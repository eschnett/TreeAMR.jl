# Ghost filling: replay a GhostSchedule.
#
# One KernelAbstractions kernel serves all three cases. Because every
# transfer is a tensor product of D one-dimensional stencils, a copy is
# just the width-1 case and needs no separate code path. Written against
# KA from the start so the CPU implementation is already the GPU one
# (M6); the only backend-specific step is `get_backend`.

# Every index this file's kernels form is built by the schedule, whose
# job is precisely to guarantee them: a target offset runs over the
# stencil's own `ntarget`, a source window is `clamp`ed into the stored
# extent at construction, and KernelAbstractions' CPU emitter guards the
# body with `__validindex`, so the global index never leaves the
# `ndrange`. Bounds checking them again in the innermost loop of the
# per-evaluation path cost more than the physics did — a quarter of the
# ghost fill, see "What the ghost fill costs" in CODE.md — so the reads
# and writes below are `@inbounds`. The claim is checked rather than
# asserted: CI runs `julia-runtest` with `check_bounds=yes`, which
# overrides `@inbounds` package-wide, so a stencil that walks out of its
# block fails there even though it would not fail locally.

# Tensor-product stencil: up to `prod(Ps)` contributions, the weight of
# each the product of its D one-dimensional weights. The widths are **per
# dimension** (M8): a transfer can be injection in one dimension and
# order-p interpolation in another, and padding the narrow one to the
# common width with zero weights would read slots that need not hold
# data at all — a G = 0 face field has none beyond its high face — and
# `0 * NaN` is `NaN`.
#
# Within a width the zero weights are skipped too on the CPU, where that
# is cheap to know: a column with a single nonzero weight (`Stencil1D`'s
# `single`) takes that one tap along its dimension instead of all
# `Ps[d]`. In vertex-centered refinement every other fine point along a
# dimension coincides with a coarse node, and its column is an exact one
# among exact zeros, so a prolongation of order 6 averages 3.5 taps
# instead of 6 along each dimension where its targets alternate.
#
# A device takes the full sum: the flat launch passes `singles =
# nothing`, and for that type the nest below is generated without the
# branches. There the skip made a prolongation stage slower, not faster
# — 2.3 times on an H200, where one EPYC thread ran it 3.7 times faster
# ("A column with one nonzero weight takes one tap" in CODE.md has the
# numbers): neighbouring lanes along dimension 1 alternate between the
# one tap and the loop, and the `2^D` bodies cost the unrolled nest its
# registers.
# Both forms compute the same bits for finite data, by the argument
# below, so the choice by backend changes no result.
#
# The skip is exact for finite data. A skipped term is `0 · u`, which is
# `±0`. The accumulator starts at `+0`, and in round-to-nearest a sum is
# `-0` only when both addends are, so it never becomes `-0`, and adding
# `±0` to it leaves it as it was. The terms that remain are the same
# products, summed in the same order. What differs is a `NaN` or an
# infinity under a zero weight: the full sum carries it into the result
# and the skip does not, which is the choice the per-dimension widths
# make for the slots outside the window.
#
# `Ps` is a compile-time constant, so the sum is *generated* as a loop
# nest with literal trip counts instead of iterating
# `CartesianIndices(Ps)`. That iterator spent its time in `__inc`, and
# it hid the fact that D-1 of the D weight loads per stencil point are
# invariant in the inner loops. With `singles` given, each dimension
# wider than one point is a branch on its `single` row — the one tap, or
# the literal loop — so the nest holds up to `2^D` copies of the
# innermost body; a width-1 dimension has no branch, and with `singles =
# nothing` none has.
#
# The order of the sum is fixed, not merely its accuracy: `m_1` runs
# innermost, as column-major `CartesianIndices` iteration does, and the
# weight product is formed as `((w₁ * w₂) * …) * w_D`, since
# floating-point multiplication does not associate and hoisting a
# partial product out of the inner loop would reassociate it. Only the
# *loads* are hoisted. That is what makes the skip exact, and what keeps
# `bracket_clamp_sum`'s full sum equal to this one.
@generated function stencil_sum(src, weights, singles, base::NTuple{D,Int},
                                wcol::NTuple{D,Int}, v, sblock,
                                ::Val{Ps}) where {D,Ps}
    idx = [:(base[$d] + $(Symbol(:m_, d)) - 1) for d in 1:D]
    wprod = foldl((a, d) -> :($a * $(Symbol(:wt_, d))), 2:D;
                  init=Symbol(:wt_, 1))
    body = quote
        acc += $wprod * transfer_load(src, ($(idx...),), v, sblock)
    end
    for d in 1:D
        m, wt, k = Symbol(:m_, d), Symbol(:wt_, d), Symbol(:k_, d)
        loop = quote
            for $m in 1:$(Ps[d])
                $wt = weights[$d][$m, wcol[$d]]
                $body
            end
        end
        body = Ps[d] == 1 || singles === Nothing ? loop : quote
            if $k != 0
                $m = $k
                $wt = weights[$d][$m, wcol[$d]]
                $body
            else
                $loop
            end
        end
    end
    # A loop, not a filtered comprehension: the generator itself is
    # compiled for every type it closes over.
    singlerows = Expr[]
    for d in 1:D
        Ps[d] > 1 && singles !== Nothing &&
            push!(singlerows, :($(Symbol(:k_, d)) = Int(singles[$d][wcol[$d]])))
    end
    return quote
        Base.@_inline_meta
        acc = zero(transfer_eltype(src))
        @inbounds begin
            $(singlerows...)
            $body
        end
        acc
    end
end

# A copy: a group whose stencils are all one point of weight one (see
# `Stencil1D`'s `unit`) loads its one value without the `D` weight loads
# and the product. The sum above starts from zero, and `0 + 1·x` is `x`
# for every `x` but `-0`, which it makes `+0`; so does this, which keeps a
# fill bit-identical to what the general path computed, and keeps the
# distributed exchange's `-0` rule ("The sender computes" in CODE.md) as
# it was.
Base.@propagate_inbounds copy_load(src, base, v, sblock) =
    zero(transfer_eltype(src)) + transfer_load(src, base, v, sblock)

# The packed buffer of a distributed exchange (M7), as a kernel sees it.
#
# Pack and unpack are transfers like any other, so they go through the
# same kernel, with a message buffer in place of the working array: the
# pack's `dest`, the unpack's `src` ("Pack and unpack are transfers" in
# CODE.md). The kernel reaches both arrays through the three accessors
# below — a load, a store, and the element type `stencil_sum` starts
# its accumulator from — each with a method for an array, which is
# exactly the indexing it always did, and one for a packed buffer.
#
# A packed buffer is a `NamedTuple` rather than a struct of our own so
# that KernelAbstractions adapts it to a device as it is, with no new
# dependency: the flat buffer `buf`, the `offsets` of the transfers'
# slots in points, and the `dims` of one transfer's target box times the
# variables. The "block" index the kernel passes is the slot: a pack's
# target slot, an unpack's source slot. A slot's box is laid out in the
# kernel's own index order, `(box…, v)` column-major, starting at
# element `offsets[slot] * nvars`.
const PackedBuffer = NamedTuple{(:buf, :offsets, :dims)}
# The form a driver passes to `run_phase!`: the buffer and the slot
# offsets of a stage, completed with each group's box in `run_group!`.
const PackedSlots = NamedTuple{(:buf, :offsets)}

@inline transfer_eltype(a) = eltype(a)
@inline transfer_eltype(p::PackedBuffer) = eltype(p.buf)

Base.@propagate_inbounds transfer_load(a, idx, v, b) = a[idx..., v, b]
Base.@propagate_inbounds transfer_load(p::PackedBuffer, idx, v, b) =
    p.buf[packed_index(p, idx, v, b)]

Base.@propagate_inbounds function transfer_store!(a, x, idx, v, b)
    a[idx..., v, b] = x
    return nothing
end
Base.@propagate_inbounds function transfer_store!(p::PackedBuffer, x, idx, v, b)
    p.buf[packed_index(p, idx, v, b)] = x
    return nothing
end

# The element of slot `slot` at box position `idx` and variable `v`.
# Recursion over the tuple rather than a loop, so that it unrolls in a
# device kernel too.
@inline packed_index(p, idx, v, slot) =
    Int(p.offsets[slot]) * last(p.dims) + boxlinear((idx..., v), p.dims) + 1
@inline boxlinear(::Tuple{}, ::Tuple{}) = 0
@inline boxlinear(i::Tuple, n::Tuple) =
    (first(i) - 1) + first(n) * boxlinear(Base.tail(i), Base.tail(n))

# The argument the kernel gets: an array as it is, and a stage's packed
# slots with this group's box.
@inline kernelarg(a, dims) = a
@inline kernelarg(p::PackedSlots, dims) = (buf=p.buf, offsets=p.offsets, dims=dims)

# The real source of a rotated transfer (M12), as a kernel sees it.
#
# A rotated transfer's stencils are built in the virtual frame, as though
# its source sat where the target sees it across the seam; what is left
# is to read the virtual source's points out of the real array, which
# this accessor does in the load, beside the packed buffer's ("Rotating
# seams" in CODE.md). `src` is the real working array — the set's own,
# or for an odd orientation in a pair its partner's — `perm` and `flip`
# the axis map of the orientation (see `axismap`), `len` the real array's
# stored size, `vars` the field set's variable table and `col` the
# orientation's column of it. Virtual stored index `k` reads the real
# point whose index along `e` is `k[perm[e]]`, or `len[e] + 1 − k[perm[e]]`
# where `flip[e]`, and variable `vars[v, col]`. A `NamedTuple` of an
# array, isbits tuples and an `Int32` table, so KernelAbstractions adapts
# it to a device as it does the packed buffer.
const RotatedSource = NamedTuple{(:src, :perm, :flip, :len, :vars, :col)}

@inline transfer_eltype(p::RotatedSource) = eltype(p.src)

Base.@propagate_inbounds transfer_load(p::RotatedSource, idx, v, b) =
    p.src[rotated_index(p, idx)..., Int(p.vars[v, p.col]), b]

@inline rotated_index(p, idx::NTuple{D,Int}) where {D} =
    ntuple(Val(D)) do e
        k = tuplepick(idx, p.perm[e])
        p.flip[e] ? p.len[e] + 1 - k : k
    end

# `t[j]` for a run-time `j`, as a chain of selects: indexing a tuple with
# a run-time index would spill it to local memory on a device.
@inline tuplepick(t::Tuple{Any}, j::Int) = first(t)
@inline tuplepick(t::Tuple, j::Int) = _tuplepick(t, j, 1)
@inline _tuplepick(t::Tuple{Any}, j::Int, d::Int) = first(t)
@inline _tuplepick(t::Tuple, j::Int, d::Int) =
    ifelse(j == d, first(t), _tuplepick(Base.tail(t), j, d + 1))

# The axis map of orientation `r` across the seam of the plane `(d1, d2)`:
# the real stored index along `d1` and `d2` from the virtual one `k`, with
# `n` the real array's stored size there (CODE.md, "Rotating seams", the
# axis-map table), and the identity along every other dimension.
#
#   r = 1:  real[d1] = k[d2],          real[d2] = n + 1 − k[d1]
#   r = 2:  real[d1] = n + 1 − k[d1],  real[d2] = n + 1 − k[d2]
#   r = 3:  real[d1] = n + 1 − k[d2],  real[d2] = k[d1]
function axismap(r::Integer, (d1, d2)::NTuple{2,Integer}, ::Val{D}) where {D}
    swapped = isodd(r)
    perm = ntuple(e -> swapped && e == d1 ? Int(d2) : swapped && e == d2 ? Int(d1) : e,
                  Val(D))
    flip = ntuple(e -> e == d1 ? r >= 2 : e == d2 ? (r == 1 || r == 2) : false, Val(D))
    return perm, flip
end

# The accessor of a rotated group's real source `src`.
function rotated_source(group::TransferGroup{T,D}, src, rotvars) where {T,D}
    perm, flip = axismap(group.orientation, group.plane, Val(D))
    return (src=src, perm=perm, flip=flip, len=ntuple(d -> size(src, d), Val(D)),
            vars=rotvars, col=Int(group.orientation) + 1)
end

# `dest` and `src` are the same array for ghost filling (targets are
# ghosts, sources interiors, so they never overlap) and different arrays
# when regridding transfers into freshly allocated storage. Neither is
# marked @Const, so the aliasing case stays well defined.
#
# On the CPU the launch is over the target box itself — `ndrange =
# (blen…, nvars, ntransfers)` — so the backend supplies the per-axis
# position by iterating the workgroup, and the kernel does no index
# arithmetic to recover it. Flattening the box into one axis and
# unflattening it here cost an integer `div` and `rem` per dimension per
# ghost point per variable: 9 % of a copy-dominated fill's self time, and
# the single largest entry after the bounds checks. On a device that same
# division happens anyway, in KernelAbstractions' own forming of the
# global index, and as a 64-bit division emulated in software it held the
# fill to 0.65 of an H200's 4.8 TB/s; so a device launch is flat,
# and the position comes from `shape`'s precomputed inverses instead
# (`launch_positional!`, and "The copy kernels on a device" in CODE.md).
#
# `factors` is `nothing` for every ordinary transfer, and then `scaled`
# is the identity and the kernel is exactly what it was before M10. A
# mirrored transfer at a reflecting face passes the field set's
# parity-factor table and its own column of it: each variable's result
# is multiplied by -1, 0 or 1, which is exact. So does a rotated
# transfer across a seam (M12), whose column holds the sign of the turned
# variable times its parity, if it is also mirrored.
#
# The bounded prolongation ("Bounded prolongation" in CODE.md). A group
# built from bounded operators reaches the kernel with `B = true` and the
# conservative order-3 stencils, whose window along each dimension is the
# parent coarse cell and its two neighbours, so the parent is at
# `base .+ 1`. The child's side along `d` is read off its stencil column —
# `(1/8, 1, -1/8)` for the low child, `(-1/8, 1, 1/8)` for the high one —
# rather than from the target position, so that a mirrored, clamped or
# rotated row, whose column is its image's, and a pack, which keeps the
# serial columns, need nothing of their own. The parity or rotation sign
# `scaled` applies afterwards commutes with the bounded sum, which is odd.
#
# Per variable: one-sided differences `a`, `b` along each dimension, the
# monotonized central slope `s_d`, the child `ū + φ Σ σ_d s_d / 4`, and
# `φ` the largest scaling in `[0, 1]` that keeps every child of the
# parent within the extremes of the parent and its `2D` face neighbours.
# The comparisons of `minmod` would turn a `NaN` neighbour into a zero
# slope, so the result carries `0 · (M − m)`, which is `NaN` exactly when
# an extreme is: an undefined source stays visible, as it does in
# `stencil_sum`.
@inline minmod(a, b) = a > 0 && b > 0 ? min(a, b) : a < 0 && b < 0 ? max(a, b) : zero(a)
@inline mc_slope(a, b) = minmod(2a, minmod(2b, (a + b) / 2))

# Left to right, so the sum is the same on every backend.
@inline tuplesum(t::Tuple{Any}) = t[1]
@inline tuplesum(t::Tuple) = t[1] + tuplesum(Base.tail(t))

Base.@propagate_inbounds function bounded_sum(src, weights, base::NTuple{D,Int},
                                              wcol::NTuple{D,Int}, v, sblock,
                                              ::Val{D}) where {D}
    c = ntuple(d -> base[d] + 1, Val(D))
    u = transfer_load(src, c, v, sblock)
    lo = ntuple(d -> transfer_load(src, Base.setindex(c, c[d] - 1, d), v, sblock), Val(D))
    hi = ntuple(d -> transfer_load(src, Base.setindex(c, c[d] + 1, d), v, sblock), Val(D))
    s = ntuple(d -> mc_slope(hi[d] - u, u - lo[d]), Val(D))
    high = ntuple(d -> weights[d][3, wcol[d]] > weights[d][1, wcol[d]], Val(D))
    Δ = tuplesum(ntuple(d -> high[d] ? s[d] : -s[d], Val(D))) / 4
    S = tuplesum(ntuple(d -> abs(s[d]), Val(D))) / 4
    M = max(u, lo..., hi...)
    m = min(u, lo..., hi...)
    φ = S > 0 ? min(one(u), (M - u) / S, (u - m) / S) : one(u)
    return u + φ * Δ + zero(u) * (M - m)
end

# The bounded point-value interpolation (AMR-10; "Bounded point values"
# in CODE.md). The stencils are `stack_linear`'s: rows `1:Ps[d]` the
# order-`p` weights, rows `Ps[d]+1:2Ps[d]` the multilinear ones over the
# same window. The order-`p` sum is formed as `stencil_sum` forms it, but
# in full, without its one-tap skip, and beside it the extremes of the
# points with a nonzero multilinear weight in every dimension — the
# target's bracket. The result is the sum clamped into them, which is
# `u_lin + φ(u_w − u_lin)` with the largest `φ ∈ [0, 1]` that stays in
# range, since `u_lin` lies in it. Where no clamp acts it is the unbounded
# sum, bit for bit but for `-0` (for finite data, since the skip is exact
# there). `min` and `max` propagate a `NaN` bracket point and `clamp` a
# `NaN` sum, and the result carries `0 · (M − m)`, as `bounded_sum` does,
# so an undefined source stays visible.
@generated function bracket_clamp_sum(src, weights, base::NTuple{D,Int},
                                      wcol::NTuple{D,Int}, v, sblock,
                                      ::Val{Ps}) where {D,Ps}
    idx = [:(base[$d] + $(Symbol(:m_, d)) - 1) for d in 1:D]
    wprod = foldl((a, d) -> :($a * $(Symbol(:wt_, d))), 2:D;
                  init=Symbol(:wt_, 1))
    inbracket = foldl((a, d) -> :($a && !iszero($(Symbol(:lt_, d)))), 2:D;
                      init=:(!iszero($(Symbol(:lt_, 1)))))
    body = quote
        u = transfer_load(src, ($(idx...),), v, sblock)
        acc += $wprod * u
        if $inbracket
            lo = min(lo, u)
            hi = max(hi, u)
        end
    end
    for d in 1:D
        body = quote
            for $(Symbol(:m_, d)) in 1:$(Ps[d])
                $(Symbol(:wt_, d)) = weights[$d][$(Symbol(:m_, d)), wcol[$d]]
                $(Symbol(:lt_, d)) = weights[$d][$(Ps[d]) + $(Symbol(:m_, d)), wcol[$d]]
                $body
            end
        end
    end
    return quote
        Base.@_inline_meta
        acc = zero(transfer_eltype(src))
        lo = typemax(transfer_eltype(src))
        hi = typemin(transfer_eltype(src))
        @inbounds $body
        clamp(acc, lo, hi) + zero(acc) * (hi - lo)
    end
end

@inline scaled(acc, ::Nothing, v, col) = acc
@inline scaled(acc, factors, v, col) = @inbounds acc * factors[v, col]

@kernel function transfer_kernel!(shape, dest, src,
                                  @Const(targetblocks), @Const(sourceblocks),
                                  srcstarts, weights, singles, unit::Bool,
                                  targetfirst::NTuple{D,Int}, toffset::Int,
                                  factors, fcol::Int, ::Val{B},
                                  ::Val{Ps}, ::Val{D}) where {B,Ps,D}
    J = @index(Global, NTuple)
    I = kernel_position(shape, J)
    # `I[1:D]` is the position within the target region, one-based.
    v = I[D + 1]
    t = I[D + 2] + toffset

    @inbounds tblock = targetblocks[t]
    @inbounds sblock = sourceblocks[t]

    wcol = ntuple(d -> I[d], Val(D))
    tidx = ntuple(d -> targetfirst[d] + I[d] - 1, Val(D))
    @inbounds base = ntuple(d -> Int(srcstarts[d][I[d]]), Val(D))

    # `unit` is the same for every item of a launch, so the branch costs
    # nothing on a device, where it is uniform across a warp. Only a
    # one-point stencil can be a copy, and `Ps` is a type parameter, so
    # every other kernel compiles without the branch.
    acc = B == BOUNDED_CONSERVATIVE ?
          (@inbounds bounded_sum(src, weights, base, wcol, v, sblock, Val(D))) :
          B == BOUNDED_POINTVALUE ?
          bracket_clamp_sum(src, weights, base, wcol, v, sblock, Val(Ps)) :
          all(isone, Ps) && unit ? (@inbounds copy_load(src, base, v, sblock)) :
          stencil_sum(src, weights, singles, base, wcol, v, sblock, Val(Ps))
    @inbounds transfer_store!(dest, scaled(acc, factors, v, fcol), tidx, v, tblock)
end

# Launch one group's transfers `range` — the whole group by default.
# `single` asks KernelAbstractions for a *single* workgroup, so the
# launch runs inline on the calling task instead of spawning its own:
# that is what lets a phase be one parallel loop over slices rather than
# a nest of parallel loops.
#
# `factors` is the field set's factor table, or `nothing`; only a
# mirrored or rotated group reads it (see `scaled`). A rotated group
# (M12) reads its real source through `rotated_source`, from `src` for an
# even orientation and from `altsrc` for an odd one — the partner's
# working array in a `RotationPair`, the set's own otherwise — with the
# variable table `rotvars`; an ordinary group launches exactly as before.
# `flat` is the launch's form, flat on a device and shaped on the CPU
# unless a test asks otherwise (see `launch_shape`).
function run_group!(dest, src, group::TransferGroup{T,D}, nvars::Integer, backend;
                    range=1:ntransfers(group), single::Bool=false,
                    factors=nothing, altsrc=src, rotvars=nothing,
                    flat::Bool=flatlaunch(backend)) where {T,D}
    n = length(range)
    n == 0 && return nothing
    group.factorcol == 0 || factors !== nothing || throw(ArgumentError(
        "this schedule mirrors ghosts across a reflecting face or turns them across " *
        "a rotating seam, but the field set has no factor table to do it with. " *
        "Build the field set over the same forest as the schedule, with `parity` " *
        "or `rotation`."))
    group.orientation == 0 || rotvars !== nothing || throw(ArgumentError(
        "this schedule turns ghosts across a rotating seam, but the field set has " *
        "no variable table to turn them with. Build the field set over the same " *
        "forest as the schedule, with `rotation`."))
    gfactors = group.factorcol == 0 ? nothing : factors
    blen = boxsize(group)
    prod(blen) == 0 && return nothing
    tfirst = ntuple(d -> group.stencils[d].targetfirst, Val(D))
    # One width per dimension, not one shared width: see the kernel.
    rows = ntuple(d -> stencilorder(group.stencils[d]), Val(D))
    # A bounded conservative group's kernel reads the parent and its face
    # neighbours out of the order-3 window, and a bounded point-value
    # group's stencils stack two sets of weights (`stack_linear`), so its
    # order is half its rows; anything else is a bug in whatever built
    # the group.
    group.bounded != BOUNDED_CONSERVATIVE || all(==(3), rows) || error(
        "a bounded conservative group has stencils of orders $rows, not 3")
    group.bounded != BOUNDED_POINTVALUE || all(iseven, rows) || error(
        "a bounded point-value group has stencils of $rows rows, not stacked pairs")
    orders = group.bounded == BOUNDED_POINTVALUE ? map(r -> r ÷ 2, rows) : rows
    # Like `Val(orders)`, a run-time `Val`: the launch below is dispatched
    # on it, so a bounded kernel is compiled only where one runs.
    bounded = Val(group.bounded)
    srcstarts = ntuple(d -> group.stencils[d].srcstart, Val(D))
    weights = ntuple(d -> group.stencils[d].weights, Val(D))
    # The row of each column's one nonzero weight, or zero: see
    # `stencil_sum`. The shaped launch reads it; the flat one, a device's,
    # takes the full sum instead.
    singles = ntuple(d -> group.stencils[d].single, Val(D))
    # A copy goes without its weights: see `copy_load`. A run-time flag
    # rather than `weights = nothing`, which would make every launch site
    # compile both forms of the kernel (see below).
    unit = all(s -> s.unit, group.stencils)
    kdest = kernelarg(dest, (blen..., Int(nvars)))
    ksrc = kernelarg(src, (blen..., Int(nvars)))

    # A slice of a group is passed as an offset into the group's own
    # block lists rather than as two `view`s: every kernel argument
    # lands in a tuple that KernelAbstractions heap-allocates on each
    # launch, and two `SubArray`s are the largest arguments here. See
    # "What the ghost fill costs" in CODE.md for what that buys and what
    # it does not.
    #
    # The flat launch's shape is a `LinearShape` of `Int32` or of `Int`, by
    # the launch's size: a run-time choice of type. Seen by inference,
    # every call site would be compiled for both, and on the CPU for a flat
    # launch only the suite makes; together with a `nothing`/weights union
    # that this kernel once had, that made the compilation-bound MPI
    # workload a fifth slower and pushed its three-rank job past CI's
    # deadline on a small runner ("The copy kernels on a device" in
    # CODE.md). So a flat launch hides the kernel from inference and is
    # dispatched on the values it gets — one dynamic call, on a device
    # where a launch costs microseconds — while the CPU's shaped launch
    # stays type-stable and allocates no more than it did.
    extents = (blen..., Int(nvars), n)
    if group.orientation != 0
        rsrc = rotated_source(group, isodd(group.orientation) ? altsrc : src, rotvars)
        launch_transfer!(backend, flat, extents, single, kdest, rsrc, group.targetblocks,
                         group.sourceblocks, srcstarts, weights, singles, unit, tfirst,
                         first(range) - 1, gfactors, Int(group.factorcol), bounded,
                         Val(orders), Val(D))
        return nothing
    end
    launch_transfer!(backend, flat, extents, single, kdest, ksrc, group.targetblocks,
                     group.sourceblocks, srcstarts, weights, singles, unit, tfirst,
                     first(range) - 1, gfactors, Int(group.factorcol), bounded,
                     Val(orders), Val(D))
    return nothing
end

# The launch itself, in its two forms (see above). Each branch makes its
# own call, so that the shaped one stays type-stable; a closure over the
# arguments instead was allocated at every launch. The flat launch drops
# `singles` for `nothing`, so a device's kernel takes the full sum (see
# `stencil_sum`); behind the inference barrier the different argument
# type costs the shaped launch nothing.
@inline function launch_transfer!(backend, flat::Bool, extents, single::Bool,
                                  dest, src, targetblocks, sourceblocks, srcstarts,
                                  weights, singles, args...)
    if flat
        shape, ndrange = launch_shape(true, extents)
        kernel! = Base.inferencebarrier(transfer_kernel!(backend))
        kernel!(shape, dest, src, targetblocks, sourceblocks, srcstarts, weights, nothing,
                args...; ndrange=ndrange, workgroupsize=(single ? ndrange : nothing))
    else
        transfer_kernel!(backend)(nothing, dest, src, targetblocks, sourceblocks, srcstarts,
                                  weights, singles, args...; ndrange=extents,
                                  workgroupsize=(single ? extents : nothing))
    end
    return nothing
end

run_group!(fs::FieldSet{T,D}, group::TransferGroup{T,D}, backend) where {T,D} =
    run_group!(fs.work, fs.work, group, fs.nvars, backend; factors=fs.factors,
               rotvars=fs.rotvars)

# One phase of the exchange — all the copies and restrictions, or all
# the prolongations onto one level — as a single parallel loop.
#
# The transfers within a phase write disjoint ghost cells, so all of
# them may run at once. They are *batched* by stencil, though, and the
# batches differ in size by orders of magnitude: a face slab is
# `G·N^(D-1)` cells, a corner `G^D`. Running the batches one launch
# after another therefore leaves the small ones with a single workgroup
# each, which is to say serial — measured in M5 as a ceiling of about
# 2.5x on the ghost fill however many threads were available, while the
# single-launch parts of the same step scaled fine.
#
# So on the CPU the phase is one parallel loop over threads, and each
# thread takes its share of *every* group: the transfers whose target
# block it owns (see `threadchunks`), launched as single inline
# workgroups. A thread therefore fills the ghosts of the blocks it
# scatters into and computes on, which is the other half of the reason
# for the loop: a block whose ghosts were written by a different core
# than the one about to read them streams at a fraction of the rate on
# a many-core node (`CODE.md`, "What one process loses"; the M5 form,
# slices of equal cell count dealt largest first to spawned tasks,
# balanced better and moved every block to a new core in every phase).
# Every group's `targetblocks` is non-decreasing, so a thread's share of
# a group is one contiguous run found by bisection. A device backend has
# neither problem — there a launch *is* the parallel unit — and takes
# the plain per-group path below.
#
# A pack (M7) writes a message buffer rather than a block, so it is the
# *source* block's owner that runs it: the thread that last wrote the
# data it reads. Its groups are sorted by source block instead, and the
# bisection runs over whichever block list says who owns the transfer.
ownerblocks(group, dest) = group.targetblocks
ownerblocks(group, ::PackedSlots) = group.sourceblocks
ownercount(dest, src) = size(dest, ndims(dest))
ownercount(::PackedSlots, src) = size(src, ndims(src))

function run_phase!(dest, src, groups, nvars::Integer, backend; factors=nothing,
                    altsrc=src, rotvars=nothing)
    for group in groups
        run_group!(dest, src, group, nvars, backend; factors=factors, altsrc=altsrc,
                   rotvars=rotvars)
    end
    return nothing
end

function run_phase!(dest, src, groups, nvars::Integer, backend::CPU;
                    factors=nothing, altsrc=src, rotvars=nothing)
    nb = ownercount(dest, src)
    if length(threadchunks(nb)) <= 1
        for group in groups
            run_group!(dest, src, group, nvars, backend; factors=factors, altsrc=altsrc,
                       rotvars=rotvars)
        end
        return nothing
    end
    threaded_chunks(nb) do _, owned
        for group in groups
            owners = ownerblocks(group, dest)
            lo = searchsortedfirst(owners, first(owned))
            hi = searchsortedlast(owners, last(owned))
            lo <= hi && run_group!(dest, src, group, nvars, backend;
                                   range=lo:hi, single=true, factors=factors,
                                   altsrc=altsrc, rotvars=rotvars)
        end
    end
    return nothing
end

run_phase!(fs::FieldSet{T,D}, groups, backend) where {T,D} =
    run_phase!(fs.work, fs.work, groups, fs.nvars, backend; factors=fs.factors,
               rotvars=fs.rotvars)

# --- Stages (M7) -----------------------------------------------------------
#
# The driver of a staged exchange. Every ordering point of the serial
# fill is a stage (see `ExchangeStage`), and a stage runs in five steps
# ("Stages" in CODE.md): post the receives, pack and synchronize, post
# the sends, run the local groups while the messages are in flight, and
# wait for the receives and unpack. The sends are waited on once, at the
# end of the call. The pieces are separate functions so that the tests
# can run every simulated rank's stage in lockstep in one process,
# wiring the buffers between ranks directly; `run_stages!` composes them
# through the communicator verbs, which is the path a real run takes.
#
# A stage without messages is a phase exactly as before M7, so a serial
# fill runs the same launches and the same barriers it always did.

# The stage's send and receive buffers for `nvars` variables, on the
# backend, taken on first use and kept with the schedule, since a
# schedule serves every field set of its layout. Over a forest they are
# leased from its buffer pool (`BufferPool`), so that a stage built
# after a regrid reuses what earlier stages held; without one (the
# in-process tests, which wire the buffers between simulated ranks
# themselves) they are allocated and zeroed, as before the pool.
function stagebuffers(remote::RemoteStage{D,GRP,VB,BUF}, nvars::Integer, backend,
                      forest=nothing) where {D,GRP,VB,BUF}
    bufs = get(remote.buffers, Int(nvars), nothing)
    bufs === nothing || return bufs
    make = cap -> new_stagebuffer(BUF, backend, cap)
    lengths = (Int(nvars) * sum(remote.sendcounts; init=0),
               Int(nvars) * sum(remote.recvcounts; init=0))
    bufs = map(lengths) do n
        forest === nothing ? make(n).full::BUF :
        lease!(make, BUF, bufferpool(forest), (:buffer, BUF, typeof(backend)), n,
               generation(forest), remote.buffers, Int(nvars))
    end
    remote.buffers[Int(nvars)] = bufs
    return bufs
end

# A fresh, zeroed stage buffer of `cap` elements. A host one (the CPU's)
# is an `Array` over a `Memory` of its own, which a lease can wrap at a
# shorter length; a device one is the backend's array, which a lease
# views.
function new_stagebuffer(::Type{BUF}, backend, cap::Int) where {BUF}
    T = eltype(BUF)
    if BUF <: Array
        mem = Memory{T}(undef, cap)
        full = fill!(Base.wrap(Array, mem, (cap,)), zero(T))
        return PooledBuffer(full, mem)
    end
    full = allocate(backend, T, cap)
    fill!(full, zero(T))
    return PooledBuffer(full, nothing)
end

# Host mirrors of the stage's buffers for `nvars` variables, for a
# communicator that cannot send from or receive into the backend's own
# memory (`hoststaging`): the same layout in host vectors, taken on
# first use and kept with the stage like the buffers themselves, so a
# staged fill allocates nothing new either. They are page-locked for the
# backend where it implements that (`KernelAbstractions.pagelock!`; CUDA
# pins them, the CPU and Metal do nothing), which is what lets a device
# copy them at the bus's rate rather than through a bounce buffer, and
# leased from the forest's pool like the buffers: page-locking is the
# slow part of allocating one, so a pooled mirror is page-locked once.
function stagemirrors(remote::RemoteStage{D,GRP,VB,BUF,HB}, nvars::Integer,
                      backend, forest=nothing) where {D,GRP,VB,BUF,HB}
    mirrors = get(remote.mirrors, Int(nvars), nothing)
    mirrors === nothing || return mirrors
    pool = forest === nothing ? nothing : bufferpool(forest)
    make = cap -> new_mirror(eltype(HB), backend, cap, pool)
    lengths = (Int(nvars) * sum(remote.sendcounts; init=0),
               Int(nvars) * sum(remote.recvcounts; init=0))
    mirrors = map(lengths) do n
        forest === nothing ? make(n).full::HB :
        lease!(make, HB, pool, (:mirror, HB, typeof(backend)), n,
               generation(forest), remote.mirrors, Int(nvars))
    end
    remote.mirrors[Int(nvars)] = mirrors
    return mirrors
end

function new_mirror(::Type{T}, backend, cap::Int, pool) where {T}
    mem = Memory{T}(undef, cap)
    full = fill!(Base.wrap(Array, mem, (cap,)), zero(T))
    pagelock_mirror!(backend, full)
    pool === nothing || cap == 0 || (pool.pagelocked += 1)
    return PooledBuffer(full, mem)
end

# Give a stage's buffers and mirrors back to the forest's pool, and
# forget them: the regrid stage's, once its sends have been waited on,
# since it lives for one call.
function release_stage!(forest::Forest, remote::RemoteStage)
    for bufs in values(remote.buffers)
        release!(bufferpool(forest), bufs)
    end
    for mirrors in values(remote.mirrors)
        release!(bufferpool(forest), mirrors)
    end
    empty!(remote.buffers)
    empty!(remote.mirrors)
    return nothing
end

# `pagelock!` is in KernelAbstractions from 0.9.40 on; where a backend
# does not implement it, it returns `missing` and the mirror is ordinary
# pageable memory, which is still correct. An empty mirror — a stage
# this rank only sends in, or only receives in — is not page-locked:
# CUDA refuses to register an empty range (found on Symmetry's H200s,
# M7 step 8), and there is nothing to copy.
pagelock_mirror!(backend, a::Vector) =
    isdefined(KernelAbstractions, :pagelock!) && !isempty(a) ?
    KernelAbstractions.pagelock!(backend, a) : nothing

# Each peer's segment of a buffer, in elements: `counts` are in points,
# and the segments follow each other in peer order.
function segment_ranges(counts::Vector{Int}, nvars::Integer)
    ranges = Vector{UnitRange{Int}}(undef, length(counts))
    lo = 1
    for (i, c) in enumerate(counts)
        ranges[i] = lo:(lo + c * Int(nvars) - 1)
        lo += c * Int(nvars)
    end
    return ranges
end

# Evaluate every send transfer of the stage into its slot of the send
# buffer, then synchronize: a device buffer is safe to hand to MPI only
# once the kernels writing it have finished. `src` is the array the
# transfers read: a field set's working array in an exchange, the old
# mesh's in the regrid transfer. A rotated pack (M12) reads through the
# variable table `rotvars`, from `altsrc` for an odd orientation (the
# partner's array in a `RotationPair`), and computes the unscaled sum.
# A group of stages packs them all and synchronizes once (`sync=false`
# and a `synchronize` of its own; see `run_stages!`).
function pack_stage!(src::AbstractArray, remote::RemoteStage, bufs, nvars::Integer,
                     backend; altsrc=src, rotvars=nothing, sync::Bool=true)
    run_phase!((buf=bufs[1], offsets=remote.sendoffsets), src, remote.packs, nvars,
               backend; altsrc=altsrc, rotvars=rotvars)
    sync && synchronize(backend)
    return nothing
end
pack_stage!(fs::FieldSet, remote::RemoteStage, bufs, backend) =
    pack_stage!(fs.work, remote, bufs, fs.nvars, backend; rotvars=fs.rotvars)

# Copy every received slot into its target box in `dest`, applying the
# parity factor of a mirrored transfer, or the sign of a rotated one,
# here, where the serial kernel applies it.
function unpack_stage!(dest::AbstractArray, remote::RemoteStage, bufs, nvars::Integer,
                       factors, backend)
    run_phase!(dest, (buf=bufs[2], offsets=remote.recvoffsets), remote.unpacks, nvars,
               backend; factors=factors)
    return nothing
end
unpack_stage!(fs::FieldSet, remote::RemoteStage, bufs, backend) =
    unpack_stage!(fs.work, remote, bufs, fs.nvars, fs.factors, backend)

# Run the stages `stages[range]` on this rank as one group through the
# forest's communicator (a run of `GhostSchedule`'s `stagegroups`, or a
# single stage), from `src` into `dest`: the same working array in an
# exchange, since its targets are ghosts and its sources interiors, and
# the old and the new mesh's arrays in the regrid transfer (step 4 of
# M7), whose packs read the old blocks and whose local groups and unpacks
# write the new ones. `locals` are the local groups of all of them, in
# stage order: a single stage's own `locals`, or the group's
# concatenation (`GhostSchedule`'s `grouplocals`), built with the
# schedule. `sends` collects the send requests of the call so far —
# `nothing` until the first one — and is returned, to be waited on at the
# end of the call. The communicator is fetched only where a stage has
# messages: the field is abstractly typed, and a call that took it as an
# argument would be dispatched at run time on every stage of a serial
# fill too.
#
# A stage runs in five steps, and a group takes each for every stage of
# it before the next: every stage's receives are posted; every stage
# packs, and the backend is synchronized once; every stage's sends are
# posted; the local groups of all of them run as one phase, one parallel
# loop on the CPU; and once all the receives are in, every stage unpacks.
# Nothing a pack or a local group of a group reads is written by another
# stage of it (`stage_groups`), so this is the stages run one after
# another without the waits between them, and the ghosts come out bit
# for bit the same. The local groups of different stages write ghosts of
# blocks at different levels, so the one phase writes each point once, as
# a phase does. Each stage keeps its own tag, buffers and host mirrors,
# so the messages are those the stages send alone, and a peer may run the
# same stages one at a time. A group without messages is one phase and
# one barrier.
#
# One driver, for a group and for a single stage alike, with the locals
# handed in as the `Vector` a stage holds: a second driver, or an
# iterator over the stages' locals, is a second specialization of the
# whole fill's call tree for every field-set type, which the
# compilation-bound MPI workload paid for in seconds ("Stages" in
# CODE.md has the count).
#
# On a device whose buffers the communicator cannot take (`hoststaging`,
# step 8 of M7) the messages go through host mirrors: the receives are
# posted into the receive mirror, the packed buffer is copied down after
# the pack's synchronization and sent from the send mirror, and the
# received mirror is copied up before the unpack. The data, the layout
# and the kernels are the device buffers' own, so the bytes on the wire
# and the ghosts written are the same either way; only where MPI reads
# and writes them differs. Each copy is ordered where it has to be: the
# download (`copyto!` into an `Array`) returns once the host holds the
# data, and the upload is queued before the unpack on the backend's
# queue, and complete by the final `synchronize`, before the next call
# can receive into the mirror again (each stage has mirrors of its own,
# and the send mirror is rewritten only after the call's sends have been
# waited on). A CPU buffer is an `Array` and goes to the communicator
# directly, as before. The buffers and mirrors are fetched again at each
# step rather than kept from the first: they are cached with the stage,
# and a container of them would hand every launch below an abstract
# type.
#
# `altsrc` and `rotvars` are what a rotated transfer (M12) reads through:
# the array its odd orientations read — the partner's in a
# `RotationPair`, `src` itself otherwise — and the variable table. The
# packs read them as the local groups do; an unpack is a plain copy with
# the factor column, so it needs neither.
Base.@nospecializeinfer function run_stages!(dest::AbstractArray, src::AbstractArray,
                                             nvars::Integer, factors,
                                             stages::Vector{<:ExchangeStage},
                                             range::UnitRange{Int},
                                             locals::Vector{<:TransferGroup},
                                             forest::Forest, backend, @nospecialize(sends),
                                             altsrc::AbstractArray, rotvars)
    if !anyremote(stages, range)
        run_phase!(dest, src, locals, nvars, backend; factors=factors, altsrc=altsrc,
                   rotvars=rotvars)
        synchronize(backend)
        return sends::Union{Nothing,Vector{Any}}
    end
    recvs, out = send_stages!(src, stages, range, nvars, forest, backend, sends, altsrc,
                              rotvars)
    run_phase!(dest, src, locals, nvars, backend; factors=factors, altsrc=altsrc,
               rotvars=rotvars)
    receive_stages!(dest, stages, range, nvars, factors, forest, backend, recvs)
    return out
end
run_stages!(fs::FieldSet, stages, range::UnitRange{Int}, locals, forest::Forest, backend,
            sends) =
    run_stages!(fs.work, fs.work, fs.nvars, fs.factors, stages, range, locals, forest,
                backend, sends, fs.work, fs.rotvars)

# The message steps of `run_stages!`, each a function of only what it
# reads, so that the larger part of the driver is compiled once per
# source and once per target rather than for every combination of the
# two. `sends` is not specialized on either: it is `nothing` or a
# `Vector{Any}`, and every call site would otherwise compile both.
#
# The receives posted, the packs and the one synchronization, and the
# sends posted: the receive requests, and the call's sends so far.
Base.@nospecializeinfer function send_stages!(src::AbstractArray,
                                              stages::Vector{<:ExchangeStage},
                                              range::UnitRange{Int}, nvars::Integer,
                                              forest::Forest, backend, @nospecialize(sends),
                                              altsrc::AbstractArray, rotvars)
    comm = forest.comm
    recvs = Any[]
    for i in range
        remote = stages[i].remote
        remote === nothing && continue
        bufs = stagebuffers(remote, nvars, backend, forest)
        wire = hoststaging(comm, bufs[1]) ? stagemirrors(remote, nvars, backend, forest) :
               bufs
        for (peer, r) in zip(remote.recvpeers, segment_ranges(remote.recvcounts, nvars))
            push!(recvs, irecv(comm, view(wire[2], r), peer, stages[i].tag))
        end
    end
    for i in range
        remote = stages[i].remote
        remote === nothing && continue
        pack_stage!(src, remote, stagebuffers(remote, nvars, backend, forest), nvars,
                    backend; altsrc=altsrc, rotvars=rotvars, sync=false)
    end
    synchronize(backend)
    out = sends === nothing ? Any[] : sends::Vector{Any}
    for i in range
        remote = stages[i].remote
        remote === nothing && continue
        bufs = stagebuffers(remote, nvars, backend, forest)
        staged = hoststaging(comm, bufs[1])
        wire = staged ? stagemirrors(remote, nvars, backend, forest) : bufs
        staged && copyto!(wire[1], bufs[1])
        for (peer, r) in zip(remote.sendpeers, segment_ranges(remote.sendcounts, nvars))
            push!(out, isend(comm, view(wire[1], r), peer, stages[i].tag))
        end
    end
    return recvs, out
end

# The wait for the receives, the unpacks, and the closing synchronization.
function receive_stages!(dest::AbstractArray, stages::Vector{<:ExchangeStage},
                         range::UnitRange{Int}, nvars::Integer, factors, forest::Forest,
                         backend, recvs::Vector{Any})
    comm = forest.comm
    waitall(comm, recvs)
    for i in range
        remote = stages[i].remote
        remote === nothing && continue
        bufs = stagebuffers(remote, nvars, backend, forest)
        staged = hoststaging(comm, bufs[1])
        staged && copyto!(bufs[2], stagemirrors(remote, nvars, backend, forest)[2])
        unpack_stage!(dest, remote, bufs, nvars, factors, backend)
    end
    synchronize(backend)
    return nothing
end

# Whether any of the stages `stages[range]` sends or receives here. A loop
# rather than `all` over a closure, which would be one more method per
# stage type.
function anyremote(stages::Vector{<:ExchangeStage}, range::UnitRange{Int})
    for i in range
        stages[i].remote === nothing || return true
    end
    return false
end

# One stage alone, which is the group of one: what the tests and the
# benchmarks run a stage with. The one-element vector is allocated per
# call, so the fill itself goes through `run_stages!` with its schedule's
# own vectors.
run_stage!(fs::FieldSet, stage::ExchangeStage, forest::Forest, backend, sends) =
    run_stages!(fs, [stage], 1:1, stage.locals, forest, backend, sends)

# A whole staged ghost fill: phase 1, the boundary hook on this rank's
# own blocks, then phase 2 by target level. This is `fill_ghosts!` once
# its checks have passed; over a serial forest every stage is a plain
# phase and no message is sent.
function exchange_ghosts!(fs::FieldSet, schedule::GhostSchedule, boundary, backend)
    forest = schedule.forest
    stages = schedule.stages
    groups = schedule.stagegroups
    locals = schedule.grouplocals
    # Phase 1: same-level copies and restrictions. Both read interiors
    # only, so they cannot race with each other.
    sends = run_stages!(fs, stages, groups[1], locals[1], forest, backend, nothing)

    # Physical boundaries, before prolongation rather than after
    # everything: a block against the domain edge has prolongation
    # stencils that reach tangentially past that edge into its coarse
    # source's outer ghosts, so those must already hold data.
    if boundary !== nothing
        apply_boundary!(fs, boundary, schedule, backend)
    end

    # Phase 2: prolongations, coarsest targets first. A prolongation may
    # read its coarse source's ghosts, which the earlier sweeps filled;
    # levels that read nothing the ones below them write run together, as
    # one group of stages (`stagegroups`). The relative hook's regions
    # that read prolongated ghosts run after the group whose stage fills
    # them (`boundary_after_stage!`), in level order.
    for k in 2:length(groups)
        sends = run_stages!(fs, stages, groups[k], locals[k], forest, backend, sends)
        boundary === nothing && continue
        for i in groups[k]
            boundary_after_stage!(fs, boundary, schedule, backend, stages[i].tag)
        end
    end
    sends === nothing || waitall(forest.comm, sends)
    return fs
end

# The part of a hook that runs after a phase-2 stage: nothing, but for the
# relative hook, whose regions next to a coarser block read the ghosts
# that stage prolongates (see `BoundaryRegion`).
boundary_after_stage!(fs, boundary, schedule, backend, tag) = nothing

# The cell-wise boundary form (M6).
#
# The region form hands the hook a `FieldSet` and a `CartesianIndices`
# and lets it do as it likes, which on a device means scalar-indexing a
# device array — the one place in the package that did. So the hook gets
# a second, narrower form: a pure per-cell function, which the package
# itself launches as a kernel. The offset arithmetic below is the same
# as `transfer_kernel!`'s, and the position is formed exactly as
# `coordinates` forms it, from the same origin and spacing, so the two
# agree bit for bit.
@kernel function boundary_kernel!(work, g, @Const(blocks), @Const(directions),
                                  @Const(firsts), @Const(origins), @Const(spacings),
                                  ::Val{D}, ::Val{G}, ::Val{C}) where {D,G,C}
    I = @index(Global, NTuple)
    v = I[D + 1]
    t = I[D + 2]
    @inbounds b = blocks[t]
    @inbounds δ = directions[t]
    @inbounds f = firsts[t]

    idx = ntuple(d -> Int(f[d]) + I[d] - 1, Val(D))

    @inbounds origin, h = origins[b], spacings[b]
    # The same expression `coordinates` forms, in the same order, from
    # the same origin and spacing, so the two agree bit for bit — half a
    # cell in a cell-centered dimension, a whole one in a vertex-like
    # dimension.
    poff = pointoffsets(h, C)
    x = ntuple(d -> origin[d] + (idx[d] - G[d] - poff[d]) * h, Val(D))
    # `g` is the user's, and is deliberately *outside* the `@inbounds`
    # above: whatever it indexes is checked as it would be anywhere else.
    val = g(x, v, ntuple(d -> Int(δ[d]), Val(D)))
    @inbounds work[idx..., v, b] = val
end

# The all-variables form of the same kernel: no variable axis in the
# ndrange, one call per cell, every slot written from the tuple that
# comes back. The index and position arithmetic is the same as above, so
# the two forms write bit-for-bit the same numbers.
@kernel function boundary_all_kernel!(work, g, @Const(blocks), @Const(directions),
                                      @Const(firsts), @Const(origins),
                                      @Const(spacings),
                                      ::Val{D}, ::Val{G}, ::Val{C},
                                      ::Val{NV}) where {D,G,C,NV}
    I = @index(Global, NTuple)
    t = I[D + 1]
    @inbounds b = blocks[t]
    @inbounds δ = directions[t]
    @inbounds f = firsts[t]

    idx = ntuple(d -> Int(f[d]) + I[d] - 1, Val(D))

    @inbounds origin, h = origins[b], spacings[b]
    poff = pointoffsets(h, C)
    x = ntuple(d -> origin[d] + (idx[d] - G[d] - poff[d]) * h, Val(D))
    vals = g(x, ntuple(d -> Int(δ[d]), Val(D)))
    # Unrolled through `Val`, as in `coordinates_all_kernel!`.
    ntuple(Val(NV)) do v
        @inbounds work[idx..., v, b] = vals[v]
        nothing
    end
end

# The relative hook (AMR-7): the ghost at `idx` is `f(x_g) + (u(r) −
# f(x_r))`, `r` the nearest evolved point (`clamps`, see
# `BoundaryRegion`) and both positions formed as `coordinates` forms them.
# In that order, so that data equal to `f` come back as `f(x_g)` bit for
# bit. `r` never lies in a region this launch writes: along a clamped
# component it is an evolved point, along every other one a ghost that
# transfers filled before this batch runs, so the kernel reads `work`
# where it writes it without a race.
@inline relative_point(idx, clamps, ::Val{D}) where {D} =
    ntuple(d -> clamps[d] == 0 ? idx[d] : Int(clamps[d]), Val(D))

@kernel function relative_kernel!(work, f, @Const(blocks), @Const(firsts),
                                  @Const(clamps), @Const(origins), @Const(spacings),
                                  ::Val{D}, ::Val{G}, ::Val{C}) where {D,G,C}
    I = @index(Global, NTuple)
    t = I[D + 1]
    @inbounds b = blocks[t]
    @inbounds first_ = firsts[t]
    @inbounds cl = clamps[t]
    idx = ntuple(d -> Int(first_[d]) + I[d] - 1, Val(D))
    r = relative_point(idx, cl, Val(D))
    @inbounds origin, h = origins[b], spacings[b]
    poff = pointoffsets(h, C)
    xg = ntuple(d -> origin[d] + (idx[d] - G[d] - poff[d]) * h, Val(D))
    xr = ntuple(d -> origin[d] + (r[d] - G[d] - poff[d]) * h, Val(D))
    @inbounds ur = work[r..., 1, b]
    @inbounds work[idx..., 1, b] = f(xg) + (ur - f(xr))
end

@kernel function relative_all_kernel!(work, f, @Const(blocks), @Const(firsts),
                                      @Const(clamps), @Const(origins), @Const(spacings),
                                      ::Val{D}, ::Val{G}, ::Val{C},
                                      ::Val{NV}) where {D,G,C,NV}
    I = @index(Global, NTuple)
    t = I[D + 1]
    @inbounds b = blocks[t]
    @inbounds first_ = firsts[t]
    @inbounds cl = clamps[t]
    idx = ntuple(d -> Int(first_[d]) + I[d] - 1, Val(D))
    r = relative_point(idx, cl, Val(D))
    @inbounds origin, h = origins[b], spacings[b]
    poff = pointoffsets(h, C)
    xg = ntuple(d -> origin[d] + (idx[d] - G[d] - poff[d]) * h, Val(D))
    xr = ntuple(d -> origin[d] + (r[d] - G[d] - poff[d]) * h, Val(D))
    fg = f(xg)
    fr = f(xr)
    ntuple(Val(NV)) do v
        @inbounds work[idx..., v, b] = fg[v] + (work[r..., v, b] - fr[v])
        nothing
    end
end

"""
    CellBoundary(g)
    CellBoundary(AllVariables(g))
    CellBoundary(Relative(f))
    CellBoundary(Relative(AllVariables(f)))

A boundary hook expressed **per cell**: `g(x, v, δ) -> value`, with `x`
the cell center, `v` the variable index, and `δ` the outward direction
of the region the cell belongs to.

This is the form that runs on a device. The package launches it as a
kernel over the outward-facing ghost cells, batched by region shape (see
[`BoundaryBatch`](@ref TreeAMR.BoundaryBatch)), so `g` must be a pure
function of its arguments — it never sees the field set and cannot read
the block's interior.

That is the trade. Conditions defined by position alone — Dirichlet
data, a manufactured solution, an analytic exterior — are exactly this
shape. A condition relative to the block's own data takes the
[`Relative`](@ref) form below, which also runs on a device. Anything else
that reads the interior stays with the region form [`fill_ghosts!`](@ref)
also accepts, which is CPU-only. A reflection needs no hook at all:
declare the face `reflecting` on the [`Forest`](@ref) and the schedule
fills it, on every backend (M10); a zero-gradient outflow neither,
declared `outflow`.

Wrapping a function of position alone in [`Relative`](@ref) selects the
relative form, `f(x) -> value`: a ghost becomes `f(x_g) + (u(r) − f(x_r))`,
the case's value plus the evolution's offset from the case at the
nearest evolved point `r`. See [`Relative`](@ref).

Wrapping `g` in [`AllVariables`](@ref) selects the once-per-cell form,
`g(x, δ) -> vals`, which drops the variable index and returns all
`fs.nvars` values as a tuple. That is what a boundary state definable
only as a whole needs — a hydrodynamics code's Dirichlet data is a
primitive state converted to a conserved one — and it matters more here
than for the initial data, because this hook runs at every ghost fill,
hence at every right-hand side evaluation. The two forms write
bit-for-bit the same numbers. The tuple's length is checked on the host
before each launch, at the first owned point of block 1 with the sample
direction `δ = (-1, 0, …, 0)`; any outward direction would do, since
what is checked is how many values come back.

    fill_ghosts!(fs, schedule; boundary = CellBoundary((x, v, δ) -> zero(eltype(x))))
    fill_ghosts!(fs, schedule;                         # nvars == 2
                 boundary = CellBoundary(AllVariables((x, δ) -> (x[1], -x[1]))))

!!! note "Callbacks on a device"
    The callback becomes a kernel argument, so everything it closes over
    must be `isbits`. A captured `Type` is the usual trip: write
    `oftype(x[1], 2)` rather than closing over `T` and calling `T(2)`.
    The same rule covers captured arrays (pass a device array, or index
    the one the callback is already given) and any mutable state, which
    the purity requirement rules out anyway.
"""
struct CellBoundary{F}
    g::F
end

"""
    Relative(f)
    Relative(AllVariables(f))

A boundary condition relative to the block's own data, for
[`CellBoundary`](@ref): every ghost `x_g` beyond an outer face is set to

    u(x_g) = f(x_g) + (u(x_r) − f(x_r)),

with `x_r` the nearest *evolved* point — the case's value `f` plus the
evolution's offset from it there. `f(x)` returns one variable's value,
and is accepted only by a field set with one variable; wrapped in
[`AllVariables`](@ref), `f(x)` returns all `nvars` values as a tuple. `f
≡ 0` is a zero-gradient outflow; `f` the case's analytic field holds the
ghosts to the case's differences while the level follows the evolution —
Dirichlet data "in the field", which keeps `curl A` of a vector
potential the case's in the ghosts while the potential's gauge level is
the evolution's (TreeGRRMHD's AMR-7). The formula is evaluated in that
order, so that data equal to `f` come back as `f(x_g)` bit for bit.

`x_r` is the ghost's point clamped, along each component of its region's
direction that leaves the domain through an outer face, onto the block's
last evolved point there: `G + N` on the high side — the shared high
plane of a vertex-like dimension counts as a ghost — and `G + 1` on the
low side, or `G + 2` for a set with symmetric walls, whose low wall
plane the fill writes. Along every other component the point keeps the
ghost's own index: in a corner region beyond an outer face along one
dimension and across another block's face along another, `x_r` is the
block's own ghost there, filled from the neighbour — never the block's
last column, which would make the value at a point two blocks share
depend on which block fills it. So the hook runs in parts, each after
the stage that fills what it reads: after phase 1, as every hook does,
or, where those ghosts are prolongated from a coarser block, after the
prolongations onto the block's level. Every fill — [`fill_ghosts!`](@ref),
[`regrid!`](@ref)'s fill before its transfer, and
[`adapt_to_initial_data!`](@ref)'s — applies the whole condition, so
nothing reads a partial value.

It is a kernel on every backend. `f` is called twice per ghost, at
`x_g` and `x_r`, and must be a pure function of position: time-dependent
data are captured in `f` by the caller, who builds a fresh hook per fill.
That costs nothing — a closure capturing a `Float64` keeps one type
whatever its value, so the kernel compiled for it is reused — as long as
what it captures is `isbits` and keeps its type from fill to fill: a
`Ref`, or a time that is sometimes a `Rational`, would compile anew.

    hook = CellBoundary(Relative(x -> A_case(x, t)))       # nvars == 1
    fill_ghosts!(fs, schedule; boundary = hook)
"""
struct Relative{F}
    f::F
end

# The cell form runs through the kernel on every backend, the CPU
# included — that is the point of it, and it is what keeps the form
# under test without a device attached. Two dispatch entries rather than
# one so that it wins against the CPU region-form method below without
# an ambiguity.
apply_boundary!(fs::FieldSet{T,D}, hook::CellBoundary, schedule::GhostSchedule{T,D},
                backend::Backend) where {T,D} = cell_boundary!(fs, hook, schedule, backend)
apply_boundary!(fs::FieldSet{T,D}, hook::CellBoundary, schedule::GhostSchedule{T,D},
                backend::CPU) where {T,D} = cell_boundary!(fs, hook, schedule, backend)

function cell_boundary!(fs::FieldSet{T,D}, hook::CellBoundary,
                        schedule::GhostSchedule{T,D}, backend) where {T,D}
    plan = schedule.boundaryplan
    for batch in plan.batches
        n = nregions(batch)
        n == 0 && continue
        blen = batch.boxlen
        boundary_kernel!(backend)(fs.work, hook.g, batch.blocks, batch.directions,
                                  batch.firsts, plan.origins, plan.spacings,
                                  Val(D), Val(fs.G), Val(staggers(fs));
                                  ndrange=(blen..., fs.nvars, n))
    end
    synchronize(backend)
    return nothing
end

# The relative form (AMR-7): only the batches whose slot is `slot`, so
# that each runs after the stage that fills what it reads. The first call
# of a fill, after phase 1, is `apply_boundary!`'s; the others come from
# `boundary_after_stage!`.
function cell_boundary!(fs::FieldSet{T,D}, hook::CellBoundary{<:Relative},
                        schedule::GhostSchedule{T,D}, backend;
                        slot::Int=PHASE1_TAG) where {T,D}
    nblocks(fs) == 0 && return nothing
    f = hook.g.f
    fs.nvars == 1 || throw(ArgumentError(
        "CellBoundary(Relative(f)) sets one variable from `f(x)`, but the field set " *
        "has nvars = $(fs.nvars); for several, wrap the function in AllVariables, " *
        "`CellBoundary(Relative(AllVariables(x -> (u1, u2, ...))))`"))
    return relative_launch!(relative_kernel!(backend), fs, f, schedule, backend, slot)
end

function cell_boundary!(fs::FieldSet{T,D}, hook::CellBoundary{<:Relative{<:AllVariables}},
                        schedule::GhostSchedule{T,D}, backend;
                        slot::Int=PHASE1_TAG) where {T,D}
    nblocks(fs) == 0 && return nothing
    f = hook.g.f.f
    check_allvariables(f(allvariables_sample(fs)), fs, "boundary hook")
    return relative_launch!(relative_all_kernel!(backend), fs, f, schedule, backend, slot,
                            Val(fs.nvars))
end

function relative_launch!(kernel!, fs::FieldSet{T,D}, f, schedule, backend, slot,
                          nv...) where {T,D}
    plan = schedule.boundaryplan
    for batch in plan.batches
        batch.slot == slot || continue
        n = nregions(batch)
        n == 0 && continue
        kernel!(fs.work, f, batch.blocks, batch.firsts, batch.clamps, plan.origins,
                plan.spacings, Val(D), Val(fs.G), Val(staggers(fs)), nv...;
                ndrange=(batch.boxlen..., n))
    end
    synchronize(backend)
    return nothing
end

boundary_after_stage!(fs::FieldSet, hook::CellBoundary{<:Relative}, schedule, backend,
                      tag) = cell_boundary!(fs, hook, schedule, backend; slot=tag)

# The all-variables form. The length check runs here rather than in
# `AllVariables` because only the field set knows `nvars`; it costs one
# host call per ghost fill, against a launch over every outward-facing
# ghost cell. A rank without blocks has no point to check the callback
# at, and no boundary to fill; the hook is not collective, so returning
# alone is safe.
function cell_boundary!(fs::FieldSet{T,D}, hook::CellBoundary{<:AllVariables},
                        schedule::GhostSchedule{T,D}, backend) where {T,D}
    nblocks(fs) == 0 && return nothing
    g = hook.g.f
    δ = ntuple(d -> d == 1 ? -1 : 0, D)            # the sample direction
    check_allvariables(g(allvariables_sample(fs), δ), fs, "boundary hook")

    plan = schedule.boundaryplan
    for batch in plan.batches
        n = nregions(batch)
        n == 0 && continue
        blen = batch.boxlen
        boundary_all_kernel!(backend)(fs.work, g, batch.blocks, batch.directions,
                                      batch.firsts, plan.origins, plan.spacings,
                                      Val(D), Val(fs.G),
                                      Val(staggers(fs)), Val(fs.nvars);
                                      ndrange=(blen..., n))
    end
    synchronize(backend)
    return nothing
end

# The region form. It may do anything to the region it is handed, which
# is exactly why it cannot be run on a device on the caller's behalf: a
# hook that scalar-indexes would fail deep inside a task, with no hint
# of what to do instead. So say it here.
function apply_boundary!(fs::FieldSet{T,D}, hook,
                         schedule::GhostSchedule{T,D}, backend) where {T,D}
    throw(ArgumentError(
        "this boundary hook takes a whole region — `(fs, b, key, δ, region)` — " *
        "which is a host form: it indexes the working array cell by cell, and " *
        "this field set lives on $(nameof(typeof(backend))). Wrap a per-cell " *
        "function in `CellBoundary((x, v, δ) -> ...)`, which the package launches " *
        "as a kernel; a condition relative to the nearest evolved point is " *
        "`CellBoundary(Relative(f))`, a kernel too, and a zero-gradient one an outflow " *
        "face. Any other condition that has to read the block's interior needs the " *
        "CPU backend."))
end

# The region form on the host, as it has always run: one parallel loop
# over the outward-facing regions, the hook free to do as it likes with
# the one it is handed.
function apply_boundary!(fs::FieldSet{T,D}, hook,
                         schedule::GhostSchedule{T,D}, backend::CPU) where {T,D}
    threaded_foreach(length(schedule.boundaries)) do i
        region = schedule.boundaries[i]
        hook(fs, Int(region.block), blockkey(fs, region.block),
             region.direction, region.region)
    end
    return nothing
end

"""
    fill_ghosts!(fs::FieldSet, schedule::GhostSchedule; boundary=nothing)
    fill_ghosts!(pair::RotationPair, (schedule_a, schedule_b); boundary=nothing)

Fill every ghost cell of every block, by replaying `schedule`.

The three cases — same-level copy, restriction from finer neighbors,
prolongation from a coarser neighbor — run in the phases described in
[`GhostSchedule`](@ref): copies and restrictions together first, then
prolongations swept coarsest target first, then the physical boundary
hook. Each phase is one parallel loop with a barrier after it; on the
CPU each thread fills the ghosts of the blocks it owns (see
[`threadchunks`](@ref TreeAMR.threadchunks)). Prolongation levels that
read nothing the levels below them write run as one such loop, and over
a distributed forest with their messages in flight together (see
[`GhostSchedule`](@ref)).

Periodic boundaries need nothing special; the tree wraps around, so they
are ordinary transfers. Reflecting faces need nothing either (M10): the
schedule fills their ghosts by mirrored transfers in the same phases,
with the parity each variable declares on the field set, and the hook
never sees them. That includes the upper wall plane of a vertex-like
dimension, which is derived — zero for an odd variable, the symmetric
interpolant of the prolongation order for an even one.

A rotating seam needs nothing either (M12): its ghosts are the turned
image of real data, filled by transfers from the real blocks across it,
read through the quarter turn and the field set's `rotation`, in the
same phases. A set whose layout is not symmetric under exchanging the
seam's two dimensions turns into another set, and if it has ghosts
there it is refused here: fill it with its partner, as a
[`RotationPair`](@ref), through the second form. That form runs the two
schedules — each set's own — merged stage by stage, each set's phase 1,
then both hooks, then each prolongation level of one set and of the
other, so that a prolongation may read its partner's coarser ghosts.
Its `boundary` is one hook for both sets, or a tuple of two.

`boundary` is called once per ghost region facing outside the domain
through an *outer* face — neither periodic nor reflecting — as

    boundary(fs, blockindex, key, δ, region)

with `key` the block's [`MortonKey`](@ref), `δ` the outward direction,
and `region` the `CartesianIndices` of the ghost cells in stored
coordinates. Use [`coordinates`](@ref) to get their positions. Passing
`nothing` leaves those ghosts untouched, which is what a fully periodic
domain wants.

In a vertex-like dimension the outward region on the domain's **high**
side starts at the shared boundary plane `G+N+1`, one plane further in
than a ghost slab: those points belong to nobody, since ownership is
half-open, so the hook fills them and the application never evolves them.
The low boundary points are owned and evolved as usual. This asymmetry is
the price of a uniform `N^D` state layout for every centering; see
"Centerings" in `CODE.md`.

!!! note "Where the boundary hook runs"
    The hook runs after the copies and restrictions but **before** the
    prolongation sweep, not after every inter-block phase. A block
    sitting against the domain edge has prolongation stencils that reach
    *tangentially* past that edge, into the coarse source's own outer
    ghosts; filling those last would feed unwritten memory into the
    interpolation. The hook may therefore read the block's interior
    (as an extrapolating condition does) but not other blocks' ghosts.
    The [`Relative`](@ref) form reads the block's own ghosts across a
    block face too, and runs each of its regions after the stage that
    fills those — phase 1, or the prolongations onto the block's level.

!!! note "Over a distributed forest"
    The fill is collective (M7): every rank calls it, with its own field
    set over its own blocks, in the same order relative to the other
    collective calls. A ghost whose source block lives on another rank is
    computed there and sent; the hook runs on each rank for its own
    blocks, in the same place as serially. The ghosts come out bit for
    bit as a serial fill writes them, whatever the number of ranks. Two
    fills over one forest must not run concurrently from two tasks.

!!! note "The hook is called concurrently"
    The boundary regions are a parallel loop like every other phase
    (M5): the hook runs on several threads at once, once per region.
    Regions are disjoint, so a hook that writes only the `region` it was
    handed needs nothing further; one that accumulates into shared state
    of its own must synchronize itself.
"""
function fill_ghosts!(fs::FieldSet{T,D}, schedule::GhostSchedule{T,D};
                      boundary=nothing) where {T,D}
    backend = check_fill(fs, schedule)
    # A set whose layout is not symmetric in the seam's plane turns into
    # its partner, so its ghosts across the seam are the partner's data
    # (M12). It has such ghosts exactly when it has ghosts along either
    # dimension of the plane — every block on a seam face then has a
    # region beyond it — which is a property of the layout, the same on
    # every rank.
    if hasrotating(fs.forest) && !symmetric_layout(fs) && seam_ghosts(fs)
        throw(ArgumentError(
            "this field set's layout is not symmetric in the rotating seam's " *
            "dimensions $(rotating_dims(fs.forest)) (G = $(fs.G), centering " *
            "$(fs.centering)), so its ghosts across the seam are a quarter turn of " *
            "another set's data, which a fill of this set alone does not have. Fill " *
            "it as a RotationPair with the set of the exchanged layout: " *
            "`fill_ghosts!(RotationPair(a, b), (schedule_a, schedule_b))`."))
    end
    return exchange_ghosts!(fs, schedule, boundary, backend)
end

# Whether a set has ghost regions across a rotating seam: ghosts along
# either dimension of the seam's plane. A set with none there, as a
# flux's `G = 0` along a stagger, has only regions on the high side of
# its blocks there, and nothing crosses the seam.
function seam_ghosts(fs::FieldSet)
    d1, d2 = rotating_dims(fs.forest)
    return fs.G[d1] > 0 || fs.G[d2] > 0
end

# The checks every fill makes of a field set and the schedule it is
# replayed with, returning the backend. Rank-local, on purpose: see
# "Collective checks" in CODE.md.
function check_fill(fs::FieldSet{T,D}, schedule::GhostSchedule{T,D}) where {T,D}
    schedule.forest === fs.forest || throw(ArgumentError(
        "schedule was built for a different forest than the field set"))
    isstale(schedule) && throw(ArgumentError(
        "the forest changed since this schedule was built (generation " *
        "$(schedule.generation) -> $(generation(schedule.forest))); rebuild it"))
    nblocks(fs) == length(blockrange(schedule.forest)) || throw(ArgumentError(
        "field set has $(nblocks(fs)) blocks but the schedule's forest has " *
        "$(length(blockrange(schedule.forest))) on this rank; rebuild both"))
    fs.G == schedule.G || throw(ArgumentError(
        "the field set has ghost width G=$(fs.G) but this schedule was built for " *
        "G=$(schedule.G); every target range and stencil in it is wrong for this " *
        "layout. A schedule belongs to a layout, not to a forest: build one with " *
        "`GhostSchedule(fs, operators)`."))
    fs.centering == schedule.centering || throw(ArgumentError(
        "the field set has centering $(fs.centering) but this schedule was built " *
        "for $(schedule.centering); the stored extent, the target ranges and the " *
        "one-dimensional operators all differ between a cell-centered and a " *
        "vertex-like dimension. Build one with `GhostSchedule(fs, operators)`."))
    fs.walls === schedule.walls || throw(ArgumentError(
        "the field set has walls = :$(fs.walls) but this schedule was built for " *
        ":$(schedule.walls); the target ranges at a low domain face differ between " *
        "the two. Build one with `GhostSchedule(fs, operators)`."))

    backend = get_backend(fs.work)
    samebackend(backend, schedule.backend) || throw(ArgumentError(
        "the field set lives on $(nameof(typeof(backend))) but this schedule was " *
        "built for $(nameof(typeof(schedule.backend))); its stencils are in the " *
        "wrong memory. Build it with " *
        "`GhostSchedule(forest, operators; backend = $(nameof(typeof(backend)))())`, " *
        "or let both default to the CPU."))
    return backend
end

# The element types have to agree exactly — the transfer accumulates in
# `eltype(dest)` and reads the weights straight out of the stencils, so a
# mismatch would silently promote in the innermost loop. Caught here with
# a reason rather than left to a `MethodError`, since the two types are
# chosen at two different call sites.
fill_ghosts!(fs::FieldSet{T}, schedule::GhostSchedule{S};
             boundary=nothing) where {T,S} = check_fill(fs, schedule)
function check_fill(fs::FieldSet{T}, schedule::GhostSchedule{S}) where {T,S}
    throw(ArgumentError(
        "the field set stores $T but this schedule carries $S weights; build " *
        "the schedule with `GhostSchedule(forest, operators; T=$T)`, or let " *
        "both default to the forest's floattype"))
end

function fill_ghosts!(pair::RotationPair, schedules::Tuple{GhostSchedule,GhostSchedule};
                      boundary=nothing)
    a, b = pair.a, pair.b
    sa, sb = schedules
    backend = check_fill(a, sa)
    check_fill(b, sb)
    hooks = boundary isa Tuple ? boundary : (boundary, boundary)
    length(hooks) == 2 || throw(ArgumentError(
        "boundary for a RotationPair is one hook for both sets or a tuple of two, " *
        "one per set; got a tuple of $(length(hooks))"))
    return exchange_pair!(pair, sa, sb, hooks, backend)
end

# The paired fill (M12), once its checks have passed: the two schedules'
# stages merged by (stage, member). Phase 1 of `a` and then of `b`, then
# the two hooks, then each phase-2 target level of `a` and then of `b`,
# a stage that only one member has running alone. A rotated
# prolongation of one member may read its partner's coarser ghosts,
# which the partner's earlier stages, phase 1 or a coarser level, have
# filled by then, as its own coarser ghosts are in a single fill. Under
# MPI each stage completes its receives before the next starts, so the
# two members share a stage's tag: messages between two ranks with one
# tag are matched in the order they were sent.
function exchange_pair!(pair::RotationPair, sa::GhostSchedule, sb::GhostSchedule, hooks,
                        backend)
    a, b = pair.a, pair.b
    forest = sa.forest
    runa(i, sends) = run_stages!(a.work, a.work, a.nvars, pair.afactors, sa.stages, i:i,
                                 sa.stages[i].locals, forest, backend, sends, b.work,
                                 pair.arotvars)
    runb(i, sends) = run_stages!(b.work, b.work, b.nvars, pair.bfactors, sb.stages, i:i,
                                 sb.stages[i].locals, forest, backend, sends, a.work,
                                 pair.brotvars)
    sends = runa(1, nothing)
    sends = runb(1, sends)
    hooks[1] === nothing || apply_boundary!(a, hooks[1], sa, backend)
    hooks[2] === nothing || apply_boundary!(b, hooks[2], sb, backend)
    tags = sort!(unique!([[st.tag for st in sa.stages[2:end]];
                          [st.tag for st in sb.stages[2:end]]]))
    for tag in tags
        i = findfirst(st -> st.tag == tag, sa.stages)
        i === nothing || (sends = runa(i, sends))
        j = findfirst(st -> st.tag == tag, sb.stages)
        j === nothing || (sends = runb(j, sends))
        hooks[1] === nothing || boundary_after_stage!(a, hooks[1], sa, backend, tag)
        hooks[2] === nothing || boundary_after_stage!(b, hooks[2], sb, backend, tag)
    end
    sends === nothing || waitall(forest.comm, sends)
    return pair
end

"""
    boundary_by_coordinates(f)
    boundary_by_coordinates(AllVariables(f))

A boundary hook that sets each outer ghost cell from `f(x, v)`, with `x`
the cell center and `v` the variable index — the same signature
[`fill_by_coordinates!`](@ref) takes.

Useful when the exact solution is known (manufactured solutions,
convergence tests); real applications supply their own hook to impose
outgoing, reflecting, or symmetry conditions.

A [`CellBoundary`](@ref) that ignores the direction, so it runs on every
backend. Wrapped in [`AllVariables`](@ref) it is
`CellBoundary(AllVariables((x, δ) -> f(x)))`: the same hook built from a
once-per-cell `f(x) -> vals`, which is the shape of a Dirichlet
condition set from initial data that is itself stated all at once.
"""
boundary_by_coordinates(f) = CellBoundary((x, v, δ) -> f(x, v))
boundary_by_coordinates(w::AllVariables) = CellBoundary(AllVariables((x, δ) -> w.f(x)))
