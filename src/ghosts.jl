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

# Tensor-product stencil: `prod(Ps)` contributions, the weight of each
# the product of its D one-dimensional weights. The widths are **per
# dimension** (M8): a transfer can be injection in one dimension and
# order-p interpolation in another, and padding the narrow one to the
# common width with zero weights would read slots that need not hold
# data at all — a G = 0 face field has none beyond its high face — and
# `0 * NaN` is `NaN`.
#
# `Ps` is a compile-time constant, so the sum is *generated* as a loop
# nest with literal trip counts instead of iterating
# `CartesianIndices(Ps)`. That iterator spent its time in `__inc`, and
# it hid the fact that D-1 of the D weight loads per stencil point are
# invariant in the inner loops.
#
# The nest reproduces the old loop exactly, not merely to the same
# accuracy: `m_1` runs innermost, as column-major `CartesianIndices`
# iteration did, so the contributions are summed in the same order; and
# the weight product is still formed as `((w₁ * w₂) * …) * w_D`, since
# floating-point multiplication does not associate and hoisting a
# partial product out of the inner loop would reassociate it. Only the
# *loads* are hoisted.
@generated function stencil_sum(src, weights, base::NTuple{D,Int},
                                wcol::NTuple{D,Int}, v, sblock,
                                ::Val{Ps}) where {D,Ps}
    idx = [:(base[$d] + $(Symbol(:m_, d)) - 1) for d in 1:D]
    wprod = foldl((a, d) -> :($a * $(Symbol(:wt_, d))), 2:D;
                  init=Symbol(:wt_, 1))
    body = quote
        acc += $wprod * src[$(idx...), v, sblock]
    end
    for d in 1:D
        body = quote
            for $(Symbol(:m_, d)) in 1:$(Ps[d])
                $(Symbol(:wt_, d)) = weights[$d][$(Symbol(:m_, d)), wcol[$d]]
                $body
            end
        end
    end
    return quote
        Base.@_inline_meta
        acc = zero(eltype(src))
        @inbounds $body
        acc
    end
end

# `dest` and `src` are the same array for ghost filling (targets are
# ghosts, sources interiors, so they never overlap) and different arrays
# when regridding transfers into freshly allocated storage. Neither is
# marked @Const, so the aliasing case stays well defined.
#
# The launch is over the target box itself — `ndrange = (blen…, nvars,
# ntransfers)` — so the backend supplies the per-axis position and the
# kernel does no index arithmetic to recover it. Flattening the box into
# one axis and unflattening it here cost an integer `div` and `rem` per
# dimension per ghost point per variable: 9 % of a copy-dominated fill's
# self time, and the single largest entry after the bounds checks.
@kernel function transfer_kernel!(dest, src,
                                  @Const(targetblocks), @Const(sourceblocks),
                                  srcstarts, weights,
                                  targetfirst::NTuple{D,Int}, toffset::Int,
                                  ::Val{Ps}, ::Val{D}) where {Ps,D}
    I = @index(Global, NTuple)
    # `I[1:D]` is the position within the target region, one-based.
    v = I[D + 1]
    t = I[D + 2] + toffset

    @inbounds tblock = targetblocks[t]
    @inbounds sblock = sourceblocks[t]

    wcol = ntuple(d -> I[d], Val(D))
    tidx = ntuple(d -> targetfirst[d] + I[d] - 1, Val(D))
    @inbounds base = ntuple(d -> Int(srcstarts[d][I[d]]), Val(D))

    acc = stencil_sum(src, weights, base, wcol, v, sblock, Val(Ps))
    @inbounds dest[tidx..., v, tblock] = acc
end

# Launch one group's transfers `range` — the whole group by default.
# `single` asks KernelAbstractions for a *single* workgroup, so the
# launch runs inline on the calling task instead of spawning its own:
# that is what lets a phase be one parallel loop over slices rather than
# a nest of parallel loops.
function run_group!(dest, src, group::TransferGroup{T,D}, nvars::Integer, backend;
                    range=1:ntransfers(group), single::Bool=false) where {T,D}
    n = length(range)
    n == 0 && return nothing
    blen = boxsize(group)
    prod(blen) == 0 && return nothing
    tfirst = ntuple(d -> group.stencils[d].targetfirst, Val(D))
    # One width per dimension, not one shared width: see the kernel.
    orders = ntuple(d -> stencilorder(group.stencils[d]), Val(D))
    srcstarts = ntuple(d -> group.stencils[d].srcstart, Val(D))
    weights = ntuple(d -> group.stencils[d].weights, Val(D))
    ndrange = (blen..., Int(nvars), n)

    # A slice of a group is passed as an offset into the group's own
    # block lists rather than as two `view`s: every kernel argument
    # lands in a tuple that KernelAbstractions heap-allocates on each
    # launch, and two `SubArray`s are the largest arguments here. See
    # "What the ghost fill costs" in CODE.md for what that buys and what
    # it does not.
    kernel! = transfer_kernel!(backend)
    kernel!(dest, src, group.targetblocks, group.sourceblocks,
            srcstarts, weights, tfirst, first(range) - 1, Val(orders), Val(D);
            ndrange=ndrange, workgroupsize=(single ? ndrange : nothing))
    return nothing
end

run_group!(fs::FieldSet{T,D}, group::TransferGroup{T,D}, backend) where {T,D} =
    run_group!(fs.work, fs.work, group, fs.nvars, backend)

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
# So the phase is flattened into `PhaseSlice`s of roughly equal cell
# count and the slices are dealt out largest first, one task per thread,
# each slice launching as a single inline workgroup. A device backend
# has no such problem — there a launch *is* the parallel unit — and
# takes the plain per-group path below.
function run_phase!(dest, src, groups, plan, nvars::Integer, backend)
    for group in groups
        run_group!(dest, src, group, nvars, backend)
    end
    return nothing
end

function run_phase!(dest, src, groups, plan, nvars::Integer, backend::CPU)
    ntasks = length(threadchunks(length(plan)))
    if ntasks <= 1
        for group in groups
            run_group!(dest, src, group, nvars, backend)
        end
        return nothing
    end
    threaded_foreach(ntasks) do c
        i = c
        while i <= length(plan)
            slice = plan[i]
            run_group!(dest, src, groups[slice.group], nvars, backend;
                       range=Int(slice.first):Int(slice.last), single=true)
            i += ntasks
        end
    end
    return nothing
end

run_phase!(fs::FieldSet{T,D}, groups, plan, backend) where {T,D} =
    run_phase!(fs.work, fs.work, groups, plan, fs.nvars, backend)

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

"""
    CellBoundary(g)
    CellBoundary(AllVariables(g))

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
shape. Conditions that read the interior (reflecting, extrapolating
outflow) are not, and stay with the region form
[`fill_ghosts!`](@ref) also accepts, which is CPU-only.

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

# The all-variables form. The length check runs here rather than in
# `AllVariables` because only the field set knows `nvars`; it costs one
# host call per ghost fill, against a launch over every outward-facing
# ghost cell.
function cell_boundary!(fs::FieldSet{T,D}, hook::CellBoundary{<:AllVariables},
                        schedule::GhostSchedule{T,D}, backend) where {T,D}
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
        "as a kernel. A condition that has to read the block's interior has no " *
        "device form yet and needs the CPU backend."))
end

# The region form on the host, as it has always run: one parallel loop
# over the outward-facing regions, the hook free to do as it likes with
# the one it is handed.
function apply_boundary!(fs::FieldSet{T,D}, hook,
                         schedule::GhostSchedule{T,D}, backend::CPU) where {T,D}
    threaded_foreach(length(schedule.boundaries)) do i
        region = schedule.boundaries[i]
        hook(fs, Int(region.block), fs.forest.leaves[region.block],
             region.direction, region.region)
    end
    return nothing
end

"""
    fill_ghosts!(fs::FieldSet, schedule::GhostSchedule; boundary=nothing)

Fill every ghost cell of every block, by replaying `schedule`.

The three cases — same-level copy, restriction from finer neighbors,
prolongation from a coarser neighbor — run in the phases described in
[`GhostSchedule`](@ref): copies and restrictions together first, then
prolongations swept coarsest target first, then the physical boundary
hook. Each phase is one parallel loop with a barrier after it; how the
phase is cut up for the threads is an implementation detail of that
loop (see `PhaseSlice`).

Periodic boundaries need nothing special; the tree wraps around, so they
are ordinary transfers.

`boundary` is called once per ghost region facing outside a non-periodic
domain, as

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
    (as reflecting and extrapolating conditions do) but not other
    blocks' ghosts.

!!! note "The hook is called concurrently"
    The boundary regions are a parallel loop like every other phase
    (M5): the hook runs on several threads at once, once per region.
    Regions are disjoint, so a hook that writes only the `region` it was
    handed needs nothing further; one that accumulates into shared state
    of its own must synchronize itself.
"""
function fill_ghosts!(fs::FieldSet{T,D}, schedule::GhostSchedule{T,D};
                      boundary=nothing) where {T,D}
    schedule.forest === fs.forest || throw(ArgumentError(
        "schedule was built for a different forest than the field set"))
    isstale(schedule) && throw(ArgumentError(
        "the forest changed since this schedule was built (generation " *
        "$(schedule.generation) -> $(generation(schedule.forest))); rebuild it"))
    nblocks(fs) == nleaves(schedule.forest) || throw(ArgumentError(
        "field set has $(nblocks(fs)) blocks but the schedule's forest has " *
        "$(nleaves(schedule.forest)) leaves; rebuild both"))
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

    backend = get_backend(fs.work)
    samebackend(backend, schedule.backend) || throw(ArgumentError(
        "the field set lives on $(nameof(typeof(backend))) but this schedule was " *
        "built for $(nameof(typeof(schedule.backend))); its stencils are in the " *
        "wrong memory. Build it with " *
        "`GhostSchedule(forest, operators; backend = $(nameof(typeof(backend)))())`, " *
        "or let both default to the CPU."))

    # Phase 1: same-level copies and restrictions. Both read interiors
    # only, so they cannot race with each other.
    run_phase!(fs, schedule.phase1, schedule.phase1plan, backend)
    synchronize(backend)

    # Physical boundaries, before prolongation rather than after
    # everything: a block against the domain edge has prolongation
    # stencils that reach tangentially past that edge into its coarse
    # source's outer ghosts, so those must already hold data.
    if boundary !== nothing
        apply_boundary!(fs, boundary, schedule, backend)
    end

    # Phase 2: prolongations, coarsest targets first. A prolongation may
    # read its coarse source's ghosts, which the earlier sweeps filled.
    for (groups, plan) in zip(schedule.phase2, schedule.phase2plans)
        run_phase!(fs, groups, plan, backend)
        synchronize(backend)
    end
    return fs
end

# The element types have to agree exactly — the transfer accumulates in
# `eltype(dest)` and reads the weights straight out of the stencils, so a
# mismatch would silently promote in the innermost loop. Caught here with
# a reason rather than left to a `MethodError`, since the two types are
# chosen at two different call sites.
function fill_ghosts!(fs::FieldSet{T}, schedule::GhostSchedule{S};
                      boundary=nothing) where {T,S}
    throw(ArgumentError(
        "the field set stores $T but this schedule carries $S weights; build " *
        "the schedule with `GhostSchedule(forest, operators; T=$T)`, or let " *
        "both default to the forest's floattype"))
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
