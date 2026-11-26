# Coupling to ODE integrators.
#
# The state vector holds leaf *interiors only*: it is what an integrator
# sees, and it never contains ghosts. The working array is scratch,
# refreshed at every RHS evaluation. So a right-hand side reads
#
#     scatter!(fs, u)          # flat vector -> working array interiors
#     fill_ghosts!(fs, sched)  # copies, restrictions, prolongations
#     map_blocks!(...)         # application kernels, writing du directly
#
# The integrator never sees ghosts and the RHS never mutates `u`. The
# cost is one scatter per evaluation, which `CODE.md` accepts in exchange
# for not spending integrator bandwidth on ghost memory.

"""
    statelength(fs::FieldSet)

Number of entries in this field set's state vector: `∏(N + c_d) · nvars ·
nblocks`, every own point of every block, with `c_d` the stagger of
dimension `d` ([`staggers`](@ref)): `N^D` per block and variable for a
cell-centered or element set, and one plane more along each vertex-like
dimension, every copy of a shared point (M22, "Closed blocks" in
`CODE.md`).
"""
statelength(fs::FieldSet{T,D}) where {T,D} = prod(ownextents(fs)) * fs.nvars * nblocks(fs)

"""
    statevector(fs::FieldSet)

A freshly allocated, zeroed state vector for `fs` — the `u` an ODE
integrator advances. Use [`gather!`](@ref) to load the field set's
current interior values into it.

Allocated on the field set's own backend, so a device-resident field set
gets a device-resident state vector and the integrator never touches
host memory (M6).
"""
function statevector(fs::FieldSet{T}) where {T}
    backend = get_backend(fs.work)
    u = allocate(backend, T, (statelength(fs),))
    zerofill!(statearray(u, fs), backend)          # first touch by block owner
    return u
end

"""
    statearray(u, fs::FieldSet)

View a flat state vector as an `(N+c[1], ..., N+c[D], nvars, nblocks)`
array, sharing memory, `c` the set's [`staggers`](@ref): a plain reshape.
This is the layout application kernels write `du` in; the index order
matches the working array, minus the ghosts.
"""
function statearray(u::AbstractVector, fs::FieldSet{T,D}; kwargs...) where {T,D}
    check_removed_appendix(kwargs, "statearray")
    check_statelength(u, fs)
    shape = (ownextents(fs)..., fs.nvars, nblocks(fs))
    return state_array(u, shape)
end

# The state as the one array type `reshape(u, …)` gives, so that every
# set's `statearray` has one type and a kernel taking it compiles once. On
# the host an `Array` over `u`'s own memory; elsewhere a reshaped vector,
# which on a device is the device's own array type.
state_array(u::AbstractVector, shape) = reshape(u, shape)
state_array(u::Vector, shape) = Base.wrap(Array, u.ref, shape)

# The keywords closed blocks (M22) replace: `appendix`, a set
# interpolated across a glue's owned glued high face planes (M20), and
# `closed`, the launch over the closed range, which is now every block's
# own range. Say so, rather than a `MethodError` about a keyword.
function check_removed_appendix(kwargs, what)
    isempty(kwargs) && return nothing
    haskey(kwargs, :appendix) && throw(ArgumentError(
        "$what no longer takes `appendix`: closed blocks (M22, \"Closed blocks\" in " *
        "CODE.md) make the planes the appendix held part of every block's own range, " *
        "which the state and the launch cover without a keyword"))
    haskey(kwargs, :closed) && throw(ArgumentError(
        "$what no longer takes `closed`: closed blocks (M22, \"Closed blocks\" in " *
        "CODE.md) make the closed range, both face planes of a vertex-like dimension, " *
        "every block's own range, which the launch covers without a keyword; " *
        "`stored = true` still covers every stored point"))
    throw(ArgumentError("$what got an unsupported keyword: $(join(keys(kwargs), ", "))"))
end

function check_statelength(u::AbstractVector, fs::FieldSet)
    length(u) == statelength(fs) || throw(DimensionMismatch(
        "state vector has $(length(u)) entries but this field set needs " *
        "$(statelength(fs))"))
    return nothing
end

# The two copies between the state vector and the working array. On a
# device they launch over one linear axis, the state vector's own index,
# and recover `(i1..iD, var, block)` through `shape` (see
# `launch_positional!`): the block-shaped launch made a device divide
# twice per axis per value to form the index, which held these copies to a
# fifth of the memory bandwidth ("The copy kernels on a device" in
# CODE.md). On the CPU the launch is block-shaped as before. The state
# vector is read and written linearly either way, at the launch's own
# linear index, which is the state vector's under both launches.
@kernel function scatter_kernel!(shape, work, @Const(state), ::Val{D},
                                 ::Val{G}) where {D,G}
    J = @index(Global, NTuple)           # (KA's CPU emitter wants it on its own)
    n = @index(Global, Linear)
    I = kernel_position(shape, J)                  # (i1..iD, var, block)
    @inbounds work[ntuple(d -> I[d] + G[d], Val(D))..., I[D + 1], I[D + 2]] = state[n]
end

@kernel function gather_kernel!(shape, state, @Const(work), ::Val{D},
                                ::Val{G}) where {D,G}
    J = @index(Global, NTuple)
    n = @index(Global, Linear)
    I = kernel_position(shape, J)
    @inbounds state[n] = work[ntuple(d -> I[d] + G[d], Val(D))..., I[D + 1], I[D + 2]]
end

# The indices are the launch's own and the shape's, so the accesses are
# `@inbounds`, which CI's `check_bounds=yes` overrides as it does for the
# transfer kernel.
function run_over_interiors!(kernel, fs::FieldSet{T,D}, a, b;
                             flat::Bool=flatlaunch(get_backend(fs.work))) where {T,D}
    backend = get_backend(fs.work)
    launch_positional!(kernel, backend, a, b, Val(D), Val(fs.G); flat=flat,
                       extents=(ownextents(fs)..., fs.nvars, nblocks(fs)))
    synchronize(backend)
    return nothing
end

"""
    scatter!(fs::FieldSet, u::AbstractVector)

Copy a state vector into the working array's interior cells, leaving
ghosts untouched. The first step of every RHS evaluation; follow it with
[`fill_ghosts!`](@ref).
"""
function scatter!(fs::FieldSet{T,D}, u::AbstractVector) where {T,D}
    check_statelength(u, fs)
    run_over_interiors!(scatter_kernel!, fs, fs.work, u)
    return fs
end

"""
    gather!(u::AbstractVector, fs::FieldSet)

Copy the working array's interior cells into a state vector — the
inverse of [`scatter!`](@ref).

An application's RHS kernels normally write `du` in state layout
directly, fusing this away; `gather!` is for setting up initial data and
for reading results back out.
"""
function gather!(u::AbstractVector, fs::FieldSet{T,D}) where {T,D}
    check_statelength(u, fs)
    run_over_interiors!(gather_kernel!, fs, u, fs.work)
    return u
end

# Several field sets in one state vector.
#
# Constrained-transport MHD evolves cell-centered hydro variables and
# face-centered `B` together, and the integrator sees one vector. The
# sets' segments are concatenated in tuple order, each in exactly the
# single-set layout, so every single-set function applied to a segment's
# view behaves as on a vector of its own ("Time integration" in
# CODE.md). The functions below are therefore loops over segments, and
# the per-segment work is the single-set code unchanged.

"""
    statelength(sets::Tuple{Vararg{FieldSet}})

Number of entries in the state vector of several field sets: the sum of
their single-set lengths. See [`statevector`](@ref) for the layout.
"""
function statelength(sets::Tuple{Vararg{FieldSet}})
    check_sets(sets)
    return sum(statelength, sets)
end

"""
    staterange(sets::Tuple{Vararg{FieldSet}}, i::Integer) -> UnitRange{Int}

The entries of the `i`-th set's segment in the state vector of `sets`.
Segments follow each other in tuple order, so the range starts after the
lengths of the sets before it.
"""
function staterange(sets::Tuple{Vararg{FieldSet}}, i::Integer)
    check_sets(sets)
    1 <= i <= length(sets) || throw(ArgumentError(
        "set index $i is outside 1:$(length(sets)): the state vector has one " *
        "segment per field set in the tuple"))
    start = sum(j -> statelength(sets[j]), 1:(i - 1); init=0)
    return (start + 1):(start + statelength(sets[i]))
end

"""
    statevector(sets::Tuple{Vararg{FieldSet}})

A freshly allocated, zeroed state vector for several field sets over one
forest: their segments concatenated in tuple order, each in exactly the
layout `statevector(sets[i])` has. That is what makes every single-set
function apply to a segment — [`scatter!`](@ref), [`gather!`](@ref),
[`statearray`](@ref), [`block_mapreduce`](@ref), [`mesh_mapreduce`](@ref)
and [`volume_weighted_norm`](@ref) on [`stateview`](@ref)`(u, sets, i)`
and `sets[i]` behave as on a separate vector.

The sets may mix centerings, `nvars` and ghost widths, but must share
the forest, the backend and the element type, and each must hold this
rank's blocks of the forest's current leaves.

```julia
u = statevector((hydro, Bx, By, Bz))
gather!(u, (hydro, Bx, By, Bz))
```
"""
function statevector(sets::Tuple{Vararg{FieldSet}})
    check_sets(sets)
    fs = first(sets)
    backend = get_backend(fs.work)
    u = allocate(backend, eltype(fs.work), (statelength(sets),))
    # Segment by segment, each through its own set's blocks, so every
    # block's entries are first touched by its owner as for one set.
    for i in eachindex(sets)
        zerofill!(statearray(u, sets, i), backend)
    end
    return u
end

"""
    stateview(u::AbstractVector, sets::Tuple{Vararg{FieldSet}}, i::Integer)

The `i`-th set's segment of the state vector `u`, as a view sharing
memory with it: `view(u, staterange(sets, i))`. Pass it with `sets[i]`
to any single-set function.
"""
function stateview(u::AbstractVector, sets::Tuple{Vararg{FieldSet}}, i::Integer)
    check_statelength(u, sets)
    return view(u, staterange(sets, i))
end

"""
    statearray(u::AbstractVector, sets::Tuple{Vararg{FieldSet}}, i::Integer)

The `i`-th set's segment of `u` as that set's `(N, ..., N, nvars,
nblocks)` array, sharing memory: `statearray(stateview(u, sets, i),
sets[i])`.
"""
function statearray(u::AbstractVector, sets::Tuple{Vararg{FieldSet}}, i::Integer;
                    kwargs...)
    check_removed_appendix(kwargs, "statearray")
    return statearray(stateview(u, sets, i), sets[i])
end

"""
    scatter!(sets::Tuple{Vararg{FieldSet}}, u::AbstractVector)

[`scatter!`](@ref) for each set from its own segment of `u`.
"""
function scatter!(sets::Tuple{Vararg{FieldSet}}, u::AbstractVector)
    check_statelength(u, sets)
    for i in eachindex(sets)
        scatter!(sets[i], stateview(u, sets, i))
    end
    return sets
end

"""
    gather!(u::AbstractVector, sets::Tuple{Vararg{FieldSet}})

[`gather!`](@ref) of each set into its own segment of `u`.
"""
function gather!(u::AbstractVector, sets::Tuple{Vararg{FieldSet}})
    check_statelength(u, sets)
    for i in eachindex(sets)
        gather!(stateview(u, sets, i), sets[i])
    end
    return u
end

# What makes a tuple of field sets one state vector: one forest, one
# backend, one element type, and every set holding this rank's blocks of
# the forest's current leaves. The block count is compared with the
# rank's own range, never with `nleaves`, which counts every rank's.
function check_sets(sets::Tuple{Vararg{FieldSet}})
    isempty(sets) && throw(ArgumentError(
        "a state vector over several field sets needs at least one set; for one " *
        "set, the single-set functions take the field set itself"))
    fs = first(sets)
    forest = fs.forest
    backend = get_backend(fs.work)
    T = eltype(fs.work)
    for (i, s) in enumerate(sets)
        s.forest === forest || throw(ArgumentError(
            "field set $i is over a different forest from field set 1: the sets of " *
            "one state vector are regridded together, so they must share the forest"))
        samebackend(get_backend(s.work), backend) || throw(ArgumentError(
            "field set $i is on $(nameof(typeof(get_backend(s.work)))) and field set 1 " *
            "on $(nameof(typeof(backend))): the state vector is one array, on one " *
            "backend"))
        eltype(s.work) === T || throw(ArgumentError(
            "field set $i has element type $(eltype(s.work)) and field set 1 has $T: " *
            "the state vector is one array, of one element type"))
        nblocks(s) == length(blockrange(forest)) || throw(ArgumentError(
            "field set $i has $(nblocks(s)) blocks but the forest has " *
            "$(length(blockrange(forest))) on this rank: the set was not resized " *
            "with the forest's last regrid (pass it to `regrid!`, as `fs => nothing` " *
            "if it holds nothing worth moving)"))
    end
    return nothing
end

function check_statelength(u::AbstractVector, sets::Tuple{Vararg{FieldSet}})
    n = statelength(sets)
    length(u) == n || throw(DimensionMismatch(
        "state vector has $(length(u)) entries but these $(length(sets)) field sets " *
        "need $n"))
    return nothing
end

"""
    map_blocks!(kernel!, fs::FieldSet, args...; stored=false, workgroupsize=nothing)

Launch a KernelAbstractions kernel over every **own** point of every
block, with `ndrange = (N+c[1], ..., N+c[D], nblocks)`, `c` the set's
[`staggers`](@ref): `N` points along a cell-like dimension, and `N + 1`
along a vertex-like one, both face planes, which the block owns (M22,
"Closed blocks" in `CODE.md`); a flux has `N+1` faces per dimension. The
kernel's global index is therefore `(i1, ..., iD, b)`, each `i` an index
into the own range, and it adds `G[d]` to reach the working array's
stored indices; it is also the index of [`statearray`](@ref).

With `stored = true` the loop runs over every **stored** point of every
block, ghosts included — `ndrange = (size(fs.work)[1:D]..., nblocks(fs))`,
that is `N + 2G[d] + c[d]` per dimension.

!!! warning "The two forms index differently"
    Under `stored = true` the kernel's global index **is** the stored
    index: there is nothing to add. Under the default it is an offset
    into the own range and the kernel adds `G[d]`. A kernel written for
    one form is wrong under the other — silently so, since both are in
    bounds — so a kernel meant for the stored form should not add `G`
    anywhere.

The `closed` keyword of half-open blocks is gone with them: its range is
the default's.

The stored form exists because a pointwise pass sometimes has to cover
the ghosts too. The case that asked for it is a hydrodynamics code
exchanging *conserved* variables and recovering *primitive* ones from
them: its reconstruction reads primitives two cells into the
neighbours, so the recovery has to have run in the ghost cells as well.
Which points a block stores is the mesh's business, not the
application's, so the launch says `stored = true` rather than the
application spelling out `N + 2G + c` for itself (see `CODE.md`,
"Application interface").

Blocks are uniform work units, so this is one flat parallel loop. On
the CPU it runs *by owner* (see [`launch_by_owner!`](@ref TreeAMR.launch_by_owner!)):
every block on the same thread as in every other per-block pass of the
package, which is worth up to 2.4x on a many-core node, and one
workgroup per block — so a kernel handed to `map_blocks!` must not
depend on the workgroup size. The same launch runs on a device; every
work item writes its own output cell, so the result does not depend on
how the loop was split.

`workgroupsize` is handed to the KernelAbstractions launch on a device,
where a kernel heavy in registers may want smaller workgroups than the
backend picks; `nothing` leaves the choice to the backend. The CPU
ignores it, since there the workgroup is the block, which is what keeps
every block on its owner's thread — so code written once for every
backend may pass it unconditionally.

```julia
@kernel function rhs!(du, @Const(work), @Const(h), ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)
    b = I[D + 1]
    c = ntuple(d -> I[d] + G[d], Val(D))       # stored (ghosted) index
    du[ntuple(d -> I[d], Val(D))..., 1, b] = work[c..., 2, b]
end
```

The same pass over the stored extent, which reaches the ghosts and does
not shift:

```julia
@kernel function recover!(prim, @Const(cons), ::Val{D}) where {D}
    I = @index(Global, NTuple)                 # already a stored index
    b = I[D + 1]
    c = ntuple(d -> I[d], Val(D))
    prim[c..., 1, b] = sqrt(cons[c..., 1, b])
end
```
"""
function map_blocks!(kernel!, fs::FieldSet{T,D}, args...;
                     stored::Bool=false, workgroupsize=nothing, kwargs...) where {T,D}
    check_removed_appendix(kwargs, "map_blocks!")
    backend = get_backend(fs.work)
    extent = stored ? ntuple(d -> size(fs.work, d), D) : ownextents(fs)
    launch_by_owner!(kernel!, backend, args...; ndrange=(extent..., nblocks(fs)),
                     workgroupsize=workgroupsize)
    synchronize(backend)
    return nothing
end

# --- The primary mask (M22) ----------------------------------------------------

"""
    PrimaryMask

Which own points of a field set are **primaries**: of the copies of a
point that several blocks hold, on a face plane of a vertex-like
dimension, the one on the block first in the forest's leaf order (M22,
"Closed blocks" in `CODE.md`). One bit per piece of a block's closed box —
along each vertex-like dimension its low face plane, its open interior or
its high face plane, `3^D` pieces — since every point of a piece lies on
the same block faces, and so has its copies on the same blocks. Every
point of a cell-centered or element set is a primary.

Built by [`primary_mask`](@ref) and asked by [`isprimary`](@ref); it is
`isbits` once on its backend, so a kernel takes it as an argument.
"""
struct PrimaryMask{D,V<:AbstractVector{UInt32}}
    pieces::V                      # per local block, bit `Σ cls_d 3^(d-1)`
    N::Int
    c::NTuple{D,Int8}
end

Adapt.adapt_structure(to, m::PrimaryMask) =
    PrimaryMask(Adapt.adapt(to, m.pieces), m.N, m.c)

"""
    isprimary(mask::PrimaryMask, i1, …, iD, b) -> Bool

Whether own point `(i1, …, iD)` of local block `b` is the primary copy of
its point (see [`PrimaryMask`](@ref)): the indices are those
[`map_blocks!`](@ref) hands a kernel, each in `1:N`, or `1:N+1` along a
vertex-like dimension, and those of [`statearray`](@ref). A right-hand side
that computes the primaries only returns early where this is `false`, and
must then call [`unify!`](@ref) on `du`, which gives every secondary its
primary's value; a reduction that takes each point once asks it through
its `primaries` keyword. It costs a load and a few comparisons, and works
in a device kernel.
"""
@inline function isprimary(m::PrimaryMask{D}, I::Vararg{Integer}) where {D}
    length(I) == D + 1 || throw(ArgumentError(
        "isprimary takes a point's $D own indices and its block, (i1, …, iD, b), " *
        "the global index of a map_blocks! kernel; got $(length(I)) numbers"))
    cls = ntuple(d -> m.c[d] == 0 ? 1 : I[d] == 1 ? 0 : I[d] == m.N + 1 ? 2 : 1, Val(D))
    piece = sum(ntuple(d -> cls[d] * 3^(d - 1), Val(D)))
    return (@inbounds(m.pieces[Int(I[D + 1])]) >> piece) & 0x1 == 0x1
end

"""
    primary_mask(fs::FieldSet) -> PrimaryMask

The [`PrimaryMask`](@ref) of `fs`'s blocks, on its backend: built from the
forest's topology and the set's centering once per change of the forest
and kept with the set. Over a distributed forest each rank's mask is its
own blocks', decided from the replicated forest by global leaf position,
so every rank agrees on every point.
"""
function primary_mask(fs::FieldSet{T,D}) where {T,D}
    backend = get_backend(fs.work)
    V = typeof(todevice(backend, UInt32[]))
    g = generation(fs.forest)
    cached = fs.primary
    cached !== nothing && cached[1] == g && return cached[2]::PrimaryMask{D,V}
    forest = fs.forest
    # The primaries across levels, mirrors, seams and glues come with the
    # steps that rebuild those (M22); refused until then, as the fill is.
    check_closed_blocks(forest, fs.centering, :half_open, "primary_mask")
    c = staggers(fs)
    range = blockrange(forest)
    pieces = Vector{UInt32}(undef, length(range))
    threaded_foreach(length(range)) do b
        pieces[b] = primary_bits(forest, first(range) + b - 1, c)
    end
    mask = PrimaryMask{D,V}(todevice(backend, pieces), forest.N, map(Int8, c))
    fs.primary = (g, mask)
    return mask
end

# The mask of a reduction that takes every own point, and the test of one.
struct NoMask end
@inline counted(::NoMask, I, b) = true
@inline counted(m::PrimaryMask, I, b) = isprimary(m, I..., b)

# Whether a reduction over `fs` takes the primaries only (M22): `primaries`
# is required for a set with a vertex-like dimension, where a sum or a
# count that silently took a shared point at every copy is the mistake to
# prevent, and means nothing for any other set, every point of which is a
# primary.
function count_primaries(fs::FieldSet, primaries, what::AbstractString)
    any(==(:vertex), fs.centering) || return false
    primaries === nothing && throw(ArgumentError(
        "$what over a set with a vertex-like dimension (centering $(fs.centering)) " *
        "needs `primaries = true` or `false`: such a set's blocks own both face planes " *
        "(M22, \"Closed blocks\" in CODE.md), so a point on a block face has a copy on " *
        "every block that holds it. `true` takes each point once, at its primary — " *
        "what a sum or a count wants; `false` takes every copy, which is right for a " *
        "maximum or a minimum."))
    return primaries::Bool
end

# The reduction masked to the primaries, as a fold: point by point, every
# selected variable at a point in turn. Behind a barrier, so that no set
# without a vertex-like dimension compiles it.
function primary_mapreduce(f, op, init, array, fs::FieldSet, g, vars::UnitRange{Int})
    rf(acc, p) = foldvalues(f, op, acc, p.u)
    return _block_fold(rf, op, init, array, fs, g, vars, nothing; mask=primary_mask(fs))
end
@inline foldvalues(f, op, acc, ::Tuple{}) = acc
@inline foldvalues(f, op, acc, u::Tuple) =
    foldvalues(f, op, op(acc, f(first(u))), Base.tail(u))

# Per-block reductions.
#
# Every diagnostic in the package has the same shape: one value per
# block, then a pass over those values on the host. Each block's value
# is bit-for-bit independent of the thread count (M5), because it is
# computed from that block's cells alone; a floating-point combination
# of the values is promised to roundoff only (`CODE.md`, "Parallelism",
# narrowed after M8), which is what lets the device path below be
# hierarchical and lets M7 `Allreduce`. Only how the per-block values
# are produced changes with the backend.
#
# On the CPU they are threaded host reductions over per-block views,
# which is what M5 measured and what the recorded numbers were taken
# with. On a device that same formulation would be one kernel launch and
# one device-to-host synchronization *per block*, issued from several
# host tasks at once; and the M6 form — one work item per block, each
# looping its own cells — was the weak row of the M6 table, 960 work
# items on a device that wants tens of thousands. So a device reduces
# in two launches: `REDUCE_LANES` lanes per block, each striding over
# the block's cells and variables in linear order (adjacent lanes read
# adjacent cells) into a private partial, then one work item per block
# folding its lanes in order. There is no barrier and no local memory,
# deliberately: KernelAbstractions realises a barrier on the CPU backend
# by splitting the kernel into separate loops over the workgroup, so a
# local does not survive a `@synchronize` unless it is `@private`, and
# the CPU backend is exactly where the suite cross-checks this path.
# Without a barrier the lanes of a block need not share a workgroup
# either, so none is imposed: a lane is a work item of an ordinary
# launch, and block and lane are read off the global index. (A static
# workgroup of 256 was the first form; Metal.jl launches one without
# checking the pipeline's own limit, so a register-heavy `f` could have
# failed to launch.) Every partial is a function of the block's cells
# and the stride, and the lane fold has a fixed order, so the device
# result is reproducible from run to run and exact for an
# order-independent `op`. `init` starts every lane, so it enters the
# fold once per lane rather than once: it must satisfy
# `op(init, init) == init`, which a neutral element does, and so does
# any `init` under an idempotent `op` such as `max`.
#
# The two paths share one specification of the reduction, `(f, op,
# init)`. They did not always: an earlier version passed a host
# block-reducer *and* a kernel-form fold, and nothing checked that the
# two agreed. `sum` over a block view is a sequential `mapfoldl` when
# the view is `IndexCartesian` but a *pairwise* one when it is
# `IndexLinear` — which `D = 1` with a scalar `vars` actually is. They
# agreed for every shape the package itself passed, by accident of
# `SubArray`'s `viewindexing` and nothing more (amended in M6).
#
# `array` is either the working array (offset `g = fs.G`, so ghosts are
# skipped) or the state array (`g` all zeros). Neither helper below works
# that out for itself; `block_mapreduce` does, which is why it and not
# these is what an application calls.

# Lanes per block in the device reductions. A power of two, so that
# block and lane come out of the global index as a shift and a mask; a
# block with fewer cells times variables than lanes leaves the surplus
# lanes at `init`.
const REDUCE_LANES = 256

# Launch one: lane `l` of block `b` folds entries `l, l + W, l + 2W, …`
# of the block's cells times selected variables, cells fastest, from
# `init`, into its own slot.
@kernel function block_partials_kernel!(partials, @Const(array), f, op, init,
                                        firstvar::Int, lastvar::Int,
                                        ::Val{D}, ::Val{G}, ::Val{E},
                                        ::Val{W}) where {D,G,E,W}
    g = @index(Global)
    b, l = divrem(g - 1, W) .+ 1
    cells = CartesianIndices(E)
    ncells = length(cells)
    acc = init
    for j in l:W:(ncells * (lastvar - firstvar + 1))
        q, r = divrem(j - 1, ncells)
        c = Tuple(cells[r + 1])
        acc = op(acc, f(array[ntuple(d -> c[d] + G[d], Val(D))..., firstvar + q, b]))
    end
    partials[l, b] = acc
end

# Launch two: one work item per block folds its lanes, in lane order,
# starting from the first lane's partial.
@kernel function fold_lanes_kernel!(values, @Const(partials), op, ::Val{W}) where {W}
    b = @index(Global)
    acc = partials[1, b]
    for l in 2:W
        acc = op(acc, partials[l, b])
    end
    values[b] = acc
end

# The device path: two launches, one synchronization, one copy back.
function _block_mapreduce_device(f, op, init::R, array, fs::FieldSet{T,D},
                                 backend::Backend, g::NTuple{D,Int},
                                 vars::UnitRange{Int}) where {R,T,D}
    n = nblocks(fs)
    n == 0 && return R[]
    W = REDUCE_LANES
    partials = allocate(backend, R, (W, n))
    values = allocate(backend, R, (n,))
    block_partials_kernel!(backend)(partials, array, f, op, init,
                                    first(vars), last(vars),
                                    Val(D), Val(g), Val(ownextents(fs)), Val(W);
                                    ndrange=W * n)
    fold_lanes_kernel!(backend)(values, partials, op, Val(W); ndrange=n)
    synchronize(backend)
    return tohost(values)
end

# The host path. It is split out under its own name rather than
# dispatched on `::CPU` so that both paths stay reachable on a machine
# with no device, which is what lets the suite check that the two
# compute the same fold.
function _block_mapreduce_host(f, op, init::R, array, fs::FieldSet{T,D},
                               g::NTuple{D,Int},
                               vars::UnitRange{Int}) where {R,T,D}
    E = ownextents(fs)
    inner = ntuple(d -> (g[d] + 1):(g[d] + E[d]), D)
    values = Vector{R}(undef, nblocks(fs))
    threaded_foreach(nblocks(fs)) do b
        values[b] = mapreduce(f, op, view(array, inner..., vars, b); init=init)
    end
    return values
end

# `vars` ends up as a pair of kernel arguments, so it has to name a
# contiguous run of variables: there is no way to hand a device an
# arbitrary index vector cell by cell, and silently reducing
# `first:last` instead would be worse than refusing.
function _varrange(vars, nvars::Integer)
    r = if vars isa Integer
        Int(vars):Int(vars)
    elseif vars isa AbstractUnitRange{<:Integer}
        Int(first(vars)):Int(last(vars))
    else
        throw(ArgumentError("vars must be an integer or a contiguous range of " *
                            "variable indices, got a $(typeof(vars)): the " *
                            "selection becomes a kernel argument, and a kernel " *
                            "cannot be handed an arbitrary index vector"))
    end
    isempty(r) || 1 <= first(r) <= last(r) <= nvars ||
        throw(ArgumentError("vars = $vars is out of range for a field set with " *
                            "$nvars variable(s); valid indices are 1:$nvars"))
    return r
end

"""
    block_mapreduce(f, op, init, fs::FieldSet; vars=1:fs.nvars)
    block_mapreduce(f, op, init, fs::FieldSet, u::AbstractVector; vars=1:fs.nvars)

Reduce each block's interior cells to one value: `f` transforms a cell
value and `op` folds the transformed values, starting from `init`. The
result is a host `Vector` of length `nblocks(fs)`, indexed by block,
with element type taken from `init`. `init` must satisfy
`op(init, init) == init` — a neutral element does, as Base's `reduce`
requires of its `init`, and so does any `init` under an idempotent `op`
such as `max` — because on a device the fold starts from it once per
lane, not once per block.

The first form reads the working array's interiors, skipping the ghosts;
the second reads a state vector, which has no ghosts to skip. `vars` is
an integer or a contiguous range of variable indices.

This is the read-side counterpart of [`map_blocks!`](@ref), and it is
the shape every diagnostic here has. Combining the values is left to the
caller, because a refinement criterion wants them per block; for one
number over the whole mesh, weighted or not, use
[`mesh_mapreduce`](@ref). Each block's value is bit-identical whatever
the thread count, since it is computed from that block's cells alone; a
floating-point combination of them is promised to roundoff only,
whichever way it is written.

The values are per **local** block (M7): over a distributed forest they
are this rank's blocks only, so a number combined from them — the
largest value of a variable, which a refinement criterion needs for its
scale, or a signal speed for a time step — is this rank's number, not
the mesh's, and would differ between ranks. Such a number is a
[`mesh_mapreduce`](@ref), the same on every rank:

```julia
scales = [mesh_mapreduce(abs, max, zero(eltype(fs.work)), fs; vars=v)
          for v in 1:fs.nvars]
```

It runs wherever the data lives: threaded host reductions over per-block
views on the CPU; on a device, one workgroup per block in two launches,
lanes striding over the block's cells and then one work item per block
folding the lanes. It synchronizes and copies back to the host, so it
belongs between steps and not inside a right-hand side.

The fold is one specification on both backends, but its *association* is
not: the host's `mapreduce` may reassociate `op`, and the lanes of the
device fold split the cells differently again. The guarantee is
per-block values that are bit-identical across thread counts and, for a
floating-point `op`, agree to roundoff across backends — identical
across backends is not claimed, and is not true. A device value is
reproducible from run to run, since the lane fold has a fixed order.

!!! note "Callbacks on a device"
    `f` and `op` become kernel arguments, so everything they close over
    must be `isbits`. A captured `Type` is the usual trip: write
    `oftype(x, 2)` rather than closing over `T` and calling `T(2)`. The
    same rule covers captured arrays and any mutable state.
"""
function block_mapreduce(f, op, init, fs::FieldSet{T,D}; vars=1:fs.nvars,
                         primaries=nothing) where {T,D}
    once = count_primaries(fs, primaries, "block_mapreduce")
    return _block_mapreduce(f, op, init, fs.work, fs, fs.G, _varrange(vars, fs.nvars),
                            once)
end

function block_mapreduce(f, op, init, fs::FieldSet{T,D}, u::AbstractVector;
                         vars=1:fs.nvars, primaries=nothing) where {T,D}
    once = count_primaries(fs, primaries, "block_mapreduce")
    return _block_mapreduce(f, op, init, statearray(u, fs), fs, ntuple(_ -> 0, D),
                            _varrange(vars, fs.nvars), once)
end

function _block_mapreduce(f, op, init::R, array, fs::FieldSet{T,D}, g::NTuple{D,Int},
                          vars::UnitRange{Int}, once::Bool=false) where {R,T,D}
    once && return Base.inferencebarrier(primary_mapreduce)(f, op, init, array, fs, g,
                                                            vars)::Vector{R}
    backend = get_backend(array)
    return backend isa CPU ?
           _block_mapreduce_host(f, op, init, array, fs, g, vars) :
           _block_mapreduce_device(f, op, init, array, fs, backend, g, vars)
end

"""
    mesh_mapreduce(f, op, init, fs::FieldSet; vars=1:fs.nvars, weight=nothing)
    mesh_mapreduce(f, op, init, fs::FieldSet, u::AbstractVector;
                   vars=1:fs.nvars, weight=nothing)

Reduce the whole mesh to one number: the per-block values of
[`block_mapreduce`](@ref), each multiplied by `weight(key)` when a
weight is given, combined with `op`. The two forms and `vars` are those
of `block_mapreduce`; `init` is returned for a mesh with no blocks and
must be a neutral element for `op`.

`weight` is a function of a block's [`MortonKey`](@ref), evaluated on the
host and converted to the type of `init` before it multiplies each
block's value — so `init` has to be a floating-point value when a
weight is given. It scales a block's contribution by its geometry: a
cell volume, `spacing(T, forest, key)^D`, is what
[`volume_weighted_norm`](@ref) and [`total_mass`](@ref) pass to `+`,
and a positive weight is equally meaningful under `max` or `min`, where
`max |v| / h` is a CFL rate.

This is the form a conserved total, a norm, or a CFL speed wants, and
the one place a reduction crosses blocks: the per-block values are
combined on the host, and under MPI (M7) across ranks as well, so an
application never sees a communicator. The floating-point result is
promised to roundoff across thread counts, rank counts and backends,
not bit for bit; see [`block_mapreduce`](@ref) for what *is* exact.

The peak of a variable over the mesh, and the mass of another:

```julia
peak = mesh_mapreduce(abs, max, zero(T), fs; vars=1)
mass = mesh_mapreduce(identity, +, zero(T), fs; vars=2,
                      weight=key -> spacing(T, fs.forest, key)^D)
```
"""
function mesh_mapreduce(f, op, init, fs::FieldSet; vars=1:fs.nvars, weight=nothing,
                        primaries=nothing)
    values = block_mapreduce(f, op, init, fs; vars=vars, primaries=primaries)
    return combine_blocks(op, init, fs, values, weight)
end

function mesh_mapreduce(f, op, init, fs::FieldSet, u::AbstractVector; vars=1:fs.nvars,
                        weight=nothing, primaries=nothing)
    values = block_mapreduce(f, op, init, fs, u; vars=vars, primaries=primaries)
    return combine_blocks(op, init, fs, values, weight)
end

# The host stage. Two details keep every recorded norm and mass exactly
# where the M3 code left it. The weight multiplies in place, in the
# order `total_mass` always used. And the combination is
# `mapreduce(identity, op, values)` *without* `init`: `sum(v)` and
# `mapreduce(identity, +, v)` are the same pairwise reduction bit for
# bit, whereas `reduce(+, v; init)` is a sequential left fold — an
# explicit `init` changes Base's association (measured, 2026-09-22).
#
# The values are this rank's blocks, and the one cross-rank step of any
# reduction is here and nowhere else (M7): every rank gathers each
# rank's partial and folds them in rank order, so that the association
# is the package's and the result the same on every rank. A rank with
# no blocks contributes no partial rather than `init`, which need only
# satisfy `op(init, init) == init` (`max` from 0 over negative data);
# `init` comes back only when every rank is empty. At one rank the
# partial is the result, exactly the serial value, and nothing is
# gathered. See "Reductions" under "Distributed meshes" in CODE.md.
function combine_blocks(op, init::R, fs::FieldSet, values::Vector{R}, weight) where {R}
    if weight !== nothing
        for b in eachindex(values)
            values[b] *= oftype(init, weight(blockkey(fs, b)))
        end
    end
    comm = fs.forest.comm
    if commsize(comm) == 1
        isempty(values) && return init
        return mapreduce(identity, op, values)
    end
    partial = isempty(values) ? (false, init) : (true, mapreduce(identity, op, values))
    # Asserted, since `comm` is abstractly typed: with more than a few
    # `Communicator` subtypes loaded, inference gives up on the call and
    # the reduction's result would be inferred as `Any` (found in step 8,
    # when the suite's fourth test communicator made it so).
    partials = allgather(comm, partial)::Vector{typeof(partial)}
    acc, found = init, false
    for (hasvalue, value) in partials
        hasvalue || continue
        acc = found ? op(acc, value) : value
        found = true
    end
    return acc
end

# Folds over points (M15).
#
# `block_mapreduce` hands its `f` one value at a time and nothing else, so
# one call yields one number, and a region smaller than a block has to be
# written into the data by a kernel of its own before it can be reduced.
# A fold hands its `rf` a whole point instead — every selected variable,
# the position, the spacing, the level and a per-block tag — and carries
# any `isbits` accumulator, so one sweep and one collective yield every
# norm of every region at once. The case that asked for it is
# TreeGeneralizedHarmonic's analysis row, which re-ran its constraint
# monitors once per spherical shell only to select the shell's points
# (`CODE.md`, "Folds over points").
#
# One kernel serves both paths. Lane `l` of block `b` folds the block's
# points `l, l + W, l + 2W, …` in linear order into its own slot, as the
# device reduction above does; the CPU runs it with `W = 1`, so that each
# block is one sequential fold in index order, on its owner's thread, and
# its value is bit-identical whatever the thread count. A device runs as
# many lanes as the accumulator's size allows, then folds them in lane
# order with `fold_lanes_kernel!`.

"""
    Sum(x)

An accumulator field that [`TreeAMR.combine`](@ref) combines with `+`.
`Sum`, [`Max`](@ref) and [`Min`](@ref) let a fold's accumulator say how
each of its fields combines, so that the caller writes only the per-point
`rf` of [`mesh_fold`](@ref) and passes `TreeAMR.combine` as its `op`:

```julia
acc = (n = Sum(0), sq = Sum(0.0), peak = Max(0.0))
```

The two sides of a `Sum` must have the same type for the accumulator to
keep its type: a `Sum(0)` and a `Sum(1.0)` do not combine to a `Sum{Int}`.
"""
struct Sum{T}
    value::T
end

"""
    Max(x)

An accumulator field that [`TreeAMR.combine`](@ref) combines with `max`.
`NaN` propagates, as it does through `max`, so a blown-up run shows up
in the result instead of being skipped. See [`Sum`](@ref).
"""
struct Max{T}
    value::T
end

"""
    Min(x)

An accumulator field that [`TreeAMR.combine`](@ref) combines with `min`,
propagating `NaN` as `min` does. See [`Sum`](@ref).
"""
struct Min{T}
    value::T
end

"""
    TreeAMR.combine(a, b)

Combine two accumulators built from [`Sum`](@ref), [`Max`](@ref) and
[`Min`](@ref), recursing through tuples and `NamedTuple`s: each leaf is
combined by its own wrapper. This is the `op` a fold built from those
wrappers passes to [`mesh_fold`](@ref) and [`block_fold`](@ref).

Accumulators of different shapes do not combine: a `Sum` against a `Max`,
tuples of different lengths or `NamedTuple`s with different names are a
`MethodError`, never a silently wrong number. It is not called `merge`
because `Base.merge` on two `NamedTuple`s does something else entirely,
and quietly.
"""
combine(a::Sum, b::Sum) = Sum(a.value + b.value)
combine(a::Max, b::Max) = Max(max(a.value, b.value))
combine(a::Min, b::Min) = Min(min(a.value, b.value))
combine(a::NTuple{N,Any}, b::NTuple{N,Any}) where {N} = map(combine, a, b)
combine(a::NamedTuple{K}, b::NamedTuple{K}) where {K} =
    NamedTuple{K}(map(combine, Tuple(a), Tuple(b)))

"""
    addto(bins::NTuple{K}, k, x)

`bins` with `x` combined into slot `k` by [`TreeAMR.combine`](@ref), and
every other slot unchanged. A `k` outside `1:K` changes nothing, so a
fold can use `0` for a point that falls in no bin.

This is how a fold bins its points, by shell or by level, say. The tuple
is rebuilt with every index a literal rather than indexed at `k`: a
run-time index into a tuple makes a device spill it to local memory,
while this keeps the bins in registers for the cost of `K` selects.
"""
@inline addto(bins::NTuple{K,Any}, k::Integer, x) where {K} =
    ntuple(i -> i == k ? combine(bins[i], x) : bins[i], Val(K))

"""
    TreeAMR.FoldPoint

What a fold's `rf` is handed for each owned point (see
[`mesh_fold`](@ref)). It is built inside the kernel and is `isbits`:

- `p.u`: the selected variables at this point, an `NTuple` in the field
  set's element type, `p.u[1]` being variable `first(vars)`;
- `p.x`: the point's position, an `NTuple{D}`, bit for bit what
  [`coordinates`](@ref) returns, so a cell center, a vertex or a face
  center as the field set's centering says;
- `p.h`: the block's spacing, so `p.h^D` is the point's volume weight;
- `p.level`: the block's refinement level;
- `p.tag`: the block's value of the fold's `tag` function, or `nothing`;
- `p.I`: the own index, each entry in `1:N`, or `1:N+1` along a
  vertex-like dimension (add `G[d]` for the stored index, as in
  [`map_blocks!`](@ref));
- `p.b`: the local block index.
"""
struct FoldPoint{D,NV,T,TG}
    u::NTuple{NV,T}
    x::NTuple{D,T}
    h::T
    level::Int
    tag::TG
    I::NTuple{D,Int}
    b::Int
end

# The tags of a fold without a `tag` function: indexing gives `nothing`,
# so `p.tag` is `nothing` without an array to upload.
struct NoTags end
@inline Base.getindex(::NoTags, ::Integer) = nothing
tagtype(::NoTags) = Nothing
tagtype(tags::AbstractArray) = eltype(tags)

# Bytes of lane partials per block a device fold may use: the lane count
# starts at `REDUCE_LANES` and halves until `W · sizeof(acc)` fits. With
# the 100-odd numbers of TreeGH's accumulators that is 32 or 64 lanes, and
# at its 856 blocks the buffer stays under 56 MB whatever the accumulator.
# A first guess, not a measurement (`CODE.md`, "Folds over points").
const FOLD_LANE_BYTES = 64 * 1024

function fold_lanes(::Type{R}) where {R}
    W = REDUCE_LANES
    while W > 1 && W * sizeof(R) > FOLD_LANE_BYTES
        W ÷= 2
    end
    return W
end

# Lane `l` of block `b` folds points `l, l + W, …` of the block, in
# linear order with the first dimension fastest, into its own slot. The
# position is formed by `coordinates_kernel!`'s expression, on the same
# corner and spacing in the same order, so it reproduces `coordinates`
# bit for bit.
@kernel function fold_points_kernel!(partials, @Const(array), rf, init,
                                     @Const(lows), @Const(bases), @Const(spacings), nodes,
                                     @Const(levels),
                                     tags, mask, firstvar::Int,
                                     ::Val{D}, ::Val{G}, ::Val{C}, ::Val{E}, ::Val{NV},
                                     ::Val{W}) where {D,G,C,E,NV,W}
    g = @index(Global)
    b, l = divrem(g - 1, W) .+ 1
    lo, B, h = lows[b], bases[b], spacings[b]
    lev, tag = levels[b], tags[b]
    cells = CartesianIndices(E)
    acc = init
    for j in l:W:length(cells)
        I = Tuple(cells[j])
        counted(mask, I, b) || continue
        x = ntuple(d -> nodeposition(lo[d], h, B[d], I[d], C[d], nodes), Val(D))
        c = ntuple(d -> I[d] + G[d], Val(D))
        u = ntuple(v -> array[c..., firstvar + v - 1, b], Val(NV))
        acc = rf(acc, FoldPoint(u, x, h, lev, tag, I, b))
    end
    partials[l, b] = acc
end

# The per-block tags, or `NoTags()`. The tag type is taken from inference
# rather than from the values, so that it is the same on every rank — a
# refusal that fired on some ranks only would leave the others waiting in
# the allgather — and a rank with no blocks gets the same type as the
# rest.
fold_tags(::Backend, ::Nothing, forest::Forest) = NoTags()
function fold_tags(backend::Backend, tag, forest::Forest{D}) where {D}
    TG = Base.promote_op(tag, MortonKey{D})
    isconcretetype(TG) && isbitstype(TG) || throw(ArgumentError(
        "a fold's tag function must return one isbits type for every block, " *
        "inferred as $TG: the tags are uploaded as one array, which needs a " *
        "concrete element type, and a kernel argument must be plain bits. " *
        "Return a Bool or a number, and the same type on every path"))
    return todevice(backend, TG[tag(k) for k in localleaves(forest)])
end

# The checks a fold can make before launching, on the types alone so that
# every rank makes the same decision.
function check_fold(rf, ::Type{R}, ::Type{P}) where {R,P}
    isbitstype(R) || throw(ArgumentError(
        "a fold's accumulator must be an isbits value — a number, a tuple, a " *
        "NamedTuple or a plain struct of them — but init is a $R: a device " *
        "holds one per lane, and every rank's partial is allgathered, which " *
        "sends plain bits only"))
    S = Base.promote_op(rf, R, P)
    isconcretetype(S) && S !== R && throw(ArgumentError(
        "a fold's rf returns a $S where the accumulator is a $R, the type of " *
        "init: rf(acc, p) must return the accumulator's own type, since a " *
        "lane holds it in place from point to point. A Sum(1) beside an init " *
        "holding Sum(0.0) is the usual cause"))
    return nothing
end

# The fold over `array` — the working array at offset `g = fs.G`, or the
# state array at `g = 0` — with `W` lanes per block: 1 on the CPU, and
# `fold_lanes(R)` on a device unless the caller says otherwise, which the
# tests do to reach the device path on `CPU()`.
#
# `mask` is `NoMask()` for every own point, or the set's `PrimaryMask` for
# the primaries only (M22).
function _block_fold(rf, op, init::R, array, fs::FieldSet{T,D}, g::NTuple{D,Int},
                     vars::UnitRange{Int}, tag, lanes=nothing;
                     mask=NoMask()) where {R,T,D}
    forest = fs.forest
    backend = get_backend(array)
    tags = fold_tags(backend, tag, forest)
    NV = length(vars)
    check_fold(rf, R, FoldPoint{D,NV,T,tagtype(tags)})
    n = nblocks(fs)
    n == 0 && return R[]
    W = lanes !== nothing ? Int(lanes) : backend isa CPU ? 1 : fold_lanes(R)
    lows = todevice(backend, block_lows(forest, T))
    bases = todevice(backend, block_bases(forest))
    spacings = todevice(backend, block_spacings(forest, T))
    levels = todevice(backend, Int[level(k) for k in localleaves(forest)])
    partials = allocate(backend, R, (W, n))
    launch_fold!(backend, partials, array, rf, init, lows, bases, spacings, levels, tags,
                 mask, fs, first(vars), g, Val(NV), Val(W))
    if W == 1
        synchronize(backend)
        return tohost(vec(partials))
    end
    values = allocate(backend, R, (n,))
    fold_lanes_kernel!(backend)(values, partials, op, Val(W); ndrange=n)
    synchronize(backend)
    return tohost(values)
end

function launch_fold!(backend, partials, array, rf, init, lows, bases, spacings,
                      levels, tags, mask, fs::FieldSet{T,D}, firstvar::Int,
                      g::NTuple{D,Int},
                      nv::Val, w::Val{W}) where {T,D,W}
    with_nodes(backend, T, fs.P) do nodes
        launch_by_owner!(fold_points_kernel!, backend, partials, array, rf, init, lows,
                         bases, spacings, nodes, levels, tags, mask, firstvar, Val(D),
                         Val(g), Val(staggers(fs)), Val(ownextents(fs)), nv, w;
                         ndrange=(W * nblocks(fs),))
    end
    return nothing
end

"""
    block_fold(rf, op, init, fs::FieldSet; vars=1:fs.nvars, tag=nothing)
    block_fold(rf, op, init, fs::FieldSet, u::AbstractVector; vars=1:fs.nvars, tag=nothing)

Fold each block's owned points into one accumulator: a host `Vector` of
length `nblocks(fs)`, indexed by local block, of the type of `init`. This
is the per-block half of [`mesh_fold`](@ref), which says what `rf`, `op`,
`init`, `vars` and `tag` are; the values are this rank's blocks only, as
for [`block_mapreduce`](@ref).
"""
function block_fold(rf, op, init, fs::FieldSet{T,D}; vars=1:fs.nvars,
                    tag=nothing, primaries=nothing) where {T,D}
    once = count_primaries(fs, primaries, "block_fold")
    return fold_masked(once, rf, op, init, fs.work, fs, fs.G, _varrange(vars, fs.nvars),
                       tag)
end

function block_fold(rf, op, init, fs::FieldSet{T,D}, u::AbstractVector; vars=1:fs.nvars,
                    tag=nothing, primaries=nothing) where {T,D}
    once = count_primaries(fs, primaries, "block_fold")
    return fold_masked(once, rf, op, init, statearray(u, fs), fs, ntuple(_ -> 0, D),
                       _varrange(vars, fs.nvars), tag)
end

# The fold over every own point, or behind a barrier over the primaries,
# so that no set without a vertex-like dimension compiles the masked one.
fold_masked(once::Bool, rf, op, init::R, args...) where {R} =
    once ? Base.inferencebarrier(fold_primaries)(rf, op, init, args...)::Vector{R} :
    _block_fold(rf, op, init, args...)
fold_primaries(rf, op, init, array, fs, g, vars, tag) =
    _block_fold(rf, op, init, array, fs, g, vars, tag; mask=primary_mask(fs))

"""
    mesh_fold(rf, op, init, fs::FieldSet; vars=1:fs.nvars, tag=nothing)
    mesh_fold(rf, op, init, fs::FieldSet, u::AbstractVector; vars=1:fs.nvars, tag=nothing)

Fold every owned point of the mesh into one accumulator, which can hold
as many results as the caller likes: the norms of several variables, of
several regions, of several weightings, in one sweep and one collective.

- `rf(acc, p) -> acc` folds one point into the accumulator. `p` is a
  [`TreeAMR.FoldPoint`](@ref): the selected variables at the point
  (`p.u`), its position (`p.x`), the block's spacing, level and tag, and
  its indices.
- `op(a, b) -> acc` combines two accumulators — of two lanes, two blocks
  or two ranks. For an accumulator built from [`Sum`](@ref),
  [`Max`](@ref) and [`Min`](@ref), it is [`TreeAMR.combine`](@ref).
- `init` is the empty accumulator, and its type is the result's. It must
  be `isbits`: a number, a tuple, a `NamedTuple` or a plain struct of
  them.
- `vars` is an integer or a contiguous range of variable indices, as for
  [`block_mapreduce`](@ref); it selects what `p.u` holds, not which
  points are visited.
- `tag = key -> t` is an optional host function of a block's
  [`MortonKey`](@ref), whose value reaches `rf` as `p.tag`: a block
  region such as "touches an outer face" is a `Bool` tag. It must return
  one `isbits` type for every block.

The first form reads the working array, skipping the ghosts; the second
reads a state vector. Either way the points are exactly the owned points
[`map_blocks!`](@ref) visits, `N` per dimension, so a vertex-centered
block's shared upper plane is not counted and every physical point is
counted once.

The laws are those that let a block be split across a device's lanes:
`op` is associative, to roundoff at least; `op(init, init) == init`; and
`rf(a, p) == op(a, rf(init, p))`. Every sum, maximum, minimum and
[`addto`](@ref) bin update obeys them. A weight is per point: `rf`
multiplies by `p.h^D` itself where a volume weight is wanted, unlike the
per-block `weight` of [`mesh_mapreduce`](@ref), so the same norm through
the two differs in its last bits.

Determinism follows the package's rules ("Parallelism" in `CODE.md`). On
the CPU each block is one sequential fold in index order on its owner's
thread, so a block's accumulator is bit-identical at every thread count,
and the blocks combine in block order. A device folds each block over
lanes and then the lanes in order, reproducible from run to run and equal
to the CPU's to roundoff. Across ranks the per-rank partials are
allgathered and folded in rank order, so maxima, minima and integer
counts are bit-identical at any rank count and sums agree to roundoff.

The point count, the sum of squares and the peak of variable 1 inside the
shell `1 ≤ |x| < 2`, and the sum of squares outside it, in one sweep:

```julia
init = (n = Sum(0), sq = (Sum(0.0), Sum(0.0)), peak = Max(0.0))
function rf(acc, p)
    inside = 1 ≤ sqrt(sum(abs2, p.x)) < 2
    q = p.u[1]
    return TreeAMR.combine(acc, (n = Sum(Int(inside)),
                                 sq = addto((Sum(0.0), Sum(0.0)), inside ? 1 : 2,
                                            Sum(q * q)),
                                 peak = Max(inside ? abs(q) : 0.0)))
end
r = mesh_fold(rf, TreeAMR.combine, init, fs; vars=1)
l2 = sqrt(r.sq[1].value / r.n.value)
```

!!! note "Callbacks on a device"
    `rf`, `op` and `init` become kernel arguments, so everything they
    hold or close over must be `isbits`: shell radii go in as a tuple,
    not a `Vector`, and a captured `Type` is the usual trip — write
    `oftype(x, 2)` rather than closing over `T` and calling `T(2)`. The
    accumulator lives in each lane's registers; a large one spills,
    which costs time but not correctness.
"""
function mesh_fold(rf, op, init, fs::FieldSet; vars=1:fs.nvars, tag=nothing,
                   primaries=nothing)
    values = block_fold(rf, op, init, fs; vars=vars, tag=tag, primaries=primaries)
    return combine_blocks(op, init, fs, values, nothing)
end

function mesh_fold(rf, op, init, fs::FieldSet, u::AbstractVector; vars=1:fs.nvars,
                   tag=nothing, primaries=nothing)
    values = block_fold(rf, op, init, fs, u; vars=vars, tag=tag, primaries=primaries)
    return combine_blocks(op, init, fs, values, nothing)
end

"""
    volume_weighted_norm(fs::FieldSet, u::AbstractVector; p=2)

The `p`-norm of a state vector with each cell weighted by its volume,
normalized by the domain volume — so the result is a resolution
independent RMS (or, for `p = Inf`, the plain maximum).

Weighting matters on an adaptive mesh: refined regions contribute more
*entries* per unit volume simply for being refined, so an unweighted
norm silently emphasizes them. An element-centered set (M16;
[`elementcentered`](@ref)) weights each node by its Gauss–Lobatto
quadrature weight instead, the tensor product of `(P+1)ŵᵢ/2` over the
dimensions times the block's `hᴰ`, so the norm is the discrete one its
summation-by-parts property holds in, exact for `|u|^p` of degree up to
`2P - 1` in each dimension. This is also the shape an adaptive
integrator's `internalnorm` needs; through M3 only fixed-`dt`
integrators are exercised, so it is used here for error measurement.

A [`mesh_mapreduce`](@ref) with the cell volume as the weight. The value
is reproducible to roundoff across thread counts and backends; on the
CPU, where the partials are combined in block order, it is exact across
thread counts as well.
"""
function volume_weighted_norm(fs::FieldSet{T,D}, u::AbstractVector; p::Real=2) where {T,D}
    R = float(real(T))
    isinf(p) && return mesh_mapreduce(abs, max, zero(R), fs, u; primaries=false)
    total, volume = norm_parts(fs, u, p)
    volume == 0 && return zero(R)
    # `inv(R(p))`, not `1 / p`: the latter is a Float64 exponent, which
    # promotes the whole result to Float64 and made this function return a
    # different type from the `isinf(p)` branch above.
    return (total / volume)^inv(R(p))
end

"""
    volume_weighted_norm(sets::Tuple{Vararg{FieldSet}}, u::AbstractVector; p=2)

The same norm over the state vector of several field sets (see
[`statevector`](@ref)): every cell of every set weighted by its volume,
normalized by the sum of the sets' domain volumes, each counted once per
variable as for one set. For `p = Inf` it is the maximum over all
segments. With one set it equals the single-set norm exactly. This is
the `internalnorm` an adaptive integrator wants for such a vector.
"""
function volume_weighted_norm(sets::Tuple{Vararg{FieldSet}}, u::AbstractVector;
                              p::Real=2)
    check_statelength(u, sets)
    R = float(real(eltype(first(sets).work)))
    if isinf(p)
        return mapreduce(i -> volume_weighted_norm(sets[i], stateview(u, sets, i); p=p),
                         max, eachindex(sets))
    end
    total, volume = zero(R), zero(R)
    for i in eachindex(sets)
        t, v = norm_parts(sets[i], stateview(u, sets, i), p)
        total += t
        volume += v
    end
    volume == 0 && return zero(R)
    return (total / volume)^inv(R(p))
end

# The quadrature of an element-centered set (M16), per block: `Σ W f(u)`
# over the owned points of `array` (the working array at offset `g =
# fs.G`, or the state array at `g = 0`) and the variables `vars`, `W` the
# tensor product of the nodes' GLL weights in units of the spacing,
# `(P+1)ŵᵢ/2` per dimension. The block's `hᴰ` is left to the caller's
# `combine_blocks` weight, as for a cell-centered set. A fold whose `rf`
# carries the weights, a small array on the backend, as a closure: a
# kernel argument is adapted to the device, closures included.
function element_quadrature(f, ::Type{R}, array, fs::FieldSet{T,D}, g::NTuple{D,Int},
                            vars::UnitRange{Int}) where {R,T,D}
    w = todevice(get_backend(array), element_tables(T, fs.P).quadrature)
    function rf(acc, p)
        P1 = length(w)
        weight = prod(ntuple(d -> @inbounds(w[(p.I[d] - 1) % P1 + 1]), Val(D)))
        return acc + weight * sum(f, p.u)
    end
    return _block_fold(rf, +, zero(R), array, fs, g, vars, nothing)
end

# The trapezoid's quadrature of a set with a vertex-like dimension (M22),
# per block: `Σ w f(u)` over the own points of `array` and the variables
# `vars`, `w` the product over the vertex-like dimensions of ½ on a face
# plane and 1 between. Summed over the blocks, a shared point then has its
# weight in the composite rule — two halves on a plane between blocks, a
# half at a wall — so a constant is its own norm with no mask. The block's
# `hᴰ` is left to `combine_blocks`, as for a cell-centered set.
function trapezoid_quadrature(f, ::Type{R}, array, fs::FieldSet{T,D}, g::NTuple{D,Int},
                              vars::UnitRange{Int}) where {R,T,D}
    c, N = staggers(fs), fs.forest.N
    function rf(acc, p)
        half = one(acc) / 2
        w = prod(ntuple(d -> c[d] == 1 && (p.I[d] == 1 || p.I[d] == N + 1) ? half :
                             one(acc), Val(D)))
        return acc + w * sum(f, p.u)
    end
    return _block_fold(rf, +, zero(R), array, fs, g, vars, nothing)
end
trapezoid_quadrature_barrier(f, ::Type{R}, args...) where {R} =
    Base.inferencebarrier(trapezoid_quadrature)(f, R, args...)::Vector{R}

# `element_quadrature` behind `Base.inferencebarrier`, for the callers
# that branch on `fs.P` at run time: inferred, the element branch would
# compile a second fold kernel for every cell or vertex set's norm and
# mass (see `with_nodes`).
element_quadrature_barrier(f, ::Type{R}, args...) where {R} =
    Base.inferencebarrier(element_quadrature)(f, R, args...)::Vector{R}

# The two sums of a finite `p`-norm over one field set: `Σ hᴰ |u|^p` over
# its cells, or for an element set (M16) the GLL quadrature of `|u|^p`,
# and its domain volume times `nvars`.
function norm_parts(fs::FieldSet{T,D}, u::AbstractVector, p::Real) where {T,D}
    forest = fs.forest
    R = float(real(T))

    # An integer exponent stays an integer: `abs(x)^2` is a squaring,
    # while `abs(x)^2.0` would drag a `Float64` operand into the
    # innermost loop — fatal on a device with no hardware fp64, and the
    # exact leak the type-genericity work went after.
    q = p isa Integer ? Int(p) : R(p)
    # `float(real(T))` inline rather than the local `R`: a closure over a
    # local holding a type stores it as a `DataType` on Julia 1.10 and
    # returns `Any`, which cost the norm a dynamic dispatch per block
    # (found in review, measured at 2x the host time on 1.10.12).
    cellvolume(key) = spacing(float(real(T)), forest, key)^D
    total = if any(==(:vertex), fs.centering)
        # The trapezoid's weights along a vertex-like dimension (M22).
        values = trapezoid_quadrature_barrier(x -> abs(x)^q, R, statearray(u, fs), fs,
                                              ntuple(_ -> 0, D), 1:(fs.nvars))
        combine_blocks(+, zero(R), fs, values, cellvolume)
    elseif fs.P == 0
        mesh_mapreduce(x -> abs(x)^q, +, zero(R), fs, u; weight=cellvolume)
    else
        # The nodes are not equally spaced, so each gets its quadrature
        # weight: the norm the summation-by-parts property holds in.
        values = element_quadrature_barrier(x -> abs(x)^q, R, statearray(u, fs), fs,
                                            ntuple(_ -> 0, D), 1:(fs.nvars))
        combine_blocks(+, zero(R), fs, values, cellvolume)
    end

    # The domain volume as the sum of the block volumes, through the same
    # combination as everything else — so `sum`'s association from M3 is
    # kept, and M7 has one place to reduce across ranks. `N^D` per block
    # whatever the centering: the trapezoid's weights of a vertex-like
    # dimension sum to `N` too.
    volumes = Vector{R}(undef, nblocks(fs))
    for b in 1:nblocks(fs)
        volumes[b] = cellvolume(blockkey(fs, b)) * (forest.N^D * fs.nvars)
    end
    volume = combine_blocks(+, zero(R), fs, volumes, nothing)
    return (total, volume)
end
