# Regridding: flag -> complete -> rebuild -> transfer.
#
# The transfer reuses the M2 stencil machinery. A block that survives is
# copied, a newly refined block is prolongated from its parent, and a
# coarsened block is restricted from its children — which are exactly
# the `δ = 0` cases of the ghost-exchange stencils, since the target
# region is a block's own interior rather than a halo slab.

"""
    RegridFlag

What the application wants done with a block: `Refine`, `Coarsen`, or
`Keep` it as is.

Flags are requests, not commands. Refinement is completed outward to
maintain 2:1 balance, and coarsening happens only where all `2^D`
siblings ask for it *and* balance still permits — see
[`complete_marks`](@ref).
"""
@enum RegridFlag Coarsen Keep Refine

"""
    flag_blocks(f, forest) -> Vector

Build a flag vector by calling `f(b, key)` for every leaf, with `b` the
block index and `key` its [`MortonKey`](@ref).

`f` may return either a bare [`RegridFlag`](@ref) or a
`(flag, box)` pair, where `box::NTuple{D,UnitRange{Int}}` is the
bounding box of the cells that fired, in that block's own interior
indices `1:N` (see [`complete_marks`](@ref) for what the box is used
for). The two forms may be mixed within one vector.

A convenience for host-side flagging; an application is free to produce
the vector any other way, which is what will let the flagging kernel run
on the device in M6 while the completion logic stays on the host.
"""
flag_blocks(f, forest::Forest) = [f(b, k) for (b, k) in enumerate(forest.leaves)]

# Whatever a flagging function reported, reduced to its flag and to the
# box of firing cells. An omitted box means the whole interior — the
# conservative isotropic case.
markflag(m::RegridFlag) = m
markflag(m::Tuple{RegridFlag,Any}) = m[1]
markflag(m) = throw(ArgumentError(
    "a flag must be a RegridFlag or a (RegridFlag, box) pair, got $(typeof(m))"))

markbox(m::RegridFlag, N::Int, ::Val{D}) where {D} = ntuple(_ -> 1:N, D)
function markbox(m::Tuple{RegridFlag,Any}, N::Int, ::Val{D}) where {D}
    box = m[2]
    box isa NTuple{D,UnitRange{Int}} || throw(ArgumentError(
        "a flag box must be an NTuple{$D,UnitRange{Int}} of interior indices, " *
        "got $(typeof(box))"))
    all(r -> !isempty(r) && first(r) >= 1 && last(r) <= N, box) || throw(ArgumentError(
        "flag box $box is empty or outside the interior indices 1:$N"))
    return box
end
markbox(m, N::Int, ::Val) = markflag(m)      # not a flag at all: complain about that

"""
    buffered_flags(forest, flags, buffer) -> Vector{RegridFlag}

The flags of `flags` widened by a `buffer`-cell margin around every
region asking to refine — step 2 of the regridding sequence in
`CODE.md`, evaluated in mark space, before any refine or coarsen is
applied and before balancing, without touching `forest`.

For each block marked `Refine`, its box of firing cells (the whole
interior when the flagging function reported no box) is dilated by
`buffer` cells per dimension *at that block's own resolution*. Every
other leaf the dilated box reaches joins the buffer:

- its `Coarsen` is demoted to `Keep` unconditionally, which is what
  suppresses coarsen/refine flicker at the edge of the refined region;
- it is marked `Refine` if its level is at or below the source's — a
  finer neighbour is already fine enough and gets only the demotion.

The dilated box reaches the neighbour in direction `δ` exactly when it
leaves the block along *every* nonzero component of `δ`, so a feature
near one face recruits that face's neighbour only, while one near a
corner recruits the face, edge, and corner neighbours on that side —
the conjunction comes out of the box geometry rather than being
computed by hand.

Recruitment is a single pass over the original marks: a block pulled
into the buffer does not itself recruit further neighbours. `buffer` is
therefore limited to `N` cells, one block width, so that the dilated
box cannot reach past the first ring of neighbours.
"""
function buffered_flags(forest::Forest{D}, flags::AbstractVector,
                        buffer::Integer) where {D}
    N = forest.N
    buffer >= 0 || throw(ArgumentError("buffer must be non-negative, got $buffer"))
    buffer <= N || throw(ArgumentError(
        "buffer of $buffer cells exceeds the block width N = $N; recruitment is a " *
        "single pass and cannot reach past the first ring of neighbours"))

    # Validate every box even when there is no buffering to do, so that a
    # malformed box is reported the same way either way.
    boxes = [markbox(m, N, Val(D)) for m in flags]
    marks = RegridFlag[markflag(m) for m in flags]
    buffer == 0 && return marks

    directions = alldirections(Val(D))
    for (b, k) in enumerate(forest.leaves)
        # The *reported* flag, not `marks[b]`: a block recruited into the
        # buffer by an earlier source must not become a source itself.
        markflag(flags[b]) === Refine || continue
        box = boxes[b]
        # The dilated box leaves the block in direction δ[d] = ∓1 only if
        # it crosses that face; a tangential dimension never restricts.
        exits = ntuple(d -> (first(box[d]) - buffer < 1, last(box[d]) + buffer > N), D)
        for δ in directions
            reaches = all(d -> δ[d] == 0 ||
                               (δ[d] < 0 ? exits[d][1] : exits[d][2]), 1:D)
            reaches || continue
            for nk in neighbor_keys(forest, k, δ)
                j = find_leaf(forest, nk)
                j === nothing && continue          # cannot happen: nk is a leaf
                marks[j] === Coarsen && (marks[j] = Keep)
                level(nk) <= level(k) && (marks[j] = Refine)
            end
        end
    end
    return marks
end

"""
    complete_marks(forest, flags; buffer=0) -> Vector{MortonKey}

The sorted leaf array that `flags` asks for, completed so that the
result is still 2:1 balanced.

`flags` holds one entry per leaf, each either a [`RegridFlag`](@ref) or
a `(flag, box)` pair as [`flag_blocks`](@ref) produces. `buffer` is a
margin in **cells**, applied first: it widens every refinement request
to the neighbouring leaves its box of firing cells comes within `buffer`
cells of — see [`buffered_flags`](@ref).

Refinement is applied first, then coarsening — but only for sibling
groups where all `2^D` children are present as leaves and all of them
ask to coarsen, since refinement is all-or-nothing. The result is then
balanced, which may refine blocks the application did not flag, and may
undo a coarsening that balance cannot support.

`forest` is not modified.
"""
function complete_marks(forest::Forest{D}, flags::AbstractVector;
                        buffer::Integer=0) where {D}
    length(flags) == nleaves(forest) || throw(DimensionMismatch(
        "got $(length(flags)) flags for $(nleaves(forest)) leaves"))
    marks = buffered_flags(forest, flags, buffer)

    # A sibling group coarsens only if it is complete and unanimous.
    wanted = Dict{MortonKey{D},Int}()
    for (k, f) in zip(forest.leaves, marks)
        f === Coarsen && level(k) > 0 || continue
        parent = parentkey(k)
        wanted[parent] = get(wanted, parent, 0) + 1
    end
    coarsening = Set{MortonKey{D}}(p for (p, n) in wanted if n == 2^D)

    candidate = Vector{MortonKey{D}}()
    sizehint!(candidate, length(forest.leaves))
    emitted = Set{MortonKey{D}}()
    for (k, f) in zip(forest.leaves, marks)
        parent = level(k) > 0 ? parentkey(k) : nothing
        if parent !== nothing && parent in coarsening
            # Descendants are contiguous on the curve, so emitting the
            # parent at the first child keeps the array sorted.
            if !(parent in emitted)
                push!(candidate, parent)
                push!(emitted, parent)
            end
        elseif f === Refine && level(k) < MAX_LEVEL
            append!(candidate, sortedchildkeys(k))
        else
            push!(candidate, k)
        end
    end

    # Balance the candidate tree without disturbing the live one.
    scratch = Forest{D}(forest.roots, forest.periodic, forest.extents,
                        forest.N, forest.G, candidate, Ref(0))
    balance!(scratch)
    return scratch.leaves
end

# Classify every new leaf by where its data comes from, and batch the
# transfers by (kind, child offset) so each batch shares one set of
# stencils — the same grouping the ghost schedule uses.
function transfer_groups(::Type{T}, forest::Forest{D}, oldleaves, newleaves,
                         operators::Operators) where {T,D}
    oldindex = Dict{MortonKey{D},Int32}(k => Int32(i) for (i, k) in enumerate(oldleaves))
    N, G = forest.N, forest.G
    zerodir = ntuple(_ -> 0, D)

    pairs = Dict{Tuple{Symbol,NTuple{D,Int}},Tuple{Vector{Int32},Vector{Int32}}}()
    record!(kind, offset, target, source) =
        push!.(get!(pairs, (kind, offset), (Int32[], Int32[])), (target, source))

    for (bn, kn) in enumerate(newleaves)
        same = get(oldindex, kn, nothing)
        if same !== nothing
            record!(:copy, zerodir, Int32(bn), same)
            continue
        end

        parent = level(kn) > 0 ? get(oldindex, parentkey(kn), nothing) : nothing
        if parent !== nothing
            record!(:prolong, childoffset(kn), Int32(bn), parent)
            continue
        end

        # Otherwise this block was coarsened, so its children were leaves.
        level(kn) < MAX_LEVEL || throw(ArgumentError(
            "cannot rebuild $kn: it is neither an old leaf, a child of one, nor a parent"))
        for kc in childkeys(kn)
            child = get(oldindex, kc, nothing)
            child === nothing && throw(ArgumentError(
                "cannot rebuild $kn: neither it, its parent, nor its child $kc was a " *
                "leaf before regridding. A single regrid may move a block by at most " *
                "one level, which holds when the previous tree was 2:1 balanced."))
            record!(:restrict, childoffset(kc), Int32(bn), child)
        end
    end

    stencils(kind, o) =
        kind === :copy ? ntuple(d -> copy_stencil(T, N, G, 0), D) :
        kind === :restrict ?
        ntuple(d -> restriction_stencil(T, N, G, 0, o[d], operators), D) :
        ntuple(d -> prolongation_stencil(T, N, G, 0, o[d], operators), D)

    return [TransferGroup{T,D}(kind, stencils(kind, o), targets, sources)
            for ((kind, o), (targets, sources)) in pairs]
end

"""
    regrid!(forest, fieldsets, schedule; flags, buffer=0, boundary=nothing,
            transfer=true)

Refine and coarsen `forest` as `flags` asks, and move every field set's
data onto the new mesh. Returns `true` if the mesh changed.

The steps are those in `CODE.md`: widen the refinement requests by a
`buffer`-cell margin ([`buffered_flags`](@ref)), complete the marks to
preserve 2:1 balance ([`complete_marks`](@ref)), build the new sorted
key list, allocate fresh block storage, and transfer — surviving blocks
copied, newly refined blocks prolongated from their parent, coarsened
blocks restricted from their children.

`flags` holds a [`RegridFlag`](@ref) or a `(flag, box)` pair per leaf,
as [`flag_blocks`](@ref) produces; `buffer` is a width in cells, the
application's choice (feature speed × regrid cadence), and defaults to
no buffering.

`fieldsets` is a single [`FieldSet`](@ref) or a collection of them, all
over `forest`. Each one's storage is replaced in place, so references an
application already holds stay valid, but **block indices do not
survive**: slots are compacted, and `forest.leaves[b]` is the only way
to say which block is which.

`schedule` must be the current schedule for `forest`. Ghosts are filled
from it before the transfer, because a prolongation stencil reads its
parent's ghost layers; pass `boundary` if the domain has non-periodic
faces. Afterwards the schedule is stale and the state vector has changed
length, so an application must rebuild both:

```julia
if regrid!(forest, fs, schedule; flags = flags)
    schedule = GhostSchedule(forest, operators)
    u = statevector(fs); gather!(u, fs)      # then reinit! the integrator
end
```

Set `transfer = false` to rebuild the mesh and storage without moving
data — what the initial-data cycle wants, since it re-evaluates the
initial data on the new mesh instead (see
[`adapt_to_initial_data!`](@ref)).
"""
function regrid!(forest::Forest{D}, fieldsets, schedule::GhostSchedule;
                 flags::AbstractVector, buffer::Integer=0, boundary=nothing,
                 transfer::Bool=true) where {D}
    sets = fieldsets isa FieldSet ? (fieldsets,) : fieldsets
    for fs in sets
        fs.forest === forest || throw(ArgumentError(
            "every field set must be over the forest being regridded"))
        nblocks(fs) == nleaves(forest) || throw(ArgumentError(
            "field set has $(nblocks(fs)) blocks but the forest has " *
            "$(nleaves(forest)) leaves"))
    end
    schedule.forest === forest || throw(ArgumentError(
        "schedule was built for a different forest"))
    isstale(schedule) && throw(ArgumentError(
        "schedule is stale; rebuild it before regridding"))

    oldleaves = copy(forest.leaves)
    newleaves = complete_marks(forest, flags; buffer=buffer)
    newleaves == oldleaves && return false

    if transfer
        # Prolongation from a parent reaches into that parent's ghost
        # layers, so they have to hold data before anything moves.
        for fs in sets
            fill_ghosts!(fs, schedule; boundary=boundary)
        end
    end

    backend = isempty(sets) ? nothing : get_backend(first(sets).work)
    stored = forest.N + 2 * forest.G
    for fs in sets
        fresh = similar(fs.work, ntuple(_ -> stored, D)..., fs.nvars, length(newleaves))
        fill!(fresh, zero(eltype(fresh)))
        if transfer
            groups = transfer_groups(eltype(fs.work), forest, oldleaves, newleaves,
                                     schedule.operators)
            for group in groups
                run_group!(fresh, fs.work, group, fs.nvars, backend)
            end
            synchronize(backend)
        end
        fs.work = fresh
    end

    rebuild_leaves!(forest, newleaves)
    return true
end

"""
    adapt_to_initial_data!(fs, operators; initial, flag, buffer=0, maxpasses=10,
                           boundary=nothing)

Run the initialization cycle from `CODE.md`: fill the initial data, flag,
regrid, then **re-evaluate** the initial data on the new mesh rather than
interpolating it, and repeat until the hierarchy stops changing.

Re-evaluating is the point: interpolating initial data onto a newly
refined block would bake in the coarse mesh's resolution, so the
refinement would never buy anything.

`initial` is an `(x, v) -> value` callback as
[`fill_by_coordinates!`](@ref) takes, and `flag` is a
`(b, key) -> RegridFlag` (or `(b, key) -> (flag, box)`) callback as
[`flag_blocks`](@ref) takes; `buffer` is passed on to
[`regrid!`](@ref). Returns
`(schedule, passes, converged)`; `converged` is `false` if the hierarchy
was still changing when `maxpasses` ran out.
"""
function adapt_to_initial_data!(fs::FieldSet{T,D}, operators::Operators;
                                initial, flag, buffer::Integer=0,
                                maxpasses::Integer=10,
                                boundary=nothing) where {T,D}
    forest = fs.forest
    schedule = GhostSchedule(forest, operators; T=T)
    fill_by_coordinates!(initial, fs)

    for pass in 1:maxpasses
        fill_ghosts!(fs, schedule; boundary=boundary)
        flags = flag_blocks(flag, forest)
        changed = regrid!(forest, fs, schedule; flags=flags, buffer=buffer,
                          boundary=boundary, transfer=false)
        schedule = GhostSchedule(forest, operators; T=T)
        fill_by_coordinates!(initial, fs)
        changed || return (schedule, pass, true)
    end
    return (schedule, Int(maxpasses), false)
end

"""
    total_mass(fs::FieldSet, var=1)

The volume integral of one variable over the domain — `Σ hᴰ u` over
every interior cell, with each block weighted by its own cell volume.

What [`regrid!`](@ref) does to this depends on the operator family:

- With [`Conservative`](@ref OperatorFamily) operators it is preserved to roundoff for
  **any** field. Restriction is the exact volume average, and
  prolongation reconstructs over the coarse cell preserving its average,
  so a parent's children always average back to it.
- With [`PointValue`](@ref OperatorFamily) operators it is preserved only for fields the
  operators reproduce exactly. Coarsening alone still conserves any
  field — order-2 restriction is the `2^D` average — but refinement does
  not: prolongation is not locally conservative, and at a refinement
  boundary the fine region draws on neighbor values through the parent's
  ghosts without those neighbors giving anything up.
"""
function total_mass(fs::FieldSet{T,D}, var::Integer=1) where {T,D}
    forest = fs.forest
    total = zero(float(real(T)))
    for b in 1:nblocks(fs)
        total += spacing(forest, blockkey(fs, b))^D * sum(interiorview(fs, b, var))
    end
    return total
end
