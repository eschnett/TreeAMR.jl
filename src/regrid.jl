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
indices `1:N`. Reporting a box is what makes a block a source of the
regrid buffer, so the form carries meaning beyond the flag — see
[`buffered_flags`](@ref). The two forms may be mixed within one vector.

`f` is called once per leaf, threaded over blocks (M5), so it must be a
pure function of its arguments — reading field data is fine, writing to
shared state of its own is not.

A convenience for host-side flagging; an application is free to produce
the vector any other way, which is what will let the flagging kernel run
on the device in M6 while the completion logic stays on the host.
"""
function flag_blocks(f, forest::Forest)
    out = Vector{Any}(undef, nleaves(forest))
    threaded_foreach(nleaves(forest)) do b
        out[b] = f(b, forest.leaves[b])
    end
    # The element type has to come from the values, not from `f`: the
    # bare and `(flag, box)` forms may be mixed within one vector.
    return identity.(out)
end

# --- device flagging (M6) -------------------------------------------------
#
# `flag_blocks` calls the application's `f(b, key)` on the host, and a
# realistic criterion reads its block's data — which is a scalar index
# into a device array. So flagging gets a device form, split where
# `CODE.md` says it should be: the mesh does the mechanical part (a
# per-cell predicate and the min/max reduction over the cells that
# fired) and the application supplies the *verdict*, which is physics
# the mesh cannot know.
#
# The same two launches as `block_mapreduce`'s device path (see
# `state.jl`): `REDUCE_LANES` lanes per block, each striding over the
# block's cells into a private count and box, then one work item per
# block folding its lanes. One work item per block, the M6 form, was
# enough parallelism for a rare operation but sat at 5.5x against the
# RHS path's 35x on the H200 (`CODE.md`, "Parallelism") — and on the
# CPU backend it was *serial*, since an ndrange of up to 1024 items is
# one workgroup there. Every lane owns its output slots, and a count, a
# min and a max are order independent, so the result is bit-identical
# to the one-item form whatever the lane count and however the lanes
# are scheduled.
#
# Widen a running bounding box by one cell. Ordinary functions rather
# than closures written inline: `lo` and `hi` are reassigned inside the
# loop below, and a closure capturing a reassigned local boxes it, which
# on a device is a dynamic `getindex` and so does not compile at all.
# (Found on Metal; it is invisible on the CPU backend, where the box
# costs only a pointer chase.) The fold over lanes below reuses them,
# with another lane's corner in place of a cell index — the same boxing
# trap, in the same loop shape.
@inline widen_lo(lo::NTuple{D,Int32}, i::NTuple{D,<:Integer}) where {D} =
    ntuple(d -> min(lo[d], Int32(i[d])), Val(D))
@inline widen_hi(hi::NTuple{D,Int32}, i::NTuple{D,<:Integer}) where {D} =
    ntuple(d -> max(hi[d], Int32(i[d])), Val(D))

@kernel function firing_lanes_kernel!(counts, los, his, @Const(work), fires,
                                      @Const(origins), @Const(spacings),
                                      ::Val{D}, ::Val{G}, ::Val{C}, ::Val{N},
                                      ::Val{W}) where {D,G,C,N,W}
    g = @index(Global)
    b, l = divrem(g - 1, W) .+ 1
    origin, h = origins[b], spacings[b]
    # The same expression `coordinates` forms, per centering.
    off = pointoffsets(h, C)
    cells = CartesianIndices(ntuple(_ -> N, Val(D)))

    n = 0
    lo = ntuple(_ -> Int32(N + 1), Val(D))
    hi = ntuple(_ -> Int32(0), Val(D))
    for k in l:W:length(cells)
        c = Tuple(cells[k])
        i = ntuple(d -> c[d], Val(D))                  # owned index, 1:N
        idx = ntuple(d -> i[d] + G[d], Val(D))         # stored index
        x = ntuple(d -> origin[d] + (i[d] - off[d]) * h, Val(D))
        if fires(work, idx, b, x)
            n += 1
            lo = widen_lo(lo, i)
            hi = widen_hi(hi, i)
        end
    end
    counts[l, b] = Int32(n)
    los[l, b] = lo
    his[l, b] = hi
end

@kernel function firing_fold_kernel!(counts, los, his, @Const(lcounts), @Const(llos),
                                     @Const(lhis), ::Val{D}, ::Val{W}) where {D,W}
    b = @index(Global)
    n = lcounts[1, b]
    lo = llos[1, b]
    hi = lhis[1, b]
    for l in 2:W
        n += lcounts[l, b]
        lo = widen_lo(lo, llos[l, b])
        hi = widen_hi(hi, lhis[l, b])
    end
    counts[b] = n
    los[b] = lo
    his[b] = hi
end

"""
    firing_boxes(fires, fs::FieldSet) -> Vector{Tuple{Int,NTuple{D,UnitRange{Int}}}}

Per block, how many interior cells satisfied `fires` and the bounding
box of those cells, in that block's own interior indices `1:N`.

`fires(work, idx, b, x) -> Bool` is evaluated at every **owned** point of
every block from a kernel, so it runs on the field set's backend and
must be a pure function of its arguments. `work` is the whole working
array, `idx` the point's **stored** index (so `work[idx..., v, b]` reads
it and `Base.setindex(idx, idx[d] + 1, d)` reaches its neighbour — the
ghosts are there, so a stencil may cross a block face), `b` the block
index, and `x` its position for this field set's centering (see
[`coordinates`](@ref)).

This is the device half of regridding: the mesh does the per-cell sweep
and the min/max reduction, and the application turns the result into
flags, which is where the physics is. A criterion with a maximum
refinement level reads

```julia
flags = map(enumerate(firing_boxes(fires, fs))) do (b, (n, box))
    n == 0 && return Coarsen
    level(forest.leaves[b]) < lmax ? (Refine, box) : (Keep, box)
end
```

and the `(flag, box)` pairs go straight to [`regrid!`](@ref), whose
buffering is driven by exactly this box (see [`buffered_flags`](@ref)).
A block where nothing fired gets the empty box `1:0` in every dimension.

[`flag_blocks`](@ref) remains the host form, for criteria that want the
tree rather than the data.

!!! note "Callbacks on a device"
    The callback becomes a kernel argument, so everything it closes over
    must be `isbits`. A captured `Type` is the usual trip: write
    `oftype(x[1], 2)` rather than closing over `T` and calling `T(2)`.
    The same rule covers captured arrays (pass a device array, or index
    the one the callback is already given) and any mutable state, which
    the purity requirement rules out anyway.
"""
function firing_boxes(fires, fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    backend = get_backend(fs.work)
    n = nblocks(fs)
    W = REDUCE_LANES
    lcounts = allocate(backend, Int32, (W, n))
    llos = allocate(backend, NTuple{D,Int32}, (W, n))
    lhis = allocate(backend, NTuple{D,Int32}, (W, n))
    counts = allocate(backend, Int32, (n,))
    los = allocate(backend, NTuple{D,Int32}, (n,))
    his = allocate(backend, NTuple{D,Int32}, (n,))
    origins = todevice(backend, block_origins(forest, T))
    spacings = todevice(backend, block_spacings(forest, T))
    firing_lanes_kernel!(backend)(lcounts, llos, lhis, fs.work, fires, origins,
                                  spacings, Val(D), Val(fs.G), Val(staggers(fs)),
                                  Val(forest.N), Val(W); ndrange=W * n)
    firing_fold_kernel!(backend)(counts, los, his, lcounts, llos, lhis, Val(D), Val(W);
                                 ndrange=n)
    synchronize(backend)

    hc, hlo, hhi = tohost(counts), tohost(los), tohost(his)
    return [(Int(hc[b]),
             ntuple(d -> hc[b] == 0 ? (1:0) : Int(hlo[b][d]):Int(hhi[b][d]), D))
            for b in 1:n]
end

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

# A block is a dilation source iff it *explicitly* reports a box, with any
# flag but `Coarsen`. With a bare flag only `Refine` is a source: a bare
# `Keep` must not recruit, or every quiescent block would hold its
# neighbours and coarsening would die globally.
issource(m::RegridFlag) = m === Refine
issource(m::Tuple{RegridFlag,Any}) = m[1] !== Coarsen

# The level a source is asking to hold around itself.
requestedlevel(f::RegridFlag, l::Int) = f === Refine ? l + 1 : l

"""
    buffered_flags(forest, flags, buffer) -> Vector{RegridFlag}

The flags of `flags` widened by a `buffer`-cell margin around every
region whose criterion fired — step 2 of the regridding sequence in
`CODE.md`, evaluated in mark space, before any refine or coarsen is
applied and before balancing, without touching `forest`.

The source of a margin is a **box**, not a flag. A block is a dilation
source iff it *explicitly* reports a `(flag, box)` pair with any flag but
`Coarsen`, or is marked with a bare `Refine` (whose box defaults to the
whole interior, the conservative isotropic case). A bare `Keep` is never
a source — else quiescent blocks would recruit their neighbours and
coarsening would die globally — and `Coarsen` is never a source even
with a box.

Each source asks for a **level** `L`: `level + 1` for `Refine`, `level`
for `Keep`. Its box is dilated by `buffer` cells per dimension *at the
source's own resolution*, and every other leaf the dilated box reaches
joins the buffer:

- it is promoted to `Refine` if its level is below `L` — a neighbour
  already at or above `L` is fine enough;
- its `Coarsen` is demoted to `Keep` if its level is at or below `L`,
  which is what suppresses coarsen/refine flicker at the edge of the
  refined region. An over-fine neighbour, above `L`, may still coarsen
  toward it.

A block holding a feature at its target level therefore returns
`(Keep, box)` and holds an equal-level margin that travels with the
feature — the proactive margin a `Refine`-keyed rule cannot give, since
a freshly refined block's box hugs the face the feature entered through.

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

    # Each source's recruits are found in parallel — the neighbor search
    # is the expensive half and touches only the tree — and the marks
    # are then rewritten in a serial pass in block order. Writing them
    # from the tasks would race: one leaf can be recruited by several
    # sources at once.
    recruits = threaded_collect(Pair{Int,Int}, nleaves(forest)) do found, b
        # The *reported* mark, not `marks[b]`: a block recruited into the
        # buffer by an earlier source must not become a source itself.
        issource(flags[b]) || return
        k = forest.leaves[b]
        L = requestedlevel(markflag(flags[b]), level(k))
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
                push!(found, j => L)
            end
        end
    end

    for (j, L) in recruits
        l = level(forest.leaves[j])
        marks[j] === Coarsen && l <= L && (marks[j] = Keep)
        l < L && (marks[j] = Refine)
    end
    return marks
end

"""
    complete_marks(forest, flags; buffer=0) -> Vector{MortonKey}

The sorted leaf array that `flags` asks for, completed so that the
result is still 2:1 balanced.

`flags` holds one entry per leaf, each either a [`RegridFlag`](@ref) or
a `(flag, box)` pair as [`flag_blocks`](@ref) produces. `buffer` is a
margin in **cells**, applied first: every block that reports a box (or
is marked with a bare `Refine`) pulls the neighbouring leaves its box
comes within `buffer` cells of up to the level it asks for — `level + 1`
for `Refine`, `level` for `Keep`. See [`buffered_flags`](@ref).

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
    scratch = typeof(forest)(forest.roots, forest.periodic, forest.reflecting,
                             forest.extents, forest.N, candidate, Ref(0))
    balance!(scratch)
    return scratch.leaves
end

# Classify every new leaf by where its data comes from, and batch the
# transfers by (kind, child offset) so each batch shares one set of
# stencils — the same grouping the ghost schedule uses.
function transfer_groups(::Type{T}, forest::Forest{D}, G::NTuple{D,Int},
                         c::NTuple{D,Int}, oldleaves, newleaves,
                         operators::Operators, backend::Backend) where {T,D}
    oldindex = Dict{MortonKey{D},Int32}(k => Int32(i) for (i, k) in enumerate(oldleaves))
    N = forest.N
    zerodir = ntuple(_ -> 0, D)

    # Classify every new block in parallel (dictionary *lookups* only —
    # nothing is inserted), then merge in block order so the batches come
    # out the same whatever the thread count.
    perblock = [Pair{Tuple{Symbol,NTuple{D,Int}},Int32}[] for _ in 1:length(newleaves)]
    threaded_foreach(length(newleaves)) do bn
        kn = newleaves[bn]
        out = perblock[bn]
        same = get(oldindex, kn, nothing)
        if same !== nothing
            push!(out, (:copy, zerodir) => same)
            return
        end

        parent = level(kn) > 0 ? get(oldindex, parentkey(kn), nothing) : nothing
        if parent !== nothing
            push!(out, (:prolong, childoffset(kn)) => parent)
            return
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
            push!(out, (:restrict, childoffset(kc)) => child)
        end
    end

    pairs = Dict{Tuple{Symbol,NTuple{D,Int}},Tuple{Vector{Int32},Vector{Int32}}}()
    for bn in 1:length(newleaves)
        for (key, source) in perblock[bn]
            push!.(get!(pairs, key, (Int32[], Int32[])), (Int32(bn), source))
        end
    end

    # Every target here is the new block's *owned* range — the δ = 0 case
    # of the same builders. A fresh block's shared plane and ghosts are
    # left to the next `fill_ghosts!`, as ghosts always are; in a
    # vertex-like dimension that makes coarsening halved injection from
    # the children's even points, all of which they own.
    stencils(kind, o) =
        kind === :copy ? ntuple(d -> copy_stencil(T, N, G[d], c[d], 0), D) :
        kind === :restrict ?
        ntuple(d -> restriction_stencil(T, N, G[d], c[d], 0, o[d], operators), D) :
        ntuple(d -> prolongation_stencil(T, N, G[d], c[d], 0, o[d], operators), D)

    # These stencils are rebuilt on every regrid — the child offsets
    # involved depend on which blocks moved — so, like the schedule's,
    # they are uploaded here, once, rather than at the launch.
    GRP = grouptype(backend, T, Val(D))
    return GRP[todevice(backend, TransferGroup{T,D}(kind, stencils(kind, o),
                                                    targets, sources))
               for ((kind, o), (targets, sources)) in pairs]
end

"""
    regrid!(forest, pairs; flags, buffer=0, boundary=nothing, transfer=true)

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

`pairs` is one `fs => schedule` or a collection of them, each field set
over `forest` and each schedule the current one *for that field set*
(amended in M8: with `G` on the field set a schedule belongs to a
layout, so a bare field set no longer says which schedule moves it).
Every set's storage is replaced in place, so references an application
already holds stay valid, but **block indices do not survive**: slots
are compacted, and `forest.leaves[b]` is the only way to say which block
is which.

The target of every transfer is the new block's **owned** range, whatever
the centering: a fresh block's shared boundary plane and its ghosts are
left to the next [`fill_ghosts!`](@ref), as ghosts always are. A test
that inspects a staggered field set after a regrid has to fill ghosts
first.

Ghosts are filled from each schedule before that set's transfer, because
a prolongation stencil reads its parent's ghost layers; pass `boundary`
if the domain has outer faces (neither periodic nor reflecting —
reflecting faces are filled by the schedule itself). Write `fs => nothing` for a set
that should only be **resized** — a computed quantity such as a flux,
which the next right-hand side overwrites anyway, and whose ghost-free
layout has no schedule to fill it from. Afterwards every schedule is
stale and the state vector has changed length, so an application must
rebuild both:

```julia
if regrid!(forest, fs => schedule; flags = flags)
    schedule = GhostSchedule(fs, operators)
    u = statevector(fs); gather!(u, fs)      # then reinit! the integrator
end
```

Set `transfer = false` to rebuild the mesh and storage without moving
data at all — what the initial-data cycle wants, since it re-evaluates
the initial data on the new mesh instead (see
[`adapt_to_initial_data!`](@ref)).
"""
function regrid!(forest::Forest{D}, pairs;
                 flags::AbstractVector, buffer::Integer=0, boundary=nothing,
                 transfer::Bool=true) where {D}
    sets = pairs isa Pair ? (pairs,) : pairs
    for p in sets
        p isa Pair && p.first isa FieldSet || throw(ArgumentError(
            "regrid! takes `fs => schedule` pairs, got a $(typeof(p)). Each field " *
            "set brings its own schedule, since a schedule belongs to a layout " *
            "(G, element type, backend) and not to the forest; write " *
            "`fs => nothing` for a set that should only be resized."))
        fs, sched = p
        fs.forest === forest || throw(ArgumentError(
            "every field set must be over the forest being regridded"))
        nblocks(fs) == nleaves(forest) || throw(ArgumentError(
            "field set has $(nblocks(fs)) blocks but the forest has " *
            "$(nleaves(forest)) leaves"))
        sched === nothing && continue
        sched.forest === forest || throw(ArgumentError(
            "schedule was built for a different forest"))
        isstale(sched) && throw(ArgumentError(
            "schedule is stale; rebuild it before regridding"))
        fs.G == sched.G || throw(ArgumentError(
            "the field set has ghost width G=$(fs.G) but its schedule was built " *
            "for G=$(sched.G); pair each field set with its own schedule"))
        fs.centering == sched.centering || throw(ArgumentError(
            "the field set has centering $(fs.centering) but its schedule was " *
            "built for $(sched.centering); pair each field set with its own " *
            "schedule"))
    end

    oldleaves = copy(forest.leaves)
    newleaves = complete_marks(forest, flags; buffer=buffer)
    newleaves == oldleaves && return false

    for (fs, sched) in sets
        # Per field set, not once from the first one: nothing says two
        # field sets over the same forest share a backend, a ghost width,
        # or an operator family.
        backend = get_backend(fs.work)
        move = transfer && sched !== nothing
        if move
            # Prolongation from a parent reaches into that parent's ghost
            # layers, so they have to hold data before anything moves.
            fill_ghosts!(fs, sched; boundary=boundary)
        end
        stored = storedsize(forest.N, fs.G, staggers(fs))
        fresh = similar(fs.work, stored..., fs.nvars, length(newleaves))
        zerofill!(fresh, backend)
        if move
            groups = transfer_groups(eltype(fs.work), forest, fs.G, staggers(fs),
                                     oldleaves, newleaves, sched.operators, backend)
            # The transfer moves every cell in the domain, so it is
            # threaded the same way a ghost phase is — its groups are
            # just as uneven, a whole block against a single child — and
            # by owner of the *new* block, which is also the thread that
            # zeroed it above and will compute on it next.
            run_phase!(fresh, fs.work, groups, fs.nvars, backend)
            synchronize(backend)
        end
        fs.work = fresh
    end

    rebuild_leaves!(forest, newleaves)
    return true
end

# The M6 form, so that a caller written against it gets told what moved
# rather than a `MethodError` on a three-argument `regrid!`.
regrid!(::Forest, ::Any, ::GhostSchedule; kwargs...) = throw(ArgumentError(
    "regrid! now takes `fs => schedule` pairs rather than field sets and one " *
    "shared schedule: write `regrid!(forest, fs => schedule; flags = ...)`. A " *
    "schedule belongs to a layout (G, element type, backend) from M8 on, and " *
    "different field sets over one forest have different layouts."))

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
[`fill_by_coordinates!`](@ref) takes — or an [`AllVariables`](@ref)`(f)`
with `f(x) -> vals`, which the cycle simply hands on, and which is what
an initial state definable only as a whole needs; `buffer` is passed on
to [`regrid!`](@ref). Returns `(schedule, passes, converged)`;
`converged` is `false` if the hierarchy was still changing when
`maxpasses` ran out.

The criterion is given exactly one of two ways:

- `flag`, a `(b, key) -> RegridFlag` (or `(b, key) -> (flag, box)`)
  callback as [`flag_blocks`](@ref) takes — the host form;
- `flags`, a callable `fs -> flagvector` producing the whole vector at
  once. This is what a device-side criterion wants, since it reduces
  every block in one kernel: `flags = fs -> map(..., firing_boxes(fires, fs))`.
  See [`firing_boxes`](@ref).

The schedule is rebuilt from `fs` itself, so it carries that field set's
ghost width, element type and backend — a device-resident field set
adapts without anything further.
"""
function adapt_to_initial_data!(fs::FieldSet{T,D}, operators::Operators;
                                initial, flag=nothing, flags=nothing,
                                buffer::Integer=0, maxpasses::Integer=10,
                                boundary=nothing) where {T,D}
    (flag === nothing) == (flags === nothing) && throw(ArgumentError(
        "pass exactly one of `flag` (a (b, key) callback, evaluated per block on " *
        "the host) and `flags` (a callable producing the whole flag vector, which " *
        "is what a device-side criterion built on `firing_boxes` produces)"))
    criterion = flags === nothing ? (f -> flag_blocks(flag, f.forest)) : flags

    forest = fs.forest
    schedule = GhostSchedule(fs, operators)
    fill_by_coordinates!(initial, fs)

    for pass in 1:maxpasses
        fill_ghosts!(fs, schedule; boundary=boundary)
        changed = regrid!(forest, fs => schedule; flags=criterion(fs), buffer=buffer,
                          boundary=boundary, transfer=false)
        schedule = GhostSchedule(fs, operators)
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
    R = float(real(T))
    # A volume-weighted `mesh_mapreduce`: one value per block, scaled by
    # its cell volume and summed on the host, so on the CPU the answer
    # does not move when the thread count does (M5). That exactness is
    # how it is written, not what is promised: a sum is guaranteed to
    # roundoff only, and the suite asserts exactly that on a device,
    # where the lanes split the cells differently.
    # `float(real(T))` inline, not the local `R`: see `volume_weighted_norm`.
    return mesh_mapreduce(identity, +, zero(R), fs; vars=var,
                          weight=key -> spacing(float(real(T)), forest, key)^D)
end
