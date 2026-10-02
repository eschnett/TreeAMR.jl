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

Build a flag vector by calling `f(b, key)` for every block this rank
stores, with `b` the local block index and `key` its [`MortonKey`](@ref)
— serially, for every leaf (see [`blockrange`](@ref)).

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
    owned = blockrange(forest)
    offset = first(owned) - 1
    out = Vector{Any}(undef, length(owned))
    threaded_foreach(length(owned)) do b
        out[b] = f(b, forest.leaves[offset + b])
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
box of those cells, in that block's own interior indices `1:N`. The
blocks are the ones this rank stores (see [`blockrange`](@ref)), so over
a distributed forest the result is this rank's part of the flags, which
is what [`regrid!`](@ref) takes; nothing is communicated.

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
    level(blockkey(fs, b)) < lmax ? (Refine, box) : (Keep, box)
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
    n == 0 && return Tuple{Int,NTuple{D,UnitRange{Int}}}[]   # a rank without blocks
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

# A flag in canonical form (M7): what `markflag`, `markbox` and
# `issource` read from either form a caller may report, as one `isbits`
# value. Over a distributed forest each rank reports flags for its own
# blocks, and `regrid!` gathers them into the global vector in curve
# order; a caller's vector may mix the bare and the `(flag, box)` forms,
# so it cannot be gathered as it is ("Regridding" under "Distributed
# meshes" in CODE.md). `explicit` records which form it was, since that
# decides whether a `Keep` is a source. The box has been validated by
# `markbox` when the mark is made.
struct RegridMark{D}
    flag::RegridFlag
    explicit::Bool
    lo::NTuple{D,Int32}
    hi::NTuple{D,Int32}
end

function RegridMark{D}(m, N::Int) where {D}
    box = markbox(m, N, Val(D))              # validates, and complains about a non-flag
    return RegridMark{D}(markflag(m), m isa Tuple, ntuple(d -> Int32(first(box[d])), D),
                         ntuple(d -> Int32(last(box[d])), D))
end

markflag(m::RegridMark) = m.flag
markbox(m::RegridMark{D}, N::Int, ::Val{D}) where {D} =
    ntuple(d -> Int(m.lo[d]):Int(m.hi[d]), D)
issource(m::RegridMark) = m.explicit ? m.flag !== Coarsen : m.flag === Refine

# The limits on `buffer`, shared by `buffered_flags` and the argument
# checks of `regrid!`, which have to refuse it before the flags are
# gathered.
function check_buffer(buffer::Integer, N::Int)
    buffer >= 0 || throw(ArgumentError("buffer must be non-negative, got $buffer"))
    buffer <= N || throw(ArgumentError(
        "buffer of $buffer cells exceeds the block width N = $N; recruitment is a " *
        "single pass and cannot reach past the first ring of neighbours"))
    return nothing
end

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

`flags` has one entry per leaf of the whole forest, in leaf order, and a
vector of another length is refused. Serially that is one per block.
Over a distributed forest (M7) a rank's [`flag_blocks`](@ref) are its
own blocks' only; pass those to [`regrid!`](@ref) with its `buffer`
keyword, which searches the buffer from each rank's own sources and
gathers the result.
"""
function buffered_flags(forest::Forest{D}, flags::AbstractVector,
                        buffer::Integer) where {D}
    N = forest.N
    check_buffer(buffer, N)
    # One flag per leaf of the whole forest, as `complete_marks` asks: the
    # recruits it writes are global leaves. Over a distributed forest a
    # rank's flags are its local blocks' only, which without this check
    # would buffer around the wrong leaves (step 9 of M7 found a
    # downstream calling it that way); `regrid!` buffers those itself.
    length(flags) == nleaves(forest) || throw(DimensionMismatch(
        "got $(length(flags)) flags for $(nleaves(forest)) leaves: buffered_flags " *
        "takes one flag per leaf of the whole forest; over a distributed forest, " *
        "pass the local flags to regrid! with its buffer keyword instead"))

    # Validate every box even when there is no buffering to do, so that a
    # malformed box is reported the same way either way.
    foreach(m -> markbox(m, N, Val(D)), flags)
    marks = RegridFlag[markflag(m) for m in flags]
    buffer == 0 && return marks
    return apply_recruits!(marks, forest, buffer_recruits(forest, flags, 0, buffer))
end

# One leaf drawn into the regrid buffer: global leaf `leaf` is asked by a
# source to hold `level`. `isbits`, so that over a distributed forest the
# recruits can be gathered (M7).
struct Recruit
    leaf::Int32
    level::Int32
end

# The recruits of the buffer's sources among `flags`, which are the marks
# of the leaves `offset + 1 : offset + length(flags)`: every leaf a
# source's dilated box reaches, with the level the source asks for, in
# source order. This is the neighbour search, the expensive half of
# `buffered_flags`, and it touches only the tree, so each source's
# recruits are found in parallel, and over a distributed forest each
# rank searches from its own blocks only (`regrid_marks`). Writing the
# marks from the tasks would race, since one leaf can be recruited by
# several sources at once; `apply_recruits!` writes them afterwards.
function buffer_recruits(forest::Forest{D}, flags::AbstractVector, offset::Int,
                         buffer::Integer) where {D}
    N = forest.N
    directions = alldirections(Val(D))
    return threaded_collect(Recruit, length(flags)) do found, b
        # The *reported* mark: a block recruited into the buffer by
        # another source must not become a source itself.
        m = flags[b]
        issource(m) || return
        k = forest.leaves[offset + b]
        L = requestedlevel(markflag(m), level(k))
        box = markbox(m, N, Val(D))
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
                push!(found, Recruit(j, L))
            end
        end
    end
end

# Rewrite the marks for the recruits: a recruit at level `L` raises a
# leaf below `L` to `Refine` and lifts its `Coarsen` to `Keep` at or
# below `L`. The result does not depend on the order of the recruits, nor
# on how often one appears (M7; "What was done about it" under step 7 in
# CODE.md): `Refine` is never undone, a `Keep` made from `Coarsen` is only
# ever raised to `Refine`, and so a leaf of level `l` ends as `Refine` if
# any recruit asks it for more than `l`, as `Keep` if it was `Coarsen`
# and some recruit asks for `l` exactly, and as it was otherwise — a
# function of its own mark and of the largest level asked of it. That is
# what lets the ranks search separately and apply the union.
function apply_recruits!(marks::Vector{RegridFlag}, forest::Forest,
                         recruits::AbstractVector{Recruit})
    for r in recruits
        j, L = Int(r.leaf), Int(r.level)
        l = level(forest.leaves[j])
        marks[j] === Coarsen && l <= L && (marks[j] = Keep)
        l < L && (marks[j] = Refine)
    end
    return marks
end

# What a rank contributes to the gathered recruits: per leaf only the
# largest level asked of it, and nothing for a leaf already finer than
# that, where a recruit changes no mark. By the argument above the marks
# come out the same, and the gather is at most one entry per leaf the
# rank's sources reach.
function strongest_recruits(forest::Forest, recruits::Vector{Recruit})
    sort!(recruits; by=r -> (r.leaf, r.level))
    out = Recruit[]
    for r in recruits
        level(forest.leaves[r.leaf]) <= r.level || continue
        if !isempty(out) && out[end].leaf == r.leaf
            out[end] = r                           # sorted, so a larger level
        else
            push!(out, r)
        end
    end
    return out
end

# The buffered marks `regrid!` completes (M7): `allmarks` the gathered
# global vector, `marks` this rank's own part of it. Each rank runs the
# neighbour search for its own sources only, and the strongest recruits
# of every rank are gathered and applied on every rank, which gives every
# rank the marks the serial `buffered_flags` gives over `allmarks`, at
# `O(local sources)` rather than `O(sources)` per rank and one more
# `allgatherv`. Serially it is `buffered_flags` itself. Every rank knows
# whether there is a buffer, since it is agreed, so they all gather or
# none does.
function regrid_marks(forest::Forest, allmarks::AbstractVector, marks::AbstractVector,
                      buffer::Integer)
    out = RegridFlag[markflag(m) for m in allmarks]
    buffer == 0 && return out
    recruits = buffer_recruits(forest, marks, first(blockrange(forest)) - 1, buffer)
    if isdistributed(forest)
        recruits = allgatherv(forest.comm, strongest_recruits(forest, recruits))
    end
    return apply_recruits!(out, forest, recruits)
end

"""
    complete_marks(forest, flags; buffer=0) -> Vector{MortonKey}

The sorted leaf array that `flags` asks for, completed so that the
result is still 2:1 balanced.

`flags` holds one entry per leaf, each either a [`RegridFlag`](@ref) or
a `(flag, box)` pair as [`flag_blocks`](@ref) produces — per leaf of the
whole forest, which serially is per block; over a distributed forest
[`regrid!`](@ref) assembles this global vector from every rank's flags
for its own blocks. `buffer` is a
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
    return completed_leaves(forest, buffered_flags(forest, flags, buffer))
end

# `complete_marks` after the buffer: the leaves that the buffered marks,
# one per leaf, ask for, balanced. `regrid!` comes in here with the marks
# of `regrid_marks`.
function completed_leaves(forest::Forest{D}, marks::Vector{RegridFlag}) where {D}
    length(marks) == nleaves(forest) || throw(DimensionMismatch(
        "got $(length(marks)) marks for $(nleaves(forest)) leaves"))

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

    # Balance the candidate tree without disturbing the live one. It
    # carries the live forest's communicator, so that it is the forest
    # the ranks will hold; balancing sends nothing, since every rank
    # holds the whole tree and arrives at the same leaves. Its state is
    # its own, so it has no buffer pool, and would make its own if it
    # exchanged anything.
    scratch = typeof(forest)(forest.roots, forest.periodic, forest.reflecting,
                             forest.extents, forest.N, candidate, ForestState(),
                             forest.comm)
    balance!(scratch)
    return scratch.leaves
end

# Classify new leaves by where their data come from, and batch the
# transfers by (kind, child offset) so each batch shares one set of
# stencils — the same grouping the ghost schedule uses, under the same
# `GroupKey`, with direction zero. Targets are new leaves and sources old
# ones, both as global leaf indices.
#
# Not every new leaf (amended in M7 after step 7, which measured the
# replicated classification at 38 % of a rank's regrid at 512 ranks): a
# rank needs the transfers of the new leaves it will own, `newrange`,
# and of those whose sources it owns now, which are the new leaves that
# overlap its old leaves `oldrange` — one contiguous range of new
# indices, since refinement and coarsening keep the curve order
# (`overlapping_leaves`). The leaves are classified in ascending order,
# so every list comes out as the whole classification's, restricted to
# them; `split_regrid` keeps only this rank's ends of it, and the stage
# layouts are sorted explicitly, so no message changes. An old leaf is
# found in the sorted old leaves by a search that starts at the previous
# target's sources (`findfrom`), rather than in a `Dict` of every old
# leaf. The default ranges are every leaf, the serial classification.
function regrid_sources(oldleaves::AbstractVector{MortonKey{D}},
                        newleaves::AbstractVector{MortonKey{D}};
                        oldrange::UnitRange{Int}=1:length(oldleaves),
                        newrange::UnitRange{Int}=1:length(newleaves)) where {D}
    targets = Int[]
    for r in union_ranges(newrange, overlapping_leaves(oldleaves, newleaves, oldrange))
        append!(targets, r)
    end

    # Classify the new blocks in parallel (lookups only), then merge in
    # block order so the batches come out the same whatever the thread
    # count. The sources of ascending new leaves are non-decreasing old
    # leaves, so within a chunk each target's lookups start where the
    # previous target's sources were found.
    perblock = [Pair{GroupKey{D},Int32}[] for _ in 1:length(targets)]
    threaded_chunks(length(targets)) do _, chunk
        from = 1
        for t in chunk
            from = classify_regrid!(perblock[t], oldleaves, newleaves[targets[t]], from)
        end
    end

    pairs = TransferPairs{D}()
    for (t, bn) in enumerate(targets)
        for (key, source) in perblock[t]
            push!.(get!(pairs, key, (Int32[], Int32[])), (Int32(bn), source))
        end
    end
    return pairs
end

# Where new leaf `kn`'s data come from, pushed onto `out` as
# `GroupKey => old leaf`: the old leaf itself (a copy), its parent (a
# prolongation), or each of its children (a restriction). Every source
# is at or after old leaf `from`; returns the last one found, where the
# next target's search can start.
function classify_regrid!(out::Vector{Pair{GroupKey{D},Int32}},
                          oldleaves::AbstractVector{MortonKey{D}}, kn::MortonKey{D},
                          from::Int) where {D}
    zerodir = ntuple(_ -> 0, D)
    same = findfrom(oldleaves, kn, from)
    if same != 0
        push!(out, GroupKey{D}(:copy, zerodir, zerodir, 0) => Int32(same))
        return same
    end

    parent = level(kn) > 0 ? findfrom(oldleaves, parentkey(kn), from) : 0
    if parent != 0
        push!(out, GroupKey{D}(:prolong, zerodir, childoffset(kn), 0) => Int32(parent))
        return parent
    end

    # Otherwise this block was coarsened, so its children were leaves.
    level(kn) < MAX_LEVEL || throw(ArgumentError(
        "cannot rebuild $kn: it is neither an old leaf, a child of one, nor a parent"))
    last_ = from
    for kc in childkeys(kn)
        child = findfrom(oldleaves, kc, from)
        child == 0 && throw(ArgumentError(
            "cannot rebuild $kn: neither it, its parent, nor its child $kc was a " *
            "leaf before regridding. A single regrid may move a block by at most " *
            "one level, which holds when the previous tree was 2:1 balanced."))
        push!(out, GroupKey{D}(:restrict, zerodir, childoffset(kc), 0) => Int32(child))
        last_ = max(last_, child)
    end
    return last_
end

# The index of `k` in the sorted `v`, or 0, given that it is not before
# `from`: a few steps along the curve first, since a regrid's next source
# is almost always among them, and a binary search over the rest
# otherwise — the first lookup of a chunk, or after a run of new leaves
# without old ones.
function findfrom(v::AbstractVector{MortonKey{D}}, k::MortonKey{D}, from::Int) where {D}
    n = length(v)
    for i in from:min(from + 3, n)
        x = v[i]
        x == k && return i
        isless(k, x) && return 0
    end
    from + 4 > n && return 0
    i = from + 3 + searchsortedfirst(view(v, (from + 4):n), k)
    return i <= n && v[i] == k ? i : 0
end

# The new leaves that overlap the old leaves `oldrange`, as one range of
# new indices. Both arrays tile the same brick in curve order, so they
# are a contiguous run: from the new leaf that covers where the first
# old leaf begins — that leaf itself or an ancestor, which precedes it,
# else its first descendant, which follows it directly — to the last new
# leaf at or before the last old leaf's deepest last descendant, the
# last node of its subtree in the curve's pre-order. An ancestor of that
# old leaf qualifies, since everything after the ancestor lies beyond
# its subtree; anything later lies beyond the old leaf's own.
function overlapping_leaves(oldleaves::AbstractVector{MortonKey{D}},
                            newleaves::AbstractVector{MortonKey{D}},
                            oldrange::UnitRange{Int}) where {D}
    isempty(oldrange) && return 1:0
    a, z = oldleaves[first(oldrange)], oldleaves[last(oldrange)]
    i = searchsortedlast(newleaves, a)
    lo = i >= 1 && (newleaves[i] == a || isancestor(newleaves[i], a)) ? i : i + 1
    shift = MAX_LEVEL - level(z)
    deepest = MortonKey{D}(z.root, MAX_LEVEL,
                           map(c -> (UInt64(c) << shift) | ((UInt64(1) << shift) - 1),
                               z.coords))
    return lo:searchsortedlast(newleaves, deepest)
end

# Two ranges as ascending, disjoint ranges covering both.
function union_ranges(a::UnitRange{Int}, b::UnitRange{Int})
    isempty(a) && return (b,)
    isempty(b) && return (a,)
    first(a) > first(b) && ((a, b) = (b, a))
    last(a) + 1 >= first(b) && return (first(a):max(last(a), last(b)),)
    return (a, b)
end

# The regrid transfers split by where their ends live (M7): targets in
# the new partition, `newrange` this rank's new leaves, and sources in
# the old one, `oldrange` its old leaves. A transfer with both ends here
# is *local*, shifted to local block indices (new ones for the target,
# old ones for the source); one with only its target here is
# *received*, one with only its source here is *sent*, both kept global
# for the stage builder; the rest belong to other ranks. Each list stays
# in the order of `pairs`, so the local targets are still
# non-decreasing.
function split_regrid(pairs::TransferPairs{D}, oldrange::UnitRange{Int},
                      newrange::UnitRange{Int}) where {D}
    local_, sent, received = TransferPairs{D}(), TransferPairs{D}(), TransferPairs{D}()
    toffset, soffset = Int32(first(newrange) - 1), Int32(first(oldrange) - 1)
    for (key, (targets, sources)) in pairs, i in eachindex(targets)
        t, s = targets[i], sources[i]
        heret, heres = t in newrange, s in oldrange
        into, tt, ss = heret && heres ? (local_, t - toffset, s - soffset) :
                       heret ? (received, t, s) : heres ? (sent, t, s) :
                       (nothing, t, s)
        into === nothing && continue
        push!.(get!(into, key, (Int32[], Int32[])), (tt, ss))
    end
    return local_, sent, received
end

# The regrid transfer of one layout as one stage (M7, tag `REGRID_TAG`),
# from the classified `pairs`. Its local groups are the serial transfer
# groups restricted to this rank's ends; its messages are built by the
# exchange's own `remote_stage`, with the targets in the new partition
# and the sources in the old, so the old owner of each source evaluates
# the transfer — a copy, a prolongation from the parent, or one child's
# share of a restriction — into a slot shaped like that transfer's target
# box, and the new owner copies it into place. Serially both ranges are
# every leaf, nothing is split, and the stage is the serial groups with
# no messages.
#
# Every target here is the new block's *owned* range — the δ = 0 case of
# the ghost builders. A fresh block's shared plane and ghosts are left
# to the next `fill_ghosts!`, as ghosts always are; in a vertex-like
# dimension that makes coarsening halved injection from the children's
# even points, all of which they own.
function regrid_stage(::Type{T}, N::Int, G::NTuple{D,Int}, c::NTuple{D,Int},
                      operators::Operators, backend::Backend, pairs::TransferPairs{D};
                      oldrange::UnitRange{Int}, newrange::UnitRange{Int},
                      nold::Int, nnew::Int, oldowner, newowner) where {T,D}
    # These stencils are rebuilt on every regrid — the child offsets
    # involved depend on which blocks moved — so, like the schedule's,
    # they are uploaded here, once, rather than at the launch.
    build1(key) =
        key.kind === :copy ? ntuple(d -> copy_stencil(T, N, G[d], c[d], 0), D) :
        key.kind === :restrict ?
        ntuple(d -> restriction_stencil(T, N, G[d], c[d], 0, key.offset[d], operators),
               D) :
        ntuple(d -> prolongation_stencil(T, N, G[d], c[d], 0, key.offset[d], operators),
               D)
    built = Dict{GroupKey{D},Any}()
    stencils(key) = get!(() -> build1(key), built, key)

    serial = oldrange == 1:nold && newrange == 1:nnew
    local_, sent, received = serial ? (pairs, nothing, nothing) :
                             split_regrid(pairs, oldrange, newrange)
    GRP = grouptype(backend, T, Val(D))
    locals = GRP[todevice(backend, TransferGroup{T,D}(key.kind, stencils(key), targets,
                                                      sources))
                 for (key, (targets, sources)) in local_]
    ST = stagetype(backend, T, Val(D))
    remote = serial ? nothing :
             remote_stage(remotetype(ST), T, backend, sent, received, stencils, key -> 0;
                          targetowner=newowner, sourceowner=oldowner,
                          targetrange=newrange, sourcerange=oldrange)
    return ST(REGRID_TAG, locals, remote)
end

# The serial regrid transfer's groups, from the old and the new leaf
# arrays: what `regrid!` runs over a forest on one rank, kept as a
# function of its own for the tests and `bench/gpu.jl`.
function transfer_groups(::Type{T}, forest::Forest{D}, G::NTuple{D,Int},
                         c::NTuple{D,Int}, oldleaves, newleaves,
                         operators::Operators, backend::Backend) where {T,D}
    nold, nnew = length(oldleaves), length(newleaves)
    stage = regrid_stage(T, forest.N, G, c, operators, backend,
                         regrid_sources(oldleaves, newleaves); oldrange=1:nold,
                         newrange=1:nnew, nold=nold, nnew=nnew, oldowner=nothing,
                         newowner=nothing)
    return stage.locals
end

# The argument checks of `regrid!`, run on every rank before anything is
# gathered so that a refusal on some ranks is raised on all of them
# (`collective_checks`). Returns this rank's flags in canonical form and
# a hash of what every rank must agree on beyond the forest: the field
# sets' layouts and schedules' operators, in order, `buffer`, `transfer`
# and whether there is a hook — a rank that moved other field sets, or
# the same ones in another order, would match its regrid messages to the
# wrong field set's.
function check_regrid(forest::Forest{D}, sets, flags, buffer::Integer, transfer::Bool,
                      boundary) where {D}
    layout = Any[transfer, Int(buffer), boundary === nothing]
    for p in sets
        p isa Pair && p.first isa FieldSet || throw(ArgumentError(
            "regrid! takes `fs => schedule` pairs, got a $(typeof(p)). Each field " *
            "set brings its own schedule, since a schedule belongs to a layout " *
            "(G, element type, backend) and not to the forest; write " *
            "`fs => nothing` for a set that should only be resized."))
        fs, sched = p
        fs.forest === forest || throw(ArgumentError(
            "every field set must be over the forest being regridded"))
        nblocks(fs) == length(blockrange(forest)) || throw(ArgumentError(
            "field set has $(nblocks(fs)) blocks but the forest has " *
            "$(length(blockrange(forest))) on this rank"))
        backend = get_backend(fs.work)
        push!(layout, (fs.nvars, fs.G, fs.centering, eltype(fs.work),
                       nameof(typeof(backend)), sched === nothing))
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
        # What `fill_ghosts!` would refuse further down, refused here,
        # where the refusal is agreed between the ranks.
        eltype(fs.work) == scheduletype(sched) || throw(ArgumentError(
            "the field set stores $(eltype(fs.work)) but its schedule carries " *
            "$(scheduletype(sched)) weights; pair each field set with its own schedule"))
        samebackend(backend, sched.backend) || throw(ArgumentError(
            "the field set lives on $(nameof(typeof(backend))) but its schedule was " *
            "built for $(nameof(typeof(sched.backend))); pair each field set with its " *
            "own schedule"))
        ops = sched.operators
        push!(layout, (Int(ops.family), ops.prolongation, ops.restriction))
    end
    check_buffer(buffer, forest.N)
    length(flags) == length(blockrange(forest)) || throw(DimensionMismatch(
        "got $(length(flags)) flags for the $(length(blockrange(forest))) blocks " *
        "this rank stores" *
        (isdistributed(forest) ? " (one per local block, not per leaf of the forest)" :
         "")))
    marks = RegridMark{D}[RegridMark{D}(m, forest.N) for m in flags]
    return marks, layouthash(layout...)
end

scheduletype(::GhostSchedule{T}) where {T} = T

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
are compacted, and [`blockkey`](@ref)`(fs, b)` is the only way to say
which block is which.

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

Over a forest distributed between ranks (M7) the call is collective:
every rank makes it, with the same field sets, schedules, `buffer` and
`transfer`, and `flags` holds one entry per **local** block, as
[`flag_blocks`](@ref) and [`firing_boxes`](@ref) produce them. The ranks
gather the flags into the global vector, complete it identically, and
so arrive at the same leaves and return the same `Bool`. The new leaves
are split over the ranks afresh, and the transfer moves each block to
its new owner: its old owner — the owner of the parent for a refined
block, of each child for a coarsened one — evaluates the transfer and
sends the result. Every block comes out bit for bit as a serial regrid
writes it, whatever the number of ranks. An argument that some ranks'
checks refuse is refused on all of them, and so is a forest that
differs between ranks; see "Distributed meshes" in `CODE.md`.
"""
function regrid!(forest::Forest{D}, pairs;
                 flags::AbstractVector, buffer::Integer=0, boundary=nothing,
                 transfer::Bool=true) where {D}
    sets = pairs isa Pair ? (pairs,) : pairs
    # The checks run on every rank, and a refusal on any of them is
    # raised on all of them, together with the forest digest: one
    # `allgather` over a distributed forest, nothing serially (M7).
    marks = collective_checks(forest, "regrid!") do
        check_regrid(forest, sets, flags, buffer, transfer, boundary)
    end
    comm = forest.comm
    distributed = isdistributed(forest)
    # Each rank flagged its own blocks; the decision is made over all of
    # them, replicated, so every rank arrives at the same new leaves and
    # returns the same answer. The buffer's neighbour search is the one
    # part that is not replicated: each rank searches from its own
    # sources, and the recruits are gathered (`regrid_marks`).
    allmarks = distributed ? allgatherv(comm, marks) : marks

    oldleaves = copy(forest.leaves)
    newleaves = completed_leaves(forest, regrid_marks(forest, allmarks, marks, buffer))
    newleaves == oldleaves && return false

    # The partitions before and after: this rank's old leaves, which its
    # field sets store now, and its new ones (`blockrange` of the forest
    # once it holds `newleaves`).
    nold, nnew = length(oldleaves), length(newleaves)
    P, rank = commsize(comm), commrank(comm)
    oldrange = blockrange(forest)
    newrange = equalsplit(nnew, P, rank + 1)
    oldowner(i) = equalsplit_part(nold, P, Int(i)) - 1
    newowner(j) = equalsplit_part(nnew, P, Int(j)) - 1
    sources = nothing                            # classified once, on first use

    for (fs, sched) in sets
        # Per field set, not once from the first one: nothing says two
        # field sets over the same forest share a backend, a ghost width,
        # or an operator family.
        backend = get_backend(fs.work)
        move = transfer && sched !== nothing
        if move
            # Prolongation from a parent reaches into that parent's ghost
            # layers, so they have to hold data before anything moves.
            # Over a distributed forest this is the distributed fill, so a
            # parent's ghosts are current on its owner.
            fill_ghosts!(fs, sched; boundary=boundary)
        end
        stored = storedsize(forest.N, fs.G, staggers(fs))
        fresh = similar(fs.work, stored..., fs.nvars, length(newrange))
        zerofill!(fresh, backend)
        if move
            sources === nothing && (sources = regrid_sources(oldleaves, newleaves;
                                                             oldrange=oldrange,
                                                             newrange=newrange))
            stage = regrid_stage(eltype(fs.work), forest.N, fs.G, staggers(fs),
                                 sched.operators, backend, sources; oldrange=oldrange,
                                 newrange=newrange, nold=nold, nnew=nnew,
                                 oldowner=oldowner, newowner=newowner)
            # The transfer moves every cell in the domain, so it is
            # threaded the same way a ghost phase is — its groups are
            # just as uneven, a whole block against a single child — and
            # by owner of the *new* block, which is also the thread that
            # zeroed it above and will compute on it next. Over a
            # distributed forest it is one stage, which is also the
            # repartitioning: a kept block whose owner changes is a copy
            # between ranks.
            sends = run_stage!(fresh, fs.work, fs.nvars, nothing, stage, forest, backend,
                               nothing)
            sends === nothing || waitall(comm, sends)
            # The stage lives for this call, so its buffers go back to the
            # forest's pool now that nothing is in flight, for the next
            # field set's stage or the next regrid to take.
            stage.remote === nothing || release_stage!(forest, stage.remote)
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

Over a distributed forest (M7) the cycle is collective, as every
[`regrid!`](@ref) in it is: `flag` is called for this rank's blocks,
`flags` returns the flags of this rank's blocks, and every rank leaves
with the same mesh after the same number of passes. A rank may start
without blocks, as all but one do from a single leaf.
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
