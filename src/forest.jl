"""
    Forest{D,T}

A `D`-dimensional brick of `roots[1] × ... × roots[D]` octree roots
covering the rectangular physical domain `extents`, refined into a
leaf-only linear octree.

- `leaves` is the sorted (by [`MortonKey`](@ref) curve order) flat
  vector of the leaves that currently tile the domain. Only leaves carry
  data; they tile the domain exactly, with no overlap.
- Refinement is **all-or-nothing**: a node is either a leaf or has
  exactly `2^D` children. There is never a partially refined node.
- `periodic[d]` selects whether dimension `d` wraps around the brick.
  Wraparound lives in the neighbor arithmetic, so periodic ghost filling
  needs no special-casing later.
- `reflecting[d]` is a `(lo, hi)` pair selecting which faces of a
  non-periodic dimension are **reflecting** (M10): the solution beyond
  them is its own mirror image, with the parity each variable declares
  on its [`FieldSet`](@ref). The tree does not see them — there is no
  neighbor across a reflecting face, as across any other non-periodic
  one — but the [`GhostSchedule`](@ref) does, and fills their ghosts by
  mirrored transfers. The faces that are neither periodic nor reflecting
  are *outer* faces, and belong to the boundary hook of
  [`fill_ghosts!`](@ref).
- `N` is the per-block interior size, and it is even: cells are the
  tree's geometry, so `N` belongs here. The ghost width `G` does **not**
  — it says how far a stencil reaches into a neighbor's data, which is a
  property of what is stored, and it belongs to the
  [`FieldSet`](@ref) (amended in M8; through M6 it was a forest
  keyword).
- Blocks are cubes, so `extents` must match the aspect ratio of `roots`.
- `T` is the floating-point type the geometry is *computed* in, not
  merely stored in — see [`floattype`](@ref) and "Precision" in
  `CODE.md`.

Root indices are linearized 0-based, dimension 1 fastest, over `roots`;
see [`root_position`](@ref) and [`root_index`](@ref).

    Forest(roots; N, periodic=all false, reflecting=all false,
           extents=one unit per root, leaves=nothing, comm=nothing)
    Forest{T}(roots; ...)                      # geometry in `T`

Without `leaves` the forest starts as its unrefined roots. With it, it
starts from that leaf list instead — keys in curve order, as
`forest.leaves` holds them — which is copied rather than aliased, and
the new forest is at [`generation`](@ref) 0. That is how a checkpoint
is restored, and how M7's ranks will build their forests. Everything
built over a forest trusts its leaves, so the list is refused unless
every key lies in the brick, the keys strictly increase, they tile the
brick exactly — no gap, no overlap — and they are 2:1 balanced (see
[`balance!`](@ref)).

`comm` is the communicator the forest is distributed over (M7), as
[`communicator`](@ref) converts it; `nothing`, the default, is a serial
forest. Every rank holds the whole forest, and the field data of a
[`FieldSet`](@ref) over it are split by [`blockrange`](@ref): each rank
stores the blocks of one contiguous run of the leaves. Every forest
mutation is then collective — the same call with the same arguments on
every rank. See "Distributed meshes" in `CODE.md`.

# Examples

```jldoctest
julia> forest = Forest((2, 2); N = 8, periodic = (true, true));

julia> nleaves(forest)
4

julia> octant = Forest((1, 1, 1); N = 8, reflecting = ntuple(_ -> (true, false), 3));
```
"""
struct Forest{D,T}
    roots::NTuple{D,Int}
    periodic::NTuple{D,Bool}
    reflecting::NTuple{D,Tuple{Bool,Bool}}       # (lo, hi) per dimension
    extents::NTuple{D,Tuple{T,T}}
    N::Int
    leaves::Vector{MortonKey{D}}
    # Bumped whenever the leaf array changes, so anything derived from
    # the tree (a GhostSchedule, say) can detect in O(1) that it is
    # stale — a same-size refine-then-coarsen would otherwise slip past
    # a leaf-count check and silently transfer the wrong data.
    generation::Base.RefValue{Int}
    # The processes the field data are split over (M7). Abstract-typed
    # on purpose, so that `Forest{D,T}` keeps its two parameters and no
    # `FieldSet` or schedule signature downstream changes; the price is
    # a dynamic dispatch per verb, a few per ghost fill and none in a
    # kernel ("Distributed meshes" in CODE.md).
    comm::Communicator
end

# `G` is still accepted as a keyword so that the move can be reported
# instead of surfacing as a bare `MethodError` on an unrecognised
# keyword. It is the first thing a caller written against M6 hits.
const no_forest_ghosts = ArgumentError(
    "the ghost width moved from the forest to the field set in M8: write " *
    "`Forest(roots; N = ...)` and `FieldSet(forest, nvars; G = ...)`. Ghosts " *
    "say how far a stencil reaches into a neighbor's data, which is a property " *
    "of what is stored, not of how space is cut up — two field sets over one " *
    "forest with different G is the normal case. `G` may be a plain integer or " *
    "an NTuple{D,Integer}, one width per dimension.")

# The geometry type is a parameter rather than a fixed `Float64` because
# the *arithmetic*, not just the storage, has to stay inside it: a device
# without hardware fp64 must never evaluate a coordinate in `Float64` on
# its way into a `Float32` field. Converting at the end would not do.
function Forest{T}(roots::NTuple{D,Integer};
                   N::Integer,
                   periodic::NTuple{D,Bool}=ntuple(_ -> false, D),
                   reflecting::NTuple{D,Tuple{Bool,Bool}}=
                       ntuple(_ -> (false, false), D),
                   extents::NTuple{D,Tuple{Real,Real}}=
                       ntuple(d -> (zero(T), T(roots[d])), D),
                   leaves::Union{Nothing,AbstractVector{MortonKey{D}}}=nothing,
                   comm=nothing, G=nothing) where {T,D}
    G === nothing || throw(no_forest_ghosts)
    all(>(0), roots) || throw(ArgumentError("roots must all be positive, got $roots"))
    N > 0 || throw(ArgumentError("N must be positive, got $N"))
    iseven(N) || throw(ArgumentError("N must be even, got $N"))
    for d in 1:D
        periodic[d] && any(reflecting[d]) && throw(ArgumentError(
            "dimension $d is both periodic and reflecting, got reflecting[$d] = " *
            "$(reflecting[d]): a periodic dimension has no faces — its last block " *
            "is the first block's neighbor — so there is nothing for a reflection " *
            "to act on. Drop one of the two."))
    end

    ext = ntuple(d -> (T(extents[d][1]), T(extents[d][2])), D)
    all(d -> ext[d][2] > ext[d][1], 1:D) ||
        throw(ArgumentError("each extent must be nonempty and increasing, got $ext"))
    # Blocks are cubes, so the root spacing must be the same in every
    # dimension.
    h = ntuple(d -> (ext[d][2] - ext[d][1]) / roots[d], D)
    # 4096 eps is the type-generic spelling of the 1e-12 this check used
    # when the geometry was always Float64: 1e-12 / eps(Float64) ≈ 4500.
    # `h` can differ from `h[1]` only by the rounding of one subtraction
    # and one division, so the slack is enormous either way; what matters
    # is that it tracks the type rather than sitting below eps(Float32).
    all(d -> isapprox(h[d], h[1]; rtol=4096 * eps(T)), 1:D) ||
        throw(ArgumentError("blocks must be cubes: extents $ext over roots $roots give " *
                            "anisotropic root spacings $h"))

    rootsI = map(Int, roots)
    if leaves === nothing
        list = [MortonKey{D}(r, 0, ntuple(_ -> 0, D)) for r in 0:(prod(rootsI) - 1)]
        sort!(list)
    else
        # Always a copy: `refine!` and friends rewrite the forest's leaf
        # array in place, which must never reach the caller's vector.
        list = collect(MortonKey{D}, leaves)
    end
    forest = Forest{D,T}(rootsI, periodic, reflecting, ext, Int(N), list, Ref(0),
                         communicator(comm))
    leaves === nothing || check_leaves(forest)
    return forest
end

# Without an explicit `T`, the geometry type follows the extents the
# caller supplied; with no extents either, it is `Float64`. Both branches
# are resolved from argument *types*, so this stays inferable.
function Forest(roots::NTuple{D,Integer};
                N::Integer,
                periodic::NTuple{D,Bool}=ntuple(_ -> false, D),
                reflecting::NTuple{D,Tuple{Bool,Bool}}=ntuple(_ -> (false, false), D),
                extents::Union{Nothing,NTuple{D,Tuple{Real,Real}}}=nothing,
                leaves::Union{Nothing,AbstractVector{MortonKey{D}}}=nothing,
                comm=nothing, G=nothing) where {D}
    G === nothing || throw(no_forest_ghosts)
    if extents === nothing
        return Forest{Float64}(roots; N=N, periodic=periodic, reflecting=reflecting,
                               leaves=leaves, comm=comm)
    end
    T = float(promote_type(ntuple(d -> promote_type(typeof(extents[d][1]),
                                                    typeof(extents[d][2])), D)...))
    return Forest{T}(roots; N=N, periodic=periodic, reflecting=reflecting,
                     extents=extents, leaves=leaves, comm=comm)
end

# Validate a caller's leaf list (the `leaves` keyword), already copied
# into the candidate `forest`. The storage, the ghost schedule and the
# regrid all trust `forest.leaves` without looking at it again — that is
# what lets them be built once per tree change — so a list read from a
# file, or received from another rank, is checked here instead: every
# key in the brick, strictly increasing in curve order, tiling the brick
# exactly, and 2:1 balanced. Each refusal names the first leaf at which
# the list goes wrong.
function check_leaves(forest::Forest{D}) where {D}
    leaves = forest.leaves
    nroots = prod(forest.roots)
    isempty(leaves) && throw(ArgumentError(
        "the leaf list is empty: the leaves tile the brick, so there is at least one " *
        "per root, and the brick $(forest.roots) has $nroots roots"))
    # The key constructor checks the level and the coordinates, but it
    # cannot check the root, which is an index into a brick it never sees.
    for (i, k) in enumerate(leaves)
        k.root < nroots || throw(ArgumentError(
            "leaf $i, $k, is in root $(k.root), but the brick $(forest.roots) has " *
            "roots 0:$(nroots - 1): a root index counts through the brick, dimension " *
            "1 fastest, so this key belongs to a larger one"))
    end
    for i in 2:length(leaves)
        a, b = leaves[i - 1], leaves[i]
        a == b && throw(ArgumentError(
            "leaves $(i - 1) and $i are both $a: a duplicate leaf would be two blocks " *
            "for one region of space"))
        isless(a, b) || throw(ArgumentError(
            "leaves $(i - 1) and $i, $a and $b, are out of curve order: the list must " *
            "be strictly increasing, as `forest.leaves` is, because block `b` of every " *
            "field set is leaf `b`, and sorting the list here would silently reorder " *
            "whatever data were stored with it"))
    end
    # One walk along the curve. `next` is the node at which the part of
    # the brick not yet covered begins, and each leaf must begin there
    # too: be `next` or one of its first-corner descendants, the only
    # nodes that start where it starts. Then the leaves tile the brick
    # exactly, and a strictly increasing list that does not has either a
    # leaf inside the one before it, or a gap.
    next = MortonKey{D}(0, 0, ntuple(_ -> 0, D))
    for (i, k) in enumerate(leaves)
        if next === nothing || !begins_at(k, next)
            # The leaves before `k` tile exactly up to `next`, and `k`
            # comes after its predecessor in curve order. So `k` either
            # lies inside that predecessor, or begins beyond `next`,
            # leaving a gap. (Once the last root is covered, `next` is
            # `nothing`, and a later leaf in the brick can only be the
            # former.)
            i > 1 && isancestor(leaves[i - 1], k) && throw(ArgumentError(
                "leaf $i, $k, overlaps leaf $(i - 1), $(leaves[i - 1]), which contains " *
                "it: the leaves tile the brick with no overlap, since a node is either " *
                "a block or refined into its 2^$D children, never both"))
            place = i == 1 ? "the brick begins, and leaf 1, $k, does not" :
                             "leaf $(i - 1), $(leaves[i - 1]), ends, and leaf $i, $k, " *
                             "begins further on"
            throw(ArgumentError(
                "the leaves leave a gap: no leaf covers the beginning of $next, which " *
                "is where $place. The leaves tile the brick exactly, and a region " *
                "without a leaf has no block to hold its data"))
        end
        next = curve_successor(k, nroots)
    end
    next === nothing || throw(ArgumentError(
        "the leaves leave a gap at the end of the brick: the last leaf, " *
        "$(length(leaves)), $(leaves[end]), ends where $next begins, and no leaf " *
        "covers anything from there through the last root, $(nroots - 1). The leaves " *
        "tile the brick exactly, and a region without a leaf has no block to hold " *
        "its data"))
    # Balance is the threaded check; the serial search that names the
    # offending pair runs only once it has failed.
    isbalanced(forest) && return nothing
    dirs = alldirections(Val(D))
    for (i, k) in enumerate(leaves), δ in dirs, nb in neighbor_keys(forest, k, δ)
        abs(level(nb) - level(k)) > 1 && throw(ArgumentError(
            "the leaves are not 2:1 balanced: leaf $i, $k, touches $nb, and their " *
            "levels differ by more than one. The list is refused rather than " *
            "rebalanced, because a balanced forest can only ever produce a balanced " *
            "list, so this one was damaged or made by hand; and because the ghost " *
            "schedule assumes the balance, and would silently build wrong " *
            "prolongations from an unbalanced mesh"))
    end
    return nothing
end

# Whether leaf `k` begins where node `e` begins, as `e` itself or one of
# its first-corner descendants: same root, no coarser, and its
# coordinates are `e`'s scaled to its level, with nothing added. Exact
# in `UInt32`: the scaled coordinates stay below `2^level(k)`, and a
# shift by the full 32 bits (from level 0 to MAX_LEVEL) is defined, as 0.
function begins_at(k::MortonKey{D}, e::MortonKey{D}) where {D}
    (k.root == e.root && k.level >= e.level) || return false
    shift = Int(k.level) - Int(e.level)
    return all(d -> k.coords[d] == e.coords[d] << shift, 1:D)
end

# The node at which the curve continues after the subtree of `k`: `k`'s
# next sibling, else that of its nearest ancestor that has one, else the
# next root; `nothing` after the last root. Level by level, so exact at
# any depth and in any `D`, with no packed curve index to overflow.
function curve_successor(k::MortonKey{D}, nroots::Int) where {D}
    lvl = Int(k.level)
    coords = k.coords
    while lvl > 0
        # A node's position among its siblings is the bits `coords .& 1`,
        # dimension 1 most significant. The next sibling adds one: it sets
        # the least significant clear bit and clears the set ones below it.
        d = findlast(d -> iseven(coords[d]), 1:D)
        if d !== nothing
            sibling = ntuple(D) do e
                e < d ? coords[e] : e == d ? coords[e] | 0x1 : coords[e] & ~UInt32(1)
            end
            return MortonKey{D}(k.root, lvl, sibling)
        end
        # The last sibling: the parent's subtree ends here too.
        lvl -= 1
        coords = map(c -> c >> 1, coords)
    end
    k.root + 1 < nroots || return nothing
    return MortonKey{D}(k.root + 1, 0, ntuple(_ -> 0, D))
end

"""
    floattype(forest::Forest)

The floating-point type `forest`'s geometry is computed in — what
[`spacing`](@ref), [`block_origin`](@ref) and friends return, and the
element type a [`FieldSet`](@ref) or [`GhostSchedule`](@ref) over this
forest takes unless told otherwise.
"""
floattype(::Forest{D,T}) where {D,T} = T

"""
    generation(forest::Forest)

A counter bumped on every change to the leaf array. Structures derived
from the tree record it so they can tell in O(1) whether they are still
valid — see [`GhostSchedule`](@ref).
"""
generation(forest::Forest) = forest.generation[]

"""
    nleaves(forest::Forest)

The number of leaves currently tiling `forest` — all of them, on every
rank. Serially that is also the number of blocks a [`FieldSet`](@ref)
over it stores; over a distributed forest a field set stores the
`length(blockrange(forest))` blocks of this rank (see
[`blockrange`](@ref)), so a per-block array is sized by
[`nblocks`](@ref), never by `nleaves`.
"""
nleaves(forest::Forest) = length(forest.leaves)

"""
    blockrange(forest::Forest) -> UnitRange{Int}

The leaves whose blocks this rank stores: a contiguous range of
`1:nleaves(forest)`, in curve order. Local block `b` of every
[`FieldSet`](@ref) over `forest` is leaf `first(blockrange(forest)) + b - 1`,
which is what [`blockkey`](@ref) returns.

The ranges of the ranks tile `1:nleaves(forest)` in rank order, and their
lengths differ by at most one, the longer ones first — the same
equal-count arithmetic that splits a rank's blocks over its threads
(`CODE.md`, "Distributed meshes"). Every block costs the same under one
global time step, so equal counts are equal work. A rank beyond the
number of leaves owns none, which is allowed. Serially this is
`1:nleaves(forest)`, and local and global block indices coincide.
"""
blockrange(forest::Forest) =
    equalsplit(nleaves(forest), commsize(forest.comm), commrank(forest.comm) + 1)

# The rank whose blocks include global leaf `i` (M7): the inverse of
# `blockrange`, from the same split arithmetic.
leafowner(forest::Forest, i::Integer) =
    equalsplit_part(nleaves(forest), commsize(forest.comm), Int(i)) - 1

# The leaves outside `range` that touch a leaf inside it, as ascending
# global leaf indices: the union of `neighbor_keys` over every direction
# around the leaves of `range`. These are the *candidate remote targets*
# of a distributed schedule (M7): adjacency is mutually discoverable, so
# a leaf of another rank whose ghosts read one of this rank's blocks is
# found from that block across some direction. Kept here because it is
# brick knowledge, as `neighbor_keys` is. Empty when `range` is every
# leaf, which is the serial case, without a search.
function remote_neighbors(forest::Forest{D}, range::UnitRange{Int}) where {D}
    length(range) == nleaves(forest) && return Int[]
    dirs = alldirections(Val(D))
    found = threaded_collect(Int, length(range)) do hits, i
        k = forest.leaves[first(range) + i - 1]
        for δ in dirs, nbr in neighbor_keys(forest, k, δ)
            j = find_leaf(forest, nbr)::Int
            j in range || push!(hits, j)
        end
    end
    return unique!(sort!(found))
end

# Whether the forest's field data are split over more than one rank —
# what an operation that still needs messages, and does not have them
# yet, refuses (M7 brings them step by step; "Distributed meshes" in
# CODE.md).
isdistributed(forest::Forest) = commsize(forest.comm) > 1

function refuse_distributed(forest::Forest, what::AbstractString, why::AbstractString)
    isdistributed(forest) || return nothing
    throw(ArgumentError(
        "$what over a forest distributed over $(commsize(forest.comm)) ranks is not " *
        "implemented yet: $why. It arrives with M7 (see \"Distributed meshes\" in " *
        "CODE.md); until then a distributed forest supports the partition, the " *
        "field set storage, the geometry and the local schedule build."))
end

"""
    maxlevel(forest::Forest)

The level of the finest leaf currently in `forest`.
"""
maxlevel(forest::Forest) = maximum(level, forest.leaves)

"""
    root_position(roots_or_forest, root::Integer)

The 0-based `D`-dimensional brick position of root index `root` (as
carried in a [`MortonKey`](@ref)); the inverse of [`root_index`](@ref).
"""
function root_position(roots::NTuple{D,Int}, root::Integer) where {D}
    pos = ntuple(_ -> 0, D)
    r = Int(root)
    for d in 1:D
        pos = Base.setindex(pos, r % roots[d], d)
        r = r ÷ roots[d]
    end
    return pos
end
root_position(forest::Forest, root::Integer) = root_position(forest.roots, root)

"""
    root_index(roots_or_forest, pos::NTuple)

The linearized 0-based root index of 0-based brick position `pos`; the
inverse of [`root_position`](@ref).
"""
function root_index(roots::NTuple{D,Int}, pos::NTuple{D,<:Integer}) where {D}
    idx = 0
    stride = 1
    for d in 1:D
        idx += Int(pos[d]) * stride
        stride *= roots[d]
    end
    return idx
end
root_index(forest::Forest{D}, pos::NTuple{D,<:Integer}) where {D} = root_index(forest.roots, pos)

"""
    alldirections(::Val{D})

All `3^D - 1` nonzero direction vectors `δ ∈ {-1,0,1}^D`. One nonzero
component names a face, two an edge, and `D` a corner — the ghost
regions that surround a block.
"""
function alldirections(::Val{D}) where {D}
    origin = ntuple(_ -> 0, D)
    return filter(!=(origin), vec(collect(Iterators.product(ntuple(_ -> (-1, 0, 1), D)...))))
end

"""
    find_leaf(forest, key) -> Union{Int,Nothing}
    find_leaf(forest, root, level, coords) -> Union{Int,Nothing}

The index into `forest.leaves` of the given leaf, or `nothing` if that
node is not currently a leaf (because it is refined, or coarsened away,
or outside the domain). `O(log nleaves)`.
"""
function find_leaf(forest::Forest{D}, key::MortonKey{D}) where {D}
    i = searchsortedfirst(forest.leaves, key)
    return (i <= length(forest.leaves) && forest.leaves[i] == key) ? i : nothing
end
find_leaf(forest::Forest{D}, root::Integer, lvl::Integer, coords::NTuple{D,<:Integer}) where {D} =
    find_leaf(forest, MortonKey{D}(root, lvl, coords))

"""
    isleaf(forest, key)

Whether `key` is currently a leaf of `forest`.
"""
isleaf(forest::Forest{D}, key::MortonKey{D}) where {D} = find_leaf(forest, key) !== nothing

# The same-level anchor node (root, coords) reached by stepping one cell
# from `k` in direction `δ`, crossing root boundaries and wrapping
# periodic ones. `nothing` when the step leaves a non-periodic boundary.
# Since |δ[d]| <= 1, a step crosses at most one root boundary.
function neighbor_anchor(forest::Forest{D}, k::MortonKey{D}, δ::NTuple{D,Int}) where {D}
    n = 1 << level(k)
    rootpos = root_position(forest, k.root)
    # Step, carrying into the root brick where the step leaves the block.
    stepped = ntuple(d -> Int(k.coords[d]) + δ[d], D)
    newcoords = ntuple(d -> mod(stepped[d], n), D)
    newrootpos = ntuple(D) do d
        stepped[d] < 0 ? rootpos[d] - 1 : stepped[d] >= n ? rootpos[d] + 1 : rootpos[d]
    end
    # Wrap periodic dimensions; a step off a non-periodic face leaves the
    # domain, and there is no neighbor there.
    for d in 1:D
        outside = newrootpos[d] < 0 || newrootpos[d] >= forest.roots[d]
        outside && !forest.periodic[d] && return nothing
    end
    wrapped = ntuple(d -> mod(newrootpos[d], forest.roots[d]), D)
    return (root_index(forest, wrapped), newcoords)
end

# Whether the forest has any reflecting face at all — what decides
# whether a field set over it must declare a parity.
hasreflecting(forest::Forest) = any(r -> r[1] || r[2], forest.reflecting)

# Split a ghost direction of block `k` at the reflecting faces it crosses
# (M10). Returns `(δ′, mask)`: `mask[d]` says that stepping from `k` by
# `δ[d]` leaves the domain through a reflecting face, and `δ′` is `δ`
# with those components zeroed — the direction whose region, mirrored
# along the masked dimensions, is the ghost region `δ`. So the mirrored
# source of region `δ` is the block itself when `δ′` is zero and
# whatever lies in direction `δ′` otherwise.
#
# Lives here rather than in the schedule because it is brick knowledge:
# which faces are the domain's, and which of them reflect.
function reflect_direction(forest::Forest{D}, k::MortonKey{D},
                           δ::NTuple{D,Int}) where {D}
    n = 1 << level(k)
    rootpos = root_position(forest, k.root)
    mask = ntuple(D) do d
        δ[d] == 0 && return false
        forest.periodic[d] && return false
        stepped = Int(k.coords[d]) + δ[d]
        exits = δ[d] < 0 ? (stepped < 0 && rootpos[d] == 0) :
                           (stepped >= n && rootpos[d] == forest.roots[d] - 1)
        return exits && forest.reflecting[d][δ[d] < 0 ? 1 : 2]
    end
    return ntuple(d -> mask[d] ? 0 : δ[d], D), mask
end

# The index of the leaf that covers the node (root, lvl, coords): either
# that node itself, or its nearest coarser ancestor. `nothing` when the
# region is refined *below* `lvl`, so no single leaf covers it.
function find_covering_leaf(forest::Forest{D}, root::Integer, lvl::Integer,
                            coords::NTuple{D,Int}) where {D}
    c = coords
    for l in Int(lvl):-1:0
        i = find_leaf(forest, root, l, c)
        i !== nothing && return i
        c = map(x -> x >> 1, c)
    end
    return nothing
end

# Descend from the refined node (root, lvl, coords) into the children
# that touch its boundary in direction δ — both children along a
# tangential dimension (δ[d] == 0), only the near one along a normal
# dimension — recursing wherever a child is itself refined.
#
# Every node reached is known to exist and to be either a leaf or (by
# the all-or-nothing invariant) fully refined, so the recursion always
# terminates at leaves.
function collect_touching_leaves!(results::Vector{MortonKey{D}}, forest::Forest{D},
                                  root::Integer, lvl::Integer, coords::NTuple{D,Int},
                                  δ::NTuple{D,Int}) where {D}
    childlevel = lvl + 1
    base = ntuple(d -> 2 * coords[d] + (δ[d] == -1 ? 1 : 0), D)
    # Tangential dimensions span both children; normal ones are pinned.
    spans = ntuple(d -> δ[d] == 0 ? (0, 1) : (0,), D)
    for offset in Iterators.product(spans...)
        childcoords = ntuple(d -> base[d] + offset[d], D)
        j = find_leaf(forest, root, childlevel, childcoords)
        if j === nothing
            collect_touching_leaves!(results, forest, root, childlevel, childcoords, δ)
        else
            push!(results, forest.leaves[j])
        end
    end
    return results
end

"""
    neighbor_keys(forest, k::MortonKey, δ::NTuple) -> Vector{MortonKey}

The leaves abutting `k` across the face, edge, or corner in direction
`δ` (see [`alldirections`](@ref)) — that is, the leaves that supply the
data for that ghost region of `k`.

Valid on any forest, balanced or not. The result is

- empty, at a non-periodic domain boundary;
- one key, when the neighbor is at `k`'s level or coarser;
- otherwise every leaf touching the shared region, at whatever depth.

Under 2:1 balance (see [`balance!`](@ref)) the last case is exactly
`2^(D - count(!=(0), δ))` keys, all one level finer.

Note that this is not symmetric under `δ -> -δ` when levels differ: a
coarse neighbor found across `k`'s *corner* also spans the face beyond
it, and reversing the direction from that larger block points elsewhere.
Adjacency is still mutually discoverable, just not necessarily across
the opposite direction.
"""
function neighbor_keys(forest::Forest{D}, k::MortonKey{D}, δ::NTuple{D,Int}) where {D}
    δ != ntuple(_ -> 0, D) || throw(ArgumentError("direction must be nonzero"))
    all(d -> -1 <= δ[d] <= 1, 1:D) ||
        throw(ArgumentError("direction components must be in -1:1, got $δ"))
    anchor = neighbor_anchor(forest, k, δ)
    anchor === nothing && return MortonKey{D}[]
    nbroot, nbcoords = anchor

    i = find_covering_leaf(forest, nbroot, level(k), nbcoords)
    i !== nothing && return [forest.leaves[i]]

    return collect_touching_leaves!(MortonKey{D}[], forest, nbroot, level(k), nbcoords, δ)
end

# Rebuild `forest.leaves` by walking it in order and replacing selected
# entries. Because a node's descendants occupy a contiguous run of the
# curve, splicing sorted children in place of their parent (or a parent
# in place of its run of children) preserves sortedness — no re-sort.
function rebuild_leaves!(forest::Forest{D}, newleaves::Vector{MortonKey{D}}) where {D}
    empty!(forest.leaves)
    append!(forest.leaves, newleaves)
    forest.generation[] += 1
    return forest
end

"""
    refine!(forest, keys)

Replace each leaf in `keys` — a single [`MortonKey`](@ref) or a
collection of them — by its `2^D` children. Errors if any key is not
currently a leaf, or is already at [`MAX_LEVEL`](@ref).

Does not restore 2:1 balance; follow with [`balance!`](@ref).
"""
function refine!(forest::Forest{D}, keys) where {D}
    ks = keys isa MortonKey{D} ? (keys,) : keys
    targets = Set{MortonKey{D}}()
    for k in ks
        isleaf(forest, k) || throw(ArgumentError("cannot refine $k: not a leaf of the forest"))
        level(k) < MAX_LEVEL || throw(ArgumentError("cannot refine $k: already at MAX_LEVEL"))
        push!(targets, k)
    end
    isempty(targets) && return forest

    newleaves = Vector{MortonKey{D}}()
    sizehint!(newleaves, length(forest.leaves) + length(targets) * (2^D - 1))
    for k in forest.leaves
        if k in targets
            append!(newleaves, sortedchildkeys(k))
        else
            push!(newleaves, k)
        end
    end
    return rebuild_leaves!(forest, newleaves)
end

"""
    coarsen!(forest, keys)

Replace the `2^D` children of each key in `keys` by that key itself.
Errors unless every child of every key is currently a leaf — coarsening
is all-or-nothing, matching refinement.

Does not restore 2:1 balance; follow with [`balance!`](@ref).
"""
function coarsen!(forest::Forest{D}, keys) where {D}
    ps = keys isa MortonKey{D} ? (keys,) : keys
    parentof = Dict{MortonKey{D},MortonKey{D}}()
    for p in ps
        for c in childkeys(p)
            isleaf(forest, c) ||
                throw(ArgumentError("cannot coarsen $p: child $c is not a leaf"))
            parentof[c] = p
        end
    end
    isempty(parentof) && return forest

    newleaves = Vector{MortonKey{D}}()
    sizehint!(newleaves, length(forest.leaves))
    emitted = Set{MortonKey{D}}()
    for k in forest.leaves
        p = get(parentof, k, nothing)
        if p === nothing
            push!(newleaves, k)
        elseif !(p in emitted)
            push!(newleaves, p)
            push!(emitted, p)
        end
    end
    return rebuild_leaves!(forest, newleaves)
end

"""
    balance!(forest)

Enforce 2:1 balance across every face, edge, and corner: refine any leaf
that is more than one level coarser than a leaf it touches, repeating
until the refinement stops rippling outward.

Afterwards every ghost region of every block touches at most one level
up or down, which is what bounds the ghost-filling cases and the
prolongation stencils.
"""
function balance!(forest::Forest{D}) where {D}
    dirs = alldirections(Val(D))
    while true
        # The scan is threaded over leaves, each task collecting into its
        # own buffer; the buffers are concatenated in leaf order (M5).
        found = threaded_collect(MortonKey{D}, nleaves(forest)) do hits, b
            k = forest.leaves[b]
            # Only a leaf at level >= 2 can have a neighbor two or more
            # levels coarser than itself.
            level(k) >= 2 || return
            for δ in dirs
                anchor = neighbor_anchor(forest, k, δ)
                anchor === nothing && continue
                nbroot, nbcoords = anchor
                # Looking only for a *coarser* neighbor, so the covering
                # leaf suffices — no need to descend into finer ones,
                # which get checked from their own side.
                i = find_covering_leaf(forest, nbroot, level(k), nbcoords)
                i === nothing && continue
                nb = forest.leaves[i]
                level(nb) < level(k) - 1 && push!(hits, nb)
            end
        end

        isempty(found) && break
        toorefined = Set{MortonKey{D}}(found)
        refine!(forest, toorefined)
    end
    return forest
end

"""
    isbalanced(forest)

Whether `forest` satisfies 2:1 balance across all faces, edges, and
corners — the postcondition of [`balance!`](@ref).
"""
function isbalanced(forest::Forest{D}) where {D}
    dirs = alldirections(Val(D))
    ok = fill(true, nleaves(forest))
    threaded_foreach(nleaves(forest)) do b
        k = forest.leaves[b]
        for δ in dirs, nb in neighbor_keys(forest, k, δ)
            if abs(level(nb) - level(k)) > 1
                ok[b] = false
                return
            end
        end
    end
    return all(ok)
end
