# The mutable part of a forest. `pool` holds the message buffers of the
# exchanges over the forest between the stages that use them: a regrid
# builds new stages, and they take their buffers, and host mirrors, from
# it instead of allocating them again (`BufferPool`). It is made on first
# use, so a serial forest, whose stages have no messages, never has one.
mutable struct ForestState
    generation::Int
    pool::Union{Nothing,BufferPool}
end
ForestState() = ForestState(0, nothing)

function bufferpool(forest)
    state = forest.state
    state.pool === nothing && (state.pool = BufferPool())
    return state.pool::BufferPool
end

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
- `rotating = (d1, d2)` declares a **rotating seam** (M12): only one
  quadrant of the `(d1, d2)` plane is simulated, and the other three are
  its images under quarter turns about the line where the low faces of
  `d1` and `d2` meet. The rotation `R` takes `e_{d1}` to `e_{d2}` and
  `e_{d2}` to `-e_{d1}`, so the order of the pair fixes its sense. Unlike
  a reflecting face, the tree sees the seam: the low face of `d1` is
  glued to the low face of `d2`, [`neighbor_keys`](@ref) finds the real
  leaves across it, and [`balance!`](@ref) keeps the leaves on either
  side of it at one level (*conformity*). The two dimensions must have
  as many roots as each other and be neither periodic nor reflecting;
  their high faces stay outer. See "Rotating seams" in `CODE.md`.
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

    Forest(roots; N, periodic=all false, reflecting=all false, rotating=nothing,
           extents=one unit per root, leaves=nothing, comm=nothing)
    Forest{T}(roots; ...)                      # geometry in `T`

Without `leaves` the forest starts as its unrefined roots. With it, it
starts from that leaf list instead — keys in curve order, as
`forest.leaves` holds them — which is copied rather than aliased, and
the new forest is at [`generation`](@ref) 0. That is how a checkpoint
is restored, and how M7's ranks will build their forests. Everything
built over a forest trusts its leaves, so the list is refused unless
every key lies in the brick, the keys strictly increase, they tile the
brick exactly — no gap, no overlap — and they are 2:1 balanced and, on a
rotating forest, conforming at the seam (see [`balance!`](@ref)).

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

julia> quadrant = Forest((2, 2, 1); N = 8, rotating = (1, 2),
                         reflecting = ((false, false), (false, false), (true, false)));
```
"""
struct Forest{D,T}
    roots::NTuple{D,Int}
    periodic::NTuple{D,Bool}
    reflecting::NTuple{D,Tuple{Bool,Bool}}       # (lo, hi) per dimension
    # The rotating seam's pair of dimensions `(d1, d2)` (M12), `(0, 0)` for
    # none; read it through `hasrotating` and `rotating_dims`. `Int8`
    # because the cost of a ninth field was its size, not its count: two
    # bytes fit in the padding before `extents` (for D ≤ 4), so the struct
    # stays as large as it was and so does the schedule build's
    # allocation, where an `NTuple{2,Int}` added 160 and 4896 bytes to
    # `bench/ghosts.jl` (CODE.md, "The buffer pool").
    rotating::NTuple{2,Int8}
    extents::NTuple{D,Tuple{T,T}}
    N::Int
    leaves::Vector{MortonKey{D}}
    # What changes while the fields above stay: the generation, bumped
    # whenever the leaf array changes, so anything derived from the tree
    # (a GhostSchedule, say) can detect in O(1) that it is stale — a
    # same-size refine-then-coarsen would otherwise slip past a
    # leaf-count check and silently transfer the wrong data — and the
    # message-buffer pool (M7). One mutable object for both, in place of
    # the `Ref` the generation had, so that the struct stays as large as
    # it was: a ninth field made the schedule build allocate more
    # (CODE.md, "The buffer pool").
    state::ForestState
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
                   rotating::Union{Nothing,Tuple{Integer,Integer}}=nothing,
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
    rot = check_rotating(rotating, roots, periodic, reflecting)

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
    forest = Forest{D,T}(rootsI, periodic, reflecting, rot, ext, Int(N), list,
                         ForestState(), communicator(comm))
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
                rotating::Union{Nothing,Tuple{Integer,Integer}}=nothing,
                extents::Union{Nothing,NTuple{D,Tuple{Real,Real}}}=nothing,
                leaves::Union{Nothing,AbstractVector{MortonKey{D}}}=nothing,
                comm=nothing, G=nothing) where {D}
    G === nothing || throw(no_forest_ghosts)
    if extents === nothing
        return Forest{Float64}(roots; N=N, periodic=periodic, reflecting=reflecting,
                               rotating=rotating, leaves=leaves, comm=comm)
    end
    T = float(promote_type(ntuple(d -> promote_type(typeof(extents[d][1]),
                                                    typeof(extents[d][2])), D)...))
    return Forest{T}(roots; N=N, periodic=periodic, reflecting=reflecting,
                     rotating=rotating, extents=extents, leaves=leaves, comm=comm)
end

# The `rotating` keyword, checked, as the forest stores it: `(0, 0)` for
# none. Each refusal says why the seam cannot be glued that way (CODE.md,
# "Rotating" under "Domain and boundaries").
function check_rotating(rotating, roots::NTuple{D,Integer}, periodic::NTuple{D,Bool},
                        reflecting::NTuple{D,Tuple{Bool,Bool}}) where {D}
    rotating === nothing && return (Int8(0), Int8(0))
    d1, d2 = Int(rotating[1]), Int(rotating[2])
    D >= 2 || throw(ArgumentError(
        "rotating = $rotating needs at least two dimensions, and this forest has " *
        "$D: the rotation turns the plane of two dimensions about the line where " *
        "their low faces meet, and a $D-dimensional forest has no such plane"))
    (1 <= d1 <= D && 1 <= d2 <= D) || throw(ArgumentError(
        "rotating = $rotating names a dimension outside 1:$D: the pair (d1, d2) is " *
        "the plane of the rotation, two of the forest's own dimensions"))
    d1 != d2 || throw(ArgumentError(
        "rotating = $rotating names dimension $d1 twice: the rotation turns the " *
        "plane of two different dimensions, gluing the low face of the first to the " *
        "low face of the second"))
    roots[d1] == roots[d2] || throw(ArgumentError(
        "rotating = $rotating needs as many roots along dimension $d1 as along $d2, " *
        "got $(roots[d1]) and $(roots[d2]): the seam glues the low face of $d1 onto " *
        "the low face of $d2, root for root, so the two faces must be the same " *
        "length"))
    for d in (d1, d2)
        periodic[d] && throw(ArgumentError(
            "rotating = $rotating, but dimension $d is periodic: its low face is " *
            "the rotating seam, glued to the other dimension's low face, and a " *
            "periodic dimension has no faces — its last block is its first block's " *
            "neighbor. Drop one of the two."))
        any(reflecting[d]) && throw(ArgumentError(
            "rotating = $rotating, but dimension $d has a reflecting face, " *
            "reflecting[$d] = $(reflecting[d]): its low face is the rotating seam, " *
            "and its high face stays outer, the boundary hook's. A reflecting high " *
            "wall together with the rotation is an open question in CODE.md, not " *
            "implemented. Reflect along a dimension outside the plane of the " *
            "rotation instead, as an octant does."))
    end
    return (Int8(d1), Int8(d2))
end

# Whether the forest has a rotating seam (M12).
hasrotating(forest::Forest) = forest.rotating[1] != 0

# The rotating pair `(d1, d2)` as the `rotating` keyword takes it: `nothing`
# when the forest has no seam. What rebuilds a forest from another one's
# parameters passes on.
rotating_dims(forest::Forest) =
    hasrotating(forest) ? (Int(forest.rotating[1]), Int(forest.rotating[2])) : nothing

# M12 is built in steps (CODE.md, the M12 entry under "Milestones"). From
# step 1 on the neighbor search finds the real leaves across a rotating
# seam, but what reads from them learns their orientation only in later
# steps; until then each such reader refuses a rotating forest, rather
# than read across the seam as though it were an ordinary face. It takes
# the forest or its `rotating_dims`: a build's argument checks run in a
# closure (`collective_checks`), and one that captures the forest copies
# it — `GhostSchedule`'s allocated 240 bytes more in `bench/ghosts.jl`.
refuse_rotating(forest::Forest, what::AbstractString) =
    refuse_rotating(rotating_dims(forest), what)
refuse_rotating(::Nothing, ::AbstractString) = nothing
function refuse_rotating(rotating::NTuple{2,Int}, what::AbstractString)
    throw(ArgumentError(
        "$what over a rotating forest (rotating = $rotating) is not " *
        "implemented yet in this step of M12: the forest finds the real leaves " *
        "across the seam, but $what does not yet turn what it reads from them — " *
        "their axes and their variables — into the frame of the block that reads, " *
        "and would deliver wrong data without noticing. See the M12 entry in " *
        "CODE.md for the steps that add it."))
end

# Validate a caller's leaf list (the `leaves` keyword), already copied
# into the candidate `forest`. The storage, the ghost schedule and the
# regrid all trust `forest.leaves` without looking at it again — that is
# what lets them be built once per tree change — so a list read from a
# file, or received from another rank, is checked here instead: every
# key in the brick, strictly increasing in curve order, tiling the brick
# exactly, 2:1 balanced and (M12) conforming at a rotating seam. Each
# refusal names the first leaf at which the list goes wrong.
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
    for (i, k) in enumerate(leaves), δ in dirs
        r, nbrs = oriented_neighbors(forest, k, δ)
        for nb in nbrs
            balance_slack(r, δ) == 0 && level(nb) != level(k) && throw(ArgumentError(
                "the leaves are not conforming at the rotating seam: leaf $i, $k, " *
                "touches $nb across the seam's face in direction $δ, and their levels " *
                "differ. On a rotating forest a leaf on the low face of one dimension " *
                "of the plane and its image on the low face of the other are at one " *
                "level, so that no coarse-fine face crosses the seam (CODE.md, " *
                "\"Rotating seams\"); balance! keeps a forest so, so this list was " *
                "damaged, made by hand, or written for a forest without the seam"))
            abs(level(nb) - level(k)) > 1 || continue
            throw(ArgumentError(
                "the leaves are not 2:1 balanced: leaf $i, $k, touches $nb, and their " *
                "levels differ by more than one. The list is refused rather than " *
                "rebalanced, because a balanced forest can only ever produce a " *
                "balanced list, so this one was damaged or made by hand; and because " *
                "the ghost schedule assumes the balance, and would silently build " *
                "wrong prolongations from an unbalanced mesh"))
        end
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
generation(forest::Forest) = forest.state.generation

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

# Whether the forest's field data are split over more than one rank:
# what takes the distributed path of an operation that has one.
isdistributed(forest::Forest) = commsize(forest.comm) > 1

# --- The forest digest (M7) ----------------------------------------------
#
# Every forest mutation is collective, and every host pass is a
# deterministic function of its inputs, so the ranks' forests agree
# without a message — as long as the application kept the contract.
# What would go wrong *silently* on a forest that diverged is checked:
# a schedule build gathers every rank's digest and refuses, on every
# rank together, if any differs ("Every forest mutation is collective"
# in CODE.md). The digest is the generation, the leaf count, a fold of
# every leaf's hash — not `hash(forest.leaves)`, which for a long
# vector samples only some of the elements — and the brick, the
# rotating seam (M12) included. Beside it
# goes a hash of the layout the build is for, since a rank that built
# its stencils from other operators would deliver wrong ghosts as
# silently, and a flag saying whether this rank's own argument checks
# refused. Every hash here is of integers and strings, never of a
# `Symbol` or an object identity, which differ between processes.
struct ForestDigest
    generation::Int
    nleaves::Int
    leaves::UInt
    brick::UInt
    layout::UInt
    refused::Bool
end

function ForestDigest(forest::Forest, layout::UInt, refused::Bool)
    h = hash(nleaves(forest))
    for k in forest.leaves
        h = hash(k, h)
    end
    brick = hash(string((forest.roots, forest.N, forest.periodic, forest.reflecting,
                         Int.(forest.rotating), forest.extents)))
    return ForestDigest(generation(forest), nleaves(forest), h, brick, layout, refused)
end

sameforest(a::ForestDigest, b::ForestDigest) =
    (a.generation, a.nleaves, a.leaves, a.brick) == (b.generation, b.nleaves, b.leaves,
                                                     b.brick)

# A hash of the values a build's layout is made of, for the digest.
layouthash(values...) = hash(string(values))

# Agree, across the ranks, that `what` may go ahead: one `allgather` of
# the digest, then the same verdict on every rank. `refusal` is this
# rank's own argument error, if its checks refused; a rank that refused
# throws its own, and every other rank says which ranks refused, so no
# rank goes on to wait in an exchange the others never enter. Serially
# nothing is gathered and the refusal, if any, is thrown as it is.
function agree_on_forest(forest::Forest, what::AbstractString;
                         layout::UInt=UInt(0), refusal=nothing)
    comm = forest.comm
    if commsize(comm) == 1
        refusal === nothing || throw(refusal)
        return nothing
    end
    digests = allgather(comm, ForestDigest(forest, layout, refusal !== nothing))
    return digest_verdict(digests, what, commrank(comm), refusal)
end

# The verdict on the gathered digests, apart from the gathering so that
# it can be tested in one process.
function digest_verdict(digests::Vector{ForestDigest}, what::AbstractString,
                        rank::Integer, refusal=nothing)
    nranks = length(digests)
    refused = [r - 1 for r in 1:nranks if digests[r].refused]
    if !isempty(refused)
        refusal === nothing || throw(refusal)
        throw(ArgumentError(
            "$what was refused on rank(s) $(join(refused, ", ")) of $nranks, and so " *
            "on this one (rank $rank) too: the call is collective, and a rank that " *
            "went on would wait for the others in its first exchange. The reason is " *
            "in the error on rank $(first(refused)); the arguments evidently differ " *
            "between ranks, which they must not."))
    end
    first_ = digests[1]
    diverged = [r - 1 for r in 2:nranks if !sameforest(digests[r], first_)]
    if !isempty(diverged)
        describe(r) = (d = digests[r + 1];
                       "rank $r has generation $(d.generation) and $(d.nleaves) leaves")
        throw(ArgumentError(
            "the forest differs between ranks, so $what is refused on every rank: " *
            "$(describe(0)), but rank(s) $(join(diverged, ", ")) hold a different " *
            "one ($(join(describe.(diverged), "; "))). Every forest mutation — the " *
            "constructors, refine!, coarsen!, balance! and regrid! — is collective: " *
            "the same call with the same arguments on every rank. Over forests that " *
            "differ, the ranks would exchange the wrong data without noticing."))
    end
    mismatched = [r - 1 for r in 2:nranks if digests[r].layout != first_.layout]
    isempty(mismatched) || throw(ArgumentError(
        "$what was called for a different layout on rank(s) " *
        "$(join(mismatched, ", ")) than on rank 0, so it is refused on every rank: " *
        "the ghost widths, the centering, the operators and the element type must " *
        "be the same everywhere (for regrid!, so must the field sets passed, their " *
        "variable counts, `buffer` and `transfer`; for interpolate, the basis, " *
        "`derivs`, `vars` and `exclude`; for save_checkpoint and load_checkpoint, " *
        "the path and every keyword but `data`, and for write_plain the item's " *
        "name), since a rank computes the data it sends with its own stencils and " *
        "lays out what it receives by its own, and the ranks of a checkpoint send " *
        "and receive its blocks by the layout each derives from its arguments."))
    return nothing
end

# A collective build's argument checks: run `check`, which returns the
# checked values and their `layouthash`, and agree on the forest and
# the layout across the ranks before going on. A refusal on some ranks
# only is raised on all of them (see `agree_on_forest`). A refusal is an
# `ArgumentError`, or the `DimensionMismatch` `regrid!` raises for a flag
# vector of the wrong length (step 4 of M7); anything else is a bug and
# is rethrown at once.
function collective_checks(check, forest::Forest, what::AbstractString)
    distributed = isdistributed(forest)
    checked, refusal = try
        check(), nothing
    catch err
        (distributed && err isa Union{ArgumentError,DimensionMismatch}) || rethrow()
        nothing, err
    end
    distributed || return first(checked)
    agree_on_forest(forest, what; layout=checked === nothing ? UInt(0) : last(checked),
                    refusal=refusal)
    return first(checked)
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

# --- The rotating seam (M12) ------------------------------------------------
#
# The arithmetic of the seam, in one place: every other function sees it
# through `neighbor_anchor` and `oriented_neighbors`. A step from a block
# is taken in *global level coordinates* — a node's position at its level
# counted across the whole brick, so that the axis sits at 0 in both
# dimensions of the plane — and a naive neighbor that lands beyond the low
# face of `d1`, of `d2`, or of both, lies in the image `R^r` of real data,
# `r = 1, 3, 2` respectively. It maps back to the real node by `R^{-r}`
# about the axis (CODE.md, "Rotating seams", the first table):
#
#     r = 1:  (g1, g2) ↦ (g2, -1 - g1)
#     r = 3:  (g1, g2) ↦ (-1 - g2, g1)
#     r = 2:  (g1, g2) ↦ (-1 - g1, -1 - g2)
#
# with (g1, g2) the coordinates along (d1, d2) and every other one kept.
# The `-1` is a cell's width: the cell at `g` spans `[g, g + 1)`, and
# `R^{-1}` takes it to the cell spanning `(-g - 1, -g]` along `d2`.

# The orientation of the naive node at global coordinates `g`: 0 off the
# seam, or with no seam at all.
@inline function seam_orientation(forest::Forest{D}, g::NTuple{D,Int}) where {D}
    hasrotating(forest) || return 0
    d1, d2 = Int(forest.rotating[1]), Int(forest.rotating[2])
    a, b = g[d1] < 0, g[d2] < 0
    return a ? (b ? 2 : 1) : (b ? 3 : 0)
end

# The real node of the naive node `g` in orientation `r` (the table above).
@inline function unrotate_node(g::NTuple{D,Int}, r::Int, d1::Int, d2::Int) where {D}
    g1, g2 = g[d1], g[d2]
    r1, r2 = r == 1 ? (g2, -1 - g1) : r == 3 ? (-1 - g2, g1) :
             r == 2 ? (-1 - g1, -1 - g2) : (g1, g2)
    return Base.setindex(Base.setindex(g, r1, d1), r2, d2)
end

# The direction `R^{-r} δ`: a ghost direction `δ` of a block as seen from
# the real leaves across a rotating seam in orientation `r` (M12). It is
# the linear part of the map from naive to real nodes, so `r = 1` takes
# `(δ1, δ2)` along `(d1, d2)` to `(δ2, -δ1)`, `r = 3` to `(-δ2, δ1)` and
# `r = 2` to `(-δ1, -δ2)`; `r = 0` is the identity.
@inline function real_direction(δ::NTuple{D,Int}, r::Integer, d1::Integer,
                                 d2::Integer) where {D}
    a, b = δ[d1], δ[d2]
    r1, r2 = r == 1 ? (b, -a) : r == 3 ? (-b, a) : r == 2 ? (-a, -b) : (a, b)
    return Base.setindex(Base.setindex(δ, r1, d1), r2, d2)
end

# The child offset `o ∈ {0,1}^D` of a real leaf across a rotating seam in
# orientation `r`, taken into the *virtual* frame, where the asking block
# sees that leaf (M12): its offset within its virtual parent, which is
# what a restriction's stencils depend on. The virtual frame is `R^r` of
# the real one, and the parent of a node maps to the parent of its image
# (`-1 - g` halves to `-1 - g ÷ 2`), so the offset turns about its
# parent's centre: `r = 1` takes `(o1, o2)` along `(d1, d2)` to
# `(1 - o2, o1)`, `r = 3` to `(o2, 1 - o1)` and `r = 2` to
# `(1 - o1, 1 - o2)`; `r = 0` is the identity.
@inline function virtual_offset(o::NTuple{D,Int}, r::Integer, d1::Integer,
                                d2::Integer) where {D}
    a, b = o[d1], o[d2]
    v1, v2 = r == 1 ? (1 - b, a) : r == 3 ? (b, 1 - a) :
             r == 2 ? (1 - a, 1 - b) : (a, b)
    return Base.setindex(Base.setindex(o, v1, d1), v2, d2)
end

# The real node one step from `k` in direction `δ`, in global level
# coordinates, and its orientation: the step taken naively, then mapped
# back across a rotating seam it crosses. Periodic dimensions are not yet
# wrapped and a high face not yet tested; `neighbor_anchor` does both.
@inline function stepped_node(forest::Forest{D}, k::MortonKey{D},
                              δ::NTuple{D,Int}) where {D}
    lvl = level(k)
    rootpos = root_position(forest, k.root)
    g = ntuple(d -> (rootpos[d] << lvl) + Int(k.coords[d]) + δ[d], D)
    r = seam_orientation(forest, g)
    r == 0 && return g, 0
    return unrotate_node(g, r, Int(forest.rotating[1]), Int(forest.rotating[2])), r
end

# The same-level anchor node reached by stepping one cell from `k` in
# direction `δ`, as `(root, coords, r)`: crossing root boundaries,
# wrapping periodic ones, and crossing a rotating seam (M12) into the
# real node, with `r` the orientation (0 off the seam). `nothing` when the
# step leaves a non-periodic face that is not the seam, or crosses the
# seam to a node that lies beyond a high face — a region beyond an outer
# face either way, and the hook's. Since |δ[d]| <= 1, a step crosses at
# most one root boundary per dimension. This is the single source of
# truth for where a step lands: the neighbor search, `balance!` and,
# through them, everything else go through it.
function neighbor_anchor(forest::Forest{D}, k::MortonKey{D}, δ::NTuple{D,Int}) where {D}
    lvl = level(k)
    g, r = stepped_node(forest, k, δ)
    for d in 1:D
        outside = g[d] < 0 || g[d] >= forest.roots[d] << lvl
        outside && !forest.periodic[d] && return nothing
    end
    wrapped = ntuple(d -> mod(g[d], forest.roots[d] << lvl), D)
    mask = (1 << lvl) - 1
    rootpos = ntuple(d -> wrapped[d] >> lvl, D)
    return (root_index(forest, rootpos), ntuple(d -> wrapped[d] & mask, D), r)
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
# which faces are the domain's, and which of them reflect. A rotating
# seam (M12) is never masked: the constructor refuses a reflecting face
# in either dimension of its plane, and the tree finds the leaves across
# the seam, so a `δ` through it is ordinary here.
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

Across a rotating seam (M12; see [`Forest`](@ref)) the keys are the real
leaves whose image under the rotation abuts `k`, the leaves that supply
that ghost region turned by a quarter turn or two. A block at the axis
is then its own neighbor in three directions of the plane, as a single
periodic root is its own neighbor.

Note that this is not symmetric under `δ -> -δ` when levels differ: a
coarse neighbor found across `k`'s *corner* also spans the face beyond
it, and reversing the direction from that larger block points elsewhere.
Adjacency is still mutually discoverable, just not necessarily across
the opposite direction.
"""
neighbor_keys(forest::Forest{D}, k::MortonKey{D}, δ::NTuple{D,Int}) where {D} =
    last(oriented_neighbors(forest, k, δ))

# `neighbor_keys` with the orientation (M12): `(r, keys)`, where the
# keys are the real leaves and `R^r` carries them to where `k` sees them
# in direction `δ`. All the neighbors of one block in one direction share
# one `r`, since the seam lies on a root boundary at every level. `r` is
# 0 off the seam; it is the region's orientation even when `keys` is
# empty, which across the seam means the real image lies beyond a high
# face. The search for finer leaves runs in the real frame, in the
# direction `R^{-r} δ` from them back to `k`'s image.
function oriented_neighbors(forest::Forest{D}, k::MortonKey{D},
                            δ::NTuple{D,Int}) where {D}
    δ != ntuple(_ -> 0, D) || throw(ArgumentError("direction must be nonzero"))
    all(d -> -1 <= δ[d] <= 1, 1:D) ||
        throw(ArgumentError("direction components must be in -1:1, got $δ"))
    anchor = neighbor_anchor(forest, k, δ)
    anchor === nothing && return last(stepped_node(forest, k, δ)), MortonKey{D}[]
    nbroot, nbcoords, r = anchor

    i = find_covering_leaf(forest, nbroot, level(k), nbcoords)
    i !== nothing && return r, [forest.leaves[i]]

    δreal = r == 0 ? δ : real_direction(δ, r, forest.rotating[1], forest.rotating[2])
    return r, collect_touching_leaves!(MortonKey{D}[], forest, nbroot, level(k), nbcoords,
                                       δreal)
end

# Rebuild `forest.leaves` by walking it in order and replacing selected
# entries. Because a node's descendants occupy a contiguous run of the
# curve, splicing sorted children in place of their parent (or a parent
# in place of its run of children) preserves sortedness — no re-sort.
function rebuild_leaves!(forest::Forest{D}, newleaves::Vector{MortonKey{D}}) where {D}
    empty!(forest.leaves)
    append!(forest.leaves, newleaves)
    forest.state.generation += 1
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
until the refinement stops rippling outward. On a forest with a rotating
seam (M12) it also enforces **conformity** there: a leaf across a seam
*face* that is coarser by any amount is refined, so the leaves on the low
face of `d1` and their images on the low face of `d2` end at one level.

Afterwards every ghost region of every block touches at most one level
up or down, which is what bounds the ghost-filling cases and the
prolongation stencils.
"""
function balance!(forest::Forest{D}) where {D}
    dirs = alldirections(Val(D))
    seam = hasrotating(forest)
    while true
        # The scan is threaded over leaves, each task collecting into its
        # own buffer; the buffers are concatenated in leaf order (M5).
        found = threaded_collect(MortonKey{D}, nleaves(forest)) do hits, b
            k = forest.leaves[b]
            # Only a leaf at level >= 2 can have a neighbor two or more
            # levels coarser than itself — but across a rotating seam's
            # face one level coarser is too coarse already, so there a
            # leaf at level 1 is asked too.
            level(k) >= 2 || (level(k) == 1 && seam) || return
            for δ in dirs
                anchor = neighbor_anchor(forest, k, δ)
                anchor === nothing && continue
                nbroot, nbcoords, r = anchor
                # Looking only for a *coarser* neighbor, so the covering
                # leaf suffices — no need to descend into finer ones,
                # which get checked from their own side.
                i = find_covering_leaf(forest, nbroot, level(k), nbcoords)
                i === nothing && continue
                nb = forest.leaves[i]
                level(nb) < level(k) - balance_slack(r, δ) && push!(hits, nb)
            end
        end

        isempty(found) && break
        toorefined = Set{MortonKey{D}}(found)
        refine!(forest, toorefined)
    end
    return forest
end

# How many levels two adjacent leaves may differ by: one, but none across
# a rotating seam's *face* (M12, conformity; CODE.md "Rotating seams").
# An edge or corner across the seam keeps the ordinary 2:1 rule. A seam
# face is a direction with one nonzero component in orientation 1 or 3;
# orientation 2 needs two.
@inline balance_slack(r::Int, δ::NTuple{D,Int}) where {D} =
    r != 0 && count(!=(0), δ) == 1 ? 0 : 1

"""
    isbalanced(forest)

Whether `forest` satisfies 2:1 balance across all faces, edges, and
corners, and conformity at a rotating seam (M12) — the postcondition of
[`balance!`](@ref).
"""
function isbalanced(forest::Forest{D}) where {D}
    dirs = alldirections(Val(D))
    ok = fill(true, nleaves(forest))
    threaded_foreach(nleaves(forest)) do b
        k = forest.leaves[b]
        for δ in dirs
            r, nbrs = oriented_neighbors(forest, k, δ)
            for nb in nbrs
                if abs(level(nb) - level(k)) > balance_slack(r, δ)
                    ok[b] = false
                    return
                end
            end
        end
    end
    return all(ok)
end
