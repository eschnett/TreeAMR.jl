"""
    Forest{D}

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
- `N` is the per-block interior size and `G` the ghost width, subject to
  the invariants in `CODE.md`: `N` even and `N ≥ 2G`.
- Blocks are cubes, so `extents` must match the aspect ratio of `roots`.

Root indices are linearized 0-based, dimension 1 fastest, over `roots`;
see [`root_position`](@ref) and [`root_index`](@ref).

    Forest(roots; N, G, periodic=all false, extents=one unit per root)

# Examples

```jldoctest
julia> forest = Forest((2, 2); N = 8, G = 2, periodic = (true, true));

julia> nleaves(forest)
4
```
"""
struct Forest{D}
    roots::NTuple{D,Int}
    periodic::NTuple{D,Bool}
    extents::NTuple{D,Tuple{Float64,Float64}}
    N::Int
    G::Int
    leaves::Vector{MortonKey{D}}
end

function Forest(roots::NTuple{D,Integer};
                N::Integer,
                G::Integer,
                periodic::NTuple{D,Bool}=ntuple(_ -> false, D),
                extents::NTuple{D,Tuple{Real,Real}}=ntuple(d -> (0.0, Float64(roots[d])), D)) where {D}
    all(>(0), roots) || throw(ArgumentError("roots must all be positive, got $roots"))
    N > 0 || throw(ArgumentError("N must be positive, got $N"))
    G >= 0 || throw(ArgumentError("G must be nonnegative, got $G"))
    iseven(N) || throw(ArgumentError("N must be even, got $N"))
    N >= 2G || throw(ArgumentError("N must be >= 2G, got N=$N, G=$G"))

    ext = ntuple(d -> (Float64(extents[d][1]), Float64(extents[d][2])), D)
    all(d -> ext[d][2] > ext[d][1], 1:D) ||
        throw(ArgumentError("each extent must be nonempty and increasing, got $ext"))
    # Blocks are cubes, so the root spacing must be the same in every
    # dimension.
    h = ntuple(d -> (ext[d][2] - ext[d][1]) / roots[d], D)
    all(d -> isapprox(h[d], h[1]; rtol=1e-12), 1:D) ||
        throw(ArgumentError("blocks must be cubes: extents $ext over roots $roots give " *
                            "anisotropic root spacings $h"))

    rootsI = map(Int, roots)
    leaves = [MortonKey{D}(r, 0, ntuple(_ -> 0, D)) for r in 0:(prod(rootsI) - 1)]
    sort!(leaves)
    return Forest{D}(rootsI, periodic, ext, Int(N), Int(G), leaves)
end

"""
    nleaves(forest::Forest)

The number of leaves currently tiling `forest`; equivalently the number
of blocks a [`FieldSet`](@ref) over it stores.
"""
nleaves(forest::Forest) = length(forest.leaves)

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
        toorefined = Set{MortonKey{D}}()
        for k in forest.leaves
            # Only a leaf at level >= 2 can have a neighbor two or more
            # levels coarser than itself.
            level(k) >= 2 || continue
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
                level(nb) < level(k) - 1 && push!(toorefined, nb)
            end
        end
        isempty(toorefined) && break
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
    for k in forest.leaves
        for δ in dirs
            for nb in neighbor_keys(forest, k, δ)
                abs(level(nb) - level(k)) <= 1 || return false
            end
        end
    end
    return true
end
