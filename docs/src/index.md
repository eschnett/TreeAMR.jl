# TreeAMR.jl

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

See the [design document](https://github.com/eschnett/TreeAMR.jl/blob/main/CODE.md)
for the full design and the milestone roadmap.

The package is at milestone **M1** (tree core): Morton keys over a brick
of octree roots, the sorted leaf array, neighbor finding, refinement and
coarsening, 2:1 balance, periodic wraparound, and block storage. Ghost
exchange and the interpolation operators arrive in M2.

## Overview

The domain is a brick of `M₁ × … × M_D` octree roots. Refinement is
all-or-nothing, and only leaves carry data, so the leaves tile the
domain exactly:

```jldoctest overview
julia> using TreeAMR

julia> forest = Forest((2, 2); N = 8, G = 2, periodic = (true, true));

julia> nleaves(forest)
4

julia> refine!(forest, first(forest.leaves));

julia> nleaves(forest)
7
```

The tree is a *linear* octree: a sorted flat vector of [`MortonKey`](@ref)s,
with no pointers. Neighbor finding is key arithmetic plus binary search,
and periodic wraparound lives in that arithmetic rather than in special
boundary code:

```jldoctest overview
julia> k = first(forest.leaves);

julia> neighbor_keys(forest, k, (1, 0))
1-element Vector{MortonKey{2}}:
 MortonKey{2}(root=0, level=1, coords=(1, 0))
```

Enforcing [`balance!`](@ref) guarantees that every ghost region touches
at most one level up or down, which is what bounds the ghost-filling
cases in M2.

Data lives in a [`FieldSet`](@ref): one big array over all leaf blocks,
cell indices fastest, ghosts included.

## Module

```@docs
TreeAMR
```

## Tree core

```@docs
MortonKey
Base.isless(::MortonKey{D}, ::MortonKey{D}) where {D}
MAX_LEVEL
level
parentkey
childkeys
sortedchildkeys
isancestor
```

## Forest

```@docs
Forest
nleaves
maxlevel
root_position
root_index
alldirections
find_leaf
isleaf
neighbor_keys
refine!
coarsen!
balance!
isbalanced
```

## Geometry

```@docs
root_spacing
spacing
minimum_spacing
block_origin
block_extent
cell_center
```

## Storage

```@docs
FieldSet
nblocks
blockkey
blockview
interiorview
fill_by_coordinates!
```

## Index

```@index
```
