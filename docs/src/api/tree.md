# Tree and geometry

The linear octree — keys, the forest of leaves, refinement and balance —
and the map from a key to physical space.

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

## Forests

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
generation
floattype
```

## Geometry

```@docs
root_spacing
spacing
minimum_spacing
block_origin
block_extent
block_spacings
block_origins
```
