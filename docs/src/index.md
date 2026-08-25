# TreeAMR.jl

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

See the [design document](https://github.com/eschnett/TreeAMR.jl/blob/main/CODE.md)
for the full design and the milestone roadmap.

The package is at milestone **M2**: the tree core (Morton keys over a
brick of octree roots, neighbor finding, refinement and coarsening, 2:1
balance, periodic wraparound, block storage) plus the cached ghost
exchange and the default interpolation operators. The wave equation and
`OrdinaryDiffEq` coupling arrive in M3.

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

## Ghost exchange

Ghost filling runs at every RHS evaluation, while neighbor finding is
only needed when the tree changes. So the exchange is split in two: a
[`GhostSchedule`](@ref) is built once and [`fill_ghosts!`](@ref) merely
replays it, with no tree query in the per-evaluation path.

```jldoctest overview
julia> schedule = GhostSchedule(forest, Operators(prolongation=2, restriction=2));

julia> state = FieldSet(forest, 2);

julia> fill_by_coordinates!((x, v) -> v * x[1], state);

julia> fill_ghosts!(state, schedule);
```

Under 2:1 balance there are only three cases — a same-level copy, a
restriction from finer neighbors, and a prolongation from a coarser one
— and all three are tensor products of one-dimensional stencils, so a
single KernelAbstractions kernel serves them all.

The schedule is tied to the tree it was built from, and says so:

```jldoctest overview
julia> isstale(schedule)
false

julia> refine!(forest, last(forest.leaves));

julia> isstale(schedule)
true
```

Interpolation order is configurable via [`Operators`](@ref), and is
constrained by the block geometry: order `p` prolongation needs
`G ≥ p/2`, which [`check_operators`](@ref) enforces when the schedule is
built.

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
generation
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

## Ghost exchange and operators

```@docs
Operators
check_operators
GhostSchedule
isstale
fill_ghosts!
boundary_by_coordinates
```

## Internals

Not exported, and not part of the public interface, but documented
because they define the shape of the schedule.

```@docs
TreeAMR.Stencil1D
TreeAMR.TransferGroup
TreeAMR.BoundaryRegion
TreeAMR.lagrange_weights
```

## Index

```@index
```
