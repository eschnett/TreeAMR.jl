# TreeAMR.jl

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

See the [design document](https://github.com/eschnett/TreeAMR.jl/blob/main/CODE.md)
for the full design and the milestone roadmap.

The package is at milestone **M4**: the tree core (Morton keys over a
brick of octree roots, neighbor finding, refinement and coarsening, 2:1
balance, periodic wraparound, block storage), the cached ghost exchange
with configurable interpolation operators, the state-vector coupling
that lets a standard ODE integrator drive the whole hierarchy, and
adaptive regridding. Multi-threading arrives in M5.

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
built. It is also constrained by your discretization — see the warning
in [`Operators`](@ref), which is worth reading before picking an order.

## Time integration

The whole hierarchy advances with one global `dt`, and the state is a
single flat vector holding leaf **interiors only**, so a standard
integrator drives it unmodified. Ghosts live in the working array, which
is scratch, refreshed at every evaluation.

The application writes its own `f!`, calling the three steps explicitly
rather than through a `semidiscretize`-style wrapper:

```julia
function rhs!(du, u, p, t)
    scatter!(p.fs, u)                    # flat vector -> working array
    fill_ghosts!(p.fs, p.schedule)       # copies, restrictions, prolongations
    map_blocks!(my_kernel!, p.fs, statearray(du, p.fs), p.fs.work, ...)
end

dt = cfl * minimum_spacing(forest)       # global step, set by the finest level
solve(ODEProblem(rhs!, u, tspan, p), RK4(); dt = dt, adaptive = false)
```

Because the integrator owns the stages, ghosts are refilled at *every*
evaluation; the classic wide-ghost optimization is unavailable by
construction, which `CODE.md` accepts.

Errors and tolerances on an adaptive mesh want
[`volume_weighted_norm`](@ref): refined regions contribute more entries
per unit volume, so an unweighted norm silently emphasizes them.

## Regridding

The application flags blocks; [`regrid!`](@ref) completes the marks to
preserve 2:1 balance, rebuilds the key list, and moves the data —
surviving blocks copied, refined blocks prolongated from their parent,
coarsened blocks restricted from their children.

```julia
flags = flag_blocks((b, key) -> needs_refining(fs, b) ? Refine : Keep, forest)
if regrid!(forest, fs, schedule; flags = flags)
    schedule = GhostSchedule(forest, operators)   # the old one is now stale
    u = statevector(fs); gather!(u, fs)           # and u changed length
    # ... then reinit! the integrator
end
```

Regridding changes both the size and the meaning of the state vector, so
in practice it means stop → rebuild → `reinit!` for anything beyond a
one-step method. Block indices are not stable across a regrid: slots are
compacted, and `forest.leaves[b]` is the only way to say which block is
which.

Building initial data iterates the same machinery, *re-evaluating* the
data on each new mesh rather than interpolating it — otherwise a newly
refined block would only ever carry the coarse mesh's resolution:

```julia
schedule, passes, converged = adapt_to_initial_data!(fs, operators;
                                                     initial = f, flag = flag)
```

Whether the transfer conserves [`total_mass`](@ref) depends on which
operator family the field set uses. With [`Conservative`](@ref OperatorFamily)
operators it is exact for arbitrary data; with [`PointValue`](@ref OperatorFamily) ones
only for fields the operators reproduce exactly, since prolongation is
not locally conservative. Coarsening alone conserves either way.

```julia
ops = Operators(prolongation = 3, restriction = 2, family = Conservative)
```

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
block_spacings
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
OperatorFamily
check_operators
GhostSchedule
isstale
fill_ghosts!
boundary_by_coordinates
```

## ODE coupling

```@docs
statelength
statevector
statearray
scatter!
gather!
map_blocks!
volume_weighted_norm
```

## Regridding

```@docs
RegridFlag
flag_blocks
complete_marks
regrid!
adapt_to_initial_data!
total_mass
```

## Internals

Not exported, and not part of the public interface, but documented
because they define the shape of the schedule.

```@docs
TreeAMR.Stencil1D
TreeAMR.TransferGroup
TreeAMR.BoundaryRegion
TreeAMR.lagrange_weights
TreeAMR.ghost_layers_read
```

## Index

```@index
```
