# TreeAMR.jl

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

See the [design document](https://github.com/eschnett/TreeAMR.jl/blob/main/CODE.md)
for the full design and the milestone roadmap.

The package is at milestone **M8**: the tree core (Morton keys over a
brick of octree roots, neighbor finding, refinement and coarsening, 2:1
balance, periodic wraparound, block storage), the cached ghost exchange
with configurable interpolation operators, the state-vector coupling
that lets a standard ODE integrator drive the whole hierarchy, adaptive
regridding, multi-threading throughout, and GPU support: the storage,
the exchange schedule and every kernel follow a KernelAbstractions
backend of the caller's choosing.

M8 added the layout and the conservation. The ghost width `G` is a
[`FieldSet`](@ref) keyword, one per dimension, rather than a
[`Forest`](@ref) one, and a field set also carries a **centering** —
[`cellcentered`](@ref), [`vertexcentered`](@ref), [`facecentered`](@ref)
or [`edgecentered`](@ref) — so a [`GhostSchedule`](@ref) belongs to a
*layout* rather than to a forest. [`InterfaceSchedule`](@ref) and
[`restrict_interfaces!`](@ref) then make a finite-volume scheme
conservative across coarse-fine faces, which under one global `dt` needs
nothing but a spatial flux fixup within each right-hand side.

Next is MPI (M7), so that the distributed exchange is built once over a
layout-generic schedule.

## Overview

The domain is a brick of `M₁ × … × M_D` octree roots. Refinement is
all-or-nothing, and only leaves carry data, so the leaves tile the
domain exactly:

```jldoctest overview
julia> using TreeAMR

julia> forest = Forest((2, 2); N = 8, periodic = (true, true));

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
cell indices fastest, ghosts included. The ghost width `G` belongs to the
field set rather than to the forest, per dimension: it says how far a
stencil reaches into a neighbor's data, which is a property of what is
stored. An evolved state with `G = 2` and the fluxes computed from it
with `G = 0`, over one forest, is the normal case.

```jldoctest overview
julia> state = FieldSet(forest, 2; G = 2);

julia> size(state.work)
(12, 12, 2, 7)
```

## Centerings

A field set also carries a **centering**: per dimension, its values sit
either at cell centers (`:cell`) or at cell boundaries (`:vertex`). The
familiar names are spellings of that tuple — [`cellcentered`](@ref),
[`vertexcentered`](@ref), [`facecentered`](@ref), [`edgecentered`](@ref)
— because every transfer is a product of `D` one-dimensional stencils,
and the stencil for dimension `d` depends on the centering in *that*
dimension alone.

A vertex-like dimension stores one plane more: the boundary plane a block
**shares** with its high-side neighbor. Ownership is half-open — a block
owns its points `0 … N−1` in every dimension, whatever the centering — so
the state vector still holds `N^D` values per block per variable, and the
shared plane is filled by the exchange exactly as a ghost is. A flux
therefore needs no ghosts at all, only that one extra plane:

```jldoctest overview
julia> flux = FieldSet(forest, 2; G = (0, 0), centering = facecentered(2, 1));

julia> size(flux.work)
(9, 8, 2, 7)
```

[`interiorview`](@ref) returns the owned points and [`closedview`](@ref)
the owned points plus the shared plane — a block's `N+1` faces in the
staggered dimension, which is what a flux kernel launched with
`map_blocks!(...; closed = true)` writes. A third launch range,
`map_blocks!(...; stored = true)`, covers every stored point, ghosts
included; there the kernel's global index *is* the stored index, while
the other two hand out an offset into the owned range that the kernel
shifts by `G`.

In a vertex-like dimension the inter-grid operators change with it:
restriction is exact **injection** (the coarse point at position `X`
coincides with fine point `2X`), and prolongation interpolates at integer
or half-integer coarse coordinates, needing `G[d] ≥ p/2 − 1` rather than
`p/2`. The conservative family is refused along a stagger, where a
staggered quantity stores a point value and there is nothing to conserve.

## Element types

The mesh is generic in its floating-point type, and not merely in what it
stores: the geometry is *computed* in that type too, so nothing on the
path from a key to a cell center evaluates in `Float64` unless that is
the type you asked for. This matters because hardware fp64 is not
available everywhere — on many GPUs it is absent or an order of magnitude
slower.

A [`Forest`](@ref) carries the type, and [`FieldSet`](@ref) and
[`GhostSchedule`](@ref) take theirs from it unless told otherwise:

```jldoctest overview
julia> floattype(forest)
Float64

julia> small = Forest{Float32}((2, 2); N = 8);

julia> eltype(FieldSet(small, 1; G = 2).work)
Float32
```

Interpolation weights are built in exact rational arithmetic and rounded
once, at the point where they enter a stencil, so they carry no error
beyond the target type's own — and the conservative family's
"children average back to their parent" is an algebraic identity rather
than a statement about roundoff. `Float64`, `Float32`, and a
double-`Float32` (MultiFloats.jl's `Float32x2`, which has no hardware
support at all) are exercised in the test suite.

## Ghost exchange

Ghost filling runs at every RHS evaluation, while neighbor finding is
only needed when the tree changes. So the exchange is split in two: a
[`GhostSchedule`](@ref) is built once and [`fill_ghosts!`](@ref) merely
replays it, with no tree query in the per-evaluation path.

A schedule belongs to a *layout* — one ghost width, one centering, one
element type, one backend — so it is built from a field set:

```jldoctest overview
julia> schedule = GhostSchedule(state, Operators(prolongation=2, restriction=2));

julia> fill_by_coordinates!((x, v) -> v * x[1], state);

julia> fill_ghosts!(state, schedule);
```

A coordinate callback is called once per point *and variable*. Wrapping
it in [`AllVariables`](@ref) selects the once-per-point form instead,
which returns every variable at once — what a state that is only
definable as a whole needs, such as a set of conserved variables built
from a primitive one:

```jldoctest overview
julia> fill_by_coordinates!(AllVariables(x -> (x[1], 2 * x[1])), state);
```

The two forms write bit-for-bit the same numbers, and
[`CellBoundary`](@ref), [`boundary_by_coordinates`](@ref) and
[`adapt_to_initial_data!`](@ref)'s `initial` take the wrapper too.

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
constrained by the block geometry, per dimension and per centering: order
`p` prolongation needs `G[d] ≥ p/2` in a cell-centered dimension and
`G[d] ≥ p/2 − 1` in a vertex-like one, which [`check_operators`](@ref)
enforces when the schedule is built. It is also constrained by your
discretization — see the warning in [`Operators`](@ref), which is worth
reading before picking an order.

## Conservation at coarse-fine faces

With one global timestep, a conservative scheme needs only that the flux
a coarse cell sees on a coarse-fine face equals the area-weighted sum of
the fine-face fluxes there. That is a purely spatial condition, settled
within a single right-hand side, so there are no flux registers and no
time-accumulated corrections — they exist only to bridge subcycled steps.

A conservative RHS is therefore three steps: compute the fluxes over each
face set's closed range, restrict them at coarse-fine faces, then apply
the divergence.

```julia
map_blocks!(flux_kernel!, flux, ...; closed = true)   # N+1 faces per dimension
restrict_interfaces!(flux, isched)                    # the fixup
map_blocks!(divergence_kernel!, state, ...)
```

[`InterfaceSchedule`](@ref) is built once per tree like a
[`GhostSchedule`](@ref), and takes no [`Operators`](@ref): the transfer
is injection and the exact two-cell average, fixed by the geometry, not
interpolation. It is built over the *flux* field set — the one whose
centering is vertex-like in the face dimension — and refuses a
cell-centered one, which has no values lying on a block's face at all. It
reads and writes closed-range values only, so a computed flux may carry
`G = 0`.

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
if regrid!(forest, fs => schedule; flags = flags)
    schedule = GhostSchedule(fs, operators)       # the old one is now stale
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

## Threading

Start Julia with threads and everything in the package uses them:

```bash
julia -t auto --project=. my_run.jl
```

There is no switch to throw and nothing to configure. Per-cell work is
KernelAbstractions kernels, whose CPU backend spreads a launch over the
available threads, and the host-side passes over blocks — neighbor
finding when a [`GhostSchedule`](@ref) is built, the mark arithmetic in
[`regrid!`](@ref), the boundary hook, the reductions — are parallel
loops over blocks.

Results are **bit-identical** whatever the thread count, for everything
that is not a floating-point sum: every parallel loop writes to its own
slot, so the state, the mesh, the schedule and every max or integer
reduction on 64 threads reproduce a run on one exactly. A floating-point
sum — a norm, a total mass — is promised to roundoff only, so that a
device may reduce hierarchically and MPI may `Allreduce`; today's CPU
fold is per-block and so still exact. That is worth relying on when
debugging: a difference between two runs beyond the last bits of a sum
is never the thread count.

Two consequences for application code:

- callbacks run concurrently. The `f(x, v)` of
  [`fill_by_coordinates!`](@ref), the `f(b, key)` of
  [`flag_blocks`](@ref), and the `boundary` hook of
  [`fill_ghosts!`](@ref) are each called from several threads at once,
  so they must be pure functions of their arguments (the boundary hook
  may write the region it was handed, and nothing else);
- a block is the unit of parallelism, so a mesh wants appreciably more
  blocks than threads — a few dozen blocks per thread is comfortable,
  a handful is not.

On a multi-socket machine, **interleave the pages**:

```bash
numactl --interleave=all julia -t 64 --project=. my_run.jl
```

This is worth 2–6× at high thread counts and is not something the
library can do for you — it is a policy for the whole process. The
reason it helps rather than first-touch placement is that the same
arrays are partitioned differently by different kernels (the working
array by stored cell, the state vector by interior cell, a ghost region
by target slab), so no single first-touch pattern serves them all.
Measured numbers are in
[CODE.md](https://github.com/eschnett/TreeAMR.jl/blob/main/CODE.md#parallelism);
`bench/scan.sh` reproduces them.

## Devices

The storage decides where the work runs. Pass a KernelAbstractions
backend when you allocate, and every kernel in the package follows:

```julia
using CUDA                                # or Metal, or any KA backend

forest   = Forest((4, 4); N = 32, periodic = (true, true),
                  extents = ((0f0, 1f0), (0f0, 1f0)))
fs       = FieldSet{Float32}(forest, 2; G = 2, backend = CUDABackend())
schedule = GhostSchedule(fs, ops)
```

There is nothing else to choose. [`statevector`](@ref) allocates where
the field set lives, [`regrid!`](@ref) reallocates there, and each
kernel takes its backend from the storage it is handed — so an RHS
written for the CPU is already the device RHS.

The schedule takes the backend too, and must be built for the same one:
its stencil weights and index vectors are read *inside* the transfer
kernel, so they have to live where that kernel runs. They are still
built on the host in exact rational arithmetic and uploaded once, when
the schedule is built, which is the same argument that put the exchange
in a cached schedule in the first place. A mismatch is reported rather
than left to fail in memory.

**Precision.** The mesh is generic in its floating-point type and
computes the geometry and the interpolation weights in it from the first
operation, so `Float32` needs no fp64 anywhere. A `Float64` field set on
a backend without hardware fp64 is refused at construction, with the
reason.

Two callbacks change shape on a device, because both used to be host
loops over block data:

- **Boundary conditions.** [`CellBoundary`](@ref) expresses the hook per
  cell — `g(x, v, δ)`, or `g(x, δ)` returning every variable at once
  under [`AllVariables`](@ref) — and the package launches it as a
  kernel. [`boundary_by_coordinates`](@ref) is one of these, so it runs
  anywhere. The older whole-region form, `boundary(fs, b, key, δ,
  region)`, is still accepted and is still what a condition that reads
  the block's interior (reflecting, extrapolating) needs — but it
  indexes the working array cell by cell, so it is CPU-only, and says so
  if handed a device field set.
- **Refinement criteria.** [`flag_blocks`](@ref) calls `f(b, key)` on
  the host, which cannot read device data. [`firing_boxes`](@ref) is the
  device form: it evaluates a per-cell predicate over every block in one
  kernel and returns each block's firing-cell count and bounding box.
  The *verdict* — which flag, against which maximum level — stays with
  the application, because that is physics the mesh cannot know:

```julia
fires(work, idx, b, x) = abs(work[idx..., 1, b]) > threshold

flags = map(enumerate(firing_boxes(fires, fs))) do (b, (n, box))
    n == 0 && return Coarsen
    level(forest.leaves[b]) < lmax ? (Refine, box) : (Keep, box)
end
regrid!(forest, fs => schedule; flags = flags, buffer = 4)
```

The box is exactly what [`regrid!`](@ref)'s buffering dilates, so the
device path feeds the same machinery the host one does.
[`adapt_to_initial_data!`](@ref) takes such a criterion through its
`flags` keyword.

`bench/gpu.jl` times the per-evaluation phases on a chosen backend, in
the format `bench/threads.jl` prints, so a device run and a host run can
be read side by side.

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

## Storage

```@docs
FieldSet
cellcentered
vertexcentered
facecentered
edgecentered
staggers
coordinates
nblocks
blockkey
blockview
interiorview
closedview
fill_by_coordinates!
AllVariables
KernelAbstractions.get_backend(::FieldSet)
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
CellBoundary
```

## Conservation at coarse-fine faces

```@docs
InterfaceSchedule
restrict_interfaces!
```

## ODE coupling

```@docs
statelength
statevector
statearray
scatter!
gather!
map_blocks!
block_mapreduce
volume_weighted_norm
```

## Regridding

```@docs
RegridFlag
flag_blocks
buffered_flags
complete_marks
regrid!
adapt_to_initial_data!
total_mass
firing_boxes
```

## Internals

Not exported, and not part of the public interface, but documented
because they define the shape of the schedule.

```@docs
TreeAMR.Stencil1D
TreeAMR.TransferGroup
TreeAMR.BoundaryRegion
TreeAMR.BoundaryBatch
TreeAMR.BoundaryPlan
TreeAMR.PhaseSlice
TreeAMR.lagrange_weights
TreeAMR.unit_lagrange_weights
TreeAMR.ghost_layers_read
```

The host-side threading primitives, for the same reason — the shape of
every parallel pass over blocks in the package:

```@docs
TreeAMR.threadchunks
TreeAMR.threaded_foreach
TreeAMR.threaded_chunks
TreeAMR.threaded_collect
```

The device-residency helpers behind the `backend` keyword:

```@docs
TreeAMR.todevice
TreeAMR.tohost
```

## Index

```@index
```
