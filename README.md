# TreeAMR.jl

TreeAMR.jl implements a tree-based (octree-style) AMR discretization
for Julia. It provides mesh, storage, and inter-grid operations.

[![CI](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://eschnett.github.io/TreeAMR.jl/dev)
[![codecov](https://codecov.io/gh/eschnett/TreeAMR.jl/graph/badge.svg?token=IHXP23WQ1H)](https://codecov.io/gh/eschnett/TreeAMR.jl)

See [CODE.md](CODE.md) for the full design document and the milestone
roadmap, or the [documentation](https://eschnett.github.io/TreeAMR.jl/dev).
The package is currently at milestone **M8** (every centering,
per-field-set ghost widths, conservation at coarse-fine faces).

This package is still under development. It is ready for experimental use.

## Overview

TreeAMR is the mesh layer of a block-structured AMR code. It stores
the data, keeps the ghost zones filled, interpolates between levels,
and adapts the mesh. The equations, fluxes, boundary conditions, and
refinement criterion are decided by the application. A
finite-difference or finite-volume code written for one uniform block
should need little additional work to run on an adaptive mesh.

**The mesh.** The domain is a brick (rectangular grid) of octree
roots, each a block of `N^D` cells. Each block can be refined with a
factor of two between levels and 2:1 balance enforced across faces,
edges, and corners. Only the leaves of the tree store data, there is
no coarse data underneath refined regions. The leaves are kept as a
sorted vector of Morton keys, so neighbour finding is just arithmetic
on keys rather than pointer chasing, and periodic directions are built
into that arithmetic. Other physical boundaries need to go through a
per-cell hook that the application needs to define. The dimension `D`
is a type parameter, and the same code runs in 1D, 2D, and 3D, and the
tests cover all three.

**Global time stepping.** There is no subcycling. Every block advances
with the same global `dt`, set by the finest level. Ghost filling does
not need to interpolate in time. The whole hierarchy is one flat state
vector, so any time integrator from OrdinaryDiffEq.jl (or your own)
can drive an adaptive run exactly as it would a uniform one.
Conservation at coarse-fine faces reduces to a purely spatial
condition: the fine fluxes are averaged onto the coarse face within
each right-hand-side evaluation, with no flux registers necessary and
no corrections accumulated over time. The disadvantage is that coarse
levels take more steps than they need.

**Numerics.** Inter-level transfer comes in two families since
finite-difference and finite-volume codes require different
properties. The *point-value* family is Lagrange interpolation of even
order, for schemes whose unknowns are values at points. The
*conservative* family restricts by exact cell averaging and prolongs
by reconstructing a polynomial of odd order and averaging it over the
fine cells, so both operators preserve the volume integral of
arbitrary data to roundoff. You would choose the order according to
the discretization scheme you are using. For example, to stay second
order across a refinement boundary, a second-derivative stencil needs
order-4 interpolation, and a flux divergence needs order-3
conservative prolongation, two orders above the differencing in either
family.

Fields (variables) can live at cell centers (cell averages), vertices,
faces, or edges, per dimension, and different field sets with
different centerings and ghost widths can coexist on one refinement
forest. A typical hydro setup might use a cell-centered state with two
ghost cells and face-centered fluxes without ghosts.

**Regridding.** The application needs to flag blocks, either directly
or through a per-cell criterion. TreeAMR does the rest automatically:
it buffers the flags (i.e. it increases the set of flagged cells
outwards to ensure that the newly refined region stays safe for some
time) and completes them to ensure the 2:1 balance. Blocks are only
coarsened once a whole sibling group agrees. TreeAMR then
rebuilds the tree, and transfers the data with the same operators it
uses for ghost filling. A block can change by at most one level per
regrid. Initial data is re-evaluated on each new mesh rather than
interpolated, iterating until the mesh stops changing.

**Performance.** All per-cell work is implemented as
KernelAbstractions kernels, so that one implementation runs
multi-threaded on the CPU and efficiently on a GPU. If you start Julia
with `--threads=auto` then everything is multi-threaded, with results
that are bit-identical across thread counts except for floating-point
sums such as norms and totals, which are promised to roundoff only (so
that a device may reduce hierarchically, as it does, and MPI may
`Allreduce`). If you allocate the state vector storage with
`backend=CUDABackend()`, then the data and the exchange schedule live
on the device, and every kernel, the reductions included, runs on the
device. Both Float64 and Float32 are supported, and
the test suite passes both on CUDA (with both precisions) and on Metal
(which does not support double precision). Some performance
measurements are listed in [CODE.md](CODE.md#parallelism).

**Tests.** The test suite contains two small applications. The scalar
wave equation, as a vertex-centered finite-difference code, converges
at second order on a refined mesh, and a travelling pulse followed by
a moving refined region matches the accuracy of a uniformly fine mesh
with fewer cells. Burgers' equation, as a cell-centered finite-volume
code, sends a shock through a refined region that regrids around it
and conserves the domain integral to one or two ulp over hundreds of
steps. A standalone sample application,
[TreeWave.jl](https://github.com/eschnett/TreeWave.jl), solves the
wave equation with a Löhner refinement criterion.

**Still missing.** There is no MPI parallelism yet. The code runs on a
single node, either on its CPU cores or on one GPU. MPI is the next
milestone. There is no I/O or visualization output yet either. The
leaf-only storage also rules out multigrid algorithms on the mesh
hierarchy.

## Installing

```julia
using Pkg
Pkg.develop(url="https://github.com/eschnett/TreeAMR.jl")
```

## Testing

```julia
using Pkg
Pkg.test("TreeAMR")
```
