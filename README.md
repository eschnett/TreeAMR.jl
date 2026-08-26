# TreeAMR.jl

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

[![CI](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://eschnett.github.io/TreeAMR.jl/dev)

See [CODE.md](CODE.md) for the full design document and the milestone
roadmap, or the [documentation](https://eschnett.github.io/TreeAMR.jl/dev).
The package is currently at milestone **M4** (regridding).

## Status

Early development. Not registered, not ready for use.

Implemented so far, serial and `D`-generic:

**M1 — tree core**

- Morton keys over a brick of `M₁ × … × M_D` octree roots, as a sorted
  linear octree of leaves.
- Neighbor finding across faces, edges, and corners, including periodic
  wraparound and coarse/fine interfaces.
- Refinement and coarsening, and 2:1 balance enforcement.
- Block storage: one array over all leaf blocks, ghosts included, plus
  the physical geometry (spacings, block extents, cell centers).

**M2 — ghost exchange and operators**

- A cached exchange schedule, built when the tree changes and replayed
  by `fill_ghosts!`, so no tree query runs per RHS evaluation.
- The phased fill: same-level copies and restrictions first, then
  prolongations swept coarsest target first.
- Polynomial interpolation operators of configurable order, with the
  `G`/`N` sufficiency checks that tie order to block geometry.
- Periodic boundaries (free, via the tree) and a physical-boundary hook.
- Written as KernelAbstractions kernels, CPU backend for now.

**M3 — ODE coupling**

- A flat state vector over leaf interiors, with `scatter!`/`gather!`
  against the ghosted working array, so a standard integrator
  (OrdinaryDiffEq) drives the whole hierarchy with one global `dt`.
- `map_blocks!` to launch application kernels over every block.
- Volume-weighted norms, so error measures are not skewed by refined
  regions contributing more entries per unit volume.
- Verified with the scalar wave equation: 2nd-order convergence in the
  volume-weighted L2 and L∞ errors against the exact sine mode, on a
  two-level periodic mesh.

**M4 — regridding**

- Flag → complete → rebuild → transfer, with marks completed so that 2:1
  balance survives and coarsening only where a whole sibling group asks.
- The initial-data cycle, which re-evaluates rather than interpolates as
  the mesh adapts, iterated to a fixed point.
- Two operator families: point-value (finite differences) and
  conservative (finite volumes — exact-average restriction,
  reconstruct-and-average prolongation).
- Conservation: with conservative operators the transfer preserves the
  volume integral to roundoff for *arbitrary* data. With point-value
  operators it does so only for fields they reproduce exactly, though
  coarsening alone conserves either way.
- Verified with a travelling pulse whose refined region follows it: the
  adaptive run matches a uniformly fine mesh's accuracy using fewer
  cells, so the moving coarse-fine interface introduces no artifacts.

Note that reaching 2nd order on a refined mesh needs **order-4**
interpolation — see the warning on `Operators`.

Next up is M5: multi-threading.

## Installation

```julia
using Pkg
Pkg.develop(url="https://github.com/eschnett/TreeAMR.jl")
```

## Testing

```julia
using Pkg
Pkg.test("TreeAMR")
```
