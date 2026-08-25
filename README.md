# TreeAMR.jl

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

[![CI](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://eschnett.github.io/TreeAMR.jl/dev)

See [CODE.md](CODE.md) for the full design document and the milestone
roadmap, or the [documentation](https://eschnett.github.io/TreeAMR.jl/dev).
The package is currently at milestone **M2** (ghost exchange).

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

Next up is M3: the scalar wave equation driven by OrdinaryDiffEq.

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
