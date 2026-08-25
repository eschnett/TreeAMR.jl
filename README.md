# TreeAMR.jl

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

[![CI](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://eschnett.github.io/TreeAMR.jl/dev)

See [CODE.md](CODE.md) for the full design document and the milestone
roadmap, or the [documentation](https://eschnett.github.io/TreeAMR.jl/dev).
The package is currently at milestone **M1** (tree core).

## Status

Early development. Not registered, not ready for use.

Implemented so far (M1, serial and `D`-generic):

- Morton keys over a brick of `M₁ × … × M_D` octree roots, as a sorted
  linear octree of leaves.
- Neighbor finding across faces, edges, and corners, including periodic
  wraparound and coarse/fine interfaces.
- Refinement and coarsening, and 2:1 balance enforcement.
- Block storage: one array over all leaf blocks, ghosts included, plus
  the physical geometry (spacings, block extents, cell centers).

Next up is M2: the phased ghost exchange and the default interpolation
operators.

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
