# TreeAMR.jl

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

[![CI](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://eschnett.github.io/TreeAMR.jl/dev)

See [CODE.md](CODE.md) for the full design document and the milestone
roadmap, or the [documentation](https://eschnett.github.io/TreeAMR.jl/dev).
The package is currently at milestone **M0** (scaffolding).

## Status

Early development; the public API does not exist yet. Not registered,
not ready for use.

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
