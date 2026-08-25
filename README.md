# TreeAMR.jl

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

See [CODE.md](CODE.md) for the full design document and the milestone
roadmap. The package is currently at milestone **M0** (scaffolding).

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
