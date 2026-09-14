# TreeAMR.jl

TreeAMR.jl implements a tree-based (octree-style) AMR discretization for
Julia. It provides the mesh, the storage, and the inter-grid operations —
no physics.

[![CI](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeAMR.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://eschnett.github.io/TreeAMR.jl/dev)

See [CODE.md](CODE.md) for the full design document and the milestone
roadmap, or the [documentation](https://eschnett.github.io/TreeAMR.jl/dev).
The package is currently at milestone **M6** (GPU support).

## Status

Early development. Not registered, not ready for use.

Implemented so far, `D`-generic, floating-point-type generic,
multi-threaded, and able to run device-resident on any
KernelAbstractions backend:

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
- Written as KernelAbstractions kernels, so the CPU implementation is
  already the device one (M6).

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

**M5 — multi-threading**

- Every per-cell kernel and every host-side pass over blocks is a
  parallel loop; start Julia with `-t auto` and there is nothing else to
  configure.
- Ghost filling runs one parallel loop per phase, with the transfers
  sliced by cell count and dealt out largest first — batching them by
  stencil alone left the small batches (edges, corners) serial and
  capped the ghost fill at about 2.5×.
- Results are **bit-identical** across thread counts, not merely equal
  to roundoff: no parallel loop shares an accumulator, and every
  reduction combines its partials in block order. The test suite checks
  this by running a full adapt/evolve/regrid/evolve cycle in
  subprocesses at different thread counts and comparing digests.
- Application callbacks (initial data, flagging, the boundary hook) are
  therefore called concurrently and must be pure.
- Measured on a 64-core AMD EPYC 7532 (960 blocks of 32³, 31.5M cells):
  **36.3×** on the RHS path and 59.5× on the compute-bound initial-data
  pass — with `numactl --interleave=all`, which is worth 2–6× at that
  thread count and which a library cannot set for itself.
  `bench/scan.sh` reproduces the measurement; the full table and the
  reasoning are in [CODE.md](CODE.md#parallelism).

**M6 — GPU**

- The storage picks the backend and everything follows it:
  `FieldSet(forest, nvars; backend = CUDABackend())` puts the leaf data
  on the device, and `statevector`, `regrid!` and every kernel allocate
  and launch there.
- The exchange schedule is device-resident too. Its stencil weights are
  read inside the transfer kernel, so they are built on the host in
  exact rational arithmetic and uploaded once, when the schedule is
  built — never per ghost fill.
- `CellBoundary` expresses a boundary condition per cell, which the
  package launches as a kernel; `firing_boxes` evaluates a per-cell
  refinement criterion and reduces each block to a firing-cell count and
  bounding box on the device, leaving the verdict to the application.
- The whole test suite passes on CUDA (NVIDIA H200) in Float64 *and*
  Float32, and on Metal (Apple M3 Pro) in Float32 — a backend with no
  hardware fp64 at all, which is the strongest available check that no
  fp64 path is load-bearing. The M3 convergence result is reproduced on
  the device: L2 rate 1.99 in either precision.
- Measured on an H200 against the same node's 16 cores (960 blocks of
  32³, 31.5M cells): **35×** on the RHS path, 32× on the ghost fill, 63×
  on initial data — tracking the 19× bandwidth ratio, which is what a
  mesh library should deliver. The two per-block reductions
  (`volume_weighted_norm`, `firing_boxes`) deliberately do not: they run
  one work item per block, which is what makes them deterministic, and
  neither is on the per-evaluation path. Numbers and reasoning in
  [CODE.md](CODE.md#parallelism); `bench/gpu.jl` reproduces them and
  `bench/symmetry_gpu.sh` is the cluster job.

Note that reaching 2nd order on a refined mesh needs **order-4**
interpolation — see the warning on `Operators`.

Next up is M7: MPI.

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

The tests run on whatever thread count they inherit; pass
`julia_args = ["--threads=8"]` to exercise the threaded paths (they are
covered either way, since the thread-independence test spawns its own
subprocesses).
