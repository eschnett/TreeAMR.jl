# TreeAMR.jl — Plan

Each milestone has a concrete acceptance test; serial correctness is
established before any parallelism. The numbers are the order the
milestones were planned in, and the list below is in execution order.

**Contents**

- [Milestones](#milestones)
  - [M0 — Scaffolding](#m0--scaffolding)
  - [M1 — Tree core](#m1--tree-core)
  - [M2 — Ghost exchange and default operators](#m2--ghost-exchange-and-default-operators)
  - [M3 — Wave equation and OrdinaryDiffEq](#m3--wave-equation-and-ordinarydiffeq)
  - [M4 — Regridding](#m4--regridding)
  - [M5 — Multi-threading](#m5--multi-threading)
  - [M6 — GPU](#m6--gpu)
  - [M8 — Every centering, per-field-set ghost width, conservation, Burgers](#m8--every-centering-per-field-set-ghost-width-conservation-burgers)
  - [M10 — Reflecting boundaries](#m10--reflecting-boundaries)
  - [M11 — Point interpolation](#m11--point-interpolation)
  - [M9a — Checkpoint and restart](#m9a--checkpoint-and-restart)
  - [M7 — MPI](#m7--mpi)
  - [M12 — Rotating symmetry](#m12--rotating-symmetry)
  - [M13 — Half-turn symmetry](#m13--half-turn-symmetry)
  - [M9b — Visualization export](#m9b--visualization-export)
  - [Release 0.2.0 with TreeIOHDF5](#release-020-with-treeiohdf5)

## Milestones

### M0 — Scaffolding

Done. Its record is in [HISTORY.md](HISTORY.md#m0--scaffolding).

### M1 — Tree core

Done. The design is under "Tree structure" in [CODE.md](CODE.md#tree-structure); the record is in [HISTORY.md](HISTORY.md#m1--tree-core).

### M2 — Ghost exchange and default operators

Done. The design is under "Ghost filling" and "Operators" in [CODE.md](CODE.md#ghost-filling); the record is in [HISTORY.md](HISTORY.md#m2--ghost-exchange-and-default-operators).

### M3 — Wave equation and OrdinaryDiffEq

Done. The design is under "Time integration" and the interface-order rule under "Operators" in [CODE.md](CODE.md#time-integration); the record is in [HISTORY.md](HISTORY.md#m3--wave-equation-and-ordinarydiffeq).

### M4 — Regridding

Done. The design is under "Regridding" in [CODE.md](CODE.md#regridding); the record is in [HISTORY.md](HISTORY.md#m4--regridding).

### M5 — Multi-threading

Done. The design and its measurements are under "Parallelism" in [CODE.md](CODE.md#parallelism); the record is in [HISTORY.md](HISTORY.md#m5--multi-threading).

### M6 — GPU

Done. The design and its measurements are under "Parallelism" in [CODE.md](CODE.md#parallelism); the record is in [HISTORY.md](HISTORY.md#m6--gpu).

### M8 — Every centering, per-field-set ghost width, conservation, Burgers

Done (M8a and M8b). The design is under "Centerings", "Operators" and "Conservation at coarse-fine faces" in [CODE.md](CODE.md#centerings); the record is in [HISTORY.md](HISTORY.md#m8--every-centering-per-field-set-ghost-width-conservation-burgers).

### M10 — Reflecting boundaries

Done. The design is under "Domain and boundaries" and "Ghost filling" in [CODE.md](CODE.md#domain-and-boundaries); the record is in [HISTORY.md](HISTORY.md#m10--reflecting-boundaries).

### M11 — Point interpolation

Done. The design is under "Point interpolation" in [CODE.md](CODE.md#point-interpolation); the record is in [HISTORY.md](HISTORY.md#m11--point-interpolation).

### M9a — Checkpoint and restart

Done. Since TreeAMR 0.2 the checkpoints are the companion package [TreeIOHDF5](https://github.com/eschnett/TreeIOHDF5.jl), whose `CODE.md` has the design; the record is in [HISTORY.md](HISTORY.md#m9a--checkpoint-and-restart).

### M7 — MPI

Done, 2026-10-02 (steps 0–9 and 6b). The design is under "Distributed meshes" in [CODE.md](CODE.md#distributed-meshes), and what it leaves for later under "Performance work left for later" there; the record is in [HISTORY.md](HISTORY.md#m7--mpi).

### M12 — Rotating symmetry

Done, 2026-10-03 (steps 0–10). The design is under "Domain and boundaries" and "Rotating seams" in "Ghost filling" in [CODE.md](CODE.md#ghost-filling), and what it leaves open under "Open questions"; the record is in [HISTORY.md](HISTORY.md#m12--rotating-symmetry).

### M13 — Half-turn symmetry

*(Specified 2026-10-21.)* A 180° rotating symmetry, the π-symmetry of
Cactus's RotatingSymmetry180: half of the `(d1, d2)` plane is
simulated, and the low face of `d1` is glued to itself, reversed about
the centre plane of `d2`. `halfturn = (d1, d2)` on the forest and a
`halfturn` variable map on every field set, as specified under "Half
turn" in [Domain and boundaries](CODE.md#domain-and-boundaries) and
*Half turns* under "Rotating seams" in
[Ghost filling](CODE.md#ghost-filling). It reuses M12's orientation 2 —
the oriented search, the virtual frame, the axis map, the tables,
conformity, the MPI path — and adds the turn about the centre line and
the shifted partition of a vertex-like `d2` in the seam slab. The change
is additive, a new keyword on `Forest` and on `FieldSet`, so it is a
`0.2.x` release, 0.2.4, with a TreeIOHDF5 release that saves the new
keywords. *Accept:*
- **refusals**, each with its reason: every case listed for the forest
  and the field set, a leaf list not conforming at the seam, and a
  `RotationPair` over a half-turn forest;
- **the neighbor oracle**: the half plane's oriented neighbors equal the
  images of the neighbors in an independently built *unfolded* forest of
  `2M₁ × M₂` roots, made by turning the leaves with exact `Rational`
  boxes, on random refinements and after `balance!`; the unfolded forest
  is 2:1 balanced and the seam conforming;
- **no ghost undefined, and none read before it is defined**, in
  `D = 2` and `3` with `z` periodic, outer or reflecting at its low
  face, for every centering, rotated restrictions and prolongations
  included (which a half turn has in 2D, at the seam's edges);
- **`halfturn_vs_doubled`**: data covariant under the half turn and not
  polynomial, on a refinement symmetric under it that reaches the seam
  and the axis, where every stored point, ghosts included, equals the
  full plane's within `1e-13`, cell, vertex and face centering, under
  half-open and symmetric walls;
- **write counts**: the schedule partitions the stored points, and every
  hook region's image leaves through an outer face;
- **the wave equation on a half plane against the full box**, a scalar
  and a vector wave: `linf` equal to the full box's to `rtol = 1e-8`,
  the vertex-centered seam half-planes equal to each other as measured,
  and the rate 2;
- **a regrid across the seam**, conformity kept, no block moved by more
  than one level, the ghosts afterwards equal to the oracle's;
- **conservation through the seam** under the rigid rotation, with the
  fixup, to roundoff;
- **interpolation beyond the seam**: values and first derivatives;
- **MPI**: lockstep packs in process, bitwise with a turned `−0`, and a
  half-turn case in `mpi_workload.jl` at `-n 2` and `-n 3`; a cycle in
  `thread_workload.jl`, a fill in `gpu_tests.jl`;
- **TreeIOHDF5**: a round trip and a restart that continues byte for
  byte.

### M9b — Visualization export

*(Not designed.)* After M7. The candidates:
- an XDMF sidecar that describes the checkpoint's own datasets as
  hyperslabs, for ParaView and VisIt (it grows by one grid per block);
- VTKHDF `OverlappingAMR` with restricted parents added, which turns
  leaf-only data into a real overlapping hierarchy and gives level of
  detail for about 1/7 more storage in 3D (ParaView only);
- Conduit Blueprint through Conduit.jl, for VisIt;
- a Parthenon-compatible export, which yt reads directly;
- sampling onto output grids (slices, a uniform box) through M11's
  `interpolate`.

An in-file view goes in a top-level group of its own and points at
the checkpoint's datasets without copying them, through links or
virtual datasets, as the layout under
[Checkpoint and restart](CODE.md#checkpoint-and-restart) leaves room for.

### Release 0.2.0 with TreeIOHDF5

The checkpoints moved to the companion package TreeIOHDF5, which removes
exported functions, so the next release is 0.2.0 (`Project.toml` says so
already). In order, each step Erik's to take:

1. Tag and register TreeAMR 0.2.0, after TreeIOHDF5's suite, TreeWave's
   and TreeHydro's (checkpoint tests included) have passed against the
   checkout.
2. Release TreeIOHDF5 0.1.0, and make the downstream changes, as
   TreeIOHDF5's `PLAN.md` lists them.

*Done when* TreeAMR 0.2.0 and TreeIOHDF5 0.1.0 are registered and the
three downstreams resolve them.

