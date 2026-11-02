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
  - [M13 — Nested-mesh ghost fill](#m13--nested-mesh-ghost-fill)
  - [M14 — Half-turn symmetry](#m14--half-turn-symmetry)
  - [M15 — Folds over points](#m15--folds-over-points)
  - [M16 — Element centering](#m16--element-centering)
  - [M9b — Visualization export](#m9b--visualization-export)
  - [Release 0.2.0 with TreeIOHDF5](#release-020-with-treeiohdf5)
  - [M17 — Multi-patch forests (uniform)](#m17--multi-patch-forests-uniform)

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

### M13 — Nested-mesh ghost fill

Done, 2026-10-20 (steps 1–4; step 4, a partition balanced per class of leaves, parked). The design is under "Ghost filling", "What the ghost fill costs" and "Stages" in "Distributed meshes" in [CODE.md](CODE.md#ghost-filling), and what it leaves open under "Performance work left for later" there; the record is in [HISTORY.md](HISTORY.md#m13--nested-mesh-ghost-fill).

### M14 — Half-turn symmetry

Done, 2026-10-21 (steps 0–8). The design is under "Half turn" in "Domain and boundaries" and *Half turns* under "Rotating seams" in "Ghost filling" in [CODE.md](CODE.md#ghost-filling), and what it leaves open under "Open questions"; the record is in [HISTORY.md](HISTORY.md#m14--half-turn-symmetry).

### M15 — Folds over points

Done, 2026-10-22. The design is under "Folds over points" in "Operations" in [CODE.md](CODE.md#folds-over-points), and what it leaves open there; the record is in [HISTORY.md](HISTORY.md#m15--folds-over-points).

### M16 — Element centering

Done, 2026-10-31 (steps 1–8). The design is under *The element centering* in "Centerings", the element-local family in "Operators", and "Ghost filling" and "Regridding" in [CODE.md](CODE.md#centerings), and what it leaves open under "Open questions"; the record is in [HISTORY.md](HISTORY.md#m16--element-centering).

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

### M17 — Multi-patch forests (uniform)

*(Proposed, not designed in detail.)* A conforming multi-patch forest on
uniform meshes, so that an application can solve on an annulus or a
cubed sphere. The design is the open question of that name in
[CODE.md](CODE.md#open-questions), whose four decisions — bricks glued
at faces, logical coordinates with the map left to the application,
uniform meshes only, and the rotating seams left as they are — are
settled; everything below follows from them. Asked for by TreeWaveOLDG,
whose element sets are filled on faces only and so never read a patch
edge or corner.

Steps, in order:

1. **The connectivity.** Several root bricks (patches), each with its
   own `roots` and `extents`, and a glue table carrying a patch face
   onto another patch's face with a signed axis permutation. The
   keywords, the refusals and the field of `Forest` that holds it, with
   `ForestDigest` covering it so that the collective checks see it. A
   one-patch forest is today's forest, bit for bit.
2. **Neighbor finding.** `neighbor_anchor` consults the table when a
   step leaves its patch; `oriented_neighbors` returns a general axis
   map `(perm, flip)` rather than a number of quarter turns;
   `wall_direction` treats a glued face as neither outer nor a wall. A
   glued face is refused as reflecting, outflow or periodic, and a
   forest with both glue and a rotating seam is refused.
3. **The schedule.** `GroupKey` carries the general axis map, and
   `rotated_source` takes it from the table instead of from `axismap`.
   The kernel, the pack, the unpack and the MPI path are unchanged. An
   oriented group with the identity variable map is allowed, for an
   application that stores physical components; a per-interface variable
   map is refused with its reason.
4. **Conformity.** A coarse-fine face across a glued face is refused, in
   `balance!`, in `isbalanced` and in the leaf list `Forest` accepts, as
   nonconformity at a seam is.
5. **Geometry.** `patch(forest, k)` and `block_patches(forest)`, beside
   `block_spacings` and `block_origins`; the cube check per patch.
6. **The checkpoint.** TreeIOHDF5 stores and restores the patch list and
   the glue table — a format version and a release of its own, with the
   new TreeAMR version appended to its `[compat]` list ("Companion
   interface" in [CODE.md](CODE.md#companion-interface)).
7. **Tests.** A four-patch 2D annulus, uniform, periodic around the
   ring, in `test/`: every ghost across a glue holds the value its donor
   owns at the same physical point, to roundoff, for a field set of each
   centering, the element centering included; the fill is bit-identical
   across thread counts; and `mpi_workload.jl` gains the annulus, so that
   one rank and three agree line for line.

*Done when* the annulus test passes serially, threaded and at three
ranks; a checkpoint of a multi-patch forest round-trips through
TreeIOHDF5; and TreeWaveOLDG's scalar wave converges at the expected
order on the annulus, with its discrete energy conserved across the
glued faces.

*Not in this milestone*, each refused with its reason: refinement across
a glued face (oriented prolongation and restriction, the oriented
interface restriction, 2:1 balance through the table); nonconforming or
overlapping patches; `N` or `P` differing across a glue; and point
interpolation across a glue.
