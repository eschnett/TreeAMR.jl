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
  - [M18 — KernelAbstractions 0.10](#m18--kernelabstractions-010)
  - [M19 — Interpolation on element sets and patches](#m19--interpolation-on-element-sets-and-patches)

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

Implemented on 2026-11-02 (steps 1–8), on the branch
`eschnett-claude/multi-patch`, and not yet done. The design is under "A
multi-patch forest" in "Domain and boundaries", "Glued faces" in "Ghost
filling" and the passages they cite in [CODE.md](CODE.md#domain-and-boundaries),
and what it leaves open under "Open questions" there; the record, steps
and measurements are in
[HISTORY.md](HISTORY.md#m17--multi-patch-forests-uniform), and
TreeIOHDF5's side, which saves a forest of patches under a new feature
`patches` with no format-version bump, is on its branch of the same
name. TreeWaveOLDG's acceptance, its scalar wave on the four-patch
annulus, passed on 2026-11-07 against `adc7edb` with no change to
TreeAMR; its numbers are in
[HISTORY.md](HISTORY.md#m17--multi-patch-forests-uniform). What remains:

1. **The releases.** A TreeAMR release (TreeWave's and TreeHydro's
   suites pass against the checkout); then TreeIOHDF5's 0.1.6 (element
   sets, pending), and a release of its own for the feature `patches`,
   with `[compat] TreeAMR` the new version alone, since its save calls
   `patch_keywords`, and its `[sources]` path entries removed
   ("Companion interface" in [CODE.md](CODE.md#companion-interface)).

*Done when* TreeWaveOLDG's scalar wave converges at the expected order on
the annulus, with its discrete energy conserved across the glued faces,
and TreeAMR and TreeIOHDF5 are released with it.

### M18 — KernelAbstractions 0.10

*(Not designed.)* `[compat]` allows KernelAbstractions 0.9 only, here
and in every Tree* package. In 0.10, `KernelAbstractions.CPU` is the
POCL (OpenCL) backend, and the native CPU backend is gone, with its
`static` option: `launch_by_owner!` calls `CPU(; static=true)`, so the
suite fails almost throughout on 0.10.5 (932 of 966 failures on
2026-11-07 are that call). Moving to 0.10 is therefore a port of the
CPU path, not a compat bump. It has to keep what the CPU backend
carries now: every block on its owner's thread and the first-touch
placement that follows ("What one process loses" in
[CODE.md](CODE.md#parallelism)), results bit-identical across thread
counts, the shaped CPU launch of the copy kernels, and the CPU
cross-check of the device folds. The candidates for the owner launch
are POCL's own threading, if it can be pinned to blocks, or a host
loop through `threaded_chunks` with one launch per block. The device
backends need releases that work with 0.10 (Metal 1.12 does; CUDA is
not checked).

*Done when* the suite passes on 0.10, thread scaling and the compile
times of `test/mpi_workload.jl` are no worse than on 0.9, and TreeAMR,
TreeIOHDF5 and the downstreams are released with `"0.10"` in their
`[compat]`.

### M19 — Interpolation on element sets and patches

*(Designed; TreeAMR's steps in progress on the branch
`eschnett-claude/m19-implementation-40c9c9`, their record in
[HISTORY.md](HISTORY.md#m19--interpolation-on-element-sets-and-patches).)*
Point interpolation on the two things M11's
`interpolate` refuses, element sets (M16) and forests of patches (M17),
for TreeWaveOLDG's annulus and seven-patch ball: a horizon finder that
needs the metric and its first derivatives at physical points, and a
viewer that samples regular grids. The design is "Interpolation on
element sets and patches" in [CODE.md](CODE.md#interpolation-on-element-sets-and-patches);
the decisions taken in it are under "Point interpolation" in
[HISTORY.md](HISTORY.md#point-interpolation). Each step has the test that
says it is done:

1. **Location split from evaluation.** *(Implemented 2026-11-09.)*
   `LocatedPoints`, `locate`, and `interpolate(fs, pts, basis)`. *Done when* M11's tests pass unchanged,
   the split path reproduces the fused one bit for bit, and the
   thread-independence line still holds.
2. **The basis interface by position** *(implemented 2026-11-09)*, `t` in spacings from the block's
   origin in place of the continuous stored index. *Done when*
   `Lagrange` is bit for bit what it was.
3. **`ElementPolynomial` and the state-vector form.** *Done when*:
   - it is exact on `Q_P` at random points, to a few eps;
   - a query at a node reproduces the node's value to rounding;
   - derivatives are exact on `Q_P`;
   - the measured rates on a smooth field over a refined forest are
     `P + 1` for the value and `P + 1 − maxₐ mₐ` for a derivative;
   - filling the ghosts with NaN leaves every result unchanged.
4. **`PatchPoint` on a forest of patches.** *Done when* logical points
   on a four-patch annulus agree bit for bit with the brick of four times
   the roots, a point on a glued face is located in the patch it names,
   and `Lagrange` is refused there.
5. **The atlas, the chain rule, and `outside = :mark`.** *Done when* a
   test atlas on a brick, one affine map and one nonlinear map, matches
   analytic derivatives up to second order, a missing method is refused
   on the host naming it, and outside points are marked and filled with
   NaN.
6. **Grids and `element_lattice`.** *Done when* a `UniformGrid` and an
   `AffineGrid` give the values of the same points passed as an array,
   bit for bit, and `element_lattice` places its points where
   `positions` says.
7. **Distributed.** *Done when*, with an atlas, `LocatedPoints`' routing
   gives the serial values bit for bit at 1 and 3 ranks (a line of
   `test/mpi_workload.jl`).
8. **TreeWaveOLDG's acceptance.** The annulus's rotating mode and the
   seven-patch ball's plane wave, at random physical points that include
   glued faces and patch corners. *Done when* the value converges at
   `P + 1` and the gradient through the chain rule at `P`, and its
   annulus viewer draws a `UniformGrid` image.
9. **A horizon finder end to end**, as a script in TreeWaveOLDG's
   `bin/`, not in either suite. Kerr–Schild `γ` and `K` are filled at the
   nodes of a seven-patch ball, and ApparentHorizonFinder's
   `find_horizon` runs through a batched callback. *Done when* the
   horizon agrees with the analytic one at the element order.

*Done when* steps 1–9 pass, and TreeAMR is released with it.
