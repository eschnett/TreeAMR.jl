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
  - [M20 — TreeMultiPatch, and finite differences on patches](#m20--treemultipatch-and-finite-differences-on-patches)
  - [M21 — Symmetries on patches](#m21--symmetries-on-patches)
  - [M22 — Closed blocks](#m22--closed-blocks)

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
3. **`ElementPolynomial` and the state-vector form.** *(Implemented
   2026-11-09.)* *Done when*:
   - it is exact on `Q_P` at random points, to a few eps;
   - a query at a node reproduces the node's value to rounding;
   - derivatives are exact on `Q_P`;
   - the measured rates on a smooth field over a refined forest are
     `P + 1` for the value and `P + 1 − maxₐ mₐ` for a derivative;
   - filling the ghosts with NaN leaves every result unchanged.
4. **`PatchPoint` on a forest of patches.** *(Implemented 2026-11-09.)*
   *Done when* logical points
   on a four-patch annulus agree bit for bit with the brick of four times
   the roots, a point on a glued face is located in the patch it names,
   and `Lagrange` is refused there.
5. **The atlas, the chain rule, and `outside = :mark`.** *(Implemented
   2026-11-09.)* *Done when* a
   test atlas on a brick, one affine map and one nonlinear map, matches
   analytic derivatives up to second order, a missing method is refused
   on the host naming it, and outside points are marked and filled with
   NaN.
6. **Grids and `element_lattice`.** *(Implemented 2026-11-09.)* *Done when* a `UniformGrid` and an
   `AffineGrid` give the values of the same points passed as an array,
   bit for bit, and `element_lattice` places its points where
   `positions` says.
7. **Distributed.** *(Implemented 2026-11-09.)* *Done when*, with an atlas, `LocatedPoints`' routing
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

### M20 — TreeMultiPatch, and finite differences on patches

*(Proposed 2026-11-22; its design decisions taken 2026-11-23; the
TreeAMR side designed in CODE.md, "Finite differences across a glue",
and implemented in step 4.)*
TreeWave's scalar wave, a finite-difference scheme, on the patch systems
TreeWaveOLDG runs, as the simplified version of what
TreeGeneralizedHarmonic needs. The patch systems move out of
TreeWaveOLDG into a package of their own, **TreeMultiPatch**, which
TreeWave and TreeWaveOLDG use now and TreeGeneralizedHarmonic and
TreeGRRMHD later. Across a glue, TreeWave's ghosts are **interpolated**
from the patch across rather than copied; why, and the couplings not
taken, are under M20 in [HISTORY.md](HISTORY.md#m20--treemultipatch-and-finite-differences-on-patches).

**The split.** TreeAMR keeps what moves data: the forest of patches, the
glue and the ghost fill across it, interpolated ghosts included, point
location by `PatchPoint`, and the atlas *interface* (`chart_locate`,
`chart_map`, `chart_jacobian`, `chart_hessian`), whose design already
leaves inverting a map to the package that owns the patch system
("Interpolation on element sets and patches" in
[CODE.md](CODE.md#interpolation-on-element-sets-and-patches)).
TreeMultiPatch owns the patch systems, which TreeWaveOLDG's
`geometry.jl` holds today:

- each layout's roots, glue and forest builder, taking the forest's own
  keywords (`N`, `comm`, …) rather than a scheme's `P` and `E`, and the
  check that a forest is that layout;
- the map to physical space, with its Jacobian, the area vectors
  `S[d] = √g ∇ξ^d`, `√g`, and the second derivatives, generic in the
  number type and isbits, so that kernels on a device and the
  extended-precision tables evaluate them alike, and defined past each
  patch's box by at least the widest ghost region, since an
  interpolated ghost's position is the target's map there;
- the atlas methods, the inverse map included;
- `affine_patch` and `affine_map`, whether `√g` factorizes along a
  dimension, and the largest logical speed of light for a time step.

The scheme stays in the application: TreeWaveOLDG's flux coefficients,
truncation tables, Cartesian gradient set and its rule that only affine
patches refine, and every application's exact solutions and boundary
hooks.

Why a package and not TreeAMR: TreeAMR's coordinates stay logical ("A
multi-patch forest" in [CODE.md](CODE.md#domain-and-boundaries)); the
maps want ForwardDiff, StaticArrays and MultiFloats, none of which
TreeAMR needs; and a new layout should not cost a TreeAMR release and
the TreeIOHDF5 release that follows every one. TreeMultiPatch uses
TreeAMR's exports only and takes an ordinary `[compat]` range, not exact
versions as a companion does: needing an internal would mean the thing
belongs in TreeAMR, exported. TreeAMR's tests keep their own glue lists
and atlases in `test/helpers/` and do not depend on it.

**Interpolated ghosts.** The TreeAMR side is designed under
"Finite differences across a glue" in
[CODE.md](CODE.md#finite-differences-across-a-glue) (proposed); what
follows is the plan's summary of it. A ghost of a block at a glued
face, edge or corner sits at the point the target patch's map, extended
past its box, names. The schedule locates that point once, at build
time, through the atlas and the search of M19's location step, in
whichever patch holds it; each fill then evaluates a Lagrange stencil
there. Between ranks the sender computes (decided): the rank that owns
the source points applies the stencil and sends the value in the
schedule's own point-to-point messages, laid out once when the schedule
is built, as a copy's are. M19's routing, an `allgather` and an
`alltoallv` on every `interpolate`, is for occasional batches and stays
out of the fill. So no tree query and no collective enters the
per-evaluation path, a patch corner has a source like any other ghost,
and the refusals of a patch corner and of a reversed vertex-like axis
across a glue do not apply to an interpolated set. The stencil reads
the patch across's points, its face planes included, and its ghosts
inside that patch, but not its ghosts across a glue, which are
interpolated from this one; so the interpolation runs as a stage of its
own, after phase 1 and the boundary hook and before phase 2. Copying
stays what an element set does across a glue, since its
face nodes are its coupling. Decided, and still to settle in step 4:

- **Global components** (decided). TreeAMR interpolates each variable
  as a scalar. The states hold global tensor components, Cartesian, as
  TreeGeneralizedHarmonic's do today, so an interpolated value needs no
  transformation, and TreeWave's `u` and `v` are scalars. Patch-local
  components would need each interpolated tensor transformed into the
  target patch's basis by the Jacobians of both maps at the point; that
  callback is not in M20.
- **Touching patches, not overlapping.** The patches tile, as M17 built
  them, sharing only their face planes; overlap is the overset case
  CODE.md rules out. The price is the off-centered stencil at the
  patch's edge, which cost no accuracy in step 3b, and the dissipation
  the patch corners need.
- **The interpolation order**, a keyword with no default, since what a
  derivative stencil of order `p` needs from its ghosts is the scheme's
  to say.
- **Closed ownership at glued faces** (decided). Every point on a
  glued face is a degree of freedom of each patch holding it, evolved
  by each with its own stencil: a block on a patch's high glued face
  owns its high face plane too, `N + 1` planes along that dimension,
  and a point on a patch edge or corner is evolved by every patch
  there. The copies of a point differ by the scheme's error, as on
  overlapping grids, and nothing averages them. Why, against the
  alternatives, is under M20 in
  [HISTORY.md](HISTORY.md#m20--treemultipatch-and-finite-differences-on-patches):
  half-open ownership, a patch owning its low face planes only, leaves
  two of the seven-patch ball's eight corner lines to no shell whatever
  the orientation, and step 3b found them impossible to extrapolate
  stably. Closed ownership owns them three times over, needs no
  orientation rule, no averaging and no exchange back to an owner,
  treats every glued face alike and so keeps the ball's symmetry, and
  was stable in every test at `ε = 0.5`, the dissipation that the
  corners where three patches meet need under any coupling. Overlap of
  one plane or more failed at least one test.
- **An appendix to the state** (decided). The core stays `N^D` per
  block per variable ("Shared points and half-open ownership" in
  [CODE.md](CODE.md#centerings)), and the high face planes that closed
  ownership adds, on blocks at a patch's high glued face, follow the
  cores in the state vector as an appendix. A brick has none, so
  nothing changes for it, and the change is additive. The planes are
  stored already, as every vertex-like set stores its non-owned high
  plane; what changes is that they are evolved: `statelength`,
  `scatter!` and `gather!` carry the appendix; `statearray` stays the
  cores' array, and `statearray(u, fs; appendix = true)` the whole state;
  `map_blocks!` on a set with an appendix requires `appendix = true` or
  `false`, so that the application's right-hand side covers the planes
  by choice; the fill skips them; the flat device launches, the MPI
  message layouts and TreeIOHDF5's format learn of them. A fold or norm
  counts a point once, at the copy of the lowest-numbered patch holding
  it. A glue's low face plane,
  owned under half-open ownership already, is unchanged. Closed
  ownership was rejected for bricks because the copies of a point at a
  coarse-fine face drift apart ([HISTORY.md](HISTORY.md#centerings));
  across a kinked glue no copy can be exact anyway, step 3b measured the
  drift stable, and the next item keeps coarse-fine faces away from it.
- **No refinement boundary at a glue** (decided, for now). A
  coarse-fine face, edge or corner never touches a glued face, and a
  block on a glued face keeps its level through `regrid!`, so that
  neither the interpolated ghosts nor the appendix meet a level
  difference, and the appendix is never prolongated or restricted.
  `GhostSchedule` refuses an interpolated set on a forest that breaks
  this, and `regrid!` refuses a change that would, each saying why.
  Lifting it, with interpolation from a leaf at another level and the
  appendix in the regrid transfers, comes after M20.
Each step has the test that says it is done:

1. **TreeMultiPatch**, with `PolarAnnulus`, `SevenPatchBall` and
   `ThirteenPatchBall` moved from TreeWaveOLDG, the balls' shells
   reoriented with the radial axis outward, so that the outer sphere is
   every shell's high face and the hook's, and the thirteen-patch ball's
   atlas declaring its aligned coordinates (`chart_aligned`: the radius
   between outer shells, both angles between an inner and an outer shell
   over one face). *Done when* the
   Jacobian, area vectors and second derivatives agree with
   ForwardDiff's, past the box too; `Σ_d ∂_d S[d] = 0` holds to rounding
   on every patch; `chart_locate` inverts `chart_map` at random points,
   glued faces and patch corners included; every declared alignment
   holds to rounding at random points past each box; each layout's `√g` integrates
   to its physical volume; every layout's forest builds and passes its
   check; and the maps run in a kernel on Metal.
2. **TreeWaveOLDG on TreeMultiPatch**, its geometry code removed and the
   scheme's kept. If M19's step 8 is not done by then, its atlases are
   written in TreeMultiPatch, not TreeWaveOLDG. *Done when* the
   annulus's, the seven-patch and the thirteen-patch ball's convergence
   tables and energies are what they were.
3. **TreeWave on the annulus.** Done 2026-11-24, before step 2, and
   vertex-centred only (Erik's decision), now on TreeWave's `main`: the
   rotating Bessel mode converges at the operator's order, two. The
   record is under M20 in
   [HISTORY.md](HISTORY.md#m20--treemultipatch-and-finite-differences-on-patches).
3b. **A prototype of the coupling.** Done 2026-11-23, on two tori and the
   cubed sphere's surface; closed ownership chosen. The record is in
   [HISTORY.md](HISTORY.md#m20--treemultipatch-and-finite-differences-on-patches).
4. **TreeAMR: closed ownership and interpolated ghosts across a
   glue.** Done 2026-11-24, serially, over MPI and on a device, and
   TreeIOHDF5 saves the appendix (released in its 0.1.11); two criteria
   restated by Erik. The design is in
   [CODE.md](CODE.md#finite-differences-across-a-glue); the record and
   the criteria as met in
   [HISTORY.md](HISTORY.md#m20--treemultipatch-and-finite-differences-on-patches).
5. **TreeWave on the seven-patch ball.** Done 2026-11-24, before step
   2, now on TreeWave's `main` against the registered 0.2.12: the plane
   wave converges at second order in L2 and near the glues, the closed
   ball's energy falls over 33 crossings at `ε = 1/2`, the quadrature
   gives the volume at second order, and both inversion and the mirror
   hold to rounding. The record is under M20 in
   [HISTORY.md](HISTORY.md#m20--treemultipatch-and-finite-differences-on-patches).
6. **The releases.** TreeAMR 0.2.12 and TreeIOHDF5 0.1.11 are
   registered (2026-11-25; the record is in
   [HISTORY.md](HISTORY.md#m20--treemultipatch-and-finite-differences-on-patches)).
   Left: TreeMultiPatch registered (its repository, CI and `[compat]
   TreeAMR = "0.2.12"` are in place); then TreeWaveOLDG and TreeWave
   released, each with its `[sources]` entry for TreeMultiPatch replaced
   by a `[compat]` bound.

Not in M20: refinement boundaries at a glue; the transformation of
patch-local components; per-patch face kinds, for a reflecting or
outflow outer sphere (the face kinds are per `(dim, side)`, and a cubed
sphere is glued on every one), which TreeWave's Dirichlet hook does not
need and the GR codes' outer boundaries will; and
TreeGeneralizedHarmonic and TreeGRRMHD on patches.

*Done when* steps 1–6 pass.

### M21 — Symmetries on patches

*(Proposed 2026-11-25; Erik's decisions on refinement taken that day.)*
*(On hold behind [M22](#m22--closed-blocks) since 2026-11-25: its steps
1–5 are on eschnett/TreeAMR.jl#50 and resume on closed blocks, which
remove the shifted slab at a glued axis, the turn's low-to-low rule and
the appendix those steps work around.)*
The symmetries a brick has — mirror planes (bitant, quadrant, octant)
and turns about an axis by 90° and 180° (M12, M14), alone or with a
mirror plane across the axis — on forests of patches, so that a
spinning black hole sits at the centre of a seven- or thirteen-patch
ball computed in a half, a quarter or an eighth of it. Both balls are
symmetric under the reflections through the coordinate planes and the
turns about `z` (step 5 of M20 measured the mirror and the inversion to
rounding), and every symmetry plane is a patch mid-plane: the cube's, or
a shell's tangential one, a root boundary when the root count across it
is even. The octant, for one, is the cube's octant and three quarter
shells; the 90° turn the cube's quarter, the `+x` and `+y` half shells
glued to each other through the turn, and the two `z` shells' quarters,
each glued to itself.

What stands in the way today ("A multi-patch forest" in
[CODE.md](CODE.md#domain-and-boundaries)): the face kinds are per
`(dim, side)` for every patch at once, and a glued face may not be
reflecting, so a mirror plane that is a different logical face on each
patch cannot be said; a forest of patches refuses `rotating` and
`halfturn`; and under `glue = :interpolate` every leaf on a glued face
keeps its level, so a seam through the central box would freeze the
black hole's neighbourhood.

**Refinement** (decided by Erik):

- **Mirror planes restrict nothing.** A mirror plane is a reflecting
  face, not a glue: nothing lies across it to match, and its ghosts
  come from the block's own mirrored interior, prolongated from a
  mirrored source where a level changes, as on a brick.
- **Turn seams keep conformity**, a brick seam's rule ("Rotating seams"
  under [Ghost filling](CODE.md#ghost-filling)): a leaf on a seam face
  and its image are at one level, and `balance!` refines the coarser.
  Against the full domain this refines more only where the full domain
  would have a coarse-fine face lying *on* the seam's half-plane; a
  pattern centred on the axis, as about a spinning black hole on it,
  conforms by symmetry. Edges and corners across a seam keep 2:1
  balance, through the rotated prolongation and restriction a brick's
  seam already has.
- **Free regridding in the central box.** A symmetry glue is aligned,
  an exact copy through a signed permutation, and is exempt from M20's
  rule that a leaf on a glued face keeps its level; that rule stays
  where it was made, at the kinked glues between the cube and the
  shells.
- **Prolongation across a seam face** is possible, and left for later,
  if a refinement pattern needs a coarse-fine face on a seam: it needs
  an oriented interface restriction for conservative sets and an
  ownership rule for a vertex-like set's two copies of the seam plane
  at different levels, which conformity makes unnecessary.

**The pieces**, each to be designed in step 1:

- **Face kinds per patch face**: reflecting, outflow or the hook, set
  for each `(patch, dim, side)` that is not glued, on a forest of
  patches; a brick's per-`(dim, side)` kinds unchanged. The same serves
  a reflecting or outflow outer sphere, which M20 left out. A set's
  parity is declared per physical axis, as the states hold Cartesian
  components (M20's global components), and each mirror face takes the
  parity of the physical axis its plane is normal to, which the layout
  names.
- **Turn seams as glue entries**: a glue entry gains a frame turn —
  none, a quarter turn either way, or a half turn — and a copy across
  it applies the set's `rotation` or `halfturn` variable map, as
  `RotatedSource` does for a brick seam. A vertex-like set's two owned
  copies of the seam plane, and the half turn's shifted seam slab, are
  re-derived for patches, where closed ownership (M20) may make the
  slab's shift unnecessary.
- **The fold at a kinked glue**: a ghost interpolated across a kinked
  glue near a mirror plane or a seam can sit outside the reduced
  domain, and its stencil can straddle the plane. The atlas folds the
  point back — an optional method giving the image point and the
  symmetry that carried it — and the ghost is the value at the image
  under that symmetry's parity signs or variable map, per ghost.
  `interpolate!` folds the same way, which a horizon finder on a
  symmetric ball needs.
- **Counting**: a seam point counted once in folds and norms, which the
  glued-point masks do once a seam is a glue entry.

**Open, for step 1**: whether a brick's seams collapse into the glue
table too ("A multi-patch forest" in
[CODE.md](CODE.md#domain-and-boundaries) leaves it open); whether the
symmetry is declared by the layout (TreeMultiPatch) or by a forest
keyword; and the checkpoint format for per-patch face kinds and turned
glue entries.

Each step has the test that says it is done:

1. **The design**, in CODE.md, marked proposed until Erik approves it,
   the open questions above settled. *Done when* Erik approves.
2. **Face kinds per patch face.** *Done when*, on TreeAMR's own test
   layouts, a forest of patches cut by a mirror plane fills its ghosts,
   regrids and evolves, as the full forest does on symmetric data, to
   rounding, in every centering, with a refinement region against the
   mirror plane; and the refusals of the per-`(dim, side)` kinds on
   patches give way.
3. **Turn seams as glue entries.** *Done when* a brick seam written as a
   glue entry fills, regrids and evolves bit for bit as M12's and M14's
   seams do (their tests, re-run on the glue form); a forest of patches
   with a turned glue does as the full forest on symmetric data, to
   rounding; `balance!` keeps conformity there; and a leaf on an aligned
   glue regrids freely under `glue = :interpolate`, serially, over MPI
   and on a device.
4. **The fold.** *Done when* interpolated ghosts across a kinked glue
   next to a mirror plane and next to a seam agree with the full
   layout's to rounding, in parity and in the turned components; and
   `interpolate!` returns the full domain's values at points beyond the
   planes.
5. **TreeMultiPatch's symmetric layouts**: both balls under the bitant,
   quadrant and octant mirrors and under the 90° and 180° turns about
   `z`, alone and with the `z` mirror, with even root counts across the
   cuts. *Done when* each layout's forest builds and passes its check,
   its atlas folds every point of the full ball, its `√g` integrates to
   its share of the volume, and `chart_aligned` declares the new pairs.
6. **TreeWave on the symmetric balls**: the sum of a plane wave's images
   under the symmetry, an exact solution with that symmetry. *Done
   when* each reduced ball's run is the full ball's, restricted, to
   rounding, at `N = 8` over `t = 2`, refined about the origin; and the
   energy stays bounded as in M20.
7. **The releases**: TreeAMR, TreeIOHDF5 with the new format,
   TreeMultiPatch, TreeWave.

Not in M21: prolongation across a seam face (above); refinement at a
kinked glue, M20's rule; patch-local components; symmetries of a turn
about any axis but the balls' `z`.

*Done when* steps 1–7 pass.
### M22 — Closed blocks

*(Proposed 2026-11-25, with Erik's decisions of that day; the
design is "Closed blocks" in [CODE.md](CODE.md#closed-blocks), the
record under M22 in [HISTORY.md](HISTORY.md#m22--closed-blocks).)*
Vertex-like dimensions own their closed range, `N + 1` points, so that
domains, refined regions, seams and glues are symmetric by
construction; shared points are stored by every block holding them and
unified by the fill. It replaces half-open ownership and its
work-arounds — symmetric walls, the derived wall row, the half turn's
shifted slab, the appendix of a set interpolated across a glue, and the
refusals that cite ownership — and it is a breaking release, 0.3.0,
for TreeAMR, TreeIOHDF5 and every downstream.

**Decided** (Erik): the state is uniform and holds every copy,
`∏(N + c_d)` per block; at a coarse-fine face the fine block is the
authority; half-open ownership is removed, not kept as an option; the
copies across a kinked glue are averaged; an outer wall is evolved
unless the hook prescribes it; unifying is a stage of its own in the
fill, to be measured; `unify!` on a state vector is offered and
optional; the interface restriction becomes `unify!` on a flux set;
positions are formed from integer coordinates, so that every copy of a
point is at one position bit for bit.

**How it is built.** Step 1 is independent of the rest and goes to
`main` by itself. Steps 2–10 are built on an integration branch,
`m22`, each as a pull request into it, so that `main` stays releasable
until the branch merges; the releases are step 11.

- *No half-open path is kept beside the new one.* What a later step
  rebuilds is refused by `GhostSchedule` until then — reflecting and
  outflow faces, a seam, a forest of patches, a mesh with two levels —
  with a message naming the step, rather than left half-working.
- *A test that needs a later step is pending*, by testset, marked in
  the test file with the step that brings it back, and by case in
  `test/mpi_workload.jl` and the thread workload. A step is done when
  the testsets it names are back and pass, on the CPU, in the flat
  form `gpu_tests.jl` runs, over simulated ranks and under MPI; the
  branch merges when none is pending.
- *Each step begins with a written plan against the code*, on its pull
  request and reviewed before any code: the functions that change, the
  keys and ranges, what is deleted, and the testsets that come back.
  The plan settles what the design leaves to the implementer, named
  under each step below, and records it in `CODE.md` as it lands.
- *The cost is measured at step 5*, as soon as the downstream's nested
  mesh fills, so that a change of the fill's partition costs the steps
  before it and not the whole milestone.

Steps:

1. **Positions from integer coordinates** — done: every position is
   formed by `nodeposition` from integer coordinates ("Positions" under
   [Centerings](CODE.md#centerings)); the record is M22 step 1 in
   [HISTORY.md](HISTORY.md#m22--closed-blocks).
2. **The layout, the uniform fill and outer faces** — done on `m22`:
   the closed own range, stage 0 and the symmetric partition, the
   primary mask with the reductions that take each point once and
   `isprimary`, the trapezoid's weights, `unify!`, and the hook at an
   outer face, the wall evolved or prescribed (stage W), on uniform
   meshes with periodic and outer faces; what later steps rebuild is
   refused until then, its tests pending. The design text is under
   [Closed blocks](CODE.md#closed-blocks) (*Stage 0, the mask and the hook*),
   the record, with what the code corrected in the design, is step 2 in
   [HISTORY.md](HISTORY.md#m22--closed-blocks).
3. **Coarse-fine faces.** Stage 0's injections, the mask across
   levels, prolongation into ghosts only, on meshes refined once and
   left alone; the interface restriction as `unify!` on a flux set,
   with `InterfaceSchedule` and `restrict_interfaces!` removed. The
   plan settles: which finer neighbour fills the mid-plane of a halved
   range; whether `N ≥ 2G + 2` is still needed anywhere along a
   stagger. *Done when* the fill of a refined mesh is mirror-symmetric
   to rounding for a vertex-centered set (4.7 × 10⁻⁵ today); the vertex
   convergence tables under [Operators](CODE.md#operators) are
   re-measured and recorded; the interface tests pass on `unify!` with
   their conservation numbers, exact at same-level faces; and an
   edge-centered set's copies around an edge beside a diagonal finer
   block are one number. *Implemented on `m22`*, every done-when item
   passing but one: the vertex convergence table is re-measured (0.97 /
   1.92 in `D = 1`, 1.04 / 1.95 in `D = 2`, against 0.99 / 1.99 and 1.01 /
   1.99) and awaits Erik's review before [CODE.md](CODE.md#operators)
   records it; the step is done then. The record is step 3 in
   [HISTORY.md](HISTORY.md#m22--closed-blocks).
4. **Reflecting and outflow faces** — done on `m22`: the mirror about
   the wall plane, stage P after stage W, the clamp to the wall row;
   `walls`, `wall_stencil` and the derived wall row deleted. Its
   done-when, as corrected: a closed box, uniform and refined, is
   mirror-symmetric to rounding under every face kind without a
   keyword; an odd variable is zero on its mirror plane; a bitant
   domain's `total_mass` is half the full domain's and its
   `volume_weighted_norm` equal, exactly on dyadic data and to a few ulp
   otherwise. The design text is under [Closed blocks](CODE.md#closed-blocks)
   (*Walls*, *Stage P*), the record step 4 in
   [HISTORY.md](HISTORY.md#m22--closed-blocks).
5. **The first measurement, and the partition settled** — done: measured
   on Symmetry and Fir against the half-open baseline, and Erik decided
   on 2026-11-26 that stage 0 stays a stage of its own and that
   `map_blocks!` does not skip secondaries by default. The numbers are
   under [Closed blocks](CODE.md#closed-blocks), the record M22 step 5
   in [HISTORY.md](HISTORY.md#m22--closed-blocks). Merging stage 0 into
   phase 1 is to be revisited after step 9's repeated measurements
   (step 9a below).
6. **Regridding** — done on `m22`, ahead of step 5's cluster numbers
   (the review's decision): transfers over closed boxes, the kept
   blocks' copies over the created ones, flagging over the closed box
   ("Regridding" under [Closed blocks](CODE.md#closed-blocks)); the
   record is M22 step 6 in
   [HISTORY.md](HISTORY.md#m22--closed-blocks).
7. **Seams** — done on `m22`: quarter and half turns on the ordinary
   partition, the seam's copies under rule 2 and the axis under rule 3
   (`seam_shift` had already gone in step 2). The design text is
   *Seams* under [Closed blocks](CODE.md#closed-blocks), the record M22
   step 7 in [HISTORY.md](HISTORY.md#m22--closed-blocks).
8. **Glues** — done on `m22`, but for its last done-when item: a glue
   that copies in any pairing and with reversed axes, the lowest patch's
   copy the primary; a kinked glue's copies averaged by stage A; group
   keys carrying the glue's axis map. The wave on the balls at M20's
   dissipation is TreeWave's, whose ball code used the appendix, and is
   checked when the downstream is ported. The design text is *Glues*
   under [Closed blocks](CODE.md#closed-blocks), the record M22 step 8
   in [HISTORY.md](HISTORY.md#m22--closed-blocks).
9. **What is left in TreeAMR** — implemented on `m22` but for its
   measurements: `interpolate!`'s windows, the schedule build of a
   vertex-like set made about as fast as a cell set's, the compile time
   cut, the documentation with the [migration
   note](docs/src/migration.md); no testset is pending. Outstanding: step
   5's measurements, `bench/mpi.jl` and the device suite on CUDA
   repeated on the finished branch (the studies are listed in M22 step
   9's record in [HISTORY.md](HISTORY.md#m22--closed-blocks)). *Done
   when* their numbers are recorded and replace step 5's in
   [CODE.md](CODE.md#closed-blocks).
   - 9a. **Stage 0 revisited**: on those numbers, whether stage 0 merges
     into phase 1 (Erik's decision of step 5, "revisit this later").
10. **TreeIOHDF5 and the downstreams**: the closed format feature and
    what a file of half-open blocks becomes; TreeWave, TreeHydro and
    TreeGRRMHD (which drops `walls_to_state!` and its symmetry
    testset's bound), then the others, sized by `N .+ staggers(fs)`
    and forming their kernels' positions through `nodeposition`, as
    the [migration note](docs/src/migration.md) lists. Step 8's last
    done-when item comes here: TreeWave's wave on the seven-patch ball
    stays bounded at M20's dissipation, once its right-hand side no
    longer uses the appendix. *Done when* each suite passes against the
    branch from a scratch copy.
11. **The releases**: `m22` merged, TreeAMR 0.3.0 first, and M21
    resumed on it.

Not in M22: refinement at a kinked glue; a patch corner's ghosts for a
set that copies; `map_blocks!` skipping the secondaries by itself,
unless step 5's numbers ask for it.

*Done when* steps 1–11 pass.
