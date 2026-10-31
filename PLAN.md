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

*(Specified 2026-10-30.)* Mesh refinement for element-local
discontinuous Galerkin schemes such as TreeWaveOLDG's OLDG, which today
runs on uniform forests only because the coarse-fine ghosts and the
regrid transfer interpolate equally spaced cell centres. M16 adds an
`:element` centering, `elementcentered(D)`, with the polynomial order `P`
a property of the field set: each block holds `N/(P+1)` elements per
dimension of `P+1` Gauss–Lobatto–Legendre nodes each, both face nodes
stored, and a fine block covers whole parent elements. Its operator
family `ElementLocal` fills element sets on faces only, in one stage:
injection normal to the face; tangentially the coarse element's
polynomial at the fine nodes (`I`) for a fine ghost and the lumped
compatible restriction `R = M_c⁻¹ Iᵀ M_f` for a coarse ghost, so that
the discrete energy of an SBP scheme stays an exact invariant across
coarse-fine faces. A regrid refines by `I` and coarsens by the exact L²
projection. The design is the "(proposed)" bullet "Element-local DG
stencils" in [CODE.md](CODE.md#open-questions), which the milestone
replaces by the built design. The change is additive, a new centering
and a new `FieldSet` field and keyword, so it is a `0.2.x` release,
0.2.7, with a TreeIOHDF5 release that saves `P`. *Accept:*
- **the tables**, built in `BigFloat` and rounded once: GLL nodes and
  weights against closed forms for `P ≤ 4`; the quadrature exact to
  degree `2P-1` and not `2P`; `I` exact to degree `P`; `M_c R = Iᵀ M_f`
  in `BigFloat` and to rounding in `Float64`, `Float32` and
  `Float32x2`; `R` exact to degree `P-1` and not `P`; the coarsening
  exact to degree `P`, conserving `Σ w u`, and a left inverse of `I`;
  every table the correctly rounded `BigFloat` one, exact `0` and `1`
  entries included;
- **refusals**, each with its reason: `:element` in some dimensions
  only, `P` missing for an element set or given for another, `P < 1`,
  a ghost width above 1, `N` not a multiple of `2(P+1)`, a rotating
  forest or a `RotationPair`, point interpolation, an interface
  schedule, and every operator option `ElementLocal` does not take;
- **positions**: `coordinates`, `fill_by_coordinates!`, `FoldPoint.x`,
  `firing_boxes` and the boundary hooks' `x` agree bit for bit with each
  other and with TreeWaveOLDG's node positions, in 1D–3D, at `G = 0`
  and `1`, in `Float32x2` too; adjacent elements' and blocks' face nodes
  coincide; ghosts sit at the face; every existing position is
  bit-identical;
- **weights**: `volume_weighted_norm` exact on polynomials of degree
  `≤ 2P-1`, and `total_mass` the GLL quadrature;
- **the fill**, against an oracle that evaluates the rules per face
  from the element polynomials, on two- and three-level forests in
  1D–3D, periodic and reflecting, fine on either side: every face ghost
  written and every edge and corner ghost left alone; fine ghosts exact
  for `Q_P` data, coarse ghosts for `Q_{P-1}`; the adjoint identity
  `Σ W_c c g_c = Σ W_f f g_f` to rounding; one stage;
- **the regrid**: refining and coarsening exact for `Q_P`, coarsening
  after refining the identity, the GLL `total_mass` conserved, and
  `adapt_to_initial_data!` sampling at the nodes;
- **distributed, threads and device**: element cases in the exchange,
  regrid-exchange, thread, MPI and GPU tests, the serial and three-rank
  output equal, the thread counts equal bit for bit, and
  `--trace-compile-timing` of `test/mpi_workload.jl` compared against
  `main`;
- **downstream**: TreeIOHDF5 saves and restores `P`, and the suites of
  TreeIOHDF5, TreeWave and TreeHydro pass against the checkout.

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

