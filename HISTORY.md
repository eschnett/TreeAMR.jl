# TreeAMR.jl — History

How the package got where it is: who decided what and when, the
alternatives that were tried or rejected and why, measurements of code
that no longer exists or that later measurements replaced, the
milestone records, the downstream checks and the release history. The
current design, with the reasons for it, is in [CODE.md](CODE.md), and
the plan in [PLAN.md](PLAN.md). The sections follow CODE.md's.

**Step numbers.** The milestones M0–M12 are those of
[PLAN.md](PLAN.md), numbered in the order they were planned and done in
the order M0–M6, M8, M10, M11, M9a, M7, M12. Several were built in
numbered steps — M8a steps 1–3 and M8b steps 4–5, M7 steps 0–9 with a
step 6b, M12 steps 0–10 — and a reference such as "M7 step 6b", or
"step 8" inside a milestone's record, names one of them. Each
milestone's steps are listed in its record under
[Milestones](#milestones).

**Contents**

- [Core concepts](#core-concepts)
  - [Blocks](#blocks)
  - [Centerings](#centerings)
  - [Tree structure](#tree-structure)
  - [Domain and boundaries](#domain-and-boundaries)
  - [Precision](#precision)
- [Operations](#operations)
  - [Ghost filling](#ghost-filling)
  - [What the ghost fill costs](#what-the-ghost-fill-costs)
  - [The copy kernels on a device](#the-copy-kernels-on-a-device)
  - [Operators](#operators)
  - [Conservation at coarse-fine faces](#conservation-at-coarse-fine-faces)
  - [Regridding](#regridding)
  - [Point interpolation](#point-interpolation)
  - [Checkpoint and restart](#checkpoint-and-restart)
- [Application interface](#application-interface)
- [Time integration](#time-integration)
- [Parallelism](#parallelism)
  - [Distributed meshes](#distributed-meshes)
- [Code structure](#code-structure)
- [Open questions](#open-questions)
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
- [Downstream checks](#downstream-checks)
- [Development environment and CI](#development-environment-and-ci)
- [Releases](#releases)

## Core concepts

### Blocks

`G` moved from the forest to the field set, per dimension, in the M8 design; through M6 it was a forest parameter.

Ghosts on all sides of every block were decided early; the earlier ideas of x-only or unstored ghosts were dropped.

The `N ≥ p` invariant (the restriction window fits inside a fine block's interior) was found in M2. The conservative family's invariants were measured in its implementation (M8b), the vertex-like ones in M8a step 2.

### Centerings

Through M6 the package was cell-centered only, deliberately: the original plan scheduled face-centered variables for M8 and left vertex and edge centering unscheduled, accepting that the retrofit would touch the core (array shapes, ghost rules). The M8 design (decided) does all `2^D` centerings at once.

*(Designed in M8, before implementation, and implemented in M8a steps
1, 2 and 3. "Decided" below records the design discussion; the three
"Implemented in M8a" notes at the end of this section record what the
implementation settled or had to correct.)*

Rejected: closed ownership, both copies of a shared point in the state
vector. At a coarse-fine face the two copies see different ghosts — one
side's injected, the other's prolongated — so their right-hand sides
differ and the copies drift; keeping them consistent needs an exchange of
`du`, or of `u` inside the RHS, which the RHS contract forbids (AMReX's
`OverrideSync` exists to repair exactly this). The volume-weighted norm
would also count every shared point twice.

`coordinates(fs, b, idx)` replaced `cell_center` in M8; `cell_center` took the forest's `G` and assumed cell centering.

Rejected: one shape for every centering, `(N+2G)^D`, with the shared
plane doubling as the first high ghost. Every index convention would
have stayed literally as it is, but a ghost-free face field becomes
impossible — there is no slot for the high face when `G = 0` — and
ghost-free fluxes are the reason `G` moved to the field set at all. The
extra plane costs `(N+2G+1)/(N+2G)` per vertex-like dimension and buys
`G = 0`.

M8a step 1 made `G` a required `FieldSet` keyword and settled four things the design had left open. `check_operators` then kept M2's blanket `G_d ≥ 1` in every dimension, so that a ghost-free field set could not have a ghost schedule at all; step 2 relaxed that along a stagger, and a review after step 2 removed it altogether. In step 2 `centering` joined the forest form of `GhostSchedule`'s keywords and `fill_ghosts!`'s layout check.

M8a step 2 implemented the centering itself. The cell-centered stencils stayed the same rational weights, so no measured number moved: the whole suite passes
unchanged, the M3 wave tables included, and the thread-independence
digests still agree byte for byte. One thing the plan expected was not needed:

- **The oracle generalization the plan expected was not needed.** The
  plan anticipated teaching `cell_average` to average along cell-like
  dimensions only, so that a staggered field set could be checked as the
  mixed average-and-point-value object it is. Nothing needs it: that
  reading belongs to the conservative family, which is refused along a
  stagger, so every staggered set under test is point-value throughout
  and is compared at `coordinates`. The oracle was left alone.

M8a step 3 made the test wave equation vertex-centered and kept the M3 study verbatim in `test/wave_cell_tests.jl`; no cell-centered number moved.

### Tree structure

All-or-nothing refinement was amended from the original sketch's "up to 2^D children": leaf-only storage requires it.

The key encoding was decided early. Hilbert was considered for the curve and rejected (better MPI partition
locality, but the rotation arithmetic is not worth it at realistic rank
counts).

The neighbor asymmetry was recorded from M1.

### Domain and boundaries

Reflecting faces came with M10, which decided that `NoParity` is refused in a dimension with a reflecting face. Until M10 the boundary hook was also the only way to express a reflection, which it could not do correctly at every edge and corner (see [Ghost filling](CODE.md#ghost-filling)).

The rotating seam is M12, designed on 2026-10-03 before implementation. The rotation map as physics with no default was decided as `parity` was. Conformity at the seam was decided on 2026-10-03 with Erik.

The M12 design said: "The forest gains a field for it, `rotating`, with `(0, 0)` for none. A ninth field once made the schedule build allocate more (see "The buffer pool" under [Distributed meshes](CODE.md#distributed-meshes)), so `bench/ghosts.jl`'s allocation is measured before and after, and if the field costs again `reflecting` and `rotating` are folded into one immutable field instead. M12's step 1 records which." Step 1 (2026-10-03) found neither: what costs is the struct's size, not its field count. The numbers are under step 1 in the M12 entry below.

### Precision

The original specification said only that the working array's element
type is generic. That is too weak. Geometry was computed from `Float64`
extents and *converted* at the end, so a `Float32` field set still needed
hardware fp64 to find a cell center — and fp64 is exactly what a device
may not have.

The precision rule was amended in M5, when the implementation showed the original one-line rule was not enough; M5 also amended "Data layout" to point to "Precision" for what carries `T`.

## Operations

### Ghost filling

The boundary hook first ran last; M2 moved it between phase 1 and the prolongation sweep, after finding that prolongation stencils at a domain edge reach tangentially into the coarse source's own outer ghosts.

Rotating seams (M12) were designed on 2026-10-03, before implementation. Erik asked whether the seam ghosts should be filled as ordinary copies and then mixed by a second pass; the design showed that is not needed. Step 1 made the direction and child-offset formulas exact. Conformity at the seam, pairs for asymmetric layouts (brought into scope then), and 90° only were decided on 2026-10-03 with Erik; the second owned seam plane was decided in the design.

The design said a set with `G = 0` in both plane dimensions "fills alone, as today"; step 3 (2026-10-03) found that vacuous, since no ghost schedule serves such a set.

The M12 design said the two members "can share a stage's tag ...; an offset per member on the tag is the alternative if it turns out cleaner". Step 7 (2026-10-03) confirmed the shared tag and made it precise (what a stage completes before the next is its receives); no ordering hazard was found, and the members keep the shared tag. Step 3 (2026-10-03) found the schedule need record nothing for the plain fill's refusal, which decides from the layout. The design had "`regrid!` and `adapt_to_initial_data!` take `pair => (sa, sb)` and fill the pair before the transfers"; step 4 (2026-10-03) amended that: `regrid!` takes the element, and `adapt_to_initial_data!` has a pair form instead.

Batching by target level was amended in M5. The original keying did span levels; it was found while threading the schedule build, and no configuration could be constructed in which it actually produced wrong ghosts — but the ordering the sweep exists to guarantee was not in fact enforced, which is enough reason to fix it. The mirror state joined the group key in M10 and the orientation in M12; the M12 design had a group carry its axis map, and step 3 (2026-10-03) amended that to the orientation and plane, with `factorcol` becoming an `Int32`.

### What the ghost fill costs

Profiling a downstream solve (TreeGeneralizedHarmonic, a uniform mesh of
120 blocks of `8^3`) found that **the ghost machinery owned most of the
non-compilation run time and essentially all of the remaining heap
allocation**: the floating-point work of the physics was about 22 % of
run self time and the transfer kernel's addressing about 19 %. That is
the right order of magnitude to care about — and it understates the
refined case, where the same application measures the fill at 79 % of an
evaluation at a coarse-fine face against 22 % uniform. Four costs were
named; all four are real, one was diagnosed wrongly, and a fifth turned
out to be the largest. Measured here with `bench/ghosts.jl`, one thread,
`D = 3`, `N = 8`, 4 roots per edge, 10 variables, `p = 4`, flat *self*
time:

| | before | after |
|---|---|---|
| uniform mesh, 64 blocks, one fill | 5.12 ms, 17552 B | **2.90 ms, 10896 B** |
| two-level mesh, 120 blocks, one fill | 28.4 ms, 157408 B | **15.4 ms, 97504 B** |
| two-level mesh, schedule build | 19.6 ms, 39.1 MiB | **3.15 ms, 4.28 MiB** |
| `p = 6` schedule build | 89.5 ms, 104.4 MiB | **4.2 ms, 6.28 MiB** |

The same fills at four threads, where a phase runs as slices rather than
as one launch per group, gain slightly more: 1.01 ms → **0.53 ms**
uniform and 6.57 ms → **2.85 ms** two-level. So this is a change to what
the kernel does per point, not an accident of the serial schedule.

The two boundary kernels had the same flattening as the transfer kernel and got the same treatment. Since this is the one change that alters launch geometry, it was checked on real hardware and not only argued: the whole suite passes on Metal (Apple M3 Pro, `Float32`). The first version of this section said that on a GPU the division moves into KernelAbstractions' own `expand`, "where it belongs"; that was amended on 2026-10-05 (see "The copy kernels on a device").

Ghost fills are bit-identical to the
previous implementation across `D = 1, 2, 3`, both centerings and both
operator families — checked by digest against the previous commit, not
inferred.

That the per-launch allocation is KernelAbstractions' corrected the downstream brief, which put it on `run_group!` reassembling the group's geometry. Four costs were named by the downstream profile; all four were real, one was diagnosed wrongly, and a fifth (bounds checking, not on the list) turned out to be the largest.

### The copy kernels on a device

Measured and changed 2026-10-05; released as 0.1.8, 2026-10-06.

Found 2026-10-06, when the change failed CI. The first version passed
`weights = nothing` for a copy and chose the shape — `nothing`, or a
`LinearShape` of `Int32` or of `Int` — by a run-time `flat`. Every launch
site compiled up to six launches of which it ran one, and the
compilation-bound `mpi_workload.jl` took 75–85 s serially against 62–65
for 0.1.7, at 0.3–0.6 GB more peak memory (`--trace-compile-timing`:
62.4 → 73.8 s of compilation, all of it in host code; the kernels run
were the same 108). On CI's macOS runners, 3 cores and 7 GB with Julia
1.13, the three-rank job runs beside its serial reference and was
already near its 900 s deadline — it had missed it once on `main`, on
2026-10-04 — and now missed it in both macOS cells. With the fix below
it still missed it in one cell, so on a small machine the three-rank job
now runs after the reference instead (M7 step 9, amended). 

### Operators

The original specification had a default order of 2; M3 showed it silently producing first-order convergence, and the default was removed. Putting physics-specific operators in application packages resolved the original sketch's open question about where hydro-specific operators belong. Per-variable operator selection through field sets was decided in the M8 design. The interface-order rule was measured in M3, and its form along a stagger in M8a step 3.

The conservative family's prediction (rates 1, 2, 2) was measured in M8b step 5 with the Burgers study; that the norm is part of the result was measured there too, and the design did not anticipate it. The record of step 5 first credited the flux-divergence form for why the defect stays local; it was amended after step 5 to credit conservation, with the prediction made before the run.

Interface stencils were decided in M2. Reducing the order instead of shifting was rejected: a symmetric window at the
first ghost layer collapses to order 2 regardless of `p` — and by the
interface-order rule above, order-2 ghost data feeding a
second-derivative stencil leaves an `O(1)` interface truncation error,
capping global convergence at *first* order (measured in M3).

The operators per centering were decided in the M8 design, with the empty conservative vertex row; the vertex rows were measured in M8a steps 2 (exactness) and 3 (convergence).

### Conservation at coarse-fine faces

The conservative right-hand side is the "two phases" of the original sketch, with the fixup named. The
original plan tied this to M8 and left applications non-conservative at
coarse-fine interfaces until then (fine for the wave equation and the
Einstein equations); the M8 design of step (ii) follows.

The interface restriction was designed in M8 and implemented in M8b step 4; the design survived contact with the code. "Face directions only" was decided correcting a first draft that also listed edge directions for edge-centered fields. Across a rotating seam it was designed for M12 on 2026-10-03 and confirmed in M12 step 4 the same day.

### Regridding

Buffering was added post-M4, and amended to *box-as-source* after the first implementation measured Refine-keyed dilation to be inert at a steady-state frontier.

The conservation of the transfer was measured in M4. The conservative operator family landed early (pre-M5).

### Point interpolation

Point interpolation was added in M11, for the horizon finder of TreeGeneralizedHarmonic, which carried a stopgap of its own; the points below were decided in its design. Compared with the stopgap: the stopgap searched each ancestor in turn, `maxlevel` searches per point; it wrapped a periodic point for location only, not for the stencil, a latent bug no Dirichlet case could see; its excluded-region guard threw, because a horizon inside the damping layer is a bug there; and it moved to `threaded_foreach` precisely so that its refusal reached the caller readable. Mirroring a point beyond a reflecting face was decided with Erik.

Folding through a rotating seam was designed for M12 on 2026-10-03 and implemented in M12 step 5 as specified. Its second-derivative rule was amended on merging the second derivatives from `main`.

Derivatives were first order until 2026-10-03, when the second order was added.

Under M7 the routing was "the one place this design will change". It was designed on 2026-10-01 under "Distributed meshes" and implemented in M7's step 5, with the kernel unchanged except for the block offset. The serial path measured the same before and after.

### Checkpoint and restart

The history of the checkpoints — the design of M9a, the parallel-I/O
facts checked before M7 and the survey of formats — moved with them to
[TreeIOHDF5's HISTORY.md](https://github.com/eschnett/TreeIOHDF5.jl/blob/main/HISTORY.md)
(TreeAMR 0.2), and so did "Parallel checkpoints", the account of the
shared-file checkpoint that M7 step 6's record cites. The milestone
records, M9a and M7 steps 6 and 6b, stay under [Milestones](#milestones)
here.

## Application interface

The interface sketch was updated for the M8 design; through M6 `G` was a forest keyword and `regrid!` took bare field sets; `reflecting` and `parity` are M10's. The third launch range of `map_blocks!` was amended for TreeHydro.

## Time integration

The interiors-only state vector was decided early. The alternative — handing the padded working array to the integrator, with `du = 0` in ghost cells — was rejected: it makes the RHS mutate `u` and spends integrator bandwidth on ghost memory.

Several field sets in one state vector were specified in the M8 design, to be implemented with the first application that needs it.

Through M3 only fixed-`dt` integrators are exercised,
with `dt` chosen by the application from a minimum-spacing query.

The volume-weighted norm for adaptive integrators was documented in the design and implemented post-M3.

## Parallelism

Bit-identical results were decided in M5 and narrowed after M8 to promise floating-point sums to roundoff only.

The phase as one parallel loop was amended in M5: each phase was flattened into slices of roughly equal cell count, never crossing a batch, dealt out largest first, one task per thread, each slice launching as a single inline workgroup. It was amended again on 2026-09-23: still one parallel loop with every batch split across all threads, but by owner rather than by size.

Pinning KernelAbstractions to its *static* schedule,
 so that a chunk of an ndrange always lands on the same thread, was
 measured as the alternative and rejected: with first-touch placement
 it reproduced the left-hand column to within noise (RHS 10.0, scatter
 5.5, norm 19.3), for exactly the reason above — stability within one
 kernel does not make a page local to all the kernels that touch it.
 *(Revisited 2026-09-23:* right for the reason given and for one kernel
 alone, but most of what it was meant to fix was not placement. With
 the ghost fill given the same block-to-thread map as the kernels, the
 static schedule recovers 2.4x on the RHS; see "What one process
 loses" below.)

The NUMA investigation of 2026-09-23 on Symmetry first read its eight unsynchronized copies as a 3x gap to one process; it was amended the same day to 2.2x after re-measuring in shared windows, and the conclusions were amended where they depended on the difference. The plan it decided (no NUMA-aware page placement inside one process) was confirmed in M7's step 7, which measured one pinned 64-thread process against 8 ranks of 8 threads.

The backend chosen once at allocation was decided in M6. That the schedule has to move with the data was found in M6; the design had not anticipated it. The second form of two callbacks was found in M6 as well.

In M6 the boundary hook's region form was the only way to express reflection, which M10 moved into the schedule. M6's `firing_boxes` ran one work item per block, each looping its own cells: integer min/max is order-independent and every item owned its output slots, so the M5 determinism discipline carried over with nothing added; after M8 it became the two-launch reduction.

- **The two per-block reductions are the weak rows, by choice**
  (revisited after M8, below). `volume_weighted_norm` and
  `firing_boxes` both run one work item per *block*, so 960 work items
  on a device that wants tens of thousands. A hierarchical reduction
  would fix that and would give up the property that makes these
  functions trustworthy: one work item per block, each accumulating
  its own cells in its own order, is deterministic without a word of
  extra care, which is the M5 discipline. Neither is on the
  per-evaluation path — one is a diagnostic, the other runs at regrid
  frequency — so the trade is paid where it is cheap. It would have to
  be revisited if `volume_weighted_norm` were ever wired in as an
  adaptive integrator's `internalnorm`, which is still an open
  question above.

The two-launch device reduction and `mesh_mapreduce` were decided after M8, amended before implementation (the first version said a tree in local memory), implemented and measured on 2026-09-22, and amended after review (the lanes were a static workgroup of 256, and the norm's domain volume was summed separately). The first design of `mesh_mapreduce` combined across ranks with `Allreduce` once M7 existed, with `+`, `max` and `min` mapped to the builtin operations and anything else to a custom one; it was amended on 2026-10-01 to an `allgather` of per-rank partials.

M8 was decided to precede M7, so that the exchange would be layout-generic before MPI existed. The original MPI bullet said prolongation and restriction happen on the owner of the finer data; it was amended on 2026-10-01, when M7 was specified: the sender computes, so a prolongation runs on the coarse side.

### Distributed meshes

*(Designed 2026-10-01 as M7, before implementation. Four choices were
decided with Erik that day and are marked so; the rest follows from
them and from the sections cited, and is open to amendment as the
implementation measures it. The steps are in the M7 entry under
[Milestones](#milestones). Implemented in them by 2026-10-02, the
amendments marked where they were made. Measured on Symmetry the same
day: steps 7 and 8 in full, and step 6, whose multi-node run lost data
and led to the decision to replace the shared-file checkpoint; the
findings are in the steps, and amend the bullets below where marked.
The replacement, checkpoints without parallel I/O, was specified,
built and measured the same day as step 6b.)*

The design left the cost of replication at thousands of ranks for the weak-scaling smoke test (step 7) to measure. Step 7 found that of the replicated passes two grew to dominate a rank's regrid — the buffer's neighbour search when many blocks report boxes, and the classification of every new leaf — both of which could be made `O(local)` without changing a result; both were, the same day.

The forest digest was first specified as `hash(forest.leaves)`, which for `MortonKey` is defined from the fields; it was amended in M7 step 3, where the check was implemented, to the bullets under "Every forest mutation is collective" in CODE.md. `regrid!`'s checks joined the same gather in step 4.

Ranks without blocks: one host-side step did not guard: the
  `AllVariables` boundary hook checks its callback's tuple length at a
  point of block 1, and an empty rank has none, so it threw there while
  the other ranks went on to the next collective. TreeHydro, the only
  caller of that form, found it when it first ran over MPI; the
  workload's empty rank had been on a periodic line, where no hook runs.
  The check now returns early on an empty rank, as
  `fill_by_coordinates!` already did. A search of `src/` for any other
  step that assumes a block 1, and a scratch run of every public
  operation at four ranks over two leaves (CPU and Metal), found
  nothing else. Fixed in 0.1.6.

MPI+GPU was implemented in M7 step 8. The design had the pack and receive buffers allocated once per schedule (amended in step 2: and variable count), sent to MPI directly when MPI is device-aware, as `MPI.has_cuda()` says for CUDA, with a CUDA-aware MPICH opting in through `JULIA_MPI_HAS_CUDA`; step 8 amended it so that the caller decides, and recorded that Symmetry's MPI is CUDA-aware. A regrid stage was first built per regrid, its mirrors with it, CUDA unpinning them when they were collected; the buffer pool was added after step 8, on 2026-10-02.

The design's example for an empty rank's partial said that `max` over negative data from `init = 0` is right serially, and would be wrong if an empty rank's `0` entered the fold; M7 step 3 found the example wrong and replaced it with the weighted one. The `Allreduce` that the `mesh_mapreduce` bullet had planned was superseded by the allgather of per-rank partials. The buffer's neighbour search was replicated until step 7. The regrid transfer's buffer was first specified as shaped like the new block's owned box; that was amended while specifying, since it is right for a copy and a prolongation, but a restriction's target is one child's part of it, and a coarsened block's children may have had up to `2^D` different owners. Until step 7 a rank classified every new leaf. The regrid's distribution was implemented in step 4, which also decided that the ghost fill's checks stay rank-local.

Distributed interpolation was implemented in M7 step 5, which amended the design in two places: the design had the outside points agreed after the values returned, and said every rank throws the same `ArgumentError`.

**What an MPI test costs** (measured the same day). One `mpiexec -n 3`
subprocess, two threads per rank, that loads MPI and TreeAMR, builds a
forest and does one `Allreduce`, takes 1.7–1.8 s of wall clock in three
runs, 1.04 s of it inside the script on rank 0. The M5 workload,
`test/thread_workload.jl`, run under `-n 3` with each rank doing the
whole of it, takes 29.8 s, against 24.0 s for one process at two
threads. The ranks compile in parallel, so a rank count costs about one
serial workload plus a quarter. Two such subprocesses (`-n 2`, `-n 3`)
would add about a minute to the suite, and the serial reference can run
in-process. CI runners have three or four cores, while `-n 3` at two
threads wants six, and MPICH polls while it waits; on CI the ranks
should therefore run one thread each.

"Performance work left for later" was recorded on 2026-10-02 after the buffer pool; none of it blocked M7. Its item on compression where the data are was added in step 6b.

The multi-block check was made on 2026-10-01 and amended by the M12 design on 2026-10-03.

## Code structure

**The checkpoints left TreeAMR** (2026-10-08, decided with Erik), into
the companion package TreeIOHDF5, to make TreeAMR smaller: it had no
other user inside the package, and it was the only reason for the HDF5
weak dependency, the `CRC32c` dependency and TreeAMR's `__init__` (the
error hint saying to load HDF5). TreeIOHDF5 owns and exports the five
functions; TreeAMR dropped them, which is an API break, so 0.2.0. The
file format did not change — the group `/TreeAMR.jl`, the format names
and version 2, and `treeamr_version` still TreeAMR's — and gained the
additive provenance field `treeiohdf5_version`, so files written before
the split load as they did. A point-interpolation package was
considered the same week and not split off: it would remove about a
tenth of the source and couple the ghost exchange's internal tables
(`fs.factors`, `fs.rotvars`) to another package.

**Physics and companion packages** (2026-10-08, Erik's distinction): a
physics package uses only the exports, which semantic versioning covers;
a companion package, which provides infrastructure itself, may use the
companion interface and pins exact TreeAMR versions. Erik asked for a
suggestion on how to express the tighter coupling in versions; the
exact-version lists of CODE.md's "Companion interface" are it, chosen
over tilde bounds and over a run-time interface version.

## Open questions

The 180° rotation was deferred on 2026-10-03 with Erik, when M12 was designed. The integrator's own passes were raised by TreeGeneralizedHarmonic on 2026-09-25, after it adopted the ownership policy; Erik decided the same day not to optimise the OrdinaryDiffEq path further, Polyester included. The working array's layout and the state that lives in the working array were raised on 2026-10-05.

## Milestones

The milestone numbers are the order the milestones were planned in; **M8 is done before M7** (decided in the M8
design, see the M8 entry), and so are M10, which was added after M8,
and M11, added after M10 for a downstream horizon finder. **So is M9a,
checkpoint and restart** (decided 2026-09-29), because the downstream
applications need to restart long runs before they need MPI. M9 is
split for it: its second half, M9b (visualization export), stays after
M7. M9a's layout was chosen so that M7 need not change it, subject to
M7's benchmarks (in the end it did change it: format version 2, M7 step
6b). The list below is in execution order. M7 is done (2026-10-02).
M12, the rotating symmetry, was added after it on 2026-10-03 and done
the same day, so it comes before M9b, which follows it.

M9b was split from M9, "I/O and visualization", on 2026-09-29, when its checkpoint half became M9a.

M12 was decided with Erik, after M7 and before M9b.

### M0 — Scaffolding

- **M0 — Scaffolding.** Package skeleton, test harness, CI, docs stub.
  *(Skeleton exists.)*

### M1 — Tree core

- **M1 — Tree core (serial, D-generic).** Morton keys over a brick of
  roots, sorted leaf array, neighbor finding, refine/coarsen, 2:1
  balance enforcement, block storage, periodic wraparound. *Accept:*
  hand-rolled property tests with a seeded RNG (tiling, balance,
  neighbor soundness/completeness — exact reciprocity only at equal
  levels, see neighbor asymmetry above — and periodicity) on random
  refinement patterns in D = 1, 2, 3. *(Done.)*

### M2 — Ghost exchange and default operators

- **M2 — Ghost exchange and default operators.** The cached exchange
  schedule, the phased ghost fill (three cases, level-ordered
  prolongation), periodic boundaries, physical-boundary hooks, default
  operators of configurable order with the `G`-sufficiency check —
  written as KernelAbstractions kernels (CPU backend). *Accept:*
  polynomial data reproduced exactly up to operator order (degree
  `p − 1` and no further) across all face/edge/corner and three-level
  configurations. Periodic wraparound is tested by its definition
  instead — an `M`-root periodic domain reproduces the middle tile of an
  explicit `3M`-root tiling bit for bit (amended in M2: polynomials are
  not periodic, so polynomial exactness across the seam is unattainable;
  only constants are periodic polynomials). *(Done.)*

### M3 — Wave equation and OrdinaryDiffEq

- **M3 — Wave equation + OrdinaryDiffEq.** Scalar wave in 2nd-order
  form (state `(u, ∂ₜu)`, 2nd-order centered Laplacian), periodic cube,
  static two-level refinement over a sub-box, manual `f!`, fixed-`dt`
  RK4. *Accept:* volume-weighted L2/L∞ errors against the exact
  sine-mode solution converge at 2nd order — which requires **order-4
  operators and `G = 2`** (amended in M3; see the interface-order rule
  under [Operators](CODE.md#operators)), verified in D = 1, 2 with a 3D smoke
  test. The wave equation lives in the tests: the package has no
  physics. *(Done.)* *(Amended in M8a: the wave study is
  **vertex-centered** from M8 on — `test/wave_tests.jl`, with its own
  rate table under [Operators](CODE.md#operators) — and this cell-centered
  study is kept verbatim as `test/wave_cell_tests.jl` so the M3 numbers
  stay under test.)*

### M4 — Regridding

- **M4 — Regridding.** Flag → balance → rebuild → transfer; the
  initial-data cycle; integrator reinit. *Accept:* the initial-data
  cycle converges to a fixed-point hierarchy; a moving refined region
  tracks a travelling pulse with the accuracy of the *uniformly finest*
  mesh at fewer cells (measured error ratio 1.00 — matching that
  reference is what "without artifacts" means operationally); transfer
  conservation as stated under [Regridding](CODE.md#regridding) (amended in M4
  from the original blanket "conservation of transferred data", which
  refinement cannot deliver without conservative operators). *(Done.)*

### M5 — Multi-threading

- **M5 — Multi-threading.** Threaded loops over blocks. *Accept:*
  results match serial to roundoff; scaling measurement on a many-core
  node. Delivered stronger than asked on the first count: results are
  **bit-identical** across thread counts, checked by running a full
  adapt/evolve/regrid/evolve cycle in subprocesses at different thread
  counts and comparing digests of the state vector, the leaf array, the
  schedule shape and the reductions — the floating-point sums among
  those narrowed to a roundoff promise after M8, see
  [Parallelism](CODE.md#parallelism); the digests still agree, because the CPU
  fold did not change. Scaling on a 64-core AMD EPYC 7532
  (8 NUMA domains, 960 blocks of `32^3`): **36.3x** on the RHS path,
  59.5x on the compute-bound initial-data pass, with the table and the
  two findings that got it there — a phase must be one parallel loop,
  and pages must be interleaved — under [Parallelism](CODE.md#parallelism).
  `bench/scan.sh` reproduces the measurement. Re-measured on a Milan
  node on 2026-09-23 ("Where the 64-thread efficiency goes" and "What
  one process loses", same section): the remaining gap is not page
  placement but the loss of data-to-core affinity between launches,
  which the block-ownership launch policy recovers in one process
  (implemented the same day: 38.6x on the RHS path at 64 threads,
  pinned). *(Done.)*

### M6 — GPU

- **M6 — GPU.** CUDA backend via KernelAbstractions; device-resident
  data. Floating-point-type genericity *(landed early, after M5)* is a
  prerequisite that is now in place: the geometry and the interpolation
  weights no longer evaluate in `Float64` on their way into a `Float32`
  field, so nothing on the per-cell path needs hardware fp64. See
  "Precision" under [Core concepts](CODE.md#core-concepts). *Accept:* M3
  convergence results reproduced on GPU; kernel benchmarks. The backend
  is a keyword on `FieldSet` and `GhostSchedule` and nothing else; what
  the milestone did *not* anticipate, and what most of the work was, is
  that the exchange schedule has to become device-resident and that two
  application callbacks — the boundary hook and the flagging function —
  needed a second, cell-wise form, both recorded under
  [Parallelism](CODE.md#parallelism). The whole suite passes on CUDA (NVIDIA
  H200) in `Float64` *and* `Float32`, and on Metal (Apple M3 Pro) in
  `Float32`, a backend with no hardware fp64 at all. The M3 convergence
  result is reproduced on both: L2 rate **1.99** in `Float64` and
  **1.99** in `Float32`, with order-4 operators and `G = 2` over the
  two-level mesh. The `Float32` study has to be run at coarser
  resolutions, and for a reason worth recording: the error measured is a
  truncation error, the same number in every precision, while the
  roundoff floor it must clear moves — and the step count grows with
  `N`, so in `D = 1` the floor is already reached at `N = 64` (the rate
  over `N = 16…128` collapses to 0.73). That is the positive-assertion
  form of the caveat "Precision" records for negative ones.
  `bench/gpu.jl` produces the kernel benchmarks in the format
  `bench/threads.jl` uses, so a device run and a host run read side by
  side; `bench/symmetry_gpu.sh` is the cluster job that runs both.
  *(Done.)*

### M8 — Every centering, per-field-set ghost width, conservation, Burgers

- **M8 — Every centering, per-field-set ghost width, conservation,
  Burgers.** *(Done.)* Done before M7 (decided): the MPI exchange is
  built over the schedule, and with the schedule layout-generic first, M7
  distributes ghost fill, interface restriction and regrid transfer for
  every centering in one design, instead of building the cell-centered
  exchange and retrofitting it twice. The design is under
  [Centerings](CODE.md#centerings), [Ghost filling](CODE.md#ghost-filling),
  [Operators](CODE.md#operators) and
  [Conservation](CODE.md#conservation-at-coarse-fine-faces); the conservative
  cell-data operator family *(landed early, pre-M5, with its
  regrid-transfer conservation already verified)* is its foundation. The
  test problem is Burgers' equation, in the tests, as the wave equation
  is; an Euler hydro toy is a separate package, as TreeWave is for the
  wave equation. Two halves:
  - **M8a — layout.** `G` (per dimension) and `centering` on the field
    set, `Forest` without `G`, `coordinates` in place of `cell_center`,
    `GhostSchedule(fs, ops)`, per-dimension stencil widths, the vertex
    rows of the operator table, `regrid!` over `fs => schedule` pairs,
    and the wave test split into vertex- and cell-centered halves.
    *(Done.)* See "**Implemented in M8a step 1**", "**step 2**" and
    "**step 3**" under [Centerings](CODE.md#centerings) for what the design
    left open and the implementation settled. Nothing changed a measured
    cell-centered number, as predicted: the whole suite passes at one
    and eight threads with the M3 wave tables unchanged, and the
    thread-independence digests still agree byte for byte. The
    centering's own acceptance tests pass — the M2 exactness claim over
    all `2^D` centerings in `D = 1, 2, 3` at `p = 2` and `p = 4`, the
    write-count partition of the stored points including the shared
    plane and with `G = 0` along a stagger, bit-for-bit injection on
    position-determined data (with the cell-centered control showing
    *no* coincident points across levels at all), the `G ≥ p/2 − 1`
    bound, the conservative refusal, and the regrid transfer exact for
    every centering — and so does the wave equation on top of them: the
    predicted rates **1 and 2** at prolongation orders 2 and 4, measured
    0.99 / 1.99 in `D = 1` and 1.01 / 1.99 in `D = 2`, at **`G = 1`**,
    with the restriction order and the second ghost plane both
    bit-for-bit inert, the M4 pulse tracked to the uniformly finest
    mesh's accuracy (0.0240 against 0.0233, at 176 cells against 256),
    and two staggered cycles in the thread workload. The table is under
    [Operators](CODE.md#operators).
    *Accept:* the M2 exactness test (degree `p − 1`, face/edge/corner,
    three levels) over all `2^D` centerings in `D = 1, 2, 3`, the oracle
    averaging along cell-like dimensions and sampling along vertex-like
    ones *(amended in M8a step 2: the averaging half has no consumer,
    since the conservative family is refused along a stagger — see
    [Centerings](CODE.md#centerings))*; vertex restriction bit-for-bit
    injection on arbitrary data; **the wave equation is vertex-centered
    from here on**, with the M3 study
    repeated on it — predicted rates **1 and 2** for prolongation orders
    2 and 4 at *any* restriction order, since restriction is exact, and
    `G = 1` sufficient at order 4 where cell centering needs 2 — while
    the cell-centered study is kept verbatim as `wave_cell` so the M3
    numbers stay under test; the thread digests. The cell-centered
    stencils are the same rational weights as before, so M8a changes no
    measured number. *(All of this is measured; see above.)*
  - **M8b — conservation.** *(Done.)* `InterfaceSchedule` /
    `restrict_interfaces!`
    *(step 4; see "**Implemented in M8b step 4**" under
    [Conservation](CODE.md#conservation-at-coarse-fine-faces))*,
    `map_blocks!(…; closed = true)`, and Burgers' equation
    `∂ₜu + Σ_d ∂_d(u²/2) = 0` on a periodic box in `test/burgers.jl`:
    finite volume, linear reconstruction (unlimited for the smooth
    studies, so the measured rate isolates the interface and not a
    limiter's clipping at extrema; minmod wherever there is a shock),
    Rusanov flux, `SSPRK33` from `OrdinaryDiffEqSSPRK` (a new test
    dependency — conservation to roundoff holds for any Runge–Kutta
    method, since every stage's `du` sums to zero, but only a
    strong-stability-preserving one keeps the shock monotone), `G = 2` on
    the state and `G = 0` on the fluxes, a gradient criterion through
    `firing_boxes` with the M4 travelling margin. Data depending on
    `s = Σ_d x_d` solve the one-dimensional equation in `s` with speed
    factor `D`, so `u = u₀(s − D·u·t)` with `u₀ = ū + a·sin(2πs/L)` is
    exact in any `D` up to `t_b = L/(2πaD)`, solved per cell by Newton and
    compared as *cell averages*; after `t_b` a shock travels at `D·ū`.
    *Accept:* **total mass conserved to roundoff across coarse-fine
    faces** — a shock crossing a refined region that follows it, regrids
    in between, `|Δ Σ h^D u| ≤ c·eps(T)·Σ h^D|u|·nsteps` — with the
    negative control (fixup skipped) measured and its drift recorded
    here; the interface-order rule for the conservative family measured
    (predicted rates 1, 2, 2 for `p = 1, 3, 5`) and tabulated under
    [Operators](CODE.md#operators); a tracked shock matching the uniformly fine
    reference at fewer cells, as the M4 pulse did; the interface
    restriction equal to a hand-computed average over the finer
    neighbors, and each target written exactly once per phase; the
    Burgers cycle in the thread-independence workload and in the device
    suite. *(All of this is measured, in M8b step 5;
    `test/burgers.jl` and `test/burgers_tests.jl` are the study.)*
    **Mass is conserved to 0.5-2.5 ulp** of the domain integral, and the
    drift does not grow with the step count; the negative control leaks
    3.8e-5 to 3.2e-4 in the same runs, eleven orders of magnitude more,
    with the full table and the `Float32` caveat under
    [Conservation](CODE.md#conservation-at-coarse-fine-faces). A **uniform**
    mesh conserves with or without the fixup, which is what pins that on
    the coarse-fine faces rather than on the scheme. The interface-order
    rule comes out as predicted — L∞ rates **1.00 / 1.97 / 1.97** in
    `D = 1` and **0.84 / 1.81 / 1.77** in `D = 2` for `p = 1, 3, 5`,
    with `p = 3` and `p = 5` matching the *unrefined control's* own rate
    — and the **norm turned out to be part of the result**: an integral
    norm sees none of it, because a flux divergence leaves the defect on
    the interface instead of radiating it as a second derivative does
    (both under [Operators](CODE.md#operators)). The tracked shock matches the
    uniformly fine reference at **6.6e-4** against that mesh, where the
    uniform coarse mesh is **5.2e-3** — 7.9x worse — using 80 cells
    against 128, and with the travelling buffer removed the refined
    region falls off the shock entirely. The Burgers cycle is in the
    thread workload (bit-identical at 1 and 8 threads) and in the device
    suite, where Metal in `Float32` reproduces the CPU numbers bit for
    bit.

### M10 — Reflecting boundaries

- **M10 — Reflecting boundaries.** *(Done.)* Done before M7, for M8's reason: M7
  then distributes mirror transfers as the ordinary transfers they are,
  rather than retrofitting them. `reflecting` per face on the forest,
  `parity` per variable and dimension on the field set, and the mirror
  transfers and the derived upper wall point under
  [Ghost filling](CODE.md#ghost-filling). Until now a reflection could only be
  written as a region-form boundary hook. That form is CPU-only, has to
  repeat the mirror index arithmetic per centering, and reads stale
  ghosts at a mixed edge or corner region whose tangential neighbor is
  coarser. *Accept:*
  - refusals, each saying why: periodic and reflecting in one dimension,
    a missing parity, `NoParity` in a reflected dimension;
  - the M2 exactness test with data even or odd about the wall, of
    per-dimension degree `p − 1`, over three levels touching the wall,
    at `p = 2, 4`, for every centering. It covers the derived upper wall
    point: zero for odd data, exact for even. The conservative family
    is tested on cell averages;
  - a reflecting domain reproducing the mirrored doubled domain on
    arbitrary data, to roundoff, including a coarser tangential neighbor
    at the wall;
  - every ghost written exactly once, the hook never handed a
    reflecting region;
  - **no ghost undefined, and none read before it is defined**, for
    every combination of periodic, outer and reflecting faces in
    `D = 1, 2, 3` and every centering: `NaN`-prefilled storage with
    finite owned data, checked for `NaN` after one fill;
  - the regrid transfer exact at a wall;
  - the wave equation with odd and with even data on a reflecting half
    domain agreeing with the full domain, and converging at the
    predicted rate at a vertex-centered upper wall;
  - a reflecting cycle in the thread workload and in the device suite.

  *(All of this is measured; `test/reflect_tests.jl` is the study, with
  its oracles at the end of `test/ghost_oracles.jl`.)* The design needed
  no amendment: the mechanism above went in as specified, and every
  mirror point landed inside the tangential stencil's range, as
  `N ≥ 2G + 2c` promises. Four things are worth recording.

  - **The doubled domain is reproduced to roundoff.** It is bit for bit
    in `D = 1`, and within 4.4e-16 in `D = 2` and 1.6e-15 in `D = 3` over
    up to 221184 stored values, on arbitrary data. That holds cell-centered
    at either wall and vertex-centered at the low one. The residue is
    summation order: a mirrored stencil adds the same terms as the
    doubled domain's, reversed.
  - **The `NaN` test catches the ordering bug it is there for.** It
    covers 235 cases: 10 in `D = 1`, 100 in `D = 2`, and in `D = 3` the
    125 face combinations with the centering rotating. All are clean, and
    the values are exact to 1.2e-14. As a check that the test is not
    vacuous, mirrored prolongations were moved into phase 1, where the
    hook used to run: it then failed 39 to 41 of the 42 2D cases with a
    reflecting face, and all 234 in 3D. 1D has no mirrored prolongation
    to misplace. The 2D count varies between processes because the order
    of the groups within a phase does: a misplaced prolongation reads its
    source stale only if it runs before the group that fills it. That
    order is free precisely because phase 1 is order independent. It costs about 11 s warm, most of it the 3D sweep.
  - **The wave equation keeps its rate.** Order-4 operators on a box
    with reflecting walls at `x₁ = 0` and `L/2`, `N = 8, 16, 32`, give
    L2 rates **1.91 / 2.02** (vertex, odd / even) and **1.99 / 2.00**
    (cell) in `D = 1`, and **1.96 / 1.99** and **1.99 / 2.00** in
    `D = 2`. The periodic box with the same refinement gives 1.99–2.00.
    The derived upper wall point therefore costs no order, as the
    interface-order rule predicts. The lowest of the eight, vertex-odd in
    1D at 1.91, is inside the 0.15 the periodic studies are held to; its
    L∞ rate is 1.95. Why that one case sits lowest was not
    investigated.

    The half box also evolves as the periodic box it folds.
    Cell-centered, at `N = 16`, the L∞ errors of the two agree to a
    relative 3.0e-11 (`D = 1`) and 2.6e-13 (`D = 2`) after 128 and 91
    RK4 steps, on half the blocks.
  - **Cost.** The suite went from 89168 tests in 3m31 to 89961 in 4m49,
    at 8 threads. The M10 testsets add up to about 42 s of that, and the
    device subset (Float32 compilation on the CPU backend) and the two
    reflecting cycles in each thread-workload subprocess about 8 s more.
    The rest is inside the run-to-run noise at this length. On Julia 1.10,
    in one thread, it was 97744 tests in 2m53. The 3D order-4 exactness
    sweep was then dropped as a duplicate of the `NaN` test's, which left
    89836 tests in 3m37 in one thread on the current Julia.

### M11 — Point interpolation

- **M11 — Point interpolation.** *(Done.)* `interpolate` and
  `locate_point`, as specified under [Point
  interpolation](CODE.md#point-interpolation). Asked for by
  TreeGeneralizedHarmonic, whose apparent-horizon provider carried a
  stopgap — 3D, vertex-centered, host-only, `maxlevel` searches per
  point — and listed it as its first upstream prerequisite; porting it
  onto this is that package's next step. *Accept:*
  - `locate_point` agreeing with exact rational leaf boxes (half-open,
    the upper domain face to the last leaf, a periodic upper face to the
    first) on random forests in `D = 1, 2, 3`, on the finest lattice as
    well as at random;
  - exactness on tensor polynomials of degree `n − 1`, value and every
    first derivative, through three levels, at nodes, on block faces and
    at the domain's corners, for `n = 2…6` in `D = 1, 2` and `n = 3, 4`
    in `D = 3`, for cell, vertex and face centering; and at `G = 0` and
    `G = 1`, where the stencil shifts;
  - not exact one degree higher, and converging at rate `n` (value) and
    `n − 1` (gradient);
  - continuity inside a block on random data, across nodes and
    midpoints;
  - a point one period away agreeing to roundoff; beyond a reflecting
    face the parity-signed mirror, gradients included, exactly for data
    of definite parity, over six combinations of face kinds and
    centerings;
  - the ellipsoid test agreeing with enumeration bit for bit, and the
    flags through `interpolate` agreeing with the stencils
    `query_stencil` names;
  - the kernel's value equal to the contraction, with exact rational
    weights, of the block's own stored points over that stencil, on
    random data;
  - refusals with reasons; `Float32` and `Float32x2` exact and never
    widened; device agreement with the host; a line in the thread
    workload.

  *(Measured 2026-09-25; `test/interpolate_tests.jl`, `bench/interpolate.jl`.)*
  The design needed no amendment. Five things are worth recording.

  - **Allocation and hidden arithmetic, audited.** For `Float64` and
    `Float32` a batch allocates a fixed ~3 KB of host setup (the origins
    and spacings vectors, the variable list) plus the 4-byte block index
    per point, and nothing per point in the kernel; the kernel's LLVM
    IR calls no `Rational`, `BigInt` or `BigFloat` code and no generic
    dispatch. `Float32x2` did not pass at first: 535 bytes and 15 µs per
    point, from `pointoffsets` (`storage.jl`) converting `1//2` through
    `BigFloat` at run time — which every position-forming kernel in the
    package shares, `fill_by_coordinates!` included. It is now `one(h) /
    2`, the same value in every binary type, and `Float32x2` allocates
    what `Float64` does; with the innermost closures marked `@inline` it
    takes 12 µs per point, 14 times `Float64`, against 10.5 times for a
    bare double-float multiply-add loop.

  - **Rates.** A smooth 2D field, `N = 8 → 16`, over a three-level
    mesh: value **2.96 / 3.20** at `n = 3` (cell / vertex), **3.97 /
    4.08** at `n = 4`, **5.04 / 5.11** at `n = 5`; gradient **1.79 /
    2.06**, **3.05 / 3.27** and **3.86 / 4.01**. The suite holds them to
    `n − 0.4` and `n − 1.4`.
  - **Cost against the stopgap** (`bench/interpolate.jl`,
    `bench/symmetry_interpolate.sh`; the stopgap is a verbatim copy of
    TreeGeneralizedHarmonic's core run on the same batches). The
    horizon finder's workload: points in a shell through three levels,
    20 variables, value and gradient, `Lagrange(4)`, vertex-centered with
    `G = 2`, 176 leaves, a 249 MiB working array; best of 50 whole
    `interpolate!` calls. On a Symmetry AMD EPYC 7532 node (64 cores, 8
    NUMA domains, threads pinned), in ns per point:

    | threads | 496 points | stopgap | 49600 points | stopgap |
    |---|---|---|---|---|
    | 1 | 2083 | 1926 | 2214 | 1904 |
    | 8 | 356 | 312 | 267 | 258 |
    | 16 | 256 | | 135 | |
    | 64 | 288 | 643 | 42.1 | 43.9 |

    So the finder's own batch takes **0.13–0.14 ms from 16 threads up**,
    against the stopgap's 0.32 ms at 64; a large batch scales 53-fold
    to 64 threads. Serially this is 8–16 % behind the stopgap on the
    EPYC, and 25 % ahead on an Apple laptop (781 against 1042 ns): the
    stopgap accumulates all 20 variables in one `SVector`, which AVX2
    vectorizes across the variables, where this contracts one variable
    at a time. Contracting the variables in static chunks is the
    recorded way to close that if it ever matters; at eight threads the
    gap is 3 %, and at 64 it has reversed. Three measured causes were
    fixed on the way: a `Val` for the derivative multi-indices (run-time
    ones made the selections dynamic tuple indexing, 21 % serially), the
    closure boxing above, and a floor of 32 points per CPU task (the
    496-point batch took 0.29 ms at 64 threads without it, 0.14 at 16).

    **NUMA does not matter here.** A point is not owned by any thread,
    so a query reads its block from wherever first touch put it — and
    eight threads bound to one domain with their memory there run at
    the pinned eight-thread rate (264 against 267 ns), and 64 pinned
    threads beat `numactl --interleave=all` (42.1 against 49.7). A batch
    touches too little of the array for remote reads to show.

    `locate_point` is 113 ns per point on the laptop and 125 on the
    EPYC, against 249 for the per-level descent. The allocation is a
    fixed ~10–20 KB of host setup per batch plus four bytes per point,
    against the stopgap's 890 bytes per point.
  - **Suite cost.** 92054 tests in 4m35–4m54 at eight threads after M11,
    against 4m49 for M10 (within the run-to-run noise at this length);
    the new file takes about 20 s on its own, almost all of it
    compiling one kernel per (dimension, order, centering) case, and
    the thread-workload subprocesses about 2 s each. On Julia 1.10, in
    one thread, 99962 tests in 3m10. The device suite passes on Metal
    (`Float32`) and on an H200 (`bench/symmetry_gpu.sh`: 93348 tests,
    `Float64` and `Float32`, in 10m09), the interpolation's agreement
    with the host included.
  - **On a device** (one H200, `bench/symmetry_interpolate.sh cuda`), the
    same batches: 496 points in **0.26 ms** (`Float64`) and 0.28
    (`Float32`), which is latency — the small uploads of the leaves and
    the geometry, the launch, the on-device check for outside points,
    the synchronization — and so no better than the node's 16 host
    cores (0.12 ms). Large batches are throughput: **6.4 ns per point**
    in `Float64` and 4.7 in `Float32` at 496000 points, against 178 on
    the 16 cores. For a device-resident run the comparison that matters
    is with the stopgap's `hostcopy` of the whole field set per find;
    caching the uploads per forest generation would take the small
    batch down further, and is not done. The host allocates a constant
    17 KB per batch: the outside check is reduced on the device, so only
    a batch with an outside point copies its block indices back (the
    first version copied them always, 2 MB at 496000 points, and ran at
    12.2 ns per point before the `Val` for the multi-indices).
  - **Second derivatives** *(amended 2026-10-03)*. The check moved from
    `|m| ≤ 1` to `|m| ≤ 2`, and the tests now claim: every second
    derivative of a degree-`(n − 1)` tensor polynomial, through the same
    three levels, centerings and `n` as the first, and at `G = 0` and
    `1` where the stencil shifts — worst error 7.4e-12 against
    8e-14 for first derivatives, the `h⁻²` the roundoff picks up, held
    to 1e-9; the Hessian beyond a reflecting face in the six face and
    centering cases; the kernel against the exact rational contraction
    with the basis polynomials' exact derivatives (expanded, not the
    package's series); `(2, 0)` and `(1, 1)` in `Float32` and
    `Float32x2` (at most 1302 `eps(T)` of the value scale, held to
    16384); `∂ₓ²` on a device against the host; the Hessian through
    the in-process distributed path and `∂ₓ∂ᵧ`, `∂ᵧ²` in the MPI
    workload; and the refusal moved to total order 3. `(1, 1)` through
    `Lagrange(2)` is allowed, being first order along each dimension.
    *Rates*, the same smooth 2D field, `N = 8 → 16`, cell / vertex:

    | `n` | `p` | `∂ₓ²` | `∂ₓ∂ᵧ` |
    |---|---|---|---|
    | 3 | 4 | 0.90 / 0.86 | 1.74 / 1.76 |
    | 4 | 4 | 1.71 / 2.08 | 3.09 / **2.14** |
    | 4 | 6 | 2.01 / 2.08 | 3.09 / 2.80 |
    | 5 | 6 | 2.99 / 2.93 | 3.80 / 3.85 |

    The bold entry is the exchange's limit, `p − |m| = 2`, and is why the
    rate is written with `p` in it: raising `p` to 6 gives `∂ₓ∂ᵧ` back
    its `n − 1`. At `N = 16 → 32 → 64` the same case runs 1.69, 2.50
    at `p = 4` and 3.05, 2.96 at `p = 6`, and the `n = 3, 5` rows do not
    move with `p`, since `p − 2` does not limit them there. The suite
    holds `∂ₓ²` to `n − 2.4` and `∂ₓ∂ᵧ` to `min(n − 1, p − 2) − 0.4`.
    *Cost*: the interpolation file takes 35.4–36.0 s against 33.6–34.9
    for the previous one on the same `src/` (two runs each, Julia
    1.13), the extra `M = 2` kernels; the suite 110993 tests in 7m27 at
    eight threads, against 110658 in 6m28 after M7 step 6b, which is
    run-to-run noise at this length since the file accounts for 1 s of
    it. The device test passes on Metal (`Float32`, `∂ₓ²` included).

### M9a — Checkpoint and restart

- **M9a — Checkpoint and restart.** *(Done.)* Done before M7
  (decided): TreeHydro's long runs and TreeGeneralizedHarmonic's
  production runs, estimated at 38–149 h, outlast any queue's day and
  need to stop and resume before they need MPI, and
  TreeGeneralizedHarmonic and TreeGRRMHD were waiting for M9.
  Serial. A forest from a validated leaf list, and `save_checkpoint`,
  `load_checkpoint`, `write_plain` / `read_plain` and
  `checkpoint_environment` in the package extension `TreeAMRHDF5Ext`,
  as specified under [Checkpoint and restart](CODE.md#checkpoint-and-restart).
  *Accept:*
  - the forest from a leaf list: random refinement patterns round-trip,
    checked against the oracles, and a gap, an overlap, a duplicate, an
    unsorted list, an unbalanced list and a root out of range are each
    refused with the reason;
  - **the round trip is bitwise**, in `D = 1, 2, 3`, for cell, vertex,
    face and edge centering, in `Float64`, `Float32` and `Float32x2`,
    with periodic, reflecting (with parity) and outer faces: after
    `scatter!` and `fill_ghosts!`, the leaves, the forest's parameters,
    the field-set metadata, the state vector byte for byte and the
    working arrays all equal the saved ones;
  - **a restart continues bit-identically through regrids**: a chunked
    driver saves after chunk `k`, drops every object, loads into fresh
    ones and continues, and its leaves and state vector equal an
    uninterrupted run's byte for byte. This is checked for the
    vertex-centered wave study (`test/wave.jl`, RK4) and for the
    conservative Burgers study (`test/burgers.jl`, `SSPRK33`, the
    interface fixup), whose face flux sets with `G = 0` are not saved.
    The time is part of the plain data;
  - refusals, each saying why: a newer `format_version`, an unknown
    feature, a `Float32x2` file loaded without `types`, a field set
    over another forest, a value outside the plain-data types;
  - an atomic write: a do-block that throws leaves the earlier file
    intact and no `.partial` file behind;
  - plain data round-tripped exactly (`Rational`s, `Symbol`s, nested
    NamedTuples and Dicts, tuples, arrays, `nothing`,
    `VersionNumber`s), and `checkpoint_environment` writing the
    `Project.toml` and `Manifest.toml` that were stored;
  - a device round trip in the device suite;
  - **throughput measured** (`bench/checkpoint.jl`): save and load rates
    and file sizes, uncompressed and with `Shuffle` + `Deflate(1)`, and
    with zstd and bitshuffle where those packages are available, on a
    smooth wave pulse and on a Burgers shock, recorded here with the
    recommended filter chosen from them;
  - TreeWave and TreeHydro still green against it, the change being
    additive.

  *(Measured 2026-09-29; `test/checkpoint_tests.jl`,
  `bench/checkpoint.jl`.)* The design needed one amendment, the name a
  limb type is recorded under (see "Element types" under
  [Checkpoint and restart](CODE.md#checkpoint-and-restart)); the
  implementation fixed the spellings the design had left to it, and
  chose `rename` over `mv(…; force = true)`, which on Julia 1.11 removes
  the old file before the new one is in place. Four things are worth
  recording.

  - **Restarts are bit-identical, and the test would see it if they
    were not.** Both studies continue byte for byte through regrids
    after the restart point, and both fail on a one-ulp error in the
    restored time, which was checked. The device round trip passes on
    the CPU backend in the suite, and on Metal in `Float32` (the testset
    run on its own).
  - **Throughput** (the table and the reasons are under "Throughput and
    filters" in [Checkpoint and restart](CODE.md#checkpoint-and-restart)). On
    the development laptop an unfiltered save reaches the page cache at
    5–7 GB/s and stable storage at 3–5, and a load of the 540 MB state
    runs at 3.6 GB/s on six threads, half of that time the HDF5 read. The
    recommendation is no filter, and `Shuffle()` with `ZstdFilter(1)`
    when size matters: 6.1-fold at 1.0 GB/s on a state that is mostly
    atmosphere, and 1.3-fold on a smooth one.
  - **Suite cost.** 93686 tests in 4m16 at one thread and 93738 in 4m47
    at eight threads, against 92054 in 4m35–4m54 at eight after M11;
    93686 in 3m48 on Julia 1.11.9 (with `sync`, 2026-09-29; 93677,
    93729 and 93677 before it). `checkpoint_tests.jl` (541 tests) adds
    about 30 s at one thread inside the suite, and takes about 57 s on
    its own, most of it compilation.
  - **The downstreams are green.** Against the M9a checkout, developed
    into scratch copies, at one thread: TreeWave 310 tests in 1m12, and
    TreeHydro 11893 in 4m16. Neither calls the new functions yet; each
    gains them with `using HDF5` once a release carries them.

### M7 — MPI

- **M7 — MPI.** *(Done, 2026-10-02. Measured on Symmetry the same day:
  the weak-scaling table of step 7 at up to four nodes and the H200 run
  of step 8, both paths, pass. The checkpoint of step 6, measured on
  four nodes the same day, lost data on BeeGFS until ROMIO's
  read-modify-write was turned off, and was replaced the same day by
  one without parallel I/O — part files per I/O process and an index
  file, decided with Erik — which is step 6b, measured on one, two and
  four nodes with every save verified and 1000 stress saves without
  damage. Left open, as later work rather than as part of the
  milestone: the items under "Performance work left for later" in
  [Distributed meshes](CODE.md#distributed-meshes) — the H200 re-measurement
  of the staged regrid with the buffer pool, `interpolate!` at 32 ranks
  under Open MPI, compression at the members under `io = :node`, the
  pool's own rules, and what only many more ranks will show.)* Curve partitioning, distributed ghost exchange (for
  every centering, and the interface restriction with it, since both are
  transfers over the same schedule machinery), distributed regridding,
  and the `Allreduce` inside `mesh_mapreduce` (the planned global
  reduction under [Parallelism](CODE.md#parallelism)), the one place a
  communicator appears in a reduction. *(Amended 2026-10-01: the
  reduction is an `allgather` of per-rank partials folded in rank
  order, not an `Allreduce`, and parallel checkpoints are part of M7;
  both are under [Distributed meshes](CODE.md#distributed-meshes), which
  specifies the milestone. Interpolation routing is added with them.)*
  *Accept:* results match serial to roundoff — bit-identical for
  everything but floating-point sums, as [Parallelism](CODE.md#parallelism)
  states; weak-scaling smoke test; then MPI+GPU with CUDA-aware MPI. In
  steps, each ending green and committed, with what it measured recorded
  here and in the commit body. Every step runs the full suite at one
  thread and at eight, with the thread-independence digests byte for
  byte, and from step 3 on the MPI tests with it:
  - **Step 0 — specification.** *(Done, 2026-10-01.)*
    [Distributed meshes](CODE.md#distributed-meshes), after two feasibility
    checks recorded there: the stock HDF5_jll is a parallel build and
    works under `mpiexec` with MPICH_jll on Julia 1.11 and MPIABI_jll
    on 1.13, and an `mpiexec -n 3` subprocess costs about a serial
    workload and a quarter.
  - **Step 1 — local blocks, no messages.** `communicator.jl`, the
    forest's `comm` field and keyword, the split helper shared with
    `threadchunks`, and `blockrange`. Every block-index site becomes
    local: allocation, `blockkey` and the coordinate and fill origins in
    `storage.jl`; `block_origins` and `block_spacings`; the schedule's
    block count and `BoundaryPlan`; the block-count checks in
    `ghosts.jl` and `interfaces.jl`; `combine_blocks` and the volume
    loop in `state.jl`; `flag_blocks`, `regrid!`, the scratch forest and
    `firing_boxes` in `regrid.jl`; and `interpolate.jl`. *Accept:* the
    whole suite passes unchanged. New in-process tests, through a
    test-only `PartitionCommunicator(rank, size)` that answers rank and
    size and sends nothing, check the partition: the ranges tile
    `1:nleaves` in order and differ in length by at most one, ranks
    beyond `nleaves` are empty, and `blockkey` on every rank names the
    leaf that rank owns. *(Done, 2026-10-01.)* What it settled, and
    where it went beyond the plan:
    - *The split.* `equalsplit(n, p, i)` in `threading.jl`, the closed
      form of `threadchunks`' old loop, is the one helper:
      `threadchunks` returns its parts, and `blockrange(forest)` is part
      `commrank + 1` of `commsize`. The test compares it against the old
      loop for every `n ≤ 40` and `p ≤ 9`, and `threadchunks` against it
      at the suite's thread count.
    - *The keyword.* `Forest(…; comm = nothing)`, and
      `communicator(nothing)` is the `SerialCommunicator`, so the default
      is spelled the same way the MPI case will be. A `Communicator`
      subtype lacking a verb is refused at that verb by name, through a
      fallback method on the abstract type, rather than with a
      `MethodError` inside a reduction; that is what lets the test-only
      communicator answer rank and size alone. The serial `isend` and
      `irecv` refuse any peer, since a serial rank has none, and the
      serial `waitall` accepts only an empty list.
    - *The geometry per local block.* `block_origins(forest)` and
      `block_spacings(forest)` take the forest, not a field set, and are
      per local block through `blockrange`, so every kernel that reads
      them is unchanged. Tree queries — `nleaves`, `find_leaf`,
      `neighbor_keys`, `locate_point` — stay global, and `flag_blocks`
      walks the local range and passes the local index.
    - *The schedules build their local part* (amended: the plan had them
      untouched until step 2). `GhostSchedule` and `InterfaceSchedule`
      walk the rank's own targets in global leaf indices and then keep
      the transfers with both ends on the rank, shifted to local
      indices, together with the boundary regions of the rank's blocks.
      That is step 2's *local* class already; the test checks it against
      the serial schedule restricted to each rank's range, mirrored
      transfers and three levels included, over 2 and 3 fake ranks.
    - *What refuses a distributed forest until its step*: `fill_ghosts!`
      and `restrict_interfaces!` (step 3), `regrid!` (step 4),
      `interpolate` (step 5) and `save_checkpoint` (step 6), each with an
      `ArgumentError` naming the step; a reduction is refused by the
      missing `allgather` of a communicator that lacks one.
    - *The reduction is already written* (amended: the plan put it in
      step 3). `combine_blocks` gathers `(hasvalue, partial)` per rank
      and folds the ranks that have a value, in rank order, as
      "Reductions" above specifies — but at one rank it returns its own
      fold without calling `allgather`, so the serial value is unchanged
      by construction and a serial reduction keeps accepting a
      non-`isbits` partial, which `allgather` refuses. The multi-rank
      fold is first exercised by step 3's workload.
    - *Empty ranks.* `launch_by_owner!` skips an empty `ndrange` on a
      device too, `firing_boxes` returns an empty vector, and the
      `AllVariables` fill returns before checking its callback at a
      block that does not exist.
    - *Docs.* The `docs/src/api/distributed.md` page that step 9 was to
      add exists now, with `communicator`, `blockrange` and the internal
      verbs, since Documenter refuses a docstring that is on no page.
    - *Suite cost.* 98818 tests in 4m44 at one thread and 98870 in 4m51
      at eight; the difference from M9a's 93686 and 93738 is exactly
      `partition_tests.jl`'s 5132, so no existing test moved. The docs
      build.
  - **Step 2 — the distributed schedule.** The candidate remote targets,
    the three classes, the sorted layouts, the stages, and the
    packed-buffer accessors. *Accept,* in process, over 1–5 fake ranks,
    in `D = 1, 2, 3`, with periodic, reflecting and outer faces, every
    centering, three levels and the interface schedule:
    - the union over ranks of the local, send and recv transfers
      equals the serial schedule's transfer set exactly;
    - rank `r`'s send layout to `s` equals `s`'s receive layout from
      `r`;
    - every target is written exactly once;
    - a pack-then-unpack round trip, with every rank's buffers wired
      directly in one process, reproduces the serial `fill_ghosts!`
      bitwise, in `Float64`, `Float32` and `Float32x2`, including an
      odd variable that is zero at a reflecting wall (the `−0` case).

    *(Done, 2026-10-01.)* What it settled, and where it went beyond the
    plan:
    - *The data structures.* `GhostSchedule` keeps `phase1`, `phase2`
      and `levels`, which now hold the transfers *local* to the rank,
      and gains `stages`, a vector of `ExchangeStage`s in tag order:
      phase 1, then one per phase-2 target level. A stage holds its
      `tag`, its local groups and a `remote` part, which is `nothing`
      when the stage has no messages on this rank. Serially every stage
      is `nothing`-remote and its local groups *are* the phase (the same
      vector). `InterfaceSchedule` gains `stages` the same way, one per
      face dimension. A `RemoteStage` holds, per direction, the peers
      ascending with their segment lengths in points; one `LayoutEntry`
      per buffer slot (peer, `GroupKey`, global target and source,
      offset and size in points), kept on the host; the pack and unpack
      groups; the slot offsets on the backend; and the buffers. A
      `GroupKey` has an explicit total order (`keyorder`: kind ranked,
      then direction, offset, level, mirror state), so a layout never
      depends on `Dict` order.
    - *The tags*: phase 1 is 1, the level-`ℓ` stage `2 + ℓ`, interface
      dimension `d` is `40 + d`, and 50 is reserved for the regrid.
      They ascend in the order the stages run.
    - *The classification.* `split_received!` splits the transfers
      found for the rank's own targets into local ones, which are
      shifted to local indices, and received ones, kept global.
      `sent_transfers` runs the builder's own search (`block_sources!`
      or `interface_sources!`) for every candidate remote target,
      `remote_neighbors` in `forest.jl`, and keeps those with a local
      source. `leafowner` inverts the split in `O(1)`, through
      `equalsplit_part` beside `equalsplit`. `remote_stage` builds one
      stage's messages. It takes the target and source owners and
      ranges separately, so that step 4's regrid stage — targets in the
      new partition, sources in the old — is a call of it, and is not
      built here.
    - *Packs and unpacks.* A pack is the serial group's stencils with
      every target range moved to start at 1, `targetblocks` the slots,
      `sourceblocks` the local sources, sorted by (source, slot), and
      no parity column. An unpack is a width-1, weight-1 group from the
      slot into the serial target box, sorted by (target, slot), with
      the parity column. On the CPU, `run_phase!` bisects
      `sourceblocks` when the destination is a packed buffer, through
      `ownerblocks` and `ownercount`; the device path is the per-group
      one, unchanged.
    - *The driver.* `run_stage!` runs the five steps through the
      communicator verbs. `exchange_ghosts!` and `exchange_interfaces!`
      are the bodies of `fill_ghosts!` and `restrict_interfaces!` after
      their checks, and a serial fill now goes through them. The two
      public functions still refuse a distributed forest, so step 3
      removes the refusal and adds the MPI methods of the verbs.
      `run_stage!` takes the forest rather than its communicator: the
      `comm` field is abstractly typed, and passing it made every stage
      of a serial fill a run-time dispatch of 48 bytes (measured, then
      removed).
    - *Tests*, in `test/exchange_tests.jl`. They cover 1–5 simulated
      ranks (and `nleaves + 1` in 1D, which leaves a rank empty), `D =
      1, 2, 3`, periodic, outer and reflecting faces, `faces_forest`'s
      three levels, every centering, both operator families, and the
      interface schedule over every centering with a face dimension.
      They check the candidates against the serial schedule's readers.
      They check the union of the local transfers and the matched
      sent and received halves against the serial set, per stage, with
      every slot consistent with its layout entry. They check both
      ends' layouts entry by entry and the per-stage write counts
      against the serial ones, at most one write per point. They run
      the bitwise lockstep round trip, the boundary hook between the
      stages included: `Float64` over every centering, `Float32` and
      `Float32x2` over a subset, and the interface restriction in
      `Float64` and `Float32x2`. Finally the staged driver itself runs
      over an in-process mailbox communicator, one task per rank, in
      `D = 2, 3` at 3 and 5 ranks, for the ghosts and the interfaces,
      bitwise against serial. `gpu_tests.jl` runs the lockstep round
      trip on every backend; on Metal (Float32) it passes, with the
      rest of that file.
    - *Serial cost* (`bench/ghosts.jl` at its defaults: `D = 3`,
      `N = 8`, 4³ roots, 10 variables, order 4; best of 400 fills and 40
      builds, HEAD before and after, alternated). The fill is unchanged
      at one thread: 2.71–2.78 ms against 2.71–2.73 ms on the uniform
      mesh, 11.8–14.5 ms against 12.0–15.0 ms on the two-level one, with
      identical allocations (10912 and 97536 bytes). At four threads
      the uniform fill measured 0.51–0.52 ms against 0.64–0.65 ms; the
      cause was not traced. The schedule build is within noise,
      0.40–0.43 ms against 0.41 ms and 3.0–3.3 ms against 3.0–3.2 ms.
      It allocates 2.8 % more on the uniform mesh (389 KB against 379 KB)
      and 3.1 % more on the two-level one (4.65 MB against 4.51 MB),
      nearly all of it the per-key stencil memo below.
    - *Distributed build cost* (one thread, `D = 3`, `N = 8`, `G = 2`,
      order 4, the bench's two-level periodic mesh, slowest rank of `P`
      fake ranks). At 120 leaves the serial build is 3.4 ms and a rank's
      is 3.6, 3.1 and 2.6 ms at `P = 2`, 4 and 8. At 960 leaves (8³
      roots) the serial build is 9.7 ms and a rank's 11.0, 9.2 and
      7.3 ms. So at these sizes a rank's build costs about the serial
      one, not `1/P` of it. A profile of one of the eight ranks at 960
      leaves puts a third of the time in the search for the sent
      transfers and another third in building the remote stages. At
      120 blocks a rank, a 3D halo is about as many blocks as the
      rank's own, so that is the `O(local + halo)` the spec predicts,
      not a defect. It is a regrid-frequency cost, about 1,300–2,100
      sent transfers a rank here. The host stencils are memoized per
      group key, which the stage builder asks for again (10 % of the
      rank build).
    - *What was not done.* The regrid's stage, which is step 4, has
      only its builder's interface, as above.
    - *Suite cost.* 108338 tests at one thread, in 5m29 and 5m51 over
      two runs, and 108390 at eight, in 5m35 and 5m54. The increase over
      step 1's 98818 and 98870 is exactly the 9506 tests of
      `exchange_tests.jl` and the 14 of the CPU entry of the new
      `gpu_tests.jl` testset (7380 of the 9506 check the owner inverse),
      so no existing test moved. The thread-independence digests are
      unchanged. The new file takes about 50 s standalone, almost all of
      it compilation of the `Float32` and `Float32x2` kernels and of the
      3D centerings; that is most of the 45–65 s the suite grew by, and
      what to trim first if the suite has to shrink. The docs build.
  - **Step 3 — the MPI extension and the exchange.** `TreeAMRMPIExt`,
    the staged `fill_ghosts!` and `restrict_interfaces!`, the
    `combine_blocks` allgather, and the forest digest check; MPI in
    `[weakdeps]` and `[extensions]`, and in `test/Project.toml`. A new
    `test/mpi_workload.jl`, a standalone script like
    `thread_workload.jl`, prints on rank 0 digests of the leaves and of
    the state gathered in curve order, the integer and max reductions
    exactly, and the sums to `%.17g`. A new `test/mpi_tests.jl` runs it
    under `MPI.mpiexec()` at `-n 2` and `-n 3`. *Accept:* against a
    serial run, every line byte-identical except the sum lines, which
    agree to roundoff, over the vertex and cell waves, Burgers with the
    fixup, a reflecting box, three levels, and a rank with no blocks;
    the digest check refuses a forest mutated on one rank only, on
    every rank; and what the subprocesses add to the suite is measured,
    with the rank counts trimmed if it is large.

    *(Done, 2026-10-01.)* What it settled, and where it went beyond the
    plan:
    - *The extension.* `ext/TreeAMRMPIExt.jl` defines `MPICommunicator`
      (the duplicate, with its rank and size read once) and one method
      per verb, each a single MPI.jl call: `MPI.Allgather` of the
      `isbits` value, `Allgatherv!` after a gather of the lengths,
      `Alltoallv!` after an `Alltoall` of the counts, `Isend`, `Irecv!`
      and `Waitall`. `communicator(::MPI.Comm)` refuses MPI not
      initialized (or finalized), `COMM_NULL` and a thread level below
      `THREAD_SERIALIZED`, each saying what to do; the duplicate and its
      cache are as amended under "The communicator layer". A message
      buffer must be an `Array` or a `UnitRange` view of one, which
      MPI.jl passes as a pointer and a count: the exchange's per-peer
      segments are exactly such views, and anything else is refused
      rather than sent as a derived datatype. The fill and the
      interface restriction lose their refusals and run through
      `run_stage!` unchanged from step 2; the reduction needed nothing,
      since step 1 had written it.
    - *The digest*, in `forest.jl` (`ForestDigest`, `agree_on_forest`,
      `digest_verdict`, `collective_checks`), as amended under "Every
      forest mutation is collective": a fold over every leaf, the brick,
      the layout, and a refusal flag, so that a refusal on some ranks is
      raised on all. `GhostSchedule` and `InterfaceSchedule` run their
      argument checks through `collective_checks`; `regrid!` will in
      step 4. The in-process fakes of steps 1 and 2 answer the digest
      gather alone, by replication — their ranks are copies of one
      forest — so they still refuse every other verb they lack.
    - *The workload.* `test/mpi_workload.jl` prints, on rank 0, lines
      that do not depend on the rank count: per case the leaf count, the
      maximum level and a digest of the leaves; the serial transfer
      count, as the sum over ranks of local and received transfers, and
      whether the totals sent and received agree; digests, gathered to
      every rank in block order, of the working arrays *with their
      ghosts* after a fill, of the state after the steps, and of the
      working arrays after a last fill; and the reductions — a count,
      `linf`, `min`, a `max` of negative data and a weighted `max` from a
      non-neutral `init` exactly, the norm and the mass on `sum` lines.
      The cases: the cell-centered wave in 1D on two periodic leaves,
      which leaves a rank empty at `-n 3`, and on three levels against
      outer faces; the vertex-centered wave in 2D on three levels with
      the hook; the cell-centered wave in 3D, periodic, three levels; a
      reflecting box in 2D, vertex-centered with walls at both ends
      (the derived wall plane) and cell-centered with low walls, the
      first variable odd across x₁; Burgers in 2D on three levels with
      the fixup, the fluxes digested after it and mass conservation
      asserted; and every centering through one fill and, where a
      dimension is vertex-like, one interface restriction — cell,
      vertex and both faces in 2D, vertex, a face and an edge in 3D —
      over pseudo-random data that is a function of the global leaf and
      the stored index, so that no stencil reproduces it by accident
      and the injection at a vertex-like face is not a no-op. Lines
      starting with `#` depend on the rank count and are checked on
      their own: the refusals, the duplicate cache, `alltoallv` and
      `allgatherv` with empty contributions, and the negative control.
      Without the argument `mpi` the forests are serial.
    - *The test.* `test/mpi_tests.jl` runs the serial reference in
      process (the script in a module of its own, printing into a
      buffer), then `MPI.mpiexec()` at `-n 2` and `-n 3`, one thread a
      rank, with `Base.julia_cmd()` and the active project, under a
      deadline, and compares: every line byte for byte except the `sum`
      lines, within `rtol = 1e-12`. It checks the comparison itself on
      the serial output (a flipped digest character is caught, a sum
      moved by `10⁻⁹` is caught, one moved by three ulps is not). The
      negative control rewrites one rank's received ghosts from its last
      receive buffer with one value moved by an ulp, and asserts that
      the gathered digest changes. The refusals are a `refine!` on rank
      1 only (`GhostSchedule`) and on rank 0 only
      (`InterfaceSchedule`), other operators on rank 1, and an argument
      only rank 1's checks refuse; each is raised on every rank, and the
      test asserts the reasons. In process, the verdict is tested on
      digests of real forests, and a forest over `COMM_WORLD` in a
      process without `MPI.Init` is refused.
    - *The launcher's environment.* `MPI.mpiexec()` returns a `Cmd`
      carrying an environment — the library paths of the MPI binary —
      which interpolating it into a larger command drops. The test
      launches `setenv(cmd, mpiexec().env)`.
    - *What was measured.* At `-n 2`, `-n 3` and `-n 4` with one thread
      a rank, and at `-n 3` with two, every line agrees with the serial
      run byte for byte except the `sum` lines, which differ in the last
      one or two digits of 17 where they differ at all (`l2`, `mass`;
      Burgers' mass 16 against 15.999999999999998); at `-n 1` the output
      is the serial output exactly. Each `mpiexec` run of the workload
      takes 27–29 s of wall clock at any of these rank counts, and the
      serial run 25 s, nearly all of it compilation. Inside the suite
      `mpi_tests.jl` takes 58.4 s at one thread and 58.9 s at eight
      (timed around its `include`): the serial reference in process,
      which reuses the suite's compiled kernels, and the two `mpiexec`
      runs. That is the minute "What an MPI test costs" predicted, so the
      rank counts were not trimmed.
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 resolves
      MPI.jl 0.20.27 with MPICH_jll (its global environment's preference),
      against MPIABI_jll on 1.13.1 here, where the global v1.13
      `LocalPreferences.toml` selects it; `Pkg.test` copies the merged
      preferences of the load path into its sandbox (Pkg's
      `Operations.jl`, read, not measured), so the suite's ranks run the
      same binary as its `MPI.mpiexec()` either way. `mpi_tests.jl` alone
      passes there (32 tests, 1m04), and so does the whole suite, 108376
      tests in 5m08.
    - *Suite cost.* 108376 tests at one thread, in 6m10 and 6m00 over
      two runs, and 108428 at eight, in 6m01. The increase over step 2's
      108338 and 108390 is 38: the 32 of `mpi_tests.jl` and the 16 of the
      in-process verdict test, less a net 10 in the refusal assertions
      steps 1 and 2 made of `fill_ghosts!` and `restrict_interfaces!`
      over a distributed forest (11 removed, and one added: a fill over
      a communicator that answers rank and size alone now stops at
      `irecv`, by name). The thread-independence digests are
      unchanged. The docs build.
  - **Step 4 — the distributed regrid.** The flag gather, the transfer
    stage with the repartitioning, and `adapt_to_initial_data!`.
    *Accept:* the tracked pulse and the Burgers shock through regrid
    cycles in the workload, with leaves and state byte-identical to
    serial and mass conserved to roundoff across ranks; a regrid that
    migrates blocks between ranks, and one that coarsens a block whose
    children had different owners.

    *(Done, 2026-10-01.)* What it settled, and where it went beyond the
    plan (the design decisions are recorded under "Regridding" in
    [Distributed meshes](CODE.md#distributed-meshes)):
    - *The driver.* `regrid!` runs its checks through
      `collective_checks`, gathers the canonical marks with one
      `allgatherv`, completes them replicated, and per field set fills
      the old mesh's ghosts (the distributed fill), zero-fills the new
      local array and runs the regrid stage through `run_stage!` from
      the old array into the new one, waiting for its sends before the
      next field set; then `rebuild_leaves!`. `fs => nothing`
      reallocates to the new local count, `transfer = false` moves
      nothing, and "nothing changed" returns `false` on every rank,
      since every rank decides it from the same gathered marks. The
      step-1 refusal is gone; `adapt_to_initial_data!`, `firing_boxes`
      and `total_mass` needed no change.
    - *The workload.* `test/mpi_workload.jl` gains the tracked pulse (a
      right-moving Gaussian, vertex-centered, outer faces with the hook,
      flagged through `firing_boxes` with a buffer, three regrids with
      RK4 steps between), Burgers' shock through three regrids with the
      fixup (conservative operators, the fluxes `=> nothing`; mass
      conserved across each regrid and the run, and on `sum` lines),
      `adapt_to_initial_data!` from a single leaf with the host callback
      and with a `firing_boxes` flag vector (two of three ranks start
      empty at `-n 3`), and a case that refines the first blocks of a
      uniform mesh and coarsens them back, and refines three of four
      roots and coarsens them back, over three field sets at once (cell,
      `nvars = 2`, `G = 2`, order 4; vertex, `nvars = 1`, `G = 1`,
      order 2; a face, `nvars = 3`, `G = 2`, order 4), then a regrid
      that changes nothing. The second half runs again in `Float32` and
      `Float32x2` over the cell-centered set, so both types cross MPI in
      the regrid stage and the exchange. After each regrid the leaves
      and every moved field set's array, ghosts zero, are digested
      before any fill, so the transfer itself is compared bit for bit.
      `#` lines count the blocks that moved up and down the ranks and
      the coarsened blocks whose children had more than one owner, and
      the test asserts at each rank count that both directions and the
      straddling case occur. Four more refusals: a bad flag box on rank
      1, a flag vector of the wrong length on rank 0, another `buffer`
      on rank 1, and a forest refined on rank 1, each refused on every
      rank, the forest untouched by the first three.
    - *In process*, `test/regrid_exchange_tests.jl`. The lockstep test
      builds the regrid stage of every simulated rank, packs from its
      slice of the old array, delivers every segment, runs the local
      groups and unpacks, and requires the result bit for bit equal to
      the serial `regrid!` on every rank's new blocks; the union of the
      local and received transfers equal to the serial transfer set;
      both ends' layouts entry for entry; and every sent transfer naming
      its source's old owner and its target's new one. It covers `D =
      1, 2, 3`, every centering, the periodic, outer and reflecting
      faces of `exchange_tests.jl`, PointValue and Conservative, 1–5
      ranks and one more than the leaves in 1D (empty ranks), and, in
      3D, 7 ranks, which is the first count at which a coarsened group
      of that mesh straddles a rank boundary; `Float32` and `Float32x2`
      over a 2D subset. Then `regrid!` and `adapt_to_initial_data!`
      themselves run with one task per rank over a communicator whose
      collectives are rendezvous between the tasks and whose messages
      go through step 2's mailbox: two field sets and a flux `=> nothing`
      with mixed bare and boxed flags and a buffer, in 2D and 3D at 3
      and 5 ranks, bitwise against serial, and `transfer = false`; both
      criterion forms from a single leaf at 3 ranks, with the serial
      passes and data; and the refusals.
    - *Measured.* The workload at `-n 2`, `3` and `4` with one thread a
      rank, and at `-n 3` with two, prints the serial lines byte for
      byte except the `sum` lines, which differ in the last one or two
      of 17 digits where they differ (`l2`; Burgers' mass 16 against
      15.999999999999998 and 16.000000000000004); 39–42 s of wall clock
      per run, against 27–29 s in step 3. Blocks moved up and down at
      every rank count (the uniform mesh: 3, 6 and 5 up on the refine,
      as many down on the coarsen, at 2, 3 and 4 ranks), and 1, 2 and
      2 coarsened blocks had children of more than one owner.
    - *Serial cost* (a scratch script, best of 15: `D = 3`, `N = 16`,
      the 4³-root mesh of `bench/threads.jl` with its middle refined, a
      field set of 2 variables, `G = 2`, order 4, and a face-centered
      flux `=> nothing`, refining a slab of 128 blocks and coarsening it
      back; HEAD before and after, alternated). At one thread the
      refine took 37.1–38.4 ms before and 37.2–38.3 ms after, the
      coarsening 23.2–23.7 against 23.4–23.7; at four threads the
      refine 12.6–13.2 ms against 12.7–14.3 and the coarsening 7.4–7.9
      against 7.5–7.9. On a host-heavier mesh (`N = 6`, 8³ roots, 1856
      leaves after the refine) 26.3–27.0 ms against 26.4–27.2 and
      28.2–28.3 against 28.0–30.7. A regrid allocates 0.1–0.8 % more
      (38.55 MB against 38.50, 34.86 against 34.57): the marks and the
      layout hash. `bench/ghosts.jl`'s fill is unchanged, 2.57–2.59 ms
      against 2.57–2.58 and 11.45–11.46 against 11.48, with 16 and 32
      bytes less allocated per fill.
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 passes
      `partition_tests.jl`, `exchange_tests.jl`,
      `regrid_exchange_tests.jl`, `regrid_tests.jl` and `mpi_tests.jl`
      (2m38 together); the whole suite was not run there.
    - *Suite cost.* 108827 tests at one thread in 6m57 (7m02 in a
      second run, timed by file) and 108879 at eight in 6m57: 451 more
      than step 3 at each — the 430 of `regrid_exchange_tests.jl`, 22
      more in `mpi_tests.jl`, and one fewer in `partition_tests.jl`,
      whose two regrid refusals became one. `mpi_tests.jl` now takes
      96.6 s inside the suite, against step 3's 58.4 s, and
      `regrid_exchange_tests.jl` 12.4 s: about 50 s more in all. The
      `Float32x2` and `Float32` workload cases cost about 2 s each per
      run and the three-field-set case the most compilation; those are
      what to trim first. The thread-independence digests are
      unchanged. The docs build.
  - **Step 5 — interpolation routing.** *Accept:* an interpolation line
    in the workload, bit-identical to serial; a rank that passes no
    points; and an outside point refused on every rank with the same
    reason. *(Amended in step 5: every rank refuses together, but a rank
    that passed an outside point names its own, and the others name the
    first rank's; see "Point interpolation" in
    [Distributed meshes](CODE.md#distributed-meshes).)*

    *(Done, 2026-10-01.)* What it settled, and where it went beyond the
    plan (the design decisions are recorded under "Point interpolation"
    in [Distributed meshes](CODE.md#distributed-meshes)):
    - *The driver.* `interpolate!` over a distributed forest goes to
      `interpolate_distributed!` in `interpolate.jl`: the serial
      argument checks, factored into `check_interpolation`, run inside
      one agreed gather together with the cheap forest check, a hash of
      the arguments that must agree and the rank's first outside point;
      then the host location, the route by a stable counting sort and
      `alltoallv`, the kernel over the received points through
      `launch_interpolation!` — the serial launch, factored out, with
      the block offset — and the return of the values, and of the flags
      when there is a region, into the caller's order. `interpolate`
      itself defers a refusal of `derivs` to `interpolate!` over a
      distributed forest, so that it too is agreed. The step-1 refusal
      is gone; `refuse_distributed` now serves only the checkpoint.
    - *In process*, `test/interpolate_exchange_tests.jl`, over
      `regrid_exchange_tests.jl`'s rendezvous communicator, which gains
      an `alltoallv`. Every simulated rank queries an uneven slice of
      one point list — rank 1 none — and must reproduce its part of the
      serial `interpolate` bit for bit: value and gradient of two
      variables in reverse order with an excluded ball, the value alone
      of every variable without a region, and `interpolate!` into
      caller-supplied outputs from points given as vectors. The points
      run beyond periodic dimensions and reflecting walls, sit on every
      leaf's lower corner (so on every rank boundary) and on the
      domain's corners, over random data in every stored point. The
      cases are `D = 1` with a reflecting wall at 2 and 3 ranks and
      periodic at `nleaves + 2` ranks (two of them without blocks);
      `D = 2` reflecting at both ends and periodic, vertex-centered, at
      3 ranks, outer and cell-centered at 5, and a face-centered
      `Float32` set at 3; `D = 3` with a reflecting, an outer and a
      periodic dimension at 4. The refusals at 3 ranks: an outside
      point on rank 1 (its message and the others'), outside points on
      ranks 0 and 2 with rank 1 passing none, `derivs` only rank 0's
      checks refuse, `vars` and `exclude` that differ on one rank, and
      a forest refined on one rank.
    - *The workload.* `test/mpi_workload.jl` gains a 2D case, periodic
      in x₁, reflecting below in x₂ and outer above, three levels and an
      odd variable, over pseudo-random data with the ghosts filled:
      every rank asks for its own uneven slice (rank 1 none) of 301
      points running half a period and half a domain beyond the faces,
      and rank 0 digests the gathered answers, which are then in
      global order — value and gradient of two variables with an
      excluded ellipse and its flags in `Float64`, the value alone in
      `Float64` and `Float32x2`. A `#` line refuses an outside point on
      rank 1, and `mpi_tests.jl` asserts it is refused on every rank
      with rank 0 naming rank 1's point.
    - *Measured.* The workload at `-n 2`, `3` and `4`, one thread a
      rank, prints the serial lines byte for byte except the `sum`
      lines, which differ in the last one or two of 17 digits where they
      differ, as in step 4; the five interpolation lines are
      byte-identical at every rank count. 44–47 s of wall clock per
      run, against 39–42 s in step 4, the serial run 38 s. On Metal, a
      scratch script found the device path bitwise equal to the serial
      Metal call (the device bullet under "Point interpolation" in
      [Distributed meshes](CODE.md#distributed-meshes)); it is not in the suite.
    - *Serial cost* (`bench/interpolate.jl` with `N = 8`, 4³ roots and
      176 leaves, 20 variables, value and gradient with an excluded
      ball, best of 200 calls; HEAD before and after, alternated, four
      runs each). At one thread the 496-point batch took 0.353–0.368 ms
      before and 0.363–0.370 ms after, 4960 points 3.61–3.66 ms against
      3.64–3.71 ms; at four threads 0.109–0.114 ms against
      0.110–0.114 ms and 0.981–0.998 ms against 0.979–0.992 ms. That is
      within the run-to-run spread. A call allocates 32 bytes more at
      one thread (10320 against 10288) and 96 more at four (13232
      against 13136), the block offset in the kernel's arguments.
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 passes `interpolate_tests.jl`, `partition_tests.jl`,
      `exchange_tests.jl`, `regrid_exchange_tests.jl`,
      `interpolate_exchange_tests.jl` and `mpi_tests.jl` (3m06
      together); the whole suite was not run there.
    - *Suite cost.* 109038 tests at one thread in 7m49 and 109090 at
      eight in 8m09: 211 more than step 4 at each — the 204 of
      `interpolate_exchange_tests.jl` and 7 more in `mpi_tests.jl`;
      `partition_tests.jl`'s interpolation refusal became a refusal by
      verb, one test for one. The thread-independence digests are
      unchanged. The docs build. The step-4 tree, run the same day on
      the same machine, took 7m02 at one thread, so the suite grew by
      about 47 s; of that the new tests account for about 24 s when run
      standalone — `interpolate_exchange_tests.jl` 13.5 s, nearly all of
      it compilation, and `mpi_tests.jl` 125 s against the step-4 tree's
      114 s — and the rest was not separated from the run-to-run spread.
      The `Float32` and 3D cases of the new file are what to trim first.
  - **Step 6 — parallel checkpoints.** `TreeAMRHDF5MPIExt` and the hooks
    in `TreeAMRHDF5Ext`. *Accept:* save at `-n 3`, then load at `-n 2`,
    at `-n 1` and serially, each continuation byte-identical to the
    uninterrupted run; a serial file loaded at `-n 3`; a rank with no
    blocks; plain data that differ between ranks refused on every rank;
    `MPI_File_sync` confirmed under the collective flush; and
    `bench/checkpoint.jl` run under MPI on Symmetry, with the shared
    file's throughput recorded on its parallel file system — the
    benchmark "Parallel I/O and M7" asked for. The per-process files
    are revisited only if the shared file does not hold up.

    *(Done, 2026-10-01; measured on Symmetry on 2026-10-02, on one node
    and then on two and four, where the shared file lost data until
    ROMIO's read-modify-write was turned off, which with its rate led to
    the decision to replace it; see the last items.)*
    What it settled, and where it went beyond the
    plan (the design decisions are recorded under "Parallel checkpoints"
    in [Distributed meshes](CODE.md#distributed-meshes)):
    - *The code.* `ext/TreeAMRHDF5MPIExt.jl` (`[extensions]`
      `TreeAMRHDF5MPIExt = ["HDF5", "MPI"]`) opens the shared file;
      `TreeAMRHDF5Ext` gains the `Access` hooks (`open_file`,
      `write_whole`, `write_slab` / `read_slab`, `put_value`,
      `agree_values`, `flush_ranks`, `publish`), the agreement
      (`agreed`), the plain-data walk (`plain_hash`) and the
      fixed-length string arrays; `src/` gains the stubs `librarycomm`
      and `open_parallel_file`, `load_checkpoint`'s `comm` keyword, and
      the docstrings' collective contract. `refuse_distributed` is gone,
      its last caller with it.
    - *The serial file is unchanged but for `nranks`.* A scratch script
      wrote the same checkpoint — two field sets, `Float64` with parity
      and a `Float32x2` vertex set, a reflecting face, every kind of
      plain data and a do-block `write_plain` — with HEAD before the step
      and after it, unfiltered and with `Shuffle` + `Deflate(1)`; `h5dump`
      of the two files differs in `created` and the new `nranks = 1`
      only, and `h5dump -p` also in the storage offsets, which the 8-byte
      dataset shifts.
    - *The workload.* `test/mpi_workload.jl` gains `checkpoint_case`:
      the tracked pulse as a chunked driver (RK4 steps, the flags and a
      regrid per chunk), run uninterrupted for three chunks, then saved
      after the first — the bare `name => fs` form, plain data with a
      Rational and an array of strings including an empty one —
      unfiltered and with `Shuffle` + `Deflate(1)`, dropped, loaded at
      the same rank count and continued; the mesh refines again after
      the save (39 leaves, then 87). Its lines must be the serial
      run's, and the continued digests the uninterrupted run's. A 1D
      forest of two leaves is saved and loaded too, which at three ranks
      leaves one rank without blocks both times. `checkpoint_cross` then
      loads every file the earlier runs wrote at another rank count, at
      this one and, under MPI, a second time over `MPI.COMM_SELF`, on
      `#` lines. `mpi_tests.jl` runs the serial reference, `-n 3` and
      `-n 2` (in that order now) over one `TREEAMR_CHECKPOINT_DIR`, then
      the serial loads in process: so the serial file loads at 3 ranks
      and at one; the 3-rank files at 2, at one inside the 2-rank job
      (a one-rank MPI communicator — what a separate `mpiexec -n 1`
      would test, without a third launch) and serially; the 2-rank files
      serially. Every load reproduces the saved digests and every
      continuation the uninterrupted ones. The refusals, at every rank
      count: plain data that differ between ranks; an `application` only
      rank 1 refuses; filters that differ on rank 1; a `write_plain` in
      the do-block of a value that differs, after which the earlier file
      is intact and no partial file is left; a load whose `fieldsets`
      differ on rank 1; a load of a missing file, which rank 0 alone
      looks for. Each is raised on every rank.
    - *In process*: `checkpoint_tests.jl` passes unchanged but for two
      deliberate assertions, `nranks == 1` and a file without `nranks`
      reading as 1; `partition_tests.jl`'s save over the test-only
      communicator now stops at its missing `allgather`, by name, like
      the interpolation.
    - *Measured by hand.* The workload at `-n 2`, `3` and `4`, one thread
      a rank, prints the serial lines byte for byte except the `sum`
      lines, which agree to roundoff, as in step 5; every cross load at
      every count reproduces the digests. 53–57 s of wall clock per run,
      against 44–47 s in step 5, and the serial run 44 s, against 38;
      most of the difference is compiling HDF5.jl's paths in each rank.
      `mpi_tests.jl` alone takes 2m34, against step 5's 125 s.
    - *Throughput, locally only* (`bench/checkpoint.jl`, now runnable
      under `mpiexec` with the argument `mpi`; the development laptop of
      "Throughput and filters", its internal SSD, one thread a rank,
      MPICH_jll 5.0.2, the same day and the same in-use conditions).
      GB/s of state, aggregate over the ranks, the slowest rank's time;
      `-n 1` is the serial path over a one-rank communicator:

      | data | filter | serial | `-n 1` | `-n 2` | `-n 4` |
      |---|---|---|---|---|---|
      | blast, save | none | 7.49 | 7.50 | 6.66 | 6.90 |
      | blast, sync | none | 4.58 | 4.66 | 4.52 | 4.47 |
      | blast, load | none | 1.24 | 1.28 | 2.28 | 3.89 |
      | blast, save | `Shuffle` + `Deflate(1)` | 0.34 | 0.34 | 0.67 | 0.92 |
      | blast, load | `Shuffle` + `Deflate(1)` | 0.51 | 0.51 | 0.96 | 1.57 |
      | pulse, save | none | 5.66 | 7.46 | 6.51 | 4.55 |
      | pulse, load | none | 1.20 | 1.21 | 2.03 | 3.09 |
      | pulse, save | `Shuffle` + `Deflate(1)` | 0.11 | 0.11 | 0.21 | 0.38 |
      | pulse, save | bitshuffle + LZ4 | 0.62 | 0.62 | 1.12 | 1.80 |

      So on one SSD the unfiltered save stays at the page cache's rate
      and the flush at the drive's whatever the rank count, while what
      is per rank — compression, the load's allocation, first touch and
      `scatter!` — divides among the ranks: the filtered save 2.7 times
      faster at 4 ranks, the unfiltered load 3.1 times. This is the
      point "Compression is serial" made, that under M7 a filter
      parallelizes. File sizes are the serial ones (90.0 MB for blast
      with Deflate). H5Zzstd and H5Zlz4 were not in the environment, so
      their rows are missing; H5Zbitshuffle was, from the default
      environment. One run each; the M9a table varied by 15–60 % run to
      run on this machine. These are not the parallel file system's
      numbers, which are the point of the measurement.
    - *`MPI_File_sync`* is confirmed by reading libhdf5 2.2.0's and
      MPICH 5.0's sources (the chain is under "Parallel checkpoints"),
      not by tracing a run: the call is made, but no test here can see
      a write reach stable storage, and macOS's `fsync` does not.
    - *Symmetry, one node only* (measured 2026-10-02, job 567854 on
      cn106, AMD EPYC 7543, 64 cores, 8 NUMA domains; Julia 1.13.1,
      MPICH_jll 5.0.2 through `srun --mpi=pmi2`, HDF5_jll's MPICH build
      of libhdf5 2.2.0 with `HDF5.has_parallel()`; the files on BeeGFS
      under `/mnt/beegfs/eschnetter/claude`; `bench/symmetry_checkpoint_mpi.sh`
      as amended in the step's script commit, `TREEAMR_BENCH_REPS = 3`,
      `64 / P` threads a rank, ranks in blocks so that 8 are one per
      domain). GB/s of state, aggregate, the slowest rank's time, save
      (`sync = false`) / sync / load; the default mesh, 3296 blocks,
      216 MB of `pulse` and 540 MB of `blast`; `-n 1` is the serial
      path over a one-rank communicator:

      | data | filter | `-n 1` | `-n 2` | `-n 4` | `-n 8` |
      |---|---|---|---|---|---|
      | blast | none | 1.53 / 1.60 / 1.52 | 1.56 / 1.26 / 2.82 | 1.44 / 0.96 / 4.03 | 1.37 / 0.84 / 4.56 |
      | blast | `Shuffle` + `Deflate(1)` | 0.22 / 0.22 / 0.42 | 0.40 / 0.38 / 0.70 | 0.52 / 0.52 / 1.40 | 0.81 / 0.77 / 2.51 |
      | blast | shuffle + zstd(1) | 0.64 / 0.64 / 0.70 | 0.89 / 0.85 / 1.02 | 1.18 / 1.01 / 2.50 | 1.46 / 1.28 / 3.83 |
      | blast | bitshuffle + zstd(1) | 0.51 / 0.51 / 0.68 | 0.77 / 0.74 / 1.28 | 1.01 / 0.96 / 2.41 | 1.15 / 1.06 / 3.65 |
      | pulse | none | 1.26 / 1.34 / 1.20 | 1.35 / 1.11 / 2.31 | 1.34 / 0.90 / 2.36 | 1.15 / 0.77 / 3.27 |
      | pulse | `Shuffle` + `Deflate(1)` | 0.07 / 0.07 / 0.19 | 0.12 / 0.12 / 0.45 | 0.19 / 0.18 / 0.74 | 0.25 / 0.23 / 1.33 |
      | pulse | shuffle + zstd(1) | 0.30 / 0.30 / 0.42 | 0.35 / 0.34 / 1.05 | 0.42 / 0.36 / 1.79 | 0.42 / 0.34 / 2.43 |

      File sizes are the serial ones (blast 89.9 MB with Deflate, 87.9
      with zstd(1); pulse 161–164 MB). The mesh with 6³ → 12³ roots has
      12648 blocks (the refinement follows a surface, so 3.8 times the
      blocks, not 8; 829 MB of pulse and 2.07 GB of blast): at `-n 1`
      pulse unfiltered 1.25 / 1.48 / 1.35, blast zstd(1) 0.62 / 0.63 /
      0.69; at `-n 2` pulse unfiltered 1.48 / 1.32 / 2.55, blast
      unfiltered 1.48 / 1.13 / 2.96 — the default mesh's rates, so the
      table is not a small-file effect. What it shows:
      1. **An unfiltered save from one node is 1.2–1.6 GB/s at every
         rank count**, and the flush costs more with more ranks (blast
         sync 1.60 → 0.84 GB/s from 1 to 8), every rank's
         `MPI_File_sync` being a BeeGFS client round trip. One node's
         client, or the servers, set that rate; which, only the
         multi-node run can say.
      2. **A filter parallelizes**, as "Compression is serial"
         predicted: blast with zstd(1) saves at 1.46 GB/s at 8 ranks,
         2.3 times its one-rank rate and about the unfiltered rate,
         for a file 6.1 times smaller; Deflate 3.7 times faster. Pulse,
         whose data compress to 1.3 only, gains less (zstd(1) 1.4
         times).
      3. **The load scales** (blast unfiltered 1.52 → 4.56 GB/s), but it
         reads a file the node just wrote, so this is the client's
         cache and the per-rank allocation and `scatter!`, not BeeGFS's
         read rate.
      4. Collective metadata reads were not needed at these counts.

      The log of 567854 was copied later and is complete; the larger
      mesh at `-n 4` and `-n 8`, which the table above lacked, is pulse
      unfiltered 1.28 / 0.94 / 3.41 and 1.43 / 0.89 / 4.24, blast
      unfiltered 1.57 / 1.08 / 4.82 and 1.63 / 0.90 / 5.64.
    - **The multi-node run lost data** (job 567855, 2026-10-02: cn092–095,
      8 ranks a node, `TREEAMR_CKPT_ONENODE = 0`). Every setting at 2
      nodes on both meshes and at 4 nodes on the default one, and at 4
      nodes on the larger mesh through shuffle + zstd(3), loaded back; then a load of blast with
      shuffle + LZ4 on the larger mesh was refused on every rank by the
      forest: leaves 5538 and 5539 "out of curve order", the second a
      key with coordinates `(0, 0, 0)`. The last file was still on
      BeeGFS and was read serially on another node (job 567907,
      `bench/checkpoint_inspect.jl`): `root`, `level` and all 63240
      chunks of `data` were intact, and in `coords` exactly the 395
      rows of rank 14 (on cn093) were zero, bytes [160993, 165733) of
      the file. Every reader agreeing, the damage was in the file, not
      in a read. `bench/checkpoint_layout.jl` showed the trigger: HDF5
      had put the first 9 chunks of `data` (rank 0's first two blocks)
      at [6672, 8110), in free space before the leaf columns at
      [31309, 246325), and the rest from 250501 on, so rank 0's file
      view for its chunks spanned the columns. ROMIO reported its
      generic driver (`romio_filesystem_type` "UFS"), `romio_cb_write`
      and `romio_ds_write` "automatic", and the BeeGFS client
      `tuneFileCacheType = buffered` (512 KiB a file),
      `tuneUseGlobalFileLocks = false`, `tuneRemoteFSync = true`; the
      file was striped over 4 targets on 2 servers in 512 KiB chunks.
      The mechanism this points to is under "Parallel checkpoints" in
      [Distributed meshes](CODE.md#distributed-meshes); here is what tested it.
    - *The reproducers* (`bench/checkpoint_stress.jl`, run by
      `bench/symmetry_checkpoint_stress.sh`; four nodes, 32 ranks,
      MPICH_jll 5.0.2, files on BeeGFS unless marked; a save is damaged
      when either of two serial readers on different nodes finds a row
      that is not what was written). The 12648-leaf partition of the
      failing run, at 4³ cells a block:

      | layer | case | damaged | jobs |
      |---|---|---|---|
      | `save_checkpoint`, Shuffle + Deflate(1) | as in 0e71c75 | 12 / 200; 13 / 200 with `sync` | 567917 (and 567915) |
      | | `romio_ds_write` off (hint file) | 0 / 200 | 567917 |
      | | `romio_ds_write` and `romio_cb_write` off | 0 / 200 | 567917 |
      | | with the fix (the hints in the code) | 0 / 1000 | 567930 |
      | `save_checkpoint`, unfiltered | as in 0e71c75 | 0 / 200; 0 / 200 with `sync` | 567915 |
      | | with the fix | 0 / 300 | 567930 |
      | HDF5 alone, contiguous hyperslabs | defaults | 0 / 200 | 567915 |
      | MPI-IO, the checkpoint's pattern, independent | defaults | 1 / 200; 3 / 2000 | 567917, 567930 |
      | | data sieving off | 0 / 200; 0 / 2000 | 567917, 567930 |
      | MPI-IO, the same, collective | defaults (falls back to independent) | 0 / 200; 0 / 200 with sync-barrier-sync | 567917 |
      | | collective buffering forced on | 151 / 200; 176 / 200 | 567917, 567930 |
      | | the same, data sieving off | 147 / 200 | 567917 |
      | | both off | 0 / 200 | 567930 |
      | MPI-IO, adjacent unaligned ranges | `write_at_all`; `write_at` | 0 / 200; 0 / 200 | 567915, 567917 |
      | POSIX `pwrite`, adjacent unaligned ranges | — | 0 / 200 | 567915 |
      | POSIX, a rank's write read by the next rank after a barrier | — | unseen 5 / 300, always across nodes; file after close intact | 567958 |
      | NFS `/home`: MPI-IO pattern, independent; collective buffering on | defaults | 0 / 1000; 0 / 200 | 567959 |
      | NFS: MPI-IO adjacent ranges; POSIX `pwrite`s | — | 0 / 200; 98 / 200, whole pages | 567959 |

      Under data sieving the loss was always one or more whole slabs
      of ranks on cn093–cn095 (31 damaged saves, ranks 9 to 30), never
      of ranks 1–7, which share rank 0's node; under collective buffering,
      whole aggregator file domains. Under the old rate, 0 damaged in
      1000 has a probability of about e⁻⁶⁰. The collective MPI-IO case
      with defaults saw no damage either way, so it says nothing about
      sync-barrier-sync. What was not done: a trace of individual
      writes, a cross-node `fcntl` lock test, a run with
      `tuneUseGlobalFileLocks = true` (root only), and any Open MPI run
      — HDF5_jll's Open MPI build failed to load HPC-X's library (job
      567937), and the pure MPI-IO runs under OMPIO were cancelled when
      the shared file was dropped.
    - *Measured with the fix* (job 567938, cn092–095, the same day,
      `TREEAMR_BENCH_REPS = 3`; every save now checked by a load, 8 a
      setting, all of which passed): GB/s save / sync / load, as above.

      | data | filter | 2 nodes, 16 ranks | 4 nodes, 32 ranks |
      |---|---|---|---|
      | blast | none | 1.69 / 0.92 / 5.45 | 1.72 / 0.93 / 6.10 |
      | blast | `Shuffle` + `Deflate(1)` | 0.76 / 0.74 / 2.94 | 1.18 / 1.05 / 3.94 |
      | blast | shuffle + zstd(1) | 1.29 / 1.16 / 4.27 | 1.79 / 1.57 / 4.76 |
      | blast | bitshuffle + zstd(1) | 1.22 / 1.03 / 4.28 | 1.68 / 1.34 / 4.68 |
      | pulse | none | 1.33 / 0.81 / 2.84 | 1.38 / 0.81 / 3.00 |
      | pulse | `Shuffle` + `Deflate(1)` | 0.38 / 0.35 / 1.62 | 0.56 / 0.48 / 1.85 |
      | pulse | shuffle + zstd(1) | 0.64 / 0.49 / 2.56 | 0.89 / 0.62 / 2.43 |

      The larger mesh: at 2 nodes pulse unfiltered 2.00 / 0.88 / 5.13,
      blast unfiltered 1.85 / 1.13 / 7.90, blast zstd(1) 1.26 / 1.06 /
      5.21; at 4 nodes 1.72 / 1.05 / 5.49, 1.98 / 0.95 / 9.18 and 1.95 /
      1.58 / 8.23. One node with the fix (job 567939, which overlapped
      the cost measurement below on the same file system, on cn112): at `-n 8` blast unfiltered
      1.47 / 0.94 / 3.82 and zstd(1) 0.80 / 0.72 / 2.96, against 1.37 /
      0.84 / 4.56 and 1.46 / 1.28 / 3.83 without (567854); `-n 1`, the
      serial path, which the hints do not touch, unchanged (blast
      zstd(1) 0.64).
    - *The hints' cost* (job 567964, 2 nodes, 16 ranks, alternating with
      and without them in a scratch copy, two rounds, at the same time
      as 567939): blast zstd(1) 1.40 and 1.40 GB/s with, 1.88 without;
      shuffle + LZ4 1.37 and 1.40 against 2.08; Deflate(1) 0.95 and 0.91
      against 1.16 and 1.15; blast unfiltered 1.73 and 1.79 against 1.68
      and 1.84; pulse unchanged, unfiltered and filtered. With the
      hints data sieving is replaced by one write a chunk (ROMIO's
      "naive" strided path, `ad_write_str_naive.c`), and blast's chunks
      compress to about 5 KB. Without the hints both rounds were
      refused by the checksums, on every rank — round 1 at
      bitshuffle + LZ4, round 2 at shuffle + zstd(1) — which is the
      checksums catching the original loss in the benchmark itself.
    - *What it says about the shared file.* An unfiltered save is 1.3–2.0
      GB/s at one, two and four nodes alike, so its rate is that of one
      file, presumably its four storage targets, and not of the
      clients; a filter parallelizes up to that rate (blast zstd(1)
      1.95 GB/s at 32 ranks on the larger mesh); a load of a file just
      written reaches 9 GB/s at 32 ranks. Whether the shared file's
      rate grows with nodes past one node's was the question; it does
      not, and with the losses that settles it: **the shared file is to be replaced** by files per I/O
      process (decided 2026-10-02 with Erik; see "Parallel
      checkpoints"). The checksums, the self-checking benchmark and the
      reproducers carry over to the replacement.
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 passes
      `partition_tests.jl`, `checkpoint_tests.jl` and `mpi_tests.jl`
      (2m53 together), the parallel checkpoints and every cross load
      included; the whole suite was not run there.
    - *Suite cost.* 109107 tests at one thread in 8m10 and 109159 at
      eight in 8m02, against step 5's 109038 in 7m49 and 109090 in 8m09:
      69 more at each — 67 in `mpi_tests.jl` (the checkpoint lines, the
      cross loads at each rank count and in process, the refusals, and
      the extension's presence) and the two `nranks` assertions of
      `checkpoint_tests.jl`. `mpi_tests.jl` takes about 28 s more than in
      step 5 standalone (2m34 against 125 s), which is the HDF5 code
      compiled in each `mpiexec` run and the in-process loads; at one
      thread the suite grew by about 20 s. The thread-independence
      digests are unchanged. The docs build.
  - **Step 6b — checkpoints without parallel I/O.** The design under
    "Checkpoints without parallel I/O" in
    [Distributed meshes](CODE.md#distributed-meshes), decided 2026-10-02 with
    Erik, replacing step 6's shared file: I/O groups and the `io`
    keyword, part files and an index (format version 2, the reader
    reading 1 and 2), the verbs `commnodes` and `bcast`, and the removal
    of `TreeAMRHDF5MPIExt`. *Accept:* serially, the single file with its
    inline part round-trips as before, and version-1 files written by
    the old writer (fixtures committed under `test/fixtures/`) still
    load; under MPI, saves at `-n 3` with `io = :all`, `io = 1` (or
    `:node` on one node) and an `io` between, loaded at `-n 2`, at one
    rank and serially, each continuation byte-identical, and the
    version-1 fixtures loaded at every rank count; refused on every
    rank: a part from another save, a block damaged in a part, a missing
    part; orphans removed and nothing else; an I/O process failing
    mid-save, after which the previous checkpoint loads and no new part
    is left; and no file opened by more than one process, by a test hook
    that records every open. Then `bench/checkpoint.jl` under MPI with
    `io = :node` and `io = :all` on one, two and four nodes of Symmetry,
    on BeeGFS, every save verified, compared with step 6's shared file;
    and the stress reproducer's `save_checkpoint` modes on four nodes,
    hundreds of verified saves, with no damage.

    *(Done, 2026-10-02.)* What it settled (the design decisions made
    while implementing are under "Checkpoints without parallel I/O" in
    [Distributed meshes](CODE.md#distributed-meshes)):
    - *The code.* `ext/TreeAMRHDF5MPIExt.jl` and its `[extensions]` entry
      are gone, with `open_parallel_file`, `librarycomm` and the hints;
      `src/communicator.jl` gains `bcast` and `commnodes`, each with a
      serial method and an MPI one; `TreeAMRHDF5Ext` replaces the
      `Access` hooks with the I/O plan (`io_plan`, `groupblocks`), the
      gathering (`send_blocks`, `write_blocks!`), the index
      (`write_index!`), the cleanup (`previous_parts`, `remove_stale`),
      the image (`share_index`, `index_image`) and the scattering
      (`part_readers`, `open_part`, `read_blocks!`), with `agree_errors`
      after every step that runs on some ranks only.
    - *The tests.* `checkpoint_tests.jl` checks the single version-2
      file (its part inside the index, the part table, the save id), a
      damaged block, a block whose checksum was recomputed with it
      (refused by the index's checksum of the part's checksums), the
      required checksums, an I/O process failing after its first write
      (a test hook, `FAIL_PART`), the NUL refusal, and two version-1
      fixtures written by the step-6 writer (`test/fixtures/`, 42 KB
      together; generated without an active project, so that they hold
      no environment texts). `mpi_workload.jl` saves the chunked pulse
      run with `io = :all`, `io = 2` and `:node` (one part on one node),
      with messages cut at 4 KiB by the hook `MAX_MESSAGE`, so that a
      rank's blocks travel in several; continues from each at the same
      rank count; loads the version-1 fixtures, on lines that must be
      the serial run's; and refuses on every rank a block damaged in
      the last part, a part from another save copied over one of this
      save's, and a missing part. It checks that a save removes an
      orphan and the previous save's parts and nothing else, that an
      I/O process failing mid-save fails the save on every rank and
      leaves the previous checkpoint loadable with no new part and no
      partial index, and that no file is opened by two processes in a
      save with two I/O groups or in a load (a hook, `OPEN_LOG`,
      records every open on every rank). `mpi_tests.jl` loads every
      file at the other rank counts, at one rank and serially, as
      before, now for the three files, and follows the external links
      of the three-rank `io = :all` index into its parts with HDF5.jl,
      as a tool would.
    - *Checked by hand.* TreeAMR 0.1.4, from the registry, refuses a
      version-2 file with its format-version refusal, saying it was
      written by a newer TreeAMR. Julia 1.11.9, in the manifest-free
      copy, passes `partition_tests.jl`, `checkpoint_tests.jl` and
      `mpi_tests.jl`, 5899 tests in 2m48.
    - *Suite cost.* 110606 tests at one thread in 6m32 and 110658 at
      eight in 6m28, against 110525 in 7m16 at one thread at 67c9153 the
      same day: 81 more, in `checkpoint_tests.jl` (593 tests, 1m08 on
      its own), `mpi_tests.jl` (173, 1m07 on its own) and
      `partition_tests.jl`. The thread-independence digests are
      unchanged. The docs build.
    - *Measured on Symmetry* (2026-10-02: AMD EPYC 7543 nodes, 8 ranks a
      node at 8 threads, one per NUMA domain; Julia 1.13.1, MPICH_jll
      5.0.2 through `srun --mpi=pmi2`, libhdf5 2.2.0 used serially; files
      on BeeGFS, then 91 % full; `bench/symmetry_checkpoint_mpi.sh`,
      `TREEAMR_BENCH_REPS = 3`, every save verified by a load). GB/s of
      state, save (`sync = false`) / sync / load, aggregate, the slowest
      rank's time; the default mesh, 3296 blocks, 216 MB of `pulse` and
      540 MB of `blast`; the parts in parentheses. `:all` is from jobs
      568075 (two and four nodes, cn095 and cn102–104) and 568077 (one
      node, cn084), `:node` from their repetition after the amendment
      to the gathering, 568119 (cn107) and 568120 (cn093–096); the last
      column is step 6's shared file on four nodes, with the hints
      (567938):

      | data | filter | 1 node, `:node` (1) | 1 node, `:all` (8) | 2 nodes, `:node` (2) | 2 nodes, `:all` (16) | 4 nodes, `:node` (4) | 4 nodes, `:all` (32) | shared, 4 nodes |
      |---|---|---|---|---|---|---|---|---|
      | blast | none | 0.77 / 0.71 / 0.99 | 2.45 / 4.16 / 4.64 | 1.28 / 1.34 / 1.58 | 2.55 / 2.59 / 6.78 | 2.22 / 2.17 / 3.18 | 2.70 / 2.37 / 8.24 | 1.72 / 0.93 / 6.10 |
      | blast | `Shuffle` + `Deflate(1)` | 0.20 / 0.20 / 0.39 | 0.94 / 0.94 / 1.94 | 0.32 / 0.32 / 0.73 | 1.45 / 1.46 / 3.66 | 0.57 / 0.57 / 1.26 | 1.94 / 1.92 / 4.99 | 1.18 / 1.05 / 3.94 |
      | blast | shuffle + zstd(1) | 0.49 / 0.48 / 0.59 | 2.96 / 2.88 / 2.77 | 0.72 / 0.73 / 1.07 | 3.49 / 3.60 / 5.36 | 1.45 / 1.46 / 2.00 | 4.50 / 4.32 / 7.46 | 1.79 / 1.57 / 4.76 |
      | blast | bitshuffle + zstd(1) | 0.39 / 0.39 / 0.53 | 2.40 / 2.42 / 2.65 | 0.60 / 0.60 / 1.00 | 2.92 / 2.92 / 5.15 | 1.23 / 1.21 / 1.81 | 3.93 / 3.80 / 6.95 | 1.68 / 1.34 / 4.68 |
      | pulse | none | 0.71 / 0.67 / 0.78 | 1.72 / 2.13 / 3.11 | 1.18 / 1.16 / 1.08 | 2.07 / 1.63 / 4.33 | 1.76 / 1.76 / 2.29 | 1.62 / 2.08 / 4.33 | 1.38 / 0.81 / 3.00 |
      | pulse | `Shuffle` + `Deflate(1)` | 0.06 / 0.06 / 0.19 | 0.38 / 0.38 / 0.99 | 0.12 / 0.12 / 0.36 | 0.46 / 0.45 / 1.98 | 0.24 / 0.24 / 0.63 | 0.55 / 0.55 / 2.94 | 0.56 / 0.48 / 1.85 |
      | pulse | shuffle + zstd(1) | 0.25 / 0.25 / 0.35 | 1.53 / 1.57 / 1.80 | 0.45 / 0.46 / 0.66 | 1.40 / 1.45 / 3.14 | 0.83 / 0.85 / 1.07 | 1.51 / 1.44 / 3.95 | 0.89 / 0.62 / 2.43 |

      The larger mesh (12648 blocks, 829 MB of pulse and 2.07 GB of
      blast), blast unfiltered: 1.37 / 1.35 / 1.69 and 3.47 / 3.06 /
      9.07 at two nodes with `:node` and `:all`, 2.01 / 2.02 / 3.66 and
      2.92 / 2.99 / 12.46 at four, against the shared file's 1.85 / 1.13
      / 7.90 and 1.98 / 0.95 / 9.18; blast with zstd(1) 0.76 / 0.75 /
      0.99 and 4.20 / 4.14 / 5.88 at two, 1.19 / 1.19 / 1.82 and 5.60 /
      5.25 / 10.60 at four, against 1.26 / 1.06 / 5.21 and 1.95 / 1.58 /
      8.23; pulse unfiltered 1.13 / 1.15 / 1.63 and 3.17 / 3.02 / 6.35 at
      two, 2.24 / 2.38 / 3.11 and 2.70 / 2.57 / 8.02 at four. On one
      node the shared file had saved blast at 1.47 / 0.94 / 3.82 and
      with zstd(1) at 0.80 / 0.72 / 2.96 (567939, 8 ranks). What it
      shows:
      1. **A part per rank (`io = :all`) is the fastest setting
         measured, everywhere.** Unfiltered it saves at 1.2–1.9 times
         the shared file's rate, and its flush to stable storage costs
         nothing measurable (sync ≈ save, where the shared file's
         per-rank `MPI_File_sync` cost 40–55 % of the rate); a filter
         parallelizes over the ranks (blast with zstd(1) at 4.50 GB/s on
         four nodes, 5.60 on the larger mesh, 2.5 and 2.9 times the
         shared file); and the load reaches 4–12 GB/s. The unfiltered
         save does not grow from two nodes to four (2.55 → 2.70 GB/s,
         3.47 → 2.92 on the larger mesh), so at about 3 GB/s it is the
         file system's rate for these writes, not the clients'.
      2. **A part per node (`io = :node`, the default) grows with the
         nodes** — blast unfiltered 0.77, 1.28 and 2.22 GB/s on one, two
         and four nodes — at 0.5–0.8 GB/s per I/O process. On one node
         that is half the shared file's rate; on four, it is above it
         unfiltered (2.22 against 1.72, and 2.17 against 0.93 synced)
         and below it filtered (1.45 against 1.79 with zstd(1)), since
         one process per node runs the node's compression, serially (see
         "Compression where the data are" under "Performance work left
         for later"). A load with one part per node is read by one rank
         per node, 0.8–3.7 GB/s.
      3. **The amendment to the gathering made no measurable
         difference** (568119/568120 against 568075/568077, the `:node`
         runs of the same jobs: blast unfiltered 0.77 against 0.80 on
         one node, 1.28 against 1.36 on two, 2.22 against 2.76 on four,
         inside this file system's run-to-run spread). It is kept for the
         memory it bounds — two 64 MiB buffers on an I/O process instead
         of two the size of a member's share — but the guess that
         motivated it was wrong. What is consistent with the numbers,
         and not traced: MPICH makes no progress on a posted receive
         while the process is inside HDF5 (it has no asynchronous
         progress by default), so the transfer from the members —
         2.7–2.8 GB/s for 1–64 MiB between two ranks of one node,
         `pingpong` job 568132 — and the write add up rather than
         overlap. A write task on another thread while the calling task
         waits in MPI, or MPICH's asynchronous progress, would test it.
      4. **Every save was verified**, by a load that compared the leaves
         and the state bit for bit on every rank and verified every
         checksum: none failed, against both rounds of step 6's
         measurement without the hints.
    - *The stress reproducer on four nodes* (job 568076, cn093–096, 32
      ranks, the 12648-leaf partition of step 6's failure at `4³` cells
      a block, every save loaded serially and compared by a reader on
      cn093 and then one on cn096): shuffle + `Deflate(1)` with `io =
      :node`, 400 saves; the same with `io = :all` and `sync = true`, 300
      saves; unfiltered with `:node`, 300 saves: **0 damaged in 1000**,
      in 10m30. Step 6's shared file lost 12 of 200 filtered saves of
      this mesh (567917); at that rate 0 in the 700 filtered saves here
      has a probability of about e⁻⁴³.
    - *What was not done.* An Open MPI run (nothing in the package needs
      MPI-IO any more, so its OMPIO question is moot for checkpoints);
      a cold-cache load; more than four nodes, where the metadata cost
      of `:all`'s file per rank is the open question; and a trace of
      where an I/O process's time goes.
  - **Step 7 — weak-scaling smoke test.** `bench/mpi.jl` holds the
    blocks per rank fixed and times the RHS, the ghost fill, a regrid
    and the norm; `bench/symmetry_mpi.sh` runs one rank per NUMA domain
    at eight threads, on one, two and four nodes, with the system MPI
    through MPIPreferences. *Accept:* the table recorded here, with
    what limits it named, the replicated regrid bookkeeping and the
    overlap of the exchange with the local groups among them.

    *(Done, 2026-10-02: built and run locally, then on Symmetry at up to
    four nodes, 32 ranks; the table and what limits it are the
    "Symmetry" items below.)* What it settled:
    - *The benchmark.* `bench/mpi.jl` (under `mpiexec` with the
      argument `mpi`, in the test environment) builds a mesh of `TILES`
      identical tiles of `ROOTS^D` roots stacked along `x_D`, periodic,
      `TILES` the rank count unless set. The roots are numbered with
      `x_D` slowest and every tile has the same leaves, so the
      equal-count split gives rank `r` exactly tile `r`: the blocks per
      rank are fixed by construction, and the halo per rank is constant
      from three ranks on. Two tiles: *two-level*, the leaves with `x₁`
      in the lower half and `x_D` in the upper half of the tile refined
      once (176 blocks at `ROOTS = 4` in 3D), so a coarse-fine face
      crosses every rank boundary and phase 1, the prolongation stage
      and the interface restriction all carry messages; and *uniform*
      (64 blocks), copies only. Timed in synchronized windows — an
      `MPI.Barrier`, the call, the slowest rank's time by `allgather` —
      as the minimum and the median over the repetitions: the wave RHS
      (`scatter!` → `fill_ghosts!` → `map_blocks!`), the fill alone and
      its three parts run separately through the internals (the stages'
      local groups, the packs, the unpacks), `scatter!`, the norm, a
      `mesh_mapreduce` max, `restrict_interfaces!` on a set
      face-centered along `x_D`, `interpolate!` of 1000 points a rank
      spread over the whole domain, the two schedule builds, a regrid
      that refines the lowest layer of roots of tile 0 — the start of
      the curve, so every later rank's range shifts and blocks migrate
      — and one that coarsens it back, and the triad. `#` lines give per
      rank, as min/mean/max, the messages, bytes, transfers and peers of
      a fill and of an interface restriction (read from the stages),
      the blocks a regrid made a rank take from another, and ns per cell
      of the whole job. `TREEAMR_BENCH_BACKEND` and
      `TREEAMR_BENCH_DEVICEAWARE` choose a device and its message path,
      `TREEAMR_BENCH_TILES` the mesh of a `P`-rank run for a serial
      control, `TREEAMR_BENCH_LABEL` the column name. The tab-separated
      lines carry the minimum *and* the median, so `bench/scan.sh`'s
      awk does not read them; `bench/mpitable.awk` is the matching
      parser (efficiency against the first run, then both times) and
      `bench/mpiscan.sh P…` launches each count through
      `MPI.mpiexec()` and prints the table.
    - *Laptop smoke run, not a scaling result* (Apple M3 Pro, 6
      performance and 6 efficiency cores, one memory system; MPIABI_jll
      through the global preference; `D = 3`, `N = 16`, `ROOTS = 4`, 2
      variables, `G = 2`, order 4, Float64, 10 windows; `bench/mpiscan.sh
      1 2 4`, one thread a rank). Minimum ms, two-level mesh, with the
      weak-scaling efficiency `t(1)/t(P)`:

      | phase | 1 rank | 2 ranks | 4 ranks | eff. 2 | eff. 4 |
      |---|---|---|---|---|---|
      | rhs | 16.65 | 20.12 | 22.56 | 0.83 | 0.74 |
      | fill_ghosts | 10.22 | 14.00 | 16.38 | 0.73 | 0.62 |
      | — its local groups | 9.91 | 11.43 | 14.16 | | |
      | — its packs | – | 1.63 | 1.68 | | |
      | — its unpacks | – | 0.29 | 0.30 | | |
      | scatter | 2.56 | 2.61 | 2.69 | 0.98 | 0.95 |
      | norm | 6.15 | 6.25 | 6.46 | 0.98 | 0.95 |
      | interfaces | 0.025 | 0.046 | 0.073 | | |
      | interpolate (1000 pts/rank) | 0.148 | 0.272 | 0.396 | 0.54 | 0.37 |
      | ghost_schedule | 3.82 | 6.30 | 6.57 | 0.61 | 0.58 |
      | interface_schedule | 0.13 | 1.23 | 1.35 | | |
      | regrid, refine | 46.0 | 57.4 | 65.5 | 0.80 | 0.70 |
      | regrid, coarsen | 30.4 | 45.5 | 44.7 | 0.67 | 0.68 |
      | triad reference | 0.61 | 1.22 | 2.20 | 0.50 | 0.28 |

      A fill sends, per rank, 2 messages at 2 ranks and 3 at 4 (phase 1
      to both neighbours, the prolongation stage to one), 717 KB each way
      in both cases, 560 transfers sent and 560 received against 4224
      local (4784 serially); the interface restriction one message of
      32 KB, 32 transfers. The uniform mesh: one stage, 410 KB, 288
      transfers; its rhs 3.26 / 4.25 / 4.56 ms, fill 1.19 / 1.70 / 2.51,
      build 0.40 / 1.20 / 1.26, refine 32.2 / 39.1 / 46.1. The refine
      makes ranks take 0/28/56 blocks from another (min/mean/max) at 2
      ranks and 0/42/84 at 4, the coarsening as many. At two threads a
      rank the two-level rhs is 8.72 / 11.53 / 22.47 ms (efficiency 0.76
      and 0.39) — eight threads on this machine reach its efficiency
      cores, and the window is the slowest rank's.
    - *What limits it here, named.* (1) **Memory bandwidth is shared**:
      the triad's aggregate is about 120 GB/s at any rank count, one
      core nearly saturates it, and so the per-rank triad halves with
      each doubling. Whatever streams memory cannot weak-scale on this
      machine, and the local groups of the fill do not (9.9 → 14.2 ms
      for *fewer* transfers); only the phases bound by one core's
      arithmetic do (`scatter`, the norm, 0.95). That is the reason the
      numbers are not a scaling result. (2) **The sender computes**: the
      packs are 1.6–1.7 ms, about a tenth of the fill. They are the
      halo's copies, restrictions and prolongations, evaluated into the
      send buffer rather than into ghosts, and the unpacks (0.3 ms) are
      the extra pass that copies them into place. (3) **The overlap**: the fill less its local groups, packs
      and unpacks is 0.7 ms at 2 ranks and 0.2 ms at 4, so over shared
      memory almost nothing of the messages is left unhidden behind the
      local groups; between nodes is what Symmetry will show. (4)
      **Per-call collectives on small work**: an interpolation of 1000
      points a rank costs an `allgather` and four `alltoallv`, 0.40 ms
      at 4 ranks against 0.06 ms in one 4-thread process over the same
      4-tile mesh (`TREEAMR_BENCH_TILES=4`); there the rhs and fill were
      within 5 % of the 4-rank run (22.1 against 22.7 ms, 15.9 against
      16.8), the schedule build 3.9 against 6.7 ms, the interface
      schedule 0.14 against 1.35 ms and the refine 48.8 against 65.4 ms.
      The interface schedule's growth is `remote_neighbors`, which
      searches all `3^D − 1` directions of every local leaf (about 1 ms
      here) for a builder that needs only faces, and which each
      schedule build repeats.
    - *Does the schedule build stay flat with `P`?* Yes. Step 2's
      finding — a rank's build costs about the serial one at 120 blocks
      a rank — was strong scaling: the halo is as large as the local
      part, so the build does not shrink with `P`. At a *fixed* number
      of blocks per rank it does not grow either. `bench/replicated.jl`
      simulates rank `P÷2` of `P` in one process, over a communicator
      that answers the digest gather by replication, hands `regrid!` the
      global marks and makes every message a no-op, over this tile mesh
      (176 blocks a rank, `N = 16`); best of 3–5, ms, at one thread and
      at four:

      | leaves (`P`) | 1408 (8) | 11264 (64) | 90112 (512) | 180224 (1024) |
      |---|---|---|---|---|
      | digest, fold over every leaf | 0.024 | 0.15 | 1.14 | 2.27 |
      | `GhostSchedule` on the rank, 1 thread | 6.3 | 6.7 | 8.1 | 9.1 |
      | the same, 4 threads | 4.3 | 4.4 | 5.6 | 6.8 |
      | `remote_neighbors`, 1 thread | 1.04 | 1.19 | 1.29 | 1.34 |
      | `complete_marks`, the slab, no buffer | 0.025 | 0.13 | 1.01 | 1.90 |
      | `balance!` alone | 0.005 | 0.033 | 0.26 | 0.52 |
      | `regrid_sources`, 1 thread | 0.17 | 1.27 | 12.1 | 29.0 |
      | the same, 4 threads | 0.17 | 1.01 | 8.7 | 19.4 |
      | `split_regrid` | 0.015 | 0.026 | 0.11 | 0.21 |
      | `complete_marks`, every block a `(Keep, box)` source, buffer 4, 1 thread | 8.2 | 74 | 672 | 1407 |
      | the same, 4 threads | 2.3 | 21 | 204 | 436 |
      | `regrid!` on the rank, refine, 1 thread | 21.7 | 21.9 | 51.7 | 67.2 |
      | `regrid!` on the rank, refine, 4 threads | 11.4 | 12.7 | 17.7 | 44.0 |
      | marks gathered per rank and regrid | 0.05 MB | 0.36 MB | 2.9 MB | 5.8 MB |

      Over 128 times the leaves the rank's build grows by 2.8 ms at one
      thread, 2.2 ms of it the digest, the rest the `log n` of the
      searches; so it is weak-scaling-flat, and the digest is the only
      `O(nleaves)` part of it (13 ns a leaf, single-threaded, folded up
      to three times per generation: `regrid!`, each `GhostSchedule`,
      each `InterfaceSchedule`). The `regrid!` times are noisy, the
      garbage collector's share included; a profile of one rank at 90112
      leaves (one thread, 20 regrids, 29 ms each) put 38 % in
      `regrid_sources`, 25 % in the fill before the transfer, 22 % in
      allocating and zeroing the new array (local, first touch), 7 % in
      `complete_marks` and
      7 % in the transfer stage.
    - *The replicated regrid bookkeeping, and what to do about it*
      (proposed here, and the first two implemented the same day; what
      was done, and measured, is the next item). Two passes become the
      regrid's cost long before the forest's memory does:
      1. **The buffer's recruit search** in `buffered_flags`, about
         5.8–7.8 µs per source leaf at one thread and 1.6–2.4 at four,
         over every source of the whole forest on every rank. With
         every block a source it overtakes the whole local regrid
         between 1408 and 11264 leaves (8.2 and 74 ms against 22) and
         is 13 times it at 90112 (672 against 52); at 32 ranks
         of 176 blocks (four Symmetry nodes) it would be some 35–40 ms
         at one thread, by the 1408- and 11264-leaf points, against a
         local regrid of 11–22 ms — an extrapolation, not measured
         there. The share of sources is the share of blocks whose
         criterion fired (`firing_boxes`), so a run with a wide feature
         pays it. *Remedy:* the recruits of a source are found from the
         tree alone, so each rank can search for its own sources only
         and gather the `(leaf, level)` pairs beside the marks; applying
         them only raises a mark to `Refine` below a level or lifts a
         `Coarsen` at or below one, which does not depend on their
         order, so the result is bit for bit the replicated one, at
         `O(local sources)` per rank and one `allgatherv` of the pairs.
      2. **`regrid_sources`** classifies every new leaf on every rank,
         with a `Dict` of every old leaf and a vector per new one: 12 ms
         at 90112 leaves, 29 ms at 180224, 38 % of a rank's regrid at
         512 ranks. *Remedy:* a rank needs only the new leaves it will
         own and those whose sources it owns now. Refinement and
         coarsening keep the curve order, so the second set is one
         contiguous range of new indices, found by two binary searches
         over the new leaves at the keys of the rank's first and last
         old leaf, and an old leaf is found by `searchsortedfirst`
         instead of the `Dict`: `O(local · log n)`. The layouts are
         sorted explicitly (`stage_layout`), so restricting the list
         changes no message.
      3. Smaller and left as they are: the digest (cache its leaf fold
         per generation if it ever shows; 1.1 ms at 90112 leaves is
         14 % of a build), `complete_marks`' own passes and `balance!`
         (1–2 ms at 10⁵ leaves, replicated in AMReX and Parthenon too),
         and the marks' `allgatherv`, 32 bytes a leaf to every rank,
         which a gather of the marks that are not a bare `Keep` would
         shrink when it matters.
    - *What was done about it* (2026-10-02, after the measurement; the
      results are unchanged, bit for bit):
      1. **The recruits.** `buffered_flags` is now two passes, both
         internal: `buffer_recruits`, the neighbour search over a run of
         sources, which returns `Recruit(leaf, level)` pairs in source
         order, and `apply_recruits!`, the serial rewrite of the marks.
         The public function runs both over every leaf, as before.
         `regrid!` runs `regrid_marks` instead: each rank searches from
         its own sources, reduces its recruits with
         `strongest_recruits`, gathers them in a second `allgatherv`,
         and applies the union to the gathered marks; then the
         completion, which is `complete_marks` after the buffer
         (`completed_leaves`). Serially nothing is gathered and the
         recruits are applied as found, which is `buffered_flags`.
         *Why the order does not matter*, checked against the code
         rather than assumed: the rewrite is two statements per recruit
         `(j, L)` at a leaf of level `l` — a `Coarsen` with `l ≤ L`
         becomes `Keep`, then anything with `l < L` becomes `Refine`.
         `Refine` is never undone, and a `Keep` made from `Coarsen` is
         only ever raised to `Refine`. So a leaf ends as `Refine` if any
         recruit asks it for more than its level, as `Keep` if it was
         `Coarsen` and some recruit asks for its level exactly, and as
         it was otherwise: a function of its own reported mark and of
         the largest level asked of it, whatever the order of the
         recruits and however often one repeats. Recruitment reads the
         *reported* marks, never the rewritten ones, so the search
         itself does not depend on the order either. That is also what
         lets a rank keep, per leaf, only its largest request, and drop
         a request at a leaf already finer, which changes no mark; so
         gathering in global source order, the fallback had the rewrite
         been order-dependent, was not needed. *Why a second gather and
         not the marks' own*: merging the two would need a
         byte-level encoding of a vector of padded structs with its
         per-rank counts, to save one collective whose cost is below
         the marks' gather. Measured over MPI on this laptop (one tile
         a rank, one thread, every block a `(Keep, box)` source, buffer
         4), at 2, 4 and 8 ranks: the recruits' `allgatherv` 0.015,
         0.034 and 0.078 ms with 1600 bytes from a rank, the marks'
         0.021, 0.074 and 0.188 ms with 5632; the whole `regrid_marks`
         0.97, 1.12 and 1.93 ms against the replicated `buffered_flags`
         1.88, 4.11 and 11.8 ms, with the same leaves at every count.
      2. **`regrid_sources`** takes `oldrange` and `newrange` and
         classifies `newrange` and `overlapping_leaves(oldrange)`: the
         new leaves from the one that covers the start of the rank's
         first old leaf (that leaf or an ancestor, at or before it in the
         curve's pre-order, else its first descendant, directly after
         it) through the last one at or before the deepest last
         descendant of its last old leaf. Without the ranges it
         classifies every leaf, as before, which is what the serial
         `regrid!`, `transfer_groups` and the tests use. A plain
         `searchsortedfirst` in place of the `Dict` was *slower* for the
         full classification — 14.1 against 11.6 ms at 90112 leaves,
         30.9 against 29.6 at 180224 — so the serial regrid would have
         paid for the distributed one. The search therefore starts at the
         previous target's sources (`findfrom`: four steps along the
         curve, then a binary search over the rest), since the sources
         of ascending new leaves are non-decreasing old leaves; that
         made the full classification faster than the `Dict`. The
         in-process lockstep test now builds every rank's stage from its
         own classification and checks that `split_regrid` keeps the
         same transfers from it as from the full one.
      3. **The digest's fold is not cached.** A cache per forest and
         generation goes stale whenever the leaves change without
         `rebuild_leaves!`, and nothing prevents that: `forest.leaves`
         is a public `Vector`, and `test/regrid_tests.jl` itself
         empties and refills one at generation 0. The digest exists to
         catch a forest that changed outside the collective contract,
         so it should not trust the generation to say the leaves are
         unchanged. Where to keep the cache is a second problem:
         `Forest` is immutable, and its positional constructor is called
         from `complete_marks`. The saving would be at most 1.1 ms a
         build at 90112 leaves.

      `bench/replicated.jl` (rank `P÷2`, 176 blocks a rank, ms, best of
      3–5) now also times the rank's own buffer and classification.
      Before and after on the same machine, run back to back, a stale
      copy of the tree for the "before":

      | leaves (`P`) | 1408 (8) | 11264 (64) | 90112 (512) | 180224 (1024) |
      |---|---|---|---|---|
      | every block a source, buffer 4: replicated `complete_marks`, 1 thread | 7.6 | 71 | 650 | 1365 |
      | on the rank: `regrid_marks` / with the completion, 1 thread | 1.0 / 1.1 | 1.1 / 1.6 | 1.4 / 5.4 | 1.6 / 9.6 |
      | replicated, 4 threads | 2.3 | 21.7 | 212 | 422 |
      | on the rank, 4 threads | 0.45 / 0.52 | 0.53 / 0.93 | 0.64 / 3.8 | 0.78 / 7.5 |
      | `regrid_sources`, every leaf, `Dict` (before), 1 thread | 0.165 | 1.26 | 11.6 | 29.6 |
      | the same by `findfrom` | 0.134 | 0.99 | 8.1 | 18.1 |
      | the rank's own (what `regrid!` runs) | 0.021 | 0.019 | 0.017 | 0.016 |
      | `regrid!` on the rank, refine / coarsen, 1 thread, before | 21.6 / 22.7 | 22.4 / 23.0 | 33.6 / 42.6 | 73.1 / 68.1 |
      | after | 21.9 / 22.9 | 22.7 / 22.8 | 20.7 / 21.0 | 23.2 / 23.6 |
      | the same, 4 threads, before | 11.1 / 11.0 | 11.4 / 11.7 | 22.5 / 17.4 | 35.7 / 41.4 |
      | after | 11.3 / 9.9 | 9.8 / 11.4 | 17.0 / 9.4 | 15.4 / 11.4 |

      So a rank's regrid no longer grows with the forest here: at
      180224 leaves it costs what it costs at 1408, within the noise
      the four-thread column shows. What remains replicated is the
      completion. It is the 4–8 ms above at 90112–180224 leaves when
      every block is a source, since these marks then refine every
      coarse leaf beside a fine one and `balance!` has a level 2 to
      check; with the slab's marks it is the 0.9–2.0 ms of item 3. A
      recruit gathers as 8 bytes, 0.82 MB a rank at 90112 leaves in
      that case, against 2.9 MB of marks.

      *Tests.* A new in-process testset in
      `test/regrid_exchange_tests.jl` runs `regrid_marks`, the
      completion and `regrid!` itself (one task per simulated rank, over
      the rendezvous communicator) on random forests with random bare and
      boxed flags, `D = 1, 2, 3`, buffer 0–4 and 1–7 ranks, a roots-only
      forest among them so that most ranks are empty, and requires every
      rank's marks and leaves to be the serial `buffered_flags` and
      `complete_marks` ones. Each of four deliberate breakages fails it
      or the lockstep test: skipping the gather, dropping a recruit at a
      leaf of the requested level, and moving either end of
      `overlapping_leaves`. The suite passes at one thread (110511
      tests, 8m08) and at eight (110563, 7m57), and the MPI workload at
      2, 3 and 4 ranks prints the serial lines byte for byte, apart from
      the 20 `sum` lines, which agree to roundoff (11, 11 and 9 of them
      byte for byte).
    - *A device.* `TREEAMR_BENCH_BACKEND=metal TREEAMR_BENCH_N=8
      TREEAMR_BENCH_REPS=3 TREEAMR_BENCH_PROJECT=<env> bench/mpiscan.sh
      1 2` (a scratch environment that develops this checkout and adds
      Metal, MPI and KernelAbstractions) runs in Float32 through the
      staging path at 2 ranks, every phase included; its numbers, two
      ranks sharing one GPU, say only that the path works.
    - *Symmetry* (measured 2026-10-02, job 567847 on cn092–cn095: AMD
      EPYC 7532, the Rome nodes of `amdq`, 64 cores and 8 NUMA domains
      each, ConnectX-6 HDR100 InfiniBand; Julia 1.13.1; HPC-X 2.20's Open
      MPI 4.1.7 over UCX 1.17, `nvhpc-hpcx-cuda12/24.9`, through
      MPIPreferences' system binary). `bench/symmetry_mpi.sh` as amended
      in the same step's script commit: one rank per NUMA domain bound by
      `numactl`, 8 threads, `D = 3`, `ROOTS = 4`, 2 variables, Float64,
      10 windows, the minimum over them. Launching took four fixes to
      the script and none to the package: `amddebugq` takes one node a
      job, so the job ran on `amdq`; SLURM 21.08 ignores
      `--ntasks-per-node` in a step unless `--ntasks` is given;
      MPICH_jll starts under `srun --mpi=pmi2` but runs TCP over IPoIB,
      2.1 GB/s and 40 µs ping-pong between nodes (job 567845), while
      HPC-X runs 12.0 GB/s and 2.2 µs (job 567846) and is not built with
      SLURM's PMI, so it is launched by its own `mpiexec`; and Open MPI
      hands each rank a pseudo-terminal, into which the juliaup launcher
      writes a terminal title, so the ranks run the julia binary itself.
      Minimum ms, two-level mesh (176 blocks a rank), one node at 1–8
      ranks, then 2 and 4 nodes at 8 a node; efficiency `t(1)/t(P)`:

      | `N = 16` | 1 | 2 | 4 | 8 | 16 | 32 | eff. 32 |
      |---|---|---|---|---|---|---|---|
      | rhs | 7.75 | 11.98 | 12.99 | 12.78 | 13.59 | 14.41 | 0.54 |
      | fill_ghosts | 4.37 | 8.40 | 8.86 | 8.87 | 8.93 | 9.64 | 0.45 |
      | — its local groups | 4.33 | 5.18 | 3.90 | 5.29 | 5.31 | 5.55 | |
      | — its packs | – | 2.73 | 2.97 | 2.98 | 2.95 | 2.93 | |
      | — its unpacks | – | 0.51 | 0.78 | 0.80 | 0.79 | 0.80 | |
      | scatter | 1.84 | 1.84 | 1.84 | 1.95 | 1.91 | 1.91 | 0.97 |
      | norm | 2.96 | 2.09 | 2.09 | 2.08 | 3.34 | 3.00 | 0.99 |
      | interfaces | 0.13 | 0.70 | 0.62 | 0.74 | 0.72 | 0.52 | |
      | interpolate (1000 pts/rank) | 0.24 | 0.77 | 0.83 | 0.61 | 0.69 | 10.24 | 0.02 |
      | ghost_schedule | 8.38 | 11.79 | 11.65 | 12.10 | 11.70 | 11.73 | 0.71 |
      | interface_schedule | 0.33 | 1.54 | 1.23 | 1.63 | 1.36 | 1.36 | |
      | regrid, refine | 25.6 | 28.9 | 26.4 | 27.0 | 24.9 | 32.5 | 0.79 |
      | regrid, coarsen | 21.9 | 28.9 | 33.9 | 30.5 | 33.7 | 38.8 | 0.56 |
      | triad reference | 1.70 | 1.74 | 2.13 | 2.18 | 2.26 | 2.15 | 0.79 |

      | `N = 32` | 1 | 2 | 4 | 8 | 16 | 32 | eff. 32 |
      |---|---|---|---|---|---|---|---|
      | rhs | 37.9 | 47.7 | 49.4 | 58.2 | 55.0 | 55.9 | 0.68 |
      | fill_ghosts | 18.7 | 29.2 | 28.6 | 27.7 | 29.8 | 30.1 | 0.62 |
      | — its local groups | 18.7 | 17.0 | 17.0 | 17.0 | 16.9 | 17.1 | |
      | — its packs | – | 8.58 | 8.58 | 8.55 | 8.96 | 9.07 | |
      | — its unpacks | – | 2.04 | 1.73 | 1.77 | 1.99 | 2.09 | |
      | scatter | 8.02 | 8.69 | 8.71 | 9.38 | 9.96 | 11.98 | 0.67 |
      | norm | 15.5 | 15.2 | 15.9 | 16.2 | 16.0 | 16.9 | 0.92 |
      | interpolate | 0.19 | 0.78 | 0.94 | 0.63 | 0.86 | 8.28 | 0.02 |
      | ghost_schedule | 13.2 | 16.0 | 16.4 | 16.2 | 16.2 | 16.3 | 0.81 |
      | regrid, refine | 162 | 174 | 192 | 163 | 154 | 162 | 1.00 |
      | regrid, coarsen | 134 | 172 | 224 | 225 | 217 | 217 | 0.62 |
      | triad reference | 13.3 | 13.6 | 13.8 | 14.5 | 14.4 | 13.8 | 0.96 |

      A fill sends, per rank, 2 messages at 2 ranks and 3 from 4 on
      (716800 bytes each way at `N = 16`, 2.3 MB at 32; 560 transfers
      sent and 560 received against 4224 local); the regrid makes ranks
      take 0/50/108 blocks from another (min/mean/max) at 32 ranks. The
      uniform mesh (64 blocks a rank, one stage): rhs 2.05 / 3.75 / 2.62 /
      3.92 / 3.85 / 3.83 ms at `N = 16` (efficiency 0.53 at 32) and 9.84
      / 13.4 / 15.3 / 16.3 / 16.2 / 16.9 at 32 (0.58). What it shows:
      1. **The step from one rank to two is the whole loss, and the
         network is not in it.** From 2 to 32 ranks, across one, two and
         four nodes, the two-level rhs grows by 20 % at `N = 16` (12.0 →
         14.4 ms) and 17 % at 32 (47.7 → 55.9): efficiency 0.83 and 0.85
         against 2 ranks. The fill less its local groups, packs and
         unpacks — what the messages left unhidden — is −0.2 to 0.4 ms
         at `N = 16` and 0.4 to 2.0 ms at 32 at every count, inside the
         error of subtracting separately measured minima; 2.3 MB over
         UCX at 12 GB/s is 0.2 ms. The overlap does its job, and
         between nodes too.
      2. **The packs are the cost**: 2.7–3.0 ms of an 8.9 ms fill at
         `N = 16`, 8.6–9.1 of 30 at 32, against local groups of 5.3 and
         17.0 ms for 7.5 times as many transfers. So a packed transfer
         costs about 4 times a local one. The likely reason, not
         measured: pack groups run by the owner of the *source* block
         ("Ownership" under "Pack and unpack are transfers"), and the
         sources of a rank's halo are the bottom and top layers of its
         tile, which sit at the two ends of its curve range and so in
         the first and last thread chunks: at 8 threads the packs for
         the rank below run on one thread, those for the rank above on
         about four. The sender-computes rule moves the halo's
         arithmetic to the sender, as designed; what costs is that the
         block-ownership partition then gives it to a few threads. A
         pack split over all threads by transfer, not by owner, would
         trade affinity for balance, which is a design question.
      3. **The schedule build and the regrid are flat from 2 ranks on**,
         as `bench/replicated.jl` predicted: the build 11.7–12.1 ms at
         `N = 16` and 16.0–16.4 at 32 at every count from 2 to 32; the
         refining regrid 25–33 ms and 154–192 ms. The coarsening regrid
         rises from 1 to 4 ranks, by 55 % at `N = 16` and 67 % at 32,
         and is flat after at `N = 32`; at 16 it rises again at 32 ranks
         (not traced).
      4. **`interpolate!` at 32 ranks jumps to 8–10 ms**, from 0.6–0.9
         ms at 2–16 ranks; at 16 ranks (two nodes) its *median* is
         already 9.9 ms. Nothing else in the table does this, and
         MPICH_jll over TCP at 16 ranks takes 2.0 ms (below), so it is
         the collectives — an `allgather` and four `alltoallv` — inside
         this Open MPI at that scale, not the work. A job that repeats
         the 16- and 32-rank runs with HPC-X's HCOLL collectives on and
         off (567858) was queued and its result not collected; the cause
         is open.
      5. **Phases without messages slow by the slowest of 32 ranks**:
         `scatter` 8.0 → 12.0 ms at `N = 32` and the uniform mesh's
         local groups 2.5 → 4.3, while the triad stays at 0.96. The
         window is the slowest rank's.
    - *One node, the same mesh: 8 ranks against one process* (the same
      job; the control is 64 threads over the 8-tile mesh in one
      process). The one process wins the per-evaluation phases. Minimum
      ms, two-level, 8 ranks / one process pinned with first touch
      (`JULIA_EXCLUSIVE=1`) / one process interleaved:

      | | `N = 16` | `N = 32` |
      |---|---|---|
      | rhs | 12.78 / 7.98 / 11.97 | 58.2 / 39.4 / 50.5 |
      | fill_ghosts | 8.87 / 4.78 / 7.82 | 27.7 / 19.8 / 28.2 |
      | ghost_schedule | 12.1 / 16.7 / 17.1 | 16.2 / 23.0 / 19.9 |
      | regrid, refine | 27.0 / 48.1 / 44.3 | 163 / 233 / 245 |
      | regrid, coarsen | 30.5 / 48.9 / 41.4 | 225 / 221 / 218 |

      One pinned process runs the rhs 1.6 times as fast at `N = 16` and
      1.5 times at 32 (the uniform mesh 1.3 and 1.6), because the 8-rank
      run pays the packs and unpacks of item 2 and the one process pays
      nothing for the same halo: its local groups are the 8 ranks' local
      groups plus the halo, at about the same time (4.95 against 5.29
      ms at `N = 16`). That settles what "What one process loses" left
      for this step: the ownership policy recovers inside one process
      everything one rank per domain was expected to, and on one node
      ranks are a loss for the rhs. The replicated host passes go the
      other way — the schedule build and the refining regrid are
      1.4–1.8 times faster over 8 ranks, being per rank `O(local)` work
      on fewer threads — but they run at regrid frequency. So the
      layout to recommend is one process per node, its threads pinned,
      and ranks between nodes.
    - *The default binary between nodes* (job 567857, cn109–cn110, the
      Milan EPYC 7543 nodes, MPICH_jll 5.0.2 through `srun --mpi=pmi2`,
      `N = 16`; compare within the job only). On one node it matches
      HPC-X's shape (two-level rhs 6.32 / 11.83 / 12.31 / 13.66 ms at
      1–8 ranks); on two nodes the rhs is 18.52 ms and the fill 13.42,
      against 13.66 and 9.48 at 8 ranks, so TCP over IPoIB adds 4 ms a
      fill that UCX does not (efficiency 0.34 at 16 ranks, against 0.57
      for HPC-X at 16 in job 567847). Between nodes the system MPI is
      not optional.
    - *The replicated costs on a rank's domain* (`bench/replicated.jl`
      in job 567847, one domain of cn092, 8 threads, 176 blocks a rank;
      ms):

      | leaves (`P`) | 1408 (8) | 11264 (64) | 90112 (512) | 180224 (1024) |
      |---|---|---|---|---|
      | digest | 0.036 | 0.19 | 1.40 | 2.79 |
      | `GhostSchedule` on the rank | 11.9 | 11.7 | 13.0 | 14.3 |
      | `regrid_sources`, every leaf / the rank's own | 0.40 / 0.17 | 1.94 / 0.18 | 15.8 / 0.16 | 32.3 / 0.17 |
      | every block a source: replicated `complete_marks` | 3.6 | 33 | 290 | 596 |
      | on the rank: `regrid_marks` / the completion | 0.78 / 1.04 | 0.84 / 2.15 | 0.89 / 12.2 | 1.26 / 20.4 |
      | `regrid!` on the rank, refine / coarsen | 23.1 / 18.5 | 19.2 / 22.8 | 19.5 / 19.4 | 22.4 / 23.9 |

      So the laptop's finding holds on the cluster: a rank's regrid
      costs at 180224 leaves what it costs at 1408. What remains
      replicated and grows is the completion after the buffer, 20 ms at
      180224 leaves when every block is a source, about a regrid's
      worth; the build grows by 2.4 ms over 128 times the leaves, the
      digest most of it.
    - *What was not done.* The benchmark itself changed nothing in
      `src/`, so the suite was not rerun for it; the follow-up above
      did, and it was.
      The fill's overlap is read from the difference of its parts, not
      traced inside `run_stage!`. The benchmark's mesh is periodic, so
      the boundary hook is not timed.
  - **Step 8 — MPI+GPU.** Device-resident buffers, the device-aware
    check and host staging. *Accept:* the workload on Metal through the
    staging path, and on an H200 on Symmetry, agreeing with the serial
    device run as [Parallelism](CODE.md#parallelism) states for a device.

    *(Done locally, 2026-10-01, and on Symmetry's H200s, both paths, on
    2026-10-02.)*
    What it settled, and where it went beyond the plan (the design
    decisions are recorded under "MPI+GPU" in
    [Distributed meshes](CODE.md#distributed-meshes)):
    - *The code.* `run_stage!` in `ghosts.jl` is the one place every
      stage runs — the ghost fill, the interface restriction and the
      regrid transfer — so it is the one place that stages: it asks the
      new verb `hoststaging(comm, buffer)`, and when the answer is yes it
      posts the receives into host mirrors, downloads the packed buffer
      after the pack's synchronization, sends from the mirror, and
      uploads the receive mirror before the unpack. `stagemirrors`
      allocates the mirrors once per stage and variable count, kept in
      `RemoteStage`'s new `mirrors` field, and page-locks them with
      `KernelAbstractions.pagelock!`. The MPI extension gains the
      `deviceaware` keyword of `communicator` and a field of
      `MPICommunicator` for it, answers `hoststaging`, and checks a
      message buffer against the setting. No new dependency, weak or
      otherwise.
    - *An inference fix found by the new test.* The in-process staging
      test defines a fourth test `Communicator` with an `allgather`
      method, and with it `type_tests.jl`'s `@inferred total_mass` and
      `@inferred volume_weighted_norm` failed: `combine_blocks` calls
      `allgather` on the abstractly typed `comm` field, and with that
      many methods inference returns `Any` for the call, and the fold
      over its result with it. Nothing about a reduction's type should
      depend on how many communicator types a process has loaded — an
      application with its own would have hit it too — so the gather is
      now asserted, `allgather(comm, partial)::Vector{typeof(partial)}`,
      which every `allgather` returns.
    - *The device workload.* `test/mpi_device_workload.jl`, standalone and
      not in `Pkg.test`, since the test environment has no device package
      and must not gain one: the vertex-centered wave on three levels with
      the hook, a reflecting box with an odd variable, a periodic 3D
      mesh, each filled and stepped with RK4; Burgers with the interface
      fixup and its conservation; the tracked pulse through two regrids
      flagged by `firing_boxes` on the device; and refinements of the
      first blocks and their coarsening, which move blocks up and down
      the ranks and coarsen siblings with different owners, over a
      conservative cell-centered set and a vertex-centered one. Rank 0
      prints digests of the arrays downloaded and gathered in block
      order, ghosts included, the exact reductions, the sums, and `#`
      lines with how many of each schedule's stages had messages and how
      many were staged, and how blocks migrated. Every callback computes
      in the coordinates' type (the forests' extents are in `T`), since
      Metal has no Float64. `TREEAMR_TEST_BACKEND` chooses `cpu`, `metal`
      or `cuda`, `TREEAMR_TEST_T` the type, and
      `TREEAMR_TEST_DEVICEAWARE=1` the direct path; on a node with several
      devices each rank takes device `local rank mod devices`.
      `test/mpi_device_tests.jl` runs it serially and under `mpiexec` at
      `TREEAMR_TEST_RANKS` (default `2 3`), one thread a rank, and
      requires every line to be the serial device run's, the `sum` lines
      to `rtol = 10⁻⁵` in Float32 and `10⁻¹²` in Float64; every stage with
      messages to have been staged on a device that is not device-aware,
      and none otherwise; and the migrations to have happened.
    - *Measured on Metal* (Apple M3 Pro, Metal.jl in a scratch environment
      that develops this checkout and adds Metal, MPI, KernelAbstractions,
      SHA and Test; MPICH_jll 5.0.2; two and three ranks sharing the one
      GPU). `TREEAMR_TEST_BACKEND=metal julia --project=<env>
      test/mpi_device_tests.jl` passes, 36 tests in 1m45: the serial run
      31 s, `-n 2` 36 s and `-n 3` 38 s of wall clock, nearly all of it
      compilation. In Float32 every line but the sums is the serial Metal
      run's byte for byte, at both counts; at `-n 3` the sums move in the
      seventh or eighth of 9 digits (Burgers' mass 15.999999 against 16,
      `l2` by 1–2 × 10⁻⁷ relative). At `-n 3` 9, 8, 6, 9 and 5 stages of
      the five schedules had messages, and all of them were staged;
      6 blocks moved up the ranks on the refinement, 6 back down on the
      coarsening, and 2 coarsened blocks had children of two owners. As a
      negative control, with the upload of the receive mirror removed the
      `-n 2` run differs from the serial one in 53 lines. The same driver
      on the CPU (the direct path, nothing staged) passes in 59 s.
    - *In process*, the staging path on the CPU: a communicator in
      `regrid_exchange_tests.jl` that answers `hoststaging` with `true`
      for every buffer, over the rendezvous communicator, and records the
      arrays every message is handed out of. At 3 ranks in 2D, vertex-
      centered, outer and reflecting faces with the hook, the staged ghost
      fill, the interface restriction and `regrid!` reproduce the serial
      results bit for bit; every buffer handed to the communicator is one
      of the stages' mirrors and none is a stage buffer; a second fill
      reuses the mirrors and gives the same bits. With the upload removed
      it fails, 7 of its 31 tests. `mpi_workload.jl` gains a `#` line that
      checks, under MPI, that a device-aware communicator shares the
      duplicate, and that a buffer which is not contiguous host memory is
      refused before anything is sent, naming the setting.
    - *The CPU path is unchanged.* `bench/ghosts.jl` at its defaults, one
      thread, best of 200, HEAD before and after, alternated twice: the
      serial fill allocates exactly what it did, 10896 bytes on the
      uniform mesh and 97504 on the two-level one, and the schedule build
      392832 and 4656880; the fill took 2.660–2.719 ms against
      2.667–2.670 ms and 12.09–12.22 ms against 11.87–11.92 ms, the second
      1.5–3 % slower in both rounds, though the serial path runs no
      changed line (the stage without messages returns before the new
      code); not traced further. A distributed CPU fill (the same
      two-level mesh, 10 variables, `mpiexec` with one thread a rank,
      best of 50) allocates exactly what it did, 130320 bytes on each rank
      at `-n 2` and 151856 / 189168 / 151856 at `-n 3`, in 7.0 ms and
      6.1–6.9 ms both before and after.
    - *The CPU workload by hand*, `test/mpi_workload.jl` at `-n 3` against
      the serial run (before the inference fix above, which changes no
      value): every line but the `sum` lines and the `#` lines identical,
      the sums agreeing to the last one or two of 17 digits, as in step 6.
    - *Symmetry* (measured 2026-10-02 on cn111, one node of 8 H200s
      with 4 of them allocated, NVLink between every pair; Julia 1.13.1,
      CUDA.jl's CUDACore 6.4.1; `bench/symmetry_mpi_gpu.sh` as amended
      in the step's script commit, on `h200q`, since `h200debugq`'s QOS
      allows a group 2 GPUs and 24 CPUs). One rank per GPU, one thread a
      rank, `TREEAMR_TEST_RANKS = "2 3 4"`:
      - **Host staging, MPICH_jll 5.0.2** (job 567852):
        `test/mpi_device_tests.jl` passes in Float64 and in Float32, 52
        tests each, every line of every distributed run the serial CUDA
        run's but the sums, every stage with messages staged. The first
        job (567850) had failed at `-n 2`: CUDA refuses to page-lock an
        empty range, and a stage this rank only sends in, or only
        receives in, has an empty mirror, which Metal and the CPU never
        refused. `pagelock_mirror!` now skips an empty mirror (the
        step's fix commit), and nothing else changed between the jobs.
      - **Symmetry has a CUDA-aware MPI**: HPC-X 2.20's Open MPI 4.1.7
        (`nvhpc-hpcx-cuda12/24.9`, through MPIPreferences' system
        binary), whose `ompi_info` says `opal_built_with_cuda_support`
        and for which `MPI.has_cuda()` is `true`. Over it (job 567856)
        the test passes through host staging and through the direct
        path, `communicator(COMM_WORLD; deviceaware = true)`, in
        Float64 and in Float32: four times 52 tests, the direct runs
        with no stage staged. So the direct path, which no device here
        could run, is checked — MPI.jl hands the `CuArray` views to the
        library by device pointer, and UCX moves them. The probe of
        `MPI.has_cuda()` has to run under `mpiexec -n 1`: a singleton
        `MPI.Init()` of this Open MPI inside a SLURM job waits forever
        for the daemon it spawns (job 567853, cancelled).
      - **What the two paths cost** (`bench/mpi.jl` on CUDA through
        `bench/mpiscan.sh 1 2 4`, `N = 32`, `ROOTS = 4`, Float64, 5.8 M
        cells a rank, the four GPUs of one node, HPC-X; minimum ms,
        staged / direct):

        | phase | 1 rank | 2 ranks | 4 ranks |
        |---|---|---|---|
        | two-level rhs | 2.07 / 2.06 | 2.89 / 2.72 | 2.92 / 2.76 |
        | two-level fill_ghosts | 1.71 / 1.70 | 2.49 / 2.29 | 2.52 / 2.35 |
        | — its local groups | 1.69 / 1.67 | 1.44 / 1.41 | 1.44 / 1.43 |
        | — its packs | – | 0.52 / 0.51 | 0.51 / 0.50 |
        | — its unpacks | – | 0.44 / 0.44 | 0.44 / 0.44 |
        | uniform rhs | 0.31 / 0.31 | 0.63 / 0.61 | 0.71 / 0.63 |
        | two-level regrid, refine | 5.2 / 5.2 | 25.4 / 7.1 | 48.0 / 8.7 |
        | two-level regrid, coarsen | 3.6 / 3.6 | 26.1 / 7.3 | 52.6 / 8.2 |
        | interpolate (1000 pts/rank) | 0.11 / 0.11 | 0.55 / 0.55 | 0.47 / 0.50 |

        The rhs weak-scales at 0.71 staged and 0.74 direct to 4 GPUs.
        As on the CPU, the messages are not the cost: the fill less its
        parts is under 0.2 ms either way. The packs and unpacks are,
        0.95 ms of a 2.5 ms fill, and on a device that is launches —
        one per group, for groups of a few thousand points (the
        uniform mesh's single stage has 0.12 + 0.12 ms of them for a
        fill of 0.16 ms serially). The direct path saves 0.16–0.2 ms a
        fill, the two copies. **The staged regrid is the surprise**: 48
        ms at 4 ranks against 8.7 direct and 5.2 serially. A regrid
        stage is built per regrid, so its host mirrors are allocated,
        zero-filled and page-locked afresh each time, for up to 84
        blocks of `32³ × 2` values; that is the likely cost (not
        traced), and MPICH_jll's staging (job 567852) shows the same
        48.4 ms. A regrid that staged through pageable memory, or kept
        its mirrors, would not pay it; left open, since a regrid runs at
        regrid frequency and the direct path exists. *(Followed up the
        same day with the buffer pool, the next bullet; the H200
        re-measurement is pending.)*
    - *The buffer pool* (the follow-up, 2026-10-02; the design is "The
      buffer pool" under "MPI+GPU" in
      [Distributed meshes](CODE.md#distributed-meshes)). Stage buffers and host
      mirrors are leased from the forest's pool, the regrid stage's
      returned at the end of its transfer and a stale schedule's
      reclaimed at the next lease, so that a regrid reuses what the
      earlier ones allocated and, on CUDA, page-locked. Measured locally
      only (Apple M3 Pro, MPICH_jll 5.0.2, one thread a rank, Symmetry
      being in use by another job):
      - *Reuse.* In process, `regrid_exchange_tests.jl`'s new test runs
        a refine-and-coarsen cycle three times at 3 ranks with every
        message staged on the CPU, schedules rebuilt after each regrid
        and filled, with the serial bits throughout and no buffer
        allocated in the third cycle. A script running five cycles of
        the same setup shows the three ranks' pools allocating 8, 10
        and 8 buffers in the first cycle (half of them mirrors) and none
        in the four after it. `mpi_device_workload.jl` runs its
        moving-blocks cycle twice and prints the ranks' pool counts after
        each: on Metal at `-n 3`, 42 buffers (21 mirrors) and 48 (24)
        after the first cycles of its two meshes, the same after the
        second, and `mpi_device_tests.jl` now asserts that, 52 tests
        passing on Metal (staged) and on the CPU (direct, no mirrors).
        `bench/mpi.jl` over its six regrid cycles at `N = 16`: rank 0's
        pool allocates 20 buffers at 2 ranks and 18 at 4 in the first
        cycle, staged on Metal or forced-staged on the CPU, 10 and 9
        direct on the CPU, and nothing after.
      - *Negative controls.* With reclaiming disabled, the pool's unit
        test fails 4 of its 9 tests; with a leased buffer left on the
        free list, the unit test fails 2 and the cycle test's overlap
        check fails on all 3 ranks — while its bits stayed serial,
        since the in-process mailbox copies a message when it is sent,
        which is why the overlap is checked directly.
      - *Host allocation per regrid*, rank 0, `bench/mpi.jl`'s last
        refine / coarsen (`@allocated`, `N = 16`, `ROOTS = 4`, 2 / 4
        ranks), before and after: forced staging on the CPU 57.0 / 48.4
        and 55.2 / 52.2 MB to 46.8 / 35.5 and 41.4 / 35.6; direct on the
        CPU 51.9 / 42.0 and 48.3 / 43.9 to 46.8 / 35.5 and 41.3 / 35.6;
        staged on Metal (Float32) 5.3 / 6.5 and 6.3 / 7.4 to 2.7 / 3.2
        and 2.8 / 3.2. What remains on the CPU is mostly the new
        working arrays, which a regrid allocates by design.
      - *Time: no change that the noise lets one see* — the cost the
        pool removes is not one this machine has. Two runs each,
        alternated, minimum ms, refine / coarsen at 2 and 4 ranks:
        forced staging on the CPU 61.2–62.8 / 50.2–50.9 and
        69.1–72.6 / 48.4–48.7 before, 61.5–63.8 / 49.6–54.5 and
        71.3–71.6 / 44.0–44.5 after; Metal 40.2–51.1 / 40.0–42.4 and
        51.4–56.0 / 50.5–50.8 before, 47.9–50.3 / 40.7 and 51.2–52.8 /
        48.2–49.9 after (the ranks share one GPU, so the Metal
        distributed numbers are not weak scaling). Neither backend
        page-locks, and the allocation and zero-filling the pool does
        save here are evidently lost in a 40–70 ms regrid.
        So the local runs show that the pool works and is safe, not
        what it buys on CUDA: whether step 8's 40 ms were the pinning
        is still to be measured on Symmetry (pending; see "Performance
        work left for later" under
        [Distributed meshes](CODE.md#distributed-meshes)).
      - *Unchanged elsewhere.* `bench/ghosts.jl` at one thread: the
        serial fill allocates 10896 and 97504 bytes and the schedule
        build 392832 and 4656880, as in step 8; the distributed CPU
        fill of step 8 allocates 130320 bytes a rank at `-n 2` and
        151856 / 189168 / 151856 at `-n 3`, as before.
    - *What was not checked* (before the Symmetry run, which checked the
      direct path on CUDA). The direct path on any device: Metal has no
      device-aware MPI, and the CPU's direct path is not the device's
      code path for MPI.jl, which hands a `CuArray` over through its
      own CUDA extension. A checkpoint or an interpolation from a device
      field set under MPI was not run (both go through the host, as steps
      5 and 6 recorded).
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 passes
      `partition_tests.jl`, `exchange_tests.jl`,
      `regrid_exchange_tests.jl` (the staging test included),
      `type_tests.jl` after them (the inference fix) and `mpi_tests.jl`
      (3m21 together); the whole suite was not run there.
    - *Suite cost.* 109140 tests at one thread in 7m55 and 109192 at eight
      in 7m51, against step 6's 109107 in 8m10 and 109159 in 8m02: 33
      more at each, the 31 of the staging test and the device-aware line
      at two rank counts. The thread-independence digests are unchanged.
      The docs build.
  - **Step 9 — wrap-up.** A `docs/src/api/distributed.md` page
    (`communicator`, `blockrange`), a guide section in `index.md`, the
    status there and in `README.md`, and CLAUDE.md's architecture row,
    MPI test commands and suite cost. *Accept:* the docs build; TreeWave
    and TreeHydro green in scratch copies against the checkout, with
    the audit for `nleaves`-sized per-block arrays done in all three
    downstreams; M7 marked *(Done.)*. Tagging the release is left to
    Erik. *(Amended in step 9: M7 is marked implemented rather than
    done, since the acceptance's weak-scaling table and the H200 run
    are scripted and not run; see the milestone's heading.)*

    *(Done, 2026-10-02.)* What it settled:
    - *Suite cost, and what was trimmed.* After step 8 the suite took
      about 8 minutes at one thread, against M9a's 4m16, and
      `mpi_tests.jl` was the largest file: 123–128 s, the serial
      reference in process and then `mpiexec -n 3` and `-n 2` one after
      the other, each launch about 55 s of wall clock and, by a timed
      copy of the workload, almost all of it compilation (a fresh serial
      process spends about 20 s compiling `main`'s call tree before its
      first case runs, and the cases' own time is about 19 s). Nothing
      orders the launches but the checkpoint cross loads, so they now
      overlap. Each run leaves an empty marker beside its checkpoints
      once they are written, and `TREEAMR_CHECKPOINT_FROM` names the
      rank counts whose files a run waits for and loads; every rank
      waits on its own, sleeping, so none spins in MPI meanwhile, and a
      run by hand without the variable loads whatever it finds, as
      before. A new helper, `test/mpi_jobs.jl`, starts both jobs at the
      start of the suite where the machine has room for five ranks
      beside it — eight threads and 24 GB, since a rank measured about
      2 GB resident — and `mpi_tests.jl` collects them; elsewhere, a CI
      runner among them, the three-rank job runs beside the serial
      reference and the two-rank job after it. *(Amended 2026-10-06:
      there the three ranks now start after the reference. Beside it,
      on CI's macOS runners — 3 cores, 7 GB, Julia 1.13 — three ranks
      of about 2.4 GB each compiling beside the reference and the
      suite's process missed the job's 900 s deadline: once on `main`
      on 2026-10-04, and in two cells of the copy-kernel change, whose
      first version had also slowed the workload's compilation (see
      [The copy kernels on a device](CODE.md#the-copy-kernels-on-a-device)).
      On a laptop the MPI test alone takes 3m46 that way, against 1m47
      with both jobs beside the reference.)*
      `TREEAMR_TEST_MPI_CONCURRENT=0` or `1` overrides the choice. Two
      type repeats went, each of a path covered elsewhere: the
      workload's `Float32` regrid case (`Float32` crosses MPI as a
      native type, its regrid stage is checked bitwise in process by
      `regrid_exchange_tests.jl`, and `Float32x2`, which MPI sends
      through a derived datatype, stays), and `exchange_tests.jl`'s
      pack/unpack round trip in `Float32` in 1D and 3D and in `Float32x2`
      in 1D, 29 tests (`Float32` stays in 2D, `Float32x2` in 2D and 3D,
      where remote mirrored transfers exist). No claim of an acceptance
      list went with them: every case the step-3 to step-6 acceptances
      name is still in the workload, at both rank counts, and every type
      the step-2 round trip names is still in it. The thread-independence
      test was not touched.

      Per file, one thread, timed around each `include` in a scratch
      copy of `runtests.jl` (s; before and after on the same machine,
      the same day):

      | file | before | after |
      |---|---|---|
      | inline M1 tests | 10.7 | 14.7 |
      | `ghost_tests.jl` | 14.2 | 18.0 |
      | `centering_tests.jl` | 25.7 | 30.5 |
      | `reflect_tests.jl` | 28.7 | 29.5 |
      | `interpolate_tests.jl` | 18.0 | 23.4 |
      | `interface_tests.jl` | 5.2 | 5.1 |
      | `partition_tests.jl` | 8.8 | 8.2 |
      | `exchange_tests.jl` | 40.1 | 32.8 |
      | `regrid_exchange_tests.jl` | 20.3 | 19.0 |
      | `interpolate_exchange_tests.jl` | 13.2 | 12.9 |
      | `allvariables_tests.jl` | 8.3 | 8.1 |
      | `state_tests.jl` | 7.6 | 7.5 |
      | `regrid_tests.jl` | 10.4 | 10.8 |
      | `wave_tests.jl` | 4.2 | 4.2 |
      | `wave_cell_tests.jl` | 2.3 | 2.3 |
      | `burgers_tests.jl` | 16.6 | 17.2 |
      | `imex_tests.jl` | 5.6 | 5.3 |
      | `type_tests.jl` | 12.8 | 13.4 |
      | `checkpoint_tests.jl` | 26.6 | 28.6 |
      | `thread_tests.jl` | 50.2 | 50.3 |
      | `mpi_tests.jl` | 127.8 | 15.0 |
      | `gpu_tests.jl` | 27.5 | 27.0 |
      | whole run | 489.6 | 386.5 |

      The files that run while the jobs compile lose 4–6 s each, about
      20 s in all, to the five ranks on the other cores (an M3 Pro, six
      performance and six efficiency cores); the net is 103 s. On the
      sequential path, which is CI's, `mpi_tests.jl` alone takes 1m47
      (130 tests), against step 6's 2m34 for three launches in a row,
      and the whole suite, forced onto that path here with
      `TREEAMR_TEST_MPI_CONCURRENT=0`, 7m52 (110484 tests), within the
      run-to-run spread of the 7m55 before (steps 5–8 measured 7m49 to
      8m10). So the trim is a local saving: a CI cell, whose runner has
      no room for the jobs beside the suite, is not expected to get
      faster, and launching them there anyway was not tried, since a
      `macos-latest` runner has 7 GB for the main process and five ranks
      of about 2 GB.
    - *CI.* `.github/workflows/CI.yml` needed no change for MPI: MPI.jl's
      default binary is MPICH_jll, an artifact on both runners'
      platforms, and the parallel HDF5_jll build matching it is chosen
      by MPIPreferences with nothing configured ("What the feasibility
      check found"); the ranks run at one thread whatever
      `JULIA_NUM_THREADS` says, and inherit `--check-bounds=yes` and,
      on the single-threaded cells, `--code-coverage` through
      `Base.julia_cmd()`, so their `.cov` files are merged with the
      rest. The job gained a `timeout-minutes` of 120, since a process
      stuck outside the test's own deadlines would hold a runner for
      GitHub's default six hours. **None of it has run on GitHub**: the
      branch is not pushed. What is unverified there: that MPICH's
      launcher starts on the hosted macOS runner; the memory of a
      `macos-latest` runner (7 GB) against the main process and three
      ranks of about 2 GB each, which may swap; and the time of the
      coverage cells, whose ranks are instrumented too (the deadline is
      900 s a launch, against about 55 s uninstrumented here).
      *(Linux, 2026-10-02: in a fresh clone of the branch at 12a6cd6 on
      one Symmetry node each — jobs 567843 on cn106 and 567844 on cn109,
      AMD EPYC 7543 — `Pkg.test()` at one thread passes on Julia 1.11.9,
      110484 tests in 13m34 (`--check-bounds=yes`, which 1.11's
      `Pkg.test` sets), and on 1.13.1, 110484 in 11m18, the MPI jobs
      started early on both, with MPICH_jll 5.0.2 resolved by default.
      `HYDRA_LAUNCHER=fork` was set, so that MPICH's launcher forked its
      ranks as on a runner without SLURM, rather than starting them
      through `srun`. So the MPI tests work on Linux; GitHub's runners
      are what CI is for.)*
    - *Docs.* A guide section, "Running distributed", in
      `docs/src/index.md`; the status there and in `README.md`; the
      distributed API page pointing at the guide. Two examples that
      were right only serially were fixed: `firing_boxes`' verdict used
      `level(forest.leaves[b])`, and the regridding section named
      `forest.leaves[b]` as the way to tell blocks apart; both are
      `blockkey(fs, b)`. The docs build.
    - *Fixed upstream from the audit.* `buffered_flags` now refuses a
      flag vector that is not one per leaf, as `complete_marks` already
      did: TreeGeneralizedHarmonic calls it with `flag_blocks`' flags,
      which over a distributed forest are a rank's local blocks, and
      without the check it buffered around leaves `1:nblocks` and wrote
      global recruit indices into the local vector — silently. The
      `DimensionMismatch` says to pass the local flags to `regrid!` with
      `buffer` instead. And `block_mapreduce`'s docstring had shown
      `maximum(block_mapreduce(…))` as a refinement scale, the
      rank-local pattern two downstreams copied; it shows
      `mesh_mapreduce` now and says why.
    - *The downstreams*, each copied to scratch with this checkout
      developed into the copy, at one thread: TreeWave 310 tests in 1m22
      (`Pkg.test`'s summary; 310 in 1m12 against the M9a checkout);
      TreeHydro 12447 in 5m00 (11893 in 4m16 then; it has grown since);
      TreeGeneralizedHarmonic, whose whole suite takes about 19 minutes,
      a subset of 1760 tests in 7m04 — `precision_`, `prerequisite_`,
      `stencils_`, `stepping_`, `interface_`, `refinement_`, `horizon_`,
      `checkpoint_` and `type_tests.jl`, the files that reach TreeAMR
      most directly (its names, the partition, the exchange, the
      buffer, `interpolate`, checkpoints, the element types). All pass,
      so the local block indices, the new check in `buffered_flags` and
      the rest of M7 change nothing serially for them.
    - *The audit* (read-only, of each downstream's `src/` and `bin/`
      against the list under "What an application must make global
      itself"). None of the three passes `comm` to a `Forest` yet, so
      today an `mpiexec` launch runs a full serial copy per rank, and
      nothing below is wrong until one does. None indexes
      `forest.leaves` by a block index or sizes a per-block array by
      `nleaves`, with the exceptions named. What would go wrong once it
      is distributed:
      - *TreeWave.* `field_scales` (`src/refinement.jl:112`) is
        `maximum(block_mapreduce(…))`, so the Löhner reference amplitude
        becomes per rank: the mesh would depend on the rank count, and an
        empty rank would throw on `maximum` of nothing and leave the
        others in the next collective. `blast_coverage`
        (`src/blast.jl:281–295`) and `track_pulse`'s tracking measure
        (`src/supergaussian.jl:188–195`) are rank-local diagnostics.
        Its integrators are fixed-step, so no adaptive norm. The viewers
        in `bin/` would plot one rank's blocks.
      - *TreeHydro.* Five functions combine `block_mapreduce` on the host
        — `max_signal_speed` (`src/evolution.jl:477`), `floor_hits`
        (`:505`), `ghost_floor_hits` (`:578–585`), `indicator_scales`
        (`src/refinement.jl:189–190`), `peak_compression`
        (`src/sedov.jl:512`) — and the first is the CFL speed: `evolve!`
        derives each chunk's step count from it (`src/driver.jl:857–860`),
        so the ranks would take different numbers of steps and hang in
        the exchange; `entropywave_errors`, `sedov_static` and the
        `check_cfl` calls do the same. Its time-based checkpoint triggers
        read each rank's own clock (`src/driver.jl:910–917`), so some
        ranks would enter the collective `save_checkpoint` and others
        not. The run state it checkpoints holds the rank-local counts and
        speeds, which `write_plain`'s agreement would refuse on every
        rank. `tracked_share`, `reduce_to_grid`, `mode_amplitude`,
        `max_y_kinetic_energy`, `shock_radius` are rank-local
        diagnostics, and `src/kelvinhelmholtz.jl:544` records
        `nblocks` where the mesh's block count is meant.
      - *TreeGeneralizedHarmonic.* Its reductions are already
        `mesh_mapreduce` (the speed, the bounds, the constraint norms),
        and its integrator is fixed-step. But `indicator_flags` passes
        local flags to the public `buffered_flags`
        (`src/refinement.jl:739`) — now refused, above — and
        `clamp_marks` (`:762`) and `refinement_centroid` (`:816`) index
        `forest.leaves[b]` by a local block, the centroid's sums staying
        per rank too; `nfiring` (`:743`) is a local count. Its
        time-based checkpoint triggers read each rank's clock
        (`src/driver.jl:1287–1293`), its run state holds wall-clock fit
        costs and the rank-local centroids (`:1312`), which the agreement
        would refuse, and its non-finite check (`:741`) reads the local
        state only, so one rank would throw alone. Its horizon finder
        passes every point on every rank, which is correct under M7's
        collective `interpolate` and only redundant.
      None of these is fixed here: they are the downstreams' to fix
      when they distribute.
    - *Julia 1.11.* In the manifest-free copy, Julia 1.11.9 passes the
      whole suite, 110484 tests in 6m46 at one thread, with the MPI jobs
      started early as on 1.13. It resolves MPICH_jll there, the binary
      CI's runners get, against MPIABI_jll on 1.13.1 here (the global
      preference), and IMEXRungeKutta 1.3.0 from `main`, against the 1.1.0
      the 1.13 test manifest holds.
    - *Suite cost.* 110484 tests at one thread in 6m25 and 110536 at
      eight in 6m21 (`Pkg.test`), against 7m55 and 7m51 after step 8.
      The differences from the step-7 follow-up's 110511 and 110563 are the
      29 tests of the round trip and two new `buffered_flags`
      assertions. The thread-independence digests are unchanged. The
      docs build.

### M12 — Rotating symmetry

- **M12 — Rotating symmetry.** *(Specified 2026-10-03. Done 2026-10-03.)*
  A 90° rotating symmetry about the axis where the low faces of two
  dimensions meet: one quadrant of the plane is simulated, as Cactus's
  RotatingSymmetry90 does, for a spinning black hole, which no
  reflection in `x` or `y` maps onto itself; with M10's reflection at
  `z = 0` it gives an octant. `rotating = (d1, d2)` on the forest,
  `rotation` on the field set, the oriented neighbor search with
  conformity at the seam, the virtual-frame transfers with the axis map
  and the signed variable map, and `RotationPair` for the field sets
  whose layout is not symmetric in the plane, all as specified under
  [Domain and boundaries](CODE.md#domain-and-boundaries) and "Rotating seams"
  in [Ghost filling](CODE.md#ghost-filling), with the additions under
  [Conservation](CODE.md#conservation-at-coarse-fine-faces),
  [Point interpolation](CODE.md#point-interpolation) and
  [Checkpoint and restart](CODE.md#checkpoint-and-restart). It comes after
  M7, so unlike M10's mirrored transfers, which M7 distributed as the
  ordinary transfers they are, the rotated ones are distributed by M12
  itself; the MPI path needs the pack to permute and the unpack to
  sign, and nothing else. The change is additive — new
  keywords and one new export — so it is a `0.1.x` release, 0.1.7.
  *Accept:*
  - **refusals**, each with its reason: every case listed for the
    forest and the field set under
    [Domain and boundaries](CODE.md#domain-and-boundaries), a leaf list that
    is not conforming at the seam, an asymmetric set filled alone where
    it has seam ghosts, and a `RotationPair` that is mismatched or
    whose maps compose to anything but the identity;
  - **the neighbor oracle**: the quadrant's oriented neighbors equal the
    images of the neighbors in an independently built *unfolded* forest
    of `2M × 2M` roots over `[−L, L]²`, made by rotating the quadrant's
    leaves with exact `Rational` boxes rather than with the package's
    arithmetic, on random refinements and then after `balance!`; the
    unfolded forest is 2:1 balanced by the oracle, and the seam is
    conforming;
  - **no ghost undefined, and none read before it is defined**:
    `NaN`-prefilled storage with finite owned data has no `NaN` after
    one fill, in `D = 2` and in `D = 3` with `z` periodic, outer or
    reflecting at its low face, for every centering symmetric under the
    swap;
  - **`rotating_vs_quadrupled`**, the definitional test after
    `reflecting_vs_doubled`: data covariant under the quarter turn and
    not polynomial — a scalar built from invariants such as
    `x⁴ − 6x²y² + y⁴`, and a vector field — on a refinement symmetric
    under the rotation that reaches the seam and the axis, where every
    stored point, ghosts included, equals the full domain's within
    `1e-13`, compared with `==` semantics so that `−0` counts; cell and
    vertex centering, `D = 2` and `3`; and the same for a paired
    face-centered `(B_x, B_y)` with `G > 0`, against the full domain
    filled set by set;
  - **write counts**: the schedule partitions the stored points, zero
    writes at owned points and one everywhere else, and every hook
    region's virtual position leaves through an outer face;
  - **the wave equation on a quadrant against the full box**, a scalar
    and a two-component vector wave that exercises the mixing: the same
    number of steps, `linf` equal to the full box's to `rtol = 1e-8`,
    the vertex-centered seam planes at `x = 0` and `y = 0` equal to each
    other — asserted as measured, bitwise if it holds — and the rate 2;
  - **a regrid across the seam**: conformity kept, no block moved by
    more than one level, and the ghosts afterwards equal to the
    oracle's;
  - **conservation through the seam**: advection by the rigid rotation
    `v = (−y, x)`, with face flux sets (`G = 0`, asymmetric, filled
    alone) and the fixup, conserves mass to roundoff while the field
    crosses the seam;
  - **interpolation beyond the seam**: the value and first derivatives
    of the covariant vector field at `r = 1, 2, 3`;
  - **checkpoints**: a round trip, a restart that continues byte for
    byte, and the old fixtures still loading;
  - **MPI in process**: a lockstep rotated pack and unpack in
    `exchange_tests.jl`, bitwise, with a rotated `−0`, and rotating
    cases in `regrid_exchange_tests.jl` and
    `interpolate_exchange_tests.jl`;
  - **the MPI workload**: a rotating case and a pair case in
    `mpi_workload.jl` at `-n 2` and `-n 3`, with seam transfers that
    cross ranks and a rank without blocks;
  - a rotating cycle in `thread_workload.jl`, and a rotating fill in
    `gpu_tests.jl`; `mpi_device_tests.jl` run by hand on Metal.

  Measured, as for every milestone: ordinary fills unchanged in time and
  allocation in `bench/ghosts.jl`, the quadrant against the full domain,
  the suite's cost, and TreeWave and TreeHydro against a scratch copy
  that develops this checkout. In steps, each ending green and
  committed, with what it measured in the commit body:
  - **Step 0 — specification.** *(Done, 2026-10-03.)* The sections
    above, before any code.
  - **Step 1 — forest.** *(Done, 2026-10-03.)* The keyword and its
    refusals, the oriented neighbor search, conformity in `balance!`,
    the checked `leaves` path and `isbalanced`, and the forest digest.
    The field, with `bench/ghosts.jl`'s allocation before and after, and
    the fold into one field with `reflecting` if a ninth field costs
    again (see "The buffer pool" under
    [Distributed meshes](CODE.md#distributed-meshes)); the outcome is recorded
    here either way.
    - *What was built.* `Forest(…; rotating = (d1, d2))` with every
      refusal listed under [Domain and boundaries](CODE.md#domain-and-boundaries);
      `neighbor_anchor` as the one place the seam's arithmetic lives, in
      global level coordinates, returning the orientation; the internal
      `oriented_neighbors(forest, k, δ) -> (r, keys)`, of which
      `neighbor_keys` returns the keys, so `remote_neighbors`,
      `buffer_recruits`, `balance!` and `isbalanced` see the seam
      through it; `real_direction` and `virtual_offset` (the table under
      "Rotating seams"); the seam rule in `balance!`, whose `level ≥ 2`
      shortcut now lets a level-1 leaf of a rotating forest through;
      the conformity refusal in the checked `leaves` path; `rotating` in
      the digest's brick. Until step 3, `GhostSchedule`,
      `InterfaceSchedule`, `interpolate` and `save_checkpoint` refuse a
      rotating forest ("not implemented yet in this step of M12"), so
      that no commit builds unrotated transfers across the seam or
      writes a checkpoint that would load without it; the later steps
      remove each refusal as they teach its reader the orientation.
    - *The field* (measured on the laptop, Julia 1.13.1, `-t 4`,
      defaults `D = 3`, `N = 8`, 4 roots, 10 variables, `p = 4`; bytes
      allocated per call, the median time of two runs). Before:
      uniform `fill_ghosts` 50208 B in 0.52 ms and `ghost_schedule`
      441888 B in 0.24–0.27 ms; two-level 310144 B in 4.03–4.06 ms and
      4976688 B in 2.65 ms. With `rotating::NTuple{2,Int}` the builds
      allocated 442048 and 4981584 B (+160, +4896). With
      `NTuple{2,Int8}`, and with the whole of step 1, every number is
      the baseline's: 50208, 441888, 310144 and 4976688 B, in 0.52,
      0.24–0.30, 4.1–4.3 and 2.63–2.65 ms, inside the run-to-run noise.
      The forest keeps its nine fields; no fold.
    - *Tests.* `test/rotate_tests.jl`, after `reflect_tests.jl`, with
      the unfolded-forest oracle at the end of `ghost_oracles.jl`
      (`unfolded_forest`, `seam_neighbor_mismatches`,
      `seam_conforming`): the refusals; the direction and offset maps
      against turned boxes; the oriented neighbors against the unfolded
      forest's, every leaf and every direction, on random quadrants
      before and after `balance!` in `D = 2` and in `D = 3` with the
      third dimension outer, periodic or reflecting below, over four
      orderings of the pair, every `(r, kind)` met; the unfolded forest
      balanced and the seam conforming; the conformity refusal and its
      acceptance after `balance!`; `complete_marks` moving no leaf by
      more than one level across the seam, with a buffer of 0 and 1;
      and mutual adjacency with `remote_neighbors` over cuts of the
      leaves. Deliberately breaking `virtual_offset` or
      `real_direction` (exchanging `r = 1` and `3`) makes the oracle
      report 58 mismatches on six balanced quadrants. The file holds
      885 tests and runs in about 10 s, nearly all compilation. The
      suite: 111503 tests at one thread in 8m12 (`Pkg.test`), every one
      passing; the docs build, doctests included.
  - **Step 2 — field sets.** *(Done, 2026-10-03, with step 3.)*
    `rotation`, the factor and `rotvars` tables, `RotationPair`.
    - *What was built.* `FieldSet(…; rotation)`, required on a rotating
      forest and checked as recorded under
      [Domain and boundaries](CODE.md#domain-and-boundaries) (step 2's note
      there says where each check lives). Two fields, `rotation` (the
      checked map) and `rotvars` (`nvars × 4` `Int32`, the composed
      variable `σ_r(v)` in column `r + 1`, on the backend), and
      `factors` grown on a rotating forest to `nvars × 3^D·4`, column
      `mirror column + 3^D·r`, each entry the target variable's parity
      factor times the sign of `Q^r`, formed in integers so that a zero
      is `+0`; `r = 0` is M10's table unchanged (`seamtables`). A
      symmetric set composes its own map three times; an asymmetric
      set alone holds zero factors and the identity for `r ≥ 1`, which
      no fill reads, since a plain fill refuses it. `RotationPair(a, b)`,
      exported, with every refusal of the design and one more (two
      symmetric sets: each turns into itself), builds both members'
      tables by alternating the maps, `(Q_a, Q_b, Q_a)` for `a`'s
      targets.
  - **Step 3 — schedule and kernel.** *(Done, 2026-10-03.)* The
    orientation in the group key and the axis map in the group, the
    rotated source accessor, the oriented source search, the single and
    paired fills, and `show`.
    - *What was built.* `GroupKey` gained `orientation::Int8`, last in
      `keyorder` (the old constructors give 0), and `TransferGroup`
      gained `orientation` and `plane` (see the amendment under "Ghost
      filling"). `block_sources!` and `mirror_sources!` search with
      `oriented_neighbors`, take a finer source's offset into the
      virtual frame (`seam_offset`), filter the wall side on it, and
      record `r`; a region with no source is the hook's whatever its
      `r`. `factorcol` is `mirror column + 3^D·r` whenever either is
      nonzero. In the kernel the source accessor
      `RotatedSource = (src, perm, flip, len, vars, col)` maps a
      virtual stored index to the real one (a run-time `perm` is read
      through a chain of selects, `tuplepick`, not a run-time tuple
      index) and reads `vars[v, col]`; `run_group!` builds it only for
      `orientation ≠ 0`, in a branch of its own, so an ordinary launch
      is the code it was. `altsrc` (the array odd orientations read:
      the partner's in a pair, the set's own otherwise) and `rotvars`
      are threaded through `run_group!`, `run_phase!`, `pack_stage!`
      and `run_stage!` beside `factors`. A rotated pack carries the
      orientation and the plane and computes the unscaled sum through
      the accessor; its unpack is the plain width-1 copy with the
      factor column, now holding the sign, so the serial `−0` stays
      bitwise. The step-1 refusal is gone from `GhostSchedule`; the
      plain fill refuses an asymmetric set with ghosts in the plane,
      naming the pair; `fill_ghosts!(pair, (sa, sb); boundary)` runs
      the merged stages (`exchange_pair!`), with one hook or two;
      `show` says "N rotated transfers" when there are any (and counts
      as mirrored only the groups whose mirror state is nonzero).
    - *Deviations*, each amended where the design states it: the
      odd-orientation refusal is decided from the layout, not recorded
      on the schedule; the group holds the orientation and the plane,
      not an `AxisMap`; the `G = 0` asymmetric case has no ghost
      schedule at all; and in 2D a half turn is only ever a copy (a
      region beyond both low faces belongs to the block at the axis,
      whose image there is itself), so rotated restrictions and
      prolongations at `r = 2` occur in 3D only, across the third
      dimension.
    - *The schedule build.* Building the groups inline, with the
      element type a run-time value there, made the new small fields
      a dynamic call's boxed arguments, 448 bytes more per build in
      `bench/ghosts.jl`. The groups are now built behind a function
      barrier (`local_groups`), where the type is static, which also
      removes the run-time type construction the build always did per
      group.
    - *Measured* (laptop, Julia 1.13.1, `-t 4`, `bench/ghosts.jl`
      defaults, bytes per call, two runs). Fills: uniform 50208 B in
      0.52–0.53 ms, two-level 310144 B in 3.8–4.2 ms, against 50208 B
      in 0.51 ms and 310144 B in 3.9 ms before, so unchanged. Schedule
      builds: 406592 B in 0.18–0.25 ms and 4653648 B in 2.08 ms,
      against 441888 B in 0.22 ms and 4976688 B in 2.63 ms: 8 % and
      6 % fewer bytes, and the two-level build 21 % faster, from the
      barrier. The quadrant against the full plane, every stored point:
      worst 2.2e-16 in 2D and 8.9e-16 in 3D (cell and vertex, both
      orders of the pair, the third dimension periodic, reflecting or
      outer), and 4.4e-16 for the pairs (face-centered `(B, F)` with the
      variables in swapped orders, `F` odd about a reflecting wall, and
      a cell-centered pair whose ghost widths alone are swapped) — not
      zero, since the formula's covariance is itself only to roundoff.
      The polynomial data of the `NaN` test is reproduced to 1e-10 or
      better in every case, with no `NaN` left. By hand (not yet in the
      suite; step 7), the lockstep driver of `exchange_tests.jl` over
      2, 3 and 5 simulated ranks reproduces the serial fill bit for
      bit, a symmetric set and a pair, in 2D and 3D, with 40 to 3184
      rotated `−0`s per case.
    - *Tests.* In `test/rotate_tests.jl`, with the oracles at the end of
      `ghost_oracles.jl` (`rotating_data`, `rotating_forest`,
      `undefined_rotated_ghosts`, `rotating_vs_quadrupled`,
      `rotating_pair_vs_quadrupled`, `hook_regions_leave`): the
      refusals of `rotation` and of `RotationPair`; the tables against
      hand-composed turns, and their `r = 0` block against M10's; the
      `NaN` test over every symmetric centering, `p = 2` and 4, `D = 2`
      and 3 with the third dimension periodic, outer and reflecting,
      every `(kind, r)` met; the quadrant and the pair against the full
      plane; the write counts and the hook regions, pairs included; and
      the plain fill's refusal and `show`.The file
      holds 1092 tests (885 after step 1) and runs in about 30 s at one
      thread and at four, nearly all compilation (the `NaN` test's 3D
      cases alone compile 12 s and compute 1.2 s, which is why the
      other orders of the pair there run cell and vertex only). The
      suite: 111710 tests at one thread in 6m51 (`Pkg.test`), every one
      passing; the docs build, doctests included.
  - **Step 4 — regrid, initial data and the interface schedule.**
    *(Done, 2026-10-03.)*
    - *What was built.* `regrid!` takes `pair => (sa, sb)` and fills the
      pair before its two transfers (`regrid_steps`); `check_regrid`
      checks both members as it checks a set (`check_regrid_set!`), adds
      the pair to the agreed layout, and refuses a plain asymmetric set
      with ghosts in the plane, with the pair named.
      `adapt_to_initial_data!` gained a pair form (amended under "Rotating
      seams"); both forms share `adapt_criterion`. The interface
      schedule's step-1 refusal is gone, its search is oriented, and a
      seam restriction is a bug check (amended under
      [Conservation](CODE.md#conservation-at-coarse-fine-faces)).
    - *Measured.* A regrid of the quadrant against a regrid of the full
      plane to the turned image of the quadrant's new leaves, the full
      plane's flags derived from the quadrant's moves in `Rational` boxes
      (`rotating_regrid_vs_quadrupled`): three passes — refine along the
      low face of `d1` near the axis, which conformity carries to the
      low face of `d2`; coarsen everything at level 2; refine along the
      low face of `d2` — take the 2D quadrant through 31, 16 and 34
      leaves and the 3D ones through 134 (or 176), 57 and 246. After
      every pass every leaf moved by at most one level, the quadrant is
      balanced and conforming, the full plane's leaves are exactly the
      four turns of the quadrant's, and every stored point after a fill
      equals the full plane's: worst 1.1e-15 in 2D and 3.6e-15 in 3D,
      cell, vertex and the face-centered pair, with the third dimension
      reflecting below or periodic. Without the pair's fill before the
      transfer the same comparison is off by 1.5. The interface
      schedule and the conservation test are recorded under
      [Conservation](CODE.md#conservation-at-coarse-fine-faces).
    - *Tests*, in `test/rotate_tests.jl` with the regrid oracle at the
      end of `ghost_oracles.jl` (`quadrant_preimage`, `regrid_move`,
      `formula_hook`, `rotating_regrid_vs_quadrupled`): the regrid
      against the full plane in `D = 2` and 3; the refusals and a pair's
      regrid, after which the pair fills again; the initial-data cycle
      on a quadrant, alone (3 passes to level 2) and as a pair, each
      reproducing polynomial data to 1e-10 after a fill; the interface
      schedule against the seamless leaves and its bug check; and the
      rigid-rotation advection, both orders of the pair. The file holds
      1334 tests (1092 after step 3) and runs in 50 s at one thread and
      42 s at four.
  - **Step 5 — interpolation.** *(Done, 2026-10-03.)*
    - *What was built.* The step-1 refusal is gone. `PointGeometry`
      carries the seam's pair; `fold_point` folds periodic and
      reflecting coordinates and then turns a point beyond the seam
      back, returning the orientation (amended under
      [Point interpolation](CODE.md#point-interpolation)); `locate_point`,
      the kernel and M7's host routing share it. The kernel reads
      variable `rotvars[v, r + 1]` and the factor column
      `mirror + 3^D·r`, and contracts with the exchanged multi-indices
      for an odd `r` (`contract_point!`, split out of
      `interpolate_point!` so that both calls see constant indices). A
      set with an asymmetric layout is handed no variable table and
      refuses a point beyond the seam after the launch, with the reason
      and the partner named. `inside`, `stencil_hits` and
      `stencil_position` are unchanged.
    - *Measured.* Polynomial data of degree 3 per dimension, which every
      operator and `Lagrange(4)` reproduce, at the images of 60 points
      under every turn (and the mirror below `z = 0` where `z` reflects),
      values and all first derivatives against the formula and its
      complex-step derivatives: within 1.6e-14 in 2D and 2.4e-14 in 3D,
      cell and vertex, both orders of the pair, the third dimension
      reflecting below or periodic. The value at a turned point is the
      preimage's, turned, **bit for bit**, every variable and derivative.
      Smooth data at 400 random points of the whole plane: within
      1.3e-16 of the full plane's value at the preimage, turned; at the
      point itself within 3.4e-14 for cell centering and 4.2e-3 (2D) and
      5.5e-3 (3D) for vertex centering, values and gradients, about 6e-5 for
      the values alone, which is the full plane's own lack of covariance
      (amended under [Point interpolation](CODE.md#point-interpolation)).
      `vars = [3, 2]` returns the same numbers bit for bit. Float32: the
      turned identity holds bit for bit there too, an `exclude` region
      flags a turned point as its preimage, and the values are within
      7.9e-6 of Float64's. Deliberately contracting an odd turn with the
      unexchanged derivatives puts the polynomial comparison off by 3.4.
    - *The ordinary path* (`bench/interpolate.jl`, laptop, Julia 1.13.1,
      defaults, two runs each against the 0.1.6 release): unchanged
      within the noise — at one thread 783–859 ns per point against
      799–919, at four 217–306 against 226–465, `locate_point` 96 ns
      against 93–97 — with 64 bytes more per batch (128 at four
      threads), the two new kernel arguments. The first version of the
      fold reassigned two tuples that its closures captured, which boxed
      them: 930 bytes allocated per point and 1.6x the time, on every
      forest. On a rotating quadrant (2D and 3D, `N = 16`, vertex,
      value and gradient, 49600 points) a batch allocates what an
      ordinary one does, and points spread over the whole plane, three
      quarters of them turned, cost 94 ns per point in 2D and 559 in 3D
      at one thread, against 75 and 455 for points inside the quadrant.
    - *Tests*, in `test/rotate_tests.jl`: the turned field in `D = 2` and
      3 (`turn_matrix`, `turned_values`, `complex_step`), Float32 with a
      region, and the asymmetric set's refusal, the outside points with
      the seam named, and `locate_point` beyond the seam. The file holds
      1388 tests (1334 after step 4) and runs in 60–84 s at one thread
      and 53–60 s at four. The suite, run once for steps 4 and 5
      together: 112006 tests at one thread in 8m27 (`Pkg.test`; 111710
      in 6m51 after step 3, on a machine less loaded), every one
      passing; the docs build, doctests included.
  - **Step 6 — checkpoint.** *(Done, 2026-10-03.)*
    - *What was built.* The step-1 refusal in `save_checkpoint` is gone,
      and with it the last caller of `refuse_rotating`, which went too.
      A forest with a seam writes `rotating`, `Int64[d1, d2]`, in
      `forest/`, and the file lists `features = ["brick", "rotating"]`;
      one without writes neither and lists `["brick"]`, as before M12,
      so that every file an unrotated run writes is still readable by
      0.1.6. The reader knows both features (`FEATURES`) and builds the
      forest with `rotating` through the checked `leaves` path. A field
      set with a map writes `rotation`, `Int64[nvars]`, in its group,
      read back into the constructor; the map is part of the save's
      agreed layout (`set_layout`). Both attributes are read only where
      present, so the version-1 fixtures and every version-2 file of
      0.1.6 load as before (amended under
      [Checkpoint and restart](CODE.md#checkpoint-and-restart)).
    - *Measured.* A quadrant with a symmetric vertex-centered set and
      a face-centered `RotationPair`, in 2D in `Float32x2` (polynomial
      data, which has no `sin`) and in 3D in `Float64` with a reflecting
      low face, round-trips bit for bit: the forest's pair, every set's
      map, `rotvars` and factor table, the state, and the working arrays
      after the same fill, the pair's through a pair rebuilt from the
      loaded sets. A quadrant wave — a ring about the axis, vertex-
      centered, refined where it is large, with a pair riding through
      every regrid as a pair — runs four chunks through 136, 142, 148,
      160 and 178 leaves, 25 to 29 refined blocks on the seam faces, and
      a restart after the second continues byte for byte, the wave and
      both members of the pair. The file without its feature, without
      its seam, with an unknown feature, with a pair out of range or not
      a pair, with a map that is not a signed permutation, of the wrong
      length, or missing, is refused with the reason each time.
    - *Tests*, in `test/checkpoint_tests.jl`, after the restarts: the
      round trip (`rotating_checkpoint_sets`), the restart
      (`quadrant_start`, `quadrant_chunk`, `quadrant_restore`, through
      `interrupted`), and the refusals; the step-1 refusal's test in
      `rotate_tests.jl` is gone. The file holds 658 tests (593 before)
      and runs in 77 s alone, the four new testsets about 10 s of it.
  - **Step 7 — MPI.** *(Done, 2026-10-03.)* The pack and unpack, and
    the workloads.
    - *What was built.* Nothing in `src/`: the rotated pack and unpack,
      the merged stages of a pair and the agreed refusals came with
      steps 3–5, and step 7 is their tests in process and over MPI. The
      transfer oracles of `exchange_tests.jl` carry the orientation
      (the pack supplies it, the unpack must be a plain copy with
      orientation 0), and `lockstep_pair!` runs a pair's stages merged
      as `fill_ghosts!(pair, …)` merges them. The paired fill's shared
      tags are safe as specified (amended under "Rotating seams" in
      [Ghost filling](CODE.md#ghost-filling)).
    - *In process.* On the quadrants of `rotating_forest` — 10 leaves in
      2D, 22 in 3D with the third dimension reflecting below and 29 with
      it periodic, 21 of 73, 155 of 517 and 217 of 737 transfers
      rotated — over 2, 3 and 5 simulated ranks, for a scalar and
      vector, cell- and vertex-centered, and for both members of a
      face-centered pair: the local and the sent-and-received transfers
      are the serial ones with their orientations and factor columns,
      both ends of every message derive one layout, and every point is
      written once per stage; 6, 8 and 14 rotated transfers cross
      ranks in 2D at 2, 3 and 5 ranks, 45–125 in 3D. In lockstep, and
      through `MailboxCommunicator` at 3 and 5 ranks, every rank's
      array, ghosts included, is the serial fill's bit for bit, the
      single set's and the pair's, with the turned `−0` of a zero
      component kept (38 to 609 per serial fill of a single set, 84 to
      996 of a pair). Over
      `GatherCommunicator`, `regrid!` of a single set and a pair
      together at 3 and 5 ranks, and `adapt_to_initial_data!` from one
      leaf at the axis at 3 ranks (two empty), alone and as a pair, give
      the serial leaves, passes and arrays bit for bit; routed
      `interpolate` over the whole plane (three quarters of the points
      turned back) is the serial one bit for bit at 3 ranks and at more
      ranks than leaves, and a point beyond the seam of a pair's member
      on one rank is refused on all three, that rank with its reason.
    - *Over MPI.* `mpi_workload.jl` gained a rotating quadrant: the
      wave, vertex-centered in 2D (`Q2v`, 27 leaves) and cell-centered
      in 3D over a reflecting low face (`Q3c`, an octant, 57 leaves); a
      single set with a zero `v_2` and a pair with zero second variables
      through a fill, a regrid that refines along the seam and coarsens
      off it, interpolation of 257 points of which 193 are turned back,
      and a checkpoint saved and loaded at this rank count and at the
      others (`rotating_cross`, `QC`), the pair rebuilt and refilled to
      the bytes it had (`Q2`); a single leaf at the axis (`QE2`), which
      leaves every rank but one empty, through the fill, the wave, a
      pair's fill, interpolation from rank 0 alone, a refinement that
      gives every rank blocks, the coarsening back and a checkpoint;
      and two refusals on rank 1 only, a point beyond the seam of a
      pair's member and such a member regridded alone where the others
      regrid the pair, each refused on every rank. Run by hand,
      serially in 53 s and at `-n 3` and `-n 2` (one thread a rank)
      in 66 s and 63 s against about 55 s before, every line agrees but
      the `#` lines, and the sums to `rtol = 1e-12`; 66 rotated
      transfers cross ranks at `-n 3`, 18 at `-n 2`. `mpi_tests.jl`
      asserts the new lines: 206 tests (179 before), in 1m11 alone on
      the concurrent path.
    - *Collective refusals.* Every refusal M12 added was checked for
      one rank refusing alone. The forest's are local and see the same
      arguments and the same replicated leaves on every rank. The field
      set's `rotation` and `RotationPair`'s checks are local, as every
      field-set check is, and depend only on the arguments; the plain
      fill's refusal of an asymmetric set depends on the layout alone
      (amended in step 3), so `fill_ghosts!`'s checks stay rank-local
      on purpose. `regrid!`'s pair checks and its refusal of an
      asymmetric set alone run inside `collective_checks`, and
      `interpolate`'s refusal beyond the seam is decided on the host
      before the routing and agreed with the outside points; the
      workload shows both refused on every rank. The checkpoint's map
      is in the agreed layout (step 6). As with `parity`, a `rotation`
      that differs between ranks is not caught: the schedule is built
      from the forest and the layout, not from the map, so it would
      fill wrong data without a hang; recorded, not changed.
    - *Tests.* `exchange_tests.jl` 9840 tests (9491 before) in 49 s
      alone (38 s), `regrid_exchange_tests.jl` 1951 (1857) in 31 s
      (24 s), `interpolate_exchange_tests.jl` 298 (204) in 16 s (14 s),
      `mpi_tests.jl` 206 (179). The suite, run once for steps 6 and 7:
      112635 tests at one thread in 7m27 (`Pkg.test`; 112006 in 8m27
      after step 5), every one passing; the docs build, doctests
      included.
  - **Step 8 — threads and device, and the wave.** *(Done, 2026-10-03.)*
    - *The wave on a quadrant against the full plane*, the acceptance
      test left from the list above (`test/rotate_tests.jl`,
      `rotating_wave`): the scalar wave and a two-component vector wave,
      state `(v_x, v_y, ∂ₜv_x, ∂ₜv_y)` with `rotation = (−2, 1, −4, 3)`,
      on the quadrant of `quadrant_and_full` (refinement to level 2 along
      both seam faces and at the axis) and on the full plane, order 4,
      RK4 to a quarter period, the exact solution in the hook on the
      outer faces of both. The exact solution is the sum of the four
      turns of one standing mode with no symmetry of its own, summed as
      `(t₀ + t₂) + (t₁ + t₃)`: at the turned point the terms come round
      as `(t₁ + t₃) + (t₂ + t₀)`, the same bits, since floating-point
      addition commutes; so the data is covariant bit for bit, and so is
      its gradient, the vector. The Laplacian adds the two neighbours
      before subtracting `2u₀`, which makes a point and its image compute
      the same bits too. *Measured* (`N = 16`, 143 steps on both):
      cell-centered, `linf` 0.0046035106829825 on the quadrant against
      0.0046035106829834 on the plane, and every stored point, ghosts
      included, within 1.9e-13 of the plane's (4.2e-13 for the vector).
      Vertex-centered, `linf` agrees to the last bit, but the stored
      points differ by 1.9e-4 (vector 3.2e-4), and that is the *full
      plane's* fault: its solution is not covariant itself, by 2.0e-3
      (3.7e-3) under a turn, because which block owns the shared plane
      of a coarse-fine face — the one above it — does not turn with the
      mesh, so a plane the coarse block evolves becomes, a half turn
      away, one the fine block evolves. The quadrant is covariant by
      construction; the test asserts it within the plane's own defect.
      The vertex-centered quadrant owns both seam planes, `x = 0` and
      `y = 0`, which are the same 63 points under the turn: after the
      evolution they are equal **bit for bit**, scalar and vector. Rates
      on the quadrant over `N = 8, 16, 32`, `l2` and `linf`: cell 2.04
      and 2.01 (scalar), 2.02 and 2.00 (vector); vertex 2.04 and 2.00,
      2.02 and 2.00. The two testsets take about 12 s, mostly
      compilation.
    - *Threads.* `thread_workload.jl` gained two rotating quadrants,
      cell-centered with `G = 2` (`D2ocq`) and vertex-centered with
      `G = 1` (`D2ovq`): a ring about the axis, so that the refinement
      reaches both seam faces and the axis, through the adapt, evolve,
      regrid, evolve cycle, interpolation over the whole plane, and a
      face-centered `RotationPair` filled as a pair (39 rotated
      transfers in the final schedule). Its output is the same at one
      thread and at four; the script takes 26.7 and 25.4 s.
    - *Device.* `gpu_tests.jl` gained a rotating fill — the `NaN` test
      of `undefined_rotated_ghosts`, cell and vertex, in 2D and in 3D
      over a reflecting low face, and a face-centered pair against the
      CPU — and interpolation through the seam against the host. The
      oracles' formulas had to be made device-clean: their constants are
      now converted to the argument's real type (`literal`), the data
      closure carries the third dimension's kind as a `Val` (a `Symbol`
      is not plain data), and `outofplane` no longer builds a `Set`;
      `Float64` results are unchanged bit for bit. On this laptop's
      Metal GPU (`Float32`, a scratch environment that develops this
      checkout and adds Metal): the rotating testsets pass, 23 tests on
      Metal and 46 on the CPU in 43 s, compilation included, and the
      `NaN` fill reproduces the polynomial data to 3.6e-7 in 2D and
      8.3e-7 in 3D on Metal, as on the CPU in `Float32`.
      `mpi_device_workload.jl` gained a rotating quadrant (`Q2v`, 15
      leaves, the vertex-centered wave) and a face-centered pair
      (`QP2`); `mpi_device_tests.jl` passes on Metal, 60 tests in 1m59
      (serial 35 s, `-n 2` 41 s, `-n 3` 42 s; 1m45 before M12), with 8
      staged messages for `Q2v` and 16 for `QP2` at `-n 3`.
  - **Step 9 — measurements**, as listed above. *(Done, 2026-10-03.)*
    All on the laptop (Apple silicon, 12 cores), Julia 1.13.1 unless
    stated.
    - *The ordinary fill.* `bench/ghosts.jl` at `-t 4` with its defaults
      (`D = 3`, `N = 8`, 4 roots, 10 variables, `p = 4`), this checkout
      against a pristine copy of 0.1.6 with the same manifest, run
      alternately twice each; seconds and bytes per call:

      | | 0.1.6 | M12 |
      |---|---|---|
      | uniform `fill_ghosts` | 0.641, 0.635 ms; 50208 B | 0.516, 0.517 ms; 50208 B |
      | uniform `ghost_schedule` | 0.217, 0.219 ms; 441888 B | 0.143, 0.141 ms; 406592 B |
      | two-level `fill_ghosts` | 3.81, 3.77 ms; 310144 B | 3.53, 3.53 ms; 310144 B |
      | two-level `ghost_schedule` | 2.52, 2.50 ms; 4976688 B | 1.86, 1.87 ms; 4653648 B |

      A fill allocates exactly what it did, and it is not slower: 19 %
      and 7 % faster in these runs, a difference not investigated (step
      3 measured the fills equal, 0.52 against 0.51 ms, on a quieter
      machine). The schedule builds are 8 % and 6 % smaller and 35 % and
      26 % faster, from step 3's function barrier (`local_groups`).
    - *The quadrant against the full plane*: step 8, and the single
      fills of steps 3–5.
    - *The suite.* 112715 tests at one thread in 7m52 (`Pkg.test`, 8m01
      wall clock) and 112767 at eight threads in 7m33 (7m43), every one
      passing, against 110606 in 6m32 and 110658 in 6m28 after M7. On
      Julia 1.11.9, on a copy without manifests (the procedure in
      `CLAUDE.md`): 112715 tests in 8m09, every one passing. M12 added
      about 2100 tests; `rotate_tests.jl` holds 1421 of them and takes
      about 56 s alone at four threads, nearly all compilation.
    - *Downstream*, each in a scratch copy whose `Project.toml` and
      `test/Project.toml` name this checkout under `[sources]`, the real
      repositories untouched: TreeWave 310 tests in 1m17 (1m22 against
      M7), TreeHydro 12447 tests in 5m08 (5m00), both at one thread and
      all passing; neither uses the seam. TreeGeneralizedHarmonic's
      suite takes about 19 minutes, so only its `prerequisite_tests.jl`,
      which names the unexported TreeAMR functions it relies on, was run:
      52 tests, passing.
  - **Step 10 — documentation and status.** *(Done, 2026-10-03.)* The
    `rotating` keyword is documented in `Forest`'s docstring and
    `rotation` in `FieldSet`'s, both already on their pages, and
    `RotationPair` is on the storage page; the guide gained a section,
    "Rotating symmetry", with a doctest of a quadrant, a velocity's map
    and a face-centered pair; README and the guide's status say M12;
    `CLAUDE.md` gained the architecture bullet "Rotating seams are
    oriented transfers", the oracles and the workloads in its tests
    paragraph, and the new timings. The docs build, doctests included.
    The repository keeps no changelog; the release commit says what
    changed. M12 is additive — the keywords `rotating` and
    `rotation`, the export `RotationPair`, the pair forms of
    `fill_ghosts!`, `regrid!` and `adapt_to_initial_data!`, and a
    checkpoint feature that only a rotating forest writes — so it is a
    `0.1.x` release: *released as 0.1.7 (2026-10-04)*, together with the
    interpolator's second derivatives from `main`.

## Downstream checks

**TreeWave.** The `[sources]` pins to `main` went when 0.1.1 was
released, which is also what dropped its Julia floor to 1.10 (raised back
to 1.11 on 2026-09-25 with all the Tree* packages). TreeWave was ported to
M8 and green: 310 tests passing against the registered 0.1.1 (measured
2026-09-22). Checkpointing reached it with 0.1.4; against the M9a
checkout its suite was unchanged, 310 tests in 1m12 (2026-09-29). M7
reached it with 0.1.5 and changed nothing serially: 310 tests in 1m22
against the M7 checkout (2026-10-02); M12, released in 0.1.7, changed
nothing either, 310 tests in 1m17 against the M12 checkout (2026-10-03).

**TreeHydro.** It is **implemented, not a sketch**: sixteen files in
`src/`, fourteen test files, two viewers in `bin/`, its own CI and
Codecov, and milestones H0–H5 marked done in its `CODE.md` — the scheme,
coarse-fine faces, regridding, Sedov with the atmosphere reset,
Kelvin–Helmholtz with its viewers — with measured results recorded
through step 11; H6 (precision, threads, device) is next. Its two
"Upstream prerequisites" (`map_blocks!(…; stored = true)` and
`AllVariables`) are satisfied and released, so that section of its
`CODE.md` is history. Its `[sources]` entry for TreeAMR went when
TreeAMR 0.1.1 reached the General registry. Its suite is much longer
than TreeWave's — 11893 tests in 4m16 at one thread against the M9a
checkout (2026-09-29), against TreeWave's 310 in 1m12. Checkpointing
(M9a) is what its long runs were waiting for, and it reached TreeHydro
with 0.1.4, through `using HDF5`. The plan (2026-09-29) saves at the
start of a chunk, *after* the regrid and the atmosphere reset — its
observer fires before the regrid, so saving there would mean replaying
both — with the chunk index, the case recipe as Rationals, the `evolve!`
keywords and the run histories as plain data; its `src/checkpoint.jl`
(`run_state`, `rotate_checkpoints!`) has since implemented it. M7
changed nothing for it serially: 12447 tests in 5m00 against the M7
checkout, at one thread (2026-10-02); nor did M12 (0.1.7), 12447 tests in
5m08 against the M12 checkout (2026-10-03). Its own MPI port
(2026-10-02) found the one TreeAMR bug so far on a rank without blocks —
the `AllVariables` boundary hook's length check, of which it is the only
caller — so it needs TreeAMR 0.1.6, which has the fix.

**TreeGeneralizedHarmonic.** It pinned TreeAMR to GitHub `main` through
`[sources]` until 2026-09-26, when it retired its stopgap interpolator for
M11's `interpolate`; since then it takes TreeAMR from General,
`TreeAMR = "0.1.4"` under `[compat]` since its checkpointing. That
includes M9a: checkpointing reached it with 0.1.4, through `using HDF5`,
together with TreeAMR's new `__init__` and the five new exports (none of
which clashes with a name of its own). Its production runs, estimated
at 38–149 h, are one of the reasons M9a went before M7. The stopgap's
`threaded_foreach` left with the stopgap. On 2026-09-25 it measured the
ownership policy from the outside, on Symmetry, and found the
integrator's own passes to be what is left (the last item of "Open
questions" in [CODE.md](CODE.md#open-questions)); Erik decided the same
day not to optimise the OrdinaryDiffEq path further, Polyester
included; what is left open there is where limiters go (Shu–Osher
against Butcher form). M7 changed nothing for it serially: its full
suite takes about 19 minutes, so M7 step 9 ran a subset, 1760 tests in
7m04 at one thread, all passing (2026-10-02); against the M12 checkout
(0.1.7) its `prerequisite_tests.jl` alone passes, 52 tests (2026-10-03).
Its production runs are the reason M7 was wanted. Of the step-9 audit's
findings for it, `indicator_flags` hands its local flags to the public
`buffered_flags`, which TreeAMR now refuses — under MPI the dilation has
to come from `regrid!`'s `buffer`, which it avoids on purpose so that
its level ceiling applies after the dilation.

## Development environment and CI

**Julia floor.** The floor was 1.10 (the LTS) through 0.1.2 and was
raised to 1.11 for 0.1.3 (2026-09-25) across all the Tree* packages, so
that unregistered dependencies can be located with `[sources]` entries,
a 1.11 key. The procedure for checking a test on 1.11 in a copy without
manifests was worked out, and measured at 97744 tests in ~2m55 after M10,
on 1.10.

**Coverage.** `julia-actions/julia-runtest` defaults `coverage` to
`true`, so until the Codecov upload was added *every* CI cell was paying
for coverage and discarding it; CI.yml then tied coverage to
`threads == 1`.

**Suite timings after M12** (as CLAUDE.md recorded them): the
thread-independence test's two subprocesses running
`test/thread_workload.jl` take ~26 s each since M12's rotating cycles,
M10's `reflect_tests.jl` ~30 s, and, each run alone with its own
compilation, `exchange_tests.jl` ~49 s, M12's `rotate_tests.jl` ~60 s and
M9a's `checkpoint_tests.jl` ~77 s. The MPI test's two `mpiexec` jobs take
~65 s each (~55 s before M12), so `mpi_tests.jl` itself costs ~15 s where
they run beside the suite; on the one-after-the-other path the suite
took about 8 min after M7 (7m52 with `TREEAMR_TEST_MPI_CONCURRENT=0`,
CI's path; not re-measured after M12). The per-file times before and
after M7's trim are in M7 step 9, M12's in its steps. Pinned with first
touch, the thread-scaling benchmark measured 0.92 against 1.00 ns/cell
interleaved (2026-09-23).

## Releases

TreeAMR was registered in General on 2026-09-21; 0.1.1 was the first
release the downstreams took from the registry. 0.1.2 is the release the
H200 run of 2026-09-23 measured. 0.1.3 (2026-09-25) raised the Julia
floor to 1.11. 0.1.4 carried checkpointing (M9a), 0.1.5 the distribution
over MPI (M7), and 0.1.6 the fix for the `AllVariables` boundary hook on
a rank without blocks. 0.1.7 (2026-10-04) carried the rotating symmetry
(M12), together with the interpolator's second derivatives from `main`,
and 0.1.8 (2026-10-06) the device copy kernels; 0.1.8 is the current
release. The repository keeps no changelog; the release commit says what
changed.
