# M8 implementation plan

For the sessions that implement M8. The **design** lives in CODE.md —
read, in this order, its sections *Centerings*, *Ghost filling*,
*Operators* (the "Operators per centering" table), *Conservation at
coarse-fine faces*, *Regridding* (step 5), *Application interface*, and
the M8 entry under *Milestones*. CODE.md is authoritative; this file is
only the work breakdown: what each step changes, what it must not change,
and what it must measure and record. Delete it when M8 is marked *(Done.)*.

## Ground rules, every step

- Read `CLAUDE.md` first. It has the commands, the conventions, and the
  invariants that are easy to violate (bit-identity across thread counts,
  KernelAbstractions for all per-cell work, no floating-point literals in
  per-cell arithmetic, Julia 1.10 compatibility, Documenter strictness).
- Work on a branch. **Do not push to `main` until Step 7 is done**:
  TreeWave pins TreeAMR's `main` and Step 1 breaks its API.
- Before and after each step: the full suite, including with
  `--threads=8`; the docs build (the only place doctests run); the suite
  on Julia 1.10 via `julia +1.10`. Metal is available on this machine —
  `test/gpu_tests.jl` in `Float32` is worth running for every step that
  touches a kernel.
- Spec-first: if the implementation shows CODE.md wrong or incomplete,
  amend CODE.md and say so in it ("amended in M8: …"), rather than
  diverging silently. Record measured numbers in CODE.md where the text
  says "the table goes here when it exists", and in the commit body.
- Testset names are claims; each opens with a comment naming the failure
  mode it guards. Property tests compare against the independent oracles
  in `test/oracles.jl` / `test/ghost_oracles.jl`, never against the
  package's own neighbor search or geometry.
- A documented function must appear in a `@docs` block in
  `docs/src/index.md`, and every `` [`name`](@ref) `` must resolve.
- One step at a time; the next step starts from a green suite.

## Sharp edges to know before starting

- **`0 × NaN = NaN`.** Never pad a narrow stencil with zero weights to a
  common width; widths are per dimension (Step 2). A `G = 0` face field
  has no slot beyond its high face, so a padded read is out of bounds,
  not merely wasteful.
- **`target_range` is the single source of truth** for the owned, closed
  and exchange ranges. The ghost schedule, the regrid transfer and the
  interface schedule all take their ranges from it; do not re-derive a
  range anywhere else.
- **Stored index → position** is `origin + (i − G_d − 1/2)·h` in a
  cell-centered dimension and `origin + (i − G_d − 1)·h` in a vertex-like
  one, with `G_d` the *field set's* ghost width. Every kernel that forms
  a position (`coordinates_kernel!`, `boundary_kernel!`, `firing_kernel!`)
  must use the same expression, in the same operation order, so that they
  agree with `coordinates` bit for bit (the M5 digests depend on it).
- **The regrid transfer fills the owned range only.** A fresh block's
  shared plane and ghosts are filled by the next `fill_ghosts!`. A test
  that checks a staggered field after `regrid!` must fill ghosts first.
- **Injection is a unit vector that the rational builder produces for
  free**: `interpolation_weights(lo, p, x, …)` at an integer `x` returns
  exact zeros and one exact one. Keep the uniform width `p` within one
  stencil (a `Stencil1D` has one width); the zero-weight reads stay inside
  filled memory because of the `G ≥ p/2 − 1` invariant — that invariant
  is what makes them safe, so check it, do not trust it.
- **Kernel arguments must be `isbits`**: `Val` of tuples for `G` and the
  centering, `oftype` instead of captured types, no closures over
  reassigned locals (found on Metal in M6).
- **`thread_workload.jl` is self-contained and must stay fast**: its own
  Runge–Kutta, no ODE package, everything it prints deterministic.
- **Julia 1.10**: no `public`, no `[sources]`; a seeded RNG stream can
  differ across versions, so assertions must follow deterministically
  from the setup.

## Step 1 — `G` moves to the field set (M8a)

Cell-centered only; no behaviour changes for cell-centered data. This is
the API move, done first so that the arithmetic retrofit in Step 2 is
about centering alone.

Changes:

- `Forest(roots; N, periodic, extents)` — no `G`. Passing `G` throws an
  `ArgumentError` that says where it went (`FieldSet(forest, nvars; G = …)`);
  that message is what TreeWave's migration will hit first.
- `FieldSet(forest, nvars; G, backend)` with `G::Union{Integer,NTuple{D,Integer}}`,
  stored as `NTuple{D,Int}`; **required**, no default. The invariant
  `N ≥ 2G_d` moves here (per dimension; Step 2 adds `+ 2c_d`).
- All index arithmetic per dimension from the start:
  `ntuple(d -> I[d] + G[d], Val(D))` in every kernel, `Val(fs.G)` at every
  launch site, `G[d]` in every stencil builder and in `target_range`.
- `GhostSchedule(fs, operators)` records `(G, T, backend)` (Step 2 adds
  the centering); `fill_ghosts!` checks them with a message.
  `check_operators(fs, ops)` takes the field set. Keep
  `GhostSchedule(forest, ops; G, T, backend)` only if it costs nothing;
  the field-set form is the documented one.
- `coordinates(fs, b, idx)` replaces `cell_center(forest, key, idx)`
  (stored indices; needs the field set's `G`). Remove `cell_center`.
  `block_origin`, `block_extent`, `block_origins`, `block_spacings` stay
  on the forest.
- `regrid!(forest, pairs; flags, buffer, boundary, transfer)` with
  `pairs` one or more `fs => schedule`; `fs => nothing` resizes without
  transfer. `adapt_to_initial_data!(fs, ops; …)` keeps its signature and
  builds the schedule from `fs`.
- Tests, docs, README, doctests updated mechanically. `test/wave.jl`,
  `thread_workload.jl`, `bench/*.jl` too.

Accept: the whole suite green with the same measured rates as before
(the wave tables in `wave_tests.jl` must not move); the thread digests
still agree; a test that `FieldSet` rejects `N < 2G_d` with a message and
that `fill_ghosts!` rejects a schedule built for another `G`.

## Step 2 — Centering (M8a)

Changes:

- `FieldSet(forest, nvars; G, centering = cellcentered(D), backend)` with
  `centering::NTuple{D,Symbol}` of `:cell` / `:vertex`, validated;
  constructors `cellcentered(D)`, `vertexcentered(D)`,
  `facecentered(D, d)`, `edgecentered(D, d)`. Stored extent
  `N + 2G_d + c_d`; invariant `N ≥ 2G_d + 2c_d`.
- `coordinates` per centering; `interiorview` unchanged (owned);
  `closedview(fs, b[, v])`; `map_blocks!(kernel!, fs, args…; closed = false)`
  launches over `N + c_d·closed` per dimension; `fill_by_coordinates!`
  fills the owned range at the set's own points.
- `target_range(N, G_d, c_d, δ_d, o_d, halved)`: high exchange region
  `G+N+1 … N+2G+c_d`.
- Stencil builders take `c_d`. Vertex-like dimension: copy is the same
  shift by `N`; restriction is width-1 weight-1 injection, source
  `G + 1 + 2·(q − o_d·N/2)` in the fine block's frame with `q` the target's
  position in the adjacent node's frame (the same `q` and `f0` arithmetic
  `restrict_stencil` already has, minus the `+ 1/2`); prolongation targets
  coarse coordinate `φ/2` (not `φ/2 − 1/4`), window
  `lo = clamp(fld(φ, 2) − p ÷ 2 + 1, 1, N + 2G + c − p + 1)`, weights from
  `interpolation_weights` as today. Conservative builders unchanged in
  cell-centered dimensions; `check_operators` **refuses** the conservative
  family on any field set with a vertex-like dimension, with the reason
  from CODE.md. Point-value constraint in a vertex-like dimension:
  `G_d ≥ p/2 − 1`.
- `transfer_kernel!` takes `::Val{Ps}` with `Ps::NTuple{D,Int}` and loops
  `CartesianIndices(Ps)`; `run_group!` builds `Ps` from the stencils.
- Boundary regions include the shared plane on the domain's high side
  (falls out of `target_range`); `boundary_kernel!` and the region form
  use the per-centering position; `firing_kernel!` likewise.
- Regrid transfer: `transfer_groups` passes `c_d`; targets are the owned
  range; coarsening in a vertex-like dimension is halved injection.
- Exports and docs for everything new.

Tests (new files or testsets; oracles in `ghost_oracles.jl` generalised
so that `makepoly` is evaluated at `coordinates` and `cell_average`
averages along cell-like dimensions only):

- *The exchange is exact to degree `p − 1` for every centering*: the M2
  test over all `2^D` centerings in `D = 1, 2, 3`, point-value family,
  face/edge/corner and three-level configurations, including the shared
  plane and the domain's upper boundary plane under a hook.
- *Vertex restriction is injection*: random data, bit-for-bit equality of
  the coarse exchange region with the coincident fine points.
- *`G ≥ p/2 − 1` is the bound*: `check_operators` rejects one less with a
  message; one more passes the exactness test.
- *The conservative family is refused along a vertex-like dimension*,
  `@test_throws` on the reason.
- *The regrid transfer is exact for every centering*: polynomial data
  through refine and coarsen, ghosts filled after the transfer.
- Type tests (`Float32`, `Float32x2`) and the Metal suite on a vertex
  field set; the thread digests.

Record in CODE.md: the vertex-like rows of the invariants and the
operator table change from "M8 design" to "measured", with anything the
implementation had to correct said explicitly.

## Step 3 — The vertex-centered wave equation (M8a)

- `test/wave.jl`: `wave_forest`, `wave_errors`, `track_pulse`,
  `uniform_pulse`, `WaveProblem` take `centering`; the kernel takes
  `Val(C)` if it needs it (it should not — it reads owned cells and their
  neighbors, and the Laplacian does not care where they sit).
- `test/wave_tests.jl` becomes the **vertex-centered** study: the M3
  table with the changed prediction — rates 1 and 2 for prolongation
  orders 2 and 4 at any restriction order, `G = 1` at order 4 — and the
  pulse tests. `test/wave_cell_tests.jl` is the current `wave_tests.jl`
  verbatim with `centering = cellcentered(D)`, asserting the M3 numbers.
  `runtests.jl` includes both.
- `thread_workload.jl` evolves the vertex-centered wave as well (keep it
  a few seconds); `gpu_tests.jl` reproduces the vertex convergence on the
  device; `type_tests.jl` gets a vertex case.

Record in CODE.md: the vertex table under Operators next to the M3
table, the M3 milestone note that the wave test is vertex-centered from
M8 on with the cell-centered study kept as `wave_cell`, and M8a marked
done in the M8 entry.

## Step 4 — Interface restriction (M8b)

- `InterfaceSchedule(fs)` (backend from the field set) and
  `restrict_interfaces!(fs, isched)`, per CODE.md *Conservation at
  coarse-fine faces*: face directions `±e_d` for vertex-like `d` only;
  one phase per direction; target the owned low plane `G+1` or the shared
  high plane `G+N+1`, tangentially the owned range (cell-like) or the
  closed range (vertex-like) split half-open among the finer neighbors
  with the top point to the top neighbor; source the finer neighbor's
  opposite boundary plane; stencils width 1 normally, the 2-cell average
  tangentially in cell-like dimensions, halved injection in vertex-like
  ones. Built from the `:restrict` face cases of the neighbor walk.
  Records generation and layout; refuses stale or mismatched. Refuses a
  cell-centered field set with a message. `Base.show`. Exported and
  documented.
- Runs through `TransferGroup` / `run_phase!`, so the device path is the
  generic per-batch one and needs nothing new.

Tests:

- *Interface restriction equals the hand-computed average*: an oracle
  using `neighbor_keys` sums the finer neighbors' boundary planes
  directly; face fields in `D = 1, 2, 3`, an edge field in 3D, a vertex
  field in 2D (the EMF `E_z`); compared at a few `eps`.
- *Every target is written exactly once per phase*: sentinel fill, count
  the writes (the ghost tests have this oracle shape already).
- *A cell-centered set is refused.*
- Thread digests (the interface restriction on a face field joins the
  workload in Step 5; a unit-level determinism check here is enough).

## Step 5 — Burgers (M8b)

`test/burgers.jl` and `test/burgers_tests.jl`, mirroring `wave.jl` /
`wave_tests.jl`. Everything generic in `T` and the backend.

- `BurgersProblem`: state set (`G = 2`), `D` face sets (`G = 0`,
  `facecentered(D, d)`), ghost schedule, `D` interface schedules,
  spacings on the backend, `Val`s for `D`, `G_u`, `G_f`.
- Flux kernel for direction `d`, launched with `closed = true` over face
  set `d`: face `i ∈ 1 … N+1` lies between cells `i−1` and `i`, stored at
  `i − 1 + G_u` and `i + G_u`; the face is stored at `i + G_f`. Linear
  reconstruction from cells `i−2 … i+1` (`limiter = :none` for smooth
  studies, `:minmod` for shocks), Rusanov flux
  `½(f_L + f_R) − ½·max(|u_L|, |u_R|)·(u_R − u_L)`, `f = u²/2`.
- Divergence kernel over the state's owned cells:
  `du = −Σ_d (F_d[i+1] − F_d[i]) / h`, written in state layout.
- `rhs!` = the three steps from CODE.md's *Application interface*.
- Exact solution `u = u₀(s − D·u·t)`, `s = Σ x_d`, `u₀ = ū + a sin(2πs/L)`,
  Newton per cell, compared as cell averages through the Gauss–Legendre
  oracle; `t_b = L/(2πaD)`.
- Refinement criterion `|u_{i+1} − u_{i−1}| > threshold` through
  `firing_boxes`, `(Keep, box)` at the target level.
- `OrdinaryDiffEqSSPRK` in `test/Project.toml`; `SSPRK33`.

Tests, in the M8b acceptance order:

- *Total mass is conserved to roundoff across coarse-fine faces*: a shock
  crossing a refined region that follows it, regrids between chunks;
  `|Δ total_mass| ≤ c·eps(T)·Σ h^D|u|·nsteps`. **Negative control**: the
  same run with `restrict_interfaces!` skipped (a keyword on the problem)
  must drift by orders of magnitude more; record both numbers.
- *The interface-order rule for the conservative family*: smooth run to
  `t_b/2` on the M3 two-level static mesh, unlimited reconstruction,
  conservative prolongation orders 1, 3, 5 (restriction 2), `D = 1, 2`;
  predicted rates 1, 2, 2. Tabulate.
- *A tracked shock matches the uniformly fine reference*: adaptive versus
  the finest uniform mesh, `L¹` difference against the coarse uniform
  error, at fewer cells — the M4 pulse test's shape.
- 3D smoke test; `Float32` conservation (relative tolerance scaled by
  `eps(T)`); the Metal suite.
- `thread_workload.jl` gains an adapt/evolve/regrid Burgers cycle with a
  hand-written SSPRK3 (no ODE package there).

Record in CODE.md: the conservation numbers (with and without the fixup),
the rate table under Operators (replacing "the table goes here when it
exists"), the shock-tracking ratio in the M8 entry, and M8 marked
*(Done.)*. Update the status paragraphs in `README.md` and
`docs/src/index.md`. Delete this file.

## Step 6 — Review pass

A fresh session reads CODE.md end to end against the code: every "M8
design" marker is now "decided" or "measured" or amended; the invariants
in the text are the ones `FieldSet` and `check_operators` enforce; the
docs build is clean; `bench/threads.jl` and `bench/gpu.jl` run (they
construct field sets and need the new keywords).

## Step 7 — TreeWave

`~/src/jl/TreeWave` pins TreeAMR's `main`. Before pushing:

| TreeWave today | after M8 |
|---|---|
| `Forest(roots; N, G, …)` | `Forest(roots; N, …)` |
| `FieldSet(forest, 2)` | `FieldSet(forest, 2; G = 2)` |
| `GhostSchedule(forest, ops; T, backend)` | `GhostSchedule(fs, ops)` |
| `cell_center(forest, key, idx)` | `coordinates(fs, b, idx)` |
| `regrid!(forest, fs, schedule; …)` | `regrid!(forest, fs => schedule; …)` |
| `adapt_to_initial_data!(fs, ops; …)` | unchanged |

Temporarily `Pkg.develop` this checkout into TreeWave's environment (root
and `bin/`), run its tests, update its `CLAUDE.md` / `CODE.md` where they
describe the API, revert the temporary `Project.toml` / `Manifest.toml`
edits, then push TreeAMR and TreeWave together. TreeWave stays
cell-centered.
