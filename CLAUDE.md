# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TreeAMR.jl is a Julia package: a tree-based (octree-style) block-structured
AMR mesh — storage, ghost exchange, inter-grid operators, regridding — with
**no physics**. `CODE.md` is the authoritative design document and milestone
roadmap. Read it before changing anything non-trivial: it states *why* things
are the way they are, and it is kept in sync with the code (see "Spec-first
workflow"). `README.md` and `docs/src/index.md` carry the public status
summary.

Current state: milestones M0–M6 and M8 are done (tree core, ghost
exchange, ODE coupling, regridding, multi-threading, GPU; then every
centering, per-field-set ghost widths, and conservation at coarse-fine
faces). Everything is `D`-generic and floating-point-type generic. Next
is MPI (M7), deliberately after M8 so the distributed exchange is built
once over a layout-generic schedule; then I/O (M9).

`TODO.md` is Erik's personal to-do list. **Do not modify it.**

## Commands

Full test suite (about 3 min — the thread-independence test spends ~50 s
of that running `test/thread_workload.jl` in two subprocesses).

**The suite is compilation-bound, not kernel-bound**, so do not try to
shorten it by making the kernels faster. Measured: annotating the test
applications' kernels `@inbounds` made `wave_rhs_kernel!` 6.8x faster
and left the suite at 3m11 against 3m14, inside the 7 % run-to-run
noise; the same treatment of `test/thread_workload.jl` moved its 18.8 s
by 2 %. CI gains nothing from such a change in any case, since
`check_bounds: yes` overrides `@inbounds` package-wide:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The tests inherit the caller's thread count; `Pkg.test` does not
propagate `-t`, so pass it explicitly to exercise the threaded paths in
the suite itself (the thread-independence test spawns its own
subprocesses either way):

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=8"])'
```

A single test file. The `test/*_tests.jl` files are `include`d by
`runtests.jl` and rely on the helper files, so include those first.
`test/Project.toml` devs TreeAMR from `..`:

```bash
julia --project=test -e 'using Test, Random, TreeAMR; include("test/oracles.jl"); include("test/ghost_oracles.jl"); include("test/wave.jl"); include("test/regrid_tests.jl")'
```

On a fresh clone the test environment has no Manifest; run this once first:

```bash
julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Coverage, the way CI's single-threaded cells measure it. Note that
`julia-actions/julia-runtest` defaults `coverage` to `true`, so until
the Codecov upload was added *every* cell was paying for coverage and
discarding it; CI.yml now ties coverage to `threads == 1`, which keeps
it on the four cheap cells (both operating systems, both Julia
versions, merged by Codecov) and drops it from the expensive threaded
ones. Instrumentation cost **2.61x** locally
(measured: 3m23.6s → 8m51.8s, 88819 tests, 4 threads both times) —
though that is not the CI factor, because the action also defaults
`check_bounds` to `yes` and a plain `Pkg.test` no longer does.
`Base.julia_cmd()` forwards `--code-coverage` to the subprocesses the
thread-independence test spawns, so those are instrumented too — three
PIDs write `.cov` files and `julia-processcoverage` merges them per
source file, which is why a single-threaded cell still sees the
threaded code paths. The `.cov` files scattered through `src/` are
gitignored; delete them before the next run, because counts from
separate runs accumulate:

```bash
julia --project=. -e 'using Pkg; Pkg.test(; coverage=true, julia_args=["--threads=4"])'
```

```bash
find src -name '*.cov' -delete
```

Docs build. This is also the **only place doctests run** — `Pkg.test` does
not run the `jldoctest` blocks in docstrings and `docs/src/index.md`:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

```bash
julia --project=docs docs/make.jl
```

Both `Pkg.develop` lines above have a side effect on a current Julia: they
write a `[sources]` entry into that environment's `Project.toml`. It is a
1.11+ feature, so it is exactly what the compat bound below forbids, and it
turns a local convenience into a build everyone else's 1.10 cannot parse.
The `Manifest.toml` they also write is the part you want, and is
gitignored. Check `git status` after running either and revert
`test/Project.toml` or `docs/Project.toml` if it moved.

Documenter is strict: every docstring in the module must appear in a `@docs`
block in `docs/src/index.md`, and every `` [`name`](@ref) `` must resolve, or
the build errors out. **Adding a documented function means adding it to
`docs/src/index.md`.**

CI (`.github/workflows/CI.yml`) tests on Julia **1.10** and latest, on Linux
and macOS. `Project.toml` says `julia = "1.10"`, so no 1.11+ features (no
`public`, no `[sources]`). Your local Julia is newer. A seeded RNG stream can
differ across Julia versions, so a test whose *assertions* depend on a
particular random draw can pass locally and fail on 1.10: use a seeded RNG
for the inputs, but make what the test asserts follow deterministically from
the setup. `juliaup` has 1.10 installed, but checking a suspect test on it
is not simply `Pkg.test()` in this checkout: 1.10 cannot read a
`Manifest.toml` a newer Julia resolved, and even against fresh manifests
its `Pkg.test()` dies with "can not merge projects" whenever
`test/Manifest.toml` exists — which the setup command above creates.
Copy the tree without any manifest and run the test file directly
(measured: 96474 tests, ~2m15):

```bash
rm -rf /tmp/amr110 && mkdir /tmp/amr110 && tar -cf - --exclude=Manifest.toml --exclude=.git --exclude=.claude . | tar -xf - -C /tmp/amr110
```

```bash
cd /tmp/amr110 && julia +1.10 --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()' && julia +1.10 --project=test test/runtests.jl
```

Thread scaling (`bench/threads.jl`, driven by `bench/scan.sh`, which
takes a list of thread counts and prints a speedup table). Sizes come
from `TREEAMR_BENCH_{D,N,ROOTS,REPS}`. On a NUMA node, pin the threads
(`JULIA_EXCLUSIVE=1`) and leave placement to first touch, which the
block-ownership policy makes domain-local (measured 2026-09-23, 0.92
against 1.00 ns/cell interleaved). Unpinned, run it under `numactl
--interleave=all` instead, which is what the M5 numbers in CODE.md were
taken with and is worth 3–7x over unpinned first touch there:

```bash
TREEAMR_BENCH_N=32 TREEAMR_BENCH_ROOTS=8 bench/scan.sh 1 2 4 8
```

`bench/symmetry_numa.sh` is the SLURM job behind the 2026-09-23 NUMA
measurement in CODE.md ("Where the 64-thread efficiency goes"): the
same benchmark as one 64-thread process against eight domain-bound
8-thread processes, plus placement and thread-count controls and the
stream microbenchmark `bench/stream.jl`. Its finding is that the
memory-streaming phases stop scaling inside one process at 32 threads
for a reason that is not page placement; anyone working on CPU thread
scaling should read that paragraph first. `bench/threads.jl` also
prints allocation and GC cost per call for the per-evaluation path.

`bench/symmetry_affinity.sh` is the same-day follow-up, and a later
CODE.md paragraph ("What one process loses") records it. The reason is
data-to-core affinity between launches. KernelAbstractions' default CPU
schedule and `run_phase!`'s former largest-first slices put a block on
a different core in every phase. The block-ownership policy that
recovers it is implemented (see the Architecture bullet below).
`bench/affinity_mesh.jl` compares it against a `spawn` control. Compare
processes
only in synchronized wall-clock windows (`bench/affinity.jl` explains
why). Best-of timings of independent processes overstate what they get
together.

There is no formatter or linter configured.

## Architecture

Twelve source files, included in dependency order from `src/TreeAMR.jl`; each
layer uses only the ones before it:

| layer | files | what |
|---|---|---|
| threading | `threading.jl` | `threadchunks` (the block-ownership partition), the three host-side parallel-loop helpers everything else is built on, and `launch_by_owner!` |
| residency | `device.jl` | `todevice` (host-built metadata uploaded once, where it is already being rebuilt) and `check_floattype` |
| tree | `morton.jl`, `forest.jl` | `MortonKey{D}` (root, level, coords; curve order computed on the fly), `Forest{D}` = sorted leaf vector + `generation` counter; neighbor finding, `refine!`/`coarsen!`, `balance!` |
| geometry | `geometry.jl` | key + stored cell index → physical coordinates |
| storage | `storage.jl` | `FieldSet`: one `(N+2G₁+c₁, …, N+2G_D+c_D, nvars, nblocks)` array over all leaves, ghosts included; the per-dimension `G` and the centering live here, not on the forest |
| operators | `operators.jl` | `Operators` (family + orders), `check_operators`, Lagrange weights |
| exchange | `schedule.jl`, `ghosts.jl` | `GhostSchedule` (built when the tree changes) and `fill_ghosts!` (replays it) |
| conservation | `interfaces.jl` | `InterfaceSchedule` and `restrict_interfaces!`: the flux fixup at coarse-fine faces, over the same `TransferGroup`/`run_phase!` machinery |
| ODE | `state.jl` | flat interior-only state vector, `scatter!`/`gather!`, `map_blocks!`, the reductions `block_mapreduce` (per block) and `mesh_mapreduce` (one number, where M7's Allreduce will go), `volume_weighted_norm` |
| regrid | `regrid.jl` | flags → `buffered_flags` → `complete_marks` → rebuild → transfer; `adapt_to_initial_data!` |

The ideas that span several files and are easy to violate:

- **Linear octree, leaf-only data.** `forest.leaves` *is* the tree: no node
  objects, no pointers, no coarse data under refined regions. Block `b` of
  any `FieldSet` is `forest.leaves[b]`. Block indices are **not** stable
  across a regrid (slots are compacted), and `FieldSet` is a `mutable struct`
  precisely so that `regrid!` can swap `fs.work` wholesale while callers keep
  their reference.
- **Staleness is by generation, not size.** `rebuild_leaves!` bumps
  `Forest.generation` on every leaf change; a `GhostSchedule` records the
  generation it was built for, and `fill_ghosts!`/`regrid!` refuse a stale
  one. A refine-then-coarsen returns to the same leaf count with different
  leaves, which is why a count check is not enough.
- **Two time scales.** Neighbor finding (`neighbor_keys`, `find_leaf`) is a
  regrid-frequency operation; ghost filling runs at every RHS evaluation.
  Tree queries must never appear in the per-evaluation path — that is the
  whole point of the schedule.
- **One kernel for every transfer.** Same-level copy, restriction,
  prolongation, and the regrid transfer (the `δ = 0` case) are all tensor
  products of `D` one-dimensional `Stencil1D`s, so `transfer_kernel!` in
  `ghosts.jl` serves all of them; a copy is a width-1 stencil with weight 1.
  Transfers sharing `(kind, direction, child offset)` share stencils and are
  batched into one `TransferGroup` = one kernel launch. Stencil construction
  lives in `schedule.jl`; `regrid.jl` reuses `prolongation_stencil` and
  `restriction_stencil` so the two cannot drift apart.
- **Phased ghost fill.** Phase 1: copies and restrictions (they read
  interiors only, so they are race free). Then the physical-boundary hook.
  Phase 2: prolongations, coarsest target level first, because a
  prolongation may read its coarse source's *own ghosts*, which an earlier
  sweep filled. The hook runs *between* the phases, not last: prolongation
  stencils at a domain edge reach tangentially into the source's outer
  ghosts.
- **Neighbor finding is asymmetric across levels** (CODE.md, "Neighbor
  asymmetry"). Ghost filling is formulated as each block asking for its own
  sources, never as reversing a neighbor lookup.
- **Two operator families.** `PointValue`: even orders, both operators are
  Lagrange interpolation at the target center, and restriction *shifts* its
  window inward near an interface. `Conservative`: odd prolongation orders
  built through the primitive function, restriction fixed at the exact
  2-cell average, nothing ever shifts. `Operators` has **no default order** —
  the right one follows from the application's differencing order (the
  interface-order rule: `p` must exceed it by two, for *both* operators). Do
  not add one. `check_operators` ties the orders to `N` and `G`; the
  constraints are listed under "Blocks" in CODE.md.
- **Interiors-only state vector.** An RHS is `scatter!` → `fill_ghosts!` →
  `map_blocks!`, with kernels writing `du` in state layout via `statearray`.
  `u` is never mutated; the working array is scratch. There is deliberately
  no `semidiscretize`-style wrapper.
- **No subcycling.** One global `dt` from `minimum_spacing`.
- **A regrid moves a block by at most one level**, which is what makes
  parent/child-only transfer sufficient; the transfer asserts it. After
  `regrid!` the schedule is stale and the state vector has a new length, so
  callers rebuild both and `reinit!` (or restart) the integrator.
  `adapt_to_initial_data!` *re-evaluates* the initial data on each new mesh
  rather than interpolating it.
- **KernelAbstractions from the start.** All per-cell work in `src/` is a
  `@kernel` launched through `get_backend(fs.work)`, with `D` and `G` as
  `Val` parameters. The CPU implementation is meant to already be the GPU
  implementation (M6). Don't write plain nested loops for cell work in
  `src/`; host-side driver logic (mark completion, balance, key rebuild)
  stays ordinary Julia — but *threaded* ordinary Julia, via
  `threading.jl`.
- **Bit-identical across thread counts, except floating-point sums.**
  Every work item owns its output slot, no parallel loop shares an
  accumulator, and collecting passes fill one buffer per task and
  concatenate in block order. That is race freedom and a hard invariant:
  the state vector, the leaf array, the schedule and every max or
  integer reduction are bit-identical whatever the thread count.
  Floating-point sums are promised to roundoff only (narrowed after M8,
  so that a device can reduce hierarchically — it does, in two launches
  with 256 lanes per block and no barrier — and MPI can `Allreduce`);
  the CPU fold is still one `mapreduce` per block summed in block
  order, and so still exact, which is why `test/thread_tests.jl`'s
  acceptance test — `test/thread_workload.jl` in subprocesses at two
  thread counts, digests compared byte for byte — passes unchanged. A
  new parallel loop that races on a slot breaks it; a reassociated sum
  would move only its `l2` and `mass` lines, which are the ones to give
  a tolerance if that day comes.
- **A ghost phase is one parallel loop, by owner.** `run_phase!` in
  `ghosts.jl` gives each thread the part of *every* transfer batch
  whose target blocks it owns; a batch is *not* the unit of
  parallelism, because batch sizes differ by orders of magnitude (face
  slab vs corner) and per-batch launches capped the ghost fill at
  ~2.5x. That needs each group's `targetblocks` non-decreasing, which
  every builder guarantees by collecting in block order and
  `test/thread_tests.jl` asserts. Only the CPU backend does this —
  `run_phase!` has a generic method that keeps per-batch launches for
  devices.
- **Every per-block pass runs a block on its owner's thread.** Block
  `b` belongs to the thread whose chunk of `threadchunks(nblocks)`
  contains it. `threaded_chunks` puts chunk `c` on thread `c` with
  sticky tasks, and `launch_by_owner!` runs block-shaped kernels on
  KernelAbstractions' static schedule with one block per workgroup,
  which splits the blocks identically. A new block-shaped CPU launch
  in `src/` goes through `launch_by_owner!`, and a new host loop over
  blocks through `threaded_chunks`. A bare `kernel(backend)(…)` or
  `Threads.@spawn` puts blocks on arbitrary cores and costs up to 2.4x
  on a many-core node (CODE.md, "What one process loses").

Index conventions: per dimension `d`, stored indices run `1:N+2G_d+c_d`
(`c_d = 1` in a vertex-like dimension, `0` in a cell-centered one); the
**owned** range is `G_d+1:G_d+N` and the **closed** range `G_d+1:G_d+N+c_d`.
`coordinates(fs, b, idx)` — which replaced `cell_center` in M8, and takes the
field set rather than the forest — takes **stored** indices, so owned point
`i` is `idx = i + G_d`. Kernels launched by `map_blocks!` get the global index
`(i1, …, iD, b)` with each `i` in `1:N` (or `1:N+c_d` under `closed = true`)
and add `G_d` to reach the working array — **except** under
`stored = true`, the third range, where the loop runs over `1:N+2G_d+c_d`
and the global index already *is* the stored index, so the kernel adds
nothing. A kernel written for one form is wrong under the other and still
in bounds. Directions are
`δ ∈ {-1, 0, 1}^D` from `alldirections(Val(D))`; `+1` names the high
ghost slab.

## Tests

`test/runtests.jl` holds the M1 tests inline and `include`s
`ghost_tests.jl`, `centering_tests.jl`, `interface_tests.jl`,
`allvariables_tests.jl`, `state_tests.jl`, `regrid_tests.jl`, `wave_tests.jl`,
`wave_cell_tests.jl`, `burgers_tests.jl`, `type_tests.jl`,
`thread_tests.jl`, `gpu_tests.jl` (M2–M8). The wave
study comes in two halves: `wave_tests.jl` is the **vertex-centered**
one (M8a), and `wave_cell_tests.jl` is the M3 cell-centered study kept
verbatim so its numbers stay under test. Five helper files are not
tests:

- `oracles.jl`, `ghost_oracles.jl` — deliberately naive, independent
  reference implementations (bit-plane Morton comparison, exact `Rational`
  box geometry, analytic polynomials, Gauss–Legendre cell averages). Property
  tests compare the package against *these*, never against the package's own
  neighbor search or geometry. Keep that independence when adding oracles.
- `wave.jl` — the scalar wave equation as an application of the mesh
  (`WaveProblem`, `wave_rhs!`, `wave_errors`, `track_pulse`,
  `uniform_pulse`). It lives in the tests because the package has no physics.
  Every entry point takes `centering`, defaulting to `vertexcentered(D)`;
  `wave_forest` does not, because a centering does not change how space is
  cut into blocks.
- `burgers.jl` — Burgers' equation as a **conservative** application
  (`BurgersProblem`, `burgers_rhs!`, `burgers_errors`, `track_shock`,
  `uniform_shock`), the M8b counterpart of `wave.jl`: the three-step
  right-hand side, a cell-centered state with `G = 2` and `D`
  face-centered flux sets with `G = 0`, and `fixup = false` as the
  negative control for conservation. It reuses `to_backend` and
  `convergence_rate` from `wave.jl` and `cell_average` from
  `ghost_oracles.jl`, so `runtests.jl` includes it after both.
- `thread_workload.jl` — a standalone script, not `include`d. The thread
  count is a command-line argument to Julia, so the M5 acceptance test runs
  this in subprocesses at two thread counts and compares their output byte
  for byte. It is deliberately self-contained (its own RK4 and SSPRK3, no
  ODE package) so a subprocess starts in a couple of seconds; keep it that
  way, and keep everything it prints deterministic.

Testset names are claims ("Coarsening conserves any field exactly", not
"coarsening test"), and each opens with a comment naming the failure mode it
guards. Convergence rates and conservation are asserted as numbers with
tolerances; when a measured number changes, the specification is updated too.

## Spec-first workflow

CODE.md is not a historical sketch; it is maintained alongside the code and
records decisions, their reasons, and *measured* results, with markers like
"(decided)", "(amended in M3)", "(measured in M4)". The commit history
follows a pattern: implement → measure → record what was learned in CODE.md,
often as its own commit ("Record the measured buffer-width lower bound in the
specification"), with the measured numbers in the commit body. When an
implementation shows the spec was wrong or incomplete, amend the spec and say
so in it rather than silently diverging.

When a milestone lands, update the status in `README.md` and
`docs/src/index.md`, and mark the milestone *(Done.)* in CODE.md.

## Conventions

- 4-space indent, wrap at about 90 columns.
- Explicit `return` on the last line of any non-trivial function.
- `ntuple(d -> …, D)` / `ntuple(d -> …, Val(D))` and `Base.setindex` on
  tuples rather than comprehensions or arrays in kernel-adjacent code.
- Unicode in mathematical contexts (`δ`, `φ`, `ω`, `∂ₜ`, `≈`).
- Keyword-heavy constructor and driver signatures, with **no default for
  anything the caller must think about** (`N`, `G`, both operator orders).
- `ArgumentError`s say *why*, not just what — see `Operators()` or
  `check_operators` for the tone. Tests assert on the reason with
  `@test_throws "substring"`.
- Docstrings and file-header comments are prose-first: what it is, then why
  it is that way, pointing at CODE.md for the argument. Internal helpers get
  `#` comments in the same voice.
- `D` is a type parameter everywhere; `N`, `G`, and the operator orders are
  runtime values. Test in `D = 1, 2, 3` (3D with smaller sizes where speed
  matters).

## Repository facts

- Remote: `github.com/eschnett/TreeAMR.jl`, branches `main` and `gh-pages`
  only. **Registered in General** since 2026-09-21; 0.1.1 is the current
  release. TagBot (`.github/workflows/TagBot.yml`) creates the tag and the
  GitHub release for each registered version, and needs the write deploy
  key behind `DOCUMENTER_KEY` to push them — the file says why. Both
  downstreams bound TreeAMR by `[compat]` over the `0.1` series, so a
  `0.1.x` release lands on their next resolve, and an API break has to go
  to `0.2`.
- All `Manifest.toml` files (root, `test/`, `docs/`) and `docs/build/` are
  gitignored.
- `.claude/worktrees/` holds a leftover git worktree from an earlier session.
  Git ignores it; you should too when searching — it is a stale copy of the
  tree and will produce duplicate grep hits.

## Downstream: TreeWave

`~/src/jl/TreeWave` (github.com/eschnett/TreeWave.jl) is the sample
application: the scalar wave equation with a Löhner refinement criterion,
ported from `test/wave.jl`. It exists so that there is a real downstream user
of the *public API only*. Facts that matter here:

- It takes TreeAMR from the **General registry**, not from `main` and not
  from this checkout: `Project.toml` bounds it with `TreeAMR = "0.1.0"`
  under `[compat]`, and `bin/Project.toml` inherits that bound through
  its `TreeWave = {path = ".."}` source. The `[sources]` pins to `main`
  went when 0.1.1 was released, which is also what dropped its Julia
  floor to 1.10. So neither a push to `main` nor an uncommitted change
  here reaches it; a change arrives with the next release. Before
  tagging one that touches the API, run TreeWave's tests (`julia
  --project=. -e 'using Pkg; Pkg.test()'` there, about 1.5 min) against
  this checkout: copy TreeWave somewhere scratch and `Pkg.develop` this
  worktree into the copy, never into the real one, whose manifest would
  keep pointing at the checkout until someone runs `Pkg.free`.
- **TreeWave is ported to M8 and green.** It takes `FieldSet(forest,
  nvars; G, centering, backend)`, the `fs => schedule` form of `regrid!`,
  `GhostSchedule(fs, ops)`, and `coordinates(fs, b, idx)` in `bin/` —
  310 tests passing against the registered 0.1.1 (measured 2026-09-22).
  `bin/` is outside `src/` and `test/`, so `Pkg.test` never runs it; its
  CI's `viewer` job renders every figure on each push, which is what
  catches breakage there.
- Its `src/` calls: `Forest`, `refine!`, `balance!`, `nleaves`, `level`,
  `maxlevel`, `spacing`, `minimum_spacing`, `block_spacings`,
  `block_extent`, `MortonKey`, `FieldSet`, `nblocks`, `blockkey`,
  `fill_by_coordinates!`, `Operators`, `GhostSchedule`, `fill_ghosts!`,
  `statelength`, `statevector`, `statearray`, `scatter!`, `gather!`,
  `map_blocks!`, `block_mapreduce`, `volume_weighted_norm`, `flag_blocks`,
  `firing_boxes`, `regrid!`, `adapt_to_initial_data!`, `cellcentered`,
  `vertexcentered`, the `RegridFlag` values, and the `(flag, box)` flag
  form; `bin/` adds `blockview`, `interiorview` and `coordinates`, and
  its tests add `buffered_flags`, `complete_marks` and `block_origin`.
  `hostcopy` is its own, in its `device.jl`, and it does not use
  `todevice`. Renaming or re-signaturing any of these breaks it.
- Its `CLAUDE.md` and `CODE.md` record API sharp edges found from the
  outside — a keyword named `maxlevel` shadows the exported
  `maxlevel(forest)` inside a function body; `coordinates` taking stored
  indices is an easy off-by-`G`. Read them when changing anything
  user-facing.
- Mesh machinery belongs here; physics belongs there. If a TreeWave change
  turns out to be about trees, ghosts, or interpolation, it comes upstream.

## Downstream: TreeHydro

`~/src/jl/TreeHydro` (github.com/eschnett/TreeHydro.jl) is the second
worked application: Newtonian ideal hydrodynamics with a
high-resolution shock-capturing finite-volume scheme — the
*conservative* counterpart of TreeWave, and the heaviest downstream user
of M8. It is **implemented, not a sketch**: sixteen files in `src/`,
fourteen test files, two viewers in `bin/`, its own CI and Codecov, and
milestones H0–H5 marked done in its `CODE.md` — the scheme, coarse-fine
faces, regridding, Sedov with the atmosphere reset, Kelvin–Helmholtz
with its viewers — with measured results recorded through step 11;
H6 (precision, threads, device) is next. Its two "Upstream
prerequisites" (`map_blocks!(…; stored = true)` and `AllVariables`) are
satisfied and released, so that section of its `CODE.md` is history.

It pins **neither `main` nor this checkout**: since TreeAMR 0.1.1 reached
the General registry, its `Project.toml` and `bin/Project.toml` have no
`[sources]` entry for TreeAMR, only `TreeAMR = "0.1.1"` under `[compat]`.
A change here reaches its tests only once it is tagged and registered,
a higher bar than a push; to try one sooner, `Pkg.develop` this checkout
into a scratch copy of TreeHydro, never the real one. Its suite is much
longer than TreeWave's — 11622 tests in 3.5 to 4.5 minutes at one
thread, against TreeWave's 310 in 1.5 — so TreeWave stays the cheap
downstream check and this is the thorough one. It is worth the minutes
for anything that touches the exchange, the interface restriction or
the operators, because it is the only place conservation at coarse-fine
faces is exercised by a real scheme rather than by Burgers in `test/`.

It is the only caller of several things, which makes it the only test
of them outside this repo: `InterfaceSchedule` / `restrict_interfaces!`,
the physical-boundary hook at all (`boundary_by_coordinates` over an
`AllVariables` callback; TreeWave is periodic throughout),
`map_blocks!(…; stored = true)`, `facecentered` field sets with `G = 0`,
`total_mass`, and `regrid!` over several `fs => schedule` pairs including
`fs => nothing`. Beyond those its `src/` calls `Forest`, `refine!`,
`balance!`, `nleaves`, `level`, `spacing`, `minimum_spacing`,
`block_spacings`, `block_extent`, `FieldSet`, `nblocks`, `blockkey`,
`coordinates`, `fill_by_coordinates!`, `Operators`, `GhostSchedule`,
`fill_ghosts!`, `statelength`, `statevector`, `statearray`, `scatter!`,
`gather!`, `map_blocks!`, `block_mapreduce`, `volume_weighted_norm`,
`firing_boxes` with the `(flag, box)` form of `Refine`/`Keep`/`Coarsen`,
and `adapt_to_initial_data!`. Its tests add `Conservative` (every
`Operators` they build), `maxlevel`, `block_origin`, `cellcentered`,
`staggers`, `RegridFlag` and `buffered_flags`; `bin/` adds
`interiorview`. It uses neither `todevice` nor a TreeAMR `hostcopy`:
`to_backend` and `hostcopy` are its own, in its `device.jl`.

Which of its tests reach TreeAMR: `prerequisite_tests.jl` tests the mesh
directly — `AllVariables` runs once per point, a `stored = true` launch
reaches every ghost, and a name list asserts the resolved release still
exports the M8 surface. The rest go through the scheme:
`interface_tests.jl` (the fixup is what conserves at a coarse-fine face;
the interface-order rule for a system), `sod_tests.jl` and
`sedov_tests.jl` (the Dirichlet hook on a face, and at a corner and an
edge — the M2 ordering case), `refinement_tests.jl` (`firing_boxes`, the
four marks, `buffered_flags`), `driver_tests.jl` and
`kelvinhelmholtz_tests.jl` (`regrid!`, `adapt_to_initial_data!`,
conservation through the regrids), `evolution_tests.jl` and
`reset_tests.jl` (`fill_ghosts!` and `map_blocks!` over `statearray`,
through the right-hand side and the reset). `eos_`, `riemann_`,
`exact_riemann_` and `precision_tests.jl` are pointwise physics and touch
nothing here. Its `bin/`, like TreeWave's, is run by its CI's `viewer`
job rather than by `Pkg.test`.

Mesh machinery belongs here; physics belongs there — the same rule as
for TreeWave.
