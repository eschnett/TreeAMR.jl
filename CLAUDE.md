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
of that running `test/thread_workload.jl` in two subprocesses):

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
from `TREEAMR_BENCH_{D,N,ROOTS,REPS}`. On a NUMA node, run it under
`numactl --interleave=all` — that is worth 3–7x at 64 threads and is
what the numbers in CODE.md were taken with:

```bash
TREEAMR_BENCH_N=32 TREEAMR_BENCH_ROOTS=8 bench/scan.sh 1 2 4 8
```

There is no formatter or linter configured.

## Architecture

Twelve source files, included in dependency order from `src/TreeAMR.jl`; each
layer uses only the ones before it:

| layer | files | what |
|---|---|---|
| threading | `threading.jl` | `threadchunks` and the three host-side parallel-loop helpers everything else is built on |
| residency | `device.jl` | `todevice` (host-built metadata uploaded once, where it is already being rebuilt) and `check_floattype` |
| tree | `morton.jl`, `forest.jl` | `MortonKey{D}` (root, level, coords; curve order computed on the fly), `Forest{D}` = sorted leaf vector + `generation` counter; neighbor finding, `refine!`/`coarsen!`, `balance!` |
| geometry | `geometry.jl` | key + stored cell index → physical coordinates |
| storage | `storage.jl` | `FieldSet`: one `(N+2G₁+c₁, …, N+2G_D+c_D, nvars, nblocks)` array over all leaves, ghosts included; the per-dimension `G` and the centering live here, not on the forest |
| operators | `operators.jl` | `Operators` (family + orders), `check_operators`, Lagrange weights |
| exchange | `schedule.jl`, `ghosts.jl` | `GhostSchedule` (built when the tree changes) and `fill_ghosts!` (replays it) |
| conservation | `interfaces.jl` | `InterfaceSchedule` and `restrict_interfaces!`: the flux fixup at coarse-fine faces, over the same `TransferGroup`/`run_phase!` machinery |
| ODE | `state.jl` | flat interior-only state vector, `scatter!`/`gather!`, `map_blocks!`, `block_mapreduce`, `volume_weighted_norm` |
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
- **No subcycling, ever.** One global `dt` from `minimum_spacing`. This is a
  permanent design commitment, not a simplification to remove later.
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
- **Bit-identical across thread counts.** This is a hard invariant, not
  an aspiration: no parallel loop shares an accumulator, reductions form
  one partial per block and sum them in block order, and collecting
  passes fill one buffer per task and concatenate in block order. A new
  parallel loop that breaks this breaks `test/thread_tests.jl`'s
  acceptance test, which runs `test/thread_workload.jl` in subprocesses
  at two thread counts and compares digests byte for byte.
- **A ghost phase is one parallel loop.** `run_phase!` in `ghosts.jl`
  flattens a phase's transfer batches into `PhaseSlice`s of roughly
  equal cell count and deals them out largest first; a batch is *not*
  the unit of parallelism, because batch sizes differ by orders of
  magnitude (face slab vs corner) and per-batch launches capped the
  ghost fill at ~2.5x. Only the CPU backend does this — `run_phase!`
  has a generic method that keeps per-batch launches for devices.

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
  only. Not registered.
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

- It pins TreeAMR to **GitHub `main`** via a `[sources]` entry, not to this
  checkout. A push to `main` here is immediately what TreeWave's tests see;
  an uncommitted change here is invisible to it. Before pushing an API
  change, run TreeWave's tests (`julia --project=. -e 'using Pkg;
  Pkg.test()'` there, about 20 s), temporarily dev'ing this checkout into
  TreeWave's environment if the change is unpushed — and revert TreeWave's
  `Project.toml` and `Manifest.toml` afterwards.
- **TreeWave is still pre-M8 and will break when `m8` lands on `main`.**
  Measured against this checkout: it calls `FieldSet(forest, nvars;
  backend)` with no `G`, the three-argument `regrid!(forest, fs, schedule;
  …)`, and `GhostSchedule(forest, ops; …)` without `G` — the first two now
  throw by construction, which is the message M8a step 1 added for exactly
  this caller — and `bin/visualize.jl` still calls `cell_center`, which no
  longer exists. Note that `bin/` is outside `src/` and `test/`, so its
  breakage does not show up in TreeWave's own test run. Porting it is part
  of landing M8, not an afterthought: that is what the pin to `main` is
  for.
- It calls: `Forest`, `refine!`, `balance!`, `nleaves`, `level`, `maxlevel`,
  `spacing`, `minimum_spacing`, `block_spacings`, `block_extent`,
  `coordinates`, `FieldSet`, `nblocks`, `blockkey`, `blockview`,
  `interiorview`, `fill_by_coordinates!`, `Operators`, `GhostSchedule`,
  `fill_ghosts!`, `statevector`, `statearray`, `scatter!`, `gather!`,
  `map_blocks!`, `block_mapreduce`, `volume_weighted_norm`,
  `flag_blocks`, `buffered_flags`,
  `complete_marks`, `regrid!`, `adapt_to_initial_data!`, the `RegridFlag`
  values, and the `(flag, box)` flag form. Renaming or re-signaturing any of
  these breaks it.
- Its `CLAUDE.md` and `CODE.md` record API sharp edges found from the
  outside — a keyword named `maxlevel` shadows the exported
  `maxlevel(forest)` inside a function body; `coordinates` taking stored
  indices is an easy off-by-`G`. Read them when changing anything
  user-facing.
- Mesh machinery belongs here; physics belongs there. If a TreeWave change
  turns out to be about trees, ghosts, or interpolation, it comes upstream.

## Downstream: TreeHydro

`~/src/jl/TreeHydro` is the second worked application: Newtonian ideal
hydrodynamics with a high-resolution shock-capturing finite-volume
scheme — the *conservative* counterpart of TreeWave, and the first
downstream user of M8. It is **design only so far**: its `CODE.md` is
the whole package, nothing is implemented, and nothing here is under
test from it yet.

It will lean on `map_blocks!(…; stored = true)` and `AllVariables` —
both added here as its prerequisites, see "Application interface" in
`CODE.md` — and on `InterfaceSchedule`/`restrict_interfaces!`,
`CellBoundary` (it is the first caller of the physical-boundary hook at
all; TreeWave is periodic throughout), `firing_boxes`, and
`block_mapreduce`. Its `CODE.md` section "Upstream prerequisites" is
where it records what it still needs from here; read it before changing
any of those.
