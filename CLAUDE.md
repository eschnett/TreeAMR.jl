# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Contents**

- [Read first, and where things go](#read-first-and-where-things-go)
- [What this is](#what-this-is)
- [Commands](#commands)
- [Rules and sharp edges](#rules-and-sharp-edges)
- [Conventions](#conventions)
- [Repository facts](#repository-facts)

## Read first, and where things go

Read `CODE.md` before changing anything non-trivial: it states *why*
things are the way they are, and it is kept in sync with the code. Its
"Code structure" section has the layer table and the rules that span
several files, and its "Tests" section the test files and their
helpers. The markdown files and what goes in each:

- `README.md` — for users: what the package does, installation, a short
  example, the status, and a pointer to `CODE.md`.
- `CLAUDE.md` — this file: what to read, the commands, short rules
  pointing at `CODE.md`, and the repository facts. Keep it short; no
  status narrative, file inventory, release history or detailed
  measurements.
- `CODE.md` — the current design and the reasons for it: the method, the
  code and its structure, the interfaces and their contracts, the tests,
  current measurements that support a current decision, known
  limitations, and open questions. Present tense, no dates or step
  numbers on decisions; mark only what is not settled, as **(open)** or
  **(proposed)**.
- `PLAN.md` — the concrete plan: milestones, each with what it delivers
  and how one knows it is done. A finished milestone is one line saying
  it is done and where its results went; keep the numbering, since other
  documents cite it.
- `HISTORY.md` — how the package got where it is: who decided what and
  when, alternatives tried or rejected and why, measurements of code
  that no longer exists or that later measurements replaced, the
  milestone records with their steps, the downstream checks, and the
  release history. Ordered by topic, following `CODE.md`'s sections.
- `TODO.md` is Erik's personal to-do list. **Do not modify it**, do not
  move its content elsewhere, and do not move content into it.

The test between `CODE.md` and `HISTORY.md`: would someone changing the
code today need this? Then it belongs in `CODE.md`, rationale included;
where a reason lives in `HISTORY.md`, `CODE.md` links to it. Every fact
lives in one file; elsewhere, link to it.

**When an implementation shows the design was wrong or incomplete**,
fix `CODE.md` to describe the current state, and record in `HISTORY.md`
what changed, when and why. Do not leave amendment markers ("amended in
step N", "(decided)") in `CODE.md`. The commit history follows a
pattern: implement → measure → record what was learned, often as its
own commit ("Record the measured buffer-width lower bound in the
specification"), with the measured numbers in the commit body. When a
milestone lands, update the status in `README.md`, in
`docs/src/index.md` and at the top of `CODE.md`, and replace the
milestone in `PLAN.md` by its one-line done entry, its record going to
`HISTORY.md`.

**Keep the tables of contents current**: `README.md`, `CLAUDE.md`,
`CODE.md`, `PLAN.md` and `HISTORY.md` each have one near the top, as
links (`##` and `###` headings for `CODE.md`, `PLAN.md` and
`HISTORY.md`, `##` only for `README.md` and `CLAUDE.md`). A heading
added, renamed or removed changes it, and a renamed or moved heading
needs every reference to it updated — docstrings, comments, tests and
the other documents cite sections by heading.

## What this is

TreeAMR.jl is a Julia package: a tree-based (octree-style) block-structured
AMR mesh — storage, ghost exchange, inter-grid operators, regridding — with
**no physics**. `CODE.md` is the authoritative design document; the status
is at its top, in `README.md` and in `docs/src/index.md`, and the next
milestone in `PLAN.md`.

## Commands

Full test suite, about 8 min at one thread and at eight; the
thread-independence test spends about 50 s of it in two subprocesses
running `test/thread_workload.jl`. The MPI test's two `mpiexec` jobs,
about a minute each and nearly all compilation, run *beside* the suite
where the machine has 8+ threads and 24+ GB (`test/mpi_jobs.jl`); on a
smaller machine, CI's runners among them, they run one after the other
(`TREEAMR_TEST_MPI_CONCURRENT=0` or `1` overrides the choice). The
per-file times are in `HISTORY.md` (M7 step 9, M12's steps).

**The suite is compilation-bound, not kernel-bound**, so do not try to
shorten it by making the kernels faster (`CODE.md`, "Tests", has the
measurement):

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
`test/Project.toml` locates TreeAMR (`..`) and the unregistered
IMEXRungeKutta (GitHub `main`) through `[sources]`:

```bash
julia --project=test -e 'using Test, Random, TreeAMR; include("test/oracles.jl"); include("test/ghost_oracles.jl"); include("test/wave.jl"); include("test/regrid_tests.jl")'
```

The MPI tests (M7) are part of `Pkg.test`: `test/mpi_tests.jl` launches
`test/mpi_workload.jl` under `MPI.mpiexec()` — the launcher of the MPI
binary MPIPreferences selects for the load path: MPICH_jll by default
(CI, Julia 1.11 here), MPIABI_jll on Julia 1.13 here, which a
`LocalPreferences.toml` in the global v1.13 environment selects — with
`setenv(cmd, mpiexec().env)`, since interpolating `mpiexec()` into a
larger command drops its library paths. On its own (it needs no helper
file; about 2 min here, where both jobs start beside the serial
reference, and about 4 on CI's one-after-the-other path):

```bash
julia --project=test -e 'using Test, TreeAMR, HDF5; @testset "mpi" begin include("test/mpi_tests.jl") end'
```

The workload by hand, serially and at three ranks; every line but the
`#` lines and the last digits of the `sum` lines must agree:

```bash
julia --project=test test/mpi_workload.jl > /tmp/serial.txt
julia --project=test -e 'using MPI; m = MPI.mpiexec(); run(pipeline(setenv(`$m -n 3 $(Base.julia_cmd()) --threads=1 --project=test test/mpi_workload.jl mpi`, m.env); stdout = "/tmp/n3.txt"))'
diff /tmp/serial.txt /tmp/n3.txt
```

The device counterpart, `test/mpi_device_tests.jl`, is not in
`Pkg.test` (the test environment has no device package and must not
gain one). It runs in a scratch environment that develops this checkout
and adds the device package, MPI, KernelAbstractions, SHA and Test
(`TREEAMR_TEST_RANKS`, default `"2 3"`; `TREEAMR_TEST_T`;
`TREEAMR_TEST_DEVICEAWARE=1` for the direct path):

```bash
TREEAMR_TEST_BACKEND=metal julia --project=<env> test/mpi_device_tests.jl
```

On a fresh clone the test environment has no Manifest; run this once
first (`Pkg.test` needs nothing, it resolves an environment of its own).
The same line with `docs` sets up the docs environment, which locates
TreeAMR the same way:

```bash
julia --project=test -e 'using Pkg; Pkg.instantiate()'
```

Coverage, the way CI's single-threaded cells measure it (`CODE.md`,
"Tests", says what CI covers and what instrumentation costs). The `.cov`
files scattered through `src/` are gitignored; delete them before the
next run, because counts from separate runs accumulate:

```bash
julia --project=. -e 'using Pkg; Pkg.test(; coverage=true, julia_args=["--threads=4"])'
```

```bash
find src -name '*.cov' -delete
```

Docs build. This is also the **only place doctests run** — `Pkg.test` does
not run the `jldoctest` blocks in docstrings and `docs/src/`:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
```

```bash
julia --project=docs docs/make.jl
```

Julia 1.11, the floor. `juliaup` has 1.11 installed, but checking a
suspect test on it is not simply `Pkg.test()` in this checkout: an older
Julia may not read a `Manifest.toml` a newer one resolved (on 1.11.9
`Pkg.test()` died with "can not merge projects" whenever
`test/Manifest.toml` existed). Copy the tree without any manifest and
run the test file directly (the whole suite takes about 8 min there):

```bash
rm -rf /tmp/amr111 && mkdir /tmp/amr111 && tar -cf - --exclude=Manifest.toml --exclude=.git --exclude=.claude . | tar -xf - -C /tmp/amr111
```

```bash
cd /tmp/amr111 && julia +1.11 --project=test -e 'using Pkg; Pkg.instantiate()' && julia +1.11 --project=test test/runtests.jl
```

Downstream checks. Before tagging a release that touches the API, run
TreeWave's tests (`julia --project=. -e 'using Pkg; Pkg.test()'` there,
about 1.5 min) against this checkout: copy TreeWave somewhere scratch and
`Pkg.develop` this worktree into the copy, never into the real one, whose
manifest would keep pointing at the checkout until someone runs
`Pkg.free`. TreeWave is the cheap check; TreeHydro (about 5 min) is the
thorough one, worth the minutes for anything that touches the exchange,
the interface restriction or the operators, because it is the only
place conservation at coarse-fine faces is exercised by a real scheme
rather than by Burgers in `test/`. The same scratch-copy rule holds for
it. What each downstream calls is under "Downstream applications" in
`CODE.md`.

Thread scaling (`bench/threads.jl`, driven by `bench/scan.sh`, which
takes a list of thread counts and prints a speedup table). Sizes come
from `TREEAMR_BENCH_{D,N,ROOTS,REPS}`. On a NUMA node, pin the threads
(`JULIA_EXCLUSIVE=1`) and leave placement to first touch, which the
block-ownership policy makes domain-local. Unpinned, run it under
`numactl --interleave=all` instead, which is what the M5 numbers in
`CODE.md` were taken with (`CODE.md`, "Parallelism"). `bench/threads.jl`
also prints allocation and GC cost per call for the per-evaluation path:

```bash
TREEAMR_BENCH_N=32 TREEAMR_BENCH_ROOTS=8 bench/scan.sh 1 2 4 8
```

`bench/symmetry_numa.sh` is the SLURM job behind the NUMA measurement in
`CODE.md` ("Where the 64-thread efficiency goes"): the same benchmark as
one 64-thread process against eight domain-bound 8-thread processes,
plus placement and thread-count controls and the stream microbenchmark
`bench/stream.jl`; anyone working on CPU thread scaling should read that
paragraph first. `bench/symmetry_affinity.sh` is the same-day follow-up,
and `CODE.md`'s "What one process loses" records it;
`bench/affinity_mesh.jl` compares the block-ownership policy against a
`spawn` control. Compare processes only in synchronized wall-clock
windows (`bench/affinity.jl` explains why). Best-of timings of
independent processes overstate what they get together.

`bench/interpolate.jl` times point interpolation (M11) on the horizon
finder's batch and larger ones, on any backend; `bench/symmetry_interpolate.sh
cpu|cuda` runs it on Symmetry across thread counts and NUMA placements, or
on an H200. `HISTORY.md`'s M11 entry has the numbers.

`bench/copies.jl` times `scatter!`, `gather!` and `fill_ghosts!` per
owned point and in TB/s against a linear copy, at the downstream
TreeGeneralizedHarmonic's sizes by default (uniform, 20 variables, `G =
3`, vertex-centered); `bench/copy_index.jl` prices seven ways of forming
the scatter's index; `bench/copy_groups.jl` times the fill group by
group, flat against shaped. `bench/symmetry_copies.sh` runs all three on
an H200 for a baseline checkout beside this one (`TREEAMR_BASE`,
`TREEAMR_COPIES_STAGES`), then the suite on CUDA. `CODE.md`'s "The copy
kernels on a device" has the numbers.

The Symmetry jobs write SimWatch status files through `bench/simwatch.sh`
(`simwatch_begin DIR NAME NSTAGES`, `simwatch_stage MESSAGE`, an exit
trap for `finished`/`failed`, a 60 s heartbeat; `simwatch_queued` on the
submit side), into `$SLURM_SUBMIT_DIR/$SLURM_JOB_NAME-$SLURM_JOB_ID`, so
`simwatch <submit dir>` on Symmetry shows them; `bench/symmetry_copies.sh`
is the first to use it. Give a new job script the same few lines, and
never point a new job at a directory or environment a running job uses.

`bench/mpi.jl` is M7's weak-scaling smoke test: a fixed number of blocks
per rank (`TILES` stacked tiles, one a rank), timed in synchronized
windows, minimum and median; `bench/mpiscan.sh P…` launches each rank
count through `MPI.mpiexec()` at `TREEAMR_BENCH_THREADS` threads a rank
and prints the table through `bench/mpitable.awk`
(`TREEAMR_BENCH_BACKEND`, `TREEAMR_BENCH_DEVICEAWARE`,
`TREEAMR_BENCH_TILES` for a serial control). `bench/replicated.jl` needs
no MPI: it simulates rank `P÷2` of `P` in one process to price the
replicated `O(nleaves)` passes at large leaf counts. `HISTORY.md`'s M7
step 7 has both tables (laptop only):

```bash
TREEAMR_BENCH_N=8 bench/mpiscan.sh 1 2 4
julia -t 4 --project=test bench/replicated.jl 8 64 512
```

Three SLURM jobs hold the M7 measurements, all run on 2026-10-02:
`bench/symmetry_mpi.sh` (weak scaling, one rank per NUMA domain at 8
threads, 1–4 nodes, plus single-process controls),
`bench/symmetry_checkpoint_mpi.sh` (checkpoint throughput on BeeGFS: the
shared file of step 6, then the part files of step 6b, once per `io` in
`TREEAMR_CKPT_IO`) and `bench/symmetry_mpi_gpu.sh` (`mpi_device_tests.jl` on H200s,
staged and, with a CUDA-aware system MPI, direct). Each launches
MPICH_jll through `srun --mpi=pmi2` by default or a system MPI through
MPIPreferences (`TREEAMR_MPI=system TREEAMR_MPI_MODULE=…`).
`bench/symmetry_checkpoint_stress.sh` runs `bench/checkpoint_stress.jl`,
the reproducers of the multi-node checkpoint loss, from
`save_checkpoint` (now the part-file writer, verified by serial loads on
two nodes; `STRESS_IO`) down to plain `pwrite`;
`bench/checkpoint_inspect.jl` and `bench/checkpoint_layout.jl` read a
damaged shared file of step 6. Nothing in the package writes one file
from several processes any more; anything that would must be checked
with them first.

`bench/stepping.jl` times one time step by integrator — OrdinaryDiffEq's
RK4 and SSPRK33 against IMEXRungeKutta's, broadcast and by owner — and
runs in the test environment, which has both:

```bash
TREEAMR_BENCH_ROOTS=8 TREEAMR_BENCH_SCRIPT=bench/stepping.jl TREEAMR_BENCH_PROJECT=test bench/scan.sh 1 8
```

`bench/checkpoint.jl` times `save_checkpoint` and `load_checkpoint` (M9a)
for each HDF5 filter setting, on a smooth wave pulse and a blast wave
with a uniform atmosphere, and prints the file size, the ratio and GB/s
for a save with `sync = false`, one with the default `sync = true`
(which adds the flush to stable storage: `fsync`, or `F_FULLFSYNC` on
macOS, where `fsync` does not wait), and a load. It
runs in the test environment with the built-in filters only; the
filter packages (H5Zzstd, H5Zlz4, H5Zbitshuffle) are used
where the environment has them, which means a scratch environment that
develops this checkout — never add them to `test/Project.toml`.
`TREEAMR_BENCH_DIR` puts the files on the file system under test. A
load right after a save reads the page cache; `CODE.md`'s "Throughput and
filters" under "Checkpoint and restart" has the numbers and that
caveat:

```bash
julia --project=test bench/checkpoint.jl
```

There is no formatter or linter configured.

## Rules and sharp edges

Each rule is explained in `CODE.md`, mostly under "Rules that span
several files" in "Code structure"; the section named after a rule is
the one that says why.

- **Leaf-only data, local block indices.** `forest.leaves` *is* the
  tree. Every block index is local to the rank, while tree queries stay
  global: walk `blockrange` and size by `nblocks`, never index
  `forest.leaves` by a block index or size a per-block array by
  `nleaves` — that passes every serial test and is wrong under MPI.
- **Staleness is by generation, not size**; a count check is not
  enough.
- **Two time scales.** Tree queries must never appear in the
  per-evaluation path ("Ghost filling").
- **One kernel for every transfer.** Keep the CPU launch path
  type-stable, hide unavoidable unions behind `Base.inferencebarrier` on
  the device path only, and compare `--trace-compile-timing` of
  `test/mpi_workload.jl` against `main` for any change to a launch path
  ("The copy kernels on a device").
- **The copy kernels launch flat on a device, shaped on the CPU**; a new
  copy-like kernel goes the same way. KA's CPU emitter only rewrites
  `@index` at statement level, so assign it before passing it to a
  function.
- **Phased ghost fill**: the boundary hook runs *between* the phases,
  not last.
- **Reflecting faces are transfers, not hooks**, and **rotating seams
  are oriented transfers**; the boundary hook sees only outer faces
  ("Domain and boundaries", "Ghost filling").
- **Neighbor finding is asymmetric across levels**: ghost filling is
  each block asking for its own sources, never a reversed neighbor
  lookup ("Tree structure").
- **`Operators` has no default order.** Do not add one ("Operators").
- **Interiors-only state vector**, and deliberately no
  `semidiscretize`-style wrapper ("Time integration").
- **No subcycling**, and **a regrid moves a block by at most one
  level**; after `regrid!` callers rebuild the schedule and `reinit!`
  ("Regridding").
- **KernelAbstractions from the start.** Don't write plain nested loops
  for cell work in `src/`; host-side driver logic stays ordinary Julia,
  threaded via `threading.jl`.
- **Bit-identical across thread counts, except floating-point sums.** A
  new parallel loop must not race on a slot ("Parallelism").
- **Every per-block pass runs a block on its owner's thread.** A new
  block-shaped CPU launch in `src/` goes through `launch_by_owner!`, and
  a new host loop over blocks through `threaded_chunks` ("What one
  process loses").
- **A checkpoint stores what cannot be recomputed**; do not add JLD2 or
  `Serialization`, and rename with `Base.Filesystem.rename`, not
  `mv(…; force = true)` ("Checkpoint and restart").
- **No file of a checkpoint has two writers or two openers.** Do not
  reintroduce parallel HDF5 or MPI-IO ("Checkpoints without parallel
  I/O").
- **Every forest mutation is collective.** Any new check that can fire
  on some ranks only goes through `collective_checks`. Do not add a
  field to `Forest`: put mutable state in `ForestState` ("Distributed
  meshes").
- **The sender computes.** `Dict` order must never reach a message
  layout, and MPI is called from the calling task only.
- **Reductions are an allgather of per-rank partials**, folded in
  `combine_blocks` (the only site); a number combined from
  `block_mapreduce` is rank-local.
- **HDF5 and MPI are weak dependencies.** Do not make either a hard
  dependency, do not call MPI from `src/`, and do not use `MPI.Comm_dup`
  ("Distributed meshes").
- **Index conventions**: `coordinates` takes *stored* indices, and a
  `map_blocks!` kernel written for the owned form is wrong under
  `stored = true`, and still in bounds ("Index conventions").
- **A seeded RNG stream can differ across Julia versions**, so a test
  whose *assertions* depend on a particular random draw can pass locally
  and fail on the floor: use a seeded RNG for the inputs, but make what
  the test asserts follow deterministically from the setup.
- **Mesh machinery belongs here; physics belongs downstream.** If a
  downstream change turns out to be about trees, ghosts, or
  interpolation, it comes upstream.

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
- Testset names are claims ("Coarsening conserves any field exactly", not
  "coarsening test"), and each opens with a comment naming the failure mode it
  guards. Convergence rates and conservation are asserted as numbers with
  tolerances; when a measured number changes, `CODE.md` is updated too.
- Keep `thread_workload.jl` and `mpi_workload.jl` self-contained, and
  everything they print deterministic (`CODE.md`, "Tests").
- Documenter is strict: every docstring in the module must appear in a
  `@docs` block, and every `` [`name`](@ref) `` must resolve, or the build
  errors out. **Adding a documented function means adding it to the API
  page of its layer**, `docs/src/api/{tree,storage,exchange,ode,regrid,interpolate,io,distributed,internals}.md`;
  `docs/src/index.md` holds no `@docs` blocks. Keep a section heading
  from spelling an exported name exactly (`## Forest` made
  `` [`Forest`](@ref) `` link to the heading, not the docstring).

## Repository facts

- Remote: `github.com/eschnett/TreeAMR.jl`, branches `main` and `gh-pages`
  only. **Registered in General**. TagBot (`.github/workflows/TagBot.yml`)
  creates the tag and the GitHub release for each registered version, and
  needs the write deploy key behind `DOCUMENTER_KEY` to push them — the
  file says why. All three downstreams bound TreeAMR by `[compat]` over
  the `0.1` series, so a `0.1.x` release lands on their next resolve, and
  an API break has to go to `0.2`.
- CI (`.github/workflows/CI.yml`) tests on Julia **1.11** and latest, on
  Linux and macOS. `Project.toml` says `julia = "1.11"`, so no 1.12+
  features. Your local Julia is newer.
- The `[sources]` entries in `test/Project.toml` and `docs/Project.toml`
  are committed and meant to be (a 1.11 key, the reason for the floor).
  Do not `Pkg.develop` TreeAMR into either any more: that is what they
  replace. IMEXRungeKutta is taken from `main`, so its next push reaches
  the next resolve here; `Pkg.update` in `test/` picks it up locally.
- All `Manifest.toml` files (root, `test/`, `docs/`) and `docs/build/` are
  gitignored.
- `.claude/worktrees/` holds a leftover git worktree from an earlier session.
  Git ignores it; you should too when searching — it is a stale copy of the
  tree and will produce duplicate grep hits.
- `TODO.md` is Erik's: do not modify it.
