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

Current state: milestones M0–M8, M10, M11, M9a and M12 are done (tree
core, ghost exchange, ODE coupling, regridding, multi-threading, GPU; then
every centering, per-field-set ghost widths, and conservation at
coarse-fine faces; then reflecting boundaries; then point interpolation;
then checkpoint and restart, through an HDF5 package extension; then MPI;
then a 90° rotating symmetry).
M7 (done 2026-10-02): the forest replicated on every rank, the blocks
distributed in contiguous curve ranges, and the ghost exchange,
interface restriction, reductions, regrid, interpolation and
checkpoints across ranks, bit-identical to serial but for floating-point
sums, through an MPI package extension. Its cluster measurements ran on
Symmetry on 2026-10-02 (CODE.md, steps 6–8). Its checkpoint was first a
shared file written through MPI-IO, which lost data between nodes on
BeeGFS; it was replaced the same day (step 6b, decided with Erik) by
one without parallel I/O — part files per I/O process and an index,
each file with one writer and one opener. What M7 leaves for later is
listed under "Performance work left for later" in CODE.md. It came after
M8 and M10 so the distributed exchange was built once over a
layout-generic schedule that already held the mirrored transfers, and
after M9a because the downstream runs needed to restart before they
needed MPI. M12 (done 2026-10-03, decided with Erik, after M7 and before
M9b): `Forest(…; rotating = (d1, d2))` stores one quadrant of a plane
whose two low faces are glued by a quarter turn, `FieldSet(…; rotation)`
declares how its variables turn, and `RotationPair` fills two sets whose
layouts are each other's swap; the ghost exchange, regrid, interface
schedule, interpolation, checkpoints and MPI all know the seam, and it
runs on devices. 180° and reflecting high walls beside the seam are
left open (CODE.md, "Open questions"). Everything is `D`-generic and
floating-point-type generic. Next is visualization export (M9b), the
other half of the old M9.

`TODO.md` is Erik's personal to-do list. **Do not modify it.**

## Commands

Full test suite (about 8 min at one thread and at eight — 112715 tests
in 7m52 and 112767 in 7m33 after M12, against 110606 in 6m32 and 110658
in 6m28 after M7's step 6b and M9a's 93686 in 4m16 — of
which the thread-independence test spends ~50 s running
`test/thread_workload.jl` in two subprocesses (~26 s each since M12's
rotating cycles), M10's `reflect_tests.jl` ~30 s, and, each run alone
with its own compilation, `exchange_tests.jl` ~49 s, M12's
`rotate_tests.jl` ~60 s and M9a's `checkpoint_tests.jl` ~77 s).
The MPI test's two `mpiexec` jobs, ~65 s each (~55 s before M12) and nearly all
compilation, run *beside* the suite where the machine has 8+ threads and
24+ GB (`test/mpi_jobs.jl`), so `mpi_tests.jl` itself costs ~15 s here;
on a smaller machine, CI's runners among them, they run one after the
other and the suite took about 8 min after M7 (7m52 here with
`TREEAMR_TEST_MPI_CONCURRENT=0`, CI's path; not re-measured after M12).
The per-file times before and after M7's trim are in CODE.md's M7 step
9, M12's in its steps.

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
file; about 1m47 on the sequential path):

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
and adds the device package, MPI, KernelAbstractions, SHA and Test:

```bash
TREEAMR_TEST_BACKEND=metal julia --project=<env> test/mpi_device_tests.jl
```

(`TREEAMR_TEST_RANKS`, default `"2 3"`; `TREEAMR_TEST_T`;
`TREEAMR_TEST_DEVICEAWARE=1` for the direct path). It passed on Metal
in 1m45 (step 8).

On a fresh clone the test environment has no Manifest; run this once
first (`Pkg.test` needs nothing, it resolves an environment of its own).
The same line with `docs` sets up the docs environment, which locates
TreeAMR the same way:

```bash
julia --project=test -e 'using Pkg; Pkg.instantiate()'
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
threaded code paths. The MPI test's five ranks launch through
`Base.julia_cmd()` too, so the MPI extension's coverage comes from them
(not measured in step 9). The `.cov` files scattered through `src/` are
gitignored; delete them before the next run, because counts from
separate runs accumulate:

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

The `[sources]` entries in `test/Project.toml` and `docs/Project.toml` are
committed and meant to be (a 1.11 key, the reason for the floor). Do not
`Pkg.develop` TreeAMR into either any more: that is what they replace.
IMEXRungeKutta is taken from `main`, so its next push reaches the next
resolve here; `Pkg.update` in `test/` picks it up locally.

HDF5 is a **weak dependency** (`[weakdeps]`, `[compat]` 0.17): the
checkpoint implementation is the package extension `TreeAMRHDF5Ext`,
which Julia loads only once HDF5 is loaded beside TreeAMR. The test and
docs environments list HDF5, and `docs/make.jl` does `using HDF5`, so
the suite and the checkpoint doctest see the extension. Without it the
five checkpoint functions have no methods, and an error hint registered
in TreeAMR's `__init__` says to load HDF5. Do not make HDF5 a hard
dependency: an application that never checkpoints should not load it.

Documenter is strict: every docstring in the module must appear in a `@docs`
block, and every `` [`name`](@ref) `` must resolve, or the build errors out.
**Adding a documented function means adding it to the API page of its
layer**, `docs/src/api/{tree,storage,exchange,ode,regrid,interpolate,io,distributed,internals}.md`.
`docs/src/index.md` is the guide (prose and doctests, plus the status) and
holds no `@docs` blocks. The split is there because Documenter's HTML writer
fails the build on any page over 200 KiB (`size_threshold`), and the single
page had reached 178 KiB; the largest page is now about 40 KiB. Keep a
section heading from spelling an exported name exactly (`## Forest` made
`` [`Forest`](@ref) `` link to the heading, not the docstring).

CI (`.github/workflows/CI.yml`) tests on Julia **1.11** and latest, on Linux
and macOS. `Project.toml` says `julia = "1.11"`, so no 1.12+ features. The
floor was 1.10 (the LTS) through 0.1.2 and was raised to 1.11 for 0.1.3
(2026-09-25) across all the Tree* packages, so that unregistered
dependencies can be located with `[sources]` entries, a 1.11 key. Your local Julia is newer. A
seeded RNG stream can differ across Julia versions, so a test whose
*assertions* depend on a particular random draw can pass locally and fail on
the floor: use a seeded RNG for the inputs, but make what the test asserts
follow deterministically from the setup. `juliaup` has 1.11 installed, but
checking a suspect test on it is not simply `Pkg.test()` in this checkout:
an older Julia may not read a `Manifest.toml` a newer one resolved. Copy the
tree without any manifest and run the test file directly (the procedure was
worked out, and measured at 97744 tests in ~2m55 after M10, on 1.10, and
at 112715 tests in 8m09 after M12, on 1.11.9, whose
`Pkg.test()` also died with "can not merge projects" whenever
`test/Manifest.toml` existed):

```bash
rm -rf /tmp/amr111 && mkdir /tmp/amr111 && tar -cf - --exclude=Manifest.toml --exclude=.git --exclude=.claude . | tar -xf - -C /tmp/amr111
```

```bash
cd /tmp/amr111 && julia +1.11 --project=test -e 'using Pkg; Pkg.instantiate()' && julia +1.11 --project=test test/runtests.jl
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

`bench/interpolate.jl` times point interpolation (M11) on the horizon
finder's batch and larger ones, on any backend; `bench/symmetry_interpolate.sh
cpu|cuda` runs it on Symmetry across thread counts and NUMA placements, or
on an H200. CODE.md's M11 entry has the numbers.

`bench/mpi.jl` is M7's weak-scaling smoke test: a fixed number of blocks
per rank (`TILES` stacked tiles, one a rank), timed in synchronized
windows, minimum and median; `bench/mpiscan.sh P…` launches each rank
count through `MPI.mpiexec()` at `TREEAMR_BENCH_THREADS` threads a rank
and prints the table through `bench/mpitable.awk`
(`TREEAMR_BENCH_BACKEND`, `TREEAMR_BENCH_DEVICEAWARE`,
`TREEAMR_BENCH_TILES` for a serial control). `bench/replicated.jl` needs
no MPI: it simulates rank `P÷2` of `P` in one process to price the
replicated `O(nleaves)` passes at large leaf counts. CODE.md's M7 step 7
has both tables (laptop only):

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
load right after a save reads the page cache; CODE.md's "Throughput and
filters" under "Checkpoint and restart" has the numbers and that
caveat:

```bash
julia --project=test bench/checkpoint.jl
```

There is no formatter or linter configured.

## Architecture

Fifteen source files, included in dependency order from `src/TreeAMR.jl`, plus
two package extensions in `ext/`; each layer uses only the ones before it:

| layer | files | what |
|---|---|---|
| threading | `threading.jl` | `threadchunks` (the block-ownership partition), the three host-side parallel-loop helpers everything else is built on, and `launch_by_owner!` |
| residency | `device.jl` | `todevice` (host-built metadata uploaded once, where it is already being rebuilt) and `check_floattype` |
| communicator | `communicator.jl`; `ext/TreeAMRMPIExt.jl` | M7: abstract `Communicator`, `SerialCommunicator` (rank 0 of 1), `communicator`, and the internal verbs (`commrank`, `commsize`, `allgather`, `allgatherv`, `alltoallv`, `bcast`, `commnodes`, `isend`/`irecv`/`waitall`, `hoststaging`), each with a serial method; the MPI extension (weak dependency MPI.jl) adds `MPICommunicator` — a cached `MPI_Comm_dup` with its rank, size and `deviceaware` setting — and one MPI.jl call per verb |
| tree | `morton.jl`, `forest.jl` | `MortonKey{D}` (root, level, coords; curve order computed on the fly), `Forest{D}` = sorted leaf vector + `generation` counter; neighbor finding (oriented across a rotating seam), `refine!`/`coarsen!`, `balance!` (conformity at the seam) |
| geometry | `geometry.jl` | key + stored cell index → physical coordinates |
| storage | `storage.jl` | `FieldSet`: one `(N+2G₁+c₁, …, N+2G_D+c_D, nvars, nblocks)` array over all leaves, ghosts included; the per-dimension `G` and the centering live here, not on the forest, and so do `parity`, `rotation` and their tables; `RotationPair` |
| operators | `operators.jl` | `Operators` (family + orders), `check_operators`, Lagrange weights |
| exchange | `schedule.jl`, `ghosts.jl` | `GhostSchedule` (built when the tree changes) and `fill_ghosts!` (replays it) |
| conservation | `interfaces.jl` | `InterfaceSchedule` and `restrict_interfaces!`: the flux fixup at coarse-fine faces, over the same `TransferGroup`/`run_phase!` machinery |
| ODE | `state.jl` | flat interior-only state vector, `scatter!`/`gather!`, `map_blocks!`, the reductions `block_mapreduce` (per block) and `mesh_mapreduce` (one number; its cross-rank step is the rank-order fold of gathered partials in `combine_blocks`), `volume_weighted_norm` |
| regrid | `regrid.jl` | flags → `buffered_flags` → `complete_marks` → rebuild → transfer; `adapt_to_initial_data!` |
| interpolation | `interpolate.jl` | `locate_point` (one binary search) and `interpolate`: a batch of arbitrary points, tensor-product `Lagrange(n)` over one block's stored array, first and second derivatives, periodic wrap and reflecting fold, `exclude` region flags |
| checkpoint | `checkpoint.jl`; `ext/TreeAMRHDF5Ext.jl` | `save_checkpoint`, `load_checkpoint`, `write_plain`/`read_plain`, `checkpoint_environment`: the stubs, docstrings and the load-HDF5 error hint in `src/`, the HDF5 implementation in the extension — serial HDF5 only, over a distributed forest the I/O groups, part files and index of M7 step 6b, the data moved by the communicator verbs |

The ideas that span several files and are easy to violate:

- **Linear octree, leaf-only data.** `forest.leaves` *is* the tree: no node
  objects, no pointers, no coarse data under refined regions. Block `b` of
  any `FieldSet` is `blockkey(fs, b)`, leaf `first(blockrange(forest)) + b
  - 1`: since M7 step 1 every block index is **local to the rank**, while
  tree queries (`nleaves`, `find_leaf`, `locate_point`, `neighbor_keys`)
  stay global. Serially the two coincide, so a site that indexes
  `forest.leaves` by a block index, or sizes a per-block array by
  `nleaves`, passes every serial test and is wrong under MPI; walk
  `blockrange` and size by `nblocks`. Block indices are **not** stable
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
- **Reflecting faces are transfers, not hooks** (M10). `reflecting` is a
  per-face `(lo, hi)` property of the `Forest`, `parity` a per-variable,
  per-dimension property of the `FieldSet` (required when the forest has
  a reflecting face; `NoParity` refused in a reflected dimension). The
  tree does not see the walls — `neighbor_keys` finds nothing across
  them — but `block_sources!` asks `reflect_direction` (in `forest.jl`,
  since it is brick knowledge) for `δ′`, the direction with the masked
  components zeroed, and takes the source that `δ′` finds. Along a
  masked dimension the stencil is the *tangential* one with its target
  rows remapped across the wall (`mirror_rows`), and the kernel
  multiplies by a parity factor from `fs.factors` (column
  `TransferGroup.factorcol`; 0 = an ordinary transfer, left unscaled).
  So mirrored transfers sit in the ordinary phases and run on devices.
  The unowned upper wall plane of a vertex-like dimension is derived by
  `wall_stencil` (odd → 0, even → the folded order-`p` interpolant).
  The boundary hook sees only *outer* faces.
- **Rotating seams are oriented transfers** (M12, CODE.md "Rotating
  seams" under "Ghost filling"). `rotating = (d1, d2)` is a property of
  the `Forest` (a ninth field, `NTuple{2,Int8}` because an `Int` pair
  made the schedule build allocate more; `(0, 0)` for none), `rotation`
  a signed permutation per variable on the `FieldSet` (required on a
  rotating forest; variable `v` beyond the seam is `sign(rotation[v])`
  times variable `abs(rotation[v])` at the preimage). The low faces of
  `d1` and `d2` are glued: `neighbor_anchor` is the one place the seam's
  arithmetic lives, and `oriented_neighbors` returns the real leaves
  with the orientation `r ∈ 0:3` (quarter turns), of which
  `neighbor_keys` returns the keys, so everything built on it sees the
  seam. `balance!` and the checked `leaves` path keep the seam
  **conforming** (equal levels across a seam face), so no coarse-fine
  face crosses it and the interface restriction never does. A turned
  transfer's stencils are built in the *virtual frame*, as if the
  source sat at its turned position (`virtual_offset`, `seam_offset`),
  so the stencil builders are unchanged; `GroupKey` carries the
  orientation (last in `keyorder`) and `TransferGroup` the orientation
  and the plane, and the kernel reads the source through the accessor
  `RotatedSource` (`(src, perm, flip, len, vars, col)`: the axis map
  from virtual to real stored index, a run-time `perm` read through
  `tuplepick`, and the variable from `fs.rotvars`, `nvars × 4`), built
  by `run_group!` only for `orientation ≠ 0` so an ordinary launch is
  unchanged. The sign goes through `fs.factors`, grown to `3^D·4`
  columns, column `mirror + 3^D·r` (`r = 0` is M10's table): a 90°
  turn of any Cartesian component is a signed permutation, so it is
  the permutation on the load and the sign on the target, and nothing
  is fixed up afterwards. A set whose layout is not symmetric under
  `d1 ↔ d2` (and has ghosts in the plane) turns into a partner and is
  filled as a `RotationPair`, whose odd turns read the partner's array
  (`altsrc`) and whose two schedules' stages run merged
  (`exchange_pair!`); a plain fill refuses it. Under MPI the pack
  permutes and the unpack, a plain width-1 copy, applies the sign, which
  keeps the serial `−0`. `interpolate` folds a point beyond the seam
  back and turns its value and gradient; a checkpoint writes `rotating`
  and each set's `rotation`, and lists the `"rotating"` feature only
  when used, so unrotated files still load in 0.1.6. A vertex-like set
  owns both seam planes (the same points under the turn) and evolves
  them at one level.
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
  with 256 lanes per block and no barrier — and ranks can fold their own
  blocks, which M7 does);
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

- **Point interpolation reads one block** (M11, CODE.md "Point
  interpolation"). A query's *stencil* is `n^D` consecutive stored
  points of the block `locate_point` finds, ghosts included, so ghosts
  must be current. The kernel sees a basis only through `stencilwidth`,
  `stencilstart` and `basisweights` — the extension point for smooth
  bases — and `derivs` are multi-indices with only `|m| ≤ 2` accepted
  until higher orders are tested. A derivative's rate is
  `min(n − max mₐ, p − |m|)`: the ghosts carry the exchange's `O(hᵖ)`
  error, so `∂ₓ∂ᵧ` through `Lagrange(4)` with `p = 4` converges at 2,
  not 3. Outside points are reported by the
  host after the launch (no throwing in kernels), and `exclude` flags
  rather than throws. `locate_point` and `isless` share `curve_less`
  in `morton.jl`, so the search and the leaf order cannot disagree.

- **A checkpoint stores what cannot be recomputed** (M9a, CODE.md
  "Checkpoint and restart"): the forest's parameters and its leaves, in
  curve order, and each field set's layout and **owned points only**,
  in state-vector layout. Ghosts, shared planes and schedules are the
  application's to rebuild with its own operators and hook. A load
  builds the forest through the validated `leaves` path
  (`Forest(roots; …, leaves)`, which refuses a list that does not tile
  the brick or is not balanced) and the field set through its
  constructor, so nothing read is trusted before they check it.
  Compatibility is the file's `format_version` plus a must-understand
  `features` list, never package versions; each refusal says why and
  points to `checkpoint_environment`. TreeAMR writes only under
  `/TreeAMR.jl` and the application only under its own top-level group,
  leaving the root free for M9b's sidecars. The write goes to
  `path * ".partial"` and is renamed over `path` with
  `Base.Filesystem.rename` — not `mv(…; force = true)`, which on 1.11
  removes the target first — and, under the default `sync = true`, the
  partial file is flushed to stable storage before the rename and its
  directory after (`F_FULLFSYNC` on macOS, `fsync` elsewhere), since
  closing a file ends in the page cache and a rename can reach the disk
  before the data it names. Element types are HDF5 natives or *limbs*
  (`Float32x2` as two `Float32`), named as a Base-only module prints
  them and matched against the loader's `types`. No Julia type
  definition reaches the file, so a converter can read an old file
  without the old package; do not add JLD2 or `Serialization`. Every
  file carries CRC-32C checksums — the leaf list's (`leaves_crc32c`)
  and one per block of each field set (`data_crc32c`), and the index
  one per part and field set over those — verified on load; HDF5's own
  Fletcher-32 is deliberately not used (it accepts an all-zero chunk).
- **No file of a checkpoint has two writers or two openers** (M7 step
  6b, CODE.md "Checkpoints without parallel I/O"). A shared file
  written from several nodes through MPI-IO lost data on Symmetry's
  BeeGFS (step 6), so HDF5 is used serially only and the data travel
  as messages: the ranks form `k` contiguous I/O groups (`io = :node`,
  the default, by `commnodes`; `:all`; or a number), each group's first
  rank writes a part file `path.<saveid>.<j>.h5` from its members'
  streamed blocks, and rank 0 writes the index (format version 2) last;
  its rename is the commit point, after which rank 0 removes the
  previous parts and orphans. Loading, only rank 0 opens the index and
  broadcasts an in-memory image of it without the data (`bcast`), each
  part is read by one rank, which sends the blocks to their owners, and
  every step that runs on some ranks only ends in `agree_errors`, so a
  failure anywhere throws everywhere and nobody waits on a message. The
  index's external links to the parts are for tools; the loader must
  never follow them (that opens the target). One part lives inside the
  index, so a serial checkpoint is one file; version-1 files (the
  fixtures in `test/fixtures/`) still load. Do not reintroduce parallel
  HDF5 or MPI-IO.
- **The forest is replicated, the blocks are distributed** (M7, CODE.md
  "Distributed meshes"). Every rank holds `forest.leaves` whole; rank
  `r` stores the contiguous curve range `blockrange(forest)`
  (`equalsplit`, the arithmetic `threadchunks` uses one level down), and
  every block index of a field set is local (the first bullet). Every
  forest mutation is **collective**: the same call with the same
  arguments on every rank, which is what keeps the copies equal without
  a message, since every host pass is deterministic. `GhostSchedule`,
  `InterfaceSchedule` and `regrid!` check it through
  `collective_checks` in `forest.jl` (`interpolate` and the checkpoint
  functions through the same `ForestDigest` and `digest_verdict`): one
  `allgather` of a
  `ForestDigest` (generation, `nleaves`, a fold of `hash(key, h)` over
  every leaf — not `hash(leaves)`, which samples a long vector — the
  brick, a layout hash and a refusal flag), so a diverged forest, a
  layout that differs, or an `ArgumentError` on some ranks is refused on
  all of them together. `fill_ghosts!`'s checks stay rank-local on
  purpose (a collective per fill would sit on the per-evaluation path);
  a one-rank refusal there is a hang, never wrong data. Serially nothing
  is gathered. Any new check that can fire on some ranks only goes
  through `collective_checks`.
- **The sender computes; messages are stages of packed buffers.** A
  transfer whose source and target live on different ranks is evaluated
  on the source's rank by the one kernel into a packed buffer (a
  `NamedTuple` `(buf, offsets, dims)` that `transfer_kernel!` reaches
  through three accessor methods), sent, and unpacked on the target's
  rank by a width-1, weight-1 transfer that carries the mirrored
  group's **parity factor**: applied when packing, the factor would
  turn the serial fill's `−0` into `+0`. Every ordering point of the
  serial fill is a stage with its own tag — phase 1 (1), each phase-2
  target level (`2 + ℓ`), each interface dimension (`40 + d`), the
  regrid (50) — run by `run_stage!` in five steps (post receives, pack
  and synchronize, send, run the local groups, wait and unpack). Both
  ends derive a message's layout from the replicated forest, sorted by
  `keyorder`, so no descriptor is ever sent and `Dict` order must never
  reach a layout. Device buffers are staged through page-locked host
  mirrors unless `communicator(comm; deviceaware = true)` (`hoststaging`).
  Stage buffers and mirrors are **leased from the forest's buffer pool**
  (`bufferpool(forest)`, in `forest.state` beside the generation): the
  regrid stage releases its leases after its sends are waited on, and a
  schedule's are reclaimed once the generation has moved on, so a
  regrid reuses what earlier stages held instead of allocating and
  page-locking it again. A lease is an `Array` (`Base.wrap` over pooled
  `Memory`) or a contiguous device `view`, never a `SubArray` of host
  memory. Do not add a field to `Forest`: put mutable state in
  `ForestState` (a ninth field made the schedule build allocate more;
  M12's `rotating` is one only because two `Int8`s fit in padding, and
  `bench/ghosts.jl` measured it at zero bytes).
  MPI is called from the calling task only, never in a threaded loop or
  kernel, and needs `THREAD_SERIALIZED`.
- **Reductions are an allgather of per-rank partials**, folded in rank
  order on every rank (`combine_blocks` in `state.jl`, the only site),
  not an `Allreduce`: the association is the package's, so every rank
  gets the same bits, any `op` and `isbits` partial works (MPI.jl
  refuses custom operators off Intel), and one rank is exactly serial. An
  empty rank contributes nothing, not `init`. `block_mapreduce` and
  `firing_boxes` stay per local block; a number combined from them is
  rank-local, which is the downstream hazard the step-9 audit found
  everywhere (CODE.md, M7 step 9).
- **MPI is a weak dependency** (`[weakdeps]`, `[compat]` 0.20), like
  HDF5: `src/` never names MPI and never branches on it — a serial run
  takes the distributed code path with every message empty, and the
  serial suite is its test. Do not make MPI a hard dependency, do not
  call MPI from `src/`, and do not use `MPI.Comm_dup` (MPI.jl's attaches
  a finalizer that frees collectively at different moments on each
  rank; the extension calls `MPI.API.MPI_Comm_dup` and caches one
  duplicate per communicator, checked with `MPI_Comm_compare`).

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
`ghost_tests.jl`, `centering_tests.jl`, `reflect_tests.jl` (M10),
`rotate_tests.jl` (M12),
`interpolate_tests.jl` (M11), `interface_tests.jl`, the four in-process
M7 files `partition_tests.jl`, `exchange_tests.jl`,
`regrid_exchange_tests.jl` and `interpolate_exchange_tests.jl`,
`allvariables_tests.jl`, `state_tests.jl`, `regrid_tests.jl`, `wave_tests.jl`,
`wave_cell_tests.jl`, `burgers_tests.jl`, `imex_tests.jl`, `type_tests.jl`,
`checkpoint_tests.jl` (M9a), `thread_tests.jl`, `mpi_tests.jl` (M7) and
`gpu_tests.jl` (M2–M8).

The M7 tests come in two kinds. **In process**, simulated ranks over
test-only communicators, each answering only the verbs its test needs:
`PartitionCommunicator(rank, size)` (`partition_tests.jl`: rank and size,
and the digest gather by replication) for the partition and each rank's
local schedule; `exchange_tests.jl`'s lockstep, which builds every
simulated rank's stages and wires their buffers directly, for the
classification, the layouts, write-once and the bitwise pack/unpack
round trip (the `−0` parity case included), and its
`MailboxCommunicator`, one task per rank with a channel per message,
for the staged driver itself; `regrid_exchange_tests.jl`'s
`GatherCommunicator` (collectives as rendezvous between the tasks, the
mailbox for messages) for `regrid!`, `adapt_to_initial_data!`, the
per-rank buffer search and the refusals, and its `StagingCommunicator`
for the host-mirror path; `interpolate_exchange_tests.jl` over the same
for routed interpolation. **Over MPI**, `mpi_tests.jl` runs the
standalone `mpi_workload.jl` serially in process (the reference) and
under `MPI.mpiexec()` at `-n 3` and `-n 2`, one thread a rank, and
requires every line byte for byte except the `sum` lines (`rtol =
1e-12`); `#` lines depend on the rank count and carry the refusals,
the checkpoint cross loads, the migrations and the negative control,
each asserted on its own. The launches are compilation-bound (about
55 s each), so `test/mpi_jobs.jl` starts both at the start of the
suite where the machine has room (`concurrent_launches`: 8+ threads and
24+ GB, or `TREEAMR_TEST_MPI_CONCURRENT=0/1`), and runs them beside the
serial reference otherwise; the workload's `TREEAMR_CHECKPOINT_FROM`
and its marker files order the cross loads either way. Keep
`mpi_workload.jl` self-contained like `thread_workload.jl`, its
non-`#` output independent of the rank count, and a new case's lines
deterministic. `mpi_device_workload.jl` and `mpi_device_tests.jl` are
its device counterpart, run by hand (see Commands), never by
`Pkg.test`.
`checkpoint_tests.jl` checks a bitwise round trip over every centering,
`Float64`/`Float32`/`Float32x2` and every face kind, restarts of the
wave and Burgers studies that continue byte for byte through regrids,
the refusals with their reasons, the checksums (a damaged block, a
recomputed one, missing ones), the atomic write and a failing I/O
process, plain data and `checkpoint_environment`, and that the
version-1 files in `test/fixtures/` (written by the step-6 writer)
still load; the leaf-list `Forest` it loads through is tested inline in
`runtests.jl`, against the oracles. The part files themselves are
`mpi_workload.jl`'s: saves with `io = :all`, `2` and `:node`, a part
from another save, a missing part, orphans, a failed I/O process, and
the `OPEN_LOG` hook showing that no file is opened by two processes. The wave
study comes in two halves: `wave_tests.jl` is the **vertex-centered**
one (M8a), and `wave_cell_tests.jl` is the M3 cell-centered study kept
verbatim so its numbers stay under test. `imex_tests.jl` runs the wave
and Burgers studies a second time through IMEXRungeKutta's explicit
`RK4` and `SSPRK33`, by owner, with a `state_partition` helper built
from `threadchunks`; it also asserts that a stage limiter's correction
never reaches the state (the drift of a conserved total is the step
limiter's injection) and that OrdinaryDiffEq's Shu–Osher SSPRK33 is
different there. Its names clash with OrdinaryDiffEq's (`RK4`,
`SSPRK33`), so it uses `import IMEXRungeKutta as IRK`. Seven helper
files are not tests:

- `oracles.jl`, `ghost_oracles.jl` — deliberately naive, independent
  reference implementations (bit-plane Morton comparison, exact `Rational`
  box geometry, analytic polynomials, Gauss–Legendre cell averages). Property
  tests compare the package against *these*, never against the package's own
  neighbor search or geometry. Keep that independence when adding oracles.
  The M10 oracles live at the end of `ghost_oracles.jl`: `faces_forest`
  (three levels against the walls), `parity_data` (polynomials of
  definite parity), `undefined_ghosts` (the `NaN`-prefill check that
  catches a ghost read before it is written) and `reflecting_vs_doubled`
  (a half domain against the doubled domain it folds). The M12 oracles
  follow them: `unfolded_forest` (the full plane built from the
  quadrant's leaves turned in exact `Rational` boxes),
  `seam_neighbor_mismatches` and `seam_conforming` against it;
  `rotating_data` (a scalar and a vector covariant under the turn and
  under no mirror, smooth or polynomial, its constants in the argument's
  real type so a `Float32` device evaluates it), `rotating_forest`,
  `undefined_rotated_ghosts`, `rotating_vs_quadrupled` and
  `rotating_pair_vs_quadrupled` (the quadrant against the full plane
  `quadrant_and_full` builds, every stored point), `hook_regions_leave`,
  and the regrid oracle `rotating_regrid_vs_quadrupled`. Beyond those,
  `rotate_tests.jl` holds the rigid-rotation advection (conservation
  through the seam), interpolation beyond it, and the wave on a quadrant
  against the full plane (`rotating_wave`, scalar and vector, which the
  vertex-centered seam planes keep equal bit for bit).
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
  way, and keep everything it prints deterministic. Its cycles cover
  periodic and outer boxes, both centerings, reflecting walls (`…r`),
  rotating quadrants with a `RotationPair` fill (`…q`, M12) and the
  conservative Burgers cycle (`B…`).
- `mpi_workload.jl` — the M7 counterpart, standalone too: the argument
  `mpi` makes its forests distributed over `MPI.COMM_WORLD`, and rank 0
  prints the digests gathered in block order. `mpi_tests.jl` also runs
  it in process, in a module of its own, printing into the `IOBuffer`
  it defines as `WORKLOAD_IO`.
- `mpi_jobs.jl` — the launcher of those `mpiexec` jobs
  (`launch_workload`, `finish_workload`, `concurrent_launches`), which
  `runtests.jl` includes before the first test so that the jobs can
  start early.

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
  only. **Registered in General** since 2026-09-21; 0.1.6 is the current
  release. TagBot (`.github/workflows/TagBot.yml`) creates the tag and the
  GitHub release for each registered version, and needs the write deploy
  key behind `DOCUMENTER_KEY` to push them — the file says why. All
  three downstreams bound TreeAMR by `[compat]` over the `0.1` series, so a
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
  floor to 1.10 (raised back to 1.11 on 2026-09-25 with all the Tree*
  packages). So neither a push to `main` nor an uncommitted change
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
- **Checkpointing reaches it with 0.1.4**: `using HDF5`
  beside TreeAMR loads the extension, and nothing else is needed. It
  calls none of the checkpoint functions yet. Against the M9a checkout
  its suite is unchanged, 310 tests in 1m12 (2026-09-29).
- **M7 reaches it with 0.1.5 and changes nothing serially**:
  310 tests in 1m22 against the M7 checkout (2026-10-02); M12 changes
  nothing either, 310 tests in 1m17 against the M12 checkout
  (2026-10-03). It does not
  pass `comm` to any `Forest` yet, so under `mpiexec` each rank would
  run a whole serial copy. Once it does, the step-9 audit (CODE.md, M7
  step 9) found what would go wrong: `field_scales`
  (`src/refinement.jl:112`) takes `maximum(block_mapreduce(…))`, a
  per-rank scale, so the mesh would depend on the rank count and an
  empty rank would throw; `blast_coverage` (`src/blast.jl:281–295`) and
  `track_pulse`'s tracking measure (`src/supergaussian.jl:188–195`) are
  rank-local; `bin/`'s viewers would draw one rank's blocks. The fix is
  `mesh_mapreduce`. It indexes nothing by a global `b` and sizes
  nothing by `nleaves`.
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
`[sources]` entry for TreeAMR, only `TreeAMR = "0.1.3"` under `[compat]`.
A change here reaches its tests only once it is tagged and registered,
a higher bar than a push; to try one sooner, `Pkg.develop` this checkout
into a scratch copy of TreeHydro, never the real one. Its suite is much
longer than TreeWave's — 11893 tests in 4m16 at one thread against the
M9a checkout (2026-09-29), against TreeWave's 310 in 1m12 — so TreeWave
stays the cheap downstream check and this is the thorough one. It is
worth the minutes for anything that touches the exchange, the interface
restriction or the operators, because it is the only place conservation
at coarse-fine faces is exercised by a real scheme rather than by
Burgers in `test/`.

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

**Checkpointing (M9a) is what its long runs were waiting for**, and it
reaches TreeHydro with 0.1.4, through `using HDF5`. The plan (2026-09-29) saves at the start of a chunk, *after*
the regrid and the atmosphere reset — its observer fires before the
regrid, so saving there would mean replaying both — with the chunk
index, the case recipe as Rationals, the `evolve!` keywords and the run
histories as plain data; its `src/checkpoint.jl` (`run_state`,
`rotate_checkpoints!`) has since implemented it.

**M7 changes nothing for it serially**: 12447 tests in 5m00 against the
M7 checkout, at one thread (2026-10-02); nor does M12, 12447 tests in
5m08 against the M12 checkout (2026-10-03). It does not pass `comm` to a
`Forest` yet. Once it does, the step-9 audit found it the most exposed
of the three, all through five host combinations of `block_mapreduce`
— `max_signal_speed` (`src/evolution.jl:477`), `floor_hits` (`:505`),
`ghost_floor_hits` (`:578–585`), `indicator_scales`
(`src/refinement.jl:189–190`), `peak_compression` (`src/sedov.jl:512`).
The first is the CFL speed, and `evolve!` takes each chunk's step count
from it (`src/driver.jl:857–860`), so the ranks would take different
numbers of steps and hang in the exchange. Its wall-clock checkpoint
triggers (`src/driver.jl:910–917`) decide per rank whether to enter the
collective `save_checkpoint`, and its run state carries the rank-local
counts and speeds, which the plain-data agreement would refuse on every
rank. The diagnostics (`tracked_share`, `reduce_to_grid`,
`mode_amplitude`, `shock_radius`, …) are rank-local, and
`src/kelvinhelmholtz.jl:544` records `nblocks` for the mesh's block
count. The fixes are `mesh_mapreduce`, and an agreed trigger.
Its own MPI port (2026-10-02) found the one TreeAMR bug so far on a
rank without blocks — the `AllVariables` boundary hook's length check,
of which it is the only caller (CODE.md, "Ranks without blocks are
allowed") — so it needs TreeAMR 0.1.6, which has the fix.

Mesh machinery belongs here; physics belongs there — the same rule as
for TreeWave.

## Downstream: TreeGeneralizedHarmonic

`~/src/jl/TreeGeneralizedHarmonic` (github.com/eschnett/TreeGeneralizedHarmonic.jl)
is the third application: the vacuum Einstein equations in the
generalized harmonic formulation, a black hole on the octree. It
pinned TreeAMR to GitHub `main` through `[sources]` until 2026-09-26,
when it retired its stopgap interpolator for M11's `interpolate`; since
then it takes TreeAMR from **General**, `TreeAMR = "0.1.4"` under
`[compat]` since its checkpointing, like the other two, so a change here reaches it only with a
release. That includes M9a: checkpointing reaches it with 0.1.4,
through `using HDF5`, together with TreeAMR's new `__init__`
and the five new exports (none of which clashes with a name of its
own). Its production runs, estimated at 38–149 h, are one of the
reasons M9a went before M7; beyond `(t, u)` its restart stores its
horizon tracking and interior fits, which its `src/checkpoint.jl` does
through 0.1.4. Every
kernel it has goes through `map_blocks!`. Its
`test/prerequisite_tests.jl` names the four unexported TreeAMR names it
relies on — `threadchunks` (its integrator's partition) and M11's
`Region` extension points `inside`, `stencil_hits` and
`stencil_position` — so renaming one breaks that suite at the top; the
stopgap's `threaded_foreach` left with the stopgap. On 2026-09-25 it
measured the ownership policy from the outside, on Symmetry, and found
the integrator's own passes to be what is left: see the last item of
"Open questions" in `CODE.md` (the serial stage updates, the
per-`solve` buffers, a first-touch anomaly that looks like NUMA
balancing, and why not Polyester). Erik decided the same day not to optimise the
OrdinaryDiffEq path further, Polyester included; what is left open there
is where limiters go (Shu–Osher against Butcher form).

**M7 changes nothing for it serially**: its full suite takes about 19
minutes, so step 9 ran a subset against the M7 checkout — `precision_`,
`prerequisite_`, `stencils_`, `stepping_`, `interface_`, `refinement_`,
`horizon_`, `checkpoint_` and `type_tests.jl`, 1760 tests in 7m04 at one
thread, all passing (2026-10-02); against the M12 checkout its
`prerequisite_tests.jl` alone, which names the unexported TreeAMR names
it relies on, passes, 52 tests (2026-10-03). Its production runs are the
reason M7
was wanted, and of the three it is the closest to ready: its reductions
are already `mesh_mapreduce` and its integrator is fixed-step. It does
not pass `comm` yet (`gh_forest`, `src/initialdata.jl`; `load_run`,
`src/checkpoint.jl`). The step-9 audit found, for when it does:
`indicator_flags` hands its local flags to the public `buffered_flags`
(`src/refinement.jl:739`), which TreeAMR now refuses (step 9 added the
length check, so this fails loudly instead of buffering the wrong
leaves) — under MPI the dilation has to come from `regrid!`'s
`buffer`, which it avoids on purpose so that its level ceiling applies
after the dilation; `clamp_marks` (`:762`) and
`refinement_centroid` (`:816`) index `forest.leaves[b]` by a local
block, and the centroid's sums stay per rank; its wall-clock checkpoint
triggers (`src/driver.jl:1287–1293`) decide per rank, and its run state
holds wall-clock fit costs and the per-rank centroids (`:1312`), which
the plain-data agreement refuses; its non-finite check (`:741`) reads
the local state, so one rank would throw alone. Its horizon finder
passing every point on every rank is correct under the collective
`interpolate`, only redundant.
