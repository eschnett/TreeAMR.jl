#!/bin/bash
# Bounding the gain of NUMA-local data placement on a Symmetry AMD node,
# without writing any placement code.
#
# The M5 table in CODE.md has the memory-bound phases at ~36x on 64
# cores with pages interleaved. The question is how much of the rest a
# domain-local layout — one MPI rank per NUMA domain, or the same
# ownership imitated with pinned thread groups in one process — could
# recover. Eight independent copies of the benchmark, each bound to one
# NUMA domain with its own memory and holding an eighth of the blocks,
# have no cross-domain traffic at all, so their throughput is an upper
# bound on any such scheme. The interleaved 64-thread run over all the
# blocks is what we have today. Two socket-bound copies sit in between.
#
#     sbatch bench/symmetry_numa.sh
#
# Every run prints one tab-separated line per phase (see
# bench/threads.jl); the outputs land in $OUTDIR and are also echoed at
# the end of the job log. Compare the per-evaluation lines (rhs,
# fill_ghosts, scatter) per cell; the regrid-frequency lines
# (ghost_schedule, complete_marks) are serial-tailed by construction and
# are not what this measures.

#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=treeamr-numa
#SBATCH --output=treeamr-numa-%j.out

set -euo pipefail

export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEAMR_REPO:-$PWD}"
OUTDIR="${TREEAMR_NUMA_OUT:-/mnt/beegfs/eschnetter/claude/numa-$SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
cd "$REPO"

echo "=== node ==="
hostname
lscpu | grep -E 'Model name|^Socket|^NUMA|Thread\(s\) per core|L3'
numactl -H

echo "=== environment ==="
julia --version
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# The same block size and refinement pattern as the M5 measurement:
# N = 32, the middle eighth refined once. ROOTS = 8 gives 960 blocks;
# ROOTS = 4 gives 120, so eight of those hold the same cells. No
# JULIA_EXCLUSIVE here: it would pin every copy's threads to the same
# first cores, fighting numactl's binding.
export TREEAMR_BENCH_D=3 TREEAMR_BENCH_N=32 TREEAMR_BENCH_REPS=20

run() {  # run <label> <roots> <threads> [numactl args...]
    local label=$1 roots=$2 threads=$3; shift 3
    echo "--- $label: roots=$roots threads=$threads numactl: ${*:-(none)}"
    local prefix=()
    [ $# -gt 0 ] && prefix=(numactl "$@")
    TREEAMR_BENCH_ROOTS=$roots "${prefix[@]}" \
        julia -t "$threads" --project=. bench/threads.jl > "$OUTDIR/$label.tsv" 2>&1
}

echo "=== warm-up (compiles the benchmark path once into the cache) ==="
TREEAMR_BENCH_N=8 TREEAMR_BENCH_ROOTS=2 TREEAMR_BENCH_REPS=1 \
    julia -t 8 --project=. bench/threads.jl > /dev/null

echo "=== A: 64 threads, all blocks, pages interleaved (today's way to run) ==="
run interleave64 8 64 --interleave=all

echo "=== B: 64 threads, all blocks, first touch (the M5 left-hand column) ==="
run firsttouch64 8 64

echo "=== C: one domain alone, an eighth of the blocks, bound to its own memory ==="
run domain0alone 4 8 --cpunodebind=0 --membind=0

echo "=== D: eight domain-bound copies at once, an eighth of the blocks each ==="
# The upper bound: no cross-domain traffic anywhere on the node.
for i in 0 1 2 3 4 5 6 7; do
    run "domain$i" 4 8 --cpunodebind=$i --membind=$i &
done
wait

echo "=== E: two socket-bound copies at once, interleaved within the socket ==="
# In between: cross-domain but not cross-socket traffic. Node ranges
# assume nodes 0-3 are socket 0 and 4-7 socket 1; numactl -H above
# says whether that holds. ROOTS = 6 gives 405 blocks per copy; compare
# per cell.
run socket0 6 32 --cpunodebind=0-3 --interleave=0-3 &
run socket1 6 32 --cpunodebind=4-7 --interleave=4-7 &
wait

echo "=== F: the same 8 threads on domain 0, memory placed elsewhere ==="
# Separates placement from process size: same thread count and
# scheduling as C, only the pages move. Interleaved over the node,
# on a same-socket domain, on a cross-socket domain.
run domain0interleaved 4 8 --cpunodebind=0 --interleave=all
run domain0samesocket  4 8 --cpunodebind=0 --membind=1
run domain0crosssocket 4 8 --cpunodebind=0 --membind=4

echo "=== G: 16 threads on two domains, interleaved over those two ==="
# With E (32 on four) and C (8 on one): process size at local placement.
run domains01 5 16 --cpunodebind=0-1 --interleave=0-1

echo "=== H: one process, all blocks, pages interleaved, thread count varied ==="
# The in-process scaling curve at fixed placement and fixed problem.
for t in 16 32 48; do
    run "interleave$t" 8 $t --interleave=all
done

echo "=== I: 64 threads interleaved, runtime variants ==="
# Same as A but for one runtime knob each: a single GC thread, threads
# pinned, and no interactive thread (the last needs Julia 1.12+ syntax
# and is allowed to fail).
TREEAMR_BENCH_ROOTS=8 numactl --interleave=all \
    julia -t 64 --gcthreads=1 --project=. bench/threads.jl > "$OUTDIR/gcthreads1.tsv" 2>&1
TREEAMR_BENCH_ROOTS=8 JULIA_EXCLUSIVE=1 numactl --interleave=all \
    julia -t 64 --project=. bench/threads.jl > "$OUTDIR/exclusive64.tsv" 2>&1
TREEAMR_BENCH_ROOTS=8 numactl --interleave=all \
    julia -t 64,0 --project=. bench/threads.jl > "$OUTDIR/nointeractive64.tsv" 2>&1 || true

echo "=== J: is it the placement or the process? ==="
# Eight 8-thread copies again, but with each copy's pages interleaved
# over the whole node: the same non-local placement as A with the
# process size of D. If they match D, one big process is the problem;
# if they match A, non-local placement is. The kernel's NUMA balancing
# and huge-page settings, and its NUMA counters around the run, say
# whether page migration is in play.
cat /proc/sys/kernel/numa_balancing /sys/kernel/mm/transparent_hugepage/enabled || true
grep -E "numa_(hint_faults|pages_migrated|pte_updates)" /proc/vmstat
for i in 0 1 2 3 4 5 6 7; do
    run "spread$i" 4 8 --cpunodebind=$i --interleave=all &
done
wait
grep -E "numa_(hint_faults|pages_migrated|pte_updates)" /proc/vmstat
echo "--- and the counters around A itself"
run interleave64again 8 64 --interleave=all
grep -E "numa_(hint_faults|pages_migrated|pte_updates)" /proc/vmstat
for i in 0 1 2 3; do
    run "spread16_$i" 5 16 --cpunodebind=$((2*i)),$((2*i+1)) --interleave=all &
done
wait

echo "=== K: locating the single-process limit ==="
# bench/stream.jl: the same stream through four launch mechanisms.
# One 64-thread process interleaved, pinned and not; one 8-thread
# process over the same 717 MB arrays (footprint without the threads);
# eight 8-thread processes over an eighth each (the D/J shape).
BIG=$((960 * 46656 * 2)); SMALL=$((BIG / 8))
TREEAMR_STREAM_N=$BIG numactl --interleave=all \
    julia -t 64 --project=. bench/stream.jl > "$OUTDIR/stream64.txt" 2>&1
TREEAMR_STREAM_N=$BIG JULIA_EXCLUSIVE=1 numactl --interleave=all \
    julia -t 64 --project=. bench/stream.jl > "$OUTDIR/stream64pinned.txt" 2>&1
TREEAMR_STREAM_N=$BIG numactl --cpunodebind=0 --interleave=all \
    julia -t 8 --project=. bench/stream.jl > "$OUTDIR/stream8big.txt" 2>&1
for i in 0 1 2 3 4 5 6 7; do
    TREEAMR_STREAM_N=$SMALL numactl --cpunodebind=$i --interleave=all \
        julia -t 8 --project=. bench/stream.jl > "$OUTDIR/stream8x8_$i.txt" 2>&1 &
done
wait
# And the mesh benchmark itself at 8 threads over all 960 blocks:
# the footprint of A with the thread count of C.
run domain0big 8 8 --cpunodebind=0 --interleave=all

echo "=== results ==="
for f in "$OUTDIR"/*.tsv; do
    echo "## $(basename "$f" .tsv)"
    cat "$f"
done
