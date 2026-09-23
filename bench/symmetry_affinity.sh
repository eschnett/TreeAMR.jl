#!/bin/bash
# The measurements behind "What one process loses: data-to-core
# affinity" in CODE.md, on one Symmetry AMD node (64-core EPYC 7543,
# 8 NUMA domains of one CCD each).
#
#     sbatch bench/symmetry_affinity.sh
#
# Every comparison between processes runs in synchronized wall-clock
# windows (bench/affinity.jl says why). The original runs were jobs
# 562406-562425, and job 562462 measured the implemented ownership
# policy; this script reproduces their comparisons in one job. The
# mesh part needs a checkout with the ownership policy in `src/`.
# The perf and IBS counters quoted in CODE.md need
# `kernel.perf_event_paranoid=-1` on the node (set by hand on cn099) and
# are not collected here.

#SBATCH --partition=amdq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2:00:00
#SBATCH --job-name=treeamr-affinity
#SBATCH --output=treeamr-affinity-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEAMR_REPO:-$PWD}"
cd "$REPO"
hostname
numactl -H | head -3
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

BIG=$((960 * 46656 * 2))        # the size of bench/threads.jl's working array
start() { echo $(( $(date +%s) + $1 )); }

stream64() {  # stream64 <tag> <modes> [env...] -- one 64-thread process, interleaved
    local tag=$1 modes=$2; shift 2
    env TREEAMR_AFFINITY_START=$(start 60) TREEAMR_AFFINITY_TAG=$tag \
        TREEAMR_AFFINITY_MODES=$modes TREEAMR_STREAM_N=$BIG "$@" \
        numactl --interleave=all julia -t 64 --project=. bench/affinity.jl
}
stream8x8() {  # stream8x8 <tag> <modes> <numactl memory policy for domain $i>
    local tag=$1 modes=$2 mem=$3 s
    s=$(start 75)
    for i in 0 1 2 3 4 5 6 7; do
        TREEAMR_AFFINITY_START=$s TREEAMR_AFFINITY_TAG=$tag$i TREEAMR_AFFINITY_MODES=$modes \
            TREEAMR_STREAM_N=$((BIG / 8)) \
            numactl --cpunodebind=$i ${mem//\$i/$i} julia -t 8 --project=. bench/affinity.jl \
            > "/tmp/affinity-$tag$i.txt" 2>&1 &
    done
    wait
    cat /tmp/affinity-$tag?.txt | awk -F'\t' '$4 ~ /GB\/s/ {split($4, a, " "); s[$2] += a[1]}
        END {for (k in s) printf "'$tag' 8x8 sum\t%s\t%.0f GB/s\n", k, s[k]}'
}

echo "=== stream: one 64-thread process, launch paths and chunk maps"
M=static,static_rot,static_scalar,spawn,ka,ka_static,persist,persist_stagger2,static
stream64 u64 $M
stream64 p64 $M,static_alt1,static_alt8,static_alt32 JULIA_EXCLUSIVE=1
echo "=== stream: hysteresis after one rotated window (pinned, interleaved)"
stream64 hyst static,static,static_rot,static,static,static,static JULIA_EXCLUSIVE=1
echo "=== stream: the same, first touch in the static partition (no numactl)"
TREEAMR_AFFINITY_START=$(start 60) TREEAMR_AFFINITY_TAG=ft JULIA_EXCLUSIVE=1 \
    TREEAMR_AFFINITY_MODES=static,static_rot,static,spawn,ka,ka_static TREEAMR_STREAM_N=$BIG \
    julia -t 64 --project=. bench/affinity.jl
echo "=== stream: eight 8-thread processes, synchronized"
stream8x8 i static,static_rot,spawn,ka,ka_static '--interleave=all'
stream8x8 l static,static_rot,spawn,ka,ka_static '--membind=$i'

echo "=== owner: idle latency and bandwidth by previous owner"
JULIA_EXCLUSIVE=1 numactl --membind=0 julia -t 64 --project=. bench/owner.jl

echo "=== mesh: 960 blocks of 32^3, one 64-thread process, by owner and the spawn control"
export TREEAMR_BENCH_D=3 TREEAMR_BENCH_N=32
mesh64() {  # mesh64 <tag> <variant> [env...] [numactl...]
    local tag=$1 v=$2; shift 2
    env TREEAMR_AFFINITY_START=$(start 150) TREEAMR_AFFINITY_TAG=$tag \
        TREEAMR_AFFINITY_VARIANT=$v TREEAMR_BENCH_ROOTS=8 "$@" \
        julia -t 64 --project=. bench/affinity_mesh.jl
}
for v in owner spawn; do
    mesh64 p64 $v JULIA_EXCLUSIVE=1 numactl --interleave=all
    mesh64 u64 $v numactl --interleave=all       # unpinned
done
mesh64 f64 owner JULIA_EXCLUSIVE=1               # first touch
mesh64 uft owner                                 # unpinned, first touch
echo "=== mesh: eight 8-thread processes, 120 blocks each, synchronized"
for mem in '--membind=$i' '--interleave=all'; do
    s=$(start 150)
    for i in 0 1 2 3 4 5 6 7; do
        TREEAMR_AFFINITY_START=$s TREEAMR_AFFINITY_TAG=d$i TREEAMR_BENCH_ROOTS=4 \
            numactl --cpunodebind=$i ${mem//\$i/$i} julia -t 8 --project=. bench/affinity_mesh.jl \
            > /tmp/affinity-mesh$i.txt 2>&1 &
    done
    wait
    echo "--- $mem (slowest copy per phase, seconds per call)"
    cat /tmp/affinity-mesh?.txt | awk -F'\t' 'NF == 7 {if ($5 > m[$3]) m[$3] = $5}
        END {for (k in m) printf "%s\t%.6f\n", k, m[k]}'
done
