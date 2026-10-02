#!/usr/bin/env bash
# Run bench/mpi.jl at a range of rank counts and print the weak-scaling
# table (bench/mpitable.awk). Usage:
#
#     bench/mpiscan.sh [rank counts...]          (default 1 2 4)
#
# Each count runs under the launcher of MPI.jl's binary for the project
# (`MPI.mpiexec()`, with its environment, as test/mpi_tests.jl does), at
# TREEAMR_BENCH_THREADS threads a rank (default 1). The other
# TREEAMR_BENCH_* variables pass through to bench/mpi.jl, so
#
#     TREEAMR_BENCH_N=8 TREEAMR_BENCH_THREADS=2 bench/mpiscan.sh 1 2 4
#
# is a laptop-sized scan. TREEAMR_BENCH_PROJECT chooses the environment
# (default test, which has MPI); on a cluster, bench/symmetry_mpi.sh
# launches through srun instead and uses the same table.
set -euo pipefail
cd "$(dirname "$0")/.."
project=${TREEAMR_BENCH_PROJECT:-test}
threads=${TREEAMR_BENCH_THREADS:-1}

counts=("$@")
if [ ${#counts[@]} -eq 0 ]; then
    counts=(1 2 4)
fi

out=$(mktemp)
for p in "${counts[@]}"; do
    echo "# ranks=$p threads=$threads" >&2
    julia --project="$project" -e '
        using MPI
        p, t, project = ARGS
        m = MPI.mpiexec()
        cmd = `$m -n $p $(Base.julia_cmd()) --threads=$t --project=$project bench/mpi.jl mpi`
        run(setenv(cmd, m.env))' "$p" "$threads" "$project" | tee -a "$out" >&2
done

awk -F'\t' -f bench/mpitable.awk "$out"
rm -f "$out"
