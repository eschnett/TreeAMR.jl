#!/usr/bin/env bash
# Run bench/threads.jl at a range of thread counts and print a speedup
# table. Usage:  bench/scan.sh [thread counts...]   (default 1 2 4 ... 128)
#
# Environment variables are passed through to bench/threads.jl, so
#
#     TREEAMR_BENCH_N=32 bench/scan.sh 1 8 64
#
# scans a realistic block size at three thread counts.
# TREEAMR_BENCH_SCRIPT and TREEAMR_BENCH_PROJECT choose another script
# in the same output format, e.g. bench/stepping.jl with project test.
set -euo pipefail
cd "$(dirname "$0")/.."
script=${TREEAMR_BENCH_SCRIPT:-bench/threads.jl}
project=${TREEAMR_BENCH_PROJECT:-.}

counts=("$@")
if [ ${#counts[@]} -eq 0 ]; then
    counts=(1 2 4 8 16 32 64 128)
fi

out=$(mktemp)
for t in "${counts[@]}"; do
    echo "# threads=$t" >&2
    julia -t "$t" --project="$project" "$script" >>"$out"
done

awk -F'\t' '
  /^threads=/ { header = $0; next }
  NF == 3 {
    key = $2 SUBSEP $1
    sec[key] = $3
    if (!seen[$2]++) phase[++np] = $2
    if (!tseen[$1]++) tc[++nt] = $1
  }
  END {
    print header
    printf "%-30s", "phase"
    for (i = 1; i <= nt; i++) printf "%10s", tc[i] "t"
    printf "   |"
    for (i = 1; i <= nt; i++) printf "%10s", tc[i] "t"
    printf "\n"
    printf "%-30s%*s   |%*s\n", "", nt * 10, "speedup", nt * 10, "seconds"
    for (j = 1; j <= np; j++) {
      p = phase[j]
      printf "%-30s", p
      for (i = 1; i <= nt; i++)
        printf "%10.2f", sec[p SUBSEP tc[1]] / sec[p SUBSEP tc[i]]
      printf "   |"
      for (i = 1; i <= nt; i++) printf "%10.5f", sec[p SUBSEP tc[i]]
      printf "\n"
    }
  }
' "$out"
rm -f "$out"
