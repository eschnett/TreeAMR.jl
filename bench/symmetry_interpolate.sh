#!/bin/bash
# The M11 point-interpolation benchmark on Symmetry.
#
#     sbatch bench/symmetry_interpolate.sh cpu                     # an AMD node
#     sbatch --partition=h200debugq --gres=gpu:h200:1 --cpus-per-task=16 \
#            bench/symmetry_interpolate.sh cuda                   # one H200
#
# `cpu` scans the thread count with Julia pinning its own threads
# (`JULIA_EXCLUSIVE=1`, the block-ownership policy placing the working
# array by first touch), then repeats 64 threads unpinned under
# `numactl --interleave=all`, and 8 threads bound to one NUMA domain —
# the three placements CODE.md's NUMA measurements compare. A batch of
# points is not owned by anyone: whichever thread takes a point reads
# whichever block holds it, so this is where remote reads would show.
#
# `cuda` runs Float64 and Float32 on the device, then the same batches
# on the node's host cores for the comparison.
#
# `TREEAMR_STOPGAP=<dir>` adds, in `cpu` mode, the same batches through
# a copy of TreeGeneralizedHarmonic's stopgap interpolator
# (`<dir>/stopgap_bench.jl`), which is what M11 replaces.

#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=treeamr-interp
#SBATCH --output=treeamr-interp-%j.out

set -euo pipefail

MODE="${1:-cpu}"
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEAMR_REPO:-$PWD}"
export TREEAMR_REPO="$REPO"
# A project of its own, under scratch, as in `symmetry_gpu.sh`: the
# repository's environments must not gain a CUDA dependency.
# One per mode, so that a CPU and a GPU job can run at once.
ENVDIR="${TREEAMR_INTERP_ENV:-/mnt/beegfs/eschnetter/claude/treeamr-interp-$MODE}"
PKGS='"KernelAbstractions", "StaticArrays"'
[ "$MODE" = cuda ] && PKGS="$PKGS, \"CUDA\""

mkdir -p "$ENVDIR"
julia --project="$ENVDIR" -e "
    using Pkg
    Pkg.develop(path = \"$REPO\")
    for p in ($PKGS)
        p in keys(Pkg.project().dependencies) || Pkg.add(p)
    end
    Pkg.instantiate()
    Pkg.precompile()"

# A command, not a shell function, so that `srun` and `numactl` can run it.
JL=(julia --project="$ENVDIR")

echo "# $(hostname) $(date -Iseconds) mode=$MODE"
if [ "$MODE" = cpu ]; then
    lscpu | grep -E 'Model name|NUMA node\(s\)'
    for t in 1 8 16 32 64; do
        JULIA_EXCLUSIVE=1 srun --cpu-bind=none "${JL[@]}" -t "$t" "$REPO/bench/interpolate.jl"
    done
    echo "# 64 threads unpinned, numactl --interleave=all"
    srun --cpu-bind=none numactl --interleave=all "${JL[@]}" -t 64 "$REPO/bench/interpolate.jl"
    echo "# 8 threads bound to NUMA domain 0, memory there too"
    srun --cpu-bind=none numactl --cpunodebind=0 --membind=0 \
        "${JL[@]}" -t 8 "$REPO/bench/interpolate.jl"
    if [ -n "${TREEAMR_STOPGAP:-}" ]; then
        for t in 1 8 64; do
            JULIA_EXCLUSIVE=1 srun --cpu-bind=none "${JL[@]}" -t "$t" \
                "$TREEAMR_STOPGAP/stopgap_bench.jl"
        done
    fi
elif [ "$MODE" = cuda ]; then
    "${JL[@]}" -e 'using CUDA; CUDA.versioninfo()'
    for T in Float64 Float32; do
        TREEAMR_BENCH_BACKEND=cuda TREEAMR_BENCH_T=$T \
            TREEAMR_BENCH_NPTS=496,4960,49600,496000 "${JL[@]}" "$REPO/bench/interpolate.jl"
    done
    echo "# the same batches on this node's host cores"
    JULIA_EXCLUSIVE=1 "${JL[@]}" -t "${SLURM_CPUS_PER_TASK:-16}" "$REPO/bench/interpolate.jl"
else
    echo "mode must be cpu or cuda, got $MODE" >&2
    exit 1
fi
