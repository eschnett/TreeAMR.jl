#!/bin/bash
# The M6 acceptance run on Symmetry: the test suite against CUDA, in
# Float64 and Float32, followed by the kernel benchmarks.
#
# `CODE.md` asks M6 for "M3 convergence results reproduced on GPU;
# kernel benchmarks", and it names CUDA. A Mac's Metal backend answers
# the Float32 half of that locally and quickly, which is what makes it
# the development target — but it has no hardware fp64 at all, so the
# Float64 claim can only be made here.
#
#     sbatch bench/symmetry_gpu.sh
#     sbatch --partition=h200q --time=4:00:00 bench/symmetry_gpu.sh   # longer
#
# h200debugq holds 2 nodes of 8x H200 with a 1-hour limit, which is more
# than the suite needs; only one GPU is used.

#SBATCH --partition=h200debugq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --job-name=treeamr-gpu
#SBATCH --output=treeamr-gpu-%j.out

set -euo pipefail

export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEAMR_REPO:-$PWD}"
# A project of its own, under scratch: the repository's own test
# environment must not gain a CUDA dependency, since CI has no GPU and
# would then install it on every run for nothing.
ENVDIR="${TREEAMR_GPU_ENV:-/mnt/beegfs/eschnetter/claude/treeamr-gpu}"

mkdir -p "$ENVDIR"
julia --project="$ENVDIR" -e "
    using Pkg
    Pkg.develop(path = \"$REPO\")
    for p in (\"CUDA\", \"KernelAbstractions\", \"MultiFloats\",
              \"OrdinaryDiffEqLowOrderRK\", \"OrdinaryDiffEqSSPRK\", \"SciMLBase\",
              \"Test\", \"Random\", \"SHA\", \"Printf\")
        p in keys(Pkg.project().dependencies) || Pkg.add(p)
    end
    Pkg.instantiate()
    using CUDA; CUDA.versioninfo()"

echo "=== test suite, CUDA backend ==="
# The suite's own Float64/Float32 sweep runs on the device: `gpu_tests.jl`
# asks the backend which types it supports, and an H200 supports both.
TREEAMR_TEST_BACKEND=cuda \
    julia --project="$ENVDIR" --threads=8 "$REPO/test/runtests.jl"

echo "=== kernel benchmarks ==="
for T in Float64 Float32; do
    TREEAMR_BENCH_BACKEND=cuda TREEAMR_BENCH_T=$T \
        TREEAMR_BENCH_N=32 TREEAMR_BENCH_ROOTS=8 \
        julia --project="$ENVDIR" "$REPO/bench/gpu.jl"
done

echo "=== the same benchmark on the host, for the comparison ==="
# Same sizes, same phases, so the two outputs read side by side. Pages
# interleaved, which M5 measured as worth 2-6x on this node type and
# which the library cannot set for itself.
for T in Float64 Float32; do
    TREEAMR_BENCH_BACKEND=cpu TREEAMR_BENCH_T=$T \
        TREEAMR_BENCH_N=32 TREEAMR_BENCH_ROOTS=8 \
        numactl --interleave=all \
        julia --project="$ENVDIR" --threads="$SLURM_CPUS_PER_TASK" "$REPO/bench/gpu.jl"
done
