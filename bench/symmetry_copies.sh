#!/bin/bash
# The copy kernels on an H200 ("The copy kernels on a device" in CODE.md):
# `bench/copies.jl` for a baseline checkout and this one, side by side on
# the same GPU, at the sizes a downstream application measured them at;
# then `bench/copy_index.jl`, which prices each way of forming the index;
# then `bench/copy_groups.jl`, the fill group by group; then the test
# suite on the device.
#
#     TREEAMR_REPO=<this checkout> TREEAMR_BASE=<baseline checkout> \
#         sbatch bench/symmetry_copies.sh
#
# Both checkouts get an environment of their own under scratch that
# develops them, so the two runs differ only in the package.
# `TREEAMR_COPIES_STAGES` picks stages (default: copies index groups
# tests). The job writes a SimWatch status file (`bench/simwatch.sh`) into
# `$SLURM_SUBMIT_DIR/$SLURM_JOB_NAME-$SLURM_JOB_ID`.

#SBATCH --partition=h200debugq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --job-name=treeamr-copies
#SBATCH --output=treeamr-copies-%j.out

set -euo pipefail

export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEAMR_REPO:-$PWD}"
BASE="${TREEAMR_BASE:-}"
SCRATCH="${TREEAMR_COPIES_ENV:-/mnt/beegfs/eschnetter/claude/treeamr-copies-env}"
# The MPI test launches mpiexec inside the allocation; Hydra forks the
# ranks locally rather than asking SLURM for a step.
export HYDRA_BOOTSTRAP=fork
STAGES="${TREEAMR_COPIES_STAGES:-copies index groups tests}"
want() { [[ " $STAGES " == *" $1 "* ]]; }

. "$REPO/bench/simwatch.sh"
nstages=1
for st in $STAGES; do nstages=$((nstages + 1)); done
simwatch_begin "${SLURM_SUBMIT_DIR:-$PWD}/${SLURM_JOB_NAME:-copies}-${SLURM_JOB_ID:-local}" \
    "copy kernels on an H200" "$nstages"

# An environment that develops checkout `$1` and has CUDA. The test
# environment's dependencies too, from its own Project.toml with the
# relative TreeAMR source made absolute, so that the suite runs here.
setup() {
    local repo="$1" env="$2"
    mkdir -p "$env"
    sed "s|TreeAMR = {path = \"..\"}|TreeAMR = {path = \"$repo\"}|" \
        "$repo/test/Project.toml" > "$env/Project.toml"
    rm -f "$env/Manifest.toml"
    julia --project="$env" -e '
        using Pkg
        Pkg.add(["CUDA"])
        Pkg.instantiate()
        using CUDA; CUDA.versioninfo()'
}

simwatch_stage "setting up environments"
setup "$REPO" "$SCRATCH/new"
[ -n "$BASE" ] && want copies && setup "$BASE" "$SCRATCH/base"

# The downstream's rows: 512 blocks of 16³ and of 32³, 8 of 128³; 20
# variables, G = 3, vertex-centered, Float64.
sizes=("16 8" "32 8" "128 2")
want copies && simwatch_stage "copies, baseline and new"
want copies && for which in base new; do
    [ "$which" = base ] && [ -z "$BASE" ] && continue
    repo=$([ "$which" = base ] && echo "$BASE" || echo "$REPO")
    echo "=== copies, $which ($repo, $(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo '?')) ==="
    for s in "${sizes[@]}"; do
        set -- $s
        TREEAMR_BENCH_BACKEND=cuda TREEAMR_BENCH_N=$1 TREEAMR_BENCH_ROOTS=$2 \
            julia --project="$SCRATCH/$which" "$REPO/bench/copies.jl"
    done
    echo "=== copies, $which, two-level, 32³ ==="
    TREEAMR_BENCH_BACKEND=cuda TREEAMR_BENCH_N=32 TREEAMR_BENCH_ROOTS=8 \
        TREEAMR_BENCH_LEVELS=2 julia --project="$SCRATCH/$which" "$REPO/bench/copies.jl"
done

want index && simwatch_stage "index forms"
want index && for n in 16 32; do
    TREEAMR_BENCH_BACKEND=cuda TREEAMR_BENCH_N=$n \
        TREEAMR_BENCH_BLOCKS=512 \
        julia --project="$SCRATCH/new" "$REPO/bench/copy_index.jl"
done

want groups && simwatch_stage "fill by group"
want groups && for s in "${sizes[@]}"; do
    set -- $s
    TREEAMR_BENCH_BACKEND=cuda TREEAMR_BENCH_N=$1 TREEAMR_BENCH_ROOTS=$2 \
        julia --project="$SCRATCH/new" "$REPO/bench/copy_groups.jl"
done

if want tests; then
    simwatch_stage "test suite, CUDA backend"
    TREEAMR_TEST_BACKEND=cuda TREEAMR_TEST_MPI_CONCURRENT=0 \
        julia --project="$SCRATCH/new" --threads=8 "$REPO/test/runtests.jl"
fi
