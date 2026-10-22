#!/bin/bash
# Folds over points on an H200 ("Folds over points" in CODE.md):
# `bench/fold.jl` at the downstream's sizes, a fold of 126 and of 161
# numbers against the one-number reductions it replaces, then the test
# suite on the device.
#
#     cd /mnt/beegfs/eschnetter/claude
#     TREEAMR_REPO=<checkout> TREEAMR_REV=<commit> sbatch <checkout>/bench/symmetry_fold.sh
#
# The checkout is usually an rsynced copy without `.git`, so the job
# prints `TREEAMR_REV` as the revision it ran. It gets an environment of
# its own under scratch that develops it. `TREEAMR_FOLD_STAGES` picks
# stages (default: fold tests). The job writes a SimWatch status file
# (`bench/simwatch.sh`) into `$SLURM_SUBMIT_DIR/$SLURM_JOB_NAME-$SLURM_JOB_ID`.

#SBATCH --partition=h200debugq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --job-name=treeamr-fold
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treeamr-fold-%j.out

set -euo pipefail

export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEAMR_REPO:-$PWD}"
REV="${TREEAMR_REV:-$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
SCRATCH="${TREEAMR_FOLD_ENV:-/mnt/beegfs/eschnetter/claude/treeamr-fold-env}"
# The MPI test launches mpiexec inside the allocation; Hydra forks the
# ranks locally rather than asking SLURM for a step.
export HYDRA_BOOTSTRAP=fork
STAGES="${TREEAMR_FOLD_STAGES:-fold tests}"
want() { [[ " $STAGES " == *" $1 "* ]]; }

. "$REPO/bench/simwatch.sh"
nstages=1
for st in $STAGES; do nstages=$((nstages + 1)); done
simwatch_begin "${SLURM_SUBMIT_DIR:-$PWD}/${SLURM_JOB_NAME:-fold}-${SLURM_JOB_ID:-local}" \
    "folds on an H200" "$nstages"

# An environment that develops the checkout and has CUDA, with the test
# environment's dependencies so that the suite runs here too.
simwatch_stage "setting up the environment"
mkdir -p "$SCRATCH"
sed "s|TreeAMR = {path = \"..\"}|TreeAMR = {path = \"$REPO\"}|" \
    "$REPO/test/Project.toml" > "$SCRATCH/Project.toml"
rm -f "$SCRATCH/Manifest.toml"
julia --project="$SCRATCH" -e '
    using Pkg
    Pkg.Registry.update()
    Pkg.add(["CUDA"])
    Pkg.instantiate()
    using CUDA; CUDA.versioninfo()'
echo "=== TreeAMR $REV on $(hostname) ==="

# The downstream's mesh at 32³ (842 blocks, 8.8 GB) and at 16³.
want fold && simwatch_stage "fold against reductions"
want fold && for n in 32 16; do
    TREEAMR_BENCH_BACKEND=cuda TREEAMR_BENCH_N=$n \
        julia --project="$SCRATCH" --threads=8 "$REPO/bench/fold.jl"
done

if want tests; then
    simwatch_stage "test suite, CUDA backend"
    TREEAMR_TEST_BACKEND=cuda TREEAMR_TEST_MPI_CONCURRENT=0 \
        julia --project="$SCRATCH" --threads=8 "$REPO/test/runtests.jl"
fi
