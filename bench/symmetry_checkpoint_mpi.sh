#!/bin/bash
# The shared-file checkpoint throughput of M7 step 6 on Symmetry: the
# measurement "Parallel I/O and M7" in CODE.md asked for, on the
# parallel file system (BeeGFS, /mnt/beegfs).
#
#     sbatch bench/symmetry_checkpoint_mpi.sh                 # 2 nodes, 8 ranks each
#     sbatch --nodes=4 bench/symmetry_checkpoint_mpi.sh 8     # 4 nodes, 8 ranks each
#     sbatch --nodes=1 bench/symmetry_checkpoint_mpi.sh 1     # 1 rank, the serial file
#
# The argument is the ranks per node (default 8, one per NUMA domain of
# an AMD node, at 64 / ranks threads each). The job runs
# `bench/checkpoint.jl mpi` first on one node at 1, 2, 4, … up to that
# many ranks, then on every node of the allocation, each time over the
# default mesh and over one with 2³ times the blocks, and writes the
# files under TREEAMR_BENCH_DIR on BeeGFS, which is removed at the end.
# Every rate is the aggregate over the ranks, the whole state over the
# slowest rank's time (see the header of bench/checkpoint.jl).
#
# What the numbers are not:
#
# - A load right after a save may read the clients' page cache. BeeGFS
#   caches on the client by default in "buffered" mode, so a load on the
#   ranks that wrote is not the servers' read rate. The larger mesh
#   (about 4.3 GB of `blast` state) is less affected than the default
#   one (540 MB); a cold-cache read needs the caches dropped on every
#   node, which takes root.
# - `sync` waits for every client's writes to reach the servers
#   (`MPI_File_sync` on every rank, then rank 0's `fsync`), but what the
#   servers do with them — their own caches, their disks — is BeeGFS's.
#
# The MPI is MPI.jl's default binary (MPICH_jll), launched by `srun`
# over PMI-2, which needs nothing installed. `TREEAMR_MPI=system` with
# `TREEAMR_MPI_MODULE=<module>` uses a system MPI instead, through
# MPIPreferences in the scratch environment; HDF5_jll then needs an
# artifact for that MPI's ABI (it has `mpich` and `openmpi` ones), which
# MPIPreferences selects. Neither path has been run on Symmetry yet:
# this script was written without access to the cluster, and the first
# job should be read with that in mind (`srun --mpi=list` shows what the
# SLURM there supports).

#SBATCH --partition=amddebugq
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=treeamr-ckpt-mpi
#SBATCH --output=treeamr-ckpt-mpi-%j.out

set -euo pipefail

RPN="${1:-8}"
THREADS=$((64 / RPN))
NODES="${SLURM_JOB_NUM_NODES:-1}"
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEAMR_REPO:-$PWD}"
# A project of its own under scratch, as in symmetry_interpolate.sh: the
# repository's environments must not gain the filter packages.
ENVDIR="${TREEAMR_CKPT_ENV:-/mnt/beegfs/eschnetter/claude/treeamr-ckpt-mpi-env}"
export TREEAMR_BENCH_DIR="${TREEAMR_BENCH_DIR:-/mnt/beegfs/eschnetter/claude/treeamr-ckpt-$SLURM_JOB_ID}"
mkdir -p "$ENVDIR" "$TREEAMR_BENCH_DIR"

if [ "${TREEAMR_MPI:-jll}" = system ]; then
    module load "${TREEAMR_MPI_MODULE:?set TREEAMR_MPI_MODULE to the MPI module}"
fi
julia --project="$ENVDIR" -e "
    using Pkg
    Pkg.develop(path = \"$REPO\")
    for p in (\"HDF5\", \"MPI\", \"MPIPreferences\", \"H5Zzstd\", \"H5Zlz4\")
        p in keys(Pkg.project().dependencies) || Pkg.add(p)
    end
    using MPIPreferences
    if get(ENV, \"TREEAMR_MPI\", \"jll\") == \"system\"
        MPIPreferences.use_system_binary()
    else
        MPIPreferences.use_jll_binary()
    end
    Pkg.instantiate()
    Pkg.precompile()"

echo "# $(hostname) $(date -Iseconds) nodes=$NODES ranks/node=$RPN threads/rank=$THREADS"
lscpu | grep -E 'Model name|NUMA node\(s\)'
julia --project="$ENVDIR" -e 'using MPI, HDF5; MPI.versioninfo(); println("HDF5 ", HDF5.libversion, " parallel ", HDF5.has_parallel())'
df -h "$TREEAMR_BENCH_DIR"

# Each rank gets its own `THREADS` cores, in blocks, so that eight ranks
# on a node are one per NUMA domain. No JULIA_EXCLUSIVE: it would pin
# every rank's threads to the same first cores.
run() {  # run <nodes> <ranks per node> <roots>
    local nodes=$1 rpn=$2 roots=$3
    local threads=$((64 / rpn))
    echo "--- nodes=$nodes ranks/node=$rpn threads/rank=$threads roots=$roots"
    TREEAMR_BENCH_ROOTS=$roots srun --mpi=pmi2 --nodes="$nodes" --ntasks-per-node="$rpn" \
        --cpus-per-task="$threads" --cpu-bind=cores --distribution=block:block \
        julia --project="$ENVDIR" -t "$threads" "$REPO/bench/checkpoint.jl" mpi
}

for roots in 6 12; do
    r=1
    while [ "$r" -le "$RPN" ]; do
        run 1 "$r" "$roots"
        r=$((r * 2))
    done
    if [ "$NODES" -gt 1 ]; then
        run "$NODES" "$RPN" "$roots"
    fi
done

rm -rf "$TREEAMR_BENCH_DIR"
