#!/bin/bash
# The distributed checkpoint throughput of M7 on Symmetry, on the
# parallel file system (BeeGFS, /mnt/beegfs): the shared file of step 6
# (the measurement "Parallel I/O and M7" in CODE.md asked for) until step
# 6b replaced it with a part file per I/O process and an index; since
# then each setting runs once per `io` in TREEAMR_CKPT_IO (default "node
# all"), which the benchmark's TREEAMR_BENCH_IO takes.
#
#     sbatch bench/symmetry_checkpoint_mpi.sh                 # 2 nodes, 8 ranks each
#     sbatch --nodes=4 bench/symmetry_checkpoint_mpi.sh 8     # 4 nodes, 8 ranks each
#     sbatch --partition=amddebugq --nodes=1 bench/symmetry_checkpoint_mpi.sh 8
#                                                             # one node only
#     TREEAMR_CKPT_ONENODE_RANKS=8 …                          # only 8 ranks on it
#
# The argument is the ranks per node (default 8, one per NUMA domain of
# an AMD node, at 64 / ranks threads each). The job runs
# `bench/checkpoint.jl mpi` first on one node at 1, 2, 4, … up to that
# many ranks (skipped with TREEAMR_CKPT_ONENODE=0, so that a multi-node
# job need not idle its other nodes through them), then on 2, 4, … nodes
# up to the allocation's, each time over the default mesh (6³ roots,
# 3296 blocks) and over one with 2³ times the roots (12648 blocks: the
# refinement follows a surface, so the blocks grow 3.8-fold), and writes
# the files under TREEAMR_BENCH_DIR on BeeGFS, which is removed at the
# end.
# Every rate is the aggregate over the ranks, the whole state over the
# slowest rank's time (see the header of bench/checkpoint.jl).
#
# What the numbers are not:
#
# - A load right after a save may read the clients' page cache. BeeGFS
#   caches on the client by default in "buffered" mode, so a load on the
#   ranks that wrote is not the servers' read rate. The larger mesh
#   (2.07 GB of `blast` state) is less affected than the default one
#   (540 MB); a cold-cache read needs the caches dropped on every
#   node, which takes root.
# - `sync` waits for every writer's files to reach the servers (each I/O
#   process's `fsync` of its part and of the directory, then rank 0's of
#   the index), but what the servers do with them — their own caches,
#   their disks — is BeeGFS's.
#
# The MPI is MPI.jl's default binary (MPICH_jll), launched by `srun`
# over PMI-2, which needs nothing installed, and HDF5_jll's MPICH
# artifact with it (since step 6b HDF5 is used serially only, so any
# HDF5 build would do). On Symmetry that works (2026-10-02); MPICH_jll's
# libfabric talks TCP over IPoIB there, which matters to the collective
# buffering of MPI-IO between nodes, not to the file system's own
# traffic. `TREEAMR_MPI=system` with `TREEAMR_MPI_MODULE=<module>` uses a
# system MPI instead, through MPIPreferences in the scratch environment;
# HDF5_jll then needs an artifact for that MPI's ABI (it has `mpich` and
# `openmpi` ones), which MPIPreferences selects, and the MPI must be one
# `srun` can start (Symmetry's HPC-X cannot: it is not built with
# SLURM's PMI). amddebugq takes one node a job (MaxNodes=1), so the
# default partition is amdq.

#SBATCH --partition=amdq
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
    for p in (\"HDF5\", \"MPI\", \"MPIPreferences\", \"H5Zzstd\", \"H5Zlz4\",
              \"H5Zbitshuffle\")
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

echo "# $(hostname) $(date -Iseconds) nodes=$NODES ranks/node=$RPN threads/rank=$THREADS $(julia --version)"
lscpu | grep -E 'Model name|NUMA node\(s\)'
julia --project="$ENVDIR" -e 'using MPI, HDF5; MPI.versioninfo(); println("HDF5 ", HDF5.libversion, " parallel ", HDF5.has_parallel())'
df -h "$TREEAMR_BENCH_DIR"

# Each rank gets its own `THREADS` cores, in blocks, so that eight ranks
# on a node are one per NUMA domain. No JULIA_EXCLUSIVE: it would pin
# every rank's threads to the same first cores. The step's task count is
# given explicitly: SLURM 21.08 (Symmetry's) otherwise takes the job's
# and ignores --ntasks-per-node.
run() {  # run <nodes> <ranks per node> <roots>, once per io setting
    local nodes=$1 rpn=$2 roots=$3
    local threads=$((64 / rpn))
    for io in ${TREEAMR_CKPT_IO:-node all}; do
        echo "--- nodes=$nodes ranks/node=$rpn threads/rank=$threads roots=$roots io=$io"
        TREEAMR_BENCH_IO=$io TREEAMR_BENCH_ROOTS=$roots srun --mpi=pmi2 \
            --nodes="$nodes" --ntasks=$((nodes * rpn)) --ntasks-per-node="$rpn" \
            --cpus-per-task="$threads" --cpu-bind=cores --distribution=block:block \
            julia --project="$ENVDIR" -t "$threads" "$REPO/bench/checkpoint.jl" mpi
    done
}

# The one-node rank counts: TREEAMR_CKPT_ONENODE_RANKS, or 1, 2, 4, … up
# to the ranks per node.
ONENODE_RANKS="${TREEAMR_CKPT_ONENODE_RANKS:-}"
if [ -z "$ONENODE_RANKS" ]; then
    r=1
    while [ "$r" -le "$RPN" ]; do
        ONENODE_RANKS="$ONENODE_RANKS $r"
        r=$((r * 2))
    done
fi
for roots in 6 12; do
    if [ "${TREEAMR_CKPT_ONENODE:-1}" = 1 ]; then
        for r in $ONENODE_RANKS; do
            run 1 "$r" "$roots"
        done
    fi
    n=2
    while [ "$n" -le "$NODES" ]; do
        run "$n" "$RPN" "$roots"
        n=$((n * 2))
    done
done

rm -rf "$TREEAMR_BENCH_DIR"
