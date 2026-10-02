#!/bin/bash
# The MPI+GPU acceptance run of M7 step 8 on Symmetry: the distributed
# mesh on CUDA, one rank per GPU of one H200 node, every rank count
# against the serial run on the same backend, through host staging and,
# where the MPI is CUDA-aware, with the device buffers handed to MPI
# directly.
#
#     sbatch bench/symmetry_mpi_gpu.sh                    # 4 GPUs, MPICH_jll
#     sbatch --gres=gpu:h200:8 --ntasks=8 bench/symmetry_mpi_gpu.sh
#     TREEAMR_MPI=system TREEAMR_MPI_MODULE=<module> sbatch bench/symmetry_mpi_gpu.sh
#
# What it runs is `test/mpi_device_tests.jl` (see its header and that of
# `test/mpi_device_workload.jl`): the serial run, then `mpiexec -n 2`,
# `-n 3` and `-n <GPUs>`, each rank on GPU `local rank mod GPUs`, every
# line of the output compared with the serial one — byte for byte, but
# for the floating-point sums — and the `#` lines checked to show that the
# path meant is the path that ran. In Float64 and in Float32.
#
# The two message paths ("MPI+GPU" under "Distributed meshes" in
# CODE.md):
#
# - *Host staging* is the default and always runs: each stage's packed
#   buffer is copied to a page-locked host mirror (CUDA.pin, through
#   KernelAbstractions.pagelock!) and sent from there.
# - *Direct*, `communicator(comm; deviceaware = true)`, runs only when
#   `MPI.has_cuda()` says the MPI is CUDA-aware, or when
#   `TREEAMR_DEVICEAWARE=1` forces it. `MPI.has_cuda()` asks Open MPI
#   (and IBM Spectrum MPI) at run time; for any MPICH it is false unless
#   `JULIA_MPI_HAS_CUDA` is set, since MPICH has no query for it. So with
#   the default binary, MPICH_jll, which is not built with CUDA, only the
#   staging path runs, and the direct one needs a CUDA-aware system MPI
#   selected with `TREEAMR_MPI=system TREEAMR_MPI_MODULE=<module>`, as in
#   `symmetry_checkpoint_mpi.sh`. On Symmetry that is HPC-X's Open MPI
#   4.1.7 (`nvhpc-hpcx-cuda12/24.9`; `ompi_info --parsable --all | grep
#   cuda_support` says it is built with CUDA), launched by its own
#   `mpiexec`, which `MPI.mpiexec()` names under the system binary.
#
# After the tests, the weak-scaling benchmark of step 7 (`bench/mpi.jl`
# through `bench/mpiscan.sh`) on CUDA at 1, 2 and `GPUs` ranks, one GPU a
# rank, through host staging and, where the MPI is CUDA-aware, directly:
# that is where the cost of the two paths is measured. Its block size is
# TREEAMR_BENCH_N (default 32 here, 5.8 M cells a rank).

#SBATCH --partition=h200debugq
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:4
#SBATCH --time=1:00:00
#SBATCH --job-name=treeamr-mpi-gpu
#SBATCH --output=treeamr-mpi-gpu-%j.out

set -euo pipefail

export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEAMR_REPO:-$PWD}"
# A project of its own under scratch: the repository's environments must
# not gain CUDA (CI has no GPU).
ENVDIR="${TREEAMR_MPI_GPU_ENV:-/mnt/beegfs/eschnetter/claude/treeamr-mpi-gpu-env}"
mkdir -p "$ENVDIR"

if [ "${TREEAMR_MPI:-jll}" = system ]; then
    module load "${TREEAMR_MPI_MODULE:?set TREEAMR_MPI_MODULE to the MPI module}"
else
    # MPICH_jll's hydra would launch its proxies through `srun`, a step
    # inside this one; on one node it can fork them.
    export HYDRA_LAUNCHER=fork
fi
julia --project="$ENVDIR" -e "
    using Pkg
    Pkg.develop(path = \"$REPO\")
    for p in (\"CUDA\", \"MPI\", \"MPIPreferences\", \"KernelAbstractions\", \"SHA\",
              \"Printf\", \"Test\")
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

NGPU=$(nvidia-smi -L | wc -l)
echo "# $(hostname) $(date -Iseconds) gpus=$NGPU mpi=${TREEAMR_MPI:-jll} $(julia --version)"
nvidia-smi -L
nvidia-smi topo -m || true
# Under the system MPI the query runs as one rank of its `mpiexec`: a
# singleton `MPI.Init()` of Open MPI 4.1 inside a SLURM job waits forever
# for the daemon it spawns (seen on Symmetry). The julia binary itself,
# not the juliaup launcher, which writes a terminal title into Open MPI's
# pseudo-terminal; and only the digit is kept.
JULIA=$(julia --startup-file=no -e 'print(joinpath(Sys.BINDIR, "julia"))')
LAUNCH=()
[ "${TREEAMR_MPI:-jll}" = system ] && LAUNCH=(mpiexec -n 1)
AWARE=$("${LAUNCH[@]}" "$JULIA" --project="$ENVDIR" -e '
    using CUDA, MPI
    MPI.Init()          # Open MPI answers MPIX_Query_cuda_support only after it
    MPI.versioninfo(stderr)
    CUDA.versioninfo(stderr)
    println(stderr, "MPI.has_cuda() = ", MPI.has_cuda())
    print(MPI.has_cuda() ? 1 : 0)' | tr -dc 01)
echo "# has_cuda=$AWARE"

RANKS="${TREEAMR_TEST_RANKS:-2 3 $NGPU}"
for T in Float64 Float32; do
    echo "=== host staging, $T, ranks $RANKS ==="
    TREEAMR_TEST_BACKEND=cuda TREEAMR_TEST_T=$T TREEAMR_TEST_RANKS="$RANKS" \
        TREEAMR_TEST_DEVICEAWARE=0 \
        julia --project="$ENVDIR" "$REPO/test/mpi_device_tests.jl"
    if [ "$AWARE" = 1 ] || [ "${TREEAMR_DEVICEAWARE:-0}" = 1 ]; then
        echo "=== device buffers to MPI directly, $T, ranks $RANKS ==="
        TREEAMR_TEST_BACKEND=cuda TREEAMR_TEST_T=$T TREEAMR_TEST_RANKS="$RANKS" \
            TREEAMR_TEST_DEVICEAWARE=1 \
            julia --project="$ENVDIR" "$REPO/test/mpi_device_tests.jl"
    else
        echo "=== the direct path is skipped: MPI.has_cuda() is false ==="
    fi
done

N="${TREEAMR_BENCH_N:-32}"
echo "=== bench/mpi.jl on CUDA, host staging, N=$N ==="
TREEAMR_BENCH_BACKEND=cuda TREEAMR_BENCH_N=$N TREEAMR_BENCH_PROJECT="$ENVDIR" \
    TREEAMR_BENCH_DEVICEAWARE=0 "$REPO/bench/mpiscan.sh" 1 2 "$NGPU"
if [ "$AWARE" = 1 ] || [ "${TREEAMR_DEVICEAWARE:-0}" = 1 ]; then
    echo "=== bench/mpi.jl on CUDA, device buffers to MPI directly, N=$N ==="
    TREEAMR_BENCH_BACKEND=cuda TREEAMR_BENCH_N=$N TREEAMR_BENCH_PROJECT="$ENVDIR" \
        TREEAMR_BENCH_DEVICEAWARE=1 "$REPO/bench/mpiscan.sh" 1 2 "$NGPU"
fi
