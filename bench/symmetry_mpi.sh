#!/bin/bash
# The weak-scaling smoke test of M7 step 7 on Symmetry's AMD nodes:
# bench/mpi.jl at a fixed number of blocks per rank, one rank per NUMA
# domain at 8 threads, on one node at 1, 2, 4 and 8 ranks, then on 2 and
# 4 nodes at 8 ranks a node, against the single-process control — one
# 64-thread process over the mesh of the 8-rank run — which is the case
# CODE.md's "What one process loses" expected one rank per domain to
# recover.
#
#     TREEAMR_MPI=system TREEAMR_MPI_MODULE=nvhpc-hpcx-cuda12/24.9 \
#         sbatch bench/symmetry_mpi.sh                     # 4 nodes, N = 16 and 32
#     sbatch --nodes=2 bench/symmetry_mpi.sh               # up to 2 nodes
#     TREEAMR_BENCH_NS=32 sbatch bench/symmetry_mpi.sh     # one block size
#     sbatch --partition=amddebugq --nodes=1 bench/symmetry_mpi.sh   # one node
#
# amddebugq takes one node a job (MaxNodes=1), so the default partition
# is amdq.
#
# Every rank holds one tile of the mesh (see the header of bench/mpi.jl):
# at the default `ROOTS = 4` in 3D, 176 blocks on the two-level mesh and
# 64 on the uniform one, so N = 16 is 0.72 M cells a rank and N = 32 is
# 5.8 M, against the M5 and NUMA measurements' 120 blocks of 32³ a
# domain. Each run prints its tab-separated lines into $OUTDIR; at the
# end the job prints, per block size, the weak-scaling table
# (bench/mpitable.awk, efficiency against 1 rank on 1 domain) and the
# same-mesh table, the 8-rank run against the two controls (efficiency
# below 1 there means the control is faster). The `#` lines of each run
# — messages and bytes per fill and rank, migrated blocks, ns per cell
# of the job — are in $OUTDIR/*.tsv.
#
# Placement. Rank `i` of a node with `R` ranks is bound by `numactl` to
# NUMA domain `i · 8 / R`, its cores and its memory (`--cpunodebind`,
# `--membind`), so 2 ranks take one domain on each socket and 4 every
# other domain, assuming domains 0–3 are socket 0 (`numactl -H` below
# says whether that holds). Within the domain the 8 threads are left to
# the kernel: JULIA_EXCLUSIVE would pin every rank's threads to the
# node's first cores. `srun --cpu-bind=none` gives each task the step's
# cores, `64 / R` a task, so that numactl can narrow them. The controls
# run one process of 64 threads: pinned with first touch (JULIA_EXCLUSIVE,
# what CLAUDE.md recommends on a NUMA node) and pages interleaved (the M5
# table's way).
#
# The MPI is MPI.jl's default binary (MPICH_jll), launched by `srun`
# over PMI-2 (TREEAMR_SRUN_MPI, default pmi2; `srun --mpi=list` shows
# what the SLURM there supports). That works on Symmetry, but MPICH_jll's
# libfabric has no InfiniBand provider and talks TCP over IPoIB (about 2
# GB/s and 40 µs between nodes, measured 2026-10-02). `TREEAMR_MPI=system`
# with `TREEAMR_MPI_MODULE=<module>` uses a system MPI instead, through
# MPIPreferences in the scratch environment, launched by that MPI's own
# `mpiexec`; on Symmetry that is HPC-X's Open MPI 4.1.7 over UCX
# (`nvhpc-hpcx-cuda12/24.9`: 12 GB/s and 2 µs), which is not built with
# SLURM's PMI and so cannot be started by `srun`. The ranks run the
# julia binary itself, not the juliaup launcher, which writes a terminal
# title into Open MPI's pseudo-terminal and so into the tables.
#
# Compare times only within one job: each window is the slowest rank's
# time from a common barrier, so ranks are measured together, but runs
# are not. The job ends with bench/replicated.jl on one domain, the
# replicated regrid and digest passes at up to 180224 leaves.

#SBATCH --partition=amdq
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=treeamr-mpi
#SBATCH --output=treeamr-mpi-%j.out

set -euo pipefail

NODES="${SLURM_JOB_NUM_NODES:-1}"
export PATH="$HOME/.juliaup/bin:$PATH"
export REPO="${TREEAMR_REPO:-$PWD}"
# A project of its own under scratch: the repository's environments stay
# as they are, and MPIPreferences is written here.
export ENVDIR="${TREEAMR_MPI_ENV:-/mnt/beegfs/eschnetter/claude/treeamr-mpi-env}"
OUTDIR="${TREEAMR_MPI_OUT:-/mnt/beegfs/eschnetter/claude/mpi-$SLURM_JOB_ID}"
SRUN_MPI="${TREEAMR_SRUN_MPI:-pmi2}"
NS="${TREEAMR_BENCH_NS:-16 32}"
mkdir -p "$ENVDIR" "$OUTDIR"

if [ "${TREEAMR_MPI:-jll}" = system ]; then
    module load "${TREEAMR_MPI_MODULE:?set TREEAMR_MPI_MODULE to the MPI module}"
fi
julia --project="$ENVDIR" -e "
    using Pkg
    Pkg.develop(path = \"$REPO\")
    for p in (\"MPI\", \"MPIPreferences\", \"KernelAbstractions\", \"Printf\")
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
export JULIA=$(julia --startup-file=no -e 'print(joinpath(Sys.BINDIR, "julia"))')

echo "# $(hostname) $(date -Iseconds) nodes=$NODES mpi=${TREEAMR_MPI:-jll} $("$JULIA" --version)"
lscpu | grep -E 'Model name|^Socket|^NUMA|L3'
numactl -H
"$JULIA" --project="$ENVDIR" -e 'using MPI; MPI.versioninfo()'

# What each rank runs: bound to its NUMA domain, by its rank on the node.
# Under Open MPI's mpiexec SLURM_LOCALID is the daemon's, so the local
# rank is Open MPI's own if it is set.
cat > "$OUTDIR/rank.sh" <<'EOF'
l=${OMPI_COMM_WORLD_LOCAL_RANK:-$SLURM_LOCALID}
r=${OMPI_COMM_WORLD_RANK:-$SLURM_PROCID}
d=$(( l * 8 / TREEAMR_RPN ))
[ "$r" -lt 2 ] && echo "# rank $r on $(hostname): domain $d" >&2
exec numactl --cpunodebind=$d --membind=$d \
    "$JULIA" --project="$ENVDIR" -t 8 "$REPO/bench/mpi.jl" mpi
EOF

# run <label> <nodes> <ranks per node>: one rank per domain, 8 threads.
# The step's task count is given explicitly: SLURM 21.08 (Symmetry's)
# otherwise takes the job's and ignores --ntasks-per-node.
run() {
    local label=$1 nodes=$2 rpn=$3
    echo "--- $label: nodes=$nodes ranks/node=$rpn threads/rank=8 N=$TREEAMR_BENCH_N"
    export TREEAMR_BENCH_LABEL=$label TREEAMR_RPN=$rpn
    if [ "${TREEAMR_MPI:-jll}" = system ]; then
        local pass=()
        for v in $(compgen -e | grep -E '^(TREEAMR_|ENVDIR$|REPO$|JULIA$)'); do
            pass+=(-x "$v")
        done
        mpiexec -n $((nodes * rpn)) --map-by ppr:"$rpn":node --bind-to none \
            "${pass[@]}" bash "$OUTDIR/rank.sh" > "$OUTDIR/N$TREEAMR_BENCH_N-$label.tsv"
    else
        srun --mpi="$SRUN_MPI" --nodes="$nodes" --ntasks=$((nodes * rpn)) \
            --ntasks-per-node="$rpn" --cpus-per-task=$((64 / rpn)) --cpu-bind=none \
            bash "$OUTDIR/rank.sh" > "$OUTDIR/N$TREEAMR_BENCH_N-$label.tsv"
    fi
    unset TREEAMR_BENCH_LABEL TREEAMR_RPN
}

# control <label> <prefix command...>: one 64-thread process over the
# 8-rank mesh, started through the prefix (env or numactl).
control() {
    local label=$1; shift
    echo "--- $label: one process, 64 threads, 8 tiles, N=$TREEAMR_BENCH_N, $*"
    TREEAMR_BENCH_LABEL=$label TREEAMR_BENCH_TILES=8 \
        srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=none \
        "$@" "$JULIA" --project="$ENVDIR" -t 64 "$REPO/bench/mpi.jl" \
        > "$OUTDIR/N$TREEAMR_BENCH_N-$label.tsv"
}

export TREEAMR_BENCH_D=3 TREEAMR_BENCH_ROOTS=4 TREEAMR_BENCH_REPS=10
for n in $NS; do
    export TREEAMR_BENCH_N=$n
    for r in 1 2 4 8; do
        run "${r}x8" 1 "$r"
    done
    for nodes in 2 4; do
        [ "$nodes" -le "$NODES" ] && run "$((8 * nodes))x8" "$nodes" 8
    done
    control 1x64pin env JULIA_EXCLUSIVE=1
    control 1x64il numactl --interleave=all
done

# The replicated O(nleaves) passes at large leaf counts, in one process
# on one domain at a rank's 8 threads (bench/replicated.jl; no MPI).
echo "=== replicated costs, one domain, 8 threads ==="
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=none \
    numactl --cpunodebind=0 --membind=0 \
    "$JULIA" --project="$ENVDIR" -t 8 "$REPO/bench/replicated.jl" 8 64 512 1024 \
    | tee "$OUTDIR/replicated.txt"

echo "=== results ==="
for n in $NS; do
    weak=("$OUTDIR/N$n-1x8.tsv" "$OUTDIR/N$n-2x8.tsv" "$OUTDIR/N$n-4x8.tsv"
          "$OUTDIR/N$n-8x8.tsv")
    for nodes in 2 4; do
        f="$OUTDIR/N$n-$((8 * nodes))x8.tsv"
        [ -f "$f" ] && weak+=("$f")
    done
    echo "## N=$n: weak scaling, one rank per domain at 8 threads"
    awk -F'\t' -f "$REPO/bench/mpitable.awk" "${weak[@]}"
    echo "## N=$n: one node, the same mesh: 8 ranks against one process"
    awk -F'\t' -f "$REPO/bench/mpitable.awk" "$OUTDIR/N$n-8x8.tsv" \
        "$OUTDIR/N$n-1x64pin.tsv" "$OUTDIR/N$n-1x64il.tsv"
done
for f in "$OUTDIR"/*.tsv; do
    echo "## $(basename "$f" .tsv)"
    grep '^#' "$f"
done
