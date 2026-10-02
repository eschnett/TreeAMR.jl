#!/bin/bash
# The MPI-IO reproducers of bench/checkpoint_stress.jl under HPC-X's Open
# MPI 4.1.7 on Symmetry, launched by its own `mpiexec` (it is not built
# with SLURM's PMI): through OMPIO, Open MPI's own MPI-IO and its default
# here, and through the ROMIO 3.2.1 it also ships (`--mca io romio321`).
#
#     sbatch --nodes=4 bench/symmetry_checkpoint_stress_hpcx.sh <iters> <case>...
#
# A case is `mode[:io[:hints[:iters]]]`, `io` being `ompio` (default) or
# `romio`, `hints` a ROMIO hint set as in symmetry_checkpoint_stress.sh
# (which OMPIO ignores). Only the modes below HDF5 run here: HDF5_jll's
# Open MPI build loads OpenMPI_jll's own library, not HPC-X's, and
# Symmetry has no parallel HDF5 built against HPC-X at hand, so the
# environment is the MPI one of bench/symmetry_mpi.sh, without HDF5.

#SBATCH --partition=amdq
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=ckpt-stress-hpcx
#SBATCH --output=stress-hpcx-%j.out

set -uo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
module load nvhpc-hpcx-cuda12/24.9
REPO="${TREEAMR_REPO:-$PWD}"
OUT="$PWD"
ENVDIR="${TREEAMR_MPI_ENV:-/mnt/beegfs/eschnetter/claude/treeamr-mpi-env-hpcx}"
DIR="${STRESS_BASEDIR:-/mnt/beegfs/eschnetter/claude}/treeamr-stress-$SLURM_JOB_ID"
NODES=$SLURM_JOB_NUM_NODES
mkdir -p "$DIR/hints"
printf "" > "$DIR/hints/default"
printf "romio_ds_write disable\n" > "$DIR/hints/dsoff"
printf "romio_cb_write enable\nromio_cb_read enable\n" > "$DIR/hints/cbon"
printf "romio_cb_write disable\nromio_ds_write disable\n" > "$DIR/hints/cboff"
# The julia binary itself, not the juliaup launcher, which writes a
# terminal title into Open MPI's pseudo-terminal (bench/symmetry_mpi.sh).
JULIA=$(julia --startup-file=no -e 'print(joinpath(Sys.BINDIR, "julia"))')
"$JULIA" --project="$ENVDIR" -e "using Pkg; Pkg.develop(path = \"$REPO\"); Pkg.precompile()" \
    2>&1 | tail -1
echo "# $(hostname) $(date -Iseconds) nodes=$NODES $SLURM_JOB_NODELIST dir=$DIR"
"$JULIA" --project="$ENVDIR" -e 'using MPI; println(MPI.Get_library_version()[1:60])'
ompi_info | grep -E "MCA (io|fcoll|fs|fbtl):"
ITERS=$1
shift
for c in "$@"; do
    IFS=: read -r mode io hints iters <<< "$c"
    io=${io:-ompio} hints=${hints:-default} iters=${iters:-$ITERS}
    mca=()
    [ "$io" = romio ] && mca=(--mca io romio321)
    echo "=== $mode io=$io hints=$hints ($(tr '\n' ' ' < "$DIR/hints/$hints")) iterations=$iters"
    out="$OUT/stress-hpcx-$SLURM_JOB_ID-$mode-$io-$hints.txt"
    mpiexec -n $((NODES * 8)) --map-by ppr:8:node --bind-to none "${mca[@]}" \
        -x STRESS_SYNC=0 -x ROMIO_HINTS="$DIR/hints/$hints" -x PATH \
        "$JULIA" --project="$ENVDIR" -t 8 "$REPO/bench/checkpoint_stress.jl" "$mode" \
        "$iters" "$DIR/" 2>&1 | grep -vE '^\s*$' > "$out"
    grep -E "^# " "$out"
    echo "  damaged iterations: $(grep '^BAD' "$out" | awk '{print $3}' | sort -u | wc -l);" \
         "ranks: $(grep '^BAD' "$out" | awk '{print $8}' | sort -u | tr '\n' ' ')"
    grep -E '^BAD' "$out" | head -4 | cut -c1-200
    grep -m3 -E 'ERROR' "$out"
done
rm -rf "$DIR"
