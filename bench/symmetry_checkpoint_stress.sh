#!/bin/bash
# The reproducers of the multi-node checkpoint corruption of M7 step 6
# (bench/checkpoint_stress.jl) on Symmetry, under MPICH_jll and its ROMIO,
# launched by `srun` over PMI-2 as in bench/symmetry_checkpoint_mpi.sh.
#
#     sbatch --nodes=4 bench/symmetry_checkpoint_stress.sh <iters> <case>...
#
# A case is `mode[:sync[:hints[:iters[:io]]]]`: a mode of
# checkpoint_stress.jl, 0 or 1 for STRESS_SYNC, a set of ROMIO hints from
# the list below (default none), its own iteration count, and for the
# `treeamr` modes STRESS_IO (default `node`). For example the
# measurement of HISTORY.md ("Parallel checkpoints", M7 step 6):
#
#     sbatch --nodes=4 bench/symmetry_checkpoint_stress.sh 200 \
#         treeamr-filt treeamr-filt:0:dsoff sieve-ind:0:default:2000 \
#         sieve-ind:0:dsoff:2000 sieve-coll:0:cbon sieve-coll:0:cboff visible
#
# Since step 6b the `treeamr` modes save with part files and an index;
# STRESS_IO (`node`, the default, `all` or a number) chooses the I/O
# processes, as in the measurement of step 6b:
#
#     sbatch --nodes=4 bench/symmetry_checkpoint_stress.sh 500 \
#         treeamr-filt:0:default:500:node treeamr-filt:1:default:500:all \
#         treeamr:0:default:500:node
#
# The files go under STRESS_BASEDIR (default BeeGFS scratch; /home is the
# NFS comparison) and are removed at the end; each case's output is kept
# as stress-<job>-<case>.txt in the submit directory, and the job's log
# has a summary line a case. Eight ranks a node, eight threads a rank.
# The corruption needs ranks on more than one node.

#SBATCH --partition=amdq
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=ckpt-stress
#SBATCH --output=stress-%j.out

set -uo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEAMR_REPO:-$PWD}"
OUT="$PWD"
ENVDIR="${TREEAMR_CKPT_ENV:-/mnt/beegfs/eschnetter/claude/treeamr-ckpt-mpi-env}"
DIR="${STRESS_BASEDIR:-/mnt/beegfs/eschnetter/claude}/treeamr-stress-$SLURM_JOB_ID"
NODES=$SLURM_JOB_NUM_NODES
mkdir -p "$DIR/hints"
echo "# $(hostname) $(date -Iseconds) nodes=$NODES $SLURM_JOB_NODELIST dir=$DIR"
df -h "$DIR" | tail -1
# The ROMIO hint sets, as ROMIO_HINTS files.
printf "" > "$DIR/hints/default"
printf "romio_ds_write disable\n" > "$DIR/hints/dsoff"
printf "romio_cb_write enable\nromio_cb_read enable\n" > "$DIR/hints/cbon"
printf "romio_cb_write enable\nromio_ds_write disable\n" > "$DIR/hints/cbondsoff"
printf "romio_cb_write disable\nromio_ds_write disable\n" > "$DIR/hints/cboff"
ITERS=$1
shift
julia --project="$ENVDIR" -e "using Pkg; Pkg.develop(path = \"$REPO\"); Pkg.precompile()" \
    2>&1 | tail -1
for c in "$@"; do
    IFS=: read -r mode sync hints iters io <<< "$c"
    sync=${sync:-0} hints=${hints:-default} iters=${iters:-$ITERS} io=${io:-node}
    [ -f "$DIR/hints/$hints" ] || { echo "no hint set $hints"; exit 1; }
    echo "=== $mode sync=$sync hints=$hints ($(tr '\n' ' ' < "$DIR/hints/$hints")) iterations=$iters io=$io"
    out="$OUT/stress-$SLURM_JOB_ID-$mode-$sync-$hints-$io.txt"
    STRESS_IO=$io STRESS_SYNC=$sync ROMIO_HINTS="$DIR/hints/$hints" \
        srun --mpi=pmi2 --nodes="$NODES" --ntasks=$((NODES * 8)) --ntasks-per-node=8 \
        --cpus-per-task=8 --cpu-bind=cores --distribution=block:block \
        julia --project="$ENVDIR" -t 8 "$REPO/bench/checkpoint_stress.jl" "$mode" "$iters" \
        "$DIR/" 2>&1 | grep -vE '^\s*$' > "$out"
    grep -E "^# " "$out"
    echo "  damaged iterations: $(grep '^BAD' "$out" | awk '{print $3}' | sort -u | wc -l);" \
         "ranks: $(grep '^BAD' "$out" | awk '{print $8}' | sort -u | tr '\n' ' ');" \
         "unseen: $(grep -c '^UNSEEN' "$out")"
    grep -E '^(BAD|UNSEEN)' "$out" | head -4 | cut -c1-200
    grep -m3 -E 'ERROR' "$out"
done
rm -rf "$DIR"
