# SimWatch status files for TreeAMR's Slurm jobs.
#
# SimWatch (https://github.com/eschnett/simwatch, whose FORMAT.md is the
# reference) shows every run below a directory with its progress, its
# Slurm state and a one-line message, from a `simwatch.toml` that the run
# rewrites about once a minute. TreeAMR runs no simulations; what runs
# long here are the benchmark and test jobs in `bench/symmetry_*.sh`,
# whose progress is a sequence of stages. So a job reports the stage it is
# in, as `message` and as `progress.fraction`, and a heartbeat rewrites the
# file every minute so that a long stage is not shown as stale. The
# downstream TreeGeneralizedHarmonic writes the same file from Julia
# (`src/simwatch.jl` there); this is the shell form of it.
#
# In a batch script, after `set -euo pipefail`:
#
#     . "$REPO/bench/simwatch.sh"
#     simwatch_begin "$RUNDIR" "copies H200" 5      # name, number of stages
#     simwatch_stage "setting up environments"       # stage 1 of 5
#     …
#     simwatch_stage "test suite"                    # stage 5 of 5
#
# The exit trap writes `finished` (exit code 0) or `failed`, with the exit
# code and the stage it failed in, and stops the heartbeat. `RUNDIR` is
# where the file goes; the jobs default it to
# `$SLURM_SUBMIT_DIR/$SLURM_JOB_NAME-$SLURM_JOB_ID`, beside their output
# file, so `simwatch <submit dir>` finds them.
#
# On the submit side, to show a job while it is queued:
#
#     jobid=$(sbatch --parsable bench/symmetry_copies.sh)
#     simwatch_queued "$PWD/treeamr-copies-$jobid" "copies H200" "$jobid"

_simwatch_quote() {
    local s=${1//\\/\\\\}
    s=${s//\"/\\\"}
    printf '"%s"' "${s//$'\n'/ }"
}

_simwatch_now() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

# Seconds in a Slurm time limit, `[D-]HH:MM:SS`, `MM:SS` or `MM`; empty if
# there is none.
_simwatch_seconds() {
    local t=$1 d=0 h=0 m=0 s=0
    case $t in ""|UNLIMITED|INVALID) return ;; esac
    if [[ $t == *-* ]]; then d=${t%%-*}; t=${t#*-}; fi
    IFS=: read -r -a parts <<< "$t"
    case ${#parts[@]} in
        3) h=${parts[0]}; m=${parts[1]}; s=${parts[2]} ;;
        2) m=${parts[0]}; s=${parts[1]} ;;
        1) m=${parts[0]} ;;
    esac
    echo $(( ((10#$d * 24 + 10#$h) * 60 + 10#$m) * 60 + 10#$s ))
}

# Write the whole file from the job's state: STATUS [MESSAGE]. The state
# lives in files beside it, so that the heartbeat, a subshell, sees the
# stages the script has reached since it started.
_simwatch_write() {
    local status=$1 message=${2-}
    local dir=$SIMWATCH_DIR f="$SIMWATCH_DIR/simwatch.toml" tmp
    local stage=0 label=""
    [ -f "$dir/.simwatch-stage" ] && read -r stage label < "$dir/.simwatch-stage"
    [ -n "$message" ] || message=$label
    local elapsed=$(( $(date +%s) - SIMWATCH_T0 ))
    tmp="$f.tmp.$BASHPID"
    {
        echo "name = $(_simwatch_quote "$SIMWATCH_NAME")"
        echo "status = $(_simwatch_quote "$status")"
        echo "code = \"TreeAMR.jl\""
        echo "group = \"treeamr-bench\""
        echo "updated = $(_simwatch_now)"
        echo "started = $SIMWATCH_STARTED"
        echo "update_interval = 60"
        echo "host = $(_simwatch_quote "$(hostname -s)")"
        echo "pid = $SIMWATCH_PID"
        echo "message = $(_simwatch_quote "$message")"
        echo "summary = [\"stage\"]"
        echo "stage = $(_simwatch_quote "$stage/$SIMWATCH_NSTAGES $label")"
        echo
        echo "[progress]"
        if [ "$status" = finished ]; then
            echo "fraction = 1.0"
        else
            # A stage counts as done when the next one begins.
            echo "fraction = $(awk -v s="$stage" -v n="$SIMWATCH_NSTAGES" \
                               'BEGIN { printf "%.4f", (s > 0 ? s - 1 : 0) / n }')"
        fi
        echo "walltime = $elapsed"
        [ -n "$SIMWATCH_LIMIT" ] && echo "walltime_limit = $SIMWATCH_LIMIT"
        echo
        echo "[resources]"
        echo "nodes = ${SLURM_JOB_NUM_NODES:-1}"
        echo "tasks = ${SLURM_NTASKS:-1}"
        echo "threads = ${SLURM_CPUS_PER_TASK:-1}"
        local gpus=${CUDA_VISIBLE_DEVICES:-}
        [ -n "$gpus" ] && echo "gpus = $(awk -F, '{ print NF }' <<< "$gpus")"
        if [ -n "${SLURM_JOB_ID:-}" ]; then
            echo
            echo "[slurm]"
            echo "job_id = $(_simwatch_quote "$SLURM_JOB_ID")"
            echo "job_name = $(_simwatch_quote "${SLURM_JOB_NAME:-}")"
            echo "partition = $(_simwatch_quote "${SLURM_JOB_PARTITION:-}")"
        fi
    } > "$tmp" && mv -f "$tmp" "$f"
    return 0
}

# simwatch_begin DIR NAME NSTAGES
simwatch_begin() {
    SIMWATCH_DIR=$1 SIMWATCH_NAME=$2 SIMWATCH_NSTAGES=$3
    SIMWATCH_T0=$(date +%s) SIMWATCH_STARTED=$(_simwatch_now) SIMWATCH_PID=$$
    SIMWATCH_LIMIT=""
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v squeue > /dev/null; then
        SIMWATCH_LIMIT=$(_simwatch_seconds "$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null)")
    fi
    mkdir -p "$SIMWATCH_DIR"
    echo "0 starting" > "$SIMWATCH_DIR/.simwatch-stage"
    _simwatch_write starting
    ( while sleep 60; do _simwatch_write running; done ) &
    SIMWATCH_HEARTBEAT=$!
    trap '_simwatch_exit $?' EXIT
}

# simwatch_stage MESSAGE: the next stage begins.
simwatch_stage() {
    local stage=0 label
    read -r stage label < "$SIMWATCH_DIR/.simwatch-stage"
    echo "$((stage + 1)) $1" > "$SIMWATCH_DIR/.simwatch-stage"
    echo "=== $1 ==="
    _simwatch_write running
}

_simwatch_exit() {
    local rc=$1 stage=0 label=""
    kill "$SIMWATCH_HEARTBEAT" 2>/dev/null || true
    read -r stage label < "$SIMWATCH_DIR/.simwatch-stage" || true
    if [ "$rc" -eq 0 ]; then
        _simwatch_write finished "all $SIMWATCH_NSTAGES stages done"
    else
        _simwatch_write failed "exit code $rc in stage $stage ($label) on $(hostname -s)"
    fi
    rm -f "$SIMWATCH_DIR/.simwatch-stage"
    return "$rc"
}

# simwatch_queued DIR NAME JOBID: a submitted job, before it starts.
simwatch_queued() {
    local dir=$1 name=$2 jobid=$3 f="$1/simwatch.toml"
    mkdir -p "$dir" || return
    {
        echo "name = $(_simwatch_quote "$name")"
        echo "status = \"queued\""
        echo "code = \"TreeAMR.jl\""
        echo "group = \"treeamr-bench\""
        echo "updated = $(_simwatch_now)"
        echo
        echo "[slurm]"
        echo "job_id = $(_simwatch_quote "$jobid")"
    } > "$f.tmp.$$" && mv -f "$f.tmp.$$" "$f"
}
