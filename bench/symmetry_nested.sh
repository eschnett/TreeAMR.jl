#!/bin/bash
# The nested-mesh ghost fill on Symmetry (`bench/nested.jl`, the mesh of
# eschnett/TreeGeneralizedHarmonic.jl#11), for a baseline checkout and
# this one in the same job, run after run, so that the two differ only
# in the package:
#
#     TREEAMR_REPO=<this checkout> TREEAMR_BASE=<baseline checkout> \
#         sbatch bench/symmetry_nested.sh            # cpu, an AMD node
#     TREEAMR_REPO=… TREEAMR_NESTED_VERSIONS="base=<dir> step1=<dir> step2=<dir>" \
#         sbatch bench/symmetry_nested.sh            # any number of versions
#     TREEAMR_REPO=… TREEAMR_BASE=… sbatch bench/symmetry_nested_gpu.sh
#
# The second is this script in mode `gpu` under an H200 job's
# directives (4 GPUs on h200q); `bash bench/symmetry_nested.sh gpu` inside
# any allocation with GPUs does the same.
#
# `cpu`, on an AMD node: `bench/nested.jl` at `N = 16` over 1, 2 and 4
# ranks of one thread, and 8 ranks of 8 threads, one rank per NUMA
# domain (rank `i` of `R` bound by `numactl` to domain `i · 8 / R`, its
# cores and its memory, as `bench/symmetry_mpi.sh` places them), under
# MPICH_jll started by `srun --mpi=pmi2`; `bench/ghosts.jl` (cell-centered,
# where Step 1's single-tap columns change nothing: the control) on one
# domain at 1 and 8 threads; and the `--trace-compile-timing` comparison
# of `test/mpi_workload.jl`, serial, three runs of each checkout, with
# the compilation times summed and the outputs compared byte for byte
# (CLAUDE.md's rule for a change to a launch path). Every checkout runs
# the same workload file, the first version's (`TREEAMR_NESTED_WORKLOAD`
# to choose another), so that a version whose workload gained a case is
# not charged for compiling it.
#
# `gpu`, on an H200 node, one rank a GPU: `bench/nested.jl` at `N = 32`
# on 1, 2 and 4 GPUs, under MPICH_jll (always staged through host
# mirrors, launched by its `mpiexec`, Hydra forking the ranks) and under
# HPC-X's Open MPI (`nvhpc-hpcx-cuda12/24.9`, CUDA-aware), staged and
# direct (`TREEAMR_BENCH_DEVICEAWARE=1`), set up through MPIPreferences
# as `bench/symmetry_mpi_gpu.sh` does. One GPU has no messages, so it
# runs under MPICH_jll only.
#
# `TREEAMR_NESTED_STAGES` picks stages (default `setup compile ghosts
# nested` for cpu, `setup nested` for gpu; `smoke` runs `nested.jl` at a
# tiny size instead of the timed sizes, to check the setup).
# `TREEAMR_NESTED_ROUNDS` (default 2) is how often each configuration is
# run per checkout, the order of the checkouts rotating between rounds.
# `TREEAMR_NESTED_VERSIONS` names the checkouts, `name=dir` each, the
# first the reference of the tables (default `base=$TREEAMR_BASE
# new=$TREEAMR_REPO`); the bench files are always `$TREEAMR_REPO`'s.
# `TREEAMR_NESTED_SHAPES` picks the cpu rank shapes (default `1x1 2x1
# 4x1 8x8`), `TREEAMR_NESTED_GPUS` the gpu rank counts (default `1 2 4`).
#
# Each checkout gets environments of its own under `TREEAMR_NESTED_ENVS`,
# per mode, so that the compile caches of the two node types do not
# evict each other. The cpu ones are copies of the checkout's test
# environment with the relative TreeAMR source made absolute (the
# compile workload needs SHA and MultiFloats); the gpu ones develop the
# checkout and add CUDA, MPI and KernelAbstractions. Every run's output
# goes into `$OUT`, one file per run, and the job ends by printing the
# tables: each quantity's minimum for both checkouts and their ratio.
# The job writes a SimWatch status file (`bench/simwatch.sh`) into
# `$SLURM_SUBMIT_DIR/$SLURM_JOB_NAME-$SLURM_JOB_ID`.

#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=treeamr-nested
#SBATCH --output=treeamr-nested-%j.out

set -euo pipefail

MODE="${1:-cpu}"
case $MODE in cpu|gpu) ;; *) echo "mode must be cpu or gpu, got $MODE" >&2; exit 2 ;; esac
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEAMR_REPO:?set TREEAMR_REPO to this checkout}"
VERSIONS="${TREEAMR_NESTED_VERSIONS:-base=${TREEAMR_BASE:?set TREEAMR_BASE or TREEAMR_NESTED_VERSIONS} new=$REPO}"
NAMES=$(for v in $VERSIONS; do printf '%s ' "${v%%=*}"; done)
SHAPES="${TREEAMR_NESTED_SHAPES:-1x1 2x1 4x1 8x8}"
GPUS="${TREEAMR_NESTED_GPUS:-1 2 4}"
ENVS="${TREEAMR_NESTED_ENVS:-$HOME/claude/treeamr-nested/envs}/$MODE"
RUNDIR="${SLURM_SUBMIT_DIR:-$PWD}/${SLURM_JOB_NAME:-nested}-${SLURM_JOB_ID:-local}"
OUT="${TREEAMR_NESTED_OUT:-$RUNDIR}"
ROUNDS="${TREEAMR_NESTED_ROUNDS:-2}"
HPCX="${TREEAMR_NESTED_HPCX:-nvhpc-hpcx-cuda12/24.9}"
if [ "$MODE" = cpu ]; then
    STAGES="${TREEAMR_NESTED_STAGES:-setup compile ghosts nested}"
else
    STAGES="${TREEAMR_NESTED_STAGES:-setup nested}"
fi
want() { [[ " $STAGES " == *" $1 "* ]]; }
mkdir -p "$ENVS" "$OUT"

# MPICH: one BLAS thread a rank, and no release-gather collectives, which
# segfault in the first collective on Symmetry (the symmetry-hpc notes).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPIR_CVAR_CH4_SHM_POSIX_RELEASE_GATHER_ENABLE=0

. "$REPO/bench/simwatch.sh"
nstages=1
for st in $STAGES; do nstages=$((nstages + 1)); done
simwatch_begin "$RUNDIR" "nested-mesh ghost fill, $MODE" "$nstages"

rev() { git -C "$1" rev-parse --short HEAD 2>/dev/null || cat "$1/.revision" 2>/dev/null || echo '?'; }
echo "# $(hostname) $(date -Iseconds) mode=$MODE stages=$STAGES rounds=$ROUNDS"
for v in $VERSIONS; do echo "# ${v%%=*} ${v#*=} ($(rev "${v#*=}"))"; done
echo "# bench files from $REPO ($(rev "$REPO"))"
echo "# output in $OUT"
lscpu | grep -E 'Model name|^Socket|^NUMA|L3' || true
[ "$MODE" = gpu ] && { nvidia-smi -L; nvidia-smi topo -m || true; }

# The checkouts by name, and the environments of each.
repo_of() {
    local v
    for v in $VERSIONS; do [ "${v%%=*}" = "$1" ] && { echo "${v#*=}"; return; }; done
    echo "no version $1" >&2
    return 1
}
cpuenv() { echo "$ENVS/$1"; }
gpuenv() { echo "$ENVS/$1-$2"; }                 # which, jll or hpcx

setup_cpu() {
    local which=$1 repo env
    repo=$(repo_of "$which") env=$(cpuenv "$which")
    mkdir -p "$env"
    sed "s|TreeAMR = {path = \"..\"}|TreeAMR = {path = \"$repo\"}|" \
        "$repo/test/Project.toml" > "$env/Project.toml"
    julia --project="$env" -e '
        using Pkg
        Pkg.Registry.update()
        Pkg.add("MPIPreferences")
        using MPIPreferences
        MPIPreferences.use_jll_binary()
        Pkg.instantiate()
        Pkg.precompile()'
}

setup_gpu() {
    local which=$1 mpi=$2 repo env
    repo=$(repo_of "$which") env=$(gpuenv "$which" "$mpi")
    mkdir -p "$env"
    (
        [ "$mpi" = hpcx ] && module load "$HPCX"
        TREEAMR_SETUP_MPI=$mpi julia --project="$env" -e "
            using Pkg
            Pkg.Registry.update()
            Pkg.develop(path = \"$repo\")
            for p in (\"CUDA\", \"MPI\", \"MPIPreferences\", \"KernelAbstractions\",
                      \"Printf\")
                p in keys(Pkg.project().dependencies) || Pkg.add(p)
            end
            using MPIPreferences
            if ENV[\"TREEAMR_SETUP_MPI\"] == \"hpcx\"
                MPIPreferences.use_system_binary()
            else
                MPIPreferences.use_jll_binary()
            end
            Pkg.instantiate()
            Pkg.precompile()"
    )
}

# What a CPU rank runs, bound to its NUMA domain by its rank on the node.
cat > "$OUT/rank.sh" <<'EOF'
l=${SLURM_LOCALID:-0}
d=$(( l * 8 / NESTED_RPN ))
exec numactl --cpunodebind=$d --membind=$d \
    "$JULIA" --project="$NESTED_ENV" -t "$NESTED_THREADS" "$NESTED_FILE" mpi
EOF

JULIA=$(julia --startup-file=no -e 'print(joinpath(Sys.BINDIR, "julia"))')
export JULIA

# nested_cpu <which> <ranks> <threads> <file stem>
nested_cpu() {
    local which=$1 ranks=$2 threads=$3 stem=$4
    echo "--- nested $which: ${ranks}x$threads N=$TREEAMR_BENCH_N -> $stem"
    NESTED_RPN=$ranks NESTED_THREADS=$threads NESTED_ENV=$(cpuenv "$which") \
        NESTED_FILE="$REPO/bench/nested.jl" TREEAMR_BENCH_LABEL="${ranks}x$threads" \
        srun --mpi=pmi2 --nodes=1 --ntasks="$ranks" --cpus-per-task=$((64 / ranks)) \
        --cpu-bind=none bash "$OUT/rank.sh" > "$OUT/$stem.tsv"
}

# nested_gpu <which> <mpi> <ranks> <deviceaware> <file stem>: through the
# launcher of the environment's MPI binary (`MPI.mpiexec()`), with its
# environment, as bench/mpiscan.sh does; HPC-X's module loaded for it.
nested_gpu() {
    local which=$1 mpi=$2 ranks=$3 aware=$4 stem=$5
    echo "--- nested $which: $mpi, $ranks GPUs, deviceaware=$aware N=$TREEAMR_BENCH_N -> $stem"
    (
        [ "$mpi" = hpcx ] && module load "$HPCX"
        [ "$mpi" = jll ] && export HYDRA_LAUNCHER=fork
        TREEAMR_BENCH_BACKEND=cuda TREEAMR_BENCH_DEVICEAWARE=$aware \
            TREEAMR_BENCH_LABEL="${ranks}gpu-$mpi-$([ "$aware" = 1 ] && echo direct || echo staged)" \
            "$JULIA" --project="$(gpuenv "$which" "$mpi")" -e '
                using MPI
                p, file = ARGS
                m = MPI.mpiexec()
                cmd = `$m -n $p $(Base.julia_cmd()) -t 1 --project=$(Base.active_project()) $file mpi`
                run(setenv(cmd, m.env))' "$ranks" "$REPO/bench/nested.jl" \
            > "$OUT/$stem.tsv"
    )
}

# The order of the checkouts in round `r`: the list rotated by `r - 1`.
order() {
    local -a n=($NAMES)
    local k=${#n[@]} i
    for ((i = 0; i < k; i++)); do printf '%s ' "${n[$(( (i + $1 - 1) % k ))]}"; done
}

if want setup; then
    simwatch_stage "setting up environments"
    for which in $NAMES; do
        if [ "$MODE" = cpu ]; then
            setup_cpu "$which"
        else
            setup_gpu "$which" jll
            setup_gpu "$which" hpcx
        fi
    done
    # Not under HPC-X: a singleton `MPI.Init` of Open MPI in a job waits
    # for a daemon (bench/symmetry_mpi_gpu.sh), so ask MPICH_jll's env.
    [ "$MODE" = gpu ] && "$JULIA" --project="$(gpuenv "${NAMES%% *}" jll)" -e '
        using CUDA, MPI
        MPI.versioninfo(); CUDA.versioninfo()' || true
fi

if want smoke; then
    simwatch_stage "smoke runs at a tiny size"
    export TREEAMR_BENCH_N=8 TREEAMR_BENCH_NVARS=4 TREEAMR_BENCH_REPS=1
    export TREEAMR_BENCH_FOOTPRINT_P=""
    for which in $NAMES; do
        if [ "$MODE" = cpu ]; then
            nested_cpu "$which" 2 1 "smoke-$which-2x1"
            nested_cpu "$which" 8 8 "smoke-$which-8x8"
        else
            nested_gpu "$which" jll 2 0 "smoke-$which-jll-2"
            nested_gpu "$which" hpcx 2 0 "smoke-$which-hpcx-staged-2"
            nested_gpu "$which" hpcx 2 1 "smoke-$which-hpcx-direct-2"
        fi
        cat "$OUT"/smoke-"$which"-*.tsv | grep -E '^(ranks|# footprint check|T.*fill)'
    done
    unset TREEAMR_BENCH_N TREEAMR_BENCH_NVARS TREEAMR_BENCH_REPS TREEAMR_BENCH_FOOTPRINT_P
fi

if want compile; then
    simwatch_stage "compile timing of test/mpi_workload.jl"
    WORKLOAD="${TREEAMR_NESTED_WORKLOAD:-$(repo_of "${NAMES%% *}")/test/mpi_workload.jl}"
    echo "# workload $WORKLOAD"
    for r in 1 2 3; do
        for which in $(order "$r"); do
            repo=$(repo_of "$which")
            stem="compile-$which-$r"
            t0=$(date +%s.%N)
            (cd "$repo" && numactl --cpunodebind=0 --membind=0 \
                "$JULIA" --project="$(cpuenv "$which")" -t 1 \
                --trace-compile="$OUT/$stem.trace" --trace-compile-timing \
                "$WORKLOAD" > "$OUT/$stem.out")
            t1=$(date +%s.%N)
            awk -v which="$which" -v r="$r" -v wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN { print b - a }')" '
                /^#= *[0-9.]+ ms =#/ { n++; t += $2; if ($0 ~ /TreeAMR/) { m++; u += $2 } }
                END { printf "C\t%s\t%d\t%.2f\t%.2f\t%d\t%.2f\t%d\n", which, r, wall,
                      t / 1000, n, u / 1000, m }' "$OUT/$stem.trace" | tee -a "$OUT/compile.tsv"
        done
    done
    first=${NAMES%% *}
    for which in $NAMES; do
        [ "$which" = "$first" ] && continue
        if cmp -s "$OUT/compile-$first-1.out" "$OUT/compile-$which-1.out"; then
            echo "# mpi_workload.jl output: $first and $which identical"
        else
            echo "# mpi_workload.jl output: $first and $which DIFFER"
            diff "$OUT/compile-$first-1.out" "$OUT/compile-$which-1.out" | head -20 || true
        fi
    done
fi

if want ghosts; then
    simwatch_stage "bench/ghosts.jl, one domain"
    for r in $(seq 1 "$ROUNDS"); do
        for which in $(order "$r"); do
            for t in 1 8; do
                for p in 4 6; do
                    echo "--- ghosts $which: t=$t p=$p round $r"
                    TREEAMR_BENCH_N=16 TREEAMR_BENCH_P=$p TREEAMR_BENCH_VARS=20 \
                        numactl --cpunodebind=0 --membind=0 \
                        "$JULIA" --project="$(cpuenv "$which")" -t "$t" \
                        "$REPO/bench/ghosts.jl" | sed "s/^/$which\tp$p\t/" \
                        | tee -a "$OUT/ghosts.tsv"
                done
            done
        done
    done
fi

if want nested; then
    simwatch_stage "bench/nested.jl, every version"
    if [ "$MODE" = cpu ]; then
        export TREEAMR_BENCH_N="${TREEAMR_BENCH_N:-16}"
        for r in $(seq 1 "$ROUNDS"); do
            for shape in $SHAPES; do
                set -- ${shape/x/ }
                for which in $(order "$r"); do
                    nested_cpu "$which" "$1" "$2" "nested-$which-$1x$2-r$r"
                done
            done
        done
    else
        export TREEAMR_BENCH_N="${TREEAMR_BENCH_N:-32}"
        for r in $(seq 1 "$ROUNDS"); do
            for ranks in $GPUS; do
                [ "$ranks" = 1 ] || continue
                for which in $(order "$r"); do
                    nested_gpu "$which" jll 1 0 "nested-$which-1gpu-jll-r$r"
                done
            done
            for ranks in $GPUS; do
                [ "$ranks" = 1 ] && continue
                for v in "jll 0 staged" "hpcx 0 staged" "hpcx 1 direct"; do
                    set -- $v
                    for which in $(order "$r"); do
                        nested_gpu "$which" "$1" "$ranks" "$2" \
                            "nested-$which-${ranks}gpu-$1-$3-r$r"
                    done
                done
            done
        done
    fi
fi

simwatch_stage "tables"
# Per configuration and quantity: the best minimum over the rounds of the
# slowest rank's time, base and new, in ms, and new / base; then the
# per-rank minima of the packs, per stage.
if ls "$OUT"/nested-*.tsv > /dev/null 2>&1; then
    echo "=== nested.jl: slowest rank, min over rounds (ms) ==="
    awk -F'\t' -v names="$NAMES" '
        BEGIN { nv = split(names, vs, " ") }
        FNR == 1 { split(FILENAME, p, "/"); f = p[length(p)]; split(f, q, "-"); which = q[2] }
        $1 == "T" {
            k = $2 SUBSEP $3
            if (!(k in seen)) { seen[k] = 1; keys[++nk] = k }
            v = $4 * 1000
            if (!((k, which) in best) || v < best[k, which]) best[k, which] = v
        }
        END {
            printf "%-24s %-26s", "config", "quantity"
            for (j = 1; j <= nv; j++) printf " %10s", vs[j]
            for (j = 2; j <= nv; j++) printf " %11s", vs[j] "/" vs[1]
            printf "\n"
            for (i = 1; i <= nk; i++) {
                split(keys[i], kk, SUBSEP)
                printf "%-24s %-26s", kk[1], kk[2]
                for (j = 1; j <= nv; j++)
                    if ((keys[i], vs[j]) in best) printf " %10.3f", best[keys[i], vs[j]]
                    else printf " %10s", "-"
                b = best[keys[i], vs[1]]
                for (j = 2; j <= nv; j++)
                    if (b > 0 && (keys[i], vs[j]) in best)
                        printf " %11.3f", best[keys[i], vs[j]] / b
                    else printf " %11s", "-"
                printf "\n"
            }
        }' "$OUT"/nested-*.tsv
    echo "=== nested.jl: per-rank minimum of the packs and stages, round 1 (ms) ==="
    for f in "$OUT"/nested-*-r1.tsv; do
        echo "## $(basename "$f" .tsv)"
        awk -F'\t' '$1 == "T" && ($3 ~ /^packs/ || $3 ~ /^stage/ || $3 ~ /^phase2/ ||
                                   $3 == "fill") {
            n = split($6, t, ","); s = ""
            for (i = 1; i <= n; i++) s = s sprintf(" %9.3f", t[i] * 1000)
            printf "%-22s%s\n", $3, s }' "$f"
    done
    for which in $NAMES; do
        echo "=== nested.jl: counts (round 1, $which) ==="
        for f in "$OUT"/nested-"$which"-*-r1.tsv; do
            [ -f "$f" ] || continue
            echo "## $(basename "$f" .tsv)"
            grep -E '^(ranks|# )' "$f" | grep -v '^# footprint P=[0-9]* L[0-9] [0-9]' || true
        done
    done
fi
if [ -f "$OUT/compile.tsv" ]; then
    echo "=== compile timing of test/mpi_workload.jl: which, run, wall s, compile s," \
         "methods, of which TreeAMR s, methods ==="
    cat "$OUT/compile.tsv"
fi
if [ -f "$OUT/ghosts.tsv" ]; then
    echo "=== bench/ghosts.jl ==="
    grep -v '#' "$OUT/ghosts.tsv"
fi
