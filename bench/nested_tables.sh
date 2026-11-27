#!/bin/bash
# The tables of the nested-mesh benchmark (`jobs/nested.toml`,
# `jobs/nested-gpu.toml`), printed by their `[after]` command:
#
#     bash bench/nested_tables.sh <spec> <job directory>...
#
# Each job directory is `<spec>-<label>/<n>-<job ID>/` and holds the
# run's `output.log`. The labels say what a run is:
#
# - `fill-<version>-<config>-r<round>`: `bench/nested.jl`. Per
#   configuration and quantity, the best minimum over the rounds of the
#   slowest rank's time, per version, in ms, and each version against the
#   first; then the per-rank minima of the packs and stages of round 1,
#   and the counts.
# - `compile-<version>-<round>`: `bench/firstcall.jl` under
#   `--trace-compile=stderr --trace-compile-timing`, the trace in its
#   `output.log` with the output. The compile time summed over all methods
#   and over TreeAMR's, and each run's time to its last step.
# - `ghosts-<version>-p<P>-r<round>-t<threads>`: `bench/ghosts.jl`, its
#   lines prefixed with the version and `P`.
# - `firstcall-<version>-<round>`: `bench/nested.jl` with
#   `TREEAMR_BENCH_FIRSTCALL=1`. Per version and round the first and second
#   call of the schedule and of the fill, and the first less the second,
#   which is the compilation; then each version's least compilation over
#   the rounds, against the first version's.
# - `build-<version>-r<round>-t<threads>`: `bench/schedule_build.jl`. Per
#   mesh, centering and threads the best build over the rounds, in ms,
#   per version, and each against the first; and each version's counts.
#
# The versions are the first one named by a run, then the others in the
# order they appear; the first is the reference. Only labels of these
# forms are read; any other run is left out.

set -u
spec=$1
shift

label() {                       # label <job directory>
    local run
    run=$(basename "$(dirname "$1")")
    echo "${run#"$spec"-}"
}

fills=() compiles=() ghosts=() firsts=() builds=() names=""
for d in "$@"; do
    l=$(label "$d")
    case $l in
        fill-*) fills+=("$d") ;;
        compile-*) compiles+=("$d") ;;
        ghosts-*) ghosts+=("$d") ;;
        firstcall-*) firsts+=("$d") ;;
        build-*) builds+=("$d") ;;
        *) continue ;;
    esac
    which=$(echo "$l" | cut -d- -f2)
    [[ " $names " == *" $which "* ]] || names="$names $which"
done
names=${names# }
first=${names%% *}

if [ ${#fills[@]} -gt 0 ]; then
    logs=()
    for d in "${fills[@]}"; do logs+=("$d/output.log"); done
    echo "=== nested.jl: slowest rank, min over rounds (ms) ==="
    awk -F'\t' -v names="$names" -v spec="$spec" '
        BEGIN { nv = split(names, vs, " ") }
        FNR == 1 {
            n = split(FILENAME, p, "/"); run = p[n - 2]
            sub("^" spec "-fill-", "", run); split(run, q, "-"); which = q[1]
        }
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
        }' "${logs[@]}"
    echo "=== nested.jl: per-rank minimum of the packs and stages, round 1 (ms) ==="
    for d in "${fills[@]}"; do
        l=$(label "$d")
        [[ $l == *-r1 ]] || continue
        echo "## $l"
        awk -F'\t' '$1 == "T" && ($3 ~ /^packs/ || $3 ~ /^stage/ || $3 ~ /^phase2/ ||
                                   $3 == "fill") {
            n = split($6, t, ","); s = ""
            for (i = 1; i <= n; i++) s = s sprintf(" %9.3f", t[i] * 1000)
            printf "%-22s%s\n", $3, s }' "$d/output.log"
    done
    for which in $names; do
        echo "=== nested.jl: counts (round 1, $which) ==="
        for d in "${fills[@]}"; do
            l=$(label "$d")
            [[ $l == fill-"$which"-*-r1 ]] || continue
            echo "## $l"
            grep -E '^(ranks|# )' "$d/output.log" |
                grep -v '^# footprint P=[0-9]* L[0-9] [0-9]' || true
        done
    done
fi

if [ ${#compiles[@]} -gt 0 ]; then
    # The trace without its colours: Julia writes the `# recompile` lines
    # in yellow even into a file.
    plain() { sed "s/$(printf '\033')\[[0-9;]*m//g" "$1"; }
    echo "=== compile timing of bench/firstcall.jl: which, run, compile s," \
         "methods, of which TreeAMR s, methods ==="
    for d in "${compiles[@]}"; do
        l=$(label "$d")
        which=$(echo "$l" | cut -d- -f2) r=$(echo "$l" | cut -d- -f3)
        plain "$d/output.log" | awk -v which="$which" -v r="$r" '
            /^#= *[0-9.]+ ms =#/ { n++; t += $2; if ($0 ~ /TreeAMR/) { m++; u += $2 } }
            END { printf "C\t%s\t%s\t%.2f\t%d\t%.2f\t%d\n", which, r,
                  t / 1000, n, u / 1000, m }'
    done
    # And each run's time from the process start to its last step, under
    # the trace (bench/firstcall.jl's `F` lines; `firstcall-` is untraced).
    for d in "${compiles[@]}"; do
        l=$(label "$d")
        plain "$d/output.log" | awk -v l="$l" -F'\t' '/^F\t/ { last = $3 }
            END { printf "# %s: start to the last step under the trace %s s\n", l, last }'
    done
fi

if [ ${#ghosts[@]} -gt 0 ]; then
    echo "=== bench/ghosts.jl ==="
    for d in "${ghosts[@]}"; do
        l=$(label "$d")
        which=$(echo "$l" | cut -d- -f2) p=$(echo "$l" | cut -d- -f3)
        grep -v '#' "$d/output.log" | sed "s/^/$which	$p	/"
    done
fi

if [ ${#firsts[@]} -gt 0 ]; then
    echo "=== nested.jl first calls: which, run, schedule 1st/2nd/compile s," \
         "fill 1st/2nd/compile s, start-to-fill s ==="
    for d in "${firsts[@]}"; do
        l=$(label "$d")
        which=$(echo "$l" | cut -d- -f2) r=$(echo "$l" | cut -d- -f3)
        awk -F'\t' -v which="$which" -v r="$r" '
            $1 == "F" { a[$3] = $4; b[$3] = $5 }
            END { printf "F\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
                  which, r, a["schedule"], b["schedule"], a["schedule"] - b["schedule"],
                  a["fill"], b["fill"], a["fill"] - b["fill"], a["start-to-fill"] }' \
            "$d/output.log"
    done | awk -F'\t' -v names="$names" '
        BEGIN { nv = split(names, vs, " ") }
        { print
          if (!($2 in s) || $6 < s[$2]) s[$2] = $6
          if (!($2 in f) || $9 < f[$2]) f[$2] = $9
          if (!($2 in t) || $10 < t[$2]) t[$2] = $10 }
        END {
            printf "# least over the rounds (s): which, schedule compile, fill compile, " \
                   "start-to-fill, each against %s\n", vs[1]
            for (j = 1; j <= nv; j++) {
                v = vs[j]
                if (!(v in s)) continue
                printf "# %s\t%.3f\t%.3f\t%.3f", v, s[v], f[v], t[v]
                if (j > 1 && (vs[1] in s))
                    printf "\t%.3f\t%.3f\t%.3f", s[v] / s[vs[1]], f[v] / f[vs[1]],
                           t[v] / t[vs[1]]
                printf "\n"
            }
        }'
fi

if [ ${#builds[@]} -gt 0 ]; then
    logs=()
    for d in "${builds[@]}"; do logs+=("$d/output.log"); done
    echo "=== schedule_build.jl: best build over the rounds (ms) ==="
    awk -F'\t' -v names="$names" -v spec="$spec" '
        BEGIN { nv = split(names, vs, " ") }
        FNR == 1 {
            n = split(FILENAME, p, "/"); run = p[n - 2]
            sub("^" spec "-build-", "", run); split(run, q, "-"); which = q[1]
        }
        $1 == "B" {
            k = $2 SUBSEP $3 SUBSEP $4
            if (!(k in seen)) { seen[k] = 1; keys[++nk] = k }
            v = $5 * 1000
            if (!((k, which) in best) || v < best[k, which]) best[k, which] = v
        }
        END {
            printf "%-8s %-7s %7s", "mesh", "center", "threads"
            for (j = 1; j <= nv; j++) printf " %10s", vs[j]
            for (j = 2; j <= nv; j++) printf " %11s", vs[j] "/" vs[1]
            printf "\n"
            for (i = 1; i <= nk; i++) {
                split(keys[i], kk, SUBSEP)
                printf "%-8s %-7s %7s", kk[1], kk[2], kk[3]
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
        }' "${logs[@]}"
    for which in $names; do
        for d in "${builds[@]}"; do
            l=$(label "$d")
            [[ $l == build-"$which"-r1-t1 ]] || continue
            echo "## counts, $which"
            grep '^# ' "$d/output.log" || true
        done
    done
fi
