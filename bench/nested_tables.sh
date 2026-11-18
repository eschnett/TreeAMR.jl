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
# - `compile-<version>-<round>`: `test/mpi_workload.jl` under
#   `--trace-compile=stderr --trace-compile-timing`, the trace in its
#   `output.log` with the output. The compile time summed over all methods
#   and over TreeAMR's, and whether the output of round 1, without the
#   trace, is the same for every version.
# - `ghosts-<version>-p<P>-r<round>-t<threads>`: `bench/ghosts.jl`, its
#   lines prefixed with the version and `P`.
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

fills=() compiles=() ghosts=() names=""
for d in "$@"; do
    l=$(label "$d")
    case $l in
        fill-*) fills+=("$d") ;;
        compile-*) compiles+=("$d") ;;
        ghosts-*) ghosts+=("$d") ;;
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
    echo "=== compile timing of test/mpi_workload.jl: which, run, compile s," \
         "methods, of which TreeAMR s, methods ==="
    for d in "${compiles[@]}"; do
        l=$(label "$d")
        which=$(echo "$l" | cut -d- -f2) r=$(echo "$l" | cut -d- -f3)
        plain "$d/output.log" | awk -v which="$which" -v r="$r" '
            /^#= *[0-9.]+ ms =#/ { n++; t += $2; if ($0 ~ /TreeAMR/) { m++; u += $2 } }
            END { printf "C\t%s\t%s\t%.2f\t%d\t%.2f\t%d\n", which, r,
                  t / 1000, n, u / 1000, m }'
    done
    output() { plain "$1" | grep -Ev '^#= *[0-9.]+ ms =#'; }
    ref=""
    for d in "${compiles[@]}"; do
        [ "$(label "$d")" = "compile-$first-1" ] && ref="$d/output.log"
    done
    for which in $names; do
        [ "$which" = "$first" ] && continue
        for d in "${compiles[@]}"; do
            [ "$(label "$d")" = "compile-$which-1" ] || continue
            if [ -z "$ref" ]; then
                echo "# mpi_workload.jl output: no round 1 of $first to compare $which with"
            elif cmp -s <(output "$ref") <(output "$d/output.log"); then
                echo "# mpi_workload.jl output: $first and $which identical"
            else
                echo "# mpi_workload.jl output: $first and $which DIFFER"
                diff <(output "$ref") <(output "$d/output.log") | head -20 || true
            fi
        done
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
