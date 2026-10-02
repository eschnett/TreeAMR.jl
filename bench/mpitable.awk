# The weak-scaling table of bench/mpi.jl's output (M7 step 7): reads the
# tab-separated lines `<ranks>x<threads>  <mesh>:<phase>  <min s>  <median s>`
# of one or more runs, concatenated, and prints per phase the efficiency
# of each run against the first one — t(first) / t(run), from the
# minimum over the windows, which at a fixed number of blocks per rank is
# 1 for perfect weak scaling — then the minimum and the median seconds.
#
#     awk -F'\t' -f bench/mpitable.awk run1.tsv run2.tsv …
#
# The column order is the order the runs appear in. Lines starting with
# `#`, and the `ranks=` header lines (the last one is printed), are
# otherwise ignored.
/^ranks=/ { header = $0; next }
/^#/ { next }
NF == 4 {
    key = $2 SUBSEP $1
    tmin[key] = $3
    tmed[key] = $4
    if (!seen[$2]++) phase[++np] = $2
    if (!rseen[$1]++) run[++nr] = $1
}
END {
    print header
    printf "%-28s", "phase (ranks x threads)"
    for (i = 1; i <= nr; i++) printf "%9s", run[i]
    printf "  |"
    for (i = 1; i <= nr; i++) printf "%10s", run[i]
    printf "  |"
    for (i = 1; i <= nr; i++) printf "%10s", run[i]
    printf "\n"
    printf "%-28s%*s  |%*s  |%*s\n", "", nr * 9, "efficiency", nr * 10, "min s",
           nr * 10, "median s"
    for (j = 1; j <= np; j++) {
        p = phase[j]
        printf "%-28s", p
        for (i = 1; i <= nr; i++) {
            k = p SUBSEP run[i]
            if (!(k in tmin) || tmin[k] == 0 || tmin[p SUBSEP run[1]] == 0) printf "%9s", "-"
            else printf "%9.2f", tmin[p SUBSEP run[1]] / tmin[k]
        }
        printf "  |"
        for (i = 1; i <= nr; i++) {
            k = p SUBSEP run[i]
            if (k in tmin) printf "%10.6f", tmin[k]; else printf "%10s", "-"
        }
        printf "  |"
        for (i = 1; i <= nr; i++) {
            k = p SUBSEP run[i]
            if (k in tmed) printf "%10.6f", tmed[k]; else printf "%10s", "-"
        }
        printf "\n"
    }
}
