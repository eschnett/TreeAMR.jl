# The collectives that `interpolate!` calls over MPI, timed on their own:
# one `Allgather` of its checks, then two rounds of an `Alltoall` of the
# counts and an `Alltoallv` of the payload, with the sizes `bench/mpi.jl`'s
# interpolation sends (1000 points a rank over the whole domain, so about
# `24000 / P` bytes to each peer out and `16000 / P` back). At 32 ranks on
# 4 nodes of Symmetry, `interpolate!` took 24–25 ms under HPC-X through
# mpi_abi_wrapper and 3.5–4.1 under MPICH_jll, while every other phase
# agreed (HISTORY.md, "Development environment and CI", *Cluster jobs*).
# This separates the collectives from the rest of the call, and sizes
# from latency, so that one MPI setting can be compared against another
# (`jobs/collectives.toml`).
#
#     julia --project=test -e 'using MPI; m = MPI.mpiexec(); run(setenv(`$m -n 4 $(Base.julia_cmd()) --project=test bench/collectives.jl`, m.env))'
#
# Every case is timed in synchronized windows, as `bench/mpi.jl` times
# its phases: a barrier, the case, and the slowest rank's time, gathered
# outside the window. The output is `bench/mpi.jl`'s, `<label> <case>
# <min s> <median s>` tab-separated, so `bench/mpitable.awk` lays several
# runs side by side; a `#` line per case adds the 10th and 90th
# percentiles, the maximum and how many windows took over 5 ms, since a
# median alone hides a time that comes in steps.
#
#     TREEAMR_BENCH_LABEL   the run's column in the table (default "<P>x<threads>")
#     TREEAMR_BENCH_REPS    windows per case (default 200)
#     TREEAMR_BENCH_NPTS    points a rank, as in bench/mpi.jl (default 1000)
using MPI
using Printf

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const P = MPI.Comm_size(COMM)
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "200"))
const NPTS = parse(Int, get(ENV, "TREEAMR_BENCH_NPTS", "1000"))
const LABEL = get(ENV, "TREEAMR_BENCH_LABEL", "$(P)x$(Threads.nthreads())")

say(s) = (RANK == 0 && (println(s); flush(stdout)); nothing)

# One synchronized window: the slowest rank's time for `f`.
function window(f)
    MPI.Barrier(COMM)
    t = @elapsed f()
    return maximum(MPI.Allgather(t, COMM))
end

quantile_sorted(s, q) = s[clamp(round(Int, q * (length(s) - 1)) + 1, 1, length(s))]

function measure(name, f)
    f()
    f()
    GC.gc()
    ts = sort!([window(f) for _ in 1:REPS])
    med = (ts[(REPS + 1) ÷ 2] + ts[REPS ÷ 2 + 1]) / 2
    say(@sprintf("%s\t%s\t%.6f\t%.6f", LABEL, name, ts[1], med))
    say(@sprintf("# %s %s: p10 %.3f ms, p90 %.3f ms, max %.3f ms, %d of %d over 5 ms",
                 LABEL, name, 1e3 * quantile_sorted(ts, 0.1), 1e3 * quantile_sorted(ts, 0.9),
                 1e3 * ts[end], count(>(5e-3), ts), REPS))
    return nothing
end

# `n` items split over the ranks as evenly as they go, the first ranks
# taking one more: each rank's count for each peer.
splitcounts(n) = Cint[n ÷ P + (r <= n % P ? 1 : 0) for r in 1:P]

# An `Alltoallv` of `T`s with `counts` to each peer: the counts first, by
# `Alltoall`, as TreeAMR's `alltoallv` does, then the payload.
function alltoallv_round(send::Vector{T}, scounts::Vector{Cint}) where {T}
    rcounts = MPI.Alltoall(MPI.UBuffer(scounts, 1), COMM)
    recv = Vector{T}(undef, sum(rcounts))
    MPI.Alltoallv!(MPI.VBuffer(send, scounts), MPI.VBuffer(recv, rcounts), COMM)
    return recv, rcounts
end

# The same exchange by `Isend` and `Irecv!` to every peer, to tell the
# collective's algorithm from the transport.
function pairwise(send::Vector{T}, scounts::Vector{Cint}) where {T}
    rcounts = MPI.Alltoall(MPI.UBuffer(scounts, 1), COMM)
    recv = Vector{T}(undef, sum(rcounts))
    soff, roff = cumsum([0; scounts[1:(end - 1)]]), cumsum([0; rcounts[1:(end - 1)]])
    reqs = MPI.Request[]
    for k in 1:(P - 1)
        src = mod(RANK - k, P)
        rcounts[src + 1] > 0 && push!(reqs, MPI.Irecv!(
            view(recv, roff[src + 1] .+ (1:rcounts[src + 1])), COMM; source=src, tag=7))
    end
    for k in 1:(P - 1)
        dst = mod(RANK + k, P)
        scounts[dst + 1] > 0 && push!(reqs, MPI.Isend(
            view(send, soff[dst + 1] .+ (1:scounts[dst + 1])), COMM; dest=dst, tag=7))
    end
    r = RANK + 1
    recv[roff[r] .+ (1:rcounts[r])] .= view(send, soff[r] .+ (1:scounts[r]))
    MPI.Waitall(reqs)
    return recv, rcounts
end

function main()
    nodes = let node = MPI.Comm_split_type(COMM, MPI.COMM_TYPE_SHARED, RANK)
        first = MPI.Comm_rank(node) == 0
        MPI.free(node)
        MPI.Allreduce(Int(first), +, COMM)
    end
    say("ranks=$P nodes=$nodes threads=$(Threads.nthreads()) npts=$NPTS reps=$REPS " *
        "label=$LABEL Julia $VERSION")
    say("# library: " * replace(strip(MPI.Get_library_version()), r"\s*\n\s*" => "; "))
    for (k, v) in sort!([kv for kv in ENV if startswith(kv[1], "OMPI_MCA_") ||
                                              startswith(kv[1], "UCX_") ||
                                              startswith(kv[1], "HCOLL_") ||
                                              startswith(kv[1], "MPIR_CVAR_")])
        say("# env: $k=$v")
    end

    check = zeros(UInt8, 80)                # the size of interpolate!'s checks
    gathered = zeros(UInt8, 80 * P)
    one = Cint[1 for _ in 1:P]
    pts = splitcounts(NPTS)                 # points to each owner
    out = rand(NTuple{3,Float64}, NPTS)     # 24 bytes a point
    back = rand(Float64, 2 * NPTS)          # 2 variables a point

    measure("barrier", () -> MPI.Barrier(COMM))
    measure("allreduce 8 B", () -> MPI.Allreduce(1.0, +, COMM))
    measure("allgather 80 B", () -> MPI.Allgather!(check, MPI.UBuffer(gathered, 80), COMM))
    measure("alltoall 4 B", () -> MPI.Alltoall(MPI.UBuffer(one, 1), COMM))
    for bytes in (8, 64, 768, 8192, 65536)
        send = zeros(UInt8, bytes * P)
        counts = fill(Cint(bytes), P)
        measure("alltoallv $bytes B/peer", () -> alltoallv_round(send, counts))
    end
    measure("pairwise 768 B/peer",
            () -> pairwise(zeros(UInt8, 768 * P), fill(Cint(768), P)))
    measure("interpolate: points out", () -> alltoallv_round(out, pts))
    measure("interpolate: values back", () -> alltoallv_round(back, Cint(2) .* pts))
    measure("interpolate: all", () -> begin
        MPI.Allgather!(check, MPI.UBuffer(gathered, 80), COMM)
        received, rcounts = alltoallv_round(out, pts)
        alltoallv_round(zeros(2 * length(received)), Cint(2) .* rcounts)
    end)
    measure("interpolate: all, pairwise", () -> begin
        MPI.Allgather!(check, MPI.UBuffer(gathered, 80), COMM)
        received, rcounts = pairwise(out, pts)
        pairwise(zeros(2 * length(received)), Cint(2) .* rcounts)
    end)
    return nothing
end

main()
MPI.Finalize()
