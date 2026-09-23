# Does it cost anything, on an idle node, to read lines whose last owner
# was another core? One pinned core writes (or writes, then re-reads) a
# buffer and evicts it by streaming 1 GB; a second core then chases a
# random cycle through the buffer (latency) or streams it (one core's
# bandwidth). Owner and reader are the same core, the same CCD, another
# CCD on the socket, and the other socket.
#
#     JULIA_EXCLUSIVE=1 numactl --membind=0 julia -t 64 --project=. bench/owner.jl
#
# On Symmetry (2026-09-23) nothing moved: 74-79 ns and 27-29 GB/s
# whoever the owner was, with the buffer in the reader's domain, and no
# systematic change interleaved. So the affinity loss of CODE.md ("What
# one process loses") is a cost under load, not a latency. Needs
# pinning: static slot t must run on CPU t - 1, which the script prints.
using Printf, Random
const nt = Threads.nthreads()
const MB = parse(Int, get(ENV, "MB", "64"))
const nl = MB * 2^20 ÷ 64                  # cache lines in the buffer
getcpu() = ccall(:sched_getcpu, Cint, ())

function on(cpu, f)
    r = Ref{Any}()
    Threads.@threads :static for s in 1:nt
        s == cpu + 1 && (r[] = f())
    end
    return r[]
end

const BUF = zeros(Int, 8nl)
const FLUSH = zeros(Float64, 2^27)        # 1 GB
const PERM = randperm(MersenneTwister(1), nl)

function build!()             # a cyclic chase through all lines, written by the caller
    @inbounds for k in 1:nl
        BUF[8(PERM[k] - 1) + 1] = 8(PERM[mod1(k + 1, nl)] - 1) + 1
        for j in 2:8; BUF[8(PERM[k] - 1) + j] = j; end
    end
end
readall() = (s = 0; @inbounds @simd for i in eachindex(BUF); s += BUF[i]; end; s)
flush!() = (s = 0.0; @inbounds @simd for i in eachindex(FLUSH); s += FLUSH[i]; end; s)
function chase()
    p = 8(PERM[1] - 1) + 1; t = time_ns()
    @inbounds for _ in 1:nl; p = BUF[p]; end
    return (time_ns() - t) / nl, p
end
function stream()
    t = time_ns(); s = readall(); return 64nl / (time_ns() - t), s   # GB/s
end

# setup by owner O, then the measuring core C chases or streams
function trial(O, C, how, measure)
    on(O, build!)
    on(O, flush!)                       # dirty lines written back
    if how == :read
        on(O, readall); on(O, flush!)   # O re-reads: clean lines owned by O, evicted by O
    end
    on(C, flush!)                       # C's caches hold nothing of BUF
    return on(C, measure)[1]
end

function main()
    on(0, flush!); build!(); chase(); stream()   # compile
    println("cpu map: ", join([on(c, getcpu) for c in (0, 1, 8, 32)], " "), "  MB=", MB)
    for (label, measure) in (("chase ns/load", chase), ("stream GB/s", stream))
        for how in (:write, :read)
            for C in (0, 33)
                for O in unique((C, C ⊻ 1, C ⊻ 8, C ⊻ 16, C ⊻ 32))
                    v = [trial(O, C, how, measure) for _ in 1:3]
                    @printf("%-14s owner-%-5s C=%2d O=%2d  %7.2f  (%s)\n", label, how, C, O, minimum(v),
                            join([@sprintf("%.2f", x) for x in v], " "))
                    flush(stdout)
                end
            end
        end
    end
end
main()
