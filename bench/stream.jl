# A memory-stream microbenchmark with no mesh in it, to locate a limit
# that bench/threads.jl found on Symmetry (2026-09-23): one 64-thread
# process streams at a quarter of the rate that eight 8-thread
# processes reach together on the same node with the same page
# placement. Each mode below moves the same three arrays with a
# different launch mechanism, so the mechanisms can be told apart from
# the hardware and the process.
#
#     TREEAMR_STREAM_N=<elements> julia -t 64 --project=. bench/stream.jl
#
# Modes: `static` is `Threads.@threads :static` over per-thread chunks;
# `spawn` is one `Threads.@spawn` per chunk under `@sync`, which is what
# the KernelAbstractions CPU backend does; `ka` is the KernelAbstractions
# triad kernel of bench/threads.jl; `ka_static` is the same kernel with
# the backend's static scheduling.

using KernelAbstractions: @kernel, @index, @Const, CPU, synchronize
using Printf: @printf

const n = parse(Int, get(ENV, "TREEAMR_STREAM_N", string(2^26)))
const REPS = 20
const nt = Threads.nthreads()

chunks(n) = [(1 + (t - 1) * n ÷ nt):(t * n ÷ nt) for t in 1:nt]

function triad_range!(c, a, b, r)
    @inbounds @simd for i in r
        c[i] = a[i] + 2 * b[i]
    end
    return nothing
end

function static!(c, a, b)
    ch = chunks(length(c))
    Threads.@threads :static for t in 1:nt
        triad_range!(c, a, b, ch[t])
    end
    return nothing
end

function spawn!(c, a, b)
    ch = chunks(length(c))
    @sync for t in 1:nt
        Threads.@spawn triad_range!(c, a, b, ch[t])
    end
    return nothing
end

@kernel function triad_kernel!(c, @Const(a), @Const(b))
    i = @index(Global, Linear)
    c[i] = a[i] + 2 * b[i]
end

function ka!(c, a, b)
    triad_kernel!(CPU())(c, a, b; ndrange=length(c))
    synchronize(CPU())
    return nothing
end

function ka_static!(c, a, b)
    triad_kernel!(CPU(; static=true))(c, a, b; ndrange=length(c))
    synchronize(CPU(; static=true))
    return nothing
end

function best(f)
    f()
    t = Inf
    for _ in 1:REPS
        t = min(t, @elapsed f())
    end
    return t
end

function main()
    a = Vector{Float64}(undef, n)
    b = Vector{Float64}(undef, n)
    c = Vector{Float64}(undef, n)
    # First touch in the static partition, so that every mode below
    # finds the pages where the static one wrote them.
    static!(a, b, c)        # writes a
    static!(b, a, c)        # writes b
    static!(c, a, b)        # writes c
    @printf("threads=%d n=%d bytes_per_array=%.0f MB\n", nt, n, 8n / 1e6)
    for (name, f) in (("static", static!), ("spawn", spawn!), ("ka", ka!),
                      ("ka_static", ka_static!))
        t = best(() -> f(c, a, b))
        @printf("%d\t%s\t%.6f\t%.1f GB/s\n", nt, name, t, 3 * 8n / t / 1e9)
    end
    return nothing
end

main()
