# The stream of bench/stream.jl, instrumented to find out why one
# 64-thread process loses bandwidth that eight 8-thread processes keep
# (CODE.md, "Parallelism", "What one process loses: data-to-core
# affinity"; Symmetry, 2026-09-23).
#
#     TREEAMR_STREAM_N=<elements> TREEAMR_AFFINITY_START=<unix time> \
#         TREEAMR_AFFINITY_MODES=static,static_rot,spawn \
#         julia -t 64 --project=. bench/affinity.jl
#
# Mode k runs during the wall-clock window [START + k (WIN + GAP),
# + WIN) and reports bytes moved inside it over its length. Processes
# given the same START therefore really run each mode at the same time,
# which best-of-REPS timings of independent processes do not: a copy
# whose repetitions happen to fall while the others compile or idle
# reports bandwidth the node never delivered to all of them at once.
# That is how the first comparison came to credit eight processes with
# 385 GB/s; measured in windows they reach 224, as one process does.
#
# Every mode streams the same triad over the same three arrays in
# `nthreads()` equal chunks; they differ only in which thread gets which
# chunk, when, and through which launch path:
#
#     static          `@threads :static`, chunk t on thread t, every launch
#     static_rot      the same, but the chunk-to-thread map rotates by
#                     one every launch
#     static_alt<K>   chunk t alternates between thread t and t + K
#                     (pinned, K = 1 stays within a CCD, 8 crosses to the
#                     next one, 32 to the other socket)
#     static_scalar   `static` with a loop the compiler cannot vectorize
#     spawn           one `Threads.@spawn` per chunk under `@sync`
#     ka, ka_static   the KernelAbstractions CPU backend's own partition,
#                     default and static schedule, timed per task
#     persist         long-lived tasks spinning on a counter, no launch
#                     at all; persist_stagger<µs> staggers their starts
#
# Per mode it prints GB/s and, per chunk: wall time, the thread's CPU
# time over the same interval (equal means the thread was never
# descheduled), and involuntary context switches. TREEAMR_AFFINITY_PERF
# names modes to run `perf stat -e $TREEAMR_AFFINITY_PERFEV` against for
# their window (needs `kernel.perf_event_paranoid` <= 0).

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, CPU
using Printf

const n = parse(Int, get(ENV, "TREEAMR_STREAM_N", string(2^26)))
const nt = Threads.nthreads()
const START = parse(Float64, ENV["TREEAMR_AFFINITY_START"])
const WIN = parse(Float64, get(ENV, "TREEAMR_AFFINITY_WIN", "4"))
const GAP = parse(Float64, get(ENV, "TREEAMR_AFFINITY_GAP", "2"))
const MODES = split(get(ENV, "TREEAMR_AFFINITY_MODES", "static,static_rot,spawn,ka,ka_static"), ",")
const TAG = get(ENV, "TREEAMR_AFFINITY_TAG", "")
getcpu() = ccall(:sched_getcpu, Cint, ())

# Per-chunk recorders. ACC accumulates, per chunk slot and over one
# window: wall ns, thread CPU ns, involuntary switches, chunks.
const T0 = zeros(UInt64, 1024)
const ACC = zeros(Float64, 4, 1024)
const S0 = zeros(Int64, 2, 1024)
const RU = [zeros(Int64, 18) for _ in 1:1024]      # struct rusage, as longs
const TS = [zeros(Int64, 2) for _ in 1:1024]       # struct timespec
@inline function threadcpu(k)
    ts = TS[k]
    ccall(:clock_gettime, Cint, (Cint, Ptr{Int64}), 3, ts)   # CLOCK_THREAD_CPUTIME_ID
    return ts[1] * 1_000_000_000 + ts[2]
end
@inline function nivcsw(k)
    r = RU[k]
    ccall(:getrusage, Cint, (Cint, Ptr{Int64}), 1, r)        # RUSAGE_THREAD
    return r[18]
end
@inline function rec0(k)
    @inbounds S0[1, k] = threadcpu(k)
    @inbounds S0[2, k] = nivcsw(k)
    @inbounds T0[k] = time_ns()
    return nothing
end
@inline function rec1(k)
    t1 = time_ns()
    @inbounds begin
        ACC[1, k] += t1 - T0[k]
        ACC[2, k] += threadcpu(k) - S0[1, k]
        ACC[3, k] += nivcsw(k) - S0[2, k]
        ACC[4, k] += 1
    end
    return nothing
end

function triad_range!(c, a, b, r)
    @inbounds @simd for i in r
        c[i] = a[i] + 2 * b[i]
    end
    return nothing
end
function triad_scalar!(c, a, b, r)
    for i in r
        @inbounds c[i] = a[i] + 2 * b[i]
        Base.donotdelete(i)            # an opaque use per element: no SIMD
    end
    return nothing
end
chunks(n, k) = [(1 + (t - 1) * n ÷ k):(t * n ÷ k) for t in 1:k]

const LAUNCH = Ref(0)
# `shift(g)` maps launch g to an offset: thread t runs chunk t - shift.
function static!(c, a, b, ch, shift, body=triad_range!)
    s = shift(LAUNCH[] += 1)
    Threads.@threads :static for t in 1:nt
        k = mod(t - 1 - s, nt) + 1
        rec0(t); body(c, a, b, ch[k]); rec1(t)
    end
    return nothing
end
function spawn!(c, a, b, ch)
    @sync for t in 1:nt
        Threads.@spawn begin rec0(t); triad_range!(c, a, b, ch[t]); rec1(t) end
    end
    return nothing
end

@kernel function triad_kernel!(c, @Const(a), @Const(b))
    i = @index(Global, Linear)
    c[i] = a[i] + 2 * b[i]
end
# KernelAbstractions' own `__run` for the CPU (src/cpu.jl), with each
# task's loop over its workgroups wrapped in the recorders. The
# partition and the launch are the backend's, unchanged.
function ka!(c, a, b, static)
    obj = triad_kernel!(CPU(; static))
    ndrange, _, iterspace, dynamic = KernelAbstractions.launch_config(obj, length(c), nothing)
    len, rem = divrem(length(iterspace), nt)
    args = (c, a, b)
    go(tid) = KernelAbstractions.__thread_run(tid, len, rem, obj, ndrange, iterspace,
                                               args, dynamic)
    if static
        Threads.@threads :static for tid in 1:nt
            rec0(tid); go(tid); rec1(tid)
        end
    else
        @sync for tid in 1:nt
            Threads.@spawn begin rec0(tid); go(tid); rec1(tid) end
        end
    end
    return nothing
end

# Long-lived workers: no task is created or woken per launch. Slot 1
# drives: it waits for every chunk of launch g, then releases g + 1.
const GEN = Threads.Atomic{Int}(0)
const DONE = Threads.Atomic{Int}(0)
const STOP = Threads.Atomic{Bool}(false)
function persist!(c, a, b, ch, tend, stagger_ns)
    GEN[] = 1; DONE[] = 0; STOP[] = false
    launches = Ref(0)
    Threads.@threads :static for t in 1:nt
        g = 1
        while true
            while GEN[] < g && !STOP[]
                ccall(:jl_cpu_pause, Cvoid, ())
            end
            STOP[] && break
            t1 = time_ns() + (t - 1) * stagger_ns
            while time_ns() < t1 end
            rec0(t); triad_range!(c, a, b, ch[t]); rec1(t)
            Threads.atomic_add!(DONE, 1)
            if t == 1
                while DONE[] < g * nt
                    ccall(:jl_cpu_pause, Cvoid, ())
                end
                launches[] = g
                time() >= tend ? (STOP[] = true) : (GEN[] = g + 1)
            end
            g += 1
        end
    end
    return launches[]
end

function window(name, f, k)
    t0 = START + k * (WIN + GAP)
    tend = t0 + WIN
    while time() < t0
        sleep(0.001)
    end
    fill!(ACC, 0)
    perf = nothing
    if name in split(get(ENV, "TREEAMR_AFFINITY_PERF", ""), ",")
        ev = ENV["TREEAMR_AFFINITY_PERFEV"]
        perf = run(`perf stat -x, -e $ev -p $(getpid()) -o perf-$TAG-$name-$k.csv -- sleep $(WIN - 0.3)`;
                   wait=false)
    end
    w0 = time()
    count = 0
    if f isa Tuple                      # persist: runs its own launches
        count = f[1](tend)
    else
        while time() < tend
            f()
            count += 1
        end
    end
    w = time() - w0
    perf === nothing || wait(perf)
    L = sum(@view ACC[4, :])
    @printf("%s\t%s\t%d launches\t%.1f GB/s\tper chunk: wall %.2f ms, thread cpu %.0f%%, invol %.3f\n",
            TAG, name, count, count * 24n / w / 1e9, sum(@view ACC[1, :]) / L / 1e6,
            100 * sum(@view ACC[2, :]) / sum(@view ACC[1, :]), sum(@view ACC[3, :]) / L)
    flush(stdout)
    return nothing
end

function main()
    a = Vector{Float64}(undef, n)
    b = similar(a)
    c = similar(a)
    ch = chunks(n, nt)
    # Write every input before timing anything (see the zero-page note
    # in bench/threads.jl), in the `static` partition.
    for (x, y, z) in ((a, b, c), (b, a, c), (c, a, b))
        static!(x, y, z, ch, _ -> 0)
    end
    fs = Any[]
    for m in MODES
        f = if m == "static"
            () -> static!(c, a, b, ch, _ -> 0)
        elseif m == "static_rot"
            () -> static!(c, a, b, ch, g -> g)
        elseif startswith(m, "static_alt")
            K = parse(Int, m[11:end])
            () -> static!(c, a, b, ch, g -> isodd(g) ? K : 0)
        elseif m == "static_scalar"
            () -> static!(c, a, b, ch, _ -> 0, triad_scalar!)
        elseif m == "spawn"
            () -> spawn!(c, a, b, ch)
        elseif m == "ka"
            () -> ka!(c, a, b, false)
        elseif m == "ka_static"
            () -> ka!(c, a, b, true)
        elseif startswith(m, "persist")
            s = m == "persist" ? 0 : parse(Int, m[16:end])     # persist_stagger<µs>
            (tend -> persist!(c, a, b, ch, tend, 1000s),)
        else
            error("unknown mode $m")
        end
        f isa Tuple ? f[1](time()) : f()               # compile outside the windows
        push!(fs, f)
    end
    time() > START && println(stderr, "WARNING: setup ran past START; the windows are late")
    cpu = zeros(Int, nt)
    Threads.@threads :static for t in 1:nt
        cpu[t] = getcpu()
    end
    println(TAG, "\tthreads=", nt, " n=", n, " cpu of static slot t: ", join(cpu, " "))
    for (k, m) in enumerate(MODES)
        window(m, fs[k], k - 1)
    end
    return nothing
end

main()
