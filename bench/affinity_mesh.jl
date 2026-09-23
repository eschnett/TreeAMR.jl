# The per-evaluation phases of bench/threads.jl, timed in synchronized
# wall-clock windows (see bench/affinity.jl for why), under two launch
# policies — the measurement behind "What one process loses: data-to-core
# affinity" in CODE.md (Symmetry, 2026-09-23).
#
#     TREEAMR_AFFINITY_START=<unix time> TREEAMR_AFFINITY_VARIANT=owner \
#         TREEAMR_BENCH_N=32 TREEAMR_BENCH_ROOTS=8 \
#         julia -t 64 --project=. bench/affinity_mesh.jl
#
#     owner  the package as it is: every per-block pass runs each block
#            on the thread that owns it (`launch_by_owner!`,
#            `threaded_chunks`, the ghost fill by owner)
#     spawn  the control: the same partitions, but every launch on
#            KernelAbstractions' default CPU schedule and every host
#            chunk on its own `Threads.@spawn`, so a block runs on
#            whichever thread is free — the placement the package had
#            before the ownership policy
#
# The control is installed below by overwriting two package methods at
# load time. Each phase prints: tag, threads, phase, cells, mean seconds
# per call over the window, best seconds, calls.

using TreeAMR
import KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
using Printf: @printf

const D = parse(Int, get(ENV, "TREEAMR_BENCH_D", "3"))
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const ROOTS = parse(Int, get(ENV, "TREEAMR_BENCH_ROOTS", "4"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "20"))
const G = 2
const OPS = Operators(prolongation=4, restriction=4)

@kernel function bench_rhs_kernel!(du, @Const(work), @Const(spacings),
                                   ::Val{DD}, ::Val{GG}) where {DD,GG}
    I = @index(Global, NTuple)
    b = I[DD + 1]
    c = ntuple(d -> I[d] + GG[d], Val(DD))
    u0 = work[c..., 1, b]
    laplacian = zero(eltype(du))
    for d in 1:DD
        up = Base.setindex(c, c[d] + 1, d)
        um = Base.setindex(c, c[d] - 1, d)
        laplacian += work[up..., 1, b] - 2 * u0 + work[um..., 1, b]
    end
    h = spacings[b]
    du[ntuple(d -> I[d], Val(DD))..., 1, b] = work[c..., 2, b]
    du[ntuple(d -> I[d], Val(DD))..., 2, b] = laplacian / (h * h)
end

# A bandwidth reference with no mesh structure at all: the same
# KernelAbstractions launch machinery over one flat array, so the mesh
# phases can be read as a fraction of what the node actually delivers.
@kernel function triad_kernel!(c, @Const(a), @Const(b))
    i = @index(Global, Linear)
    c[i] = a[i] + 2 * b[i]
end

# The inputs must be written before they are read. An untouched
# allocation on Linux is backed by the kernel's single shared zero page,
# so reading it costs nothing and a "triad" over unwritten `a` and `b`
# is a write stream that reports three times its bandwidth (found on
# Symmetry, 2026-09-23, when eight NUMA domains of two DDR4 channels
# each reported 130 GB/s apiece).
@kernel function fill_kernel!(x, v)
    i = @index(Global, Linear)
    x[i] = v
end

"""A two-level mesh: the middle eighth of the domain refined once."""
function build_forest(::Val{DD}) where {DD}
    forest = Forest(ntuple(_ -> ROOTS, DD); N=N,
                    periodic=ntuple(_ -> true, DD),
                    extents=ntuple(_ -> (0.0, 1.0), DD))
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> 0.25 < (ext[d][1] + ext[d][2]) / 2 < 0.75, 1:DD)
    end
    refine!(forest, targets)
    balance!(forest)
    return forest
end


const START = parse(Float64, ENV["TREEAMR_AFFINITY_START"])
const WIN = parse(Float64, get(ENV, "TREEAMR_AFFINITY_WIN", "5"))
const GAP = parse(Float64, get(ENV, "TREEAMR_AFFINITY_GAP", "2"))
const TAG = get(ENV, "TREEAMR_AFFINITY_TAG", "")
const VARIANT = get(ENV, "TREEAMR_AFFINITY_VARIANT", "owner")

if VARIANT == "spawn"
    @eval TreeAMR function launch_by_owner!(kernel, backend::KernelAbstractions.CPU,
                                            args...; ndrange)
        kernel(backend)(args...; ndrange=ndrange)
        return nothing
    end
    @eval TreeAMR function threaded_chunks(f, n::Integer)
        @sync for (c, range) in enumerate(threadchunks(n))
            Threads.@spawn f(c, range)
        end
        return nothing
    end
elseif VARIANT != "owner"
    error("TREEAMR_AFFINITY_VARIANT is `owner` or `spawn`, not $VARIANT")
end

function triad_static!(c, a, b)
    n = length(c)
    nt = Threads.nthreads()
    Threads.@threads :static for t in 1:nt
        @inbounds @simd for i in (1 + (t - 1) * n ÷ nt):(t * n ÷ nt)
            c[i] = a[i] + 2 * b[i]
        end
    end
    return nothing
end

function main()
    forest = build_forest(Val(D))
    fs = FieldSet(forest, 2; G=G)
    schedule = GhostSchedule(fs, OPS)
    spacings = block_spacings(forest)
    fill_by_coordinates!((x, v) -> sin(2π * x[1]) + v, fs)
    u = statevector(fs)
    gather!(u, fs)
    du = similar(u)
    dua = statearray(du, fs)
    rhs!() = begin
        scatter!(fs, u)
        fill_ghosts!(fs, schedule)
        map_blocks!(bench_rhs_kernel!, fs, dua, fs.work, spacings, Val(D), Val(fs.G))
    end
    backend = get_backend(fs.work)
    n = length(fs.work)
    a, b, c = (similar(fs.work, n) for _ in 1:3)
    fill_kernel!(backend)(a, one(eltype(a)); ndrange=n)
    fill_kernel!(backend)(b, one(eltype(a)); ndrange=n)
    triad!() = (triad_kernel!(backend)(c, a, b; ndrange=n); synchronize(backend))
    phases = (("rhs", rhs!),
              ("fill_ghosts", () -> fill_ghosts!(fs, schedule)),
              ("scatter", () -> scatter!(fs, u)),
              ("fill_by_coordinates",
               () -> fill_by_coordinates!((x, v) -> sin(2π * x[1]) + v, fs)),
              ("map_blocks", () -> map_blocks!(bench_rhs_kernel!, fs, dua, fs.work,
                                               spacings, Val(D), Val(fs.G))),
              ("triad_ka", triad!),
              ("triad_static", () -> triad_static!(c, a, b)))
    for (_, f) in phases
        f(); f()
    end
    time() > START && println(stderr, "WARNING: setup ran past START; the windows are late")
    println(stderr, "variant ", VARIANT, ", backend ", get_backend(fs.work))
    cells = nleaves(forest) * N^D
    for (k, (name, f)) in enumerate(phases)
        t0 = START + (k - 1) * (WIN + GAP)
        while time() < t0
            sleep(0.001)
        end
        calls = 0
        tbest = Inf
        w0 = time()
        while time() < t0 + WIN
            t = @elapsed f()
            calls += 1
            tbest = min(tbest, t)
        end
        w = time() - w0
        @printf("%s\t%d\t%s\t%d\t%.6f\t%.6f\t%d\n", TAG * VARIANT, Threads.nthreads(),
                name, cells, w / calls, tbest, calls)
        flush(stdout)
    end
    return nothing
end

main()
