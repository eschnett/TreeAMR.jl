# Device kernel benchmarks for M6.
#
#     TREEAMR_BENCH_BACKEND=cuda  julia --project=. bench/gpu.jl
#     TREEAMR_BENCH_BACKEND=metal julia --project=. bench/gpu.jl
#     julia -t N --project=. bench/gpu.jl            # cpu, for the comparison
#
# The same phases `bench/threads.jl` times, in the same tab-separated
# format, so a device run and a CPU run can be read side by side. The
# axis here is the *backend*, not the thread count, which is why this is
# a second script rather than another loop in `bench/scan.sh`: a device
# run has one process and one launch queue.
#
# Two phases are new relative to the thread benchmark, both because they
# are the per-evaluation work M6 moved rather than merely retargeted:
# `firing_boxes` (the device flagging sweep) and `regrid_transfer` (the
# transfer itself, separated from the host driver logic around it).
#
#     TREEAMR_BENCH_D        dimension (default 3)
#     TREEAMR_BENCH_N        cells per block edge (default 16)
#     TREEAMR_BENCH_ROOTS    roots per edge (default 4)
#     TREEAMR_BENCH_REPS     timed repetitions (default 20)
#     TREEAMR_BENCH_T        element type, Float64 or Float32 (default:
#                            Float64, or Float32 where the backend has no
#                            hardware fp64)
#     TREEAMR_BENCH_BACKEND  cpu (default), cuda, or metal

using TreeAMR
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize, CPU,
                          supports_float64
using Printf: @printf

const D = parse(Int, get(ENV, "TREEAMR_BENCH_D", "3"))
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const ROOTS = parse(Int, get(ENV, "TREEAMR_BENCH_ROOTS", "4"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "20"))
const G = 2
const OPS = Operators(prolongation=4, restriction=4)

const BNAME = lowercase(get(ENV, "TREEAMR_BENCH_BACKEND", "cpu"))

# At top level, in its own statement, so that everything after it is
# compiled in a world that can see the package.
if BNAME == "cuda"
    using CUDA
elseif BNAME == "metal"
    using Metal
elseif BNAME != "cpu"
    error("TREEAMR_BENCH_BACKEND must be cpu, cuda, or metal; got \"$BNAME\"")
end

const BACKEND =
    BNAME == "cuda" ? (CUDA.functional() ? CUDABackend() :
                       error("CUDA is not functional here")) :
    BNAME == "metal" ? (Metal.functional() ? MetalBackend() :
                        error("Metal is not functional here")) :
    CPU()
const T = let want = get(ENV, "TREEAMR_BENCH_T", "")
    isempty(want) ? (supports_float64(BACKEND) ? Float64 : Float32) :
    want == "Float64" ? Float64 : want == "Float32" ? Float32 :
    error("TREEAMR_BENCH_T must be Float64 or Float32, got \"$want\"")
end

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

# A bandwidth reference with no mesh structure at all, so the mesh
# phases can be read as a fraction of what the hardware delivers.
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

"""A two-level mesh: the middle half of the domain refined once."""
function build_forest(::Val{DD}) where {DD}
    forest = Forest(ntuple(_ -> ROOTS, DD); N=N,
                    periodic=ntuple(_ -> true, DD),
                    extents=ntuple(_ -> (zero(T), one(T)), DD))
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> T(0.25) < (ext[d][1] + ext[d][2]) / 2 < T(0.75), 1:DD)
    end
    refine!(forest, targets)
    balance!(forest)
    return forest
end

# Best of `REPS`, after a warm-up call. The warm-up matters far more on
# a device than on a host: the first call pays for kernel compilation.
# Every timed phase ends in a `synchronize`, which is what makes host
# wall-clock the right thing to measure.
function best(f, reps=REPS)
    f()
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f())
    end
    return t
end

function main()
    forest = build_forest(Val(D))
    fs = FieldSet{T}(forest, 2; G=G, backend=BACKEND)
    schedule = GhostSchedule(fs, OPS)
    spacings = let h = block_spacings(forest, T)
        BACKEND isa CPU ? h : (dev = similar(fs.work, T, length(h)); copyto!(dev, h); dev)
    end
    initial = (x, v) -> sin(2 * T(π) * x[1]) + T(v)
    fill_by_coordinates!(initial, fs)

    u = statevector(fs)
    gather!(u, fs)
    du = similar(u)
    dua = statearray(du, fs)

    rhs!() = begin
        scatter!(fs, u)
        fill_ghosts!(fs, schedule)
        map_blocks!(bench_rhs_kernel!, fs, dua, fs.work, spacings, Val(D), Val(fs.G))
    end

    # The device flagging sweep, over a threshold that fires somewhere.
    thr = T(0.5)
    fires(work, idx, b, x) = work[idx..., 1, b] > thr

    t_rhs = best(rhs!)
    t_ghosts = best(() -> fill_ghosts!(fs, schedule))
    t_scatter = best(() -> scatter!(fs, u))
    t_fill = best(() -> fill_by_coordinates!(initial, fs))
    t_norm = best(() -> volume_weighted_norm(fs, u))
    t_flag = best(() -> firing_boxes(fires, fs), max(3, REPS ÷ 4))
    t_schedule = best(() -> GhostSchedule(fs, OPS),
                      max(3, REPS ÷ 4))

    # The regrid transfer, measured on its own: a fresh array plus the
    # move, with the host driver logic (marks, balance, key rebuild)
    # excluded, since that is not what a backend changes.
    oldleaves = copy(forest.leaves)
    groups = TreeAMR.transfer_groups(T, forest, fs.G, staggers(fs),
                                     oldleaves, oldleaves, OPS, BACKEND)
    plan = TreeAMR.phase_plan(groups)
    fresh = similar(fs.work)
    transfer!() = begin
        TreeAMR.run_phase!(fresh, fs.work, groups, plan, fs.nvars, BACKEND)
        synchronize(BACKEND)
    end
    t_transfer = best(transfer!)

    n = length(fs.work)
    a, b, c = (similar(fs.work, n) for _ in 1:3)
    triad!() = begin
        triad_kernel!(BACKEND)(c, a, b; ndrange=n)
        synchronize(BACKEND)
    end
    fill_kernel!(BACKEND)(a, one(eltype(a)); ndrange=n)      # the first touch, in
    fill_kernel!(BACKEND)(b, one(eltype(a)); ndrange=n)      # the partition of the
    triad!()                                     # kernel that reads them
    t_triad = best(triad!)

    cells = nleaves(forest) * N^D
    @printf("backend=%s T=%s threads=%d D=%d N=%d roots=%d blocks=%d cells=%d\n",
            BNAME, T, Threads.nthreads(), D, N, ROOTS, nleaves(forest), cells)
    for (name, t) in (("rhs", t_rhs), ("fill_ghosts", t_ghosts), ("scatter", t_scatter),
                      ("fill_by_coordinates", t_fill), ("norm", t_norm),
                      ("firing_boxes", t_flag), ("regrid_transfer", t_transfer),
                      ("ghost_schedule", t_schedule), ("triad_reference", t_triad))
        @printf("%s\t%s\t%.6f\n", BNAME, name, t)
    end
    @printf("# triad %.1f GB/s on %s in %s\n", 3 * n * sizeof(T) / t_triad / 1e9,
            BNAME, T)
    return nothing
end

main()
