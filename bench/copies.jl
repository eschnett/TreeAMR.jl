# The copy kernels around an application's right-hand side: `scatter!`,
# `gather!` and `fill_ghosts!`, per owned point and as a fraction of the
# memory bandwidth.
#
#     TREEAMR_BENCH_BACKEND=cuda julia --project=<env> bench/copies.jl
#     TREEAMR_BENCH_BACKEND=metal julia --project=<env> bench/copies.jl
#     julia -t N --project=. bench/copies.jl            # cpu
#
# A downstream application (TreeGeneralizedHarmonic, 2026-10-05) measured
# these at 0.65–0.96 TB/s on an H200 against its 4.8 TB/s, once its own
# kernel had become fast enough for them to be 40 % of an evaluation
# ("The copy kernels on a device" in CODE.md). Its sizes are the
# defaults here: a uniform periodic mesh, vertex-centered, `G = 3`, 20
# variables, `Float64`, 512 blocks of `16³` — `TREEAMR_BENCH_N=32` and
# `TREEAMR_BENCH_ROOTS=2 TREEAMR_BENCH_N=128` for its other two rows.
#
# Bandwidth is counted as the bytes a copy must move: a value read and a
# value written per owned point and variable for the scatter and the
# gather, and per ghost point and variable written for the fill
# (a prolongation reads more than one value, so on a refined mesh the
# fill's figure is a lower bound). `copy_floor` is a plain linear copy
# of the scatter's bytes, the most any of them can reach.
#
#     TREEAMR_BENCH_D          dimension (default 3)
#     TREEAMR_BENCH_N          cells per block edge (default 16)
#     TREEAMR_BENCH_ROOTS      roots per edge (default 8)
#     TREEAMR_BENCH_VARS       variables (default 20)
#     TREEAMR_BENCH_G          ghost width (default 3)
#     TREEAMR_BENCH_CENTERING  vertex (default) or cell
#     TREEAMR_BENCH_LEVELS     1 (default, uniform) or 2 (middle half refined)
#     TREEAMR_BENCH_P          operator order, both operators (default 4)
#     TREEAMR_BENCH_REPS       timed repetitions (default 20)
#     TREEAMR_BENCH_T          Float64 (default where the backend has it) or Float32
#     TREEAMR_BENCH_BACKEND    cpu (default), cuda, or metal

using TreeAMR
using KernelAbstractions: @kernel, @index, @Const, synchronize, CPU, supports_float64
using Printf: @printf

const D = parse(Int, get(ENV, "TREEAMR_BENCH_D", "3"))
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const ROOTS = parse(Int, get(ENV, "TREEAMR_BENCH_ROOTS", "8"))
const NVARS = parse(Int, get(ENV, "TREEAMR_BENCH_VARS", "20"))
const G = parse(Int, get(ENV, "TREEAMR_BENCH_G", "3"))
const CENTERING = Symbol(get(ENV, "TREEAMR_BENCH_CENTERING", "vertex"))
const LEVELS = parse(Int, get(ENV, "TREEAMR_BENCH_LEVELS", "1"))
const P = parse(Int, get(ENV, "TREEAMR_BENCH_P", "4"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "20"))
const BNAME = lowercase(get(ENV, "TREEAMR_BENCH_BACKEND", "cpu"))

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

@kernel function copy_kernel!(dst, @Const(src))
    i = @index(Global, Linear)
    @inbounds dst[i] = src[i]
end

function best(f, reps=REPS)
    f()
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f())
    end
    return t
end

function main()
    centering = CENTERING === :vertex ? vertexcentered(D) :
                CENTERING === :cell ? cellcentered(D) :
                error("TREEAMR_BENCH_CENTERING must be vertex or cell")
    forest = Forest{T}(ntuple(_ -> ROOTS, D); N=N, periodic=ntuple(_ -> true, D),
                       extents=ntuple(_ -> (zero(T), one(T)), D))
    if LEVELS == 2
        refine!(forest, filter(forest.leaves) do k
            ext = block_extent(forest, k)
            all(d -> T(0.25) < (ext[d][1] + ext[d][2]) / 2 < T(0.75), 1:D)
        end)
        balance!(forest)
    end
    fs = FieldSet(forest, NVARS; G=ntuple(_ -> G, D), centering=centering,
                  backend=BACKEND)
    ops = Operators(prolongation=P, restriction=P)
    schedule = GhostSchedule(fs, ops)
    fill_by_coordinates!((x, v) -> sin(2 * T(π) * x[1]) + T(v), fs)
    u = statevector(fs)
    gather!(u, fs)

    t_scatter = best(() -> scatter!(fs, u))
    t_gather = best(() -> gather!(u, fs))
    t_fill = best(() -> fill_ghosts!(fs, schedule))
    v = similar(u)
    t_floor = best() do
        copy_kernel!(BACKEND)(v, u; ndrange=length(u))
        synchronize(BACKEND)
    end

    owned = nblocks(fs) * N^D
    groups = Iterators.flatten((schedule.phase1, schedule.phase2...))
    ghosts = sum(g -> prod(TreeAMR.boxsize(g)) * TreeAMR.ntransfers(g), groups; init=0)
    nbytes = 2 * sizeof(T) * NVARS
    @printf("backend=%s T=%s threads=%d D=%d N=%d G=%d %s blocks=%d vars=%d levels=%d\n",
            BNAME, T, Threads.nthreads(), D, N, G, CENTERING, nblocks(fs), NVARS, LEVELS)
    @printf("# phase\tseconds\tns per owned point\tTB/s\n")
    for (name, t, points) in (("scatter", t_scatter, owned), ("gather", t_gather, owned),
                              ("fill_ghosts", t_fill, ghosts),
                              ("copy_floor", t_floor, owned))
        @printf("%s\t%.6f\t%.4f\t%.3f\n", name, t, 1e9 * t / owned,
                points * nbytes / t / 1e12)
    end
    @printf("# %d ghost points written by %d transfer groups\n", ghosts,
            count(_ -> true, groups))
    return nothing
end

main()
