# Folds over points (M15) against the one-number reductions they replace,
# at the sizes of the downstream that asked for them.
#
#     TREEAMR_BENCH_BACKEND=cuda julia --project=<env> bench/fold.jl
#     TREEAMR_BENCH_BACKEND=metal TREEAMR_BENCH_T=Float32 julia --project=<env> bench/fold.jl
#     julia -t N --project=test bench/fold.jl            # cpu
#
# TreeGeneralizedHarmonic's analysis row (2026-10-22) made about 280
# `mesh_mapreduce`s, one number each, over a `diag` field set of 22 slots
# on 856 blocks of `32³`, vertex-centered with `G = 3`, on a quadrant
# `[0, 64]² × [−64, 64]` refined in five levels about the hole — and the
# fold was to make that one sweep, or three ("Folds over points" in
# CODE.md, and HISTORY.md for the row's measurement). The defaults here
# build a mesh of that shape: the same root brick, refined about the
# origin in `LEVELS` levels, with the same field set. What is timed:
#
# - `reduce`: one `mesh_mapreduce` of one slot, and `REDUCTIONS` of them,
#   which is what the row cost in reductions;
# - `fold shells`: the row's shells and half-shells, 18 bins of 7 numbers;
# - `fold levels`: the whole region, the boundary band and the levels;
# - `fold fused`: all of those in one accumulator, 25 bins of 7;
#
# each the best of `REPS`. The device must hold the field set: about 9 GB
# at the defaults; `TREEAMR_BENCH_N=16` is an eighth of that.
#
#     TREEAMR_BENCH_N           points per block edge (default 32)
#     TREEAMR_BENCH_LEVELS      refinement levels above the roots (default 3:
#                               842 blocks against the downstream's 856)
#     TREEAMR_BENCH_VARS        variables of the field set (default 22)
#     TREEAMR_BENCH_REDUCTIONS  one-number reductions in a row (default 280)
#     TREEAMR_BENCH_REPS        timed repetitions (default 5)
#     TREEAMR_BENCH_T           Float64 (default where the backend has it) or Float32
#     TREEAMR_BENCH_BACKEND     cpu (default), cuda, or metal

using TreeAMR
using TreeAMR: combine
using KernelAbstractions: CPU, supports_float64
using Printf: @printf

const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "32"))
const LEVELS = parse(Int, get(ENV, "TREEAMR_BENCH_LEVELS", "3"))
const NVARS = parse(Int, get(ENV, "TREEAMR_BENCH_VARS", "22"))
const REDUCTIONS = parse(Int, get(ENV, "TREEAMR_BENCH_REDUCTIONS", "280"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "5"))
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

function best(f, reps=REPS)
    f()
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f())
    end
    return t
end

# The quadrant's root brick, refined level by level about the origin,
# where the downstream's hole sits: each level refines the leaves whose
# center lies within half the previous level's radius.
function build_forest()
    forest = Forest((4, 4, 8); N=N, extents=((0.0, 64.0), (0.0, 64.0), (-64.0, 64.0)))
    radius = 48.0
    for _ in 1:LEVELS
        marked = filter(forest.leaves) do k
            ext = block_extent(forest, k)
            c = ntuple(d -> (ext[d][1] + ext[d][2]) / 2, 3)
            sqrt(sum(abs2, c)) < radius
        end
        for k in marked
            isleaf(forest, k) && refine!(forest, k)
        end
        balance!(forest)
        radius /= 2
    end
    return forest
end

# The slots of the downstream's `diag`: ℋ, ℳᵢ, C_a, the error, the mask.
const HAM, MOM, CGH, ERR, MASK = 1, 2, 5, 9, 10
const SLOTS = 1:10

# Seven numbers a bin: the count, Σ ℋ², max |ℋ|, Σ |ℳ|², Σ |C|², Σ err²,
# max |err|.
bin(::Type{R}) where {R} = (n = Sum(0), ham2 = Sum(zero(R)), hammax = Max(zero(R)),
                            mom2 = Sum(zero(R)), gauge2 = Sum(zero(R)),
                            err2 = Sum(zero(R)), errmax = Max(zero(R)))
@inline function contrib(u)
    ℋ, e = u[HAM], u[ERR]
    m = u[MOM]^2 + u[MOM + 1]^2 + u[MOM + 2]^2
    c = u[CGH]^2 + u[CGH + 1]^2 + u[CGH + 2]^2 + u[CGH + 3]^2
    return (n = Sum(1), ham2 = Sum(ℋ * ℋ), hammax = Max(abs(ℋ)), mom2 = Sum(m),
            gauge2 = Sum(c), err2 = Sum(e * e), errmax = Max(abs(e)))
end

# The shell index of radius `r`: 0 below the innermost edge.
@inline function shellindex(edges::NTuple{K}, r) where {K}
    k = 0
    for i in 1:(K - 1)
        edges[i] <= r && (k = i)
    end
    return k
end

struct Shells{R}
    edges::NTuple{7,R}
    center::NTuple{3,R}
    dir::NTuple{3,R}
end
function (s::Shells)(acc, p)
    iszero(p.u[MASK]) && return acc
    d = (p.x[1] - s.center[1], p.x[2] - s.center[2], p.x[3] - s.center[3])
    k = shellindex(s.edges, sqrt(d[1]^2 + d[2]^2 + d[3]^2))
    k == 0 && return acc
    q = contrib(p.u)
    w = d[1] * s.dir[1] + d[2] * s.dir[2] + d[3] * s.dir[3]
    return (shell = addto(acc.shell, k, q), trail = addto(acc.trail, w < 0 ? k : 0, q),
            lead = addto(acc.lead, w > 0 ? k : 0, q))
end

struct Levels end
function (::Levels)(acc, p)
    iszero(p.u[MASK]) && return acc
    q = contrib(p.u)
    return (all = combine(acc.all, q), band = p.tag ? combine(acc.band, q) : acc.band,
            lev = addto(acc.lev, p.level + 1, q))
end

struct Fused{R}
    shells::Shells{R}
end
(f::Fused)(acc, p) = (shells = f.shells(acc.shells, p), levels = Levels()(acc.levels, p))

function main()
    forest = build_forest()
    fs = FieldSet{T}(forest, NVARS; G=3, centering=vertexcentered(3), backend=BACKEND)
    fill_by_coordinates!((x, v) -> v == MASK ? one(x[1]) : sin(x[1] + v * x[2]) * x[3],
                         fs)
    npoints = nblocks(fs) * N^3
    @printf("# %s, %s: %d blocks of %d³ (%.2f M points), %d variables, %d levels\n",
            BNAME, T, nblocks(fs), N, npoints / 1e6, NVARS, maxlevel(forest) + 1)

    R = T
    edges = (R(1.5), R(2), R(2.25), R(3), R(5), R(8), R(Inf))
    shells = Shells{R}(edges, (zero(R), zero(R), zero(R)), (zero(R), zero(R), one(R)))
    sinit = (shell = ntuple(_ -> bin(R), 6), trail = ntuple(_ -> bin(R), 6),
             lead = ntuple(_ -> bin(R), 6))
    L = maxlevel(forest) + 1
    linit = (all = bin(R), band = bin(R), lev = ntuple(_ -> bin(R), L))
    band = k -> any(d -> block_extent(forest, k)[d][2] >= forest.extents[d][2], 1:3)

    report(name, t, k=1) = @printf("%-22s %10.3f ms  %8.3f ns/point\n", name, 1e3 * t,
                                   1e9 * t / (k * npoints))
    t1 = best(() -> mesh_mapreduce(x -> x * x, +, zero(R), fs; vars=HAM))
    report("reduce, one slot", t1)
    tk = best(() -> for i in 1:REDUCTIONS
                  v = 1 + i % length(SLOTS)
                  isodd(i) ? mesh_mapreduce(x -> x * x, +, zero(R), fs; vars=v) :
                  mesh_mapreduce(abs, max, zero(R), fs; vars=v)
              end, max(1, REPS ÷ 2))
    report("reduce, $REDUCTIONS slots", tk)
    @printf("%-22s %10d bytes\n", "  shell accumulator", sizeof(sinit))
    ts = best(() -> mesh_fold(shells, combine, sinit, fs; vars=SLOTS))
    report("fold shells", ts)
    tl = best(() -> mesh_fold(Levels(), combine, linit, fs; vars=SLOTS, tag=band))
    report("fold levels", tl)
    finit = (shells = sinit, levels = linit)
    @printf("%-22s %10d bytes\n", "  fused accumulator", sizeof(finit))
    tf = best(() -> mesh_fold(Fused(shells), combine, finit, fs; vars=SLOTS, tag=band))
    report("fold fused", tf)
    return nothing
end

main()
