# The ghost fill group by group: where a fill's time goes once the index
# is cheap ("The copy kernels on a device" in CODE.md).
#
#     TREEAMR_BENCH_BACKEND=cuda julia --project=<env> bench/copy_groups.jl
#
# Each transfer group of a uniform mesh's schedule is launched alone,
# three ways: as `fill_ghosts!` launches it on this backend (`flat` on a
# device), shaped (KernelAbstractions' own `NTuple` index, which is what
# every device launch was through 0.1.7), and for a copy through its
# weights instead of the copy path. Per group it prints the box, the
# number of transfers, the time of each form and the bandwidth of the
# first, counting a value read and a value written per ghost point and
# variable; then the whole fill, and the sum of the groups' times by the
# length of their box's innermost run, which is what the memory system
# sees.
#
# Sizes as in `bench/copies.jl`: `TREEAMR_BENCH_{D,N,ROOTS,VARS,G,
# CENTERING,P,REPS,T,BACKEND}`, defaults 3, 16, 8, 20, 3, vertex, 4, 20.

using TreeAMR
using KernelAbstractions: synchronize, CPU, supports_float64
using Printf: @printf

const D = parse(Int, get(ENV, "TREEAMR_BENCH_D", "3"))
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const ROOTS = parse(Int, get(ENV, "TREEAMR_BENCH_ROOTS", "8"))
const NVARS = parse(Int, get(ENV, "TREEAMR_BENCH_VARS", "20"))
const G = parse(Int, get(ENV, "TREEAMR_BENCH_G", "3"))
const CENTERING = Symbol(get(ENV, "TREEAMR_BENCH_CENTERING", "vertex"))
const P = parse(Int, get(ENV, "TREEAMR_BENCH_P", "4"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "20"))
const BNAME = lowercase(get(ENV, "TREEAMR_BENCH_BACKEND", "cpu"))

if BNAME == "cuda"
    using CUDA
elseif BNAME == "metal"
    using Metal
end
const BACKEND = BNAME == "cuda" ? CUDABackend() : BNAME == "metal" ? MetalBackend() : CPU()
const T = let want = get(ENV, "TREEAMR_BENCH_T", "")
    isempty(want) ? (supports_float64(BACKEND) ? Float64 : Float32) :
    want == "Float64" ? Float64 : Float32
end

function best(f, reps=REPS)
    f()
    synchronize(BACKEND)
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed (f(); synchronize(BACKEND)))
    end
    return t
end

# The group with every stencil marked not unit: the copy through its weights.
function weighted(g::TreeAMR.TransferGroup{TT,DD}) where {TT,DD}
    stencils = map(s -> TreeAMR.Stencil1D{TT}(s.targetfirst, s.srcstart, s.weights, false),
                   g.stencils)
    return TreeAMR.TransferGroup{TT,DD}(g.kind, stencils, g.targetblocks, g.sourceblocks,
                                        g.factorcol, g.orientation, g.plane)
end

function main()
    centering = CENTERING === :vertex ? vertexcentered(D) : cellcentered(D)
    forest = Forest{T}(ntuple(_ -> ROOTS, D); N=N, periodic=ntuple(_ -> true, D),
                       extents=ntuple(_ -> (zero(T), one(T)), D))
    fs = FieldSet(forest, NVARS; G=ntuple(_ -> G, D), centering=centering,
                  backend=BACKEND)
    schedule = GhostSchedule(fs, Operators(prolongation=P, restriction=P))
    fill_by_coordinates!((x, v) -> sin(2 * T(π) * x[1]) + T(v), fs)
    flat = TreeAMR.flatlaunch(BACKEND)
    launch(g, f) = () -> TreeAMR.run_group!(fs.work, fs.work, g, fs.nvars, BACKEND;
                                            flat=f, factors=fs.factors,
                                            rotvars=fs.rotvars)
    @printf("backend=%s T=%s D=%d N=%d G=%d %s blocks=%d vars=%d\n", BNAME, T, D, N, G,
            CENTERING, nblocks(fs), NVARS)
    @printf("# box\ttransfers\tunit\tus\tus shaped\tus weighted\tTB/s\n")
    byrun = Dict{Int,Tuple{Float64,Int}}()
    total = 0.0
    for g in Iterators.flatten((schedule.phase1, schedule.phase2...))
        box = TreeAMR.boxsize(g)
        unit = all(s -> s.unit, g.stencils)
        t = best(launch(g, flat))
        ts = best(launch(g, !flat))
        tw = unit ? best(launch(weighted(g), flat)) : NaN
        bytes = 2 * sizeof(T) * NVARS * prod(box) * TreeAMR.ntransfers(g)
        @printf("%s\t%d\t%s\t%.1f\t%.1f\t%.1f\t%.3f\n", join(box, "x"),
                TreeAMR.ntransfers(g), unit, 1e6t, 1e6ts, 1e6tw, bytes / t / 1e12)
        total += t
        tr, br = get(byrun, box[1], (0.0, 0))
        byrun[box[1]] = (tr + t, br + bytes)
    end
    tfill = best(() -> fill_ghosts!(fs, schedule))
    @printf("# groups one at a time %.1f us, the fill %.1f us\n", 1e6total, 1e6tfill)
    for run in sort!(collect(keys(byrun)))
        t, b = byrun[run]
        @printf("# innermost run %d: %.1f us, %.3f TB/s\n", run, 1e6t, b / t / 1e12)
    end
    return nothing
end

main()
