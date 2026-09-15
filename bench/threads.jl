# Thread-scaling measurement for M5.
#
#     julia -t N --project=. bench/threads.jl
#
# Prints one tab-separated line per phase for the thread count Julia was
# started with; run it once per thread count and compare. `bench/scan.sh`
# does that and formats the result.
#
# What is measured is the per-evaluation path an application actually
# pays for — scatter, ghost fill, RHS kernel — plus the two
# regrid-frequency host-side passes (building the schedule, regridding),
# because those are the parts that would otherwise cap the speedup by
# Amdahl's law.
#
# Sizes come from the environment so the same script serves a laptop and
# a compute node:
#
#     TREEAMR_BENCH_D      dimension (default 3)
#     TREEAMR_BENCH_N      cells per block edge (default 16)
#     TREEAMR_BENCH_ROOTS  roots per edge (default 4)
#     TREEAMR_BENCH_REPS   timed repetitions (default 20)

using TreeAMR
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

# Best of `REPS`, after a warm-up call: the minimum is what a scaling
# study wants, since noise only ever adds time.
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

    # Refine a slab, so the regrid really moves blocks around.
    flags = flag_blocks(forest) do b, k
        ext = block_extent(forest, k)
        level(k) == 0 && ext[1][1] < 0.25 ? Refine : Keep
    end

    t_rhs = best(rhs!)
    t_ghosts = best(() -> fill_ghosts!(fs, schedule))
    t_scatter = best(() -> scatter!(fs, u))
    t_fill = best(() -> fill_by_coordinates!((x, v) -> sin(2π * x[1]) + v, fs))
    t_norm = best(() -> volume_weighted_norm(fs, u))
    t_schedule = best(() -> GhostSchedule(fs, OPS), max(3, REPS ÷ 4))
    t_marks = best(() -> complete_marks(forest, flags), max(3, REPS ÷ 4))

    # The bandwidth reference, on arrays the size of the working array.
    backend = get_backend(fs.work)
    n = length(fs.work)
    a, b, c = (similar(fs.work, n) for _ in 1:3)
    triad!() = begin
        triad_kernel!(backend)(c, a, b; ndrange=n)
        synchronize(backend)
    end
    triad!()                                     # also the first touch
    t_triad = best(triad!)

    cells = nleaves(forest) * N^D
    @printf("threads=%d D=%d N=%d roots=%d blocks=%d cells=%d\n",
            Threads.nthreads(), D, N, ROOTS, nleaves(forest), cells)
    for (name, t) in (("rhs", t_rhs), ("fill_ghosts", t_ghosts), ("scatter", t_scatter),
                      ("fill_by_coordinates", t_fill), ("norm", t_norm),
                      ("ghost_schedule", t_schedule), ("complete_marks", t_marks),
                      ("triad_reference", t_triad))
        @printf("%d\t%s\t%.6f\n", Threads.nthreads(), name, t)
    end
    @printf("# triad %.1f GB/s at %d thread(s)\n", 3 * n * 8 / t_triad / 1e9,
            Threads.nthreads())
    return nothing
end

main()
