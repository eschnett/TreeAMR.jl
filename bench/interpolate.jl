# Point interpolation benchmark (M11).
#
#     julia -t N --project=. bench/interpolate.jl
#     TREEAMR_BENCH_BACKEND=cuda julia --project=<env with CUDA> bench/interpolate.jl
#
# The horizon finder's workload, scaled: a 3D vertex-centered field set
# with `G = 2` and 20 variables over three levels refined around the
# center, and batches of points in a shell there, interpolated with
# `Lagrange(4)`, value and gradient, with an excluded ball. The smallest
# batch is the finder's own (496 points, `EquiangularGrid(15)`); the
# larger ones show where launch overhead stops mattering and what the
# per-point cost is once it has.
#
# Printed, tab separated, one line per batch size:
#
#     backend  threads  T  npts  seconds  ns/point  bytes/batch
#
# `seconds` is the best of `REPS` whole `interpolate!` calls — uploads of
# the leaves and the geometry, the launch, the synchronization and the
# host-side check of the block indices included, since that is what a
# caller pays. `bytes/batch` is what `@allocated` sees on the host; on a
# device it counts only the host side. One line more times
# `locate_point` alone, on the host.
#
#     TREEAMR_BENCH_N        cells per block edge (default 16)
#     TREEAMR_BENCH_ROOTS    roots per edge (default 4)
#     TREEAMR_BENCH_NVARS    variables (default 20, TreeGH's state)
#     TREEAMR_BENCH_NPTS     comma-separated batch sizes (default 496,4960,49600)
#     TREEAMR_BENCH_REPS     timed repetitions (default 50)
#     TREEAMR_BENCH_T        Float64 or Float32 (default Float64, or Float32
#                            where the backend has no hardware fp64)
#     TREEAMR_BENCH_BACKEND  cpu (default), cuda, or metal

using TreeAMR
using KernelAbstractions: CPU, supports_float64, allocate
using Printf: @printf

const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const ROOTS = parse(Int, get(ENV, "TREEAMR_BENCH_ROOTS", "4"))
const NVARS = parse(Int, get(ENV, "TREEAMR_BENCH_NVARS", "20"))
const NPTS = parse.(Int, split(get(ENV, "TREEAMR_BENCH_NPTS", "496,4960,49600"), ","))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "50"))
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

# The domain is [-L, L]³ with L = 2·ROOTS, refined twice around the
# center, so a shell of radius L/4 runs through all three levels.
function setup()
    L = T(2 * ROOTS)
    forest = Forest{T}(ntuple(_ -> ROOTS, 3); N=N, extents=ntuple(_ -> (-L, L), 3))
    for (lvl, r) in ((0, L / 2), (1, L / 3))
        targets = filter(forest.leaves) do k
            ext = block_extent(forest, k)
            c = ntuple(d -> (ext[d][1] + ext[d][2]) / 2, 3)
            level(k) == lvl && sqrt(sum(abs2, c)) < r
        end
        refine!(forest, targets)
        balance!(forest)
    end
    fs = FieldSet{T}(forest, NVARS; G=2, centering=vertexcentered(3), backend=BACKEND)
    # Written on the host side of the data only through the public API,
    # so the device run fills on the device.
    fill_by_coordinates!((x, v) -> sin(x[1] / 3 + v) * cos(x[2] / 5) + x[3] / 7, fs)
    fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=4, restriction=4));
                 boundary=boundary_by_coordinates((x, v) -> sin(x[1] / 3 + v) *
                                                            cos(x[2] / 5) + x[3] / 7))
    return forest, fs, L
end

# `npts` points in a shell of radius `L/4 ± h`, a deterministic
# low-discrepancy sequence so every run and backend sees the same ones.
function shell(npts, L)
    φ = (sqrt(5.0) - 1) / 2
    return map(1:npts) do j
        z = 1 - 2 * mod(j * φ, 1.0)
        a = 2π * mod(j * φ^2, 1.0)
        r = L / 4 * (1 + T(mod(j * 0.7548776662466927, 1.0) - 0.5) / 16)
        s = sqrt(1 - z^2)
        (T(r * s * cos(a)), T(r * s * sin(a)), T(r * z))
    end
end

function main()
    forest, fs, L = setup()
    derivs = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    ball = Ellipsoid(ntuple(_ -> T(0), 3), ntuple(_ -> L / 5, 3))
    basis = Lagrange(4)
    println("# leaves=", nleaves(forest), " maxlevel=", maxlevel(forest), " N=", N,
            " nvars=", NVARS, " work=", round(sizeof(fs.work) / 2^20; digits=1), " MiB")
    for npts in NPTS
        xs = TreeAMR.todevice(BACKEND, shell(npts, L))
        vals = allocate(BACKEND, T, (NVARS, length(derivs), npts))
        exc = allocate(BACKEND, Bool, npts)
        run() = interpolate!(vals, exc, fs, xs, basis; derivs=derivs, exclude=ball)
        run()
        run()
        bytes = @allocated run()
        t = minimum(@elapsed(run()) for _ in 1:REPS)
        @printf("%s\t%d\t%s\t%d\t%.6e\t%.1f\t%d\n", BNAME, Threads.nthreads(), T, npts,
                t, t / npts * 1e9, bytes)
    end
    hs = shell(first(NPTS), L)
    foreach(x -> locate_point(forest, x), hs)
    tl = minimum(@elapsed(foreach(x -> locate_point(forest, x), hs)) for _ in 1:REPS)
    @printf("# locate_point on the host: %.1f ns per point\n", tl / length(hs) * 1e9)
    return nothing
end

# Run when executed, not when included (the comparison against a
# downstream stopgap reuses `setup` and `shell`).
abspath(PROGRAM_FILE) == @__FILE__() && main()
