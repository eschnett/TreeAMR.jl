# Checkpoint throughput (M9a).
#
#     julia -t N --project=<env> bench/checkpoint.jl
#
# Times `save_checkpoint` and `load_checkpoint` on two states over one
# refined mesh, for each HDF5 filter setting available, and prints one
# line per (data set, filter): the state's size, the file's, the
# compression ratio, and the throughput in GB/s of state data (1 GB =
# 1e9 bytes of owned points) for a save, a save followed by a flush to
# stable storage, and a load. The numbers in `CODE.md` ("Throughput and
# filters") are the defaults at `-t 1` and `-t 6`.
#
# The environment needs TreeAMR and HDF5; the test environment has both
# (`--project=test`). The filter packages H5Zzstd, H5Zlz4, H5Zbitshuffle
# and H5Zblosc are used where they can be loaded — from the active
# environment, or from the default one Julia stacks under it — and
# skipped otherwise. They are not test dependencies, so the numbers with
# them come from a scratch environment that develops this checkout and
# adds them:
#
#     julia --project=<scratch> -e 'using Pkg; Pkg.develop(path="<this checkout>");
#         Pkg.add(["HDF5", "H5Zzstd", "H5Zlz4", "H5Zbitshuffle", "H5Zblosc"])'
#
# The two states are the two kinds of data the downstream applications
# checkpoint. The mesh is refined twice around the sphere `r = 1/2` in
# `[-1, 1]³`, and both states have their feature there:
#
# - `pulse`: a smooth outgoing spherical wave pulse, `(u, ∂ₜu)`,
#   vertex-centered, as TreeWave and TreeGeneralizedHarmonic evolve —
#   every value different, the low mantissa bits noise;
# - `blast`: a blast wave in the conserved variables of ideal
#   hydrodynamics, `(ρ, ρv, E)`, cell-centered, as TreeHydro evolves — a
#   smooth interior behind a discontinuity at the shell and a uniform,
#   bit-for-bit constant atmosphere outside it, which is what a filter
#   can exploit.
#
# Each time is the best of `REPS` calls after a first one that compiles,
# with a garbage collection before each call so that none runs inside
# one. What the numbers are and are not:
#
# - **A save ends in the page cache.** HDF5 closes the file without
#   syncing it, so `save` is the rate at which the data reach the
#   operating system; `sync` adds the flush that writes them to stable
#   storage (`fsync`, or `F_FULLFSYNC` on macOS, below), which is what a
#   job about to hit its wall-clock limit needs.
# - **A load right after a save reads from the page cache**, since the
#   file was just written, so `load` is decompression and memory
#   bandwidth, not the device's read rate. A cold-cache load needs the
#   cache dropped first (`purge` on macOS, `/proc/sys/vm/drop_caches` on
#   Linux, both root-only), or a file larger than the node's memory.
# - `load` is the whole of `load_checkpoint`: the validated forest, the
#   field set's allocation and zero fill, the state vector's first touch,
#   the read and the `scatter!`. The ghosts are not filled, as in the
#   function itself.
#
# Every load is checked to return the saved state bit for bit.
#
#     TREEAMR_BENCH_D       dimension (default 3)
#     TREEAMR_BENCH_N       cells per block edge (default 16)
#     TREEAMR_BENCH_ROOTS   roots per edge (default 6)
#     TREEAMR_BENCH_LEVELS  levels of refinement around the shell (default 2)
#     TREEAMR_BENCH_REPS    timed repetitions (default 5)
#     TREEAMR_BENCH_DIR     directory for the files (default a fresh temporary
#                           one); on a cluster, the file system under test

using TreeAMR
using HDF5
using HDF5.Filters: Shuffle, Deflate
using Printf: @printf

const D = parse(Int, get(ENV, "TREEAMR_BENCH_D", "3"))
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const ROOTS = parse(Int, get(ENV, "TREEAMR_BENCH_ROOTS", "6"))
const LEVELS = parse(Int, get(ENV, "TREEAMR_BENCH_LEVELS", "2"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "5"))
const DIR = get(ENV, "TREEAMR_BENCH_DIR", "")

# The filter packages that can be loaded here. They register their filters
# with HDF5 when loaded; the constructors are called only from `main`,
# which runs after the imports are visible.
const OPTIONAL = filter(p -> Base.find_package(p) !== nothing,
                        ["H5Zzstd", "H5Zlz4", "H5Zbitshuffle", "H5Zblosc"])
for p in OPTIONAL
    Core.eval(Main, :(import $(Symbol(p))))
end

"""The filter settings to measure, as `(name, filters)`."""
function filter_settings()
    settings = Tuple{String,Tuple}[("none", ()),
                                   ("shuffle+deflate(1)", (Shuffle(), Deflate(1)))]
    if "H5Zzstd" in OPTIONAL
        push!(settings, ("shuffle+zstd(1)", (Shuffle(), H5Zzstd.ZstdFilter(1))))
        push!(settings, ("shuffle+zstd(3)", (Shuffle(), H5Zzstd.ZstdFilter(3))))
    end
    if "H5Zlz4" in OPTIONAL
        push!(settings, ("shuffle+lz4", (Shuffle(), H5Zlz4.Lz4Filter())))
    end
    if "H5Zbitshuffle" in OPTIONAL
        push!(settings, ("bitshuffle+lz4",
                         (H5Zbitshuffle.BitshuffleFilter(; compressor=:lz4),)))
    end
    if "H5Zblosc" in OPTIONAL
        push!(settings, ("blosc(lz4,shuffle)",
                         (H5Zblosc.BloscFilter(; level=5, compressor="lz4"),)))
    end
    return settings
end

"""
Refined `LEVELS` times around the sphere `r = r₀`: a leaf is refined
when the sphere passes within half a block width of it, then balanced.
"""
function build_forest(r₀)
    forest = Forest(ntuple(_ -> ROOTS, D); N=N, extents=ntuple(_ -> (-1.0, 1.0), D))
    for lvl in 0:LEVELS-1
        targets = filter(forest.leaves) do k
            level(k) == lvl || return false
            ext = block_extent(forest, k)
            w = ext[1][2] - ext[1][1]
            near = sqrt(sum(d -> max(ext[d][1], 0.0, -ext[d][2])^2, 1:D))
            far = sqrt(sum(d -> max(abs(ext[d][1]), abs(ext[d][2]))^2, 1:D))
            return near <= r₀ + w / 2 && far >= r₀ - w / 2
        end
        refine!(forest, targets)
        balance!(forest)
    end
    return forest
end

# An outgoing spherical Gaussian pulse at radius r₀ and its time
# derivative, ∂ₜu = −∂ᵣu.
function pulse(forest, r₀)
    fs = FieldSet(forest, 2; G=2, centering=vertexcentered(D))
    σ = 0.1
    fill_by_coordinates!(fs) do x, v
        r = sqrt(sum(abs2, x))
        g = exp(-(r - r₀)^2 / (2σ^2))
        return v == 1 ? g : (r - r₀) / σ^2 * g
    end
    return fs
end

# A blast wave of radius r₀ in (ρ, ρv₁, …, ρv_D, E), γ = 5/3: behind the
# shock a smooth profile rising to the strong-shock density ratio 4, with
# a velocity linear in r and a pressure falling outward; ahead of it a
# uniform atmosphere at rest, the same bits everywhere.
function blast(forest, r₀)
    fs = FieldSet(forest, D + 2; G=2, centering=cellcentered(D))
    γ = 5 / 3
    ρatm, Patm = 1.0, 1e-5
    state = function (x)
        r = sqrt(sum(abs2, x))
        if r >= r₀
            return (ρatm, ntuple(_ -> 0.0, D)..., Patm / (γ - 1))
        end
        s = r / r₀
        ρ = ρatm * (0.05 + 3.95 * s^6)
        vr = 0.75 * s                                  # v / v_shock
        P = 0.3 + 0.2 * (1 - s^2)
        m = ntuple(d -> ρ * vr * x[d] / max(r, eps()), D)
        return (ρ, m..., P / (γ - 1) + ρ * vr^2 / 2)
    end
    fill_by_coordinates!(AllVariables(state), fs)
    return fs
end

function best(f, reps=REPS)
    GC.gc()
    f()
    t = Inf
    for _ in 1:reps
        GC.gc()
        t = min(t, @elapsed f())
    end
    return t
end

# On macOS `fsync` only hands the data to the drive, and returned in 1 ms
# for 540 MB on the machine the numbers in CODE.md come from; the flush
# to stable storage is `fcntl(F_FULLFSYNC)`, which took 125 ms there. So
# that is what `sync` means on macOS, and `fsync` elsewhere.
const F_FULLFSYNC = Cint(51)

function fsync_file(path)
    open(path) do io
        status = Sys.isapple() ?
                 ccall(:fcntl, Cint, (Cint, Cint, Cint...), fd(io), F_FULLFSYNC, 0) :
                 ccall(:fsync, Cint, (Cint,), fd(io))
        status == 0 || error("syncing $path failed")
    end
    return nothing
end

function main()
    dir = isempty(DIR) ? mktempdir() : DIR
    r₀ = 0.5
    forest = build_forest(r₀)
    datasets = (("pulse", pulse(forest, r₀)), ("blast", blast(forest, r₀)))
    settings = filter_settings()

    @printf("threads=%d D=%d N=%d roots=%d levels=%d blocks=%d REPS=%d Julia %s\n",
            Threads.nthreads(), D, N, ROOTS, LEVELS, nleaves(forest), REPS, VERSION)
    @printf("# blocks per level: %s\n",
            join([count(k -> level(k) == l, forest.leaves) for l in 0:LEVELS], ", "))
    @printf("# filter packages: %s\n", isempty(OPTIONAL) ? "none" : join(OPTIONAL, ", "))
    @printf("# files in %s\n", dir)
    @printf("%-6s %-20s %9s %9s %7s %8s %8s %8s\n", "data", "filter", "state MB",
            "file MB", "ratio", "save", "sync", "load")
    @printf("%-6s %-20s %9s %9s %7s %8s %8s %8s\n", "", "", "", "", "", "GB/s", "GB/s",
            "GB/s")
    for (dname, fs) in datasets
        u = statevector(fs)
        gather!(u, fs)
        bytes = sizeof(u)
        for (fname, filters) in settings
            path = joinpath(dir, "checkpoint-$dname.h5")
            save() = save_checkpoint(path, forest; fieldsets=("U" => (fs, u),),
                                     application="bench" => 1, data=(; t=0.0),
                                     filters=filters)
            t_save = best(save)
            t_sync = best(() -> (save(); fsync_file(path)))
            fsize = filesize(path)
            t_load = best(() -> load_checkpoint(path))
            ck = load_checkpoint(path)
            ck.forest.leaves == forest.leaves &&
                reinterpret(UInt8, ck.fieldsets["U"].state) == reinterpret(UInt8, u) ||
                error("$dname with $fname did not round-trip bit for bit")
            @printf("%-6s %-20s %9.1f %9.1f %7.2f %8.2f %8.2f %8.2f\n", dname, fname,
                    bytes / 1e6, fsize / 1e6, bytes / fsize, bytes / t_save / 1e9,
                    bytes / t_sync / 1e9, bytes / t_load / 1e9)
            rm(path)
        end
    end
    isempty(DIR) && rm(dir; recursive=true)
    return nothing
end

main()
