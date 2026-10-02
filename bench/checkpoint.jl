# Checkpoint throughput (M9a; distributed, M7 steps 6 and 6b).
#
#     julia -t N --project=<env> bench/checkpoint.jl
#     mpiexec -n P julia -t N --project=<env> bench/checkpoint.jl mpi
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
# (`--project=test`). The filter packages H5Zzstd, H5Zlz4 and H5Zbitshuffle
# are used where they can be loaded — from the active
# environment, or from the default one Julia stacks under it — and
# skipped otherwise. They are not test dependencies, so the numbers with
# them come from a scratch environment that develops this checkout and
# adds them:
#
#     julia --project=<scratch> -e 'using Pkg; Pkg.develop(path="<this checkout>");
#         Pkg.add(["HDF5", "H5Zzstd", "H5Zlz4", "H5Zbitshuffle"])'
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
# - **`save` ends in the page cache.** It is `save_checkpoint(…; sync =
#   false)`, the rate at which the data reach the operating system;
#   `sync` is the default, `sync = true`, which adds the flushes of the
#   file and its directory to stable storage (`fsync`, or `F_FULLFSYNC`
#   on macOS, where `fsync` does not wait for the drive's cache).
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
# Every save is checked, untimed, by a load that must return the saved
# leaves and state bit for bit on every rank, so a file damaged while it
# was written cannot pass (on a cluster file system one did: "Parallel
# checkpoints" under "Distributed meshes" in CODE.md, M7 step 6). The
# load verifies the file's checksums too, and refuses damage with its
# reason.
#
# **Under MPI** (the argument `mpi`, launched by `mpiexec`; the test
# environment has MPI) the forest is distributed over `MPI.COMM_WORLD`,
# and the save writes one part file per I/O process beside an index
# (step 6b: no file is written or opened by more than one process; until
# then, step 6, one shared file through parallel HDF5): `TREEAMR_BENCH_IO`
# chooses the I/O processes, `node` (the default), `all` or a number. Every
# rank loads its own blocks back (`load_checkpoint(path; comm)`), each
# part read by one rank and sent to the blocks' owners. Each call is timed
# between two barriers, as the slowest rank's time, and the rates are the
# aggregate over all ranks: the whole state over that time. Rank 0
# prints, and adds each rank's share of the state, the largest one, and
# the number of parts. The file size is the index's and the parts'
# together. `TREEAMR_BENCH_DIR` is then a directory every rank sees — on
# a cluster, the parallel file system under test
# (`bench/symmetry_checkpoint_mpi.sh`).
#
#     TREEAMR_BENCH_D       dimension (default 3)
#     TREEAMR_BENCH_N       cells per block edge (default 16)
#     TREEAMR_BENCH_ROOTS   roots per edge (default 6)
#     TREEAMR_BENCH_LEVELS  levels of refinement around the shell (default 2)
#     TREEAMR_BENCH_REPS    timed repetitions (default 5)
#     TREEAMR_BENCH_DIR     directory for the files (default a fresh temporary
#                           one); on a cluster, the file system under test
#     TREEAMR_BENCH_IO      the `io` of save_checkpoint: node (default), all, or
#                           a number of I/O processes

const USE_MPI = "mpi" in ARGS
if USE_MPI
    using MPI
    MPI.Init()
end
using TreeAMR
using HDF5
using HDF5.Filters: Shuffle, Deflate
using Printf: Printf

const COMM = USE_MPI ? MPI.COMM_WORLD : nothing
const RANK = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const NRANKS = USE_MPI ? MPI.Comm_size(MPI.COMM_WORLD) : 1

const D = parse(Int, get(ENV, "TREEAMR_BENCH_D", "3"))
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const ROOTS = parse(Int, get(ENV, "TREEAMR_BENCH_ROOTS", "6"))
const LEVELS = parse(Int, get(ENV, "TREEAMR_BENCH_LEVELS", "2"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "5"))
const DIR = get(ENV, "TREEAMR_BENCH_DIR", "")
const IO_SETTING = let io = get(ENV, "TREEAMR_BENCH_IO", "node")
    io in ("node", "all") ? Symbol(io) : parse(Int, io)
end

# The filter packages that can be loaded here. They register their filters
# with HDF5 when loaded; the constructors are called only from `main`,
# which runs after the imports are visible.
const OPTIONAL = filter(p -> Base.find_package(p) !== nothing,
                        ["H5Zzstd", "H5Zlz4", "H5Zbitshuffle"])
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
        push!(settings, ("bitshuffle+zstd(1)",
                         (H5Zbitshuffle.BitshuffleFilter(; compressor=:zstd,
                                                          comp_level=1),)))
    end
    return settings
end

"""
Refined `LEVELS` times around the sphere `r = r₀`: a leaf is refined
when the sphere passes within half a block width of it, then balanced.
"""
function build_forest(r₀)
    forest = Forest(ntuple(_ -> ROOTS, D); N=N, extents=ntuple(_ -> (-1.0, 1.0), D),
                    comm=COMM)
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

# Over several ranks a call takes as long as its slowest rank, between a
# barrier before it and one after: the `allgather` of every rank's time.
function timed(f, comm)
    TreeAMR.allgather(comm, true)                    # a barrier
    t = @elapsed f()
    return maximum(TreeAMR.allgather(comm, t))
end

# The best of `reps` timed calls after an untimed first one; `check`, if
# given, runs untimed after every call, the first included.
function best(f, comm, reps=REPS; check=nothing)
    GC.gc()
    f()
    check === nothing || check()
    t = Inf
    for _ in 1:reps
        GC.gc()
        t = min(t, timed(f, comm))
        check === nothing || check()
    end
    return t
end

# Whether the checkpoint at `path`, as loaded, holds `forest`'s leaves and
# the state `u` bit for bit, on every rank; an error saying which rank's
# part differs, on every rank, if not.
function verify(path, forest, u, what)
    ck = load_checkpoint(path; comm=COMM)
    exact = ck.forest.leaves == forest.leaves &&
            reinterpret(UInt8, ck.fieldsets["U"].state) == reinterpret(UInt8, u)
    ok = TreeAMR.allgather(forest.comm, exact)
    all(ok) || error("$what did not round-trip bit for bit on rank(s) ",
                     join(findall(!, ok) .- 1, ", "))
    return nothing
end

# The files of the checkpoint at `path`: the index and its parts, which
# are named after it.
checkpoint_files(path) =
    [joinpath(dirname(path), n) for n in readdir(dirname(path))
     if n == basename(path) || startswith(n, basename(path) * ".")]

# A line printed by rank 0 alone.
say(fmt, args...) = (RANK == 0 && print(Printf.format(Printf.Format(fmt), args...)); nothing)

function main()
    r₀ = 0.5
    forest = build_forest(r₀)
    comm = forest.comm
    # One directory for every rank: rank 0's, named to the others.
    dir = !isempty(DIR) ? DIR :
          String(TreeAMR.allgatherv(comm, RANK == 0 ? collect(codeunits(mktempdir())) :
                                          UInt8[]))
    datasets = (("pulse", pulse(forest, r₀)), ("blast", blast(forest, r₀)))
    settings = filter_settings()

    say("ranks=%d threads=%d D=%d N=%d roots=%d levels=%d blocks=%d REPS=%d Julia %s\n",
        NRANKS, Threads.nthreads(), D, N, ROOTS, LEVELS, nleaves(forest), REPS, VERSION)
    say("# blocks per level: %s\n",
        join([count(k -> level(k) == l, forest.leaves) for l in 0:LEVELS], ", "))
    say("# blocks per rank: %s\n",
        join(TreeAMR.allgather(comm, length(blockrange(forest))), ", "))
    say("# filter packages: %s\n", isempty(OPTIONAL) ? "none" : join(OPTIONAL, ", "))
    say("# files in %s, io = %s\n", dir, repr(IO_SETTING))
    say("%-6s %-20s %9s %9s %9s %7s %8s %8s %8s %6s\n", "data", "filter", "state MB",
        "rank MB", "file MB", "ratio", "save", "sync", "load", "parts")
    say("%-6s %-20s %9s %9s %9s %7s %8s %8s %8s %6s\n", "", "", "", "max", "", "", "GB/s",
        "GB/s", "GB/s", "")
    for (dname, fs) in datasets
        u = statevector(fs)
        gather!(u, fs)
        shares = TreeAMR.allgather(comm, sizeof(u))
        bytes = sum(shares)
        for (fname, filters) in settings
            path = joinpath(dir, "checkpoint-$dname.h5")
            save(sync) = save_checkpoint(path, forest; fieldsets=("U" => (fs, u),),
                                         application="bench" => 1, data=(; t=0.0),
                                         filters=filters, sync=sync, io=IO_SETTING)
            check() = verify(path, forest, u, "$dname with $fname")
            t_save = best(() -> save(false), comm; check=check)
            t_sync = best(() -> save(true), comm; check=check)
            TreeAMR.allgather(comm, true)        # rank 0 removes the old parts first
            fsize = sum(filesize, checkpoint_files(path))
            nparts = load_checkpoint(path; comm=COMM).provenance.nparts
            t_load = best(() -> load_checkpoint(path; comm=COMM), comm)
            say("%-6s %-20s %9.1f %9.1f %9.1f %7.2f %8.2f %8.2f %8.2f %6d\n", dname,
                fname, bytes / 1e6, maximum(shares) / 1e6, fsize / 1e6, bytes / fsize,
                bytes / t_save / 1e9, bytes / t_sync / 1e9, bytes / t_load / 1e9, nparts)
            TreeAMR.allgather(comm, true)
            RANK == 0 && foreach(rm, checkpoint_files(path))
        end
    end
    TreeAMR.allgather(comm, true)
    isempty(DIR) && RANK == 0 && rm(dir; recursive=true)
    return nothing
end

main()
