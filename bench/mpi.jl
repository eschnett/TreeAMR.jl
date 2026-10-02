# Weak scaling over MPI ranks (M7 step 7).
#
#     mpiexec -n P julia -t T --project=test bench/mpi.jl mpi
#     julia -t T --project=test bench/mpi.jl              # serial, the control
#     bench/mpiscan.sh 1 2 4                              # P = 1, 2, 4, and a table
#
# Every rank holds the same number of blocks whatever the rank count: the
# mesh is `TILES` copies of one tile of `ROOTS^D` roots stacked along the
# last dimension (`TILES` = the rank count unless set), periodic in every
# dimension, and the equal-count split of the curve gives rank `r`
# exactly tile `r`, since the roots are numbered with the last dimension
# slowest and every tile has the same leaves. So the rank boundaries are
# the tiles' two faces normal to `x_D`, with a constant halo per rank
# from three ranks on (at one rank the tile is its own neighbor, at two
# both faces face the same peer). Two meshes:
#
# - `twolevel`: in every tile, the leaves with `x₁` in the lower half
#   and `x_D` in the upper half of the tile refined once. The top face of
#   a tile is then fine below and coarse above, so a coarse-fine face
#   crosses every rank boundary and phase 2 (the prolongations), the
#   restrictions of phase 1 and the interface restriction all carry
#   messages; the other coarse-fine faces are inside the tile. At
#   `ROOTS = 4` in 3D a tile has 176 blocks.
# - `uniform`: the tile unrefined, `ROOTS^D` blocks (64 at `ROOTS = 4`),
#   only copies.
#
# Timed, per mesh, in synchronized windows: every repetition starts at an
# `MPI.Barrier`, its time is the slowest rank's (an `allgather` of the
# per-rank times, the barrier and the gather outside the window), and
# what is reported is the minimum and the median over the repetitions,
# after one untimed call that compiles. Phases:
#
#     rhs                 scatter! → fill_ghosts! → map_blocks! of a wave
#                         right-hand side (bench/threads.jl's, over every
#                         pair of variables)
#     fill_ghosts         the ghost fill alone
#     fill_locals         its local groups alone, stage by stage, no packs,
#                         unpacks or messages (internal calls)
#     fill_packs          its packs alone, into the send buffers
#     fill_unpacks        its unpacks alone, from the receive buffers
#     scatter             scatter! alone (no messages: the ideal line)
#     norm                volume_weighted_norm (two reductions)
#     maxabs              mesh_mapreduce(abs, max, …) (one reduction)
#     interfaces          restrict_interfaces! on a set face-centered along x_D,
#                         G = 0, whose coarse-fine faces cross the rank boundaries
#     interpolate         interpolate! of NPTS points per rank, spread over
#                         the whole domain, so most route to other ranks
#     ghost_schedule      GhostSchedule(fs, ops): the build, digest included
#     interface_schedule  InterfaceSchedule(fluxes)
#     regrid_refine       regrid! refining the lowest layer of roots of tile
#                         0, the start of the curve: every later rank's range
#                         shifts, so at P ≥ 2 blocks migrate (up to the
#                         whole refinement per rank, half of it on average)
#     regrid_coarsen      regrid! coarsening it back (the schedules are
#                         rebuilt between the two, outside the windows)
#     triad_reference     c = a + 2b over arrays the size of the working
#                         array, on every rank at once (bench/threads.jl's)
#
# `fill_ghosts − (fill_locals + fill_packs + fill_unpacks)` is what the
# messages cost beyond what the local groups hid. On a device every
# phase ends in a `synchronize`.
#
# Output (rank 0): one header line `ranks=…`, then per phase one
# tab-separated line
#
#     <ranks>x<threads>  <mesh>:<phase>  <min seconds>  <median seconds>
#
# which `bench/mpitable.awk` turns into a weak-scaling table (efficiency
# = the first column's time over each column's), and `#` lines: the
# blocks and cells, what a ghost fill sends and receives per rank
# (messages, bytes, transfers, peers; min, mean and max over ranks), the
# blocks the regrid moved between ranks, and ns per cell of the whole
# job for the per-evaluation phases (the slowest rank's time over every
# cell of every rank, comparable with bench/threads.jl's numbers for one
# process over the same mesh: `TREEAMR_BENCH_TILES`).
#
#     TREEAMR_BENCH_D           dimension (default 3)
#     TREEAMR_BENCH_N           cells per block edge (default 16)
#     TREEAMR_BENCH_ROOTS       roots per tile edge (default 4)
#     TREEAMR_BENCH_TILES       tiles (default: the rank count); set it for the
#                               serial control over the mesh of a P-rank run
#     TREEAMR_BENCH_NVARS       variables, even (default 2)
#     TREEAMR_BENCH_REPS        timed repetitions (default 10; the regrid and
#                               schedule phases take a third, at least 3)
#     TREEAMR_BENCH_NPTS        interpolation points per rank (default 1000)
#     TREEAMR_BENCH_MESHES      comma-separated, of twolevel and uniform
#                               (default both)
#     TREEAMR_BENCH_BACKEND     cpu (default), cuda, or metal (an environment
#                               with the device package and MPI; one device
#                               per rank, `local rank mod devices`)
#     TREEAMR_BENCH_T           Float64 or Float32 (default Float64, or
#                               Float32 where the backend has no fp64)
#     TREEAMR_BENCH_LABEL       the first column (default <ranks>x<threads>), to
#                               tell apart runs of one shape, e.g. two placements
#     TREEAMR_BENCH_DEVICEAWARE 1 to hand MPI the device buffers
#                               (`communicator(…; deviceaware = true)`);
#                               default 0, staging through host mirrors
#     TREEAMR_BENCH_FORCESTAGING 1 to stage every message through host
#                               mirrors even on the CPU, whose buffers are
#                               host memory already: the staging path's
#                               host-side cost, without a device (default 0)
#
# The test environment has MPI and KernelAbstractions; `--project=.`
# does not (MPI is a weak dependency).

const USE_MPI = "mpi" in ARGS
if USE_MPI
    using MPI
    MPI.Init()
end
using TreeAMR
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize, CPU,
                          supports_float64, allocate
using Printf: Printf, @sprintf

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

const RANK = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const NRANKS = USE_MPI ? MPI.Comm_size(MPI.COMM_WORLD) : 1
const DEVICEAWARE = get(ENV, "TREEAMR_BENCH_DEVICEAWARE", "0") == "1"
const FORCESTAGING = get(ENV, "TREEAMR_BENCH_FORCESTAGING", "0") == "1"

# The MPI communicator with every buffer staged (`TREEAMR_BENCH_FORCESTAGING`),
# as `test/regrid_exchange_tests.jl`'s `StagingCommunicator` does in process.
struct ForcedStaging{C<:TreeAMR.Communicator} <: TreeAMR.Communicator
    inner::C
end
TreeAMR.hoststaging(::ForcedStaging, ::AbstractVector) = true
for verb in (:commrank, :commsize, :librarycomm)
    @eval TreeAMR.$verb(c::ForcedStaging) = TreeAMR.$verb(c.inner)
end
TreeAMR.allgather(c::ForcedStaging, x) = TreeAMR.allgather(c.inner, x)
TreeAMR.allgatherv(c::ForcedStaging, v::AbstractVector) = TreeAMR.allgatherv(c.inner, v)
TreeAMR.waitall(c::ForcedStaging, requests::AbstractVector) =
    TreeAMR.waitall(c.inner, requests)
TreeAMR.alltoallv(c::ForcedStaging, buf::AbstractVector, counts::AbstractVector{<:Integer}) =
    TreeAMR.alltoallv(c.inner, buf, counts)
TreeAMR.isend(c::ForcedStaging, buf::AbstractVector, peer::Integer, tag::Integer) =
    TreeAMR.isend(c.inner, buf, peer, tag)
TreeAMR.irecv(c::ForcedStaging, buf::AbstractVector, peer::Integer, tag::Integer) =
    TreeAMR.irecv(c.inner, buf, peer, tag)

const COMM = !USE_MPI ? nothing :
             FORCESTAGING ? ForcedStaging(communicator(MPI.COMM_WORLD)) :
             communicator(MPI.COMM_WORLD; deviceaware=DEVICEAWARE)
const LOCALRANK = USE_MPI ?
                  MPI.Comm_rank(MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED,
                                                    RANK)) : 0
if BNAME == "cuda"
    CUDA.functional() || error("CUDA is not functional here")
    CUDA.device!(LOCALRANK % length(CUDA.devices()))
end
const BACKEND = BNAME == "cuda" ? CUDABackend() :
                BNAME == "metal" ? (Metal.functional() ? MetalBackend() :
                                    error("Metal is not functional here")) :
                CPU()
const T = let want = get(ENV, "TREEAMR_BENCH_T", "")
    isempty(want) ? (supports_float64(BACKEND) ? Float64 : Float32) :
    want == "Float64" ? Float64 : want == "Float32" ? Float32 :
    error("TREEAMR_BENCH_T must be Float64 or Float32, got \"$want\"")
end

const D = parse(Int, get(ENV, "TREEAMR_BENCH_D", "3"))
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const ROOTS = parse(Int, get(ENV, "TREEAMR_BENCH_ROOTS", "4"))
const TILES = parse(Int, get(ENV, "TREEAMR_BENCH_TILES", string(NRANKS)))
const NVARS = parse(Int, get(ENV, "TREEAMR_BENCH_NVARS", "2"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "10"))
const NPTS = parse(Int, get(ENV, "TREEAMR_BENCH_NPTS", "1000"))
const MESHES = split(get(ENV, "TREEAMR_BENCH_MESHES", "twolevel,uniform"), ",")
const G = 2
const OPS = Operators(prolongation=4, restriction=4)
const LABEL = get(ENV, "TREEAMR_BENCH_LABEL", "$(NRANKS)x$(Threads.nthreads())")

iseven(NVARS) || error("TREEAMR_BENCH_NVARS must be even: the variables are wave pairs")
ROOTS % 2 == 0 || error("TREEAMR_BENCH_ROOTS must be even: the refinement takes halves")

# bench/threads.jl's wave right-hand side, over every pair of variables
# (u, ∂ₜu): du = ∂ₜu, d∂ₜu = Δu, second order.
@kernel function bench_rhs_kernel!(du, @Const(work), @Const(spacings),
                                   ::Val{DD}, ::Val{GG}, ::Val{NV}) where {DD,GG,NV}
    I = @index(Global, NTuple)
    b = I[DD + 1]
    c = ntuple(d -> I[d] + GG[d], Val(DD))
    o = ntuple(d -> I[d], Val(DD))
    h = spacings[b]
    for v in 1:2:NV
        u0 = work[c..., v, b]
        laplacian = zero(eltype(du))
        for d in 1:DD
            up = Base.setindex(c, c[d] + 1, d)
            um = Base.setindex(c, c[d] - 1, d)
            laplacian += work[up..., v, b] - 2 * u0 + work[um..., v, b]
        end
        du[o..., v, b] = work[c..., v + 1, b]
        du[o..., v + 1, b] = laplacian / (h * h)
    end
end

@kernel function triad_kernel!(c, @Const(a), @Const(b))
    i = @index(Global, Linear)
    c[i] = a[i] + 2 * b[i]
end

# Written before they are read (see bench/threads.jl: an untouched
# allocation on Linux reads the shared zero page).
@kernel function fill_kernel!(x, v)
    i = @index(Global, Linear)
    x[i] = v
end

const ROOTDIMS = ntuple(d -> d == D ? TILES * ROOTS : ROOTS, D)

function build_forest(mesh)
    forest = Forest(ROOTDIMS; N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(d -> (zero(T), T(ROOTDIMS[d])), D), comm=COMM)
    mesh == "uniform" && return forest
    mesh == "twolevel" || error("unknown mesh \"$mesh\": twolevel or uniform")
    half = T(ROOTS) / 2
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        x1 = (ext[1][1] + ext[1][2]) / 2
        xD = (ext[D][1] + ext[D][2]) / 2
        return x1 < half && mod(xD, T(ROOTS)) > half
    end
    refine!(forest, targets)
    balance!(forest)
    return forest
end

# The slab the regrid refines: the level-0 leaves of the lowest layer of
# roots, at the start of the curve.
inslab(forest, k) = root_position(forest, k.root)[D] == 0

barrier() = USE_MPI ? MPI.Barrier(MPI.COMM_WORLD) : nothing

# One synchronized window: the slowest rank's time for `f`.
function window(f, comm)
    barrier()
    t = @elapsed f()
    return maximum(TreeAMR.allgather(comm, t))
end

median(ts) = (s = sort(ts); n = length(s); (s[(n + 1) ÷ 2] + s[n ÷ 2 + 1]) / 2)

# Minimum and median over `reps` windows, after one untimed call.
function measure(f, comm, reps=REPS)
    f()
    GC.gc()
    ts = [window(f, comm) for _ in 1:reps]
    return minimum(ts), median(ts)
end

say(s) = (RANK == 0 && println(s); nothing)
line(mesh, phase, (tmin, tmed)) =
    say(@sprintf("%s\t%s:%s\t%.6f\t%.6f", LABEL, mesh, phase, tmin, tmed))

# min, mean and max over the ranks of each field of a tuple of numbers.
function spread(comm, values::NTuple{K,Int}) where {K}
    all = TreeAMR.allgather(comm, values)
    return ntuple(i -> (minimum(v[i] for v in all),
                        sum(v[i] for v in all) / length(all),
                        maximum(v[i] for v in all)), K)
end
fmt((lo, mean, hi)) = @sprintf("%d/%.1f/%d", lo, mean, hi)

# What one fill sends and receives on this rank, from the schedule's
# stages: stages with messages, messages, bytes, transfers and peers.
function message_counts(stages, nvars)
    remotes = [s.remote for s in stages if s.remote !== nothing]
    nstages = length(remotes)
    nsend = sum(r -> length(r.sendpeers), remotes; init=0)
    nrecv = sum(r -> length(r.recvpeers), remotes; init=0)
    bsend = sum(r -> sum(r.sendcounts; init=0), remotes; init=0) * nvars * sizeof(T)
    brecv = sum(r -> sum(r.recvcounts; init=0), remotes; init=0) * nvars * sizeof(T)
    tsend = sum(r -> length(r.sendlayout), remotes; init=0)
    trecv = sum(r -> length(r.recvlayout), remotes; init=0)
    tlocal = sum(s -> sum(TreeAMR.ntransfers, s.locals; init=0), stages; init=0)
    peers = length(unique(reduce(vcat, [[r.sendpeers; r.recvpeers] for r in remotes];
                                 init=Int[])))
    return (nstages, nsend, nrecv, bsend, brecv, tsend, trecv, tlocal, peers)
end

function report_messages(comm, what, stages, nvars)
    s = spread(comm, message_counts(stages, nvars))
    say("# $what per rank (min/mean/max): stages with messages $(fmt(s[1])), " *
        "messages sent $(fmt(s[2])) received $(fmt(s[3])), bytes sent $(fmt(s[4])) " *
        "received $(fmt(s[5])), transfers sent $(fmt(s[6])) received $(fmt(s[7])) " *
        "local $(fmt(s[8])), peers $(fmt(s[9]))")
end

# How many of this rank's new blocks take data from another rank: a kept
# block that changed owner, a child whose parent, or a parent any of
# whose children, lived elsewhere.
function moved_in(oldlocal::Set, oldall::Set, fs)
    count(1:nblocks(fs)) do b
        k = blockkey(fs, b)
        k in oldall && return !(k in oldlocal)
        level(k) > 0 && parentkey(k) in oldall && return !(parentkey(k) in oldlocal)
        return !all(in(oldlocal), childkeys(k))
    end
end

# NPTS points per rank over the whole domain, a Kronecker sequence that
# differs between ranks.
function points(::Val{DD}) where {DD}
    α = (sqrt(2.0), sqrt(3.0), sqrt(5.0), sqrt(7.0))
    return [ntuple(d -> T(ROOTDIMS[d] * mod(0.5 + (j + RANK * NPTS) * α[d], 1.0)), DD)
            for j in 1:NPTS]
end

function run_mesh(mesh)
    forest = build_forest(mesh)
    comm = forest.comm
    fs = FieldSet{T}(forest, NVARS; G=G, backend=BACKEND)
    initial = (x, v) -> sin(2 * T(π) * x[1] / ROOTS + v) * cos(2 * T(π) * x[D] / ROOTS)
    fill_by_coordinates!(initial, fs)
    schedule = GhostSchedule(fs, OPS)
    fluxes = FieldSet{T}(forest, NVARS; G=0, centering=facecentered(D, D), backend=BACKEND)
    fill_by_coordinates!(initial, fluxes)
    isched = InterfaceSchedule(fluxes)
    spacings = let h = block_spacings(forest, T)
        BACKEND isa CPU ? h : TreeAMR.todevice(BACKEND, h)
    end

    u = statevector(fs)
    gather!(u, fs)
    du = similar(u)
    dua = statearray(du, fs)
    sync() = synchronize(BACKEND)
    fill!() = (fill_ghosts!(fs, schedule); sync())
    rhs!() = begin
        scatter!(fs, u)
        fill_ghosts!(fs, schedule)
        map_blocks!(bench_rhs_kernel!, fs, dua, fs.work, spacings, Val(D), Val(fs.G),
                    Val(NVARS))
        sync()
    end
    # The pieces of a fill, through the internals `run_stage!` calls.
    locals!() = begin
        for st in schedule.stages
            TreeAMR.run_phase!(fs.work, fs.work, st.locals, fs.nvars, BACKEND;
                               factors=fs.factors)
        end
        sync()
    end
    remotes = [st.remote for st in schedule.stages if st.remote !== nothing]
    packs!() = begin
        for r in remotes
            TreeAMR.pack_stage!(fs, r, TreeAMR.stagebuffers(r, fs.nvars, BACKEND), BACKEND)
        end
        sync()
    end
    unpacks!() = begin
        for r in remotes
            TreeAMR.unpack_stage!(fs, r, TreeAMR.stagebuffers(r, fs.nvars, BACKEND),
                                  BACKEND)
        end
        sync()
    end

    nb = nblocks(fs)
    cellsrank = TreeAMR.allgather(comm, nb * N^D)
    cells = sum(cellsrank)
    say("# $mesh: $(nleaves(forest)) blocks, levels 0:$(maxlevel(forest)), blocks per " *
        "rank $(join(TreeAMR.allgather(comm, nb), ", ")), $cells cells, " *
        "$(round(sizeof(fs.work) / 2^20; digits=1)) MiB of working array on rank 0")
    report_messages(comm, "$mesh: a ghost fill", schedule.stages, NVARS)
    report_messages(comm, "$mesh: an interface restriction", isched.stages, NVARS)

    res = Dict{String,Tuple{Float64,Float64}}()
    record(phase, f, reps=REPS) =
        (res[phase] = measure(f, comm, reps); line(mesh, phase, res[phase]))

    record("rhs", rhs!)
    record("fill_ghosts", fill!)
    record("fill_locals", locals!)
    record("fill_packs", packs!)
    record("fill_unpacks", unpacks!)
    record("scatter", () -> (scatter!(fs, u); sync()))
    record("norm", () -> volume_weighted_norm(fs, u))
    record("maxabs", () -> mesh_mapreduce(abs, max, zero(T), fs, u))
    record("interfaces", () -> (restrict_interfaces!(fluxes, isched); sync()))

    xs = TreeAMR.todevice(BACKEND, points(Val(D)))
    vals = allocate(BACKEND, T, (NVARS, 1, NPTS))
    exc = allocate(BACKEND, Bool, NPTS)
    record("interpolate", () -> (interpolate!(vals, exc, fs, xs, Lagrange(4)); sync()))

    bytes = @allocated rhs!()
    say(@sprintf("# %s: rhs allocates %.1f kB per call on rank 0", mesh, bytes / 1e3))

    slow = max(3, REPS ÷ 3)
    record("ghost_schedule", () -> GhostSchedule(fs, OPS), slow)
    record("interface_schedule", () -> InterfaceSchedule(fluxes), slow)

    # The regrid cycle. Each repetition refines the slab and coarsens it
    # back, so every one starts from the same mesh; the schedules are
    # rebuilt between the two calls, outside the windows.
    original = copy(forest.leaves)
    refine_flags = flag_blocks((b, k) -> inslab(forest, k) && level(k) == 0 ? Refine :
                                         Keep, forest)
    t_refine, t_coarsen = Float64[], Float64[]
    moved = (0, 0)
    bytes_refine = bytes_coarsen = 0               # host bytes allocated, last repetition
    # The forest's buffer pool, where it has one (`forest.state.pool`).
    pooled(forest) = hasfield(typeof(forest), :state) ? forest.state.pool : nothing
    firstcycle = nothing
    current = schedule
    for rep in 0:slow
        oldlocal = Set(blockkey(fs, b) for b in 1:nblocks(fs))
        oldall = Set(forest.leaves)
        t = window(comm) do
            bytes_refine = @allocated regrid!(forest, (fs => current, fluxes => nothing);
                                              flags=refine_flags)
        end
        rep > 0 && push!(t_refine, t)
        m1 = moved_in(oldlocal, oldall, fs)
        current = GhostSchedule(fs, OPS)
        coarsen_flags = flag_blocks((b, k) -> inslab(forest, k) && level(k) == 1 ?
                                              Coarsen : Keep, forest)
        oldlocal = Set(blockkey(fs, b) for b in 1:nblocks(fs))
        oldall = Set(forest.leaves)
        t = window(comm) do
            bytes_coarsen = @allocated regrid!(forest, (fs => current, fluxes => nothing);
                                               flags=coarsen_flags)
        end
        rep > 0 && push!(t_coarsen, t)
        moved = (m1, moved_in(oldlocal, oldall, fs))
        current = GhostSchedule(fs, OPS)
        forest.leaves == original || error("the regrid cycle did not return to the mesh")
        if rep == 0 && pooled(forest) !== nothing
            p = pooled(forest)
            firstcycle = (p.allocated, p.pagelocked, p.dropped)
        end
        GC.gc()
    end
    res["regrid_refine"] = (minimum(t_refine), median(t_refine))
    line(mesh, "regrid_refine", res["regrid_refine"])
    res["regrid_coarsen"] = (minimum(t_coarsen), median(t_coarsen))
    line(mesh, "regrid_coarsen", res["regrid_coarsen"])
    mv = spread(comm, moved)
    nslab = ROOTS^(D - 1)
    say("# $mesh: the regrid refines $nslab blocks into $(nslab * 2^D) and back; " *
        "blocks taking data from another rank (min/mean/max over ranks): refine " *
        "$(fmt(mv[1])), coarsen $(fmt(mv[2]))")
    # What the last repetition allocated on the host, and, where the forest
    # has a buffer pool, how many message buffers it ever allocated.
    say(@sprintf("# %s: the last regrids allocate %.1f / %.1f MB (refine / coarsen) on rank 0",
                 mesh, bytes_refine / 1e6, bytes_coarsen / 1e6))
    pool = pooled(forest)
    if pool !== nothing
        say("# $mesh: rank 0's buffer pool after $(slow + 1) regrid cycles: " *
            "$(pool.allocated) buffers allocated, $(pool.pagelocked) mirrors " *
            "page-locked, $(pool.dropped) dropped" *
            (firstcycle === nothing ? "" :
             "; after the first cycle $(join(firstcycle, ", "))"))
    end

    # The bandwidth reference, on every rank at once.
    n = length(fs.work)
    a, b, c = (similar(fs.work, n) for _ in 1:3)
    fill_kernel!(BACKEND)(a, one(T); ndrange=n)
    fill_kernel!(BACKEND)(b, one(T); ndrange=n)
    record("triad_reference", () -> (triad_kernel!(BACKEND)(c, a, b; ndrange=n); sync()))
    say(@sprintf("# %s: triad %.1f GB/s per rank (min window), %.1f GB/s over the job",
                 mesh, 3 * n * sizeof(T) / res["triad_reference"][1] / 1e9,
                 3 * n * sizeof(T) * NRANKS / res["triad_reference"][1] / 1e9))

    per(phase) = @sprintf("%s %.3f", phase, res[phase][1] / cells * 1e9)
    say("# $mesh: ns per cell of the job (min): " *
        join(per.(["rhs", "fill_ghosts", "scatter", "norm", "interfaces",
                   "triad_reference"]), ", "))
    return nothing
end

function main()
    say("ranks=$NRANKS threads=$(Threads.nthreads()) D=$D N=$N roots=$ROOTS tiles=$TILES " *
        "nvars=$NVARS backend=$BNAME T=$T deviceaware=$DEVICEAWARE " *
        "forcestaging=$FORCESTAGING reps=$REPS " *
        "npts=$NPTS Julia $VERSION")
    for mesh in MESHES
        run_mesh(mesh)
    end
    return nothing
end

main()
