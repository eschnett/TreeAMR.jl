# The M7 rank-count independence check on a device (step 8 of M7).
#
# `mpi_workload.jl` checks the distributed mesh on the CPU, inside the
# suite. This script checks the same claim for field sets on a device
# backend, where the exchange's message buffers live in device memory and
# either go through host mirrors or, over a device-aware MPI, straight to
# the library ("MPI+GPU" under "Distributed meshes" in CODE.md). It is not
# part of `Pkg.test`: the test environment has no device package, and must
# not gain one. Run it in a scratch environment that develops this
# checkout and adds MPI, KernelAbstractions, SHA and the device package,
# through `mpi_device_tests.jl`, which runs it serially and under
# `mpiexec` and compares:
#
#     TREEAMR_TEST_BACKEND=metal julia --project=<env> test/mpi_device_tests.jl
#
# or by hand, one run at a time:
#
#     TREEAMR_TEST_BACKEND=metal julia --project=<env> test/mpi_device_workload.jl
#     TREEAMR_TEST_BACKEND=metal mpiexec -n 3 julia --project=<env> \
#         test/mpi_device_workload.jl mpi
#
# Environment:
#
#     TREEAMR_TEST_BACKEND      cpu, metal or cuda (default cpu)
#     TREEAMR_TEST_T            Float64 or Float32 (default: Float64 where the
#                               backend has hardware fp64, else Float32)
#     TREEAMR_TEST_DEVICEAWARE  1 to build the forests over
#                               `communicator(COMM_WORLD; deviceaware = true)`,
#                               which hands MPI the device buffers; only for an
#                               MPI that reads device memory
#
# As in `mpi_workload.jl`, rank 0 prints lines that must not depend on the
# rank count — digests of the leaves and of the field data gathered to
# the host in block order, ghosts included, and the exact reductions —
# with the floating-point sums on lines whose second word is `sum`, which
# agree to roundoff only, and `#` lines that depend on the rank count.
# The claim is the device's own: each line equals the *serial run on the
# same backend*, not the CPU's, since a device's `exp` or `tanh` may
# differ from the host's by an ulp. The cases are a subset of
# `mpi_workload.jl`'s, enough to cross every kind of stage on the device:
# the wave on three levels with the hook, a reflecting box with an odd
# variable (the `−0` case), and a 3D periodic mesh, each filled and
# stepped with RK4; Burgers with the interface fixup; and regrids — the
# tracked pulse through `firing_boxes` on the device, and a refinement of
# the first blocks and its coarsening, which move blocks up and down the
# ranks and coarsen siblings with different owners.

using TreeAMR
using MPI: MPI
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, CPU,
                          supports_float64, allocate
using Printf: @sprintf
using SHA: sha256

const BNAME = lowercase(get(ENV, "TREEAMR_TEST_BACKEND", "cpu"))

# At top level, in its own statement, so that everything after it is
# compiled in a world that can see the package.
if BNAME == "cuda"
    using CUDA
elseif BNAME == "metal"
    using Metal
elseif BNAME != "cpu"
    error("TREEAMR_TEST_BACKEND must be cpu, cuda or metal; got \"$BNAME\"")
end

const USE_MPI = "mpi" in ARGS
USE_MPI && MPI.Init()
const RANK = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const NRANKS = USE_MPI ? MPI.Comm_size(MPI.COMM_WORLD) : 1
const DEVICEAWARE = get(ENV, "TREEAMR_TEST_DEVICEAWARE", "0") == "1"
const COMM = USE_MPI ? communicator(MPI.COMM_WORLD; deviceaware=DEVICEAWARE) : nothing

# One device per rank on a node with several: the ranks of a node are
# numbered through a shared-memory split, and take the devices in turn.
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
const T = let want = get(ENV, "TREEAMR_TEST_T", "")
    isempty(want) ? (supports_float64(BACKEND) ? Float64 : Float32) :
    want == "Float64" ? Float64 : want == "Float32" ? Float32 :
    error("TREEAMR_TEST_T must be Float64 or Float32, got \"$want\"")
end

const OUT = stdout
emit(words...) = (RANK == 0 && println(OUT, join(words, " ")); nothing)

digest(bytes::AbstractVector{UInt8}) = bytes2hex(sha256(bytes))[1:32]
digest(v::Vector{<:Number}) = digest(reinterpret(UInt8, v))
digest(s::AbstractString) = digest(codeunits(s))

# Every rank's part of a per-block array, downloaded and gathered in rank
# order, which is block order.
gathered(forest, v) = TreeAMR.allgatherv(forest.comm, vec(Array(v)))

todev(a::AbstractArray) = (d = allocate(BACKEND, eltype(a), size(a)); copyto!(d, a); d)

# A forest whose coordinates are of the element type, so that every
# callback computes in `T` on the device; refined twice around `centre`.
function forest_of(roots::NTuple{D,Int}, N; periodic=ntuple(_ -> false, D),
                   reflecting=ntuple(_ -> (false, false), D), centre=nothing) where {D}
    forest = Forest(roots; N=N, periodic=periodic, reflecting=reflecting,
                    extents=ntuple(d -> (zero(T), T(roots[d])), D), comm=COMM)
    centre === nothing && return forest
    for lvl in 0:1
        r = 0.6 / 2^lvl
        targets = filter(forest.leaves) do k
            level(k) == lvl || return false
            ext = block_extent(forest, k)
            return all(d -> abs((ext[d][1] + ext[d][2]) / 2 - centre[d]) <= r, 1:D)
        end
        refine!(forest, targets)
        balance!(forest)
    end
    return forest
end

hasouter(forest::Forest{D}) where {D} =
    any(d -> !forest.periodic[d] && !all(forest.reflecting[d]), 1:D)

# Which stages of a schedule went through host mirrors on this rank, and
# how many have messages at all: a `#` line, since it depends on the
# partition, which `mpi_device_tests.jl` reads to assert that the path it
# meant to test is the one that ran.
function staging(forest, stages)
    remote = [st.remote for st in stages if st.remote !== nothing]
    counts = TreeAMR.allgather(forest.comm,
                               (length(remote), count(r -> !isempty(r.mirrors), remote)))
    return "messages", sum(first, counts), "staged", sum(last, counts)
end

fmt(x) = T === Float64 ? @sprintf("%.17g", x) : @sprintf("%.9g", x)

function reductions(tag, fs, u)
    emit(tag, "count", mesh_mapreduce(x -> 1, +, 0, fs))
    emit(tag, "linf", fmt(volume_weighted_norm(fs, u; p=Inf)))
    emit(tag, "max", fmt(mesh_mapreduce(identity, max, T(-Inf), fs; vars=1)))
    emit(tag, "sum l2", fmt(volume_weighted_norm(fs, u)))
    emit(tag, "sum mass", fmt(total_mass(fs, 1)))
    return nothing
end

# --- the wave equation ------------------------------------------------------

@kernel function wave_kernel!(du, @Const(work), @Const(spacings), ::Val{D},
                              ::Val{G}) where {D,G}
    I = @index(Global, NTuple)
    b = I[D + 1]
    c = ntuple(d -> I[d] + G[d], Val(D))
    u0 = work[c..., 1, b]
    laplacian = zero(eltype(du))
    for d in 1:D
        up = Base.setindex(c, c[d] + 1, d)
        um = Base.setindex(c, c[d] - 1, d)
        laplacian += work[up..., 1, b] - 2 * u0 + work[um..., 1, b]
    end
    h = spacings[b]
    du[ntuple(d -> I[d], Val(D))..., 1, b] = work[c..., 2, b]
    du[ntuple(d -> I[d], Val(D))..., 2, b] = laplacian / (h * h)
end

function wave_rhs!(du, u, fs, sched, spacings, ::Val{D}, ::Val{G}, boundary) where {D,G}
    scatter!(fs, u)
    fill_ghosts!(fs, sched; boundary=boundary)
    map_blocks!(wave_kernel!, fs, statearray(du, fs), fs.work, spacings, Val(D), Val(G))
    return du
end

function rk4!(u, fs, sched, dt, nsteps, ::Val{D}, ::Val{G}, boundary) where {D,G}
    spacings = todev(block_spacings(fs.forest, T))
    k1, k2, k3, k4, tmp = (similar(u) for _ in 1:5)
    for _ in 1:nsteps
        wave_rhs!(k1, u, fs, sched, spacings, Val(D), Val(G), boundary)
        @. tmp = u + (dt / 2) * k1
        wave_rhs!(k2, tmp, fs, sched, spacings, Val(D), Val(G), boundary)
        @. tmp = u + (dt / 2) * k2
        wave_rhs!(k3, tmp, fs, sched, spacings, Val(D), Val(G), boundary)
        @. tmp = u + dt * k3
        wave_rhs!(k4, tmp, fs, sched, spacings, Val(D), Val(G), boundary)
        @. u += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    end
    return u
end

# A Gaussian and its x₁-derivative-shaped partner, computed in the
# coordinates' type: a device kernel has no Float64 on Metal, and a
# captured `Type` is not `isbits`, so the constants are converted with
# `oftype` and the centre is captured as a tuple of `T`.
function pulse(x0::NTuple{D}, σ) where {D}
    c = map(T, x0)
    s = T(σ)
    return (x, v) -> begin
        r2 = zero(x[1])
        for d in 1:D
            r2 += (x[d] - c[d])^2
        end
        g = exp(-r2 / (2 * s * s))
        v == 1 ? g : (x[1] - c[1]) / (s * s) * g
    end
end

function wave_case(tag, forest::Forest{D}; G, centering, ops, steps,
                   parity=nothing) where {D}
    fs = FieldSet{T}(forest, 2; G=G, centering=centering, parity=parity, backend=BACKEND)
    initial = pulse(ntuple(d -> 0.37 * forest.roots[d] + 0.05d, D), 0.3)
    boundary = hasouter(forest) ? boundary_by_coordinates(initial) : nothing
    fill_by_coordinates!(initial, fs)
    sched = GhostSchedule(fs, ops)
    emit(tag, "leaves", nleaves(forest), maxlevel(forest), digest(string(forest.leaves)))
    fill_ghosts!(fs, sched; boundary=boundary)
    emit(tag, "filled", digest(gathered(forest, fs.work)))
    u = statevector(fs)
    gather!(u, fs)
    dt = T(0.2) * T(minimum_spacing(forest))
    rk4!(u, fs, sched, dt, steps, Val(D), Val(fs.G), boundary)
    emit(tag, "state", digest(gathered(forest, u)))
    scatter!(fs, u)
    fill_ghosts!(fs, sched; boundary=boundary)
    emit(tag, "work", digest(gathered(forest, fs.work)))
    reductions(tag, fs, u)
    emit("#", tag, "stages", staging(forest, sched.stages)...)
    return nothing
end

# --- Burgers' equation with the interface fixup -----------------------------

@inline function minmod(back, forw)
    z = zero(back)
    back * forw <= z && return z
    return abs(back) < abs(forw) ? back : forw
end

@kernel function flux_kernel!(flux, @Const(work), ::Val{D}, ::Val{GU}, ::Val{GF},
                              ::Val{d}) where {D,GU,GF,d}
    I = @index(Global, NTuple)
    b = I[D + 1]
    c = ntuple(e -> I[e] + GU[e], Val(D))
    m1 = Base.setindex(c, c[d] - 1, d)
    m2 = Base.setindex(c, c[d] - 2, d)
    p1 = Base.setindex(c, c[d] + 1, d)
    um2 = work[m2..., 1, b]
    um1 = work[m1..., 1, b]
    u0 = work[c..., 1, b]
    up1 = work[p1..., 1, b]
    uL = um1 + minmod(um1 - um2, u0 - um1) / 2
    uR = u0 - minmod(u0 - um1, up1 - u0) / 2
    a = max(abs(uL), abs(uR))
    flux[ntuple(e -> I[e] + GF[e], Val(D))..., 1, b] =
        (uL * uL / 2 + uR * uR / 2 - a * (uR - uL)) / 2
end

@kernel function divergence_kernel!(du, fluxes, @Const(spacings), ::Val{D},
                                    ::Val{GF}) where {D,GF}
    I = @index(Global, NTuple)
    b = I[D + 1]
    c = ntuple(e -> I[e] + GF[e], Val(D))
    acc = zero(eltype(du))
    for d in 1:D
        hi = Base.setindex(c, c[d] + 1, d)
        acc += fluxes[d][hi..., 1, b] - fluxes[d][c..., 1, b]
    end
    du[ntuple(e -> I[e], Val(D))..., 1, b] = -acc / spacings[b]
end

function burgers_rhs!(du, u, state, fluxes, sched, ischeds, spacings, ::Val{D},
                      ::Val{GU}, ::Val{GF}) where {D,GU,GF}
    scatter!(state, u)
    fill_ghosts!(state, sched)
    ntuple(Val(D)) do d
        map_blocks!(flux_kernel!, fluxes[d], fluxes[d].work, state.work, Val(D),
                    Val(GU), Val(GF), Val(d); closed=true)
        restrict_interfaces!(fluxes[d], ischeds[d])
        nothing
    end
    map_blocks!(divergence_kernel!, state, statearray(du, state),
                map(f -> f.work, fluxes), spacings, Val(D), Val(GF))
    return du
end

function burgers_case(tag, forest::Forest{D}; G, ops, steps) where {D}
    state = FieldSet{T}(forest, 1; G=G, backend=BACKEND)
    k0 = T(2π) / T(forest.roots[1])
    initial = (x, v) -> begin
        s = zero(x[1])
        for d in 1:D
            s += x[d]
        end
        one(s) + tanh(sin(k0 * s) * 10) / 2
    end
    fill_by_coordinates!(initial, state)
    sched = GhostSchedule(state, ops)
    fluxes = ntuple(d -> FieldSet{T}(forest, 1; G=0, centering=facecentered(D, d),
                                     backend=BACKEND), D)
    ischeds = ntuple(d -> InterfaceSchedule(fluxes[d]), D)
    emit(tag, "leaves", nleaves(forest), maxlevel(forest), digest(string(forest.leaves)))
    u = statevector(state)
    gather!(u, state)
    mass0 = total_mass(state, 1)
    spacings = todev(block_spacings(forest, T))
    dt = T(0.2) * T(minimum_spacing(forest)) / T(D * 1.5)
    args = (state, fluxes, sched, ischeds, spacings, Val(D), Val(state.G),
            Val(first(fluxes).G))
    k, u1, u2 = (similar(u) for _ in 1:3)
    for _ in 1:steps
        burgers_rhs!(k, u, args...)
        @. u1 = u + dt * k
        burgers_rhs!(k, u1, args...)
        @. u2 = (3 * u + u1 + dt * k) / 4
        burgers_rhs!(k, u2, args...)
        @. u = (u + 2 * (u2 + dt * k)) / 3
    end
    emit(tag, "state", digest(gathered(forest, u)))
    burgers_rhs!(k, u, args...)
    for d in 1:D
        emit(tag, "flux$d", digest(gathered(forest, fluxes[d].work)))
    end
    scatter!(state, u)
    reductions(tag, state, u)
    mass = total_mass(state, 1)
    # Conserved to the precision's roundoff over the steps, the fixup's
    # claim; the bound is a few thousand ulps of the total.
    emit(tag, "conserved", abs(mass - mass0) <= 4096 * eps(T) * abs(mass0))
    emit("#", tag, "stages", staging(forest, sched.stages)...)
    emit("#", tag, "interface stages",
         staging(forest, reduce(vcat, map(s -> s.stages, ischeds)))...)
    return nothing
end

# --- regridding --------------------------------------------------------------

function migration(old, new)
    owner(n, i) = TreeAMR.equalsplit_part(n, NRANKS, i) - 1
    oldindex = Dict(k => i for (i, k) in enumerate(old))
    up, down, straddling = 0, 0, 0
    for (j, k) in enumerate(new)
        i = get(oldindex, k, nothing)
        if i !== nothing
            up += owner(length(new), j) > owner(length(old), i)
            down += owner(length(new), j) < owner(length(old), i)
        elseif all(c -> haskey(oldindex, c), TreeAMR.childkeys(k))
            owners = Set(owner(length(old), oldindex[c]) for c in TreeAMR.childkeys(k))
            straddling += length(owners) > 1
        end
    end
    return up, down, straddling
end

function regridded(tag, forest, pairs; flags, buffer=0, boundary=nothing)
    old = copy(forest.leaves)
    changed = regrid!(forest, pairs; flags=flags, buffer=buffer, boundary=boundary)
    emit(tag, "regrid", changed, nleaves(forest), maxlevel(forest),
         digest(string(forest.leaves)))
    for (i, (fs, sched)) in enumerate(pairs)
        sched === nothing && continue
        emit(tag, "transferred$i", digest(gathered(forest, fs.work)))
    end
    emit("#", tag, "migrated", migration(old, forest.leaves)...)
    return changed
end

# The tracked pulse, flagged on the device through `firing_boxes`.
function tracked_pulse_case(tag; cycles, steps)
    D = 2
    forest = forest_of((4, 3), 8)
    fs = FieldSet{T}(forest, 2; G=1, centering=vertexcentered(D), backend=BACKEND)
    initial = pulse((1.3, 1.4), 0.35)
    boundary = boundary_by_coordinates(initial)
    fill_by_coordinates!(initial, fs)
    ops = Operators(prolongation=4, restriction=4)
    sched = GhostSchedule(fs, ops)
    threshold = T(0.2)
    fires(work, idx, b, x) = abs(work[idx..., 1, b]) > threshold
    for cycle in 1:cycles
        fill_ghosts!(fs, sched; boundary=boundary)
        flags = map(enumerate(firing_boxes(fires, fs))) do (b, (n, box))
            k = blockkey(fs, b)
            n == 0 && return level(k) > 0 ? Coarsen : Keep
            return level(k) < 2 ? (Refine, box) : (Keep, box)
        end
        regridded("$tag.$cycle", forest, (fs => sched,); flags=flags, buffer=2,
                  boundary=boundary)
        sched = GhostSchedule(fs, ops)
        u = statevector(fs)
        gather!(u, fs)
        rk4!(u, fs, sched, T(0.2) * T(minimum_spacing(forest)), steps, Val(D), Val(fs.G),
             boundary)
        emit("$tag.$cycle", "state", digest(gathered(forest, u)))
        scatter!(fs, u)
        emit("$tag.$cycle", "sum l2", fmt(volume_weighted_norm(fs, u)))
    end
    return nothing
end

# The first blocks of a uniform periodic mesh refined, which moves later
# blocks up the ranks, then coarsened, which moves them back down: over
# 4×4 roots with two refined, and over 2×2 roots with three refined,
# where at 2 and 3 ranks a group of siblings straddles a rank boundary
# when it is coarsened again (as in `mpi_workload.jl`). A cell-centered
# set with conservative operators, whose mass the transfer conserves,
# beside a vertex-centered one. The cycle runs twice, with the schedules
# rebuilt for every regrid as an application rebuilds them, and a `#`
# line after each says how many message buffers and host mirrors the
# ranks' pools have allocated so far: the second cycle must allocate
# none, since the first one's are kept (`BufferPool`).
function moving_blocks_case(tag)
    D = 2
    cons = Operators(family=Conservative, prolongation=3, restriction=2)
    lagr = Operators(prolongation=2, restriction=2)
    conserved = true
    for (shape, refined) in (((4, 4), 2), ((2, 2), 3))
        forest = forest_of(shape, 8; periodic=(true, true))
        a = FieldSet{T}(forest, 2; G=2, backend=BACKEND)
        b = FieldSet{T}(forest, 1; G=1, centering=vertexcentered(D), backend=BACKEND)
        fill_by_coordinates!(pulse((0.8, 1.1), 0.4), a)
        fill_by_coordinates!(pulse((1.2, 0.7), 0.5), b)
        pairs() = (a => GhostSchedule(a, cons), b => GhostSchedule(b, lagr))
        name = "$tag$(shape[1])"
        mass = total_mass(a, 1)
        for round in 1:2
            suffix = round == 1 ? "" : string(round)
            offset = first(blockrange(forest)) - 1
            flags = RegridFlag[offset + i <= refined ? Refine : Keep for i in 1:nblocks(a)]
            regridded("$name.refine$suffix", forest, pairs(); flags=flags)
            offset = first(blockrange(forest)) - 1
            flags = RegridFlag[level(forest.leaves[offset + i]) > 0 ? Coarsen : Keep
                               for i in 1:nblocks(a)]
            regridded("$name.coarsen$suffix", forest, pairs(); flags=flags)
            pool = forest.state.pool
            counts = TreeAMR.allgather(forest.comm, pool === nothing ? (0, 0) :
                                                    (pool.allocated, pool.pagelocked))
            emit("#", name, "pool", round, "allocated", sum(first, counts), "mirrors",
                 sum(last, counts))
        end
        conserved &= abs(total_mass(a, 1) - mass) <= 256 * eps(T) * abs(mass)
    end
    emit(tag, "conserved", conserved)
    return nothing
end

function main()
    emit("# ranks", NRANKS, "backend", BNAME, "T", T, "deviceaware", DEVICEAWARE)
    wave_case("W2v", forest_of((4, 4), 8; centre=(1.6, 2.3)); G=1,
              centering=vertexcentered(2), ops=Operators(prolongation=4, restriction=4),
              steps=4)
    parity = [(OddParity, EvenParity), (EvenParity, EvenParity)]
    wave_case("R2c", forest_of((3, 3), 8; reflecting=((true, false), (true, false)),
                               centre=(0.4, 0.5)); G=2, centering=cellcentered(2),
              ops=Operators(prolongation=4, restriction=4), steps=4, parity=parity)
    wave_case("W3c", forest_of((2, 2, 2), 4; periodic=(true, true, true),
                               centre=(0.7, 0.8, 1.2)); G=1, centering=cellcentered(3),
              ops=Operators(prolongation=2, restriction=2), steps=3)
    burgers_case("B2", forest_of((4, 4), 8; periodic=(true, true), centre=(1.6, 2.3));
                 G=2, ops=Operators(family=Conservative, prolongation=3, restriction=2),
                 steps=4)
    tracked_pulse_case("TP"; cycles=2, steps=3)
    moving_blocks_case("M")
    return nothing
end

try
    main()
catch err
    showerror(stderr, err, catch_backtrace())
    println(stderr)
    USE_MPI && MPI.Abort(MPI.COMM_WORLD, 1)
    rethrow()
end
