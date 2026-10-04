# The workload behind the M7 rank-count independence test.
#
# Run as a standalone script, it prints on rank 0 a digest of everything
# a distributed mesh produces, in lines that must not depend on the
# number of ranks:
#
#     mpiexec -n 3 julia --project=test test/mpi_workload.jl mpi
#     julia --project=test test/mpi_workload.jl                 # serially
#
# The field data are gathered to every rank in block order before they
# are digested — the state vector after several time steps, and the
# working arrays *with their ghosts* after a fill, so the exchange is
# checked bit for bit, ghosts included, and not only through what the
# ghosts did to the interior. Integer, max and min reductions are
# printed exactly. The floating-point sums are printed on lines whose
# second word is `sum`; they are promised to roundoff across rank counts
# only ("Reductions" under "Distributed meshes" in CODE.md), and the
# test compares them with a tolerance. Every other line must be
# identical byte for byte. Lines starting with `#` depend on the rank
# count — the refusals and the negative control — and are checked
# separately by `mpi_tests.jl`.
#
# Without the argument `mpi` the forests are serial, which is the
# reference; `mpi_tests.jl` runs it in its own process, into an
# `IOBuffer` it defines as `WORKLOAD_IO`. Deliberately self-contained —
# its own RK4 and SSPRK3, no ODE package, no test helpers — so that a
# rank starts in seconds. From step 4 of M7 on it regrids too: the
# leaves and the transferred arrays after every regrid are digested
# before any ghost fill, so the transfer itself is checked bit for bit.
# From step 5 on it interpolates: every rank queries its own slice of one
# global point list, and the answers gathered in rank order are the
# serial answers in global order. From step 6 on it checkpoints: a run
# saved after a regrid, with one part file per rank, per two groups and
# per node (step 6b), and continued from each file at the same rank count
# must print what the uninterrupted run prints; the files of other runs,
# written at other rank counts, are loaded at this one on `#` lines
# (`TREEAMR_CHECKPOINT_DIR`, and `TREEAMR_CHECKPOINT_FROM` for the rank
# counts whose files to wait for and load, which lets `mpi_tests.jl` run
# the launches at once); and the version-1 fixtures, written by the
# shared-file writer's version, load at every rank count. From M12 on it
# runs a rotating quadrant too: the wave, a set that turns into itself
# and a `RotationPair` through a fill, a regrid, interpolation beyond the
# seam and a checkpoint, loaded at the other rank counts as well.

using TreeAMR
using MPI: MPI
using KernelAbstractions: @kernel, @index, @Const
using Printf: @sprintf
using SHA: sha256
using MultiFloats: Float32x2
using HDF5: HDF5
using HDF5.Filters: Shuffle, Deflate

const USE_MPI = "mpi" in ARGS
USE_MPI && MPI.Init()
const COMM = USE_MPI ? MPI.COMM_WORLD : nothing
const RANK = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const NRANKS = USE_MPI ? MPI.Comm_size(MPI.COMM_WORLD) : 1
const OUT = isdefined(@__MODULE__, :WORKLOAD_IO) ? WORKLOAD_IO : stdout
# Under MPI the checkpoints' messages are cut at 4 KiB (a test hook of
# the HDF5 extension; 64 MiB otherwise), so that a rank's blocks travel to
# its I/O process, and a part's from its reader, in several messages.
const CKPT = Base.get_extension(TreeAMR, :TreeAMRHDF5Ext)
USE_MPI && (CKPT.MAX_MESSAGE[] = 4096)

emit(words...) = (RANK == 0 && println(OUT, join(words, " ")); nothing)

digest(bytes::AbstractVector{UInt8}) = bytes2hex(sha256(bytes))[1:32]
digest(v::Vector{<:AbstractFloat}) = digest(reinterpret(UInt8, v))
digest(s::AbstractString) = digest(codeunits(s))

# Every rank's part of a per-block vector, in rank order, which is block
# order: the ranks own contiguous runs of the curve.
gathered(forest, v) = TreeAMR.allgatherv(forest.comm, collect(vec(v)))

# A forest refined twice around `centre`, so that three levels meet. A
# forest mutation is collective: every rank makes the same calls.
function forest_of(roots::NTuple{D,Int}, N; periodic=ntuple(_ -> false, D),
                   reflecting=ntuple(_ -> (false, false), D), rotating=nothing,
                   centre=nothing, levels=2) where {D}
    forest = Forest(roots; N=N, periodic=periodic, reflecting=reflecting,
                    rotating=rotating, comm=COMM)
    centre === nothing && return forest
    for lvl in 0:(levels - 1)
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

# The serial schedule's transfer count, summed over the ranks: each
# rank's local transfers and those it receives. The ones it sends are
# another rank's received ones, so the totals sent and received agree.
function transfers(forest, groups, stages)
    nlocal = sum(TreeAMR.ntransfers, groups; init=0)
    layouts(f) = sum(st -> st.remote === nothing ? 0 : length(f(st.remote)), stages;
                     init=0)
    nrecv, nsent = layouts(r -> r.recvlayout), layouts(r -> r.sendlayout)
    counts = TreeAMR.allgather(forest.comm, (nlocal + nrecv, nsent, nrecv))
    total = sum(first, counts)
    balanced = sum(c -> c[2], counts) == sum(c -> c[3], counts)
    return "$total balanced $balanced"
end
transfers(forest, s::GhostSchedule) =
    transfers(forest, [s.phase1; reduce(vcat, s.phase2; init=eltype(s.phase1)[])],
              s.stages)
transfers(forest, s::InterfaceSchedule) =
    transfers(forest, reduce(vcat, s.phases; init=eltype(eltype(s.phases))[]), s.stages)

# The reductions of one variable: exact ones as they are, sums marked.
function reductions(tag, fs, u)
    emit(tag, "count", mesh_mapreduce(x -> 1, +, 0, fs))
    emit(tag, "linf", @sprintf("%.17g", volume_weighted_norm(fs, u; p=Inf)))
    emit(tag, "min", @sprintf("%.17g", mesh_mapreduce(identity, min, Inf, fs; vars=1)))
    emit(tag, "negmax",
         @sprintf("%.17g", mesh_mapreduce(x -> -abs(x) - 1, max, -Inf, fs; vars=1)))
    # An `init` that is not neutral, under a weight: every block's value
    # is `0.5 max(1, …) = 0.5`, so a rank with no blocks must contribute
    # nothing rather than its `init`, or this would print 1.
    emit(tag, "initmax",
         @sprintf("%.17g", mesh_mapreduce(x -> -abs(x), max, 1.0, fs; vars=1,
                                          weight=k -> 0.5)))
    emit(tag, "sum l2", @sprintf("%.17g", volume_weighted_norm(fs, u)))
    emit(tag, "sum mass", @sprintf("%.17g", total_mass(fs, 1)))
    return nothing
end

# --- the wave equation, as in thread_workload.jl ---------------------------

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
    spacings = block_spacings(fs.forest)
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

pulse(x0, σ) = (x, v) -> begin
    g = exp(-sum(d -> (x[d] - x0[d])^2, 1:length(x)) / (2σ^2))
    v == 1 ? g : (x[1] - x0[1]) * g
end

# A message corrupted in flight, as a negative control: rewrite one
# rank's received ghosts from its last receive buffer with one value
# changed, and report whether the gathered digest notices. Collective.
function perturbed_digest_differs(forest, fs, sched, before)
    stage = findlast(st -> st.remote !== nothing && !isempty(st.remote.recvlayout),
                     sched.stages)
    has = TreeAMR.allgather(forest.comm, stage !== nothing)
    victim = findlast(has)
    victim === nothing && return nothing
    if RANK == victim - 1
        remote = sched.stages[stage].remote
        send, recv = remote.buffers[fs.nvars]
        recv[1] = nextfloat(recv[1])
        TreeAMR.unpack_stage!(fs, remote, (send, recv), TreeAMR.CPU())
    end
    return digest(gathered(forest, fs.work)) != before
end

function wave_case(tag, forest::Forest{D}; G, centering, ops, steps, parity=nothing,
                   rotation=nothing, control=false) where {D}
    fs = FieldSet(forest, 2; G=G, centering=centering, parity=parity, rotation=rotation)
    initial = pulse(ntuple(d -> 0.37 * forest.roots[d] + 0.05d, D), 0.3)
    boundary = hasouter(forest) ? boundary_by_coordinates(initial) : nothing
    fill_by_coordinates!(initial, fs)
    sched = GhostSchedule(fs, ops)
    emit(tag, "leaves", nleaves(forest), maxlevel(forest), digest(string(forest.leaves)))
    emit(tag, "transfers", transfers(forest, sched))
    fill_ghosts!(fs, sched; boundary=boundary)
    emit(tag, "filled", digest(gathered(forest, fs.work)))

    u = statevector(fs)
    gather!(u, fs)
    dt = 0.2 * minimum_spacing(forest)
    rk4!(u, fs, sched, dt, steps, Val(D), Val(fs.G), boundary)
    emit(tag, "state", digest(gathered(forest, u)))
    scatter!(fs, u)
    fill_ghosts!(fs, sched; boundary=boundary)
    work = digest(gathered(forest, fs.work))
    emit(tag, "work", work)
    reductions(tag, fs, u)
    if control
        differs = perturbed_digest_differs(forest, fs, sched, work)
        differs === nothing || emit("#", tag, "perturbed-ghost-changes-digest", differs)
    end
    return nothing
end

# --- every centering through one fill, and the interface restriction ------

# Data that no stencil reproduces by accident: a function of the global
# leaf, the stored index and the variable alone, so it is the same at
# any rank count, written over every stored point of a block. Smooth
# data would let a wrong source or a wrong weight go unnoticed wherever
# the two interpolants agree, and would make the injection at a
# vertex-like coarse-fine face a no-op.
function pseudorandom!(fs::FieldSet{T,D}) where {T,D}
    offset = first(blockrange(fs.forest)) - 1
    for b in 1:nblocks(fs), v in 1:fs.nvars
        block = blockview(fs, b, v)
        for idx in CartesianIndices(block)
            h = hash((offset + b, Tuple(idx), v))
            block[idx] = T((h >> 11) * 0x1p-53) - T(1) / 2
        end
    end
    return fs
end

function fill_case(tag, forest::Forest{D}, centerings; G, ops) where {D}
    emit(tag, "leaves", nleaves(forest), maxlevel(forest), digest(string(forest.leaves)))
    walls = findall(d -> any(forest.reflecting[d]), 1:D)
    f = (x, v) -> sum(d -> (d + v) * x[d]^2 - x[d], 1:D) + v
    for (name, C) in centerings
        parity = isempty(walls) ? nothing :
                 [ntuple(d -> d == first(walls) ? OddParity : EvenParity, D),
                  ntuple(_ -> EvenParity, D)]
        fs = FieldSet(forest, 2; G=G, centering=C, parity=parity)
        pseudorandom!(fs)
        sched = GhostSchedule(fs, ops)
        boundary = hasouter(forest) ? boundary_by_coordinates(f) : nothing
        fill_ghosts!(fs, sched; boundary=boundary)
        emit(tag, name, "transfers", transfers(forest, sched))
        emit(tag, name, "filled", digest(gathered(forest, fs.work)))
        any(==(:vertex), C) || continue
        isched = InterfaceSchedule(fs)
        restrict_interfaces!(fs, isched)
        emit(tag, name, "interfaces", transfers(forest, isched))
        emit(tag, name, "restricted", digest(gathered(forest, fs.work)))
    end
    return nothing
end

# --- Burgers' equation with the interface fixup, as in thread_workload.jl --

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

@inline function minmod(back, forw)
    z = zero(back)
    back * forw <= z && return z
    return abs(back) < abs(forw) ? back : forw
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
    state = FieldSet(forest, 1; G=G)
    initial = (x, v) -> 1.0 + 0.5 * tanh(sin(2π * sum(x) / forest.roots[1]) / 0.1)
    fill_by_coordinates!(initial, state)
    sched = GhostSchedule(state, ops)
    fluxes = ntuple(d -> FieldSet(forest, 1; G=0, centering=facecentered(D, d)), D)
    ischeds = ntuple(d -> InterfaceSchedule(fluxes[d]), D)
    emit(tag, "leaves", nleaves(forest), maxlevel(forest), digest(string(forest.leaves)))
    emit(tag, "transfers", transfers(forest, sched))
    for d in 1:D
        emit(tag, "interfaces$d", transfers(forest, ischeds[d]))
    end

    u = statevector(state)
    gather!(u, state)
    mass0 = total_mass(state, 1)
    spacings = block_spacings(forest)
    dt = 0.2 * minimum_spacing(forest) / (D * 1.5)
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
    # The fluxes of the final state, after the fixup, ghosts and all.
    burgers_rhs!(k, u, args...)
    for d in 1:D
        emit(tag, "flux$d", digest(gathered(forest, fluxes[d].work)))
    end
    scatter!(state, u)
    reductions(tag, state, u)
    mass = total_mass(state, 1)
    emit(tag, "conserved", abs(mass - mass0) <= 1e-13 * abs(mass0))
    return nothing
end

# --- regridding (step 4 of M7) ---------------------------------------------

# How a regrid moved blocks between ranks, from the leaves before and
# after: kept blocks whose owner rose and fell, and coarsened blocks
# whose `2^D` children had more than one owner. A function of the rank
# count, so it goes on `#` lines.
function migration(forest::Forest{D}, old, new) where {D}
    P = NRANKS
    owner(n, i) = TreeAMR.equalsplit_part(n, P, i) - 1
    oldindex = Dict(k => i for (i, k) in enumerate(old))
    up, down, straddling = 0, 0, 0
    for (j, k) in enumerate(new)
        i = get(oldindex, k, nothing)
        if i !== nothing
            up += owner(length(new), j) > owner(length(old), i)
            down += owner(length(new), j) < owner(length(old), i)
        elseif level(k) < TreeAMR.MAX_LEVEL &&
               all(c -> haskey(oldindex, c), TreeAMR.childkeys(k))
            owners = Set(owner(length(old), oldindex[c]) for c in TreeAMR.childkeys(k))
            straddling += length(owners) > 1
        end
    end
    return up, down, straddling
end

# A regrid of every pair, followed by the digests that do not depend on
# the rank count — the leaves, and each field set's arrays as the
# transfer left them, ghosts zero — and the `#` line that does.
function regridded(tag, forest, pairs; flags, buffer=0, boundary=nothing)
    old = copy(forest.leaves)
    changed = regrid!(forest, pairs; flags=flags, buffer=buffer, boundary=boundary)
    emit(tag, "regrid", changed, nleaves(forest), maxlevel(forest),
         digest(string(forest.leaves)))
    for (i, (fs, sched)) in enumerate(pairs)
        sched === nothing && continue
        if fs isa RotationPair                      # both members (M12)
            emit(tag, "transferred$i", digest(gathered(forest, fs.a.work)),
                 digest(gathered(forest, fs.b.work)))
        else
            emit(tag, "transferred$i", digest(gathered(forest, fs.work)))
        end
    end
    up, down, straddling = migration(forest, old, forest.leaves)
    emit("#", tag, "migrated", up, "up", down, "down", straddling, "straddling")
    return changed
end

# The tracked pulse: a right-moving Gaussian, vertex-centered, against
# outer faces with the hook, refined where it is large through
# `firing_boxes` (local) and regridded between RK4 steps.
pulse_fires(work, idx, b, x) = abs(work[idx..., 1, b]) > 0.2

function moving_pulse(x0, σ)
    return (x, v) -> begin
        g = exp(-sum(d -> (x[d] - x0[d])^2, 1:length(x)) / (2σ^2))
        v == 1 ? g : (x[1] - x0[1]) / σ^2 * g
    end
end

function pulse_flags(fs, lmax)
    return map(enumerate(firing_boxes(pulse_fires, fs))) do (b, (n, box))
        k = blockkey(fs, b)
        n == 0 && return level(k) > 0 ? Coarsen : Keep
        return level(k) < lmax ? (Refine, box) : (Keep, box)
    end
end

function tracked_pulse_case(tag; cycles, steps)
    forest = forest_of((4, 3), 8)
    D = 2
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(D))
    initial = moving_pulse((1.3, 1.4), 0.35)
    boundary = boundary_by_coordinates(initial)
    fill_by_coordinates!(initial, fs)
    sched = GhostSchedule(fs, OPS4)
    for cycle in 1:cycles
        fill_ghosts!(fs, sched; boundary=boundary)
        regridded("$tag.$cycle", forest, (fs => sched,); flags=pulse_flags(fs, 2),
                  buffer=2, boundary=boundary)
        sched = GhostSchedule(fs, OPS4)
        u = statevector(fs)
        gather!(u, fs)
        rk4!(u, fs, sched, 0.2 * minimum_spacing(forest), steps, Val(D), Val(fs.G),
             boundary)
        emit("$tag.$cycle", "state", digest(gathered(forest, u)))
        scatter!(fs, u)
        emit("$tag.$cycle", "sum l2", @sprintf("%.17g", volume_weighted_norm(fs, u)))
    end
    return nothing
end

# Burgers' shock through regrid cycles, with the fixup: conservative
# operators, so the regrid conserves mass as the steps do, and the flux
# sets are only resized (`fs => nothing`).
function burgers_fires(work, idx, b, x)
    m = 0.0
    for d in 1:length(idx)
        hi = Base.setindex(idx, idx[d] + 1, d)
        lo = Base.setindex(idx, idx[d] - 1, d)
        m = max(m, abs(work[hi..., 1, b] - work[lo..., 1, b]))
    end
    return m > 0.25
end

function burgers_regrid_case(tag; cycles, steps)
    D = 2
    forest = forest_of((4, 4), 8; periodic=(true, true))
    state = FieldSet(forest, 1; G=2)
    initial = (x, v) -> 1.0 + 0.5 * tanh((sin(π * x[1] / 2) + 0.3 * sin(π * x[2] / 2)) / 0.1)
    fill_by_coordinates!(initial, state)
    fluxes = ntuple(d -> FieldSet(forest, 1; G=0, centering=facecentered(D, d)), D)
    mass0 = total_mass(state, 1)
    conserved = true
    sched = GhostSchedule(state, OPSC)
    for cycle in 1:cycles
        fill_ghosts!(state, sched)
        flags = map(enumerate(firing_boxes(burgers_fires, state))) do (b, (n, box))
            k = blockkey(state, b)
            n == 0 && return level(k) > 0 ? Coarsen : Keep
            return level(k) < 2 ? (Refine, box) : (Keep, box)
        end
        before = total_mass(state, 1)
        pairs = (state => sched, map(f -> f => nothing, fluxes)...)
        regridded("$tag.$cycle", forest, pairs; flags=flags, buffer=3)
        after = total_mass(state, 1)
        conserved &= abs(after - before) <= 1e-13 * abs(before)
        sched = GhostSchedule(state, OPSC)
        ischeds = ntuple(d -> InterfaceSchedule(fluxes[d]), D)
        u = statevector(state)
        gather!(u, state)
        spacings = block_spacings(forest)
        dt = 0.2 * minimum_spacing(forest) / (D * 1.5)
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
        scatter!(state, u)
        emit("$tag.$cycle", "state", digest(gathered(forest, u)))
        emit("$tag.$cycle", "sum mass", @sprintf("%.17g", total_mass(state, 1)))
    end
    mass = total_mass(state, 1)
    emit(tag, "conserved", conserved && abs(mass - mass0) <= 1e-13 * abs(mass0))
    return nothing
end

# The initial-data cycle from a single leaf, so that at three ranks two
# start without blocks: once with the host flag callback, vertex-
# centered, and once with a flag vector from `firing_boxes`,
# cell-centered.
function adapt_case(tag)
    D = 2
    ring = (x, v) -> tanh((sqrt((x[1] - 0.45)^2 + (x[2] - 0.55)^2) - 0.25) / 0.05) + v
    forest = forest_of((1, 1), 8)
    fs = FieldSet(forest, 1; G=1, centering=vertexcentered(D))
    flag = (b, k) -> begin
        ext = block_extent(forest, k)
        c = ntuple(d -> (ext[d][1] + ext[d][2]) / 2, D)
        r = sqrt((c[1] - 0.45)^2 + (c[2] - 0.55)^2)
        abs(r - 0.25) < 0.75 / 2^level(k) && level(k) < 3 ? Refine : Keep
    end
    sched, passes, converged = adapt_to_initial_data!(fs, OPS2; initial=ring, flag=flag,
                                                      boundary=boundary_by_coordinates(ring))
    emit(tag, "flag", passes, converged, nleaves(forest), maxlevel(forest),
         digest(string(forest.leaves)), digest(gathered(forest, fs.work)))

    forest = forest_of((1, 1), 8)
    fs = FieldSet(forest, 1; G=2)
    fires(work, idx, b, x) = abs(work[idx..., 1, b]) < 0.9
    flags = fs -> map(enumerate(firing_boxes(fires, fs))) do (b, (n, box))
        n == 0 && return Keep
        return level(blockkey(fs, b)) < 3 ? (Refine, box) : (Keep, box)
    end
    sched, passes, converged = adapt_to_initial_data!(fs, OPS4; initial=ring, flags=flags,
                                                      buffer=2,
                                                      boundary=boundary_by_coordinates(ring))
    emit(tag, "flags", passes, converged, nleaves(forest), maxlevel(forest),
         digest(string(forest.leaves)), digest(gathered(forest, fs.work)))
    return nothing
end

# Blocks moving between ranks in both directions, and coarsenings whose
# siblings had different owners, over several field sets of different
# variable counts, ghost widths and centerings at once. 2×2 roots with
# three of them refined are 13 leaves, so at 2, 3 and 4 ranks a sibling
# group straddles a rank boundary; refining the first blocks of a
# uniform mesh moves every later block up the ranks, and coarsening them
# again moves them back. In another element type (`full = false`) only
# the cell-centered set over the 13 leaves, whose kernels are what each
# element type compiles anew.
function moving_blocks_case(tag, ::Type{T}; full=true) where {T}
    D = 2
    sets(forest) = full ?
                   (FieldSet{T}(forest, 2; G=2) => OPS4,
                    FieldSet{T}(forest, 1; G=1, centering=vertexcentered(D)) => OPS2,
                    FieldSet{T}(forest, 3; G=2, centering=facecentered(D, 2)) => OPS4) :
                   (FieldSet{T}(forest, 2; G=2) => OPS4,)
    first_blocks(forest, n, flag) = begin
        offset = first(blockrange(forest)) - 1
        RegridFlag[offset + b <= n ? flag : Keep for b in 1:length(blockrange(forest))]
    end
    for (shape, refined) in (full ? (((4, 4), 2), ((2, 2), 3)) : (((2, 2), 3),))
        forest = forest_of(shape, 8; periodic=(true, true))
        made = sets(forest)
        fss = map(first, made)
        foreach(pseudorandom!, fss)
        scheds() = map(p -> GhostSchedule(p.first, p.second), made)
        name = "$tag$(shape[1])"
        pairs = map(=>, fss, scheds())
        regridded("$name.refine", forest, pairs; flags=first_blocks(forest, refined, Refine))
        # Every child of the refined roots asks to coarsen.
        offset = first(blockrange(forest)) - 1
        flags = RegridFlag[level(forest.leaves[offset + b]) > 0 ? Coarsen : Keep
                           for b in 1:length(blockrange(forest))]
        pairs = map(=>, fss, scheds())
        foreach(p -> fill_ghosts!(p...), pairs)
        emit(name, "filled", join((digest(gathered(forest, fs.work)) for fs in fss), " "))
        regridded("$name.coarsen", forest, pairs; flags=flags)
        # Nothing asked for: no change, on every rank.
        pairs = map(=>, fss, scheds())
        emit(name, "unchanged",
             regrid!(forest, pairs; flags=fill(Keep, length(blockrange(forest)))))
    end
    return nothing
end

# --- point interpolation (step 5 of M7) ------------------------------------

# A global list of points, the same at any rank count, of which rank `r`
# queries a contiguous slice — unevenly, rank 1 none — so that gathering
# the answers in rank order gives them in global order. The points cover
# the domain and run half a period beyond the periodic dimension and
# beyond the reflecting wall, so they are wrapped and mirrored, and
# nearly every one lives on another rank than the one that asks.
function interpolation_points(n)
    φ = (sqrt(5.0) - 1) / 2
    xs = [(-1.5 + 6.0 * mod(j * φ, 1.0), -1.5 + 4.5 * mod(j * φ^2, 1.0)) for j in 1:n]
    weights = [(2, 0, 5, 1, 3)[r % 5 + 1] for r in 0:(NRANKS - 1)]
    cuts = [0; round.(Int, n .* cumsum(weights) ./ sum(weights))]
    return xs[(cuts[RANK + 1] + 1):cuts[RANK + 2]]
end

function interpolate_case(tag)
    D = 2
    forest = forest_of((3, 3), 8; periodic=(true, false),
                       reflecting=((false, false), (true, false)), centre=(1.4, 0.5))
    emit(tag, "leaves", nleaves(forest), maxlevel(forest), digest(string(forest.leaves)))
    parity = [(EvenParity, OddParity), (EvenParity, EvenParity)]
    xs = interpolation_points(301)
    for T in (Float64, Float32x2)
        fs = FieldSet{T}(forest, 2; G=2, parity=parity)
        pseudorandom!(fs)
        f = (x, v) -> v + x[1] / 7 + x[2]^2 / 5
        fill_ghosts!(fs, GhostSchedule(fs, OPS4); boundary=boundary_by_coordinates(f))
        name = "$tag." * (T === Float64 ? "F64" : "F32x2")
        if T === Float64
            r = interpolate(fs, xs, Lagrange(4);
                            derivs=((0, 0), (1, 0), (0, 1), (1, 1), (0, 2)),
                            vars=[2, 1], exclude=Ellipsoid((1.4, 0.5), (0.4, 0.3)))
            flags = gathered(forest, r.excluded)
            emit(name, "excluded", count(flags), digest(reinterpret(UInt8, flags)))
            emit(name, "derivatives", digest(gathered(forest, r.values)))
        end
        r = interpolate(fs, xs, Lagrange(4))
        emit(name, "values", length(gathered(forest, r.excluded)),
             digest(gathered(forest, r.values)))
    end
    return nothing
end

# --- ranks without blocks ---------------------------------------------------

# Two leaves, so that at three ranks one rank holds no block, through
# every operation an application makes on it: the all-variables forms of
# the fill and the boundary hook, the wave equation, the interface
# restriction, interpolation, regrids that give the empty rank blocks
# and take them away again, the initial-data cycle and a checkpoint. A
# step that assumed a rank has a block 1 throws there while the others
# wait, which `main` turns into an abort: the all-variables hook's length
# check did, until it was guarded in 0.1.6.
const EMPTY_PARITY = [(OddParity, EvenParity), (EvenParity, EvenParity)]
const EMPTY_DATA = AllVariables(x -> (sin(2x[1]) + x[2] / 3, x[1] * x[2] - 1))
const EMPTY_HOOK = CellBoundary(AllVariables((x, δ) -> EMPTY_DATA.f(x)))

function empty_rank_case(tag)
    D = 2
    forest = forest_of((2, 1), 8; reflecting=((true, false), (false, false)))
    emit(tag, "leaves", nleaves(forest), maxlevel(forest), digest(string(forest.leaves)))
    empties() = count(iszero, TreeAMR.allgather(forest.comm, length(blockrange(forest))))
    emit("#", tag, "empty ranks", empties())
    fs = FieldSet(forest, 2; G=2, parity=EMPTY_PARITY)
    fill_by_coordinates!(EMPTY_DATA, fs)
    sched = GhostSchedule(fs, OPS4)
    fill_ghosts!(fs, sched; boundary=EMPTY_HOOK)
    emit(tag, "filled", digest(gathered(forest, fs.work)))
    u = statevector(fs)
    gather!(u, fs)
    rk4!(u, fs, sched, 0.2 * minimum_spacing(forest), 3, Val(D), Val(fs.G), EMPTY_HOOK)
    emit(tag, "state", digest(gathered(forest, u)))
    scatter!(fs, u)
    reductions(tag, fs, u)

    vfs = FieldSet(forest, 2; G=0, centering=vertexcentered(D), parity=EMPTY_PARITY)
    pseudorandom!(vfs)
    restrict_interfaces!(vfs, InterfaceSchedule(vfs))
    emit(tag, "restricted", digest(gathered(forest, vfs.work)))

    # Rank 0 asks for every point, beyond the wall too; the others for none.
    fill_ghosts!(fs, sched; boundary=EMPTY_HOOK)
    pts = [(-0.4 + 2.35 * mod(j * 0.618034, 1.0), 0.97 * mod(j * 0.381966, 1.0))
           for j in 1:23]
    r = interpolate(fs, RANK == 0 ? pts : similar(pts, 0), Lagrange(4);
                    derivs=((0, 0), (1, 0)))
    emit(tag, "interpolated", digest(gathered(forest, r.values)))

    # Refined, every rank has blocks; coarsened again, the third has none.
    regridded("$tag.refine", forest, (fs => sched, vfs => nothing);
              flags=fill(Refine, length(blockrange(forest))), boundary=EMPTY_HOOK)
    sched = GhostSchedule(fs, OPS4)
    fill_ghosts!(fs, sched; boundary=EMPTY_HOOK)
    emit(tag, "refined", digest(gathered(forest, fs.work)))
    regridded("$tag.coarsen", forest, (fs => sched,);
              flags=fill(Coarsen, length(blockrange(forest))), boundary=EMPTY_HOOK)
    sched = GhostSchedule(fs, OPS4)
    fill_ghosts!(fs, sched; boundary=EMPTY_HOOK)
    emit(tag, "coarsened", digest(gathered(forest, fs.work)))
    emit("#", tag, "empty ranks after the regrids", empties())

    path = joinpath(checkpoint_dir(), "$tag-n$NRANKS.h5")
    save_checkpoint(path, forest; fieldsets=("u" => fs,), application="Empty" => 1)
    ck = load_checkpoint(path; comm=COMM)
    back = ck.fieldsets["u"].fieldset
    v = statevector(back)
    gather!(v, back)
    gather!(u, fs)
    emit(tag, "checkpoint", nleaves(ck.forest), gathered(forest, u) == gathered(forest, v))

    # The initial-data cycle from the two leaves, in both callback forms.
    for (name, initial) in (("all", EMPTY_DATA), ("each", (x, v) -> EMPTY_DATA.f(x)[v]))
        forest = forest_of((2, 1), 8; reflecting=((true, false), (false, false)))
        fs = FieldSet(forest, 2; G=2, parity=EMPTY_PARITY)
        _, passes, converged = adapt_to_initial_data!(
            fs, OPS4; initial=initial, boundary=EMPTY_HOOK,
            flag=(b, k) -> level(k) < 1 && block_extent(forest, k)[1][1] < 0.5 ?
                           Refine : Keep)
        emit(tag, "adapted", name, passes, converged, nleaves(forest),
             digest(gathered(forest, fs.work)))
    end
    return nothing
end

# --- rotating seams (M12) ---------------------------------------------------

# A quadrant `[0, 3]^2` with a rotating seam, refined at the axis and the
# seam, through every operation: the fill of a set that turns into itself
# — a scalar and a vector, cell-centered — and of a `RotationPair` of
# face-centered sets, `(B_1, F_1)` and `(F_2, B_2)`, whose stages are
# merged and share their tags; a regrid of both, the pair filled as a pair
# before its transfer; interpolation over the whole plane, three quarters
# of the points turned back across the seam; and a checkpoint, saved and
# loaded at this rank count and at the others (`rotating_cross`). Each
# set has a variable that is zero everywhere, the vector's `v_2` and each
# member's second, so the turned ghosts hold `−0`, which a pack that
# applied the sign would turn into `+0`; the digests are of the bytes.
const QUAD_DATA = (x, v) -> v + x[1] / 7 - x[2]^2 / 5 + x[1] * x[2] / 11
const QUAD_HOOK = boundary_by_coordinates(QUAD_DATA)

# The rotated transfers this rank receives from another, summed over the
# ranks: none serially, some wherever the seam's transfers cross ranks.
function rotated_received(forest, scheds...)
    mine = sum(scheds) do sched
        sum(st -> st.remote === nothing ? 0 :
                  count(e -> e.key.orientation != 0, st.remote.recvlayout),
            sched.stages; init=0)
    end
    return sum(TreeAMR.allgather(forest.comm, mine))
end

# Pseudorandom data with variable `zero` set to zero, ghosts included.
function quad_data!(fs, zero)
    pseudorandom!(fs)
    for b in 1:nblocks(fs)
        blockview(fs, b, zero) .= 0
    end
    return fs
end

negzeros(forest, fs) = count(x -> iszero(x) && signbit(x), gathered(forest, fs.work))

function quad_sets(forest)
    fs = FieldSet(forest, 3; G=2, rotation=(1, -3, 2))
    a = FieldSet(forest, 2; G=2, centering=facecentered(2, 1), rotation=(-2, -1))
    b = FieldSet(forest, 2; G=2, centering=facecentered(2, 2), rotation=(2, 1))
    return fs, RotationPair(a, b)
end

function quad_fill!(forest, fs, pair)
    sched = GhostSchedule(fs, OPS4)
    pscheds = (GhostSchedule(pair.a, OPS4), GhostSchedule(pair.b, OPS4))
    fill_ghosts!(fs, sched; boundary=QUAD_HOOK)
    fill_ghosts!(pair, pscheds; boundary=QUAD_HOOK)
    return sched, pscheds
end

quad_digests(forest, fs, pair) =
    (digest(gathered(forest, fs.work)), digest(gathered(forest, pair.a.work)),
     digest(gathered(forest, pair.b.work)))

function quad_states(forest, sets...)
    return map(sets) do fs
        u = statevector(fs)
        gather!(u, fs)
        digest(gathered(forest, u))
    end
end

# A global list of points over `[-2.9, 2.9]^2`, three quarters of them
# beyond the seam, of which rank `r` queries its slice as in
# `interpolation_points`. (Not `φ` and `φ²` as there: their fractional
# parts sum to one, which would put every point on one diagonal.)
seam_points_all(n) = [(-2.9 + 5.8 * mod(j * (sqrt(5.0) - 1) / 2, 1.0),
                       -2.9 + 5.8 * mod(j * (sqrt(2.0) - 1), 1.0)) for j in 1:n]

function seam_points(n)
    xs = seam_points_all(n)
    weights = [(2, 0, 5, 1, 3)[r % 5 + 1] for r in 0:(NRANKS - 1)]
    cuts = [0; round.(Int, n .* cumsum(weights) ./ sum(weights))]
    return xs[(cuts[RANK + 1] + 1):cuts[RANK + 2]]
end

function rotating_case(tag)
    forest = forest_of((3, 3), 8; rotating=(1, 2), centre=(0.4, 0.5))
    emit(tag, "leaves", nleaves(forest), maxlevel(forest), digest(string(forest.leaves)))
    fs, pair = quad_sets(forest)
    quad_data!(fs, 3)
    quad_data!(pair.a, 2)
    quad_data!(pair.b, 2)
    sched, pscheds = quad_fill!(forest, fs, pair)
    emit(tag, "transfers", transfers(forest, sched), transfers(forest, pscheds[1]),
         transfers(forest, pscheds[2]))
    emit(tag, "filled", quad_digests(forest, fs, pair)...)
    emit(tag, "negzero", negzeros(forest, fs), negzeros(forest, pair.a),
         negzeros(forest, pair.b))
    emit("#", tag, "rotated received", rotated_received(forest, sched, pscheds...))

    # Refine along the low face of the first dimension, which conformity
    # carries to the low face of the second, and coarsen the finest
    # blocks off the seam.
    flags = map(1:nblocks(fs)) do b
        k = blockkey(fs, b)
        ext = block_extent(forest, k)
        onseam = ext[1][1] == 0 || ext[2][1] == 0
        level(k) < 2 && ext[1][1] == 0 && ext[2][1] < 2 ? Refine :
        level(k) == 2 && !onseam ? Coarsen : Keep
    end
    regridded("$tag.regrid", forest, (fs => sched, pair => pscheds); flags=flags,
              boundary=QUAD_HOOK)
    sched, pscheds = quad_fill!(forest, fs, pair)
    emit(tag, "refilled", quad_digests(forest, fs, pair)...)

    xs = seam_points(257)
    r = interpolate(fs, xs, Lagrange(4); derivs=((0, 0), (1, 0), (0, 1)), vars=[2, 3, 1])
    emit(tag, "interpolated", length(gathered(forest, r.excluded)),
         count(x -> x[1] < 0 || x[2] < 0, seam_points_all(257)),
         digest(gathered(forest, r.values)))

    # The checkpoint: the bare form, after the fill; loaded at this rank
    # count, the pair rebuilt from its two sets and filled again.
    dir = checkpoint_dir()
    path = joinpath(dir, "QC-n$NRANKS.h5")
    emit("QC", "saved", quad_states(forest, fs, pair.a, pair.b)...)
    save_checkpoint(path, forest; fieldsets=("u" => fs, "a" => pair.a, "b" => pair.b),
                    application="Quadrant" => 1, io=:all)
    loaded = quad_load(path, COMM)
    emit("QC", "loaded", quad_states(loaded...)...)
    lforest, lfs, lpair = loaded[1], loaded[2], RotationPair(loaded[3], loaded[4])
    quad_fill!(lforest, lfs, lpair)
    emit("QC", "refilled", quad_digests(lforest, lfs, lpair)...)
    RANK == 0 && touch(joinpath(dir, "QC-n$NRANKS.done"))
    rotating_cross("QC")
    return nothing
end

function quad_load(path, comm)
    ck = load_checkpoint(path; comm=comm)
    sets = ck.fieldsets
    return (ck.forest, sets["u"].fieldset, sets["a"].fieldset, sets["b"].fieldset)
end

# The rotating checkpoints of the runs at other rank counts, loaded at
# this one and, under MPI, at one rank over `MPI.COMM_SELF`: their states
# must be the saved ones. `#` lines, as in `checkpoint_cross`.
function rotating_cross(tag)
    dir = checkpoint_dir()
    counts = checkpoint_sources(dir, tag)
    names = filter(readdir(dir)) do name
        m = match(r"^(.*)-n(\d+)\.h5$", name)
        m === nothing && return false
        n = parse(Int, m[2])
        m[1] == tag && n != NRANKS && (counts === nothing || n in counts)
    end
    comms = USE_MPI ? ((NRANKS, COMM), (1, MPI.COMM_SELF)) : ((1, nothing),)
    for name in sort(names), (n, comm) in comms
        loaded = quad_load(joinpath(dir, name), comm)
        emit("#", replace(name, ".h5" => ""), "at", n, "loaded",
             quad_states(loaded...)...)
    end
    return nothing
end

# The quadrant wave, as `wave_case` runs it, on a single leaf at the axis
# — its own neighbor across the seam three times over — so that at two
# and three ranks only one rank has a block: the fill, the wave, a pair's
# fill, interpolation beyond the seam from rank 0 alone, regrids that
# give every rank blocks and take them away again, and a checkpoint.
function rotating_empty_case(tag)
    forest = Forest((1, 1); N=8, rotating=(1, 2), comm=COMM)
    empties() = count(iszero, TreeAMR.allgather(forest.comm, length(blockrange(forest))))
    emit("#", tag, "empty ranks", empties())
    fs = FieldSet(forest, 2; G=2, centering=vertexcentered(2), rotation=(1, 2))
    fill_by_coordinates!(QUAD_DATA, fs)
    sched = GhostSchedule(fs, OPS4)
    fill_ghosts!(fs, sched; boundary=QUAD_HOOK)
    emit(tag, "filled", digest(gathered(forest, fs.work)))
    u = statevector(fs)
    gather!(u, fs)
    rk4!(u, fs, sched, 0.2 * minimum_spacing(forest), 3, Val(2), Val(fs.G), QUAD_HOOK)
    emit(tag, "state", digest(gathered(forest, u)))
    scatter!(fs, u)
    reductions(tag, fs, u)
    _, pair = quad_sets(forest)
    fill_by_coordinates!(QUAD_DATA, pair.a)
    fill_by_coordinates!(QUAD_DATA, pair.b)
    pscheds = (GhostSchedule(pair.a, OPS4), GhostSchedule(pair.b, OPS4))
    fill_ghosts!(pair, pscheds; boundary=QUAD_HOOK)
    emit(tag, "pair", digest(gathered(forest, pair.a.work)),
         digest(gathered(forest, pair.b.work)))
    fill_ghosts!(fs, sched; boundary=QUAD_HOOK)
    pts = [(-0.95 + 1.9 * mod(j * 0.618034, 1.0), -0.95 + 1.9 * mod(j * 0.381966, 1.0))
           for j in 1:19]
    r = interpolate(fs, RANK == 0 ? pts : similar(pts, 0), Lagrange(4);
                    derivs=((0, 0), (0, 1)))
    emit(tag, "interpolated", digest(gathered(forest, r.values)))
    regridded("$tag.refine", forest, (fs => sched, pair => pscheds);
              flags=fill(Refine, length(blockrange(forest))), boundary=QUAD_HOOK)
    emit("#", tag, "empty ranks after refining", empties())
    sched = GhostSchedule(fs, OPS4)
    pscheds = (GhostSchedule(pair.a, OPS4), GhostSchedule(pair.b, OPS4))
    regridded("$tag.coarsen", forest, (fs => sched, pair => pscheds);
              flags=fill(Coarsen, length(blockrange(forest))), boundary=QUAD_HOOK)
    emit("#", tag, "empty ranks after coarsening", empties())
    path = joinpath(checkpoint_dir(), "$tag-n$NRANKS.h5")
    save_checkpoint(path, forest; fieldsets=("u" => fs, "a" => pair.a, "b" => pair.b),
                    application="QuadrantEmpty" => 1)
    loaded = quad_load(path, COMM)
    emit(tag, "checkpoint", nleaves(loaded[1]),
         quad_states(forest, fs, pair.a, pair.b) == quad_states(loaded...))
    return nothing
end

# --- the refusals, which only a distributed run can show -------------------

# A forest mutated on one rank only, a layout that differs on one rank,
# and an argument one rank alone refuses: each must be refused on every
# rank, with the same reason on those that did not refuse themselves.
function refusals()
    NRANKS > 1 || return nothing
    world = TreeAMR.communicator(COMM)
    function attempt(f)
        msg = try
            f()
            ""
        catch err
            err isa Union{ArgumentError,DimensionMismatch} || rethrow()
            err.msg
        end
        refused = TreeAMR.allgather(world, !isempty(msg))
        return count(refused), msg
    end
    ops4 = Operators(prolongation=4, restriction=4)

    forest = Forest((4, 4); N=8, comm=COMM)
    if RANK == 1
        refine!(forest, [forest.leaves[1]])
        balance!(forest)
    end
    n, msg = attempt(() -> GhostSchedule(FieldSet(forest, 1; G=2), ops4))
    emit("# diverged refused on", n, "of", NRANKS, "ranks:", msg)

    forest = Forest((4, 4); N=8, comm=COMM)
    ops = RANK == 1 ? Operators(prolongation=2, restriction=2) : ops4
    n, msg = attempt(() -> GhostSchedule(FieldSet(forest, 1; G=2), ops))
    emit("# layout refused on", n, "of", NRANKS, "ranks:", msg)

    # G = 1 is too narrow for order 4, which only rank 1 asks for.
    n, msg = attempt(() -> GhostSchedule(FieldSet(forest, 1; G=1),
                                         RANK == 1 ? ops4 :
                                         Operators(prolongation=2, restriction=2)))
    emit("# partial refused on", n, "of", NRANKS, "ranks:", msg)

    # A diverged forest is refused by the interface schedule too.
    forest = Forest((4, 4); N=8, comm=COMM)
    RANK == 0 && (refine!(forest, [forest.leaves[end]]); balance!(forest))
    n, msg = attempt(() -> InterfaceSchedule(FieldSet(forest, 1; G=0,
                                                      centering=facecentered(2, 1))))
    emit("# interface diverged refused on", n, "of", NRANKS, "ranks:", msg)

    # regrid!'s argument checks are agreed the same way (step 4): a flag
    # box out of range on rank 1 only, a flag vector of the wrong length
    # on rank 0 only (a `DimensionMismatch` there, as serially), another
    # `buffer` on rank 1, and a forest refined on rank 1 only, with a
    # field set that rank 1's own checks accept.
    forest = Forest((4, 4); N=8, comm=COMM)
    fs = FieldSet(forest, 1; G=2)
    sched = GhostSchedule(fs, ops4)
    nb = length(blockrange(forest))
    flags = Any[RANK == 1 && b == 1 ? (Refine, (1:9, 1:8)) : Keep for b in 1:nb]
    n, msg = attempt(() -> regrid!(forest, fs => sched; flags=flags))
    emit("# regrid box refused on", n, "of", NRANKS, "ranks:", msg)
    n, msg = attempt(() -> regrid!(forest, fs => sched; flags=fill(Keep, nb + (RANK == 0))))
    emit("# regrid length refused on", n, "of", NRANKS, "ranks:", msg)
    n, msg = attempt(() -> regrid!(forest, fs => sched; flags=fill(Refine, nb),
                                   buffer=RANK == 1 ? 1 : 0))
    emit("# regrid buffer refused on", n, "of", NRANKS, "ranks:", msg)
    emit("# regrid refusals left the forest alone", all(TreeAMR.allgather(world,
                                                                          nleaves(forest) == 16)))
    if RANK == 1
        refine!(forest, [forest.leaves[end]])
        balance!(forest)
        fs = FieldSet(forest, 1; G=2)
    end
    n, msg = attempt(() -> regrid!(forest, fs => nothing;
                                   flags=fill(Keep, length(blockrange(forest)))))
    emit("# regrid diverged refused on", n, "of", NRANKS, "ranks:", msg)

    # A point outside the domain on rank 1 only (step 5): refused on every
    # rank before anything is routed, rank 0 naming rank 1's point.
    forest = Forest((4, 4); N=8, comm=COMM)
    fs = FieldSet(forest, 1; G=2)
    n, msg = attempt(() -> interpolate(fs, RANK == 1 ? [(0.5, 0.5), (0.5, 9.0)] :
                                           [(0.5, 0.5)], Lagrange(4)))
    emit("# interpolate outside refused on", n, "of", NRANKS, "ranks:", msg)

    # A rotating seam (M12): a point beyond it on rank 1 only, for a set
    # that turns into its partner, which `interpolate` does not read; and
    # such a set regridded alone on rank 1, where the others regrid the
    # pair. Each refused on every rank, rank 1 with its own reason.
    forest = Forest((2, 2); N=8, rotating=(1, 2), comm=COMM)
    _, pair = quad_sets(forest)
    n, msg = attempt(() -> interpolate(pair.a, RANK == 1 ? [(0.5, 0.5), (-0.5, 0.7)] :
                                               [(0.5, 0.5)], Lagrange(4)))
    emit("# rotating interpolate refused on", n, "of", NRANKS, "ranks:", msg)
    pscheds = (GhostSchedule(pair.a, ops4), GhostSchedule(pair.b, ops4))
    nb = length(blockrange(forest))
    n, msg = attempt(() -> regrid!(forest, RANK == 1 ?
                                           (pair.a => pscheds[1], pair.b => pscheds[2]) :
                                           (pair => pscheds,); flags=fill(Keep, nb)))
    emit("# rotating regrid refused on", n, "of", NRANKS, "ranks:", msg)
    return nothing
end

# The verbs the exchange does not use (`alltoallv` routes interpolation
# points from step 5 on), with empty contributions, and the one
# duplicate per communicator: every forest over `COMM` holds the same
# `MPICommunicator`.
function verbs()
    NRANKS > 1 || return nothing
    world = TreeAMR.communicator(COMM)
    shared = all(_ -> Forest((2, 2); N=4, comm=COMM).comm === world, 1:20)
    emit("# one duplicate per communicator", shared && TreeAMR.communicator(COMM) === world)
    # The device-aware setting (step 8) shares the duplicate, and a buffer
    # that is not a contiguous host vector is refused before anything is
    # sent, naming the setting; over a device-aware communicator a range
    # is no dense device vector either.
    aware = TreeAMR.communicator(COMM; deviceaware=true)
    refused(c) = try
        TreeAMR.isend(c, 1:3, (RANK + 1) % NRANKS, 99)
        false
    catch err
        err isa ArgumentError && occursin("deviceaware = true", sprint(showerror, err))
    end
    emit("# device-aware shares the duplicate",
         aware.comm === world.comm && aware.deviceaware && !world.deviceaware &&
         Forest((2, 2); N=4, comm=aware).comm === aware &&
         !TreeAMR.hoststaging(aware, zeros(1)) && !TreeAMR.hoststaging(world, zeros(1)) &&
         refused(world) && !refused(aware) &&
         (try TreeAMR.isend(aware, 1:3, (RANK + 1) % NRANKS, 99); false
          catch err; err isa ArgumentError end))
    # Rank r sends s + 1 copies of 100r + s to rank s, and nothing to
    # itself when r is even.
    counts = [s == RANK && iseven(RANK) ? 0 : s + 1 for s in 0:(NRANKS - 1)]
    send = reduce(vcat, [fill(100RANK + s, counts[s + 1]) for s in 0:(NRANKS - 1)])
    recv, rcounts = TreeAMR.alltoallv(world, send, counts)
    expect = [r == RANK && iseven(RANK) ? 0 : RANK + 1 for r in 0:(NRANKS - 1)]
    good = rcounts == expect &&
           recv == reduce(vcat, [fill(100r + RANK, expect[r + 1]) for r in 0:(NRANKS - 1)])
    # `allgatherv` with an empty contribution from rank 0.
    v = TreeAMR.allgatherv(world, collect(1.0:RANK))
    good &= v == reduce(vcat, [collect(1.0:r) for r in 0:(NRANKS - 1)])
    emit("# verbs agree on every rank", all(TreeAMR.allgather(world, good)))
    return nothing
end

# --- checkpoints (step 6 of M7) --------------------------------------------

# Where the checkpoints go: `TREEAMR_CHECKPOINT_DIR`, which `mpi_tests.jl`
# sets to one directory for the serial run and every `mpiexec` run, so
# that each run can load what the runs before it wrote, at another rank
# count; otherwise a fresh directory, rank 0's, named to every rank.
function checkpoint_dir()
    dir = get(ENV, "TREEAMR_CHECKPOINT_DIR", "")
    isempty(dir) || return dir
    USE_MPI || return mktempdir()
    name = RANK == 0 ? collect(codeunits(mktempdir())) : UInt8[]
    return String(TreeAMR.allgatherv(TreeAMR.communicator(COMM), name))
end

# The tracked pulse as a chunked driver, as in `checkpoint_tests.jl`: a
# chunk is a few RK4 steps, the flags and a regrid, and the checkpoint
# goes at a chunk boundary, after the regrid, where the run holds nothing
# but the mesh, the owned points and `(t, chunk)`.
const CK_PULSE = moving_pulse((1.3, 1.4), 0.35)
const CK_HOOK = boundary_by_coordinates(CK_PULSE)

function ck_start(comm)
    forest = Forest((4, 3); N=8, comm=comm)
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(2))
    fill_by_coordinates!(CK_PULSE, fs)
    return (; forest, fs, t=0.0, chunk=0)
end

function ck_chunk(run; steps=3)
    (; forest, fs, t, chunk) = run
    sched = GhostSchedule(fs, OPS4)
    u = statevector(fs)
    gather!(u, fs)
    dt = 0.2 * minimum_spacing(forest)
    rk4!(u, fs, sched, dt, steps, Val(2), Val(fs.G), CK_HOOK)
    scatter!(fs, u)
    fill_ghosts!(fs, sched; boundary=CK_HOOK)
    regrid!(forest, fs => sched; flags=pulse_flags(fs, 2), buffer=2, boundary=CK_HOOK)
    return (; forest, fs, t=t + steps * dt, chunk=chunk + 1)
end

function ck_digests(run)
    u = statevector(run.fs)
    gather!(u, run.fs)
    return (digest(string(run.forest.leaves)), digest(gathered(run.forest, u)))
end

# The bare form, `name => fs`: `regrid!` has just filled the working
# array's owned points. The plain data carry the run state and an array
# of strings.
function ck_save(path, run; filters=(), io=:node)
    return save_checkpoint(path, run.forest; fieldsets=("pulse" => run.fs,),
                           application="PulseRestart" => 1, filters=filters, io=io,
                           data=(; t=run.t, chunk=run.chunk, σ=7 // 20,
                                 tags=["pulse", "", "vertex-centered"]))
end

function ck_restore(path, comm)
    ck = load_checkpoint(path; comm=comm)
    ck.data.tags == ["pulse", "", "vertex-centered"] && ck.data.σ === 7 // 20 ||
        error("the plain data did not come back exactly: $(ck.data)")
    return (; forest=ck.forest, fs=ck.fieldsets["pulse"].fieldset, t=ck.data.t,
            chunk=ck.data.chunk), ck.provenance
end

function ck_finish(run; chunks)
    while run.chunk < chunks
        run = ck_chunk(run)
    end
    return run
end

# One leaf per rank or fewer: a 1D cell-centered state on two leaves, so
# that at three ranks one rank holds no blocks when the file is written
# and when it is read.
function ck_small(comm)
    forest = Forest((2,); N=8, periodic=(true,), comm=comm)
    fs = FieldSet(forest, 1; G=2)
    fill_by_coordinates!((x, v) -> sin(π * x[1]) + x[1]^2 / 7, fs)
    return forest, fs
end

# The mesh refines in the first chunk and again in the second, so a
# checkpoint after the first is followed by a regrid that changes the
# mesh: a regrid that depended on something the file does not hold would
# show.
function checkpoint_case(tag; chunks=3, k=1)
    dir = checkpoint_dir()
    run = ck_finish(ck_start(COMM); chunks=chunks)
    emit(tag, "uninterrupted", ck_digests(run)...)
    final = nleaves(run.forest)
    # Saved after chunk `k`, everything dropped, and continued from each
    # file at this rank count: unfiltered with a part per rank, filtered
    # with two I/O groups (at three ranks, one of two ranks and one of
    # one), and with the default, a part per node, which on one node is
    # the one part inside the index.
    run = ck_finish(ck_start(COMM); chunks=k)
    emit(tag, "mesh", nleaves(run.forest), "then", final)
    emit(tag, "saved", ck_digests(run)...)
    plain = joinpath(dir, "$tag-n$NRANKS.h5")
    filtered = joinpath(dir, "$tag-n$NRANKS-filtered.h5")
    node = joinpath(dir, "$tag-n$NRANKS-node.h5")
    ck_save(plain, run; io=:all)
    ck_save(filtered, run; filters=(Shuffle(), Deflate(1)), io=2)
    ck_save(node, run)
    run = nothing
    for (name, path) in (("restarted", plain), ("restarted-filtered", filtered),
                         ("restarted-node", node))
        resumed, provenance = ck_restore(path, COMM)
        emit(tag, name, "loaded", ck_digests(resumed)...)
        emit(tag, name, "continued", ck_digests(ck_finish(resumed; chunks=chunks))...)
        emit("#", tag, name, "nranks", provenance.nranks, "nparts", provenance.nparts)
    end
    # The version-1 fixtures (`checkpoint_tests.jl`), loaded at this rank
    # count: their digests must be the serial run's.
    for name in ("plain", "filtered")
        ck = load_checkpoint(joinpath(@__DIR__, "fixtures", "checkpoint-v1-$name.h5");
                             comm=COMM, types=(Float32x2,))
        emit("V1", name, "loaded", digest(string(ck.forest.leaves)),
             digest(gathered(ck.forest, ck.fieldsets["u"].state)),
             digest(reinterpret(UInt8, gathered(ck.forest, ck.fieldsets["w"].state))),
             digest(string(ck.data)))
    end
    forest, fs = ck_small(COMM)
    u = statevector(fs)
    gather!(u, fs)
    emit(tag * "1", "saved", digest(gathered(forest, u)))
    save_checkpoint(joinpath(dir, "$(tag)1-n$NRANKS.h5"), forest;
                    fieldsets=("u" => (fs, u),), application="Small" => 1)
    ck = load_checkpoint(joinpath(dir, "$(tag)1-n$NRANKS.h5"); comm=COMM)
    emit(tag * "1", "loaded", digest(gathered(ck.forest, ck.fieldsets["u"].state)))
    emit("#", tag * "1", "empty ranks", count(TreeAMR.allgather(forest.comm,
                                                               nblocks(fs) == 0)))
    # Every file of this run is in place (each save returns on every rank
    # after the rename), which a run waiting to load them is told by an
    # empty marker file.
    RANK == 0 && touch(joinpath(dir, "$tag-n$NRANKS.done"))
    checkpoint_cross(tag; chunks=chunks)
    return nothing
end

# The rank counts whose files `checkpoint_cross` loads: those that
# `TREEAMR_CHECKPOINT_FROM` lists, once each run's marker is there, every
# rank waiting on its own so that none spins in a collective meanwhile;
# without the variable, every other rank count whose files are there.
function checkpoint_sources(dir, tag)
    from = get(ENV, "TREEAMR_CHECKPOINT_FROM", nothing)
    from === nothing && return nothing
    counts = parse.(Int, split(from))
    deadline = time() + parse(Float64, get(ENV, "TREEAMR_CHECKPOINT_WAIT", "900"))
    for n in counts
        marker = joinpath(dir, "$tag-n$n.done")
        while !isfile(marker)
            time() < deadline ||
                error("the checkpoints of the run at $n rank(s) did not appear: no $marker")
            sleep(0.2)
        end
    end
    return counts
end

# The files the other runs wrote, at other rank counts, loaded at this
# one and continued: a `#` line each, since which files exist depends on
# the order of the runs. Under MPI each is loaded a second time over
# `MPI.COMM_SELF`, every rank on its own, which is a load at one rank of
# MPI. Collective.
function checkpoint_cross(tag; chunks=3)
    dir = checkpoint_dir()
    counts = checkpoint_sources(dir, tag)
    names = filter(readdir(dir)) do name
        m = match(r"^(.*)-n(\d+)(-filtered|-node)?\.h5$", name)
        m === nothing && return false
        n = parse(Int, m[2])
        m[1] in (tag, tag * "1") && n != NRANKS && (counts === nothing || n in counts)
    end
    comms = USE_MPI ? ((NRANKS, COMM), (1, MPI.COMM_SELF)) : ((1, nothing),)
    for name in sort(names), (n, comm) in comms
        path = joinpath(dir, name)
        what = replace(name, ".h5" => "")
        if startswith(name, tag * "1")
            ck = load_checkpoint(path; comm=comm)
            emit("#", what, "at", n, "loaded",
                 digest(gathered(ck.forest, ck.fieldsets["u"].state)))
        else
            resumed, _ = ck_restore(path, comm)
            emit("#", what, "at", n, "loaded", ck_digests(resumed)...)
            emit("#", what, "at", n, "continued",
                 ck_digests(ck_finish(resumed; chunks=chunks))...)
        end
    end
    return nothing
end

# The files of the directory that belong to the checkpoint `path`: the
# index and every file named after it.
ck_files(path) = sort(filter(n -> n == basename(path) || startswith(n, basename(path) * "."),
                             readdir(dirname(path))))

# Whether no file was opened by more than one process during `f()`: every
# rank records the files it opens (a test hook of the HDF5 extension),
# and the records are gathered.
function opened_once(f)
    CKPT.OPEN_LOG[] = String[]
    try
        f()
    finally
        mine = unique(CKPT.OPEN_LOG[])
        CKPT.OPEN_LOG[] = nothing
        world = TreeAMR.communicator(COMM)
        all_ = split(String(TreeAMR.allgatherv(world, collect(codeunits(join(mine, "\n") *
                                                                       "\n")))), '\n';
                     keepempty=false)
        return allunique(all_) && !isempty(all_)
    end
end

# A refusal on some ranks, or arguments that differ between them, must be
# refused on every rank before the file is created, and leave the
# previous checkpoint at the path as it was.
function checkpoint_refusals()
    NRANKS > 1 || return nothing
    world = TreeAMR.communicator(COMM)
    function attempt(f)
        msg = try
            f()
            ""
        catch err
            err isa ArgumentError || rethrow()
            err.msg
        end
        return count(TreeAMR.allgather(world, !isempty(msg))), msg
    end
    dir = checkpoint_dir()
    path = joinpath(dir, "refused-n$NRANKS.h5")
    forest = Forest((4, 4); N=8, comm=COMM)
    fs = FieldSet(forest, 1; G=1)
    fill_by_coordinates!((x, v) -> x[1] - x[2], fs)
    save(; kwargs...) = save_checkpoint(path, forest; fieldsets=("u" => fs,),
                                        application="Refused" => 1, io=1, kwargs...)
    save(; data=(; t=1.0))
    n, msg = attempt(() -> save(; data=(; t=1.0, rank=RANK)))
    emit("# checkpoint data refused on", n, "of", NRANKS, "ranks:", msg)
    n, msg = attempt(() -> save_checkpoint(path, forest; fieldsets=("u" => fs,),
                                           application=(RANK == 1 ? "TreeAMR.jl" :
                                                        "Refused") => 1))
    emit("# checkpoint partial refused on", n, "of", NRANKS, "ranks:", msg)
    n, msg = attempt(() -> save(; filters=RANK == 1 ? (Deflate(1),) : ()))
    emit("# checkpoint layout refused on", n, "of", NRANKS, "ranks:", msg)
    # In the do-block, after the file is created: refused on every rank
    # before the item is written, the file closed and the partial removed.
    n, msg = attempt(() -> save_checkpoint(path, forest; fieldsets=("u" => fs,),
                                           application="Refused" => 1) do app
                         write_plain(app, "mine", RANK)
                     end)
    emit("# checkpoint write_plain refused on", n, "of", NRANKS, "ranks:", msg)
    intact = load_checkpoint(path; comm=COMM).data == (; t=1.0) &&
             !ispath(path * ".partial")
    emit("# checkpoint refusals left the file alone",
         all(TreeAMR.allgather(world, intact)))
    n, msg = attempt(() -> load_checkpoint(path; comm=COMM,
                                           fieldsets=RANK == 1 ? ("v",) : ("u",)))
    emit("# load layout refused on", n, "of", NRANKS, "ranks:", msg)
    n, msg = attempt(() -> load_checkpoint(joinpath(dir, "missing.h5"); comm=COMM))
    emit("# load missing refused on", n, "of", NRANKS, "ranks:", msg)
    # Damage only the last part's reader can see — one value of the last
    # block, changed by rank 0 alone in the last part file between the save
    # and the load — is refused by the checksums on every rank, which is
    # the agreement: the other parts are intact. This is the multi-node
    # corruption of step 6 made by hand; no file is shared any more, so
    # none can happen on its own.
    damaged = joinpath(dir, "damaged-n$NRANKS.h5")
    save_checkpoint(damaged, forest; fieldsets=("u" => fs,), application="Refused" => 1,
                    io=:all)
    world_barrier() = TreeAMR.allgather(world, true)
    parts = filter(n -> n != basename(damaged), ck_files(damaged))
    if RANK == 0
        lastpart = last(sort(parts; by=n -> parse(Int, split(n, '.')[end - 1])))
        HDF5.h5open(joinpath(dir, lastpart), "r+") do file
            data = file["TreeAMR.jl/fieldsets/u/data"]
            d = read(data)
            d[2, 3, 1, end] += 1
            data[:, :, :, :] = d
        end
    end
    world_barrier()
    n, msg = attempt(() -> load_checkpoint(damaged; comm=COMM))
    emit("# checkpoint damage refused on", n, "of", NRANKS, "ranks:", msg)
    # A part of another save in place of one of this save's, and a part
    # missing: both refused on every rank, before any data move.
    other = joinpath(dir, "other-n$NRANKS.h5")
    save_checkpoint(other, forest; fieldsets=("u" => fs,), application="Refused" => 1,
                    io=:all)
    save_checkpoint(damaged, forest; fieldsets=("u" => fs,), application="Refused" => 1,
                    io=:all)
    partof(path, j) = only(filter(n -> endswith(n, ".$j.h5"), ck_files(path)))
    if RANK == 0
        cp(joinpath(dir, partof(other, 1)), joinpath(dir, partof(damaged, 1)); force=true)
    end
    world_barrier()
    n, msg = attempt(() -> load_checkpoint(damaged; comm=COMM))
    emit("# checkpoint foreign part refused on", n, "of", NRANKS, "ranks:", msg)
    RANK == 0 && rm(joinpath(dir, partof(other, NRANKS - 1)))
    world_barrier()
    n, msg = attempt(() -> load_checkpoint(other; comm=COMM))
    emit("# checkpoint missing part refused on", n, "of", NRANKS, "ranks:", msg)
    # Orphans: a file of the part form for the index, from a save no index
    # names, is removed by the next save, and so are the previous save's
    # parts; files of other forms are left alone.
    orphans = joinpath(dir, "orphans-n$NRANKS.h5")
    keep = [basename(orphans) * ".notapart.h5", basename(orphans) * "." * "a"^31 * ".0.h5",
            basename(orphans) * "x." * "a"^32 * ".0.h5"]
    if RANK == 0
        touch(joinpath(dir, basename(orphans) * "." * "b"^32 * ".7.h5"))
        foreach(name -> touch(joinpath(dir, name)), keep)
    end
    world_barrier()
    save_checkpoint(orphans, forest; fieldsets=("u" => fs,), application="Refused" => 1,
                    io=:all)
    # Rank 0 removes the stale files after the commit, when the other ranks
    # have returned: they wait for it here before they look.
    world_barrier()
    first_parts = ck_files(orphans)
    save_checkpoint(orphans, forest; fieldsets=("u" => fs,), application="Refused" => 1,
                    io=:all)
    world_barrier()
    after = ck_files(orphans)
    # The index, its parts, and the two of `keep` named after the index.
    cleaned = length(after) == NRANKS + 3 &&
              !(basename(orphans) * "." * "b"^32 * ".7.h5" in after) &&
              isempty(intersect(setdiff(first_parts, [basename(orphans)], keep), after)) &&
              all(name -> isfile(joinpath(dir, name)), keep)
    emit("# checkpoint orphans removed", all(TreeAMR.allgather(world, cleaned)))
    # An I/O process that fails after its first write (a test hook): the
    # save is refused on every rank, the previous checkpoint still loads,
    # and none of the new parts, nor the partial index, is left.
    world_barrier()
    before = ck_files(orphans)
    CKPT.FAIL_PART[] = NRANKS - 1
    n, msg = try
        attempt(() -> save_checkpoint(orphans, forest; fieldsets=("u" => fs,),
                                      application="Refused" => 1, io=:all,
                                      data=(; t=2.0)))
    catch err
        # Not an ArgumentError: an I/O error, which `attempt` rethrows.
        count(TreeAMR.allgather(world, true)), sprint(showerror, err)
    finally
        CKPT.FAIL_PART[] = -1
    end
    emit("# checkpoint failed part on", n, "of", NRANKS, "ranks:", first(split(msg, '\n')))
    intact = ck_files(orphans) == before &&
             load_checkpoint(orphans; comm=COMM).data == (;)
    emit("# checkpoint failed part left the previous one", all(TreeAMR.allgather(world,
                                                                                  intact)))
    # No file opened by more than one process, saving with two I/O groups
    # and loading.
    once = joinpath(dir, "once-n$NRANKS.h5")
    saved = opened_once(() -> save_checkpoint(once, forest; fieldsets=("u" => fs,),
                                              application="Refused" => 1, io=2))
    loaded = opened_once(() -> load_checkpoint(once; comm=COMM))
    emit("# checkpoint each file opened by one process", saved, loaded)
    return nothing
end

# --- the cases --------------------------------------------------------------

const OPS4 = Operators(prolongation=4, restriction=4)
const OPS2 = Operators(prolongation=2, restriction=2)
const OPSC = Operators(family=Conservative, prolongation=3, restriction=2)

function main()
    emit("# ranks", NRANKS)
    # D = 1: two leaves on a periodic line, so a third rank has no blocks;
    # then three levels against outer faces, which brings in the hook.
    wave_case("W1p", forest_of((2,), 8; periodic=(true,)); G=2,
              centering=cellcentered(1), ops=OPS4, steps=8)
    wave_case("W1o", forest_of((4,), 8; centre=(1.3,)); G=2, centering=cellcentered(1),
              ops=OPS4, steps=8)
    # D = 2, vertex-centered, outer faces, three levels; with the
    # negative control.
    wave_case("W2v", forest_of((4, 4), 8; centre=(1.6, 2.3)); G=1,
              centering=vertexcentered(2), ops=OPS4, steps=6, control=true)
    # D = 3, cell-centered, periodic, three levels.
    wave_case("W3c", forest_of((2, 2, 2), 4; periodic=(true, true, true),
                               centre=(0.7, 0.8, 1.2)); G=1,
              centering=cellcentered(3), ops=OPS2, steps=3)
    # A reflecting box: walls at both ends of every dimension, vertex-
    # centered so that the derived wall plane occurs, and the low walls
    # only, cell-centered, with outer faces opposite. The first variable
    # is odd across the x₁ walls.
    parity = [(OddParity, EvenParity), (EvenParity, EvenParity)]
    wave_case("R2v", forest_of((3, 3), 8; reflecting=((true, true), (true, true)),
                               centre=(0.4, 0.5)); G=1,
              centering=vertexcentered(2), ops=OPS4, steps=6, parity=parity)
    wave_case("R2c", forest_of((3, 3), 8; reflecting=((true, false), (true, false)),
                               centre=(0.4, 0.5)); G=2,
              centering=cellcentered(2), ops=OPS4, steps=6, parity=parity)
    # Burgers' equation with the fixup at coarse-fine faces.
    burgers_case("B2", forest_of((4, 4), 8; periodic=(true, true), centre=(1.6, 2.3));
                 G=2, ops=OPSC, steps=4)
    # Every centering through one fill.
    fill_case("F2", forest_of((3, 3), 8; periodic=(true, false),
                              reflecting=((false, false), (true, true)),
                              centre=(1.4, 0.5)),
              ("cell" => cellcentered(2), "vertex" => vertexcentered(2),
               "face1" => facecentered(2, 1), "face2" => facecentered(2, 2));
              G=2, ops=OPS4)
    fill_case("F3", forest_of((2, 2, 2), 4; periodic=(true, false, false),
                              reflecting=((false, false), (true, false), (false, false)),
                              centre=(0.7, 0.4, 1.2)),
              ("vertex" => vertexcentered(3), "face3" => facecentered(3, 3),
               "edge1" => edgecentered(3, 1));
              G=1, ops=OPS2)
    # Regridding: the tracked pulse, Burgers' shock with mass conserved,
    # the initial-data cycle from one leaf, and blocks moving between
    # ranks over several field sets, in Float64 and Float32x2. Float32 is
    # not repeated here (dropped in step 9): it crosses MPI as a native
    # type, its regrid stage is checked bitwise in process by
    # `regrid_exchange_tests.jl`, and Float32x2 is the element type MPI
    # sends through a derived datatype, so it is the one that adds a code
    # path.
    tracked_pulse_case("TP"; cycles=3, steps=4)
    burgers_regrid_case("BR"; cycles=3, steps=3)
    adapt_case("A2")
    moving_blocks_case("M", Float64)
    moving_blocks_case("M32x2-", Float32x2; full=false)
    # Point interpolation, routed to the owners and back.
    interpolate_case("I2")
    # A rank without blocks through everything, at three ranks.
    empty_rank_case("E2")
    # A rotating quadrant (M12): the vertex-centered wave in 2D, and in 3D
    # cell-centered over a reflecting low face below the plane, an octant;
    # a set that turns into itself and a pair through a fill, a regrid,
    # interpolation and a checkpoint; and a single leaf at the axis, which
    # leaves every rank but one without blocks.
    wave_case("Q2v", forest_of((3, 3), 8; rotating=(1, 2), centre=(0.4, 0.5)); G=1,
              centering=vertexcentered(2), ops=OPS4, steps=6, rotation=(1, 2))
    wave_case("Q3c", forest_of((2, 2, 2), 4; rotating=(1, 2),
                               reflecting=((false, false), (false, false), (true, false)),
                               centre=(0.5, 0.6, 0.4)); G=1,
              centering=cellcentered(3), ops=OPS2, steps=3, rotation=(1, 2),
              parity=[ntuple(_ -> EvenParity, 3), ntuple(_ -> EvenParity, 3)])
    rotating_case("Q2")
    rotating_empty_case("QE2")
    # Checkpoints: saved and loaded at this rank count, and the files of
    # the runs before this one loaded at it.
    checkpoint_case("C")
    refusals()
    checkpoint_refusals()
    verbs()
    return nothing
end

try
    main()
catch err
    # A rank that fails must not leave the others waiting in a
    # collective: take the whole job down.
    showerror(stderr, err, catch_backtrace())
    println(stderr)
    USE_MPI && MPI.Abort(MPI.COMM_WORLD, 1)
    rethrow()
end
