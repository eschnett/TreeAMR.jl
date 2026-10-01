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
# rank starts in seconds. No regrid: that is step 4 of M7.

using TreeAMR
using MPI: MPI
using KernelAbstractions: @kernel, @index, @Const
using Printf: @sprintf
using SHA: sha256

const USE_MPI = "mpi" in ARGS
USE_MPI && MPI.Init()
const COMM = USE_MPI ? MPI.COMM_WORLD : nothing
const RANK = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const NRANKS = USE_MPI ? MPI.Comm_size(MPI.COMM_WORLD) : 1
const OUT = isdefined(@__MODULE__, :WORKLOAD_IO) ? WORKLOAD_IO : stdout

emit(words...) = (RANK == 0 && println(OUT, join(words, " ")); nothing)

digest(bytes::AbstractVector{UInt8}) = bytes2hex(sha256(bytes))[1:32]
digest(v::Vector{Float64}) = digest(reinterpret(UInt8, v))
digest(s::AbstractString) = digest(codeunits(s))

# Every rank's part of a per-block vector, in rank order, which is block
# order: the ranks own contiguous runs of the curve.
gathered(forest, v) = TreeAMR.allgatherv(forest.comm, collect(vec(v)))

# A forest refined twice around `centre`, so that three levels meet. A
# forest mutation is collective: every rank makes the same calls.
function forest_of(roots::NTuple{D,Int}, N; periodic=ntuple(_ -> false, D),
                   reflecting=ntuple(_ -> (false, false), D), centre=nothing,
                   levels=2) where {D}
    forest = Forest(roots; N=N, periodic=periodic, reflecting=reflecting, comm=COMM)
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
                   control=false) where {D}
    fs = FieldSet(forest, 2; G=G, centering=centering, parity=parity)
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
            err isa ArgumentError || rethrow()
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
    return nothing
end

# The verbs the exchange does not use yet (step 5 routes points with
# `alltoallv`), and the one duplicate per communicator: every forest
# over `COMM` holds the same `MPICommunicator`.
function verbs()
    NRANKS > 1 || return nothing
    world = TreeAMR.communicator(COMM)
    shared = all(_ -> Forest((2, 2); N=4, comm=COMM).comm === world, 1:20)
    emit("# one duplicate per communicator", shared && TreeAMR.communicator(COMM) === world)
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
    refusals()
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
