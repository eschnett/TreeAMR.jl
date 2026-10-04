# Checkpoint and restart (M9a).
#
# A checkpoint is worth something only if a run restarted from it is the
# run that was interrupted, bit for bit, and only if a file it cannot
# interpret is refused rather than misread. So the claims here are
# byte-for-byte equalities and refusals with reasons: a field set, its
# forest and the application's plain data come back exactly, in every
# centering, dimension, element type and face kind; a chunked driver
# with regrids continues from a checkpoint exactly as it would have run
# on; a newer format, an unknown feature, a type the reader cannot name
# and a value outside the plain-data vocabulary are each refused with a
# message saying why; a write that fails leaves the previous checkpoint
# as it was; and a version-1 file, written before the part files of M7
# step 6b, still loads (`test/fixtures/`). Serially a checkpoint is one
# file, its one part inside the index; the part files themselves are the
# MPI workload's (`mpi_workload.jl`).
#
# The drivers reuse the wave and Burgers applications of `wave.jl` and
# `burgers.jl`, and the round trip the face-kind oracles of
# `ghost_oracles.jl` (`faces_forest`, `parity_data`, `ghosts_for`), so
# `runtests.jl` includes this file after all three.

using HDF5
using MultiFloats: Float32x2

# A struct, which plain data refuses: the application converts it.
struct CheckpointParams
    cfl::Float64
end

# Open a copy of `path` for writing and let `edit!` change it, the way a
# newer writer, or damage, would have: this is how the refusals are made
# to happen without a second version of the package.
function edited_copy(edit!, path, name)
    copy = joinpath(dirname(path), name)
    cp(path, copy; force=true)
    h5open(edit!, copy, "r+")
    return copy
end

function replace_attribute!(obj, name, value)
    delete_attribute(obj, name)
    write_attribute(obj, name, value)
    return obj
end

bytes(a) = reinterpret(UInt8, vec(Array(a)))

# Where a serial checkpoint keeps a field set's data: in the one part,
# which lives inside the index (format version 2, M7 step 6b).
const PART = "TreeAMR.jl/parts/0000"

# --- the round trip --------------------------------------------------------

# The cases: every centering in every dimension, in the element types
# (geometry, field) and the face kinds, so that every type meets
# reflecting parity, periodic wrap and an outer hook. The last type pair
# is a field set narrower than its geometry, the one case where the two
# element types differ. In D = 1 there are only two centerings, so each
# runs in all four type pairs, against all four face kinds; in D = 2
# and 3 each centering runs once, the types and kinds rotating so that
# the pairings differ between the two. Every other case is filtered, so
# the chunked layout is read back as often as the contiguous one.
const ROUNDTRIP_TYPES = ((Float64, Float64), (Float32, Float32), (Float32x2, Float32x2),
                         (Float64, Float32))
const ROUNDTRIP_KINDS = Dict(
    1 => ((:periodic,), (:reflect_lo,), (:outer,), (:reflect_both,)),
    2 => ((:reflect_lo, :outer), (:periodic, :reflect_hi), (:outer, :periodic),
          (:reflect_both, :outer)),
    3 => ((:reflect_lo, :outer, :periodic), (:outer, :reflect_hi, :periodic),
          (:periodic, :periodic, :reflect_lo), (:outer, :outer, :reflect_both)))

function roundtrip_cases()
    cases = []
    for D in 1:3
        Cs = unique([cellcentered(D), vertexcentered(D), facecentered(D, D),
                     edgecentered(D, D)])
        for (ci, C) in enumerate(Cs), j in (D == 1 ? (1:4) : (0:0))
            t, k = D == 1 ? (j, j + 2(ci - 1)) : (ci + D - 2, ci + D - 1)
            push!(cases, (; D, C, types=ROUNDTRIP_TYPES[mod1(t, 4)],
                          kinds=ROUNDTRIP_KINDS[D][mod1(k, 4)],
                          filtered=isodd(length(cases))))
        end
    end
    return cases
end

@testset "A checkpoint round-trips bit for bit: D=$(c.D), C=$(c.C), $(c.types), $(c.kinds)" for
        c in roundtrip_cases()
    # The failure: anything a load gets not quite right — a block out of
    # order, a limb swapped, an extent rounded through Float64, a parity
    # dropped, the shared plane or a wall row left for the file to supply
    # — which a comparison to tolerance would let through and a restart
    # would carry into the interior. So everything is compared as bytes:
    # the state vector, and the working arrays once both sets have been
    # filled by the same schedule and the same hook, which is how an
    # application rebuilds what the file does not store.
    (; D, C, kinds) = c
    R, T = c.types
    p = D == 3 ? 2 : 4
    ops = Operators(prolongation=p, restriction=p)
    forest = faces_forest(kinds; T=R, N=D == 3 ? 4 : 8)
    f, parity = parity_data(T, kinds, p)
    reflects = any(forest.reflecting) do r
        r[1] || r[2]
    end
    fs = FieldSet{T}(forest, 2; G=ghosts_for(C, p), centering=C,
                     parity=reflects ? parity : nothing)
    fill_by_coordinates!(f, fs)
    # Structureless owned data on top of the polynomial, so that a
    # misplaced block or variable cannot happen to hold the right value.
    u = statevector(fs)
    gather!(u, fs)
    u .+= T.(rand(MersenneTwister(7), length(u))) ./ 64
    scatter!(fs, u)
    hook = boundary_by_coordinates(f)

    path = joinpath(mktempdir(), "roundtrip.h5")
    filters = c.filtered ? (HDF5.Filters.Shuffle(), HDF5.Filters.Deflate(1)) : ()
    @test save_checkpoint(path, forest; fieldsets=("state" => (fs, u),),
                          application="RoundTrip" => 3, data=(; D), filters) == path
    ck = load_checkpoint(path; types=(Float32x2,))
    loaded, v = ck.fieldsets["state"].fieldset, ck.fieldsets["state"].state

    @test ck.forest !== forest
    @test ck.forest.leaves == forest.leaves
    @test generation(ck.forest) == 0
    @test ck.forest.roots == forest.roots
    @test ck.forest.N == forest.N
    @test ck.forest.periodic == forest.periodic
    @test ck.forest.reflecting == forest.reflecting
    @test ck.forest.extents === forest.extents                 # bitwise, in R
    @test floattype(ck.forest) === R
    @test loaded.forest === ck.forest
    @test eltype(loaded.work) === T
    @test (loaded.nvars, loaded.G, loaded.centering) == (fs.nvars, fs.G, fs.centering)
    @test loaded.parity == fs.parity
    @test size(loaded.work) == size(fs.work)
    @test eltype(v) === T
    @test bytes(v) == bytes(u)
    @test ck.application == ("RoundTrip" => 3)
    @test ck.data === (; D)

    fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=hook)
    fill_ghosts!(loaded, GhostSchedule(loaded, ops); boundary=hook)
    @test bytes(loaded.work) == bytes(fs.work)
end

# --- restarts --------------------------------------------------------------

# The restart drivers: `track_pulse`'s and `track_shock`'s loops, cut
# down to what a checkpoint has to reproduce. A run is a NamedTuple of
# the forest, the evolved field set, whatever is rebuilt from them, and
# the run state (`t`, the chunk index) that goes in the plain data. Each
# chunk is a fresh fixed-step `solve`, then the flags, the regrid and the
# rebuild; a checkpoint is taken at a chunk boundary, after the regrid,
# which is where `CODE.md` puts it.
const PULSE = (; D=1, N=8, G=2, roots=8, L=1.0, σ=0.05, x0=0.25, chunk=0.05, cfl=0.25,
               maxlevel=2, threshold=0.05, buffer=4,
               ops=Operators(prolongation=4, restriction=4))

function pulse_flags(fs, P)
    thr = P.threshold
    fires(work, idx, b, x) = abs(work[idx..., 1, b]) > thr
    boxes = firing_boxes(fires, fs)
    return map(1:nblocks(fs)) do b
        n, box = boxes[b]
        k = blockkey(fs, b)
        n == 0 && return level(k) > 0 ? Coarsen : Keep
        return level(k) >= P.maxlevel ? (Keep, box) : (Refine, box)
    end
end

function pulse_start(P)
    D = P.D
    forest = Forest(ntuple(_ -> P.roots, D); N=P.N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, P.L), D))
    fs = FieldSet(forest, 2; G=P.G, centering=vertexcentered(D))
    initial = pulse_exact(D, P.L, P.x0, P.σ, 0.0)
    fill_by_coordinates!(initial, fs)
    schedule, _, _ = adapt_to_initial_data!(fs, P.ops; initial=initial,
                                            flags=_ -> pulse_flags(fs, P),
                                            buffer=P.buffer, maxpasses=8)
    return (; forest, fs, schedule, t=0.0, chunk=0)
end

function pulse_chunk(run, P)
    (; forest, fs, schedule, t, chunk) = run
    stop = (chunk + 1) * P.chunk
    u = statevector(fs)
    gather!(u, fs)
    dt = P.cfl * minimum_spacing(forest)
    nsteps = max(1, ceil(Int, (stop - t) / dt))
    sol = solve(ODEProblem(wave_rhs!, u, (t, stop), WaveProblem(fs, schedule)), RK4();
                dt=(stop - t) / nsteps, adaptive=false, save_everystep=false)
    scatter!(fs, sol.u[end])
    fill_ghosts!(fs, schedule)
    if regrid!(forest, fs => schedule; flags=pulse_flags(fs, P), buffer=P.buffer)
        schedule = GhostSchedule(fs, P.ops)
    end
    return (; forest, fs, schedule, t=stop, chunk=chunk + 1)
end

# The saved form: the bare field set, whose working array `regrid!` has
# just filled, so that the owned points are what the next chunk gathers.
pulse_save(path, run) =
    save_checkpoint(path, run.forest; fieldsets=("pulse" => run.fs,),
                    application="PulseRestart" => 1,
                    data=(; t=run.t, chunk=run.chunk, recipe=(; cfl=1//4, σ=1//20)))

function pulse_restore(path, P)
    ck = load_checkpoint(path)
    fs = ck.fieldsets["pulse"].fieldset
    @test ck.data.recipe === (; cfl=1//4, σ=1//20)
    return (; forest=ck.forest, fs, schedule=GhostSchedule(fs, P.ops), t=ck.data.t,
            chunk=ck.data.chunk)
end

const SHOCK = (; D=1, N=8, roots=8, G=2, L=1.0, ubar=1.0, amp=0.5, chunk=0.1, cfl=0.4,
               maxlevel=1, threshold=0.15, buffer=4, limiter=:minmod,
               ops=Operators(family=Conservative, prolongation=3, restriction=2))

function shock_flags(state, P)
    boxes = firing_boxes(burgers_fires(Val(P.D), P.threshold), state)
    return map(1:nblocks(state)) do b
        n, box = boxes[b]
        k = blockkey(state, b)
        n == 0 && return level(k) > 0 ? Coarsen : Keep
        return level(k) >= P.maxlevel ? (Keep, box) : (Refine, box)
    end
end

function shock_start(P)
    D = P.D
    forest = Forest(ntuple(_ -> P.roots, D); N=P.N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, P.L), D))
    state = FieldSet(forest, 1; G=P.G)
    initial = burgers_pointwise(P.L, P.ubar, P.amp)
    fill_by_coordinates!(initial, state)
    adapt_to_initial_data!(state, P.ops; initial=initial, flags=fs -> shock_flags(fs, P),
                           buffer=P.buffer, maxpasses=8)
    return (; forest, state, p=BurgersProblem(state, P.ops; limiter=P.limiter), t=0.0,
            chunk=0)
end

function shock_chunk(run, P)
    (; forest, state, p, t, chunk) = run
    stop = (chunk + 1) * P.chunk
    u = statevector(state)
    gather!(u, state)
    dt = burgers_dt(forest, P.cfl, P.ubar + P.amp, P.D)
    u = burgers_solve!(p, u, t, stop, max(1, ceil(Int, (stop - t) / dt)))
    scatter!(state, u)
    fill_ghosts!(state, p.schedule)
    # The flux sets ride along as `fs => nothing`: resized, not moved,
    # since every right-hand side overwrites them.
    pairs = (state => p.schedule, ntuple(d -> p.fluxes[d] => nothing, P.D)...)
    if regrid!(forest, pairs; flags=shock_flags(state, P), buffer=P.buffer)
        p = BurgersProblem(state, P.ops; limiter=P.limiter, fluxes=p.fluxes)
    end
    return (; forest, state, p, t=stop, chunk=chunk + 1)
end

# The recommended form, `(fs, u)`, with `u` gathered after the regrid.
# The flux sets are not saved: they are scratch, rebuilt by
# `BurgersProblem` on the loaded mesh.
function shock_save(path, run)
    u = statevector(run.state)
    gather!(u, run.state)
    return save_checkpoint(path, run.forest; fieldsets=("u" => (run.state, u),),
                           application="ShockRestart" => 1,
                           data=(; t=run.t, chunk=run.chunk))
end

function shock_restore(path, P)
    ck = load_checkpoint(path)
    state = ck.fieldsets["u"].fieldset
    @test collect(keys(ck.fieldsets)) == ["u"]
    return (; forest=ck.forest, state, p=BurgersProblem(state, P.ops; limiter=P.limiter),
            t=ck.data.t, chunk=ck.data.chunk)
end

# Run `n` chunks without a break, and again with a checkpoint after chunk
# `k`, every object dropped, and the rest run from what was loaded.
# Returns both final runs and the uninterrupted run's leaf history.
function interrupted(start, step, save, restore, P, n, k)
    run = start(P)
    history = [copy(run.forest.leaves)]
    for _ in 1:n
        run = step(run, P)
        push!(history, copy(run.forest.leaves))
    end
    path = joinpath(mktempdir(), "restart.h5")
    broken = start(P)
    for _ in 1:k
        broken = step(broken, P)
    end
    save(path, broken)
    broken = nothing                              # nothing survives but the file
    resumed = restore(path, P)
    for _ in (k + 1):n
        resumed = step(resumed, P)
    end
    return run, resumed, history
end

@testset "A restart continues bit-identically through regrids: $name" for
        (name, start, step, save, restore, P, n, k, field) in
        (("the vertex-centered wave", pulse_start, pulse_chunk, pulse_save,
          pulse_restore, PULSE, 4, 2, :fs),
         ("the Burgers shock, with the interface fixup", shock_start, shock_chunk,
          shock_save, shock_restore, SHOCK, 6, 3, :state))
    # The failure: a restarted run that drifts from the uninterrupted one
    # — through state the checkpoint does not hold (a time rounded, an
    # integrator or regrid input left behind, a flux set assumed to carry
    # something over), or through data the load puts back wrongly. The
    # mesh has to change after the restart point, or a dependence of the
    # regrid on something unsaved would go unseen; and the Burgers run's
    # flux sets, which are not saved, are rebuilt fresh.
    run, resumed, history = interrupted(start, step, save, restore, P, n, k)
    @test any(c -> history[c + 1] != history[c], (k + 1):n)  # the mesh really moved
    @test resumed.chunk == run.chunk == n
    @test resumed.t === run.t
    @test resumed.forest.leaves == run.forest.leaves
    fs, fs′ = getfield(run, field), getfield(resumed, field)
    u, u′ = statevector(fs), statevector(fs′)
    gather!(u, fs)
    gather!(u′, fs′)
    @test length(u′) == length(u)
    @test bytes(u′) == bytes(u)
end

# --- rotating forests (M12) ----------------------------------------------------

# A rotating quadrant with a symmetric set — a scalar and a vector, which
# the seam turns into themselves — and a `RotationPair` of face-centered
# sets, `(B_{d1}, F_{d1})` and `(F_{d2}, B_{d2})`, which turn into each
# other: the sets of `rotating_vs_quadrupled` and
# `rotating_pair_vs_quadrupled` (`ghost_oracles.jl`), in element type `T`.
function rotating_checkpoint_sets(::Val{D}, ::Type{T}; other, p=4, poly=0) where {D,T}
    rotating = (1, 2)
    d1, d2 = rotating
    z = outofplane(D, rotating)
    forest = rotating_forest(Val(D); rotating=rotating, other=other, N=D == 3 ? 4 : 8)
    f = rotating_data(D, rotating, other; poly=poly)
    fs = FieldSet{T}(forest, D + 1; G=ghosts_for(vertexcentered(D), p),
                     centering=vertexcentered(D), rotation=vector_rotation(D, rotating),
                     parity=vector_parity(D, rotating, other))
    field(x, k) = covariant_vector(x[d1], x[d2], k; poly=poly)
    zf(x, odd) = outofplane_factor(x, D, rotating, other, odd; poly=poly)
    fa = (x, v) -> v == 1 ? field(x, 1)[1] * zf(x, false) : field(x, 2)[1] * zf(x, true)
    fb = (x, v) -> v == 1 ? field(x, 2)[2] * zf(x, true) : field(x, 1)[2] * zf(x, false)
    parity(odd) = other === :reflect_lo ?
                  [ntuple(d -> d != z ? NoParity : o ? OddParity : EvenParity, D)
                   for o in odd] : nothing
    Ca = facecentered(D, d1)
    swap(t) = Base.setindex(Base.setindex(t, t[d2], d1), t[d1], d2)
    a = FieldSet{T}(forest, 2; G=ghosts_for(Ca, p), centering=Ca,
                    parity=parity((false, true)), rotation=(-2, -1))
    b = FieldSet{T}(forest, 2; G=swap(ghosts_for(Ca, p)), centering=swap(Ca),
                    parity=parity((true, false)), rotation=(2, 1))
    hooks = (boundary_by_coordinates(f), boundary_by_coordinates(fa),
             boundary_by_coordinates(fb))
    return forest, (fs, a, b), (f, fa, fb), hooks
end

file_features(path) = h5open(file -> read_attribute(file["TreeAMR.jl"], "features"), path)

@testset "A rotating checkpoint round-trips bit for bit, a pair included: D=$D, $T" for
        (D, T, other) in ((2, Float32x2, :outer), (3, Float64, :reflect_lo))
    # The failure: a load that drops the seam — the forest comes back with
    # two outer faces where the seam was, or a set without its rotation —
    # which only the next fill would show, as ghosts the hook wrote
    # instead of the turned data; or a reader that does not know the seam
    # and reads it all the same. So the forest's pair, every set's map and
    # tables, the state, and the working arrays after both sides fill the
    # same way, the pair through a pair rebuilt from the two loaded sets,
    # are compared as bytes; and the file lists the feature `rotating`,
    # which a reader from before M12 does not know and so refuses.
    p = D == 3 ? 2 : 4
    ops = Operators(prolongation=p, restriction=p)
    # Polynomial data in `Float32x2`, which has no `sin`; the noise below
    # makes every value structureless all the same.
    forest, sets, formulas, hooks = rotating_checkpoint_sets(Val(D), T; other=other, p=p,
                                                             poly=T === Float64 ? 0 : p)
    rng = MersenneTwister(12)
    states = map(sets, formulas) do fs, f
        fill_by_coordinates!(f, fs)
        u = statevector(fs)
        gather!(u, fs)
        u .+= T.(rand(rng, length(u))) ./ 64
        scatter!(fs, u)
        u
    end
    names = ("u", "a", "b")
    path = joinpath(mktempdir(), "rotating.h5")
    save_checkpoint(path, forest; fieldsets=Tuple(n => (fs, u) for (n, fs, u) in
                                                  zip(names, sets, states)),
                    application="Rotating" => 1)
    @test file_features(path) == ["brick", "rotating"]
    ck = load_checkpoint(path; types=(Float32x2,))
    @test ck.forest.rotating == forest.rotating
    @test TreeAMR.rotating_dims(ck.forest) == (1, 2)
    @test ck.forest.leaves == forest.leaves
    @test ck.forest.reflecting == forest.reflecting
    loaded = map(n -> ck.fieldsets[n].fieldset, names)
    for (fs, back, n, u) in zip(sets, loaded, names, states)
        @test back.rotation == fs.rotation
        @test (back.G, back.centering, back.parity) == (fs.G, fs.centering, fs.parity)
        @test Array(back.rotvars) == Array(fs.rotvars)
        @test bytes(back.factors) == bytes(fs.factors)
        @test bytes(ck.fieldsets[n].state) == bytes(u)
    end

    fs, a, b = sets
    fs′, a′, b′ = loaded
    for (s, set) in ((fs, sets), (fs′, loaded))
        fill_ghosts!(set[1], GhostSchedule(set[1], ops); boundary=hooks[1])
        fill_ghosts!(RotationPair(set[2], set[3]),
                     (GhostSchedule(set[2], ops), GhostSchedule(set[3], ops));
                     boundary=hooks[2:3])
    end
    @test bytes(fs′.work) == bytes(fs.work)
    @test bytes(a′.work) == bytes(a.work)
    @test bytes(b′.work) == bytes(b.work)
end

# The quadrant wave as a chunked driver, as `pulse_start` and
# `pulse_chunk` are the periodic one's: a ring around the axis, which the
# seam turns into itself, vertex-centered, with outer high faces whose
# ghosts the hook writes from the initial data. Beside it a `RotationPair`
# rides through every regrid unevolved, so that what a pair's fill and
# transfer do after a restart is compared too.
const QUADRANT = (; D=2, N=8, G=2, roots=4, L=1.0, σ=0.08, r0=0.45, chunk=0.04,
                  cfl=0.25, maxlevel=2, threshold=0.1, buffer=3,
                  ops=Operators(prolongation=4, restriction=4))

quadrant_ring(P) = (x, v) -> v == 1 ? exp(-(hypot(x[1], x[2]) - P.r0)^2 / (2 * P.σ^2)) :
                                 0.0
quadrant_pair(x, v) = v == 1 ? covariant_vector(x[1], x[2], 1)[1] :
                      covariant_vector(x[1], x[2], 1)[2]

struct QuadrantWave{W,H}
    wave::W
    hook::H
end

function quadrant_rhs!(du, u, q, t)
    p = q.wave
    scatter!(p.fs, u)
    fill_ghosts!(p.fs, p.schedule; boundary=q.hook)
    map_blocks!(wave_rhs_kernel!, p.fs, statearray(du, p.fs), p.fs.work, p.spacings,
                p.valD, p.valG)
    return nothing
end

function quadrant_run(forest, fs, pair, t, chunk, P)
    hook = formula_hook(fs => quadrant_ring(P), pair.a => quadrant_pair,
                        pair.b => quadrant_pair)
    return (; forest, fs, pair, schedule=GhostSchedule(fs, P.ops),
            pairschedules=(GhostSchedule(pair.a, P.ops), GhostSchedule(pair.b, P.ops)),
            hook, t, chunk)
end

function quadrant_start(P)
    forest = Forest((P.roots, P.roots); N=P.N, rotating=(1, 2),
                    extents=((0.0, P.L), (0.0, P.L)))
    fs = FieldSet(forest, 2; G=P.G, centering=vertexcentered(2), rotation=(1, 2))
    ring = quadrant_ring(P)
    hook = boundary_by_coordinates(ring)
    adapt_to_initial_data!(fs, P.ops; initial=ring, boundary=hook,
                           flags=_ -> pulse_flags(fs, P), buffer=P.buffer, maxpasses=8)
    # The pair starts on the adapted mesh, which its regrids carry on.
    a = FieldSet(forest, 1; G=ghosts_for(facecentered(2, 1), 4),
                 centering=facecentered(2, 1), rotation=(-1,))
    b = FieldSet(forest, 1; G=ghosts_for(facecentered(2, 2), 4),
                 centering=facecentered(2, 2), rotation=(1,))
    fill_by_coordinates!(quadrant_pair, a)
    fill_by_coordinates!((x, v) -> covariant_vector(x[1], x[2], 1)[2], b)
    pair = RotationPair(a, b)
    return quadrant_run(forest, fs, pair, 0.0, 0, P)
end

function quadrant_chunk(run, P)
    (; forest, fs, pair, schedule, pairschedules, hook, t, chunk) = run
    stop = (chunk + 1) * P.chunk
    u = statevector(fs)
    gather!(u, fs)
    dt = P.cfl * minimum_spacing(forest)
    nsteps = max(1, ceil(Int, (stop - t) / dt))
    problem = QuadrantWave(WaveProblem(fs, schedule), hook)
    sol = solve(ODEProblem(quadrant_rhs!, u, (t, stop), problem), RK4();
                dt=(stop - t) / nsteps, adaptive=false, save_everystep=false)
    scatter!(fs, sol.u[end])
    fill_ghosts!(fs, schedule; boundary=hook)
    regrid!(forest, (fs => schedule, pair => pairschedules); flags=pulse_flags(fs, P),
            buffer=P.buffer, boundary=hook)
    return quadrant_run(forest, fs, pair, stop, chunk + 1, P)
end

quadrant_save(path, run) =
    save_checkpoint(path, run.forest;
                    fieldsets=("wave" => run.fs, "a" => run.pair.a, "b" => run.pair.b),
                    application="QuadrantRestart" => 1, data=(; t=run.t, chunk=run.chunk))

function quadrant_restore(path, P)
    ck = load_checkpoint(path)
    sets = ck.fieldsets
    return quadrant_run(ck.forest, sets["wave"].fieldset,
                        RotationPair(sets["a"].fieldset, sets["b"].fieldset), ck.data.t,
                        ck.data.chunk, P)
end

@testset "A restart of a rotating quadrant continues bit-identically through regrids" begin
    # The failure: a restarted quadrant that drifts from the uninterrupted
    # one — a seam lost or a map dropped on the way through the file,
    # which turns the ghosts across the seam into the hook's, or a pair
    # that a load cannot rebuild as it was. The mesh moves after the
    # restart point, and the ring reaches the seam, so the regrids there
    # are conforming ones; the pair is filled and transferred as a pair
    # in every regrid, before and after the restart.
    P = QUADRANT
    n, k = 4, 2
    run, resumed, history = interrupted(quadrant_start, quadrant_chunk, quadrant_save,
                                        quadrant_restore, P, n, k)
    @test any(c -> history[c + 1] != history[c], (k + 1):n)
    @test all(h -> isbalanced(Forest((P.roots, P.roots); N=P.N, rotating=(1, 2),
                                     leaves=h)), history)
    @test resumed.chunk == run.chunk == n
    @test resumed.t === run.t
    @test resumed.forest.leaves == run.forest.leaves
    for (x, y) in ((run.fs, resumed.fs), (run.pair.a, resumed.pair.a),
                   (run.pair.b, resumed.pair.b))
        u, u′ = statevector(x), statevector(y)
        gather!(u, x)
        gather!(u′, y)
        @test bytes(u′) == bytes(u)
    end
end

@testset "A rotating file without its feature or its seam is refused, saying why" begin
    # The failure: a file whose seam and feature disagree, or whose maps
    # are damaged, read anyway — with the seam as two outer faces, or a
    # map the constructor would refuse from a caller. And a file without
    # a seam must not list the feature, so that every reader of its
    # format version still reads it.
    forest, sets, formulas, _ = rotating_checkpoint_sets(Val(2), Float64; other=:outer)
    foreach(fill_by_coordinates!, formulas, sets)
    dir = mktempdir()
    path = joinpath(dir, "rotating.h5")
    save_checkpoint(path, forest; fieldsets=("u" => sets[1], "a" => sets[2]),
                    application="Rotating" => 1)
    @test load_checkpoint(path).forest.rotating == forest.rotating
    plain = joinpath(dir, "plain.h5")
    save_checkpoint(plain, Forest((2, 2); N=4); fieldsets=(),
                    application="Rotating" => 1)
    @test file_features(plain) == ["brick"]

    unlisted = edited_copy(path, "unlisted.h5") do file
        replace_attribute!(file["TreeAMR.jl"], "features", ["brick"])
    end
    @test_throws "does not list the feature \"rotating\"" load_checkpoint(unlisted)
    @test_throws "checkpoint_environment(path, dir)" load_checkpoint(unlisted)
    seamless = edited_copy(path, "seamless.h5") do file
        delete_attribute(file["TreeAMR.jl/forest"], "rotating")
    end
    @test_throws "lists the feature \"rotating\", but its forest has no" (
        load_checkpoint(seamless))
    unknown = edited_copy(path, "unknown.h5") do file
        replace_attribute!(file["TreeAMR.jl"], "features",
                           ["brick", "rotating", "halfturn"])
    end
    @test_throws "the feature \"halfturn\"" load_checkpoint(unknown)
    # The pair as stored is checked by the forest: a dimension out of
    # range, and the seam on a periodic dimension.
    outside = edited_copy(path, "outside.h5") do file
        replace_attribute!(file["TreeAMR.jl/forest"], "rotating", [1, 3])
    end
    @test_throws "names a dimension outside 1:2" load_checkpoint(outside)
    unpaired = edited_copy(path, "unpaired.h5") do file
        replace_attribute!(file["TreeAMR.jl/forest"], "rotating", [1, 2, 3])
    end
    @test_throws "is not a pair of dimensions" load_checkpoint(unpaired)
    # A map that is not a signed permutation, one of the wrong length, and
    # none at all on a rotating forest.
    unturned = edited_copy(path, "unturned.h5") do file
        replace_attribute!(file["TreeAMR.jl/fieldsets/u"], "rotation", [1, 2, 2])
    end
    @test_throws "is not a signed permutation" load_checkpoint(unturned)
    short = edited_copy(path, "short.h5") do file
        replace_attribute!(file["TreeAMR.jl/fieldsets/a"], "rotation", [-2])
    end
    @test_throws "field set \"a\" has the rotation [-2], not 2 signed" load_checkpoint(short)
    mapless = edited_copy(path, "mapless.h5") do file
        delete_attribute(file["TreeAMR.jl/fieldsets/u"], "rotation")
    end
    @test_throws "so the field set needs `rotation`" load_checkpoint(mapless)
end

# --- refusals ----------------------------------------------------------------

@testset "A file this version cannot interpret is refused, saying why" begin
    # The failure: a reader that goes ahead with a file it does not
    # understand — a newer format, a must-understand feature it lacks, a
    # type it cannot name or whose layout differs, a damaged leaf list —
    # and returns a run that is silently not the one that was saved; or a
    # refusal that does not say what to do about it.
    dir = mktempdir()
    forest = Forest((2, 2); N=4, periodic=(true, true))
    fs = FieldSet(forest, 1; G=1)
    fill_by_coordinates!((x, v) -> x[1] + 2x[2], fs)
    path = joinpath(dir, "good.h5")
    save_checkpoint(path, forest; fieldsets=("u" => fs,), application="Refusals" => 1)
    @test load_checkpoint(path).forest.leaves == forest.leaves

    newer = edited_copy(path, "newer.h5") do file
        replace_attribute!(file["TreeAMR.jl"], "format_version", 3)
    end
    @test_throws "format version 3" load_checkpoint(newer)
    @test_throws "reads format versions 1 and 2" load_checkpoint(newer)
    @test_throws "written by a newer TreeAMR" load_checkpoint(newer)
    @test_throws "The file was written by TreeAMR $(pkgversion(TreeAMR))" load_checkpoint(
        newer)
    @test_throws "checkpoint_environment(path, dir)" load_checkpoint(newer)

    feature = edited_copy(path, "feature.h5") do file
        replace_attribute!(file["TreeAMR.jl"], "features", ["brick", "multiblock"])
    end
    @test_throws "the feature \"multiblock\"" load_checkpoint(feature)
    @test_throws "refused rather than misread" load_checkpoint(feature)

    # A limb type the reader cannot name, in the geometry and in a field
    # set narrower than it, and one whose layout is not the file's.
    wide = Forest{Float32x2}((2,); N=4, periodic=(true,))
    widepath = joinpath(dir, "wide.h5")
    save_checkpoint(widepath, wide; fieldsets=("u" => FieldSet(wide, 1; G=1),),
                    application="Refusals" => 1)
    @test_throws "the forest's geometry is stored in MultiFloats.MultiFloat{Float32, 2}" (
        load_checkpoint(widepath))
    @test_throws "pass the type in `types`" load_checkpoint(widepath)
    @test load_checkpoint(widepath; types=(Float32x2,)).forest.extents === wide.extents
    narrow = Forest((2,); N=4, periodic=(true,))
    narrowpath = joinpath(dir, "narrow.h5")
    save_checkpoint(narrowpath, narrow;
                    fieldsets=("u" => FieldSet{Float32x2}(narrow, 1; G=1),),
                    application="Refusals" => 1)
    @test_throws "field set \"u\" is stored in MultiFloats" load_checkpoint(narrowpath)
    @test_throws "types" load_checkpoint(narrowpath; types=(Float32,))
    relimbed = edited_copy(narrowpath, "relimbed.h5") do file
        replace_attribute!(file["TreeAMR.jl/fieldsets/u"], "nlimbs", 4)
    end
    @test_throws "does not match the file's" load_checkpoint(relimbed; types=(Float32x2,))
    @test_throws "collection of types" load_checkpoint(narrowpath; types=(1,))

    # A leaf list out of order is refused by the forest, as it would be
    # from any other caller: here with the checksum recomputed, which would
    # refuse it first.
    swapped = edited_copy(path, "swapped.h5") do file
        g = file["TreeAMR.jl/forest"]
        r = read(g["root"])
        g["root"][1:2] = r[[2, 1]]
        ext = Base.get_extension(TreeAMR, :TreeAMRHDF5Ext)
        replace_attribute!(g, "leaves_crc32c",
                           ext.leaves_checksum(read(g["root"]), read(g["level"]),
                                               read(g["coords"])))
    end
    @test_throws "out of curve order" load_checkpoint(swapped)

    # Damage that leaves a valid file: a leaf column's rows zeroed, as one
    # rank's were in the four-node run that showed the need (M7 step 6),
    # and one value of a field set changed, in the contiguous layout and
    # in the chunked one; and the value changed with its block's checksum
    # recomputed, which the index's checksum of the part's checksums
    # refuses. In format version 2 the checksums are required: a file
    # without them is damaged (a version-1 file without them loads
    # unchecked, below).
    big = Forest((4, 4); N=4, periodic=(true, true))
    refine!(big, big.leaves[[1, 6]])
    bigfs = FieldSet(big, 2; G=1)
    fill_by_coordinates!((x, v) -> x[1] + 2x[2] + v, bigfs)
    for filters in ((), (HDF5.Filters.Shuffle(), HDF5.Filters.Deflate(1)))
        intact = joinpath(dir, "intact.h5")
        save_checkpoint(intact, big; fieldsets=("u" => bigfs,), application="Refusals" => 1,
                        filters=filters)
        zeroed = edited_copy(intact, "zeroed.h5") do file
            coords = file["TreeAMR.jl/forest/coords"]
            c = read(coords)
            c[:, 9:14] .= 0
            coords[:, :] = c
        end
        @test_throws "the leaf list does not match the checksum" load_checkpoint(zeroed)
        @test_throws "refused rather than read into a wrong mesh" load_checkpoint(zeroed)
        @test_throws "checkpoint_environment(path, dir)" load_checkpoint(zeroed)
        changed = edited_copy(intact, "changed.h5") do file
            data = file["$PART/fieldsets/u/data"]
            d = read(data)
            d[3, 2, 1, 7] += 1
            data[:, :, :, :] = d
        end
        @test_throws("the data of field set \"u\" do not match the checksums stored " *
                     "with them in 1 of its $(nleaves(big)) blocks, the first being block 7",
                     load_checkpoint(changed))
        resummed = edited_copy(changed, "resummed.h5") do file
            ext = Base.get_extension(TreeAMR, :TreeAMRHDF5Ext)
            sums = file["$PART/fieldsets/u/data_crc32c"]
            d = read(file["$PART/fieldsets/u/data"])
            s = read(sums)
            s[7] = ext.crc_of(collect(vec(d[:, :, :, 7])))
            sums[:] = s
        end
        @test_throws("holds checksums of field set \"u\" that do not match the index's",
                     load_checkpoint(resummed))
        unsummed = edited_copy(intact, "unsummed.h5") do file
            delete_object(file["$PART/fieldsets/u"], "data_crc32c")
        end
        @test_throws "has no checksums of field set \"u\"" load_checkpoint(unsummed)
        unleaved = edited_copy(intact, "unleaved.h5") do file
            delete_attribute(file["TreeAMR.jl/forest"], "leaves_crc32c")
        end
        @test_throws "the leaf list has no checksum" load_checkpoint(unleaved)
    end

    @test_throws "no field set named \"nope\"; it holds \"u\"" load_checkpoint(
        path; fieldsets=("nope",))

    # Not a checkpoint at all: no file, not HDF5, HDF5 without our group.
    @test_throws "no such file" load_checkpoint(joinpath(dir, "missing.h5"))
    text = joinpath(dir, "text.h5")
    write(text, "not HDF5")
    @test_throws "not a TreeAMR checkpoint: it is not an HDF5 file" load_checkpoint(text)
    other = joinpath(dir, "other.h5")
    h5open(file -> write(file, "x", [1, 2, 3]), other, "w")
    @test_throws "not a TreeAMR checkpoint: it is an HDF5 file with no /TreeAMR.jl" (
        load_checkpoint(other))
end

@testset "What cannot be saved is refused before anything is written" begin
    # The failure: a checkpoint that stores something it cannot restore —
    # a field set over another mesh, a state vector of the wrong length or
    # type, a struct whose type name would tie the file to its definition,
    # an application group that collides with TreeAMR's — or, having
    # started, leaves a partial file behind.
    dir = mktempdir()
    path = joinpath(dir, "run.h5")
    forest = Forest((2, 2); N=4, periodic=(true, true))
    fs = FieldSet(forest, 1; G=1)
    save(; kw...) = save_checkpoint(path, forest;
                                    (; fieldsets=("u" => fs,), application="Saves" => 1,
                                     kw...)...)
    twin = Forest((2, 2); N=4, periodic=(true, true))          # equal, not the same
    @test_throws "over another forest" save(fieldsets=("u" => FieldSet(twin, 1; G=1),))
    @test_throws "two field sets are named \"u\"" save(fieldsets=("u" => fs, "u" => fs))
    @test_throws "has 3 entries, but the field set needs 64" save(
        fieldsets=("u" => (fs, zeros(3)),))
    @test_throws "has element type Float32" save(
        fieldsets=("u" => (fs, zeros(Float32, statelength(fs))),))
    @test_throws "cannot name an HDF5 object" save(fieldsets=("a/b" => fs,))
    @test_throws "no default `fieldsets`" save_checkpoint(path, forest;
                                                         application="Saves" => 1)
    @test_throws "no default `application`" save_checkpoint(path, forest;
                                                           fieldsets=("u" => fs,))
    @test_throws "cannot be named \"TreeAMR.jl\"" save(application="TreeAMR.jl" => 1)
    @test_throws "cannot name an HDF5 object" save(application="My/App" => 1)
    @test_throws "`application` is `name => version`" save(application="MyApp")
    @test_throws "/Saves/data/params is a CheckpointParams, which is not plain data" save(
        data=(; t=0.0, params=CheckpointParams(0.25)))
    @test_throws "Convert anything else to a NamedTuple" save(
        data=(; params=CheckpointParams(0.25)))
    @test readdir(dir) == []                         # neither the file nor a partial one
end

@testset "A write that fails leaves the previous checkpoint intact" begin
    # The failure: a crash or an error in the middle of a write that
    # destroys the checkpoint a restart would need, or a `.partial` file
    # left for the next write to trip over; here an error of the do-block,
    # and the I/O process failing after its first write (a test hook).
    dir = mktempdir()
    path = joinpath(dir, "run.h5")
    forest = Forest((2,); N=4, periodic=(true,))
    fs = FieldSet(forest, 1; G=1)
    save(f, which) = save_checkpoint(f, path, forest; fieldsets=("u" => fs,),
                                     application="Atomic" => 1, data=(; which))
    save(_ -> nothing, 1)
    @test_throws "interrupted while writing" save(2) do app
        write_plain(app, "extra", [1, 2, 3])
        error("interrupted while writing")
    end
    @test load_checkpoint(path).data.which == 1
    @test readdir(dir) == ["run.h5"]
    ext = Base.get_extension(TreeAMR, :TreeAMRHDF5Ext)
    ext.FAIL_PART[] = 0
    try
        @test_throws "failed on purpose after its first write" save(_ -> nothing, 4)
    finally
        ext.FAIL_PART[] = -1
    end
    @test load_checkpoint(path).data.which == 1
    @test readdir(dir) == ["run.h5"]
    # And a write that succeeds replaces it, with what its do-block added.
    save(3) do app
        write_plain(app, "extra", [4, 5, 6])
    end
    ck = load_checkpoint(app -> read_plain(app, "extra"), path)
    @test ck.data.which == 3
    @test ck.result == [4, 5, 6]
    @test readdir(dir) == ["run.h5"]
end

@testset "A synced write and an unsynced one store the same checkpoint" begin
    # The failure: the flush to stable storage that `sync` adds changing
    # what is written, failing on an ordinary file or directory, or
    # passing over an error in silence. That the data then survive a
    # power loss is the operating system's promise, and not testable
    # here.
    dir = mktempdir()
    forest = Forest((2, 2); N=4, periodic=(true, false))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    fs = FieldSet(forest, 2; G=1)
    fill_by_coordinates!((x, v) -> v * x[1] - x[2], fs)
    paths = map((true, false)) do sync
        path = joinpath(dir, "sync-$sync.h5")
        @test save_checkpoint(path, forest; fieldsets=("u" => fs,),
                              application="Sync" => 1, data=(; sync), sync) == path
        return path
    end
    synced, unsynced = load_checkpoint.(paths)
    @test synced.forest.leaves == unsynced.forest.leaves == forest.leaves
    @test bytes(synced.fieldsets["u"].state) == bytes(unsynced.fieldsets["u"].state)
    @test (synced.data.sync, unsynced.data.sync) == (true, false)
    @test sort(readdir(dir)) == ["sync-false.h5", "sync-true.h5"]
    # One file, of format version 2, its one part inside it, and a save id.
    h5open(paths[1]) do file
        root = file["TreeAMR.jl"]
        @test read_attribute(root, "format_version") == 2
        @test occursin(r"^[0-9a-f]{32}$", read_attribute(root, "save_id"))
        @test read(root["parttable/file"]) == [""]
        @test (read(root["parttable/first_block"]), read(root["parttable/last_block"])) ==
              ([1], [nleaves(forest)])
        @test read_attribute(file[PART], "save_id") == read_attribute(root, "save_id")
        @test haskey(file, "$PART/fieldsets/u/data")
        @test !haskey(root, "fieldsets/u/data")
    end
    # The flush itself, on a file, on a directory, and refusing a path
    # that is not there rather than skipping it.
    ext = Base.get_extension(TreeAMR, :TreeAMRHDF5Ext)
    @test ext.flush_to_storage(paths[1]) === nothing
    @test ext.flush_to_storage(dir; directory=true) === nothing
    Sys.iswindows() ||
        @test_throws SystemError ext.flush_to_storage(joinpath(dir, "missing.h5"))
end

# --- plain data ----------------------------------------------------------------

@testset "Plain data round-trip exactly" begin
    # The failure: application state that comes back as something else —
    # a Rational through Float64, a Symbol as a String, a Bool as a
    # bitfield another reader cannot read, a NamedTuple's fields
    # reordered, a Dict's keys changing type, an integer narrowed or
    # widened, a Float16 or a subnormal rounded.
    values = (; r8=Int8(-3) // Int8(7), r32=Int32(1) // Int32(3),
              r64=typemax(Int64) // 2, ru16=UInt16(5) // UInt16(9), rinf=1 // 0,
              sym=:alpha, spaced=Symbol("with space"), empty="", text="ünïcödé",
              version=v"0.1.3-rc1+build.5", nothing_=nothing, yes=true, no=false,
              i8=Int8(-128), u64=typemax(UInt64), i64=typemin(Int64),
              f16=Float16(0.1), f32=1.0f0 / 3, negzero=-0.0, sub=floatmin(Float64) / 3,
              inf=-Inf, nan=NaN, c64=1.0 + 2.0im, c32i=Complex{Int32}(1, -2), cbool=im,
              c16=ComplexF16(0.5, -1),
              ints=[1, 2, 3], matrix=[1.0 2.0; 3.0 4.0], noints=Int[],
              f32s=Float32[1, 2, 3], bools=Bool[true, false, true],
              complexes=[1.0im, 2.0 + 0im], strings=["a" "b"; "c" "d"], nostrings=String[],
              tuple=(1, "two", 3.0, :four, nothing, 5 // 6), same=(1.5, 2.5, 3.5),
              sameints=(1, 2, 3), notuple=(), one=(7,),
              nested=(; z=1, a=(; c=2.5, b=(; deep=:yes)), m=[1 2]),
              strdict=Dict("x" => 1, "y" => [1.0, 2.0]),
              symdict=Dict(:p => (; q=1), :r => "s"), emptydict=Dict{String,Int}())
    path = joinpath(mktempdir(), "plain.h5")
    forest = Forest((2,); N=4, periodic=(true,))
    save_checkpoint(path, forest; fieldsets=(), application="Plain" => 7, data=values)
    ck = load_checkpoint(path)
    @test ck.application == ("Plain" => 7)
    @test isempty(ck.fieldsets)
    @test keys(ck.data) == keys(values)                        # field order kept
    @test keys(ck.data.nested) == (:z, :a, :m)
    for k in keys(values)
        want, got = values[k], ck.data[k]
        if want isa AbstractDict
            @test got isa Dict{keytype(want),Any}
            @test isequal(got, want)
        else
            @test typeof(got) === typeof(want)
            @test isequal(got, want)
            isbits(want) && @test got === want
        end
    end

    # The same through `write_plain` and `read_plain` on a bare file, and
    # the values that are refused, with the item's path in the message.
    bare = joinpath(mktempdir(), "bare.h5")
    h5open(bare, "w") do file
        write_plain(file, "recipe", (; cfl=2 // 5, name="sod"))
        g = create_group(file, "nested")
        write_plain(g, "x", 1.5)
        @test_throws "/char is a Char, which is not plain data" write_plain(file, "char", 'c')
        @test_throws "BigFloat" write_plain(file, "big", big(1.0))
        @test_throws "MultiFloat" write_plain(file, "limbs", Float32x2(1))
        @test_throws "a Dict with keys of type Int64" write_plain(file, "d", Dict(1 => 2))
        @test_throws "an array of Any" write_plain(file, "a", Any[1, 2])
        @test_throws "/nested/f/f is a" write_plain(g, "f", (; f=sin))
        @test_throws "cannot name an HDF5 object" write_plain(file, "a/b", 1)
        @test_throws "cannot name an HDF5 object" write_plain(
            file, "t", NamedTuple{(Symbol("a/b"),)}((1,)))
        @test_throws "already holds an item named \"recipe\"" write_plain(file, "recipe", 1)
        # A NUL, which an HDF5 string cannot hold, refused before anything
        # is written.
        @test_throws "/nul holds a string with a NUL character" write_plain(file, "nul",
                                                                         "a\0b")
        @test_throws "/nuls/n holds a string with a NUL" write_plain(file, "nuls",
                                                                  (; ok="x", n=["\0"]))
        @test !haskey(file, "nul") && !haskey(file, "nuls")
        write(file, "foreign", [1, 2])
    end
    h5open(bare, "r") do file
        @test read_plain(file, "recipe") === (; cfl=2 // 5, name="sod")
        @test read_plain(file["nested"], "x") === 1.5
        @test_throws "no `type` attribute" read_plain(file, "foreign")
        @test_throws "holds no item named \"missing\"" read_plain(file, "missing")
    end
end

# --- environment, provenance and subsets -----------------------------------------

@testset "checkpoint_environment writes the stored environment, even for a refused file" begin
    # The failure: a refusal whose remedy does not work — the texts
    # written out differ from the writer's, or the function itself refuses
    # the file whose format it exists to get around — and provenance that
    # does not say who wrote the file.
    dir = mktempdir()
    path = joinpath(dir, "run.h5")
    forest = Forest((2,); N=4, periodic=(true,))
    save_checkpoint(path, forest; fieldsets=(), application="Environment" => 1)
    p = load_checkpoint(path).provenance
    @test p.project == read(Base.active_project(), String)
    @test !isempty(p.manifest)
    @test p.treeamr_version == pkgversion(TreeAMR)
    @test p.julia_version == VERSION
    @test p.nthreads == Threads.nthreads()
    # The rank count, added by M7 (an additive field): a serial file says
    # 1, and a file written before there was one reads as 1 too.
    @test p.nranks == 1
    @test p.nparts == 1
    older = edited_copy(path, "older.h5") do file
        delete_object(file, "TreeAMR.jl/provenance/nranks")
        delete_object(file, "TreeAMR.jl/provenance/nparts")
    end
    @test load_checkpoint(older).provenance.nranks == 1
    @test load_checkpoint(older).provenance.nparts == 1
    @test p.hostname == gethostname()
    @test occursin(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", p.created)

    env = joinpath(dir, "env")
    @test checkpoint_environment(path, env) == env
    @test read(joinpath(env, "Project.toml"), String) == p.project
    @test read(joinpath(env, "Manifest.toml"), String) == p.manifest
    @test_throws "pass `force = true`" checkpoint_environment(path, env)
    @test checkpoint_environment(path, env; force=true) == env

    refused = edited_copy(path, "refused.h5") do file
        replace_attribute!(file["TreeAMR.jl"], "format_version", 99)
    end
    @test_throws "format version 99" load_checkpoint(refused)
    env2 = checkpoint_environment(refused, joinpath(dir, "env2"))
    @test read(joinpath(env2, "Project.toml"), String) == p.project
    @test read(joinpath(env2, "Manifest.toml"), String) == p.manifest

    # A writer without a manifest stored an empty text: only the project
    # is written, and a warning says so.
    bare = edited_copy(path, "bare.h5") do file
        delete_object(file, "TreeAMR.jl/provenance/manifest")
        write(file["TreeAMR.jl/provenance"], "manifest", "")
    end
    env3 = joinpath(dir, "env3")
    @test_logs (:warn, r"stores no Manifest.toml") checkpoint_environment(bare, env3)
    @test isfile(joinpath(env3, "Project.toml"))
    @test !ispath(joinpath(env3, "Manifest.toml"))
    @test_throws "not a TreeAMR checkpoint" checkpoint_environment(
        joinpath(dir, "env", "Project.toml"), joinpath(dir, "env4"))
end

# --- version 1 ------------------------------------------------------------------

# The two version-1 files in `test/fixtures/`, written by the M7 step-6
# writer (HEAD 67c9153) before the part files replaced it, unfiltered and
# with Shuffle + Deflate(1), from an environment without a project file
# so that they stay small:
#
#     save_checkpoint(path, fixture_forest(); fieldsets=("u" => u, "w" => w),
#                     application="Fixture" => 1, filters, sync=false,
#                     data=(; t=1//3, chunk=7, tags=["a", "", "c"], name="fixture")) do app
#         write_plain(app, "extra", [1, 2, 3])
#     end
#
# with `u, w = fixture_sets(forest)`. The data are exact in binary, so
# rebuilding them gives the saved bits on any Julia version.
function fixture_forest(; comm=nothing)
    forest = Forest((2, 2); N=4, periodic=(true, false),
                    reflecting=((false, false), (false, true)), comm=comm)
    refine!(forest, forest.leaves[1])
    balance!(forest)
    return forest
end

function fixture_sets(forest)
    u = FieldSet(forest, 2; G=1, parity=[(EvenParity, EvenParity), (OddParity, OddParity)])
    w = FieldSet{Float32x2}(forest, 1; G=1, centering=vertexcentered(2),
                            parity=[(EvenParity, OddParity)])
    fill_by_coordinates!((x, v) -> x[1] + 2x[2] + 3v, u)
    fill_by_coordinates!((x, v) -> Float32x2(x[1] - x[2] / 4), w)
    return u, w
end

const FIXTURES = joinpath(@__DIR__, "fixtures")

@testset "A version-1 file, written before the part files, still loads" begin
    # The failure: a reader of format version 2 that no longer reads the
    # single files every earlier version wrote — those of M9a, and the
    # shared files of M7 step 6 — or reads them differently; and one that
    # refuses a version-1 file without checksums, which are optional there.
    forest = fixture_forest()
    u, w = fixture_sets(forest)
    want(fs) = (v = statevector(fs); gather!(v, fs); bytes(v))
    for name in ("plain", "filtered")
        path = joinpath(FIXTURES, "checkpoint-v1-$name.h5")
        ck = load_checkpoint(app -> read_plain(app, "extra"), path; types=(Float32x2,))
        @test ck.forest.leaves == forest.leaves
        @test ck.forest.reflecting == forest.reflecting
        @test bytes(ck.fieldsets["u"].state) == want(u)
        @test bytes(ck.fieldsets["w"].state) == want(w)
        @test ck.fieldsets["w"].fieldset.parity == w.parity
        @test ck.data == (; t=1 // 3, chunk=7, tags=["a", "", "c"], name="fixture")
        @test ck.result == [1, 2, 3]
        @test (ck.provenance.nranks, ck.provenance.nparts) == (1, 1)
        @test h5open(file -> read_attribute(file["TreeAMR.jl"], "format_version"),
                     path) == 1
    end
    # Edited in a copy outside the fixtures' directory.
    original = joinpath(mktempdir(), "v1.h5")
    cp(joinpath(FIXTURES, "checkpoint-v1-filtered.h5"), original)
    unsummed = edited_copy(original, "unsummed.h5") do file
        delete_attribute(file["TreeAMR.jl/forest"], "leaves_crc32c")
        delete_object(file["TreeAMR.jl/fieldsets/u"], "data_crc32c")
    end
    ck = load_checkpoint(unsummed; fieldsets=("u",))
    @test ck.forest.leaves == forest.leaves
    @test bytes(ck.fieldsets["u"].state) == want(u)
end

@testset "A subset of the field sets loads on its own" begin
    # The failure: a subset that loads other sets too, or cannot be
    # loaded without them; and two sets of different layouts and types
    # over one forest that do not both come back.
    dir = mktempdir()
    path = joinpath(dir, "sets.h5")
    forest = Forest((2, 2); N=4, periodic=(true, false))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    a = FieldSet(forest, 2; G=1)
    b = FieldSet{Float32}(forest, 1; G=0, centering=facecentered(2, 2))
    fill_by_coordinates!((x, v) -> x[1] - v * x[2], a)
    fill_by_coordinates!((x, v) -> x[1] * x[2], b)
    ua, ub = statevector(a), statevector(b)
    gather!(ua, a)
    gather!(ub, b)
    save_checkpoint(path, forest; fieldsets=["a" => (a, ua), "b" => b],
                    application="Sets" => 1)
    only_b = load_checkpoint(path; fieldsets=("b",))
    @test collect(keys(only_b.fieldsets)) == ["b"]
    @test only_b.fieldsets["b"].fieldset.centering == facecentered(2, 2)
    @test bytes(only_b.fieldsets["b"].state) == bytes(ub)
    @test collect(keys(load_checkpoint(path; fieldsets="a").fieldsets)) == ["a"]
    both = load_checkpoint(path)
    @test sort(collect(keys(both.fieldsets))) == ["a", "b"]
    @test bytes(both.fieldsets["a"].state) == bytes(ua)
    @test both.fieldsets["a"].fieldset.forest === both.fieldsets["b"].fieldset.forest
    @test_throws "it holds \"a\", \"b\"" load_checkpoint(path; fieldsets=("a", "c"))
end

@testset "Without HDF5 the checkpoint functions say to load it" begin
    # The failure: a bare `MethodError` for a function that exists but
    # whose implementation is in an extension that is not loaded, which
    # says nothing about HDF5; and a hint that goes on appearing once it
    # is loaded, when a `MethodError` means wrong arguments.
    script = """
        using TreeAMR
        try
            save_checkpoint("x.h5", Forest((2,); N=4); fieldsets=(), application="A" => 1)
        catch err
            showerror(stdout, err)
        end
        """
    out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`, String)
    @test occursin("no method matching save_checkpoint", out)
    @test occursin("run `using HDF5`", out)
    @test occursin("an application that never checkpoints", out)
    loaded = try
        load_checkpoint(42)
    catch err
        sprint(showerror, err)
    end
    @test occursin("no method matching load_checkpoint", loaded)
    @test !occursin("using HDF5", loaded)
end
