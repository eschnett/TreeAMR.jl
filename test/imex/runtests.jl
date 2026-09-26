# TreeAMR under IMEXRungeKutta's explicit tableaus: an opt-in suite.
#
# Not part of `Pkg.test`. IMEXRungeKutta is not in the General registry,
# and the main suite runs on Julia 1.10, where `test/Project.toml` cannot
# name an unregistered package (`[sources]` is 1.11+). This directory is
# an environment of its own that takes IMEXRungeKutta from GitHub `main`
# and TreeAMR from this checkout. Set it up once, then run it — at more
# than one thread, or the by-owner path is a plain loop and the placement
# test has nothing to check:
#
#     julia --project=test/imex -e 'using Pkg; Pkg.instantiate()'
#     julia --project=test/imex -t 4 test/imex/runtests.jl
#
# What it claims (see "Open questions" in `CODE.md`): the stage
# arithmetic can run by owner with the partition TreeAMR's block
# ownership defines, with no change to the result; the wave and Burgers
# studies keep their rates and their conservation under IMEXRungeKutta's
# `RK4` and `SSPRK33`; and a stage limiter's correction never reaches the
# state, so that with a conservative right-hand side the drift of a
# conserved total *is* the step limiter's injection.

using Test, Random, TreeAMR
using TreeAMR: threadchunks
import IMEXRungeKutta as IRK
using KernelAbstractions: @kernel, @index

const TESTDIR = dirname(@__DIR__)
include(joinpath(TESTDIR, "oracles.jl"))
include(joinpath(TESTDIR, "ghost_oracles.jl"))
include(joinpath(TESTDIR, "wave.jl"))        # also brings OrdinaryDiffEq's RK4
include(joinpath(TESTDIR, "burgers.jl"))     # and its SSPRK33

@info "TreeAMR with IMEXRungeKutta $(pkgversion(IRK)), $(Threads.nthreads()) thread(s)"

"""
    state_partition(fs::FieldSet) -> Vector{UnitRange{Int}}

The ownership partition of `fs`'s state vector: element `c` holds the
entries of the blocks in chunk `c` of `threadchunks(nblocks(fs))`, which
is the chunk `launch_by_owner!` and `threaded_chunks` run on
default-pool thread `c`. One element per thread, padded with empty
ranges, as IMEXRungeKutta's `partition` keyword takes it.

A candidate for TreeAMR's own API (the open question of the same name in
both packages' `CODE.md`), kept here until it is named.
"""
function state_partition(fs::FieldSet{T,D}) where {T,D}
    L = fs.forest.N^D * fs.nvars
    parts = UnitRange{Int}[(first(r) - 1) * L + 1:last(r) * L
                           for r in threadchunks(nblocks(fs))]
    while length(parts) < Threads.nthreads()
        push!(parts, 1:0)
    end
    return parts
end

# One fixed-step IMEXRungeKutta solve of `nsteps` steps, returning the
# final state. `solve_imp! = nothing`: an explicit tableau makes no stage
# solve.
imex_solve(f!, u, t0, t1, nsteps, p, tab; kw...) =
    IRK.solve(IRK.IMEXProblem(f!, nothing, u, (t0, t1), p), tab;
              dt=(t1 - t0) / nsteps, kw...).u

# The M8a wave study, through IMEXRungeKutta rather than OrdinaryDiffEq.
function imex_wave_errors(::Val{D}; N, ops, G=1, L=1.0, m=1, cfl=0.25,
                          periods=0.25, partition=:owner) where {D}
    forest = wave_forest(Val(D), N; L=L)
    fs = FieldSet(forest, 2; G=G, centering=vertexcentered(D))
    problem = WaveProblem(fs, GhostSchedule(fs, ops))
    fill_by_coordinates!(wave_exact(D, L, m, 0.0), fs)
    u0 = statevector(fs); gather!(u0, fs)
    h = minimum_spacing(forest)
    t_end = periods * 2π / wave_omega(D, L, m)
    nsteps = ceil(Int, t_end / (cfl * h))
    part = partition === :owner ? state_partition(fs) : partition
    u = imex_solve(wave_rhs!, u0, 0.0, t_end, nsteps, problem, IRK.RK4();
                   partition=part)
    final = FieldSet(forest, 2; G=G, centering=vertexcentered(D))
    fill_by_coordinates!(wave_exact(D, L, m, t_end), final)
    uexact = statevector(final); gather!(uexact, final)
    return (l2=volume_weighted_norm(fs, u .- uexact), h=h)
end

state_mass(fs::FieldSet{T,D}, u) where {T,D} =
    mesh_mapreduce(identity, +, zero(T), fs, u;
                   weight=key -> spacing(T, fs.forest, key)^D)

@kernel function owner_thread_kernel!(tid)
    I = @index(Global, NTuple)
    tid[I[end]] = Threads.threadid()
end

@testset "The partition gives each state entry to the thread that runs its block" begin
    # The failure mode is a partition that covers the state exactly — so
    # IMEXRungeKutta accepts it and every result is still right — but
    # hands a block's entries to another thread than the one
    # `map_blocks!` runs that block on, which is the cross-core migration
    # the ownership policy exists to remove, and invisible in any value.
    forest = wave_forest(Val(2), 8; roots=5)       # 49 blocks: chunks of unequal size
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(2))
    part = state_partition(fs)
    @test length(part) == Threads.nthreads()
    @test reduce(vcat, collect.(part)) == 1:statelength(fs)
    tid = zeros(Int, nblocks(fs))
    map_blocks!(owner_thread_kernel!, fs, tid)
    offset = Threads.threadpoolsize(:interactive)
    L = fs.forest.N^2 * fs.nvars
    for (c, r) in enumerate(part), b in unique(cld.(r, L))
        # IMEXRungeKutta places range `c` on default-pool thread `c`, as
        # TreeAMR's `threaded_chunks` does; one thread runs everything
        # on the calling task, whichever thread that is.
        Threads.nthreads() > 1 && @test tid[b] == offset + c
    end
end

@testset "RK4 by owner is bitwise the broadcast, and OrdinaryDiffEq's RK4 to roundoff: D=$D" for
        D in (1, 2, 3)
    # Guards a stage arithmetic that differs by path (a partition that
    # drops, doubles or reorders an entry, or a fold that reassociates),
    # and a tableau or abscissa that differs from the method the rest of
    # the suite measured its rates with.
    N = D == 3 ? 8 : 16
    forest = wave_forest(Val(D), N)
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(D))
    problem = WaveProblem(fs, GhostSchedule(fs, Operators(prolongation=4,
                                                          restriction=4)))
    fill_by_coordinates!(wave_exact(D, 1.0, 1, 0.0), fs)
    u0 = statevector(fs); gather!(u0, fs)
    dt = 0.25 * minimum_spacing(forest)
    nsteps = 16
    owner = imex_solve(wave_rhs!, u0, 0.0, nsteps * dt, nsteps, problem, IRK.RK4();
                       partition=state_partition(fs))
    bcast = imex_solve(wave_rhs!, u0, 0.0, nsteps * dt, nsteps, problem, IRK.RK4())
    ode = solve(ODEProblem(wave_rhs!, u0, (0.0, nsteps * dt), problem), RK4();
                dt=dt, adaptive=false, save_everystep=false).u[end]
    @test owner == bcast
    # The two sum the stages in a different order; the Laplacian's `1/h²`
    # amplifies that difference, so it is a few hundred ulp and not one.
    err = maximum(abs, owner .- ode) / maximum(abs, ode)
    @info "D = $D: |IMEX − ODE| / |u| = $err after $nsteps steps"
    @test err ≤ 1e-11
end

@testset "The wave equation converges at second order through IMEXRungeKutta's RK4: D=$D" for
        D in (1, 2)
    # The M8a rate (1.99 at order-4 operators, G = 1) must not depend on
    # which package forms the stages. A mistimed stage or a wrong weight
    # would drop it to 1 or below.
    ops = Operators(prolongation=4, restriction=4)
    rs = [imex_wave_errors(Val(D); N=N, ops=ops) for N in (8, 16, 32)]
    rates = [log2(rs[i].l2 / rs[i + 1].l2) for i in 1:2]
    @info "D = $D: L2 errors $(getfield.(rs, :l2)), rates $rates"
    @test all(r -> abs(r - 2) ≤ 0.15, rates)
end

@testset "Burgers conserves mass to roundoff through IMEXRungeKutta's SSPRK33, by owner: D=$D" for
        D in (1, 2)
    # M8b's claim on a static two-level mesh, past the breaking time: a
    # shock crosses the coarse-fine faces. It holds for any Runge–Kutta
    # method, because every stage's `du` sums to zero, so a drift beyond
    # an ulp would mean the update is not the tableau's convex sum of
    # tendencies. The control without the fixup shows the faces are hit.
    T = Float64
    ops = Operators(family=Conservative, prolongation=3, restriction=2)
    run(fixup) = begin
        forest = burgers_forest(Val(D), 8)
        state = FieldSet(forest, 1; G=2, centering=cellcentered(D))
        fill_burgers_averages!(state, 1.0, 1.0, 0.5)
        p = BurgersProblem(state, ops; limiter=:minmod, fixup=fixup)
        u0 = statevector(state); gather!(u0, state)
        t1 = 1.5 * burgers_breaktime(D, 1.0, 0.5)
        nsteps = ceil(Int, t1 / burgers_dt(forest, 0.5, 1.5, D))
        u = imex_solve(burgers_rhs!, u0, 0.0, t1, nsteps, p, IRK.SSPRK33();
                       partition=state_partition(state))
        drift = abs(state_mass(state, u) - state_mass(state, u0))
        (; drift, scale=burgers_mass_scale(state), nsteps)
    end
    r, control = run(true), run(false)
    @info "D = $D: drift $(r.drift) (scale $(r.scale), $(r.nsteps) steps); " *
          "without the fixup $(control.drift)"
    @test r.drift ≤ 8 * eps(T) * r.scale
    @test control.drift > 1e8 * r.drift
end

@testset "Only the step limiter's correction reaches the conserved total: D=$D" for
        D in (1, 2)
    # IMEXRungeKutta limits a stage value only for `f_exp!` to read and
    # builds the next stage from `uⁿ` and the tendencies, so a stage
    # correction never enters the state. With a conservative right-hand
    # side the mass drift is then exactly the step limiter's injection —
    # the equality TreeHydro measured only under its per-step cadence.
    # Guards an integrator that folds a stage correction into the state
    # (OrdinaryDiffEq's Shu–Osher SSPRK33 does, the control below), and a
    # limiter that is installed nowhere (the injections must be nonzero).
    #
    # The limiter caps `u` at 1.4, below the initial maximum 1.5, and the
    # reconstruction is unlimited, so it overshoots near the flat top and
    # the cap keeps firing. The initial data are capped outside the
    # integrator, as an application caps its initial data and its regrid
    # output.
    T = Float64
    cap = 1.4
    ops = Operators(family=Conservative, prolongation=3, restriction=2)
    forest = burgers_forest(Val(D), 8)
    state = FieldSet(forest, 1; G=2, centering=cellcentered(D))
    fill_burgers_averages!(state, 1.0, 1.0, 0.5)
    p = BurgersProblem(state, ops; limiter=:none)
    u0 = statevector(state); gather!(u0, state)
    u0 .= min.(u0, cap)
    injected = Dict(:stage => 0.0, :step => 0.0, :calls => 0)
    capper(kind) = (u, integ, p, t) -> begin
        m = state_mass(state, u)
        u .= min.(u, cap)
        injected[kind] += state_mass(state, u) - m
        injected[:calls] += 1
        return nothing
    end
    t1 = 0.5 * burgers_breaktime(D, 1.0, 0.5)
    nsteps = ceil(Int, t1 / burgers_dt(forest, 0.5, 1.5, D))
    u = imex_solve(burgers_rhs!, u0, 0.0, t1, nsteps, p, IRK.SSPRK33();
                   stage_limiter=capper(:stage), step_limiter=capper(:step),
                   partition=state_partition(state))
    drift = state_mass(state, u) - state_mass(state, u0)
    scale = burgers_mass_scale(state)
    @info "D = $D, IMEXRungeKutta: drift $drift, step injection $(injected[:step]), " *
          "stage injection $(injected[:stage]) over $(injected[:calls]) calls"
    @test maximum(u) ≤ cap
    @test injected[:stage] < -1e-6 * scale && injected[:step] < -1e-6 * scale
    @test abs(drift - injected[:step]) ≤ 8 * eps(T) * scale * nsteps

    # The control: the same limiters in OrdinaryDiffEq's SSPRK33, whose
    # Shu–Osher stages carry the capped value forward. Its drift is no
    # longer its step limiter's injection.
    injected[:stage] = injected[:step] = 0.0
    v = solve(ODEProblem(burgers_rhs!, u0, (0.0, t1), p), SSPRK33();
              dt=t1 / nsteps, adaptive=false, save_everystep=false,
              stage_limiter=capper(:stage), step_limiter=capper(:step)).u[end]
    odrift = state_mass(state, v) - state_mass(state, u0)
    @info "D = $D, OrdinaryDiffEq: drift $odrift, step injection $(injected[:step]), " *
          "stage injection $(injected[:stage])"
    @test abs(odrift - injected[:step]) > 1e6 * 8 * eps(T) * scale * nsteps
end
