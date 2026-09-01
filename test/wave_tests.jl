# M3: the scalar wave equation driven by OrdinaryDiffEq.
#
# The acceptance criterion is that volume-weighted L2/L∞ errors against
# the exact sine-mode solution converge at 2nd order.

@testset "Wave equation on a uniform grid: D=$D" for D in (1, 2)
    # Control: with no coarse-fine interfaces the 2nd-order Laplacian and
    # fixed-step RK4 must give a clean 2nd-order rate. Anything the
    # refined runs below lose is then attributable to the interface.
    hs, l2 = Float64[], Float64[]
    for N in (8, 16, 32)
        r = wave_errors(Val(D); N=N, G=1, refined=false)
        push!(hs, r.h)
        push!(l2, r.l2)
    end
    @test all(l2[i] > l2[i + 1] for i in 1:(length(l2) - 1))
    @test convergence_rate(hs, l2) ≈ 2.0 atol = 0.15
end

@testset "Wave equation on a two-level mesh: D=$D" for D in (1, 2)
    # Interpolation order must exceed the differencing order by two:
    # a prolongated ghost carries an O(h^p) error, and the 2nd-order
    # Laplacian divides it by h², leaving an O(h^(p-2)) truncation error
    # along the coarse-fine interface. With p = 4 that is O(h²) and does
    # not pollute the interior scheme.
    ops = Operators(prolongation=4, restriction=4)
    hs, l2, linf = Float64[], Float64[], Float64[]
    for N in (8, 16, 32)
        r = wave_errors(Val(D); N=N, G=2, ops=ops)
        push!(hs, r.h)
        push!(l2, r.l2)
        push!(linf, r.linf)
        @test r.nblocks > 2^D                     # refinement really happened
        @test isfinite(r.l2)
    end

    @test all(l2[i] > l2[i + 1] for i in 1:(length(l2) - 1))
    @test all(linf[i] > linf[i + 1] for i in 1:(length(linf) - 1))
    @test convergence_rate(hs, l2) ≈ 2.0 atol = 0.15
    @test convergence_rate(hs, linf) ≈ 2.0 atol = 0.2
end

@testset "Interface order limits the global rate: D=$D" for D in (1,)
    # Documents the mechanism above, and guards it: with order-2
    # operators the interface error is O(1) and drags the global rate
    # down to first order, even though the interior scheme is 2nd order.
    # Both operators matter -- raising only one leaves the other side of
    # the interface first order.
    rate(ops, G) = begin
        hs, l2 = Float64[], Float64[]
        for N in (8, 16, 32)
            r = wave_errors(Val(D); N=N, G=G, ops=ops)
            push!(hs, r.h)
            push!(l2, r.l2)
        end
        convergence_rate(hs, l2)
    end

    @test rate(Operators(prolongation=2, restriction=2), 1) ≈ 1.0 atol = 0.2
    @test rate(Operators(prolongation=4, restriction=2), 2) < 1.5
    @test rate(Operators(prolongation=2, restriction=4), 2) < 1.5
    @test rate(Operators(prolongation=4, restriction=4), 2) ≈ 2.0 atol = 0.15
end

@testset "Wave equation in 3D" begin
    # Smoke test only: 3D convergence runs are expensive, so this checks
    # that the same code path works and that the solution stays sane.
    r = wave_errors(Val(3); N=8, G=2, ops=Operators(prolongation=4, restriction=4))
    @test r.nblocks > 8
    @test isfinite(r.l2)
    @test r.l2 < 0.05
    @test r.linf < 0.2
end

@testset "A moving refined region tracks a propagating pulse" begin
    # The measure of "without artifacts" is that the adaptive run matches
    # the *uniformly finest* mesh: if the moving coarse-fine interface
    # were reflecting or smearing the pulse, the error would exceed it.
    σ = 0.08
    coarse = uniform_pulse(Val(1); roots=8, N=8, σ=σ)     # level-0 equivalent
    fine = uniform_pulse(Val(1); roots=8, N=32, σ=σ)      # level-2 equivalent
    # A tight flagging threshold -- 0.05 of the peak, not 1e-3 far down
    # the tail -- with buffers of one block width, of a few cells, and of
    # nothing at all.
    amr = track_pulse(Val(1); roots=8, N=8, σ=σ, threshold=0.05, buffer=8, chunk=0.02)
    narrow = track_pulse(Val(1); roots=8, N=8, σ=σ, threshold=0.05, buffer=4, chunk=0.02)
    unbuffered = track_pulse(Val(1); roots=8, N=8, σ=σ, threshold=0.05, buffer=0,
                             chunk=0.02)

    # Refinement is worth having at all: the coarse mesh is far worse.
    @test coarse.err > 10 * fine.err

    # The refined region never lets the pulse peak escape onto a coarse
    # block over the whole run.
    @test amr.tracking == 1.0
    @test amr.maxlevel == 2

    # The buffer did real work: refining ahead of the pulse measurably
    # improves the run over the same criterion with no margin at all.
    @test amr.worst < unbuffered.worst
    @test unbuffered.tracking == 1.0                  # the criterion alone still tracks

    # And a buffer of a *few cells* now does that work too, which is why
    # dilation is keyed on the reported box rather than on the Refine
    # flag. Under the Refine-keyed rule this run was bit-identical to the
    # unbuffered one: a block ahead of the pulse refined the moment the
    # criterion fired anywhere in it, so its box hugged the face the
    # pulse entered through and only a dilation of most of a block width
    # reached out the far face. Now the block that holds the pulse is
    # already at the target level and reports (Keep, box), so it carries
    # an equal-level margin along with it and the width is free to be
    # what the physics asks for -- feature speed times regrid cadence.
    @test narrow.worst < unbuffered.worst
    @test narrow.tracking == 1.0
    @test narrow.maxlevel == 2

    # And the adaptive run is as accurate as the uniform fine one ...
    @test amr.worst ≈ fine.err rtol = 0.1
    @test amr.worst < coarse.err / 5
    # ... for fewer cells, which is the point of doing this at all.
    @test amr.nblocks * 8 < fine.cells
end

@testset "RHS does not mutate the state vector" begin
    # The integrator's `u` is authoritative; the working array is
    # scratch. A RHS that wrote back into `u` would corrupt multi-stage
    # methods like RK4.
    forest = wave_forest(Val(1), 8, 2)
    fs = FieldSet(forest, 2)
    problem = WaveProblem(fs, GhostSchedule(forest, Operators(prolongation=4,
                                                              restriction=4)))
    fill_by_coordinates!(wave_exact(1, 1.0, 1, 0.0), fs)
    u = statevector(fs)
    gather!(u, fs)

    before = copy(u)
    du = statevector(fs)
    wave_rhs!(du, u, problem, 0.0)
    @test u == before
    @test !all(iszero, du)
    @test all(isfinite, du)

    # ∂ₜu = v, so the first half of du is exactly the second field.
    state, dstate = statearray(u, fs), statearray(du, fs)
    @test dstate[:, 1, :] == state[:, 2, :]

    # Re-evaluating gives the same answer: the RHS is a pure function of
    # (u, t), with no state carried in the working array between calls.
    du2 = statevector(fs)
    wave_rhs!(du2, u, problem, 0.0)
    @test du2 == du
end

@testset "Energy stays bounded" begin
    # A standing mode neither grows nor decays; a wrong interface
    # treatment usually shows up as slow drift long before it shows up
    # as an outright instability.
    D, L, m = 1, 1.0, 1
    forest = wave_forest(Val(D), 16, 2)
    fs = FieldSet(forest, 2)
    problem = WaveProblem(fs, GhostSchedule(forest, Operators(prolongation=4,
                                                              restriction=4)))
    fill_by_coordinates!(wave_exact(D, L, m, 0.0), fs)
    u0 = statevector(fs)
    gather!(u0, fs)

    t_end = 4 * 2π / wave_omega(D, L, m)          # four full periods
    dt = 0.25 * minimum_spacing(forest)
    nsteps = ceil(Int, t_end / dt)
    sol = solve(ODEProblem(wave_rhs!, u0, (0.0, t_end), problem), RK4();
                dt=t_end / nsteps, adaptive=false, save_everystep=false)

    @test all(isfinite, sol.u[end])
    amplitude(u) = volume_weighted_norm(fs, u; p=Inf)
    @test amplitude(sol.u[end]) ≈ amplitude(u0) rtol = 0.05
end
