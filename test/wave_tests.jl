# M8a: the scalar wave equation, **vertex-centered**.
#
# From M8 on this is the wave study: the values sit on the cell
# boundaries, so a block owns its points 0 … N-1 and the plane at N is
# shared with its high-side neighbor and exchange-filled exactly as a
# ghost is. The M3 study is kept beside it in `wave_cell_tests.jl` with
# `centering = cellcentered(D)` said out loud, so the M3 numbers stay
# under test.
#
# Nothing in the application changed. `wave_rhs_kernel!` takes no
# `Val(C)`: it reads its own point and its two neighbours a spacing
# away, which is the same stencil wherever those points sit. What
# changes is what the mesh does at a coarse-fine interface, and with it
# two predictions from the operator table under "Operators" in CODE.md:
#
#   * **Restriction along a stagger is injection** — the coincident fine
#     point copied, exact for arbitrary data — so the restriction order
#     does not enter the global rate at all. M3's "raising one operator
#     alone does not help" becomes "only the prolongation order
#     matters", and that is asserted here bit for bit rather than as a
#     rate.
#   * **Prolongation reaches `p/2 - 1` planes past the shared plane**,
#     against `p/2` past an interface, so `G = 1` suffices at order 4
#     where cell centering needs 2.
#
# The interface-order rule itself is unchanged: an order-`p` ghost
# carries an `O(h^p)` error and the 2nd-order Laplacian divides it by
# `h²`, so the global rate is `min(2, p - 1)` — 1 at `p = 2`, 2 at
# `p = 4`.

@testset "Vertex-centered wave equation on a uniform grid: D=$D" for D in (1, 2)
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

@testset "Vertex-centered wave equation on a two-level mesh: D=$D" for D in (1, 2)
    # The M3 claim on a staggered layout, at the ghost width the vertex
    # row of the operator table allows: order-4 prolongation reaches one
    # plane past the source's shared plane, and the Laplacian reaches one
    # point past the owned range, so G = 1 is enough for both.
    ops = Operators(prolongation=4, restriction=4)
    hs, l2, linf = Float64[], Float64[], Float64[]
    for N in (8, 16, 32)
        r = wave_errors(Val(D); N=N, G=1, ops=ops)
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

@testset "Only prolongation limits the vertex rate: D=$D" for D in (1,)
    # The cell-centered study needs *both* orders raised, because each
    # side of the interface gets its ghosts from a different operator.
    # Along a stagger the restriction side is injection — a coincident
    # point, exact for any data, with no order to raise — so the whole
    # rate is the prolongation's: 1 at order 2, 2 at order 4, whatever
    # the restriction order says.
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
    @test rate(Operators(prolongation=2, restriction=4), 1) ≈ 1.0 atol = 0.2
    @test rate(Operators(prolongation=4, restriction=2), 1) ≈ 2.0 atol = 0.15
    @test rate(Operators(prolongation=4, restriction=4), 1) ≈ 2.0 atol = 0.15
end

@testset "The restriction order is inert along a stagger: D=$D" for D in (1, 2)
    # Stronger than the rates above, and the reason they come in pairs:
    # the restriction stencil in a vertex-like dimension is width 1 with
    # weight 1 *regardless of the order asked for*, so the two runs are
    # not merely equally accurate, they are the same computation. A
    # future restriction builder that quietly used `p` along a stagger
    # would break this long before it moved a rate.
    for p in (2, 4)
        a = wave_errors(Val(D); N=16, G=1, ops=Operators(prolongation=p, restriction=2))
        b = wave_errors(Val(D); N=16, G=1, ops=Operators(prolongation=p, restriction=4))
        @test a.l2 === b.l2
        @test a.linf === b.linf
    end
end

@testset "G = 1 is the whole requirement at order 4: D=$D" for D in (1, 2)
    # The vertex row of the operator table says G >= p/2 - 1, so a
    # second ghost plane buys nothing at order 4 — not "almost nothing",
    # nothing: the same stencils read the same points. Cell centering
    # needs G >= p/2 and refuses G = 1 outright, which is what makes the
    # relaxation worth a test rather than a remark.
    ops = Operators(prolongation=4, restriction=4)
    one_ghost = wave_errors(Val(D); N=16, G=1, ops=ops)
    two_ghosts = wave_errors(Val(D); N=16, G=2, ops=ops)
    @test one_ghost.l2 === two_ghosts.l2
    @test one_ghost.linf === two_ghosts.linf

    @test_throws "G >= 2" wave_errors(Val(D); N=16, G=1, ops=ops,
                                      centering=cellcentered(D))
end

@testset "Vertex-centered wave equation in 3D" begin
    # Smoke test only: 3D convergence runs are expensive, so this checks
    # that the same code path works and that the solution stays sane.
    r = wave_errors(Val(3); N=8, G=1, ops=Operators(prolongation=4, restriction=4))
    @test r.nblocks > 8
    @test isfinite(r.l2)
    @test r.l2 < 0.05
    @test r.linf < 0.2
end

@testset "A moving refined region tracks a propagating pulse" begin
    # M4's claim, on the staggered layout: "without artifacts" means the
    # adaptive run matches the *uniformly finest* mesh, so a moving
    # coarse-fine interface that reflected or smeared the pulse would
    # show up as an error above it.
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
    # improves the run over the same criterion with no margin at all,
    # and a buffer of a few *cells* does that work too, because dilation
    # is keyed on the reported box rather than on the Refine flag (M4).
    @test amr.worst < unbuffered.worst
    @test unbuffered.tracking == 1.0                  # the criterion alone still tracks
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
    # methods like RK4. Asserted on the staggered layout because the
    # working array has a plane there that the state vector does not —
    # the shared boundary plane — and scatter/gather must still be
    # inverse to each other over the owned range alone.
    forest = wave_forest(Val(1), 8)
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(1))
    problem = WaveProblem(fs, GhostSchedule(fs, Operators(prolongation=4,
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

    # The state vector is N^D per variable per block for every
    # centering, so its shape has not moved either.
    @test length(u) == forest.N * 2 * nblocks(fs)

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
    forest = wave_forest(Val(D), 16)
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(D))
    problem = WaveProblem(fs, GhostSchedule(fs, Operators(prolongation=4,
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
