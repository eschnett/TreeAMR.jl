# M8b: Burgers' equation, the acceptance test for conservation at
# coarse-fine faces.
#
# The wave equation (M3, M8a) is not conservative and does not need to
# be; Burgers is the problem that makes the three-step right-hand side
# of CODE.md's "Application interface" mean something. Two claims, in
# the order the M8 milestone states them:
#
#   * **Total mass is conserved to roundoff**, with a shock crossing a
#     refined region that follows it and regrids in between. The
#     negative control -- the identical run with `restrict_interfaces!`
#     skipped -- is what makes the claim a measurement rather than a
#     tautology.
#   * **The interface-order rule carries over to the conservative
#     family**: a flux divergence takes one derivative, not two, so `p`
#     must exceed the scheme's order by one rather than by two, and the
#     odd conservative orders 1, 3, 5 give rates 1, 2, 2 under a
#     second-order finite-volume scheme.
#
# The second one is measured in the **maximum** norm, and that is a
# finding rather than a detail (recorded in CODE.md): the defect an
# order-`p` prolongation leaves is confined to the coarse-fine
# interface, whose measure shrinks with `h`, so a volume-weighted L1
# norm converges at the scheme's own rate for every `p` and only `L∞`
# exposes the rule. M3's wave study saw it in L2 because a
# second-derivative stencil spreads the interface error over the whole
# domain; a flux divergence does not.

@testset "Same-level faces conserve without any fixup: D=$D" for D in (1, 2)
    # The control that localizes the claim. On a uniform mesh every face
    # is a same-level face, and the two blocks sharing one compute its
    # flux from the same four cell values — their own and their ghosts,
    # which are bit-for-bit copies — with the same kernel. So mass is
    # already conserved to roundoff there, `restrict_interfaces!` or not,
    # and everything the fixup is measured on below is attributable to
    # the coarse-fine faces alone.
    #
    # This is also what pins the state's `G = 2`: the reconstruction at a
    # block's own boundary face reads cells `i-2 … i+1`, and with one
    # ghost the two sides would read different numbers there.
    T = Float64
    for fixup in (true, false)
        r = uniform_shock(Val(D); roots=(D == 1 ? 8 : 4), N=8, grid=(D == 1 ? 64 : 32),
                          t_end=0.2, fixup=fixup)
        @test r.mass ≈ 1.0 atol = 256 * eps(T)
    end
end

@testset "The fixup is what conserves at a coarse-fine face: D=$D" for D in (1, 2, 3)
    # The claim in its cheapest, most direct form, and the only one that
    # runs in 3D: the M3 static two-level mesh, a smooth solution short
    # of the breaking time, no regridding and no limiter — so the fixup
    # is the *single* difference between the two runs here. In 3D a
    # coarse-fine face carries four fine faces, so this is also where the
    # tangential stencil is a 2x2 average rather than a single cell.
    #
    # The adaptive shock below is the milestone's acceptance test; this
    # is the one that says which line of the right-hand side is
    # responsible.
    T = Float64
    ops = Operators(family=Conservative, prolongation=3, restriction=2)
    N, roots = D == 3 ? (4, 4) : (8, 4)
    common = (; N=N, roots=roots, ops=ops, fraction=0.8, limiter=:none)
    r = burgers_errors(Val(D); common...)
    control = burgers_errors(Val(D); common..., fixup=false)

    @test r.nblocks > roots^D                      # the mesh really is two-level
    @test isfinite(r.l1)
    @test r.drift <= 8 * eps(T) * r.scale * r.nsteps
    @test control.drift > 1e8 * eps(T) * control.scale * control.nsteps
    @test control.drift > 1e-5 * control.scale
end

@testset "Total mass is conserved to roundoff across coarse-fine faces: D=$D" for
        D in (1, 2)
    # The M8b acceptance test. A sine steepens into a shock, the refined
    # region follows it, the mesh is rebuilt between chunks, and the
    # domain integral must not move by more than roundoff — which is what
    # the flux fixup buys and nothing else does.
    T = Float64
    # The criterion is tuned per dimension so that the mesh comes out
    # genuinely *mixed*. `s = x + y` steepens twice as fast in 2D, and a
    # buffer of a whole block width there refines the entire domain —
    # which would leave no coarse-fine face at all and make the negative
    # control below drift by nothing, passing every assertion for the
    # wrong reason.
    common = D == 1 ?
        (; N=8, roots=8, t_end=0.6, chunk=0.05, threshold=0.15, buffer=8) :
        (; N=8, roots=4, t_end=0.4, chunk=0.05, threshold=0.3, buffer=4)
    r = track_shock(Val(D); common..., maxlevel_wanted=1)
    control = track_shock(Val(D); common..., maxlevel_wanted=1, fixup=false)

    # The mesh really did what the test claims it did: two levels, so
    # there are coarse-fine faces, and the refined half followed the
    # shock.
    @test r.maxlevel == 1
    @test sort(unique(level.(r.forest.leaves))) == [0, 1]
    @test r.tracking == 1.0
    @test (r.nblocks, r.maxlevel) == (control.nblocks, control.maxlevel)
    @test r.nsteps == control.nsteps             # the two runs differ in one step only

    # `c · eps(T) · Σ hᴰ|u| · nsteps`, CODE.md's form. Measured far
    # inside it: the drift is one or two ulp of the total mass, whatever
    # the step count.
    @test r.drift <= 8 * eps(T) * r.scale * r.nsteps
    @test r.drift <= 8 * eps(T) * r.scale

    # The negative control: the same run without step (ii) leaks at the
    # interfaces by orders of magnitude more. Without this the assertion
    # above would pass on a scheme that never saw a coarse-fine face.
    @test control.drift > 1e8 * r.drift
    @test control.drift > 1e-5 * control.scale
end

@testset "Conservation survives Float32" begin
    # The bound is relative to the precision, not an absolute constant:
    # the fixup adds no arithmetic, only target ranges, so the drift is
    # still an ulp of the total mass. The control's drift, by contrast,
    # is a *discretization* error and therefore the same number in every
    # precision — which is what shrinks the separation to a couple of
    # hundred rather than the thirteen orders of magnitude Float64 shows.
    for T in (Float32, Float64)
        common = (; N=8, roots=8, t_end=0.4, chunk=0.05, maxlevel_wanted=1,
                  threshold=0.15, buffer=8, T=T)
        r = track_shock(Val(1); common...)
        control = track_shock(Val(1); common..., fixup=false)
        @test r.maxlevel == 1
        @test r.drift <= 8 * eps(T) * r.scale * r.nsteps
        @test control.drift > 100 * r.drift
    end
end

@testset "The interface-order rule for the conservative family: D=$D" for D in (1, 2)
    # CODE.md's prediction, measured. A flux divergence divides an
    # `O(hᵖ)` ghost error by `h` once, so the interface caps the global
    # rate at `p - 1 + 1 = p`; against a second-order scheme that is 1 at
    # `p = 1` and 2 from `p = 3` on, and raising `p` further buys
    # nothing. The restriction order does not appear because conservative
    # restriction is the exact volume average, exact for any field.
    #
    # Measured in L∞. The defect sits at the interface and nowhere else,
    # so the volume-weighted L1 norm — which weights it by the shrinking
    # measure of the region it occupies — converges at the scheme's own
    # rate whatever `p` is; that is asserted below too, because it is the
    # part of this that is easy to get wrong by picking a norm and
    # believing it. The defect stays local *because the fixup makes it
    # zero-mean*: a first-order hyperbolic operator carries a zero-mean
    # residual nowhere, but it carries a leak downstream as an O(h)
    # plateau, so without the fixup the L1 rate at order 1 falls to first
    # order too. Predicted before it was run; asserted at the end.
    Ns = D == 1 ? (8, 16, 32, 64) : (8, 16, 32)
    unlimited = :none                            # a limiter clips at smooth extrema

    rates(ops; refined=true, fixup=true) = begin
        hs, l1, linf = Float64[], Float64[], Float64[]
        for N in Ns
            r = burgers_errors(Val(D); N=N, ops=ops, roots=4, refined=refined,
                               limiter=unlimited, fixup=fixup)
            push!(hs, r.h)
            push!(l1, r.l1)
            push!(linf, r.linf)
            @test isfinite(r.l1)
            if refined
                @test r.nblocks > 4^D             # refinement really happened
            end
        end
        @test all(linf[i] > linf[i + 1] for i in 1:(length(linf) - 1))
        (linf=convergence_rate(hs, linf), l1=convergence_rate(hs, l1))
    end

    ops(p) = Operators(family=Conservative, prolongation=p, restriction=2)

    # The unrefined control: with no coarse-fine interface the scheme is
    # second order, so anything the refined runs lose below is the
    # interface's doing and not the scheme's.
    control = rates(ops(3); refined=false)
    @test control.linf > 1.65

    first_order = rates(ops(1))
    @test first_order.linf ≈ 1.0 atol = 0.25

    for p in (3, 5)
        r = rates(ops(p))
        # Not "second order" in the abstract: the *same* rate as the
        # unrefined control, which is the claim that the interface has
        # stopped being what limits it.
        @test r.linf ≈ control.linf atol = 0.15
        @test r.linf > 1.65
        # And the L1 rate never sees any of this.
        @test r.l1 > 1.65
    end
    @test first_order.l1 > 1.65

    # The negative control on the *rate*: the same order-1 run without
    # the fixup leaks O(h) of mass per unit time at each interface, and
    # the leak is transported downstream, so the L1 norm now sees the
    # interface — measured 1.12 in D = 1 and 1.25 in D = 2 against the
    # scheme's own 1.98 and 1.80 with the fixup — while L∞ never depended
    # on the fixup at all. This is what pins the localization on
    # conservation rather than on the flux-divergence form.
    leaky = rates(ops(1); fixup=false)
    @test leaky.l1 < 1.45
    @test leaky.l1 < first_order.l1 - 0.4
    @test leaky.linf ≈ 1.0 atol = 0.25
end

@testset "A tracked shock matches the uniformly fine reference" begin
    # M4's claim on a conservative scheme: "tracks the shock without
    # artifacts" means the adaptive run agrees with the *uniformly finest*
    # mesh — the one at its own finest level — at fewer cells, while the
    # uniform mesh at its coarsest level does not.
    #
    # The two runs live on different meshes, so they are compared where
    # they can be: reduced onto a common uniform grid by exact volume
    # averaging (`reduce_to_grid`), which knows positions and spacings
    # and nothing about how either run produced its numbers.
    t_end = 0.6
    coarse = uniform_shock(Val(1); roots=8, N=8, grid=64, t_end=t_end)
    fine = uniform_shock(Val(1); roots=8, N=16, grid=64, t_end=t_end)
    amr = track_shock(Val(1); N=8, roots=8, grid=64, t_end=t_end, chunk=0.05,
                      maxlevel_wanted=1, threshold=0.15, buffer=8)
    unbuffered = track_shock(Val(1); N=8, roots=8, grid=64, t_end=t_end, chunk=0.05,
                             maxlevel_wanted=1, threshold=0.15, buffer=0)

    # The adaptive run's finest spacing is the uniform fine run's, so
    # these two resolve the shock equally well ...
    @test l1_difference(amr.reduced, fine.reduced) < 1e-3
    # ... and far better than the coarse mesh does, which is what says
    # the refinement was worth having.
    @test l1_difference(coarse.reduced, fine.reduced) >
          5 * l1_difference(amr.reduced, fine.reduced)
    # ... for fewer cells, which is the point of doing this at all.
    @test amr.cells < fine.cells

    # The travelling margin does real work here as it did for the M4
    # pulse: the shock moves about six fine cells per chunk, and with no
    # buffer at all the refined region falls behind it.
    @test amr.tracking == 1.0
    @test unbuffered.tracking < 1.0
    @test l1_difference(amr.reduced, fine.reduced) <
          l1_difference(unbuffered.reduced, fine.reduced)
end

@testset "The conservative RHS is a pure function of the state" begin
    # The integrator's `u` is authoritative; the working arrays — the
    # state's and all `D` flux sets' — are scratch. A right-hand side
    # that wrote back into `u`, or that carried state between calls,
    # would corrupt a multi-stage method, and SSPRK33 is one.
    #
    # Asserted with the fixup on, because the fixup is the one operation
    # in the package that overwrites a block's own computed values: if it
    # depended on what was in the flux arrays beforehand, the second
    # evaluation would differ from the first.
    D = 2
    forest = burgers_forest(Val(D), 8; roots=4)
    state = FieldSet(forest, 1; G=2)
    p = BurgersProblem(state, Operators(family=Conservative, prolongation=3,
                                        restriction=2))
    fill_burgers_averages!(state, 1.0, 1.0, 0.5)
    u = statevector(state)
    gather!(u, state)

    before = copy(u)
    du = statevector(state)
    burgers_rhs!(du, u, p, 0.0)
    @test u == before
    @test !all(iszero, du)
    @test all(isfinite, du)

    du2 = statevector(state)
    burgers_rhs!(du2, u, p, 0.0)
    @test du2 == du

    # And `du` sums to zero over the domain, cell by cell weighted by
    # volume — the discrete statement of conservation, one evaluation at
    # a time and independent of the time integrator.
    scratch = FieldSet(forest, 1; G=2)
    scatter!(scratch, du)
    @test abs(total_mass(scratch)) <= 64 * eps(Float64) * burgers_mass_scale(scratch)
end

@testset "A Burgers problem refuses a layout its stencils do not fit" begin
    # The two obligations the application carries, said out loud where
    # they are cheap to check rather than discovered as a silent
    # conservation leak: the reconstruction reaches two cells past a
    # block face, and the state is cell-centered.
    forest = burgers_forest(Val(1), 8; roots=4, refined=false)
    ops = Operators(family=Conservative, prolongation=3, restriction=2)
    @test_throws "G >= 2" BurgersProblem(FieldSet(forest, 1; G=1), ops)
    @test_throws "cell-centered" BurgersProblem(
        FieldSet(forest, 1; G=2, centering=vertexcentered(1)), ops)
    @test_throws "limiter" BurgersProblem(FieldSet(forest, 1; G=2), ops;
                                          limiter=:vanleer)
end
