# The all-variables form of the coordinate callbacks, added for the
# TreeHydro application: `AllVariables(f)` is called once per point and
# returns every variable at once, beside the `(x, v)` form that is called
# once per point and variable.
#
# The claim throughout is *bit-for-bit agreement* with the per-variable
# form on position-determined data, not agreement to roundoff. Both forms
# have to build the position from the same origin and spacing in the same
# order, or an application that switched forms would see its numbers move
# in the last bit — and, through the initial data, the M5
# thread-independence digests with them.

const OPS2A = Operators(prolongation=2, restriction=2)

# The same function written both ways, so that "the two forms agree" is a
# statement about the package and not about the two closures. `Val` on
# the variable count keeps the tuple version inferable, as an
# application's own would be.
const PER_VARIABLE = (x, v) -> sum(x) * (1 + v) + oftype(x[1], 7v) - x[1]^2
all_variables(nv::Integer) = all_variables(Val(Int(nv)))
all_variables(::Val{NV}) where {NV} =
    AllVariables(x -> ntuple(v -> sum(x) * (1 + v) + oftype(x[1], 7v) - x[1]^2, Val(NV)))

# Centerings and per-dimension ghost widths to sweep, a zero among them.
# Every layout here also satisfies `check_operators` at order 2, since
# the boundary testset below exchanges ghosts over it: a ghost width of
# zero is legal along a stagger and nowhere else.
allvariables_cases(D) =
    D == 1 ? ((cellcentered(1), (1,)), (vertexcentered(1), (0,))) :
    D == 2 ? ((cellcentered(2), (1, 1)), (facecentered(2, 1), (0, 1)),
              (vertexcentered(2), (1, 0))) :
             ((cellcentered(3), (1, 1, 1)), (facecentered(3, 2), (1, 0, 1)),
              (edgecentered(3, 1), (1, 1, 1)))

# A three-level non-periodic hierarchy, with smaller blocks in 3D.
allvariables_forest(::Val{D}) where {D} = nested_forest(Val(D); N=D == 3 ? 4 : 8)

@testset "The all-variables fill reproduces the per-variable fill: D=$D" for
        D in (1, 2, 3)
    # The whole working array is compared, not a norm of it: the two
    # kernels differ in their ndrange and in how they reach the variable
    # axis, and the position each forms must still be the identical
    # floating-point number.
    forest = allvariables_forest(Val(D))
    for (C, G) in allvariables_cases(D), nv in (1, 3)
        onebyone = FieldSet(forest, nv; G=G, centering=C)
        atonce = FieldSet(forest, nv; G=G, centering=C)
        fill_by_coordinates!(PER_VARIABLE, onebyone)
        @test fill_by_coordinates!(all_variables(nv), atonce) === atonce
        @test atonce.work == onebyone.work
        # Not vacuous, and the ghosts are left to the exchange by both
        # forms alike: the same cells are still zero.
        @test !all(iszero, atonce.work)
        @test count(iszero, atonce.work) == count(iszero, onebyone.work)
    end
end

@testset "The all-variables boundary hook reproduces the per-variable one: D=$D" for
        D in (1, 2, 3)
    # The hook is the form that matters most, because it runs at every
    # ghost fill rather than once at setup. The comparison is over the
    # whole array after a full `fill_ghosts!` on a *non-periodic*
    # three-level mesh, so face, edge and corner regions are all covered,
    # and so are the prolongations that read the hook's output
    # tangentially.
    forest = allvariables_forest(Val(D))
    @test length(unique(level.(forest.leaves))) >= 3
    nv = 2

    for (C, G) in allvariables_cases(D)
        per = FieldSet(forest, nv; G=G, centering=C)
        whole = FieldSet(forest, nv; G=G, centering=C)
        schedule = GhostSchedule(per, OPS2A)
        @test !isempty(schedule.boundaries)          # there are outer regions
        fill_by_coordinates!(PER_VARIABLE, per)
        fill_by_coordinates!(PER_VARIABLE, whole)

        fill_ghosts!(per, schedule; boundary=boundary_by_coordinates(PER_VARIABLE))
        fill_ghosts!(whole, GhostSchedule(whole, OPS2A);
                     boundary=boundary_by_coordinates(all_variables(nv)))
        @test whole.work == per.work

        # And the direction-dependent form, which is what a real hook
        # uses: the two spellings must agree there too.
        perδ = FieldSet(forest, nv; G=G, centering=C)
        allδ = FieldSet(forest, nv; G=G, centering=C)
        fill_by_coordinates!(PER_VARIABLE, perδ)
        fill_by_coordinates!(PER_VARIABLE, allδ)
        fill_ghosts!(perδ, GhostSchedule(perδ, OPS2A);
                     boundary=CellBoundary((x, v, δ) -> sum(x) + v * δ[1]))
        fill_ghosts!(allδ, GhostSchedule(allδ, OPS2A);
                     boundary=CellBoundary(AllVariables(
                         (x, δ) -> ntuple(v -> sum(x) + v * δ[1], Val(2)))))
        @test allδ.work == perδ.work
        @test allδ.work != whole.work                # the direction really is read
    end
end

@testset "adapt_to_initial_data! takes either callback form: D=$D" for D in (1, 2)
    # The cycle forwards `initial` to `fill_by_coordinates!` and
    # re-evaluates it on every mesh it produces, so the two forms must
    # give the same *hierarchy* as well as the same data: a criterion
    # reading a field that differed in the last bit could flag
    # differently, and the two meshes would then diverge for good.
    per = (x, v) -> exp(-sum(y -> (y - 1) * (y - 1), x) * 8) + v
    whole = AllVariables(x -> ntuple(v -> exp(-sum(y -> (y - 1) * (y - 1), x) * 8) + v,
                                     Val(2)))
    criterion = (b, k) -> level(k) < 2 ? Refine : Keep

    results = map((per, whole)) do initial
        forest = Forest(ntuple(_ -> 2, D); N=8, periodic=ntuple(_ -> false, D),
                        extents=ntuple(_ -> (0.0, 2.0), D))
        fs = FieldSet(forest, 2; G=1)
        _, passes, converged = adapt_to_initial_data!(
            fs, OPS2A; initial=initial, flag=criterion, maxpasses=5,
            boundary=boundary_by_coordinates(initial))
        (copy(forest.leaves), passes, converged, copy(fs.work))
    end

    @test results[1][1] == results[2][1]             # the same leaves
    @test results[1][2] == results[2][2]             # in the same number of passes
    @test results[1][3] && results[2][3]
    @test results[1][4] == results[2][4]             # and the same data
    @test length(results[1][1]) > 2^D                # refinement really happened
end

@testset "A callback returning the wrong number of variables is refused" begin
    # A kernel cannot report this: an out-of-range tuple index inside a
    # launch names neither the callback nor the count, and on a backend
    # that does not check it would write a wrong number instead. So the
    # length is checked once on the host, before anything is launched,
    # and the message names both numbers.
    forest = Forest((2, 2); N=8, periodic=(false, false))
    fs = FieldSet(forest, 3; G=1)

    @test_throws ArgumentError fill_by_coordinates!(AllVariables(x -> (sum(x),)), fs)
    @test_throws "nvars = 3" fill_by_coordinates!(AllVariables(x -> (sum(x),)), fs)
    @test_throws "returned 1" fill_by_coordinates!(AllVariables(x -> (sum(x),)), fs)
    @test_throws "returned 4" fill_by_coordinates!(
        AllVariables(x -> (1.0, 2.0, 3.0, 4.0)), fs)
    # A scalar has length 1, so it is caught by the same check rather
    # than silently filling every variable with it.
    @test_throws "returned 1" fill_by_coordinates!(AllVariables(sum), fs)
    # The right length passes, so the check is not simply refusing.
    @test fill_by_coordinates!(AllVariables(x -> (1.0, 2.0, 3.0)), fs) === fs

    schedule = GhostSchedule(fs, OPS2A)
    short = CellBoundary(AllVariables((x, δ) -> (sum(x), sum(x))))
    @test_throws ArgumentError fill_ghosts!(fs, schedule; boundary=short)
    @test_throws "nvars = 3" fill_ghosts!(fs, schedule; boundary=short)
    @test_throws "returned 2" fill_ghosts!(fs, schedule; boundary=short)
    @test_throws "returned 2" fill_ghosts!(
        fs, schedule; boundary=boundary_by_coordinates(AllVariables(x -> (1.0, 2.0))))
    @test fill_ghosts!(fs, schedule;
                       boundary=CellBoundary(AllVariables(
                           (x, δ) -> (1.0, 2.0, 3.0)))) === fs

    # And through the initial-data cycle, before it has moved anything.
    @test_throws "nvars = 3" adapt_to_initial_data!(
        fs, OPS2A; initial=AllVariables(x -> (1.0,)), flag=(b, k) -> Keep)
    @test nleaves(forest) == 4
end

@testset "AllVariables is isbits when its callback is" begin
    # It is a kernel argument, so this is a requirement and not a
    # coincidence: a wrapper with a mutable field, or one that boxed the
    # callback, would fail on a device rather than here.
    @test isbits(AllVariables(sum))
    @test isbits(AllVariables(x -> (sum(x), one(eltype(x)))))
    two = 2.0
    @test isbits(AllVariables(x -> (two * sum(x),)))       # an isbits capture
    @test !isbits(AllVariables([1.0, 2.0]))                # an array one is not
    @test AllVariables(sum).f === sum
end
