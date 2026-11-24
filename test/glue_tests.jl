# Ghosts interpolated across a glue (M20): "Finite differences across a
# glue" in CODE.md. The forests and atlases are in `helpers/glue.jl`.

using TreeAMR: GluePlan, chart_aligned

# A polynomial of degree five in each logical coordinate, different in
# each patch, which a six-point stencil reproduces exactly.
glue_poly(ξ, p) = 0.5 + p * ξ[1]^5 + 2ξ[1]^3 * ξ[2]^4 - (p + 1) * ξ[2]^5 + ξ[1] * ξ[2]

# Owned points from `f(ξ, p)`, every other stored point NaN, then the fill.
function fill_from_owned!(fs, sched, f)
    forest = fs.forest
    G, N = fs.G, forest.N
    fs.work .= NaN
    for b in 1:nblocks(fs)
        k = forest.leaves[b]
        p, h, o = patch(forest, k), spacing(forest, k), block_origin(forest, k)
        for I in CartesianIndices(ntuple(d -> (G[d] + 1):(G[d] + N), 2))
            ξ = ntuple(d -> o[d] + (I[d] - G[d] - 1) * h, 2)
            fs.work[I, 1, b] = f(ξ, p)
        end
    end
    fill_ghosts!(fs, sched)
    return fs
end

@testset "Interpolated ghosts across a kinked glue reproduce a patch's polynomial exactly" begin
    # Guards the stencil: the window, its clamping into the patch across,
    # the weights and the source block's index arithmetic. Data that is a
    # polynomial of degree glue - 1 in each patch's logical coordinates is
    # interpolated exactly wherever the ghost lands, kink or not.
    for n in (1, 2)
        forest = kinked_forest(n)
        fs = kinked_set(forest)
        sched = GhostSchedule(fs, KINKED_OPS; atlas=KINKED)
        fill_from_owned!(fs, sched, glue_poly)
        ghosts = glue_ghosts(fs)
        @test length(ghosts) == length(sched.glueplan)
        # A ghost on a glued face plane is read from where that plane is
        # owned, and each patch has its own polynomial.
        lvl = level(first(forest.leaves))
        err = maximum(ghosts) do (b, idx, ξ, p)
            ξ′, q = TreeAMR.chart_locate(KINKED, TreeAMR.chart_map(KINKED, ξ, p))
            ξ″, q″ = TreeAMR.owned_side(forest, forest.conn, Float64, ξ′, q, lvl, (1, 1))
            return abs(fs.work[idx..., 1, b] - glue_poly(ξ″, q″))
        end
        @test err < 1e-11
    end
end

@testset "Interpolated ghosts across a kinked glue converge at the glue order" begin
    # Guards the location: a ghost read at the wrong physical point stays
    # wrong at every resolution. Six-point stencils converge at 6.
    errs = map((1, 2, 4)) do n
        forest = kinked_forest(n)
        fs = kinked_set(forest)
        sched = GhostSchedule(fs, KINKED_OPS; atlas=KINKED)
        fill_from_owned!(fs, sched,
                         (ξ, p) -> kinked_wave(KINKED, TreeAMR.chart_map(KINKED, ξ, p)))
        return maximum(glue_ghosts(fs)) do (b, idx, ξ, p)
            abs(fs.work[idx..., 1, b] - kinked_wave(KINKED, TreeAMR.chart_map(KINKED, ξ, p)))
        end
    end
    rates = [log2(errs[i] / errs[i+1]) for i in 1:2]
    @test all(r -> r > 5.5, rates)
end

# The annulus, whose maps join smoothly: an aligned atlas, so that every
# ghost lands on a node.
struct AlignedPolar end
TreeAMR.chart_locate(::AlignedPolar, x) = TreeAMR.chart_locate(PolarAtlas(), x)
TreeAMR.chart_map(::AlignedPolar, ξ, p) = TreeAMR.chart_map(PolarAtlas(), ξ, p)
TreeAMR.chart_aligned(::AlignedPolar, p, q) = (true, true)

ring8() = Forest([(roots=(2, 1),) for _ in 1:4]; N=8,
                 glue=reduce(vcat, [patch_glue_both((p, 1, 2), (mod1(p + 1, 4), 1, 1),
                                                    (1, 2)) for p in 1:4]))

@testset "On the annulus interpolated ghosts are the copied ones bit for bit" begin
    # Guards the alignment: along an aligned dimension the stencil has
    # width one and weight one, so where the maps join smoothly the
    # interpolation is the copy, exactly.
    forest = ring8()
    fi = FieldSet(forest, 1; G=2, centering=vertexcentered(2), glue=:interpolate)
    fc = FieldSet(forest, 1; G=2, centering=vertexcentered(2))
    ops = Operators(prolongation=4, restriction=4, glue=6)
    si = GhostSchedule(fi, ops; atlas=AlignedPolar())
    sc = GhostSchedule(fc, ops)
    @test all(==(1), reduce(vcat, collect.(si.glueplan.width)))
    f(ξ, p) = sin(3 * (2(p - 1) + ξ[1])) * (1 + ξ[2])^2
    fill_from_owned!(fi, si, f)
    fill_from_owned!(fc, sc, f)
    @test isequal(fi.work, fc.work)
end

@testset "An alignment the maps do not have is refused" begin
    # Guards the declaration's check: a wrong declaration would read a
    # node that is not where the ghost lies.
    struct WronglyAligned end
    TreeAMR.chart_locate(::WronglyAligned, x) = TreeAMR.chart_locate(KINKED, x)
    TreeAMR.chart_map(::WronglyAligned, ξ, p) = TreeAMR.chart_map(KINKED, ξ, p)
    TreeAMR.chart_aligned(::WronglyAligned, p, q) = (true, true)
    fs = kinked_set(kinked_forest(1))
    @test_throws "not at a node" GhostSchedule(fs, KINKED_OPS; atlas=WronglyAligned())
end

@testset "A set interpolated across a glue refuses what it cannot fill" begin
    # Guards the refusals of the schedule build, each with its reason.
    forest = kinked_forest(1)
    fs = kinked_set(forest)
    @test_throws "glue = k" GhostSchedule(fs, Operators(prolongation=4, restriction=4);
                                          atlas=KINKED)
    @test_throws "needs the application's atlas" GhostSchedule(fs, KINKED_OPS)
    @test_throws "k ≤ 2(G + c)" GhostSchedule(kinked_set(forest; G=1), KINKED_OPS;
                                              atlas=KINKED)
    @test_throws "glue must be" FieldSet(forest, 1; G=1, glue=:mirror)
    @test_throws "element-centered" FieldSet(kinked_forest(2), 1; G=1,
                                             centering=elementcentered(2), P=1,
                                             glue=:interpolate)
    @test_throws "glue order must be even" Operators(prolongation=4, restriction=4,
                                                     glue=5)
    @test_throws "takes no glue order" Operators(family=ElementLocal, glue=4)
    # A refinement boundary at a glue: refine one block on patch 1's high
    # face, then balance.
    f2 = kinked_forest(2)
    refine!(f2, first(k for k in f2.leaves if patch(f2, k) == 1 && k.coords == (0, 0) &&
                      TreeAMR.root_position(f2, k.root)[1] == 1))
    balance!(f2)
    @test_throws "refinement boundary touches a glued face" GhostSchedule(
        kinked_set(f2), KINKED_OPS; atlas=KINKED)
    # The schedule and the set must agree on the glue mode.
    ring = ring8()
    fr = FieldSet(ring, 1; G=2, centering=vertexcentered(2), glue=:interpolate)
    sc = GhostSchedule(FieldSet(ring, 1; G=2, centering=vertexcentered(2)), KINKED_OPS)
    @test_throws "glue = :interpolate but this schedule was built for :copy" fill_ghosts!(
        fr, sc)
end

@testset "A brick ignores the glue mode" begin
    # Guards that a brick's layout and fill do not change: nothing on a
    # brick is glued, so interpolating across its glue is copying nothing.
    forest = Forest((2, 2); N=8, periodic=(true, true))
    fi = FieldSet(forest, 1; G=2, glue=:interpolate)
    fc = FieldSet(forest, 1; G=2)
    ops = Operators(prolongation=4, restriction=4)
    si = GhostSchedule(fi, ops)
    sc = GhostSchedule(fc, ops)
    @test isempty(si.glueplan)
    fi.work .= reshape(1.0:length(fi.work), size(fi.work))
    fc.work .= fi.work
    fill_ghosts!(fi, si)
    fill_ghosts!(fc, sc)
    @test fi.work == fc.work
end
