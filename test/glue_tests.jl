# Ghosts interpolated across a glue (M20): "Finite differences across a
# glue" in CODE.md. The forests and atlases are in `helpers/glue.jl`.

using TreeAMR: GluePlan, chart_aligned

# A polynomial of degree five in each logical coordinate, different in
# each patch, which a six-point stencil reproduces exactly. Its terms that
# differ between the patches vanish on the glued faces `ξ₁ ∈ {0, 1}`, and
# every term in `ξ₂` at the wrap's `ξ₂ ∈ {0, 1}`, so that the copies of a
# glued point agree bit for bit and stage A's mean (M22) leaves them as
# they are.
glue_poly(ξ, p) = 0.5 + p * ξ[1]^4 * (1 - ξ[1]) +
                  2ξ[1]^3 * (1 - ξ[1])^2 * ξ[2]^3 * (1 - ξ[2])^2 -
                  3ξ[2]^4 * (1 - ξ[2]) + ξ[1] * (1 - ξ[1]) * ξ[2] * (1 - ξ[2])

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
        # Each patch has its own polynomial, equal on the glued faces, so a
        # ghost on a glued face takes the value of the patch the atlas
        # gives it to.
        err = maximum(ghosts) do (b, idx, ξ, p)
            ξ′, q = TreeAMR.chart_locate(KINKED, TreeAMR.chart_map(KINKED, ξ, p))
            return abs(fs.work[idx..., 1, b] - glue_poly(ξ′, q))
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
    # Periodic around the ring, and the same bits on both sides of a glue,
    # where the patches' angles `2(p - 1) + ξ₁` are the same integer.
    f(ξ, p) = cospi(3 * (2(p - 1) + ξ[1]) / 4) * (1 + ξ[2])^2
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

@testset "A kinked set's state holds every copy of a glued face's point" begin
    # Guards the closed layout across a kinked glue (M22): no appendix, every
    # block's closed box in the state, `(N + 1)²` points a block, and a
    # state that round-trips through `scatter!` and `gather!`; the
    # appendix's keyword is refused with its reason.
    forest = kinked_forest(2)
    fs = FieldSet(forest, 2; G=3, centering=vertexcentered(2), glue=:interpolate)
    N = forest.N
    @test statelength(fs) == 2 * nblocks(fs) * (N + 1)^2
    u = collect(1.0:statelength(fs))
    scatter!(fs, u)
    w = zero(u)
    gather!(w, fs)
    @test w == u
    @test size(statearray(u, fs)) == (N + 1, N + 1, 2, nblocks(fs))
    @test_throws "no longer takes `appendix`" statearray(u, fs; appendix=true)
end

# Two patches of `n × n` roots side by side along dimension 1, glued at
# one pair of faces with the normal reversed: low face to low face, or
# high face to high face. Every other face is outer.
function folded_pair(n, side; N=8)
    glue = patch_glue_both((1, 1, side), (2, 1, side), (-1, 2))
    patches = [(roots=(n, n), extents=((0.0, 1.0), (0.0, 1.0))) for _ in 1:2]
    return Forest(patches; N=N, glue=glue)
end

# A state vector holding, at every own point, the patch of its block, so
# that a fold can tell which copy it read.
function patch_state(fs)
    u = fill(NaN, statelength(fs))
    A = statearray(u, fs)
    for b in 1:nblocks(fs)
        A[:, :, 1, b] .= patch(fs.forest, fs.forest.leaves[b])
    end
    return u
end

@testset "A fold over the primaries counts every glued point once, at its lowest patch" begin
    # Guards the mask across a kinked glue (M22): every copy of a point is
    # stored, on each patch holding it and on each block holding it there,
    # and `primaries = true` takes each point once, at the copy of the
    # lowest patch, which is the one stage A's mean is the same in.
    count1(acc, p) = acc + 1
    for n in (1, 2)
        # The torus: 2(nN)² distinct points; patch 1 holds both of its
        # glued planes along dimension 1, (nN + 1) nN points, and the wrap
        # along dimension 2 counts one plane.
        forest = kinked_forest(n)
        N = forest.N
        fs = kinked_set(forest)
        u = patch_state(fs)
        @test mesh_fold(count1, +, 0, fs, u; primaries=true) == 2 * (n * N)^2
        ones1 = mesh_fold((acc, p) -> acc + (p.u[1] == 1), +, 0, fs, u; vars=1,
                          primaries=true)
        @test ones1 == (n * N + 1) * n * N
        @test mesh_mapreduce(x -> 1, +, 0, fs, u; primaries=true) == 2 * (n * N)^2
        @test mesh_mapreduce(identity, max, 0.0, fs, u; primaries=false) == 2
        scatter!(fs, u)
        @test mesh_fold(count1, +, 0, fs; primaries=true) == 2 * (n * N)^2
        # A low face glued to a low face and a high face to a high face:
        # (nN + 1)² points a patch, one plane of them shared, patch 1's.
        for side in (1, 2)
            fp = FieldSet(folded_pair(n, side), 1; G=3, centering=vertexcentered(2),
                          glue=:interpolate)
            v = patch_state(fp)
            @test mesh_fold(count1, +, 0, fp, v; primaries=true) ==
                  2 * (n * N + 1)^2 - (n * N + 1)
            @test mesh_fold((acc, p) -> acc + (p.u[1] == 1), +, 0, fp, v;
                            primaries=true) == (n * N + 1)^2
        end
    end
end

@testset "Norms over a kinked set weight each copy by its share" begin
    # Guards the trapezoid's weights across a kinked glue (M22): a point on
    # a glued face is half on each patch's side, so a constant has its own
    # value as its norm and the torus's two unit patches a mass of 2.
    for side in (1, 2)
        fs = FieldSet(folded_pair(2, side), 2; G=3, centering=vertexcentered(2),
                      glue=:interpolate)
        u = fill(3.0, statelength(fs))
        @test volume_weighted_norm(fs, u) ≈ 3.0 rtol = 1e-14
        @test volume_weighted_norm(fs, u; p=1) ≈ 3.0 rtol = 1e-14
        @test volume_weighted_norm(fs, u; p=Inf) == 3.0
    end
    fs = kinked_set(kinked_forest(2))
    u = fill(-2.0, statelength(fs))
    @test volume_weighted_norm(fs, u) ≈ 2.0 rtol = 1e-14
    scatter!(fs, u)
    @test total_mass(fs) ≈ -2.0 * 2 rtol = 1e-14
end

# The leaves of `forest` on a glued face of their patch.
on_glue(forest, k) = !isempty(TreeAMR.glued_faces(forest.conn, k))

@testset "regrid! keeps every leaf on a glued face at its level, and its points" begin
    # Guards the regrid of a set interpolated across a glue: a flag on a
    # glued face is dropped, the interior refines, the own points of the
    # kept blocks, their face planes included, come through unchanged, and
    # the schedule builds again, with no refinement boundary at a glue.
    forest = kinked_forest(4)
    N = forest.N
    fs = kinked_set(forest)
    sched = GhostSchedule(fs, KINKED_OPS; atlas=KINKED)
    owned(u) = let A = statearray(u, fs)
        Dict(k => vec(A[:, :, 1, b])
             for (b, k) in enumerate(forest.leaves) if on_glue(forest, k))
    end
    # Unified data, as a fill leaves it: the regrid fills before its
    # transfer, which would otherwise move the copies to their mean.
    fill_from_owned!(fs, sched, (ξ, p) -> glue_poly(ξ, p))
    u = statevector(fs)
    gather!(u, fs)
    before = owned(u)
    # A flag on a glued face alone changes nothing.
    flags = [on_glue(forest, k) ? Refine : Keep for k in forest.leaves]
    @test !regrid!(forest, fs => sched; flags=flags)
    # Every block flagged: only the interior ones refine.
    nglued = count(k -> on_glue(forest, k), forest.leaves)
    @test regrid!(forest, fs => sched; flags=fill(Refine, nblocks(fs)))
    @test nleaves(forest) == nglued + 4 * (32 - nglued)
    @test all(k -> level(k) == (on_glue(forest, k) ? 0 : 1), forest.leaves)
    v = statevector(fs)
    gather!(v, fs)
    @test owned(v) == before
    sched = GhostSchedule(fs, KINKED_OPS; atlas=KINKED)
    # A refinement next to a glued face, which balance would carry into
    # it, is refused, saying why.
    neighbors(k) = (n for δ in TreeAMR.alldirections(Val(2))
                    for n in oriented_neighbors(forest, k, δ)[2])
    flags = [level(k) == 1 && any(n -> on_glue(forest, n), neighbors(k)) ? Refine : Keep
             for k in forest.leaves]
    @test any(==(Refine), flags)
    @test_throws "2:1 balance carries a refinement nearby" regrid!(forest, fs => sched;
                                                                   flags=flags)
end

@testset "fill_by_coordinates! fills every own point of a kinked set in every form" begin
    # Guards the initial-data path across a kinked glue: the face planes
    # are state too. Each form of the callback fills every own point with
    # the same numbers.
    forest = kinked_forest(2)
    N = forest.N
    f(x, v) = v + x[1] - 3x[2]
    forms = (f, AllVariables(x -> (f(x, 1), f(x, 2))), WithPatch((x, p, v) -> f(x, v)),
             WithPatch(AllVariables((x, p) -> (f(x, 1), f(x, 2)))))
    for form in forms
        fs = FieldSet(forest, 2; G=3, centering=vertexcentered(2), glue=:interpolate)
        fill_by_coordinates!(form, fs)
        u = statevector(fs)
        gather!(u, fs)
        A = statearray(u, fs)
        @test all(1:nblocks(fs)) do b
            k = blockkey(fs, b)
            o, h = block_origin(forest, k), spacing(forest, k)
            all(CartesianIndices((N + 1, N + 1))) do I
                x = (o[1] + (I[1] - 1) * h, o[2] + (I[2] - 1) * h)
                A[I[1], I[2], 1, b] == f(x, 1) && A[I[1], I[2], 2, b] == f(x, 2)
            end
        end
    end
    # And the adaptation cycle takes the atlas for its schedules.
    fs = kinked_set(forest)
    sched, passes, converged = adapt_to_initial_data!(fs, KINKED_OPS; atlas=KINKED,
                                                      initial=(x, v) -> x[1],
                                                      flag=(b, k) -> Keep)
    @test converged && passes == 1 && length(sched.glueplan) > 0
end

@testset "A fold counts mixed centerings once across a glue that turns the axes" begin
    # Guards the search for a point's copies: a glue that sends dimension
    # 1 of one patch onto dimension 2 of the other carries a vertex-like
    # face onto a cell-like one, where the set has no point, and the fold
    # counted a copy there that does not exist. Over the primaries a patch
    # of a vertex-like dimension `i` holds `(nN + 1)` points along it, a
    # cell-like one `nN`, and only a vertex set shares the glued plane.
    n, N = 2, 8
    forest = Forest([(roots=(n, n),), (roots=(n, n),)]; N=N,
                    glue=patch_glue_both((1, 1, 2), (2, 2, 1), (2, -1)))
    m = n * N
    for (cen, total) in (((:vertex, :vertex), 2 * (m + 1)^2 - (m + 1)),
                         ((:vertex, :cell), 2 * (m + 1) * m),
                         ((:cell, :vertex), 2 * m * (m + 1)))
        fs = FieldSet(forest, 1; G=3, centering=cen, glue=:interpolate)
        @test mesh_fold((acc, p) -> acc + 1, +, 0, fs, zeros(statelength(fs));
                        primaries=true) == total
    end
end

@testset "A glue stencil reads only what phase 1 and the hook have filled" begin
    # Guards the overhang rule. A ghost that lands deep in the patch across
    # reads its source block's ghosts past its high plane, which a coarser
    # neighbor fills by prolongation only in phase 2, after the glue's
    # stage: the build refuses it rather than leave the ghost a fill
    # behind. A finer neighbor's ghosts are restricted in phase 1, and are
    # read. The high plane itself is the block's own (M22), so a stencil
    # that reaches it and no further is accepted: `a = 0.22` lands the
    # deepest ghost 13.6 spacings into the patch, its window `11 … 16`
    # ending on that plane, and `a = 0.2` at 15, its window `13 … 18`.
    ops = Operators(prolongation=4, restriction=4, glue=6)
    for (a, refused) in ((0.2, true), (0.22, false), (0.8, false))
        forest = stretched_forest(2)
        for k in copy(forest.leaves)
            p, pos = patch(forest, k), TreeAMR.root_position(forest, k.root)[1]
            ((p == 1 && pos == 1) || (p == 2 && pos == 0)) && refine!(forest, k)
        end
        balance!(forest)
        fs = FieldSet(forest, 1; G=3, centering=vertexcentered(2), glue=:interpolate)
        if refused
            @test_throws "no block of the same level or finer" GhostSchedule(
                fs, ops; atlas=StretchedPair(a))
        else
            atlas = StretchedPair(a)
            f(ξ, p) = (x = TreeAMR.chart_map(atlas, ξ, p); 1 + x[1] - 2x[1]^2 * x[2]^3)
            sched = GhostSchedule(fs, ops; atlas=atlas)
            fill_from_owned!(fs, sched, f;
                             boundary=CellBoundary(WithPatch((x, p, v, δ) -> f(x, p))))
            @test !any(((b, idx, ξ, p),) -> isnan(fs.work[idx..., 1, b]), glue_ghosts(fs))
        end
    end
    # regrid! keeps what the stencils read, so the schedule builds again:
    # a finer neighbor of a glued source is accepted, a coarser one never
    # arises.
    forest = stretched_forest(4)
    atlas = StretchedPair(0.5)
    fs = FieldSet(forest, 1; G=3, centering=vertexcentered(2), glue=:interpolate)
    sched = GhostSchedule(fs, ops; atlas=atlas)
    read = sched.glueplan.reads
    @test any(i -> isempty(TreeAMR.glued_faces(forest.conn, forest.leaves[i])), read)
    before = forest.leaves[read]
    @test regrid!(forest, fs => sched; flags=fill(Refine, nblocks(fs)))
    @test all(k -> k in forest.leaves, before)
    @test any(k -> level(k) == 1, forest.leaves)
    @test length(GhostSchedule(fs, ops; atlas=atlas).glueplan) > 0
end

# The kinked pair: the torus's two patches glued at one face only, every
# other face outer, so that a set copying across the glue meets no patch
# corner and can be compared with one interpolating.
kinked_pair(n) = Forest([(roots=(n, n), extents=((0.0, 1.0), (0.0, 1.0))) for _ in 1:2];
                        N=8, glue=patch_glue_both((1, 1, 2), (2, 1, 1), (1, 2)))

# The torus's smooth function as each patch's map places it, and its
# first and second derivatives along logical dimension `d`, by central
# differences of that function in 256-bit arithmetic.
kinked_data(ξ, p) = kinked_wave(KINKED, TreeAMR.chart_map(KINKED, ξ, p))
function kinked_derivatives(ξ, p, d)
    setprecision(BigFloat, 256) do
        h = big"1e-25"
        x = big.(ξ)
        e = ntuple(i -> i == d ? h : zero(h), 2)
        fc, fp, fm = kinked_data(x, p), kinked_data(x .+ e, p), kinked_data(x .- e, p)
        return Float64((fp - fm) / 2h), Float64((fp - 2 * fc + fm) / h^2)
    end
end

# The largest errors of the fourth-order centered first and second
# derivatives over every core point, each along both dimensions, after a
# fill of the kinked pair in `glue` mode.
function kinked_derivative_errors(n, glue)
    forest = kinked_pair(n)
    fs = kinked_set(forest; glue=glue)
    sched = glue === :interpolate ? GhostSchedule(fs, KINKED_OPS; atlas=KINKED) :
            GhostSchedule(fs, Operators(prolongation=4, restriction=4))
    fill_from_owned!(fs, sched, kinked_data;
                     boundary=CellBoundary(WithPatch((x, p, v, δ) -> kinked_data(x, p))))
    G, N = fs.G, forest.N
    c1 = (1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12)
    c2 = (-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12)
    e1 = e2 = 0.0
    for b in 1:nblocks(fs)
        k = blockkey(fs, b)
        p, h, o = patch(forest, k), spacing(forest, k), block_origin(forest, k)
        for I in CartesianIndices((1:N, 1:N)), d in 1:2
            u(j) = fs.work[ntuple(e -> G[e] + I[e] + (e == d ? j : 0), 2)..., 1, b]
            x1, x2 = kinked_derivatives(ntuple(e -> o[e] + (I[e] - 1) * h, 2), p, d)
            e1 = max(e1, abs(sum(c1[j + 3] * u(j) for j in -2:2) / h - x1))
            e2 = max(e2, abs(sum(c2[j + 3] * u(j) for j in -2:2) / h^2 - x2))
        end
    end
    return e1, e2
end

@testset "A smooth solution's derivatives converge at the operator's order across a kink" begin
    # Guards what the interpolation is for: a fourth-order derivative read
    # across a glue where the map kinks. With interpolated ghosts the first
    # and second derivatives converge at 4 everywhere, the points next to
    # the glue included; with copied ones the ghosts hold a function with a
    # kink in the reader's coordinate, and the first derivative's largest
    # error stays O(1) while the second's grows like 1/h (HISTORY.md, M20,
    # "the derivatives across a glue", has the numbers).
    ns = (2, 4, 8)
    rate(errs, i) = [log2(errs[j][i] / errs[j + 1][i]) for j in 1:(length(errs) - 1)]
    interpolated = [kinked_derivative_errors(n, :interpolate) for n in ns]
    @test all(>(3.8), rate(interpolated, 1)) && all(>(3.8), rate(interpolated, 2))
    copied = [kinked_derivative_errors(n, :copy) for n in ns]
    @test all(e -> e[1] > 0.3, copied)
    @test all(<(-0.9), rate(copied, 2))
end

# A polynomial of degree `k - 1` in each logical coordinate, different in
# each patch, which a `k`-point stencil reproduces exactly; its terms that
# differ between the patches vanish on the glued faces, and its terms in
# `ξ₂` at the wrap, as `glue_poly`'s do.
glue_poly_k(k) = (ξ, p) -> 0.5 + (ξ[1] * (1 - ξ[1]) * ξ[2] * (1 - ξ[2]))^(k ÷ 2 - 1) +
                           (k >= 4 ? p * ξ[1]^(k - 2) * (1 - ξ[1]) -
                                     3ξ[2]^(k - 2) * (1 - ξ[2]) : 0.0)

@testset "Interpolated ghosts are exact for every centering and order" begin
    # Guards the stencil's window and weights away from the one layout the
    # other tests use: a cell-like dimension has no point on a face, a
    # mixed set has one along one dimension only, and the
    # window's clamping and centering change with the order. Each patch's
    # polynomial of degree `k - 1` comes back to rounding at every ghost
    # across the kinked torus's glues and wrap, patch corners included.
    for C in ((:cell, :cell), (:vertex, :cell), (:cell, :vertex), (:vertex, :vertex)),
        k in (2, 4, 6)
        forest = kinked_forest(2)
        fs = FieldSet(forest, 1; G=3, centering=C, glue=:interpolate)
        sched = GhostSchedule(fs, Operators(prolongation=4, restriction=4, glue=k);
                              atlas=KINKED)
        f = glue_poly_k(k)
        fill_from_owned!(fs, sched, f)
        ghosts = glue_ghosts(fs)
        @test length(ghosts) == length(sched.glueplan)
        err = maximum(ghosts) do (b, idx, ξ, p)
            ξ′, q = TreeAMR.chart_locate(KINKED, TreeAMR.chart_map(KINKED, ξ, p))
            return abs(fs.work[idx..., 1, b] - f(ξ′, q))
        end
        @test err < 1e-11
        # A block's state is its own points, closed along a vertex-like
        # dimension.
        c = staggers(fs)
        @test statelength(fs) == nblocks(fs) * prod(ntuple(d -> fs.forest.N + c[d], 2))
    end
end
