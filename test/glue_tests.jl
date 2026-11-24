# Ghosts interpolated across a glue (M20): "Finite differences across a
# glue" in CODE.md. The forests and atlases are in `helpers/glue.jl`.

using TreeAMR: GluePlan, chart_aligned

# A polynomial of degree five in each logical coordinate, different in
# each patch, which a six-point stencil reproduces exactly.
glue_poly(ξ, p) = 0.5 + p * ξ[1]^5 + 2ξ[1]^3 * ξ[2]^4 - (p + 1) * ξ[2]^5 + ξ[1] * ξ[2]

# Owned points, the appendix included, from `f(ξ, p)`, every other stored
# point NaN, then the fill.
function fill_from_owned!(fs, sched, f)
    forest = fs.forest
    G, N = fs.G, forest.N
    dims = TreeAMR.appendix_layout(fs).dims
    fs.work .= NaN
    for b in 1:nblocks(fs)
        k = forest.leaves[b]
        p, h, o = patch(forest, k), spacing(forest, k), block_origin(forest, k)
        for I in CartesianIndices(ntuple(d -> (G[d] + 1):(G[d] + N + dims[b][d]), 2))
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
        # Each patch has its own polynomial, and owns its face planes, so
        # a ghost on a glued face takes the value of the patch the atlas
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

@testset "Closed ownership adds each patch's glued high face planes to the state, once" begin
    # Guards the appendix's layout: a patch of n × n roots of N points owns
    # (nN + 1)² points under closed ownership against (nN)² in its cores,
    # every point of the planes once, the edge point included.
    for n in (1, 2)
        forest = kinked_forest(n)
        fs = FieldSet(forest, 3; G=3, centering=vertexcentered(2), glue=:interpolate)
        N = forest.N
        @test statelength(fs) - TreeAMR.corelength(fs) == 3 * 2 * (2n * N + 1)
        @test statelength(FieldSet(forest, 3; G=3, centering=vertexcentered(2))) ==
              TreeAMR.corelength(fs)
        # A cell-centered set has no face plane to own.
        @test statelength(FieldSet(forest, 3; G=3, glue=:interpolate)) ==
              TreeAMR.corelength(fs)
    end
end

@testset "The appendix round-trips through scatter!, gather! and statearray" begin
    # Guards the order of the appendix: scatter! and gather! move each
    # entry to the stored point statearray(…; appendix = true) names.
    forest = kinked_forest(2)
    fs = FieldSet(forest, 2; G=3, centering=vertexcentered(2), glue=:interpolate)
    u = collect(1.0:statelength(fs))
    scatter!(fs, u)
    w = zero(u)
    gather!(w, fs)
    @test w == u
    A = statearray(u, fs; appendix=true)
    L = TreeAMR.appendix_layout(fs)
    N, G = forest.N, fs.G
    seen = 0
    for b in 1:nblocks(fs), v in 1:2,
        I in CartesianIndices(ntuple(d -> 1:(N + L.dims[b][d]), 2))
        @test A[Tuple(I)..., v, b] == fs.work[(Tuple(I) .+ G)..., v, b]
        seen += 1
    end
    @test seen == length(u)
    @test statearray(u, fs) == reshape(u[1:TreeAMR.corelength(fs)], N, N, 2, nblocks(fs))
    # A write to a point a block does not own is dropped.
    b = findfirst(e -> !e[1], L.dims)
    A[N + 1, 1, 1, b] = -1.0
    @test u == collect(1.0:statelength(fs))
end

@kernel function glue_position_kernel!(du, @Const(origins), @Const(h), ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[3]
    du[I[1], I[2], 1, b] = origins[b][1] + (I[1] - 1) * h[b] + 10 * (origins[b][2] + (I[2] - 1) * h[b])
end

@testset "map_blocks! with appendix = true writes every point the state owns" begin
    # Guards the launch over the closed range: a right-hand side written
    # once reaches the appendix, and a set with one must say whether.
    forest = kinked_forest(2)
    fs = FieldSet(forest, 1; G=3, centering=vertexcentered(2), glue=:interpolate)
    du = fill(NaN, statelength(fs))
    origins = [block_origin(forest, k) for k in forest.leaves]
    h = [spacing(forest, k) for k in forest.leaves]
    @test_throws "needs `appendix = true` or `false`" map_blocks!(
        glue_position_kernel!, fs, statearray(du, fs; appendix=true), origins, h, Val(fs.G))
    map_blocks!(glue_position_kernel!, fs, statearray(du, fs; appendix=true), origins, h,
                Val(fs.G); appendix=true)
    @test !any(isnan, du)
    A = statearray(du, fs; appendix=true)
    L = TreeAMR.appendix_layout(fs)
    N = forest.N
    @test all(b -> all(I -> A[Tuple(I)..., 1, b] ==
                            origins[b][1] + (I[1] - 1) * h[b] + 10 * (origins[b][2] + (I[2] - 1) * h[b]),
                       CartesianIndices(ntuple(d -> 1:(N + L.dims[b][d]), 2))),
              1:nblocks(fs))
end

# Two patches of `n × n` roots side by side along dimension 1, glued at
# one pair of faces with the normal reversed: low face to low face, or
# high face to high face. Every other face is outer.
function folded_pair(n, side; N=8)
    glue = patch_glue_both((1, 1, side), (2, 1, side), (-1, 2))
    patches = [(roots=(n, n), extents=((0.0, 1.0), (0.0, 1.0))) for _ in 1:2]
    return Forest(patches; N=N, glue=glue)
end

# A state vector holding, at every owned point, the patch and the owned
# index, so that a fold can tell which copy it read.
function patch_state(fs)
    u = fill(NaN, statelength(fs))
    A = statearray(u, fs; appendix=true)
    dims = TreeAMR.appendix_layout(fs).dims
    N = fs.forest.N
    for b in 1:nblocks(fs)
        p = patch(fs.forest, fs.forest.leaves[b])
        for I in CartesianIndices(ntuple(d -> 1:(N + dims[b][d]), 2))
            A[Tuple(I)..., 1, b] = p
        end
    end
    return u
end

@testset "A fold counts every glued face point once, at its lowest patch" begin
    # Guards the count under closed ownership: the cores alone count a
    # point on a low face glued to a low face twice and one on a high face
    # glued to a high face not at all, and the appendix holds the high
    # planes. Each point is read at the copy of the lowest patch, which
    # here is patch 1 wherever it holds the point.
    count1(acc, p) = acc + 1
    for n in (1, 2)
        # The torus: 2(nN)² distinct points; patch 1 holds both of its
        # glued planes along dimension 1, (nN + 1) nN points, and the wrap
        # along dimension 2 counts its low plane.
        forest = kinked_forest(n)
        N = forest.N
        fs = kinked_set(forest)
        u = patch_state(fs)
        @test mesh_fold(count1, +, 0, fs, u) == 2 * (n * N)^2
        ones1 = mesh_fold((acc, p) -> acc + (p.u[1] == 1), +, 0, fs, u; vars=1)
        @test ones1 == (n * N + 1) * n * N
        @test mesh_mapreduce(x -> 1, +, 0, fs, u) == 2 * (n * N)^2
        @test mesh_mapreduce(identity, max, 0.0, fs, u) == 2
        # The working array after scatter! counts the same points.
        scatter!(fs, u)
        @test mesh_fold(count1, +, 0, fs) == 2 * (n * N)^2
        @test mesh_mapreduce(x -> 1, +, 0, fs) == 2 * (n * N)^2
        # The device's lane split, on the CPU, visits the same points.
        @test sum(TreeAMR._block_fold(count1, +, 0, fs.work, fs, fs.G, 1:1, nothing,
                                      4)) == 2 * (n * N)^2
        # A fold visits the appendix: its owned index reaches N + 1.
        @test mesh_fold((acc, p) -> max(acc, maximum(p.I)), max, 0, fs, u) == N + 1
        # Low face to low face: patch 2's low plane is patch 1's.
        for (side, total, mine) in ((1, 2 * (n * N)^2 - n * N, (n * N)^2),
                                    (2, 2 * (n * N)^2 + n * N, (n * N)^2 + n * N))
            fp = FieldSet(folded_pair(n, side), 1; G=3, centering=vertexcentered(2),
                          glue=:interpolate)
            v = patch_state(fp)
            @test mesh_fold(count1, +, 0, fp, v) == total
            @test mesh_fold((acc, p) -> acc + (p.u[1] == 1), +, 0, fp, v) == mine
            # Without the appendix the cores count the shared plane twice
            # or not at all.
            fc = FieldSet(folded_pair(n, side), 1; G=3, centering=vertexcentered(2))
            @test mesh_fold(count1, +, 0, fc, zeros(statelength(fc))) == 2 * (n * N)^2
        end
    end
end

@testset "Norms over a set with an appendix weight the points the fold counts" begin
    # Guards the norm's volume: a constant has its own value as its norm,
    # which needs the volume to count the points the sum counts.
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

@testset "regrid! keeps every leaf on a glued face at its level, and its appendix" begin
    # Guards the regrid of a set interpolated across a glue: a flag on a
    # glued face is dropped, the interior refines, the owned points of the
    # kept blocks, their appendix planes included, come through
    # unchanged, and the schedule builds again, with no refinement
    # boundary at a glue.
    forest = kinked_forest(4)
    N = forest.N
    fs = kinked_set(forest)
    sched = GhostSchedule(fs, KINKED_OPS; atlas=KINKED)
    owned(u) = let A = statearray(u, fs; appendix=true), L = TreeAMR.appendix_layout(fs)
        Dict(k => [A[Tuple(I)..., 1, b]
                   for I in CartesianIndices(ntuple(d -> 1:(N + L.dims[b][d]), 2))]
             for (b, k) in enumerate(forest.leaves) if on_glue(forest, k))
    end
    u = collect(1.0:statelength(fs))
    scatter!(fs, u)
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
