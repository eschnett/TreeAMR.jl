# Ghosts interpolated across a glue (M20): "Finite differences across a
# glue" in CODE.md. The forests and atlases are in `helpers/glue.jl`.

using TreeAMR: GluePlan, chart_aligned

# A polynomial of degree five in each logical coordinate, different in
# each patch, which a six-point stencil reproduces exactly.
glue_poly(ξ, p) = 0.5 + p * ξ[1]^5 + 2ξ[1]^3 * ξ[2]^4 - (p + 1) * ξ[2]^5 + ξ[1] * ξ[2]

@pending 8 "Interpolated ghosts across a kinked glue reproduce a patch's polynomial exactly" begin
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

@pending 8 "Interpolated ghosts across a kinked glue converge at the glue order" begin
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

@pending 8 "On the annulus interpolated ghosts are the copied ones bit for bit" begin
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

@pending 8 "An alignment the maps do not have is refused" begin
    # Guards the declaration's check: a wrong declaration would read a
    # node that is not where the ghost lies.
    struct WronglyAligned end
    TreeAMR.chart_locate(::WronglyAligned, x) = TreeAMR.chart_locate(KINKED, x)
    TreeAMR.chart_map(::WronglyAligned, ξ, p) = TreeAMR.chart_map(KINKED, ξ, p)
    TreeAMR.chart_aligned(::WronglyAligned, p, q) = (true, true)
    fs = kinked_set(kinked_forest(1))
    @test_throws "not at a node" GhostSchedule(fs, KINKED_OPS; atlas=WronglyAligned())
end

@pending 8 "A set interpolated across a glue refuses what it cannot fill" begin
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

@pending 8 "Closed ownership adds each patch's glued high face planes to the state, once" begin
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

@pending 8 "The appendix round-trips through scatter!, gather! and statearray" begin
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

@pending 8 "map_blocks! with appendix = true writes every point the state owns" begin
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

@pending 8 "A fold counts every glued face point once, at its lowest patch" begin
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

@pending 8 "Norms over a set with an appendix weight the points the fold counts" begin
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

@pending 8 "regrid! keeps every leaf on a glued face at its level, and its appendix" begin
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

@pending 8 "Reading a state through its appendix allocates nothing" begin
    # Guards the inlining of AppendixArray's accessors: a call per point
    # allocated its index tuples, 150 B and 70 ns an entry, in every fold
    # and norm over a state vector and every right-hand side written
    # through `appendix = true`.
    fs = kinked_set(kinked_forest(2))
    u = collect(1.0:statelength(fs))
    A = statearray(u, fs; appendix=true)
    function total(A, N, nb)
        s = 0.0
        for b in 1:nb, j in 1:N, i in 1:N
            s += A[i, j, 1, b]
            A[i, j, 1, b] = s
        end
        return s
    end
    total(A, fs.forest.N, nblocks(fs))
    @test (@allocated total(A, fs.forest.N, nblocks(fs))) == 0
    @test eltype(A) === Float64
end

@pending 8 "fill_by_coordinates! fills the appendix in every form" begin
    # Guards the initial-data path of a set with an appendix: the owned
    # high face planes are state too, and were left at zero. Each form of
    # the callback fills every owned point with the same numbers.
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
        A = statearray(u, fs; appendix=true)
        L = TreeAMR.appendix_layout(fs)
        @test all(1:nblocks(fs)) do b
            k = blockkey(fs, b)
            o, h = block_origin(forest, k), spacing(forest, k)
            all(CartesianIndices(ntuple(d -> 1:(N + L.dims[b][d]), 2))) do I
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

@pending 8 "A fold counts mixed centerings once across a glue that turns the axes" begin
    # Guards the search for a point's copies: a glue that sends dimension
    # 1 of one patch onto dimension 2 of the other carries a vertex-like
    # face onto a cell-like one, where the set has no point, and the fold
    # counted a copy there that does not exist.
    n, N = 2, 8
    forest = Forest([(roots=(n, n),), (roots=(n, n),)]; N=N,
                    glue=patch_glue_both((1, 1, 2), (2, 2, 1), (2, -1)))
    for (cen, shared) in (((:vertex, :vertex), n * N), ((:vertex, :cell), 0),
                          ((:cell, :vertex), 0))
        fs = FieldSet(forest, 1; G=3, centering=cen, glue=:interpolate)
        @test mesh_fold((acc, p) -> acc + 1, +, 0, fs, zeros(statelength(fs))) ==
              statelength(fs) - shared
    end
end

@pending 8 "A glue stencil reads only what phase 1 and the hook have filled" begin
    # Guards the overhang rule. A ghost that lands deep in the patch across
    # reads its source block's high plane, which a coarser neighbor fills
    # by prolongation only in phase 2, after the glue's stage: the build
    # refuses it rather than leave the ghost a fill behind. A finer
    # neighbor's ghosts are restricted in phase 1, and are read.
    ops = Operators(prolongation=4, restriction=4, glue=6)
    for (a, refused) in ((0.22, true), (0.8, false))
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

@pending 8 "A smooth solution's derivatives converge at the operator's order across a kink" begin
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

@pending 8 "isowned names the points a block owns, and a multi-set state reaches its appendix" begin
    # Guards the closed range's bookkeeping a right-hand side relies on:
    # the points `isowned` accepts are exactly the state's, so a kernel
    # that skips the others computes every owned point once; and a set
    # with an appendix inside a state vector of several sets reads and
    # writes it through the same `appendix` keyword.
    forest = kinked_forest(2)
    N = forest.N
    fs = kinked_set(forest)
    other = FieldSet(forest, 2; G=1)
    u = collect(1.0:statelength(fs))
    A = statearray(u, fs; appendix=true)
    owned = count(isowned(A, Tuple(I)..., 1, b)
                  for b in 1:nblocks(fs) for I in CartesianIndices((1:(N + 1), 1:(N + 1))))
    @test owned == statelength(fs)
    @test isowned(statearray(u, fs), N, N, 1, 1) && !isowned(statearray(u, fs), N + 1, 1, 1, 1)
    sets = (other, fs)
    w = statevector(sets)
    W = statearray(w, sets, 2; appendix=true)
    @test W isa TreeAMR.AppendixArray
    for b in 1:nblocks(fs), I in CartesianIndices((1:(N + 1), 1:(N + 1)))
        isowned(W, Tuple(I)..., 1, b) && (W[Tuple(I)..., 1, b] = A[Tuple(I)..., 1, b])
    end
    @test stateview(w, sets, 2) == u
    @test statearray(w, sets, 2) isa AbstractArray{Float64,4}
end

# A polynomial of degree `k - 1` in each logical coordinate, different in
# each patch, which a `k`-point stencil reproduces exactly.
glue_poly_k(k) = (ξ, p) -> 0.5 + p * ξ[1]^(k - 1) - (p + 1) * ξ[2]^(k - 1) +
                           ξ[1]^(k ÷ 2) * ξ[2]^(k ÷ 2 - 1)

@testset "Interpolated ghosts are exact for every centering and order" begin
    # Guards the stencil's window and weights away from the one layout the
    # other tests use: a cell-like dimension has no point on a face and no
    # appendix, a mixed set has one along one dimension only, and the
    # window's clamping and centering change with the order. Each patch's
    # polynomial of degree `k - 1` comes back to rounding at every ghost
    # across the kinked torus's glues and wrap, patch corners included.
    for C in ((:cell, :cell), (:vertex, :cell), (:cell, :vertex), (:vertex, :vertex)),
        k in (2, 4, 6)
        vertexlike(C) && (pending(8, "interpolated ghosts exact, $C, k = $k"); continue)
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
        # Each patch's closed box, `nN + 1` points along a vertex-like
        # dimension, beyond its `(nN)²` cores.
        @test statelength(fs) - TreeAMR.corelength(fs) ==
              2 * (prod(16 + (C[d] === :vertex) for d in 1:2) - 16^2)
    end
end
