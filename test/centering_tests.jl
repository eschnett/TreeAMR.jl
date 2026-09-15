# M8a step 2: centering. A field set's values live, per dimension, at
# cell centers or at cell boundaries; the exchange, the geometry and the
# regrid transfer all follow from that one tuple.

using KernelAbstractions: @kernel, @index

const OPS2C = Operators(prolongation=2, restriction=2)

# Writes 1 at every stored index the launch reaches, so that counting
# ones says exactly which points `map_blocks!` covered.
@kernel function mark_kernel!(work, ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)
    work[ntuple(d -> I[d] + G[d], Val(D))..., 1, I[D + 1]] = 1.0
end

@testset "The familiar centerings are spellings of tuples" begin
    # A tuple rather than an enumeration of the 2^D cases, because every
    # transfer is a product of D one-dimensional stencils and the stencil
    # for dimension d depends on the centering in *that* dimension alone.
    @test cellcentered(3) == (:cell, :cell, :cell)
    @test vertexcentered(3) == (:vertex, :vertex, :vertex)
    # An x-face is normal to x, so it is vertex-like in x alone; an
    # x-edge runs *along* x, so it is the exact complement.
    @test facecentered(3, 1) == (:vertex, :cell, :cell)
    @test facecentered(3, 2) == (:cell, :vertex, :cell)
    @test edgecentered(3, 1) == (:cell, :vertex, :vertex)
    @test edgecentered(3, 2) == (:vertex, :cell, :vertex)
    for D in (1, 2, 3), d in 1:D
        @test all(e -> facecentered(D, d)[e] != edgecentered(D, d)[e], 1:D)
    end
    # In 1D a face and a vertex are the same thing, and an edge is a cell.
    @test facecentered(1, 1) == vertexcentered(1)
    @test edgecentered(1, 1) == cellcentered(1)

    @test_throws ArgumentError facecentered(2, 3)
    @test_throws ArgumentError edgecentered(2, 0)
    @test_throws "must be in 1:2" facecentered(2, 3)

    # c_d is the arithmetic's view of the same tuple.
    @test staggers((:cell, :vertex, :cell)) == (0, 1, 0)
end

@testset "A centering adds one stored plane per stagger: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=8, extents=ntuple(_ -> (0.0, 2.0), D))
    for C in allcenterings(Val(D))
        c = staggers(C)
        fs = FieldSet(forest, 2; G=1, centering=C)
        @test fs.centering == C
        @test size(fs.work) == (ntuple(d -> 8 + 2 + c[d], D)..., 2, 2^D)
        # The owned range is `N` wide whatever the centering — that is
        # what keeps the state layout uniform — and the closed range adds
        # the shared boundary plane.
        @test size(interiorview(fs, 1, 1)) == ntuple(_ -> 8, D)
        @test size(closedview(fs, 1, 1)) == ntuple(d -> 8 + c[d], D)
        @test size(closedview(fs, 1)) == (ntuple(d -> 8 + c[d], D)..., 2)
        @test statelength(fs) == 8^D * 2 * 2^D
    end
    # In a cell-centered set the two views are the same range.
    cells = FieldSet(forest, 1; G=2)
    fill!(interiorview(cells, 1, 1), 5.0)
    @test all(==(5.0), closedview(cells, 1, 1))
    @test size(closedview(cells, 1, 1)) == size(interiorview(cells, 1, 1))

    # The default is cell-centered, as every field set was through M6.
    @test FieldSet(forest, 1; G=1).centering == cellcentered(D)

    @test_throws ArgumentError FieldSet(forest, 1; G=1, centering=ntuple(_ -> :node, D))
    @test_throws ":cell or :vertex" FieldSet(forest, 1; G=1,
                                             centering=ntuple(_ -> :node, D))
    @test_throws ArgumentError FieldSet(forest, 1; G=1,
                                        centering=ntuple(_ -> :cell, D + 1))
    @test_throws "one entry per dimension" FieldSet(forest, 1; G=1,
                                                    centering=ntuple(_ -> :cell, D + 1))
    @test_throws ArgumentError FieldSet(forest, 1; G=1, centering=:vertex)
end

@testset "N >= 2G + 2c: the exchange region fits one ring of neighbours" begin
    # A vertex-like dimension needs one coarse cell more than a
    # cell-centered one, because its shared boundary plane makes the high
    # exchange region one plane longer, and a finer neighbour spans only
    # N/2 of this block's cells.
    set(N, G, C) = FieldSet(Forest((2, 2); N=N), 1; G=G, centering=C)
    @test set(8, 4, cellcentered(2)) isa FieldSet
    @test_throws ArgumentError set(8, 4, vertexcentered(2))
    @test set(8, 3, vertexcentered(2)) isa FieldSet
    @test_throws "N must be >= 2G + 2c" set(8, 4, vertexcentered(2))
    # Per dimension: wide in a cell-centered x, at the limit in a
    # vertex-like y.
    @test set(8, (4, 3), (:cell, :vertex)) isa FieldSet
    @test_throws ArgumentError set(8, (4, 4), (:cell, :vertex))
    # G = 0 along a stagger is legal and is the point of the whole move:
    # a face field still has its shared plane, and nothing else.
    @test size(set(8, (0, 1), (:vertex, :cell)).work) == (9, 10, 1, 4)
end

@testset "Positions follow the centering: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=8, extents=ntuple(_ -> (0.0, 2.0), D))
    G = 2
    for C in allcenterings(Val(D))
        c = staggers(C)
        fs = FieldSet(forest, 1; G=G, centering=C)
        for b in 1:nblocks(fs)
            k = blockkey(fs, b)
            origin = block_origin(forest, k)
            h = spacing(forest, k)
            # The first owned point sits at the block's lower corner in a
            # vertex-like dimension and half a cell in from it in a
            # cell-centered one ...
            first = coordinates(fs, b, ntuple(_ -> G + 1, D))
            @test all(d -> first[d] ≈ origin[d] + (c[d] == 1 ? 0 : h / 2), 1:D)
            # ... and the last point of the closed range is the block's
            # upper corner exactly where it is vertex-like.
            last = coordinates(fs, b, ntuple(d -> G + forest.N + c[d], D))
            ext = block_extent(forest, k)
            @test all(d -> last[d] ≈ (c[d] == 1 ? ext[d][2] : ext[d][2] - h / 2), 1:D)
        end
        # Neighbouring points are one spacing apart, ghosts included.
        h = spacing(forest, blockkey(fs, 1))
        for d in 1:D
            lo = coordinates(fs, 1, ntuple(_ -> 1, D))
            hi = coordinates(fs, 1, ntuple(e -> e == d ? 2 : 1, D))
            @test hi[d] - lo[d] ≈ h
        end
    end
end

@testset "fill_by_coordinates! fills the owned range, at its own points: D=$D" for
        D in (1, 2)
    # The M5 claim (thread_tests.jl), now with the centering varying as
    # well: the kernel must land on the same floating-point position
    # `coordinates` gives, not merely a close one, or the boundary hook
    # and the exchange would disagree at the last bit -- and the M5
    # digests depend on it. The two expressions cancel `G` differently
    # (the kernel never forms it, `coordinates` subtracts it) and now
    # the offset they subtract depends on the centering too.
    forest = nested_forest(Val(D); N=8)
    f = (x, v) -> sum(x) + 100v
    for C in allcenterings(Val(D)), G in (1, 2)
        fs = FieldSet(forest, 2; G=G, centering=C)
        fill_by_coordinates!(f, fs)
        owned = CartesianIndices(ntuple(_ -> (G + 1):(G + forest.N), D))
        identical = all(1:nblocks(fs)) do b
            all(v -> all(idx -> blockview(fs, b, v)[idx] ===
                                f(coordinates(fs, b, Tuple(idx)), v), owned),
                1:fs.nvars)
        end
        @test identical
        # The shared plane is *not* owned, so it is left to the exchange
        # exactly as a ghost is.
        c = staggers(C)
        if any(==(1), c)
            plane = ntuple(d -> c[d] == 1 ? (G + forest.N + 1) : (G + 1), D)
            @test iszero(fs.work[plane..., 1, 1])
        end
    end
end

@testset "map_blocks! over the closed range covers the shared plane: D=$D" for
        D in (1, 2, 3)
    # A flux has N+1 faces per dimension, not N: the extra one is the
    # block's own high face, which lives on the shared plane.
    forest = Forest(ntuple(_ -> 2, D); N=4, periodic=ntuple(_ -> true, D))
    for C in allcenterings(Val(D))
        c = staggers(C)
        fs = FieldSet(forest, 1; G=1, centering=C)
        map_blocks!(mark_kernel!, fs, fs.work, Val(D), Val(fs.G); closed=true)
        @test count(==(1.0), fs.work) == nblocks(fs) * prod(ntuple(d -> 4 + c[d], D))
        @test all(==(1.0), closedview(fs, 1, 1))

        fill!(fs.work, 0.0)
        map_blocks!(mark_kernel!, fs, fs.work, Val(D), Val(fs.G))
        @test count(==(1.0), fs.work) == nblocks(fs) * 4^D
        @test all(==(1.0), interiorview(fs, 1, 1))
    end
end

@testset "The flagging kernel forms the same position: D=$D" for D in (1, 2, 3)
    # `firing_kernel!` hands the application's criterion a position, and
    # it must be the one `coordinates` gives -- for every centering, and
    # bit for bit, or a criterion written against positions would fire on
    # different cells than the same criterion written against the
    # geometry. `fill_by_coordinates!` is already under test for exactly
    # that agreement, so comparing against what it stored is transitive.
    forest = nested_forest(Val(D); N=8)
    for C in allcenterings(Val(D))
        fs = FieldSet(forest, 1; G=1, centering=C)
        fill_by_coordinates!((x, v) -> x[1], fs)
        agrees(work, idx, b, x) = work[idx..., 1, b] == x[1]
        @test all(g -> g[1] == forest.N^D, firing_boxes(agrees, fs))
    end
end

@testset "The exchange is exact to degree p-1 for every centering: D=$D" for
        D in (1, 2, 3)
    # The M2 claim, now for all 2^D centerings: face, edge, corner and
    # three-level transfers all reproduce a polynomial of degree p-1
    # exactly and one of degree p not at all. `max_deviation` compares
    # every stored point, so this covers the shared boundary plane and
    # the domain's upper boundary plane that the hook fills.
    forest = nested_forest(Val(D); N=8)
    @test isbalanced(forest)
    @test length(unique(level.(forest.leaves))) >= 3

    for C in allcenterings(Val(D))
        for (p, G) in ((2, 1), (4, 2))
            ops = Operators(prolongation=p, restriction=p)
            # All three transfer kinds really occur, so exactness is not
            # vacuous.
            counts = transfer_counts(GhostSchedule(FieldSet(forest, 1; G=G,
                                                            centering=C), ops))
            @test counts[:copy] > 0 && counts[:restrict] > 0 && counts[:prolong] > 0

            @test exchange_error(forest, ops, makepoly(D, p - 1); G=G,
                                 centering=C) < 1e-10
            @test exchange_error(forest, ops, makepoly(D, p); G=G,
                                 centering=C) > 1e-8
        end
    end
end

@testset "The exchange partitions the stored points for every centering: D=$D" for
        D in (1, 2, 3)
    # Every point outside the owned range -- ghosts *and* the shared
    # boundary plane -- is written exactly once, and no owned point is
    # touched. With G = 0 along a stagger that leaves only the shared
    # plane, which is the whole exchange a second-order face field has.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> isodd(D), D))
    N = forest.N
    for C in allcenterings(Val(D))
        c = staggers(C)
        for G in (ntuple(_ -> 1, D), ntuple(d -> c[d] == 1 ? 0 : 1, D))
            fs = FieldSet(forest, 1; G=G, centering=C)
            counts = write_counts(GhostSchedule(fs, OPS2C))
            owned = ntuple(d -> (G[d] + 1):(G[d] + N), D)
            ok = true
            for b in 1:nleaves(forest)
                slab = view(counts, ntuple(_ -> Colon(), D)..., b)
                for idx in CartesianIndices(slab)
                    inside = all(d -> Tuple(idx)[d] in owned[d], 1:D)
                    slab[idx] == (inside ? 0 : 1) || (ok = false)
                end
            end
            @test ok
            stored = prod(ntuple(d -> N + 2G[d] + c[d], D))
            @test sum(counts) == nleaves(forest) * (stored - N^D)
        end
    end
end

@testset "Vertex restriction is injection: D=$D" for D in (1, 2, 3)
    # Injection is exact for *any* data, which is what tells it apart
    # from an averaging or interpolating restriction. So the blocks are
    # filled with values that depend on nothing but a point's exact
    # position -- no polynomial structure at all -- and every exchange
    # point that coincides with an owned point of a *finer* block must
    # come back bit for bit. The oracle finds the owner from the exact
    # rational box geometry in `oracles.jl`, not from the package's
    # neighbour search.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> true, D))
    @test length(unique(level.(forest.leaves))) >= 3

    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(D))
    fill_arbitrary!(fs)
    fill_ghosts!(fs, GhostSchedule(fs, OPS2C))
    nfiner, nsame, bad = injection_report(fs)
    @test nfiner > 0                       # restriction from finer blocks happens
    @test nsame > 0                        # so do same-level copies
    @test bad == 0                         # and both reproduce the owner exactly

    # A cell-centered set has no coincident points across levels at all:
    # a coarse cell center falls on the interface between two fine cells,
    # which is why its restriction has to average and cannot inject.
    cells = FieldSet(forest, 2; G=1)
    fill_arbitrary!(cells)
    fill_ghosts!(cells, GhostSchedule(cells, OPS2C))
    cfiner, csame, cbad = injection_report(cells)
    @test cfiner == 0
    @test csame > 0 && cbad == 0           # copies are still exact
end

@testset "G >= p/2 - 1 is the bound along a stagger" begin
    # Prolongation into a vertex-like dimension reads p/2 - 1 planes past
    # the source's shared plane, against p/2 past the interface in a
    # cell-centered one -- one ghost layer fewer, and none at all at
    # order 2.
    vset(G, D=1) = FieldSet(Forest(ntuple(_ -> 2, D); N=8), 1; G=G,
                            centering=vertexcentered(D))
    ops(p) = Operators(prolongation=p, restriction=p)

    @test check_operators(vset(0), ops(2)) === nothing
    @test check_operators(vset(1), ops(4)) === nothing
    @test check_operators(vset(2), ops(6)) === nothing
    @test_throws ArgumentError check_operators(vset(0), ops(4))
    @test_throws ArgumentError check_operators(vset(1), ops(6))
    @test_throws "vertex-like dimension needs G >= 1" check_operators(vset(0), ops(4))

    # The same order in a cell-centered dimension needs one more, and the
    # message names the dimension.
    @test_throws ArgumentError check_operators(
        FieldSet(Forest((2, 2); N=8), 1; G=1, centering=(:vertex, :cell)), ops(4))
    @test check_operators(
        FieldSet(Forest((2, 2); N=8), 1; G=(1, 2), centering=(:vertex, :cell)),
        ops(4)) === nothing

    # A cell-centered dimension with no ghosts has nothing to exchange
    # and is still refused; a vertex-like one has its shared plane and is
    # accepted. That is the step-1 rule, relaxed exactly as far as the
    # geometry allows.
    @test_throws "no ghosts" GhostSchedule(FieldSet(Forest((2,); N=8), 1; G=0), OPS2C)
    @test GhostSchedule(vset(0), OPS2C) isa GhostSchedule

    # One less than the bound fails; one more passes the exactness test.
    forest = nested_forest(Val(1); N=8)
    @test exchange_error(forest, ops(4), makepoly(1, 3); G=1,
                         centering=vertexcentered(1)) < 1e-10
    @test exchange_error(forest, ops(4), makepoly(1, 4); G=1,
                         centering=vertexcentered(1)) > 1e-8
end

@testset "The conservative family is refused along a vertex-like dimension" begin
    # What a staggered quantity stores is an average along its cell-like
    # dimensions and a *point value* along its vertex-like ones, so there
    # the family has nothing to conserve and would merely interpolate --
    # at an even order that its odd orders do not name. Refused rather
    # than guessed at.
    cons = Operators(prolongation=3, restriction=2, family=Conservative)
    forest = Forest((2, 2); N=8)
    face = FieldSet(forest, 1; G=1, centering=facecentered(2, 1))
    @test_throws ArgumentError GhostSchedule(face, cons)
    @test_throws "not defined along a vertex-like dimension" GhostSchedule(face, cons)
    @test_throws "nothing to conserve" check_operators(face, cons)
    @test_throws ArgumentError check_operators(
        FieldSet(forest, 1; G=1, centering=vertexcentered(2)), cons)
    # A cell-centered set is of course still fine.
    @test GhostSchedule(FieldSet(forest, 1; G=1), cons) isa GhostSchedule
end

@testset "A schedule belongs to a centering: D=$D" for D in (1, 2)
    # Every target range, stored extent and one-dimensional operator in a
    # schedule differs between a cell-centered and a vertex-like
    # dimension, so a schedule built for one layout is refused for the
    # other -- and nothing else would have caught it.
    forest = Forest(ntuple(_ -> 2, D); N=8, periodic=ntuple(_ -> true, D))
    cells = FieldSet(forest, 1; G=1)
    verts = FieldSet(forest, 1; G=1, centering=vertexcentered(D))
    csched, vsched = GhostSchedule(cells, OPS2C), GhostSchedule(verts, OPS2C)
    @test csched.centering == cellcentered(D)
    @test vsched.centering == vertexcentered(D)
    @test fill_ghosts!(verts, vsched) === verts
    @test_throws ArgumentError fill_ghosts!(verts, csched)
    @test_throws "but this schedule was built for" fill_ghosts!(verts, csched)
    @test_throws ArgumentError fill_ghosts!(cells, vsched)

    # The forest form spells the layout out and produces the same thing.
    byforest = GhostSchedule(forest, OPS2C; G=1, centering=vertexcentered(D))
    @test byforest.centering == verts.centering
    @test sprint(show, byforest) == sprint(show, vsched)

    # And `regrid!` says the same thing before it moves anything.
    @test_throws "pair each field set with its own schedule" regrid!(
        forest, verts => csched; flags=fill(Keep, nleaves(forest)))
end

@testset "The domain's upper boundary plane belongs to the hook: D=$D" for D in (1, 2)
    # Ownership is half-open, so at a non-periodic boundary the domain's
    # *upper* boundary points belong to nobody: they are the first plane
    # of an outward-facing region, and the hook fills them. The lower
    # ones are owned and evolved. A vertex-centered application with
    # physical boundaries sees this asymmetry; a periodic domain has none.
    forest = Forest(ntuple(_ -> 2, D); N=8)
    G = 1
    fs = FieldSet(forest, 1; G=G, centering=vertexcentered(D))
    schedule = GhostSchedule(fs, OPS2C)

    # The high-side outward regions start at the shared plane, one plane
    # further in than a ghost slab; the low-side ones are plain ghosts.
    highs = filter(r -> r.direction == ntuple(d -> d == 1 ? 1 : 0, D),
                   schedule.boundaries)
    lows = filter(r -> r.direction == ntuple(d -> d == 1 ? -1 : 0, D),
                  schedule.boundaries)
    @test !isempty(highs) && !isempty(lows)
    @test all(r -> first(r.region.indices[1]) == G + forest.N + 1, highs)
    @test all(r -> length(r.region.indices[1]) == G + 1, highs)
    @test all(r -> r.region.indices[1] == 1:G, lows)

    # And the hook really does write that plane: with it, the linear
    # field is exact everywhere; without it, the upper plane is left at
    # its initial zero.
    f = makepoly(D, 1)
    @test exchange_error(forest, OPS2C, f; nvars=1, G=G,
                         centering=vertexcentered(D)) < 1e-12
    fill_by_coordinates!(f, fs)
    fill_ghosts!(fs, schedule)
    topblock = findfirst(k -> block_extent(forest, k)[1][2] == 2.0, forest.leaves)
    @test iszero(fs.work[ntuple(d -> d == 1 ? G + forest.N + 1 : G + 1, D)..., 1,
                         topblock])
end

@testset "The regrid transfer is exact for every centering: D=$D" for D in (1, 2, 3)
    # Refine, coarsen and copy all move a polynomial of degree p-1 onto
    # the new mesh exactly, whatever the centering -- coarsening a
    # vertex-like dimension by injection from the children's even points,
    # refining it by the same prolongation the exchange uses. The
    # transfer fills the *owned* range only, so the shared plane and the
    # ghosts of a fresh block need a ghost fill afterwards before
    # anything may be compared there.
    N = D == 3 ? 4 : 8
    f = makepoly(D, 1)
    hook = boundary_by_coordinates(f)
    for C in allcenterings(Val(D))
        forest = Forest(ntuple(_ -> 2, D); N=N, extents=ntuple(_ -> (0.0, 1.0), D))
        refine!(forest, forest.leaves[1])
        balance!(forest)
        fs = FieldSet(forest, 1; G=1, centering=C)
        fill_by_coordinates!(f, fs)
        fill_ghosts!(fs, GhostSchedule(fs, OPS2C); boundary=hook)

        # One pass that refines the coarse half and coarsens the fine
        # one, so every kind of transfer occurs.
        flags = flag_blocks(forest) do b, k
            level(k) == 0 ? Refine : Coarsen
        end
        @test regrid!(forest, fs => GhostSchedule(fs, OPS2C); flags=flags,
                      boundary=hook)
        fill_ghosts!(fs, GhostSchedule(fs, OPS2C); boundary=hook)
        @test max_deviation(fs, f) < 1e-12
    end
end
