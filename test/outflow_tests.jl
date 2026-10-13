# Outflow faces: zero-gradient extrapolation as transfers.
#
# An outflow face is a property of the domain, like a reflecting one, and
# the schedule fills its ghosts the same way: the tangential stencil of
# the source's kind, with its target rows clamped onto the nearest owned
# row instead of mirrored, and no parity sign. Every claim is stated
# against an oracle that knows only the geometry — which stored index a
# ghost repeats — never the schedule's group keys.

using TreeAMR: wall_direction, OuterFace, ReflectingFace, OutflowFace, reflecting_faces,
               outflow_faces, facekind

const OUTFLOW_KINDS = (:outflow_lo, :outflow_hi, :outflow_both)

"""
The stored index whose value the ghost `idx` of block `b` holds: `idx`
itself clamped into the owned range along every dimension in which it
lies beyond an outflow face of the domain — or `nothing` when it lies
beyond an outer face, which makes it the boundary hook's, holding the
data at its own position. Beyond a periodic or a reflecting face the
data themselves are periodic or of definite parity, so the index stays.
"""
function outflow_source(fs::FieldSet{T,D}, b, idx::NTuple{D,Int}) where {T,D}
    forest = fs.forest
    N = forest.N
    ext = block_extent(forest, blockkey(fs, b))
    hook = false
    j = ntuple(D) do d
        G = fs.G[d]
        lo = idx[d] <= G && ext[d][1] == forest.extents[d][1]
        hi = idx[d] >= G + N + 1 && ext[d][2] == forest.extents[d][2]
        (lo || hi) && !forest.periodic[d] || return idx[d]
        kind = facekind(forest, d, lo ? 1 : 2)
        kind == OuterFace && (hook = true)
        return kind == OutflowFace ? (lo ? G + 1 : G + N) : idx[d]
    end
    return hook ? nothing : j
end

"""
Fill a forest with `f` after setting every stored value to `NaN`,
exchange ghosts once, and report `(nnan, worst)`: the values still `NaN`,
and the worst deviation of a stored value from `f` at its source — its
own position, or the clamped one beyond an outflow face (see
[`outflow_source`](@ref)).
"""
function outflow_deviation(forest, C, f, parity; p=4)
    ops = Operators(prolongation=p, restriction=p)
    fs = FieldSet(forest, 2; G=ghosts_for(C, p), centering=C, parity=parity)
    fill!(fs.work, NaN)
    fill_by_coordinates!(f, fs)
    fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=boundary_by_coordinates(f))
    nnan = count(isnan, fs.work)
    worst = 0.0
    for b in 1:nblocks(fs), v in 1:2
        for idx in CartesianIndices(size(fs.work)[1:length(C)])
            j = outflow_source(fs, b, Tuple(idx))
            x = coordinates(fs, b, j === nothing ? Tuple(idx) : j)
            worst = max(worst, abs(fs.work[idx, v, b] - f(x, v)))
        end
    end
    return nnan, worst
end

@testset "Outflow faces are refused where they cannot mean anything" begin
    # An outflow face declared where it has no meaning must be refused
    # with the reason, not discovered later as a wrong ghost.
    @test_throws "both periodic and outflow" Forest((2, 2); N=8, periodic=(true, false),
                                                     outflow=((true, false),
                                                              (false, false)))
    @test_throws "both reflecting and outflow" Forest((2, 2); N=8,
                                                       reflecting=((true, false),
                                                                   (false, false)),
                                                       outflow=((true, true),
                                                                (false, false)))
    @test_throws "has an outflow face" Forest((2, 2, 1); N=8, rotating=(1, 2),
                                              outflow=((false, true), (false, false),
                                                       (false, false)))
    # Outside the plane of the rotation an outflow face is legal, as a
    # reflecting one is.
    @test outflow_faces(Forest((2, 2, 1); N=8, rotating=(1, 2),
                               outflow=((false, false), (false, false),
                                        (false, true)))) ==
          ((false, false), (false, false), (false, true))

    forest = Forest((2, 2); N=8, reflecting=((true, false), (false, false)),
                    outflow=((false, true), (true, true)))
    @test reflecting_faces(forest) == ((true, false), (false, false))
    @test outflow_faces(forest) == ((false, true), (true, true))
    @test facekind(forest, 1, 1) == ReflectingFace
    # The properties physics packages read, as the keywords' pairs: the
    # forest stored `reflecting` itself before the outflow faces came.
    @test forest.reflecting == ((true, false), (false, false))
    @test forest.outflow == ((false, true), (true, true))
    @test :reflecting in propertynames(forest) && :outflow in propertynames(forest)
    @test facekind(forest, 1, 2) == OutflowFace
    @test outflow_faces(Forest((2, 2); N=8)) == ((false, false), (false, false))
    # An outflow face needs no parity: whether a variable changes sign
    # is a question only a mirror asks.
    plain = Forest((2, 2); N=8, outflow=((true, true), (true, true)))
    @test FieldSet(plain, 2; G=1).parity === nothing
end

@testset "The wall split finds outflow faces as it finds reflecting ones" begin
    # The schedule looks for a clamped ghost's source in direction δ′,
    # so a wrong split sends it to the wrong block, and a wrong kind
    # mirrors where it should clamp.
    R, F, O = ReflectingFace, OutflowFace, OuterFace
    forest = Forest((2, 2); N=8, reflecting=((true, false), (false, false)),
                    outflow=((false, true), (true, false)))
    at(pos) = forest.leaves[findfirst(k -> root_position(forest, k.root) == pos,
                                      forest.leaves)]
    @test wall_direction(forest, at((0, 0)), (-1, -1)) == ((0, 0), (R, F))
    @test wall_direction(forest, at((0, 0)), (0, -1)) == ((0, 0), (O, F))
    @test wall_direction(forest, at((1, 0)), (1, 1)) == ((0, 1), (F, O))
    # The high y face is outer: not a wall, even where it leaves.
    @test wall_direction(forest, at((1, 1)), (1, 1)) == ((0, 1), (F, O))
    @test wall_direction(forest, at((0, 1)), (0, 1)) == ((0, 1), (O, O))
end

@testset "An outflow ghost repeats the nearest owned point: D=$D" for D in (1, 2, 3)
    # The definition, for data that vary along the outflow dimension: on
    # an unrefined brick every transfer is a copy, so each ghost beyond
    # an outflow face holds exactly the value of the owned point it is
    # clamped onto — and that point, not its mirror image, which data
    # varying along the dimension tell apart. Every centering, and the
    # outflow faces against periodic, outer and reflecting neighbors,
    # edges and corners included.
    others = (:periodic, :outer, :reflect_lo)
    cases = [ks for ks in Iterators.product(ntuple(_ -> (OUTFLOW_KINDS..., others...),
                                                   D)...)
             if any(k -> k in OUTFLOW_KINDS, ks)]
    centerings = allcenterings(Val(D))
    bad = []
    for (i, kinds) in enumerate(cases)
        forest = faces_forest(kinds; refined=false)
        # A general polynomial along the outflow dimensions, as along an
        # outer one; the reflecting data keep their parity.
        f, parity = parity_data(Float64, map(k -> k in OUTFLOW_KINDS ? :outer : k, kinds),
                                4)
        Cs = D == 3 ? [centerings[mod1(i, length(centerings))]] : centerings
        for C in Cs
            nnan, worst = outflow_deviation(forest, C, f, parity)
            (nnan == 0 && worst < 1e-12) || push!(bad, (kinds, C, nnan, worst))
        end
    end
    @test isempty(bad)
    isempty(bad) || foreach(println, bad)
end

@testset "Every ghost is defined and none is read undefined at outflow faces: D=$D" for
        D in (1, 2, 3)
    # The ordering failure the reflecting faces guard against, at an
    # outflow face: on a mesh refined into the faces, a clamped ghost may
    # be the block's own prolongated ghost, or a coarse neighbor's
    # clamped one that a fine block's prolongation reads. With every
    # stored value `NaN` beforehand, an unwritten ghost or a read before
    # the write stays `NaN`. The data are constant along the outflow
    # dimensions (see `parity_data`), so every value is also exact.
    others = (:periodic, :outer, :reflect_lo, :reflect_hi)
    cases = [ks for ks in Iterators.product(ntuple(_ -> (OUTFLOW_KINDS..., others...),
                                                   D)...)
             if any(k -> k in OUTFLOW_KINDS, ks)]
    centerings = allcenterings(Val(D))
    bad = []
    for (i, kinds) in enumerate(cases)
        Cs = D == 3 ? [centerings[mod1(i, length(centerings))]] : centerings
        for C in Cs
            nnan, worst = undefined_ghosts(kinds, C)
            (nnan == 0 && worst < 1e-10) || push!(bad, (kinds, C, nnan, worst))
        end
    end
    @test isempty(bad)
    isempty(bad) || foreach(println, bad)
end

@testset "Outflow and periodic faces need no boundary hook: D=$D" for D in (1, 2, 3)
    # Outflow faces belong to the schedule, so a domain bounded by them
    # alone hands the hook nothing, and a fill without one is complete.
    kinds = ntuple(d -> d == 1 ? :outflow_both : :periodic, D)
    forest = faces_forest(kinds)
    for C in (cellcentered(D), vertexcentered(D))
        fs = FieldSet(forest, 1; G=ghosts_for(C, 4), centering=C)
        schedule = GhostSchedule(fs, OPS4)
        @test isempty(schedule.boundaries)
        fill!(fs.work, NaN)
        fill_by_coordinates!((x, v) -> 1.0, fs)
        fill_ghosts!(fs, schedule)
        @test all(==(1.0), fs.work)
    end
end

@testset "The regrid transfer is exact at an outflow face: D=$D" for D in (1, 2, 3)
    # A prolongation onto a fresh child at the face reads its parent's
    # clamped ghosts, which `regrid!` fills first; ghosts left unfilled
    # there would put a wrong value into the child's interior.
    N = D == 3 ? 4 : 8
    kinds = ntuple(d -> isodd(d) ? :outflow_lo : :reflect_hi, D)
    for C in allcenterings(Val(D))
        forest = faces_forest(kinds; N=N)
        f, parity = parity_data(Float64, kinds, 2)
        hook = boundary_by_coordinates(f)            # for the outer faces
        fs = FieldSet(forest, 2; G=1, centering=C, parity=parity)
        fill!(fs.work, NaN)
        fill_by_coordinates!(f, fs)
        flags = flag_blocks(forest) do b, k
            level(k) == 0 ? Refine : level(k) == maxlevel(forest) ? Coarsen : Keep
        end
        @test regrid!(forest, fs => GhostSchedule(fs, OPS2C); flags=flags,
                      boundary=hook)
        fill_ghosts!(fs, GhostSchedule(fs, OPS2C); boundary=hook)
        @test !any(isnan, fs.work)
        @test max_deviation(fs, f) < 1e-12
    end
end
