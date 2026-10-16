# Symmetric walls (TreeGRRMHD's AMR-8).
#
# Ownership is half-open, so along a vertex-like dimension the domain's
# high wall plane is filled and its low one evolved, and a closed box is
# not mirror-symmetric. `FieldSet(…; walls = :symmetric)` has the fill
# write the owned low wall too, by the rule the high wall gets. The
# oracles know only the geometry: a block's mirror image under the
# point reflection of the domain, and which owned planes lie on a low
# domain face (`low_wall_planes`).

# Point symmetric about the centre of a `faces_forest` box of side `L`,
# so even under every reflection of a dimension through the centre, and
# with no polynomial structure.
point_symmetric(D, L) = (x, v) -> prod(d -> cos(1.7 * (x[d] - L / 2)) +
                                            0.3 * (x[d] - L / 2)^2, 1:D) + v

# The worst difference between every stored value of `fs` and the value
# its mirror block holds at the mirrored index, under the point
# reflection `x → L − x` of the domain; and how many values were
# compared. A block's mirror is the leaf whose box is its own reflected,
# and stored index `i` maps to `n + 1 − i`, `n` the stored extent, since
# the reflection turns a block's low ghosts into the mirror's high ones.
function point_mirror_error(fs::FieldSet{T,D}, L) where {T,D}
    forest = fs.forest
    lower(k) = (level(k), ntuple(d -> block_extent(forest, k)[d][1], D))
    upper(k) = (level(k), ntuple(d -> L - block_extent(forest, k)[d][2], D))
    index = Dict(lower(blockkey(fs, b)) => b for b in 1:nblocks(fs))
    n = ntuple(d -> size(fs.work, d), D)
    worst, count = 0.0, 0
    for b in 1:nblocks(fs)
        m = get(index, upper(blockkey(fs, b)), nothing)
        m === nothing && return nothing
        for v in 1:fs.nvars, idx in CartesianIndices(n)
            j = ntuple(d -> n[d] + 1 - idx[d], D)
            worst = max(worst, abs(fs.work[Tuple(idx)..., v, b] - fs.work[j..., v, m]))
            count += 1
        end
    end
    return worst, count
end

# One kind for both faces of every dimension, so that the box is
# symmetric under its point reflection: the hook's, mirrored, clamped,
# or a mix of the three.
const WALL_BOXES = Dict(1 => [(:outer,), (:reflect_both,), (:outflow_both,)],
                        2 => [(:outer, :outer), (:reflect_both, :reflect_both),
                              (:outflow_both, :outflow_both), (:outflow_both, :reflect_both)],
                        3 => [(:outer, :reflect_both, :outflow_both)])

staggered_or_vertex(D) = filter(C -> any(==(:vertex), C), allcenterings(Val(D)))

@testset "A closed box fills mirror-symmetrically with symmetric walls: D=$D" for
        D in (1, 2, 3)
    # The claim of AMR-8: symmetric data fill symmetrically, both walls
    # included, at every face kind. On a uniform box every value is a
    # copy, a hook value, a mirror, a clamp or a derived wall, and the
    # fill is symmetric bit for bit — but where two derived walls meet, in
    # a corner of two reflecting vertex-like dimensions, whose four-term
    # stencil the two corners sum in mirrored order. A regrid that refines every block
    # keeps the box uniform, and the fill after it symmetric to rounding:
    # a prolongation sums its window in the mirrored order on the two
    # sides. Half-open walls are the negative control: the low wall keeps
    # the data while the high one is derived, an error of the scheme's
    # order. (On a mesh refined somewhere a vertex-like set is not mirror
    # symmetric under either kind of wall: half-open ownership puts a
    # coarse-fine face at fine resolution on one side only; see
    # "Centerings" in CODE.md.)
    for kinds in WALL_BOXES[D], C in (D == 3 ? [vertexcentered(3), edgecentered(3, 2)] :
                                       staggered_or_vertex(D))
        L = D == 1 ? 3 : 2
        reflecting = map(k -> k === :reflect_both, kinds)
        parity = any(reflecting) ?
                 [ntuple(d -> reflecting[d] ? EvenParity : NoParity, D)] : nothing
        f = point_symmetric(D, L)
        hook = boundary_by_coordinates(f)
        ops = Operators(prolongation=4, restriction=4)
        function filled(walls)
            forest = faces_forest(kinds; N=8, refined=false)
            fs = FieldSet(forest, 1; G=ghosts_for(C, 4), centering=C, parity=parity,
                          walls=walls)
            fill_ghosts!(fill_by_coordinates!(f, fs), GhostSchedule(fs, ops);
                         boundary=hook)
            return forest, fs
        end
        forest, fs = filled(:symmetric)
        result = point_mirror_error(fs, L)
        corners = count(d -> reflecting[d] && C[d] === :vertex, 1:D) >= 2
        @test result !== nothing && result[2] > 0
        @test result !== nothing && (corners ? result[1] < 1e-15 : result[1] == 0)
        regrid!(forest, fs => GhostSchedule(fs, ops); flags=fill(Refine, nleaves(forest)),
                boundary=hook)
        fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=hook)
        result = point_mirror_error(fs, L)
        @test result !== nothing && result[1] < 1e-14
        # At an outer face the hook writes the data on both walls, so
        # half-open walls are symmetric there too; elsewhere they are not.
        if !all(==(:outer), kinds)
            _, fs = filled(:half_open)
            result = point_mirror_error(fs, L)
            @test result !== nothing && result[1] > 1e-6
        end
    end
end

@testset "The fill writes the owned low wall, and reads it only after: D=$D" for
        D in (1, 2, 3)
    # The owned low wall planes are set to NaN before the fill. With
    # symmetric walls the fill must write every one of them, by the hook,
    # the reflection or the clamp, and no transfer may read one before it
    # is written: then nothing is NaN, and every value is the data, which
    # these operators reproduce. Half-open walls leave them as they were.
    for kinds in Iterators.product(ntuple(_ -> (:outer, :reflect_lo, :outflow_lo,
                                                :periodic), D)...),
        C in (D == 3 ? [vertexcentered(3), facecentered(3, 1)] : staggered_or_vertex(D))
        D == 3 && count(==(:periodic), kinds) > 1 && continue
        nnan, worst, nwalls = undefined_walls(kinds, C)
        @test nnan == 0
        @test worst < 1e-12
        @test (nwalls > 0) == any(d -> C[d] === :vertex && kinds[d] !== :periodic, 1:D)
        if nwalls > 0
            nnan, _, _ = undefined_walls(kinds, C; walls=:half_open)
            @test nnan > 0
        end
    end
    # Bounded point values bound the derived low wall as the high one.
    kinds = ntuple(d -> isodd(d) ? :reflect_lo : :outflow_lo, D)
    nnan, _, _ = undefined_walls(kinds, vertexcentered(D); bounded=true)
    @test nnan == 0
end

@testset "Symmetric walls partition the ghosts and the owned low walls: D=$D" for
        D in (1, 2, 3)
    # Every ghost written exactly once, every owned low wall plane at a
    # domain face exactly once, and no other owned point at all — so no
    # two regions overlap, which would be a race in one phase.
    for kinds in WALL_BOXES[D], C in (D == 3 ? [vertexcentered(3)] : staggered_or_vertex(D))
        forest = faces_forest(kinds; N=8)
        fs = FieldSet(forest, 1; G=ghosts_for(C, 4), centering=C, walls=:symmetric,
                      parity=any(==(:reflect_both), kinds) ?
                             [ntuple(d -> kinds[d] === :reflect_both ? EvenParity :
                                          NoParity, D)] : nothing)
        counts = write_counts(GhostSchedule(fs, Operators(prolongation=4, restriction=4)))
        expect = ones(Int, size(counts))
        G, N = fs.G, forest.N
        for b in 1:nblocks(fs)
            owned = ntuple(d -> (G[d] + 1):(G[d] + N), D)
            view(expect, owned..., b) .= 0
        end
        for (b, d) in low_wall_planes(fs)
            sel = ntuple(e -> e == d ? (G[d] + 1:G[d] + 1) : (G[e] + 1):(G[e] + N), D)
            view(expect, sel..., b) .= 1
        end
        @test counts == expect
    end
end

@testset "Half-open walls are the default and change nothing" begin
    # The default must be today's layout: no group key marks a trimmed
    # range, and a schedule built for the default fills a set that says
    # so explicitly, bit for bit.
    forest = faces_forest((:outflow_both, :reflect_both); N=8)
    parity = [(NoParity, EvenParity)]
    fs = FieldSet(forest, 1; G=1, centering=vertexcentered(2), parity=parity)
    @test fs.walls === :half_open
    explicit = FieldSet(forest, 1; G=1, centering=vertexcentered(2), parity=parity,
                        walls=:half_open)
    ops = Operators(prolongation=4, restriction=4)
    sched = GhostSchedule(fs, ops)
    @test sched.walls === :half_open
    f = point_symmetric(2, 2)
    fill_ghosts!(fill_by_coordinates!(f, fs), sched)
    fill_ghosts!(fill_by_coordinates!(f, explicit), GhostSchedule(explicit, ops))
    @test fs.work == explicit.work
end

@testset "A schedule belongs to its walls" begin
    # The target ranges at a low domain face differ between the two, so a
    # schedule built for one is refused for the other, in the fill and in
    # the regrid, and an unknown kind of wall is refused with the two
    # there are.
    forest = faces_forest((:outer, :outer); N=8)
    ops = Operators(prolongation=4, restriction=4)
    half = FieldSet(forest, 1; G=1, centering=vertexcentered(2))
    sym = FieldSet(forest, 1; G=1, centering=vertexcentered(2), walls=:symmetric)
    @test_throws "walls = :symmetric but this schedule was built for :half_open" fill_ghosts!(
        sym, GhostSchedule(half, ops))
    @test_throws "walls = :symmetric but its schedule was built for :half_open" regrid!(
        forest, sym => GhostSchedule(half, ops); flags=fill(Keep, nleaves(forest)))
    @test_throws "walls must be :half_open" FieldSet(forest, 1; G=1, walls=:closed)
end
