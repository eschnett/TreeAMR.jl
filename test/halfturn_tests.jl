# M14: a half-turn seam — the forest (step 1), the field sets (step 2), the
# turned ghosts (step 3), regrid and conservation (step 4), interpolation
# (step 5), and the wave equation on a half plane against the full plane
# (step 7).
#
# Only the half of the plane above the low face of `d1` is stored, and that
# face is glued to itself, reversed along `d2`, by a half turn about the
# line where it meets the centre plane of `d2`. It is M12's orientation 2
# with the turn moved to the centre line, plus one new thing: the turn
# flips `d2`, which is tangential to the seam, so a vertex-like `d2` has its
# seam slab partitioned one plane up. Every claim is stated against the
# unfolded forest of `ghost_oracles.jl`, built from turned `Rational`
# boxes, or against the full plane holding the turned data.
#
# It shares M12's test helpers — `complex_step`, `turned_values`, the
# advection kernels and `advect_rhs!`, the wave's — so `rotate_tests.jl`
# is included before it.

using TreeAMR: oriented_neighbors, hasrotating, ishalfturn, halfturn_dims, rotating_dims,
               seam_plane, seam_keywords

@testset "A half-turn seam is refused where it cannot be glued" begin
    # A seam declared where the face cannot be glued to itself, or across
    # a face that already means something else, must be refused with the
    # reason, not found later as a wrong neighbor.
    @test_throws "needs at least two dimensions" Forest((2,); N=4, halfturn=(1, 2))
    @test_throws "outside 1:2" Forest((2, 2); N=4, halfturn=(1, 3))
    @test_throws "names dimension 2 twice" Forest((2, 2); N=4, halfturn=(2, 2))
    @test_throws "even number of roots along dimension 2" Forest((2, 3); N=4,
                                                                 halfturn=(1, 2))
    @test_throws "even number of roots along dimension 2" Forest((2, 1); N=4,
                                                                 halfturn=(1, 2))
    @test_throws "a forest has one seam at most" Forest((2, 2); N=4, rotating=(1, 2),
                                                        halfturn=(1, 2))
    @test_throws "dimension 2 is periodic" Forest((1, 2); N=4, periodic=(false, true),
                                                  halfturn=(1, 2))
    @test_throws "dimension 1 is periodic" Forest((1, 2); N=4, periodic=(true, false),
                                                  halfturn=(1, 2))
    @test_throws "dimension 1 has a reflecting face" Forest((1, 2); N=4,
                                                            reflecting=((false, true),
                                                                        (false, false)),
                                                            halfturn=(1, 2))
    @test_throws "dimension 2 has a reflecting face" Forest((1, 2); N=4,
                                                            reflecting=((false, false),
                                                                        (true, false)),
                                                            halfturn=(1, 2))
    @test_throws "dimension 2 has an outflow face" Forest((1, 2); N=4,
                                                          outflow=((false, false),
                                                                   (false, true)),
                                                          halfturn=(1, 2))
    # The roots along `d1` are free, and the third dimension may be
    # anything.
    @test ishalfturn(Forest((3, 2); N=4, halfturn=(1, 2)))
    @test ishalfturn(Forest((1, 2, 1); N=4, periodic=(false, false, true),
                            halfturn=(1, 2)))
    @test ishalfturn(Forest((2, 1, 2); N=4, halfturn=(2, 3),
                            reflecting=((true, false), (false, false), (false, false))))
    # The pair as given, under its own keyword; the stored form is signed,
    # and every reader sees the plane.
    forest = Forest((2, 1, 2); N=4, halfturn=(2, 3))
    @test forest.rotating == (2, -3)
    @test hasrotating(forest)
    @test halfturn_dims(forest) == (2, 3)
    @test rotating_dims(forest) === nothing
    @test seam_plane(forest) == (2, 3)
    @test seam_keywords(forest) == (rotating=nothing, halfturn=(2, 3))
    quad = Forest((2, 2); N=4, rotating=(2, 1))
    @test !ishalfturn(quad)
    @test halfturn_dims(quad) === nothing
    @test seam_keywords(quad) == (rotating=(2, 1), halfturn=nothing)
    @test seam_keywords(Forest((2, 2); N=4)) == (rotating=nothing, halfturn=nothing)
    @test Forest{Float32}((1, 2); N=4, halfturn=(1, 2)).rotating == (1, -2)
    @test Forest((1, 2); N=4, halfturn=(1, 2),
                 extents=((0.0f0, 1.0f0), (-1.0f0, 1.0f0))).rotating == (1, -2)

    # The digest's brick tells a half turn from a quarter turn and from
    # none, so ranks that disagree about it are refused.
    digest(f) = TreeAMR.ForestDigest(f, UInt(0), false)
    half = Forest((2, 2); N=8, halfturn=(1, 2))
    @test digest(half).brick != digest(Forest((2, 2); N=8, rotating=(1, 2))).brick
    @test digest(half).brick != digest(Forest((2, 2); N=8)).brick
end

@testset "The half plane's neighbors are the unfolded forest's, turned back" begin
    # The tree glues the face to itself by a half turn about the centre
    # line: a wrong real node, a wrong orientation, or a finer search in
    # the wrong direction would hand the schedule the wrong source.
    # Against the unfolded forest, on random unbalanced half planes and
    # then after balance!, in D = 2 and in D = 3 with the third dimension
    # outer, periodic or reflecting at its low face.
    rng = MersenneTwister(1313)
    seen = Dict{Tuple{Int,Symbol},Int}()
    cases = [(Val(2), (1, 2), :outer), (Val(2), (2, 1), :outer),
             (Val(3), (1, 2), :outer), (Val(3), (1, 2), :periodic),
             (Val(3), (1, 2), :reflecting), (Val(3), (3, 1), :periodic),
             (Val(3), (2, 3), :reflecting)]
    for (V, plane, other) in cases, trial in 1:(V === Val(2) ? 6 : 3)
        half = random_halfturn_forest(rng, V; halfturn=plane, M1=rand(rng, 1:2),
                                      M2=2 * rand(rng, 1:2), other=other,
                                      nsteps=V === Val(2) ? 30 : 12, maxlvl=3)
        bad, s = halfturn_neighbor_mismatches(half)
        @test bad == 0
        mergewith!(+, seen, s)

        balance!(half)
        @test isbalanced(half)
        bad, s = halfturn_neighbor_mismatches(half)
        @test bad == 0
        mergewith!(+, seen, s)
        full, _, _ = unfolded_halfturn_forest(half)
        # Balanced and conforming, judged on the unfolded forest, which
        # has no seam for the package to know about.
        @test isbalanced(full)
        @test halfturn_conforming(full, plane[1])
        V === Val(2) && trial == 1 &&
            @test balanced_by_geometry(full, full.leaves) && tiles_brick(full, full.leaves)
        # And accepted by the checked leaf path, as it was produced.
        @test Forest(half.roots; N=4, periodic=half.periodic,
                     reflecting=TreeAMR.reflecting_faces(half), halfturn=plane,
                     leaves=half.leaves).leaves == half.leaves
    end
    # Orientation 2 met every kind of neighbor, and none at all: across
    # the seam and out through a face of `d2`.
    for r in (0, 2), kind in (:same, :coarser, :finer, :none)
        @test get(seen, (r, kind), 0) > 0
    end
    @test all(((r, _),) -> r in (0, 2), keys(seen))
end

@testset "Balance makes the half-turn seam conforming, and the leaf path refuses it otherwise" begin
    # Conformity keeps every coarse-fine face off the seam, here between a
    # leaf on the seam face and its image on the same face; a list that
    # breaks it must be refused with the reason.
    forest = Forest((1, 2); N=4, halfturn=(1, 2))
    bottom = only(filter(k -> root_position(forest, k.root) == (0, 0), forest.leaves))
    top = only(filter(k -> root_position(forest, k.root) == (0, 1), forest.leaves))
    refine!(forest, bottom)
    lopsided = copy(forest.leaves)
    @test balanced_by_geometry(Forest((1, 2); N=4), lopsided)
    @test !isbalanced(forest)
    @test_throws "not conforming at the rotating seam" Forest((1, 2); N=4,
                                                              halfturn=(1, 2),
                                                              leaves=lopsided)
    @test_throws "on the same face for a half turn" Forest((1, 2); N=4, halfturn=(1, 2),
                                                           leaves=lopsided)
    balance!(forest)
    # The image root is refined to match, and nothing else.
    @test !isleaf(forest, top) && all(c -> isleaf(forest, c), childkeys(top))
    @test nleaves(forest) == 8
    @test Forest((1, 2); N=4, halfturn=(1, 2), leaves=forest.leaves).leaves ==
          forest.leaves

    # Deeper, off the axis: a leaf on the seam face a quarter of the way up
    # wants its image three quarters of the way up at its level, and a
    # leaf away from the face does not.
    forest = Forest((2, 2); N=4, halfturn=(1, 2))
    refine!(forest, copy(forest.leaves))
    seamleaf = MortonKey{2}(0, 1, (0, 0))       # root (0, 0): x ∈ [0, 1/2), y ∈ [0, 1/2)
    refine!(forest, seamleaf)
    balance!(forest)
    image = MortonKey{2}(2, 1, (0, 1))          # root (0, 1): y ∈ [3/2, 2)
    @test !isleaf(forest, image)
    @test halfturn_conforming(first(unfolded_halfturn_forest(forest)), 1)
    inner = Forest((2, 2); N=4, halfturn=(1, 2))
    refine!(inner, copy(inner.leaves))
    refine!(inner, MortonKey{2}(1, 1, (1, 0)))  # root (1, 0), away from the face
    n = nleaves(inner)
    balance!(inner)
    @test nleaves(inner) == n
end

@testset "Completing marks moves no leaf by more than one level across the half-turn seam" begin
    # The regrid transfer is parent/child only; the seam rule raises a
    # leaf to its image's new level, which must never be two levels from
    # its old one.
    rng = MersenneTwister(4343)
    for (V, plane, other) in ((Val(2), (1, 2), :outer), (Val(2), (2, 1), :outer),
                              (Val(3), (1, 2), :reflecting), (Val(3), (1, 3), :periodic)),
        trial in 1:(V === Val(2) ? 8 : 3), buffer in (0, 1)
        forest = random_halfturn_forest(rng, V; halfturn=plane, M1=rand(rng, 1:2), M2=2,
                                        other=other, nsteps=V === Val(2) ? 24 : 10,
                                        maxlvl=3)
        balance!(forest)
        flags = [rand(rng, (Refine, Keep, Coarsen, Coarsen)) for _ in forest.leaves]
        new = complete_marks(forest, flags; buffer=buffer)
        old = Set(forest.leaves)
        moved = count(new) do k
            k in old && return false
            level(k) > 0 && parentkey(k) in old && return false
            return !all(c -> c in old, childkeys(k))
        end
        @test moved == 0
        rebuilt = Forest(forest.roots; N=4, periodic=forest.periodic,
                         reflecting=TreeAMR.reflecting_faces(forest),
                         seam_keywords(forest)..., leaves=new)
        @test halfturn_conforming(first(unfolded_halfturn_forest(rebuilt)), plane[1])
    end
end

@testset "Adjacency across the half-turn seam is mutual" begin
    # A distributed schedule finds the remote targets of its blocks as
    # their neighbors, which needs adjacency to be discoverable from both
    # sides, across the seam as anywhere.
    rng = MersenneTwister(78)
    for (V, plane, other) in ((Val(2), (1, 2), :outer), (Val(3), (2, 1), :periodic)),
        trial in 1:3
        forest = random_halfturn_forest(rng, V; halfturn=plane, M1=2, M2=2, other=other,
                                        nsteps=V === Val(2) ? 24 : 10, maxlvl=3)
        balance!(forest)
        n = nleaves(forest)
        touching = [Set(find_leaf(forest, nb)
                        for δ in alldirections(V)
                        for nb in neighbor_keys(forest, forest.leaves[i], δ))
                    for i in 1:n]
        @test all(i -> all(j -> i in touching[j], touching[i]), 1:n)
        for cut in (1, n ÷ 2, n - 1)
            range = 1:cut
            expected = sort!(unique!([j for i in range for j in touching[i]
                                      if !(j in range)]))
            @test TreeAMR.remote_neighbors(forest, range) == expected
        end
    end
end

# --- Steps 2 and 3: the field sets, and the ghosts across the seam ----------

const HALF_OPS4 = Operators(prolongation=4, restriction=4)

@testset "A field set's halfturn map is refused where it cannot turn the variables" begin
    # A map that is not a half turn of the variables would fill the seam's
    # ghosts with the wrong variable or sign, silently.
    forest = Forest((1, 2); N=8, halfturn=(1, 2))
    @test_throws "needs `halfturn`" FieldSet(forest, 3; G=1)
    # `rotation` is the quarter turn's map, and is no substitute.
    @test_throws "needs `halfturn`" FieldSet(forest, 3; G=1, rotation=(1, -3, 2))
    @test_throws "one entry per variable" FieldSet(forest, 3; G=1, halfturn=(1, -2))
    @test_throws "vector or tuple" FieldSet(forest, 1; G=1, halfturn=1)
    @test_throws "nonzero integer" FieldSet(forest, 1; G=1, halfturn=(1.0,))
    @test_throws "not a signed permutation" FieldSet(forest, 2; G=1, halfturn=(1, 1))
    @test_throws "applied twice is not the identity" FieldSet(forest, 2; G=1,
                                                              halfturn=(-2, 1))
    @test_throws "applied twice is not the identity" FieldSet(forest, 3; G=1,
                                                              halfturn=(2, 3, 1))
    # A vector in the plane, an exchange, and an axial vector's signs.
    @test FieldSet(forest, 3; G=1, halfturn=(1, -2, -3)).halfturn == [1, -2, -3]
    @test FieldSet(forest, 2; G=1, halfturn=(2, 1)).halfturn == [2, 1]
    @test FieldSet(forest, 2; G=1, halfturn=[-2, -1]).halfturn == [-2, -1]
    # Over a half-turn forest `rotation` is ignored, as over a forest
    # without a seam.
    @test FieldSet(forest, 1; G=1, halfturn=(1,), rotation=(-1,)).rotation == [-1]
    @test_throws "not a signed permutation" FieldSet(forest, 1; G=1, halfturn=(1,),
                                                     rotation=(2,))

    # The mirror and the turn commute, so a variable and its image need
    # one parity in a reflecting dimension.
    octant = Forest((1, 2, 1); N=8, halfturn=(1, 2),
                    reflecting=((false, false), (false, false), (true, false)))
    @test_throws "must have the same parity there" FieldSet(octant, 2; G=1,
        halfturn=(2, 1), parity=[(NoParity, NoParity, EvenParity),
                                 (NoParity, NoParity, OddParity)])

    # Over a forest without a half turn the map may be omitted, and if
    # given is checked for its shape and otherwise ignored.
    plain = Forest((2, 2); N=8)
    @test FieldSet(plain, 1; G=1).halfturn === nothing
    @test FieldSet(plain, 2; G=1, halfturn=(-2, 1)).rotvars === nothing
    @test_throws "not a signed permutation" FieldSet(plain, 2; G=1, halfturn=(2, 2))
    quad = Forest((2, 2); N=8, rotating=(1, 2))
    @test FieldSet(quad, 1; G=1, rotation=(1,), halfturn=(-1,)).halfturn == [-1]
    @test_throws "needs `rotation`" FieldSet(quad, 1; G=1, halfturn=(1,))

    # Every set turns into itself, so there is no pair.
    a = FieldSet(forest, 1; G=1, centering=facecentered(2, 1), halfturn=(-1,))
    b = FieldSet(forest, 1; G=1, centering=facecentered(2, 2), halfturn=(-1,))
    @test_throws "a half turn exchanges no dimensions" RotationPair(a, b)
end

@testset "The half-turn tables hold the map at orientation 2 and keep the parity columns" begin
    # The kernel reads variable `rotvars[v, 3]` and multiplies by the
    # factor at column `mirror + 3^D·2`: that is the half turn itself, not
    # the square of anything, and the `r = 0` columns stay M10's.
    forest = Forest((1, 2, 1); N=8, halfturn=(1, 2),
                    reflecting=((false, false), (false, false), (true, false)))
    parity = [(NoParity, NoParity, EvenParity), (NoParity, NoParity, EvenParity),
              (NoParity, NoParity, EvenParity), (NoParity, NoParity, OddParity)]
    fs = FieldSet(forest, 4; G=1, halfturn=(1, -2, -3, 4), parity=parity)
    @test size(fs.factors) == (4, 4 * 27)
    @test fs.rotvars[:, 3] == Int32[1, 2, 3, 4]
    @test fs.factors[:, 1 + 27 * 2] == [1, -1, -1, 1]
    @test fs.factors[:, 1:27] == FieldSet(Forest((1, 2, 1); N=8,
                                                 reflecting=((false, false),
                                                             (false, false),
                                                             (true, false))),
                                          4; G=1, parity=parity).factors
    # A mirrored and turned column multiplies the two: mirrored in z
    # (state 1 along the third dimension, column 1 + 9), the odd vz flips.
    @test fs.factors[:, 1 + 9 + 27 * 2] == [1, -1, -1, -1]
    # A permuting map lands in the variable column.
    pair = FieldSet(Forest((1, 2); N=8, halfturn=(1, 2)), 2; G=1, halfturn=(-2, -1))
    @test pair.rotvars[:, 3] == Int32[2, 1]
    @test pair.factors[:, 1 + 9 * 2] == [-1, -1]
    # Every layout turns into itself.
    @test TreeAMR.symmetric_layout(FieldSet(forest, 4; G=(1, 2, 1),
                                            centering=facecentered(3, 1),
                                            halfturn=(1, -2, -3, 4), parity=parity))
end

# The centerings the ghost tests run: every one in 2D; in 3D cell, vertex
# and the faces and an edge whose stagger lies along `d1`, along `d2` and
# out of the plane, which between them put a vertex-like `d2` in and out
# of the seam slab.
halfturn_centerings(D) =
    D == 2 ? [cellcentered(2), vertexcentered(2), facecentered(2, 1), facecentered(2, 2)] :
    [cellcentered(3), vertexcentered(3), facecentered(3, 1), facecentered(3, 2),
     facecentered(3, 3), edgecentered(3, 3)]

@testset "Every ghost across the half-turn seam is defined and none is read undefined: D=$D" for
        D in (2, 3)
    # The seam reads real blocks at the mirrored position on the same face,
    # at other levels, through the turn, and flips the dimension along it:
    # a region read before it is written, or never written, stays `NaN`,
    # and a wrong axis map, variable or sign shows as a deviation from data
    # the operators reproduce exactly. Without the shifted slab a
    # vertex-like `d2` reads its source's shared plane while it is being
    # written; under symmetric walls the low walls are `NaN` until the hook
    # writes them, so a transfer that reads one early shows too.
    cases = D == 2 ? [((1, 2), :none), ((2, 1), :none)] :
            [((1, 2), :periodic), ((1, 2), :outer), ((1, 2), :reflect_lo),
             ((3, 1), :periodic), ((2, 3), :reflect_lo)]
    bad = []
    seen = Dict{Tuple{Symbol,Int},Int}()
    for (plane, other) in cases, p in (2, 4), walls in (:half_open, :symmetric),
        C in (plane == cases[1][1] && walls === :half_open ? halfturn_centerings(D) :
              [cellcentered(D), vertexcentered(D)])
        nnan, worst, schedule = undefined_halfturn_ghosts(Val(D); plane=plane,
                                                          other=other, C=C, p=p,
                                                          walls=walls)
        (nnan == 0 && worst < 1e-10) || push!(bad, (plane, other, C, p, walls, nnan, worst))
        mergewith!(+, seen, rotation_counts(schedule))
    end
    @test isempty(bad)
    isempty(bad) || foreach(println, bad)
    # Every kind crosses the seam, in 2D too, where the edges of the seam
    # face meet other levels; and no quarter turn appears.
    for kind in (:copy, :restrict, :prolong)
        @test get(seen, (kind, 2), 0) > 0
        @test get(seen, (kind, 1), 0) == get(seen, (kind, 3), 0) == 0
    end
end

@testset "A half plane reproduces the turned full plane: D=$D" for D in (2, 3)
    # The definitional test, for data covariant under the half turn and
    # under neither a quarter turn nor a mirror: every stored point of the
    # half plane, ghosts included, equals the full plane's, which has no
    # seam and holds the turned data written out. Exactly where `d2` is
    # cell-like. Where it is vertex-like the two agree to interpolation
    # accuracy only, at the points where levels meet along `d2` in the
    # seam slab: half-open ownership puts a shared plane on a block's high
    # side, and the turn maps the high side to the low one, so a point the
    # full plane interpolates the half plane owns, and the other way round
    # at its image (CODE.md, *Half turns*). The polynomial test above
    # shows those stencils exact.
    cases = D == 2 ? [((1, 2), :none), ((2, 1), :none)] :
            [((1, 2), :periodic), ((2, 1), :reflect_lo), ((3, 1), :outer)]
    worst = Dict(:cell => 0.0, :vertex => 0.0)
    for (plane, other) in cases,
        C in (cellcentered(D), vertexcentered(D), facecentered(D, plane[1])),
        walls in (:half_open, :symmetric)
        result = halfturn_vs_doubled(Val(D); plane=plane, centering=C, other=other,
                                     walls=walls)
        @test result !== nothing
        result === nothing && continue
        w, npoints = result
        @test npoints > 0
        kind = C[plane[2]] === :vertex ? :vertex : :cell
        @test w < (kind === :cell ? 1e-13 : D == 2 ? 1e-6 : 5e-4)
        worst[kind] = max(worst[kind], w)
    end
    println("halfturn_vs_doubled D=$D: worst deviation $(worst[:cell]) cell-like d2, " *
            "$(worst[:vertex]) vertex-like d2")
end

@testset "The schedule partitions the ghosts across the half-turn seam: D=$D" for D in (2, 3)
    # Every stored point outside the owned range is written exactly once,
    # by a transfer — turned or not — or by the hook, and the hook is
    # handed only regions whose image leaves through an outer face; the
    # shifted slab keeps the partition. A double write would race; a gap
    # would be stale.
    cases = D == 2 ? [((1, 2), :none), ((2, 1), :none)] :
            [((1, 2), :outer), ((1, 2), :reflect_lo), ((2, 3), :periodic)]
    N = 8
    for (plane, other) in cases
        forest = halfturn_forest(Val(D); plane=plane, other=other, N=N)
        for C in halfturn_centerings(D), walls in (:half_open, :symmetric)
            G = ghosts_for(C, 4)
            fs = FieldSet(forest, D + 1; G=G, centering=C, walls=walls,
                          halfturn=vector_halfturn(D, plane),
                          parity=vector_parity(D, plane, other))
            schedule = GhostSchedule(fs, HALF_OPS4)
            counts = write_counts(schedule)
            c = staggers(C)
            # Under symmetric walls the fill writes the owned low wall
            # planes too, once.
            ok = true
            for b in 1:nleaves(forest)
                lw = TreeAMR.lowwalls(forest, forest.leaves[b], c, walls === :symmetric)
                for idx in CartesianIndices(size(counts)[1:D])
                    i = Tuple(idx)
                    inside = all(d -> G[d] + 1 + lw[d] <= i[d] <= G[d] + N, 1:D)
                    counts[idx, b] == (inside ? 0 : 1) || (ok = false)
                end
            end
            @test ok
            @test halfturn_hook_regions_leave(schedule)
            @test occursin("rotated transfers", sprint(show, schedule))
        end
    end
end

# --- Step 5: interpolation beyond the seam -----------------------------------

# The half turn `P` of the plane as an integer matrix, then, if `mirror`, the
# reflection of dimension `z` at its low face; the axis is at the origin.
function halfturn_matrix(D, (d1, d2); mirror::Bool=false, z::Int=0)
    return [Int(i == j) * (i == d1 || i == d2 || (mirror && i == z) ? -1 : 1)
            for i in 1:D, j in 1:D]
end

@testset "Interpolation beyond the half-turn seam is the turned field: D=$D" for D in (2, 3)
    # A point beyond the seam is answered from its preimage in the half
    # plane, turned about the centre line: a wrong fold, variable, sign or
    # derivative sign gives a wrong value there, which the polynomial data
    # would show at once, and the turned value must be the preimage's bit
    # for bit. The smooth data are compared against the full plane, at the
    # preimage, turned, and at the point itself, which agrees to roundoff
    # where `d2` is cell-like only (see the ghost comparison above).
    cases = D == 2 ? [((1, 2), :none), ((2, 1), :none)] :
            [((1, 2), :reflect_lo), ((3, 1), :periodic)]
    derivs = (ntuple(_ -> 0, D), ntuple(k -> ntuple(d -> Int(d == k), D), D)...)
    worst_poly = 0.0
    worst_full = Dict(:cell => 0.0, :vertex => 0.0)
    worst_direct = Dict(:cell => 0.0, :vertex => 0.0)
    for (plane, other) in cases, C in (cellcentered(D), vertexcentered(D))
        d1, d2 = plane
        z = outofplane(D, plane)
        mirrors = other === :reflect_lo
        rng = Xoshiro(hash((D, plane, C, :half)))
        # Points inside the half plane, away from its outer faces, and
        # their images (and mirrored below z = 0, where z reflects).
        qs = [ntuple(d -> d == d2 ? 3.9 * rand(rng) - 1.95 : 0.05 + 1.9 * rand(rng), D)
              for _ in 1:60]
        image(q, r, m) = Tuple((r == 2 ? halfturn_matrix(D, plane; mirror=m, z=z) :
                                halfturn_matrix(D, (0, 0); mirror=m, z=z)) * collect(q))
        turns = [(r, m) for r in (0, 2) for m in (mirrors ? (false, true) : (false,))]

        p = 4
        f = halfturn_data(D, plane, other; poly=p)
        fs = FieldSet(halfturn_forest(Val(D); plane=plane, other=other), D + 1;
                      G=ghosts_for(C, p), centering=C, halfturn=vector_halfturn(D, plane),
                      parity=vector_parity(D, plane, other))
        fill_by_coordinates!(f, fs)
        fill_ghosts!(fs, GhostSchedule(fs, HALF_OPS4); boundary=boundary_by_coordinates(f))
        xs = [image(q, r, m) for q in qs for (r, m) in turns]
        res = interpolate(fs, xs, Lagrange(4); derivs=derivs)
        for (j, x) in enumerate(xs), v in 1:(D + 1)
            worst_poly = max(worst_poly, abs(res.values[v, 1, j] - f(x, v)))
            for d in 1:D
                worst_poly = max(worst_poly,
                                 abs(res.values[v, 1 + d, j] - complex_step(f, x, v, d)))
            end
        end

        # The turned value is the preimage's, turned, bit for bit.
        vq = interpolate(fs, qs, Lagrange(4); derivs=derivs).values
        exact = true
        for (iq, q) in enumerate(qs), (r, m) in turns
            O = r == 2 ? halfturn_matrix(D, plane; mirror=m, z=z) :
                halfturn_matrix(D, (0, 0); mirror=m, z=z)
            vp = interpolate(fs, [image(q, r, m)], Lagrange(4); derivs=derivs).values
            exact &= vp[:, :, 1] == turned_values(vq[:, :, iq], O)
        end
        @test exact
        @test all(locate_point(fs.forest, image(q, r, m)) == locate_point(fs.forest, q)
                  for q in qs[1:10] for (r, m) in turns)

        # Smooth data against the full plane, at random points anywhere.
        half = halfturn_forest(Val(D); plane=plane, other=other)
        full = halfturn_forest(Val(D); plane=plane, other=other, full=true)
        g = halfturn_data(D, plane, other)
        par = vector_parity(D, plane, other)
        fsh = FieldSet(half, D + 1; G=ghosts_for(C, 4), centering=C, parity=par,
                       halfturn=vector_halfturn(D, plane))
        fsf = FieldSet(full, D + 1; G=ghosts_for(C, 4), centering=C, parity=par)
        for s in (fsh, fsf)
            fill_by_coordinates!(g, s)
            fill_ghosts!(s, GhostSchedule(s, HALF_OPS4); boundary=boundary_by_coordinates(g))
        end
        ys = [ntuple(d -> d == z && !mirrors ? 0.05 + 1.9 * rand(rng) :
                          3.9 * rand(rng) - 1.95, D) for _ in 1:400]
        a = interpolate(fsh, ys, Lagrange(4); derivs=derivs)
        b = interpolate(fsf, ys, Lagrange(4); derivs=derivs)
        Os = map(ys) do y
            y[d1] < 0 ? halfturn_matrix(D, plane; mirror=mirrors && y[z] < 0, z=z) :
            halfturn_matrix(D, (0, 0); mirror=mirrors && y[z] < 0, z=z)
        end
        pre = [Tuple(O' * collect(y)) for (O, y) in zip(Os, ys)]
        bq = interpolate(fsf, pre, Lagrange(4); derivs=derivs).values
        kind = first(C)
        for j in eachindex(ys)
            worst_full[kind] = max(worst_full[kind],
                                   maximum(abs.(a.values[:, :, j] .-
                                                turned_values(bq[:, :, j], Os[j]))))
        end
        worst_direct[kind] = max(worst_direct[kind], maximum(abs.(a.values .- b.values)))
        @test any(y -> y[d1] < 0, ys) && any(y -> y[d1] >= 0, ys)
        sub = interpolate(fsh, ys, Lagrange(4); derivs=derivs, vars=[3, 2])
        @test sub.values == a.values[[3, 2], :, :]
    end
    @test worst_poly < 1e-9
    @test worst_full[:cell] < 1e-12
    @test worst_direct[:cell] < 1e-12
    @test worst_full[:vertex] < 2e-2
    @test worst_direct[:vertex] < 2e-2               # values and gradients
    println("interpolation beyond the half-turn seam D=$D: polynomial data within " *
            "$worst_poly; smooth data against the full plane at the turned preimage " *
            "$(worst_full[:cell]) (cell) and $(worst_full[:vertex]) (vertex), at the " *
            "point itself $(worst_direct[:cell]) (cell) and $(worst_direct[:vertex]) " *
            "(vertex)")
end

@testset "Interpolation beyond the half-turn seam in Float32, and the refusal outside" begin
    # The fold in another element type, and a point the turn takes outside
    # the domain refused with the seam named.
    f = halfturn_data(2, (1, 2), :none; poly=4)
    fs = FieldSet{Float32}(halfturn_forest(Val(2); plane=(1, 2), other=:none, T=Float32),
                           3; G=2, halfturn=vector_halfturn(2, (1, 2)))
    fill_by_coordinates!(f, fs)
    fill_ghosts!(fs, GhostSchedule(fs, HALF_OPS4); boundary=boundary_by_coordinates(f))
    q = (0.3f0, 1.1f0)
    a = interpolate(fs, [q, (-0.3f0, -1.1f0)], Lagrange(4)).values
    @test a[:, 1, 2] == [a[1, 1, 1], -a[2, 1, 1], -a[3, 1, 1]]
    @test maximum(abs.(a[:, 1, 1] .- [f(Float64.(q), v) for v in 1:3])) < 1e-4
    @test_throws "half-turn seam below" interpolate(fs, [(-0.5f0, 2.5f0)], Lagrange(4))
end

# --- Step 4: regrid, initial data, the interface schedule, conservation ------

@testset "A regrid across the half-turn seam keeps it conforming and the turned full plane: D=$D" for
        D in (2, 3)
    # A regrid's transfers never cross the seam, but a prolongation reads
    # its parent's ghosts, which do; and the completion must keep the seam
    # conforming while moving no leaf by more than one level. Pass one
    # refines along the seam face on one side of the axis only, so that
    # conformity must refine its image on the other side; pass two
    # coarsens everything at level 2, as far as balance lets it; pass
    # three refines along the low face of `d2`. Exact where `d2` is
    # cell-like; to interpolation accuracy where it is vertex-like, as for
    # a single fill, and more loosely, since a prolongation carries the
    # differing ghosts of its parent into the interior.
    seamface(forest, k, d) = leafbox(forest, k)[1][d] == 0
    cases = D == 2 ? [((1, 2), :none), ((2, 1), :none)] :
            [((1, 2), :reflect_lo), ((3, 1), :periodic)]
    for (plane, other) in cases
        d1, d2 = plane
        # In root units the axis is at 2 along d2.
        upper(forest, k) = 2 <= leafbox(forest, k)[1][d2] < 3
        passes = [(f, k) -> seamface(f, k, d1) && upper(f, k) && level(k) < 3 ? Refine :
                            Keep,
                  (f, k) -> level(k) >= 2 ? Coarsen : Keep,
                  (f, k) -> seamface(f, k, d2) && level(k) < 2 ? Refine : Keep]
        for C in (cellcentered(D), vertexcentered(D))
            results = halfturn_regrid_vs_doubled(Val(D); plane=plane, other=other,
                                                 passes=passes, C=C)
            counts = [r.nleaves for r in results]
            for (i, r) in enumerate(results)
                @test r.moved
                @test r.conforming
                @test r.images
                @test r.npoints > 0
                @test r.worst < (C[d2] === :cell ? 1e-13 : D == 2 ? 1e-5 : 1e-3)
            end
            @test counts[1] > 10 && counts[2] < counts[1] && counts[3] > counts[2]
            println("regrid across the half-turn seam D=$D $plane $other $(first(C)): " *
                    "leaves $counts, worst $(maximum(r -> r.worst, results))")
        end
    end
end

@testset "The initial-data cycle adapts a half plane" begin
    # The cycle fills ghosts before it flags; it must converge to a
    # conforming mesh and leave data that a fill reproduces exactly.
    p = 4
    ops = Operators(prolongation=p, restriction=p)
    rule(key, forest) = begin
        lo, hi = leafbox(forest, key)
        c = (lo .+ hi) ./ 2
        # Off the axis, on one side of it: conformity refines the other.
        hypot(c[1], c[2] - 2.6) < 0.8 && level(key) < 2 ? Refine : Keep
    end
    f = halfturn_data(2, (1, 2), :none; poly=p)
    forest = Forest((2, 4); N=8, halfturn=(1, 2), extents=((0.0, 2.0), (-2.0, 2.0)))
    fs = FieldSet(forest, 3; G=ghosts_for(cellcentered(2), p),
                  halfturn=vector_halfturn(2, (1, 2)))
    schedule, passes, converged = adapt_to_initial_data!(fs, ops; initial=f,
        flag=(b, key) -> rule(key, forest), boundary=boundary_by_coordinates(f))
    @test converged
    @test maxlevel(forest) == 2
    @test isbalanced(forest)
    @test halfturn_conforming(first(unfolded_halfturn_forest(forest)), 1)
    # The image of the refined spot, below the axis, is refined too.
    @test any(k -> level(k) == 2 && leafbox(forest, k)[1][2] < 2, forest.leaves)
    fill_ghosts!(fs, schedule; boundary=boundary_by_coordinates(f))
    worst = maximum(Iterators.flatten(
        (abs(fs.work[idx, v, b] - f(coordinates(fs, b, Tuple(idx)), v))
         for idx in CartesianIndices(size(fs.work)[1:2]), v in 1:3) for b in 1:nblocks(fs)))
    @test worst < 1e-10
end

@testset "The interface schedule records nothing across the conforming half-turn seam: D=$D" for
        D in (2, 3)
    # A coarse-fine face across the seam would need a turned restriction,
    # which conformity rules out; elsewhere the schedule restricts exactly
    # as the same leaves without the seam.
    plane = (1, 2)
    other = D == 2 ? :none : :periodic
    forest = halfturn_forest(Val(D); plane=plane, other=other)
    plain = Forest(forest.roots; N=forest.N, periodic=forest.periodic,
                   extents=forest.extents, leaves=copy(forest.leaves))
    rng = Xoshiro(13)
    nrestrict = 0
    for d in 1:D
        halfturn = d in plane ? (-1,) : (1,)
        fh = FieldSet(forest, 1; G=0, centering=facecentered(D, d), halfturn=halfturn)
        fp = FieldSet(plain, 1; G=0, centering=facecentered(D, d))
        is = InterfaceSchedule(fh)
        groups = collect(Iterators.flatten(is.phases))
        @test all(g -> g.orientation == 0, groups)
        nrestrict += sum(ntransfers, groups; init=0)
        fh.work .= randn(rng, size(fh.work))
        fp.work .= fh.work
        restrict_interfaces!(fh, is)
        restrict_interfaces!(fp, InterfaceSchedule(fp))
        @test fh.work == fp.work
    end
    @test nrestrict > 0

    # A seam that is not conforming is a bug the schedule reports.
    broken = Forest((1, 2); N=8, halfturn=(1, 2))
    refine!(broken, only(filter(k -> leafbox(broken, k)[1] == (0, 0), broken.leaves)))
    fx = FieldSet(broken, 1; G=0, centering=facecentered(2, 1), halfturn=(-1,))
    @test_throws "this is a bug" InterfaceSchedule(fx)
end

"""
The rigid-rotation advection of `advection_problem` on a half plane: the
flux and the velocity, vectors, change sign under the half turn, and the
high face of `d1` and both faces of `d2` are walls.
"""
function halfturn_advection_problem(forest::Forest{D}; fixup::Bool) where {D}
    d1, d2 = TreeAMR.halfturn_dims(forest)
    state = FieldSet(forest, 1; G=2, halfturn=(1,))
    fluxes = ntuple(d -> FieldSet(forest, 1; G=0, centering=facecentered(D, d),
                                  halfturn=(-1,)), D)
    vel = ntuple(d -> FieldSet(forest, 1; G=0, centering=facecentered(D, d),
                               halfturn=(-1,)), D)
    for d in 1:D, b in 1:nblocks(vel[d])
        lo, hi = forest.extents[d]
        for idx in CartesianIndices(ntuple(e -> 1:(forest.N + (e == d)), D))
            x = coordinates(vel[d], b, Tuple(idx))
            wall = x[d] > hi - 1e-9 || (d == d2 && x[d] < lo + 1e-9)
            vel[d].work[idx, 1, b] = wall ? 0.0 :
                                     d == d1 ? -x[d2] : d == d2 ? x[d1] : 0.0
        end
    end
    return (; state, fluxes, vel, fixup, schedule=GhostSchedule(state, HALF_OPS4),
            ischeds=ntuple(d -> InterfaceSchedule(fluxes[d]), D),
            spacings=block_spacings(forest, Float64),
            hook=CellBoundary((x, v, δ) -> zero(eltype(x))))
end

"""
Advect a compact bump by the rigid rotation on a half plane and return
`(mass0, mass1, crossed)`, `crossed` the mass that ended below the axis,
having started above it at 65°, so that it can only have got there
through the seam.
"""
function halfturn_advect(forest::Forest{D}; fixup::Bool, nsteps::Int,
                         dt::Float64) where {D}
    d1, d2 = TreeAMR.halfturn_dims(forest)
    p = halfturn_advection_problem(forest; fixup=fixup)
    centre, w = (cos(1.13), sin(1.13)), 0.35
    fill_by_coordinates!(p.state) do x, _
        ρ2 = ((x[d1] - centre[1])^2 + (x[d2] - centre[2])^2) / w^2
        return ρ2 < 1 ? (1 - ρ2)^3 : 0.0
    end
    u = statevector(p.state)
    gather!(u, p.state)
    mass() = (scatter!(p.state, u); total_mass(p.state))
    mass0 = mass()
    k, u1, u2 = similar(u), similar(u), similar(u)
    for _ in 1:nsteps
        advect_rhs!(k, u, p)
        @. u1 = u + dt * k
        advect_rhs!(k, u1, p)
        @. u2 = 3 / 4 * u + 1 / 4 * (u1 + dt * k)
        advect_rhs!(k, u2, p)
        @. u = 1 / 3 * u + 2 / 3 * (u2 + dt * k)
    end
    mass1 = mass()
    crossed = 0.0
    for b in 1:nblocks(p.state)
        h = spacing(forest, blockkey(p.state, b))
        for idx in CartesianIndices(ntuple(_ -> 3:(forest.N + 2), D))
            x = coordinates(p.state, b, Tuple(idx))
            x[d2] < 0 && (crossed += p.state.work[idx, 1, b] * h^D)
        end
    end
    return mass0, mass1, crossed
end

@testset "Rigid rotation conserves mass through the half-turn seam and coarse-fine faces" begin
    # Mass that leaves the half plane through the seam above the axis
    # enters it through the seam below, and the two parts of the face
    # exchange exactly the flux one loses and the other gains only because
    # they are each other's image; the coarse-fine faces need the fixup,
    # as anywhere. The negative control drops the fixup and must drift,
    # far above roundoff though less than in M12's quadrant, since less of
    # the bump crosses a refinement boundary here.
    for plane in ((1, 2), (2, 1))
        forest = halfturn_forest(Val(2); plane=plane, other=:none)
        @test maxlevel(forest) == 2
        hmin = minimum(k -> spacing(forest, k), forest.leaves)
        dt = 0.4 * hmin / (2 * sqrt(2.0))
        nsteps = ceil(Int, 0.6 / dt)
        m0, m1, crossed = halfturn_advect(forest; fixup=true, nsteps=nsteps, dt=dt)
        drift = abs(m1 - m0) / m0
        @test drift < 1e-13
        @test crossed > 0.1 * m0
        n0, n1, _ = halfturn_advect(forest; fixup=false, nsteps=nsteps, dt=dt)
        leak = abs(n1 - n0) / n0
        @test n0 == m0
        @test leak > 1e-5
        println("rigid rotation through the half-turn seam $plane: $nsteps steps, " *
                "crossed $(crossed / m0) of the mass, drift $drift with the fixup, " *
                "$leak without")
    end
end

# --- Step 7: the wave equation on a half plane -------------------------------
#
# The standing wave of M12's test, made invariant under the half turn
# rather than the quarter turn: `φ(p) = f(p) + f(−p)` about the axis,
# whose terms come round in the other order at `−p`, the same bits, and
# its gradient `∇f(p) − ∇f(−p)`, which the turn negates exactly, since
# `a − b = −(b − a)`. So the two seam half-planes of a vertex-centered
# set start out each other's image bit for bit.

halfwave_phi(p) = rotwave_f(p) + rotwave_f((-p[1], -p[2]))
function halfwave_gradient(p)
    g, h = rotwave_∇f(p), rotwave_∇f((-p[1], -p[2]))
    return (g[1] - h[1], g[2] - h[2])
end

"""The exact standing wave at time `t`, as `rotwave_exact` for the half turn."""
function halfwave_exact(K::Int, (d1, d2), t)
    c, s = cos(ROTWAVE_ω * t), -ROTWAVE_ω * sin(ROTWAVE_ω * t)
    return function (x, v)
        p = (x[d1], x[d2])
        k = v > K ? v - K : v
        value = K == 1 ? halfwave_phi(p) : halfwave_gradient(p)[k]
        return (v > K ? s : c) * value
    end
end

# The half turn's map of the wave's state: a scalar keeps its sign, and a
# vector in the plane and its time derivative change it.
halfwave_map(K::Int) = K == 1 ? [1, 2] : [-1, -2, -3, -4]

function halfwave_rhs!(du, u, p, t)
    scatter!(p.fs, u)
    fill_ghosts!(p.fs, p.schedule;
                 boundary=boundary_by_coordinates(halfwave_exact(p.K, p.plane, t)))
    map_blocks!(rotwave_kernel!, p.fs, statearray(du, p.fs), p.fs.work, p.spacings,
                Val(2), p.valG, Val(p.K))
    return nothing
end

"""
`rotating_wave` on the half plane of `halfturn_forest(Val(2); plane)` or,
with `full = true`, on the full plane.
"""
function halfturn_wave(; N, C, K, plane=(1, 2), full::Bool, p=4, cfl=0.25, periods=0.25,
                       refine::Bool=true, walls::Symbol=:half_open)
    forest = halfturn_forest(Val(2); plane=plane, other=:none, N=N, full=full,
                             refine=refine)
    G = ghosts_for(C, p)
    halfturn = full ? nothing : halfwave_map(K)
    fs = FieldSet(forest, 2K; G=G, centering=C, halfturn=halfturn, walls=walls)
    schedule = GhostSchedule(fs, Operators(prolongation=p, restriction=p))
    problem = (; fs, schedule, K, plane, valG=Val(G),
               spacings=block_spacings(forest, Float64))
    fill_by_coordinates!(halfwave_exact(K, plane, 0.0), fs)
    u0 = statevector(fs)
    gather!(u0, fs)
    h = minimum_spacing(forest)
    t_end = periods * 2π / ROTWAVE_ω
    nsteps = ceil(Int, t_end / (cfl * h))
    prob = ODEProblem(halfwave_rhs!, u0, (0.0, t_end), problem)
    sol = solve(prob, RK4(); dt=t_end / nsteps, adaptive=false, save_everystep=false)
    exact = FieldSet(forest, 2K; G=G, centering=C, halfturn=halfturn, walls=walls)
    fill_by_coordinates!(halfwave_exact(K, plane, t_end), exact)
    uexact = statevector(exact)
    gather!(uexact, exact)
    err = sol.u[end] .- uexact
    scatter!(fs, sol.u[end])
    fill_ghosts!(fs, schedule;
                 boundary=boundary_by_coordinates(halfwave_exact(K, plane, t_end)))
    return (l2=volume_weighted_norm(fs, err), linf=volume_weighted_norm(fs, err; p=Inf),
            h=h, nsteps=nsteps, nblocks=nleaves(forest), fs=fs)
end

"""
The worst difference between the vertex-centered half plane's owned seam
points `(0, y)` and `(0, −y)`, one point under the half turn: the first
must be the second's values through the signed map. Returns `(worst,
npairs)`, with `==` semantics; the axis itself is skipped.
"""
function seam_halfplane_mismatch(fs::FieldSet{T,2}) where {T}
    d1, d2 = halfturn_dims(fs.forest)
    plane = Dict{T,Vector{T}}()
    for b in 1:nblocks(fs)
        block_extent(fs.forest, blockkey(fs, b))[d1][1] == 0 || continue
        for j in 1:(fs.forest.N)
            idx = Base.setindex(Base.setindex((0, 0), fs.G[d1] + 1, d1), fs.G[d2] + j, d2)
            plane[coordinates(fs, b, idx)[d2]] = fs.work[idx..., :, b]
        end
    end
    worst, npairs = 0.0, 0
    for (y, here) in plane
        y > 0 || continue
        there = plane[-y]
        npairs += 1
        for v in eachindex(here)
            q = fs.halfturn[v]
            turned = sign(q) * there[abs(q)]
            worst = here[v] == turned ? worst : max(worst, abs(here[v] - turned))
        end
    end
    return worst, npairs
end

"""
How far the full plane's own solution is from covariant under the half
turn: the worst difference, over its owned points `p` with `x_{d1} ≥ 0`
whose image is owned too, between its values at `−p` and those at `p`
through the map.
"""
function halfturn_defect(fs::FieldSet{T,2}, map_) where {T}
    owned = Dict{NTuple{2,T},Vector{T}}()
    for b in 1:nblocks(fs)
        for i in CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + fs.forest.N), 2))
            owned[coordinates(fs, b, Tuple(i))] = fs.work[i, :, b]
        end
    end
    worst = 0.0
    for (p, u) in owned
        p[1] >= 0 || continue
        there = get(owned, (-p[1], -p[2]), nothing)
        there === nothing && continue
        worst = max(worst, maximum(abs.(there .- [sign(q) * u[abs(q)] for q in map_])))
    end
    return worst
end

@testset "The wave on a half plane evolves as the full plane it folds: K=$K" for K in (1, 2)
    # Many fills compound whatever one fill gets wrong. Cell-centered the
    # half plane owns exactly the full plane's points in its half and the
    # two agree to roundoff at every stored point. Vertex-centered the full
    # plane is not covariant itself where levels meet along `d2`, and the
    # half plane must lie within the full plane's own defect of it. It owns
    # the seam points `(0, y)` and `(0, −y)` twice, as one point. On a
    # uniform half plane with symmetric walls they stay each other's image
    # bit for bit. Under half-open walls the turn exchanges an evolved low
    # wall of `d2` with a hook-filled high one, and where levels meet along
    # the seam half-open ownership puts the shared point of a coarse-fine
    # face on its upper block, which the turn makes the lower one; either
    # way they drift apart by the scheme's error, within the full plane's
    # own defect.
    for C in (cellcentered(2), vertexcentered(2))
        full = halfturn_wave(; N=16, C=C, K=K, full=true)
        half = halfturn_wave(; N=16, C=C, K=K, full=false)
        @test half.nsteps == full.nsteps
        @test 2 * half.nblocks == full.nblocks
        worst, npoints = compare_matching_blocks(half.fs, full.fs)
        defect = halfturn_defect(full.fs, halfwave_map(K))
        @test npoints > 0
        line = "wave on a half plane, K=$K, $(C[1]): $(half.nsteps) steps, " *
               "linf $(half.linf) against $(full.linf), every stored point within " *
               "$worst, the full plane covariant to $defect"
        if C == cellcentered(2)
            @test half.linf ≈ full.linf rtol = 1e-8
            @test worst < 1e-12
            @test defect < 1e-12
        else
            @test abs(half.linf - full.linf) < defect
            @test 0 < worst < defect < 1e-2
            seam, npairs = seam_halfplane_mismatch(half.fs)
            @test npairs > 0
            @test seam < defect
            uniform = halfturn_wave(; N=16, C=C, K=K, full=false, refine=false,
                                    walls=:symmetric)
            useam, _ = seam_halfplane_mismatch(uniform.fs)
            @test useam == 0
            line *= ", seam half-planes $seam apart over $npairs pairs ($useam uniform)"
        end
        println(line)
    end
end

@testset "The wave on a half plane converges at the interface-order rate" begin
    # Order-4 operators against the 2nd-order Laplacian: rate 2, as on the
    # quadrant; an inconsistency at the seam (a ghost off by one plane in
    # the shifted slab, say) shows as a lower rate.
    for C in (cellcentered(2), vertexcentered(2)), K in (1, 2)
        hs, l2, linf = Float64[], Float64[], Float64[]
        for N in (8, 16, 32)
            r = halfturn_wave(; N=N, C=C, K=K, full=false)
            push!(hs, r.h)
            push!(l2, r.l2)
            push!(linf, r.linf)
        end
        rl2, rlinf = convergence_rate(hs, l2), convergence_rate(hs, linf)
        @test rl2 ≈ 2.0 atol = 0.15
        @test rlinf ≈ 2.0 atol = 0.2
        println("wave on a half plane, K=$K, $(C[1]): rates $rl2 (l2), $rlinf (linf); " *
                "l2 $l2")
    end
end
