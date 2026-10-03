# M12: a rotating seam, the forest (step 1).
#
# Only one quadrant of the plane is stored, and the low face of `d1` is
# glued to the low face of `d2` by a quarter turn. The tree sees the
# seam: the neighbor search returns the real leaves across it with their
# orientation, and balance keeps the two sides of it at one level. Every
# claim is stated against the unfolded forest of `ghost_oracles.jl`,
# built from turned `Rational` boxes, never against the seam's own key
# arithmetic.

using TreeAMR: oriented_neighbors, virtual_offset, real_direction, rotating_dims,
               hasrotating

@testset "A rotating seam is refused where it cannot be glued" begin
    # A seam declared where the two low faces cannot be glued root for
    # root, or across a face that already means something else, must be
    # refused with the reason, not found later as a wrong neighbor.
    @test_throws "needs at least two dimensions" Forest((2,); N=4, rotating=(1, 2))
    @test_throws "outside 1:2" Forest((2, 2); N=4, rotating=(1, 3))
    @test_throws "outside 1:2" Forest((2, 2); N=4, rotating=(0, 1))
    @test_throws "names dimension 2 twice" Forest((2, 2); N=4, rotating=(2, 2))
    @test_throws "as many roots along dimension 1 as along 2" Forest((2, 1); N=4,
                                                                     rotating=(1, 2))
    @test_throws "dimension 2 is periodic" Forest((2, 2); N=4, periodic=(false, true),
                                                  rotating=(1, 2))
    @test_throws "dimension 1 has a reflecting face" Forest((2, 2); N=4,
                                                            reflecting=((false, true),
                                                                        (false, false)),
                                                            rotating=(1, 2))
    @test_throws "open question" Forest((2, 2, 1); N=4,
                                        reflecting=((false, false), (true, false),
                                                    (false, false)),
                                        rotating=(2, 1))
    # The third dimension may be anything.
    @test hasrotating(Forest((2, 2, 1); N=4, periodic=(false, false, true),
                             rotating=(1, 2)))
    @test hasrotating(Forest((2, 2, 1); N=4, rotating=(2, 1),
                             reflecting=((false, false), (false, false), (true, false))))
    # Either order; the pair as given, and none by default.
    @test rotating_dims(Forest((2, 1, 2); N=4, rotating=(3, 1))) == (3, 1)
    @test rotating_dims(Forest((2, 2); N=4)) === nothing
    @test Forest((2, 2); N=4).rotating == (0, 0)
    @test Forest{Float32}((2, 2); N=4, rotating=(1, 2)).rotating == (1, 2)
    @test Forest((2, 2); N=4, rotating=(1, 2),
                 extents=((0.0f0, 1.0f0), (0.0f0, 1.0f0))).rotating == (1, 2)

    # Until the interface schedule, interpolation and the checkpoint
    # learn the orientation, each refuses the seam rather than read
    # across it as though it were an ordinary face. (The ghost schedule
    # learned it in step 3.)
    forest = Forest((2, 2); N=8, rotating=(1, 2))
    fs = FieldSet(forest, 1; G=1, rotation=(1,))
    vs = FieldSet(forest, 1; G=0, centering=facecentered(2, 1), rotation=(-1,))
    @test_throws "not implemented yet in this step of M12" InterfaceSchedule(vs)
    @test_throws "InterfaceSchedule over a rotating forest" InterfaceSchedule(vs)
    @test_throws "interpolate over a rotating forest" interpolate(fs, [(0.5, 0.5)],
                                                                  Lagrange(2))
    mktempdir() do dir
        @test_throws "save_checkpoint over a rotating forest" save_checkpoint(
            joinpath(dir, "c.h5"), forest; fieldsets=())
    end

    # The digest's brick tells a seam from none, so ranks that disagree
    # about it are refused.
    plain = Forest((2, 2); N=8)
    @test TreeAMR.ForestDigest(forest, UInt(0), false).brick !=
          TreeAMR.ForestDigest(plain, UInt(0), false).brick
    @test !TreeAMR.sameforest(TreeAMR.ForestDigest(forest, UInt(0), false),
                              TreeAMR.ForestDigest(plain, UInt(0), false))
end

@testset "The seam's direction and offset maps turn by the rotation" begin
    # A wrong sign or swap here sends the finer-neighbor search to the
    # wrong children, or a restriction to the wrong stencil. Stated
    # against the turned boxes: the direction is the turn of a unit step,
    # and the offset that of a child's box inside its parent.
    for (d1, d2) in ((1, 2), (2, 1), (1, 3), (3, 2)), r in 0:3
        for δ in alldirections(Val(3))
            unit = (ntuple(_ -> 0//1, 3), ntuple(d -> δ[d] // 1, 3))
            turned = rotate_box(unit, -r, d1, d2)
            expected = ntuple(d -> Int(turned[1][d] + turned[2][d]), 3)
            @test real_direction(δ, r, d1, d2) == expected
        end
        for o in Iterators.product(0:1, 0:1, 0:1)
            # The child's box in its parent of size 2 centred on the axis,
            # turned forward by r into the virtual frame.
            box = (ntuple(d -> o[d] - 1//1, 3), ntuple(d -> o[d] + 0//1, 3))
            lo, _ = rotate_box(box, r, d1, d2)
            @test virtual_offset(o, r, d1, d2) == ntuple(d -> Int(lo[d]) + 1, 3)
        end
    end
end

@testset "The seam's neighbors are the unfolded forest's, turned back" begin
    # The tree glues the two low faces by a quarter turn: a wrong
    # orientation, a wrong real node, or a finer search in the wrong
    # direction would hand the schedule the wrong source. Against the
    # unfolded forest, on random unbalanced quadrants and then after
    # balance!, in D = 2 and in D = 3 with the third dimension outer,
    # periodic or reflecting at its low face.
    rng = MersenneTwister(1212)
    seen = Dict{Tuple{Int,Symbol},Int}()
    cases = [(Val(2), (1, 2), :outer), (Val(2), (2, 1), :outer),
             (Val(3), (1, 2), :outer), (Val(3), (1, 2), :periodic),
             (Val(3), (1, 2), :reflecting), (Val(3), (3, 1), :periodic),
             (Val(3), (2, 3), :reflecting)]
    for (V, rot, other) in cases, trial in 1:(V === Val(2) ? 6 : 3)
        quad = random_rotating_forest(rng, V; rotating=rot, M=rand(rng, 1:2),
                                      other=other, nsteps=V === Val(2) ? 30 : 12,
                                      maxlvl=3)
        bad, s = seam_neighbor_mismatches(quad)
        @test bad == 0
        mergewith!(+, seen, s)

        balance!(quad)
        @test isbalanced(quad)
        bad, s = seam_neighbor_mismatches(quad)
        @test bad == 0
        mergewith!(+, seen, s)
        full, _, _ = unfolded_forest(quad)
        d1, d2 = rot
        # Balanced and conforming, judged on the unfolded forest, which
        # has no seam for the package to know about.
        @test isbalanced(full)
        @test seam_conforming(full, d1, d2)
        V === Val(2) && trial == 1 &&
            @test balanced_by_geometry(full, full.leaves) && tiles_brick(full, full.leaves)
        # And accepted by the checked leaf path, as it was produced.
        @test Forest(quad.roots; N=4, periodic=quad.periodic, reflecting=quad.reflecting,
                     rotating=rot, leaves=quad.leaves).leaves == quad.leaves
    end
    # Every orientation met every kind of neighbor, and none at all:
    # beyond an outer face, or across the seam and out again.
    for r in 0:3, kind in (:same, :coarser, :finer, :none)
        @test get(seen, (r, kind), 0) > 0
    end
end

@testset "A block at the axis is its own neighbor in three directions" begin
    # The quarter turns fix the axis, so the block there sees itself
    # beyond the seam on either side and across the corner, as a single
    # periodic root sees itself.
    forest = Forest((1, 1); N=4, rotating=(1, 2))
    k = only(forest.leaves)
    @test oriented_neighbors(forest, k, (-1, 0)) == (1, [k])
    @test oriented_neighbors(forest, k, (0, -1)) == (3, [k])
    @test oriented_neighbors(forest, k, (-1, -1)) == (2, [k])
    # Beyond the seam and the high face at once: nothing, but the
    # orientation of the region stepped into.
    @test oriented_neighbors(forest, k, (-1, 1)) == (1, MortonKey{2}[])
    @test oriented_neighbors(forest, k, (1, -1)) == (3, MortonKey{2}[])
    @test oriented_neighbors(forest, k, (1, 0)) == (0, MortonKey{2}[])
    @test neighbor_keys(forest, k, (-1, 0)) == [k]

    # The seam is not a reflecting face: reflect_direction masks only the
    # wall outside the plane.
    octant = Forest((1, 1, 1); N=4, rotating=(1, 2),
                    reflecting=((false, false), (false, false), (true, false)))
    k = only(octant.leaves)
    @test TreeAMR.reflect_direction(octant, k, (-1, 0, -1)) ==
          ((-1, 0, 0), (false, false, true))
    @test oriented_neighbors(octant, k, (-1, 0, 0)) == (1, [k])
end

@testset "Balance makes the seam conforming, and the leaf path refuses it otherwise" begin
    # Conformity is what keeps every coarse-fine face off the seam; a
    # list that breaks it would give the interface restriction a face it
    # cannot handle, so it must be refused with the reason.
    forest = Forest((2, 2); N=4, rotating=(1, 2))
    # Root (0, 1) lies on the low face of dimension 1; across the seam
    # its face neighbor is root (1, 0), on the low face of dimension 2.
    up = only(filter(k -> root_position(forest, k.root) == (0, 1), forest.leaves))
    right = only(filter(k -> root_position(forest, k.root) == (1, 0), forest.leaves))
    refine!(forest, up)
    lopsided = copy(forest.leaves)
    # One level apart is 2:1 balanced, so only the seam rule is broken.
    @test balanced_by_geometry(Forest((2, 2); N=4), lopsided)
    @test !isbalanced(forest)
    @test Forest((2, 2); N=4, leaves=lopsided).leaves == lopsided
    @test_throws "not conforming at the rotating seam" Forest((2, 2); N=4,
                                                              rotating=(1, 2),
                                                              leaves=lopsided)
    @test_throws "across the seam's face" Forest((2, 2); N=4, rotating=(1, 2),
                                                 leaves=lopsided)
    balance!(forest)
    # The image root is refined to match, and nothing else.
    @test !isleaf(forest, right) && all(c -> isleaf(forest, c), childkeys(right))
    @test nleaves(forest) == 2 + 2 * 4
    @test isbalanced(forest)
    @test Forest((2, 2); N=4, rotating=(1, 2), leaves=forest.leaves).leaves ==
          forest.leaves

    # A level deeper, inside one root: the seam's face still wants one
    # level, from the leaves on either side of it.
    forest = Forest((1, 1); N=4, rotating=(1, 2))
    refine!(forest, only(forest.leaves))
    corner = only(filter(k -> k.coords == (1, 0), forest.leaves))
    refine!(forest, corner)                 # on the low face of dimension 2
    balance!(forest)
    image = MortonKey{2}(0, 1, (0, 1))      # its image on the low face of dimension 1
    @test !isleaf(forest, image)
    @test seam_conforming(first(unfolded_forest(forest)), 1, 2)

    # The ordinary 2:1 refusal is still the ordinary one.
    deep = Forest((2, 2); N=4, rotating=(1, 2))
    far = only(filter(k -> root_position(deep, k.root) == (1, 1), deep.leaves))
    refine!(deep, far)
    refine!(deep, first(childkeys(far)))
    @test_throws "not 2:1 balanced" Forest((2, 2); N=4, rotating=(1, 2),
                                           leaves=deep.leaves)
end

@testset "Completing marks moves no leaf by more than one level across the seam" begin
    # The regrid transfer is parent/child only; the seam rule raises a
    # leaf to its image's new level, and that must never be two levels
    # from its old one, or the transfer would have no source.
    rng = MersenneTwister(4242)
    for (V, rot, other) in ((Val(2), (1, 2), :outer), (Val(2), (2, 1), :outer),
                            (Val(3), (1, 2), :reflecting), (Val(3), (1, 3), :periodic)),
        trial in 1:(V === Val(2) ? 8 : 3), buffer in (0, 1)
        forest = random_rotating_forest(rng, V; rotating=rot, M=rand(rng, 1:2),
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
                         reflecting=forest.reflecting, rotating=rot, leaves=new)
        @test seam_conforming(first(unfolded_forest(rebuilt)), rot...)
    end
end

@testset "Adjacency across the seam is mutual, so every rank finds its targets" begin
    # A distributed schedule finds the remote targets of its blocks as
    # their neighbors (`remote_neighbors`), which needs adjacency to be
    # discoverable from both sides, across the seam as anywhere.
    rng = MersenneTwister(77)
    for (V, rot, other) in ((Val(2), (1, 2), :outer), (Val(3), (2, 1), :periodic)),
        trial in 1:3
        forest = random_rotating_forest(rng, V; rotating=rot, M=2, other=other,
                                        nsteps=V === Val(2) ? 24 : 10, maxlvl=3)
        balance!(forest)
        n = nleaves(forest)
        touching = [Set(find_leaf(forest, nb)
                        for δ in alldirections(V)
                        for nb in neighbor_keys(forest, forest.leaves[i], δ))
                    for i in 1:n]
        @test all(i -> all(j -> i in touching[j], touching[i]), 1:n)
        # Across the seam, by the unfolded forest: some pair is adjacent
        # only through it.
        full, image, back = unfolded_forest(forest)
        @test any(1:n) do i
            any(alldirections(V)) do δ
                any(nf -> first(back(nf)) != 0,
                    neighbor_keys(full, image(forest.leaves[i], 0), δ))
            end
        end
        for cut in (1, n ÷ 2, n - 1)
            range = 1:cut
            expected = sort!(unique!([j for i in range for j in touching[i]
                                      if !(j in range)]))
            @test TreeAMR.remote_neighbors(forest, range) == expected
        end
    end
end

# --- Steps 2 and 3: the field sets, and the ghosts across the seam ----------
#
# A ghost beyond the seam is the turned image of real data: read through
# the quarter turn's axis map and the field set's signed variable map, in
# the ordinary phases. Every claim is stated against data covariant under
# the turn, written out by formula, and against the full plane the
# quadrant folds, which holds the turned data explicitly.

const ROT_OPS4 = Operators(prolongation=4, restriction=4)

@testset "A field set's rotation is refused where it cannot turn the variables" begin
    # A map that is not a quarter turn of the variables would fill the
    # seam's ghosts with the wrong variable or the wrong sign, silently;
    # it must be refused with the reason.
    forest = Forest((2, 2); N=8, rotating=(1, 2))
    @test_throws "needs `rotation`" FieldSet(forest, 3; G=1)
    @test_throws "one entry per variable" FieldSet(forest, 3; G=1, rotation=(1, -3))
    @test_throws "vector or tuple" FieldSet(forest, 1; G=1, rotation=1)
    @test_throws "nonzero integer" FieldSet(forest, 1; G=1, rotation=(1.0,))
    @test_throws "not a signed permutation" FieldSet(forest, 2; G=1, rotation=(1, 1))
    @test_throws "not a signed permutation" FieldSet(forest, 2; G=1, rotation=(0, 2))
    @test_throws "not a signed permutation" FieldSet(forest, 2; G=1, rotation=(1, 3))
    # Four quarter turns are none: a three-cycle is not a turn of
    # anything, and a sign that does not come back is not either.
    @test_throws "turned four times is not the identity" FieldSet(forest, 3; G=1,
                                                                  rotation=(2, 3, 1))
    @test_throws "turned four times is not the identity" FieldSet(forest, 4; G=1,
                                                                  rotation=(2, 3, 4, -1))
    # A vector, either order of the pair, and an axial vector's pseudo
    # sign are all quarter turns.
    @test FieldSet(forest, 3; G=1, rotation=(1, -3, 2)).rotation == [1, -3, 2]
    @test FieldSet(forest, 3; G=1, rotation=[1, 3, -2]).rotation == [1, 3, -2]
    @test FieldSet(forest, 1; G=1, rotation=(-1,)).rotation == [-1]

    # The mirror and the turn commute, so a variable and its image need
    # one parity in a reflecting dimension.
    octant = Forest((2, 2, 1); N=8, rotating=(1, 2),
                    reflecting=((false, false), (false, false), (true, false)))
    @test_throws "must have the same parity there" FieldSet(octant, 2; G=1,
        rotation=(-2, 1), parity=[(NoParity, NoParity, EvenParity),
                                  (NoParity, NoParity, OddParity)])
    @test FieldSet(octant, 2; G=1, rotation=(-2, 1),
                   parity=[(NoParity, NoParity, OddParity),
                           (NoParity, NoParity, OddParity)]).rotation == [-2, 1]

    # Over a forest without a seam the rotation may be omitted, and if
    # given is checked for its shape and otherwise ignored, as parity is,
    # so nothing written before M12 changes.
    plain = Forest((2, 2); N=8)
    @test FieldSet(plain, 1; G=1).rotation === nothing
    @test FieldSet(plain, 1; G=1).rotvars === nothing
    @test FieldSet(plain, 2; G=1, rotation=(2, 1)).rotvars === nothing
    @test FieldSet(plain, 2; G=1, rotation=(2, 1)).factors === nothing
    @test_throws "not a signed permutation" FieldSet(plain, 2; G=1, rotation=(2, 2))
    # An asymmetric set's map is its partner's business, so a lone one is
    # not judged by its fourth power.
    @test FieldSet(forest, 3; G=(1, 2), rotation=(2, 3, 1)).rotation == [2, 3, 1]
end

@testset "The rotation tables compose the quarter turns and keep the parity columns" begin
    # The kernel reads variable `rotvars[v, r+1]` and multiplies by the
    # factor at column `mirror + 3^D·r`: a wrong composition puts the
    # wrong component or sign two or three turns away, and a moved
    # `r = 0` column would change every mirrored transfer of M10.
    forest = Forest((2, 2, 1); N=8, rotating=(1, 2),
                    reflecting=((false, false), (false, false), (true, false)))
    parity = [(NoParity, NoParity, EvenParity), (NoParity, NoParity, EvenParity),
              (NoParity, NoParity, EvenParity), (NoParity, NoParity, OddParity)]
    fs = FieldSet(forest, 4; G=1, rotation=(1, -3, 2, 4), parity=parity)
    @test size(fs.factors) == (4, 4 * 27)
    @test fs.rotvars == Int32[1 1 1 1; 2 3 2 3; 3 2 3 2; 4 4 4 4]
    nomirror(r) = fs.factors[:, 1 + 27r]
    @test nomirror(0) == [1, 1, 1, 1]
    @test nomirror(1) == [1, -1, 1, 1]           # vx′ = −vy, vy′ = vx
    @test nomirror(2) == [1, -1, -1, 1]          # a half turn negates the plane
    @test nomirror(3) == [1, 1, -1, 1]
    # The mirror columns, block r = 0, are M10's table; a mirrored and
    # turned column multiplies the two.
    flat = Forest((2, 2, 1); N=8,
                  reflecting=((false, false), (false, false), (true, false)))
    @test fs.factors[:, 1:27] == FieldSet(flat, 4; G=1, parity=parity).factors
    zmirror = TreeAMR.mirrorcolumn((0, 0, 1))
    @test fs.factors[:, zmirror + 27] == [1, -1, 1, -1]
    # A rotating forest without a reflecting face still has the table,
    # for the signs; a forest with neither has none.
    @test FieldSet(Forest((2, 2); N=8, rotating=(1, 2)), 1; G=1,
                   rotation=(1,)).factors == ones(1, 36)
end

@testset "A RotationPair is refused unless its two sets turn into each other" begin
    # The pair's tables read one set's data for the other's ghosts; a
    # mismatched pair would read the wrong layout, variable or sign.
    forest = Forest((2, 2); N=8, rotating=(1, 2))
    Bx(; kw...) = FieldSet(forest, 1; G=(1, 2), centering=facecentered(2, 1),
                           rotation=(-1,), kw...)
    By(; kw...) = FieldSet(forest, 1; G=(2, 1), centering=facecentered(2, 2),
                           rotation=(1,), kw...)
    pair = RotationPair(Bx(), By())
    @test pair isa RotationPair
    @test pair.arotvars == Int32[1 1 1 1]
    @test pair.afactors[1, 1 .+ 9 .* (0:3)] == [1, -1, -1, 1]
    @test pair.bfactors[1, 1 .+ 9 .* (0:3)] == [1, 1, -1, -1]

    other = Forest((2, 2); N=8, rotating=(1, 2))
    @test_throws "over one forest" RotationPair(Bx(), FieldSet(other, 1; G=(2, 1),
        centering=facecentered(2, 2), rotation=(1,)))
    plain = Forest((2, 2); N=8)
    @test_throws "needs a forest with a rotating seam" RotationPair(
        FieldSet(plain, 1; G=(1, 2), centering=facecentered(2, 1)),
        FieldSet(plain, 1; G=(2, 1), centering=facecentered(2, 2)))
    @test_throws "each other's layout" RotationPair(Bx(), Bx())
    @test_throws "each other's layout" RotationPair(Bx(), FieldSet(forest, 1; G=(1, 2),
        centering=facecentered(2, 2), rotation=(1,)))
    @test_throws "turns into itself" RotationPair(FieldSet(forest, 1; G=1, rotation=(1,)),
                                                  FieldSet(forest, 1; G=1, rotation=(1,)))
    @test_throws "same number of variables" RotationPair(Bx(), FieldSet(forest, 2;
        G=(2, 1), centering=facecentered(2, 2), rotation=(1, 2)))
    @test_throws "one element type" RotationPair(Bx(), FieldSet{Float32}(forest, 1;
        G=(2, 1), centering=facecentered(2, 2), rotation=(1,)))
    # Q_a Q_b a quarter turn of the variables themselves: its square is
    # −I, not I.
    @test_throws "Q_a Q_b Q_a Q_b are not the identity" RotationPair(
        FieldSet(forest, 2; G=(1, 2), centering=facecentered(2, 1), rotation=(-2, 1)),
        FieldSet(forest, 2; G=(2, 1), centering=facecentered(2, 2), rotation=(1, 2)))
    octant = Forest((2, 2, 1); N=8, rotating=(1, 2),
                    reflecting=((false, false), (false, false), (true, false)))
    @test_throws "must have the same parity there" RotationPair(
        FieldSet(octant, 1; G=(1, 2, 1), centering=facecentered(3, 1), rotation=(-1,),
                 parity=[(NoParity, NoParity, EvenParity)]),
        FieldSet(octant, 1; G=(2, 1, 1), centering=facecentered(3, 2), rotation=(1,),
                 parity=[(NoParity, NoParity, OddParity)]))
end

# The centerings symmetric under exchanging the plane's two dimensions:
# cell and vertex, and in 3D the two staggered along the third dimension
# only, the face and the edge normal to and along it.
function symmetric_centerings(D, rotating)
    D == 2 && return [cellcentered(2), vertexcentered(2)]
    z = outofplane(D, rotating)
    return [cellcentered(3), vertexcentered(3), facecentered(3, z), edgecentered(3, z)]
end

@testset "Every ghost across the seam is defined and none is read undefined: D=$D" for
        D in (2, 3)
    # The seam reads real blocks elsewhere in the tree, at other levels,
    # through the turn: a region read before it is written, or never
    # written, stays `NaN`; a wrong axis map, variable or sign shows as a
    # deviation from data the operators reproduce exactly. Three levels
    # meet the seam side by side, and the axis, for every symmetric
    # centering, both orders, and in 3D the third dimension periodic,
    # outer (the hook's, across the seam's image too) and reflecting.
    cases = D == 2 ? [((1, 2), :none), ((2, 1), :none)] :
            [((1, 2), :periodic), ((1, 2), :outer), ((1, 2), :reflect_lo),
             ((3, 1), :periodic), ((2, 3), :reflect_lo)]
    bad = []
    seen = Dict{Tuple{Symbol,Int},Int}()
    # The staggered centerings are new kernels for each third dimension,
    # and the suite is compilation-bound, so the other orders of the pair
    # run cell and vertex only.
    for (rot, other) in cases, p in (2, 4),
        C in (rot == cases[1][1] ? symmetric_centerings(D, rot) :
              [cellcentered(D), vertexcentered(D)])
        nnan, worst, schedule = undefined_rotated_ghosts(Val(D); rotating=rot,
                                                         other=other, C=C, p=p)
        (nnan == 0 && worst < 1e-10) || push!(bad, (rot, other, C, p, nnan, worst))
        mergewith!(+, seen, rotation_counts(schedule))
    end
    @test isempty(bad)
    isempty(bad) || foreach(println, bad)
    # Every kind of transfer crosses the seam, in every orientation, so
    # the claim is not vacuous — but in 2D a half turn is only ever a
    # copy: a region beyond both low faces belongs to the block at the
    # axis, whose image there is itself. In 3D the half turn also reaches
    # that block's neighbors along the third dimension, at other levels.
    for kind in (:copy, :restrict, :prolong), r in 1:3
        D == 2 && r == 2 && kind !== :copy && continue
        @test get(seen, (kind, r), 0) > 0
    end
end

@testset "A rotating quadrant reproduces the turned full plane: D=$D" for D in (2, 3)
    # The definitional test, for data that is covariant under the turn
    # and no polynomial: every stored point of the quadrant, ghosts
    # included, equals the full plane's, which has no seam and holds the
    # turned data written out. The refinement reaches the seam and the
    # axis with three levels side by side.
    cases = D == 2 ? [((1, 2), :none), ((2, 1), :none)] :
            [((1, 2), :periodic), ((2, 1), :reflect_lo), ((3, 1), :outer)]
    worst = 0.0
    for (rot, other) in cases, C in (cellcentered(D), vertexcentered(D))
        result = rotating_vs_quadrupled(Val(D); rotating=rot, centering=C, other=other)
        @test result !== nothing
        result === nothing && continue
        w, npoints = result
        @test npoints > 0
        @test w < 1e-13
        worst = max(worst, w)
    end
    println("rotating_vs_quadrupled D=$D: worst deviation $worst")
end

@testset "A RotationPair reproduces the turned full plane set by set: D=$D" for
        D in (2, 3)
    # B_x across the seam is −B_y: a pair fill reads each set's ghosts
    # there out of the other, through the composed maps, with a
    # prolongation free to read its partner's coarser ghosts. Face
    # centering with G > 0 in the plane, with the variables stored in
    # swapped orders so the maps permute, and a cell-centered pair whose
    # ghost widths alone are swapped.
    cases = D == 2 ? [((1, 2), :none), ((2, 1), :none)] :
            [((1, 2), :reflect_lo), ((3, 2), :periodic)]
    worst = 0.0
    for (rot, other) in cases
        for (Ca, Ga, p) in ((facecentered(D, rot[1]), nothing, 4),
                            (cellcentered(D), ntuple(d -> d == rot[1] ? 2 : 1, D), 2))
            G = Ga === nothing ? ghosts_for(Ca, p) : Ga
            result = rotating_pair_vs_quadrupled(Val(D); rotating=rot, other=other, p=p,
                                                 Ca=Ca, Ga=G)
            @test result !== nothing
            result === nothing && continue
            w, npoints, (sa, sb) = result
            @test npoints > 0
            @test w < 1e-13
            worst = max(worst, w)
            # Each set's schedule partitions its own ghosts.
            for s in (sa, sb)
                @test occursin("rotated transfers", sprint(show, s))
                @test hook_regions_leave(s)
            end
        end
    end
    println("RotationPair vs the full plane D=$D: worst deviation $worst")
end

@testset "The schedule partitions the ghosts across the seam: D=$D" for D in (2, 3)
    # Every stored point outside the owned range is written exactly once,
    # by a transfer — rotated or not — or by the hook, and the hook is
    # handed only regions whose image leaves through an outer face. A
    # double write would race; a gap would be stale.
    cases = D == 2 ? [((1, 2), :none), ((2, 1), :none)] :
            [((1, 2), :outer), ((1, 2), :reflect_lo), ((2, 3), :periodic)]
    N = 8
    for (rot, other) in cases
        forest = rotating_forest(Val(D); rotating=rot, other=other, N=N)
        d1, d2 = rot
        swap(t) = Base.setindex(Base.setindex(t, t[d2], d1), t[d1], d2)
        layouts = [(C, ghosts_for(C, 4)) for C in symmetric_centerings(D, rot)]
        Cf = facecentered(D, d1)
        push!(layouts, (Cf, ghosts_for(Cf, 4)), (swap(Cf), swap(ghosts_for(Cf, 4))))
        for (C, G) in layouts
            fs = FieldSet(forest, D + 1; G=G, centering=C,
                          rotation=vector_rotation(D, rot),
                          parity=vector_parity(D, rot, other))
            schedule = GhostSchedule(fs, ROT_OPS4)
            counts = write_counts(schedule)
            owned = ntuple(d -> (G[d] + 1):(G[d] + N), D)
            ok = true
            for b in 1:nleaves(forest), idx in CartesianIndices(size(counts)[1:D])
                inside = all(d -> Tuple(idx)[d] in owned[d], 1:D)
                counts[idx, b] == (inside ? 0 : 1) || (ok = false)
            end
            @test ok
            @test hook_regions_leave(schedule)
        end
    end
end

@testset "An asymmetric set is refused alone where its ghosts are its partner's" begin
    # Filled alone, a set whose layout is not symmetric in the plane would
    # read its own data where its partner's belongs, the wrong component
    # in the wrong layout; the plain fill refuses it and says what to do.
    forest = rotating_forest(Val(2); rotating=(1, 2))
    Bx = FieldSet(forest, 1; G=(1, 2), centering=facecentered(2, 1), rotation=(-1,))
    By = FieldSet(forest, 1; G=(2, 1), centering=facecentered(2, 2), rotation=(1,))
    sx, sy = GhostSchedule(Bx, ROT_OPS4), GhostSchedule(By, ROT_OPS4)
    @test_throws "Fill it as a RotationPair" fill_ghosts!(Bx, sx)
    @test_throws "not symmetric in the rotating seam's dimensions" fill_ghosts!(By, sy)
    # One hook for both sets, or one each; and the schedules in order.
    hook = CellBoundary((x, v, δ) -> zero(eltype(x)))
    @test fill_ghosts!(RotationPair(Bx, By), (sx, sy); boundary=hook) isa RotationPair
    @test_throws "a tuple of two" fill_ghosts!(RotationPair(Bx, By), (sx, sy);
                                                boundary=(hook, hook, hook))
    @test_throws "ghost width" fill_ghosts!(RotationPair(Bx, By), (sy, sx))
    # A symmetric set needs no partner, and says how much it turns.
    fs = FieldSet(forest, 3; G=2, rotation=(1, -3, 2))
    s = GhostSchedule(fs, ROT_OPS4)
    @test occursin("rotated transfers", sprint(show, s))
    @test !occursin("mirrored", sprint(show, s))
    @test fill_ghosts!(fs, s) === fs
    # Off the seam nothing is said.
    plain = FieldSet(Forest((2, 2); N=8), 1; G=2)
    @test !occursin("rotated", sprint(show, GhostSchedule(plain, ROT_OPS4)))
    # A set with no ghosts in the plane has nothing across the seam: a
    # face-centered flux with G = 0 needs no partner, though no ghost
    # schedule serves it (check_operators wants G ≥ 1 along a
    # cell-centered dimension), and it is regridded without one.
    flux = FieldSet(forest, 1; G=0, centering=facecentered(2, 1), rotation=(-1,))
    @test flux.rotvars == Int32[1 1 1 1]
end
