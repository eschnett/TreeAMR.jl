# M12: a rotating seam — the forest (step 1), the turned ghosts (steps
# 2–3), regrid and conservation (step 4), interpolation (step 5), and the
# wave equation on a quadrant against the full plane (step 8).
#
# Only one quadrant of the plane is stored, and the low face of `d1` is
# glued to the low face of `d2` by a quarter turn. The tree sees the
# seam: the neighbor search returns the real leaves across it with their
# orientation, and balance keeps the two sides of it at one level. Every
# claim is stated against the unfolded forest of `ghost_oracles.jl`,
# built from turned `Rational` boxes, never against the seam's own key
# arithmetic.

using KernelAbstractions: @kernel, @index, @Const
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

    forest = Forest((2, 2); N=8, rotating=(1, 2))

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

# --- M12 step 4: regrid, initial data and the interface schedule -------------

@testset "A regrid across the seam keeps it conforming and the turned full plane: D=$D" for
        D in (2, 3)
    # A regrid moves blocks by parent/child transfers that never cross the
    # seam, but a prolongation reads its parent's ghosts, which do: a
    # regrid that skipped the fill, or filled a pair alone, would prolong
    # from stale or wrong seam ghosts. And the completion must keep the
    # seam conforming while moving no leaf by more than one level. Pass
    # one refines along the low face of d1 only, near the axis, so that
    # conformity must refine the image on the low face of d2; pass two
    # coarsens everything at level 2, as far as balance lets it; pass
    # three refines along the low face of d2 out to the high faces.
    lowface(forest, k, d) = leafbox(forest, k)[1][d] == 0
    near(forest, k, d, w) = leafbox(forest, k)[1][d] < w
    cases = D == 2 ? [((1, 2), :none), ((2, 1), :none)] :
            [((1, 2), :reflect_lo), ((3, 1), :periodic)]
    for (rot, other) in cases
        d1, d2 = rot
        passes = [(f, k) -> lowface(f, k, d1) && near(f, k, d2, 1) && level(k) < 3 ?
                            Refine : Keep,
                  (f, k) -> level(k) >= 2 ? Coarsen : Keep,
                  (f, k) -> lowface(f, k, d2) && level(k) < 2 ? Refine : Keep]
        for (label, kw) in (("cell", (; C=cellcentered(D))),
                            ("vertex", (; C=vertexcentered(D))),
                            ("pair", (; pair=true)))
            results = rotating_regrid_vs_quadrupled(Val(D); rotating=rot, other=other,
                                                    passes=passes, kw...)
            counts = [r.nleaves for r in results]
            for (i, r) in enumerate(results)
                @test r.moved
                @test r.conforming
                @test r.images
                @test r.npoints > 0
                @test r.worst < 1e-13
                (r.worst < 1e-13 && r.images) ||
                    println("regrid D=$D $rot $other $label pass $i: $r")
            end
            # Every pass changed the mesh, so the claim is not vacuous.
            @test counts[1] > 10 && counts[2] < counts[1] && counts[3] > counts[2]
            println("regrid across the seam D=$D $rot $other $label: leaves $counts, " *
                    "worst $(maximum(r -> r.worst, results))")
        end
    end
end

# Refine the level-0 leaves touching the low face of dimension 1.
lowface_rule(forest, k) = leafbox(forest, k)[1][1] == 0 && level(k) == 0 ? Refine : Keep

@testset "regrid! refuses an asymmetric set alone and takes a pair with its schedules" begin
    # Moved alone, a set whose ghosts across the seam are its partner's
    # would be filled from its own data before the transfer; the regrid
    # must refuse it with the hint, as the fill does, and agree on it on
    # every rank. Resizing it alone fills nothing and stays allowed.
    forest = rotating_forest(Val(2); rotating=(1, 2))
    Bx = FieldSet(forest, 1; G=(1, 2), centering=facecentered(2, 1), rotation=(-1,))
    By = FieldSet(forest, 1; G=(2, 1), centering=facecentered(2, 2), rotation=(1,))
    sx, sy = GhostSchedule(Bx, ROT_OPS4), GhostSchedule(By, ROT_OPS4)
    keep = fill(Keep, nleaves(forest))
    @test_throws "Regrid the two sets as a pair" regrid!(forest, Bx => sx; flags=keep)
    @test_throws "Regrid the two sets as a pair" regrid!(forest, [By => sy]; flags=keep)
    @test_throws "with its two schedules" regrid!(forest, RotationPair(Bx, By) => sx;
                                                  flags=keep)
    @test_throws "`pair => (schedule_a, schedule_b)`" regrid!(forest, RotationPair(Bx, By);
                                                              flags=keep)
    @test_throws "ghost width" regrid!(forest, RotationPair(Bx, By) => (sy, sx);
                                       flags=keep)
    @test regrid!(forest, [Bx => nothing, By => nothing]; flags=keep) == false
    @test regrid!(forest, RotationPair(Bx, By) => (sx, sy); flags=keep) == false
    # A refusal changes nothing, and a pair's regrid moves both sets.
    g = generation(forest)
    flags = [lowface_rule(forest, k) for k in forest.leaves]
    pair = RotationPair(Bx, By)
    @test regrid!(forest, pair => (sx, sy); flags=flags,
                  boundary=CellBoundary((x, v, δ) -> zero(eltype(x))))
    @test generation(forest) > g
    @test nblocks(Bx) == nblocks(By) == nleaves(forest)
    @test isbalanced(forest)
    # The pair's tables survive, since the sets' storage was replaced in
    # place: the pair fills again with fresh schedules.
    @test fill_ghosts!(pair, (GhostSchedule(Bx, ROT_OPS4), GhostSchedule(By, ROT_OPS4));
                       boundary=CellBoundary((x, v, δ) -> zero(eltype(x)))) === pair
end

@testset "The initial-data cycle adapts a rotating quadrant, alone or as a pair" begin
    # The cycle fills ghosts before it flags, and a pair's ghosts across
    # the seam are each other's; it must converge to a conforming mesh
    # and leave data that a fill reproduces exactly.
    p = 4
    ops = Operators(prolongation=p, restriction=p)
    rule(b, key, forest) = begin
        lo, hi = leafbox(forest, key)
        c = (lo .+ hi) ./ 2
        hypot(c[1], c[2]) < 0.8 && level(key) < 2 ? Refine : Keep
    end
    f = rotating_data(2, (1, 2), :none; poly=p)
    forest = Forest((2, 2); N=8, rotating=(1, 2), extents=((0.0, 2.0), (0.0, 2.0)))
    fs = FieldSet(forest, 3; G=ghosts_for(cellcentered(2), p),
                  rotation=vector_rotation(2, (1, 2)))
    schedule, passes, converged = adapt_to_initial_data!(fs, ops; initial=f,
        flag=(b, key) -> rule(b, key, forest), boundary=boundary_by_coordinates(f))
    @test converged
    @test passes == 3
    @test maxlevel(forest) == 2
    @test isbalanced(forest)
    fill_ghosts!(fs, schedule; boundary=boundary_by_coordinates(f))
    worst = maximum(Iterators.flatten(
        (abs(fs.work[idx, v, b] - f(coordinates(fs, b, Tuple(idx)), v))
         for idx in CartesianIndices(size(fs.work)[1:2]), v in 1:3) for b in 1:nblocks(fs)))
    @test worst < 1e-10

    # The pair form: two callbacks, the flags of the whole pair at once.
    field(x, k) = covariant_vector(x[1], x[2], k; poly=p)
    fa(x, v) = field(x, v)[1]
    fb(x, v) = field(x, 3 - v)[2]
    forest = Forest((2, 2); N=8, rotating=(1, 2), extents=((0.0, 2.0), (0.0, 2.0)))
    a = FieldSet(forest, 2; G=ghosts_for(facecentered(2, 1), p),
                 centering=facecentered(2, 1), rotation=(-2, -1))
    b = FieldSet(forest, 2; G=reverse(ghosts_for(facecentered(2, 1), p)),
                 centering=facecentered(2, 2), rotation=(2, 1))
    pair = RotationPair(a, b)
    @test_throws "Fill it as a RotationPair" adapt_to_initial_data!(a, ops; initial=fa,
        flag=(b, key) -> Keep)
    @test_throws "a tuple of two" adapt_to_initial_data!(pair, ops;
        initial=(fa, fb, fa), flag=(b, key) -> Keep)
    hooks = (boundary_by_coordinates(fa), boundary_by_coordinates(fb))
    (sa, sb), passes, converged = adapt_to_initial_data!(pair, ops; initial=(fa, fb),
        flags=pr -> [rule(i, key, forest) for (i, key) in enumerate(pr.a.forest.leaves)],
        boundary=hooks)
    @test converged
    @test maxlevel(forest) == 2
    @test isbalanced(forest)
    fill_ghosts!(pair, (sa, sb); boundary=hooks)
    worst = 0.0
    for (fs, g) in ((a, fa), (b, fb)), blk in 1:nblocks(fs),
        idx in CartesianIndices(size(fs.work)[1:2]), v in 1:2
        worst = max(worst, abs(fs.work[idx, v, blk] - g(coordinates(fs, blk, Tuple(idx)), v)))
    end
    @test worst < 1e-10
end

@testset "The interface schedule records nothing across the conforming seam: D=$D" for
        D in (2, 3)
    # A coarse-fine face across the seam would need a turned restriction,
    # which the interface schedule does not build; conformity rules it
    # out. So a rotating forest's schedule must hold no rotated transfer,
    # and elsewhere restrict exactly as the same leaves without the seam,
    # whose seam faces are outer faces that nothing restricts either.
    rot = (1, 2)
    other = D == 2 ? :none : :periodic
    forest = rotating_forest(Val(D); rotating=rot, other=other)
    plain = Forest(forest.roots; N=forest.N, periodic=forest.periodic,
                   extents=forest.extents, leaves=copy(forest.leaves))
    rng = Xoshiro(12)
    nrestrict = 0
    for d in 1:D
        rotation = d == 1 ? (-1,) : (1,)
        fr = FieldSet(forest, 1; G=0, centering=facecentered(D, d), rotation=rotation)
        fp = FieldSet(plain, 1; G=0, centering=facecentered(D, d))
        is = InterfaceSchedule(fr)
        groups = collect(Iterators.flatten(is.phases))
        @test all(g -> g.orientation == 0, groups)
        @test !occursin("rotated", sprint(show, is))
        nrestrict += sum(ntransfers, groups; init=0)
        fr.work .= randn(rng, size(fr.work))
        fp.work .= fr.work
        restrict_interfaces!(fr, is)
        restrict_interfaces!(fp, InterfaceSchedule(fp))
        @test fr.work == fp.work
    end
    @test nrestrict > 0

    # A seam that is not conforming is a bug the schedule reports, not a
    # restriction it skips: refine one leaf on the low face of d2 without
    # balancing, so that its image across the low face of d1 is coarser.
    broken = Forest((2, 2); N=8, rotating=(1, 2))
    refine!(broken, only(filter(k -> leafbox(broken, k)[1] == (1, 0), broken.leaves)))
    fx = FieldSet(broken, 1; G=0, centering=facecentered(2, 1), rotation=(-1,))
    @test_throws "this is a bug" InterfaceSchedule(fx)
end

# A conservative advection by the rigid rotation `v = (−y, x)` in the
# plane of the seam, about its axis: the three-step right-hand side of
# `burgers.jl` with a linear reconstruction and an upwind flux. Mass that
# leaves the quadrant through the low face of `d1` enters it through the
# low face of `d2`, and the two faces' fluxes cancel only because they
# are each other's image: the flux is `F = u v`, a vector, so
# `F_{d1}(0, s) = −F_{d2}(s, 0)`, and the scheme computes it covariantly.
# The block on one seam face reconstructs from its cells and its turned
# ghosts, which are the image block's cells, and the image block
# reconstructs the same numbers in the other order; the centered slope
# is antisymmetric under that reversal, `a − b = −(b − a)` exactly, and
# the upwind choice is made by the sign of the *normal* velocity, which
# the turn negates on one side, so both pick the same state and the
# fluxes cancel bit for bit. A Rusanov flux with `|v|` would too; a flux
# that chose its upwind side by a fixed axis would not. The high faces
# are walls, a zero normal velocity there, so that nothing else enters or
# leaves, and the hook fills the outer ghosts with zeros.
@kernel function advect_flux_kernel!(flux, @Const(work), @Const(vel), ::Val{D},
                                     ::Val{GU}, ::Val{d}) where {D,GU,d}
    I = @index(Global, NTuple)                     # (i1..iD, block), face indices
    b = I[D + 1]
    c = ntuple(e -> I[e] + GU[e], Val(D))
    m1 = Base.setindex(c, c[d] - 1, d)
    m2 = Base.setindex(c, c[d] - 2, d)
    p1 = Base.setindex(c, c[d] + 1, d)
    um2, um1 = work[m2..., 1, b], work[m1..., 1, b]
    u0, up1 = work[c..., 1, b], work[p1..., 1, b]
    uL = um1 + (u0 - um2) / 4                      # centered slope, halved
    uR = u0 - (up1 - um1) / 4
    v = vel[ntuple(e -> I[e], Val(D))..., 1, b]    # G = 0: stored = face index
    z = zero(v)
    flux[ntuple(e -> I[e], Val(D))..., 1, b] = max(v, z) * uL + min(v, z) * uR
end

@kernel function advect_divergence_kernel!(du, fluxes, @Const(spacings),
                                           ::Val{D}) where {D}
    I = @index(Global, NTuple)
    b = I[D + 1]
    c = ntuple(e -> I[e], Val(D))
    acc = zero(eltype(du))
    for d in 1:D
        hi = Base.setindex(c, c[d] + 1, d)
        acc += fluxes[d][hi..., 1, b] - fluxes[d][c..., 1, b]
    end
    du[c..., 1, b] = -acc / spacings[b]
end

"""
The state, flux, velocity and schedules of the rigid-rotation advection
on a rotating quadrant; `fixup = false` drops `restrict_interfaces!`.
"""
function advection_problem(forest::Forest{D}; fixup::Bool) where {D}
    d1, d2 = TreeAMR.rotating_dims(forest)
    rotation(d) = d == d1 ? (-1,) : (1,)       # F_{d1} is −F_{d2} a turn away
    state = FieldSet(forest, 1; G=2, rotation=(1,))
    fluxes = ntuple(d -> FieldSet(forest, 1; G=0, centering=facecentered(D, d),
                                  rotation=rotation(d)), D)
    vel = ntuple(d -> FieldSet(forest, 1; G=0, centering=facecentered(D, d),
                               rotation=rotation(d)), D)
    hi = ntuple(d -> forest.extents[d][2], D)
    for d in 1:D, b in 1:nblocks(vel[d])
        # The normal velocity on each face of the closed range, both of a
        # block's own faces, zero on the high walls. (`fill_by_coordinates!`
        # fills owned points only, which leaves the high face at zero, and
        # a block whose high face carries no flux leaks.)
        for idx in CartesianIndices(ntuple(e -> 1:(forest.N + (e == d)), D))
            x = coordinates(vel[d], b, Tuple(idx))
            vel[d].work[idx, 1, b] = x[d] > hi[d] - 1e-9 ? 0.0 :
                                     d == d1 ? -x[d2] : d == d2 ? x[d1] : 0.0
        end
    end
    return (; state, fluxes, vel, fixup, schedule=GhostSchedule(state, ROT_OPS4),
            ischeds=ntuple(d -> InterfaceSchedule(fluxes[d]), D),
            spacings=block_spacings(forest, Float64),
            hook=CellBoundary((x, v, δ) -> zero(eltype(x))))
end

function advect_rhs!(du, u, p)
    D = length(p.fluxes)
    scatter!(p.state, u)
    fill_ghosts!(p.state, p.schedule; boundary=p.hook)
    ntuple(Val(D)) do d
        map_blocks!(advect_flux_kernel!, p.fluxes[d], p.fluxes[d].work, p.state.work,
                    p.vel[d].work, Val(D), Val(p.state.G), Val(d); closed=true)
        p.fixup && restrict_interfaces!(p.fluxes[d], p.ischeds[d])
        nothing
    end
    map_blocks!(advect_divergence_kernel!, p.state, statearray(du, p.state),
                map(f -> f.work, p.fluxes), p.spacings, Val(D))
    return du
end

"""
Advect a compact bump by the rigid rotation for `nsteps` SSPRK3 steps of
`dt` and return `(mass0, mass1, crossed)`: the total mass before and
after, and the mass that ended up within 30° of the low face of `d2`,
having started between 45° and 85° from it, next to the low face of
`d1`, so that it can only have got there through the seam.
"""
function advect_through_seam(forest::Forest{D}; fixup::Bool, nsteps::Int,
                             dt::Float64) where {D}
    d1, d2 = TreeAMR.rotating_dims(forest)
    p = advection_problem(forest; fixup=fixup)
    centre, w = (cos(1.13), sin(1.13)), 0.35          # 65° from the d1 axis
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
            if x[d2] < tan(π / 6) * x[d1]
                crossed += p.state.work[idx, 1, b] * h^D
            end
        end
    end
    return mass0, mass1, crossed
end

@testset "Rigid rotation conserves mass through the seam and coarse-fine faces" begin
    # The seam's two faces must exchange exactly the flux one loses and
    # the other gains, and the coarse-fine faces inside the quadrant need
    # the fixup, as anywhere: a leak at the seam or at an interface shows
    # as a drift of the total mass. The bump crosses the seam during the
    # run and the refinement boundary at radius 1.2 cuts through it; the
    # negative control drops the fixup and must drift.
    for rot in ((1, 2), (2, 1))
        forest = rotating_forest(Val(2); rotating=rot)
        @test maxlevel(forest) == 2
        hmin = minimum(k -> spacing(forest, k), forest.leaves)
        dt = 0.4 * hmin / (2 * sqrt(2.0))
        nsteps = ceil(Int, 0.45 / dt)               # a quarter of a radian and more
        m0, m1, crossed = advect_through_seam(forest; fixup=true, nsteps=nsteps, dt=dt)
        drift = abs(m1 - m0) / m0
        @test drift < 1e-13
        @test crossed > 0.1 * m0
        n0, n1, _ = advect_through_seam(forest; fixup=false, nsteps=nsteps, dt=dt)
        leak = abs(n1 - n0) / n0
        @test n0 == m0
        @test leak > 1e-3
        println("rigid rotation through the seam $rot: $nsteps steps, crossed " *
                "$(crossed / m0) of the mass, drift $drift with the fixup, $leak without")
    end
end

# --- M12 step 5: interpolation beyond the seam -------------------------------

# The orthogonal signed permutation that carries the quadrant to where a
# point lies: `r` quarter turns `R` (`R e_{d1} = e_{d2}`, `R e_{d2} =
# −e_{d1}`), then, if `mirror`, the reflection of dimension `z` at its
# low face. Written from the definition, as integer matrices.
function turn_matrix(D, (d1, d2), r::Int; mirror::Bool=false, z::Int=0)
    R = Matrix{Int}(I_(D))
    R[d1, d1], R[d2, d2], R[d2, d1], R[d1, d2] = 0, 0, 1, -1
    O = R^r
    if mirror
        M = Matrix{Int}(I_(D))
        M[z, z] = -1
        O = M * O
    end
    return O
end
I_(D) = [Int(i == j) for i in 1:D, j in 1:D]

# The value and gradient at `R^r q` (mirrored or not) of the scalar and
# the vector of `rotating_data`, from those at `q`: `s(Op) = s(p)`,
# `V(Op) = O V(p)`, and each gradient turned by `O` too. Each sum has a
# single nonzero term of weight ±1, so this is exact.
function turned_values(vq::AbstractMatrix, O::Matrix{Int})
    D = size(O, 1)
    out = similar(vq)
    comp(i, k) = i == 1 ? vq[1, k] : sum(O[i - 1, j] * vq[1 + j, k] for j in 1:D)
    for i in 1:(D + 1)
        out[i, 1] = comp(i, 1)
        for k in 1:D
            out[i, 1 + k] = sum(O[k, l] * comp(i, 1 + l) for l in 1:D)
        end
    end
    return out
end

# A complex-step first derivative of the formula `f(x, v)` along `d`:
# exact to roundoff for the polynomial data, and needing no package.
function complex_step(f, x::NTuple{D}, v, d) where {D}
    ε = 1e-20
    return imag(f(ntuple(e -> e == d ? complex(x[e], ε) : complex(x[e]), D), v)) / ε
end

@testset "Interpolation beyond the seam is the turned field: D=$D" for D in (2, 3)
    # A point beyond the seam is answered from its preimage in the
    # quadrant: a wrong fold, variable, sign or derivative remap gives a
    # wrong value there, which the polynomial data would show at once, and
    # the turned value must be the preimage's bit for bit, since the
    # stencil, its block and its arithmetic are the same. The smooth data
    # are compared against the full plane, which holds them written out:
    # at the preimage, turned, which is where the quadrant's stored points
    # are the full plane's; and at the point itself, which agrees to
    # roundoff for cell centering only. A vertex-like full plane is not
    # itself covariant under the turn at coarse-fine faces — half-open
    # ownership puts the shared plane, and a different ghost stencil, on
    # a block's high side — so there the two agree to interpolation
    # accuracy, and the quadrant is the covariant one.
    cases = D == 2 ? [((1, 2), :none), ((2, 1), :none)] :
            [((1, 2), :reflect_lo), ((3, 1), :periodic)]
    derivs = (ntuple(_ -> 0, D), ntuple(k -> ntuple(d -> Int(d == k), D), D)...)
    worst_poly, worst_full = 0.0, 0.0
    worst_direct = Dict(:cell => 0.0, :vertex => 0.0)
    for (rot, other) in cases, C in (cellcentered(D), vertexcentered(D))
        d1, d2 = rot
        z = outofplane(D, rot)
        mirrors = other === :reflect_lo
        rng = Xoshiro(hash((D, rot, C)))
        # Points inside the quadrant, away from the high faces, and their
        # images R^r q (and mirrored below z = 0, where z reflects).
        qs = [ntuple(d -> 0.05 + 1.9 * rand(rng), D) for _ in 1:60]
        image(q, r, m) = Tuple(turn_matrix(D, rot, r; mirror=m, z=z) * collect(q))
        turns = [(r, m) for r in 0:3 for m in (mirrors ? (false, true) : (false,))]

        # Polynomial data, which every operator and the interpolant
        # reproduce: every point anywhere in the plane is exact.
        p = 4
        f = rotating_data(D, rot, other; poly=p)
        fs = FieldSet(rotating_forest(Val(D); rotating=rot, other=other), D + 1;
                      G=ghosts_for(C, p), centering=C,
                      rotation=vector_rotation(D, rot),
                      parity=vector_parity(D, rot, other))
        fill_by_coordinates!(f, fs)
        fill_ghosts!(fs, GhostSchedule(fs, ROT_OPS4); boundary=boundary_by_coordinates(f))
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
            vp = interpolate(fs, [image(q, r, m)], Lagrange(4); derivs=derivs).values
            exact &= vp[:, :, 1] == turned_values(vq[:, :, iq],
                                                  turn_matrix(D, rot, r; mirror=m, z=z))
        end
        @test exact
        # `locate_point` folds the same way.
        @test all(locate_point(fs.forest, image(q, r, m)) == locate_point(fs.forest, q)
                  for q in qs[1:10] for (r, m) in turns)

        # Smooth data against the full plane, at random points anywhere.
        quad, full = quadrant_and_full(Val(D); rotating=rot, other=other, N=8)
        g = rotating_data(D, rot, other)
        par = vector_parity(D, rot, other)
        fsq = FieldSet(quad, D + 1; G=ghosts_for(C, 4), centering=C, parity=par,
                       rotation=vector_rotation(D, rot))
        fsf = FieldSet(full, D + 1; G=ghosts_for(C, 4), centering=C, parity=par)
        for s in (fsq, fsf)
            fill_by_coordinates!(g, s)
            fill_ghosts!(s, GhostSchedule(s, ROT_OPS4); boundary=boundary_by_coordinates(g))
        end
        ys = [ntuple(d -> d == z && !mirrors ? 0.05 + 1.9 * rand(rng) :
                          3.9 * rand(rng) - 1.95, D) for _ in 1:400]
        a = interpolate(fsq, ys, Lagrange(4); derivs=derivs)
        b = interpolate(fsf, ys, Lagrange(4); derivs=derivs)
        # The preimage of each point, by the test's own turns: `O' = O⁻¹`.
        Os = map(ys) do y
            r = y[d1] < 0 ? (y[d2] < 0 ? 2 : 1) : (y[d2] < 0 ? 3 : 0)
            turn_matrix(D, rot, r; mirror=mirrors && y[z] < 0, z=z)
        end
        pre = [Tuple(O' * collect(y)) for (O, y) in zip(Os, ys)]
        bq = interpolate(fsf, pre, Lagrange(4); derivs=derivs).values
        for j in eachindex(ys)
            worst_full = max(worst_full, maximum(abs.(a.values[:, :, j] .-
                                                      turned_values(bq[:, :, j], Os[j]))))
        end
        kind = first(C)
        worst_direct[kind] = max(worst_direct[kind], maximum(abs.(a.values .- b.values)))
        # Every orientation is met.
        seen = Set(Int(y[d1] < 0) + 2 * Int(y[d2] < 0) for y in ys)
        @test length(seen) == 4

        # A subset of the variables, reordered, is the same numbers.
        sub = interpolate(fsq, ys, Lagrange(4); derivs=derivs, vars=[3, 2])
        @test sub.values == a.values[[3, 2], :, :]
    end
    @test worst_poly < 1e-9
    @test worst_full < 1e-12
    @test worst_direct[:cell] < 1e-12
    @test worst_direct[:vertex] < 2e-2               # values and gradients
    println("interpolation beyond the seam D=$D: polynomial data within $worst_poly, " *
            "smooth data within $worst_full of the full plane at the turned preimage; " *
            "at the point itself $(worst_direct[:cell]) (cell) and " *
            "$(worst_direct[:vertex]) (vertex)")
end

@testset "Interpolation beyond the seam in Float32, with a region" begin
    # The fold and the turned tables in another element type, and a
    # region's flags, which are tested where the stencil is read: a
    # turned point and its preimage share both.
    rot = (1, 2)
    quad, _ = quadrant_and_full(Val(2); rotating=rot, other=:none, N=8)
    quad32 = Forest{Float32}(quad.roots; N=quad.N, rotating=rot,
                             extents=((0.0f0, 2.0f0), (0.0f0, 2.0f0)),
                             leaves=copy(quad.leaves))
    g = rotating_data(2, rot, :none)
    derivs = ((0, 0), (1, 0), (0, 1))
    res = map((quad, quad32)) do forest
        T = floattype(forest)
        fs = FieldSet{T}(forest, 3; G=2, rotation=vector_rotation(2, rot))
        fill_by_coordinates!(g, fs)
        fill_ghosts!(fs, GhostSchedule(fs, ROT_OPS4); boundary=boundary_by_coordinates(g))
        rng = Xoshiro(5)
        qs = [(T(0.05 + 1.9 * rand(rng)), T(0.05 + 1.9 * rand(rng))) for _ in 1:40]
        ps = [(-q[2], q[1]) for q in qs]                 # R q, r = 1
        region = Ellipsoid((T(0.6), T(0.4)), (T(0.3), T(0.5)))
        vq = interpolate(fs, qs, Lagrange(4); derivs=derivs, exclude=region)
        vp = interpolate(fs, ps, Lagrange(4); derivs=derivs, exclude=region)
        O = turn_matrix(2, rot, 1)
        @test all(j -> vp.values[:, :, j] == turned_values(vq.values[:, :, j], O), 1:40)
        @test vp.excluded == vq.excluded
        @test 0 < count(vp.excluded) < 40
        vp.values
    end
    @test eltype(res[2]) === Float32
    @test maximum(abs.(res[2] .- res[1])) < 1e-4
    println("Float32 beyond the seam: within $(maximum(abs.(res[2] .- res[1]))) of Float64")
end

@testset "A turned point reads variables it was not asked for" begin
    # The failure mode: the turn takes a requested component to one that is
    # not in `vars` — the metric's g_xz to g_yz — and an implementation that
    # only loaded the requested variables would read the wrong one. The
    # kernel reads the turned variable from the whole field set, so `vars`
    # needs no closure under the rotation. (Metric order xx, xy, xz, yy, yz,
    # zz; under R, g_xz′ = −g_yz.)
    rot = (4, -2, -5, 1, 3, 6)
    par = [ntuple(d -> d == 3 && v in (3, 5) ? OddParity : EvenParity, 3) for v in 1:6]
    forest = Forest((2, 2, 2); N=8, rotating=(1, 2),
                    reflecting=((false, false), (false, false), (true, false)))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    fs = FieldSet(forest, 6; G=2, centering=vertexcentered(3), parity=par, rotation=rot)
    f(x, v) = sin(0.7x[1] + 0.3v) * cos(0.4x[2] - 0.2v) * (1 + 0.1v * x[3]^2) +
              0.05v * x[1] * x[2]
    fill_by_coordinates!(f, fs)
    fill_ghosts!(fs, GhostSchedule(fs, ROT_OPS4); boundary=boundary_by_coordinates(f))
    derivs = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
              (2, 0, 0), (1, 1, 0), (0, 2, 0))
    q = (0.83, 0.41, 0.57)
    p = (-q[2], q[1], q[3])                         # R q, beyond the low x face
    a = interpolate(fs, [p], Lagrange(4); derivs=derivs, vars=[3]).values[1, :, 1]
    b = interpolate(fs, [q], Lagrange(4); derivs=derivs, vars=[5]).values[1, :, 1]
    # g_xz(p) = −g_yz(q) with q = (p_y, −p_x, p_z): ∂ₓ ↦ −∂_y, ∂_y ↦ ∂ₓ,
    # once per order, so ∂ₓ² ↦ ∂_y², ∂_y² ↦ ∂ₓ² and ∂ₓ∂_y ↦ −∂_y∂ₓ.
    @test a == [-b[1], b[3], -b[2], -b[4], -b[7], b[6], -b[5]]
end

@testset "A set that turns into its partner refuses points beyond the seam" begin
    # A face-centered set's value beyond the seam is its partner's, turned;
    # read from itself it would be the wrong component in the wrong
    # layout, so the point is refused with the reason, after the launch,
    # as an outside point is — and inside the quadrant it interpolates.
    forest = rotating_forest(Val(2); rotating=(1, 2))
    Bx = FieldSet(forest, 1; G=(1, 2), centering=facecentered(2, 1), rotation=(-1,))
    By = FieldSet(forest, 1; G=(2, 1), centering=facecentered(2, 2), rotation=(1,))
    fill_by_coordinates!((x, v) -> x[1] + 2x[2], Bx)
    fill_ghosts!(RotationPair(Bx, By), (GhostSchedule(Bx, ROT_OPS4),
                                        GhostSchedule(By, ROT_OPS4));
                 boundary=CellBoundary((x, v, δ) -> zero(eltype(x))))
    @test interpolate(Bx, [(0.3, 0.7)], Lagrange(2)).values[1] ≈ 0.3 + 1.4
    @test_throws "lies 1 quarter turn(s) away across the rotating seam" interpolate(
        Bx, [(0.3, 0.7), (-0.2, 0.5)], Lagrange(2))
    @test_throws "Interpolate the partner at the turned point" interpolate(
        By, [(-0.2, -0.5)], Lagrange(2))
    # Beyond a high face, turned or not, is outside, and the faces say
    # which are the seam.
    @test_throws "rotating seam below" interpolate(Bx, [(2.5, 0.5)], Lagrange(2))
    @test_throws "is outside the domain" interpolate(Bx, [(-0.5, 2.5)], Lagrange(2))
    fs = FieldSet(forest, 1; G=2, rotation=(1,))
    @test_throws "is outside the domain" interpolate(fs, [(-2.5, 0.5)], Lagrange(2))
    @test locate_point(forest, (-2.5, 0.5)) === nothing
end

# --- M12 step 8: the wave equation on a quadrant ------------------------------
#
# The acceptance test of the whole seam: a quadrant evolved through many
# fills must stay the quarter of the full plane it stands for. The scalar
# wave and a two-component vector wave (each Cartesian component of a
# covariant vector field obeys the wave equation, and the turn mixes
# them) are evolved on the quadrant of `quadrant_and_full` and on the
# full plane it folds, with the same refinement, the same steps and the
# exact solution in the hook on the outer faces of both.
#
# The exact solution is a sum of the four turns of one standing mode `f`,
# an eigenfunction of the Laplacian that has no symmetry of its own,
# so the sum is invariant under the quarter turn and under no mirror.
# It is summed as `(t₀ + t₂) + (t₁ + t₃)`, `t_r = f(Rʳp)`: at `Rp` the
# terms come round as `(t₁ + t₃) + (t₂ + t₀)`, which floating-point
# addition, being commutative, makes the same bits, so the data is
# invariant bit for bit, not only to roundoff. The vector is its
# gradient, `∇φ(p) = Σ R⁻ʳ ∇f(Rʳp)`, summed the same way, and covariant
# bit for bit for the same reason. That is what lets the vertex-centered
# seam planes, which the quadrant owns twice, start out equal.

const ROTWAVE_α, ROTWAVE_β = π / 2, 3π / 4
const ROTWAVE_ω = sqrt(ROTWAVE_α^2 + ROTWAVE_β^2)

# `R^r (a, b)`, `R(a, b) = (−b, a)`: exact, signs and swaps only.
function rotwave_turn((a, b), r::Int)
    for _ in 1:mod(r, 4)
        a, b = -b, a
    end
    return a, b
end

rotwave_f((a, b)) = cos(ROTWAVE_α * a + 0.3) * cos(ROTWAVE_β * b + 0.7)
rotwave_∇f((a, b)) = (-ROTWAVE_α * sin(ROTWAVE_α * a + 0.3) * cos(ROTWAVE_β * b + 0.7),
                      -ROTWAVE_β * cos(ROTWAVE_α * a + 0.3) * sin(ROTWAVE_β * b + 0.7))

function rotwave_phi(p)
    t = ntuple(r -> rotwave_f(rotwave_turn(p, r - 1)), 4)
    return (t[1] + t[3]) + (t[2] + t[4])
end

function rotwave_gradient(p)
    w = ntuple(r -> rotwave_turn(rotwave_∇f(rotwave_turn(p, r - 1)), -(r - 1)), 4)
    return ntuple(e -> (w[1][e] + w[3][e]) + (w[2][e] + w[4][e]), 2)
end

"""
The exact standing wave at time `t` as `(x, v) -> value`: with `K = 1`
the scalar `(u, ∂ₜu)`, with `K = 2` the vector `(v_{d1}, v_{d2}, ∂ₜv_{d1},
∂ₜv_{d2})` in the coordinates about the axis, `(a, b) = (x_{d1}, x_{d2})`.
"""
function rotwave_exact(K::Int, (d1, d2), t)
    c, s = cos(ROTWAVE_ω * t), -ROTWAVE_ω * sin(ROTWAVE_ω * t)
    return function (x, v)
        p = (x[d1], x[d2])
        k = v > K ? v - K : v
        value = K == 1 ? rotwave_phi(p) : rotwave_gradient(p)[k]
        return (v > K ? s : c) * value
    end
end

"""
The rotation map of the wave's state: the scalar's `(1, 2)`, or the
vector's from `vector_rotation` (the oracle's, written from `v(Rp) =
Rv(p)`), its components numbered from 1, then the same for `∂ₜv`.
"""
function rotwave_rotation(K::Int, rotating)
    K == 1 && return [1, 2]
    m = [sign(q) * (abs(q) - 1) for q in vector_rotation(2, rotating)[2:end]]
    return [m; [sign(q) * (abs(q) + 2) for q in m]]
end

# The Laplacian of each of `K` components with the two neighbours added
# first, `(u₊ + u₋) − 2u₀`, so that a point and its image under the turn,
# whose neighbours are each other's in the other order, compute the same
# bits; `wave_rhs_kernel!` subtracts first, which only roundoff tells
# apart.
@kernel function rotwave_kernel!(du, @Const(work), @Const(spacings), ::Val{D},
                                 ::Val{G}, ::Val{K}) where {D,G,K}
    I = @index(Global, NTuple)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))
    c = ntuple(d -> I[d] + G[d], Val(D))
    h = spacings[b]
    for k in 1:K
        u0 = work[c..., k, b]
        lap = zero(eltype(du))
        for d in 1:D
            up = Base.setindex(c, c[d] + 1, d)
            um = Base.setindex(c, c[d] - 1, d)
            lap += (work[up..., k, b] + work[um..., k, b]) - 2 * u0
        end
        du[inner..., k, b] = work[c..., K + k, b]
        du[inner..., K + k, b] = lap / (h * h)
    end
end

function rotwave_rhs!(du, u, p, t)
    scatter!(p.fs, u)
    fill_ghosts!(p.fs, p.schedule;
                 boundary=boundary_by_coordinates(rotwave_exact(p.K, p.rotating, t)))
    map_blocks!(rotwave_kernel!, p.fs, statearray(du, p.fs), p.fs.work, p.spacings,
                Val(2), p.valG, Val(p.K))
    return nothing
end

"""
Evolve the standing wave with `K` components on the quadrant of
`quadrant_and_full(Val(2); rotating)` or, with `full = true`, on the full
plane, by fixed-step RK4 to a quarter period, and return the errors
against the exact solution, the step count, and the field set holding
the final state with its ghosts filled.
"""
function rotating_wave(; N, C, K, rotating=(1, 2), full::Bool, p=4, cfl=0.25,
                       periods=0.25)
    quad, plane = quadrant_and_full(Val(2); rotating=rotating, other=:periodic, N=N)
    forest = full ? plane : quad
    G = ghosts_for(C, p)
    rotation = full ? nothing : rotwave_rotation(K, rotating)
    fs = FieldSet(forest, 2K; G=G, centering=C, rotation=rotation)
    schedule = GhostSchedule(fs, Operators(prolongation=p, restriction=p))
    problem = (; fs, schedule, K, rotating, valG=Val(G),
               spacings=block_spacings(forest, Float64))
    fill_by_coordinates!(rotwave_exact(K, rotating, 0.0), fs)
    u0 = statevector(fs)
    gather!(u0, fs)
    h = minimum_spacing(forest)
    t_end = periods * 2π / ROTWAVE_ω
    nsteps = ceil(Int, t_end / (cfl * h))
    prob = ODEProblem(rotwave_rhs!, u0, (0.0, t_end), problem)
    sol = solve(prob, RK4(); dt=t_end / nsteps, adaptive=false, save_everystep=false)
    exact = FieldSet(forest, 2K; G=G, centering=C, rotation=rotation)
    fill_by_coordinates!(rotwave_exact(K, rotating, t_end), exact)
    uexact = statevector(exact)
    gather!(uexact, exact)
    err = sol.u[end] .- uexact
    scatter!(fs, sol.u[end])
    fill_ghosts!(fs, schedule; boundary=boundary_by_coordinates(rotwave_exact(K, rotating,
                                                                                t_end)))
    return (l2=volume_weighted_norm(fs, err), linf=volume_weighted_norm(fs, err; p=Inf),
            h=h, nsteps=nsteps, nblocks=nleaves(forest), fs=fs)
end

"""
The worst difference between the vertex-centered quadrant's two owned
seam planes, the points `(0, s)` on the low face of `d1` and `(s, 0)` on
the low face of `d2`, which are one point under the turn: the first must
be the second's values through the signed map. Returns `(worst, npairs)`,
`worst` compared with `==` semantics; the axis itself is skipped.
"""
function seam_plane_mismatch(fs::FieldSet{T,2}) where {T}
    d1, d2 = rotating_dims(fs.forest)
    planes = (Dict{T,Vector{T}}(), Dict{T,Vector{T}}())
    for b in 1:nblocks(fs), (side, d, e) in ((1, d1, d2), (2, d2, d1))
        block_extent(fs.forest, blockkey(fs, b))[d][1] == 0 || continue
        for j in 1:(fs.forest.N)
            idx = (0, 0)
            idx = Base.setindex(idx, fs.G[d] + 1, d)
            idx = Base.setindex(idx, fs.G[e] + j, e)
            s = coordinates(fs, b, idx)[e]
            s == 0 && continue
            planes[side][s] = fs.work[idx..., :, b]
        end
    end
    @assert sort!(collect(keys(planes[1]))) == sort!(collect(keys(planes[2])))
    worst = 0.0
    for (s, here) in planes[1]
        there = planes[2][s]
        for v in eachindex(here)
            q = fs.rotation[v]
            turned = sign(q) * there[abs(q)]
            worst = here[v] == turned ? worst : max(worst, abs(here[v] - turned))
        end
    end
    return worst, length(planes[1])
end

"""
How far the full plane's own solution is from covariant: the worst
difference, over its owned points `p` in the quadrant `x_{d1}, x_{d2} ≥ 0`
and each turn `r = 1, 2, 3`, between its values at `Rʳp`, where those are
owned too, and its values at `p` turned by the signed map `rotation`.
"""
function turned_defect(fs::FieldSet{T,2}, rotation) where {T}
    d1, d2 = 1, 2
    owned = Dict{NTuple{2,T},Vector{T}}()
    for b in 1:nblocks(fs)
        for i in CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + fs.forest.N), 2))
            owned[coordinates(fs, b, Tuple(i))] = fs.work[i, :, b]
        end
    end
    worst = 0.0
    for (p, u) in owned, r in 1:3
        (p[d1] >= 0 && p[d2] >= 0) || continue
        there = get(owned, rotwave_turn(p, r), nothing)
        there === nothing && continue
        for _ in 1:r
            u = [sign(q) * u[abs(q)] for q in rotation]
        end
        worst = max(worst, maximum(abs.(there .- u)))
    end
    return worst
end

@testset "The wave on a quadrant evolves as the full plane it folds: K=$K" for K in (1, 2)
    # Many fills compound whatever one fill gets wrong: a wrong turn of
    # the ghosts, of the vector's components or of their signs, or a
    # seam treated as an outer face, makes the quadrant's solution drift
    # from the full plane's. Cell-centered the quadrant owns exactly the
    # full plane's points in its quarter and the two agree to roundoff at
    # every stored point. Vertex-centered they cannot: the full plane is
    # not covariant itself, because which side owns the shared plane of a
    # coarse-fine face (the block above it) is not turned with the mesh,
    # so a face whose plane the coarse block evolves becomes, a half turn
    # away, one whose plane the fine block evolves. The quadrant, which
    # is covariant by construction, must then lie within the full plane's
    # own defect of it; and it owns the two seam planes twice, as the
    # same points, and must keep them equal.
    rotation = rotwave_rotation(K, (1, 2))
    for C in (cellcentered(2), vertexcentered(2))
        full = rotating_wave(; N=16, C=C, K=K, full=true)
        quad = rotating_wave(; N=16, C=C, K=K, full=false)
        @test quad.nsteps == full.nsteps
        @test 4 * quad.nblocks == full.nblocks
        @test quad.linf ≈ full.linf rtol = 1e-8
        worst, npoints = compare_matching_blocks(quad.fs, full.fs)
        defect = turned_defect(full.fs, rotation)
        @test npoints > 0
        line = "wave on a quadrant, K=$K, $(C[1]): $(quad.nsteps) steps, " *
               "linf $(quad.linf) against $(full.linf), every stored point within " *
               "$worst, the full plane covariant to $defect"
        if C == cellcentered(2)
            @test worst < 1e-12
            @test defect < 1e-12
        else
            @test 0 < worst < defect < 1e-2
            seam, npairs = seam_plane_mismatch(quad.fs)
            @test npairs > 0
            @test seam == 0
            line *= ", seam planes $seam apart over $npairs pairs"
        end
        println(line)
    end
end

@testset "The wave on a quadrant converges at the interface-order rate" begin
    # Order-4 operators against the 2nd-order Laplacian: rate 2, as on
    # the periodic box and the reflecting half box. The turned ghosts are
    # copies and the prolongations and restrictions next to the seam the
    # ordinary ones read in the virtual frame, so the seam may cost
    # nothing; an inconsistency there (a ghost off by one point, a
    # half-turned edge) shows as a rate below 2 long before a single fill
    # looks wrong.
    for C in (cellcentered(2), vertexcentered(2)), K in (1, 2)
        hs, l2, linf = Float64[], Float64[], Float64[]
        for N in (8, 16, 32)
            r = rotating_wave(; N=N, C=C, K=K, full=false)
            push!(hs, r.h)
            push!(l2, r.l2)
            push!(linf, r.linf)
        end
        rl2, rlinf = convergence_rate(hs, l2), convergence_rate(hs, linf)
        @test rl2 ≈ 2.0 atol = 0.15
        @test rlinf ≈ 2.0 atol = 0.2
        println("wave on a quadrant, K=$K, $(C[1]): rates $rl2 (l2), $rlinf (linf); " *
                "l2 $l2")
    end
end
