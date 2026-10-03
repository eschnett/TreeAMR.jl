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

    # Until the schedule, the interface schedule, interpolation and the
    # checkpoint learn the orientation, each refuses the seam rather
    # than read across it as though it were an ordinary face.
    forest = Forest((2, 2); N=8, rotating=(1, 2))
    fs = FieldSet(forest, 1; G=1)
    ops = Operators(prolongation=4, restriction=4)
    @test_throws "not implemented yet in this step of M12" GhostSchedule(fs, ops)
    @test_throws "GhostSchedule over a rotating forest" GhostSchedule(fs, ops)
    vs = FieldSet(forest, 1; G=0, centering=facecentered(2, 1))
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
