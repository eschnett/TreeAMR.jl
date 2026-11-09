# M17: a forest of patches — several bricks of roots glued at their faces by
# signed axis permutations. This file holds the forest (step 1), the
# neighbor search across the glue (step 2), the ghost fill across it
# (step 3), conformity at the glue (step 4), and the patch handed to the
# coordinate callbacks, with the refusals of point location (step 5).
#
# Every claim about where a step lands is stated against a continuous map
# written out independently of the glue table: by hand for the small
# forests, and for the cubed sphere by folding a point that leaves one face
# of the cube over the edge onto the next. The tree's answer is a node at
# a level; the oracle's is the region of logical space beyond a block,
# carried across the face and intersected with the leaves of the patch it
# lands in.
#
# This file's definitions are in `helpers/patch.jl`.

using TreeAMR: Crossing, OneBrick, Patches, ForestDigest, patch_keywords,
               patch_roots, glue_slot, glue_axes, neighbor_anchor, oriented_neighbors,
               wall_direction, virtual_offset, childoffset

@testset "Patches are refused where they cannot be glued" begin
    # A list of patches or a glue table that does not describe bricks
    # meeting face to face, root for root, must be refused with its
    # reason, not found later as a wrong neighbor or a ghost read from the
    # wrong place.
    ring = [(roots=(2, 1),), (roots=(2, 1),)]
    ringglue = [patch_glue_both((1, 1, 2), (2, 1, 1), (1, 2));
                patch_glue_both((2, 1, 2), (1, 1, 1), (1, 2))]
    pair = patch_glue_both((1, 1, 2), (2, 1, 1), (1, 2))
    # The seams are the brick form's, and a wrap is a glue.
    @test_throws "takes no seam" Forest(ring; N=4, glue=ringglue, rotating=(1, 2))
    @test_throws "takes no seam" Forest(ring; N=4, glue=ringglue, halfturn=(2, 1))
    @test_throws "a wrap is a patch glued to its own opposite face" Forest(
        ring; N=4, glue=ringglue, periodic=(true, false))
    @test_throws "a wrap is a patch glued to its own opposite face" Forest(
        [(roots=(2, 1),)]; N=4, periodic=(false, true))
    @test_throws "patch by patch" Forest(ring; N=4, glue=pair,
                                         extents=((0, 2), (0, 1)))
    # One cell size, and root counts that agree along a glued face.
    @test_throws "differ in cell size" Forest(
        [(roots=(2, 1),), (roots=(2, 1), extents=((0, 4), (0, 2)))]; N=4, glue=pair)
    @test_throws "root for root" Forest(
        [(roots=(2, 1),), (roots=(2, 2),)]; N=4, glue=pair)
    @test_throws "root for root" Forest(
        [(roots=(2, 3),), (roots=(2, 3),)]; N=4,
        glue=patch_glue_both((1, 1, 2), (2, 2, 1), (2, -1)))
    # The map: a signed permutation, of the right length, carrying the
    # face onto the face it is glued to.
    glue_with(σ, τ=σ) = [(from=(1, 1, 2), to=(2, 1, 1), map=σ),
                         (from=(2, 1, 1), to=(1, 1, 2), map=τ)]
    two = [(roots=(2, 2),), (roots=(2, 2),)]
    @test_throws "sends two axes onto one" Forest(two; N=4, glue=glue_with((1, 1)))
    @test_throws "±1 through ±2" Forest(two; N=4, glue=glue_with((1, 3)))
    @test_throws "±1 through ±2" Forest(two; N=4, glue=glue_with((1, 0)))
    @test_throws "with 3 entries" Forest(two; N=4, glue=glue_with((1, 2, 3)))
    @test_throws "not a tuple of integers" Forest(two; N=4, glue=glue_with([1, 2]))
    @test_throws "sends the normal" Forest(two; N=4, glue=glue_with((2, 1)))
    @test_throws "sends the normal" Forest(two; N=4, glue=glue_with((-1, 2)))
    # Both directions, mutually inverse, and each face glued once.
    @test_throws "is not glued back" Forest(two; N=4, glue=glue_with((1, 2))[1:1])
    @test_throws "each the inverse of the other" Forest(two; N=4,
                                                        glue=glue_with((1, -2), (1, 2)))
    @test_throws "each the inverse of the other" Forest(
        [(roots=(2, 2),) for _ in 1:3]; N=4,
        glue=[(from=(1, 1, 2), to=(2, 1, 1), map=(1, 2)),
              (from=(2, 1, 1), to=(3, 1, 2), map=(1, 2)),
              (from=(3, 1, 2), to=(2, 1, 1), map=(1, 2))])
    @test_throws "glued twice" Forest(two; N=4,
                                      glue=[glue_with((1, 2)); glue_with((1, 2))[1:1]])
    # A glued face is not a wall.
    @test_throws "is not a wall" Forest(two; N=4, glue=glue_with((1, 2)),
                                        reflecting=((false, true), (false, false)))
    @test_throws "is not a wall" Forest(two; N=4, glue=glue_with((1, 2)),
                                        outflow=((true, false), (false, false)))
    # The shape of the arguments.
    @test_throws "the forest has patches 1:2" Forest(
        two; N=4, glue=[(from=(3, 1, 2), to=(2, 1, 1), map=(1, 2))])
    @test_throws "outside 1:2" Forest(
        two; N=4, glue=[(from=(1, 3, 2), to=(2, 1, 1), map=(1, 2))])
    @test_throws "a side is 1 for the low face" Forest(
        two; N=4, glue=[(from=(1, 1, 3), to=(2, 1, 1), map=(1, 2))])
    @test_throws "of three integers" Forest(
        two; N=4, glue=[(from=(1, 1), to=(2, 1, 1), map=(1, 2))])
    @test_throws "is not a named tuple `(from" Forest(two; N=4, glue=[((1, 1, 2), (2, 1, 1))])
    @test_throws "list of patches is empty" Forest(NamedTuple[]; N=4)
    @test_throws "is not a named tuple `(roots" Forest([(2, 2)]; N=4)
    @test_throws "has only `roots` and `extents`" Forest([(roots=(2, 2), glue=())]; N=4)
    @test_throws "different lengths" Forest([(roots=(2, 2),), (roots=(2,),)]; N=4)
    @test_throws "must all be positive" Forest([(roots=(2, 0),)]; N=4)
    @test_throws "of patch 2 give anisotropic" Forest(
        [(roots=(2, 2),), (roots=(2, 2), extents=((0, 2), (0, 3)))]; N=4)
    @test_throws "pairs `(lo, hi)`" Forest([(roots=(2, 2), extents=(0, 2))]; N=4)
    @test_throws "at most 127" Forest([(roots=(1, 1, 1),) for _ in 1:22]; N=4)
    @test_throws "N must be even" Forest(two; N=3)
    @test_throws "moved from the forest" Forest(two; N=4, G=2)
    # A leaf list over every patch, checked as over a brick.
    @test_throws "the 2 patches have roots 0:7" Forest(
        two; N=4, leaves=[MortonKey{2}(r, 0, (0, 0)) for r in 0:8])
    # What is not refused: patches with the same extents, the cubed
    # sphere's natural layout, and patches with no glue at all.
    @test npatches_of(Forest(two; N=4, glue=glue_with((1, 2)))) == 2
    @test npatches_of(Forest([(roots=(1, 1),), (roots=(3, 3),)]; N=4)) == 2
end

@testset "A one-patch forest is the brick, and the brick is what it was: D=$D" for D in
                                                                                    (1, 2, 3)
    # The patches form must not change a forest that has nothing to glue,
    # and the brick form must not change at all: its connectivity is the
    # singleton, and its digest is the hash it was before M17, so that the
    # ranks of a run and a TreeIOHDF5 checkpoint see the same forest.
    roots = ntuple(d -> d + 1, D)
    extents = ntuple(d -> (-1, roots[d] - 1), D)
    reflecting = ntuple(d -> (d == 1, false), D)
    brick = Forest(roots; N=4, extents=extents, reflecting=reflecting)
    one = Forest([(roots=roots, extents=extents)]; N=4, reflecting=reflecting)
    @test brick.conn === OneBrick{D}()
    @test one.conn isa Patches{D,Float64}
    @test !one.conn.corners
    @test ForestDigest(brick, UInt(0), false).brick ==
          hash(string((brick.roots, brick.N, brick.periodic, brick.faces,
                       Int.(brick.rotating), brick.extents)))
    @test ForestDigest(one, UInt(0), false).brick != ForestDigest(brick, UInt(0), false).brick
    @test patch_keywords(brick) === nothing
    # Refined the same way — the first leaf, three times over, then
    # balanced — they agree on everything the tree and the geometry say.
    for forest in (brick, one), _ in 1:3
        refine!(forest, first(forest.leaves))
        balance!(forest)
    end
    @test brick.leaves == one.leaves
    @test (brick.roots, brick.extents, brick.faces, brick.periodic) ==
          (one.roots, one.extents, one.faces, one.periodic)
    @test block_origins(brick) == block_origins(one)
    @test block_spacings(brick) == block_spacings(one)
    @test block_patches(brick) == block_patches(one) == ones(Int32, nleaves(brick))
    @test isbalanced(one)
    for k in brick.leaves
        @test patch(brick, k) == patch(one, k) == 1
        @test root_position(brick, k.root) == root_position(one, k.root)
        @test root_index(one, root_position(one, k.root)) == k.root
        @test root_index(one, 1, root_position(one, k.root)) == k.root
        for δ in alldirections(Val(D))
            @test neighbor_keys(brick, k, δ) == neighbor_keys(one, k, δ)
            @test wall_direction(brick, k, δ) == wall_direction(one, k, δ)
        end
    end
end

@testset "Patches are numbered in order, each a brick of its own" begin
    # The accessors must place a root in its own patch, at its own
    # position there, and refuse what is ambiguous on several patches; a
    # forest rebuilt from its patch keywords must be the same forest.
    forest = patch_pair()
    @test nleaves(forest) == 12
    @test [patch(forest, r) for r in 0:11] == [fill(1, 6); fill(2, 6)]
    @test patch(forest, forest.leaves[7]) == 2
    @test forest.roots == (2, 3) && forest.extents == ((0.0, 2.0), (0.0, 3.0))
    @test root_position(forest, 0) == (0, 0)
    @test root_position(forest, 5) == (1, 2)
    @test root_position(forest, 6) == (0, 0)
    @test root_position(forest, 11) == (2, 1)
    @test root_index(forest, 1, (1, 2)) == 5
    @test root_index(forest, 2, (2, 1)) == 11
    @test all(r -> root_index(forest, patch(forest, r), root_position(forest, r)) == r,
              0:11)
    @test_throws "is ambiguous on a forest of 2 patches" root_index(forest, (0, 0))
    @test_throws "is not a patch of the forest" root_index(forest, 3, (0, 0))
    @test block_patches(forest) == Int32[fill(1, 6); fill(2, 6)]
    # Each patch's own logical box, with the one cell size.
    @test block_origin(forest, forest.leaves[7]) == (0.0, 5.0)
    @test block_origin(forest, forest.leaves[12]) == (2.0, 6.0)
    @test all(==(root_spacing(forest)), block_spacings(forest))
    # Rebuilt from its keywords, refined, it is the same forest, in canonical
    # order whatever order the glue was given in.
    patch_refine_all!(forest)
    kw = patch_keywords(forest)
    @test length(kw.glue) == 4
    rebuilt = Forest(kw.patches; N=4, glue=reverse(kw.glue), leaves=forest.leaves)
    digest(f) = (d = ForestDigest(f, UInt(0), false); (d.nleaves, d.leaves, d.brick))
    @test digest(rebuilt) == digest(forest)
    @test rebuilt.conn.glue == forest.conn.glue
    @test patch_keywords(rebuilt) == kw
    # A different map is a different forest.
    turned = Forest([(roots=(2, 2),), (roots=(2, 2),)]; N=4,
                    glue=patch_glue_both((1, 1, 2), (2, 1, 1), (1, -2)))
    plain = Forest([(roots=(2, 2),), (roots=(2, 2),)]; N=4,
                   glue=patch_glue_both((1, 1, 2), (2, 1, 1), (1, 2)))
    @test ForestDigest(turned, UInt(0), false).brick != ForestDigest(plain, UInt(0), false).brick
end

@testset "A patch corner is recorded where some patch is glued along two dimensions" begin
    # The schedule refuses a ghost region that crosses two glued faces from
    # this flag (step 3), so it must be set exactly where such a region
    # exists: never on a ring, whose patches are glued along one dimension
    # only, always on a cubed sphere.
    @test !patch_ring(4, (2, 1)).conn.corners
    @test !patch_ring(3, (2, 2, 2)).conn.corners
    @test !patch_loop().conn.corners
    @test patch_pair().conn.corners
    sphere = cubed_sphere()
    @test sphere.conn.corners
    @test nleaves(sphere) == 24
    # The cubed sphere's glue has swapped axes and reversed ones, so the
    # tests below exercise both.
    maps = [g.map for g in cubed_sphere_glue()]
    @test any(σ -> abs(σ[1]) == 2, maps)
    @test any(σ -> any(<(0), σ), maps)
end

@testset "A step across a glued face lands where the face's map says: $name" for (name,
                                                                                   make,
                                                                                   across) in
    (("1D loop", patch_loop, patch_loop_across),
     ("2D ring", () -> patch_ring(4, (2, 1)), patch_ring_across),
     ("2D swapped and reversed", patch_pair, patch_pair_across),
     ("3D cubed sphere", cubed_sphere, cubed_sphere_across))
    # The neighbor search must carry a step out of a patch onto the node
    # the glue's affine map takes it to, at every level and under every
    # axis permutation and reversal, and find no neighbor for a step out
    # through two faces at once — beyond a patch corner — nor through a
    # face that is not glued. Checked at the roots, refined uniformly once,
    # and, below the cubed sphere, with two levels side by side, where an
    # edge across a glue meets finer leaves and the search descends in the
    # frame of the patch across.
    forest = make()
    D = length(forest.roots)
    for stage in 1:(name == "3D cubed sphere" ? 2 : 3)
        stage == 2 && patch_refine_all!(forest)
        if stage == 3
            refine!(forest, forest.leaves[[1, length(forest.leaves) ÷ 2 + 1]])
            balance!(forest)
        end
        mismatches, seen = patch_neighbor_mismatches(forest, across)
        @test mismatches == 0
        @test get(seen, :glued, 0) > 0
        @test get(seen, :inside, 0) > 0
        forest.conn.corners && @test get(seen, :corner, 0) > 0
        stage == 3 && D == 2 && name != "2D ring" && @test get(seen, :finer, 0) > 0
        @test isbalanced(forest)
    end
end

@testset "A patch glued to its own opposite face is a periodic brick" begin
    # A wrap is a self-glue on a forest of patches: the leaves it finds
    # must be the periodic brick's, crossing the glue where the brick
    # wraps. Only one dimension wraps, since two would meet at a patch
    # corner, which a periodic brick does not have.
    brick = Forest((3, 2); N=4, periodic=(true, false))
    wrap = Forest([(roots=(3, 2),)]; N=4,
                  glue=patch_glue_both((1, 1, 2), (1, 1, 1), (1, 2)))
    @test !wrap.conn.corners
    patch_refine_all!(brick)
    patch_refine_all!(wrap)
    crossings = 0
    for k in brick.leaves, δ in alldirections(Val(2))
        @test neighbor_keys(brick, k, δ) == neighbor_keys(wrap, k, δ)
        crossings += first(oriented_neighbors(wrap, k, δ)).glue != 0
    end
    @test crossings > 0
end

@testset "A uniform forest of patches is balanced, and stays so" begin
    # The balance machinery must accept a forest of patches as it accepts a
    # brick: balance! leaves a uniform one alone, isbalanced says so, and
    # the checked leaf list takes it back.
    for make in (() -> patch_ring(4, (2, 1)), patch_pair, cubed_sphere)
        forest = make()
        for _ in 1:(length(forest.roots) == 3 ? 1 : 2)
            patch_refine_all!(forest)
        end
        leaves = copy(forest.leaves)
        @test isbalanced(forest)
        balance!(forest)
        @test forest.leaves == leaves
        kw = patch_keywords(forest)
        @test Forest(kw.patches; N=4, glue=kw.glue, leaves=leaves).leaves == leaves
    end
end

@testset "balance! keeps the leaves on either side of a glued face at one level" begin
    # No coarse-fine face may cross a glue (M17 is uniform across them),
    # so a refinement beside a glued face must ripple across it, as across
    # a seam's face, where a brick would allow one level of difference.
    ring = patch_ring(4, (2, 1))
    brick = Forest((8, 1); N=4)
    # The leaf at the high end of patch 1, against the glue to patch 2, and
    # the brick's leaf in the same place.
    refine!(ring, ring.leaves[2])
    refine!(brick, brick.leaves[2])
    @test !isbalanced(ring)
    @test isbalanced(brick)
    balance!(ring)
    @test isbalanced(ring)
    @test nleaves(ring) == 8 + 3 + 3
    @test all(k -> level(k) == (k.root in (1, 2) ? 1 : 0), ring.leaves)
end

# --- The ghost fill across the glue (step 3) ----------------------------------

@testset "A brick cut into patches glued as it was fills as the brick: D=$D" for D in (1, 2,
                                                                                       3)
    # A glued face with the identity map is the brick's own face under
    # another name: every stored point of every block, ghosts included, must
    # come back bit for bit as the brick's, for every centering, refined
    # beside the glue too, so that the edges across it are coarse-fine. A
    # wrong source, a wrong offset, a glued transfer missing or turned when
    # it should not be all show.
    roots = ntuple(d -> d == 1 ? 4 : 2, D)
    f = makepoly(D, 2)
    for refined in (false, true), C in patch_centerings(Val(D))
        refined && first(C) === :element && continue
        patched, place = brick_cut(roots, ((2,),))
        refined && patch_refine_one!(patched, place)
        brick = Forest(roots; N=4, leaves=laid_out_leaves(patched, place, roots))
        fsp, fsb = patch_fieldset(patched, C), patch_fieldset(brick, C)
        for fs in (fsp, fsb)
            fill_by_coordinates!(f, fs)
            fill_ghosts!(fs, GhostSchedule(fs, patch_ops(fs));
                         boundary=boundary_by_coordinates(f))
        end
        result = compare_matching_blocks(fsp, fsb)
        @test result !== nothing
        result === nothing && continue
        worst, npoints = result
        @test worst == 0
        @test npoints == length(fsp.work)
    end
end

@testset "Cut at a patch corner, an element set fills as the brick: D=$D" for D in (2, 3)
    # Cut along two dimensions, every patch has a patch corner. An element
    # set is filled on faces only and never reaches one, so it must come
    # back as the brick's; so must a set whose ghost widths leave the corner
    # region empty. Any other is refused at schedule build, with its reason,
    # rather than handed to the boundary hook as though the corner were an
    # outer face.
    roots = ntuple(d -> d == 1 ? 4 : 2, D)
    patched, place = brick_cut(roots, ((2,), (1,)))
    @test patched.conn.corners
    brick = Forest(roots; N=4)
    f = makepoly(D, 2)
    # No ghosts along dimension 1 takes the conservative family's order 1.
    cons1 = Operators(prolongation=1, restriction=2, family=Conservative)
    for (C, G, ops) in ((elementcentered(D), 1, PATCH_ELEMENT_OPS),
                        (cellcentered(D), ntuple(d -> Int(d != 1), D), cons1))
        fsp, fsb = patch_fieldset(patched, C; G=G), patch_fieldset(brick, C; G=G)
        for fs in (fsp, fsb)
            fill_by_coordinates!(f, fs)
            fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=boundary_by_coordinates(f))
        end
        result = compare_matching_blocks(fsp, fsb)
        @test result !== nothing && result[1] == 0
    end
    for C in allcenterings(Val(D))
        @test_throws "a patch corner" GhostSchedule(patch_fieldset(patched, C), PATCH_OPS)
    end
    # The low side of a vertex-like dimension with no ghosts has no region,
    # but its high side has the shared plane.
    vertex = ntuple(d -> d == 1 ? :vertex : :cell, D)
    @test_throws "a patch corner" GhostSchedule(
        patch_fieldset(patched, vertex; G=ntuple(d -> Int(d != 1), D)), PATCH_OPS)
end

@testset "Patches with swapped and reversed axes fill as the brick: $name" for (name, roots,
                                                                                σs, reversed,
                                                                                lowlow) in
                                                                               PATCH_LAYOUTS
    # The end-to-end test of the axis map: a brick cut in two, the second
    # half laid out with its axes permuted and reversed, glued back. Filled
    # with data that is a function of the brick's point alone, every stored
    # point must be the brick's at the point the layout map carries it to,
    # bit for bit, and no `NaN` may reach a ghost the fill writes. Refined
    # beside the glue, the turned prolongations and restrictions across its
    # edges must agree with the brick's to rounding (their sums run in the
    # patch's order of dimensions). A layout the axis map does not keep, and
    # a vertex-like dimension the glue reverses, are refused with their
    # reasons: the first would read with the wrong ghost widths, the second
    # a plane the block across does not own.
    D = length(roots)
    swapped = any(σ -> map(abs, σ) != ntuple(identity, D), σs)
    for refined in (false, true), C in patch_centerings(Val(D))
        refined && C != cellcentered(D) && continue
        patched, place = laid_out(roots, σs)
        refined && patch_refine_one!(patched, place)
        fsp = patch_fieldset(patched, C)
        if swapped && !all(==(C[1]), C)
            @test_throws "hold their points alike" GhostSchedule(fsp, patch_ops(fsp))
            continue
        end
        if any(b -> C[b] === :vertex && !(b == 1 && lowlow), reversed)
            @test_throws "vertex-like" GhostSchedule(fsp, patch_ops(fsp))
            continue
        end
        brick = Forest(roots; N=4, leaves=laid_out_leaves(patched, place, roots))
        fsb = patch_fieldset(brick, C)
        nnan, schedule = undefined_patch_ghosts(fsp, patch_ops(fsp), place)
        @test nnan == 0
        @test occursin("turned across a glue", sprint(show, schedule))
        fill_patch_arbitrary!(fsb, (p, x) -> x)
        fill_ghosts!(fsb, GhostSchedule(fsb, patch_ops(fsb));
                     boundary=patch_arbitrary_hook((p, x) -> x))
        result = compare_through_layout(fsp, fsb, place)
        @test result !== nothing
        result === nothing && continue
        worst, npoints = result
        @test refined ? worst < 1e-14 : worst == 0
        @test npoints > 0
        refined && @test any(g -> g.turned, Iterators.flatten(schedule.phase2))
    end
end

@testset "The annulus fills every ghost across its glue from the patch across: $C" for C in
                                                                                         PATCH_CENTERINGS_2D
    # The milestone's acceptance mesh: a ring of patches over one logical box
    # each, glued end to end, with no patch corner, so every centering runs.
    # Given different data on every patch, every ghost across a glued face
    # must hold what the donor the ring's own map names owns at that point,
    # every ghost inside a patch what its neighbor owns, and no `NaN` may
    # reach a ghost the fill writes. The glue is the identity, so its
    # transfers are ordinary copies, turned by nothing.
    forest = patch_ring(4, (2, 1))
    patch_refine_all!(forest)
    fs = patch_fieldset(forest, C)
    nnan, schedule = undefined_patch_ghosts(fs, patch_ops(fs), (p, x) -> (p, x))
    @test nnan == 0
    mismatches, seen = patch_ghost_mismatches(fs, patch_ring_across)
    @test mismatches == 0
    @test seen[:glued] > 0 && seen[:inside] > 0
    @test !occursin("turned", sprint(show, schedule))
    @test fs.rotvars == reshape(Int32[1, 2], :, 1)
end

@testset "A cubed sphere refuses its patch corners and fills an element set exactly" begin
    # Six patches over one logical box each, glued along both tangential
    # dimensions with swapped and reversed axes: a block in the corner of a
    # face has a ghost region beyond two glued faces at once, which a cell-
    # or vertex-centered set would read and the fill cannot fill, so its
    # schedule is refused with the reason. An element set is filled on faces
    # only, and must build its schedule, exchange, and hold in every face
    # ghost what the face across owns at the point the cube's own geometry
    # folds it onto, with no `NaN` left in a ghost the fill writes.
    for refine in (false, true)
        sphere = cubed_sphere()
        refine && patch_refine_all!(sphere)
        @test sphere.conn.corners
        for C in allcenterings(Val(3))
            @test_throws "a patch corner" GhostSchedule(patch_fieldset(sphere, C),
                                                        PATCH_OPS)
        end
        fs = patch_fieldset(sphere, elementcentered(3))
        nnan, schedule = undefined_patch_ghosts(fs, PATCH_ELEMENT_OPS, (p, x) -> (p, x))
        @test nnan == 0
        mismatches, seen = patch_ghost_mismatches(fs, cubed_sphere_across)
        @test mismatches == 0
        @test seen[:glued] > 0
        @test occursin("turned across a glue", sprint(show, schedule))
        @test isempty(schedule.phase2)
    end
    # So on the 2D pair with swapped and reversed axes, whose first patch is
    # glued along both dimensions.
    pair = patch_pair()
    @test_throws "a patch corner" GhostSchedule(patch_fieldset(pair, cellcentered(2)),
                                                PATCH_OPS)
    fs = patch_fieldset(pair, elementcentered(2))
    nnan, _ = undefined_patch_ghosts(fs, PATCH_ELEMENT_OPS, (p, x) -> (p, x))
    @test nnan == 0
    mismatches, seen = patch_ghost_mismatches(fs, patch_pair_across)
    @test mismatches == 0 && seen[:glued] > 0
end

@testset "A glue refuses a layout its axis map does not carry, with the reason" begin
    # A turned ghost is read with the target block's own layout through the
    # axis map, so a dimension glued onto another must hold its points as
    # that one does, and a vertex-like dimension may not be reversed, except
    # the normal of a low face glued to a low face: the error says which, and
    # why, rather than the fill reading the wrong points.
    swapped, _ = laid_out((4, 3), ((1, 2), (2, 1)))
    @test_throws "hold their points alike" GhostSchedule(
        patch_fieldset(swapped, cellcentered(2); G=(1, 2)), PATCH_OPS)
    @test_throws "hold their points alike" GhostSchedule(
        patch_fieldset(swapped, (:cell, :vertex)), PATCH_OPS)
    tangent, _ = laid_out((4, 3), ((1, 2), (1, -2)))
    @test_throws "with axis 2 of patch 1 reversed" GhostSchedule(
        patch_fieldset(tangent, vertexcentered(2)), PATCH_OPS)
    highhigh, _ = laid_out((4, 3), ((1, 2), (-1, 2)))
    @test_throws "a high face to a high face" GhostSchedule(
        patch_fieldset(highhigh, (:vertex, :cell)), PATCH_OPS)
    # What is not refused: the same with the reversed dimension cell-like,
    # and a low face glued to a low face, vertex-like along its normal.
    @test GhostSchedule(patch_fieldset(tangent, (:vertex, :cell)), PATCH_OPS) isa
          GhostSchedule
    @test GhostSchedule(patch_fieldset(highhigh, (:cell, :vertex)), PATCH_OPS) isa
          GhostSchedule
    lowlow, _ = laid_out((4, 3), ((-1, 2), (1, 2)))
    @test GhostSchedule(patch_fieldset(lowlow, vertexcentered(2)), PATCH_OPS) isa
          GhostSchedule
end

@testset "Groups across different glued faces are told apart, in one order" begin
    # Two ranks lay a message out by sorting its transfers' keys, so a key
    # must carry the glued face its source lies across — the axis map the
    # kernel reads by — and order by it, or two faces' transfers would share
    # one group, or two ranks disagree on where a transfer's slot is.
    z = (Int8(0), Int8(0))
    key(g) = TreeAMR.GroupKey{2}(:copy, (1, 0), (0, 0), 0, z, Int8(0), Int8(g),
                                 (false, false), z)
    @test key(1) != key(2)
    @test isless(key(1), key(2)) && !isless(key(2), key(1))
    @test isless(key(0), key(1))
end

# --- Conformity at the glue (step 4) ------------------------------------------

@testset "A leaf list that is coarse-fine across a glued face is refused, naming the face" begin
    # A list that puts a coarse-fine face across the glue — damaged, made by
    # hand, or written for another forest — must be refused by the checked
    # leaves path with the glue's own reason, not the seam's, and naming the
    # face, rather than trusted by a schedule that has no turned restriction.
    ring = patch_ring(4, (2, 1))
    refine!(ring, ring.leaves[2])
    @test !isbalanced(ring)
    kw = patch_keywords(ring)
    err = try
        Forest(kw.patches; N=4, glue=kw.glue, leaves=ring.leaves)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("not conforming at a glued face", err.msg)
    @test occursin("the high face of dimension 1 of patch 1, glued to the low face of " *
                   "dimension 1 of patch 2", err.msg)
    @test !occursin("not conforming at the rotating seam", err.msg)
    # The same through a swapped and reversed glue: the leaf across the high
    # face of dimension 1 of patch 1 is on the low face of dimension 2 of
    # patch 2, and balance! refines it there.
    pair = patch_pair()
    k = only(filter(k -> patch(pair, k) == 1 && patch_leafbox(pair, k)[1] == (1, 2),
                    pair.leaves))
    refine!(pair, k)
    @test !isbalanced(pair)
    kw = patch_keywords(pair)
    @test_throws "of patch 1, glued to the low face of dimension 2 of patch 2" Forest(
        kw.patches; N=4, glue=kw.glue, leaves=pair.leaves)
    balance!(pair)
    @test isbalanced(pair)
    @test Forest(kw.patches; N=4, glue=kw.glue, leaves=pair.leaves).leaves == pair.leaves
    for k in pair.leaves, δ in alldirections(Val(2))
        count(!=(0), δ) == 1 || continue
        crossing, nbrs = oriented_neighbors(pair, k, δ)
        crossing.glue == 0 && continue
        @test all(nb -> level(nb) == level(k), nbrs)
    end
end

@testset "Refined beside the glue, a forest keeps 2:1 across its edges and is accepted" begin
    # Conformity is a rule for glued faces only: an edge or corner across the
    # glue keeps the ordinary 2:1 rule, as across a seam. The meshes the
    # ghost-fill tests refine beside the glue must be balanced, accepted by
    # the checked leaves path, conforming at every glued face, and still
    # coarse-fine across an edge of the glue, or those tests would not
    # exercise the turned prolongations they claim to.
    for (forest, place) in (brick_cut((4, 2), ((2,),)), brick_cut((4, 2, 2), ((2,),)),
                            laid_out((4, 3), ((1, 2), (-2, -1))))
        patch_refine_one!(forest, place)
        @test isbalanced(forest)
        kw = patch_keywords(forest)
        @test Forest(kw.patches; N=4, glue=kw.glue, leaves=forest.leaves).leaves ==
              forest.leaves
        D = length(forest.roots)
        faces = edges = 0
        for k in forest.leaves, δ in alldirections(Val(D))
            crossing, nbrs = oriented_neighbors(forest, k, δ)
            crossing.glue == 0 && continue
            if count(!=(0), δ) == 1
                faces += all(nb -> level(nb) == level(k), nbrs) ? 0 : 1
            else
                edges += any(nb -> level(nb) != level(k), nbrs)
            end
        end
        @test faces == 0
        @test edges > 0
    end
end

@testset "The interface schedule reports a coarse-fine face across a glue as a bug" begin
    # A finer neighbor across a glued face can only mean a forest that is not
    # conforming, which no checked path produces: the interface schedule
    # says so, rather than skip the restriction, or restrict untransformed.
    broken = patch_ring(4, (2, 1))
    refine!(broken, broken.leaves[3])
    fx = FieldSet(broken, 1; G=0, centering=facecentered(2, 1))
    @test_throws "not across a glue" InterfaceSchedule(fx)
    @test_throws "this is a bug" InterfaceSchedule(fx)
    # Balanced, the same forest has coarse-fine faces inside its patches only.
    balance!(broken)
    fx = FieldSet(broken, 1; G=0, centering=facecentered(2, 1))
    @test InterfaceSchedule(fx) isa InterfaceSchedule
end

# --- The patch as a number (step 5) -------------------------------------------

@testset "Patches with identical extents fill and exchange correctly: D=$D" for D in (1, 2,
                                                                                    3)
    # Two patches on one logical box: a position alone does not say which
    # of them a point is in, so a callback handed the position only would
    # give them the same data. Handed the patch through WithPatch, each
    # patch is filled through its own map, initial data and boundary values
    # alike, and after the exchange every stored point must be the brick's
    # at the point that map carries it to — across the glue too, where the
    # ghosts come from the other patch's data. The all-variables form must
    # write bit for bit what the per-variable one does.
    roots = ntuple(d -> d == 1 ? 4 : 2, D)
    f = makepoly(D, 2)
    fp(x, p, v) = f(p == 1 ? x : Base.setindex(x, x[1] + 2, 1), v)
    all2(x, p) = ntuple(v -> fp(x, p, v), 2)
    for refined in (false, true), C in patch_centerings(Val(D))
        refined && first(C) === :element && continue
        patched, place = same_box(Val(D))
        @test patched.extents == ntuple(_ -> (0.0, 2.0), D)
        refined && patch_refine_one!(patched, place)
        brick = Forest(roots; N=4, leaves=laid_out_leaves(patched, place, roots))
        fsp, fsa = patch_fieldset(patched, C), patch_fieldset(patched, C)
        fsb = patch_fieldset(brick, C)
        fill_by_coordinates!(WithPatch(fp), fsp)
        fill_by_coordinates!(WithPatch(AllVariables(all2)), fsa)
        fill_by_coordinates!(f, fsb)
        # The two patches hold different data at the same logical position
        # (refined, a 1D pair has no two such blocks at one level).
        ps, origins = block_patches(patched), block_origins(patched)
        twins = [(a, b) for a in 1:nblocks(fsp), b in 1:nblocks(fsp)
                 if ps[a] == 1 && ps[b] == 2 && origins[a] == origins[b] &&
                    level(blockkey(fsp, a)) == level(blockkey(fsp, b))]
        @test refined || !isempty(twins)
        @test all(((a, b),) -> blockview(fsp, a, 1) != blockview(fsp, b, 1), twins)
        sched = GhostSchedule(fsp, patch_ops(fsp))
        fill_ghosts!(fsp, sched; boundary=boundary_by_coordinates(WithPatch(fp)))
        fill_ghosts!(fsa, sched;
                     boundary=boundary_by_coordinates(WithPatch(AllVariables(all2))))
        fill_ghosts!(fsb, GhostSchedule(fsb, patch_ops(fsb));
                     boundary=boundary_by_coordinates(f))
        @test fsa.work == fsp.work
        result = compare_through_layout(fsp, fsb, place)
        @test result !== nothing
        result === nothing && continue
        worst, npoints = result
        # Dyadic positions are exact, shifted or not; an element set's
        # Gauss–Lobatto nodes are not, so there the two maps round apart.
        @test worst <= (first(C) === :element ? 1e-13 : 0.0)
        @test npoints > 0
    end
end

@testset "WithPatch hands each outer ghost its block's patch, on every form" begin
    # The annulus' four patches share one logical box, and its radial faces
    # are outer: a boundary hook must be handed, at every point of every
    # region it fills, that point's position, the variable, the region's
    # direction and the patch of the block, which the position alone does
    # not determine; boundary_by_coordinates must hand the same as
    # fill_by_coordinates!; and the all-variables forms the same values.
    ring = patch_ring(4, (2, 1))
    refine!(ring, ring.leaves[2])
    balance!(ring)
    @test all(==(ring.extents), patch_keywords(ring).patches .|> (q -> q.extents))
    code(x, p, v, δ) = 1000 * p + 100 * v + 10 * δ[2] + δ[1] + x[1] / 4 + x[2] / 64
    ps = block_patches(ring)
    for C in patch_centerings(Val(2))
        fs = patch_fieldset(ring, C)
        sched = GhostSchedule(fs, patch_ops(fs))
        hooks = (CellBoundary(WithPatch((x, p, v, δ) -> oftype(x[1], code(x, p, v, δ)))),
                 CellBoundary(WithPatch(AllVariables(
                     (x, p, δ) -> ntuple(v -> oftype(x[1], code(x, p, v, δ)), 2)))),
                 boundary_by_coordinates(WithPatch(
                     (x, p, v) -> oftype(x[1], code(x, p, v, (0, 0))))),
                 boundary_by_coordinates(WithPatch(AllVariables(
                     (x, p) -> ntuple(v -> oftype(x[1], code(x, p, v, (0, 0))), 2)))))
        works = map(enumerate(hooks)) do (h, hook)
            fill!(fs.work, NaN)
            fill_by_coordinates!(WithPatch((x, p, v) -> oftype(x[1], p)), fs)
            fill_ghosts!(fs, sched; boundary=hook)
            mismatches = points = 0
            seen = Set{Int}()
            for r in sched.boundaries, idx in r.region, v in 1:fs.nvars
                δ = h <= 2 ? r.direction : (0, 0)
                x = coordinates(fs, r.block, Tuple(idx))
                points += 1
                push!(seen, ps[r.block])
                mismatches += fs.work[Tuple(idx)..., v, r.block] != code(x, ps[r.block], v, δ)
            end
            @test points > 0
            @test mismatches == 0
            @test seen == Set(1:4)
            return copy(fs.work)
        end
        @test isequal(works[1], works[2])
        @test isequal(works[3], works[4])
    end
end

@testset "On a brick WithPatch hands patch 1, and fills as the form without it" begin
    # Every forest has patches, a brick one: so a callback written for a
    # forest of patches runs on a brick, handed `p = 1`, and writes bit for
    # bit what the same callback without the patch does.
    brick = Forest((2, 2); N=4)
    refine!(brick, brick.leaves[1])
    balance!(brick)
    f = makepoly(2, 2)
    fp(x, p, v) = p == 1 ? f(x, v) : NaN
    a, b = FieldSet(brick, 2; G=1), FieldSet(brick, 2; G=1)
    fill_by_coordinates!(f, a)
    fill_by_coordinates!(WithPatch(fp), b)
    @test b.work == a.work
    fill_ghosts!(a, GhostSchedule(a, PATCH_OPS); boundary=boundary_by_coordinates(f))
    fill_ghosts!(b, GhostSchedule(b, PATCH_OPS);
                 boundary=boundary_by_coordinates(WithPatch(fp)))
    @test b.work == a.work
end

@testset "WithPatch is refused where it cannot go, with the reason" begin
    # A wrapper nested the wrong way round, or handed to the relative hook,
    # which is called by position alone, must say so on the host, before a
    # kernel calls the wrapper itself; an all-variables WithPatch callback
    # is held to one value per variable as the form without it is; and the
    # wrapper is `isbits` when its callback is, so a device can take it.
    ring = patch_ring(4, (2, 1))
    fs = patch_fieldset(ring, cellcentered(2))
    sched = GhostSchedule(fs, PATCH_OPS)
    @test_throws "it goes outside" fill_by_coordinates!(
        AllVariables(WithPatch((x, p) -> (1.0, 2.0))), fs)
    @test_throws "it goes outside" fill_ghosts!(
        fs, sched; boundary=CellBoundary(AllVariables(WithPatch((x, p, δ) -> (1.0, 2.0)))))
    @test_throws "returned 1" fill_by_coordinates!(
        WithPatch(AllVariables((x, p) -> (1.0,))), fs)
    @test_throws "returned 3" fill_ghosts!(
        fs, sched; boundary=boundary_by_coordinates(WithPatch(AllVariables(
            (x, p) -> (1.0, 2.0, 3.0)))))
    one_var = patch_fieldset(ring, cellcentered(2); nvars=1)
    @test_throws "is not handed the patch" fill_ghosts!(
        one_var, GhostSchedule(one_var, PATCH_OPS);
        boundary=CellBoundary(Relative(WithPatch((x, p) -> 0.0))))
    @test isbits(WithPatch((x, p, v) -> sum(x) + p + v))
    @test isbits(WithPatch(AllVariables((x, p) -> (sum(x), p))))
end

@testset "PatchPoints on the annulus are the brick of four times the roots, bit for bit" begin
    # A logical point in patch p of the four-patch ring is the point two
    # roots per patch along the brick of eight roots, so with the same
    # data in the same blocks the two must read the same nodes with the
    # same weights. A search that ignored the patch's first root, or a
    # geometry from another patch, would read another block. Dyadic
    # points keep every position and every shift exact in both.
    ring = patch_ring(4, (2, 1))
    brick = Forest((8, 1); N=4)
    refine!(ring, copy(ring.leaves))
    refine!(brick, copy(brick.leaves))
    fr = FieldSet(ring, 2; G=1, centering=elementcentered(2), P=1)
    fb = FieldSet(brick, 2; G=1, centering=elementcentered(2), P=1)
    fill_by_coordinates!((x, v) -> sin(x[1] + v) * cos(2x[2]), fb)
    @test nblocks(fr) == nblocks(fb)
    copyto!(fr.work, fb.work)
    rng = Xoshiro(19)
    grid(n) = rand(rng, 0:(n - 1)) / n
    pps = [PatchPoint((2grid(1024), grid(1024)), rand(rng, 1:4)) for _ in 1:300]
    filter!(q -> q.ξ[1] < 2, pps)
    xs = [(q.ξ[1] + 2(q.p - 1), q.ξ[2]) for q in pps]
    derivs = ((0, 0), (1, 0), (0, 1), (1, 1))
    a = interpolate(fr, pps, ElementPolynomial(); derivs=derivs)
    b = interpolate(fb, xs, ElementPolynomial(); derivs=derivs)
    @test isequal(a.values, b.values)
    @test [locate_point(ring, q) for q in pps] == [locate_point(brick, x) for x in xs]
    pts = locate(ring, pps)
    @test Array(pts.patch) == [q.p for q in pps]
    @test isequal(interpolate(fr, pts, ElementPolynomial(); derivs=derivs).values,
                  a.values)
    # On a brick, patch 1 is the brick.
    @test isequal(interpolate(fb, [PatchPoint(x, 1) for x in xs], ElementPolynomial();
                              derivs=derivs).values, b.values)
end

@testset "A PatchPoint on a glued face is located in the patch it names" begin
    # TreeAMR follows no glue: the high face of patch p is patch p's, its
    # last leaf there, even though the same physical point is the low face
    # of the patch across, which a half-open search would otherwise hand
    # it to.
    ring = patch_ring(4, (2, 1))
    for p in 1:4, y in (0.0, 0.3, 1.0)
        b = locate_point(ring, PatchPoint((2.0, y), p))
        k = ring.leaves[b]
        @test patch(ring, k) == p
        @test block_extent(ring, k)[1][2] == 2
    end
    @test patch(ring, ring.leaves[locate_point(ring, PatchPoint((0.0, 0.5), 3))]) == 3
end

@testset "A forest of patches refuses plain positions, Lagrange, and unknown patches" begin
    # A position names a point in every patch whose box holds it, so it
    # must be refused with the way out, not answered from patch 1; a
    # stencil reading ghosts across a glue reads another chart's points;
    # a patch that does not exist, or a point outside its patch's box,
    # must not be read from a neighbouring block.
    ring = patch_ring(4, (2, 1))
    @test_throws "Pass each point as a PatchPoint" locate_point(ring, (0.5, 0.5))
    @test_throws "Pass each point as a PatchPoint" locate(ring, [(0.5, 0.5)])
    es = FieldSet(ring, 1; G=1, centering=elementcentered(2), P=1)
    @test_throws "Pass each point as a PatchPoint" interpolate(es, [(0.5, 0.5)],
                                                               ElementPolynomial())
    cs = patch_fieldset(ring, cellcentered(2))
    @test_throws "on a forest of patches no stencil reads them" interpolate(
        cs, [PatchPoint((0.5, 0.5), 2)], Lagrange(2))
    @test_throws "names patch 5, but the forest has patches 1:4" locate(
        ring, [PatchPoint((0.5, 0.5), 1), PatchPoint((0.5, 0.5), 5)])
    @test_throws "names patch 0" locate_point(ring, PatchPoint((0.5, 0.5), 0))
    @test_throws "outside the box" interpolate(es, [PatchPoint((2.5, 0.5), 1)],
                                               ElementPolynomial())
    @test_throws "one brick, which is patch 1" locate_point(Forest((2, 2); N=4),
                                                            PatchPoint((0.5, 0.5), 2))
    m = interpolate(es, [PatchPoint((2.5, 0.5), 1), PatchPoint((0.5, 0.5), 1)],
                    ElementPolynomial(); outside=:mark)
    @test m.inside == [false, true] && isnan(m.values[1])
    # One patch is a forest of patches too: there is no brick to fall back on.
    one = Forest([(roots=(2, 2),)]; N=4)
    @test_throws "Pass each point as a PatchPoint" locate_point(one, (0.5, 0.5))
    @test locate_point(one, PatchPoint((0.5, 0.5), 1)) isa Int
end
