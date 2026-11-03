# M17: a forest of patches — several bricks of roots glued at their faces by
# signed axis permutations. This file holds the forest (step 1) and the
# neighbor search across the glue (step 2).
#
# Every claim about where a step lands is stated against a continuous map
# written out independently of the glue table: by hand for the small
# forests, and for the cubed sphere by folding a point that leaves one face
# of the cube over the edge onto the next. The tree's answer is a node at
# a level; the oracle's is the region of logical space beyond a block,
# carried across the face and intersected with the leaves of the patch it
# lands in.

using TreeAMR: Crossing, OneBrick, Patches, ForestDigest, patch_keywords,
               patch_roots, glue_slot, glue_axes, neighbor_anchor, oriented_neighbors,
               wall_direction, virtual_offset, childoffset

# Both directions of the gluing of face `from` to face `to` by the signed
# permutation `σ`: the entry as given, and its inverse.
function patch_glue_both(from, to, σ)
    D = length(σ)
    τ = ntuple(D) do a
        e = findfirst(e -> abs(σ[e]) == a, 1:D)
        return sign(σ[e]) * e
    end
    return [(from=from, to=to, map=σ), (from=to, to=from, map=τ)]
end

# A ring of `n` patches of `roots`, each glued at the high face of
# dimension 1 to the low face of the next with the identity: an annulus,
# angle along dimension 1 and radius along 2, which has no patch corner.
patch_ring(n, roots; kw...) =
    Forest([(roots=roots,) for _ in 1:n]; N=4,
           glue=reduce(vcat, [patch_glue_both((p, 1, 2), (mod1(p + 1, n), 1, 1),
                                              ntuple(identity, length(roots)))
                              for p in 1:n]), kw...)

# Two 2D patches glued at two faces, with their axes swapped and one of them
# reversed: the high face of dimension 1 of patch 1 onto the low face of
# dimension 2 of patch 2, by `(2, -1)`, and the low face of dimension 2 of
# patch 1 onto the high face of dimension 1 of patch 2, by `(2, 1)`.
# Patch 1 then has glued faces along both dimensions, so a patch corner.
const PATCH_PAIR_GLUE = [patch_glue_both((1, 1, 2), (2, 2, 1), (2, -1));
                         patch_glue_both((1, 2, 1), (2, 1, 2), (2, 1))]
patch_pair(; kw...) =
    Forest([(roots=(2, 3),), (roots=(3, 2), extents=((0, 3), (5, 7)))]; N=4,
           glue=PATCH_PAIR_GLUE, kw...)

# The pair's maps, by hand, in root units: a point `x` of patch `p` beyond
# its face `(d, s)` as `(q, y)`, the point of patch `q` it is; `nothing`
# for a face that is not glued.
function patch_pair_across(p, x, d, s)
    p == 1 && (d, s) == (1, 2) && return 2, (3 - x[2], x[1] - 2)
    p == 1 && (d, s) == (2, 1) && return 2, (x[2] + 3, x[1])
    p == 2 && (d, s) == (2, 1) && return 1, (x[2] + 2, 3 - x[1])
    p == 2 && (d, s) == (1, 2) && return 1, (x[2], x[1] - 3)
    return nothing
end

# Three 1D patches of two roots in a loop, one gluing reversed at either
# end of patch 3, so that a step can turn round.
const PATCH_LOOP_GLUE = [patch_glue_both((1, 1, 2), (2, 1, 1), (1,));
                         patch_glue_both((2, 1, 2), (3, 1, 2), (-1,));
                         patch_glue_both((3, 1, 1), (1, 1, 1), (-1,))]
patch_loop() = Forest([(roots=(2,),) for _ in 1:3]; N=4, glue=PATCH_LOOP_GLUE)
function patch_loop_across(p, x, d, s)
    p == 1 && return s == 2 ? (2, (x[1] - 2,)) : (3, (-x[1],))
    p == 2 && return s == 1 ? (1, (x[1] + 2,)) : (3, (4 - x[1],))
    return s == 2 ? (2, (4 - x[1],)) : (1, (-x[1],))
end

# The six faces of the cube `[-1, 1]³`, each a patch `(ξ, η, r)` over
# `[-1, 1]² × [1, 2]`, with the frame `(n, u, v)` its point
# `n + ξ u + η v` is placed by. The frames are chosen so that the glue has
# swapped axes, reversed ones, and both at once.
const CUBE_FRAMES = (([1, 0, 0], [0, 1, 0], [0, 0, 1]),
                     ([-1, 0, 0], [0, 0, 1], [0, 1, 0]),
                     ([0, 1, 0], [0, 0, 1], [1, 0, 0]),
                     ([0, -1, 0], [-1, 0, 0], [0, 0, 1]),
                     ([0, 0, 1], [1, 0, 0], [0, 1, 0]),
                     ([0, 0, -1], [0, 1, 0], [-1, 0, 0]))

# The glue of the cubed sphere, derived from the frames: face `(d, s)` of
# patch `p` is the edge at `±a_d`, shared with the patch whose `n` that
# is, and there it is the face at `n_p`. A step out of the one face is a
# step into the other; the tangent along the edge is parallel in both
# frames; the radius is the radius. Every face is a `from` once, so both
# directions come out of the loop.
function cubed_sphere_glue()
    glue = []
    for (p, (n, u, v)) in enumerate(CUBE_FRAMES), d in 1:2, s in 1:2
        a = (u, v)
        q = findfirst(f -> f[1] == (s == 2 ? 1 : -1) .* a[d], CUBE_FRAMES)
        aq = CUBE_FRAMES[q][2:3]
        d2, s2 = only((d2, s2) for d2 in 1:2 for s2 in 1:2
                      if (s2 == 2 ? 1 : -1) .* aq[d2] == n)
        t, t2 = 3 - d, 3 - d2
        @assert a[t] == aq[t2] || a[t] == -aq[t2]
        σ = Base.setindex(Base.setindex((0, 0, 3),
                                        (s2 == 1 ? 1 : -1) * (s == 2 ? 1 : -1) * d2, d),
                          (a[t] == aq[t2] ? 1 : -1) * t2, t)
        push!(glue, (from=(p, d, s), to=(q, d2, s2), map=σ))
    end
    return glue
end
cubed_sphere(; kw...) =
    Forest([(roots=(2, 2, 1), extents=((-1, 1), (-1, 1), (1, 2))) for _ in 1:6]; N=4,
           glue=cubed_sphere_glue(), kw...)

# A point of patch `p` beyond its face `(d, s)` folded over the cube's edge
# onto the next face, in root units (`ξ = x[1] - 1`, `η = x[2] - 1`,
# `r = x[3] + 1`). Beyond the edge by `ε` along `a_d` is the point on the
# edge moved by `ε` along `-n`, which lies on the face across; there its
# coordinates are its projections on that face's `u` and `v`. The radial
# faces are not glued.
function cubed_sphere_across(p, x, d, s)
    d == 3 && return nothing
    n, u, v = CUBE_FRAMES[p]
    a = (u, v)
    c = (x[1] - 1, x[2] - 1)
    sgn = s == 2 ? 1 : -1
    ε = sgn * c[d] - 1
    point = n .+ sgn .* a[d] .+ c[3 - d] .* a[3 - d] .- ε .* n
    q = findfirst(f -> f[1] == sgn .* a[d], CUBE_FRAMES)
    _, uq, vq = CUBE_FRAMES[q]
    return q, (sum(point .* uq) + 1, sum(point .* vq) + 1, x[3])
end

# A leaf's box in its patch, in root units.
function patch_leafbox(forest::Forest{D}, k::MortonKey{D}) where {D}
    w = 1 // 2^level(k)
    pos = root_position(forest, k.root)
    lo = ntuple(d -> pos[d] + Int(k.coords[d]) * w, D)
    return lo, lo .+ w
end

"""
The leaves the oracle says abut leaf `k` of `forest` in direction `δ`, and
how the step goes: `:inside` its patch, out through the one face `(d, s)`,
or out through several at once, `:corner` if every one of them is glued
and `:outer` otherwise. The leaves are those of the patch the region lands
in whose boxes overlap the thin slab beyond `k` in direction `δ`, carried
across the face by `across`; the slab is thinner than the finest leaf, so
the leaves it overlaps are the ones touching `k`.
"""
function patch_expected_neighbors(forest::Forest{D}, k::MortonKey{D}, δ, across) where {D}
    p = patch(forest, k)
    lo, hi = patch_leafbox(forest, k)
    ε = 1 // 2^(maxlevel(forest) + 2)
    slo = ntuple(d -> δ[d] > 0 ? hi[d] : δ[d] < 0 ? lo[d] - ε : lo[d], D)
    shi = ntuple(d -> δ[d] > 0 ? hi[d] + ε : δ[d] < 0 ? lo[d] : hi[d], D)
    mid = (slo .+ shi) ./ 2
    roots = patch_roots(forest, p)
    exits = [(d, mid[d] < 0 ? 1 : 2) for d in 1:D if !(0 < mid[d] < roots[d])]
    if length(exits) > 1
        glued = all(e -> across(p, mid, e...) !== nothing, exits)
        return MortonKey{D}[], glued ? :corner : :outer
    end
    if isempty(exits)
        q, box, how = p, (slo, shi), :inside
    else
        how = only(exits)
        a, b = across(p, slo, how...), across(p, shi, how...)
        a === nothing && return MortonKey{D}[], :outer
        q, box = a[1], (min.(a[2], b[2]), max.(a[2], b[2]))
    end
    found = filter(forest.leaves) do kk
        patch(forest, kk) == q || return false
        klo, khi = patch_leafbox(forest, kk)
        return all(d -> max(klo[d], box[1][d]) < min(khi[d], box[2][d]), 1:D)
    end
    return found, how
end

"""
Compare the neighbor search of `forest` with the oracle for every leaf and
every direction: the keys, the crossing — the glue index of the face the
step leaves through, or none — and, for a finer neighbor across a glue,
its child offset turned into the asking block's frame by
`virtual_offset`, against the offset of its image there. Returns
`(mismatches, seen)`, `seen` counting the cases met: `:inside`, `:glued`
(a neighbor across a glue at the same level), `:finer` (finer ones), `:outer` and `:corner` (out
through several glued faces at once).
"""
function patch_neighbor_mismatches(forest::Forest{D}, across) where {D}
    mismatches = 0
    seen = Dict{Symbol,Int}()
    count!(c) = (seen[c] = get(seen, c, 0) + 1)
    for k in forest.leaves, δ in alldirections(Val(D))
        expected, how = patch_expected_neighbors(forest, k, δ, across)
        crossing, keys = oriented_neighbors(forest, k, δ)
        ok = Set(keys) == Set(expected) && neighbor_keys(forest, k, δ) == keys
        if how isa Symbol
            count!(how)
            ok &= crossing == Crossing()
        elseif isempty(keys)
            ok = false
        else
            exit = how
            p = patch(forest, k)
            ok &= crossing == Crossing(0, glue_slot(p, exit..., D))
            count!(level(first(keys)) > level(k) ? :finer : :glued)
            # The offset of a finer neighbor's image, within its parent's
            # image, in the asking block's frame.
            q = patch(forest, first(keys))
            back = forest.conn.glue[glue_slot(p, exit..., D)].face
            for nbr in keys
                level(nbr) > level(k) || continue
                nlo, nhi = patch_leafbox(forest, nbr)
                a, b = across(q, nlo, back...), across(q, nhi, back...)
                image = min.(a[2], b[2])
                w = 1 // 2^level(nbr)
                ok &= virtual_offset(childoffset(nbr), glue_axes(forest, crossing)...) ==
                      ntuple(d -> mod(Int(image[d] / w), 2), D)
            end
        end
        mismatches += !ok
    end
    return mismatches, seen
end

# Refine every leaf once.
patch_refine_all!(forest) = refine!(forest, copy(forest.leaves))

npatches_of(forest) = TreeAMR.npatches(forest)

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
     ("2D ring", () -> patch_ring(4, (2, 1)), (p, x, d, s) ->
         d == 1 ? (mod1(p + (s == 2 ? 1 : -1), 4), (x[1] + (s == 2 ? -2 : 2), x[2])) :
         nothing),
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
