# The definitions of `patch_tests.jl`, split off so that each test file runs
# on its own: every test process includes all of `helpers/` ("Parallel
# groups" in CODE.md). `exchange_tests.jl` uses them too.

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

# The ring of four patches of `(2, 1)` roots, by hand: a point beyond the
# high face of dimension 1 is in the next patch, two roots back.
patch_ring_across(p, x, d, s) =
    d == 1 ? (mod1(p + (s == 2 ? 1 : -1), 4), (x[1] + (s == 2 ? -2 : 2), x[2])) : nothing

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

const PATCH_OPS = Operators(prolongation=2, restriction=2)
const PATCH_ELEMENT_OPS = Operators(family=ElementLocal)

# A field set of `centering` over `forest` with one ghost layer, of order
# `P = 1` if it is an element set, and the operators that fill it.
patch_fieldset(forest, centering; nvars=2, G=1) =
    FieldSet(forest, nvars; G=G, centering=centering,
             P=first(centering) === :element ? 1 : nothing)
patch_ops(fs) = fs.P > 0 ? PATCH_ELEMENT_OPS : PATCH_OPS

# Every centering of one kind per dimension, and the element one.
patch_centerings(::Val{D}) where {D} = [vec(allcenterings(Val(D))); [elementcentered(D)]]

# A brick of `roots` cut at the root planes `cuts[d]` along each dimension
# `d` into patches glued face to face with the identity, each over the
# brick's own extents there, so that its blocks lie where the brick's do,
# and `place` the exact map from a patch's root units to the brick's. Cut
# along two dimensions, every patch has a patch corner.
function brick_cut(roots::NTuple{D,Int}, cuts; N=4) where {D}
    bounds = ntuple(d -> [0; collect(d <= length(cuts) ? cuts[d] : ()); roots[d]], D)
    shape = map(b -> length(b) - 1, bounds)
    cells = vec(CartesianIndices(shape))
    number(I) = LinearIndices(shape)[I]
    patches = [(roots=ntuple(d -> bounds[d][I[d] + 1] - bounds[d][I[d]], D),
                extents=ntuple(d -> (Float64(bounds[d][I[d]]),
                                     Float64(bounds[d][I[d] + 1])), D))
               for I in cells]
    glue = []
    for I in cells, d in 1:D
        I[d] < shape[d] || continue
        J = I + CartesianIndex(ntuple(e -> e == d ? 1 : 0, D))
        append!(glue, patch_glue_both((number(I), d, 2), (number(J), d, 1),
                                      ntuple(identity, D)))
    end
    lows = [ntuple(d -> bounds[d][I[d]], D) for I in cells]
    place(p, x) = x .+ lows[p]
    return Forest(patches; N=N, glue=glue), place
end

# A brick of `roots` cut at `cut` along dimension 1 into two patches whose
# axes are laid out by the signed permutations `σs`: axis `a` of patch `p`
# is the brick's axis `|σs[p][a]|`, reversed where it is negative. The glue
# of the cut follows from the two layouts; `place` is the exact affine map
# from a patch's root units to the brick's, written out from the layouts
# alone.
function laid_out(roots::NTuple{D,Int}, σs; cut=2, N=4) where {D}
    zero = ntuple(_ -> 0, D)
    boxes = [(zero, Base.setindex(roots, cut, 1)), (Base.setindex(zero, cut, 1), roots)]
    brickaxis(σ, b) = findfirst(a -> abs(σ[a]) == b, 1:D)::Int
    patchroots = [ntuple(a -> hi[abs(σ[a])] - lo[abs(σ[a])], D)
                  for ((lo, hi), σ) in zip(boxes, σs)]
    function place(p, x)
        local (lo, hi), τ = boxes[p], σs[p]
        return ntuple(D) do b
            a = brickaxis(τ, b)
            return τ[a] > 0 ? lo[b] + x[a] : hi[b] - x[a]
        end
    end
    # The cut is a high face of patch 1 where its axis runs along the brick's
    # and a low face of patch 2 where its does; a step along axis `e` of
    # patch 1 is a step along its brick axis, and so along that axis' image
    # in patch 2.
    a1, a2 = brickaxis(σs[1], 1), brickaxis(σs[2], 1)
    from = (1, a1, σs[1][a1] > 0 ? 2 : 1)
    to = (2, a2, σs[2][a2] > 0 ? 1 : 2)
    gluemap = ntuple(D) do e
        a = brickaxis(σs[2], abs(σs[1][e]))
        return sign(σs[1][e]) * sign(σs[2][a]) * a
    end
    forest = Forest([(roots=r,) for r in patchroots]; N=N,
                    glue=patch_glue_both(from, to, gluemap))
    return forest, place
end

# The brick's leaves that are the images of a forest of patches' under
# `place`, in curve order.
function laid_out_leaves(forest, place, roots)
    keys = map(forest.leaves) do k
        lo, hi = patch_leafbox(forest, k)
        p = patch(forest, k)
        a, b = place(p, lo), place(p, hi)
        return box_key(roots, (min.(a, b), max.(a, b)))
    end
    return sort!(keys)
end

# Refine the leaf of patch 2 at the cut, `x = 2` in the brick, lowest in the
# brick's other dimensions, and balance: the leaf across the glue is refined
# with it, and the glue has coarse-fine edges beside them.
function patch_refine_one!(forest, place)
    images = map(forest.leaves) do k
        lo, hi = patch_leafbox(forest, k)
        p = patch(forest, k)
        return min.(place(p, lo), place(p, hi))
    end
    atcut = filter(i -> patch(forest, forest.leaves[i]) == 2 && images[i][1] == 2,
                   eachindex(images))
    refine!(forest, forest.leaves[argmin(i -> reverse(images[i]), atcut)])
    balance!(forest)
    return forest
end

# The layouts of the test below: a name, the brick, the two patches' axes, the
# brick axes the glue reverses, and whether the reversal of the normal is a
# low face glued to a low face, which a vertex-like normal allows.
const PATCH_LAYOUTS = [
    ("2D swapped", (4, 3), ((1, 2), (2, 1)), (), false),
    ("2D low face to low face", (4, 3), ((-1, 2), (1, 2)), (1,), true),
    ("2D the tangent reversed", (4, 3), ((1, 2), (1, -2)), (2,), false),
    ("2D swapped, the tangent reversed", (4, 3), ((1, 2), (-2, 1)), (2,), false),
    ("2D high face to high face", (4, 3), ((1, 2), (-1, 2)), (1,), false),
    ("2D swapped, both reversed", (4, 3), ((1, 2), (-2, -1)), (1, 2), false),
    ("3D cyclic", (4, 2, 3), ((1, 2, 3), (2, 3, 1)), (), false),
    ("3D cyclic, two reversed", (4, 2, 3), ((1, 2, 3), (-2, 3, -1)), (1, 2), false)]

const PATCH_CENTERINGS_2D = patch_centerings(Val(2))
# A brick of `(4, 2, …)` roots cut in two at `x = 2`, both halves on the same
# logical box `[0, 2] × [0, 2] × …`, glued with the identity, and `place` the
# map from a patch's root units to the brick's: the second patch is the
# brick's upper half, two roots on.
function same_box(::Val{D}) where {D}
    half = ntuple(d -> 2, D)
    forest = Forest([(roots=half,), (roots=half,)]; N=4,
                    glue=patch_glue_both((1, 1, 2), (2, 1, 1), ntuple(identity, D)))
    place(p, x) = p == 1 ? x : Base.setindex(x, x[1] + 2, 1)
    return forest, place
end
