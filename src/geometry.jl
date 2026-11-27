# Physical geometry of the forest: mapping keys and cell indices to
# coordinates. Blocks are cubes with uniform spacing, halving per level.
#
# Every function here takes an optional leading element type and does its
# arithmetic *in* that type from the first operation, defaulting to the
# forest's own `floattype`. That is the point of the parameter: a
# `Float64` intermediate narrowed at the end would still need hardware
# fp64, which is exactly what a device may not have.

"""
    root_spacing([T], forest)

The physical width of one cell of a root-level block — the coarsest cell
size in the hierarchy. Isotropic by construction.

As everywhere in this file, the optional leading `T` is the type the
arithmetic is done in; it defaults to [`floattype`](@ref)`(forest)`.
"""
root_spacing(::Type{T}, forest::Forest) where {T} =
    (T(forest.extents[1][2]) - T(forest.extents[1][1])) / forest.roots[1] / forest.N
root_spacing(forest::Forest{D,T}) where {D,T} = root_spacing(T, forest)

"""
    spacing([T], forest, level::Integer)
    spacing([T], forest, k::MortonKey)

The physical cell size at the given refinement level, halving per level.
"""
spacing(::Type{T}, forest::Forest, lvl::Integer) where {T} =
    root_spacing(T, forest) / (1 << Int(lvl))
spacing(forest::Forest{D,T}, lvl::Integer) where {D,T} = spacing(T, forest, lvl)
spacing(::Type{T}, forest::Forest{D}, k::MortonKey{D}) where {T,D} =
    spacing(T, forest, level(k))
spacing(forest::Forest{D,T}, k::MortonKey{D}) where {D,T} = spacing(T, forest, k)

"""
    minimum_spacing([T], forest)

The cell size of the finest leaf present. This is what sets the global
timestep: with no subcycling, the whole hierarchy advances at the CFL
limit of its finest cells.
"""
minimum_spacing(::Type{T}, forest::Forest) where {T} =
    spacing(T, forest, maxlevel(forest))
minimum_spacing(forest::Forest{D,T}) where {D,T} = minimum_spacing(T, forest)

"""
    block_spacings(forest, T=floattype(forest))

The cell size of every block this rank stores, indexed by local block —
what a kernel needs to scale a finite-difference stencil, since blocks
at different levels have different spacings. Serially that is every
leaf; over a distributed forest it is the leaves of
[`blockrange`](@ref), so that entry `b` belongs to block `b` of every
[`FieldSet`](@ref) over `forest`.
"""
block_spacings(forest::Forest{D,R}, ::Type{T}=R) where {D,R,T} =
    T[spacing(T, forest, k) for k in localleaves(forest)]

# The leaves whose blocks this rank stores, in local block order.
localleaves(forest::Forest) = view(forest.leaves, blockrange(forest))

# Where positions are anchored, patch by patch and dimension by
# dimension, as `(m, kind)`: `m` the number of level-0 spacings from the
# patch's low face to the anchor, and `kind` which anchor it is. The rule,
# in order (M22 steps 10b and 10d; CODE.md, "Positions" under
# "Centerings"):
#
# 1. `ANCHOR_ZERO`: coordinate zero, where zero is one of the patch's
#    level-0 nodes along `d`, `lo + K·(hi − lo)/n` for an integer `K` in
#    `0…n`, `n` the roots times `N`; then `m = K`.
# 2. `ANCHOR_LOW` or `ANCHOR_HIGH`: a symmetry face of `d`, a reflecting
#    face or a face of the rotating seam (a quarter turn's low faces of
#    `d1` and `d2`, a half turn's low face of `d1`); `m = 0` or `n`. Both
#    faces symmetric: the low one. An outer or outflow face is none.
# 3. `ANCHOR_CENTRE`: the centre, `m = n/2`; `n` is even, since `N` is.
#
# The test for zero is exact, in `Rational{BigInt}` of the extents in the
# forest's own type. It is made once per forest, on the host, and kept in
# its `ForestState`: it depends on the extents, the roots, `N`, the face
# kinds and the seam, which a forest never changes, the same on every
# rank. A zero that is a cell centre and not a node does not anchor.
const ANCHOR_ZERO, ANCHOR_LOW, ANCHOR_HIGH, ANCHOR_CENTRE = Int8(0), Int8(1), Int8(2), Int8(3)

# Whether face `side` (1 low, 2 high) of dimension `d` is a symmetry face:
# reflecting, or a face of the rotating seam.
function symmetry_face(forest::Forest, d::Integer, side::Integer)
    facekind(forest, d, side) == ReflectingFace && return true
    hasrotating(forest) && side == 1 || return false
    d1, d2 = seam_plane(forest)
    return d == d1 || (!ishalfturn(forest) && d == d2)
end

function anchor_table(forest::Forest{D}) where {D}
    cached = forest.state.anchors
    cached === nothing || return cached::Vector{NTuple{D,Tuple{Int,Int8}}}
    anchors = map(1:npatches(forest)) do p
        extents, roots = patch_extents(forest, p), patch_roots(forest, p)
        return ntuple(D) do d
            lo = Rational{BigInt}(extents[d][1])
            hi = Rational{BigInt}(extents[d][2])
            n = roots[d] * forest.N
            K = -lo * n / (hi - lo)
            denominator(K) == 1 && 0 <= K <= n && return (Int(K), ANCHOR_ZERO)
            symmetry_face(forest, d, 1) && return (0, ANCHOR_LOW)
            symmetry_face(forest, d, 2) && return (n, ANCHOR_HIGH)
            @assert iseven(n)
            return (n ÷ 2, ANCHOR_CENTRE)
        end
    end
    forest.state.anchors = anchors
    return anchors
end

# The anchors' integer coordinates `m`, patch by patch.
position_anchors(forest::Forest{D}) where {D} =
    [map(first, a) for a in anchor_table(forest)]

# The anchor of patch `p` along each dimension, in `T`, and its integer
# coordinate `m` in level-0 spacings. Its value: zero; the face's extent
# itself, `T(lo)` or `T(hi)`; or the centre, `(T(lo) + T(hi)) / 2`, the
# extents converted, summed with one rounding and halved exactly, which
# is zero for extents `(−L, L)` (CODE.md, "Positions" under "Centerings").
function patch_anchor(::Type{T}, forest::Forest{D}, p::Integer) where {T,D}
    table = anchor_table(forest)[p]
    extents = patch_extents(forest, p)
    a = ntuple(D) do d
        lo, hi = T(extents[d][1]), T(extents[d][2])
        kind = table[d][2]
        kind == ANCHOR_ZERO && return zero(T)
        kind == ANCHOR_LOW && return lo
        kind == ANCHOR_HIGH && return hi
        return (lo + hi) / 2
    end
    return a, map(first, table)
end

# The integer corner of block `k` along each dimension: the number of
# spacings of its level from its patch's anchor to its low face — from
# the low face of the patch, `g·N` for `g` its integer coordinate at its
# level within the patch, less the anchor's `m` at that level, so that it
# is negative below a zero anchor. An `Int`, since `MAX_LEVEL` levels
# below a few roots overflow 32 bits. Positions are formed from it by
# `nodeposition` (CODE.md, "Positions" under "Centerings").
function block_base(forest::Forest{D}, k::MortonKey{D}) where {D}
    rootpos = root_position(forest, k.root)
    lvl = level(k)
    m = position_anchors(forest)[patch(forest, k)]
    return ntuple(d -> ((rootpos[d] << lvl) + Int(k.coords[d])) * forest.N - (m[d] << lvl),
                  D)
end

# The anchor of block `k`'s patch, in `T` (`patch_anchor`): zero where
# zero is on the patch's grid, else its symmetry face, else its centre.
block_low(::Type{T}, forest::Forest{D}, k::MortonKey{D}) where {T,D} =
    patch_anchor(T, forest, patch(forest, k))[1]

# The box points of patch `p` are located in, in `T`: from the smaller of
# the low corner it was given and its low face as positions form it to
# the larger of the high corner and its high face, the faces formed from
# the patch's anchor (`patch_anchor`), `a + (K − m)·h` for `K` its
# spacings across at level 0 (the same bits at every level). The two differ by rounding: one cell
# size serves every dimension and every patch, and the extents need only
# agree with it to `check_brick`'s tolerance. The larger one, so that
# every point `coordinates` returns is located, and every point of the
# box the forest was given still is (CODE.md, "Positions" under
# "Centerings").
function location_box(::Type{T}, forest::Forest{D}, p::Integer) where {T,D}
    extents, roots = patch_extents(forest, p), patch_roots(forest, p)
    a, m = patch_anchor(T, forest, p)
    h = spacing(T, forest, 0)
    return ntuple(D) do d
        lo, hi = T(extents[d][1]), T(extents[d][2])
        bottom = nodeposition(a[d], h, -m[d], 1, 1, nothing)
        top = nodeposition(a[d], h, -m[d], roots[d] * forest.N + 1, 1, nothing)
        return (min(lo, bottom), max(hi, top))
    end
end

"""
    block_origin([T], forest, k::MortonKey)

The physical position of the lower corner of block `k`'s interior (the
outer corner of its first interior cell, not that cell's center). On a
forest of patches (M17) it is in the logical coordinates of `k`'s patch.

It is a node's position, formed by [`nodeposition`](@ref
TreeAMR.nodeposition) as `a + B·h` from the patch's anchor `a` (zero
where zero is on the patch's grid, else a symmetry face, else the
centre; [`block_lows`](@ref)), the block's integer corner `B` counted
from it and its level's spacing `h`, so that it
is bit for bit the corner the blocks around it see and the first vertex
`coordinates` gives.
"""
function block_origin(::Type{T}, forest::Forest{D}, k::MortonKey{D}) where {T,D}
    lo, B, h = block_low(T, forest, k), block_base(forest, k), spacing(T, forest, k)
    return ntuple(d -> nodeposition(lo[d], h, B[d], 1, 1, nothing), D)
end
block_origin(forest::Forest{D,T}, k::MortonKey{D}) where {D,T} = block_origin(T, forest, k)

"""
    block_origins(forest, T=floattype(forest))

The lower corner of the interior of every block this rank stores,
indexed by local block — the companion of [`block_spacings`](@ref), for
what in a kernel is not a point's position: a point is placed through
[`nodeposition`](@ref TreeAMR.nodeposition) from [`block_lows`](@ref)
and [`block_bases`](@ref) instead, which puts every copy of a point that
several blocks hold at one position bit for bit. The corners themselves
are formed by the same rule, so `origins[b]` is the position of block
`b`'s first vertex.

Together these arrays are the whole geometry a device-side kernel
sees: plain `isbits` arrays indexed by block, no keys and no forest.
"""
block_origins(forest::Forest{D,R}, ::Type{T}=R) where {D,R,T} =
    NTuple{D,T}[block_origin(T, forest, k) for k in localleaves(forest)]

"""
    block_lows(forest, T=floattype(forest))

The anchor of the patch (M17) of every block this rank stores, as an
`NTuple{D,T}` indexed by local block, dimension by dimension: zero where
zero is a node of the patch's level-0 grid; else the extent of a
symmetry face (a reflecting face or a face of the rotating seam, the
lower of two); else the centre, `(T(lo) + T(hi)) / 2`. The name stays
from when it was the low corner. With [`block_bases`](@ref) and
[`block_spacings`](@ref) it is what [`nodeposition`](@ref
TreeAMR.nodeposition) forms a point's position from, `a + k·h` with `k`
an integer counted from the anchor, and why every block holding a point puts it at the same
position bit for bit. A number per block rather than per patch, so that
a kernel reads it as it reads the spacing.
"""
block_lows(forest::Forest{D,R}, ::Type{T}=R) where {D,R,T} =
    NTuple{D,T}[block_low(T, forest, k) for k in localleaves(forest)]

"""
    block_bases(forest)

The integer corner of every block this rank stores, as an
`NTuple{D,Int}` indexed by local block: along each dimension the number
of spacings of the block's level from its patch's low face to the
block's low face. Owned index `I` of block `b` has the integer
coordinate `bases[b][d] + I − 1` there, which every block holding the
point agrees on and which [`nodeposition`](@ref TreeAMR.nodeposition)
turns into its position. `Int` on every backend, since a deep forest's
coordinates overflow 32 bits.
"""
block_bases(forest::Forest{D}) where {D} =
    NTuple{D,Int}[block_base(forest, k) for k in localleaves(forest)]

"""
    block_patches(forest)

The patch (M17; see [`patch`](@ref)) of every block this rank stores, as
`Int32`, indexed by local block — the companion of
[`block_origins`](@ref) and [`block_spacings`](@ref) for a forest of
patches, whose extents may overlap or coincide: a block's origin then
does not say which patch it is in, and so which of the application's
maps it is under, and this does. All ones on a forest of one brick.
"""
block_patches(forest::Forest) = Int32[patch(forest, k) for k in localleaves(forest)]

"""
    block_extent([T], forest, k::MortonKey)

The `(lo, hi)` physical extent of block `k`'s interior, per dimension.
Both ends are node positions ([`nodeposition`](@ref
TreeAMR.nodeposition)), so the leaves' extents tile the domain bit for
bit: a block's `hi` is its neighbour's `lo`, at one level or across two.
The domain's high face is then at `lo + K·h` for `K` the number of
spacings across it, which is the extent the forest was given up to
rounding (CODE.md, "Positions" under "Centerings").
"""
function block_extent(::Type{T}, forest::Forest{D}, k::MortonKey{D}) where {T,D}
    lo, B, h = block_low(T, forest, k), block_base(forest, k), spacing(T, forest, k)
    N = forest.N
    return ntuple(d -> (nodeposition(lo[d], h, B[d], 1, 1, nothing),
                        nodeposition(lo[d], h, B[d], N + 1, 1, nothing)), D)
end
block_extent(forest::Forest{D,T}, k::MortonKey{D}) where {D,T} = block_extent(T, forest, k)
