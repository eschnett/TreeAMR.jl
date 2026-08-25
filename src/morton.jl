"""
    MAX_LEVEL

The deepest refinement level a [`MortonKey`](@ref) can represent. Bounded
by the `UInt32` coordinate storage: a coordinate at level `L` runs over
`0:2^L-1`, so `L` cannot exceed 32.
"""
const MAX_LEVEL = 32

"""
    MortonKey{D}

A key identifying one node of the forest: the root octree it belongs to,
its refinement level, and its per-dimension integer coordinates at that
level (each in `0:2^level-1`).

Keys are totally ordered by [`isless`](@ref): root index first, then
Morton (Z-order) curve order, with an ancestor sorting before all of its
descendants. The bit interleaving is computed on the fly from `level`
and `coords` — there is no packed-integer representation, so the depth
limit is [`MAX_LEVEL`](@ref) rather than `64 ÷ D`.

Dimension 1 is the *most* significant dimension of the interleaving.
"""
struct MortonKey{D}
    root::Int32
    level::Int8
    coords::NTuple{D,UInt32}

    function MortonKey{D}(root::Integer, level::Integer, coords::NTuple{D,<:Integer}) where {D}
        root >= 0 || throw(ArgumentError("root index must be nonnegative, got $root"))
        0 <= level <= MAX_LEVEL ||
            throw(ArgumentError("level must be in 0:$MAX_LEVEL, got $level"))
        maxcoord = (UInt64(1) << level) - 1
        all(c -> 0 <= c <= maxcoord, coords) ||
            throw(ArgumentError("coordinates $coords out of range 0:$maxcoord for level $level"))
        return new{D}(Int32(root), Int8(level), map(UInt32, coords))
    end
end

MortonKey(root::Integer, level::Integer, coords::NTuple{D,<:Integer}) where {D} =
    MortonKey{D}(root, level, coords)

Base.:(==)(::MortonKey, ::MortonKey) = false
Base.:(==)(a::MortonKey{D}, b::MortonKey{D}) where {D} =
    a.root == b.root && a.level == b.level && a.coords == b.coords
Base.hash(k::MortonKey, h::UInt) = hash(k.coords, hash(k.level, hash(k.root, h)))

Base.show(io::IO, k::MortonKey{D}) where {D} =
    print(io, "MortonKey{", D, "}(root=", Int(k.root), ", level=", Int(k.level),
          ", coords=", map(Int, k.coords), ")")

"""
    level(k::MortonKey)

The refinement level of `k`, as an `Int`.
"""
level(k::MortonKey) = Int(k.level)

"""
    parentkey(k::MortonKey)

The key of the node that `k` is one of the `2^D` children of. Errors if
`k` is at root level.
"""
function parentkey(k::MortonKey{D}) where {D}
    k.level > 0 || throw(ArgumentError("a root-level key ($k) has no parent"))
    return MortonKey{D}(k.root, k.level - 1, map(c -> c >> 1, k.coords))
end

"""
    childkeys(k::MortonKey)

The `2^D` keys `k` splits into under refinement. The order within the
tuple is unspecified; use [`sortedchildkeys`](@ref) when curve order
matters.
"""
function childkeys(k::MortonKey{D}) where {D}
    k.level < MAX_LEVEL || throw(ArgumentError("a key at MAX_LEVEL ($k) cannot be refined"))
    return ntuple(2^D) do i
        offset = ntuple(d -> ((i - 1) >> (d - 1)) & 1, D)
        MortonKey{D}(k.root, k.level + 1, map((c, o) -> 2c + o, k.coords, offset))
    end
end

"""
    sortedchildkeys(k::MortonKey)

The `2^D` children of `k` in ascending curve order. Because a node's
descendants occupy a contiguous run of the Morton curve, splicing these
in place of `k` keeps a sorted leaf array sorted.
"""
sortedchildkeys(k::MortonKey{D}) where {D} = sort!(collect(childkeys(k))) # :: Vector{MortonKey{D}}

"""
    isancestor(a::MortonKey, b::MortonKey)

Whether `a` is a strict ancestor of `b` — that is, `b` lies inside `a`
and is strictly finer.
"""
function isancestor(a::MortonKey{D}, b::MortonKey{D}) where {D}
    a.root == b.root || return false
    a.level < b.level || return false
    shift = b.level - a.level
    return all(d -> (b.coords[d] >> shift) == a.coords[d], 1:D)
end

# The coordinate of `c` at level `level`, left-shifted into a common
# MAX_LEVEL-deep frame so that coordinates from different levels can be
# compared bit for bit.
@inline padded_coord(c::UInt32, lvl::Int8) = UInt64(c) << (MAX_LEVEL - Int(lvl))

# Whether the most significant set bit of `y` is strictly above that of
# `x`. The standard branch-free MSB comparison; see e.g. Chan, "A
# minimalist's implementation of an approximate nearest neighbor
# algorithm in fixed dimensions".
@inline less_msb(x::UInt64, y::UInt64) = x < y && x < (x ⊻ y)

"""
    isless(a::MortonKey{D}, b::MortonKey{D})

Curve order: by root index, then by Morton (Z-order) position, with an
ancestor immediately preceding its descendants.

Rather than walking all `MAX_LEVEL` bit planes, this finds the dimension
whose coordinates first differ at the highest bit and compares only
that one — `D` cheap operations instead of `32D`.
"""
function Base.isless(a::MortonKey{D}, b::MortonKey{D}) where {D}
    a.root != b.root && return a.root < b.root
    pa = ntuple(d -> padded_coord(a.coords[d], a.level), D)
    pb = ntuple(d -> padded_coord(b.coords[d], b.level), D)
    # Dimension 1 is most significant, so it wins ties in bit position.
    msd = 1
    xmax = pa[1] ⊻ pb[1]
    for d in 2:D
        x = pa[d] ⊻ pb[d]
        if less_msb(xmax, x)
            msd = d
            xmax = x
        end
    end
    # Identical position: the coarser node (the ancestor) comes first.
    xmax == 0 && return a.level < b.level
    return pa[msd] < pb[msd]
end
