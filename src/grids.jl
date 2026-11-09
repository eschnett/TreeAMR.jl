# Point sets computed from the index (M19): regular grids of points for
# `locate` and `interpolate`, and the lattice of sub-centres of every
# element, located without a search.
#
# A viewer samples 10⁵–10⁶ points on a line, a plane or a box at every
# output; an `isbits` array that computes each point from its index
# reaches a device kernel without an array of points, and the values come
# out in the grid's order. Visualization export (M9b) writes what they
# sample. See "Interpolation on element sets and patches" in CODE.md.

"""
    AbstractPointGrid{D,T,K} <: AbstractArray{NTuple{D,T},K}

A `K`-dimensional array of points in `D` dimensions computed from the
index, which [`locate`](@ref) and [`interpolate`](@ref) take as they
take an array of points, on any backend: [`UniformGrid`](@ref),
[`AffineGrid`](@ref), and [`PatchGrid`](@ref) of either.
"""
abstract type AbstractPointGrid{D,T,K} <: AbstractArray{NTuple{D,T},K} end

"""
    AffineGrid(origin, axes, dims)

The points `origin + Σₖ (iₖ − 1)·axes[k]` for `1 ≤ iₖ ≤ dims[k]`, with
`K = length(axes) ≤ D` axes, each a `D`-vector: a line, a plane or a box
at any orientation, through a `D`-dimensional domain. An `isbits` array,
so it reaches a kernel as one argument; `collect` gives the same points.
"""
struct AffineGrid{D,T,K} <: AbstractPointGrid{D,T,K}
    origin::NTuple{D,T}
    axes::NTuple{K,NTuple{D,T}}
    dims::NTuple{K,Int}
    # Inner, so that the checking constructor below is the only outer one.
    AffineGrid{D,T,K}(origin, axes, dims) where {D,T,K} = new{D,T,K}(origin, axes, dims)
end

function AffineGrid(origin, axes, dims)
    D = length(origin)
    K = length(axes)
    1 <= K <= D || throw(ArgumentError(
        "an AffineGrid in $D dimensions has between 1 and $D axes, got $K"))
    all(a -> length(a) == D, axes) || throw(ArgumentError(
        "each axis of an AffineGrid is a $D-vector, as its origin is; got " *
        "$(map(length, Tuple(axes)))"))
    length(dims) == K && all(>=(1), dims) || throw(ArgumentError(
        "an AffineGrid has one positive point count per axis, $K, got $(Tuple(dims))"))
    T = float(promote_type(eltype(Tuple(origin)), map(a -> eltype(Tuple(a)), Tuple(axes))...))
    return AffineGrid{D,T,K}(ntuple(d -> T(origin[d]), D),
                             ntuple(k -> ntuple(d -> T(axes[k][d]), D), K),
                             ntuple(k -> Int(dims[k]), K))
end

"""
    UniformGrid(lo, hi, dims)

The `prod(dims)` points of a regular grid filling the box from `lo` to
`hi`, both corners included, `dims[d]` points along dimension `d` (a
single point lies at `lo`): `lo[d] + (i_d − 1)·(hi[d] − lo[d])/(dims[d] −
1)`, with the step formed once. An `isbits` array, so it reaches a
kernel as one argument; `collect` gives the same points.
"""
struct UniformGrid{D,T} <: AbstractPointGrid{D,T,D}
    lo::NTuple{D,T}
    step::NTuple{D,T}
    dims::NTuple{D,Int}
    # Inner, so that the checking constructor below is the only outer one.
    UniformGrid{D,T}(lo, step, dims) where {D,T} = new{D,T}(lo, step, dims)
end

function UniformGrid(lo, hi, dims)
    D = length(lo)
    length(hi) == D && length(dims) == D || throw(ArgumentError(
        "a UniformGrid has a low corner, a high corner and a point count per " *
        "dimension, all of one length; got $(length(lo)), $(length(hi)) and " *
        "$(length(dims))"))
    all(>=(1), dims) || throw(ArgumentError(
        "a UniformGrid has at least one point per dimension, got $(Tuple(dims))"))
    T = float(promote_type(eltype(Tuple(lo)), eltype(Tuple(hi))))
    step = ntuple(d -> dims[d] == 1 ? zero(T) : (T(hi[d]) - T(lo[d])) / T(dims[d] - 1), D)
    return UniformGrid{D,T}(ntuple(d -> T(lo[d]), D), step, ntuple(d -> Int(dims[d]), D))
end

Base.size(g::Union{AffineGrid,UniformGrid}) = g.dims
Base.IndexStyle(::Type{<:AbstractPointGrid}) = IndexLinear()

# The subscripts of linear index `j`, by division, so that a kernel's
# linear index needs no `CartesianIndices`. Recursive rather than a
# closure that reassigns the remainder, which would box it (see
# `lagrange_series`).
@inline grid_subscripts(dims::Tuple, j::Int) = _subscripts(dims, j - 1)
@inline _subscripts(::Tuple{}, r::Int) = ()
@inline function _subscripts(dims::Tuple, r::Int)
    q, i = divrem(r, dims[1])
    return (i + 1, _subscripts(Base.tail(dims), q)...)
end

@inline function Base.getindex(g::UniformGrid{D,T}, j::Int) where {D,T}
    @boundscheck checkbounds(g, j)
    i = grid_subscripts(g.dims, j)
    return ntuple(d -> g.lo[d] + T(i[d] - 1) * g.step[d], Val(D))
end

@inline function Base.getindex(g::AffineGrid{D,T,K}, j::Int) where {D,T,K}
    @boundscheck checkbounds(g, j)
    i = grid_subscripts(g.dims, j)
    return ntuple(Val(D)) do d
        x = g.origin[d]
        for k in 1:K
            x += T(i[k] - 1) * g.axes[k][d]
        end
        x
    end
end

"""
    PatchGrid(grid, p)

The points of `grid` — a [`UniformGrid`](@ref) or an [`AffineGrid`](@ref)
of logical positions — as [`PatchPoint`](@ref)s of patch `p` of a forest
of patches, in the grid's shape and order.
"""
struct PatchGrid{D,T,K,G<:AbstractPointGrid{D,T,K}} <: AbstractArray{PatchPoint{D,T},K}
    grid::G
    p::Int
    PatchGrid(grid::AbstractPointGrid{D,T,K}, p::Integer) where {D,T,K} =
        new{D,T,K,typeof(grid)}(grid, Int(p))
end

Base.size(g::PatchGrid) = size(g.grid)
Base.IndexStyle(::Type{<:PatchGrid}) = IndexLinear()
@inline function Base.getindex(g::PatchGrid{D,T}, j::Int) where {D,T}
    @boundscheck checkbounds(g, j)
    return PatchPoint{D,T}(@inbounds(g.grid[j]), g.p)
end

# Whether an array of points is computed from its index, and so is taken
# on any backend.
isgrid(xs) = xs isa AbstractPointGrid || xs isa PatchGrid

# --- the element lattice ----------------------------------------------------------

# Point `j` of the lattice: block `b` (local, from 1), then its sub-centre
# `s` along each dimension, `s_d` from 1 to `E·k`, at `origin + h·W(2s −
# 1)/(2k)` with `W` the element's width in spacings — `P + 1`, or 1 for a
# set without elements, whose cells are the unit.
@inline function lattice_point!(ξs, ys, leaf, patch, codes, inside, origins, spacings,
                                patches, ::Val{D}, n, W, k, boffset, j) where {D}
    T = eltype(eltype(ξs))
    per = n^D
    b, r = divrem(j - 1, per)
    b += 1
    s = grid_subscripts(ntuple(_ -> n, Val(D)), r + 1)
    o, h = origins[b], spacings[b]
    x = ntuple(d -> o[d] + h * (T(W * (2s[d] - 1)) / T(2k)), Val(D))
    ξs[j] = x
    ys[j] = x
    leaf[j] = Int32(b + boffset)
    patch[j] = patches[b]
    codes[j] = 0x00
    inside[j] = true
    return nothing
end

@kernel function lattice_kernel!(ξs, ys, leaf, patch, codes, inside, @Const(origins),
                                 @Const(spacings), @Const(patches), ::Val{D}, n, W, k,
                                 boffset) where {D}
    j = @index(Global, Linear)
    lattice_point!(ξs, ys, leaf, patch, codes, inside, origins, spacings, patches, Val(D),
                   n, W, k, boffset, j)
end

"""
    element_lattice(fs, k; atlas = nothing) -> LocatedPoints

The centres of a `k`-fold subdivision of every element of every local
block of `fs` — of every cell, for a set without elements — as located
points, without a search: `kᴰ` per element, block by block and, within
a block, in the lexicographic order of the sub-cells along each
dimension. No point is on a face, so each is in the element it was made
in. [`positions`](@ref) gives them, physical through the atlas's
[`chart_map`](@ref TreeAMR.chart_map) when there is one, which also
makes [`interpolate`](@ref)'s derivatives at them physical.

The points are those of this rank's blocks, so over a distributed
forest every rank evaluates its own, with no messages.
"""
function element_lattice(fs::FieldSet{T,D}, k::Integer; atlas=nothing) where {T,D}
    k >= 1 || throw(ArgumentError(
        "element_lattice subdivides each element k-fold, so k is at least 1, got $k"))
    atlas === nothing || check_chart(chart_locate, atlas, T, Val(D), "a located batch")
    forest = fs.forest
    backend = get_backend(fs.work)
    W = fs.P == 0 ? 1 : fs.P + 1
    n = (forest.N ÷ W) * Int(k)
    npts = nblocks(fs) * n^D
    leaf = allocate(backend, Int32, npts)
    patch = allocate(backend, Int32, npts)
    ξ = allocate(backend, NTuple{D,T}, npts)
    y = allocate(backend, NTuple{D,T}, npts)
    code = allocate(backend, UInt8, npts)
    inside = allocate(backend, Bool, npts)
    if npts > 0
        origins = todevice(backend, block_origins(forest, T))
        spacings = todevice(backend, block_spacings(forest, T))
        patches = todevice(backend, block_patches(forest))
        pointkernel(lattice_kernel!, backend, npts)(
            ξ, y, leaf, patch, code, inside, origins, spacings, patches, Val(D), n, W,
            Int(k), first(blockrange(forest)) - 1; ndrange=npts)
        synchronize(backend)
    end
    return LocatedPoints{D,T}(forest, generation(forest), leaf, patch, ξ, y, code, inside,
                              atlas, :error, LocalBatch())
end

# --- positions --------------------------------------------------------------------

@inline position_of(::Nothing, ξ, p, inside) = ξ
@inline position_of(atlas, ξ, p, inside) = inside ? chart_map(atlas, ξ, Int(p)) : ξ

@kernel function positions_kernel!(xs, @Const(ξs), @Const(patches), @Const(inside), atlas)
    j = @index(Global, Linear)
    xs[j] = position_of(atlas, ξs[j], patches[j], inside[j])
end

"""
    positions(pts::LocatedPoints) -> AbstractVector

The position of every located point, on the points' backend: logical
without an atlas — on a forest of patches the position in its patch,
`pts.patch` saying which — and physical through
[`chart_map`](@ref TreeAMR.chart_map) with one. A point outside keeps the
position it was given.
"""
function positions(pts::LocatedPoints{D,T}) where {D,T}
    pts.atlas === nothing || check_chart(chart_map, pts.atlas, T, Val(D), "positions")
    backend = get_backend(pts)
    xs = allocate(backend, NTuple{D,T}, length(pts))
    length(pts) == 0 && return xs
    pointkernel(positions_kernel!, backend, length(pts))(xs, pts.ξ, pts.patch, pts.inside,
                                                         pts.atlas; ndrange=length(pts))
    synchronize(backend)
    return xs
end
