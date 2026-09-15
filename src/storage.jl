"""
    FieldSet{T,D,R,A}

Block storage for `nvars` variables over every leaf of a
[`Forest`](@ref): one big persistent array holding all leaf blocks
including their ghosts,

    work :: A   # size (N+2G[1], ..., N+2G[D], nvars, nblocks)

Cell indices vary fastest (so a GPU reads them coalesced), then the
variable index, then the block index; blocks are ordered by the forest's
Morton key order, so `forest.leaves[b]` is the key of block `b`.

`G` is the **ghost width, per dimension**, and it lives here rather than
on the forest (amended in M8): it says how far a stencil reaches into a
neighbor's data, which is a property of what is stored and not of how
space is cut up. Two field sets over one forest with different `G` — an
evolved state with `G = 2` and the fluxes computed from it with `G = 0` —
is the normal case. It is **required**: like the operator orders, it
follows from the application's discretization, which the mesh cannot
know. Pass a plain integer for the uniform case or an `NTuple{D,Integer}`
for one width per dimension; it is stored as an `NTuple{D,Int}`. The
invariant `N ≥ 2G[d]` is checked here, per dimension.

Element type `T` is generic: `Float32` for GPUs, `Float64` on a host,
or a software type such as a double-`Float32` where no hardware fp64
exists. It defaults to the forest's own [`floattype`](@ref) `R`, which is
also the type the geometry is computed in; the two are separate
parameters only so that a field set may deliberately store something
narrower than its coordinates.

`backend` says where the storage lives, and thereby where every kernel
over it runs (M6): each one takes its backend from `get_backend(fs.work)`,
so this one keyword is the whole switch. A [`GhostSchedule`](@ref) used
with this field set must be built for the same backend. On a device with
no hardware fp64 a `Float64` field set is rejected here, with a message,
rather than failing later inside a kernel compilation.

A field set is tied to the forest's *current* leaf array. Block indices
are deliberately not stable across regridding (M4), which compacts the
block slots and rebuilds the storage.

    FieldSet(forest, nvars; G)          # element type = floattype(forest)
    FieldSet{Float32}(forest, nvars; G = (2, 0))
    FieldSet{Float32}(forest, nvars; G = 2, backend = CUDABackend())

# Examples

```jldoctest
julia> forest = Forest((2,); N = 4);

julia> fs = FieldSet(forest, 3; G = 1);

julia> size(fs.work)
(6, 3, 2)

julia> FieldSet(Forest((1, 1); N = 8), 1; G = (2, 0)).G
(2, 0)
```
"""
mutable struct FieldSet{T,D,R,A<:AbstractArray{T}}
    const forest::Forest{D,R}
    const nvars::Int
    const G::NTuple{D,Int}
    # Replaced wholesale by regridding, which compacts the block slots
    # into a freshly sized array. Mutable so that references an
    # application already holds stay valid across a regrid.
    work::A
end

# `G` as a sentinel-defaulted keyword rather than a required one, so that
# omitting it reports *why* there is no default — the same reason
# `Operators` has no default order.
function FieldSet{T}(forest::Forest{D,R}, nvars::Integer;
                     G::Union{Integer,Tuple{Vararg{Integer}},Nothing}=nothing,
                     backend::Backend=CPU()) where {T,D,R}
    nvars > 0 || throw(ArgumentError("nvars must be positive, got $nvars"))
    G === nothing && throw(ArgumentError(
        "FieldSet has no default ghost width: pass `G` explicitly. It follows " *
        "from what this field set stores and from the operator orders that read " *
        "it — an evolved state needs G >= prolongation ÷ 2, a computed flux needs " *
        "none at all — which the mesh cannot know."))
    check_floattype(T, backend)
    ghosts = ghostwidths(G, Val(D))
    stored = storedsize(forest.N, ghosts)
    work = allocate(backend, T, (stored..., Int(nvars), nleaves(forest)))
    # Through the kernel rather than `fill!`, for the first-touch reason
    # in `zerofill!` below.
    zerofill!(work, backend)
    return FieldSet{T,D,R,typeof(work)}(forest, Int(nvars), ghosts, work)
end
FieldSet(forest::Forest{D,R}, nvars::Integer; kwargs...) where {D,R} =
    FieldSet{R}(forest, nvars; kwargs...)

# The uniform shorthand, and the per-dimension invariant. `N ≥ 2G[d]` is
# what makes a block's high exchange region reachable from one ring of
# neighbors even when those neighbors are finer and each spans only
# `N/2` coarse cells (see "Blocks" in CODE.md).
ghostwidths(G::Integer, ::Val{D}) where {D} = ghostwidths(ntuple(_ -> G, D), Val(D))
function ghostwidths(G::Tuple{Vararg{Integer}}, ::Val{D}) where {D}
    length(G) == D || throw(ArgumentError(
        "G must have one width per dimension: got $(length(G)) for a $D-dimensional " *
        "forest, $G. Pass a plain integer for the uniform case."))
    all(>=(0), G) || throw(ArgumentError("G must be nonnegative, got $G"))
    return ntuple(d -> Int(G[d]), D)
end

function storedsize(N::Int, G::NTuple{D,Int}) where {D}
    all(d -> N >= 2 * G[d], 1:D) || throw(ArgumentError(
        "N must be >= 2G in every dimension, got N=$N, G=$G. A block's high " *
        "ghost layers must be reachable from one ring of neighbors, and a finer " *
        "neighbor spans only N/2 of this block's cells."))
    return ntuple(d -> N + 2 * G[d], D)
end

"""
    get_backend(fs::FieldSet)

The KernelAbstractions backend this field set's storage lives on — the
backend every kernel over it is launched with.
"""
KernelAbstractions.get_backend(fs::FieldSet) = get_backend(fs.work)

"""
    nblocks(fs::FieldSet)

The number of blocks stored — one per leaf of the underlying forest.
"""
nblocks(fs::FieldSet) = size(fs.work, ndims(fs.work))

"""
    blockkey(fs::FieldSet, b::Integer)

The [`MortonKey`](@ref) of block `b`.
"""
blockkey(fs::FieldSet, b::Integer) = fs.forest.leaves[b]

"""
    blockview(fs::FieldSet, b::Integer)
    blockview(fs::FieldSet, b::Integer, v::Integer)

A view of block `b` **including ghosts** — shape `(N+2G[1], ...,
N+2G[D], nvars)`, or without the trailing `nvars` for a single variable
`v`.
"""
blockview(fs::FieldSet{T,D}, b::Integer) where {T,D} =
    view(fs.work, ntuple(_ -> Colon(), D + 1)..., b)
blockview(fs::FieldSet{T,D}, b::Integer, v::Integer) where {T,D} =
    view(fs.work, ntuple(_ -> Colon(), D)..., v, b)

"""
    interiorview(fs::FieldSet, b::Integer)
    interiorview(fs::FieldSet, b::Integer, v::Integer)

A view of block `b` **excluding ghosts** — shape `(N, ..., N, nvars)`,
or `(N, ..., N)` for a single variable `v`. This is the part that tiles
the domain and that the ODE state vector holds.
"""
interiorview(fs::FieldSet{T,D}, b::Integer) where {T,D} =
    view(fs.work, interiorranges(fs)..., :, b)
interiorview(fs::FieldSet{T,D}, b::Integer, v::Integer) where {T,D} =
    view(fs.work, interiorranges(fs)..., v, b)

# The owned range per dimension, `G[d]+1 … G[d]+N`.
interiorranges(fs::FieldSet{T,D}) where {T,D} =
    ntuple(d -> (fs.G[d] + 1):(fs.G[d] + fs.forest.N), D)

"""
    coordinates([S], fs::FieldSet, b::Integer, idx::NTuple{D,Integer})

The physical position of point `idx` of block `b` of `fs`. `idx` is
1-based over the **stored** array, so the owned cells are
`G[d]+1 : G[d]+N` and values outside that range name ghosts, whose
positions are still well defined and lie outside the block.

The position depends on the *field set's* ghost width, which is why this
takes a field set and a block index rather than a forest and a key
(replacing `cell_center`, which assumed the forest carried `G`).

The optional leading `S` is the type the arithmetic is done in, as
everywhere in `geometry.jl`; it defaults to the field set's own element
type `T`, which is what [`fill_by_coordinates!`](@ref) and
[`boundary_by_coordinates`](@ref) hand their callbacks.
"""
function coordinates(::Type{S}, fs::FieldSet{T,D}, b::Integer,
                     idx::NTuple{D,<:Integer}) where {S,T,D}
    forest = fs.forest
    origin = block_origin(S, forest, blockkey(fs, b))
    h = spacing(S, forest, blockkey(fs, b))
    # `1//2` rather than `0.5`: the literal would be a `Float64` operand
    # and would drag the whole expression into fp64. The conversion is
    # exact and folds away at compile time.
    half = oftype(h, 1//2)
    return ntuple(d -> origin[d] + (Int(idx[d]) - fs.G[d] - half) * h, D)
end
coordinates(fs::FieldSet{T,D}, b::Integer, idx::NTuple{D,<:Integer}) where {T,D} =
    coordinates(T, fs, b, idx)

@kernel function coordinates_kernel!(work, f, @Const(origins), @Const(spacings),
                                     ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                     # (i1..iD, var, block)
    v, b = I[D + 1], I[D + 2]
    origin, h = origins[b], spacings[b]
    # The stored index of interior cell i is i + G[d], so `coordinates`'
    # (idx - G[d] - 1/2) is just (i - 1/2) — the same arithmetic, in the
    # same order, on the same origin and spacing, so this reproduces
    # `coordinates` bit for bit. `1//2` rather than `0.5` for the same
    # reason as there: the literal would be an fp64 operand.
    x = ntuple(d -> origin[d] + (I[d] - oftype(h, 1//2)) * h, Val(D))
    work[ntuple(d -> I[d] + G[d], Val(D))..., v, b] = f(x, v)
end

"""
    fill_by_coordinates!(f, fs::FieldSet)

Set every interior cell of every block from the callback
`f(x, v) -> value`, where `x` is the cell center (an `NTuple{D,T}` in the
field set's own element type, see [`coordinates`](@ref)) and `v` the
variable index. Ghosts are left untouched — they are filled by the ghost
exchange (M2).

`f` is called once per cell from a KernelAbstractions kernel, so it runs
concurrently across blocks (M5) and must be a pure function of its
arguments. The tree is not consulted: the kernel gets the geometry as
the two plain per-block arrays [`block_origins`](@ref) and
[`block_spacings`](@ref), which is what makes it a device kernel (M6).

!!! note "Callbacks on a device"
    The callback becomes a kernel argument, so everything it closes over
    must be `isbits`. A captured `Type` is the usual trip: write
    `oftype(x[1], 2)` rather than closing over `T` and calling `T(2)`.
    The same rule covers captured arrays (pass a device array, or index
    the one the callback is already given) and any mutable state, which
    the purity requirement rules out anyway.
"""
function fill_by_coordinates!(f, fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    backend = get_backend(fs.work)
    # The geometry is built on the host and moved to wherever the kernel
    # runs. This is a setup-frequency call (initial data, and one pass of
    # `adapt_to_initial_data!`), not part of the per-evaluation path, so
    # the upload is not worth caching.
    origins = todevice(backend, block_origins(forest, T))
    spacings = todevice(backend, block_spacings(forest, T))
    coordinates_kernel!(backend)(fs.work, f, origins, spacings,
                                 Val(D), Val(fs.G);
                                 ndrange=(ntuple(_ -> forest.N, D)..., fs.nvars,
                                          nblocks(fs)))
    synchronize(backend)
    return fs
end

@kernel function zero_kernel!(work)
    I = @index(Global, NTuple)
    work[I...] = zero(eltype(work))
end

# Zeroing fresh block storage through a kernel rather than `fill!` is
# not about speed: on a multi-socket node it is the *first touch* that
# decides which NUMA domain each page lands in, and a serial `fill!`
# would park the whole array on whichever domain the driver thread sits
# on. The kernel touches each block from the same chunk of the ndrange
# that will later compute on it.
function zerofill!(work, backend)
    zero_kernel!(backend)(work; ndrange=size(work))
    synchronize(backend)
    return work
end
