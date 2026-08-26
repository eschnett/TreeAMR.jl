# The ghost exchange schedule.
#
# Neighbor finding is a regridding-frequency operation; ghost filling
# runs at every RHS evaluation. So the schedule — the flat list of
# source/target region pairs — is built once whenever the tree changes,
# and `fill_ghosts!` only replays it. No tree query appears in the
# per-evaluation path.
#
# Every transfer, whatever its kind, is a tensor product of D
# one-dimensional stencils. That is what makes one kernel serve copies,
# restrictions, and prolongations alike: a copy is just a width-1
# stencil with weight 1.
#
# The 1D stencils depend only on the direction `δ` and on a child offset
# `o` — never on which particular blocks are involved — so all transfers
# sharing `(kind, δ, o)` share one set of stencils and are batched into a
# single `TransferGroup` holding just the block-index pairs.

"""
    Stencil1D{T}

One dimension of a transfer: for each cell of the target's index range,
the first source cell and the `order` weights to apply from there.

`srcstart[i]` and `weights[:, i]` describe target cell
`targetfirst + i - 1`.
"""
struct Stencil1D{T}
    targetfirst::Int
    srcstart::Vector{Int32}
    weights::Matrix{T}
end

ntarget(s::Stencil1D) = length(s.srcstart)
Base.@propagate_inbounds stencilorder(s::Stencil1D) = size(s.weights, 1)

"""
    TransferGroup{T,D}

All transfers that share one set of 1D stencils — same kind, same
direction, same child offset — reduced to a list of
`(targetblock, sourceblock)` index pairs. One kernel launch per group.
"""
struct TransferGroup{T,D}
    kind::Symbol                      # :copy, :restrict, or :prolong
    stencils::NTuple{D,Stencil1D{T}}
    targetblocks::Vector{Int32}
    sourceblocks::Vector{Int32}
end

boxsize(g::TransferGroup{T,D}) where {T,D} = ntuple(d -> ntarget(g.stencils[d]), D)
ntransfers(g::TransferGroup) = length(g.targetblocks)

"""
    BoundaryRegion{D}

A ghost region of a block that faces outside a non-periodic domain, and
so is filled by the user's boundary hook rather than by an inter-block
transfer.
"""
struct BoundaryRegion{D}
    block::Int32
    direction::NTuple{D,Int}
    region::CartesianIndices{D,NTuple{D,UnitRange{Int}}}
end

"""
    GhostSchedule{T,D}

The precomputed ghost exchange for one forest, replayed by
[`fill_ghosts!`](@ref).

The phasing follows `CODE.md`:

- `phase1` holds all same-level copies and all restrictions. Each reads
  only *interior* cells of other blocks and writes only ghosts, so the
  whole phase is race free and order independent.
- `phase2` holds the prolongations, grouped by target level and ordered
  **coarsest target first**. The sweep is required because a
  prolongation stencil may read the coarse source's own ghosts, which
  may themselves have been prolongated from a still-coarser block —
  legal under 2:1 balance, where levels `l-2, l-1, l` can meet.
- `boundaries` lists the ghost regions facing outside a non-periodic
  domain, filled by the user hook after the inter-block phases.

Periodic boundaries appear nowhere special here: the tree wraps around,
so they are ordinary copies, restrictions, and prolongations.

A schedule is tied to the forest's leaf array as it was when built. It
must be rebuilt after any refinement, coarsening, or regridding.

    GhostSchedule(forest, operators::Operators; T=Float64)

`operators` is required: interpolation order follows from the
application's discretization, so there is no order the mesh could
sensibly default to. See [`Operators`](@ref).
"""
struct GhostSchedule{T,D}
    forest::Forest{D}
    generation::Int                              # forest generation it was built for
    operators::Operators
    phase1::Vector{TransferGroup{T,D}}
    phase2::Vector{Vector{TransferGroup{T,D}}}   # by target level, coarsest first
    levels::Vector{Int}                          # target level of each phase2 entry
    boundaries::Vector{BoundaryRegion{D}}
end

"""
    isstale(schedule::GhostSchedule)

Whether the forest has changed since `schedule` was built, in which case
it must be rebuilt before [`fill_ghosts!`](@ref) will accept it.
"""
isstale(s::GhostSchedule) = generation(s.forest) != s.generation

# --- 1D stencil construction ---------------------------------------------
#
# Index conventions, all in *stored* indices (1:N+2G, interior G+1:G+N):
#
#   δ_d = +1  target is the high ghost slab  G+N+1 : G+N+G
#   δ_d = -1  target is the low  ghost slab      1 : G
#   δ_d =  0  target spans the block's own extent
#
# For restriction the tangential extent is halved, since each of the
# 2^(tangential) fine neighbors supplies one half.

# Weights for a window of `p` consecutive source cells starting at `lo`,
# evaluated at `x`.
#
# Shifting a window (as restriction does near a coarse-fine interface)
# costs no accuracy — Lagrange interpolation through any p distinct
# nodes is exact for degree < p — but it must never shift so far that
# the target leaves the node hull, which would turn interpolation into
# extrapolation and amplify error instead of damping it. Checked here,
# at schedule-build time, so it costs nothing per evaluation.
function interpolation_weights(lo::Int, p::Int, x::Real, what::AbstractString)
    lo <= x <= lo + p - 1 || throw(ArgumentError(
        "$what would extrapolate: target $x lies outside the source window " *
        "[$lo, $(lo + p - 1)]. The block geometry cannot support this order."))
    return lagrange_weights(collect(lo:(lo + p - 1)), x)
end

# Target range of a transfer in dimension d.
function target_range(N::Int, G::Int, δd::Int, od::Int, halved::Bool)
    δd == 1 && return (G + N + 1):(G + N + G)
    δd == -1 && return 1:G
    halved && return (G + 1 + od * (N ÷ 2)):(G + od * (N ÷ 2) + N ÷ 2)
    return (G + 1):(G + N)
end

# Same-level copy: a pure shift of N cells against the direction.
function copy_stencil(::Type{T}, N::Int, G::Int, δd::Int) where {T}
    rng = target_range(N, G, δd, 0, false)
    srcstart = Int32[j - N * δd for j in rng]
    return Stencil1D{T}(first(rng), srcstart, ones(T, 1, length(rng)))
end

# Restriction, fine -> coarse. `od` is the source's child offset within
# the (refined) node adjacent to the target, so it selects which half of
# the target's extent this source covers.
function restrict_stencil(::Type{T}, N::Int, G::Int, δd::Int, od::Int, p::Int) where {T}
    rng = target_range(N, G, δd, od, true)
    # Step into the adjacent node's frame, where the source's parent has
    # the target block's own layout.
    shift = -N * δd
    srcstart = Vector{Int32}(undef, length(rng))
    weights = Matrix{T}(undef, p, length(rng))
    for (i, j) in enumerate(rng)
        q = j + shift - G - 1                       # interior cell of the adjacent node
        f0 = G + 1 + 2 * (q - od * (N ÷ 2))         # first of its two fine cells
        # The coarse cell center falls on the interface between the two
        # fine cells, at f0 + 1/2. Center the window there, then shift it
        # to stay inside the fine block's interior — restriction must
        # never read another block's ghosts.
        lo = clamp(f0 - p ÷ 2 + 1, G + 1, G + N - p + 1)
        srcstart[i] = lo
        weights[:, i] = interpolation_weights(lo, p, f0 + 0.5, "restriction")
    end
    return Stencil1D{T}(first(rng), srcstart, weights)
end

# Direction from the target's *parent* to the source, per dimension.
#
# A ghost slab leaves the parent only on the side the target itself sits
# on: a high ghost of a low child is still inside the parent, covering
# the sibling's ground. For a corner this is per-dimension — some
# dimensions leave the parent while others do not — so the source can be
# a diagonal neighbor of the target while being a *face* neighbor of the
# target's parent.
source_direction(δd::Int, od::Int) =
    (δd == 1 && od == 1) ? 1 : (δd == -1 && od == 0) ? -1 : 0

# Prolongation, coarse -> fine. `od` is the *target*'s child offset
# within its own parent, which fixes where the target's cells fall
# inside the coarse frame.
function prolong_stencil(::Type{T}, N::Int, G::Int, δd::Int, od::Int, p::Int) where {T}
    rng = target_range(N, G, δd, od, false)
    srcstart = Vector{Int32}(undef, length(rng))
    weights = Matrix{T}(undef, p, length(rng))
    for (i, j) in enumerate(rng)
        # Fine offset from the parent's interior start, so that coarse
        # cell c covers fine cells 2c and 2c+1.
        φ = od * N + j - G - 1
        # Fine cell φ sits at coarse coordinate φ/2 - 1/4; center a
        # window of p coarse cells on it. Translating into the source's
        # stored frame costs N cells per unit of direction.
        origin = -N * source_direction(δd, od) + G + 1
        lo = clamp(cld(φ, 2) - p ÷ 2 + origin, 1, N + 2G - p + 1)
        srcstart[i] = lo
        weights[:, i] = interpolation_weights(lo, p, φ / 2 - 0.25 + origin, "prolongation")
    end
    return Stencil1D{T}(first(rng), srcstart, weights)
end

# --- Schedule construction -----------------------------------------------

# A block's offset within its parent, per dimension.
childoffset(k::MortonKey{D}) where {D} = ntuple(d -> Int(k.coords[d]) & 1, D)

function GhostSchedule(forest::Forest{D}, operators::Operators;
                       T::Type=Float64) where {D}
    check_operators(forest, operators)
    N, G = forest.N, forest.G
    dirs = alldirections(Val(D))

    # Transfers are collected keyed by (kind, δ, o): everything sharing
    # that key shares one set of 1D stencils.
    pairs = Dict{Tuple{Symbol,NTuple{D,Int},NTuple{D,Int}},
                 Tuple{Vector{Int32},Vector{Int32}}}()
    prolonglevel = Dict{Tuple{Symbol,NTuple{D,Int},NTuple{D,Int}},Int}()
    boundaries = BoundaryRegion{D}[]

    zerooffset = ntuple(_ -> 0, D)
    for (b, k) in enumerate(forest.leaves)
        for δ in dirs
            nbrs = neighbor_keys(forest, k, δ)
            if isempty(nbrs)
                region = CartesianIndices(ntuple(d -> target_range(N, G, δ[d], 0, false), D))
                push!(boundaries, BoundaryRegion{D}(b, δ, region))
                continue
            end
            nblevel = level(first(nbrs))
            if nblevel == level(k)
                s = find_leaf(forest, only(nbrs))
                key = (:copy, δ, zerooffset)
                push!.(get!(pairs, key, (Int32[], Int32[])), (Int32(b), Int32(s)))
            elseif nblevel < level(k)
                # Coarser neighbor: this block's ghosts are prolongated.
                # The stencil geometry depends on where this block sits
                # inside its own parent.
                s = find_leaf(forest, only(nbrs))
                key = (:prolong, δ, childoffset(k))
                push!.(get!(pairs, key, (Int32[], Int32[])), (Int32(b), Int32(s)))
                prolonglevel[key] = level(k)
            else
                # Finer neighbors: each supplies one part of this block's
                # ghost region, selected by its offset within its parent.
                for nb in nbrs
                    s = find_leaf(forest, nb)
                    key = (:restrict, δ, childoffset(nb))
                    push!.(get!(pairs, key, (Int32[], Int32[])), (Int32(b), Int32(s)))
                end
            end
        end
    end

    build(kind, δ, o) =
        kind === :copy ? ntuple(d -> copy_stencil(T, N, G, δ[d]), D) :
        kind === :restrict ? ntuple(d -> restrict_stencil(T, N, G, δ[d], o[d],
                                                          operators.restriction), D) :
        ntuple(d -> prolong_stencil(T, N, G, δ[d], o[d], operators.prolongation), D)

    phase1 = TransferGroup{T,D}[]
    bylevel = Dict{Int,Vector{TransferGroup{T,D}}}()
    for (key, (targets, sources)) in pairs
        kind, δ, o = key
        group = TransferGroup{T,D}(kind, build(kind, δ, o), targets, sources)
        if kind === :prolong
            push!(get!(bylevel, prolonglevel[key], TransferGroup{T,D}[]), group)
        else
            push!(phase1, group)
        end
    end

    levels = sort!(collect(keys(bylevel)))          # coarsest targets first
    phase2 = [bylevel[l] for l in levels]
    return GhostSchedule{T,D}(forest, generation(forest), operators, phase1, phase2,
                              levels, boundaries)
end

function Base.show(io::IO, s::GhostSchedule{T,D}) where {T,D}
    ncopy = sum(ntransfers, filter(g -> g.kind === :copy, s.phase1); init=0)
    nrest = sum(ntransfers, filter(g -> g.kind === :restrict, s.phase1); init=0)
    nprol = sum(gs -> sum(ntransfers, gs; init=0), s.phase2; init=0)
    print(io, "GhostSchedule{", T, ",", D, "}(", ncopy, " copies, ", nrest,
          " restrictions, ", nprol, " prolongations over ", length(s.phase2),
          " level(s), ", length(s.boundaries), " boundary regions)")
end
