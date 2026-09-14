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
# sharing `(kind, δ, o)` are batched into a single `TransferGroup`
# holding just the block-index pairs. Prolongation groups are
# additionally split by target level, because the phase-2 sweep
# schedules a group by the level of its targets (see `GroupKey`).

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
direction, same child offset, and for prolongations the same target
level — reduced to a list of `(targetblock, sourceblock)` index pairs.

A group is a batch of work, not a unit of scheduling: a phase runs as
one parallel loop over [`PhaseSlice`](@ref TreeAMR.PhaseSlice)s, and a
big group is launched in several of them.
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
    PhaseSlice

A contiguous run of one [`TransferGroup`](@ref TransferGroup)'s
transfers — the unit of work one thread takes when a whole phase is run
as a single parallel loop, and the reason `cells` is carried: the
slices are dealt out largest first, so a phase made of one enormous face
group and a hundred tiny corner ones still balances.
"""
struct PhaseSlice
    group::Int32
    first::Int32
    last::Int32
    cells::Int32                      # target cells, the cost of the slice
end

# Cells per slice. Big enough that a slice is worth a kernel launch (a
# few microseconds of work), small enough that a hundred-odd threads
# each get several slices of the largest group.
const SLICE_CELLS = 4096

# Cut a phase's groups into slices of about `SLICE_CELLS` target cells,
# never crossing a group boundary (a group is one set of stencils, hence
# one kernel), and order them largest first so that dealing them round
# robin balances the phase.
function phase_plan(groups::AbstractVector{<:TransferGroup})
    slices = PhaseSlice[]
    for (g, group) in enumerate(groups)
        n = ntransfers(group)
        n == 0 && continue
        box = prod(boxsize(group))
        per = max(1, cld(SLICE_CELLS, max(box, 1)))
        lo = 1
        while lo <= n
            hi = min(n, lo + per - 1)
            push!(slices, PhaseSlice(Int32(g), Int32(lo), Int32(hi),
                                     Int32(min(box * (hi - lo + 1), typemax(Int32)))))
            lo = hi + 1
        end
    end
    # `sort!` is stable, so equal-cost slices keep their group order and
    # the plan is a function of the schedule alone.
    return sort!(slices; by=s -> -s.cells)
end

"""
    GhostSchedule{T,D,R}

The precomputed ghost exchange for one forest, replayed by
[`fill_ghosts!`](@ref).

The phasing follows `CODE.md`:

- `phase1` holds all same-level copies and all restrictions. Each reads
  only *interior* cells of other blocks and writes only ghosts, so the
  whole phase is race free and order independent.
- `phase2` holds the prolongations, grouped by target level and ordered
  **coarsest target first**; every transfer in a group has its target at
  that group's level. The sweep is required because a
  prolongation stencil may read the coarse source's own ghosts, which
  may themselves have been prolongated from a still-coarser block —
  legal under 2:1 balance, where levels `l-2, l-1, l` can meet.
- `boundaries` lists the ghost regions facing outside a non-periodic
  domain, filled by the user hook after the inter-block phases.
- `phase1plan` and `phase2plans` cut each phase into balanced
  [`PhaseSlice`](@ref TreeAMR.PhaseSlice)s, so that a phase runs as one
  parallel loop rather than as a sequence of separate launches.

Periodic boundaries appear nowhere special here: the tree wraps around,
so they are ordinary copies, restrictions, and prolongations.

A schedule is tied to the forest's leaf array as it was when built. It
must be rebuilt after any refinement, coarsening, or regridding.

    GhostSchedule(forest, operators::Operators; T=floattype(forest))

`operators` is required: interpolation order follows from the
application's discretization, so there is no order the mesh could
sensibly default to. See [`Operators`](@ref).
"""
struct GhostSchedule{T,D,R}
    forest::Forest{D,R}
    generation::Int                              # forest generation it was built for
    operators::Operators
    phase1::Vector{TransferGroup{T,D}}
    phase2::Vector{Vector{TransferGroup{T,D}}}   # by target level, coarsest first
    levels::Vector{Int}                          # target level of each phase2 entry
    boundaries::Vector{BoundaryRegion{D}}
    # How each phase is cut up for the threads, precomputed here rather
    # than at every ghost fill (see `PhaseSlice`).
    phase1plan::Vector{PhaseSlice}
    phase2plans::Vector{Vector{PhaseSlice}}
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
function interpolation_weights(lo::Int, p::Int, x::Rational, what::AbstractString)
    lo <= x <= lo + p - 1 || throw(ArgumentError(
        "$what would extrapolate: target $x lies outside the source window " *
        "[$lo, $(lo + p - 1)]. The block geometry cannot support this order."))
    return lagrange_weights([Rational(j) for j in lo:(lo + p - 1)], x)
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
        weights[:, i] = interpolation_weights(lo, p, f0 + 1//2, "restriction")
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
        weights[:, i] = interpolation_weights(lo, p, φ//2 - 1//4 + origin, "prolongation")
    end
    return Stencil1D{T}(first(rng), srcstart, weights)
end

# Conservative prolongation weights: reconstruct a degree-(p-1)
# polynomial over the coarse cell from the averages of the p cells
# centered on it, then average that reconstruction over each half.
#
# Working through the primitive W (whose increments across cell
# boundaries are the cell averages) turns "reconstruct from averages"
# into ordinary interpolation: W is the degree-p polynomial through the
# p+1 boundary values, and a subcell average is a difference of W
# divided by the subcell width.
#
# The result is exactly conservative: the two subcell weight vectors
# average to the unit vector on the center cell, so the children always
# average back to their parent whatever the data. In rational arithmetic
# that is an algebraic identity rather than a roundoff claim — it holds
# exactly, at every order and for every element type the weights are
# later rounded into.
function conservative_prolong_weights(p::Int)
    r = (p - 1) ÷ 2
    boundaries = [-r - 1//2 + i for i in 0:p]       # p+1 cell boundaries
    # W(x) = Σ_i L_i(x) W_i and W_i = Σ_{t<i} ū_t, so ū_t carries weight
    # Σ_{i>t} L_i(x).
    L = lagrange_weights(boundaries, 0//1)          # at the cell's own center
    tail = [sum(L[(t + 2):(p + 1)]) for t in 0:(p - 1)]
    low = [2 * (tail[t + 1] - (t < r ? 1//1 : 0//1)) for t in 0:(p - 1)]
    high = [2 * ((t < r + 1 ? 1//1 : 0//1) - tail[t + 1]) for t in 0:(p - 1)]
    return low, high
end

function conservative_prolong_stencil(::Type{T}, N::Int, G::Int, δd::Int, od::Int,
                                      p::Int) where {T}
    rng = target_range(N, G, δd, od, false)
    low, high = conservative_prolong_weights(p)
    r = (p - 1) ÷ 2
    srcstart = Vector{Int32}(undef, length(rng))
    weights = Matrix{T}(undef, p, length(rng))
    for (i, j) in enumerate(rng)
        φ = od * N + j - G - 1
        # Unlike the point-value case, the fine cell *is* a subcell of
        # coarse cell fld(φ, 2) rather than a point near it.
        c = fld(φ, 2)
        subcell = φ - 2c                            # 0 = low half, 1 = high
        origin = -N * source_direction(δd, od) + G + 1
        srcstart[i] = c - r + origin
        weights[:, i] = subcell == 0 ? low : high
    end
    return Stencil1D{T}(first(rng), srcstart, weights)
end

# The stencils a given operator family uses, so that the schedule and
# the regrid transfer cannot drift apart.
prolongation_stencil(::Type{T}, N, G, δd, od, ops::Operators) where {T} =
    ops.family === Conservative ?
    conservative_prolong_stencil(T, N, G, δd, od, ops.prolongation) :
    prolong_stencil(T, N, G, δd, od, ops.prolongation)

# Conservative restriction is the exact volume average, which the
# point-value builder already produces at order 2: its window is a
# cell's own two children and its weights are 1/2.
restriction_stencil(::Type{T}, N, G, δd, od, ops::Operators) where {T} =
    restrict_stencil(T, N, G, δd, od, ops.restriction)

# --- Schedule construction -----------------------------------------------

# A block's offset within its parent, per dimension.
childoffset(k::MortonKey{D}) where {D} = ntuple(d -> Int(k.coords[d]) & 1, D)

# What one ghost region of one block needs, as found by the neighbor
# search. Transfers sharing a `GroupKey` share their 1D stencils and are
# batched into one `TransferGroup`, hence one kernel launch.
#
# For prolongations the key also carries the target's *level*. The
# stencils do not depend on it — but the phase-2 sweep does: a group is
# scheduled at the level of its targets, so merging two levels into one
# group would file them both under one of the two and silently defeat
# the coarsest-target-first ordering that a prolongation reading its
# source's own prolongated ghosts relies on. (Found in M5, while
# threading this loop; see `CODE.md`.) Phase 1 is order independent by
# construction, so copies and restrictions carry `level = 0` and stay
# batched across levels — one launch instead of one per level.
struct GroupKey{D}
    kind::Symbol
    direction::NTuple{D,Int}
    offset::NTuple{D,Int}
    level::Int
end

# The per-key transfer lists a schedule is assembled from: for each
# group key, the target blocks and the source blocks of its transfers,
# in the order the blocks were walked.
const TransferPairs{D} = Dict{GroupKey{D},Tuple{Vector{Int32},Vector{Int32}}}

# The neighbor search for a single block: every transfer its ghosts
# need, appended to `pairs`, plus the regions that face out of the
# domain and so belong to the boundary hook instead. Depends on the tree
# alone, which is what makes it the part that threads — each task owns
# its own `pairs` and `boundaries`.
function block_sources!(pairs::TransferPairs{D},
                        boundaries::Vector{BoundaryRegion{D}},
                        forest::Forest{D}, b::Int, dirs) where {D}
    k = forest.leaves[b]
    N, G = forest.N, forest.G
    zerooffset = ntuple(_ -> 0, D)
    record!(kind, δ, offset, lvl, s) =
        push!.(get!(pairs, GroupKey{D}(kind, δ, offset, lvl), (Int32[], Int32[])),
               (Int32(b), Int32(s)))
    for δ in dirs
        nbrs = neighbor_keys(forest, k, δ)
        if isempty(nbrs)
            region = CartesianIndices(ntuple(d -> target_range(N, G, δ[d], 0, false), D))
            push!(boundaries, BoundaryRegion{D}(Int32(b), δ, region))
            continue
        end
        nblevel = level(first(nbrs))
        if nblevel == level(k)
            record!(:copy, δ, zerooffset, 0, find_leaf(forest, only(nbrs)))
        elseif nblevel < level(k)
            # Coarser neighbor: this block's ghosts are prolongated. The
            # stencil geometry depends on where this block sits inside
            # its own parent.
            record!(:prolong, δ, childoffset(k), level(k), find_leaf(forest, only(nbrs)))
        else
            # Finer neighbors: each supplies one part of this block's
            # ghost region, selected by its offset within its parent.
            for nbr in nbrs
                record!(:restrict, δ, childoffset(nbr), 0, find_leaf(forest, nbr))
            end
        end
    end
    return nothing
end

# Concatenate per-task transfer lists in task order, which is block
# order — so the schedule does not depend on how the blocks were split.
function merge_pairs!(into::TransferPairs{D}, from::TransferPairs{D}) where {D}
    for (key, (targets, sources)) in from
        slot = get!(into, key, (Int32[], Int32[]))
        append!(slot[1], targets)
        append!(slot[2], sources)
    end
    return into
end

function GhostSchedule(forest::Forest{D,R}, operators::Operators;
                       T::Type=R) where {D,R}
    check_operators(forest, operators)
    N, G = forest.N, forest.G
    dirs = alldirections(Val(D))
    nb = nleaves(forest)

    # Neighbor finding, threaded over blocks: it reads nothing but the
    # tree, and each task collects into buffers of its own. Those are
    # concatenated in block order, so the schedule that comes out is
    # identical whatever `Threads.nthreads()` happens to be.
    chunks = threadchunks(nb)
    perpairs = [TransferPairs{D}() for _ in chunks]
    perboundaries = [BoundaryRegion{D}[] for _ in chunks]
    threaded_chunks(nb) do c, range
        for b in range
            block_sources!(perpairs[c], perboundaries[c], forest, b, dirs)
        end
    end

    # Merging per-task lists rather than per-block ones keeps the serial
    # tail proportional to the number of *groups*, not to the number of
    # transfers — which is what it costs, since the neighbor search
    # itself threads perfectly (measured in M5).
    pairs = TransferPairs{D}()
    boundaries = BoundaryRegion{D}[]
    for c in eachindex(chunks)
        merge_pairs!(pairs, perpairs[c])
        append!(boundaries, perboundaries[c])
    end

    build(key) =
        key.kind === :copy ?
        ntuple(d -> copy_stencil(T, N, G, key.direction[d]), D) :
        key.kind === :restrict ?
        ntuple(d -> restriction_stencil(T, N, G, key.direction[d], key.offset[d],
                                        operators), D) :
        ntuple(d -> prolongation_stencil(T, N, G, key.direction[d], key.offset[d],
                                         operators), D)

    phase1 = TransferGroup{T,D}[]
    bylevel = Dict{Int,Vector{TransferGroup{T,D}}}()
    for (key, (targets, sources)) in pairs
        group = TransferGroup{T,D}(key.kind, build(key), targets, sources)
        if key.kind === :prolong
            push!(get!(bylevel, key.level, TransferGroup{T,D}[]), group)
        else
            push!(phase1, group)
        end
    end

    levels = sort!(collect(keys(bylevel)))          # coarsest targets first
    phase2 = [bylevel[l] for l in levels]
    return GhostSchedule{T,D,R}(forest, generation(forest), operators, phase1, phase2,
                                levels, boundaries, phase_plan(phase1),
                                [phase_plan(groups) for groups in phase2])
end

function Base.show(io::IO, s::GhostSchedule{T,D}) where {T,D}
    ncopy = sum(ntransfers, filter(g -> g.kind === :copy, s.phase1); init=0)
    nrest = sum(ntransfers, filter(g -> g.kind === :restrict, s.phase1); init=0)
    nprol = sum(gs -> sum(ntransfers, gs; init=0), s.phase2; init=0)
    print(io, "GhostSchedule{", T, ",", D, "}(", ncopy, " copies, ", nrest,
          " restrictions, ", nprol, " prolongations over ", length(s.phase2),
          " level(s), ", length(s.boundaries), " boundary regions)")
end
