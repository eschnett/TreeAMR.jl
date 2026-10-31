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
    Stencil1D{T,VI,MW}

One dimension of a transfer: for each cell of the target's index range,
the first source cell and the `order` weights to apply from there.

`srcstart[i]` and `weights[:, i]` describe target cell
`targetfirst + i - 1`.

`srcstart` and `weights` are read *inside* the transfer kernel, so their
array types are parameters: a schedule built for a device holds them
there (M6). They are always constructed on the host first — the weights
are exact rational arithmetic, which is precisely the kind of work a
device should not be asked to do — and uploaded once, when the schedule
is built. `targetfirst` is a plain `Int` passed by value and stays on
the host.

A stencil of a bounded point-value group (`BOUNDED_POINTVALUE`) stacks
two stencils over the same windows: rows `1:p` are the order-`p`
weights, and rows `p+1:2p` the multilinear ones, nonzero at the bracket
of the target only (see `stack_linear`). Everything that remaps target
rows (mirrors, clamps, packs) moves whole columns, so it carries both;
`run_group!` halves the order.

`unit` says that the stencil is a plain copy along its dimension: one
point, weight exactly one. A group whose stencils are all unit is
launched with a flag that skips them, so that a copy does no stencil
arithmetic (see `copy_load`). It is decided from the weights themselves on the
host, where they are built, and carried to a device with them. It is
not the group's kind: a mirrored copy at a vertex-like high wall derives
the wall plane it does not own by interpolation (`wall_stencil`), and a
vertex-centered `PointValue` restriction is an injection, which is unit.

`single[i]` is the row of column `i`'s only nonzero weight, or zero when
the column has more than one (or none). Along that dimension the CPU's
kernel then takes the one tap instead of all `order` of them (see
`stencil_sum`): in vertex-centered refinement every other fine point
coincides with a coarse node, and its column is an exact one among exact
zeros. Like `unit`, it is decided from the weights on the host and
carried to a device beside them, where the flat launch takes the full sum
instead and does not read it. A stacked bounded point-value stencil
has it zero: its columns hold two sets of weights, so none has a single
nonzero one, and its kernel does not read it.
"""
struct Stencil1D{T,VI<:AbstractVector{Int32},MW<:AbstractMatrix{T}}
    targetfirst::Int
    srcstart::VI
    weights::MW
    unit::Bool
    single::VI
end

Stencil1D{T}(targetfirst::Int, srcstart::AbstractVector{Int32},
             weights::Matrix{T}) where {T} =
    Stencil1D{T}(targetfirst, srcstart, weights,
                 size(weights, 1) == 1 && all(isone, weights), single_rows(weights))
Stencil1D{T}(targetfirst::Int, srcstart::AbstractVector{Int32},
             weights::AbstractMatrix{T}, unit::Bool,
             single::AbstractVector{Int32}) where {T} =
    Stencil1D{T,typeof(srcstart),typeof(weights)}(targetfirst, srcstart, weights, unit,
                                                  single)

# Per column, the row of its only nonzero weight, or zero.
function single_rows(weights::Matrix)
    return Int32[count(!iszero, col) == 1 ? Int32(findfirst(!iszero, col)) : Int32(0)
                 for col in eachcol(weights)]
end

ntarget(s::Stencil1D) = length(s.srcstart)
Base.@propagate_inbounds stencilorder(s::Stencil1D) = size(s.weights, 1)

todevice(backend::Backend, s::Stencil1D{T}) where {T} =
    Stencil1D{T}(s.targetfirst, todevice(backend, s.srcstart),
                 todevice(backend, s.weights), s.unit, todevice(backend, s.single))

"""
    TransferGroup{T,D}

All transfers that share one set of 1D stencils — same kind, same
direction, same child offset, and for prolongations the same target
level — reduced to a list of `(targetblock, sourceblock)` index pairs.

A group is a batch of work, not a unit of scheduling. On the CPU a phase
runs as one parallel loop over threads, and each thread launches the
part of every group whose target blocks it owns (see
[`threadchunks`](@ref TreeAMR.threadchunks)); that is why `targetblocks`
is **non-decreasing**, which every builder guarantees by collecting
transfers in block order, so that a thread's part is one contiguous run.

`factorcol` is nonzero for the mirrored transfers at a reflecting face
(M10) and the rotated ones across a rotating seam (M12): the column of
the field set's factor table the kernel multiplies each variable's
result by. Zero means an ordinary transfer, which the kernel leaves
unscaled.

`orientation` is the number of quarter turns `r` that carry a rotated
transfer's real source to where the target sees it, across the seam of
the forest's `plane`, `(d1, d2)` (M12; see "Rotating seams" in
`CODE.md`); a half-turn seam (M14) has orientation 2 only, and `plane`
is its pair as positive dimensions. Its stencils are built in that virtual frame, and the kernel
reads the real source through the axis map of `r` (see `axismap`) and the
field set's variable table. Zero means an ordinary source, read as it
is. The three small fields share the eight bytes `factorcol` had alone,
so a group is no larger than before M12.

`bounded` is `UNBOUNDED` for every group but one built from bounded
operators whose operator can make a new extremum (`boundedmode`). A
conservative prolongation (`BOUNDED_CONSERVATIVE`) then bounds the
reconstruction instead of summing the stencil; a point-value group
(`BOUNDED_POINTVALUE`) sums it and clamps the result into the range of
the target's bracket (see "Bounded prolongation" and "Bounded point
values" in `CODE.md`). It is the fourth small field in those eight bytes.
"""
struct TransferGroup{T,D,S<:Stencil1D{T},VB<:AbstractVector{Int32}}
    kind::Symbol                      # :copy, :restrict, or :prolong
    stencils::NTuple{D,S}
    targetblocks::VB
    sourceblocks::VB
    factorcol::Int32                  # 0, or the factor-table column
    orientation::Int8                 # 0, or the quarter turns of the source
    plane::NTuple{2,Int8}             # the seam's (d1, d2), or (0, 0)
    bounded::Int8                     # UNBOUNDED, or how the kernel bounds
end

TransferGroup{T,D}(kind::Symbol, stencils::NTuple{D,S},
                   targetblocks::VB, sourceblocks::VB, factorcol::Integer=0,
                   orientation::Integer=0,
                   plane::NTuple{2,Integer}=(0, 0),
                   bounded::Integer=UNBOUNDED) where {T,D,S,VB} =
    TransferGroup{T,D,S,VB}(kind, stencils, targetblocks, sourceblocks,
                            Int32(factorcol), Int8(orientation), Int8.(plane),
                            Int8(bounded))

boxsize(g::TransferGroup{T,D}) where {T,D} = ntuple(d -> ntarget(g.stencils[d]), D)
ntransfers(g::TransferGroup) = length(g.targetblocks)

# The whole group never reaches a kernel — `kind` is a `Symbol`, so the
# struct is not `isbits` — and it does not need to: `run_group!`
# destructures it into the four arrays the kernel actually reads. Only
# those four move.
todevice(backend::Backend, g::TransferGroup{T,D}) where {T,D} =
    TransferGroup{T,D}(g.kind, ntuple(d -> todevice(backend, g.stencils[d]), D),
                       todevice(backend, g.targetblocks),
                       todevice(backend, g.sourceblocks), g.factorcol, g.orientation,
                       g.plane, g.bounded)
# The concrete group type on a given backend, so that a schedule's
# `phase1`/`phase2` are concretely typed even when they are empty (a
# uniform single-root forest has no restrictions and no prolongations).
# Two throwaway allocations, once per schedule.
function grouptype(backend::Backend, ::Type{T}, ::Val{D}) where {T,D}
    VI = typeof(todevice(backend, Int32[0]))
    MW = typeof(todevice(backend, Matrix{T}(undef, 1, 1)))
    return TransferGroup{T,D,Stencil1D{T,VI,MW},VI}
end

"""
    BoundaryRegion{D}

A ghost region of a block that faces outside the domain through an
outer face — neither periodic, reflecting nor outflow — and so is filled
by the user's boundary hook rather than by an inter-block transfer. A
corner or edge region crossing a reflecting or outflow face as well is
one of these when its mirror (or clamped) image still lies outside
through an outer face.

Two more fields serve the relative hook ([`Relative`](@ref), AMR-7),
which sets a ghost from the value at its nearest evolved point. `clamps`
is, per dimension, the stored index that point has along a component of
the direction that leaves the domain through an outer face — `G + N` on
the high side, `G + 1` on the low one, `G + 2` for a set with symmetric
walls — and `0` along every other dimension, where the point keeps the
ghost's own index: across a block face, a reflecting or an outflow face,
in the block's own ghosts, which transfers fill. `slot` is the tag of
the stage after which those ghosts are filled: phase 1, or the
prolongations onto the block's level when they come from a coarser
block. Every other hook ignores both and runs once, after phase 1.
"""
struct BoundaryRegion{D}
    block::Int32
    direction::NTuple{D,Int}
    region::CartesianIndices{D,NTuple{D,UnitRange{Int}}}
    clamps::NTuple{D,Int32}
    slot::Int
end

BoundaryRegion{D}(block, direction, region) where {D} =
    BoundaryRegion{D}(block, direction, region, ntuple(_ -> Int32(0), D), PHASE1_TAG)

"""
    BoundaryBatch{D}

All outward-facing ghost regions of one *shape*, reduced to per-region
arrays — the boundary counterpart of a [`TransferGroup`](@ref
TransferGroup), and for the same reason.

A region's extent is `G` along every nonzero component of `δ` and `N`
along every zero one, so there are only a handful of distinct shapes
however large the domain is. Batching by shape gives every batch a
uniform `ndrange`, which is what lets the cell-wise boundary form
([`CellBoundary`](@ref)) run as a few kernel launches instead of one per
region.

The arrays live wherever the schedule's backend is (M6). A batch also
holds one `slot`, so that the relative hook runs each batch after the
stage that fills what it reads (see [`BoundaryRegion`](@ref
TreeAMR.BoundaryRegion)), and the regions' `clamps`.
"""
struct BoundaryBatch{D,VB<:AbstractVector{Int32},VD<:AbstractVector{NTuple{D,Int8}},
                     VF<:AbstractVector{NTuple{D,Int32}}}
    boxlen::NTuple{D,Int}
    slot::Int
    blocks::VB
    directions::VD
    firsts::VF                        # first stored index of the region
    clamps::VF                        # the nearest evolved point, or 0
end

nregions(b::BoundaryBatch) = length(b.blocks)

todevice(backend::Backend, b::BoundaryBatch{D}) where {D} =
    BoundaryBatch{D}(b.boxlen, b.slot, todevice(backend, b.blocks),
                     todevice(backend, b.directions), todevice(backend, b.firsts),
                     todevice(backend, b.clamps))
BoundaryBatch{D}(boxlen, slot, blocks::VB, directions::VD, firsts::VF,
                 clamps::VF) where {D,VB,VD,VF} =
    BoundaryBatch{D,VB,VD,VF}(boxlen, slot, blocks, directions, firsts, clamps)

"""
    BoundaryPlan{D}

Everything the cell-wise boundary kernel reads that is not field data:
the [`BoundaryBatch`](@ref TreeAMR.BoundaryBatch)es, and the per-block
origin and spacing it turns a stored index into a position with, and for
an element-centered layout (M16) the node offsets, `nothing` otherwise.

The geometry is here rather than recomputed per fill for the same reason
the exchange itself is precomputed: it depends only on the tree, which
is exactly what a schedule is rebuilt for.
"""
struct BoundaryPlan{D,BT<:BoundaryBatch{D},VO<:AbstractVector,VS<:AbstractVector,VN}
    batches::Vector{BT}
    origins::VO
    spacings::VS
    nodes::VN               # `element_offsets` of the layout, on the backend
end

function batchtype(backend::Backend, ::Val{D}) where {D}
    VB = typeof(todevice(backend, Int32[0]))
    VD = typeof(todevice(backend, [ntuple(_ -> Int8(0), D)]))
    VF = typeof(todevice(backend, [ntuple(_ -> Int32(0), D)]))
    return BoundaryBatch{D,VB,VD,VF}
end

# Group the outward-facing regions by the stage after which the relative
# hook may run them and by shape, then move the result to wherever the
# kernels will run.
function BoundaryPlan(backend::Backend, ::Type{T}, forest::Forest{D},
                      boundaries::Vector{BoundaryRegion{D}}, P::Int) where {T,D}
    byshape = Dict{Tuple{Int,NTuple{D,Int}},
                   Tuple{Vector{Int32},Vector{NTuple{D,Int8}},Vector{NTuple{D,Int32}},
                         Vector{NTuple{D,Int32}}}}()
    for r in boundaries
        shape = ntuple(d -> length(r.region.indices[d]), D)
        lists = get!(byshape, (r.slot, shape),
                     (Int32[], NTuple{D,Int8}[], NTuple{D,Int32}[], NTuple{D,Int32}[]))
        push!(lists[1], r.block)
        push!(lists[2], ntuple(d -> Int8(r.direction[d]), D))
        push!(lists[3], ntuple(d -> Int32(first(r.region.indices[d])), D))
        push!(lists[4], r.clamps)
    end
    # Sorted by slot and shape so the batch order is a function of the
    # tree alone, not of dictionary iteration order.
    keys_ = sort!(collect(keys(byshape)))
    # Concretely typed even when empty — a fully periodic domain has no
    # outward-facing regions at all, which is the common case.
    BT = batchtype(backend, Val(D))
    batches = BT[todevice(backend, BoundaryBatch{D}(k[2], k[1], byshape[k]...))
                 for k in keys_]
    origins = todevice(backend, block_origins(forest, T))
    spacings = todevice(backend, block_spacings(forest, T))
    nodes = element_offsets(backend, T, P)
    return BoundaryPlan{D,eltype(batches),typeof(origins),typeof(spacings),
                        typeof(nodes)}(batches, origins, spacings, nodes)
end

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
#
# `mirror` is the mirror state per dimension of a transfer across a
# reflecting or outflow face (M10): 0 for an ordinary dimension, 1 for
# rows mirrored across a reflecting wall, 2 for the derived upper wall
# row of a vertex-like dimension, 3 for rows clamped onto the nearest
# owned row at an outflow face. It changes the stencils, and — through
# the parity factor of states 1 and 2 — what the kernel does with their
# result, so it is part of the key. Everything that is not a mirror
# transfer has all zeros.
#
# `orientation` is the number of quarter turns that carry the source
# across a rotating seam to where the target sees it (M12): it changes
# where the kernel reads and the sign it applies, so it is part of the
# key too. The direction and the offset are the *virtual* ones, as the
# target sees them, since those fix the stencils. Zero off the seam.
#
# `low` marks, per dimension, a target whose owned low wall plane the
# fill writes (a set with `walls = :symmetric`, a vertex-like dimension,
# a block at a low domain face; AMR-8): there the low region is one plane
# longer, `1 : G+1`, and the block's own range one plane shorter,
# `G+2 : G+N`, as the high side always is. It changes the target range,
# so it is part of the key. False everywhere for a half-open set, and
# along a dimension whose `δ` is `+1`, which it does not change.
#
# `shift` moves the partition of a target's seam slab along the flipped
# tangential dimension of a half turn (M14), where that dimension is
# vertex-like: 1 moves every boundary between the regions up one plane,
# so that each region is the image of one real block's owned points, and
# 2 moves all but the upper one, at a block on the high face of `d2`
# whose image there is a symmetric low wall (see `target_range`, and
# *Half turns* in CODE.md). It changes the target range, so it is part of
# the key. Zero everywhere else.
struct GroupKey{D}
    kind::Symbol
    direction::NTuple{D,Int}
    offset::NTuple{D,Int}
    level::Int
    mirror::NTuple{D,Int8}
    orientation::Int8
    low::NTuple{D,Bool}
    shift::NTuple{D,Int8}
end

GroupKey{D}(kind::Symbol, direction::NTuple{D,Int}, offset::NTuple{D,Int},
            level::Int) where {D} =
    GroupKey{D}(kind, direction, offset, level, ntuple(_ -> Int8(0), D), Int8(0))
GroupKey{D}(kind::Symbol, direction::NTuple{D,Int}, offset::NTuple{D,Int},
            level::Int, mirror::NTuple{D,Int8}) where {D} =
    GroupKey{D}(kind, direction, offset, level, mirror, Int8(0))
GroupKey{D}(kind::Symbol, direction::NTuple{D,Int}, offset::NTuple{D,Int},
            level::Int, mirror::NTuple{D,Int8}, orientation::Integer) where {D} =
    GroupKey{D}(kind, direction, offset, level, mirror, Int8(orientation),
                ntuple(_ -> false, D))
GroupKey{D}(kind::Symbol, direction::NTuple{D,Int}, offset::NTuple{D,Int},
            level::Int, mirror::NTuple{D,Int8}, orientation::Integer,
            low::NTuple{D,Bool}) where {D} =
    GroupKey{D}(kind, direction, offset, level, mirror, Int8(orientation), low,
                ntuple(_ -> Int8(0), D))

# A total order over group keys (M7). Within a single rank the order in
# which groups run does not matter, and phase 1's come out of a `Dict`.
# Across ranks it does: both ends of a message lay their buffers out in
# the same order without exchanging a descriptor, so that order has to
# be a function of the keys alone. The kinds are ranked explicitly
# rather than compared as strings, which keeps a comparison free of
# allocation when a layout sorts one entry per remote transfer.
kindrank(kind::Symbol) =
    kind === :copy ? 1 : kind === :restrict ? 2 : kind === :prolong ? 3 :
    throw(ArgumentError("no transfer kind $kind"))
keyorder(k::GroupKey) =
    (kindrank(k.kind), k.direction, k.offset, k.level, k.mirror, k.orientation, k.low,
     k.shift)
Base.isless(a::GroupKey{D}, b::GroupKey{D}) where {D} = isless(keyorder(a), keyorder(b))

# --- Distributed stages (M7) ----------------------------------------------
#
# Over a forest distributed between ranks a transfer whose target and
# source blocks live on different ranks becomes a message, and the
# *sender computes* it: the source's rank evaluates the transfer into a
# flat buffer, and the target's rank copies it into place ("Distributed
# meshes" in CODE.md). Every ordering point of the serial exchange — the
# end of phase 1, each phase-2 target level, each interface face
# dimension — becomes a *stage*, and the types below are one stage as
# one rank holds it.

"""
    LayoutEntry{D}

One transfer of a stage's message buffer, as both of its ends describe
it: the peer rank at the other end, the transfer's group key (the
kind, direction, child offset, target level, mirror state and
orientation that fix its stencils and how it reads), its target and
source as *global* leaf indices, and where it sits in the buffer.

`offset` is the transfer's first point, counted from the start of the
stage buffer, and `npoints` the size of its target box; both are in
points, so the element offset is `offset * nvars` and the transfer
occupies `npoints * nvars` elements, its target box times the
variables, in the transfer kernel's index order. A schedule is built
for a layout and not for a variable count (one schedule serves every
field set of the layout), which is why the factor is applied when the
buffer is addressed rather than stored here.

Both ends derive the entries from the replicated forest — the sender
from its send transfers, the receiver from its receive transfers, which
are the same transfers — in one order: by peer, then by `GroupKey`,
then by global target, then by global source. That is why no layout
descriptor ever travels with the data.
"""
struct LayoutEntry{D}
    peer::Int
    key::GroupKey{D}
    target::Int
    source::Int
    offset::Int
    npoints::Int
end

"""
    RemoteStage{D,GRP,VB,BUF,HB}

The messages of one stage on one rank: what it packs and sends, and what
it receives and unpacks, with the layouts of both buffers.

- `sendpeers` / `recvpeers` are the ranks this one sends to and receives
  from in this stage, ascending, and `sendcounts` / `recvcounts` the
  points of each peer's segment; segments follow each other in peer
  order.
- `sendlayout` / `recvlayout` hold one [`LayoutEntry`](@ref
  TreeAMR.LayoutEntry) per buffer *slot*, in buffer order; `sendoffsets`
  / `recvoffsets` are the same offsets, on the backend, which is what a
  kernel reads.
- `packs` are the send transfers as groups whose target is a slot of the
  send buffer: the stencils of the serial group with every target range
  starting at 1, `targetblocks` the slots and `sourceblocks` the local
  sources, sorted by source block so that the owner of the source runs
  them. They compute the **unscaled** sum, even for a mirrored transfer.
- `unpacks` are width-1, weight-1 transfers from a slot of the receive
  buffer into the local target's box, `sourceblocks` the slots, sorted
  by target block; a mirrored transfer's unpack carries its parity
  column, so the factor is applied here, where the serial kernel
  applies it. (Applied when packing, `0 + 1·x` in the unpack would turn
  the `−0` of a mirrored zero into `+0`; see "Pack and unpack are
  transfers" in CODE.md.)
- `buffers` holds the send and receive buffers per variable count, on
  the backend, taken on first use and kept with the schedule.
- `mirrors` holds their host mirrors, of type `HB` (a `Vector` of the
  element type), per variable count: taken, page-locked for the
  backend, the first time the stage runs over a communicator that cannot
  take the buffers themselves ([`hoststaging`](@ref TreeAMR.hoststaging)).
  That is a device without a device-aware MPI; a CPU buffer is host
  memory already, and MPI is handed it directly.

Both are leased from the forest's buffer pool, which keeps them once the
stage is done with them — the regrid's stage at the end of its call, a
schedule's once the forest has moved on — so that the stages a regrid
builds reuse earlier stages' memory rather than allocating, and
page-locking, their own; see "MPI+GPU" under "Distributed meshes" in
`CODE.md`.
"""
struct RemoteStage{D,GRP<:TransferGroup,VB<:AbstractVector{Int32},BUF<:AbstractVector,
                   HB<:Vector}
    sendpeers::Vector{Int}
    sendcounts::Vector{Int}
    recvpeers::Vector{Int}
    recvcounts::Vector{Int}
    sendlayout::Vector{LayoutEntry{D}}
    recvlayout::Vector{LayoutEntry{D}}
    packs::Vector{GRP}
    unpacks::Vector{GRP}
    sendoffsets::VB
    recvoffsets::VB
    buffers::Dict{Int,Tuple{BUF,BUF}}
    mirrors::Dict{Int,Tuple{HB,HB}}
end

"""
    ExchangeStage{GRP,RS}

One stage of an exchange as this rank runs it: the `tag` that names it
(and matches its messages), the `locals` — the groups with target and
source on this rank, exactly the serial groups of that phase when there
is one rank — and the `remote` part, a [`RemoteStage`](@ref
TreeAMR.RemoteStage), or `nothing` when the stage sends and receives
nothing here, which is every stage of a serial schedule.
"""
struct ExchangeStage{GRP<:TransferGroup,RS<:RemoteStage}
    tag::Int
    locals::Vector{GRP}
    remote::Union{Nothing,RS}
end

hasmessages(s::ExchangeStage) = s.remote !== nothing

# The concrete stage type on a backend, as `grouptype` is for groups.
function stagetype(backend::Backend, ::Type{T}, ::Val{D}) where {T,D}
    GRP = grouptype(backend, T, Val(D))
    VB = typeof(todevice(backend, Int32[0]))
    BUF = typeof(allocate(backend, T, 0))
    return ExchangeStage{GRP,RemoteStage{D,GRP,VB,BUF,Vector{T}}}
end

# The message tags, one per stage, so that a stage's messages can only
# match the same stage on the peer (CODE.md, "Deadlock freedom and
# message matching"). Ascending in the order the stages run, which the
# in-process tests rely on to run every rank's stages in lockstep.
const PHASE1_TAG = 1
prolongation_tag(level::Integer) = 2 + Int(level)       # 2 … 2 + MAX_LEVEL
interface_tag(d::Integer) = 40 + Int(d)                  # one per face dimension
const REGRID_TAG = 50                                     # the regrid transfer

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
- `boundaries` lists the ghost regions facing outside the domain through
  an *outer* face — neither periodic, reflecting nor outflow — filled by
  the user hook between the two phases;
  `boundaryplan` is the same information batched by region shape and
  resident on the backend, which is what the cell-wise hook form
  ([`CellBoundary`](@ref)) is launched over;
- `stages` is the order [`fill_ghosts!`](@ref) runs them in: phase 1,
  then one stage per phase-2 target level, each an
  [`ExchangeStage`](@ref TreeAMR.ExchangeStage). Serially a stage's
  local groups are exactly `phase1` or one entry of `phase2`, and it has
  no messages. Over a forest distributed between ranks (M7), `phase1`,
  `phase2` and `levels` hold the transfers *local* to this rank, in
  local block indices, and a stage also carries the transfers this rank
  computes for another one's ghosts and those it receives — together
  exactly the serial schedule's transfers, split by where their two
  blocks live; see "Distributed meshes" in `CODE.md`.
- `stagegroups` are runs of consecutive `stages` that run as one on this
  rank: phase 1 alone, and the prolongation levels in runs within which
  no level reads or writes a point another one writes here — no
  prolongation reads a ghost of its coarse source that the level below
  prolongated, nor one that the relative hook fills after it. Those
  levels then run together: their messages in flight at once, their
  local groups in one parallel loop, one wait. A level with a rotated
  prolongation runs alone. The decision is the rank's own, from the
  transfers it holds, and is permission rather than obligation: each
  stage keeps its tag and its messages, so ranks may group differently;
  see "Ghost filling" in `CODE.md`. `grouplocals[k]` holds the local
  groups of `stagegroups[k]`, its stages' in order, which is what the
  group runs as one phase.

Periodic boundaries appear nowhere special here: the tree wraps around,
so they are ordinary copies, restrictions, and prolongations. Reflecting
faces (M10) are ordinary transfers too, from mirrored sources: a copy,
restriction or prolongation with its target rows remapped across the
wall, whose result the kernel multiplies by each variable's parity
sign. They sit in the same two phases as every other transfer, so the
hook never sees them; see "Ghost filling" in `CODE.md`. Outflow faces
are the same transfers with the target rows clamped onto the nearest
owned row instead of mirrored, and no sign. So are the
ghosts across a rotating seam (M12): each is a transfer from the real
leaf across the seam, built as though that leaf sat where the block sees
it and read through the quarter turns between the two, which `show`
reports as rotated transfers. A half-turn seam (M14) is the same, two
quarter turns about the centre plane of `d2`, with the seam slab of a
vertex-like `d2` partitioned one plane up. A region whose turned image
leaves through an outer face is the hook's.

A schedule is tied to the forest's leaf array as it was when built. It
must be rebuilt after any refinement, coarsening, or regridding.

    GhostSchedule(fs::FieldSet, operators::Operators)

`operators` is required: interpolation order follows from the
application's discretization, so there is no order the mesh could
sensibly default to. See [`Operators`](@ref).

A schedule is built for one *layout* — one ghost width, one centering,
one element type, one backend — which is exactly what a field set is, so
it takes one (amended in M8, when `G` moved off the forest).
[`fill_ghosts!`](@ref) refuses a field set whose layout differs from the
one the schedule was built for. The forest form
`GhostSchedule(forest, ops; G, centering, T, backend)` spells the four
out instead, for a caller with no field set in hand, and an
element-centered layout (M16) its order `P` as well, which is `0`, no
elements, by default.

`backend` must be the backend of every field set the schedule is
replayed over (M6). The stencil weights and index vectors are read
*inside* the transfer kernel, so they have to live where it runs; they
are built on the host in exact rational arithmetic and uploaded once,
here, rather than at every ghost fill. That is the same argument that
put the exchange in a cached schedule in the first place, applied one
level down.

Over a distributed forest (M7) the build is collective. The ranks
gather a digest of their forests — generation, leaf count, every leaf
and the brick — and of the layout and operators asked for, and a forest
that differs between ranks, or a layout, is refused on every rank
together, saying which ranks differ: every forest mutation must be the
same call on every rank, and over forests that differ the exchange would
deliver the wrong data without noticing. An argument that one rank's
checks refuse is refused on every rank too, so that no rank goes on to
wait in an exchange the others never enter.
"""
struct GhostSchedule{T,D,R,BK<:Backend,GRP<:TransferGroup{T,D},BP<:BoundaryPlan{D},
                     ST<:ExchangeStage{GRP}}
    forest::Forest{D,R}
    generation::Int                              # forest generation it was built for
    G::NTuple{D,Int}                             # ghost width it was built for
    centering::NTuple{D,Symbol}                  # centering it was built for
    P::Int                                       # element order it was built for (M16)
    walls::Symbol                                # walls it was built for (AMR-8)
    operators::Operators
    backend::BK
    phase1::Vector{GRP}
    phase2::Vector{Vector{GRP}}                  # by target level, coarsest first
    levels::Vector{Int}                          # target level of each phase2 entry
    boundaries::Vector{BoundaryRegion{D}}
    boundaryplan::BP
    stages::Vector{ST}                           # phase 1, then phase 2 by level
    stagegroups::Vector{UnitRange{Int}}          # runs of `stages` that run together
    grouplocals::Vector{Vector{GRP}}             # each run's local groups, in stage order

    # The constructor the build calls, which derives `grouplocals`: inside
    # it every type is static, which the build's own body is not.
    function GhostSchedule{T,D,R,BK,GRP,BP,ST}(
        forest, generation, G, centering, P, walls, operators, backend, phase1, phase2,
        levels, boundaries, boundaryplan, stages, stagegroups) where {T,D,R,BK,GRP,BP,ST}
        return new{T,D,R,BK,GRP,BP,ST}(forest, generation, G, centering, P, walls,
                                       operators,
                                       backend, phase1, phase2, levels, boundaries,
                                       boundaryplan, stages, stagegroups,
                                       group_locals(stages, stagegroups))
    end
end

"""
    isstale(schedule::GhostSchedule)
    isstale(schedule::InterfaceSchedule)

Whether the forest has changed since `schedule` was built, in which case
it must be rebuilt before [`fill_ghosts!`](@ref) — or
[`restrict_interfaces!`](@ref) — will accept it.
"""
isstale(s::GhostSchedule) = generation(s.forest) != s.generation

# --- 1D stencil construction ---------------------------------------------
#
# Index conventions, all in *stored* indices, per dimension `d` against
# that dimension's ghost width `G = G[d]` and stagger `c = c[d]` (`1` in
# a vertex-like dimension, `0` in a cell-centered one). The stored extent
# is 1:N+2G+c and the owned range is G+1:G+N whatever the centering:
#
#   δ_d = +1  target is the high exchange region  G+N+1 : N+2G+c
#   δ_d = -1  target is the low  ghost slab           1 : G
#   δ_d =  0  target spans the block's own owned extent
#
# The high region is one plane longer in a vertex-like dimension: that is
# the boundary plane the block *shares* with its high-side neighbor,
# which the neighbor owns and the exchange therefore fills, exactly as it
# fills a ghost. Everything above the owned range is exchange-filled, so
# the asymmetry is bookkeeping in the target ranges and nothing more.
#
# For restriction the tangential extent is halved, since each of the
# 2^(tangential) fine neighbors supplies one half. That split is of the
# *owned* range, so it does not depend on the centering.
#
# Every builder below takes scalar `G` and `c` — the width and stagger in
# *its own* dimension. `target_range` is the single source of truth for
# the ranges; nothing re-derives one.

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
    # Asked for in the window's own frame, which is what makes the
    # answer cacheable across every window of this width in the mesh.
    return unit_lagrange_weights(p, x - lo)
end

# Target range of a transfer in dimension d.
#
# `low` is a target whose owned low wall plane `G + 1` the fill writes
# (`walls = :symmetric`, AMR-8): its low region takes the wall,
# `1 : G+1`, and its own range starts above it, `G+2`, the halved one's
# low half included — the mirror image of the high side, where the wall
# `G + N + 1` is the first plane of the high region.
#
# `closed` extends the block's *own* range (δd = 0) by the shared plane,
# turning the owned range into the closed one and giving the top half of
# a halved range the extra point. The ghost exchange never wants it — a
# block's shared plane is the target of its δ_d = +1 region, not of a
# tangential one — but the interface restriction does: it overwrites the
# block's own closed-range values, and in a vertex-like dimension the
# boundary line of a coarse-fine face is part of what must agree. Keeping
# it here, rather than in a range the interface schedule derives for
# itself, is what keeps this function the single source of truth.
#
# `shift` is a half turn's seam slab along its flipped vertex-like
# tangential dimension (M14): the image of the real blocks' half-open
# partition, every boundary between regions one plane up (1), or all but
# the upper one (2, a block whose image above is a symmetric low wall).
# Then the low region is `1 : G+1`, the block's own range `G+2 : G+N+1`
# (`G+2 : G+N` under 2), its halves meeting between `G+1+N/2` and
# `G+2+N/2`, and the high region `G+N+2 : N+2G+c` (`G+N+1 : …` under 2).
function target_range(N::Int, G::Int, c::Int, δd::Int, od::Int, halved::Bool,
                      closed::Bool=false; low::Bool=false, shift::Integer=0)
    shift != 0 && return shifted_range(N, G, c, δd, od, halved, shift == 1)
    δd == 1 && return (G + N + 1):(N + 2G + c)
    δd == -1 && return 1:(G + low)
    top = closed ? c : 0                            # the shared plane, to the top half
    halved && return (G + 1 + od * (N ÷ 2) + (low && od == 0)):(G + od * (N ÷ 2) + N ÷ 2 +
                                                                 od * top)
    return (G + 1 + low):(G + N + top)
end

# The shifted ranges of `target_range` (M14); `up` moves the upper
# boundary too.
function shifted_range(N::Int, G::Int, c::Int, δd::Int, od::Int, halved::Bool, up::Bool)
    δd == 1 && return (G + N + 1 + up):(N + 2G + c)
    δd == -1 && return 1:(G + 1)
    top = G + N + up
    halved && return (G + 2 + od * (N ÷ 2)):(od == 0 ? G + 1 + N ÷ 2 : top)
    return (G + 2):top
end

# Same-level copy: a pure shift of N cells against the direction, for
# every centering. In a vertex-like dimension the high target is one
# plane longer and its first entry, the shared plane G+N+1, reads the
# neighbor's first *owned* point G+1 — which is the same point. So a copy
# still reads interiors only, which is what makes phase 1 race free.
function copy_stencil(::Type{T}, N::Int, G::Int, c::Int, δd::Int;
                      low::Bool=false, shift::Integer=0) where {T}
    rng = target_range(N, G, c, δd, 0, false; low=low, shift=shift)
    srcstart = Int32[j - N * δd for j in rng]
    return Stencil1D{T}(first(rng), srcstart, ones(T, 1, length(rng)))
end

# Restriction, fine -> coarse. `od` is the source's child offset within
# the (refined) node adjacent to the target, so it selects which half of
# the target's extent this source covers.
#
# In a vertex-like dimension restriction is **injection**: the coarse
# point at position `q` coincides with fine point `2q`, so the stencil is
# width one with weight one. It is exact for any data, carries no order,
# and never has to shift — the circularity that forces the cell-centered
# window inward cannot arise, because the coincident fine point is always
# owned by the fine block (that is the `N ≥ 2G + 2` invariant).
restrict_stencil(::Type{T}, N::Int, G::Int, c::Int, δd::Int, od::Int,
                 p::Int; low::Bool=false, shift::Integer=0) where {T} =
    restrict_stencil_over(T, target_range(N, G, c, δd, od, true; low=low, shift=shift),
                          N, G, c, δd, od, p)

# The body, over an explicit target range. The interface restriction
# (M8b) is the same transfer over a different range — one plane in the
# face dimension, the closed range tangentially — so it calls this rather
# than repeating the arithmetic that maps a target point to its source.
function restrict_stencil_over(::Type{T}, rng::UnitRange{Int}, N::Int, G::Int, c::Int,
                               δd::Int, od::Int, p::Int) where {T}
    # Step into the adjacent node's frame, where the source's parent has
    # the target block's own layout.
    shift = -N * δd
    if c == 1
        srcstart = Int32[G + 1 + 2 * (j + shift - G - 1 - od * (N ÷ 2)) for j in rng]
        return Stencil1D{T}(first(rng), srcstart, ones(T, 1, length(rng)))
    end
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
function prolong_stencil(::Type{T}, N::Int, G::Int, c::Int, δd::Int, od::Int,
                         p::Int; low::Bool=false, shift::Integer=0) where {T}
    rng = target_range(N, G, c, δd, od, false; low=low, shift=shift)
    stored = N + 2G + c
    srcstart = Vector{Int32}(undef, length(rng))
    weights = Matrix{T}(undef, p, length(rng))
    for (i, j) in enumerate(rng)
        # Fine offset from the parent's interior start, so that coarse
        # cell c covers fine cells 2c and 2c+1 (cell-centered) or coarse
        # point c coincides with fine point 2c (vertex-like).
        φ = od * N + j - G - 1
        # Translating into the source's stored frame costs N cells per
        # unit of direction.
        origin = -N * source_direction(δd, od) + G + 1
        if c == 1
            # Fine point φ sits at coarse coordinate φ/2: an integer for
            # even φ, where the Lagrange weights through any window
            # containing that node are the unit vector *exactly* (they
            # are built in rational arithmetic), and a half-integer for
            # odd φ, where a symmetric window of even width p applies.
            # One builder serves both.
            lo = clamp(fld(φ, 2) - p ÷ 2 + 1 + origin, 1, stored - p + 1)
            x = φ//2 + origin
        else
            # Fine cell φ sits at coarse coordinate φ/2 - 1/4; center a
            # window of p coarse cells on it.
            lo = clamp(cld(φ, 2) - p ÷ 2 + origin, 1, stored - p + 1)
            x = φ//2 - 1//4 + origin
        end
        srcstart[i] = lo
        weights[:, i] = interpolation_weights(lo, p, x, "prolongation")
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
function build_conservative_prolong_weights(p::Int)
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

# Memoised on `p`, for the same reason as the Lagrange weights it is
# built from: a schedule rebuild asks for the same order over and over,
# and a regridding run rebuilds at every regrid. The two vectors are
# shared, so nothing may mutate one — the one caller copies them into a
# `Stencil1D`'s element type.
const CONSERVATIVE_CACHE =
    Dict{Int,Tuple{Vector{WeightRational},Vector{WeightRational}}}()

conservative_prolong_weights(p::Int) =
    lock(LAGRANGE_LOCK) do
        get!(() -> build_conservative_prolong_weights(p), CONSERVATIVE_CACHE, p)
    end

function conservative_prolong_stencil(::Type{T}, N::Int, G::Int, c::Int, δd::Int,
                                      od::Int, p::Int) where {T}
    # Only ever called along a cell-like dimension (`prolongation_stencil`
    # takes point values along a stagger), so `c` only reaches
    # `target_range`.
    rng = target_range(N, G, c, δd, od, false)
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
# A conservative staggered set (AMR-9) is a tensor product of the two
# families: the conservative reconstruction along its cell-like
# dimensions, point values of width `vertex_prolongation` along its
# vertex-like ones.
# `low` and `shift` only ever reach a vertex-like dimension (see
# `target_range`).
prolongation_stencil(::Type{T}, N, G, c, δd, od, ops::Operators;
                     low::Bool=false, shift::Integer=0) where {T} =
    ops.family === Conservative && c == 0 ?
    conservative_prolong_stencil(T, N, G, c, δd, od, ops.prolongation) :
    prolong_stencil(T, N, G, c, δd, od, vertex_order(ops); low=low, shift=shift)

# Conservative restriction is the exact volume average, which the
# point-value builder already produces at order 2: its window is a
# cell's own two children and its weights are 1/2.
restriction_stencil(::Type{T}, N, G, c, δd, od, ops::Operators;
                    low::Bool=false, shift::Integer=0) where {T} =
    restrict_stencil(T, N, G, c, δd, od, ops.restriction; low=low, shift=shift)

# One dimension of an unmirrored transfer of `kind`, as the ghost schedule
# and the regrid transfer both build it. With `mode` the group's bound
# (`boundedmode`), a bounded point-value group's stencil carries the
# multilinear weights below its own (`stack_linear`).
# `low` trims the target range for a symmetric wall, and `shift` moves it
# in a half turn's seam slab (see `target_range`).
function kind_stencil(::Type{T}, N, G, c, kind::Symbol, δd, od, ops::Operators;
                      low::Bool=false, shift::Integer=0) where {T}
    kind === :copy && return copy_stencil(T, N, G, c, δd; low=low, shift=shift)
    kind === :restrict &&
        return restriction_stencil(T, N, G, c, δd, od, ops; low=low, shift=shift)
    return prolongation_stencil(T, N, G, c, δd, od, ops; low=low, shift=shift)
end

function kind_stencil(::Type{T}, N, G, c, kind::Symbol, δd, od, ops::Operators,
                      mode::Int8; low::Bool=false, shift::Integer=0) where {T}
    s = kind_stencil(T, N, G, c, kind, δd, od, ops; low=low, shift=shift)
    mode == BOUNDED_POINTVALUE || return s
    return stack_linear(s, kind_stencil(T, N, G, c, kind, δd, od, LINEAR_OPERATORS;
                                        low=low, shift=shift))
end

# Multilinear interpolation: the point-value operators of width 2, whose
# window is the target's bracket — the two source points either side of
# it, or the one it coincides with.
const LINEAR_OPERATORS = Operators(; prolongation=2, restriction=2)

# A bounded point-value stencil (AMR-10; "Bounded point values" in
# CODE.md): `s`'s order-`p` weights in rows `1:p`, and below them, in rows
# `p+1:2p`, the weights of the linear stencil `lin` over the same targets,
# placed at their points' positions within `s`'s window. The kernel takes
# the range of the source points whose linear weight is nonzero — the
# bracket — and clamps the order-`p` sum into it.
#
# The bracket lies inside every window: a window is `p` consecutive
# integer nodes and contains its target, so it contains the integers
# either side of it. A zero linear weight outside the window (a target on
# a node, whose width-2 window may stick out past it) is dropped; a
# nonzero one there is a bug in the builders.
function stack_linear(s::Stencil1D{T}, lin::Stencil1D{T}) where {T}
    p, q, n = stencilorder(s), stencilorder(lin), ntarget(s)
    s.targetfirst == lin.targetfirst && ntarget(lin) == n || error(
        "the linear stencil covers other targets than the order-$p one")
    weights = zeros(T, 2p, n)
    weights[1:p, :] = s.weights
    for i in 1:n, k in 1:q
        w = lin.weights[k, i]
        iszero(w) && continue
        off = Int(lin.srcstart[i]) + k - 1 - Int(s.srcstart[i])
        0 <= off < p || error(
            "the bracket of target $(s.targetfirst + i - 1) lies outside its " *
            "order-$p window")
        weights[p + 1 + off, i] = w
    end
    return Stencil1D{T}(s.targetfirst, s.srcstart, weights, false, zeros(Int32, n))
end

# --- Mirrored stencils (M10) ---------------------------------------------
#
# A ghost row `j` beyond a reflecting wall holds the value at its mirror
# point `j*` inside the block, which is what the ordinary tangential
# transfer writes at `j*`. So the mirrored stencil is that tangential
# stencil with its *target rows* remapped: the source windows and the
# weights are untouched, and the kernel does not change. The parity sign
# is applied by the kernel, per variable, not here.
#
#   low wall   j* = 2G + 1 + c − j
#   high wall  j* = 2(G + N) + 1 + c − j
#
# with `c = 1` along a vertex-like dimension, whose wall is a stored
# point, and `0` along a cell-centered one, whose wall lies between two.

# --- Clamped stencils (outflow faces) -------------------------------------
#
# A ghost row `j` beyond an outflow face holds the value at the nearest
# owned row — zero-gradient extrapolation — which is what the ordinary
# tangential transfer writes there. So, as for a mirror, the clamped
# stencil is the tangential stencil with its target rows remapped:
#
#   low face   j* = G + 1
#   high face  j* = G + N
#
# over the whole ghost slab, which on the high side of a vertex-like
# dimension includes the wall row `G + N + 1` that no block owns: it is
# a ghost like the rows beyond it, and clamped like them. With a
# symmetric low wall (`low`, AMR-8) the low side is the mirror of that:
# the slab includes the owned wall row `G + 1`, and the nearest evolved
# row is `G + 2`.
clamp_range(N::Int, G::Int, c::Int, δd::Int; low::Bool=false) =
    δd < 0 ? (1:(G + low)) : ((G + N + 1):(N + 2G + c))

function clamp_rows(s::Stencil1D{T}, N::Int, G::Int, c::Int, δd::Int;
                    low::Bool=false) where {T}
    rng = clamp_range(N, G, c, δd; low=low)
    row = (δd < 0 ? G + 1 + low : G + N) - s.targetfirst + 1
    # The nearest owned row is on the wall side of the owned range, so
    # it lies inside whatever range the tangential stencil covers, halved
    # or not, as a mirror point does. A failure is a bug in the schedule.
    1 <= row <= ntarget(s) || error(
        "clamped row $row falls outside the tangential stencil's $(ntarget(s)) " *
        "targets (N=$N, G=$G, c=$c, δ=$δd)")
    rows = fill(row, length(rng))
    return Stencil1D{T}(first(rng), s.srcstart[rows], s.weights[:, rows])
end

# Target range of the mirrored rows: the whole ghost slab, less the wall
# row itself on the high side of a vertex-like dimension (that row is
# `wall_stencil`'s).
mirror_range(N::Int, G::Int, c::Int, δd::Int) =
    δd < 0 ? (1:G) : ((G + N + 1 + c):(N + 2G + c))

mirror_index(N::Int, G::Int, c::Int, δd::Int, j::Int) =
    δd < 0 ? 2G + 1 + c - j : 2(G + N) + 1 + c - j

function mirror_rows(s::Stencil1D{T}, N::Int, G::Int, c::Int, δd::Int) where {T}
    rng = mirror_range(N, G, c, δd)
    rows = Int[mirror_index(N, G, c, δd, j) - s.targetfirst + 1 for j in rng]
    # Every mirror point lies inside the owned range, within the half on
    # the wall side — `N ≥ 2G + 2c` is exactly what guarantees it — and
    # so inside whatever range the tangential stencil covers, halved or
    # not. A failure here is a bug in the schedule, not in the input.
    all(r -> 1 <= r <= ntarget(s), rows) || error(
        "mirror rows $rows fall outside the tangential stencil's " *
        "$(ntarget(s)) targets (N=$N, G=$G, c=$c, δ=$δd)")
    return Stencil1D{T}(first(rng), s.srcstart[rows], s.weights[:, rows])
end

# The derived upper wall row of a vertex-like dimension at a reflecting
# face: the symmetric Lagrange interpolant at the wall through the `p`
# points `±1, …, ±p/2` beside it, folded by symmetry onto the `p/2`
# one-sided points with doubled weights — `(4u₁ − u₂)/3` at `p = 4`. The
# parity factor is `(1 + σ)/2`, so an odd variable gets exactly zero.
# Every source has its own wall at `G + N + 1` in its own frame, which
# is why the row is the same whatever kind the transfer is. See "Ghost
# filling" in CODE.md.
#
# `low` is the derived low wall row of a set with symmetric walls
# (AMR-8): the wall `G + 1` from the points `G + 2 … G + 1 + h` above it,
# the same weights in mirrored order. A source at a low domain face has
# its own wall at `G + 1` in its own frame, as every source has its high
# one at `G + N + 1`.
function wall_stencil(::Type{T}, N::Int, G::Int, p::Int; low::Bool=false) where {T}
    h = p ÷ 2
    nodes = [Rational{Int}(i) for i in vcat(-h:-1, 1:h)]
    w = lagrange_weights(nodes, 0//1)
    weights = Matrix{T}(undef, h, 1)
    for i in 1:h
        # Nodes -h … -1 ascending for the high wall; +1 … +h for the low
        # one, whose weights are the same by symmetry.
        weights[i, 1] = T(2 * (low ? w[h + i] : w[i]))
    end
    low && return Stencil1D{T}(G + 1, Int32[G + 2], weights)
    wall = G + N + 1
    return Stencil1D{T}(wall, Int32[wall - h], weights)
end

# --- Schedule construction -----------------------------------------------

# A block's offset within its parent, per dimension.
childoffset(k::MortonKey{D}) where {D} = ntuple(d -> Int(k.coords[d]) & 1, D)

# The per-key transfer lists a schedule is assembled from: for each
# group key, the target blocks and the source blocks of its transfers,
# in the order the blocks were walked.
const TransferPairs{D} = Dict{GroupKey{D},Tuple{Vector{Int32},Vector{Int32}}}

# The neighbor search for a single block: every transfer its ghosts
# need, appended to `pairs`, plus the regions that face out of the
# domain and so belong to the boundary hook instead. Depends on the tree
# alone, which is what makes it the part that threads — each task owns
# its own `pairs` and `boundaries`.
#
# Across a rotating seam (M12) the search returns the real leaves with
# their orientation `r`, and a transfer is recorded as the target sees
# it, in the *virtual* frame: a finer source's child offset is turned
# into its virtual one, and `r` goes into the key, which the kernel reads
# the real source by. A region whose image leaves through an outer face
# is the hook's, as any other such region is.
function block_sources!(pairs::TransferPairs{D},
                        boundaries::Vector{BoundaryRegion{D}},
                        forest::Forest{D}, G::NTuple{D,Int}, c::NTuple{D,Int},
                        b::Int, dirs, symmetric::Bool=false) where {D}
    k = forest.leaves[b]
    N = forest.N
    zerooffset = ntuple(_ -> 0, D)
    walls = lowwalls(forest, k, c, symmetric)
    for δ in dirs
        # Where this block's low wall plane is written (AMR-8), along every
        # dimension that does not step up.
        low = ntuple(d -> walls[d] && δ[d] != 1, D)
        shift = seam_shift(forest, k, c, δ, symmetric)
        record!(kind, offset, lvl, s, r) =
            push!.(get!(pairs, GroupKey{D}(kind, δ, offset, lvl, ntuple(_ -> Int8(0), D),
                                           Int8(r), low, shift),
                        (Int32[], Int32[])),
                   (Int32(b), Int32(s)))
        region = CartesianIndices(ntuple(d -> target_range(N, G[d], c[d], δ[d], 0, false;
                                                           low=low[d], shift=shift[d]),
                                         D))
        # A cell-centered dimension with `G_d = 0` has no low slab and no
        # high one, so every direction that leaves the block along it has
        # nothing to fill. Skipping here rather than launching an empty
        # kernel also spares the tree query. (Halving a tangential range
        # cannot empty it, so testing the unhalved region covers every
        # kind.)
        isempty(region) && continue
        δ′, faces = wall_direction(forest, k, δ)
        if any(!=(OuterFace), faces)
            mirror_sources!(pairs, boundaries, forest, G, c, b, δ, δ′, faces, region, low,
                            shift)
            continue
        end
        r, nbrs = oriented_neighbors(forest, k, δ)
        if isempty(nbrs)
            push!(boundaries, boundary_region(forest, k, b, δ, region, G, low, shift))
            continue
        end
        nblevel = level(first(nbrs))
        if nblevel == level(k)
            record!(:copy, zerooffset, 0, find_leaf(forest, only(nbrs)), r)
        elseif nblevel < level(k)
            # Coarser neighbor: this block's ghosts are prolongated. The
            # stencil geometry depends on where this block sits inside
            # its own parent.
            record!(:prolong, childoffset(k), level(k), find_leaf(forest, only(nbrs)), r)
        else
            # Finer neighbors: each supplies one part of this block's
            # ghost region, selected by its offset within its parent —
            # its virtual parent, where this block sees it.
            for nbr in nbrs
                record!(:restrict, seam_offset(forest, nbr, r), 0, find_leaf(forest, nbr),
                        r)
            end
        end
    end
    return nothing
end

# A hook region of block `b` (key `k`) in direction `δ`, with what the
# relative hook needs (see `BoundaryRegion`). A component of `δ` leaves
# through an outer face when the single step along it does: there is no
# block across, and the face is not reflecting or outflow. Along such a
# component the nearest evolved point is clamped into the block; along
# every other one it keeps the ghost's index, in the block's own ghost
# region in the remaining direction `δ″`, which transfers fill — at
# phase 1, unless they are prolongations from a coarser block, which
# land with the stage of the block's level. Should `δ″` find no source
# at all (a corner whose turned image leaves through an outer face), the
# point is clamped along every component instead, so that the hook
# never reads another hook region.
function boundary_region(forest::Forest{D}, k::MortonKey{D}, b::Int, δ::NTuple{D,Int},
                         region, G::NTuple{D,Int}, low::NTuple{D,Bool},
                         shift::NTuple{D,Int8}=ntuple(_ -> Int8(0), D)) where {D}
    N = forest.N
    exits = ntuple(D) do d
        δ[d] == 0 && return false
        step = ntuple(e -> e == d ? δ[d] : 0, D)
        _, faces = wall_direction(forest, k, step)
        return faces[d] == OuterFace && isempty(oriented_neighbors(forest, k, step)[2])
    end
    δ″ = ntuple(d -> exits[d] ? 0 : δ[d], D)
    slot = PHASE1_TAG
    if any(!=(0), δ″)
        δ‴, _ = wall_direction(forest, k, δ″)
        if any(!=(0), δ‴)
            nbrs = oriented_neighbors(forest, k, δ‴)[2]
            if isempty(nbrs)
                exits = map(!=(0), δ)
            elseif level(first(nbrs)) < level(k)
                slot = prolongation_tag(level(k))
            end
        end
    end
    # The nearest evolved point, where a half turn's shifted slab (M14)
    # puts it: one plane up on either side under a full shift.
    clamps = ntuple(D) do d
        exits[d] || return Int32(0)
        return Int32(δ[d] > 0 ? G[d] + N + (shift[d] == 1) :
                     G[d] + 1 + (low[d] || shift[d] != 0))
    end
    return BoundaryRegion{D}(Int32(b), δ, region, clamps, slot)
end

# Along which dimensions the fill writes block `k`'s owned low wall
# plane (AMR-8): with symmetric walls, along every vertex-like dimension
# whose low face is a face of the domain — one with no block across it.
# That is an outer, reflecting or outflow face, never a periodic one nor
# a rotating seam, both of which have neighbours.
function lowwalls(forest::Forest{D}, k::MortonKey{D}, c::NTuple{D,Int},
                  symmetric::Bool) where {D}
    return ntuple(D) do d
        symmetric && c[d] == 1 || return false
        δ = ntuple(e -> e == d ? -1 : 0, D)
        return isempty(oriented_neighbors(forest, k, δ)[2])
    end
end

# The shift of block `k`'s target ranges in direction `δ` (M14; see
# `GroupKey`): nonzero only across a half turn's seam — `δ[d1] = -1` from
# a block on the low face of `d1` — and along its flipped dimension `d2`
# where that is vertex-like. There the
# partition is the image of the real blocks', every boundary one plane
# up (1), except under symmetric walls on a block at the high face of
# `d2`, whose upper boundary's image is a low wall the fill writes (2).
function seam_shift(forest::Forest{D}, k::MortonKey{D}, c::NTuple{D,Int},
                    δ::NTuple{D,Int}, symmetric::Bool) where {D}
    zero = ntuple(_ -> Int8(0), D)
    ishalfturn(forest) || return zero
    d1, d2 = seam_plane(forest)
    δ[d1] == -1 && c[d2] == 1 || return zero
    lvl = level(k)
    pos = root_position(forest, k.root)
    (pos[d1] << lvl) + Int(k.coords[d1]) == 0 || return zero
    g2 = (pos[d2] << lvl) + Int(k.coords[d2])
    top = g2 == (forest.roots[d2] << lvl) - 1
    return Base.setindex(zero, Int8(symmetric && top ? 2 : 1), d2)
end

# The child offset of a finer source as the target sees it: its own
# offset off the seam, and across it in orientation `r` the offset turned
# into the virtual frame (M12).
seam_offset(forest::Forest, nbr::MortonKey, r::Integer) =
    r == 0 ? childoffset(nbr) :
    virtual_offset(childoffset(nbr), r, seam_plane(forest)...)

# The sources of a ghost region that crosses a reflecting or an outflow
# face (M10; see "Ghost filling" in CODE.md). Mirrored (or clamped)
# along the walls' dimensions, the region lies inside the domain: in the
# block itself when `δ′` is zero, and otherwise in the node adjacent to
# it in direction `δ′`, at the block's own extent along every wall
# dimension. So the source is whatever the ordinary search finds in
# `δ′`, and the transfer is recorded under the original `δ`, which fixes
# the target range. A `δ′` that leaves through an outer face makes the
# region the hook's.
function mirror_sources!(pairs::TransferPairs{D},
                         boundaries::Vector{BoundaryRegion{D}},
                         forest::Forest{D}, G::NTuple{D,Int}, c::NTuple{D,Int},
                         b::Int, δ::NTuple{D,Int}, δ′::NTuple{D,Int},
                         walls::NTuple{D,FaceKind}, region,
                         low::NTuple{D,Bool}=ntuple(_ -> false, D),
                         shift::NTuple{D,Int8}=ntuple(_ -> Int8(0), D)) where {D}
    k = forest.leaves[b]
    zerooffset = ntuple(_ -> 0, D)
    mask = map(!=(OuterFace), walls)
    # A reflecting vertex-like dimension on the high side splits its
    # target into the derived wall row (state 2) and the mirrored rows
    # beyond it (state 1), which are batched apart because their parity
    # factors differ. So does the low side of a set with symmetric walls
    # (`low`, AMR-8), whose owned wall row is derived the same way. Each
    # part may be empty — at `G = 0` only the wall row is left — and an
    # empty part records nothing. An outflow dimension clamps its whole
    # slab, wall row included (state 3).
    choices = ntuple(D) do d
        !mask[d] && return (Int8(0),)
        walls[d] == OutflowFace && return (Int8(3),)
        c[d] == 1 && (δ[d] == 1 || (δ[d] == -1 && low[d])) || return (Int8(1),)
        return G[d] == 0 ? (Int8(2),) : (Int8(1), Int8(2))
    end
    states = vec(collect(Iterators.product(choices...)))
    record!(kind, offset, lvl, s, r) =
        for state in states
            push!.(get!(pairs, GroupKey{D}(kind, δ, offset, lvl, state, Int8(r), low,
                                           shift),
                        (Int32[], Int32[])),
                   (Int32(b), Int32(s)))
        end

    if δ′ == zerooffset
        record!(:copy, zerooffset, 0, b, 0)
        return nothing
    end
    # `δ′` may still cross a rotating seam (M12), the mirror being outside
    # its plane: the two compose, the source turned and then mirrored.
    r, nbrs = oriented_neighbors(forest, k, δ′)
    if isempty(nbrs)
        push!(boundaries, boundary_region(forest, k, b, δ, region, G, low, shift))
        return nothing
    end
    nblevel = level(first(nbrs))
    if nblevel == level(k)
        record!(:copy, zerooffset, 0, find_leaf(forest, only(nbrs)), r)
    elseif nblevel < level(k)
        record!(:prolong, childoffset(k), level(k), find_leaf(forest, only(nbrs)), r)
    else
        # Only the children on the wall side along every masked dimension
        # cover the mirror image; the others supply the half of the
        # tangential extent it does not reach. The side is the virtual
        # one, where the target sees the child.
        wallside = ntuple(d -> δ[d] < 0 ? 0 : 1, D)
        for nbr in nbrs
            o = seam_offset(forest, nbr, r)
            all(d -> !mask[d] || o[d] == wallside[d], 1:D) || continue
            record!(:restrict, o, 0, find_leaf(forest, nbr), r)
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

# Split the transfers `pairs`, collected for this rank's own targets in
# global leaf indices, by where their source lives (M7). Those whose
# source is on this rank as well stay in `pairs`, shifted to local block
# indices: the *local* class, today's groups. Those whose source is on
# another rank are the *recv* class, and are returned, still in global
# indices, for the stage builder. Serially `range` is every leaf, so
# nothing is returned and `pairs` is untouched.
function split_received!(pairs::TransferPairs{D}, range::UnitRange{Int},
                         n::Int) where {D}
    received = TransferPairs{D}()
    range == 1:n && return received
    offset = Int32(first(range) - 1)
    for key in collect(keys(pairs))
        targets, sources = pairs[key]
        here = [i for i in eachindex(targets) if sources[i] in range]
        there = [i for i in eachindex(targets) if !(sources[i] in range)]
        isempty(there) || (received[key] = (targets[there], sources[there]))
        if isempty(here)
            delete!(pairs, key)
        else
            pairs[key] = (targets[here] .- offset, sources[here] .- offset)
        end
    end
    return received
end

# The *send* class (M7): the transfers this rank computes for another
# rank's targets. `sources!(pairs, j)` is the neighbor search a builder
# runs for its own targets — `block_sources!`, `interface_sources!` —
# and it is run here for every *candidate remote target* `j`, a leaf of
# another rank that touches one of this rank's (`remote_neighbors`),
# keeping the transfers whose source is on this rank. Adjacency is
# mutually discoverable, so the candidates miss no target, and the
# search is the target owner's own, so the sender finds exactly the
# transfers the receiver does. Threaded and merged in candidate order
# like the local search, hence a function of the tree alone. Serially
# there are no candidates and nothing is searched.
function sent_transfers(sources!, forest::Forest{D}, range::UnitRange{Int}) where {D}
    sent = TransferPairs{D}()
    candidates = remote_neighbors(forest, range)
    isempty(candidates) && return sent
    chunks = threadchunks(length(candidates))
    perpairs = [TransferPairs{D}() for _ in chunks]
    threaded_chunks(length(candidates)) do c, part
        for i in part
            sources!(perpairs[c], candidates[i])
        end
    end
    for c in eachindex(chunks)
        merge_pairs!(sent, perpairs[c])
    end
    for key in collect(keys(sent))
        targets, sources = sent[key]
        keep = [i for i in eachindex(targets) if sources[i] in range]
        if isempty(keep)
            delete!(sent, key)
        else
            sent[key] = (targets[keep], sources[keep])
        end
    end
    return sent
end

# The transfers of `pairs` filed by the stage they belong to, `stageof(key)`.
function bystage(stageof, pairs::TransferPairs{D}) where {D}
    out = Dict{Int,TransferPairs{D}}()
    for (key, lists) in pairs
        get!(TransferPairs{D}, out, stageof(key))[key] = lists
    end
    return out
end

# One buffer's layout: every transfer of `pairs` as a `LayoutEntry`, in
# the order both ends derive — peer, `GroupKey`, global target, global
# source — with the offsets accumulated in that order. `peerof(t, s)` is
# the rank at the other end of the transfer from target `t` to source
# `s`, and `npoints(key)` the size of the group's target box.
function stage_layout(pairs::TransferPairs{D}, peerof, npoints) where {D}
    entries = Tuple{Int,GroupKey{D},Int,Int}[]
    for (key, (targets, sources)) in pairs, i in eachindex(targets)
        t, s = Int(targets[i]), Int(sources[i])
        push!(entries, (peerof(t, s), key, t, s))
    end
    sort!(entries)
    layout = Vector{LayoutEntry{D}}(undef, length(entries))
    offset = 0
    for (i, (peer, key, t, s)) in enumerate(entries)
        n = npoints(key)
        layout[i] = LayoutEntry{D}(peer, key, t, s, offset, n)
        offset += n
    end
    offset <= typemax(Int32) || throw(ArgumentError(
        "a stage buffer of $offset points does not fit the Int32 offsets the " *
        "transfer kernel reads; this rank exchanges far more ghost data than a " *
        "block decomposition should"))
    return layout
end

# The peers of a layout, ascending, and the points of each one's segment.
function layout_segments(layout::Vector{<:LayoutEntry})
    peers, counts = Int[], Int[]
    for e in layout
        if isempty(peers) || last(peers) != e.peer
            push!(peers, e.peer)
            push!(counts, 0)
        end
        counts[end] += e.npoints
    end
    return peers, counts
end

# The slots of a layout, by group key.
function slots_by_key(layout::Vector{LayoutEntry{D}}) where {D}
    out = Dict{GroupKey{D},Vector{Int}}()
    for (slot, e) in enumerate(layout)
        push!(get!(Vector{Int}, out, e.key), slot)
    end
    return out
end

# A stage's messages on this rank (M7), from the transfers it sends and
# those it receives, both in global leaf indices.
#
# `stencils(key)` builds a group's host stencils and `factorcol(key)`
# its parity column, exactly as the builder does for its local groups,
# so a pack evaluates the serial transfer. `plane` is the forest's
# rotating seam (M12): a rotated pack reads its source through the axis
# map and the variable table, as the serial group does, and computes the
# unscaled sum; its unpack applies the sign through the factor column, as
# it applies a parity. `targetowner(t)` and
# `sourceowner(s)` are the ranks of a global target and source, and
# `targetrange` / `sourcerange` this rank's own leaves in the partitions
# the targets and the sources are local to. For the ghost and interface
# exchanges both are `blockrange(forest)`; the regrid transfer (`regrid_stage`)
# has its targets in the new partition and its sources in the old.
# The bound of a group with no bound to apply (see `boundedmode`).
nobound(key) = UNBOUNDED

# `bounded(key)` is the bound of a pack's group (`boundedmode`): the
# sender computes, so it is the pack that bounds, and the unpack stays a
# copy.
# Returns `nothing` when the stage has no messages here.
function remote_stage(::Type{RS}, ::Type{T}, backend::Backend,
                      sent::TransferPairs{D}, received::TransferPairs{D}, stencils,
                      factorcol; targetowner, sourceowner, targetrange::UnitRange{Int},
                      sourcerange::UnitRange{Int},
                      plane::NTuple{2,Int8}=(Int8(0), Int8(0)),
                      @nospecialize(bounded=nobound)) where {RS<:RemoteStage,T,D}
    isempty(sent) && isempty(received) && return nothing
    built = Dict{GroupKey{D},Any}()
    host(key) = get!(() -> stencils(key), built, key)
    npoints(key) = prod(ntarget, host(key))
    sendlayout = stage_layout(sent, (t, s) -> targetowner(t), npoints)
    recvlayout = stage_layout(received, (t, s) -> sourceowner(s), npoints)
    sendpeers, sendcounts = layout_segments(sendlayout)
    recvpeers, recvcounts = layout_segments(recvlayout)

    GRP = eltype(fieldtype(RS, :packs))
    # A pack is the serial transfer with its target box moved to the
    # start of a buffer slot: the same source windows, the same weights,
    # summed in the same order, read through the same rotation, and no
    # factor. Sorted by source block, so that on the CPU the owner of the
    # source runs it.
    packs = GRP[]
    sendslots = slots_by_key(sendlayout)
    for key in sort!(collect(keys(sendslots)))
        order = sort!([(sendlayout[slot].source - first(sourcerange) + 1, slot)
                       for slot in sendslots[key]])
        st = map(s -> Stencil1D{T}(1, s.srcstart, s.weights, s.unit, s.single),
                 host(key))
        push!(packs, todevice(backend, TransferGroup{T,D}(
            key.kind, st, Int32[o[2] for o in order], Int32[o[1] for o in order], 0,
            key.orientation, plane, bounded(key)::Int8)))
    end
    # An unpack is a width-1, weight-1 transfer from a slot into the
    # target box the serial group writes, carrying its parity column.
    # Sorted by target block, so that the owner of the target runs it.
    unpacks = GRP[]
    recvslots = slots_by_key(recvlayout)
    for key in sort!(collect(keys(recvslots)))
        order = sort!([(recvlayout[slot].target - first(targetrange) + 1, slot)
                       for slot in recvslots[key]])
        st = map(host(key)) do s
            n = ntarget(s)
            Stencil1D{T}(s.targetfirst, Int32.(1:n), ones(T, 1, n))
        end
        push!(unpacks, todevice(backend, TransferGroup{T,D}(
            :copy, st, Int32[o[1] for o in order], Int32[o[2] for o in order],
            factorcol(key))))
    end
    sendoffsets = todevice(backend, Int32[e.offset for e in sendlayout])
    recvoffsets = todevice(backend, Int32[e.offset for e in recvlayout])
    return RS(sendpeers, sendcounts, recvpeers, recvcounts, sendlayout, recvlayout,
              packs, unpacks, sendoffsets, recvoffsets,
              fieldtype(RS, :buffers)(), fieldtype(RS, :mirrors)())
end

remotetype(::Type{ExchangeStage{GRP,RS}}) where {GRP,RS} = RS

# The stages of an exchange, in tag order: one per tag that this rank
# has local groups or messages in, `localsof(tag)` giving the local
# groups (possibly none) and `sentby` / `receivedby` the remote transfers
# filed by stage. `required` tags are present even when empty here, as
# phase 1 is, after which the boundary hook runs.
function build_stages(::Type{ST}, ::Type{T}, backend::Backend, localtags, localsof,
                      sentby::Dict{Int,TransferPairs{D}},
                      receivedby::Dict{Int,TransferPairs{D}}, stencils, factorcol,
                      forest::Forest{D}; required=Int[],
                      plane::NTuple{2,Int8}=(Int8(0), Int8(0)),
                      @nospecialize(bounded=nobound)) where {ST,T,D}
    owned = blockrange(forest)
    owner(i) = leafowner(forest, i)
    tags = sort!(unique!([required; localtags; collect(keys(sentby));
                          collect(keys(receivedby))]))
    nopairs = TransferPairs{D}()
    return ST[ST(tag, localsof(tag),
                 remote_stage(remotetype(ST), T, backend, get(sentby, tag, nopairs),
                              get(receivedby, tag, nopairs), stencils, factorcol;
                              targetowner=owner, sourceowner=owner, targetrange=owned,
                              sourcerange=owned, plane=plane, bounded=bounded))
              for tag in tags]
end

# --- Stage groups ----------------------------------------------------------
#
# Which phase-2 stages run together on this rank (`GhostSchedule`'s
# `stagegroups`; "Ghost filling" in CODE.md). The prolongations onto
# level ℓ wait for those onto ℓ − 1 only because one of them may read a
# ghost of its coarse source that the stage below wrote. Where none
# does, the two stages can run as one: their messages in flight
# together, their local groups in one parallel loop, one wait. A group
# is permission, not obligation — running its stages one after another
# is always correct — so the decision is this rank's own, and ranks may
# decide differently: a message's tag and layout are its stage's, not
# its group's.
#
# The test is on the points themselves, per local block, from the host
# stencils the groups are built from, before anything goes to a device.
# A stage writes the target boxes of its local groups and its unpacks,
# and the relative hook's regions that run after it (`BoundaryRegion`'s
# `slot`); it reads the source windows of its local groups and its packs
# on their source blocks, and the hook regions' nearest evolved points.
# Every write to a block is made on the block's own rank, so a rank sees
# every write that a read of its blocks could meet. A level with any
# rotated prolongation never joins: its source windows are in the
# virtual frame, which the axis map turns into the real source's only at
# the launch, and an odd orientation of a `RotationPair` reads the
# partner's array, which this rank's boxes do not describe. The cost is
# one pass over the transfers and one over the hook regions, with a few
# boxes per block.

const Box{D} = NTuple{D,UnitRange{Int}}

# The target box of a group's transfers, per dimension, in stored indices.
function write_box(stencils::NTuple{D,Stencil1D}) where {D}
    return ntuple(d -> stencils[d].targetfirst:(stencils[d].targetfirst +
                                                ntarget(stencils[d]) - 1), D)
end

# The window a group's transfers read on their source block: along each
# dimension from the first column's start to the last one's end. The
# width is the kernel's (see `run_group!`): a bounded point-value
# stencil stacks two sets of weights over one window, so it reads half
# its rows, and a bounded conservative one reads its three. A column that
# takes a single tap reads less, which the window covers. Mirrored and
# clamped rows keep their image's columns, which are in the source's
# frame, and so does the derived wall row, which reads the source beside
# its own wall.
function read_box(stencils::NTuple{D,Stencil1D}, mode::Int8) where {D}
    return ntuple(D) do d
        s = stencils[d]
        w = stencilorder(s)
        mode == BOUNDED_POINTVALUE && (w ÷= 2)
        return Int(minimum(s.srcstart)):(Int(maximum(s.srcstart)) + w - 1)
    end
end

# The points the relative hook reads for a region: the nearest evolved
# point along each dimension it clamps, the ghost's own index along every
# other (see `BoundaryRegion`).
hook_read_box(r::BoundaryRegion{D}) where {D} =
    ntuple(d -> r.clamps[d] == 0 ? r.region.indices[d] : (Int(r.clamps[d]):Int(r.clamps[d])),
           D)

overlaps(a::Box{D}, b::Box{D}) where {D} =
    all(d -> max(first(a[d]), first(b[d])) <= min(last(a[d]), last(b[d])), 1:D)

# The boxes a group's transfers write and read, or `nothing` for a group
# with no target. The one step that sees the stencils, whose type is the
# element type's as well as the dimension's; `stage_groups` around it
# depends on `D` alone.
function stage_boxes(stencils::NTuple{D,Stencil1D}, mode::Int8) where {D}
    write = write_box(stencils)
    any(isempty, write) && return nothing
    return (write, read_box(stencils, mode))
end

# What one phase-2 stage touches on this rank, each box on a local block,
# and whether any of its prolongations is rotated.
struct StageAccess{D}
    writes::Vector{Tuple{Int32,Box{D}}}
    reads::Vector{Tuple{Int32,Box{D}}}
    rotated::Base.RefValue{Bool}
end
StageAccess{D}() where {D} =
    StageAccess{D}(Tuple{Int32,Box{D}}[], Tuple{Int32,Box{D}}[], Ref(false))

# Boxes by local block, for the stages of the group being formed.
const BlockBoxes{D} = Dict{Int32,Vector{Box{D}}}

# Whether a box of `list` meets one of `boxes` on its block. Loops rather
# than `any` over closures, each of which would be methods of their own
# for every `D`.
function meets(boxes::BlockBoxes{D}, list::Vector{Tuple{Int32,Box{D}}}) where {D}
    for (b, box) in list
        others = get(boxes, b, nothing)
        others === nothing && continue
        for w in others
            overlaps(w, box) && return true
        end
    end
    return false
end

# Record `box` on each of `blocks`, shifted by `shift` to local blocks:
# zero for blocks already local, this rank's first leaf less one for
# global leaf indices.
function record_boxes!(list::Vector{Tuple{Int32,Box{D}}}, blocks, shift::Int,
                       box::Box{D}) where {D}
    for b in blocks
        push!(list, (Int32(b - shift), box))
    end
    return list
end

function add_boxes!(boxes::BlockBoxes{D}, list::Vector{Tuple{Int32,Box{D}}}) where {D}
    for (b, box) in list
        push!(get!(Vector{Box{D}}, boxes, b), box)
    end
    return boxes
end

# The stage groups of a schedule whose stages have the tags `tags`, phase
# 1 first. `pairs` are the local transfers in local block indices; `sent`
# and `received` the remote ones in global leaf indices, `offset` this
# rank's first leaf less one; `boundaries` the hook regions in local
# indices; `build(key)` a group's host stencils and `boundedof(key)` its
# bound. Neither of the two is specialized on: their types carry the
# element type, and the boxes they give are the same `Box{D}` for every
# one, so the function is compiled once per dimension.
#
# Phase 1 is alone: the boundary hook runs after it. A stage joins the
# group below it when it is the next level up, when none of its
# prolongations is rotated, and when it conflicts with no stage already
# in the group: none of its points is written by one of them and read or
# written by the other. Only the stage just below can write what it
# reads — a level-ℓ prolongation reads level-(ℓ − 1) blocks, which
# nothing but the level-(ℓ − 1) stage and its hook regions write — and no
# stage of a group writes a block another one reads; but testing every
# pair costs the same and needs neither argument to be right.
function stage_groups(tags::Vector{Int}, pairs::TransferPairs{D},
                      sent::TransferPairs{D}, received::TransferPairs{D},
                      boundaries::Vector{BoundaryRegion{D}}, @nospecialize(build),
                      @nospecialize(boundedof), offset::Int) where {D}
    length(tags) <= 2 && return [i:i for i in eachindex(tags)]
    index = Dict{Int,Int}()
    access = Vector{StageAccess{D}}(undef, length(tags))
    for (i, tag) in enumerate(tags)
        index[tag] = i
        access[i] = StageAccess{D}()
    end
    for (lists, shift, reads, writes) in ((pairs, 0, true, true), (sent, offset, true, false),
                                          (received, offset, false, true))
        for (key, (targets, sources)) in lists
            key.kind === :prolong || continue
            i = get(index, prolongation_tag(key.level), 0)
            i == 0 && continue
            boxes = stage_boxes(build(key), boundedof(key)::Int8)::Union{Nothing,
                                                                         NTuple{2,Box{D}}}
            boxes === nothing && continue
            a = access[i]
            key.orientation == 0 || (a.rotated[] = true)
            reads && record_boxes!(a.reads, sources, shift, boxes[2])
            writes && record_boxes!(a.writes, targets, shift, boxes[1])
        end
    end
    for r in boundaries
        i = get(index, r.slot, 0)
        i <= 1 && continue
        push!(access[i].writes, (r.block, r.region.indices))
        push!(access[i].reads, (r.block, hook_read_box(r)))
    end

    groups = [1:1]
    written, read = BlockBoxes{D}(), BlockBoxes{D}()   # by the stages of the open group
    for i in 2:length(tags)
        a = access[i]
        joins = i > 2 && tags[i] == tags[i - 1] + 1 && !a.rotated[] &&
                !meets(written, a.reads) && !meets(written, a.writes) &&
                !meets(read, a.writes)
        if joins
            groups[end] = first(groups[end]):i
        else
            push!(groups, i:i)
            empty!(written)
            empty!(read)
        end
        add_boxes!(written, a.writes)
        add_boxes!(read, a.reads)
    end
    return groups
end

# The local groups each stage group runs as one phase, in stage order: a
# stage alone runs its own `locals`, the very vector, and a group of
# several the concatenation of theirs, made once here. So the fill hands
# `run_phase!` a `Vector` of the schedule's group type for a group as for
# a stage, and compiles one phase per field-set type, not two.
function group_locals(stages::Vector{<:ExchangeStage{GRP}},
                      groups::Vector{UnitRange{Int}}) where {GRP}
    out = Vector{Vector{GRP}}(undef, length(groups))
    for (k, r) in enumerate(groups)
        if length(r) == 1
            out[k] = stages[first(r)].locals
        else
            joined = GRP[]
            for i in r
                append!(joined, stages[i].locals)
            end
            out[k] = joined
        end
    end
    return out
end

function localize_boundaries(boundaries::Vector{BoundaryRegion{D}},
                             range::UnitRange{Int}) where {D}
    offset = Int32(first(range) - 1)
    iszero(offset) && return boundaries
    return [BoundaryRegion{D}(r.block - offset, r.direction, r.region, r.clamps, r.slot)
            for r in boundaries]
end

# A schedule's local groups from its transfer lists: phase 1, and the
# prolongations by target level. Behind a function barrier, since the
# builder takes its element type as a run-time value: here `T` is static,
# so each group is built by a static call. (Built inline, M12's small
# fields made the call box them once per group, which `bench/ghosts.jl`
# showed as 448 bytes more per build.) `bounded(key)` is the group's
# bound, `boundedmode`.
function local_groups(::Type{GRP}, ::Type{T}, ::Val{D}, backend, pairs::TransferPairs{D},
                      build, factorcol, plane, bounded) where {GRP,T,D}
    phase1 = GRP[]
    bylevel = Dict{Int,Vector{GRP}}()
    for (key, (targets, sources)) in pairs
        # The stencils come out of a `Dict{…,Any}`.
        stencils = build(key)::NTuple{D,Stencil1D{T,Vector{Int32},Matrix{T}}}
        group = todevice(backend, TransferGroup{T,D}(key.kind, stencils, targets, sources,
                                                     factorcol(key), key.orientation,
                                                     plane, bounded(key)::Int8))
        if key.kind === :prolong
            push!(get!(bylevel, key.level, GRP[]), group)
        else
            push!(phase1, group)
        end
    end
    return phase1, bylevel
end

GhostSchedule(fs::FieldSet{T,D}, operators::Operators) where {T,D} =
    GhostSchedule(fs.forest, operators; G=fs.G, centering=fs.centering, P=fs.P,
                  walls=fs.walls, T=T, backend=get_backend(fs.work))

function GhostSchedule(forest::Forest{D,R}, operators::Operators;
                       G::Union{Integer,Tuple{Vararg{Integer}}},
                       centering=cellcentered(D), P::Integer=0,
                       walls::Symbol=:half_open,
                       T::Type=R, backend::Backend=CPU()) where {D,R}
    N = forest.N
    # The build is collective over a distributed forest (M7): the ranks
    # agree that their forests and layouts are the same, and a refusal
    # on any of them is raised on all of them.
    checked = collective_checks(forest, "GhostSchedule") do
        gs = ghostwidths(G, Val(D))
        cs = centerings(centering, Val(D))
        ss = staggers(cs)
        storedsize(N, gs, ss)                    # the N >= 2G[d] + 2c[d] invariant
        # `0` is the forest form's "no elements", as `FieldSet` stores it.
        order = element_order(P == 0 ? nothing : P, cs, N, gs, forest)
        check_operators(N, gs, ss, operators)
        check_floattype(T, backend)
        check_walls(walls)
        layout = layouthash(gs, cs, order, walls, Int(operators.family),
                            operators.prolongation,
                            operators.restriction, operators.bounded,
                            operators.vertex_prolongation, T,
                            nameof(typeof(backend)))
        return (gs, cs, ss, order), layout
    end
    ghosts::NTuple{D,Int}, centers::NTuple{D,Symbol}, stags::NTuple{D,Int}, order::Int =
        checked
    symmetric = walls === :symmetric
    dirs = alldirections(Val(D))
    # The targets are this rank's blocks (M7), walked in global leaf
    # indices, since that is what the tree answers in; `split_received!`
    # turns the local transfers into local block indices below.
    owned = blockrange(forest)
    offset = first(owned) - 1
    nb = length(owned)

    # Neighbor finding, threaded over blocks: it reads nothing but the
    # tree, and each task collects into buffers of its own. Those are
    # concatenated in block order, so the schedule that comes out is
    # identical whatever `Threads.nthreads()` happens to be.
    chunks = threadchunks(nb)
    perpairs = [TransferPairs{D}() for _ in chunks]
    perboundaries = [BoundaryRegion{D}[] for _ in chunks]
    threaded_chunks(nb) do c, range
        for b in range
            block_sources!(perpairs[c], perboundaries[c], forest, ghosts, stags,
                           offset + b, dirs, symmetric)
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
    # Over a distributed forest (M7) the transfers split three ways: the
    # local ones stay in `pairs`, in local block indices; those with a
    # remote source are received; and those this rank computes for a
    # remote target are sent, found by the target owner's own search.
    received = split_received!(pairs, owned, nleaves(forest))
    boundaries = localize_boundaries(boundaries, owned)
    sent = sent_transfers(forest, owned) do into, j
        block_sources!(into, BoundaryRegion{D}[], forest, ghosts, stags, j, dirs, symmetric)
    end

    # One dimension of a group's stencils. Along a wall dimension of a
    # mirror transfer the stencil is the ordinary *tangential* one with
    # its target rows remapped across the wall (or clamped onto the
    # nearest owned row at an outflow face), or the derived wall row.
    boundedof(key) = boundedmode(operators, key.kind, key.mirror, stags)
    function build1(key, d)
        δd, od, state, low = key.direction[d], key.offset[d], key.mirror[d], key.low[d]
        shift = key.shift[d]
        mode = boundedof(key)
        if state == 2
            # The derived wall row: the high one, or a symmetric low one.
            wall = wall_stencil(T, N, ghosts[d], vertex_order(operators); low=δd < 0)
            mode == BOUNDED_POINTVALUE || return wall
            return stack_linear(wall, wall_stencil(T, N, ghosts[d], 2; low=δd < 0))
        end
        # Along a wall dimension the tangential stencil is built whole and
        # its rows remapped; along any other the target range is trimmed
        # where the block's low wall is written.
        δs = state == 0 ? δd : 0
        s = kind_stencil(T, N, ghosts[d], stags[d], key.kind, δs, od, operators, mode;
                         low=state == 0 && low, shift=state == 0 ? shift : 0)
        state == 0 && return s
        state == 3 && return clamp_rows(s, N, ghosts[d], stags[d], δd; low=low)
        return mirror_rows(s, N, ghosts[d], stags[d], δd)
    end
    # Memoized per key: over a distributed forest the stage builder asks
    # again for the keys the local groups were built from.
    built = Dict{GroupKey{D},Any}()
    build(key) = get!(() -> ntuple(d -> build1(key, d), D), built, key)
    # The factor column covers the mirror state and the orientation
    # (M12): `mirror column + 3^D·r`, so an ordinary transfer keeps 0 and
    # a mirrored one off the seam the column it had. A clamped dimension
    # takes no sign, so it counts as unmirrored, and a transfer clamped
    # only is an ordinary one to the kernel.
    function factorcol(key)
        parity = map(m -> m == 3 ? Int8(0) : m, key.mirror)
        return any(!iszero, parity) || key.orientation != 0 ?
               mirrorcolumn(parity) + 3^D * Int(key.orientation) : 0
    end
    plane = map(Int8, seam_plane(forest))

    GRP = grouptype(backend, T, Val(D))
    phase1, bylevel = local_groups(GRP, T, Val(D), backend, pairs, build, factorcol, plane,
                                   boundedof)

    levels = sort!(collect(keys(bylevel)))          # coarsest targets first
    phase2 = [bylevel[l] for l in levels]
    bplan = BoundaryPlan(backend, T, forest, boundaries, order)

    # Phase 1 is one stage and phase 2 one per target level. Serially
    # each stage is one of the phases above and sends nothing.
    stageof(key) = key.kind === :prolong ? prolongation_tag(key.level) : PHASE1_TAG
    leveltags = prolongation_tag.(levels)
    localsof(tag) = tag == PHASE1_TAG ? phase1 :
                    (i = findfirst(==(tag), leveltags); i === nothing ? GRP[] : phase2[i])
    ST = stagetype(backend, T, Val(D))
    stages = build_stages(ST, T, backend, leveltags, localsof, bystage(stageof, sent),
                          bystage(stageof, received), build, factorcol, forest;
                          required=[PHASE1_TAG], plane=plane, bounded=boundedof)
    # Which of them may run together here, from the host stencils.
    groups = stage_groups([st.tag for st in stages], pairs, sent, received, boundaries,
                          build, boundedof, offset)
    return GhostSchedule{T,D,R,typeof(backend),GRP,typeof(bplan),ST}(
        forest, generation(forest), ghosts, centers, order, walls, operators, backend,
        phase1, phase2, levels, boundaries, bplan, stages, groups)
end

function Base.show(io::IO, s::GhostSchedule{T,D}) where {T,D}
    ncopy = sum(ntransfers, filter(g -> g.kind === :copy, s.phase1); init=0)
    nrest = sum(ntransfers, filter(g -> g.kind === :restrict, s.phase1); init=0)
    nprol = sum(gs -> sum(ntransfers, gs; init=0), s.phase2; init=0)
    # Of all of those, the ones mirrored across a reflecting face (M10)
    # and the ones rotated across a seam (M12), each said only when there
    # are any, so a schedule without either prints as it always has.
    groups = Iterators.flatten((s.phase1, Iterators.flatten(s.phase2)))
    nmirror = sum(g -> ismirrored(g) ? ntransfers(g) : 0, groups; init=0)
    nrot = sum(g -> g.orientation == 0 ? 0 : ntransfers(g), groups; init=0)
    print(io, "GhostSchedule{", T, ",", D, "}(", ncopy, " copies, ", nrest,
          " restrictions, ", nprol, " prolongations over ", length(s.phase2),
          " level(s), ", nmirror == 0 ? "" : "$nmirror mirrored transfers, ",
          nrot == 0 ? "" : "$nrot rotated transfers, ",
          length(s.boundaries), " boundary regions", groups_summary(s),
          messages_summary(s.stages), ")")
end

# The prolongation levels that run together, said only when some do, so
# that a schedule whose stages each run alone prints as it always has.
function groups_summary(s::GhostSchedule)
    runs = filter(r -> length(r) > 1, s.stagegroups)
    isempty(runs) && return ""
    lvl(i) = s.stages[i].tag - prolongation_tag(0)
    return "; levels " * join(("$(lvl(first(r)))-$(lvl(last(r)))" for r in runs), ", ") *
           " run together"
end

# Whether a group mirrors across a reflecting face: a factor column
# whose mirror state, the column within its orientation's block of `3^D`,
# is not the unmirrored one.
ismirrored(g::TransferGroup{T,D}) where {T,D} =
    g.factorcol != 0 && mod(g.factorcol - 1, 3^D) != 0

# What a distributed schedule (M7) sends and receives, said only when it
# does, so that a serial schedule prints as it always has.
function messages_summary(stages)
    nsent = sum(st -> st.remote === nothing ? 0 : length(st.remote.sendlayout), stages;
                init=0)
    nrecv = sum(st -> st.remote === nothing ? 0 : length(st.remote.recvlayout), stages;
                init=0)
    nsent + nrecv == 0 && return ""
    return "; $nsent transfers sent and $nrecv received"
end
