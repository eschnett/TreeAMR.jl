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
"""
struct Stencil1D{T,VI<:AbstractVector{Int32},MW<:AbstractMatrix{T}}
    targetfirst::Int
    srcstart::VI
    weights::MW
end

Stencil1D{T}(targetfirst::Int, srcstart::AbstractVector{Int32},
             weights::AbstractMatrix{T}) where {T} =
    Stencil1D{T,typeof(srcstart),typeof(weights)}(targetfirst, srcstart, weights)

ntarget(s::Stencil1D) = length(s.srcstart)
Base.@propagate_inbounds stencilorder(s::Stencil1D) = size(s.weights, 1)

todevice(backend::Backend, s::Stencil1D{T}) where {T} =
    Stencil1D{T}(s.targetfirst, todevice(backend, s.srcstart),
                 todevice(backend, s.weights))

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
`CODE.md`). Its stencils are built in that virtual frame, and the kernel
reads the real source through the axis map of `r` (see `axismap`) and the
field set's variable table. Zero means an ordinary source, read as it
is. The three small fields share the eight bytes `factorcol` had alone,
so a group is no larger than before M12.
"""
struct TransferGroup{T,D,S<:Stencil1D{T},VB<:AbstractVector{Int32}}
    kind::Symbol                      # :copy, :restrict, or :prolong
    stencils::NTuple{D,S}
    targetblocks::VB
    sourceblocks::VB
    factorcol::Int32                  # 0, or the factor-table column
    orientation::Int8                 # 0, or the quarter turns of the source
    plane::NTuple{2,Int8}             # the seam's (d1, d2), or (0, 0)
end

TransferGroup{T,D}(kind::Symbol, stencils::NTuple{D,S},
                   targetblocks::VB, sourceblocks::VB, factorcol::Integer=0,
                   orientation::Integer=0,
                   plane::NTuple{2,Integer}=(0, 0)) where {T,D,S,VB} =
    TransferGroup{T,D,S,VB}(kind, stencils, targetblocks, sourceblocks,
                            Int32(factorcol), Int8(orientation), Int8.(plane))

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
                       g.plane)
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
outer face — neither periodic nor reflecting — and so is filled by the
user's boundary hook rather than by an inter-block transfer. A corner
or edge region crossing a reflecting face as well is one of these when
its mirror image still lies outside through an outer face.
"""
struct BoundaryRegion{D}
    block::Int32
    direction::NTuple{D,Int}
    region::CartesianIndices{D,NTuple{D,UnitRange{Int}}}
end

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

The three arrays live wherever the schedule's backend is (M6).
"""
struct BoundaryBatch{D,VB<:AbstractVector{Int32},VD<:AbstractVector{NTuple{D,Int8}},
                     VF<:AbstractVector{NTuple{D,Int32}}}
    boxlen::NTuple{D,Int}
    blocks::VB
    directions::VD
    firsts::VF                        # first stored index of the region
end

nregions(b::BoundaryBatch) = length(b.blocks)

todevice(backend::Backend, b::BoundaryBatch{D}) where {D} =
    BoundaryBatch{D}(b.boxlen, todevice(backend, b.blocks),
                     todevice(backend, b.directions), todevice(backend, b.firsts))
BoundaryBatch{D}(boxlen, blocks::VB, directions::VD, firsts::VF) where {D,VB,VD,VF} =
    BoundaryBatch{D,VB,VD,VF}(boxlen, blocks, directions, firsts)

"""
    BoundaryPlan{D}

Everything the cell-wise boundary kernel reads that is not field data:
the [`BoundaryBatch`](@ref TreeAMR.BoundaryBatch)es, and the per-block
origin and spacing it turns a stored index into a position with.

The geometry is here rather than recomputed per fill for the same reason
the exchange itself is precomputed: it depends only on the tree, which
is exactly what a schedule is rebuilt for.
"""
struct BoundaryPlan{D,BT<:BoundaryBatch{D},VO<:AbstractVector,VS<:AbstractVector}
    batches::Vector{BT}
    origins::VO
    spacings::VS
end

function batchtype(backend::Backend, ::Val{D}) where {D}
    VB = typeof(todevice(backend, Int32[0]))
    VD = typeof(todevice(backend, [ntuple(_ -> Int8(0), D)]))
    VF = typeof(todevice(backend, [ntuple(_ -> Int32(0), D)]))
    return BoundaryBatch{D,VB,VD,VF}
end

# Group the outward-facing regions by shape, then move the result to
# wherever the kernels will run.
function BoundaryPlan(backend::Backend, ::Type{T}, forest::Forest{D},
                      boundaries::Vector{BoundaryRegion{D}}) where {T,D}
    byshape = Dict{NTuple{D,Int},Tuple{Vector{Int32},Vector{NTuple{D,Int8}},
                                       Vector{NTuple{D,Int32}}}}()
    for r in boundaries
        shape = ntuple(d -> length(r.region.indices[d]), D)
        slot = get!(byshape, shape,
                    (Int32[], NTuple{D,Int8}[], NTuple{D,Int32}[]))
        push!(slot[1], r.block)
        push!(slot[2], ntuple(d -> Int8(r.direction[d]), D))
        push!(slot[3], ntuple(d -> Int32(first(r.region.indices[d])), D))
    end
    # Sorted by shape so the batch order is a function of the tree
    # alone, not of dictionary iteration order.
    shapes = sort!(collect(keys(byshape)))
    # Concretely typed even when empty — a fully periodic domain has no
    # outward-facing regions at all, which is the common case.
    BT = batchtype(backend, Val(D))
    batches = BT[todevice(backend, BoundaryBatch{D}(sh, byshape[sh]...)) for sh in shapes]
    origins = todevice(backend, block_origins(forest, T))
    spacings = todevice(backend, block_spacings(forest, T))
    return BoundaryPlan{D,eltype(batches),typeof(origins),typeof(spacings)}(
        batches, origins, spacings)
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
# reflecting face (M10): 0 for an ordinary dimension, 1 for rows
# mirrored across the wall, 2 for the derived upper wall row of a
# vertex-like dimension. It changes the stencils, and — through the
# parity factor — what the kernel does with their result, so it is part
# of the key. Everything that is not a mirror transfer has all zeros.
#
# `orientation` is the number of quarter turns that carry the source
# across a rotating seam to where the target sees it (M12): it changes
# where the kernel reads and the sign it applies, so it is part of the
# key too. The direction and the offset are the *virtual* ones, as the
# target sees them, since those fix the stencils. Zero off the seam.
struct GroupKey{D}
    kind::Symbol
    direction::NTuple{D,Int}
    offset::NTuple{D,Int}
    level::Int
    mirror::NTuple{D,Int8}
    orientation::Int8
end

GroupKey{D}(kind::Symbol, direction::NTuple{D,Int}, offset::NTuple{D,Int},
            level::Int) where {D} =
    GroupKey{D}(kind, direction, offset, level, ntuple(_ -> Int8(0), D), Int8(0))
GroupKey{D}(kind::Symbol, direction::NTuple{D,Int}, offset::NTuple{D,Int},
            level::Int, mirror::NTuple{D,Int8}) where {D} =
    GroupKey{D}(kind, direction, offset, level, mirror, Int8(0))

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
    (kindrank(k.kind), k.direction, k.offset, k.level, k.mirror, k.orientation)
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
  an *outer* face — neither periodic nor reflecting — filled by the user
  hook between the two phases;
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

Periodic boundaries appear nowhere special here: the tree wraps around,
so they are ordinary copies, restrictions, and prolongations. Reflecting
faces (M10) are ordinary transfers too, from mirrored sources: a copy,
restriction or prolongation with its target rows remapped across the
wall, whose result the kernel multiplies by each variable's parity
sign. They sit in the same two phases as every other transfer, so the
hook never sees them; see "Ghost filling" in `CODE.md`. So are the
ghosts across a rotating seam (M12): each is a transfer from the real
leaf across the seam, built as though that leaf sat where the block sees
it and read through the quarter turns between the two, which `show`
reports as rotated transfers. A region whose turned image leaves through
an outer face is the hook's.

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
out instead, for a caller with no field set in hand.

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
    operators::Operators
    backend::BK
    phase1::Vector{GRP}
    phase2::Vector{Vector{GRP}}                  # by target level, coarsest first
    levels::Vector{Int}                          # target level of each phase2 entry
    boundaries::Vector{BoundaryRegion{D}}
    boundaryplan::BP
    stages::Vector{ST}                           # phase 1, then phase 2 by level
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
# `closed` extends the block's *own* range (δd = 0) by the shared plane,
# turning the owned range into the closed one and giving the top half of
# a halved range the extra point. The ghost exchange never wants it — a
# block's shared plane is the target of its δ_d = +1 region, not of a
# tangential one — but the interface restriction does: it overwrites the
# block's own closed-range values, and in a vertex-like dimension the
# boundary line of a coarse-fine face is part of what must agree. Keeping
# it here, rather than in a range the interface schedule derives for
# itself, is what keeps this function the single source of truth.
function target_range(N::Int, G::Int, c::Int, δd::Int, od::Int, halved::Bool,
                      closed::Bool=false)
    δd == 1 && return (G + N + 1):(N + 2G + c)
    δd == -1 && return 1:G
    top = closed ? c : 0                            # the shared plane, to the top half
    halved && return (G + 1 + od * (N ÷ 2)):(G + od * (N ÷ 2) + N ÷ 2 + od * top)
    return (G + 1):(G + N + top)
end

# Same-level copy: a pure shift of N cells against the direction, for
# every centering. In a vertex-like dimension the high target is one
# plane longer and its first entry, the shared plane G+N+1, reads the
# neighbor's first *owned* point G+1 — which is the same point. So a copy
# still reads interiors only, which is what makes phase 1 race free.
function copy_stencil(::Type{T}, N::Int, G::Int, c::Int, δd::Int) where {T}
    rng = target_range(N, G, c, δd, 0, false)
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
                 p::Int) where {T} =
    restrict_stencil_over(T, target_range(N, G, c, δd, od, true), N, G, c, δd, od, p)

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
                         p::Int) where {T}
    rng = target_range(N, G, c, δd, od, false)
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
    # `check_operators` refuses this family on a field set with any
    # vertex-like dimension, so `c` only ever reaches `target_range`.
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
prolongation_stencil(::Type{T}, N, G, c, δd, od, ops::Operators) where {T} =
    ops.family === Conservative ?
    conservative_prolong_stencil(T, N, G, c, δd, od, ops.prolongation) :
    prolong_stencil(T, N, G, c, δd, od, ops.prolongation)

# Conservative restriction is the exact volume average, which the
# point-value builder already produces at order 2: its window is a
# cell's own two children and its weights are 1/2.
restriction_stencil(::Type{T}, N, G, c, δd, od, ops::Operators) where {T} =
    restrict_stencil(T, N, G, c, δd, od, ops.restriction)

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
function wall_stencil(::Type{T}, N::Int, G::Int, p::Int) where {T}
    h = p ÷ 2
    nodes = [Rational{Int}(i) for i in vcat(-h:-1, 1:h)]
    w = lagrange_weights(nodes, 0//1)
    weights = Matrix{T}(undef, h, 1)
    for i in 1:h
        weights[i, 1] = T(2 * w[i])                 # nodes -h … -1, ascending
    end
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
                        b::Int, dirs) where {D}
    k = forest.leaves[b]
    N = forest.N
    zerooffset = ntuple(_ -> 0, D)
    record!(kind, δ, offset, lvl, s, r) =
        push!.(get!(pairs, GroupKey{D}(kind, δ, offset, lvl, ntuple(_ -> Int8(0), D),
                                       Int8(r)),
                    (Int32[], Int32[])),
               (Int32(b), Int32(s)))
    for δ in dirs
        region = CartesianIndices(ntuple(d -> target_range(N, G[d], c[d], δ[d], 0,
                                                           false), D))
        # A cell-centered dimension with `G_d = 0` has no low slab and no
        # high one, so every direction that leaves the block along it has
        # nothing to fill. Skipping here rather than launching an empty
        # kernel also spares the tree query. (Halving a tangential range
        # cannot empty it, so testing the unhalved region covers every
        # kind.)
        isempty(region) && continue
        δ′, mask = reflect_direction(forest, k, δ)
        if any(mask)
            mirror_sources!(pairs, boundaries, forest, G, c, b, δ, δ′, mask, region)
            continue
        end
        r, nbrs = oriented_neighbors(forest, k, δ)
        if isempty(nbrs)
            push!(boundaries, BoundaryRegion{D}(Int32(b), δ, region))
            continue
        end
        nblevel = level(first(nbrs))
        if nblevel == level(k)
            record!(:copy, δ, zerooffset, 0, find_leaf(forest, only(nbrs)), r)
        elseif nblevel < level(k)
            # Coarser neighbor: this block's ghosts are prolongated. The
            # stencil geometry depends on where this block sits inside
            # its own parent.
            record!(:prolong, δ, childoffset(k), level(k), find_leaf(forest, only(nbrs)),
                    r)
        else
            # Finer neighbors: each supplies one part of this block's
            # ghost region, selected by its offset within its parent —
            # its virtual parent, where this block sees it.
            for nbr in nbrs
                record!(:restrict, δ, seam_offset(forest, nbr, r), 0,
                        find_leaf(forest, nbr), r)
            end
        end
    end
    return nothing
end

# The child offset of a finer source as the target sees it: its own
# offset off the seam, and across it in orientation `r` the offset turned
# into the virtual frame (M12).
seam_offset(forest::Forest, nbr::MortonKey, r::Integer) =
    r == 0 ? childoffset(nbr) :
    virtual_offset(childoffset(nbr), r, forest.rotating[1], forest.rotating[2])

# The sources of a ghost region that crosses a reflecting face (M10; see
# "Ghost filling" in CODE.md). Mirrored along the masked dimensions, the
# region lies inside the domain: in the block itself when `δ′` is zero,
# and otherwise in the node adjacent to it in direction `δ′`, at the
# block's own extent along every masked dimension. So the source is
# whatever the ordinary search finds in `δ′`, and the transfer is
# recorded under the original `δ`, which fixes the target range. A `δ′`
# that leaves through an outer face makes the region the hook's.
function mirror_sources!(pairs::TransferPairs{D},
                         boundaries::Vector{BoundaryRegion{D}},
                         forest::Forest{D}, G::NTuple{D,Int}, c::NTuple{D,Int},
                         b::Int, δ::NTuple{D,Int}, δ′::NTuple{D,Int},
                         mask::NTuple{D,Bool}, region) where {D}
    k = forest.leaves[b]
    zerooffset = ntuple(_ -> 0, D)
    # A masked vertex-like dimension on the high side splits its target
    # into the derived wall row (state 2) and the mirrored rows beyond it
    # (state 1), which are batched apart because their parity factors
    # differ. Each part may be empty — at `G = 0` only the wall row is
    # left — and an empty part records nothing.
    choices = ntuple(D) do d
        !mask[d] && return (Int8(0),)
        c[d] == 1 && δ[d] == 1 || return (Int8(1),)
        return G[d] == 0 ? (Int8(2),) : (Int8(1), Int8(2))
    end
    states = vec(collect(Iterators.product(choices...)))
    record!(kind, offset, lvl, s, r) =
        for state in states
            push!.(get!(pairs, GroupKey{D}(kind, δ, offset, lvl, state, Int8(r)),
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
        push!(boundaries, BoundaryRegion{D}(Int32(b), δ, region))
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
# Returns `nothing` when the stage has no messages here.
function remote_stage(::Type{RS}, ::Type{T}, backend::Backend,
                      sent::TransferPairs{D}, received::TransferPairs{D}, stencils,
                      factorcol; targetowner, sourceowner, targetrange::UnitRange{Int},
                      sourcerange::UnitRange{Int},
                      plane::NTuple{2,Int8}=(Int8(0), Int8(0))) where {RS<:RemoteStage,T,D}
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
        st = map(s -> Stencil1D{T}(1, s.srcstart, s.weights), host(key))
        push!(packs, todevice(backend, TransferGroup{T,D}(
            key.kind, st, Int32[o[2] for o in order], Int32[o[1] for o in order], 0,
            key.orientation, plane)))
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
                      plane::NTuple{2,Int8}=(Int8(0), Int8(0))) where {ST,T,D}
    owned = blockrange(forest)
    owner(i) = leafowner(forest, i)
    tags = sort!(unique!([required; localtags; collect(keys(sentby));
                          collect(keys(receivedby))]))
    nopairs = TransferPairs{D}()
    return ST[ST(tag, localsof(tag),
                 remote_stage(remotetype(ST), T, backend, get(sentby, tag, nopairs),
                              get(receivedby, tag, nopairs), stencils, factorcol;
                              targetowner=owner, sourceowner=owner, targetrange=owned,
                              sourcerange=owned, plane=plane))
              for tag in tags]
end

function localize_boundaries(boundaries::Vector{BoundaryRegion{D}},
                             range::UnitRange{Int}) where {D}
    offset = Int32(first(range) - 1)
    iszero(offset) && return boundaries
    return [BoundaryRegion{D}(r.block - offset, r.direction, r.region) for r in boundaries]
end

# A schedule's local groups from its transfer lists: phase 1, and the
# prolongations by target level. Behind a function barrier, since the
# builder takes its element type as a run-time value: here `T` is static,
# so each group is built by a static call. (Built inline, M12's small
# fields made the call box them once per group, which `bench/ghosts.jl`
# showed as 448 bytes more per build.)
function local_groups(::Type{GRP}, ::Type{T}, ::Val{D}, backend, pairs::TransferPairs{D},
                      build, factorcol, plane) where {GRP,T,D}
    phase1 = GRP[]
    bylevel = Dict{Int,Vector{GRP}}()
    for (key, (targets, sources)) in pairs
        # The stencils come out of a `Dict{…,Any}`.
        stencils = build(key)::NTuple{D,Stencil1D{T,Vector{Int32},Matrix{T}}}
        group = todevice(backend, TransferGroup{T,D}(key.kind, stencils, targets, sources,
                                                     factorcol(key), key.orientation,
                                                     plane))
        if key.kind === :prolong
            push!(get!(bylevel, key.level, GRP[]), group)
        else
            push!(phase1, group)
        end
    end
    return phase1, bylevel
end

GhostSchedule(fs::FieldSet{T,D}, operators::Operators) where {T,D} =
    GhostSchedule(fs.forest, operators; G=fs.G, centering=fs.centering, T=T,
                  backend=get_backend(fs.work))

function GhostSchedule(forest::Forest{D,R}, operators::Operators;
                       G::Union{Integer,Tuple{Vararg{Integer}}},
                       centering=cellcentered(D),
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
        check_operators(N, gs, ss, operators)
        check_floattype(T, backend)
        layout = layouthash(gs, cs, Int(operators.family), operators.prolongation,
                            operators.restriction, T, nameof(typeof(backend)))
        return (gs, cs, ss), layout
    end
    ghosts::NTuple{D,Int}, centers::NTuple{D,Symbol}, stags::NTuple{D,Int} = checked
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
                           offset + b, dirs)
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
        block_sources!(into, BoundaryRegion{D}[], forest, ghosts, stags, j, dirs)
    end

    # One dimension of a group's stencils. Along a masked dimension of a
    # mirror transfer the stencil is the ordinary *tangential* one with
    # its target rows remapped across the wall, or the derived wall row.
    function build1(key, d)
        δd, od, state = key.direction[d], key.offset[d], key.mirror[d]
        state == 2 && return wall_stencil(T, N, ghosts[d], operators.prolongation)
        δs = state == 0 ? δd : 0
        s = key.kind === :copy ? copy_stencil(T, N, ghosts[d], stags[d], δs) :
            key.kind === :restrict ?
            restriction_stencil(T, N, ghosts[d], stags[d], δs, od, operators) :
            prolongation_stencil(T, N, ghosts[d], stags[d], δs, od, operators)
        state == 0 && return s
        return mirror_rows(s, N, ghosts[d], stags[d], δd)
    end
    # Memoized per key: over a distributed forest the stage builder asks
    # again for the keys the local groups were built from.
    built = Dict{GroupKey{D},Any}()
    build(key) = get!(() -> ntuple(d -> build1(key, d), D), built, key)
    # The factor column covers the mirror state and the orientation
    # (M12): `mirror column + 3^D·r`, so an ordinary transfer keeps 0 and
    # a mirrored one off the seam the column it had.
    factorcol(key) = any(!iszero, key.mirror) || key.orientation != 0 ?
                     mirrorcolumn(key.mirror) + 3^D * Int(key.orientation) : 0
    plane = forest.rotating

    GRP = grouptype(backend, T, Val(D))
    phase1, bylevel = local_groups(GRP, T, Val(D), backend, pairs, build, factorcol, plane)

    levels = sort!(collect(keys(bylevel)))          # coarsest targets first
    phase2 = [bylevel[l] for l in levels]
    bplan = BoundaryPlan(backend, T, forest, boundaries)

    # Phase 1 is one stage and phase 2 one per target level. Serially
    # each stage is one of the phases above and sends nothing.
    stageof(key) = key.kind === :prolong ? prolongation_tag(key.level) : PHASE1_TAG
    leveltags = prolongation_tag.(levels)
    localsof(tag) = tag == PHASE1_TAG ? phase1 :
                    (i = findfirst(==(tag), leveltags); i === nothing ? GRP[] : phase2[i])
    ST = stagetype(backend, T, Val(D))
    stages = build_stages(ST, T, backend, leveltags, localsof, bystage(stageof, sent),
                          bystage(stageof, received), build, factorcol, forest;
                          required=[PHASE1_TAG], plane=plane)
    return GhostSchedule{T,D,R,typeof(backend),GRP,typeof(bplan),ST}(
        forest, generation(forest), ghosts, centers, operators, backend, phase1,
        phase2, levels, boundaries, bplan, stages)
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
          length(s.boundaries), " boundary regions", messages_summary(s.stages), ")")
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
