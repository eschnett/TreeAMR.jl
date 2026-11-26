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
transfer's real source to where the target sees it, across a rotating
seam (M12; see "Rotating seams" in `CODE.md`); a half-turn seam (M14) has
orientation 2 only. It picks the field set's variable table column and,
odd, the partner's array in a [`RotationPair`](@ref). Zero off a seam,
across a glued face (M17) too.

`perm` and `flip` are the axis map the kernel reads a turned source by,
in the gather form of `RotatedSource`: virtual stored index `k` reads the
real point whose index along `e` is `k[perm[e]]`, reversed where
`flip[e]`. A seam's is `axismap` of its orientation, a glued face's the
glue entry's own (see `GlueEntry`), and both are computed on the host
when the group is built and carried by value, since the group is what
`run_group!` has in hand and it never crosses a rank. Its stencils are
built in the virtual frame, as though the source sat where the target
sees it. `turned` says the map is not the identity, so that the kernel's
dispatch is one test: a glued group has orientation 0 and may still be
turned, and one glued with the identity is not, and runs as an ordinary
copy.

`bounded` is `UNBOUNDED` for every group but one built from bounded
operators whose operator can make a new extremum (`boundedmode`). A
conservative prolongation (`BOUNDED_CONSERVATIVE`) then bounds the
reconstruction instead of summing the stencil; a point-value group
(`BOUNDED_POINTVALUE`) sums it and clamps the result into the range of
the target's bracket (see "Bounded prolongation" and "Bounded point
values" in `CODE.md`).

Before M17 the small fields — `orientation`, the seam's plane as two
bytes, and `bounded` — shared the eight bytes `factorcol` had alone. The
axis map takes `2D + 1` bytes in the plane's place, so a group is larger
than it was. It does not matter: a schedule holds groups by the hundred,
not by the block, and the struct never reaches a kernel whole (see
`todevice`).
"""
struct TransferGroup{T,D,S<:Stencil1D{T},VB<:AbstractVector{Int32}}
    kind::Symbol                      # :copy, :restrict, or :prolong
    stencils::NTuple{D,S}
    targetblocks::VB
    sourceblocks::VB
    factorcol::Int32                  # 0, or the factor-table column
    orientation::Int8                 # 0, or the seam's quarter turns of the source
    perm::NTuple{D,Int8}              # the axis map of a turned source,
    flip::NTuple{D,Bool}              # the identity for any other
    turned::Bool                      # the map is not the identity
    bounded::Int8                     # UNBOUNDED, or how the kernel bounds
end

# `axes` is `(perm, flip)`, as `group_axes` gives it.
TransferGroup{T,D}(kind::Symbol, stencils::NTuple{D,S},
                   targetblocks::VB, sourceblocks::VB, factorcol::Integer=0,
                   orientation::Integer=0,
                   axes::Tuple{NTuple{D,Int8},NTuple{D,Bool}}=identity_axes(Val(D)),
                   bounded::Integer=UNBOUNDED) where {T,D,S,VB} =
    TransferGroup{T,D,S,VB}(kind, stencils, targetblocks, sourceblocks,
                            Int32(factorcol), Int8(orientation), axes[1], axes[2],
                            axes != identity_axes(Val(D)), Int8(bounded))

# The identity axis map.
identity_axes(::Val{D}) where {D} =
    (ntuple(e -> Int8(e), Val(D)), ntuple(_ -> false, Val(D)))

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
                       (g.perm, g.flip), g.bounded)
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
the high side, `G + 1` on the low one — and `0` along every other dimension, where the point keeps the
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
patch corner, integer corner and spacing ([`block_lows`](@ref),
[`block_bases`](@ref), [`block_spacings`](@ref)) it turns a stored index
into a position with through [`nodeposition`](@ref
TreeAMR.nodeposition), the per-block patch (M17,
[`block_patches`](@ref)) that a [`WithPatch`](@ref) hook is handed, and
for an element-centered layout (M16) the node offsets, empty otherwise,
so that the plan's type, and the schedule's, does not depend on the
layout.

The geometry is here rather than recomputed per fill for the same reason
the exchange itself is precomputed: it depends only on the tree, which
is exactly what a schedule is rebuilt for.
"""
struct BoundaryPlan{D,BT<:BoundaryBatch{D},VL<:AbstractVector,VB<:AbstractVector,
                    VS<:AbstractVector,VP<:AbstractVector,VN}
    batches::Vector{BT}
    walls::Vector{BT}       # stage W (M22): the wall primaries a hook may prescribe
    lows::VL                # `block_lows`
    bases::VB               # `block_bases`
    spacings::VS
    patches::VP             # `block_patches`, all ones on a brick
    nodes::VN               # `node_offsets` of the layout, on the backend
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
                      boundaries::Vector{BoundaryRegion{D}}, P::Int,
                      walls::Vector{BoundaryRegion{D}}=BoundaryRegion{D}[]) where {T,D}
    batches = boundary_batches(backend, boundaries)
    lows = todevice(backend, block_lows(forest, T))
    bases = todevice(backend, block_bases(forest))
    spacings = todevice(backend, block_spacings(forest, T))
    patches = todevice(backend, block_patches(forest))
    nodes = node_offsets(backend, T, P)
    return BoundaryPlan{D,eltype(batches),typeof(lows),typeof(bases),typeof(spacings),
                        typeof(patches),typeof(nodes)}(batches,
                                                       boundary_batches(backend, walls),
                                                       lows, bases, spacings, patches,
                                                       nodes)
end

function boundary_batches(backend::Backend, boundaries::Vector{BoundaryRegion{D}}) where {D}
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
    return BT[todevice(backend, BoundaryBatch{D}(k[2], k[1], byshape[k]...))
              for k in keys_]
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
# `glue` is the index into the forest's glue table of the glued face the
# source lies across (M17; `Crossing.glue`), 0 for none. It fixes the axis
# map the kernel reads the source by, so it is part of the key, and the
# group takes the map from the table at construction (`group_axes`). An
# index rather than a table of maps collected during the build: the
# forest's table is canonically ordered and covered by `ForestDigest`, so
# the index is the same on every rank by construction, and `keyorder`
# stays a function of the keys alone.
#
#
# `span` marks a transfer of stage 0 (M22, "Closed blocks" in CODE.md),
# which writes secondary copies of points from their primaries: per
# dimension the span of piece classes of the target's closed box it
# writes (`span_code`), and `-1` everywhere for any other transfer, which
# writes the region beyond the box that `direction` names. The direction
# alone does not fix the span: along a dimension with `δ_d = 0` the
# target is the open interior and the high face plane, or the whole own
# range where no block lies below. It changes the target range, so it is
# part of the key.
struct GroupKey{D}
    kind::Symbol
    direction::NTuple{D,Int}
    offset::NTuple{D,Int}
    level::Int
    mirror::NTuple{D,Int8}
    orientation::Int8
    glue::Int8
    span::NTuple{D,Int8}
end
GroupKey{D}(kind::Symbol, direction::NTuple{D,Int}, offset::NTuple{D,Int}, level::Int,
            mirror::NTuple{D,Int8}, orientation::Int8, glue::Int8) where {D} =
    GroupKey{D}(kind, direction, offset, level, mirror, orientation, glue, nospan(Val(D)))

# The `span` of a key that is not stage 0's, and the test for one that is.
nospan(::Val{D}) where {D} = ntuple(_ -> Int8(-1), Val(D))
isunify(key::GroupKey) = key.span[1] >= 0

# A span of sub-classes along one dimension, `lo:hi`, as one code, `8lo +
# hi` (three bits a class). The sub-classes divide a block's closed range
# where finer sources divide it (`closed_subpieces`): along a vertex-like
# dimension 0 the low face plane, 1 the lower open half, 2 the mid-plane,
# 3 the upper open half and 4 the high face plane; along a cell-like one 1
# the lower half and 3 the upper. A copy on one level writes `0:0` (the
# low plane, from the block below), `1:4` (the open interior and the high
# plane) or `0:4` (the whole own range, where no block lies below) along a
# vertex-like dimension, and `1:3` along a cell-like one; a restriction
# from a finer block writes its half of one piece, `1:2` or `3:3` across
# an open interior (the mid-plane is the lower child's), `0:0` or `4:4`
# along the dimensions it steps in, and `1:1` or `3:3` along a cell-like
# one; where sub-pieces do not form a box, each is written alone
# (`merged_spans`).
span_code(lo::Integer, hi::Integer) = Int8(8lo + hi)
span_classes(code::Integer) = (Int(code) >> 3, Int(code) & 7)

GroupKey{D}(kind::Symbol, direction::NTuple{D,Int}, offset::NTuple{D,Int},
            level::Int) where {D} =
    GroupKey{D}(kind, direction, offset, level, ntuple(_ -> Int8(0), D), Int8(0))
GroupKey{D}(kind::Symbol, direction::NTuple{D,Int}, offset::NTuple{D,Int},
            level::Int, mirror::NTuple{D,Int8}) where {D} =
    GroupKey{D}(kind, direction, offset, level, mirror, Int8(0))
GroupKey{D}(kind::Symbol, direction::NTuple{D,Int}, offset::NTuple{D,Int},
            level::Int, mirror::NTuple{D,Int8}, orientation::Integer) where {D} =
    GroupKey{D}(kind, direction, offset, level, mirror, Int8(orientation), Int8(0))

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
    (kindrank(k.kind), k.direction, k.offset, k.level, k.mirror, k.orientation, k.glue,
     k.span)
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
const UNIFY_TAG = 0                                       # stage 0 (M22)
const PHASE1_TAG = 1
prolongation_tag(level::Integer) = 2 + Int(level)       # 2 … 2 + MAX_LEVEL
interface_tag(d::Integer) = 40 + Int(d)                  # one per face dimension
const REGRID_TAG = 50                                     # the regrid transfer

# The stage a ghost transfer of group key `key` runs in, for a layout of
# element order `P` (`0` for none): the prolongations onto level ℓ in
# the stage of that level, since one may read a ghost of its coarse
# source that a coarser level's prolongation wrote, and everything else
# in phase 1. An element set's prolongations read owned points only and
# write only ghosts, which is phase 1's invariant, so they run there,
# and an element fill is one stage (M16). A transfer that writes a
# secondary copy from its primary is stage 0's, which runs before phase 1
# (M22). The one place this is decided: the local groups, the messages
# and the stage groups all ask it.
transfer_stage(key, P::Int) =
    isunify(key) ? UNIFY_TAG :
    key.kind === :prolong && P == 0 ? prolongation_tag(key.level) : PHASE1_TAG

"""
    GhostSchedule{T,D,R}

The precomputed ghost exchange for one forest, replayed by
[`fill_ghosts!`](@ref).

The phasing follows `CODE.md`:

- `unify` (M22, "Closed blocks" in `CODE.md`), for a layout with a
  vertex-like dimension, holds stage 0: the copies that give every
  secondary copy of a shared point, on a face plane of a block's closed
  box, its primary's value. Each reads a primary, which no transfer of
  the stage writes, so the stage is race free; it runs before phase 1,
  alone, and `stateunify` holds the same transfers for `unify!` on a
  state. A hook that prescribes the wall writes the wall's primaries
  before it (`BoundaryPlan.walls`, stage W).
- `phase1` holds all same-level copies and all restrictions. Each reads
  only *own* points of other blocks, final after stage 0, and writes
  only ghosts, so the whole phase is race free and order independent. For an
  element-centered set (M16) it holds the prolongations too, which read
  owned points only (see *Element sets* below).
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
- `stages` is the order [`fill_ghosts!`](@ref) runs them in: stage 0
  where there is one, phase 1, then one stage per phase-2 target level, each an
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
leaves through an outer face is the hook's. Across a glued face of a
forest of patches (M17) the transfers are the same, read through the
glue's signed axis permutation and carrying each variable onto itself.
A layout that would have a ghost region beyond a patch corner — beyond
two glued faces at once, where no one block lies across — is refused
here: a cell- or vertex-centered set wherever some patch is glued along
two dimensions, never an element set, which is filled on faces only. So
is one that the axis map does not carry onto itself, or that reverses a
vertex-like dimension (see `check_glue_layout`).

*Interpolated across a glue* (M20). A set built with `glue =
:interpolate` takes its ghosts across a glue by Lagrange interpolation
in the patch across, of the order `Operators(…; glue = k)` gives, at the
point the target patch's map names: pass the application's atlas,
`GhostSchedule(fs, operators; atlas)`, which the build calls
(`chart_map`, `chart_locate`, and `chart_aligned` where it has one) and
the fill never does. A patch corner is filled like any other ghost. A
set with a vertex-like dimension is refused over a glued forest until
M22 step 8 (see "Closed blocks" in `CODE.md`). Over a distributed forest the ranks compare the
atlas by `hash`, as the rest of the layout: an atlas holding an array or
mutable state hashes by identity, differently in every process, and
needs a `Base.hash` method of its own over its values. See "Finite
differences across a glue" in `CODE.md`.

*Element sets.* An element-centered set ([`elementcentered`](@ref),
M16), whose schedule is built with the [`ElementLocal`](@ref) family, is
filled on **faces only**: the one ghost plane across each face of a
block, and never an edge or corner ghost, which along a dimension
normal to the block can land on an element face inside a coarse block,
where two values are stored. Those ghosts are never written — no
transfer and no hook region targets them — so they hold whatever they
held, and a kernel that reads them, under `map_blocks!(…; stored =
true)` say, reads stale data. Every element transfer reads owned points
only and writes only ghosts, which is phase 1's invariant, so the
prolongations run there too: the fill is one stage, one message round,
and `phase2` and `levels` are empty.

A schedule is tied to the forest's leaf array as it was when built. It
must be rebuilt after any refinement, coarsening, or regridding.

    GhostSchedule(fs::FieldSet, operators::Operators; atlas = nothing)

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
    glue::Symbol                                 # glue mode it was built for (M20)
    operators::Operators
    backend::BK
    unify::Vector{GRP}                           # stage 0 (M22): secondaries
    phase1::Vector{GRP}
    phase2::Vector{Vector{GRP}}                  # by target level, coarsest first
    levels::Vector{Int}                          # target level of each phase2 entry
    boundaries::Vector{BoundaryRegion{D}}
    boundaryplan::BP
    stages::Vector{ST}                           # phase 1, then phase 2 by level
    stagegroups::Vector{UnitRange{Int}}          # runs of `stages` that run together
    grouplocals::Vector{Vector{GRP}}             # each run's local groups, in stage order
    phase1group::Int                             # the run that holds phase 1
    stateunify::Vector{ST}                       # stage 0 on a state, for `unify!`
    glueplan::GluePlan{T,D}                      # the ghosts interpolated across a glue

    # The constructor the build calls, which derives `grouplocals`: inside
    # it every type is static, which the build's own body is not.
    function GhostSchedule{T,D,R,BK,GRP,BP,ST}(
        forest, generation, G, centering, P, walls, glue, operators, backend, unify,
        phase1, phase2, levels, boundaries, boundaryplan, stages, stagegroups,
        stateunify, glueplan) where {T,D,R,BK,GRP,BP,ST}
        p1 = findfirst(r -> any(i -> stages[i].tag == PHASE1_TAG, r), stagegroups)::Int
        return new{T,D,R,BK,GRP,BP,ST}(forest, generation, G, centering, P, walls, glue,
                                       operators,
                                       backend, unify, phase1, phase2, levels, boundaries,
                                       boundaryplan, stages, stagegroups,
                                       group_locals(stages, stagegroups), p1,
                                       stateunify, glueplan)
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

# Target range of a transfer in dimension d: one symmetric partition of
# the stored range for every block (M22, "Closed blocks" in CODE.md), the
# low ghosts `1 … G`, the block's own range `G+1 … G+N+c` (its closed
# range: both face planes of a vertex-like dimension are its own), and
# the high ghosts `G+N+c+1 … N+2G+c`. Along a cell-like dimension that is
# the partition it always was.
#
# A halved own range is the half of it that one of two finer sources
# restricts into, `od` the source's child offset. Along a vertex-like
# dimension the two halves share the mid-plane, which the lower child
# writes, as it is the mid-plane's primary in stage 0 (M22 step 3): `G+1 …
# G+1+N/2` and `G+2+N/2 … G+N+1`.
function target_range(N::Int, G::Int, c::Int, δd::Int, od::Int, halved::Bool)
    δd == 1 && return (G + N + c + 1):(N + 2G + c)
    δd == -1 && return 1:G
    if halved
        h = N ÷ 2
        c == 1 && return od == 0 ? ((G + 1):(G + 1 + h)) : ((G + 2 + h):(G + N + 1))
        return (G + 1 + od * h):(G + (od + 1) * h)
    end
    return (G + 1):(G + N + c)
end

# The stored range of a span of sub-pieces of a block's closed box along a
# dimension (M22, `span_code`).
function piece_range(N::Int, G::Int, c::Int, code::Integer)
    lo, hi = span_classes(code)
    h = N ÷ 2
    c == 0 && return (lo == 1 ? G + 1 : G + h + 1):(hi == 1 ? G + h : G + N)
    first = (G + 1, G + 2, G + h + 1, G + h + 2, G + N + 1)[lo + 1]
    last = (G + 1, G + h, G + h + 1, G + N, G + N + 1)[hi + 1]
    return first:last
end

# The span of a whole piece, by its class.
piece_span(cls::Integer) =
    cls == 0 ? span_code(0, 0) : cls == 2 ? span_code(4, 4) : span_code(1, 3)

# Same-level copy: a pure shift of N cells against the direction, for
# every centering. A ghost reads the neighbor's own points only, which
# stage 0 has made equal to their primaries; a span of stage 0 (`span`
# its code, M22) reads the same points of its primary's block, a face
# plane the block across's opposite one, which is the same point.
function copy_stencil(::Type{T}, N::Int, G::Int, c::Int, δd::Int;
                      span::Integer=-1) where {T}
    rng = span >= 0 ? piece_range(N, G, c, span) : target_range(N, G, c, δd, 0, false)
    srcstart = Int32[j - N * δd for j in rng]
    return Stencil1D{T}(first(rng), srcstart, ones(T, 1, length(rng)))
end

# One dimension of a transfer of stage 0 (M22): a copy over the target's
# span from the same span of a block of its level (`copy_stencil`), or a
# restriction over it from a finer block, by the set's restriction of
# order `p` along a cell-like dimension (injection along a vertex-like
# one), with the multilinear weights stacked below where the group is
# bounded, as phase 1's restrictions are.
function unify_stencil(::Type{T}, N::Int, G::Int, c::Int, key, d::Int, p::Int,
                       mode::Int8) where {T}
    δd, code = key.direction[d], key.span[d]
    key.kind === :copy && return copy_stencil(T, N, G, c, δd; span=code)
    rng = piece_range(N, G, c, code)
    s = restrict_stencil_over(T, rng, N, G, c, δd, key.offset[d], p)
    mode == BOUNDED_POINTVALUE || return s
    return stack_linear(s, restrict_stencil_over(T, rng, N, G, c, δd, key.offset[d], 2))
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
# an own point of the fine block: a coarse ghost `g ≤ G` planes past the
# face is fine point `2g ≤ N` past it (the `N ≥ 2G` invariant), and a coarse
# point of the closed range the fine point there.
restrict_stencil(::Type{T}, N::Int, G::Int, c::Int, δd::Int, od::Int, p::Int) where {T} =
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
prolongation_stencil(::Type{T}, N, G, c, δd, od, ops::Operators) where {T} =
    ops.family === Conservative && c == 0 ?
    conservative_prolong_stencil(T, N, G, c, δd, od, ops.prolongation) :
    prolong_stencil(T, N, G, c, δd, od, vertex_order(ops))

# Conservative restriction is the exact volume average, which the
# point-value builder already produces at order 2: its window is a
# cell's own two children and its weights are 1/2.
restriction_stencil(::Type{T}, N, G, c, δd, od, ops::Operators) where {T} =
    restrict_stencil(T, N, G, c, δd, od, ops.restriction)

# One dimension of an unmirrored transfer of `kind`, as the ghost schedule
# and the regrid transfer both build it. With `mode` the group's bound
# (`boundedmode`), a bounded point-value group's stencil carries the
# multilinear weights below its own (`stack_linear`).
# `P` is the element order of the layout, `0` for none, and is required:
# an element set stores and owns as a cell-centered one does, so nothing
# else here can tell the two apart, and a caller that forgot it would
# build cell-centre stencils for Gauss–Lobatto nodes without a word. For
# `P > 0` the operators are the element's (`element_stencil`), and
# `regrid` picks the coarsening of a regrid over the ghost fill's
# restriction. They are called behind `Base.inferencebarrier`, as the
# element variants of the position kernels are (`with_nodes`), so that
# building a cell or vertex set's schedule does not compile them.
function kind_stencil(::Type{T}, N, G, c, kind::Symbol, δd, od, ops::Operators;
                      P::Integer, regrid::Bool=false) where {T}
    if P > 0
        c == 0 || error("an element stencil asked for with stagger $c; element sets are " *
                        "cell-like, so this is a bug")
        S = Stencil1D{T,Vector{Int32},Matrix{T}}
        return Base.inferencebarrier(element_stencil)(T, N, G, kind, δd, od, Int(P),
                                                      regrid)::S
    end
    kind === :copy && return copy_stencil(T, N, G, c, δd)
    kind === :restrict && return restriction_stencil(T, N, G, c, δd, od, ops)
    return prolongation_stencil(T, N, G, c, δd, od, ops)
end

function kind_stencil(::Type{T}, N, G, c, kind::Symbol, δd, od, ops::Operators,
                      mode::Int8; P::Integer, regrid::Bool=false) where {T}
    s = kind_stencil(T, N, G, c, kind, δd, od, ops; P=P, regrid=regrid)
    mode == BOUNDED_POINTVALUE || return s
    return stack_linear(s, kind_stencil(T, N, G, c, kind, δd, od, LINEAR_OPERATORS;
                                        P=P, regrid=regrid))
end

# One dimension of a transfer of an element-centered set (M16; see
# `ElementLocal`), in stored indices, `G ≤ 1`. Owned point `j` is node
# `(j - G - 1) % (P+1)` of element `(j - G - 1) ÷ (P+1)`, and a block of
# `N` points holds `N/(P+1)` whole elements, a fine block's half of its
# parent whole parent elements (`N % 2(P+1) == 0`).
#
# Normal to the face (`δd ≠ 0`), for every kind, the ghost is the
# neighbour's face-adjacent owned plane, a copy: the ghost sits at the
# face, and so does that plane, on the other side. The two face nodes are
# the same point, and the only ambiguity is which element's polynomial a
# value there belongs to, which the ghost resolves as the neighbour's.
#
# Tangentially (`δd = 0`) the element operators of `element_tables`:
#
# - a copy is the identity;
# - a prolongation, with `od` the *target*'s child offset, gives fine node
#   `i` of fine element `qf` in the parent's frame the coarse element
#   `qf ÷ 2`'s polynomial there: row `i` of the half `qf % 2` of `I`, over
#   that coarse element's `P+1` nodes, in the coarse source's stored
#   frame, which along a tangential dimension is the parent's;
# - a restriction, with `od` the *source*'s child offset and the target
#   range halved by it, gives coarse node `k` of coarse element `qc` row
#   `k` of `R` over the `2(P+1)` nodes of fine elements `2qc` and `2qc+1`
#   in the fine source's frame, or of the exact coarsening `C` when the
#   stencil is a regrid's (`regrid`), where energy compatibility buys
#   nothing and `C` keeps a degree `R` loses.
#
# Every window lies inside the source's owned range — the element
# operators read whole elements, and `N % 2(P+1) == 0` keeps those inside
# a block — which is what makes every element transfer phase 1's. Checked
# here, at build time.
function element_stencil(::Type{T}, N::Int, G::Int, kind::Symbol, δd::Int, od::Int,
                         P::Int, regrid::Bool) where {T}
    P1 = P + 1
    if δd != 0 || kind === :copy
        s = copy_stencil(T, N, G, 0, δd)
    elseif kind === :prolong
        t = element_tables(T, P)
        rng = target_range(N, G, 0, 0, od, false)
        srcstart = Vector{Int32}(undef, length(rng))
        weights = Matrix{T}(undef, P1, length(rng))
        for (i, j) in enumerate(rng)
            qf, node = divrem(od * N + j - G - 1, P1)   # fine element, node
            qc, half = divrem(qf, 2)                    # coarse element, half
            srcstart[i] = G + 1 + qc * P1
            weights[:, i] = view(t.interpolation[half + 1], node + 1, :)
        end
        s = Stencil1D{T}(first(rng), srcstart, weights)
    else
        t = element_tables(T, P)
        table = regrid ? t.coarsening : t.restriction
        rng = target_range(N, G, 0, 0, od, true)
        srcstart = Vector{Int32}(undef, length(rng))
        weights = Matrix{T}(undef, 2P1, length(rng))
        for (i, j) in enumerate(rng)
            qc, node = divrem(j - G - 1, P1)            # coarse element, node
            srcstart[i] = G + 1 + 2qc * P1 - od * N
            weights[:, i] = view(table, node + 1, :)
        end
        s = Stencil1D{T}(first(rng), srcstart, weights)
    end
    width = stencilorder(s)
    all(st -> G + 1 <= st && st + width - 1 <= G + N, s.srcstart) || error(
        "an element $kind stencil (δ = $δd, offset $od, P = $P, N = $N, G = $G) reads " *
        "outside its source's owned range $(G + 1):$(G + N); this is a bug")
    return s
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
# a ghost like the rows beyond it, and clamped like them.
clamp_range(N::Int, G::Int, c::Int, δd::Int) =
    δd < 0 ? (1:G) : ((G + N + 1):(N + 2G + c))

function clamp_rows(s::Stencil1D{T}, N::Int, G::Int, c::Int, δd::Int) where {T}
    c == 0 || error(step4_bug("clamped rows"))
    rng = clamp_range(N, G, c, δd)
    row = (δd < 0 ? G + 1 : G + N) - s.targetfirst + 1
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
    c == 0 || error(step4_bug("mirrored rows"))
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
    error(step4_bug("a derived wall row"))
    h = p ÷ 2
    nodes = [Rational{Int}(i) for i in vcat(-h:-1, 1:h)]
    w = lagrange_weights(nodes, 0//1)
    weights = Matrix{T}(undef, h, 1)
    for i in 1:h
        weights[i, 1] = T(2 * w[i])                # nodes -h … -1 ascending
    end
    wall = G + N + 1
    return Stencil1D{T}(wall, Int32[wall - h], weights)
end

# The mirror and the clamp above, and the derived wall row, are those of
# half-open ownership along a vertex-like dimension; M22 step 4 rebuilds
# them about the wall plane, and a set with a vertex-like dimension is
# refused over reflecting and outflow faces until then.
step4_bug(what) = "$what along a vertex-like dimension asked for; reflecting and " *
                  "outflow faces of such a set come back in M22 step 4, and " *
                  "GhostSchedule refuses them until then, so this is a bug"

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
# the real source by. Across a glued face (M17) the same, with the glue's
# index in the key in place of `r`. A region whose image leaves through
# an outer face is the hook's, as any other such region is; one that
# crosses two glued faces at once, a patch corner, finds no neighbor
# either, and `check_patch_corners` has refused every layout that has
# one before the search begins.
function block_sources!(pairs::TransferPairs{D},
                        boundaries::Vector{BoundaryRegion{D}},
                        forest::Forest{D}, G::NTuple{D,Int}, c::NTuple{D,Int},
                        b::Int, dirs; interpolated::Bool=false,
                        glueregions::Vector{GlueRegion{D}}=GlueRegion{D}[]) where {D}
    k = forest.leaves[b]
    N = forest.N
    zerooffset = ntuple(_ -> 0, D)
    for δ in dirs
        record!(kind, offset, lvl, s, via::Crossing) =
            push!.(get!(pairs, GroupKey{D}(kind, δ, offset, lvl, ntuple(_ -> Int8(0), D),
                                           via.turns, via.glue),
                        (Int32[], Int32[])),
                   (Int32(b), Int32(s)))
        region = CartesianIndices(ntuple(d -> target_range(N, G[d], c[d], δ[d], 0, false),
                                         D))
        # A cell-centered dimension with `G_d = 0` has no low slab and no
        # high one, so every direction that leaves the block along it has
        # nothing to fill. Skipping here rather than launching an empty
        # kernel also spares the tree query. (Halving a tangential range
        # cannot empty it, so testing the unhalved region covers every
        # kind.)
        isempty(region) && continue
        # Across a glue of a set interpolated there (M20): located through
        # the atlas once the search is done, not searched for here.
        if interpolated && interpolated_region(forest.conn, k, δ)
            push!(glueregions, (b, δ, region))
            continue
        end
        δ′, faces = wall_direction(forest, k, δ)
        if any(!=(OuterFace), faces)
            mirror_sources!(pairs, boundaries, forest, G, c, b, δ, δ′, faces, region)
            continue
        end
        crossing, nbrs = oriented_neighbors(forest, k, δ)
        if isempty(nbrs)
            push!(boundaries, boundary_region(forest, k, b, δ, region, G, c))
            continue
        end
        nblevel = level(first(nbrs))
        if nblevel == level(k)
            record!(:copy, zerooffset, 0, find_leaf(forest, only(nbrs)), crossing)
        elseif nblevel < level(k)
            # Coarser neighbor: this block's ghosts are prolongated. The
            # stencil geometry depends on where this block sits inside
            # its own parent.
            record!(:prolong, childoffset(k), level(k), find_leaf(forest, only(nbrs)),
                    crossing)
        else
            # Finer neighbors: each supplies one part of this block's
            # ghost region, selected by its offset within its parent —
            # its virtual parent, where this block sees it.
            for nbr in nbrs
                record!(:restrict, seam_offset(forest, nbr, crossing), 0,
                        find_leaf(forest, nbr), crossing)
            end
        end
    end
    # The secondaries of the block's closed box, from their primaries (M22).
    any(==(1), c) && unify_sources!(pairs, forest, c, b)
    return nothing
end

# --- The primary of a shared point (M22) ---------------------------------
#
# A point on a block face of a vertex-like dimension is held by every block
# whose closed box contains it. One copy is the primary, and the others
# take its value in stage 0. A block's closed box falls into at most `3^D`
# pieces, by the class of each dimension (0 low face plane, 1 open
# interior, 2 high face plane; always 1 along a cell-like dimension), and
# whether the block's copy of a point is the primary is the same at every
# point of a piece: whether a copy is the primary is decided per piece,
# one bit (`primary_bits`).
#
# The primary is the copy on the finest level holding the point (M22 rule
# 1: a coarser copy takes the fine value by the set's restriction), and
# among the copies there the one whose class tuple is greatest, dimension
# 1 first: the copy on the block lying lowest along the dimensions the
# point is shared in, for which the point is on high face planes. On one
# level the copies of a piece are its own and one across each nonempty
# subset of the faces it lies on that have a block across (a brick of
# roots, so a block across each of two faces has one across their edge),
# the class flipped, 0 to 2, along each step; distinct copies have
# distinct class tuples, so the greatest is one copy, the one high along
# every shared dimension. The direction from a secondary to its primary on
# its own level is then `-1` exactly where its class is the low plane and
# a block lies below, and never `+1`, across a periodic wrap too. A block
# that is its own neighbour, across a periodic dimension of one root,
# holds two copies; its high plane is the primary.
#
# Across levels (step 3) the blocks holding the points of one piece are
# not the same everywhere: a coarse piece facing finer blocks is covered
# by several of them, which share its mid-plane, and a hanging node of a
# fine face has no coarse copy. What is uniform along a piece is whether
# the block's copy is the primary. Under 2:1 balance, counting edges and
# corners, the blocks across a face, edge or corner of a block are one
# block of the same level or coarser, or finer blocks that tile the
# whole of what they share with it; so either every point of a piece has
# a finer copy, and the piece is secondary, or none has, and its copies
# on the block's own level, one block across each step and covering the
# piece, decide it as on one level. Coarser copies never win.
#
# Stage 0's transfers are found per **sub-piece**, which divides a piece
# where finer sources divide it: along a vertex-like dimension 0 the low
# plane, 1 the lower open half, 2 the mid-plane, 3 the upper open half, 4
# the high plane; along a cell-like one 1 the lower half and 3 the upper.
# Every point of a sub-piece has its copies on the same blocks, at the
# same classes, so its primary is one copy. The mid-plane is held by both
# finer children across a face, the lower one at its high plane, so the
# lower one is its primary, and phase 1's halved ghost ranges give it the
# mid-plane too (`target_range`).
#
# The rule is local: it asks for no index, so every rank — both ends of a
# message, and every rank's mask — decides alike by construction, and the
# secondaries of a block share their few directions, which is what lets
# stage 0 take one transfer group per direction on one level
# (`merged_spans`). Step 8 adds copies across a glue, by a patch entry of
# the key after the level.

# The index of a piece among a block's `3^D`, `Σ cls_d 3^(d-1)`.
piece_index(cls::NTuple{D,Int}) where {D} = sum(ntuple(d -> cls[d] * 3^(d - 1), D))

# The pieces of a block's closed box, by class per dimension.
closed_pieces(c::NTuple{D,Int}) where {D} =
    Iterators.product(ntuple(d -> c[d] == 1 ? (0:2) : (1:1), D)...)

# The sub-pieces of a block's closed box, by sub-class per dimension, and
# the piece a sub-piece lies in. Along a vertex-like dimension of `N = 2`
# the open halves are empty, and left out.
closed_subpieces(c::NTuple{D,Int}, N::Int) where {D} =
    Iterators.product(ntuple(d -> c[d] == 0 ? (1, 3) : N == 2 ? (0, 2, 4) : (0:4), D)...)
piece_class(sub::Integer) = sub == 0 ? 0 : sub == 4 ? 2 : 1
piece_of(sub::NTuple{D,Int}) where {D} = map(piece_class, sub)

# A sub-piece of a piece, at which the piece's primary is decided: any
# would do; the mid-plane is never empty.
piece_representative(cls::NTuple{D,Int}, c::NTuple{D,Int}) where {D} =
    ntuple(d -> c[d] == 0 ? 1 : 2 * cls[d], D)

# Every copy of the sub-piece `sub` of global leaf `j` (staggers `c`) on
# `j`'s level or a finer one, as `(level, leaf, class, direction, offset)`:
# the leaf holding it, at which piece of its own, the direction from `j`
# to it, and, for a finer leaf, its child offset in its parent (what a
# restriction from it is keyed by); `j`'s own copy first. On `j`'s level
# the copies are the neighbours across the faces the sub-piece lies on,
# the class flipped along each step; on the finer level the children
# across, chosen along each dimension the step leaves alone by which half
# the sub-piece lies in, both at the mid-plane. A coarser neighbour's copy
# never wins, and is left out.
function piece_copies(forest::Forest{D}, j::Int, sub::NTuple{D,Int},
                      c::NTuple{D,Int}) where {D}
    k = forest.leaves[j]
    zero = ntuple(_ -> 0, D)
    cls = piece_of(sub)
    copies = Tuple{Int,Int,NTuple{D,Int},NTuple{D,Int},NTuple{D,Int}}[
        (level(k), j, cls, zero, zero)]
    steps = ntuple(d -> c[d] == 1 && cls[d] != 1 ? (0, cls[d] == 0 ? -1 : 1) : (0,), D)
    for δ in Iterators.product(steps...)
        all(iszero, δ) && continue
        crossing, nbrs = oriented_neighbors(forest, k, δ)
        isempty(nbrs) && continue                   # an outer face
        crossing.turns == 0 && crossing.glue == 0 || error(
            "a copy of a point across a rotating seam or a glue asked for; those come " *
            "back in M22 steps 7 and 8, and GhostSchedule refuses them until then, so " *
            "this is a bug")
        across = ntuple(d -> δ[d] == 0 ? cls[d] : 2 - cls[d], D)
        lvl = level(first(nbrs))
        lvl < level(k) && continue                  # coarser: never the primary
        if lvl == level(k)
            push!(copies, (lvl, find_leaf(forest, only(nbrs)), across, δ, zero))
            continue
        end
        for nbr in nbrs
            o = childoffset(nbr)
            # Along a dimension the step leaves alone the child holds the
            # sub-piece when it lies in the child's half, the mid-plane in
            # both; its class there is the sub-class seen from that half.
            holds = all(d -> δ[d] != 0 || (c[d] == 0 ? o[d] == (sub[d] == 3) :
                                           sub[d] == 2 || o[d] == (sub[d] >= 3)), 1:D)
            holds || continue
            fcls = ntuple(D) do d
                δ[d] != 0 && return across[d]
                c[d] == 0 && return 1
                sub[d] == 2 && return o[d] == 0 ? 2 : 0
                return sub[d] == 0 || sub[d] == 4 ? cls[d] : 1
            end
            push!(copies, (lvl, find_leaf(forest, nbr), fcls, δ, o))
        end
    end
    return copies
end

# The primary among `piece_copies`: the finest level, then the greatest
# class tuple.
piece_primary(copies) = argmin(x -> (-x[1], map(-, x[3])), copies)

# One bit per piece of global leaf `j`'s closed box, set where its copy is
# the primary, for a set of staggers `c`: decided at one sub-piece of each
# piece, since the answer is the same along a piece.
function primary_bits(forest::Forest{D}, j::Int, c::NTuple{D,Int}) where {D}
    bits = UInt32(0)
    for cls in closed_pieces(c)
        _, s, scls, _, _ = piece_primary(piece_copies(forest, j,
                                                      piece_representative(cls, c), c))
        s == j && scls == cls && (bits |= UInt32(1) << piece_index(cls))
    end
    return bits
end

# Stage 0's transfers into global leaf `b`: every sub-piece of its closed
# box whose copy is a secondary, from its primary's — a copy from a block
# of its own level, keyed by the direction to it, or a restriction from a
# finer one, keyed by the direction to the finer block's parent node and
# the finer block's child offset in it — merged as `merged_transfers`
# says, and keyed by the spans each writes.
function unify_sources!(pairs::TransferPairs{D}, forest::Forest{D}, c::NTuple{D,Int},
                        b::Int) where {D}
    entries = Tuple{NTuple{D,Int},Symbol,Int,NTuple{D,Int},NTuple{D,Int}}[]
    lvl = level(forest.leaves[b])
    for sub in closed_subpieces(c, forest.N)
        all(==(1), piece_of(sub)) && continue       # the open interior: no copies
        slvl, s, scls, δ, o = piece_primary(piece_copies(forest, b, sub, c))
        s == b && scls == piece_of(sub) && continue
        push!(entries, (sub, slvl == lvl ? :copy : :restrict, s, δ, o))
    end
    for (kind, s, δ, o, spans) in merged_transfers(entries,
                                                   closed_subpieces(c, forest.N).iterators)
        key = GroupKey{D}(kind, δ, o, 0, ntuple(_ -> Int8(0), D), Int8(0), Int8(0), spans)
        push!.(get!(pairs, key, (Int32[], Int32[])), (Int32(b), Int32(s)))
    end
    return nothing
end

# Stage 0's merging policy, the one place it is decided: the transfers,
# as `(kind, source, direction, offset, spans)`, that write one target's
# secondary sub-pieces, `entries` as `(sub-piece, kind, source, direction,
# offset)`, with `allowed` the sub-classes each dimension has
# (`closed_subpieces`). Sub-pieces with one source, direction and offset
# merge into one transfer over their bounding box where they fill it:
# every allowed sub-class between the least and the greatest, along every
# dimension, in every combination. Copies try the whole set first, which
# on one level always fills its box, since the direction to a primary is
# decided per dimension; where it does not — the copies of a coarse block
# whose other pieces face finer blocks — each piece's sub-pieces merge,
# which always fill their box, the primary of a piece being one block.
# Restrictions merge within one piece of the target only, so that a coarse
# piece facing finer blocks takes one transfer a child, with spans that do
# not depend on what lies around it. Sorted, so that the transfers follow
# no `Dict` order.
function merged_transfers(entries::Vector{Tuple{NTuple{D,Int},Symbol,Int,NTuple{D,Int},
                                                NTuple{D,Int}}}, allowed) where {D}
    bysource = Dict{Tuple{Int,Int,NTuple{D,Int},NTuple{D,Int}},Vector{NTuple{D,Int}}}()
    for (sub, kind, s, δ, o) in entries
        push!(get!(bysource, (kindrank(kind), s, δ, o), NTuple{D,Int}[]), sub)
    end
    out = Tuple{Symbol,Int,NTuple{D,Int},NTuple{D,Int},NTuple{D,Int8}}[]
    for g in sort!(collect(keys(bysource)))
        rank, s, δ, o = g
        kind = rank == kindrank(:copy) ? :copy : :restrict
        subs = unique(bysource[g])
        spans = kind === :copy ? box_spans(subs, allowed) : nothing
        if spans === nothing
            for piece in sort!(unique(map(piece_of, subs)))
                part = filter(p -> piece_of(p) == piece, subs)
                spans = box_spans(part, allowed)
                if spans === nothing
                    for p in part
                        push!(out, (kind, s, δ, o, map(x -> span_code(x, x), p)))
                    end
                else
                    push!(out, (kind, s, δ, o, spans))
                end
            end
        else
            push!(out, (kind, s, δ, o, spans))
        end
    end
    return out
end

# The span codes of the bounding box of the distinct sub-pieces `subs`
# where they fill it, or `nothing`.
function box_spans(subs::Vector{NTuple{D,Int}}, allowed) where {D}
    lo = ntuple(d -> minimum(p -> p[d], subs), D)
    hi = ntuple(d -> maximum(p -> p[d], subs), D)
    box = prod(ntuple(d -> count(x -> lo[d] <= x <= hi[d], allowed[d]), D))
    return box == length(subs) ? ntuple(d -> span_code(lo[d], hi[d]), D) : nothing
end

# The regions of stage W (M22) on global leaf `j`, local block `b`: every
# piece of its closed box that lies on an outer face of a vertex-like
# dimension and whose copy is the primary, the piece's own range, with the
# outward direction along the outer faces it lies on. A hook that
# prescribes the wall writes these before stage 0, which carries them to
# every other copy, so that the copies of a prescribed point are one
# number. The relative hook's nearest evolved point is one plane in from
# the wall along each such dimension, `G + 2` or `G + N`: a point held by
# the same blocks as the wall point, since neither has a neighbour along
# the normal, at the same classes along every other dimension and class 1
# along the normal on each, so the rule ranks its copies as it ranks the
# wall point's, and it is a primary of this same block, which stage W does
# not write. It needs `N ≥ 2`, which every layout has.
function wall_regions!(walls::Vector{BoundaryRegion{D}}, forest::Forest{D},
                       G::NTuple{D,Int}, c::NTuple{D,Int}, j::Int, b::Int) where {D}
    k = forest.leaves[j]
    N = forest.N
    outer = ntuple(D) do d
        lo = c[d] == 1 && isempty(oriented_neighbors(forest, k,
                                                     ntuple(e -> e == d ? -1 : 0, D))[2])
        hi = c[d] == 1 && isempty(oriented_neighbors(forest, k,
                                                     ntuple(e -> e == d ? 1 : 0, D))[2])
        (lo, hi)
    end
    any(o -> o[1] || o[2], outer) || return walls
    bits = primary_bits(forest, j, c)
    for cls in closed_pieces(c)
        δ = ntuple(d -> cls[d] == 0 && outer[d][1] ? -1 : cls[d] == 2 && outer[d][2] ? 1 : 0,
                   D)
        all(iszero, δ) && continue
        (bits >> piece_index(cls)) & 0x1 == 0x1 || continue
        region = CartesianIndices(ntuple(d -> piece_range(N, G[d], c[d], piece_span(cls[d])),
                                         D))
        clamps = ntuple(d -> Int32(δ[d] < 0 ? G[d] + 2 : δ[d] > 0 ? G[d] + N : 0), D)
        push!(walls, BoundaryRegion{D}(Int32(b), δ, region, clamps, UNIFY_TAG))
    end
    return walls
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
                         region, G::NTuple{D,Int}, c::NTuple{D,Int}) where {D}
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
    # The nearest evolved point: the wall itself along a vertex-like
    # dimension, on both sides, which is the block's own (M22).
    clamps = ntuple(D) do d
        exits[d] || return Int32(0)
        return Int32(δ[d] > 0 ? G[d] + N + c[d] : G[d] + 1)
    end
    return BoundaryRegion{D}(Int32(b), δ, region, clamps, slot)
end

# The child offset of a finer source as the target sees it: its own
# offset off the seam, and across it in orientation `r` the offset turned
# into the virtual frame (M12); across a glued face (M17), turned by the
# glue's axis map.
seam_offset(forest::Forest, nbr::MortonKey, c::Crossing) =
    c.turns != 0 ? virtual_offset(childoffset(nbr), c.turns, seam_plane(forest)...) :
    c.glue != 0 ? virtual_offset(childoffset(nbr), glue_axes(forest, c)...) :
    childoffset(nbr)

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
                         walls::NTuple{D,FaceKind}, region) where {D}
    k = forest.leaves[b]
    zerooffset = ntuple(_ -> 0, D)
    mask = map(!=(OuterFace), walls)
    # A reflecting vertex-like dimension on the high side splits its
    # target into the derived wall row (state 2) and the mirrored rows
    # beyond it (state 1), which are batched apart because their parity
    # factors differ. Each part may be empty — at `G = 0` only the wall row is left — and an
    # empty part records nothing. An outflow dimension clamps its whole
    # slab, wall row included (state 3).
    choices = ntuple(D) do d
        !mask[d] && return (Int8(0),)
        walls[d] == OutflowFace && return (Int8(3),)
        c[d] == 1 && δ[d] == 1 || return (Int8(1),)
        return G[d] == 0 ? (Int8(2),) : (Int8(1), Int8(2))
    end
    states = vec(collect(Iterators.product(choices...)))
    record!(kind, offset, lvl, s, via::Crossing) =
        for state in states
            push!.(get!(pairs, GroupKey{D}(kind, δ, offset, lvl, state, via.turns,
                                           via.glue),
                        (Int32[], Int32[])),
                   (Int32(b), Int32(s)))
        end

    if δ′ == zerooffset
        record!(:copy, zerooffset, 0, b, Crossing())
        return nothing
    end
    # `δ′` may still cross a rotating seam (M12), the mirror being outside
    # its plane, or a glued face (M17): the two compose, the source turned
    # and then mirrored.
    crossing, nbrs = oriented_neighbors(forest, k, δ′)
    if isempty(nbrs)
        push!(boundaries, boundary_region(forest, k, b, δ, region, G, c))
        return nothing
    end
    nblevel = level(first(nbrs))
    if nblevel == level(k)
        record!(:copy, zerooffset, 0, find_leaf(forest, only(nbrs)), crossing)
    elseif nblevel < level(k)
        record!(:prolong, childoffset(k), level(k), find_leaf(forest, only(nbrs)),
                crossing)
    else
        # Only the children on the wall side along every masked dimension
        # cover the mirror image; the others supply the half of the
        # tangential extent it does not reach. The side is the virtual
        # one, where the target sees the child.
        wallside = ntuple(d -> δ[d] < 0 ? 0 : 1, D)
        for nbr in nbrs
            o = seam_offset(forest, nbr, crossing)
            all(d -> !mask[d] || o[d] == wallside[d], 1:D) || continue
            record!(:restrict, o, 0, find_leaf(forest, nbr), crossing)
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
# so a pack evaluates the serial transfer. `axesof(key)` is its axis map
# (`group_axes`; the identity off a seam and a glue): a turned pack (M12,
# M17) reads its source through the axis map and the variable table, as
# the serial group does, and computes the unscaled sum; its unpack applies
# the sign through the factor column, as it applies a parity.
# `targetowner(t)` and `sourceowner(s)` are the ranks of a global target and source, and
# `targetrange` / `sourcerange` this rank's own leaves in the partitions
# the targets and the sources are local to. For the ghost and interface
# exchanges both are `blockrange(forest)`; the regrid transfer (`regrid_stage`)
# has its targets in the new partition and its sources in the old.
# The bound of a group with no bound to apply (see `boundedmode`).
nobound(key) = UNBOUNDED

# The axis map a group of `key` reads its source by, as `TransferGroup`'s
# `(perm, flip)`: a seam's is `axismap` of the orientation, a glued face's
# the glue entry's own, and anything else the identity (`noaxes`, the
# default of a builder whose transfers are neither, as the regrid's).
function group_axes(forest::Forest{D}, key::GroupKey{D}) where {D}
    if key.orientation != 0
        perm, flip = axismap(key.orientation, seam_plane(forest), Val(D))
        return map(Int8, perm), flip
    end
    key.glue != 0 && return glue_axes(forest, Crossing(0, key.glue))
    return identity_axes(Val(D))
end
noaxes(key::GroupKey{D}) where {D} = identity_axes(Val(D))

# `bounded(key)` is the bound of a pack's group (`boundedmode`): the
# sender computes, so it is the pack that bounds, and the unpack stays a
# copy.
# Returns `nothing` when the stage has no messages here.
function remote_stage(::Type{RS}, ::Type{T}, backend::Backend,
                      sent::TransferPairs{D}, received::TransferPairs{D}, stencils,
                      factorcol; targetowner, sourceowner, targetrange::UnitRange{Int},
                      sourcerange::UnitRange{Int}, @nospecialize(axesof=noaxes),
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
            key.orientation, axesof(key)::Tuple{NTuple{D,Int8},NTuple{D,Bool}},
            bounded(key)::Int8)))
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
                      forest::Forest{D}; required=Int[], @nospecialize(axesof=noaxes),
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
                              sourcerange=owned, axesof=axesof, bounded=bounded))
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
# partner's array, which this rank's boxes do not describe. So does a
# level with a prolongation across a glued face (M17), turned or not:
# conservative, for the same reason, and only an edge or a corner can
# be coarse-fine across a glue. The cost is
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
# indices; `build(key)` a group's host stencils, `boundedof(key)` its
# bound and `stageof(key)` its stage (`transfer_stage`). None of the
# three is specialized on: their types carry the element type, and the
# boxes they give are the same `Box{D}` for every one, so the function is
# compiled once per dimension.
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
                      @nospecialize(boundedof), @nospecialize(stageof),
                      offset::Int) where {D}
    # Stage 0 (M22), where there is one, and phase 1 run alone, in front.
    p1 = something(findfirst(==(PHASE1_TAG), tags), 1)
    length(tags) <= p1 + 1 && return [i:i for i in eachindex(tags)]
    index = Dict{Int,Int}()
    access = Vector{StageAccess{D}}(undef, length(tags))
    for (i, tag) in enumerate(tags)
        index[tag] = i
        access[i] = StageAccess{D}()
    end
    for (lists, shift, reads, writes) in ((pairs, 0, true, true), (sent, offset, true, false),
                                          (received, offset, false, true))
        for (key, (targets, sources)) in lists
            # Phase 1 runs alone, whatever it holds: an element set's
            # prolongations are there (`transfer_stage`).
            i = get(index, stageof(key)::Int, 0)
            i <= p1 && continue
            boxes = stage_boxes(build(key), boundedof(key)::Int8)::Union{Nothing,
                                                                         NTuple{2,Box{D}}}
            boxes === nothing && continue
            a = access[i]
            key.orientation == 0 && key.glue == 0 || (a.rotated[] = true)
            reads && record_boxes!(a.reads, sources, shift, boxes[2])
            writes && record_boxes!(a.writes, targets, shift, boxes[1])
        end
    end
    for r in boundaries
        i = get(index, r.slot, 0)
        i <= p1 && continue
        push!(access[i].writes, (r.block, r.region.indices))
        push!(access[i].reads, (r.block, hook_read_box(r)))
    end

    groups = [i:i for i in 1:p1]
    written, read = BlockBoxes{D}(), BlockBoxes{D}()   # by the stages of the open group
    for i in (p1 + 1):length(tags)
        a = access[i]
        joins = i > p1 + 1 && tags[i] == tags[i - 1] + 1 && !a.rotated[] &&
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
# bound, `boundedmode`, and `stageof(key)` its stage, `transfer_stage`.
function local_groups(::Type{GRP}, ::Type{T}, ::Val{D}, backend, pairs::TransferPairs{D},
                      build, factorcol, axesof, bounded, stageof) where {GRP,T,D}
    unify, phase1 = GRP[], GRP[]
    bylevel = Dict{Int,Vector{GRP}}()
    for (key, (targets, sources)) in pairs
        # The stencils come out of a `Dict{…,Any}`.
        stencils = build(key)::NTuple{D,Stencil1D{T,Vector{Int32},Matrix{T}}}
        group = todevice(backend, TransferGroup{T,D}(key.kind, stencils, targets, sources,
                                                     factorcol(key), key.orientation,
                                                     axesof(key), bounded(key)::Int8))
        stage = stageof(key)
        if stage == UNIFY_TAG
            push!(unify, group)
        elseif stage == PHASE1_TAG
            push!(phase1, group)
        else
            push!(get!(bylevel, key.level, GRP[]), group)
        end
    end
    return unify, phase1, bylevel
end

# A ghost region that crosses two glued faces at once lies beyond a patch
# corner (M17), where three or more patches meet and no one block lies
# across: the neighbor search finds nothing there (`glued_anchor`), which
# `block_sources!` would hand to the boundary hook as though it were an
# outer face. So it is refused here, once, from the connectivity and the
# layout, before any search. An element set never reaches one: it is
# filled on faces only, and `GhostSchedule`'s filter removes every edge
# and corner direction before the search. Any other set has such a region
# at the blocks in a corner of every patch glued along two dimensions,
# unless its ghost widths leave it empty: the region is nonempty exactly
# when its range along each of the two dimensions is (`target_range`).
check_patch_corners(::OneBrick, N::Int, G, c, order::Int) = nothing
function check_patch_corners(conn::Patches{D}, N::Int, G::NTuple{D,Int},
                             c::NTuple{D,Int}, order::Int) where {D}
    (order > 0 || !conn.corners) && return nothing
    for p in eachindex(conn.roots), d in 1:D, e in (d + 1):D, sd in 1:2, se in 1:2
        conn.glue[glue_slot(p, d, sd, D)].patch != 0 || continue
        conn.glue[glue_slot(p, e, se, D)].patch != 0 || continue
        isempty(target_range(N, G[d], c[d], sd == 2 ? 1 : -1, 0, false)) && continue
        isempty(target_range(N, G[e], c[e], se == 2 ? 1 : -1, 0, false)) && continue
        throw(ArgumentError(
            "patch $p is glued at its $(sidename(sd)) face of dimension $d and at its " *
            "$(sidename(se)) face of dimension $e, so a block in the corner between " *
            "them has a ghost region beyond both faces at once: a patch corner, where " *
            "three or more patches meet and no one block lies across, which the ghost " *
            "fill does not fill. An element-centered set is filled on faces only and " *
            "never reaches one; a cell- or vertex-centered set does unless its ghost " *
            "widths leave that region empty."))
    end
    return nothing
end
sidename(s::Integer) = s == 1 ? "low" : "high"

# The layouts a glued face can carry (M17). A turned source is read with
# the target's own stored size and offsets, through the axis map, so a
# dimension must hold its points as the one it reads from does: the same
# ghost width and centering along axis `a` across the glue as along
# `perm[a]` here, as a seam asks of a set it turns into itself
# (`symmetric_layout`). And a reversal must carry owned points onto owned
# points. It does along a cell-like dimension, whose stored range is
# symmetric, but not along a vertex-like one, where a block owns its low
# wall plane and not its high one: reversed along a tangent, a ghost
# region's last plane would read the shared high plane of the block
# across, a ghost of its own, and reversed along the normal, a high face
# glued to a high face leaves the plane on it to neither block. A low face
# glued to a low face owns it on both sides, as a rotating seam does, and
# is allowed.
check_glue_layout(::OneBrick, G, centering, c) = nothing
function check_glue_layout(conn::Patches{D}, G::NTuple{D,Int}, centering::NTuple{D,Symbol},
                           c::NTuple{D,Int}) where {D}
    for p in eachindex(conn.roots), d in 1:D, s in 1:2
        entry = conn.glue[glue_slot(p, d, s, D)]
        entry.patch == 0 && continue
        face, there = (p, d, s), (entry.patch, entry.face...)
        for a in 1:D
            e = Int(entry.perm[a])
            G[a] == G[e] && centering[a] == centering[e] || throw(ArgumentError(
                "face $face is glued to face $there with axis $e of patch $p onto axis " *
                "$a of patch $(entry.patch), and this layout holds the two differently " *
                "(G = $G, centering $centering): a ghost across a glued face is read " *
                "with the block's own layout through the glue's axis map, so a " *
                "dimension and the one it is glued onto hold their points alike"))
        end
        for a in 1:D
            e = Int(entry.perm[a])
            entry.flip[a] && c[e] == 1 || continue
            e == d && s == 1 && continue
            throw(ArgumentError(e == d ?
                "face $face is glued to face $there, a high face to a high face, and " *
                "this layout holds dimension $d vertex-like: a block owns the low wall " *
                "plane of a vertex-like dimension and not the high one, so the plane on " *
                "a high face glued to a high face belongs to neither block, and nothing " *
                "fills it. Glue a high face to a low one, or hold dimension $d " *
                "cell-centered." :
                "face $face is glued to face $there with axis $e of patch $p reversed, " *
                "and this layout holds dimension $e vertex-like: a block owns the low " *
                "wall plane of a vertex-like dimension and not the high one, so a ghost " *
                "region turned across the glue would read the high plane of the block " *
                "across, a ghost of that block's own and not yet filled when it is " *
                "read. Glue the patches without reversing it, or hold dimension $e " *
                "cell-centered."))
        end
    end
    return nothing
end

# What a set with a vertex-like dimension cannot be filled over yet (M22,
# "Closed blocks"). Such a set owns its closed range, and the fill is
# rebuilt on it step by step; what a later step rebuilds is refused until
# then, with the step that brings it, rather than left half-working. Every
# condition is a property of the forest and the layout, the same on every
# rank, and the refusal is raised inside `GhostSchedule`'s
# `collective_checks`. Cell-centered and element sets have no shared
# points and are not affected. `what` names the caller.
function check_closed_blocks(forest::Forest{D}, cs::NTuple{D,Symbol}, walls::Symbol,
                             what::AbstractString) where {D}
    any(==(:vertex), cs) || return nothing
    why = "a set with a vertex-like dimension (centering $cs) owns its closed range, " *
          "both boundary planes, and its fill is being rebuilt on that layout in M22 " *
          "(\"Closed blocks\" in CODE.md)"
    walls === :symmetric && throw(ArgumentError(
        "$what: walls = :symmetric is refused for $why. Closed blocks make every wall " *
        "symmetric: an outer wall plane is the block's own on both sides, evolved by " *
        "default; a wall the boundary hook prescribes is `CellBoundary(…; wall = " *
        ":prescribed)`. The `walls` keyword goes in M22 step 4."))
    any(f -> f[1] != OuterFace || f[2] != OuterFace, getfield(forest, :faces)) &&
        throw(ArgumentError(
            "$what: the forest has a reflecting or an outflow face, which is refused " *
            "for $why: the mirror and the clamp about the wall plane come back in M22 " *
            "step 4."))
    (hasrotating(forest) || ishalfturn(forest)) && throw(ArgumentError(
        "$what: the forest has a rotating seam, which is refused for $why: quarter " *
        "and half turns, and so a RotationPair, come back in M22 step 7."))
    ispatched(forest) && has_glue(forest.conn) && throw(ArgumentError(
        "$what: the forest is a forest of patches with a glued face, which is " *
        "refused for $why: glues come back in M22 step 8."))
    return nothing
end

# Whether a forest of patches glues any face.
has_glue(conn::Patches) = any(e -> e.patch != 0, conn.glue)

GhostSchedule(fs::FieldSet{T,D}, operators::Operators; atlas=nothing) where {T,D} =
    GhostSchedule(fs.forest, operators; G=fs.G, centering=fs.centering, P=fs.P,
                  walls=fs.walls, glue=fs.glue, atlas=atlas, T=T,
                  backend=get_backend(fs.work))

function GhostSchedule(forest::Forest{D,R}, operators::Operators;
                       G::Union{Integer,Tuple{Vararg{Integer}}},
                       centering=cellcentered(D), P::Integer=0,
                       walls::Symbol=:half_open, glue::Symbol=:copy, atlas=nothing,
                       T::Type=R, backend::Backend=CPU()) where {D,R}
    N = forest.N
    # The build is collective over a distributed forest (M7): the ranks
    # agree that their forests and layouts are the same, and a refusal
    # on any of them is raised on all of them.
    checked = collective_checks(forest, "GhostSchedule") do
        gs = ghostwidths(G, Val(D))
        cs = centerings(centering, Val(D))
        ss = staggers(cs)
        storedsize(N, gs, ss)                    # the N >= 2G[d] invariant
        # `0` is the forest form's "no elements", as `FieldSet` stores it.
        order = element_order(P == 0 ? nothing : P, cs, N, gs, forest)
        check_operators(N, gs, cs, order, operators)
        check_glue(glue)
        check_glue_order(glue, order)
        if glue === :interpolate && ispatched(forest)
            # Interpolated across its glue (M20): a patch corner has a
            # source like any other ghost, and the source is read in its
            # own patch's frame, so neither refusal below applies.
            # Behind a barrier, as everything only such a set runs.
            Base.inferencebarrier(check_glue_interpolation)(forest, N, gs, ss, operators,
                                                            atlas)
            Base.inferencebarrier(check_glue_levels)(forest, blockrange(forest))
        else
            # What a forest of patches (M17) cannot fill: a region beyond a
            # patch corner, which an element set's faces-only filter below
            # never asks for, and a layout the glue's axis map does not keep.
            check_patch_corners(forest.conn, N, gs, ss, order)
            check_glue_layout(forest.conn, gs, cs, ss)
        end
        check_floattype(T, backend)
        check_walls(walls)
        # After the refusals above, which keep their own reasons (M22).
        check_closed_blocks(forest, cs, walls, "GhostSchedule")
        layout = layouthash(gs, cs, order, walls, Int(operators.family),
                            operators.prolongation,
                            operators.restriction, operators.bounded,
                            operators.vertex_prolongation, glue, operators.glue,
                            glue === :interpolate ? hash(atlas) : UInt(0), T,
                            nameof(typeof(backend)))
        return (gs, cs, ss, order), layout
    end
    ghosts::NTuple{D,Int}, centers::NTuple{D,Symbol}, stags::NTuple{D,Int}, order::Int =
        checked
    # An element set is filled on faces only (M16): along a dimension with
    # `δd ≠ 0` an edge or corner ghost can land on an element face inside
    # a coarse block, where two values are stored, and the element-local
    # schemes it serves never read one. That one filter covers the local
    # search, the transfers sent for other ranks' blocks, and the hook
    # regions — and is why an element set never meets a patch corner (M17),
    # which `check_patch_corners` has refused for every other layout.
    dirs = alldirections(Val(D))
    order > 0 && filter!(δ -> count(!=(0), δ) == 1, dirs)
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
    interpolated = glue === :interpolate && ispatched(forest)
    perglue = [GlueRegion{D}[] for _ in chunks]
    threaded_chunks(nb) do c, range
        for b in range
            block_sources!(perpairs[c], perboundaries[c], forest, ghosts, stags,
                           offset + b, dirs;
                           interpolated=interpolated, glueregions=perglue[c])
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
    # Every rank builds it, regions or not: over a distributed forest the
    # build is collective (M20).
    glueplan = interpolated ?
               Base.inferencebarrier(glue_plan)(T, forest,
                                                reduce(vcat, perglue; init=GlueRegion{D}[]),
                                                ghosts, stags, operators, atlas, owned,
                                                backend)::GluePlan{T,D} : GluePlan{T,D}()
    # Over a distributed forest (M7) the transfers split three ways: the
    # local ones stay in `pairs`, in local block indices; those with a
    # remote source are received; and those this rank computes for a
    # remote target are sent, found by the target owner's own search.
    received = split_received!(pairs, owned, nleaves(forest))
    boundaries = localize_boundaries(boundaries, owned)
    # A remote target's regions across a glue are its own rank's to
    # locate, and its owned high face planes (M20) shape what it is sent,
    # so the search for it runs in the same mode as its owner's.
    sent = sent_transfers(forest, owned) do into, j
        block_sources!(into, BoundaryRegion{D}[], forest, ghosts, stags, j, dirs;
                       interpolated=interpolated, glueregions=GlueRegion{D}[])
    end

    # One dimension of a group's stencils. Along a wall dimension of a
    # mirror transfer the stencil is the ordinary *tangential* one with
    # its target rows remapped across the wall (or clamped onto the
    # nearest owned row at an outflow face), or the derived wall row.
    boundedof(key) = boundedmode(operators, key.kind, key.mirror, stags)
    function build1(key, d)
        # A span of stage 0 (M22): a copy or a restriction into the
        # target's closed box.
        isunify(key) && return unify_stencil(T, N, ghosts[d], stags[d], key, d,
                                             operators.restriction, boundedof(key))
        δd, od, state = key.direction[d], key.offset[d], key.mirror[d]
        mode = boundedof(key)
        if state == 2
            # The derived high wall row.
            wall = wall_stencil(T, N, ghosts[d], vertex_order(operators))
            mode == BOUNDED_POINTVALUE || return wall
            return stack_linear(wall, wall_stencil(T, N, ghosts[d], 2))
        end
        # Along a wall dimension the tangential stencil is built whole and
        # its rows remapped.
        δs = state == 0 ? δd : 0
        s = kind_stencil(T, N, ghosts[d], stags[d], key.kind, δs, od, operators, mode;
                         P=order)
        state == 0 && return s
        state == 3 && return clamp_rows(s, N, ghosts[d], stags[d], δd)
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
    # only is an ordinary one to the kernel. A glued face (M17) has no
    # orientation and takes no sign, however its axes turn: a glued
    # transfer keeps 0 unless it is also mirrored.
    function factorcol(key)
        parity = map(m -> m == 3 ? Int8(0) : m, key.mirror)
        return any(!iszero, parity) || key.orientation != 0 ?
               mirrorcolumn(parity) + 3^D * Int(key.orientation) : 0
    end
    axesof(key) = group_axes(forest, key)

    # The stage of each transfer (`transfer_stage`), one function for the
    # local groups, the messages and the stage groups alike.
    stageof(key) = transfer_stage(key, order)
    GRP = grouptype(backend, T, Val(D))
    unify, phase1, bylevel = local_groups(GRP, T, Val(D), backend, pairs, build, factorcol,
                                          axesof, boundedof, stageof)

    levels = sort!(collect(keys(bylevel)))          # coarsest targets first
    phase2 = [bylevel[l] for l in levels]
    # Stage W's regions (M22), for a hook that prescribes the walls.
    wallregs = BoundaryRegion{D}[]
    if any(==(1), stags)
        for b in 1:nb
            wall_regions!(wallregs, forest, ghosts, stags, offset + b, b)
        end
    end
    bplan = BoundaryPlan(backend, T, forest, boundaries, order, wallregs)

    # Stage 0 (M22) for a layout with a vertex-like dimension, phase 1,
    # and one stage per phase-2 target level. Serially each stage is one
    # of the phases above and sends nothing.
    leveltags = prolongation_tag.(levels)
    closed = any(==(1), stags)
    localsof(tag) = tag == UNIFY_TAG ? unify : tag == PHASE1_TAG ? phase1 :
                    (i = findfirst(==(tag), leveltags); i === nothing ? GRP[] : phase2[i])
    ST = stagetype(backend, T, Val(D))
    stages = build_stages(ST, T, backend, leveltags, localsof, bystage(stageof, sent),
                          bystage(stageof, received), build, factorcol, forest;
                          required=closed ? [UNIFY_TAG, PHASE1_TAG] : [PHASE1_TAG],
                          axesof=axesof, bounded=boundedof)
    # Stage 0 again for `unify!` on a state-shaped array (M22), which has
    # no ghosts: the same transfers, groups and messages, with every
    # stencil moved down by the ghost width. Its own buffers, its own
    # stage; the tag is stage 0's, since the two never run at once.
    stateunify = ST[]
    if closed
        unghost(key) = ntuple(D) do d
            st = build(key)[d]::Stencil1D{T,Vector{Int32},Matrix{T}}
            Stencil1D{T}(st.targetfirst - ghosts[d], st.srcstart .- Int32(ghosts[d]),
                         st.weights, st.unit, st.single)
        end
        ulocal = GRP[todevice(backend, TransferGroup{T,D}(key.kind, unghost(key), targets,
                                                          sources, factorcol(key),
                                                          key.orientation, axesof(key),
                                                          boundedof(key)::Int8))
                     for (key, (targets, sources)) in pairs if stageof(key) == UNIFY_TAG]
        nopairs = TransferPairs{D}()
        owner(i) = leafowner(forest, i)
        uremote = remote_stage(remotetype(ST), T, backend,
                               get(bystage(stageof, sent), UNIFY_TAG, nopairs),
                               get(bystage(stageof, received), UNIFY_TAG, nopairs), unghost,
                               factorcol; targetowner=owner, sourceowner=owner,
                               targetrange=owned, sourcerange=owned, axesof=axesof,
                               bounded=boundedof)
        push!(stateunify, ST(UNIFY_TAG, ulocal, uremote))
    end
    # Which of them may run together here, from the host stencils.
    groups = stage_groups([st.tag for st in stages], pairs, sent, received, boundaries,
                          build, boundedof, stageof, offset)
    return GhostSchedule{T,D,R,typeof(backend),GRP,typeof(bplan),ST}(
        forest, generation(forest), ghosts, centers, order, walls, glue, operators,
        backend, unify, phase1, phase2, levels, boundaries, bplan, stages, groups,
        stateunify, glueplan)
end

function Base.show(io::IO, s::GhostSchedule{T,D}) where {T,D}
    # By kind across both phases: an element set's prolongations are in
    # phase 1 (M16).
    groups = Iterators.flatten((s.phase1, Iterators.flatten(s.phase2)))
    bykind(kind) = sum(g -> g.kind === kind ? ntransfers(g) : 0, groups; init=0)
    ncopy, nrest, nprol = bykind(:copy), bykind(:restrict), bykind(:prolong)
    # Of all of those, the ones mirrored across a reflecting face (M10)
    # and the ones rotated across a seam (M12), each said only when there
    # are any, so a schedule without either prints as it always has.
    nmirror = sum(g -> ismirrored(g) ? ntransfers(g) : 0, groups; init=0)
    nrot = sum(g -> g.orientation == 0 ? 0 : ntransfers(g), groups; init=0)
    # The ones turned across a glued face (M17) likewise.
    nturn = sum(g -> g.turned && g.orientation == 0 ? ntransfers(g) : 0, groups; init=0)
    # The copies of stage 0 (M22), said only where there are any.
    nunify = sum(ntransfers, s.unify; init=0)
    print(io, "GhostSchedule{", T, ",", D, "}(", nunify == 0 ? "" : "$nunify unifying, ",
          ncopy, " copies, ", nrest,
          " restrictions, ", nprol, " prolongations over ", length(s.phase2),
          " level(s), ", nmirror == 0 ? "" : "$nmirror mirrored transfers, ",
          nrot == 0 ? "" : "$nrot rotated transfers, ",
          nturn == 0 ? "" : "$nturn transfers turned across a glue, ",
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
