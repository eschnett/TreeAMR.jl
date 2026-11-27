# Ghosts interpolated across a glue (M20).
#
# A field set built with `glue = :interpolate` over a forest of patches
# is filled across a glued face by interpolation in the patch across, at
# the physical point the target's own map names past its box, located
# through the application's atlas when the schedule is built. Each fill
# then evaluates one tensor-product Lagrange stencil per ghost point and
# variable. See "Finite differences across a glue" in CODE.md, which this
# file implements serially and over a distributed forest, on the CPU and
# on a device. On closed blocks (M22) every patch holding a point of a
# glued face holds a copy of it, and stage A, at the end of this file,
# makes the copies their mean.

"""
    TreeAMR.chart_aligned(atlas, p, q) -> NTuple{D,Bool}

Along which dimensions of patch `q` a point at a node of patch `p`'s grid
lands exactly on a node of `q`'s grid at the same level: the radius between
two of a thirteen-patch ball's outer shells, both angles between an inner
and an outer shell over one cube face, both coordinates across an
annulus's glue. An optional method of an atlas (M20). A field set filled
across a glue by interpolation then reads no neighbor along such a
dimension: the schedule checks every ghost against the declaration, to a
thousandth of a spacing, and takes the node exactly, so that the stencil
there has width one. Without the method nothing is aligned. See "Finite
differences across a glue" in `CODE.md`.
"""
function chart_aligned end

# A ghost region across a glue, as the search finds it: the global leaf,
# the direction and the stored indices.
const GlueRegion{D} = Tuple{Int,NTuple{D,Int},CartesianIndices{D,NTuple{D,UnitRange{Int}}}}

# The entries of a list grouped by the local block they belong to, for the
# launch by owner: their order, stable, and where each block's run starts,
# with the length of the longest run.
struct GlueRuns
    order::Vector{Int32}
    start::Vector{Int32}
    longest::Int
end
GlueRuns() = GlueRuns(Int32[], Int32[1], 0)
function GlueRuns(blocks::Vector{Int32}, nb::Int)
    counts = zeros(Int, nb)
    foreach(b -> counts[b] += 1, blocks)
    start = Int32.(cumsum([1; counts]))
    return GlueRuns(Int32.(sortperm(blocks; alg=Base.Sort.DEFAULT_STABLE)), start,
                    maximum(counts; init=0))
end

# The interpolated ghosts of one schedule. Those whose source block is on
# this rank (every one, serially): per ghost point the target block and
# stored index, the source block, the stored index of the stencil's first
# point and its width per dimension (`k`, or 1 along an aligned
# dimension), and its weights, `k` per dimension, column `j`. Over a
# distributed forest (M20) the sender computes: the stencils this rank
# evaluates for other ranks' ghosts (`send…`, in the order the targets'
# ranks located them, peer by peer), and the ghosts of this rank's blocks
# whose values other ranks send (`recv…`, in the order this rank located
# them), each fill one message per peer and direction under `GLUE_TAG`,
# point by point with the variables fastest. On a device the same tables
# are uploaded once, at the build (`device`), and the message buffers live
# there, with host mirrors where the communicator stages (`buffers`). One
# concrete type whether empty or not, so that a schedule's type does not
# say whether it interpolates.
struct GluePlan{T,D}
    k::Int
    target::Vector{Int32}
    index::Vector{NTuple{D,Int32}}
    source::Vector{Int32}
    first::Vector{NTuple{D,Int32}}
    width::Vector{NTuple{D,Int8}}
    weights::Matrix{T}
    sendsource::Vector{Int32}
    sendfirst::Vector{NTuple{D,Int32}}
    sendwidth::Vector{NTuple{D,Int8}}
    sendweights::Matrix{T}
    sendpeers::Vector{Int}
    sendcounts::Vector{Int}
    recvtarget::Vector{Int32}
    recvindex::Vector{NTuple{D,Int32}}
    recvpeers::Vector{Int}
    recvcounts::Vector{Int}
    reads::Vector{Int}
    localruns::GlueRuns
    sendruns::GlueRuns
    recvruns::GlueRuns
    device::Any
    buffers::Any
end
GluePlan{T,D}() where {T,D} =
    GluePlan{T,D}(0, Int32[], NTuple{D,Int32}[], Int32[], NTuple{D,Int32}[],
                  NTuple{D,Int8}[], zeros(T, 0, 0), Int32[], NTuple{D,Int32}[],
                  NTuple{D,Int8}[], zeros(T, 0, 0), Int[], Int[], Int32[],
                  NTuple{D,Int32}[], Int[], Int[], Int[], GlueRuns(), GlueRuns(),
                  GlueRuns(), nothing, GlueBuffers())

# The message buffers of a glue plan, on the backend, and the host mirrors
# a staging communicator sends from and receives into; sized for the
# variable count of the last fill.
mutable struct GlueBuffers
    send::Any
    recv::Any
    wiresend::Any
    wirerecv::Any
end
GlueBuffers() = GlueBuffers(nothing, nothing, nothing, nothing)

# The tables the kernels read, uploaded to `backend`, under the plan's
# own field names.
function upload_glue(plan::GluePlan, backend)
    dev(a) = todevice(backend, a)
    return (target=dev(plan.target), index=dev(plan.index), source=dev(plan.source),
            first=dev(plan.first), width=dev(plan.width), weights=dev(plan.weights),
            sendsource=dev(plan.sendsource), sendfirst=dev(plan.sendfirst),
            sendwidth=dev(plan.sendwidth), sendweights=dev(plan.sendweights),
            recvtarget=dev(plan.recvtarget), recvindex=dev(plan.recvindex))
end

# The buffers for `n` points of `nv` variables each way, made or remade
# when the size changes.
function glue_buffers!(plan::GluePlan{T}, backend, comm, nv::Int) where {T}
    bufs = plan.buffers::GlueBuffers
    nsend, nrecv = length(plan.sendsource) * nv, length(plan.recvtarget) * nv
    if bufs.send === nothing || length(bufs.send) != nsend || length(bufs.recv) != nrecv
        bufs.send = allocate(backend, T, nsend)
        bufs.recv = allocate(backend, T, nrecv)
        staged = hoststaging(comm, bufs.send)
        bufs.wiresend = staged ? Vector{T}(undef, nsend) : bufs.send
        bufs.wirerecv = staged ? Vector{T}(undef, nrecv) : bufs.recv
    end
    return bufs
end
Base.isempty(plan::GluePlan) =
    isempty(plan.target) && isempty(plan.sendsource) && isempty(plan.recvtarget)
# The ghosts of this rank's blocks it fills, computed here or received.
Base.length(plan::GluePlan) = length(plan.target) + length(plan.recvtarget)

# The message tag of the interpolated ghosts: a round of its own between
# phase 1 and phase 2, past the prolongation tags (CODE.md, "Deadlock
# freedom and message matching").
const GLUE_TAG = 36

# The faces of its patch a step `δ` from leaf `k` leaves through: per
# dimension 0 where it stays, 1 through the low face, 2 through the high.
function patch_exits(conn::Patches{D}, k::MortonKey{D}, δ::NTuple{D,Int}) where {D}
    lvl = level(k)
    p = searchsortedlast(conn.firstroot, Int(k.root))
    n = conn.roots[p]
    rootpos = root_position(n, Int(k.root) - conn.firstroot[p])
    return ntuple(D) do e
        g = (rootpos[e] << lvl) + Int(k.coords[e]) + δ[e]
        return g < 0 ? 1 : g >= n[e] << lvl ? 2 : 0
    end
end

isglued(conn::Patches{D}, p::Integer, e::Integer, side::Integer) where {D} =
    conn.glue[glue_slot(p, e, side, D)].patch != 0

# The glued faces of its patch that leaf `k` lies on, as `(dim, side)`.
function glued_faces(conn::Patches{D}, k::MortonKey{D}) where {D}
    lvl = level(k)
    p = searchsortedlast(conn.firstroot, Int(k.root))
    n = conn.roots[p]
    rootpos = root_position(n, Int(k.root) - conn.firstroot[p])
    g = ntuple(e -> (rootpos[e] << lvl) + Int(k.coords[e]), D)
    return [(e, s) for e in 1:D for s in 1:2
            if (s == 1 ? g[e] == 0 : g[e] == (n[e] << lvl) - 1) && isglued(conn, p, e, s)]
end

# Whether the ghost region of leaf `k` in direction `δ` lies across a
# glue and nowhere else: it leaves `k`'s patch, through glued faces only.
# Such a region is interpolated; one that leaves through an outer face
# too is the boundary hook's, as beyond any outer face.
function interpolated_region(conn::Patches{D}, k::MortonKey{D},
                             δ::NTuple{D,Int}) where {D}
    ex = patch_exits(conn, k, δ)
    all(iszero, ex) && return false
    p = searchsortedlast(conn.firstroot, Int(k.root))
    return all(e -> ex[e] == 0 || isglued(conn, p, e, ex[e]), 1:D)
end

# The refusal of a refinement boundary at a glue (M20): a face, edge or
# corner between leaves of two levels that touches a glued face, on
# either side. Checked over this rank's blocks; the build agrees on it
# across the ranks. Along a glued face (e, side) that block `k` touches,
# the interface in direction `δ` touches it unless `δ` points away from
# it.
function check_glue_levels(forest::Forest{D}, range) where {D}
    conn = forest.conn::Patches{D}
    for b in range
        k = forest.leaves[b]
        p = patch(forest, k)
        lvl = level(k)
        touched = glued_faces(conn, k)
        isempty(touched) && continue
        for δ in alldirections(Val(D))
            any(((e, s),) -> δ[e] != (s == 1 ? 1 : -1), touched) || continue
            nbrs = oriented_neighbors(forest, k, δ)[2]
            isempty(nbrs) && continue
            level(first(nbrs)) == lvl && continue
            e, s = first(((e, s) for (e, s) in touched if δ[e] != (s == 1 ? 1 : -1)))
            throw(ArgumentError(
                "a refinement boundary touches a glued face: leaf $b, $k, in patch $p, " *
                "lies on the $(s == 1 ? "low" : "high") face of dimension $e, which is " *
                "glued, and its neighbor $(first(nbrs)) in direction $δ is at level " *
                "$(level(first(nbrs))), not $lvl. A set interpolated across a glue " *
                "needs every block within reach of the glue at one level, so that no " *
                "ghost is interpolated across levels (CODE.md, \"Finite differences " *
                "across a glue\"); keep the refinement away from the patches' glued " *
                "faces"))
        end
    end
    return nothing
end

# The checks of an interpolated set's schedule that need no tree search,
# against its layout, its operators and its atlas.
function check_glue_interpolation(forest::Forest{D,R}, N::Int, G::NTuple{D,Int},
                                  c::NTuple{D,Int}, ops::Operators, atlas) where {D,R}
    k = ops.glue
    k > 0 || throw(ArgumentError(
        "a field set with glue = :interpolate needs the order of the interpolation " *
        "across a glue: pass `Operators(…; glue = k)`. There is no default, since " *
        "what a derivative stencil needs from its ghosts is the scheme's to say: a " *
        "second derivative of order 4 needs k = 6 (CODE.md, \"Finite differences " *
        "across a glue\")."))
    atlas === nothing && throw(ArgumentError(
        "a field set with glue = :interpolate over a forest of patches needs the " *
        "application's atlas: pass `GhostSchedule(fs, operators; atlas)`. A ghost " *
        "across a glue lies where the target patch's map puts it, which only the " *
        "atlas knows."))
    check_chart(chart_map, atlas, R, Val(D),
                "a field set interpolated across a glue, whose ghosts lie where the map puts them,")
    check_chart(chart_locate, atlas, R, Val(D),
                "a field set interpolated across a glue, which locates its ghosts,")
    for d in 1:D
        k <= 2 * (G[d] + c[d]) || throw(ArgumentError(
            "the glue order $k does not fit the field set's stored array along " *
            "dimension $d: a centered stencil of k points about a point of the source " *
            "block reaches k ÷ 2 points beyond its owned range, so k ≤ 2(G + c) = " *
            "$(2 * (G[d] + c[d])) with G = $(G[d]) and c = $(c[d]) there"))
    end
    any(f -> any(!=(OuterFace), f), getfield(forest, :faces)) && throw(ArgumentError(
        "a field set interpolated across a glue over a forest with reflecting or " *
        "outflow faces is not implemented: the edge regions beyond such a face and a " *
        "glued one are neither mirrored nor located yet (M20 step 4)"))
    return nothing
end

# The Lagrange weights of the points `first … first + k - 1` at `t`, in
# the arithmetic type `S`, rounded once into `T`, into `w[at + 1 … at + k]`.
function glue_weights!(w::Vector{T}, at::Int, ::Type{S}, t, first::Int, k::Int) where {T,S}
    ts = S(t)
    for i in 1:k
        x = one(S)
        for j in 1:k
            j == i && continue
            x *= (ts - (first + j - 1)) / S(i - j)
        end
        w[at + i] = T(x)
    end
    return w
end

# One interpolated ghost as the target block's rank locates it: the
# target's global leaf, stored index and patch, the source's global leaf
# and the position in the source's patch. Plain bits, so that over a
# distributed forest the build's one `alltoallv` hands it to the source's
# owner, who builds the stencil from it exactly as the target's rank did.
struct GlueGhost{D,R}
    target::Int
    index::NTuple{D,Int32}
    from::Int32
    source::Int
    ξ::NTuple{D,R}
end

# The glue plan of one schedule: every interpolated region `(b, δ,
# region)` of this rank's blocks — `b` a global leaf, `region` stored
# indices — located through the atlas, with its stencil. Refuses, saying
# why, a ghost the atlas puts in no patch, one whose source is at another
# level, an alignment that does not hold, and a stencil that would read a
# point that phase 1 and the hook leave unfilled. Every rank comes here, whether it has regions or not: over a
# distributed forest the refusals are agreed on (`collective_checks`) and
# the ghosts whose source another rank owns go to that rank in one
# `alltoallv`, which builds their stencils and evaluates them each fill.
function glue_plan(::Type{T}, forest::Forest{D,R}, regions, G::NTuple{D,Int},
                   c::NTuple{D,Int}, ops::Operators, atlas, owned::UnitRange{Int},
                   backend::Backend) where {T,D,R}
    offset = first(owned) - 1
    k = ops.glue
    # Asked once, not per ghost: `hasmethod` costs a method lookup.
    aligned = hasmethod(chart_aligned, Tuple{typeof(atlas),Int,Int})
    # Located and checked in threaded chunks of regions, each with its
    # own buffers and its own cache of what the stencils read, and
    # concatenated in chunk order, so that the plan is the same at any
    # thread count; a refusal is the first chunk's that has one.
    located = collective_checks(forest, "GhostSchedule") do
        nr = length(regions)
        parts = Vector{Any}(undef, length(threadchunks(nr)))
        threaded_chunks(nr) do ci, range
            ghosts = locate_glue_ghosts(forest, view(regions, range), G, c, atlas)
            parts[ci] = (ghosts, glue_stencils(T, forest, atlas, aligned, G, c, k, ghosts)...)
        end
        isempty(parts) && return (GlueGhost{D,R}[], NTuple{D,Int32}[], NTuple{D,Int8}[], T[],
                                  Int[]), UInt(0)
        ghosts = reduce(vcat, (x[1] for x in parts))
        firsts = reduce(vcat, (x[2] for x in parts))
        widths = reduce(vcat, (x[3] for x in parts))
        weights = reduce(vcat, (x[4] for x in parts))
        reads = sort!(unique!(reduce(vcat, (x[5] for x in parts))))
        return (ghosts, firsts, widths, weights, reads), UInt(0)
    end
    ghosts, allfirst, allwidth, allweights, reads = located
    W = reshape(allweights, k * D, length(ghosts))
    # The ghosts this rank computes, and those it asks other ranks for,
    # counted by owner and kept in the order located.
    P = commsize(forest.comm)
    mine = [i for i in eachindex(ghosts) if ghosts[i].source in owned]
    asked = GlueGhost{D,R}[]
    counts = zeros(Int, P)
    if P > 1
        theirs = [i for i in eachindex(ghosts) if !(ghosts[i].source in owned)]
        owner = [leafowner(forest, ghosts[i].source) for i in theirs]
        perm = sortperm(owner; alg=Base.Sort.DEFAULT_STABLE)
        asked = ghosts[theirs[perm]]
        foreach(o -> counts[o + 1] += 1, owner)
    end
    # What the other ranks ask this one for, in their rank order.
    received, rcounts = P > 1 ? alltoallv(forest.comm, asked, counts) :
                        (GlueGhost{D,R}[], Int[0])
    sfirst, swidth, sweights, _ = glue_stencils(T, forest, atlas, aligned, G, c, k, received)
    nonzero(cs) = [r - 1 for r in eachindex(cs) if cs[r] > 0]
    plan = GluePlan{T,D}(k,
                         Int32[ghosts[i].target - offset for i in mine],
                         NTuple{D,Int32}[ghosts[i].index for i in mine],
                         Int32[ghosts[i].source - offset for i in mine],
                         allfirst[mine], allwidth[mine], W[:, mine],
                         Int32[g.source - offset for g in received],
                         sfirst, swidth, reshape(sweights, k * D, length(received)),
                         nonzero(rcounts), filter(>(0), rcounts),
                         Int32[g.target - offset for g in asked],
                         NTuple{D,Int32}[g.index for g in asked],
                         nonzero(counts), filter(>(0), counts), reads,
                         GlueRuns(), GlueRuns(), GlueRuns(), nothing, GlueBuffers())
    if backend isa CPU
        nb = length(owned)
        return GluePlan{T,D}((getfield(plan, f) for f in fieldnames(GluePlan)[1:18])...,
                             GlueRuns(plan.target, nb), GlueRuns(plan.sendsource, nb),
                             GlueRuns(plan.recvtarget, nb), nothing, GlueBuffers())
    end
    return GluePlan{T,D}((getfield(plan, f) for f in fieldnames(GluePlan)[1:(end - 2)])...,
                         upload_glue(plan, backend), GlueBuffers())
end

# The logical position of stored index `idx` of leaf `key`.
function ghost_position(::Type{R}, forest::Forest{D}, key::MortonKey{D},
                        idx::NTuple{D,<:Integer}, G::NTuple{D,Int},
                        c::NTuple{D,Int}) where {R,D}
    h = spacing(R, forest, level(key))
    lo, B = block_low(R, forest, key), block_base(forest, key)
    return ntuple(d -> nodeposition(lo[d], h, B[d], Int(idx[d]) - G[d], c[d], nothing), D)
end

# Every interpolated ghost of `regions`, located: the atlas names the
# patch and the position there, the search the source leaf, which must
# be at the target's level.
function locate_glue_ghosts(forest::Forest{D,R}, regions, G::NTuple{D,Int},
                            c::NTuple{D,Int}, atlas) where {D,R}
    out = GlueGhost{D,R}[]
    for (b, δ, region) in regions
        key = forest.leaves[b]
        p = patch(forest, key)
        lvl = level(key)
        for I in region
            idx = Tuple(I)
            ξ = ghost_position(R, forest, key, idx, G, c)
            x = chart_map(atlas, ξ, p)
            loc = chart_locate(atlas, x)
            loc === nothing && throw(ArgumentError(
                "the atlas $(typeof(atlas)) puts the ghost at $ξ of patch $p, at $x, in " *
                "no patch, though it lies across a glued face of patch $p in direction " *
                "$δ: a map must be defined past its patch's box, by the ghost width at " *
                "least (CODE.md, \"Finite differences across a glue\")"))
            # The patch across, or `p` itself across a face glued to its own
            # opposite face, a wrap. A point on a glued face may be given to
            # either side: both own their face planes (closed ownership).
            ξ′, q = loc
            s = locate_point(forest, PatchPoint(ξ′, q))
            s === nothing && throw(ArgumentError(
                "no leaf of patch $q holds $ξ′, where the atlas locates the ghost at $ξ " *
                "of patch $p"))
            skey = forest.leaves[s]
            level(skey) == lvl || throw(ArgumentError(
                "the ghost at $ξ of patch $p (level $lvl) lies in leaf $skey of patch " *
                "$q, at level $(level(skey)): a set interpolated across a glue needs " *
                "every block within reach of the glue at one level"))
            push!(out, GlueGhost{D,R}(b, ntuple(d -> Int32(idx[d]), D), Int32(p), s,
                                      ntuple(d -> R(ξ′[d]), D)))
        end
    end
    return out
end

# The stencils of the located ghosts `ghosts`: per ghost the stored index
# of its first point in the source block and its width per dimension,
# its weights, `k` per dimension, one after the other in one vector, and
# the leaves the stencils read (their sources, and the neighbors whose
# ghosts they read). A function of the ghosts and the replicated tree
# alone, so that the source's owner builds what the target's rank
# checked, bit for bit. What a stencil reads beyond its source is checked
# once per source block and direction (`cache`), not per ghost.
function glue_stencils(::Type{T}, forest::Forest{D}, atlas, aligned::Bool,
                       G::NTuple{D,Int}, c::NTuple{D,Int}, k::Int, ghosts) where {T,D}
    firsts = Vector{NTuple{D,Int32}}(undef, length(ghosts))
    widths = Vector{NTuple{D,Int8}}(undef, length(ghosts))
    weights = zeros(T, k * D * length(ghosts))
    cache = Dict{Tuple{Int,NTuple{D,Int}},Vector{Int}}()
    for (j, g) in enumerate(ghosts)
        firsts[j], widths[j] = glue_stencil!(weights, (j - 1) * k * D, cache, forest, atlas,
                                             aligned, G, c, k, g)
    end
    reads = Int[]
    for ((s, _), nbrs) in cache
        push!(reads, s)
        append!(reads, nbrs)
    end
    return firsts, widths, weights, reads
end

# One located ghost's stencil, its weights into `weights[at + 1 … at + kD]`.
function glue_stencil!(weights::Vector{T}, at::Int, cache, forest::Forest{D,R}, atlas,
                       aligned::Bool, G::NTuple{D,Int}, c::NTuple{D,Int}, k::Int,
                       g::GlueGhost{D}) where {T,D,R}
    N = forest.N
    S = promote_type(T, Float64)
    p = Int(g.from)
    skey = forest.leaves[g.source]
    q = patch(forest, skey)
    lvl = level(skey)
    h = spacing(R, forest, lvl)
    ξ′ = g.ξ
    al = aligned ? chart_aligned(atlas, p, q) : ntuple(_ -> false, D)
    lo = patch_extents(forest, q)
    npts = patch_roots(forest, q)
    spos = root_position(forest, skey.root)
    start = ntuple(e -> ((spos[e] << lvl) + Int(skey.coords[e])) * N, D)
    where_() = (ghost_position(R, forest, forest.leaves[g.target], g.index, G, c), p)
    fs = ntuple(_ -> Int32(0), D)
    ws = ntuple(_ -> Int8(0), D)
    overhang = ntuple(_ -> 0, D)
    for e in 1:D
        t = (S(ξ′[e]) - S(lo[e][1])) / S(h) - (c[e] == 1 ? 0 : S(1) / 2)
        n = (npts[e] << lvl) * N
        # The window: the patch's own points, its high face plane included
        # along a vertex-like dimension, every block's own (M22).
        hi = c[e] == 1 ? n : n - 1
        if al[e]
            r = round(Int, t)
            abs(t - r) <= S(1) / 1000 || throw(ArgumentError(
                "the atlas $(typeof(atlas)) declares dimension $e of patch $q aligned " *
                "with patch $p (chart_aligned), but the ghost at $(where_()[1]) of patch " *
                "$p lands at $(t) spacings along it, not at a node"))
            f, w = r, 1
            weights[at + (e - 1) * k + 1] = one(T)
        else
            hi - k + 1 >= 0 || throw(ArgumentError(
                "patch $q has $(hi + 1) points along dimension $e at level $lvl, fewer " *
                "than the glue order $k"))
            f = clamp(floor(Int, t) - k ÷ 2 + 1, 0, hi - k + 1)
            w = k
            glue_weights!(weights, at + (e - 1) * k, S, t, f, k)
        end
        first = f - start[e] + G[e] + 1
        last = first + w - 1
        1 <= first && last <= N + 2G[e] + c[e] || throw(ArgumentError(
            "the stencil of the ghost at $(where_()[1]) of patch $p reaches outside the " *
            "stored array of its source block $skey along dimension $e"))
        # Beyond the own range, closed along a vertex-like dimension (M22):
        # the ghosts, which the neighbor within the patch fills.
        overhang = Base.setindex(overhang,
                                 first <= G[e] ? -1 : last > G[e] + N + c[e] ? 1 : 0, e)
        fs = Base.setindex(fs, Int32(first), e)
        ws = Base.setindex(ws, Int8(w), e)
    end
    haskey(cache, (g.source, overhang)) ||
        (cache[(g.source, overhang)] = check_overhang(forest, skey, overhang, where_()...))
    return fs, ws
end

# What a stencil reads beyond its source block's owned range is the
# source's ghosts inside its own patch, which phase 1 fills before this
# stage: a copy from a neighbor at the same level, a restriction from finer
# ones. Refused if a neighbor in one of those directions is missing (the
# stencil would read across a glue) or coarser, whose prolongation comes
# in phase 2, after this stage. Returns the neighbors' leaves, which
# `regrid!` keeps at their levels.
function check_overhang(forest::Forest{D}, skey::MortonKey{D}, overhang, ξ, p) where {D}
    reads = Int[]
    for δ in alldirections(Val(D))
        all(e -> δ[e] == 0 || δ[e] == overhang[e], 1:D) || continue
        crossing, nbrs = oriented_neighbors(forest, skey, δ)
        ok = !isempty(nbrs) && all(nb -> level(nb) >= level(skey), nbrs) &&
             !iscrossing(crossing)
        ok || throw(ArgumentError(
            "the stencil of the ghost at $ξ of patch $p reads the ghosts of its source " *
            "block $skey in direction $δ, which no block of the same level or finer " *
            "in its patch fills before the ghosts across the glue are: the ghost lies " *
            "too deep in the patch across, or a coarser block is within reach of the " *
            "glue"))
        foreach(nb -> push!(reads, find_leaf(forest, nb)), nbrs)
    end
    return reads
end

# The value of the stencil `j` for variable `v`: the tensor product of its
# weights over the source block `s`'s points (M20), summed one dimension
# at a time, the first innermost, each loop as long as the stencil's width
# there — one along an aligned dimension, so that a ghost with shared
# coordinates costs its taps and not `K^D`. The same arithmetic for a
# ghost filled in place and for one evaluated into a message, on every
# backend, so that both are the serial value bit for bit.
@inline function glue_value(work, s, f, w, weights, j, v, ::Val{1}, ::Val{K}) where {K}
    acc = zero(eltype(work))
    @inbounds for i1 in 1:Int(w[1])
        acc += weights[i1, j] * work[Int(f[1]) + i1 - 1, v, s]
    end
    return acc
end
@inline function glue_value(work, s, f, w, weights, j, v, ::Val{2}, ::Val{K}) where {K}
    acc = zero(eltype(work))
    @inbounds for i2 in 1:Int(w[2])
        a1 = zero(eltype(work))
        for i1 in 1:Int(w[1])
            a1 += weights[i1, j] * work[Int(f[1]) + i1 - 1, Int(f[2]) + i2 - 1, v, s]
        end
        acc += weights[K + i2, j] * a1
    end
    return acc
end
@inline function glue_value(work, s, f, w, weights, j, v, ::Val{3}, ::Val{K}) where {K}
    acc = zero(eltype(work))
    @inbounds for i3 in 1:Int(w[3])
        a2 = zero(eltype(work))
        for i2 in 1:Int(w[2])
            a1 = zero(eltype(work))
            for i1 in 1:Int(w[1])
                a1 += weights[i1, j] *
                      work[Int(f[1]) + i1 - 1, Int(f[2]) + i2 - 1, Int(f[3]) + i3 - 1, v, s]
            end
            a2 += weights[K + i2, j] * a1
        end
        acc += weights[2K + i3, j] * a2
    end
    return acc
end

# On the CPU the three kernels run by owner, as every per-block pass does:
# one workgroup per block, its entries the block's (`order[start[b] …
# start[b + 1] - 1]`), every variable of an entry in turn. A local ghost
# and a received one belong to their target block, a ghost evaluated for
# another rank to its source block.
@kernel function glue_owner_kernel!(work, @Const(order), @Const(start), @Const(target),
                                    @Const(index), @Const(source), @Const(firsts),
                                    @Const(widths), @Const(weights), nvars,
                                    ::Val{D}, ::Val{K}) where {D,K}
    i, b = @index(Global, NTuple)
    @inbounds begin
        r = Int(start[b]) + i - 1
        if r < Int(start[b + 1])
            j = Int(order[r])
            idx = ntuple(d -> Int(index[j][d]), Val(D))
            for v in 1:nvars
                work[idx..., v, Int(target[j])] =
                    glue_value(work, Int(source[j]), firsts[j], widths[j], weights, j, v,
                               Val(D), Val(K))
            end
        end
    end
end
@kernel function glue_owner_pack_kernel!(buf, @Const(work), @Const(order), @Const(start),
                                         @Const(source), @Const(firsts), @Const(widths),
                                         @Const(weights), nvars,
                                         ::Val{D}, ::Val{K}) where {D,K}
    i, b = @index(Global, NTuple)
    @inbounds begin
        r = Int(start[b]) + i - 1
        if r < Int(start[b + 1])
            j = Int(order[r])
            for v in 1:nvars
                buf[(j - 1) * nvars + v] =
                    glue_value(work, Int(source[j]), firsts[j], widths[j], weights, j, v,
                               Val(D), Val(K))
            end
        end
    end
end
@kernel function glue_owner_unpack_kernel!(work, @Const(buf), @Const(order), @Const(start),
                                           @Const(target), @Const(index), nvars,
                                           ::Val{D}) where {D}
    i, b = @index(Global, NTuple)
    @inbounds begin
        r = Int(start[b]) + i - 1
        if r < Int(start[b + 1])
            j = Int(order[r])
            idx = ntuple(d -> Int(index[j][d]), Val(D))
            for v in 1:nvars
                work[idx..., v, Int(target[j])] = buf[(j - 1) * nvars + v]
            end
        end
    end
end


# A ghost whose source is on this rank: one work item per point and
# variable, each writing its own slot.
@kernel function glue_kernel!(work, @Const(target), @Const(index), @Const(source),
                              @Const(firsts), @Const(widths), @Const(weights),
                              ::Val{D}, ::Val{K}) where {D,K}
    j, v = @index(Global, NTuple)
    @inbounds begin
        acc = glue_value(work, Int(source[j]), firsts[j], widths[j], weights, j, v,
                         Val(D), Val(K))
        work[ntuple(d -> Int(index[j][d]), Val(D))..., v, Int(target[j])] = acc
    end
end

# A ghost of another rank's block, evaluated into the send buffer, the
# variables of a point together.
@kernel function glue_pack_kernel!(buf, @Const(work), @Const(source), @Const(firsts),
                                   @Const(widths), @Const(weights), nvars,
                                   ::Val{D}, ::Val{K}) where {D,K}
    j, v = @index(Global, NTuple)
    @inbounds buf[(j - 1) * nvars + v] =
        glue_value(work, Int(source[j]), firsts[j], widths[j], weights, j, v, Val(D),
                   Val(K))
end

# A received ghost, copied into place.
@kernel function glue_unpack_kernel!(work, @Const(buf), @Const(target), @Const(index),
                                     nvars, ::Val{D}) where {D}
    j, v = @index(Global, NTuple)
    @inbounds work[ntuple(d -> Int(index[j][d]), Val(D))..., v, Int(target[j])] =
        buf[(j - 1) * nvars + v]
end

# The fill of the interpolated ghosts, after phase 1 and the boundary hook:
# the receives posted, the sends evaluated and sent, the local ghosts
# filled in place while the messages travel, then the received ones
# copied in. Its sends are waited for here, so that the fill's other
# stages see nothing of this one; every rank posts its receives before it
# waits, so the wait cannot deadlock.
function run_glue!(fs::FieldSet{T,D}, plan::GluePlan{T,D}, backend, comm) where {T,D}
    isempty(plan) && return nothing
    tables = plan.device === nothing ? plan : plan.device
    run_glue!(fs, plan, tables, backend, comm)
    return nothing
end

function run_glue!(fs::FieldSet{T,D}, plan::GluePlan{T,D}, tables, backend,
                   comm) where {T,D}
    nv = fs.nvars
    bufs = glue_buffers!(plan, backend, comm, nv)
    recvs = Any[]
    for (peer, r) in zip(plan.recvpeers, segment_ranges(plan.recvcounts, nv))
        push!(recvs, irecv(comm, view(bufs.wirerecv, r), peer, GLUE_TAG))
    end
    sends = Any[]
    nb = nblocks(fs)
    if !isempty(plan.sendpeers)
        if backend isa CPU
            runs = plan.sendruns
            launch_by_owner!(glue_owner_pack_kernel!, backend, bufs.send, fs.work,
                             runs.order, runs.start, plan.sendsource, plan.sendfirst,
                             plan.sendwidth, plan.sendweights, nv, Val(D), Val(plan.k);
                             ndrange=(runs.longest, nb))
        else
            glue_pack_kernel!(backend)(bufs.send, fs.work, tables.sendsource,
                                       tables.sendfirst, tables.sendwidth,
                                       tables.sendweights, nv, Val(D), Val(plan.k);
                                       ndrange=(length(plan.sendsource), nv))
        end
        synchronize(backend)
        bufs.wiresend === bufs.send || copyto!(bufs.wiresend, bufs.send)
        for (peer, r) in zip(plan.sendpeers, segment_ranges(plan.sendcounts, nv))
            push!(sends, isend(comm, view(bufs.wiresend, r), peer, GLUE_TAG))
        end
    end
    if !isempty(plan.target)
        if backend isa CPU
            runs = plan.localruns
            launch_by_owner!(glue_owner_kernel!, backend, fs.work, runs.order, runs.start,
                             plan.target, plan.index, plan.source, plan.first, plan.width,
                             plan.weights, nv, Val(D), Val(plan.k);
                             ndrange=(runs.longest, nb))
        else
            glue_kernel!(backend)(fs.work, tables.target, tables.index, tables.source,
                                  tables.first, tables.width, tables.weights, Val(D),
                                  Val(plan.k); ndrange=(length(plan.target), nv))
        end
    end
    if !isempty(recvs)
        waitall(comm, recvs)
        bufs.wirerecv === bufs.recv || copyto!(bufs.recv, bufs.wirerecv)
        if backend isa CPU
            runs = plan.recvruns
            launch_by_owner!(glue_owner_unpack_kernel!, backend, fs.work, bufs.recv,
                             runs.order, runs.start, plan.recvtarget, plan.recvindex, nv,
                             Val(D); ndrange=(runs.longest, nb))
        else
            glue_unpack_kernel!(backend)(fs.work, bufs.recv, tables.recvtarget,
                                         tables.recvindex, nv, Val(D);
                                         ndrange=(length(plan.recvtarget), nv))
        end
    end
    synchronize(backend)
    isempty(sends) || waitall(comm, sends)
    return nothing
end

# --- The copies of a point across glues --------------------------------------
#
# What stage A's members and the mask's primary at a kinked glue are found
# from (M22): the copies of a point across the glues of a forest of
# patches.

# Every copy of the point at doubled coordinates `y` of patch `p` (twice
# its position in spacings of level `lvl`, so that a cell center is odd
# and never on a face), following each glued face it lies on, as
# `(patch, y)` pairs, itself first. A glue that turns the axes may carry
# the point onto a dimension of the other patch with another centering,
# where no point of the set lies (`c` the staggers): that image is no copy.
function glue_copies(conn::Patches{D}, p::Int, y::NTuple{D,Int}, lvl::Int, N::Int,
                     c::NTuple{D,Int}) where {D}
    copies = [(p, y)]
    i = 1
    while i <= length(copies)
        q, z = copies[i]
        n = conn.roots[q]
        for e in 1:D, side in 1:2
            z[e] == (side == 1 ? 0 : 2N * (n[e] << lvl)) || continue
            entry = conn.glue[glue_slot(q, e, side, D)]
            entry.patch == 0 && continue
            there = ntuple(D) do a
                x = z[Int(entry.perm[a])]
                o = 2N * (entry.offset[a] << lvl)
                return entry.flip[a] ? o - x : x + o
            end
            all(a -> iseven(there[a]) == (c[a] == 1), 1:D) || continue
            (entry.patch, there) in copies || push!(copies, (entry.patch, there))
        end
        i += 1
    end
    return copies
end

# --- Stage A: the copies of a point on a kinked glue (M22) -------------------
#
# Across a glue a set is interpolated at (`glue = :interpolate`), the
# patches holding a point of a glued face evolve it by different stencils,
# and their copies differ by the scheme's error. Stage A makes them one
# number, their mean (M22 rule 4; the prototype's "averaged right-hand
# sides", M20 in HISTORY.md), before stages W, P and 0, which carry it to
# every other copy inside each patch. The members of a point are the
# within-patch primaries of the patches holding it, one a patch, or more
# where a patch is glued to itself: found by following the glues of every
# glued face the point lies on (`glue_copies`), in each patch's integer
# coordinates, so exactly, and ordered by patch and then by position. Every
# block holding a member writes the same mean, computed by every rank alike
# from the members in that order (`mean_gather_kernel!`), so the copies are
# one number bit for bit.
#
# Over a distributed forest the members another rank holds come in one
# message round under `GLUE_MEAN_TAG`, the sender packing its members'
# values unchanged; the means are computed where they are written, so no
# second round is needed. Both ends derive the layout from the tree: per
# peer, the entries sorted by the receiving member's global leaf and stored
# index, then the sending member's, the variables of a value together.
#
# The buffers — the means, `nvars × n`; the values sent and received —
# are made on the first fill and kept with the plan, remade when the
# variable count changes, and freed with the schedule, as the glue plan's
# are.

# The tag of stage A's round, past the interpolated ghosts'.
const GLUE_MEAN_TAG = 37

mutable struct MeanBuffers
    means::Any
    send::Any
    recv::Any
    wiresend::Any
    wirerecv::Any
    nvars::Int
end
MeanBuffers() = MeanBuffers(nothing, nothing, nothing, nothing, nothing, 0)

struct GlueMeans{MP}
    targets::MP            # the members of this rank's within-patch primaries
    sends::MP              # one member each: a value another rank needs
    sendpeers::Vector{Int}
    sendcounts::Vector{Int}
    recvpeers::Vector{Int}
    recvcounts::Vector{Int}
    buffers::MeanBuffers
end
Base.isempty(g::GlueMeans) = length(g.targets) == 0 && length(g.sends) == 0

# The doubled patch coordinates of stored index `idx` of leaf `k` (twice its
# position in spacings of `k`'s level, a cell centre odd), as `glue_copies`
# takes a point.
function doubled_point(forest::Forest{D}, k::MortonKey{D}, idx::NTuple{D,Int},
                       G::NTuple{D,Int}, c::NTuple{D,Int}) where {D}
    lvl = level(k)
    N = forest.N
    rootpos = root_position(forest, k.root)
    return ntuple(D) do e
        g = (rootpos[e] << lvl) + Int(k.coords[e])
        return 2 * (g * N + idx[e] - G[e] - 1) + 1 - c[e]
    end
end

# The within-patch primary of the point at doubled coordinates `z` of patch
# `q`, at level `lvl`: the global leaf holding it on its high planes where a
# block lies below along a dimension it is shared in (the greatest class
# tuple, as `piece_primary` decides inside a patch), and its stored index.
# Every leaf within reach of a kinked glue is at one level (*No refinement
# boundary at a glue*), so the leaf is at `lvl`.
function patch_primary(forest::Forest{D}, q::Int, z::NTuple{D,Int}, lvl::Int,
                       G::NTuple{D,Int}, c::NTuple{D,Int}) where {D}
    N = forest.N
    conn = forest.conn::Patches{D}
    g = ntuple(e -> (c[e] == 1 && z[e] > 0 && z[e] % (2N) == 0) ? z[e] ÷ (2N) - 1 :
                    z[e] ÷ (2N), D)
    root, coords, _ = glued_node(conn, q, g, lvl, Crossing())
    j = find_leaf(forest, MortonKey{D}(Int32(root), Int8(lvl), map(UInt32, coords)))
    j === nothing && error(
        "a point on a kinked glue at $z of patch $q has no leaf at level $lvl " *
        "holding it; every leaf within reach of the glue is at one level, so this " *
        "is a bug")
    idx = ntuple(e -> (z[e] - (1 - c[e])) ÷ 2 - g[e] * N + G[e] + 1, D)
    return j, idx
end

# The members of the point at stored index `idx` of global leaf `j`: every
# patch's within-patch primary of it, as `(patch, doubled point, leaf,
# stored index)`, by patch and then by position.
function glue_members(forest::Forest{D}, j::Int, idx::NTuple{D,Int}, G::NTuple{D,Int},
                      c::NTuple{D,Int}) where {D}
    k = forest.leaves[j]
    lvl = level(k)
    y = doubled_point(forest, k, idx, G, c)
    copies = sort!(glue_copies(forest.conn::Patches{D}, patch(forest, k), y, lvl,
                               forest.N, c))
    return [(q, z, patch_primary(forest, q, z, lvl, G, c)...) for (q, z) in copies]
end

# Stage A's plan for a set of ghost widths `G` and staggers `c` over a
# forest of patches it is interpolated across, or `nothing` where no own
# point lies on a glued face (a set cell-like along every glued normal).
function glue_means(forest::Forest{D}, G::NTuple{D,Int}, c::NTuple{D,Int},
                    backend) where {D}
    conn = forest.conn::Patches{D}
    N = forest.N
    owned = blockrange(forest)
    offset = first(owned) - 1
    me = commrank(forest.comm)
    # This rank's members: per block on a glued face, the pieces of its
    # closed box on that face whose copy is the within-patch primary.
    targets = Tuple{Int,NTuple{D,Int},Vector{Tuple{Int,NTuple{D,Int},Int,NTuple{D,Int}}}}[]
    for b in eachindex(owned)
        j = offset + b
        k = forest.leaves[j]
        faces = filter(((e, s),) -> c[e] == 1, glued_faces(conn, k))
        isempty(faces) && continue
        bits = primary_bits(forest, j, c; acrossglue=false)
        for cls in closed_pieces(c)
            (bits >> piece_index(cls)) & 0x1 == 0x1 || continue
            any(((e, s),) -> cls[e] == (s == 1 ? 0 : 2), faces) || continue
            rng = ntuple(d -> piece_range(N, G[d], c[d], piece_span(cls[d])), D)
            for I in CartesianIndices(rng)
                idx = Tuple(I)
                members = glue_members(forest, j, idx, G, c)
                length(members) > 1 || continue
                any(m -> m[3] == j && m[4] == idx, members) || error(
                    "the within-patch primary at $idx of leaf $j is not among the " *
                    "members of its point on a kinked glue; this is a bug")
                push!(targets, (j, idx, members))
            end
        end
    end
    isempty(targets) && !isdistributed(forest) && return nothing
    # What is sent and received: a member held elsewhere, per pair of a
    # target and a member, keyed by both, so both ends sort alike.
    Key = Tuple{Int,Int,NTuple{D,Int},Int,NTuple{D,Int}}
    recv = Key[]
    send = Key[]
    for (j, idx, members) in targets, (_, _, mj, midx) in members
        (mj == j && midx == idx) && continue
        o = leafowner(forest, mj)
        o == me && continue
        # Received: member `(mj, midx)` of target `(j, idx)`, from `o`.
        push!(recv, (o, j, idx, mj, midx))
        # Sent: this target's value to `o`, for its target `(mj, midx)`,
        # keyed as `o` keys the receive, so that both sort alike.
        push!(send, (o, mj, midx, j, idx))
    end
    sort!(recv)
    sort!(send)
    M = isempty(targets) ? 1 : maximum(t -> length(t[3]), targets)
    host = host_means(Val(D), length(targets), M, false)
    slot = Dict{Tuple{Int,NTuple{D,Int},Int,NTuple{D,Int}},Int}(
        (r[2], r[3], r[4], r[5]) => i for (i, r) in enumerate(recv))
    for (i, (j, idx, members)) in enumerate(targets)
        host.blocks[i] = Int32(j - offset)
        host.points[i] = map(Int32, idx)
        host.statepoints[i] = ntuple(d -> Int32(idx[d] - G[d]), D)
        host.mcount[i] = Int32(length(members))
        for (m, (_, _, mj, midx)) in enumerate(members)
            if leafowner(forest, mj) == me
                host.mblock[m, i] = Int32(mj - offset)
                host.mpoint[m, i] = map(Int32, midx)
                host.mstatepoint[m, i] = ntuple(d -> Int32(midx[d] - G[d]), D)
            else
                host.mslot[m, i] = Int32(slot[(j, idx, mj, midx)])
            end
        end
    end
    sends = host_means(Val(D), length(send), 1, false)
    for (i, (_, _, _, j, idx)) in enumerate(send)
        sends.mcount[i] = Int32(1)
        sends.mblock[1, i] = Int32(j - offset)
        sends.mpoint[1, i] = map(Int32, idx)
        sends.mstatepoint[1, i] = ntuple(d -> Int32(idx[d] - G[d]), D)
    end
    function peers(list)
        ps = unique(first.(list))
        return ps, [count(x -> x[1] == p, list) for p in ps]
    end
    sendpeers, sendcounts = peers(send)
    recvpeers, recvcounts = peers(recv)
    return GlueMeans(todevice(backend, host), todevice(backend, sends), sendpeers,
                     sendcounts, recvpeers, recvcounts, MeanBuffers())
end

# The buffers of a run of `nv` variables, made or remade when the count
# changes.
function mean_buffers!(g::GlueMeans, ::Type{T}, backend, comm, nv::Int) where {T}
    bufs = g.buffers
    if bufs.means === nothing || bufs.nvars != nv
        ns, nr = sum(g.sendcounts; init=0), sum(g.recvcounts; init=0)
        bufs.means = allocate(backend, T, (nv, length(g.targets)))
        bufs.send = allocate(backend, T, (nv, ns))
        bufs.recv = allocate(backend, T, (nv, nr))
        staged = hoststaging(comm, vec(bufs.send))
        bufs.wiresend = staged ? Vector{T}(undef, nv * ns) : vec(bufs.send)
        bufs.wirerecv = staged ? Vector{T}(undef, nv * nr) : vec(bufs.recv)
        bufs.nvars = nv
    end
    return bufs
end

# Stage A on `A`, `fs`'s working array or, with `state`, a state-shaped
# one: the receives posted, this rank's members packed and sent, the
# means formed once the receives are in and written back. Its sends are
# waited for here, so the stages after it see nothing of this one.
function run_glue_means!(A::AbstractArray{T}, fs::FieldSet, g::GlueMeans, backend, comm;
                         state::Bool=false) where {T}
    nv = fs.nvars
    bufs = mean_buffers!(g, T, backend, comm, nv)
    recvs = Any[]
    for (peer, r) in zip(g.recvpeers, segment_ranges(g.recvcounts, nv))
        push!(recvs, irecv(comm, view(bufs.wirerecv, r), peer, GLUE_MEAN_TAG))
    end
    sends = Any[]
    # Every member is read as stored (orientation 0), so the set's sign
    # and variable tables are not read; a set without a seam has none, and
    # the means buffer stands in for them.
    dummy = bufs.means
    if !isempty(g.sendpeers)
        gather_means!(bufs.send, A, bufs.recv, g.sends, nv, dummy, dummy, backend;
                      state=state)
        synchronize(backend)
        bufs.wiresend === vec(bufs.send) || copyto!(bufs.wiresend, vec(bufs.send))
        for (peer, r) in zip(g.sendpeers, segment_ranges(g.sendcounts, nv))
            push!(sends, isend(comm, view(bufs.wiresend, r), peer, GLUE_MEAN_TAG))
        end
    end
    if !isempty(recvs)
        waitall(comm, recvs)
        bufs.wirerecv === vec(bufs.recv) || copyto!(vec(bufs.recv), bufs.wirerecv)
    end
    gather_means!(bufs.means, A, bufs.recv, g.targets, nv, dummy, dummy, backend;
                  state=state)
    write_means!(A, bufs.means, g.targets, nv, backend; state=state)
    synchronize(backend)
    isempty(sends) || waitall(comm, sends)
    return nothing
end

# Stage A where a schedule has one, behind a barrier, so that a forest
# without a kinked glue compiles none of it.
function run_stage_a!(A, fs::FieldSet, means, backend, comm; state::Bool=false)
    means === nothing && return nothing
    Base.inferencebarrier(run_glue_means!)(A, fs, means, backend, comm; state=state)
    return nothing
end

# The mask's bits of global leaf `j` of a set interpolated across its
# glues: of the within-patch primaries `bits`, those of a piece on a glued
# face only where the block's copy is the member on the lowest patch,
# then at the greatest position within it, which is the primary the
# reductions count once. The patches holding a point are the same along a
# piece, since a glue joins a whole face to a whole face.
function glue_lowest_bits(forest::Forest{D}, j::Int, c::NTuple{D,Int}, G::NTuple{D,Int},
                          bits::UInt32) where {D}
    conn = forest.conn::Patches{D}
    k = forest.leaves[j]
    N = forest.N
    faces = filter(((e, s),) -> c[e] == 1, glued_faces(conn, k))
    isempty(faces) && return bits
    for cls in closed_pieces(c)
        (bits >> piece_index(cls)) & 0x1 == 0x1 || continue
        any(((e, s),) -> cls[e] == (s == 1 ? 0 : 2), faces) || continue
        idx = ntuple(d -> c[d] == 0 || cls[d] == 0 ? G[d] + 1 :
                          cls[d] == 2 ? G[d] + N + 1 : G[d] + 1 + N ÷ 2, D)
        members = glue_members(forest, j, idx, G, c)
        _, _, mj, midx = argmin(m -> (m[1], map(-, m[2])), members)
        mj == j && midx == idx || (bits &= ~(UInt32(1) << piece_index(cls)))
    end
    return bits
end
