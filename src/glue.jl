# Ghosts interpolated across a glue (M20).
#
# A field set built with `glue = :interpolate` over a forest of patches
# is filled across a glued face by interpolation in the patch across, at
# the physical point the target's own map names past its box, located
# through the application's atlas when the schedule is built. Each fill
# then evaluates one tensor-product Lagrange stencil per ghost point and
# variable. See "Finite differences across a glue" in CODE.md, which this
# file implements serially and over a distributed forest, on the CPU and
# on a device, with closed ownership's appendix.

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

# Along which dimensions leaf `k` owns its high face plane under closed
# ownership (M20): the vertex-like ones along which its high face lies on
# a glued high face of its patch. That plane is the block's appendix.
function appendix_dims(forest::Forest{D}, k::MortonKey{D}, c::NTuple{D,Int}) where {D}
    conn = forest.conn::Patches{D}
    lvl = level(k)
    p = searchsortedlast(conn.firstroot, Int(k.root))
    n = conn.roots[p]
    rootpos = root_position(n, Int(k.root) - conn.firstroot[p])
    return ntuple(D) do e
        c[e] == 1 || return false
        g = (rootpos[e] << lvl) + Int(k.coords[e])
        return g == (n[e] << lvl) - 1 && isglued(conn, p, e, 2)
    end
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
# the arithmetic type `S`, rounded once into `T`.
function glue_weights(::Type{T}, ::Type{S}, t, first::Int, k::Int) where {T,S}
    ts = S(t)
    return ntuple(k) do i
        w = one(S)
        for j in 1:k
            j == i && continue
            w *= (ts - (first + j - 1)) / S(i - j)
        end
        return T(w)
    end
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
    stencil(g) = glue_stencil(T, forest, atlas, G, c, k, g)
    located = collective_checks(forest, "GhostSchedule") do
        ghosts = locate_glue_ghosts(forest, regions, G, c, atlas)
        return (ghosts, map(stencil, ghosts)), UInt(0)
    end
    ghosts, stencils = located
    # The leaves this rank's ghosts read, which `regrid!` keeps.
    reads = Int[]
    foreach(s -> append!(reads, s[4]), stencils)
    sort!(unique!(reads))
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
    sent = map(stencil, received)
    flat(list) = isempty(list) ? zeros(T, k * D, 0) : reduce(hcat, list)
    nonzero(cs) = [r - 1 for r in eachindex(cs) if cs[r] > 0]
    plan = GluePlan{T,D}(k,
                         Int32[ghosts[i].target - offset for i in mine],
                         NTuple{D,Int32}[ghosts[i].index for i in mine],
                         Int32[ghosts[i].source - offset for i in mine],
                         NTuple{D,Int32}[stencils[i][1] for i in mine],
                         NTuple{D,Int8}[stencils[i][2] for i in mine],
                         flat([stencils[i][3] for i in mine]),
                         Int32[g.source - offset for g in received],
                         NTuple{D,Int32}[s[1] for s in sent],
                         NTuple{D,Int8}[s[2] for s in sent],
                         flat([s[3] for s in sent]),
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
    origin = block_origin(R, forest, key)
    return ntuple(d -> nodeposition(origin[d], h, Int(idx[d]) - G[d], c[d], nothing), D)
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

# The stencil of one located ghost: the stored index of its first point
# in the source block, its width per dimension, and its weights, `k` per
# dimension. A function of the ghost and the replicated tree alone, so
# that the source's owner builds what the target's rank checked, bit for
# bit.
function glue_stencil(::Type{T}, forest::Forest{D,R}, atlas, G::NTuple{D,Int},
                      c::NTuple{D,Int}, k::Int, g::GlueGhost{D}) where {T,D,R}
    N = forest.N
    S = promote_type(T, Float64)
    p = Int(g.from)
    skey = forest.leaves[g.source]
    q = patch(forest, skey)
    lvl = level(skey)
    h = spacing(R, forest, lvl)
    ξ′ = g.ξ
    al = hasmethod(chart_aligned, Tuple{typeof(atlas),Int,Int}) ?
         chart_aligned(atlas, p, q) : ntuple(_ -> false, D)
    lo = patch_extents(forest, q)
    npts = patch_roots(forest, q)
    spos = root_position(forest, skey.root)
    start = ntuple(e -> ((spos[e] << lvl) + Int(skey.coords[e])) * N, D)
    where_() = (ghost_position(R, forest, forest.leaves[g.target], g.index, G, c), p)
    fs = Vector{Int32}(undef, D)
    ws = Vector{Int8}(undef, D)
    col = zeros(T, k * D)
    overhang = zeros(Int, D)
    for e in 1:D
        t = (S(ξ′[e]) - S(lo[e][1])) / S(h) - (c[e] == 1 ? 0 : S(1) / 2)
        n = (npts[e] << lvl) * N
        # The window: the patch's own points, its high face plane included
        # along a vertex-like dimension, owned where it is glued (the
        # appendix) and the hook's at an outer face.
        hi = c[e] == 1 ? n : n - 1
        if al[e]
            r = round(Int, t)
            abs(t - r) <= S(1) / 1000 || throw(ArgumentError(
                "the atlas $(typeof(atlas)) declares dimension $e of patch $q aligned " *
                "with patch $p (chart_aligned), but the ghost at $(where_()[1]) of patch " *
                "$p lands at $(t) spacings along it, not at a node"))
            f, w = r, 1
            col[(e - 1) * k + 1] = one(T)
        else
            hi - k + 1 >= 0 || throw(ArgumentError(
                "patch $q has $(hi + 1) points along dimension $e at level $lvl, fewer " *
                "than the glue order $k"))
            f = clamp(floor(Int, t) - k ÷ 2 + 1, 0, hi - k + 1)
            w = k
            col[(e - 1) * k .+ (1:k)] .= glue_weights(T, S, t, f, k)
        end
        first = f - start[e] + G[e] + 1
        last = first + w - 1
        1 <= first && last <= N + 2G[e] + c[e] || throw(ArgumentError(
            "the stencil of the ghost at $(where_()[1]) of patch $p reaches outside the " *
            "stored array of its source block $skey along dimension $e"))
        # Beyond the owned range: the ghosts below, or from the high plane
        # of a vertex-like dimension up, which the neighbor within the
        # patch fills — except the plane at the patch's high face, which
        # the block owns at a glue (the appendix) and the hook fills at an
        # outer face, both before this stage.
        top = c[e] == 1 && start[e] + N == n
        overhang[e] = first <= G[e] ? -1 : last > G[e] + N + top ? 1 : 0
        fs[e] = first
        ws[e] = w
    end
    reads = check_overhang(forest, skey, overhang, where_()...)
    pushfirst!(reads, g.source)
    return ntuple(d -> fs[d], D), ntuple(d -> ws[d], D), col, reads
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

# --- The appendix (M20) ---------------------------------------------------------
#
# Under closed ownership a block whose high face lies on a glued high face
# of its patch owns that face plane too, along every such vertex-like
# dimension. Those planes follow the cores in the state vector: block by
# block, variable by variable, the block's points in a fixed order — the
# plane of each dimension in turn, every point once (one on two planes,
# at an edge or a corner, in the plane of the lower dimension), each
# plane lexicographic in its other dimensions, the first fastest. See
# "Closed ownership and the appendix" in CODE.md, "Finite differences
# across a glue".

# Whether a field set has an appendix: interpolated across the glue of a
# forest of patches, with a vertex-like dimension.
hasappendix(fs::FieldSet) =
    fs.glue === :interpolate && ispatched(fs.forest) && any(==(:vertex), fs.centering)

# The appendix of one field set over its forest's current leaves: per
# local block its dimensions, its offset in the appendix and its number
# of points per variable; and every point, owned index `1 … N+1` per
# dimension, block after block in the order above, with its block.
struct AppendixLayout{D}
    dims::Vector{NTuple{D,Bool}}
    base::Vector{Int}
    count::Vector{Int}
    points::Vector{NTuple{D,Int32}}
    block::Vector{Int32}
    first::Vector{Int}
    total::Int
end

# The layout, built once per change of the forest and kept in the field
# set with its tables on the set's backend: `scatter!`, `gather!` and
# `statearray` run on every evaluation, and a device kernel reads the
# tables where it runs.
struct AppendixCache{D,VD,VI,VP,VB}
    generation::Int
    host::AppendixLayout{D}
    dims::VD
    base::VI
    count::VI
    points::VP
    block::VB
    first::VI
end

function appendix_cache(fs::FieldSet{T,D}) where {T,D}
    cached = fs.appendix
    g = generation(fs.forest)
    cached !== nothing && cached.generation == g && return cached
    L = build_appendix_layout(fs)
    backend = get_backend(fs.work)
    dev(a) = todevice(backend, a)
    fresh = AppendixCache(g, L, dev(L.dims), dev(L.base), dev(L.count), dev(L.points),
                          dev(L.block), dev(L.first))
    fs.appendix = fresh
    return fresh
end

appendix_layout(fs::FieldSet{T,D}) where {T,D} = appendix_cache(fs).host::AppendixLayout{D}

function build_appendix_layout(fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    N = forest.N
    c = staggers(fs.centering)
    range = blockrange(forest)
    dims = NTuple{D,Bool}[]
    base = Int[]
    count = Int[]
    points = NTuple{D,Int32}[]
    block = Int32[]
    first = Int[]
    total = 0
    for (b, j) in enumerate(range)
        e = hasappendix(fs) ? appendix_dims(forest, forest.leaves[j], c) :
            ntuple(_ -> false, D)
        push!(dims, e)
        push!(base, total)
        push!(first, length(points) + 1)
        n0 = length(points)
        for d in 1:D
            e[d] || continue
            ranges = ntuple(D) do d′
                d′ == d && return (N + 1):(N + 1)
                return d′ < d ? (1:N) : (1:(N + e[d′]))
            end
            for I in CartesianIndices(ranges)
                push!(points, ntuple(d′ -> Int32(I[d′]), D))
                push!(block, Int32(b))
            end
        end
        push!(count, length(points) - n0)
        total += fs.nvars * (length(points) - n0)
    end
    return AppendixLayout{D}(dims, base, count, points, block, first, total)
end

# The position, from 1, of the point `I` (owned index `1 … N+1`) among its
# block's appendix points, `e` the block's appendix dimensions: the planes
# of the lower dimensions first, then lexicographic within its own.
@inline function appendix_rank(I::NTuple{D,Int}, e::NTuple{D,Bool}, N::Int) where {D}
    # The first dimension at the high plane, by a loop: a device kernel
    # reads through this, and `findfirst` returns a union.
    d = D
    for d′ in D:-1:1
        I[d′] == N + 1 && (d = d′)
    end
    off = 0
    for d2 in 1:(d - 1)
        e[d2] || continue
        size = 1
        for d′ in 1:D
            d′ == d2 && continue
            size *= d′ < d2 ? N : N + e[d′]
        end
        off += size
    end
    r = 0
    stride = 1
    for d′ in 1:D
        d′ == d && continue
        r += (I[d′] - 1) * stride
        stride *= d′ < d ? N : N + e[d′]
    end
    return off + r + 1
end

"""
    AppendixArray

The whole state of a set with an appendix, as `statearray(u, fs;
appendix = true)` returns it: an `(i1, …, iD, var, b)` array whose `i_d`
runs to `N + 1` along a dimension where block `b` owns its high face
plane, reading and writing the core or the appendix as the index says. A
write to a point the block does not own is discarded, so that a kernel
launched over the closed range may write every point it visits; reading
one is an error. See "Finite differences across a glue" in `CODE.md`.
"""
struct AppendixArray{T,D,C<:AbstractArray{T},A<:AbstractVector{T},
                     VD<:AbstractVector{NTuple{D,Bool}},VI<:AbstractVector{Int}}
    core::C
    appendix::A
    dims::VD
    base::VI
    count::VI
    N::Int
end
AppendixArray(core::AbstractArray{T}, appendix::AbstractVector{T},
              dims::AbstractVector{NTuple{D,Bool}}, base::AbstractVector{Int},
              count::AbstractVector{Int}, N::Int) where {T,D} =
    AppendixArray{T,D,typeof(core),typeof(appendix),typeof(dims),typeof(base)}(
        core, appendix, dims, base, count, N)

# Into a device kernel, as a right-hand side launched with `map_blocks!(…;
# appendix = true)` takes it: every array converted for the device.
Adapt.adapt_structure(to, a::AppendixArray) =
    AppendixArray(Adapt.adapt(to, a.core), Adapt.adapt(to, a.appendix),
                  Adapt.adapt(to, a.dims), Adapt.adapt(to, a.base),
                  Adapt.adapt(to, a.count), a.N)

@inline function appendix_slot(a::AppendixArray{T,D}, I::NTuple{D,Int}, v::Int,
                               b::Int) where {T,D}
    e = a.dims[b]
    all(d -> I[d] <= a.N + e[d], 1:D) || return 0
    return a.base[b] + (v - 1) * a.count[b] + appendix_rank(I, e, a.N)
end

# Inlined, so that a kernel's index arithmetic folds into them: on the
# CPU a call per point allocated its tuples (90 ns and 160 B a read).
@inline function Base.getindex(a::AppendixArray{T,D}, I::Vararg{Int}) where {T,D}
    i = ntuple(d -> I[d], Val(D))
    v, b = I[D + 1], I[D + 2]
    all(d -> i[d] <= a.N, 1:D) && return a.core[i..., v, b]
    s = appendix_slot(a, i, v, b)
    @boundscheck s == 0 && throw(BoundsError(a, I))
    return @inbounds a.appendix[s]
end

@inline function Base.setindex!(a::AppendixArray{T,D}, x, I::Vararg{Int}) where {T,D}
    i = ntuple(d -> I[d], Val(D))
    v, b = I[D + 1], I[D + 2]
    if all(d -> i[d] <= a.N, 1:D)
        a.core[i..., v, b] = x
        return a
    end
    s = appendix_slot(a, i, v, b)
    s == 0 || (a.appendix[s] = x)
    return a
end

Base.eltype(::Type{<:AppendixArray{T}}) where {T} = T
Base.size(a::AppendixArray{T,D}) where {T,D} =
    (ntuple(_ -> a.N + 1, D)..., size(a.core, D + 1), size(a.core, D + 2))

# The appendix's points between the working array and the state vector,
# one work item per point and variable.
@kernel function appendix_scatter_kernel!(work, @Const(app), @Const(points),
                                          @Const(block), @Const(first), @Const(base),
                                          @Const(count), ::Val{D}, ::Val{G}) where {D,G}
    j, v = @index(Global, NTuple)
    @inbounds begin
        b = block[j]
        I = points[j]
        s = base[b] + (v - 1) * count[b] + (j - first[b]) + 1
        work[ntuple(d -> Int(I[d]) + G[d], Val(D))..., v, b] = app[s]
    end
end
@kernel function appendix_gather_kernel!(app, @Const(work), @Const(points),
                                         @Const(block), @Const(first), @Const(base),
                                         @Const(count), ::Val{D}, ::Val{G}) where {D,G}
    j, v = @index(Global, NTuple)
    @inbounds begin
        b = block[j]
        I = points[j]
        s = base[b] + (v - 1) * count[b] + (j - first[b]) + 1
        app[s] = work[ntuple(d -> Int(I[d]) + G[d], Val(D))..., v, b]
    end
end

# The appendix's points of `fill_by_coordinates!` (M20): every form of its
# callback as one that returns all the variables from the position and
# the patch. `all` and `patch` say which form `f` is: `f(x, v)`,
# `f(x) -> vals`, `f(x, p, v)` or `f(x, p) -> vals`.
struct AppendixCallback{F,NV,A,P}
    f::F
end
AppendixCallback(f, ::Val{NV}, all::Bool, patch::Bool) where {NV} =
    AppendixCallback{typeof(f),NV,all,patch}(f)
@inline (g::AppendixCallback{F,NV,false,false})(x, p) where {F,NV} =
    ntuple(v -> g.f(x, v), Val(NV))
@inline (g::AppendixCallback{F,NV,true,false})(x, p) where {F,NV} = g.f(x)
@inline (g::AppendixCallback{F,NV,false,true})(x, p) where {F,NV} =
    ntuple(v -> g.f(x, p, v), Val(NV))
@inline (g::AppendixCallback{F,NV,true,true})(x, p) where {F,NV} = g.f(x, p)

# One work item per appendix point: its position formed by the expression
# `coordinates_kernel!` uses, from the owned index `N + 1` on its plane, so
# that a point the patches share is filled from the same arithmetic.
@kernel function appendix_coordinates_kernel!(work, g, @Const(points), @Const(block),
                                              @Const(origins), @Const(spacings),
                                              @Const(patches), nodes, ::Val{D}, ::Val{G},
                                              ::Val{C}, ::Val{NV}) where {D,G,C,NV}
    j = @index(Global, Linear)
    @inbounds begin
        b = Int(block[j])
        I = points[j]
        origin, h, p = origins[b], spacings[b], Int(patches[b])
        x = ntuple(d -> nodeposition(origin[d], h, Int(I[d]), C[d], nodes), Val(D))
        vals = g(x, p)
        idx = ntuple(d -> Int(I[d]) + G[d], Val(D))
        ntuple(Val(NV)) do v
            work[idx..., v, b] = vals[v]
            nothing
        end
    end
end

function fill_appendix_coordinates!(g::AppendixCallback, fs::FieldSet{T,D}) where {T,D}
    L = appendix_cache(fs)
    isempty(L.host.points) && return nothing
    forest = fs.forest
    backend = get_backend(fs.work)
    origins = todevice(backend, block_origins(forest, T))
    spacings = todevice(backend, block_spacings(forest, T))
    patches = todevice(backend, block_patches(forest))
    with_nodes(backend, T, fs.P) do nodes
        appendix_coordinates_kernel!(backend)(fs.work, g, L.points, L.block, origins,
                                              spacings, patches, nodes, Val(D), Val(fs.G),
                                              Val(staggers(fs)), Val(fs.nvars);
                                              ndrange=length(L.host.points))
    end
    synchronize(backend)
    return nothing
end

function move_appendix!(kernel, fs::FieldSet{T,D}, a, b, L::AppendixCache) where {T,D}
    isempty(L.host.points) && return nothing
    backend = get_backend(fs.work)
    kernel(backend)(a, b, L.points, L.block, L.first, L.base, L.count, Val(D), Val(fs.G);
                    ndrange=(length(L.host.points), fs.nvars))
    synchronize(backend)
    return nothing
end

# --- Counting a glued face point once (M20) ----------------------------------------
#
# Under closed ownership a point on a glued face is owned by every patch
# holding it, so a fold or a norm that read every owned point would count
# it twice, or three or four times at an edge or a corner. It is counted
# at one copy: the one with the lowest `(patch, position)`, the position
# in the patch's own integer coordinates, so that a patch glued to itself
# (a wrap) counts its low plane and not its high one. See "Closed
# ownership and the appendix" in CODE.md, "Finite differences across a
# glue".
#
# A block's owned points fall into at most `3^D` pieces: along each
# dimension, the low face plane where it lies on a glued face of the
# patch (class 0), the high appendix plane (class 2), or neither (class
# 1). Every point of a piece lies on the same glued faces, and a glue
# joins a whole face to a whole face, so its copies lie on the same
# patches, and the decision is one bit per piece. The one exception, two
# copies on the lowest patch, is checked point by point.

# Which of a block's owned points a fold visits: the dimensions with a
# glued low face plane and with an appendix plane, and one bit per piece,
# piece `Σ class_d 3^(d-1)`.
struct FoldMask{D}
    low::NTuple{D,Bool}
    high::NTuple{D,Bool}
    pieces::UInt32
end

# The masks of a set without an appendix: every block visits its `N^D`
# owned points, and the kernel compiles none of this.
struct NoMasks end
@inline Base.getindex(::NoMasks, ::Integer) = nothing
fold_extra(::NoMasks) = 0
fold_extra(::AbstractVector) = 1

@inline counted(::Nothing, I, N) = true
@inline function counted(m::FoldMask{D}, I::NTuple{D,Int}, N::Int) where {D}
    piece, stride = 0, 1
    for d in 1:D
        i = I[d]
        cls = i > N ? 2 : (i == 1 && m.low[d]) ? 0 : 1
        cls == 2 && !m.high[d] && return false
        piece += cls * stride
        stride *= 3
    end
    return (m.pieces >> piece) & 0x1 == 0x1
end

# The number of points a mask counts, of `N^D` per block without one.
function counted_points(m::FoldMask{D}, N::Int) where {D}
    total = 0
    for cls in CartesianIndices(ntuple(_ -> 0:2, D))
        piece = sum(ntuple(d -> cls[d] * 3^(d - 1), D))
        (m.pieces >> piece) & 0x1 == 0x1 || continue
        total += prod(ntuple(D) do d
            cls[d] == 0 ? Int(m.low[d]) : cls[d] == 2 ? Int(m.high[d]) : N - m.low[d]
        end)
    end
    return total
end

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

# Whether the copy of the point `y` of patch `p` is the one counted.
counted_copy(conn::Patches{D}, p::Int, y::NTuple{D,Int}, lvl::Int, N::Int,
             c::NTuple{D,Int}) where {D} =
    minimum(glue_copies(conn, p, y, lvl, N, c)) == (p, y)

# The masks of this rank's blocks, on `backend`, for a set with an
# appendix; a set without one passes `NoMasks()`.
fold_masks(backend, fs::FieldSet) = todevice(backend, host_fold_masks(fs))

# The points a fold counts in each block, which a norm's volume weights.
counted_per_block(fs::FieldSet) = [counted_points(m, fs.forest.N) for m in host_fold_masks(fs)]

function host_fold_masks(fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    conn = forest.conn::Patches{D}
    N = forest.N
    c = staggers(fs.centering)
    masks = FoldMask{D}[]
    # The refusal below depends on the block, so over a distributed forest
    # with a patch glued to itself every rank asks it of every leaf, and
    # all refuse alike (every forest's other folds pay nothing for it).
    range = blockrange(forest)
    selfglued = any(p -> any(e -> e.patch == p, conn.glue[glue_slot(p, d, s, D)]
                                                for d in 1:D, s in 1:2),
                    eachindex(conn.roots))
    blocks = isdistributed(forest) && selfglued ? (1:nleaves(forest)) : range
    for j in blocks
        k = forest.leaves[j]
        lvl = level(k)
        p = patch(forest, k)
        n = conn.roots[p]
        rootpos = root_position(n, Int(k.root) - conn.firstroot[p])
        o = ntuple(e -> N * ((rootpos[e] << lvl) + Int(k.coords[e])), D)
        low = ntuple(e -> c[e] == 1 && o[e] == 0 && isglued(conn, p, e, 1), D)
        high = appendix_dims(forest, k, c)
        position(I) = ntuple(e -> 2 * (o[e] + I[e] - 1) + 1 - c[e], D)
        pieces = UInt32(0)
        for cls in CartesianIndices(ntuple(_ -> 0:2, D))
            ranges = ntuple(D) do d
                cls[d] == 0 ? (low[d] ? (1:1) : (1:0)) :
                cls[d] == 2 ? (high[d] ? ((N + 1):(N + 1)) : (1:0)) : ((1 + low[d]):N)
            end
            any(isempty, ranges) && continue
            piece = sum(ntuple(d -> cls[d] * 3^(d - 1), D))
            I0 = map(first, ranges)
            copies = glue_copies(conn, p, position(I0), lvl, N, c)
            bit = minimum(copies) == (p, position(I0))
            # Two copies on the lowest patch, a patch glued to itself:
            # which counts may depend on the point, so every point is
            # asked. A wrap answers alike for the whole piece.
            if count(x -> x[1] == minimum(copies)[1], copies) > 1
                for I in CartesianIndices(ranges)
                    counted_copy(conn, p, position(Tuple(I)), lvl, N, c) == bit && continue
                    throw(ArgumentError(
                        "patch $p is glued to itself so that the points of one face of " *
                        "leaf $k, $(Tuple(I)) and $I0, are counted at different copies: " *
                        "a fold over a set with closed ownership counts each glued face " *
                        "point once, at the copy with the lowest patch and position, " *
                        "and decides it per face of a block, which this glue does not " *
                        "allow (CODE.md, \"Finite differences across a glue\")"))
                end
            end
            bit && (pieces |= UInt32(1) << piece)
        end
        push!(masks, FoldMask{D}(low, high, pieces))
    end
    return blocks == range ? masks : masks[range]
end

KernelAbstractions.get_backend(a::AppendixArray) = get_backend(a.core)
