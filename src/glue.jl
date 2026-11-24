# Ghosts interpolated across a glue (M20).
#
# A field set built with `glue = :interpolate` over a forest of patches
# is filled across a glued face by interpolation in the patch across, at
# the physical point the target's own map names past its box, located
# through the application's atlas when the schedule is built. Each fill
# then evaluates one tensor-product Lagrange stencil per ghost point and
# variable. See "Finite differences across a glue" in CODE.md, which this
# file implements serially and on the CPU, with closed ownership's
# appendix (the sender's evaluation across ranks and the device path are
# later steps of M20).

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

# The interpolated ghosts of one schedule: per ghost point the target
# block and stored index, the source block, the stored index of the
# stencil's first point and its width per dimension (`k`, or 1 along an
# aligned dimension), and its weights, `k` per dimension, column `j`.
# One concrete type whether empty or not, so that a schedule's type does
# not say whether it interpolates.
struct GluePlan{T,D}
    k::Int
    target::Vector{Int32}
    index::Vector{NTuple{D,Int32}}
    source::Vector{Int32}
    first::Vector{NTuple{D,Int32}}
    width::Vector{NTuple{D,Int8}}
    weights::Matrix{T}
end
GluePlan{T,D}() where {T,D} =
    GluePlan{T,D}(0, Int32[], NTuple{D,Int32}[], Int32[], NTuple{D,Int32}[],
                  NTuple{D,Int8}[], zeros(T, 0, 0))
Base.isempty(plan::GluePlan) = isempty(plan.target)
Base.length(plan::GluePlan) = length(plan.target)

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
    commsize(forest.comm) == 1 || throw(ArgumentError(
        "a field set interpolated across a glue over a distributed forest is not " *
        "implemented yet: the sending rank's evaluation is a later part of M20 step 4"))
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

# The glue plan of one schedule: every interpolated region `(b, δ,
# region)` of this rank's blocks — `b` a global leaf, `region` stored
# indices — located through the atlas, with its stencil. Refuses, saying
# why, a ghost the atlas puts in no patch or back in its own, one whose
# source is at another level, an alignment that does not hold, and a
# stencil that would read a point no copy within the source's patch fills.
function glue_plan(::Type{T}, forest::Forest{D,R}, regions, G::NTuple{D,Int},
                   c::NTuple{D,Int}, ops::Operators, atlas, offset::Int) where {T,D,R}
    isempty(regions) && return GluePlan{T,D}()
    conn = forest.conn::Patches{D}
    N = forest.N
    k = ops.glue
    S = promote_type(T, Float64)
    aligned = hasmethod(chart_aligned, Tuple{typeof(atlas),Int,Int})
    target = Int32[]
    index = NTuple{D,Int32}[]
    source = Int32[]
    firsts = NTuple{D,Int32}[]
    widths = NTuple{D,Int8}[]
    weights = T[]
    for (b, δ, region) in regions
        key = forest.leaves[b]
        p = patch(forest, key)
        lvl = level(key)
        h = spacing(R, forest, lvl)
        origin = block_origin(R, forest, key)
        for I in region
            idx = Tuple(I)
            ξ = ntuple(d -> nodeposition(origin[d], h, idx[d] - G[d], c[d], nothing), D)
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
            al = aligned ? chart_aligned(atlas, p, q) : ntuple(_ -> false, D)
            lo = patch_extents(forest, q)
            npts = patch_roots(forest, q)
            spos = root_position(forest, skey.root)
            start = ntuple(e -> ((spos[e] << lvl) + Int(skey.coords[e])) * N, D)
            fs = Vector{Int32}(undef, D)
            ws = Vector{Int8}(undef, D)
            col = zeros(T, k * D)
            overhang = zeros(Int, D)
            for e in 1:D
                t = (S(ξ′[e]) - S(lo[e][1])) / S(h) - (c[e] == 1 ? 0 : S(1) / 2)
                n = (npts[e] << lvl) * N
                # The window: the patch's own points, its high face plane
                # included along a vertex-like dimension, owned where it is
                # glued (the appendix) and the hook's at an outer face.
                hi = c[e] == 1 ? n : n - 1
                if al[e]
                    r = round(Int, t)
                    abs(t - r) <= S(1) / 1000 || throw(ArgumentError(
                        "the atlas $(typeof(atlas)) declares dimension $e of patch $q " *
                        "aligned with patch $p (chart_aligned), but the ghost at $ξ of " *
                        "patch $p lands at $(t) spacings along it, not at a node"))
                    f, w = r, 1
                    col[(e - 1) * k + 1] = one(T)
                else
                    hi - k + 1 >= 0 || throw(ArgumentError(
                        "patch $q has $(hi + 1) points along dimension $e at level " *
                        "$lvl, fewer than the glue order $k"))
                    f = clamp(floor(Int, t) - k ÷ 2 + 1, 0, hi - k + 1)
                    w = k
                    col[(e - 1) * k .+ (1:k)] .= glue_weights(T, S, t, f, k)
                end
                first = f - start[e] + G[e] + 1
                last = first + w - 1
                1 <= first && last <= N + 2G[e] + c[e] || throw(ArgumentError(
                    "the stencil of the ghost at $ξ of patch $p reaches outside the " *
                    "stored array of its source block $skey along dimension $e"))
                # Beyond the owned range: the ghosts below, or above the
                # shared plane of a vertex-like dimension, which a copy
                # within the patch fills, or at an outer high face the hook.
                overhang[e] = first <= G[e] ? -1 : last > G[e] + N + c[e] ? 1 : 0
                fs[e] = first
                ws[e] = w
            end
            check_overhang(forest, skey, overhang, ξ, p)
            push!(target, Int32(b - offset))
            push!(index, ntuple(d -> Int32(idx[d]), D))
            push!(source, Int32(s - offset))
            push!(firsts, ntuple(d -> fs[d], D))
            push!(widths, ntuple(d -> ws[d], D))
            append!(weights, col)
        end
    end
    return GluePlan{T,D}(k, target, index, source, firsts, widths,
                         reshape(weights, k * D, length(target)))
end

# What a stencil reads beyond its source block's owned range is the
# source's ghosts inside its own patch, which phase 1's same-level copies
# fill: refused if a neighbor in one of those directions is missing (the
# stencil would read across a glue) or at another level.
function check_overhang(forest::Forest{D}, skey::MortonKey{D}, overhang, ξ, p) where {D}
    for δ in alldirections(Val(D))
        all(e -> δ[e] == 0 || δ[e] == overhang[e], 1:D) || continue
        nbrs = oriented_neighbors(forest, skey, δ)[2]
        ok = !isempty(nbrs) && level(first(nbrs)) == level(skey) &&
             !iscrossing(oriented_neighbors(forest, skey, δ)[1])
        ok || throw(ArgumentError(
            "the stencil of the ghost at $ξ of patch $p reads the ghosts of its source " *
            "block $skey in direction $δ, where no block of the same level and patch " *
            "fills them: the ghost lies too deep in the patch across, or a refinement " *
            "boundary is within reach of the glue"))
    end
    return nothing
end

# Each ghost point and variable sums the tensor product of its weights over
# the source block's points (M20); along an aligned dimension the width is
# one. One work item per point and variable, each writing its own slot.
@kernel function glue_kernel!(work, @Const(target), @Const(index), @Const(source),
                              @Const(firsts), @Const(widths), @Const(weights),
                              ::Val{D}, ::Val{K}) where {D,K}
    j, v = @index(Global, NTuple)
    @inbounds begin
        b = target[j]
        s = source[j]
        f = firsts[j]
        w = widths[j]
        acc = zero(eltype(work))
        for tap in 0:(K^D - 1)
            digits = ntuple(d -> (tap ÷ K^(d - 1)) % K, Val(D))
            any(d -> digits[d] >= w[d], 1:D) && continue
            c = one(eltype(work))
            for d in 1:D
                c *= weights[(d - 1) * K + digits[d] + 1, j]
            end
            acc += c * work[ntuple(d -> Int(f[d]) + digits[d], Val(D))..., v, s]
        end
        work[ntuple(d -> Int(index[j][d]), Val(D))..., v, b] = acc
    end
end

# The fill of the interpolated ghosts, after phase 1 and the boundary hook.
function run_glue!(fs::FieldSet{T,D}, plan::GluePlan{T,D}, backend) where {T,D}
    isempty(plan) && return nothing
    kernel = glue_kernel!(backend)
    kernel(fs.work, plan.target, plan.index, plan.source, plan.first, plan.width,
           plan.weights, Val(D), Val(plan.k); ndrange=(length(plan), fs.nvars))
    synchronize(backend)
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

function appendix_layout(fs::FieldSet{T,D}) where {T,D}
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
    d = findfirst(d′ -> I[d′] == N + 1, 1:D)::Int
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
struct AppendixArray{T,D,C<:AbstractArray{T},A<:AbstractVector{T}}
    core::C
    appendix::A
    dims::Vector{NTuple{D,Bool}}
    base::Vector{Int}
    count::Vector{Int}
    N::Int
end

@inline function appendix_slot(a::AppendixArray{T,D}, I::NTuple{D,Int}, v::Int,
                               b::Int) where {T,D}
    e = a.dims[b]
    all(d -> I[d] <= a.N + e[d], 1:D) || return 0
    return a.base[b] + (v - 1) * a.count[b] + appendix_rank(I, e, a.N)
end

function Base.getindex(a::AppendixArray{T,D}, I::Vararg{Int}) where {T,D}
    i = ntuple(d -> I[d], Val(D))
    v, b = I[D + 1], I[D + 2]
    all(d -> i[d] <= a.N, 1:D) && return a.core[i..., v, b]
    s = appendix_slot(a, i, v, b)
    s == 0 && throw(BoundsError(a, I))
    return a.appendix[s]
end

function Base.setindex!(a::AppendixArray{T,D}, x, I::Vararg{Int}) where {T,D}
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

function move_appendix!(kernel, fs::FieldSet{T,D}, a, b, L::AppendixLayout{D}) where {T,D}
    isempty(L.points) && return nothing
    backend = get_backend(fs.work)
    kernel(backend)(a, b, L.points, L.block, L.first, L.base, L.count, Val(D), Val(fs.G);
                    ndrange=(length(L.points), fs.nvars))
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
# `(patch, y)` pairs, itself first.
function glue_copies(conn::Patches{D}, p::Int, y::NTuple{D,Int}, lvl::Int,
                     N::Int) where {D}
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
            (entry.patch, there) in copies || push!(copies, (entry.patch, there))
        end
        i += 1
    end
    return copies
end

# Whether the copy of the point `y` of patch `p` is the one counted.
counted_copy(conn::Patches{D}, p::Int, y::NTuple{D,Int}, lvl::Int, N::Int) where {D} =
    minimum(glue_copies(conn, p, y, lvl, N)) == (p, y)

# The masks of this rank's blocks, on `backend`, for a set with an
# appendix; a set without one passes `NoMasks()`.
fold_masks(backend, fs::FieldSet) = todevice(backend, host_fold_masks(fs))

function host_fold_masks(fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    conn = forest.conn::Patches{D}
    N = forest.N
    c = staggers(fs.centering)
    masks = FoldMask{D}[]
    for j in blockrange(forest)
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
            copies = glue_copies(conn, p, position(I0), lvl, N)
            bit = minimum(copies) == (p, position(I0))
            # Two copies on the lowest patch, a patch glued to itself:
            # which counts may depend on the point, so every point is
            # asked. A wrap answers alike for the whole piece.
            if count(x -> x[1] == minimum(copies)[1], copies) > 1
                for I in CartesianIndices(ranges)
                    counted_copy(conn, p, position(Tuple(I)), lvl, N) == bit && continue
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
    return masks
end

KernelAbstractions.get_backend(a::AppendixArray) = get_backend(a.core)
