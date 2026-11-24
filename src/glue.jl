# Ghosts interpolated across a glue (M20).
#
# A field set built with `glue = :interpolate` over a forest of patches
# is filled across a glued face by interpolation in the patch across, at
# the physical point the target's own map names past its box, located
# through the application's atlas when the schedule is built. Each fill
# then evaluates one tensor-product Lagrange stencil per ghost point and
# variable. See "Finite differences across a glue" in CODE.md, which this
# file implements serially and on the CPU, under half-open ownership (the
# appendix of closed ownership, the sender's evaluation across ranks and
# the device path are later steps of M20).

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
        n = conn.roots[p]
        rootpos = root_position(n, Int(k.root) - conn.firstroot[p])
        g = ntuple(e -> (rootpos[e] << lvl) + Int(k.coords[e]), D)
        touched = [(e, s) for e in 1:D for s in 1:2
                   if (s == 1 ? g[e] == 0 : g[e] == (n[e] << lvl) - 1) && isglued(conn, p, e, s)]
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
            # opposite face, a wrap. A point on a glued high face plane, which
            # the atlas may give either side of, is taken to the side where
            # that plane is owned.
            ξ′, q = loc
            ξ′, q = owned_side(forest, conn, R, ξ′, q, lvl, c)
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
                # only where that is not glued, since under half-open
                # ownership a glued high plane is a ghost filled here too.
                hi = c[e] == 1 && !isglued(conn, q, e, 2) ? n : n - 1
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

# A point `ξ` of patch `q` at the level `lvl`, moved across every glued
# high face of a vertex-like dimension it lies on, by the glue's map in
# point coordinates, to the patch where that face plane is owned (under
# half-open ownership a glued high plane is a ghost, filled by the very
# stage that reads here). A plane glued high to high has no owner, and is
# refused until closed ownership gives it one.
function owned_side(forest::Forest{D,R}, conn::Patches{D}, ::Type{R}, ξ, q::Int, lvl::Int,
                    c::NTuple{D,Int}) where {D,R}
    N = forest.N
    h = spacing(R, forest, lvl)
    scale = (1 << lvl) * N
    t = ntuple(e -> (ξ[e] - patch_extents(forest, q)[e][1]) / h, D)
    for _ in 1:D
        npts = patch_roots(forest, q)
        e = findfirst(e -> c[e] == 1 && isglued(conn, q, e, 2) &&
                           abs(t[e] - npts[e] * scale) <= 1e-6, 1:D)
        e === nothing && break
        # On the face exactly, so that the map lands on the face across
        # exactly, in point coordinates that are integers there.
        t = Base.setindex(t, R(npts[e] * scale), e)
        entry = conn.glue[glue_slot(q, e, 2, D)]
        entry.face[2] == 2 && throw(ArgumentError(
            "a ghost lands on the high face of dimension $e of patch $q, glued to the " *
            "high face of dimension $(entry.face[1]) of patch $(entry.patch): under " *
            "half-open ownership neither patch owns that plane, and interpolating " *
            "from it waits for closed ownership (M20 step 4)"))
        t = ntuple(D) do a
            x = t[Int(entry.perm[a])]
            return entry.flip[a] ? entry.offset[a] * scale - x : x + entry.offset[a] * scale
        end
        q = entry.patch
    end
    lo = patch_extents(forest, q)
    return ntuple(e -> lo[e][1] + t[e] * h, D), q
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
