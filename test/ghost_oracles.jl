# Helpers for the ghost-exchange tests. As in oracles.jl, these check
# the implementation against something independent of it — analytic
# polynomial values, an explicit tiling, and a direct count of writes.

using TreeAMR: TransferGroup, boxsize, ntransfers, target_range

"""
A polynomial of degree `deg` in each coordinate, varying per variable.
An operator of order `p` reproduces it exactly iff `deg < p`.
"""
makepoly(D, deg) = (x, v) -> sum(0.3v + 0.7d + 0.11 * (d + v) * x[d]^e
                                 for d in 1:D for e in 0:deg)

# 5-point Gauss-Legendre on [-1, 1]: exact through degree 9, so the cell
# averages below are exact for every polynomial the conservative family
# claims to reproduce.
const GAUSS_X = [-0.906179845938664, -0.5384693101056831, 0.0,
                 0.5384693101056831, 0.906179845938664]
const GAUSS_W = [0.23692688505618908, 0.47862867049936647, 0.5688888888888889,
                 0.47862867049936647, 0.23692688505618908]

"""
The exact average of `f` over the cell of width `h` centered at `x` —
what the conservative family treats a stored number as meaning.
"""
function cell_average(f, x::NTuple{D,<:Real}, h::Real) where {D}
    total = zero(promote_type(eltype(x), typeof(h)))
    for idx in CartesianIndices(ntuple(_ -> 1:length(GAUSS_X), D))
        i = Tuple(idx)
        weight = prod(GAUSS_W[i[d]] for d in 1:D) / 2^D
        point = ntuple(d -> x[d] + h / 2 * GAUSS_X[i[d]], D)
        total += weight * f(point)
    end
    return total
end

"""Fill every interior cell with the exact cell average of `f`."""
function fill_cell_averages!(fs::FieldSet{T,D}, f) where {T,D}
    N = fs.forest.N
    for b in 1:nblocks(fs)
        h = spacing(fs.forest, blockkey(fs, b))
        block = blockview(fs, b, 1)
        for idx in CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + N), D))
            block[idx] = cell_average(f, coordinates(fs, b, Tuple(idx)), h)
        end
    end
    return fs
end

"""A boundary hook imposing exact cell averages of `f`."""
boundary_cell_averages(f) =
    function (fs, b, key, δ, region)
        h = spacing(fs.forest, key)
        block = blockview(fs, b, 1)
        for idx in region
            block[idx] = cell_average(f, coordinates(fs, b, Tuple(idx)), h)
        end
        return nothing
    end

"""Worst deviation of any stored cell from the exact cell average of `f`."""
function max_average_deviation(fs::FieldSet{T,D}, f) where {T,D}
    worst = 0.0
    for b in 1:nblocks(fs)
        h = spacing(fs.forest, blockkey(fs, b))
        block = blockview(fs, b, 1)
        for idx in CartesianIndices(block)
            exact = cell_average(f, coordinates(fs, b, Tuple(idx)), h)
            worst = max(worst, abs(block[idx] - exact))
        end
    end
    return worst
end

"""
Fill a hierarchy with exact cell averages of `f`, exchange ghosts with
the conservative family, and report the worst error anywhere.
"""
function conservative_exchange_error(forest, ops, f; G)
    fs = FieldSet(forest, 1; G=G)
    fill_cell_averages!(fs, f)
    fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=boundary_cell_averages(f))
    return max_average_deviation(fs, f)
end

"""A polynomial of total degree `deg`, as a plain function of position."""
scalarpoly(D, deg) = x -> sum(0.7d + 0.31 * (d + 1) * x[d]^e for d in 1:D for e in 0:deg)

"""Largest deviation of any stored cell (interior *and* ghost) from `f`."""
function max_deviation(fs, f)
    worst = 0.0
    for b in 1:nblocks(fs), v in 1:fs.nvars
        blk = blockview(fs, b, v)
        for idx in CartesianIndices(blk)
            x = coordinates(fs, b, Tuple(idx))
            worst = max(worst, abs(blk[idx] - f(x, v)))
        end
    end
    return worst
end

"""
Fill with `f`, exchange ghosts, and report the worst error anywhere.

`max_deviation` evaluates `f` at [`coordinates`](@ref), which knows the
field set's centering, so this is the same claim for every centering: a
face-centered set is filled and compared at face centers, a vertex set at
the vertices, including the shared boundary plane and the domain's upper
boundary plane that the hook fills.
"""
function exchange_error(forest::Forest{D}, ops, f; nvars=2, G=1,
                        centering=cellcentered(D)) where {D}
    fs = FieldSet(forest, nvars; G=G, centering=centering)
    fill_ghosts!(fill_by_coordinates!(f, fs), GhostSchedule(fs, ops);
                 boundary=boundary_by_coordinates(f))
    return max_deviation(fs, f)
end

"""Every one of the `2^D` centerings, in a fixed order."""
allcenterings(::Val{D}) where {D} =
    [NTuple{D,Symbol}(c) for c in Iterators.product(ntuple(_ -> (:cell, :vertex), D)...)]

"""
The position of stored index `idx` of leaf `k`, in exact rational units
of root cells, from the oracle's own box geometry (`leafbox`) rather than
from the package's floating-point `coordinates`. Wrapped into the domain
where a dimension is periodic, so that two blocks abutting across a seam
name a shared point identically.
"""
function exact_point(forest::Forest{D}, k::MortonKey{D}, G::NTuple{D,Int},
                     c::NTuple{D,Int}, idx::NTuple{D,Int}) where {D}
    lo, _ = leafbox(forest, k)
    n = forest.N * (1 << level(k))
    return ntuple(D) do d
        off = c[d] == 1 ? 1//1 : 1//2
        p = lo[d] + (idx[d] - G[d] - off) // n
        Rational{Int}(forest.periodic[d] ? mod(p, forest.roots[d]) : p)
    end
end

exact_point(fs::FieldSet{T,D}, b::Integer, idx::NTuple{D,Int}) where {T,D} =
    exact_point(fs.forest, blockkey(fs, b), fs.G, staggers(fs), idx)

"""
A value determined by a point's exact position and nothing else — data
with no polynomial structure at all, so that no interpolation reproduces
it, but the *same* number wherever two blocks name the same point, which
is what makes a bit-for-bit comparison across blocks meaningful.
"""
arbitrary_value(::Type{T}, p, v::Integer) where {T} =
    T(2 * (hash((p, v)) / typemax(UInt64)) - 1)

"""
Fill every owned point of every block from [`arbitrary_value`](@ref), or
every *closed*-range point with `closed = true` — which is what a block
computes for itself when it holds a flux or an EMF, the shared plane
included, and so what the interface restriction finds in place.
"""
function fill_arbitrary!(fs::FieldSet{T,D}; closed::Bool=false) where {T,D}
    c = closed ? staggers(fs) : ntuple(_ -> 0, D)
    owned = CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + fs.forest.N + c[d]), D))
    for b in 1:nblocks(fs), v in 1:fs.nvars, idx in owned
        fs.work[Tuple(idx)..., v, b] =
            arbitrary_value(T, exact_point(fs, b, Tuple(idx)), v)
    end
    return fs
end

"""
Every owned point of every block, keyed by its exact position: the
level of the block that owns it and that block's stored values.

Ownership is half-open in every dimension and the owned boxes tile the
domain, so each position appears exactly once.
"""
function owned_points(fs::FieldSet{T,D}) where {T,D}
    out = Dict{NTuple{D,Rational{Int}},Tuple{Int,Vector{T}}}()
    owned = CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + fs.forest.N), D))
    for b in 1:nblocks(fs), idx in owned
        p = exact_point(fs, b, Tuple(idx))
        out[p] = (level(blockkey(fs, b)),
                  T[fs.work[Tuple(idx)..., v, b] for v in 1:fs.nvars])
    end
    return out
end

"""
Compare every exchange-filled point of `fs` against the block that
*owns* that position, and report `(nfiner, nsame, mismatches)`: how many
exchange points coincide with an owned point of a finer block, how many
with one at the same level, and how many of those did not come back bit
for bit.

Injection and a same-level copy both reproduce the owner's stored value
exactly, whatever the data; an averaging or interpolating restriction
does not. Points whose owner is *coarser* were prolongated and are
skipped — prolongation is genuine interpolation.
"""
function injection_report(fs::FieldSet{T,D}) where {T,D}
    owners = owned_points(fs)
    G, N = fs.G, fs.forest.N
    nfiner = nsame = bad = 0
    stored = CartesianIndices(ntuple(d -> axes(fs.work, d), D))
    for b in 1:nblocks(fs)
        l = level(blockkey(fs, b))
        for idx in stored
            i = Tuple(idx)
            all(d -> G[d] < i[d] <= G[d] + N, 1:D) && continue   # owned, not exchanged
            entry = get(owners, exact_point(fs, b, i), nothing)
            entry === nothing && continue
            ownerlevel, values = entry
            ownerlevel < l && continue                           # prolongated
            ownerlevel > l ? (nfiner += 1) : (nsame += 1)
            for v in 1:fs.nvars
                fs.work[i..., v, b] == values[v] || (bad += 1)
            end
        end
    end
    return (nfiner, nsame, bad)
end

"""
How many times the schedule writes each stored cell. Every ghost cell
must be written exactly once and no interior cell may be touched, which
together prove the ghost regions are partitioned without gaps or
double-writes.
"""
function write_counts(schedule::GhostSchedule{T,D}) where {T,D}
    forest = schedule.forest
    c = staggers(schedule.centering)
    stored = ntuple(d -> forest.N + 2 * schedule.G[d] + c[d], D)
    counts = zeros(Int, stored..., nleaves(forest))

    function tally!(group::TransferGroup)
        blen = boxsize(group)
        first = ntuple(d -> group.stencils[d].targetfirst, D)
        for t in 1:ntransfers(group)
            b = group.targetblocks[t]
            for off in CartesianIndices(ntuple(d -> 0:(blen[d] - 1), D))
                counts[ntuple(d -> first[d] + off[d], D)..., b] += 1
            end
        end
    end

    foreach(tally!, schedule.phase1)
    for groups in schedule.phase2
        foreach(tally!, groups)
    end
    for r in schedule.boundaries
        for idx in r.region
            counts[Tuple(idx)..., r.block] += 1
        end
    end
    return counts
end

"""How many transfers of each kind the schedule holds."""
function transfer_counts(s::GhostSchedule)
    counts = Dict(:copy => 0, :restrict => 0, :prolong => 0)
    for g in s.phase1
        counts[g.kind] += ntransfers(g)
    end
    for groups in s.phase2, g in groups
        counts[g.kind] += ntransfers(g)
    end
    return counts
end

"""
A hierarchy with three levels meeting, built by nesting two refinement
regions so that distant blocks stay at level 0. A single broad region
would not do: balancing would lift everything off the coarsest level and
leave only two levels in play.
"""
function nested_forest(::Val{D}; T=Float64, N=4, roots=4,
                      periodic=ntuple(_ -> false, D)) where {D}
    forest = Forest{T}(ntuple(_ -> roots, D); N=N, periodic=periodic,
                       extents=ntuple(_ -> (0, roots), D))
    center = ntuple(_ -> 1.5, D)
    near(c, r) = all(d -> abs(c[d] - center[d]) <= r, 1:D)
    refine_where!(forest, (c, lvl) -> (lvl == 0 && near(c, 1.0)) ||
                                      (lvl == 1 && near(c, 0.3)), 2)
    return forest
end

"""Refine wherever `pred(block center, level)` holds, then rebalance."""
function refine_where!(forest, pred, passes)
    for _ in 1:passes
        targets = filter(forest.leaves) do k
            ext = block_extent(forest, k)
            pred(ntuple(d -> (ext[d][1] + ext[d][2]) / 2, length(ext)), level(k))
        end
        isempty(targets) && break
        refine!(forest, targets)
        balance!(forest)
    end
    return forest
end

"""
Compare an `M`-root periodic domain against the middle tile of an
explicit `3M`-root tiling of the same data.

Periodicity *is* the domain wrapping onto itself, so this is the
definitional test — and unlike a polynomial (which is discontinuous
across the seam, so no interpolation can reproduce it there) it works
for arbitrary data. Refinement is driven towards the seam so the
coarse/fine interfaces sit exactly where the wraparound happens.

Returns `(maxdiff, ncells)`, or `nothing` if the two refinement patterns
failed to correspond.
"""
function periodic_vs_tiled(::Val{D}, M::Int; N=4, G=1, nvars=2,
                           ops=Operators(prolongation=2, restriction=2),
                           passes=2) where {D}
    L = Float64(M)
    seam(c, lvl) = lvl < passes &&
        all(d -> min(mod(c[d], L), L - mod(c[d], L)) < 0.55, 1:D)

    periodic = Forest(ntuple(_ -> M, D); N=N, periodic=ntuple(_ -> true, D),
                      extents=ntuple(_ -> (0.0, L), D))
    refine_where!(periodic, seam, passes)

    tiled = Forest(ntuple(_ -> 3M, D); N=N, extents=ntuple(_ -> (-L, 2L), D))
    refine_where!(tiled, seam, passes)

    data = (x, v) -> sum(sin(3.1 * mod(x[d], L) + 0.7v) * (1 + 0.3d) for d in 1:D)

    fsp = FieldSet(periodic, nvars; G=G)
    fill_by_coordinates!(data, fsp)
    fill_ghosts!(fsp, GhostSchedule(fsp, ops))

    fst = FieldSet(tiled, nvars; G=G)
    fill_by_coordinates!(data, fst)
    fill_ghosts!(fst, GhostSchedule(fst, ops); boundary=boundary_by_coordinates(data))

    lower(forest, k) = ntuple(d -> block_extent(forest, k)[d][1], D)
    index = Dict((level(k), lower(tiled, k)) => b for (b, k) in enumerate(tiled.leaves))

    worst = 0.0
    ncells = 0
    for (b, k) in enumerate(periodic.leaves)
        tb = get(index, (level(k), lower(periodic, k)), nothing)
        tb === nothing && return nothing
        for v in 1:nvars
            a, c = blockview(fsp, b, v), blockview(fst, tb, v)
            for i in CartesianIndices(a)
                worst = max(worst, abs(a[i] - c[i]))
                ncells += 1
            end
        end
    end
    return (worst, ncells)
end

# --- Interface restriction (M8b) -----------------------------------------

"""
Every closed-range point of every block, keyed by `(level, exact
position)`.

Two blocks at one level that name the same point store the same number
once the data comes from [`arbitrary_value`](@ref), so the map is well
defined even though the closed ranges of neighbouring blocks overlap on
their shared plane.
"""
function closed_values(fs::FieldSet{T,D}) where {T,D}
    out = Dict{Tuple{Int,NTuple{D,Rational{Int}}},Vector{T}}()
    c = staggers(fs)
    closed = CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + fs.forest.N + c[d]), D))
    for b in 1:nblocks(fs), idx in closed
        out[(level(blockkey(fs, b)), exact_point(fs, b, Tuple(idx)))] =
            T[fs.work[Tuple(idx)..., v, b] for v in 1:fs.nvars]
    end
    return out
end

"""
The interface restriction computed by hand, as `(block, stored index) =>
expected value` for every point it should touch.

For every block, every vertex-like dimension and both sides,
`neighbor_keys` says whether the neighbours across that face are finer;
where they are, the block's boundary plane must come back as the average
of the coincident fine values. *Which* fine values those are is decided
by position and nothing else — a cell-like tangential dimension
contributes the two fine points half a fine spacing to either side, a
vertex-like one the single coincident point — from the exact rational
geometry of [`exact_point`](@ref), never from the package's stencils or
target ranges.

`values` is [`closed_values`](@ref) of the *pristine* field set, taken
before the restriction runs.
"""
function interface_targets(fs::FieldSet{T,D}, values) where {T,D}
    forest = fs.forest
    N, G, c = forest.N, fs.G, staggers(fs)
    wrap(d, x) = forest.periodic[d] ? mod(x, forest.roots[d]) : x
    out = Dict{Tuple{Int,NTuple{D,Int}},Vector{T}}()
    for b in 1:nblocks(fs)
        k = blockkey(fs, b)
        l = level(k)
        half = 1 // (2 * N * (1 << (l + 1)))        # half a fine spacing, in root cells
        for d in 1:D
            c[d] == 1 || continue
            for s in (-1, 1)
                δ = ntuple(e -> e == d ? s : 0, D)
                nbrs = neighbor_keys(forest, k, δ)
                (isempty(nbrs) || level(first(nbrs)) <= l) && continue
                # 2:1 balance, so the finer side is exactly one level down.
                @assert all(nbr -> level(nbr) == l + 1, nbrs)
                plane = s == 1 ? G[d] + N + 1 : G[d] + 1
                rng = ntuple(e -> e == d ? (plane:plane) :
                             ((G[e] + 1):(G[e] + N + c[e])), D)
                shifts = Iterators.product(ntuple(e -> (e != d && c[e] == 0) ?
                                                  (-1, 1) : (0,), D)...)
                for idx in CartesianIndices(rng)
                    p = exact_point(fs, b, Tuple(idx))
                    acc = zeros(T, fs.nvars)
                    n = 0
                    for sh in shifts
                        q = ntuple(e -> wrap(e, p[e] + sh[e] * half), D)
                        acc .+= values[(l + 1, q)]
                        n += 1
                    end
                    out[(b, Tuple(idx))] = acc ./ n
                end
            end
        end
    end
    return out
end

"""
How many times each phase of an [`InterfaceSchedule`](@ref) writes each
stored point: one count array per phase, in phase order.

A phase must write every target exactly once — the fixup overwrites a
block's own computed values, so a double write would make the result
depend on which of two fine sources landed last.
"""
function interface_write_counts(isched::InterfaceSchedule{T,D}) where {T,D}
    forest = isched.forest
    c = staggers(isched.centering)
    stored = ntuple(d -> forest.N + 2 * isched.G[d] + c[d], D)
    return map(isched.phases) do groups
        counts = zeros(Int, stored..., nleaves(forest))
        for group in groups
            blen = boxsize(group)
            first = ntuple(d -> group.stencils[d].targetfirst, D)
            for t in 1:ntransfers(group)
                b = group.targetblocks[t]
                for off in CartesianIndices(ntuple(d -> 0:(blen[d] - 1), D))
                    counts[ntuple(d -> first[d] + off[d], D)..., b] += 1
                end
            end
        end
        counts
    end
end

"""
Every `(block, stored index)` an [`InterfaceSchedule`](@ref) writes and
every one it reads, as two sets over all phases.
"""
function interface_touches(isched::InterfaceSchedule{T,D}) where {T,D}
    targets = Set{Tuple{Int,NTuple{D,Int}}}()
    sources = Set{Tuple{Int,NTuple{D,Int}}}()
    for groups in isched.phases, group in groups
        blen = boxsize(group)
        tfirst = ntuple(d -> group.stencils[d].targetfirst, D)
        srcstart = ntuple(d -> group.stencils[d].srcstart, D)
        width = ntuple(d -> size(group.stencils[d].weights, 1), D)
        for t in 1:ntransfers(group)
            tb, sb = Int(group.targetblocks[t]), Int(group.sourceblocks[t])
            for off in CartesianIndices(ntuple(d -> 0:(blen[d] - 1), D))
                o = Tuple(off)
                push!(targets, (tb, ntuple(d -> tfirst[d] + o[d], D)))
                base = ntuple(d -> Int(srcstart[d][o[d] + 1]), D)
                for m in CartesianIndices(ntuple(d -> 0:(width[d] - 1), D))
                    push!(sources, (sb, ntuple(d -> base[d] + Tuple(m)[d], D)))
                end
            end
        end
    end
    return targets, sources
end
