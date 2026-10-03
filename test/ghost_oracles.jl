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

# --- Reflecting faces (M10) ----------------------------------------------

"""
What the two faces of one dimension can be: wrapped onto each other,
both outer (the hook's), one of the two reflecting, or both reflecting.
"""
const FACE_KINDS = (:periodic, :outer, :reflect_lo, :reflect_hi, :reflect_both)

face_periodic(kind::Symbol) = kind === :periodic
face_reflecting(kind::Symbol) = (kind === :reflect_lo || kind === :reflect_both,
                                 kind === :reflect_hi || kind === :reflect_both)

"""
A brick of side `roots` (3 in 1D, 2 otherwise) whose faces are `kinds`,
refined twice at the low and at the high corner.

That puts all three levels against the walls, with the coarse-fine faces
meeting a wall *tangentially* — a level-2 block, a level-1 block and a
level-0 block in a row along it — which is the case where a mirrored
ghost is the block's own prolongated ghost, and where filling it at the
wrong time reads it undefined.
"""
function faces_forest(kinds::NTuple{D,Symbol}; T::Type=Float64, N=8) where {D}
    roots = D == 1 ? 3 : 2
    forest = Forest{T}(ntuple(_ -> roots, D); N=N,
                       periodic=map(face_periodic, kinds),
                       reflecting=map(face_reflecting, kinds),
                       extents=ntuple(_ -> (0, roots), D))
    corner(c, r) = all(d -> c[d] < r, 1:D) || all(d -> c[d] > roots - r, 1:D)
    refine_where!(forest, (c, lvl) -> (lvl == 0 && corner(c, 1.0)) ||
                                      (lvl == 1 && corner(c, 0.5)), 2)
    return forest
end

"""
Data with a definite parity about every reflecting wall, polynomial of
degree `< p` along every dimension so that operators of order `p`
reproduce it exactly — together with the `parity` declaring it.

Along each dimension the data is a cubic `c₀ + c₁y + c₂y² + c₃y³` in
`y = x - w`, with the coefficients chosen by the kind of the two faces:

- one reflecting wall at `w`: even (`c₀ + c₂y²`) for variable 1, odd
  (`c₁y + c₃y³`) for variable 2;
- reflecting at both ends: constant, since no nonconstant polynomial is
  even about two points, and so both variables are even there;
- periodic: constant, since no nonconstant polynomial is periodic;
- outer at both ends: a general polynomial, and the hook supplies it.

Terms of degree `≥ p` are dropped. The parity is `NoParity` wherever
the dimension has no reflecting face, which is the only place the
package accepts it.

The coefficients are captured as `T` values, so the callback is a pure
`T` function and can be launched on a device without fp64.
"""
function parity_data(::Type{T}, kinds::NTuple{D,Symbol}, p::Int;
                     flip::Bool=false) where {T,D}
    roots = D == 1 ? 3 : 2
    walls = ntuple(d -> T(kinds[d] === :reflect_hi ? roots : 0), D)
    keep(e) = e < p ? 1 : 0
    coef(v, d) = begin
        kind = kinds[d]
        if kind === :periodic || kind === :reflect_both
            (1 + d / 10, 0, 0, 0)
        elseif kind === :outer
            (0.3 + (d + v) / 10, 0.2, 0.1 * keep(2), 0.05 * keep(3))
        elseif v == 1
            (0.5, 0, 0.3 * keep(2), 0)
        else
            (0, 0.7, 0, 0.2 * keep(3))
        end
    end
    cs = ntuple(v -> ntuple(d -> map(T, coef(v, d)), D), 2)
    f = function (x, v)
        acc = one(eltype(x))
        for d in 1:D
            c = cs[v][d]
            y = x[d] - walls[d]
            acc *= ((c[4] * y + c[3]) * y + c[2]) * y + c[1]
        end
        return acc
    end
    single(kind) = kind === :reflect_lo || kind === :reflect_hi
    parity = [ntuple(D) do d
                  kind = kinds[d]
                  kind === :reflect_both && return EvenParity
                  single(kind) || return NoParity
                  odd = (v == 2) ⊻ flip
                  return odd ? OddParity : EvenParity
              end for v in 1:2]
    return f, parity
end

"""
The ghost width order-`p` operators need along each dimension of
centering `C`: `p/2` in a cell-centered dimension, `p/2 − 1` along a
stagger (see [`check_operators`](@ref)), and at least one.
"""
ghosts_for(C::NTuple{D,Symbol}, p::Int) where {D} =
    ntuple(d -> max(1, C[d] === :vertex ? p ÷ 2 - 1 : cld(p, 2)), D)

"""
Fill a [`faces_forest`](@ref) with [`parity_data`](@ref) after first
setting **every stored value to `NaN`**, exchange ghosts once, and
report `(nnan, worst)`: how many values are still `NaN`, and the worst
deviation of any stored value from the data.

`NaN` is what makes the fill's *order* visible. A ghost the schedule
never writes stays `NaN`; one it writes from a ghost that has not been
written yet becomes `NaN`, since every stencil weight is nonzero and
`NaN` survives any finite combination. A zero-initialized field set shows
neither: a stale zero is just a wrong number, and may even be the right
one.
"""
function undefined_ghosts(kinds::NTuple{D,Symbol}, C::NTuple{D,Symbol};
                          T::Type=Float64, p::Int=4, N::Int=8,
                          family=PointValue, backend=CPU()) where {D}
    forest = faces_forest(kinds; T=T, N=N)
    f, parity = parity_data(T, kinds, p)
    ops = family === Conservative ?
          Operators(prolongation=p, restriction=2, family=Conservative) :
          Operators(prolongation=p, restriction=p)
    fs = FieldSet{T}(forest, 2; G=ghosts_for(C, p), centering=C, parity=parity,
                     backend=backend)
    fill!(fs.work, T(NaN))
    fill_by_coordinates!(f, fs)
    fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=boundary_by_coordinates(f))
    work = Array(fs.work)
    nnan = count(isnan, work)
    worst = 0.0
    for b in 1:nblocks(fs), v in 1:2
        for idx in CartesianIndices(size(work)[1:D])
            x = coordinates(fs, b, Tuple(idx))
            worst = max(worst, Float64(abs(work[idx, v, b] - f(x, v))))
        end
    end
    return nnan, worst
end

"""
How many transfers of each kind the schedule *mirrors* across a
reflecting face — so a test can show that each kind actually occurs.
"""
function mirror_counts(s::GhostSchedule)
    counts = Dict(:copy => 0, :restrict => 0, :prolong => 0)
    for g in Iterators.flatten((s.phase1, Iterators.flatten(s.phase2)))
        g.factorcol == 0 || (counts[g.kind] += ntransfers(g))
    end
    return counts
end

"""
Compare a domain with a reflecting face at `x₁ = 0` against the doubled
domain it is the half of, holding the mirrored data explicitly.

The mirror *is* the doubled domain folded onto itself, so this is the
definitional test, as [`periodic_vs_tiled`](@ref) is for periodicity,
and it works for data with no polynomial structure. The refinement is
mirror symmetric and runs into the wall, with a coarse-fine face meeting
it tangentially, so mirrored copies, restrictions and prolongations all
occur. The remaining dimensions are periodic on both domains.
`side = :lo` puts the half domain on `[0, 2]` with its wall below,
`:hi` on `[-2, 0]` with its wall above.

Returns `(maxdiff, ncells)`, or `nothing` if the two refinement patterns
failed to correspond.
"""
function reflecting_vs_doubled(::Val{D}; side::Symbol, centering::NTuple{D,Symbol},
                               N=8, p=4) where {D}
    ops = Operators(prolongation=p, restriction=p)
    G = ghosts_for(centering, p)
    others = ntuple(d -> d == 1 ? false : true, D)
    xext = side === :lo ? (0.0, 2.0) : (-2.0, 0.0)
    half = Forest(ntuple(_ -> 2, D); N=N, periodic=others,
                  reflecting=ntuple(d -> d == 1 ? (side === :lo, side === :hi) :
                                                  (false, false), D),
                  extents=ntuple(d -> d == 1 ? xext : (0.0, 2.0), D))
    full = Forest(ntuple(d -> d == 1 ? 4 : 2, D); N=N, periodic=others,
                  extents=ntuple(d -> d == 1 ? (-2.0, 2.0) : (0.0, 2.0), D))
    # Symmetric under x₁ -> -x₁, and reaching the wall at x₁ = 0.
    pred(c, lvl) = (lvl == 0 && abs(c[1]) < 1 && (D == 1 || c[2] < 1)) ||
                   (lvl == 1 && abs(c[1]) < 0.5 && (D == 1 || c[2] < 0.5))
    refine_where!(half, pred, 2)
    refine_where!(full, pred, 2)

    # Even and odd about x₁ = 0, periodic in the rest, and no polynomial.
    shape(x, v) = sin(3.1 * abs(x[1]) + 0.7v) *
                  prod((1 + 0.3 * sin(π * x[d] + 0.2d) for d in 2:D); init=1.0)
    data(x, v) = v == 1 ? shape(x, v) : sign(x[1]) * shape(x, v)
    parity = [ntuple(d -> d == 1 ? EvenParity : NoParity, D),
              ntuple(d -> d == 1 ? OddParity : NoParity, D)]

    fsh = FieldSet(half, 2; G=G, centering=centering, parity=parity)
    fill_by_coordinates!(data, fsh)
    fill_ghosts!(fsh, GhostSchedule(fsh, ops); boundary=boundary_by_coordinates(data))

    fsf = FieldSet(full, 2; G=G, centering=centering)
    fill_by_coordinates!(data, fsf)
    fill_ghosts!(fsf, GhostSchedule(fsf, ops); boundary=boundary_by_coordinates(data))

    lower(forest, k) = ntuple(d -> block_extent(forest, k)[d][1], D)
    index = Dict((level(k), lower(full, k)) => b for (b, k) in enumerate(full.leaves))
    worst = 0.0
    ncells = 0
    for (b, k) in enumerate(half.leaves)
        fb = get(index, (level(k), lower(half, k)), nothing)
        fb === nothing && return nothing
        for v in 1:2
            a, c = blockview(fsh, b, v), blockview(fsf, fb, v)
            for i in CartesianIndices(a)
                worst = max(worst, abs(a[i] - c[i]))
                ncells += 1
            end
        end
    end
    return (worst, ncells)
end

# --- M12: rotating seams ----------------------------------------------------
#
# The quadrant's forest is checked against the *unfolded* forest: the whole
# plane, with the quadrant's leaves copied into the other three quadrants
# by turning their boxes, in exact `Rational` arithmetic. The unfolded
# forest has no seam, so its neighbors are found by the ordinary search,
# and none of the seam's key arithmetic is reused to build it.

"""
The box `(lo, hi)` turned by `r` quarter turns, `R^r`, about the axis at
the origin of the `(d1, d2)` plane, where `R` takes `e_{d1}` to `e_{d2}`
and `e_{d2}` to `−e_{d1}`: `R(x1, x2) = (−x2, x1)`. A negative `r` turns
the other way.
"""
function rotate_box(box, r::Int, d1::Int, d2::Int)
    lo, hi = box
    for _ in 1:mod(r, 4)
        lo, hi = Base.setindex(Base.setindex(lo, -hi[d2], d1), lo[d1], d2),
                 Base.setindex(Base.setindex(hi, -lo[d2], d1), hi[d1], d2)
    end
    return lo, hi
end

"""
Which quarter turn of the quadrant `x_{d1}, x_{d2} ≥ 0` the box lies in,
with the axis at the origin: 1 for `x_{d1} ≤ 0 ≤ x_{d2}`, 2 for both
negative, 3 for `x_{d2} ≤ 0 ≤ x_{d1}`, 0 in the quadrant itself.
"""
function box_orientation(box, d1::Int, d2::Int)
    lo, hi = box
    a, b = hi[d1] <= 0, hi[d2] <= 0
    return a ? (b ? 2 : 1) : (b ? 3 : 0)
end

"""
The key of the node whose box, in root units from the brick's low
corner, is `(lo, hi)` in a brick of `roots`: its level from its size, its
root from where it begins, dimension 1 fastest.
"""
function box_key(roots::NTuple{D,Int}, (lo, hi)) where {D}
    size = hi[1] - lo[1]
    @assert all(d -> hi[d] - lo[d] == size, 1:D) && numerator(size) == 1
    lvl = trailing_zeros(denominator(size))
    @assert denominator(size) == 1 << lvl
    pos = ntuple(d -> floor(Int, lo[d]), D)
    root = sum((pos[d] * prod(roots[1:(d - 1)]; init=1) for d in 1:D); init=0)
    coords = ntuple(d -> Int((lo[d] - pos[d]) * (1 << lvl)), D)
    return MortonKey{D}(root, lvl, coords)
end

"""
The unfolded forest of the rotating quadrant `quad`: `2M × 2M` roots in
the plane of the rotation, the quadrant's `M × M` turned into each of the
four quarters, the other dimensions as the quadrant's. Its leaves are the
quadrant's leaves turned by `r = 0, 1, 2, 3`. Returns `(full, image,
back)`: `image(k, r)` is the key in `full` of quadrant leaf `k` turned by
`r`, and `back(kf)` is `(r, k)` for a leaf `kf` of `full`. The leaf list is
installed without the package's checks, since a quadrant drawn at random
is not balanced.
"""
function unfolded_forest(quad::Forest{D}) where {D}
    d1, d2 = TreeAMR.rotating_dims(quad)
    M = quad.roots[d1]
    inplane(d) = d == d1 || d == d2
    roots = ntuple(d -> inplane(d) ? 2M : quad.roots[d], D)
    full = Forest(roots; N=quad.N, periodic=quad.periodic, reflecting=quad.reflecting)
    # From coordinates about the axis to the unfolded brick's, and back.
    shift = ntuple(d -> inplane(d) ? M : 0, D)
    image(k, r) = box_key(roots, map(c -> c .+ shift, rotate_box(leafbox(quad, k), r,
                                                                 d1, d2)))
    function back(kf)
        box = map(c -> c .- shift, leafbox(full, kf))
        r = box_orientation(box, d1, d2)
        return r, box_key(quad.roots, rotate_box(box, -r, d1, d2))
    end
    leaves = sort!([image(k, r) for k in quad.leaves for r in 0:3]; lt=naive_isless)
    TreeAMR.rebuild_leaves!(full, leaves)
    return full, image, back
end

"""A block's offset within its parent, from its coordinates alone."""
parity_offset(k::MortonKey{D}) where {D} = ntuple(d -> Int(k.coords[d]) & 1, D)

"""
Compare the quadrant's oriented neighbor search against the unfolded
forest, for every leaf and every direction: the keys `oriented_neighbors`
returns must be the unfolded forest's neighbors turned back into the
quadrant, all of one orientation, which is the `r` it returns; with no
neighbor, `r` must still be the orientation of the region stepped into.
Every finer neighbor's `virtual_offset` must be its child offset in the
unfolded forest, where the asking block sees it. Returns `(mismatches,
seen)`, `seen` counting the `(r, kind)` cases met, so that a test can
show that each occurs.
"""
function seam_neighbor_mismatches(quad::Forest{D}) where {D}
    d1, d2 = TreeAMR.rotating_dims(quad)
    full, image, back = unfolded_forest(quad)
    mismatches = 0
    seen = Dict{Tuple{Int,Symbol},Int}()
    for k in quad.leaves, δ in alldirections(Val(D))
        r, keys = TreeAMR.oriented_neighbors(quad, k, δ)
        unfolded = neighbor_keys(full, image(k, 0), δ)
        expected = map(back, unfolded)
        ok = if isempty(expected)
            lo, hi = leafbox(quad, k)
            s = hi[1] - lo[1]
            stepped = (lo .+ s .* δ, hi .+ s .* δ)
            isempty(keys) && r == box_orientation(stepped, d1, d2)
        else
            all(e -> first(e) == r, expected) &&
                sort(keys; lt=naive_isless) == sort(last.(expected); lt=naive_isless) &&
                all(zip(unfolded, expected)) do (kf, (_, kq))
                    level(kf) <= level(k) ||
                        TreeAMR.virtual_offset(parity_offset(kq), r, d1, d2) ==
                        parity_offset(kf)
                end
        end
        mismatches += !ok
        kind = isempty(keys) ? :none : level(first(keys)) == level(k) ? :same :
               level(first(keys)) < level(k) ? :coarser : :finer
        seen[(r, kind)] = get(seen, (r, kind), 0) + 1
    end
    return mismatches, seen
end

"""
Whether the unfolded forest `full` is conforming across the images of
the seam, the lines where `x_{d1}` or `x_{d2}` is zero: every two leaves
that share a face across one of them are at one level.
"""
function seam_conforming(full::Forest{D}, d1::Int, d2::Int) where {D}
    M = full.roots[d1] ÷ 2
    for kf in full.leaves, d in (d1, d2), s in (-1, 1)
        δ = ntuple(e -> e == d ? s : 0, D)
        hi = leafbox(full, kf)[2][d]
        for nf in neighbor_keys(full, kf, δ)
            across = (hi <= M) != (leafbox(full, nf)[2][d] <= M)
            across && level(nf) != level(kf) && return false
        end
    end
    return true
end

"""
A rotating quadrant refined and coarsened at random, unbalanced: `M`
roots along each dimension of the plane `rotating`, and the other
dimensions `other` — `:outer`, `:periodic` or `:reflecting` at their
low face — with one or two roots each.
"""
function random_rotating_forest(rng, ::Val{D}; rotating, M, other=:outer, nsteps,
                                maxlvl) where {D}
    inplane(d) = d in rotating
    roots = ntuple(d -> inplane(d) ? M : rand(rng, 1:2), D)
    periodic = ntuple(d -> !inplane(d) && other === :periodic, D)
    reflecting = ntuple(d -> (!inplane(d) && other === :reflecting, false), D)
    forest = Forest(roots; N=4, periodic=periodic, reflecting=reflecting,
                    rotating=rotating)
    for _ in 1:nsteps
        k = rand(rng, forest.leaves)
        if level(k) < maxlvl && rand(rng) < 0.75
            refine!(forest, k)
        elseif level(k) > 0
            p = parentkey(k)
            all(c -> isleaf(forest, c), childkeys(p)) && coarsen!(forest, p)
        end
    end
    return forest
end

# --- M12: rotated ghosts (steps 2–3) -----------------------------------------
#
# The ghosts across a rotating seam are checked against data covariant
# under the quarter turn, written out by formula, and against the full
# plane the quadrant is a quarter of, which holds the turned data
# explicitly and has no seam: the definitional test, as
# `reflecting_vs_doubled` is for a mirror. The formulas are in the
# coordinates about the axis, `(a, b) = (x_{d1}, x_{d2})`, where the turn
# is `R(a, b) = (−b, a)`; the quadrant spans `[0, 2]` in the plane and the
# full plane `[−2, 2]`.

"""
The dimension outside the plane `rotating` in 3D, or 0 in 2D: by
arithmetic rather than `setdiff`, which allocates a `Set` that a device
kernel evaluating the data cannot.
"""
outofplane(D, rotating) = D == 2 ? 0 : 6 - rotating[1] - rotating[2]

"""
The signed map of a quarter turn for a scalar followed by a vector's
components in Cartesian order: the component along `d1` becomes minus
the one along `d2`, the one along `d2` the one along `d1`, and anything
else itself — written from the definition, `v(Rp) = R v(p)`, not from
the package.
"""
function vector_rotation(D, (d1, d2))
    comp(e) = 1 + e
    return [1; [e == d1 ? -comp(d2) : e == d2 ? comp(d1) : comp(e) for e in 1:D]]
end

"""
The parity of [`vector_rotation`](@ref)'s variables when the dimension
outside the plane reflects at its low face: the scalar and the in-plane
components even, the component along the wall's normal odd; `nothing`
otherwise.
"""
function vector_parity(D, rotating, other)
    other === :reflect_lo || return nothing
    z = outofplane(D, rotating)
    return [ntuple(d -> d != z ? NoParity : v == 1 + z ? OddParity : EvenParity, D)
            for v in 1:(D + 1)]
end

"""
The quarter-turn-covariant vector field `k` (1 or 2) at `(a, b)`: a radial
part `ρ(a, b)` and a swirl `σ(−b, a)`, with `ρ` and `σ` invariant under
the turn, as `(v_a, v_b)`. Invariants are functions of `a² + b²` and of
`χ = ab(a² − b²)`, which the turn keeps and a mirror negates, so the
field is covariant under rotations and not under reflections.
`poly = p` keeps it a polynomial of degree `< p` per dimension, computed
in the type of `a`, so that a `Float32` device can evaluate it.
"""
function covariant_vector(a, b, k::Int; poly::Int=0)
    c = literal(a)
    if poly > 0
        ρ2 = poly > 2 ? a^2 + b^2 : zero(a)
        radial = c(k == 1 ? 0.8 : -0.5) + c(0.25) * ρ2
        swirl = c(k == 1 ? -0.6 : 0.9) + c(0.15) * ρ2
    else
        ρ2 = a * a + b * b
        χ = a * b * (a * a - b * b)
        radial = k == 1 ? sin(c(0.9) * ρ2) + c(0.5) :
                 cos(c(0.7) * ρ2) - c(0.2) * sin(c(0.4) * χ)
        swirl = k == 1 ? c(0.7) * cos(c(0.6) * ρ2) + c(0.3) * sin(c(0.5) * χ) :
                exp(c(-0.3) * ρ2)
    end
    return radial * a - swirl * b, radial * b + swirl * a
end

"""
The quarter-turn-invariant scalar at `(a, b)`, chiral like the vectors.
"""
function invariant_scalar(a, b; poly::Int=0)
    c = literal(a)
    if poly > 0
        ρ2 = poly > 2 ? a^2 + b^2 : zero(a)
        χ = poly > 3 ? a^3 * b - a * b^3 : zero(a)
        return c(1.3) + c(0.4) * ρ2 + c(0.1) * χ
    end
    ρ2 = a * a + b * b
    χ = a * b * (a * a - b * b)
    return exp(c(-0.4) * ρ2) + c(0.3) * sin(c(0.7) * χ) +
           c(0.2) * cos(c(0.5) * (a^4 - 6a^2 * b^2 + b^4))
end

"""
The factor along the dimension outside the plane: 1 in 2D; with that
dimension periodic, outer or reflecting at its low face, a function
periodic on `[0, 2]`, a general one, or one even (`odd = false`) or odd
about 0. `poly = p` keeps it a polynomial of degree `< p` (a periodic
one is then constant).
"""
function outofplane_factor(x, D, rotating, other, odd::Bool; poly::Int=0)
    D == 2 && return one(eltype(x))
    t = x[outofplane(D, rotating)]
    keep(e) = poly == 0 || e < poly
    c = literal(t)
    if poly > 0
        z = zero(t)
        other === :periodic && return c(odd ? 0.5 : 1.0)
        other === :outer &&
            return c(0.4) + c(0.3) * t + (keep(2) ? c(0.2) * t^2 : z) +
                   (keep(3) ? c(0.05) * t^3 : z)
        return odd ? c(0.7) * t + (keep(3) ? c(0.2) * t^3 : z) :
               c(0.5) + (keep(2) ? c(0.3) * t^2 : z)
    end
    other === :periodic &&
        return odd ? c(0.6) + c(0.3) * sin(π * t + c(0.4)) :
               1 + c(0.3) * sin(π * t + c(0.2))
    other === :outer && return 1 + c(0.3) * sin(c(1.7) * t + c(0.2))
    return odd ? sin(c(1.3) * t) : cos(c(1.1) * t) + c(0.5)
end

# The formulas' constants in the real type of their argument, so that a
# `Float32` device evaluates them without `Float64`: the identity for
# `Float64`, and for the complex step, whose real type is `Float64`.
literal(a) = y -> convert(real(typeof(a)), y)

"""
A scalar and a covariant vector, as [`vector_rotation`](@ref) orders
them, as `f(x, v)`: smooth and not polynomial, or with `poly = p` of
degree `< p` per dimension, which order-`p` operators reproduce exactly.
"""
function rotating_data(D, rotating, other; poly::Int=0)
    d1, d2 = rotating
    z = outofplane(D, rotating)
    kind = Val(other)           # a `Symbol` is not plain data; a device kernel takes this
    return function (x, v)
        o = valsymbol(kind)
        a, b = x[d1], x[d2]
        v == 1 && return invariant_scalar(a, b; poly=poly) *
                         outofplane_factor(x, D, rotating, o, false; poly=poly)
        e = v - 1                                    # the component along e
        if e == z
            c = literal(a)
            return (c(0.9) + c(0.2) * invariant_scalar(a, b; poly=min(poly, 3))) *
                   outofplane_factor(x, D, rotating, o, true; poly=poly)
        end
        va, vb = covariant_vector(a, b, 1; poly=poly)
        return (e == d1 ? va : vb) * outofplane_factor(x, D, rotating, o, false; poly=poly)
    end
end

valsymbol(::Val{S}) where {S} = S

"""
The refinement both the quadrant and the full plane get: symmetric under
every quarter turn and every mirror of the plane, so that the seam's
conformity holds in the full plane by symmetry and the two balance the
same. Two passes put levels 2, 1 and 0 side by side along each seam
face, starting at the axis; in 3D they reach the low face of the third
dimension too, and stop halfway up it.
"""
function rotating_refinement(D, rotating)
    d1, d2 = rotating
    z = outofplane(D, rotating)
    low(c, r) = D == 2 || c[z] < r
    return (c, lvl) -> (lvl == 0 && hypot(c[d1], c[d2]) < 1.2 && low(c, 1.0)) ||
                       (lvl == 1 && hypot(c[d1], c[d2]) < 0.6 && low(c, 0.5))
end

"""
The quadrant of side 2 roots in the plane `rotating` (and 2 along the
third dimension, which is `other`: `:periodic`, `:outer` or `:reflect_lo`),
over `[0, 2]` everywhere, refined by [`rotating_refinement`](@ref).
"""
function rotating_forest(::Val{D}; rotating, other=:periodic, T::Type=Float64,
                         N=8) where {D}
    z = outofplane(D, rotating)
    forest = Forest{T}(ntuple(_ -> 2, D); N=N,
                       periodic=ntuple(d -> d == z && other === :periodic, D),
                       reflecting=ntuple(d -> (d == z && other === :reflect_lo, false), D),
                       rotating=rotating, extents=ntuple(_ -> (0, 2), D))
    return refine_where!(forest, rotating_refinement(D, rotating), 2)
end

"""
Fill a [`rotating_forest`](@ref) with the polynomial
[`rotating_data`](@ref) after setting every stored value to `NaN`,
exchange ghosts once, and report `(nnan, worst, schedule)`: the values
still `NaN`, the worst deviation of any stored value from the data, and
the schedule. As for [`undefined_ghosts`](@ref), a ghost never written
stays `NaN` and one written from an unwritten one becomes `NaN`, and the
data is reproduced exactly, so a wrong axis map, a wrong variable or a
wrong sign shows as a deviation.
"""
function undefined_rotated_ghosts(::Val{D}; rotating, other, C, p, T::Type=Float64,
                                  N=8, backend=CPU()) where {D}
    forest = rotating_forest(Val(D); rotating=rotating, other=other, T=T, N=N)
    f = rotating_data(D, rotating, other; poly=p)
    fs = FieldSet{T}(forest, D + 1; G=ghosts_for(C, p), centering=C,
                     rotation=vector_rotation(D, rotating),
                     parity=vector_parity(D, rotating, other), backend=backend)
    fill!(fs.work, T(NaN))
    fill_by_coordinates!(f, fs)
    schedule = GhostSchedule(fs, Operators(prolongation=p, restriction=p))
    fill_ghosts!(fs, schedule; boundary=boundary_by_coordinates(f))
    work = Array(fs.work)
    nnan = count(isnan, work)
    worst = 0.0
    for b in 1:nblocks(fs), idx in CartesianIndices(size(work)[1:D])
        x = coordinates(fs, b, Tuple(idx))
        for v in 1:fs.nvars
            worst = max(worst, Float64(abs(work[idx, v, b] - f(x, v))))
        end
    end
    return nnan, worst, schedule
end

"""How many transfers of each `(kind, orientation)` the schedule holds."""
function rotation_counts(s::GhostSchedule)
    counts = Dict{Tuple{Symbol,Int},Int}()
    for g in Iterators.flatten((s.phase1, Iterators.flatten(s.phase2)))
        key = (g.kind, Int(g.orientation))
        counts[key] = get(counts, key, 0) + ntransfers(g)
    end
    return counts
end

"""
Whether every region the schedule hands the hook leaves the domain
through an outer face, judged on the unfolded plane: its image, the
block's box stepped by the direction with the reflected components
dropped, lies beyond `±M` in the plane or beyond a non-periodic face of
the third dimension. In exact `Rational` units of root cells, with the
axis at the quadrant's low corner.
"""
function hook_regions_leave(schedule::GhostSchedule{T,D}) where {T,D}
    forest = schedule.forest
    d1, d2 = TreeAMR.rotating_dims(forest)
    M = forest.roots[d1]
    return all(schedule.boundaries) do region
        k = forest.leaves[region.block]
        δ′, _ = TreeAMR.reflect_direction(forest, k, region.direction)
        lo, hi = leafbox(forest, k)
        s = hi[1] - lo[1]
        slo, shi = lo .+ s .* δ′, hi .+ s .* δ′
        any(1:D) do d
            d == d1 || d == d2 ? (shi[d] > M || slo[d] < -M) :
            !forest.periodic[d] && (slo[d] < 0 || shi[d] > forest.roots[d])
        end
    end
end

"""
The worst difference between every stored point of every block of
`fsq`, ghosts included, and the block of `fsf` at the same level and
lower corner, with the count of points compared; `nothing` if some block
of `fsq` has no match, which means the two refinements differ.
"""
function compare_matching_blocks(fsq::FieldSet{T,D}, fsf::FieldSet{T,D}) where {T,D}
    lower(fs, b) = ntuple(d -> block_extent(fs.forest, blockkey(fs, b))[d][1], D)
    index = Dict((level(blockkey(fsf, b)), lower(fsf, b)) => b for b in 1:nblocks(fsf))
    worst = 0.0
    npoints = 0
    for b in 1:nblocks(fsq)
        fb = get(index, (level(blockkey(fsq, b)), lower(fsq, b)), nothing)
        fb === nothing && return nothing
        for v in 1:fsq.nvars
            a, c = blockview(fsq, b, v), blockview(fsf, fb, v)
            for i in CartesianIndices(a)
                # `==` semantics: −0 and +0 agree, and a `NaN` on either
                # side is a mismatch that `max` keeps.
                worst = a[i] == c[i] ? worst : max(worst, Float64(abs(a[i] - c[i])))
                npoints += 1
            end
        end
    end
    return worst, npoints
end

"""
The quadrant and the full plane it is a quarter of, refined alike by
[`rotating_refinement`](@ref), the third dimension `other` in both.
"""
function quadrant_and_full(::Val{D}; rotating, other, N) where {D}
    z = outofplane(D, rotating)
    inplane(d) = d in rotating
    periodic = ntuple(d -> d == z && other === :periodic, D)
    reflecting = ntuple(d -> (d == z && other === :reflect_lo, false), D)
    quad = Forest(ntuple(_ -> 2, D); N=N, periodic=periodic, reflecting=reflecting,
                  rotating=rotating, extents=ntuple(_ -> (0.0, 2.0), D))
    full = Forest(ntuple(d -> inplane(d) ? 4 : 2, D); N=N, periodic=periodic,
                  reflecting=reflecting,
                  extents=ntuple(d -> inplane(d) ? (-2.0, 2.0) : (0.0, 2.0), D))
    pred = rotating_refinement(D, rotating)
    refine_where!(quad, pred, 2)
    refine_where!(full, pred, 2)
    return quad, full
end

"""
Compare a quadrant with a rotating seam against the full plane holding
the turned data explicitly, for the smooth [`rotating_data`](@ref) — a
scalar and a vector, covariant under the quarter turn and not
polynomial — with `centering`. Both are `NaN`-prefilled, filled and
exchanged once, the outer faces through the hook from the same formula.
Returns [`compare_matching_blocks`](@ref)'s `(worst, npoints)`, or
`nothing` if the refinements failed to correspond.
"""
function rotating_vs_quadrupled(::Val{D}; rotating, centering::NTuple{D,Symbol},
                                other=:periodic, N=8, p=4) where {D}
    quad, full = quadrant_and_full(Val(D); rotating=rotating, other=other, N=N)
    ops = Operators(prolongation=p, restriction=p)
    G = ghosts_for(centering, p)
    f = rotating_data(D, rotating, other)
    parity = vector_parity(D, rotating, other)
    fsq = FieldSet(quad, D + 1; G=G, centering=centering, parity=parity,
                   rotation=vector_rotation(D, rotating))
    fsf = FieldSet(full, D + 1; G=G, centering=centering, parity=parity)
    for fs in (fsq, fsf)
        fill!(fs.work, NaN)
        fill_by_coordinates!(f, fs)
        fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=boundary_by_coordinates(f))
    end
    return compare_matching_blocks(fsq, fsf)
end

"""
The same for a `RotationPair`: by default `a` face-centered normal to
`d1`, holding `(B_{d1}, F_{d1})`, and `b` normal to `d2`, holding
`(F_{d2}, B_{d2})` — the order swapped, so that the maps permute — with
`B` and `F` the two covariant vector fields, `F` odd about a reflecting
low face of the third dimension and `B` even. `Ca` and `Ga` give `a`
another layout; `b`'s is its swap. The quadrant fills them as a pair,
the full plane set by set. Returns `(worst, npoints, (sa, sb))`, or
`nothing`.
"""
function rotating_pair_vs_quadrupled(::Val{D}; rotating, other=:periodic, N=8, p=4,
                                     Ca=facecentered(D, rotating[1]),
                                     Ga=ghosts_for(Ca, p)) where {D}
    d1, d2 = rotating
    swap(t) = Base.setindex(Base.setindex(t, t[d2], d1), t[d1], d2)
    Cb, Gb = swap(Ca), swap(Ga)
    quad, full = quadrant_and_full(Val(D); rotating=rotating, other=other, N=N)
    ops = Operators(prolongation=p, restriction=p)
    field(x, k) = covariant_vector(x[d1], x[d2], k)
    zf(x, odd) = outofplane_factor(x, D, rotating, other, odd)
    B(x, e) = field(x, 1)[e] * zf(x, false)
    F(x, e) = field(x, 2)[e] * zf(x, true)
    fa(x, v) = v == 1 ? B(x, 1) : F(x, 1)            # (B_{d1}, F_{d1})
    fb(x, v) = v == 1 ? F(x, 2) : B(x, 2)            # (F_{d2}, B_{d2})
    z = outofplane(D, rotating)
    parity(odd) = other === :reflect_lo ?
                  [ntuple(d -> d != z ? NoParity : o ? OddParity : EvenParity, D)
                   for o in odd] : nothing
    qa = FieldSet(quad, 2; G=Ga, centering=Ca, parity=parity((false, true)),
                  rotation=(-2, -1))
    qb = FieldSet(quad, 2; G=Gb, centering=Cb, parity=parity((true, false)),
                  rotation=(2, 1))
    sa, sb = GhostSchedule(qa, ops), GhostSchedule(qb, ops)
    for (fs, f) in ((qa, fa), (qb, fb))
        fill!(fs.work, NaN)
        fill_by_coordinates!(f, fs)
    end
    fill_ghosts!(RotationPair(qa, qb), (sa, sb);
                 boundary=(boundary_by_coordinates(fa), boundary_by_coordinates(fb)))
    worst, npoints = 0.0, 0
    for (q, C, G, f, par) in ((qa, Ca, Ga, fa, (false, true)),
                              (qb, Cb, Gb, fb, (true, false)))
        fsf = FieldSet(full, 2; G=G, centering=C, parity=parity(par))
        fill!(fsf.work, NaN)
        fill_by_coordinates!(f, fsf)
        fill_ghosts!(fsf, GhostSchedule(fsf, ops); boundary=boundary_by_coordinates(f))
        result = compare_matching_blocks(q, fsf)
        result === nothing && return nothing
        worst = max(worst, result[1])
        npoints += result[2]
    end
    return worst, npoints, (sa, sb)
end

# --- M12: regrid across the seam (step 4) ------------------------------------
#
# A regrid of the quadrant is checked against a regrid of the full plane
# to the turned image of the quadrant's new leaves: the full plane's
# flags are derived from where the quadrant went, in exact `Rational`
# boxes, so that both meshes stay each other's image, and then every
# stored point of the two is compared, as `rotating_vs_quadrupled` does
# for a single fill. The transfers move data that the fill before them
# turned across the seam; a wrong fill there shows in the prolonged
# blocks.

"""
The leaf of `quad` (in the quadrant, a key) that the full plane's leaf
`kf` is the image of, with the turn `r` that carries it there, by the
boxes: the full plane spans `2M` roots in the plane, with the axis at
`M`.
"""
function quadrant_preimage(quad::Forest{D}, full::Forest{D}, kf) where {D}
    d1, d2 = TreeAMR.rotating_dims(quad)
    M = quad.roots[d1]
    shift = ntuple(d -> d == d1 || d == d2 ? M : 0, D)
    box = map(c -> c .- shift, leafbox(full, kf))
    r = box_orientation(box, d1, d2)
    return r, box_key(quad.roots, rotate_box(box, -r, d1, d2))
end

"""
How a leaf `k` of the old leaves became the new leaves `new` (a `Set`):
`Keep` if it is still a leaf, `Refine` if its children are, `Coarsen` if
its parent is, and `nothing` for anything else — a move by more than one
level, which a regrid must never make.
"""
function regrid_move(k, new)
    k in new && return Keep
    all(c -> c in new, childkeys(k)) && return Refine
    level(k) > 0 && parentkey(k) in new && return Coarsen
    return nothing
end

"""
A boundary hook that writes `f(x, v)` into the region it is handed, for
the field set `fs`, and to `other(…)` for any other: a plain function
hook, so that several sets with different formulas can share one
`regrid!`.
"""
function formula_hook(pairs...)
    return function (fs, b, key, δ, region)
        f = last(pairs[findfirst(p -> first(p) === fs, pairs)])
        for idx in region, v in 1:fs.nvars
            fs.work[idx, v, b] = f(coordinates(fs, b, Tuple(idx)), v)
        end
        return nothing
    end
end

"""
Regrid a rotating quadrant through `passes`, each a `(forest, key) ->
RegridFlag` rule over the quadrant's leaves, and the full plane after it
to the turned image of each result; then fill both and compare. With
`pair = false` the quadrant holds the scalar and vector of
[`rotating_data`](@ref) with centering `C`; with `pair = true` it holds
the face-centered `(B, F)` pair of [`rotating_pair_vs_quadrupled`](@ref),
regridded as a `RotationPair`. Returns per pass `(moved, conforming,
images, worst, npoints, nleaves)`: whether every quadrant leaf moved by at
most one level, whether the quadrant is balanced and conforming, whether
the full plane's leaves are the four turns of the quadrant's, and the
worst deviation over every stored point.
"""
function rotating_regrid_vs_quadrupled(::Val{D}; rotating, other=:periodic, N=8, p=4,
                                       C=cellcentered(D), pair::Bool=false,
                                       passes) where {D}
    d1, d2 = rotating
    quad, full = quadrant_and_full(Val(D); rotating=rotating, other=other, N=N)
    ops = Operators(prolongation=p, restriction=p)
    z = outofplane(D, rotating)
    if pair
        Ca = facecentered(D, d1)
        swap(t) = Base.setindex(Base.setindex(t, t[d2], d1), t[d1], d2)
        Ga = ghosts_for(Ca, p)
        field(x, k) = covariant_vector(x[d1], x[d2], k)
        zf(x, odd) = outofplane_factor(x, D, rotating, other, odd)
        fa = (x, v) -> v == 1 ? field(x, 1)[1] * zf(x, false) : field(x, 2)[1] * zf(x, true)
        fb = (x, v) -> v == 1 ? field(x, 2)[2] * zf(x, true) : field(x, 1)[2] * zf(x, false)
        parity(odd) = other === :reflect_lo ?
                      [ntuple(d -> d != z ? NoParity : o ? OddParity : EvenParity, D)
                       for o in odd] : nothing
        qa = FieldSet(quad, 2; G=Ga, centering=Ca, parity=parity((false, true)),
                      rotation=(-2, -1))
        qb = FieldSet(quad, 2; G=swap(Ga), centering=swap(Ca),
                      parity=parity((true, false)), rotation=(2, 1))
        ffa = FieldSet(full, 2; G=Ga, centering=Ca, parity=parity((false, true)))
        ffb = FieldSet(full, 2; G=swap(Ga), centering=swap(Ca),
                       parity=parity((true, false)))
        qsets, fsets, formulas = [qa, qb], [ffa, ffb], [fa, fb]
    else
        f = rotating_data(D, rotating, other)
        par = vector_parity(D, rotating, other)
        G = ghosts_for(C, p)
        qsets = [FieldSet(quad, D + 1; G=G, centering=C, parity=par,
                          rotation=vector_rotation(D, rotating))]
        fsets = [FieldSet(full, D + 1; G=G, centering=C, parity=par)]
        formulas = [f]
    end
    for (fs, f) in Iterators.flatten((zip(qsets, formulas), zip(fsets, formulas)))
        fill_by_coordinates!(f, fs)
    end
    qhook = formula_hook((qsets .=> formulas)...)
    fhook = formula_hook((fsets .=> formulas)...)
    rpair = pair ? RotationPair(qsets...) : nothing
    qentries() = pair ? [rpair => Tuple(GhostSchedule(fs, ops) for fs in qsets)] :
                 [fs => GhostSchedule(fs, ops) for fs in qsets]
    function fillall!()
        if pair
            fill_ghosts!(rpair, Tuple(GhostSchedule(fs, ops) for fs in qsets);
                         boundary=qhook)
        else
            foreach(fs -> fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=qhook), qsets)
        end
        foreach(fs -> fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=fhook), fsets)
    end
    results = []
    for rule in passes
        old = copy(quad.leaves)
        regrid!(quad, qentries(); flags=[rule(quad, k) for k in quad.leaves],
                boundary=qhook)
        new = Set(quad.leaves)
        moved = all(k -> regrid_move(k, new) !== nothing, old)
        fflags = map(full.leaves) do kf
            _, kq = quadrant_preimage(quad, full, kf)
            something(regrid_move(kq, new), Keep)
        end
        regrid!(full, [fs => GhostSchedule(fs, ops) for fs in fsets]; flags=fflags,
                boundary=fhook)
        images = sort!([last(quadrant_preimage(quad, full, kf)) for kf in full.leaves];
                       lt=naive_isless) ==
                 sort!(repeat(quad.leaves, 4); lt=naive_isless) &&
                 all(kf -> isleaf(quad, last(quadrant_preimage(quad, full, kf))),
                     full.leaves)
        fillall!()
        worst, npoints = 0.0, 0
        for (q, f) in zip(qsets, fsets)
            result = compare_matching_blocks(q, f)
            result === nothing && (worst = Inf; break)
            worst = max(worst, result[1])
            npoints += result[2]
        end
        push!(results, (moved=moved, conforming=isbalanced(quad), images=images,
                        worst=worst, npoints=npoints, nleaves=nleaves(quad)))
    end
    return results
end
