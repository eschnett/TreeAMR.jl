# Point interpolation: the values of a field set, and their derivatives,
# at arbitrary points of the domain.
#
# Everything else in the package moves data between the mesh's own
# points; this is the one operation that answers "what is the field
# *here*" for a point the mesh did not choose — a horizon finder's
# surface, a tracer, a sampled ray. It is an analysis-cadence operation,
# batched over a whole array of points so that it runs as one launch.
# See "Point interpolation" in CODE.md for the argument; the short form:
#
#   * **Locate by one search.** A point is mapped to the finest-level
#     node containing it and the covering leaf is the last leaf not after
#     that node in curve order — one binary search, because an ancestor
#     sorts immediately before its contiguous subtree.
#   * **Read one block.** The *stencil* of a query is the `n^D` stored
#     points of the containing block the interpolant reads, ghosts
#     included. It never crosses into another block's array, so ghosts
#     must be current, as for any other stencil.
#   * **The basis is the extension point.** The kernel knows a basis only
#     through `stencilwidth`, `stencilstart` and `basisweights`; Lagrange
#     is the one implemented.
#   * **Errors after the kernel.** A point outside the domain is recorded
#     in its own slot and reported by the host once the launch is done: a
#     device cannot throw, and the CPU would wrap the exception.

# --- bases -------------------------------------------------------------------

"""
    InterpolationBasis

The one-dimensional interpolant [`interpolate`](@ref) builds its tensor
product from. [`Lagrange`](@ref) is the only one implemented.

A basis is the extension point for interpolants with other properties —
a smooth (`C¹`) one built from the same nodal data, say, where
Lagrange interpolation is only continuous. A new basis is a subtype
that is `isbits` (it becomes a kernel argument) and implements three
internal functions, which are everything the kernel asks of it:

- `TreeAMR.stencilwidth(basis)`: the number `n` of consecutive stored
  points per dimension it reads, a compile-time constant of the type;
- `TreeAMR.stencilstart(basis, s)`: the first of them, given the query's
  position `s` as a continuous stored index (stored point `i` sits at
  `s = i`); the caller clamps the result into the stored array;
- `TreeAMR.basisweights(basis, ξ, Val(M))`: the `n` weights and their
  first `M` derivatives at `ξ = s - first`, in node units, as an
  `(M+1)`-tuple of `n`-tuples.

and `TreeAMR.check_derivative(basis, m)` if it cannot produce every
derivative order below `n`.
"""
abstract type InterpolationBasis end

"""
    Lagrange(n)

Tensor-product Lagrange interpolation through `n` consecutive stored
points per dimension: exact on polynomials of degree `n − 1` in each
dimension, error `O(hⁿ)` for the value and `O(hⁿ⁻ᵐ)` for an `m`-th
derivative.

`n` has no default: the right order follows from what the field is used
for, as for [`Operators`](@ref). Interpolating an evolved state one order
above the scheme's accuracy keeps the interpolation error below the
solution's.

The stencil is centered on the query for even `n` and half a point off
for odd `n`, and in both cases it changes only *at a stored point*, where
every choice of stencil interpolates the same value. So the interpolant
is continuous inside a block — `C⁰`, not `C¹`: its derivative jumps at
the stored points. Near the edge of the stored array the stencil shifts
inward rather than extrapolate (see [`interpolate`](@ref)).
"""
struct Lagrange{n} <: InterpolationBasis
    function Lagrange{n}() where {n}
        n isa Int && n >= 1 || throw(ArgumentError(
            "a Lagrange basis needs at least one point per dimension, got n = $n"))
        return new{n}()
    end
end
Lagrange(n::Integer) = Lagrange{Int(n)}()

Base.show(io::IO, ::Lagrange{n}) where {n} = print(io, "Lagrange(", n, ")")

@inline stencilwidth(::Lagrange{n}) where {n} = n

# Centered for even `n` (the point lies between the two middle nodes),
# half a node low for odd `n`; either way the start moves only when `s`
# crosses an integer, i.e. a node — which is what makes the interpolant
# continuous. `floor(s − (n−1)/2)` would be the naive centering, and for
# even `n` it switches at the midpoints between nodes, where two stencils
# give different values.
@inline stencilstart(::Lagrange{n}, s) where {n} = floorindex(s) - (n - 1) ÷ 2

# A derivative the basis cannot produce, with the reason; `nothing` when
# it can. Lagrange's `m`-th derivative through `n` points is identically
# zero from `m = n` on, which is never what a caller meant.
function check_derivative(::Lagrange{n}, m::Int) where {n}
    m < n && return nothing
    return "a derivative of order $m through Lagrange($n) is identically zero: " *
           "the interpolant through $n points is a polynomial of degree $(n - 1) " *
           "in each dimension. Use at least Lagrange($(m + 1))."
end

# The Lagrange basis of the nodes `0, 1, …, n−1` at `ξ`, with its first
# `M` derivatives: `w[m+1][k]` is `ℓₖ⁽ᵐ⁾(ξ)` for node `k − 1`.
#
# The numerator `∏_{j≠k} (ξ − j)` is formed as a *truncated Taylor
# series* in `ε`, `∏_{j≠k} ((ξ − j) + ε)` kept to order `ε^M`, so its
# `m`-th coefficient times `m!` is the `m`-th derivative. That never
# divides by `ξ − j` — the barycentric form's `0/0` when the query sits on
# a node, which is the first thing an exactness test does — and needs no
# rational arithmetic, unlike the schedule's weights, which are exact
# because they are built once, and cached per offset, which a stream of
# arbitrary offsets would grow without bound. The denominators are
# integers, formed exactly and converted once. No floating-point literals
# (see "Precision" in CODE.md).
@inline function basisweights(::Lagrange{n}, ξ::T, ::Val{M}) where {n,T,M}
    w = ntuple(k -> lagrange_series(Val(n), k, ξ, Val(M)), Val(n))
    return ntuple(m -> ntuple(k -> w[k][m], Val(n)), Val(M + 1))
end

# `ℓₖ` and its first `M` derivatives at `ξ`. The series is threaded
# through `series_times` rather than rebuilt by a closure in the loop: a
# variable that is reassigned *and* captured by a closure is boxed, which
# costs an allocation per update and does not compile for a device.
@inline function lagrange_series(::Val{n}, k::Int, ξ::T, ::Val{M}) where {n,T,M}
    c = ntuple(m -> m == 1 ? one(T) : zero(T), Val(M + 1))
    den = 1
    for j in 1:n
        j == k && continue
        c = series_times(c, ξ - T(j - 1))
        den *= k - j
    end
    return series_derivatives(c, T(den))
end

# `c · (a + ε)`, truncated.
@inline series_times(c::NTuple{L,T}, a::T) where {L,T} =
    ntuple(m -> m == 1 ? a * c[1] : a * c[m] + c[m - 1], Val(L))

# The derivatives `m!·cₘ` of the series, divided by the denominator.
@inline series_derivatives(c::NTuple{L,T}, den::T) where {L,T} =
    ntuple(m -> T(factorial(m - 1)) * c[m] / den, Val(L))

# `floor(s)` as an `Int`, for the bounded arguments this file gives it (a
# stored index, a root position). A hardware float truncates directly.
# Another `AbstractFloat` may not convert to an integer at all —
# MultiFloats' `Float32x2` does not — but every one converts to
# `Float32`, which is exact on the small integer `floor(s)` is, and the
# exact comparisons in `T` repair the one case where it is not.
@inline floorindex(s::Base.IEEEFloat) = unsafe_trunc(Int, floor(s))
@inline function floorindex(s::T) where {T}
    f = floor(s)
    k = unsafe_trunc(Int, Float32(f))
    k -= T(k) > f
    k += T(k + 1) <= f
    return k
end

# --- excluded regions --------------------------------------------------------

"""
    Region

A region of the domain whose data an interpolation must not silently use:
[`interpolate`](@ref)'s `exclude` flags every query whose stencil has a
point inside it. [`Ellipsoid`](@ref) is the one implemented.

A new region is an `isbits` subtype implementing `TreeAMR.inside(region,
x)` for a point `x::NTuple{D,T}`, and `TreeAMR.convertregion(T, region)`
if its parameters must be brought to the field set's type. It may also
override `TreeAMR.stencil_hits` with something cheaper than the default,
which tests all `n^D` stencil points.
"""
abstract type Region end

"""
    Ellipsoid(center, semiaxes)

The open axis-aligned ellipsoid `Σ_d ((x_d − c_d)/a_d)² < 1`; a ball is
`Ellipsoid(c, (r, r, r))`. Used as [`interpolate`](@ref)'s `exclude`
region, for data that is not a solution inside it — the interior of a
black hole kept only as a damping layer, say.

Testing a stencil against it costs `O(D·n)`, not `O(n^D)`, and is exact
rather than conservative: the sum is separable, so the stencil point
nearest the center in the scaled metric is the per-dimension nearest,
and the stencil reaches inside iff that one point does.
"""
struct Ellipsoid{D,T} <: Region
    center::NTuple{D,T}
    semiaxes::NTuple{D,T}
    function Ellipsoid{D,T}(center, semiaxes) where {D,T}
        all(a -> a > 0, semiaxes) || throw(ArgumentError(
            "an ellipsoid's semiaxes must be positive, got $semiaxes: a zero or " *
            "negative one would describe an empty or inverted region, and nothing " *
            "could be flagged against it."))
        return new{D,T}(map(T, Tuple(center)), map(T, Tuple(semiaxes)))
    end
end
function Ellipsoid(center, semiaxes)
    D = length(center)
    length(semiaxes) == D || throw(ArgumentError(
        "an ellipsoid needs one semiaxis per dimension of its center, got a " *
        "$D-dimensional center and $(length(semiaxes)) semiaxes"))
    T = float(promote_type(eltype(Tuple(center)), eltype(Tuple(semiaxes))))
    return Ellipsoid{D,T}(center, semiaxes)
end

convertregion(::Type{T}, ::Nothing) where {T} = nothing
convertregion(::Type{T}, r::Region) where {T} = r
convertregion(::Type{T}, e::Ellipsoid{D}) where {T,D} = Ellipsoid{D,T}(e.center, e.semiaxes)

@inline function inside(e::Ellipsoid{D,T}, x) where {D,T}
    r² = zero(T)
    for d in 1:D
        y = (x[d] - e.center[d]) / e.semiaxes[d]
        r² += y * y
    end
    return r² < one(T)
end

# Whether any stencil point lies in the region. Stencil point `k` (from
# 0) along `d` sits at `origin + ((base + k) − off)·h`, with `base =
# first − G` — the expression `coordinates` evaluates, in the same order,
# so a stencil point is flagged exactly when its `coordinates` are inside.
@inline stencil_hits(::Nothing, origin, h, base, off, ::Val{n}) where {n} = false

@inline stencil_position(origin, h, base, off, d, k) =
    origin[d] + ((base[d] + k) - off[d]) * h

@inline function stencil_hits(region::Region, origin::NTuple{D}, h, base, off,
                              ::Val{n}) where {D,n}
    for J in CartesianIndices(ntuple(_ -> n, Val(D)))
        x = ntuple(d -> stencil_position(origin, h, base, off, d, J[d] - 1), Val(D))
        inside(region, x) && return true
    end
    return false
end

# The separable test. The brute-force sum at the nearest point adds the
# same per-dimension terms in the same order, and floating-point addition
# is monotone, so this agrees with the default method bit for bit.
@inline function stencil_hits(e::Ellipsoid{D,T}, origin::NTuple{D}, h, base, off,
                              ::Val{n}) where {D,T,n}
    r² = zero(T)
    for d in 1:D
        best = zero(T)
        for k in 0:(n - 1)
            y = (stencil_position(origin, h, base, off, d, k) - e.center[d]) /
                e.semiaxes[d]
            best = k == 0 ? y * y : min(best, y * y)
        end
        r² += best
    end
    return r² < one(T)
end

# --- point location ----------------------------------------------------------

# What locating a point needs of the forest, as an `isbits` value a kernel
# can take: the domain in the field set's type, the brick, the face
# kinds, and the finest level present (an `O(nleaves)` scan, done once
# per batch on the host).
struct PointGeometry{D,T}
    extents::NTuple{D,Tuple{T,T}}
    roots::NTuple{D,Int}
    periodic::NTuple{D,Bool}
    reflecting::NTuple{D,Tuple{Bool,Bool}}
    L::Int
end

PointGeometry(::Type{T}, forest::Forest{D}) where {T,D} =
    PointGeometry{D,T}(ntuple(d -> (T(forest.extents[d][1]), T(forest.extents[d][2])), D),
                       forest.roots, forest.periodic, forest.reflecting,
                       maxlevel(forest))

# One coordinate brought into the domain: wrapped along a periodic
# dimension, mirrored once across a reflecting face it lies beyond.
# Returns `(y, folded, inside)`. A point already inside is returned
# untouched, so its interpolation does not depend on the wrap. `NaN` and
# infinities fail the final test.
@inline function fold_coordinate(y::T, lo::T, hi::T, periodic::Bool,
                                 reflecting::Tuple{Bool,Bool}) where {T}
    folded = false
    if periodic
        if !(lo <= y < hi)
            w = hi - lo
            y -= floor((y - lo) / w) * w
            # The subtraction can round onto `hi` itself, or just below `lo`.
            y >= hi && (y = lo)
            y < lo && (y = lo)
        end
    elseif y < lo && reflecting[1]
        y = lo + (lo - y)
        folded = true
    elseif y > hi && reflecting[2]
        y = hi - (y - hi)
        folded = true
    end
    return y, folded, lo <= y <= hi
end

@inline function fold_point(g::PointGeometry{D,T}, x) where {D,T}
    f = ntuple(d -> fold_coordinate(T(x[d]), g.extents[d][1], g.extents[d][2],
                                    g.periodic[d], g.reflecting[d]), Val(D))
    return ntuple(d -> f[d][1], Val(D)), ntuple(d -> f[d][2], Val(D)),
           all(ntuple(d -> f[d][3], Val(D)))
end

# The block index of the leaf containing the in-domain point `x`.
#
# The level-`L` node containing `x`: the root brick position, then the
# node's coordinates bit by bit from the fraction within the root —
# doubling is exact, so this needs no integer conversion of `2^L`, which
# `T` could not hold exactly at deep levels. A point on the upper face of
# the domain lands in the last node (the fraction reaches one and every
# bit is set). Then the last leaf not after that node in curve order,
# which is the covering leaf: the leaves tile the domain, and an
# ancestor sorts immediately before its descendants, which are contiguous.
@inline function locate_leaf(leaves, g::PointGeometry{D,T}, x) where {D,T}
    cell = ntuple(Val(D)) do d
        xlo, xhi = g.extents[d]
        t = (x[d] - xlo) / (xhi - xlo) * T(g.roots[d])
        rp = clamp(floorindex(t), 0, g.roots[d] - 1)
        r = t - T(rp)
        c = 0
        for _ in 1:(g.L)
            r += r
            bit = r >= one(T)
            bit && (r -= one(T))
            c = 2c + bit
        end
        (rp, c)
    end
    root = root_index(g.roots, ntuple(d -> cell[d][1], Val(D)))
    pc = ntuple(d -> UInt64(cell[d][2]) << (MAX_LEVEL - g.L), Val(D))
    # Largest `i` with `leaves[i] ≤ query`; `lo` and `hi` bracket it.
    # (Named apart from the extents above: a closure that assigned to a
    # captured `lo` would box it, and the search would infer as `Any`.)
    lo, hi = 0, length(leaves) + 1
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        k = leaves[mid]
        pk = ntuple(d -> padded_coord(k.coords[d], k.level), Val(D))
        if curve_less(root, pc, g.L, Int(k.root), pk, Int(k.level))
            hi = mid
        else
            lo = mid
        end
    end
    return lo
end

"""
    locate_point(forest, x) -> Union{Int,Nothing}

The index into `forest.leaves` — and so the block index of every
[`FieldSet`](@ref) over `forest` — of the leaf containing the point `x`,
or `nothing` when `x` is outside the domain.

Leaves own half-open boxes, so a point on a face shared by two leaves
belongs to the upper one; a point on the domain's upper face belongs to
the last leaf there. Along a periodic dimension `x` is wrapped into the
domain first, and beyond a reflecting face it is mirrored once — which is
where [`interpolate`](@ref) reads its value. `O(D·maxlevel + log nleaves)`:
one binary search over the sorted leaves.
"""
function locate_point(forest::Forest{D,R}, x) where {D,R}
    length(x) == D || throw(ArgumentError(
        "a point in a $D-dimensional forest has $D coordinates, got $(length(x))"))
    g = PointGeometry(R, forest)
    xf, _, ok = fold_point(g, x)
    ok || return nothing
    return locate_leaf(forest.leaves, g, xf)
end

# --- the stencil -------------------------------------------------------------

# The stencil of an in-domain query in block `b`: its first stored index
# per dimension, and the query's offset `ξ` from it in node units. The
# continuous stored index is the inverse of `coordinates`, `s = G + off +
# (x − origin)/h`; the start is the basis's, clamped into the stored
# array `1 : S − n + 1`. Clamping moves the stencil toward the point, so
# it never extrapolates — near the array's edge it shifts off center, as
# restriction shifts near an interface.
@inline function query_stencil(basis, x::NTuple{D,T}, origin, h, ::Val{G}, ::Val{C},
                               ::Val{S}) where {D,T,G,C,S}
    n = stencilwidth(basis)
    off = pointoffsets(h, C)
    t = ntuple(d -> (x[d] - origin[d]) / h, Val(D))
    first = ntuple(d -> clamp(stencilstart(basis, t[d] + (T(G[d]) + off[d])), 1,
                              S[d] - n + 1), Val(D))
    base = ntuple(d -> first[d] - G[d], Val(D))
    ξ = ntuple(d -> t[d] - (T(base[d]) - off[d]), Val(D))
    return first, base, off, ξ
end

# The whole of one query, written into slot `j` of the outputs.
@inline function interpolate_point!(values, excluded, blocks, xs, leaves, origins,
                                    spacings, work, factors, vars, region, basis,
                                    g::PointGeometry{D,T}, derivs::NTuple{K},
                                    ::Val{M}, ::Val{G}, ::Val{C}, ::Val{S},
                                    j) where {D,T,K,M,G,C,S}
    x, fold, ok = fold_point(g, xs[j])
    if !ok
        blocks[j] = 0
        return nothing
    end
    b = locate_leaf(leaves, g, x)
    blocks[j] = b
    origin = origins[b]
    h = spacings[b]
    n = stencilwidth(basis)
    first, base, off, ξ = query_stencil(basis, x, origin, h, Val(G), Val(C), Val(S))
    excluded[j] = stencil_hits(region, origin, h, base, off, Val(n))
    W = ntuple(d -> basisweights(basis, ξ[d], Val(M)), Val(D))

    # Physical units, and the sign a derivative picks up across each wall
    # the point was mirrored over.
    ih = inv(h)
    scale = ntuple(Val(K)) do k
        s = one(T)
        flips = 0
        for d in 1:D
            for _ in 1:derivs[k][d]
                s *= ih
            end
            fold[d] && (flips += derivs[k][d])
        end
        isodd(flips) ? -s : s
    end
    col, stride = 1, 1
    for d in 1:D
        fold[d] && (col += stride)
        stride *= 3
    end

    # Sum factorization along dimension 1: each row of `n` points is
    # contracted once per derivative order in `x₁`, and only those partial
    # sums meet the other dimensions' weights. Everything that does not
    # depend on the variable is formed once per point: the product of the
    # outer dimensions' weights for every row and requested derivative,
    # and each row's offset into the working array, which is indexed
    # linearly from there. `derivs` arrives as a constant (the kernel takes
    # it as a `Val`): with run-time multi-indices the selections below are
    # dynamic tuple indexing, which cost 21 % serially.
    vNR = rowcount(Val(n), Val(D))
    rows = rows_of(Val(n), Val(D))
    st = strides_of(size(work))
    rowoff = ntuple(jr -> row_offset(Tuple(rows[jr]), st), vNR)
    wouter = ntuple(k -> ntuple(jr -> outer_weight(W, derivs[k], Tuple(rows[jr])), vNR),
                    Val(K))
    sel = ntuple(k -> derivs[k][1] + 1, Val(K))
    corner = 1 + sum(ntuple(d -> (first[d] - 1) * st[d], Val(D))) + (b - 1) * st[D + 2]
    for iv in eachindex(vars)
        v = Int(vars[iv])
        vcorner = corner + (v - 1) * st[D + 1]
        acc = ntuple(_ -> zero(T), Val(K))
        for jr in 1:length(rows)
            r = ntuple(_ -> zero(T), Val(M + 1))
            for i1 in 1:n
                # In bounds by construction: the start is clamped into
                # `1 : S − n + 1` in every dimension (`query_stencil`), and
                # `b` is a leaf index. As in `stencil_sum`, a CI run with
                # `check_bounds = yes` still checks it.
                @inbounds u = work[vcorner + rowoff[jr] + (i1 - 1)]
                r = row_update(r, W[1], i1, u)
            end
            acc = outer_update(acc, r, sel, wouter, jr)
        end
        σ = paritysign(factors, v, col, T)
        for k in 1:K
            values[iv, k, j] = σ * (scale[k] * acc[k])
        end
    end
    return nothing
end

# The two accumulations of the contraction, as functions of their
# accumulators rather than closures over them, for the boxing reason in
# `lagrange_series`. `row_update` adds one point of a row to its partial
# sums, one per derivative order in `x₁`; `outer_update` adds a finished
# row, times its outer weight, to each requested derivative.
@inline row_update(r::NTuple{L,T}, W1, i1, u) where {L,T} =
    ntuple(@inline(m -> r[m] + W1[m][i1] * u), Val(L))

@inline outer_update(acc::NTuple{K,T}, r, sel, wouter, jr) where {K,T} =
    ntuple(@inline(k -> acc[k] + wouter[k][jr] * r[sel[k]]), Val(K))

# The product of the weights of dimensions `2…D` at row `J` (the stencil
# offsets along those dimensions), for derivative multi-index `m`.
@inline function outer_weight(W::NTuple{D}, m, J) where {D}
    w = one(eltype(W[1][1]))
    for d in 2:D
        w *= W[d][m[d] + 1][J[d - 1]]
    end
    return w
end

# The rows of a stencil — its offsets along dimensions `2…D` — and their
# number as a `Val`. Generated so that `n^(D−1)` is a constant: computed
# in the body, the power is not folded, and every tuple sized by it
# would be built at run time.
@inline rows_of(::Val{n}, ::Val{D}) where {n,D} =
    CartesianIndices(ntuple(_ -> n, Val(D - 1)))
@generated rowcount(::Val{n}, ::Val{D}) where {n,D} = :(Val($(n^(D - 1))))

# Column-major strides of an array of size `sz`, as a tuple.
@inline strides_of(sz::Tuple{}) = ()
@inline strides_of(sz::Tuple) = _strides(1, sz)
@inline _strides(acc, sz::Tuple{Any}) = (acc,)
@inline _strides(acc, sz::Tuple) = (acc, _strides(acc * sz[1], Base.tail(sz))...)

# The linear offset of row `J` from the stencil's corner.
@inline row_offset(J::NTuple{E,Int}, st) where {E} =
    sum(ntuple(d -> (J[d] - 1) * st[d + 1], Val(E)); init=0)

# The parity sign of variable `v` in mirror state `col` — the table the
# mirrored transfers multiply by, whose single-fold entries are the
# products of the variable's signs. A forest without reflecting faces has
# no table and no folds.
@inline paritysign(::Nothing, v, col, ::Type{T}) where {T} = one(T)
@inline paritysign(factors, v, col, ::Type{T}) where {T} = factors[v, col]

@kernel function interpolate_kernel!(values, excluded, blocks, @Const(xs),
                                     @Const(leaves), @Const(origins), @Const(spacings),
                                     @Const(work), factors, @Const(vars), region, basis,
                                     g, ::Val{DV}, ::Val{M}, ::Val{G}, ::Val{C},
                                     ::Val{S}) where {DV,M,G,C,S}
    j = @index(Global, Linear)
    interpolate_point!(values, excluded, blocks, xs, leaves, origins, spacings, work,
                       factors, vars, region, basis, g, DV, Val(M), Val(G),
                       Val(C), Val(S), j)
end

# --- the driver --------------------------------------------------------------

# `derivs` validated into an `NTuple{K,NTuple{D,Int}}`. The machinery is
# written for any multi-index — weights to any order, the contraction, the
# `h^|m|` scaling and the mirror signs — and only this check limits it to
# first derivatives until higher ones are tested.
function check_derivs(basis, derivs, ::Val{D}) where {D}
    derivs isa Tuple && !isempty(derivs) || throw(ArgumentError(
        "derivs must be a nonempty tuple of multi-indices, one NTuple{$D,Int} per " *
        "requested quantity — ($(ntuple(_ -> 0, D)),) for the value alone — got " *
        "$(repr(derivs))"))
    for m in derivs
        m isa Tuple && length(m) == D && all(c -> c isa Integer && c >= 0, m) ||
            throw(ArgumentError(
                "each entry of derivs is a multi-index of $D nonnegative integers, the " *
                "derivative order along each dimension, got $(repr(m))"))
        sum(m) <= 1 || throw(ArgumentError(
            "the derivative $(repr(m)) has total order $(sum(m)), and only values and " *
            "first derivatives are implemented so far. The weights, the contraction " *
            "and the scaling are written for any order; what second derivatives still " *
            "need is the tests that would claim them."))
        for c in m
            why = check_derivative(basis, Int(c))
            why === nothing || throw(ArgumentError(why))
        end
    end
    return map(m -> ntuple(d -> Int(m[d]), D), derivs)
end

# The smallest share of a batch worth a task of its own on the CPU.
const MIN_POINTS_PER_TASK = 32

function check_stencil_fits(fs::FieldSet{T,D}, basis) where {T,D}
    n = stencilwidth(basis)
    S = size(fs.work)[1:D]
    for d in 1:D
        n <= S[d] || throw(ArgumentError(
            "$basis reads $n consecutive stored points per dimension, but along " *
            "dimension $d a block of this field set stores only $(S[d]) (N = " *
            "$(fs.forest.N), G = $(fs.G[d]), $(fs.centering[d])): the stencil must lie " *
            "inside one block's array, ghosts included, so that no query reads " *
            "another block's data."))
    end
    return nothing
end

function outside_error(fs::FieldSet, x, j)
    forest = fs.forest
    kinds = ntuple(length(forest.extents)) do d
        forest.periodic[d] ? "periodic" :
        forest.reflecting[d] == (true, true) ? "reflecting" :
        forest.reflecting[d][1] ? "reflecting below" :
        forest.reflecting[d][2] ? "reflecting above" : "outer"
    end
    return ArgumentError(
        "point $j, $(Tuple(x)), is outside the domain $(forest.extents) (faces " *
        "$(kinds)), so no block holds data there. A periodic dimension wraps and a " *
        "reflecting face mirrors a point once; beyond an outer face there is " *
        "nothing to interpolate from, and taking the nearest block would " *
        "extrapolate without saying so.")
end

"""
    interpolate(fs, xs, basis; derivs = (value,), vars = 1:fs.nvars,
                exclude = nothing) -> (; values, excluded)

The variables `vars` of `fs`, and the derivatives `derivs` of them, at every
point of `xs`, by tensor-product interpolation in `basis` — today
[`Lagrange`](@ref)`(n)`.

- `xs` is an array of points — anything indexable as `x[d]`, an
  `NTuple{D}` or an `SVector` — on the field set's backend, of any shape.
- `values[iv, k, j]` is derivative `derivs[k]` of variable `vars[iv]` at
  point `xs[j]` (linear index), so `values[:, k, j]` is contiguous. It has
  the field set's element type and lives on its backend.
- `derivs` is a tuple of multi-indices, the derivative order along each
  dimension: `(0, 0, 0)` is the value, `(1, 0, 0)` is `∂ₓ`, and the value
  with the gradient is `((0,0,0), (1,0,0), (0,1,0), (0,0,1))`. Derivatives
  are in physical units. **Only values and first derivatives are
  implemented so far**; the multi-index form is there so that `(2,0,0)`
  and `(1,1,0)` need no change of interface when they are.
- `exclude` is an optional [`Region`](@ref), and `excluded[j]` says
  whether any point of query `j`'s stencil lies inside it. The value is
  computed either way; what to do with a flagged one is the caller's
  decision. Without a region every entry is `false`.

The **stencil** of a query is the `n^D` stored points of one block the
interpolant reads: the block containing the point ([`locate_point`](@ref)),
and `n` consecutive stored indices per dimension around it, ghosts
included — so **the ghosts must be current**, filled by
[`fill_ghosts!`](@ref) with the boundary hook, as for any stencil.
Near the edge of the stored array the stencil shifts inward rather than
extrapolate: centered whenever `G ≥ n/2` in a cell-centered dimension and
`G ≥ n/2 − 1` in a vertex-like one, off center otherwise, and exact on
the same polynomials either way — which is what makes a field set with
`G = 0` interpolable at all.

A point is wrapped along a periodic dimension and mirrored once across a
reflecting face; a mirrored value takes the variable's parity sign and a
derivative across the wall its own sign too, as the ghosts beyond the
wall do. A point outside the domain after that is an `ArgumentError`,
raised once the whole batch has run.

Every query writes only its own slots, so the result does not depend on
the thread count. The batch is one kernel launch; see "Point
interpolation" in `CODE.md`.
"""
function interpolate(fs::FieldSet{T,D}, xs::AbstractArray, basis::InterpolationBasis;
                     derivs=(ntuple(_ -> 0, D),), vars=1:fs.nvars,
                     exclude=nothing) where {T,D}
    backend = get_backend(fs.work)
    ms = check_derivs(basis, derivs, Val(D))
    values = allocate(backend, T, (length(vars), length(ms), length(xs)))
    excluded = allocate(backend, Bool, length(xs))
    interpolate!(values, excluded, fs, xs, basis; derivs=ms, vars=vars, exclude=exclude)
    return (; values, excluded)
end

"""
    interpolate!(values, excluded, fs, xs, basis; derivs, vars, exclude)

[`interpolate`](@ref) into caller-supplied outputs: `values` of size
`(length(vars), length(derivs), length(xs))` and `excluded` of length
`length(xs)`, both on the field set's backend.
"""
function interpolate!(values::AbstractArray, excluded::AbstractArray, fs::FieldSet{T,D},
                      xs::AbstractArray, basis::InterpolationBasis;
                      derivs=(ntuple(_ -> 0, D),), vars=1:fs.nvars,
                      exclude=nothing) where {T,D}
    ms = check_derivs(basis, derivs, Val(D))
    check_stencil_fits(fs, basis)
    all(v -> 1 <= v <= fs.nvars, vars) || throw(ArgumentError(
        "vars must name variables of the field set, 1:$(fs.nvars), got $vars"))
    npts = length(xs)
    size(values) == (length(vars), length(ms), npts) || throw(ArgumentError(
        "values must have size (length(vars), length(derivs), length(xs)) = " *
        "$((length(vars), length(ms), npts)), got $(size(values))"))
    eltype(values) === T || throw(ArgumentError(
        "values must have the field set's element type $T, got $(eltype(values))"))
    length(excluded) == npts || throw(ArgumentError(
        "excluded must have one entry per point, $npts, got $(length(excluded))"))
    backend = get_backend(fs.work)
    for (name, a) in (("values", values), ("excluded", excluded), ("xs", xs))
        samebackend(get_backend(a), backend) || throw(ArgumentError(
            "$name lives on $(nameof(typeof(get_backend(a)))) but the field set on " *
            "$(nameof(typeof(backend))): the interpolation is one launch where the " *
            "data is, so the points and the outputs must be there too."))
    end
    if xs isa Array && npts > 0
        length(first(xs)) == D || throw(ArgumentError(
            "a point in a $D-dimensional field set has $D coordinates, got " *
            "$(length(first(xs)))"))
    end
    npts == 0 && return values, excluded

    forest = fs.forest
    g = PointGeometry(T, forest)
    # Uploaded per call: this is an analysis-cadence operation, and on the
    # CPU the leaves are not copied at all.
    leaves = todevice(backend, forest.leaves)
    origins = todevice(backend, block_origins(forest, T))
    spacings = todevice(backend, block_spacings(forest, T))
    vs = todevice(backend, Int32[v for v in vars])
    blocks = allocate(backend, Int32, npts)
    region = convertregion(T, exclude)
    M = maximum(m -> maximum(m), ms)
    S = size(fs.work)[1:D]
    # One launch over the points. They are not blocks, so this is not a
    # by-owner launch; on the CPU the workgroup is sized to give every
    # thread a share, since a batch is a few hundred points and one
    # default-sized workgroup would run it on one thread — but not below
    # `MIN_POINTS_PER_TASK` points, since a task per handful of points
    # costs more than it saves (measured on a 64-core node: the 496-point
    # horizon batch took 0.29 ms at 64 threads and 0.14 ms at 16).
    kernel! = backend isa CPU ?
              interpolate_kernel!(backend, max(MIN_POINTS_PER_TASK,
                                               cld(npts, Threads.nthreads()))) :
              interpolate_kernel!(backend)
    kernel!(values, excluded, blocks, xs, leaves, origins, spacings, fs.work,
            fs.factors, vs, region, basis, g, Val(ms), Val(M), Val(fs.G),
            Val(staggers(fs)), Val(S); ndrange=npts)
    synchronize(backend)

    # Reduced where the indices are, so that a device batch copies back one
    # flag rather than an index per point; only a batch that has an
    # outside point pays for finding which.
    if any(iszero, blocks)
        j = findfirst(iszero, tohost(blocks))
        throw(outside_error(fs, tohost(xs)[j], j))
    end
    return values, excluded
end
