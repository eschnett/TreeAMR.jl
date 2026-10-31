# --- The reference element of the element centering (M16) -----------------
#
# An element-centered field set (`elementcentered(D)`) holds, per
# dimension, `N/(P+1)` elements of `P+1` Gauss–Lobatto–Legendre nodes
# each, both face nodes stored. Everything the mesh needs to know about
# such an element is a handful of one-dimensional tables: where the nodes
# sit, their quadrature weights, and the three operators that move an
# element's polynomial between a coarse element and the two fine ones
# covering it. They are built here, once per order `P`, and handed to
# the rest of the package rounded to the working type.
#
# The nodes are irrational from `P = 3` on, so the exact `Rational`
# construction the uniform stencils use (`lagrange_weights`) does not
# apply. The tables are built in 512-bit `BigFloat` instead and rounded
# once, which keeps the rule of "Precision" in CODE.md: the only rounding
# in the construction is the conversion into the working type. The
# nodes are built with the algorithm TreeWaveOLDG's `gll_nodes_weights`
# uses, at the same precision, so that the two packages place every node
# at bit-identical positions.

# The one lock over the package's memoised weight tables: the Lagrange
# weights of `operators.jl` and the element tables below. Defined here,
# since this file is included first.
const LAGRANGE_LOCK = ReentrantLock()

# The precision every element table is built in, as TreeWaveOLDG's.
const ELEMENT_PRECISION = 512

# Legendre polynomials `L₀ … L_n` at `x`, by the three-term recurrence.
function legendre_all(n::Integer, x)
    L = Vector{typeof(x)}(undef, n + 1)
    L[1] = one(x)
    n >= 1 && (L[2] = x)
    for k in 2:n
        L[k + 1] = ((2k - 1) * x * L[k] - (k - 1) * L[k - 1]) / k
    end
    return L
end

# The `P + 1` Gauss–Lobatto–Legendre nodes on `[-1, 1]`, ascending, and
# their weights, in `BigFloat` at `ELEMENT_PRECISION`. Newton's iteration
# on `(1 - x²) L_P'(x)` written through the recurrence, from the
# Chebyshev–Gauss–Lobatto points (Hesthaven and Warburton's `JacobiGL`
# for `α = β = 0`); the same algorithm, line for line, as TreeWaveOLDG's
# `gll_nodes_weights`. The endpoints are fixed points of the iteration and
# come out as `±1` exactly; the nodes are then symmetrized, so `ξᵢ ==
# -ξ_{P-i}` holds bit for bit after rounding to any type, and the centre
# node of an even `P` is `0` exactly.
function gll_nodes_weights(P::Integer)
    return setprecision(BigFloat, ELEMENT_PRECISION) do
        x = [-cospi(big(i) / P) for i in 0:P]
        tol = ldexp(one(BigFloat), 8 - ELEMENT_PRECISION)
        for _ in 1:200
            delta = zero(x)
            for i in eachindex(x)
                L = legendre_all(P, x[i])
                delta[i] = (x[i] * L[P + 1] - L[P]) / ((P + 1) * L[P + 1])
            end
            x .-= delta
            maximum(abs, delta) <= tol && break
        end
        x = [(x[i + 1] - x[P + 1 - i]) / 2 for i in 0:P]
        x[1], x[P + 1] = -one(BigFloat), one(BigFloat)
        w = [2 / (P * (P + 1) * legendre_all(P, xi)[P + 1]^2) for xi in x]
        return x, w
    end
end

# The `n` Gauss–Legendre nodes and weights on `[-1, 1]`, in `BigFloat`,
# by Newton's iteration on `L_n` from the usual cosine guesses. Only the
# coarsening below uses them, to integrate a product of degree `2P`
# exactly, which `n = P + 1` points already do.
function gauss_nodes_weights(n::Integer)
    return setprecision(BigFloat, ELEMENT_PRECISION) do
        x = [-cospi((big(i) - big(1) / 4) / (n + big(1) / 2)) for i in 1:n]
        tol = ldexp(one(BigFloat), 8 - ELEMENT_PRECISION)
        dL(L, xi) = n * (xi * L[n + 1] - L[n]) / (xi^2 - 1)
        for _ in 1:200
            delta = zero(x)
            for i in eachindex(x)
                L = legendre_all(n, x[i])
                delta[i] = L[n + 1] / dL(L, x[i])
            end
            x .-= delta
            maximum(abs, delta) <= tol && break
        end
        w = [2 / ((1 - xi^2) * dL(legendre_all(n, xi), xi)^2) for xi in x]
        return x, w
    end
end

# The Lagrange basis polynomial `ℓ_k` of the nodes `ξ` at `x`, as a
# product. At a node it is a product of exact factors, `0` and `1` by
# construction, which is what keeps the exact entries of the tables
# exact.
function lagrange_basis(ξ::AbstractVector, k::Int, x)
    l = one(x)
    for m in eachindex(ξ)
        m == k && continue
        l *= (x - ξ[m]) / (ξ[k] - ξ[m])
    end
    return l
end

"""
    ElementTables{T}

The one-dimensional tables of the reference element of order `P`, for
the element centering ([`elementcentered`](@ref)), in type `T`. Index
`i + 1` is node `i = 0 … P` of an element; for the two-element operators
columns `1 : P+1` are the nodes of the low fine element (the half
`[-1, 0]` of the coarse one) and `P+2 : 2P+2` those of the high one.

- `nodes`: the Gauss–Lobatto–Legendre nodes `ξᵢ ∈ [-1, 1]`, ascending,
  `±1` at the faces and `0` at the centre of an even `P`, exactly.
- `weights`: their quadrature weights `ŵᵢ` on `[-1, 1]`, summing to 2.
- `offsets`: the node's position in its element in units of the spacing
  `h`, `(P+1)(1 + ξᵢ)/2`, so `offsets[1] == 0` and `offsets[P+1] == P+1`
  exactly, and two face nodes at one position get the same bits.
- `quadrature`: the node's quadrature weight in units of `h`,
  `(P+1)ŵᵢ/2`: an element is `P+1` spacings wide.
- `interpolation`: the two halves of `I`, `(P+1) × (P+1)` each, row `j`
  a fine node and column `k` a coarse one, `I[j, k] = ℓ_k(ξ′_j)` with
  `ξ′_j = (ξ_j ∓ 1)/2` the fine node in the coarse element's coordinate:
  the coarse element's degree-`P` polynomial at the fine nodes. Exact to
  degree `P`.
- `restriction`: the lumped compatible restriction `R = M_c⁻¹ Iᵀ M_f`,
  `(P+1) × 2(P+1)`, `R[k, j] = ŵ_j ℓ_k(ξ′_j)/(2ŵ_k)` over the nodes of
  both fine elements. It is what keeps a summation-by-parts scheme's
  energy an exact invariant across a coarse-fine face, `M_c R = Iᵀ M_f`
  with the diagonal GLL mass matrices; it conserves `Σ ŵ u` and is exact
  to degree `P - 1` but not `P`.
- `coarsening`: the exact `L²` projection of the two fine elements'
  degree-`P` polynomials onto the coarse element's, evaluated at its
  nodes, `(P+1) × 2(P+1)`. Exact to degree `P`, conserving `Σ ŵ u`, and
  a left inverse of `I`.

Built once per `P` in 512-bit `BigFloat`, as TreeWaveOLDG builds its
element, and rounded once into `T`; see [`element_tables`](@ref
TreeAMR.element_tables).
"""
struct ElementTables{T}
    P::Int
    nodes::Vector{T}
    weights::Vector{T}
    offsets::Vector{T}
    quadrature::Vector{T}
    interpolation::NTuple{2,Matrix{T}}
    restriction::Matrix{T}
    coarsening::Matrix{T}
end

# Every table in `BigFloat`, from the nodes and weights.
function build_element_tables(P::Int)
    ξ, ŵ = gll_nodes_weights(P)
    P1 = P + 1
    return setprecision(BigFloat, ELEMENT_PRECISION) do
        offsets = [P1 * (1 + x) / 2 for x in ξ]
        offsets[1], offsets[P1] = 0, P1
        quadrature = [P1 * w / 2 for w in ŵ]
        # The fine nodes in the coarse element's coordinate, low half then
        # high half. `(ξ ∓ 1)/2` is exact for the face nodes, so the face
        # and (even `P`) centre rows of `I` are exact unit rows.
        fine = [[(x - 1) / 2 for x in ξ], [(x + 1) / 2 for x in ξ]]
        interpolation = ntuple(s -> [lagrange_basis(ξ, k, fine[s][j])
                                     for j in 1:P1, k in 1:P1], 2)
        restriction = zeros(BigFloat, P1, 2P1)
        for s in 1:2, j in 1:P1, k in 1:P1
            restriction[k, (s - 1) * P1 + j] = ŵ[j] * interpolation[s][j, k] / (2 * ŵ[k])
        end
        # The L² projection through Legendre moments: the coarse
        # polynomial is `Σ_m a_m L_m` with `a_m = (2m+1)/2 ∫ f L_m`, and on
        # half `s` the integral is `½∫ f_s(η) L_m((η ∓ 1)/2) dη` over the
        # fine element's own coordinate, a product of degree `2P` that the
        # `P+1`-point Gauss rule integrates exactly.
        gx, gw = gauss_nodes_weights(P1)
        Lc = [legendre_all(P, x) for x in ξ]
        coarsening = zeros(BigFloat, P1, 2P1)
        for s in 1:2, j in 1:P1
            σ = s == 1 ? -1 : 1
            moments = zeros(BigFloat, P1)
            for q in eachindex(gx)
                lj = lagrange_basis(ξ, j, gx[q])
                Lm = legendre_all(P, (gx[q] + σ) / 2)
                for m in 0:P
                    moments[m + 1] += gw[q] * lj * Lm[m + 1] / 2
                end
            end
            for k in 1:P1
                coarsening[k, (s - 1) * P1 + j] =
                    sum((2m + 1) * moments[m + 1] * Lc[k][m + 1] for m in 0:P) / 2
            end
        end
        return ElementTables{BigFloat}(P, ξ, ŵ, offsets, quadrature, interpolation,
                                       restriction, coarsening)
    end
end

# Round every table once from `BigFloat` into `T`.
function round_element_tables(::Type{T}, t::ElementTables{BigFloat}) where {T}
    r(a) = T.(a)
    return ElementTables{T}(t.P, r(t.nodes), r(t.weights), r(t.offsets),
                            r(t.quadrature), map(r, t.interpolation), r(t.restriction),
                            r(t.coarsening))
end

# Memoised per `(P, T)`. The `BigFloat` tables are built once per `P` and
# every other type rounds from them, so each entry of every type is the
# correctly rounded value of one and the same `BigFloat`. The cached
# tables are shared, so nothing may mutate one.
const ELEMENT_CACHE = Dict{Tuple{Int,Type},ElementTables}()

"""
    element_tables(T, P)

The [`ElementTables`](@ref TreeAMR.ElementTables) of the reference
element of order `P` in type `T`, memoised: built once per `P` in 512-bit
`BigFloat`, with the algorithm and precision of TreeWaveOLDG's
`gll_nodes_weights` so that both packages place the nodes at the same
bits, and rounded once into `T`. `T = BigFloat` returns the unrounded
tables. See "Precision" in `CODE.md` for why weights are built exactly
or nearly so, and rounded once.

`P ≥ 1`: an element needs both of its face nodes.
"""
function element_tables(::Type{T}, P::Integer) where {T}
    P >= 1 || throw(ArgumentError(
        "the element order P must be at least 1, got $P: an element stores both of " *
        "its face nodes, so it holds at least two"))
    p = Int(P)
    return lock(LAGRANGE_LOCK) do
        get!(ELEMENT_CACHE, (p, T)) do
            T === BigFloat ? build_element_tables(p) :
            round_element_tables(T, element_tables(BigFloat, p))
        end
    end::ElementTables{T}
end
