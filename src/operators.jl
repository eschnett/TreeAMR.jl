# Inter-grid interpolation operators.
#
# Prolongation and restriction are symmetric: both are polynomial
# interpolation of a configurable accuracy order `p`, evaluated at the
# target cell centers. Cell-centered geometry sets where those centers
# fall relative to the source cells:
#
#   - Prolongation (coarse -> fine): fine cell `f` has its center at
#     coarse coordinate `f/2 - 1/4`, a quarter cell either side of the
#     coarse center it sits in.
#   - Restriction (fine -> coarse): a coarse cell center falls exactly
#     on the interface between its two fine children, at `2c + 1/2` in
#     fine coordinates.
#
# At order 2 the restriction weights come out to 1/2, 1/2 — the plain
# `2^D` average — and the prolongation weights to 1/4, 3/4. Higher
# orders are correspondingly wider.

"""
    Operators(; prolongation=2, restriction=2)

The interpolation orders a [`GhostSchedule`](@ref) is built with.
Both must be even and at least 2; the order is the number of source
cells per dimension, and the operator is exact for polynomials of
degree up to `order - 1`.

Operators are configured per field set, not per variable — see
`CODE.md`; per-variable selection is deferred to M8.

!!! warning "Choose the order against your discretization, not the default"
    Interpolation order must exceed the application's differencing order
    **by two**, for *both* operators, or the coarse-fine interface caps
    global convergence.

    A ghost filled by an order-`p` operator carries an `O(hᵖ)` error. A
    second-derivative stencil divides it by `h²`, so the truncation
    error along the interface is `O(h^{p-2})` — with `p = 2` that is
    `O(1)`, and it does not shrink under refinement at all.

    Measured with the M3 wave equation (2nd-order Laplacian, two-level
    mesh), global L2 convergence comes out as

    | prolongation | restriction | rate |
    |---|---|---|
    | 2 | 2 | 1.0 |
    | 4 | 2 | 0.9 |
    | 2 | 4 | 1.0 |
    | 4 | 4 | 2.0 |

    Raising one operator alone does not help: each side of the interface
    gets its ghosts from a different operator, so whichever stays at
    order 2 keeps its side first order. The same mesh with no refinement
    converges at 2.0 with order-2 operators, so this is the interface
    and not the scheme.

    The default of 2 is the cheapest correct *interpolation*, not the
    right choice for a second-order-in-space application.

The two operators behave differently at a coarse-fine interface.

**Prolongation stays symmetric.** Its window may reach into the coarse
source block's own ghost layers — exactly `p ÷ 2` of them, at the fine
ghost layer nearest the interface — which is what the level-ordered
sweep in [`GhostSchedule`](@ref) exists to guarantee are already filled.

**Restriction shifts.** A symmetric window at the coarse ghost layer
nearest the interface would have to read fine cells across it, and those
are themselves prolongated coarse data — a circularity. So the window is
shifted inward instead, by `p ÷ 2 - 1` fine cells at that layer. This
costs no accuracy: Lagrange interpolation through any `p` distinct nodes
is exact for degree `< p`, and the shifted window still brackets the
target, so it stays interpolation and never becomes extrapolation (which
the schedule asserts).

The orders are therefore constrained by the block geometry (see
[`check_operators`](@ref)):

- `G ≥ prolongation ÷ 2`, so a fine block's ghost stencil fits within
  its coarse neighbor's interior plus that neighbor's own ghosts;
- `N ≥ 2G + restriction ÷ 2 - 1` and `N ≥ restriction`, so a coarse
  block's ghost layers can be restricted from fine *interior* cells
  alone.
"""
struct Operators
    prolongation::Int
    restriction::Int

    function Operators(; prolongation::Integer=2, restriction::Integer=2)
        for (name, p) in (("prolongation", prolongation), ("restriction", restriction))
            p >= 2 || throw(ArgumentError("$name order must be at least 2, got $p"))
            iseven(p) ||
                throw(ArgumentError("$name order must be even (symmetric cell-centered " *
                                    "stencils), got $p"))
        end
        return new(Int(prolongation), Int(restriction))
    end
end

Base.show(io::IO, ops::Operators) =
    print(io, "Operators(prolongation=", ops.prolongation, ", restriction=", ops.restriction, ")")

"""
    check_operators(forest, ops::Operators)

Verify that `forest`'s `N` and `G` support the requested interpolation
orders, throwing an `ArgumentError` naming the violated invariant
otherwise. Called when a [`GhostSchedule`](@ref) is built.
"""
function check_operators(forest::Forest, ops::Operators)
    N, G = forest.N, forest.G
    G >= 1 || throw(ArgumentError("ghost filling needs G >= 1, got G=$G"))

    pp = ops.prolongation
    G >= pp ÷ 2 || throw(ArgumentError(
        "prolongation of order $pp needs G >= $(pp ÷ 2) ghost layers so its stencil fits " *
        "within the coarse neighbor's interior plus ghosts, but G=$G"))

    pr = ops.restriction
    N >= pr || throw(ArgumentError(
        "restriction of order $pr needs N >= $pr so the stencil fits within a fine " *
        "block's interior, but N=$N"))
    N >= 2G + pr ÷ 2 - 1 || throw(ArgumentError(
        "restriction of order $pr into G=$G ghost layers needs N >= $(2G + pr ÷ 2 - 1) " *
        "fine interior cells, but N=$N"))
    return nothing
end

"""
    lagrange_weights(nodes, x)

Weights `w` with `sum(w[i] * u(nodes[i])) == u(x)` for every polynomial
`u` of degree less than `length(nodes)`.

Exactness holds for *any* distinct nodes, which is what lets the
restriction stencil be shifted away from the coarse-fine interface (to
stay inside the fine block's interior) without losing order.
"""
function lagrange_weights(nodes::AbstractVector{<:Real}, x::Real)
    n = length(nodes)
    w = Vector{Float64}(undef, n)
    for i in 1:n
        num = 1.0
        den = 1.0
        for j in 1:n
            j == i && continue
            num *= (x - nodes[j])
            den *= (nodes[i] - nodes[j])
        end
        w[i] = num / den
    end
    return w
end
