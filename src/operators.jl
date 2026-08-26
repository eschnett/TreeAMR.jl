# Inter-grid interpolation operators.
#
# Two families, differing in what a stored number *means*.
#
# PointValue (finite-difference semantics: samples at cell centers).
# Both operators are polynomial interpolation of order `p`, evaluated at
# the target cell center. Cell-centered geometry sets where those
# centers fall relative to the source cells:
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
# orders are correspondingly wider. Symmetry wants an *even* stencil,
# since the target sits a quarter cell off a source center.
#
# Conservative (finite-volume semantics: cell averages).
# Restriction is the exact volume average, which is the 2-cell stencil
# and needs no order. Prolongation reconstructs a polynomial over the
# coarse cell from its neighbours' averages and takes subcell averages
# of it; because the reconstruction reproduces the cell's own average,
# the two subcell averages average back to it exactly. Symmetry here
# wants an *odd* stencil, since the reconstruction is centered on a
# cell rather than between cells.
#
# Both families are linear with per-cell weights, so both run through
# the same tensor-product stencil machinery.

"""
    OperatorFamily

Which semantics the stored numbers carry, and so which pair of
inter-grid operators applies:

- `PointValue` — samples at cell centers (finite differences). Both
  operators are polynomial interpolation at the target center.
- `Conservative` — cell averages (finite volumes). Restriction is the
  exact volume average; prolongation reconstructs over the coarse cell
  and takes subcell averages, so it is locally conservative by
  construction.

See [`Operators`](@ref).
"""
@enum OperatorFamily PointValue Conservative

"""
    Operators(; prolongation, restriction, family=PointValue)

The operator family and orders a [`GhostSchedule`](@ref) is built with.
Both orders are required — there is deliberately **no default**. The
order is the number of source cells per dimension, and the operator is
exact to degree `order - 1`.

Two families ship, differing in what a stored number means:

- [`PointValue`](@ref OperatorFamily) (default) — finite-difference
  semantics, data are samples at cell centers. Both operators are
  polynomial interpolation evaluated at the target center, and both
  orders must be **even** and at least 2 so the stencil is symmetric
  about the target.
- [`Conservative`](@ref OperatorFamily) — finite-volume semantics, data
  are cell averages. Restriction is the exact volume average, so `restriction`
  must be `2` (its stencil is a cell's two children per dimension, and
  it is exact for *any* field, not merely to some order).
  `prolongation` reconstructs a polynomial over the coarse cell and
  takes subcell averages of it, and must be **odd** and at least 1,
  symmetric about the reconstructed cell. Order 1 is piecewise constant,
  order 3 the familiar `±1/8` slope.

Only the conservative family makes the regridding transfer exactly
mass-conserving for arbitrary data — see [`regrid!`](@ref) and
[`total_mass`](@ref). The point-value family conserves only fields it
reproduces exactly.

Operators are configured per field set, not per variable — see
`CODE.md`; per-variable selection is deferred to M8.

!!! warning "Choose the order against your discretization"
    For the point-value family, interpolation order must exceed the
    application's differencing order **by two**, for *both* operators, or
    the coarse-fine interface caps global convergence. This is why there
    is no default: the right order follows from the application's
    discretization, which the mesh cannot know. (How the rule carries
    over to the conservative family is to be measured in M8.)

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

    Order 2 is the cheapest correct *interpolation*, not the right choice
    for a second-order-in-space application — which is what a default of
    2 used to hide.

The two operators behave differently at a coarse-fine interface.

**Prolongation stays symmetric.** Its window may reach into the coarse
source block's own ghost layers — `p ÷ 2` of them for the point-value
family, `(p - 1) ÷ 2` for the conservative one, at the fine ghost layer
nearest the interface — which is what the level-ordered sweep in
[`GhostSchedule`](@ref) exists to guarantee are already filled.

**Restriction shifts.** A symmetric window at the coarse ghost layer
nearest the interface would have to read fine cells across it, and those
are themselves prolongated coarse data — a circularity. So the window is
shifted inward instead, by `p ÷ 2 - 1` fine cells at that layer. This
costs no accuracy: Lagrange interpolation through any `p` distinct nodes
is exact for degree `< p`, and the shifted window still brackets the
target, so it stays interpolation and never becomes extrapolation (which
the schedule asserts). Conservative restriction never shifts: its window
is exactly a cell's own children.

The orders are therefore constrained by the block geometry (see
[`check_operators`](@ref)):

- `G ≥ prolongation ÷ 2` (point-value) or `(prolongation - 1) ÷ 2`
  (conservative), so a fine block's ghost stencil fits within its coarse
  neighbor's interior plus that neighbor's own ghosts;
- `N ≥ 2G + restriction ÷ 2 - 1` and `N ≥ restriction`, so a coarse
  block's ghost layers can be restricted from fine *interior* cells
  alone.
"""
struct Operators
    family::OperatorFamily
    prolongation::Int
    restriction::Int

    # Sentinel defaults rather than required keywords, so that omitting
    # one reports *why* there is no default instead of a bare
    # UndefKeywordError.
    function Operators(; prolongation::Union{Integer,Nothing}=nothing,
                       restriction::Union{Integer,Nothing}=nothing,
                       family::OperatorFamily=PointValue)
        for (name, p) in (("prolongation", prolongation), ("restriction", restriction))
            p === nothing && throw(ArgumentError(
                "Operators has no default order: pass $name explicitly. The order " *
                "must exceed the application's differencing order by two. Order 2 " *
                "against a second-derivative stencil leaves an O(1) error at " *
                "coarse-fine interfaces, capping global convergence at first order."))
        end

        if family === PointValue
            for (name, p) in (("prolongation", prolongation), ("restriction", restriction))
                p >= 2 || throw(ArgumentError("$name order must be at least 2, got $p"))
                iseven(p) || throw(ArgumentError(
                    "$name order must be even for the point-value family (the target " *
                    "sits a quarter cell off a source center, so a symmetric stencil " *
                    "has even width), got $p"))
            end
        else
            prolongation >= 1 || throw(ArgumentError(
                "conservative prolongation order must be at least 1, got $prolongation"))
            isodd(prolongation) || throw(ArgumentError(
                "conservative prolongation order must be odd (the reconstruction is " *
                "centered on the coarse cell, so a symmetric stencil has odd width), " *
                "got $prolongation"))
            restriction == 2 || throw(ArgumentError(
                "conservative restriction is the exact volume average, whose stencil is " *
                "a cell's two children per dimension, so restriction must be 2, got " *
                "$restriction. It is exact for any field, so there is no higher order " *
                "to ask for."))
        end
        return new(family, Int(prolongation), Int(restriction))
    end
end

Base.show(io::IO, ops::Operators) =
    print(io, "Operators(prolongation=", ops.prolongation,
          ", restriction=", ops.restriction, ", family=", ops.family, ")")

"""
    ghost_layers_read(ops::Operators)

How many of the coarse source block's own ghost layers a prolongation
stencil reaches into — the quantity that ties the operator order to `G`.
"""
ghost_layers_read(ops::Operators) =
    ops.family === Conservative ? (ops.prolongation - 1) ÷ 2 : ops.prolongation ÷ 2

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
    needed = ghost_layers_read(ops)
    G >= needed || throw(ArgumentError(
        "$(ops.family) prolongation of order $pp needs G >= $needed ghost layers so its " *
        "stencil fits within the coarse neighbor's interior plus ghosts, but G=$G"))

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
