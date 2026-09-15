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

Operators are configured per field set, not per variable — and from M8
on that *is* per-variable selection: a field set is the unit of
centering, ghost width and operators alike, so conservative operators for
a density and point-value ones for a velocity are two field sets over one
forest, each with its own schedule.

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

The orders are therefore constrained by the block geometry, per
dimension against that dimension's ghost width *and centering* (see
[`check_operators`](@ref)). In a cell-centered dimension:

- `G ≥ prolongation ÷ 2` (point-value) or `(prolongation - 1) ÷ 2`
  (conservative), so a fine block's ghost stencil fits within its coarse
  neighbor's interior plus that neighbor's own ghosts;
- `N ≥ 2G + restriction ÷ 2 - 1` and `N ≥ restriction`, so a coarse
  block's ghost layers can be restricted from fine *interior* cells
  alone.

In a vertex-like dimension neither operator is the same object:
restriction is injection (no order, no shifting) and prolongation
evaluates at integer or half-integer coarse coordinates, so the whole
requirement is `G ≥ prolongation ÷ 2 - 1` — and the conservative family
is refused there outright.
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
    check_operators(fs::FieldSet, ops::Operators)

Verify that the forest's `N` and the *field set's* per-dimension `G` and
centering support the requested interpolation orders, throwing an
`ArgumentError` naming the violated invariant otherwise. Called when a
[`GhostSchedule`](@ref) is built.

Every constraint is per dimension, against that dimension's `G[d]`
(amended in M8, with the ghost width) and that dimension's centering
(M8a step 2): a family is a rule giving one-dimensional operators per
centering, so what a dimension needs depends on where its values sit.

| `d` | family | restriction | prolongation | needs, in `d` |
|---|---|---|---|---|
| cell | `PointValue` | shifted Lagrange | quarter offsets | `G ≥ p/2`, `N ≥ 2G+p/2-1` |
| cell | `Conservative` | 2-cell average | subcell averages | `G ≥ (p-1)/2` |
| vertex | `PointValue` | injection | integer / half-integer | `G ≥ p/2 - 1` |
| vertex | `Conservative` | *refused* | *refused* | — |

A cell-centered dimension additionally needs `N ≥ restriction`, so that
the restriction window fits inside a fine block's interior; along a
stagger there is no window.

A vertex-like dimension needs less, in both directions: restriction is
injection, which carries no order and reads a point the fine block owns,
and prolongation reaches only `p/2 - 1` planes past the shared plane
instead of `p/2` past the interface. In particular `G_d = 0` is legal
there — a block still has its shared plane to exchange — and is exactly
the second-order constrained-transport layout for an evolved face field.

There is no blanket requirement of `G_d ≥ 1` anywhere: the table implies
it wherever interpolation reads a neighbor, which is every order except
conservative prolongation of order 1. Piecewise-constant prolongation
reads only the coarse cell containing the fine one and the exact average
reads only a cell's own children, so a cell-centered dimension with
`G_d = 0` is a legal layout for `Conservative` `(1, 2)` operators — a
ghost-free auxiliary field set carried across regrids, say. Its exchange
in that dimension is simply empty.
"""
check_operators(fs::FieldSet, ops::Operators) =
    check_operators(fs.forest.N, fs.G, staggers(fs), ops)

function check_operators(N::Integer, G::NTuple{D,Int}, c::NTuple{D,Int},
                         ops::Operators) where {D}
    pp = ops.prolongation
    pr = ops.restriction
    needed = ghost_layers_read(ops)

    for d in 1:D
        g = G[d]
        if c[d] == 1
            # What a face- or edge-centered quantity stores is an average
            # along its cell-like dimensions and a *point value* along
            # its vertex-like ones, so along a vertex dimension the
            # conservative family has nothing to conserve and would
            # merely interpolate — at an even order its odd `p` does not
            # name. Refused rather than guessed at; see "Operators" in
            # CODE.md.
            ops.family === Conservative && throw(ArgumentError(
                "the Conservative family is not defined along a vertex-like " *
                "dimension (dimension $d of centering with c=$c). What a staggered " *
                "quantity stores is a point value there, not an average, so there " *
                "is nothing to conserve and the family's odd orders do not name " *
                "the interpolation that would be wanted. Use PointValue for a " *
                "staggered field set; fluxes and EMFs are never ghost-filled at " *
                "all."))
            g >= pp ÷ 2 - 1 || throw(ArgumentError(
                "prolongation of order $pp into a vertex-like dimension needs " *
                "G >= $(pp ÷ 2 - 1), because its stencil reaches that many planes " *
                "past the source's shared boundary plane, but G=$G (dimension $d)"))
        else
            # No blanket `g >= 1` here: the order constraints below imply
            # it wherever a stencil reads a neighbor, and conservative
            # order 1 — the one case they do not — reads nothing outside
            # the containing cell, so a ghost-free dimension is legal for
            # it and its exchange there is simply empty.
            g >= needed || throw(ArgumentError(
                "$(ops.family) prolongation of order $pp needs G >= $needed ghost " *
                "layers so its stencil fits within the coarse neighbor's interior " *
                "plus ghosts, but G=$G (dimension $d)"))
            N >= 2g + pr ÷ 2 - 1 || throw(ArgumentError(
                "restriction of order $pr into G=$g ghost layers (dimension $d) " *
                "needs N >= $(2g + pr ÷ 2 - 1) fine interior cells, but N=$N"))
            N >= pr || throw(ArgumentError(
                "restriction of order $pr needs N >= $pr so the stencil fits " *
                "within a fine block's interior (dimension $d), but N=$N"))
        end
    end
    return nothing
end

# `BigInt` rather than `Int`: the *running* products below outgrow a
# 64-bit numerator at order 16 (measured: 5.79e20 there, 1.78e17 at order
# 14), and `Rational` arithmetic is checked, so that would be a hard
# error rather than a wrong answer. A bignum removes the ceiling
# altogether instead of moving it, which costs nothing that matters: this
# runs once per stencil entry when a schedule is built, never in the
# per-evaluation path. It also buys correct rounding on the way out —
# `T(::Rational{BigInt})` divides through `BigFloat`, so the quotient is
# rounded once from the exact value rather than from a numerator and
# denominator that were each rounded to `T` first.
const WeightRational = Rational{BigInt}

"""
    lagrange_weights(nodes, x)

Weights `w` with `sum(w[i] * u(nodes[i])) == u(x)` for every polynomial
`u` of degree less than `length(nodes)`.

Exactness holds for *any* distinct nodes, which is what lets the
restriction stencil be shifted away from the coarse-fine interface (to
stay inside the fine block's interior) without losing order.

Nodes and target are **exact rationals**, and so is the result: every
position a stencil is ever evaluated at is an integer or a quarter
integer, so the weights are exact rational numbers and the only rounding
in the whole construction is the single conversion into a
[`Stencil1D`](@ref TreeAMR.Stencil1D)'s element type. Doing this in
floating point instead would fix an accuracy ceiling at whatever type the
weights were built in — and would need hardware fp64 to reach it.
"""
function lagrange_weights(nodes::AbstractVector{<:Rational}, x::Rational)
    n = length(nodes)
    ns = WeightRational.(nodes)
    xq = WeightRational(x)
    w = Vector{WeightRational}(undef, n)
    for i in 1:n
        num = one(WeightRational)
        den = one(WeightRational)
        for j in 1:n
            j == i && continue
            num *= (xq - ns[j])
            den *= (ns[i] - ns[j])
        end
        w[i] = num / den
    end
    return w
end
