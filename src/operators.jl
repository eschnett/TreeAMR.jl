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
- [`ElementLocal`](@ref) — the polynomials of Gauss–Lobatto elements
  (M16), the one family of an element-centered set
  ([`elementcentered`](@ref)).

See [`Operators`](@ref).
"""
@enum OperatorFamily PointValue Conservative ElementLocal

"""
    ElementLocal

The [`OperatorFamily`](@ref) of an element-centered field set
([`elementcentered`](@ref), M16), whose values are the nodal values of a
degree-`P` polynomial per element. It is the only family such a set
takes, and no other set takes it. It has no orders to choose: every
operator follows from the element and its order `P`, which is the field
set's, so `Operators(family = ElementLocal)` takes no `prolongation`,
`restriction`, `bounded` or `vertex_prolongation`.

The ghost fill of an element set writes **faces only**, one plane deep,
each ghost the value the neighbour holds at the shared face:

- normal to the face, for every kind of transfer, the neighbour's
  face-adjacent owned plane, which sits at the same position;
- tangentially across a coarse-fine face, a fine ghost is the coarse
  element's polynomial at the fine nodes (`I`, exact to degree `P`), and
  a coarse ghost the compatible restriction `R = M_c⁻¹ Iᵀ M_f` of the
  two fine elements covering it, exact to degree `P - 1` and the adjoint
  of `I` in the Gauss–Lobatto inner products, which is what keeps a
  summation-by-parts scheme's energy an exact invariant across the face.

Every transfer reads owned points only, so the fill is one stage. A
[`regrid!`](@ref) refines by `I` and coarsens by the exact `L²`
projection, exact to degree `P`, conserving [`total_mass`](@ref), and a
left inverse of the refinement. See [`ElementTables`](@ref
TreeAMR.ElementTables) for the tables, and "Element-local DG stencils"
in `CODE.md`.
"""
ElementLocal

"""
    Operators(; prolongation, restriction, family=PointValue, bounded=false,
              vertex_prolongation=nothing)
    Operators(; family=ElementLocal)

The operator family and orders a [`GhostSchedule`](@ref) is built with.
Both orders are required — there is deliberately **no default**. The
order is the number of source cells per dimension, and the operator is
exact to degree `order - 1`. The exception is the element family
([`ElementLocal`](@ref), M16), the one family of an element-centered
set, which takes no orders at all: its operators follow from the
element order `P` the field set carries.

Two families of equally spaced data ship, differing in what a stored
number means:

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
    discretization, which the mesh cannot know. The `+2` is the
    second-derivative case; a flux divergence takes one derivative, so
    for the conservative family the order must exceed the scheme's by
    **one** — measured below.

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

    Along a **vertex-like** dimension only the prolongation order
    enters, because restriction there is injection — a coincident point
    copied, exact for any data, with no order to raise. The same study
    on a vertex-centered field set measures (M8a)

    | prolongation | restriction | rate, `D = 1` | rate, `D = 2` |
    |---|---|---|---|
    | 2 | 2 | 0.99 | 1.01 |
    | 2 | 4 | 0.99 | 1.01 |
    | 4 | 2 | 1.99 | 1.99 |
    | 4 | 4 | 1.99 | 1.99 |

    at `G = 1`, where cell centering needs 2. The rows come in pairs
    because the two runs are the *same computation*, bit for bit.

    For the **conservative** family under a second-order finite-volume
    scheme the measured rates are (M8b, Burgers' equation, maximum norm;
    the restriction order does not appear because conservative
    restriction is exact)

    | prolongation | rate, `D = 1` | rate, `D = 2` |
    |---|---|---|
    | 1 | 1.00 | 0.84 |
    | 3 | 1.97 | 1.81 |
    | 5 | 1.97 | 1.77 |

    against 1.93 and 1.76 for the same mesh unrefined, so order 3 is the
    first that does not degrade the scheme and order 5 buys nothing
    more. The *maximum* norm: a flux divergence leaves the interface
    defect where it is instead of radiating it, so an integral norm
    converges at the scheme's own rate even at order 1. See `CODE.md`.

    Order 2 is the cheapest correct *interpolation*, not the right choice
    for a second-order-in-space application — which is what a default of
    2 used to hide.

**Bounded operators.** `bounded = true` makes the inter-grid operators
create no new extrema. With the conservative family, allowed at
`prolongation = 3` or `1`, a ghost filled across a coarse-fine face, or a
child made by [`regrid!`](@ref), lies between the smallest and the
largest of its parent cell and the parent's `2D` face neighbours. With
the point-value family, at any order, see "Bounded point values" below.
That is a property of
the interpolation, as conservation is, and it is what keeps a fine ghost
from leaving the range of the data around it where the unlimited order-3
slope overshoots — across a shock, or a stellar surface where the
density drops by orders of magnitude. Whether a state is *physically*
admissible is the application's question, answered after the fill. At
order 1 a child is a copy of its parent, so the bound holds already and
`bounded = true` changes nothing; it is accepted so that an application
can switch between the two orders without touching it. At order 3, per
variable, with `ū` the parent's average, `a = ū₊ − ū` and `b = ū − ū₋`
its one-sided differences along dimension `d`,

- the slope is the monotonized central `s_d = minmod(2a, 2b, (a + b)/2)`,
  which is the unlimited `(a + b)/2` wherever the data are smooth and
  monotone;
- a child on side `σ_d = ±1` gets `ū + φ Σ_d σ_d s_d / 4`;
- `φ = min(1, (M − ū)/S, (ū − m)/S)` with `S = Σ_d |s_d|/4` and `M`, `m`
  the largest and smallest of `ū` and its face neighbours (`φ = 1` where
  every slope is zero) — the largest scaling that keeps every child in
  `[m, M]`.

The bound comes from `φ`, not from the slope, so the slope is a matter
of accuracy only, and there is no choice of slope to make. The `2^D`
children of a cell still average to it, so the transfer stays
conservative, and linear data are still reproduced exactly. Unlike the
unbounded operator it is not a tensor product: it drops the mixed terms
of the tensor-product stencil, so on smooth data it differs from the
unbounded operator by `O(h²)` in `D ≥ 2`, and it stays second order.
`bounded = false`, the default, is the tensor-product operator,
unchanged. See "Bounded prolongation" under "Operators" in `CODE.md`.

**Bounded point values.** With the point-value family `bounded = true`
keeps every interpolated value — a prolongated ghost or child, a
restricted ghost or coarse point where restriction interpolates (along a
cell-like dimension at `restriction > 2`), and the derived wall point of
a reflecting vertex-like dimension — within `[m, M]`, the smallest and
largest of the `2^D` source points that enclose it, the points
multilinear interpolation would read. It is `u_lin + φ(u_w − u_lin)` with
`u_w` the order-`p` interpolant, `u_lin` the multilinear one and `φ ∈
[0, 1]` the largest value that stays in range; since `u_lin` lies in
`[m, M]`, that is `u_w` clamped to `[m, M]`, which is how it is computed.
Monotone data stay monotone, a positive field stays positive, linear data
are reproduced exactly, and where no clamp is active the value is the
unbounded one bit for bit, of order `p`. Width 2 is linear interpolation,
bounded already, and is accepted so that an application can change
orders without touching the flag. It is for primitive variables in the
ghost fill (TreeGRRMHD's AMR-10). See "Bounded point values" under
"Operators" in `CODE.md`.

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
requirement is `G ≥ prolongation ÷ 2 - 1`.

**Staggered conservative sets.** What a face- or edge-centered quantity
stores is an average along its cell-like dimensions and a point value
along its vertex-like ones — a magnetic flux through a face, a vector
potential's line integral along an edge. The conservative family serves
such a set as a tensor product of the two: conservative operators along
the cell-like dimensions (the two-cell average, and subcell averages of
the order-`prolongation` reconstruction), and along the vertex-like ones
injection and point-value Lagrange prolongation of width
`vertex_prolongation`, which must be even and is then required — there
is no default, as for the other orders. A coarse edge's line integral is
then the sum of its fine edges' after every restriction, prolongation,
ghost fill and regrid, so the circulation around a coarse face is the sum
around its fine faces, and a coarse face's flux the sum of its fine
faces' (TreeGRRMHD's AMR-9, for `B = curl A`). A cell-centered set
ignores `vertex_prolongation`, and without it a conservative set with a
vertex-like dimension is refused. `bounded = true` is not implemented
together with it. The vertex-like requirement is `G ≥
vertex_prolongation ÷ 2 - 1`.
"""
struct Operators
    family::OperatorFamily
    prolongation::Int
    restriction::Int
    bounded::Bool                      # the operators create no new extrema
    vertex_prolongation::Int           # conservative, along a stagger; 0 if none

    # Sentinel defaults rather than required keywords, so that omitting
    # one reports *why* there is no default instead of a bare
    # UndefKeywordError.
    function Operators(; prolongation::Union{Integer,Nothing}=nothing,
                       restriction::Union{Integer,Nothing}=nothing,
                       family::OperatorFamily=PointValue, bounded::Bool=false,
                       vertex_prolongation::Union{Integer,Nothing}=nothing)
        # Before the missing-order check: the element family has no
        # orders, so asking for them is the error, not omitting them.
        if family === ElementLocal
            for (name, p) in (("prolongation", prolongation),
                              ("restriction", restriction),
                              ("vertex_prolongation", vertex_prolongation))
                p === nothing || throw(ArgumentError(
                    "the ElementLocal family takes no $name: its operators are the " *
                    "element's own, of the order P the element-centered field set " *
                    "carries, so there is no order to choose here. Write " *
                    "`Operators(family = ElementLocal)`."))
            end
            bounded && throw(ArgumentError(
                "bounded = true is not available with the ElementLocal family: its " *
                "operators are the element's polynomial interpolation and its " *
                "compatible restriction, fixed by the order P of the field set, and " *
                "bounding them would break the compatibility that keeps a " *
                "summation-by-parts scheme's energy invariant across a coarse-fine face"))
            return new(family, 0, 0, false, 0)
        end
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
        !bounded || family === PointValue || prolongation in (1, 3) ||
            throw(ArgumentError(
                "bounded = true with the conservative family needs prolongation = 3 or " *
                "1, got $prolongation: the bound limits the slope of the order-3 " *
                "reconstruction, the one conservative prolongation with a slope to " *
                "limit, and order 1, which copies the parent, is bounded already"))
        if vertex_prolongation !== nothing
            family === Conservative || throw(ArgumentError(
                "vertex_prolongation names the point-value width the conservative " *
                "family uses along a vertex-like dimension; the point-value family " *
                "uses its own prolongation order there, so pass that instead"))
            vertex_prolongation >= 2 && iseven(vertex_prolongation) ||
                throw(ArgumentError(
                    "vertex_prolongation must be even and at least 2 (a vertex-like " *
                    "point sits half a coarse spacing off its neighbours, so a " *
                    "symmetric stencil has even width), got $vertex_prolongation"))
            bounded && throw(ArgumentError(
                "bounded = true together with vertex_prolongation is not implemented: " *
                "the bounded conservative prolongation is built for an order-3 cell " *
                "stencil in every dimension. Use the unbounded operators for a " *
                "staggered conservative set"))
        end
        return new(family, Int(prolongation), Int(restriction), bounded,
                   vertex_prolongation === nothing ? 0 : Int(vertex_prolongation))
    end
end

# How a transfer group's kernel bounds what it computes (see
# `TransferGroup`): not at all, by the bounded conservative prolongation,
# or by clamping the point-value interpolant into its bracket.
const UNBOUNDED = Int8(0)
const BOUNDED_CONSERVATIVE = Int8(1)
const BOUNDED_POINTVALUE = Int8(2)

# The bound a group of `kind` takes, from its operators, its mirror state
# per dimension and the set's staggers `c`. Only an operator that can make
# a new extremum takes one. Conservative order 1 copies the parent, and a
# point-value stencil of width 2 is linear interpolation, a convex
# combination of its bracket, so both are bounded already and take the
# plain kernel; so does injection, a vertex-like restriction. A point-value
# restriction interpolates along every cell-like dimension, and the
# derived wall row of a reflecting vertex-like dimension (mirror state 2)
# interpolates at the prolongation order whatever the group's kind.
function boundedmode(ops::Operators, kind::Symbol, mirror::NTuple{D,Int8},
                     c::NTuple{D,Int}) where {D}
    ops.bounded || return UNBOUNDED
    if ops.family === Conservative
        return kind === :prolong && ops.prolongation == 3 ? BOUNDED_CONSERVATIVE :
               UNBOUNDED
    end
    kind === :prolong && ops.prolongation > 2 && return BOUNDED_POINTVALUE
    kind === :restrict && ops.restriction > 2 && any(iszero, c) &&
        return BOUNDED_POINTVALUE
    any(==(Int8(2)), mirror) && ops.prolongation > 2 && return BOUNDED_POINTVALUE
    return UNBOUNDED
end

Base.show(io::IO, ops::Operators) =
    ops.family === ElementLocal ? print(io, "Operators(family=ElementLocal)") :
    print(io, "Operators(prolongation=", ops.prolongation,
          ", restriction=", ops.restriction, ", family=", ops.family,
          ops.bounded ? ", bounded=true" : "",
          ops.vertex_prolongation == 0 ? "" :
          ", vertex_prolongation=$(ops.vertex_prolongation)", ")")

# The prolongation width along a vertex-like dimension: the family's own
# order for point values, `vertex_prolongation` for the conservative
# family (0 when it has none, which `check_operators` refuses there).
vertex_order(ops::Operators) =
    ops.family === Conservative ? ops.vertex_prolongation : ops.prolongation

"""
    ghost_layers_read(ops::Operators)

How many of the coarse source block's own ghost layers a prolongation
stencil reaches into — the quantity that ties the operator order to `G`.
None for the [`ElementLocal`](@ref) family, whose transfers read owned
points only.
"""
ghost_layers_read(ops::Operators) =
    ops.family === ElementLocal ? 0 :
    ops.family === Conservative ? (ops.prolongation - 1) ÷ 2 : ops.prolongation ÷ 2

"""
    check_operators(fs::FieldSet, ops::Operators)
    check_operators(N, G, centering, P, ops::Operators)

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
| vertex | `Conservative` | injection | point values of width `vertex_prolongation` | `G ≥ vp/2 - 1` |

A conservative set with a vertex-like dimension needs
`vertex_prolongation`, and is refused without it.

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

An element-centered set ([`elementcentered`](@ref), order `P > 0`, M16)
takes the [`ElementLocal`](@ref) family and no other, and that family
serves no other set: the two equally spaced families would interpolate
Gauss–Lobatto nodes as if they were cell centres, and the element family
has no operators for equally spaced data. It needs nothing of `G` and
`N` beyond what the field set already checked (`G ≤ 1`, whole parent
elements per fine block), since every element transfer reads owned
points only.

The form `check_operators(N, G, centering, P, ops)` is the one a
schedule build calls, with the centering as symbols and the element
order `P`, `0` for none.
"""
check_operators(fs::FieldSet, ops::Operators) =
    check_operators(fs.forest.N, fs.G, fs.centering, fs.P, ops)

function check_operators(N::Integer, G::NTuple{D,Int}, centering::NTuple{D,Symbol},
                         P::Integer, ops::Operators) where {D}
    # The element family and the element centering go together, both
    # ways: neither has operators for the other's data.
    if P > 0
        ops.family === ElementLocal || throw(ArgumentError(
            "an element-centered field set (P = $P) takes Operators(family = " *
            "ElementLocal), not the $(ops.family) family: its values are the " *
            "Gauss–Lobatto nodes of degree-$P polynomials, which $(ops.family) " *
            "operators would interpolate as equally spaced points, silently wrong at " *
            "every coarse-fine face and across every regrid"))
        return nothing
    end
    ops.family === ElementLocal && throw(ArgumentError(
        "the ElementLocal family moves the polynomials of Gauss–Lobatto elements, " *
        "and this field set has none (centering $centering): its values are " *
        "equally spaced. Use the PointValue or the Conservative family, or make " *
        "the set element-centered, `centering = elementcentered($D)` with its " *
        "order `P`"))
    c = staggers(centering)
    pp = ops.prolongation
    pr = ops.restriction
    needed = ghost_layers_read(ops)

    for d in 1:D
        g = G[d]
        if c[d] == 1
            # What a face- or edge-centered quantity stores is an average
            # along its cell-like dimensions and a *point value* along
            # its vertex-like ones, so along a vertex dimension the
            # conservative family interpolates point values, at a width
            # its odd `p` does not name: `vertex_prolongation`, required
            # here. See "Operators" in CODE.md.
            pv = vertex_order(ops)
            pv == 0 && throw(ArgumentError(
                "a Conservative field set with a vertex-like dimension (dimension $d " *
                "of centering with c=$c) needs vertex_prolongation, the even width of " *
                "the point-value interpolation along it. What a staggered quantity " *
                "stores is a point value there, not an average, and the family's odd " *
                "orders do not name that interpolation; there is no default, as for " *
                "the other orders"))
            g >= pv ÷ 2 - 1 || throw(ArgumentError(
                "prolongation of order $pv into a vertex-like dimension needs " *
                "G >= $(pv ÷ 2 - 1), because its stencil reaches that many planes " *
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

# Lagrange weights are translation invariant: shifting every node and
# the target by the same amount leaves every difference
# `x - nodes[j]` and `nodes[i] - nodes[j]` unchanged, and in exact
# rational arithmetic that is an identity, not an approximation. So a
# window of `p` consecutive integer nodes is fully described by `p` and
# by where the target sits inside it, and the whole package only ever
# asks for integer or quarter-integer targets — a handful of distinct
# `(p, ξ)` pairs however big the mesh is.
#
# That is what makes the bignum arithmetic affordable. It was never in
# the per-evaluation path, but it *is* in the per-schedule path, which a
# regridding run pays at every regrid and a test suite pays once per
# problem it builds: a `p = 6` schedule over a two-level 3D mesh spent
# 90 ms and 104 MiB, almost all of it recomputing the same few weight
# vectors. See "What the ghost fill costs" in CODE.md.
#
# The cached vectors are shared, so nothing may mutate one; the two
# callers copy them into a `Stencil1D`'s element type on the way out.
#
# `LAGRANGE_LOCK` guards this cache and the element tables' alike, and is
# defined in `elements.jl`, which is included first.
const LAGRANGE_CACHE = Dict{Tuple{Int,WeightRational},Vector{WeightRational}}()

"""
    unit_lagrange_weights(p, ξ)

Weights for a window of `p` consecutive integer nodes `0:(p-1)`,
evaluated at `ξ`, memoised. See [`lagrange_weights`](@ref
TreeAMR.lagrange_weights) for what they mean and why they are exact.
"""
function unit_lagrange_weights(p::Int, ξ::Rational)
    key = (p, WeightRational(ξ))
    return lock(LAGRANGE_LOCK) do
        get!(LAGRANGE_CACHE, key) do
            lagrange_weights([Rational(j) for j in 0:(p - 1)], WeightRational(ξ))
        end
    end
end
