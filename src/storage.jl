# --- Centerings ----------------------------------------------------------
#
# Per dimension a variable lives either at cell centers (`:cell`, `N`
# values per block at the half-integer positions) or at cell boundaries
# (`:vertex`, the integer positions). A centering is therefore a
# `D`-tuple, and the familiar names are spellings of tuples — see
# "Centerings" in CODE.md for why a tuple rather than an enumeration of
# the `2^D` cases: every transfer is a product of `D` one-dimensional
# stencils, and the stencil for dimension `d` depends on the centering in
# *that dimension* alone.

"""
    cellcentered(D)

The all-`:cell` centering: `N^D` values per block at the cell centers.
This is what every field set was through M6, and what
[`FieldSet`](@ref) defaults to.
"""
cellcentered(D::Integer) = ntuple(_ -> :cell, Int(D))

"""
    vertexcentered(D)

The all-`:vertex` centering: values at the integer positions, so a block
stores its `N` owned points per dimension plus the boundary plane it
shares with its high-side neighbor.
"""
vertexcentered(D::Integer) = ntuple(_ -> :vertex, Int(D))

"""
    facecentered(D, d)

The centering of a quantity living on the faces **normal to** dimension
`d` — `:vertex` in `d`, `:cell` in every other dimension. A flux `F_d`
or a face-centered magnetic field `B_d` has this centering.
"""
facecentered(D::Integer, d::Integer) = staggeredalong(D, d, :vertex, :cell)

"""
    edgecentered(D, d)

The centering of a quantity living on the edges running **along**
dimension `d` — `:cell` in `d`, `:vertex` in every other dimension, the
complement of [`facecentered`](@ref). A constrained-transport EMF `E_d`
or a vector potential `A_d` has this centering.
"""
edgecentered(D::Integer, d::Integer) = staggeredalong(D, d, :cell, :vertex)

function staggeredalong(D::Integer, d::Integer, along::Symbol, across::Symbol)
    1 <= d <= D || throw(ArgumentError(
        "the staggered dimension must be in 1:$D, got $d"))
    return ntuple(e -> e == d ? along : across, Int(D))
end

# Validate a user-supplied centering into an `NTuple{D,Symbol}` — the
# same shape `ghostwidths` gives `G`.
function centerings(C::Tuple{Vararg{Symbol}}, ::Val{D}) where {D}
    length(C) == D || throw(ArgumentError(
        "centering must have one entry per dimension: got $(length(C)) for a " *
        "$D-dimensional forest, $C. Spell it with cellcentered($D), " *
        "vertexcentered($D), facecentered($D, d) or edgecentered($D, d)."))
    all(s -> s === :cell || s === :vertex, C) || throw(ArgumentError(
        "each centering entry must be :cell or :vertex, got $C. A dimension is " *
        "either cell-centered (values at the half-integer positions) or " *
        "vertex-like (values at the integer positions, with a shared boundary " *
        "plane)."))
    return ntuple(d -> C[d], D)
end
centerings(C, ::Val{D}) where {D} = throw(ArgumentError(
    "centering must be a tuple of :cell / :vertex symbols, got a $(typeof(C))"))

"""
    staggers(centering::NTuple{D,Symbol})
    staggers(fs::FieldSet)

The centering as the arithmetic sees it: `c_d = 1` where dimension `d`
is vertex-like, `0` where it is cell-centered. This is the `c_d` of
CODE.md — the extra stored plane, the offset from a stored index to a
position, and the length a `closed` loop adds.
"""
staggers(C::NTuple{D,Symbol}) where {D} = ntuple(d -> C[d] === :vertex ? 1 : 0, D)

"""
    Parity

How a variable behaves under a mirror across a reflecting face (M10):
`EvenParity` (a scalar, a tangential vector component: the value beyond
the wall is the value at its mirror point) or `OddParity` (the normal
component of a vector: the value changes sign). A product of components
takes the product of their parities, so `T_xy` is odd across a wall
normal to `x` and across one normal to `y`.

`NoParity` is for dimensions without a reflecting face, where the
question does not arise. It is refused in a dimension with one: a
variable without a parity has no value beyond the wall, and ghosts
without a value do not stay in the ghosts — the first regrid
prolongation that reads them carries them into the interior.

Declared per variable and per dimension through the `parity` keyword of
[`FieldSet`](@ref); see "Domain and boundaries" in `CODE.md`.
"""
@enum Parity EvenParity OddParity NoParity

"""
    FieldSet{T,D,R,A}

Block storage for `nvars` variables over every leaf of a
[`Forest`](@ref): one big persistent array holding all leaf blocks
including their ghosts,

    work :: A   # size (N+2G[1]+c[1], ..., N+2G[D]+c[D], nvars, nblocks)

Cell indices vary fastest (so a GPU reads them coalesced), then the
variable index, then the block index; blocks are ordered by the forest's
Morton key order, so [`blockkey`](@ref)`(fs, b)` is the key of block `b`.
Over a distributed forest (M7) the array holds this rank's blocks only,
the leaves of [`blockrange`](@ref), and `b` is a **local** block index;
serially that is every leaf, and `forest.leaves[b]` is the same key.

`G` is the **ghost width, per dimension**, and it lives here rather than
on the forest (amended in M8): it says how far a stencil reaches into a
neighbor's data, which is a property of what is stored and not of how
space is cut up. Two field sets over one forest with different `G` — an
evolved state with `G = 2` and the fluxes computed from it with `G = 0` —
is the normal case. It is **required**: like the operator orders, it
follows from the application's discretization, which the mesh cannot
know. Pass a plain integer for the uniform case or an `NTuple{D,Integer}`
for one width per dimension; it is stored as an `NTuple{D,Int}`.

`centering` says, per dimension, whether a value sits at a cell center
(`:cell`) or at a cell boundary (`:vertex`), and defaults to
[`cellcentered`](@ref)`(D)`. Spell it with [`cellcentered`](@ref),
[`vertexcentered`](@ref), [`facecentered`](@ref) or
[`edgecentered`](@ref). A vertex-like dimension stores one plane more —
the boundary plane the block **shares** with its high-side neighbor,
which that neighbor owns and the exchange fills exactly as it fills a
ghost. Three ranges per dimension have names, with `c_d = 1` in a
vertex-like dimension and `0` otherwise:

| stored index | role | view |
|---|---|---|
| `1 … G` | low ghosts | |
| `G+1 … G+N` | **owned** — what the state vector holds | [`interiorview`](@ref) |
| `G+1 … G+N+c` | **closed** — both boundary planes included | [`closedview`](@ref) |
| `G+N+1 … N+2G+c` | high exchange region | |

Everything outside the owned range is exchange-filled, so the asymmetry
is bookkeeping in the target ranges and nothing more. The invariant
`N ≥ 2G[d] + 2c[d]` is checked here, per dimension: a block's high
exchange region must be reachable from one ring of neighbors, and a
finer neighbor spans only `N/2` of this block's cells.

Element type `T` is generic: `Float32` for GPUs, `Float64` on a host,
or a software type such as a double-`Float32` where no hardware fp64
exists. It defaults to the forest's own [`floattype`](@ref) `R`, which is
also the type the geometry is computed in; the two are separate
parameters only so that a field set may deliberately store something
narrower than its coordinates.

`backend` says where the storage lives, and thereby where every kernel
over it runs (M6): each one takes its backend from `get_backend(fs.work)`,
so this one keyword is the whole switch. A [`GhostSchedule`](@ref) used
with this field set must be built for the same backend. On a device with
no hardware fp64 a `Float64` field set is rejected here, with a message,
rather than failing later inside a kernel compilation.

`parity` gives each variable its [`Parity`](@ref) under a mirror, and
is **required** when the forest has a reflecting face (M10): whether a
variable changes sign in a mirror is physics, which the mesh cannot
know. It holds one entry per variable, each either a single `Parity` for
every dimension or an `NTuple{D,Parity}`; every dimension with a
reflecting face needs `EvenParity` or `OddParity`. Over a forest without
reflecting faces it may be omitted, and is ignored if given.

`rotation` says how the variables turn under the quarter turn `R` of a
rotating seam, and is **required** when the forest has one (M12; see
[`Forest`](@ref)'s `rotating`), for the reason `parity` is: it is
physics. It is a signed map with one nonzero integer per variable:
variable `v` at `Rp` equals `sign(rotation[v])` times variable
`abs(rotation[v])` at `p` of the set it rotates *from*. That is the set
itself when its layout is symmetric under exchanging the seam's two
dimensions `(d1, d2)` — equal `G` and equal centering along both, as for
a cell- or vertex-centered set — and its partner in a
[`RotationPair`](@ref) otherwise. For `(ρ, vx, vy, vz)` with
`rotating = (1, 2)` it is `rotation = (1, -3, 2, 4)`: the turned velocity
has `vx′ = −vy` and `vy′ = vx`. A 90° turn sends every Cartesian tensor
component to plus or minus another one, so a signed map covers any
variable stored in Cartesian components. Refused, with the reason: a map
that is not a signed permutation of `1:nvars`; for a symmetric set, one
whose fourth power is not the identity; and one that sends a variable to
another of a different parity in a reflecting dimension. Over a forest
without a seam it may be omitted, and if given is checked for its shape
and otherwise ignored, as `parity` is.

A field set is tied to the forest's *current* leaf array. Block indices
are deliberately not stable across regridding (M4), which compacts the
block slots and rebuilds the storage.

    FieldSet(forest, nvars; G)          # element type = floattype(forest)
    FieldSet{Float32}(forest, nvars; G = (2, 0))
    FieldSet(forest, nvars; G = 0, centering = facecentered(3, 1))
    FieldSet{Float32}(forest, nvars; G = 2, backend = CUDABackend())
    FieldSet(octant, 4; G = 2,          # ρ, and a vector (vx, vy, vz)
             parity = [EvenParity, (OddParity, EvenParity, EvenParity),
                       (EvenParity, OddParity, EvenParity),
                       (EvenParity, EvenParity, OddParity)])
    FieldSet(quadrant, 4; G = 2,        # the same, on a rotating forest
             rotation = (1, -3, 2, 4))

# Examples

```jldoctest
julia> forest = Forest((2,); N = 4);

julia> fs = FieldSet(forest, 3; G = 1);

julia> size(fs.work)
(6, 3, 2)

julia> FieldSet(Forest((1, 1); N = 8), 1; G = (2, 0)).G
(2, 0)

julia> size(FieldSet(Forest((1, 1); N = 8), 1;
                     G = 0, centering = facecentered(2, 1)).work)
(9, 8, 1, 1)
```
"""
mutable struct FieldSet{T,D,R,A<:AbstractArray{T}}
    const forest::Forest{D,R}
    const nvars::Int
    const G::NTuple{D,Int}
    const centering::NTuple{D,Symbol}
    const parity::Union{Nothing,Vector{NTuple{D,Parity}}}
    # The signed variable map of a rotating seam (M12), as checked by
    # `rotations_of`; `nothing` when none was given.
    const rotation::Union{Nothing,Vector{Int}}
    # The per-variable factor each mirrored or rotated transfer
    # multiplies by, `factors[v, col]`, on the storage's backend: the
    # `3^D` mirror states of a transfer group (see `parityfactors`), and
    # over a rotating forest those `3^D` columns once per orientation
    # `r = 0…3`, column `mirror column + 3^D·r`, each times the sign of
    # the turned variable (see `seamtables`). `nothing` over a forest
    # with neither reflecting faces nor a seam. It lives here rather than
    # in the schedule because parity belongs to the variables, while a
    # schedule belongs to a layout and serves every field set of it.
    const factors::Union{Nothing,AbstractMatrix{T}}
    # The variable a rotated transfer reads for each target variable,
    # `rotvars[v, r + 1]` (M12; see `seamtables`), as `Int32` on the
    # storage's backend; `nothing` over a forest without a seam.
    const rotvars::Union{Nothing,AbstractMatrix{Int32}}
    # Replaced wholesale by regridding, which compacts the block slots
    # into a freshly sized array. Mutable so that references an
    # application already holds stay valid across a regrid.
    work::A
end

# `G` as a sentinel-defaulted keyword rather than a required one, so that
# omitting it reports *why* there is no default — the same reason
# `Operators` has no default order. `centering` does have one:
# cell-centered is what a field set was before M8, and it is the answer
# for everything that is not deliberately staggered.
function FieldSet{T}(forest::Forest{D,R}, nvars::Integer;
                     G::Union{Integer,Tuple{Vararg{Integer}},Nothing}=nothing,
                     centering=cellcentered(D),
                     parity=nothing, rotation=nothing,
                     backend::Backend=CPU()) where {T,D,R}
    nvars > 0 || throw(ArgumentError("nvars must be positive, got $nvars"))
    G === nothing && throw(ArgumentError(
        "FieldSet has no default ghost width: pass `G` explicitly. It follows " *
        "from what this field set stores and from the operator orders that read " *
        "it — an evolved state needs G >= prolongation ÷ 2, a computed flux needs " *
        "none at all — which the mesh cannot know."))
    check_floattype(T, backend)
    ghosts = ghostwidths(G, Val(D))
    centers = centerings(centering, Val(D))
    stored = storedsize(forest.N, ghosts, staggers(centers))
    parities = parities_of(parity, forest, Int(nvars))
    rotations = rotations_of(rotation, forest, Int(nvars), ghosts, staggers(centers),
                             parities)
    factors, rotvars = if hasrotating(forest)
        # A symmetric set turns into itself, so every orientation applies
        # its own map; an asymmetric one turns into its partner, whose
        # map it does not know (see `seamtables`).
        steps = symmetric_layout(forest, ghosts, staggers(centers)) ?
                (rotations, rotations, rotations) : nothing
        f, v = seamtables(T, parities, steps, Int(nvars), Val(D))
        todevice(backend, f), todevice(backend, v)
    elseif hasreflecting(forest)
        todevice(backend, parityfactors(T, parities, Val(D))), nothing
    else
        nothing, nothing
    end
    work = allocate(backend, T, (stored..., Int(nvars), length(blockrange(forest))))
    # Through the kernel rather than `fill!`, for the first-touch reason
    # in `zerofill!` below.
    zerofill!(work, backend)
    return FieldSet{T,D,R,typeof(work)}(forest, Int(nvars), ghosts, centers, parities,
                                        rotations, factors, rotvars, work)
end
FieldSet(forest::Forest{D,R}, nvars::Integer; kwargs...) where {D,R} =
    FieldSet{R}(forest, nvars; kwargs...)

staggers(fs::FieldSet) = staggers(fs.centering)

# Validate a user-supplied `parity` into one `NTuple{D,Parity}` per
# variable — the same shape `ghostwidths` gives `G` — against the faces
# the forest actually reflects at.
function parities_of(parity, forest::Forest{D}, nvars::Int) where {D}
    if parity === nothing
        hasreflecting(forest) && throw(ArgumentError(
            "this forest has reflecting faces, so the field set needs `parity`: one " *
            "entry per variable, EvenParity or OddParity in each dimension with a " *
            "reflecting face. Whether a variable changes sign in a mirror is physics " *
            "— a scalar is even, the d component of a vector is odd in d and even " *
            "elsewhere — which the mesh cannot know."))
        return nothing
    end
    (parity isa AbstractVector || parity isa Tuple) || throw(ArgumentError(
        "parity must be a vector or tuple with one entry per variable, got a " *
        "$(typeof(parity))"))
    length(parity) == nvars || throw(ArgumentError(
        "parity must have one entry per variable: got $(length(parity)) for " *
        "nvars = $nvars"))
    parities = map(collect(parity)) do p
        p isa Parity && return ntuple(_ -> p, D)
        (p isa Tuple && length(p) == D && all(q -> q isa Parity, p)) ||
            throw(ArgumentError(
                "each parity entry must be a Parity, for every dimension, or an " *
                "NTuple{$D,Parity}, one per dimension; got $(repr(p))"))
        return ntuple(d -> p[d], D)
    end
    for (v, p) in enumerate(parities), d in 1:D
        any(forest.reflecting[d]) && p[d] === NoParity && throw(ArgumentError(
            "variable $v has NoParity in dimension $d, which has a reflecting face. " *
            "A variable without a parity has no value beyond the wall, and its " *
            "ghosts there would not stay ghosts: the first prolongation that reads " *
            "them carries them into the interior. Give it EvenParity or OddParity."))
    end
    return Vector{NTuple{D,Parity}}(parities)
end

# The per-variable factor table of the mirrored transfers: column `col`
# is the mirror state `s` of a transfer group, `col = 1 + Σ_d s_d 3^(d-1)`
# with `s_d` = 0 (not mirrored along `d`), 1 (mirrored rows) or 2 (the
# derived upper wall row of a vertex-like dimension), and the entry is
# the product over `d` of `1`, the sign `σ` of the variable's parity, or
# `(1 + σ)/2` respectively — see "Ghost filling" in CODE.md. Every entry
# is -1, 0 or 1, so the multiplication in the kernel is exact.
function parityfactors(::Type{T}, parities::Vector{NTuple{D,Parity}},
                       ::Val{D}) where {T,D}
    nstates = 3^D
    factors = Matrix{T}(undef, length(parities), nstates)
    for (v, p) in enumerate(parities), col in 1:nstates
        f = 1
        for d in 1:D
            state = ((col - 1) ÷ 3^(d - 1)) % 3
            σ = p[d] === OddParity ? -1 : 1        # NoParity is never mirrored
            f *= state == 0 ? 1 : state == 1 ? σ : (1 + σ) ÷ 2
        end
        factors[v, col] = T(f)
    end
    return factors
end

# The factor-table column of a mirror state (see `parityfactors`).
mirrorcolumn(state::NTuple{D,<:Integer}) where {D} =
    1 + sum(Int(state[d]) * 3^(d - 1) for d in 1:D)

# --- Rotating seams (M12) ----------------------------------------------------
#
# A ghost region beyond a rotating seam lies in the image `R^r` of real
# data, and since `u(R^r q) = Q^r u(q)` its values are the composed
# signed map `Q^r` applied to the data at `R^{-r} p` (CODE.md, "Rotating
# seams"). A signed map splits into a permutation, which the transfer
# kernel applies on the source load through `rotvars`, and a sign, which
# it applies on the target through the factor table, where it applies a
# parity: the same two places, so a rotated transfer that is also
# mirrored multiplies one factor.

# Whether a layout turns into itself under the quarter turn: equal ghost
# widths and equal staggers along the seam's two dimensions. Such a set
# rotates from itself; any other from its partner.
function symmetric_layout(forest::Forest, G::NTuple{D,Int}, c::NTuple{D,Int}) where {D}
    d1, d2 = rotating_dims(forest)
    return G[d1] == G[d2] && c[d1] == c[d2]
end
symmetric_layout(fs::FieldSet) = symmetric_layout(fs.forest, fs.G, staggers(fs))

# Validate a user-supplied `rotation` into a `Vector{Int}`, the signed
# map one quarter turn applies, against the forest's seam. Shape is
# checked on any forest, as `parity`'s is; the rest needs the seam, and
# the fourth-power and parity checks need the set the map reads from,
# which is the set itself only for a symmetric layout. A pair's are made
# by `RotationPair`.
function rotations_of(rotation, forest::Forest{D}, nvars::Int, G::NTuple{D,Int},
                      c::NTuple{D,Int}, parities) where {D}
    if rotation === nothing
        hasrotating(forest) && throw(ArgumentError(
            "this forest has a rotating seam, rotating = $(rotating_dims(forest)), so " *
            "the field set needs `rotation`: one signed variable index per variable, " *
            "saying which variable, and with which sign, each one equals a quarter " *
            "turn away. How a variable turns — a scalar into itself, the x component " *
            "of a vector into minus its y component — is physics, which the mesh " *
            "cannot know."))
        return nothing
    end
    (rotation isa AbstractVector || rotation isa Tuple) || throw(ArgumentError(
        "rotation must be a vector or tuple with one signed variable index per " *
        "variable, got a $(typeof(rotation))"))
    length(rotation) == nvars || throw(ArgumentError(
        "rotation must have one entry per variable: got $(length(rotation)) for " *
        "nvars = $nvars"))
    q = map(collect(rotation)) do x
        x isa Integer || throw(ArgumentError(
            "each rotation entry must be a nonzero integer, a signed variable index; " *
            "got $(repr(x))"))
        return Int(x)
    end
    sort!(abs.(q)) == 1:nvars || throw(ArgumentError(
        "rotation = $(Tuple(q)) is not a signed permutation of 1:$nvars: a quarter " *
        "turn sends each variable to plus or minus exactly one variable, so the " *
        "absolute values must name every variable once, and none may be zero"))
    hasrotating(forest) || return q
    if symmetric_layout(forest, G, c)
        steps = (q, q, q, q)
        σ, s = composed_rotation(steps, nvars)
        all(v -> σ[v] == v && s[v] == 1, 1:nvars) || throw(ArgumentError(
            "rotation = $(Tuple(q)) turned four times is not the identity: four " *
            "quarter turns are no turn at all, so applying the map four times must " *
            "return every variable to itself with its sign. (This set's layout is " *
            "symmetric in the seam's dimensions $(rotating_dims(forest)), so it " *
            "turns into itself.)"))
        check_rotation_parity(forest, parities, parities, q, "rotation")
    end
    return q
end

# Compose a sequence of signed maps, one per quarter turn: after the
# steps, target variable `v` reads variable `σ[v]` of the set the last
# step reads from, times `s[v]`. A step reads the next set's variable
# `abs(q[v])` with the sign of `q[v]`, so `σ_r = σ_{r−1}` followed by step
# `r`, and the sign is the product along the way.
function composed_rotation(steps, nvars::Int)
    σ = collect(1:nvars)
    s = ones(Int, nvars)
    for q in steps, v in 1:nvars
        x = q[σ[v]]
        s[v] *= sign(x)
        σ[v] = abs(x)
    end
    return σ, s
end

# A variable and its image must have one parity in every dimension with
# a reflecting face, which lies outside the seam's plane: the rotation
# and the mirror commute, so the two declarations must agree, or a ghost
# that is both turned and mirrored would depend on which came first.
function check_rotation_parity(forest::Forest{D}, target, source, q::Vector{Int},
                               what::AbstractString) where {D}
    (target === nothing || source === nothing) && return nothing
    for v in eachindex(q), d in 1:D
        any(forest.reflecting[d]) || continue
        w = abs(q[v])
        target[v][d] == source[w][d] || throw(ArgumentError(
            "$what sends variable $v to variable $w, but in dimension $d, which has a " *
            "reflecting face, the first has $(target[v][d]) and the second " *
            "$(source[w][d]). The rotation and the mirror commute, so a variable and " *
            "its image under the quarter turn must have the same parity there."))
    end
    return nothing
end

# The factor and variable tables of the rotated transfers that fill one
# set's ghosts. `steps` holds the signed maps of the first three quarter
# turns, the set's own three times for a symmetric layout and alternating
# with its partner's for a pair; then `rotvars[v, r + 1]` is the variable
# `Q^r` reads for target `v`, and the factor at column `mirror column +
# 3^D·r` is the target variable's parity factor for that mirror state
# (see `parityfactors`; 1 without reflecting faces) times the sign of
# `Q^r`. `r = 0` is the identity, so the first `3^D` columns are the
# parity table as it was. Each entry is formed in integers and converted
# once, so a zero is `+0` whatever the sign.
#
# An asymmetric set alone, `steps = nothing`, cannot turn into itself:
# its seam ghosts come from its partner, through the tables
# `RotationPair` builds. Its own tables hold the `r = 0` columns only,
# with zero factors and the identity map for `r ≥ 1`, which no fill
# reads — a schedule that has such transfers is refused for it alone.
function seamtables(::Type{T}, parities, steps, nvars::Int, ::Val{D}) where {T,D}
    nstates = 3^D
    factors = Matrix{T}(undef, nvars, 4 * nstates)
    rotvars = Matrix{Int32}(undef, nvars, 4)
    for r in 0:3
        σ, s = r == 0 ? (collect(1:nvars), ones(Int, nvars)) :
               steps === nothing ? (collect(1:nvars), zeros(Int, nvars)) :
               composed_rotation(steps[1:r], nvars)
        rotvars[:, r + 1] .= σ
        for v in 1:nvars, col in 1:nstates
            f = 1
            for d in 1:D
                parities === nothing && break
                state = ((col - 1) ÷ 3^(d - 1)) % 3
                σd = parities[v][d] === OddParity ? -1 : 1
                f *= state == 0 ? 1 : state == 1 ? σd : (1 + σd) ÷ 2
            end
            factors[v, col + nstates * r] = T(f * s[v])
        end
    end
    return factors, rotvars
end

"""
    RotationPair(a::FieldSet, b::FieldSet)

Two field sets over one forest with a rotating seam (M12; see
[`Forest`](@ref)'s `rotating`), each of which turns into the other: the
case of a set whose layout is not symmetric under exchanging the seam's
two dimensions `(d1, d2)`. `B_x`, face-centered along `x`, is `−B_y`
across the low face of `x`, so `B_x`'s ghosts there come from `B_y`'s
data, and the other way round.

The two sets must have each other's layout with `d1` and `d2` exchanged
— ghost widths and centering — and the same number of variables,
element type and backend. Each declares, with [`FieldSet`](@ref)'s
`rotation`, how its variables turn into the *other* set's: for
`a = (B_x,)` and `b = (B_y,)` with `rotating = (1, 2)`,
`rotation = (-1,)` on `a` and `(1,)` on `b`. Four quarter turns are the
identity, so the maps must compose to it, `Q_a Q_b Q_a Q_b = I`; a
mismatched pair, or one whose maps do not, is refused with the reason.
A set whose layout is symmetric turns into itself and is never part of
a pair.

The pair holds the tables the rotated transfers of each set read: for
`a`'s targets the maps `Q_a`, `Q_a Q_b` and `Q_a Q_b Q_a` for one, two
and three quarter turns, the first and the last reading `b` and the
middle one `a` itself, since two quarter turns map a layout onto itself;
and symmetrically for `b`. It is an immutable value: [`regrid!`](@ref)
replaces each set's storage in place, so a pair stays valid across
regrids.

Fill it with [`fill_ghosts!`](@ref)`(pair, (schedule_a, schedule_b))`;
each set keeps its own [`GhostSchedule`](@ref). A set whose ghost widths
are zero in both dimensions of the plane, as a flux's are, has no ghost
across the seam at all and needs no partner.

    a = FieldSet(quadrant, 1; G = (1, 2), centering = facecentered(2, 1),
                 rotation = (-1,))                       # B_x
    b = FieldSet(quadrant, 1; G = (2, 1), centering = facecentered(2, 2),
                 rotation = (1,))                        # B_y
    pair = RotationPair(a, b)
    fill_ghosts!(pair, (GhostSchedule(a, ops), GhostSchedule(b, ops)))
"""
struct RotationPair{FA<:FieldSet,FB<:FieldSet,MF<:AbstractMatrix,MV<:AbstractMatrix{Int32}}
    a::FA
    b::FB
    # The tables of `a`'s targets and of `b`'s, as `FieldSet`'s `factors`
    # and `rotvars` are for a symmetric set (see `seamtables`).
    afactors::MF
    bfactors::MF
    arotvars::MV
    brotvars::MV
end

function RotationPair(a::FieldSet{T,D}, b::FieldSet{S,E}) where {T,D,S,E}
    a.forest === b.forest || throw(ArgumentError(
        "the two sets of a RotationPair must be over one forest: each fills its " *
        "ghosts across the seam from the other's blocks, block for block"))
    forest = a.forest
    hasrotating(forest) || throw(ArgumentError(
        "a RotationPair needs a forest with a rotating seam: without one no set " *
        "turns into another, and each fills its ghosts alone"))
    d1, d2 = rotating_dims(forest)
    swap(t) = Base.setindex(Base.setindex(t, t[d2], d1), t[d1], d2)
    (a.G == swap(b.G) && a.centering == swap(b.centering)) || throw(ArgumentError(
        "the two sets of a RotationPair must have each other's layout with the " *
        "seam's dimensions $d1 and $d2 exchanged, since a quarter turn of one is the " *
        "other: got G = $(a.G) and $(b.G), centering $(a.centering) and " *
        "$(b.centering)"))
    symmetric_layout(a) && throw(ArgumentError(
        "these sets have a layout symmetric in the seam's dimensions $d1 and $d2 " *
        "(G = $(a.G), centering $(a.centering)), so each turns into itself and fills " *
        "its ghosts alone; a RotationPair is for two sets whose layouts are each " *
        "other's swap, such as the face-centered B_x and B_y"))
    a.nvars == b.nvars || throw(ArgumentError(
        "the two sets of a RotationPair must have the same number of variables, " *
        "since each turns variable for variable into the other: got $(a.nvars) and " *
        "$(b.nvars)"))
    T === S || throw(ArgumentError(
        "the two sets of a RotationPair must store one element type, since a " *
        "transfer reads one and writes the other: got $T and $S"))
    backend = get_backend(a.work)
    samebackend(backend, get_backend(b.work)) || throw(ArgumentError(
        "the two sets of a RotationPair must live on one backend, since a transfer " *
        "reads one and writes the other: got $(nameof(typeof(backend))) and " *
        "$(nameof(typeof(get_backend(b.work))))"))
    qa, qb = a.rotation, b.rotation
    σ, s = composed_rotation((qa, qb, qa, qb), a.nvars)
    all(v -> σ[v] == v && s[v] == 1, 1:a.nvars) || throw(ArgumentError(
        "the rotations of the pair, $(Tuple(qa)) and $(Tuple(qb)), composed four " *
        "times as Q_a Q_b Q_a Q_b are not the identity: four quarter turns are no " *
        "turn at all, so a variable carried from one set to the other and back " *
        "twice must return to itself with its sign"))
    check_rotation_parity(forest, a.parity, b.parity, qa, "the rotation of the first set")
    check_rotation_parity(forest, b.parity, a.parity, qb, "the rotation of the second set")
    af, av = seamtables(T, a.parity, (qa, qb, qa), a.nvars, Val(D))
    bf, bv = seamtables(T, b.parity, (qb, qa, qb), b.nvars, Val(D))
    af, bf = todevice(backend, af), todevice(backend, bf)
    av, bv = todevice(backend, av), todevice(backend, bv)
    return RotationPair{typeof(a),typeof(b),typeof(af),typeof(av)}(a, b, af, bf, av, bv)
end

# The uniform shorthand, and the per-dimension invariant. `N ≥ 2G[d] +
# 2c[d]` is what makes a block's high exchange region reachable from one
# ring of neighbors even when those neighbors are finer and each spans
# only `N/2` coarse cells (see "Blocks" in CODE.md).
ghostwidths(G::Integer, ::Val{D}) where {D} = ghostwidths(ntuple(_ -> G, D), Val(D))
function ghostwidths(G::Tuple{Vararg{Integer}}, ::Val{D}) where {D}
    length(G) == D || throw(ArgumentError(
        "G must have one width per dimension: got $(length(G)) for a $D-dimensional " *
        "forest, $G. Pass a plain integer for the uniform case."))
    all(>=(0), G) || throw(ArgumentError("G must be nonnegative, got $G"))
    return ntuple(d -> Int(G[d]), D)
end

function storedsize(N::Int, G::NTuple{D,Int}, c::NTuple{D,Int}) where {D}
    all(d -> N >= 2 * G[d] + 2 * c[d], 1:D) || throw(ArgumentError(
        "N must be >= 2G + 2c in every dimension, got N=$N, G=$G, c=$c. A block's " *
        "high exchange region must be reachable from one ring of neighbors, and a " *
        "finer neighbor spans only N/2 of this block's cells; a vertex-like " *
        "dimension (c=1) needs one more, because its shared boundary plane makes " *
        "that region one plane longer."))
    return ntuple(d -> N + 2 * G[d] + c[d], D)
end

"""
    get_backend(fs::FieldSet)

The KernelAbstractions backend this field set's storage lives on — the
backend every kernel over it is launched with.
"""
KernelAbstractions.get_backend(fs::FieldSet) = get_backend(fs.work)

"""
    nblocks(fs::FieldSet)

The number of blocks stored — one per leaf of the underlying forest, or
over a distributed forest one per leaf of this rank's
[`blockrange`](@ref). Every per-block array is sized by this, and never
by [`nleaves`](@ref), which counts the leaves of every rank.
"""
nblocks(fs::FieldSet) = size(fs.work, ndims(fs.work))

"""
    blockkey(fs::FieldSet, b::Integer)

The [`MortonKey`](@ref) of local block `b`: leaf
`first(blockrange(fs.forest)) + b - 1`, which serially is
`fs.forest.leaves[b]`.
"""
blockkey(fs::FieldSet, b::Integer) = fs.forest.leaves[blockrange(fs.forest)[b]]

"""
    blockview(fs::FieldSet, b::Integer)
    blockview(fs::FieldSet, b::Integer, v::Integer)

A view of block `b` **including ghosts** — shape `(N+2G[1]+c[1], ...,
N+2G[D]+c[D], nvars)`, or without the trailing `nvars` for a single
variable `v`.
"""
blockview(fs::FieldSet{T,D}, b::Integer) where {T,D} =
    view(fs.work, ntuple(_ -> Colon(), D + 1)..., b)
blockview(fs::FieldSet{T,D}, b::Integer, v::Integer) where {T,D} =
    view(fs.work, ntuple(_ -> Colon(), D)..., v, b)

"""
    interiorview(fs::FieldSet, b::Integer)
    interiorview(fs::FieldSet, b::Integer, v::Integer)

A view of block `b`'s **owned** points — shape `(N, ..., N, nvars)`, or
`(N, ..., N)` for a single variable `v`. This is the part that tiles the
domain and that the ODE state vector holds, for every centering: in a
vertex-like dimension a block owns its points `0 … N-1` and the boundary
point `N` belongs to the neighbor on that side (see
[`closedview`](@ref)).
"""
interiorview(fs::FieldSet{T,D}, b::Integer) where {T,D} =
    view(fs.work, interiorranges(fs)..., :, b)
interiorview(fs::FieldSet{T,D}, b::Integer, v::Integer) where {T,D} =
    view(fs.work, interiorranges(fs)..., v, b)

# The owned range per dimension, `G[d]+1 … G[d]+N`.
interiorranges(fs::FieldSet{T,D}) where {T,D} =
    ntuple(d -> (fs.G[d] + 1):(fs.G[d] + fs.forest.N), D)

"""
    closedview(fs::FieldSet, b::Integer)
    closedview(fs::FieldSet, b::Integer, v::Integer)

A view of block `b`'s **closed** range — shape `(N+c[1], ..., N+c[D],
nvars)`, or without the trailing `nvars` for a single variable `v`: the
owned points plus the shared boundary plane at the high end of every
vertex-like dimension.

This is what a quantity defined on a block's faces or edges is computed
over — a flux `F_d` has `N+1` faces in `d`, not `N` — and it is what
[`map_blocks!`](@ref)`(...; closed = true)` launches over. In a
cell-centered field set it is exactly [`interiorview`](@ref).

The extra plane is not owned: it belongs to the high-side neighbor, and
the exchange fills it. A kernel launched over the closed range
nevertheless *writes* it, because a flux is needed on both of a block's
faces; making the two sides of a coarse-fine face agree afterwards is
the interface restriction's job (see `CODE.md`, "Conservation at
coarse-fine faces").
"""
closedview(fs::FieldSet{T,D}, b::Integer) where {T,D} =
    view(fs.work, closedranges(fs)..., :, b)
closedview(fs::FieldSet{T,D}, b::Integer, v::Integer) where {T,D} =
    view(fs.work, closedranges(fs)..., v, b)

# The closed range per dimension, `G[d]+1 … G[d]+N+c[d]`.
function closedranges(fs::FieldSet{T,D}) where {T,D}
    c = staggers(fs)
    return ntuple(d -> (fs.G[d] + 1):(fs.G[d] + fs.forest.N + c[d]), D)
end

# The offset from a stored index to a position, per dimension: half a
# cell in a cell-centered dimension (the value sits at the center), a
# whole one in a vertex-like dimension (it sits on the boundary). `one(h)
# / 2` rather than `0.5`: a floating-point literal would be an fp64
# operand and would drag the whole position into fp64. And rather than
# `oftype(h, 1//2)`, which it was until M11: that folds away for a
# hardware float, but a `Float32x2` converts a `Rational` through
# `BigFloat`, at run time, in every kernel that forms a position — 530
# bytes and some 15 µs per point, measured on the point interpolator.
# Halving one is exact in any binary type, so the value is the same.
@inline pointoffsets(h, c::NTuple{D,Int}) where {D} =
    ntuple(d -> c[d] == 1 ? one(h) : one(h) / 2, Val(D))

"""
    coordinates([S], fs::FieldSet, b::Integer, idx::NTuple{D,Integer})

The physical position of point `idx` of block `b` of `fs`. `idx` is
1-based over the **stored** array, so the owned points are
`G[d]+1 : G[d]+N` and values outside that range name the shared plane
and the ghosts, whose positions are still well defined and lie on or
outside the block's boundary.

The position depends on the field set's ghost width *and* on its
centering — `origin + (i - G[d] - 1/2)·h` in a cell-centered dimension,
`origin + (i - G[d] - 1)·h` in a vertex-like one — which is why this
takes a field set and a block index rather than a forest and a key
(replacing `cell_center`, which assumed the forest carried `G` and that
every dimension was cell-centered).

The optional leading `S` is the type the arithmetic is done in, as
everywhere in `geometry.jl`; it defaults to the field set's own element
type `T`, which is what [`fill_by_coordinates!`](@ref) and
[`boundary_by_coordinates`](@ref) hand their callbacks.
"""
function coordinates(::Type{S}, fs::FieldSet{T,D}, b::Integer,
                     idx::NTuple{D,<:Integer}) where {S,T,D}
    forest = fs.forest
    origin = block_origin(S, forest, blockkey(fs, b))
    h = spacing(S, forest, blockkey(fs, b))
    off = pointoffsets(h, staggers(fs))
    return ntuple(d -> origin[d] + (Int(idx[d]) - fs.G[d] - off[d]) * h, D)
end
coordinates(fs::FieldSet{T,D}, b::Integer, idx::NTuple{D,<:Integer}) where {T,D} =
    coordinates(T, fs, b, idx)

"""
    AllVariables(f)

Wraps a coordinate callback that is called **once per point** and returns
**every variable at once**, as an `NTuple{nvars}`, instead of once per
point and variable. It selects that form wherever the package takes such
a callback: [`fill_by_coordinates!`](@ref)`(AllVariables(f), fs)` with
`f(x) -> vals`, [`CellBoundary`](@ref)`(AllVariables(g))` with
`g(x, δ) -> vals`, [`boundary_by_coordinates`](@ref)`(AllVariables(f))`,
and [`adapt_to_initial_data!`](@ref)'s `initial`. The per-variable forms
are unchanged and remain the default.

The reason is that some states are only definable as a whole (added for
the TreeHydro application; see `CODE.md`, "Application interface"). A
hydrodynamics code states its initial and boundary data as *primitive*
variables and stores *conserved* ones, and the conversion between them
needs all the primitives of a point together: the energy density is
built from the density, the velocity and the pressure. Called once per
variable, such a callback would redo the whole conversion `nvars` times
per point and throw all but one number away — at setup for the initial
data, and at **every** right-hand side evaluation for the boundary hook.

It is a wrapper type rather than an arity test on `f` because a closure
does not advertise its arity reliably, and because the package already
spells a callback's form as a type ([`CellBoundary`](@ref)). A single
field, so `AllVariables(f)` is `isbits` whenever `f` is and can be a
kernel argument like the bare callback.

The tuple's length is checked once on the host before the launch, since
a kernel cannot report it usefully; the message names both numbers.

    fill_by_coordinates!(AllVariables(x -> (sum(x), prod(x))), fs)     # nvars == 2

!!! note "Callbacks on a device"
    The wrapped callback becomes a kernel argument, so everything it
    closes over must be `isbits` — the rule the per-variable forms state
    too. Return a tuple, not a vector. One consequence is specific to
    this form: the length check evaluates the callback **on the host**,
    so one that closes over a *device array* — which a kernel argument
    may legally do, since the backend adapts it on the way in — would
    scalar-index it there. Such a callback needs the per-variable form.
"""
struct AllVariables{F}
    f::F
end

# The sample point the host-side length check evaluates at: the first
# owned point of block 1. Under MPI a rank may hold no blocks, so every
# caller returns before this on such a rank. Any point would do
# — the check is about how many values come back, and a callback whose
# tuple length varies with position is broken in a way no single
# evaluation could catch.
allvariables_sample(fs::FieldSet{T,D}) where {T,D} =
    coordinates(T, fs, 1, ntuple(d -> fs.G[d] + 1, D))

# Check, once on the host, that an `AllVariables` callback returns one
# value per variable. A kernel cannot say this usefully: an out-of-range
# tuple index inside a launch is an error with no context on a good day
# and a wrong number on a bad one. Evaluating the callback on the host is
# always legal — everything it closes over is `isbits` by the device
# rule, which is what lets it be a kernel argument at all — and it costs
# one call at setup, or one per ghost fill for the boundary form.
function check_allvariables(vals, fs::FieldSet, what::AbstractString)
    length(vals) == fs.nvars && return nothing
    throw(ArgumentError(
        "an AllVariables $what must return one value per variable, as a tuple: " *
        "this field set has nvars = $(fs.nvars), but the callback returned " *
        "$(length(vals)). A per-variable callback takes the variable index and " *
        "returns one number; an AllVariables one takes no index and returns the " *
        "whole tuple."))
end

@kernel function coordinates_kernel!(work, f, @Const(origins), @Const(spacings),
                                     ::Val{D}, ::Val{G}, ::Val{C}) where {D,G,C}
    I = @index(Global, NTuple)                     # (i1..iD, var, block)
    v, b = I[D + 1], I[D + 2]
    origin, h = origins[b], spacings[b]
    # The stored index of owned point i is i + G[d], so `coordinates`'
    # (idx - G[d] - off[d]) is just (i - off[d]) — the same arithmetic,
    # in the same order, on the same origin and spacing, so this
    # reproduces `coordinates` bit for bit.
    off = pointoffsets(h, C)
    x = ntuple(d -> origin[d] + (I[d] - off[d]) * h, Val(D))
    work[ntuple(d -> I[d] + G[d], Val(D))..., v, b] = f(x, v)
end

# The all-variables form: no variable axis in the ndrange, one call per
# point, every slot written from the tuple that comes back. The position
# is formed by the same expression as `coordinates_kernel!` above, on the
# same origin and spacing in the same order, so the two forms fill a
# field set with bit-for-bit the same numbers.
@kernel function coordinates_all_kernel!(work, f, @Const(origins), @Const(spacings),
                                         ::Val{D}, ::Val{G}, ::Val{C},
                                         ::Val{NV}) where {D,G,C,NV}
    I = @index(Global, NTuple)                     # (i1..iD, block)
    b = I[D + 1]
    origin, h = origins[b], spacings[b]
    off = pointoffsets(h, C)
    x = ntuple(d -> origin[d] + (I[d] - off[d]) * h, Val(D))
    vals = f(x)
    idx = ntuple(d -> I[d] + G[d], Val(D))
    # Unrolled through `Val`, so every tuple index is a literal: a
    # runtime index into a tuple would spill it to local memory.
    ntuple(Val(NV)) do v
        work[idx..., v, b] = vals[v]
        nothing
    end
end

"""
    fill_by_coordinates!(f, fs::FieldSet)
    fill_by_coordinates!(AllVariables(f), fs::FieldSet)

Set every **owned** point of every block from the callback
`f(x, v) -> value`, where `x` is that point's position — a cell center,
a face center, a vertex, whatever this field set's centering says (an
`NTuple{D,T}` in its own element type, see [`coordinates`](@ref)) — and
`v` the variable index. The ghosts and the shared boundary plane are
left untouched: they are filled by the ghost exchange (M2).

`f` is called once per point from a KernelAbstractions kernel, so it runs
concurrently across blocks (M5) and must be a pure function of its
arguments. The tree is not consulted: the kernel gets the geometry as
the two plain per-block arrays [`block_origins`](@ref) and
[`block_spacings`](@ref), which is what makes it a device kernel (M6).

Wrapping the callback in [`AllVariables`](@ref) selects the second form:
`f(x) -> vals` is then called **once per point** and returns all
`fs.nvars` values as a tuple, which is what a state that is only
definable as a whole needs. The two forms fill a field set with
bit-for-bit the same numbers — the position is formed by the same
expression, on the same origin and spacing, in the same order — so
switching between them is not a numerical change. The tuple's length is
checked once on the host, at the first owned point of block 1, before
anything is launched.

!!! note "Callbacks on a device"
    The callback becomes a kernel argument, so everything it closes over
    must be `isbits`. A captured `Type` is the usual trip: write
    `oftype(x[1], 2)` rather than closing over `T` and calling `T(2)`.
    The same rule covers captured arrays (pass a device array, or index
    the one the callback is already given) and any mutable state, which
    the purity requirement rules out anyway.
"""
function fill_by_coordinates!(f, fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    backend = get_backend(fs.work)
    # The geometry is built on the host and moved to wherever the kernel
    # runs. This is a setup-frequency call (initial data, and one pass of
    # `adapt_to_initial_data!`), not part of the per-evaluation path, so
    # the upload is not worth caching.
    origins = todevice(backend, block_origins(forest, T))
    spacings = todevice(backend, block_spacings(forest, T))
    launch_by_owner!(coordinates_kernel!, backend, fs.work, f, origins, spacings,
                     Val(D), Val(fs.G), Val(staggers(fs));
                     ndrange=(ntuple(_ -> forest.N, D)..., fs.nvars, nblocks(fs)))
    synchronize(backend)
    return fs
end

function fill_by_coordinates!(w::AllVariables, fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    backend = get_backend(fs.work)
    # A rank without blocks has no point to check the callback at, and
    # nothing to fill.
    nblocks(fs) == 0 && return fs
    check_allvariables(w.f(allvariables_sample(fs)), fs, "fill callback")
    origins = todevice(backend, block_origins(forest, T))
    spacings = todevice(backend, block_spacings(forest, T))
    launch_by_owner!(coordinates_all_kernel!, backend, fs.work, w.f, origins, spacings,
                     Val(D), Val(fs.G), Val(staggers(fs)), Val(fs.nvars);
                     ndrange=(ntuple(_ -> forest.N, D)..., nblocks(fs)))
    synchronize(backend)
    return fs
end

@kernel function zero_kernel!(work)
    I = @index(Global, NTuple)
    work[I...] = zero(eltype(work))
end

# Zeroing fresh block storage through a kernel rather than `fill!` is
# not about speed: on a multi-socket node it is the *first touch* that
# decides which NUMA domain each page lands in, and a serial `fill!`
# would park the whole array on whichever domain the driver thread sits
# on. `work` is block-shaped (last axis the block), and the launch is by
# owner, so each block's pages land on the domain of the thread that
# will compute on it — which is what makes first-touch placement
# domain-local without `numactl` (`CODE.md`, "What one process loses").
function zerofill!(work, backend)
    launch_by_owner!(zero_kernel!, backend, work; ndrange=size(work))
    synchronize(backend)
    return work
end
