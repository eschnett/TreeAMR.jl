# The mutable part of a forest. `pool` holds the message buffers of the
# exchanges over the forest between the stages that use them: a regrid
# builds new stages, and they take their buffers, and host mirrors, from
# it instead of allocating them again (`BufferPool`). It is made on first
# use, so a serial forest, whose stages have no messages, never has one.
mutable struct ForestState
    generation::Int
    pool::Union{Nothing,BufferPool}
end
ForestState() = ForestState(0, nothing)

function bufferpool(forest)
    state = forest.state
    state.pool === nothing && (state.pool = BufferPool())
    return state.pool::BufferPool
end

# What fills the ghosts beyond a face of the brick: the boundary hook
# (an outer face), a mirror (M10), or the nearest interior cell. A
# periodic dimension's faces are `OuterFace` and never asked about.
@enum FaceKind::Int8 OuterFace ReflectingFace OutflowFace

# --- The connectivity (M17) ------------------------------------------------
#
# How the forest's roots fit together. A forest built from `roots` is one
# brick, and every step between its roots is index arithmetic on it, its
# periodic wraps and its seam (`OneBrick`). A forest built from `patches`
# is several bricks glued at their faces (`Patches`): inside a patch the
# arithmetic is the same, and only a step that leaves one consults the
# glue table, so the common case pays one type test for it (CODE.md, "A
# multi-patch forest" under "Domain and boundaries").
#
# Abstract-typed in the forest, as `comm` is, so that `Forest{D,T}` keeps
# its two parameters and no signature downstream changes.
abstract type Connectivity{D} end

# One brick of `forest.roots`, no glue: today's forest, and what
# `Forest(roots; …)` builds.
struct OneBrick{D} <: Connectivity{D} end

# One face of a patch as the glue sees it: `patch` is the patch across it
# (0 for a face that is not glued) and `face` the `(dim, side)` of the face
# there. The rest is the affine map from this patch's *level* coordinates
# to that patch's, in gather form: axis `a` over there is axis `perm[a]`
# here, reversed where `flip[a]`, and shifted by `offset[a]` roots. So the
# cell at `g` here, at level `l`, is the cell at
#
#     y[a] = g[perm[a]] + (offset[a] << l)          (no flip)
#     y[a] = (offset[a] << l) - 1 - g[perm[a]]      (flip)
#
# there, the `-1` being a cell's width as in the seam's tables below. In
# root units, not level ones, because the offset scales with the level,
# as the seam's `n2` does. The gather form is the one `RotatedSource`
# reads a turned source by, so the schedule can take `perm` and `flip`
# from here as they are.
struct GlueEntry{D}
    patch::Int
    face::NTuple{2,Int}
    perm::NTuple{D,Int8}
    flip::NTuple{D,Bool}
    offset::NTuple{D,Int}
end
GlueEntry{D}() where {D} =
    GlueEntry{D}(0, (0, 0), ntuple(e -> Int8(e), D), ntuple(_ -> false, D),
                 ntuple(_ -> 0, D))

# Several bricks of roots — patches — glued at faces. Patch `p` owns the
# flat roots `firstroot[p] .+ (0:prod(roots[p]) - 1)`, numbered within it
# as a brick's are, so root numbering stays flat and ascending across the
# patches and the curve walks over all of them unchanged. `glue` has one
# entry per patch face, at `glue_slot(p, d, side, D)`: dense, so that a
# step finds its face's entry by arithmetic, and canonically ordered
# whatever order the caller listed the glue in, so that an index into it
# is the same on every rank. `corners` says whether some patch has glued
# faces along two different dimensions, so that some step leaves it
# through two glued faces at once: a patch corner, which the neighbor
# search answers with no neighbor and the schedule refuses where a field
# set would read it. A property of the table alone, so computed here,
# once.
struct Patches{D,T} <: Connectivity{D}
    roots::Vector{NTuple{D,Int}}
    extents::Vector{NTuple{D,Tuple{T,T}}}
    firstroot::Vector{Int}
    glue::Vector{GlueEntry{D}}
    corners::Bool
end

# The entry of face `side` (1 low, 2 high) of dimension `d` of patch `p`.
@inline glue_slot(p::Integer, d::Integer, side::Integer, D::Integer) =
    ((Int(p) - 1) * D + (Int(d) - 1)) * 2 + Int(side)

# The face `(patch, dim, side)` of the entry at `slot`, the inverse of
# `glue_slot`, for a refusal that names the face a step crossed.
@inline glue_face(slot::Integer, D::Integer) =
    (fld(Int(slot) - 1, 2D) + 1, mod(fld(Int(slot) - 1, 2), D) + 1, mod(Int(slot) - 1, 2) + 1)

"""
    Forest{D,T}

A `D`-dimensional brick of `roots[1] × ... × roots[D]` octree roots
covering the rectangular physical domain `extents`, refined into a
leaf-only linear octree.

- `leaves` is the sorted (by [`MortonKey`](@ref) curve order) flat
  vector of the leaves that currently tile the domain. Only leaves carry
  data; they tile the domain exactly, with no overlap.
- Refinement is **all-or-nothing**: a node is either a leaf or has
  exactly `2^D` children. There is never a partially refined node.
- `periodic[d]` selects whether dimension `d` wraps around the brick.
  Wraparound lives in the neighbor arithmetic, so periodic ghost filling
  needs no special-casing later.
- `reflecting[d]` is a `(lo, hi)` pair selecting which faces of a
  non-periodic dimension are **reflecting** (M10): the solution beyond
  them is its own mirror image, with the parity each variable declares
  on its [`FieldSet`](@ref). The tree does not see them — there is no
  neighbor across a reflecting face, as across any other non-periodic
  one — but the [`GhostSchedule`](@ref) does, and fills their ghosts by
  mirrored transfers.
- `outflow[d]` is a `(lo, hi)` pair selecting the **outflow** faces of a
  non-periodic dimension: the solution beyond them repeats the nearest
  interior cell (zero-gradient extrapolation). Like a reflecting face it
  is the schedule's, filled by transfers with clamped rather than
  mirrored target rows, so it needs no hook and no parity, and runs on
  every backend and across ranks. A nonlinear condition, such as
  forbidding inflow, is the application's, applied to the ghosts after
  the fill. A face is reflecting or outflow, not both. The faces that
  are neither periodic, reflecting nor outflow are *outer* faces, and
  belong to the boundary hook of [`fill_ghosts!`](@ref).
- `rotating = (d1, d2)` declares a **rotating seam** (M12): only one
  quadrant of the `(d1, d2)` plane is simulated, and the other three are
  its images under quarter turns about the line where the low faces of
  `d1` and `d2` meet. The rotation `R` takes `e_{d1}` to `e_{d2}` and
  `e_{d2}` to `-e_{d1}`, so the order of the pair fixes its sense. Unlike
  a reflecting face, the tree sees the seam: the low face of `d1` is
  glued to the low face of `d2`, [`neighbor_keys`](@ref) finds the real
  leaves across it, and [`balance!`](@ref) keeps the leaves on either
  side of it at one level (*conformity*). The two dimensions must have
  as many roots as each other and be neither periodic, reflecting nor
  outflow; their high faces stay outer. See "Rotating seams" in `CODE.md`.
- `halfturn = (d1, d2)` declares a **half-turn seam** (M14) instead: only
  the half of the `(d1, d2)` plane above the low face of `d1` is
  simulated, and the other half is its image under the rotation by 180°
  about the line where that face meets the centre plane of `d2`. The low
  face of `d1` is glued to itself, reversed along `d2`; the tree sees it
  as it sees a quarter turn's seam, and keeps a leaf and its image at one
  level. `roots[d2]` must be even, so that the axis lies on a root
  boundary, and `d1` and `d2` must be neither periodic, reflecting nor
  outflow. A forest has one seam at most. See "Half turn" in `CODE.md`.
- A forest may instead be several bricks — **patches** — glued at their
  faces (M17; the second form below), each with its own `roots` and its
  own logical `extents`. `glue` lists, for each glued face, the face
  across it and the signed axis permutation `map` that carries this
  patch's axes onto that patch's: `map[e] = ±a` sends axis `e` here to
  axis `a` there, reversed for a minus. Both directions of every gluing
  are listed, mutually inverse. The coordinates stay logical, patch by
  patch — the map from a patch's box to physical space is the
  application's — so patch extents may overlap or coincide, and
  [`patch`](@ref) and [`block_patches`](@ref) say which patch a block is
  in. Every patch has one cell size, and a glued face is neither
  periodic, reflecting nor outflow; a wrap is a patch glued to its own
  opposite face, so `periodic` is refused, and so is a seam. A step
  that leaves a patch through two glued faces at once, at a patch
  corner, finds no neighbor. No coarse-fine face crosses a glued face:
  [`balance!`](@ref) keeps the leaves on either side of one at one level
  (*conformity*), as across a seam. The root indices count through patch 1,
  then patch 2, and so on, and `forest.roots` and `forest.extents` are
  patch 1's. See "A multi-patch forest" in `CODE.md`.
- `N` is the per-block interior size, and it is even: cells are the
  tree's geometry, so `N` belongs here. The ghost width `G` does **not**
  — it says how far a stencil reaches into a neighbor's data, which is a
  property of what is stored, and it belongs to the
  [`FieldSet`](@ref) (amended in M8; through M6 it was a forest
  keyword).
- Blocks are cubes, so `extents` must match the aspect ratio of `roots`.
- `T` is the floating-point type the geometry is *computed* in, not
  merely stored in — see [`floattype`](@ref) and "Precision" in
  `CODE.md`.

Root indices are linearized 0-based, dimension 1 fastest, over `roots`;
see [`root_position`](@ref) and [`root_index`](@ref).

    Forest(roots; N, periodic=all false, reflecting=all false, outflow=all false,
           rotating=nothing, halfturn=nothing, extents=one unit per root,
           leaves=nothing, comm=nothing)
    Forest{T}(roots; ...)                      # geometry in `T`
    Forest(patches; N, glue, reflecting=all false, outflow=all false,
           leaves=nothing, comm=nothing)
    Forest{T}(patches; ...)

`patches` is a vector (or tuple) of `(roots = (…), extents = (…))`, the
extents defaulting as above, and `glue` a vector of
`(from = (patch, dim, side), to = (patch, dim, side), map = (…))`, with
`side` 1 for the low face and 2 for the high one. `reflecting` and
`outflow` apply to the faces of every patch that are not glued.

Without `leaves` the forest starts as its unrefined roots. With it, it
starts from that leaf list instead — keys in curve order, as
`forest.leaves` holds them — which is copied rather than aliased, and
the new forest is at [`generation`](@ref) 0. That is how a checkpoint
package such as TreeIOHDF5 restores a forest. Everything
built over a forest trusts its leaves, so the list is refused unless
every key lies in the brick, the keys strictly increase, they tile the
brick exactly — no gap, no overlap — and they are 2:1 balanced and, on a
forest with a seam or glued faces, conforming there (see [`balance!`](@ref)).

`comm` is the communicator the forest is distributed over (M7), as
[`communicator`](@ref) converts it; `nothing`, the default, is a serial
forest. Every rank holds the whole forest, and the field data of a
[`FieldSet`](@ref) over it are split by [`blockrange`](@ref): each rank
stores the blocks of one contiguous run of the leaves. Every forest
mutation is then collective — the same call with the same arguments on
every rank. See "Distributed meshes" in `CODE.md`.

# Examples

```jldoctest
julia> forest = Forest((2, 2); N = 8, periodic = (true, true));

julia> nleaves(forest)
4

julia> octant = Forest((1, 1, 1); N = 8, reflecting = ntuple(_ -> (true, false), 3));

julia> quadrant = Forest((2, 2, 1); N = 8, rotating = (1, 2),
                         reflecting = ((false, false), (false, false), (true, false)));

julia> star = Forest((1, 1, 1); N = 8, reflecting = ntuple(_ -> (true, false), 3),
                     outflow = ntuple(_ -> (false, true), 3));

julia> half = Forest((1, 2); N = 8, halfturn = (1, 2), extents = ((0, 1), (-1, 1)));

julia> next(p) = mod1(p + 1, 4);

julia> ring = Forest([(roots = (2, 1),) for p in 1:4]; N = 8,
                     glue = [g for p in 1:4 for g in
                             ((from = (p, 1, 2), to = (next(p), 1, 1), map = (1, 2)),
                              (from = (next(p), 1, 1), to = (p, 1, 2), map = (1, 2)))]);

julia> nleaves(ring), patch(ring, ring.leaves[end])
(8, 4)
```
"""
struct Forest{D,T}
    roots::NTuple{D,Int}
    periodic::NTuple{D,Bool}
    # The kind of each face, `(lo, hi)` per dimension; read it through
    # `reflecting_faces`, `outflow_faces` and `facekind`, or as the
    # properties `reflecting` and `outflow` (see `getproperty`). One byte a face,
    # as the `(lo, hi)` `Bool` pair of the reflecting faces was before the
    # outflow faces came, so the struct stays as large as it was (see
    # `rotating` below).
    faces::NTuple{D,NTuple{2,FaceKind}}
    # The rotating seam's pair of dimensions `(d1, d2)` (M12), `(d1, -d2)`
    # for a half turn (M14), `(0, 0)` for none; read it through
    # `hasrotating`, `ishalfturn`, `seam_plane`, `rotating_dims` and
    # `halfturn_dims`. The sign, not a third byte, because at D = 2 the two
    # bytes fill the padding exactly. `Int8`
    # because the cost of a ninth field was its size, not its count: two
    # bytes fit in the padding before `extents` (for D ≤ 4), so the struct
    # stays as large as it was and so does the schedule build's
    # allocation, where an `NTuple{2,Int}` added 160 and 4896 bytes to
    # `bench/ghosts.jl` (CODE.md, "The buffer pool").
    rotating::NTuple{2,Int8}
    extents::NTuple{D,Tuple{T,T}}
    N::Int
    leaves::Vector{MortonKey{D}}
    # What changes while the fields above stay: the generation, bumped
    # whenever the leaf array changes, so anything derived from the tree
    # (a GhostSchedule, say) can detect in O(1) that it is stale — a
    # same-size refine-then-coarsen would otherwise slip past a
    # leaf-count check and silently transfer the wrong data — and the
    # message-buffer pool (M7). One mutable object for both, in place of
    # the `Ref` the generation had, so that the struct stays as large as
    # it was: a ninth field made the schedule build allocate more
    # (CODE.md, "The buffer pool").
    state::ForestState
    # The processes the field data are split over (M7). Abstract-typed
    # on purpose, so that `Forest{D,T}` keeps its two parameters and no
    # `FieldSet` or schedule signature downstream changes; the price is
    # a dynamic dispatch per verb, a few per ghost fill and none in a
    # kernel ("Distributed meshes" in CODE.md).
    comm::Communicator
    # How the roots fit together (M17): `OneBrick()` for a forest built
    # from `roots`, `Patches` for one built from `patches`. Abstract-typed
    # as `comm` is, and for the same reason; immutable, and covered by the
    # digest, which is why it is a field and not part of `state`. Unlike
    # `rotating` it does not fit in padding: the struct grows by its eight
    # bytes, and the schedule build's allocation by about what a ninth
    # field's would have (CODE.md, "The buffer pool").
    conn::Connectivity{D}
end


# `G` is still accepted as a keyword so that the move can be reported
# instead of surfacing as a bare `MethodError` on an unrecognised
# keyword. It is the first thing a caller written against M6 hits.
const no_forest_ghosts = ArgumentError(
    "the ghost width moved from the forest to the field set in M8: write " *
    "`Forest(roots; N = ...)` and `FieldSet(forest, nvars; G = ...)`. Ghosts " *
    "say how far a stencil reaches into a neighbor's data, which is a property " *
    "of what is stored, not of how space is cut up — two field sets over one " *
    "forest with different G is the normal case. `G` may be a plain integer or " *
    "an NTuple{D,Integer}, one width per dimension.")

# The geometry type is a parameter rather than a fixed `Float64` because
# the *arithmetic*, not just the storage, has to stay inside it: a device
# without hardware fp64 must never evaluate a coordinate in `Float64` on
# its way into a `Float32` field. Converting at the end would not do.
function Forest{T}(roots::NTuple{D,Integer};
                   N::Integer,
                   periodic::NTuple{D,Bool}=ntuple(_ -> false, D),
                   reflecting::NTuple{D,Tuple{Bool,Bool}}=
                       ntuple(_ -> (false, false), D),
                   outflow::NTuple{D,Tuple{Bool,Bool}}=ntuple(_ -> (false, false), D),
                   rotating::Union{Nothing,Tuple{Integer,Integer}}=nothing,
                   halfturn::Union{Nothing,Tuple{Integer,Integer}}=nothing,
                   extents::NTuple{D,Tuple{Real,Real}}=
                       ntuple(d -> (zero(T), T(roots[d])), D),
                   leaves::Union{Nothing,AbstractVector{MortonKey{D}}}=nothing,
                   comm=nothing, G=nothing) where {T,D}
    G === nothing || throw(no_forest_ghosts)
    all(>(0), roots) || throw(ArgumentError("roots must all be positive, got $roots"))
    check_N(N)
    faces = face_kinds(periodic, reflecting, outflow)
    rot = halfturn === nothing ?
          check_rotating(rotating, roots, periodic, reflecting, outflow) :
          check_halfturn(halfturn, rotating, roots, periodic, reflecting, outflow)
    ext = check_brick(T, roots, extents)

    rootsI = map(Int, roots)
    list = leaf_list(Val(D), prod(rootsI), leaves)
    forest = Forest{D,T}(rootsI, periodic, faces, rot, ext, Int(N), list,
                         ForestState(), communicator(comm), OneBrick{D}())
    leaves === nothing || check_leaves(forest)
    return forest
end

check_N(N) = (N > 0 || throw(ArgumentError("N must be positive, got $N"));
              iseven(N) || throw(ArgumentError("N must be even, got $N"));
              nothing)

# The kinds of the faces, `(lo, hi)` per dimension, from the keywords that
# select them, refusing a face that would be two kinds at once.
function face_kinds(periodic::NTuple{D,Bool}, reflecting::NTuple{D,Tuple{Bool,Bool}},
                    outflow::NTuple{D,Tuple{Bool,Bool}}) where {D}
    for d in 1:D
        periodic[d] && any(reflecting[d]) && throw(ArgumentError(
            "dimension $d is both periodic and reflecting, got reflecting[$d] = " *
            "$(reflecting[d]): a periodic dimension has no faces — its last block " *
            "is the first block's neighbor — so there is nothing for a reflection " *
            "to act on. Drop one of the two."))
        periodic[d] && any(outflow[d]) && throw(ArgumentError(
            "dimension $d is both periodic and outflow, got outflow[$d] = " *
            "$(outflow[d]): a periodic dimension has no faces — its last block is " *
            "the first block's neighbor — so nothing flows out of it. Drop one of " *
            "the two."))
        for s in 1:2
            reflecting[d][s] && outflow[d][s] && throw(ArgumentError(
                "the $(s == 1 ? "low" : "high") face of dimension $d is both " *
                "reflecting and outflow: its ghosts are either the mirror image of " *
                "the interior or a copy of the nearest interior cell, not both"))
        end
    end
    return ntuple(d -> ntuple(s -> reflecting[d][s] ? ReflectingFace :
                                   outflow[d][s] ? OutflowFace : OuterFace, 2), D)
end

# The extents of one brick of `roots`, in `T`, checked: nonempty and
# increasing, and blocks that are cubes. `what` names the brick in a
# refusal.
function check_brick(::Type{T}, roots::NTuple{D,Integer}, extents,
                     what::AbstractString="") where {T,D}
    ext = ntuple(d -> (T(extents[d][1]), T(extents[d][2])), D)
    all(d -> ext[d][2] > ext[d][1], 1:D) ||
        throw(ArgumentError("each extent$what must be nonempty and increasing, got $ext"))
    # Blocks are cubes, so the root spacing must be the same in every
    # dimension.
    h = ntuple(d -> (ext[d][2] - ext[d][1]) / roots[d], D)
    # 4096 eps is the type-generic spelling of the 1e-12 this check used
    # when the geometry was always Float64: 1e-12 / eps(Float64) ≈ 4500.
    # `h` can differ from `h[1]` only by the rounding of one subtraction
    # and one division, so the slack is enormous either way; what matters
    # is that it tracks the type rather than sitting below eps(Float32).
    all(d -> isapprox(h[d], h[1]; rtol=4096 * eps(T)), 1:D) ||
        throw(ArgumentError("blocks must be cubes: extents $ext over roots $roots$what " *
                            "give anisotropic root spacings $h"))
    return ext
end

# The forest's leaf list: its unrefined roots, or a copy of the caller's.
# Always a copy: `refine!` and friends rewrite the forest's leaf array in
# place, which must never reach the caller's vector.
leaf_list(::Val{D}, nroots::Int, ::Nothing) where {D} =
    sort!([MortonKey{D}(r, 0, ntuple(_ -> 0, D)) for r in 0:(nroots - 1)])
leaf_list(::Val{D}, nroots::Int, leaves) where {D} = collect(MortonKey{D}, leaves)

# Without an explicit `T`, the geometry type follows the extents the
# caller supplied; with no extents either, it is `Float64`. Both branches
# are resolved from argument *types*, so this stays inferable.
function Forest(roots::NTuple{D,Integer};
                N::Integer,
                periodic::NTuple{D,Bool}=ntuple(_ -> false, D),
                reflecting::NTuple{D,Tuple{Bool,Bool}}=ntuple(_ -> (false, false), D),
                outflow::NTuple{D,Tuple{Bool,Bool}}=ntuple(_ -> (false, false), D),
                rotating::Union{Nothing,Tuple{Integer,Integer}}=nothing,
                halfturn::Union{Nothing,Tuple{Integer,Integer}}=nothing,
                extents::Union{Nothing,NTuple{D,Tuple{Real,Real}}}=nothing,
                leaves::Union{Nothing,AbstractVector{MortonKey{D}}}=nothing,
                comm=nothing, G=nothing) where {D}
    G === nothing || throw(no_forest_ghosts)
    if extents === nothing
        return Forest{Float64}(roots; N=N, periodic=periodic, reflecting=reflecting,
                               outflow=outflow, rotating=rotating, halfturn=halfturn,
                               leaves=leaves, comm=comm)
    end
    T = float(promote_type(ntuple(d -> promote_type(typeof(extents[d][1]),
                                                    typeof(extents[d][2])), D)...))
    return Forest{T}(roots; N=N, periodic=periodic, reflecting=reflecting,
                     outflow=outflow, rotating=rotating, halfturn=halfturn,
                     extents=extents, leaves=leaves, comm=comm)
end

# --- The patches form (M17) --------------------------------------------------
#
# A forest of several bricks glued at their faces. Every refusal says why
# the patches cannot be glued that way (CODE.md, "A multi-patch forest").

# A list of patches: a vector, or a nonempty tuple, of named tuples. The
# tuple is nonempty so that `()` stays the brick form's, as it was.
const PatchList = Union{AbstractVector,Tuple{NamedTuple,Vararg{NamedTuple}}}

# The keywords the brick form takes and this one refuses, with its reason,
# are accepted only to be refused: a bare `MethodError` on an unknown
# keyword would not say what to write instead.
function Forest{T}(patches::PatchList; N::Integer, glue=(), periodic=nothing,
                   reflecting=nothing, outflow=nothing, rotating=nothing,
                   halfturn=nothing, extents=nothing,
                   leaves::Union{Nothing,AbstractVector{<:MortonKey}}=nothing,
                   comm=nothing, G=nothing) where {T}
    G === nothing || throw(no_forest_ghosts)
    D = patch_dimension(patches)
    return patch_forest(Val(D), T, patches, glue, N, periodic, reflecting, outflow,
                        rotating, halfturn, extents, leaves, comm)
end

# Without an explicit `T`, the geometry type follows the patches' extents,
# promoted over all of them; with none, it is `Float64`. Not inferable,
# since the dimension is a length; a forest is built once.
function Forest(patches::PatchList; kwargs...)
    patch_dimension(patches)
    types = [typeof(x) for p in patches if haskey(p, :extents) for e in p.extents
             for x in e]
    T = isempty(types) ? Float64 : float(promote_type(types...))
    return Forest{T}(patches; kwargs...)
end

# The dimension of a list of patches, checking the shape of each.
function patch_dimension(patches)
    isempty(patches) && throw(ArgumentError(
        "the list of patches is empty: a forest of patches has at least one, each a " *
        "brick of roots"))
    for (p, patch) in enumerate(patches)
        patch isa NamedTuple && haskey(patch, :roots) &&
            patch.roots isa Tuple{Integer,Vararg{Integer}} || throw(ArgumentError(
            "patch $p, $(repr(patch)), is not a named tuple `(roots = (…), extents = " *
            "(…))` whose roots are a tuple of integers: each patch is a brick of " *
            "roots, with its own logical extents"))
        extra = [n for n in keys(patch) if !(n in (:roots, :extents))]
        isempty(extra) || throw(ArgumentError(
            "patch $p, $(repr(patch)), has the field(s) $(join(extra, ", ")), and a " *
            "patch has only `roots` and `extents`: the glue between patches is the " *
            "forest's `glue` keyword, and the faces' kinds its `reflecting` and " *
            "`outflow`"))
    end
    D = length(first(patches).roots)
    all(p -> length(p.roots) == D, patches) || throw(ArgumentError(
        "the patches have roots of different lengths, " *
        "$(join((p.roots for p in patches), ", ")): every patch is a brick of the " *
        "forest's dimension"))
    return D
end

function patch_forest(::Val{D}, ::Type{T}, patches, glue, N, periodic, reflecting,
                      outflow, rotating, halfturn, extents, leaves, comm) where {D,T}
    (rotating === nothing && halfturn === nothing) || throw(ArgumentError(
        "a forest of patches takes no seam, got rotating = $(repr(rotating)) and " *
        "halfturn = $(repr(halfturn)): its patches are glued by the `glue` table, and " *
        "a seam is the brick form's other way of gluing faces, by a quarter or a half " *
        "turn. The two mechanisms are not combined. A turn between patches is a glue " *
        "entry whose map turns the axes; a seam on one brick is " *
        "`Forest(roots; rotating = …)` or `halfturn = …`."))
    periodic === nothing || !any(periodic) || throw(ArgumentError(
        "a forest of patches is not periodic, got periodic = $periodic: a wrap is a " *
        "patch glued to its own opposite face, which the glue table says directly — " *
        "for dimension d of patch p, `(from = (p, d, 2), to = (p, d, 1), map = " *
        "(1, 2, …))` and its reverse — so the two ways of saying it are not " *
        "combined"))
    extents === nothing || throw(ArgumentError(
        "a forest of patches takes its extents patch by patch, as `(roots = (…), " *
        "extents = (…))`, not as a keyword: each patch has its own logical box"))
    check_N(N)
    faces = face_kinds(ntuple(_ -> false, D),
                       something(reflecting, ntuple(_ -> (false, false), D)),
                       something(outflow, ntuple(_ -> (false, false), D)))
    conn = patch_connectivity(Val(D), T, patches, glue, faces)
    list = leaf_list(Val(D), last(conn.firstroot) + prod(last(conn.roots)), leaves)
    forest = Forest{D,T}(conn.roots[1], ntuple(_ -> false, D), faces, (Int8(0), Int8(0)),
                         conn.extents[1], Int(N), list, ForestState(),
                         communicator(comm), conn)
    leaves === nothing || check_leaves(forest)
    return forest
end

# The patches and the glue table, checked, as the forest stores them.
function patch_connectivity(::Val{D}, ::Type{T}, patches, glue,
                            faces::NTuple{D,NTuple{2,FaceKind}}) where {D,T}
    np = length(patches)
    roots = NTuple{D,Int}[]
    extents = NTuple{D,Tuple{T,T}}[]
    for (p, patch) in enumerate(patches)
        r = patch.roots
        all(>(0), r) || throw(ArgumentError(
            "patch $p has roots $r: roots must all be positive"))
        ext = haskey(patch, :extents) ? patch.extents : ntuple(d -> (zero(T), T(r[d])), D)
        ext isa NTuple{D,Tuple{Real,Real}} || throw(ArgumentError(
            "patch $p has extents $(repr(ext)), which are not $D pairs `(lo, hi)`: a " *
            "patch's extents are its logical box, one interval per dimension"))
        push!(roots, map(Int, r))
        push!(extents, check_brick(T, r, ext, " of patch $p"))
    end
    # One cell size for every patch, so that `root_spacing` and the cell
    # size of each level stay the forest's, and a glued face, whose root
    # counts agree, is conforming by construction.
    h = [(e[1][2] - e[1][1]) / r[1] for (r, e) in zip(roots, extents)]
    for p in 2:np
        isapprox(h[p], h[1]; rtol=4096 * eps(T)) || throw(ArgumentError(
            "patch $p has roots of width $(h[p]), and patch 1 roots of width $(h[1]): " *
            "the patches differ in cell size, and every patch of a forest has the one " *
            "cell size of each level, in logical units — `root_spacing` is the " *
            "forest's — so that a glued face meets the face across it cell for cell. " *
            "Give the patches extents in proportion to their roots"))
    end
    firstroot = zeros(Int, np)
    for p in 2:np
        firstroot[p] = firstroot[p - 1] + prod(roots[p - 1])
    end
    nslots = 2D * np
    nslots <= typemax(Int8) || throw(ArgumentError(
        "$np patches in $D dimensions have $nslots faces, and a forest of patches has " *
        "at most $(typemax(Int8)): a step across a glued face is identified by its " *
        "face's index, and the ghost schedule carries that in one byte"))

    table = fill(GlueEntry{D}(), nslots)
    maps = fill(ntuple(identity, D), nslots)
    for (i, item) in enumerate(glue)
        item isa NamedTuple && all(n -> haskey(item, n), (:from, :to, :map)) ||
            throw(ArgumentError(
                "glue entry $i, $(repr(item)), is not a named tuple `(from = (patch, " *
                "dim, side), to = (patch, dim, side), map = (…))`: each entry glues one " *
                "face of a patch to a face of another, or of the same one"))
        p, d, s = from = check_face(item.from, :from, i, np, D)
        q, d2, s2 = to = check_face(item.to, :to, i, np, D)
        σ = check_axis_map(item.map, i, D)
        slot = glue_slot(p, d, s, D)
        table[slot].patch == 0 || throw(ArgumentError(
            "face $from is glued twice, by glue entry $i and an earlier one: a face is " *
            "glued to one face only, so it is the `from` of one entry"))
        faces[d][s] == OuterFace || throw(ArgumentError(
            "glue entry $i glues face $from, and the $(s == 1 ? "low" : "high") face of " *
            "dimension $d is $(faces[d][s] == ReflectingFace ? "reflecting" : "outflow"): " *
            "a glued face has a neighbor across it, the face it is glued to, so it is " *
            "not a wall. `reflecting` and `outflow` select a face of every patch at " *
            "once, so they cannot be set where some patch is glued"))
        outward, inward = s == 2 ? 1 : -1, s2 == 1 ? 1 : -1
        abs(σ[d]) == d2 && sign(σ[d]) * outward == inward || throw(ArgumentError(
            "glue entry $i has map = $σ, which sends the normal of face $from, axis " *
            "$d, to $(σ[d] < 0 ? "-" : "+")$(abs(σ[d])), and a step out of face $from " *
            "must be a step into face $to, along $(inward * outward > 0 ? "+" : "-")$d2: " *
            "map[$d] = $(inward * outward * d2)"))
        for e in 1:D
            e == d && continue
            a = abs(σ[e])
            roots[p][e] == roots[q][a] || throw(ArgumentError(
                "glue entry $i glues face $from to face $to, and its map = $σ sends axis " *
                "$e of patch $p, $(roots[p][e]) roots long, onto axis $a of patch $q, " *
                "$(roots[q][a]) roots long: a glued face meets the face across it root " *
                "for root, so the root counts along it agree under the map"))
        end
        # The map in gather form (see `GlueEntry`): axis `a` there is axis
        # `src[a]` here. Along the normal the offset carries this face onto
        # that one; along a reversed tangential axis it is that axis'
        # length, `x ↦ n - x`.
        src = ntuple(a -> findfirst(e -> abs(σ[e]) == a, 1:D)::Int, D)
        flip = ntuple(a -> σ[src[a]] < 0, D)
        here, there = s == 2 ? roots[p][d] : 0, s2 == 2 ? roots[q][d2] : 0
        offset = ntuple(D) do a
            src[a] == d && return there - sign(σ[d]) * here
            return flip[a] ? roots[q][a] : 0
        end
        table[slot] = GlueEntry{D}(q, (d2, s2), map(Int8, src), flip, offset)
        maps[slot] = σ
    end
    # Both directions listed, each the other's inverse: then a step from
    # either side finds its neighbor by the same table, and the two sides
    # agree on what they share.
    for p in 1:np, d in 1:D, s in 1:2
        slot = glue_slot(p, d, s, D)
        entry = table[slot]
        entry.patch == 0 && continue
        to = (entry.patch, entry.face...)
        back = table[glue_slot(to..., D)]
        σ = maps[slot]
        inverse = ntuple(a -> sign(σ[entry.perm[a]]) * Int(entry.perm[a]), D)
        back.patch == 0 && throw(ArgumentError(
            "face $((p, d, s)) is glued to face $to, but face $to is not glued back: " *
            "both directions of every gluing are listed, each the inverse of the other, " *
            "so here `(from = $to, to = $((p, d, s)), map = $inverse)`"))
        (back.patch, back.face) == (p, (d, s)) && maps[glue_slot(to..., D)] == inverse ||
            throw(ArgumentError(
                "face $((p, d, s)) is glued to face $to by map = $σ, but face $to is " *
                "glued back to face $((back.patch, back.face...)) by map = " *
                "$(maps[glue_slot(to..., D)]): both directions of every gluing are " *
                "listed, each the inverse of the other, so face $to is glued to " *
                "$((p, d, s)) by map = $inverse"))
    end
    corners = any(1:np) do p
        count(d -> any(s -> table[glue_slot(p, d, s, D)].patch != 0, 1:2), 1:D) >= 2
    end
    return Patches{D,T}(roots, extents, firstroot, table, corners)
end

# A face `(patch, dim, side)` of a glue entry, checked.
function check_face(face, which::Symbol, i::Int, np::Int, D::Int)
    face isa Tuple{Integer,Integer,Integer} || throw(ArgumentError(
        "glue entry $i has $which = $(repr(face)), which is not a face `(patch, dim, " *
        "side)` of three integers"))
    p, d, s = map(Int, face)
    1 <= p <= np || throw(ArgumentError(
        "glue entry $i has $which = $face, naming patch $p, and the forest has patches " *
        "1:$np"))
    1 <= d <= D || throw(ArgumentError(
        "glue entry $i has $which = $face, naming dimension $d outside 1:$D"))
    s in (1, 2) || throw(ArgumentError(
        "glue entry $i has $which = $face, on side $s: a side is 1 for the low face of " *
        "a dimension and 2 for the high one"))
    return (p, d, s)
end

# The `map` of a glue entry, checked: a signed permutation of the axes.
# Not `signed_map`, which checks a map of *variables* and says so.
function check_axis_map(m, i::Int, D::Int)
    m isa Tuple{Vararg{Integer}} || throw(ArgumentError(
        "glue entry $i has map = $(repr(m)), which is not a tuple of integers: the " *
        "map is a signed permutation of the axes, such as (2, -1, 3), carrying axis e " *
        "of the patch to axis abs(map[e]) of the patch across, reversed where " *
        "map[e] < 0"))
    length(m) == D || throw(ArgumentError(
        "glue entry $i has map = $m, with $(length(m)) entries, and the forest has $D " *
        "dimensions: the map has one signed axis per axis of the patch, the one it is " *
        "carried to across the face"))
    σ = ntuple(e -> Int(m[e]), D)
    all(e -> 1 <= abs(σ[e]) <= D, 1:D) || throw(ArgumentError(
        "glue entry $i has map = $m, which is not a signed permutation: each entry is " *
        "a signed axis of the patch across, ±1 through ±$D"))
    allunique(map(abs, σ)) || throw(ArgumentError(
        "glue entry $i has map = $m, which is not a signed permutation: it sends two " *
        "axes onto one, and a face carried onto a face sends each axis to a different " *
        "one"))
    return σ
end

# The patches form's arguments, `(; patches, glue)`, for a forest rebuilt
# from another one's parameters (`Forest(kw.patches; glue = kw.glue, …)`),
# with the glue in canonical order; `nothing` for a forest of one brick,
# which is rebuilt from `forest.roots` and `seam_keywords`, as before M17.
function patch_keywords(forest::Forest{D,T}) where {D,T}
    forest.conn isa OneBrick && return nothing
    conn = forest.conn::Patches{D,T}
    patches = [(roots=conn.roots[p], extents=conn.extents[p])
               for p in eachindex(conn.roots)]
    glue = NamedTuple{(:from, :to, :map),
                      Tuple{NTuple{3,Int},NTuple{3,Int},NTuple{D,Int}}}[]
    for p in eachindex(conn.roots), d in 1:D, s in 1:2
        entry = conn.glue[glue_slot(p, d, s, D)]
        entry.patch == 0 && continue
        σ = ntuple(D) do e
            a = findfirst(==(e), entry.perm)::Int
            return entry.flip[a] ? -a : a
        end
        push!(glue, (from=(p, d, s), to=(entry.patch, entry.face...), map=σ))
    end
    return (patches=patches, glue=glue)
end

# Whether the forest is made of patches (M17), and how many it has.
ispatched(forest::Forest) = forest.conn isa Patches
npatches(forest::Forest) = forest.conn isa Patches ? length(forest.conn.roots) : 1

# The total number of roots, over every patch.
nroots(forest::Forest{D}) where {D} =
    forest.conn isa Patches ?
    (conn = forest.conn::Patches{D}; last(conn.firstroot) + prod(last(conn.roots))) :
    prod(forest.roots)

"""
    patch(forest, k::MortonKey)
    patch(forest, root::Integer)

The patch (M17) that leaf `k`, or flat root index `root`, lies in: 1 on a
forest of one brick, and otherwise the index into the list of patches the
forest was built from. Patch extents may overlap or coincide, since the
coordinates are logical patch by patch, so it is the patch number, not a
position, that says which of the application's maps a block is under; see
[`block_patches`](@ref) for every block at once.
"""
patch(forest::Forest{D}, root::Integer) where {D} =
    forest.conn isa Patches ?
    searchsortedlast((forest.conn::Patches{D}).firstroot, Int(root)) : 1
patch(forest::Forest{D}, k::MortonKey{D}) where {D} = patch(forest, k.root)

# The roots and extents of patch `p`: the forest's own on a forest of one
# brick.
patch_roots(forest::Forest{D}, p::Integer) where {D} =
    forest.conn isa Patches ? (forest.conn::Patches{D}).roots[p] : forest.roots
patch_extents(forest::Forest{D,T}, p::Integer) where {D,T} =
    forest.conn isa Patches ? (forest.conn::Patches{D,T}).extents[p] : forest.extents

# The first flat root index of patch `p`.
patch_firstroot(forest::Forest{D}, p::Integer) where {D} =
    forest.conn isa Patches ? (forest.conn::Patches{D}).firstroot[p] : 0

# The `rotating` keyword, checked, as the forest stores it: `(0, 0)` for
# none. Each refusal says why the seam cannot be glued that way (CODE.md,
# "Rotating" under "Domain and boundaries").
function check_rotating(rotating, roots::NTuple{D,Integer}, periodic::NTuple{D,Bool},
                        reflecting::NTuple{D,Tuple{Bool,Bool}},
                        outflow::NTuple{D,Tuple{Bool,Bool}}) where {D}
    rotating === nothing && return (Int8(0), Int8(0))
    d1, d2 = Int(rotating[1]), Int(rotating[2])
    D >= 2 || throw(ArgumentError(
        "rotating = $rotating needs at least two dimensions, and this forest has " *
        "$D: the rotation turns the plane of two dimensions about the line where " *
        "their low faces meet, and a $D-dimensional forest has no such plane"))
    (1 <= d1 <= D && 1 <= d2 <= D) || throw(ArgumentError(
        "rotating = $rotating names a dimension outside 1:$D: the pair (d1, d2) is " *
        "the plane of the rotation, two of the forest's own dimensions"))
    d1 != d2 || throw(ArgumentError(
        "rotating = $rotating names dimension $d1 twice: the rotation turns the " *
        "plane of two different dimensions, gluing the low face of the first to the " *
        "low face of the second"))
    roots[d1] == roots[d2] || throw(ArgumentError(
        "rotating = $rotating needs as many roots along dimension $d1 as along $d2, " *
        "got $(roots[d1]) and $(roots[d2]): the seam glues the low face of $d1 onto " *
        "the low face of $d2, root for root, so the two faces must be the same " *
        "length"))
    for d in (d1, d2)
        periodic[d] && throw(ArgumentError(
            "rotating = $rotating, but dimension $d is periodic: its low face is " *
            "the rotating seam, glued to the other dimension's low face, and a " *
            "periodic dimension has no faces — its last block is its first block's " *
            "neighbor. Drop one of the two."))
        any(reflecting[d]) && throw(ArgumentError(
            "rotating = $rotating, but dimension $d has a reflecting face, " *
            "reflecting[$d] = $(reflecting[d]): its low face is the rotating seam, " *
            "and its high face stays outer, the boundary hook's. A reflecting high " *
            "wall together with the rotation is an open question in CODE.md, not " *
            "implemented. Reflect along a dimension outside the plane of the " *
            "rotation instead, as an octant does."))
        any(outflow[d]) && throw(ArgumentError(
            "rotating = $rotating, but dimension $d has an outflow face, " *
            "outflow[$d] = $(outflow[d]): its low face is the rotating seam, and its " *
            "high face stays outer, the boundary hook's. An outflow high face " *
            "together with the rotation is an open question in CODE.md, not " *
            "implemented."))
    end
    return (Int8(d1), Int8(d2))
end

# The `halfturn` keyword (M14), checked, as the forest stores it:
# `(d1, -d2)`, the sign telling it from a quarter turn's pair. Each refusal
# says why the seam cannot be glued that way (CODE.md, "Half turn" under
# "Domain and boundaries").
function check_halfturn(halfturn, rotating, roots::NTuple{D,Integer},
                        periodic::NTuple{D,Bool}, reflecting::NTuple{D,Tuple{Bool,Bool}},
                        outflow::NTuple{D,Tuple{Bool,Bool}}) where {D}
    rotating === nothing || throw(ArgumentError(
        "rotating = $rotating and halfturn = $halfturn are both given: a forest has " *
        "one seam at most, and the two glue the low face of $(halfturn[1]) " *
        "differently — to another face by a quarter turn, or to itself by a half " *
        "turn. Declare one."))
    d1, d2 = Int(halfturn[1]), Int(halfturn[2])
    D >= 2 || throw(ArgumentError(
        "halfturn = $halfturn needs at least two dimensions, and this forest has " *
        "$D: the half turn rotates the plane of two dimensions about a line in " *
        "it, and a $D-dimensional forest has no such plane"))
    (1 <= d1 <= D && 1 <= d2 <= D) || throw(ArgumentError(
        "halfturn = $halfturn names a dimension outside 1:$D: the pair (d1, d2) is " *
        "the plane of the rotation, two of the forest's own dimensions"))
    d1 != d2 || throw(ArgumentError(
        "halfturn = $halfturn names dimension $d1 twice: the half turn glues the " *
        "low face of the first dimension to itself, reversed along the second, a " *
        "different one"))
    iseven(roots[d2]) || throw(ArgumentError(
        "halfturn = $halfturn needs an even number of roots along dimension $d2, " *
        "got $(roots[d2]): the axis of the turn lies on the centre plane of $d2, " *
        "and it must lie on a root boundary, so that no block is its own image " *
        "across the seam"))
    for d in (d1, d2)
        periodic[d] && throw(ArgumentError(
            "halfturn = $halfturn, but dimension $d is periodic: the low face of " *
            "$d1 is the seam and the faces of $d2 are mapped onto each other by " *
            "the turn, and a periodic dimension has no faces. A periodic $d2 with " *
            "a half turn is an open question in CODE.md, not implemented."))
        any(reflecting[d]) && throw(ArgumentError(
            "halfturn = $halfturn, but dimension $d has a reflecting face, " *
            "reflecting[$d] = $(reflecting[d]): the low face of $d1 is the seam, " *
            "and the high face of $d1 and both faces of $d2 stay outer, the " *
            "boundary hook's. A reflecting face together with the half turn is an " *
            "open question in CODE.md, not implemented. Reflect along a dimension " *
            "outside the plane of the rotation instead."))
        any(outflow[d]) && throw(ArgumentError(
            "halfturn = $halfturn, but dimension $d has an outflow face, " *
            "outflow[$d] = $(outflow[d]): the low face of $d1 is the seam, and the " *
            "high face of $d1 and both faces of $d2 stay outer, the boundary " *
            "hook's. An outflow face together with the half turn is an open " *
            "question in CODE.md, not implemented."))
    end
    return (Int8(d1), Int8(-d2))
end

# Whether the forest has a rotating seam, a quarter turn's (M12) or a half
# turn's (M14).
hasrotating(forest::Forest) = forest.rotating[1] != 0

# Whether the forest's seam is a half turn's (M14).
ishalfturn(forest::Forest) = forest.rotating[2] < 0

# The plane `(d1, d2)` of either seam, as positive dimensions: what the
# kernels and the stencils need, whatever the turn.
seam_plane(forest::Forest) = (Int(forest.rotating[1]), abs(Int(forest.rotating[2])))

# The rotating pair `(d1, d2)` as the `rotating` keyword takes it: `nothing`
# when the forest has no quarter-turn seam. What rebuilds a forest from
# another one's parameters passes on, through `seam_keywords`.
rotating_dims(forest::Forest) =
    hasrotating(forest) && !ishalfturn(forest) ? seam_plane(forest) : nothing

# The pair `(d1, d2)` as the `halfturn` keyword takes it, `nothing` when the
# forest has no half-turn seam (M14).
halfturn_dims(forest::Forest) = ishalfturn(forest) ? seam_plane(forest) : nothing

# The seam as the keywords that rebuild it, `(; rotating, halfturn)`, for a
# forest rebuilt from another one's parameters to splat.
seam_keywords(forest::Forest) =
    (rotating=rotating_dims(forest), halfturn=halfturn_dims(forest))

# Validate a caller's leaf list (the `leaves` keyword), already copied
# into the candidate `forest`. The storage, the ghost schedule and the
# regrid all trust `forest.leaves` without looking at it again — that is
# what lets them be built once per tree change — so a list read from a
# file, or received from another rank, is checked here instead: every
# key in the brick, strictly increasing in curve order, tiling the brick
# exactly, 2:1 balanced and conforming at a rotating seam (M12) and at
# a glued face (M17). Each refusal names the first leaf at which the list
# goes wrong.
function check_leaves(forest::Forest{D}) where {D}
    leaves = forest.leaves
    total = nroots(forest)
    # What the leaves tile, as a refusal names it.
    brick = ispatched(forest) ? "the $(npatches(forest)) patches" :
                                "the brick $(forest.roots)"
    isempty(leaves) && throw(ArgumentError(
        "the leaf list is empty: the leaves tile the brick, so there is at least one " *
        "per root, and $brick $(ispatched(forest) ? "have" : "has") $total roots"))
    # The key constructor checks the level and the coordinates, but it
    # cannot check the root, which is an index into a brick it never sees.
    for (i, k) in enumerate(leaves)
        k.root < total || throw(ArgumentError(
            "leaf $i, $k, is in root $(k.root), but $brick " *
            "$(ispatched(forest) ? "have" : "has") roots 0:$(total - 1): a root index " *
            "counts through the brick, dimension 1 fastest, so this key belongs to a " *
            "larger one"))
    end
    for i in 2:length(leaves)
        a, b = leaves[i - 1], leaves[i]
        a == b && throw(ArgumentError(
            "leaves $(i - 1) and $i are both $a: a duplicate leaf would be two blocks " *
            "for one region of space"))
        isless(a, b) || throw(ArgumentError(
            "leaves $(i - 1) and $i, $a and $b, are out of curve order: the list must " *
            "be strictly increasing, as `forest.leaves` is, because block `b` of every " *
            "field set is leaf `b`, and sorting the list here would silently reorder " *
            "whatever data were stored with it"))
    end
    # One walk along the curve. `next` is the node at which the part of
    # the brick not yet covered begins, and each leaf must begin there
    # too: be `next` or one of its first-corner descendants, the only
    # nodes that start where it starts. Then the leaves tile the brick
    # exactly, and a strictly increasing list that does not has either a
    # leaf inside the one before it, or a gap.
    next = MortonKey{D}(0, 0, ntuple(_ -> 0, D))
    for (i, k) in enumerate(leaves)
        if next === nothing || !begins_at(k, next)
            # The leaves before `k` tile exactly up to `next`, and `k`
            # comes after its predecessor in curve order. So `k` either
            # lies inside that predecessor, or begins beyond `next`,
            # leaving a gap. (Once the last root is covered, `next` is
            # `nothing`, and a later leaf in the brick can only be the
            # former.)
            i > 1 && isancestor(leaves[i - 1], k) && throw(ArgumentError(
                "leaf $i, $k, overlaps leaf $(i - 1), $(leaves[i - 1]), which contains " *
                "it: the leaves tile the brick with no overlap, since a node is either " *
                "a block or refined into its 2^$D children, never both"))
            place = i == 1 ? "the brick begins, and leaf 1, $k, does not" :
                             "leaf $(i - 1), $(leaves[i - 1]), ends, and leaf $i, $k, " *
                             "begins further on"
            throw(ArgumentError(
                "the leaves leave a gap: no leaf covers the beginning of $next, which " *
                "is where $place. The leaves tile the brick exactly, and a region " *
                "without a leaf has no block to hold its data"))
        end
        next = curve_successor(k, total)
    end
    next === nothing || throw(ArgumentError(
        "the leaves leave a gap at the end of the brick: the last leaf, " *
        "$(length(leaves)), $(leaves[end]), ends where $next begins, and no leaf " *
        "covers anything from there through the last root, $(total - 1). The leaves " *
        "tile the brick exactly, and a region without a leaf has no block to hold " *
        "its data"))
    # Balance is the threaded check; the serial search that names the
    # offending pair runs only once it has failed.
    isbalanced(forest) && return nothing
    dirs = alldirections(Val(D))
    for (i, k) in enumerate(leaves), δ in dirs
        crossing, nbrs = oriented_neighbors(forest, k, δ)
        for nb in nbrs
            if balance_slack(crossing, δ) == 0 && level(nb) != level(k)
                crossing.glue != 0 &&
                    throw(ArgumentError(glue_conformity(forest, i, k, nb, δ, crossing)))
                throw(ArgumentError(
                    "the leaves are not conforming at the rotating seam: leaf $i, $k, " *
                    "touches $nb across the seam's face in direction $δ, and their levels " *
                    "differ. On a rotating forest a leaf on the seam and its image " *
                    "across it — on the low face of the other dimension of the plane " *
                    "for a quarter turn, on the same face for a half turn — are at one " *
                    "level, so that no coarse-fine face crosses the seam (CODE.md, " *
                    "\"Rotating seams\"); balance! keeps a forest so, so this list was " *
                    "damaged, made by hand, or written for a forest without the seam"))
            end
            abs(level(nb) - level(k)) > 1 || continue
            throw(ArgumentError(
                "the leaves are not 2:1 balanced: leaf $i, $k, touches $nb, and their " *
                "levels differ by more than one. The list is refused rather than " *
                "rebalanced, because a balanced forest can only ever produce a " *
                "balanced list, so this one was damaged or made by hand; and because " *
                "the ghost schedule assumes the balance, and would silently build " *
                "wrong prolongations from an unbalanced mesh"))
        end
    end
    return nothing
end

# The conformity refusal of `check_leaves` at a glued face (M17), naming
# the face leaf `i`, `k`, crosses to touch `nb`, and the face across it.
function glue_conformity(forest::Forest{D}, i::Int, k::MortonKey{D}, nb::MortonKey{D},
                         δ::NTuple{D,Int}, crossing) where {D}
    p, d, s = glue_face(crossing.glue, D)
    entry = (forest.conn::Patches{D}).glue[crossing.glue]
    side(s) = s == 1 ? "low" : "high"
    return "the leaves are not conforming at a glued face: leaf $i, $k, in patch $p, " *
           "touches $nb across the $(side(s)) face of dimension $d of patch $p, glued " *
           "to the $(side(entry.face[2])) face of dimension $(entry.face[1]) of patch " *
           "$(entry.patch), in direction $δ, and their levels differ. A forest of " *
           "patches is conforming at its glue, a leaf and the one across a glued face " *
           "at one level, as at a rotating seam, so that no coarse-fine face crosses " *
           "the glue and the interface restriction never has to be turned across it " *
           "(CODE.md, \"A multi-patch forest\"); balance! keeps a forest so, so this " *
           "list was damaged, made by hand, or written for a forest without the glue"
end

# Whether leaf `k` begins where node `e` begins, as `e` itself or one of
# its first-corner descendants: same root, no coarser, and its
# coordinates are `e`'s scaled to its level, with nothing added. Exact
# in `UInt32`: the scaled coordinates stay below `2^level(k)`, and a
# shift by the full 32 bits (from level 0 to MAX_LEVEL) is defined, as 0.
function begins_at(k::MortonKey{D}, e::MortonKey{D}) where {D}
    (k.root == e.root && k.level >= e.level) || return false
    shift = Int(k.level) - Int(e.level)
    return all(d -> k.coords[d] == e.coords[d] << shift, 1:D)
end

# The node at which the curve continues after the subtree of `k`: `k`'s
# next sibling, else that of its nearest ancestor that has one, else the
# next root; `nothing` after the last root. Level by level, so exact at
# any depth and in any `D`, with no packed curve index to overflow.
function curve_successor(k::MortonKey{D}, nroots::Int) where {D}
    lvl = Int(k.level)
    coords = k.coords
    while lvl > 0
        # A node's position among its siblings is the bits `coords .& 1`,
        # dimension 1 most significant. The next sibling adds one: it sets
        # the least significant clear bit and clears the set ones below it.
        d = findlast(d -> iseven(coords[d]), 1:D)
        if d !== nothing
            sibling = ntuple(D) do e
                e < d ? coords[e] : e == d ? coords[e] | 0x1 : coords[e] & ~UInt32(1)
            end
            return MortonKey{D}(k.root, lvl, sibling)
        end
        # The last sibling: the parent's subtree ends here too.
        lvl -= 1
        coords = map(c -> c >> 1, coords)
    end
    k.root + 1 < nroots || return nothing
    return MortonKey{D}(k.root + 1, 0, ntuple(_ -> 0, D))
end

"""
    floattype(forest::Forest)

The floating-point type `forest`'s geometry is computed in — what
[`spacing`](@ref), [`block_origin`](@ref) and friends return, and the
element type a [`FieldSet`](@ref) or [`GhostSchedule`](@ref) over this
forest takes unless told otherwise.
"""
floattype(::Forest{D,T}) where {D,T} = T

"""
    generation(forest::Forest)

A counter bumped on every change to the leaf array. Structures derived
from the tree record it so they can tell in O(1) whether they are still
valid — see [`GhostSchedule`](@ref).
"""
generation(forest::Forest) = forest.state.generation

"""
    nleaves(forest::Forest)

The number of leaves currently tiling `forest` — all of them, on every
rank. Serially that is also the number of blocks a [`FieldSet`](@ref)
over it stores; over a distributed forest a field set stores the
`length(blockrange(forest))` blocks of this rank (see
[`blockrange`](@ref)), so a per-block array is sized by
[`nblocks`](@ref), never by `nleaves`.
"""
nleaves(forest::Forest) = length(forest.leaves)

"""
    blockrange(forest::Forest) -> UnitRange{Int}

The leaves whose blocks this rank stores: a contiguous range of
`1:nleaves(forest)`, in curve order. Local block `b` of every
[`FieldSet`](@ref) over `forest` is leaf `first(blockrange(forest)) + b - 1`,
which is what [`blockkey`](@ref) returns.

The ranges of the ranks tile `1:nleaves(forest)` in rank order, and their
lengths differ by at most one, the longer ones first — the same
equal-count arithmetic that splits a rank's blocks over its threads
(`CODE.md`, "Distributed meshes"). Every block costs the same under one
global time step, so equal counts are equal work. A rank beyond the
number of leaves owns none, which is allowed. Serially this is
`1:nleaves(forest)`, and local and global block indices coincide.
"""
blockrange(forest::Forest) =
    equalsplit(nleaves(forest), commsize(forest.comm), commrank(forest.comm) + 1)

# The rank whose blocks include global leaf `i` (M7): the inverse of
# `blockrange`, from the same split arithmetic.
leafowner(forest::Forest, i::Integer) =
    equalsplit_part(nleaves(forest), commsize(forest.comm), Int(i)) - 1

# The leaves outside `range` that touch a leaf inside it, as ascending
# global leaf indices: the union of `neighbor_keys` over every direction
# around the leaves of `range`. These are the *candidate remote targets*
# of a distributed schedule (M7): adjacency is mutually discoverable, so
# a leaf of another rank whose ghosts read one of this rank's blocks is
# found from that block across some direction. Kept here because it is
# brick knowledge, as `neighbor_keys` is. Empty when `range` is every
# leaf, which is the serial case, without a search.
function remote_neighbors(forest::Forest{D}, range::UnitRange{Int}) where {D}
    length(range) == nleaves(forest) && return Int[]
    dirs = alldirections(Val(D))
    found = threaded_collect(Int, length(range)) do hits, i
        k = forest.leaves[first(range) + i - 1]
        for δ in dirs, nbr in neighbor_keys(forest, k, δ)
            j = find_leaf(forest, nbr)::Int
            j in range || push!(hits, j)
        end
    end
    return unique!(sort!(found))
end

# Whether the forest's field data are split over more than one rank:
# what takes the distributed path of an operation that has one.
isdistributed(forest::Forest) = commsize(forest.comm) > 1

# --- The forest digest (M7) ----------------------------------------------
#
# Every forest mutation is collective, and every host pass is a
# deterministic function of its inputs, so the ranks' forests agree
# without a message — as long as the application kept the contract.
# What would go wrong *silently* on a forest that diverged is checked:
# a schedule build gathers every rank's digest and refuses, on every
# rank together, if any differs ("Every forest mutation is collective"
# in CODE.md). The digest is the generation, the leaf count, a fold of
# every leaf's hash — not `hash(forest.leaves)`, which for a long
# vector samples only some of the elements — and the brick, the
# rotating seam (M12) and the patches and their glue (M17) included. Beside it
# goes a hash of the layout the build is for, since a rank that built
# its stencils from other operators would deliver wrong ghosts as
# silently, and a flag saying whether this rank's own argument checks
# refused. Every hash here is of integers and strings, never of a
# `Symbol` or an object identity, which differ between processes.
struct ForestDigest
    generation::Int
    nleaves::Int
    leaves::UInt
    brick::UInt
    layout::UInt
    refused::Bool
end

function ForestDigest(forest::Forest, layout::UInt, refused::Bool)
    h = hash(nleaves(forest))
    for k in forest.leaves
        h = hash(k, h)
    end
    shape = (forest.roots, forest.N, forest.periodic, forest.faces,
             Int.(forest.rotating), forest.extents)
    # The patches and the glue table (M17), on a forest that has them only,
    # so that a brick's digest is what it was.
    brick = hash(string(ispatched(forest) ? (shape..., connectivity_digest(forest)) :
                        shape))
    return ForestDigest(generation(forest), nleaves(forest), h, brick, layout, refused)
end

# The patches and the glue table as integers and strings, which hash the
# same in every process.
function connectivity_digest(forest::Forest{D,T}) where {D,T}
    conn = forest.conn::Patches{D,T}
    return (conn.roots, conn.extents,
            [(e.patch, e.face, Int.(e.perm), e.flip, e.offset) for e in conn.glue])
end

sameforest(a::ForestDigest, b::ForestDigest) =
    (a.generation, a.nleaves, a.leaves, a.brick) == (b.generation, b.nleaves, b.leaves,
                                                     b.brick)

# A hash of the values a build's layout is made of, for the digest.
layouthash(values...) = hash(string(values))

# Agree, across the ranks, that `what` may go ahead: one `allgather` of
# the digest, then the same verdict on every rank. `refusal` is this
# rank's own argument error, if its checks refused; a rank that refused
# throws its own, and every other rank says which ranks refused, so no
# rank goes on to wait in an exchange the others never enter. Serially
# nothing is gathered and the refusal, if any, is thrown as it is.
function agree_on_forest(forest::Forest, what::AbstractString;
                         layout::UInt=UInt(0), refusal=nothing)
    comm = forest.comm
    if commsize(comm) == 1
        refusal === nothing || throw(refusal)
        return nothing
    end
    digests = allgather(comm, ForestDigest(forest, layout, refusal !== nothing))
    return digest_verdict(digests, what, commrank(comm), refusal)
end

# The verdict on the gathered digests, apart from the gathering so that
# it can be tested in one process.
function digest_verdict(digests::Vector{ForestDigest}, what::AbstractString,
                        rank::Integer, refusal=nothing)
    nranks = length(digests)
    refused = [r - 1 for r in 1:nranks if digests[r].refused]
    if !isempty(refused)
        refusal === nothing || throw(refusal)
        throw(ArgumentError(
            "$what was refused on rank(s) $(join(refused, ", ")) of $nranks, and so " *
            "on this one (rank $rank) too: the call is collective, and a rank that " *
            "went on would wait for the others in its first exchange. The reason is " *
            "in the error on rank $(first(refused)); the arguments evidently differ " *
            "between ranks, which they must not."))
    end
    first_ = digests[1]
    diverged = [r - 1 for r in 2:nranks if !sameforest(digests[r], first_)]
    if !isempty(diverged)
        describe(r) = (d = digests[r + 1];
                       "rank $r has generation $(d.generation) and $(d.nleaves) leaves")
        throw(ArgumentError(
            "the forest differs between ranks, so $what is refused on every rank: " *
            "$(describe(0)), but rank(s) $(join(diverged, ", ")) hold a different " *
            "one ($(join(describe.(diverged), "; "))). Every forest mutation — the " *
            "constructors, refine!, coarsen!, balance! and regrid! — is collective: " *
            "the same call with the same arguments on every rank. Over forests that " *
            "differ, the ranks would exchange the wrong data without noticing."))
    end
    mismatched = [r - 1 for r in 2:nranks if digests[r].layout != first_.layout]
    isempty(mismatched) || throw(ArgumentError(
        "$what was called for a different layout on rank(s) " *
        "$(join(mismatched, ", ")) than on rank 0, so it is refused on every rank: " *
        "the ghost widths, the centering (and an element set's order `P`), the " *
        "operators and the element type must be the same everywhere (for " *
        "regrid!, so must the field sets passed, their " *
        "variable counts, `buffer` and `transfer`; for interpolate, the basis, " *
        "`derivs`, `vars` and `exclude`; for TreeIOHDF5's save_checkpoint and " *
        "load_checkpoint, " *
        "the path and every keyword but `data`, and for write_plain the item's " *
        "name), since a rank computes the data it sends with its own stencils and " *
        "lays out what it receives by its own, and the ranks of a checkpoint send " *
        "and receive its blocks by the layout each derives from its arguments."))
    return nothing
end

# A collective build's argument checks: run `check`, which returns the
# checked values and their `layouthash`, and agree on the forest and
# the layout across the ranks before going on. A refusal on some ranks
# only is raised on all of them (see `agree_on_forest`). A refusal is an
# `ArgumentError`, or the `DimensionMismatch` `regrid!` raises for a flag
# vector of the wrong length (step 4 of M7); anything else is a bug and
# is rethrown at once.
function collective_checks(check, forest::Forest, what::AbstractString)
    distributed = isdistributed(forest)
    checked, refusal = try
        check(), nothing
    catch err
        (distributed && err isa Union{ArgumentError,DimensionMismatch}) || rethrow()
        nothing, err
    end
    distributed || return first(checked)
    agree_on_forest(forest, what; layout=checked === nothing ? UInt(0) : last(checked),
                    refusal=refusal)
    return first(checked)
end

"""
    maxlevel(forest::Forest)

The level of the finest leaf currently in `forest`.
"""
maxlevel(forest::Forest) = maximum(level, forest.leaves)

"""
    root_position(roots_or_forest, root::Integer)

The 0-based `D`-dimensional brick position of root index `root` (as
carried in a [`MortonKey`](@ref)); the inverse of [`root_index`](@ref).
On a forest of patches (M17) it is the position within the root's own
patch, which the flat index determines (see [`patch`](@ref)).
"""
function root_position(roots::NTuple{D,Int}, root::Integer) where {D}
    pos = ntuple(_ -> 0, D)
    r = Int(root)
    for d in 1:D
        pos = Base.setindex(pos, r % roots[d], d)
        r = r ÷ roots[d]
    end
    return pos
end
function root_position(forest::Forest, root::Integer)
    ispatched(forest) || return root_position(forest.roots, root)
    p = patch(forest, root)
    return root_position(patch_roots(forest, p), Int(root) - patch_firstroot(forest, p))
end

"""
    root_index(roots_or_forest, pos::NTuple)
    root_index(forest, p::Integer, pos::NTuple)

The linearized 0-based root index of 0-based brick position `pos`; the
inverse of [`root_position`](@ref). On a forest of several patches (M17)
a position names a root in each of them, so the two-argument form refuses
there, and the second form takes the patch `p` the position is in.
"""
function root_index(roots::NTuple{D,Int}, pos::NTuple{D,<:Integer}) where {D}
    idx = 0
    stride = 1
    for d in 1:D
        idx += Int(pos[d]) * stride
        stride *= roots[d]
    end
    return idx
end
function root_index(forest::Forest{D}, pos::NTuple{D,<:Integer}) where {D}
    npatches(forest) == 1 || throw(ArgumentError(
        "root_index(forest, $pos) is ambiguous on a forest of $(npatches(forest)) " *
        "patches: a position is within a patch, and names a root in each of them. " *
        "Name the patch, `root_index(forest, p, $pos)`"))
    return root_index(patch_roots(forest, 1), pos)
end
function root_index(forest::Forest{D}, p::Integer, pos::NTuple{D,<:Integer}) where {D}
    1 <= p <= npatches(forest) || throw(ArgumentError(
        "patch $p is not a patch of the forest, whose patches are 1:$(npatches(forest))"))
    return patch_firstroot(forest, p) + root_index(patch_roots(forest, p), pos)
end

"""
    alldirections(::Val{D})

All `3^D - 1` nonzero direction vectors `δ ∈ {-1,0,1}^D`. One nonzero
component names a face, two an edge, and `D` a corner — the ghost
regions that surround a block.
"""
function alldirections(::Val{D}) where {D}
    origin = ntuple(_ -> 0, D)
    return filter(!=(origin), vec(collect(Iterators.product(ntuple(_ -> (-1, 0, 1), D)...))))
end

"""
    find_leaf(forest, key) -> Union{Int,Nothing}
    find_leaf(forest, root, level, coords) -> Union{Int,Nothing}

The index into `forest.leaves` of the given leaf, or `nothing` if that
node is not currently a leaf (because it is refined, or coarsened away,
or outside the domain). `O(log nleaves)`.
"""
function find_leaf(forest::Forest{D}, key::MortonKey{D}) where {D}
    i = searchsortedfirst(forest.leaves, key)
    return (i <= length(forest.leaves) && forest.leaves[i] == key) ? i : nothing
end
find_leaf(forest::Forest{D}, root::Integer, lvl::Integer, coords::NTuple{D,<:Integer}) where {D} =
    find_leaf(forest, MortonKey{D}(root, lvl, coords))

"""
    isleaf(forest, key)

Whether `key` is currently a leaf of `forest`.
"""
isleaf(forest::Forest{D}, key::MortonKey{D}) where {D} = find_leaf(forest, key) !== nothing

# --- The rotating seam (M12) ------------------------------------------------
#
# The arithmetic of the seam, in one place: every other function sees it
# through `neighbor_anchor` and `oriented_neighbors`. A step from a block
# is taken in *global level coordinates* — a node's position at its level
# counted across the whole brick, so that the axis sits at 0 in both
# dimensions of the plane — and a naive neighbor that lands beyond the low
# face of `d1`, of `d2`, or of both, lies in the image `R^r` of real data,
# `r = 1, 3, 2` respectively. It maps back to the real node by `R^{-r}`
# about the axis (CODE.md, "Rotating seams", the first table):
#
#     r = 1:  (g1, g2) ↦ (g2, -1 - g1)
#     r = 3:  (g1, g2) ↦ (-1 - g2, g1)
#     r = 2:  (g1, g2) ↦ (-1 - g1, -1 - g2)
#
# with (g1, g2) the coordinates along (d1, d2) and every other one kept.
# The `-1` is a cell's width: the cell at `g` spans `[g, g + 1)`, and
# `R^{-1}` takes it to the cell spanning `(-g - 1, -g]` along `d2`.
#
# A half turn (M14) has orientation 2 only, entered by stepping beyond the
# low face of `d1` alone — a face of `d2` is an outer face — and turns
# about the centre plane of `d2` rather than its low face (CODE.md, *Half
# turns*):
#
#     r = 2:  (g1, g2) ↦ (-1 - g1, n2 - 1 - g2)
#
# with `n2 = roots[d2] << level` the nodes along `d2` at the level. A
# quarter turn's `r = 2` is the same map with `n2 = 0`, which is how
# `unrotate_node` takes both.

# How a step reached the node it lands on: across a rotating seam, by
# `turns` quarter turns (M12, M14; the `r` of the tables above, which
# indexes a field set's `rotvars` and `factors`), or across a glued face
# (M17), whose index into the forest's glue table is `glue`. Both are 0
# for a step that crosses neither, and a forest has seams or glue, never
# both. The glue is an index and not its axis map because two patches
# glued with the identity map still differ from one brick: a coarse-fine
# face may not cross the glue (conformity, `balance_slack`). The few
# callers that need the map itself look it up, through `glue_axes`.
struct Crossing
    turns::Int8
    glue::Int8
end
Crossing() = Crossing(0, 0)

# Whether a step crosses a seam or a glued face.
iscrossing(c::Crossing) = c.turns != 0 || c.glue != 0

# The axis map of a step across a glued face, as `(perm, flip)` in the
# gather form of `GlueEntry`: axis `a` beyond the glue is axis `perm[a]`
# of the asking block, reversed where `flip[a]`.
function glue_axes(forest::Forest{D}, c::Crossing) where {D}
    entry = (forest.conn::Patches{D}).glue[c.glue]
    return entry.perm, entry.flip
end

# The orientation of the naive node at global coordinates `g`: 0 off the
# seam, or with no seam at all.
@inline function seam_orientation(forest::Forest{D}, g::NTuple{D,Int}) where {D}
    hasrotating(forest) || return 0
    d1, d2 = seam_plane(forest)
    ishalfturn(forest) && return g[d1] < 0 ? 2 : 0
    a, b = g[d1] < 0, g[d2] < 0
    return a ? (b ? 2 : 1) : (b ? 3 : 0)
end

# The real node of the naive node `g` in orientation `r` (the tables above):
# `n2` is 0 for a quarter turn, and the nodes along `d2` at the level for a
# half turn.
@inline function unrotate_node(g::NTuple{D,Int}, r::Int, d1::Int, d2::Int,
                               n2::Int=0) where {D}
    g1, g2 = g[d1], g[d2]
    r1, r2 = r == 1 ? (g2, -1 - g1) : r == 3 ? (-1 - g2, g1) :
             r == 2 ? (-1 - g1, n2 - 1 - g2) : (g1, g2)
    return Base.setindex(Base.setindex(g, r1, d1), r2, d2)
end

# The direction `R^{-r} δ`: a ghost direction `δ` of a block as seen from
# the real leaves across a rotating seam in orientation `r` (M12). It is
# the linear part of the map from naive to real nodes, so `r = 1` takes
# `(δ1, δ2)` along `(d1, d2)` to `(δ2, -δ1)`, `r = 3` to `(-δ2, δ1)` and
# `r = 2` to `(-δ1, -δ2)`; `r = 0` is the identity.
@inline function real_direction(δ::NTuple{D,Int}, r::Integer, d1::Integer,
                                 d2::Integer) where {D}
    a, b = δ[d1], δ[d2]
    r1, r2 = r == 1 ? (b, -a) : r == 3 ? (-b, a) : r == 2 ? (-a, -b) : (a, b)
    return Base.setindex(Base.setindex(δ, r1, d1), r2, d2)
end

# The same, across a glued face (M17), from its axis map in gather form:
# the step `δ` of the asking block, as the patch across sees it. Along axis
# `a` there it is the step along `perm[a]` here, reversed where `flip[a]`.
@inline real_direction(δ::NTuple{D,Int}, perm::NTuple{D,Integer},
                       flip::NTuple{D,Bool}) where {D} =
    ntuple(a -> flip[a] ? -δ[Int(perm[a])] : δ[Int(perm[a])], D)

# The child offset `o ∈ {0,1}^D` of a real leaf across a rotating seam in
# orientation `r`, taken into the *virtual* frame, where the asking block
# sees that leaf (M12): its offset within its virtual parent, which is
# what a restriction's stencils depend on. The virtual frame is `R^r` of
# the real one, and the parent of a node maps to the parent of its image
# (`-1 - g` halves to `-1 - g ÷ 2`), so the offset turns about its
# parent's centre: `r = 1` takes `(o1, o2)` along `(d1, d2)` to
# `(1 - o2, o1)`, `r = 3` to `(o2, 1 - o1)` and `r = 2` to
# `(1 - o1, 1 - o2)`; `r = 0` is the identity.
@inline function virtual_offset(o::NTuple{D,Int}, r::Integer, d1::Integer,
                                d2::Integer) where {D}
    a, b = o[d1], o[d2]
    v1, v2 = r == 1 ? (1 - b, a) : r == 3 ? (b, 1 - a) :
             r == 2 ? (1 - a, 1 - b) : (a, b)
    return Base.setindex(Base.setindex(o, v1, d1), v2, d2)
end

# The same, across a glued face (M17), from its axis map in gather form:
# a leaf's offset along axis `a` of the patch across is its offset along
# `perm[a]` in the asking block's frame, turned about its parent's centre,
# `1 - o`, where `flip[a]`.
function virtual_offset(o::NTuple{D,Int}, perm::NTuple{D,Integer},
                        flip::NTuple{D,Bool}) where {D}
    return ntuple(D) do e
        a = findfirst(==(e), perm)::Int
        return flip[a] ? 1 - o[a] : o[a]
    end
end

# The real node one step from `k` in direction `δ`, in global level
# coordinates, and its orientation: the step taken naively, then mapped
# back across a rotating seam it crosses. Periodic dimensions are not yet
# wrapped and a high face not yet tested; `neighbor_anchor` does both.
@inline function stepped_node(forest::Forest{D}, k::MortonKey{D},
                              δ::NTuple{D,Int}) where {D}
    lvl = level(k)
    rootpos = root_position(forest.roots, k.root)
    g = ntuple(d -> (rootpos[d] << lvl) + Int(k.coords[d]) + δ[d], D)
    r = seam_orientation(forest, g)
    r == 0 && return g, 0
    d1, d2 = seam_plane(forest)
    n2 = ishalfturn(forest) ? forest.roots[d2] << lvl : 0
    return unrotate_node(g, r, d1, d2, n2), r
end

# The same-level anchor node reached by stepping one cell from `k` in
# direction `δ`, as `(root, coords, crossing)`: crossing root boundaries,
# wrapping periodic ones, and crossing a rotating seam (M12) into the
# real node, with `crossing.turns` the orientation (0 off the seam).
# `nothing` when the step leaves a non-periodic face that is not the seam,
# or crosses the seam to a node that lies beyond a high face — a region
# beyond an outer face either way, and the hook's. Since |δ[d]| <= 1, a
# step crosses at most one root boundary per dimension. This is the single
# source of truth for where a step lands: the neighbor search, `balance!`
# and, through them, everything else go through it. On a forest of
# patches (M17) the step is `glued_anchor`'s instead.
function neighbor_anchor(forest::Forest{D}, k::MortonKey{D}, δ::NTuple{D,Int}) where {D}
    forest.conn isa Patches && return glued_anchor(forest.conn::Patches{D}, k, δ)
    lvl = level(k)
    g, r = stepped_node(forest, k, δ)
    for d in 1:D
        outside = g[d] < 0 || g[d] >= forest.roots[d] << lvl
        outside && !forest.periodic[d] && return nothing
    end
    wrapped = ntuple(d -> mod(g[d], forest.roots[d] << lvl), D)
    mask = (1 << lvl) - 1
    rootpos = ntuple(d -> wrapped[d] >> lvl, D)
    return (root_index(forest.roots, rootpos), ntuple(d -> wrapped[d] & mask, D),
            Crossing(r, 0))
end

# `neighbor_anchor` on a forest of patches (M17). The step is taken in the
# patch's own level coordinates, as on a brick; a step that stays in the
# patch lands where it lands. One that leaves it through exactly one face
# crosses that face's glue, by its affine map (see `GlueEntry`), into the
# patch across, and is checked against that patch's bounds again, since
# only a step out through one face is a step in through another. A step
# out through an unglued face is beyond an outer face, and one out through
# two faces or more at once is beyond a patch corner — if both are glued,
# the region where three or more patches meet, which has no one node to
# land on. Both are `nothing`: the same "no neighbor" a step through an
# outer face gives on a brick, so that nothing downstream of the search
# meets a new case. The schedule tells a corner from an outer face by the
# connectivity's `corners` flag and the field set's centering, and refuses
# it where it would be read. There is no periodic wrap here: a wrap is a
# patch glued to itself.
function glued_anchor(conn::Patches{D}, k::MortonKey{D}, δ::NTuple{D,Int}) where {D}
    lvl = level(k)
    p = searchsortedlast(conn.firstroot, Int(k.root))
    n = conn.roots[p]
    rootpos = root_position(n, Int(k.root) - conn.firstroot[p])
    g = ntuple(d -> (rootpos[d] << lvl) + Int(k.coords[d]) + δ[d], D)
    exits, d = 0, 0
    for e in 1:D
        if g[e] < 0 || g[e] >= n[e] << lvl
            exits += 1
            d = e
        end
    end
    exits == 0 && return glued_node(conn, p, g, lvl, Crossing())
    exits == 1 || return nothing
    slot = glue_slot(p, d, g[d] < 0 ? 1 : 2, D)
    entry = conn.glue[slot]
    entry.patch == 0 && return nothing
    there = ntuple(D) do a
        x = g[Int(entry.perm[a])]
        return entry.flip[a] ? (entry.offset[a] << lvl) - 1 - x :
                               x + (entry.offset[a] << lvl)
    end
    m = conn.roots[entry.patch]
    any(e -> there[e] < 0 || there[e] >= m[e] << lvl, 1:D) && return nothing
    return glued_node(conn, entry.patch, there, lvl, Crossing(0, slot))
end

# The anchor `(root, coords, crossing)` of the node at level coordinates
# `g` of patch `p`, inside it.
@inline function glued_node(conn::Patches{D}, p::Int, g::NTuple{D,Int}, lvl::Int,
                            crossing::Crossing) where {D}
    mask = (1 << lvl) - 1
    root = conn.firstroot[p] + root_index(conn.roots[p], ntuple(e -> g[e] >> lvl, D))
    return (root, ntuple(e -> g[e] & mask, D), crossing)
end

# The kind of face `side` (1 low, 2 high) of dimension `d`, and the
# faces of each kind as `(lo, hi)` pairs per dimension, the form the
# `reflecting` and `outflow` keywords take.
# They read the field with `getfield`, since `getproperty` calls them,
# and a recursion there would widen what `forest.reflecting` infers to.
facekind(forest::Forest, d::Integer, side::Integer) =
    getfield(forest, :faces)[d][side]
reflecting_faces(forest::Forest{D}) where {D} =
    map(f -> map(==(ReflectingFace), f),
        getfield(forest, :faces))::NTuple{D,NTuple{2,Bool}}
outflow_faces(forest::Forest{D}) where {D} =
    map(f -> map(==(OutflowFace), f), getfield(forest, :faces))::NTuple{D,NTuple{2,Bool}}

# `forest.reflecting` and `forest.outflow` read as the keywords' pairs,
# derived from `faces`. The forest stored `reflecting` itself until the
# outflow faces came, and physics packages read it as a field (TreeHydro,
# TreeWaveGR, TreeGeneralizedHarmonic), so it stays readable; `outflow`
# beside it, for symmetry. A constant name folds away, so a field read
# costs what it did.
@inline function Base.getproperty(forest::Forest, name::Symbol)
    name === :reflecting && return reflecting_faces(forest)
    name === :outflow && return outflow_faces(forest)
    return getfield(forest, name)
end
Base.propertynames(forest::Forest, private::Bool=false) =
    (fieldnames(typeof(forest))..., :reflecting, :outflow)

# Whether the forest has any reflecting face at all — what decides
# whether a field set over it must declare a parity. An outflow face
# needs none.
hasreflecting(forest::Forest) = any(f -> ReflectingFace in f, forest.faces)

# Split a ghost direction of block `k` at the reflecting and outflow
# faces it crosses (M10). Returns `(δ′, walls)`: `walls[d]` is the kind of
# face that stepping from `k` by `δ[d]` leaves the domain through, when
# that face is reflecting or outflow, and `OuterFace` otherwise; `δ′` is
# `δ` with those components zeroed — the direction whose region, mirrored
# (reflecting) or clamped (outflow) along those dimensions, is the ghost
# region `δ`. So the source of region `δ` is the block itself when `δ′`
# is zero and whatever lies in direction `δ′` otherwise.
#
# Lives here rather than in the schedule because it is brick knowledge:
# which faces are the domain's, and of what kind. A rotating seam (M12)
# is never a wall: the constructor refuses a reflecting or outflow face
# in either dimension of its plane, and the tree finds the leaves across
# the seam, so a `δ` through it is ordinary here. Nor is a glued face
# (M17), for the same two reasons: the faces of a patch are its own, and
# the kind of each glued one is refused as anything but `OuterFace`, so a
# step through it keeps its component and finds the patch across.
function wall_direction(forest::Forest{D}, k::MortonKey{D},
                        δ::NTuple{D,Int}) where {D}
    n = 1 << level(k)
    rootpos = root_position(forest, k.root)
    roots = patch_roots(forest, patch(forest, k))
    walls = ntuple(D) do d
        δ[d] == 0 && return OuterFace
        forest.periodic[d] && return OuterFace
        stepped = Int(k.coords[d]) + δ[d]
        exits = δ[d] < 0 ? (stepped < 0 && rootpos[d] == 0) :
                           (stepped >= n && rootpos[d] == roots[d] - 1)
        return exits ? forest.faces[d][δ[d] < 0 ? 1 : 2] : OuterFace
    end
    return ntuple(d -> walls[d] == OuterFace ? δ[d] : 0, D), walls
end

# The index of the leaf that covers the node (root, lvl, coords): either
# that node itself, or its nearest coarser ancestor. `nothing` when the
# region is refined *below* `lvl`, so no single leaf covers it.
function find_covering_leaf(forest::Forest{D}, root::Integer, lvl::Integer,
                            coords::NTuple{D,Int}) where {D}
    c = coords
    for l in Int(lvl):-1:0
        i = find_leaf(forest, root, l, c)
        i !== nothing && return i
        c = map(x -> x >> 1, c)
    end
    return nothing
end

# Descend from the refined node (root, lvl, coords) into the children
# that touch its boundary in direction δ — both children along a
# tangential dimension (δ[d] == 0), only the near one along a normal
# dimension — recursing wherever a child is itself refined.
#
# Every node reached is known to exist and to be either a leaf or (by
# the all-or-nothing invariant) fully refined, so the recursion always
# terminates at leaves.
function collect_touching_leaves!(results::Vector{MortonKey{D}}, forest::Forest{D},
                                  root::Integer, lvl::Integer, coords::NTuple{D,Int},
                                  δ::NTuple{D,Int}) where {D}
    childlevel = lvl + 1
    base = ntuple(d -> 2 * coords[d] + (δ[d] == -1 ? 1 : 0), D)
    # Tangential dimensions span both children; normal ones are pinned.
    spans = ntuple(d -> δ[d] == 0 ? (0, 1) : (0,), D)
    for offset in Iterators.product(spans...)
        childcoords = ntuple(d -> base[d] + offset[d], D)
        j = find_leaf(forest, root, childlevel, childcoords)
        if j === nothing
            collect_touching_leaves!(results, forest, root, childlevel, childcoords, δ)
        else
            push!(results, forest.leaves[j])
        end
    end
    return results
end

"""
    neighbor_keys(forest, k::MortonKey, δ::NTuple) -> Vector{MortonKey}

The leaves abutting `k` across the face, edge, or corner in direction
`δ` (see [`alldirections`](@ref)) — that is, the leaves that supply the
data for that ghost region of `k`.

Valid on any forest, balanced or not. The result is

- empty, at a non-periodic domain boundary;
- one key, when the neighbor is at `k`'s level or coarser;
- otherwise every leaf touching the shared region, at whatever depth.

Under 2:1 balance (see [`balance!`](@ref)) the last case is exactly
`2^(D - count(!=(0), δ))` keys, all one level finer.

Across a rotating seam (M12; see [`Forest`](@ref)) the keys are the real
leaves whose image under the rotation abuts `k`, the leaves that supply
that ghost region turned by a quarter turn or two. A block at the axis
is then its own neighbor in three directions of the plane, as a single
periodic root is its own neighbor. Across a half-turn seam (M14) they
are the leaves on the same face, at the mirrored position along `d2`,
turned by a half turn.

Note that this is not symmetric under `δ -> -δ` when levels differ: a
coarse neighbor found across `k`'s *corner* also spans the face beyond
it, and reversing the direction from that larger block points elsewhere.
Adjacency is still mutually discoverable, just not necessarily across
the opposite direction.
"""
neighbor_keys(forest::Forest{D}, k::MortonKey{D}, δ::NTuple{D,Int}) where {D} =
    last(oriented_neighbors(forest, k, δ))

# `neighbor_keys` with the crossing (M12, M17): `(crossing, keys)`, where
# the keys are the real leaves and the crossing carries them to where `k`
# sees them in direction `δ` — `R^r` for `r = crossing.turns` across a
# seam, the glue's axis map across a glued face. All the neighbors of one
# block in one direction share one crossing, since a seam and a glued
# face lie on root boundaries at every level. Across a seam `turns` is
# the region's orientation even when `keys` is empty, which there means
# the real image lies beyond a high face; on a forest of patches an empty
# `keys` has no crossing, whether the region lies beyond an outer face or
# a patch corner (see `glued_anchor`). The search for finer leaves runs
# in the real frame, in the direction from them back to `k`'s image.
function oriented_neighbors(forest::Forest{D}, k::MortonKey{D},
                            δ::NTuple{D,Int}) where {D}
    δ != ntuple(_ -> 0, D) || throw(ArgumentError("direction must be nonzero"))
    all(d -> -1 <= δ[d] <= 1, 1:D) ||
        throw(ArgumentError("direction components must be in -1:1, got $δ"))
    anchor = neighbor_anchor(forest, k, δ)
    if anchor === nothing
        crossing = ispatched(forest) ? Crossing() :
                   Crossing(last(stepped_node(forest, k, δ)), 0)
        return crossing, MortonKey{D}[]
    end
    nbroot, nbcoords, crossing = anchor

    i = find_covering_leaf(forest, nbroot, level(k), nbcoords)
    i !== nothing && return crossing, [forest.leaves[i]]

    δreal = crossing.turns != 0 ?
            real_direction(δ, crossing.turns, seam_plane(forest)...) :
            crossing.glue != 0 ? real_direction(δ, glue_axes(forest, crossing)...) : δ
    return crossing, collect_touching_leaves!(MortonKey{D}[], forest, nbroot, level(k),
                                              nbcoords, δreal)
end

# Rebuild `forest.leaves` by walking it in order and replacing selected
# entries. Because a node's descendants occupy a contiguous run of the
# curve, splicing sorted children in place of their parent (or a parent
# in place of its run of children) preserves sortedness — no re-sort.
function rebuild_leaves!(forest::Forest{D}, newleaves::Vector{MortonKey{D}}) where {D}
    empty!(forest.leaves)
    append!(forest.leaves, newleaves)
    forest.state.generation += 1
    return forest
end

"""
    refine!(forest, keys)

Replace each leaf in `keys` — a single [`MortonKey`](@ref) or a
collection of them — by its `2^D` children. Errors if any key is not
currently a leaf, or is already at [`MAX_LEVEL`](@ref).

Does not restore 2:1 balance; follow with [`balance!`](@ref).
"""
function refine!(forest::Forest{D}, keys) where {D}
    ks = keys isa MortonKey{D} ? (keys,) : keys
    targets = Set{MortonKey{D}}()
    for k in ks
        isleaf(forest, k) || throw(ArgumentError("cannot refine $k: not a leaf of the forest"))
        level(k) < MAX_LEVEL || throw(ArgumentError("cannot refine $k: already at MAX_LEVEL"))
        push!(targets, k)
    end
    isempty(targets) && return forest

    newleaves = Vector{MortonKey{D}}()
    sizehint!(newleaves, length(forest.leaves) + length(targets) * (2^D - 1))
    for k in forest.leaves
        if k in targets
            append!(newleaves, sortedchildkeys(k))
        else
            push!(newleaves, k)
        end
    end
    return rebuild_leaves!(forest, newleaves)
end

"""
    coarsen!(forest, keys)

Replace the `2^D` children of each key in `keys` by that key itself.
Errors unless every child of every key is currently a leaf — coarsening
is all-or-nothing, matching refinement.

Does not restore 2:1 balance; follow with [`balance!`](@ref).
"""
function coarsen!(forest::Forest{D}, keys) where {D}
    ps = keys isa MortonKey{D} ? (keys,) : keys
    parentof = Dict{MortonKey{D},MortonKey{D}}()
    for p in ps
        for c in childkeys(p)
            isleaf(forest, c) ||
                throw(ArgumentError("cannot coarsen $p: child $c is not a leaf"))
            parentof[c] = p
        end
    end
    isempty(parentof) && return forest

    newleaves = Vector{MortonKey{D}}()
    sizehint!(newleaves, length(forest.leaves))
    emitted = Set{MortonKey{D}}()
    for k in forest.leaves
        p = get(parentof, k, nothing)
        if p === nothing
            push!(newleaves, k)
        elseif !(p in emitted)
            push!(newleaves, p)
            push!(emitted, p)
        end
    end
    return rebuild_leaves!(forest, newleaves)
end

"""
    balance!(forest)

Enforce 2:1 balance across every face, edge, and corner: refine any leaf
that is more than one level coarser than a leaf it touches, repeating
until the refinement stops rippling outward. On a forest with a rotating
seam (M12) it also enforces **conformity** there: a leaf across a seam
*face* that is coarser by any amount is refined, so the leaves on the low
face of `d1` and their images on the low face of `d2` end at one level —
or, across a half-turn seam (M14), their images on the same face. On a
forest of patches (M17) it enforces conformity at every glued face in
the same way, so no coarse-fine face crosses the glue; an edge or corner
across a glue keeps the 2:1 rule.

Afterwards every ghost region of every block touches at most one level
up or down, which is what bounds the ghost-filling cases and the
prolongation stencils.
"""
function balance!(forest::Forest{D}) where {D}
    dirs = alldirections(Val(D))
    # Conformity is kept across a seam and across a glued face (M17).
    seam = hasrotating(forest) || ispatched(forest)
    while true
        # The scan is threaded over leaves, each task collecting into its
        # own buffer; the buffers are concatenated in leaf order (M5).
        found = threaded_collect(MortonKey{D}, nleaves(forest)) do hits, b
            k = forest.leaves[b]
            # Only a leaf at level >= 2 can have a neighbor two or more
            # levels coarser than itself — but across a rotating seam's
            # face one level coarser is too coarse already, so there a
            # leaf at level 1 is asked too.
            level(k) >= 2 || (level(k) == 1 && seam) || return
            for δ in dirs
                anchor = neighbor_anchor(forest, k, δ)
                anchor === nothing && continue
                nbroot, nbcoords, crossing = anchor
                # Looking only for a *coarser* neighbor, so the covering
                # leaf suffices — no need to descend into finer ones,
                # which get checked from their own side.
                i = find_covering_leaf(forest, nbroot, level(k), nbcoords)
                i === nothing && continue
                nb = forest.leaves[i]
                level(nb) < level(k) - balance_slack(crossing, δ) && push!(hits, nb)
            end
        end

        isempty(found) && break
        toorefined = Set{MortonKey{D}}(found)
        refine!(forest, toorefined)
    end
    return forest
end

# How many levels two adjacent leaves may differ by: one, but none across
# a rotating seam's *face* (M12, conformity; CODE.md "Rotating seams"),
# nor across a glued face (M17), which no coarse-fine face may cross.
# An edge or corner across the seam keeps the ordinary 2:1 rule. A seam
# face is a direction with one nonzero component and a nonzero
# orientation: 1 or 3 for a quarter turn, whose orientation 2 needs two
# components and is a corner, and 2 for a half turn (M14), whose only
# face across the seam is the low face of `d1`, glued to itself. A glued
# face is likewise a direction with one nonzero component that crosses
# the glue; an edge across it keeps the 2:1 rule.
@inline balance_slack(c::Crossing, δ::NTuple{D,Int}) where {D} =
    iscrossing(c) && count(!=(0), δ) == 1 ? 0 : 1

"""
    isbalanced(forest)

Whether `forest` satisfies 2:1 balance across all faces, edges, and
corners, and conformity at a rotating seam (M12, M14) and at a glued face
(M17) — the postcondition of [`balance!`](@ref).
"""
function isbalanced(forest::Forest{D}) where {D}
    dirs = alldirections(Val(D))
    ok = fill(true, nleaves(forest))
    threaded_foreach(nleaves(forest)) do b
        k = forest.leaves[b]
        for δ in dirs
            crossing, nbrs = oriented_neighbors(forest, k, δ)
            for nb in nbrs
                if abs(level(nb) - level(k)) > balance_slack(crossing, δ)
                    ok[b] = false
                    return
                end
            end
        end
    end
    return all(ok)
end
