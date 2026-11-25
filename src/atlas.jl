# The application's atlas (M19): the maps between physical space and the
# logical coordinates of a forest's patches, through which `locate` takes
# physical positions and `interpolate` returns physical derivatives.
#
# TreeAMR carries no maps ("A multi-patch forest" in CODE.md); an atlas is
# any value the application defines these four methods for, recognised by
# its methods alone, with no supertype (Erik's choice, "Point
# interpolation" in HISTORY.md). It folds the global topology itself —
# wrapping an angle, choosing a ball's patch — and returns a `ξ` inside
# its patch's closed box; TreeAMR follows no glue. The methods run inside
# kernels, so an atlas must be `isbits` and its methods must compile on
# the backend. See "Interpolation on element sets and patches" in CODE.md.

"""
    TreeAMR.chart_locate(atlas, x) -> (ξ, p) or nothing

The logical position `ξ::NTuple{D}` in patch `p::Int` of the physical
point `x::NTuple{D,T}`, or `nothing` when `x` lies in no patch — in an
annulus's hole, beyond a ball's surface. The application defines it for
its atlas; [`locate`](@ref) calls it for every point.

It folds the domain's global topology itself, wrapping an angle or
choosing which of a ball's patches a point is in, and returns a `ξ` in
patch `p`'s closed box: a `ξ` outside it, but for one past a reflecting
face, or past a symmetry's mirror face or turned glue of a patch (M21),
is refused as a broken contract, naming the atlas.

On a forest with a symmetry of its own patches the atlas may instead
fold a point itself, returning `(ξ, p, (mirrors, turn))`: the position
of the point's image, and the symmetry that carries the image back to
the point — `mirrors` an `NTuple{D,Bool}` by mirror label, `turn` the
quarter turns of the forest's turn. TreeAMR then applies that symmetry's
variable map, sign and derivative transform, as for a fold of its own.
An atlas returns one shape for every point, `(0, (false, …))` where it
folds nothing, so that a kernel sees one type. On a brick,
`p` is 1 and the brick's own folds apply to `ξ` as to a position. Inverting
the map is the atlas's: TreeAMR offers no Newton iteration.
"""
function chart_locate end

# The symmetry an atlas reports with a located point (M21), as a fold's
# mask of mirror labels and turns: none for `(ξ, p)`.
@inline located_symmetry(c::Tuple{Any,Any}) = 0, 0
@inline function located_symmetry(c::Tuple{Any,Any,Any})
    mirrors, turn = c[3]
    mask = 0
    for k in 1:length(mirrors)
        mirrors[k] && (mask |= 1 << (k - 1))
    end
    return mask, Int(turn)
end

"""
    TreeAMR.chart_map(atlas, ξ, p) -> x

The physical position of logical position `ξ` in patch `p`: the inverse
of [`chart_locate`](@ref TreeAMR.chart_locate). Needed for an `exclude`
region with an atlas, which is physical, and for [`positions`](@ref).
"""
function chart_map end

"""
    TreeAMR.chart_jacobian(atlas, ξ, p) -> J

The Jacobian of [`chart_map`](@ref TreeAMR.chart_map) at `ξ` in patch
`p`, as an `NTuple{D}` of `NTuple{D}`s with `J[i][a] = ∂xⁱ/∂ξᵃ`. Needed
for a first or second derivative with an atlas, which
[`interpolate`](@ref) returns in physical units through the chain rule.
"""
function chart_jacobian end

"""
    TreeAMR.chart_hessian(atlas, ξ, p) -> H

The second derivatives of [`chart_map`](@ref TreeAMR.chart_map) at `ξ` in
patch `p`, `H[i][a][b] = ∂²xⁱ/∂ξᵃ∂ξᵇ`. Needed only for a second
derivative with an atlas.
"""
function chart_hessian end

"""
    multi_indices(Val(D), maxorder) -> NTuple

Every derivative multi-index in `D` dimensions up to total order
`maxorder ≤ 2`, in a fixed order: the value, then the first derivatives
by dimension, then the second derivatives `∂_a∂_b` with `a ≤ b` in
lexicographic order. `derivs = multi_indices(Val(3), 1)` asks
[`interpolate`](@ref) for the value with the gradient.
"""
function multi_indices(::Val{D}, maxorder::Integer) where {D}
    0 <= maxorder <= 2 || throw(ArgumentError(
        "multi_indices goes up to the second order, the highest interpolate " *
        "implements, got maxorder = $maxorder"))
    z = ntuple(_ -> 0, Val(D))
    o1 = ntuple(a -> ntuple(d -> Int(d == a), Val(D)), Val(D))
    o2 = Tuple(ntuple(d -> Int(d == a) + Int(d == b), Val(D)) for a in 1:D for b in a:D)
    return maxorder == 0 ? (z,) : maxorder == 1 ? (z, o1...) : (z, o1..., o2...)
end

# The position of `∂_a∂_b`, `a ≤ b`, among the second derivatives of
# `multi_indices`.
@inline pairindex(a::Int, b::Int, D::Int) = (a - 1) * D - ((a - 1) * (a - 2)) ÷ 2 + (b - a + 1)

# --- the host's checks --------------------------------------------------------

# The atlas has `f` for a position in `T` and a patch, or the call that
# needs it is refused, naming it — rather than failing inside a kernel's
# compilation.
function check_chart(f, atlas, ::Type{T}, ::Val{D}, why) where {T,D}
    args = f === chart_locate ? Tuple{typeof(atlas),NTuple{D,T}} :
           Tuple{typeof(atlas),NTuple{D,T},Int}
    hasmethod(f, args) && return nothing
    sig = f === chart_locate ? "(::$(typeof(atlas)), ::NTuple{$D,$T})" :
          "(::$(typeof(atlas)), ::NTuple{$D,$T}, ::Int)"
    throw(ArgumentError(
        "the atlas has no method TreeAMR.$(nameof(f))$sig, which $why needs. An atlas " *
        "is recognised by its methods alone: define it for the atlas's type, with the " *
        "position before the patch (CODE.md, \"Interpolation on element sets and " *
        "patches\")."))
end

# Every method `interpolate` with an atlas needs for these derivatives and
# this region.
function check_atlas_interpolation(atlas, ::Type{T}, ::Val{D}, ms, exclude) where {T,D}
    order = maximum(sum, ms)
    order >= 1 && check_chart(chart_jacobian, atlas, T, Val(D),
                              "a first or second derivative with an atlas")
    order >= 2 && check_chart(chart_hessian, atlas, T, Val(D),
                              "a second derivative with an atlas")
    exclude === nothing || check_chart(chart_map, atlas, T, Val(D),
                                       "an exclude region with an atlas, which is physical,")
    return nothing
end

# The refusal of point `j`, `x`, whose `chart_locate` result `(ξ, p)` is
# not a point of the forest: a patch it does not have, or a `ξ` outside
# the patch's closed box.
function broken_chart(forest::Forest, atlas, x, ξ, p, j)
    where = !(1 <= p <= npatches(forest)) ?
            "patch $p, but the forest has " *
            (ispatched(forest) ? "patches 1:$(npatches(forest))" : "one brick, patch 1") :
            "a position outside the closed box $(patch_extents(forest, p)) of patch $p"
    return ArgumentError(
        "the atlas $(typeof(atlas)) broke its contract at point $j, $(Tuple(x)): " *
        "chart_locate returned (ξ, p) = ($ξ, $p), $where. An atlas folds the global " *
        "topology itself and returns a ξ inside its patch's box (clamping it there is " *
        "one line); TreeAMR follows no glue, and mirrors only at a reflecting face.")
end

# The refusal of point `j`, `x`, for which `chart_locate` returned nothing.
atlas_outside(atlas, x, j) = ArgumentError(
    "point $j, $(Tuple(x)), is in no patch: the atlas $(typeof(atlas))'s chart_locate " *
    "returned nothing for it, so no block holds data there. Pass outside = :mark to " *
    "keep such points, marked and filled.")

# --- the excluded region through the maps --------------------------------------

# An `exclude` region with an atlas is physical: every stencil point is
# mapped through `chart_map` and tested, `nᴰ` evaluations per query,
# since a curved chart breaks the ellipsoid's separable test.
struct MappedRegion{R,A}
    region::R
    atlas::A
    p::Int
end

@inline mapped_region(::Nothing, atlas, p) = nothing
@inline mapped_region(region, ::Nothing, p) = region
@inline mapped_region(region, atlas, p) = MappedRegion(region, atlas, Int(p))
@inline mapped_region(::Nothing, ::Nothing, p) = nothing

@inline function stencil_hits(m::MappedRegion, place, ::Val{n}) where {n}
    D = length(place.origin)
    for J in CartesianIndices(ntuple(_ -> n, Val(D)))
        ξ = ntuple(d -> stencil_position(place, d, J[d] - 1), Val(D))
        inside(m.region, chart_map(m.atlas, ξ, m.p)) && return true
    end
    return false
end

# --- the chain rule ------------------------------------------------------------

# `K = J⁻¹`, `K[a][i] = ∂ξᵃ/∂xⁱ`, in closed form in `T`, from `J[i][a] =
# ∂xⁱ/∂ξᵃ`.
@inline inv_jacobian(J::NTuple{1}) = ((inv(J[1][1]),),)
@inline function inv_jacobian(J::NTuple{2})
    det = J[1][1] * J[2][2] - J[1][2] * J[2][1]
    return ((J[2][2] / det, -J[1][2] / det), (-J[2][1] / det, J[1][1] / det))
end
@inline function inv_jacobian(J::NTuple{3})
    m11, m12, m13 = J[1]
    m21, m22, m23 = J[2]
    m31, m32, m33 = J[3]
    c11 = m22 * m33 - m23 * m32
    c12 = m23 * m31 - m21 * m33
    c13 = m21 * m32 - m22 * m31
    det = m11 * c11 + m12 * c12 + m13 * c13
    return ((c11 / det, (m13 * m32 - m12 * m33) / det, (m12 * m23 - m13 * m22) / det),
            (c12 / det, (m11 * m33 - m13 * m31) / det, (m13 * m21 - m11 * m23) / det),
            (c13 / det, (m12 * m31 - m11 * m32) / det, (m11 * m22 - m12 * m21) / det))
end

# The dimensions a multi-index of total order two differentiates along,
# `(i, j)` with `i ≤ j`.
@inline function second_dims(m::NTuple{D,Int}) where {D}
    i, j = 0, 0
    for d in 1:D
        if m[d] == 2
            i, j = d, d
        elseif m[d] == 1
            i == 0 ? (i = d) : (j = d)
        end
    end
    return i, j
end

# Physical derivative `m` of variable `iv` at point `j`, from the logical
# derivatives of `multi_indices(Val(D), order)` in `logical`:
#
#   ∂ᵢu    = Kᵃᵢ ∂ₐũ
#   ∂ᵢ∂ⱼu  = Kᵃᵢ Kᵇⱼ ∂ₐ∂ᵦũ − (Kᵃₖ Hᵏ_bc Kᵇᵢ Kᶜⱼ) ∂ₐũ
@inline function physical_derivative(logical, iv, j, m::NTuple{D,Int}, K, H) where {D}
    T = eltype(logical)
    o = sum(m)
    o == 0 && return logical[iv, 1, j]
    if o == 1
        i = 0
        for d in 1:D
            m[d] == 1 && (i = d)
        end
        s = zero(T)
        for a in 1:D
            s += K[a][i] * logical[iv, 1 + a, j]
        end
        return s
    end
    i, jj = second_dims(m)
    s = zero(T)
    for a in 1:D, b in 1:D
        l2 = logical[iv, 1 + D + pairindex(min(a, b), max(a, b), D), j]
        s += K[a][i] * K[b][jj] * l2
    end
    for a in 1:D
        c = zero(T)
        for k in 1:D, b in 1:D, cc in 1:D
            c += K[a][k] * H[k][b][cc] * K[b][i] * K[cc][jj]
        end
        s -= c * logical[iv, 1 + a, j]
    end
    return s
end

@inline chart_K(atlas, ξ, p, ::Val{O}) where {O} =
    O >= 1 ? inv_jacobian(chart_jacobian(atlas, ξ, p)) : nothing
@inline chart_H(atlas, ξ, p, ::Val{O}) where {O} =
    O >= 2 ? chart_hessian(atlas, ξ, p) : nothing

# The requested physical derivatives of point `j`, or `fill` for a point
# outside.
@inline function chain_point!(values, logical, ξs, patches, inside, atlas, ::Val{MS},
                              ::Val{O}, fillvalue, j) where {MS,O}
    nv = size(values, 1)
    if !inside[j]
        for k in 1:length(MS), iv in 1:nv
            values[iv, k, j] = fillvalue
        end
        return nothing
    end
    ξ = ξs[j]
    p = Int(patches[j])
    K = chart_K(atlas, ξ, p, Val(O))
    H = chart_H(atlas, ξ, p, Val(O))
    for k in 1:length(MS), iv in 1:nv
        values[iv, k, j] = physical_derivative(logical, iv, j, MS[k], K, H)
    end
    return nothing
end

@kernel function chain_kernel!(values, @Const(logical), @Const(ξs), @Const(patches),
                               @Const(inside), atlas, ::Val{MS}, ::Val{O},
                               fillvalue) where {MS,O}
    j = @index(Global, Linear)
    chain_point!(values, logical, ξs, patches, inside, atlas, Val(MS), Val(O), fillvalue,
                 j)
end

# The chain rule over a batch, on the backend of `values`: the logical
# derivatives `logical` of `multi_indices(Val(D), order)` mapped to the
# physical ones `ms` at each point's `ξ` in its patch.
function launch_chain!(values, logical, pts::LocatedPoints, atlas, ms, fillvalue)
    npts = length(pts)
    npts == 0 && return values
    backend = get_backend(values)
    O = maximum(sum, ms)
    forest = pts.forest
    if symmetric_patches(forest)
        # At the point the fold reached (M21), in its patch, and carried
        # back by the symmetry.
        turn = patch_turn(forest)
        plane = turn === nothing ? (0, 0) : turn[1]
        pointkernel(chain_folded_kernel!, backend, npts)(
            values, logical, pts.y, folded_patches(forest, pts), pts.code, pts.inside,
            atlas, plane, Val(ms), Val(O), fillvalue; ndrange=npts)
    else
        pointkernel(chain_kernel!, backend, npts)(values, logical, pts.ξ, pts.patch,
                                                  pts.inside, atlas, Val(ms), Val(O),
                                                  fillvalue; ndrange=npts)
    end
    synchronize(backend)
    return values
end

# The physical symmetry of a fold (M21), as a signed permutation of the
# axes: with `x = M x′` the point and the image it was folded to, `M` the
# mirrors of the labels in `mask` then `turn` quarter turns in the plane
# `(k1, k2)` (`e_k1 → e_k2`, `e_k2 → -e_k1`), `∂ᵢu(x) = sᵢ ∂′_{π(i)}` of
# the turned variables at `x′`. Returns `(π, s)`.
@inline function fold_axes(mask::Int, turn::Int, plane::NTuple{2,Int}, ::Val{D}) where {D}
    k1, k2 = plane
    # `M e_j = σ_j e_{τ(j)}`.
    function column(j)
        i, σ = j, (mask >> (j - 1)) & 1 == 1 ? -1 : 1
        for _ in 1:turn
            if i == k1
                i = k2
            elseif i == k2
                i, σ = k1, -σ
            end
        end
        return i, σ
    end
    cols = ntuple(column, Val(D))
    π = ntuple(i -> something(findfirst(c -> c[1] == i, cols), i), Val(D))
    s = ntuple(i -> cols[π[i]][2], Val(D))
    return π, s
end

# The requested physical derivatives of point `j` of a symmetric forest
# (M21): at its folded position `y` in the patch `patches[j]` it landed in,
# each multi-index carried back by the fold's signed permutation.
@inline function chain_folded_point!(values, logical, ys, patches, codes, inside, atlas,
                                     plane, ::Val{MS}, ::Val{O}, fillvalue,
                                     j) where {MS,O}
    nv = size(values, 1)
    if !inside[j]
        for k in 1:length(MS), iv in 1:nv
            values[iv, k, j] = fillvalue
        end
        return nothing
    end
    y = ys[j]
    D = length(y)
    q = Int(patches[j])
    # The carried-back multi-index is known at run time only, so every order
    # is compiled: a lower `O` gets a zero Hessian and, at 0, the identity,
    # which its derivatives never read.
    T = eltype(values)
    K = O >= 1 ? chart_K(atlas, y, q, Val(O)) :
        ntuple(a -> ntuple(i -> T(a == i), Val(D)), Val(D))
    H = O >= 2 ? chart_H(atlas, y, q, Val(O)) :
        ntuple(_ -> ntuple(_ -> ntuple(_ -> zero(T), Val(D)), Val(D)), Val(D))
    c = Int(codes[j])
    π, s = fold_axes(c & ((1 << D) - 1), c >> D, plane, Val(D))
    for k in 1:length(MS)
        m = MS[k]
        m′ = ntuple(e -> sum(ntuple(i -> π[i] == e ? m[i] : 0, Val(D))), Val(D))
        σ = prod(ntuple(i -> isodd(m[i]) ? s[i] : 1, Val(D)))
        for iv in 1:nv
            values[iv, k, j] = σ * physical_derivative(logical, iv, j, m′, K, H)
        end
    end
    return nothing
end

@kernel function chain_folded_kernel!(values, @Const(logical), @Const(ys), @Const(patches),
                                      @Const(codes), @Const(inside), atlas, plane,
                                      ::Val{MS}, ::Val{O}, fillvalue) where {MS,O}
    j = @index(Global, Linear)
    chain_folded_point!(values, logical, ys, patches, codes, inside, atlas, plane,
                        Val(MS), Val(O), fillvalue, j)
end
