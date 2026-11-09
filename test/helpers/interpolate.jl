# The definitions of `interpolate_tests.jl`, split off so that each test file runs
# on its own: every test process includes all of `helpers/` ("Parallel
# groups" in CODE.md).

using TreeAMR: query_stencil, stencil_hits, lagrange_weights, PointGeometry

# A tensor polynomial of degree `deg` in each dimension, per variable,
# with its gradient and its derivative for any multi-index `m`:
# `Π_d p_{v,d}(x_d)`. Coefficients of order one on the domain `[-1, 1]^D`,
# so an absolute tolerance means something.
function tensorpoly(D, deg)
    c(v, d, e) = (0.3 + 0.1v + 0.05d) * (-1)^e / (1 + e)
    # The `k`-th derivative of one factor, `Σ e!/(e−k)! c yᵉ⁻ᵏ`.
    p(v, d, y, k) = sum(prod((e - k + 1):e; init=1) * c(v, d, e) * y^(e - k)
                        for e in k:deg; init=0.0)
    f(x, v) = prod(p(v, d, x[d], 0) for d in 1:D)
    df(x, v, a) = prod(p(v, d, x[d], Int(d == a)) for d in 1:D)
    dm(x, v, m) = prod(p(v, d, x[d], m[d]) for d in 1:D)
    return f, df, dm
end

# Two roots per dimension over `[-1, 1]^D`, a block refined twice at the
# low corner and once more around it, so that queries meet coarse-fine
# faces at three levels.
function interp_forest(D; N=8, periodic=ntuple(_ -> false, D))
    forest = Forest(ntuple(_ -> 2, D); N=N, periodic=periodic,
                    extents=ntuple(_ -> (-1.0, 1.0), D))
    refine_where!(forest, (c, lvl) -> (lvl == 0 && all(<(0), c)) ||
                                      (lvl == 1 && all(<(-0.5), c)), 2)
    return forest
end

# A field set of `f` with ghosts filled exactly for polynomials of degree
# `< p`: point-value operators of even order `p`, the hook from `f`.
function interp_fieldset(forest::Forest{D}, f, C, p; G=ghosts_for(C, p),
                         nvars=2) where {D}
    fs = FieldSet(forest, nvars; G=G, centering=C)
    fill_by_coordinates!(f, fs)
    ops = Operators(prolongation=p, restriction=p)
    fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=boundary_by_coordinates(f))
    return fs
end

# Query points that probe the edge cases: random, every leaf's lower
# corner (a point on a block face and on a node of every centering's
# lattice), the domain's upper corner, and owned stored points.
function probe_points(rng, fs::FieldSet{T,D}; nrandom=60) where {T,D}
    forest = fs.forest
    lo = ntuple(d -> forest.extents[d][1], D)
    hi = ntuple(d -> forest.extents[d][2], D)
    xs = NTuple{D,Float64}[ntuple(d -> lo[d] + (hi[d] - lo[d]) * rand(rng), D)
                           for _ in 1:nrandom]
    for k in forest.leaves
        push!(xs, ntuple(d -> block_extent(forest, k)[d][1], D))
    end
    push!(xs, hi, lo)
    for b in 1:nblocks(fs)
        push!(xs, coordinates(fs, b, ntuple(d -> fs.G[d] + 1 + (b + d) % forest.N, D)))
    end
    return xs
end

gradient_derivs(D) = (ntuple(_ -> 0, D), ntuple(a -> ntuple(d -> Int(d == a), D), D)...)

# Every multi-index of total order two — the Hessian — leaving out the
# pure ones where `Lagrange(n)` cannot produce them (`n < 3`).
hessian_derivs(D, n=3) =
    Tuple(m for m in (ntuple(d -> Int(d == a) + Int(d == b), D) for a in 1:D for b in a:D)
          if n >= 3 || maximum(m) == 1)

# The largest error of `values` against `dm` per derivative order: the
# value, the first derivatives, the second.
function errors_by_order(values, xs, derivs, dm; nvars=2)
    worst = zeros(3)
    for (j, x) in enumerate(xs), v in 1:nvars, (k, m) in enumerate(derivs)
        o = sum(m) + 1
        worst[o] = max(worst[o], abs(values[v, k, j] - dm(x, v, m)))
    end
    return worst
end

# `TREEAMR_SHOW_ERRORS=1` prints those errors, which is how the bounds
# below were set.
showerrors() = get(ENV, "TREEAMR_SHOW_ERRORS", "") == "1"

# Per dimension, a factor of definite parity about the reflecting wall
# (even for variable 1, odd for variable 2), a general cubic at outer
# faces, a constant where periodic — the only polynomials those faces
# admit — with its first and second derivatives. Its natural continuation
# beyond the wall *is* the mirror image, so the analytic polynomial is the
# oracle there.
function parity_factor(kind, v, d, y, wall)
    s = y - wall
    (kind === :periodic || kind === :reflect_both) && return (1 + d / 10, 0.0, 0.0)
    kind === :outer && return (0.3 + 0.2y + 0.1y^2 + 0.05y^3, 0.2 + 0.2y + 0.15y^2,
                               0.2 + 0.3y)
    return v == 1 ? (0.5 + 0.3s^2, 0.6s, 0.6) : (0.7s + 0.2s^3, 0.7 + 0.6s^2, 1.2s)
end

# The `m`-th derivative of every Lagrange basis polynomial on `nodes` at
# `ξ`, exactly: each `ℓₖ` expanded into its coefficients, differentiated
# `m` times and evaluated — independent of the package's truncated
# series.
function lagrange_derivative_weights(nodes, ξ, m)
    return map(eachindex(nodes)) do k
        c = [one(ξ)]                         # lowest order first
        for j in eachindex(nodes)
            j == k && continue
            c = ([zero(ξ); c] .- nodes[j] .* [c; zero(ξ)]) ./ (nodes[k] - nodes[j])
        end
        for _ in 1:m
            c = [i * c[i + 1] for i in 1:(length(c) - 1)]
        end
        sum(c[i] * ξ^(i - 1) for i in eachindex(c); init=zero(ξ))
    end
end

# Whether the split path — `locate` in the field set's type, then
# `interpolate` at the located points — gives the fused path's values and
# flags bit for bit (M19). `isequal`, so that the same NaN agrees too.
function split_matches(fs, xs, basis; kwargs...)
    a = interpolate(fs, xs, basis; kwargs...)
    b = interpolate(fs, locate(eltype(fs.work), fs.forest, xs), basis; kwargs...)
    return isequal(a.values, b.values) && a.excluded == b.excluded && all(b.inside)
end

# --- test atlases (M19) --------------------------------------------------------
#
# Each is a map of a brick's or a patch's logical coordinates `ξ` to
# physical ones `x`, with its inverse in closed form for `chart_locate`,
# and an independent `inverse_derivs(atlas, x)`: `ξ`, `K[a][i] = ∂ξᵃ/∂xⁱ`
# and `S[a][i][j] = ∂²ξᵃ/∂xⁱ∂xʲ`, differentiated by hand from the inverse,
# which is what the tests compare the package's chain rule against.

# `x = A ξ + c`.
struct AffineAtlas{D}
    A::NTuple{D,NTuple{D,Float64}}
    Ainv::NTuple{D,NTuple{D,Float64}}
    c::NTuple{D,Float64}
end
function AffineAtlas(A::Matrix, c)
    D = length(c)
    rows(M) = ntuple(i -> ntuple(k -> Float64(M[i, k]), D), D)
    return AffineAtlas{D}(rows(A), rows(inv(A)), Tuple(Float64.(c)))
end
matvec(M::NTuple{D}, y) where {D} =
    ntuple(i -> (s = 0.0; for k in 1:D; s += M[i][k] * y[k]; end; s), D)
TreeAMR.chart_map(a::AffineAtlas{D}, ξ, p) where {D} =
    ntuple(i -> a.c[i] + matvec(a.A, ξ)[i], D)
TreeAMR.chart_locate(a::AffineAtlas{D}, x) where {D} =
    (matvec(a.Ainv, ntuple(i -> x[i] - a.c[i], D)), 1)
TreeAMR.chart_jacobian(a::AffineAtlas, ξ, p) = a.A
TreeAMR.chart_hessian(a::AffineAtlas{D}, ξ, p) where {D} =
    ntuple(_ -> ntuple(_ -> ntuple(_ -> 0.0, D), D), D)
inverse_derivs(a::AffineAtlas{D}, x) where {D} =
    (TreeAMR.chart_locate(a, x)[1], a.Ainv, ntuple(_ -> ntuple(_ -> ntuple(_ -> 0.0, D), D), D))

# `xᵢ = sinh(ξᵢ)`, separable and nonlinear.
struct SinhAtlas{D} end
TreeAMR.chart_map(::SinhAtlas{D}, ξ, p) where {D} = ntuple(i -> sinh(ξ[i]), D)
TreeAMR.chart_locate(::SinhAtlas{D}, x) where {D} = (ntuple(i -> asinh(x[i]), D), 1)
TreeAMR.chart_jacobian(::SinhAtlas{D}, ξ, p) where {D} =
    ntuple(i -> ntuple(a -> a == i ? cosh(ξ[i]) : 0.0, D), D)
TreeAMR.chart_hessian(::SinhAtlas{D}, ξ, p) where {D} =
    ntuple(i -> ntuple(a -> ntuple(b -> a == i && b == i ? sinh(ξ[i]) : 0.0, D), D), D)
inverse_derivs(::SinhAtlas{D}, x) where {D} =
    (ntuple(i -> asinh(x[i]), D),
     ntuple(a -> ntuple(i -> a == i ? 1 / sqrt(1 + x[i]^2) : 0.0, D), D),
     ntuple(a -> ntuple(i -> ntuple(j -> a == i == j ? -x[i] / (1 + x[i]^2)^1.5 : 0.0,
                                    D), D), D))

# `x₁ = ξ₁, x₂ = ξ₂ + c ξ₁²` in 2D, a shear that mixes the dimensions.
struct ShearAtlas
    c::Float64
end
TreeAMR.chart_map(a::ShearAtlas, ξ, p) = (ξ[1], ξ[2] + a.c * ξ[1]^2)
TreeAMR.chart_locate(a::ShearAtlas, x) = ((x[1], x[2] - a.c * x[1]^2), 1)
TreeAMR.chart_jacobian(a::ShearAtlas, ξ, p) = ((1.0, 0.0), (2a.c * ξ[1], 1.0))
TreeAMR.chart_hessian(a::ShearAtlas, ξ, p) = (((0.0, 0.0), (0.0, 0.0)),
                                              ((2a.c, 0.0), (0.0, 0.0)))
inverse_derivs(a::ShearAtlas, x) =
    (TreeAMR.chart_locate(a, x)[1], ((1.0, 0.0), (-2a.c * x[1], 1.0)),
     (((0.0, 0.0), (0.0, 0.0)), ((-2a.c, 0.0), (0.0, 0.0))))

# The physical derivative `m` of `u = g ∘ ξ(x)` at `x`, from the exact
# logical derivatives `dm(ξ, v, ·)` of `g` and the inverse's derivatives.
function physical_oracle(atlas, dm, x, v, m)
    D = length(x)
    ξ, K, S = inverse_derivs(atlas, x)
    e(a) = ntuple(d -> Int(d == a), D)
    o = sum(m)
    o == 0 && return dm(ξ, v, m)
    if o == 1
        i = findfirst(==(1), m)
        return sum(dm(ξ, v, e(a)) * K[a][i] for a in 1:D)
    end
    i = findfirst(>(0), m)
    j = m[i] == 2 ? i : findlast(>(0), m)
    return sum(dm(ξ, v, e(a) .+ e(b)) * K[a][i] * K[b][j] for a in 1:D, b in 1:D) +
           sum(dm(ξ, v, e(a)) * S[a][i][j] for a in 1:D)
end

# The identity map, through every method, so that the chain rule
# multiplies by exact ones and zeros.
struct IdentityAtlas{D} end
TreeAMR.chart_locate(::IdentityAtlas, x) = (x, 1)
TreeAMR.chart_map(::IdentityAtlas, ξ, p) = ξ
TreeAMR.chart_jacobian(::IdentityAtlas{D}, ξ, p) where {D} =
    ntuple(i -> ntuple(a -> a == i ? 1.0 : 0.0, D), D)
TreeAMR.chart_hessian(::IdentityAtlas{D}, ξ, p) where {D} =
    ntuple(_ -> ntuple(_ -> ntuple(_ -> 0.0, D), D), D)

# A polar atlas on the four-patch ring: patch `p` covers the angles
# `[(p − 1)π/2, pπ/2]`, its `ξ₁ ∈ [0, 2]`, and `ξ₂ ∈ [0, 1]` is `r − 1`. A
# point in the hole or beyond the rim is in no patch.
struct PolarAtlas end
function TreeAMR.chart_locate(::PolarAtlas, x)
    r = hypot(x[1], x[2])
    1 <= r <= 2 || return nothing
    s = 4 * mod(atan(x[2], x[1]), 2π) / π
    p = clamp(floor(Int, s / 2) + 1, 1, 4)
    return ((clamp(s - 2(p - 1), 0.0, 2.0), r - 1), p)
end
function TreeAMR.chart_map(::PolarAtlas, ξ, p)
    θ = π / 4 * (2(p - 1) + ξ[1])
    return ((1 + ξ[2]) * cos(θ), (1 + ξ[2]) * sin(θ))
end
function TreeAMR.chart_jacobian(::PolarAtlas, ξ, p)
    θ = π / 4 * (2(p - 1) + ξ[1])
    r = 1 + ξ[2]
    return ((-r * sin(θ) * π / 4, cos(θ)), (r * cos(θ) * π / 4, sin(θ)))
end

# An atlas that returns a position outside its patch's box.
struct OverreachingAtlas end
TreeAMR.chart_locate(::OverreachingAtlas, x) = ((x[1] + 3, x[2]), 1)
