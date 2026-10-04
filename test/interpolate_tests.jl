# Point interpolation.
#
# `interpolate` answers "what is the field here" for points the mesh did
# not choose, so every claim is stated against something the package did
# not compute: exact rational leaf boxes for the location, analytic
# polynomials and their derivatives for the values, exact rational
# Lagrange weights over the block's own stored array for the stencil, and
# brute-force enumeration for the excluded-region test.

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

@testset "locate_point finds the leaf whose half-open box holds the point: D=$D" for
        D in (1, 2, 3)
    # A wrong covering leaf is a wrong value everywhere downstream, and
    # the single search depends on the curve order putting an ancestor
    # just before its subtree. The oracle is the exact rational leaf box:
    # the unique leaf with lo ≤ x < hi, the upper domain face going to
    # the last leaf there.
    rng = Xoshiro(20260925 + D)
    for trial in 1:8
        forest = random_forest(rng, Val(D); nsteps=12 * D, maxlvl=3)
        balance!(forest)
        roots = forest.roots
        function oracle(x)
            # A periodic dimension's upper face *is* its lower face.
            x = ntuple(d -> forest.periodic[d] && x[d] == roots[d] ? 0.0 : x[d], D)
            hits = findall(forest.leaves) do k
                lo, hi = leafbox(forest, k)
                all(d -> lo[d] <= x[d] < hi[d] || (x[d] == roots[d] == hi[d]), 1:D)
            end
            return only(hits)
        end
        # Random points, and points on the lattice of the finest level,
        # which sit exactly on leaf faces and corners.
        L = maxlevel(forest)
        pts = [ntuple(d -> roots[d] * rand(rng), D) for _ in 1:40]
        append!(pts, [ntuple(d -> rand(rng, 0:(roots[d] << L)) / (1 << L), D)
                      for _ in 1:40])
        for x in pts
            @test locate_point(forest, x) == oracle(x)
        end
        # Outside a non-periodic dimension there is nothing; a periodic
        # one wraps.
        for d in 1:D
            x = ntuple(e -> e == d ? roots[d] + 0.25 : 0.5, D)
            if forest.periodic[d]
                @test locate_point(forest, x) == oracle(ntuple(e -> e == d ? 0.25 : 0.5, D))
            else
                @test locate_point(forest, x) === nothing
            end
        end
    end
    @test_throws "has 2 coordinates" locate_point(Forest((1, 1); N=4), (0.5,))
end

@testset "Interpolation is exact on polynomials of degree n − 1: D=$D, n=$n, C=$C" for
        (D, ns, Cs) in ((1, 2:6, (cellcentered(1), vertexcentered(1))),
                        (2, 2:6, (cellcentered(2), vertexcentered(2), facecentered(2, 1))),
                        (3, (3, 4), (cellcentered(3), vertexcentered(3)))),
        n in ns, C in Cs
    # The claim that makes the interpolant usable at all: through a
    # three-level mesh, at nodes, on block faces, at the domain's corners,
    # every value, first and second derivative of a degree-(n−1) tensor
    # polynomial is reproduced to roundoff. It holds only if the stencil
    # is inside one block's stored array, the weights are right, and the
    # ghosts the stencil reads are the exchange's. A second derivative
    # divides the roundoff by h², hence its looser bound.
    f, _, dm = tensorpoly(D, n - 1)
    p = n + isodd(n)                         # point-value orders are even
    forest = interp_forest(D; N=D == 3 ? 6 : 8)
    fs = interp_fieldset(forest, f, C, p)
    xs = probe_points(Xoshiro(n + 10D), fs)
    derivs = (gradient_derivs(D)..., hessian_derivs(D, n)...)
    r = interpolate(fs, xs, Lagrange(n); derivs=derivs)
    @test size(r.values) == (2, length(derivs), length(xs))
    @test !any(r.excluded)
    worst = errors_by_order(r.values, xs, derivs, dm)
    showerrors() && @info "exactness" D n C worst = Tuple(worst)
    @test worst[1] < 1e-11 && worst[2] < 1e-11
    @test worst[3] < 1e-9
end

@testset "A stencil shifts inward and stays exact where ghosts are narrow: C=$C" for
        C in (cellcentered(2), vertexcentered(2), facecentered(2, 2))
    # With `G = 0` (a flux) or `G` below `n/2`, the centered stencil would
    # leave the stored array; clamping moves it inward instead of
    # extrapolating or reading another block, and polynomials stay exact.
    f, _, dm = tensorpoly(2, 3)
    forest = interp_forest(2)
    derivs = (gradient_derivs(2)..., hessian_derivs(2)...)
    for G in (0, 1)
        fs = FieldSet(forest, 2; G=G, centering=C)
        # Every stored point from the polynomial itself: no exchange of
        # order 4 fits in ghosts this narrow, and the claim is about the
        # stencil, not about the exchange.
        for b in 1:nblocks(fs), v in 1:2, idx in CartesianIndices(size(fs.work)[1:2])
            fs.work[idx, v, b] = f(coordinates(fs, b, Tuple(idx)), v)
        end
        xs = probe_points(Xoshiro(7 + G), fs)
        r = interpolate(fs, xs, Lagrange(4); derivs=derivs)
        worst = errors_by_order(r.values, xs, derivs, dm)
        showerrors() && @info "shifted" C G worst = Tuple(worst)
        @test worst[1] < 1e-11
        @test worst[2] < 1e-10
        @test worst[3] < 1e-9
    end
end

@testset "Interpolation of order n is not exact at degree n, and converges at rate n" begin
    # The order claim must be sharp: exactness one degree higher would
    # mean the test above proves less than it says. The rates are the
    # numbers a caller chooses `n` by — `n` for the value, `n − 1` for a
    # first derivative (measured 2026-09-25, recorded in CODE.md), `n − 2`
    # for a pure second derivative and `n − 1` for a mixed one, whose
    # error is that of one first derivative in each of two dimensions —
    # but no better than `p − |m|`, since the ghosts a stencil reads near
    # a coarse-fine face carry the exchange's `O(hᵖ)` error and an `m`-th
    # derivative divides it by `h^|m|`. So `p = 4` holds `∂ₓ∂ᵧ` through
    # `Lagrange(4)` to rate 2, and `p = 6` gives it back its 3 (measured
    # 2026-10-03).
    f, _ = tensorpoly(2, 4)
    fs = interp_fieldset(interp_forest(2), f, cellcentered(2), 6)
    xs = probe_points(Xoshiro(3), fs)
    r = interpolate(fs, xs, Lagrange(4))
    @test maximum(abs(r.values[1, 1, j] - f(xs[j], 1)) for j in eachindex(xs)) > 1e-6

    g(x, v) = sin(2x[1] + v) * cos(3x[2] - 0.5)
    gx(x, v) = 2cos(2x[1] + v) * cos(3x[2] - 0.5)
    gxx(x, v) = -4sin(2x[1] + v) * cos(3x[2] - 0.5)
    gxy(x, v) = -6cos(2x[1] + v) * sin(3x[2] - 0.5)
    rng = Xoshiro(11)
    xs = [(2rand(rng) - 1, 2rand(rng) - 1) for _ in 1:200]
    for (n, p) in ((3, 4), (4, 4), (4, 6), (5, 6)),
        C in (cellcentered(2), vertexcentered(2))
        errs = map((8, 16)) do N
            fs = interp_fieldset(interp_forest(2; N=N), g, C, p)
            r = interpolate(fs, xs, Lagrange(n); derivs=((0, 0), (1, 0), (2, 0), (1, 1)))
            map(enumerate((g, gx, gxx, gxy))) do (k, exact)
                maximum(abs(r.values[1, k, j] - exact(xs[j], 1)) for j in eachindex(xs))
            end
        end
        rv, rg, rxx, rxy = ntuple(k -> log2(errs[1][k] / errs[2][k]), 4)
        get(ENV, "TREEAMR_SHOW_RATES", "") == "1" && @info "rates" n p C rv rg rxx rxy
        @test rv > n - 0.4
        @test rg > n - 1 - 0.4
        @test rxx > n - 2 - 0.4
        @test rxy > min(n - 1, p - 2) - 0.4
    end
end

@testset "The interpolant is continuous inside a block: n=$n" for n in 2:5
    # Which stencil a query uses must change only at a stored point, where
    # every stencil agrees; a start that switched between nodes (the naive
    # centering for even n) makes the interpolant jump there. Random data,
    # so no polynomial hides a jump.
    forest = Forest((1, 1); N=8)
    fs = FieldSet(forest, 1; G=3)
    fs.work .= rand(Xoshiro(n), size(fs.work)...)
    h = spacing(forest, 0)
    ε = 1e-9 * h
    worst = 0.0
    for i in 2:6, frac in (0.0, 0.25, 0.5, 0.75)
        # Along x₁ across the node / midpoint at (i + frac)·h − h/2.
        x = (i + frac - 0.5) * h
        y = 0.37
        r = interpolate(fs, [(x - ε, y), (x + ε, y)], Lagrange(n))
        worst = max(worst, abs(r.values[1, 1, 1] - r.values[1, 1, 2]))
    end
    @test worst < 1e-6
end

@testset "A periodic point is interpolated where it wraps to: D=$D" for D in (1, 2)
    # A point one period away is the same point. The wrap is applied
    # before location and before the stencil (a stencil from the
    # unwrapped point would be clamped, and wrong); only the rounding of
    # the wrap itself may differ.
    g(x, v) = prod(sin(π * x[d] + 0.3v + d) for d in 1:D)
    forest = interp_forest(D; periodic=ntuple(_ -> true, D))
    fs = FieldSet(forest, 2; G=2)
    fill_by_coordinates!(g, fs)
    fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=4, restriction=4)))
    rng = Xoshiro(5)
    xs = [ntuple(_ -> 2rand(rng) - 1, D) for _ in 1:50]
    shifted = [ntuple(d -> x[d] + 2 * rand(rng, (-2, -1, 1, 3)), D) for x in xs]
    derivs = (gradient_derivs(D)..., hessian_derivs(D)...)
    a = interpolate(fs, xs, Lagrange(4); derivs=derivs).values
    b = interpolate(fs, shifted, Lagrange(4); derivs=derivs).values
    @test maximum(abs, a[:, 1:(D + 1), :] - b[:, 1:(D + 1), :]) < 1e-11
    @test maximum(abs, a - b) < 1e-9
    # And it is the function, to the interpolation error.
    @test maximum(abs(a[v, 1, j] - g(xs[j], v)) for j in eachindex(xs), v in 1:2) < 1e-3
end

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

@testset "Beyond a reflecting face the value is the parity-signed mirror: kinds=$kinds, C=$C" for
        (kinds, C) in (((:reflect_lo,), cellcentered(1)),
                       ((:reflect_hi,), vertexcentered(1)),
                       ((:reflect_lo, :outer), cellcentered(2)),
                       ((:reflect_hi, :reflect_lo), vertexcentered(2)),
                       ((:outer, :reflect_hi), facecentered(2, 2)),
                       ((:reflect_lo, :periodic), cellcentered(2)))
    # A symmetric run's horizon finder asks for points across the wall.
    # The answer is the mirror image's value times the variable's parity,
    # and a derivative across the wall flips once more per order — so
    # `∂ₓ²` across a wall in x does not flip and `∂ₓ∂ᵧ` does; getting a
    # sign wrong gives a gradient pointing the wrong way at the wall, or a
    # curvature of the wrong sign.
    D = length(kinds)
    forest = faces_forest(kinds)
    roots = D == 1 ? 3 : 2
    walls = ntuple(d -> kinds[d] === :reflect_hi ? Float64(roots) : 0.0, D)
    fac(x, v, d) = parity_factor(kinds[d], v, d, x[d], walls[d])
    f(x, v) = prod(fac(x, v, d)[1] for d in 1:D)
    dm(x, v, m) = prod(fac(x, v, d)[m[d] + 1] for d in 1:D)
    single(k) = k === :reflect_lo || k === :reflect_hi
    parity = [ntuple(d -> single(kinds[d]) ? (v == 1 ? EvenParity : OddParity) :
                          kinds[d] === :reflect_both ? EvenParity : NoParity, D)
              for v in 1:2]
    fs = FieldSet(forest, 2; G=ghosts_for(C, 4), centering=C, parity=parity)
    fill_by_coordinates!(f, fs)
    fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=4, restriction=4));
                 boundary=boundary_by_coordinates(f))
    rng = Xoshiro(17 + D)
    xs = map(1:80) do _
        ntuple(D) do d
            k = kinds[d]
            k === :reflect_lo ? -roots / 2 + (3roots / 2) * rand(rng) :
            k === :reflect_hi ? (3roots / 2) * rand(rng) :
            k === :periodic ? -roots + 3roots * rand(rng) : roots * rand(rng)
        end
    end
    push!(xs, ntuple(d -> walls[d] + (single(kinds[d]) ? (walls[d] > 0 ? 0.3 : -0.3) : 0.7), D))
    derivs = (gradient_derivs(D)..., hessian_derivs(D)...)
    r = interpolate(fs, xs, Lagrange(4); derivs=derivs)
    worst = errors_by_order(r.values, xs, derivs, dm)
    showerrors() && @info "reflecting" kinds C worst = Tuple(worst)
    @test worst[1] < 1e-11
    @test worst[2] < 1e-10
    @test worst[3] < 1e-9
    # A point beyond the wall by more than the domain is still outside.
    far = ntuple(d -> single(kinds[d]) ? (walls[d] > 0 ? 3roots + 0.5 : -2roots - 0.5) :
                                         0.5, D)
    @test_throws "outside the domain" interpolate(fs, [far], Lagrange(4))
end

@testset "An excluded region flags exactly the stencils that reach into it" begin
    # The guard a caller builds on: a stencil point inside the region
    # must never go unflagged, and one outside must never be flagged. The
    # separable ellipsoid test is compared against enumerating every
    # stencil point with the region's own `inside`, bit for bit.
    rng = Xoshiro(99)
    sig = Tuple{Region,NTuple{3,Float64},Any,Any,Any,Val{4}}
    mismatches, hits = 0, 0
    for _ in 1:2000
        e = Ellipsoid(ntuple(_ -> 2rand(rng) - 1, 3), ntuple(_ -> 0.05 + 0.5rand(rng), 3))
        h = 0.02 + 0.2rand(rng)
        # Near the center, so that both answers are common.
        origin = ntuple(d -> e.center[d] - 4h + 2h * rand(rng) - 0.3rand(rng), 3)
        base = ntuple(_ -> rand(rng, -3:3), 3)
        off = ntuple(_ -> rand(rng, (0.5, 1.0)), 3)
        fast = stencil_hits(e, origin, h, base, off, Val(4))
        slow = invoke(stencil_hits, sig, e, origin, h, base, off, Val(4))
        mismatches += fast != slow
        hits += fast
    end
    @test mismatches == 0
    @test 0 < hits < 2000                    # both answers were exercised
    # And one of each by construction: a stencil point at the center, and
    # a stencil whose nearest point is just outside along one axis.
    e = Ellipsoid((0.0, 0.0, 0.0), (0.5, 0.3, 0.2))
    @test stencil_hits(e, (-0.1, -0.1, -0.1), 0.1, (0, 0, 0), (0.0, 0.0, 0.0), Val(4))
    @test !stencil_hits(e, (0.5, -0.1, -0.1), 0.1, (0, 0, 0), (0.0, 0.0, 0.0), Val(4))

    # Through `interpolate`: the flag is the oracle's, the values are not
    # changed by asking, and a query far from the region is not flagged.
    f, _ = tensorpoly(3, 2)
    fs = interp_fieldset(interp_forest(3; N=6), f, vertexcentered(3), 4)
    xs = probe_points(Xoshiro(4), fs; nrandom=200)
    ball = Ellipsoid((-0.4, -0.3, -0.2), (0.3, 0.2, 0.25))
    plain = interpolate(fs, xs, Lagrange(4))
    r = interpolate(fs, xs, Lagrange(4); exclude=ball)
    @test r.values == plain.values
    @test !any(plain.excluded)
    g = PointGeometry(Float64, fs.forest)
    expected = map(xs) do x
        b = locate_point(fs.forest, x)
        k = blockkey(fs, b)
        first, _, _, _ = query_stencil(Lagrange(4), x, block_origin(fs.forest, k),
                                       spacing(fs.forest, k), Val(fs.G),
                                       Val(staggers(fs)), Val(size(fs.work)[1:3]))
        any(TreeAMR.inside(ball, coordinates(fs, b, Tuple(first) .+ Tuple(J) .- 1))
            for J in CartesianIndices((4, 4, 4)))
    end
    @test r.excluded == expected
    @test 0 < count(r.excluded) < length(xs)
    @test_throws "semiaxes must be positive" Ellipsoid((0, 0), (1, 0))
    @test_throws "one semiaxis per dimension" Ellipsoid((0, 0), (1, 1, 1))
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

@testset "The kernel reads the stencil query_stencil names, with exact weights" begin
    # Everything else checks the answer; this checks the mechanism. With
    # random data no polynomial can mask a stencil one point off: the
    # kernel's value and derivatives must be the contraction of *this*
    # block's stored points over the stencil, with exact rational Lagrange
    # weights and their exact derivatives, scaled by h^|m|.
    forest = interp_forest(2)
    derivs = ((0, 0), (0, 1), (1, 1), (0, 2))
    for (C, n) in ((cellcentered(2), 4), (vertexcentered(2), 3), (facecentered(2, 1), 5))
        fs = FieldSet(forest, 1; G=2, centering=C)
        fs.work .= rand(Xoshiro(n), size(fs.work)...)
        xs = probe_points(Xoshiro(n + 1), fs)
        r = interpolate(fs, xs, Lagrange(n); derivs=derivs)
        nodes = Rational{BigInt}.(0:(n - 1))
        worst = 0.0
        for (j, x) in enumerate(xs)
            b = locate_point(forest, x)
            k = blockkey(fs, b)
            h = spacing(forest, k)
            first, _, _, ξ = query_stencil(Lagrange(n), x, block_origin(forest, k), h,
                                           Val(fs.G), Val(staggers(fs)),
                                           Val(size(fs.work)[1:2]))
            w = [[lagrange_derivative_weights(nodes, Rational{BigInt}(ξ[d]), o)
                  for o in 0:2] for d in 1:2]
            @test w[1][1] == lagrange_weights(nodes, Rational{BigInt}(ξ[1]))
            u = view(fs.work, first[1]:(first[1] + n - 1), first[2]:(first[2] + n - 1), 1, b)
            for (kd, m) in enumerate(derivs)
                exact = sum(w[1][m[1] + 1][i] * w[2][m[2] + 1][l] *
                            Rational{BigInt}(u[i, l]) for i in 1:n, l in 1:n)
                worst = max(worst, abs(h^sum(m) * r.values[1, kd, j] - Float64(exact)))
            end
        end
        @test worst < 1e-12
    end
end

@testset "interpolate refuses what it cannot answer, and says why" begin
    # Each of these would otherwise be a wrong number, a bounds error
    # deep in a kernel, or a silent extrapolation.
    f, _ = tensorpoly(2, 2)
    fs = interp_fieldset(interp_forest(2), f, cellcentered(2), 4)
    xs = [(0.1, 0.2), (0.3, -0.4)]
    @test_throws "outside the domain" interpolate(fs, [(0.1, 0.2), (1.5, 0.0)], Lagrange(4))
    @test_throws "point 2" interpolate(fs, [(0.1, 0.2), (1.5, 0.0)], Lagrange(4))
    @test_throws "outside the domain" interpolate(fs, [(NaN, 0.0)], Lagrange(4))
    @test_throws "stores only 12" interpolate(fs, xs, Lagrange(13))
    @test_throws "identically zero" interpolate(fs, xs, Lagrange(1); derivs=((1, 0),))
    @test_throws "up to second order" interpolate(fs, xs, Lagrange(4); derivs=((3, 0),))
    @test_throws "up to second order" interpolate(fs, xs, Lagrange(4); derivs=((2, 1),))
    @test_throws "identically zero" interpolate(fs, xs, Lagrange(2); derivs=((2, 0),))
    # A mixed second derivative is first order in each dimension, which
    # two points per dimension produce.
    @test size(interpolate(fs, xs, Lagrange(2); derivs=((1, 1),)).values) == (2, 1, 2)
    @test_throws "multi-index of 2" interpolate(fs, xs, Lagrange(4); derivs=((0, 0, 0),))
    @test_throws "nonempty tuple" interpolate(fs, xs, Lagrange(4); derivs=())
    @test_throws "vars must name" interpolate(fs, xs, Lagrange(4); vars=1:3)
    @test_throws "has 2 coordinates" interpolate(fs, [(0.1,)], Lagrange(4))
    @test_throws "at least one point" Lagrange(0)
    @test_throws "must have size" interpolate!(zeros(2, 2, 2), falses(2), fs, xs, Lagrange(4))
    @test_throws "one entry per point" interpolate!(zeros(2, 1, 2), falses(3), fs, xs,
                                                    Lagrange(4))
    @test_throws "element type" interpolate!(zeros(Float32, 2, 1, 2), falses(2), fs, xs,
                                             Lagrange(4))
    # A subset of the variables, in the order asked for, and an empty batch.
    r = interpolate(fs, xs, Lagrange(4); vars=[2, 1])
    @test r.values[1, 1, :] ≈ [f(x, 2) for x in xs]
    @test r.values[2, 1, :] ≈ [f(x, 1) for x in xs]
    @test size(interpolate(fs, NTuple{2,Float64}[], Lagrange(4)).values) == (2, 1, 0)
    @test sprint(show, Lagrange(4)) == "Lagrange(4)"
end
