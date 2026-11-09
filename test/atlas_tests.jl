# Point interpolation through the application's atlas (M19): physical
# positions located through `chart_locate`, physical derivatives through
# the chain rule, physical `exclude` regions, and outside points.
#
# The oracle of every derivative is the inverse map differentiated by
# hand (`inverse_derivs` in `helpers/interpolate.jl`), never the
# package's `J⁻¹` and `H`; the fields are polynomials in the logical
# coordinates, so that the interpolation itself is exact and every
# difference is the chain rule's.

# An element set of order `P` over `interp_forest`'s three levels, filled
# with `f` in logical coordinates.
function atlas_element_set(D, P, f; N=2 * (P + 1))
    fs = FieldSet(interp_forest(D; N=N), 2; G=1, centering=elementcentered(D), P=P)
    fill_by_coordinates!(f, fs)
    return fs
end

# Physical points: random logical ones in `[-1, 1]^D` mapped forward.
atlas_points(rng, atlas, D, n) =
    [TreeAMR.chart_map(atlas, ntuple(_ -> 2rand(rng) - 1, D), 1) for _ in 1:n]

@testset "Physical derivatives through an atlas are the analytic ones: $name" for
        (name, atlas, D) in (("affine 2D", AffineAtlas([1.3 0.4; -0.2 0.9], (0.5, -1.0)), 2),
                             ("affine 3D", AffineAtlas([1.1 0.2 0.0; 0.1 0.8 -0.3;
                                                        0.0 0.4 1.2], (0.0, 1.0, 2.0)), 3),
                             ("sinh 2D", SinhAtlas{2}(), 2),
                             ("sinh 3D", SinhAtlas{3}(), 3),
                             ("shear", ShearAtlas(0.35), 2))
    # The chain rule to second order: a wrong index of K, a transposed
    # Jacobian, or the Hessian term's sign would each show far above
    # roundoff for a map that is not the identity, and only the
    # nonlinear maps exercise the Hessian at all.
    P = 3
    f, _, dm = tensorpoly(D, P)
    fs = atlas_element_set(D, P, f)
    xs = atlas_points(Xoshiro(23 + D), atlas, D, D == 3 ? 60 : 150)
    derivs = multi_indices(Val(D), 2)
    r = interpolate(fs, xs, ElementPolynomial(); derivs=derivs, atlas=atlas)
    @test all(r.inside) && !any(r.excluded)
    worst = zeros(3)
    for (j, x) in enumerate(xs), v in 1:2, (k, m) in enumerate(derivs)
        o = sum(m) + 1
        worst[o] = max(worst[o], abs(r.values[v, k, j] - physical_oracle(atlas, dm, x, v, m)))
    end
    showerrors() && @info "atlas" name worst = Tuple(worst)
    @test worst[1] < 1e-13
    @test worst[2] < 1e-11
    @test worst[3] < 1e-9
    # Located once, the same bits; and Lagrange on a brick takes an atlas too.
    pts = locate(fs.forest, xs; atlas=atlas)
    @test isequal(interpolate(fs, pts, ElementPolynomial(); derivs=derivs).values, r.values)
    if D == 2
        cs = interp_fieldset(interp_forest(2), f, cellcentered(2), 4)
        rl = interpolate(cs, xs, Lagrange(4); derivs=derivs, atlas=atlas)
        @test maximum(abs(rl.values[v, k, j] - physical_oracle(atlas, dm, xs[j], v, m))
                      for j in eachindex(xs), v in 1:2,
                          (k, m) in enumerate(derivs)) < 1e-9
    end
end

@testset "multi_indices lists the value, the gradient and the Hessian in order" begin
    # The chain rule finds each logical derivative by its position in
    # this list; a reordering would mix the Hessian's entries.
    @test multi_indices(Val(2), 0) == ((0, 0),)
    @test multi_indices(Val(2), 1) == ((0, 0), (1, 0), (0, 1))
    @test multi_indices(Val(3), 2) ==
          ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0), (1, 0, 1),
           (0, 2, 0), (0, 1, 1), (0, 0, 2))
    for D in 1:3, a in 1:D, b in a:D
        m = ntuple(d -> Int(d == a) + Int(d == b), D)
        @test multi_indices(Val(D), 2)[1 + D + TreeAMR.pairindex(a, b, D)] == m
    end
    @test_throws "up to the second order" multi_indices(Val(2), 3)
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

@testset "The identity atlas changes nothing, across a reflecting wall too" begin
    # The fold's signs apply to the logical derivatives before the chain
    # rule, and the chain rule through the identity multiplies by exact
    # ones and zeros, so the values must be the ones without an atlas bit
    # for bit — beyond the wall as well, where a sign applied twice or not
    # at all would flip a derivative.
    kinds = (:reflect_lo, :outer)
    forest = faces_forest(kinds)
    fs = FieldSet(forest, 2; G=1, centering=elementcentered(2), P=3,
                  parity=[(EvenParity, NoParity), (OddParity, NoParity)])
    fill_by_coordinates!((x, v) -> (v == 1 ? 1 + x[1]^2 : x[1] - x[1]^3) * (1 + x[2]), fs)
    rng = Xoshiro(29)
    xs = [(-1 + 3rand(rng), 2rand(rng)) for _ in 1:120]
    derivs = multi_indices(Val(2), 2)
    a = interpolate(fs, xs, ElementPolynomial(); derivs=derivs)
    b = interpolate(fs, xs, ElementPolynomial(); derivs=derivs, atlas=IdentityAtlas{2}())
    # `==`, not `isequal`: the chain rule's sum from zero turns a `-0.0`
    # into `0.0`, which is the same number.
    @test a.values == b.values
    @test any(x -> x[1] < 0, xs)
end

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

@testset "On the annulus the atlas chooses each point's patch" begin
    # The atlas folds the angle and picks the patch; TreeAMR then locates
    # the logical position there, as for the PatchPoint it amounts to. A
    # point in the hole is marked, not taken from the nearest patch.
    ring = patch_ring(4, (2, 1))
    fs = FieldSet(ring, 1; G=1, centering=elementcentered(2), P=1)
    fill_by_coordinates!(WithPatch((ξ, p, v) -> sin(ξ[1] + p) * ξ[2]), fs)
    rng = Xoshiro(31)
    pps = [PatchPoint((2rand(rng), rand(rng)), rand(rng, 1:4)) for _ in 1:200]
    xs = [TreeAMR.chart_map(PolarAtlas(), q.ξ, q.p) for q in pps]
    pts = locate(ring, xs; atlas=PolarAtlas())
    @test Array(pts.patch) == [q.p for q in pps]
    at = [PatchPoint(ξ, p) for (ξ, p) in zip(Array(pts.ξ), Array(pts.patch))]
    @test isequal(interpolate(fs, pts, ElementPolynomial()).values,
                  interpolate(fs, at, ElementPolynomial()).values)
    @test maximum(abs(Array(pts.ξ)[j][d] - pps[j].ξ[d]) for j in eachindex(pps), d in 1:2) <
          1e-13
    hole = [(0.2, 0.1), (1.5, 0.0), (3.0, 0.0)]
    m = interpolate(fs, hole, ElementPolynomial(); atlas=PolarAtlas(), outside=:mark)
    @test m.inside == [false, true, false]
    @test isnan(m.values[1, 1, 1]) && !isnan(m.values[1, 1, 2])
    @test_throws "is in no patch" interpolate(fs, hole, ElementPolynomial();
                                              atlas=PolarAtlas())
end

@testset "A physical exclude region is tested at every stencil node's image" begin
    # With an atlas the region is physical, and a curved chart has no
    # separable test: each stencil node is mapped and tested. The oracle
    # maps the nodes' own coordinates.
    P = 2
    f, _ = tensorpoly(2, P)
    fs = atlas_element_set(2, P, f)
    atlas = SinhAtlas{2}()
    xs = atlas_points(Xoshiro(37), atlas, 2, 300)
    ball = Ellipsoid((-0.4, 0.1), (0.35, 0.3))
    r = interpolate(fs, xs, ElementPolynomial(); atlas=atlas, exclude=ball)
    expected = map(xs) do x
        ξ = TreeAMR.chart_locate(atlas, x)[1]
        b = locate_point(fs.forest, ξ)
        k = blockkey(fs, b)
        o, h = block_origin(fs.forest, k), spacing(fs.forest, k)
        q = ntuple(d -> clamp(floor(Int, (ξ[d] - o[d]) / h / (P + 1)), 0,
                              fs.forest.N ÷ (P + 1) - 1), 2)
        any(TreeAMR.inside(ball, TreeAMR.chart_map(atlas, coordinates(fs, b,
                q .* (P + 1) .+ Tuple(J) .+ fs.G), 1)) for J in CartesianIndices((3, 3)))
    end
    @test r.excluded == expected
    @test 0 < count(r.excluded) < length(xs)
end

# An atlas with nothing but the location; and one that returns a position
# outside its patch's box.
struct LocateOnlyAtlas end
TreeAMR.chart_locate(::LocateOnlyAtlas, x) = (x, 1)
struct NoHessianAtlas end
TreeAMR.chart_locate(::NoHessianAtlas, x) = (x, 1)
TreeAMR.chart_jacobian(::NoHessianAtlas, ξ, p) = ((1.0, 0.0), (0.0, 1.0))
struct OverreachingAtlas end
TreeAMR.chart_locate(::OverreachingAtlas, x) = ((x[1] + 3, x[2]), 1)
struct WrongPatchAtlas end
TreeAMR.chart_locate(::WrongPatchAtlas, x) = (x, 2)

@testset "An atlas without a needed method, or breaking its contract, is refused" begin
    # A missing method would otherwise fail deep in a kernel's compilation
    # with no word of what the atlas lacks; a position outside its patch's
    # box would be read from a neighbouring block, or extrapolated.
    f, _ = tensorpoly(2, 2)
    fs = atlas_element_set(2, 2, f)
    xs = [(0.1, 0.2), (-0.3, 0.4)]
    plain = interpolate(fs, xs, ElementPolynomial()).values
    @test isequal(interpolate(fs, xs, ElementPolynomial(); atlas=LocateOnlyAtlas()).values,
                  plain)
    @test_throws "no method TreeAMR.chart_jacobian" interpolate(
        fs, xs, ElementPolynomial(); derivs=gradient_derivs(2), atlas=LocateOnlyAtlas())
    @test_throws "no method TreeAMR.chart_map" interpolate(
        fs, xs, ElementPolynomial(); atlas=LocateOnlyAtlas(),
        exclude=Ellipsoid((0.0, 0.0), (0.1, 0.1)))
    @test size(interpolate(fs, xs, ElementPolynomial(); derivs=gradient_derivs(2),
                           atlas=NoHessianAtlas()).values) == (2, 3, 2)
    @test_throws "no method TreeAMR.chart_hessian" interpolate(
        fs, xs, ElementPolynomial(); derivs=((2, 0),), atlas=NoHessianAtlas())
    @test_throws "no method TreeAMR.chart_locate" locate(fs.forest, xs; atlas=PolarAtlas)
    @test_throws "broke its contract" locate(fs.forest, xs; atlas=OverreachingAtlas())
    @test_throws "outside the closed box" locate(fs.forest, xs; atlas=OverreachingAtlas())
    @test_throws "one brick, patch 1" locate(fs.forest, xs; atlas=WrongPatchAtlas())
    @test_throws "a PatchPoint names its patch already" locate(
        fs.forest, [PatchPoint((0.1, 0.2), 1)]; atlas=LocateOnlyAtlas())
    # A physical second derivative mixes every logical one of its order,
    # which an element of order 1 does not have.
    e1 = atlas_element_set(2, 1, f)
    @test_throws "mixes every logical one" interpolate(
        e1, xs, ElementPolynomial(); derivs=((1, 1),), atlas=IdentityAtlas{2}())
end
