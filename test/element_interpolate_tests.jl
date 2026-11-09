# Point interpolation on element sets (M19): `ElementPolynomial`, the
# degree-`P` polynomial of the element containing the point.
#
# Every claim is stated against something the package did not compute:
# analytic tensor polynomials and their derivatives for exactness, a
# smooth analytic field for the rates, the stored node values themselves
# for a query at a node, and the nodes' own `coordinates` for the
# excluded region.

# An element set over `interp_forest`'s three levels, filled with `f` at
# its nodes. No ghosts are filled: the basis reads none.
function element_interp_set(D, P, f; N=2 * (P + 1), G=1, nvars=2, kwargs...)
    forest = interp_forest(D; N=N, kwargs...)
    fs = FieldSet(forest, nvars; G=G, centering=elementcentered(D), P=P)
    fill_by_coordinates!(f, fs)
    return fs
end

# Every multi-index of total order up to two whose order along each
# dimension is at most `P`, the derivatives an element of order `P` has.
element_derivs(D, P) =
    Tuple(m for m in (gradient_derivs(D)..., hessian_derivs(D)...) if maximum(m) <= P)

@testset "ElementPolynomial is exact on Q_P with its derivatives: D=$D, P=$P, G=$G" for
        (D, Ps) in ((1, 1:4), (2, 1:4), (3, (2, 3))), P in Ps, G in (0, 1)
    # The claim that makes the basis the element's own interpolant: a
    # polynomial of degree P in each dimension, filled at the nodes, comes
    # back at any point with every derivative the element has, through a
    # three-level forest, at random points, block corners and nodes. A
    # wrong offset, a wrong barycentric weight or the wrong element would
    # each break it far above roundoff.
    f, _, dm = tensorpoly(D, P)
    fs = element_interp_set(D, P, f; G=G)
    xs = probe_points(Xoshiro(100P + 10D + G), fs)
    derivs = element_derivs(D, P)
    r = interpolate(fs, xs, ElementPolynomial(); derivs=derivs)
    @test size(r.values) == (2, length(derivs), length(xs))
    @test !any(r.excluded) && all(r.inside)
    worst = errors_by_order(r.values, xs, derivs, dm)
    showerrors() && @info "element exactness" D P G worst = Tuple(worst)
    @test worst[1] < 1e-13
    @test worst[2] < 1e-11
    @test worst[3] < 1e-9
    @test split_matches(fs, xs, ElementPolynomial(); derivs=derivs)
end

@testset "A query at a node reproduces the node's value: D=$D, P=$P" for
        (D, P) in ((1, 3), (2, 2), (2, 4), (3, 2))
    # The node offsets are `nodeposition`'s, rounded once from the same
    # BigFloat, so a query at a node's coordinates evaluates the element's
    # polynomial where it interpolates that node; anything above a few
    # roundings means the stencil and the stored positions disagree. A
    # node on a block's upper face is left out: that point belongs to the
    # block above, as every face point does, whose element interpolates
    # it only to its own accuracy where the levels differ. A node on an
    # element face inside a block is in, and is read from the upper
    # element, which holds the same value.
    g(x, v) = prod(sin(2x[d] + v + d) for d in 1:D)
    fs = element_interp_set(D, P, g)
    worst = 0.0
    N = fs.forest.N
    for b in 1:nblocks(fs), I in CartesianIndices(ntuple(_ -> N - 1, D))
        idx = Tuple(I) .+ fs.G
        x = coordinates(fs, b, idx)
        r = interpolate(fs, [x], ElementPolynomial(); vars=[2]).values[1]
        worst = max(worst, abs(r - fs.work[idx..., 2, b]))
    end
    @test worst <= 8eps()
end

@testset "ElementPolynomial converges at P + 1, a derivative at P + 1 − maxₐ mₐ: P=$P" for
        P in 1:4
    # The rates the design promises, measured on a smooth field over a
    # refined forest: no ghosts are read, so no exchange error lowers
    # them at the coarse-fine faces, and a mixed derivative keeps the
    # rate of its highest single order.
    g(x, v) = sin(2x[1] + v) * cos(3x[2] - 0.5)
    dg(x, v, m) = m == (0, 0) ? g(x, v) :
                  m == (1, 0) ? 2cos(2x[1] + v) * cos(3x[2] - 0.5) :
                  m == (0, 1) ? -3sin(2x[1] + v) * sin(3x[2] - 0.5) :
                  m == (2, 0) ? -4g(x, v) :
                  m == (1, 1) ? -6cos(2x[1] + v) * sin(3x[2] - 0.5) :
                  m == (0, 2) ? -9g(x, v) : error("no such derivative")
    derivs = element_derivs(2, P)
    rng = Xoshiro(41 + P)
    xs = [(2rand(rng) - 1, 2rand(rng) - 1) for _ in 1:300]
    err(N) = begin
        fs = element_interp_set(2, P, g; N=N)
        r = interpolate(fs, xs, ElementPolynomial(); derivs=derivs)
        [maximum(abs(r.values[v, k, j] - dg(xs[j], v, m)) for j in eachindex(xs), v in 1:2)
         for (k, m) in enumerate(derivs)]
    end
    N = 2 * (P + 1)
    rates = log2.(err(N) ./ err(2N))
    get(ENV, "TREEAMR_SHOW_RATES", "") == "1" &&
        println("element rates P=$P ", derivs, " ", round.(rates; digits=2))
    for (k, m) in enumerate(derivs)
        @test rates[k] > P + 1 - maximum(m) - 0.4
    end
end

@testset "An element set's interpolant reads no ghosts, and reads a state vector alike" begin
    # A basis that read a ghost would turn NaN there into NaN here, and
    # would read garbage from a state vector, which has no ghosts; and the
    # two sources must give the same bits, since they hold the same nodes.
    for (D, P) in ((1, 2), (2, 3), (3, 1))
        g(x, v) = prod(cos(x[d] + 0.2v * d) for d in 1:D)
        fs = element_interp_set(D, P, g)
        xs = probe_points(Xoshiro(7D), fs)
        derivs = element_derivs(D, P)
        a = interpolate(fs, xs, ElementPolynomial(); derivs=derivs)
        u = statevector(fs)
        gather!(u, fs)
        fill!(fs.work, NaN)
        scatter!(fs, u)
        b = interpolate(fs, xs, ElementPolynomial(); derivs=derivs)
        c = interpolate(fs, u, xs, ElementPolynomial(); derivs=derivs)
        @test isequal(a.values, b.values)
        @test isequal(a.values, c.values)
        @test !any(isnan, c.values)
        pts = locate(fs.forest, xs)
        @test isequal(interpolate(fs, u, pts, ElementPolynomial(); derivs=derivs).values,
                      a.values)
        v = similar(a.values)
        interpolate!(v, zeros(Bool, length(xs)), fs, u, xs, ElementPolynomial();
                     derivs=derivs)
        @test isequal(v, a.values)
    end
end

@testset "Beyond a reflecting wall an element set's value is the parity-signed mirror" begin
    # The one fold an element set keeps: across the wall a point reads its
    # mirror image, the value times the variable's parity and the normal
    # derivative once more, so that an even variable has zero slope at the
    # wall and an odd one its value zero. The mirror of a point about a
    # wall at 0 is exact, so the comparison is bit for bit.
    P = 3
    forest = Forest((2, 2); N=8, reflecting=((true, false), (false, false)),
                    extents=((0.0, 1.0), (0.0, 1.0)))
    refine!(forest, first(forest.leaves))
    balance!(forest)
    fs = FieldSet(forest, 2; G=1, centering=elementcentered(2), P=P,
                  parity=[(EvenParity, NoParity), (OddParity, NoParity)])
    fill_by_coordinates!((x, v) -> (v == 1 ? 1 + x[1]^2 : x[1] - x[1]^3) * (1 + x[2]), fs)
    rng = Xoshiro(3)
    inner = [(0.4rand(rng), rand(rng)) for _ in 1:60]
    outer = [(-x[1], x[2]) for x in inner]
    derivs = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1))
    a = interpolate(fs, inner, ElementPolynomial(); derivs=derivs).values
    b = interpolate(fs, outer, ElementPolynomial(); derivs=derivs).values
    for (k, m) in enumerate(derivs), v in 1:2
        σ = (v == 1 ? 1 : -1) * (-1)^m[1]
        @test b[v, k, :] == σ .* a[v, k, :]
    end
    @test maximum(abs(b[1, 1, j] - (1 + outer[j][1]^2) * (1 + outer[j][2]))
                  for j in eachindex(outer)) < 1e-13
end

@testset "An excluded region flags exactly the element stencils that reach into it" begin
    # The stencil of an element query is the element's own nodes, at
    # Gauss–Lobatto positions, not the equally spaced ones the uniform
    # bases read; a region tested at the wrong positions would flag the
    # wrong points near its boundary.
    P = 3
    f, _ = tensorpoly(2, P)
    fs = element_interp_set(2, P, f; N=8)
    xs = probe_points(Xoshiro(12), fs; nrandom=300)
    ball = Ellipsoid((-0.35, -0.2), (0.3, 0.22))
    r = interpolate(fs, xs, ElementPolynomial(); exclude=ball)
    expected = map(xs) do x
        b = locate_point(fs.forest, x)
        k = blockkey(fs, b)
        o, h = block_origin(fs.forest, k), spacing(fs.forest, k)
        q = ntuple(d -> clamp(floor(Int, (x[d] - o[d]) / h / (P + 1)), 0,
                              fs.forest.N ÷ (P + 1) - 1), 2)
        any(TreeAMR.inside(ball, coordinates(fs, b, q .* (P + 1) .+ Tuple(J) .+ fs.G))
            for J in CartesianIndices((P + 1, P + 1)))
    end
    @test r.excluded == expected
    @test 0 < count(r.excluded) < length(xs)
    @test r.values == interpolate(fs, xs, ElementPolynomial()).values
end

@testset "Each basis refuses the other's sets, and a state vector a basis that reads ghosts" begin
    # Lagrange through Gauss–Lobatto nodes, or an element polynomial on
    # equally spaced points, would each return wrong numbers silently; a
    # stencil reading ghosts from a state vector would read the next
    # block's points.
    f, _ = tensorpoly(2, 2)
    es = element_interp_set(2, 1, f)
    cs = interp_fieldset(interp_forest(2), f, cellcentered(2), 4)
    xs = [(0.1, 0.2), (-0.3, 0.4)]
    @test_throws "use ElementPolynomial()" interpolate(es, xs, Lagrange(2))
    @test_throws "use ElementPolynomial()" interpolate(es, locate(es.forest, xs),
                                                       Lagrange(2))
    @test_throws "which Lagrange(n) interpolates" interpolate(cs, xs, ElementPolynomial())
    @test_throws "identically zero" interpolate(es, xs, ElementPolynomial();
                                                derivs=((2, 0),))
    @test size(interpolate(es, xs, ElementPolynomial(); derivs=((1, 1),)).values) ==
          (2, 1, 2)
    u = statevector(cs)
    @test_throws "a state vector holds only the owned ones" interpolate(cs, u, xs,
                                                                         Lagrange(4))
    @test_throws "the field set's length" interpolate(es, zeros(3), xs,
                                                      ElementPolynomial())
    @test sprint(show, ElementPolynomial()) == "ElementPolynomial()"
end

@testset "element_lattice places its points where positions says, inside elements: D=$D" for
        D in (1, 2, 3)
    # The lattice is located without a search, so if it named the wrong
    # block or put a point on an element face, its values would not be
    # those of the same points located by search; and `positions` must
    # say where they are, so that a viewer draws them there.
    P, k = 2, 3
    g(x, v) = prod(cos(1.3x[d] + v) for d in 1:D)
    fs = element_interp_set(D, P, g)
    lat = element_lattice(fs, k)
    E = fs.forest.N ÷ (P + 1)
    @test length(lat) == nblocks(fs) * (E * k)^D
    xs = Array(positions(lat))
    @test xs == Array(lat.ξ)
    leaf = Array(lat.leaf)
    for j in eachindex(xs)
        b = leaf[j]
        ext = block_extent(fs.forest, fs.forest.leaves[b])
        @test all(d -> ext[d][1] < xs[j][d] < ext[d][2], 1:D)
    end
    @test leaf == [locate_point(fs.forest, x) for x in xs]
    derivs = gradient_derivs(D)
    @test isequal(interpolate(fs, lat, ElementPolynomial(); derivs=derivs).values,
                  interpolate(fs, xs, ElementPolynomial(); derivs=derivs).values)
    # Of every cell, for a set without elements.
    cs = FieldSet(fs.forest, 1; G=2)
    cl = element_lattice(cs, 2)
    @test length(cl) == nblocks(cs) * (2fs.forest.N)^D
    @test Array(cl.leaf) == [locate_point(fs.forest, x) for x in Array(positions(cl))]
    @test_throws "at least 1" element_lattice(fs, 0)
end

@testset "positions with an atlas are physical" begin
    # A viewer draws in physical space: with an atlas the located points'
    # positions are their images (an outside point keeps the position it
    # was given, `atlas_tests.jl`).
    f, _ = tensorpoly(2, 2)
    fs = element_interp_set(2, 2, f)
    atlas = SinhAtlas{2}()
    lat = element_lattice(fs, 2; atlas=atlas)
    @test Array(positions(lat)) == [TreeAMR.chart_map(atlas, ξ, 1) for ξ in Array(lat.ξ)]
    pts = locate(fs.forest, [(0.3, -0.2)]; atlas=atlas)
    @test all(Array(positions(pts))[1] .≈ (0.3, -0.2))
end
