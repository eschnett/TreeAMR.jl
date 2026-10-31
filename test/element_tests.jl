# M16: the element centering. A field set of elements of `P + 1`
# Gauss–Lobatto–Legendre nodes, both face nodes stored, for element-local
# DG schemes such as OLDG. These tests cover the reference element's
# tables, the centering and its refusals, and the positions and weights
# of the nodes.

using MultiFloats: Float32x2

# The tables at full precision, and the closed forms they must match.
elementbig(P) = TreeAMR.element_tables(BigFloat, P)

# Closed-form GLL nodes and weights for `P ≤ 4`, in 512-bit `BigFloat`.
function gll_closed_form(P)
    return setprecision(BigFloat, 512) do
        if P == 1
            ξ, w = big.([-1, 1]), big.([1, 1])
        elseif P == 2
            ξ, w = big.([-1, 0, 1]), [big(1) / 3, big(4) / 3, big(1) / 3]
        elseif P == 3
            a = 1 / sqrt(big(5))
            ξ, w = [-1, -a, a, 1], [big(1) / 6, big(5) / 6, big(5) / 6, big(1) / 6]
        else
            a = sqrt(big(3) / 7)
            ξ = [-1, -a, 0, a, 1]
            w = [big(1) / 10, big(49) / 90, big(32) / 45, big(49) / 90, big(1) / 10]
        end
        return BigFloat.(ξ), w
    end
end

# The fine nodes of the two halves in the coarse element's coordinate.
fine_nodes(t) = [[(x - 1) / 2 for x in t.nodes]; [(x + 1) / 2 for x in t.nodes]]

@testset "The GLL nodes and weights are the closed forms: P=$P" for P in 1:4
    # Guards a wrong node or weight construction (a different family of
    # points, an unconverged iteration, weights off by a factor), which
    # every position, norm and transfer of an element set inherits.
    ξ, w = gll_closed_form(P)
    t = elementbig(P)
    setprecision(BigFloat, 512) do
        @test maximum(abs, t.nodes - ξ) < 1e-140
        @test maximum(abs, t.weights - w) < 1e-140
    end
    for T in (Float64, Float32, Float32x2)
        tt = TreeAMR.element_tables(T, P)
        @test tt.nodes == T.(ξ)
        @test tt.weights == T.(w)
        # Exact where it must be: the faces, the centre of an even P, the
        # face offsets, and the mirror symmetry bit for bit.
        @test tt.nodes[1] == -1 && tt.nodes[end] == 1
        iseven(P) && @test tt.nodes[P ÷ 2 + 1] === zero(T)
        @test tt.offsets[1] == 0 && tt.offsets[end] == P + 1
        @test tt.nodes == -reverse(tt.nodes)
        @test tt.weights == reverse(tt.weights)
    end
end

@testset "The GLL quadrature is exact to degree 2P-1 and not 2P: P=$P" for P in 1:6
    # Guards weights that integrate the wrong polynomials, which would
    # make the element norm and the compatible restriction wrong.
    t = elementbig(P)
    setprecision(BigFloat, 512) do
        moment(k) = sum(t.weights .* t.nodes .^ k)
        exact(k) = iseven(k) ? big(2) / (k + 1) : big(0)
        @test all(k -> abs(moment(k) - exact(k)) < 1e-140, 0:(2P - 1))
        @test abs(moment(2P) - exact(2P)) > 1e-6
        # In units of the spacing an element is P+1 wide.
        @test abs(sum(t.quadrature) - (P + 1)) < 1e-140
    end
end

@testset "I is the coarse polynomial at the fine nodes, exact to degree P: P=$P" for P in 1:6
    # Guards a prolongation that does not reproduce the coarse element's
    # polynomial, or reads its halves the wrong way round.
    t = elementbig(P)
    Ic = vcat(t.interpolation...)
    setprecision(BigFloat, 512) do
        xf = fine_nodes(t)
        for k in 0:P
            @test maximum(abs, Ic * t.nodes .^ k - xf .^ k) < 1e-140
        end
        @test maximum(abs, Ic * t.nodes .^ (P + 1) - xf .^ (P + 1)) > 1e-6
    end
end

@testset "R is compatible with I: M_c R = Iᵀ M_f, P=$P" for P in 1:6
    # Guards the property the coarse-fine energy argument rests on: the
    # coarse side's ghost operator must be the adjoint of the fine
    # side's in the GLL inner products, or the derivative stops being
    # skew-adjoint across the face.
    t = elementbig(P)
    Ic = vcat(t.interpolation...)
    setprecision(BigFloat, 512) do
        wf = [t.weights; t.weights] ./ 2        # the fine mass, in coarse units
        @test maximum(abs, t.weights .* t.restriction - Ic' .* wf') < 1e-140
    end
    for T in (Float64, Float32, Float32x2)
        tt = TreeAMR.element_tables(T, P)
        Ict = vcat(tt.interpolation...)
        wft = [tt.weights; tt.weights] ./ 2
        @test maximum(abs, tt.weights .* tt.restriction - Ict' .* wft') <= 8 * eps(T)
    end
end

@testset "R is exact to degree P-1 and not P, and conserves Σŵu: P=$P" for P in 1:6
    # Guards the measured loss of one degree in the coarse ghost: a
    # restriction exact to P would not be the adjoint of I, and one exact
    # to less would cost the scheme its order. Float64 only: "not exact"
    # is a truncation error, not a rounding one (see "Precision").
    t = TreeAMR.element_tables(Float64, P)
    xf = fine_nodes(t)
    for k in 0:(P - 1)
        @test maximum(abs, t.restriction * xf .^ k - t.nodes .^ k) < 1e-13
    end
    @test maximum(abs, t.restriction * xf .^ P - t.nodes .^ P) > 1e-3
    u = [2 + cos(3x) + x^P for x in xf]
    @test sum(t.weights .* (t.restriction * u)) ≈ sum([t.weights; t.weights] .* u) / 2 rtol=1e-14
end

@testset "The coarsening is the exact L² projection: P=$P" for P in 1:6
    # Guards a regrid coarsening that loses a degree, does not conserve
    # the GLL integral, or does not undo a refinement.
    t = elementbig(P)
    Ic = vcat(t.interpolation...)
    setprecision(BigFloat, 512) do
        xf = fine_nodes(t)
        wf = [t.weights; t.weights] ./ 2
        for k in 0:P
            @test maximum(abs, t.coarsening * xf .^ k - t.nodes .^ k) < 1e-140
        end
        # Conservation for any data, here a piecewise polynomial of
        # degree P that is not one polynomial (a kink at the centre).
        u = [abs(x) + x^P for x in xf]
        @test abs(sum(t.weights .* (t.coarsening * u)) - sum(wf .* u)) < 1e-140
        # The projection of a coarse polynomial is itself.
        id = [big(i == j) for i in 1:(P + 1), j in 1:(P + 1)]
        @test maximum(abs, t.coarsening * Ic - id) < 1e-140
    end
end

@testset "The tables are the BigFloat ones rounded once: $T" for T in
    (Float64, Float32, Float32x2)
    # Guards a table rounded twice, or built in a narrower type than
    # BigFloat, and an exact 0 or 1 that comes out as a rounding error,
    # which would cost `Stencil1D` its unit and single-source fast paths.
    for P in 1:6
        b, t = elementbig(P), TreeAMR.element_tables(T, P)
        @test t isa TreeAMR.ElementTables{T}
        @test TreeAMR.element_tables(T, P) === t            # memoised
        for f in (:nodes, :weights, :offsets, :quadrature, :restriction, :coarsening)
            @test getfield(t, f) == T.(getfield(b, f))
        end
        @test all(s -> t.interpolation[s] == T.(b.interpolation[s]), 1:2)
        P1 = P + 1
        unit(k) = T.(1:P1 .== k)
        # The face rows of I are unit rows, and so are the rows at the
        # coarse centre node of an even P.
        @test t.interpolation[1][1, :] == unit(1)
        @test t.interpolation[2][P1, :] == unit(P1)
        if iseven(P)
            @test t.interpolation[1][P1, :] == unit(P ÷ 2 + 1)
            @test t.interpolation[2][1, :] == unit(P ÷ 2 + 1)
        end
        # Every exact zero survives in every table.
        for f in (:restriction, :coarsening)
            @test all(iszero.(getfield(b, f)) .== iszero.(getfield(t, f)))
        end
    end
    @test_throws "at least 1" TreeAMR.element_tables(T, 0)
end

@testset "The element centering stores as cells, and P belongs to it: D=$D" for D in (1, 2, 3)
    # Guards an element set laid out differently from a cell-centered one
    # (the state layout and every range must be the same), and a `P`
    # that is lost, defaulted, or carried by a set without elements.
    @test elementcentered(D) == ntuple(_ -> :element, D)
    @test staggers(elementcentered(D)) == ntuple(_ -> 0, D)
    forest = Forest(ntuple(_ -> 2, D); N=8)
    for G in (0, 1)
        fs = FieldSet(forest, 2; G=G, centering=elementcentered(D), P=3)
        @test fs.P == 3
        @test fs.centering == elementcentered(D)
        @test size(fs.work) == size(FieldSet(forest, 2; G=G).work)
        @test TreeAMR.closedranges(fs) == TreeAMR.interiorranges(fs)
    end
    @test FieldSet(forest, 1; G=1, centering=elementcentered(D), P=1).P == 1
    @test FieldSet(forest, 1; G=1).P == 0
    @test FieldSet(forest, 1; G=1, centering=vertexcentered(D)).P == 0
    efs = FieldSet(forest, 1; G=1, centering=elementcentered(D), P=3)
    @test GhostSchedule(efs, Operators(family=ElementLocal)).P == 3
    @test GhostSchedule(FieldSet(forest, 1; G=1),
                        Operators(prolongation=2, restriction=2)).P == 0
end

@testset "An element set is refused where it cannot hold whole elements: D=$D" for D in
    (1, 2, 3)
    # Guards a layout the element operators cannot serve being accepted
    # and found later as wrong data: each refusal names its reason.
    forest = Forest(ntuple(_ -> 2, D); N=8)
    E = elementcentered(D)
    if D > 1
        @test_throws "in every dimension or in none" FieldSet(
            forest, 1; G=1, centering=Base.setindex(E, :cell, 1), P=3)
    end
    @test_throws "needs `P`" FieldSet(forest, 1; G=1, centering=E)
    @test_throws "has no elements" FieldSet(forest, 1; G=1, P=3)
    @test_throws "has no elements" FieldSet(forest, 1; G=1, centering=vertexcentered(D),
                                            P=3)
    @test_throws "at least 1" FieldSet(forest, 1; G=1, centering=E, P=0)
    @test_throws "must be an integer" FieldSet(forest, 1; G=1, centering=E, P=3.0)
    @test_throws "at most one ghost layer" FieldSet(forest, 1; G=2, centering=E, P=3)
    @test_throws "at most one ghost layer" FieldSet(forest, 1; G=ntuple(d -> d == D ? 2 : 1, D),
                                                    centering=E, P=3)
    # N = 8 holds two elements of order 3, but a fine block's half of it
    # does not hold whole parent elements for order 2 (N/2 = 4 nodes
    # against 3) or order 4, nor any element of order 5.
    for P in (2, 4, 5)
        @test_throws "whole parent elements" FieldSet(forest, 1; G=1, centering=E, P=P)
    end
    @test FieldSet(Forest(ntuple(_ -> 1, D); N=12), 1; G=1, centering=E, P=2).P == 2
    # The forest form of the schedule takes the same checks.
    ops = Operators(family=ElementLocal)
    @test_throws "needs `P`" GhostSchedule(forest, ops; G=1, centering=E)
    @test_throws "has no elements" GhostSchedule(forest, ops; G=1, P=3)
    @test_throws "whole parent elements" GhostSchedule(forest, ops; G=1, centering=E, P=2)
    # A schedule for another order is refused by the fill.
    fs = FieldSet(forest, 1; G=1, centering=E, P=3)
    @test_throws "element order P = 3" fill_ghosts!(
        fs, GhostSchedule(forest, ops; G=1, centering=E, P=1))
    # No rotating seam, quarter or half turn, and so no RotationPair.
    if D > 1
        for seam in ((rotating=(1, 2),), (halfturn=(1, 2),))
            rotating = Forest(ntuple(_ -> 2, D); N=8, seam...)
            @test_throws "rotating seam" FieldSet(rotating, 1; G=1, centering=E, P=3)
        end
    end
    @test_throws "cannot hold an element-centered set" RotationPair(fs, fs)
end

@testset "The element family takes no orders, and goes with element sets only: D=$D" for D in
    (1, 2, 3)
    # Guards an element set filled by operators that treat its
    # Gauss–Lobatto nodes as equally spaced points (wrong at every
    # coarse-fine face and regrid, without an error), an equally spaced
    # set handed the element operators, and an order asked of a family
    # whose order is the field set's.
    ops = Operators(family=ElementLocal)
    @test ops.family === ElementLocal
    @test sprint(show, ops) == "Operators(family=ElementLocal)"
    @test TreeAMR.ghost_layers_read(ops) == 0
    @test_throws "takes no prolongation" Operators(family=ElementLocal, prolongation=4)
    @test_throws "takes no restriction" Operators(family=ElementLocal, restriction=2)
    @test_throws "takes no prolongation" Operators(family=ElementLocal, prolongation=2,
                                                   restriction=2)
    @test_throws "takes no vertex_prolongation" Operators(family=ElementLocal,
                                                          vertex_prolongation=2)
    @test_throws "bounded = true is not available" Operators(family=ElementLocal,
                                                             bounded=true)
    # The other families still need their orders.
    @test_throws "no default order" Operators(family=PointValue)
    @test_throws "no default order" Operators(family=Conservative, prolongation=3)

    forest = Forest(ntuple(_ -> 2, D); N=8)
    E = elementcentered(D)
    efs = FieldSet(forest, 1; G=1, centering=E, P=3)
    @test check_operators(efs, ops) === nothing
    @test check_operators(FieldSet(forest, 1; G=0, centering=E, P=3), ops) === nothing
    for other in (Operators(prolongation=2, restriction=2),
                  Operators(prolongation=1, restriction=2, family=Conservative))
        @test_throws "takes Operators(family = ElementLocal)" check_operators(efs, other)
        @test_throws "takes Operators(family = ElementLocal)" GhostSchedule(efs, other)
    end
    for C in (cellcentered(D), vertexcentered(D), facecentered(D, 1))
        fs = FieldSet(forest, 1; G=1, centering=C)
        @test_throws "this field set has none" check_operators(fs, ops)
        @test_throws "this field set has none" GhostSchedule(fs, ops)
    end
    @test_throws "this field set has none" GhostSchedule(forest, ops; G=1)
end

# A forest whose block origins and spacings are dyadic, so that the
# positions the tests compare are exact where they must coincide.
element_forest(T, D; roots=2, N=8) =
    Forest{T}(ntuple(_ -> roots, D); N=N, extents=ntuple(_ -> (zero(T), one(T)), D))

# Every stored index of block `b`, as tuples.
stored_indices(fs) = CartesianIndices(size(fs.work)[1:(ndims(fs.work) - 2)])

@testset "Every position of an element set is its node, bit for bit: D=$D, P=$P, G=$G, $T" for
    (D, P, T) in ((1, 1, Float64), (1, 3, Float64), (2, 1, Float64), (2, 3, Float64),
                  (3, 3, Float64), (2, 3, Float32), (2, 3, Float32x2)),
    G in (0, 1)
    # Guards a position site that still computes cell centres for an
    # element set, or forms the node by a different expression than
    # `coordinates` does: the initial data, the folds, the flagging sweep
    # and the boundary hooks would then disagree with each other and with
    # TreeWaveOLDG about where a node is.
    forest = element_forest(T, D)
    E = elementcentered(D)
    fs = FieldSet(forest, D; G=G, centering=E, P=P)
    t = TreeAMR.element_tables(T, P)
    N, P1 = forest.N, P + 1
    nodepos(b, I) = let o = block_origin(T, forest, blockkey(fs, b)),
                        h = spacing(T, forest, blockkey(fs, b))
        ntuple(d -> o[d] + h * (((I[d] - 1) ÷ P1) * P1 + t.offsets[(I[d] - 1) % P1 + 1]), D)
    end
    owned(b) = CartesianIndices(ntuple(_ -> 1:N, D))
    # `coordinates` is TreeWaveOLDG's `node_position` at every owned
    # point, and a ghost sits at the block's face.
    for b in 1:nblocks(fs)
        o = block_origin(T, forest, blockkey(fs, b))
        h = spacing(T, forest, blockkey(fs, b))
        @test all(I -> coordinates(fs, b, Tuple(I) .+ G) === nodepos(b, Tuple(I)), owned(b))
        if G == 1
            @test coordinates(fs, b, ntuple(_ -> 1, D)) === o
            @test coordinates(fs, b, ntuple(_ -> N + 2, D)) === ntuple(d -> o[d] + h * N, D)
        end
        # Adjacent elements' face nodes coincide inside a block.
        @test all(q -> coordinates(fs, b, ntuple(_ -> G + q * P1, D)) ===
                       coordinates(fs, b, ntuple(_ -> G + q * P1 + 1, D)), 1:(N ÷ P1 - 1))
    end
    # Adjacent blocks' face nodes coincide: the last node of one is the
    # first of its neighbour along each dimension.
    for b in 1:nblocks(fs), c in 1:nblocks(fs), d in 1:D
        ob = block_origin(T, forest, blockkey(fs, b))
        oc = block_origin(T, forest, blockkey(fs, c))
        h = spacing(T, forest, blockkey(fs, b))
        all(e -> e == d ? oc[e] == ob[e] + h * N : oc[e] == ob[e], 1:D) || continue
        i = ntuple(e -> G + 1, D)
        @test coordinates(fs, b, Base.setindex(i, G + N, d))[d] === coordinates(fs, c, i)[d]
    end

    # Both forms of `fill_by_coordinates!` write the positions.
    fill_by_coordinates!((x, v) -> x[v], fs)
    for b in 1:nblocks(fs)
        @test all(I -> ntuple(v -> fs.work[(Tuple(I) .+ G)..., v, b], D) ===
                       coordinates(fs, b, Tuple(I) .+ G), owned(b))
    end
    gs = FieldSet(forest, D; G=G, centering=E, P=P)
    fill_by_coordinates!(AllVariables(x -> x), gs)
    @test gs.work == fs.work
    # The folds and the flagging sweep hand their callbacks the same x.
    mismatch(acc, p) = acc + (p.u === p.x ? 0 : 1)
    @test mesh_fold(mismatch, +, 0, fs) == 0
    # The device's lane split too, run on the CPU.
    @test sum(TreeAMR._block_fold(mismatch, +, 0, fs.work, fs, fs.G, 1:D, nothing, 4)) == 0
    @test sum(first, firing_boxes((w, idx, b, x) ->
                                      x !== ntuple(d -> w[idx..., d, b], length(x)), fs)) == 0

    # The boundary hooks write their x at the faces, every form of them.
    # Same-level copies bring the neighbour's face node, which sits at
    # the same position, so after a fill every stored value is its own
    # position.
    G == 0 && continue
    ops = Operators(family=ElementLocal)
    sched = GhostSchedule(fs, ops)
    for hook in (CellBoundary((x, v, δ) -> x[v]), CellBoundary(AllVariables((x, δ) -> x)),
                 CellBoundary(Relative(AllVariables(x -> x))))
        fs.work .= T(NaN)
        fill_by_coordinates!((x, v) -> x[v], fs)
        fill_ghosts!(fs, sched; boundary=hook)
        for b in 1:nblocks(fs)
            @test all(I -> ntuple(v -> fs.work[Tuple(I)..., v, b], D) ===
                           coordinates(fs, b, Tuple(I)), stored_indices(fs))
        end
    end
    one_ = FieldSet(forest, 1; G=G, centering=E, P=P)
    fill_by_coordinates!((x, v) -> x[1], one_)
    fill_ghosts!(one_, GhostSchedule(one_, ops); boundary=CellBoundary(Relative(x -> x[1])))
    for b in 1:nblocks(one_)
        @test all(I -> one_.work[Tuple(I)..., 1, b] === coordinates(one_, b, Tuple(I))[1],
                  stored_indices(one_))
    end
end

@testset "Equally spaced sets keep their positions bit for bit: D=$D" for D in (1, 2, 3)
    # Guards the position helper changing a cell or vertex position in
    # its last bit, which every recorded number of the suite would move
    # with: it must be the expression these positions always were.
    for T in (Float64, Float32), C in (cellcentered(D), vertexcentered(D),
                                       facecentered(D, 1)), G in (0, 2)
        forest = Forest{T}(ntuple(_ -> 3, D); N=6,
                           extents=ntuple(_ -> (T(-1) / 3, T(5) / 7), D))
        fs = FieldSet(forest, 1; G=G, centering=C)
        c = staggers(fs)
        for b in (1, nblocks(fs)), I in stored_indices(fs)
            o = block_origin(T, forest, blockkey(fs, b))
            h = spacing(T, forest, blockkey(fs, b))
            off = ntuple(d -> c[d] == 1 ? one(h) : one(h) / 2, D)
            @test coordinates(fs, b, Tuple(I)) ===
                  ntuple(d -> o[d] + (I[d] - G - off[d]) * h, D)
        end
    end
end

@testset "The element norm and mass are the GLL quadrature: D=$D, P=$P" for D in (1, 2),
    P in (1, 2, 3)
    # Guards an element set weighted as equally spaced cells, which is
    # not the norm its summation-by-parts property holds in and is not
    # exact for any polynomial beyond the constants on GLL nodes.
    N = 2 * (P + 1)
    forest = Forest(ntuple(_ -> 1, D); N=N, extents=ntuple(_ -> (0.0, 1.0), D))
    refine!(forest, only(forest.leaves))
    refine!(forest, first(forest.leaves))          # two levels below the root
    balance!(forest)
    fs = FieldSet(forest, 1; G=1, centering=elementcentered(D), P=P)
    u = statevector(fs)
    norm1(f) = (fill_by_coordinates!((x, v) -> f(x), fs); gather!(u, fs);
                volume_weighted_norm(fs, u; p=1))
    # |u| of degree 2P-1 in each dimension is integrated exactly, and the
    # volume is the domain's: the unit cube's.
    @test norm1(x -> prod(xd -> 2 + xd^(2P - 1), x)) ≈ (2 + 1 / (2P))^D rtol = 1e-13
    @test total_mass(fs) ≈ (2 + 1 / (2P))^D rtol = 1e-13
    @test norm1(x -> 3.0) ≈ 3 rtol = 1e-14
    # ... but degree 2P is not.
    @test abs(norm1(x -> prod(xd -> 2 + xd^(2P), x)) - (2 + 1 / (2P + 1))^D) > 1e-10
    # The 2-norm squares a polynomial of degree P-1 into one of 2P-2.
    fill_by_coordinates!((x, v) -> prod(xd -> 1 + xd^(P - 1), x), fs)
    gather!(u, fs)
    exact2 = (1 + 2 / P + 1 / (2P - 1))^D
    @test volume_weighted_norm(fs, u; p=2) ≈ sqrt(exact2) rtol = 1e-13
    @test volume_weighted_norm((fs,), u; p=2) == volume_weighted_norm(fs, u; p=2)
    @test volume_weighted_norm(fs, u; p=Inf) == maximum(abs, u)
end

@testset "Point interpolation refuses an element set" begin
    # Guards interpolating Gauss–Lobatto nodes as if they were equally
    # spaced, which would silently return wrong values.
    forest = element_forest(Float64, 2)
    fs = FieldSet(forest, 1; G=1, centering=elementcentered(2), P=3)
    @test_throws "does not take an element-centered set" interpolate(fs, [(0.3, 0.4)],
                                                                      Lagrange(2))
end
