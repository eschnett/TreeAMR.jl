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
