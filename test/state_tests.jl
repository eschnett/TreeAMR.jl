# M3: the state-vector coupling to ODE integrators.

using KernelAbstractions: @kernel, @index, @Const

# A minimal application kernel: read a ghosted block, write state layout.
@kernel function double_interior!(du, @Const(work), ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                    # (i1..iD, block)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))
    c = ntuple(d -> I[d] + G, Val(D))
    du[inner..., 1, b] = 2 * work[c..., 1, b]
end

@testset "State vector layout: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, G=1, periodic=ntuple(_ -> true, D))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    fs = FieldSet(forest, 2)

    @test statelength(fs) == forest.N^D * fs.nvars * nblocks(fs)
    u = statevector(fs)
    @test length(u) == statelength(fs)
    @test all(iszero, u)
    @test eltype(u) == Float64

    # The state array shares memory with the flat vector, and its layout
    # matches the working array minus the ghosts.
    sa = statearray(u, fs)
    @test size(sa) == (ntuple(_ -> forest.N, D)..., fs.nvars, nblocks(fs))
    sa[ntuple(_ -> 1, D)..., 1, 1] = 42.0
    @test u[1] == 42.0

    @test_throws DimensionMismatch statearray(zeros(3), fs)
    @test_throws DimensionMismatch scatter!(fs, zeros(3))
end

@testset "scatter!/gather! round trip: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, G=1, periodic=ntuple(_ -> true, D))
    refine!(forest, forest.leaves[1])
    balance!(forest)

    fs = FieldSet(forest, 2)
    f = (x, v) -> v + sum(x) + prod(x)
    fill_by_coordinates!(f, fs)

    u = statevector(fs)
    @test gather!(u, fs) === u
    @test !all(iszero, u)

    # Scattering into a fresh field set reproduces every interior cell...
    other = FieldSet(forest, 2)
    @test scatter!(other, u) === other
    @test all(b -> interiorview(other, b) == interiorview(fs, b), 1:nblocks(fs))

    # ... and leaves the ghosts alone: the integrator never sees them.
    stored = forest.N + 2 * forest.G
    ghostcells = (stored^D - forest.N^D) * nblocks(fs) * fs.nvars
    @test count(iszero, other.work) == ghostcells

    # Round trip is exact, not merely close.
    u2 = statevector(other)
    gather!(u2, other)
    @test u2 == u
end

@testset "map_blocks!: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, G=1, periodic=ntuple(_ -> true, D))
    fs = FieldSet(forest, 1)
    fill!(fs.work, 1.0)

    # Doubling every interior cell touches exactly N^D * nblocks cells.
    du = statevector(fs)
    map_blocks!(double_interior!, fs, statearray(du, fs), fs.work, Val(D), Val(forest.G))
    @test all(==(2.0), du)
    @test length(du) == forest.N^D * nblocks(fs)
end

@testset "Volume-weighted norm: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, G=1, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    fs = FieldSet(forest, 1)

    # A constant field has that constant for its norm, whatever the
    # refinement -- this is what "volume weighted" buys.
    fill_by_coordinates!((x, v) -> 3.0, fs)
    u = statevector(fs)
    gather!(u, fs)
    @test volume_weighted_norm(fs, u) ≈ 3.0
    @test volume_weighted_norm(fs, u; p=Inf) ≈ 3.0
    @test volume_weighted_norm(fs, u; p=1) ≈ 3.0

    @test volume_weighted_norm(fs, zero(u)) == 0.0
    @test volume_weighted_norm(fs, zero(u); p=Inf) == 0.0

    # The weighting is what makes a refined mesh agree with a uniform one
    # on the same function: an unweighted norm would over-count the
    # refined region, which contributes more entries per unit volume.
    g = (x, v) -> sum(x)
    fill_by_coordinates!(g, fs)
    gather!(u, fs)

    uniform = Forest(ntuple(_ -> 2, D); N=8, G=1, periodic=ntuple(_ -> true, D),
                     extents=ntuple(_ -> (0.0, 1.0), D))
    ufs = FieldSet(uniform, 1)
    fill_by_coordinates!(g, ufs)
    uu = statevector(ufs)
    gather!(uu, ufs)

    # Both are midpoint quadratures of the same smooth function, so they
    # agree to the quadrature error rather than exactly.
    @test volume_weighted_norm(fs, u) ≈ volume_weighted_norm(ufs, uu) rtol = 1e-3

    # Linf really is the maximum.
    @test volume_weighted_norm(fs, u; p=Inf) ≈ maximum(abs, u)
end
