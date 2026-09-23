# M3: the state-vector coupling to ODE integrators.

using KernelAbstractions: @kernel, @index, @Const

# A minimal application kernel: read a ghosted block, write state layout.
@kernel function double_interior!(du, @Const(work), ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                    # (i1..iD, block)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))
    c = ntuple(d -> I[d] + G[d], Val(D))
    du[inner..., 1, b] = 2 * work[c..., 1, b]
end

@testset "State vector layout: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, periodic=ntuple(_ -> true, D))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    fs = FieldSet(forest, 2; G=1)

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
    forest = Forest(ntuple(_ -> 2, D); N=4, periodic=ntuple(_ -> true, D))
    refine!(forest, forest.leaves[1])
    balance!(forest)

    fs = FieldSet(forest, 2; G=1)
    f = (x, v) -> v + sum(x) + prod(x)
    fill_by_coordinates!(f, fs)

    u = statevector(fs)
    @test gather!(u, fs) === u
    @test !all(iszero, u)

    # Scattering into a fresh field set reproduces every interior cell...
    other = FieldSet(forest, 2; G=1)
    @test scatter!(other, u) === other
    @test all(b -> interiorview(other, b) == interiorview(fs, b), 1:nblocks(fs))

    # ... and leaves the ghosts alone: the integrator never sees them.
    stored = forest.N + 2 * other.G[1]
    ghostcells = (stored^D - forest.N^D) * nblocks(fs) * fs.nvars
    @test count(iszero, other.work) == ghostcells

    # A field set with a different ghost width holds the same state:
    # the owned range is `N` wide whatever `G` is.
    lopsided = FieldSet(forest, 2; G=ntuple(d -> d - 1, D))
    @test statelength(lopsided) == statelength(fs)
    scatter!(lopsided, u)
    u3 = statevector(lopsided)
    gather!(u3, lopsided)
    @test u3 == u

    # Round trip is exact, not merely close.
    u2 = statevector(other)
    gather!(u2, other)
    @test u2 == u
end

@testset "map_blocks!: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, periodic=ntuple(_ -> true, D))
    fs = FieldSet(forest, 1; G=1)
    fill!(fs.work, 1.0)

    # Doubling every interior cell touches exactly N^D * nblocks cells.
    du = statevector(fs)
    map_blocks!(double_interior!, fs, statearray(du, fs), fs.work, Val(D), Val(fs.G))
    @test all(==(2.0), du)
    @test length(du) == forest.N^D * nblocks(fs)
end

@testset "Volume-weighted norm: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    fs = FieldSet(forest, 1; G=1)

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

    uniform = Forest(ntuple(_ -> 2, D); N=8, periodic=ntuple(_ -> true, D),
                     extents=ntuple(_ -> (0.0, 1.0), D))
    ufs = FieldSet(uniform, 1; G=1)
    fill_by_coordinates!(g, ufs)
    uu = statevector(ufs)
    gather!(uu, ufs)

    # Both are midpoint quadratures of the same smooth function, so they
    # agree to the quadrature error rather than exactly.
    @test volume_weighted_norm(fs, u) ≈ volume_weighted_norm(ufs, uu) rtol = 1e-3

    # Linf really is the maximum.
    @test volume_weighted_norm(fs, u; p=Inf) ≈ maximum(abs, u)
end

# Per-block reductions (M3 shape, made public in M6).

using TreeAMR: _block_mapreduce_host, _block_mapreduce_device
using KernelAbstractions: CPU

# A refined, periodic mesh with poisoned ghosts: if a reduction picked
# the wrong window, the poison shows up as a wrong number rather than a
# near-miss that a tolerance would swallow.
function poisoned_fieldset(::Val{D}, nvars; N=4, G=2, seed=7) where {D}
    forest = Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    fs = FieldSet(forest, nvars; G=G)
    rng = MersenneTwister(seed)
    fill!(fs.work, 1e6)                       # poison, ghosts included
    for b in 1:nblocks(fs), v in 1:nvars
        iv = interiorview(fs, b, v)
        for i in eachindex(iv)
            iv[i] = randn(rng)
        end
    end
    return fs
end

@testset "Per-block reductions read interiors only: D=$D" for D in (1, 2, 3)
    # Guards the off-by-G that a `g` the caller had to supply invited:
    # the working-array form must skip the ghosts and the state-vector
    # form must not shift by G.
    fs = poisoned_fieldset(Val(D), 2)
    oracle = [sum(sum(interiorview(fs, b, v)) for v in 1:2) for b in 1:nblocks(fs)]

    @test block_mapreduce(identity, +, 0.0, fs) ≈ oracle
    @test maximum(block_mapreduce(abs, max, 0.0, fs)) < 1e5     # no poison read

    u = statevector(fs)
    gather!(u, fs)
    @test block_mapreduce(identity, +, 0.0, fs, u) ≈ oracle

    # One variable at a time, as a scalar and as a range.
    for v in 1:2
        per_v = [sum(interiorview(fs, b, v)) for b in 1:nblocks(fs)]
        @test block_mapreduce(identity, +, 0.0, fs; vars=v) ≈ per_v
        @test block_mapreduce(identity, +, 0.0, fs; vars=v:v) ≈ per_v
        @test block_mapreduce(identity, +, 0.0, fs, u; vars=v) ≈ per_v
    end
end

@testset "Host and kernel reductions compute the same fold: D=$D" for D in (1, 2, 3)
    # The two backends carried separate specifications of the reduction
    # until M6, and nothing checked they agreed — `sum` over a block view
    # is pairwise for an `IndexLinear` view and sequential for an
    # `IndexCartesian` one, and D=1 with a scalar `vars` is the first.
    # Both paths are reachable on `CPU()`, so this is checked on every
    # run and not only where there is a device.
    fs = poisoned_fieldset(Val(D), 2)
    G = fs.G
    half = 0.5
    cases = (("sum", identity, +, 0.0),
             ("max|x|", abs, max, 0.0),
             ("count", x -> abs(x) > half, +, 0))
    for (name, f, op, init) in cases, vars in (1, 1:1, 1:2)
        r = vars isa Integer ? (Int(vars):Int(vars)) : vars
        host = _block_mapreduce_host(f, op, init, fs.work, fs, G, r)
        device = _block_mapreduce_device(f, op, init, fs.work, fs, CPU(), G, r)
        # Exact for `max` and for the integer count; `+` may reassociate.
        if init isa Integer || op === max
            @test host == device
        else
            @test host ≈ device rtol = 1e-12
        end
        @test block_mapreduce(f, op, init, fs; vars=vars) == host
    end
end

@testset "Per-block reductions take their element type from init" begin
    # A predicate count into an `Int` accumulator over a `Float64` field
    # set: the result type follows `init`, not `eltype(fs.work)`.
    fs = poisoned_fieldset(Val(2), 1)
    counts = block_mapreduce(x -> abs(x) > 0.5, +, 0, fs)
    @test counts isa Vector{Int}
    @test counts == [count(x -> abs(x) > 0.5, interiorview(fs, b, 1))
                     for b in 1:nblocks(fs)]

    @test block_mapreduce(identity, +, 0.0f0, fs) isa Vector{Float32}
end

@testset "A variable selection that a kernel cannot take is refused" begin
    # Silently reducing `first:last` instead — which the device path once
    # did — would give a different answer from the host for the same call.
    fs = poisoned_fieldset(Val(2), 3)
    @test_throws "contiguous range" block_mapreduce(identity, +, 0.0, fs; vars=[1, 3])
    @test_throws "contiguous range" block_mapreduce(identity, +, 0.0, fs; vars=1:2:3)
    @test_throws "out of range" block_mapreduce(identity, +, 0.0, fs; vars=0:2)
    @test_throws "out of range" block_mapreduce(identity, +, 0.0, fs; vars=4)
    @test_throws "out of range" block_mapreduce(identity, +, 0.0, fs; vars=1:4)
    # An empty selection is not an error: every block reduces to `init`.
    @test block_mapreduce(identity, +, 0.0, fs; vars=1:0) == zeros(nblocks(fs))
end

@testset "The device fold is reproducible and handles any block size: D=$D, N=$N" for
        (D, N) in ((1, 4), (2, 12), (2, 20), (3, 8))
    # The device path folds each block over `REDUCE_LANES` lanes. A block
    # with fewer cells than lanes leaves the surplus lanes at `init`; one
    # whose cell count is not a multiple of the lane count leaves some
    # lanes a cell short; 512 is an exact multiple. All must agree with
    # the host — exactly for `max` and an integer count, to roundoff for
    # a sum — and two device calls must return identical bits, since the
    # lane fold has a fixed order. Reachable on `CPU()`, so checked here.
    fs = poisoned_fieldset(Val(D), 2; N=N)
    G = fs.G
    half = 0.5
    for (f, op, init, exact) in ((identity, +, 0.0, false), (abs, max, 0.0, true),
                                 (x -> abs(x) > half, +, 0, true))
        host = _block_mapreduce_host(f, op, init, fs.work, fs, G, 1:2)
        dev = _block_mapreduce_device(f, op, init, fs.work, fs, CPU(), G, 1:2)
        @test dev == _block_mapreduce_device(f, op, init, fs.work, fs, CPU(), G, 1:2)
        if exact
            @test host == dev
        else
            @test host ≈ dev rtol = 1e-12
        end
    end
end

@testset "mesh_mapreduce combines the per-block values as M3 did: D=$D" for D in (1, 2, 3)
    # The whole-mesh form must reproduce `sum` over the per-block vector
    # bit for bit — `mapreduce(identity, +, v)` does, `reduce(+, v; init)`
    # does not — or every recorded norm and mass moves in its last bits.
    # The weight scales each block's value before the combination, in
    # the order `total_mass` has always used.
    fs = poisoned_fieldset(Val(D), 2)
    forest = fs.forest
    values = block_mapreduce(identity, +, 0.0, fs)
    @test mesh_mapreduce(identity, +, 0.0, fs) === sum(values)
    @test mesh_mapreduce(abs, max, 0.0, fs) ===
          maximum(block_mapreduce(abs, max, 0.0, fs))
    @test mesh_mapreduce(x -> abs(x) > 0.5, +, 0, fs; vars=1) ==
          sum(block_mapreduce(x -> abs(x) > 0.5, +, 0, fs; vars=1))

    vol(key) = spacing(Float64, forest, key)^D
    weighted = copy(values)
    for b in 1:nblocks(fs)
        weighted[b] *= vol(blockkey(fs, b))
    end
    @test mesh_mapreduce(identity, +, 0.0, fs; weight=vol) === sum(weighted)
    @test total_mass(fs) === mesh_mapreduce(identity, +, 0.0, fs; vars=1, weight=vol)

    # The state-vector form, and the weight converted to the value type
    # before it multiplies, so a Float64 weight does not widen a Float32
    # reduction.
    u = statevector(fs)
    gather!(u, fs)
    @test mesh_mapreduce(identity, +, 0.0, fs, u) ===
          sum(block_mapreduce(identity, +, 0.0, fs, u))
    fs32 = FieldSet{Float32}(forest, 1; G=2)
    fill!(fs32.work, 1.0f0)
    @test mesh_mapreduce(identity, +, 0.0f0, fs32; weight=key -> 0.5) isa Float32
end
