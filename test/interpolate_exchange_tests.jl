# Point interpolation over a distributed forest (M7, step 5).
#
# `interpolate` is collective there: each rank passes its own points,
# which are located on the host, routed to the rank that owns their
# block, evaluated by the serial kernel with the block offset, and routed
# back into the caller's order. The claim is that every value and flag is
# the serial one bit for bit, whatever the partition, so the reference is
# a serial `interpolate` over the whole point list, and each simulated
# rank must reproduce its own slice of it. The ranks run as tasks over
# `regrid_exchange_tests.jl`'s rendezvous communicator, whose `alltoallv`
# is defined there; the helpers come from `interpolate_tests.jl` and
# `exchange_tests.jl`.

# Points that probe what routing could get wrong: random ones over the
# domain and, along a periodic dimension, up to half a period beyond it
# (wrapped, perhaps onto another rank's block), and beyond a reflecting
# face (mirrored); every leaf's lower corner, which puts a point on the
# first block of every rank, i.e. on a rank boundary; and the domain's
# corners.
function routing_points(rng, forest::Forest{D}; nrandom=120) where {D}
    lo = ntuple(d -> Float64(forest.extents[d][1]), D)
    hi = ntuple(d -> Float64(forest.extents[d][2]), D)
    w = hi .- lo
    below(d) = forest.periodic[d] || forest.reflecting[d][1] ? w[d] / 2 : 0.0
    above(d) = forest.periodic[d] || forest.reflecting[d][2] ? w[d] / 2 : 0.0
    xs = NTuple{D,Float64}[ntuple(d -> lo[d] - below(d) +
                                       (w[d] + below(d) + above(d)) * rand(rng), D)
                           for _ in 1:nrandom]
    for k in forest.leaves
        push!(xs, ntuple(d -> Float64(block_extent(forest, k)[d][1]), D))
    end
    push!(xs, hi, lo)
    return xs
end

# An uneven split of `1:n` over `P` ranks, rank 1 always empty.
function uneven_parts(n, P)
    weights = [(2, 0, 5, 1, 3)[r % 5 + 1] for r in 0:(P - 1)]
    cuts = [0; round.(Int, n .* cumsum(weights) ./ sum(weights))]
    return [(cuts[r] + 1):cuts[r + 1] for r in 1:P]
end

@testset "A distributed interpolate is the serial one, point for point: D=$D, $kinds, $(nameof(T))" for
        (D, kinds, C, T, Ps) in
        ((1, (:reflect_lo,), cellcentered(1), Float64, (2, 3)),
         (1, (:periodic,), vertexcentered(1), Float64, (:empty,)),
         (2, (:reflect_both, :periodic), vertexcentered(2), Float64, (3,)),
         (2, (:outer, :outer), cellcentered(2), Float64, (5,)),
         (2, (:reflect_hi, :outer), facecentered(2, 1), Float32, (3,)),
         (3, (:reflect_lo, :outer, :periodic), cellcentered(3), Float64, (4,)))
    # A point routed to the wrong rank, evaluated on the wrong local block,
    # or put back in the wrong slot gives a wrong value; one evaluated from
    # other inputs than the serial kernel's gives one that differs in the
    # last bit. Random data over every stored point, ghosts included, so
    # no smooth field hides either.
    serial = faces_forest(kinds; N=D == 3 ? 6 : 8)
    walls = any(d -> any(serial.reflecting[d]), 1:D)
    parity = walls ? EXCHANGE_PARITY : nothing
    G = ghosts_for(C, 4)
    sfs = FieldSet{T}(serial, 3; G=G, centering=C, parity=parity)
    data = T.(rand(MersenneTwister(D), size(sfs.work)...) .- 0.5)
    copyto!(sfs.work, data)
    xs = routing_points(Xoshiro(30 + D), serial)
    mid = ntuple(d -> (serial.extents[d][1] + serial.extents[d][2]) / 2, D)
    ball = Ellipsoid(mid, ntuple(d -> 0.3, D))
    basis = Lagrange(4)
    derivs = (gradient_derivs(D)..., hessian_derivs(D)...)
    full = interpolate(sfs, xs, basis; derivs=derivs, vars=[3, 1], exclude=ball)
    plain = interpolate(sfs, xs, basis)
    @test 0 < count(full.excluded) < length(xs)
    for P in Ps
        P = P === :empty ? nleaves(serial) + 2 : P      # ranks without blocks
        parts = uneven_parts(length(xs), P)
        comms = gather_ranks(P)
        results = on_ranks(P) do r
            forest = Forest(serial.roots; N=serial.N, periodic=serial.periodic,
                            reflecting=serial.reflecting, extents=serial.extents,
                            leaves=serial.leaves, comm=comms[r])
            fs = FieldSet{T}(forest, 3; G=G, centering=C, parity=parity)
            copyto!(fs.work, data[ntuple(_ -> :, D + 1)..., blockrange(forest)])
            mine = xs[parts[r]]
            a = interpolate(fs, mine, basis; derivs=derivs, vars=[3, 1], exclude=ball)
            b = interpolate(fs, mine, basis)
            # Into caller-supplied outputs, over the points as SVector-like
            # vectors rather than tuples.
            vals = fill(T(NaN), 3, length(derivs), length(mine))
            exc = fill(true, length(mine))
            interpolate!(vals, exc, fs, [collect(x) for x in mine], basis; derivs=derivs,
                         vars=1:3)
            (a, b, vals, exc, nblocks(fs))
        end
        P > nleaves(serial) && @test any(res -> res[5] == 0, results)
        @test isempty(parts[2]) && size(results[2][1].values) == (2, length(derivs), 0)
        for (r, (a, b, vals, exc, _)) in enumerate(results)
            js = parts[r]
            @test bitwise_equal(a.values, full.values[:, :, js])
            @test a.excluded == full.excluded[js]
            @test bitwise_equal(b.values, plain.values[:, :, js])
            @test !any(b.excluded)
            ref = interpolate(sfs, xs[js], basis; derivs=derivs)
            @test bitwise_equal(vals, ref.values)
            @test !any(exc)
        end
    end
end

@testset "A distributed interpolate refuses on every rank what any rank refuses" begin
    # A rank that threw while the others went on would leave them waiting
    # in the routing: an outside point, an argument only some ranks' checks
    # refuse, arguments that differ between ranks, and a forest mutated on
    # one rank must each be refused on every rank, before anything is sent.
    f, _ = tensorpoly(2, 2)
    serial = interp_forest(2)
    P = 3
    function attempt(make; mutate=(forest, r) -> nothing)
        comms = gather_ranks(P)
        return on_ranks(P) do r
            forest = Forest(serial.roots; N=serial.N, extents=serial.extents,
                            leaves=serial.leaves, comm=comms[r])
            mutate(forest, r)
            fs = FieldSet(forest, 2; G=2)
            try
                make(fs, r)
                ""
            catch err
                err isa ArgumentError || rethrow()
                err.msg
            end
        end
    end
    inside = [(0.1, 0.2), (-0.7, 0.4)]
    # An outside point on rank 1 only: it names its own point, the others
    # name rank 1's.
    msgs = attempt((fs, r) -> interpolate(fs, r == 2 ? [inside; [(1.5, 0.0)]] : inside,
                                          Lagrange(4)))
    @test all(!isempty, msgs)
    @test startswith(msgs[2], "point 3, (1.5, 0.0), is outside the domain")
    @test occursin("rank(s) 1 of 3 passed a point outside the domain, so it is refused " *
                   "on every rank", msgs[2])
    for r in (1, 3)
        @test occursin("this one (rank $(r - 1)) included", msgs[r])
        @test occursin("On rank 1, point 3, (1.5, 0.0), is outside the domain", msgs[r])
    end
    # Outside points on ranks 0 and 2, rank 1 passing none at all.
    points = ([(NaN, 0.0)], NTuple{2,Float64}[], [(0.1, 0.2), (0.0, -9.0)])
    msgs = attempt((fs, r) -> interpolate(fs, points[r], Lagrange(4)))
    @test startswith(msgs[1], "point 1, (NaN, 0.0)")
    @test startswith(msgs[3], "point 2, (0.0, -9.0)")
    @test occursin("rank(s) 0, 2 of 3", msgs[3])
    @test occursin("On rank 0, point 1, (NaN, 0.0)", msgs[2])
    # An argument only rank 0's checks refuse, through `interpolate`.
    msgs = attempt((fs, r) -> interpolate(fs, inside, Lagrange(4);
                                          derivs=r == 1 ? ((3, 0),) : ((0, 0),)))
    @test occursin("up to second order", msgs[1])
    @test all(r -> occursin("interpolate was refused on rank(s) 0 of 3", msgs[r]), 2:3)
    # Arguments that differ: the variables on rank 2, the region on rank 1.
    msgs = attempt((fs, r) -> interpolate(fs, inside, Lagrange(4); vars=r == 3 ? [2] : [1]))
    @test all(m -> occursin("different layout on rank(s) 2", m) &&
                   occursin("for interpolate, the basis", m), msgs)
    msgs = attempt((fs, r) -> interpolate(fs, inside, Lagrange(4);
                                          exclude=r == 2 ? Ellipsoid((0, 0), (1, 1)) :
                                                  nothing))
    @test all(m -> occursin("different layout on rank(s) 1", m), msgs)
    # A forest refined on rank 2 only.
    msgs = attempt((fs, r) -> interpolate(fs, inside, Lagrange(4));
                   mutate=(forest, r) -> r == 3 && (refine!(forest, [forest.leaves[end]]);
                                                    balance!(forest)))
    @test all(m -> startswith(m, "the forest differs between ranks, so interpolate is " *
                                 "refused on every rank"), msgs)
end
