# Folds over points (M15): `mesh_fold` and `block_fold`, the accumulator
# wrappers `Sum`, `Max` and `Min`, `TreeAMR.combine` and `addto`.
#
# Relies on `poisoned_fieldset` from `state_tests.jl`.

using TreeAMR: _block_fold, fold_lanes, FOLD_LANE_BYTES, REDUCE_LANES, combine

# A refined, periodic mesh in either centering with poisoned ghosts, as
# `poisoned_fieldset` builds for the cell-centered case: a fold that read
# outside the owned points would show the poison as a wrong number.
function fold_fieldset(::Val{D}, nvars, centering; N=8, G=2, seed=11,
                       T=Float64) where {D}
    forest = Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    fs = FieldSet{T}(forest, nvars; G=G, centering=centering(D))
    rng = MersenneTwister(seed)
    fill!(fs.work, T(1e6))
    for b in 1:nblocks(fs), v in 1:nvars
        iv = interiorview(fs, b, v)
        for i in eachindex(iv)
            iv[i] = randn(rng, T)
        end
    end
    return fs
end

# Every owned point of block `b`, in the fold's order: the owned index,
# its stored index, and its position.
function owned_points(fs::FieldSet{T,D}, b) where {T,D}
    return [(Tuple(I), Tuple(I) .+ fs.G, coordinates(fs, b, Tuple(I) .+ fs.G))
            for I in CartesianIndices(ntuple(_ -> fs.forest.N, D))]
end

@testset "A fold visits every owned point once, where coordinates puts it: D=$D, $cname" for
        D in (1, 2, 3), (cname, centering) in (("cell", cellcentered),
                                               ("vertex", vertexcentered))
    # Guards a fold that visits the wrong window — the ghosts, or a
    # vertex-like block's shared upper plane, which would count a physical
    # point twice — and one whose `p.x` drifts from `coordinates` by an
    # ulp, which a shell boundary would turn into a point in the wrong
    # shell. Every field of the point is checked against the tree.
    fs = fold_fieldset(Val(D), 2, centering)
    forest, G = fs.forest, fs.G
    init = (n = Sum(0), xerr = Max(0.0), uerr = Max(0.0), wrong = Sum(0),
            Ilo = Min(typemax(Int)), Ihi = Max(typemin(Int)))
    rf(acc, p) = combine(acc,
        (n = Sum(1),
         xerr = Max(maximum(abs.(p.x .- coordinates(fs, p.b, p.I .+ G)))),
         uerr = Max(maximum(v -> abs(p.u[v] - fs.work[(p.I .+ G)..., v, p.b]), 1:2)),
         wrong = Sum(Int(p.h != spacing(Float64, forest, blockkey(fs, p.b))) +
                     Int(p.level != level(blockkey(fs, p.b)))),
         Ilo = Min(minimum(p.I)), Ihi = Max(maximum(p.I))))
    r = mesh_fold(rf, combine, init, fs)
    @test r.n.value == nblocks(fs) * forest.N^D
    @test r.xerr.value == 0                        # bit for bit
    @test r.uerr.value == 0
    @test r.wrong.value == 0
    @test (r.Ilo.value, r.Ihi.value) == (1, forest.N)

    # Per block, the count is the block's and `p.b` is the block.
    counts = block_fold((acc, p) -> combine(acc, (Sum(1), Max(p.b))), combine,
                        (Sum(0), Max(0)), fs)
    @test [c[1].value for c in counts] == fill(forest.N^D, nblocks(fs))
    @test [c[2].value for c in counts] == 1:nblocks(fs)

    # No poison: the largest value read is a normal deviate's.
    @test mesh_fold((acc, p) -> combine(acc, Max(maximum(abs, p.u))), combine,
                    Max(0.0), fs).value < 10
end

@testset "A fold sees several variables at once, from either array: D=$D" for D in (1, 2, 3)
    # Guards the restriction `block_mapreduce` has — one value at a time
    # — and an off-by-`G` between the working-array form and the
    # state-vector form. Each block's fold is sequential in index order,
    # so a host loop in the same order reproduces it bit for bit.
    fs = fold_fieldset(Val(D), 3, cellcentered)
    oracle = map(1:nblocks(fs)) do b
        foldl(owned_points(fs, b); init=0.0) do s, (_, idx, _)
            s + fs.work[idx..., 2, b] * fs.work[idx..., 3, b]
        end
    end
    rf(acc, p) = combine(acc, Sum(p.u[1] * p.u[2]))
    @test [s.value for s in block_fold(rf, combine, Sum(0.0), fs; vars=2:3)] == oracle

    u = statevector(fs)
    gather!(u, fs)
    @test block_fold(rf, combine, Sum(0.0), fs, u; vars=2:3) ==
          block_fold(rf, combine, Sum(0.0), fs; vars=2:3)
    @test mesh_fold(rf, combine, Sum(0.0), fs, u; vars=2:3) ==
          mesh_fold(rf, combine, Sum(0.0), fs; vars=2:3)

    # `vars` selects what `p.u` holds, not which points are visited.
    @test mesh_fold((acc, p) -> combine(acc, (Sum(1), Sum(length(p.u)))), combine,
                    (Sum(0), Sum(0)), fs; vars=1:0) ==
          (Sum(nblocks(fs) * fs.forest.N^D), Sum(0))
end

@testset "A fold computes what mesh_mapreduce computes: D=$D" for D in (1, 2, 3)
    # One fold, five results, against five reductions: exact for the
    # extrema and the counts, to roundoff for the sums, whose association
    # differs (a sequential fold against `mapreduce`'s pairwise one).
    fs = poisoned_fieldset(Val(D), 2)
    half = 0.5
    init = (sum = Sum(0.0), peak = Max(0.0), low = Min(Inf), count = Sum(0),
            sq = Sum(0.0))
    rf(acc, p) = combine(acc, (sum = Sum(p.u[1]), peak = Max(abs(p.u[1])),
                               low = Min(p.u[2]), count = Sum(Int(abs(p.u[2]) > half)),
                               sq = Sum(p.u[1]^2 + p.u[2]^2)))
    r = mesh_fold(rf, combine, init, fs)
    @test r.sum.value ≈ mesh_mapreduce(identity, +, 0.0, fs; vars=1) rtol = 1e-12
    @test r.peak.value == mesh_mapreduce(abs, max, 0.0, fs; vars=1)
    @test r.low.value == mesh_mapreduce(identity, min, Inf, fs; vars=2)
    @test r.count.value == mesh_mapreduce(x -> abs(x) > half, +, 0, fs; vars=2)
    @test r.sq.value ≈ mesh_mapreduce(x -> x^2, +, 0.0, fs) rtol = 1e-12

    # A volume weight is per point in a fold and per block in a
    # reduction: the same number to roundoff.
    rv(acc, p) = combine(acc, Sum(p.u[1] * p.h^D))
    @test mesh_fold(rv, combine, Sum(0.0), fs).value ≈ total_mass(fs, 1) rtol = 1e-12
end

@testset "A fold bins points by shell and half-shell as a host loop does: D=$D" for
        D in (2, 3)
    # The case the fold exists for: TreeGH's spherical shells about a
    # center, and each shell's two halves by the sign of `(x − c)·v̂`, in
    # one sweep instead of one masked monitor run per shell. The center
    # sits on a vertex plane, so some points have `s == 0` and must land
    # in neither half; a point below the innermost edge is in no shell.
    fs = fold_fieldset(Val(D), 1, vertexcentered)
    c = ntuple(d -> 0.5, D)
    v̂ = ntuple(d -> d == 1 ? 1.0 : 0.0, D)
    edges = (0.05, 0.15, 0.3, 0.45, Inf)
    K = length(edges) - 1
    shellof(r) = count(e -> e <= r, edges[1:K])    # 0 below the innermost edge
    bin = (n = Sum(0), sq = Sum(0.0), peak = Max(0.0))
    init = (shell = ntuple(_ -> bin, K), trail = ntuple(_ -> bin, K),
            lead = ntuple(_ -> bin, K))
    function rf(acc, p)
        d = p.x .- c
        r = sqrt(sum(abs2, d))
        k = 0
        for i in 1:K
            edges[i] <= r < edges[i + 1] && (k = i)
        end
        q = p.u[1]
        x = (n = Sum(1), sq = Sum(q * q), peak = Max(abs(q)))
        s = sum(d .* v̂)
        return (shell = addto(acc.shell, k, x),
                trail = addto(acc.trail, s < 0 ? k : 0, x),
                lead = addto(acc.lead, s > 0 ? k : 0, x))
    end
    # Not `r`: `rf` assigns an `r` of its own, which a local `r` here would
    # turn into one variable shared by every thread running `rf`.
    res = mesh_fold(rf, combine, init, fs)

    n = zeros(Int, 3, K)
    sq = zeros(3, K)
    peak = zeros(3, K)
    onplane = 0
    for b in 1:nblocks(fs), (_, idx, x) in owned_points(fs, b)
        d = x .- c
        k = shellof(sqrt(sum(abs2, d)))
        k == 0 && continue
        q = fs.work[idx..., 1, b]
        s = sum(d .* v̂)
        onplane += s == 0
        for row in (1, s < 0 ? 2 : 0, s > 0 ? 3 : 0)
            row == 0 && continue
            n[row, k] += 1
            sq[row, k] += q * q
            peak[row, k] = max(peak[row, k], abs(q))
        end
    end
    @test onplane > 0                              # the plane is exercised
    for (row, bins) in enumerate((res.shell, res.trail, res.lead)), k in 1:K
        @test bins[k].n.value == n[row, k]
        @test bins[k].peak.value == peak[row, k]
        @test bins[k].sq.value ≈ sq[row, k] rtol = 1e-12
    end
    halves(k) = res.trail[k].n.value + res.lead[k].n.value
    @test all(k -> halves(k) <= res.shell[k].n.value, 1:K)
    @test sum(k -> res.shell[k].n.value - halves(k), 1:K) == onplane
end

@testset "A tag selects blocks as a 0/1 weight does: D=$D" for D in (1, 2, 3)
    # The block regions of TreeGH's row — a level, the boundary band —
    # were 0/1 weights; as a tag they are one bin of the same sweep.
    fs = poisoned_fieldset(Val(D), 1)
    fine(k) = level(k) == 1
    rf(acc, p) = p.tag ? combine(acc, (Sum(1), Sum(p.u[1]), Max(abs(p.u[1])))) : acc
    r = mesh_fold(rf, combine, (Sum(0), Sum(0.0), Max(0.0)), fs; tag=fine)
    w = k -> fine(k) ? 1.0 : 0.0
    @test r[1].value == count(b -> fine(blockkey(fs, b)), 1:nblocks(fs)) * fs.forest.N^D
    @test r[2].value ≈ mesh_mapreduce(identity, +, 0.0, fs; weight=w) rtol = 1e-12
    @test r[3].value == mesh_mapreduce(abs, max, 0.0, fs; weight=w)
end

@testset "The host and lane paths compute the same fold: D=$D, N=$N" for
        (D, N) in ((1, 4), (2, 12), (3, 8))
    # The CPU folds each block in one lane; a device in up to 256, a
    # block with fewer points than lanes leaving the surplus at `init`.
    # Both run on `CPU()`, so their agreement is checked on every run:
    # exactly for the extrema and the count, to roundoff for the sum, and
    # two lane-path calls must give identical bits.
    fs = poisoned_fieldset(Val(D), 2; N=N)
    G = fs.G
    init = (Sum(0), Sum(0.0), Max(-Inf), Min(Inf))
    rf(acc, p) = combine(acc, (Sum(1), Sum(p.u[1] * p.u[2]), Max(p.u[1]), Min(p.u[2])))
    host = block_fold(rf, combine, init, fs)
    @test host == _block_fold(rf, combine, init, fs.work, fs, G, 1:2, nothing, 1)
    for W in (7, 64, REDUCE_LANES, fold_lanes(typeof(init)))
        lanes = _block_fold(rf, combine, init, fs.work, fs, G, 1:2, nothing, W)
        @test isequal(lanes, _block_fold(rf, combine, init, fs.work, fs, G, 1:2, nothing, W))
        @test [l[1] for l in lanes] == [h[1] for h in host]
        @test [l[3] for l in lanes] == [h[3] for h in host]
        @test [l[4] for l in lanes] == [h[4] for h in host]
        @test [l[2].value for l in lanes] ≈ [h[2].value for h in host] rtol = 1e-12
    end
end

@testset "The accumulator wrappers combine as they say, and only with their own kind" begin
    # Guards a fold whose `op` silently combined a sum with a maximum, a
    # bin index that fell outside the bins and landed somewhere, and a
    # NaN from a blown-up run that a maximum skipped.
    @test combine(Sum(1), Sum(2)) === Sum(3)
    @test combine(Max(1.0), Max(2.0)) === Max(2.0)
    @test combine(Min(1.0), Min(2.0)) === Min(1.0)
    @test isnan(combine(Max(NaN), Max(1.0)).value)
    @test isnan(combine(Max(1.0), Max(NaN)).value)
    @test isnan(combine(Min(1.0), Min(NaN)).value)
    @test combine((a = Sum(1), b = (Max(1), Min(1))), (a = Sum(2), b = (Max(0), Min(0)))) ===
          (a = Sum(3), b = (Max(1), Min(0)))
    @test combine((), ()) === ()
    @test_throws MethodError combine(Sum(1), Max(1))
    @test_throws MethodError combine((Sum(1),), (Sum(1), Sum(2)))
    @test_throws MethodError combine((a = Sum(1),), (b = Sum(1),))

    bins = (Sum(0), Sum(0), Sum(0))
    @test addto(bins, 2, Sum(5)) === (Sum(0), Sum(5), Sum(0))
    @test addto(bins, 0, Sum(5)) === bins
    @test addto(bins, 4, Sum(5)) === bins

    # A NaN at one owned point reaches the mesh's maximum.
    fs = poisoned_fieldset(Val(2), 1)
    interiorview(fs, 2, 1)[1] = NaN
    @test isnan(mesh_fold((acc, p) -> combine(acc, Max(abs(p.u[1]))), combine,
                          Max(0.0), fs).value)
end

@testset "The lane count shrinks with the accumulator to fit the buffer" begin
    # Guards a partials buffer of `256 × nblocks × sizeof(acc)` for a
    # large accumulator: TreeGH's 100-odd numbers at 856 blocks would be
    # 175 MB at 256 lanes.
    for R in (Float64, NTuple{7,Float64}, NTuple{100,Float64}, NTuple{126,Float64},
              NTuple{9000,Float64})
        W = fold_lanes(R)
        @test ispow2(W) && 1 <= W <= REDUCE_LANES
        @test W == 1 || W * sizeof(R) <= FOLD_LANE_BYTES
        @test W == REDUCE_LANES || 2W * sizeof(R) > FOLD_LANE_BYTES
    end
    @test fold_lanes(Float64) == REDUCE_LANES
    @test fold_lanes(NTuple{100,Float64}) == 64
    @test fold_lanes(NTuple{9000,Float64}) == 1
end

@testset "A fold follows the field set's element type and is type-stable" begin
    # Guards an fp64 position in a Float32 fold — the leak the
    # type-genericity work went after — and a result inferred as `Any`.
    fs = fold_fieldset(Val(2), 1, vertexcentered; T=Float32)
    init = (Max(-Inf32), Sum(0.0f0))
    rf(acc, p) = combine(acc, (Max(p.x[1]), Sum(p.u[1] * p.h^2)))
    r = @inferred mesh_fold(rf, combine, init, fs)
    @test r isa Tuple{Max{Float32},Sum{Float32}}
    @test r[1].value == maximum(b -> coordinates(fs, b, ntuple(d -> fs.G[d] + fs.forest.N, 2))[1],
                                1:nblocks(fs))
end

@testset "A fold refuses what a kernel cannot hold, and says why" begin
    # Each of these would fail inside a launch — with no context on a
    # device — or deadlock a distributed run if it were decided per rank.
    fs = poisoned_fieldset(Val(2), 3)
    rf(acc, p) = combine(acc, Sum(p.u[1]))
    @test_throws "isbits value" mesh_fold((acc, p) -> acc, (a, b) -> a, [0.0], fs)
    @test_throws "the accumulator's own type" mesh_fold((acc, p) -> Sum(1), combine,
                                                         Sum(0.0), fs)
    @test_throws "tag function" mesh_fold(rf, combine, Sum(0.0), fs; tag=k -> "fine")
    @test_throws "tag function" mesh_fold(rf, combine, Sum(0.0), fs;
                                          tag=k -> level(k) == 1 ? 1 : nothing)
    @test_throws "contiguous range" mesh_fold(rf, combine, Sum(0.0), fs; vars=[1, 3])
    @test_throws "out of range" mesh_fold(rf, combine, Sum(0.0), fs; vars=1:4)
end
