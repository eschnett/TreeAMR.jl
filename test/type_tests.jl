# Element-type genericity: the mesh must run in a caller-chosen float
# type, with no Float64 anywhere in the arithmetic.
#
# The failure mode these guard is code that is generic in name only —
# computing in Float64 and converting at the end. That is invisible in a
# Float64 run and fatal on a device with no hardware fp64.
#
# Both non-default types earn their place, and for opposite reasons:
#
#   Float32   is the *leak detector*. A stray Float64 operand widens the
#             result, so a return type of Float64 names the leak.
#   Float32x2 is the *does it work off the beaten path* detector: a
#             software type, built from two Float32 limbs, which no
#             `Float64` fast path can serve. It cannot detect leaks on
#             its own — MultiFloats promotes Float64 *downward*
#             (`promote_rule(Float32x2, Float64) = Float32x2`), so a leak
#             is silently absorbed and even `isfinite` looks healthy.
#
# Everything here is polynomial. MultiFloats implements no trigonometric
# functions at all (they `error`), so `wave.jl` and the oracles that use
# `sin` are structurally Float64-only; do not try to generalize them.

using MultiFloats: Float32x2

const FLOATTYPES = (Float64, Float32, Float32x2)

# Tolerances track the type instead of being absolute Float64 constants.
# eps(Float32x2) = 1.4e-14, about 46 bits — coarser than Float64, so
# this is not an "at least as accurate" drop-in.
reltol(::Type{T}, k=64) where {T} = k * eps(T)

@testset "A forest carries the type its geometry is computed in" begin
    # Without this parameter the extents are Float64 and every
    # coordinate is computed in Float64 however the field set is stored.
    @test floattype(Forest((2, 2); N=4)) === Float64
    @test floattype(Forest{Float32}((2, 2); N=4)) === Float32
    @test floattype(Forest{Float32x2}((2, 2); N=4)) === Float32x2

    # With no explicit type the extents decide, and integer extents mean
    # Float64 — so every existing caller keeps what it had.
    @test floattype(Forest((2,); N=4, extents=((0, 2),))) === Float64
    @test floattype(Forest((2,); N=4, extents=((0.0, 2.0),))) === Float64
    @test floattype(Forest((2,); N=4, extents=((0.0f0, 2.0f0),))) === Float32

    # `Forest{D}` must keep matching once D is not the only parameter,
    # or every signature in the package would have to change.
    @test Forest((2, 2); N=4) isa Forest{2}
    @test Forest{Float32}((2, 2); N=4) isa Forest{2}

    # The cube check is type relative, not a Float64 constant. A unit
    # square over a 1x2 brick gives anisotropic root spacings; the
    # default extents would not, since they scale with `roots`.
    for T in FLOATTYPES
        @test_throws "blocks must be cubes" Forest{T}((1, 2); N=4,
                                                      extents=((0, 1), (0, 1)))
        @test Forest{T}((1, 2); N=4, extents=((0, 1), (0, 2))) isa Forest{2,T}
    end
end

@testset "Geometry is computed in the forest's type: T=$T, D=$D" for T in FLOATTYPES,
                                                                    D in (1, 2, 3)
    forest = Forest{T}(ntuple(_ -> 2, D); N=4)
    refine!(forest, first(forest.leaves))
    balance!(forest)
    k = first(forest.leaves)

    @test root_spacing(forest) isa T
    @test spacing(forest, 0) isa T
    @test spacing(forest, k) isa T
    @test minimum_spacing(forest) isa T
    @test block_origin(forest, k) isa NTuple{D,T}
    @test block_extent(forest, k) isa NTuple{D,Tuple{T,T}}
    @test block_spacings(forest) isa Vector{T}
    @test block_origins(forest) isa Vector{NTuple{D,T}}

    # Positions come from the field set, whose own element type is the
    # default rather than the forest's.
    fs = FieldSet(forest, 1; G=1)
    @test coordinates(fs, 1, ntuple(_ -> 2, D)) isa NTuple{D,T}
    @test coordinates(FieldSet{Float32}(forest, 1; G=1), 1,
                      ntuple(_ -> 2, D)) isa NTuple{D,Float32}

    # An explicit type still overrides, and still computes in it.
    @test block_spacings(forest, Float32) isa Vector{Float32}
    @test coordinates(Float32, fs, 1, ntuple(_ -> 2, D)) isa NTuple{D,Float32}
end

@testset "Cell centers match exact rational geometry: T=$T, D=$D" for T in FLOATTYPES,
                                                                     D in (1, 2)
    # `leafbox` (oracles.jl) is exact Rational arithmetic in root-cell
    # units and never touches the package's floating-point geometry. The
    # default extents are one unit per root, so root-cell units *are*
    # physical units here, and with N a power of two every cell center is
    # a dyadic rational — exactly representable in all three types. So
    # this is an equality, not an approximation.
    forest = Forest{T}(ntuple(_ -> 2, D); N=4)
    refine!(forest, first(forest.leaves))
    balance!(forest)
    G, N = 1, forest.N
    fs = FieldSet(forest, 1; G=G)

    for (b, k) in enumerate(forest.leaves)
        lo, hi = leafbox(forest, k)
        got = coordinates(fs, b, ntuple(_ -> G + 1, D))
        for d in 1:D
            w = (hi[d] - lo[d]) // N                   # exact cell width
            @test got[d] == T(lo[d] + (1 - 1//2) * w)
        end
    end
end

@testset "fill_by_coordinates! reproduces coordinates bit for bit: T=$T, D=$D" for
        T in FLOATTYPES, D in (1, 2)
    # storage.jl claims the coordinate kernel and `coordinates` agree bit
    # for bit. Until the geometry followed the element type that was only
    # true when both happened to be Float64; assert it for every type.
    forest = Forest{T}(ntuple(_ -> 2, D); N=4)
    refine!(forest, first(forest.leaves))
    balance!(forest)
    G = 1
    fs = FieldSet(forest, 1; G=G)
    fill_by_coordinates!((x, v) -> x[1], fs)

    for b in 1:nblocks(fs)
        interior = interiorview(fs, b, 1)
        for idx in CartesianIndices(interior)
            stored = ntuple(d -> Tuple(idx)[d] + G, D)
            @test interior[idx] === coordinates(fs, b, stored)[1]
        end
    end
end

@testset "Interpolation weights are exact rationals, rounded once" begin
    # Computed in Rational, so these are equalities. Previously the same
    # claims could only be made to a Float64 roundoff tolerance.
    @test TreeAMR.lagrange_weights([0//1, 1//1], 3//4) == [1//4, 3//4]
    @test TreeAMR.lagrange_weights([0//1, 1//1], 1//2) == [1//2, 1//2]
    @test TreeAMR.lagrange_weights([0//1, 1//1, 2//1, 3//1], 3//2) ==
          [-1//16, 9//16, 9//16, -1//16]
    @test TreeAMR.interpolation_weights(1, 4, 5//2, "test") ==
          [-1//16, 9//16, 9//16, -1//16]
    @test TreeAMR.interpolation_weights(1, 4, 4//1, "test") == [0, 0, 0, 1]

    # The conservative family's defining identity: the two subcell weight
    # vectors average to the unit vector on the center cell, so children
    # always average back to their parent. In rational arithmetic that is
    # an algebraic identity and holds exactly at every order.
    for p in 1:2:11
        low, high = TreeAMR.conservative_prolong_weights(p)
        r = (p - 1) ÷ 2
        @test (low .+ high) .// 2 == [t == r ? 1//1 : 0//1 for t in 0:(p - 1)]
    end

    # Rounded once into the stencil, in the schedule's element type.
    for T in FLOATTYPES
        forest = Forest{T}((2, 2); N=8, periodic=(true, true))
        refine!(forest, first(forest.leaves))
        balance!(forest)
        schedule = GhostSchedule(FieldSet(forest, 1; G=2),
                                 Operators(prolongation=4, restriction=4))
        for group in schedule.phase1, d in 1:2
            @test eltype(group.stencils[d].weights) === T
        end
    end
end

@testset "Ghost exchange reproduces representable polynomials: T=$T, D=$D" for
        T in FLOATTYPES, D in (1, 2)
    # The same nested hierarchy the Float64 ghost tests use, so all three
    # transfer kinds are exercised. Non-periodic, because `makepoly` is
    # not a periodic function: the boundary hook supplies the exact
    # values outside the domain.
    forest = nested_forest(Val(D); T=T, N=8)

    for p in (2, 4)
        ops = Operators(prolongation=p, restriction=p)
        @test exchange_error(forest, ops, makepoly(D, p - 1); G=p ÷ 2) < reltol(T, 4096)
    end

    # That the exactness above is not vacuous is asserted as a *ratio*.
    # An absolute threshold cannot work across types: the existing
    # `> 1e-9` is roundoff in Float64 and larger than the entire effect
    # in Float32, while the truncation error being measured is the same
    # number in every type. Stated at order 2, where the margin is three
    # decades wide even in Float32. At order 4 it is not: the degree-4
    # truncation error on this grid is ~3e-5 against a Float32 roundoff
    # floor of ~3e-6, so the sharp order-boundary claims stay in the
    # Float64 suite (ghost_tests.jl), which has the dynamic range for
    # them.
    ops = Operators(prolongation=2, restriction=2)
    @test exchange_error(forest, ops, makepoly(D, 2)) >
          100 * exchange_error(forest, ops, makepoly(D, 1))
end

@testset "A staggered layout is type generic too: T=$T, D=$D, C=$C" for
        T in FLOATTYPES, D in (1, 2), C in (facecentered(D, 1), vertexcentered(D))
    # The centering changes the position arithmetic — a vertex-like
    # dimension subtracts a whole cell where a cell-centered one
    # subtracts half — so the same three claims are made again for a
    # staggered set: exact rational geometry, bit-for-bit agreement
    # between the kernel and `coordinates`, and an exact exchange.
    #
    # Both a partly and a wholly staggered set, because the arithmetic
    # is per dimension and a face field in D = 2 leaves one dimension
    # cell-centered: only `vertexcentered` exercises the vertex-like
    # branch in *every* dimension at once. The vertex-centered wave
    # equation itself is Float64/Float32 only and lives in
    # `gpu_tests.jl`, whose CPU backend runs both — MultiFloats
    # implements no trigonometry, so `wave.jl` cannot be generalized
    # past those two (see the header above).
    c = staggers(C)
    forest = Forest{T}(ntuple(_ -> 2, D); N=8)
    refine!(forest, first(forest.leaves))
    balance!(forest)
    G, N = 1, forest.N
    fs = FieldSet(forest, 1; G=G, centering=C)
    @test eltype(fs.work) === T

    # An owned point sits *on* the block's lower boundary in a
    # vertex-like dimension and half a cell in across it. Dyadic
    # rationals throughout, so these are equalities.
    for (b, k) in enumerate(forest.leaves)
        lo, hi = leafbox(forest, k)
        got = coordinates(fs, b, ntuple(_ -> G + 1, D))
        for d in 1:D
            w = (hi[d] - lo[d]) // N
            @test got[d] == T(lo[d] + (c[d] == 1 ? 0//1 : 1//2) * w)
        end
    end

    fill_by_coordinates!((x, v) -> x[1], fs)
    for b in 1:nblocks(fs)
        owned = interiorview(fs, b, 1)
        for idx in CartesianIndices(owned)
            stored = ntuple(d -> Tuple(idx)[d] + G, D)
            @test owned[idx] === coordinates(fs, b, stored)[1]
        end
    end

    nested = nested_forest(Val(D); T=T, N=8)
    ops = Operators(prolongation=2, restriction=2)
    @test exchange_error(nested, ops, makepoly(D, 1); G=G,
                         centering=C) < reltol(T, 4096)
    @test exchange_error(nested, ops, makepoly(D, 2); G=G, centering=C) >
          100 * exchange_error(nested, ops, makepoly(D, 1); G=G, centering=C)
end

@testset "Nothing widens to Float64: T=$T" for T in FLOATTYPES
    forest = Forest{T}((2, 2); N=4, periodic=(true, true))
    refine!(forest, first(forest.leaves))
    balance!(forest)
    fs = FieldSet(forest, 2; G=1)

    # The element type follows the forest with nothing passed, at both
    # construction sites — which is also what keeps them consistent.
    @test eltype(fs.work) === T
    schedule = GhostSchedule(fs, Operators(prolongation=2, restriction=2))
    @test eltype(schedule.phase1[1].stencils[1].weights) === T

    fill_by_coordinates!((x, v) -> x[1] * v, fs)
    u = statevector(fs)
    gather!(u, fs)

    @test u isa Vector{T}
    @test volume_weighted_norm(fs, u) isa T
    @test volume_weighted_norm(fs, u; p=1) isa T
    @test volume_weighted_norm(fs, u; p=Inf) isa T
    @test total_mass(fs) isa T

    # Inference, which is the direct check that the forest field stayed
    # concrete when Forest gained a parameter: were it abstract, these
    # would go through a dynamic `spacing(fs.forest, ...)` lookup.
    @test (@inferred total_mass(fs)) isa T
    @test (@inferred volume_weighted_norm(fs, u)) isa T
    @test (@inferred coordinates(fs, 1, (2, 2))) isa NTuple{2,T}
    @test isconcretetype(fieldtype(typeof(fs), :forest))
end

@testset "Point interpolation is exact and stays in T: T=$T, D=$D" for
        T in FLOATTYPES, D in (1, 2)
    # The weights are formed in the field set's type from integers, and
    # the stencil start is a floor that `Float32x2` cannot convert to an
    # integer directly; either going through `Float64` would be a
    # widening, and either getting the floor wrong would be a wrong
    # stencil and a lost polynomial.
    forest = nested_forest(Val(D); T=T, N=8)
    poly = makepoly(D, 3)
    dpoly(x, v, a) = sum(0.11 * (a + v) * e * x[a]^(e - 1) for e in 1:3)
    fs = FieldSet(forest, 2; G=2)
    fill_by_coordinates!(poly, fs)
    fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=4, restriction=4));
                 boundary=boundary_by_coordinates(poly))
    xs = [ntuple(d -> T((7j + 3d) % 64 // 16), D) for j in 1:40]
    push!(xs, ntuple(_ -> T(3 // 2), D), ntuple(_ -> T(1 // 8), D))  # in the ball, far
    derivs = (ntuple(_ -> 0, D), ntuple(d -> Int(d == 1), D))
    r = interpolate(fs, xs, Lagrange(4); derivs=derivs,
                    exclude=Ellipsoid(ntuple(_ -> 1.5, D), ntuple(_ -> 0.5, D)))
    @test eltype(r.values) === T
    scale = maximum(abs(T(poly(x, 2))) for x in xs)
    @test maximum(abs(r.values[v, 1, j] - T(poly(xs[j], v)))
                  for j in eachindex(xs), v in 1:2) < reltol(T, 4096) * scale
    @test maximum(abs(r.values[v, 2, j] - T(dpoly(xs[j], v, 1)))
                  for j in eachindex(xs), v in 1:2) < reltol(T, 65536) * scale
    @test any(r.excluded) && !all(r.excluded)
    @test (@inferred Union{Int,Nothing} locate_point(forest, xs[1])) isa Int
end

@testset "Conservative regridding conserves mass in any precision: T=$T, D=$D" for
        T in FLOATTYPES, D in (1, 2)
    # Assert properties, never values: a seeded RNG gives a different
    # stream per type, since MultiFloats has a `rand` of its own.
    rng = MersenneTwister(4242 + D)
    ops = Operators(prolongation=3, restriction=2, family=Conservative)
    forest = Forest{T}(ntuple(_ -> 3, D); N=4, periodic=ntuple(_ -> true, D))
    fs = FieldSet(forest, 1; G=1)
    for b in 1:nblocks(fs)
        interiorview(fs, b, 1) .= rand(rng, T, size(interiorview(fs, b, 1))...)
    end
    before = total_mass(fs)

    for _ in 1:5
        schedule = GhostSchedule(fs, ops)
        flags = flag_blocks(forest) do b, k
            r = rand(rng)
            r < 0.35 && level(k) < 2 ? Refine : r < 0.7 ? Coarsen : Keep
        end
        regrid!(forest, fs => schedule; flags=flags)
        @test isbalanced(forest)
    end
    @test total_mass(fs) ≈ before rtol = reltol(T, 4096)
end

@testset "A field set and schedule of different types say so" begin
    # They must agree exactly: the transfer accumulates in eltype(dest)
    # and reads weights straight from the stencils, so a mismatch would
    # promote in the innermost loop. It used to be a bare MethodError.
    forest = Forest((2, 2); N=4, periodic=(true, true))
    fs = FieldSet{Float32}(forest, 1; G=1)
    schedule = GhostSchedule(FieldSet(forest, 1; G=1),
                             Operators(prolongation=2, restriction=2))
    @test_throws "field set stores Float32" fill_ghosts!(fs, schedule)
end
