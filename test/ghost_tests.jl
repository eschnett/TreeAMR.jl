# M2: ghost exchange and the interpolation operators.

@testset "Operators" begin
    ops = Operators(prolongation=4, restriction=6)
    @test ops.prolongation == 4
    @test ops.restriction == 6
    @test sprint(show, Operators(prolongation=2, restriction=2)) ==
          "Operators(prolongation=2, restriction=2, family=PointValue)"

    # Both orders are required. There is no order the mesh could default
    # to, because the right one follows from the application's
    # differencing order (see the interface-order rule below).
    @test_throws ArgumentError Operators()
    @test_throws ArgumentError Operators(prolongation=2)
    @test_throws ArgumentError Operators(restriction=2)
    # The message says why, rather than just that a keyword is missing.
    @test_throws "no default order" Operators()
    @test_throws "restriction" Operators(prolongation=2)

    @test_throws ArgumentError Operators(prolongation=1, restriction=2)   # too low
    @test_throws ArgumentError Operators(prolongation=3, restriction=2)   # odd
    @test_throws ArgumentError Operators(prolongation=2, restriction=0)

    # Exact for polynomials of degree < length(nodes), for any nodes --
    # which is what lets restriction shift its stencil off-center near a
    # coarse/fine interface without losing order.
    # The weights are built in exact rational arithmetic, so these are
    # equalities rather than tolerances.
    for nodes in ([0//1, 1//1], [-1//1, 0//1, 1//1, 2//1], [3//1, 4//1, 5//1, 6//1])
        p = length(nodes)
        for deg in 0:(p - 1), x in (-1//4, 1//2, 5//4, 9//2)
            w = TreeAMR.lagrange_weights(nodes, x)
            @test sum(w[i] * nodes[i]^deg for i in 1:p) == x^deg
            @test sum(w) == 1                                 # partition of unity
        end
    end
    # Order 2 gives the familiar weights: 1/4, 3/4 a quarter cell off
    # center, and 1/2, 1/2 at a midpoint (the 2^D average).
    @test TreeAMR.lagrange_weights([0//1, 1//1], 3//4) == [1//4, 3//4]
    @test TreeAMR.lagrange_weights([0//1, 1//1], 1//2) == [1//2, 1//2]
    # Order 4 restriction at a midpoint: the standard -1/16, 9/16 stencil.
    @test TreeAMR.lagrange_weights([0//1, 1//1, 2//1, 3//1], 3//2) ==
          [-1//16, 9//16, 9//16, -1//16]

    # A shifted window is fine as long as the target stays bracketed;
    # shifting past it would be extrapolation, and is refused.
    @test TreeAMR.interpolation_weights(1, 4, 5//2, "test") ==
          [-1//16, 9//16, 9//16, -1//16]
    @test TreeAMR.interpolation_weights(1, 4, 4//1, "test") == [0, 0, 0, 1]
    @test_throws ArgumentError TreeAMR.interpolation_weights(1, 4, 9//2, "test")
    @test_throws ArgumentError TreeAMR.interpolation_weights(1, 4, 1//2, "test")
end

@testset "Conservative operator family" begin
    ops = Operators(prolongation=3, restriction=2, family=Conservative)
    @test ops.family === Conservative
    @test ops.prolongation == 3
    @test occursin("Conservative", sprint(show, ops))
    @test Operators(prolongation=2, restriction=2).family === PointValue

    # Odd stencils, because the reconstruction is centered on the coarse
    # cell rather than a quarter cell off a source center.
    @test Operators(prolongation=1, restriction=2, family=Conservative) isa Operators
    @test Operators(prolongation=5, restriction=2, family=Conservative) isa Operators
    @test_throws ArgumentError Operators(prolongation=2, restriction=2, family=Conservative)
    @test_throws ArgumentError Operators(prolongation=0, restriction=2, family=Conservative)
    # ... and the point-value family still wants even ones.
    @test_throws ArgumentError Operators(prolongation=3, restriction=2)

    # Conservative restriction is the exact volume average, so there is
    # no higher order to ask for.
    @test_throws ArgumentError Operators(prolongation=3, restriction=4, family=Conservative)
    @test_throws "exact volume average" Operators(prolongation=3, restriction=4,
                                                  family=Conservative)

    # The reconstruction weights are a partition of unity, and the two
    # subcell weight vectors average to the unit vector on the centre
    # cell -- which is local conservation, independent of the data.
    for p in (1, 3, 5)
        low, high = TreeAMR.conservative_prolong_weights(p)
        @test length(low) == length(high) == p
        @test sum(low) == 1
        @test sum(high) == 1
        # Exact in rational arithmetic: this is an algebraic identity,
        # not a statement about Float64 roundoff.
        centre = [t == (p - 1) ÷ 2 ? 1//1 : 0//1 for t in 0:(p - 1)]
        @test (low .+ high) .// 2 == centre
    end
    # Order 3 is the familiar ±1/8 conservative slope.
    low, high = TreeAMR.conservative_prolong_weights(3)
    @test low == [1//8, 1, -1//8]
    @test high == [-1//8, 1, 1//8]
    # Order 1 is piecewise constant.
    @test TreeAMR.conservative_prolong_weights(1) == ([1//1], [1//1])
end

@testset "Conservative exchange is exact for cell averages: D=$D" for D in (1, 2)
    # The conservative family reproduces the cell *averages* of a
    # polynomial of degree p-1 exactly, and no further -- the analogue of
    # the point-value family's pointwise exactness.
    for p in (1, 3, 5)
        G = max(1, (p - 1) ÷ 2)
        forest = nested_forest(Val(D); N=8)
        ops = Operators(prolongation=p, restriction=2, family=Conservative)
        @test conservative_exchange_error(forest, ops, scalarpoly(D, p - 1); G=G) < 1e-11
        @test conservative_exchange_error(forest, ops, scalarpoly(D, p); G=G) > 1e-9
    end
end

@testset "Operator/geometry compatibility" begin
    # Each constraint is isolated by pinning the other order low enough
    # that it cannot be the one that trips. The constraints are against
    # the *field set's* ghost width from M8 on, so these go through a
    # field set.
    prolong(p) = Operators(prolongation=p, restriction=2)
    restrict(p) = Operators(prolongation=2, restriction=p)
    set(N, G) = FieldSet(Forest((2,); N=N), 1; G=G)

    # G >= p/2, so a fine block's prolongation stencil fits inside its
    # coarse neighbor's interior plus that neighbor's own ghosts.
    @test check_operators(set(8, 1), prolong(2)) === nothing
    @test_throws ArgumentError check_operators(set(8, 1), prolong(4))
    @test check_operators(set(8, 2), prolong(4)) === nothing
    @test_throws ArgumentError check_operators(set(8, 2), prolong(6))
    @test check_operators(set(8, 3), prolong(6)) === nothing

    # N >= 2G + p/2 - 1, so restriction reads fine *interior* cells only.
    @test check_operators(set(4, 2), restrict(2)) === nothing
    @test_throws ArgumentError check_operators(set(4, 2), restrict(4))
    @test check_operators(set(6, 2), restrict(4)) === nothing

    # The conservative family needs one fewer ghost layer at the same
    # order, because its stencil is centered on a cell rather than
    # straddling one.
    cons(p) = Operators(prolongation=p, restriction=2, family=Conservative)
    @test TreeAMR.ghost_layers_read(cons(3)) == 1
    @test TreeAMR.ghost_layers_read(cons(5)) == 2
    @test TreeAMR.ghost_layers_read(Operators(prolongation=4, restriction=2)) == 2
    @test check_operators(set(8, 1), cons(3)) === nothing
    @test_throws ArgumentError check_operators(set(8, 1), cons(5))
    @test check_operators(set(8, 2), cons(5)) === nothing

    # The constraints are per dimension, against that dimension's own
    # ghost width: wide in x and narrow in y is refused for order 4 by
    # the y bound alone, and the message names the dimension.
    @test check_operators(FieldSet(Forest((2, 2); N=8), 1; G=(2, 2)),
                          prolong(4)) === nothing
    @test_throws "dimension 2" check_operators(FieldSet(Forest((2, 2); N=8), 1;
                                                        G=(2, 1)), prolong(4))

    # Ghost filling is meaningless without ghosts.
    @test_throws ArgumentError GhostSchedule(FieldSet(Forest((2,); N=4), 1; G=0),
                                             Operators(prolongation=2, restriction=2))
    @test_throws "no ghosts" GhostSchedule(FieldSet(Forest((2,); N=4), 1; G=0),
                                           Operators(prolongation=2, restriction=2))

    # The schedule requires operators too, for the same reason.
    @test_throws MethodError GhostSchedule(FieldSet(Forest((2,); N=4), 1; G=1))
end

@testset "The ghost width belongs to the field set" begin
    # The M8 move. A forest that is handed `G` says where it went rather
    # than failing on an unrecognised keyword, and a field set with no
    # `G` says why there is no default -- both are the first thing a
    # caller written against M6 hits.
    @test_throws ArgumentError Forest((2,); N=4, G=1)
    @test_throws "moved from the forest to the field set" Forest((2,); N=4, G=1)
    @test_throws ArgumentError Forest{Float32}((2,); N=4, G=1)
    @test_throws ArgumentError FieldSet(Forest((2,); N=4), 1)
    @test_throws "no default ghost width" FieldSet(Forest((2,); N=4), 1)

    # Per dimension, with a plain integer as the uniform shorthand.
    forest = Forest((2, 2); N=8, periodic=(true, true))
    @test FieldSet(forest, 1; G=2).G == (2, 2)
    @test size(FieldSet(forest, 3; G=(2, 0)).work) == (12, 8, 3, 4)
    @test size(FieldSet(forest, 1; G=0).work) == (8, 8, 1, 4)
    @test_throws ArgumentError FieldSet(forest, 1; G=(1, -1))
    @test_throws "one width per dimension" FieldSet(forest, 1; G=(1, 1, 1))

    # N >= 2G[d], per dimension: N = 8 takes G = 4 but not 5, and the
    # violation in *either* dimension is enough.
    @test FieldSet(forest, 1; G=4) isa FieldSet
    @test_throws ArgumentError FieldSet(forest, 1; G=5)
    @test_throws ArgumentError FieldSet(forest, 1; G=(2, 5))
    @test_throws "N must be >= 2G" FieldSet(forest, 1; G=5)

    # Two field sets over one forest with different widths is the case
    # the move exists for; each gets its own schedule, and a schedule
    # built for one layout is refused for the other.
    ops = Operators(prolongation=2, restriction=2)
    narrow, wide = FieldSet(forest, 1; G=1), FieldSet(forest, 1; G=2)
    @test fill_ghosts!(narrow, GhostSchedule(narrow, ops)) === narrow
    @test fill_ghosts!(wide, GhostSchedule(wide, ops)) === wide
    @test_throws ArgumentError fill_ghosts!(wide, GhostSchedule(narrow, ops))
    @test_throws "built for G=(1, 1)" fill_ghosts!(wide, GhostSchedule(narrow, ops))

    # The forest form of the constructor spells the layout out instead,
    # and produces the same schedule the field-set form does.
    byforest = GhostSchedule(forest, ops; G=2)
    @test byforest.G == wide.G
    @test sprint(show, byforest) == sprint(show, GhostSchedule(wide, ops))
end

@testset "Schedule partitions the ghost cells: D=$D" for D in (1, 2, 3)
    rng = MersenneTwister(500 + D)
    for trial in 1:3
        forest = Forest(ntuple(_ -> 2, D); N=4,
                        periodic=ntuple(d -> isodd(d + trial), D))
        for _ in 1:4
            k = rand(rng, forest.leaves)
            level(k) < 2 && refine!(forest, k)
        end
        balance!(forest)

        fs = FieldSet(forest, 1; G=1)
        schedule = GhostSchedule(fs, Operators(prolongation=2, restriction=2))
        counts = write_counts(schedule)
        G, N = 1, forest.N
        interior = ntuple(_ -> (G + 1):(G + N), D)

        # Every ghost cell is written exactly once: the ghost regions
        # tile the halo with no gaps and no double-writes.
        for b in 1:nleaves(forest)
            slab = view(counts, ntuple(_ -> Colon(), D)..., b)
            for idx in CartesianIndices(slab)
                inside = all(d -> Tuple(idx)[d] in interior[d], 1:D)
                @test slab[idx] == (inside ? 0 : 1)
            end
        end

        # And that accounts for exactly the halo of every block.
        stored = N + 2G
        @test sum(counts) == nleaves(forest) * (stored^D - N^D)
    end
end

@testset "Exchange is exact for polynomials: D=$D" for D in (1, 2, 3)
    # A three-level hierarchy, non-periodic, with the exact solution
    # imposed on the outer boundary: every ghost cell must then hold the
    # polynomial, whether it came from a copy, a restriction, or a
    # prolongation.
    forest = nested_forest(Val(D))
    @test isbalanced(forest)
    @test length(unique(level.(forest.leaves))) >= 3

    ops = Operators(prolongation=2, restriction=2)

    # All three cases really are exercised, so the exactness below is
    # not vacuous.
    counts = transfer_counts(GhostSchedule(FieldSet(forest, 1; G=1), ops))
    @test counts[:copy] > 0
    @test counts[:restrict] > 0
    @test counts[:prolong] > 0

    @test exchange_error(forest, ops, makepoly(D, 1)) < 1e-12   # linear: exact
    @test exchange_error(forest, ops, makepoly(D, 2)) > 1e-6    # quadratic: not

    # Higher order needs wider ghosts and bigger blocks; order p is exact
    # through degree p-1 and no further.
    wide = nested_forest(Val(D); N=8)
    ops4 = Operators(prolongation=4, restriction=4)
    @test exchange_error(wide, ops4, makepoly(D, 3); G=2) < 1e-10
    @test exchange_error(wide, ops4, makepoly(D, 4); G=2) > 1e-8
end

@testset "Order 6 exchange: D=$D" for D in (1, 2)
    forest = nested_forest(Val(D); N=8)
    ops = Operators(prolongation=6, restriction=6)
    @test exchange_error(forest, ops, makepoly(D, 5); G=3) < 1e-9
    @test exchange_error(forest, ops, makepoly(D, 6); G=3) > 1e-9
end

@testset "Three-level corners: D=$D" for D in (1, 2, 3)
    # Levels l-1, l and l+1 meeting is legal under 2:1 balance, and is
    # what forces the prolongation sweep to run coarsest-target-first:
    # the level l+1 blocks prolongate from level l ghosts that were
    # themselves prolongated from level l-1.
    forest = nested_forest(Val(D))
    @test isbalanced(forest)
    @test length(unique(level.(forest.leaves))) >= 3

    ops = Operators(prolongation=2, restriction=2)
    schedule = GhostSchedule(FieldSet(forest, 1; G=1), ops)
    # More than one prolongation sweep, ordered coarsest target first.
    @test length(schedule.levels) >= 2
    @test issorted(schedule.levels)
    @test length(schedule.phase2) == length(schedule.levels)

    @test exchange_error(forest, ops, makepoly(D, 1)) < 1e-12
end

@testset "Periodic wraparound equals an explicit tiling: D=$D" for D in (1, 2, 3)
    # M = 1 is the self-neighbor case: a lone periodic root abuts itself
    # on every side.
    for M in (D == 3 ? (1,) : (1, 2))
        result = periodic_vs_tiled(Val(D), M)
        @test result !== nothing
        result === nothing && continue
        diff, ncells = result
        @test diff == 0.0                          # bit for bit, not just close
        @test ncells > 0
    end
end

@testset "Ghost filling leaves interiors alone: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, periodic=ntuple(_ -> true, D))
    refine!(forest, forest.leaves[1])
    balance!(forest)

    fs = FieldSet(forest, 2; G=1)
    f = makepoly(D, 1)
    fill_by_coordinates!(f, fs)
    before = [copy(interiorview(fs, b)) for b in 1:nblocks(fs)]

    fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=2, restriction=2)))
    @test all(b -> interiorview(fs, b) == before[b], 1:nblocks(fs))

    # Ghosts really were written: nothing is left at its initial zero.
    @test !all(iszero, fs.work)
end

@testset "Schedule staleness" begin
    forest = Forest((2, 2); N=4, periodic=(true, true))
    fs = FieldSet(forest, 1; G=1)
    schedule = GhostSchedule(fs, Operators(prolongation=2, restriction=2))
    @test !isstale(schedule)
    @test fill_ghosts!(fs, schedule) === fs

    # Refining invalidates the schedule, and a same-size round trip must
    # not slip past the check -- which is why the forest tracks a
    # generation rather than just a leaf count.
    gen = generation(forest)
    refine!(forest, forest.leaves[1])
    @test generation(forest) > gen
    @test isstale(schedule)
    coarsen!(forest, forest.leaves[1] |> parentkey)
    @test nleaves(forest) == nblocks(fs)           # same size again ...
    @test isstale(schedule)                        # ... but still stale
    @test_throws ArgumentError fill_ghosts!(fs, schedule)

    rebuilt = GhostSchedule(fs, Operators(prolongation=2, restriction=2))
    @test !isstale(rebuilt)
    @test fill_ghosts!(fs, rebuilt) === fs

    # A field set over a different forest is rejected outright.
    other = Forest((2, 2); N=4, periodic=(true, true))
    @test_throws ArgumentError fill_ghosts!(FieldSet(other, 1; G=1), rebuilt)
end

@testset "Element types" begin
    forest = Forest((2, 2); N=4, periodic=(true, true))
    refine!(forest, forest.leaves[1])
    balance!(forest)

    f = makepoly(2, 1)
    for T in (Float64, Float32)
        fs = FieldSet{T}(forest, 1; G=1)
        schedule = GhostSchedule(fs, Operators(prolongation=2, restriction=2))
        fill_by_coordinates!(f, fs)
        fill_ghosts!(fs, schedule)
        @test eltype(fs.work) == T
        @test all(isfinite, fs.work)
    end

    schedule = GhostSchedule(FieldSet(forest, 1; G=1),
                             Operators(prolongation=2, restriction=2))
    @test occursin("GhostSchedule{Float64,2}", sprint(show, schedule))
    @test occursin("copies", sprint(show, schedule))
end
