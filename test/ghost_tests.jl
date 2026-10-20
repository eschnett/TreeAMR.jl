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

    # There is no blanket "needs ghosts" rule: whether a ghost-free set
    # can have a schedule follows from the orders alone. Point-value
    # order 2 reads a neighbor and is refused by its own bound;
    # conservative order 1 reads nothing outside the containing cell and
    # gets a schedule -- an empty one; centering_tests.jl says what that
    # buys.
    @test_throws ArgumentError GhostSchedule(FieldSet(Forest((2,); N=4), 1; G=0),
                                             Operators(prolongation=2, restriction=2))
    @test_throws "needs G >= 1" GhostSchedule(FieldSet(Forest((2,); N=4), 1; G=0),
                                              Operators(prolongation=2, restriction=2))
    @test GhostSchedule(FieldSet(Forest((2,); N=4), 1; G=0),
                        Operators(prolongation=1, restriction=2,
                                  family=Conservative)) isa GhostSchedule

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
    # Levels l-1, l and l+1 meeting is legal under 2:1 balance, and the
    # prolongation sweep runs coarsest target first, one stage per level:
    # a level l+1 block may prolongate from level l ghosts that were
    # themselves prolongated from level l-1. (Here none does, and the two
    # levels run as one group; see the stage-group testsets below.)
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

@testset "Independent prolongation levels run as one group: D=$D" for D in (2, 3)
    # On a nested mesh whose shells are two blocks thick no prolongation
    # reads a ghost that the level below prolongated, so the levels need
    # not wait for each other; run one after another, the fill's
    # critical path under MPI is the sum over the levels, each packed on
    # one rank. A dependency check that refused too much would leave them
    # apart, and one that ignored the rotation rule would trust windows
    # in a rotated source's virtual frame.
    octant = nested_octant(Val(D))
    @test sort!(unique(level.(octant.leaves))) == 0:3
    ops = Operators(prolongation=6, restriction=6)
    parity = [ntuple(_ -> EvenParity, D)]
    for C in (vertexcentered(D), cellcentered(D))
        s = GhostSchedule(FieldSet(octant, 1; G=3, centering=C, parity=parity), ops)
        @test s.levels == [1, 2, 3]
        @test s.stagegroups == [1:1, 2:4]
        @test occursin("levels 1-3 run together", sprint(show, s))
    end
    # A level with rotated prolongations runs alone, and so does every
    # level of a schedule printed as before.
    quad = rotating_forest(Val(D); rotating=(1, 2))
    s = GhostSchedule(FieldSet(quad, 1; G=2, centering=vertexcentered(D), rotation=[1]),
                      Operators(prolongation=4, restriction=4))
    @test length(s.levels) == 2
    @test all(gs -> any(g -> g.orientation != 0, gs), s.phase2)
    @test s.stagegroups == [1:1, 2:2, 3:3]
    @test !occursin("run together", sprint(show, s))
end

@testset "A level that reads what the level below writes runs after it" begin
    # No mesh the tests build has a prolongation that reads what the
    # level below writes, so the rule is checked on transfer lists made
    # for it, in one dimension. Level 1 (`lo`) writes its target's
    # `1:2` and reads its source's `7:8`; level 2 (`hi`) reads and writes
    # as the case says. A rule that missed a read of a written point — by
    # a local group or a pack, of a target box, an unpack's or a hook
    # region's, or the other way round, a write over a point the level
    # below reads — would let a level run before the one it depends on.
    S(first, starts) =
        (TreeAMR.Stencil1D{Float64}(first, Int32.(starts), ones(2, length(starts))),)
    key(l, r=0) = TreeAMR.GroupKey{1}(:prolong, (-1,), (0,), l, (Int8(0),), r)
    lo, hi, turned, lvl3 = key(1), key(2), key(2, 1), key(3)
    tag(l) = TreeAMR.prolongation_tag(l)
    pairs(kv...) = TreeAMR.TransferPairs{1}(k => (Int32[t], Int32[s]) for (k, t, s) in kv)
    nobound(k) = TreeAMR.UNBOUNDED
    # The groups of phase 1 and levels 1 and 2 (or `levels`) on a rank
    # whose first leaf is 11, `hi` read through `stencil`.
    function grouped(local_, stencil; sent=pairs(), received=pairs(), boundaries=[],
                     levels=(1, 2), upper=hi)
        stencils = Dict(lo => S(1, [7, 7]), upper => stencil)
        return TreeAMR.stage_groups([TreeAMR.PHASE1_TAG; collect(tag.(levels))], local_,
                                    sent, received,
                                    TreeAMR.BoundaryRegion{1}[boundaries...],
                                    k -> stencils[k], nobound, 10)
    end
    apart, together = [1:1, 2:2, 3:3], [1:1, 2:3]
    # Level 2 reads block 1 at `3:5`, beside what level 1 writes there,
    # at `2:4`, over it, and at `1:2` of block 4, another block.
    @test grouped(pairs((lo, 1, 2), (hi, 3, 1)), S(1, [3, 4])) == together
    @test grouped(pairs((lo, 1, 2), (hi, 3, 1)), S(1, [2, 3])) == apart
    @test grouped(pairs((lo, 1, 2), (hi, 3, 4)), S(1, [1, 1])) == together
    # The same write as an unpack, and the same read as a pack, in global
    # leaf indices.
    @test grouped(pairs((hi, 3, 1)), S(1, [2, 3]); received=pairs((lo, 11, 99))) == apart
    @test grouped(pairs((lo, 1, 2)), S(1, [2, 3]); sent=pairs((hi, 99, 11))) == apart
    # A relative hook region on block 1 at `1:2`, which level 2 reads,
    # run after level 1, against one run after phase 1.
    hook(slot, clamp=0) = TreeAMR.BoundaryRegion{1}(Int32(1), (-1,),
                                                    CartesianIndices((1:2,)),
                                                    (Int32(clamp),), slot)
    @test grouped(pairs((lo, 2, 5), (hi, 3, 1)), S(1, [1, 1]);
                  boundaries=[hook(tag(1))]) == apart
    @test grouped(pairs((lo, 2, 5), (hi, 3, 1)), S(1, [1, 1]);
                  boundaries=[hook(TreeAMR.PHASE1_TAG)]) == together
    # Level 2 writing block 3 where level 1 reads it, at `7:8`, or beside
    # it; and writing block 1 where level 1's hook region reads it, at
    # `4`, or beside it.
    @test grouped(pairs((lo, 1, 3), (hi, 3, 5)), S(7, [1, 1])) == apart
    @test grouped(pairs((lo, 1, 3), (hi, 3, 5)), S(1, [1, 1])) == together
    @test grouped(pairs((lo, 2, 5), (hi, 1, 6)), S(3, [1, 1]);
                  boundaries=[hook(tag(1), 4)]) == apart
    @test grouped(pairs((lo, 2, 5), (hi, 1, 6)), S(5, [1, 1]);
                  boundaries=[hook(tag(1), 4)]) == together
    # A rotated level runs alone whatever it reads, and levels that are
    # not consecutive here never group.
    @test grouped(pairs((lo, 1, 2), (turned, 3, 4)), S(1, [1, 1]); upper=turned) == apart
    @test grouped(pairs((lo, 1, 2), (lvl3, 3, 4)), S(1, [1, 1]); upper=lvl3,
                  levels=(1, 3)) == apart
end

@testset "A grouped fill is the same in any order within the group: D=$D" for D in (2, 3)
    # The oracle for the dependency check. Run in reverse, a level that
    # read a ghost the level below writes would read it before it was
    # written — NaN here, from the prefill — and a level that read a
    # hook region filled after the level below would read it NaN even in
    # level order, against the fill with every stage alone. Each fill
    # starts from NaN in every ghost, and the three must agree in every
    # bit and leave no NaN. Serial: a level wrongly grouped would also
    # race in the one parallel loop at more threads, which the suite at
    # eight threads exercises.
    f = (x, v) -> sin(sum(x) + v) + x[1]^2
    relative = CellBoundary(Relative(x -> cos(sum(x))))
    even = [ntuple(_ -> EvenParity, D)]
    ops6, ops4 = Operators(prolongation=6, restriction=6), Operators(prolongation=4,
                                                                     restriction=4)
    octant = nested_octant(Val(D))
    kinds = D == 2 ? (:reflect_lo, :outer) : (:reflect_lo, :outer, :outflow_hi)
    cases = Any[(octant, vertexcentered(D), 3, ops6, boundary_by_coordinates(f), even),
                (octant, cellcentered(D), 3, ops6, relative, even),
                (faces_forest(kinds), vertexcentered(D), 2, ops4, relative, even),
                (nested_forest(Val(D)), cellcentered(D), 1,
                 Operators(prolongation=2, restriction=2), boundary_by_coordinates(f),
                 nothing),
                (rotating_forest(Val(D); rotating=(1, 2)), vertexcentered(D), 2, ops4,
                 boundary_by_coordinates(f), nothing)]
    D == 2 && append!(cases, [(octant, vertexcentered(D), 3, ops6, relative, even),
                              (octant, cellcentered(D), 3, ops6, boundary_by_coordinates(f),
                               even),
                              (faces_forest(kinds), cellcentered(D), 2, ops4,
                               boundary_by_coordinates(f), even)])
    ngrouped = 0
    for (forest, C, G, ops, hook, parity) in cases
        rotating = TreeAMR.hasrotating(forest)
        fs = FieldSet(forest, 1; G=G, centering=C, parity=parity,
                      rotation=rotating ? [1] : nothing)
        s = GhostSchedule(fs, ops)
        @test length(s.stages) >= 3                 # two levels or more to group
        ngrouped += any(r -> length(r) >= 2, s.stagegroups)
        fills = map((:grouped, :reversed, :alone)) do how
            fill!(fs.work, NaN)
            fill_by_coordinates!(f, fs)
            if how === :grouped
                fill_ghosts!(fs, s; boundary=hook)
            elseif how === :reversed
                fill_stagewise!(fs, s, s.stagegroups; boundary=hook, reverse=true)
            else
                fill_stagewise!(fs, s, [i:i for i in eachindex(s.stages)]; boundary=hook)
            end
            @test !any(isnan, fs.work)
            copy(reinterpret(UInt64, vec(fs.work)))
        end
        @test fills[1] == fills[2] == fills[3]
    end
    @test ngrouped >= 3                             # the order was free somewhere
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

@testset "A flat launch recovers every position, past 2³¹ too" begin
    # A device launches the copy kernels over one axis and recovers the
    # position by multiplying with inverses: an off-by-one there writes
    # a neighbouring ghost silently, and an extent of 1 (D = 1, one
    # variable, a width-1 slab) is the inverse's special case.
    for extents in ((5,), (1, 7), (3, 1, 4), (7, 1, 1, 3, 2), (4, 4, 4, 20, 3),
                    (1, 1, 1, 1, 1))
        shape = TreeAMR.LinearShape(extents)
        @test shape isa TreeAMR.LinearShape{length(extents) - 1,Int32}
        @test [TreeAMR.linear_position(shape, n) for n in 1:prod(extents)] ==
              [Tuple(I) for I in vec(CartesianIndices(extents))]
    end
    # Past `typemax(Int32)` items the arithmetic is `Int`, and still exact.
    extents = (39, 39, 39, 20, 30_000)
    shape = TreeAMR.LinearShape(extents)
    @test shape isa TreeAMR.LinearShape{4,Int}
    ci = CartesianIndices(extents)
    for n in (1, 2^31 - 1, 2^31, 2^31 + 1, 2^32 + 12345, prod(extents))
        @test TreeAMR.linear_position(shape, n) == Tuple(ci[n])
    end
end

# A group with every stencil marked not unit: the same transfer, computed
# through the weights.
function weighted(g::TreeAMR.TransferGroup{T,D}) where {T,D}
    stencils = map(s -> TreeAMR.Stencil1D{T}(s.targetfirst, s.srcstart, s.weights, false,
                                             s.single),
                   g.stencils)
    return TreeAMR.TransferGroup{T,D}(g.kind, stencils, g.targetblocks, g.sourceblocks,
                                      g.factorcol, g.orientation, g.plane)
end

# Every value a copy could meet that an arithmetic shortcut could change:
# signed zeros, NaN, infinities, subnormals, ordinary numbers.
function awkward_fill!(work::AbstractArray{T}) where {T}
    specials = T[-zero(T), zero(T), T(NaN), -T(Inf), floatmin(T) / 4, -floatmin(T) / 3]
    for (i, I) in enumerate(eachindex(work))
        work[I] = iszero(i % 7) ? specials[1 + (i ÷ 7) % length(specials)] :
                  T(sin(i))
    end
    return work
end

# A group with every column's `single` row cleared: the same transfer,
# computed as the full tensor-product sum over every weight.
function full_sum(g::TreeAMR.TransferGroup{T,D}) where {T,D}
    stencils = map(s -> TreeAMR.Stencil1D{T}(s.targetfirst, s.srcstart, s.weights, s.unit,
                                             zero(s.single)),
                   g.stencils)
    return TreeAMR.TransferGroup{T,D}(g.kind, stencils, g.targetblocks, g.sourceblocks,
                                      g.factorcol, g.orientation, g.plane, g.bounded)
end

@testset "The flat launch and the copy path fill bit for bit: D=$D" for D in (1, 2, 3)
    # The device's form of the transfer launch (flat, the position from
    # inverses) runs on no device in CI, and the copy path skips the
    # weights: either one going wrong would show as a ghost that is
    # right to roundoff, or off by a sign of zero. Every group of a
    # schedule with copies, restrictions, prolongations, mirrored and
    # turned transfers is run from the same array flat, shaped with the
    # full sum the flat form takes (the data hold a `NaN` and an infinity,
    # which the CPU's one-tap skip drops under a zero weight), and a copy
    # also through its weights, and the arrays must agree in every bit.
    N = D == 3 ? 6 : 8
    setups = Any[]
    two = Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> true, D))
    refine!(two, two.leaves[1]); balance!(two)
    push!(setups, FieldSet(two, 3; G=2))
    push!(setups, FieldSet(two, 3; G=2, centering=vertexcentered(D)))
    kinds = ntuple(d -> d == 1 ? :reflect_hi : :periodic, D)
    walls = faces_forest(kinds; N=N)
    push!(setups, FieldSet(walls, 2; G=2, centering=vertexcentered(D),
                           parity=[ntuple(d -> d == 1 ? OddParity : NoParity, D),
                                   ntuple(d -> d == 1 ? EvenParity : NoParity, D)]))
    if D >= 2
        quad = rotating_forest(Val(D); rotating=(1, 2), N=N)
        push!(setups, FieldSet(quad, D + 1; G=2, centering=cellcentered(D),
                               rotation=vector_rotation(D, (1, 2))))
    end
    ops = Operators(prolongation=4, restriction=4)
    nunit = 0
    for fs in setups
        schedule = GhostSchedule(fs, ops)
        for g in Iterators.flatten((schedule.phase1, schedule.phase2...))
            fillwith(group; flat) = begin
                work = awkward_fill!(similar(fs.work))
                TreeAMR.run_group!(work, work, group, fs.nvars, TreeAMR.CPU(); flat=flat,
                                   factors=fs.factors, rotvars=fs.rotvars)
                reinterpret(UInt64, vec(work))
            end
            unit = all(s -> s.unit, g.stencils)
            nunit += unit
            @test fillwith(g; flat=true) == fillwith(full_sum(g); flat=false)
            unit && @test fillwith(weighted(g); flat=false) == fillwith(g; flat=false)
        end
    end
    @test nunit > 0                                # the copy path was taken
    # A copy at a vertex-like high wall derives the wall plane it does not
    # own by interpolation, so a `:copy` group is not unit by its kind.
    wallsched = GhostSchedule(setups[3], ops)
    @test any(g -> g.kind === :copy && !all(s -> s.unit, g.stencils),
              wallsched.phase1)
end

# Finite data with what a dropped `±0` term could still change: signed
# zeros and subnormals, beside ordinary numbers of both signs.
function finite_fill!(rng, work::AbstractArray{T}) where {T}
    specials = T[-zero(T), zero(T), floatmin(T) / 4, -floatmin(T) / 3]
    for I in eachindex(work)
        work[I] = rand(rng) < 0.1 ? rand(rng, specials) : T(randn(rng))
    end
    return work
end

@testset "Skipping zero-weight taps fills bit for bit: D=$D" for D in (1, 2, 3)
    # On the CPU a column with one nonzero weight takes one tap along its
    # dimension instead of the whole window; a device, the flat launch,
    # takes the full sum. Got wrong — the wrong row, the row of a column a
    # mirror, a clamp or a turn remapped, a product formed in another
    # order — it shows as a ghost that is right to roundoff, or off by a
    # sign of zero, and only along the vertex-like dimensions that have
    # such columns, and the CPU and a device would disagree. Every group
    # of schedules over both centerings, both families, the prolongation
    # orders, reflecting and outflow faces and a rotating seam is run with
    # the skip (shaped) and without it (shaped with `single` cleared, and
    # flat), from the same finite data, and the arrays must agree in every
    # bit.
    rng = Xoshiro(20261019)
    N = 8
    kinds = ((:reflect_hi,), (:reflect_hi, :outflow_lo),
             (:reflect_hi, :reflect_lo, :outflow_hi))[D]
    configs = Any[]
    for C in (vertexcentered(D), cellcentered(D)), p in (2, 4, 6, 8)
        push!(configs, (C, Operators(prolongation=p, restriction=2), p))
    end
    for p in (3, 5, 7)
        push!(configs, (cellcentered(D), Operators(prolongation=p, restriction=2,
                                                   family=Conservative), p))
    end
    push!(configs, (facecentered(D, 1),
                    Operators(prolongation=3, restriction=2, family=Conservative,
                              vertex_prolongation=4), 4))
    nsingle = 0
    for (C, ops, p) in configs
        G = ghosts_for(C, p)
        _, parity = parity_data(Float64, kinds, p)
        setups = Any[FieldSet(faces_forest(kinds; N=N), 2; G=G, centering=C,
                              parity=parity)]
        # The seam reads through the axis map whatever the order, so one
        # order per family turns it: each order is kernels of its own to
        # compile, and the suite is compilation-bound.
        if D >= 2 && p in (3, 4)
            quad = rotating_forest(Val(D); rotating=(1, 2), N=N)
            push!(setups, FieldSet(quad, D + 1; G=G, centering=C,
                                   rotation=vector_rotation(D, (1, 2))))
        end
        for fs in setups
            schedule = GhostSchedule(fs, ops)
            vertexprolong = 0
            data = finite_fill!(rng, similar(fs.work))
            work, full = similar(data), Vector{UInt64}(undef, length(data))
            for g in Iterators.flatten((schedule.phase1, schedule.phase2...))
                fillwith(group; flat) = begin
                    copyto!(work, data)
                    TreeAMR.run_group!(work, work, group, fs.nvars, TreeAMR.CPU();
                                       flat=flat, factors=fs.factors, rotvars=fs.rotvars)
                    reinterpret(UInt64, vec(work))
                end
                skips = any(s -> any(!iszero, s.single), g.stencils)
                nsingle += skips
                g.kind === :prolong && ops.family === PointValue && C[1] === :vertex &&
                    (vertexprolong += skips)
                copyto!(full, fillwith(full_sum(g); flat=false))
                @test fillwith(g; flat=false) == full
                @test fillwith(g; flat=true) == full
            end
            # The vertex-centered prolongations are what the skip is for.
            ops.family === PointValue && C[1] === :vertex && @test vertexprolong > 0
        end
    end
    @test nsingle > 0                              # the one-tap path was taken
end

@testset "The flat scatter and gather match the shaped ones: D=$D" for D in (1, 2, 3)
    # The same for the state copies, which a device also launches flat.
    forest = Forest(ntuple(_ -> 2, D); N=6, periodic=ntuple(_ -> true, D))
    refine!(forest, forest.leaves[1]); balance!(forest)
    fs = FieldSet(forest, 3; G=ntuple(d -> d == 1 ? 1 : 2, D), centering=vertexcentered(D))
    u = awkward_fill!(statevector(fs))
    works = map((false, true)) do flat
        fill!(fs.work, 0)
        TreeAMR.run_over_interiors!(TreeAMR.scatter_kernel!, fs, fs.work, u; flat=flat)
        copy(fs.work)
    end
    @test reinterpret(UInt64, vec(works[1])) == reinterpret(UInt64, vec(works[2]))
    back = map((false, true)) do flat
        v = statevector(fs)
        TreeAMR.run_over_interiors!(TreeAMR.gather_kernel!, fs, v, works[1]; flat=flat)
        reinterpret(UInt64, v)
    end
    @test back[1] == back[2] == reinterpret(UInt64, u)
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
