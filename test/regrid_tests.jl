# M4: regridding -- flag, complete, rebuild, transfer.

const OPS2 = Operators(prolongation=2, restriction=2)

"""A forest with random interior data, for transfer tests."""
function noisy_forest(rng, ::Val{D}; roots=3, N=4, G=1, periodic=true, nvars=1) where {D}
    forest = Forest(ntuple(_ -> roots, D); N=N, G=G,
                    periodic=ntuple(_ -> periodic, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    fs = FieldSet(forest, nvars)
    for b in 1:nblocks(fs), v in 1:nvars
        interiorview(fs, b, v) .= rand(rng, size(interiorview(fs, b, v))...)
    end
    return forest, fs
end

@testset "complete_marks: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, G=1, periodic=ntuple(_ -> true, D))
    before = copy(forest.leaves)

    # Keep everywhere is a no-op.
    keep = fill(Keep, nleaves(forest))
    @test complete_marks(forest, keep) == before
    @test forest.leaves == before                      # never modifies the forest

    # Refining one block adds its 2^D children, and keeps the array sorted.
    flags = copy(keep)
    flags[1] = Refine
    planned = complete_marks(forest, flags)
    @test length(planned) == length(before) - 1 + 2^D
    @test issorted(planned)
    @test allunique(planned)
    @test all(c -> c in planned, childkeys(before[1]))

    @test_throws DimensionMismatch complete_marks(forest, fill(Keep, nleaves(forest) + 1))
end

@testset "Coarsening needs a complete unanimous sibling group: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, G=1, periodic=ntuple(_ -> true, D))
    parent = forest.leaves[1]
    refine!(forest, parent)
    balance!(forest)
    children = [k for k in forest.leaves if isancestor(parent, k)]
    @test length(children) == 2^D

    # One sibling short of unanimous: nothing coarsens.
    flags = [(k in children[1:(end - 1)]) ? Coarsen : Keep for k in forest.leaves]
    @test complete_marks(forest, flags) == forest.leaves

    # All siblings agree: the group collapses back to the parent.
    flags = [(k in children) ? Coarsen : Keep for k in forest.leaves]
    planned = complete_marks(forest, flags)
    @test parent in planned
    @test !any(c -> c in planned, children)
    @test issorted(planned)

    # A root-level block cannot coarsen; asking is simply ignored.
    flat = Forest(ntuple(_ -> 2, D); N=4, G=1, periodic=ntuple(_ -> true, D))
    @test complete_marks(flat, fill(Coarsen, nleaves(flat))) == flat.leaves
end

@testset "Marks are completed for 2:1 balance: D=$D" for D in (1, 2)
    # Refining one corner of a brick twice would leave a 2-level jump
    # against its neighbors, so completion must refine them too.
    forest = Forest(ntuple(_ -> 4, D); N=4, G=1)
    refine!(forest, last(filter(k -> k.root == 0, forest.leaves)))
    balance!(forest)
    @test isbalanced(forest)

    deep = last(filter(k -> level(k) == 1, forest.leaves))
    flags = [k == deep ? Refine : Keep for k in forest.leaves]
    planned = complete_marks(forest, flags)

    scratch = Forest(ntuple(_ -> 4, D); N=4, G=1)
    empty!(scratch.leaves)
    append!(scratch.leaves, planned)
    @test isbalanced(scratch)
    # More blocks than the bare refinement asks for: the ripple is real.
    @test length(planned) > length(forest.leaves) - 1 + 2^D
end

@testset "regrid! mechanics: D=$D" for D in (1, 2)
    forest = Forest(ntuple(_ -> 4, D); N=4, G=1, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    fs = FieldSet(forest, 2)
    fill_by_coordinates!((x, v) -> v + sum(x), fs)
    schedule = GhostSchedule(forest, OPS2)

    # No flags set: nothing happens, and the schedule stays usable.
    @test regrid!(forest, fs, schedule; flags=fill(Keep, nleaves(forest))) == false
    @test !isstale(schedule)

    before = nleaves(forest)
    gen = generation(forest)
    flags = flag_blocks((b, k) -> block_extent(forest, k)[1][2] <= 0.5 ? Refine : Keep, forest)
    @test regrid!(forest, fs, schedule; flags=flags) == true

    @test nleaves(forest) > before
    @test generation(forest) > gen
    @test isbalanced(forest)
    @test issorted(forest.leaves)
    @test isstale(schedule)                            # must be rebuilt
    # Storage was reallocated to match, and the field set stayed valid.
    @test nblocks(fs) == nleaves(forest)
    @test size(fs.work)[end] == nleaves(forest)
    @test all(isfinite, fs.work)

    rebuilt = GhostSchedule(forest, OPS2)
    @test !isstale(rebuilt)
    @test fill_ghosts!(fs, rebuilt) === fs
end

@testset "regrid! argument checking" begin
    forest = Forest((4,); N=4, G=1, periodic=(true,))
    fs = FieldSet(forest, 1)
    schedule = GhostSchedule(forest, OPS2)
    keep = fill(Keep, nleaves(forest))

    @test_throws DimensionMismatch regrid!(forest, fs, schedule;
                                           flags=fill(Keep, nleaves(forest) + 1))

    other = Forest((4,); N=4, G=1, periodic=(true,))
    @test_throws ArgumentError regrid!(other, fs, GhostSchedule(other, OPS2); flags=keep)
    @test_throws ArgumentError regrid!(forest, FieldSet(other, 1), schedule; flags=keep)

    # A schedule that predates a tree change cannot be trusted to fill
    # the ghosts the transfer reads.
    refine!(forest, forest.leaves[1])
    @test_throws ArgumentError regrid!(forest, fs, schedule;
                                       flags=fill(Keep, nleaves(forest)))
end

@testset "Transfer refuses a two-level jump" begin
    # Defensive: a single regrid may move a block by at most one level,
    # which holds whenever the previous tree was balanced. If that
    # invariant were ever broken, the transfer must say so rather than
    # silently produce wrong data.
    forest = Forest((2,); N=4, G=1, periodic=(true,))
    old = copy(forest.leaves)
    grandchildren = collect(Iterators.flatten(childkeys(c) for c in childkeys(old[1])))
    new = sort!(vcat(grandchildren, old[2:end]))
    @test_throws ArgumentError TreeAMR.transfer_groups(Float64, forest, old, new, OPS2)
end

@testset "Untouched blocks are copied bit-exactly: D=$D" for D in (1, 2)
    rng = MersenneTwister(700 + D)
    forest, fs = noisy_forest(rng, Val(D); roots=4)
    schedule = GhostSchedule(forest, OPS2)

    untouched = [k for k in forest.leaves if block_extent(forest, k)[1][1] >= 0.5]
    saved = Dict(k => copy(interiorview(fs, findfirst(==(k), forest.leaves), 1))
                 for k in untouched)

    flags = flag_blocks((b, k) -> block_extent(forest, k)[1][2] <= 0.25 ? Refine : Keep,
                        forest)
    @test regrid!(forest, fs, schedule; flags=flags)

    for k in untouched
        b = findfirst(==(k), forest.leaves)
        @test b !== nothing
        @test interiorview(fs, b, 1) == saved[k]       # exactly, not approximately
    end
end

@testset "Coarsening conserves any field exactly: D=$D" for D in (1, 2, 3)
    # Restriction at order 2 is the 2^D average, so a coarse cell is the
    # exact mean of its children and the volume integral is untouched --
    # whatever the data.
    rng = MersenneTwister(800 + D)
    forest = Forest(ntuple(_ -> 2, D); N=4, G=1, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    refine!(forest, copy(forest.leaves))
    balance!(forest)

    fs = FieldSet(forest, 1)
    for b in 1:nblocks(fs)
        interiorview(fs, b, 1) .= rand(rng, size(interiorview(fs, b, 1))...)
    end
    before = total_mass(fs)

    fine = nleaves(forest)
    @test regrid!(forest, fs, GhostSchedule(forest, OPS2);
                  flags=fill(Coarsen, nleaves(forest)))
    @test nleaves(forest) < fine
    @test total_mass(fs) ≈ before rtol = 1e-14
end

@testset "Regridding conserves representable fields: D=$D" for D in (1, 2, 3)
    # For a field the operators reproduce exactly, prolongation is exact
    # pointwise and midpoint quadrature is exact, so the volume integral
    # survives regridding to roundoff.
    rng = MersenneTwister(900 + D)

    # A constant is representable at any order, on a periodic domain.
    forest = Forest(ntuple(_ -> 3, D); N=4, G=1, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    fs = FieldSet(forest, 1)
    fill_by_coordinates!((x, v) -> 2.0, fs)
    before = total_mass(fs)
    for _ in 1:5
        schedule = GhostSchedule(forest, OPS2)
        flags = flag_blocks(forest) do b, k
            r = rand(rng)
            r < 0.35 && level(k) < 3 ? Refine : r < 0.7 ? Coarsen : Keep
        end
        regrid!(forest, fs, schedule; flags=flags)
        @test isbalanced(forest)
    end
    @test total_mass(fs) ≈ before rtol = 1e-12

    # A linear field is representable at order 2 -- but only where it is
    # single valued, so this one needs a non-periodic domain with the
    # exact solution on the boundary. (A polynomial is discontinuous
    # across a periodic seam, so mass would legitimately drift there.)
    linear = (x, v) -> 1.0 + sum(x)
    open = Forest(ntuple(_ -> 3, D); N=4, G=1, extents=ntuple(_ -> (0.0, 1.0), D))
    ofs = FieldSet(open, 1)
    fill_by_coordinates!(linear, ofs)
    before = total_mass(ofs)
    for _ in 1:5
        schedule = GhostSchedule(open, OPS2)
        flags = flag_blocks(open) do b, k
            r = rand(rng)
            r < 0.35 && level(k) < 3 ? Refine : r < 0.7 ? Coarsen : Keep
        end
        regrid!(open, ofs, schedule; flags=flags,
                boundary=boundary_by_coordinates(linear))
        @test isbalanced(open)
    end
    @test total_mass(ofs) ≈ before rtol = 1e-12
end

@testset "Conservative family: transfer conserves any field: D=$D" for D in (1, 2, 3)
    # The point of the conservative family. Prolongation reconstructs
    # over the coarse cell preserving its average, so the children always
    # average back to their parent -- exactly, for arbitrary data, not
    # merely for fields the operators reproduce.
    rng = MersenneTwister(1100 + D)
    for p in (1, 3)
        G = max(1, (p - 1) ÷ 2)
        ops = Operators(prolongation=p, restriction=2, family=Conservative)
        forest = Forest(ntuple(_ -> 3, D); N=4, G=G, periodic=ntuple(_ -> true, D),
                        extents=ntuple(_ -> (0.0, 1.0), D))
        fs = FieldSet(forest, 1)
        for b in 1:nblocks(fs)
            interiorview(fs, b, 1) .= rand(rng, size(interiorview(fs, b, 1))...)
        end
        before = total_mass(fs)

        for _ in 1:5
            schedule = GhostSchedule(forest, ops)
            flags = flag_blocks(forest) do b, k
                r = rand(rng)
                r < 0.35 && level(k) < 3 ? Refine : r < 0.7 ? Coarsen : Keep
            end
            regrid!(forest, fs, schedule; flags=flags)
            @test isbalanced(forest)
        end
        @test total_mass(fs) ≈ before rtol = 1e-12
    end
end

@testset "Point-value transfer does not conserve arbitrary data" begin
    # The contrast that makes the family selection meaningful: with
    # point-value operators the same random field drifts, because
    # prolongation draws on neighbour values through the parent's ghosts
    # without those neighbours giving anything up.
    rng = MersenneTwister(1234)
    ops = Operators(prolongation=2, restriction=2)
    forest = Forest((3,); N=4, G=1, periodic=(true,), extents=((0.0, 1.0),))
    fs = FieldSet(forest, 1)
    for b in 1:nblocks(fs)
        interiorview(fs, b, 1) .= rand(rng, size(interiorview(fs, b, 1))...)
    end
    before = total_mass(fs)
    for _ in 1:5
        schedule = GhostSchedule(forest, ops)
        flags = flag_blocks(forest) do b, k
            r = rand(rng)
            r < 0.35 && level(k) < 3 ? Refine : r < 0.7 ? Coarsen : Keep
        end
        regrid!(forest, fs, schedule; flags=flags)
    end
    @test !isapprox(total_mass(fs), before; rtol=1e-8)
end

@testset "Several field sets regrid together: D=$D" for D in (1, 2)
    forest = Forest(ntuple(_ -> 3, D); N=4, G=1, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    state = FieldSet(forest, 2)
    aux = FieldSet(forest, 1)
    fill_by_coordinates!((x, v) -> 2.0, state)
    fill_by_coordinates!((x, v) -> 5.0, aux)

    schedule = GhostSchedule(forest, OPS2)
    flags = flag_blocks((b, k) -> block_extent(forest, k)[1][2] <= 0.4 ? Refine : Keep,
                        forest)
    @test regrid!(forest, [state, aux], schedule; flags=flags)

    @test nblocks(state) == nleaves(forest)
    @test nblocks(aux) == nleaves(forest)
    # Constants survive transfer exactly, in every field set.
    for b in 1:nblocks(state)
        @test all(≈(2.0), interiorview(state, b, 1))
        @test all(≈(5.0), interiorview(aux, b, 1))
    end
end

@testset "transfer=false rebuilds the mesh without moving data: D=$D" for D in (1, 2)
    forest = Forest(ntuple(_ -> 3, D); N=4, G=1, periodic=ntuple(_ -> true, D))
    fs = FieldSet(forest, 1)
    fill_by_coordinates!((x, v) -> 7.0, fs)
    schedule = GhostSchedule(forest, OPS2)

    flags = flag_blocks((b, k) -> b == 1 ? Refine : Keep, forest)
    @test regrid!(forest, fs, schedule; flags=flags, transfer=false)
    @test nblocks(fs) == nleaves(forest)
    @test all(iszero, fs.work)                         # storage is fresh, not carried
end

@testset "Initial-data cycle: D=$D" for D in (1, 2)
    L = 1.0
    bump = (x, v) -> exp(-sum((x[d] - 0.5L)^2 for d in 1:D) / (2 * 0.06^2))
    forest = Forest(ntuple(_ -> 4, D); N=4, G=1, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, L), D))
    fs = FieldSet(forest, 1)

    # Refine towards the bump, in nested shells.
    function flag(b, k)
        ext = block_extent(forest, k)
        c = ntuple(d -> (ext[d][1] + ext[d][2]) / 2, D)
        r = sqrt(sum((c[d] - 0.5L)^2 for d in 1:D))
        want = r < 0.10 ? 3 : r < 0.20 ? 2 : r < 0.32 ? 1 : 0
        return level(k) < want ? Refine : level(k) > want ? Coarsen : Keep
    end

    schedule, passes, converged = adapt_to_initial_data!(fs, OPS2; initial=bump,
                                                         flag=flag, maxpasses=12)
    @test converged
    @test passes > 1                                   # it really did adapt
    @test isbalanced(forest)
    @test maxlevel(forest) == 3
    @test !isstale(schedule)

    # The hierarchy is a fixed point: running the cycle again converges
    # immediately without changing the mesh.
    leaves = copy(forest.leaves)
    _, again, converged2 = adapt_to_initial_data!(fs, OPS2; initial=bump, flag=flag)
    @test converged2
    @test again == 1
    @test forest.leaves == leaves

    # Data is the *re-evaluated* initial data, not something interpolated
    # from the coarse mesh it started on.
    for b in 1:nblocks(fs)
        k = blockkey(fs, b)
        block = blockview(fs, b, 1)
        for idx in CartesianIndices(ntuple(_ -> (forest.G + 1):(forest.G + forest.N), D))
            @test block[idx] ≈ bump(cell_center(forest, k, Tuple(idx)), 1)
        end
    end
end
