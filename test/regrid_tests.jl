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

# --- Buffering (CODE.md regridding step 2) ----------------------------

"""A uniform periodic brick and its centre block, for buffer tests."""
function buffer_forest(::Val{D}; roots=3, N=8, G=1) where {D}
    forest = Forest(ntuple(_ -> roots, D); N=N, G=G, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    mid = only(filter(k -> root_position(forest, k.root) == ntuple(_ -> roots ÷ 2, D),
                      forest.leaves))
    return forest, mid
end

"""The leaves `neighbor_keys` reaches from `k` over the directions `δs`."""
neighbors_over(forest, k, δs) =
    unique(reduce(vcat, [neighbor_keys(forest, k, δ) for δ in δs]))

"""Which leaves of `forest` the buffer moved, and to what."""
function buffer_recruits(forest, flags, buffer)
    plain = RegridFlag[TreeAMR.markflag(m) for m in flags]
    marks = buffered_flags(forest, flags, buffer)
    return Dict(forest.leaves[b] => marks[b]
                for b in eachindex(marks) if marks[b] !== plain[b])
end

"""Fill every interior cell from a fresh seeded RNG, in block order."""
function fill_noise!(fs, seed)
    rng = MersenneTwister(seed)
    for b in 1:nblocks(fs), v in 1:fs.nvars
        interiorview(fs, b, v) .= rand(rng, size(interiorview(fs, b, v))...)
    end
    return fs
end

@testset "buffer=0 changes nothing: D=$D" for D in (1, 2)
    # The default must reproduce the unbuffered behaviour exactly -- both
    # the completed marks and the transferred data, bit for bit.
    build() = begin
        f = Forest(ntuple(_ -> 3, D); N=4, G=1, periodic=ntuple(_ -> true, D),
                   extents=ntuple(_ -> (0.0, 1.0), D))
        refine!(f, f.leaves[1])
        balance!(f)
        f
    end
    forest = build()
    fs = fill_noise!(FieldSet(forest, 1), 1300 + D)

    rng = MersenneTwister(1400 + D)
    flags = flag_blocks(forest) do b, k
        r = rand(rng)
        r < 0.3 ? Refine : r < 0.6 ? Coarsen : Keep
    end
    @test complete_marks(forest, flags; buffer=0) == complete_marks(forest, flags)
    # A box spanning the whole interior is what an omitted box means.
    boxed = [(f, ntuple(_ -> 1:(forest.N), D)) for f in flags]
    @test complete_marks(forest, boxed; buffer=0) == complete_marks(forest, flags)

    # And the same through regrid!, data included.
    twin = build()
    tfs = fill_noise!(FieldSet(twin, 1), 1300 + D)
    @test twin.leaves == forest.leaves && tfs.work == fs.work
    @test regrid!(forest, fs, GhostSchedule(forest, OPS2); flags=flags)
    @test regrid!(twin, tfs, GhostSchedule(twin, OPS2); flags=flags, buffer=0)
    @test twin.leaves == forest.leaves
    @test tfs.work == fs.work                          # exactly
end

@testset "A box well inside a block recruits nobody: D=$D" for D in (1, 2, 3)
    forest, mid = buffer_forest(Val(D))
    flags = flag_blocks((b, k) -> k == mid ? (Refine, ntuple(_ -> 4:5, D)) : Keep, forest)
    @test isempty(buffer_recruits(forest, flags, 2))   # margin 3 > buffer 2
    @test !isempty(buffer_recruits(forest, flags, 4))  # a wider one does reach out
end

@testset "A box near one face recruits only that face: D=$D" for D in (1, 2, 3)
    forest, mid = buffer_forest(Val(D))
    # Near the low-x face, mid-range in every other dimension.
    box = ntuple(d -> d == 1 ? (1:2) : (4:5), D)
    flags = flag_blocks((b, k) -> k == mid ? (Refine, box) : Keep, forest)
    recruits = buffer_recruits(forest, flags, 2)

    lox = ntuple(d -> d == 1 ? -1 : 0, D)
    @test sort(collect(keys(recruits))) == sort(neighbors_over(forest, mid, (lox,)))
    @test all(==(Refine), values(recruits))            # equal level: promoted
    if D >= 2
        # In particular no corner: the box never leaves the block in y.
        corner = only(neighbor_keys(forest, mid, ntuple(d -> d <= 2 ? -1 : 0, D)))
        @test !(corner in keys(recruits))
    end
end

@testset "A box near a corner recruits that corner's side: D=$D" for D in (2, 3)
    forest, mid = buffer_forest(Val(D))
    flags = flag_blocks((b, k) -> k == mid ? (Refine, ntuple(_ -> 1:2, D)) : Keep, forest)
    recruits = buffer_recruits(forest, flags, 2)

    # Every direction with only nonpositive components -- faces, edges,
    # and the corner on the low side -- and nothing on the high side.
    δs = filter(δ -> all(<=(0), δ), alldirections(Val(D)))
    @test length(δs) == 2^D - 1
    @test sort(collect(keys(recruits))) == sort(neighbors_over(forest, mid, δs))
    @test all(==(Refine), values(recruits))

    # The conjunction is the point: near low-x but mid-y, the (-1,-1,...)
    # corner is *not* reached, even though the same buffer reaches -x.
    corner = only(neighbor_keys(forest, mid, ntuple(_ -> -1, D)))
    @test corner in keys(recruits)
    edgebox = ntuple(d -> d == 1 ? (1:2) : (4:5), D)
    edgeflags = flag_blocks((b, k) -> k == mid ? (Refine, edgebox) : Keep, forest)
    @test !(corner in keys(buffer_recruits(forest, edgeflags, 2)))
end

@testset "An enum-only Refine buffers isotropically: D=$D" for D in (1, 2, 3)
    forest, mid = buffer_forest(Val(D))
    plain = flag_blocks((b, k) -> k == mid ? Refine : Keep, forest)
    whole = flag_blocks((b, k) -> k == mid ? (Refine, ntuple(_ -> 1:(forest.N), D)) : Keep,
                        forest)
    @test buffered_flags(forest, plain, 2) == buffered_flags(forest, whole, 2)

    # Every one of the 3^D - 1 surrounding directions joins the buffer.
    recruits = buffer_recruits(forest, plain, 2)
    @test sort(collect(keys(recruits))) ==
          sort(neighbors_over(forest, mid, alldirections(Val(D))))
end

@testset "Coarsen inside the buffer is demoted, finer neighbours only so" begin
    # The centre block is refined; the level-0 block across its low-x face
    # flags Refine with a box against that face. The level-1 leaves it
    # reaches are already finer, so they are demoted to Keep and not
    # promoted -- which breaks the unanimity their group needed to coarsen.
    forest, mid = buffer_forest(Val(2))
    refine!(forest, mid)
    balance!(forest)
    children = filter(k -> isancestor(mid, k), forest.leaves)
    @test length(children) == 4

    source = only(filter(k -> level(k) == 0 && root_position(forest, k.root) == (0, 1),
                         forest.leaves))
    flags = [k == source ? (Refine, (7:8, 4:5)) : (k in children ? Coarsen : Keep)
             for k in forest.leaves]

    touching = neighbor_keys(forest, source, (1, 0))
    @test length(touching) == 2                        # two children share the face
    marks = buffered_flags(forest, flags, 2)
    for (b, k) in enumerate(forest.leaves)
        if k in touching
            @test marks[b] === Keep                    # demoted, not promoted
        elseif k in children
            @test marks[b] === Coarsen                 # untouched by the buffer
        end
    end

    # Without the buffer the group is unanimous and collapses; with it the
    # children survive. That is the flicker suppression, operationally.
    @test !any(k -> isancestor(mid, k), complete_marks(forest, flags))
    @test count(k -> isancestor(mid, k), complete_marks(forest, flags; buffer=2)) == 4
end

@testset "A coarser neighbour in the buffer is promoted, and regrid! copes" begin
    forest, mid = buffer_forest(Val(2); N=8, G=2)
    refine!(forest, mid)
    balance!(forest)
    fs = FieldSet(forest, 1)
    fill_by_coordinates!((x, v) -> 3.0, fs)

    # A level-1 child with a box against the low-x face of its root: the
    # leaf across that face is a level-0 block, coarser than the source.
    source = MortonKey{2}(mid.root, 1, (0, 0))
    @test source in forest.leaves
    coarse = only(neighbor_keys(forest, source, (-1, 0)))
    @test level(coarse) == 0

    flags = [k == source ? (Refine, (1:2, 4:5)) : Keep for k in forest.leaves]
    @test buffer_recruits(forest, flags, 2) == Dict(coarse => Refine)

    @test regrid!(forest, fs, GhostSchedule(forest, OPS2); flags=flags, buffer=2)
    @test isbalanced(forest)
    @test issorted(forest.leaves)
    @test maxlevel(forest) == 2
    @test !(coarse in forest.leaves)                   # it really did refine
    @test all(c -> c in forest.leaves, childkeys(coarse))
    # A constant survives the transfer, so the promotion moved real data.
    for b in 1:nblocks(fs)
        @test all(≈(3.0), interiorview(fs, b, 1))
    end
end

@testset "Buffer argument checking" begin
    forest, mid = buffer_forest(Val(2); N=8)
    keep = fill(Keep, nleaves(forest))
    @test_throws ArgumentError buffered_flags(forest, keep, -1)
    @test_throws ArgumentError buffered_flags(forest, keep, 9)   # wider than N
    @test buffered_flags(forest, keep, 8) == keep                # exactly N is fine

    bad(box) = [k == mid ? (Refine, box) : Keep for k in forest.leaves]
    @test_throws ArgumentError buffered_flags(forest, bad((0:2, 1:2)), 1)   # below 1
    @test_throws ArgumentError buffered_flags(forest, bad((1:2, 4:9)), 1)   # above N
    @test_throws ArgumentError buffered_flags(forest, bad((3:2, 1:2)), 1)   # empty
    @test_throws ArgumentError buffered_flags(forest, bad((1:2,)), 1)       # wrong rank
    @test_throws ArgumentError buffered_flags(forest, fill(1, nleaves(forest)), 1)
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
