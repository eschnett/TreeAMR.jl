# M8b step 4: interface restriction, the flux fixup at coarse-fine
# faces. A coarse block's boundary value is replaced by the restriction
# of the finer side's coincident one — injection along vertex-like
# dimensions, the exact two-cell average along cell-like ones — which is
# what makes the two sides of a coarse-fine face agree on the
# area-weighted flux, and hence what makes a conservative scheme
# conservative under a global timestep.

# The centerings that have an admissible face direction, one set per
# dimension: the fluxes of a conservative scheme (face-centered, one
# vertex-like dimension), a 2D constrained-transport EMF `E_z` (the
# corners, vertex-like in both), and a 3D EMF `E_x` (edge-centered, so
# vertex-like in two dimensions and cell-like in the one it runs along).
interface_cases(::Val{1}) = [facecentered(1, 1)]
interface_cases(::Val{2}) = [facecentered(2, 1), facecentered(2, 2), vertexcentered(2)]
interface_cases(::Val{3}) = [facecentered(3, 1), facecentered(3, 3), edgecentered(3, 1)]

@testset "Interface restriction is the average of the finer side: D=$D" for
        D in (1, 2, 3)
    # The claim, against an oracle that knows only positions: every point
    # on a coarse block's coarse-fine boundary face comes back as the
    # mean of the fine values at the coincident points, and nothing else
    # in the array moves. The data is `arbitrary_value`, which depends on
    # a point's exact position and nothing else, so no smoothness hides a
    # wrong source: a stencil off by one cell reads a number unrelated to
    # the right one.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> isodd(D), D))
    @test length(unique(level.(forest.leaves))) >= 3
    for C in interface_cases(Val(D)), G in (0, 1)
        fs = FieldSet(forest, 2; G=G, centering=C)
        # NaN everywhere first: the fixup must read and write closed-range
        # values only, which is what lets a computed flux carry G = 0, and
        # a single ghost read would poison a target.
        fill!(fs.work, NaN)
        fill_arbitrary!(fs; closed=true)
        before = copy(fs.work)
        expected = interface_targets(fs, closed_values(fs))

        isched = InterfaceSchedule(fs)
        restrict_interfaces!(fs, isched)

        # Not vacuous: this hierarchy really has coarse-fine faces.
        @test !isempty(expected)
        worst = 0.0
        for ((b, idx), want) in expected, v in 1:fs.nvars
            worst = max(worst, abs(fs.work[idx..., v, b] - want[v]))
        end
        @test worst <= 4eps(Float64)

        # Everything else is untouched, NaN ghosts included (`isequal`,
        # so an unwritten NaN compares equal to itself).
        stored = CartesianIndices(ntuple(d -> axes(fs.work, d), D))
        untouched = all(1:nblocks(fs)) do b
            all(stored) do idx
                haskey(expected, (b, Tuple(idx))) ||
                    all(v -> isequal(fs.work[Tuple(idx)..., v, b],
                                     before[Tuple(idx)..., v, b]), 1:fs.nvars)
            end
        end
        @test untouched
        # And no target came back NaN, which is the same claim stated
        # from the other side: no ghost reached a stencil.
        @test !any(isnan, view(fs.work, ntuple(_ -> Colon(), D + 2)...)[
            ntuple(d -> (G + 1):(G + forest.N + staggers(C)[d]), D)..., :, :])
    end
end

@testset "Every interface target is written exactly once per phase: D=$D" for
        D in (2, 3)
    # There is one phase per face dimension. Within a phase a double
    # write would make the result depend on which fine source landed
    # last — this operation overwrites the block's own computed values,
    # so unlike a ghost fill there is no "unwritten" state to fall back
    # on. Across phases a point on the line where two coarse-fine faces
    # of the same block meet *is* written twice, by two same-level fine
    # blocks that computed the same value there, which is why the counts
    # are taken per phase.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> false, D))
    for C in interface_cases(Val(D))
        fs = FieldSet(forest, 1; G=1, centering=C)
        isched = InterfaceSchedule(fs)
        counts = interface_write_counts(isched)
        @test length(counts) == count(==(:vertex), C)
        @test all(phase -> all(<=(1), phase), counts)

        # The points the schedule writes are exactly the ones the
        # position oracle says should move — same set, counted once per
        # phase that touches them.
        fill_arbitrary!(fs; closed=true)
        expected = interface_targets(fs, closed_values(fs))
        written = Set{Tuple{Int,NTuple{D,Int}}}()
        for phase in counts, idx in CartesianIndices(phase)
            phase[idx] == 0 && continue
            t = Tuple(idx)
            push!(written, (t[D + 1], ntuple(d -> t[d], D)))
        end
        @test written == Set(keys(expected))
    end
end

@testset "A cell-centered field set has no interface to restrict" begin
    # Silently building an empty schedule would be the wrong answer: a
    # cell-centered field has no values lying *on* a block's face, so
    # asking for the fixup over it means the caller reached for the state
    # set instead of the flux set.
    forest = nested_forest(Val(2); N=8)
    @test_throws ArgumentError InterfaceSchedule(FieldSet(forest, 1; G=2))
    @test_throws "no coarse-fine interface" InterfaceSchedule(FieldSet(forest, 1; G=2))
    @test_throws "facecentered(2, d)" InterfaceSchedule(FieldSet(forest, 1; G=2))
end

@testset "An interface schedule refuses a mesh or a layout it was not built for" begin
    # Every target plane in a schedule is an index into one layout over
    # one leaf array. A stale or mismatched one would write the wrong
    # cells of the wrong blocks rather than fail.
    forest = nested_forest(Val(2); N=8)
    fs = FieldSet(forest, 1; G=1, centering=facecentered(2, 1))
    isched = InterfaceSchedule(fs)
    @test !isstale(isched)
    @test restrict_interfaces!(fs, isched) === fs

    @test_throws "ghost width" restrict_interfaces!(
        FieldSet(forest, 1; G=2, centering=facecentered(2, 1)), isched)
    @test_throws "centering" restrict_interfaces!(
        FieldSet(forest, 1; G=1, centering=vertexcentered(2)), isched)
    @test_throws "carries" restrict_interfaces!(
        FieldSet{Float32}(forest, 1; G=1, centering=facecentered(2, 1)), isched)
    @test_throws "different forest" restrict_interfaces!(
        FieldSet(nested_forest(Val(2); N=8), 1; G=1, centering=facecentered(2, 1)),
        isched)

    refine!(forest, first(forest.leaves))
    @test isstale(isched)
    @test_throws "rebuild it" restrict_interfaces!(fs, isched)

    @test occursin("InterfaceSchedule{Float64,2}", sprint(show, isched))
end

@testset "The interface schedule is a function of the tree alone: D=$D" for D in (2, 3)
    # The neighbour walk is threaded and its per-task lists are
    # concatenated in chunk order, which is block order. That is what
    # makes the schedule — and therefore every number the fixup produces
    # — identical whatever `Threads.nthreads()` happens to be; a group
    # whose target blocks were not ascending would mean the merge had
    # gone out of order.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> true, D))
    fs = FieldSet(forest, 1; G=0, centering=facecentered(D, 1))
    isched = InterfaceSchedule(fs)
    @test all(g -> issorted(g.targetblocks), Iterators.flatten(isched.phases))
    @test isched.dimensions == [1]

    # And replaying it twice lands on the same numbers: the fixup is a
    # pure function of the data it is handed.
    fill_arbitrary!(fs; closed=true)
    once = copy(restrict_interfaces!(fs, isched).work)
    @test restrict_interfaces!(fs, isched).work == once
end

@testset "No plane the fixup writes is one it reads: D=$D" for D in (1, 2, 3)
    # The phases would otherwise not commute: a field set with more than
    # one vertex-like dimension runs several of them, and a fine block
    # that is itself the coarse side of a deeper interface could hand a
    # later phase a value an earlier one had already replaced. It cannot
    # happen, and 2:1 balance is why. Such a point lies on the line where
    # two of that block's faces meet, so the level-(l+2) block that wrote
    # it and the level-l block that would read it touch across a corner —
    # which `balance!` forbids, corner directions included.
    #
    # With that, the fixup is a pure function of the fluxes it is handed,
    # whatever order the phases run in and however the slices are dealt
    # out to threads.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> isodd(D), D))
    @test isbalanced(forest)
    for C in interface_cases(Val(D)), G in (0, 1)
        fs = FieldSet(forest, 1; G=G, centering=C)
        targets, sources = interface_touches(InterfaceSchedule(fs))
        @test !isempty(targets)
        @test isdisjoint(targets, sources)
        # And it reads nothing outside a block's closed range, which is
        # what lets a computed flux carry no ghosts at all.
        c = staggers(C)
        @test all(sources) do (_, idx)
            all(d -> G + 1 <= idx[d] <= G + forest.N + c[d], 1:D)
        end
    end
end
