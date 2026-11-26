# The interface restriction (M8b) as `unify!` on a flux set (M22 step 3):
# a coarse block's copy of a value on a coarse-fine face is replaced by
# the restriction of the finer side's — injection along vertex-like
# dimensions, the exact two-cell average along cell-like ones — which is
# what makes the two sides of a coarse-fine face agree on the
# area-weighted flux, and hence what makes a conservative scheme
# conservative under a global timestep. A `UnifySchedule` holds stage 0
# of the fill alone, so it also gives every same-level secondary its
# primary's value, and an edge only a diagonal finer block touches its
# fine value.

# The centerings that have shared points, one set per dimension: the
# fluxes of a conservative scheme (face-centered, one vertex-like
# dimension), a 2D constrained-transport EMF `E_z` (the corners,
# vertex-like in both), and a 3D EMF `E_x` (edge-centered, so vertex-like
# in two dimensions and cell-like in the one it runs along).
interface_cases(::Val{1}) = [facecentered(1, 1)]
interface_cases(::Val{2}) = [facecentered(2, 1), facecentered(2, 2), vertexcentered(2)]
interface_cases(::Val{3}) = [facecentered(3, 1), facecentered(3, 3), edgecentered(3, 1)]

@testset "unify! makes every coarse copy the average of the finer side: D=$D" for
        D in (1, 2, 3)
    # The claim, against an oracle that knows only positions: every point
    # a block one level finer also holds — across a face, an edge or a
    # corner — comes back as the mean of the fine values at the
    # coincident points, and nothing else in the array moves. The data is
    # `arbitrary_value`, which depends on a point's exact position and
    # nothing else, so no smoothness hides a wrong source: a stencil off by
    # one cell reads a number unrelated to the right one.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> isodd(D), D))
    @test length(unique(level.(forest.leaves))) >= 3
    for C in interface_cases(Val(D)), G in (0, 1)
        fs = FieldSet(forest, 2; G=G, centering=C)
        # NaN everywhere first: the fixup must read and write own values
        # only, which is what lets a computed flux carry G = 0, and a
        # single ghost read would poison a target.
        fill!(fs.work, NaN)
        fill_arbitrary!(fs; closed=true)
        before = copy(fs.work)
        expected = interface_targets(fs, closed_values(fs))

        s = UnifySchedule(fs)
        @test unify!(fs, s) === fs

        # Not vacuous: this hierarchy really has coarse-fine faces.
        @test !isempty(expected)
        worst = 0.0
        for ((b, idx), want) in expected, v in 1:fs.nvars
            worst = max(worst, abs(fs.work[idx..., v, b] - want[v]))
        end
        @test worst <= 4eps(Float64)

        # Everything else keeps its value, NaN ghosts included (`isequal`,
        # so an unwritten NaN compares equal to itself): a same-level
        # secondary is written, with the number it already holds.
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

@testset "unify! writes every secondary exactly once, from its primary: D=$D" for
        D in (2, 3)
    # One stage, where the interface restriction ran one phase per face
    # dimension and wrote a point on the line where two coarse-fine faces
    # meet twice. A double write would make the result depend on which
    # source landed last — this operation overwrites the block's own
    # computed values, so unlike a ghost fill there is no "unwritten"
    # state to fall back on. What it writes is what the mask calls a
    # secondary, and the coarse copies the position oracle names among it.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> false, D))
    for C in interface_cases(Val(D))
        fs = FieldSet(forest, 1; G=1, centering=C)
        s = UnifySchedule(fs)
        counts = only(interface_write_counts(s))
        @test all(<=(1), counts)
        mask = primary_mask(fs)
        c = staggers(C)
        written = Set{Tuple{Int,NTuple{D,Int}}}()
        for idx in CartesianIndices(counts)
            counts[idx] == 0 && continue
            t = Tuple(idx)
            push!(written, (t[D + 1], ntuple(d -> t[d], D)))
        end
        secondaries = Set{Tuple{Int,NTuple{D,Int}}}()
        for b in 1:nblocks(fs), I in CartesianIndices(ntuple(d -> forest.N + c[d], D))
            isprimary(mask, Tuple(I)..., b) ||
                push!(secondaries, (b, ntuple(d -> I[d] + fs.G[d], D)))
        end
        @test written == secondaries
        fill_arbitrary!(fs; closed=true)
        @test issubset(keys(interface_targets(fs, closed_values(fs))), written)
    end
end

@testset "A cell-centered field set has no interface to restrict" begin
    # Silently building an empty schedule would be the wrong answer: a
    # cell-centered field has no values lying *on* a block's face, so
    # asking for the fixup over it means the caller reached for the state
    # set instead of the flux set.
    forest = nested_forest(Val(2); N=8)
    @test_throws ArgumentError UnifySchedule(FieldSet(forest, 1; G=2))
    @test_throws "no shared point" UnifySchedule(FieldSet(forest, 1; G=2))
    @test_throws "facecentered(2, d)" UnifySchedule(FieldSet(forest, 1; G=2))
end

@testset "A unify schedule refuses a mesh or a layout it was not built for" begin
    # Every target range in a schedule is an index into one layout over
    # one leaf array. A stale or mismatched one would write the wrong
    # points of the wrong blocks rather than fail.
    forest = nested_forest(Val(2); N=8)
    fs = FieldSet(forest, 1; G=1, centering=facecentered(2, 1))
    s = UnifySchedule(fs)
    @test !isstale(s)
    @test unify!(fs, s) === fs

    @test_throws "ghost width" unify!(
        FieldSet(forest, 1; G=2, centering=facecentered(2, 1)), s)
    @test_throws "centering" unify!(
        FieldSet(forest, 1; G=1, centering=vertexcentered(2)), s)
    @test_throws "carries" unify!(
        FieldSet{Float32}(forest, 1; G=1, centering=facecentered(2, 1)), s)
    @test_throws "different forest" unify!(
        FieldSet(nested_forest(Val(2); N=8), 1; G=1, centering=facecentered(2, 1)), s)

    refine!(forest, first(forest.leaves))
    @test isstale(s)
    @test_throws "rebuild it" unify!(fs, s)

    @test occursin("UnifySchedule{Float64,2}", sprint(show, s))
end

@testset "The unify schedule is a function of the tree alone: D=$D" for D in (2, 3)
    # The neighbour walk is threaded and its per-task lists are
    # concatenated in chunk order, which is block order. That is what
    # makes the schedule — and therefore every number the fixup produces
    # — identical whatever `Threads.nthreads()` happens to be; a group
    # whose target blocks were not ascending would mean the merge had
    # gone out of order.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> true, D))
    fs = FieldSet(forest, 1; G=0, centering=facecentered(D, 1))
    s = UnifySchedule(fs)
    @test all(g -> issorted(g.targetblocks), s.unify)
    @test [st.tag for st in s.stages] == [TreeAMR.UNIFY_TAG]

    # And replaying it twice lands on the same numbers: the fixup is a
    # pure function of the data it is handed.
    fill_arbitrary!(fs; closed=true)
    once = copy(unify!(fs, s).work)
    @test unify!(fs, s).work == once
end

@testset "No point the fixup writes is one it reads: D=$D" for D in (1, 2, 3)
    # Stage 0 reads primaries and writes secondaries, which is what lets it
    # run as one stage whatever order its groups run in and however the
    # slices are dealt out to threads; and it reads nothing outside a
    # block's own range, which is what lets a computed flux carry no
    # ghosts at all.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> isodd(D), D))
    @test isbalanced(forest)
    for C in interface_cases(Val(D)), G in (0, 1)
        fs = FieldSet(forest, 1; G=G, centering=C)
        targets, sources = interface_touches(UnifySchedule(fs))
        @test !isempty(targets)
        @test isdisjoint(targets, sources)
        c = staggers(C)
        @test all(sources) do (_, idx)
            all(d -> G + 1 <= idx[d] <= G + forest.N + c[d], 1:D)
        end
    end
end

@testset "A flux's copies at a same-level face are one number after unify!: D=$D" for
        D in (1, 2, 3)
    # Guards the exactness at a same-level face that the interface
    # restriction left to the application: two blocks that computed the
    # flux through their common face from their own data may differ by
    # rounding, and after `unify!` they hold the same bits, so the
    # divergence each applies takes the same flux and conservation there
    # is exact rather than to rounding.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> true, D))
    for C in interface_cases(Val(D))
        fs = FieldSet(forest, 1; G=0, centering=C)
        fs.work .= randn(Xoshiro(41 + D), size(fs.work))
        unify!(fs, UnifySchedule(fs))
        groups = Dict{Any,Vector{Float64}}()
        c = staggers(C)
        for b in 1:nblocks(fs), I in CartesianIndices(ntuple(d -> forest.N + c[d], D))
            key = (level(blockkey(fs, b)), exact_point(fs, b, Tuple(I)))
            push!(get!(groups, key, Float64[]), fs.work[Tuple(I)..., 1, b])
        end
        @test any(v -> length(v) > 1, values(groups))
        @test all(v -> all(x -> x === first(v), v), values(groups))
    end
end
