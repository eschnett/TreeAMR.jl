# Multi-threading (M5).
#
# Two things are asserted here. The host-side helpers partition work
# correctly and thread-safely, and — the milestone's acceptance test —
# a complete cycle produces *bit-identical* results whatever thread
# count Julia was started with. For the state vector, the leaf array,
# the schedule and the max norm that is what `CODE.md` promises: every
# parallel loop writes to its own slot. For the two floating-point sums
# in the digest (`l2`, `mass`) `CODE.md` promises roundoff only, since
# M8; they come out exact because the CPU fold sums its per-block
# partials in block order, and they are the lines to give a tolerance
# if that fold is ever reassociated. Everything else must stay exact —
# that is the race detector.

using TreeAMR: threadchunks, threaded_foreach, threaded_chunks, TransferGroup, ntransfers
using KernelAbstractions: @kernel, @index, CPU

@testset "Thread chunks tile 1:n without gaps or overlap" begin
    # A chunking that dropped or repeated an index would corrupt a
    # schedule rather than merely slow it down, so this is checked for
    # its own sake and not only through the loops that use it.
    for n in 0:37
        chunks = threadchunks(n)
        @test sum(length, chunks; init=0) == n
        @test all(!isempty, chunks)
        @test vcat(collect.(chunks)...) == collect(1:n)
        @test length(chunks) <= max(Threads.nthreads(), 1)
        @test length(chunks) <= max(n, 1)
        # Balanced to within one item, so no task does twice the work.
        isempty(chunks) ||
            @test maximum(length, chunks) - minimum(length, chunks) <= 1
    end
    @test threadchunks(0) == UnitRange{Int}[]
    @test threadchunks(1) == [1:1]
end

@testset "Threaded loops visit every index exactly once" begin
    for n in (0, 1, 3, 100, 1000)
        seen = zeros(Int, n)
        threaded_foreach(n) do i
            seen[i] += 1
        end
        @test all(==(1), seen)
    end
end

@testset "An error in a threaded loop arrives unwrapped" begin
    # The package's ArgumentErrors say *why* something is wrong, and
    # tests match on that text. Raising one on a worker task must not
    # bury it inside a TaskFailedException.
    err = try
        threaded_foreach(200) do i
            i == 137 && throw(ArgumentError("the reason, stated plainly"))
        end
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("stated plainly", err.msg)
end

@testset "Prolongation groups are homogeneous in target level: D=$D" for D in (1, 2, 3)
    # A regression guard. Prolongations used to be batched by direction
    # and child offset alone, so targets at different levels landed in
    # one group, which was then filed under whichever level happened to
    # be recorded last. That silently defeats the coarsest-target-first
    # sweep phase 2 exists for — a prolongation may read its coarse
    # source's own prolongated ghosts.
    forest = nested_forest(Val(D))
    @test length(unique(level.(forest.leaves))) >= 3

    schedule = GhostSchedule(FieldSet(forest, 1; G=1),
                             Operators(prolongation=2, restriction=2))
    @test issorted(schedule.levels)
    @test length(schedule.phase2) == length(schedule.levels)
    @test length(schedule.levels) >= 2

    for (sweep, lvl) in zip(schedule.phase2, schedule.levels)
        for group in sweep
            @test group.kind === :prolong
            @test ntransfers(group) > 0
            for t in group.targetblocks
                @test level(forest.leaves[t]) == lvl
            end
        end
    end

    # Every prolongation in the schedule is still present exactly once.
    prolongations = sum(g -> ntransfers(g), Iterators.flatten(schedule.phase2); init=0)
    expected = 0
    for k in forest.leaves, δ in alldirections(Val(D))
        nbrs = neighbor_keys(forest, k, δ)
        isempty(nbrs) && continue
        level(first(nbrs)) < level(k) && (expected += 1)
    end
    @test prolongations == expected
end

@testset "fill_by_coordinates! reproduces coordinates exactly: D=$D" for D in (1, 2, 3)
    # The kernel computes positions from the per-block origin and
    # spacing arrays rather than from the tree; it has to land on the
    # same floating-point value `coordinates` gives, not merely a close
    # one, or ghost exchange and boundary hooks would disagree at the
    # last bit. Checked at two ghost widths, since the two expressions
    # cancel `G` differently: the kernel never forms it, `coordinates`
    # subtracts it.
    forest = nested_forest(Val(D); N=4)
    f = (x, v) -> sum(x) + 100v
    for G in (1, 2)
        fs = FieldSet(forest, 2; G=G)
        fill_by_coordinates!(f, fs)

        N = forest.N
        interior = CartesianIndices(ntuple(_ -> (G + 1):(G + N), D))
        identical = all(1:nblocks(fs)) do b
            all(1:fs.nvars) do v
                block = blockview(fs, b, v)
                all(idx -> block[idx] === f(coordinates(fs, b, Tuple(idx)), v), interior)
            end
        end
        @test identical
    end
end

@testset "Every transfer group lists its targets in block order: D=$D" for D in (1, 2, 3)
    # The ghost fill, the interface fixup and the regrid transfer give
    # each thread the transfers whose target block it owns, found by
    # bisection in `targetblocks`. A group whose targets were out of
    # order would have transfers silently skipped and their ghosts left
    # stale, so the order every builder produces is asserted for all
    # three kinds of schedule.
    forest = nested_forest(Val(D))
    ops = Operators(prolongation=2, restriction=2)
    schedule = GhostSchedule(FieldSet(forest, 1; G=1), ops)
    ghostgroups = vcat(schedule.phase1, schedule.phase2...)
    @test length(ghostgroups) > 1
    @test all(g -> issorted(g.targetblocks), ghostgroups)

    isched = InterfaceSchedule(FieldSet(forest, 1; G=0, centering=facecentered(D, 1)))
    facegroups = vcat(isched.phases...)
    @test sum(ntransfers, facegroups; init=0) > 0
    @test all(g -> issorted(g.targetblocks), facegroups)

    finest = maxlevel(forest)
    flags = [level(k) == finest ? Coarsen : level(k) == 0 ? Refine : Keep
             for k in forest.leaves]
    newleaves = complete_marks(forest, flags)
    groups = TreeAMR.transfer_groups(Float64, forest, ntuple(_ -> 1, D),
                                     ntuple(_ -> 0, D), copy(forest.leaves), newleaves,
                                     ops, CPU())
    @test Set(g.kind for g in groups) == Set((:copy, :prolong, :restrict))
    @test all(g -> issorted(g.targetblocks), groups)
end

@kernel function threadid_kernel!(ids)
    I = @index(Global, NTuple)
    ids[I...] = Threads.threadid()
end

@testset "Every per-block pass runs a block on the thread that owns it" begin
    # The block-ownership policy (CODE.md, "What one process loses"):
    # block `b` belongs to the thread whose chunk of
    # `threadchunks(nblocks)` contains it, and the host loops and the
    # kernel launches must both put it there, since data that changes
    # core from one pass to the next streams at a fraction of the rate
    # on a many-core node. Only a threaded run can tell the difference;
    # single-threaded, everything runs on the calling thread.
    forest = nested_forest(Val(2))
    fs = FieldSet(forest, 1; G=1)
    nb = nblocks(fs)
    owner = fill(Threads.threadid(), nb)
    if Threads.nthreads() > 1
        offset = Threads.threadpoolsize(:interactive)
        for (c, range) in enumerate(threadchunks(nb))
            owner[range] .= offset + c
        end
    end

    seen = zeros(Int, nb)
    threaded_chunks(nb) do _, range
        for b in range
            seen[b] = Threads.threadid()
        end
    end
    @test seen == owner

    N = forest.N
    ids = zeros(Int, N, N, nb)
    map_blocks!(threadid_kernel!, fs, ids)
    @test all(b -> all(==(owner[b]), view(ids, :, :, b)), 1:nb)
    ids = zeros(Int, size(fs.work, 1), size(fs.work, 2), nb)
    map_blocks!(threadid_kernel!, fs, ids; stored=true)
    @test all(b -> all(==(owner[b]), view(ids, :, :, b)), 1:nb)
end

@testset "Per-block passes still run inside a caller's own threaded loop" begin
    # The static schedule behind `launch_by_owner!` is `@threads
    # :static`, which refuses to run inside another threaded region, so
    # there the launch has to fall back to the default schedule rather
    # than throw; and `threaded_chunks` uses sticky tasks of its own
    # precisely so that it nests. Both must give the same answer as the
    # unnested call.
    forest = nested_forest(Val(2))
    fs = FieldSet(forest, 1; G=1)
    schedule = GhostSchedule(fs, Operators(prolongation=2, restriction=2))
    step!() = (fill_by_coordinates!((x, v) -> x[1] + 2x[2], fs); fill_ghosts!(fs, schedule))
    fs.work .= 0
    step!()
    expected = copy(fs.work)
    fs.work .= 0
    Threads.@threads :static for i in 1:Threads.nthreads()
        i == 1 && step!()
    end
    @test fs.work == expected

    visits = zeros(Int, 12, 12)
    threaded_foreach(12) do i
        threaded_foreach(12) do j
            visits[i, j] += 1
        end
    end
    @test all(==(1), visits)
end

@testset "Results are bit-identical across thread counts" begin
    # The acceptance test for M5. The thread count is a command-line
    # argument to Julia, so the only way to vary it is to run the
    # workload in subprocesses; `test/thread_workload.jl` prints a
    # digest of the state vector, the leaf array, the schedule shape,
    # and the reductions after a full adapt/evolve/regrid/evolve cycle
    # in D = 1, 2, 3, periodic and not.
    script = joinpath(@__DIR__, "thread_workload.jl")
    # The *active* environment, not `test/`: under `Pkg.test` the tests
    # run in a sandbox, and `test/` may have no manifest at all.
    project = Base.active_project()
    run_with(n) = read(`$(Base.julia_cmd()) --threads=$n --project=$project $script`,
                       String)

    serial = run_with(1)
    @test occursin("state", serial)
    @test count(==('\n'), serial) >= 4

    @test run_with(max(2, min(4, Sys.CPU_THREADS))) == serial
end
