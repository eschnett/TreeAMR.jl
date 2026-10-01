# M7 step 1: local blocks, no messages.
#
# Every rank holds the whole forest and stores the blocks of one
# contiguous run of its leaves, `blockrange(forest)`, and from step 1 on
# every block index in the package is local to the rank. Nothing here
# sends a message: a rank is simulated in process by a communicator that
# answers its rank and the rank count and nothing else, and each claim
# is stated against the serial forest over the same leaves, which is the
# oracle — whatever a rank stores or computes for its block `b`, the
# serial forest stores or computes for leaf `first(blockrange) + b - 1`.

using TreeAMR: equalsplit, threadchunks, commrank, commsize, allgather, allgatherv,
               alltoallv, isend, irecv, waitall, SerialCommunicator

# A rank of a communicator that does not exist: it answers the two verbs
# the partition needs, and every other verb is refused by the package's
# fallback, which is what lets a test see which operations need more.
struct PartitionCommunicator <: TreeAMR.Communicator
    rank::Int
    size::Int
end
TreeAMR.commrank(c::PartitionCommunicator) = c.rank
TreeAMR.commsize(c::PartitionCommunicator) = c.size

# The forest `serial` as rank `r` of `P` holds it: the same leaves, the
# same brick, through the validated `leaves` path.
rank_forest(serial::Forest{D,T}, r, P) where {D,T} =
    Forest{T}(serial.roots; N=serial.N, periodic=serial.periodic,
              reflecting=serial.reflecting, extents=serial.extents,
              leaves=serial.leaves, comm=PartitionCommunicator(r, P))

# The equal-count split as `threadchunks` computed it through 0.1.4, a
# running loop rather than a closed form: an independent statement of
# the arithmetic that the shared helper must reproduce.
function loop_split(n, p)
    len, extra = divrem(n, p)
    ranges = UnitRange{Int}[]
    lo = 1
    for t in 1:p
        hi = lo + len - 1 + (t <= extra)
        push!(ranges, lo:hi)
        lo = hi + 1
    end
    return ranges
end

# Every transfer of a schedule as a value that names the whole stencil,
# with its blocks shifted by `offset` into global leaf indices.
function transfer_set(groups, offset)
    out = Set{Any}()
    for g in groups
        st = map(s -> (s.targetfirst, Vector(s.srcstart), Matrix(s.weights)), g.stencils)
        for t in 1:ntransfers(g)
            push!(out, (g.kind, g.factorcol, st, Int(g.targetblocks[t]) + offset,
                        Int(g.sourceblocks[t]) + offset))
        end
    end
    return out
end
schedule_groups(s::GhostSchedule) = [s.phase1; reduce(vcat, s.phase2; init=eltype(s.phase1)[])]
schedule_groups(s::InterfaceSchedule) =
    reduce(vcat, s.phases; init=eltype(eltype(s.phases))[])

using TreeAMR: ntransfers

@testset "The rank split is threadchunks' arithmetic and tiles the leaves" begin
    # A rank split that left a gap or an overlap would lose or double a
    # block, one out of order would break the curve-order layout every
    # buffer is derived from, and one that drifted from `threadchunks`
    # would make the two levels of the split disagree.
    for n in 0:40, p in 1:9
        parts = [equalsplit(n, p, i) for i in 1:p]
        @test parts == loop_split(n, p)
        @test reduce(vcat, collect.(parts); init=Int[]) == collect(1:n)
        @test all(i -> first(parts[i + 1]) == last(parts[i]) + 1, 1:(p - 1))
        lens = length.(parts)
        @test maximum(lens) - minimum(lens) <= 1
        @test issorted(lens; rev=true)              # the longer ones first
        @test all(isempty, parts[(n + 1):end])      # ranks beyond the leaves
    end
    # And `threadchunks` itself, at this process's thread count.
    for n in 0:100
        @test threadchunks(n) == (n == 0 ? UnitRange{Int}[] :
                                  loop_split(n, min(Threads.nthreads(), n)))
    end
end

@testset "Every rank stores the blocks of its range, as the serial forest does: D=$D" for
        D in (1, 2, 3)
    # A block index left global anywhere — in the storage, the geometry,
    # the key lookup or the flagging — names another rank's leaf, and on
    # a rank with offset zero it would even look right. So every rank of
    # every count, including counts beyond the leaves, is checked against
    # the serial forest at the global index.
    rng = MersenneTwister(70 + D)
    serial = random_forest(rng, Val(D); nsteps=D == 3 ? 12 : 30, maxlvl=3)
    balance!(serial)
    n = nleaves(serial)
    G = 1
    C = D == 2 ? facecentered(2, 1) : vertexcentered(D)
    sfs = FieldSet(serial, 2; G=G, centering=C)
    @test serial.comm === SerialCommunicator()
    @test blockrange(serial) == 1:n
    sorigins = block_origins(serial)
    sspacings = block_spacings(serial)
    sflags = flag_blocks((b, k) -> (b, k), serial)
    f(x, v) = v + sum(d -> d * x[d]^2, 1:D)
    fill_by_coordinates!(f, sfs)
    for P in (1, 2, 3, 5, n + 2)
        ranges = UnitRange{Int}[]
        for r in 0:(P - 1)
            forest = rank_forest(serial, r, P)
            @test nleaves(forest) == n            # tree queries stay global
            range = blockrange(forest)
            push!(ranges, range)
            @test range == equalsplit(n, P, r + 1)
            fs = FieldSet(forest, 2; G=G, centering=C)
            m = length(range)
            @test nblocks(fs) == m
            @test size(fs.work) == (size(sfs.work)[1:(D + 1)]..., m)
            @test statelength(fs) == forest.N^D * 2 * m
            @test [blockkey(fs, b) for b in 1:m] == serial.leaves[range]
            @test_throws BoundsError blockkey(fs, m + 1)
            @test block_origins(forest) == sorigins[range]
            @test block_spacings(forest) == sspacings[range]
            @test flag_blocks((b, k) -> (b, k), forest) ==
                  [(b - first(range) + 1, k) for (b, k) in sflags[range]]
            corner = ntuple(_ -> 1, D)
            far = size(fs.work)[1:D]
            @test all(b -> coordinates(fs, b, corner) ==
                           coordinates(sfs, first(range) + b - 1, corner) &&
                           coordinates(fs, b, far) ==
                           coordinates(sfs, first(range) + b - 1, far), 1:m)
            # The fill reads only the per-block geometry, so it is
            # bit-identical to the serial fill on the rank's blocks, and an
            # empty rank fills nothing in either form.
            fill_by_coordinates!(f, fs)
            @test fs.work == sfs.work[ntuple(_ -> :, D + 1)..., range]
            fill_by_coordinates!(AllVariables(x -> (f(x, 1), f(x, 2))), fs)
            @test fs.work == sfs.work[ntuple(_ -> :, D + 1)..., range]
            # The per-block reductions are per local block.
            @test block_mapreduce(abs, max, 0.0, fs) ==
                  block_mapreduce(abs, max, 0.0, sfs)[range]
            fires(work, idx, b, x) = x[1] < 0.6 * sum(serial.extents[1])
            @test firing_boxes(fires, fs) == firing_boxes(fires, sfs)[range]
        end
        @test reduce(vcat, collect.(ranges); init=Int[]) == collect(1:n)
        P > n && @test count(isempty, ranges) == P - n
    end
end

@testset "A rank's schedules hold exactly the serial transfers local to it: D=$D" for
        D in (1, 2, 3)
    # The schedule is built for the rank's own targets, in local block
    # indices. A builder that walked every leaf, or forgot to shift a
    # source, would fill the wrong block's ghosts; one that kept a remote
    # source would read past the rank's array. Until the distributed
    # schedule (step 2) a rank keeps the transfers with both ends local
    # and the boundary regions of its own blocks, and that is exactly the
    # serial schedule restricted to its range.
    kinds = D == 1 ? (:reflect_lo,) : D == 2 ? (:reflect_both, :periodic) :
            (:reflect_lo, :outer, :periodic)
    serial = faces_forest(kinds; N=8)
    n = nleaves(serial)
    ops = Operators(prolongation=4, restriction=4)
    parity = [EvenParity, OddParity]
    for C in (cellcentered(D), vertexcentered(D))
        sched = GhostSchedule(FieldSet(serial, 2; G=2, centering=C, parity=parity), ops)
        stransfers = transfer_set(schedule_groups(sched), 0)
        sbounds = Set((Int(r.block), r.direction, r.region) for r in sched.boundaries)
        crossing = 0                  # serial transfers with their ends on two ranks
        for P in (2, 3), r in 0:(P - 1)
            forest = rank_forest(serial, r, P)
            range = blockrange(forest)
            offset = first(range) - 1
            local_sched = GhostSchedule(FieldSet(forest, 2; G=2, centering=C,
                                                 parity=parity), ops)
            @test transfer_set(schedule_groups(local_sched), offset) ==
                  Set(t for t in stransfers if t[4] in range && t[5] in range)
            crossing += count(t -> t[4] in range && !(t[5] in range), stransfers)
            @test Set((Int(b.block) + offset, b.direction, b.region)
                      for b in local_sched.boundaries) ==
                  Set(b for b in sbounds if b[1] in range)
            @test all(g -> issorted(g.targetblocks), schedule_groups(local_sched))
            @test local_sched.boundaryplan.origins == block_origins(forest)
        end
        @test crossing > 0            # so the filter had something to drop
    end
    # The interface schedule the same way, over a face-centered flux.
    flux(forest) = FieldSet(forest, 1; G=0, centering=facecentered(D, 1),
                            parity=[EvenParity])
    sisched = InterfaceSchedule(flux(serial))
    sinterfaces = transfer_set(schedule_groups(sisched), 0)
    @test !isempty(sinterfaces)
    for P in (2, 3), r in 0:(P - 1)
        forest = rank_forest(serial, r, P)
        range = blockrange(forest)
        isched = InterfaceSchedule(flux(forest))
        @test transfer_set(schedule_groups(isched), first(range) - 1) ==
              Set(t for t in sinterfaces if t[4] in range && t[5] in range)
    end
end

@testset "What needs a message refuses a distributed forest, and says why" begin
    # Until the exchange, the regrid, the routing and the parallel file
    # exist, an operation that needs one must refuse rather than act on
    # this rank's blocks as if they were the whole mesh — a ghost fill
    # that silently skipped every remote source would look like a fill.
    serial = nested_forest(Val(2); N=8)
    forest = rank_forest(serial, 1, 3)
    fs = FieldSet(forest, 1; G=2)
    ops = Operators(prolongation=4, restriction=4)
    sched = GhostSchedule(fs, ops)
    @test_throws "fill_ghosts! over a forest distributed over 3 ranks" fill_ghosts!(fs,
                                                                                  sched)
    @test_throws "step 3" fill_ghosts!(fs, sched)
    flux = FieldSet(forest, 1; G=0, centering=facecentered(2, 1))
    @test_throws "restrict_interfaces! over a forest distributed" restrict_interfaces!(
        flux, InterfaceSchedule(flux))
    flags = fill(Keep, nblocks(fs))
    @test_throws "regrid! over a forest distributed" regrid!(forest, fs => sched;
                                                             flags=flags)
    @test_throws "step 4" regrid!(forest, fs => sched; flags=flags)
    @test_throws "step 5" interpolate(fs, [(0.5, 0.5)], Lagrange(4))
    @test_throws "step 6" save_checkpoint(tempname(), forest; fieldsets=("u" => fs,))
    # A reduction needs `allgather`, which this communicator lacks; the
    # refusal names the verb rather than failing inside the fold.
    @test_throws "does not implement `allgather`" mesh_mapreduce(abs, max, 0.0, fs)
    # And a forest built over something that is not a communicator.
    @test_throws "is not a communicator TreeAMR knows" Forest((2, 2); N=8, comm=42)
end

@testset "The serial communicator is rank 0 of 1 and sends nothing" begin
    # Every verb has a serial method, so that the package never branches
    # on whether MPI is loaded: the collectives return the caller's own
    # contribution, and a message to a peer is refused, since a serial
    # rank has none and a self-transfer is a local group.
    comm = SerialCommunicator()
    @test communicator(comm) === comm
    @test communicator(nothing) === comm
    @test Forest((2,); N=4, comm=comm).comm === comm
    @test (commrank(comm), commsize(comm)) == (0, 1)
    @test allgather(comm, (true, 1.5)) == [(true, 1.5)]
    @test_throws "isbits" allgather(comm, [1.0])
    @test allgatherv(comm, [1, 2, 3]) == [1, 2, 3]
    @test alltoallv(comm, [1.0, 2.0], [2]) == ([1.0, 2.0], [2])
    @test_throws "one entry" alltoallv(comm, [1.0, 2.0], [1, 1])
    @test_throws "no peer 1" isend(comm, [1.0], 1, 0)
    @test_throws "no peer 0" irecv(comm, [1.0], 0, 0)
    @test waitall(comm, []) === nothing
    # A communicator that answers only rank and size is refused at the
    # first verb it lacks, by name.
    fake = PartitionCommunicator(0, 2)
    @test_throws "does not implement `allgatherv`" allgatherv(fake, [1])
    @test_throws "does not implement `isend`" isend(fake, [1.0], 1, 0)
end
