# M7 step 2: the distributed schedule, without MPI.
#
# Over a forest distributed between ranks every transfer of the serial
# schedule is either *local* (target and source on one rank), *sent*
# (the source's rank computes it into a buffer) or *received* (the
# target's rank copies it into place), and every ordering point of the
# serial exchange is a stage. Nothing here sends a message: the ranks
# are simulated in one process over `PartitionCommunicator`
# (`partition_tests.jl`), and the serial schedule over the same leaves
# is the oracle throughout — for the transfers, for the layouts both
# ends of a message derive, for the writes, and bit for bit for the
# values, with every rank's buffers wired to its peers directly. A
# last test runs the staged driver itself over an in-process mailbox.
#
# This file's definitions are in `helpers/exchange.jl`.

using TreeAMR: PHASE1_TAG, prolongation_tag, interface_tag, ntransfers, boxsize,
               stagebuffers, segment_ranges, pack_stage!, unpack_stage!, run_phase!,
               apply_boundary!, exchange_ghosts!, exchange_interfaces!, leafowner,
               remote_neighbors, equalsplit_part, ntarget
using KernelAbstractions: CPU, synchronize
using MultiFloats: Float32x2

@testset "A leaf's owner inverts the rank split" begin
    # The stage builder names the rank at the other end of every
    # transfer by inverting the split; an off-by-one there would send a
    # transfer to the wrong rank, or expect it from one.
    for n in 1:40, p in 1:9, i in 1:n
        @test i in equalsplit(n, p, equalsplit_part(n, p, i))
    end
end

@testset "The candidate remote targets are every remote leaf touching this rank: D=$D" for
        D in (1, 2, 3)
    # A missing candidate is a remote target whose ghosts this rank never
    # packs, and its owner then waits for a message that is never sent.
    # The oracle is the serial schedule itself: every leaf that takes a
    # transfer from one of this rank's blocks must be a candidate.
    for kinds in EXCHANGE_KINDS[D]
        serial = faces_forest(kinds; N=8)
        n = nleaves(serial)
        sched = GhostSchedule(FieldSet(serial, 1; G=2, centering=vertexcentered(D),
                                       parity=[OddParity]), XOPS4)
        readers = [Set{Int}() for _ in 1:n]          # who reads each leaf
        for t in serial_transfers(sched)
            push!(readers[t[7]], t[6])
        end
        @test isempty(remote_neighbors(serial, 1:n))  # serially, no search at all
        for P in (2, 3, 5), r in 0:(P - 1)
            range = equalsplit(n, P, r + 1)
            candidates = remote_neighbors(serial, range)
            @test issorted(candidates) && allunique(candidates)
            @test !any(in(range), candidates)
            needed = setdiff(union(Set{Int}(), readers[range]...), range)
            @test issubset(needed, candidates)
        end
    end
end

@testset "Local, sent and received transfers are the serial schedule's, stage by stage: D=$D" for
        D in (1, 2, 3)
    # A transfer lost between ranks leaves a ghost unwritten, one found
    # twice writes it twice, and one filed under the wrong stage runs
    # before what it reads. So over 1–5 ranks the local groups of every
    # rank, and every sent transfer matched with its received half, must
    # be the serial transfers exactly, each under the stage the serial
    # schedule runs it in; and every slot must name its transfer in the
    # layout of its buffer, the owner of each end included.
    nonempty = 0
    for kinds in EXCHANGE_KINDS[D], C in allcenterings(Val(D)),
        family in (all(==(:cell), C) ? (PointValue, Conservative) : (PointValue,))
        serial = faces_forest(kinds; N=8)
        n = nleaves(serial)
        ops = exchange_operators(C, family)
        G = exchange_ghosts(C, family)
        stransfers = serial_transfers(GhostSchedule(FieldSet(serial, 1; G=G,
                                                             centering=C,
                                                             parity=[OddParity]), ops))
        for P in 1:5
            scheds = [GhostSchedule(FieldSet(rank_forest(serial, r, P), 1; G=G,
                                             centering=C, parity=[OddParity]), ops)
                      for r in 0:(P - 1)]
            locals, remote, unmatched, bad, nsent = distributed_transfers(scheds)
            @test bad == 0
            @test unmatched == 0
            @test length(locals) + length(remote) == length(stransfers)
            @test Set(locals) ∪ Set(remote) == stransfers
            P == 1 ? (@test nsent == 0) : (nonempty += nsent > 0)
            # A stage's local groups run in the order the serial phases
            # did: the tags ascend, and phase 1 is there on every rank.
            @test all(s -> issorted([st.tag for st in s.stages]) &&
                           first(s.stages).tag == PHASE1_TAG, scheds)
        end
    end
    @test nonempty > 0
end

@testset "Both ends of every message derive the same layout: D=$D" for D in (1, 2, 3)
    # The receiver unpacks a segment by the layout it derived itself,
    # never by one sent along, so the two derivations must agree element
    # for element — the same transfers in the same order at the same
    # offsets — or every ghost after the first disagreement is another's.
    for kinds in EXCHANGE_KINDS[D], C in (cellcentered(D), vertexcentered(D))
        serial = faces_forest(kinds; N=8)
        G = exchange_ghosts(C, PointValue)
        for P in 2:5
            scheds = [GhostSchedule(FieldSet(rank_forest(serial, r, P), 1; G=G,
                                             centering=C, parity=[OddParity]), XOPS4)
                      for r in 0:(P - 1)]
            bad, nonempty = layout_mismatches(scheds)
            @test bad == 0
            @test nonempty > 0
        end
    end
end

@testset "Every ghost is written exactly once, in the stage the serial fill writes it: D=$D" for
        D in (1, 2, 3)
    # Within a stage the order of the writes is free only because no
    # point is written twice; across stages a point written in the wrong
    # one is read before it is written. So the writes of every rank's
    # local groups and unpacks, per stage, must be the serial writes of
    # that stage, at most one per point — which the serial write counts
    # (exactly one per ghost, `ghost_tests.jl`) then make exactly one.
    for kinds in EXCHANGE_KINDS[D], C in allcenterings(Val(D))
        serial = faces_forest(kinds; N=8)
        G = exchange_ghosts(C, PointValue)
        stored = ntuple(d -> serial.N + 2G[d] + (C[d] === :vertex), D)
        sched = GhostSchedule(FieldSet(serial, 1; G=G, centering=C, parity=[OddParity]),
                              XOPS4)
        scounts = stage_write_counts([sched], stored, nleaves(serial))
        for P in (2, 3, 5)
            scheds = [GhostSchedule(FieldSet(rank_forest(serial, r, P), 1; G=G,
                                             centering=C, parity=[OddParity]), XOPS4)
                      for r in 0:(P - 1)]
            counts = stage_write_counts(scheds, stored, nleaves(serial))
            @test counts == scounts
            @test all(c -> maximum(c) <= 1, values(counts))
        end
    end
end

@testset "The interface stages hold the serial restrictions, split by rank: D=$D" for
        D in (1, 2, 3)
    # The interface restriction is the other exchange that crosses
    # ranks: a coarse-fine face whose fine side is another rank's. The
    # same three claims as for the ghosts — the transfers, both ends'
    # layouts, and one write per point per stage — over every centering
    # with a face dimension, one stage per face dimension.
    nonempty = 0
    for kinds in EXCHANGE_KINDS[D], C in allcenterings(Val(D))
        any(==(:vertex), C) || continue
        serial = faces_forest(kinds; N=8)
        G = ntuple(d -> C[d] === :vertex ? 0 : 1, D)
        flux(forest) = FieldSet(forest, 1; G=G, centering=C, parity=[OddParity])
        isched = InterfaceSchedule(flux(serial))
        @test [st.tag for st in isched.stages] == interface_tag.(isched.dimensions)
        stransfers = serial_transfers(isched)
        stored = ntuple(d -> serial.N + 2G[d] + (C[d] === :vertex), D)
        scounts = stage_write_counts([isched], stored, nleaves(serial))
        for P in 1:5
            scheds = [InterfaceSchedule(flux(rank_forest(serial, r, P))) for r in 0:(P - 1)]
            locals, remote, unmatched, bad, nsent = distributed_transfers(scheds)
            @test bad == 0
            @test unmatched == 0
            @test length(locals) + length(remote) == length(stransfers)
            @test Set(locals) ∪ Set(remote) == stransfers
            @test layout_mismatches(scheds)[1] == 0
            counts = stage_write_counts(scheds, stored, nleaves(serial))
            @test counts == scounts
            @test all(c -> maximum(c) <= 1, values(counts))
            nonempty += nsent > 0
        end
    end
    @test nonempty > 0
end

# `Float32` in 2D only, and `Float32x2` from 2D on (trimmed in step 9, for
# the suite's time: each pair of `D` and type compiles the kernels anew).
# Neither type changes what a pack or an unpack does, only the arithmetic
# the round trip must preserve bit for bit, which one dimension checks as
# well as three; and in 1D every mirrored transfer is a block's own, so
# the `−0` case that needs a remote mirror exists from 2D on.
@testset "Packing, delivering and unpacking reproduces the serial fill bitwise: D=$D, T=$T" for
        (D, T) in ((1, Float64), (2, Float64), (3, Float64), (2, Float32),
                   (2, Float32x2), (3, Float32x2))
    # The values themselves. The sender computes each remote ghost with
    # the serial stencil from the same source values and the unpack adds
    # an identity, so every rank's working array, ghosts included, must
    # equal the serial fill on its blocks bit for bit, the boundary hook
    # between the stages included. Two things could break that silently:
    # a pack that applied the parity factor would turn the serial fill's
    # `−0` from a mirrored zero into `+0` (so the data has an odd
    # variable that is zero, and the test asserts that `−0` occurs and
    # that mirrored transfers do cross ranks), and an unpack `0 + 1·x`
    # that is not the identity on an element type — the reason for
    # `Float32x2`, whose limbs must survive it. A rank with no blocks
    # takes part too.
    nnegzero = nmirror = 0
    for kinds in EXCHANGE_KINDS[D]
        serial = faces_forest(kinds; T=T, N=8)
        centers = T === Float64 ? allcenterings(Val(D)) :
                  D == 3 ? [vertexcentered(D)] : [cellcentered(D), vertexcentered(D)]
        counts = T === Float64 ? (D == 1 ? (1, 2, 3, 5, nleaves(serial) + 1) : (2, 3, 5)) :
                 (3,)
        for C in centers, P in counts
            same, nz, nm, bad = lockstep_ghost_fill(serial, P, C, PointValue, T)
            @test same
            @test bad == 0
            nnegzero += nz
            nmirror += nm
        end
        if T === Float64
            same, _, _, bad = lockstep_ghost_fill(serial, 3, cellcentered(D), Conservative, T)
            @test same && bad == 0
            # A bounded prolongation is computed by the sender too, its
            # child side read off the stencil column the pack keeps.
            same, _, _, bad = lockstep_ghost_fill(serial, 3, cellcentered(D), Conservative,
                                                  T; bounded=true)
            @test same && bad == 0
            # So is a bounded point-value interpolation, its bracket read
            # off the multilinear rows the pack keeps; vertex-centered, so
            # that the derived wall row is bounded too.
            for C in (cellcentered(D), vertexcentered(D))
                same, _, _, bad = lockstep_ghost_fill(serial, 3, C, PointValue, T;
                                                      bounded=true)
                @test same && bad == 0
            end
        end
    end
    # MultiFloats' product returns `+0` for `0 · (−1)`, so the serial fill
    # has no `−0` to preserve in `Float32x2`; what that type checks is
    # that `0 + 1·x` returns the limbs of every value a pack produces.
    T === Float32x2 || @test nnegzero > 0
    # In 1D every mirrored region is the block's own reflection, which is
    # always local; from 2D on an edge region's mirror image lies in a
    # tangential neighbor, which may be another rank's.
    D == 1 || @test nmirror > 0
end

@testset "The staged interface restriction reproduces the serial one bitwise: D=$D, T=$T" for
        D in (1, 2, 3), T in (Float64, Float32x2)
    # The interface restriction overwrites a coarse block's own face
    # values with the fine side's, so a fine side on another rank is a
    # message; delivered, every rank's array must be the serial one on
    # its blocks bit for bit, over every centering with a face dimension.
    nremote = 0
    for kinds in EXCHANGE_KINDS[D]
        serial = faces_forest(kinds; T=T, N=8)
        centers = filter(C -> any(==(:vertex), C), allcenterings(Val(D)))
        T === Float64 || (centers = [last(centers)])
        for C in centers, P in (2, 3, 5)
            G = ntuple(d -> C[d] === :vertex ? 0 : 1, D)
            same, nr, bad = lockstep_interfaces(serial, P, C, G, T)
            @test same
            @test bad == 0
            nremote += nr
        end
    end
    @test nremote > 0
end

# --- The driver through a communicator ------------------------------------
#
# The lockstep above runs the pieces of a stage; the staged driver that a
# real run takes composes them through the communicator verbs, and step 3
# adds only MPI's methods for those verbs. A mailbox in one process can
# stand in for MPI: every rank's fill runs as its own task, a send is a
# copy put into a channel per (sender, receiver, tag), and a wait takes
# from it, yielding to the other ranks' tasks until the data is there.

@testset "The staged driver over a communicator reproduces the serial exchange: D=$D" for
        D in (2, 3)
    # The composition: receives posted before the pack, sends after it,
    # the local groups while the messages are in flight, the unpack after
    # the wait, the hook after the first stage, the sends waited on at
    # the end — run by every rank as its own task, so that a rank which
    # waited for a message its peer had not yet posted would deadlock
    # rather than pass. The ghosts and the interfaces must come out bit
    # for bit as the serial ones.
    kinds = D == 2 ? (:outer, :reflect_both) : (:reflect_lo, :outer, :periodic)
    serial = faces_forest(kinds; N=8)
    nvars = length(EXCHANGE_PARITY)
    for C in (cellcentered(D), vertexcentered(D)), P in (3, 5)
        G = exchange_ghosts(C, PointValue)
        sfs = FieldSet(serial, nvars; G=G, centering=C, parity=EXCHANGE_PARITY)
        data = exchange_data(MersenneTwister(7), Float64, size(sfs.work))
        copyto!(sfs.work, data)
        hook = exchange_hook()
        fill_ghosts!(sfs, GhostSchedule(sfs, XOPS4); boundary=hook)
        box = Mailbox()
        forests = [Forest(serial.roots; N=serial.N, periodic=serial.periodic,
                          reflecting=TreeAMR.reflecting_faces(serial),
                          outflow=TreeAMR.outflow_faces(serial), extents=serial.extents,
                          TreeAMR.seam_keywords(serial)...,
                          leaves=serial.leaves, comm=MailboxCommunicator(r, P, box))
                   for r in 0:(P - 1)]
        sets = map(forests) do forest
            fs = FieldSet(forest, nvars; G=G, centering=C, parity=EXCHANGE_PARITY)
            copyto!(fs.work, data[ntuple(_ -> :, D + 1)..., blockrange(forest)])
            fs
        end
        scheds = [GhostSchedule(fs, XOPS4) for fs in sets]
        # Through `fill_ghosts!` itself, which runs a distributed forest
        # from step 3 on.
        @sync for r in 1:P
            @async fill_ghosts!(sets[r], scheds[r]; boundary=hook)
        end
        @test box.nmessages > 0
        @test all(isempty ∘ last, box.channels)          # every message consumed
        @test all(r -> bitwise_equal(sets[r].work,
                                     sfs.work[ntuple(_ -> :, D + 1)...,
                                              blockrange(forests[r])]), 1:P)

        # The interface restriction, over the vertex-like dimensions.
        any(==(:vertex), C) || continue
        sflux = FieldSet(serial, nvars; G=0, centering=C, parity=EXCHANGE_PARITY)
        fdata = exchange_data(MersenneTwister(8), Float64, size(sflux.work))
        copyto!(sflux.work, fdata)
        restrict_interfaces!(sflux, InterfaceSchedule(sflux))
        fluxes = map(forests) do forest
            fs = FieldSet(forest, nvars; G=0, centering=C, parity=EXCHANGE_PARITY)
            copyto!(fs.work, fdata[ntuple(_ -> :, D + 1)..., blockrange(forest)])
            fs
        end
        ischeds = [InterfaceSchedule(fs) for fs in fluxes]
        before = box.nmessages
        @sync for r in 1:P
            @async restrict_interfaces!(fluxes[r], ischeds[r])
        end
        @test box.nmessages > before
        @test all(r -> bitwise_equal(fluxes[r].work,
                                     sflux.work[ntuple(_ -> :, D + 1)...,
                                                blockrange(forests[r])]), 1:P)
    end
end

@testset "Grouped prolongation stages over a communicator fill as serially: D=$D" for
        D in (2, 3)
    # A rank that runs several prolongation levels as one group posts
    # every level's receives and sends before it waits for any, and its
    # peers may group otherwise. A send posted only after a wait, a
    # message under another level's tag or into another level's buffer,
    # an unpack before its wait, or a hook region run before the level
    # that fills what it reads would deadlock here, or change a ghost.
    # The nested octant groups the levels each rank holds, and its
    # levels' sources lie on different ranks, so the groups carry
    # messages.
    serial = nested_octant(Val(D))
    parity = [ntuple(_ -> EvenParity, D)]
    ops = Operators(prolongation=6, restriction=6)
    f = (x, v) -> sin(sum(x)) + x[1]^2
    for (C, hook) in ((vertexcentered(D), CellBoundary(Relative(x -> cos(sum(x))))),
                      (cellcentered(D), boundary_by_coordinates(f))), P in (2, 3)
        sfs = FieldSet(serial, 1; G=3, centering=C, parity=parity)
        data = rand(MersenneTwister(11), size(sfs.work)...) .- 0.5
        copyto!(sfs.work, data)
        fill_ghosts!(sfs, GhostSchedule(sfs, ops); boundary=hook)
        box = Mailbox()
        forests = [Forest(serial.roots; N=serial.N,
                          reflecting=TreeAMR.reflecting_faces(serial),
                          extents=serial.extents, leaves=serial.leaves,
                          comm=MailboxCommunicator(r, P, box))
                   for r in 0:(P - 1)]
        sets = map(forests) do forest
            fs = FieldSet(forest, 1; G=3, centering=C, parity=parity)
            copyto!(fs.work, data[ntuple(_ -> :, D + 1)..., blockrange(forest)])
            fs
        end
        scheds = [GhostSchedule(fs, ops) for fs in sets]
        # Some rank runs a group of levels with messages: of one of them at
        # two ranks, of two or three at three.
        @test any(s -> any(r -> length(r) >= 2 &&
                                count(i -> s.stages[i].remote !== nothing, r) >= P - 1,
                           s.stagegroups), scheds)
        @sync for r in 1:P
            @async fill_ghosts!(sets[r], scheds[r]; boundary=hook)
        end
        @test all(isempty ∘ last, box.channels)          # every message consumed
        @test all(r -> bitwise_equal(sets[r].work,
                                     sfs.work[ntuple(_ -> :, D + 1)...,
                                              blockrange(forests[r])]), 1:P)
    end
end

@testset "An AllVariables boundary hook runs on a rank with no blocks: D=$D" for D in (2, 3)
    # The all-variables hook's length check evaluates the callback at a
    # point of block 1, and a rank of a forest with fewer leaves than
    # ranks has no block 1: it threw there while its peers went on to
    # the next collective, which under MPI is a hang. The empty rank must
    # skip the hook, and the others must still match the serial fill
    # with the per-variable hook bit for bit.
    kinds = D == 2 ? (:outer, :reflect_both) : (:reflect_lo, :outer, :periodic)
    serial = faces_forest(kinds; N=8)
    nvars = length(EXCHANGE_PARITY)
    P = nleaves(serial) + 1
    C = cellcentered(D)
    G = exchange_ghosts(C, PointValue)
    sfs = FieldSet(serial, nvars; G=G, centering=C, parity=EXCHANGE_PARITY)
    data = exchange_data(MersenneTwister(9), Float64, size(sfs.work))
    copyto!(sfs.work, data)
    fill_ghosts!(sfs, GhostSchedule(sfs, XOPS4); boundary=exchange_hook())
    whole = CellBoundary(AllVariables((x, δ) -> ntuple(v -> exchange_hook().g(x, v, δ),
                                                       Val(nvars))))
    box = Mailbox()
    forests = [Forest(serial.roots; N=serial.N, periodic=serial.periodic,
                      reflecting=TreeAMR.reflecting_faces(serial),
                      outflow=TreeAMR.outflow_faces(serial), extents=serial.extents,
                      TreeAMR.seam_keywords(serial)...,
                      leaves=serial.leaves, comm=MailboxCommunicator(r, P, box))
               for r in 0:(P - 1)]
    @test count(f -> isempty(blockrange(f)), forests) == 1
    sets = map(forests) do forest
        fs = FieldSet(forest, nvars; G=G, centering=C, parity=EXCHANGE_PARITY)
        copyto!(fs.work, data[ntuple(_ -> :, D + 1)..., blockrange(forest)])
        fs
    end
    scheds = [GhostSchedule(fs, XOPS4) for fs in sets]
    @sync for r in 1:P
        @async fill_ghosts!(sets[r], scheds[r]; boundary=whole)
    end
    @test all(isempty ∘ last, box.channels)
    @test all(r -> bitwise_equal(sets[r].work,
                                 sfs.work[ntuple(_ -> :, D + 1)...,
                                          blockrange(forests[r])]), 1:P)
end

# --- Element sets (M16) -----------------------------------------------------
#
# An element set is filled on faces only, in one stage: its
# prolongations read owned points only and write only ghosts, so they
# run in phase 1 beside the copies and the restrictions, and a fill is
# one round of messages. The three-level forests of `faces_forest`, with
# periodic, outer and reflecting faces, split over 1–5 ranks, put copies,
# prolongations and restrictions on rank boundaries, and the claims are
# the ones above: the transfers, both ends' layouts, one write per point,
# and the values bit for bit, in lockstep and through the driver.

@testset "An element fill is one stage of the serial transfers, split by rank: D=$D" for
        D in (1, 2, 3)
    # A phase-1 prolongation filed under a later tag on one end of a
    # message, a face transfer lost or doubled at a rank boundary, or an
    # edge transfer that one end derives and the other does not would
    # leave a ghost unwritten, write one twice, or leave a rank waiting
    # for a second round its peers never post.
    nremote = Dict(:copy => 0, :prolong => 0, :restrict => 0)
    for kinds in EXCHANGE_KINDS[D], order in (1, 3)
        serial = faces_forest(kinds; N=8)
        layout(forest) = element_exchange_set(forest, 1, order; parity=[OddParity])
        sched = GhostSchedule(layout(serial), EOPS)
        @test [st.tag for st in sched.stages] == [PHASE1_TAG]
        stransfers = serial_transfers(sched)
        stored = ntuple(_ -> serial.N + 2, D)
        scounts = stage_write_counts([sched], stored, nleaves(serial))
        for P in 1:5
            scheds = [GhostSchedule(layout(rank_forest(serial, r, P)), EOPS)
                      for r in 0:(P - 1)]
            locals, remote, unmatched, bad, nsent = distributed_transfers(scheds)
            @test bad == 0
            @test unmatched == 0
            @test length(locals) + length(remote) == length(stransfers)
            @test Set(locals) ∪ Set(remote) == stransfers
            @test all(s -> [st.tag for st in s.stages] == [PHASE1_TAG], scheds)
            @test layout_mismatches(scheds)[1] == 0
            counts = stage_write_counts(scheds, stored, nleaves(serial))
            @test counts == scounts
            @test all(c -> maximum(c) <= 1, values(counts))
            foreach(t -> nremote[t[2]] += 1, remote)
        end
    end
    # Every kind crossed a rank boundary somewhere.
    @test all(>(0), values(nremote))
end

@testset "Packing and unpacking an element fill reproduces the serial fill bitwise: D=$D, T=$T" for
        (D, T) in ((1, Float64), (2, Float64), (3, Float64), (2, Float32x2))
    # The sender evaluates an element prolongation (P+1 taps of I) and a
    # restriction (2(P+1) taps of R) from its own owned points, and the
    # receiver adds it in as `0 + 1·x`: every rank's array, ghosts
    # included, must be the serial fill's on its blocks bit for bit, the
    # hook at the outer faces included. The edge and corner ghosts, which
    # an element fill never writes, hold random data on every rank, so a
    # rank that wrote one would show here too.
    for kinds in EXCHANGE_KINDS[D], order in (1, 3), P in (2, 3, 5)
        serial = faces_forest(kinds; T=T, N=8)
        same, _, _, bad = lockstep_ghost_fill(serial, P, elementcentered(D), ElementLocal,
                                              T; order=order)
        @test same
        @test bad == 0
    end
end

@testset "The staged driver fills an element set in one round of messages: D=$D" for
        D in (2, 3)
    # Through `fill_ghosts!` over the mailbox, every rank its own task: a
    # rank that waited for a prolongation level's messages, which an
    # element fill does not have, would deadlock, and one that sent its
    # prolongations under another tag than its peer receives them would
    # leave a message unconsumed.
    kinds = D == 2 ? (:outer, :reflect_both) : (:reflect_lo, :outer, :periodic)
    serial = faces_forest(kinds; N=8)
    nvars = length(EXCHANGE_PARITY)
    hook = exchange_hook()
    sfs = element_exchange_set(serial, nvars, 3; parity=EXCHANGE_PARITY)
    data = exchange_data(MersenneTwister(13), Float64, size(sfs.work))
    copyto!(sfs.work, data)
    fill_ghosts!(sfs, GhostSchedule(sfs, EOPS); boundary=hook)
    for P in (3, 5)
        box = Mailbox()
        forests = [Forest(serial.roots; N=serial.N, periodic=serial.periodic,
                          reflecting=TreeAMR.reflecting_faces(serial),
                          outflow=TreeAMR.outflow_faces(serial), extents=serial.extents,
                          leaves=serial.leaves, comm=MailboxCommunicator(r, P, box))
                   for r in 0:(P - 1)]
        sets = map(forests) do forest
            fs = element_exchange_set(forest, nvars, 3; parity=EXCHANGE_PARITY)
            copyto!(fs.work, data[ntuple(_ -> :, D + 1)..., blockrange(forest)])
            fs
        end
        scheds = [GhostSchedule(fs, EOPS) for fs in sets]
        @sync for r in 1:P
            @async fill_ghosts!(sets[r], scheds[r]; boundary=hook)
        end
        @test box.nmessages > 0
        @test all(k -> k[3] == PHASE1_TAG, keys(box.channels))   # one round
        @test all(isempty ∘ last, box.channels)                   # all consumed
        @test all(r -> bitwise_equal(sets[r].work,
                                     sfs.work[ntuple(_ -> :, D + 1)...,
                                              blockrange(forests[r])]), 1:P)
    end
end

# --- Rotating seams (M12) ---------------------------------------------------
#
# A rotated transfer whose source and target are on two ranks is packed
# by the source's rank — the axis map and the variable permutation, the
# unscaled sum — and unpacked by the target's, with the factor column
# that holds the sign of the turn, as a mirrored one holds the parity.
# The quadrant of `rotating_forest` (`ghost_oracles.jl`), refined to
# three levels at the seam and the axis, split over 2, 3 and 5 ranks,
# puts seam transfers of every kind on rank boundaries. The same claims
# as above: the transfers, the layouts, one write per point, and the
# values bit for bit — for a set that turns into itself and for a
# `RotationPair`, whose stages are merged by (stage, member).

@testset "Across a rotating seam the ranks hold the serial transfers, laid out alike and written once: D=$D" for
        D in (2, 3)
    # A rotated transfer lost between ranks leaves a ghost unwritten; one
    # whose orientation a rank derives differently from its peer reads
    # its source through the wrong turn; one filed under another stage
    # runs before what it reads. So over 2, 3 and 5 ranks, for every
    # layout a quadrant takes, the transfers of all ranks together —
    # local, and sent matched with received — are the serial ones with
    # their orientations and factor columns, both ends of every message
    # derive one layout, and every point is written once per stage, as
    # serially. A rotated pack is unscaled; its unpack is a plain copy.
    nrotated = 0
    N = D == 3 ? 4 : 8
    p = D == 3 ? 2 : 4
    ops = Operators(prolongation=p, restriction=p)
    for other in ROTATING_OTHERS[D], layout in rotating_exchange_layouts(D, other; p=p)
        serial = rotating_forest(Val(D); rotating=(1, 2), other=other, N=N)
        sched = GhostSchedule(rotating_set(serial, layout), ops)
        stransfers = serial_transfers(sched)
        @test any(t -> t[4] != 0, stransfers)
        _, _, G, C, _, _ = layout
        stored = ntuple(d -> N + 2G[d] + (C[d] === :vertex), D)
        scounts = stage_write_counts([sched], stored, nleaves(serial))
        for P in (2, 3, 5)
            scheds = [GhostSchedule(rotating_set(rank_forest(serial, r, P), layout), ops)
                      for r in 0:(P - 1)]
            locals, remote, unmatched, bad, nsent = distributed_transfers(scheds)
            @test bad == 0
            @test unmatched == 0
            @test length(locals) + length(remote) == length(stransfers)
            @test Set(locals) ∪ Set(remote) == stransfers
            nrotated += count(t -> t[4] != 0, remote)
            @test layout_mismatches(scheds)[1] == 0
            counts = stage_write_counts(scheds, stored, nleaves(serial))
            @test counts == scounts
            @test all(c -> maximum(c) <= 1, values(counts))
        end
    end
    @test nrotated > 0
end

@testset "Packing and unpacking across a rotating seam reproduces the serial fill bitwise: D=$D" for
        D in (2, 3)
    # The values. A pack that applied the turn's sign would turn the
    # serial fill's `−0` — a component that is zero, negated by the turn
    # — into `+0`; one that read the wrong member of a pair, or turned
    # the wrong way, a whole region. So every rank's working array,
    # ghosts included, must equal the serial fill on its blocks bit for
    # bit, the hook between the stages included, for a set that turns into
    # itself (its `v_{d2}` zero) and for a pair (the second variable of
    # each member zero), with `−0` present and rotated transfers received
    # across rank boundaries.
    N = D == 3 ? 4 : 8
    p = D == 3 ? 2 : 4
    ops = Operators(prolongation=p, restriction=p)
    hook = exchange_hook()
    colons = ntuple(_ -> :, D + 1)
    nnegzero = npairzero = nrot = 0
    for other in ROTATING_OTHERS[D]
        serial = rotating_forest(Val(D); rotating=(1, 2), other=other, N=N)
        layouts = rotating_exchange_layouts(D, other; p=p)
        for layout in layouts[1:2]
            sfs = rotating_set(serial, layout)
            data = rotating_exchange_data(MersenneTwister(21), Float64, size(sfs.work), 3)
            copyto!(sfs.work, data)
            fill_ghosts!(sfs, GhostSchedule(sfs, ops); boundary=hook)
            reference = Array(sfs.work)
            nnegzero += count(x -> iszero(x) && signbit(x), reference)
            for P in (2, 3, 5)
                sets = map(0:(P - 1)) do r
                    fs = rotating_set(rank_forest(serial, r, P), layout)
                    copyto!(fs.work, data[colons..., blockrange(fs.forest)])
                    fs
                end
                scheds = [GhostSchedule(fs, ops) for fs in sets]
                bad = lockstep!(sets, scheds, CPU();
                                between=r -> apply_boundary!(sets[r], hook, scheds[r],
                                                             CPU()))
                @test bad == 0
                @test all(r -> bitwise_equal(sets[r].work,
                                             reference[colons...,
                                                       blockrange(sets[r].forest)]), 1:P)
                nrot += nrotated_remote(scheds)
            end
        end
        # The pair, each member's second variable zero.
        la, lb = layouts[3], layouts[4]
        sa, sb = rotating_set(serial, la), rotating_set(serial, lb)
        rng = MersenneTwister(22)
        da = rotating_exchange_data(rng, Float64, size(sa.work), 2)
        db = rotating_exchange_data(rng, Float64, size(sb.work), 2)
        copyto!(sa.work, da)
        copyto!(sb.work, db)
        fill_ghosts!(RotationPair(sa, sb), (GhostSchedule(sa, ops), GhostSchedule(sb, ops));
                     boundary=hook)
        npairzero += count(x -> iszero(x) && signbit(x), sa.work) +
                     count(x -> iszero(x) && signbit(x), sb.work)
        for P in (2, 3, 5)
            pairs = map(0:(P - 1)) do r
                forest = rank_forest(serial, r, P)
                a, b = rotating_set(forest, la), rotating_set(forest, lb)
                copyto!(a.work, da[colons..., blockrange(forest)])
                copyto!(b.work, db[colons..., blockrange(forest)])
                RotationPair(a, b)
            end
            schedsa = [GhostSchedule(q.a, ops) for q in pairs]
            schedsb = [GhostSchedule(q.b, ops) for q in pairs]
            @test lockstep_pair!(pairs, schedsa, schedsb, CPU(); hooks=(hook, hook)) == 0
            @test all(1:P) do r
                range = blockrange(pairs[r].a.forest)
                bitwise_equal(pairs[r].a.work, sa.work[colons..., range]) &&
                    bitwise_equal(pairs[r].b.work, sb.work[colons..., range])
            end
            nrot += nrotated_remote(schedsa) + nrotated_remote(schedsb)
        end
    end
    @test nnegzero > 0
    @test npairzero > 0
    @test nrot > 0
end

@testset "The staged driver over a communicator fills across a rotating seam: D=$D" for
        D in (2, 3)
    # The composition through the communicator verbs, every rank its own
    # task: the single fill of a set that turns into itself, and the
    # pair's fill, whose two members share each stage's tag — the
    # mailbox, like MPI, delivers the messages between two ranks with one
    # tag in the order they were sent, which is the order of the merged
    # stages on both ends. Bit for bit the serial fills.
    N = D == 3 ? 4 : 8
    p = D == 3 ? 2 : 4
    ops = Operators(prolongation=p, restriction=p)
    other = D == 3 ? :reflect_lo : :outer
    serial = rotating_forest(Val(D); rotating=(1, 2), other=other, N=N)
    layouts = rotating_exchange_layouts(D, other; p=p)
    hook = exchange_hook()
    colons = ntuple(_ -> :, D + 1)
    sfs = rotating_set(serial, layouts[1])
    data = rotating_exchange_data(MersenneTwister(23), Float64, size(sfs.work), 3)
    copyto!(sfs.work, data)
    fill_ghosts!(sfs, GhostSchedule(sfs, ops); boundary=hook)
    sa, sb = rotating_set(serial, layouts[3]), rotating_set(serial, layouts[4])
    da = rotating_exchange_data(MersenneTwister(24), Float64, size(sa.work), 2)
    db = rotating_exchange_data(MersenneTwister(25), Float64, size(sb.work), 2)
    copyto!(sa.work, da)
    copyto!(sb.work, db)
    fill_ghosts!(RotationPair(sa, sb), (GhostSchedule(sa, ops), GhostSchedule(sb, ops));
                 boundary=hook)
    for P in (3, 5)
        box = Mailbox()
        forests = [Forest(serial.roots; N=serial.N, periodic=serial.periodic,
                          reflecting=TreeAMR.reflecting_faces(serial),
                          outflow=TreeAMR.outflow_faces(serial), extents=serial.extents,
                          TreeAMR.seam_keywords(serial)...,
                          leaves=serial.leaves, comm=MailboxCommunicator(r, P, box))
                   for r in 0:(P - 1)]
        sets = map(forests) do forest
            fs = rotating_set(forest, layouts[1])
            copyto!(fs.work, data[colons..., blockrange(forest)])
            fs
        end
        scheds = [GhostSchedule(fs, ops) for fs in sets]
        @sync for r in 1:P
            @async fill_ghosts!(sets[r], scheds[r]; boundary=hook)
        end
        @test box.nmessages > 0
        @test all(isempty ∘ last, box.channels)
        @test all(r -> bitwise_equal(sets[r].work,
                                     sfs.work[colons..., blockrange(forests[r])]), 1:P)
        pairs = map(forests) do forest
            a, b = rotating_set(forest, layouts[3]), rotating_set(forest, layouts[4])
            copyto!(a.work, da[colons..., blockrange(forest)])
            copyto!(b.work, db[colons..., blockrange(forest)])
            RotationPair(a, b)
        end
        pscheds = [(GhostSchedule(q.a, ops), GhostSchedule(q.b, ops)) for q in pairs]
        before = box.nmessages
        @sync for r in 1:P
            @async fill_ghosts!(pairs[r], pscheds[r]; boundary=(hook, hook))
        end
        @test box.nmessages > before
        @test all(isempty ∘ last, box.channels)
        @test all(1:P) do r
            range = blockrange(forests[r])
            bitwise_equal(pairs[r].a.work, sa.work[colons..., range]) &&
                bitwise_equal(pairs[r].b.work, sb.work[colons..., range])
        end
    end
end

# --- Half-turn seams (M14) --------------------------------------------------
#
# A half turn's transfers are orientation 2, packed and unpacked as a
# quarter turn's are; what is new is the shifted seam slab of a
# vertex-like `d2`, whose ranges both ends of a message must derive alike.
# The half plane of `halfturn_forest` (`ghost_oracles.jl`), refined at the
# axis and off it, split over 2, 3 and 5 ranks: the transfers, the layouts,
# one write per point, and the values bit for bit, with `−0` from the
# negated component.

@testset "Across a half-turn seam the ranks hold the serial transfers and its values bitwise: D=$D" for
        D in (2, 3)
    # A shifted slab whose range one end derives differently from the other
    # misplaces a whole message; a pack that applied the sign turns the
    # serial fill's `−0` into `+0`. So over 2, 3 and 5 ranks the transfers
    # of all ranks are the serial ones, both ends of every message derive
    # one layout, every point is written once per stage, and every rank's
    # working array, ghosts included, equals the serial fill on its blocks
    # bit for bit, the hook between the stages included.
    N = D == 3 ? 4 : 8
    p = D == 3 ? 2 : 4
    ops = Operators(prolongation=p, restriction=p)
    hook = exchange_hook()
    colons = ntuple(_ -> :, D + 1)
    nrotated = nnegzero = nrot = 0
    for other in ROTATING_OTHERS[D], layout in halfturn_exchange_layouts(D, other; p=p)
        serial = halfturn_forest(Val(D); plane=(1, 2), other=other, N=N)
        sfs = halfturn_set(serial, layout)
        sched = GhostSchedule(sfs, ops)
        stransfers = serial_transfers(sched)
        @test any(t -> t[4] == 2, stransfers)
        @test all(t -> t[4] in (0, 2), stransfers)
        _, _, G, C, _, _ = layout
        stored = ntuple(d -> N + 2G[d] + (C[d] === :vertex), D)
        scounts = stage_write_counts([sched], stored, nleaves(serial))
        # The serial fill, with `v_{d1}` zero: the turn negates it.
        data = rotating_exchange_data(MersenneTwister(31), Float64, size(sfs.work), 2)
        copyto!(sfs.work, data)
        fill_ghosts!(sfs, sched; boundary=hook)
        reference = Array(sfs.work)
        nnegzero += count(x -> iszero(x) && signbit(x), reference)
        for P in (2, 3, 5)
            sets = map(0:(P - 1)) do r
                fs = halfturn_set(rank_forest(serial, r, P), layout)
                copyto!(fs.work, data[colons..., blockrange(fs.forest)])
                fs
            end
            scheds = [GhostSchedule(fs, ops) for fs in sets]
            locals, remote, unmatched, bad, nsent = distributed_transfers(scheds)
            @test bad == 0
            @test unmatched == 0
            @test Set(locals) ∪ Set(remote) == stransfers
            nrotated += count(t -> t[4] != 0, remote)
            @test layout_mismatches(scheds)[1] == 0
            counts = stage_write_counts(scheds, stored, nleaves(serial))
            @test counts == scounts
            @test lockstep!(sets, scheds, CPU();
                            between=r -> apply_boundary!(sets[r], hook, scheds[r],
                                                         CPU())) == 0
            @test all(r -> bitwise_equal(sets[r].work,
                                         reference[colons..., blockrange(sets[r].forest)]),
                      1:P)
            nrot += nrotated_remote(scheds)
        end
    end
    @test nrotated > 0
    @test nnegzero > 0
    @test nrot > 0
end

@testset "A serial schedule's stages are its phases, and send nothing" begin
    # The serial path must stay what it was: one stage per phase, the
    # phase's own groups, no messages and no buffers, so a serial fill
    # runs the launches and barriers it always ran.
    forest = faces_forest((:reflect_lo, :outer); N=8)
    fs = FieldSet(forest, 1; G=2, parity=[OddParity])
    sched = GhostSchedule(fs, XOPS4)
    @test [st.tag for st in sched.stages] ==
          [PHASE1_TAG; prolongation_tag.(sched.levels)]
    @test sched.stages[1].locals === sched.phase1
    @test all(i -> sched.stages[i + 1].locals === sched.phase2[i], eachindex(sched.phase2))
    @test all(st -> st.remote === nothing, sched.stages)
    @test !occursin("sent", sprint(show, sched))
    distributed = GhostSchedule(FieldSet(rank_forest(forest, 0, 3), 1; G=2,
                                         parity=[OddParity]), XOPS4)
    @test occursin("transfers sent and", sprint(show, distributed))
end

using TreeAMR: ForestDigest, digest_verdict, layouthash

@testset "A forest or layout that differs between ranks is refused on every rank" begin
    # Every forest mutation is collective, and the ranks' forests agree
    # without a message only if the application kept that contract. A
    # schedule over forests that diverged would exchange the wrong data
    # silently, so the build gathers each rank's digest and every rank
    # reaches the same verdict from the same gathered values. The verdict
    # is checked here on digests of real forests; `mpi_tests.jl` checks
    # it over MPI, where one rank refines on its own.
    forest = faces_forest((:outer, :outer); N=8)
    other = faces_forest((:outer, :outer); N=8)
    refine!(other, [other.leaves[1]])
    balance!(other)
    layout = layouthash((2, 2), (:cell, :cell))
    same = ForestDigest(forest, layout, false)
    @test digest_verdict(fill(same, 3), "GhostSchedule", 1) === nothing
    # The fold covers every leaf, not the samples `hash(::Vector)` takes.
    @test ForestDigest(other, layout, false).leaves != same.leaves
    diverged = [same, same, ForestDigest(other, layout, false)]
    for r in 0:2
        @test_throws "the forest differs between ranks" digest_verdict(diverged,
                                                                       "GhostSchedule", r)
        @test_throws "rank(s) 2 hold a different one" digest_verdict(diverged,
                                                                     "GhostSchedule", r)
        @test_throws "is collective" digest_verdict(diverged, "GhostSchedule", r)
    end
    # Every rank throws the same message, whichever rank it is.
    message(r) = try
        digest_verdict(diverged, "GhostSchedule", r)
    catch err
        err.msg
    end
    @test message(0) == message(1) == message(2)
    # A generation differs even when the leaves agree: a refine undone by
    # a coarsen on one rank only.
    bumped = ForestDigest(same.generation + 1, same.nleaves, same.leaves, same.brick,
                          same.layout, false)
    @test_throws "generation" digest_verdict([same, bumped], "GhostSchedule", 0)
    # Another layout on one rank.
    odd = ForestDigest(forest, layouthash((1, 1), (:cell, :cell)), false)
    @test_throws "different layout on rank(s) 1" digest_verdict([same, odd, same],
                                                                "GhostSchedule", 2)
    # A rank whose own checks refused: it raises its own error, the others
    # say which rank refused.
    refused = ForestDigest(forest, UInt(0), true)
    own = ArgumentError("its own reason")
    @test_throws "its own reason" digest_verdict([same, refused], "GhostSchedule", 1, own)
    @test_throws "refused on rank(s) 1 of 2" digest_verdict([same, refused],
                                                            "GhostSchedule", 0)
end
