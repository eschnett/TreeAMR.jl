# M7 step 4: the distributed regrid, without MPI.
#
# Over a forest distributed between ranks the regrid transfer is one
# stage over the new partition: every transfer of the serial regrid — a
# copy, a prolongation from the parent, or one child's share of a
# restriction — is local, sent (its source's *old* owner evaluates it
# into a buffer) or received (its target's *new* owner copies it into
# place). The ranks are simulated in one process, as in
# `exchange_tests.jl`, whose helpers this file uses: first in lockstep
# with the buffers wired directly, which checks the stage's layouts and
# values against the serial `regrid!`, and then through `regrid!` and
# `adapt_to_initial_data!` themselves, one task per rank, over a
# communicator whose collectives are real rendezvous between the tasks.

using TreeAMR: regrid_sources, regrid_stage, equalsplit, MAX_LEVEL

# Refine the coarsest leaves and coarsen the finest, so that a regrid
# has copies, prolongations and restrictions at once.
regrid_test_flags(forest) =
    [level(k) == maxlevel(forest) ? Coarsen : level(k) == 0 ? Refine : Keep
     for k in forest.leaves]

# Coarsened blocks whose children had more than one owner, over `P` ranks.
function straddling(old, new, P)
    oldindex = Dict(k => i for (i, k) in enumerate(old))
    return count(new) do k
        haskey(oldindex, k) && return false
        level(k) < MAX_LEVEL && all(c -> haskey(oldindex, c), childkeys(k)) || return false
        return length(Set(equalsplit_part(length(old), P, oldindex[c])
                          for c in childkeys(k))) > 1
    end
end

# The serial regrid, from a field set whose ghosts are filled, and the
# same transfer over `P` simulated ranks in lockstep: every rank packs
# from its slice of the old array, every send segment is copied into
# the peer's receive segment, and every rank runs its local groups and
# unpacks into a fresh array of its new blocks. Returns whether every
# rank's new array equals the serial one on its blocks bit for bit; the
# transfers as `(kind, target, source)` over all ranks, local and
# received; the number of mismatched layout entries and segments; and
# the number of transfers that crossed a rank boundary.
function lockstep_regrid(serial::Forest{D}, P, C, family, ::Type{T}) where {D,T}
    ops = exchange_operators(C, family)
    G = exchange_ghosts(C, family)
    nvars = length(EXCHANGE_PARITY)
    forest = Forest{T}(serial.roots; N=serial.N, periodic=serial.periodic,
                       reflecting=serial.reflecting, extents=serial.extents,
                       leaves=serial.leaves)
    fs = FieldSet{T}(forest, nvars; G=G, centering=C, parity=EXCHANGE_PARITY)
    copyto!(fs.work, exchange_data(MersenneTwister(5), T, size(fs.work)))
    hook = exchange_hook()
    fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=hook)
    old = copy(fs.work)
    oldleaves = copy(forest.leaves)
    # The serial regrid fills the ghosts again, from the same interiors,
    # which writes the same values.
    regrid!(forest, fs => GhostSchedule(fs, ops); flags=regrid_test_flags(forest),
            boundary=hook)
    reference, newleaves = fs.work, forest.leaves
    nold, nnew = length(oldleaves), length(newleaves)
    pairs = regrid_sources(oldleaves, newleaves)
    colons = ntuple(_ -> :, D + 1)

    oldranges = [equalsplit(nold, P, r) for r in 1:P]
    newranges = [equalsplit(nnew, P, r) for r in 1:P]
    stages = [regrid_stage(T, serial.N, fs.G, staggers(fs), ops, CPU(), pairs;
                           oldrange=oldranges[r], newrange=newranges[r], nold=nold,
                           nnew=nnew, oldowner=i -> equalsplit_part(nold, P, i) - 1,
                           newowner=j -> equalsplit_part(nnew, P, j) - 1)
              for r in 1:P]
    olds = [old[colons..., oldranges[r]] for r in 1:P]
    fresh = [zeros(T, size(old)[1:(D + 1)]..., length(newranges[r])) for r in 1:P]
    bufs(r) = stagebuffers(stages[r].remote, nvars, CPU())

    for r in 1:P
        stages[r].remote === nothing && continue
        pack_stage!(olds[r], stages[r].remote, bufs(r), nvars, CPU())
    end
    bad = 0
    for r in 1:P
        remote = stages[r].remote
        remote === nothing && continue
        for (peer, range) in zip(remote.sendpeers, segment_ranges(remote.sendcounts, nvars))
            other = stages[peer + 1].remote
            i = findfirst(==(r - 1), other.recvpeers)
            into = segment_ranges(other.recvcounts, nvars)[i]
            bad += length(into) != length(range)
            copyto!(view(bufs(peer + 1)[2], into), view(bufs(r)[1], range))
            # Both ends' entries for this segment, offsets taken from the
            # segment's start: the same transfers, in the same order.
            mine = filter(e -> e.peer == peer, remote.sendlayout)
            theirs = filter(e -> e.peer == r - 1, other.recvlayout)
            rel(es) = [(e.key, e.target, e.source, e.offset - first(es).offset, e.npoints)
                       for e in es]
            bad += rel(mine) != rel(theirs)
        end
    end
    transfers = Tuple{Symbol,Int,Int}[]
    ncrossing = 0
    for r in 1:P
        st = stages[r]
        run_phase!(fresh[r], olds[r], st.locals, nvars, CPU())
        toff, soff = first(newranges[r]) - 1, first(oldranges[r]) - 1
        for g in st.locals, t in 1:ntransfers(g)
            push!(transfers, (g.kind, Int(g.targetblocks[t]) + toff,
                              Int(g.sourceblocks[t]) + soff))
        end
        st.remote === nothing && continue
        unpack_stage!(fresh[r], st.remote, bufs(r), nvars, nothing, CPU())
        for e in st.remote.recvlayout
            push!(transfers, (e.key.kind, e.target, e.source))
            bad += e.peer != equalsplit_part(nold, P, e.source) - 1
            bad += !(e.target in newranges[r])
        end
        bad += any(e -> !(e.source in oldranges[r]) ||
                        e.peer != equalsplit_part(nnew, P, e.target) - 1,
                   st.remote.sendlayout)
        ncrossing += length(st.remote.recvlayout)
    end
    same = all(r -> bitwise_equal(fresh[r], reference[colons..., newranges[r]]), 1:P)
    return same, transfers, bad, ncrossing, straddling(oldleaves, newleaves, P),
           any(r -> isempty(oldranges[r]) || isempty(newranges[r]), 1:P)
end

@testset "The regrid stage moves every block as the serial regrid does: D=$D" for
        D in (1, 2, 3)
    # Over a partition that changes with the mesh, a transfer lost
    # between the old owner and the new one leaves a block zero, and one
    # laid out differently at the two ends puts another block's data in
    # its place. So over 1–5 simulated ranks, and one more rank than
    # leaves in 1D, the transfers held by all ranks together — local and
    # received — must be the serial regrid's exactly, both ends of every
    # message must derive the same layout, every sent transfer must
    # name its source's old owner and its target's new one, and every
    # rank's new blocks must come out bit for bit as the serial
    # regrid's, for every centering. The cases must include coarsened
    # blocks whose children had different owners and ranks without
    # blocks before or after.
    nstraddling = nempty = ncross = 0
    for kinds in EXCHANGE_KINDS[D]
        serial = faces_forest(kinds; N=8)
        old = copy(serial.leaves)
        new = complete_marks(serial, regrid_test_flags(serial))
        stransfers = Set((g.kind, Int(g.targetblocks[t]), Int(g.sourceblocks[t]))
                         for g in TreeAMR.transfer_groups(Float64, serial,
                                                          ntuple(_ -> 1, D),
                                                          ntuple(_ -> 0, D), old, new,
                                                          XOPS4, CPU())
                         for t in 1:ntransfers(g))
        # In 3D the two coarsened groups are leaves 1–8 and 43–50 of 50,
        # which first straddle a rank boundary at 7 ranks.
        counts = D == 1 ? (1, 2, 3, 5, max(length(old), length(new)) + 1) :
                 D == 2 ? (2, 3, 5) : (2, 3, 7)
        for C in allcenterings(Val(D)), P in counts
            same, transfers, bad, nc, ns, empty = lockstep_regrid(serial, P, C,
                                                                  PointValue, Float64)
            @test same
            @test bad == 0
            @test length(transfers) == length(stransfers)
            @test Set(transfers) == stransfers
            P == 1 && @test nc == 0
            nstraddling += ns
            nempty += empty
            ncross += nc
        end
        same, _, bad, _, _, _ = lockstep_regrid(serial, 3, cellcentered(D), Conservative,
                                                Float64)
        @test same && bad == 0
    end
    @test ncross > 0
    @test nstraddling > 0
    D == 1 && @test nempty > 0
end

@testset "The regrid stage is bitwise in Float32 and Float32x2" begin
    # The unpack is `0 + 1·x`, which must return every value a pack
    # produces in each element type, the limbs of a Float32x2 included.
    for T in (Float32, Float32x2)
        serial = faces_forest((:outer, :reflect_both); T=T, N=8)
        same, _, bad, nc, _, _ = lockstep_regrid(serial, 3, cellcentered(2), PointValue, T)
        @test same && bad == 0 && nc > 0
    end
end

# --- regrid! itself, through a communicator ---------------------------------
#
# `regrid!` gathers the forest digest and the flags, so the ranks cannot
# be run one after another as the schedules are above: every rank runs
# as its own task, and the collectives meet in a rendezvous — each rank
# deposits its contribution for the `k`-th collective it enters and
# waits until all have — while the messages go through the mailbox of
# `exchange_tests.jl`.

mutable struct Rendezvous
    lock::ReentrantLock
    rounds::Dict{Int,Vector{Any}}
end
Rendezvous() = Rendezvous(ReentrantLock(), Dict{Int,Vector{Any}}())

struct GatherCommunicator <: TreeAMR.Communicator
    rank::Int
    size::Int
    meet::Rendezvous
    round::Base.RefValue{Int}
    mail::MailboxCommunicator
end
GatherCommunicator(rank, size, meet, box) =
    GatherCommunicator(rank, size, meet, Ref(0), MailboxCommunicator(rank, size, box))
TreeAMR.commrank(c::GatherCommunicator) = c.rank
TreeAMR.commsize(c::GatherCommunicator) = c.size

function rendezvous(c::GatherCommunicator, x)
    k = (c.round[] += 1)
    slot = lock(c.meet.lock) do
        v = get!(() -> Any[nothing for _ in 1:c.size], c.meet.rounds, k)
        v[c.rank + 1] = Some(x)
        v
    end
    timedwait(() -> all(!isnothing, slot), 120.0; pollint=0.001) === :ok ||
        error("rank $(c.rank) waited 120 s in collective $k: the ranks diverged")
    return [something(y) for y in slot]
end
function TreeAMR.allgather(c::GatherCommunicator, x)
    isbits(x) || throw(ArgumentError("allgather sends one isbits value per rank"))
    return typeof(x)[rendezvous(c, x)...]
end
function TreeAMR.allgatherv(c::GatherCommunicator, v::AbstractVector)
    out = eltype(v)[]
    foreach(part -> append!(out, part), rendezvous(c, collect(v)))
    return out
end
# `interpolate` routes points (step 5): every rank deposits its buffer
# and counts, and takes its own segment of each.
function TreeAMR.alltoallv(c::GatherCommunicator, sendbuf::AbstractVector,
                           sendcounts::AbstractVector{<:Integer})
    length(sendcounts) == c.size && sum(sendcounts; init=0) == length(sendbuf) ||
        error("rank $(c.rank) passed $(length(sendcounts)) counts summing to " *
              "$(sum(sendcounts; init=0)) for $(length(sendbuf)) elements")
    recv, counts = eltype(sendbuf)[], Int[]
    for (buf, cnt) in rendezvous(c, (collect(sendbuf), collect(Int, sendcounts)))
        lo = sum(cnt[1:c.rank]; init=0)
        append!(recv, buf[(lo + 1):(lo + cnt[c.rank + 1])])
        push!(counts, cnt[c.rank + 1])
    end
    return recv, counts
end
TreeAMR.isend(c::GatherCommunicator, buf::AbstractVector, peer::Integer, tag::Integer) =
    TreeAMR.isend(c.mail, buf, peer, tag)
TreeAMR.irecv(c::GatherCommunicator, buf::AbstractVector, peer::Integer, tag::Integer) =
    TreeAMR.irecv(c.mail, buf, peer, tag)
TreeAMR.waitall(c::GatherCommunicator, requests::AbstractVector) =
    TreeAMR.waitall(c.mail, requests)

# One communicator per simulated rank, shared by every forest that rank
# builds, so that its collectives are counted in the order it enters them.
gather_ranks(P) = (meet = Rendezvous(); box = Mailbox();
                   [GatherCommunicator(r, P, meet, box) for r in 0:(P - 1)])

# Run `f(r)` for every simulated rank `r in 1:P` as its own task, and
# return the results in rank order.
function on_ranks(f, P)
    results = Vector{Any}(undef, P)
    @sync for r in 1:P
        @async results[r] = f(r)
    end
    return results
end

@testset "regrid! over a distributed forest is the serial regrid, block for block: D=$D" for
        D in (2, 3)
    # The whole driver: the agreed checks, the gather of each rank's own
    # flags (bare flags and (flag, box) pairs mixed, with a buffer), the
    # replicated completion, the distributed ghost fill the prolongation
    # reads, the transfer stage, and a block count that changes on every
    # rank. Several field sets regrid together — different variable
    # counts, ghost widths and centerings, and one only resized — and
    # every rank must end with the serial leaves and the serial arrays
    # on its new blocks, bit for bit, and the same answer.
    kinds = D == 2 ? (:outer, :reflect_both) : (:reflect_lo, :outer, :periodic)
    serial = faces_forest(kinds; N=8)
    hook = exchange_hook()
    layouts = ((3, cellcentered(D)), (2, vertexcentered(D)))
    sets(forest) = map(layouts) do (nvars, C)
        FieldSet(forest, nvars; G=exchange_ghosts(C, PointValue), centering=C,
                 parity=EXCHANGE_PARITY[1:nvars])
    end
    flux(forest) = FieldSet(forest, 1; G=0, centering=facecentered(D, 1),
                            parity=EXCHANGE_PARITY[1:1])
    # A box in a corner of every refined block, so the buffer recruits.
    mark(forest, k) = begin
        f = regrid_test_flags(forest)[findfirst(==(k), forest.leaves)]
        f === Refine ? (Refine, ntuple(_ -> 1:2, D)) : f
    end
    reference = Forest(serial.roots; N=serial.N, periodic=serial.periodic,
                       reflecting=serial.reflecting, extents=serial.extents,
                       leaves=serial.leaves)
    sfs = sets(reference)
    alldata = map(enumerate(sfs)) do (i, fs)
        copyto!(fs.work, rand(MersenneTwister(10 + i), size(fs.work)...) .- 0.5)
        copy(fs.work)
    end
    sflags = Any[mark(reference, k) for k in reference.leaves]
    changed = regrid!(reference, (map(fs -> fs => GhostSchedule(fs, XOPS4), sfs)...,
                                  flux(reference) => nothing);
                      flags=sflags, buffer=2, boundary=hook)
    @test changed

    for P in (3, 5)
        comms = gather_ranks(P)
        results = on_ranks(P) do r
            forest = Forest(serial.roots; N=serial.N, periodic=serial.periodic,
                            reflecting=serial.reflecting, extents=serial.extents,
                            leaves=serial.leaves, comm=comms[r])
            owned = blockrange(forest)
            fss = sets(forest)
            for (fs, all_) in zip(fss, alldata)
                copyto!(fs.work, all_[ntuple(_ -> :, D + 1)..., owned])
            end
            fl = flux(forest)
            pairs = (map(fs -> fs => GhostSchedule(fs, XOPS4), fss)..., fl => nothing)
            flags = Any[mark(forest, forest.leaves[i]) for i in owned]
            got = regrid!(forest, pairs; flags=flags, buffer=2, boundary=hook)
            (got, copy(forest.leaves), blockrange(forest), map(fs -> copy(fs.work), fss),
             size(fl.work))
        end
        @test first(comms).mail.box.nmessages > 0       # blocks did cross ranks
        @test all(isempty ∘ last, first(comms).mail.box.channels)   # all consumed
        @test all(res -> res[1] == changed, results)
        @test all(res -> res[2] == reference.leaves, results)
        for (got, _, owned, works, fluxsize) in results
            for (w, fs) in zip(works, sfs)
                @test bitwise_equal(w, fs.work[ntuple(_ -> :, D + 1)..., owned])
            end
            @test last(fluxsize) == length(owned)
        end
        # Rebuilding the mesh without moving data: every rank's new
        # blocks are zero, and its old flags still decide the same mesh.
        comms = gather_ranks(P)
        results = on_ranks(P) do r
            forest = Forest(serial.roots; N=serial.N, periodic=serial.periodic,
                            reflecting=serial.reflecting, extents=serial.extents,
                            leaves=serial.leaves, comm=comms[r])
            fs = first(sets(forest))
            fs.work .= 1
            flags = Any[mark(forest, forest.leaves[i]) for i in blockrange(forest)]
            regrid!(forest, fs => GhostSchedule(fs, XOPS4); flags=flags, buffer=2,
                    transfer=false)
            (copy(forest.leaves), all(iszero, fs.work), nblocks(fs) == length(blockrange(forest)))
        end
        @test all(res -> res[1] == reference.leaves && res[2] && res[3], results)
    end
end

# --- Host staging (step 8 of M7) ---------------------------------------------
#
# On a device whose memory the MPI cannot read, every stage's messages go
# through host mirrors of its buffers: download after the pack, send and
# receive from the mirrors, upload before the unpack. On the CPU the
# buffers are host memory and go to MPI directly, so that path would
# only ever run on a device, which the suite does not have. This
# communicator forces it on the CPU — it answers `hoststaging` with
# `true` for every buffer — and records the arrays the messages were
# handed out of, so the test can tell that the path it means to test is
# the one that ran.

struct StagingCommunicator <: TreeAMR.Communicator
    inner::GatherCommunicator
    handed::Vector{Any}
end
StagingCommunicator(inner) = StagingCommunicator(inner, Any[])
TreeAMR.hoststaging(::StagingCommunicator, ::AbstractVector) = true
TreeAMR.commrank(c::StagingCommunicator) = TreeAMR.commrank(c.inner)
TreeAMR.commsize(c::StagingCommunicator) = TreeAMR.commsize(c.inner)
TreeAMR.allgather(c::StagingCommunicator, x) = TreeAMR.allgather(c.inner, x)
TreeAMR.allgatherv(c::StagingCommunicator, v::AbstractVector) =
    TreeAMR.allgatherv(c.inner, v)
TreeAMR.alltoallv(c::StagingCommunicator, buf::AbstractVector,
                  counts::AbstractVector{<:Integer}) = TreeAMR.alltoallv(c.inner, buf, counts)
function TreeAMR.isend(c::StagingCommunicator, buf::AbstractVector, peer::Integer,
                       tag::Integer)
    push!(c.handed, parent(buf))
    return TreeAMR.isend(c.inner, buf, peer, tag)
end
function TreeAMR.irecv(c::StagingCommunicator, buf::AbstractVector, peer::Integer,
                       tag::Integer)
    push!(c.handed, parent(buf))
    return TreeAMR.irecv(c.inner, buf, peer, tag)
end
TreeAMR.waitall(c::StagingCommunicator, requests::AbstractVector) =
    TreeAMR.waitall(c.inner, requests)

@testset "Staging the messages through host mirrors delivers the same bits" begin
    # A staged message that lost its download, its upload, or a segment
    # boundary would deliver stale or shifted ghosts. So the ghost fill
    # with the hook, the interface restriction and `regrid!` itself run
    # with every message staged, at 3 ranks, and must reproduce the
    # serial results bit for bit; every buffer handed to the
    # communicator must be one of the stages' mirrors, never a stage
    # buffer; and a second fill must reuse the mirrors, the ones it
    # allocated on first use, and give the same bits.
    D = 2
    serial = faces_forest((:outer, :reflect_both); N=8)
    hook = exchange_hook()
    nvars = length(EXCHANGE_PARITY)
    C = vertexcentered(D)
    G = exchange_ghosts(C, PointValue)
    rebuilt(comm) = Forest(serial.roots; N=serial.N, periodic=serial.periodic,
                           reflecting=serial.reflecting, extents=serial.extents,
                           leaves=serial.leaves, comm=comm)
    reference = rebuilt(nothing)
    sfs = FieldSet(reference, nvars; G=G, centering=C, parity=EXCHANGE_PARITY)
    data = exchange_data(MersenneTwister(21), Float64, size(sfs.work))
    copyto!(sfs.work, data)
    ssched = GhostSchedule(sfs, XOPS4)
    fill_ghosts!(sfs, ssched; boundary=hook)
    filled = copy(sfs.work)
    sflux = FieldSet(reference, nvars; G=0, centering=C, parity=EXCHANGE_PARITY)
    fdata = exchange_data(MersenneTwister(22), Float64, size(sflux.work))
    copyto!(sflux.work, fdata)
    restrict_interfaces!(sflux, InterfaceSchedule(sflux))
    sflags = Any[f === Refine ? (Refine, ntuple(_ -> 1:2, D)) : f
                 for f in regrid_test_flags(reference)]
    @test regrid!(reference, sfs => ssched; flags=sflags, buffer=2, boundary=hook)

    P = 3
    comms = map(StagingCommunicator, gather_ranks(P))
    results = on_ranks(P) do r
        comm = comms[r]
        forest = rebuilt(comm)
        owned = blockrange(forest)
        fs = FieldSet(forest, nvars; G=G, centering=C, parity=EXCHANGE_PARITY)
        copyto!(fs.work, data[:, :, :, owned])
        sched = GhostSchedule(fs, XOPS4)
        fill_ghosts!(fs, sched; boundary=hook)
        work = copy(fs.work)
        mirrors = [m for st in sched.stages if st.remote !== nothing
                   for m in st.remote.mirrors[nvars]]
        buffers = [x for st in sched.stages if st.remote !== nothing
                   for x in st.remote.buffers[nvars]]
        ghosts_handed = copy(comm.handed)
        # The same fill again, now that the mirrors exist.
        fill_ghosts!(fs, sched; boundary=hook)
        again = bitwise_equal(fs.work, work) &&
                all(splat(===), zip(mirrors, [m for st in sched.stages
                                              if st.remote !== nothing
                                              for m in st.remote.mirrors[nvars]]))
        flux = FieldSet(forest, nvars; G=0, centering=C, parity=EXCHANGE_PARITY)
        copyto!(flux.work, fdata[:, :, :, owned])
        isched = InterfaceSchedule(flux)
        restrict_interfaces!(flux, isched)
        imirrors = [m for st in isched.stages if st.remote !== nothing
                    for m in st.remote.mirrors[nvars]]
        empty!(comm.handed)
        flags = Any[sflags[i] for i in owned]
        regrid!(forest, fs => sched; flags=flags, buffer=2, boundary=hook)
        (owned, work, again, mirrors, buffers, ghosts_handed, copy(flux.work), imirrors,
         copy(comm.handed), copy(forest.leaves), blockrange(forest), copy(fs.work))
    end
    for (owned, work, again, mirrors, buffers, handed, fluxwork, imirrors, regridhanded,
         leaves, newowned, regridded) in results
        @test bitwise_equal(work, filled[:, :, :, owned])
        @test again
        @test !isempty(mirrors) && !isempty(handed)
        @test all(h -> any(m -> h === m, mirrors), handed)
        @test !any(h -> any(x -> h === x, buffers), handed)
        @test bitwise_equal(fluxwork, sflux.work[:, :, :, owned])
        @test !isempty(imirrors)
        @test leaves == reference.leaves
        @test bitwise_equal(regridded, sfs.work[:, :, :, newowned])
        # The regrid stage's messages were staged too: handed out of host
        # vectors that are none of the field set's arrays.
        @test !isempty(regridhanded) && all(h -> h isa Vector{Float64}, regridhanded)
    end
end

@testset "adapt_to_initial_data! over a distributed forest starts from ranks without blocks" begin
    # From a single leaf, every rank but one starts empty, and each pass
    # gathers flags from ranks with and without blocks. Both criterion
    # forms — the host callback and a flag vector from `firing_boxes` —
    # must arrive at the serial mesh in the same number of passes, with
    # the serial initial data on every rank's blocks.
    ring = (x, v) -> tanh((sqrt((x[1] - 0.45)^2 + (x[2] - 0.55)^2) - 0.25) / 0.05) + v
    flag(forest) = (b, k) -> begin
        ext = block_extent(forest, k)
        c = ntuple(d -> (ext[d][1] + ext[d][2]) / 2, 2)
        abs(sqrt((c[1] - 0.45)^2 + (c[2] - 0.55)^2) - 0.25) < 0.75 / 2^level(k) &&
            level(k) < 3 ? Refine : Keep
    end
    fires(work, idx, b, x) = abs(work[idx..., 1, b]) < 0.9
    flags(fs) = map(enumerate(firing_boxes(fires, fs))) do (b, (n, box))
        n == 0 && return Keep
        return level(blockkey(fs, b)) < 3 ? (Refine, box) : (Keep, box)
    end
    criteria = (forest -> (; flag=flag(forest)), forest -> (; flags=flags, buffer=2))
    ops = Operators(prolongation=2, restriction=2)
    for criterion in criteria
        reference = Forest((1, 1); N=8)
        sfs = FieldSet(reference, 1; G=1, centering=vertexcentered(2))
        _, spasses, sconv = adapt_to_initial_data!(sfs, ops; initial=ring,
                                                   boundary=boundary_by_coordinates(ring),
                                                   criterion(reference)...)
        @test sconv && spasses > 1
        P = 3
        comms = gather_ranks(P)
        results = on_ranks(P) do r
            forest = Forest((1, 1); N=8, comm=comms[r])
            fs = FieldSet(forest, 1; G=1, centering=vertexcentered(2))
            empty = nblocks(fs) == 0
            _, passes, conv = adapt_to_initial_data!(fs, ops; initial=ring,
                                                     boundary=boundary_by_coordinates(ring),
                                                     criterion(forest)...)
            (empty, passes, conv, copy(forest.leaves), blockrange(forest), copy(fs.work))
        end
        @test count(res -> res[1], results) == P - 1
        for (_, passes, conv, leaves, owned, work) in results
            @test (passes, conv) == (spasses, sconv)
            @test leaves == reference.leaves
            @test bitwise_equal(work, sfs.work[:, :, :, owned])
        end
    end
end

@testset "regrid! refuses an argument some ranks refuse on every rank" begin
    # A refusal raised on one rank alone would leave the others waiting
    # in the gather of the flags, or in the transfer. So a flag box out
    # of range on rank 1 only is refused on all three, rank 1 with its
    # own reason and the others naming it; so is a flag vector of the
    # wrong length on rank 0 (a `DimensionMismatch` there, as serially);
    # and a forest that changed on rank 2 alone is refused by the digest.
    P = 3
    function attempt(make_flags; diverge=false)
        comms = gather_ranks(P)
        on_ranks(P) do r
            forest = Forest((4, 4); N=8, comm=comms[r])
            if diverge && r == 3
                refine!(forest, [forest.leaves[1]])
                balance!(forest)
            end
            fs = FieldSet(forest, 1; G=2)
            # A schedule build is collective too, and would refuse the
            # diverged forest itself; the regrid then only resizes.
            sched = diverge ? nothing : GhostSchedule(fs, XOPS4)
            try
                regrid!(forest, fs => sched; flags=make_flags(r, length(blockrange(forest))))
                nothing
            catch err
                err
            end
        end
    end
    errs = attempt((r, n) -> Any[r == 2 && b == 1 ? (Refine, (1:9, 1:8)) : Keep
                                 for b in 1:n])
    @test errs[2] isa ArgumentError && occursin("outside the interior", errs[2].msg)
    @test all(i -> errs[i] isa ArgumentError &&
                   occursin("regrid! was refused on rank(s) 1 of 3", errs[i].msg), (1, 3))
    errs = attempt((r, n) -> fill(Keep, n + (r == 1)))
    @test errs[1] isa DimensionMismatch
    @test all(i -> occursin("regrid! was refused on rank(s) 0", errs[i].msg), (2, 3))
    errs = attempt((r, n) -> fill(Keep, n); diverge=true)
    @test all(e -> e isa ArgumentError &&
                   occursin("the forest differs between ranks, so regrid! is refused",
                            e.msg), errs)
end
