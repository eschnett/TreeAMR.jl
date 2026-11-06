# The definitions of `regrid_exchange_tests.jl`, split off so that each test file runs
# on its own: every test process includes all of `helpers/` ("Parallel
# groups" in CODE.md).

using TreeAMR: regrid_sources, regrid_stage, split_regrid, equalsplit, MAX_LEVEL

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
# received; the number of mismatched layout entries and segments, and of
# ranks whose own classification kept other transfers than the whole
# one; and the number of transfers that crossed a rank boundary. An
# element set (M16) is `C = elementcentered(D)`, `family = ElementLocal`
# and its element order as `order`.
function lockstep_regrid(serial::Forest{D}, P, C, family, ::Type{T};
                         bounded=false, order=nothing) where {D,T}
    ops = exchange_operators(C, family; bounded=bounded)
    G = exchange_ghosts(C, family)
    nvars = length(EXCHANGE_PARITY)
    forest = Forest{T}(serial.roots; N=serial.N, periodic=serial.periodic,
                       reflecting=TreeAMR.reflecting_faces(serial),
                       outflow=TreeAMR.outflow_faces(serial), extents=serial.extents,
                       TreeAMR.seam_keywords(serial)...,
                       leaves=serial.leaves)
    fs = FieldSet{T}(forest, nvars; G=G, centering=C, P=order, parity=EXCHANGE_PARITY)
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
    # Each rank classifies only the new leaves it owns and those whose
    # sources it owns, as `regrid!` does; what it keeps of them must be
    # what it keeps of the whole classification.
    rpairs = [regrid_sources(oldleaves, newleaves; oldrange=oldranges[r],
                             newrange=newranges[r]) for r in 1:P]
    bad = count(r -> split_regrid(rpairs[r], oldranges[r], newranges[r]) !=
                     split_regrid(pairs, oldranges[r], newranges[r]), 1:P)
    stages = [regrid_stage(T, serial.N, fs.G, staggers(fs), ops, CPU(), rpairs[r];
                           P=fs.P, oldrange=oldranges[r], newrange=newranges[r], nold=nold,
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
