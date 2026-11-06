# The definitions of `exchange_tests.jl`, split off so that each test file runs
# on its own: every test process includes all of `helpers/` ("Parallel
# groups" in CODE.md).

using TreeAMR: PHASE1_TAG, prolongation_tag, interface_tag, ntransfers, boxsize,
               stagebuffers, segment_ranges, pack_stage!, unpack_stage!, run_phase!,
               apply_boundary!, exchange_ghosts!, exchange_interfaces!, leafowner,
               remote_neighbors, equalsplit_part, ntarget
using KernelAbstractions: CPU, synchronize
using MultiFloats: Float32x2
using TreeAMR: ForestDigest, digest_verdict, layouthash

# The face kinds each dimension is tested over: periodic, outer and
# reflecting faces all occur, in every dimension count.
const EXCHANGE_KINDS = Dict(1 => [(:reflect_hi,), (:periodic,), (:outer,)],
                            2 => [(:reflect_lo, :periodic), (:outer, :reflect_both)],
                            3 => [(:reflect_lo, :outer, :periodic)])

# Three variables: two generic ones of either parity and an odd one that
# is identically zero — the normal momentum of a fluid at rest — whose
# mirrored ghosts are `−0` in the serial fill.
const EXCHANGE_PARITY = [EvenParity, OddParity, OddParity]

const XOPS4 = Operators(prolongation=4, restriction=4)

exchange_operators(C, family; bounded=false) =
    family === ElementLocal ? Operators(family=ElementLocal) :
    family === Conservative ?
    Operators(prolongation=3, restriction=2, family=Conservative, bounded=bounded) :
    Operators(prolongation=4, restriction=4, bounded=bounded)

# An element set (M16) has one ghost layer, the face's.
exchange_ghosts(C, family) =
    family === ElementLocal ? ntuple(_ -> 1, length(C)) :
    ghosts_for(C, family === Conservative ? 3 : 4)

# The schedule's stencils as plain values, so that two schedules' can be
# compared by `==`.
stencil_values(s) = (s.targetfirst, Vector(s.srcstart), Matrix(s.weights))

# Every transfer of a serial schedule, filed by the stage it runs in:
# `(tag, kind, factorcol, orientation, stencils, target, source)`, the
# orientation that of a rotating seam (M12), 0 elsewhere.
function serial_transfers(sched)
    out = Set{Any}()
    for st in sched.stages, g in st.locals
        sv = map(stencil_values, g.stencils)
        for t in 1:ntransfers(g)
            push!(out, (st.tag, g.kind, g.factorcol, Int(g.orientation), sv,
                        Int(g.targetblocks[t]), Int(g.sourceblocks[t])))
        end
    end
    return out
end

# The same set assembled from the schedules of every rank, and what went
# wrong on the way. Local groups are shifted to global indices; a sent
# transfer is matched to its received half by `(tag, key, target,
# source)`, and the pair is the serial transfer: the pack supplies the
# kind, the orientation, the source windows and the weights, the unpack
# the target box and the factor column — the parity, and the sign of a
# rotated transfer. `bad` counts every inconsistency between a
# group and the layout entry of its slot.
function distributed_transfers(scheds)
    locals = Any[]
    sends = Dict{Any,Any}()
    recvs = Dict{Any,Any}()
    bad = 0
    for sched in scheds
        forest = sched.forest
        offset = first(blockrange(forest)) - 1
        for st in sched.stages
            for g in st.locals
                sv = map(stencil_values, g.stencils)
                for t in 1:ntransfers(g)
                    push!(locals, (st.tag, g.kind, g.factorcol, Int(g.orientation), sv,
                                   Int(g.targetblocks[t]) + offset,
                                   Int(g.sourceblocks[t]) + offset))
                end
            end
            remote = st.remote
            remote === nothing && continue
            for g in remote.packs
                bad += count(s -> s.targetfirst != 1, g.stencils)
                bad += g.factorcol != 0             # packs are unscaled
                bad += !issorted(g.sourceblocks)    # run by owner of the source
                for t in 1:ntransfers(g)
                    e = remote.sendlayout[g.targetblocks[t]]
                    bad += e.source != g.sourceblocks[t] + offset
                    bad += e.peer != leafowner(forest, e.target)
                    bad += e.npoints != prod(boxsize(g))
                    key = (st.tag, e.key, e.target, e.source)
                    bad += haskey(sends, key)
                    sends[key] = (g.kind, Int(g.orientation),
                                  map(s -> (Vector(s.srcstart), Matrix(s.weights)),
                                      g.stencils))
                end
            end
            for g in remote.unpacks
                bad += !issorted(g.targetblocks)    # run by owner of the target
                bad += g.orientation != 0           # an unpack is a plain copy
                bad += count(s -> Vector(s.srcstart) != 1:ntarget(s) ||
                                  !all(isone, s.weights), g.stencils)
                for t in 1:ntransfers(g)
                    e = remote.recvlayout[g.sourceblocks[t]]
                    bad += e.target != g.targetblocks[t] + offset
                    bad += e.peer != leafowner(forest, e.source)
                    bad += e.npoints != prod(boxsize(g))
                    key = (st.tag, e.key, e.target, e.source)
                    bad += haskey(recvs, key)
                    recvs[key] = (g.factorcol, map(s -> s.targetfirst, g.stencils))
                end
            end
            # Every slot is packed once and unpacked once, so every element
            # of both buffers is written once.
            bad += sort!(reduce(vcat, [Vector(g.targetblocks) for g in remote.packs];
                                init=Int32[])) != 1:length(remote.sendlayout)
            bad += sort!(reduce(vcat, [Vector(g.sourceblocks) for g in remote.unpacks];
                                init=Int32[])) != 1:length(remote.recvlayout)
        end
    end
    remote = Any[]
    for (key, (kind, orientation, sw)) in sends
        haskey(recvs, key) || continue
        factorcol, firsts = recvs[key]
        tag, _, target, source = key
        sv = ntuple(d -> (firsts[d], sw[d]...), length(sw))
        push!(remote, (tag, kind, factorcol, orientation, sv, target, source))
    end
    unmatched = length(symdiff(keys(sends), keys(recvs)))
    return locals, remote, unmatched, bad, length(sends)
end

# Rank `r`'s send layout to `s` against `s`'s receive layout from `r`,
# per stage: the same transfers, in the same order, at the same offsets
# within the segment, of the same sizes. Returns the number of
# mismatching (stage, pair) segments and the number of nonempty ones.
function layout_mismatches(scheds)
    P = length(scheds)
    stageof(sched, tag) = (i = findfirst(st -> st.tag == tag, sched.stages);
                           i === nothing ? nothing : sched.stages[i].remote)
    segment(layout, peer) = begin
        es = filter(e -> e.peer == peer, layout)
        base = isempty(es) ? 0 : first(es).offset
        [(e.key, e.target, e.source, e.offset - base, e.npoints) for e in es]
    end
    tags = sort!(unique!([st.tag for sched in scheds for st in sched.stages]))
    bad = nonempty = 0
    for tag in tags, r in 1:P, s in 1:P
        r == s && continue
        a, b = stageof(scheds[r], tag), stageof(scheds[s], tag)
        out = a === nothing ? [] : segment(a.sendlayout, s - 1)
        in = b === nothing ? [] : segment(b.recvlayout, r - 1)
        isempty(out) || (nonempty += 1)
        bad += out != in
        # The segment lengths the counts give agree with the entries.
        a === nothing || (i = findfirst(==(s - 1), a.sendpeers);
                          bad += (i === nothing ? 0 : a.sendcounts[i]) !=
                                 sum(e -> e[5], out; init=0))
        b === nothing || (i = findfirst(==(r - 1), b.recvpeers);
                          bad += (i === nothing ? 0 : b.recvcounts[i]) !=
                                 sum(e -> e[5], in; init=0))
    end
    return bad, nonempty
end

# How often each stage writes each stored point of each leaf, from the
# local groups and the unpacks of every rank (or from the serial stages,
# for one schedule), as one count array per tag.
function stage_write_counts(scheds, stored, n)
    counts = Dict{Int,Array{Int}}()
    for sched in scheds
        offset = first(blockrange(sched.forest)) - 1
        for st in sched.stages
            c = get!(() -> zeros(Int, stored..., n), counts, st.tag)
            groups = st.remote === nothing ? st.locals : [st.locals; st.remote.unpacks]
            for g in groups
                blen = boxsize(g)
                tfirst = map(s -> s.targetfirst, g.stencils)
                for t in 1:ntransfers(g), off in CartesianIndices(map(l -> 0:(l - 1), blen))
                    c[ntuple(d -> tfirst[d] + off[d], length(blen))...,
                      Int(g.targetblocks[t]) + offset] += 1
                end
            end
        end
    end
    filter!(p -> any(!iszero, p.second), counts)
    return counts
end

# Generic data in every stored value, ghosts included, the third variable
# zero: random bits, so that nothing about a value is special, made from
# `Float64` so that every element type gets the same draws.
function exchange_data(rng, ::Type{T}, dims) where {T}
    data = T.(rand(rng, dims...) .- 0.5)
    data[ntuple(_ -> :, length(dims) - 2)..., 3, :] .= zero(T)
    return data
end

# A boundary hook for the outer faces, a pure `T` function of position.
exchange_hook() = CellBoundary((x, v, δ) -> v == 3 ? zero(x[1]) :
                                            x[1] * oftype(x[1], 0.7) + oftype(x[1], v))

# Run every simulated rank's stages in lockstep, tag by tag, as a real run
# interleaves them: pack on every rank, deliver each send segment into the
# peer's receive segment, then run the local groups and unpack on every
# rank. After the first stage, `between(r)` runs on every rank — the
# boundary hook of a ghost fill. Returns the number of segments whose two
# ends disagreed in length.
function lockstep!(sets, scheds, backend; between=nothing)
    P = length(sets)
    tags = sort!(unique!([st.tag for sched in scheds for st in sched.stages]))
    stageat(r, tag) = (i = findfirst(st -> st.tag == tag, scheds[r].stages);
                       i === nothing ? nothing : scheds[r].stages[i])
    bad = 0
    for (k, tag) in enumerate(tags)
        for r in 1:P
            st = stageat(r, tag)
            (st === nothing || st.remote === nothing) && continue
            pack_stage!(sets[r], st.remote, stagebuffers(st.remote, sets[r].nvars, backend),
                        backend)
        end
        for r in 1:P
            st = stageat(r, tag)
            (st === nothing || st.remote === nothing) && continue
            nv = sets[r].nvars
            sendbuf = stagebuffers(st.remote, nv, backend)[1]
            for (peer, range) in zip(st.remote.sendpeers,
                                     segment_ranges(st.remote.sendcounts, nv))
                other = stageat(peer + 1, tag).remote
                i = findfirst(==(r - 1), other.recvpeers)
                into = segment_ranges(other.recvcounts, nv)[i]
                bad += length(into) != length(range)
                copyto!(view(stagebuffers(other, nv, backend)[2], into),
                        view(sendbuf, range))
            end
        end
        for r in 1:P
            st = stageat(r, tag)
            st === nothing && continue
            run_phase!(sets[r], st.locals, backend)
            st.remote === nothing ||
                unpack_stage!(sets[r], st.remote,
                              stagebuffers(st.remote, sets[r].nvars, backend), backend)
            synchronize(backend)
        end
        if k == 1 && between !== nothing
            foreach(between, 1:P)
        end
    end
    return bad
end

# Bitwise equality, `−0` against `+0` and NaN payloads included.
bitwise_equal(a, b) = size(a) == size(b) && all(map(===, a, b))

# The serial ghost fill and the lockstep one over `P` simulated ranks,
# from the same data. Returns whether every rank's working array,
# ghosts included, equals the serial one on its blocks bit for bit; the
# number of `−0` in the serial result; how many mirrored transfers were
# received across a rank boundary; and the lockstep's segment mismatches.
# An element set (M16) is `C = elementcentered(D)`, `family =
# ElementLocal` and its element order as `order`.
function lockstep_ghost_fill(serial::Forest{D}, P, C, family, ::Type{T};
                             backend=CPU(), seed=1, bounded=false,
                             order=nothing) where {D,T}
    ops = exchange_operators(C, family; bounded=bounded)
    G = exchange_ghosts(C, family)
    nvars = length(EXCHANGE_PARITY)
    sfs = FieldSet{T}(serial, nvars; G=G, centering=C, P=order, parity=EXCHANGE_PARITY,
                      backend=backend)
    data = exchange_data(MersenneTwister(seed), T, size(sfs.work))
    copyto!(sfs.work, data)
    hook = exchange_hook()
    fill_ghosts!(sfs, GhostSchedule(sfs, ops); boundary=hook)
    reference = Array(sfs.work)
    sets, scheds = [], []
    for r in 0:(P - 1)
        forest = rank_forest(serial, r, P)
        fs = FieldSet{T}(forest, nvars; G=G, centering=C, P=order, parity=EXCHANGE_PARITY,
                         backend=backend)
        copyto!(fs.work, data[ntuple(_ -> :, D + 1)..., blockrange(forest)])
        push!(sets, fs)
        push!(scheds, GhostSchedule(fs, ops))
    end
    bad = lockstep!(sets, scheds, backend;
                    between=r -> apply_boundary!(sets[r], hook, scheds[r], backend))
    same = all(1:P) do r
        bitwise_equal(Array(sets[r].work),
                      reference[ntuple(_ -> :, D + 1)..., blockrange(sets[r].forest)])
    end
    nnegzero = count(x -> iszero(x) && signbit(x), reference)
    nmirror = sum(scheds) do sched
        sum(st -> st.remote === nothing ? 0 :
                  sum(g -> g.factorcol == 0 ? 0 : ntransfers(g), st.remote.unpacks;
                      init=0), sched.stages; init=0)
    end
    return same, nnegzero, nmirror, bad
end

# The same for the interface restriction, over a flux-like field set.
function lockstep_interfaces(serial::Forest{D}, P, C, G, ::Type{T};
                             backend=CPU(), seed=2) where {D,T}
    nvars = length(EXCHANGE_PARITY)
    sfs = FieldSet{T}(serial, nvars; G=G, centering=C, parity=EXCHANGE_PARITY,
                      backend=backend)
    data = exchange_data(MersenneTwister(seed), T, size(sfs.work))
    copyto!(sfs.work, data)
    restrict_interfaces!(sfs, InterfaceSchedule(sfs))
    reference = Array(sfs.work)
    sets, scheds = [], []
    for r in 0:(P - 1)
        forest = rank_forest(serial, r, P)
        fs = FieldSet{T}(forest, nvars; G=G, centering=C, parity=EXCHANGE_PARITY,
                         backend=backend)
        copyto!(fs.work, data[ntuple(_ -> :, D + 1)..., blockrange(forest)])
        push!(sets, fs)
        push!(scheds, InterfaceSchedule(fs))
    end
    bad = lockstep!(sets, scheds, backend)
    nremote = sum(s -> sum(st -> st.remote === nothing ? 0 : length(st.remote.recvlayout),
                           s.stages; init=0), scheds)
    same = all(1:P) do r
        bitwise_equal(Array(sets[r].work),
                      reference[ntuple(_ -> :, D + 1)..., blockrange(sets[r].forest)])
    end
    return same, nremote, bad
end

mutable struct Mailbox
    lock::ReentrantLock
    channels::Dict{NTuple{3,Int},Channel{Any}}
    nmessages::Int
end

Mailbox() = Mailbox(ReentrantLock(), Dict{NTuple{3,Int},Channel{Any}}(), 0)

struct MailboxCommunicator <: TreeAMR.Communicator
    rank::Int
    size::Int
    box::Mailbox
end

TreeAMR.commrank(c::MailboxCommunicator) = c.rank

TreeAMR.commsize(c::MailboxCommunicator) = c.size

# The schedules are built one rank after another, so the digest gather
# cannot be a real collective here; the forests are copies of one, as in
# `PartitionCommunicator`'s.
TreeAMR.allgather(c::MailboxCommunicator, d::TreeAMR.ForestDigest) = fill(d, c.size)

mailbox_channel(box, from, to, tag) =
    lock(box.lock) do
        get!(() -> Channel{Any}(Inf), box.channels, (from, to, tag))
    end

struct SentMessage end

struct PendingReceive{B}
    channel::Channel{Any}
    buf::B
end

function TreeAMR.isend(c::MailboxCommunicator, buf::AbstractVector, peer::Integer,
                       tag::Integer)
    peer != c.rank || error("a rank sends to itself")
    lock(() -> (c.box.nmessages += 1), c.box.lock)
    put!(mailbox_channel(c.box, c.rank, peer, tag), copy(buf))
    return SentMessage()
end

TreeAMR.irecv(c::MailboxCommunicator, buf::AbstractVector, peer::Integer,
              tag::Integer) = PendingReceive(mailbox_channel(c.box, peer, c.rank, tag), buf)

function TreeAMR.waitall(c::MailboxCommunicator, requests::AbstractVector)
    for q in requests
        q isa PendingReceive || continue
        # A wait that never returns is a deadlock of the driver; say so
        # instead of hanging the suite.
        timedwait(() -> isready(q.channel), 120.0) === :ok ||
            error("rank $(c.rank) waited 120 s for a message: the stages deadlock")
        data = take!(q.channel)
        length(data) == length(q.buf) ||
            error("rank $(c.rank) received $(length(data)) values for a segment of " *
                  "$(length(q.buf))")
        copyto!(q.buf, data)
    end
    return nothing
end

const EOPS = Operators(family=ElementLocal)

# The element layout of the tests below: order `order`, one ghost layer.
element_exchange_set(forest::Forest{D,T}, nvars, order; parity) where {D,T} =
    FieldSet{T}(forest, nvars; G=ntuple(_ -> 1, D), centering=elementcentered(D), P=order,
                parity=parity)

# The layouts over a quadrant with `other` the third dimension: a scalar
# and a vector, cell- and vertex-centered, which turn into themselves;
# and the two face-centered members of a pair, `(B_{d1}, F_{d1})` and
# `(F_{d2}, B_{d2})`, as in `rotating_pair_vs_quadrupled`. Each is
# `(name, nvars, G, centering, parity, rotation)`.
function rotating_exchange_layouts(D, other; p=4)
    rotating = (1, 2)
    z = outofplane(D, rotating)
    swap(t) = (t[2], t[1], t[3:end]...)
    parity(odd) = other === :reflect_lo ?
                  [ntuple(d -> d != z ? NoParity : o ? OddParity : EvenParity, D)
                   for o in odd] : nothing
    vector = vector_rotation(D, rotating)
    Ca = facecentered(D, 1)
    return [("cell", D + 1, ghosts_for(cellcentered(D), p), cellcentered(D),
             vector_parity(D, rotating, other), vector),
            ("vertex", D + 1, ghosts_for(vertexcentered(D), p), vertexcentered(D),
             vector_parity(D, rotating, other), vector),
            ("a", 2, ghosts_for(Ca, p), Ca, parity((false, true)), [-2, -1]),
            ("b", 2, swap(ghosts_for(Ca, p)), swap(Ca), parity((true, false)), [2, 1])]
end

rotating_set(forest, (_, nvars, G, C, parity, rotation), ::Type{T}=Float64) where {T} =
    FieldSet{T}(forest, nvars; G=G, centering=C, parity=parity, rotation=rotation)

# The quadrants: in 2D, and in 3D with the third dimension reflecting at
# its low face — so that a transfer can be mirrored and turned at once —
# or periodic.
const ROTATING_OTHERS = Dict(2 => (:outer,), 3 => (:reflect_lo, :periodic))

# Random data over every stored value, ghosts included, with the
# variable `zero` identically zero: a component whose turned image is
# negated, so that the serial fill writes `−0` across the seam.
function rotating_exchange_data(rng, ::Type{T}, dims, zero) where {T}
    data = T.(rand(rng, dims...) .- 0.5)
    data[ntuple(_ -> :, length(dims) - 2)..., zero, :] .= T(0)
    return data
end

# The pair's stages in lockstep, merged as `fill_ghosts!(pair, …)` merges
# them: per tag, `a`'s stage on every rank — pack, deliver, local groups
# and unpack — then `b`'s, the hooks of both after phase 1. Each member
# packs from its partner for an odd turn and reads its own pair tables.
function lockstep_pair!(pairs, schedsa, schedsb, backend; hooks)
    P = length(pairs)
    member(r, m) = m == 1 ?
                   (pairs[r].a, pairs[r].b, pairs[r].afactors, pairs[r].arotvars, schedsa[r]) :
                   (pairs[r].b, pairs[r].a, pairs[r].bfactors, pairs[r].brotvars, schedsb[r])
    stageat(sched, tag) = (i = findfirst(st -> st.tag == tag, sched.stages);
                           i === nothing ? nothing : sched.stages[i])
    tags = sort!(unique!([st.tag for s in [schedsa; schedsb] for st in s.stages]))
    bad = 0
    for (k, tag) in enumerate(tags), m in 1:2
        for r in 1:P
            fs, alt, _, rotvars, sched = member(r, m)
            st = stageat(sched, tag)
            (st === nothing || st.remote === nothing) && continue
            pack_stage!(fs.work, st.remote, stagebuffers(st.remote, fs.nvars, backend),
                        fs.nvars, backend; altsrc=alt.work, rotvars=rotvars)
        end
        for r in 1:P
            fs, _, _, _, sched = member(r, m)
            st = stageat(sched, tag)
            (st === nothing || st.remote === nothing) && continue
            nv = fs.nvars
            sendbuf = stagebuffers(st.remote, nv, backend)[1]
            for (peer, range) in zip(st.remote.sendpeers,
                                     segment_ranges(st.remote.sendcounts, nv))
                other = stageat(member(peer + 1, m)[5], tag).remote
                i = findfirst(==(r - 1), other.recvpeers)
                into = segment_ranges(other.recvcounts, nv)[i]
                bad += length(into) != length(range)
                copyto!(view(stagebuffers(other, nv, backend)[2], into),
                        view(sendbuf, range))
            end
        end
        for r in 1:P
            fs, alt, factors, rotvars, sched = member(r, m)
            st = stageat(sched, tag)
            st === nothing && continue
            run_phase!(fs.work, fs.work, st.locals, fs.nvars, backend; factors=factors,
                       altsrc=alt.work, rotvars=rotvars)
            st.remote === nothing ||
                unpack_stage!(fs.work, st.remote,
                              stagebuffers(st.remote, fs.nvars, backend), fs.nvars,
                              factors, backend)
            synchronize(backend)
        end
        if k == 1 && m == 2
            for r in 1:P
                apply_boundary!(pairs[r].a, hooks[1], schedsa[r], backend)
                apply_boundary!(pairs[r].b, hooks[2], schedsb[r], backend)
            end
        end
    end
    return bad
end

# Received rotated transfers, over the schedules of every rank: those
# whose pack turned them.
nrotated_remote(scheds) =
    sum(scheds) do sched
        sum(st -> st.remote === nothing ? 0 :
                  sum(g -> g.orientation == 0 ? 0 : ntransfers(g), st.remote.packs; init=0),
            sched.stages; init=0)
    end

# Cell, vertex and a face normal to `d2`, each a scalar and a vector, as
# `(name, nvars, G, centering, parity, halfturn)`.
function halfturn_exchange_layouts(D, other; p=4)
    plane = (1, 2)
    return [(string(C), D + 1, ghosts_for(C, p), C, vector_parity(D, plane, other),
             vector_halfturn(D, plane))
            for C in (cellcentered(D), vertexcentered(D), facecentered(D, 2))]
end

halfturn_set(forest, (_, nvars, G, C, parity, map_), ::Type{T}=Float64) where {T} =
    FieldSet{T}(forest, nvars; G=G, centering=C, parity=parity, halfturn=map_)
