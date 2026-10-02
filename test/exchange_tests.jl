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

using TreeAMR: PHASE1_TAG, prolongation_tag, interface_tag, ntransfers, boxsize,
               stagebuffers, segment_ranges, pack_stage!, unpack_stage!, run_phase!,
               apply_boundary!, exchange_ghosts!, exchange_interfaces!, leafowner,
               remote_neighbors, equalsplit_part, ntarget
using KernelAbstractions: CPU, synchronize
using MultiFloats: Float32x2

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

exchange_operators(C, family) =
    family === Conservative ?
    Operators(prolongation=3, restriction=2, family=Conservative) :
    Operators(prolongation=4, restriction=4)
exchange_ghosts(C, family) = ghosts_for(C, family === Conservative ? 3 : 4)

# The schedule's stencils as plain values, so that two schedules' can be
# compared by `==`.
stencil_values(s) = (s.targetfirst, Vector(s.srcstart), Matrix(s.weights))

# Every transfer of a serial schedule, filed by the stage it runs in:
# `(tag, kind, factorcol, stencils, target, source)`.
function serial_transfers(sched)
    out = Set{Any}()
    for st in sched.stages, g in st.locals
        sv = map(stencil_values, g.stencils)
        for t in 1:ntransfers(g)
            push!(out, (st.tag, g.kind, g.factorcol, sv, Int(g.targetblocks[t]),
                        Int(g.sourceblocks[t])))
        end
    end
    return out
end

# The same set assembled from the schedules of every rank, and what went
# wrong on the way. Local groups are shifted to global indices; a sent
# transfer is matched to its received half by `(tag, key, target,
# source)`, and the pair is the serial transfer: the pack supplies the
# kind, the source windows and the weights, the unpack the target box
# and the parity column. `bad` counts every inconsistency between a
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
                    push!(locals, (st.tag, g.kind, g.factorcol, sv,
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
                    sends[key] = (g.kind, map(s -> (Vector(s.srcstart), Matrix(s.weights)),
                                              g.stencils))
                end
            end
            for g in remote.unpacks
                bad += !issorted(g.targetblocks)    # run by owner of the target
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
    for (key, (kind, sw)) in sends
        haskey(recvs, key) || continue
        factorcol, firsts = recvs[key]
        tag, _, target, source = key
        sv = ntuple(d -> (firsts[d], sw[d]...), length(sw))
        push!(remote, (tag, kind, factorcol, sv, target, source))
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
function lockstep_ghost_fill(serial::Forest{D}, P, C, family, ::Type{T};
                             backend=CPU(), seed=1) where {D,T}
    ops = exchange_operators(C, family)
    G = exchange_ghosts(C, family)
    nvars = length(EXCHANGE_PARITY)
    sfs = FieldSet{T}(serial, nvars; G=G, centering=C, parity=EXCHANGE_PARITY,
                      backend=backend)
    data = exchange_data(MersenneTwister(seed), T, size(sfs.work))
    copyto!(sfs.work, data)
    hook = exchange_hook()
    fill_ghosts!(sfs, GhostSchedule(sfs, ops); boundary=hook)
    reference = Array(sfs.work)
    sets, scheds = [], []
    for r in 0:(P - 1)
        forest = rank_forest(serial, r, P)
        fs = FieldSet{T}(forest, nvars; G=G, centering=C, parity=EXCHANGE_PARITY,
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
            push!(readers[t[6]], t[5])
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
                          reflecting=serial.reflecting, extents=serial.extents,
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
