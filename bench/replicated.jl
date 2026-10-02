# The replicated O(nleaves) costs of M7 at large leaf counts (step 7), in
# one process: rank P÷2 of P simulated by a fake communicator, over the
# tile mesh of bench/mpi.jl (its two-level tile, 176 blocks per rank, 3D,
# N = 16), for growing P. No MPI is needed.
#
#     julia -t T --project=test bench/replicated.jl [P ...]   (default 1 8 64 512)
#
# The fake answers the forest-digest gather by replication, `allgatherv`
# with the global marks the caller stored, and its messages are no-ops,
# so a whole `regrid!` runs on one rank's blocks with every replicated
# pass at the size of the whole forest and no communication time; the
# data it "receives" are not meaningful, only the time. Per P it prints,
# in ms, best of 3–5: the digest's fold over every leaf; the rank's
# `GhostSchedule` build (digest included) and its candidate search
# `remote_neighbors`; the regrid's replicated passes over the global
# marks of bench/mpi.jl's slab refinement — `buffered_flags` with a
# buffer, `complete_marks` (which balances a scratch forest), `balance!`
# alone, the classification `regrid_sources` and its split
# `split_regrid`, and the comparison of the leaf arrays — then the whole
# `regrid!` refining and coarsening on the rank, and as `#` lines
# `complete_marks` when *every* block is a `(Keep, box)` source of a
# 4-cell buffer, and the bytes the marks' `allgatherv` brings every rank.
# CODE.md's M7 step 7 entry has the numbers.
using TreeAMR
using TreeAMR: ForestDigest, buffered_flags, complete_marks, regrid_sources,
               split_regrid, remote_neighbors, equalsplit, RegridMark
using Printf

mutable struct FakeComm <: TreeAMR.Communicator
    rank::Int
    size::Int
    marks::Any
end
TreeAMR.commrank(c::FakeComm) = c.rank
TreeAMR.commsize(c::FakeComm) = c.size
TreeAMR.allgather(c::FakeComm, d::ForestDigest) = fill(d, c.size)
TreeAMR.allgatherv(c::FakeComm, v::AbstractVector) = c.marks
TreeAMR.isend(::FakeComm, buf::AbstractVector, peer::Integer, tag::Integer) = nothing
TreeAMR.irecv(::FakeComm, buf::AbstractVector, peer::Integer, tag::Integer) = nothing
TreeAMR.waitall(::FakeComm, requests::AbstractVector) = nothing

const D, N, R = 3, 16, 4
const OPS = Operators(prolongation=4, restriction=4)

function tile_forest(P; comm=nothing, leaves=nothing)
    roots = ntuple(d -> d == D ? P * R : R, D)
    kw = (; N=N, periodic=ntuple(_ -> true, D),
          extents=ntuple(d -> (0.0, Float64(roots[d])), D))
    leaves === nothing || return Forest(roots; kw..., leaves=leaves, comm=comm)
    forest = Forest(roots; kw...)
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        (ext[1][1] + ext[1][2]) / 2 < R / 2 && mod((ext[D][1] + ext[D][2]) / 2, R) > R / 2
    end
    refine!(forest, targets)
    balance!(forest)
    return forest
end

inslab(f, k) = root_position(f, k.root)[D] == 0
best(f, reps) = (f(); minimum(@elapsed(f()) for _ in 1:reps))
ms(t) = @sprintf("%8.3f", t * 1e3)

function main(Ps)
    @printf("threads=%d D=%d N=%d tile=%d^%d roots (176 blocks per rank)\n",
            Threads.nthreads(), D, N, R, D)
    println("# ms; the schedule and regrid! on rank P÷2")
    println("     P  nleaves | digest  sched  remnbr | buffd  compl balnce  rsrcs  " *
            "split   eq   | regrid↑ regrid↓")
    for P in Ps
        serial = tile_forest(P)
        nl = nleaves(serial)
        rank = P ÷ 2
        comm = FakeComm(rank, P, nothing)
        forest = tile_forest(P; comm=comm, leaves=serial.leaves)
        reps = nl > 50_000 ? 3 : 5

        t_digest = best(() -> ForestDigest(forest, UInt(0), false), 10)
        t_sched = best(() -> GhostSchedule(forest, OPS; G=2), reps)
        t_nbr = best(() -> remote_neighbors(forest, blockrange(forest)), reps)

        flags = [inslab(serial, k) && level(k) == 0 ? Refine : Keep for k in serial.leaves]
        marks = [RegridMark{D}(f, N) for f in flags]
        t_buffered = best(() -> buffered_flags(forest, marks, 4), reps)
        t_complete = best(() -> complete_marks(forest, marks), reps)
        boxed = [RegridMark{D}((Keep, ntuple(_ -> 1:N, D)), N) for _ in serial.leaves]
        t_allsrc = best(() -> complete_marks(forest, boxed; buffer=4), reps)
        newleaves = complete_marks(forest, marks)
        # balance! alone, on the candidate complete_marks balances.
        cand = Forest(serial.roots; N=N, periodic=serial.periodic, extents=serial.extents,
                      leaves=newleaves)
        t_balance = best(() -> balance!(cand), reps)
        t_sources = best(() -> regrid_sources(serial.leaves, newleaves), reps)
        pairs = regrid_sources(serial.leaves, newleaves)
        oldrange = blockrange(forest)
        newrange = equalsplit(length(newleaves), P, rank + 1)
        t_split = best(() -> split_regrid(pairs, oldrange, newrange), reps)
        t_eq = best(() -> newleaves == serial.leaves, 10)

        # The whole regrid! on this rank, refine then coarsen back, messages
        # no-ops: what it costs this rank apart from the communication.
        fs = FieldSet(forest, 2; G=2)
        fill_by_coordinates!((x, v) -> sin(x[1]) + v, fs)
        flux = FieldSet(forest, 2; G=0, centering=facecentered(D, 1))
        tr, tc = Inf, Inf
        for rep in 1:(reps + 1)
            sched = GhostSchedule(fs, OPS)
            up(k) = inslab(forest, k) && level(k) == 0 ? Refine : Keep
            comm.marks = [RegridMark{D}(up(k), N) for k in forest.leaves]
            local_ = flag_blocks((b, k) -> up(k), forest)
            t = @elapsed regrid!(forest, (fs => sched, flux => nothing); flags=local_)
            rep > 1 && (tr = min(tr, t))
            sched = GhostSchedule(fs, OPS)
            down(k) = inslab(forest, k) && level(k) == 1 ? Coarsen : Keep
            comm.marks = [RegridMark{D}(down(k), N) for k in forest.leaves]
            local_ = flag_blocks((b, k) -> down(k), forest)
            t = @elapsed regrid!(forest, (fs => sched, flux => nothing); flags=local_)
            rep > 1 && (tc = min(tc, t))
            forest.leaves == serial.leaves || error("cycle did not return")
        end
        @printf("%6d %8d | %s %s %s | %s %s %s %s %s %s | %s %s\n", P, nl, ms(t_digest),
                ms(t_sched), ms(t_nbr), ms(t_buffered), ms(t_complete), ms(t_balance),
                ms(t_sources), ms(t_split), ms(t_eq), ms(tr), ms(tc))
        @printf("#   complete_marks, every block a (Keep, box) source, buffer 4: %s ms\n", ms(t_allsrc))
        @printf("#   marks gathered per regrid: %.2f MB per rank (%d B a mark)\n",
                nl * sizeof(RegridMark{D}) / 1e6, sizeof(RegridMark{D}))
        flush(stdout)
    end
end

abspath(PROGRAM_FILE) == @__FILE__() && main(isempty(ARGS) ? [1, 8, 64, 512] : parse.(Int, ARGS))
