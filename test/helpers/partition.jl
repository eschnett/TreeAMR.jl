# The definitions of `partition_tests.jl`, split off so that each test file runs
# on its own: every test process includes all of `helpers/` ("Parallel
# groups" in CODE.md).

using TreeAMR: equalsplit, threadchunks, commrank, commsize, allgather, allgatherv,
               alltoallv, isend, irecv, waitall, bcast, commnodes, SerialCommunicator
using TreeAMR: ntransfers

# A rank of a communicator that does not exist: it answers the two verbs
# the partition needs, and every other verb is refused by the package's
# fallback, which is what lets a test see which operations need more.
struct PartitionCommunicator <: TreeAMR.Communicator
    rank::Int
    size::Int
end

TreeAMR.commrank(c::PartitionCommunicator) = c.rank

TreeAMR.commsize(c::PartitionCommunicator) = c.size

# A schedule build gathers every rank's forest digest (step 3). The
# ranks simulated here are copies of one serial forest, built through
# `rank_forest`, so every rank's digest is the one this rank holds: the
# fake answers that one gather, and only it.
TreeAMR.allgather(c::PartitionCommunicator, d::TreeAMR.ForestDigest) = fill(d, c.size)

# The forest `serial` as rank `r` of `P` holds it: the same leaves, the
# same brick, through the validated `leaves` path.
rank_forest(serial::Forest{D,T}, r, P) where {D,T} =
    Forest{T}(serial.roots; N=serial.N, periodic=serial.periodic,
              reflecting=TreeAMR.reflecting_faces(serial),
              outflow=TreeAMR.outflow_faces(serial), extents=serial.extents,
              TreeAMR.seam_keywords(serial)...,
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
