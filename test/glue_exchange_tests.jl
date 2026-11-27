# Ghosts interpolated across a glue over a distributed forest (M20). The
# sender computes: each rank locates its own blocks' ghosts, hands those
# whose source another rank owns to that rank in the build's one
# `alltoallv`, and each fill that rank evaluates them into a message.
# Simulated ranks run as tasks over `regrid_exchange_tests.jl`'s
# rendezvous communicator (`helpers/regrid_exchange.jl`), whose collective
# rounds count the collective calls; the forests and atlases are in
# `helpers/glue.jl` and `helpers/ball.jl`. The claim is the serial fill,
# bit for bit, whatever the split.

# The forest of patches `serial` as a rank holds it, over `comm`.
function glue_rank_forest(serial::Forest{D,T}, comm) where {D,T}
    kw = TreeAMR.patch_keywords(serial)
    return Forest{T}(kw.patches; N=serial.N, glue=kw.glue, leaves=serial.leaves,
                     comm=comm)
end

# The layouts: the kinked torus in 2D, the seven-patch ball in 3D with two
# variables, so that a message's slots cannot mix them up, each with its
# set, data and hook. The data differs between the patches by a term in the
# patch number, so that the copies of a glued point differ and stage A's
# mean (M22) is not the identity.
const GLUE_EXCHANGE_CASES = [
    ("the kinked torus", () -> kinked_forest(2), KINKED, KINKED_OPS,
     forest -> kinked_set(forest),
     (ξ, p) -> kinked_wave(KINKED, TreeAMR.chart_map(KINKED, ξ, p)) + 1e-3 * p * ξ[2],
     nothing),
    ("the seven-patch ball", () -> ball_forest(seven_ball(2, 1)), seven_ball(2, 1),
     BALL_OPS, forest -> ball_set(forest; nvars=2),
     (ξ, p) -> sin(sum(TreeAMR.chart_map(seven_ball(2, 1), ξ, p))) + 1e-3 * p * ξ[1],
     ball_hook((ξ, p) -> sin(sum(TreeAMR.chart_map(seven_ball(2, 1), ξ, p)))))]

@testset "Over simulated ranks the interpolated ghosts are the serial ones bit for bit" begin
    # Guards the sender's evaluation: a ghost located on one rank and
    # evaluated on another, through a stencil the owner rebuilds from the
    # location, must come out as the serial fill's, in the slot the
    # target expects; stage A's mean of a glued point whose members lie on
    # several ranks must be the serial one on every rank holding a member,
    # bit for bit; and the fill itself makes no collective call.
    for (name, make, atlas, ops, makeset, f, hook) in GLUE_EXCHANGE_CASES
        serial = make()
        sfs = makeset(serial)
        ssched = GhostSchedule(sfs, ops; atlas=atlas)
        fill_from_owned!(sfs, ssched, f; boundary=hook)
        for P in (2, 3, 5)
            comms = gather_ranks(P)
            results = on_ranks(P) do r
                comm = comms[r]
                forest = glue_rank_forest(serial, comm)
                fs = makeset(forest)
                sched = GhostSchedule(fs, ops; atlas=atlas)
                before = comm.round[]
                fill_from_owned!(fs, sched, f; boundary=hook)
                collectives = comm.round[] - before
                plan = sched.glueplan
                means = sched.means
                serialpart = selectdim(sfs.work, ndims(sfs.work), blockrange(forest))
                return (isequal(fs.work, serialpart), collectives,
                        length(plan.sendsource), length(plan.recvtarget),
                        length(plan.target) + length(plan.recvtarget),
                        mesh_fold((acc, p) -> acc + 1, +, 0, fs; primaries=true),
                        sum(means.sendcounts; init=0), sum(means.recvcounts; init=0))
            end
            @testset "$name over $P ranks" begin
                @test all(first, results)
                @test all(r -> r[2] == 0, results)
                # Some ghosts travel, and every one sent is received.
                @test sum(r -> r[4], results) > 0
                @test sum(r -> r[3], results) == sum(r -> r[4], results)
                @test sum(r -> r[5], results) == length(ssched.glueplan)
                # Stage A's members travel too, every one sent received.
                @test sum(r -> r[8], results) > 0
                @test sum(r -> r[7], results) == sum(r -> r[8], results)
                # A fold over the primaries counts each point once over
                # the ranks.
                @test all(r -> r[6] == mesh_fold((acc, p) -> acc + 1, +, 0, sfs;
                                                 primaries=true), results)
            end
        end
    end
end

# The kinked torus declared aligned, which its maps are not.
struct MisalignedTorus end
TreeAMR.chart_locate(::MisalignedTorus, x) = TreeAMR.chart_locate(KINKED, x)
TreeAMR.chart_map(::MisalignedTorus, ξ, p) = TreeAMR.chart_map(KINKED, ξ, p)
TreeAMR.chart_aligned(::MisalignedTorus, p, q) = (true, true)

@testset "Over simulated ranks a refusal of the glue plan is raised on every rank" begin
    # Guards the collective agreement: a wrong alignment is found on the
    # ranks whose blocks have the ghost, and every rank must refuse rather
    # than wait in the build's alltoallv.
    serial = kinked_forest(2)
    P = 3
    comms = gather_ranks(P)
    results = on_ranks(P) do r
        forest = glue_rank_forest(serial, comms[r])
        fs = kinked_set(forest)
        try
            GhostSchedule(fs, KINKED_OPS; atlas=MisalignedTorus())
            return "built"
        catch err
            return sprint(showerror, err)
        end
    end
    @test all(m -> occursin("not at a node", m) || occursin("was refused on rank", m),
              results)
end

@testset "Over simulated ranks regrid! moves a kinked set's blocks as the serial regrid" begin
    # Guards the regrid of a set interpolated across a glue between ranks:
    # a kept block whose owner changes is a copy between ranks, its face
    # planes included, and the fill before the transfer, stage A's means
    # with members on several ranks among it, is the serial one.
    serial = kinked_forest(4)
    sfs = kinked_set(serial)
    ssched = GhostSchedule(sfs, KINKED_OPS; atlas=KINKED)
    u = collect(1.0:statelength(sfs))
    scatter!(sfs, u)
    @test regrid!(serial, sfs => ssched; flags=fill(Refine, nblocks(sfs)))
    v = statevector(sfs)
    gather!(v, sfs)
    reference = statearray(v, sfs)
    P = 3
    comms = gather_ranks(P)
    results = on_ranks(P) do r
        forest = glue_rank_forest(kinked_forest(4), comms[r])
        fs = kinked_set(forest)
        sched = GhostSchedule(fs, KINKED_OPS; atlas=KINKED)
        # Rank r's share of the serial state, block by block.
        U = statearray(u, kinked_set(kinked_forest(4)))
        w = statevector(fs)
        W = statearray(w, fs)
        for (b, j) in enumerate(blockrange(forest))
            W[:, :, :, b] .= U[:, :, :, j]
        end
        scatter!(fs, w)
        regrid!(forest, fs => sched; flags=fill(Refine, nblocks(fs)))
        x = statevector(fs)
        gather!(x, fs)
        X = statearray(x, fs)
        same = all(((b, j),) -> X[:, :, :, b] == reference[:, :, :, j],
                   enumerate(blockrange(forest)))
        # And the schedule builds again over the new partition.
        GhostSchedule(fs, KINKED_OPS; atlas=KINKED)
        return same, nleaves(forest)
    end
    @test all(first, results)
    @test all(r -> r[2] == nleaves(serial), results)
end
