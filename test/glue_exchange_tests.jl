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
# set, data and hook.
const GLUE_EXCHANGE_CASES = [
    ("the kinked torus", () -> kinked_forest(2), KINKED, KINKED_OPS,
     forest -> kinked_set(forest),
     (ξ, p) -> kinked_wave(KINKED, TreeAMR.chart_map(KINKED, ξ, p)), nothing),
    ("the seven-patch ball", () -> ball_forest(seven_ball(2, 1)), seven_ball(2, 1),
     BALL_OPS, forest -> ball_set(forest; nvars=2),
     (ξ, p) -> sin(sum(TreeAMR.chart_map(seven_ball(2, 1), ξ, p))),
     ball_hook((ξ, p) -> sin(sum(TreeAMR.chart_map(seven_ball(2, 1), ξ, p)))))]

@pending 8 "Over simulated ranks the interpolated ghosts are the serial ones bit for bit" begin
    # Guards the sender's evaluation: a ghost located on one rank and
    # evaluated on another, through a stencil the owner rebuilds from the
    # location, must come out as the serial fill's, in the slot the
    # target expects; the phase-1 copies sent for a block that owns its
    # glued high face planes must carry them; and the fill itself makes
    # no collective call.
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
                return (isequal(fs.work, selectdim(sfs.work, ndims(sfs.work), blockrange(forest))), collectives,
                        length(plan.sendsource), length(plan.recvtarget),
                        length(plan.target) + length(plan.recvtarget),
                        mesh_fold((acc, p) -> acc + 1, +, 0, fs))
            end
            @testset "$name over $P ranks" begin
                @test all(first, results)
                @test all(r -> r[2] == 0, results)
                # Some ghosts travel, and every one sent is received.
                @test sum(r -> r[4], results) > 0
                @test sum(r -> r[3], results) == sum(r -> r[4], results)
                @test sum(r -> r[5], results) == length(ssched.glueplan)
                # A fold counts each glued face point once over the ranks.
                @test all(r -> r[6] == mesh_fold((acc, p) -> acc + 1, +, 0, sfs), results)
            end
        end
    end
end

# The kinked torus declared aligned, which its maps are not.
struct MisalignedTorus end
TreeAMR.chart_locate(::MisalignedTorus, x) = TreeAMR.chart_locate(KINKED, x)
TreeAMR.chart_map(::MisalignedTorus, ξ, p) = TreeAMR.chart_map(KINKED, ξ, p)
TreeAMR.chart_aligned(::MisalignedTorus, p, q) = (true, true)

@pending 8 "Over simulated ranks a refusal of the glue plan is raised on every rank" begin
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

@pending 8 "Over simulated ranks regrid! moves the appendix with its block" begin
    # Guards the regrid of a set with an appendix between ranks: a kept
    # block whose owner changes is a copy between ranks, which must carry
    # its owned high face planes under the same key on both ends.
    serial = kinked_forest(4)
    sfs = kinked_set(serial)
    ssched = GhostSchedule(sfs, KINKED_OPS; atlas=KINKED)
    u = collect(1.0:statelength(sfs))
    scatter!(sfs, u)
    @test regrid!(serial, sfs => ssched; flags=fill(Refine, nblocks(sfs)))
    v = statevector(sfs)
    gather!(v, sfs)
    reference = statearray(v, sfs; appendix=true)
    P = 3
    comms = gather_ranks(P)
    results = on_ranks(P) do r
        forest = glue_rank_forest(kinked_forest(4), comms[r])
        fs = kinked_set(forest)
        sched = GhostSchedule(fs, KINKED_OPS; atlas=KINKED)
        # Rank r's share of the serial state, block by block.
        A = statearray(statevector(fs), fs; appendix=true)
        U = statearray(u, kinked_set(kinked_forest(4)); appendix=true)
        L = TreeAMR.appendix_layout(fs)
        N = forest.N
        owned = blockrange(forest)
        w = statevector(fs)
        W = statearray(w, fs; appendix=true)
        for (b, j) in enumerate(owned),
            I in CartesianIndices(ntuple(d -> 1:(N + L.dims[b][d]), 2))
            W[Tuple(I)..., 1, b] = U[Tuple(I)..., 1, j]
        end
        scatter!(fs, w)
        regrid!(forest, fs => sched; flags=fill(Refine, nblocks(fs)))
        x = statevector(fs)
        gather!(x, fs)
        X = statearray(x, fs; appendix=true)
        L = TreeAMR.appendix_layout(fs)
        same = all(enumerate(blockrange(forest))) do (b, j)
            all(I -> X[Tuple(I)..., 1, b] == reference[Tuple(I)..., 1, j],
                CartesianIndices(ntuple(d -> 1:(N + L.dims[b][d]), 2)))
        end
        # And the schedule builds again over the new partition.
        GhostSchedule(fs, KINKED_OPS; atlas=KINKED)
        return same, nleaves(forest)
    end
    @test all(first, results)
    @test all(r -> r[2] == nleaves(serial), results)
end
