# The ghost fill on a nested mesh, piece by piece and rank by rank.
#
#     julia -t T --project=test bench/nested.jl                   # serial
#     mpiexec -n P julia -t T --project=test bench/nested.jl mpi  # P ranks
#
# TreeGeneralizedHarmonic reported (eschnett/TreeGeneralizedHarmonic.jl#11)
# that on its nested mesh a distributed `fill_ghosts!` does not
# strong-scale: 9.5 ms on one H200 and 15–16 ms on four with HPC-X, 42 ms
# with MPICH_jll, and on the CPU the prolongations are half of a
# right-hand side. Two reasons are TreeAMR's: the prolongation stages run
# one after another, each waiting for its messages, and one rank — the
# one owning that level's coarse sources, a single stretch of the Morton
# curve — packs all of a level; and every prolonged point summed the
# full `p^D` tensor product, although in vertex-centered refinement every
# other fine point along a dimension sits on a coarse node. This is that
# mesh, so that both can be measured where they were reported.
#
# The mesh is the downstream's: the reflecting octant `[0,32]³` (low
# faces reflecting, high faces outer), `4³` roots of edge 8, with the
# cubes `[0,16)³`, `[0,8)³` and `[0,4)³` refined to levels 1, 2 and 3 —
# 232 leaves, three coarse-fine shells each two blocks thick. `N³`
# points a block, `NVARS` variables (20, the downstream's), `G = 3`,
# prolongation and restriction of order 6, vertex-centered, Float64. The
# outer faces get a trivial boundary hook (zero, through `CellBoundary`
# and `AllVariables`, the downstream's form); the reflecting ones need
# none.
#
# Timed per rank in synchronized windows, as `bench/mpi.jl` does: every
# repetition starts at a barrier and each rank times its own call; a
# line reports the slowest rank's time, its minimum and median over the
# repetitions after one untimed call, and each rank's own minimum. The
# quantities:
#
#     fill            fill_ghosts! with the hook
#     boundary        the hook alone
#     stage:P1        phase 1 alone (copies and restrictions; `run_stage!`)
#     stage:L<l>      the prolongation stage onto level l alone
#     phase2          every prolongation stage, one after another, as a
#                     fill without stage groups runs them
#     phase2:grouped  the same through this rank's stage groups but phase 1's,
#                     where the schedule has `stagegroups`, as the fill
#                     runs them: each through `run_stages!` if this
#                     TreeAMR has it, else its stages one after another,
#                     said in the name
#     group:<a>-<b>   each stage group alone, where every rank groups alike
#                     (otherwise a rank would wait in one group's window
#                     for messages its peer sends in the next one's)
#     locals[:<s>]    the local groups (`run_phase!` of `locals`), of every
#                     stage or of one
#     packs[:<s>]     the packs into the send buffers (`pack_stage!`)
#     unpacks[:<s>]   the unpacks from the receive buffers (`unpack_stage!`)
#
# A rank holds a stage only where it has transfers of that tag, so the
# windows run over every rank's tags, a rank without the stage idle in
# them. The pieces are internal calls, so a stage's packs are timed
# without its messages, and the per-rank columns of `packs:L<l>` show which rank does
# a level's packing. Over one rank there are no packs or unpacks.
#
# Counted rather than timed, on rank 0, from the serial schedule with the
# ranks' equal split of the curve as the owner function (`P` from
# `TREEAMR_BENCH_FOOTPRINT_P`, and the run's own rank count), per
# prolongation stage and per (sender, receiver) pair: the fine target
# points the stage's messages carry, which is what is sent today, and the
# *footprint*, the distinct coarse points — on the source block, its
# ghosts included — that those transfers' stencils read with a nonzero
# weight, which is what receiver-side prolongation would send instead.
# `CODE.md` estimated the two as about equal. Under MPI the predicted
# target points are checked against the schedule's own send counts. Also
# per stage: the taps per prolonged point, the full tensor product
# against the nonzero-weight ones a single-tap column takes.
#
# Output (rank 0): a header line, `#` lines with the counts, and one
# tab-separated line per timed quantity,
#
#     T  <label>  <quantity>  <min s>  <median s>  <rank 0 min s>,<rank 1 min s>,…
#
# Everything but the times is deterministic. The file reads only
# internals that `main` had before stage groups (`run_stage!`,
# `run_phase!`, `pack_stage!`, `unpack_stage!`, `stagebuffers`), so the
# same file runs against a baseline checkout's project.
#
#     TREEAMR_BENCH_N            points per block edge (default 16)
#     TREEAMR_BENCH_NVARS        variables (default 20)
#     TREEAMR_BENCH_REPS         timed repetitions (default 10)
#     TREEAMR_BENCH_BACKEND      cpu (default) or cuda (an environment with
#                                CUDA and MPI; one GPU per rank, local
#                                rank mod GPUs)
#     TREEAMR_BENCH_DEVICEAWARE  1 to hand MPI the device buffers; default 0,
#                                staging through host mirrors
#     TREEAMR_BENCH_LABEL        the label column (default <ranks>x<threads>)
#     TREEAMR_BENCH_FOOTPRINT_P  rank counts to count messages for (default
#                                "2 4 8"; empty for none)
#     TREEAMR_BENCH_PIECES       0 to time the fill, the boundary and the
#                                stages only (default 1)
#
# The `# ownership` line gives per rank the blocks and the blocks per
# level, from the forest's own `leafowner`.

const USE_MPI = "mpi" in ARGS || get(ENV, "TREEAMR_BENCH_MPI", "0") == "1"
if USE_MPI
    using MPI
    MPI.Init()
end
using TreeAMR
using KernelAbstractions: get_backend, synchronize, CPU
using Printf: @sprintf

const BNAME = lowercase(get(ENV, "TREEAMR_BENCH_BACKEND", "cpu"))
if BNAME == "cuda"
    using CUDA
elseif BNAME != "cpu"
    error("TREEAMR_BENCH_BACKEND must be cpu or cuda; got \"$BNAME\"")
end

const RANK = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const NRANKS = USE_MPI ? MPI.Comm_size(MPI.COMM_WORLD) : 1
const DEVICEAWARE = get(ENV, "TREEAMR_BENCH_DEVICEAWARE", "0") == "1"
const COMM = USE_MPI ? communicator(MPI.COMM_WORLD; deviceaware=DEVICEAWARE) : nothing
const LOCALRANK = USE_MPI ?
                  MPI.Comm_rank(MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED,
                                                    RANK)) : 0
if BNAME == "cuda"
    CUDA.functional() || error("CUDA is not functional here")
    CUDA.device!(LOCALRANK % length(CUDA.devices()))
end
const BACKEND = BNAME == "cuda" ? CUDABackend() : CPU()
const T = Float64

const D = 3
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const NVARS = parse(Int, get(ENV, "TREEAMR_BENCH_NVARS", "20"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "10"))
const LABEL = get(ENV, "TREEAMR_BENCH_LABEL", "$(NRANKS)x$(Threads.nthreads())")
const FOOTPRINT_P = [parse(Int, s)
                     for s in split(get(ENV, "TREEAMR_BENCH_FOOTPRINT_P", "2 4 8"))]
const PIECES = get(ENV, "TREEAMR_BENCH_PIECES", "1") != "0"
const G = 3
const OPS = Operators(prolongation=6, restriction=6)
const ROOTS = 4
const EXTENT = 32

# The variables' parities: the even ones scalars, each odd one the
# component of a vector along one dimension, so that every mirrored
# transfer has odd variables to flip, as the downstream's tensors do.
parities() = [isodd(v) ? EvenParity :
              ntuple(d -> d == mod1(v ÷ 2, D) ? OddParity : EvenParity, D)
              for v in 1:NVARS]

function build_forest(comm)
    forest = Forest(ntuple(_ -> ROOTS, D); N=N, reflecting=ntuple(_ -> (true, false), D),
                    extents=ntuple(_ -> (zero(T), T(EXTENT)), D), comm=comm)
    # Level ℓ covers [0, 32/2^ℓ)³: refine every leaf of level ℓ-1 inside it.
    for l in 1:3
        edge = T(EXTENT) / 2^l
        targets = filter(forest.leaves) do k
            level(k) == l - 1 || return false
            ext = block_extent(forest, k)
            return all(d -> ext[d][2] <= edge, 1:D)
        end
        refine!(forest, targets)
    end
    balance!(forest)
    return forest
end

const BOUNDARY = CellBoundary(AllVariables((x, δ) -> ntuple(_ -> zero(T), Val(NVARS))))

barrier() = USE_MPI ? MPI.Barrier(MPI.COMM_WORLD) : nothing
gather(x) = USE_MPI ? TreeAMR.allgather(COMM, x) : [x]
gatherv(v) = USE_MPI ? TreeAMR.allgatherv(COMM, v) : collect(v)
say(s) = (RANK == 0 && println(s); nothing)

median(ts) = (s = sort(ts); n = length(s); (s[(n + 1) ÷ 2] + s[n ÷ 2 + 1]) / 2)

# One synchronized window: every rank's time for `f`, from a barrier.
function window(f)
    barrier()
    t = @elapsed f()
    return gather(t)
end

# `reps` windows after one untimed call: the slowest rank's minimum and
# median, and each rank's own minimum.
function measure(f, reps=REPS)
    f()
    GC.gc()
    ws = [window(f) for _ in 1:reps]
    slowest = maximum.(ws)
    perrank = [minimum(w[r] for w in ws) for r in 1:NRANKS]
    return minimum(slowest), median(slowest), perrank
end

function report(quantity, (tmin, tmed, perrank))
    say(@sprintf("T\t%s\t%s\t%.6e\t%.6e\t%s", LABEL, quantity, tmin, tmed,
                 join([@sprintf("%.6e", t) for t in perrank], ",")))
end

stagename(st) = st.tag == TreeAMR.PHASE1_TAG ? "P1" : "L$(st.tag - 2)"

# --- Counts: target points against footprints ------------------------------

# Per dimension, the source rows a stencil reads with a nonzero weight,
# over all its target columns. The target box is the product of the
# columns and each point reads the product of its columns' rows, so the
# union over the box is the product of these unions.
function readrows(s)
    w = Array(s.weights)
    start = Array(s.srcstart)
    rows = Set{Int}()
    for i in axes(w, 2), m in axes(w, 1)
        iszero(w[m, i]) || push!(rows, Int(start[i]) + m - 1)
    end
    return sort!(collect(rows))
end

# Taps per target point of a group, summed over its box: the full tensor
# product, and with a single-nonzero column taking one tap.
function taps(g)
    full = prod(size(s.weights, 1) * length(s.srcstart) for s in g.stencils)
    single = prod(begin
                      w = Array(s.weights)
                      sum(i -> count(!iszero, w[:, i]) == 1 ? 1 : size(w, 1), axes(w, 2))
                  end for s in g.stencils)
    return full, single
end

boxpoints(g) = prod(length(s.srcstart) for s in g.stencils)

# Per prolongation level of the serial schedule `sched` and per (sender,
# receiver) pair under the equal split over `P` ranks: the transfers, the
# fine target points, and the set of (source block, coarse point) read.
function message_counts(sched, P)
    n = nleaves(sched.forest)
    owner(i) = TreeAMR.equalsplit_part(n, P, Int(i)) - 1
    out = Dict{Int,Dict{Tuple{Int,Int},Tuple{Int,Int,Set{Tuple{Int32,CartesianIndex{D}}}}}}()
    for (l, groups) in zip(sched.levels, sched.phase2)
        bypair = get!(out, l, Dict{Tuple{Int,Int},
                                   Tuple{Int,Int,Set{Tuple{Int32,CartesianIndex{D}}}}}())
        for g in groups
            rows = map(readrows, g.stencils)
            box = CartesianIndices(ntuple(d -> length(rows[d]), D))
            npts = boxpoints(g)
            tb, sb = Array(g.targetblocks), Array(g.sourceblocks)
            for j in eachindex(tb)
                pair = (owner(sb[j]), owner(tb[j]))
                pair[1] == pair[2] && P > 1 && continue
                ntr, npt, fp = get!(() -> (0, 0, Set{Tuple{Int32,CartesianIndex{D}}}()),
                                    bypair, pair)
                for I in box
                    push!(fp, (sb[j], CartesianIndex(ntuple(d -> rows[d][I[d]], D))))
                end
                bypair[pair] = (ntr + 1, npt + npts, fp)
            end
        end
    end
    return out
end

function report_counts(sched, P)
    counts = message_counts(sched, P)
    what = P == 1 ? "every prolongation, as if each crossed ranks" :
           "messages under the equal split over $P ranks"
    for l in sort!(collect(keys(counts)))
        bypair = counts[l]
        tot = [0, 0, 0]
        for pair in sort!(collect(keys(bypair)))
            ntr, npt, fp = bypair[pair]
            tot .+= (ntr, npt, length(fp))
            P == 1 && continue
            say(@sprintf("# footprint P=%d L%d %d->%d: %d transfers, %d target points, \
                          %d coarse points read (%.3f of the targets)", P, l, pair[1],
                         pair[2], ntr, npt, length(fp), length(fp) / npt))
        end
        say(@sprintf("# footprint P=%d L%d total (%s): %d transfers, %d target points, \
                      %d coarse points read (%.3f of the targets)", P, l, what, tot[1],
                     tot[2], tot[3],
                     tot[2] == 0 ? NaN : tot[3] / tot[2]))
    end
    return counts
end

function report_taps(sched)
    for (l, groups) in zip(sched.levels, sched.phase2)
        npts = sum(g -> boxpoints(g) * TreeAMR.ntransfers(g), groups; init=0)
        full = sum(g -> taps(g)[1] * TreeAMR.ntransfers(g), groups; init=0)
        single = sum(g -> taps(g)[2] * TreeAMR.ntransfers(g), groups; init=0)
        say(@sprintf("# taps L%d: %d groups, %d transfers, %d target points; taps per \
                      point %.2f full product, %.2f with single-tap columns",
                     l, length(groups), sum(TreeAMR.ntransfers, groups; init=0), npts,
                     full / npts, single / npts))
    end
end

# The schedule's own send counts per stage and peer, gathered from every
# rank, against rank 0's prediction of `message_counts` for this run.
function check_counts(schedule, predicted)
    mine = Int[]
    for st in schedule.stages
        st.tag == TreeAMR.PHASE1_TAG && continue
        r = st.remote
        r === nothing && continue
        for (peer, c) in zip(r.sendpeers, r.sendcounts)
            append!(mine, (st.tag - 2, RANK, peer, c))
        end
    end
    g = reshape(gatherv(mine), 4, :)
    got = sort!([Tuple(g[:, j]) for j in axes(g, 2)])
    RANK == 0 || return nothing
    want = sort!([(l, pair[1], pair[2], v[2])
                  for (l, bypair) in predicted for (pair, v) in bypair])
    say(got == want ?
        "# footprint check: the schedule's send counts are the predicted target points" :
        "# footprint check: MISMATCH between the schedule's send counts $got and the " *
        "predicted ones $want")
    return nothing
end

# --- The run ------------------------------------------------------------------

# A stage group, a range of `schedule.stages`: through `run_stages!`
# where this TreeAMR has it — with the group's `grouplocals` where the
# schedule holds them, else in the form that gathered them itself — else
# its stages one after another. Either takes the call's sends so far and
# returns them, as `run_stage!` does.
function group_runner(fs, schedule, forest, g)
    if isdefined(TreeAMR, :run_stages!)
        f = TreeAMR.run_stages!
        if hasfield(typeof(schedule), :grouplocals)
            locals = schedule.grouplocals[findfirst(==(g), schedule.stagegroups)]
            largs(sends) = (fs, schedule.stages, g, locals, forest, BACKEND, sends)
            applicable(f, largs(nothing)...) && return "", sends -> f(largs(sends)...)
        end
        args(sends) = (fs.work, fs.work, fs.nvars, fs.factors, schedule.stages, g, forest,
                       BACKEND, sends, fs.work, fs.rotvars)
        applicable(f, args(nothing)...) && return "", sends -> f(args(sends)...)
    end
    return "(sequential)", sends -> begin
        for st in schedule.stages[g]
            sends = TreeAMR.run_stage!(fs, st, forest, BACKEND, sends)
        end
        sends
    end
end

function main()
    say("ranks=$NRANKS threads=$(Threads.nthreads()) N=$N nvars=$NVARS G=$G order=6 " *
        "vertex-centered backend=$BNAME T=$T deviceaware=$DEVICEAWARE reps=$REPS " *
        "label=$LABEL Julia $VERSION TreeAMR $(pkgversion(TreeAMR))")
    forest = build_forest(COMM)
    fs = FieldSet{T}(forest, NVARS; G=G, centering=vertexcentered(D), parity=parities(),
                     backend=BACKEND)
    fill_by_coordinates!((x, v) -> sin(T(0.1) * x[1] + v) * cos(T(0.07) * x[2] - x[3] / 50),
                         fs)
    schedule = GhostSchedule(fs, OPS)
    comm = forest.comm
    stages = schedule.stages
    nb = TreeAMR.allgather(comm, nblocks(fs))
    say("# mesh: $(nleaves(forest)) leaves, levels 0:$(maxlevel(forest)), per level " *
        join(["L$l $(count(k -> level(k) == l, forest.leaves))"
              for l in 0:maxlevel(forest)], ", ") *
        "; blocks per rank $(join(nb, ", "))")
    # Who owns what: per rank the blocks, and the blocks per level.
    own = [TreeAMR.leafowner(forest, i) for i in 1:nleaves(forest)]
    lv = level.(forest.leaves)
    say("# ownership per rank (blocks, blocks per level L0/L1/…): " *
        join(["r$q $(count(==(q), own)) blocks, " *
              join([count(i -> own[i] == q && lv[i] == l, eachindex(own))
                    for l in 0:maxlevel(forest)], "/") for q in 0:(NRANKS - 1)], "; "))
    # A rank holds a stage only where it has transfers of that tag, so
    # everything collective below runs over the tags of every rank, a
    # rank without the stage taking part with nothing to do.
    tags = sort!(unique(gatherv([st.tag for st in stages])))
    stageof(tag) = (i = findfirst(st -> st.tag == tag, stages);
                    i === nothing ? nothing : stages[i])
    tagname(tag) = tag == TreeAMR.PHASE1_TAG ? "P1" : "L$(tag - 2)"
    say("# stages: " * join(tagname.(tags), " "))
    # Stage groups, by tag, rank by rank. Their windows are timed one by
    # one only where every rank groups alike: otherwise a rank whose
    # group spans more levels would wait in its window for messages a
    # peer sends only in the next.
    grouped = hasfield(typeof(schedule), :stagegroups)
    samegroups = false
    if grouped
        mygroups = [[st.tag for st in stages[g]] for g in schedule.stagegroups]
        gs = reshape(gatherv(reduce(vcat, [[RANK, first(g), last(g)] for g in mygroups];
                                    init=Int[])), 3, :)
        bygroup = [[(gs[2, j], gs[3, j]) for j in axes(gs, 2) if gs[1, j] == r]
                   for r in 0:(NRANKS - 1)]
        samegroups = all(==(bygroup[1]), bygroup)
        say("# stagegroups per rank: " *
            join(["r$(r - 1) " * join(["$(tagname(a))-$(tagname(b))" for (a, b) in
                                       bygroup[r]], ",") for r in 1:NRANKS], "; ") *
            (samegroups ? "" : " (they differ between ranks)"))
    else
        say("# stagegroups: none (this TreeAMR has no stage groups)")
    end
    # Per stage and rank: messages sent and received, target points sent.
    for tag in tags
        st = stageof(tag)
        r = st === nothing ? nothing : st.remote
        row = r === nothing ? (0, 0, 0, 0) :
              (length(r.sendpeers), length(r.recvpeers), sum(r.sendcounts; init=0),
               sum(r.recvcounts; init=0))
        loc = st === nothing ? 0 : sum(g -> TreeAMR.ntransfers(g), st.locals; init=0)
        rows = reshape(gatherv(Int[row..., loc]), 5, :)
        say("# stage $(tagname(tag)) per rank (sends, receives, points sent, points " *
            "received, local transfers): " *
            join(["r$(i - 1) $(join(rows[:, i], "/"))" for i in axes(rows, 2)], "; "))
    end

    # The counts, on rank 0 from a serial copy of the mesh.
    if RANK == 0
        sforest = build_forest(nothing)
        sched = GhostSchedule(sforest, OPS; G=G, centering=vertexcentered(D), T=T)
        report_taps(sched)
        report_counts(sched, 1)
        for P in unique([FOOTPRINT_P; NRANKS > 1 ? [NRANKS] : Int[]])
            P > 1 && report_counts(sched, P)
        end
        predicted = NRANKS > 1 ? message_counts(sched, NRANKS) : nothing
    else
        predicted = nothing
    end
    NRANKS > 1 && check_counts(schedule, predicted)

    sync() = synchronize(BACKEND)
    waitsends(sends) = (sends === nothing || TreeAMR.waitall(comm, sends); nothing)
    report("fill", measure(() -> (fill_ghosts!(fs, schedule; boundary=BOUNDARY); sync())))
    if applicable(TreeAMR.apply_boundary!, fs, BOUNDARY, schedule, BACKEND)
        report("boundary",
               measure(() -> (TreeAMR.apply_boundary!(fs, BOUNDARY, schedule, BACKEND);
                              sync())))
    end
    for tag in tags
        st = stageof(tag)
        report("stage:$(tagname(tag))",
               measure(() -> (st === nothing ||
                              waitsends(TreeAMR.run_stage!(fs, st, forest, BACKEND,
                                                           nothing)); sync())))
    end
    # Phase 2 whole, the stages one after another as `main` runs them, and
    # through this rank's groups.
    prolongs = [st for st in stages if st.tag != TreeAMR.PHASE1_TAG]
    report("phase2", measure(() -> begin
        sends = nothing
        for st in prolongs
            sends = TreeAMR.run_stage!(fs, st, forest, BACKEND, sends)
        end
        waitsends(sends)
        sync()
    end))
    if grouped
        # Phase 2 through this rank's groups, as the fill runs it: every
        # group but phase 1's, the sends waited on at the end.
        runners = [(g, group_runner(fs, schedule, forest, g)...)
                   for g in schedule.stagegroups]
        how = isempty(runners) ? "" : runners[1][2]
        later = [f for (g, _, f) in runners
                 if !any(i -> stages[i].tag == TreeAMR.PHASE1_TAG, g)]
        report("phase2:grouped$how", measure(() -> begin
            sends = nothing
            for f in later
                sends = f(sends)
            end
            waitsends(sends)
            sync()
        end))
        if samegroups
            for (g, _, f) in runners
                name = "group:$(stagename(stages[first(g)]))-" *
                       "$(stagename(stages[last(g)]))$how"
                report(name, measure(() -> (waitsends(f(nothing)); sync())))
            end
        end
    end
    PIECES || return nothing

    locals!(sts) = () -> begin
        for st in sts
            TreeAMR.run_phase!(fs.work, fs.work, st.locals, fs.nvars, BACKEND;
                               factors=fs.factors)
        end
        sync()
    end
    remotes(sts) = [st.remote for st in sts if st.remote !== nothing]
    packs!(sts) = (rs = remotes(sts); () -> begin
        for r in rs
            TreeAMR.pack_stage!(fs, r, TreeAMR.stagebuffers(r, fs.nvars, BACKEND), BACKEND)
        end
        sync()
    end)
    unpacks!(sts) = (rs = remotes(sts); () -> begin
        for r in rs
            TreeAMR.unpack_stage!(fs, r, TreeAMR.stagebuffers(r, fs.nvars, BACKEND),
                                  BACKEND)
        end
        sync()
    end)
    report("locals", measure(locals!(stages)))
    NRANKS > 1 && report("packs", measure(packs!(stages)))
    NRANKS > 1 && report("unpacks", measure(unpacks!(stages)))
    for tag in tags
        sts = filter(st -> st.tag == tag, stages)
        s = tagname(tag)
        report("locals:$s", measure(locals!(sts)))
        NRANKS > 1 && report("packs:$s", measure(packs!(sts)))
        NRANKS > 1 && report("unpacks:$s", measure(unpacks!(sts)))
    end
    return nothing
end

main()
