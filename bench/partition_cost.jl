# What each candidate ownership rule would cost the ghost fill, counted
# from the serial schedule: no field data, nothing timed.
#
#     julia --project=test bench/partition_cost.jl
#
# The sender computes, so a prolongation is evaluated on the rank that
# owns its coarse source block, and today's contiguous split of the curve
# puts each level's sources — one stretch of the curve on a nested mesh —
# on one or two ranks. A partition balanced per class of leaves spreads
# them. This script prices the candidate classes before any of them is
# built (M13 step 4), so that the rule the option offers is chosen from
# numbers. The rule chosen, `level-face-rot`, was built as `partition =
# :levels`, timed, and parked: it multiplied the phase-1 halo, which
# cost more than the balance gained ("Balancing prolongation over ranks"
# in CODE.md; the branch `eschnett-claude/partition-levels`). The script
# needs no partition code, and stays as the record of the counts.
#
# Every prolongation transfer of the serial schedule is charged to
# `owner(source)`, at its target level's stage, as target points ×
# nonzero taps: the product over dimensions of the nonzero weights of
# each 1D column, summed over the target box, which is what the kernel
# costs once it skips zero-weight taps (a vertex-centred fine point on a
# coarse node is one tap, not `p`). The full tensor, target points ×
# `prod(orders)`, is given for reference. Two views of the fill:
#
#   seq  the stages in sequence, the critical path today: the sum over
#        stages of the most loaded rank's cost in that stage;
#   grp  the stages run as one group, once independent levels may: the
#        most loaded rank's total over all stages.
#
# Both are printed against the ideal, the total over P. A rule must also
# keep equal block counts per rank (every block costs the same in the
# right-hand side under one global step), so the table gives their
# min–max, and the halo it adds: phase-1 transfers (copies and
# restrictions) whose source and target lie on different ranks, in
# target points and in ordered (sender, receiver) pairs, and the same
# points for the prolongations. `runs` is the most curve runs one rank
# owns. A rule is a function of the leaves, their classes and P only —
# never of cumulative counts across classes, or one regrid that changes
# one class would migrate the others.
#
#     TREEAMR_BENCH_N      cells per block edge (default 16)
#     TREEAMR_BENCH_D      dimension (default 3)
#     TREEAMR_BENCH_ROOTS  roots per edge (default 4)
#     TREEAMR_BENCH_EDGES  edges of the nested refined cubes `[0,a)^D`,
#                          coarsest first (default "16,8,4": the reported
#                          mesh, shells two blocks thick; "8,4,2" makes
#                          them one block thick)
#     TREEAMR_BENCH_P      rank counts (default "2,4,8")
#
# The domain is the reflecting octant `[0, 8·ROOTS]^D`, reflecting at the
# low face of every dimension, vertex-centred, `G = 3`, order 6 for both
# operators.
using TreeAMR
using TreeAMR: equalsplit_part
using Printf

const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const D = parse(Int, get(ENV, "TREEAMR_BENCH_D", "3"))
const ROOTS = parse(Int, get(ENV, "TREEAMR_BENCH_ROOTS", "4"))
const EDGES = parse.(Int, split(get(ENV, "TREEAMR_BENCH_EDGES", "16,8,4"), ","))
const PS = parse.(Int, split(get(ENV, "TREEAMR_BENCH_P", "2,4,8"), ","))
const G = 3
const ORDER = 6
const OPS = Operators(prolongation=ORDER, restriction=ORDER)

# The nested mesh: refining, level by level, every leaf inside the cube
# `[0, EDGES[ℓ])^D`. Balanced by construction when the cubes nest with a
# block of margin; checked.
function build_forest()
    forest = Forest(ntuple(_ -> ROOTS, D); N=N,
                    reflecting=ntuple(_ -> (true, false), D),
                    extents=ntuple(_ -> (0.0, 8.0 * ROOTS), D))
    for (ℓ, a) in enumerate(EDGES)
        targets = filter(forest.leaves) do k
            level(k) == ℓ - 1 &&
                all(d -> block_extent(forest, k)[d][2] <= a, 1:D)
        end
        refine!(forest, targets)
    end
    isbalanced(forest) || error("the nested cubes $EDGES do not give a balanced mesh")
    return forest
end

# The cost of one transfer of group `g`: target points, target points ×
# nonzero taps, and target points × the full tensor.
function transfercost(g)
    npts = prod(s -> size(s.weights, 2), g.stencils)
    taps = prod(s -> count(!iszero, s.weights), g.stencils)
    full = prod(s -> length(s.weights), g.stencils)
    return npts, taps, full
end

# A transfer as the script needs it: its stage (0 for phase 1), target,
# source and costs.
struct Transfer
    stage::Int
    target::Int
    source::Int
    npts::Int
    taps::Int
    full::Int
end

function transfers(schedule)
    out = Transfer[]
    for g in schedule.phase1, (t, s) in zip(g.targetblocks, g.sourceblocks)
        push!(out, Transfer(0, t, s, transfercost(g)...))
    end
    for (k, groups) in enumerate(schedule.phase2), g in groups
        g.kind === :prolong || error("phase 2 holds a $(g.kind) group")
        for (t, s) in zip(g.targetblocks, g.sourceblocks)
            push!(out, Transfer(k, t, s, transfercost(g)...))
        end
    end
    return out
end

# --- the candidate owner functions ---------------------------------------
#
# Each returns the 0-based rank of every leaf, in curve order. A class
# split counts each class, then walks the curve with a running index per
# class: the class's leaf `i` goes to piece `equalsplit_part(n_c, P, i)`
# (or `i` itself, round robin), and piece `k` to rank
# `(k − 1 + shift(c)) mod P`. The rules:
#
#   contiguous        today's `blockrange`, one class
#   level             class = level
#   level-rotated     the same, shifted by the level
#   level-source      class = (level, is a prolongation source)
#   level-source-rot  the same, shifted by the class index
#   source-rot        class = is a source, whatever the level
#   level-face-rot    class = (level, not a source / a source across an
#                     edge or corner only / across a face), shifted by the
#                     class index: a face source costs about a hundred
#                     times a corner source
#   level-source-rr   (level, source) dealt round robin, a bound on what
#                     equal counts per class can do, not a candidate: it
#                     gives up contiguity entirely

function classsplit(class::Vector{Int}, P::Int; shift=c -> 0, roundrobin=false)
    counts = Dict{Int,Int}()
    for c in class
        counts[c] = get(counts, c, 0) + 1
    end
    seen = Dict{Int,Int}()
    owner = similar(class)
    for (i, c) in enumerate(class)
        j = seen[c] = get(seen, c, 0) + 1
        piece = roundrobin ? j : equalsplit_part(counts[c], P, j)
        owner[i] = mod(piece - 1 + shift(c), P)
    end
    return owner
end

function rules(levels::Vector{Int}, issource::Vector{Bool}, isface::Vector{Bool})
    ls = 2 .* levels .+ issource                   # class (level, source), indexed
    lf = 3 .* levels .+ issource .+ isface         # class (level, none/edge/face)
    return [
        "contiguous" => P -> [equalsplit_part(length(levels), P, i) - 1
                              for i in eachindex(levels)],
        "level" => P -> classsplit(levels, P),
        "level-rotated" => P -> classsplit(levels, P; shift=identity),
        "level-source" => P -> classsplit(ls, P),
        "level-source-rot" => P -> classsplit(ls, P; shift=identity),
        "source-rot" => P -> classsplit(Int.(issource), P; shift=identity),
        "level-face-rot" => P -> classsplit(lf, P; shift=identity),
        "level-source-rr" => P -> classsplit(ls, P; shift=identity, roundrobin=true),
    ]
end

# --- the accounting --------------------------------------------------------

struct Row
    seq::Int                         # Σ over stages of the max over ranks
    grp::Int                         # max over ranks of the total
    grpfull::Int                     # the same in full-tensor taps
    stagemax::Vector{Int}            # max over ranks, per stage
    srcseq::Int                      # source blocks, as `seq`
    srcgrp::Int                      # source blocks, as `grp`
    blocks::Tuple{Int,Int}           # min and max blocks per rank
    runs::Int                        # the most curve runs of one rank
    halo1::Int                       # phase-1 target points across ranks
    pairs1::Int                      # phase-1 (sender, receiver) pairs
    halo2::Int                       # prolonged target points across ranks
end

function account(owner::Vector{Int}, P::Int, xs::Vector{Transfer}, nstages::Int)
    cost = zeros(Int, P, nstages)
    costfull = zeros(Int, P, nstages)
    sources = [Set{Int}() for _ in 1:P, _ in 1:nstages]
    halo1 = halo2 = 0
    pairs1 = Set{Tuple{Int,Int}}()
    for x in xs
        r, q = owner[x.source], owner[x.target]
        if x.stage == 0
            if r != q
                halo1 += x.npts
                push!(pairs1, (r, q))
            end
        else
            cost[r+1, x.stage] += x.taps
            costfull[r+1, x.stage] += x.full
            push!(sources[r+1, x.stage], x.source)
            r != q && (halo2 += x.npts)
        end
    end
    nsrc = length.(sources)
    stagemax = vec(maximum(cost; dims=1))
    blocks = [count(==(r), owner) for r in 0:(P - 1)]
    runs = zeros(Int, P)
    for i in eachindex(owner)
        (i == 1 || owner[i] != owner[i-1]) && (runs[owner[i]+1] += 1)
    end
    return Row(sum(stagemax), maximum(sum(cost; dims=2)),
               maximum(sum(costfull; dims=2)), stagemax,
               sum(maximum(nsrc; dims=1)), maximum(sum(nsrc; dims=2)),
               extrema(blocks), maximum(runs), halo1, length(pairs1), halo2)
end

function main()
    forest = build_forest()
    schedule = GhostSchedule(forest, OPS; G=G, centering=vertexcentered(D))
    xs = transfers(schedule)
    nstages = length(schedule.phase2)
    levels = level.(forest.leaves)
    issource = falses(nleaves(forest))
    for x in xs
        x.stage > 0 && (issource[x.source] = true)
    end
    issource = Vector{Bool}(issource)
    # A face source feeds a finer neighbour across a face, a whole ghost
    # slab; an edge or corner source feeds a sliver `G` or `G²` thinner.
    faces = [ntuple(e -> e == d ? s : 0, D) for d in 1:D for s in (-1, 1)]
    isface = [any(δ -> any(n -> level(n) == level(k) + 1,
                           neighbor_keys(forest, k, δ)), faces)
              for k in forest.leaves]

    stagetotal = zeros(Int, nstages)
    stagefull = zeros(Int, nstages)
    for x in xs
        x.stage > 0 || continue
        stagetotal[x.stage] += x.taps
        stagefull[x.stage] += x.full
    end
    total, totalfull = sum(stagetotal), sum(stagefull)

    @printf("# D = %d, N = %d, G = %d, order %d, vertex-centred, reflecting octant [0,%d]^%d\n",
            D, N, G, ORDER, 8ROOTS, D)
    @printf("# %d^%d roots, nested cubes %s: %d leaves\n", ROOTS, D, join(EDGES, ","),
            nleaves(forest))
    for ℓ in 0:maximum(levels)
        n = count(==(ℓ), levels)
        s = count(i -> levels[i] == ℓ && issource[i], eachindex(levels))
        f = count(i -> levels[i] == ℓ && isface[i], eachindex(levels))
        @printf("#   L%d: %3d leaves, %3d prolongation sources, %3d of them across a face\n",
                ℓ, n, s, f)
    end
    @printf("# phase 1: %.1fk target points; prolongation: %.1fk\n",
            sum(x -> x.stage == 0 ? x.npts : 0, xs) / 1e3,
            sum(x -> x.stage > 0 ? x.npts : 0, xs) / 1e3)
    @printf("# prolongation, Mtaps (nonzero / full tensor): total %.3f / %.3f\n",
            total / 1e6, totalfull / 1e6)
    for k in 1:nstages
        # How unequal the sources of one stage are: a source at the
        # corner of a finer cube feeds more targets than one on its face.
        persource = Dict{Int,Int}()
        for x in xs
            x.stage == k && (persource[x.source] = get(persource, x.source, 0) + x.taps)
        end
        c = collect(values(persource))
        @printf("#   stage %d (target L%d): %.3f / %.3f; per source %.3f-%.3f, mean %.3f\n",
                k, schedule.levels[k], stagetotal[k] / 1e6, stagefull[k] / 1e6,
                minimum(c) / 1e6, maximum(c) / 1e6, sum(c) / length(c) / 1e6)
    end
    println("# seq, grp: max-rank cost / ideal (total/P) in nonzero taps; grpF the same")
    println("# in full-tensor taps; Lℓ: that stage's max-rank cost / (stage total/P);")
    println("# src: source blocks, seq/grp; halo1 and halo2 in target points, k = 1000")

    stagehdr = join((@sprintf("%5s", "L$(schedule.levels[k])") for k in 1:nstages), " ")
    for P in PS
        println()
        @printf("P = %d: ideal %.3f Mtaps a rank (%.3f full tensor)\n", P,
                total / P / 1e6, totalfull / P / 1e6)
        @printf("%-17s %5s %5s %5s %s %7s %6s %4s %8s %6s %5s\n", "rule", "seq", "grp",
                "grpF", stagehdr, "src", "blocks", "runs", "halo1", "pairs", "halo2")
        for (name, rule) in rules(levels, issource, isface)
            owner = rule(P)
            row = account(owner, P, xs, nstages)
            stages = join((@sprintf("%5.2f", row.stagemax[k] * P / stagetotal[k])
                           for k in 1:nstages), " ")
            @printf("%-17s %5.2f %5.2f %5.2f %s %7s %6s %4d %7.1fk %6d %4.0fk\n", name,
                    row.seq * P / total, row.grp * P / total, row.grpfull * P / totalfull,
                    stages, "$(row.srcseq)/$(row.srcgrp)",
                    "$(row.blocks[1])-$(row.blocks[2])", row.runs, row.halo1 / 1e3,
                    row.pairs1, row.halo2 / 1e3)
        end
    end
    return nothing
end

main()
