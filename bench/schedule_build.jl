# The cost of building a ghost schedule, on the two meshes M22 step 5
# measures, cell- and vertex-centered.
#
#     julia -t T --project=test bench/schedule_build.jl
#
# A schedule is built once a regrid, so its cost is the price of a
# regrid, not of an evaluation; closed blocks (M22) add stage 0, stage P
# and the per-piece merging to it, and this measures what they add
# against a baseline checkout's project running the same file. The
# meshes:
#
#     uniform  8³ periodic roots of 16³, the downstream's uniform mesh
#              (`bench/copies.jl`'s default)
#     nested   `bench/nested.jl`'s: the reflecting octant [0,32]³, 4³
#              roots, refined to levels 1–3 about the origin, 232 leaves
#
# each with 20 variables (the nested mesh's with odd vector components,
# so that its mirrored transfers and stage P have something to flip),
# `G = 3`, operators of order 6. A line per mesh and centering, the best
# and the median of `TREEAMR_BENCH_REPS` builds after one untimed one:
#
#     B  <mesh>  <centering>  <threads>  <best s>  <median s>
#
# and `#` lines with the schedule's exact group and transfer counts by
# stage, where this TreeAMR has the stage. Serial only.
#
#     TREEAMR_BENCH_N      points per block edge (default 16)
#     TREEAMR_BENCH_REPS   timed builds (default 5)
#     TREEAMR_BENCH_MESH   uniform, nested, or both (default both)

using TreeAMR
using Printf: @sprintf

const T = Float64
const D = 3
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "5"))
const MESHES = let m = get(ENV, "TREEAMR_BENCH_MESH", "both")
    m == "both" ? (:uniform, :nested) : (Symbol(m),)
end
const NVARS = 20
const G = 3
const OPS = Operators(prolongation=6, restriction=6)

parities() = [isodd(v) ? EvenParity :
              ntuple(d -> d == mod1(v ÷ 2, D) ? OddParity : EvenParity, D)
              for v in 1:NVARS]

function build_forest(mesh)
    if mesh === :uniform
        return Forest{T}(ntuple(_ -> 8, D); N=N, periodic=ntuple(_ -> true, D),
                         extents=ntuple(_ -> (zero(T), one(T)), D))
    end
    mesh === :nested || error("TREEAMR_BENCH_MESH must be uniform, nested or both")
    forest = Forest(ntuple(_ -> 4, D); N=N, reflecting=ntuple(_ -> (true, false), D),
                    extents=ntuple(_ -> (zero(T), T(32)), D))
    for l in 1:3
        edge = T(32) / 2^l
        refine!(forest, filter(forest.leaves) do k
                    level(k) == l - 1 || return false
                    ext = block_extent(forest, k)
                    return all(d -> ext[d][2] <= edge, 1:D)
                end)
    end
    balance!(forest)
    return forest
end

median(ts) = (s = sort(ts); n = length(s); (s[(n + 1) ÷ 2] + s[n ÷ 2 + 1]) / 2)

ngroups(gs) = length(gs)
ntrans(gs) = sum(TreeAMR.ntransfers, gs; init=0)
field(s, f) = hasfield(typeof(s), f) ? getfield(s, f) : nothing

function counts(s)
    parts = String[]
    for (name, f) in (("stage 0", :unify), ("stage P", :selfimage), ("phase 1", :phase1))
        gs = field(s, f)
        gs === nothing && continue
        push!(parts, "$name $(ngroups(gs)) groups/$(ntrans(gs)) transfers")
    end
    p2 = reduce(vcat, s.phase2; init=eltype(s.phase1)[])
    push!(parts, "phase 2 $(ngroups(p2)) groups/$(ntrans(p2)) transfers")
    sg = field(s, :stagegroups)
    sg === nothing || push!(parts, "$(length(s.stages)) stages in $(length(sg)) stage groups")
    return join(parts, ", ")
end

function main()
    println("threads=$(Threads.nthreads()) N=$N nvars=$NVARS G=$G order=6 reps=$REPS " *
            "Julia $VERSION TreeAMR $(pkgversion(TreeAMR))")
    for mesh in MESHES, cname in (:cell, :vertex)
        forest = build_forest(mesh)
        centering = cname === :cell ? cellcentered(D) : vertexcentered(D)
        fs = mesh === :nested ?
             FieldSet{T}(forest, NVARS; G=G, centering=centering, parity=parities()) :
             FieldSet{T}(forest, NVARS; G=G, centering=centering)
        s = GhostSchedule(fs, OPS)
        ts = [@elapsed(GhostSchedule(fs, OPS)) for _ in 1:REPS]
        println("# $mesh $cname: $(nleaves(forest)) leaves; $(counts(s))")
        println(@sprintf("B\t%s\t%s\t%d\t%.6f\t%.6f", mesh, cname, Threads.nthreads(),
                         minimum(ts), median(ts)))
    end
    return nothing
end

main()
