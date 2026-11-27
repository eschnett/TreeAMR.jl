# The time to a first ghost fill in a fresh process, and what it compiles:
# the measurement of M22 step 9 (b), on the API every TreeAMR since 0.2
# has, so that a baseline checkout of half-open blocks runs the same file
# (`jobs/nested.toml`'s `compile-` runs; the study's `test/mpi_workload.jl`
# calls what a baseline does not have).
#
#     julia --project=test bench/firstcall.jl
#     julia --project=test --trace-compile=stderr --trace-compile-timing bench/firstcall.jl
#
# A line per step, the seconds since the process started and the step's
# own, every step's first call:
#
#     F  <step>  <since start s>  <step s>
#
# The steps: loading the package; then, on a 3D nested mesh with outer
# faces (four roots a side, refined three times toward a corner, `N =
# 16`) and a 2D periodic brick, for a cell- and a vertex-centered set of
# 20 variables, `G = 3`, operators of order 6, each in turn: building the
# forest, building the schedule, filling it, gathering the state, a volume-weighted norm, a
# regrid that refines one block and coarsens it back, and the fill again.
# The order is fixed, so that two versions compile the same things.

const T0 = time()
const STEPS = Tuple{String,Float64,Float64}[]
function step(f, name)
    t = time()
    r = f()
    push!(STEPS, (name, time() - T0, time() - t))
    return r
end

step(() -> (@eval using TreeAMR), "using")

function nested3()
    f = Forest((4, 4, 4); N=16, extents=((0.0, 32.0), (0.0, 32.0), (0.0, 32.0)))
    for (lvl, side) in ((0, 16), (1, 8), (2, 4))
        inside = filter(k -> level(k) == lvl &&
                             all(e -> e[2] <= side, block_extent(f, k)), f.leaves)
        refine!(f, inside)
        balance!(f)
    end
    return f
end
brick2() = Forest((4, 4); N=16, periodic=(true, true))

const OPS = Operators(prolongation=6, restriction=6)
for (mesh, make) in (("nested3", nested3), ("brick2", brick2)),
    (cname, centering) in (("cell", cellcentered), ("vertex", vertexcentered))
    tag = "$mesh.$cname"
    forest = step(make, "$tag forest")
    D = length(forest.roots)
    fs = FieldSet(forest, 20; G=3, centering=centering(D))
    fill!(fs.work, 1.0)
    hook = boundary_by_coordinates((x, v) -> 1.0)
    sched = step(() -> GhostSchedule(fs, OPS), "$tag schedule")
    step(() -> fill_ghosts!(fs, sched; boundary=hook), "$tag fill")
    u = step(() -> (u = statevector(fs); gather!(u, fs); u), "$tag gather")
    step(() -> volume_weighted_norm(fs, u), "$tag norm")
    step("$tag regrid") do
        flags = [i == 1 ? Refine : Keep for i in 1:nblocks(fs)]
        regrid!(forest, fs => sched; flags=flags, boundary=hook)
    end
    sched = GhostSchedule(fs, OPS)
    step(() -> fill_ghosts!(fs, sched; boundary=hook), "$tag refill")
end

for (name, since, own) in STEPS
    println("F\t", replace(name, ' ' => '_'), "\t", round(since; digits=3), "\t",
            round(own; digits=3))
end
