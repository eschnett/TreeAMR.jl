# Ghost-exchange measurement.
#
#     julia -t N --project=. bench/ghosts.jl
#
# `bench/threads.jl` measures how the whole per-evaluation path scales;
# this one isolates the ghost exchange itself, because that is where a
# downstream profile put most of a realistic solve's non-compilation
# run time and essentially all of its remaining heap allocation.
#
# Two meshes, because they exercise different halves of the kernel. On a
# uniform mesh every transfer is a same-level copy — one stencil point
# per ghost point, so the *addressing* is the whole cost. On a two-level
# mesh the prolongations dominate — `p^D` stencil points per ghost
# point, so the inner sum is. A change that helps one and not the other
# is easy to mistake for a change that helps.
#
# Also timed: building the schedule. That is the regrid-frequency half
# of the same machinery (exact rational interpolation weights), which a
# regridding run pays at every regrid and a test suite pays once per
# problem it builds.
#
# Sizes come from the environment, as in `bench/threads.jl`:
#
#     TREEAMR_BENCH_D      dimension (default 3)
#     TREEAMR_BENCH_N      cells per block edge (default 8)
#     TREEAMR_BENCH_ROOTS  roots per edge (default 4)
#     TREEAMR_BENCH_VARS   variables per point (default 10)
#     TREEAMR_BENCH_P      operator order, both operators (default 4)
#     TREEAMR_BENCH_REPS   timed repetitions (default 50)
#
# With `TREEAMR_BENCH_PROFILE=1` it also prints a flat *self*-time
# profile of each fill — which entries the samples land on, not who
# called them. That is the instrument the three costs recorded in
# CODE.md ("What the ghost fill costs") were found with, and the one
# that shows them gone: `div`/`rem` from unflattening a
# one-dimensional launch, `__inc` from iterating `CartesianIndices`
# over the stencil, and `checkbounds_indices` from re-checking indices
# the schedule constructed.

using TreeAMR
using Printf: @printf
using Profile

const D = parse(Int, get(ENV, "TREEAMR_BENCH_D", "3"))
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "8"))
const ROOTS = parse(Int, get(ENV, "TREEAMR_BENCH_ROOTS", "4"))
const NVARS = parse(Int, get(ENV, "TREEAMR_BENCH_VARS", "10"))
const P = parse(Int, get(ENV, "TREEAMR_BENCH_P", "4"))
const REPS = parse(Int, get(ENV, "TREEAMR_BENCH_REPS", "50"))
const PROFILE = get(ENV, "TREEAMR_BENCH_PROFILE", "0") != "0"
const OPS = Operators(prolongation=P, restriction=P)

"""A uniform forest, or one with the middle eighth refined once."""
function build_forest(::Val{DD}, refined::Bool) where {DD}
    forest = Forest(ntuple(_ -> ROOTS, DD); N=N,
                    periodic=ntuple(_ -> true, DD),
                    extents=ntuple(_ -> (0.0, 1.0), DD))
    if refined
        targets = filter(forest.leaves) do k
            ext = block_extent(forest, k)
            all(d -> 0.25 < (ext[d][1] + ext[d][2]) / 2 < 0.75, 1:DD)
        end
        refine!(forest, targets)
        balance!(forest)
    end
    return forest
end

# Best of `REPS`, after a warm-up call: the minimum is what a
# measurement wants, since noise only ever adds time.
function best(f, reps=REPS)
    f()
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f())
    end
    return t
end

# Flat self time: only the leaf frame of each backtrace is counted, so
# an entry's share is the work done *in* it, not under it. `Profile`'s
# own printer reports cumulative time by default, which buries a kernel's
# innermost loop under every frame that called it.
function selftime(f)
    Profile.clear()
    Profile.init(; n=10^7, delay=0.0005)
    Profile.@profile for _ in 1:REPS
        f()
    end
    data, ldict = Profile.retrieve()
    counts = Dict{String,Int}()
    total = 0
    i = firstindex(data)
    while i <= lastindex(data)
        j = i
        while j <= lastindex(data) && data[j] != 0
            j += 1
        end
        if j > i                                    # a backtrace, leaf first
            total += 1
            frames = get(ldict, data[i], nothing)
            if frames !== nothing && !isempty(frames)
                fr = frames[1]
                key = string(fr.func, " @ ", basename(string(fr.file)), ":", fr.line)
                counts[key] = get(counts, key, 0) + 1
            end
        end
        i = j + 1
    end
    return counts, total
end

function report_profile(label, f)
    counts, total = selftime(f)
    total == 0 && return nothing
    println("# $label flat self time, $total samples")
    for (k, v) in first(sort(collect(counts); by=last, rev=true), 15)
        @printf("#   %6.2f%%  %s\n", 100v / total, k)
    end
    return nothing
end

function main()
    @printf("threads=%d D=%d N=%d roots=%d nvars=%d p=%d\n",
            Threads.nthreads(), D, N, ROOTS, NVARS, P)
    for refined in (false, true)
        forest = build_forest(Val(D), refined)
        fs = FieldSet(forest, NVARS; G=P ÷ 2)
        schedule = GhostSchedule(fs, OPS)
        fill_by_coordinates!((x, v) -> sin(2π * x[1]) + v, fs)

        label = refined ? "two_level" : "uniform"
        t_fill = best(() -> fill_ghosts!(fs, schedule))
        fill_ghosts!(fs, schedule)
        bytes = @allocated fill_ghosts!(fs, schedule)
        t_schedule = best(() -> GhostSchedule(fs, OPS), max(3, REPS ÷ 10))
        sbytes = @allocated GhostSchedule(fs, OPS)

        @printf("%d\t%s\tfill_ghosts\t%.6f\t%d\n",
                Threads.nthreads(), label, t_fill, bytes)
        @printf("%d\t%s\tghost_schedule\t%.6f\t%d\n",
                Threads.nthreads(), label, t_schedule, sbytes)
        @printf("# %s: %d blocks, %d ghost points filled, %.1f ns per point-variable\n",
                label, nleaves(forest), ghostpoints(fs),
                1e9 * t_fill / (ghostpoints(fs) * NVARS))
        if PROFILE
            report_profile(label, () -> fill_ghosts!(fs, schedule))
        end
    end
    return nothing
end

# Filled points per fill: the stored extent less the owned one, summed
# over blocks. An overestimate at a domain edge without a hook, where
# nothing writes the outermost slab, but it is the same count before and
# after a change, which is what a per-point figure is compared on.
function ghostpoints(fs)
    stored = ntuple(d -> size(fs.work, d), D)
    owned = ntuple(d -> stored[d] - 2 * fs.G[d], D)
    return nblocks(fs) * (prod(stored) - prod(owned))
end

main()
