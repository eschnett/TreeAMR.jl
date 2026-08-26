# Helpers for the ghost-exchange tests. As in oracles.jl, these check
# the implementation against something independent of it — analytic
# polynomial values, an explicit tiling, and a direct count of writes.

using TreeAMR: TransferGroup, boxsize, ntransfers, target_range

"""
A polynomial of degree `deg` in each coordinate, varying per variable.
An operator of order `p` reproduces it exactly iff `deg < p`.
"""
makepoly(D, deg) = (x, v) -> sum(0.3v + 0.7d + 0.11 * (d + v) * x[d]^e
                                 for d in 1:D for e in 0:deg)

"""Largest deviation of any stored cell (interior *and* ghost) from `f`."""
function max_deviation(fs, f)
    forest = fs.forest
    worst = 0.0
    for b in 1:nblocks(fs), v in 1:fs.nvars
        k = blockkey(fs, b)
        blk = blockview(fs, b, v)
        for idx in CartesianIndices(blk)
            x = cell_center(forest, k, Tuple(idx))
            worst = max(worst, abs(blk[idx] - f(x, v)))
        end
    end
    return worst
end

"""Fill with `f`, exchange ghosts, and report the worst error anywhere."""
function exchange_error(forest, ops, f; nvars=2)
    schedule = GhostSchedule(forest, ops)
    fs = FieldSet(forest, nvars)
    fill_by_coordinates!(f, fs)
    fill_ghosts!(fs, schedule; boundary=boundary_by_coordinates(f))
    return max_deviation(fs, f)
end

"""
How many times the schedule writes each stored cell. Every ghost cell
must be written exactly once and no interior cell may be touched, which
together prove the ghost regions are partitioned without gaps or
double-writes.
"""
function write_counts(schedule::GhostSchedule{T,D}) where {T,D}
    forest = schedule.forest
    stored = forest.N + 2 * forest.G
    counts = zeros(Int, ntuple(_ -> stored, D)..., nleaves(forest))

    function tally!(group::TransferGroup)
        blen = boxsize(group)
        first = ntuple(d -> group.stencils[d].targetfirst, D)
        for t in 1:ntransfers(group)
            b = group.targetblocks[t]
            for off in CartesianIndices(ntuple(d -> 0:(blen[d] - 1), D))
                counts[ntuple(d -> first[d] + off[d], D)..., b] += 1
            end
        end
    end

    foreach(tally!, schedule.phase1)
    for groups in schedule.phase2
        foreach(tally!, groups)
    end
    for r in schedule.boundaries
        for idx in r.region
            counts[Tuple(idx)..., r.block] += 1
        end
    end
    return counts
end

"""How many transfers of each kind the schedule holds."""
function transfer_counts(s::GhostSchedule)
    counts = Dict(:copy => 0, :restrict => 0, :prolong => 0)
    for g in s.phase1
        counts[g.kind] += ntransfers(g)
    end
    for groups in s.phase2, g in groups
        counts[g.kind] += ntransfers(g)
    end
    return counts
end

"""
A hierarchy with three levels meeting, built by nesting two refinement
regions so that distant blocks stay at level 0. A single broad region
would not do: balancing would lift everything off the coarsest level and
leave only two levels in play.
"""
function nested_forest(::Val{D}; N=4, G=1, roots=4, periodic=ntuple(_ -> false, D)) where {D}
    forest = Forest(ntuple(_ -> roots, D); N=N, G=G, periodic=periodic,
                    extents=ntuple(_ -> (0.0, Float64(roots)), D))
    center = ntuple(_ -> 1.5, D)
    near(c, r) = all(d -> abs(c[d] - center[d]) <= r, 1:D)
    refine_where!(forest, (c, lvl) -> (lvl == 0 && near(c, 1.0)) ||
                                      (lvl == 1 && near(c, 0.3)), 2)
    return forest
end

"""Refine wherever `pred(block center, level)` holds, then rebalance."""
function refine_where!(forest, pred, passes)
    for _ in 1:passes
        targets = filter(forest.leaves) do k
            ext = block_extent(forest, k)
            pred(ntuple(d -> (ext[d][1] + ext[d][2]) / 2, length(ext)), level(k))
        end
        isempty(targets) && break
        refine!(forest, targets)
        balance!(forest)
    end
    return forest
end

"""
Compare an `M`-root periodic domain against the middle tile of an
explicit `3M`-root tiling of the same data.

Periodicity *is* the domain wrapping onto itself, so this is the
definitional test — and unlike a polynomial (which is discontinuous
across the seam, so no interpolation can reproduce it there) it works
for arbitrary data. Refinement is driven towards the seam so the
coarse/fine interfaces sit exactly where the wraparound happens.

Returns `(maxdiff, ncells)`, or `nothing` if the two refinement patterns
failed to correspond.
"""
function periodic_vs_tiled(::Val{D}, M::Int; N=4, G=1, nvars=2,
                           ops=Operators(prolongation=2, restriction=2),
                           passes=2) where {D}
    L = Float64(M)
    seam(c, lvl) = lvl < passes &&
        all(d -> min(mod(c[d], L), L - mod(c[d], L)) < 0.55, 1:D)

    periodic = Forest(ntuple(_ -> M, D); N=N, G=G, periodic=ntuple(_ -> true, D),
                      extents=ntuple(_ -> (0.0, L), D))
    refine_where!(periodic, seam, passes)

    tiled = Forest(ntuple(_ -> 3M, D); N=N, G=G, extents=ntuple(_ -> (-L, 2L), D))
    refine_where!(tiled, seam, passes)

    data = (x, v) -> sum(sin(3.1 * mod(x[d], L) + 0.7v) * (1 + 0.3d) for d in 1:D)

    fsp = FieldSet(periodic, nvars)
    fill_by_coordinates!(data, fsp)
    fill_ghosts!(fsp, GhostSchedule(periodic, ops))

    fst = FieldSet(tiled, nvars)
    fill_by_coordinates!(data, fst)
    fill_ghosts!(fst, GhostSchedule(tiled, ops); boundary=boundary_by_coordinates(data))

    lower(forest, k) = ntuple(d -> block_extent(forest, k)[d][1], D)
    index = Dict((level(k), lower(tiled, k)) => b for (b, k) in enumerate(tiled.leaves))

    worst = 0.0
    ncells = 0
    for (b, k) in enumerate(periodic.leaves)
        tb = get(index, (level(k), lower(periodic, k)), nothing)
        tb === nothing && return nothing
        for v in 1:nvars
            a, c = blockview(fsp, b, v), blockview(fst, tb, v)
            for i in CartesianIndices(a)
                worst = max(worst, abs(a[i] - c[i]))
                ncells += 1
            end
        end
    end
    return (worst, ncells)
end
