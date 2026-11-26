# Closed blocks across a rotating seam (M22 step 7): the copies of a seam
# point are one number through the turn (rule 2, the primary the orbit's
# lexicographically greatest real position), a point on the axis is
# projected onto the turn's invariant part (rule 3), and the reductions
# count a seam point and the axis as their share of the full domain.
#
# The forests are quadrants of `4 × 4` roots on `(0, 4)²` and half planes
# of `4 × 8` roots on `(0, 4) × (0, 8)`, in 3D with two roots along `z`,
# refined twice about the axis by a rule that is a staircase in the plane,
# so that the refined regions have concave corners, and in 3D refined
# further on the lower half of `z` only, so that the axis line crosses two
# levels.

using TreeAMR: ghost_plane_parts, oriented_neighbors, find_leaf, run_axis!

const SEAM_OPS = Operators(prolongation=4, restriction=4)

# The seam's forest: `half` for a half turn about `x₂ = 4`, else a quarter
# turn about the origin; `other` the kind of the low face of `z` in 3D.
function seam_forest(D; half::Bool=false, other=:outer, N=8)
    roots = ntuple(d -> d == 2 && half ? 8 : d <= 2 ? 4 : 2, D)
    extents = ntuple(d -> (0.0, Float64(roots[d])), D)
    seam = half ? (; halfturn=(1, 2)) : (; rotating=(1, 2))
    forest = Forest(roots; N=N, extents=extents,
                    periodic=ntuple(d -> d == 3 && other === :periodic, D),
                    reflecting=ntuple(d -> (d == 3 && other === :reflect_lo, false), D),
                    seam...)
    refine_where!(forest, (x, l) -> l == 0 && seam_rule1(x, half), 1)
    refine_where!(forest, (x, l) -> l == 1 && seam_rule2(x, half), 1)
    return forest
end

# The refinement about the axis, by a block's centre in the reduced domain:
# with `p` the distance from the seam's face along `x` and `q` from the
# axis along `y`, a level-0 root within `p + q < 2` refines, a staircase;
# a level-1 block within `p + q < 1` on the lower half of `z`, and in 3D
# on the upper half a ring `½ < max(p, q) < 1` below `z = 1.5`, so that
# the axis line crosses two levels and a block on the seam sees a block of
# its level across the seam diagonally where the one straight across is
# coarser (a turned face-plane row, `ghost_plane_parts`).
seam_pq(x, half) = (x[1], abs(x[2] - (half ? 4 : 0)))
seam_rule1(x, half) = sum(seam_pq(x, half)) < 2
function seam_rule2(x, half)
    p, q = seam_pq(x, half)
    length(x) == 2 && return p + q < 1
    return (p + q < 1 && x[3] < 1) || (0.5 < max(p, q) < 1 && 1 < x[3] < 1.5)
end

# The variables: a scalar and a vector in the plane (and `z` in 3D), with
# the quarter turn's map, or the half turn's, which negates the in-plane
# components.
seam_map(D, half) = half ? [1; -2; -3; (D == 3 ? [4] : Int[])] :
                    [1; -3; 2; (D == 3 ? [4] : Int[])]
forestdim(::Forest{D}) where {D} = D
seam_set(forest, C, half; parity=nothing) =
    half ? FieldSet(forest, forestdim(forest) + 1; G=ghosts_for(C, 4), centering=C,
                    halfturn=seam_map(forestdim(forest), true), parity=parity) :
           FieldSet(forest, forestdim(forest) + 1; G=ghosts_for(C, 4), centering=C,
                    rotation=seam_map(forestdim(forest), false), parity=parity)

# The centerings symmetric in the plane with a vertex-like dimension.
seam_centerings(D) = D == 2 ? [vertexcentered(2)] :
                     [vertexcentered(3), facecentered(3, 3), edgecentered(3, 3)]

# The real point of a virtual point `x` (exact) and its orientation: the
# naive point beyond the seam mapped back as the seam's tables do, and
# wrapped along a periodic `z`.
function real_point(forest, x, half)
    a, b = x[1], x[2]
    r, a2, b2 = half ? (a < 0 ? (2, -a, 8 - b) : (0, a, b)) :
                a < 0 && b < 0 ? (2, -a, -b) : a < 0 ? (1, b, -a) :
                b < 0 ? (3, -b, a) : (0, a, b)
    y = Base.setindex(Base.setindex(x, a2, 1), b2, 2)
    if length(x) == 3 && forest.periodic[3]
        y = Base.setindex(y, mod(y[3], 2), 3)
    end
    return y, r
end

inside(forest, y) = all(d -> 0 <= y[d] <= forest.extents[d][2], eachindex(y))

# The exact position of stored index `idx` of local block `b`.
function exact_position(fs, b, idx)
    forest = fs.forest
    ext = block_extent(forest, blockkey(fs, b))
    c = staggers(fs)
    return ntuple(length(idx)) do d
        lo, hi = Rational{Int}.(ext[d])
        h = (hi - lo) / forest.N
        lo + (idx[d] - fs.G[d] - 1 + (1 - c[d]) // 2) * h
    end
end

# Every stored copy of a point, own or ghost, at a level where some block
# of that level owns the point (by its real position), against that own
# copy through the turn: `f_r(v) u_{σ_r(v)} + 0`. Returns `(compared,
# differing, worst)`: the copies that are not that value bit for bit, and
# the largest difference.
function seam_copies(fs, half)
    forest = fs.forest
    D = length(fs.G)
    c = staggers(fs)
    owners = Dict{Any,Dict{Int,Tuple{Int,Any}}}()
    for b in 1:nblocks(fs), I in CartesianIndices(ntuple(d -> forest.N + c[d], D))
        idx = Tuple(I) .+ fs.G
        x = exact_position(fs, b, idx)
        get!(get!(owners, x, Dict{Int,Tuple{Int,Any}}()), level(blockkey(fs, b)), (b, idx))
    end
    compared = differing = 0
    worst = 0.0
    for b in 1:nblocks(fs), I in CartesianIndices(size(fs.work)[1:D])
        idx = Tuple(I)
        y, r = real_point(forest, exact_position(fs, b, idx), half)
        inside(forest, y) || continue
        o = get(get(owners, y, Dict{Int,Tuple{Int,Any}}()), level(blockkey(fs, b)), nothing)
        o === nothing && continue
        ob, oidx = o
        compared += 1
        same = true
        for v in 1:fs.nvars
            σ = r == 0 ? v : Int(fs.rotvars[v, r + 1])
            f = fs.factors[v, 1 + 3^D * r]
            want = r == 0 ? fs.work[oidx..., v, ob] : f * fs.work[oidx..., σ, ob] + 0.0
            got = fs.work[idx..., v, b]
            reinterpret(UInt64, got) == reinterpret(UInt64, want) && continue
            same = false
            worst = max(worst, abs(got - want))
        end
        same || (differing += 1)
    end
    return compared, differing, worst
end

# How many ghosts split a face-plane row off a prolongation onto a copy
# across the seam (`ghost_plane_parts`'s turned diagonal).
function turned_splits(forest::Forest{D}, c) where {D}
    n = 0
    for k in forest.leaves, δ in TreeAMR.alldirections(Val(D))
        _, nbrs = oriented_neighbors(forest, k, δ)
        isempty(nbrs) && continue
        level(first(nbrs)) < level(k) || continue
        parts = ghost_plane_parts(forest, k, δ, δ, c, find_leaf(forest, first(nbrs)))
        parts === nothing && continue
        n += count(p -> p[1] === :copy && p[5].turns != 0, parts)
    end
    return n
end

@testset "Every stored copy across a seam is its level's own copy, turned: D=$D" for
        D in (2, 3)
    # Guards a seam copy filled from the wrong member of its orbit, a ghost
    # on the extension of a face plane turned the wrong way, a primary that
    # two ranks choose differently, and an axis left unprojected (M22 rules
    # 2 and 3): random data, filled, and every stored copy of a point at a
    # level that owns it is that own copy through the turn's map, `+0` for
    # a turned zero, bit for bit where the path is copies and injections:
    # a set vertex-like in the plane. A set cell-like in the plane (the
    # `z`-face) restricts a coarse copy across the seam with the turn
    # reversing its window, so there its copies agree to rounding, a few
    # ulp of the data. Quarter and half turns, refined about the axis with
    # concave corners, in 3D with the axis crossing two levels and a turned
    # face-plane row, over outer, reflecting and periodic `z`, every
    # centering symmetric in the plane, bounded and not, serially and over
    # 2 and 3 simulated ranks.
    nturned = 0
    for half in (false, true), other in (D == 2 ? (:outer,) : (:outer, :reflect_lo, :periodic)),
        C in seam_centerings(D), bounded in (false, true)
        ops = Operators(prolongation=4, restriction=4, bounded=bounded)
        forest = seam_forest(D; half=half, other=other)
        parity = other === :reflect_lo ?
                 [(EvenParity, EvenParity, EvenParity), (EvenParity, EvenParity, EvenParity),
                  (EvenParity, EvenParity, EvenParity), (EvenParity, EvenParity, OddParity)] :
                 nothing
        fs = seam_set(forest, C, half; parity=parity)
        data = randn(Xoshiro(30 + D), size(fs.work))
        fs.work .= data
        hook = boundary_by_coordinates((x, v) -> x[1] - x[2] + v)
        fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=hook)
        compared, differing, worst = seam_copies(fs, half)
        @test compared > 0
        C[1] === :vertex ? (@test differing == 0) :
                           (@test worst <= 16 * eps() * maximum(abs, fs.work))
        nturned += turned_splits(forest, staggers(fs))
        for P in (2, 3)
            sets = map(0:(P - 1)) do r
                g = seam_set(rank_forest(forest, r, P), C, half; parity=parity)
                g.work .= data[ntuple(_ -> :, D + 1)..., blockrange(g.forest)]
                g
            end
            scheds = [GhostSchedule(g, ops) for g in sets]
            bad = lockstep!(sets, scheds, CPU();
                            between=r -> apply_boundary!(sets[r], hook, scheds[r], CPU()))
            @test bad == 0
            @test all(r -> bitwise_equal(sets[r].work,
                                         fs.work[ntuple(_ -> :, D + 1)...,
                                                 blockrange(sets[r].forest)]), 1:P)
        end
    end
    D == 3 && @test nturned > 0
end

@testset "The axis is projected onto the turn's invariant part: D=$D" for D in (2, 3)
    # Guards rule 3: a point on the axis is its own image, so a scalar is
    # left alone, a turned vector's in-plane components vanish, and a
    # tensor's are averaged over the orbit, exactly and idempotently (the
    # orbit is summed in pairs a half turn apart). In 3D on a reflecting
    # `z` wall the mirror's projection and the axis's commute, bit for
    # bit, whichever runs first.
    for half in (false, true)
        forest = seam_forest(D; half=half, other=D == 3 ? :reflect_lo : :outer)
        # A scalar, a vector in the plane and the plane's symmetric tensor
        # `(Txx, Tyy, Txy)`: a quarter turn exchanges `Txx` and `Tyy` and
        # negates `Txy`; a half turn leaves all three alone.
        map_ = half ? [1, -2, -3, 4, 5, 6] : [1, -3, 2, 5, 4, -6]
        parity = D == 3 ? [ntuple(d -> d == 3 && v == 0 ? OddParity : EvenParity, 3)
                           for v in (1, 1, 1, 1, 1, 0)] : nothing
        kw = half ? (; halfturn=map_) : (; rotation=map_)
        fs = FieldSet(forest, 6; G=ghosts_for(vertexcentered(D), 4),
                      centering=vertexcentered(D), parity=parity, kw...)
        fs.work .= randn(Xoshiro(40 + D), size(fs.work))
        sched = GhostSchedule(fs, SEAM_OPS)
        hook = boundary_by_coordinates((x, v) -> 0.0)
        fill_ghosts!(fs, sched; boundary=hook)
        once = copy(fs.work)
        fill_ghosts!(fs, sched; boundary=hook)
        @test bitwise_equal(fs.work, once)
        axis = sched.axis
        @test axis !== nothing && length(axis.blocks) > 0
        ok = all(eachindex(Array(axis.blocks))) do i
            b, p = Int(axis.blocks[i]), Tuple(axis.points[i])
            u = fs.work[p..., :, b]
            half ? u[2] == 0 && u[3] == 0 :
                   u[2] == 0 && u[3] == 0 && u[4] == u[5] && u[6] == 0
        end
        @test ok
        if D == 3
            # The two projections in either order.
            fs.work .= randn(Xoshiro(41), size(fs.work))
            start = copy(fs.work)
            TreeAMR.run_phase!(fs, sched.selfimage, CPU())
            run_axis!(fs.work, fs, axis, CPU())
            first_mirror = copy(fs.work)
            fs.work .= start
            run_axis!(fs.work, fs, axis, CPU())
            TreeAMR.run_phase!(fs, sched.selfimage, CPU())
            @test !isempty(sched.selfimage)
            @test bitwise_equal(fs.work, first_mirror)
        end
    end
    # A set whose map turns nothing (a scalar) launches no projection.
    forest = seam_forest(D)
    fs = FieldSet(forest, 1; G=1, centering=vertexcentered(D), rotation=[1])
    @test !TreeAMR.turns_variables(fs)
end

@testset "A reduced domain's reductions are its share of the full one's" begin
    # Guards the counting (M22): with the trapezoid's weights a seam point
    # is half on each of its faces, so a quadrant's `total_mass` is a
    # quarter of the full plane's and a half plane's a half, with no seam
    # mask, and the normalized norm is the full plane's; a count over the
    # primaries counts each orbit once: a quadrant of a `(2n+1)²` grid
    # holds `n² + n + 1` orbits.
    n = 16                                          # nodes per quadrant side, less one
    for half in (false, true)
        reduced = half ? Forest((2, 4); N=8, halfturn=(1, 2), extents=((0.0, 2.0), (-2.0, 2.0))) :
                         Forest((2, 2); N=8, rotating=(1, 2), extents=((0.0, 2.0), (0.0, 2.0)))
        full = Forest((4, 4); N=8, extents=((-2.0, 2.0), (-2.0, 2.0)))
        # A field invariant under the turn: a function of the radius.
        f(x, v) = exp(-(x[1]^2 + x[2]^2)) + 0.25 * cos(x[1]^2 + x[2]^2)
        kw = half ? (; halfturn=[1]) : (; rotation=[1])
        fr = FieldSet(reduced, 1; G=1, centering=vertexcentered(2), kw...)
        ff = FieldSet(full, 1; G=1, centering=vertexcentered(2))
        fill_by_coordinates!(f, fr)
        fill_by_coordinates!(f, ff)
        share = half ? 1 / 2 : 1 / 4
        @test total_mass(fr) ≈ share * total_mass(ff) rtol = 64 * eps()
        ur, uf = statevector(fr), statevector(ff)
        gather!(ur, fr)
        gather!(uf, ff)
        @test volume_weighted_norm(fr, ur) ≈ volume_weighted_norm(ff, uf) rtol = 64 * eps()
        count = mesh_mapreduce(x -> 1, +, 0, fr; primaries=true)
        @test count == (half ? (n * (2n + 1) + n + 1) : n^2 + n + 1)
    end
end

@testset "A stencil pass and unify! keep a reduced domain the full one's: D=$D" for
        D in (2, 3)
    # Guards what a fill alone does not show (step 6's review found a fill
    # symmetric while its ghosts were wrong): a covariant field on the
    # quadrant and the half plane, filled, one 2nd-order Laplacian over
    # every own point, then `unify!`, against the full domain doing the
    # same, at every point of the reduced domain, to rounding; refined
    # about the axis with concave corners touching the seam and the axis.
    for half in (false, true)
        reduced = seam_forest(D; half=half)
        rootsfull = ntuple(d -> d == 1 ? 8 : (d == 2 && half ? 8 : d == 2 ? 8 : 2), D)
        extfull = ntuple(d -> d == 1 ? (-4.0, 4.0) : d == 2 ? (half ? (0.0, 8.0) : (-4.0, 4.0)) :
                              (0.0, 2.0), D)
        full = Forest(rootsfull; N=8, extents=extfull)
        # The same refinement, by the image of each block's centre in the
        # reduced domain.
        toreduced(x) = half ? (x[1] < 0 ? Base.setindex(Base.setindex(x, -x[1], 1),
                                                         8 - x[2], 2) : x) :
                       (x[1] >= 0 && x[2] >= 0 ? x :
                        x[1] < 0 && x[2] >= 0 ? Base.setindex(Base.setindex(x, x[2], 1), -x[1], 2) :
                        x[1] < 0 ? Base.setindex(Base.setindex(x, -x[1], 1), -x[2], 2) :
                        Base.setindex(Base.setindex(x, -x[2], 1), x[1], 2))
        refine_where!(full, (x, l) -> l == 0 && seam_rule1(toreduced(x), half), 1)
        refine_where!(full, (x, l) -> l == 1 && seam_rule2(toreduced(x), half), 1)
        @test nleaves(full) == (half ? 2 : 4) * nleaves(reduced)
        # A scalar and a vector, covariant under the turn.
        cen = half ? (0.0, 4.0) : (0.0, 0.0)
        function g(x, v)
            p, q = x[1] - cen[1], x[2] - cen[2]
            r2 = p^2 + q^2
            v == 1 && return exp(-r2 / 4) + 0.1 * (D == 3 ? x[3] : 0.0)
            v == 2 && return -q * exp(-r2 / 3) + p * cos(r2 / 5)
            v == 3 && return p * exp(-r2 / 3) + q * cos(r2 / 5)
            return sin(r2 / 7)
        end
        nv = D + 1
        map_ = seam_map(D, half)
        kw = half ? (; halfturn=map_) : (; rotation=map_)
        C = vertexcentered(D)
        G = ghosts_for(C, 4)
        fr = FieldSet(reduced, nv; G=G, centering=C, kw...)
        ff = FieldSet(full, nv; G=G, centering=C)
        hook = boundary_by_coordinates(g)
        sr, sf = GhostSchedule(fr, SEAM_OPS), GhostSchedule(ff, SEAM_OPS)
        for (fs, s) in ((fr, sr), (ff, sf))
            fill_by_coordinates!(g, fs)
            fill_ghosts!(fs, s; boundary=hook)
            w = copy(fs.work)
            sp = block_spacings(fs.forest, Float64)
            for b in 1:nblocks(fs), I in CartesianIndices(ntuple(d -> fs.forest.N + 1, D)),
                v in 1:nv
                i = Tuple(I) .+ G
                lap = -2D * w[i..., v, b]
                for d in 1:D
                    lap += w[Base.setindex(i, i[d] + 1, d)..., v, b] +
                           w[Base.setindex(i, i[d] - 1, d)..., v, b]
                end
                fs.work[i..., v, b] = lap / sp[b]^2
            end
            unify!(fs, s)
        end
        # The full domain's value at each own point of its finest level.
        value = Dict{Any,Tuple{Int,Vector{Float64}}}()
        for b in 1:nblocks(ff), I in CartesianIndices(ntuple(d -> 9, D))
            x = exact_position(ff, b, Tuple(I) .+ G)
            l = level(blockkey(ff, b))
            get(value, x, (-1, Float64[]))[1] <= l && (value[x] = (l, ff.work[(Tuple(I) .+ G)..., :, b]))
        end
        worst, n = 0.0, 0
        for b in 1:nblocks(fr), I in CartesianIndices(ntuple(d -> 9, D))
            x = exact_position(fr, b, Tuple(I) .+ G)
            haskey(value, x) || continue
            n += 1
            worst = max(worst, maximum(abs, fr.work[(Tuple(I) .+ G)..., :, b] .- value[x][2]))
        end
        scale = maximum(abs, ff.work)
        @test n > 0
        @test worst <= 256 * eps() * scale
    end
end

@testset "A regrid across a seam keeps the kept copies and the turn's symmetry: D=$D" for
        D in (2, 3)
    # Guards the kept-over-created pass through the turn (M22 steps 6 and
    # 7): after a regrid that refines blocks along the seam and leaves
    # their neighbours, before any fill, every created copy of a point a
    # kept block holds at the finest level equals the kept block's old
    # value through the turn, bit for bit; and a radially symmetric firing
    # pattern flags boxes whose forest is symmetric under exchanging the
    # plane's axes, for buffer 0, 1 and 2.
    for half in (false,)
        forest = seam_forest(D; half=half)
        C = vertexcentered(D)
        fs = seam_set(forest, C, half)
        fs.work .= randn(Xoshiro(50 + D), size(fs.work))
        hook = boundary_by_coordinates((x, v) -> 0.0)
        sched = GhostSchedule(fs, SEAM_OPS)
        fill_ghosts!(fs, sched; boundary=hook)
        oldfs = deepcopy(fs)
        oldleaves = copy(forest.leaves)
        # Refine the level-1 blocks on the seam faces next to the axis.
        flags = [level(k) == 1 && block_extent(forest, k)[1][1] == 0 &&
                 block_extent(forest, k)[2][2] <= 1.5 ? Refine : Keep for k in forest.leaves]
        @test regrid!(forest, fs => sched; flags=flags, boundary=hook)
        # Every own copy of a point against the old storage's copy at the
        # same real point and level, through the turn, where the old
        # copy's block is kept and is the finest holder.
        owners = Dict{Any,Dict{Int,Tuple{Int,Any}}}()
        for b in 1:nblocks(oldfs), I in CartesianIndices(ntuple(d -> 9, D))
            idx = Tuple(I) .+ oldfs.G
            x = exact_position(oldfs, b, idx)
            get!(get!(owners, x, Dict{Int,Tuple{Int,Any}}()), level(blockkey(oldfs, b)),
                 (b, idx))
        end
        ncreated = 0
        same = all(1:nblocks(fs)) do b
            k = blockkey(fs, b)
            k in oldleaves && return true
            all(CartesianIndices(ntuple(d -> 9, D))) do I
                idx = Tuple(I) .+ fs.G
                x = exact_position(fs, b, idx)
                # The point itself, and its image across the seam: a point
                # `(0, y)` is the turn of `(y, 0)`, `u(0, y) = Q u(y, 0)`,
                # and `(x, 0)` the inverse turn of `(0, x)`.
                images = [(x, 0)]
                x[1] == 0 && push!(images, (Base.setindex(Base.setindex(x, x[2], 1), 0, 2), 1))
                x[2] == 0 && push!(images, (Base.setindex(Base.setindex(x, 0, 1), x[1], 2), 3))
                all(images) do (y, r)
                    o = get(get(owners, y, Dict{Int,Tuple{Int,Any}}()), level(k), nothing)
                    o === nothing && return true
                    ob, oidx = o
                    blockkey(oldfs, ob) in forest.leaves || return true
                    ncreated += 1
                    all(1:fs.nvars) do v
                        σ = r == 0 ? v : Int(fs.rotvars[v, r + 1])
                        f = fs.factors[v, 1 + 3^D * r]
                        want = r == 0 ? oldfs.work[oidx..., v, ob] :
                               f * oldfs.work[oidx..., σ, ob] + 0.0
                        reinterpret(UInt64, fs.work[idx..., v, b]) == reinterpret(UInt64, want)
                    end
                end
            end
        end
        @test same && ncreated > 0
    end
    # The flags of a radially symmetric pattern, and the regridded forests.
    for buffer in (0, 1, 2)
        forest = Forest(ntuple(d -> d <= 2 ? 4 : 2, D); N=8, rotating=(1, 2),
                        extents=ntuple(d -> (0.0, d <= 2 ? 4.0 : 2.0), D))
        for round in 1:2
            fs = FieldSet(forest, 1; G=1, centering=vertexcentered(D), rotation=[1])
            fill_by_coordinates!((x, v) -> x[1]^2 + x[2]^2, fs)
            boxes = firing_boxes((w, idx, b, x) -> 1.5 < w[idx..., 1, b] < 4.5, fs)
            flags = [n == 0 ? Keep : (level(forest.leaves[b]) < 2 ? Refine : Keep, box)
                     for (b, (n, box)) in enumerate(boxes)]
            regrid!(forest, (fs => nothing,); flags=flags, buffer=buffer)
            swapped = Set(map(k -> (block_extent(forest, k)[2], block_extent(forest, k)[1],
                                    block_extent(forest, k)[3:end]...), forest.leaves))
            @test swapped == Set(map(k -> block_extent(forest, k), forest.leaves))
        end
        @test maxlevel(forest) == 2
    end
end
