# Closed blocks (M22, "Closed blocks" in CODE.md), step 2: the layout,
# the uniform fill and outer faces. A vertex-like dimension owns its
# closed range, `N + 1` points, so a point on a block face is held by
# every block whose closed box contains it; the fill makes every copy
# its primary's, the copy on the block lying lowest along the dimensions
# the point is shared in.

using TreeAMR: UNIFY_TAG, PHASE1_TAG, primary_bits, piece_copies, piece_primary,
               closed_pieces, run_group!

# The vertex-like centerings, and a cell-centered control.
closed_centerings(D) = vcat([vertexcentered(D)], [facecentered(D, d) for d in 1:D],
                            D >= 2 ? [edgecentered(D, d) for d in 1:D] : [])

# Uniform forests, periodic and with outer faces; one root across a
# periodic dimension makes a block its own neighbour there.
closed_forests(D; N=4) = [
    ("one root periodic", Forest(ntuple(_ -> 1, D); N=N, periodic=ntuple(_ -> true, D))),
    ("two roots periodic", Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> true, D))),
    ("outer", Forest(ntuple(_ -> 2, D); N=N)),
    ("mixed", Forest(ntuple(d -> (1, 3, 2)[d], D); N=N,
                     periodic=ntuple(d -> isodd(d), D)))]

# Every stored point of every block, by its exact position (wrapped where
# periodic): the block, the stored index, and whether it is an own point.
function closed_point_groups(fs::FieldSet{T,D}) where {T,D}
    c = staggers(fs)
    groups = Dict{NTuple{D,Rational{Int}},Vector{Tuple{Int,NTuple{D,Int},Bool}}}()
    for b in 1:nblocks(fs), I in CartesianIndices(size(fs.work)[1:D])
        idx = Tuple(I)
        own = all(d -> fs.G[d] < idx[d] <= fs.G[d] + fs.forest.N + c[d], 1:D)
        push!(get!(groups, exact_point(fs, b, idx), []), (b, idx, own))
    end
    return groups
end

# The primary of a point among its own copies by the rule: the greatest
# class tuple, dimension 1 first, which is the greatest own index, since
# the copies differ only along the dimensions the point is shared in, at
# the low plane on one block and the high plane on the other.
function closed_primary(copies, offset)
    idx, b = maximum((idx, offset + b) for (b, idx, own) in copies if own)
    return (b, idx)
end

bits(x) = reinterpret(UInt, x)

# Operators a layout of either ghost width takes: a set without ghosts along
# a cell-like dimension needs the conservative family at order 1.
closed_ops(G) = G == 0 ?
    Operators(family=Conservative, prolongation=1, restriction=2, vertex_prolongation=2) :
    Operators(prolongation=2, restriction=2)

@testset "Every copy of a point is its primary's after a fill: D=$D" for D in (1, 2, 3)
    # Guards a copy left out of stage 0, a ghost read before its source
    # was unified, or a primary chosen by another rule than the greatest
    # class: random data on every own point, every copy different, and
    # after the fill every stored copy of a point holds its primary's
    # value bit for bit — and the mask says which copy that is.
    rng = Xoshiro(2200 + D)
    hook = boundary_by_coordinates((x, v) -> sum(x) + v)
    for (name, forest) in closed_forests(D), C in closed_centerings(D), G in (0, 1)
        fs = FieldSet(forest, 2; G=G, centering=C)
        fs.work .= randn(rng, size(fs.work))
        before = copy(fs.work)
        sched = GhostSchedule(fs, closed_ops(G))
        fill_ghosts!(fs, sched; boundary=hook)
        mask = primary_mask(fs)
        ok, okmask, shared = true, true, 0
        for (p, copies) in closed_point_groups(fs)
            any(x -> x[3], copies) || continue          # beyond an outer face
            pb, pidx = closed_primary(copies, 0)
            count(x -> x[3], copies) > 1 && (shared += 1)
            for (b, idx, own) in copies, v in 1:2
                bits(fs.work[idx..., v, b]) == bits(before[pidx..., v, pb]) || (ok = false)
            end
            for (b, idx, own) in copies
                own || continue
                I = ntuple(d -> idx[d] - fs.G[d], D)
                isprimary(mask, I..., b) == ((b, idx) == (pb, pidx)) || (okmask = false)
            end
        end
        @test ok
        @test okmask
        @test shared > 0
    end
end

@testset "fill_by_coordinates! gives every copy the same value without a fill: D=$D" for
        D in (1, 2, 3)
    # Guards a position formed per block: the copies of a point are at
    # one position bit for bit, so a callback of position fills them alike.
    # (Across a periodic wrap the copies are a period apart, and a callback
    # gets each its own position; a periodic one would agree to rounding.)
    for (name, forest) in closed_forests(D), C in closed_centerings(D)
        fs = FieldSet(forest, 1; G=1, centering=C)
        fill_by_coordinates!((x, v) -> sum(d -> sin(3x[d] + d), 1:D), fs)
        byposition = Dict{Any,Vector{Float64}}()
        for b in 1:nblocks(fs), I in CartesianIndices(TreeAMR.interiorranges(fs))
            x = coordinates(fs, b, Tuple(I))
            push!(get!(byposition, bits.(x), Float64[]), fs.work[I, 1, b])
        end
        @test all(v -> all(y -> bits(y) == bits(first(v)), v), values(byposition))
        name in ("two roots periodic", "outer") &&
            @test any(v -> length(v) > 1, values(byposition))
    end
end

@testset "A block that is its own neighbour has its high plane primary: D=$D" for
        D in (1, 2, 3)
    # Guards a point a block shares with itself across a periodic
    # dimension of one root: its copies are distinct by class, and the high
    # plane's copy is the primary, along each such dimension at once.
    forest = Forest(ntuple(_ -> 1, D); N=4, periodic=ntuple(_ -> true, D))
    for C in closed_centerings(D)
        c = staggers(C)
        b = primary_bits(forest, 1, c)
        for cls in Iterators.product(ntuple(d -> c[d] == 1 ? (0:2) : (1:1), D)...)
            piece = sum(cls[d] * 3^(d - 1) for d in 1:D)
            @test ((b >> piece) & 1 == 1) == !any(==(0), cls)
            @test length(piece_copies(forest, 1, cls)) == 2^count(!=(1), cls)
        end
    end
end

@testset "A secondary's primary lies below it, never above: D=$D" for D in (1, 2, 3)
    # Guards the rule's locality: the direction from a secondary to its
    # primary is -1 exactly where the piece is the low plane and a block
    # lies below, across a periodic wrap too, and 0 everywhere else, so
    # the secondaries of a block have few directions.
    for (name, forest) in closed_forests(D), C in closed_centerings(D)
        c = staggers(C)
        for j in 1:nleaves(forest), cls in closed_pieces(c)
            k = forest.leaves[j]
            below = ntuple(D) do d
                δ = ntuple(e -> e == d ? -1 : 0, D)
                !isempty(TreeAMR.oriented_neighbors(forest, k, δ)[2])
            end
            _, _, _, δ = piece_primary(piece_copies(forest, j, cls))
            @test δ == ntuple(d -> cls[d] == 0 && below[d] ? -1 : 0, D)
        end
    end
end

@testset "Stage 0 takes one transfer group per direction to a primary: D=$D" for
        D in (1, 2, 3)
    # Guards a launch per piece where one per direction serves: on a
    # uniform mesh a set with `v` vertex-like dimensions has `2^v - 1`
    # directions to a primary, and, periodic, one group each, its pieces
    # merged into one box a block; with outer faces, a block with no block
    # below along a dimension writes its whole own range there, a second
    # span, so `3^v - 2^v`. Phase 1 keeps its `3^D - 1`.
    for C in closed_centerings(D), (periodic, roots) in ((true, 8), (true, 1), (true, 2),
                                                         (false, 3))
        forest = Forest(ntuple(_ -> roots, D); N=4, periodic=ntuple(_ -> periodic, D))
        sched = GhostSchedule(FieldSet(forest, 1; G=1, centering=C), closed_ops(1))
        v = count(==(1), staggers(C))
        @test length(sched.unify) == (periodic ? 2^v - 1 : 3^v - 2^v)
        @test length(sched.phase1) == 3^D - 1
    end
end

@testset "A count over a set is its number of distinct points: D=$D" for D in (1, 2, 3)
    # Guards a count or a sum that takes a shared point at every copy.
    for (name, forest) in closed_forests(D), C in closed_centerings(D)
        fs = FieldSet(forest, 2; G=1, centering=C)
        c = staggers(C)
        distinct = prod(forest.roots[d] * forest.N + (forest.periodic[d] ? 0 : c[d])
                        for d in 1:D)
        @test mesh_mapreduce(x -> 1, +, 0, fs; vars=1, primaries=true) == distinct
        @test mesh_mapreduce(x -> 1, +, 0, fs; vars=1, primaries=false) ==
              nblocks(fs) * prod(forest.N + c[d] for d in 1:D)
        @test mesh_fold((acc, p) -> acc + 1, +, 0, fs; vars=1, primaries=true) == distinct
        @test_throws "primaries" mesh_mapreduce(x -> 1, +, 0, fs)
        @test_throws "primaries" block_fold((acc, p) -> acc + 1, +, 0, fs)
    end
    # A cell-centered set has no shared point, and needs no keyword.
    cells = FieldSet(Forest(ntuple(_ -> 2, D); N=4), 1; G=1)
    @test mesh_mapreduce(x -> 1, +, 0, cells) == 2^D * 4^D
end

@testset "A constant is its own norm, and a unit its domain's volume: D=$D" for
        D in (1, 2, 3)
    # Guards a norm that weights a shared point at every copy: the
    # trapezoid's weights, a half on each face plane of a block, give
    # every point its weight in the composite rule.
    for (name, forest) in closed_forests(D), C in closed_centerings(D)
        fs = FieldSet(forest, 2; G=1, centering=C)
        u = statevector(fs)
        fill!(u, 3.0)
        @test volume_weighted_norm(fs, u) ≈ 3.0 rtol = 4eps()
        @test volume_weighted_norm(fs, u; p=1) ≈ 3.0 rtol = 4eps()
        fs.work .= 1.0
        @test total_mass(fs, 1) ≈ prod(forest.roots) rtol = 4eps()
    end
end

@testset "The ranks hold the serial mask and stage 0 and fill alike: D=$D" for D in (1, 2, 3)
    # Guards a primary chosen by a rank-local index: every rank must
    # decide each piece as the serial forest does, and a fill over
    # simulated ranks must give the serial ghosts bit for bit.
    for (name, serial) in closed_forests(D; N=8), C in closed_centerings(D)
        n = nleaves(serial)
        c = staggers(C)
        sbits = [primary_bits(serial, j, c) for j in 1:n]
        ssched = GhostSchedule(FieldSet(serial, 1; G=1, centering=C), closed_ops(1))
        sunify = transfer_set(ssched.unify, 0)
        for P in (2, 3)
            mine = Set{Any}()
            for r in 0:(P - 1)
                forest = rank_forest(serial, r, P)
                range = blockrange(forest)
                fs = FieldSet(forest, 1; G=1, centering=C)
                @test Array(primary_mask(fs).pieces) == sbits[range]
                sched = GhostSchedule(fs, closed_ops(1))
                union!(mine, transfer_set(sched.unify, first(range) - 1))
                @test all(g -> issorted(g.targetblocks), sched.unify)
            end
            @test issubset(mine, sunify)
        end
        for P in (2, 3)
            same, _, _, bad = lockstep_ghost_fill(serial, P, C, PointValue, Float64)
            @test same && bad == 0
        end
    end
end

@testset "Stage 0 and phase 1 fill alike flat and shaped: D=$D" for D in (1, 2, 3)
    # Guards the flat launch a device takes over the new pieces.
    for (name, forest) in closed_forests(D), C in closed_centerings(D)
        fs = FieldSet(forest, 2; G=1, centering=C)
        sched = GhostSchedule(fs, closed_ops(1))
        @test !isempty(sched.unify)
        data = randn(Xoshiro(7), size(fs.work))
        for g in vcat(sched.unify, sched.phase1)
            fills = map((false, true)) do flat
                work = copy(data)
                run_group!(work, work, g, fs.nvars, TreeAMR.CPU(); flat=flat)
                bits.(work)
            end
            @test fills[1] == fills[2]
        end
    end
end

# --- the test wave equation on closed blocks ------------------------------

# The wave's right-hand side with a boundary hook, and a variant that
# skips the secondaries, as an application kernel would, with `unify!`
# after it.
@kernel function closed_wave_kernel!(du, @Const(work), @Const(spacings), mask,
                                     ::Val{D}, ::Val{G}, ::Val{S}) where {D,G,S}
    I = @index(Global, NTuple)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))
    if !S || isprimary(mask, inner..., b)
        c = ntuple(d -> I[d] + G[d], Val(D))
        u0 = work[c..., 1, b]
        lap = zero(eltype(du))
        for d in 1:D
            lap += work[Base.setindex(c, c[d] + 1, d)..., 1, b] - 2 * u0 +
                   work[Base.setindex(c, c[d] - 1, d)..., 1, b]
        end
        h = spacings[b]
        du[inner..., 1, b] = work[c..., 2, b]
        du[inner..., 2, b] = lap / (h * h)
    end
end

function closed_wave_rhs!(du, u, fs, sched, hook, spacings, skip::Bool, unify::Bool)
    scatter!(fs, u)
    fill_ghosts!(fs, sched; boundary=hook)
    map_blocks!(closed_wave_kernel!, fs, statearray(du, fs), fs.work, spacings,
                primary_mask(fs), Val(length(fs.G)), Val(fs.G), Val(skip))
    unify && unify!(du, fs, sched)
    return du
end

function closed_rk4!(u, fs, sched, hook, dt, steps; skip=false, unify=false)
    sp = block_spacings(fs.forest)
    k1, k2, k3, k4, tmp = (similar(u) for _ in 1:5)
    f!(du, v) = closed_wave_rhs!(du, v, fs, sched, hook, sp, skip, unify)
    for _ in 1:steps
        f!(k1, u)
        tmp .= u .+ (dt / 2) .* k1
        f!(k2, tmp)
        tmp .= u .+ (dt / 2) .* k2
        f!(k3, tmp)
        tmp .= u .+ dt .* k3
        f!(k4, tmp)
        u .+= (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return u
end

# A box `[0, 1]^D` with outer faces, a pulse symmetric about its centre.
closed_pulse(x, v) = v == 1 ? exp(-sum(d -> (x[d] - 0.5)^2, 1:length(x)) / 0.02) : 0.0

# The largest difference between the copies of every own point, over the
# state vector (no fill: what the integrator holds).
function closed_copy_spread(fs::FieldSet{T,D}, u) where {T,D}
    scatter!(fs, u)
    worst = 0.0
    for copies in values(closed_point_groups(fs)), v in 1:fs.nvars
        own = [fs.work[idx..., v, b] for (b, idx, o) in copies if o]
        isempty(own) || (worst = max(worst, maximum(own) - minimum(own)))
    end
    return worst
end

@testset "The closed wave converges at second order with outer faces: D=$D" for D in (1, 2)
    # Guards the hook at an outer face, whose wall plane is now evolved:
    # the sine mode with the exact solution as the hook on a box of outer
    # faces converges at the rate of the periodic study.
    hs, l2 = Float64[], Float64[]
    for N in (8, 16, 32)
        forest = Forest(ntuple(_ -> 2, D); N=N, extents=ntuple(_ -> (0.0, 1.0), D))
        fs = FieldSet(forest, 2; G=1, centering=vertexcentered(D))
        sched = GhostSchedule(fs, Operators(prolongation=2, restriction=2))
        L, m = 1.0, 1
        tend = 0.1
        fill_by_coordinates!(wave_exact(D, L, m, 0.0), fs)
        u = statevector(fs)
        gather!(u, fs)
        h = minimum_spacing(forest)
        steps = ceil(Int, tend / (0.25h))
        dt = tend / steps
        # The hook holds the exact solution at its time, which RK4's stages
        # would need per stage; a static hook at the final time's data is
        # the error's own order at the wall, so the exact one per step.
        t = 0.0
        sp = block_spacings(forest)
        k = [similar(u) for _ in 1:4]
        tmp = similar(u)
        f!(du, v, s) = closed_wave_rhs!(du, v, fs, sched,
                                        boundary_by_coordinates(wave_exact(D, L, m, s)),
                                        sp, false, false)
        for _ in 1:steps
            f!(k[1], u, t)
            tmp .= u .+ (dt / 2) .* k[1]
            f!(k[2], tmp, t + dt / 2)
            tmp .= u .+ (dt / 2) .* k[2]
            f!(k[3], tmp, t + dt / 2)
            tmp .= u .+ dt .* k[3]
            f!(k[4], tmp, t + dt)
            u .+= (dt / 6) .* (k[1] .+ 2 .* k[2] .+ 2 .* k[3] .+ k[4])
            t += dt
        end
        exact = FieldSet(forest, 2; G=1, centering=vertexcentered(D))
        fill_by_coordinates!(wave_exact(D, L, m, t), exact)
        ue = statevector(exact)
        gather!(ue, exact)
        push!(hs, h)
        push!(l2, volume_weighted_norm(fs, u .- ue))
    end
    @test convergence_rate(hs, l2) ≈ 2.0 atol = 0.15
end


@testset "A closed box is mirror-symmetric, its walls evolved or prescribed: D=$D" for
        D in (1, 2)
    # Guards a wall treated by which way an axis runs: symmetric data in
    # a box of outer faces stay symmetric to rounding, whether the hook
    # fills the ghosts beyond an evolved wall, prescribes the wall plane,
    # or does either relative to the data.
    # Three roots of unit width, so that the mirror image of a block is a
    # block whose corner is exact, and a pulse about the centre.
    L = 3.0
    forest = Forest(ntuple(_ -> 3, D); N=8, extents=ntuple(_ -> (0.0, L), D))
    pulse = x -> exp(-sum(d -> (x[d] - L / 2)^2, 1:D) / 0.2)
    data = (x, v) -> v == 1 ? pulse(x) : zero(x[1])
    hooks = ("evolved" => boundary_by_coordinates(data),
             "prescribed" => boundary_by_coordinates(data; wall=:prescribed),
             "relative" => CellBoundary(Relative(AllVariables(x -> (pulse(x), 0.0)))),
             "relative prescribed" =>
                 CellBoundary(Relative(AllVariables(x -> (pulse(x), 0.0)));
                              wall=:prescribed))
    for (name, hook) in hooks
        fs = FieldSet(forest, 2; G=1, centering=vertexcentered(D))
        sched = GhostSchedule(fs, Operators(prolongation=2, restriction=2))
        fill_by_coordinates!(data, fs)
        u = statevector(fs)
        gather!(u, fs)
        closed_rk4!(u, fs, sched, hook, 0.25 * minimum_spacing(forest), 20)
        scatter!(fs, u)
        fill_ghosts!(fs, sched; boundary=hook)
        result = point_mirror_error(fs, L)
        @test result !== nothing
        worst, count = result
        @test count > 0
        @test worst < 1e-13
        @test maximum(abs, u) > 0.1                       # the pulse is still there
    end
end

@testset "A prescribed wall takes the hook's value at every copy, an evolved one keeps its own: D=$D" for
        D in (1, 2, 3)
    # Guards stage W: a hook declared to prescribe the wall writes every
    # wall point, through its primary and stage 0 at every other copy, and
    # one that does not leaves the wall to the state.
    for C in closed_centerings(D)
        c = staggers(C)
        forest = Forest(ntuple(_ -> 2, D); N=4, periodic=ntuple(d -> d == D && D > 1, D))
        data = (x, v) -> 7.0 + v + sum(x)
        for (wall, prescribed) in ((:evolved, false), (:prescribed, true))
            fs = FieldSet(forest, 2; G=1, centering=C)
            fs.work .= randn(Xoshiro(5), size(fs.work))
            before = copy(fs.work)
            sched = GhostSchedule(fs, closed_ops(1))
            fill_ghosts!(fs, sched; boundary=boundary_by_coordinates(data; wall=wall))
            nwall, ok = 0, true
            for b in 1:nblocks(fs), I in CartesianIndices(TreeAMR.interiorranges(fs))
                x = coordinates(fs, b, Tuple(I))
                onwall = any(d -> c[d] == 1 && !forest.periodic[d] &&
                                  (x[d] == 0.0 || x[d] == forest.roots[d]), 1:D)
                onwall || continue
                # A copy across a periodic wrap, the low plane, is a period
                # from its primary, whose value it takes: the hook's at the
                # primary.
                any(d -> forest.periodic[d] && x[d] == 0.0, 1:D) && continue
                nwall += 1
                for v in 1:2
                    prescribed ? (fs.work[I, v, b] == data(x, v) || (ok = false)) :
                                 (fs.work[I, v, b] != data(x, v) || (ok = false))
                end
            end
            @test ok
            @test nwall > 0 || all(==(0), c[1:max(1, D - 1)])
        end
    end
end

@testset "Skipping the secondaries with unify! is the full run at the primaries: D=$D" for
        D in (1, 2)
    # Guards `unify!` on a state vector: a right-hand side that computes
    # the primaries only and unifies `du` evolves the primaries exactly as
    # one that computes every copy, and with `unify!` alone a state that
    # starts unified stays so.
    for (name, forest) in (("outer", Forest(ntuple(_ -> 3, D); N=8,
                                            extents=ntuple(_ -> (0.0, 1.0), D))),
                           ("periodic", Forest(ntuple(d -> d == 1 ? 1 : 3, D); N=8,
                                               periodic=ntuple(_ -> true, D),
                                               extents=ntuple(d -> (0.0, d == 1 ? 1.0 : 3.0),
                                                              D))))
        hook = boundary_by_coordinates(closed_pulse)
        runs = map(((false, false), (true, true), (false, true))) do (skip, unify)
            fs = FieldSet(forest, 2; G=1, centering=vertexcentered(D))
            sched = GhostSchedule(fs, Operators(prolongation=2, restriction=2))
            fill_by_coordinates!(closed_pulse, fs)
            # A unified state to start from: across a periodic wrap the
            # copies of a point are a period apart, and the initial data
            # put each its own value until a fill unifies them.
            fill_ghosts!(fs, sched; boundary=hook)
            u = statevector(fs)
            gather!(u, fs)
            closed_rk4!(u, fs, sched, hook, 0.2 / 24, 12; skip=skip, unify=unify)
            (fs, u)
        end
        fs, full = runs[1]
        mask = primary_mask(fs)
        A, B = statearray(full, fs), statearray(runs[2][2], fs)
        same = all(CartesianIndices(size(A)[1:D])) do I
            all(b -> !isprimary(mask, Tuple(I)..., b) ||
                     all(v -> bits(A[I, v, b]) == bits(B[I, v, b]), 1:2), 1:nblocks(fs))
        end
        @test same
        @test closed_copy_spread(fs, runs[2][2]) == 0
        @test closed_copy_spread(fs, runs[3][2]) <= 1e-14
    end
end

@testset "unify! on a field set makes every copy its primary's: D=$D" for D in (1, 2, 3)
    # Guards the flux set's use of stage 0: a set without ghosts, computed
    # on every copy, unified.
    for (name, forest) in closed_forests(D), C in closed_centerings(D)
        fs = FieldSet(forest, 1; G=0, centering=C)
        fs.work .= randn(Xoshiro(3), size(fs.work))
        before = copy(fs.work)
        sched = GhostSchedule(fs, closed_ops(0))
        @test unify!(fs, sched) === fs
        ok = all(values(closed_point_groups(fs))) do copies
            pb, pidx = closed_primary(copies, 0)
            all(((b, idx, o),) -> fs.work[idx..., 1, b] === before[pidx..., 1, pb], copies)
        end
        @test ok
        # The state-vector form does the same on a state.
        u = statevector(fs)
        u .= randn(Xoshiro(4), length(u))
        unify!(u, fs, sched)
        @test closed_copy_spread(fs, u) == 0
    end
end

@testset "What later steps rebuild is refused, naming the step" begin
    # Guards a half-open path left reachable on closed blocks.
    ops = Operators(prolongation=2, restriction=2)
    two = Forest((2, 2); N=8)
    refine!(two, two.leaves[1])
    @test_throws "M22 step 3" GhostSchedule(FieldSet(two, 1; G=1,
                                                     centering=vertexcentered(2)), ops)
    @test_throws "M22 step 3" InterfaceSchedule(FieldSet(two, 1; G=0,
                                                         centering=facecentered(2, 1)))
    walled = Forest((2, 2); N=8, reflecting=((true, false), (false, false)))
    @test_throws "M22 step 4" GhostSchedule(FieldSet(walled, 1; G=1,
                                                     centering=vertexcentered(2),
                                                     parity=[EvenParity]), ops)
    out = Forest((2, 2); N=8, outflow=((true, false), (false, false)))
    @test_throws "M22 step 4" GhostSchedule(FieldSet(out, 1; G=1,
                                                     centering=vertexcentered(2)), ops)
    uniform = Forest((2, 2); N=8)
    vs = FieldSet(uniform, 1; G=1, centering=vertexcentered(2))
    @test_throws "M22 step 6" regrid!(uniform, vs => GhostSchedule(vs, ops);
                                      flags=fill(Keep, 4))
    @test_throws "M22 step 6" firing_boxes((w, i, b, x) -> true, vs)
    quad = Forest((2, 2); N=8, rotating=(1, 2))
    @test_throws "M22 step 7" GhostSchedule(FieldSet(quad, 1; G=1,
                                                     centering=vertexcentered(2),
                                                     rotation=(1,)), ops)
    @test_throws "M22 step 4" GhostSchedule(FieldSet(uniform, 1; G=1,
                                                     centering=vertexcentered(2),
                                                     walls=:symmetric), ops)
    @test_throws "wall = :prescribed" GhostSchedule(FieldSet(uniform, 1; G=1,
                                                             centering=vertexcentered(2),
                                                             walls=:symmetric), ops)
    # Cell-centered sets are not affected.
    @test GhostSchedule(FieldSet(two, 1; G=1), ops) isa GhostSchedule
    @test GhostSchedule(FieldSet(walled, 1; G=1, parity=[EvenParity]), ops) isa GhostSchedule
    # The keywords closed blocks replace say what replaced them.
    @test_throws "own range" map_blocks!((args...) -> nothing, vs; closed=true)
    @test_throws "appendix" statearray(statevector(vs), vs; appendix=true)
    @test_throws "wall must be" CellBoundary((x, v, δ) -> 0.0; wall=:symmetric)
end
