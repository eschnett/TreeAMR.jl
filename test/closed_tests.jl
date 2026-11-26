# Closed blocks (M22, "Closed blocks" in CODE.md), step 2: the layout,
# the uniform fill and outer faces. A vertex-like dimension owns its
# closed range, `N + 1` points, so a point on a block face is held by
# every block whose closed box contains it; the fill makes every copy
# its primary's, the copy on the block lying lowest along the dimensions
# the point is shared in.

using TreeAMR: UNIFY_TAG, PHASE1_TAG, primary_bits, piece_copies, piece_primary,
               facekind, ReflectingFace, ntransfers,
               closed_pieces, run_group!, touching_cells

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
            sub = TreeAMR.piece_representative(cls, c)
            @test length(piece_copies(forest, 1, sub, c)) == 2^count(!=(1), cls)
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
            sub = TreeAMR.piece_representative(cls, c)
            _, _, _, δ, _ = piece_primary(piece_copies(forest, j, sub, c))
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

# Meshes of two and three levels (M22 step 3), each balanced: the
# nested mesh with outer faces and periodic, two refined corners of a
# box, two fine blocks diagonal to each other with coarse blocks in the
# other quadrants, and in 3D one refined corner, whose fine blocks touch
# the coarse block across from it at a corner only.
function refined_forests(D; N=4)
    forests = Any[("nested", nested_forest(Val(D); N=N)),
                  ("nested periodic", nested_forest(Val(D); N=N,
                                                    periodic=ntuple(_ -> true, D))),
                  ("two corners", faces_forest(ntuple(_ -> :outer, D); N=N))]
    if D >= 2
        f = Forest(ntuple(d -> d <= 2 ? 2 : 1, D); N=N)
        refine!(f, filter(k -> block_extent(f, k)[1][1] == block_extent(f, k)[2][1],
                          f.leaves))
        balance!(f)
        push!(forests, ("diagonal", f))
    end
    if D == 3
        f = Forest((2, 2, 2); N=N)
        refine!(f, [f.leaves[1]])
        balance!(f)
        push!(forests, ("one corner", f))
    end
    return forests
end

# The primary of the point at exact position `p` among its own copies on
# a mesh of several levels, or `nothing` where it has none: the finest
# level whose closed boxes contain the point — a coarse copy along a
# cell-like dimension has no coincident fine point, and takes the
# restriction of the fine cells about it — then the greatest own index,
# which on that level is the greatest class tuple.
function refined_primary(fs, p, copies)
    forest = fs.forest
    inside(k) = all(1:length(p)) do d
        lo, hi = leafbox(forest, k)
        any(x -> lo[d] <= x <= hi[d],
            forest.periodic[d] ? (p[d], p[d] + forest.roots[d]) : (p[d],))
    end
    finest = maximum(level(k) for k in forest.leaves if inside(k))
    own = [(idx, b) for (b, idx, o) in copies if o && level(blockkey(fs, b)) == finest]
    isempty(own) && return nothing
    idx, b = maximum(own)
    return (b, idx)
end

@testset "Whether a copy is the primary is one bit a piece across levels: D=$D" for
        D in (1, 2, 3)
    # Guards the mask's claim that the answer is the same along a piece
    # when finer blocks cover part of what a block shares: every own point
    # of every block is marked primary exactly when it is its point's
    # finest copy, high along every dimension the point is shared in on
    # that level — at a hanging node, a mid-plane, a coincident point, and
    # beside a diagonal or corner-only finer block.
    for (name, forest) in refined_forests(D), C in closed_centerings(D)
        fs = FieldSet(forest, 1; G=1, centering=C)
        mask = primary_mask(fs)
        ok, shared = true, 0
        for (p, copies) in closed_point_groups(fs)
            any(x -> x[3], copies) || continue
            count(x -> x[3], copies) > 1 && (shared += 1)
            primary = refined_primary(fs, p, copies)
            for (b, idx, own) in copies
                own || continue
                I = ntuple(d -> idx[d] - fs.G[d], D)
                isprimary(mask, I..., b) == ((b, idx) == primary) || (ok = false)
            end
        end
        @test ok
        @test shared > 0
    end
end

@testset "A count over a refined set is its number of distinct points: D=$D" for
        D in (1, 2, 3)
    # Guards a count with `primaries = true` that takes a coincident point
    # at its coarse copy as well as its fine one, or misses a hanging node.
    # Along a cell-like dimension a coarse face value beside finer blocks
    # is no point of its own: it is the restriction of the fine ones.
    for (name, forest) in refined_forests(D), C in closed_centerings(D)
        fs = FieldSet(forest, 1; G=1, centering=C)
        distinct = count(pairs(closed_point_groups(fs))) do (p, copies)
            any(x -> x[3], copies) && refined_primary(fs, p, copies) !== nothing
        end
        @test mesh_mapreduce(x -> 1, +, 0, fs; vars=1, primaries=true) == distinct
        @test mesh_fold((acc, p) -> acc + 1, +, 0, fs; vars=1, primaries=true) == distinct
    end
end

@testset "A constant is its own norm on a refined mesh, a unit its volume: D=$D" for
        D in (1, 2, 3)
    # Guards the trapezoid's weights across levels: each leaf integrates
    # its own closed box, so the sum over the leaves is the domain's
    # composite rule with no mask.
    for (name, forest) in refined_forests(D), C in closed_centerings(D)
        fs = FieldSet(forest, 1; G=1, centering=C)
        u = statevector(fs)
        fill!(u, 3.0)
        @test volume_weighted_norm(fs, u) ≈ 3.0 rtol = 4eps()
        @test volume_weighted_norm(fs, u; p=1) ≈ 3.0 rtol = 4eps()
        fs.work .= 1.0
        @test total_mass(fs, 1) ≈ prod(forest.roots) rtol = 4eps()
    end
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
    # simulated ranks must give the serial ghosts bit for bit, on meshes of
    # several levels too, where stage 0 restricts.
    for (name, serial) in vcat(closed_forests(D; N=8), refined_forests(D; N=8)),
        C in closed_centerings(D)
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
    # Guards the flat launch a device takes over the new pieces, stage 0's
    # restrictions and the halved ranges of phases 1 and 2 included.
    for (name, forest) in vcat(closed_forests(D), refined_forests(D)),
        C in closed_centerings(D)
        fs = FieldSet(forest, 2; G=1, centering=C)
        sched = GhostSchedule(fs, closed_ops(1))
        @test !isempty(sched.unify)
        data = randn(Xoshiro(7), size(fs.work))
        for g in vcat(sched.unify, sched.phase1, sched.phase2...)
            fills = map((false, true)) do flat
                work = copy(data)
                run_group!(work, work, g, fs.nvars, TreeAMR.CPU(); flat=flat)
                bits.(work)
            end
            @test fills[1] == fills[2]
        end
    end
end

# The values the own copies of a point hold after a fill, by position: a
# copy of a point the primary holds too is its value bit for bit; along a
# cell-like dimension beside a finer block a coarse face value has no
# coincident point, and is the mean of the fine values at the cell
# centres about it (the order-2 restriction of `closed_ops`), to rounding.
# Ghosts are not compared: across a coarse-fine face a ghost at a point
# some block owns is the coarse source's interpolant there, and equals
# the owner's value only to the scheme's order. `average = false` skips
# the means, for a restriction of higher order. Returns the count of
# points checked each way and whether all held.
function refined_copies_ok(fs::FieldSet{T,D}, groups; average::Bool=true) where {T,D}
    c = staggers(fs)
    exact, averaged, ok = 0, 0, true
    for (p, copies) in groups
        any(x -> x[3], copies) || continue
        primary = refined_primary(fs, p, copies)
        if primary !== nothing
            pb, pidx = primary
            for (b, idx, own) in copies, v in 1:fs.nvars
                own || continue
                bits(fs.work[idx..., v, b]) == bits(fs.work[pidx..., v, pb]) || (ok = false)
            end
            exact += 1
            continue
        end
        average || continue
        # The fine level's cell centres about `p` along every cell-like
        # dimension, a quarter of the coarse spacing either side.
        b0, idx0, _ = first(x for x in copies if x[3])
        hf = 1 // (fs.forest.N * 2^(level(blockkey(fs, b0)) + 1))
        offsets = Iterators.product(ntuple(d -> c[d] == 1 ? (0 // 1,) :
                                                (-hf // 2, hf // 2), D)...)
        fine = [get(groups, ntuple(d -> p[d] + o[d], D), nothing) for o in offsets]
        any(isnothing, fine) && (ok = false; continue)
        for v in 1:fs.nvars
            want = sum(fine) do cs
                b, idx, _ = first(x for x in cs if x[3])
                fs.work[idx..., v, b]
            end / length(fine)
            for (b, idx, own) in copies
                own || continue
                isapprox(fs.work[idx..., v, b], want; rtol=8eps(T), atol=8eps(T)) ||
                    (ok = false)
            end
        end
        averaged += 1
    end
    return exact, averaged, ok
end

@testset "Every copy is its primary's after a fill on a refined mesh: D=$D" for
        D in (1, 2, 3)
    # Guards stage 0's restrictions: a coarse copy of a point on a
    # coarse-fine face must take the finest copy's value — injected, bit
    # for bit, where the point is the fine block's too, and the mean of the
    # fine values about it along a cell-like dimension — and a mid-plane
    # or a hanging node its fine primary's; and every own point of every
    # block is written only by its primary.
    hook = boundary_by_coordinates((x, v) -> sum(x) + v)
    for (name, forest) in refined_forests(D; N=4), C in closed_centerings(D), G in (1, 2)
        fs = FieldSet(forest, 2; G=G, centering=C)
        fs.work .= randn(Xoshiro(2300 + D), size(fs.work))
        fill_ghosts!(fs, GhostSchedule(fs, closed_ops(G)); boundary=hook)
        exact, averaged, ok = refined_copies_ok(fs, closed_point_groups(fs))
        @test ok
        @test exact > 0
        # Not vacuous where a coarse-fine face lies across a cell-like
        # dimension of the set (the diagonal mesh has none normal to z).
        @test averaged > 0 || all(==(1), staggers(C)) || D == 1 ||
              (name == "diagonal" && staggers(C)[3] == 1)
    end
end

@testset "N = 2G fills a refined mesh, and its copies agree: D=$D" for D in (1, 2, 3)
    # Guards the bound itself (M22 step 3): at `N = 2G` a coarse ghost
    # `G` planes past a coarse-fine face is injected from the fine block's
    # far plane, its last own point, and a finer neighbour spans exactly
    # the ghost region. Nothing ran there before the bound was relaxed.
    hook = boundary_by_coordinates((x, v) -> sum(x) + v)
    for (G, N, ops) in ((1, 2, Operators(prolongation=2, restriction=2)),
                        (2, 4, Operators(prolongation=4, restriction=2)))
        for (name, forest) in refined_forests(D; N=N), C in closed_centerings(D)
            any(forest.periodic) && continue        # the data is linear, not periodic
            fs = FieldSet(forest, 1; G=G, centering=C)
            data = (x, v) -> 1.0 + sum(d -> d * x[d], 1:D)
            fill_by_coordinates!(data, fs)
            fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=boundary_by_coordinates(data))
            _, _, ok = refined_copies_ok(fs, closed_point_groups(fs))
            @test ok
            # Linear data, which both orders reproduce, in every ghost.
            worst = 0.0
            for b in 1:nblocks(fs), I in CartesianIndices(size(fs.work)[1:D])
                x = coordinates(fs, b, Tuple(I))
                worst = max(worst, abs(fs.work[I, 1, b] - data(x, 1)))
            end
            @test worst < 1e-12
        end
    end
end

@testset "An edge-centered set's copies around an edge beside a diagonal finer block are one number" begin
    # Guards rule 1 at an edge only a diagonal finer block touches, which
    # the interface restriction left out: every coarse copy of a point on
    # the edge takes the fine primary's value, so the four blocks around
    # the edge hold one electromotive force there.
    forest = Forest((2, 2, 1); N=8)
    refine!(forest, filter(k -> block_extent(forest, k)[1][1] ==
                                block_extent(forest, k)[2][1], forest.leaves))
    balance!(forest)
    fs = FieldSet(forest, 1; G=0, centering=edgecentered(3, 3))
    fs.work .= randn(Xoshiro(23), size(fs.work))
    unify!(fs, GhostSchedule(fs, closed_ops(0)))
    groups = closed_point_groups(fs)
    _, averaged, ok = refined_copies_ok(fs, groups)
    @test ok
    @test averaged > 0
    # The central edge, x = y = 1: its coarse copies are one number.
    central = [copies for (p, copies) in groups if p[1] == 1 && p[2] == 1 &&
               count(x -> x[3] && level(blockkey(fs, x[1])) == 0, copies) >= 2]
    @test !isempty(central)
    @test all(central) do copies
        vals = [fs.work[idx..., 1, b] for (b, idx, own) in copies
                if own && level(blockkey(fs, b)) == 0]
        all(v -> bits(v) == bits(first(vals)), vals)
    end
end

# The face kinds of a box symmetric under its point reflection: every one
# alike, and in 2D and 3D a mix of the three.
mirror_boxes(D) = vcat([ntuple(_ -> k, D) for k in (:outer, :reflect_both, :outflow_both)],
                       D >= 2 ? [ntuple(d -> (:reflect_both, :outflow_both, :outer)[d], D)] :
                       [])

# One variable even across every reflecting face and one odd there.
mirror_parity(kinds) = [ntuple(_ -> EvenParity, length(kinds)),
                        ntuple(d -> kinds[d] === :reflect_both ? OddParity : EvenParity,
                               length(kinds))]

@testset "A closed box fills mirror-symmetrically under every face kind: D=$D" for
        D in (1, 2, 3)
    # Guards a rule that depends on which way an axis runs (M22): under
    # half-open ownership a refined box was asymmetric at 4.7e-5, and a
    # reflecting or outflow box needed `walls = :symmetric`. Point-symmetric
    # data on a box, uniform and refined at two opposite corners, outer,
    # reflecting, outflow and mixed faces, the data as hook, order-4
    # operators, bounded and not: every value must equal its mirror
    # image's to rounding, with no keyword. The two sides sum a window in
    # mirrored order and form their positions as `lo + k h` from opposite
    # ends, so the bound is a few roundings of the data's scale, `64 eps ·
    # max |f|` (an order-4 window's weights sum in absolute value to 1.25
    # a dimension, 2 in 3D); what is exact is asserted separately: every
    # copy is its primary's, bit for bit, and an odd variable is zero on
    # its mirror plane.
    L = D == 1 ? 3 : 2
    f = point_symmetric(D, L)
    for kinds in mirror_boxes(D), refined in (false, true), bounded in (false, true),
        C in closed_centerings(D)
        ops = Operators(prolongation=4, restriction=4, bounded=bounded)
        forest = faces_forest(kinds; N=8, refined=refined)
        fs = FieldSet(forest, 2; G=ghosts_for(C, 4), centering=C,
                      parity=any(==(:reflect_both), kinds) ? mirror_parity(kinds) : nothing)
        fill_ghosts!(fill_by_coordinates!(f, fs), GhostSchedule(fs, ops);
                     boundary=boundary_by_coordinates(f))
        err, n = point_mirror_error(fs, L)
        scale = maximum(abs, fs.work)
        @test n > 0
        @test err <= 64 * eps() * scale
        showerrors() && @info "mirror" D kinds refined bounded C err scale
        exact, _, ok = refined_copies_ok(fs, closed_point_groups(fs); average=false)
        @test ok && exact > 0
    end
end

@testset "A closed box stays mirror-symmetric through refining and coarsening regrids: D=$D" for
        D in (1, 2, 3)
    # Guards a regrid transfer with a direction-dependent rule (walls_tests'
    # claim, on closed blocks): a regrid that refines every block keeps the
    # box uniform, and the fill after it symmetric to rounding, a
    # prolongation summing its window in mirrored order on the two sides;
    # then a regrid that coarsens every block again, and from the box
    # refined at two opposite corners one that coarsens the corners and
    # refines the rest, at once, each followed by a fill. At `N = 8` and at
    # `N = 2G`, where a coarsened block's two halves meet at its mid-plane
    # and the mirrored fill's restriction takes the closed half. The
    # regrid's own restriction reads a child's own points only, so it
    # never mirrors.
    L = D == 1 ? 3 : 2
    f = point_symmetric(D, L)
    ops = Operators(prolongation=4, restriction=4)
    for kinds in mirror_boxes(D), C in closed_centerings(D), small in (false, true),
        refined in (false, true)
        G = ghosts_for(C, 4)
        # `N = 2G` for the vertex set alone: a cell-like dimension's
        # restriction of order 4 needs more than `2G` fine cells.
        small && C != vertexcentered(D) && continue
        N = small ? 2 * maximum(G) : 8
        forest = faces_forest(kinds; N=N, refined=refined)
        fs = FieldSet(forest, 2; G=G, centering=C,
                      parity=any(==(:reflect_both), kinds) ? mirror_parity(kinds) : nothing)
        hook = boundary_by_coordinates(f)
        fill_ghosts!(fill_by_coordinates!(f, fs), GhostSchedule(fs, ops); boundary=hook)
        rounds = refined ? (k -> level(k) == 0 ? Refine : Coarsen,) :
                 (k -> Refine, k -> Coarsen)
        for flag in rounds
            @test regrid!(forest, fs => GhostSchedule(fs, ops); flags=map(flag, forest.leaves),
                          boundary=hook)
            fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=hook)
            err, n = point_mirror_error(fs, L)
            @test n > 0 && err <= 64 * eps() * maximum(abs, fs.work)
        end
    end
end

# Every own point of `fs` on a reflecting wall plane of a vertex-like
# dimension, as `(block, stored index)`.
function mirror_plane_points(fs::FieldSet{T,D}) where {T,D}
    forest = fs.forest
    c = staggers(fs)
    out = Tuple{Int,NTuple{D,Int}}[]
    for b in 1:nblocks(fs), I in CartesianIndices(ntuple(d -> forest.N + c[d], D))
        idx = ntuple(d -> I[d] + fs.G[d], D)
        k = blockkey(fs, b)
        onwall = any(1:D) do d
            c[d] == 1 || return false
            ext = block_extent(forest, k)[d]
            (I[d] == 1 && ext[1] == forest.extents[d][1] && facekind(forest, d, 1) ==
                                                            ReflectingFace) ||
                (I[d] == forest.N + 1 && ext[2] == forest.extents[d][2] &&
                 facekind(forest, d, 2) == ReflectingFace)
        end
        onwall && push!(out, (b, idx))
    end
    return out
end

@testset "An odd variable is zero on its mirror plane at every copy: D=$D" for D in (1, 2, 3)
    # Guards stage P (M22 rule 3): a point on a reflecting wall plane is
    # its own image, and an odd variable must vanish there at every copy,
    # after a fill and after `unify!` on a state, at a coarse-fine line on
    # the wall and where two reflecting walls meet; an even one is not
    # written by it, and keeps the bits its primary holds.
    kinds = ntuple(d -> d == 2 ? :outer : :reflect_both, D)
    for refined in (false, true), C in closed_centerings(D)
        forest = faces_forest(kinds; N=8, refined=refined)
        fs = FieldSet(forest, 2; G=1, centering=C, parity=mirror_parity(kinds))
        data = randn(Xoshiro(41 + D), size(fs.work))
        fs.work .= data
        sched = GhostSchedule(fs, closed_ops(1))
        fill_ghosts!(fs, sched; boundary=boundary_by_coordinates((x, v) -> 0.0))
        walls = mirror_plane_points(fs)
        @test !isempty(walls) || all(d -> staggers(C)[d] == 0 || kinds[d] === :outer, 1:D)
        @test all(((b, idx),) -> fs.work[idx..., 2, b] == 0, walls)
        mask = primary_mask(fs)
        @test all(walls) do (b, idx)
            I = ntuple(d -> idx[d] - fs.G[d], D)
            !isprimary(mask, I..., b) || bits(fs.work[idx..., 1, b]) == bits(data[idx..., 1, b])
        end
        u = statevector(fs)
        u .= randn(Xoshiro(43 + D), length(u))
        unify!(u, fs, sched)
        A = statearray(u, fs)
        @test all(((b, idx),) -> A[ntuple(d -> idx[d] - fs.G[d], D)..., 2, b] == 0, walls)
    end
end

@testset "Stage P writes primaries only, and nothing for even variables: D=$D" for
        D in (1, 2, 3)
    # Guards stage P's targets: a projection written at a secondary would
    # be overwritten by stage 0, and one at a point whose variables are
    # all even would cost a launch for nothing. Its groups copy a block's
    # own points onto themselves.
    kinds = ntuple(_ -> :reflect_both, D)
    forest = faces_forest(kinds; N=8)
    for C in closed_centerings(D)
        even = FieldSet(forest, 2; G=1, centering=C,
                        parity=[ntuple(_ -> EvenParity, D), ntuple(_ -> EvenParity, D)])
        @test isempty(GhostSchedule(even, closed_ops(1)).selfimage)
        fs = FieldSet(forest, 2; G=1, centering=C, parity=mirror_parity(kinds))
        sched = GhostSchedule(fs, closed_ops(1))
        @test !isempty(sched.selfimage)
        @test length(UnifySchedule(fs).selfimage) == length(sched.selfimage)
        mask = primary_mask(fs)
        @test all(sched.selfimage) do g
            g.targetblocks == g.sourceblocks || return false
            first = ntuple(d -> g.stencils[d].targetfirst, D)
            all(1:ntransfers(g)) do t
                b = g.targetblocks[t]
                all(CartesianIndices(TreeAMR.boxsize(g))) do off
                    I = ntuple(d -> first[d] + off[d] - 1 - fs.G[d], D)
                    isprimary(mask, I..., b)
                end
            end
        end
    end
end

@testset "A schedule refuses a set with other parities than it was built for" begin
    # Guards stage P's parities: a schedule built for even variables has no
    # transfer on a reflecting wall plane, and handed a set whose variable
    # is odd there would leave its wall value standing.
    forest = Forest((2, 2); N=8, reflecting=((true, true), (true, true)))
    even = [(EvenParity, EvenParity)]
    odd = [(OddParity, EvenParity)]
    for C in (vertexcentered(2), facecentered(2, 1))
        a = FieldSet(forest, 1; G=C === vertexcentered(2) ? 1 : 0, centering=C, parity=even)
        b = FieldSet(forest, 1; G=a.G, centering=C, parity=odd)
        u = statevector(b)
        us = UnifySchedule(a)
        @test_throws "parities" unify!(b, us)
        @test_throws "parities" unify!(u, b, us)
        if C === vertexcentered(2)
            gs = GhostSchedule(a, closed_ops(1))
            @test_throws "parities" fill_ghosts!(b, gs)
            @test_throws "parities" unify!(b, gs)
            @test_throws "parities" unify!(u, b, gs)
        end
        # Without reflecting faces a schedule serves every set of its layout.
        plain = Forest((2, 2); N=8)
        p1 = FieldSet(plain, 1; G=a.G, centering=C, parity=even)
        p2 = FieldSet(plain, 1; G=a.G, centering=C, parity=odd)
        @test unify!(p2, UnifySchedule(p1)) === p2
    end
end

@testset "A bitant's mass is half the full domain's, and its norm the same: D=$D" for
        D in (1, 2, 3)
    # Guards the trapezoid's weights at a mirror plane: half weight on the
    # wall plane, which the full domain counts once between two blocks.
    # Exact on data whose values and partial sums are dyadic; to a few ulp
    # on any other, since the two halves of the full domain sum their
    # points in mirrored order.
    N = 8
    half = Forest(ntuple(d -> d == 1 ? 2 : 2, D); N=N,
                  extents=ntuple(d -> d == 1 ? (0.0, 2.0) : (0.0, 2.0), D),
                  reflecting=ntuple(d -> d == 1 ? (true, false) : (false, false), D))
    full = Forest(ntuple(d -> d == 1 ? 4 : 2, D); N=N,
                  extents=ntuple(d -> d == 1 ? (-2.0, 2.0) : (0.0, 2.0), D))
    for (g, exact) in (((x, v) -> 1 + x[1]^2 / 4 + sum(x) / 8 - x[1] / 8, true),
                       ((x, v) -> cos(x[1]) * (1 + sin(sum(x) - x[1])), false))
        for C in closed_centerings(D)
            a = FieldSet(half, 1; G=1, centering=C, parity=[ntuple(_ -> EvenParity, D)])
            b = FieldSet(full, 1; G=1, centering=C)
            fill_by_coordinates!(g, a)
            fill_by_coordinates!(g, b)
            ua, ub = statevector(a), statevector(b)
            gather!(ua, a)
            gather!(ub, b)
            if exact
                @test 2 * total_mass(a, 1) == total_mass(b, 1)
                @test volume_weighted_norm(a, ua) == volume_weighted_norm(b, ub)
            else
                @test 2 * total_mass(a, 1) ≈ total_mass(b, 1) rtol = 8eps()
                @test volume_weighted_norm(a, ua) ≈ volume_weighted_norm(b, ub) rtol = 8eps()
            end
        end
    end
end

@testset "A flux normal to a mirror plane is zero there after unify!: D=$D" for D in (1, 2, 3)
    # Guards stage P in a `UnifySchedule`: every field set over a forest
    # with reflecting faces declares its parities, so a mass flux normal to
    # the wall, odd across it, is projected to zero there, and an even one
    # keeps its bits.
    kinds = ntuple(_ -> :reflect_both, D)
    forest = faces_forest(kinds; N=8)
    fs = FieldSet(forest, 2; G=0, centering=facecentered(D, 1),
                  parity=[ntuple(d -> d == 1 ? OddParity : EvenParity, D),
                          ntuple(_ -> EvenParity, D)])
    data = randn(Xoshiro(7 + D), size(fs.work))
    fs.work .= data
    unify!(fs, UnifySchedule(fs))
    walls = mirror_plane_points(fs)
    @test !isempty(walls)
    @test all(((b, idx),) -> fs.work[idx..., 1, b] == 0, walls)
    mask = primary_mask(fs)
    @test all(walls) do (b, idx)
        I = ntuple(d -> idx[d] - fs.G[d], D)
        !isprimary(mask, I..., b) || bits(fs.work[idx..., 2, b]) == bits(data[idx..., 2, b])
    end
end

@testset "N = 2G mirrors a ghost onto the mid-plane of a finer source: D=$D" for D in (2, 3)
    # Guards the mirrored restriction at the bound: beyond a high
    # reflecting wall the outermost ghost of a coarse block mirrors onto
    # the mid-plane of the finer blocks across its tangential face, which
    # the upper child holds at its low plane. Data even across the wall
    # and quadratic, so the mirror, the injection and an order-4
    # prolongation reproduce it: every stored value must be the data at
    # its own position, against the position oracle.
    for (G, N) in ((1, 2), (2, 4))
        forest = Forest(ntuple(d -> d <= 2 ? 2 : 1, D); N=N,
                        reflecting=ntuple(d -> d == 1 ? (false, true) : (false, false), D))
        refine!(forest, filter(k -> all(d -> block_extent(forest, k)[d][1] >= (d <= 2 ? 1 : 0),
                                        1:D), forest.leaves))
        balance!(forest)
        g = (x, v) -> 1 + (x[1] - 2)^2 + x[2] / 3 + (D == 3 ? x[3] / 5 : 0.0)
        # The mirror dimension vertex-like, where the mid-plane is; order 4
        # along a cell-like dimension needs two ghosts, and the order-2
        # restriction there is exact for the linear data only.
        for C in (G == 1 ? [vertexcentered(D)] :
                  filter(C -> C[1] === :vertex, closed_centerings(D)))
            fs = FieldSet(forest, 1; G=G, centering=C, parity=[ntuple(_ -> EvenParity, D)])
            fill!(fs.work, NaN)
            fill_by_coordinates!(g, fs)
            fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=4, restriction=2));
                         boundary=boundary_by_coordinates(g))
            worst = 0.0
            for b in 1:nblocks(fs), I in CartesianIndices(size(fs.work)[1:D])
                worst = max(worst, abs(fs.work[I, 1, b] - g(coordinates(fs, b, Tuple(I)), 1)))
            end
            @test worst < 1e-12
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
    walled = Forest((2, 2); N=8, reflecting=((true, false), (false, false)))
    uniform = Forest((2, 2); N=8)
    vs = FieldSet(uniform, 1; G=1, centering=vertexcentered(2))
    # A regrid of a vertex-like set is back (M22 step 6): a mesh kept as it is.
    @test !regrid!(uniform, vs => GhostSchedule(vs, ops); flags=fill(Keep, 4))
    quad = Forest((2, 2); N=8, rotating=(1, 2))
    @test_throws "M22 step 7" firing_boxes((w, i, b, x) -> true,
                                           FieldSet(quad, 1; G=1,
                                                    centering=vertexcentered(2),
                                                    rotation=(1,)))
    @test_throws "M22 step 7" GhostSchedule(FieldSet(quad, 1; G=1,
                                                     centering=vertexcentered(2),
                                                     rotation=(1,)), ops)
    # `walls` is gone (M22 step 4): every wall is symmetric on closed blocks.
    @test_throws MethodError FieldSet(uniform, 1; G=1, centering=vertexcentered(2),
                                      walls=:symmetric)
    # Cell-centered sets are not affected.
    @test GhostSchedule(FieldSet(two, 1; G=1), ops) isa GhostSchedule
    @test GhostSchedule(FieldSet(walled, 1; G=1, parity=[EvenParity]), ops) isa GhostSchedule
    # The keywords closed blocks replace say what replaced them.
    @test_throws "own range" map_blocks!((args...) -> nothing, vs; closed=true)
    @test_throws "appendix" statearray(statevector(vs), vs; appendix=true)
    @test_throws "wall must be" CellBoundary((x, v, δ) -> 0.0; wall=:symmetric)
end

@testset "Flagging over the closed box is mirror-symmetric: D=$D" for D in (1, 2, 3)
    # Guards the half-open bias in flagging (M22 step 6): a vertex-like
    # dimension's firing nodes give the cells touching them, so a pattern
    # mirror-symmetric across the domain's mid-plane gives mirror-image
    # boxes, and through the buffer a mirror-symmetric forest. A clamp of
    # node `N + 1` to cell `N` would recruit the high neighbour of a node
    # at `I = N` and not the low one of its image at `I = 2`.
    N = 8
    @test all(lo -> all(hi -> touching_cells(lo, hi, N, 1) ==
                              (N + 1) .- reverse(touching_cells(N + 2 - hi, N + 2 - lo, N, 1)),
                        lo:(N + 1)), 1:(N + 1))
    @test touching_cells(3, 5, N, 0) == 3:5
    mirror(e) = (1 - e[2], 1 - e[1])               # about x₁ = 1/2
    shape(forest) = Set(map(k -> (mirror(block_extent(forest, k)[1]),
                                  Base.tail(block_extent(forest, k))...),
                            forest.leaves)) ==
                    Set(map(k -> block_extent(forest, k), forest.leaves))
    # Dyadic positions and a pattern in `(x₁ − 1/2)²`, exact at every node,
    # that fires near the mid-plane and in a ring off it.
    f(x) = (x[1] - 0.5)^2 + sum(d -> (x[d] - 0.25)^2, 2:D; init=0.0)
    fires(w, idx, b, x) = (y = w[idx..., 1, b]; y < 1 / 64 || 9 / 64 < y < 10 / 64)
    for C in (vertexcentered(D), facecentered(D, 1)), buffer in (0, 1, 2)
        forest = Forest(ntuple(_ -> 4, D); N=N, extents=ntuple(_ -> (0.0, 1.0), D))
        for round in 1:2
            fs = FieldSet(forest, 1; G=1, centering=C)
            fill_by_coordinates!((x, v) -> f(x), fs)
            boxes = firing_boxes(fires, fs)
            @test any(g -> g[1] > 0, boxes) && any(g -> g[1] == 0, boxes)
            # Each block's box is the mirror image of its image block's.
            ext = [block_extent(forest, k) for k in forest.leaves]
            for b in eachindex(ext)
                m = findfirst(==((mirror(ext[b][1]), Base.tail(ext[b])...)), ext)
                (nb, box), (nm, mbox) = boxes[b], boxes[m]
                @test nb == nm
                nb == 0 || @test box == ((N + 1) .- reverse(mbox[1]), Base.tail(mbox)...)
            end
            flags = [n == 0 ? Keep : (level(forest.leaves[b]) < 2 ? Refine : Keep, box)
                     for (b, (n, box)) in enumerate(boxes)]
            regrid!(forest, (fs => nothing,); flags=flags, buffer=buffer)
            @test shape(forest)
        end
        @test maxlevel(forest) == 2
    end
end

# The kept-over-created pass of `regrid!` (M22 step 6): a forest refined
# over the high half along dimension 1, periodic, whose low half the
# regrid refines, so that every created fine block meets kept fine blocks
# at x₁ = 1, where the created block's high plane is the primary by the
# class rule, and at x₁ = 0, where the kept block's is.
function half_refined(D; comm=nothing)
    forest = Forest(ntuple(_ -> 2, D); N=8, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 2.0), D), comm=comm)
    refine!(forest, filter(k -> block_extent(forest, k)[1][1] >= 1, forest.leaves))
    balance!(forest)
    return forest
end

# Per point of `groups` (`closed_point_groups` of the set after a regrid),
# the own copies at the finest level holding it, split into those of kept
# blocks — with their old block index — and of created ones.
function kept_created(fs, groups, oldleaves)
    out = []
    for (p, copies) in groups
        own = [(b, idx) for (b, idx, o) in copies if o]
        finest = maximum(x -> level(blockkey(fs, x[1])), own)
        atf = filter(x -> level(blockkey(fs, x[1])) == finest, own)
        kept = [(b, idx, findfirst(==(blockkey(fs, b)), oldleaves)) for (b, idx) in atf
                if blockkey(fs, b) in oldleaves]
        created = [(b, idx) for (b, idx) in atf if !(blockkey(fs, b) in oldleaves)]
        isempty(kept) || isempty(created) || push!(out, (kept, created))
    end
    return out
end

@testset "A regrid gives a created copy the kept block's value, before any fill: D=$D" for
        D in (1, 2, 3)
    # Guards the pass that keeps evolved values: without it a created fine
    # block's prolongated copy of a plane it shares with a kept fine block
    # would be the primary at x₁ = 1, and the next fill's stage 0 would
    # carry the interpolated value over the evolved one. After `regrid!`
    # and before any fill, every created copy of a point a kept block
    # holds at the finest level is the kept block's old value bit for bit;
    # after the first fill the kept blocks' values there are unchanged;
    # and over 2 and 3 simulated ranks, with kept and created blocks on
    # different ranks, the regrid is the serial one bit for bit.
    ncross = 0
    for C in closed_centerings(D), bounded in (false, true)
        ops = Operators(prolongation=4, restriction=4, bounded=bounded)
        G = ghosts_for(C, 4)
        forest = half_refined(D)
        fs = FieldSet(forest, 2; G=G, centering=C)
        data = randn(Xoshiro(60 + D), size(fs.work))
        fs.work .= data
        oldleaves = copy(forest.leaves)
        flags = [level(k) == 0 ? Refine : Keep for k in forest.leaves]
        @test regrid!(forest, fs => GhostSchedule(fs, ops); flags=flags)
        # The old storage as the regrid moved it: filled, so unified.
        ref = FieldSet(half_refined(D), 2; G=G, centering=C)
        ref.work .= data
        fill_ghosts!(ref, GhostSchedule(ref, ops))
        old = ref.work
        moved = copy(fs.work)
        pairs = kept_created(fs, closed_point_groups(fs), oldleaves)
        mask = primary_mask(fs)
        nprimary = 0
        same = all(pairs) do (kept, created)
            b0, i0, o0 = first(kept)
            all(v -> all(((b, idx, o),) -> bits(old[idx..., v, o]) == bits(old[i0..., v, o0]),
                         kept) &&
                     all(((b, idx),) -> bits(moved[idx..., v, b]) == bits(old[i0..., v, o0]),
                         created), 1:2)
        end
        nprimary = count(pairs) do (kept, created)
            any(((b, idx),) -> isprimary(mask, (idx .- G)..., b), created)
        end
        # Only a set vertex-like along dimension 1 shares the planes x₁ = 0
        # and 1 between kept and created blocks of one level.
        if staggers(C)[1] == 1
            @test !isempty(pairs) && same
            @test nprimary > 0
        else
            @test isempty(pairs)
        end
        fill_ghosts!(fs, GhostSchedule(fs, ops))
        @test all(pairs) do (kept, created)
            all(((b, idx, o),) -> all(v -> bits(fs.work[idx..., v, b]) ==
                                           bits(old[idx..., v, o]), 1:2), kept)
        end
        # The same regrid over simulated ranks, before any fill.
        for P in (2, 3)
            n = nleaves(forest)
            kp = TreeAMR.regrid_kept_pairs(TreeAMR.leaves_view(forest, forest.leaves),
                                           oldleaves, forest.leaves, staggers(fs))
            ncross += count(((t, s),) -> equalsplit_part(length(oldleaves), P, Int(s)) !=
                                         equalsplit_part(n, P, Int(t)),
                            [(t, s) for (ts, ss) in values(kp) for (t, s) in zip(ts, ss)])
            comms = gather_ranks(P)
            results = on_ranks(P) do r
                rf = half_refined(D; comm=comms[r])
                owned = blockrange(rf)
                rfs = FieldSet(rf, 2; G=G, centering=C)
                rfs.work .= data[ntuple(_ -> :, D + 1)..., owned]
                regrid!(rf, rfs => GhostSchedule(rfs, ops); flags=flags[owned])
                (blockrange(rf), copy(rfs.work))
            end
            @test all(((range, work),) -> bitwise_equal(work,
                                                        moved[ntuple(_ -> :, D + 1)...,
                                                              range]), results)
        end
    end
    @test ncross > 0
end

@testset "Two created blocks of different parents agree on their common plane: D=$D" for
        D in (1, 2, 3)
    # Guards the claim the kept-over-created pass leaves out: two blocks
    # created by refinement from different parents take a common plane
    # from each parent's copy of it, which the fill before the regrid
    # unified, through the same weights in the same order, so they agree
    # bit for bit before any fill — bounded or not, periodic and at
    # reflecting walls.
    for C in closed_centerings(D), bounded in (false, true),
        kinds in (ntuple(_ -> :periodic, D), ntuple(_ -> :reflect_both, D))
        ops = Operators(prolongation=4, restriction=4, bounded=bounded)
        forest = faces_forest(kinds; N=8, refined=false)
        fs = FieldSet(forest, 2; G=ghosts_for(C, 4), centering=C,
                      parity=kinds[1] === :reflect_both ? mirror_parity(kinds) : nothing)
        fs.work .= randn(Xoshiro(70 + D), size(fs.work))
        oldleaves = copy(forest.leaves)
        regrid!(forest, fs => GhostSchedule(fs, ops); flags=fill(Refine, nleaves(forest)))
        nshared = 0
        agree = all(closed_point_groups(fs)) do (p, copies)
            own = [(b, idx) for (b, idx, o) in copies if o]
            isempty(own) && return true                 # ghosts beyond a wall
            parents = unique(parentkey(blockkey(fs, b)) for (b, _) in own)
            length(parents) >= 2 && (nshared += 1)
            b0, i0 = first(own)
            all(((b, idx),) -> all(v -> bits(fs.work[idx..., v, b]) ==
                                        bits(fs.work[i0..., v, b0]), 1:2), own)
        end
        @test agree && nshared > 0
    end
end
