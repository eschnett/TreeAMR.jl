# M22 step 1: positions from integer coordinates. A point has an integer
# coordinate `k` along each dimension, counted from its patch's low face
# in its level's spacings, and every block that stores the point forms its
# position from `k` the same way (`nodeposition`), so that every copy of a
# point — a shared plane, a ghost, a coarse point under a fine one — is at
# one position bit for bit (CODE.md, "Positions" under "Centerings").

using MultiFloats: Float32x2

# A forest whose extents are not dyadic, so that a position formed from a
# block's own floating-point corner would differ from its neighbour's in
# the last bit: three roots of width 1/3 from -0.3, refined about a corner
# and balanced, so that it has three levels and every kind of coarse-fine
# face.
function position_forest(::Type{T}, D; N=8) where {T}
    forest = Forest{T}(ntuple(_ -> 3, D); N=N, extents=ntuple(_ -> (-0.3, 0.7), D))
    refine!(forest, filter(k -> all(iszero, root_position(forest, k.root)), forest.leaves))
    balance!(forest)
    refine!(forest, [first(forest.leaves)])
    balance!(forest)
    return forest
end

# A 1D forest refined toward a point to level `L`: with four roots of `N
# = 8` the integer coordinates of the finest level reach `2^(L+5)`,
# beyond what a `Float32` holds exactly from `L = 20` on.
function deep_forest(::Type{T}, L; N=8) where {T}
    forest = Forest{T}((4,); N=N, extents=((-0.3, 0.9),))
    x = 0.4123
    for _ in 1:L
        refine!(forest, [forest.leaves[locate_point(forest, (x,))]])
        balance!(forest)
    end
    return forest
end

# Two patches with their own boxes, neither dyadic, glued to nothing:
# each forms its positions from its own low corner.
patch_position_forest(::Type{T}; N=8) where {T} =
    Forest{T}([(roots=(2, 2), extents=((-0.3, 0.3), (0.1, 0.7))),
               (roots=(3, 2), extents=((1 / 3, 1 / 3 + 0.9), (-0.7, -0.1)))]; N=N)

# The exact place of stored index `idx` of block `b` along each dimension,
# as an integer in half-spacings of level `L` (the finest), or for an
# element set's interior node the triple that names it: equal keys are one
# point. A ghost of an element set sits at its block's face.
function position_key(fs, b, idx, L)
    forest = fs.forest
    k = blockkey(fs, b)
    B = TreeAMR.block_base(forest, k)
    s = 1 << (L - level(k))
    c = staggers(fs)
    P1 = fs.P + 1
    return (patch(forest, k), ntuple(length(idx)) do d
        I = idx[d] - fs.G[d]
        if fs.P == 0
            return (2 * (B[d] + I - 1) + 1 - c[d]) * s, 0
        end
        I <= 0 && return 2 * B[d] * s, 0
        q, i = divrem(I - 1, P1)
        i == 0 && return 2 * (B[d] + q * P1) * s, 0
        i == P1 - 1 && return 2 * (B[d] + (q + 1) * P1) * s, 0
        return 2 * (B[d] + q * P1) * s, (level(k), i)
    end)
end

# Every stored index of every block grouped by its point; returns the
# groups' positions and how many have copies on two levels.
function position_groups(fs)
    L = maxlevel(fs.forest)
    groups = Dict{Any,Vector{Any}}()
    levels = Dict{Any,Set{Int}}()
    D = length(fs.G)
    for b in 1:nblocks(fs), I in CartesianIndices(size(fs.work)[1:D])
        key = position_key(fs, b, Tuple(I), L)
        push!(get!(groups, key, Any[]), coordinates(fs, b, Tuple(I)))
        push!(get!(levels, key, Set{Int}()), level(blockkey(fs, b)))
    end
    return groups, count(k -> length(levels[k]) > 1, keys(levels))
end

# A field set of centering `C`, with elements of order `P` when `P > 0`.
position_set(forest, nvars, C, P; G) =
    P == 0 ? FieldSet(forest, nvars; G=G, centering=C) :
    FieldSet(forest, nvars; G=G, centering=C, P=P)

position_cases(D) = vcat([(cellcentered(D), 0), (vertexcentered(D), 0)],
                         D >= 2 ? [(facecentered(D, 1), 0)] : [],
                         D == 3 ? [(edgecentered(D, 2), 0)] : [],
                         [(elementcentered(D), P) for P in (1, 3)])

@testset "Every copy of a point is at one position, bit for bit: D=$D, $T" for
    D in (1, 2, 3), T in (Float64, Float32)
    # Guards a position formed from a block's own floating-point corner,
    # `origin + (I − 1)·h`: across a shared plane, in a ghost and between a
    # coarse point and the fine point on it, two blocks would put one
    # point at positions that differ in the last bit, and a right-hand
    # side reading the position would give the copies different values.
    forest = position_forest(T, D)
    for (C, P) in position_cases(D)
        fs = position_set(forest, 1, C, P; G=P == 0 ? 2 : 1)
        groups, crosslevel = position_groups(fs)
        @test all(g -> all(x -> x === first(g), g), values(groups))
        # The forest has what the claim is about: shared points, and copies
        # on two levels wherever two levels have points in common (along a
        # cell-like dimension no point of two levels coincides).
        @test count(g -> length(g) > 1, values(groups)) > 0
        @test (crosslevel > 0) == (P > 0 || all(==(1), staggers(fs)))
    end
end

# The exact place of stored index `idx` of block `b` along each dimension,
# as a `BigFloat` wide enough to hold it: the patch's low corner `lo` plus
# the point's offset times the spacing `h`, with `lo` and `h` the
# floating-point numbers they are. The offset is the integer `k` or `k +
# ½` — as `T` holds it, which past `2²⁴` in `Float32` it does not
# exactly, and no single rounding can then give the exact place — or for
# an element's interior node `T(B + q(P+1)) + oᵢ`, the sum `nodeposition`
# rounds before its `fma` (an offset that is not an integer cannot be
# added without a rounding of its own).
function exact_position(fs::FieldSet{T}, b, idx) where {T}
    forest = fs.forest
    k = blockkey(fs, b)
    B, lo = TreeAMR.block_base(forest, k), TreeAMR.block_low(T, forest, k)
    h = BigFloat(spacing(T, forest, k))
    c = staggers(fs)
    P1 = fs.P + 1
    nodes = fs.P == 0 ? nothing : TreeAMR.element_offsets(T, fs.P)
    return ntuple(length(idx)) do d
        I = idx[d] - fs.G[d]
        if fs.P == 0
            return BigFloat(lo[d]) + BigFloat(T(2 * (B[d] + I - 1) + 1 - c[d])) * (h / 2)
        end
        I <= 0 && return BigFloat(lo[d]) + BigFloat(T(B[d])) * h
        q, i = divrem(I - 1, P1)
        i == P1 - 1 && return BigFloat(lo[d]) + BigFloat(T(B[d] + (q + 1) * P1)) * h
        return BigFloat(lo[d]) + BigFloat(T(B[d] + q * P1) + nodes[i + 1]) * h
    end
end

@testset "Positions are correctly rounded: D=$D, $T" for D in (1, 2, 3),
                                                         T in (Float64, Float32)
    # Guards a position rounded twice, as `lo + k·h` formed by a
    # multiplication and then an addition is, or fused on one backend and
    # not on another: every stored position must be the exact `lo + k·h`
    # (`lo + (k + ½)·h` cell-like) rounded once to `T`, which is what an
    # explicit `fma` gives on the host and on every device alike. Near
    # `lo + k·h = 0` the twice-rounded form was off by up to 13 ulps on
    # CUDA, which fuses.
    forests = Any[position_forest(T, D)]
    D == 1 && push!(forests, deep_forest(T, 21))
    D == 2 && push!(forests, patch_position_forest(T))
    setprecision(BigFloat, 512) do
        for forest in forests, (C, P) in position_cases(D)
            fs = position_set(forest, 1, C, P; G=P == 0 ? 2 : 1)
            bad = 0
            for b in 1:nblocks(fs), I in CartesianIndices(size(fs.work)[1:D])
                x = coordinates(fs, b, Tuple(I))
                bad += x === map(T, exact_position(fs, b, Tuple(I))) ? 0 : 1
            end
            @test bad == 0
        end
    end
end

@testset "Copies agree where the integer coordinate is not exact in the type: $T" for
    T in (Float32, Float64, Float32x2)
    # Guards the argument that the copies need no exact `T(k)`: past
    # `2^24` a `Float32` rounds `k`, and the copies must still round it
    # alike, a coarse point's `T(k)·2h` being the fine one's `T(2k)·h`,
    # and an element's two face nodes the same integer.
    forest = deep_forest(T, 21)
    @test maxlevel(forest) == 21
    @test 4 * forest.N << 21 > 1 << 24
    for (C, P) in position_cases(1)
        fs = position_set(forest, 1, C, P; G=1)
        groups, crosslevel = position_groups(fs)
        @test all(g -> all(x -> x === first(g), g), values(groups))
        @test (crosslevel > 0) == (P > 0 || all(==(1), staggers(fs)))
    end
end

@testset "The kernels form coordinates' positions, bit for bit: D=$D, $T" for
    D in (1, 2, 3), T in (Float64, Float32)
    # Guards a kernel that forms a position by its own expression rather
    # than through `nodeposition`: the initial data, the folds, the
    # flagging sweep and the boundary hooks would put a point where
    # `coordinates` does not.
    forest = position_forest(T, D)
    for (C, P) in position_cases(D)
        # The flagging sweep (M22 step 6) on a refined mesh (step 3).
        vertexlike(C) && (pending(6, "kernels form coordinates' positions, $C, D=$D, $T");
                          continue)
        fs = position_set(forest, D, C, P; G=1)
        fill_by_coordinates!((x, v) -> x[v], fs)
        N = forest.N
        @test all(b -> all(I -> ntuple(v -> fs.work[(Tuple(I) .+ fs.G)..., v, b], D) ===
                                coordinates(fs, b, Tuple(I) .+ fs.G),
                           CartesianIndices(ntuple(_ -> N, D))), 1:nblocks(fs))
        mismatch(acc, p) = acc + (p.u === p.x ? 0 : 1)
        @test mesh_fold(mismatch, +, 0, fs) == 0
        @test sum(TreeAMR._block_fold(mismatch, +, 0, fs.work, fs, fs.G, 1:D, nothing,
                                      4)) == 0
        @test sum(first, firing_boxes((w, idx, b, x) ->
                                          x !== ntuple(d -> w[idx..., d, b], D), fs)) == 0
        # Every outer ghost the hook writes is its own position.
        fill_ghosts!(fs, GhostSchedule(fs, P == 0 ?
                                           Operators(prolongation=2, restriction=2) :
                                           Operators(family=ElementLocal));
                     boundary=CellBoundary((x, v, δ) -> x[v]))
        for r in GhostSchedule(fs, P == 0 ? Operators(prolongation=2, restriction=2) :
                                       Operators(family=ElementLocal)).boundaries
            b = Int(r.block)
            @test all(I -> ntuple(v -> fs.work[Tuple(I)..., v, b], D) ===
                           coordinates(fs, b, Tuple(I)), r.region)
        end
    end
end

@testset "On a uniform brick a fill of the positions leaves every ghost its own: D=$D" for
    D in (1, 2, 3)
    # Guards the shared plane and the same-level ghosts: a ghost is a copy
    # of the neighbour's own point, so with one position per point the
    # value copied is the ghost's own position, bit for bit, where the
    # neighbour's corner plus the block's width was not.
    for T in (Float64, Float32), C in (cellcentered(D), vertexcentered(D))
        forest = Forest{T}(ntuple(_ -> 3, D); N=6,
                           extents=ntuple(_ -> (-0.3, 0.7), D))
        fs = FieldSet(forest, D; G=2, centering=C)
        fill_by_coordinates!((x, v) -> x[v], fs)
        fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=2, restriction=2));
                     boundary=CellBoundary((x, v, δ) -> x[v]))
        @test all(b -> all(I -> ntuple(v -> fs.work[Tuple(I)..., v, b], D) ===
                                coordinates(fs, b, Tuple(I)),
                           CartesianIndices(size(fs.work)[1:D])), 1:nblocks(fs))
    end
end

@testset "Block extents tile the domain bit for bit: D=$D, $T" for
    D in (1, 2, 3), T in (Float64, Float32)
    # Guards a block's high face formed as its corner plus its width,
    # which need not be the bits of the next block's corner: the extents
    # would overlap or leave a gap of an ulp. The domain's high face is
    # `lo + K·h` at every level.
    for forest in (position_forest(T, D; N=4),
                   D == 2 ? patch_position_forest(T) : position_forest(T, D; N=6))
        for k in forest.leaves, d in 1:D
            ext = block_extent(forest, k)
            @test ext[d][1] === block_origin(forest, k)[d]
            nbrs = neighbor_keys(forest, k, ntuple(e -> e == d ? 1 : 0, D))
            for n in nbrs
                @test block_extent(forest, n)[d][1] === ext[d][2]
            end
            if isempty(nbrs)
                p = patch(forest, k)
                lo = T(TreeAMR.patch_extents(forest, p)[d][1])
                K = TreeAMR.patch_roots(forest, p)[d] * forest.N << level(k)
                @test ext[d][2] === fma(T(K), spacing(T, forest, k), lo)
                @test ext[d][2] === fma(T(K >> level(k)), spacing(T, forest, 0), lo)
            end
        end
    end
end

@testset "Every own point is located where coordinates puts it: D=$D, $T" for
    D in (1, 2), T in (Float64, Float32)
    # Guards a point geometry whose box is not the one positions are
    # formed in: a node on the high outer face, at `lo + K·h`, may lie an
    # ulp past the declared extent, and the interpolator would refuse a
    # position `coordinates` returned as outside the domain.
    forests = Any[position_forest(T, D)]
    D == 2 && push!(forests, patch_position_forest(T))
    for forest in forests, (C, P) in ((cellcentered(D), 0), (vertexcentered(D), 0),
                                      (elementcentered(D), 1))
        # On a forest of patches a stencil reads no ghost, which only an
        # element set's does not.
        TreeAMR.ispatched(forest) && P == 0 && continue
        vertexlike(C) && (pending(3, "own points located, $C, D=$D, $T"); continue)
        fs = position_set(forest, 1, C, P; G=P == 0 ? 2 : 1)
        fill_by_coordinates!((x, v) -> 1 + sum(x), fs)
        fill_ghosts!(fs, GhostSchedule(fs, P == 0 ?
                                           Operators(prolongation=2, restriction=2) :
                                           Operators(family=ElementLocal));
                     boundary=CellBoundary((x, v, δ) -> 1 + sum(x)))
        c = staggers(fs)
        xs = []
        for b in 1:nblocks(fs)
            p = patch(forest, blockkey(fs, b))
            for I in CartesianIndices(ntuple(d -> 1:(forest.N + c[d]), D))
                x = coordinates(fs, b, Tuple(I) .+ fs.G)
                push!(xs, TreeAMR.ispatched(forest) ? PatchPoint(x, p) : x)
            end
        end
        xs = [x for x in xs]
        basis = P == 0 ? Lagrange(2) : ElementPolynomial()
        r = interpolate(fs, xs, basis)
        @test all(r.inside)
        @test vec(r.values) ≈ [1 + sum(x isa PatchPoint ? x.ξ : x) for x in xs]
    end
end
