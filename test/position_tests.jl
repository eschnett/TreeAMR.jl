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
        fs = position_set(forest, D, C, P; G=1)
        fill_by_coordinates!((x, v) -> x[v], fs)
        N = forest.N
        @test all(b -> all(I -> ntuple(v -> fs.work[(Tuple(I) .+ fs.G)..., v, b], D) ===
                                coordinates(fs, b, Tuple(I) .+ fs.G),
                           CartesianIndices(ntuple(d -> N + staggers(fs)[d], D))),
                   1:nblocks(fs))
        mismatch(acc, p) = acc + (p.u === p.x ? 0 : 1)
        # Every copy of a shared point (M22): a count of mismatches over all.
        @test mesh_fold(mismatch, +, 0, fs; primaries=false) == 0
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

# --- M22 step 10b: positions anchored at zero ---------------------------------
#
# Where zero is a level-0 node of a patch along a dimension, positions are
# formed from zero, `0 + k·h`, rather than from the low corner (CODE.md,
# "Positions" under "Centerings"). The spacing `5/48`, which a double
# cannot hold, is the case the anchor exists for.

# A box `[−L, L]^D` of `2^D` roots, refined about the low corner twice, so
# that it has every kind of coarse-fine face on both sides of zero.
function centred_forest(::Type{T}, D; N=12, L=T(5 // 2), half=ntuple(_ -> false, D)) where {T}
    forest = Forest{T}(ntuple(d -> half[d] ? 1 : 2, D); N=N,
                       extents=ntuple(d -> (half[d] ? zero(T) : -L, L), D))
    for _ in 1:2
        refine!(forest, [first(forest.leaves)])
        balance!(forest)
    end
    return forest
end

# Every own point's position over a set, as a `Set` of tuples.
function own_positions(fs)
    c = staggers(fs)
    D = length(c)
    out = Set{NTuple{D,eltype(fs.work)}}()
    for b in 1:nblocks(fs), I in CartesianIndices(ntuple(d -> 1:(fs.forest.N + c[d]), D))
        push!(out, Tuple(coordinates(fs, b, Tuple(I) .+ fs.G)))
    end
    return out
end

@testset "Zero on a patch's grid is found exactly, and only there" begin
    # Guards the integer test: an anchor where zero misses the grid by a
    # representation error would move every position by it.
    anchor(ext, roots; N=8) = TreeAMR.position_anchors(
        Forest((roots,); N=N, extents=(ext,)))[1][1]
    @test anchor((-1.0, 1.0), 2) == 8
    @test anchor((-2.5, 2.5), 4; N=12) == 24
    @test anchor((0.0, 1.0), 2) == 0                 # the low corner is zero
    @test anchor((-1.0, 0.0), 2) == 16               # the high corner is
    @test anchor((-0.25, 0.75), 1) == 2              # a dyadic third of the way
    @test anchor((0.3, 1.3), 2) == 0                 # zero outside
    @test anchor((-0.3, 0.7), 1; N=10) == 0          # −0.3 is not −3/10 in a double
    @test anchor((-0.125, 0.875), 1; N=4) == 0       # zero a cell centre only
    f32 = Forest{Float32}((1,); N=8, extents=((-0.25f0, 0.75f0),))
    @test TreeAMR.position_anchors(f32)[1] == (2,)
end

@testset "A box centred on zero is antisymmetric bit for bit: D=$D, $T" for
    D in (1, 2, 3), T in (Float64, Float32)
    # Guards a position formed from the low corner, whose rounding differs
    # on the two sides of zero: `x(n − j) == −x(j)` along every dimension,
    # at every level and centering, at a spacing a double cannot hold.
    forest = centred_forest(T, D; N=D == 3 ? 6 : 12)
    for C in unique([vertexcentered(D), cellcentered(D), facecentered(D, 1)])
        fs = FieldSet{T}(forest, 1; G=1, centering=C)
        xs = own_positions(fs)
        # Every position is an exact multiple of its level's spacing,
        # formed from zero; on the uniform box every position's negation
        # along each dimension is a position too (`+ zero(T)` makes `−0`
        # the `0` it is).
        h0 = spacing(T, forest, 0)
        multiple(v) = any(L -> (h = h0 / (1 << L); k = round(v / h * 2) / 2;
                                v == (isinteger(k) ? fma(T(k), h, zero(T)) :
                                      fma(T(2k), h / 2, zero(T)))), 0:2)
        @test all(x -> all(d -> multiple(x[d]), 1:D), xs)
        coarse = FieldSet{T}(Forest{T}(ntuple(_ -> 2, D); N=forest.N,
                                       extents=forest.extents), 1; G=1, centering=C)
        cx = own_positions(coarse)
        @test all(x -> all(d -> Base.setindex(x, -x[d] + zero(T), d) in cx, 1:D), cx)
    end
end

@testset "A quadrant's, bitant's and half-turn domain's positions are the box's: D=$D" for
    D in (2, 3)
    # Guards the agreement of a symmetric domain with its full box at a
    # spacing a double cannot hold, which a position formed from each
    # domain's own low corner breaks by an ulp (TreeGeneralizedHarmonic's
    # quadrant against its box drifted to 1e−8 from it).
    for T in (Float64, Float32), C in (vertexcentered(D), cellcentered(D))
        L = T(5 // 2)
        box = FieldSet{T}(Forest{T}(ntuple(_ -> 4, D); N=12,
                                    extents=ntuple(_ -> (-L, L), D)), 1; G=1, centering=C)
        bx = own_positions(box)
        halves = [ntuple(d -> d >= 2, D), ntuple(d -> d == D, D)]
        for half in halves
            f = Forest{T}(ntuple(d -> half[d] ? 2 : 4, D); N=12,
                          extents=ntuple(d -> (half[d] ? zero(T) : -L, L), D))
            hx = own_positions(FieldSet{T}(f, 1; G=1, centering=C))
            @test issubset(hx, bx)
        end
    end
end

@testset "A domain without zero on its grid keeps its positions: D=$D" for D in (1, 2)
    # Guards the fallback: every position is `lo + k·h` from the low corner,
    # as before the anchor, so a domain off zero keeps its bits and digests.
    T = Float64
    forest = Forest{T}(ntuple(_ -> 3, D); N=8, extents=ntuple(_ -> (0.3, 1.3), D))
    @test TreeAMR.position_anchors(forest)[1] == ntuple(_ -> 0, D)
    fs = FieldSet{T}(forest, 1; G=1, centering=vertexcentered(D))
    h = spacing(T, forest, 0)
    for b in 1:nblocks(fs)
        o = block_origin(forest, blockkey(fs, b))
        B = TreeAMR.block_base(forest, blockkey(fs, b))
        @test o == ntuple(d -> fma(T(B[d]), h, T(0.3)), D)
    end
end

@testset "Every node, block face and domain edge is located as the forward map places it: D=$D, $T" for
    D in (1, 2, 3), T in (Float64, Float32)
    # Guards an inverse map that rounds differently from the forward one: a
    # point on a face must land in a block whose closed box holds it, the
    # finest such (M22 step 10c), and among blocks of one level in the one
    # above it; a node at the domain's high edge in the last block,
    # whichever way the division in `locate_leaf` rounds.
    for forest in (centred_forest(T, D; N=D == 3 ? 6 : 12), position_forest(T, D; N=6))
        n = 0
        exts = [block_extent(forest, k) for k in forest.leaves]
        holds(e, x) = all(d -> e[d][1] <= x[d] <= e[d][2], 1:D)
        for k in forest.leaves
            ext = block_extent(forest, k)
            h = spacing(T, forest, k)
            B = TreeAMR.block_base(forest, k)
            a = TreeAMR.block_low(T, forest, k)
            for I in CartesianIndices(ntuple(_ -> 1:(forest.N + 1), D))
                x = ntuple(d -> TreeAMR.nodeposition(a[d], h, B[d], I[d], 1, nothing), D)
                j = locate_point(forest, x)
                j === nothing && (n += 1; continue)
                e = exts[j]
                top = ntuple(d -> TreeAMR.location_box(T, forest, 1)[d][2], D)
                levels = [Int(forest.leaves[i].level) for i in eachindex(exts)
                          if holds(exts[i], x)]
                ok = holds(e, x) && forest.leaves[j].level == maximum(levels)
                ok &= minimum(levels) < maximum(levels) ||
                      all(d -> x[d] < e[d][2] || e[d][2] >= top[d], 1:D)
                ok || (n += 1)
            end
        end
        @test n == 0
        # Both edges of the domain, as given and as formed, are located.
        box = TreeAMR.location_box(T, forest, 1)
        for x in (ntuple(d -> T(forest.extents[d][1]), D), ntuple(d -> T(forest.extents[d][2]), D),
                  ntuple(d -> box[d][1], D), ntuple(d -> box[d][2], D))
            @test locate_point(forest, x) isa Int
        end
    end
end

@testset "A centred box's edges are each other's negation: D=1, $T" for T in (Float64, Float32)
    # Guards the edges of a zero anchor, `∓(n/2)·h`, which are `∓L` only
    # where the product rounds there: they must at least mirror each
    # other, so that the domain is symmetric.
    for L in (T(5 // 2), T(1), T(1 // 3), T(7 // 10)), roots in (2, 4)
        f = Forest{T}((roots,); N=12, extents=((-L, L),))
        lo = block_extent(f, first(f.leaves))[1][1]
        hi = block_extent(f, last(f.leaves))[1][2]
        @test lo === -hi
    end
end

@testset "Element nodes mirror, and so does an element set in a centred box: $T" for
    T in (Float64, Float32)
    # Guards the conditional claim for element sets: their nodes are
    # antisymmetric about a zero anchor only if the tabulated offsets
    # mirror exactly, `oᵢ + o_{P−i} == P + 1`, which they do for every
    # order tested (checked here), and then so are the positions.
    for P in 1:8
        o = TreeAMR.element_offsets(T, P)
        @test all(i -> o[i + 1] + o[P - i + 1] == T(P + 1), 0:P)
    end
    for D in (1, 2), P in (1, 2)
        f = Forest{T}(ntuple(_ -> 2, D); N=12, extents=ntuple(_ -> (-T(5 // 2), T(5 // 2)), D))
        fs = FieldSet{T}(f, 1; G=1, centering=elementcentered(D), P=P)
        xs = own_positions(fs)
        @test all(x -> all(d -> Base.setindex(x, -x[d] + zero(T), d) in xs, 1:D), xs)
    end
end

@testset "A centred box takes its own edges and refuses the next float out: D=$D, $T" for
    D in (1, 2), T in (Float64, Float32)
    # Guards what depends on an edge: anchored at zero the edges are
    # `∓(n/2)·h`, an ulp off `∓L` where the product rounds away, and a
    # point at the extent given, at the edge formed, or at a node a hook
    # writes must still be inside; a point past both is outside.
    L = T(7 // 10)
    f = Forest{T}(ntuple(_ -> 2, D); N=12, extents=ntuple(_ -> (-L, L), D))
    fs = FieldSet{T}(f, 1; G=2, centering=vertexcentered(D))
    fill_by_coordinates!((x, v) -> 1 + sum(x), fs)
    fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=2, restriction=2));
                 boundary=CellBoundary((x, v, δ) -> 1 + sum(x)))
    box = TreeAMR.location_box(T, f, 1)
    edges = [ntuple(_ -> -L, D), ntuple(_ -> L, D), ntuple(d -> box[d][1], D),
             ntuple(d -> box[d][2], D)]
    r = interpolate(fs, edges, Lagrange(2))
    @test all(r.inside)
    @test vec(r.values) ≈ [1 + sum(x) for x in edges]
    out = [ntuple(d -> prevfloat(min(-L, box[d][1])), D), ntuple(d -> nextfloat(max(L, box[d][2])), D)]
    @test_throws "outside the domain" interpolate(fs, out, Lagrange(2))
end
