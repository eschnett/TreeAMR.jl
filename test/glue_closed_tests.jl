# Closed blocks across glues (M22 step 8): the copies of a point on a glued
# face are one number, by the lowest patch's copy across a glue that copies
# (rule 2) and by stage A's mean across a glue the set is interpolated at
# (rule 4). The claims are made against a brick laid out as patches, cut
# where the glues are: a fill, a stencil pass and `unify!` over the patches
# must give the brick's numbers, bit for bit where the brick's own rule
# picks the same copy as the patches' and where the copies agree.

# A brick of `roots` cut along dimension 1 at `cut` into two patches whose
# extents are the brick's, numbered `order` (`(1, 2)` the lower half
# first), glued with the identity; `reflecting` as for any forest.
function cut_pair(roots::NTuple{D,Int}; cut=2, order=(1, 2), N=8,
                  reflecting=ntuple(_ -> (false, false), D)) where {D}
    halves = [(ntuple(_ -> 0, D), Base.setindex(roots, cut, 1)),
              (Base.setindex(ntuple(_ -> 0, D), cut, 1), roots)]
    box(p) = halves[order[p]]
    patches = [(roots=box(p)[2] .- box(p)[1],
                extents=ntuple(d -> (Float64(box(p)[1][d]), Float64(box(p)[2][d])), D))
               for p in 1:2]
    lower, upper = findfirst(==(1), order), findfirst(==(2), order)
    glue = patch_glue_both((lower, 1, 2), (upper, 1, 1), ntuple(identity, D))
    return Forest(patches; N=N, glue=glue, reflecting=reflecting)
end

# Refine the root blocks with lower corners `corners` (brick coordinates)
# and balance. No coarse-fine face may cross a glued face (conformity), so
# a refined region meets the glued plane `x = 2` on both sides alike: in
# 2D a strip across it, whose coarse-fine line crosses the glue; in 3D an
# L in `(y, z)` on both sides, whose concave edge, along `x`, crosses the
# glued plane at `(2, 1, 1)`, where a ghost's face-plane rows are held by a
# block of its level across the glue (`ghost_plane_parts`).
function refine_corners!(forest::Forest{D}, corners) where {D}
    targets = filter(k -> ntuple(d -> block_extent(forest, k)[d][1], D) in corners,
                     forest.leaves)
    refine!(forest, targets)
    balance!(forest)
    return forest
end
glue_l(D) = D == 2 ? ((1.0, 0.0), (2.0, 0.0)) :
            Tuple((x, y, z) for x in (1.0, 2.0)
                  for (y, z) in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

# Copy every block of `from` into the block of `to` at the same level and
# lower corner, ghosts included.
function copy_matching!(to::FieldSet{T,D}, from::FieldSet{T,D}) where {T,D}
    lower(fs, b) = ntuple(d -> block_extent(fs.forest, blockkey(fs, b))[d][1], D)
    index = Dict((level(blockkey(from, b)), lower(from, b)) => b for b in 1:nblocks(from))
    for b in 1:nblocks(to)
        bb = index[(level(blockkey(to, b)), lower(to, b))]
        selectdim(to.work, D + 2, b) .= selectdim(from.work, D + 2, bb)
    end
    return to
end

# A stencil pass on the working array's own points, the `2D + 1`-point
# Laplacian in index space, written back over them: what a right-hand side
# leaves before `unify!`, every copy computed by its own block.
function stencil_pass!(fs::FieldSet{T,D}) where {T,D}
    c = staggers(fs)
    own = CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + fs.forest.N + c[d]), D))
    out = copy(fs.work)
    for b in 1:nblocks(fs), v in 1:fs.nvars, I in own
        acc = -2D * fs.work[I, v, b]
        for d in 1:D
            e = CartesianIndex(ntuple(a -> a == d ? 1 : 0, D))
            acc += fs.work[I + e, v, b] + fs.work[I - e, v, b]
        end
        out[I, v, b] = acc
    end
    fs.work .= out
    return fs
end

# The worst difference over every own point of two sets over the same
# blocks, matched by level and lower corner, and the set of positions
# where they differ at all.
function own_differences(fsp::FieldSet{T,D}, fsb::FieldSet{T,D}) where {T,D}
    lower(fs, b) = ntuple(d -> block_extent(fs.forest, blockkey(fs, b))[d][1], D)
    index = Dict((level(blockkey(fsb, b)), lower(fsb, b)) => b for b in 1:nblocks(fsb))
    c = staggers(fsp)
    own = CartesianIndices(ntuple(d -> (fsp.G[d] + 1):(fsp.G[d] + fsp.forest.N + c[d]), D))
    worst = 0.0
    where_ = Set{Any}()
    for b in 1:nblocks(fsp)
        bb = index[(level(blockkey(fsp, b)), lower(fsp, b))]
        for v in 1:fsp.nvars, I in own
            x, y = fsp.work[I, v, b], fsb.work[I, v, bb]
            isequal(x, y) && continue
            worst = max(worst, abs(x - y))
            push!(where_, coordinates(fsp, b, Tuple(I)))
        end
    end
    return worst, where_
end

const GLUE_CLOSED_OPS = Operators(prolongation=4, restriction=4)

@testset "A copying glue fills and unifies as the brick, refined across it: D=$D" for
        D in (2, 3)
    # Guards stage 0 and the ghost stages across a glue that copies (M22):
    # with a refined region across the glued plane, in 3D with a concave
    # edge crossing it, where a ghost's face-plane rows are copied from a
    # block of its level across the glue (`ghost_plane_parts`), and
    # coarse-fine edges across the glue, every stored point after a fill, and
    # every own point after a stencil pass and `unify!`, must be the
    # brick's bit for bit, since the lowest patch is the lower half, whose
    # copy is the brick's primary. Vertex-, face- and edge-centered.
    roots = ntuple(d -> d == 1 ? 4 : 2, D)
    rng = Xoshiro(80 + D)
    centerings = [vertexcentered(D), facecentered(D, 1), facecentered(D, 2)]
    D == 3 && push!(centerings, (:vertex, :vertex, :cell))
    for C in centerings
        patched = refine_corners!(cut_pair(roots), glue_l(D))
        brick = refine_corners!(Forest(roots; N=8), glue_l(D))
        @test nleaves(patched) == nleaves(brick) > prod(roots)
        fsp = FieldSet(patched, 2; G=2, centering=C)
        fsb = FieldSet(brick, 2; G=2, centering=C)
        fsb.work .= randn(rng, size(fsb.work))
        copy_matching!(fsp, fsb)
        hook = boundary_by_coordinates((x, v) -> v + x[1] - x[2])
        sp, sb = GhostSchedule(fsp, GLUE_CLOSED_OPS), GhostSchedule(fsb, GLUE_CLOSED_OPS)
        fill_ghosts!(fsp, sp; boundary=hook)
        fill_ghosts!(fsb, sb; boundary=hook)
        result = compare_matching_blocks(fsp, fsb)
        @test result !== nothing && result[1] == 0
        stencil_pass!(fsp)
        stencil_pass!(fsb)
        unify!(fsp, sp)
        unify!(fsb, sb)
        worst, _ = own_differences(fsp, fsb)
        @test worst == 0
    end
end

@testset "Relabelling the patches moves only what the lowest-patch rule decides: D=$D" for
        D in (2, 3)
    # Guards the order-independence of the rule (M22): numbering the upper
    # half patch 1 makes its copy of a point on the glued plane the
    # primary, where the brick keeps the lower half's. So from data whose
    # copies differ, a fill differs from the brick at the glued plane's
    # points only (and in the ghosts that read them); from unified data,
    # whose copies agree, a fill, a stencil pass and `unify!` give the
    # brick's bits under either numbering, since every copy is computed
    # from the same values.
    roots = ntuple(d -> d == 1 ? 4 : 2, D)
    C = vertexcentered(D)
    for order in ((1, 2), (2, 1))
        patched = refine_corners!(cut_pair(roots; order=order), glue_l(D))
        brick = refine_corners!(Forest(roots; N=8), glue_l(D))
        fsp = FieldSet(patched, 1; G=2, centering=C)
        fsb = FieldSet(brick, 1; G=2, centering=C)
        hook = boundary_by_coordinates((x, v) -> x[1] - x[2])
        sp, sb = GhostSchedule(fsp, GLUE_CLOSED_OPS), GhostSchedule(fsb, GLUE_CLOSED_OPS)
        fsb.work .= randn(Xoshiro(90 + D), size(fsb.work))
        copy_matching!(fsp, fsb)
        fill_ghosts!(fsp, sp; boundary=hook)
        fill_ghosts!(fsb, sb; boundary=hook)
        _, moved = own_differences(fsp, fsb)
        if order == (1, 2)
            @test isempty(moved)
        else
            @test !isempty(moved) && all(x -> x[1] == 2, moved)
        end
        # From unified data: the brick's fill, copied over.
        copy_matching!(fsp, fsb)
        fill_ghosts!(fsp, sp; boundary=hook)
        fill_ghosts!(fsb, sb; boundary=hook)
        result = compare_matching_blocks(fsp, fsb)
        @test result !== nothing && result[1] == 0
        stencil_pass!(fsp)
        stencil_pass!(fsb)
        unify!(fsp, sp)
        unify!(fsb, sb)
        @test isempty(own_differences(fsp, fsb)[2])
    end
end

@testset "A copying glue beside a reflecting face fills and unifies as the brick: D=$D" for
        D in (2, 3)
    # Guards stage P and the mirror beside a glue (M22): the low face of
    # dimension 2 reflects on both patches, a variable odd across it, and
    # the glued plane meets the wall plane. Every stored point after a
    # fill and every own point after a stencil pass and `unify!` must be
    # the brick's bit for bit, and the odd variable zero on the wall.
    roots = ntuple(d -> d == 1 ? 4 : 2, D)
    reflecting = ntuple(d -> (d == 2, false), D)
    parity = [ntuple(_ -> EvenParity, D), ntuple(d -> d == 2 ? OddParity : EvenParity, D)]
    patched = refine_corners!(cut_pair(roots; reflecting=reflecting), glue_l(D))
    brick = refine_corners!(Forest(roots; N=8, reflecting=reflecting), glue_l(D))
    fsp = FieldSet(patched, 2; G=2, centering=vertexcentered(D), parity=parity)
    fsb = FieldSet(brick, 2; G=2, centering=vertexcentered(D), parity=parity)
    fsb.work .= randn(Xoshiro(100 + D), size(fsb.work))
    copy_matching!(fsp, fsb)
    hook = boundary_by_coordinates((x, v) -> v == 2 ? 0.0 : x[1])
    sp, sb = GhostSchedule(fsp, GLUE_CLOSED_OPS), GhostSchedule(fsb, GLUE_CLOSED_OPS)
    @test !isempty(sp.selfimage)
    fill_ghosts!(fsp, sp; boundary=hook)
    fill_ghosts!(fsb, sb; boundary=hook)
    result = compare_matching_blocks(fsp, fsb)
    @test result !== nothing && result[1] == 0
    wall = [fsp.work[I, 2, b] for b in 1:nblocks(fsp)
            for I in CartesianIndices(size(fsp.work)[1:D])
            if coordinates(fsp, b, Tuple(I))[2] == 0]
    @test !isempty(wall) && all(iszero, wall)
    stencil_pass!(fsp)
    stencil_pass!(fsb)
    unify!(fsp, sp)
    unify!(fsb, sb)
    @test own_differences(fsp, fsb)[1] == 0
end

# The brick laid out as patches at once, its maps the brick's own
# placement, every coordinate shared: interpolation across a glue with
# this atlas is the copy, so a set interpolated across its glues must fill
# as the brick does where its copies agree.
struct BrickAtlas{D}
    lows::NTuple{4,NTuple{D,Float64}}
    highs::NTuple{4,NTuple{D,Float64}}
end
TreeAMR.chart_map(a::BrickAtlas, ξ, p) = ξ
function TreeAMR.chart_locate(a::BrickAtlas{D}, x) where {D}
    p = findfirst(q -> all(d -> a.lows[q][d] <= x[d] <= a.highs[q][d], 1:D), 1:4)
    return p === nothing ? nothing : (Tuple(x), p)
end
TreeAMR.chart_aligned(::BrickAtlas{D}, p, q) where {D} = ntuple(_ -> true, D)

# A brick of `2n × 2n` roots cut into four patches at `n` along both
# dimensions, glued with the identity, patch corners at the cut.
function cut_four(n; N=8)
    lows = ((0, 0), (n, 0), (0, n), (n, n))
    patches = [(roots=(n, n), extents=ntuple(d -> (Float64(lo[d]), Float64(lo[d] + n)), 2))
               for lo in lows]
    glue = vcat(patch_glue_both((1, 1, 2), (2, 1, 1), (1, 2)),
                patch_glue_both((3, 1, 2), (4, 1, 1), (1, 2)),
                patch_glue_both((1, 2, 2), (3, 2, 1), (1, 2)),
                patch_glue_both((2, 2, 2), (4, 2, 1), (1, 2)))
    atlas = BrickAtlas{2}(map(lo -> Float64.(lo), lows), map(lo -> Float64.(lo .+ n), lows))
    return Forest(patches; N=N, glue=glue), atlas
end

@testset "Stage A at a patch corner makes the copies the brick's" begin
    # Guards stage A (M22 rule 4) where four patches meet: a set
    # interpolated across its glues holds a copy of a corner point on each
    # of the four, stage A's members, and of a glued face's point on two.
    # From data whose copies agree, the fill, a stencil pass and `unify!`
    # give the brick's bits, the mean of equal members being that member;
    # a second fill changes nothing; and from data whose copies differ, the
    # fill moves the glued points only, to the mean of their copies.
    patched, atlas = cut_four(1)
    brick = Forest((2, 2); N=8)
    ops = Operators(prolongation=4, restriction=4, glue=2)
    fsp = FieldSet(patched, 1; G=2, centering=vertexcentered(2), glue=:interpolate)
    fsb = FieldSet(brick, 1; G=2, centering=vertexcentered(2))
    sp = GhostSchedule(fsp, ops; atlas=atlas)
    sb = GhostSchedule(fsb, Operators(prolongation=4, restriction=4))
    @test patched.conn.corners
    @test sort(unique(Array(sp.means.targets.mcount))) == [2, 4]
    hook = boundary_by_coordinates((x, v) -> x[1] - x[2])
    # Data whose copies differ: the means.
    fsb.work .= randn(Xoshiro(110), size(fsb.work))
    copy_matching!(fsp, fsb)
    raw = copy(fsp.work)
    fill_ghosts!(fsp, sp; boundary=hook)
    once = copy(fsp.work)
    fill_ghosts!(fsp, sp; boundary=hook)
    @test isequal(once, fsp.work)
    fill_ghosts!(fsb, sb; boundary=hook)
    _, moved = own_differences(fsp, fsb)
    @test !isempty(moved) && all(x -> x[1] == 1 || x[2] == 1, moved)
    # Unified data: the brick's fill, copied over.
    copy_matching!(fsp, fsb)
    fill_ghosts!(fsp, sp; boundary=hook)
    fill_ghosts!(fsb, sb; boundary=hook)
    result = compare_matching_blocks(fsp, fsb)
    @test result !== nothing && result[1] == 0
    stencil_pass!(fsp)
    stencil_pass!(fsb)
    unify!(fsp, sp)
    unify!(fsb, sb)
    @test isempty(own_differences(fsp, fsb)[2])
end
