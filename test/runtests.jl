using Test
using Random
using TreeAMR

include("oracles.jl")
include("ghost_oracles.jl")
include("wave.jl")

@testset "TreeAMR.jl" begin

@testset "MortonKey" begin
    k = MortonKey(0, 2, (1, 3))
    @test k == MortonKey{2}(0, 2, (1, 3))
    @test k != MortonKey{2}(0, 2, (1, 2))
    @test k != MortonKey{2}(1, 2, (1, 3))
    @test level(k) == 2
    @test hash(k) == hash(MortonKey{2}(0, 2, (1, 3)))
    @test sprint(show, k) == "MortonKey{2}(root=0, level=2, coords=(1, 3))"

    @test_throws ArgumentError MortonKey{2}(0, -1, (0, 0))
    @test_throws ArgumentError MortonKey{2}(-1, 0, (0, 0))
    @test_throws ArgumentError MortonKey{2}(0, 1, (2, 0))            # coord out of range
    @test_throws ArgumentError MortonKey{2}(0, MAX_LEVEL + 1, (0, 0))

    # Level 32 is representable; refining it is not.
    deep = MortonKey{1}(0, MAX_LEVEL, (typemax(UInt32),))
    @test level(deep) == MAX_LEVEL
    @test_throws ArgumentError childkeys(deep)
end

@testset "Parent/child relations: D=$D" for D in (1, 2, 3)
    root = MortonKey{D}(0, 0, ntuple(_ -> 0, D))
    @test_throws ArgumentError parentkey(root)

    kids = childkeys(root)
    @test length(kids) == 2^D
    @test allunique(kids)
    @test all(c -> parentkey(c) == root, kids)
    @test all(c -> level(c) == 1, kids)
    @test all(c -> isancestor(root, c), kids)
    @test !isancestor(root, root)                       # strict
    @test all(c -> !isancestor(c, root), kids)

    # Grandchildren are still descendants; siblings never are.
    grandkids = collect(Iterators.flatten(childkeys(c) for c in kids))
    @test all(g -> isancestor(root, g), grandkids)
    @test all(((a, b),) -> a == b || !isancestor(a, b),
              Iterators.product(kids, kids))

    @test sortedchildkeys(root) == sort(collect(kids))
    @test issorted(sortedchildkeys(root))
end

@testset "Curve order matches the naive bit-plane order: D=$D" for D in (1, 2, 3)
    rng = MersenneTwister(20 + D)
    keys = MortonKey{D}[]
    for _ in 1:400
        lvl = rand(rng, 0:6)
        push!(keys, MortonKey{D}(rand(rng, 0:2), lvl, ntuple(_ -> rand(rng, 0:(1 << lvl) - 1), D)))
    end
    @test all(((a, b),) -> isless(a, b) == naive_isless(a, b), Iterators.product(keys, keys))

    # A total (strict weak) order: irreflexive and antisymmetric.
    @test all(a -> !isless(a, a), keys)
    @test all(((a, b),) -> !(isless(a, b) && isless(b, a)), Iterators.product(keys, keys))

    # An ancestor always precedes its descendants.
    for k in keys
        level(k) == MAX_LEVEL && continue
        @test all(c -> isless(k, c), childkeys(k))
    end

    @test sort(keys; lt=isless) == sort(keys; lt=naive_isless)
end

@testset "Forest construction" begin
    @test_throws ArgumentError Forest((1, 1); N=3, G=1)              # N odd
    @test_throws ArgumentError Forest((1, 1); N=2, G=2)              # N < 2G
    @test_throws ArgumentError Forest((0, 1); N=4, G=1)              # empty brick
    @test_throws ArgumentError Forest((1, 1); N=0, G=0)              # N not positive
    # Non-cubic blocks: a 2:1 domain over a 1:1 root brick.
    @test_throws ArgumentError Forest((1, 1); N=4, G=1, extents=((0.0, 2.0), (0.0, 1.0)))
    # ... but the same domain over a matching 2:1 brick is fine.
    @test Forest((2, 1); N=4, G=1, extents=((0.0, 2.0), (0.0, 1.0))) isa Forest{2}

    forest = Forest((2, 3); N=4, G=1)
    @test nleaves(forest) == 6
    @test maxlevel(forest) == 0
    @test issorted(forest.leaves)
    @test allunique(forest.leaves)
    @test all(r -> root_index(forest, root_position(forest, r)) == r, 0:5)
    @test forest.periodic == (false, false)

    @test length(alldirections(Val(1))) == 2
    @test length(alldirections(Val(2))) == 8
    @test length(alldirections(Val(3))) == 26
    @test !any(δ -> all(==(0), δ), alldirections(Val(3)))
end

@testset "Refine/coarsen: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 1, D); N=4, G=1)
    root = only(forest.leaves)

    refine!(forest, root)
    @test nleaves(forest) == 2^D
    @test issorted(forest.leaves)
    @test maxlevel(forest) == 1
    @test !isleaf(forest, root)
    @test all(c -> isleaf(forest, c), childkeys(root))

    coarsen!(forest, root)
    @test forest.leaves == [root]

    # Refining twice in a row, then coarsening back, is the identity.
    refine!(forest, root)
    first_child = forest.leaves[1]
    refine!(forest, first_child)
    @test nleaves(forest) == 2^D - 1 + 2^D
    @test issorted(forest.leaves)
    coarsen!(forest, first_child)
    coarsen!(forest, root)
    @test forest.leaves == [root]

    # Errors: not a leaf, and children not all leaves.
    @test_throws ArgumentError refine!(forest, childkeys(root)[1])   # not a leaf yet
    @test_throws ArgumentError coarsen!(forest, root)                # children absent
    refine!(forest, root)
    @test_throws ArgumentError refine!(forest, root)                 # no longer a leaf

    # Refining several keys at once matches refining them one at a time.
    a = Forest(ntuple(_ -> 1, D); N=4, G=1)
    b = Forest(ntuple(_ -> 1, D); N=4, G=1)
    refine!(a, only(a.leaves))
    refine!(b, only(b.leaves))
    targets = [a.leaves[1], a.leaves[end]]
    refine!(a, targets)
    for t in targets
        refine!(b, t)
    end
    @test a.leaves == b.leaves
    @test issorted(a.leaves)
end

@testset "Property tests: D=$D" for D in (1, 2, 3)
    rng = MersenneTwister(1000 + D)
    ntrials, nsteps, maxlvl = D == 3 ? (8, 12, 2) : (15, 25, 3)

    for _trial in 1:ntrials
        forest = random_forest(rng, Val(D); nsteps=nsteps, maxlvl=maxlvl)
        leaves = copy(forest.leaves)
        n = length(leaves)

        # --- Sorted, duplicate-free linear octree -------------------
        @test issorted(forest.leaves)
        @test allunique(forest.leaves)

        # --- The leaves tile the domain exactly ---------------------
        # No two leaves overlap ...
        @test !any(((i, j),) -> i < j && overlaps(forest, leaves[i], leaves[j]),
                   Iterators.product(1:n, 1:n))
        # ... and together they fill it, in exact rational arithmetic.
        volume = sum(leaves) do k
            lo, hi = leafbox(forest, k)
            prod(hi .- lo)
        end
        @test volume == prod(forest.roots)

        # --- balance! ------------------------------------------------
        balance!(forest)
        bleaves = copy(forest.leaves)
        bn = length(bleaves)

        @test issorted(forest.leaves)
        @test allunique(forest.leaves)
        @test isbalanced(forest)
        # Balancing only refines, so it never loses coverage.
        @test bn >= n
        bvolume = sum(bleaves) do k
            lo, hi = leafbox(forest, k)
            prod(hi .- lo)
        end
        @test bvolume == prod(forest.roots)
        @test !any(((i, j),) -> i < j && overlaps(forest, bleaves[i], bleaves[j]),
                   Iterators.product(1:bn, 1:bn))

        # Check 2:1 balance against brute-force geometry, not against
        # the package's own neighbor search.
        @test all(Iterators.product(1:bn, 1:bn)) do (i, j)
            i < j || return true
            adjacent(forest, bleaves[i], bleaves[j]) || return true
            abs(level(bleaves[i]) - level(bleaves[j])) <= 1
        end

        # balance! reaches a fixed point.
        balance!(forest)
        @test forest.leaves == bleaves

        # --- neighbor_keys against the geometric oracle --------------
        dirs = alldirections(Val(D))
        for (i, k) in enumerate(bleaves)
            found = Set{MortonKey{D}}()
            for δ in dirs
                nbrs = neighbor_keys(forest, k, δ)
                @test allunique(nbrs)
                for nb in nbrs
                    # Soundness: everything reported really does touch k,
                    # and really is a leaf.
                    @test isleaf(forest, nb)
                    @test adjacent(forest, k, nb)
                    # A neighbor at k's level or finer lies squarely in
                    # the direction it was reported for. A coarser one
                    # need not: it also spans the regions beyond, so it
                    # legitimately answers several directions at once.
                    @test level(nb) < level(k) || δ in adjacency_directions(forest, k, nb)
                    push!(found, nb)
                    # Same-level neighbors are exactly reciprocal.
                    if level(nb) == level(k)
                        @test k in neighbor_keys(forest, nb, map(-, δ))
                    end
                end
                # Under 2:1 balance a finer neighbor region is exactly
                # one level down and fully covered.
                if !isempty(nbrs) && level(first(nbrs)) > level(k)
                    @test all(nb -> level(nb) == level(k) + 1, nbrs)
                    @test length(nbrs) == 2^count(==(0), δ)
                end
            end
            # Completeness: every leaf that touches k is reported for
            # some direction.
            for (j, q) in enumerate(bleaves)
                i == j && continue
                adjacent(forest, k, q) || continue
                @test q in found
            end
        end
    end
end

@testset "Periodicity: D=$D" for D in (1, 2, 3)
    dirs = alldirections(Val(D))

    # A lone root, periodic everywhere, is its own neighbor in every
    # direction — the M_i = 1 self-neighbor case.
    forest = Forest(ntuple(_ -> 1, D); N=4, G=1, periodic=ntuple(_ -> true, D))
    k = only(forest.leaves)
    @test all(δ -> neighbor_keys(forest, k, δ) == [k], dirs)

    # The same brick, non-periodic, has no neighbors at all.
    open = Forest(ntuple(_ -> 1, D); N=4, G=1)
    ko = only(open.leaves)
    @test all(δ -> isempty(neighbor_keys(open, ko, δ)), dirs)

    # Refined and periodic: each child's outward faces wrap onto the
    # sibling across the domain, so every child has a full neighbor set.
    refine!(forest, k)
    @test all(forest.leaves) do c
        all(δ -> length(neighbor_keys(forest, c, δ)) == 1, dirs)
    end
end

@testset "Periodic wraparound across roots" begin
    # Two roots along dimension 1, periodic there only.
    forest = Forest((2, 2); N=4, G=1, periodic=(true, false))
    r0 = forest.leaves[findfirst(k -> k.root == root_index(forest, (0, 0)), forest.leaves)]
    r1 = forest.leaves[findfirst(k -> k.root == root_index(forest, (1, 0)), forest.leaves)]

    @test neighbor_keys(forest, r0, (-1, 0)) == [r1]        # wraps
    @test neighbor_keys(forest, r1, (1, 0)) == [r0]         # wraps back
    @test neighbor_keys(forest, r0, (1, 0)) == [r1]         # ordinary step
    @test isempty(neighbor_keys(forest, r0, (0, -1)))       # closed dimension

    # A coarse/fine interface across the periodic seam: refining r1
    # makes r0 see 2^(D-1) fine neighbors through the wrap.
    refine!(forest, r1)
    wrapped = neighbor_keys(forest, r0, (-1, 0))
    @test length(wrapped) == 2
    @test all(nb -> level(nb) == 1 && nb.root == r1.root, wrapped)
    @test all(nb -> r0 in neighbor_keys(forest, nb, (1, 0)), wrapped)
end

@testset "Balance ripples outward" begin
    # Nesting refinement into one corner of a *single* root keeps every
    # intermediate level present, so it is balanced already.
    nested = Forest((1, 1); N=4, G=1)
    for _ in 1:3
        refine!(nested, first(nested.leaves))
    end
    @test maxlevel(nested) == 3
    @test isbalanced(nested)

    # Driving the refinement into the corner where root 0 meets the
    # other three roots does create an imbalance: those roots sit at
    # level 0 against level-3 cells.
    forest = Forest((2, 2); N=4, G=1)
    deepest = nothing
    for _ in 1:3
        # The last leaf of root 0 in curve order is its max-coordinate
        # corner, the one touching the rest of the brick.
        deepest = last(filter(k -> k.root == 0, forest.leaves))
        refine!(forest, deepest)
    end
    @test maxlevel(forest) == 3
    @test !isbalanced(forest)

    before = nleaves(forest)
    balance!(forest)
    @test isbalanced(forest)
    @test nleaves(forest) > before

    # Balancing only refines: it never coarsens, so the level-3 cells
    # that forced the imbalance are untouched, and no leaf is pushed
    # deeper than the level that forced it.
    @test maxlevel(forest) == 3
    @test count(k -> level(k) == 3, forest.leaves) == 2^2
    @test all(c -> isleaf(forest, c), childkeys(deepest))

    # The result is graded rather than uniformly refined: the ripple
    # decays outward, leaving coarser leaves further away.
    @test minimum(level, forest.leaves) == 1
    @test count(k -> level(k) == 1, forest.leaves) > 0
    @test count(k -> level(k) == 2, forest.leaves) > 0
end

@testset "Geometry: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, G=1,
                    extents=ntuple(_ -> (-1.0, 1.0), D))
    # Two roots spanning [-1,1] with N=4 gives cells of width 1/4.
    @test root_spacing(forest) ≈ 0.25
    @test spacing(forest, 0) ≈ 0.25
    @test spacing(forest, 2) ≈ 0.0625
    @test minimum_spacing(forest) ≈ 0.25

    k = forest.leaves[findfirst(k -> k.root == root_index(forest, ntuple(_ -> 0, D)),
                                forest.leaves)]
    @test all(block_origin(forest, k) .≈ -1.0)
    @test all(e -> e[1] ≈ -1.0 && e[2] ≈ 0.0, block_extent(forest, k))
    # First interior cell (index G+1) is half a spacing in from the corner.
    @test all(cell_center(forest, k, ntuple(_ -> forest.G + 1, D)) .≈ -0.875)
    # The first ghost cell lies half a spacing outside.
    @test all(cell_center(forest, k, ntuple(_ -> forest.G, D)) .≈ -1.125)

    # Refining halves the spacing, and children tile the parent's extent.
    refine!(forest, k)
    @test minimum_spacing(forest) ≈ 0.125
    kids = filter(c -> isancestor(k, c), forest.leaves)
    @test length(kids) == 2^D
    @test sum(kids) do c
        ext = block_extent(forest, c)
        prod(e -> e[2] - e[1], ext)
    end ≈ prod(e -> e[2] - e[1], block_extent(forest, k))

    # Leaf extents tile the whole domain.
    total = sum(c -> prod(e -> e[2] - e[1], block_extent(forest, c)), forest.leaves)
    @test total ≈ 2.0^D
end

@testset "FieldSet: D=$D" for D in (1, 2, 3)
    forest = Forest(ntuple(_ -> 2, D); N=4, G=1, extents=ntuple(_ -> (0.0, 2.0), D))
    nvars = 3
    fs = FieldSet(forest, nvars)

    stored = forest.N + 2 * forest.G
    @test size(fs.work) == (ntuple(_ -> stored, D)..., nvars, 2^D)
    @test nblocks(fs) == nleaves(forest) == 2^D
    @test eltype(fs.work) == Float64
    @test all(iszero, fs.work)
    @test FieldSet{Float32}(forest, 1).work isa Array{Float32}
    @test_throws ArgumentError FieldSet(forest, 0)

    @test all(b -> blockkey(fs, b) == forest.leaves[b], 1:nblocks(fs))

    @test size(blockview(fs, 1)) == (ntuple(_ -> stored, D)..., nvars)
    @test size(blockview(fs, 1, 2)) == ntuple(_ -> stored, D)
    @test size(interiorview(fs, 1)) == (ntuple(_ -> forest.N, D)..., nvars)
    @test size(interiorview(fs, 1, 2)) == ntuple(_ -> forest.N, D)

    # Views alias the working array, and the interior really is the
    # ghost-free part of the block.
    fill!(interiorview(fs, 1, 2), 7.0)
    @test all(==(7.0), interiorview(fs, 1, 2))
    @test count(==(7.0), blockview(fs, 1, 2)) == forest.N^D
    @test count(==(7.0), fs.work) == forest.N^D
    @test all(iszero, interiorview(fs, 1, 1))

    # Filling by coordinates touches interiors only, and reproduces a
    # linear function exactly.
    fs2 = FieldSet(forest, 2)
    fill_by_coordinates!((x, v) -> v == 1 ? sum(x) : 1.0, fs2)
    for b in 1:nblocks(fs2)
        k = blockkey(fs2, b)
        block = blockview(fs2, b, 1)
        for idx in CartesianIndices(ntuple(_ -> (forest.G + 1):(forest.G + forest.N), D))
            @test block[idx] ≈ sum(cell_center(forest, k, Tuple(idx)))
        end
        @test all(==(1.0), interiorview(fs2, b, 2))
    end
    # Ghosts stay untouched; M2 fills them.
    ghostcount = (stored^D - forest.N^D) * nblocks(fs2) * 2
    @test count(iszero, fs2.work) >= ghostcount - nblocks(fs2) * 2

    # The interiors tile the domain: total cell count times cell volume
    # equals the domain volume.
    cellvol = prod(_ -> spacing(forest, 0), 1:D)
    @test nblocks(fs) * forest.N^D * cellvol ≈ 2.0^D
end

include("ghost_tests.jl")
include("state_tests.jl")
include("wave_tests.jl")

end
