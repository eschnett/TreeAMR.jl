# The conservative family on staggered field sets (TreeGRRMHD's AMR-9).
#
# An edge-centered value is the line integral of a vector potential along
# its edge, divided by the edge's length; a face-centered one the flux
# through its face, divided by its area. `B = curl A` makes the flux
# through a face the circulation around it, and the circulation around a
# coarse face equals the sum around its fine faces exactly when every
# coarse edge's value is the mean of the two fine edges along it — the
# fine edges strictly inside the coarse face cancel. So the claim is
# checked edge by edge, which is the stronger statement: wherever a coarse
# value and the fine values that tile it coexist, after a ghost fill and
# across a regrid, the coarse one is their mean. The oracle knows only
# exact positions (`exact_point`) and the centering, never a stencil.

const STAGGERED_CONS = Operators(prolongation=3, restriction=2, family=Conservative,
                                 vertex_prolongation=4)

# Every stored value of `fs` (`owned = false`) or every owned one, keyed
# by its level and exact position. Ghost and owned values at the same
# level and position agree to rounding, so the first seen stands for all.
function staggered_values(fs::FieldSet{T,D}; owned::Bool=false) where {T,D}
    out = Dict{Tuple{Int,NTuple{D,Rational{Int}}},T}()
    G, N = fs.G, fs.forest.N
    range = owned ? CartesianIndices(ntuple(d -> (G[d] + 1):(G[d] + N), D)) :
            CartesianIndices(ntuple(d -> axes(fs.work, d), D))
    for b in 1:nblocks(fs), idx in range
        key = (level(blockkey(fs, b)), exact_point(fs, b, Tuple(idx)))
        haskey(out, key) || (out[key] = fs.work[Tuple(idx)..., 1, b])
    end
    return out
end

# The fine values that tile the coarse value at `p` on level `l`: along a
# cell-like dimension the two halves, a quarter of the coarse spacing
# either side; along a vertex-like one the coincident point. Wrapped like
# `exact_point`, since the forests here are periodic.
function fine_tiles(forest::Forest{D}, c::NTuple{D,Int}, l::Int, p) where {D}
    h = 1 // (forest.N * (1 << (l + 1)))            # the fine spacing
    offsets = ntuple(d -> c[d] == 1 ? (0 // 1,) : (-h / 2, h / 2), D)
    return [(l + 1, ntuple(d -> Rational{Int}(mod(p[d] + o[d], forest.roots[d])), D))
            for o in Iterators.product(offsets...)]
end

# The worst deviation of a coarse value in `coarse` from the mean of its
# fine tiles in `fine`, over every coarse value whose tiles are all there,
# and how many were compared.
function tiling_error(coarse, fine, forest, c)
    worst, n = 0.0, 0
    for ((l, p), v) in coarse
        tiles = fine_tiles(forest, c, l, p)
        all(t -> haskey(fine, t), tiles) || continue
        worst = max(worst, abs(v - sum(t -> fine[t], tiles) / length(tiles)))
        n += 1
    end
    return worst, n
end

staggered_centerings(D) =
    D == 2 ? [edgecentered(2, 1), edgecentered(2, 2), vertexcentered(2)] :
    [edgecentered(3, 1), edgecentered(3, 3), facecentered(3, 2)]

@testset "A coarse edge is the mean of its fine edges after a ghost fill: D=$D" for
        D in (2, 3)
    # Restricted coarse ghosts against the fine edges they cover, and
    # prolongated fine ghosts against the coarse edges they lie along, on
    # data with no structure at all. The point-value family, which
    # interpolates along the edge, is the negative control: it is what
    # lost TreeGRRMHD's flux across levels.
    forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> true, D))
    for C in staggered_centerings(D)
        c = staggers(C)
        results = map((STAGGERED_CONS, Operators(prolongation=4, restriction=4))) do ops
            fs = FieldSet(forest, 1; G=2, centering=C)
            fill_arbitrary!(fs)
            fill_ghosts!(fs, GhostSchedule(fs, ops))
            all = staggered_values(fs)
            return tiling_error(all, all, forest, c)
        end
        (worst, n), (control, _) = results
        @test n > 0
        @test worst < 1e-14
        all(==(1), c) || @test control > 1e-3
    end
end

@testset "A regrid keeps every coarse edge the mean of its fine edges: D=$D" for
        D in (2, 3)
    # Across regrids that refine and then coarsen: a refined block's new
    # fine edges along each old coarse edge average to it, and a coarsened
    # block's new coarse edge is the mean of the old fine edges along it,
    # so the circulation around every face a regrid refined or coarsened is
    # unchanged. The flags follow from the block alone: `flag_blocks` calls
    # its function on threads, so a random stream would hand the draws out
    # in another order at another thread count.
    for C in staggered_centerings(D)
        c = staggers(C)
        forest = nested_forest(Val(D); N=D == 3 ? 4 : 8, roots=D == 3 ? 2 : 4,
                               periodic=ntuple(_ -> true, D))
        fs = FieldSet(forest, 1; G=1, centering=C)
        fill_arbitrary!(fs)
        nrefined = ncoarsened = 0
        for pass in 1:2
            old = staggered_values(fs; owned=true)
            flags = flag_blocks(forest) do b, k
                pass == 1 ? (isodd(b) && level(k) < 2 ? Refine : Keep) :
                (level(k) >= 1 ? Coarsen : Keep)
            end
            regrid!(forest, fs => GhostSchedule(fs, STAGGERED_CONS); flags=flags)
            new = staggered_values(fs; owned=true)
            worst, n = tiling_error(old, new, forest, c)      # refined
            @test worst < 1e-14
            nrefined += n
            worst, n = tiling_error(new, old, forest, c)      # coarsened
            @test worst < 1e-14
            ncoarsened += n
        end
        @test nrefined > 0 && ncoarsened > 0
    end
end

@testset "The staggered conservative operators reproduce linear data: D=$D" for
        D in (2, 3)
    # A linear field's average along a cell-like dimension is its value at
    # the centre, so linear point data are what the operators must return.
    forest = nested_forest(Val(D); N=8)
    f = makepoly(D, 1)
    for C in allcenterings(Val(D))
        @test exchange_error(forest, STAGGERED_CONS, f; centering=C, G=1) < 1e-12
    end
end

@testset "A cell-centered set ignores vertex_prolongation, bit for bit: D=$D" for
        D in (1, 2, 3)
    # The keyword only names the operator along a stagger; a cell-centered
    # conservative set must fill and regrid exactly as without it.
    plain = Operators(prolongation=3, restriction=2, family=Conservative)
    results = map((plain, STAGGERED_CONS)) do ops
        forest = nested_forest(Val(D); N=8, periodic=ntuple(_ -> true, D))
        fs = FieldSet(forest, 1; G=1)
        fill_arbitrary!(fs)
        fill_ghosts!(fs, GhostSchedule(fs, ops))
        filled = copy(fs.work)
        flags = flag_blocks((b, k) -> isodd(b) ? Refine : Keep, forest)
        regrid!(forest, fs => GhostSchedule(fs, ops); flags=flags)
        return filled, copy(fs.work)
    end
    @test results[1][1] == results[2][1]
    @test results[1][2] == results[2][2]
end
