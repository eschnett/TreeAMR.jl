# The definitions of `reflect_tests.jl`, split off so that each test file runs
# on its own: every test process includes all of `helpers/` ("Parallel
# groups" in CODE.md).

using TreeAMR: wall_direction, OuterFace, ReflectingFace

const OPS4 = Operators(prolongation=4, restriction=4)

# The wave equation on a box with reflecting walls at x₁ = 0 and L/2,
# periodic in the other dimensions: the same two-level refinement as
# `wave_forest`'s, which is symmetric about both walls, so the half box
# is exactly the fold of the periodic box.
function reflecting_wave_forest(::Val{D}, N; L=1.0) where {D}
    forest = Forest(ntuple(d -> d == 1 ? 2 : 4, D); N=N,
                    periodic=ntuple(d -> d != 1, D),
                    reflecting=ntuple(d -> d == 1 ? (true, true) : (false, false), D),
                    extents=ntuple(d -> d == 1 ? (0.0, L / 2) : (0.0, L), D))
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> L / 4 < (ext[d][1] + ext[d][2]) / 2 < 3L / 4, 1:D)
    end
    refine!(forest, targets)
    balance!(forest)
    return forest
end

wave_parity(D, p) = [ntuple(d -> d == 1 ? p : NoParity, D) for _ in 1:2]
