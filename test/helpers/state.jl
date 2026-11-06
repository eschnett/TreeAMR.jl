# The definitions of `state_tests.jl`, split off so that each test file runs
# on its own: every test process includes all of `helpers/` ("Parallel
# groups" in CODE.md).

using KernelAbstractions: @kernel, @index, @Const
using TreeAMR: _block_mapreduce_host, _block_mapreduce_device
using KernelAbstractions: CPU

# A minimal application kernel: read a ghosted block, write state layout.
@kernel function double_interior!(du, @Const(work), ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                    # (i1..iD, block)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))
    c = ntuple(d -> I[d] + G[d], Val(D))
    du[inner..., 1, b] = 2 * work[c..., 1, b]
end

# A refined, periodic mesh with poisoned ghosts: if a reduction picked
# the wrong window, the poison shows up as a wrong number rather than a
# near-miss that a tolerance would swallow.
function poisoned_fieldset(::Val{D}, nvars; N=4, G=2, seed=7) where {D}
    forest = Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    refine!(forest, forest.leaves[1])
    balance!(forest)
    fs = FieldSet(forest, nvars; G=G)
    rng = MersenneTwister(seed)
    fill!(fs.work, 1e6)                       # poison, ghosts included
    for b in 1:nblocks(fs), v in 1:nvars
        iv = interiorview(fs, b, v)
        for i in eachindex(iv)
            iv[i] = randn(rng)
        end
    end
    return fs
end
