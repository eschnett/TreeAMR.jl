# The definitions of `centering_tests.jl`, split off so that each test file runs
# on its own: every test process includes all of `helpers/` ("Parallel
# groups" in CODE.md).

using KernelAbstractions: @kernel, @index

const OPS2C = Operators(prolongation=2, restriction=2)

# Writes 1 at every stored index the launch reaches, so that counting
# ones says exactly which points `map_blocks!` covered.
@kernel function mark_kernel!(work, ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)
    work[ntuple(d -> I[d] + G[d], Val(D))..., 1, I[D + 1]] = 1.0
end

# The stored form's counterpart. Two things differ and both are the
# point: it does *not* add `G` — under `stored = true` the global index
# already is the stored index — and it writes a value that encodes that
# index, so a launch that covered the right *number* of cells but the
# wrong ones is caught as well. `S` is a radix larger than any stored
# extent here.
@kernel function mark_stored_kernel!(work, ::Val{D}, ::Val{S}) where {D,S}
    I = @index(Global, NTuple)                     # already a stored index
    idx = ntuple(d -> I[d], Val(D))
    code = 0
    for d in 1:D
        code = code * S + idx[d]
    end
    work[idx..., 1, I[D + 1]] = code
end

storedcode(idx, S) = foldl((acc, i) -> acc * S + i, idx; init=0)
