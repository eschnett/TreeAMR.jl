# What forming a copy kernel's index costs on a device: the scatter's
# copy, state vector to working array, written seven ways that differ
# only in how a work item finds its `(i, j, k, var, block)`.
#
#     TREEAMR_BENCH_BACKEND=cuda  julia --project=<env> bench/copy_index.jl
#     TREEAMR_BENCH_BACKEND=metal julia --project=<env> bench/copy_index.jl
#
# - `ntuple`: KernelAbstractions' `@index(Global, NTuple)` over the
#   five-dimensional ndrange, which is what `scatter!` did through 0.1.7.
#   KA forms it with two run-time integer divisions per axis.
# - `inverse64`, `inverse32`: a flat launch, the position recovered by
#   multiplying with precomputed inverses (`SignedMultiplicativeInverse`)
#   in 64-bit and in 32-bit arithmetic — the second is what `scatter!` now
#   does.
# - `inverse32lin`: the same, with the store's linear index formed in
#   32-bit arithmetic as well.
# - `shift`: `N` a power of two, so the three box axes are shifts and
#   masks by a run-time shift count, and only the variable axis takes an
#   inverse.
# - `table`: no arithmetic at all, each item's store index read from a
#   precomputed table, which bounds what any index arithmetic can save.
# - `copy`: a linear copy of the same values, the bandwidth floor.
#
# Every variant's result is checked against `ntuple`'s.
#
#     TREEAMR_BENCH_N      cells per block edge, a power of two (default 16)
#     TREEAMR_BENCH_G      ghost width (default 3, vertex-centered)
#     TREEAMR_BENCH_VARS   variables (default 20)
#     TREEAMR_BENCH_BLOCKS blocks (default 512)
#     TREEAMR_BENCH_T      Float64 (default where supported) or Float32

using KernelAbstractions
using KernelAbstractions: synchronize, supports_float64
using Base.MultiplicativeInverses: SignedMultiplicativeInverse
using Printf: @printf

const BNAME = lowercase(get(ENV, "TREEAMR_BENCH_BACKEND", "cpu"))
if BNAME == "cuda"
    using CUDA
elseif BNAME == "metal"
    using Metal
end
const BK = BNAME == "cuda" ? CUDABackend() : BNAME == "metal" ? MetalBackend() : CPU()
const N = parse(Int, get(ENV, "TREEAMR_BENCH_N", "16"))
const G = parse(Int, get(ENV, "TREEAMR_BENCH_G", "3"))
const NV = parse(Int, get(ENV, "TREEAMR_BENCH_VARS", "20"))
const NB = parse(Int, get(ENV, "TREEAMR_BENCH_BLOCKS", "512"))
const T = let want = get(ENV, "TREEAMR_BENCH_T", "")
    isempty(want) ? (supports_float64(BK) ? Float64 : Float32) :
    want == "Float64" ? Float64 : Float32
end
const S = N + 2G + 1
ispow2(N) || error("TREEAMR_BENCH_N must be a power of two for the `shift` variant")

@inline pos(::Tuple{}, x) = (x + one(x),)
@inline function pos(invs::Tuple, x)
    q = div(x, first(invs))
    return (x - q * oftype(x, first(invs).divisor) + one(x), pos(Base.tail(invs), q)...)
end

@kernel function k_ntuple!(w, @Const(u))
    I = @index(Global, NTuple)
    n = @index(Global, Linear)
    @inbounds w[I[1] + G, I[2] + G, I[3] + G, I[4], I[5]] = u[n]
end
@kernel function k_inverse!(w, @Const(u), invs)
    n = @index(Global, Linear)
    I = map(Int, pos(invs, (n - 1) % typeof(first(invs).divisor)))
    @inbounds w[I[1] + G, I[2] + G, I[3] + G, I[4], I[5]] = u[n]
end
@kernel function k_inverse32lin!(w, @Const(u), invs, strides)
    n = @index(Global, Linear)
    I = pos(invs, (n - 1) % Int32)
    g = Int32(G - 1)
    lin = (I[1] + g) + strides[1] * (I[2] + g) + strides[2] * (I[3] + g) +
          strides[3] * (I[4] - Int32(1)) + strides[4] * (I[5] - Int32(1))
    @inbounds w[Int(lin) + 1] = u[n]
end
@kernel function k_shift!(w, @Const(u), k::Int32, inv)
    n = @index(Global, Linear)
    x = (n - 1) % Int32
    m = (Int32(1) << k) - Int32(1)
    i = x & m; x >>= k
    j = x & m; x >>= k
    l = x & m; x >>= k
    b = div(x, inv)
    v = x - b * inv.divisor
    I = (Int(i) + G + 1, Int(j) + G + 1, Int(l) + G + 1, Int(v) + 1, Int(b) + 1)
    @inbounds w[I...] = u[n]
end
@kernel function k_table!(w, @Const(u), @Const(tab))
    n = @index(Global, Linear)
    @inbounds w[tab[n]] = u[n]
end
@kernel function k_copy!(v, @Const(u))
    n = @index(Global, Linear)
    @inbounds v[n] = u[n]
end

function best(f)
    f()
    synchronize(BK)
    t = Inf
    for _ in 1:30
        t = min(t, @elapsed (f(); synchronize(BK)))
    end
    return t
end

function main()
    L = N^3 * NV * NB
    host = rand(T, L)
    u = KernelAbstractions.allocate(BK, T, L)
    copyto!(u, host)
    w = KernelAbstractions.zeros(BK, T, S, S, S, NV, NB)
    v = similar(u)
    ref = KernelAbstractions.zeros(BK, T, S, S, S, NV, NB)
    k_ntuple!(BK)(ref, u; ndrange=(N, N, N, NV, NB))
    synchronize(BK)
    li = LinearIndices((S, S, S, NV, NB))
    tab = KernelAbstractions.allocate(BK, Int32, L)
    copyto!(tab, Int32[li[i + G, j + G, k + G, a, b]
                       for i in 1:N, j in 1:N, k in 1:N, a in 1:NV, b in 1:NB][:])
    inv(I, d) = SignedMultiplicativeInverse{I}(I(d))
    variants = (
        "ntuple" => () -> k_ntuple!(BK)(w, u; ndrange=(N, N, N, NV, NB)),
        "inverse64" => () -> k_inverse!(BK)(w, u, map(d -> inv(Int, d), (N, N, N, NV));
                                            ndrange=L),
        "inverse32" => () -> k_inverse!(BK)(w, u, map(d -> inv(Int32, d), (N, N, N, NV));
                                            ndrange=L),
        "inverse32lin" => () -> k_inverse32lin!(BK)(w, u, map(d -> inv(Int32, d),
                                                               (N, N, N, NV)),
                                                     Int32.((S, S^2, S^3, S^3 * NV));
                                                     ndrange=L),
        "shift" => () -> k_shift!(BK)(w, u, Int32(trailing_zeros(N)), inv(Int32, NV);
                                      ndrange=L),
        "table" => () -> k_table!(BK)(w, u, tab; ndrange=L),
        "copy" => () -> k_copy!(BK)(v, u; ndrange=L))
    @printf("backend=%s T=%s N=%d G=%d vars=%d blocks=%d\n", BNAME, T, N, G, NV, NB)
    @printf("# variant\tns per point\tTB/s\n")
    for (name, f) in variants
        fill!(w, zero(T))
        t = best(f)
        name == "copy" || Array(w) == Array(ref) || error("$name is wrong")
        @printf("%s\t%.4f\t%.3f\n", name, 1e9 * t / (N^3 * NB),
                2 * sizeof(T) * L / t / 1e12)
    end
    return nothing
end

main()
