include("sphere.jl")
function growth(kw; eps, n, tend, cfl)
    S = Sphere(n; kw..., eps)
    rng = Xoshiro(17)
    for T in S.P
        T.u .= 1e-3 .* randn(rng, size(T.u)); T.v .= 1e-3 .* randn(rng, size(T.v))
    end
    E0 = energy(S); tblow = NaN
    every = max(1, round(Int, 1.0 / timestep(S, cfl)))
    evolve!(S, tend, cfl; every, monitor = t -> begin
        E = energy(S); ok = isfinite(E) && E < 1e3 * E0
        ok || (tblow = t); ok
    end)
    return energy(S) / E0, tblow
end
cases = [(m, eps, cfl) for m in (4, 6, 8) for eps in (0.2, 0.5, 1.0) for cfl in (0.25, 0.1)]
res = Vector{Any}(undef, length(cases))
Threads.@threads :greedy for k in eachindex(cases)
    m, eps, cfl = cases[k]
    res[k] = growth((corner=:own, m=m); eps, n=16, tend=20.0, cfl)
end
println("Owned corners, n = 16, random data to t = 20: final E/E₀ or blow-up time")
for (k, (m, eps, cfl)) in enumerate(cases)
    r = res[k]
    @printf("  m=%d ε=%.1f cfl=%.2f  %s\n", m, eps, cfl, isnan(r[2]) ? @sprintf("%.2e", r[1]) : @sprintf("t=%.1f", r[2]))
end
