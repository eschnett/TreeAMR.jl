include("sphere.jl")
function noise(kw; eps, n, tend)
    S = Sphere(n; kw..., eps)
    rng = Xoshiro(17)
    for T in S.P
        T.u .= 1e-3 .* randn(rng, size(T.u)); T.v .= 1e-3 .* randn(rng, size(T.v))
    end
    E0 = energy(S); tblow = NaN
    every = max(1, round(Int, 1.0 / timestep(S, 0.25)))
    evolve!(S, tend, 0.25; every, monitor = t -> begin
        E = energy(S); ok = isfinite(E) && E < 1e3 * E0
        ok || (tblow = t); ok
    end)
    return energy(S) / E0, tblow
end
w = Harmonic([0.3, -0.5, 1.0], 3)
vars = [("extrapolate, 6 points", (corner=:avg, mc=6)), ("extrapolate, 4", (corner=:avg, mc=4)),
        ("extrapolate, 3", (corner=:avg, mc=3)), ("extrapolate, 2", (corner=:avg, mc=2)),
        ("owned (hypothetical)", (corner=:own,)), ("owned + avg rhs", (corner=:own, avg=true))]
res = Vector{Any}(undef, length(vars)); conv = Vector{Any}(undef, length(vars))
Threads.@threads :greedy for k in eachindex(vars)
    kw = vars[k][2]
    res[k] = [noise(kw; eps=e, n=32, tend=100.0) for e in (0.05, 0.1, 0.2)]
    conv[k] = map((16, 32, 64)) do n
        S = Sphere(n; kw...); set_state!(S, X -> exact(w, X, 0.0))
        errors(S, w, evolve!(S, 2.0, 0.25))[1]
    end
end
println("Cubed sphere, the unowned corners: L2 at t = 2 (n = 16, 32, 64), and random data at n = 32 to t = 100 (ε = 0.05, 0.1, 0.2)")
for k in eachindex(vars)
    @printf("  %-22s %s |", vars[k][1], join([@sprintf("%.2e", x) for x in conv[k]], " "))
    for r in res[k]
        isnan(r[2]) ? @printf(" %-10.2e", r[1]) : @printf(" t=%-8.1f", r[2])
    end
    println()
end
