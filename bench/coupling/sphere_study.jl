include("sphere.jl")
function growth(kw; eps, n, tend)
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
vars = [("half-open, corners extrap. 3", (corner=:avg, mc=3, m=6)),
        ("half-open, owned corners", (corner=:own, m=6)),
        ("  same, m = 4", (corner=:own, m=4)),
        ("  same + avg rhs", (corner=:own, m=6, avg=true)),
        ("overlap o=0", (overlap=0, m=6)), ("overlap o=1", (overlap=1, m=6)),
        ("overlap o=2", (overlap=2, m=6)), ("overlap o=3", (overlap=3, m=6)),
        ("overlap o=2, m = 4", (overlap=2, m=4))]
epss = (0.1, 0.2, 0.5, 1.0)
tasks = vcat([(:g, v, e) for v in eachindex(vars) for e in epss],
             [(:c, v, n) for v in eachindex(vars) for n in (16, 32, 64)])
res = Dict{Any,Any}(); lk = ReentrantLock()
Threads.@threads :greedy for t in tasks
    r = if t[1] === :g
        growth(vars[t[2]][2]; eps=t[3], n=32, tend=100.0)
    else
        S = Sphere(t[3]; vars[t[2]][2]..., eps=0.5)
        set_state!(S, X -> exact(w, X, 0.0))
        errors(S, w, evolve!(S, 2.0, 0.25))[1]
    end
    lock(() -> (res[t] = r), lk)
end
println("Cubed sphere. L2 at t = 2 with ε = 0.5 (n = 16, 32, 64; rates), then random data at n = 32 to t = 100: final E/E₀ or blow-up time")
@printf("  %-30s %-28s", "", ""); foreach(e -> @printf(" ε=%-8.1f", e), epss); println()
for v in eachindex(vars)
    c = [res[(:c, v, n)] for n in (16, 32, 64)]
    @printf("  %-30s %.1e %.1e %.1e %4.1f %4.1f", vars[v][1], c..., log2(c[1] / c[2]), log2(c[2] / c[3]))
    for e in epss
        r = res[(:g, v, e)]
        isnan(r[2]) ? @printf(" %-10.1e", r[1]) : @printf(" t=%-8.1f", r[2])
    end
    println()
end
