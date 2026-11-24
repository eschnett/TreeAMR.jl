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
function symmetry(kw; eps, n, tend)
    S = Sphere(n; kw..., eps)
    ws = Harmonic([1.0, 0, 0], 4)     # even under X → −X and under x → −x
    set_state!(S, X -> exact(ws, X, 0.0))
    inv = -Matrix(1.0I, 3, 3); mir = Matrix(Diagonal([-1.0, 1, 1]))
    out = Tuple{Float64,Float64}[]; t0 = 0.0
    for tm in (1.0, 10.0, tend)
        evolve!(S, tm - t0, 0.25); t0 = tm
        push!(out, (asymmetry(S, inv), asymmetry(S, mir)))
    end
    return out
end
vars = [("half-open, owned corners", (corner=:own, m=6)),
        ("  same + avg rhs", (corner=:own, m=6, avg=true)),
        ("overlap o=0", (overlap=0, m=6)), ("overlap o=0, m = 4", (overlap=0, m=4)),
        ("overlap o=2, m = 4", (overlap=2, m=4)), ("overlap o=3, m = 4", (overlap=3, m=4))]
tasks = vcat([(:g, v, e) for v in eachindex(vars) for e in (0.2, 0.5)], [(:s, v) for v in eachindex(vars)],
             [(:g32, v, e) for v in (4, 6) for e in (0.1, 0.2, 0.5, 1.0)])
res = Dict{Any,Any}(); lk = ReentrantLock()
Threads.@threads :greedy for t in tasks
    r = t[1] === :g ? growth(vars[t[2]][2]; eps=t[3], n=64, tend=50.0) :
        t[1] === :g32 ? growth(vars[t[2]][2]; eps=t[3], n=32, tend=100.0) :
        symmetry(vars[t[2]][2]; eps=0.5, n=32, tend=50.0)
    lock(() -> (res[t] = r), lk)
end
fmt(r) = isnan(r[2]) ? @sprintf("%-10.1e", r[1]) : @sprintf("t=%-8.1f", r[2])
println("n = 64, random data to t = 50 (ε = 0.2, 0.5); symmetric data at n = 32, ε = 0.5: asymmetry under X → −X and x → −x at t = 1, 10, 50")
for v in eachindex(vars)
    @printf("  %-26s %s %s | %s\n", vars[v][1], fmt(res[(:g, v, 0.2)]), fmt(res[(:g, v, 0.5)]),
            join([@sprintf("%.0e/%.0e", a...) for a in res[(:s, v)]], "  "))
end
println("n = 32, random data to t = 100 (ε = 0.1, 0.2, 0.5, 1.0)")
for v in (4, 6)
    @printf("  %-26s %s\n", vars[v][1], join([fmt(res[(:g32, v, e)]) for e in (0.1, 0.2, 0.5, 1.0)], " "))
end
