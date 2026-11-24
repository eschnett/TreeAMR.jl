# Robustness of the torus results: a stronger kink (spacing 0.6 → 2.4,
# shear 0.6), CFL 0.4, n = 128, and t = 1000 at n = 32.
include("torus.jl")
function noise(s, var; eps, n, tend, cfl)
    sim = Sim(s, var, n; eps)
    rng = Xoshiro(17)
    for T in sim.P
        T.u .= 1e-3 .* randn(rng, size(T.u)); T.v .= 1e-3 .* randn(rng, size(T.v))
    end
    E0 = energy(sim); tblow = NaN
    every = max(1, round(Int, 1.0 / timestep(sim, cfl)))
    evolve!(sim, tend, cfl; every, monitor = t -> begin
        E = energy(sim)
        ok = isfinite(E) && E < 1e3 * E0
        ok || (tblow = t)
        ok
    end)
    return energy(sim) / E0, tblow
end
mild = Setup(0.8, 0.4, 0.3)
harsh = Setup(0.6, 0.9, 0.6)
vars = [Variant("halfopen", :halfopen, 0, false, 6),
        Variant("halfopen+avg", :halfopen, 0, true, 6),
        Variant("overlap o=2", :overlap, 2, false, 6)]
cases = [("mild", mild, 0.25, 128, 200.0), ("mild", mild, 0.25, 32, 1000.0),
         ("mild", mild, 0.4, 32, 200.0), ("harsh", harsh, 0.25, 32, 200.0),
         ("harsh", harsh, 0.25, 64, 200.0)]
epss = (0.05, 0.1, 0.2)
jobs = [(c, v, e) for c in cases for v in vars for e in epss]
res = Vector{Any}(undef, length(jobs))
Threads.@threads :greedy for k in eachindex(jobs)
    (name, s, cfl, n, tend), v, e = jobs[k]
    res[k] = noise(s, v; eps=e, n, tend, cfl)
end
# Convergence on the harsh torus, ε = 0, t = 2.
w = PlaneWave(harsh, 2, 1, 0.3)
println("Harsh torus convergence, L2 at t = 2, n = 16…128")
for v in vars
    e = map((16, 32, 64, 128)) do n
        sim = Sim(harsh, v, n); set_state!(sim, (x, y) -> exact(w, x, y, 0.0))
        errors(sim, w, evolve!(sim, 2.0, 0.25))[1]
    end
    @printf("  %-14s %s  rates %s\n", v.name, join([@sprintf("%.2e", x) for x in e], " "),
            join([@sprintf("%.2f", log2(e[i] / e[i+1])) for i in 1:3], " "))
end
println("Random data: final E/E₀, or blow-up time")
@printf("%-6s %4s %4s %6s %-14s", "torus", "cfl", "n", "t", "variant")
foreach(e -> @printf("  ε=%-8.2f", e), epss); println()
for c in cases, v in vars
    @printf("%-6s %4.2f %4d %6.0f %-14s", c[1], c[3], c[4], c[5], v.name)
    for e in epss
        r = res[findfirst(j -> j[1] === c && j[2] === v && j[3] == e, jobs)]
        isnan(r[2]) ? @printf("  %-10.2e", r[1]) : @printf("  t=%-8.1f", r[2])
    end
    println()
end
