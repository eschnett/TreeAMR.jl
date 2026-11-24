include("torus.jl")
# Random data, energy ratio E(t)/E(0): its maximum, its value at the end,
# and the time it first exceeds 10³ (blow-up), if it does.
function noise(s, var; eps, n, tend)
    sim = Sim(s, var, n; eps)
    rng = Xoshiro(17)
    for T in sim.P
        T.u .= 1e-3 .* randn(rng, size(T.u)); T.v .= 1e-3 .* randn(rng, size(T.v))
    end
    E0 = energy(sim); Emax = E0; tblow = NaN
    every = max(1, round(Int, 1.0 / timestep(sim, 0.25)))
    evolve!(sim, tend, 0.25; every, monitor = t -> begin
        E = energy(sim); Emax = max(Emax, E)
        ok = isfinite(E) && E < 1e3 * E0
        ok || (tblow = t)
        ok
    end)
    return Emax / E0, energy(sim) / E0, tblow
end

