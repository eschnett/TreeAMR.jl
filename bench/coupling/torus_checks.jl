include("torus.jl")
const VARIANTS = [Variant("copy", :copy, 0, false, 6),
                  Variant("halfopen", :halfopen, 0, false, 6),
                  Variant("halfopen+avg", :halfopen, 0, true, 6),
                  Variant("overlap0", :overlap, 0, false, 6),
                  Variant("overlap2", :overlap, 2, false, 6)]
smooth = Setup(1.0, 0.0, 0.0)
kinked = Setup(0.8, 0.4, 0.3)

# 1. The discrete Laplacian of exact data (ghosts exact too) converges at 4.
println("Laplacian of exact data, kinked setup, max error over owned points")
for n in (16, 32, 64)
    sim = Sim(kinked, VARIANTS[4], n)
    w = PlaneWave(kinked, 2, 1, 0.3)
    set_state!(sim, (x, y) -> exact(w, x, y, 0.0))
    K = workspace(sim)[1]
    for T in sim.P
        rhs_patch!(K[T.p][1], K[T.p][2], T, T.dlo, T.dhi, n, 0.0)
    end
    e = 0.0
    for T in sim.P, i in T.dlo:T.dhi, j in 0:n-1
        x, y = position(kinked, T.p, i / n, j / n)
        e = max(e, abs(K[T.p][2][row(T, i), j+1] + w.ω^2 * exact(w, x, y, 0.0)[1]))
    end
    @printf("  n = %3d  %.3e\n", n, e)
end

# 2. Interpolated ghosts of exact data, max error per variant.
println("Ghost fill of exact data, kinked setup, max ghost error")
for var in VARIANTS[2:end], n in (16, 32, 64)
    sim = Sim(kinked, var, n)
    w = PlaneWave(kinked, 2, 1, 0.3)
    set_state!(sim, (x, y) -> exact(w, x, y, 0.0))
    fill_ghosts!(sim)
    e = 0.0
    for T in sim.P, i in T.ilo:T.ihi, j in 0:n-1
        T.dlo <= i <= T.dhi && continue
        x, y = position(kinked, T.p, i / n, j / n)
        e = max(e, abs(T.u[row(T, i), j+1] - exact(w, x, y, 0.0)[1]))
    end
    @printf("  %-14s n = %3d  %.3e\n", var.name, n, e)
end

# 3. On the smooth setup every coupling is the periodic brick.
println("Smooth setup: L2 error at t = 1")
for var in VARIANTS, n in (16, 32, 64)
    sim = Sim(smooth, var, n)
    w = PlaneWave(smooth, 2, 1, 0.3)
    set_state!(sim, (x, y) -> exact(w, x, y, 0.0))
    t = evolve!(sim, 1.0, 0.25)
    @printf("  %-14s n = %3d  %.6e\n", var.name, n, errors(sim, w, t)[1])
end
