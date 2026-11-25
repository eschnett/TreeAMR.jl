include("torus.jl")
w0 = Wavy(0.2, 0.3)
println("Wavy control: Laplacian of exact data, and the fill")
for n in (16, 32, 64)
    sim = Sim(w0, Variant("halfopen", :halfopen, 0, false, 6), n)
    w = PlaneWave(w0, 2, 1, 0.3)
    set_state!(sim, (x, y) -> exact(w, x, y, 0.0))
    K = workspace(sim)[1]
    for T in sim.P
        rhs_patch!(K[T.p][1], K[T.p][2], T, T.dlo, T.dhi, n, 0.0)
    end
    e = 0.0
    for T in sim.P, i in T.dlo:T.dhi, j in 0:n-1
        x, y = position(w0, T.p, i / n, j / n)
        e = max(e, abs(K[T.p][2][row(T, i), j+1] + w.ω^2 * exact(w, x, y, 0.0)[1]))
    end
    fill_ghosts!(sim)
    eg = 0.0
    for T in sim.P, i in T.ilo:T.ihi, j in 0:n-1
        T.dlo <= i <= T.dhi && continue
        x, y = position(w0, T.p, i / n, j / n)
        eg = max(eg, abs(T.u[row(T, i), j+1] - exact(w, x, y, 0.0)[1]))
    end
    @printf("  n = %3d  Laplacian %.3e  ghosts %.3e\n", n, e, eg)
end
