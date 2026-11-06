# The definitions of `rotate_tests.jl`, split off so that each test file runs
# on its own: every test process includes all of `helpers/` ("Parallel
# groups" in CODE.md).

using KernelAbstractions: @kernel, @index, @Const
using TreeAMR: oriented_neighbors, virtual_offset, real_direction, rotating_dims,
               hasrotating

const ROT_OPS4 = Operators(prolongation=4, restriction=4)

# The centerings symmetric under exchanging the plane's two dimensions:
# cell and vertex, and in 3D the two staggered along the third dimension
# only, the face and the edge normal to and along it.
function symmetric_centerings(D, rotating)
    D == 2 && return [cellcentered(2), vertexcentered(2)]
    z = outofplane(D, rotating)
    return [cellcentered(3), vertexcentered(3), facecentered(3, z), edgecentered(3, z)]
end

# Refine the level-0 leaves touching the low face of dimension 1.
lowface_rule(forest, k) = leafbox(forest, k)[1][1] == 0 && level(k) == 0 ? Refine : Keep

# A conservative advection by the rigid rotation `v = (−y, x)` in the
# plane of the seam, about its axis: the three-step right-hand side of
# `burgers.jl` with a linear reconstruction and an upwind flux. Mass that
# leaves the quadrant through the low face of `d1` enters it through the
# low face of `d2`, and the two faces' fluxes cancel only because they
# are each other's image: the flux is `F = u v`, a vector, so
# `F_{d1}(0, s) = −F_{d2}(s, 0)`, and the scheme computes it covariantly.
# The block on one seam face reconstructs from its cells and its turned
# ghosts, which are the image block's cells, and the image block
# reconstructs the same numbers in the other order; the centered slope
# is antisymmetric under that reversal, `a − b = −(b − a)` exactly, and
# the upwind choice is made by the sign of the *normal* velocity, which
# the turn negates on one side, so both pick the same state and the
# fluxes cancel bit for bit. A Rusanov flux with `|v|` would too; a flux
# that chose its upwind side by a fixed axis would not. The high faces
# are walls, a zero normal velocity there, so that nothing else enters or
# leaves, and the hook fills the outer ghosts with zeros.
@kernel function advect_flux_kernel!(flux, @Const(work), @Const(vel), ::Val{D},
                                     ::Val{GU}, ::Val{d}) where {D,GU,d}
    I = @index(Global, NTuple)                     # (i1..iD, block), face indices
    b = I[D + 1]
    c = ntuple(e -> I[e] + GU[e], Val(D))
    m1 = Base.setindex(c, c[d] - 1, d)
    m2 = Base.setindex(c, c[d] - 2, d)
    p1 = Base.setindex(c, c[d] + 1, d)
    um2, um1 = work[m2..., 1, b], work[m1..., 1, b]
    u0, up1 = work[c..., 1, b], work[p1..., 1, b]
    uL = um1 + (u0 - um2) / 4                      # centered slope, halved
    uR = u0 - (up1 - um1) / 4
    v = vel[ntuple(e -> I[e], Val(D))..., 1, b]    # G = 0: stored = face index
    z = zero(v)
    flux[ntuple(e -> I[e], Val(D))..., 1, b] = max(v, z) * uL + min(v, z) * uR
end

@kernel function advect_divergence_kernel!(du, fluxes, @Const(spacings),
                                           ::Val{D}) where {D}
    I = @index(Global, NTuple)
    b = I[D + 1]
    c = ntuple(e -> I[e], Val(D))
    acc = zero(eltype(du))
    for d in 1:D
        hi = Base.setindex(c, c[d] + 1, d)
        acc += fluxes[d][hi..., 1, b] - fluxes[d][c..., 1, b]
    end
    du[c..., 1, b] = -acc / spacings[b]
end

"""
The state, flux, velocity and schedules of the rigid-rotation advection
on a rotating quadrant; `fixup = false` drops `restrict_interfaces!`.
"""
function advection_problem(forest::Forest{D}; fixup::Bool) where {D}
    d1, d2 = TreeAMR.rotating_dims(forest)
    rotation(d) = d == d1 ? (-1,) : (1,)       # F_{d1} is −F_{d2} a turn away
    state = FieldSet(forest, 1; G=2, rotation=(1,))
    fluxes = ntuple(d -> FieldSet(forest, 1; G=0, centering=facecentered(D, d),
                                  rotation=rotation(d)), D)
    vel = ntuple(d -> FieldSet(forest, 1; G=0, centering=facecentered(D, d),
                               rotation=rotation(d)), D)
    hi = ntuple(d -> forest.extents[d][2], D)
    for d in 1:D, b in 1:nblocks(vel[d])
        # The normal velocity on each face of the closed range, both of a
        # block's own faces, zero on the high walls. (`fill_by_coordinates!`
        # fills owned points only, which leaves the high face at zero, and
        # a block whose high face carries no flux leaks.)
        for idx in CartesianIndices(ntuple(e -> 1:(forest.N + (e == d)), D))
            x = coordinates(vel[d], b, Tuple(idx))
            vel[d].work[idx, 1, b] = x[d] > hi[d] - 1e-9 ? 0.0 :
                                     d == d1 ? -x[d2] : d == d2 ? x[d1] : 0.0
        end
    end
    return (; state, fluxes, vel, fixup, schedule=GhostSchedule(state, ROT_OPS4),
            ischeds=ntuple(d -> InterfaceSchedule(fluxes[d]), D),
            spacings=block_spacings(forest, Float64),
            hook=CellBoundary((x, v, δ) -> zero(eltype(x))))
end

function advect_rhs!(du, u, p)
    D = length(p.fluxes)
    scatter!(p.state, u)
    fill_ghosts!(p.state, p.schedule; boundary=p.hook)
    ntuple(Val(D)) do d
        map_blocks!(advect_flux_kernel!, p.fluxes[d], p.fluxes[d].work, p.state.work,
                    p.vel[d].work, Val(D), Val(p.state.G), Val(d); closed=true)
        p.fixup && restrict_interfaces!(p.fluxes[d], p.ischeds[d])
        nothing
    end
    map_blocks!(advect_divergence_kernel!, p.state, statearray(du, p.state),
                map(f -> f.work, p.fluxes), p.spacings, Val(D))
    return du
end

"""
Advect a compact bump by the rigid rotation for `nsteps` SSPRK3 steps of
`dt` and return `(mass0, mass1, crossed)`: the total mass before and
after, and the mass that ended up within 30° of the low face of `d2`,
having started between 45° and 85° from it, next to the low face of
`d1`, so that it can only have got there through the seam.
"""
function advect_through_seam(forest::Forest{D}; fixup::Bool, nsteps::Int,
                             dt::Float64) where {D}
    d1, d2 = TreeAMR.rotating_dims(forest)
    p = advection_problem(forest; fixup=fixup)
    centre, w = (cos(1.13), sin(1.13)), 0.35          # 65° from the d1 axis
    fill_by_coordinates!(p.state) do x, _
        ρ2 = ((x[d1] - centre[1])^2 + (x[d2] - centre[2])^2) / w^2
        return ρ2 < 1 ? (1 - ρ2)^3 : 0.0
    end
    u = statevector(p.state)
    gather!(u, p.state)
    mass() = (scatter!(p.state, u); total_mass(p.state))
    mass0 = mass()
    k, u1, u2 = similar(u), similar(u), similar(u)
    for _ in 1:nsteps
        advect_rhs!(k, u, p)
        @. u1 = u + dt * k
        advect_rhs!(k, u1, p)
        @. u2 = 3 / 4 * u + 1 / 4 * (u1 + dt * k)
        advect_rhs!(k, u2, p)
        @. u = 1 / 3 * u + 2 / 3 * (u2 + dt * k)
    end
    mass1 = mass()
    crossed = 0.0
    for b in 1:nblocks(p.state)
        h = spacing(forest, blockkey(p.state, b))
        for idx in CartesianIndices(ntuple(_ -> 3:(forest.N + 2), D))
            x = coordinates(p.state, b, Tuple(idx))
            if x[d2] < tan(π / 6) * x[d1]
                crossed += p.state.work[idx, 1, b] * h^D
            end
        end
    end
    return mass0, mass1, crossed
end

# The orthogonal signed permutation that carries the quadrant to where a
# point lies: `r` quarter turns `R` (`R e_{d1} = e_{d2}`, `R e_{d2} =
# −e_{d1}`), then, if `mirror`, the reflection of dimension `z` at its
# low face. Written from the definition, as integer matrices.
function turn_matrix(D, (d1, d2), r::Int; mirror::Bool=false, z::Int=0)
    R = Matrix{Int}(I_(D))
    R[d1, d1], R[d2, d2], R[d2, d1], R[d1, d2] = 0, 0, 1, -1
    O = R^r
    if mirror
        M = Matrix{Int}(I_(D))
        M[z, z] = -1
        O = M * O
    end
    return O
end

I_(D) = [Int(i == j) for i in 1:D, j in 1:D]

# The value and gradient at `R^r q` (mirrored or not) of the scalar and
# the vector of `rotating_data`, from those at `q`: `s(Op) = s(p)`,
# `V(Op) = O V(p)`, and each gradient turned by `O` too. Each sum has a
# single nonzero term of weight ±1, so this is exact.
function turned_values(vq::AbstractMatrix, O::Matrix{Int})
    D = size(O, 1)
    out = similar(vq)
    comp(i, k) = i == 1 ? vq[1, k] : sum(O[i - 1, j] * vq[1 + j, k] for j in 1:D)
    for i in 1:(D + 1)
        out[i, 1] = comp(i, 1)
        for k in 1:D
            out[i, 1 + k] = sum(O[k, l] * comp(i, 1 + l) for l in 1:D)
        end
    end
    return out
end

# A complex-step first derivative of the formula `f(x, v)` along `d`:
# exact to roundoff for the polynomial data, and needing no package.
function complex_step(f, x::NTuple{D}, v, d) where {D}
    ε = 1e-20
    return imag(f(ntuple(e -> e == d ? complex(x[e], ε) : complex(x[e]), D), v)) / ε
end

const ROTWAVE_α, ROTWAVE_β = π / 2, 3π / 4

const ROTWAVE_ω = sqrt(ROTWAVE_α^2 + ROTWAVE_β^2)

# `R^r (a, b)`, `R(a, b) = (−b, a)`: exact, signs and swaps only.
function rotwave_turn((a, b), r::Int)
    for _ in 1:mod(r, 4)
        a, b = -b, a
    end
    return a, b
end

rotwave_f((a, b)) = cos(ROTWAVE_α * a + 0.3) * cos(ROTWAVE_β * b + 0.7)

rotwave_∇f((a, b)) = (-ROTWAVE_α * sin(ROTWAVE_α * a + 0.3) * cos(ROTWAVE_β * b + 0.7),
                      -ROTWAVE_β * cos(ROTWAVE_α * a + 0.3) * sin(ROTWAVE_β * b + 0.7))

function rotwave_phi(p)
    t = ntuple(r -> rotwave_f(rotwave_turn(p, r - 1)), 4)
    return (t[1] + t[3]) + (t[2] + t[4])
end

function rotwave_gradient(p)
    w = ntuple(r -> rotwave_turn(rotwave_∇f(rotwave_turn(p, r - 1)), -(r - 1)), 4)
    return ntuple(e -> (w[1][e] + w[3][e]) + (w[2][e] + w[4][e]), 2)
end

"""
The exact standing wave at time `t` as `(x, v) -> value`: with `K = 1`
the scalar `(u, ∂ₜu)`, with `K = 2` the vector `(v_{d1}, v_{d2}, ∂ₜv_{d1},
∂ₜv_{d2})` in the coordinates about the axis, `(a, b) = (x_{d1}, x_{d2})`.
"""
function rotwave_exact(K::Int, (d1, d2), t)
    c, s = cos(ROTWAVE_ω * t), -ROTWAVE_ω * sin(ROTWAVE_ω * t)
    return function (x, v)
        p = (x[d1], x[d2])
        k = v > K ? v - K : v
        value = K == 1 ? rotwave_phi(p) : rotwave_gradient(p)[k]
        return (v > K ? s : c) * value
    end
end

"""
The rotation map of the wave's state: the scalar's `(1, 2)`, or the
vector's from `vector_rotation` (the oracle's, written from `v(Rp) =
Rv(p)`), its components numbered from 1, then the same for `∂ₜv`.
"""
function rotwave_rotation(K::Int, rotating)
    K == 1 && return [1, 2]
    m = [sign(q) * (abs(q) - 1) for q in vector_rotation(2, rotating)[2:end]]
    return [m; [sign(q) * (abs(q) + 2) for q in m]]
end

# The Laplacian of each of `K` components with the two neighbours added
# first, `(u₊ + u₋) − 2u₀`, so that a point and its image under the turn,
# whose neighbours are each other's in the other order, compute the same
# bits; `wave_rhs_kernel!` subtracts first, which only roundoff tells
# apart.
@kernel function rotwave_kernel!(du, @Const(work), @Const(spacings), ::Val{D},
                                 ::Val{G}, ::Val{K}) where {D,G,K}
    I = @index(Global, NTuple)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))
    c = ntuple(d -> I[d] + G[d], Val(D))
    h = spacings[b]
    for k in 1:K
        u0 = work[c..., k, b]
        lap = zero(eltype(du))
        for d in 1:D
            up = Base.setindex(c, c[d] + 1, d)
            um = Base.setindex(c, c[d] - 1, d)
            lap += (work[up..., k, b] + work[um..., k, b]) - 2 * u0
        end
        du[inner..., k, b] = work[c..., K + k, b]
        du[inner..., K + k, b] = lap / (h * h)
    end
end

function rotwave_rhs!(du, u, p, t)
    scatter!(p.fs, u)
    fill_ghosts!(p.fs, p.schedule;
                 boundary=boundary_by_coordinates(rotwave_exact(p.K, p.rotating, t)))
    map_blocks!(rotwave_kernel!, p.fs, statearray(du, p.fs), p.fs.work, p.spacings,
                Val(2), p.valG, Val(p.K))
    return nothing
end

"""
Evolve the standing wave with `K` components on the quadrant of
`quadrant_and_full(Val(2); rotating)` or, with `full = true`, on the full
plane, by fixed-step RK4 to a quarter period, and return the errors
against the exact solution, the step count, and the field set holding
the final state with its ghosts filled.
"""
function rotating_wave(; N, C, K, rotating=(1, 2), full::Bool, p=4, cfl=0.25,
                       periods=0.25)
    quad, plane = quadrant_and_full(Val(2); rotating=rotating, other=:periodic, N=N)
    forest = full ? plane : quad
    G = ghosts_for(C, p)
    rotation = full ? nothing : rotwave_rotation(K, rotating)
    fs = FieldSet(forest, 2K; G=G, centering=C, rotation=rotation)
    schedule = GhostSchedule(fs, Operators(prolongation=p, restriction=p))
    problem = (; fs, schedule, K, rotating, valG=Val(G),
               spacings=block_spacings(forest, Float64))
    fill_by_coordinates!(rotwave_exact(K, rotating, 0.0), fs)
    u0 = statevector(fs)
    gather!(u0, fs)
    h = minimum_spacing(forest)
    t_end = periods * 2π / ROTWAVE_ω
    nsteps = ceil(Int, t_end / (cfl * h))
    prob = ODEProblem(rotwave_rhs!, u0, (0.0, t_end), problem)
    sol = solve(prob, RK4(); dt=t_end / nsteps, adaptive=false, save_everystep=false)
    exact = FieldSet(forest, 2K; G=G, centering=C, rotation=rotation)
    fill_by_coordinates!(rotwave_exact(K, rotating, t_end), exact)
    uexact = statevector(exact)
    gather!(uexact, exact)
    err = sol.u[end] .- uexact
    scatter!(fs, sol.u[end])
    fill_ghosts!(fs, schedule; boundary=boundary_by_coordinates(rotwave_exact(K, rotating,
                                                                                t_end)))
    return (l2=volume_weighted_norm(fs, err), linf=volume_weighted_norm(fs, err; p=Inf),
            h=h, nsteps=nsteps, nblocks=nleaves(forest), fs=fs)
end

"""
The worst difference between the vertex-centered quadrant's two owned
seam planes, the points `(0, s)` on the low face of `d1` and `(s, 0)` on
the low face of `d2`, which are one point under the turn: the first must
be the second's values through the signed map. Returns `(worst, npairs)`,
`worst` compared with `==` semantics; the axis itself is skipped.
"""
function seam_plane_mismatch(fs::FieldSet{T,2}) where {T}
    d1, d2 = rotating_dims(fs.forest)
    planes = (Dict{T,Vector{T}}(), Dict{T,Vector{T}}())
    for b in 1:nblocks(fs), (side, d, e) in ((1, d1, d2), (2, d2, d1))
        block_extent(fs.forest, blockkey(fs, b))[d][1] == 0 || continue
        for j in 1:(fs.forest.N)
            idx = (0, 0)
            idx = Base.setindex(idx, fs.G[d] + 1, d)
            idx = Base.setindex(idx, fs.G[e] + j, e)
            s = coordinates(fs, b, idx)[e]
            s == 0 && continue
            planes[side][s] = fs.work[idx..., :, b]
        end
    end
    @assert sort!(collect(keys(planes[1]))) == sort!(collect(keys(planes[2])))
    worst = 0.0
    for (s, here) in planes[1]
        there = planes[2][s]
        for v in eachindex(here)
            q = fs.rotation[v]
            turned = sign(q) * there[abs(q)]
            worst = here[v] == turned ? worst : max(worst, abs(here[v] - turned))
        end
    end
    return worst, length(planes[1])
end

"""
How far the full plane's own solution is from covariant: the worst
difference, over its owned points `p` in the quadrant `x_{d1}, x_{d2} ≥ 0`
and each turn `r = 1, 2, 3`, between its values at `Rʳp`, where those are
owned too, and its values at `p` turned by the signed map `rotation`.
"""
function turned_defect(fs::FieldSet{T,2}, rotation) where {T}
    d1, d2 = 1, 2
    owned = Dict{NTuple{2,T},Vector{T}}()
    for b in 1:nblocks(fs)
        for i in CartesianIndices(ntuple(d -> (fs.G[d] + 1):(fs.G[d] + fs.forest.N), 2))
            owned[coordinates(fs, b, Tuple(i))] = fs.work[i, :, b]
        end
    end
    worst = 0.0
    for (p, u) in owned, r in 1:3
        (p[d1] >= 0 && p[d2] >= 0) || continue
        there = get(owned, rotwave_turn(p, r), nothing)
        there === nothing && continue
        for _ in 1:r
            u = [sign(q) * u[abs(q)] for q in rotation]
        end
        worst = max(worst, maximum(abs.(there .- u)))
    end
    return worst
end
