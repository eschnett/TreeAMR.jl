# The TreeAMR logo: a recursively defined tree of rectangles, drawn as an AMR
# mesh.
#
# The domain is [-1, 1]². Iteration 0 is one rectangle, the trunk
# [-1/8, 1/8] × [-1, 0]. Iteration n+1 is iteration n, overlaid with three
# copies of iteration n scaled by 1/2: one translated to (0, 1/2), one rotated
# 90° counter-clockwise and translated to (-1/2, 0), and one rotated 90°
# clockwise and translated to (+1/2, 0).
#
# Each region carries a refinement level: the exterior is level 0, and a
# rectangle first drawn in iteration k is level k + 1, so the trunk is level 1.
# Where rectangles of different levels overlap (at the joints), the finer level
# wins, as in AMR: rectangles are drawn coarse to fine.
#
# Arithmetic is exact (Rational), so duplicate rectangles can be removed and
# overlaps detected without tolerances. The script writes iteration 4, the
# logo, to `docs/src/assets/logo.svg`, where Documenter picks it up. As
# previews it writes `logo-<n>.svg` for iterations n = 0:4 next to itself, and a
# PNG of each if `rsvg-convert` is on the path; those are not committed.
#
#     julia logo/logo.jl

const Q = Rational{Int}

# An axis-aligned rectangle [x0, x1] × [y0, y1]. Rotations by multiples of 90°
# keep rectangles axis-aligned, so this is closed under the maps below.
struct Rect
    x0::Q
    x1::Q
    y0::Q
    y1::Q
end

# An affine map p ↦ A p + b with A a 2×2 matrix stored as a tuple
# (a11, a12, a21, a22).
struct Affine
    A::NTuple{4,Q}
    b::NTuple{2,Q}
end

apply(T::Affine, (x, y)) = (T.A[1] * x + T.A[2] * y + T.b[1],
                            T.A[3] * x + T.A[4] * y + T.b[2])

function apply(T::Affine, r::Rect)
    p = apply(T, (r.x0, r.y0))
    q = apply(T, (r.x1, r.y1))
    return Rect(min(p[1], q[1]), max(p[1], q[1]), min(p[2], q[2]), max(p[2], q[2]))
end

const h = Q(1, 2)
const maps = [
    Affine((h, 0, 0, h), (0, h)),       # scaled, shifted up
    Affine((0, -h, h, 0), (-h, 0)),     # scaled, rotated 90° ccw, shifted left
    Affine((0, h, -h, 0), (h, 0)),      # scaled, rotated 90° cw, shifted right
]

const trunk = Rect(-1 // 8, 1 // 8, -1, 0)

# Iteration n, literally as defined, as (rectangle, level) pairs: iteration
# n-1 plus the three images of iteration n-1. An image of a rectangle is first
# drawn one iteration after the rectangle, so its level is one higher.
# `unique` keeps the first occurrence of a rectangle that occurs more than once
# (iteration n-1 already contains iteration n-2, whose images are also in
# iteration n-1), which is the one with the lowest level.
function iteration(n)
    n == 0 && return [(trunk, 1)]
    prev = iteration(n - 1)
    rects = [prev; [(apply(T, r), l + 1) for T in maps for (r, l) in prev]]
    return unique(first, rects)
end

# Whether two rectangles' interiors overlap.
overlaps(a::Rect, b::Rect) =
    max(a.x0, b.x0) < min(a.x1, b.x1) && max(a.y0, b.y0) < min(a.y1, b.y1)

# Colours by level. Level 0, the exterior, is light gray; levels 1 to 5 sample
# the plasma colour map evenly from 0 to 7/8, so that the trunk is dark blue
# and the finest level gold. Plasma's last eighth, a light yellow, is left out:
# it hardly stands out against the light gray (the values of
# ColorSchemes.plasma, written out so that the script needs no packages).
const colours = ["#eeeeee", "#0d0887", "#7201a8", "#bb3587", "#eb7555", "#fdc328"]

function svg(rects; size = 512)
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$size" height="$size" \
                viewBox="-1 -1 2 2">""")
    println(io, """<rect x="-1" y="-1" width="2" height="2" fill="$(colours[1])"/>""")
    # SVG's y axis points down; flip it so that the trunk is at the bottom.
    println(io, """<g transform="scale(1,-1)">""")
    f(v) = string(Float64(v))
    for (r, l) in sort(rects; by = last)
        println(io, """<rect x="$(f(r.x0))" y="$(f(r.y0))" \
                    width="$(f(r.x1 - r.x0))" height="$(f(r.y1 - r.y0))" \
                    fill="$(colours[l + 1])"/>""")
    end
    println(io, "</g>")
    println(io, "</svg>")
    return String(take!(io))
end

function main()
    dir = @__DIR__
    for n in 0:4
        rects = iteration(n)
        # Overlaps between distinct rectangles: a branch overlapping the
        # rectangle it grows out of, or a sibling, is expected (the joint);
        # anything else means the tree runs into itself.
        same = cross = 0
        for i in eachindex(rects), j in (i + 1):lastindex(rects)
            (a, la), (b, lb) = rects[i], rects[j]
            overlaps(a, b) || continue
            la == lb ? (same += 1) : (cross += 1)
        end
        counts = join(["$(count(==(l), last.(rects))) at level $l"
                       for l in 1:(n + 1)], ", ")
        println("iteration $n: $counts; overlapping pairs: $cross across levels, " *
                "$same within a level")
        file = joinpath(dir, "logo-$n.svg")
        write(file, svg(rects))
        if Sys.which("rsvg-convert") !== nothing
            run(`rsvg-convert -o $(joinpath(dir, "logo-$n.png")) $file`)
        end
    end
    write(joinpath(dir, "..", "docs", "src", "assets", "logo.svg"), svg(iteration(4)))
    return
end

main()
