using Documenter
using KernelAbstractions
using TreeAMR

DocMeta.setdocmeta!(TreeAMR, :DocTestSetup, :(using TreeAMR); recursive=true)

makedocs(;
    sitename="TreeAMR.jl",
    modules=[TreeAMR],
    # One page per layer rather than one page for everything: the single page
    # had reached 178 KiB of the 200 KiB at which Documenter's HTML writer
    # (`size_threshold`) fails the build.
    pages=[
        "Home" => "index.md",
        "API reference" => [
            "Tree and geometry" => "api/tree.md",
            "Storage" => "api/storage.md",
            "Ghost exchange and conservation" => "api/exchange.md",
            "ODE coupling" => "api/ode.md",
            "Regridding" => "api/regrid.md",
            "Point interpolation" => "api/interpolate.md",
            "Internals" => "api/internals.md",
            "Index" => "api/genindex.md",
        ],
    ],
)

deploydocs(; repo="github.com/eschnett/TreeAMR.jl.git")
