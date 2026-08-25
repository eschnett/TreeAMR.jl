using Documenter
using TreeAMR

DocMeta.setdocmeta!(TreeAMR, :DocTestSetup, :(using TreeAMR); recursive=true)

makedocs(;
    sitename="TreeAMR.jl",
    modules=[TreeAMR],
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/eschnett/TreeAMR.jl.git")
