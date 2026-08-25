using Documenter
using TreeAMR

makedocs(;
    sitename="TreeAMR.jl",
    modules=[TreeAMR],
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/eschnett/TreeAMR.jl.git")  # TODO: adjust once the repo has a home
