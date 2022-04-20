using Mimir
using Documenter

DocMeta.setdocmeta!(Mimir, :DocTestSetup, :(using Mimir); recursive=true)

makedocs(;
    modules=[Mimir],
    authors="Kiran Shila <me@kiranshila.com> and contributors",
    repo="https://github.com/kiranshila/Mimir.jl/blob/{commit}{path}#{line}",
    sitename="Mimir.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kiranshila.github.io/Mimir.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kiranshila/Mimir.jl",
    devbranch="main",
)
