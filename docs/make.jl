using DiagonalArrays: DiagonalArrays
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(DiagonalArrays, :DocTestSetup, :(using DiagonalArrays); recursive = true)

include("make_index.jl")

makedocs(;
    modules = [DiagonalArrays],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "DiagonalArrays.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/DiagonalArrays.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"],
    ),
    pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
    repo = "github.com/ITensor/DiagonalArrays.jl", devbranch = "main", push_preview = true
)
