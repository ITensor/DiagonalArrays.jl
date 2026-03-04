using DiagonalArrays: DiagonalArrays
using Documenter: Documenter, DocMeta, deploydocs, makedocs
using ITensorFormatter: ITensorFormatter

DocMeta.setdocmeta!(
    DiagonalArrays,
    :DocTestSetup,
    :(using DiagonalArrays);
    recursive = true
)

ITensorFormatter.make_index!(pkgdir(DiagonalArrays))

makedocs(;
    modules = [DiagonalArrays],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "DiagonalArrays.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/DiagonalArrays.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"]
    ),
    pages = ["Home" => "index.md", "Reference" => "reference.md"]
)

deploydocs(;
    repo = "github.com/ITensor/DiagonalArrays.jl", devbranch = "main", push_preview = true
)
