using DiagonalArrays: DiagonalArrays
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(DiagonalArrays, :DocTestSetup, :(using DiagonalArrays); recursive=true)

include("make_index.jl")

makedocs(;
  modules=[DiagonalArrays],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="DiagonalArrays.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/DiagonalArrays.jl",
    edit_link="main",
    assets=String[],
  ),
  pages=["Home" => "index.md", "Library" => "library.md"],
)

deploydocs(;
  repo="github.com/ITensor/DiagonalArrays.jl", devbranch="main", push_preview=true
)
