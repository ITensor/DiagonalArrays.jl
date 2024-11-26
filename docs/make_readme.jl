using Literate: Literate
using DiagonalArrays: DiagonalArrays

Literate.markdown(
  joinpath(pkgdir(DiagonalArrays), "examples", "README.jl"),
  joinpath(pkgdir(DiagonalArrays));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)
