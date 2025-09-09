# TODO: Define `TensorProducts.dual`.
dual(x) = x
issquare(a::AbstractMatrix) = (axes(a, 1) == dual(axes(a, 2)))
# Like `LinearAlgebra.checksquare` but based on `DiagonalArrays.issquare`,
# which checks the axes and allows customizing to check that the
# codomain is the dual of the domain.
# Returns the codomain if the check passes.
function checksquare(a::AbstractMatrix)
  issquare(a) || throw(DimensionMismatch(lazy"matrix is not square: axes are $(axes(a))"))
  return axes(a, 1)
end
