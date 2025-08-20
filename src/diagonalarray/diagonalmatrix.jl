const DiagonalMatrix{T,Diag<:AbstractVector{T},Unstored<:AbstractMatrix{T}} = DiagonalArray{
  T,2,Diag,Unstored
}

# LinearAlgebra

using LinearAlgebra: LinearAlgebra

function mul_diagviews(a1, a2)
  # TODO: Compare that duals are equal, or define a function to overload.
  axes(a1, 2) == axes(a2, 1) || throw(
    DimensionMismatch(
      lazy"Incompatible dimensions for multiplication: $(axes(a1)) and $(axes(a2))"
    ),
  )
  d1 = diagview(a1)
  d2 = diagview(a2)
  l = min(length(d1), length(d2))
  d1′ = view(d1, Base.OneTo(l))
  d2′ = view(d2, Base.OneTo(l))
  return (d1′, d2′)
end

function mul!_diagviews(a_dest, a1, a2)
  axes(a_dest, 1) == axes(a1, 1) || throw(
    DimensionMismatch(
      lazy"Incompatible dimensions for multiplication: $(axes(a_dest)) and $(axes(a1))"
    ),
  )
  axes(a_dest, 2) == axes(a2, 2) || throw(
    DimensionMismatch(
      lazy"Incompatible dimensions for multiplication: $(axes(a_dest)) and $(axes(a2))"
    ),
  )
  d_dest = diagview(a_dest)
  d1, d2 = mul_diagviews(a1, a2)
  return d_dest, d1, d2
end

function Base.:*(a1::DiagonalMatrix, a2::DiagonalMatrix)
  d1, d2 = mul_diagviews(a1, a2)
  # TODO: Handle the rack-deficient case, for example:
  # δ(3, 2) * δ(2, 3)
  # Maybe pack the diagonal with zeros or allow rank-deficient DiagonalArrays?
  return DiagonalMatrix(d1 .* d2, (axes(a1, 1), axes(a2, 2)))
end
function LinearAlgebra.mul!(a_dest::DiagonalMatrix, a1::DiagonalMatrix, a2::DiagonalMatrix)
  d_dest, d1, d2 = mul!_diagviews(a_dest, a1, a2)
  # TODO: Handle the rack-deficient case.
  d_dest .= d1 .* d2
  return a_dest
end
function LinearAlgebra.mul!(
  a_dest::DiagonalMatrix, a1::DiagonalMatrix, a2::DiagonalMatrix, α::Number, β::Number
)
  d_dest, d1, d2 = mul!_diagviews(a_dest, a1, a2)
  # TODO: Handle the rack-deficient case.
  d_dest .= d1 .* d2 .* α .+ d_dest .* β
  return a_dest
end
