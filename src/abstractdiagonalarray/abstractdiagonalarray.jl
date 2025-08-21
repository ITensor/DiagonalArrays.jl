using SparseArraysBase: AbstractSparseArray

abstract type AbstractDiagonalArray{T,N} <: AbstractSparseArray{T,N} end
const AbstractDiagonalMatrix{T} = AbstractDiagonalArray{T,2}
const AbstractDiagonalVector{T} = AbstractDiagonalArray{T,1}

using LinearAlgebra: LinearAlgebra, ishermitian, isposdef, issymmetric
LinearAlgebra.ishermitian(a::AbstractDiagonalMatrix{<:Real}) = issquare(a)
function LinearAlgebra.ishermitian(a::AbstractDiagonalMatrix{<:Number})
  return issquare(a) && isreal(diagview(a))
end
function LinearAlgebra.ishermitian(a::AbstractDiagonalMatrix)
  return issquare(a) && all(ishermitian, diagview(a))
end
LinearAlgebra.issymmetric(a::AbstractDiagonalMatrix{<:Number}) = issquare(a)
function LinearAlgebra.issymmetric(a::AbstractDiagonalMatrix)
  return issquare(a) && all(issymmetric, diagview(a))
end
function LinearAlgebra.isposdef(a::AbstractDiagonalMatrix)
  return issquare(a) && all(isposdef, diagview(a))
end
