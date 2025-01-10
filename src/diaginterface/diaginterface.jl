# TODO: Turn these into `@interface ::AbstractDiagonalArrayInterface` functions.

using LinearAlgebra: LinearAlgebra

diaglength(a::AbstractArray{<:Any,0}) = 1

function diaglength(a::AbstractArray)
  return minimum(size(a))
end

function isdiagindex(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N}
  @boundscheck checkbounds(a, I)
  return allequal(Tuple(I))
end

function diagstride(a::AbstractArray)
  s = 1
  p = 1
  for i in 1:(ndims(a) - 1)
    p *= size(a, i)
    s += p
  end
  return s
end

# Iterator over the diagonal cartesian indices.
# For an AbstractArray `a`, `DiagCartesianIndices(a)` is equivalent
# to `@view CartesianIndices(a)[diagindices(a)]` but should be
# faster because it avoids conversions from linear to cartesian indices.
struct DiagCartesianIndices{N} <: AbstractVector{CartesianIndex{N}}
  diaglength::Int
end
function DiagCartesianIndices(axes::Tuple{Vararg{AbstractUnitRange}})
  # Check the ranges are one-based.
  @assert all(isone, first.(axes))
  return DiagCartesianIndices{length(axes)}(minimum(length.(axes)))
end
function DiagCartesianIndices(dims::Tuple{Vararg{Int}})
  return DiagCartesianIndices(Base.OneTo.(dims))
end
function DiagCartesianIndices(a::AbstractArray)
  return DiagCartesianIndices(axes(a))
end
Base.size(I::DiagCartesianIndices) = (I.diaglength,)
function Base.getindex(I::DiagCartesianIndices{N}, i::Int) where {N}
  return CartesianIndex(ntuple(Returns(i), N))
end

function diagindices(a::AbstractArray)
  return diagindices(IndexStyle(a), a)
end
function diagindices(::IndexLinear, a::AbstractArray)
  maxdiag = LinearIndices(a)[CartesianIndex(ntuple(Returns(diaglength(a)), ndims(a)))]
  return 1:diagstride(a):maxdiag
end
function diagindices(::IndexCartesian, a::AbstractArray)
  return DiagCartesianIndices(a)
  # TODO: Define a special iterator for this, i.e. `DiagCartesianIndices`?
  return Iterators.map(
    i -> CartesianIndex(ntuple(Returns(i), ndims(a))), Base.OneTo(diaglength(a))
  )
end

function diagindices(a::AbstractArray{<:Any,0})
  return Base.OneTo(1)
end

function diagview(a::AbstractArray)
  return @view a[diagindices(a)]
end

function getdiagindex(a::AbstractArray, i::Integer)
  return diagview(a)[i]
end

function setdiagindex!(a::AbstractArray, v, i::Integer)
  diagview(a)[i] = v
  return a
end

function getdiagindices(a::AbstractArray, I)
  return @view diagview(a)[I]
end

function getdiagindices(a::AbstractArray, I::Colon)
  return diagview(a)
end

function setdiagindices!(a::AbstractArray, v, i::Colon)
  diagview(a) .= v
  return a
end

"""
    diagonal(v::AbstractVector) -> AbstractMatrix

Return a diagonal matrix from a vector `v`. This is a replacement of `LinearAlgebra.Diagonal`
that does not imply an output type. Defaults to `Diagonal(v)`.
"""
diagonal(A::AbstractVector) = LinearAlgebra.Diagonal(A)
