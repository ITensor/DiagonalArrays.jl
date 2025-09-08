# TODO: Turn these into `@interface ::AbstractDiagonalArrayInterface` functions.

using LinearAlgebra: LinearAlgebra

diaglength(a::AbstractArray{<:Any,0}) = 1

function diaglength(a::AbstractArray)
  return minimum(size(a))
end

@inline function isdiagindex(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N}
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
function DiagCartesianIndices(axes::Tuple{AbstractUnitRange,Vararg{AbstractUnitRange}})
  # Check the ranges are one-based.
  @assert all(isone, first.(axes))
  return DiagCartesianIndices{length(axes)}(minimum(length.(axes)))
end
function DiagCartesianIndices(dims::Tuple{Int,Vararg{Int}})
  return DiagCartesianIndices(Base.OneTo.(dims))
end
function DiagCartesianIndices(dims::Tuple{})
  return DiagCartesianIndices{0}(0)
end
function DiagCartesianIndices(a::AbstractArray)
  return DiagCartesianIndices(axes(a))
end
Base.size(I::DiagCartesianIndices) = (I.diaglength,)
function Base.getindex(I::DiagCartesianIndices{N}, i::Int) where {N}
  return CartesianIndex(ntuple(Returns(i), N))
end

function checkdiagbounds(::Type{Bool}, a::AbstractArray, i::Integer)
  Base.require_one_based_indexing(a)
  return i ∈ 1:diaglength(a)
end
function checkdiagbounds(a::AbstractArray, i::Integer)
  checkdiagbounds(Bool, a, i) || throw(BoundsError(a, ntuple(Returns(i), ndims(a))))
  return nothing
end

# Convert a linear index along the diagonal to the corresponding
# CartesianIndex.
@inline function diagindex(a::AbstractArray, i::Integer)
  @boundscheck checkdiagbounds(a, i)
  return CartesianIndex(ntuple(Returns(i), ndims(a)))
end

function diagindices(a::AbstractArray)
  return diagindices(IndexStyle(a), a)
end
function diagindices(::IndexLinear, a::AbstractArray)
  maxdiag = isempty(a) ? 0 : @inbounds LinearIndices(a)[diagindex(a, diaglength(a))]
  return 1:diagstride(a):maxdiag
end
function diagindices(::IndexCartesian, a::AbstractArray)
  return DiagCartesianIndices(a)
end

function diagindices(a::AbstractArray{<:Any,0})
  return Base.OneTo(1)
end

function diagview(a::AbstractArray)
  return @view a[diagindices(a)]
end

using LinearAlgebra: Diagonal
diagview(a::Diagonal) = a.diag

function getdiagindex(a::AbstractArray, i::Integer)
  return diagview(a)[i]
end

function setdiagindex!(a::AbstractArray, v, i::Integer)
  diagview(a)[i] = v
  return a
end

function getdiagindices(a::AbstractArray, I)
  # TODO: Should this be a view?
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

Return a diagonal matrix from a vector `v`.
This is an extension of `LinearAlgebra.Diagonal`, designed to avoid
the implication of the output type.
Defaults to `Diagonal(v)`.
"""
diagonal(v::AbstractVector) = LinearAlgebra.Diagonal(v)

"""
    diagonal(m::AbstractMatrix) -> AbstractMatrix

Return a diagonal matrix from a matrix `m` where the diagonal
values are copied from the diagonal of `m`.
This is an extension of `LinearAlgebra.Diagonal`, designed to avoid
the implication of the output type.
Defaults to `diagonal(copy(diagview(m)))`, which in general is
equivalent to `Diagonal(m)`.
"""
diagonal(m::AbstractMatrix) = diagonal(copy(diagview(m)))

"""
    diagonaltype(::AbstractVector) -> Type{<:AbstractMatrix}
    diagonaltype(::Type{<:AbstractVector}) -> Type{<:AbstractMatrix}
    diagonaltype(::AbstractMatrix) -> Type{<:AbstractMatrix}
    diagonaltype(::Type{<:AbstractMatrix}) -> Type{<:AbstractMatrix}

Return the type of diagonal matrix that would be created from a vector or matrix
using the [`diagonal`](@ref) function.
"""
diagonaltype

diagonaltype(v::AbstractVector) = diagonaltype(typeof(v))
diagonaltype(V::Type{<:AbstractVector}) = Base.promote_op(diagonal, V)
diagonaltype(m::AbstractMatrix) = diagonaltype(typeof(m))
diagonaltype(M::Type{<:AbstractMatrix}) = Base.promote_op(diagonal, M)
