# TODO: Turn these into `@interface ::AbstractDiagonalArrayInterface` functions.

diagview(a::AbstractDiagonalArray) = throw(MethodError(diagview, Tuple{typeof(a)}))

using DerivableInterfaces: DerivableInterfaces, @interface
using SparseArraysBase:
  SparseArraysBase, AbstractSparseArrayInterface, AbstractSparseArrayStyle

abstract type AbstractDiagonalArrayInterface{N} <: AbstractSparseArrayInterface{N} end

struct DiagonalArrayInterface{N} <: AbstractDiagonalArrayInterface{N} end
DiagonalArrayInterface{M}(::Val{N}) where {M,N} = DiagonalArrayInterface{N}()
DiagionalArrayInterface(::Val{N}) where {N} = DiagonalArrayInterface{N}()
DiagonalArrayInterface() = DiagonalArrayInterface{Any}()

function Base.similar(::AbstractDiagonalArrayInterface, elt::Type, ax::Tuple)
  return similar(DiagonalArray{elt}, ax)
end
function DerivableInterfaces.interface(::Type{<:AbstractDiagonalArray{<:Any,N}}) where {N}
  return DiagonalArrayInterface{N}()
end

abstract type AbstractDiagonalArrayStyle{N} <: AbstractSparseArrayStyle{N} end

function DerivableInterfaces.interface(::Type{<:AbstractDiagonalArrayStyle{N}}) where {N}
  return DiagonalArrayInterface{N}()
end

struct DiagonalArrayStyle{N} <: AbstractDiagonalArrayStyle{N} end

DiagonalArrayStyle{M}(::Val{N}) where {M,N} = DiagonalArrayStyle{N}()

@interface ::AbstractDiagonalArrayInterface function Broadcast.BroadcastStyle(type::Type)
  return DiagonalArrayStyle{ndims(type)}()
end

function SparseArraysBase.isstored(
  a::AbstractDiagonalArray{<:Any,N}, I::Vararg{Int,N}
) where {N}
  return allequal(I)
end
function SparseArraysBase.getstoredindex(
  a::AbstractDiagonalArray{<:Any,N}, I::Vararg{Int,N}
) where {N}
  # TODO: Make this check optional, define `checkstored` like `checkbounds`
  # in SparseArraysBase.jl.
  # allequal(I) || error("Not a diagonal index.")
  return getdiagindex(a, first(I))
end
function SparseArraysBase.setstoredindex!(
  a::AbstractDiagonalArray{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  # TODO: Make this check optional, define `checkstored` like `checkbounds`
  # in SparseArraysBase.jl.
  # allequal(I) || error("Not a diagonal index.")
  setdiagindex!(a, value, first(I))
  return a
end
function SparseArraysBase.eachstoredindex(::IndexCartesian, a::AbstractDiagonalArray)
  return diagindices(a)
end

# Fix ambiguity error with SparseArraysBase.
function Base.getindex(a::AbstractDiagonalArray, I::DiagIndices)
  # TODO: Use `@interface` rather than `invoke`.
  return invoke(getindex, Tuple{AbstractArray,DiagIndices}, a, I)
end
# Fix ambiguity error with SparseArraysBase.
function Base.getindex(a::AbstractDiagonalArray, I::DiagIndex)
  # TODO: Use `@interface` rather than `invoke`.
  return invoke(getindex, Tuple{AbstractArray,DiagIndex}, a, I)
end
# Fix ambiguity error with SparseArraysBase.
function Base.setindex!(a::AbstractDiagonalArray, value, I::DiagIndices)
  # TODO: Use `@interface` rather than `invoke`.
  return invoke(setindex!, Tuple{AbstractArray,Any,DiagIndices}, a, value, I)
end
# Fix ambiguity error with SparseArraysBase.
function Base.setindex!(a::AbstractDiagonalArray, value, I::DiagIndex)
  # TODO: Use `@interface` rather than `invoke`.
  return invoke(setindex!, Tuple{AbstractArray,Any,DiagIndex}, a, value, I)
end

## SparseArraysBase.StorageIndex(i::DiagIndex) = StorageIndex(index(i))

## function Base.getindex(a::AbstractDiagonalArray, i::DiagIndex)
##   return a[StorageIndex(i)]
## end

## function Base.setindex!(a::AbstractDiagonalArray, value, i::DiagIndex)
##   a[StorageIndex(i)] = value
##   return a
## end

## SparseArraysBase.StorageIndices(i::DiagIndices) = StorageIndices(indices(i))

## function Base.getindex(a::AbstractDiagonalArray, i::DiagIndices)
##   return a[StorageIndices(i)]
## end

## function Base.setindex!(a::AbstractDiagonalArray, value, i::DiagIndices)
##   a[StorageIndices(i)] = value
##   return a
## end
