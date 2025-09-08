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

@interface ::AbstractDiagonalArrayInterface function Broadcast.BroadcastStyle(type::Type)
  return DiagonalArrayStyle{ndims(type)}()
end

using Base.Broadcast: Broadcasted, broadcasted
using MapBroadcast: Mapped
# Map to a flattened broadcast expression of the diagonals of the arrays,
# also checking that the function preserves zeros.
function broadcasted_diagview(bc::Broadcasted)
  m = Mapped(bc)
  iszero(m.f(map(zero ∘ eltype, m.args)...)) || error(
    "Broadcasting DiagonalArrays with function that doesn't preserve zeros isn't supported yet.",
  )
  return broadcasted(m.f, map(diagview, m.args)...)
end
function Base.copy(bc::Broadcasted{<:DiagonalArrayStyle})
  return DiagonalArray(copy(broadcasted_diagview(bc)), axes(bc))
end
function Base.copyto!(dest::AbstractArray, bc::Broadcasted{<:DiagonalArrayStyle})
  copyto!(diagview(dest), broadcasted_diagview(bc))
  return dest
end
