function getzero(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return zero(eltype(a))
end

function _DiagonalArray end

struct DiagonalArray{T,N,Diag<:AbstractVector{T},F} <: AbstractDiagonalArray{T,N}
  diag::Diag
  dims::Dims{N}
  getunstored::F
  global @inline function _DiagonalArray(
    diag::Diag, dims::Dims{N}, getunstored::F
  ) where {T,N,Diag<:AbstractVector{T},F}
    all(≥(0), dims) || throw(ArgumentError("Invalid dimensions: $dims"))
    length(diag) == minimum(dims) ||
      throw(ArgumentError("Length of diagonals doesn't match dimensions"))
    return new{T,N,Diag,F}(diag, dims, getunstored)
  end
end

function DiagonalArray{T,N}(
  diag::AbstractVector, dims::Dims{N}; getunstored=getzero
) where {T,N}
  return _DiagonalArray(convert(AbstractVector{T}, diag), dims, getunstored)
end

function DiagonalArray{T,N}(
  diag::AbstractVector, dims::Vararg{Int,N}; kwargs...
) where {T,N}
  return DiagonalArray{T,N}(diag, dims; kwargs...)
end

function DiagonalArray{T}(diag::AbstractVector, dims::Dims{N}; kwargs...) where {T,N}
  return DiagonalArray{T,N}(diag, dims; kwargs...)
end

function DiagonalArray{T}(diag::AbstractVector, dims::Vararg{Int,N}; kwargs...) where {T,N}
  return DiagonalArray{T,N}(diag, dims; kwargs...)
end

function DiagonalArray{<:Any,N}(
  diag::AbstractVector{T}, dims::Dims{N}; kwargs...
) where {T,N}
  return DiagonalArray{T,N}(diag, dims; kwargs...)
end

function DiagonalArray{<:Any,N}(
  diag::AbstractVector{T}, dims::Vararg{Int,N}; kwargs...
) where {T,N}
  return DiagonalArray{T,N}(diag, dims; kwargs...)
end

function DiagonalArray(diag::AbstractVector{T}, dims::Dims{N}; kwargs...) where {T,N}
  return DiagonalArray{T,N}(diag, dims; kwargs...)
end

function DiagonalArray(diag::AbstractVector{T}, dims::Vararg{Int,N}; kwargs...) where {T,N}
  return DiagonalArray{T,N}(diag, dims; kwargs...)
end

# Infer size from diagonal
function DiagonalArray{T,N}(diag::AbstractVector; kwargs...) where {T,N}
  return DiagonalArray{T,N}(diag, ntuple(Returns(length(diag)), N); kwargs...)
end

function DiagonalArray{<:Any,N}(diag::AbstractVector{T}; kwargs...) where {T,N}
  return DiagonalArray{T,N}(diag; kwargs...)
end

# undef
function DiagonalArray{T,N}(::UndefInitializer, dims::Dims{N}; kwargs...) where {T,N}
  return DiagonalArray{T,N}(Vector{T}(undef, minimum(dims)), dims; kwargs...)
end

function DiagonalArray{T,N}(::UndefInitializer, dims::Vararg{Int,N}; kwargs...) where {T,N}
  return DiagonalArray{T,N}(undef, dims; kwargs...)
end

function DiagonalArray{T}(::UndefInitializer, dims::Dims{N}; kwargs...) where {T,N}
  return DiagonalArray{T,N}(undef, dims; kwargs...)
end

function DiagonalArray{T}(::UndefInitializer, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(undef, dims)
end

# Axes version
function DiagonalArray{T}(
  ::UndefInitializer, axes::NTuple{N,Base.OneTo{Int}}; kwargs...
) where {T,N}
  return DiagonalArray{T,N}(undef, length.(axes); kwargs...)
end

# Minimal `AbstractArray` interface
Base.size(a::DiagonalArray) = a.dims

function Base.similar(a::DiagonalArray, elt::Type, dims::Tuple{Vararg{Int}})
  function getzero(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
    return convert(elt, a.getunstored(a, I...))
  end
  return DiagonalArray{elt}(undef, dims; getunstored=getzero)
end

# DiagonalArrays interface.
diagview(a::DiagonalArray) = a.diag

# Minimal `SparseArraysBase` interface
## SparseArraysBase.sparse_storage(a::DiagonalArray) = a.diag

# `SparseArraysBase`
# Defines similar when the output can't be `DiagonalArray`,
# such as in `reshape`.
# TODO: Put into `DiagonalArraysSparseArraysBaseExt`?
# TODO: Special case 2D to output `SparseMatrixCSC`?
## function SparseArraysBase.sparse_similar(
##   a::DiagonalArray, elt::Type, dims::Tuple{Vararg{Int}}
## )
##   return SparseArrayDOK{elt}(undef, dims, getindex_zero_function(a))
## end

## function SparseArraysBase.getindex_zero_function(a::DiagonalArray)
##   return a.zero
## end
