using FillArrays: Zeros
using SparseArraysBase: Unstored, unstored

function _DiagonalArray end

struct DiagonalArray{T,N,Diag<:AbstractVector{T},Unstored<:AbstractArray{T,N}} <:
       AbstractDiagonalArray{T,N}
  diag::Diag
  unstored::Unstored
  global @inline function _DiagonalArray(
    diag::Diag, unstored::Unstored
  ) where {T,N,Diag<:AbstractVector{T},Unstored<:AbstractArray{T,N}}
    length(diag) == minimum(size(unstored)) ||
      throw(ArgumentError("Length of diagonals doesn't match dimensions"))
    return new{T,N,Diag,Unstored}(diag, unstored)
  end
end

SparseArraysBase.unstored(a::DiagonalArray) = a.unstored
Base.size(a::DiagonalArray) = size(unstored(a))
Base.axes(a::DiagonalArray) = axes(unstored(a))

function DiagonalArray(::UndefInitializer, unstored::Unstored)
  return _DiagonalArray(
    Vector{eltype(unstored)}(undef, minimum(size(unstored))), parent(unstored)
  )
end

# Constructors accepting axes.
function DiagonalArray(
  diag::AbstractVector{T},
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
) where {T}
  return DiagonalArray{T,length(ax)}(diag, ax)
end
function DiagonalArray(
  diag::AbstractVector,
  ax1::AbstractUnitRange{<:Integer},
  axs::AbstractUnitRange{<:Integer}...,
)
  return DiagonalArray(diag, (ax1, axs...))
end
function DiagonalArray{T,N}(
  diag::AbstractVector,
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
) where {T,N}
  N == length(ax) || throw(ArgumentError("Wrong number of axes"))
  return _DiagonalArray(convert(AbstractVector{T}, diag), Zeros{T}(ax))
end
function DiagonalArray{T,N}(
  diag::AbstractVector,
  ax1::AbstractUnitRange{<:Integer},
  axs::AbstractUnitRange{<:Integer}...,
) where {T,N}
  return DiagonalArray{T,N}(diag, (ax1, axs...))
end

# undef constructors accepting axes.
function DiagonalArray{T,N}(
  ::UndefInitializer,
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
) where {T,N}
  return DiagonalArray{T,N}(Vector{T}(undef, minimum(length, ax)), ax)
end
function DiagonalArray{T,N}(
  ::UndefInitializer,
  ax1::AbstractUnitRange{<:Integer},
  axs::AbstractUnitRange{<:Integer}...,
) where {T,N}
  return DiagonalArray{T,N}(undef, (ax1, axs...))
end
function DiagonalArray{T}(
  ::UndefInitializer,
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
) where {T}
  return DiagonalArray{T,length(ax)}(undef, ax)
end
function DiagonalArray{T}(
  ::UndefInitializer,
  ax1::AbstractUnitRange{<:Integer},
  axs::AbstractUnitRange{<:Integer}...,
) where {T,N}
  return DiagonalArray{T}(undef, (ax1, axs...))
end

function DiagonalArray{T,N}(diag::AbstractVector, dims::Dims{N}) where {T,N}
  return _DiagonalArray(convert(AbstractVector{T}, diag), Zeros{T}(dims))
end

function DiagonalArray{T,N}(diag::AbstractVector, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray{T}(diag::AbstractVector, dims::Dims{N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray{T}(diag::AbstractVector, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray{<:Any,N}(diag::AbstractVector{T}, dims::Dims{N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray{<:Any,N}(diag::AbstractVector{T}, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray(diag::AbstractVector{T}, dims::Dims{N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray(diag::AbstractVector{T}, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

# Infer size from diagonal
function DiagonalArray{T,N}(diag::AbstractVector) where {T,N}
  return DiagonalArray{T,N}(diag, ntuple(Returns(length(diag)), N))
end

function DiagonalArray{<:Any,N}(diag::AbstractVector{T}) where {T,N}
  return DiagonalArray{T,N}(diag)
end

# undef
function DiagonalArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
  return DiagonalArray{T,N}(Vector{T}(undef, minimum(dims)), dims)
end

function DiagonalArray{T,N}(::UndefInitializer, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(undef, dims)
end

function DiagonalArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N}
  return DiagonalArray{T,N}(undef, dims)
end

function DiagonalArray{T}(::UndefInitializer, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(undef, dims)
end

# Axes version
function DiagonalArray{T}(::UndefInitializer, axes::NTuple{N,Base.OneTo{Int}}) where {T,N}
  return DiagonalArray{T,N}(undef, length.(axes))
end

function Base.similar(a::DiagonalArray, unstored::Unstored)
  return DiagonalArray(undef, unstored)
end

# DiagonalArrays interface.
diagview(a::DiagonalArray) = a.diag
