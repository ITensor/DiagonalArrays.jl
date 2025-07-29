using FillArrays: Ones, OnesVector

const Delta{T,N,V<:OnesVector{T},Axes} = DiagonalArray{T,N,V,Axes}
function Delta{T}(
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}}
) where {T}
  uniquelens = unique(length, ax)
  if !isone(length(uniquelens))
    throw(ArgumentError("All axes must have the same length for Delta."))
  end
  return DiagonalArray{T}(Ones{T}(only(uniquelens)), ax)
end
function Delta{T}(
  ax1::AbstractUnitRange{<:Integer}, ax_rest::AbstractUnitRange{<:Integer}...
) where {T}
  return Delta{T}((ax1, ax_rest...))
end
function Delta{T}(sz::Tuple{Integer,Vararg{Integer}}) where {T}
  return Delta{T}(map(Base.OneTo, sz))
end
function Delta{T}(sz1::Integer, sz_rest::Integer...) where {T}
  return Delta{T}((sz1, sz_rest...))
end
function Delta{T}(ax::Tuple{}) where {T}
  return DiagonalArray{T}(Ones{T}(0), ax)
end

function delta(
  elt::Type, ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}}
)
  return Delta{elt}(ax)
end
function δ(
  elt::Type, ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}}
)
  return delta(elt, ax)
end
function delta(ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}})
  return delta(Float64, ax)
end
function δ(ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}})
  return delta(Float64, ax)
end

function delta(
  elt::Type, ax1::AbstractUnitRange{<:Integer}, axs::AbstractUnitRange{<:Integer}...
)
  return delta(elt, (ax1, axs...))
end
function δ(
  elt::Type, ax1::AbstractUnitRange{<:Integer}, axs::AbstractUnitRange{<:Integer}...
)
  return delta(elt, (ax1, axs...))
end
function delta(ax1::AbstractUnitRange{<:Integer}, axs::AbstractUnitRange{<:Integer}...)
  return delta(Float64, (ax1, axs...))
end
function δ(ax1::AbstractUnitRange{<:Integer}, axs::AbstractUnitRange{<:Integer}...)
  return delta(Float64, (ax1, axs...))
end

function delta(elt::Type, size::Tuple{Vararg{Int}})
  return Delta{elt}(size)
end
function δ(elt::Type, size::Tuple{Vararg{Int}})
  return delta(elt, size)
end
function delta(elt::Type, size::Int...)
  return delta(elt, size)
end
function δ(elt::Type, size::Int...)
  return delta(elt, size...)
end
function delta(size::Tuple{Vararg{Int}})
  return delta(Float64, size)
end
function δ(size::Tuple{Vararg{Int}})
  return delta(size)
end
function delta(size::Int...)
  return delta(size)
end
function δ(size::Int...)
  return delta(size...)
end
