using FillArrays: Ones

function delta(
  elt::Type, ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}}
)
  return DiagonalArray(Ones{elt}(minimum(length, ax)), ax)
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
  return DiagonalArray(Ones{elt}(minimum(size)), size)
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
