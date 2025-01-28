using FillArrays: Ones

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
