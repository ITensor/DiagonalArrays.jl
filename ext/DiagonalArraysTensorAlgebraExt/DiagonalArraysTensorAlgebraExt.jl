module DiagonalArraysTensorAlgebraExt

using DiagonalArrays: Delta
using FillArrays: Eye
using TensorAlgebra:
  TensorAlgebra,
  AbstractBlockPermutation,
  BlockedTrivialPermutation,
  BlockedTuple,
  FusionStyle,
  fuseaxes,
  matricize

struct DeltaFusion <: FusionStyle end
TensorAlgebra.FusionStyle(::Delta) = DeltaFusion()
function matricize_delta(a::AbstractArray, biperm::AbstractBlockPermutation{2})
  ax = fuseaxes(axes(a), biperm)
  return Eye{eltype(a)}(ax)
end
function TensorAlgebra.matricize(
  ::DeltaFusion, a::AbstractArray, biperm::AbstractBlockPermutation{2}
)
  return matricize_delta(a, biperm)
end
function TensorAlgebra.matricize(
  ::DeltaFusion, a::AbstractArray, biperm::BlockedTrivialPermutation{2}
)
  return matricize_delta(a, biperm)
end

function TensorAlgebra.unmatricize(::DeltaFusion, a::Eye, ax::BlockedTuple{2})
  length(a) == prod(length, ax) || throw(DimensionMismatch("reshape sizes don't match"))
  return Delta{eltype(a)}(Tuple(ax))
end

end
