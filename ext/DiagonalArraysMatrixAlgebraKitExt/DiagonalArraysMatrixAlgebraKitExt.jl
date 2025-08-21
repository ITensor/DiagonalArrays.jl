module DiagonalArraysMatrixAlgebraKitExt

using DiagonalArrays:
  AbstractDiagonalMatrix,
  DeltaMatrix,
  DiagonalMatrix,
  ScaledDeltaMatrix,
  δ,
  diagview,
  dual,
  issquare
using LinearAlgebra: LinearAlgebra, isdiag, ishermitian
using MatrixAlgebraKit:
  MatrixAlgebraKit,
  AbstractAlgorithm,
  check_input,
  eig_full,
  eig_full!,
  eig_vals,
  eig_vals!,
  eigh_full,
  eigh_full!,
  eigh_vals,
  eigh_vals!,
  left_null,
  left_null!,
  left_orth,
  left_orth!,
  left_polar,
  left_polar!,
  lq_compact,
  lq_compact!,
  lq_full,
  lq_full!,
  qr_compact,
  qr_compact!,
  qr_full,
  qr_full!,
  right_null,
  right_null!,
  right_orth,
  right_orth!,
  right_polar,
  right_polar!,
  svd_compact,
  svd_compact!,
  svd_full,
  svd_full!,
  svd_vals,
  svd_vals!

abstract type AbstractDiagonalAlgorithm <: AbstractAlgorithm end

struct DeltaAlgorithm{KWargs<:NamedTuple} <: AbstractDiagonalAlgorithm
  kwargs::KWargs
end
DeltaAlgorithm(; kwargs...) = DeltaAlgorithm((; kwargs...))

struct ScaledDeltaAlgorithm{KWargs<:NamedTuple} <: AbstractDiagonalAlgorithm
  kwargs::KWargs
end
ScaledDeltaAlgorithm(; kwargs...) = ScaledDeltaAlgorithm((; kwargs...))

for f in [
  :eig_full,
  :eig_vals,
  :eigh_full,
  :eigh_vals,
  :qr_compact,
  :qr_full,
  :left_null,
  :left_orth,
  :left_polar,
  :lq_compact,
  :lq_full,
  :right_null,
  :right_orth,
  :right_polar,
  :svd_compact,
  :svd_full,
  :svd_vals,
]
  @eval begin
    MatrixAlgebraKit.copy_input(::typeof($f), a::AbstractDiagonalMatrix) = copy(a)
  end
end

for f in [
  :default_eig_algorithm,
  :default_eigh_algorithm,
  :default_lq_algorithm,
  :default_qr_algorithm,
  :default_polar_algorithm,
  :default_svd_algorithm,
]
  @eval begin
    function MatrixAlgebraKit.$f(::Type{<:DeltaMatrix}; kwargs...)
      return DeltaAlgorithm(; kwargs...)
    end
    function MatrixAlgebraKit.$f(::Type{<:ScaledDeltaMatrix}; kwargs...)
      return ScaledDeltaAlgorithm(; kwargs...)
    end
  end
end

for f in [
  :eig_full!,
  :eig_vals!,
  :eigh_full!,
  :eigh_vals!,
  :left_null!,
  :left_orth!,
  :left_polar!,
  :lq_compact!,
  :lq_full!,
  :qr_compact!,
  :qr_full!,
  :right_null!,
  :right_orth!,
  :right_polar!,
  :svd_compact!,
  :svd_full!,
  :svd_vals!,
]
  for Alg in [:ScaledDeltaAlgorithm, :DeltaAlgorithm]
    @eval begin
      function MatrixAlgebraKit.initialize_output(::typeof($f), a, alg::$Alg)
        return nothing
      end
    end
  end
end

for f in [
  :left_null!,
  :left_orth!,
  :left_polar!,
  :lq_compact!,
  :lq_full!,
  :qr_compact!,
  :qr_full!,
  :right_null!,
  :right_orth!,
  :right_polar!,
  :svd_compact!,
  :svd_full!,
  :svd_vals!,
]
  @eval begin
    function MatrixAlgebraKit.check_input(::typeof($f), a, F, alg::DeltaAlgorithm)
      @assert size(a, 1) == size(a, 2)
      @assert isdiag(a)
      @assert all(isone, diagview(a))
      return nothing
    end
    function MatrixAlgebraKit.check_input(::typeof($f), a, F, alg::ScaledDeltaAlgorithm)
      @assert size(a, 1) == size(a, 2)
      @assert isdiag(a)
      @assert allequal(diagview(a))
      return nothing
    end
  end
end
for f in [:eig_full!, :eig_vals!, :eigh_full!, :eigh_vals!]
  @eval begin
    function MatrixAlgebraKit.check_input(::typeof($f), a, F, alg::DeltaAlgorithm)
      @assert issquare(a)
      @assert isdiag(a)
      @assert all(isone, diagview(a))
      return nothing
    end
    function MatrixAlgebraKit.check_input(::typeof($f), a, F, alg::ScaledDeltaAlgorithm)
      @assert issquare(a)
      @assert isdiag(a)
      @assert allequal(diagview(a))
      return nothing
    end
  end
end

# eig
for Alg in [:DeltaAlgorithm, :ScaledDeltaAlgorithm]
  @eval begin
    function MatrixAlgebraKit.eig_full!(a, F, alg::$Alg)
      check_input(eig_full!, a, F, alg)
      d = complex(a)
      v = δ(complex(eltype(a)), axes(a))
      return (d, v)
    end
    function MatrixAlgebraKit.eigh_full!(a, F, alg::$Alg)
      check_input(eigh_full!, a, F, alg)
      ishermitian(a) || throw(ArgumentError("Matrix must be Hermitian"))
      d = real(a)
      v = δ(eltype(a), axes(a))
      return (d, v)
    end
    function MatrixAlgebraKit.eig_vals!(a, F, alg::$Alg)
      check_input(eig_vals!, a, F, alg)
      return complex(diagview(a))
    end
    function MatrixAlgebraKit.eigh_vals!(a, F, alg::$Alg)
      check_input(eigh_vals!, a, F, alg)
      return real(diagview(a))
    end
  end
end

# svd
for f in [:svd_compact!, :svd_full!]
  @eval begin
    function MatrixAlgebraKit.$f(a, F, alg::DeltaAlgorithm)
      check_input($f, a, F, alg)
      u = δ(eltype(a), (axes(a, 1), dual(axes(a, 1))))
      s = real(a)
      v = δ(eltype(a), (dual(axes(a, 2)), axes(a, 2)))
      return (u, s, v)
    end
    function MatrixAlgebraKit.$f(a, F, alg::ScaledDeltaAlgorithm)
      check_input($f, a, F, alg)
      diagvalue = only(unique(diagview(a)))
      u = δ(eltype(a), (axes(a, 1), dual(axes(a, 1))))
      s = abs(diagvalue) * δ(Bool, axes(a))
      # Sign is applied arbitarily to `v`, alternatively
      # we could apply it to `u`.
      v = sign(diagvalue) * δ(Bool, (dual(axes(a, 2)), axes(a, 2)))
      return (u, s, v)
    end
  end
end
function MatrixAlgebraKit.svd_vals!(a, F, alg::DeltaAlgorithm)
  check_input(svd_vals!, a, F, alg)
  # Using `real` instead of `abs.` helps to preserve `Ones`.
  return real(diagview(a))
end
function MatrixAlgebraKit.svd_vals!(a, F, alg::ScaledDeltaAlgorithm)
  check_input(svd_vals!, a, F, alg)
  return abs.(diagview(a))
end

# orth
for f in [:left_orth!, :left_polar!, :qr_compact!, :qr_full!]
  @eval begin
    function MatrixAlgebraKit.$f(a, F, alg::DeltaAlgorithm)
      check_input($f, a, F, alg)
      q = δ(eltype(a), (axes(a, 1), dual(axes(a, 1))))
      r = copy(a)
      return (q, r)
    end
    function MatrixAlgebraKit.$f(a, F, alg::ScaledDeltaAlgorithm)
      check_input($f, a, F, alg)
      diagvalue = only(unique(diagview(a)))
      q = sign(diagvalue) * δ(Bool, (axes(a, 1), dual(axes(a, 1))))
      # We're a bit pessimistic about the element type for type stability,
      # since in the future we might provide the option to do non-positive QR.
      r = eltype(a)(abs(diagvalue)) * δ(Bool, axes(a))
      return (q, r)
    end
  end
end
for f in [:right_orth!, :right_polar!, :lq_compact!, :lq_full!]
  @eval begin
    function MatrixAlgebraKit.$f(a, F, alg::DeltaAlgorithm)
      check_input($f, a, F, alg)
      l = copy(a)
      q = δ(eltype(a), (dual(axes(a, 2)), axes(a, 2)))
      return (l, q)
    end
    function MatrixAlgebraKit.$f(a, F, alg::ScaledDeltaAlgorithm)
      check_input($f, a, F, alg)
      diagvalue = only(unique(diagview(a)))
      # We're a bit pessimistic about the element type for type stability,
      # since in the future we might provide the option to do non-positive LQ.
      l = eltype(a)(abs(diagvalue)) * δ(Bool, axes(a))
      q = sign(diagvalue) * δ(Bool, (dual(axes(a, 2)), axes(a, 2)))
      return (l, q)
    end
  end
end

# null
for Alg in [:DeltaAlgorithm, :ScaledDeltaAlgorithm]
  @eval begin
    function MatrixAlgebraKit.left_null!(a, F, alg::$Alg)
      check_input(left_null!, a, F, alg)
      return error("Not implemented.")
    end
    function MatrixAlgebraKit.right_null!(a, F, alg::$Alg)
      check_input(right_null!, a, F, alg)
      return error("Not implemented.")
    end
  end
end

end
