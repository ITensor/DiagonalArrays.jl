const DiagonalMatrix{T,Diag<:AbstractVector{T},Unstored<:AbstractMatrix{T}} = DiagonalArray{
  T,2,Diag,Unstored
}

# LinearAlgebra

using LinearAlgebra: LinearAlgebra

function mul_diagviews(a1, a2)
  # TODO: Compare that duals are equal, or define a function to overload.
  dual(axes(a1, 2)) == axes(a2, 1) || throw(
    DimensionMismatch(
      lazy"Incompatible dimensions for multiplication: $(axes(a1)) and $(axes(a2))"
    ),
  )
  d1 = diagview(a1)
  d2 = diagview(a2)
  l = min(length(d1), length(d2))
  d1′ = view(d1, Base.OneTo(l))
  d2′ = view(d2, Base.OneTo(l))
  return (d1′, d2′)
end

function mul!_diagviews(a_dest, a1, a2)
  axes(a_dest, 1) == axes(a1, 1) || throw(
    DimensionMismatch(
      lazy"Incompatible dimensions for multiplication: $(axes(a_dest)) and $(axes(a1))"
    ),
  )
  axes(a_dest, 2) == axes(a2, 2) || throw(
    DimensionMismatch(
      lazy"Incompatible dimensions for multiplication: $(axes(a_dest)) and $(axes(a2))"
    ),
  )
  d_dest = diagview(a_dest)
  d1, d2 = mul_diagviews(a1, a2)
  return d_dest, d1, d2
end

function Base.:*(a1::DiagonalMatrix, a2::DiagonalMatrix)
  d1, d2 = mul_diagviews(a1, a2)
  # TODO: Handle the rack-deficient case, for example:
  # δ(3, 2) * δ(2, 3)
  # Maybe pack the diagonal with zeros or allow rank-deficient DiagonalArrays?
  return DiagonalMatrix(d1 .* d2, (axes(a1, 1), axes(a2, 2)))
end
function LinearAlgebra.mul!(a_dest::DiagonalMatrix, a1::DiagonalMatrix, a2::DiagonalMatrix)
  d_dest, d1, d2 = mul!_diagviews(a_dest, a1, a2)
  # TODO: Handle the rack-deficient case.
  d_dest .= d1 .* d2
  return a_dest
end
function LinearAlgebra.mul!(
  a_dest::DiagonalMatrix, a1::DiagonalMatrix, a2::DiagonalMatrix, α::Number, β::Number
)
  d_dest, d1, d2 = mul!_diagviews(a_dest, a1, a2)
  # TODO: Handle the rack-deficient case.
  d_dest .= d1 .* d2 .* α .+ d_dest .* β
  return a_dest
end

# Adapted from https://github.com/JuliaLang/LinearAlgebra.jl/blob/release-1.12/src/diagonal.jl#L866-L928.
function LinearAlgebra.tr(a::DiagonalMatrix)
  checksquare(a)
  # TODO: Define as `sum(tr, diagview(a))` like LinearAlgebra.jl?
  return sum(diagview(a))
end
# TODO: Special case for FillArrays diagonals.
function LinearAlgebra.det(a::DiagonalMatrix)
  checksquare(a)
  # TODO: Define as `prod(det, diagview(a))` like LinearAlgebra.jl?
  return prod(diagview(a))
end
# TODO: Special case for FillArrays diagonals.
function LinearAlgebra.logabsdet(a::DiagonalMatrix)
  checksquare(a)
  return mapreduce(((d1, s1), (d2, s2)) -> (d1 + d2, s1 * s2), diagview(a)) do x
    return (log(abs(x)), sign(x))
  end
end
# TODO: Special case for FillArrays diagonals.
function LinearAlgebra.logdet(a::DiagonalMatrix{<:Complex})
  checksquare(a)
  z = sum(log, diagview(a))
  return complex(real(z), rem2pi(imag(z), RoundNearest))
end

# Matrix functions
for f in (
  :exp,
  :cis,
  :log,
  :sqrt,
  :cos,
  :sin,
  :tan,
  :csc,
  :sec,
  :cot,
  :cosh,
  :sinh,
  :tanh,
  :csch,
  :sech,
  :coth,
  :acos,
  :asin,
  :atan,
  :acsc,
  :asec,
  :acot,
  :acosh,
  :asinh,
  :atanh,
  :acsch,
  :asech,
  :acoth,
)
  @eval begin
    function Base.$f(a::DiagonalMatrix)
      checksquare(a)
      return DiagonalMatrix(_broadcast($f, diagview(a)), axes(a))
    end
  end
end

# Cube root of a real-valued diagonal matrix
function Base.cbrt(a::DiagonalMatrix{<:Real})
  checksquare(a)
  return DiagonalMatrix(_broadcast(cbrt, diagview(a)), axes(a))
end

function LinearAlgebra.inv(a::DiagonalMatrix)
  checksquare(a)
  # `DiagonalArrays._broadcast` works around issues like https://github.com/JuliaArrays/FillArrays.jl/issues/416
  # when the diagonal is a FillArray or similar lazy array.
  d⁻¹ = _broadcast(inv, diagview(a))
  any(isinf, d⁻¹) && error("Singular Exception")
  return DiagonalMatrix(d⁻¹, axes(a))
end

# TODO: Support `atol` and `rtol` keyword arguments:
# https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.pinv
using LinearAlgebra: pinv
function LinearAlgebra.pinv(a::DiagonalMatrix)
  checksquare(a)
  return DiagonalMatrix(_broadcast(pinv, diagview(a)), axes(a))
end
