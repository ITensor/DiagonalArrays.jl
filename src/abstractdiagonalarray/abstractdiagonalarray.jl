using SparseArraysBase: AbstractSparseArray

abstract type AbstractDiagonalArray{T, N} <: AbstractSparseArray{T, N} end
const AbstractDiagonalMatrix{T} = AbstractDiagonalArray{T, 2}
const AbstractDiagonalVector{T} = AbstractDiagonalArray{T, 1}

# Define for type stability, for some reason the generic versions
# in SparseArraysBase.jl is not type stable.
# TODO: Investigate type stability of `iszero` in SparseArraysBase.jl.
function Base.iszero(a::AbstractDiagonalArray)
    return iszero(diagview(a))
end

using FillArrays: AbstractFill, getindex_value
using LinearAlgebra: norm
# TODO: `_norm` works around:
# https://github.com/JuliaArrays/FillArrays.jl/issues/417
# Change back to `norm` when that is fixed.
_norm(a, p::Int = 2) = norm(a, p)
function _norm(a::AbstractFill, p::Int = 2)
    nrm1 = norm(getindex_value(a))
    return (length(a))^(1 / oftype(nrm1, p)) * nrm1
end
function LinearAlgebra.norm(a::AbstractDiagonalArray, p::Int = 2)
    # TODO: `_norm` works around:
    # https://github.com/JuliaArrays/FillArrays.jl/issues/417
    # Change back to `norm` when that is fixed.
    return _norm(diagview(a), p)
end

using LinearAlgebra: LinearAlgebra, ishermitian, isposdef, issymmetric
LinearAlgebra.ishermitian(a::AbstractDiagonalMatrix{<:Real}) = issquare(a)
function LinearAlgebra.ishermitian(a::AbstractDiagonalMatrix{<:Number})
    return issquare(a) && isreal(diagview(a))
end
function LinearAlgebra.ishermitian(a::AbstractDiagonalMatrix)
    return issquare(a) && all(ishermitian, diagview(a))
end
LinearAlgebra.issymmetric(a::AbstractDiagonalMatrix{<:Number}) = issquare(a)
function LinearAlgebra.issymmetric(a::AbstractDiagonalMatrix)
    return issquare(a) && all(issymmetric, diagview(a))
end
function LinearAlgebra.isposdef(a::AbstractDiagonalMatrix)
    return issquare(a) && all(isposdef, diagview(a))
end
