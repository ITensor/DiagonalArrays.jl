const DiagonalVector{T, Diag <: AbstractVector{T}, Unstored <: AbstractVector{T}} =
    DiagonalArray{
    T, 1, Diag, Unstored,
}

function DiagonalVector(diag::AbstractVector)
    return DiagonalArray{<:Any, 1}(diag)
end
