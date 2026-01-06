# TODO: Turn these into `@interface ::AbstractDiagonalArrayInterface` functions.

diagview(a::AbstractDiagonalArray) = throw(MethodError(diagview, Tuple{typeof(a)}))

using FunctionImplementations: FunctionImplementations
using SparseArraysBase: SparseArraysBase as SA, AbstractSparseArrayStyle

abstract type AbstractDiagonalArrayStyle <: AbstractSparseArrayStyle end

struct DiagonalArrayStyle <: AbstractDiagonalArrayStyle end
const diag_style = DiagonalArrayStyle()

## function Base.similar(::AbstractDiagonalArrayInterface, elt::Type, ax::Tuple)
##     return similar(DiagonalArray{elt}, ax)
## end
function FunctionImplementations.Style(::Type{<:AbstractDiagonalArray})
    return DiagonalArrayStyle()
end

module Broadcast
    import SparseArraysBase as SA
    abstract type AbstractDiagonalArrayStyle{N} <: SA.Broadcast.AbstractSparseArrayStyle{N} end
    struct DiagonalArrayStyle{N} <: AbstractDiagonalArrayStyle{N} end
    DiagonalArrayStyle{M}(::Val{N}) where {M, N} = DiagonalArrayStyle{N}()
end

function SA.isstored(
        a::AbstractDiagonalArray{<:Any, N}, I::Vararg{Int, N}
    ) where {N}
    return allequal(I)
end
function SA.getstoredindex(
        a::AbstractDiagonalArray{<:Any, N}, I::Vararg{Int, N}
    ) where {N}
    # TODO: Make this check optional, define `checkstored` like `checkbounds`
    # in SparseArraysBase.jl.
    # allequal(I) || error("Not a diagonal index.")
    return getdiagindex(a, first(I))
end
function SA.getstoredindex(a::AbstractDiagonalArray{<:Any, 0})
    return getdiagindex(a, 1)
end
function SA.setstoredindex!(
        a::AbstractDiagonalArray{<:Any, N}, value, I::Vararg{Int, N}
    ) where {N}
    # TODO: Make this check optional, define `checkstored` like `checkbounds`
    # in SparseArraysBase.jl.
    # allequal(I) || error("Not a diagonal index.")
    setdiagindex!(a, value, first(I))
    return a
end
function SA.setstoredindex!(a::AbstractDiagonalArray{<:Any, 0}, value)
    setdiagindex!(a, value, 1)
    return a
end
function SA.eachstoredindex(::IndexCartesian, a::AbstractDiagonalArray)
    return diagindices(a)
end

# Fix ambiguity error with SparseArraysBase.
function Base.getindex(a::AbstractDiagonalArray, I::DiagIndices)
    # TODO: Use `@interface` rather than `invoke`.
    return invoke(getindex, Tuple{AbstractArray, DiagIndices}, a, I)
end
# Fix ambiguity error with SparseArraysBase.
function Base.getindex(a::AbstractDiagonalArray, I::DiagIndex)
    # TODO: Use `@interface` rather than `invoke`.
    return invoke(getindex, Tuple{AbstractArray, DiagIndex}, a, I)
end
# Fix ambiguity error with SparseArraysBase.
function Base.setindex!(a::AbstractDiagonalArray, value, I::DiagIndices)
    # TODO: Use `@interface` rather than `invoke`.
    return invoke(setindex!, Tuple{AbstractArray, Any, DiagIndices}, a, value, I)
end
# Fix ambiguity error with SparseArraysBase.
function Base.setindex!(a::AbstractDiagonalArray, value, I::DiagIndex)
    # TODO: Use `@interface` rather than `invoke`.
    return invoke(setindex!, Tuple{AbstractArray, Any, DiagIndex}, a, value, I)
end

using SparseArraysBase: sparse_style
const getindex_diag = diag_style(getindex)
function getindex_diag(a::AbstractArray, I...)
    return sparse_style(getindex)(a, I...)
end
const copyto!_diag = diag_style(copyto!)
function copyto!_diag(dst::AbstractArray, src::AbstractArray)
    return sparse_style(copyto!)(dst, src)
end

function Base.Broadcast.BroadcastStyle(type::Type{<:AbstractDiagonalArray})
    return Broadcast.DiagonalArrayStyle{ndims(type)}()
end

using Base.Broadcast: Broadcasted, broadcasted
using MapBroadcast: Mapped
# Map to a flattened broadcast expression of the diagonals of the arrays,
# also checking that the function preserves zeros.
function broadcasted_diagview(bc::Broadcasted)
    m = Mapped(bc)
    iszero(m.f(map(zero ∘ eltype, m.args)...)) || error(
        "Broadcasting DiagonalArrays with function that doesn't preserve zeros isn't supported yet.",
    )
    return broadcasted(m.f, map(diagview, m.args)...)
end
function Base.copy(bc::Broadcasted{<:Broadcast.DiagonalArrayStyle})
    return DiagonalArray(copy(broadcasted_diagview(bc)), axes(bc))
end
function Base.copyto!(dest::AbstractArray, bc::Broadcasted{<:Broadcast.DiagonalArrayStyle})
    copyto!(diagview(dest), broadcasted_diagview(bc))
    return dest
end
