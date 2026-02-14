# TODO: Turn these into `@interface ::AbstractDiagonalArrayInterface` functions.

diagview(a::AbstractDiagonalArray) = throw(MethodError(diagview, Tuple{typeof(a)}))

using FunctionImplementations: FunctionImplementations
using SparseArraysBase: SparseArraysBase as SA, AbstractSparseArrayImplementationStyle

abstract type AbstractDiagonalArrayImplementationStyle <:
AbstractSparseArrayImplementationStyle end

struct DiagonalArrayImplementationStyle <: AbstractDiagonalArrayImplementationStyle end
const diag_style = DiagonalArrayImplementationStyle()

function FunctionImplementations.ImplementationStyle(::Type{<:AbstractDiagonalArray})
    return DiagonalArrayImplementationStyle()
end

using SparseArraysBase: AbstractSparseArrayStyle
abstract type AbstractDiagonalArrayStyle{N} <: AbstractSparseArrayStyle{N} end
struct DiagonalArrayStyle{N} <: AbstractDiagonalArrayStyle{N} end
DiagonalArrayStyle{M}(::Val{N}) where {M, N} = DiagonalArrayStyle{N}()

using SparseArraysBase: getstoredindex
const getstoredindex_diag = diag_style(getstoredindex)
function getstoredindex_diag(
        a::AbstractArray{<:Any, N}, I::Vararg{Int, N}
    ) where {N}
    # TODO: Make this check optional, define `checkstored` like `checkbounds`
    # in SparseArraysBase.jl.
    # allequal(I) || error("Not a diagonal index.")
    return getdiagindex(a, first(I))
end
function getstoredindex_diag(a::AbstractArray{<:Any, 0})
    return getdiagindex(a, 1)
end
function getstoredindex_diag(a::AbstractArray, I::Int...)
    return sparse_style(getstoredindex)(a, I...)
end
using SparseArraysBase: setstoredindex!
const setstoredindex!_diag = diag_style(setstoredindex!)
function setstoredindex!_diag(
        a::AbstractArray{<:Any, N}, value, I::Vararg{Int, N}
    ) where {N}
    # TODO: Make this check optional, define `checkstored` like `checkbounds`
    # in SparseArraysBase.jl.
    # allequal(I) || error("Not a diagonal index.")
    setdiagindex!(a, value, first(I))
    return a
end
function setstoredindex!_diag(a::AbstractArray{<:Any, 0}, value)
    setdiagindex!(a, value, 1)
    return a
end
using SparseArraysBase: eachstoredindex
const eachstoredindex_diag = diag_style(eachstoredindex)
function eachstoredindex_diag(::IndexCartesian, a::AbstractArray)
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
getindex_diag(a::AbstractArray, I...) = sparse_style(getindex)(a, I...)
const setindex!_diag = diag_style(setindex!)
setindex!_diag(a::AbstractArray, value, I...) = sparse_style(setindex!)(a, value, I...)
const copyto!_diag = diag_style(copyto!)
copyto!_diag(dst::AbstractArray, src::AbstractArray) = sparse_style(copyto!)(dst, src)
const map_diag = diag_style(map)
map_diag(f, as::AbstractArray...) = sparse_style(map)(f, as...)
const map!_diag = diag_style(map!)
map!_diag(f, dst::AbstractArray, as::AbstractArray...) = sparse_style(map!)(f, dst, as...)
const fill!_diag = diag_style(fill!)
fill!_diag(a::AbstractArray, value) = sparse_style(fill!)(a, value)
using FunctionImplementations: zero!
const zero!_diag = diag_style(zero!)
zero!_diag(a::AbstractArray) = sparse_style(zero!)(a)
using SparseArraysBase: isstored
const isstored_diag = diag_style(isstored)
function isstored_diag(
        a::AbstractArray{<:Any, N}, I::Vararg{Int, N}
    ) where {N}
    return allequal(I)
end
isstored_diag(a::AbstractArray, I::Int...) = sparse_style(isstored)(a, I...)
using SparseArraysBase: storedvalues
const storedvalues_diag = diag_style(storedvalues)
storedvalues_diag(a::AbstractArray) = diagview(a)
using SparseArraysBase: storedpairs
const storedpairs_diag = diag_style(storedpairs)
storedpairs_diag(a::AbstractArray) = sparse_style(storedpairs)(a)

function Base.Broadcast.BroadcastStyle(type::Type{<:AbstractDiagonalArray})
    return DiagonalArrayStyle{ndims(type)}()
end

using Base.Broadcast: Broadcasted, broadcasted
using MapBroadcast: Mapped
# Map to a flattened broadcast expression of the diagonals of the arrays,
# also checking that the function preserves zeros.
function broadcasted_diagview(bc::Broadcasted)
    m = Mapped(bc)
    iszero(m.f(map(zero ∘ eltype, m.args)...)) || error(
        "Broadcasting DiagonalArrays with function that doesn't preserve zeros isn't supported yet."
    )
    return broadcasted(m.f, map(diagview, m.args)...)
end
function Base.copy(bc::Broadcasted{<:DiagonalArrayStyle})
    return DiagonalArray(copy(broadcasted_diagview(bc)), axes(bc))
end
function Base.copyto!(dest::AbstractArray, bc::Broadcasted{<:DiagonalArrayStyle})
    copyto!(diagview(dest), broadcasted_diagview(bc))
    return dest
end
