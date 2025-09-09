using FillArrays: Zeros
using SparseArraysBase: Unstored, unstored

diaglength_from_shape(sz::Tuple{Integer,Vararg{Integer}}) = minimum(sz)
function diaglength_from_shape(
  sz::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}}
)
  return minimum(length, sz)
end
diaglength_from_shape(sz::Tuple{}) = 1

struct DiagonalArray{T,N,D<:AbstractVector{T},U<:AbstractArray{T,N}} <:
       AbstractDiagonalArray{T,N}
  diag::D
  unstored::U
  function DiagonalArray{T,N,D,U}(
    diag::AbstractVector, unstored::Unstored
  ) where {T,N,D<:AbstractVector{T},U<:AbstractArray{T,N}}
    length(diag) == diaglength_from_shape(size(unstored)) ||
      throw(ArgumentError("Length of diagonals doesn't match dimensions"))
    return new{T,N,D,U}(diag, parent(unstored))
  end
end

SparseArraysBase.unstored(a::DiagonalArray) = a.unstored
Base.size(a::DiagonalArray) = size(unstored(a))
Base.axes(a::DiagonalArray) = axes(unstored(a))

function DiagonalArray{T,N,D}(
  diag::D, unstored::Unstored{T,N,U}
) where {T,N,D<:AbstractVector{T},U<:AbstractArray{T,N}}
  return DiagonalArray{T,N,D,U}(diag, unstored)
end
function DiagonalArray{T,N}(
  diag::D, unstored::Unstored{T,N}
) where {T,N,D<:AbstractVector{T}}
  return DiagonalArray{T,N,D}(diag, unstored)
end
function DiagonalArray{T}(diag::AbstractVector{T}, unstored::Unstored{T,N}) where {T,N}
  return DiagonalArray{T,N}(diag, unstored)
end
function DiagonalArray(diag::AbstractVector{T}, unstored::Unstored{T}) where {T}
  return DiagonalArray{T}(diag, unstored)
end

function DiagonalArray(::UndefInitializer, unstored::Unstored)
  return DiagonalArray(
    Vector{eltype(unstored)}(undef, diaglength_from_shape(size(unstored))), unstored
  )
end

# Indicate we will construct an array just from the shape,
# for example for a Base.OneTo or FillArrays.Ones or Zeros.
# All the elements should be uniquely defined by the input axes.
struct ShapeInitializer end

# This is used to create custom constructors for arrays,
# in this case a generic constructor of a vector from a length.
function construct(vect::Type{<:AbstractVector}, ::ShapeInitializer, len::Integer)
  if applicable(vect, len)
    return vect(len)
  elseif applicable(vect, (Base.OneTo(len),))
    return vect((Base.OneTo(len),))
  else
    error(lazy"Can't construct $(vect) from length.")
  end
end

# This helps to support diagonals where the elements are known
# from the types, for example diagonals that are `Zeros` and `Ones`.
function DiagonalArray{T,N,D,U}(
  init::ShapeInitializer, unstored::Unstored{T,N,U}
) where {T,N,D<:AbstractVector{T},U<:AbstractArray{T,N}}
  return DiagonalArray{T,N,D,U}(
    construct(D, init, diaglength_from_shape(axes(unstored))), unstored
  )
end
function DiagonalArray{T,N,D}(
  init::ShapeInitializer, unstored::Unstored{T,N,U}
) where {T,N,D<:AbstractVector{T},U<:AbstractArray{T,N}}
  return DiagonalArray{T,N,D,U}(init, unstored)
end

# This helps to support diagonals where the elements are known
# from the types, for example diagonals that are `Zeros` and `Ones`.
# These versions use the default unstored type `Zeros{T,N}`.
function DiagonalArray{T,N,D,U}(
  init::ShapeInitializer, ax::Tuple{Vararg{AbstractUnitRange{<:Integer}}}
) where {T,N,D<:AbstractVector{T},U<:AbstractArray{T,N}}
  return DiagonalArray{T,N,D,U}(init, Unstored(U(ax)))
end
function DiagonalArray{T,N,D}(
  init::ShapeInitializer, ax::Tuple{Vararg{AbstractUnitRange{<:Integer}}}
) where {T,N,D<:AbstractVector{T}}
  return DiagonalArray{T,N,D}(init, Unstored(Zeros{T,N}(ax)))
end
function DiagonalArray{T,N,D,U}(
  init::ShapeInitializer, ax::AbstractUnitRange{<:Integer}...
) where {T,N,D<:AbstractVector{T},U<:AbstractArray{T,N}}
  return DiagonalArray{T,N,D,U}(init, ax)
end
function DiagonalArray{T,N,D}(
  init::ShapeInitializer, ax::AbstractUnitRange{<:Integer}...
) where {T,N,D<:AbstractVector{T}}
  return DiagonalArray{T,N,D}(init, ax)
end
function DiagonalArray{T,N,D,U}(
  init::ShapeInitializer, sz::Tuple{Integer,Vararg{Integer}}
) where {T,N,D<:AbstractVector{T},U<:AbstractArray{T,N}}
  return DiagonalArray{T,N,D,U}(init, Base.OneTo.(sz))
end
function DiagonalArray{T,N,D}(
  init::ShapeInitializer, sz::Tuple{Integer,Vararg{Integer}}
) where {T,N,D<:AbstractVector{T}}
  return DiagonalArray{T,N,D}(init, Base.OneTo.(sz))
end
function DiagonalArray{T,N,D,U}(
  init::ShapeInitializer, sz1::Integer, sz_rest::Integer...
) where {T,N,D<:AbstractVector{T},U<:AbstractArray{T,N}}
  return DiagonalArray{T,N,D,U}(init, (sz1, sz_rest...))
end
function DiagonalArray{T,N,D}(
  init::ShapeInitializer, sz1::Integer, sz_rest::Integer...
) where {T,N,D<:AbstractVector{T}}
  return DiagonalArray{T,N,D}(init, (sz1, sz_rest...))
end

# Constructor from diagonal entries accepting axes.
function DiagonalArray{T,N}(
  diag::AbstractVector,
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
) where {T,N}
  N == length(ax) || throw(ArgumentError("Wrong number of axes"))
  return DiagonalArray(convert(AbstractVector{T}, diag), Unstored(Zeros{T}(ax)))
end
function DiagonalArray{T,N}(
  diag::AbstractVector,
  ax1::AbstractUnitRange{<:Integer},
  axs::AbstractUnitRange{<:Integer}...,
) where {T,N}
  return DiagonalArray{T,N}(diag, (ax1, axs...))
end
function DiagonalArray{T}(
  diag::AbstractVector,
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
) where {T}
  return DiagonalArray{T,length(ax)}(diag, ax)
end
function DiagonalArray{T}(
  diag::AbstractVector,
  ax1::AbstractUnitRange{<:Integer},
  axs::AbstractUnitRange{<:Integer}...,
) where {T}
  return DiagonalArray{T}(diag, (ax1, axs...))
end
function DiagonalArray(
  diag::AbstractVector{T},
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
) where {T}
  return DiagonalArray{T,length(ax)}(diag, ax)
end
function DiagonalArray(
  diag::AbstractVector,
  ax1::AbstractUnitRange{<:Integer},
  axs::AbstractUnitRange{<:Integer}...,
)
  return DiagonalArray(diag, (ax1, axs...))
end

# undef constructors accepting axes.
function DiagonalArray{T,N}(
  ::UndefInitializer,
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
) where {T,N}
  return DiagonalArray{T,N}(Vector{T}(undef, minimum(length, ax)), ax)
end
function DiagonalArray{T,N}(
  ::UndefInitializer,
  ax1::AbstractUnitRange{<:Integer},
  axs::AbstractUnitRange{<:Integer}...,
) where {T,N}
  return DiagonalArray{T,N}(undef, (ax1, axs...))
end
function DiagonalArray{T}(
  ::UndefInitializer,
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
) where {T}
  return DiagonalArray{T,length(ax)}(undef, ax)
end
function DiagonalArray{T}(
  ::UndefInitializer,
  ax1::AbstractUnitRange{<:Integer},
  axs::AbstractUnitRange{<:Integer}...,
) where {T}
  return DiagonalArray{T}(undef, (ax1, axs...))
end

function DiagonalArray{T,N}(diag::AbstractVector, dims::Dims{N}) where {T,N}
  return DiagonalArray(convert(AbstractVector{T}, diag), Unstored(Zeros{T}(dims)))
end

function DiagonalArray{T,N}(diag::AbstractVector, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray{T}(diag::AbstractVector, dims::Dims{N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray{T}(diag::AbstractVector, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray{<:Any,N}(diag::AbstractVector{T}, dims::Dims{N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray{<:Any,N}(diag::AbstractVector{T}, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray{<:Any,N}(
  diag::AbstractVector{T},
  ax::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
) where {T,N}
  return DiagonalArray{T,N}(diag, ax)
end

function DiagonalArray(diag::AbstractVector{T}, dims::Dims{N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

function DiagonalArray(diag::AbstractVector{T}, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(diag, dims)
end

# Infer size from diagonal
function DiagonalArray{T,N}(diag::AbstractVector) where {T,N}
  return DiagonalArray{T,N}(diag, ntuple(Returns(length(diag)), N))
end

function DiagonalArray{<:Any,N}(diag::AbstractVector{T}) where {T,N}
  return DiagonalArray{T,N}(diag)
end

# undef
function DiagonalArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
  return DiagonalArray{T,N}(Vector{T}(undef, diaglength_from_shape(dims)), dims)
end

function DiagonalArray{T,N}(::UndefInitializer, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(undef, dims)
end

function DiagonalArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N}
  return DiagonalArray{T,N}(undef, dims)
end

function DiagonalArray{T}(::UndefInitializer, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(undef, dims)
end

# Axes version
function DiagonalArray{T}(
  ::UndefInitializer, axes::Tuple{Base.OneTo{Int},Vararg{Base.OneTo{Int}}}
) where {T}
  return DiagonalArray{T,length(axes)}(undef, length.(axes))
end

function Base.similar(a::DiagonalArray, unstored::Unstored)
  return DiagonalArray(undef, unstored)
end

# These definitions are helpful for immutable diagonals
# such as FillArrays.
for f in [:complex, :copy, :imag, :real]
  @eval begin
    Base.$f(a::DiagonalArray) = DiagonalArray($f(diagview(a)), axes(a))
  end
end

# DiagonalArrays interface.
diagview(a::DiagonalArray) = a.diag

# Special case for permutedims that is friendlier for immutable storage.
function Base.permutedims(a::DiagonalArray, perm)
  ((ndims(a) == length(perm)) && isperm(perm)) ||
    throw(ArgumentError("Not a valid permutation"))
  ax_perm = ntuple(d -> axes(a)[perm[d]], ndims(a))
  # Unlike `permutedims(::Diagonal, perm)`, we copy here.
  return DiagonalArray(copy(diagview(a)), ax_perm)
end

function DerivableInterfaces.permuteddims(a::DiagonalArray, perm)
  ((ndims(a) == length(perm)) && isperm(perm)) ||
    throw(ArgumentError("Not a valid permutation"))
  ax_perm = ntuple(d -> axes(a)[perm[d]], ndims(a))
  # Unlike `permutedims(::Diagonal, perm)`, we copy here.
  return DiagonalArray(diagview(a), ax_perm)
end

# Scalar indexing.
using DerivableInterfaces: @interface, interface
one_based_range(r) = false
one_based_range(r::Base.OneTo) = true
one_based_range(r::Base.Slice) = true
function _diag_axes(a::DiagonalArray, I...)
  return map(ntuple(identity, ndims(a))) do d
    return Base.axes1(axes(a, d)[I[d]])
  end
end
# A view that preserves the diagonal structure.
function _view_diag(a::DiagonalArray, I...)
  ax = _diag_axes(a, I...)
  return DiagonalArray(view(diagview(a), Base.OneTo(minimum(length, I))), ax)
end
function _view_diag(a::DiagonalArray, I1::Base.Slice, Irest::Base.Slice...)
  ax = _diag_axes(a, I1, Irest...)
  return DiagonalArray(view(diagview(a), :), ax)
end
# A slice that preserves the diagonal structure.
function _getindex_diag(a::DiagonalArray, I...)
  ax = _diag_axes(a, I...)
  return DiagonalArray(diagview(a)[Base.OneTo(minimum(length, I))], ax)
end
function _getindex_diag(a::DiagonalArray, I1::Base.Slice, Irest::Base.Slice...)
  ax = _diag_axes(a, I1, Irest...)
  return DiagonalArray(diagview(a)[:], ax)
end
function Base.view(a::DiagonalArray, I...)
  I′ = to_indices(a, I)
  return if all(one_based_range, I′)
    _view_diag(a, I′...)
  else
    invoke(view, Tuple{AbstractArray,Vararg}, a, I′...)
  end
end
function Base.getindex(a::DiagonalArray, I::Int...)
  return @interface interface(a) a[I...]
end
function Base.getindex(a::DiagonalArray, I::DiagIndex)
  return getdiagindex(a, index(I))
end
function Base.getindex(a::DiagonalArray, I::DiagIndices)
  # TODO: Should this be a view?
  return @view diagview(a)[indices(I)]
end
function Base.getindex(a::DiagonalArray, I...)
  I′ = to_indices(a, I)
  return if all(i -> i isa Real, I′)
    # Catch scalar indexing case.
    @interface interface(a) a[I...]
  elseif all(one_based_range, I′)
    _getindex_diag(a, I′...)
  else
    copy(view(a, I′...))
  end
end

# Define in order to preserve immutable diagonals such as FillArrays.
function DiagonalArray{T,N}(a::DiagonalArray{T,N}) where {T,N}
  # TODO: Should this copy? This matches the design of `LinearAlgebra.Diagonal`:
  # https://github.com/JuliaLang/LinearAlgebra.jl/blob/release-1.12/src/diagonal.jl#L110-L112
  return a
end
function DiagonalArray{T,N}(a::DiagonalArray{<:Any,N}) where {T,N}
  return DiagonalArray{T,N}(diagview(a))
end
function DiagonalArray{T}(a::DiagonalArray) where {T}
  return DiagonalArray{T,ndims(a)}(a)
end
function DiagonalArray(a::DiagonalArray)
  return DiagonalArray{eltype(a),ndims(a)}(a)
end
function Base.AbstractArray{T,N}(a::DiagonalArray{<:Any,N}) where {T,N}
  return DiagonalArray{T,N}(a)
end

# TODO: These definitions work around this issue:
# https://github.com/JuliaArrays/FillArrays.jl/issues/416
# when the diagonal is a FillArrays.Ones or Zeros.
using Base.Broadcast: Broadcast, broadcast, broadcasted
using FillArrays: AbstractFill, Ones, Zeros
_broadcasted(f::F, a::AbstractArray) where {F} = broadcasted(f, a)
_broadcasted(::typeof(identity), a::Ones) = a
_broadcasted(::typeof(identity), a::Zeros) = a
_broadcasted(::typeof(complex), a::Ones) = Ones{complex(eltype(a))}(axes(a))
_broadcasted(::typeof(complex), a::Zeros) = Zeros{complex(eltype(a))}(axes(a))
_broadcasted(elt::Type, a::Ones) = Ones{elt}(axes(a))
_broadcasted(elt::Type, a::Zeros) = Zeros{elt}(axes(a))
_broadcasted(::typeof(inv), a::Ones) = _broadcasted(typeof(inv(oneunit(eltype(a)))), a)
using LinearAlgebra: pinv
_broadcasted(::typeof(pinv), a::Ones) = _broadcasted(typeof(inv(oneunit(eltype(a)))), a)
_broadcasted(::typeof(pinv), a::Zeros) = _broadcasted(typeof(inv(zero(eltype(a)))), a)
_broadcasted(::typeof(sqrt), a::Ones) = _broadcasted(typeof(sqrt(one(eltype(a)))), a)
_broadcasted(::typeof(sqrt), a::Zeros) = _broadcasted(typeof(sqrt(zero(eltype(a)))), a)
_broadcasted(::typeof(cbrt), a::Ones) = _broadcasted(typeof(cbrt(one(eltype(a)))), a)
_broadcasted(::typeof(cbrt), a::Zeros) = _broadcasted(typeof(cbrt(zero(eltype(a)))), a)
_broadcasted(::typeof(exp), a::Zeros) = Ones{typeof(exp(zero(eltype(a))))}(axes(a))
_broadcasted(::typeof(cis), a::Zeros) = Ones{typeof(cis(zero(eltype(a))))}(axes(a))
_broadcasted(::typeof(log), a::Ones) = Zeros{typeof(log(one(eltype(a))))}(axes(a))
_broadcasted(::typeof(cos), a::Zeros) = Ones{typeof(cos(zero(eltype(a))))}(axes(a))
_broadcasted(::typeof(sin), a::Zeros) = _broadcasted(typeof(sin(zero(eltype(a)))), a)
_broadcasted(::typeof(tan), a::Zeros) = _broadcasted(typeof(tan(zero(eltype(a)))), a)
_broadcasted(::typeof(sec), a::Zeros) = Ones{typeof(sec(zero(eltype(a))))}(axes(a))
_broadcasted(::typeof(cosh), a::Zeros) = Ones{typeof(cosh(zero(eltype(a))))}(axes(a))
# Eager version of `_broadcasted`.
_broadcast(f::F, a::AbstractArray) where {F} = copy(_broadcasted(f, a))

function Broadcast.broadcasted(
  ::DiagonalArrayStyle{N}, f::F, a::DiagonalArray{T,N,Diag}
) where {F,T,N,Diag<:AbstractFill{T}}
  # TODO: Check that `f` preserves zeros?
  return DiagonalArray(_broadcasted(f, diagview(a)), axes(a))
end
