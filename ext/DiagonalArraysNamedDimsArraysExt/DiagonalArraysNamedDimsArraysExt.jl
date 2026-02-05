module DiagonalArraysNamedDimsArraysExt

using DiagonalArrays: DiagonalArrays, δ, delta
using NamedDimsArrays: AbstractNamedUnitRange, denamed, name, nameddims

function DiagonalArrays.delta(
        elt::Type{<:Number},
        is::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}},
    )
    return nameddims(delta(elt, length.(denamed.(is))), name.(is))
end
# TODO: Make `δ` an alias for `delta` so this isn't needed.
function DiagonalArrays.δ(
        elt::Type{<:Number},
        is::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}},
    )
    return delta(elt, is)
end
# TODO: Is this needed?
function DiagonalArrays.delta(
        is::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}
    )
    return delta(Bool, is)
end
# TODO: Is this needed?
function DiagonalArrays.δ(
        is::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}
    )
    return delta(is)
end
# TODO: Is this needed?
function DiagonalArrays.delta(
        elt::Type{<:Number}, i1::AbstractNamedUnitRange, i_rest::AbstractNamedUnitRange...
    )
    return delta(elt, (i1, i_rest...))
end
# TODO: Is this needed?
function DiagonalArrays.δ(
        elt::Type{<:Number}, i1::AbstractNamedUnitRange, i_rest::AbstractNamedUnitRange...
    )
    return delta(elt, i1, i_rest...)
end
# TODO: Is this needed?
function DiagonalArrays.delta(i1::AbstractNamedUnitRange, i_rest::AbstractNamedUnitRange...)
    return delta((i1, i_rest...))
end
# TODO: Is this needed?
function DiagonalArrays.δ(i1::AbstractNamedUnitRange, i_rest::AbstractNamedUnitRange...)
    return delta(i1, i_rest...)
end

end
