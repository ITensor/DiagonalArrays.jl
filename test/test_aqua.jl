using Aqua: Aqua
using DiagonalArrays: DiagonalArrays
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    # TODO: fix ambiguities
    Aqua.test_all(DiagonalArrays; ambiguities = false)
end
