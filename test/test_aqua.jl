using DiagonalArrays: DiagonalArrays
using Aqua: Aqua
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
  # TODO: fix ambiguities
  Aqua.test_all(DiagonalArrays; ambiguities=false)
end
