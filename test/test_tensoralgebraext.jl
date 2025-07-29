using DiagonalArrays: Delta
using FillArrays: Eye
using TensorAlgebra: FusionStyle, matricize, tuplemortar, unmatricize
using Test: @test, @testset

@testset "matricize, unmatricize" begin
  a = Delta{Float32}(2, 2, 2)
  m = matricize(a, (1,), (2, 3))
  @test m ≡ Eye{Float32}(2, 4)
  @test unmatricize(FusionStyle(a), m, tuplemortar(((axes(a, 1),), (axes(a, 2), axes(a, 3))))) ≡ a
end
