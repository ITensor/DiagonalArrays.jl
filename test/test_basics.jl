using Test: @test, @testset, @test_broken, @inferred
using DiagonalArrays:
  DiagonalArrays, DiagonalArray, DiagonalMatrix, δ, delta, diaglength, diagonal, diagview
using FillArrays: Fill, Ones
using SparseArraysBase: SparseArrayDOK, sparsezeros, storedlength
using LinearAlgebra: Diagonal

@testset "Test DiagonalArrays" begin
  @testset "DiagonalArray (eltype=$elt)" for elt in (
    Float32, Float64, Complex{Float32}, Complex{Float64}
  )
    @testset "Basics" begin
      a = fill(one(elt), 2, 3)
      @test diaglength(a) == 2
      a = fill(one(elt))
      @test diaglength(a) == 1
    end
    @testset "Matrix multiplication" begin
      a1 = DiagonalArray{elt}(undef, (2, 3))
      a1[1, 1] = 11
      a1[2, 2] = 22
      a2 = DiagonalArray{elt}(undef, (3, 4))
      a2[1, 1] = 11
      a2[2, 2] = 22
      a2[3, 3] = 33
      a_dest = a1 * a2
      # TODO: Use `densearray` to make generic to GPU.
      @test Array(a_dest) ≈ Array(a1) * Array(a2)
      # TODO: Make this work with `ArrayLayouts`.
      @test storedlength(a_dest) == 2
      @test a_dest isa DiagonalMatrix{elt}

      # TODO: Make generic to GPU, use `allocate_randn`?
      a2 = randn(elt, (3, 4))
      a_dest = a1 * a2
      # TODO: Use `densearray` to make generic to GPU.
      @test Array(a_dest) ≈ Array(a1) * Array(a2)
      @test storedlength(a_dest) == 8
      @test a_dest isa Matrix{elt}

      a2 = sparsezeros(elt, (3, 4))
      a2[1, 1] = 11
      a2[2, 2] = 22
      a2[3, 3] = 33
      a_dest = a1 * a2
      # TODO: Use `densearray` to make generic to GPU.
      @test Array(a_dest) ≈ Array(a1) * Array(a2)
      # TODO: Define `SparseMatrixDOK`.
      # TODO: Make this work with `ArrayLayouts`.
      @test storedlength(a_dest) == 2
      @test a_dest isa SparseArrayDOK{elt,2}
    end
    @testset "diagonal" begin
      @test @inferred(diagonal(rand(2))) isa AbstractMatrix
      @test diagonal(zeros(Int, 2)) isa Diagonal
    end
    @testset "delta" begin
      for (a, elt′) in (
        (delta(2, 3), Float64),
        (δ(2, 3), Float64),
        (delta((2, 3)), Float64),
        (δ((2, 3)), Float64),
        (delta(Bool, 2, 3), Bool),
        (δ(Bool, 2, 3), Bool),
        (delta(Bool, (2, 3)), Bool),
        (δ(Bool, (2, 3)), Bool),
      )
        @test eltype(a) === elt′
        @test diaglength(a) == 2
        @test a isa DiagonalArray{elt′}
        @test size(a) == (2, 3)
        @test diaglength(a) == 2
        @test storedlength(a) == 2
        @test a == DiagonalArray(ones(2), (2, 3))
        @test diagview(a) == ones(2)
        @test diagview(a) isa Ones{elt′}

        a′ = 2a
        @test diagview(a′) == 2ones(2)
        # TODO: Fix this. Mapping doesn't preserve
        # the diagonal structure properly.
        # https://github.com/ITensor/DiagonalArrays.jl/issues/7
        @test_broken diagview(a′) isa Fill

        b = randn(elt, (3, 4))
        a_dest = a * b
        @test a_dest ≈ Array(a) * Array(b)
      end
    end
  end
end
