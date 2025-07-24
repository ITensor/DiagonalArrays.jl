using Test: @test, @testset, @test_broken, @inferred
using DiagonalArrays:
  DiagonalArrays,
  DiagonalArray,
  DiagonalMatrix,
  δ,
  delta,
  diagindices,
  diaglength,
  diagonal,
  diagonaltype,
  diagview
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
    @testset "diagindices" begin
      a = randn(elt, ())
      @test diagindices(a) == diagindices(IndexLinear(), a) == 1:1:1
      @test isempty(diagindices(IndexCartesian(), a))

      for a in (
        randn(elt, (0,)),
        randn(elt, (0, 0)),
        randn(elt, (0, 3)),
        randn(elt, (3, 0)),
        randn(elt, (0, 0, 0)),
        randn(elt, (3, 3, 0)),
      )
        @test diagindices(a) == diagindices(IndexLinear(), a) == 1:1:0
        @test isempty(diagindices(IndexCartesian(), a))
      end

      a = randn(elt, (3,))
      @test diagindices(a) == diagindices(IndexLinear(), a) == 1:1:3
      @test diagindices(IndexCartesian(), a) == CartesianIndex.(1:3)

      a = randn(elt, (4,))
      @test diagindices(a) == diagindices(IndexLinear(), a) == 1:1:4
      @test diagindices(IndexCartesian(), a) == CartesianIndex.(1:4)

      for a in (randn(elt, (3, 3)), randn(elt, (3, 4)))
        @test diagindices(a) == diagindices(IndexLinear(), a) == 1:4:9
        @test diagindices(IndexCartesian(), a) == CartesianIndex.(Iterators.zip(1:3, 1:3))
      end

      a = randn(elt, (4, 3))
      @test diagindices(a) == diagindices(IndexLinear(), a) == 1:5:11
      @test diagindices(IndexCartesian(), a) == CartesianIndex.(Iterators.zip(1:3, 1:3))

      for a in (randn(elt, (3, 3, 3)), randn(elt, (3, 3, 4)))
        @test diagindices(a) == diagindices(IndexLinear(), a) == 1:13:27
        @test diagindices(IndexCartesian(), a) ==
          CartesianIndex.(Iterators.zip(1:3, 1:3, 1:3))
      end

      a = randn(elt, (3, 4, 3))
      @test diagindices(a) == diagindices(IndexLinear(), a) == 1:16:33
      @test diagindices(IndexCartesian(), a) ==
        CartesianIndex.(Iterators.zip(1:3, 1:3, 1:3))
    end
    @testset "DiagonalArray constructors" begin
      v = randn(elt, 2)
      @test DiagonalArray(v, 2, 2) ≡
        DiagonalArray(v, (2, 2)) ≡
        DiagonalArray(v, Base.OneTo(2), Base.OneTo(2)) ≡
        DiagonalArray(v, (Base.OneTo(2), Base.OneTo(2))) ≡
        DiagonalArray{elt}(v, 2, 2) ≡
        DiagonalArray{elt}(v, (2, 2)) ≡
        DiagonalArray{elt}(v, Base.OneTo(2), Base.OneTo(2)) ≡
        DiagonalArray{elt}(v, (Base.OneTo(2), Base.OneTo(2))) ≡
        DiagonalArray{elt,2}(v, 2, 2) ≡
        DiagonalArray{elt,2}(v, (2, 2)) ≡
        DiagonalArray{elt,2}(v, Base.OneTo(2), Base.OneTo(2)) ≡
        DiagonalArray{elt,2}(v, (Base.OneTo(2), Base.OneTo(2)))
      @test size(DiagonalArray{elt}(undef, 2, 2)) ≡
        size(DiagonalArray{elt}(undef, (2, 2))) ≡
        size(DiagonalArray{elt}(undef, Base.OneTo(2), Base.OneTo(2))) ≡
        size(DiagonalArray{elt}(undef, (Base.OneTo(2), Base.OneTo(2)))) ≡
        size(DiagonalArray{elt,2}(undef, 2, 2)) ≡
        size(DiagonalArray{elt,2}(undef, (2, 2))) ≡
        size(DiagonalArray{elt,2}(undef, Base.OneTo(2), Base.OneTo(2))) ≡
        size(DiagonalArray{elt,2}(undef, (Base.OneTo(2), Base.OneTo(2))))
      @test elt ≡
        eltype(DiagonalArray{elt}(undef, 2, 2)) ≡
        eltype(DiagonalArray{elt}(undef, (2, 2))) ≡
        eltype(DiagonalArray{elt}(undef, Base.OneTo(2), Base.OneTo(2))) ≡
        eltype(DiagonalArray{elt}(undef, (Base.OneTo(2), Base.OneTo(2)))) ≡
        eltype(DiagonalArray{elt,2}(undef, 2, 2)) ≡
        eltype(DiagonalArray{elt,2}(undef, (2, 2))) ≡
        eltype(DiagonalArray{elt,2}(undef, Base.OneTo(2), Base.OneTo(2))) ≡
        eltype(DiagonalArray{elt,2}(undef, (Base.OneTo(2), Base.OneTo(2))))
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
      v = randn(2)
      d = @inferred diagonal(v)
      @test d isa Diagonal{eltype(v)}
      @test diagview(d) === v
      @test diagonaltype(v) === typeof(d)

      a = randn(2, 2)
      d = @inferred diagonal(a)
      @test d isa Diagonal{eltype(v)}
      @test diagview(d) == diagview(a)
      @test diagonaltype(a) === typeof(d)
    end
    @testset "delta" begin
      for (a, elt′) in (
        (delta(2, 3), Float64),
        (delta(Base.OneTo(2), Base.OneTo(3)), Float64),
        (δ(2, 3), Float64),
        (δ(Base.OneTo(2), Base.OneTo(3)), Float64),
        (delta((2, 3)), Float64),
        (delta(Base.OneTo.((2, 3))), Float64),
        (δ((2, 3)), Float64),
        (δ(Base.OneTo.((2, 3))), Float64),
        (delta(Bool, 2, 3), Bool),
        (delta(Bool, Base.OneTo(2), Base.OneTo(3)), Bool),
        (δ(Bool, 2, 3), Bool),
        (δ(Bool, Base.OneTo(2), Base.OneTo(3)), Bool),
        (delta(Bool, (2, 3)), Bool),
        (delta(Bool, Base.OneTo.((2, 3))), Bool),
        (δ(Bool, (2, 3)), Bool),
        (δ(Bool, Base.OneTo.((2, 3))), Bool),
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
