using Test: @test, @testset, @test_broken, @inferred
using DerivableInterfaces: permuteddims
using DiagonalArrays:
  DiagonalArrays,
  ShapeInitializer,
  Delta,
  DeltaMatrix,
  DiagonalArray,
  DiagonalMatrix,
  ScaledDelta,
  ScaledDeltaMatrix,
  Unstored,
  δ,
  delta,
  diagindices,
  diaglength,
  diagonal,
  diagonaltype,
  diagview
using FillArrays: Fill, Ones, Zeros
using SparseArraysBase: SparseArrayDOK, SparseMatrixDOK, sparsezeros, storedlength
using LinearAlgebra:
  Diagonal, det, ishermitian, isposdef, issymmetric, logdet, mul!, pinv, tr

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

      # Special constructors for immutable diagonal.
      init = ShapeInitializer()
      @test DiagonalMatrix(Base.OneTo(UInt32(2))) ≡
        DiagonalArray{UInt32,2,Base.OneTo{UInt32}}(init, Base.OneTo.((2, 2))) ≡
        DiagonalArray{UInt32,2,Base.OneTo{UInt32}}(init, Base.OneTo.((2, 2))...) ≡
        DiagonalArray{UInt32,2,Base.OneTo{UInt32}}(init, (2, 2)) ≡
        DiagonalArray{UInt32,2,Base.OneTo{UInt32}}(init, 2, 2) ≡
        DiagonalArray{UInt32,2,Base.OneTo{UInt32}}(init, Unstored(Zeros{UInt32}(2, 2)))

      # 0-dim constructors
      v = randn(elt, 1)
      @test DiagonalArray(v) ≡
        DiagonalArray(v, ()) ≡
        DiagonalArray{elt}(v) ≡
        DiagonalArray{elt}(v, ()) ≡
        DiagonalArray{elt,0}(v) ≡
        DiagonalArray{elt,0}(v, ())
      @test size(DiagonalArray{elt}(undef)) ≡
        size(DiagonalArray{elt}(undef, ())) ≡
        size(DiagonalArray{elt,0}(undef)) ≡
        size(DiagonalArray{elt,0}(undef, ()))
      @test elt ≡
        eltype(DiagonalArray{elt}(undef)) ≡
        eltype(DiagonalArray{elt}(undef, ())) ≡
        eltype(DiagonalArray{elt,0}(undef)) ≡
        eltype(DiagonalArray{elt,0}(undef, ()))

      # Special constructors for immutable diagonal.
      init = ShapeInitializer()
      @test DiagonalArray{<:Any,0}(Base.OneTo(UInt32(1))) ≡
        DiagonalArray{UInt32,0,Base.OneTo{UInt32}}(init, ()) ≡
        DiagonalArray{UInt32,0,Base.OneTo{UInt32}}(init) ≡
        DiagonalArray{UInt32,0,Base.OneTo{UInt32}}(init, Unstored(Zeros{UInt32}()))
    end
    @testset "0-dim operations" begin
      diag = randn(elt, 1)
      a = DiagonalArray(diag)
      @test a[] == diag[1]
      a[] = 2
      @test a[] == 2
    end
    @testset "Conversion" begin
      a = DiagonalMatrix(randn(elt, 2))
      @test DiagonalMatrix{elt}(a) ≡ a
      @test DiagonalMatrix{ComplexF64}(a) == DiagonalMatrix(ComplexF64.(diagview(a)))
      @test DiagonalArray{ComplexF64}(a) == DiagonalMatrix(ComplexF64.(diagview(a)))
      @test DiagonalArray(a) ≡ a
      @test AbstractMatrix{elt}(a) ≡ a
      @test AbstractMatrix{ComplexF64}(a) == DiagonalMatrix(ComplexF64.(diagview(a)))
      @test AbstractArray{elt}(a) ≡ a
      @test AbstractArray{ComplexF64}(a) == DiagonalMatrix(ComplexF64.(diagview(a)))
    end
    @testset "Slicing" begin
      # Slicing that preserves the diagonal structure.
      a = DiagonalMatrix(randn(elt, 3))
      b = @view a[:, :]
      @test b isa DiagonalMatrix{elt,<:SubArray{elt,1}}
      @test diagview(b) ≡ view(diagview(a), :)

      a = DiagonalMatrix(randn(elt, 3))
      b = @view a[Base.OneTo(2), Base.OneTo(2)]
      @test b isa DiagonalMatrix{elt,<:SubArray{elt,1}}
      @test diagview(b) ≡ view(diagview(a), Base.OneTo(2))

      a = DiagonalMatrix(randn(elt, 3))
      b = a[:, :]
      @test typeof(b) == typeof(a)
      @test diagview(b) == diagview(a)

      a = DiagonalMatrix(randn(elt, 3))
      b = a[Base.OneTo(2), Base.OneTo(2)]
      @test typeof(b) == typeof(a)
      @test diagview(b) == diagview(a)[Base.OneTo(2)]

      # Slicing that doesn't preserve the diagonal structure.
      a = DiagonalMatrix(randn(elt, 3))
      b = @view a[2:3, 2:3]
      @test b isa SubArray
      @test b == Matrix(a)[2:3, 2:3]

      a = DiagonalMatrix(randn(elt, 3))
      b = a[2:3, 2:3]
      @test b isa SparseMatrixDOK{elt}
      @test b == Matrix(a)[2:3, 2:3]
      @test storedlength(b) == 2
    end
    @testset "permutedims" begin
      a = DiagonalArray(randn(elt, 2), (2, 3, 4))
      b = permutedims(a, (3, 1, 2))
      @test diagview(b) == diagview(a)
      @test diagview(b) ≢ diagview(a)
      @test size(b) === (4, 2, 3)
    end
    @testset "DerivableInterfaces.permuteddims" begin
      a = DiagonalArray(randn(elt, 2), (2, 3, 4))
      b = permuteddims(a, (3, 1, 2))
      @test diagview(b) ≡ diagview(a)
      @test size(b) === (4, 2, 3)
    end
    @testset "Broadcasting" begin
      a = DiagonalArray(randn(elt, 2), (2, 3))
      b = DiagonalArray(randn(elt, 2), (2, 3))
      c = a .+ 2 .* b
      @test c ≈ Array(a) + 2 * Array(b)
      # Non-zero-preserving functions not supported yet.
      @test_broken a .+ 2

      c = DiagonalArray{elt}(undef, (2, 3))
      c .= a .+ 2 .* b
      @test c ≈ Array(a) + 2 * Array(b)

      # Non-zero-preserving functions not supported yet.
      c = DiagonalArray{elt}(undef, (2, 3))
      @test_broken c .= a .+ 2

      a_ones = DiagonalMatrix(Ones{elt}(2))
      a_zeros = DiagonalMatrix(Zeros{elt}(2))
      @test identity.(a_ones) ≡ DiagonalMatrix(Ones{elt}(2))
      @test identity.(a_zeros) ≡ DiagonalMatrix(Zeros{elt}(2))
      @test complex.(a_ones) ≡ DiagonalMatrix(Ones{complex(elt)}(2))
      @test complex.(a_zeros) ≡ DiagonalMatrix(Zeros{complex(elt)}(2))
      @test Float32.(a_ones) ≡ DiagonalMatrix(Ones{Float32}(2))
      @test Float32.(a_zeros) ≡ DiagonalMatrix(Zeros{Float32}(2))
      @test inv.(a_ones) ≡ DiagonalMatrix(Ones{elt}(2))
      @test inv.(a_zeros) ≡ DiagonalMatrix(Fill(inv(zero(elt)), 2))
      @test pinv.(a_ones) ≡ DiagonalMatrix(Ones{elt}(2))
      @test pinv.(a_zeros) ≡ DiagonalMatrix(Zeros{elt}(2))
      @test sqrt.(a_ones) ≡ DiagonalMatrix(Ones{elt}(2))
      @test sqrt.(a_zeros) ≡ DiagonalMatrix(Zeros{elt}(2))
      if elt <: Real
        @test cbrt.(a_ones) ≡ DiagonalMatrix(Ones{elt}(2))
        @test cbrt.(a_zeros) ≡ DiagonalMatrix(Zeros{elt}(2))
      end
      @test exp.(a_ones) ≡ DiagonalMatrix(Fill(exp(one(elt)), 2))
      @test exp.(a_zeros) ≡ DiagonalMatrix(Ones{typeof(exp(zero(elt)))}(2))
      @test cis.(a_ones) ≡ DiagonalMatrix(Fill(cis(one(elt)), 2))
      @test cis.(a_zeros) ≡ DiagonalMatrix(Ones{typeof(cis(zero(elt)))}(2))
      @test log.(a_ones) ≡ DiagonalMatrix(Zeros{typeof(log(one(elt)))}(2))
      @test log.(a_zeros) ≡ DiagonalMatrix(Fill(log(zero(elt)), 2))
      @test cos.(a_ones) ≡ DiagonalMatrix(Fill(cos(one(elt)), 2))
      @test cos.(a_zeros) ≡ DiagonalMatrix(Ones{typeof(cos(zero(elt)))}(2))
      @test sin.(a_ones) ≡ DiagonalMatrix(Fill(sin(one(elt)), 2))
      @test sin.(a_zeros) ≡ DiagonalMatrix(Zeros{typeof(sin(zero(elt)))}(2))
      @test tan.(a_ones) ≡ DiagonalMatrix(Fill(tan(one(elt)), 2))
      @test tan.(a_zeros) ≡ DiagonalMatrix(Zeros{typeof(tan(zero(elt)))}(2))
      @test sec.(a_ones) ≡ DiagonalMatrix(Fill(sec(one(elt)), 2))
      @test sec.(a_zeros) ≡ DiagonalMatrix(Ones{typeof(sec(zero(elt)))}(2))
      @test cosh.(a_ones) ≡ DiagonalMatrix(Fill(cosh(one(elt)), 2))
      @test cosh.(a_zeros) ≡ DiagonalMatrix(Ones{typeof(cosh(zero(elt)))}(2))
    end
    @testset "LinearAlgebra matrix properties" begin
      @test ishermitian(DiagonalMatrix([1, 2]))
      @test !ishermitian(DiagonalMatrix([1, 2], (2, 3)))
      @test !ishermitian(DiagonalMatrix([1 + 1im, 2 + 2im]))
      @test ishermitian(DiagonalMatrix([ones(2, 2), ones(3, 3)]))
      @test !ishermitian(DiagonalMatrix([randn(2, 2), randn(3, 3)]))

      @test issymmetric(DiagonalMatrix([1, 2]))
      @test !issymmetric(DiagonalMatrix([1, 2], (2, 3)))
      @test issymmetric(DiagonalMatrix([1 + 1im, 2 + 2im]))
      @test issymmetric(DiagonalMatrix([ones(2, 2), ones(3, 3)]))
      @test !issymmetric(DiagonalMatrix([randn(2, 2), randn(3, 3)]))
      @test !issymmetric(DiagonalMatrix([randn(2, 2), randn(2, 3)]))

      @test isposdef(DiagonalMatrix([1, 2]))
      @test !isposdef(DiagonalMatrix([1, -2]))
      @test !isposdef(DiagonalMatrix([1, 2], (2, 3)))
      @test !isposdef(DiagonalMatrix([1 + 1im, 2 + 2im]))
      @test isposdef(DiagonalMatrix([[1 0; 0 1], [2 0; 0 2]]))
      @test !isposdef(DiagonalMatrix([randn(2, 2), randn(3, 3)]))
      @test !isposdef(DiagonalMatrix([randn(2, 2), randn(2, 3)]))
    end
    @testset "LinearAlgebra matrix functions" begin
      diag = randn(elt, 2)
      a = DiagonalMatrix(diag)
      @test tr(a) ≈ sum(diag)
      @test det(a) ≈ prod(diag)

      # Use a positive diagonal in order to take the `log`.
      diag = rand(elt, 2)
      a = DiagonalMatrix(diag)
      @test real(logdet(a)) ≈ real(sum(log, diag))
      @test imag(logdet(a)) ≈ rem2pi(imag(sum(log, diag)), RoundNearest)

      for f in [
        :exp,
        :cis,
        :log,
        :sqrt,
        :cos,
        :sin,
        :tan,
        :csc,
        :sec,
        :cot,
        :cosh,
        :sinh,
        :tanh,
        :csch,
        :sech,
        :coth,
        :acos,
        :asin,
        :atan,
        :acot,
        :asinh,
        :atanh,
        :acsch,
        :asech,
      ]
        @eval begin
          a = DiagonalMatrix(rand($elt, 2))
          @test $f(a) ≈ DiagonalMatrix($f.(diagview(a)))
        end
      end

      for f in [:acsc, :asec, :acosh, :acoth]
        @eval begin
          a = DiagonalMatrix(inv.(rand($elt, 2)))
          @test $f(a) ≈ DiagonalMatrix($f.(diagview(a)))
        end
      end

      if elt <: Real
        a = DiagonalMatrix(randn(elt, 2))
        @test cbrt(a) ≈ DiagonalMatrix(cbrt.(diagview(a)))
      end

      a = DiagonalMatrix(randn(elt, 2))
      @test inv(a) ≈ DiagonalMatrix(inv.(diagview(a)))

      a = DiagonalMatrix(randn(elt, 2))
      @test pinv(a) ≈ DiagonalMatrix(pinv.(diagview(a)))
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

      a_dest = DiagonalArray{elt}(undef, (2, 4))
      mul!(a_dest, a1, a2)
      @test Array(a_dest) ≈ Array(a1) * Array(a2)

      a_dest = DiagonalArray(randn(elt, 2), (2, 4))
      a_dest′ = copy(a_dest)
      mul!(a_dest′, a1, a2, 2, 3)
      @test Array(a_dest′) ≈ Array(a1) * Array(a2) * 2 + Array(a_dest) * 3

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
        (delta(2, 2), Float64),
        (delta(Base.OneTo(2), Base.OneTo(2)), Float64),
        (δ(2, 2), Float64),
        (δ(Base.OneTo(2), Base.OneTo(2)), Float64),
        (delta((2, 2)), Float64),
        (delta(Base.OneTo.((2, 2))), Float64),
        (δ((2, 2)), Float64),
        (δ(Base.OneTo.((2, 2))), Float64),
        (delta(Bool, 2, 2), Bool),
        (delta(Bool, Base.OneTo(2), Base.OneTo(2)), Bool),
        (Delta{Bool}((2, 2)), Bool),
        (Delta{Bool}(Base.OneTo.((2, 2))), Bool),
        (δ(Bool, 2, 2), Bool),
        (δ(Bool, Base.OneTo(2), Base.OneTo(2)), Bool),
        (delta(Bool, (2, 2)), Bool),
        (delta(Bool, Base.OneTo.((2, 2))), Bool),
        (δ(Bool, (2, 2)), Bool),
        (δ(Bool, Base.OneTo.((2, 2))), Bool),
      )
        @test eltype(a) === elt′
        @test diaglength(a) == 2
        @test a isa DiagonalArray{elt′,2}
        @test a isa DiagonalMatrix{elt′}
        @test a isa Delta{elt′,2}
        @test a isa DeltaMatrix{elt′}
        @test size(a) == (2, 2)
        @test diaglength(a) == 2
        @test storedlength(a) == 2
        @test a == DiagonalArray(ones(2), (2, 2))
        @test diagview(a) == ones(2)
        @test diagview(a) isa Ones{elt′}
        @test copy(a) ≡ a

        a′ = 2a
        @test diagview(a′) == 2ones(2)
        # TODO: Fix this. Mapping doesn't preserve
        # the diagonal structure properly.
        # https://github.com/ITensor/DiagonalArrays.jl/issues/7
        @test diagview(a′) isa Fill{promote_type(Int, elt′)}
        @test a′ isa ScaledDelta{promote_type(Int, elt′),2}
        @test a′ isa ScaledDeltaMatrix{promote_type(Int, elt′)}

        b = randn(elt, (2, 3))
        a_dest = a * b
        @test a_dest ≈ Array(a) * Array(b)

        a_dest = a * a
        @test a_dest ≈ Array(a) * Array(a)
        @test diagview(a_dest) isa Ones{elt′}
      end
    end
  end
end
