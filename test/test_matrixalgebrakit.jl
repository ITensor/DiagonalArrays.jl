using Test: @test, @testset
using LinearAlgebra: Diagonal
using DiagonalArrays: DiagonalArrays, DeltaMatrix, ScaledDeltaMatrix, δ, dual
using FillArrays: Ones
using MatrixAlgebraKit:
  eig_full,
  eig_vals,
  eigh_full,
  eigh_vals,
  left_orth,
  left_polar,
  lq_compact,
  lq_full,
  qr_compact,
  qr_full,
  right_orth,
  right_polar,
  svd_compact,
  svd_full,
  svd_vals
using StableRNGs: StableRNG

struct SU2 <: AbstractUnitRange{Int}
  j::Int
  isdual::Bool
end
SU2(j::Int) = SU2(j, false)
Base.:(==)(s1::SU2, s2::SU2) = ((s1.j == s2.j) && (s1.isdual == s2.isdual))
Base.first(s::SU2) = 1
Base.last(s::SU2) = 2 * s.j + 1
DiagonalArrays.dual(s::SU2) = SU2(s.j, !s.isdual)

@testset "MatrixAlgebraKitExt" begin
  @testset "DeltaMatrix factorizations (eltype=$elt)" for elt in (
    Float32, Float64, ComplexF32, ComplexF64
  )
    @testset "SVD" begin
      for f in (svd_compact, svd_full)
        ax = SU2(2)
        a = δ(elt, (ax, ax))
        u, s, v = f(a)
        @test u * s * v ≡ a
        @test u ≡ δ(elt, (ax, dual(ax)))
        @test s ≡ δ(real(elt), (ax, ax))
        @test v ≡ δ(elt, (dual(ax), ax))
      end
    end
    @testset "SVD values" begin
      ax = SU2(2)
      a = δ(elt, (ax, ax))
      s = svd_vals(a)
      @test s ≡ Ones(real(elt), length(ax))
    end
    @testset "left orth" begin
      for f in (left_orth, left_polar, qr_compact, qr_full)
        ax = SU2(2)
        a = δ(elt, (ax, ax))
        q, r = f(a)
        @test q * r ≡ a
        @test q ≡ δ(elt, (ax, dual(ax)))
        @test r ≡ δ(elt, (ax, ax))
      end
    end
    @testset "right orth" begin
      for f in (lq_compact, lq_full, right_orth, right_polar)
        ax = SU2(2)
        a = δ(elt, (ax, ax))
        l, q = f(a)
        @test l * q ≡ a
        @test l ≡ δ(elt, (ax, ax))
        @test q ≡ δ(elt, (dual(ax), ax))
      end
    end
    @testset "Eigendecomposition" begin
      ax = SU2(2)
      a = δ(elt, (dual(ax), ax))
      d, v = eig_full(a)
      @test a * v ≡ v * d
      @test d ≡ δ(complex(elt), (dual(ax), ax))
      @test v ≡ δ(complex(elt), (dual(ax), ax))
    end
    @testset "Hermitian eigendecomposition" begin
      ax = SU2(2)
      a = δ(elt, (dual(ax), ax))
      d, v = eigh_full(a)
      @test a * v ≡ v * d
      @test d ≡ δ(real(elt), (dual(ax), ax))
      @test v ≡ δ(elt, (dual(ax), ax))
    end
    @testset "Eigenvalues" begin
      ax = SU2(2)
      a = δ(elt, (dual(ax), ax))
      d = eig_vals(a)
      @test d ≡ Ones{complex(elt)}(length(ax))
    end
    @testset "Hermitian eigenvalues" begin
      ax = SU2(2)
      a = δ(elt, (dual(ax), ax))
      d = eigh_vals(a)
      @test d ≡ Ones{real(elt)}(length(ax))
    end
    @testset "left null" begin
      ax = SU2(2)
      a = δ(elt, (ax, ax))
      @test_broken left_null(a)
    end
    @testset "right null" begin
      ax = SU2(2)
      a = δ(elt, (ax, ax))
      @test_broken right_null(a)
    end
  end

  @testset "ScaledDeltaMatrix factorizations (eltype=$elt)" for elt in (
    Float32, Float64, ComplexF32, ComplexF64
  )
    @testset "SVD" begin
      for f in (svd_compact, svd_full)
        ax = SU2(2)
        rng = StableRNG(1234)
        scale = randn(rng, elt)
        a = scale * δ(elt, (ax, ax))
        u, s, v = f(a)
        @test u * s * v ≡ a
        @test u ≡ δ(elt, (ax, dual(ax)))
        @test s ≡ abs(scale) * δ(real(elt), (ax, ax))
        @test v ≡ sign(scale) * δ(elt, (dual(ax), ax))
      end
    end
    @testset "SVD values" begin
      ax = SU2(2)
      rng = StableRNG(1234)
      scale = randn(rng, elt)
      a = scale * δ(elt, (ax, ax))
      s = svd_vals(a)
      @test s ≡ abs(scale) * Ones(real(elt), length(ax))
    end
    @testset "left orth" begin
      for f in (left_orth, left_polar, qr_compact, qr_full)
        ax = SU2(2)
        rng = StableRNG(1234)
        scale = randn(rng, elt)
        a = scale * δ(elt, (ax, ax))
        q, r = f(a)
        @test q * r ≡ a
        @test q ≡ sign(scale) * δ(elt, (ax, dual(ax)))
        @test r ≡ abs(scale) * δ(elt, (ax, ax))
      end
    end
    @testset "right orth" begin
      for f in (lq_compact, lq_full, right_orth, right_polar)
        ax = SU2(2)
        rng = StableRNG(1234)
        scale = randn(rng, elt)
        a = scale * δ(elt, (ax, ax))
        l, q = f(a)
        @test l * q ≡ a
        @test l ≡ abs(scale) * δ(elt, (ax, ax))
        @test q ≡ sign(scale) * δ(elt, (dual(ax), ax))
      end
    end
    @testset "Eigendecomposition" begin
      ax = SU2(2)
      rng = StableRNG(1234)
      scale = randn(rng, elt)
      a = scale * δ(elt, (dual(ax), ax))
      d, v = eig_full(a)
      @test a * v ≡ v * d
      @test d ≡ scale * δ(complex(elt), (dual(ax), ax))
      @test v ≡ δ(complex(elt), (dual(ax), ax))
    end
    @testset "Hermitian eigendecomposition" begin
      ax = SU2(2)
      rng = StableRNG(1234)
      scale = randn(rng, real(elt))
      a = scale * δ(elt, (dual(ax), ax))
      d, v = eigh_full(a)
      @test a * v ≡ v * d
      @test d ≡ scale * δ(real(elt), (dual(ax), ax))
      @test v ≡ δ(elt, (dual(ax), ax))
    end
    @testset "Eigenvalues" begin
      ax = SU2(2)
      rng = StableRNG(1234)
      scale = randn(rng, elt)
      a = scale * δ(elt, (dual(ax), ax))
      d = eig_vals(a)
      @test d ≡ scale * Ones{complex(elt)}(length(ax))
    end
    @testset "Hermitian eigenvalues" begin
      ax = SU2(2)
      rng = StableRNG(1234)
      scale = randn(rng, real(elt))
      a = scale * δ(elt, (dual(ax), ax))
      d = eigh_vals(a)
      @test d ≡ scale * Ones{real(elt)}(length(ax))
    end
    @testset "left null" begin
      ax = SU2(2)
      rng = StableRNG(1234)
      scale = randn(rng, real(elt))
      a = scale * δ(elt, (ax, ax))
      @test_broken left_null(a)
    end
    @testset "right null" begin
      ax = SU2(2)
      rng = StableRNG(1234)
      scale = randn(rng, real(elt))
      a = scale * δ(elt, (ax, ax))
      @test_broken right_null(a)
    end
  end
end
