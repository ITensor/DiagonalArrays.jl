using Adapt: adapt
using DiagonalArrays: DiagonalArray, delta, diagview, δ
using JLArrays: JLArray
using NamedDimsArrays: denamed, inds, nameddims, namedoneto
using SparseArraysBase: dense
using Test: @test, @test_broken, @testset

@testset "NamedDimsArrayExt (eltype=$elt, arraytype=$arrayt)" for elt in
        (Float64, ComplexF64),
        arrayt in (Array, JLArray)

    dev = adapt(arrayt)
    @testset "delta" begin
        i, j = namedoneto.((2, 2), (:i, :j))
        for a in (
                delta(i, j),
                delta(Bool, i, j),
                delta((i, j)),
                delta(Bool, (i, j)),
                δ(i, j),
                δ(Bool, i, j),
                δ((i, j)),
                δ(Bool, (i, j)),
            )
            @test eltype(a) ≡ Bool
            # TODO: Fix this.
            @test_broken diagview(a)
            @test diagview(denamed(a)) == ones(2)
        end
    end
    @testset "DiagonalArrays" begin
        s = dev(DiagonalArray(randn(elt, 3), (3, 3)))
        a = nameddims(s, (:a, :b))
        b = dense(a)
        @test denamed(b) == dense(denamed(a))
        @test denamed(b) isa arrayt{elt, 2}
        @test inds(b) == inds(a)
    end
end
