# DiagonalArrays.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://itensor.github.io/DiagonalArrays.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://itensor.github.io/DiagonalArrays.jl/dev/)
[![Build Status](https://github.com/ITensor/DiagonalArrays.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/DiagonalArrays.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/DiagonalArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/DiagonalArrays.jl)
[![Code Style](https://img.shields.io/badge/code_style-ITensor-purple)](https://github.com/ITensor/ITensorFormatter.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A n-dimensional diagonal array type in Julia.

## Support

<picture>
  <source media="(prefers-color-scheme: dark)" width="20%" srcset="docs/src/assets/CCQ-dark.png">
  <img alt="Flatiron Center for Computational Quantum Physics logo." width="20%" src="docs/src/assets/CCQ.png">
</picture>


DiagonalArrays.jl is supported by the Flatiron Institute, a division of the Simons Foundation.

## Installation instructions

This package resides in the `ITensor/ITensorRegistry` local registry.
In order to install, simply add that registry through your package manager.
This step is only required once.
```julia
julia> using Pkg: Pkg

julia> Pkg.Registry.add(url="https://github.com/ITensor/ITensorRegistry")
```
or:
```julia
julia> Pkg.Registry.add(url="git@github.com:ITensor/ITensorRegistry.git")
```
if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.

Then, the package can be added as usual through the package manager:

```julia
julia> Pkg.add("DiagonalArrays")
```

## Examples

````julia
using DiagonalArrays:
    DiagIndex, DiagIndices, DiagonalArray, DiagonalMatrix, diaglength, isdiagindex
using Test: @test

function main()
    d = DiagonalMatrix([1.0, 2.0, 3.0])
    @test eltype(d) == Float64
    @test diaglength(d) == 3
    @test size(d) == (3, 3)
    @test d[1, 1] == 1
    @test d[2, 2] == 2
    @test d[3, 3] == 3
    @test d[1, 2] == 0

    d = DiagonalArray([1.0, 2.0, 3.0], 3, 4, 5)
    @test eltype(d) == Float64
    @test diaglength(d) == 3
    @test d[1, 1, 1] == 1
    @test d[2, 2, 2] == 2
    @test d[3, 3, 3] == 3
    @test d[1, 2, 1] == 0

    d[2, 2, 2] = 22
    @test d[2, 2, 2] == 22

    d_r = reshape(d, 3, 20)
    @test size(d_r) == (3, 20)
    @test all(I -> d_r[I] == d[I], LinearIndices(d))

    @test length(d[DiagIndices(:)]) == 3
    @test Array(d) == d
    @test d[DiagIndex(2)] == d[2, 2, 2]

    d[DiagIndex(2)] = 222
    @test d[2, 2, 2] == 222

    a = randn(3, 4, 5)
    new_diag = randn(3)
    a[DiagIndices(:)] = new_diag
    d[DiagIndices(:)] = a[DiagIndices(:)]

    @test a[DiagIndices(:)] == new_diag
    @test d[DiagIndices(:)] == new_diag

    permuted_d = permutedims(d, (3, 2, 1))
    @test permuted_d isa DiagonalArray
    @test permuted_d[DiagIndices(:)] == d[DiagIndices(:)]
    @test size(d) == (3, 4, 5)
    @test size(permuted_d) == (5, 4, 3)
    for I in eachindex(d)
        if !isdiagindex(d, I)
            @test iszero(d[I])
        else
            @test !iszero(d[I])
        end
    end

    mapped_d = map(x -> 2x, d)
    @test mapped_d isa DiagonalArray
    @test mapped_d == map(x -> 2x, Array(d))

    return nothing
end

main()
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

