var documenterSearchIndex = {"docs":
[{"location":"library/#Library","page":"Library","title":"Library","text":"","category":"section"},{"location":"library/","page":"Library","title":"Library","text":"Modules = [DiagonalArrays]","category":"page"},{"location":"library/#DiagonalArrays.diagonal-Tuple{AbstractVector}","page":"Library","title":"DiagonalArrays.diagonal","text":"diagonal(v::AbstractVector) -> AbstractMatrix\n\nReturn a diagonal matrix from a vector v. This is an extension of LinearAlgebra.Diagonal, designed to avoid the implication of the output type. Defaults to Diagonal(v).\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"../../examples/README.jl\"","category":"page"},{"location":"#DiagonalArrays.jl","page":"Home","title":"DiagonalArrays.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Stable) (Image: Dev) (Image: Build Status) (Image: Coverage) (Image: Code Style: Blue) (Image: Aqua)","category":"page"},{"location":"","page":"Home","title":"Home","text":"A n-dimensional diagonal array type in Julia.","category":"page"},{"location":"#Installation-instructions","page":"Home","title":"Installation instructions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package resides in the ITensor/ITensorRegistry local registry. In order to install, simply add that registry through your package manager. This step is only required once.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using Pkg: Pkg\n\njulia> Pkg.Registry.add(url=\"https://github.com/ITensor/ITensorRegistry\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"or:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> Pkg.Registry.add(url=\"git@github.com:ITensor/ITensorRegistry.git\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Then, the package can be added as usual through the package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> Pkg.add(\"DiagonalArrays\")","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using DiagonalArrays:\n  DiagonalArray, DiagonalMatrix, DiagIndex, DiagIndices, diaglength, isdiagindex\nusing Test: @test\n\nfunction main()\n  d = DiagonalMatrix([1.0, 2.0, 3.0])\n  @test eltype(d) == Float64\n  @test diaglength(d) == 3\n  @test size(d) == (3, 3)\n  @test d[1, 1] == 1\n  @test d[2, 2] == 2\n  @test d[3, 3] == 3\n  @test d[1, 2] == 0\n\n  d = DiagonalArray([1.0, 2.0, 3.0], 3, 4, 5)\n  @test eltype(d) == Float64\n  @test diaglength(d) == 3\n  @test d[1, 1, 1] == 1\n  @test d[2, 2, 2] == 2\n  @test d[3, 3, 3] == 3\n  @test d[1, 2, 1] == 0\n\n  d[2, 2, 2] = 22\n  @test d[2, 2, 2] == 22\n\n  d_r = reshape(d, 3, 20)\n  @test size(d_r) == (3, 20)\n  @test all(I -> d_r[I] == d[I], LinearIndices(d))\n\n  @test length(d[DiagIndices(:)]) == 3\n  @test Array(d) == d\n  @test d[DiagIndex(2)] == d[2, 2, 2]\n\n  d[DiagIndex(2)] = 222\n  @test d[2, 2, 2] == 222\n\n  a = randn(3, 4, 5)\n  new_diag = randn(3)\n  a[DiagIndices(:)] = new_diag\n  d[DiagIndices(:)] = a[DiagIndices(:)]\n\n  @test a[DiagIndices(:)] == new_diag\n  @test d[DiagIndices(:)] == new_diag\n\n  permuted_d = permutedims(d, (3, 2, 1))\n  @test permuted_d isa DiagonalArray\n  @test permuted_d[DiagIndices(:)] == d[DiagIndices(:)]\n  @test size(d) == (3, 4, 5)\n  @test size(permuted_d) == (5, 4, 3)\n  for I in eachindex(d)\n    if !isdiagindex(d, I)\n      @test iszero(d[I])\n    else\n      @test !iszero(d[I])\n    end\n  end\n\n  mapped_d = map(x -> 2x, d)\n  @test mapped_d isa DiagonalArray\n  @test mapped_d == map(x -> 2x, Array(d))\n\n  return nothing\nend\n\nmain()","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"This page was generated using Literate.jl.","category":"page"}]
}
