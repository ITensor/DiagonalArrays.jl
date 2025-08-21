# TODO: Define `TensorProducts.dual`.
dual(x) = x
issquare(a::AbstractMatrix) = (axes(a, 1) == dual(axes(a, 2)))
