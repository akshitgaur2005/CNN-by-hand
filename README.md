# Convolutional Neural Network by hand using Numpy

In a nutshell, a Convolution layer transforms an n-dimensional tensor to another m-dimensional tensor. One thing to be kept in mind is that the tern [Convolution](https://en.wikipedia.org/wiki/Convolution)
is not accurate to this (or most for that matter) implementation. Mathematically speaking, the function we are using is [Cross-Correlation](https://en.wikipedia.org/wiki/Cross-correlation).
$$Z_{i, j} = \sum X_{i + u, j + v} \circ W_{u, v}$$

The main difference between Convolution and Cross-correlation is that in a convolution we would have to take transpose of the weights first which would provide no imporvement for us as weights are randomly initialised anyway.

For example, a Convolution layer may transform a tensor(matrix) of shape (3, 28, 28) to a tensor of shape (16, 14, 14).