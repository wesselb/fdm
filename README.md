# [FDM: Finite Difference Methods](http://github.com/wesselb/fdm)

[![Build](https://travis-ci.org/wesselb/fdm.svg?branch=master)](https://travis-ci.org/wesselb/fdm)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/fdm/badge.svg?branch=master)](https://coveralls.io/github/wesselb/fdm?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://fdm-docs.readthedocs.io/en/latest)

FDM estimates derivatives with finite differences.

See also [FDM.jl](https://github.com/invenia/FDM.jl).

## Basic Usage
```python
>>> from fdm import central_fdm
```

Let's try to estimate the first derivative of `np.sin` at `1` with a 
second-order method, where we know that `np.sin` is well conditioned.

```python
>>> central_fdm(order=2, deriv=1, condition=1)(np.sin, 1) - np.cos(1)  
9.681416779372398e-10
```

And let's try to estimate the second derivative of `np.sin` at `1` with a 
third-order method.

```python
>>> central_fdm(order=3, deriv=2, condition=1)(np.sin, 1) + np.sin(1)  
-2.1165562702485374e-07
```

Hm.
Let's check the accuracy of this third-order method.
The step size and accuracy of the method are computed upon calling
`FDM.estimate(A)`.

```python
>>> central_fdm(order=3, deriv=2, condition=1).estimate().acc
8.733476581980376e-06
```

We might want a little more accuracy. Let's check the accuracy of a 
fifth-order method.

```python
>>> central_fdm(order=5, deriv=2, condition=1).estimate().acc
7.343652562575155e-10
```

And let's estimate the second derivative of `np.sin` at `1` with a 
fifth-order method.

```python
>>> central_fdm(order=5, deriv=2, condition=1)(np.sin, 1) + np.sin(1)   
-5.742628594873622e-12
```

Hooray!

Finally, let us verify that increasing the order indeed reliably increases 
the accuracy.

```python
>>> for i in range(3, 11):
...      central_fdm(order=i, deriv=2, condition=1)(np.sin, 1) + np.sin(1)
-2.1165562702485374e-07
7.855732286898842e-09
-5.742628594873622e-12
2.0503265751870003e-11
3.3451019731955967e-13
1.0655920590352252e-12
-7.194245199571014e-14
8.37108160567368e-14
```

Moar? See the [docs](https://fdm-docs.readthedocs.io/en/latest)!