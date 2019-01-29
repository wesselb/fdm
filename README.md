# [FDM: Finite Difference Methods](http://github.com/wesselb/fdm)

[![Build](https://travis-ci.org/wesselb/fdm.svg?branch=master)](https://travis-ci.org/wesselb/fdm)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/fdm/badge.svg?branch=master)](https://coveralls.io/github/wesselb/fdm?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://fdm-docs.readthedocs.io/en/latest)

FDM estimates derivatives with finite differences.

See also [FDM.jl](https://github.com/invenia/FDM.jl).

See the [docs](https://fdm-docs.readthedocs.io/en/latest)!

## Basic Usage

```python
from fdm import gradient, directional, jacobian, hvp
```

For the purpose of illustration, let us consider a quadratic function:

```python
>>> a = np.random.randn(3, 3); a = a @ a.T
>>> a
array([[ 1.56063275,  0.80421633, -2.35877318],
       [ 0.80421633,  4.22782295,  0.22956733],
       [-2.35877318,  0.22956733,  4.41663688]])
       
>>> def f(x):
...     return 0.5 * x @ a @ x
```

Consider the following input value:

```python
>>> x = np.array([1.0, 2.0, 3.0])
```

### Gradients

```python
>>> grad = gradient(f)
>>> grad(x)
array([-3.90725414,  9.94856421, 11.3502721 ])

>>> a @ x
array([-3.90725414,  9.94856421, 11.3502721 ])
```


### Directional Derivatives

```python
>>> v = np.array([0.5, 0.6, 0.7])  # A direction

>>> dir_deriv = directional(f, v)
>>> dir_deriv(x)
11.960701923320736

>>> np.sum(grad(x) * v)
11.960701923321531
```


### Jacobians

```python

>>> jac = jacobian(f)
>>> jac(x)
array([[-3.90725414,  9.94856421, 11.3502721 ]])

>>> a @ x
array([-3.90725414,  9.94856421, 11.3502721 ])
```

But `jacobian` also works for multi-valued functions.

```python
>>> def f2(x):
...     return a @ x

>>> jac2 = jacobian(f2)
>>> jac2(x)
array([[ 1.56063275,  0.80421633, -2.35877318],
       [ 0.80421633,  4.22782295,  0.22956733],
       [-2.35877318,  0.22956733,  4.41663688]])
       
>>> a
array([[ 1.56063275,  0.80421633, -2.35877318],
       [ 0.80421633,  4.22782295,  0.22956733],
       [-2.35877318,  0.22956733,  4.41663688]])
```

### Hessian-Vector Products

```python
>>> prod = hvp(f, v)
>>> prod(x)
array([[-0.38829506,  3.09949906,  2.04999963]])

>>> 0.5 * (a + a.T) @ v
array([-0.38829506,  3.09949906,  2.04999962])
```



## Low-Level Usage
```python
>>> from fdm import central_fdm
```

Let's try to estimate the first derivative of `np.sin` at `1` with a 
second-order method, where we know that `np.sin` is well conditioned.

```python
>>> central_fdm(order=2, deriv=1, condition=1)(np.sin, 1) - np.cos(1)  
4.307577627926662e-10
```

And let's try to estimate the second derivative of `np.sin` at `1` with a 
third-order method.

```python
>>> central_fdm(order=3, deriv=2, condition=1)(np.sin, 1) + np.sin(1)  
-1.263876664436836e-07
```

Hm.
Let's check the accuracy of this third-order method.
The step size and accuracy of the method are computed upon calling
`FDM.estimate()`.

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
-9.145184609593571e-11
```

Hooray!

Finally, let us verify that increasing the order indeed reliably increases 
the accuracy.

```python
>>> for i in range(3, 11):
...      print(central_fdm(order=i, deriv=2, condition=1)(np.sin, 1) + np.sin(1))
-1.263876664436836e-07
6.341286606925678e-09
-9.145184609593571e-11
2.7335911312320604e-12
6.588063428125679e-13
2.142730437526552e-13
2.057243264630415e-13
8.570921750106208e-14
```
