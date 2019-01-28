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

Moar? See the [docs](https://fdm-docs.readthedocs.io/en/latest)!