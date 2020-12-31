# [FDM: Finite Difference Methods](http://github.com/wesselb/fdm)

[![CI](https://github.com/wesselb/fdm/workflows/CI/badge.svg?branch=master)](https://github.com/wesselb/fdm/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/fdm/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/fdm?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/fdm)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

FDM estimates derivatives with finite differences.
See also [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl).

* [Installation](#installation)
* [Multivariate Derivatives](#multivariate-derivatives)
    - [Gradients](#gradients)
    - [Jacobians](#jacobians)
    - [Jacobian-Vector Products (Directional Derivatives)](#jacobian-vector-products-directional-derivatives)
    - [Hessian-Vector Products](#hessian-vector-products)
* [Scalar Derivatives](#scalar-derivatives)
* [Testing Sensitivities in a Reverse-Mode Automatic Differentation Framework](#testing-sensitivities-in-a-reverse-mode-automatic-differentation-framework)

## Installation

FDM requires Python 3.6 or higher.

```bash
pip install fdm
```

## Multivariate Derivatives

```python
from fdm import gradient, jacobian, jvp, hvp
```

For the purpose of illustration, let us consider a quadratic function:

```python
>>> a = np.random.randn(3, 3); a = a @ a.T
>>> a
array([[ 3.57224794,  0.22646662, -1.80432262],
       [ 0.22646662,  4.72596213,  3.46435663],
       [-1.80432262,  3.46435663,  3.70938152]])
       
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
array([-1.38778668, 20.07146076, 16.25253519])

>>> a @ x
array([-1.38778668, 20.07146076, 16.25253519])
```

### Jacobians

```python
>>> jac = jacobian(f)
>>> jac(x)
array([[-1.38778668, 20.07146076, 16.25253519]])

>>> a @ x
array([-1.38778668, 20.07146076, 16.25253519])
```

But `jacobian` also works for multi-valued functions.

```python
>>> def f2(x):
...     return a @ x

>>> jac2 = jacobian(f2)
>>> jac2(x)
array([[ 3.57224794,  0.22646662, -1.80432262],
       [ 0.22646662,  4.72596213,  3.46435663],
       [-1.80432262,  3.46435663,  3.70938152]])
       
>>> a
array([[ 3.57224794,  0.22646662, -1.80432262],
       [ 0.22646662,  4.72596213,  3.46435663],
       [-1.80432262,  3.46435663,  3.70938152]])
```

### Jacobian-Vector Products (Directional Derivatives)

In the scalar case, `jvp` computes directional derivatives:

```python
>>> v = np.array([0.5, 0.6, 0.7])  # A direction

>>> dir_deriv = jvp(f, v) 
>>> dir_deriv(x)
22.725757753354657

>>> np.sum(grad(x) * v)
22.72575775335481
```

In the multivariate case, `jvp` generalises to Jacobian-vector products:

```python
>>> prod = jvp(f2, v)
>>> prod(x)
array([0.65897811, 5.37386023, 3.77301973])

>>> a @ v
array([0.65897811, 5.37386023, 3.77301973])
```

### Hessian-Vector Products

```python
>>> prod = hvp(f, v)
>>> prod(x)
array([[0.6589781 , 5.37386023, 3.77301973]])

>>> 0.5 * (a + a.T) @ v
array([0.65897811, 5.37386023, 3.77301973])
```

## Scalar Derivatives
```python
>>> from fdm import central_fdm
```

Let's try to estimate the first derivative of `np.sin` at `1` with a 
second-order method, where we know that `np.sin` is well conditioned.

```python
>>> central_fdm(order=2, deriv=1, condition=1)(np.sin, 1) - np.cos(1)  
-3.2093916413344914e-10
```

And let's try to estimate the second derivative of `np.sin` at `1` with a 
third-order method.

```python
>>> central_fdm(order=3, deriv=2, condition=1)(np.sin, 1) + np.sin(1)  
-2.4126405273605656e-07
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
7.343652562575157e-10
```

And let's estimate the second derivative of `np.sin` at `1` with a 
fifth-order method.

```python
>>> central_fdm(order=5, deriv=2, condition=1)(np.sin, 1) + np.sin(1)   
-9.530454203598993e-11
```

Hooray!

Finally, let us verify that increasing the order generally increases the accuracy.

```python
>>> for i in range(3, 16):
...      print(central_fdm(order=i, deriv=2, condition=1)(np.sin, 1) + np.sin(1))
-2.4126405273605656e-07
1.0358558566458953e-08
-9.530454203598993e-11
1.8791523892502937e-11
-1.1972645097557688e-12
-7.390754674929667e-13
2.3780977187470853e-13
1.127986593019159e-13
-7.760458942129844e-14
1.092459456231154e-13
-1.687538997430238e-14
9.43689570931383e-15
2.886579864025407e-15
```

## Testing Sensitivities in a Reverse-Mode Automatic Differentation Framework

Consider the function

```python
def mul(a, b):
    return a * b
```

and its sensitivity

```python
def s_mul(s_y, y, a, b):
    return s_y * b, a * s_y
```

The sensitivity `s_mul` takes in the sensitivity `s_y` of the output `y`, 
the output `y`, and  the arguments of the function `mul`; and returns a tuple 
containing the sensitivities with respect to `a` and `b`.
Then function `check_sensitivity` can be used to assert that the 
implementation of `s_mul` is correct:

```python
>>> from fdm import check_sensitivity

>>> check_sensitivity(mul, s_mul, (2, 3))  # Test at arguments `2` and `3`.
```

Suppose that the implementation were wrong, for example

```python
def s_mul_wrong(s_y, y, a, b):
    return s_y * b, b * s_y  # Used `b` instead of `a` for the second sensitivity!
```

Then `check_sensitivity` should throw an `AssertionError`:

```python
>>> check_sensitivity(mul, s_mul, (2, 3)) 
AssertionError: Sensitivity of argument 2 of function "mul" did not match numerical estimate.
```
