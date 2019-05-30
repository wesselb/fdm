`FDM: Finite Difference Methods <http://github.com/wesselb/fdm>`__
==================================================================

|Build| |Coverage Status| |Latest Docs|

FDM estimates derivatives with finite differences.

See also `FDM.jl <https://github.com/invenia/FDM.jl>`__.

See the `docs <https://wesselb.github.io/fdm>`__.

-  `Installation <#installation>`__
-  `Multivariate Derivatives <#multivariate-derivatives>`__

   -  `Gradients <#gradients>`__
   -  `Directional Derivatives <#directional-derivatives>`__
   -  `Jacobians <#jacobians>`__
   -  `Hessian-Vector Products <#hessian-vector-products>`__

-  `Scalar Derivatives <#scalar-derivatives>`__
-  `Testing Sensitivities in a Reverse-Mode Automatic Differentation
   Framework <#testing-sensitivities-in-a-reverse-mode-automatic-differentation-framework>`__

Installation
------------

The package is tested for Python 2.7 and Python 3.6, which are the
versions recommended to use. To install the package, use the following
sequence of commands:

::

    git clone https://github.com/wesselb/fdm
    cd fdm
    make install

Multivariate Derivatives
------------------------

.. code:: python

    from fdm import gradient, directional, jacobian, jvp, hvp

For the purpose of illustration, let us consider a quadratic function:

.. code:: python

    >>> a = np.random.randn(3, 3); a = a @ a.T
    >>> a
    array([[ 3.57224794,  0.22646662, -1.80432262],
           [ 0.22646662,  4.72596213,  3.46435663],
           [-1.80432262,  3.46435663,  3.70938152]])
           
    >>> def f(x):
    ...     return 0.5 * x @ a @ x

Consider the following input value:

.. code:: python

    >>> x = np.array([1.0, 2.0, 3.0])

Gradients
~~~~~~~~~

.. code:: python

    >>> grad = gradient(f)
    >>> grad(x)
    array([-1.38778668, 20.07146076, 16.25253519])

    >>> a @ x
    array([-1.38778668, 20.07146076, 16.25253519])

Directional Derivatives
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> v = np.array([0.5, 0.6, 0.7])  # A direction

    >>> dir_deriv = directional(f, v)
    >>> dir_deriv(x)
    22.725757753354657

    >>> np.sum(grad(x) * v)
    22.72575775335481

Jacobians
~~~~~~~~~

.. code:: python


    >>> jac = jacobian(f)
    >>> jac(x)
    array([[-1.38778668, 20.07146076, 16.25253519]])

    >>> a @ x
    array([-1.38778668, 20.07146076, 16.25253519])

But ``jacobian`` also works for multi-valued functions.

.. code:: python

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

Jacobian-Vector Products
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> prod = jvp(f2, v)
    >>> prod(x)
    array([0.65897811, 5.37386023, 3.77301973])

    >>> a @ v
    array([0.65897811, 5.37386023, 3.77301973])

Hessian-Vector Products
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> prod = hvp(f, v)
    >>> prod(x)
    array([[0.6589781 , 5.37386023, 3.77301973]])

    >>> 0.5 * (a + a.T) @ v
    array([0.65897811, 5.37386023, 3.77301973])

Scalar Derivatives
------------------

.. code:: python

    >>> from fdm import central_fdm

Let's try to estimate the first derivative of ``np.sin`` at ``1`` with a
second-order method, where we know that ``np.sin`` is well conditioned.

.. code:: python

    >>> central_fdm(order=2, deriv=1, condition=1)(np.sin, 1) - np.cos(1)  
    4.307577627926662e-10

And let's try to estimate the second derivative of ``np.sin`` at ``1``
with a third-order method.

.. code:: python

    >>> central_fdm(order=3, deriv=2, condition=1)(np.sin, 1) + np.sin(1)  
    -1.263876664436836e-07

Hm. Let's check the accuracy of this third-order method. The step size
and accuracy of the method are computed upon calling ``FDM.estimate()``.

.. code:: python

    >>> central_fdm(order=3, deriv=2, condition=1).estimate().acc
    8.733476581980376e-06

We might want a little more accuracy. Let's check the accuracy of a
fifth-order method.

.. code:: python

    >>> central_fdm(order=5, deriv=2, condition=1).estimate().acc
    7.343652562575155e-10

And let's estimate the second derivative of ``np.sin`` at ``1`` with a
fifth-order method.

.. code:: python

    >>> central_fdm(order=5, deriv=2, condition=1)(np.sin, 1) + np.sin(1)   
    -9.145184609593571e-11

Hooray!

Finally, let us verify that increasing the order indeed reliably
increases the accuracy.

.. code:: python

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

Testing Sensitivities in a Reverse-Mode Automatic Differentation Framework
--------------------------------------------------------------------------

Consider the function

.. code:: python

    def mul(a, b):
        return a * b

and its sensitivity

.. code:: python

    def s_mul(s_y, y, a, b):
        return s_y * b, a * s_y

The sensitivity ``s_mul`` takes in the sensitivity ``s_y`` of the output
``y``, the output ``y``, and the arguments of the function ``mul``; and
returns a tuple containing the sensitivities with respect to ``a`` and
``b``. Then function ``check_sensitivity`` can be used to assert that
the implementation of ``s_mul`` is correct:

.. code:: python

    >>> from fdm import check_sensitivity

    >>> check_sensitivity(mul, s_mul, (2, 3))  # Test at arguments `2` and `3`.

Suppose that the implementation were wrong, for example

.. code:: python

    def s_mul_wrong(s_y, y, a, b):
        return s_y * b, b * s_y  # Used `b` instead of `a` for the second sensitivity!

Then ``check_sensitivity`` should throw an ``AssertionError``:

.. code:: python

    >>> check_sensitivity(mul, s_mul, (2, 3)) 
    AssertionError: Sensitivity of argument 2 of function "mul" did not match numerical estimate.

.. |Build| image:: https://travis-ci.org/wesselb/fdm.svg?branch=master
   :target: https://travis-ci.org/wesselb/fdm
.. |Coverage Status| image:: https://coveralls.io/repos/github/wesselb/fdm/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/wesselb/fdm?branch=master
.. |Latest Docs| image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://wesselb.github.io/fdm
