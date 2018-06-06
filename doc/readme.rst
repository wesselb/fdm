`FDM: Finite Difference Methods <http://github.com/wesselb/fdm>`__
==================================================================

|Build| |Coverage Status| |Latest Docs|

Compute derivatives with finite difference methods

See also `FDM.jl <https://github.com/invenia/FDM.jl>`__.

Basic Usage
-----------

.. code:: python

    >>> from fdm import central_fdm

Let's try to estimate the first derivative of ``np.sin`` at ``1`` with a
second-order method.

.. code:: python

    >>> central_fdm(2, 1)(np.sin, 1) - np.cos(1)  
    9.681416779372398e-10

And let's try to estimate the second derivative of ``np.sin`` at ``1``
with a third-order method.

.. code:: python

    >>> central_fdm(3, 2)(np.sin, 1) + np.sin(1)  
    -2.1165562702485374e-07

Hm. Let's check the accuracy of this third-order method.

.. code:: python

    >>> central_fdm(3, 2).acc
    8.733476581980376e-06

We might want a little more accuracy. Let's check the accuracy of a
fifth-order method.

.. code:: python

    >>> central_fdm(5, 2).acc
    7.343652562575155e-10

And let's estimate the second derivative of ``np.sin`` at ``1`` with a
fifth-order method.

.. code:: python

    >>> central_fdm(5, 2)(np.sin, 1) + np.sin(1)  
    -5.742628594873622e-12

Hooray!

.. |Build| image:: https://travis-ci.org/wesselb/fdm.svg?branch=master
   :target: https://travis-ci.org/wesselb/fdm
.. |Coverage Status| image:: https://coveralls.io/repos/github/wesselb/fdm/badge.svg?branch=master
   :target: https://coveralls.io/github/wesselb/fdm?branch=master
.. |Latest Docs| image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://fdm-docs.readthedocs.io/en/latest
