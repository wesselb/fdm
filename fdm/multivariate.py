# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import numpy as np

from .fdm import central_fdm

__all__ = ['directional',
           'gradient',
           'jacobian',
           'hvp',
           'default_adaptive_method',
           'default_robust_method']
log = logging.getLogger(__name__)

default_adaptive_method = central_fdm(order=5, deriv=1, adapt=1)
""":class:`.fdm.FDM`: Default adaptive method."""

default_robust_method = central_fdm(order=5,
                                    deriv=1,
                                    adapt=0,
                                    condition=1e4)
""":class:`.fdm.FDM`: Default robust method."""


def directional(f, v, method=default_adaptive_method):
    """Compute the directional derivative of `f` in the direction `v`.

    Args:
        f (function): Function to compute directional derivative of.
        v (tensor): Direction.
        method (:class:`.fdm.FDM`, optional): Finite difference method to use.
            Defaults to :data:`.multivariate.default_adaptive_method`.

    Returns:
        function: Directional derivative.
    """

    def compute_derivative(x):
        zero = np.array(0).astype(np.array(x).dtype)
        return method(lambda eps: f(x + eps * v), zero)

    return compute_derivative


def gradient(f, method=default_adaptive_method):
    """Compute the gradient of a `f`.

    The function `f` should be scalar valued.

    Args:
        f (function): Function to compute gradient of. This should be a
            scalar-valued function.
        method (:class:`.fdm.FDM`, optional): Finite difference method to use.
            Defaults to :data:`.multivariate.default_adaptive_method`.

    Returns:
        function: Gradient.
    """

    def compute_gradient(x):
        x = np.array(x)  # Cast to a NumPy object.

        # Handle edge case where `x` is a scalar.
        if x.shape == ():
            return method(f, x)

        # Construct the gradient.
        grad = np.empty_like(x)

        # Loop over a basis for `x` to compute the gradient.
        e = np.zeros_like(x)
        one = np.array(1).astype(x.dtype)
        zero = np.array(0).astype(x.dtype)

        for i in range(e.size):
            # Linearly index into the array.
            ind = np.unravel_index(i, e.shape)

            # Compute element of the gradient.
            e[ind] = one
            grad[ind] = directional(f, e, method)(x)
            e[ind] = zero

        return grad

    return compute_gradient


def jacobian(f, method=default_adaptive_method):
    """Compute the Jacobian of `f`.

    Args:
        f (function): Function to compute Jacobian of.
        method (:class:`.fdm.FDM`, optional): Finite difference method to use.
            Defaults to :data:`.multivariate.default_adaptive_method`.

    Returns:
        function: Jacobian.
    """

    def compute_jacobian(x):
        x = np.array(x)  # Cast to a NumPy object.

        # Query the function once to check the size of the output.
        size_in = np.size(x)
        size_out = np.size(f(x))

        # Construct the Jacobian.
        jac = np.empty((size_out, size_in), dtype=x.dtype)

        # Loop over outputs to fill the Jacobian.
        for i in range(size_out):
            grad = gradient(lambda y: _get_at_index(f(y), i), method)
            jac[i, :] = np.reshape(grad(x), size_in)

        return jac

    return compute_jacobian


def _get_at_index(x, i):
    if x.shape == ():
        if i == 0:
            return x
        else:
            raise RuntimeError('Cannot index into a scalar.')
    else:
        return x[np.unravel_index(i, x.shape)]


def hvp(f,
        v,
        jac_method=default_adaptive_method,
        dir_method=default_robust_method):
    """Compute a Hessian--vector product.

    Args:
        f (function): Function to compute Hessian of.
        v (tensor): Vector to multiply Hessian with. Should be of the same form
            as the arguments supplied to `f`.
        jac_method (:class:`.fdm.FDM`, optional): Finite difference method to
            use for Jacobian computation. Defaults to
            :data:`.multivariate.default_adaptive_method`.
        dir_method (:class:`.fdm.FDM`, optional): Finite difference method to
            use for directional derivative computation. Defaults to
            :data:`.multivariate.default_robust_method`.

    Returns:
        function: Hessian of `f` multiplied by `v`.
    """
    return directional(jacobian(f, jac_method), v, dir_method)