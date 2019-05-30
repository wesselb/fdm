# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import numpy as np

from .fdm import central_fdm

__all__ = ['gradient',
           'jvp',
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


def _get_at_index(x, i):
    if x.shape == ():
        if i == 0:
            return x
        else:
            raise RuntimeError('Cannot index into a scalar.')
    else:
        return x[np.unravel_index(i, x.shape)]


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
        dtype = np.array(f(x)).dtype  # Query the object once to get the dtype.

        # Handle edge case where `x` is a scalar.
        if np.shape(x) == ():
            return method(f, x)

        # Construct the gradient.
        grad = np.empty_like(x, dtype=dtype)

        # Loop over a basis for `x` to compute the gradient.
        e = np.zeros_like(x)
        one = np.array(1).astype(dtype)
        zero = np.array(0).astype(dtype)

        for i in range(e.size):
            # Linearly index into the array.
            ind = np.unravel_index(i, e.shape)

            # Compute element of the gradient.
            e[ind] = one
            grad[ind] = jvp(f, e, method)(x)
            e[ind] = zero

        return grad

    return compute_gradient


def jvp(f, v, method=default_adaptive_method):
    """Compute a Jacobian-vector product, also known as a directional
    derivative.

    Args:
        f (function): Function to compute Jacobian of.
        v (tensor): Vector to multiply Jacobian with. Should be of the same form
            as the arguments supplied to `f`.
        method (:class:`.fdm.FDM`, optional): Finite difference method to use.
            Defaults to :data:`.multivariate.default_adaptive_method`.

    Returns:
        function: Jacobian of `f` multiplied by `v`, or directional
            derivative in the direction `v`.
    """

    def compute_jvp(x):
        dtype = np.array(f(x)).dtype  # Query the object once to get the dtype.
        zero = np.array(0).astype(dtype)
        return method(lambda eps: f(x + eps * v), zero)

    return compute_jvp


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
        size_in = np.size(x)  # Size of input.

        # Query the object once to get the dtype and output size.
        fx = np.array(f(x))
        dtype = fx.dtype
        size_out = fx.size

        # Construct the Jacobian.
        jac = np.empty((size_out, size_in), dtype=dtype)

        # Loop over outputs to fill the Jacobian.
        for i in range(size_out):
            grad = gradient(lambda y: _get_at_index(f(y), i), method)
            jac[i, :] = np.reshape(grad(x), size_in)

        return jac

    return compute_jacobian


def hvp(f,
        v,
        jac_method=default_adaptive_method,
        dir_method=default_robust_method):
    """Compute a Hessian-vector product.

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
    return jvp(jacobian(f, jac_method), v, dir_method)
