# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import logging

__all__ = ['FDM',
           'forward_fdm',
           'backward_fdm',
           'central_fdm',
           'default_condition']
log = logging.getLogger(__name__)

default_condition = 100  #: Default condition number.


def _default_bound_estimator(f, x, condition=default_condition):
    if f is None:
        # No function is given. Just assume a constant approximation of 1.
        return condition
    return condition * np.max(np.abs(f(x)))


def _adaptive_bound_estimator(method, order, condition, adapt, **kw_args):
    if adapt >= 1:
        # Estimate the `order`th derivative of the function `f` at `x`.
        estimate_derivative = \
            method(order + 1, order, adapt - 1, condition, **kw_args)

        # Adaptive estimator:
        def estimator(f, x, condition_=condition):
            # If no function is supplied, fall back to the default estimator.
            if f is None:
                return _default_bound_estimator(f, x, condition_)
            else:
                return np.max(np.abs(estimate_derivative(f, x)))

    else:
        # Non-adaptive estimator:
        def estimator(f, x, condition_=condition):
            return _default_bound_estimator(f, x, condition_)

    return estimator


class FDM(object):
    """A finite difference method.

    Call the finite difference method with a function `f`, an input location
    `x` (optional: defaults to `0`), and a step size (optional: defaults to an
    estimate) to estimate the derivative of `f` at `x`. For example, if `m`
    is an instance of :class:`.fdm.FDM`, then `m(f, 1)` estimates the derivative
    of `f` at `1`.

    Args:
        grid (list): Relative spacing of samples of the function used by the
            method.
        deriv (int): Order of the derivative to estimate.
        bound_estimator (function, optional): Estimator that taken in a
            function `f` and a location `x` and gives back an estimate of an
            upper bound on the `len(grid)`th order derivative of `f` at `x`.
            Defaults to :data:`.fdm._default_bound_estimator`.
        factor (float, optional): Estimate of the relative error on function
            evaluations as a multiple of the machine epsilon. Defaults to `1`.
        step_max (float, optional): Maximum step size, to prevent very
            large step sizes in the case of high-order methods. Defaults to
            `0.1`.

    Attributes:
        grid (list): Relative spacing of samples of the function used by the
            method.
        order (int): Order of the estimator.
        deriv (int): Order of the derivative to estimate.
        eps (float): Estimated absolute round-off error of the function
            evaluation. This is used to estimate the step size.
        bound (float): Estimated upper bound of the absolute value of the
            function and all its derivatives. This is used to estimate the
            step size.
        coefs (list): Weighting of the samples used to estimate the derivative.
        step (float): Estimate of the step size. This estimate depends on
            `eps` and `bound`, and may be inadequate if `eps` and `bound` are
            inadequate.
        step_max (float, optional): Maximum step size, to prevent very
            large step sizes in the case of high-order methods.
        acc (float): Estimate of the accuracy of the method. This estimate
            depends on `eps` and `bound`, and may be inadequate if `eps` and
            `bound` are inadequate.
    """

    def __init__(self,
                 grid,
                 deriv,
                 bound_estimator=_default_bound_estimator,
                 factor=1,
                 step_max=1):
        self.grid = np.array(grid)
        self.order = self.grid.shape[0]
        self.deriv = deriv
        self.bound_estimator = bound_estimator
        self.factor = factor
        self.bound = None
        self.eps = None
        self.acc = None
        self.step = None
        self.step_max = step_max

        if self.order <= self.deriv:
            raise ValueError('Order of the method must be strictly greater '
                             'than that of the derivative.')

        # Compute coefficients.
        C = np.stack([self.grid ** i for i in range(self.order)], axis=0)
        x = np.zeros(self.order)
        x[self.deriv] = np.math.factorial(self.deriv)
        self.coefs = np.linalg.solve(C, x)

    def estimate(self, f=None, x=np.float64(0)):
        """Estimate step size and accuracy of the method.

        Args:
            f (function, optional): Function to estimate step size and accuracy
                for. Defaults to the constant function `1`.
            x (tensor, optional): Location to estimate derivative at. Defaults
                to `0`.

        Returns:
            :class:`.fdm.FDM`: Returns itself.
        """
        # Obtain a sample function value, defaulting to the constant function 1.
        f_value = f(x) if f else np.float64(1)

        # Get info and a tiny number to prevent divide-by-zero errors.
        finfo = np.finfo(np.array(f_value).dtype)
        tiny = 1e-20

        # Estimate the bound and epsilon.
        self.bound = self.bound_estimator(f, x) + tiny
        self.eps = finfo.eps * (np.max(np.abs(f_value)) + tiny) * self.factor

        # Estimate step size.
        c1 = self.eps * np.sum(np.abs(self.coefs))
        c2 = self.bound
        c2 *= np.sum(np.abs(self.coefs * self.grid ** self.order))
        c2 /= np.math.factorial(self.order)
        self.step = (self.deriv / (self.order - self.deriv) * c1 / c2) \
                    ** (1. / self.order)

        # Cap the step size.
        self.step = min(self.step, self.step_max)

        # Estimate accuracy.
        self.acc = c1 * self.step ** (-self.deriv) + \
                   c2 * self.step ** (self.order - self.deriv)

        return self

    def __call__(self, f, x=np.float64(0), step=None):
        if step is None:
            # Dynamically estimate the step size and accuracy given this bound.
            self.estimate(f, x)
        else:
            # Don't do anything dynamically.
            self.bound = None
            self.eps = None
            self.step = step
            self.acc = None

        # Execute finite difference estimate.
        ws = [c * f(x + self.step * loc)
              for c, loc in zip(self.coefs, self.grid)]
        return np.sum(ws, axis=0) / self.step ** self.deriv


def forward_fdm(order, deriv, adapt=1, condition=default_condition, **kw_args):
    """Construct a forward finite difference method.

    Further takes in keyword arguments of the constructor of :class:`.fdm.FDM`.

    Args:
        order (int): Order of the method.
        deriv (int): Order of the derivative to estimate.
        adapt (int, optional): Number of recursive calls to higher-order
            derivatives to dynamically determine the step size. Defaults to `1`.
        condition (float, optional): Amplification of the infinity norm when
            passed to the function's derivatives. Defaults to
            :data:`.fdm.default_condition`.

    Returns:
        :class:`.fdm.FDM`: The desired finite difference method.
    """
    return FDM(np.arange(order),
               deriv,
               bound_estimator=_adaptive_bound_estimator(
                   forward_fdm, order, condition, adapt, **kw_args
               ),
               **kw_args)


def backward_fdm(order, deriv, adapt=1, condition=default_condition, **kw_args):
    """Construct a backward finite difference method.

    Further takes in keyword arguments of the constructor of :class:`.fdm.FDM`.

    Args:
        order (int): Order of the method.
        deriv (int): Order of the derivative to estimate.
        adapt (int, optional): Number of recursive calls to higher-order
            derivatives to dynamically determine the step size. Defaults to `1`.
        condition (float, optional): Amplification of the infinity norm when
            passed to the function's derivatives. Defaults to
            :data:`.fdm.default_condition`.

    Returns:
        :class:`.fdm.FDM`: The desired finite difference method.
    """
    return FDM(-np.arange(order)[::-1],
               deriv,
               bound_estimator=_adaptive_bound_estimator(
                   backward_fdm, order, condition, adapt, **kw_args
               ),
               **kw_args)


def central_fdm(order, deriv, adapt=1, condition=default_condition, **kw_args):
    """Construct a central finite difference method.

    Further takes in keyword arguments of the constructor of :class:`.fdm.FDM`.

    Args:
        order (int): Order of the method.
        deriv (int): Order of the derivative to estimate.
        adapt (int, optional): Number of recursive calls to higher-order
            derivatives to dynamically determine the step size. Defaults to `1`.
        condition (float, optional): Amplification of the infinity norm when
            passed to the function's derivatives. Defaults to
            :data:`.fdm.default_condition`.

    Returns:
        :class:`.fdm.FDM`: The desired finite difference method.
    """
    return FDM(np.linspace(-1, 1, order),
               deriv,
               bound_estimator=_adaptive_bound_estimator(
                   central_fdm, order, condition, adapt, **kw_args
               ),
               **kw_args)
