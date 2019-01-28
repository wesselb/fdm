# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import logging

__all__ = ['FDM', 'forward_fdm', 'backward_fdm', 'central_fdm']
log = logging.getLogger(__name__)


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
        condition (float, optional): Amplification of the infinity norm when
            passed to the function's derivatives. Defaults to `100`.

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
        acc (float): Estimate of the accuracy of the method. This estimate
            depends on `eps` and `bound`, and may be inadequate if `eps` and
            `bound` are inadequate.
    """

    def __init__(self, grid, deriv, condition=100):
        self.grid = np.array(grid)
        self.order = self.grid.shape[0]
        self.deriv = deriv
        self.condition = condition
        self.bound = None
        self.eps = None
        self.acc = None
        self.step = None

        if self.order <= self.deriv:
            raise ValueError('Order of the method must be strictly greater '
                             'than that of the derivative.')

        # Compute coefficients.
        C = np.stack([self.grid ** i for i in range(self.order)], axis=0)
        x = np.zeros(self.order)
        x[self.deriv] = np.math.factorial(self.deriv)
        self.coefs = np.linalg.solve(C, x)

    def estimate(self, f_value=np.float64(1)):
        """Estimate step size and accuracy of the method.

        Args:
            f_value (float, optional): Function evaluation. Defaults to
                `np.float64(1)`.

        Returns:
            :class:`.fdm.FDM`: Returns itself.
        """
        # Estimate bound and epsilon.
        f_max = np.max(np.abs(f_value))
        self.bound = self.condition * f_max
        self.eps = np.finfo(f_value.dtype).eps * f_max

        # Estimate step size.
        c1 = self.eps * np.sum(np.abs(self.coefs))
        c2 = self.bound
        c2 *= np.sum(np.abs(self.coefs * self.grid ** self.order))
        c2 /= np.math.factorial(self.order)
        self.step = (self.deriv / (self.order - self.deriv) * c1 / c2) \
                    ** (1. / self.order)

        # Estimate accuracy.
        self.acc = c1 * self.step ** (-self.deriv) + \
                   c2 * self.step ** (self.order - self.deriv)

        return self

    def __call__(self, f, x=0, step=None):
        if step is None:
            self.estimate(f(x))
            step = self.step
        else:
            self.bound = None
            self.eps = None
            self.step = step
            self.acc = None
        ws = [c * f(x + step * loc) for c, loc in zip(self.coefs, self.grid)]
        return np.sum(ws) / self.step ** self.deriv


def forward_fdm(order, deriv, **kw_args):
    """Construct a forward finite difference method.

    Further takes in keyword arguments of the constructor of :class:`.fdm.FDM`.

    Args:
        order (int): Order of the method.
        deriv (int): Order of the derivative to estimate.

    Returns:
        :class:`.fdm.FDM`: The desired finite difference method.
    """
    return FDM(np.arange(order), deriv, **kw_args)


def backward_fdm(order, deriv, **kw_args):
    """Construct a backward finite difference method.

    Further takes in keyword arguments of the constructor of :class:`.fdm.FDM`.

    Args:
        order (int): Order of the method.
        deriv (int): Order of the derivative to estimate.

    Returns:
        :class:`.fdm.FDM`: The desired finite difference method.
    """
    return FDM(-np.arange(order)[::-1], deriv, **kw_args)


def central_fdm(order, deriv, **kw_args):
    """Construct a central finite difference method.

    Further takes in keyword arguments of the constructor of :class:`.fdm.FDM`.

    Args:
        order (int): Order of the method.
        deriv (int): Order of the derivative to estimate.

    Returns:
        :class:`.fdm.FDM`: The desired finite difference method.
    """
    return FDM(np.linspace(-1, 1, order), deriv, **kw_args)
