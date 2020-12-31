import logging

import sympy as sp
import numpy as np

__all__ = ["FDM", "forward_fdm", "backward_fdm", "central_fdm"]

log = logging.getLogger(__name__)


def _get_dtype(x):
    return np.array(x, copy=False).dtype


def _ensure_float(x):
    if np.issubdtype(_get_dtype(x), np.floating):
        return x
    else:
        return float(x)


def _eps(x):
    return np.finfo(_get_dtype(_ensure_float(x))).eps


_cache = {}


def _compute_coefs_mults(grid, deriv):
    order = len(grid)
    cache_key = (tuple(grid), deriv)

    try:
        return _cache[cache_key]
    except KeyError:
        # Compute coefficients.
        mat = sp.Matrix([[g ** i for g in grid] for i in range(order)])
        coefs = mat.inv()[:, deriv] * np.math.factorial(deriv)

        # Compute parts of the FDM.
        coefs = np.array([float(c) for c in coefs])
        bound_mult = float(
            sum([abs(c * g ** order) for c, g in zip(coefs, grid)])
            / np.math.factorial(order)
        )
        err_mult = float(sum([abs(c) for c in coefs]))

        # Save result.
        _cache[cache_key] = coefs, bound_mult, err_mult

        return coefs, bound_mult, err_mult


class FDM:
    """A finite difference method.

    Call the finite difference method with a function `f`, an input location
    `x` (optional: defaults to `0`), and a step size (optional: defaults to an
    estimate) to estimate the derivative of `f` at `x`. For example, if `m`
    is an instance of :class:`.fdm.FDM`, then `m(f, 1)` estimates the derivative
    of `f` at `1`.

    Args:
        grid (list[int] or :class:`np.ndarray`): Relative spacing of samples of the
            function used by the method.
        deriv (int, optional): Order of the derivative to estimate. Defaults to `1`.
        bound_estimator (:class:`.fdm.FDM`, optional): A finite difference method that
            is tuned to perform adaptation for this method.
        condition (float, optional): Amplification of the infinity norm when passed
            to the function's derivatives. Defaults to `100.0`.
        factor (float, optional): Amplification of the round-off error on the function's
            evaluations. Defaults to `1.0`.

    Attributes:
        grid (:class:`np.ndarray`): Relative spacing of samples of the function used by
            the method.
        order (int): Order of the estimator.
        deriv (int): Order of the derivative to estimate.
        bound_estimator (function or `None`): Estimator that taken in a function `f`
            and a location `x` and gives back an estimate of an upper bound on the
            `len(grid)`th order derivative of `f` at `x`.
        condition (float): Amplification of the infinity norm when passed to the
            function's derivatives.
        factor (float): Amplification of the round-off error on the function's
            evaluations.
        step (float or `None`): Estimate of the step size.
        acc (float or `None`): Estimate of the accuracy of the method.
        coefs (:class:`np.ndarray`): Weighting of the samples used to estimate the
            derivative.
        df_magnitude_mult (float): Multiplier associated with the truncation error.
        f_error_mult (float): Multiplier associated with the round-off error.
    """

    def __init__(
        self,
        grid,
        deriv=1,
        bound_estimator=None,
        condition=100.0,
        factor=1.0,
    ):
        self.grid = np.array(grid)
        self.order = len(self.grid)
        self.deriv = deriv
        self.bound_estimator = bound_estimator
        self.condition = condition
        self.factor = factor

        self.step = None
        self.acc = None

        if self.order <= self.deriv:
            raise ValueError(
                "Order of the method must be strictly greater "
                "than that of the derivative."
            )

        self.coefs, self.df_magnitude_mult, self.f_error_mult = _compute_coefs_mults(
            self.grid, self.deriv
        )

    def estimate(self, f=None, x=0, max_range=None):
        """Estimate step size and accuracy of the method.

        Args:
            f (function, optional): Function to estimate step size and accuracy for.
                Defaults to the constant function `1`.
            x (tensor, optional): Location to estimate derivative at. Defaults to `0`.
            max_range (float, optional): Maximum amount to deviate from `x`.

        Returns:
            :class:`.fdm.FDM`: Returns itself.
        """
        x = _ensure_float(x)

        if f and self.bound_estimator:
            # Estimate bound.
            df_magnitude, f_magnitude = self.bound_estimator(
                f, x, magnitude=True, max_range=max_range
            )

            # Estimate error.
            f_error = f_magnitude * _eps(f_magnitude)

            # Adaptation requires that `f_error > 0` and `df_magnitude > 0`. If those
            # conditions are not satisfied, default to the default step estimator.
            if f_error == 0 or df_magnitude == 0:
                self._estimate_default(x, max_range)
            else:
                self._estimate_adapt(x, f_error, df_magnitude, max_range)
        else:
            self._estimate_default(x, max_range)

        return self

    def _compute_step_acc(self, f_error, df_magnitude):
        c1 = f_error * self.f_error_mult * self.factor
        c2 = df_magnitude * self.df_magnitude_mult
        step = self.deriv / (self.order - self.deriv) * (c1 / c2)
        step = step ** (1.0 / self.order)
        acc = c1 * step ** (-self.deriv) + c2 * step ** (self.order - self.deriv)
        return step, acc

    def _compute_default(self, x):
        step, acc = self._compute_step_acc(_eps(x), self.condition)
        return step, acc

    def _estimate_default(self, x, max_range):
        self.step, self.acc = self._compute_default(x)
        self._limit_step(x, max_range)

    def _estimate_adapt(self, x, f_error, df_magnitude, max_range):
        self.step, self.acc = self._compute_step_acc(f_error, df_magnitude)
        self._limit_step(x, max_range)

    def _limit_step(self, x, max_range):
        if max_range:
            step_max = max_range / np.max(np.abs(self.grid))
            if self.step > step_max:
                self.step = step_max
                self.acc = None
        else:
            step_default, _ = self._compute_default(x)
            step_max_default = 1000 * step_default
            if self.step > step_max_default:
                self.step = step_max_default
                self.acc = None

    def __call__(self, f, x=0, step=None, magnitude=False, max_range=None):
        """Execute the finite difference method.

        Args:
            f (function, optional): Function to estimate step size and accuracy for.
                Defaults to the constant function `1`.
            x (tensor, optional): Location to estimate derivative at. Defaults to `0`.
            step (float, optional): Step size. If not given, the step size is
                automatically estimated.
            magnitude (bool, optional): Estimate the magnitude of the derivative of
                `f` and `f` in a neighbourhood of `x`.
            max_range (float, optional): Maximum amount to deviate from `x`.

        Returns:
            float or tuple[float, float]: Estimate of the derivative if
                `magnitude = False` and estimates of the magnitudes of the derivative
                of `f` and `f` if `magnitude = True`.
        """
        x = _ensure_float(x)

        if step is None:
            # Dynamically estimate the step size and accuracy.
            self.estimate(f, x, max_range=max_range)
        else:
            # Don't do anything dynamically.
            self.step = step
            self.acc = None

        # Perform setup for finite difference estimate..
        fs = [f(x + g * self.step) for g in self.grid]

        def _execute(coefs):
            ws = [c * f for c, f in zip(coefs, fs)]
            return np.sum(ws, axis=0) / self.step ** self.deriv

        if magnitude:
            # Estimate magnitude of function in neighbourhood.
            magnitude_f = np.max(np.abs(fs))

            # Estimate magnitude of derivative in neighbourhood.
            estimates = []
            for offset in -self.grid:
                coefs, _, _ = _compute_coefs_mults(self.grid + offset, self.deriv)
                estimates.append(np.abs(_execute(coefs)))
            magnitude_df = np.max(estimates)

            # Return magnitudes instead of an estimate of the derivative.
            return magnitude_df, magnitude_f
        else:
            # Just estimate the derivative.
            return _execute(self.coefs)


def _construct_bound_estimator(method, order, adapt, **kw_args):
    if adapt >= 1:
        return method(order + 2, order, adapt - 1, **kw_args)
    else:
        return None


def forward_fdm(order, deriv, adapt=1, **kw_args):
    """Construct a forward finite difference method.

    Further takes in keyword arguments of the constructor of :class:`.fdm.FDM`.

    Args:
        order (int): Order of the method.
        deriv (int): Order of the derivative to estimate.
        adapt (int, optional): Number of recursive calls to higher-order
            derivatives to dynamically determine the step size. Defaults to `1`.

    Returns:
        :class:`.fdm.FDM`: The desired finite difference method.
    """
    return FDM(
        list(range(order)),
        deriv,
        bound_estimator=_construct_bound_estimator(
            forward_fdm, order, adapt, **kw_args
        ),
        **kw_args
    )


def backward_fdm(order, deriv, adapt=1, **kw_args):
    """Construct a backward finite difference method.

    Further takes in keyword arguments of the constructor of :class:`.fdm.FDM`.

    Args:
        order (int): Order of the method.
        deriv (int): Order of the derivative to estimate.
        adapt (int, optional): Number of recursive calls to higher-order derivatives
            to dynamically determine the step size. Defaults to `1`.

    Returns:
        :class:`.fdm.FDM`: The desired finite difference method.
    """
    return FDM(
        [-g for g in reversed(range(order))],
        deriv,
        bound_estimator=_construct_bound_estimator(
            backward_fdm, order, adapt, **kw_args
        ),
        **kw_args
    )


def central_fdm(order, deriv, adapt=1, **kw_args):
    """Construct a central finite difference method.

    Further takes in keyword arguments of the constructor of :class:`.fdm.FDM`.

    Args:
        order (int): Order of the method.
        deriv (int): Order of the derivative to estimate.
        adapt (int, optional): Number of recursive calls to higher-order
            derivatives to dynamically determine the step size. Defaults to `1`.

    Returns:
        :class:`.fdm.FDM`: The desired finite difference method.
    """
    if order % 2 == 0:
        half = int(order / 2)
        grid = list(range(-half, half + 1))
        grid.pop(half)
    else:
        half = int((order - 1) / 2)
        grid = list(range(-half, half + 1))
    return FDM(
        grid,
        deriv,
        bound_estimator=_construct_bound_estimator(
            central_fdm, order, adapt, **kw_args
        ),
        **kw_args
    )
