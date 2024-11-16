import logging

import numpy as np

from .multivariate import default_adaptive_method, jvp

__all__ = ["approx_equal", "check_sensitivity"]

log = logging.getLogger(__name__)

_float_eps = np.finfo(float).eps
_float_eps_sqrt = np.sqrt(_float_eps)


def approx_equal(x, y, eps_abs=1e2 * _float_eps, eps_rel=_float_eps_sqrt):
    """Check whether `x` and `y` are approximately equal.

    Let `eps_z = eps_abs / eps_rel`. Call `x` and `y` small if
    `abs(x) < eps_z` and `abs(y) < eps_z`, and call `x` and `y` large
    otherwise.  If this function returns `True`, then it is guaranteed that
    `abs(x - y) < 2 eps_rel max(abs(x), abs(y))` if `x` and `y` are large,
    and `abs(x - y) < 2 eps_abs` if `x` and `y` are small.

    Args:
        x (object): First object to compare.
        y (object): Second object to compare.
        eps_abs (float, optional): Absolute tolerance.
        eps_rel (float, optional): Relative tolerance.

    Returns:
        bool: Approximate equality of `x` and `y`.
    """
    return np.all(np.abs(x - y) <= eps_abs + eps_rel * np.maximum(np.abs(x), np.abs(y)))


def _make_f_i(f, i, args, kw_args):
    def f_i(x):
        return f(*(args[:i] + (x,) + args[i + 1 :]), **kw_args)

    return f_i


def check_sensitivity(
    f,
    s_f,
    args,
    kw_args=None,
    eps_abs=1e4 * _float_eps,
    eps_rel=1e1 * _float_eps_sqrt,
    method=default_adaptive_method,
):
    """Check the sensitivity of a function.

    Args:
        f (function): Function to check.
        s_f (function): Sensitivity of `f`. Should have signature
            `s_f(s_y, y, *args, **kw_args)`, where `s_y` is the sensitivity of
            the output, `y = f(*args, **kw_args)`, and `*args` and `**kw_args`
            the arguments and keyword arguments that `f` was evaluated at.
        args (tuple): Arguments to test `f` at.
        kw_args (dict, optional): Keyword arguments to test `f` at. Defaults to
            `{}`.
        eps_abs (float, optional): Absolute tolerance.
        eps_rel (float, optional): Relative tolerance.
        method (:class:`.fdm.FDM`, optional): Finite difference method to use.
            Defaults to :data:`.multivariate.default_adaptive_method`.

    """
    # Set default `kw_args`.
    if kw_args is None:
        kw_args = {}

    # Evaluate the function.
    y = f(*args, **kw_args)

    # Create a random sensitivity for the output.
    y_sens = np.random.randn(*np.array(y).shape)

    # Evaluate the sensitivity of the function.
    s_args = s_f(y_sens, y, *args, **kw_args)

    # Ensure that `s_args` is a tuple.
    if not isinstance(s_args, tuple):
        s_args = (s_args,)

    # Check that the correct number of sensitivities is given.
    if not len(args) == len(s_args):
        raise AssertionError(
            f"Number of sensitivities ({len(args)}) does not "
            f"match the number of arguments ({len(s_args)})."
        )

    # Walk through the arguments.
    for i in range(len(args)):
        # Create a function that only varies the `i`th argument.
        f_i = _make_f_i(f, i, args, kw_args)

        # Pick a random direction.
        v = np.random.randn(*np.array(args[i]).shape)

        # Compute the directional directive w.r.t. the inner product of the
        # output and `y_sens` numerically and exactly.
        estimate = np.sum(jvp(f_i, v, method=method)(args[i]) * y_sens)
        exact = np.sum(v * s_args[i])

        # Assert that the results match.
        if not approx_equal(estimate, exact, eps_abs=eps_abs, eps_rel=eps_rel):
            raise AssertionError(
                f"Sensitivity of argument {i + 1} of "
                f'function "{f.__name__}" did not match '
                f"numerical estimate. The error is "
                f"{np.abs(exact - estimate):.3e}."
            )
