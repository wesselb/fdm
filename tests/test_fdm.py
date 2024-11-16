import numpy as np
import pytest

from .util import approx
from fdm import FDM, backward_fdm, central_fdm, forward_fdm


def test_construction():
    with pytest.raises(ValueError):
        FDM([-1, 0, 1], 3)


@pytest.mark.parametrize("f", [forward_fdm, backward_fdm, central_fdm])
def test_correctness(f):
    approx(f(10, 1)(np.sin, 1), np.cos(1))
    approx(f(10, 2)(np.sin, 1), -np.sin(1))

    approx(f(10, 1)(np.exp, 1), np.exp(1))
    approx(f(10, 2)(np.exp, 1), np.exp(1))

    approx(f(10, 1)(np.sqrt, 1), 0.5)
    approx(f(10, 2)(np.sqrt, 1), -1 / 4)


def cosc(x):
    if x == 0:
        return 0.0
    else:
        # Does this give cancellation errors?
        return (np.cos(np.pi * x) - np.sinc(x)) / x


@pytest.mark.parametrize(
    "f,x,df,atol,atol_central",
    [
        (lambda x: 0, 0, 0, 0, 0),
        (lambda x: x, 0, 1, 5e-15, 5e-16),
        (np.exp, 0, 1, 7.5e-13, 5e-14),
        (np.sin, 0, 1, 1e-14, 5e-16),
        (np.cos, 0, 0, 5e-13, 5e-14),
        (np.exp, 1, np.exp(1), 5e-12, 5e-13),
        (np.sin, 1, np.cos(1), 5e-13, 5e-14),
        (np.cos, 1, -np.sin(1), 7.5e-13, 5e-14),
        (np.sinc, 0, 0, 5e-12, 5e-14),
        # `cosc` is hard.
        (cosc, 0, -(np.pi**2) / 3, 5e-10, 5e-11),
    ],
)
@pytest.mark.parametrize(
    "constructor,is_central",
    [
        (central_fdm, True),
        (forward_fdm, False),
        (backward_fdm, False),
    ],
)
def test_accuracy(f, x, df, atol, atol_central, constructor, is_central):
    if is_central:
        approx(constructor(6, 1)(f, x), df, atol=atol_central, rtol=0)
        approx(constructor(7, 1)(f, x), df, atol=atol_central, rtol=0)
        if f == cosc:
            # `cosc` is hard.
            approx(constructor(14, 1)(f, x), df, atol=1e-13, rtol=0)
        else:
            approx(constructor(14, 1)(f, x), df, atol=5e-15, rtol=0)
    else:
        approx(constructor(6, 1)(f, x), df, atol=atol, rtol=0)
        approx(constructor(7, 1)(f, x), df, atol=atol, rtol=0)


def test_estimation():
    m = central_fdm(2, 1)

    assert isinstance(m.coefs, np.ndarray)
    assert isinstance(m.df_magnitude_mult, float)
    assert isinstance(m.f_error_mult, float)
    assert m.step is None
    assert m.acc is None

    m.estimate()

    assert isinstance(m.step, float)
    assert isinstance(m.acc, float)

    m(np.sin, 0, step=1e-3)

    assert isinstance(m.step, float)
    assert m.acc is None


def test_adaptation():
    def f(x):
        return 1 / x

    def df(x):
        return -1 / x**2

    err1 = np.abs(forward_fdm(3, 1, adapt=0)(f, 1e-3) - df(1e-3))
    err2 = np.abs(forward_fdm(3, 1, adapt=2)(f, 1e-3) - df(1e-3))

    # Check that adaptation helped.
    assert err2 <= 0.01 * err1

    # Check that adaptation gets it right.
    assert err2 <= 1e-4


def test_order_monotonicity():
    err_ref = 1e-4

    for i in range(3, 10):
        method = central_fdm(i, 2, adapt=1)

        # Check order of method.
        assert method.order == i

        err = np.abs(method(np.sin, 1) + np.sin(1))

        # Check that it did better than the previous estimator.
        assert err <= err_ref

        err_ref = err


def test_default_step():
    approx(
        central_fdm(2, 1).estimate().step, np.sqrt(2 * np.finfo(np.float64).eps / 10)
    )


def test_zero_bound_fixed():
    m = central_fdm(2, 1)
    f = np.sin
    x = 0

    assert m.bound_estimator(f, x) == 0
    assert m.bound_estimator(f, x, magnitude=True)[0] > 0
    assert m.bound_estimator(f, x, magnitude=True)[1] > 0

    approx(m(f, x), 1)


def test_zero_bound_zero_error_not_fixed():
    m = central_fdm(2, 1)
    x = 0

    def f(_):
        return 0

    assert m.bound_estimator(f, x) == 0
    assert m.bound_estimator(f, x, magnitude=True)[0] == 0
    assert m.bound_estimator(f, x, magnitude=True)[1] == 0

    approx(m(f, x), 0)


def test_step_limiting():
    m = central_fdm(2, 1)
    x = 0

    def f(x):
        return np.exp(1e-3 * x)

    step_max = m.estimate().step * 1000
    assert m.estimate(f, x).step == step_max


def test_factor():
    assert (
        central_fdm(2, 1, factor=4).estimate().step
        == 2 * central_fdm(2, 1, factor=1).estimate().step
    )


def test_condition():
    assert (
        central_fdm(2, 1, condition=1).estimate().step
        == 2 * central_fdm(2, 1, condition=4).estimate().step
    )


class _TrackEvals:
    def __init__(self, f):
        self.evals_x = []
        self.f = f

    def __call__(self, x):
        self.evals_x.append(x)
        return self.f(x)

    def true_range(self, x0):
        return np.max(np.abs([x - x0 for x in self.evals_x]))


@pytest.mark.parametrize("factor", [10, 100, 1000])
def test_range_max(factor):
    f = _TrackEvals(np.sin)

    # Determine the range.
    central_fdm(8, 1, adapt=2)(f, 1)
    true_range = f.true_range(1)

    # Limit the range.
    f.evals_x.clear()
    central_fdm(8, 1, adapt=2)(f, 1, max_range=true_range / factor)
    approx(f.true_range(1), true_range / factor)
