import numpy as np
import pytest

from fdm import forward_fdm, backward_fdm, central_fdm, FDM
from .util import approx


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
        return -1 / x ** 2

    err1 = np.abs(forward_fdm(3, 1, adapt=0)(f, 1e-3) - df(1e-3))
    err2 = np.abs(forward_fdm(3, 1, adapt=1)(f, 1e-3) - df(1e-3))

    # Check that adaptation helped.
    assert err2 <= 1e-2 * err1

    # Check that adaptation gets it right.
    assert err2 <= 1e-4


def test_order_monotonicity():
    err_ref = 1e-4

    for i in range(3, 8):
        method = central_fdm(i, 2, condition=1, adapt=1)

        # Check order of method.
        assert method.order == i

        err = np.abs(method(np.sin, 1) + np.sin(1))

        # Check that it did better than the previous estimator.
        assert err <= err_ref

        err_ref = err


def test_default_step():
    approx(
        central_fdm(2, 1).estimate().step, np.sqrt(2 * np.finfo(np.float64).eps / 100)
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
    f = lambda _: 0
    x = 0

    assert m.bound_estimator(f, x) == 0
    assert m.bound_estimator(f, x, magnitude=True)[0] == 0
    assert m.bound_estimator(f, x, magnitude=True)[1] == 0

    approx(m(f, x), 0)


def test_step_limiting():
    m = central_fdm(2, 1)
    f = lambda x: np.exp(1e-3 * x)
    x = 0

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


def cosc(x):
    if x == 0:
        return 0.0
    else:
        return np.cos(np.pi * x) / x - np.sin(np.pi * x) / (np.pi * x ** 2)


@pytest.mark.parametrize(
    "f,x,df,factor",
    [
        (np.sin, 0, 1, 1),
        (np.cos, 0, 0, 1),
        (np.exp, 0, 1, 1),
        (np.sinc, 0, 0, 1),
        (cosc, 0, -np.pi ** 2 / 3, 200),
    ],
)
@pytest.mark.parametrize(
    "constructor",
    [
        central_fdm,
        forward_fdm,
        backward_fdm,
    ],
)
def test_accuracy(f, x, df, factor, constructor):
    approx(constructor(4, 1)(f, x), df, atol=1e-8 * factor, rtol=0)
    approx(constructor(5, 1)(f, x), df, atol=1e-8 * factor, rtol=0)
    approx(constructor(8, 1)(f, x), df, atol=1e-12 * factor, rtol=0)
    approx(constructor(9, 1)(f, x), df, atol=1e-12 * factor, rtol=0)
