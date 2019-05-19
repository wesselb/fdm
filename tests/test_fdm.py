# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_allclose as close

from fdm import forward_fdm, backward_fdm, central_fdm, FDM
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, ok


def test_construction():
    yield raises, ValueError, lambda: FDM([-1, 0, 1], 3)


def test_correctness():
    for f in [forward_fdm, backward_fdm, central_fdm]:
        yield close, f(10, 1)(np.sin, 1), np.cos(1)
        yield close, f(10, 2)(np.sin, 1), -np.sin(1)

        yield close, f(10, 1)(np.exp, 1), np.exp(1)
        yield close, f(10, 2)(np.exp, 1), np.exp(1)

        yield close, f(10, 1)(np.sqrt, 1), .5
        yield close, f(10, 2)(np.sqrt, 1), -.25


def test_estimation():
    m = central_fdm(2, 1)

    yield eq, m.eps, None
    yield eq, m.bound, None
    yield eq, m.step, None
    yield eq, m.acc, None

    m.estimate()

    yield isinstance, m.eps, float
    yield isinstance, m.bound, float
    yield isinstance, m.step, float
    yield isinstance, m.acc, float

    m(np.sin, 0, step=1e-3)

    yield eq, m.eps, None
    yield eq, m.bound, None
    yield isinstance, m.step, float
    yield eq, m.acc, None


def test_adaptation():
    def f(x):
        return 1 / x

    def df(x):
        return -1 / x ** 2

    err1 = np.abs(forward_fdm(3, 1, adapt=0)(f, 1e-3) - df(1e-3))
    err2 = np.abs(forward_fdm(3, 1, adapt=1)(f, 1e-3) - df(1e-3))

    # Check that adaptation helped.
    yield le, err2, 1e-2 * err1

    # Check that adaptation gets it right.
    yield le, err2, 1e-4


def test_order_monotonicity():
    err_ref = 1e-4

    for i in range(3, 8):
        err = np.abs(central_fdm(i, 2, condition=1)(np.sin, 1) + np.sin(1))

        # Check that it did better than the previous estimator.
        yield le, err, err_ref

        err_ref = err


def test_tiny():
    # Check that `tiny` added in :meth:`.fdm.FDM.estimate` stabilises the
    # numerics.
    yield eq, central_fdm(2, 1, adapt=0)(lambda x: 0.0, 1.0), 0.0
    yield eq, central_fdm(2, 1, adapt=1)(lambda x: x, 1.0), 1.0
