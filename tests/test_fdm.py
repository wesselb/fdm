# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_allclose as close
from fdm import forward_fdm, backward_fdm, central_fdm, FDM

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam


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


