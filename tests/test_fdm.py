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
