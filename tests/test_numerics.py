# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from fdm import approx_equal

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam


def test_approx_equal():
    yield ok, not approx_equal(1, 1 + 1e-5, 1e-10, 1e-6)
    yield ok, approx_equal(1, 1 + 1e-7, 1e-10, 1e-6)
    yield ok, not approx_equal(0, 1e-9, 1e-10, 1e-6)
    yield ok, approx_equal(0, 1e-11, 1e-10, 1e-6)
