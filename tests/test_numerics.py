# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import pytest

from fdm import approx_equal, check_sensitivity


def test_approx_equal():
    assert not approx_equal(1, 1 + 1e-5, 1e-10, 1e-6)
    assert approx_equal(1, 1 + 1e-7, 1e-10, 1e-6)
    assert not approx_equal(0, 1e-9, 1e-10, 1e-6)
    assert approx_equal(0, 1e-11, 1e-10, 1e-6)


def test_sensitivity():
    # Test two-argument case.
    def f(a, b):
        return a * b

    def s_f_correct(s_y, y, a, b):
        return s_y * b, a * s_y

    def s_f_incorrect1(s_y, y, a, b):
        return s_y * b + 1, a * s_y

    def s_f_incorrect2(s_y, y, a, b):
        return s_y * b, a * s_y * 1.1

    check_sensitivity(f, s_f_correct, (1, 2))
    with pytest.raises(AssertionError):
        check_sensitivity(f, s_f_incorrect1, (2, 3))
    with pytest.raises(AssertionError):
        check_sensitivity(f, s_f_incorrect2, (4, 5))

    # Test one-argument case with a keyword argument.
    def g(a, option=False):
        assert option
        return 2 * a

    def s_g(s_y, y, a, option=False):
        assert option
        return s_y * 2

    with pytest.raises(AssertionError):
        check_sensitivity(g, s_g, (1,))
    check_sensitivity(g, s_g, (1,), {'option': True})

    # Check that the the number of sensitivities must match the number of
    # arguments.
    def s_f_too_few(s_y, y, a, b):
        return s_y * a

    def s_g_too_many(s_y, y, a, option=True):
        return 2, 3

    with pytest.raises(AssertionError):
        check_sensitivity(f, s_f_too_few, (1, 2))
    with pytest.raises(AssertionError):
        check_sensitivity(g, s_g_too_many, (1,))
