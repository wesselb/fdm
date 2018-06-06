# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import logging

__all__ = ['approx_equal']
log = logging.getLogger(__name__)


def approx_equal(x, y,
                 eps_abs=10 ** 2 * np.finfo(float).eps,
                 eps_rel=np.sqrt(np.finfo(float).eps)):
    """Check whether `x` and `y` are approximately equal.

    Let `eps_z = eps_abs / eps_rel`. Call `x` and `y` small if
    `abs(x) + abs(y) < eps_z`, and call `x` and `y` large otherwise. If this
    function returns `True`, then it is guaranteed that
    `abs(x - y) < 2eps_rel * (abs(x) + abs(y))` if `x` and `y` are large, and
    `abs(x - y) < 2eps_abs` if `x` and `y` are small.

    Args:
        x (object): First object to compare.
        y (object): Second object to compare.
        eps_abs (float, optional): Absolute tolerance.
        eps_rel (float, optional): Relative tolerance.
    """
    return np.all(np.abs(x - y) <= eps_abs + eps_rel * (np.abs(x) + np.abs(y)))
