from numpy.testing import assert_allclose, assert_array_almost_equal

__all__ = ['allclose', 'approx']

allclose = assert_allclose


def approx(x, y, digits=7):
    return assert_array_almost_equal(x, y, decimal=digits)
