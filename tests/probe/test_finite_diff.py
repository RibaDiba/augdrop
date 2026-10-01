import numpy as np
import pytest

from probe.finite_diff import fd_grad, fd_points


def test_central_difference_is_exact_on_a_quadratic():
    """Central differences cancel the second-order term, so a quadratic's slope comes
    back exactly. A one-sided difference would be off by step * curvature."""
    a = np.array([1.0, -2.0, 0.5])
    m = np.array([0.3, 0.5, 0.7])
    got = fd_grad(lambda x: a * x**2, m, step=0.1)
    np.testing.assert_allclose(got, 2 * a * m, rtol=1e-10)


def test_each_image_gets_its_own_magnitude():
    m = np.array([0.2, 0.8])
    seen = []

    def evaluate(x):
        seen.append(x.copy())
        return x

    fd_grad(evaluate, m, step=0.05)
    np.testing.assert_allclose(seen[0], [0.25, 0.85])
    np.testing.assert_allclose(seen[1], [0.15, 0.75])


def test_one_sided_at_the_rails():
    """An op is undefined past its range (negative blur sigma), so an image within one
    step of a rail is differenced on the inside only."""
    lower, upper = fd_points(np.array([0.0, 0.02, 0.5, 1.0]), 0.05, 0.0, 1.0)
    np.testing.assert_allclose(lower, [0.0, 0.0, 0.45, 0.95])
    np.testing.assert_allclose(upper, [0.05, 0.07, 0.55, 1.0])
    got = fd_grad(lambda x: 3 * x, np.array([0.0, 1.0]), step=0.05)
    np.testing.assert_allclose(got, [3.0, 3.0])


def test_works_in_an_ops_own_units():
    """vocab ops carry their own ranges: degrees, pixels, enhance factors."""
    got = fd_grad(lambda x: x / 30.0, np.array([-30.0, 0.0, 30.0]), step=3.0, lo=-30.0, hi=30.0)
    np.testing.assert_allclose(got, [1 / 30.0] * 3)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="positive"):
        fd_points(np.array([0.5]), 0.0)
    with pytest.raises(ValueError, match="lo < hi"):
        fd_points(np.array([0.5]), 0.1, 1.0, 1.0)
    with pytest.raises(ValueError, match="outside"):
        fd_points(np.array([1.5]), 0.1)
