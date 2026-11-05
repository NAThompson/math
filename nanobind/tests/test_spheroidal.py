# Copyright 2026 Nick Thompson
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt
import numpy as np
import pytest

import boost.math as bm

PROLATE = [bm.prolate_spheroidal_angular, bm.prolate_spheroidal_radial_1, bm.prolate_spheroidal_radial_2]
OBLATE = [bm.oblate_spheroidal_angular, bm.oblate_spheroidal_radial_1, bm.oblate_spheroidal_radial_2]
PRIME = {bm.prolate_spheroidal_angular: bm.prolate_spheroidal_angular_prime,
         bm.prolate_spheroidal_radial_1: bm.prolate_spheroidal_radial_1_prime,
         bm.prolate_spheroidal_radial_2: bm.prolate_spheroidal_radial_2_prime,
         bm.oblate_spheroidal_angular: bm.oblate_spheroidal_angular_prime,
         bm.oblate_spheroidal_radial_1: bm.oblate_spheroidal_radial_1_prime,
         bm.oblate_spheroidal_radial_2: bm.oblate_spheroidal_radial_2_prime}


def test_eigenvalues():
    # At c = 0 both are n(n + 1); pro_cv and obl_cv differ by ±c².
    assert bm.prolate_spheroidal_eigenvalue(3, 1, 0.0) == pytest.approx(12.0, rel=1e-15)
    assert bm.oblate_spheroidal_eigenvalue(3, 1, 0.0) == pytest.approx(12.0, rel=1e-15)
    assert bm.prolate_spheroidal_eigenvalue(2, 1, 1.0) == pytest.approx(5.4246991437751347200, rel=2e-16)
    with pytest.raises(ValueError):
        bm.prolate_spheroidal_eigenvalue(1, 2, 1.0)  # n < m


def test_angular_reduces_to_ferrers():
    eta = np.linspace(-0.9, 0.9, 7)
    for f in (bm.prolate_spheroidal_angular, bm.oblate_spheroidal_angular):
        np.testing.assert_allclose(f(4, 2, 0.0, eta), bm.legendre_p(4, 2, eta), rtol=1e-13, atol=1e-14)


def test_radial_against_arb():
    assert bm.prolate_spheroidal_radial_2(2, 1, 2.0, 1.5) == pytest.approx(-0.4498876213596331, rel=8e-16)


@pytest.mark.parametrize("oblate", [False, True])
def test_radial_wronskian(oblate):
    r1, r2 = (OBLATE if oblate else PROLATE)[1:]
    r1p, r2p = PRIME[r1], PRIME[r2]
    c, xi = 3.0, np.array([0.2, 1.3, 2.0, 7.0]) if oblate else np.array([1.01, 1.3, 2.0, 7.0])
    w = r1(2, 1, c, xi) * r2p(2, 1, c, xi) - r1p(2, 1, c, xi) * r2(2, 1, c, xi)
    np.testing.assert_allclose(w, 1 / (c * (xi * xi + 1 if oblate else xi * xi - 1)), rtol=1e-12)


@pytest.mark.parametrize("f", PROLATE + OBLATE, ids=lambda f: f.__name__)
def test_prime_matches_finite_difference(f):
    fp = PRIME[f]
    x = 0.3 if "angular" in f.__name__ else (1.7 if "prolate" in f.__name__ else 0.6)
    h = 1e-5
    fd = (f(3, 1, 2.5, x + h) - f(3, 1, 2.5, x - h)) / (2 * h)
    assert fp(3, 1, 2.5, x) == pytest.approx(fd, rel=1e-8)
    xs = np.array([x, x + 0.1])
    np.testing.assert_array_equal(fp(3, 1, 2.5, xs), [fp(3, 1, 2.5, v) for v in xs])


@pytest.mark.parametrize("oblate", [False, True])
def test_classes_match_functions(oblate):
    w = (bm.oblate_spheroidal if oblate else bm.prolate_spheroidal)(3, 1, 2.5)
    p = "oblate_spheroidal_" if oblate else "prolate_spheroidal_"
    assert w.eigenvalue == getattr(bm, p + "eigenvalue")(3, 1, 2.5)
    eta, xi = np.array([-0.4, 0.3]), np.array([0.6, 1.7]) if oblate else np.array([1.2, 1.7])
    for name, x in [("angular", eta), ("radial_1", xi), ("radial_2", xi)]:
        for suffix in ("", "_prime"):
            np.testing.assert_array_equal(getattr(w, name + suffix)(x), getattr(bm, p + name + suffix)(3, 1, 2.5, x))
    with pytest.raises(ValueError):
        bm.prolate_spheroidal(1, 2, 1.0)  # n < m
