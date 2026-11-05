# Copyright 2026 Nick Thompson
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt
import mpmath as mp
import numpy as np
import pytest

import boost.math as bm

mp.mp.dps = 30
POINTS = [(0.0, 0.5), (0.0, 7.0), (1.0, 3.0), (2.5, 12.0), (-0.5, 1.5), (1.0 / 3, 40.0)]
PAIRS = [(bm.struve_h, bm.struve_h_prime, mp.struveh), (bm.struve_l, bm.struve_l_prime, mp.struvel)]


@pytest.mark.parametrize("f, fp, ref", PAIRS, ids=["H", "L"])
def test_against_mpmath(f, fp, ref):
    for v, x in POINTS:
        assert f(v, x) == pytest.approx(float(ref(v, x)), rel=1e-13), (v, x)
        assert fp(v, x) == pytest.approx(float(mp.diff(lambda t: ref(v, t), x)), rel=1e-12), (v, x)
    v, x = np.array([[0.0], [1.5]]), np.array([0.5, 2.0, 9.0])
    np.testing.assert_array_equal(fp(v, x), [[fp(a, b) for b in x[:]] for a in v[:, 0]])


@pytest.mark.parametrize("f, fp, ref", PAIRS, ids=["H", "L"])
def test_negative_x_integer_order(f, fp, ref):
    for n in (0, 1, 2):
        assert f(n, -2.5) == pytest.approx(float(ref(n, -2.5)), rel=1e-13)
        assert f(n, -2.5) == pytest.approx((-1) ** (n + 1) * f(n, 2.5), rel=1e-15)
        assert fp(n, -2.5) == pytest.approx((-1) ** n * fp(n, 2.5), rel=1e-15)
    with pytest.raises(ValueError):
        f(0.5, -1.0)
