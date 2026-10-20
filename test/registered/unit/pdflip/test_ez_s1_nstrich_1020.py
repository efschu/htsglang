"""EZ-S1 Nach-Review 2, Nachtrag N2' (nf-review2-ez-s1-1019.md; SOLL R3): ``_x_min_over_depths`` is the EXACT minimum of
X(depth) also over FOLDS (X jumps down at a depth that is no kink) and over flip-price crossings inside a raster interval.
H4 (RN2d): the bisection keeps BOTH sides of a crossing, not only ``a``.
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from types import SimpleNamespace  # noqa: E402

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import x_curves as xc  # noqa: E402
import test_pdflip_x_curves_1006 as T  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _setup(curves, cap):
    return SimpleNamespace(tp_prefill_max_tokens=cap, _x_setup=SimpleNamespace(curves=curves, beyond=xc.BEYOND_CLAMP))


def test_n2s_fold_fixture_review_case_498():
    """Seed 4242 set 14 (cap 2048, rows (0, 2852), chunks P256/D512): the fold at depth 822 -> 823 and the true minimum 498
    at depth 840 lie inside one raster interval whose ends both stay above the flip price. The old table said 567 / 512."""
    cv = xc.load(os.path.join(HERE, "ez_s1_rand14_fold.xcurves.json"))
    f = _setup(cv, 2048)
    assert F.Front._x_at_depth(f, 822) == 583 and F.Front._x_at_depth(f, 823) == 511      # the fold itself
    assert min(F.Front._x_at_depth(f, d) for d in range(0, 2853)) == 498
    assert F.Front._x_min_over_depths(f, 950) == 498
    assert F.Front._x_min_over_depths(f, 2852) == 498


def _random_set(rnd):
    depths = tuple(sorted({0} | {rnd.randrange(500, 30000) for _ in range(rnd.randint(1, 4))}))
    ns = (64, 128, 256, 512)
    chunk_p, chunk_d = rnd.choice([256, 512, 0]), rnd.choice([256, 512, 0])
    a_d, b_d, c_d = rnd.uniform(5, 60), rnd.uniform(0.3, 1.5), rnd.uniform(0.0, 0.08)
    a_p, b_p, c_p = rnd.uniform(20, 150), rnd.uniform(0.02, 0.3), rnd.uniform(0.0, 0.02)
    dm = lambda d, n, a=a_d, b=b_d, c=c_d: a + (b + c * d / 1000.0) * n + rnd.uniform(0, 3)  # noqa: E731
    pm = lambda d, n, a=a_p, b=b_p, c=c_p: a + (b + c * d / 1000.0) * n + rnd.uniform(0, 3)  # noqa: E731
    fdeps = tuple(sorted({0, depths[-1] + 2000} | {rnd.randrange(100, depths[-1] + 1000) for _ in range(rnd.randint(0, 3))}))
    fsecs = tuple(round(rnd.uniform(0.05, 0.6), 3) for _ in fdeps)
    cv = xc.XCurves(format=xc.X_CURVES_FORMAT, identity=T._curves().identity,
                    p_curve=T._curve(pm, chunk=chunk_p, depths=depths, ns=ns),
                    d_curve=T._curve(dm, chunk=chunk_d, depths=depths, ns=ns),
                    flip_price=xc.FlipPrice(depth=fdeps, seconds=fsecs, attribution="calib_prefix", pairs=9))
    reach = depths[-1]
    nsel = sorted({reach, reach // 2, reach // 3, rnd.randrange(1, reach + 1), rnd.randrange(1, reach + 1)})
    return cv, nsel


def test_n2s_property_full_scan_step_1_equals_the_table_seed_4242_60_sets():
    """The property: for 60 random curve sets (Seed 4242: the review's generator, set 14 is the fold) the minimum over the
    whole depth range with step 1 equals ``_x_min_over_depths`` for every probed ``n`` and for ``None``."""
    rnd = random.Random(4242)
    bad = []
    for i in range(60):
        cv, nsel = _random_set(rnd)
        f = _setup(cv, 2048)
        hi = max(nsel)
        xs = [F.Front._x_at_depth(f, d) for d in range(0, hi + 1)]
        for n in nsel:
            true, got = min(xs[: n + 1]), F.Front._x_min_over_depths(f, n)
            if got != true:
                bad.append((i, n, got, true))
        if hi == max(r.depth for r in cv.d_curve.rows):
            got, true = F.Front._x_min_over_depths(f, None), min(xs)
            if got > true:
                bad.append((i, None, got, true))
    assert not bad, bad


def test_n2s_h4_both_sides_of_every_flip_crossing_are_candidates():
    """H4 (RN2d): where ``depth + X(depth) - fd`` changes sign, BOTH adjacent integers are in the table (a minimum on the
    right side of the crossing is found)."""
    price = xc.FlipPrice(depth=(0, 20000, 90000, 262144), seconds=(4.0, 2.0, 6.0, 7.0), attribution="calib_prefix", pairs=9)
    f = _setup(T._curves(price=price), 12288)
    depths, _ = F.Front._x_kink_table(f)
    have = set(depths)
    seen = 0
    for fd in (20000, 90000):
        prev = None
        for d in range(max(0, fd - 20000), fd + 1):
            g = d + F.Front._x_at_depth(f, d) - fd
            if prev is not None and (prev[1] < 0) != (g < 0):
                seen += 1
                assert prev[0] in have and d in have, (fd, prev[0], d)
            prev = (d, g)
    assert seen >= 1, "the scenario must contain a crossing, else the test proves nothing"


def test_n2s_h4_flip_crossing_keeps_both_integers_of_the_sign_change():
    """H4 (RN2d) at the unit: the integer bisection of ``depth + X(depth) - fd`` keeps ``a`` AND ``b = a + 1`` (+ the outer
    neighbours). X drops from 100 to 1 at depth 41: ``depth + X - 90`` is +50 at 40 and -48 at 41 -- a minimum on ``b`` must
    stay a candidate."""
    xat = lambda d: 100 if d <= 40 else 1  # noqa: E731
    got = F.Front._x_flip_crossings(xat, [0, 60], 90)
    assert got == {39, 40, 41, 42}, got
    assert F.Front._x_flip_crossings(xat, [0, 30], 90) == set(), "no sign change: nothing to bisect"
