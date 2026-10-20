"""S86d (review S86c A1, done/nf-review-s86c-1020.md): after S86c, W86 hardly saw anything -- the agreed-set digest
is equal by construction, and the residual compared the candidates with the release's OWN list of what it released,
never whether the reference really went away.

The residual now reads the state AFTER the release: every agreed slot a node of this rank's tree still names in its
Full host rows. A rank that "releases" (its list says so) but keeps the reference -- the host rows -- raises W86 on
every rank in the same pass. Not the per-process RefLedger: Form A workers queue their arena references (S83) and
give them back before each #1470 round, a correct release would read as a residual there (false W86).

Same hermetic set-up as the S86 tests (real C arena, two real ArenaMHAHostPool ranks on one file, threads). RED on
177190875e (the old residual is 0, no W86)."""
import os
import sys
import threading
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_s86_sleep_l3_release_1020 as B  # noqa: E402
from test_s86_sleep_l3_release_1020 import _thread_roles, rc19  # noqa: E402,F401

from flliper.srt.pdflip import sleep_l3_release as S86  # noqa: E402

FULL, SLOTS = B.FULL, B.SLOTS


def _both(g):
    grp = B._Group(2)
    res = [None, None]

    def run(i, role, t, p):
        B._ROLE.r = role
        try:
            res[i] = S86.run(t, grp.member(i), p)
        except BaseException as e:  # noqa: BLE001
            res[i] = e
        finally:
            B._ROLE.r = None

    ths = [threading.Thread(target=run, args=(0, "host", g.t0, g.p0)),
           threading.Thread(target=run, args=(1, "worker", g.t1, g.p1))]
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
    return res, grp


def test_a_rank_that_releases_but_keeps_the_reference_raises_w86_everywhere(rc19):
    """The worker's release runs (the candidates go into its released list), but the eviction keeps the host rows:
    the reference is still there. The residual from the tree's state names those slots on the worker; every rank
    reads it from the digest and raises W86 DIVERGED."""
    g = rc19
    real = g.t1._evict_component_and_detach_lru

    def keep_rows(node, comp, target=None, tracker=None):
        return 0, 0                                     # "released" -- the rows (and the arena references) stay

    with mock.patch.object(g.t1, "_evict_component_and_detach_lru", keep_rows):
        res, grp = _both(g)
    assert all(isinstance(r, S86.SleepL3ReleaseDivergence) for r in res), res
    assert all("DIVERGED" in str(r) and "residual" in str(r) for r in res)
    assert grp.calls[0] == grp.calls[1]
    assert real is not None


def test_a_correct_release_has_residual_zero_on_every_rank(rc19):
    g = rc19
    res, grp = _both(g)
    assert not any(isinstance(r, BaseException) for r in res), res
    assert res[0]["residual"] == res[1]["residual"] == 0
    assert B._refs(g) == [0] * SLOTS


def test_the_kept_vetoed_nodes_are_no_residual(rc19):
    """Nodes the group vetoed keep their host rows by design (here: a host lock on both ranks) -- no residual."""
    g = rc19
    g.n0[1].component_data[FULL].host_lock_ref = 1
    g.n1[4].component_data[FULL].host_lock_ref = 1
    res, grp = _both(g)
    assert not any(isinstance(r, BaseException) for r in res), res
    assert res[0]["residual"] == res[1]["residual"] == 0
    assert B._refs(g)[1] == 2 and B._refs(g)[4] == 2
