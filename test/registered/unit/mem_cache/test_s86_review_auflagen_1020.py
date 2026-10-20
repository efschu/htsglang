"""S86 review conditions (Reviewer 2, nf-review-s86-1020.md, JA MIT AUFLAGE).

A1  ``DIVERGED`` -- and a rank's error after the first collective -- stop the group by NAME
    (``W86 SleepL3ReleaseDivergence``, RAENGE-NIE-UNEINS), on every rank in the same pass, never a log line only.
A2  the number and order of the three collectives depend on no local value (pool missing, other slot count, veto
    error, nothing to release, release error, divergence): every rank posts [slots, veto, digest] -- the reviewer's
    S8 (the count following the LOCAL slot value) hangs here and turns red.
A3a an unpaired page is never secured, whatever FLLIPER_PDFLIP_L3_PRESENT_STRICT says (data-loss direction).
A3b a rank without pool/arena, or with another slot count, vetoes every slot (fail closed).
    And the guards: a write in flight, staging rows, a host-only node keep their slots on EVERY rank.

Same hermetic set-up as test_s86_sleep_l3_release_1020 (real C arena, two ArenaMHAHostPool ranks on one file, the
ranks in two threads, the group MAX behind a barrier)."""
import os
import sys
import threading
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_s86_sleep_l3_release_1020 as B  # noqa: E402
from test_s86_sleep_l3_release_1020 import _thread_roles, rc19  # noqa: E402,F401

from flliper.srt.pdflip import sleep_l3_release as S86  # noqa: E402

FULL, SLOTS = B.FULL, B.SLOTS
LABELS = ["s86_sleep_l3_release/slots", "s86_sleep_l3_release/veto", "s86_sleep_l3_release/digest"]


class _Group(B._Group):
    def __init__(self, n, timeout=6):
        super().__init__(n)
        self.bar = threading.Barrier(n, timeout=timeout)


def _both(g, pools=None, grp=None):
    """Both ranks run the step; returns [(result | exception)] per rank and the group."""
    grp = grp or _Group(2)
    p0, p1 = pools or (g.p0, g.p1)
    res = [None, None]

    def run(i, role, t, p):
        B._ROLE.r = role
        try:
            res[i] = S86.run(t, grp.member(i), p)
        except BaseException as e:  # noqa: BLE001
            res[i] = e
        finally:
            B._ROLE.r = None

    ths = [threading.Thread(target=run, args=(0, "host", g.t0, p0)),
           threading.Thread(target=run, args=(1, "worker", g.t1, p1))]
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
    return res, grp


def _kept(g, i):
    return (g.n0[i].component_data[FULL].host_value is not None,
            g.n1[i].component_data[FULL].host_value is not None)


def _assert_agreed(res, grp):
    assert not any(isinstance(r, BaseException) for r in res), res
    assert grp.calls[0] == grp.calls[1] == LABELS


# ---- A3a / A3 guards: kept on EVERY rank ------------------------------------------------------------------------

def test_a3a_an_unpaired_page_is_never_secured_even_with_the_strict_switch_off(rc19, monkeypatch):
    g = rc19
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_PRESENT_STRICT", "0")
    victim_rows = [g.n0[4].component_data[FULL].host_value, g.n1[4].component_data[FULL].host_value]
    for p in (g.p0, g.p1):
        orig = p.secure_rows_to_l3

        def wrapped(hv, _orig=orig):
            sec = dict(_orig(hv))
            if any(torch.equal(hv, v) for v in victim_rows):
                # the QS pair gate held this KV page back (no QSA index): NOT on disk
                sec.update(unpaired=1, on_disk=max(0, int(sec.get("on_disk", 0)) - 1))
            return sec
        p.secure_rows_to_l3 = wrapped
    res, grp = _both(g)
    _assert_agreed(res, grp)
    assert _kept(g, 4) == (True, True), "an unpaired page left L2 (data loss with the strict switch off)"
    assert B._refs(g)[4] == 2


def test_a_write_in_flight_keeps_its_slot_on_every_rank(rc19):
    g = rc19
    g.n1[1].write_through_pending_id = 4711                    # the worker's write of this node is in flight
    g.t0.ongoing_write_through[g.n0[6].id] = object()          # TP0's write of another node is in flight
    res, grp = _both(g)
    _assert_agreed(res, grp)
    assert _kept(g, 1) == (True, True) and _kept(g, 6) == (True, True)
    assert B._refs(g)[1] == 2 and B._refs(g)[6] == 2


def test_staging_rows_keep_their_node_on_every_rank(rc19):
    g = rc19
    cd = g.n0[2].component_data[FULL]
    cd.host_value = torch.cat([torch.tensor([0], dtype=torch.int64), cd.host_value])   # a staging row (< S)
    res, grp = _both(g)
    _assert_agreed(res, grp)
    assert _kept(g, 2) == (True, True)


def test_a_host_only_node_keeps_its_slot_on_every_rank(rc19):
    g = rc19
    g.n0[5].component_data[FULL].value = None                  # TP0 has this node off the device: host-only
    res, grp = _both(g)
    _assert_agreed(res, grp)
    assert _kept(g, 5) == (True, True)
    assert B._refs(g)[5] == 2


# ---- A3b: fail closed --------------------------------------------------------------------------------------------

def test_a3b_a_rank_without_a_pool_vetoes_everything(rc19):
    """The worker cannot read its pool: it vetoes every slot -- nothing leaves anywhere. Before: a zero veto (it
    agreed). With the reviewer's S8 (the count following the LOCAL slot value) this rank returns after the first
    collective and its peer hangs at the second: red."""
    g = rc19
    g.t1._pdflip_direct_pool = lambda: None
    res, grp = _both(g, pools=(g.p0, None))
    _assert_agreed(res, grp)
    assert B._refs(g) == [2] * SLOTS
    assert res[1]["why"] == {"no_pool": 1}


def test_a3b_another_slot_count_vetoes_everything(rc19):
    g = rc19
    real = g.a1.slots
    try:
        g.a1.slots = real - 1                                  # this rank reads another arena geometry
        res, grp = _both(g)
    finally:
        g.a1.slots = real
    _assert_agreed(res, grp)
    assert B._refs(g) == [2] * SLOTS
    assert res[1]["why"] == {"slot_count": 1}


# ---- A1: DIVERGED and errors stop the group by name ----------------------------------------------------------------

def test_a1_diverged_stops_every_rank_by_name(rc19):
    """The worker's node 2 spans the pages a2 AND a3, TP0's only a2; the worker keeps a3 (host lock). The group
    veto names slot 3: TP0 lets go of slot 2, the worker cannot (its node holds slot 3 too) -- the ranks let go of
    different slots. Every rank reads the same reduced digest and raises W86."""
    g = rc19
    n = g.n1[2]
    cd = n.component_data[FULL]
    cd.host_value = torch.cat([cd.host_value, g.n1[3].component_data[FULL].host_value])
    n.hash_value = ["a2", "a3"]
    g.n1[3].component_data[FULL].host_lock_ref = 1
    res, grp = _both(g)
    assert all(isinstance(r, S86.SleepL3ReleaseDivergence) for r in res), res
    assert all("DIVERGED" in str(r) and "W86" in str(r) for r in res)
    assert grp.calls[0] == grp.calls[1] == LABELS


def test_a1_an_error_after_the_first_collective_stops_every_rank_by_name(rc19):
    """The worker's release raises half-way: the error rides the digest collective (no peer left waiting), and
    every rank raises W86 in the same pass instead of flushing on."""
    g = rc19
    real = S86.apply

    def boom(tree, pool, cands, agreed, released=None):
        if tree is g.t1:
            raise RuntimeError("release failed on the worker")
        return real(tree, pool, cands, agreed, released)

    with mock.patch.object(S86, "apply", boom):
        res, grp = _both(g)
    assert all(isinstance(r, S86.SleepL3ReleaseDivergence) for r in res), res
    assert grp.calls[0] == grp.calls[1] == LABELS


def test_a1_the_caller_lets_the_named_stop_through():
    from flliper.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager as WU

    wu = object.__new__(WU)
    with mock.patch.object(S86, "run", side_effect=S86.SleepL3ReleaseDivergence("W86 x")):
        with pytest.raises(S86.SleepL3ReleaseDivergence):
            wu._pdflip_sleep_l3_release(object(), object())
    with mock.patch.object(S86, "run", side_effect=RuntimeError("before the first collective")):
        assert wu._pdflip_sleep_l3_release(object(), object()) == {}


# ---- A2: the collectives depend on no local value -------------------------------------------------------------------

@pytest.mark.parametrize("case", ["plain", "no_pool", "slot_count", "veto_error", "nothing", "diverged", "apply_error"])
def test_a2_every_rank_posts_the_same_three_collectives_whatever_it_holds(rc19, case):
    g = rc19
    patches = []
    pools = (g.p0, g.p1)
    if case == "no_pool":
        g.t1._pdflip_direct_pool = lambda: None
        pools = (g.p0, None)
    elif case == "slot_count":
        g.a1.slots = SLOTS - 1
    elif case == "veto_error":
        real_r = S86._reason
        patches.append(mock.patch.object(S86, "_reason", lambda t, n, p: (_ for _ in ()).throw(RuntimeError("x"))
                                         if t is g.t1 else real_r(t, n, p)))
    elif case == "nothing":
        for n in g.n0 + g.n1:
            n.component_data[FULL].host_lock_ref = 1
    elif case == "diverged":
        n = g.n1[2]
        n.component_data[FULL].host_value = torch.cat([n.component_data[FULL].host_value,
                                                       g.n1[3].component_data[FULL].host_value])
        n.hash_value = ["a2", "a3"]
        g.n1[3].component_data[FULL].host_lock_ref = 1
    elif case == "apply_error":
        real_a = S86.apply
        patches.append(mock.patch.object(S86, "apply", lambda t, p, c, a, r=None: (_ for _ in ()).throw(
            RuntimeError("x")) if t is g.t1 else real_a(t, p, c, a, r)))
    try:
        for p in patches:
            p.start()
        res, grp = _both(g, pools=pools)
    finally:
        for p in patches:
            p.stop()
        g.a1.slots = SLOTS
    assert grp.calls[0] == grp.calls[1] == LABELS, (case, dict(grp.calls), res)


def test_a1_an_error_with_equal_digests_still_stops_every_rank_by_name(rc19):
    """Nothing is releasable anywhere (every node host-locked on both ranks), so both digests are equal (0 slots);
    the worker's release raises anyway. Only the error flag carried in the digest collective can name it -- every
    rank raises W86 (a rank that failed after the first collective never lets the group go on)."""
    g = rc19
    for n in g.n0 + g.n1:
        n.component_data[FULL].host_lock_ref = 1
    real = S86.apply

    def boom(tree, pool, cands, agreed, released=None):
        if tree is g.t1:
            raise RuntimeError("release failed on the worker")
        return real(tree, pool, cands, agreed, released)

    with mock.patch.object(S86, "apply", boom):
        res, grp = _both(g)
    assert all(isinstance(r, S86.SleepL3ReleaseDivergence) for r in res), res
    assert all("ERROR" in str(r) for r in res)
    assert B._refs(g) == [2] * SLOTS
