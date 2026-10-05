# SPDX-License-Identifier: Apache-2.0
"""#1920 HEAD-BYPASS-FLOOR (dual P, PP0): a P head that D's live floor keeps short does not hold younger,
immediately grantable requests back.

Evidence (deskq/done/1890 lever B, 1910): b9p/top112 still had P-KV waits of 63-127 s and ADMISSION-WEDGEs of
65 s; `D-LIVE-FLOOR mapped=200704 need=114688 floor=198899 live_row=198899` -- a RUNNING D request on a high row
blocks the D shrink, and the oldest P waiter then holds (hold=older-head, `_older_waits`) every younger request
once it is older than SGLANG_WEG2_DUAL_BYPASS_HEAD_AGE_S (60 s), although the younger ones fit the free bytes.

The rule, all behind SGLANG_WEG2_DUAL_HEAD_BYPASS_FLOOR (default OFF), dual P only:
  * D TP0 publishes the flag `SHRINK-BLOCKED reason=live_floor` (from the group values of its tick, a /dev/shm
    file, refreshed every 0.5 s, False written once on the way back); PP0 reads it, old flag = no flag;
  * a head older than the head age, whose grant is NOT satisfiable from the ledgers' free bytes now, is
    passed by a younger request whose OWN atomic group grant succeeds now (group_grant unchanged);
  * starvation guard: a head that fits the free bytes keeps its priority; at most
    SGLANG_WEG2_DUAL_HEAD_BYPASS_FLOOR_MAX overtakers pass one head, then hold=older-head as before;
  * rank agreement: only PP0 runs pp0_grant (the told carries the verdict), the decision uses no clock beyond
    the existing monotonic head age; D's flag is written by TP0 only from group-collective values.
Marker: '#1920 HEAD-BYPASS-FLOOR rid=... head=... overtakers=n'.
"""
from __future__ import annotations

import inspect
import json
import os
import tempfile
import time
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_d_kv_stage as DK
from sglang.srt.weg2 import dual_d_priority as DDP
from sglang.srt.weg2 import dual_parallel as DPAR
from sglang.srt.weg2 import dual_p_kv_stage as S
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# card 0: pool 6000, D holds 3000 (free 3000). step 4096: <=4096 tokens -> index 1, <=8192 -> index 2.
POOL, D_HOLD = 6000, 3000
BYTES = [0, 500, 4000] + [4000] * 62          # younger (1064 tokens) needs 500, head (level 8000) needs 4000


def _ledger(path):
    K.CardKvLedger(path, "D").contribute(POOL, committed=D_HOLD)
    K.CardKvLedger(path, "P").join(POOL)


def _stage(path_ledger):
    return {"ledger": path_ledger, "step": 4096, "top": 196608, "bytes": list(BYTES)}


class _Req:
    def __init__(self, rid, n=1000, wait=False):
        self.rid = rid
        self.origin_input_ids = [0] * n
        self._dual_kv_wait = wait


class _Actor:
    page = 64
    _committed = 0
    ledger = None

    def __init__(self):
        self.mapped = []

    def map_granted(self, lvl, charged=0):
        self.mapped.append((lvl, charged))


class _Sched:
    def __init__(self, pp_rank=0, held=None):
        self.ps = type("ps", (), {"pp_rank": pp_rank, "pp_size": 1})()
        self._weg2_store_held = held or {}


class _Fixture(CustomTestCase):
    HEAD, YOUNG = "weg2-0-9", "weg2-0-10"

    def setUp(self):
        S._RETRY.clear()
        S._STAGE_CACHE.clear()
        S._reset_wait_log()
        self.d = tempfile.mkdtemp(prefix="g1920")
        self.led = os.path.join(self.d, "wkv-0")
        _ledger(self.led)
        self.stage_path = os.path.join(self.d, "wkvs-pp0.json")
        with open(self.stage_path, "w") as f:
            json.dump(_stage(self.led), f)
        self.flag_path = os.path.join(self.d, "wkvf-x.json")
        self.clock = [1000.0]
        self.lines = []
        self.actor = _Actor()
        self.on, self.max_n = True, 4

    def publish_flag(self, blocked=True, age=0.0):
        DDP.publish_floor_signal(self.flag_path, blocked=blocked, mapped=200704, need=114688, floor=198899,
                                 p_wait_s=70.0, now=time.time() - age)

    def head(self, age_s=100.0, level=8000, record=True):
        """An old held head (past the 60 s head age) that already tried at ``level`` tokens."""
        head = _Req(self.HEAD, wait=True)
        S._WAITS[self.HEAD] = [self.clock[0] - age_s, self.clock[0], 1.0, 5, 0]
        if record:
            S._HEAD_LV[self.HEAD] = level
        return head, _Sched(held={"x": head})

    def grant(self, rid, sched, *, n=1000, layout=True):
        patches = [
            mock.patch.object(S, "_actor", lambda s: self.actor),
            mock.patch.object(S, "stage_file", lambda tag, r, root="/dev/shm": self.stage_path),
            mock.patch.object(S, "_dual_layout_env", lambda: layout),
            mock.patch.object(S, "_retry_ms", lambda: 0),
            mock.patch.object(S, "return_untold_grant", lambda *a, **k: None),
            mock.patch.object(S, "live_grant_tokens", lambda *a, **k: 0),
            mock.patch.object(S, "_now", lambda: self.clock[0]),
            mock.patch.object(S.logger, "info", lambda msg, *a: self.lines.append(msg % a)),
            mock.patch.object(DPAR, "head_bypass_floor_armed", lambda: self.on),
            mock.patch.object(DPAR, "head_bypass_floor_max", lambda: self.max_n),
            mock.patch.object(DDP, "floor_signal_file", lambda tag, root="/dev/shm": self.flag_path),
            mock.patch("sglang.srt.weg2.dual_grant_wait.grant_held_by_d", lambda *a, **k: None),
        ]
        for p in patches:
            p.start()
        try:
            with mock.patch.object(S, "group_grant", wraps=S.group_grant) as gg:
                r = S.pp0_grant(sched, _Req(rid, n=n))
            return r, gg.call_count
        finally:
            for p in patches:
                p.stop()

    def marker(self):
        return [l for l in self.lines if "#1920 HEAD-BYPASS-FLOOR" in l]

    def older_head_lines(self):
        return [l for l in self.lines if "hold=older-head" in l]


class DefaultsAndOffPath(_Fixture):
    def test_switch_default_is_off_and_cap_default_is_four(self):
        from sglang.srt.environ import envs

        self.assertFalse(envs.SGLANG_WEG2_DUAL_HEAD_BYPASS_FLOOR.get())
        self.assertEqual(envs.SGLANG_WEG2_DUAL_HEAD_BYPASS_FLOOR_MAX.get(), 4)
        self.assertFalse(DPAR.head_bypass_floor_armed())
        self.assertEqual(DPAR.head_bypass_floor_max(), 4)

    def test_off_old_behaviour_even_with_a_published_flag_and_a_known_head_level(self):
        self.on = False
        self.publish_flag()
        _head, sched = self.head()
        r, ggn = self.grant(self.YOUNG, sched)
        self.assertEqual((r, ggn), (0, 0), "switch off: hold=older-head exactly as before, group_grant not reached")
        self.assertTrue(self.older_head_lines())
        self.assertEqual(self.marker(), [])
        self.assertEqual(S._OVERTAKERS, {})

    def test_off_a_waiting_head_records_no_level(self):
        self.on = False
        _head, sched = self.head(record=False)
        S._WAITS.pop(self.HEAD)                      # the head's own attempt (nobody older)
        r, _ = self.grant(self.HEAD, _Sched(), n=7500)   # level 8000 -> need 4000 > free 3000 -> waits
        self.assertEqual(r, 0)
        self.assertEqual(S._HEAD_LV, {}, "no per-attempt state while the switch is off")

    def test_off_mid_age_head_is_passed_as_before(self):
        self.on = False
        _head, sched = self.head(age_s=10.0)         # younger than the 60 s age: the old Q-670 bypass
        r, ggn = self.grant(self.YOUNG, sched)
        self.assertGreater(r, 0)
        self.assertEqual(self.marker(), [], "the plain age bypass is not the floor rule and writes no #1920 marker")


class FloorBypass(_Fixture):
    def test_on_floor_blocked_head_is_passed_by_a_grantable_younger_one(self):
        self.publish_flag(blocked=True)
        _head, sched = self.head()
        r, ggn = self.grant(self.YOUNG, sched)
        self.assertGreater(r, 0, "the younger request fits (500 <= free 3000) and is granted")
        self.assertEqual(ggn, 1)
        self.assertFalse(self.older_head_lines())
        m = self.marker()
        self.assertEqual(len(m), 1)
        self.assertIn("rid=%s head=%s overtakers=1" % (self.YOUNG, self.HEAD), m[0])
        self.assertEqual(S._OVERTAKERS, {self.HEAD: 1})
        self.assertEqual(len(self.actor.mapped), 1, "PP0 mapped the overtaker's grant")

    def test_the_head_stays_head(self):
        self.publish_flag()
        head, sched = self.head()
        self.grant(self.YOUNG, sched)
        self.assertIn(self.HEAD, S._WAITS, "the overtake neither grants nor drops the head's wait")
        self.assertTrue(head._dual_kv_wait)
        self.assertEqual([k for k, _t in S._older_waits(sched, "weg2-0-11")], [self.HEAD])
        # and the head itself, when it finally fits, is granted and forgets its counters
        _ledger_free = K.peek(self.led).free
        self.assertEqual(_ledger_free, POOL - D_HOLD - BYTES[1])
        K.CardKvLedger(self.led, "D").release(2000)     # D shrinks: free 5000 -> head (4000) fits
        S._WAITS.pop(self.HEAD)
        S._WAITS[self.HEAD] = [self.clock[0] - 100.0, self.clock[0], 1.0, 5, 0]
        r, _ = self.grant(self.HEAD, _Sched(), n=7500)
        self.assertGreater(r, 0)
        self.assertNotIn(self.HEAD, S._OVERTAKERS)
        self.assertNotIn(self.HEAD, S._HEAD_LV)

    def test_an_ungrantable_younger_one_is_not_counted_and_takes_nothing(self):
        self.publish_flag()
        _head, sched = self.head()
        big = 7500                                     # level 8000 -> 4000 > free 3000
        r, ggn = self.grant(self.YOUNG, sched, n=big)
        self.assertEqual((r, ggn), (0, 1), "it reached group_grant (no older-head hold) and the grant was short")
        self.assertEqual(S._OVERTAKERS, {})
        self.assertEqual(self.marker(), [])
        self.assertEqual(K.peek(self.led).free, POOL - D_HOLD, "a short atomic grant leaves the ledger untouched")

    def test_no_floor_flag_means_the_old_hold(self):
        _head, sched = self.head()                     # no flag file at all
        r, ggn = self.grant(self.YOUNG, sched)
        self.assertEqual((r, ggn), (0, 0))
        self.assertTrue(self.older_head_lines())
        self.assertEqual(self.marker(), [])

    def test_flag_false_or_stale_means_the_old_hold(self):
        for kw in ({"blocked": False}, {"blocked": True, "age": 10.0}):
            S._reset_wait_log()
            self.lines.clear()
            self.publish_flag(**kw)
            _head, sched = self.head()
            r, ggn = self.grant(self.YOUNG, sched)
            self.assertEqual((r, ggn), (0, 0), kw)
            self.assertTrue(self.older_head_lines(), kw)

    def test_garbage_flag_file_means_the_old_hold(self):
        with open(self.flag_path, "w") as f:
            f.write("{not json")
        _head, sched = self.head()
        r, ggn = self.grant(self.YOUNG, sched)
        self.assertEqual((r, ggn), (0, 0))

    def test_unknown_head_level_means_the_old_hold(self):
        self.publish_flag()
        _head, sched = self.head(record=False)
        r, ggn = self.grant(self.YOUNG, sched)
        self.assertEqual((r, ggn), (0, 0), "no recorded attempt of the head: it cannot be proven short, it is protected")

    def test_head_that_fits_the_free_bytes_keeps_its_priority(self):
        self.publish_flag()
        _head, sched = self.head(level=4096)           # level index 1: 500 <= free 3000 -> grantable NOW
        r, ggn = self.grant(self.YOUNG, sched)
        self.assertEqual((r, ggn), (0, 0), "the head is servable: no overtaking, its next retry takes the bytes")
        self.assertTrue(self.older_head_lines())
        self.assertEqual(S._OVERTAKERS, {})

    def test_starvation_cap_n_overtakers_then_the_head_holds_again(self):
        self.publish_flag()
        self.max_n = 2
        _head, sched = self.head()
        got = []
        for i in range(4):
            self.clock[0] += 1.0                       # past the 0.25 s flag cache
            self.publish_flag()
            r, ggn = self.grant("weg2-0-%d" % (20 + i), sched)
            got.append((r > 0, ggn))
        self.assertEqual(got, [(True, 1), (True, 1), (False, 0), (False, 0)])
        self.assertEqual(S._OVERTAKERS, {self.HEAD: 2})
        self.assertEqual(len(self.marker()), 2)
        self.assertIn("overtakers=2", self.marker()[1])
        self.assertEqual(len(self.older_head_lines()), 2, "the third and fourth are held by the head again")

    def test_cap_zero_disables_the_rule(self):
        self.publish_flag()
        self.max_n = 0
        _head, sched = self.head()
        r, ggn = self.grant(self.YOUNG, sched)
        self.assertEqual((r, ggn), (0, 0))

    def test_head_that_becomes_servable_after_overtakers_is_protected_again(self):
        self.publish_flag()
        _head, sched = self.head()
        self.grant(self.YOUNG, sched)                  # one overtaker passes (free 3000 -> 2500)
        K.CardKvLedger(self.led, "D").release(2000)    # D finally shrinks: free 4500 >= head's 4000
        self.clock[0] += 1.0
        self.publish_flag()
        r, ggn = self.grant("weg2-0-11", sched)
        self.assertEqual((r, ggn), (0, 0), "the head fits now: it goes first, nobody else")
        self.assertEqual(S._OVERTAKERS, {self.HEAD: 1})

    def test_young_head_is_passed_by_the_plain_age_rule_and_not_counted(self):
        self.publish_flag()
        _head, sched = self.head(age_s=5.0)
        r, ggn = self.grant(self.YOUNG, sched)
        self.assertGreater(r, 0)
        self.assertEqual(S._OVERTAKERS, {}, "within the head age nothing is an order violation of the floor rule")
        self.assertEqual(self.marker(), [])

    def test_map_short_wait_is_not_counted(self):
        from sglang.srt.weg2.dual_p_kv_stage import Weg2DualKvMapShort

        self.publish_flag()
        _head, sched = self.head()

        def boom(lvl, charged=0):
            raise Weg2DualKvMapShort("card short")
        self.actor.map_granted = boom
        with mock.patch.object(S, "phys_free_bytes", lambda: None):
            r, _ = self.grant(self.YOUNG, sched)
        self.assertEqual(r, 0)
        self.assertEqual(S._OVERTAKERS, {})
        self.assertEqual(self.marker(), [])


class RankAgreementAndFlipUnchanged(_Fixture):
    def test_a_non_pp0_rank_decides_nothing(self):
        self.publish_flag()
        _head, _sched = self.head()
        r, ggn = self.grant(self.YOUNG, _Sched(pp_rank=1))
        self.assertIsNone(r)
        self.assertEqual((ggn, S._OVERTAKERS, S._HEAD_LV), (0, {}, {self.HEAD: 8000}))

    def test_the_decision_is_a_pure_function_of_group_facts_and_the_monotonic_head_age(self):
        """Same inputs -> same verdict, no wall clock inside floor_overtake_head (the flag's ts is checked in
        d_floor_blocked, PP0 only); every PP0 re-run with the same facts decides alike."""
        src = inspect.getsource(S.floor_overtake_head)
        self.assertNotIn("time.time", src)
        self.assertNotIn("random", src)
        self.publish_flag()
        _head, sched = self.head()
        stages = [_stage(self.led)]
        older = S._older_waits(sched, "weg2-0-11")
        with mock.patch.object(DPAR, "head_bypass_floor_armed", lambda: True), \
                mock.patch.object(S, "d_floor_blocked", lambda tag: True):
            a = S.floor_overtake_head(older, stages, 0, "t")
            b = S.floor_overtake_head(older, stages, 0, "t")
        self.assertEqual((a, b), (self.HEAD, self.HEAD))

    def test_flip_form_without_the_dual_layout_never_consults_the_rule(self):
        self.publish_flag()
        _head, sched = self.head()
        with mock.patch.object(S, "floor_overtake_head", side_effect=AssertionError("flip path reached #1920")), \
                mock.patch.object(S, "d_floor_blocked", side_effect=AssertionError("flip path read the flag")):
            r, ggn = self.grant(self.YOUNG, sched, layout=False)
        self.assertGreater(r, 0, "flip form: no older-head wait at all, the base grant path")
        self.assertEqual((S._OVERTAKERS, self.marker()), ({}, []))

    def test_pp0_grant_follower_path_is_the_told(self):
        """Followers never call pp0_grant: the intake step does so on pp_rank 0 only (the told carries the verdict)."""
        from sglang.srt.managers import weg2_store_told as ST

        src = inspect.getsource(ST.intake)
        self.assertIn("int(scheduler.ps.pp_rank) == 0", src)
        self.assertLess(src.index("int(scheduler.ps.pp_rank) == 0"), src.index("_dpk.pp0_grant"))


class FloorSignalHelpers(CustomTestCase):
    def test_floor_signal_blocked_pure(self):
        now = 1000.0
        self.assertTrue(DDP.floor_signal_blocked({"ts": 999.0, "blocked": 1}, now=now))
        self.assertFalse(DDP.floor_signal_blocked({"ts": 999.0, "blocked": 0}, now=now))
        self.assertFalse(DDP.floor_signal_blocked({"ts": 990.0, "blocked": 1}, now=now))
        self.assertFalse(DDP.floor_signal_blocked(None, now=now))
        self.assertFalse(DDP.floor_signal_blocked({"ts": "x", "blocked": 1}, now=now))
        self.assertFalse(DDP.floor_signal_blocked({"blocked": 1}, now=now))

    def test_head_grantable_now_unreadable_means_protected(self):
        self.assertTrue(S.head_grantable_now([], {}, 100))
        self.assertTrue(S.head_grantable_now([{"nope": 1}], {}, 100))
        self.assertTrue(S.head_grantable_now([_stage("/nonexistent/ledger")], {}, 8000))


class _DActor:
    mapped_tokens = 200704


class DSideFlag(CustomTestCase):
    """D TP0 publishes the flag; switch off / other TP rank / nothing blocked = no file."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="g1920d")
        self.path = os.path.join(self.d, "wkvf-t.json")
        self.clock = [500.0]
        self.sched = type("S", (), {"tp_rank": 0})()
        self.actor = _DActor()

    def _pub(self, blocked, *, armed=True, sched=None):
        with mock.patch.object(DPAR, "head_bypass_floor_armed", lambda: armed), \
                mock.patch.object(DDP, "floor_signal_file", lambda tag, root="/dev/shm": self.path), \
                mock.patch.object(DK._pk, "_now", lambda: self.clock[0]):
            DK._publish_floor_flag(sched or self.sched, self.actor, blocked, 114688, 198899, 70.0)

    def _read(self):
        return DDP.read_floor_signal(self.path)

    def test_off_writes_nothing(self):
        self._pub(True, armed=False)
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(getattr(self.actor, "_floor_sig_on", False))

    def test_other_tp_rank_writes_nothing(self):
        self._pub(True, sched=type("S", (), {"tp_rank": 1})())
        self.assertFalse(os.path.exists(self.path))

    def test_nothing_blocked_and_nothing_published_writes_nothing(self):
        self._pub(False)
        self.assertFalse(os.path.exists(self.path))

    def test_blocked_is_published_refreshed_rate_limited_and_taken_back_once(self):
        self._pub(True)
        sig = self._read()
        self.assertEqual((sig["blocked"], sig["mapped"], sig["need"], sig["floor"]), (1, 200704, 114688, 198899))
        t0 = sig["ts"]
        time.sleep(0.01)
        self.clock[0] += 0.1
        self._pub(True)
        self.assertEqual(self._read()["ts"], t0, "within 0.5 s: no rewrite")
        self.clock[0] += 0.5
        self._pub(True)
        self.assertGreater(self._read()["ts"], t0, "refreshed after FLOOR_SIGNAL_EVERY_S")
        self._pub(False)
        self.assertEqual(self._read()["blocked"], 0, "the way back is written at once")
        self.assertFalse(self.actor._floor_sig_on)
        before = self._read()["ts"]
        time.sleep(0.01)
        self._pub(False)
        self.assertEqual(self._read()["ts"], before, "False is written once, not on every tick")

    def test_tick_calls_the_flag_from_the_shrink_blocked_branch_only(self):
        src = inspect.getsource(DK.tick)
        self.assertIn('_publish_floor_flag(sched, actor, blocked_reason == "live_floor"', src)
        self.assertIn("_publish_floor_flag(sched, actor, False", src)

    def test_publish_failure_never_reaches_the_tick(self):
        with mock.patch.object(DPAR, "head_bypass_floor_armed", lambda: True), \
                mock.patch.object(DDP, "floor_signal_file", lambda tag, root="/dev/shm": "/nonexistent-dir/x.json"):
            DK._publish_floor_flag(self.sched, self.actor, True, 1, 2, 3.0)    # must not raise


if __name__ == "__main__":
    import unittest

    unittest.main()
