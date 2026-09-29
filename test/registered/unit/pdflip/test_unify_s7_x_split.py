"""UNIFY S7: the 27B RC7-X additions beyond the shared X-SOLO band, behind the
profile switch FLLIPER_PDFLIP_X_IDLE_REGRANT (ModelProfile.x_split), and the 27B
store-short tail behind FLLIPER_PDFLIP_STORE_SHORT_TAIL (ModelProfile.store_short_tail).

DANGER DIRECTIONS guarded here:
* nextflash (and no form) keeps the NF form exactly: no re-grant, no X_busy
  cap on the SHORT drain, no store-short tail X in D's env;
* qwen27b gets the 27B form: a deferred band backlog is served on D once D is
  idle and quiet (law 4 per request AND summed), never while D holds work,
  never past a non-deferrable request in the queue (law 1);
* an explicitly set switch wins over the profile.
Hermetic: no GPU, no boot, no HTTP.
"""
from __future__ import annotations

import asyncio
import os
import time
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import front as front_mod
from flliper.srt.pdflip import launcher as L
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

X = 4096
_SWITCHES = ("FLLIPER_PDFLIP_X_IDLE_REGRANT", "FLLIPER_PDFLIP_STORE_SHORT_TAIL")


def _form_env(profile, model="m"):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile != "nextflash"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return pdflip_form.PdFlipForm(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                              flip="family", vision="off", profile=profile, model=model).env_value()


class _Env:
    """Set the published form's profile (and optionally a switch) for a block."""

    def __init__(self, profile, **explicit):
        self.profile, self.explicit, self.saved = profile, explicit, {}

    def __enter__(self):
        for k in (pdflip_form.FORM_ENV,) + _SWITCHES:
            self.saved[k] = os.environ.pop(k, None)
        if self.profile is not None:
            os.environ[pdflip_form.FORM_ENV] = _form_env(self.profile)
        os.environ.update({k: str(v) for k, v in self.explicit.items()})
        return self

    def __exit__(self, *a):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def _front():
    f = front_mod.Front(
        prefill="http://p", decode="http://d", awake="D", tag="s7",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, tp_prefill_max_tokens=X, flip_min_work_tokens=X,
    )
    f.stops = []
    f.do_stop = lambda name, detail: f.stops.append((name, detail))
    return f


def _pending(rid, tokens, *, deferred=True, eligible=False, t_arrive=None):
    fut = asyncio.get_running_loop().create_future()
    return front_mod.Pending(rid, "/generate", {}, "x", t_arrive or time.time(), fut,
                             est_prompt=tokens, est_uncached=tokens,
                             d_eligible=eligible, x_deferred=deferred)


class TestProfileSwitches(CustomTestCase):
    def test_profile_rows_derive_the_two_switches(self):
        d27 = pdflip_form.PROFILE_SWITCH_DEFAULTS["qwen27b"]
        dnf = pdflip_form.PROFILE_SWITCH_DEFAULTS["nextflash"]
        self.assertIs(d27["FLLIPER_PDFLIP_X_IDLE_REGRANT"], True)
        self.assertIs(d27["FLLIPER_PDFLIP_STORE_SHORT_TAIL"], True)
        self.assertIs(dnf["FLLIPER_PDFLIP_X_IDLE_REGRANT"], False)
        self.assertIs(dnf["FLLIPER_PDFLIP_STORE_SHORT_TAIL"], False)

    def test_env_follows_the_form_and_explicit_wins(self):
        from flliper.srt.environ import envs
        from flliper.srt.managers import scheduler as S

        with _Env("qwen27b"):
            self.assertTrue(envs.FLLIPER_PDFLIP_X_IDLE_REGRANT.get())
            self.assertTrue(S._pdflip_store_short_tail_on())
        with _Env("nextflash"):
            self.assertFalse(envs.FLLIPER_PDFLIP_X_IDLE_REGRANT.get())
            self.assertFalse(S._pdflip_store_short_tail_on())
        with _Env(None):
            self.assertFalse(envs.FLLIPER_PDFLIP_X_IDLE_REGRANT.get())
            self.assertTrue(S._pdflip_store_short_tail_on())  # 27B code default, no form
        with _Env("nextflash", FLLIPER_PDFLIP_X_IDLE_REGRANT=1, FLLIPER_PDFLIP_STORE_SHORT_TAIL=1):
            self.assertTrue(envs.FLLIPER_PDFLIP_X_IDLE_REGRANT.get())
            self.assertTrue(S._pdflip_store_short_tail_on())
        with _Env("qwen27b", FLLIPER_PDFLIP_STORE_SHORT_TAIL=0):
            self.assertFalse(S._pdflip_store_short_tail_on())


class TestStoreShortTailEnv(CustomTestCase):
    def test_d_env_only_with_the_split_and_a_raised_guard(self):
        self.assertEqual(L.store_short_tail_env(4096, 12288, x_split=True),
                         {L.STORE_SHORT_TAIL_X_ENV: "4096"})
        self.assertEqual(L.store_short_tail_env(4096, 4096, x_split=True), {})
        self.assertEqual(L.store_short_tail_env(4096, 12288, x_split=False), {})
        with _Env("nextflash"):
            self.assertEqual(L.store_short_tail_env(4096, 12288), {})
        with _Env("qwen27b"):
            self.assertEqual(L.store_short_tail_env(4096, 12288), {L.STORE_SHORT_TAIL_X_ENV: "4096"})

    def test_scheduler_prices_the_tail_at_the_launch_x(self):
        from types import SimpleNamespace

        from flliper.srt.managers import scheduler as S

        sched = SimpleNamespace(server_args=SimpleNamespace(tp_prefill_max_tokens=12288))
        saved = os.environ.pop(S.STORE_SHORT_TAIL_X_ENV, None)
        try:
            self.assertEqual(S._pdflip_store_short_tail_x(sched), 12288)
            os.environ[S.STORE_SHORT_TAIL_X_ENV] = "4096"
            self.assertEqual(S._pdflip_store_short_tail_x(sched), 4096)
        finally:
            os.environ.pop(S.STORE_SHORT_TAIL_X_ENV, None)
            if saved is not None:
                os.environ[S.STORE_SHORT_TAIL_X_ENV] = saved


class TestIdleRegrant(CustomTestCase):
    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_off_is_the_nf_form(self):
        async def go():
            with _Env("nextflash"):
                f = _front()
            self.assertFalse(f.x_split)
            f.queue.append(_pending("a", 3000))
            self.assertEqual(f._x_idle_regrant(time.time() + 10), "none")
            self.assertEqual(len(f.queue), 1)
        self._run(go())

    def test_on_moves_a_deferred_backlog_once_quiet(self):
        async def go():
            with _Env("qwen27b"):
                f = _front()
            self.assertTrue(f.x_split)
            f.queue.append(_pending("a", 1500))
            f.queue.append(_pending("b", 1200, deferred=False, eligible=True))
            now = time.time()
            f._x_last_arrival = now
            self.assertEqual(f._x_idle_regrant(now), "wait")  # an arrival inside the window
            self.assertEqual(f._x_idle_regrant(now + 5), "moved")
            self.assertEqual(len(f.queue), 0)
            self.assertEqual([p.rid for p in f._ready_for_d], ["a", "b"])
            self.assertTrue(all(p.d_direct for p in f._ready_for_d))
            self.assertEqual(f.counters["x_idle_regrant_tokens"], 2700)
        self._run(go())

    def test_law4_summed_and_law1(self):
        async def go():
            with _Env("qwen27b"):
                f = _front()
            f.queue.append(_pending("a", 3000))
            f.queue.append(_pending("b", 2000))  # sum 5000 > X
            self.assertEqual(f._x_idle_regrant(time.time() + 10), "none")
            f.queue.clear()
            f.queue.append(_pending("a", 1000))
            f.queue.append(_pending("long", 1000, deferred=False, eligible=False))  # a P-only request
            self.assertEqual(f._x_idle_regrant(time.time() + 10), "none")
            self.assertEqual(len(f.queue), 2)
        self._run(go())

    def test_never_while_d_holds_work_or_without_a_deferred_request(self):
        async def go():
            with _Env("qwen27b"):
                f = _front()
            f.queue.append(_pending("a", 1000, deferred=False, eligible=True))
            self.assertEqual(f._x_idle_regrant(time.time() + 10), "none")
            f.queue[0].x_deferred = True
            f.groups["D"].outstanding["busy"] = time.time()
            self.assertEqual(f._x_idle_regrant(time.time() + 10), "none")
            f.groups["D"].outstanding.clear()
            f.admit_d = False  # the fairness bound switched: nothing moves
            self.assertEqual(f._x_idle_regrant(time.time() + 10), "none")
        self._run(go())

    def test_overrun_counted_only_for_a_busy_grant(self):
        with _Env("qwen27b"):
            f = _front()
        f._note_x_grant_realized("r", (4096, True), pt=6000, ct=1000)
        self.assertEqual(f.counters["x_busy_overrun"], 1)
        f._note_x_grant_realized("r", (4096, False), pt=9000, ct=0)
        f._note_x_grant_realized("r", None, pt=9000, ct=0)
        self.assertEqual(f.counters["x_busy_overrun"], 1)


if __name__ == "__main__":
    unittest.main()
