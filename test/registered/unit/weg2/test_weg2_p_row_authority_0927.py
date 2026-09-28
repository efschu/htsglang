# SPDX-License-Identifier: Apache-2.0
"""Fix B: #631 row authority re-armed on group P (weg2/p_row_authority.py).

fb3631c434 (#1233 S0) pinned pp_flip_counters / pp_chain_receiver to None (un-weave, not a crash);
this re-builds both under SGLANG_WEG2_P_ROW_AUTHORITY (DEFAULT OFF), keeps the told carrier armed,
puts every PP0 term (#1066/#1175, #1039, #794) behind its own switch (default off), refuses a
half-armed start by name, bounds the chain receive, and pins the pacemaker argument (a follower
that blocks in the chain recv misaligns its slots -- 631row5).
"""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import p_row_authority as PR  # noqa: E402


def _env(**kv):
    return mock.patch.dict(os.environ, {k: str(v) for k, v in kv.items()})


def _sched(pp_rank=1, pp_size=3, counters=None, receiver=None):
    ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size, tp_size=1, attn_tp_rank=0,
                               attn_cp_rank=0, attn_dp_rank=0, attn_cp_size=1, attn_tp_size=1)
    return types.SimpleNamespace(ps=ps, pp_flip_counters=counters, pp_chain_receiver=receiver,
                                 world_group=types.SimpleNamespace(cpu_group="g"))


class Switches(unittest.TestCase):
    def test_default_off_everything(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in [PR.ENV, *PR.TERM_ENVS.values()]:
                os.environ.pop(k, None)
            self.assertFalse(PR.enabled())
            self.assertFalse(any(PR.term_on(t) for t in PR.TERM_ENVS))
            self.assertFalse(PR.applies(_sched()))

    def test_profile_default_followed_and_explicit_env_wins(self):
        from sglang.srt.weg2 import form as F

        with mock.patch.object(F, "profile_switch_default", lambda name, fb, env=None: name == PR.ENV):
            self.assertTrue(PR.enabled({}))
            self.assertFalse(PR.enabled({PR.ENV: "0"}), "an explicit env wins")


def _form_env(profile):
    """The published weg2 form of a 27B / NF boot (as test_27b_park_immediate builds it)."""
    from sglang.srt.weg2 import form as F

    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == F.PROFILE_QWEN27B
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return F.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                      flip="family", vision="off", profile=profile, model="m").env_value()


class RegistryDefaultPerModel(unittest.TestCase):
    """Fix-B registry flip (operator 28.09.): the 27B row arms Fix B by DEFAULT (no profile/
    CONTAINER_ENV switch needed); the NF row stays off unless NF sets the switch itself. Only the
    main switch -- the PP0 terms stay off, exactly the metal proof form (27b-row-authority.env)."""

    def test_the_rows(self):
        from sglang.srt.weg2 import form as F

        self.assertIs(F.PROFILES[F.PROFILE_QWEN27B].p_row_authority, True)
        self.assertIs(F.PROFILES[F.PROFILE_NEXTFLASH].p_row_authority, False)
        self.assertIs(F.PROFILE_SWITCH_DEFAULTS[F.PROFILE_QWEN27B][PR.ENV], True)
        self.assertIs(F.PROFILE_SWITCH_DEFAULTS[F.PROFILE_NEXTFLASH][PR.ENV], False)
        self.assertEqual(set(F.PROFILES), {F.PROFILE_QWEN27B, F.PROFILE_NEXTFLASH},
                         "a new registry row must state its Fix-B default here")

    def test_enabled_follows_the_published_form_per_model(self):
        from sglang.srt.weg2 import form as F

        e27 = {F.FORM_ENV: _form_env(F.PROFILE_QWEN27B)}
        enf = {F.FORM_ENV: _form_env(F.PROFILE_NEXTFLASH)}
        self.assertTrue(PR.enabled(e27), "27B: on without any switch")
        self.assertFalse(PR.enabled(enf), "NF: unchanged, off")
        self.assertFalse(PR.enabled({}), "no form (desk/upstream): the code default, off")
        self.assertFalse(PR.enabled({**e27, PR.ENV: "0"}), "27B: an explicit 0 still turns it off")
        self.assertTrue(PR.enabled({**enf, PR.ENV: "1"}), "NF: switches it on itself")

    def test_27b_default_arms_only_the_main_switch(self):
        from sglang.srt.weg2 import form as F

        e27 = {F.FORM_ENV: _form_env(F.PROFILE_QWEN27B)}
        for t in PR.TERM_ENVS:
            self.assertFalse(PR.term_on(t, e27), t)
        with mock.patch.dict(os.environ, e27, clear=False):
            for k in [PR.ENV, *PR.TERM_ENVS.values()]:
                os.environ.pop(k, None)
            self.assertTrue(PR.applies(_sched(pp_size=3)), "group P (pp>1) re-arms the row form")
            self.assertFalse(PR.applies(_sched(pp_rank=0, pp_size=1)), "group D (pp=1) is untouched")

    def test_stall_bound_finite(self):
        with _env(**{PR.STALL_ENV: ""}):
            self.assertEqual(PR.stall_s(), 120.0)
        with _env(**{PR.STALL_ENV: "0"}):
            self.assertEqual(PR.stall_s(), 120.0, "0 would be the unbounded receive")


class Builders(unittest.TestCase):
    def test_counters_built_and_swept_only_when_on(self):
        import tempfile

        from sglang.srt.managers import phase_flip_presence as pres

        with tempfile.TemporaryDirectory() as d, mock.patch.object(pres, "DEFAULT_PRESENCE_DIR", d):
            with _env(**{PR.ENV: 0}):
                self.assertIsNone(PR.build_counters(_sched()))
            with _env(**{PR.ENV: 1}):
                c = PR.build_counters(_sched(pp_rank=0))
            self.assertIsNotNone(c)
            self.assertEqual((c.n_ranks, c.rank), (3, 0))

    def test_chain_receiver_gets_the_finite_stall_and_the_counter_hook(self):
        seen = {}

        class FakeRecv:
            def __init__(self, **kw):
                seen.update(kw)

        counters = types.SimpleNamespace(bump_consumed=lambda ch: seen.setdefault("bumped", ch))
        with _env(**{PR.ENV: 1, PR.STALL_ENV: 90}), \
             mock.patch("sglang.srt.managers.pp_chain_receiver.PpChainReceiver", FakeRecv):
            self.assertIsNone(PR.build_chain_receiver(_sched(pp_rank=0), counters), "PP0 has no upstream")
            r = PR.build_chain_receiver(_sched(pp_rank=2), counters)
        self.assertIsInstance(r, FakeRecv)
        self.assertEqual((seen["src"], seen["dst"], seen["stall_timeout_s"]), (1, 2, 90.0))
        seen["on_consumed"](1)
        self.assertIn("bumped", seen)


class StartCheck(unittest.TestCase):
    def test_half_armed_follower_is_refused_by_name(self):
        s = _sched(pp_rank=1, counters=object(), receiver=None)
        with _env(**{PR.ENV: 1}):
            with self.assertRaisesRegex(RuntimeError, "W-P-ROW Weg2RowAuthorityIncomplete.*pp_chain_receiver"):
                PR.start_check(s)
        s = _sched(pp_rank=0, counters=None)
        with _env(**{PR.ENV: 1}):
            with self.assertRaisesRegex(RuntimeError, "pp_flip_counters"):
                PR.start_check(s)

    def test_complete_arms_and_off_is_silent(self):
        s = _sched(pp_rank=1, counters=object(), receiver=object())
        with _env(**{PR.ENV: 1}):
            PR.start_check(s)
        self.assertTrue(getattr(s, PR.ROW_ONLY_ATTR))
        s2 = _sched(pp_rank=1)
        with _env(**{PR.ENV: 0}):
            PR.start_check(s2)
        self.assertFalse(hasattr(s2, PR.ROW_ONLY_ATTR))

    def test_scheduler_wiring(self):
        from sglang.srt.managers import scheduler as S

        src = open(S.__file__).read()
        i = src.index("def init_request_receiver(self)")
        blk = src[i:i + 7000]
        a = blk.index("_prow.build_counters(self)")
        b = blk.index("_prow.build_chain_receiver(self, self.pp_flip_counters)")
        c = blk.index("_prow.start_check(self)")
        self.assertLess(a, b)
        self.assertLess(b, c)
        self.assertIn("chain_receiver=self.pp_chain_receiver", blk)


class CarrierSplit(unittest.TestCase):
    def _row_sched(self):
        s = _sched(pp_rank=0, counters=object(), receiver=None)
        setattr(s, PR.ROW_ONLY_ATTR, True)
        s.enable_hicache_storage = True
        return s

    def test_no_term_is_false_and_each_term_answers_its_switch(self):
        from sglang.srt.managers.pp_admission_congruence import pp_row_carrier_present as C

        s = self._row_sched()
        with _env(**{PR.ENV: 1}):
            self.assertFalse(C(s))
            for t, e in PR.TERM_ENVS.items():
                self.assertFalse(C(s, term=t))
                with _env(**{e: 1}):
                    self.assertTrue(C(s, term=t))

    def test_old_form_keeps_the_counter_answer(self):
        from sglang.srt.managers.pp_admission_congruence import pp_row_carrier_present as C

        self.assertFalse(C(_sched(counters=None)))
        self.assertTrue(C(_sched(counters=object())), "flip-era answer unchanged off the row form")

    def test_told_stays_armed_on_the_row_form(self):
        from sglang.srt.managers import weg2_store_told as T

        s = self._row_sched()
        with _env(**{PR.ENV: 1}):
            self.assertTrue(T.armed(s))

    def test_the_three_pp0_sites_pass_their_term(self):
        from sglang.srt.managers import scheduler as S

        src = open(S.__file__).read()
        for term in ("corridor", "withhold", "floor_clamp"):
            self.assertIn(f'pp_row_carrier_present(self, term="{term}")', src)


class Pacemaker(unittest.TestCase):
    """The loop-order argument, modelled: PP0 per pass posts its chain message at the TOP
    and its proxy frame for slot k AFTER planning. A follower that blocks in the chain recv
    each iteration visits slot k before frame k exists, skips it, and from then on the
    head frame always names an earlier slot (631row5 class). With the counter-gated chain
    (the receiver) the follower keeps cycling and takes each frame at its own slot."""

    SLOTS = 3

    def _run(self, gated: bool, passes: int = 30):
        frames = []          # posted, FIFO of slot ids
        executed = []
        pp0_pass = 0
        slot = 0
        chain_msgs = 0       # posted chain messages
        consumed_chain = 0
        steps = 0
        while pp0_pass < passes and steps < 10 * passes:
            steps += 1
            # PP0: top of pass -> chain message; plan; frame (slot = pass % SLOTS)
            chain_msgs += 1
            # follower iteration
            if not gated:
                consumed_chain += 1          # blocking recv returns this pass's message...
                head_ready = bool(frames)    # ...before PP0 posted this pass's frame
            frames.append(pp0_pass % self.SLOTS)
            pp0_pass += 1
            if gated:
                consumed_chain = chain_msgs  # counter-gated: probe after the frame landed
                head_ready = bool(frames)
            if head_ready and frames[0] == slot:
                executed.append(frames.pop(0))
            slot = (slot + 1) % self.SLOTS
        return executed

    def test_counter_gated_follower_executes_every_frame_in_order(self):
        ex = self._run(gated=True)
        self.assertEqual(ex, [i % self.SLOTS for i in range(30)])

    def test_mutant_blocking_chain_misaligns(self):
        ex = self._run(gated=False)
        self.assertLess(len(ex), 30, "the blocking-chain follower falls behind (crawl mutant)")


class Cost(unittest.TestCase):
    def test_cost_line(self):
        s = _sched()
        s._pp_row_probe_stats = {"calls": 10, "drained": 4, "delivered": 4, "quiet": 6, "head_other": 0}
        with mock.patch.object(PR, "COST_EVERY_S", 0.0), self.assertLogs(PR.logger, level="INFO") as cap:
            for ms in (1.0, 2.0, 9.0):
                PR.note_plan(s, ms)
        self.assertIn("P-ROW-COST pp_rank=1", cap.output[0])
        self.assertIn("plan_ms_p50=", cap.output[0])
        self.assertIn("delivered=4", cap.output[0])

    def test_hook_in_the_follower_plan_branch(self):
        from sglang.srt.managers import scheduler_pp_mixin as M

        src = open(M.__file__).read()
        i = src.index("elif _pre_proxy is not None:")
        blk = src[i:i + 3000]
        self.assertIn("_prow_t0 = time.perf_counter()", blk)
        self.assertIn("_prow.note_plan(self, (time.perf_counter() - _prow_t0) * 1000.0)", blk)


if __name__ == "__main__":
    unittest.main()
