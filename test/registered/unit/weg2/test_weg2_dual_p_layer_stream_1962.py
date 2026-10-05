# SPDX-License-Identifier: Apache-2.0
"""#1962 P-LAYER-STREAM (dual P, PP0): weights out, KV in, staged by the prompt's level.

User order 05.10.: Dual-P must prefill 262144 tokens. 1959/1960: K0's pool is 3.33 GB against 5.998 GB
for level 266240 (11 FA layers), and with D running no static cut carries 262144. PP0 pauses whole
weight-chunk tags of its own P-private bytes, lends them to the card pool and streams those layers'
tensors per forward from pinned host images.

DANGER DIRECTIONS guarded here:
* default off: not armed, nothing built, the adapter's ``owns`` is False, graphs never forced eager,
  ``infeasible_cards`` and ``pp0_grant`` unchanged (FlipUnchanged1962 + the default-path tests);
* the outputs are IDENTICAL with a unit paused (the paused original is poisoned in the fake, so a
  forward that read it would differ), and identical again after the regain;
* a unit with a live block nobody knows, or a tensor without a layer index, is REFUSED at catalog
  time (it would be read at an unmapped address);
* streaming happens only when PP0's card is the ONLY short card, and only when the units cover the
  deficit (+ the staging ring once); D's committed bytes are never touched (P never presses D);
* the regain needs the ledger's reclaim (free covers it): when D grew into the loan, P keeps
  streaming and the loan stands; a refused resume re-lends and keeps the unit paused;
* while a unit is paused the adapter skips it for every caller but the streamer (the P sleep/wake
  legs are not idempotent, weight_updater.py #1285).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.weg2 import card_kv_ledger as K  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as S  # noqa: E402
from sglang.srt.weg2 import p_layer_stream as L  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ARM = {"SGLANG_WEG2_DUAL_P_LAYER_STREAM": "1", "SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}


def _toy(n=6, width=8, seed=0):
    torch.manual_seed(seed)
    m = torch.nn.Module()
    m.layers = torch.nn.ModuleList([torch.nn.Linear(width, width) for _ in range(n)])
    return m


def _fwd(m, x):
    for lay in m.layers:
        x = torch.tanh(lay(x))
    return x


class _FakeSaver:
    """pause = the original storage becomes garbage (an unmapped page would be read as anything);
    resume = the content comes back (TMS's own CPU backup)."""

    def __init__(self, units):
        self.units = {u.tag: u for u in units}
        self.saved = {}
        self.calls = []

    def pause(self, tag):
        self.calls.append(("pause", tag))
        u = self.units[tag]
        self.saved[tag] = [(t, t.data.clone()) for ts in u.tensors.values() for t in ts]
        for t, _c in self.saved[tag]:
            t.data.fill_(1e6)

    def resume(self, tag):
        self.calls.append(("resume", tag))
        for t, c in self.saved.pop(tag):
            t.data.copy_(c)


def _units(m, groups, nbytes=1000):
    """groups: {tag: [layer ids]} -> StreamUnits over the toy's layer tensors."""
    out = []
    for tag, ids in groups.items():
        out.append(L.StreamUnit(tag, nbytes, {i: [m.layers[i].weight, m.layers[i].bias] for i in ids}))
    return out


class _Base(CustomTestCase):
    def setUp(self):
        L.reset_for_tests()

    def tearDown(self):
        L.reset_for_tests()


class PlanByLevel(_Base):
    def test_nothing_when_it_fits(self):
        self.assertEqual(L.plan_units([("weights_5", 800)], [], 0, 200), [])
        self.assertEqual(L.plan_units([("weights_5", 800)], [], -5, 200), [])

    def test_staging_charged_once(self):
        u = [("weights_5", 800), ("weights_4", 800), ("weights_3", 800)]
        self.assertEqual(L.plan_units(u, [], 700, 200), ["weights_5", "weights_4"], "700 + ring 200 > 800")
        self.assertEqual(L.plan_units(u, ["weights_5"], 700, 200), ["weights_4"], "ring already paid")

    def test_uncoverable_pauses_nothing(self):
        self.assertEqual(L.plan_units([("weights_5", 800), ("weights_4", 800)], [], 1500, 200), [])

    def test_order_is_the_callers(self):
        u = [("weights_5", 800), ("weights_0", 800)]
        self.assertEqual(L.plan_units(u, [], 100, 0), ["weights_5"])

    def test_need_on_card0_same_rule_as_group_grant(self):
        st = {"step": 4096, "top": 266240, "bytes": [i * 10 for i in range(66)]}
        self.assertEqual(L.level_need_on_card0(st, 262145, 0), 650)     # 262145 -> 266240 -> index 65
        self.assertEqual(L.level_need_on_card0(st, 4096, 30), 0)        # covered


class Catalog(_Base):
    def _segs(self, tensors_by_tag, extra_live=None):
        """One segment per storage (real segments are disjoint; a CPU heap interleaves the toy's tensors)."""
        out = {}
        for tag, ts in tensors_by_tag.items():
            segs = []
            for t in ts:
                a, n = t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()
                segs.append({"address": a, "total_size": n,
                             "blocks": [{"address": a, "size": n, "state": "active_allocated"}]})
            if extra_live:
                segs.append({"address": extra_live["address"], "total_size": extra_live["size"],
                             "blocks": [extra_live]})
            out[tag] = segs
        return out

    def _cat(self, named, segs):
        # CPU tensors stand in for device tensors
        return L.catalog(named, segs, is_device=lambda t: True)

    def test_units_tail_first_with_their_layers(self):
        m = _toy()
        segs = self._segs({"weights_0": [m.layers[0].weight, m.layers[0].bias, m.layers[1].weight,
                                         m.layers[1].bias],
                           "weights_1": [m.layers[4].weight, m.layers[4].bias]})
        units, refused = self._cat(m.named_modules(), segs)
        self.assertEqual(refused, {})
        self.assertEqual([u.tag for u in units], ["weights_1", "weights_0"])
        self.assertEqual(units[1].layers, (0, 1))
        self.assertIs(units[0].tensors[4][0], m.layers[4].weight)

    def test_tensor_shared_by_two_layers_is_swapped_at_both(self):
        m = _toy()
        ws = torch.zeros(16)
        m.layers[1].ws = ws                                   # one workspace object, two layers' modules
        m.layers[2].ws = ws
        segs = self._segs({"weights_0": [m.layers[1].weight, m.layers[1].bias, ws]})
        units, refused = self._cat(m.named_modules(), segs)
        self.assertEqual(refused, {})
        self.assertTrue(any(t is ws for t in units[0].tensors[1]))
        self.assertTrue(any(t is ws for t in units[0].tensors[2]))

    def test_unknown_live_block_refuses_the_unit(self):
        m = _toy()
        ghost = torch.zeros(4)
        segs = self._segs({"weights_0": [m.layers[0].weight, m.layers[0].bias]},
                          extra_live={"address": ghost.untyped_storage().data_ptr(), "size": 16,
                                      "state": "active_allocated"})
        units, refused = self._cat(m.named_modules(), segs)
        self.assertEqual(units, [])
        self.assertIn("coverage", refused["weights_0"])

    def test_tensor_without_layer_index_refuses(self):
        m = _toy()
        m.stray = torch.nn.Parameter(torch.zeros(3))
        segs = self._segs({"weights_0": [m.layers[0].weight, m.stray]})
        units, refused = self._cat(m.named_modules(), segs)
        self.assertEqual(units, [])
        self.assertIn("without a layer index", refused["weights_0"])


class StreamSameOutputs(_Base):
    def _setup(self):
        m = _toy()
        x = torch.randn(3, 8)
        ref = _fwd(m, x).clone()
        units = _units(m, {"weights_1": [3, 4], "weights_0": [1]})
        saver = _FakeSaver(units)
        st = L.LayerStreamer(units, pause=saver.pause, resume=saver.resume, prefetch=1, device=None,
                             pin=False)
        st.install_hooks(m.layers, 0, len(m.layers))
        L.install(st)
        return m, x, ref, units, saver, st

    def test_poisoned_original_proves_the_swap(self):
        m, x, ref, units, saver, st = self._setup()
        self.assertFalse(L.force_eager())
        freed = st.stream_out(["weights_1"])
        self.assertEqual(freed, 1000)
        self.assertTrue(L.force_eager(), "a paused unit forces PP0 eager")
        self.assertTrue(float(m.layers[3].weight.abs().max()) >= 1e6, "the original is poisoned (unmapped)")
        for _ in range(3):
            torch.testing.assert_close(_fwd(m, x), ref)
        self.assertGreaterEqual(st.counters["swapped"], 6)
        self.assertTrue(float(m.layers[3].weight.abs().max()) >= 1e6, "after the forward the original is back")
        st.stream_out(["weights_0"])
        torch.testing.assert_close(_fwd(m, x), ref)

    def test_swap_under_inference_mode(self):
        m, x, ref, units, saver, st = self._setup()
        st.stream_out(["weights_1", "weights_0"])
        with torch.inference_mode():                         # the serving forward's mode
            out = _fwd(m, x)
        torch.testing.assert_close(out, ref)
        self.assertFalse(m.layers[3].weight.is_inference(), "the parameter never took an inference tensor")

    def test_regain_restores_and_unforces(self):
        m, x, ref, units, saver, st = self._setup()
        st.stream_out(["weights_1", "weights_0"])
        self.assertEqual(st.paused(), ("weights_1", "weights_0"))
        self.assertEqual(st.regain("weights_0"), 1000)
        self.assertEqual(st.regain("weights_1"), 1000)
        self.assertFalse(L.force_eager())
        self.assertEqual(st.lent(), 0)
        torch.testing.assert_close(_fwd(m, x), ref)
        self.assertEqual(st.counters["swapped"], 0)

    def test_pause_that_frees_nothing_is_undone(self):
        m, x, ref, units, saver, st = self._setup()
        reads = iter([100, 100])
        st._phys_free = lambda: next(reads)
        self.assertEqual(st.stream_out(["weights_1"]), 0)
        self.assertEqual(saver.calls, [("pause", "weights_1"), ("resume", "weights_1")])
        self.assertFalse(L.force_eager())
        torch.testing.assert_close(_fwd(m, x), ref)

    def test_layer_split_across_two_units_merges(self):
        m = _toy()
        x = torch.randn(3, 8)
        ref = _fwd(m, x).clone()
        units = [L.StreamUnit("weights_1", 1000, {3: [m.layers[3].weight]}),
                 L.StreamUnit("weights_0", 1000, {3: [m.layers[3].bias], 2: [m.layers[2].weight]})]
        saver = _FakeSaver(units)
        st = L.LayerStreamer(units, pause=saver.pause, resume=saver.resume, prefetch=2, device=None, pin=False)
        st.install_hooks(m.layers, 0, len(m.layers))
        st.stream_out(["weights_1"])                         # layer 3 staged with ONE row (the weight)
        torch.testing.assert_close(_fwd(m, x), ref)
        st.stream_out(["weights_0"])                         # now its bias too: the staged copy must be redone
        torch.testing.assert_close(_fwd(m, x), ref)
        st.regain("weights_1")
        torch.testing.assert_close(_fwd(m, x), ref, msg="the other unit's half of layer 3 still streams")
        self.assertEqual(st.paused(), ("weights_0",))
        st.regain("weights_0")
        torch.testing.assert_close(_fwd(m, x), ref)

    def test_adapter_skips_owned_tags_but_not_the_streamer(self):
        from sglang.srt.utils.torch_memory_saver_adapter import _p_layer_stream_owns

        m, x, ref, units, saver, st = self._setup()
        self.assertFalse(_p_layer_stream_owns("weights_1"), "nothing paused: the default path")
        st.stream_out(["weights_1"])
        self.assertTrue(_p_layer_stream_owns("weights_1"))
        self.assertFalse(_p_layer_stream_owns("weights_0"))
        L._ACTING[0] = True
        try:
            self.assertFalse(_p_layer_stream_owns("weights_1"), "the streamer itself passes")
        finally:
            L._ACTING[0] = False


def _ledger(path, budget, d_committed, p_committed=0):
    K.CardKvLedger(path, "D").contribute(budget, committed=d_committed)
    if p_committed:
        pl = K.CardKvLedger(path, "P")
        pl.join(budget)
        got, _ = pl.request(p_committed)
        assert got == p_committed


def _stage(path_ledger, bytes_level):
    return {"ledger": path_ledger, "step": 4096, "top": 266240, "bytes": [0] + [bytes_level] * 65}


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.origin_input_ids = [0] * 1000


class _Sched:
    def __init__(self):
        self.ps = type("ps", (), {"pp_rank": 0, "pp_size": 2})()
        self._weg2_store_held = {}


class _Actor:
    page = 64
    _committed = 0

    def __init__(self, ledger, streamer=None):
        self.ledger = ledger
        self.mapped_tokens = 0
        if streamer is not None:
            self.streamer = streamer

    def map_granted(self, lvl, charged=0):
        self.mapped = (lvl, charged)


class GrantThroughTheStreamer(_Base):
    """pp0_grant with REAL card ledgers + real group_grant: card 0 pool 3100, D holds 2000, need 4080."""

    def setUp(self):
        super().setUp()
        S._RETRY.clear()
        S._STAGE_CACHE.clear()
        S._reset_wait_log()
        self.d = tempfile.mkdtemp(prefix="g1962")
        self.led = [os.path.join(self.d, "wkv-%d" % i) for i in range(2)]
        _ledger(self.led[0], 3100, d_committed=2000)
        _ledger(self.led[1], 6000, d_committed=100)
        self.paths = [os.path.join(self.d, "wkvs-pp%d.json" % i) for i in range(2)]
        self._write(4080, 500)
        self.m = _toy()
        self.units = _units(self.m, {"weights_5": [5], "weights_4": [4], "weights_3": [3]}, nbytes=1200)
        self.saver = _FakeSaver(self.units)
        self.st = L.LayerStreamer(self.units, pause=self.saver.pause, resume=self.saver.resume, prefetch=1,
                                  device=None, pin=False)
        self.st.max_layer_bytes = 100                       # ring = (prefetch 1 + 2) x 100 = 300 B

    def _write(self, b0, b1):
        for p, st in zip(self.paths, (_stage(self.led[0], b0), _stage(self.led[1], b1))):
            with open(p, "w") as f:
                json.dump(st, f)
        S._STAGE_CACHE.clear()

    def _grant(self, actor):
        patches = [
            mock.patch.object(S, "_actor", lambda s: actor),
            mock.patch.object(S, "stage_file", lambda tag, r, root="/dev/shm": self.paths[r]),
            mock.patch.object(S, "_dual_layout_env", lambda: True),
            mock.patch.object(S, "_retry_ms", lambda: 0),
            mock.patch.object(S, "return_untold_grant", lambda *a, **k: None),
            mock.patch.object(S, "live_grant_tokens", lambda *a, **k: 0),
            mock.patch("sglang.srt.weg2.dual_grant_wait.grant_held_by_d", lambda *a, **k: None),
        ]
        for p in patches:
            p.start()
        try:
            return S.pp0_grant(_Sched(), _Req("weg2-0-1"))
        finally:
            for p in patches:
                p.stop()

    def test_without_streamer_the_old_wait(self):
        actor = _Actor(K.CardKvLedger(self.led[0], "P"))
        self.assertEqual(self._grant(actor), 0)
        self.assertEqual(self.saver.calls, [])

    def test_streams_exactly_the_deficit_and_grants(self):
        actor = _Actor(K.CardKvLedger(self.led[0], "P"), self.st)
        lvl = self._grant(actor)
        self.assertEqual(lvl, 4096)
        # deficit 4080 - 1100 = 2980 + ring 300 -> 3 units of 1200
        self.assertEqual(self.st.paused(), ("weights_5", "weights_4", "weights_3"))
        self.assertEqual(actor._stream_lent, 3300, "3 x 1200 freed, the ring (300) the prefetch keeps is not lent")
        self.assertEqual(self.st.loan, {"weights_5": 900, "weights_4": 1200, "weights_3": 1200})
        st = K.peek(self.led[0])
        self.assertEqual(st.committed["D"], 2000, "D untouched: P never presses D")
        self.assertEqual(st.committed["P"], 4080)

    def test_other_card_short_streams_nothing(self):
        self._write(4080, 7000)                              # card 1 short as well
        actor = _Actor(K.CardKvLedger(self.led[0], "P"), self.st)
        self.assertEqual(self._grant(actor), 0)
        self.assertEqual(self.saver.calls, [])

    def test_uncoverable_level_streams_nothing(self):
        self._write(9000, 500)                               # deficit 7900 > 3 x 1200
        actor = _Actor(K.CardKvLedger(self.led[0], "P"), self.st)
        self.assertEqual(self._grant(actor), 0)
        self.assertEqual(self.saver.calls, [])

    def test_infeasible_counts_the_stream_room_only_when_on(self):
        st = [_stage(self.led[0], 4080), _stage(self.led[1], 500)]
        self.assertEqual([c[0] for c in S.infeasible_cards(st, {}, 1064)], [0])
        self.assertEqual(S.infeasible_cards(st, {}, 1064, self.st.room()), [], "3600 - ring 300 covers it")

    def test_regain_at_idle_only_through_reclaim(self):
        actor = _Actor(K.CardKvLedger(self.led[0], "P"), self.st)
        self.assertEqual(self._grant(actor), 4096)
        led = K.CardKvLedger(self.led[0], "P")
        self.assertEqual(L.regain_at_idle(actor), 0, "P still maps KV")  # mapped_tokens is the actor's view
        actor.mapped_tokens = 0
        # P's grant still committed -> free < loan: held, nothing resumed
        self.assertEqual(L.regain_at_idle(actor), 0)
        self.assertEqual(len(self.st.paused()), 3)
        led.release(4080)                                    # the request ended, P released its KV
        got = L.regain_at_idle(actor)
        self.assertEqual(got, 3300)
        self.assertEqual(self.st.paused(), ())
        self.assertEqual(actor._stream_lent, 0)
        self.assertEqual(K.peek(self.led[0]).budget, 3100, "the loan is back out of the pool")

    def test_on_idle_regains_only_without_a_grant_waiter(self):
        actor = _Actor(K.CardKvLedger(self.led[0], "P"), self.st)
        self._grant(actor)
        K.CardKvLedger(self.led[0], "P").release(4080)
        waiter = type("W", (), {"_dual_kv_wait": True, "rid": "weg2-0-2"})()
        sched = _Sched()
        sched._weg2_store_held = {"w": waiter}
        with mock.patch.object(S, "_actor", lambda s: actor), mock.patch.object(S, "phys_free_bytes", lambda: None):
            self.assertEqual(S.on_idle(sched), 0)
            self.assertEqual(len(self.st.paused()), 3, "a waiter would pause them again on its next attempt")
            sched._weg2_store_held = {}
            S.on_idle(sched)
        self.assertEqual(self.st.paused(), ())

    def test_d_grew_into_the_loan_p_keeps_streaming(self):
        actor = _Actor(K.CardKvLedger(self.led[0], "P"), self.st)
        self._grant(actor)
        K.CardKvLedger(self.led[0], "P").release(4080)
        dl = K.CardKvLedger(self.led[0], "D")
        got, _ = dl.request(3000)                            # D legally takes the loan
        self.assertEqual(got, 3000)
        self.assertEqual(L.regain_at_idle(actor), 1200, "only what is free comes back, newest first")
        self.assertEqual(self.st.paused(), ("weights_5", "weights_4"))

    def test_refused_resume_relends(self):
        actor = _Actor(K.CardKvLedger(self.led[0], "P"), self.st)
        self._grant(actor)
        K.CardKvLedger(self.led[0], "P").release(4080)
        budget = K.peek(self.led[0]).budget

        def boom(tag):
            raise RuntimeError("W119")

        self.st._resume = boom
        with self.assertRaises(RuntimeError):
            L.regain_at_idle(actor)
        self.assertEqual(K.peek(self.led[0]).budget, budget, "the loan stands again")
        self.assertEqual(len(self.st.paused()), 3)


class FlipUnchanged1962(_Base):
    """The flip form (no dual layout, no group) and the dual form with the switch OFF: inert."""

    def test_not_armed_without_every_key(self):
        self.assertFalse(L.armed({}))
        self.assertFalse(L.armed({"SGLANG_WEG2_DUAL_P_LAYER_STREAM": "1"}))
        self.assertFalse(L.armed({"SGLANG_WEG2_DUAL_P_LAYER_STREAM": "1", "SGLANG_WEG2_DUAL_LAYOUT": "1"}))
        self.assertFalse(L.armed({"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}))
        self.assertFalse(L.armed(dict(ARM, SGLANG_WEG2_GROUP="D")))
        self.assertTrue(L.armed(ARM))
        self.assertTrue(L.armed(dict(ARM, SGLANG_WEG2_DUAL_P_LAYER_STREAM="yes")), "EnvBool's own set")
        self.assertFalse(L.armed(dict(ARM, SGLANG_WEG2_DUAL_P_LAYER_STREAM="on")), "EnvBool rejects 'on'")

    def test_build_refuses_unarmed_and_non_pp0(self):
        runner = type("R", (), {"pp_rank": 0})()
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ARM:
                os.environ.pop(k, None)
            self.assertIsNone(L.build_for_runner(runner))
        with mock.patch.dict(os.environ, ARM, clear=False):
            self.assertIsNone(L.build_for_runner(type("R", (), {"pp_rank": 1})()))

    def test_inert_globals(self):
        from sglang.srt.utils.torch_memory_saver_adapter import _p_layer_stream_owns

        self.assertFalse(L.force_eager())
        self.assertFalse(_p_layer_stream_owns("weights_0"))
        self.assertFalse(L.owns("weights_0"))
        self.assertIsNone(L.active())

    def test_on_idle_and_grant_helpers_without_streamer(self):
        actor = type("A", (), {"mapped_tokens": 0})()
        self.assertEqual(S._stream_room(actor), 0)
        self.assertEqual(L.regain_at_idle(actor), 0)
        self.assertEqual(L.try_stream_for_grant(actor, [{"step": 4096}], 1, 0, lambda p: None), 0)

    def test_env_declared_default_off(self):
        from sglang.srt.environ import envs

        self.assertFalse(envs.SGLANG_WEG2_DUAL_P_LAYER_STREAM.get())
        self.assertEqual(envs.SGLANG_WEG2_DUAL_P_LAYER_STREAM_PREFETCH.get(), 2)


class GraphsEagerWhilePaused(_Base):
    """Both PP0 graph runners refuse a replay while a unit is paused (graphs read the paused VA)."""

    def _prefill(self):
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner

        notes = []
        fake = type("R", (), {})()
        fake._is_full_backend = True
        fake._note_eager = lambda reason, fb: notes.append(reason)
        fake._full_graph_ineligible_reason = lambda fb: None
        return PrefillCudaGraphRunner.can_run_graph, fake, notes

    def test_prefill_runner(self):
        fn, fake, notes = self._prefill()
        self.assertTrue(fn(fake, object()), "nothing paused: the old verdict")
        self.assertEqual(notes, [])
        L._OWNED.add("weights_5")
        self.assertFalse(fn(fake, object()))
        self.assertEqual(notes, ["layer_stream"])
        L._OWNED.clear()
        self.assertTrue(fn(fake, object()), "after the regain the graphs replay again")

    def test_decode_runner(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner

        L._OWNED.add("weights_5")
        self.assertFalse(DecodeCudaGraphRunner.can_run_graph(object(), object()))


class ReviewMutants1971(_Base):
    """The four mutants review 1971 found undetected, plus F2/F4/F5/F6b/F7 guards."""

    def _st(self, m, groups, prefetch=1, saver_cls=None, **kw):
        units = _units(m, groups)
        saver = (saver_cls or _FakeSaver)(units)
        st = L.LayerStreamer(units, pause=saver.pause, resume=saver.resume, prefetch=prefetch, device=None,
                             pin=False, **kw)
        st.install_hooks(m.layers, 0, len(m.layers))
        return st, saver, units

    def test_m1_post_layer_points_back_at_the_original(self):
        m = _toy()
        ptr = m.layers[3].weight.data_ptr()
        st, _s, _u = self._st(m, {"weights_1": [3]})
        st.stream_out(["weights_1"])
        _fwd(m, torch.randn(2, 8))
        self.assertEqual(m.layers[3].weight.data_ptr(), ptr, "after the layer the tensor is its paused original")

    def test_m2_m3_ring_is_bounded_and_sized(self):
        m = _toy(n=12)
        st, _s, _u = self._st(m, {"weights_1": list(range(1, 12))}, prefetch=2)
        st.stream_out(["weights_1"])
        for _ in range(3):
            _fwd(m, torch.randn(2, 8))
        self.assertLessEqual(st.counters["peak_live_sets"], st.prefetch + 2,
                             "staged + executing + in flight never exceed the ring")
        self.assertEqual(st.staging_bytes(), (st.prefetch + 2) * st.max_layer_bytes)
        self.assertEqual(L._rounded(10), 512)
        self.assertEqual(L._rounded((1 << 20) + 1), 2 << 20)

    def test_m4_regain_restores_content_without_the_savers_backup(self):
        class _NoBackup(_FakeSaver):
            def resume(self, tag):                      # a weights-resident profile: pages come back undefined
                self.calls.append(("resume", tag))
                self.saved.pop(tag)

        m = _toy()
        x = torch.randn(3, 8)
        ref = _fwd(m, x).clone()
        st, _s, _u = self._st(m, {"weights_1": [3, 4]}, saver_cls=_NoBackup)
        st.stream_out(["weights_1"])
        st.regain("weights_1")
        torch.testing.assert_close(_fwd(m, x), ref, msg="the host image was copied back")

    def test_m5_stage_file_lent_excludes_the_stream_loan(self):
        d = tempfile.mkdtemp(prefix="s1962")
        actor = type("A", (), {})()
        actor.ledger = type("Led", (), {"path": os.path.join(d, "led")})()
        actor.step, actor.top = 4096, 8192
        actor.table = lambda: [0, 1, 2]
        actor._stream_lent = 999
        with mock.patch.object(S, "stage_file", lambda tag, r, root="/dev/shm": os.path.join(d, "st.json")):
            path = S.publish_stage(actor, "t", 0)
        with open(path) as f:
            self.assertEqual(json.load(f)["lent"], 0, "the front's wake gate never waits for the stream loan")

    def test_f4_measures_with_the_tags_own_mapped_bytes(self):
        m = _toy()
        mapped = {"weights_1": 800}

        class _S(_FakeSaver):
            def pause(self, tag):
                super().pause(tag)
                mapped[tag] = 0

        st, _s, _u = self._st(m, {"weights_1": [3]}, saver_cls=_S, tag_mapped=lambda t: mapped.get(t),
                              phys_free=lambda: 10 ** 12)          # a D release would inflate this one
        self.assertEqual(st.stream_out(["weights_1"]), 800)

    def test_f6b_no_cascade_for_the_same_level(self):
        m = _toy()
        st, _s, _u = self._st(m, {"weights_2": [2], "weights_1": [1], "weights_0": [0]})
        st.max_layer_bytes = 0
        actor = type("A", (), {})()
        actor.streamer = st
        lent = []
        actor.ledger = type("Led", (), {"lend": lambda self, n: lent.append(n)})()
        stages = [{"ledger": "a", "step": 4096, "top": 8192, "bytes": [0, 1000, 1000]}]   # deficit 900: one unit
        free = {"a": 100}
        peek = lambda p: type("St", (), {"free": free[p]})()
        self.assertGreater(L.try_stream_for_grant(actor, stages, 4096, 0, peek), 0)
        n_paused = len(st.paused())
        self.assertEqual(n_paused, 1)
        # D took the loan: still short at the same level -> no further pause
        self.assertEqual(L.try_stream_for_grant(actor, stages, 4096, 0, peek), 0)
        self.assertEqual(len(st.paused()), n_paused)

    def test_f5_forbidden_storage_refuses_the_unit(self):
        m = _toy()
        segs = Catalog()._segs({"weights_0": [m.layers[0].weight, m.layers[0].bias]})
        forb = {m.layers[0].bias.untyped_storage().data_ptr(): "hull buffer x (sleep static-state export)"}
        units, refused = L.catalog(m.named_modules(), segs, is_device=lambda t: True, forbidden=forb)
        self.assertEqual(units, [])
        self.assertIn("outside the forward", refused["weights_0"])

    def test_f7_python_holder_outside_the_modules_is_found(self):
        m = _toy()
        units = _units(m, {"weights_0": [0]})
        stray = m.layers[0].weight.data.t()           # a cached view nobody swaps
        bases = {"weights_0": [m.layers[0].weight.untyped_storage().data_ptr()]}
        out = L.uncatalogued_holders(units, bases, [stray, m.layers[0].weight], is_device=lambda t: True)
        self.assertEqual(out, {"weights_0": 1})
