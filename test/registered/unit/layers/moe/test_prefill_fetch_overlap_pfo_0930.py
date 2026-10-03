"""PFO (#287 Hebel A, 30.09.): expert fetch overlapped with compute in the
expert-major multi-wave prefill.

DER BEFUND (y4k/y4l P-Logs, PP0 16k-Chunk): FWD-TIMING-PREFILL ``moe_fetch``
(der Compute-Strom steht) == MOE-OFFLOAD-TIMING-PREFILL ``fetch_ms`` (die
Kopier-Events), z. B. 1453.5 == 1453.5 ms -- heute ueberlappt nichts; 279 Wellen,
5859-7592 Spill-Experten, 1.13-1.48 s Kopie gegen apply 0.42-0.47 s.

Gepinnt (CPU, echter Planer, echter Wellen-Loop, Spielzeug-apply, das die
Gewichte AUS DEN SLOTS liest -- ein falscher Slot gibt ein falsches Ergebnis):
* Schalter an: Ergebnis bitgleich zum seriellen Pfad und zur echten MoE;
  die Spill-Wellen sind auf <= scratch // 2 neu geschnitten (Wellenzahl ~x2),
  Welle j liest Haelfte (j-1) % 2;
* die Reihenfolge der Stroeme: die Kopie von Welle w+1 wird VOR dem apply von
  Welle w ausgegeben; ab w >= 3 wartet sie auf das apply-Event von w-2 (der
  letzte Leser ihrer Haelfte), davor auf den Forward-Strom; der Forward wartet
  vor jedem apply auf das Kopier-Event seiner Welle -- nur Events;
* nicht genommen: Schalter aus, stream-partials, H107-Slot-Plaene, eine
  Spill-Welle, scratch < 2; der Einwellen-/Decode-Pfad fragt PFO nie;
* ``_fetch`` ohne ``stream`` bleibt der alte Aufruf (gibt None, gleiche Kopien).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import types
import unittest
from typing import NamedTuple
from unittest import mock

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_offload as eo
from sglang.test.test_utils import CustomTestCase

E, R, S, K, H = 48, 8, 6, 4, 3


class _Topk(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: object = None


class _Disp(NamedTuple):
    hidden_states: torch.Tensor
    topk_output: _Topk
    hidden_states_scale: object = None


class _Comb(NamedTuple):
    hidden_states: torch.Tensor


def _value(e):
    return float((e * 7) % 23 + 1)


def _cache(scratch=S):
    c = object.__new__(eo.MoEExpertOffloadCache)
    c.layer = types.SimpleNamespace(layer_id=7, moe_runner_config=types.SimpleNamespace(
        routed_scaling_factor=1.0))
    c.num_local_experts = E
    c.resident_count = R
    c.scratch = scratch
    c.planner = eo.ExpertResidencyPlanner(num_local_experts=E, resident_count=R, scratch=scratch)
    res = torch.zeros(R + scratch, 1, dtype=torch.float64)
    for e in range(R):
        res[e, 0] = _value(e)
    c._resident = {"w": res}
    c._pinned = {"w": torch.tensor([[_value(e)] for e in range(R, E)], dtype=torch.float64)}
    c._stream = None
    c._cold_tier = None
    c._spill_pool_index = None
    c._scratch_holds = {}
    c._side_copies_in_flight = False
    c._remote_ids = frozenset()
    c._nan_trace = None
    return c


def _apply(cache):
    def apply(sub):
        slots = sub.topk_output.topk_ids.long()
        w = sub.topk_output.topk_weights
        per = cache._resident["w"][slots.clamp(min=0), 0] * w  # [n, 1]
        return _Comb(hidden_states=sub.hidden_states * per)
    return apply


def _combine(partials, out, rsf):
    out.copy_(partials.sum(dim=1) * (1.0 if rsf is None else rsf))
    return out


def _case(T, seed):
    rng = np.random.default_rng(seed)
    ids = np.argsort(rng.random((T, E)), axis=1)[:, :K].astype(np.int64)
    ids[::5, -1] = -1
    x = torch.from_numpy(rng.random((T, H)) + 0.5)
    w = torch.from_numpy(rng.random((T, K)))
    disp = _Disp(hidden_states=x, topk_output=_Topk(topk_weights=w, topk_ids=torch.from_numpy(ids)))
    true = torch.tensor([[(_value(int(e)) if e >= 0 else 0.0) for e in row] for row in ids],
                        dtype=torch.float64)
    want = (x.unsqueeze(1) * (true * w).unsqueeze(-1)).sum(1)
    return ids, disp, want


def _run(cache, ids, disp):
    flat = ids.reshape(-1)
    ru, sw = eo.plan_expert_waves_np(ids, cache.resident_count, cache.scratch, None, num_experts=E)
    return cache._run_waves_expert_major(disp, _apply(cache), flat, ru, sw), sw


class TestBitIdentical(CustomTestCase):
    def setUp(self):
        p = mock.patch.object(eo, "combine_topk_partials", _combine)
        p.start()
        self.addCleanup(p.stop)

    def test_overlap_equals_serial_and_the_true_moe(self):
        for T, seed in ((40, 1), (64, 2), (9, 3)):
            ids, disp, want = _case(T, seed)
            with envs.SGLANG_WEG2_ENABLE_PREFILL_FETCH_OVERLAP.override(False):
                serial, sw = _run(_cache(), ids, disp)
            fetched = []
            c = _cache()
            real_fetch = c._fetch

            def spy(plan, join=True, stream=None, after=None, _f=real_fetch):
                fetched.append([s for _e, s in plan])
                return _f(plan, join=join, stream=stream, after=after)

            c._fetch = spy
            with envs.SGLANG_WEG2_ENABLE_PREFILL_FETCH_OVERLAP.override(True):
                over, _ = _run(c, ids, disp)
            self.assertTrue(torch.equal(over.hidden_states, serial.hidden_states), (T, seed))
            torch.testing.assert_close(serial.hidden_states, want, rtol=0, atol=1e-12)
            # re-chunked to <= S // 2, and each spill wave reads its half
            spill = [f for f in fetched if f]
            self.assertTrue(all(len(f) <= S // 2 for f in spill))
            self.assertGreater(len(spill), len(sw))
            for j, slots in enumerate(spill, start=1):
                lo = R + ((j - 1) % 2) * (S // 2)
                self.assertTrue(all(lo <= s < lo + S // 2 for s in slots), (j, slots))

    def test_not_taken_outside_its_case(self):
        c = _cache()
        with envs.SGLANG_WEG2_ENABLE_PREFILL_FETCH_OVERLAP.override(False):
            self.assertIsNone(eo.prefill_fetch_overlap_form(c, None, 5))
        with envs.SGLANG_WEG2_ENABLE_PREFILL_FETCH_OVERLAP.override(True):
            self.assertIsNotNone(eo.prefill_fetch_overlap_form(c, None, 5))
            self.assertIsNone(eo.prefill_fetch_overlap_form(c, {0: None}, 5))   # H107 slot plans
            self.assertIsNone(eo.prefill_fetch_overlap_form(_cache(scratch=1), None, 5))
            self.assertIsNone(eo.prefill_fetch_overlap_form(c, None, 0))
            with mock.patch.object(eo, "partials_mode", lambda: "stream"):
                self.assertIsNone(eo.prefill_fetch_overlap_form(c, None, 5))

    def test_switch_default_off(self):
        self.assertFalse(envs.SGLANG_WEG2_ENABLE_PREFILL_FETCH_OVERLAP.get())


class TestStreamOrder(CustomTestCase):
    """The event protocol, recorded on stand-in streams/events."""

    def test_copy_ahead_of_apply_and_war_by_event(self):
        log = []

        class _Ev:
            def __init__(self, name):
                self.name = name

            def record(self, stream):
                log.append(("record", self.name))

        names = iter(range(1000))

        def event():
            return _Ev("apply%d" % next(names))

        class _Cur:
            def wait_event(self, ev):
                log.append(("forward_waits", ev))

        def resolve(needed):
            spill = [e for e in needed if e >= R]
            slot = {e: e for e in needed if e < R}
            slot.update({e: R + i for i, e in enumerate(spill)})
            return slot, [(e, R + i) for i, e in enumerate(spill)]

        fake_cache = types.SimpleNamespace(planner=types.SimpleNamespace(resident_count=R, resolve=resolve))

        def fetch(plan, join=True, stream=None, after=None):
            if not plan:
                return None
            n = len([x for x in log if x[0] == "copy"]) + 1
            log.append(("copy", [s for _e, s in plan], after.name if after is not None else "forward"))
            return "copied%d" % n

        fake_cache._fetch = fetch
        pfo = eo.PrefillFetchOverlap(stream="prefetch", half=2)
        waves = [[0, 1], [10, 11], [12, 13], [14, 15], [16]]   # resident wave, then 4 spill waves
        pfo.begin(fake_cache, waves)
        with mock.patch.object(torch.cuda, "Event", side_effect=event), \
                mock.patch.object(torch.cuda, "current_stream", return_value=_Cur()):
            for w in range(len(waves)):
                pfo.ensure(w)
                pfo.ensure(w + 1)
                pfo.wait(w)
                log.append(("apply", w))
                pfo.applied(w)
        copies = [x for x in log if x[0] == "copy"]
        # spill wave j uses half (j-1) % 2: slots [8, 9] / [10, 11]
        self.assertEqual([c[1] for c in copies], [[8, 9], [10, 11], [8, 9], [10]])
        # the first use of each half orders behind the forward stream; later
        # ones behind the apply of wave j-2 (its last reader) -- events only
        self.assertEqual([c[2] for c in copies], ["forward", "forward", "apply1", "apply2"])
        # the copy of wave w+1 is issued before apply(w) is enqueued
        for w in range(len(waves) - 1):
            self.assertLess(log.index(copies[w]), log.index(("apply", w)))
        # the forward waits on each spill wave's copy event before its apply
        for j in range(1, len(waves)):
            self.assertLess(log.index(("forward_waits", "copied%d" % j)), log.index(("apply", j)))
        self.assertTrue(all(x[0] != "synchronize" for x in log))


class TestDecodeUntouched(CustomTestCase):
    def test_single_wave_path_never_asks_pfo(self):
        c = _cache(scratch=E)
        c._plan_vector, c._wave_order = True, "expert"
        c._router_stats, c._hot_enabled, c._hot_frozen, c._heat = None, False, False, None
        c._eager_lru_armed = False
        c._route_note = False
        c.planner = types.SimpleNamespace(resident_ids=None, stats=types.SimpleNamespace(
            overflow_forwards=0, lookahead_dropped=0))
        seen = {}
        c._run_single_wave = lambda d, f, ids_list, prefetch: seen.update(n=len(ids_list))
        ids, disp, _ = _case(16, 5)
        with envs.SGLANG_WEG2_ENABLE_PREFILL_FETCH_OVERLAP.override(True), \
                mock.patch.object(eo, "prefill_fetch_overlap_form",
                                  side_effect=AssertionError("PFO asked on a single wave")):
            c._run_waves_vector(disp, apply_fn=None, lookahead=None)
        self.assertEqual(seen["n"], 16)

    def test_fetch_without_stream_is_the_old_call(self):
        c = _cache()
        self.assertIsNone(c._fetch([(R + 1, R), (R + 3, R + 1)]))
        self.assertEqual(float(c._resident["w"][R, 0]), _value(R + 1))
        self.assertEqual(c._scratch_holds, {R: R + 1, R + 1: R + 3})


if __name__ == "__main__":
    unittest.main()
