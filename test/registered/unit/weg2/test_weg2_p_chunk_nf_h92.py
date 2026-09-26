"""H92: --p-chunk-policy on the Next-Flash line (weg2/p_chunk_nf.py + the NF launcher).

Pinned without a GPU:
  * the NF PROFILE is found by the checkpoint's model key only, a 27B key or a
    27B source is refused, and its fixed prediction reproduces the measured
    single-request P drains of its own boots (x178 front.log 552/654, 325);
  * the NF HARD LIMITS: ceiling = the priced P chunk, the measured transient
    support, the 4096 raster as ChunkLimits.page (every non-final chunk end
    on an absolute multiple of 4096), fixed = the P chunk, floor >= raster;
  * the FR_P coupling: a_s follows the expert stream, measured FR = profile;
  * the STREAM budget: a P stau plans the stream, not the head alone;
  * FIXED IS IDENTICAL: no env, no line, no planner, no stream flag.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.weg2 import p_chunk_nf as NF
from sglang.srt.weg2 import p_chunk_policy as P
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NF_KEY = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
KEY_27B = "Qwen3.8-27B-INT8-gdncov-vocabembed"
NF_MODEL = "/spinning/llm_stuff/club-3090/models-cache/" + NF_KEY
MODEL_27B = "/spinning/llm_stuff/club-3090/models-cache/" + KEY_27B
FR_REF = "0.332,0.64,0.733887"


def _limits(**kw):
    kw.setdefault("ceiling", 16384)
    kw.setdefault("transient_support_max", 16384)
    return NF.nf_limits(**kw)


def _check(tc, chunks, start, n, lim):
    tc.assertEqual(sum(chunks), n)
    tc.assertTrue(all(0 < c <= lim.max_tokens for c in chunks), chunks)
    p = start
    for c in chunks[:-1]:
        p += c
        tc.assertEqual(p % NF.NF_RASTER_TOKENS, 0, (chunks, p))


class TestProfile(unittest.TestCase):
    def test_found_by_model_key_only(self):
        prof = NF.resolve_profile("auto", NF_KEY)
        self.assertEqual(prof.model_key, NF_KEY)
        self.assertEqual(prof.profile, "nextflash")
        self.assertEqual(prof.ceiling_tokens, 16384)
        self.assertEqual(len(prof.stages), 3)
        with self.assertRaises(NF.NfChunkRefused):
            NF.resolve_profile("auto", KEY_27B)
        for src in ("builtin-int8", "fit:/x/P.log"):
            with self.assertRaises(NF.NfChunkRefused):
                NF.resolve_profile(src, NF_KEY)

    def test_a_foreign_profile_path_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            src = NF.find_profile(NF_KEY)
            with open(src) as fh:
                data = json.load(fh)
            data["model_key"] = KEY_27B
            path = os.path.join(d, "x.pchunk.json")
            with open(path, "w") as fh:
                json.dump(data, fh)
            with self.assertRaises(NF.NfChunkRefused):
                NF.resolve_profile(path, NF_KEY)
            self.assertEqual(NF.find_profile(KEY_27B, d), path)
            with open(os.path.join(d, "y.pchunk.json"), "w") as fh:
                json.dump(data, fh)
            with self.assertRaises(NF.NfChunkRefused):  # two profiles, one key: no silent pick
                NF.find_profile(KEY_27B, d)

    def test_shape_is_the_nf_physics(self):
        """a_s is the expert stream: PP0 carries the largest fixed part, and
        a_0 is larger than PP0's per-token work of a 4096 chunk."""
        st = NF.resolve_profile("auto", NF_KEY).stages
        a = [s.base_ms(0) for s in st]
        b = [s.base_ms(1) - s.base_ms(0) for s in st]
        self.assertGreater(a[0], a[1])
        self.assertGreater(a[1], a[2])
        self.assertGreater(a[0], b[0] * 4096)
        self.assertTrue(all(x > 0 for x in b))

    def test_reproduces_its_boots(self):
        """Fixed prediction against the measured single-request P drains
        (x178 front.log:552 8361 -> 4.7 s, :654 8466 -> 4.9 s within 5 %;
        :325 97841 -> 24.2 s within 10 %: the drain also holds the front and
        tail hand-off, which the stage model does not price)."""
        st = NF.resolve_profile("auto", NF_KEY).stages
        lim = _limits()
        for n, wall, tol in ((8361, 4700.0, 0.05), (8466, 4900.0, 0.05), (97841, 24200.0, 0.10)):
            res = P.plan_detail(n, 3, st, lim)
            self.assertLess(abs(res.fixed_ms - wall) / wall, tol, (n, res.fixed_ms))


class TestLimits(unittest.TestCase):
    def test_defaults(self):
        lim = _limits()
        self.assertEqual((lim.max_tokens, lim.min_tokens, lim.fixed_tokens), (16384, 4096, 16384))
        self.assertEqual((lim.page, lim.grid), (4096, 0))
        self.assertEqual(lim.ladder(), (4096, 8192, 16384))
        self.assertEqual(lim.dynamic_min_tokens, 16384)

    def test_refusals(self):
        for kw in (dict(max_tokens=32768), dict(max_tokens=8192, fixed_tokens=16384), dict(min_tokens=2048),
                   dict(raster=512), dict(max_tokens=16384, transient_support_max=8192),
                   dict(min_tokens=6000)):
            with self.assertRaises(NF.NfChunkRefused, msg=kw):
                _limits(**kw)

    def test_plans_keep_every_limit(self):
        st = NF.resolve_profile("auto", NF_KEY).stages
        lim = _limits(dynamic_min_tokens=0)
        for n in (1, 4095, 8361, 12656, 19806, 33918, 97841, 118836, 262144):
            for start in (0, 64, 6080, 16384):
                ch = P.chunk_plan(n, 3, st, lim, start=start)
                _check(self, ch, start, n, lim)

    def test_rechnung_pins(self):
        """The H92 Rechnung: under today's ceiling the plan is the fixed plan
        for the 97k single and the stau6 stream; only a stream that drains in
        a short tail gets a halving tail ramp (burst 8 x 4.2k)."""
        st = NF.resolve_profile("auto", NF_KEY).stages
        lim = _limits()
        r97 = P.plan_detail(97841, 3, st, lim)
        self.assertEqual(list(r97.chunks), [16384] * 5 + [15921])
        self.assertEqual(r97.candidate, "fixed")
        r6 = P.plan_detail(118836, 3, st, lim)
        self.assertEqual(list(r6.chunks), [16384] * 7 + [4148])
        rb = P.plan_detail(33918, 3, st, lim)
        self.assertEqual(list(rb.chunks), [16384, 8192, 8192, 1150])
        self.assertLess(rb.gain, 0.05)


class TestFrCoupling(unittest.TestCase):
    def test_measured_fr_is_the_profile(self):
        prof = NF.resolve_profile("auto", NF_KEY)
        st, note = NF.fr_adjusted_stages(prof, NF.parse_fractions(FR_REF))
        self.assertEqual(st, prof.stages)
        self.assertIn("= profile", note)
        st, note = NF.fr_adjusted_stages(prof, None)
        self.assertEqual(st, prof.stages)

    def test_lower_fr_streams_more(self):
        prof = NF.resolve_profile("auto", NF_KEY)
        st, note = NF.fr_adjusted_stages(prof, (0.2, 0.64, 0.733887))
        self.assertIn("HOCHRECHNUNG", note)
        want = prof.stages[0].base_ms(0) + prof.fetch_ref_ms[0] * ((1 - 0.2) / (1 - 0.332) - 1)
        self.assertAlmostEqual(st[0].base_ms(0), want, places=6)
        self.assertAlmostEqual(st[0].base_ms(1) - st[0].base_ms(0),
                               prof.stages[0].base_ms(1) - prof.stages[0].base_ms(0), places=9)
        self.assertEqual(st[1:], prof.stages[1:])
        with self.assertRaises(NF.NfChunkRefused):
            NF.fr_adjusted_stages(prof, (1.5, 0.6, 0.7))


def _req(rid, n, pos=0):
    return SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(n)), origin_input_ids=[],
                           output_ids=[], prefix_indices=list(range(pos)))


class TestStreamBudget(unittest.TestCase):
    def _sched(self, stream):
        from sglang.srt.managers.scheduler import Scheduler

        class _S:
            _p_chunk_policy_width = Scheduler._p_chunk_policy_width

        prof = NF.resolve_profile("auto", NF_KEY)
        s = _S()
        s.chunked_prefill_size = 16384
        s._p_chunk_planner = P.ChunkPlanner(P.PolicySpec(prof.stages, _limits(), "t"))
        s._p_chunk_stream = stream
        return s

    def test_stream_end(self):
        others = [_req("b", 19806), _req("c", 19806, pos=4096)]
        self.assertEqual(NF.stream_end(16384, 32768, others), 32768 + 19806 + 15710)
        self.assertEqual(NF.stream_end(5, 5, []), 5)

    def test_stau_plans_the_stream_not_the_head(self):
        """Head 49152 with five 19806-token requests waiting: alone it drains
        with a halving tail (16384 at 16384, then 8192 at 32768), inside the
        stau the pipeline keeps running at the ceiling."""
        waiting = [_req(f"w{i}", 19806) for i in range(5)]
        for stream, want in ((False, [16384, 8192]), (True, [16384, 16384])):
            s = self._sched(stream)
            s.waiting_queue = list(waiting)
            got = []
            for pos in (16384, 32768):
                s.chunked_req = _req("head", 49152, pos=pos)
                got.append(s._p_chunk_policy_width())
            self.assertEqual(got, want, stream)

    def test_head_from_the_queue_and_short_stream(self):
        s = self._sched(True)
        s.chunked_req = None
        s.waiting_queue = [_req("a", 4231), _req("b", 4303)]
        # a stream at or below the P chunk is not planned: the fixed budget
        self.assertEqual(s._p_chunk_policy_width(), 16384)
        s.waiting_queue = []
        self.assertEqual(s._p_chunk_policy_width(), 0)

    def test_budget_env(self):
        self.assertTrue(NF.budget_is_stream({NF.BUDGET_ENV: "stream"}))
        self.assertFalse(NF.budget_is_stream({}))
        self.assertFalse(NF.budget_is_stream({NF.BUDGET_ENV: "request"}))


try:
    from sglang.srt.weg2 import launcher as L
except Exception:  # pragma: no cover
    L = None


@unittest.skipIf(L is None, "weg2 launcher unavailable")
class TestLauncherNF(unittest.TestCase):
    def setUp(self):
        self._c = dict(L._P_CHUNK)
        self._pc = L.P_CHUNKED_PREFILL_TOKENS
        L.P_CHUNKED_PREFILL_TOKENS = 16384  # the arm's SGLANG_WEG2_P_CHUNKED_PREFILL_TOKENS

    def tearDown(self):
        L._P_CHUNK.clear()
        L._P_CHUNK.update(self._c)
        L.P_CHUNKED_PREFILL_TOKENS = self._pc

    def _apply(self, *extra, model=NF_MODEL):
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--model", model, *extra])
        L.apply_p_chunk_policy(ns)
        return ns

    def test_default_fixed_is_byte_identical(self):
        ns = self._apply()
        self.assertEqual(ns.p_chunk_policy, "fixed")
        env = {"SGLANG_MOE_RESIDENT_EXPERT_FRACTION": FR_REF}
        self.assertEqual(L.p_chunk_policy_install(env), ({}, []))
        self.assertIsNone(L.p_chunk_policy_spec())
        # dynamic-only knobs are inert under fixed, even refused values
        self._apply("--p-chunk-max", "65536", model=MODEL_27B)
        self.assertEqual(L.p_chunk_policy_install(env), ({}, []))

    def test_dynamic_ships_the_nf_spec(self):
        self._apply("--p-chunk-policy", "dynamic")
        env, lines = L.p_chunk_policy_install({"SGLANG_MOE_RESIDENT_EXPERT_FRACTION": FR_REF})
        self.assertEqual(env[P.POLICY_ENV], "dynamic")
        self.assertEqual(env[NF.BUDGET_ENV], "stream")
        spec = P.PolicySpec.from_json(env[P.SPEC_ENV])
        prof = NF.resolve_profile("auto", NF_KEY)
        self.assertEqual(spec.stages, prof.stages)
        self.assertEqual(spec.limits.key(), _limits().key())
        self.assertEqual(len(lines), 1 + len(NF.NF_DRY_RUN_TOKENS))
        self.assertIn("P-CHUNK-POLICY armed policy=dynamic group=P line=NF budget=stream", lines[0])
        self.assertIn("= profile (measured)", lines[0])
        self.assertIn("HOCHRECHNUNG", lines[1])
        # the ceiling is the P chunk argv_p already carries
        self.assertEqual(spec.limits.max_tokens, L.P_CHUNKED_PREFILL_TOKENS)

    def test_fr_from_the_shipped_env(self):
        self._apply("--p-chunk-policy", "dynamic")
        env, lines = L.p_chunk_policy_install({"SGLANG_MOE_RESIDENT_EXPERT_FRACTION": "0.2,0.64,0.733887"})
        spec = P.PolicySpec.from_json(env[P.SPEC_ENV])
        self.assertGreater(spec.stages[0].base_ms(0), NF.resolve_profile("auto", NF_KEY).stages[0].base_ms(0))
        self.assertIn("HOCHRECHNUNG", lines[0])

    def test_refusals(self):
        for extra, model in ((("--p-chunk-policy", "dynamic"), MODEL_27B),
                             (("--p-chunk-policy", "dynamic", "--p-chunk-model", "builtin-int8"), NF_MODEL),
                             (("--p-chunk-policy", "dynamic", "--p-chunk-max", "32768"), NF_MODEL),
                             (("--p-chunk-policy", "dynamic", "--p-chunk-min", "512"), NF_MODEL),
                             (("--p-chunk-policy", "dynamic", "--p-chunk-max", "8192",
                               "--p-chunk-fixed", "16384"), NF_MODEL),
                             (("--p-chunk-policy", "dynamic", "--p-chunk-raster", "512"), NF_MODEL)):
            with self.assertRaises(SystemExit, msg=(extra, model)):
                self._apply(*extra, model=model)

    def test_forced_width_probe(self):
        """min = max = fixed forces one width (a metal probe of a_s/b_s at
        another M), still on the raster and under the ceiling."""
        self._apply("--p-chunk-policy", "dynamic", "--p-chunk-min", "8192", "--p-chunk-max", "8192")
        env, _ = L.p_chunk_policy_install({"SGLANG_MOE_RESIDENT_EXPERT_FRACTION": FR_REF})
        spec = P.PolicySpec.from_json(env[P.SPEC_ENV])
        self.assertEqual(P.chunk_plan(97841, 3, spec.stages, spec.limits), [8192] * 11 + [7729])

    def test_profile_belongs_to_its_p_chunk(self):
        """The profile was measured at P chunk 16384; a boot pricing another
        P chunk (the env default 4096) cannot use its a_s."""
        L.P_CHUNKED_PREFILL_TOKENS = 4096
        with self.assertRaises(SystemExit):
            self._apply("--p-chunk-policy", "dynamic")


if __name__ == "__main__":
    unittest.main()
