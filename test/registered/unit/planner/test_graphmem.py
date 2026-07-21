"""Unit tests for the CUDA-graph memory module (planner/graphmem.py):
capture-line parser (real boot-log lines), per-rung ladder itemization,
measured-anchor store (measured-overrides-estimate), and the calibrated
heuristic staying inside its stated error band on real measured values.

The embedded log lines are VERBATIM lines from real boots of this rig
(/tmp/sglang_boot_*.log, /tmp/energy_boot_*.log: Qwen3.6-27B-FP8 tp=3 with
and without the 5-rung adaptive ladder, Qwen3-0.6B tp=1) -- the calibration
data the heuristic constants were fit against.
"""

import os
import tempfile
import unittest

from sglang.srt.planner import graphmem
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


# Real capture lines (verbatim) from /tmp/sglang_boot_30100.log (tp=3 decode),
# /tmp/sglang_boot_30099.log (tp=1 prefill+decode) and
# /tmp/energy_boot_31008.log (spec: verify + adaptive draft ladder).
_REAL_DECODE_TP3 = """\
[2026-07-21 13:35:33 TP1] Capture target decode CUDA graph end. elapsed=4.81 s, mem usage=0.27 GB, avail mem=5.34 GB.
[2026-07-21 13:35:33 TP2] Capture target decode CUDA graph end. elapsed=4.82 s, mem usage=0.27 GB, avail mem=5.47 GB.
[2026-07-21 13:35:33 TP0] Capture target decode CUDA graph end. elapsed=4.83 s, mem usage=0.24 GB, avail mem=7.28 GB.
"""

_REAL_PREFILL_TP1 = """\
[2026-07-21 10:36:54] Capture target prefill CUDA graph end. elapsed=3.83 s, mem usage=0.28 GB, avail mem=5.23 GB.
[2026-07-21 10:36:55] Capture target decode CUDA graph end. elapsed=0.80 s, mem usage=0.11 GB, avail mem=5.12 GB.
"""

_REAL_SPEC_LADDER = """\
[2026-07-21 08:21:55 TP0] Capture target verify CUDA graph begin. backend=full, num_tokens_per_bs=6, bs=[1, 2, 3, 4, 5, 6], avail mem=7.77 GB
[2026-07-21 08:22:00 TP0] Capture target verify CUDA graph end. elapsed=4.56 s, mem usage=0.38 GB, avail mem=7.39 GB.
[2026-07-21 08:22:00 TP0] Capture draft decode CUDA graph begin. backend=full, num_tokens_per_bs=1, bs=[1, 2, 3, 4, 5, 6], avail mem=7.39 GB
[2026-07-21 08:22:01 TP0] Capture draft decode CUDA graph end. elapsed=1.56 s, mem usage=0.38 GB, avail mem=7.01 GB.
[2026-07-21 08:22:01 TP0] Capture draft extend CUDA graph begin. backend=full, num_tokens_per_bs=6, bs=[1, 2, 3, 4, 5, 6], avail mem=7.01 GB
[2026-07-21 08:22:02 TP0] Capture draft extend CUDA graph end. elapsed=0.74 s, mem usage=0.14 GB, avail mem=6.87 GB.
[2026-07-21 08:22:02 TP0] Capture draft decode CUDA graph begin. backend=full, num_tokens_per_bs=1, bs=[1, 2, 3, 4, 5, 6], avail mem=6.77 GB
[2026-07-21 08:22:03 TP0] Capture draft decode CUDA graph end. elapsed=0.85 s, mem usage=0.09 GB, avail mem=6.68 GB.
[2026-07-21 08:22:03 TP0] Capture draft extend CUDA graph begin. backend=full, num_tokens_per_bs=5, bs=[1, 2, 3, 4, 5, 6], avail mem=6.68 GB
[2026-07-21 08:22:04 TP0] Capture draft extend CUDA graph end. elapsed=0.74 s, mem usage=0.08 GB, avail mem=6.60 GB.
"""

_SERVER_ARGS_SNIPPET = (
    "[2026-07-21 08:21:32] server_args=ServerArgs(model_path='/models/"
    "Qwen3.6-27B-FP8', tp_size=3, kv_cache_dtype='fp8_e4m3', "
    "cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(backend='full', "
    "max_bs=16, bs=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16], "
    "tc_compiler='eager', full_prefill_max_req=None)), "
    "speculative_algorithm='EAGLE', speculative_num_steps=5, "
    "speculative_eagle_topk=1, speculative_num_draft_tokens=6, "
    "speculative_adaptive=True)\n"
)


class TestCaptureParser(CustomTestCase):
    def test_real_decode_lines_parse_exact_values(self):
        # Three real tp=3 target-decode lines: per-rank MiB must match the
        # printed GB values exactly (0.27 GB = 276.48 MiB, 0.24 = 245.76).
        entries = graphmem.parse_capture_lines(_REAL_DECODE_TP3)
        self.assertEqual(len(entries), 3)
        by_rank = {e["rank"]: e for e in entries}
        self.assertAlmostEqual(by_rank[1]["mib"], 0.27 * 1024, places=2)
        self.assertAlmostEqual(by_rank[0]["mib"], 0.24 * 1024, places=2)
        self.assertTrue(all(e["kind"] == "target decode" for e in entries))
        self.assertTrue(all(e["rung"] == 0 for e in entries))

    def test_real_prefill_line_and_rankless_format(self):
        # tp=1 logs carry no "TPn" tag: rank None, kinds prefill + decode.
        entries = graphmem.parse_capture_lines(_REAL_PREFILL_TP1)
        kinds = [e["kind"] for e in entries]
        self.assertEqual(kinds, ["target prefill", "target decode"])
        self.assertAlmostEqual(entries[0]["mib"], 0.28 * 1024, places=2)
        self.assertIsNone(entries[0]["rank"])

    def test_draft_ladder_rungs_keyed_separately(self):
        # The adaptive ladder captures each rung as its OWN graph: repeated
        # draft kinds get increasing rung indices, and the begin lines pin
        # the rung's token count (extend k=6 then k=5).
        entries = graphmem.parse_capture_lines(_REAL_SPEC_LADDER)
        drafts = [e for e in entries if e["kind"].startswith("draft")]
        self.assertEqual(len(drafts), 4)
        extends = [e for e in drafts if e["kind"] == "draft extend"]
        self.assertEqual([e["rung"] for e in extends], [0, 1])
        self.assertEqual([e["tokens_per_bs"] for e in extends], [6, 5])
        decodes = [e for e in drafts if e["kind"] == "draft decode"]
        self.assertEqual([e["rung"] for e in decodes], [0, 1])
        # first-capture workspace is real: rung 0 decode >> rung 1 decode.
        self.assertGreater(decodes[0]["mib"], 3 * decodes[1]["mib"])

    def test_summarize_itemizes_per_kind_and_rung(self):
        s = graphmem.summarize_captures(
            graphmem.parse_capture_lines(_REAL_SPEC_LADDER)
        )
        labels = [it["label"] for it in s["items"]]
        self.assertIn("target verify", labels)
        self.assertIn("draft extend k=6", labels)
        self.assertIn("draft extend k=5", labels)
        # per-rank totals: everything sits on TP0 here.
        self.assertAlmostEqual(
            s["per_rank_mib"][0],
            (0.38 + 0.38 + 0.14 + 0.09 + 0.08) * 1024,
            places=1,
        )
        self.assertEqual(s["n_captures"], 5)

    def test_boot_meta_from_server_args_line(self):
        meta = graphmem.parse_boot_meta(_SERVER_ARGS_SNIPPET)
        self.assertEqual(meta["model_path"], "/models/Qwen3.6-27B-FP8")
        self.assertEqual(meta["tp_size"], 3)
        self.assertEqual(meta["speculative_num_draft_tokens"], 6)
        self.assertTrue(meta["speculative_adaptive"])
        self.assertEqual(len(meta["decode_bs"]), 12)
        self.assertIsNone(graphmem.parse_boot_meta("no server args here"))


class TestAnchorStore(CustomTestCase):
    def _store(self):
        tmp = tempfile.mkdtemp()
        return graphmem.AnchorStore(os.path.join(tmp, "anchors.json"))

    def test_scan_and_lookup_roundtrip(self):
        tmp = tempfile.mkdtemp()
        log = os.path.join(tmp, "sglang_boot_1.log")
        with open(log, "w") as f:
            f.write(_SERVER_ARGS_SNIPPET + _REAL_SPEC_LADDER)
        store = self._store()
        n = graphmem.scan_boot_logs([log], store=store)
        self.assertEqual(n, 1)
        hit = store.lookup(
            {
                "model_path": "/models/Qwen3.6-27B-FP8",
                "tp_size": 3,
                "kv_cache_dtype": "fp8_e4m3",
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 5,
                "speculative_num_draft_tokens": 6,
                "speculative_adaptive": True,
                "decode_bs": list(range(12)),
            }
        )
        self.assertIsNotNone(hit)
        self.assertEqual(hit["provenance"], "measured")

    def test_anchor_beats_heuristic_and_key_shape_changes_fall_back(self):
        # measured-overrides-estimate: an anchored config shape returns the
        # MEASURED numbers; changing k (draft tokens) changes the key and
        # falls back to the (ladder-aware) heuristic.
        tmp = tempfile.mkdtemp()
        log = os.path.join(tmp, "sglang_boot_2.log")
        with open(log, "w") as f:
            f.write(_SERVER_ARGS_SNIPPET + _REAL_SPEC_LADDER)
        store = self._store()
        graphmem.scan_boot_logs([log], store=store)
        geom = {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "tp_size": 3,
            "max_running_requests": 32,
            "decode_bs": list(range(12)),
        }
        spec = {
            "speculative_algorithm": "EAGLE",
            "speculative_num_steps": 5,
            "speculative_num_draft_tokens": 6,
            "speculative_adaptive": True,
        }
        hit = graphmem.estimate(
            geom, spec, model_path="/models/Qwen3.6-27B-FP8",
            kv_cache_dtype="fp8_e4m3", store=store, scan=False,
        )
        self.assertEqual(hit["provenance"], "measured")
        self.assertAlmostEqual(
            hit["per_rank_mib"],
            (0.38 + 0.38 + 0.14 + 0.09 + 0.08) * 1024,
            places=1,
        )
        miss = graphmem.estimate(
            geom, dict(spec, speculative_num_draft_tokens=4),
            model_path="/models/Qwen3.6-27B-FP8",
            kv_cache_dtype="fp8_e4m3", store=store, scan=False,
        )
        self.assertEqual(miss["provenance"], "heuristic")


class TestHeuristic(CustomTestCase):
    """The heuristic must stay inside its STATED error band on the real
    measured calibration points (the same boots the constants were fit on)."""

    def _assert_in_band(self, predicted, measured):
        band = graphmem.ERROR_BAND_PCT / 100.0
        self.assertLess(
            abs(predicted - measured) / measured,
            band,
            f"predicted {predicted:.0f} MiB vs measured {measured:.0f} MiB "
            f"exceeds the stated +-{graphmem.ERROR_BAND_PCT}% band",
        )

    def test_27b_tp3_decode_within_band(self):
        # Qwen3.6-27B (hidden 5120, 64 layers) tp=3, 12 decode bs entries:
        # measured 0.24-0.27 GB per rank (sglang_boot_30100.log).
        est = graphmem.heuristic_estimate(
            {
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "tp_size": 3,
                "max_running_requests": 32,
                "decode_bs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
            },
            {},
        )
        self._assert_in_band(est["per_rank_mib"], 0.27 * 1024)
        self._assert_in_band(est["per_rank_mib"], 0.24 * 1024)

    def test_0p6b_tp1_decode_within_band(self):
        # Qwen3-0.6B (hidden 1024, 28 layers) tp=1, 6 decode bs entries:
        # measured 0.11 GB (sglang_boot_30099.log).
        est = graphmem.heuristic_estimate(
            {
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "tp_size": 1,
                "max_running_requests": 16,
                "decode_bs": [1, 2, 4, 8, 12, 16],
            },
            {},
        )
        self._assert_in_band(est["per_rank_mib"], 0.11 * 1024)

    def test_27b_spec_ladder_total_within_band(self):
        # The full spec shape (verify + 5-rung adaptive draft ladder) on the
        # 27B tp=3: rank-0 measured total = verify 0.33 GB + draft captures
        # 1.01 GB (energy_boot_31007.log) = ~1372 MiB.
        est = graphmem.heuristic_estimate(
            {
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "tp_size": 3,
                "max_running_requests": 32,
            },
            {
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 5,
                "speculative_num_draft_tokens": 6,
                "speculative_adaptive": True,
            },
        )
        measured_rank0 = (0.33 + 0.35 + 0.14 + 0.09 + 0.08 + 0.07 + 0.08
                          + 0.07 + 0.06 + 0.07) * 1024
        self._assert_in_band(est["per_rank_mib"], measured_rank0)

    def test_ladder_awareness_changes_prediction(self):
        base_geom = {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "tp_size": 3,
            "max_running_requests": 32,
        }
        spec = {
            "speculative_algorithm": "EAGLE",
            "speculative_num_steps": 5,
            "speculative_num_draft_tokens": 6,
            "speculative_adaptive": True,
        }
        full = graphmem.heuristic_estimate(base_geom, spec)
        # adaptive off -> ONE rung -> fewer items, less memory.
        single = graphmem.heuristic_estimate(
            base_geom, dict(spec, speculative_adaptive=False)
        )
        self.assertLess(single["per_rank_mib"], full["per_rank_mib"])
        self.assertLess(len(single["items"]), len(full["items"]))
        # smaller k -> fewer rungs than the 5-rung ladder.
        small_k = graphmem.heuristic_estimate(
            base_geom,
            dict(spec, speculative_num_steps=3,
                 speculative_num_draft_tokens=4),
        )
        self.assertLess(small_k["per_rank_mib"], full["per_rank_mib"])
        # the itemization names each rung separately.
        labels = " ".join(it["label"] for it in full["items"])
        self.assertIn("k=6", labels)
        self.assertIn("k=2", labels)

    def test_rung_list_shapes(self):
        self.assertEqual(
            graphmem.ladder_rungs(
                {"speculative_num_draft_tokens": 6,
                 "speculative_num_steps": 5,
                 "speculative_adaptive": True}
            ),
            [6, 5, 4, 3, 2],
        )
        self.assertEqual(
            graphmem.ladder_rungs(
                {"speculative_num_draft_tokens": 4,
                 "speculative_adaptive": False}
            ),
            [4],
        )
        self.assertEqual(graphmem.ladder_rungs({}), [])

    def test_provenance_and_band_are_stated(self):
        est = graphmem.heuristic_estimate(
            {"hidden_size": 2048, "num_hidden_layers": 24, "tp_size": 2}, {}
        )
        self.assertEqual(est["provenance"], "heuristic")
        self.assertEqual(est["error_band_pct"], graphmem.ERROR_BAND_PCT)
        self.assertIn("estimate", est["formula"])


if __name__ == "__main__":
    unittest.main()
