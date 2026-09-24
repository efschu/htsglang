"""H25b: the draft's post on P's draft card goes to that stage's experts.

Numbers are the x137 front log's own (1e49837c84, 24.09.): ``PP-CUT POOL
TERM ... experts 1238 MiB/layer (512 experts, 2.42 MiB/row) at fractions
[0.26, 0.45, 0.39]``, layers 29/11/8, and ``PLATZTAUSCH-KARTE ... Store 378
Plaetze je Layer``; D's vectors ``183,137,168|0.06,0.44,0.365``.  The draft
checkpoint is replaced by a sparse safetensors fixture whose HEADER carries the
measured tensor classes (MTP rest 1522.8, embed 1212.5, lm_head 1212.5 MiB).
"""

import json
import os
import struct
import tempfile
import unittest
from types import SimpleNamespace

import pytest

try:
    from sglang.srt.layers.moe import expert_map as em
    from sglang.srt.weg2 import draft_post as dp
    from sglang.srt.weg2 import launcher as L
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

H25_OWN_DRAFT_FORM = True

MIB = 1 << 20
ROW_MIB = 1238.0 / 512
REST, EMBED, HEAD = int(1522.8 * MIB), int(1212.5 * MIB), int(1212.5 * MIB)


def _draft_dir():
    d = tempfile.mkdtemp(prefix="h25draft")
    header = {"mtp.layers.0.w": {"dtype": "U8", "shape": [REST], "data_offsets": [0, REST]},
              "model.embed_tokens.weight": {"dtype": "U8", "shape": [EMBED],
                                            "data_offsets": [REST, REST + EMBED]},
              "lm_head.weight": {"dtype": "U8", "shape": [HEAD],
                                 "data_offsets": [REST + EMBED, REST + EMBED + HEAD]}}
    raw = json.dumps(header).encode()
    with open(os.path.join(d, "mtp.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.truncate(8 + len(raw) + REST + EMBED + HEAD)  # sparse
    return d


class _Card:
    def __init__(self, i):
        self.nvml_index = i


CARDS = [_Card(1), _Card(0), _Card(2)]


class TestDraftPostArithmetic(unittest.TestCase):
    def test_p_post_is_checkpoint_minus_head_plus_buffers_plus_transient(self):
        w, t = dp.p_draft_post_mib(_draft_dir())
        # x137 PP2 WEG2-XCHG-COVER: planned 2735.2 + buffers 64.1 (walk 2799.4)
        self.assertAlmostEqual(w, 1522.8 + 1212.5 + 64.1, places=1)
        self.assertEqual(t, dp.P_DRAFT_PRODUCER_TRANSIENT_MIB)

    def test_x137_numbers_raise_pp2_by_176_rows(self):
        new, post, why = dp.raise_for_draft_post(
            fracs=[0.26, 0.45, 0.39], stage_layers=[29, 11, 8], row_mib=ROW_MIB,
            num_experts=512, draft_path=_draft_dir(), cards=CARDS)
        self.assertEqual(why, "")
        self.assertEqual((post.stage, post.card), (2, "nvml2"))
        # 3422 MiB / (8 layers x 2.418 MiB) = 176.9 -> 176 whole rows
        self.assertEqual(post.rows, 176)
        self.assertEqual(new[:2], [0.26, 0.45])
        # the rank's own count (max(1, ceil(f*E))) lands EXACTLY on 200 + 176
        self.assertEqual(dp.resident_count(512, 0.39), 200)
        self.assertEqual(dp.resident_count(512, float(dp._fmt(new[2]))), 376)
        self.assertIn("PP-CUT draft post (H25) card=nvml2", post.line())
        self.assertIn("-> experts +176 rows", post.line())

    def test_cap_is_the_h5_buffer_rule(self):
        rows, f = dp.expert_rows_for(1e9, 8, ROW_MIB, 512, 0.9)
        self.assertEqual(dp.resident_count(512, f), 510)  # (E-2), a buffer stays

    def test_no_checkpoint_changes_nothing(self):
        new, post, why = dp.raise_for_draft_post(
            fracs=[0.26, 0.45, 0.39], stage_layers=[29, 11, 8], row_mib=ROW_MIB,
            num_experts=512, draft_path=tempfile.mkdtemp(), cards=CARDS)
        self.assertIsNone(post)
        self.assertEqual(new, [0.26, 0.45, 0.39])
        self.assertIn("no *.safetensors", why)


class TestPublishedToAllThreePlaces(unittest.TestCase):
    """The arm: "FR_P steht an DREI Stellen" -- a reader left on the old value
    is the argv-doppelung class (Memory argv-doppelung-arm-schlaegt-planer)."""

    def test_apply_rewrites_flag_extra_and_env(self):
        d = _draft_dir()
        ns = SimpleNamespace(
            pp_cut_expert_device_fraction="0.26,0.45,0.39",
            extra_p=f"--speculative-draft-model-path {d} --rank-moe-resident-fraction "
                    f"0.26,0.45,0.39 --x --rank-moe-resident-fraction=0.26,0.45,0.39",
            extra_d="", env_p="SGLANG_MOE_SCRATCH_SLOTS=32;"
                              "SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.26,0.45,0.39;A=1")
        lines = []
        new = L.apply_p_draft_post(ns, CARDS, [0.26, 0.45, 0.39], [29, 11, 8],
                                   ROW_MIB, 512, lines.append)
        want = ",".join(dp._fmt(x) for x in new)
        self.assertEqual(ns.pp_cut_expert_device_fraction, want)
        self.assertEqual(L._argv_scalar(ns.extra_p, "--rank-moe-resident-fraction"), want)
        self.assertNotIn("0.26,0.45,0.39", ns.extra_p)
        self.assertEqual(L.parse_group_env(ns.env_p)["SGLANG_MOE_RESIDENT_EXPERT_FRACTION"], want)
        self.assertEqual(L.parse_group_env(ns.env_p)["SGLANG_MOE_SCRATCH_SLOTS"], "32")
        self.assertTrue(any(line.startswith("PP-CUT draft post (H25) card=nvml2") for line in lines))

    def test_solve_p_cut_applies_it_only_when_p_has_no_draft(self):
        import inspect

        src = inspect.getsource(L.solve_p_cut)
        i = src.index("apply_p_draft_post(")
        self.assertIn('== "off"', src[src.rindex("if ", 0, i):i])
        # before the pool term the model prices
        self.assertLess(i, src.index("model_pool = _pp_cut.PhasePoolModel("))


class TestDraftSwapHostBalance(unittest.TestCase):
    """Praezisierung 08:30Z ("nullsummenspiel"): the balance is COMPUTED from
    the Platztausch map, and the map says where the zero-sum premise breaks:
    the store is ``total - |common of all stages|``, and PP2's new rows are
    P-only extras (D holds 138 per layer, PP0 binds at 134) -- the store keeps
    378 slots, so the parked draft is a net host cost in this form."""

    def _slots(self, fr_pp):
        ls = [s for s, n in enumerate([29, 11, 8]) for _ in range(n)]
        return em.build_nested(total=512, ratios=[183, 137, 168], fr_pp=fr_pp,
                               fr_tp=[0.06, 0.44, 0.365], p_layer_stage=ls, pad_tp=1)["slots"]

    def test_x137_store_is_378_before_and_after(self):
        before = self._slots([0.26, 0.45, 0.39])
        after = self._slots([0.26, 0.45, dp.fraction_for_count(512, 376)])
        self.assertEqual((before, after), (378, 378))
        host = dp.d_draft_host_mib(_draft_dir(), share_embed=True)
        self.assertAlmostEqual(host, 1522.8 + 64.1, places=1)
        line = dp.draft_swap_line(slots_before=before, slots_after=after, layers=48,
                                  row_mib=ROW_MIB, d_draft_host=host)
        self.assertIn("store_before_mib=43872 store_after_mib=43872", line)
        self.assertIn("net_host_mib=+1587", line)

    def test_the_store_only_follows_the_binding_stage(self):
        # raising PP0 (the binding stage) to D's 138 shrinks the store to D's floor
        self.assertEqual(self._slots([dp.fraction_for_count(512, 140), 0.45, 0.39]), 374)


if __name__ == "__main__":
    unittest.main()
