# SPDX-License-Identifier: Apache-2.0
"""H68c: the launcher's flag-reading probes build P's argv WITHOUT a cut.

``_max_running_requests``, ``_p_page_size`` and
``p_activation_reserve_provenance`` read one flag each off an ``argv_p`` built
for the question. They called it with ``stage_ratio=None``, and ``None`` means
"hand over the incumbent cut" -- which first checks the model's PP calibration
(#1362, W99 Weg2ModelIdentityMismatch). The first NVFP4 dry run (48 layers, no
calibration record yet) died there although its arm PINS the cut
(--pp-stage-ratio 29,11,8), whose own argv never asks for the incumbent:

    solve_p_cut -> _max_running_requests -> argv_p -> refuse_foreign_calibration

Desk only. A synthetic 48-layer checkpoint (the NVFP4 export's own config shape,
a one-name index -> a digest no calibration record carries): the probes answer,
the incumbent path still refuses by name, a pinned cut still builds.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher as lc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 12


def _model_dir(d):
    cfg = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 48,
            "layer_types": LAYER_TYPES,
            "hidden_size": 2560,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
        },
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {"model.language_model.layers.0.h68c_probe.weight": "x.safetensors"}}, f)
    return d


def _no_record(digest, calib_dir=hl.CALIB_DIR):
    return None, f"no PP calibration for model {digest}: (test) no record"


class ProbesAskNoCut(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = _model_dir(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(hl, "read_pp_calibration", _no_record)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _argv(self, **kw):
        return lc.argv_p("py", self.model, [1, 1, 1], 1, 1, lc.RING_FORM_SENTINEL_STORE_CFG, [], p_bs=4, **kw)

    def test_the_incumbent_path_still_refuses_by_name(self):
        with self.assertRaises(hl.Weg2ModelIdentityMismatch) as ctx:
            self._argv()
        self.assertIn("W99", str(ctx.exception))

    def test_a_pinned_cut_builds_and_states_it(self):
        argv = self._argv(stage_ratio="29,11,8", attn_stage_ratio="7,3,2")
        self.assertEqual(argv[argv.index("--pp-stage-ratio") + 1], "29,11,8")
        self.assertEqual(argv[argv.index("--pp-attn-stage-ratio") + 1], "7,3,2")

    def test_the_probes_answer_without_a_calibration(self):
        self.assertEqual(lc._max_running_requests(self.model, "P", 4), 4)
        self.assertEqual(lc._p_page_size(self.model, 4), int(self._argv(stage_ratio="", attn_stage_ratio="")[
            self._argv(stage_ratio="", attn_stage_ratio="").index("--page-size") + 1]))
        mib, line = lc.p_activation_reserve_provenance(self.model, 4)
        self.assertGreaterEqual(float(mib), 0.0)
        self.assertTrue(line)

    def test_the_probe_argv_carries_no_cut(self):
        argv = self._argv(stage_ratio="", attn_stage_ratio="")
        self.assertNotIn("--pp-stage-ratio", argv)
        self.assertNotIn("--pp-attn-stage-ratio", argv)


def _strip_cut(argv):
    out, skip = [], 0
    for tok in argv:
        if skip:
            skip -= 1
            continue
        if tok in ("--pp-stage-ratio", "--pp-attn-stage-ratio"):
            skip = 1
            continue
        out.append(tok)
    return out


class ACalibratedModelReadsTheSameFlags(CustomTestCase):
    """The INT4 line (a model WITH a calibration record): the probes read the
    very values they read before -- the probe argv IS the incumbent-cut argv
    minus the two ratio flags, token for token, so 0010 moves no INT4 number."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = _model_dir(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        digest, _ = hl.checkpoint_digest(self.model)
        rec = {"schema": hl.CALIB_SCHEMA, "model_digest": digest, "measured_counts": [29, 11, 8],
               "measured_attn_counts": [7, 3, 2], "measured_ms_per_layer": [102.12, 228.35, 178.21]}
        patcher = mock.patch.object(hl, "read_pp_calibration", lambda d, c=hl.CALIB_DIR: (rec, "test record"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_probe_argv_is_the_incumbent_argv_without_the_cut(self):
        for p_bs in (1, 4):
            full = lc.argv_p("py", self.model, [1, 1, 1], 1, 1, lc.RING_FORM_SENTINEL_STORE_CFG, [], p_bs=p_bs)
            self.assertEqual(full[full.index("--pp-stage-ratio") + 1], "29,11,8")  # the calibrated incumbent
            probe = lc.argv_p("py", self.model, [1, 1, 1], 1, 1, lc.RING_FORM_SENTINEL_STORE_CFG, [], p_bs=p_bs,
                              stage_ratio="", attn_stage_ratio="")
            self.assertEqual(_strip_cut(full), probe)
            self.assertEqual(lc._max_running_requests(self.model, "P", p_bs),
                             int(full[full.index("--max-running-requests") + 1]))
            self.assertEqual(lc._p_page_size(self.model, p_bs), int(full[full.index("--page-size") + 1]))


if __name__ == "__main__":
    unittest.main()
