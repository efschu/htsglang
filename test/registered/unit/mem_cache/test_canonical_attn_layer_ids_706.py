# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#706 must resolve a hybrid's attention layers, never guess them dense.

THE SPECIMEN, 2026-08-17. The Flip+HiCache boot died at readiness with

    CanonicalPageError: attention layers [51, 55, 59, 63] map to page slots
    [51, 55, 59, 63], which is not a contiguous run.

The slots should have been [12, 13, 14, 15]. The caller resolved the id list as
``full_attention_layer_ids or range(num_hidden_layers)`` under the comment "a
model with no hybrid split" -- a guard that cannot tell NOT HYBRID from NOT
POPULATED. The deployed checkpoint is a GDN hybrid with 16 full-attention layers
at 3, 7, ... 63; it took the dense branch, and the page was cut against 64 slots.

WHY THE FIX IS NOT IN ``get_hybrid_layer_ids``. That helper is SWA-scoped:
ModelConfig calls it only under ``if self.is_hybrid_swa``. A GDN hybrid is
correctly not ``is_hybrid_swa``, so a generic branch added there would either
misroute GDN hybrids into SWA machinery or never run at all. The authority
already exists instead -- ``Qwen3NextConfig.full_attention_layer_ids``, derived
from ``layers_block_type`` and inherited by ``Qwen3_5TextConfig``.

The first test reads the REAL deployed config off disk, so this cannot pass
against a fixture that has drifted from the checkpoint being served.
"""

import os
import unittest

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageError
from sglang.srt.mem_cache.canonical_page_store import resolve_attn_layer_ids
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

CHECKPOINT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-yarn1.5"


class _Cfg:
    """Minimal ModelConfig stand-in; only the attributes the ladder reads."""

    def __init__(self, *, full=None, text=None, n=64, hybrid=None, hybrid_swa=None,
                 arch=("StandInForCausalLM",)):
        self.full_attention_layer_ids = full
        self.hf_text_config = text
        self.num_hidden_layers = n
        self.is_hybrid = hybrid
        self.is_hybrid_swa = hybrid_swa
        self.hf_config = type("HF", (), {"architectures": list(arch)})()


class _Text:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class TheRealCheckpointResolvesToItsOwnLayers(unittest.TestCase):
    """Red-first against the deployed model, not against a fixture."""

    @unittest.skipUnless(
        os.path.isdir(CHECKPOINT), f"deployed checkpoint not present: {CHECKPOINT}"
    )
    def test_the_deployed_hybrid_gives_16_attention_layers(self):
        from sglang.srt.configs.model_config import ModelConfig

        mc = ModelConfig(model_path=CHECKPOINT, trust_remote_code=True)
        # The SWA-scoped list is EMPTY for this model -- that is the trap.
        self.assertFalse(getattr(mc, "full_attention_layer_ids", None))
        ids = resolve_attn_layer_ids(mc)
        self.assertEqual(16, len(ids))
        self.assertEqual([3, 7, 11, 15], ids[:4])
        self.assertEqual(63, ids[-1])

    @unittest.skipUnless(os.path.isdir(CHECKPOINT), "deployed checkpoint not present")
    def test_the_last_stage_maps_to_a_contiguous_run(self):
        """The exact failure: layers [51,55,59,63] must be slots [12..16)."""
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.mem_cache.canonical_kv_page import attn_layer_index

        mc = ModelConfig(model_path=CHECKPOINT, trust_remote_code=True)
        ids = resolve_attn_layer_ids(mc)
        slots = [attn_layer_index(i, ids) for i in (51, 55, 59, 63)]
        self.assertEqual([12, 13, 14, 15], slots)
        self.assertEqual(slots, list(range(slots[0], slots[0] + len(slots))))


class TheLadderPicksTheRightRung(unittest.TestCase):
    def test_a_populated_swa_list_wins(self):
        cfg = _Cfg(full=[1, 5], text=_Text(full_attention_layer_ids=[9, 9]))
        self.assertEqual([1, 5], resolve_attn_layer_ids(cfg))

    def test_the_checkpoint_property_covers_the_gdn_hybrid(self):
        cfg = _Cfg(
            full=[],
            text=_Text(
                full_attention_layer_ids=[3, 7, 11],
                layers_block_type=["linear", "linear", "linear", "full_attention"],
            ),
        )
        self.assertEqual([3, 7, 11], resolve_attn_layer_ids(cfg))

    def test_a_proven_dense_model_is_still_dense(self):
        """No declared kinds and no hybrid flag: range(n), byte-identical."""
        cfg = _Cfg(full=None, text=_Text(), n=32)
        self.assertEqual(list(range(32)), resolve_attn_layer_ids(cfg))

    def test_no_text_config_at_all_is_still_dense(self):
        cfg = _Cfg(full=None, text=None, n=8)
        self.assertEqual(list(range(8)), resolve_attn_layer_ids(cfg))


class TheUnresolvableHybridRefuses(unittest.TestCase):
    """CAN-FAIL: declared kinds with no id list must NOT fall through to dense.

    This is the specimen's shape. If this ever returns range(n) again, the page
    is cut against the wrong slots and only the contiguity check stands between
    that and a key naming bytes nobody can read.
    """

    def test_declared_kinds_without_ids_raises(self):
        cfg = _Cfg(
            full=[],
            text=_Text(layer_types=["linear_attention", "full_attention"]),
            arch=("MysteryHybridForCausalLM",),
        )
        with self.assertRaises(CanonicalPageError) as cm:
            resolve_attn_layer_ids(cfg)
        msg = str(cm.exception)
        self.assertIn("MysteryHybridForCausalLM", msg)
        self.assertIn("full_attention", msg)
        self.assertIn("refusing instead of guessing", msg)

    def test_a_hybrid_flag_without_ids_or_kinds_also_refuses(self):
        cfg = _Cfg(full=None, text=_Text(), hybrid=True)
        with self.assertRaises(CanonicalPageError):
            resolve_attn_layer_ids(cfg)


if __name__ == "__main__":
    unittest.main()
