# SPDX-License-Identifier: Apache-2.0
"""PP-CUT DRAFT-ZELLE: the pool model prices the draft KV the last P stage carries in its cell.

METAL dkr27browauthorityyarn2bar1fs09292006 (z30y3i, YaRN x2, cut 34,17,13 / attn 8,4,4): the solver
priced pool_tokens=508382 (floor 508048); P sized
  PP2 "KV pool sizing: available_bytes=5045059584 (4.699 GiB), cell_size=18432 -> max_total_num_tokens=273712"
= 4 attention layers x 2048 + the DFlash2 draft's 5 x 2 x 8 x 128 B -- and leg1 of the 300k needle died with
"Input length (299771 tokens) exceeds the maximum allowed length (273706)".
Record (non-YaRN, dkr27browauthoritybar1fs09291750, cut 43,11,10 / attn 10,3,3): PP2 cell_size=16384 = 3 x 2048
+ 10240, available_bytes=6931251200 -> 423050.
"""
import json
import os
import unittest
from unittest import mock

try:
    from flliper.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover

    def register_cpu_ci(*args, **kwargs):
        return None


from flliper.srt.planner import pp_cut as P
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TARGET = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
DRAFT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-DFlash2-W8-lued"
ACK = ("mamba pre-capture reserve", "speculative intermediate state", "GGUF dequant scratch")


def _model(free, fixed, extra=()):
    """The launcher's PhasePoolModel of the YaRN boot, transcribed from its 'PP-CUT budget posts' line."""
    return P.PhasePoolModel(
        free_mib=tuple(free), weight_mib_per_layer=363.4, kv_mib_per_token_per_attn_layer=2048 / P.MIB,
        arming_floor_mib=(1229.0,) * 3, stage_fixed_mib=tuple(fixed), activation_reserve_mib=1024.0,
        corridor_holdback_mib=1800.0, mamba_mib_per_linear_layer_per_slot=1.5588, mamba_slots=3,
        prefill_graph_pool_mib=(159.99,) * 3, extra_cell_bytes_by_stage=tuple(extra),
        zero_posts_acknowledged=ACK)


class DraftCellTest(CustomTestCase):
    def test_runtime_arithmetic_of_the_metal(self):
        """The model's capacity function with the draft cell IS the runtime's: boot bytes in, boot tokens out."""
        m = _model([1, 1, 1], [0, 0, 0], (0, 0, 10240))
        self.assertEqual(P.stage_capacity_tokens(5045059584 / P.MIB, 4, m, 10240), 273712)   # YaRN PP2
        self.assertEqual(P.stage_capacity_tokens(6931251200 / P.MIB, 3, m, 10240), 423050)   # 1750 PP2
        self.assertEqual(P.stage_capacity_tokens(11521163264 / P.MIB, 8, m, 0), 703196)      # YaRN PP0

    def test_solver_no_longer_overprices_the_last_stage(self):
        cut = (34, 17, 13)
        attn = (8, 4, 4)
        free, fixed = [26128, 15704, 15432], [2406.0, 1169.5, 3710.0]
        old = P.stage_pp_capacities(cut, attn, _model(free, fixed))
        new = P.stage_pp_capacities(cut, attn, _model(free, fixed, (0, 0, 10240)))
        self.assertAlmostEqual(min(old), 508382, delta=5)            # what the launcher logged (CLEARS 508048)
        self.assertGreater(min(old), 273712)                         # the metal: +86 %, the killing direction
        self.assertLessEqual(new[2], 273712)                         # priced at or under what P really holds
        self.assertEqual(new[:2], old[:2])                           # the other stages carry no draft
        self.assertLess(min(new), 506000)                            # the cut no longer clears the cap

    def test_empty_is_byte_identical(self):
        m = _model([26128, 15704, 15432], [2406.0, 1169.5, 3710.0])
        self.assertEqual(m.extra_cell_bytes(2), 0)
        self.assertEqual(P.stage_capacity_tokens(4000.0, 4, m), P.stage_capacity_tokens(4000.0, 4, m, 0))


@unittest.skipUnless(os.path.isdir(DRAFT) and os.path.isdir(TARGET), "rig checkpoints not mounted")
class LauncherDraftCellTest(CustomTestCase):
    def _cell(self, on_p, dflash):
        from flliper.srt.pdflip import launcher as L

        tcfg = json.load(open(os.path.join(TARGET, "config.json")))
        tcfg = tcfg.get("text_config") or tcfg
        ns = mock.Mock(dflash_draft_path=DRAFT)
        with mock.patch.dict(L._SPEC_FORM, {"draft_kv_on_p": on_p, "form": "DFLASH" if dflash else "NEXTN"}):
            return L.p_draft_kv_cell_bytes_by_stage(ns, 3, tcfg)

    def test_dflash_draft_geometry(self):
        cell, line = self._cell(True, True)
        self.assertEqual(cell, (0, 0, 10240))                        # 5 x 2 x 8 x 128 x 1 B, from the draft's config
        self.assertIn("DFLASH", line)

    def test_no_draft_on_p(self):
        self.assertEqual(self._cell(False, True), ((), None))

    def test_nextn_is_one_target_layer(self):
        cell, _ = self._cell(True, False)
        self.assertEqual(cell, (0, 0, 2048))


if __name__ == "__main__":
    unittest.main()
