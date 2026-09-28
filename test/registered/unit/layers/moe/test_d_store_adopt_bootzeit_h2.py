"""NF-Bootzeit H2: group D does not read (nor rewrite) the expert rows P put
into the shared store.

MEASURED (rc12z10 0928_083156 / 091438): every D rank read its whole expert
window from the checkpoint and ``write_rows`` wrote the cold rows into the
store (288 ``ct-stream-presplit`` lines), rows P had already written and
published (``L<n>-<attr>.bin.r0.written.json``). Under the Platztausch map D
holds [11, 73, 84] of [192, 144, 176] experts per layer resident, so 94 / 49 /
52 % of what the D ranks read were rows the store already had.

Toy map below (12 experts, 2 D ranks of 6, pad expert on each rank): rank 1,
layer 0 holds prefix [6] + extra [7]; P published every store slot for
w13_weight_packed, and every slot but expert 9's for w2_weight_packed.
Expected veto: {8, 10, 11} -- not the residents, not the half-published 9.

RED on 7b2c6ee5ef (no store_adopt; weight_name_needed reads every owned
expert). GREEN with the fix.
"""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from unittest import mock

import torch

try:
    from sglang.srt.layers.moe import store_adopt as sa
except ImportError:  # pragma: no cover - base
    sa = None

LAYER = 0
KARTE = {
    "version": 2,
    "total": 12,
    "slots": 10,
    "p_layer_stage": [0],
    "phases": {
        "P": {"resident": [[0, 6]], "common": [[0, 6]],
              "slot_of": {}},
        "D": {"resident": [[0], [6, 7]],
              "prefix_by_stage": [[[0]], [[6]]],
              "extra_by_stage": [[[]], [[7]]],
              "slot_of": {}},
    },
}
_cold = [g for g in range(12) if g not in (0, 6)]
SLOT = {g: i for i, g in enumerate(_cold)}
for ph in ("P", "D"):
    KARTE["phases"][ph]["slot_of"] = {str(g): s for g, s in SLOT.items()}


def _layer(rank=1):
    lo = 6 * rank
    return types.SimpleNamespace(
        layer_id=LAYER, num_local_experts=7, num_experts=12,
        _expert_shard_generic=True, _gguf_expert_range=(lo, lo + 6),
        moe_tp_rank=rank,
        w13_weight_packed=torch.zeros(7, 2), w2_weight_packed=torch.zeros(7, 3))


class _Env(unittest.TestCase):
    def setUp(self):
        if sa is not None:
            sa.reset_for_tests()
        self.store = tempfile.mkdtemp(prefix="bootzeit_h2_store_")
        # P (tp=1 -> rank 0) published its rows
        w13 = sorted(SLOT.values())
        w2 = sorted(s for g, s in SLOT.items() if g != 9)
        for attr, rows in (("w13_weight_packed", w13), ("w2_weight_packed", w2)):
            open(os.path.join(self.store, f"L0-{attr}.bin"), "wb").close()
            with open(os.path.join(self.store, f"L0-{attr}.bin.r0.written.json"), "w") as fh:
                json.dump({"rank": 0, "rows": rows}, fh)
        self._env = mock.patch.dict(os.environ, {
            "SGLANG_WEG2_GROUP": "D", "SGLANG_MOE_EXPERT_STORE_DIR": self.store})
        self._env.start()
        self._map = mock.patch("sglang.srt.layers.moe.expert_store.expert_map",
                               return_value=KARTE)
        self._map.start()

    def tearDown(self):
        self._map.stop()
        self._env.stop()


class TestVetoSet(_Env):
    def test_vetoes_cold_published_rows_only(self):
        self.assertIsNotNone(sa, "store_adopt missing (base)")
        self.assertEqual(sa.vetoed_global_ids(_layer()), frozenset({8, 10, 11}))

    def test_inert_on_group_p_and_when_disabled(self):
        self.assertIsNotNone(sa, "store_adopt missing (base)")
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "P"}):
            self.assertEqual(sa.vetoed_global_ids(_layer()), frozenset())
        from sglang.srt.environ import envs

        with envs.SGLANG_WEG2_ENABLE_D_STORE_ADOPT.override(False):
            self.assertEqual(sa.vetoed_global_ids(_layer()), frozenset())

    def test_snapshot_is_taken_once_before_d_writes(self):
        """A D rank overwriting the r0 sentinel later must not shrink the set."""
        self.assertIsNotNone(sa, "store_adopt missing (base)")
        first = sa.vetoed_global_ids(_layer())
        with open(os.path.join(self.store, "L0-w13_weight_packed.bin.r0.written.json"), "w") as fh:
            json.dump({"rank": 0, "rows": []}, fh)
        lay = _layer()
        self.assertEqual(sa.vetoed_global_ids(lay), first)


class TestPresplitFilter(_Env):
    def test_vetoed_rows_are_not_rewritten(self):
        self.assertIsNotNone(sa, "store_adopt missing (base)")
        lay = _layer()
        sa.vetoed_global_ids(lay)
        # local -> slot of the cold ids 8..11 (local = g - 6 + 1)
        rows = {g - 5: SLOT[g] for g in (8, 9, 10, 11)}
        keep, n = sa.filter_store_rows(lay, "w13_weight_packed", rows,
                                       resident_local=(0, 1, 2), lo=6, pad=True)
        self.assertEqual(n, 3)
        self.assertEqual(keep, {4: SLOT[9]})

    def test_vetoed_resident_is_refused(self):
        self.assertIsNotNone(sa, "store_adopt missing (base)")
        lay = _layer()
        sa.vetoed_global_ids(lay)
        with self.assertRaises(sa.StoreAdoptBroken):
            sa.filter_store_rows(lay, "w13_weight_packed", {3: SLOT[8]},
                                 resident_local=(0, 3), lo=6, pad=True)

    def test_presplit_writes_the_filtered_rows(self):
        import inspect

        from sglang.srt.layers.moe import expert_offload as eo

        src = inspect.getsource(eo.presplit_expert_offload_after_repack)
        self.assertIn("filter_store_rows", src)
        self.assertIn("rows=_rows_w", src)


class TestLoaderVeto(_Env):
    """The checkpoint tensor of a vetoed expert is never read."""

    def _model(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration as C

        lay = _layer()
        lay._gguf_expert_shard = True
        ns = {k: getattr(C, k) for k in (
            "_EXPERT_ID_RE", "weight_name_needed", "_owned_expert_range",
            "_num_routed_experts_for_form_a", "_ple_ngram_embedding",
            ) if hasattr(C, k)}
        Stub = type("Stub", (), ns)
        m = Stub()
        m.language_model_only = True
        m.config = types.SimpleNamespace(num_hidden_layers=1, num_experts=12)
        m.model = types.SimpleNamespace(
            start_layer=0, end_layer=1,
            layers=[types.SimpleNamespace(mlp=types.SimpleNamespace(experts=lay))])
        return m

    def test_weight_name_needed_skips_adopted_experts(self):
        m = self._model()
        name = "model.language_model.layers.0.mlp.experts.{}.down_proj.weight_packed"
        got = {g: bool(m.weight_name_needed(name.format(g))) for g in range(6, 12)}
        self.assertEqual(got, {6: True, 7: True, 8: False, 9: True, 10: False, 11: False})
        # foreign experts stay vetoed as before
        self.assertFalse(m.weight_name_needed(name.format(2)))


if __name__ == "__main__":
    unittest.main()
