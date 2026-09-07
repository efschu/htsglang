# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, spec T15/T16): the P-side scope.

The MTP head on group P's last stage never runs with a ``PPMissingLayer``
embedding (an ``nn.Identity`` that would hand int64 token ids on as a
``[T, 5120]`` embedding); ``draft_pp_scope`` publishes the single-rank pp
group only for the duration of the scope; the host ledger carries both
draft host pool terms.
"""

import types
import unittest

import torch

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, Qwen3_5MtpEmbeddingAbsent
from sglang.srt.weg2 import host_ledger
from sglang.test.test_utils import CustomTestCase


class TestEmbeddingRefusal(CustomTestCase):
    def test_t15_mtp_forward_refuses_a_missing_embedding_before_any_tensor_op(self):
        calls = []
        head = types.SimpleNamespace(
            model=types.SimpleNamespace(embed_tokens=PPMissingLayer()),
            quant_config=object(),
            pre_fc_norm_embedding=lambda x: calls.append("norm") or x,
        )
        fb = types.SimpleNamespace(
            mm_input_embeds=None,
            forward_mode=types.SimpleNamespace(is_extend=lambda: False, is_idle=lambda: False),
            spec_info=types.SimpleNamespace(hidden_states=torch.zeros(1, 4)),
        )
        with self.assertRaisesRegex(Qwen3_5MtpEmbeddingAbsent, "never an Identity"):
            Qwen3_5ForCausalLMMTP.forward(head, torch.tensor([1, 2]), torch.tensor([0, 1]), fb)
        self.assertEqual(calls, [])


class TestDraftPpScope(CustomTestCase):
    def setUp(self):
        self._saved = (ps._PP, ps._DRAFT_PP, ps._DRAFT_PP_ACTIVE)
        self.primary = object()
        self.draft = object()
        ps._PP = self.primary
        ps._DRAFT_PP = self.draft
        ps._DRAFT_PP_ACTIVE = False

    def tearDown(self):
        ps._PP, ps._DRAFT_PP, ps._DRAFT_PP_ACTIVE = self._saved

    def test_t16_scope_publishes_the_single_rank_group_only_inside(self):
        self.assertIs(ps.get_pp_group(), self.primary)
        with ps.draft_pp_scope():
            self.assertIs(ps.get_pp_group(), self.draft)
        self.assertIs(ps.get_pp_group(), self.primary)
        with self.assertRaises(RuntimeError):
            with ps.draft_pp_scope():
                raise RuntimeError("inside")
        self.assertIs(ps.get_pp_group(), self.primary)
        ps._DRAFT_PP = None
        with self.assertRaises(RuntimeError):
            with ps.draft_pp_scope():
                pass


class TestHostLedgerDraftTerms(CustomTestCase):
    def test_t16_price_carries_both_draft_terms_and_choose_steps_down(self):
        gib = host_ledger.GIB
        arm = host_ledger.price(int(200 * gib), int(150 * gib), 1, 2400, weight_chunks=8)
        self.assertIn("draft_host_p_gib", arm.terms)
        self.assertIn("draft_host_d_gib", arm.terms)
        self.assertAlmostEqual(arm.terms["draft_host_p_gib"] * 1024, 119.2, delta=0.2)
        self.assertAlmostEqual(arm.terms["draft_host_d_gib"] * 1024, 59.6, delta=0.2)
        self.assertGreater(host_ledger.STORE_DRAFT_FRACTION, 0.06)
        self.assertLess(host_ledger.STORE_DRAFT_FRACTION, 0.07)
        # an arm whose launch leftover would be positive WITHOUT the draft terms
        # but negative with them steps down to the next arm
        draft_terms = arm.terms["draft_host_p_gib"] + arm.terms["draft_host_d_gib"]
        # memavail chosen so that arm (1, 2400)'s launch leftover is positive
        # WITHOUT the draft terms and negative WITH them: half a draft term
        # below the break-even of the priced form above.
        memavail = int((150 - arm.launch_leftover_gib - draft_terms / 2) * gib)
        chosen, _store, _lines = host_ledger.choose(
            int(200 * gib), memavail, store_min_gib=0.0, weight_chunks=8
        )
        self.assertNotEqual((chosen.s_gb, chosen.m_mib), (1, 2400))


class TestLauncherW10(CustomTestCase):
    def test_w10_drafter_identity_gate_reads_both_logs(self):
        import os
        import tempfile

        from sglang.srt.weg2.launcher import check_drafter_identity

        reg = "HiCache draft KV registered: MHATokenToKVPool (host 30518 slots), owner_phase=None, binding_generation=None, drafter={}\n"
        act = "#706 canonical DRAFT page active: 1 slot, 2048 B; heads [0,2) of 4; extents [(0, 512), (1024, 512)]; draft keys carry content+drafter only (suffix=_x) layout=v{} drafter={}\n"
        with tempfile.TemporaryDirectory() as d:
            p, q = os.path.join(d, "P.log"), os.path.join(d, "D.log")
            with open(p, "w") as f:
                f.write(reg.format("a30db4b7c362c786") + act.format(1, "a30db4b7c362c786"))
            with open(q, "w") as f:
                f.write((reg.format("a30db4b7c362c786") + act.format(1, "a30db4b7c362c786")) * 3)
            out = check_drafter_identity(p, q)
            self.assertTrue(out["match"], out)
            self.assertEqual((out["n_P"], out["n_D"]), (1, 3))
            with open(p, "w") as f:
                f.write(reg.format("09136fad58d6849c") + act.format(1, "09136fad58d6849c"))
            self.assertFalse(check_drafter_identity(p, q)["match"])
            with open(p, "w") as f:
                f.write("")  # P registered no drafter at all: the pre-fix shape
            self.assertFalse(check_drafter_identity(p, q)["match"])


if __name__ == "__main__":
    unittest.main()
