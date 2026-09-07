# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, spec T15/T16): the P-side scope.

The MTP head on group P's last stage never runs with a ``PPMissingLayer``
embedding (an ``nn.Identity`` that would hand int64 token ids on as a
``[T, 5120]`` embedding); ``draft_pp_scope`` publishes the single-rank pp
group only for the duration of the scope; the host ledger carries both
draft host pool terms.
"""

import os
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

    def test_fix1_scope_masks_the_targets_pp_layer_env(self):
        # Boot weg2dk1 killer: the TARGET exports SGLANG_PP_LAYER_PARTITION
        # (server_args.py, --pp-stage-ratio) process-wide, and the draft
        # model build inside the scope reads it back through get_pp_indices
        # with pp_size=1 -> `len(partitions)=3 does not match pp_size=1`.
        # The scope is the pp_size-1 world; it masks both layer-split env
        # vars and restores them on every exit path, nested included.
        from sglang.srt.distributed.utils import PP_LAYER_SET_ENV, get_pp_indices

        part, lset = "SGLANG_PP_LAYER_PARTITION", PP_LAYER_SET_ENV
        saved = {k: os.environ.get(k) for k in (part, lset)}

        def restore():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(restore)
        os.environ[part] = "32,18,14"
        os.environ[lset] = "0-31;32-49;50-63"
        with self.assertRaisesRegex(ValueError, "does not match pp_size=1"):
            get_pp_indices(1, 0, 1)
        with ps.draft_pp_scope():
            self.assertIsNone(os.environ.get(part))
            self.assertIsNone(os.environ.get(lset))
            self.assertEqual(get_pp_indices(1, 0, 1), (0, 1))
            with ps.draft_pp_scope():  # nested (alloc/init hooks re-enter)
                self.assertIsNone(os.environ.get(part))
            self.assertIsNone(os.environ.get(part))
            self.assertIs(ps.get_pp_group(), self.draft)
        self.assertEqual(os.environ.get(part), "32,18,14")
        self.assertEqual(os.environ.get(lset), "0-31;32-49;50-63")
        with self.assertRaises(RuntimeError):
            with ps.draft_pp_scope():
                raise RuntimeError("inside")
        self.assertEqual(os.environ.get(part), "32,18,14")
        self.assertEqual(os.environ.get(lset), "0-31;32-49;50-63")
        # unset before the scope stays unset after it
        os.environ.pop(part)
        os.environ.pop(lset)
        with ps.draft_pp_scope():
            self.assertIsNone(os.environ.get(part))
        self.assertIsNone(os.environ.get(part))
        self.assertIsNone(os.environ.get(lset))


class TestProducerBuildUnderTheTargetsPartition(CustomTestCase):
    """The blind spot of boot weg2dk1: the producer test never exported the
    target's partition string. This one builds the producer exactly the way
    PP2 does -- inside the scope, with the target's 3-way partition in the
    environment -- against a draft-worker double that runs the one call the
    real build runs (`make_layers` -> `get_pp_indices` under pp_size=1)."""

    def setUp(self):
        self._saved = (ps._PP, ps._DRAFT_PP, ps._DRAFT_PP_ACTIVE)
        ps._PP = types.SimpleNamespace(is_last_rank=True, rank_in_group=2, world_size=3)
        ps._DRAFT_PP = types.SimpleNamespace(is_last_rank=True, rank_in_group=0, world_size=1)
        ps._DRAFT_PP_ACTIVE = False
        self._env = os.environ.get("SGLANG_PP_LAYER_PARTITION")
        os.environ["SGLANG_PP_LAYER_PARTITION"] = "32,18,14"

    def tearDown(self):
        ps._PP, ps._DRAFT_PP, ps._DRAFT_PP_ACTIVE = self._saved
        if self._env is None:
            os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
        else:
            os.environ["SGLANG_PP_LAYER_PARTITION"] = self._env

    def test_fix1_producer_builds_its_one_layer_under_the_targets_partition(self):
        from unittest import mock

        from sglang.srt.distributed.utils import get_pp_indices
        from sglang.srt.speculative import draft_kv_producer as dkp
        from sglang.srt.speculative import eagle_worker_v2

        seen = {}

        class FakeDraftWorker:
            def __init__(self, server_args, **kw):
                pp = ps.get_pp_group()
                seen["pp_size"] = pp.world_size
                seen["args_pp_size"] = server_args.pp_size
                # what qwen3_5_mtp -> make_layers -> get_pp_indices does
                seen["indices"] = get_pp_indices(1, pp.rank_in_group, pp.world_size)
                self.draft_runner = object()

        class Args(types.SimpleNamespace):
            def override(self, source, **fields):
                for k, v in fields.items():
                    setattr(self, k, v)

        scheduler = types.SimpleNamespace(
            server_args=Args(pp_size=3, pp_stage_ratio="32,18,14", pp_attn_stage_ratio="8,4,4",
                             pp_layer_ratio=None, skip_tokenizer_init=False),
            ps=types.SimpleNamespace(gpu_id=0, tp_rank=0, dp_rank=0, moe_ep_rank=0,
                                     attn_cp_rank=0, moe_dp_rank=0),
            nccl_port=0,
            tp_worker=object(),
        )
        with mock.patch.object(eagle_worker_v2, "EagleDraftWorker", FakeDraftWorker):
            producer = dkp.DraftKvProducer(scheduler, "NEXTN")
        self.assertEqual(seen, {"pp_size": 1, "args_pp_size": 1, "indices": (0, 1)})
        self.assertTrue(producer.draft_worker.draft_kv_only)
        self.assertEqual(os.environ.get("SGLANG_PP_LAYER_PARTITION"), "32,18,14")


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
