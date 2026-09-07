# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, spec T15/T16): the P-side scope.

The MTP head on group P's last stage never runs with a ``PPMissingLayer``
embedding (an ``nn.Identity`` that would hand int64 token ids on as a
``[T, 5120]`` embedding); ``draft_pp_scope`` publishes the single-rank pp
group only for the duration of the scope; the host ledger carries both
draft host pool terms.
"""

import os
import time
import types
import unittest

import torch

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, Qwen3_5MtpEmbeddingAbsent
from sglang.srt.weg2 import host_ledger
from sglang.test.test_utils import CustomTestCase


def _rank_main(rank: int, world: int, port: int, q) -> None:
    """One P stage of boot weg2dk2, on gloo/CPU: every rank builds the draft
    groups (collective), ONLY the last stage constructs the producer's
    collectives inside the scope -- `broadcast_pyobj` over the world group
    exactly as `TpModelWorker.__init__` does (tp_worker.py:316) and the
    world barrier of `_profile_available_bytes`
    (model_runner_kv_cache_mixin.py:775) -- and then every stage enters the
    target's own world barrier. At 35bdc9e310 the last stage sits in a
    3-rank broadcast while the other two sit in a barrier: the dk2 wedge."""
    import os
    from datetime import timedelta

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch.distributed as dist

    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.utils.common import broadcast_pyobj

    try:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=world,
            rank=rank,
            timeout=timedelta(seconds=20),
        )
        ps._WORLD = ps.init_world_group(list(range(world)), 0, "gloo")
        ps._PP = ps.init_model_parallel_group(
            [list(range(world))], 0, "gloo", use_pynccl=False,
            use_custom_allreduce=False, group_name="pp",
        )
        ps._TP = ps.init_model_parallel_group(
            [[r] for r in range(world)], 0, "gloo", use_pynccl=False,
            use_custom_allreduce=False, group_name="tp",
        )
        ps.initialize_draft_pp_group(local_rank=0, backend="gloo")
        q.put((rank, "built", None))
        seed = None
        if ps.get_pp_group().is_last_rank:
            with ps.draft_pp_scope():
                wg = ps.get_world_group()
                seed = broadcast_pyobj([4242], 0, wg.cpu_group, src=wg.ranks[0])[0]
                if wg.world_size > 1:
                    dist.barrier(group=wg.cpu_group)
        dist.barrier(group=ps.get_world_group().cpu_group)
        q.put((rank, "ok", seed))
    except BaseException as e:  # noqa: BLE001 - the verdict travels the queue
        q.put((rank, f"{type(e).__name__}: {e}"[:300], None))
        raise


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
        self._saved = (ps._PP, ps._DRAFT_PP, ps._DRAFT_PP_ACTIVE, ps._WORLD, ps._DRAFT_WORLD)
        self.primary = object()
        self.draft = object()
        ps._PP = self.primary
        ps._DRAFT_PP = self.draft
        ps._WORLD = object()
        ps._DRAFT_WORLD = object()  # fix 2: S7 covers both axes
        ps._DRAFT_PP_ACTIVE = False

    def tearDown(self):
        ps._PP, ps._DRAFT_PP, ps._DRAFT_PP_ACTIVE, ps._WORLD, ps._DRAFT_WORLD = self._saved

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


    def test_fix2_scope_publishes_a_single_rank_world_group_too(self):
        # Boot weg2dk2 killer: the draft build's TpModelWorker.__init__
        # broadcasts its seed on get_world_group() (tp_worker.py:316) and
        # _profile_available_bytes barriers on it; the scope masked only the
        # pp axis, so the last stage entered a 3-rank collective alone.
        saved = ps._WORLD, getattr(ps, "_DRAFT_WORLD", None)
        primary_world, draft_world = object(), object()
        ps._WORLD = primary_world
        ps._DRAFT_WORLD = draft_world
        try:
            self.assertIs(ps.get_world_group(), primary_world)
            with ps.draft_pp_scope():
                self.assertIs(ps.get_world_group(), draft_world)
                self.assertIs(ps.get_pp_group(), self.draft)
            self.assertIs(ps.get_world_group(), primary_world)
            with self.assertRaises(RuntimeError):
                with ps.draft_pp_scope():
                    raise RuntimeError("inside")
            self.assertIs(ps.get_world_group(), primary_world)
            # S7 covers BOTH axes: a pp group without its world twin refuses
            ps._DRAFT_WORLD = None
            with self.assertRaisesRegex(RuntimeError, "S7"):
                with ps.draft_pp_scope():
                    pass
            self.assertIs(ps.get_world_group(), primary_world)
        finally:
            ps._WORLD, ps._DRAFT_WORLD = saved

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
        self._saved = (ps._PP, ps._DRAFT_PP, ps._DRAFT_PP_ACTIVE, ps._WORLD, ps._DRAFT_WORLD)
        ps._PP = types.SimpleNamespace(is_last_rank=True, rank_in_group=2, world_size=3)
        ps._DRAFT_PP = types.SimpleNamespace(is_last_rank=True, rank_in_group=0, world_size=1)
        ps._WORLD = types.SimpleNamespace(world_size=3, ranks=[0, 1, 2])
        ps._DRAFT_WORLD = types.SimpleNamespace(world_size=1, ranks=[2])
        ps._DRAFT_PP_ACTIVE = False
        self._env = os.environ.get("SGLANG_PP_LAYER_PARTITION")
        os.environ["SGLANG_PP_LAYER_PARTITION"] = "32,18,14"

    def tearDown(self):
        ps._PP, ps._DRAFT_PP, ps._DRAFT_PP_ACTIVE, ps._WORLD, ps._DRAFT_WORLD = self._saved
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
                seen["world_size"] = ps.get_world_group().world_size  # fix 2
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
        self.assertEqual(seen, {"pp_size": 1, "world_size": 1, "args_pp_size": 1, "indices": (0, 1)})
        self.assertEqual(ps.get_world_group().world_size, 3)  # restored after the build
        self.assertTrue(producer.draft_worker.draft_kv_only)
        self.assertEqual(os.environ.get("SGLANG_PP_LAYER_PARTITION"), "32,18,14")


class TestProducerCollectivesOnTheLastStageOnly(CustomTestCase):
    """The multi-process test both dk1 and dk2 lacked: three gloo ranks,
    the producer's collectives on the last stage only (boot weg2dk2 stacks:
    PP2 in broadcast_pyobj, PP0/PP1 in dist.barrier). Real reason for a
    process test: a collective mismatch is invisible in one process."""

    def test_fix2_last_stage_builds_alone_and_the_group_still_meets(self):
        import multiprocessing as mp
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        world = 3
        procs = [ctx.Process(target=_rank_main, args=(r, world, port, q)) for r in range(world)]
        for pr in procs:
            pr.start()
        deadline = time.monotonic() + 150
        for pr in procs:
            pr.join(max(1.0, deadline - time.monotonic()))
        verdicts = {}
        while not q.empty():
            r, v, seed = q.get_nowait()
            if v != "built":
                verdicts[r] = (v, seed)
        alive = [pr.pid for pr in procs if pr.is_alive()]
        for pr in procs:
            if pr.is_alive():
                pr.kill()
        codes = [pr.exitcode for pr in procs]
        self.assertEqual(alive, [], f"wedged ranks (the dk2 shape): {alive}; verdicts={verdicts}")
        self.assertEqual(codes, [0] * world, f"exit codes {codes}; verdicts={verdicts}")
        self.assertEqual({r for r in verdicts}, set(range(world)), verdicts)
        self.assertTrue(all(v == "ok" for v, _ in verdicts.values()), verdicts)
        self.assertEqual(verdicts[world - 1][1], 4242, "the last stage's seed broadcast is the identity on a group of one")


class TestResidentEmbeddingLoad(CustomTestCase):
    """Placement A pays the embedding ONCE: the checkpoint rows go INTO the
    tensor the MTP build materialised, the head's own lm_head is released
    (not left to GC), and a checkpoint that does not match the built
    parameters is refused by name instead of cast (int8 codes into a bf16
    table without their scale = token soup)."""

    def _producer(self, embed, old_head, target_head):
        from sglang.srt.speculative import draft_kv_producer as dkp

        draft_model = types.SimpleNamespace(model=types.SimpleNamespace(embed_tokens=embed), lm_head=old_head)
        draft_model.set_lm_head_from_target = lambda h: setattr(draft_model, "lm_head", h)
        producer = dkp.DraftKvProducer.__new__(dkp.DraftKvProducer)
        producer.draft_runner = types.SimpleNamespace(model=draft_model)
        producer.draft_worker = types.SimpleNamespace(
            target_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(model=types.SimpleNamespace(lm_head=target_head)))
        )
        return producer, draft_model

    def test_fix2_loads_into_the_built_tensor_and_releases_the_own_head(self):
        import weakref
        from unittest import mock

        from sglang.srt.speculative import draft_kv_producer as dkp

        embed = torch.nn.Module()
        embed.weight = torch.nn.Parameter(torch.zeros(8, 4, dtype=torch.int8), requires_grad=False)
        embed.weight_scale = torch.nn.Parameter(torch.zeros(8, 1, dtype=torch.bfloat16), requires_grad=False)
        loads = []
        for n, p_ in embed.named_parameters():
            p_.weight_loader = (lambda n: (lambda param, t: (loads.append(n), param.data.copy_(t))))(n)
        built_weight = embed.weight
        old_head = torch.nn.Module()
        old_head.weight = torch.nn.Parameter(torch.zeros(8, 4, dtype=torch.bfloat16), requires_grad=False)
        target_head = object()
        producer, draft_model = self._producer(embed, old_head, target_head)
        ref = weakref.ref(old_head)
        ck = [
            ("model.language_model.embed_tokens.weight", torch.full((8, 4), 3, dtype=torch.int8)),
            ("model.language_model.embed_tokens.weight_scale", torch.full((8, 1), 0.5, dtype=torch.bfloat16)),
        ]
        with mock.patch.object(dkp, "_iter_checkpoint_tensors", lambda path, needle: iter(ck)), \
                mock.patch.object(torch.cuda, "empty_cache") as ec:
            mib = producer.load_resident_embedding("/nonexistent")
        self.assertEqual(sorted(loads), ["weight", "weight_scale"])
        self.assertIs(draft_model.model.embed_tokens.weight, built_weight)  # into the built tensor
        self.assertTrue(bool((built_weight == 3).all()))
        self.assertIs(draft_model.lm_head, target_head)
        del old_head
        self.assertIsNone(ref(), "the head's own lm_head must be released, not left to a GC that the profiler runs before")
        self.assertTrue(ec.called, "the released transient must reach the driver before the profiler reads free VRAM")
        self.assertAlmostEqual(mib, (8 * 4 + 8 * 2) / float(2**20))

    def test_fix2_refuses_a_dtype_cast_and_an_orphan_scale(self):
        from unittest import mock

        from sglang.srt.speculative import draft_kv_producer as dkp

        embed = torch.nn.Module()
        embed.weight = torch.nn.Parameter(torch.zeros(8, 4, dtype=torch.bfloat16), requires_grad=False)
        producer, _ = self._producer(embed, torch.nn.Module(), object())
        ck = [("model.language_model.embed_tokens.weight", torch.zeros(8, 4, dtype=torch.int8))]
        with mock.patch.object(dkp, "_iter_checkpoint_tensors", lambda path, needle: iter(ck)):
            with self.assertRaisesRegex(RuntimeError, "int8.*bfloat16|dtype"):
                producer.load_resident_embedding("/nonexistent")
        ck = [
            ("model.language_model.embed_tokens.weight", torch.zeros(8, 4, dtype=torch.bfloat16)),
            ("model.language_model.embed_tokens.weight_scale", torch.zeros(8, 1, dtype=torch.bfloat16)),
        ]
        with mock.patch.object(dkp, "_iter_checkpoint_tensors", lambda path, needle: iter(ck)):
            with self.assertRaisesRegex(RuntimeError, "weight_scale"):
                producer.load_resident_embedding("/nonexistent")


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
