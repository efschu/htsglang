# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233) -- FIX 3, the R1 hook itself.

Boot weg2dk3 (bc31554f90) killed all three P ranks on the FIRST prefill
chunk with ``AttributeError: 'ModelRunner' object has no attribute
'eager_runner'`` and wrote zero draft pages, while all 51 desk tests were
green: nothing pinned C7 -- neither the producer's runner construction, nor
the scheduler hook that drives it. These are those pins.

Every test here is red at bc31554f90:
  * the producer's ``init_cuda_graphs`` returned ``None`` (no runner built),
  * ``_draft_server_args`` did not exist and the copy carried no
    ``disable_draft_cuda_graph``,
  * ``_draft_kv_full_capture`` did not exist (the FULL-capture restore sat
    in ``run_batch`` with no ``finally``),
  * ``load_resident_embedding`` released the never-loaded lm_head by module
    swap only, and reported the NVML free delta as the residue.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase


# --------------------------------------------------------------------------
# R1 / the dk3 killer: the eager-only terminal state is CONSTRUCTED
# --------------------------------------------------------------------------


class _RecordingScope:
    """Stands in for ``draft_pp_scope``; records enter/exit ordering so a
    runner built outside the scope is a failure, not a coincidence."""

    def __init__(self):
        self.events = []

    def __call__(self):
        outer = self

        class _Ctx:
            def __enter__(self):
                outer.events.append("enter")
                return None

            def __exit__(self, *a):
                outer.events.append("exit")
                return False

        return _Ctx()


class TestProducerBuildsTheEagerRunner(CustomTestCase):
    def test_fix3_init_cuda_graphs_calls_the_draft_worker_inside_the_scope(self):
        """The producer must not skip ``init_cuda_graphs``: that call is what
        builds ``eager_runner`` (model_runner.py:1484-1499), and the #656
        item-8 note at model_runner.py:1528-1540 already records this exact
        death. It must also happen INSIDE ``draft_pp_scope`` -- the draft
        runner's geometry is the single-rank group's, not the target's."""
        from sglang.srt.speculative import draft_kv_producer as dkp

        scope = _RecordingScope()
        calls = []
        producer = dkp.DraftKvProducer.__new__(dkp.DraftKvProducer)
        producer.draft_worker = types.SimpleNamespace(
            init_cuda_graphs=lambda: calls.append(list(scope.events))
        )
        with mock.patch(
            "sglang.srt.distributed.parallel_state.draft_pp_scope", scope
        ):
            producer.init_cuda_graphs()
        self.assertEqual(len(calls), 1, "the draft worker's init_cuda_graphs was skipped")
        self.assertEqual(calls[0], ["enter"], "built outside draft_pp_scope")
        self.assertEqual(scope.events, ["enter", "exit"])

    def test_fix3_the_boot_path_actually_calls_the_producers_hook(self):
        """A correct ``init_cuda_graphs`` nobody calls is still no eager
        runner. At bc31554f90 ``init_all_cuda_graphs`` had no producer
        branch at all -- the other two lifecycle hooks were wired, this one
        was the hole, and that is the second half of the dk3 killer."""
        from sglang.srt.managers.scheduler import Scheduler

        called = []
        s = Scheduler.__new__(Scheduler)
        s.tp_worker = types.SimpleNamespace(init_cuda_graphs=lambda: called.append("tp"))
        s.draft_worker = None
        s.draft_kv_producer = types.SimpleNamespace(
            init_cuda_graphs=lambda: called.append("producer")
        )
        Scheduler.init_all_cuda_graphs(s)
        self.assertEqual(called, ["tp", "producer"])

        called.clear()
        s.draft_kv_producer = None
        Scheduler.init_all_cuda_graphs(s)
        self.assertEqual(called, ["tp"], "no producer, no extra call")

    def test_fix3_draft_args_disable_draft_cuda_graph(self):
        """No captures, through the tree's own refusal: the flag
        ``should_capture_draft_graphs`` reads lives on the producer's OWN
        args copy, so the target's graph settings are untouched and no fork
        twin of the eager-only state is built."""
        from sglang.srt.speculative.draft_kv_producer import _draft_server_args

        class _Args:
            def __init__(self):
                self.pp_size = 3
                self.pp_stage_ratio = [32, 18, 14]
                self.pp_attn_stage_ratio = [8, 4, 4]
                self.pp_layer_ratio = [32, 18, 14]
                self.skip_tokenizer_init = False
                self.disable_draft_cuda_graph = False
                self.disable_cuda_graph = False
                self.sources = []

            def override(self, source, **fields):
                self.sources.append(source)
                for k, v in fields.items():
                    setattr(self, k, v)

        base = _Args()
        out = _draft_server_args(base)
        self.assertEqual(out.sources, ["weg2.draft_kv_producer"])
        self.assertTrue(out.disable_draft_cuda_graph)
        self.assertEqual(out.pp_size, 1)
        self.assertIsNone(out.pp_layer_ratio)
        self.assertTrue(out.skip_tokenizer_init)
        # the target's own copy is untouched (deepcopy, not in-place)
        self.assertEqual(base.pp_size, 3)
        self.assertFalse(base.disable_draft_cuda_graph)

        from sglang.srt.speculative.base_spec_worker import should_capture_draft_graphs

        self.assertFalse(
            should_capture_draft_graphs(out),
            "the producer's args must reach the upstream refusal",
        )
        self.assertTrue(should_capture_draft_graphs(base))


def _stub_model_runner(**kw):
    """A ModelRunner instance with only the fields the eager-only terminal
    state and the eager dispatch of ``_forward_raw`` read. Built with
    ``__new__`` on the REAL class so the REAL methods run."""
    from sglang.srt.model_executor.model_runner import ModelRunner

    mr = ModelRunner.__new__(ModelRunner)
    mr.server_args = types.SimpleNamespace(
        disable_draft_cuda_graph=True,
        disable_cuda_graph=False,
        enable_phase_flip=False,
        cuda_graph_config=None,
    )
    mr.is_draft_worker = True
    mr.is_phase_flip_tp_stack = False
    mr.device = "cpu"
    mr.is_weightless_head = False
    mr.is_weightless_worker = False
    mr.hisparse_coordinator = None
    mr.decode_cuda_graph_runner = None
    mr.prefill_cuda_graph_runner = None
    mr.attn_backend = None
    mr.device_timer = None
    mr.prefill_rank_timer = None
    mr._layer_fingerprint = None
    mr._forward_peak = None
    mr.pp_group = types.SimpleNamespace(is_last_rank=True)
    for k, v in kw.items():
        setattr(mr, k, v)
    return mr


class _Mode:
    """Minimal ForwardMode stand-in for one plain extend chunk."""

    def is_cuda_graph(self):
        return False

    def is_cpu_graph(self):
        return False

    def is_decode(self):
        return False

    def is_decode_or_idle(self):
        return False

    def is_target_verify(self):
        return False

    def is_split_prefill(self):
        return False

    def is_plain_prefill(self):
        return True

    def is_extend(self, include_draft_extend_v2=False):
        return True


class TestEagerRunnerTerminalState(CustomTestCase):
    """The producer's runner, driven through the REAL
    ``ModelRunner.init_cuda_graphs``, reaches the eager-only terminal state
    and ``_forward_raw`` dispatches an extend chunk to it. At bc31554f90 the
    producer never called this at all, so ``eager_runner`` did not exist and
    the first chunk raised."""

    def _arm(self):
        from sglang.srt.model_executor import model_runner as mr_mod
        from sglang.srt.speculative import draft_kv_producer as dkp

        real_eager = mr_mod.EagerRunner
        runner = _stub_model_runner()
        seen = []

        def _fake_eager(model_runner):
            er = real_eager.__new__(real_eager)
            er.model_runner = model_runner
            er.execute = lambda fb, **kwargs: (seen.append(fb), "LOGITS")[1]
            return er

        producer = dkp.DraftKvProducer.__new__(dkp.DraftKvProducer)
        producer.draft_worker = types.SimpleNamespace(
            init_cuda_graphs=lambda: runner.init_cuda_graphs(
                capture_decode_cuda_graph=False
            )
        )
        scope = _RecordingScope()
        with mock.patch(
            "sglang.srt.distributed.parallel_state.draft_pp_scope", scope
        ), mock.patch.object(mr_mod, "EagerRunner", _fake_eager), mock.patch.object(
            mr_mod.GraphSharedOutput,
            "create_for_model_runner",
            classmethod(lambda cls, m: None),
        ):
            producer.init_cuda_graphs()
        return runner, seen, real_eager

    def test_fix3_eager_runner_exists_and_the_three_runners_are_aliased(self):
        runner, _seen, real_eager = self._arm()
        self.assertTrue(
            hasattr(runner, "eager_runner"),
            "the eager path is not the absence of the graph path -- it has to be built",
        )
        self.assertIsInstance(runner.eager_runner, real_eager)
        self.assertIs(runner.prefill_cuda_graph_runner, runner.eager_runner)
        self.assertIs(runner.decode_cuda_graph_runner, runner.eager_runner)
        self.assertEqual(runner.graph_mem_usage, 0)

    def test_fix3_forward_raw_dispatches_an_extend_chunk_to_the_eager_runner(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        runner, seen, _real = self._arm()
        runner._prepare_eager_forward_batch = lambda fb: None
        runner._maybe_execute_deferred_mamba_cow_and_clear = lambda fb: None
        fb = types.SimpleNamespace(
            forward_mode=_Mode(),
            input_ids=torch.zeros(4, dtype=torch.long),
            global_num_tokens_cpu=None,
        )
        out = ModelRunner._forward_raw(runner, fb, None)
        self.assertEqual(seen, [fb], "the extend chunk did not reach eager_runner.execute")
        self.assertEqual(out.logits_output, "LOGITS")
        self.assertFalse(out.can_run_graph)


# --------------------------------------------------------------------------
# C7: the scheduler hook that drives the producer (mutant M8's blind spot)
# --------------------------------------------------------------------------


def _sched(**kw):
    from sglang.srt.managers.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s.draft_kv_producer = kw.pop("producer", object())
    s.draft_kv_producer_algorithm = kw.pop("algorithm", "NEXTN")
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _batch(mode, **kw):
    b = types.SimpleNamespace(
        forward_mode=mode,
        reqs=[types.SimpleNamespace(rid="r0")],
        chunked_req=None,
        spec_algorithm="NONE",
        spec_info=None,
        capture_hidden_mode="LAST",
    )
    for k, v in kw.items():
        setattr(b, k, v)
    return b


class TestDraftKvProducerWants(CustomTestCase):
    """C7 fires on the last stage's extend chunk and on nothing else. Mutant
    M8 (`return False`) left all 51 tests green at bc31554f90; it is red
    here."""

    def _wants(self, producer, mode):
        from sglang.srt.managers.scheduler import Scheduler

        return Scheduler._draft_kv_producer_wants(
            _sched(producer=producer), _batch(mode)
        )

    def test_fix3_wants_matrix(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.assertTrue(self._wants(object(), ForwardMode.EXTEND))
        self.assertTrue(self._wants(object(), ForwardMode.MIXED))
        self.assertFalse(self._wants(None, ForwardMode.EXTEND), "no producer, no C7")
        self.assertFalse(self._wants(object(), ForwardMode.DECODE))
        self.assertFalse(self._wants(object(), ForwardMode.IDLE))


class TestDraftKvFullCapture(CustomTestCase):
    def _cm(self, batch):
        from sglang.srt.managers.scheduler import Scheduler

        return Scheduler._draft_kv_full_capture(_sched(), batch)

    def test_fix3_full_capture_is_set_and_restored(self):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        b = _batch(None, capture_hidden_mode=CaptureHiddenMode.LAST)
        with self._cm(b):
            self.assertIs(b.capture_hidden_mode, CaptureHiddenMode.FULL)
        self.assertIs(b.capture_hidden_mode, CaptureHiddenMode.LAST)

    def test_fix3_full_capture_is_restored_when_the_producer_raises(self):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        b = _batch(None, capture_hidden_mode=CaptureHiddenMode.LAST)
        with self.assertRaises(RuntimeError):
            with self._cm(b):
                raise RuntimeError("producer died")
        self.assertIs(
            b.capture_hidden_mode,
            CaptureHiddenMode.LAST,
            "a producer that raises must not leave the batch in FULL capture",
        )


class TestDraftKvProduce(CustomTestCase):
    def _produce(self, sched, batch, result):
        from sglang.srt.managers.scheduler import Scheduler

        return Scheduler._draft_kv_produce(sched, batch, result)

    def _result(self, hidden="H", ids="N"):
        return types.SimpleNamespace(
            logits_output=types.SimpleNamespace(hidden_states=hidden),
            next_token_ids=ids,
        )

    def test_fix3_produce_runs_the_producer_and_restores_spec_state(self):
        seen = {}
        producer = types.SimpleNamespace(
            _chunks=0,
            _rows=0,
            produce=lambda b, h, n: (
                seen.update(
                    hidden=h, ids=n, algo=b.spec_algorithm, chunks=producer._chunks
                ),
                {"rows": 7, "ms": 1.0, "peak_mib": 2.0},
            )[1],
        )
        s = _sched(producer=producer, algorithm="NEXTN")
        b = _batch(None, spec_algorithm="NONE", spec_info="SI")
        self._produce(s, b, self._result())
        self.assertEqual(seen["hidden"], "H")
        self.assertEqual(seen["ids"], "N")
        self.assertEqual(seen["algo"], "NEXTN", "the producer runs under its own algorithm")
        self.assertEqual(b.spec_algorithm, "NONE", "spec_algorithm not restored")
        self.assertEqual(b.spec_info, "SI", "spec_info not restored")

    def test_fix3_produce_restores_spec_state_when_the_producer_raises(self):
        def _boom(b, h, n):
            raise RuntimeError("draft extend died")

        producer = types.SimpleNamespace(_chunks=0, _rows=0, produce=_boom)
        s = _sched(producer=producer, algorithm="NEXTN")
        b = _batch(None, spec_algorithm="NONE", spec_info="SI")
        with self.assertRaises(RuntimeError):
            self._produce(s, b, self._result())
        self.assertEqual(b.spec_algorithm, "NONE")
        self.assertEqual(b.spec_info, "SI")

    def test_fix3_a_chunk_without_full_hidden_states_raises_never_skips(self):
        producer = types.SimpleNamespace(
            _chunks=0, _rows=0, produce=lambda *a: self.fail("must not run")
        )
        s = _sched(producer=producer)
        for result in (self._result(hidden=None), self._result(ids=None)):
            with self.assertRaisesRegex(RuntimeError, "never silently skipped"):
                self._produce(s, _batch(None), result)


# --------------------------------------------------------------------------
# FIX 3 / W11: the never-loaded lm_head is released, and the residue is
# measured by an instrument that can see the release
# --------------------------------------------------------------------------


def _head(rows=8, cols=4, dtype=torch.bfloat16):
    m = torch.nn.Module()
    m.weight = torch.nn.Parameter(torch.zeros(rows, cols, dtype=dtype), requires_grad=False)
    return m


class TestHeadRelease(CustomTestCase):
    def test_fix3_drop_parameters_frees_the_table_and_reports_its_mib(self):
        from sglang.srt.speculative.draft_kv_producer import _drop_parameters

        head = _head(64, 32)
        w = head.weight
        ref = w.data_ptr()
        mib = _drop_parameters(head)
        self.assertAlmostEqual(mib, 64 * 32 * 2 / float(2**20))
        self.assertEqual(list(head.named_parameters()), [])
        self.assertFalse(hasattr(head, "weight"))
        del w, ref

    def _producer(self, embed, old_head, target_head):
        from sglang.srt.speculative import draft_kv_producer as dkp

        draft_model = torch.nn.Module()
        draft_model.model = torch.nn.Module()
        draft_model.model.embed_tokens = embed
        draft_model.lm_head = old_head
        draft_model.set_lm_head_from_target = lambda h: setattr(draft_model, "lm_head", h)
        producer = dkp.DraftKvProducer.__new__(dkp.DraftKvProducer)
        producer._free_before_mib = -1.0
        producer.resident_mib = -1.0
        producer.nvml_delta_mib = -1.0
        producer.head_released_mib = 0.0
        producer.embed_dtype = "?"
        producer.draft_runner = types.SimpleNamespace(model=draft_model)
        producer.draft_worker = types.SimpleNamespace(
            target_worker=types.SimpleNamespace(
                model_runner=types.SimpleNamespace(
                    model=types.SimpleNamespace(lm_head=target_head)
                )
            )
        )
        return producer, draft_model

    def _embed(self):
        embed = torch.nn.Module()
        embed.weight = torch.nn.Parameter(
            torch.zeros(8, 4, dtype=torch.int8), requires_grad=False
        )
        for n, p in embed.named_parameters():
            p.weight_loader = lambda param, t: param.data.copy_(t)
        return embed

    def _ck(self):
        return [("model.language_model.embed_tokens.weight", torch.full((8, 4), 3, dtype=torch.int8))]

    def test_fix3_the_own_heads_parameters_are_deleted_not_just_unbound(self):
        """A module swap frees nothing while any other holder remains -- boot
        weg2dk3 shipped the swap and measured a residue equal to the build.
        The upstream release form deletes the parameter."""
        from sglang.srt.speculative import draft_kv_producer as dkp

        embed = self._embed()
        old_head = _head(64, 32)
        keeper = old_head  # a second holder, exactly what the swap cannot beat
        target_head = _head(64, 32)
        producer, draft_model = self._producer(embed, old_head, target_head)
        with mock.patch.object(
            dkp, "_iter_checkpoint_tensors", lambda p, n: iter(self._ck())
        ), mock.patch.object(torch.cuda, "empty_cache"):
            producer.load_resident_embedding("/nonexistent")
        self.assertIs(draft_model.lm_head, target_head)
        self.assertEqual(
            list(keeper.named_parameters()),
            [],
            "the never-loaded table is still live behind the second holder",
        )
        self.assertAlmostEqual(producer.head_released_mib, 64 * 32 * 2 / float(2**20))

    def test_fix3_a_tied_head_is_never_gutted(self):
        """Under tie_word_embeddings the head IS the resident embedding this
        method just loaded; gutting it would hand the producer token soup."""
        from sglang.srt.speculative import draft_kv_producer as dkp

        embed = self._embed()
        producer, draft_model = self._producer(embed, embed, None)
        # tie: set_lm_head_from_target returns early, lm_head stays the embed
        draft_model.set_lm_head_from_target = lambda h: None
        with mock.patch.object(
            dkp, "_iter_checkpoint_tensors", lambda p, n: iter(self._ck())
        ), mock.patch.object(torch.cuda, "empty_cache"):
            producer.load_resident_embedding("/nonexistent")
        self.assertIs(draft_model.lm_head, embed)
        self.assertTrue(hasattr(embed, "weight"))
        self.assertTrue(bool((embed.weight == 3).all()))
        self.assertEqual(producer.head_released_mib, 0.0)


class TestResidentInstrument(CustomTestCase):
    """W11 grades the producer's live weight bytes. The NVML free delta
    cannot measure that under --enable-memory-saver: the weights live in
    torch_memory_saver's ``torch.cuda.MemPool`` (entrypoint.py:89-91), which
    ``empty_cache()`` does not hand back to the driver (memsaver.md N4)."""

    def test_fix3_live_weight_mib_excludes_the_shared_target_head(self):
        from sglang.srt.speculative.draft_kv_producer import _live_weight_mib

        model = torch.nn.Module()
        model.mtp = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
        model.lm_head = _head(64, 32)
        own = 16 * 16 * 2 / float(2**20)
        both = own + 64 * 32 * 2 / float(2**20)
        self.assertAlmostEqual(_live_weight_mib(model), both)
        self.assertAlmostEqual(_live_weight_mib(model, shared=model.lm_head), own)

    def test_fix3_a_storage_shared_twice_is_counted_once(self):
        from sglang.srt.speculative.draft_kv_producer import _live_weight_mib

        model = torch.nn.Module()
        p = torch.nn.Parameter(torch.zeros(64, 32, dtype=torch.bfloat16), requires_grad=False)
        model.a = torch.nn.Module()
        model.a.weight = p
        model.b = torch.nn.Module()
        model.b.weight = p
        self.assertAlmostEqual(_live_weight_mib(model), 64 * 32 * 2 / float(2**20))


if __name__ == "__main__":
    unittest.main()
