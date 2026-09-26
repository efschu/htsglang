# SPDX-License-Identifier: Apache-2.0
"""Group D: the deferred host-length read (SGLANG_WEG2_D_DEFER_SEQ_LENS_CPU) and
the #DGAP round instrument (SGLANG_WEG2_D_HOSTGAP) -- managers/weg2_d_hostgap.py.

What is pinned, on CPU (streams and events are fakes that record their calls):

* the old read is untouched when the switch is off: ``resolve_seq_lens_cpu``
  synchronizes the private D2H stream up front and returns None;
* the deferred read queues the SAME device work, synchronizes nothing, leaves
  the mirror None and hands back a pending read whose completion produces the
  byte-identical mirror (tensor and sum) -- and re-applies it after the
  scheduler's isolation restore without a second wait;
* every path that is not the pinned D2H path ignores the flag (GPU-only
  backends, the bootstrap ``.cpu()`` read, the CI poison mode);
* eligibility: decode only, worker must complete the read itself, the
  reservation bound must be there, no grammar / extend-in-batch / DSpark
  confidence prepare, switch cached per scheduler;
* the worker: completes before its first exact read of the decode round (the
  draft prep), sizes the DCP prebuild by the reservation bound while the
  mirror is still pending, and the scheduler re-applies after the forward;
* the #DGAP accounting (period, result_wait split, hicache, host_other, crit,
  restart after a decode pause) and its loud-but-not-fatal config parse.
"""

import ast
import inspect
import os
import textwrap
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

from sglang.srt.managers import overlap_utils as ou  # noqa: E402
from sglang.srt.managers import weg2_d_hostgap as dg  # noqa: E402


class _Calls(list):
    pass


class _FakeStream:
    def __init__(self, log):
        self.log = log

    def wait_event(self, ev):
        self.log.append(("d2h.wait_event", ev.name))

    def synchronize(self):
        self.log.append(("d2h.synchronize",))


class _FakeEvent:
    def __init__(self, log, name):
        self.log = log
        self.name = name

    def wait(self, stream=None):
        self.log.append((f"{self.name}.wait",))

    def synchronize(self):
        self.log.append((f"{self.name}.synchronize",))

    def record(self, stream=None):
        self.log.append((f"{self.name}.record", type(stream).__name__))


class _Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_decode(self):
        return self.kind == "decode"

    def is_extend(self):
        return self.kind == "extend"

    def is_idle(self):
        return self.kind == "idle"


def _future_map(log, *, pool=16, needs_cpu=True, with_stream=True):
    fm = ou.FutureMap.__new__(ou.FutureMap)
    fm.device = torch.device("cpu")
    fm.needs_cpu_seq_lens = needs_cpu
    fm.req_pool_size = pool
    fm.max_context_len = 1 << 20
    fm._publish_fresh = True
    g = torch.Generator().manual_seed(7)
    fm.new_seq_lens_buf = torch.randint(100, 5000, (pool,), generator=g, dtype=torch.int64)
    fm.new_seq_lens_cpu_pinned = torch.full((pool,), -1, dtype=torch.int64)
    fm.fwd_prepare_d2h_stream = _FakeStream(log) if with_stream else None
    fm.publish_ready = _FakeEvent(log, "publish_ready") if with_stream else None
    fm._seq_lens_d2h_done = _FakeEvent(log, "d2h_done")
    return fm


def _batch(idx):
    idx_t = torch.tensor(idx, dtype=torch.int64)
    return SimpleNamespace(
        spec_info=SimpleNamespace(future_indices=idx_t.clone(),
                                  nxt_kv_lens_cpu=torch.tensor([9] * len(idx))),
        req_pool_indices_cpu=idx_t.clone(),
        seq_lens=None,
        seq_lens_cpu="SEEDED",
        seq_lens_sum=-1,
        forward_mode=_Mode("decode"),
        has_grammar=False,
        is_extend_in_batch=False,
    )


class _NoDebug:
    def __enter__(self):
        self._p = mock.patch.object(ou, "_DEBUG_ASSERT", False)
        self._p.start()

    def __exit__(self, *a):
        self._p.stop()


class TestResolveSeqLensCpuDefer(CustomTestCase):
    def setUp(self):
        dg.reset_for_tests()
        os.environ.pop(dg.D_HOSTGAP_ENV, None)

    def tearDown(self):
        dg.reset_for_tests()

    def test_switch_off_is_the_old_read(self):
        log = _Calls()
        fm = _future_map(log)
        b = _batch([3, 5, 11])
        with _NoDebug():
            ret = fm.resolve_seq_lens_cpu(b)
        self.assertIsNone(ret)
        self.assertEqual(
            log,
            [("publish_ready.wait",), ("d2h.wait_event", "publish_ready"),
             ("d2h.synchronize",)],
        )
        exp = fm.new_seq_lens_buf[[3, 5, 11]]
        torch.testing.assert_close(b.seq_lens_cpu, exp, rtol=0, atol=0)
        self.assertEqual(b.seq_lens_sum, int(exp.sum()))
        torch.testing.assert_close(b.seq_lens, exp, rtol=0, atol=0)

    def test_deferred_queues_same_device_work_and_waits_nowhere(self):
        log = _Calls()
        fm = _future_map(log)
        b = _batch([3, 5, 11])
        with _NoDebug():
            pending = fm.resolve_seq_lens_cpu(b, defer=True)
        self.assertIsInstance(pending, dg.PendingSeqLensCpu)
        self.assertNotIn(("d2h.synchronize",), log)
        self.assertEqual(
            log,
            [("publish_ready.wait",), ("d2h.wait_event", "publish_ready"),
             ("d2h_done.record", "_FakeStream")],
        )
        self.assertIsNone(b.seq_lens_cpu)
        self.assertIsNone(b.seq_lens_sum)
        # device half done: the relayed device lengths are already gathered
        torch.testing.assert_close(
            b.seq_lens, fm.new_seq_lens_buf[[3, 5, 11]], rtol=0, atol=0
        )
        self.assertFalse(pending.done)

    def test_completion_is_byte_identical_to_the_old_read(self):
        idx = [0, 2, 7, 15, 4]
        log_a, log_b = _Calls(), _Calls()
        fm_a, fm_b = _future_map(log_a), _future_map(log_b)
        a, b = _batch(idx), _batch(idx)
        with _NoDebug():
            fm_a.resolve_seq_lens_cpu(a)
            pending = fm_b.resolve_seq_lens_cpu(b, defer=True)
            pending.complete(b)
        self.assertEqual(log_b[-1], ("d2h_done.synchronize",))
        self.assertTrue(pending.done)
        self.assertEqual(a.seq_lens_cpu.dtype, b.seq_lens_cpu.dtype)
        torch.testing.assert_close(a.seq_lens_cpu, b.seq_lens_cpu, rtol=0, atol=0)
        self.assertEqual(a.seq_lens_sum, b.seq_lens_sum)
        self.assertIsInstance(b.seq_lens_sum, int)

    def test_second_completion_reapplies_without_waiting(self):
        log = _Calls()
        fm = _future_map(log)
        b = _batch([1, 2])
        with _NoDebug():
            pending = fm.resolve_seq_lens_cpu(b, defer=True)
        pending.complete(b)
        want_cpu, want_sum = b.seq_lens_cpu, b.seq_lens_sum
        n_sync = log.count(("d2h_done.synchronize",))
        # the scheduler's isolation restore puts the pre-forward snapshot back
        b.seq_lens_cpu, b.seq_lens_sum = None, None
        pending.complete(b)
        self.assertEqual(log.count(("d2h_done.synchronize",)), n_sync)
        self.assertIs(b.seq_lens_cpu, want_cpu)
        self.assertEqual(b.seq_lens_sum, want_sum)

    def test_debug_assert_mode_keeps_the_old_read(self):
        log = _Calls()
        fm = _future_map(log)
        b = _batch([3])
        with mock.patch.object(ou, "_DEBUG_ASSERT", True), mock.patch.object(
            ou, "_assert_nonneg_and_invalidate"
        ) as poison:
            ret = fm.resolve_seq_lens_cpu(b, defer=True)
        self.assertIsNone(ret)
        self.assertIn(("d2h.synchronize",), log)
        self.assertEqual(poison.call_count, 1)
        self.assertIsNotNone(b.seq_lens_cpu)

    def test_gpu_only_backend_and_bootstrap_ignore_the_flag(self):
        log = _Calls()
        fm = _future_map(log, needs_cpu=False)
        b = _batch([3])
        with _NoDebug():
            self.assertIsNone(fm.resolve_seq_lens_cpu(b, defer=True))
        self.assertIsNone(b.seq_lens_cpu)
        self.assertNotIn(("d2h_done.record", "_FakeStream"), log)

        log2 = _Calls()
        fm2 = _future_map(log2, with_stream=False)
        b2 = _batch([3, 4])
        with _NoDebug():
            self.assertIsNone(fm2.resolve_seq_lens_cpu(b2, defer=True))
        torch.testing.assert_close(
            b2.seq_lens_cpu, fm2.new_seq_lens_buf[[3, 4]], rtol=0, atol=0
        )

    def test_no_future_indices_is_a_noop(self):
        fm = _future_map(_Calls())
        b = _batch([3])
        b.spec_info.future_indices = None
        with _NoDebug():
            self.assertIsNone(fm.resolve_seq_lens_cpu(b, defer=True))
        self.assertEqual(b.seq_lens_cpu, "SEEDED")

    def test_lazy_event_is_created_on_the_device_module(self):
        log = _Calls()
        fm = _future_map(log)
        fm._seq_lens_d2h_done = None
        b = _batch([3])
        with _NoDebug():
            pending = fm.resolve_seq_lens_cpu(b, defer=True)
        self.assertIsInstance(fm._seq_lens_d2h_done, torch.cpu.Event)
        pending.complete(b)
        self.assertEqual(b.seq_lens_sum, int(fm.new_seq_lens_buf[3]))


class TestDeferEligible(CustomTestCase):
    def _sched(self, supports=True, confidence=None):
        worker = SimpleNamespace()
        if supports:
            worker.supports_deferred_seq_lens_cpu = True
        return SimpleNamespace(model_worker=worker,
                               _confidence_budget_prepare=confidence)

    def test_off_by_default_and_cached(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(dg.D_DEFER_SEQ_LENS_CPU_ENV, None)
            s = self._sched()
            self.assertFalse(dg.defer_eligible(s, _batch([1])))
            os.environ[dg.D_DEFER_SEQ_LENS_CPU_ENV] = "1"
            # cached per scheduler: flipping the env later changes nothing
            self.assertFalse(dg.defer_eligible(s, _batch([1])))
            os.environ.pop(dg.D_DEFER_SEQ_LENS_CPU_ENV, None)

    def test_conditions(self):
        with mock.patch.dict(os.environ, {dg.D_DEFER_SEQ_LENS_CPU_ENV: "1"}):
            self.assertTrue(dg.defer_eligible(self._sched(), _batch([1])))
            self.assertFalse(dg.defer_eligible(self._sched(supports=False), _batch([1])))
            self.assertFalse(
                dg.defer_eligible(self._sched(confidence=object()), _batch([1]))
            )
            for mut in (
                lambda b: setattr(b, "forward_mode", _Mode("extend")),
                lambda b: setattr(b, "forward_mode", _Mode("idle")),
                lambda b: setattr(b, "is_extend_in_batch", True),
                lambda b: setattr(b, "has_grammar", True),
                lambda b: setattr(b, "spec_info", None),
                lambda b: setattr(b.spec_info, "future_indices", None),
                lambda b: setattr(b.spec_info, "nxt_kv_lens_cpu", None),
            ):
                b = _batch([1])
                mut(b)
                self.assertFalse(dg.defer_eligible(self._sched(), b))
            self.assertFalse(dg.defer_eligible(self._sched(), None))

    def test_worker_declares_support(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        self.assertTrue(DFlashWorkerV2.supports_deferred_seq_lens_cpu)
        params = inspect.signature(DFlashWorkerV2.forward_batch_generation).parameters
        self.assertIn("seq_lens_cpu_ready", params)
        self.assertIsNone(params["seq_lens_cpu_ready"].default)


def _fn_ast(obj):
    src = textwrap.dedent(inspect.getsource(obj))
    return ast.parse(src).body[0]


def _is_ready_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "seq_lens_cpu_ready"
    )


def _batch_mirror_loads(tree):
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "batch"
            and node.attr in ("seq_lens_cpu", "seq_lens_sum")
            and isinstance(node.ctx, ast.Load)
        ):
            out.append(node.lineno)
    return sorted(out)


class TestWorkerCompletesBeforeFirstRead(CustomTestCase):
    """The worker side, on the source: the decode round must complete the
    pending read before its FIRST host read of the mirror, and the only call
    that runs before it with the mirror still pending is the DCP prebuild,
    which sizes by the reservation bound (next test)."""

    def test_decode_reads_follow_the_completion(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        fn = _fn_ast(DFlashWorkerV2.forward_batch_generation)
        ready_lines = sorted(n.lineno for n in ast.walk(fn) if _is_ready_call(n))
        self.assertGreaterEqual(len(ready_lines), 3, ready_lines)
        # the decode region starts after the idle early return
        idle_ret = max(
            n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.If) and "is_idle" in ast.unparse(n.test)
        )
        embed_calls = sorted(
            n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "embed_module"
        )
        decode_ready = [ln for ln in ready_lines if ln > idle_ret]
        self.assertTrue(decode_ready, "no completion in the decode round")
        first_ready = decode_ready[0]
        # after the (host-path) embedding, i.e. the embedding is queued first
        self.assertTrue(any(e < first_ready for e in embed_calls))
        for ln in _batch_mirror_loads(fn):
            if ln > idle_ret:
                self.assertGreater(
                    ln, first_ready,
                    f"batch mirror read at relative line {ln} precedes the "
                    f"completion at {first_ready}",
                )
        # the only mirror consumer before the completion is the prebuild
        before = [
            ast.unparse(n.func) for n in ast.walk(fn)
            if isinstance(n, ast.Call) and idle_ret < n.lineno < first_ready
            and "batch" in {a.id for a in n.args if isinstance(a, ast.Name)}
        ]
        self.assertIn("self._dcp_verify_prebuild", before)
        allowed = {"self._dcp_verify_prebuild", "_prepare_dflash_draft_block_unchecked"}
        self.assertTrue(set(before) <= allowed, before)

    def test_prebuild_sizes_by_the_reservation_bound_while_pending(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        seen = {}

        class _Backend:
            def dcp_verify_prebuild(self, rpi, seq_lens, host):
                seen["host"] = host
                return "prebuilt"

        fake_self = SimpleNamespace(_target_dcp_verify_backend=lambda: _Backend())
        bound = torch.tensor([40, 72], dtype=torch.int32)
        b = SimpleNamespace(req_pool_indices=torch.tensor([1, 2]),
                            seq_lens=torch.tensor([30, 60]), seq_lens_cpu=None)
        di = SimpleNamespace(nxt_kv_lens_cpu=bound)
        out = DFlashWorkerV2._dcp_verify_prebuild(fake_self, b, di)
        self.assertEqual(out, "prebuilt")
        self.assertIs(seen["host"], bound)
        # with the exact mirror present it is preferred, as before
        b.seq_lens_cpu = torch.tensor([30, 60])
        DFlashWorkerV2._dcp_verify_prebuild(fake_self, b, di)
        self.assertIs(seen["host"], b.seq_lens_cpu)

    def test_non_decode_paths_complete_first(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        fn = _fn_ast(DFlashWorkerV2.forward_batch_generation)
        first_if = next(
            n for n in fn.body
            if isinstance(n, ast.If) and any(_is_ready_call(c) for c in ast.walk(n))
        )
        test_src = ast.unparse(first_if.test)
        for needle in ("is_extend()", "is_extend_in_batch", "is_idle()"):
            self.assertIn(needle, test_src)
        extend_if = next(
            n for n in fn.body
            if isinstance(n, ast.If) and "is_extend()" in ast.unparse(n.test)
            and n is not first_if
        )
        self.assertLess(first_if.lineno, extend_if.lineno)


class TestSchedulerWiring(CustomTestCase):
    """Scheduler._run_batch_forward (the body of run_batch), on the source (the
    method needs a live scheduler):
    eligibility -> deferred resolve -> callback into the forward -> re-apply
    after the isolation block, before the batch is handed on."""

    def test_order(self):
        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._run_batch_forward)
        keys = [
            "_d_hostgap.defer_eligible(self, batch)",
            "resolve_seq_lens_cpu(\n                        batch, defer=True",
            'fwd_kwargs["seq_lens_cpu_ready"] = partial(',
            "self.model_worker.forward_batch_generation(",
            "_pending_lens.complete(batch)",
            "batch.input_ids = None",
        ]
        pos = [src.find(k) for k in keys]
        self.assertTrue(all(p >= 0 for p in pos), list(zip(keys, pos)))
        self.assertEqual(pos, sorted(pos), list(zip(keys, pos)))
        # the re-apply sits OUTSIDE the isolation block (after its restore)
        tail = src[src.find("_pending_lens.complete(batch)") - 400:
                   src.find("_pending_lens.complete(batch)")]
        self.assertIn("if _pending_lens is not None:", tail)
        line = next(l for l in src.splitlines() if "_pending_lens.complete(batch)" in l)
        iso = next(l for l in src.splitlines()
                   if "with self._forward_isolation(batch, overlap=True):" in l)
        indent = lambda l: len(l) - len(l.lstrip())
        self.assertLess(indent(line) - 4, indent(iso))


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class TestDGapMeter(CustomTestCase):
    def test_split_and_crit(self):
        clk = _Clock()
        m = dg.DHostGap(4, clock=clk)
        self.assertIsNone(m.end_round(deferred=False))  # opens the window
        line = None
        for i in range(4):
            clk.t += 0.030  # 30 ms per round
            m.add("publish_wait", 20.0)
            m.add("copy_done_wait", 1.0)
            m.add("hicache", 0.5)
            m.mark_wake()
            clk.t += 0.002
            m.mark_draft_launched()
            clk.t -= 0.002
            line = m.end_round(deferred=(i % 2 == 0))
        self.assertIsNotNone(line)
        self.assertIn("rounds=4 period_ms=30.000", line)
        self.assertIn("result_wait_ms=21.000 (publish 20.000, copy_done 1.000)", line)
        self.assertIn("hicache_ms=0.500", line)
        self.assertIn("host_other_ms=8.500", line)
        self.assertIn("crit_ms=2.000", line)
        self.assertIn("deferred=2/4", line)

    def test_pause_restarts_the_window(self):
        clk = _Clock()
        m = dg.DHostGap(2, clock=clk)
        m.end_round(False)
        clk.t += 0.030
        m.add("publish_wait", 999.0)
        m.end_round(False)
        clk.t += 5.0  # decode paused: this interval is not a round
        m.add("publish_wait", 5000.0)
        self.assertIsNone(m.end_round(False))
        clk.t += 0.020
        m.end_round(False)
        clk.t += 0.020
        line = m.end_round(False)
        self.assertIn("period_ms=20.000", line)
        self.assertIn("publish 0.000", line)

    def test_draft_launch_without_wake_is_ignored(self):
        m = dg.DHostGap(1, clock=_Clock())
        m.mark_draft_launched()
        m.end_round(False)
        self.assertIn("crit_ms=nan", m.end_round(False) or "crit_ms=nan")

    def test_config_parse(self):
        with mock.patch.dict(os.environ, {dg.D_HOSTGAP_ENV: ""}):
            self.assertEqual(dg.hostgap_rounds(), 0)
        with mock.patch.dict(os.environ, {dg.D_HOSTGAP_ENV: "1"}):
            self.assertEqual(dg.hostgap_rounds(), 512)
        with mock.patch.dict(os.environ, {dg.D_HOSTGAP_ENV: "64"}):
            self.assertEqual(dg.hostgap_rounds(), 64)
        with mock.patch.dict(os.environ, {dg.D_HOSTGAP_ENV: "abc"}):
            with self.assertRaises(ValueError):
                dg.hostgap_rounds()
            dg.reset_for_tests()
            with self.assertLogs(dg.logger, level="WARNING"):
                self.assertIsNone(dg.meter())
        dg.reset_for_tests()

    def test_pending_completion_feeds_the_meter(self):
        dg.reset_for_tests()
        try:
            with mock.patch.dict(os.environ, {dg.D_HOSTGAP_ENV: "8"}):
                m = dg.meter()
            self.assertIsNotNone(m)
            log = _Calls()
            pinned = torch.tensor([5, 6, 7], dtype=torch.int64)
            p = dg.PendingSeqLensCpu(pinned, _FakeEvent(log, "ev"),
                                     torch.tensor([2, 0]))
            b = SimpleNamespace(seq_lens_cpu=None, seq_lens_sum=None)
            p.complete(b)
            self.assertEqual(log, [("ev.synchronize",)])
            self.assertIsNotNone(m._wake_t)
            self.assertEqual(b.seq_lens_sum, 12)
            torch.testing.assert_close(b.seq_lens_cpu, torch.tensor([7, 5]))
        finally:
            dg.reset_for_tests()

    def test_launcher_env_helpers(self):
        self.assertEqual(dg.launcher_env_d_defer_seq_lens_cpu(),
                         {"SGLANG_WEG2_D_DEFER_SEQ_LENS_CPU": "1"})
        self.assertEqual(dg.launcher_env_d_hostgap(), {"SGLANG_WEG2_D_HOSTGAP": "1"})


if __name__ == "__main__":
    unittest.main()


class TestDeferRebuildStage2(CustomTestCase):
    """SGLANG_WEG2_D_DEFER_REBUILD: the compact sync-free window rebuild is
    queued before the host wait, with the WIDTH taken from the compact
    envelope of the reservation bound. The two properties that make that
    correct are pinned where they live: the envelope is >= every exact compact
    length (test_dflash_overlap_hostsync.TestCompactSeqLensHostBound) and a
    width >= the exact max rebuilds the identical rows
    (test_dflash_solo_pool.TestWindowRowsSyncFree, random max_len slack). Here:
    the switch, and the order in forward_batch_generation."""

    def test_switch_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(dg.D_DEFER_REBUILD_ENV, None)
            self.assertFalse(dg.defer_rebuild_on())
            os.environ[dg.D_DEFER_REBUILD_ENV] = "1"
            self.assertTrue(dg.defer_rebuild_on())
            os.environ.pop(dg.D_DEFER_REBUILD_ENV, None)

    def test_order_in_the_draft_prep(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        src = textwrap.dedent(inspect.getsource(DFlashWorkerV2.forward_batch_generation))
        keys = [
            "noise_embedding = embed_module(block_ids)",
            "_defer_rebuild = (",
            "if seq_lens_cpu_ready is not None and not _defer_rebuild:",
            "if _defer_rebuild:\n",
            "draft_input.nxt_kv_lens_cpu, out=seq_lens_cpu",
            "rebuild_window_rows_sync_free(",
            "max_len=int(seq_lens_cpu.max()) if bs > 0 else 0",
            "block_loc = mapper.translate_write(verify_out_cache_loc)",
            "# The device part of the draft prep is queued",
            "draft_seq_lens_sum = int(seq_lens_cpu.sum().item())",
            "draft_out = self.draft_model_runner.forward(forward_batch)",
        ]
        pos = [src.find(k) for k in keys]
        self.assertTrue(all(p >= 0 for p in pos), list(zip(keys, pos)))
        self.assertEqual(pos, sorted(pos), list(zip(keys, pos)))
        # inside the stage-2 block: wait, THEN the exact compact mirror
        blk = src[pos[keys.index("# The device part of the draft prep is queued")]:
                  pos[keys.index("draft_seq_lens_sum = int(seq_lens_cpu.sum().item())")]]
        seq = ["seq_lens_cpu_ready()", "batch.seq_lens_cpu, out=seq_lens_cpu",
               "draft_host_lens_exact = self._compact_draft_host_lens_exact()"]
        bpos = [blk.find(k) for k in seq]
        self.assertTrue(all(p >= 0 for p in bpos), list(zip(seq, bpos)))
        self.assertEqual(bpos, sorted(bpos), list(zip(seq, bpos)))
        guard = src[src.find("_defer_rebuild = ("): src.find("if seq_lens_cpu_ready is not None and not _defer_rebuild:")]
        for need in ("seq_lens_cpu_ready is not None", "_defer_rebuild", "use_compact_draft_cache",
                     "_solo_pool_mapper.sync_free", "draft_input.nxt_kv_lens_cpu is not None"):
            self.assertIn(need, guard)

    def test_worker_reads_the_switch_once(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        init_src = inspect.getsource(DFlashWorkerV2.__init__)
        self.assertIn("self._defer_rebuild = defer_rebuild_on()", init_src)
