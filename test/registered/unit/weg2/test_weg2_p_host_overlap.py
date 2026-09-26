"""P-HOST-OVERLAP (--p-host-overlap) and the #PGAP instrument (--p-hostgap).

Measured reason (weg2xsn420, 4 x 98k, PP3 42/11/11, depth 0): every P stage's
scheduler thread waited for the forward it had launched last before it planned
and launched the next chunk, so the card idled for that host work once per chunk
(PP2: pass 873 ms, gpu-ms 733). See managers/weg2_p_overlap.py for the two host
syncs and the three changes. These tests pin: off = byte-identical launcher
output and today's inline publish; on = the two env variables, the deferred
publish in order, the last-rank device fence, and an instrument that never syncs.
"""
import inspect
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler_pp_mixin as ppm
from sglang.srt.managers import weg2_p_overlap as pov
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

_ENVS = (pov.P_HOST_OVERLAP_ENV, pov.P_HOSTGAP_ENV, pov.SKIP_PURE_CHUNK_OUTPUT_ENV,
         pov.P_NOSYNC_ENV)


def _loop_src() -> str:
    text = open(inspect.getsourcefile(ppm)).read()
    return text[text.index("    def _event_loop_pp_body(self"):text.index("    def event_loop_pp_disagg_prefill(")]


class _Env:
    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in _ENVS}
        for k in _ENVS:
            os.environ.pop(k, None)
        os.environ.update(self.kv)

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestLauncherSwitch(unittest.TestCase):
    def test_off_adds_nothing(self):
        # byte-identity: both switches off -> no variable, no provenance line
        self.assertEqual(launcher.p_host_overlap_env(False, False), {})
        self.assertEqual(launcher.p_host_overlap_lines(False, False), [])

    def test_flags_exist_and_default_off(self):
        base = ["--tree", "/t", "--tag", "t"]
        ns = launcher.build_parser().parse_args(base)
        self.assertFalse(ns.p_host_overlap)
        self.assertFalse(ns.p_hostgap)
        ns = launcher.build_parser().parse_args(base + ["--p-host-overlap", "--p-hostgap"])
        self.assertTrue(ns.p_host_overlap)
        self.assertTrue(ns.p_hostgap)

    def test_overlap_sets_both_variables_the_runtime_reads(self):
        env = launcher.p_host_overlap_env(True, False)
        self.assertEqual(env, {"SGLANG_WEG2_P_HOST_OVERLAP": "1",
                               "SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM": "1",
                               "SGLANG_WEG2_P_NOSYNC": "1"})
        # the name the upstream predicate reads is the one the launcher sets
        src = inspect.getsource(ppm._pp_can_skip_output_comm)
        self.assertIn("SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM", src)
        self.assertEqual(len(launcher.p_host_overlap_lines(True, False)), 1)

    def test_hostgap_is_separate(self):
        self.assertEqual(launcher.p_host_overlap_env(False, True),
                         {"SGLANG_WEG2_P_HOSTGAP": "1"})
        both = launcher.p_host_overlap_env(True, True)
        self.assertEqual(set(both), set(_ENVS))

    def test_env_p_is_the_only_group_that_gets_it(self):
        src = inspect.getsource(launcher.main)
        i = src.index("env_p.update(p_host_overlap_env(")
        self.assertNotIn("env_d.update(p_host_overlap_env(", src)
        # after build_env for P, i.e. P's own environment
        self.assertLess(src.index('group="P"'), i)


class TestSkipPredicate(unittest.TestCase):
    """The upstream predicate the launcher turns on: middle chunks only."""

    def _batch(self, last=False, n=1, logprob=False):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        return types.SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, reqs=[object()] * n,
            contains_last_prefill_chunk=last, return_logprob=logprob)

    def test_only_a_pure_middle_chunk_skips(self):
        with _Env():
            self.assertFalse(ppm._pp_can_skip_output_comm(self._batch()))
        with _Env(**{pov.SKIP_PURE_CHUNK_OUTPUT_ENV: "1"}):
            self.assertTrue(ppm._pp_can_skip_output_comm(self._batch()))
            self.assertFalse(ppm._pp_can_skip_output_comm(self._batch(last=True)))
            self.assertFalse(ppm._pp_can_skip_output_comm(self._batch(n=2)))
            self.assertFalse(ppm._pp_can_skip_output_comm(self._batch(logprob=True)))
            # both ends of a hop ask this one predicate (#753)
            self.assertFalse(ppm._pp_output_exchange_due(self._batch()))
            self.assertTrue(ppm._pp_output_exchange_due(self._batch(last=True)))


class TestDeferredPublish(unittest.TestCase):
    def _cache(self):
        calls = []
        c = types.SimpleNamespace()
        c._weg2_publish_at_chunk = lambda req, key: calls.append((req, key))
        c._weg2_defer_chunk_publish = types.MethodType(
            urc.UnifiedRadixCache._weg2_defer_chunk_publish, c)
        c.weg2_flush_deferred_chunk_publish = types.MethodType(
            urc.UnifiedRadixCache.weg2_flush_deferred_chunk_publish, c)
        return c, calls

    def test_flush_runs_in_order_once(self):
        c, calls = self._cache()
        self.assertEqual(c.weg2_flush_deferred_chunk_publish(), 0)   # nothing deferred
        c._weg2_defer_chunk_publish("r", "k1")
        c._weg2_defer_chunk_publish("r", "k2")
        self.assertEqual(calls, [])
        self.assertEqual(c.weg2_flush_deferred_chunk_publish(), 2)
        self.assertEqual(calls, [("r", "k1"), ("r", "k2")])          # parents first
        self.assertEqual(c.weg2_flush_deferred_chunk_publish(), 0)

    def test_cache_unfinished_req_defers_only_when_on(self):
        src = inspect.getsource(urc.UnifiedRadixCache.cache_unfinished_req)
        # the inline call stays exactly where xsn342 pins it (after the cleanup)
        i_clean = src.index("cleanup_after_caching_req")
        i_gate = src.index("_weg2_p_overlap.p_host_overlap_on()")
        self.assertLess(i_clean, i_gate)
        self.assertIn("self._weg2_defer_chunk_publish(req, radix_key)", src[i_gate:])
        self.assertIn("self._weg2_publish_at_chunk(req, radix_key)", src[i_gate:])
        # never EARLIER than the inline position (xsn358's race, 476fa26ca5)
        self.assertNotIn("_weg2_defer_chunk_publish", src[:i_clean])

    def test_finish_flushes_first(self):
        src = inspect.getsource(urc.UnifiedRadixCache.cache_finished_req)
        self.assertLess(src.index("self.weg2_flush_deferred_chunk_publish()"),
                        src.index("req.pop_committed_kv_cache()"))

    def test_finish_retires_done_copies_before_it_serves(self):
        # xsn422/423 HOLD-REFETCH: the N-1 publish's copy is done at the finish
        # but its ack was never polled (PP0 sat in the last chunk's output
        # wait), so the page was not COMPLETE when the response went out
        src = inspect.getsource(urc.UnifiedRadixCache.cache_finished_req)
        i_flush = src.index("self.weg2_flush_deferred_chunk_publish()")
        i_gate = src.index("if _weg2_p_overlap.p_host_overlap_on():", i_flush)
        i_poll = src.index("self.writing_check()", i_gate)
        # the NON-blocking branch (write_back=False): never waits for a copy
        self.assertNotIn("write_back=True", src[i_poll:i_poll + 40])
        for later in ("req.pop_committed_kv_cache()", "self._weg2_handoff_write(req, radix_key)",
                      "self._weg2_publish_at_retain(req, radix_key)"):
            self.assertLess(i_poll, src.index(later), later)

    def test_poll_retires_only_complete_acks(self):
        # writing_check(write_back=False): Event.query() decides, a copy still
        # running stays pending -- the finish never waits for it
        done_ids, events = [], []

        class _Ev:
            def __init__(self, done):
                self.done = done

            def query(self):
                return self.done

            def synchronize(self):
                if not self.done:
                    raise AssertionError("the finish poll waited for a running copy")

            def elapsed_time(self, other):
                return 0.0

        c = types.SimpleNamespace()
        c.cache_controller = types.SimpleNamespace(ack_write_queue=[
            (_Ev(True), _Ev(True), [1]), (_Ev(False), _Ev(False), [2])])
        c._count_ready_acks = types.MethodType(urc.UnifiedRadixCache._count_ready_acks, c)
        c._drain_depth_every = 0
        c.pp_rank = 0
        c._finish_write_through_ack = lambda ack_id: done_ids.append(ack_id)
        urc.UnifiedRadixCache.writing_check(c)
        self.assertEqual(done_ids, [1])
        self.assertEqual(len(c.cache_controller.ack_write_queue), 1)


class TestLoopWiring(unittest.TestCase):
    def test_flush_sits_after_the_launch_and_before_the_output_commit(self):
        src = _loop_src()
        i_launch = src.index("result, self.launch_event = self._pp_launch_batch(")
        i_flush = src.index('"weg2_flush_deferred_chunk_publish"')
        i_commit = src.index("self._pp_commit_send_output_work_and_preprocess_output_tensors(",
                             i_launch)
        self.assertLess(i_launch, i_flush)
        self.assertLess(i_flush, i_commit)
        self.assertIn("if _pov.p_host_overlap_on():", src[i_launch:i_flush])

    def test_last_rank_device_fence_is_gated_and_host_free(self):
        src = _loop_src()
        i_if = src.index("if not self.pp_group.is_last_rank:\n                    # #969J/#968")
        i_elif = src.index("elif cur_batch and _pov.p_host_overlap_on():", i_if)
        block = src[i_elif:src.index("self.pp_outputs = next_pp_outputs", i_elif)]
        self.assertIn("current_stream().wait_event(", block)
        self.assertIn("self.launch_event", block)
        # a device fence, never a host wait
        self.assertNotIn(".synchronize()", block)

    def test_gap_meter_only_behind_the_instrument(self):
        src = inspect.getsource(ppm.SchedulerPPMixin._pp_launch_batch)
        self.assertIn("_gap = self._pp_gap_meter() if _pov.hostgap_on() else None", src)
        self.assertLess(src.index("_gap.begin()"), src.index("result = self.run_batch("))


class TestLeadBound(unittest.TestCase):
    class _W:
        def __init__(self):
            self.waited = 0

        def wait(self):
            self.waited += 1

    def _pw(self, cuda=True):
        w = self._W()
        payload = types.SimpleNamespace(is_cuda=cuda)
        return types.SimpleNamespace(work=w, payload=payload), w

    def test_only_device_halves_are_frames_and_old_ones_are_fenced(self):
        h = types.SimpleNamespace()
        works = []
        for _ in range(3):                     # three passes, each: 1 gloo + 2 NCCL works
            meta, wm = self._pw(cuda=False)
            d1, w1 = self._pw()
            d2, w2 = self._pw()
            pov.note_proxy_send(h, [meta, d1, d2])
            works.append((wm, w1, w2))
        self.assertEqual(len(h._weg2_proxy_lead), 3)
        self.assertEqual(pov.bound_proxy_lead(h, lead=1), 2)   # the two oldest frames
        (wm0, a0, b0), (wm1, a1, b1), (wm2, a2, b2) = works
        self.assertEqual((a0.waited, b0.waited, a1.waited, b1.waited), (1, 1, 1, 1))
        self.assertEqual((a2.waited, b2.waited), (0, 0))       # the newest stays free
        self.assertEqual((wm0.waited, wm1.waited), (0, 0))     # gloo metadata never
        self.assertEqual(pov.bound_proxy_lead(h, lead=1), 0)

    def test_default_lead_and_env(self):
        with _Env():
            self.assertEqual(pov.overlap_lead(), 1)
        os.environ[pov.P_OVERLAP_LEAD_ENV] = "3"
        try:
            self.assertEqual(pov.overlap_lead(), 3)
        finally:
            os.environ.pop(pov.P_OVERLAP_LEAD_ENV, None)

    def test_wired_before_the_launch_and_after_the_send(self):
        src = _loop_src()
        i_bound = src.index("_pov.bound_proxy_lead(self)")
        i_launch = src.index("result, self.launch_event = self._pp_launch_batch(")
        self.assertLess(i_bound, i_launch)
        i_send = src.index("self.send_proxy_work = self._pp_send_dict_to_next_stage(")
        i_note = src.index("_pov.note_proxy_send(self, self.send_proxy_work)")
        self.assertLess(i_send, i_note)


class _Ev:
    """A fake timing event: done after `ready` query() calls, t = record order."""
    t = 0.0

    def __init__(self):
        self.recorded = None
        self.done = True

    def record(self, *a):
        _Ev.t += 10.0
        self.recorded = _Ev.t

    def query(self):
        return self.done

    def elapsed_time(self, other):
        return other.recorded - self.recorded

    def synchronize(self):  # the instrument may never create the stall it measures
        raise AssertionError("GapMeter synchronised")


class TestInstrument(unittest.TestCase):
    def test_span_is_inert_when_off(self):
        with _Env():
            pov.take_spans()
            with pov.span("plan"):
                pass
            self.assertEqual(pov.take_spans(), {})

    def test_span_accumulates_when_on(self):
        with _Env(**{pov.P_HOSTGAP_ENV: "1"}):
            pov.take_spans()
            with pov.span("publish"):
                pass
            with pov.span("publish"):
                pass
            d = pov.take_spans()
            self.assertIn("publish", d)
            self.assertEqual(pov.take_spans(), {})

    def test_fi_plan_is_a_launch_term_on_the_line(self):
        # fi_plan is PART OF launch (flashinfer's blocking plan read), printed
        # right after it so `launch - fi_plan` reads off one line
        self.assertEqual(pov._ORDER[pov._ORDER.index("launch") + 1], "fi_plan")
        line = pov.format_line(1, 7, 4096, 2.0, 500.0, {"launch": 480.0, "fi_plan": 430.0})
        self.assertIn("launch=480 fi_plan=430", line)

    def test_spans_opened_inside_the_launch_go_on_this_forwards_line(self):
        src = inspect.getsource(ppm.SchedulerPPMixin._pp_launch_batch)
        run = src.index("result = self.run_batch(")
        merge = src.index("for _k, _v in _pov.take_spans().items():", run)
        self.assertLess(merge, src.index("_gap.end(", run))

    def test_both_flashinfer_prefill_plans_sit_in_the_span(self):
        # text, not import: the backend module pulls flashinfer in
        path = os.path.join(os.path.dirname(inspect.getsourcefile(ppm)), "..", "layers",
                            "attention", "flashinfer_backend.py")
        text = open(os.path.abspath(path)).read()
        for call in ("wrapper_ragged.begin_forward(", "wrapper_paged.begin_forward("):
            self.assertEqual(text.count(call), 1)
            i = text.index(call)
            head = text[text.rindex("\n", 0, text.rindex("\n", 0, i)):i]
            self.assertIn('with _weg2_p_overlap.span("fi_plan"):', head)
        self.assertIn("from sglang.srt.managers import weg2_p_overlap as _weg2_p_overlap", text)

    def test_gap_meter_measures_on_the_card_and_never_syncs(self):
        m = pov.GapMeter(2, event_factory=_Ev)
        m.begin()
        m.end(41, 4096, {"plan": 30.0})
        m.begin()
        m.end(42, 4096, {"plan": 31.0, "publish": 12.0})
        lines = m.harvest()
        self.assertEqual(len(lines), 2)
        self.assertIn("gpu_gap_ms=-", lines[0])            # first forward: no previous end
        self.assertIn("gpu_gap_ms=10.0", lines[1])         # prev end -> this start
        self.assertIn("gpu_fwd_ms=10.0", lines[1])
        self.assertIn("pp_rank=2 fwd=42 tokens=4096", lines[1])
        self.assertIn("plan=31", lines[1])
        self.assertIn("publish=12", lines[1])

    def test_gap_meter_waits_for_completion_without_blocking(self):
        m = pov.GapMeter(0, event_factory=_Ev)
        m.begin()
        m.end(1, 4096, {})
        m._pending[0][2].done = False                        # the forward still runs
        self.assertEqual(m.harvest(), [])
        m._pending[0][2].done = True
        self.assertEqual(len(m.harvest()), 1)

    def test_gap_meter_is_bounded(self):
        m = pov.GapMeter(0, event_factory=_Ev, cap=3)
        for i in range(10):
            m.begin()
            m.end(i, 1, {})
        self.assertEqual(len(m._pending), 3)


class TestLadderScript(unittest.TestCase):
    def test_reads_p_wall_and_rank_sums_from_a_synthetic_boot(self):
        import importlib.util
        import tempfile

        path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                            "scripts", "weg2_p_prefill_ladder.py")
        spec = importlib.util.spec_from_file_location("ladder", os.path.abspath(path))
        lad = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lad)
        with tempfile.TemporaryDirectory() as d:
            fl, pl = os.path.join(d, "f.log"), os.path.join(d, "p.log")
            with open(fl, "w") as f:
                f.write("[2026-09-24 12:40:10,000] INFO weg2.front: WEG2-ROUTE rid=weg2-1-2 LONG -> P leg 1 (x)\n")
                f.write("[2026-09-24 12:40:12,500] INFO weg2.front: WEG2-SERVED group=P leg=1 "
                        "rid=weg2-1-2 prompt_tokens=8192 cached_tokens=0 wall=1.50s epoch=3\n")
            with open(pl, "w") as f:
                f.write("[2026-09-24 12:40:05 PP0] Prefill rank batch, #new-token: 4096, #cached-token: 0, "
                        "#chunks: 1, gpu-ms: 999.0 (compute 999.0, wait 0.0)\n")      # before the leg
                for r in (0, 1, 2):
                    f.write("[2026-09-24 12:40:11 PP%d] Prefill rank batch, #new-token: 4096, "
                            "#cached-token: 0, #chunks: 1, gpu-ms: 300.0 (compute 300.0, wait 0.0)\n" % r)
                    f.write("[2026-09-24 12:40:12 PP%d] Prefill rank batch, #new-token: 4096, "
                            "#cached-token: 0, #chunks: 1, gpu-ms: 350.0 (compute 350.0, wait 0.0)\n" % r)
            route, served = lad._read_front(fl, "weg2-1-2")
            self.assertEqual(route, "LONG")
            self.assertEqual(served["p_wall_s"], 1.5)
            per = lad._read_p(pl, served["end_epoch"] - served["p_wall_s"], served["end_epoch"])
            self.assertEqual(sorted(per), [0, 1, 2])
            self.assertEqual(per[0]["chunks"], 2)
            self.assertAlmostEqual(per[0]["gpu_ms"], 650.0)


if __name__ == "__main__":
    unittest.main()
