"""fnFL2 H69 (SGLANG_WEG2_PLE_STAGE_BEHIND_REPLAY): the verify round's PLE
stage is filled AFTER the verify graph was launched; a one-warp gate on the
prefetch stream holds only layer 1's PLE gather until the host publishes the
round.

Desk only (CPU, real pread worker processes, real files; the kernels run in
the Triton interpreter on host memory, the table and stage pointers being
plain host addresses there, exactly as under HMM on the rig). Cases:

* the gate kernel: not armed -> no poll, ``go`` 0; armed and published ->
  ``go`` 1; armed and never published -> gives up after MAX_SPINS and counts
  a timeout; published by ANOTHER thread while it spins -> passes;
* the gated gather: ``go`` 1 is the H40 staged kernel bit for bit (it takes
  staged rows, a lying stage shows through); ``go`` 0 never touches the
  stage (a lying stage does NOT show through) and equals the plain kernel;
* the stager: the ``done`` word sits right behind the ids (gated only; the
  switch-off layout is unchanged), the gate state is made by a non-capturing
  launch only, arm/publish/disarm move ``expect``/``done``;
* the protocol end to end: the "device" (gate + gather) is started first,
  the host stages and publishes afterwards, the gather serves every row from
  the stage -- and a round whose host never publishes still reads the right
  bytes (HMM) and is counted as a timeout;
* ``finish_ple_verify_stage`` publishes even when staging raises;
* the post-replay hook: arm/fire/disarm semantics, and
  ``DecodeCudaGraphRunner.execute`` fires it right after ``backend.replay``;
* verify() wiring: arm -> park the stage on the hook (graphed) or stage it
  (eager) -> forward -> disarm in ``finally`` -> ``launch_ms`` minus the
  nested stage; an unfired hook is staged late (and releases the gate);
* the proof line carries gate_pass/gate_timeout when gated; the switch
  (default off) reaches the stager through ``make_ple_decode_stager``.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import contextlib
import logging
import pathlib
import re
import threading
import time
import types
import unittest
from unittest import mock

import torch
from triton.runtime.interpreter import InterpretedFunction

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import decode_host_split as dhs
from sglang.srt.model_executor.runner import post_replay_hook as hook
from sglang.srt.models import qwen4_exp_ple_decode_pread as dp
from sglang.srt.models import qwen4_exp_ple_prefetch as pf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# the H40 file's fixtures: two checkpoint-like files, the hash stand-in, the
# interpreted plain and staged kernels
from test_qwen4_exp_ple_decode_pread_h40 import (  # noqa: E402
    DIM,
    STAGED,
    TOTAL,
    _ctx,
    _Files,
    _run_plain,
    _run_staged,
    _same,
    _small_emb,
    _stager,
)

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

LOGGER = "sglang.srt.models.qwen4_exp_ple_decode_pread"
SRC = pathlib.Path(dp.__file__).resolve().parents[1]
GATE = InterpretedFunction(dp._ple_stage_gate_kernel.fn)
GATED = InterpretedFunction(dp._gather_ple_embedding_gated_kernel.fn)


def _gate_state(done_word, expect=-1):
    return (
        torch.tensor([expect], dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([done_word.data_ptr()], dtype=torch.int64),
    )


def _run_gate(gs, spins):
    expect, go, ctr, addr = gs
    GATE[(1,)](expect, addr, go, ctr, MAX_SPINS=spins)
    return int(go[0]), ctr.tolist()


def _run_gated(table, ids, stage_ids, stage_rows, go):
    out = torch.full((ids.numel(), DIM), 3.0, dtype=torch.bfloat16)
    addrs = torch.tensor([stage_ids.data_ptr(), stage_rows.data_ptr()], dtype=torch.int64)
    ctr = torch.zeros(2, dtype=torch.int32)
    GATED[(ids.numel(),)](
        torch.tensor(table.bases, dtype=torch.int64), table.shard_rows, ids, addrs,
        torch.tensor([int(go)], dtype=torch.int32), ctr, out,
        embedding_dim=DIM, tp_vocab_start=0, tp_vocab_end=TOTAL, is_fp8=False, BLOCK_D=256,
    )
    return out, ctr.tolist()


class TestGateKernel(CustomTestCase):
    def setUp(self):
        self.done = torch.zeros(1, dtype=torch.int64)

    def test_not_armed_never_polls_and_says_no(self):
        go, ctr = _run_gate(_gate_state(self.done, expect=-1), spins=1 << 30)
        self.assertEqual((go, ctr), (0, [0, 0]))  # returns at once: no timeout either

    def test_armed_and_published_passes(self):
        self.done[0] = 7
        for expect in (1, 7):
            go, ctr = _run_gate(_gate_state(self.done, expect=expect), spins=4)
            self.assertEqual((go, ctr), (1, [1, 0]))

    def test_armed_and_never_published_gives_up_and_counts_it(self):
        self.done[0] = 6
        t0 = time.monotonic()
        go, ctr = _run_gate(_gate_state(self.done, expect=7), spins=50)
        self.assertEqual((go, ctr), (0, [0, 1]))
        self.assertLess(time.monotonic() - t0, 30.0)

    def test_a_publish_from_another_thread_releases_a_spinning_gate(self):
        gs = _gate_state(self.done, expect=3)
        res = {}
        t = threading.Thread(target=lambda: res.update(r=_run_gate(gs, spins=1 << 30)))
        t.start()
        time.sleep(0.05)
        self.assertTrue(t.is_alive())  # spinning: done is still 0
        self.done[0] = 3  # the host's publish, a plain store
        t.join(timeout=60)
        self.assertFalse(t.is_alive())
        self.assertEqual(res["r"], (1, [1, 0]))


class TestGatedKernel(CustomTestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.f = _Files(self.tmp.name)
        g = torch.Generator().manual_seed(9)
        self.ids = torch.randint(0, TOTAL + 40, (64,), generator=g)  # some out of range
        self.rows = torch.empty(64, DIM, dtype=torch.bfloat16)
        self.rows.view(torch.int16).random_(-30000, 30000)  # a lying stage
        self.sid = self.ids.clone()  # every id "staged" -- over garbage bytes

    def tearDown(self):
        self.tmp.cleanup()

    def test_go_is_the_h40_staged_kernel(self):
        want, want_ctr = _run_staged(self.f.table, self.ids, self.sid, self.rows)
        got, ctr = _run_gated(self.f.table, self.ids, self.sid, self.rows, go=1)
        self.assertTrue(_same(got, want))
        self.assertEqual(ctr, want_ctr)
        in_range = int((self.ids < TOTAL).sum())
        self.assertEqual(ctr, [in_range, in_range])  # the lying stage WAS read

    def test_no_go_never_touches_the_stage(self):
        got, ctr = _run_gated(self.f.table, self.ids, self.sid, self.rows, go=0)
        self.assertTrue(_same(got, _run_plain(self.f.table, self.ids)))  # lies not taken
        self.assertEqual(ctr, [int((self.ids < TOTAL).sum()), 0])
        self.assertTrue(torch.all(got[self.ids >= TOTAL].float() == 0))


def _gated_stager(test, **kw):
    kw.setdefault("gated", True)
    kw.setdefault("gate_spins", 1 << 30)
    st = _stager(test.f.table, test.emb, **kw)
    test.stagers.append(st)
    return st


@contextlib.contextmanager
def _interpreted_kernels():
    with mock.patch.object(dp, "_ple_stage_gate_kernel", GATE), \
            mock.patch.object(dp, "_gather_ple_embedding_gated_kernel", GATED), \
            mock.patch.object(dp, "_gather_ple_embedding_staged_kernel", STAGED):
        yield


class TestStagerGate(CustomTestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.f = _Files(self.tmp.name)
        self.emb = _small_emb()
        self.p = dp.PleHashParamsPy.of(pf.PleHashParams.of(self.emb))
        self.stagers = []
        dp._MODEL_EMBEDDINGS.clear()

    def tearDown(self):
        for st in self.stagers:
            st.close()
        self.tmp.cleanup()

    def test_layout_gated_and_off(self):
        off = _stager(self.f.table, self.emb)
        self.stagers.append(off)
        st = _gated_stager(self)
        cap, rb = st.capacity, self.f.table.row_bytes
        self.assertEqual(off._nbytes, cap * (rb + 8))  # switch off: H40's layout
        self.assertIsNone(off.done_word)
        self.assertFalse(off.gated)
        self.assertEqual(st._nbytes, cap * (rb + 8) + dp._GATE_BYTES)
        self.assertTrue(st.gated)
        # the done word right behind the ids, in the same (registered) region
        self.assertEqual(st.done_word.data_ptr(), st.stage_ids.data_ptr() + cap * 8)
        self.assertEqual(st.stage_ids.numel(), cap)
        self.assertEqual(int(st.done_word[0]), 0)

    def test_gate_state_only_from_a_non_capturing_launch(self):
        st = _gated_stager(self)
        cpu = torch.device("cpu")
        self.assertIsNone(st._gate_state(cpu, capturing=True))
        made = st._gate_state(cpu, capturing=False)
        self.assertIs(st._gate_state(cpu, capturing=True), made)
        expect, go, ctr, addr = made
        self.assertEqual(int(expect[0]), -1)
        self.assertEqual(addr.tolist(), [st.done_word.data_ptr()])
        st.arm_gate(5)
        self.assertEqual(int(expect[0]), 5)
        st.publish(5)
        self.assertEqual(int(st.done_word[0]), 5)
        st.disarm_gate()
        self.assertEqual(int(expect[0]), -1)

    def _round(self, st, seed):
        ctx = _ctx(1, 4, seed=seed, hi=100)
        stage = dp.PleVerifyStage((st,), ctx, None)
        return stage, torch.tensor(dp.ple_verify_row_ids(ctx.tolist(), self.p))

    def _device(self, st, ids, out, errors=None):
        """The verify graph's PLE part: gate + gather on the prefetch stream."""
        try:
            with _interpreted_kernels():
                served = st.launch(ids, out, vocab_start=0, vocab_end=TOTAL, block_d=256)
            if not served:
                raise AssertionError("the gated stage did not serve the gather")
        except BaseException as exc:  # noqa: BLE001 - re-raised by the test thread
            if errors is None:
                raise
            errors.append(exc)

    def test_stage_behind_the_launch_serves_every_row_from_the_stage(self):
        st = _gated_stager(self, delay_s=0.05)
        cpu = torch.device("cpu")
        st._device_state(cpu, capturing=False)
        st._gate_state(cpu, capturing=False)
        stage, ids = self._round(st, seed=1)
        stage = dp.arm_ple_verify_gate(stage)
        self.assertEqual(stage.seq, dp._GATE_SEQ)
        self.assertEqual(int(st._gate_state(cpu, capturing=False)[0][0]), stage.seq)
        out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16)
        errors = []
        # the device first: the gate spins, nothing staged yet
        dev = threading.Thread(target=self._device, args=(st, ids, out, errors))
        dev.start()
        time.sleep(0.05)
        self.assertTrue(dev.is_alive())
        self.assertEqual(st.stage_ids[:64].tolist(), [-1] * 64)
        # then the host, as the post-replay hook would run it
        took = dp.finish_ple_verify_stage(stage)
        self.assertGreater(took, 0.0)
        self.assertEqual(int(st.done_word[0]), stage.seq)
        dev.join(timeout=120)
        self.assertFalse(dev.is_alive())
        if errors:
            raise errors[0]
        self.assertTrue(_same(out, _run_plain(self.f.table, ids)))
        self.assertEqual(st.counters.tolist(), [64, 64])  # every row from the stage
        self.assertEqual(st._gate_state(cpu, capturing=False)[2].tolist(), [1, 0])
        dp.disarm_ple_verify_gate(stage)
        self.assertEqual(int(st._gate_state(cpu, capturing=False)[0][0]), -1)

    def test_a_round_the_host_never_publishes_reads_hmm_and_counts_a_timeout(self):
        st = _gated_stager(self, gate_spins=64)
        cpu = torch.device("cpu")
        st._device_state(cpu, capturing=False)
        st._gate_state(cpu, capturing=False)
        stage, ids = self._round(st, seed=2)
        stage = dp.arm_ple_verify_gate(stage)
        # a lying stage under the right ids: must not show through
        st.stage_ids[:64] = ids
        st.stage_rows[:64].view(torch.int16).random_(-30000, 30000)
        out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16)
        self._device(st, ids, out)
        self.assertTrue(_same(out, _run_plain(self.f.table, ids)))
        self.assertEqual(st.counters.tolist(), [64, 0])
        self.assertEqual(st._gate_state(cpu, capturing=False)[2].tolist(), [0, 1])

    def test_a_gather_after_disarm_reads_hmm(self):
        st = _gated_stager(self)
        cpu = torch.device("cpu")
        st._device_state(cpu, capturing=False)
        st._gate_state(cpu, capturing=False)
        stage, ids = self._round(st, seed=3)
        stage = dp.arm_ple_verify_gate(stage)
        dp.finish_ple_verify_stage(stage)
        dp.disarm_ple_verify_gate(stage)
        out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16)
        self._device(st, ids, out)  # expect -1: no poll, no stage
        self.assertTrue(_same(out, _run_plain(self.f.table, ids)))
        self.assertEqual(st.counters.tolist(), [64, 0])
        self.assertEqual(st._gate_state(cpu, capturing=False)[2].tolist(), [0, 0])

    def test_eager_order_stage_first_passes_at_once(self):
        st = _gated_stager(self, gate_spins=4)
        cpu = torch.device("cpu")
        st._device_state(cpu, capturing=False)
        st._gate_state(cpu, capturing=False)
        stage, ids = self._round(st, seed=4)
        stage = dp.arm_ple_verify_gate(stage)
        dp.finish_ple_verify_stage(stage)  # H40 order: before the launch
        out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16)
        self._device(st, ids, out)
        self.assertTrue(_same(out, _run_plain(self.f.table, ids)))
        self.assertEqual(st.counters.tolist(), [64, 64])
        self.assertEqual(st._gate_state(cpu, capturing=False)[2].tolist(), [1, 0])

    def test_publish_even_when_staging_raises(self):
        st = _gated_stager(self)
        stage, _ = self._round(st, seed=5)
        stage = dp.arm_ple_verify_gate(stage)
        with mock.patch.object(st, "stage", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                dp.finish_ple_verify_stage(stage)
        self.assertEqual(int(st.done_word[0]), stage.seq)

    def test_not_gated_is_untouched(self):
        st = _stager(self.f.table, self.emb)
        self.stagers.append(st)
        stage, _ = self._round(st, seed=6)
        self.assertFalse(dp.ple_stage_is_gated(stage))
        self.assertFalse(dp.ple_stage_is_gated(None))
        armed = dp.arm_ple_verify_gate(stage)
        self.assertIs(armed, stage)
        self.assertEqual(armed.seq, -1)
        self.assertIsNone(dp.arm_ple_verify_gate(None))
        dp.disarm_ple_verify_gate(stage)
        dp.disarm_ple_verify_gate(None)
        self.assertEqual(dp.finish_ple_verify_stage(None), 0.0)

    def test_proof_line_carries_the_gate_counts(self):
        st = _gated_stager(self, log_every=1)
        cpu = torch.device("cpu")
        st._device_state(cpu, capturing=False)
        st._gate_state(cpu, capturing=False)
        with self.assertLogs(LOGGER, logging.INFO) as cm:
            for seed in (7, 8):
                stage, ids = self._round(st, seed=seed)
                stage = dp.arm_ple_verify_gate(stage)
                st.snapshot_counters()  # what begin() does, behind the draft
                dp.finish_ple_verify_stage(stage)
                out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16)
                self._device(st, ids, out)
                dp.disarm_ple_verify_gate(stage)
        lines = [r for r in cm.output if "PLE-DECODE-PREAD rounds=" in r]
        self.assertEqual(len(lines), 2)
        m = re.search(r"procs=\d+ gate_pass=(\d+) gate_timeout=(\d+)$", lines[1])
        self.assertIsNotNone(m, lines[1])
        # the counters seen at the second round's begin cover the first gate
        self.assertEqual((int(m[1]), int(m[2])), (1, 0))

    def test_switch_reaches_the_stager_default_off(self):
        self.assertFalse(envs.SGLANG_WEG2_PLE_STAGE_BEHIND_REPLAY.get())
        fn = lambda: pf.PleHashParams.of(self.emb)  # noqa: E731
        for on in (False, True):
            with envs.SGLANG_WEG2_PLE_STAGE_BEHIND_REPLAY.override(on):
                st = dp.make_ple_decode_stager(
                    self.f.table, fn, vocab_start=0, vocab_end=TOTAL, device=torch.device("cpu")
                )
            self.stagers.append(st)
            self.assertEqual(st.gated, on)
            self.assertEqual(st.done_word is not None, on)
        self.assertEqual(st._gate_spins, dp.ple_gate_spins(envs.SGLANG_QWEN4_PLE_DECODE_PREAD_BUDGET_MS.get()))
        self.assertEqual(dp.ple_gate_spins(8.0), 32000)
        self.assertEqual(dp.ple_gate_spins(0.1), dp._GATE_SPINS_MIN)


class TestPostReplayHook(CustomTestCase):
    def setUp(self):
        hook._PENDING.clear()

    def tearDown(self):
        hook._PENDING.clear()

    def test_arm_fire_disarm(self):
        a, b = object(), object()
        seen = []
        self.assertFalse(hook.fire(a))  # nothing armed: one truth test
        hook.arm(a, lambda: seen.append("a"))
        self.assertTrue(hook.pending(a))
        with self.assertRaises(RuntimeError):
            hook.arm(a, lambda: None)
        self.assertFalse(hook.fire(b))  # another runner's replay
        self.assertTrue(hook.fire(a))
        self.assertEqual(seen, ["a"])
        self.assertFalse(hook.fire(a))  # one-shot
        self.assertIsNone(hook.disarm(a))  # fired: nothing handed back
        fn = lambda: seen.append("late")  # noqa: E731
        hook.arm(b, fn)
        self.assertIs(hook.disarm(b), fn)  # unfired: handed back, not run
        self.assertEqual(seen, ["a"])

    def test_decode_graph_runner_fires_right_after_the_replay(self):
        from sglang.srt.model_executor.runner import decode_cuda_graph_runner as dcgr

        order = []
        runner = types.SimpleNamespace(
            device_timer=None,
            spec_algorithm=types.SimpleNamespace(is_dflash_family=lambda: False),
        )

        class _Backend:
            def replay_session(self):
                return contextlib.nullcontext()

            def replay(self, key, fb):
                order.append("replay")
                self.hook_pending_at_replay = hook.pending(runner)
                return None

        backend = _Backend()
        fake = types.SimpleNamespace(
            model_runner=runner,
            attn_backend=types.SimpleNamespace(
                use_captured_forward_metadata_for_breakable_cuda_graph=False
            ),
            backend=backend,
            load_batch=lambda fb, pp: order.append("load"),
            _replay_graph_key=types.SimpleNamespace(size=1),
            _clock_graph_key=lambda key: ("key",),
        )
        fb = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(
                is_decode=lambda: False, is_target_verify=lambda: True, name="TARGET_VERIFY"
            )
        )
        hook.arm(runner, lambda: order.append("hook"))
        with mock.patch.object(
            dcgr, "collective_clock",
            lambda: types.SimpleNamespace(note_graph_replay=lambda key: None),
        ), mock.patch.object(dcgr.graph_replay_census, "maybe_census", lambda *a: None):
            out = dcgr.DecodeCudaGraphRunner.execute(fake, fb)
        self.assertIsNone(out)
        self.assertEqual(order, ["load", "replay", "hook"])
        self.assertTrue(backend.hook_pending_at_replay)  # not before the launch
        self.assertFalse(hook.pending(runner))


def _between(text, start, end):
    i = text.index(start)
    return text[i : text.index(end, i + len(start))]


class TestVerifyWiring(CustomTestCase):
    def setUp(self):
        text = (SRC / "speculative" / "eagle_worker_v2.py").read_text()
        self.ver = _between(text, "    def verify(self, batch: ScheduleBatch):",
                            "    def _finalize_accept_tree_path(")

    def test_graphed_rounds_park_the_stage_on_the_hook(self):
        v = self.ver
        order = [
            "ple_stage = begin_ple_verify_stage(",
            "eagle_prepare_for_verify(",
            '_h58.note_span("vprep_ms", _h58_t)',
            "ple_stage = arm_ple_verify_gate(ple_stage)",
            "ple_stage_is_gated(ple_stage) and can_run_cuda_graph and not eager_round",
            "post_replay_hook.arm(",
            "        else:\n            finish_ple_verify_stage(ple_stage)",
            "        try:\n",
            "_h58.mark(_h58.MARK_VERIFY_LAUNCH)",
            "self.target_worker.forward_batch_generation(",
            "        finally:\n",
            "_stage_ple_unfired(post_replay_hook.disarm(ple_runner))",
            "disarm_ple_verify_gate(ple_stage)",
            '_h58_t = _h58.note_span("launch_ms", _h58_t)',
            '_h58.exclude_span("launch_ms", 1000.0 * sum(ple_hook_s))',
            "eagle_sample(",
        ]
        pos = [v.index(o) for o in order]
        self.assertEqual(pos, sorted(pos), list(zip(order, pos)))
        # the hook stages exactly this round's (armed) stage and keeps its time
        arm = _between(v, "post_replay_hook.arm(", "        else:")
        self.assertIn("ple_hook_s.append(finish_ple_verify_stage(ple_stage))", arm)
        self.assertIn("ple_runner", arm)
        self.assertIn("ple_runner = self.target_worker.model_runner", v)

    def test_an_unfired_hook_is_staged_late_and_warned_once(self):
        from sglang.srt.speculative import eagle_worker_v2 as ew

        ran = []
        ew._PLE_UNFIRED_WARNED = False
        with self.assertLogs(ew.logger, logging.WARNING) as cm:
            ew._stage_ple_unfired(lambda: ran.append(1))
            ew._stage_ple_unfired(lambda: ran.append(2))
        ew._stage_ple_unfired(None)  # fired by the replay: nothing to do
        self.assertEqual(ran, [1, 2])
        self.assertEqual(sum("PLE-STAGE-BEHIND-REPLAY" in r for r in cm.output), 1)

    def test_launch_span_excludes_the_nested_stage(self):
        before = dhs.SPLIT.launch_ms
        dhs.SPLIT.launch_ms += 3.0  # the launch, the stage inside it
        dhs.exclude_span("launch_ms", 1.25)
        self.assertAlmostEqual(dhs.SPLIT.launch_ms - before, 1.75, places=9)


if __name__ == "__main__":
    unittest.main()
