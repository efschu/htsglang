"""D-HEALTH (10.10.): image preprocessing blocked the tokenizer loop that serves /health.

THE BUG (boot dkr27bnvfp4dualvwweightsbar1fs10100032, D group). A 4096x4096
image (16384 image tokens) was tokenized synchronously on the tokenizer
process's asyncio loop -- HF image processor, sha256 over pixel_values, shm
copy. On the metal that took ~41 s, the loop answered nothing, two /health
probes timed out (8 s each) and the front stopped the healthy group (W17).

THE MODEL OF THE STALL. A ``time.sleep`` inside ``process_and_combine_mm_data``.
That is faithful because the GIL probe (job 1ab4cd30, gil_probe.py) measured
every step of that chain as GIL-releasing on the CPU (max loop lag per step
with the step in a thread: 6.2 ms PIL decode, 5.5 ms HF processor, 0.4 ms
sha256, 0.4 ms shm wrap; 2.7 ms for the whole process_and_combine chain; table
in ``mm_tokenize_offload``), and the metal (boot jzmxnp) showed the 22-33 s as
kernel time (THP compaction, swap: stime >95 %) -- a wait outside the
interpreter, which ``time.sleep`` is too.

Red on 86ff356d0d (the module does not exist; ``test_off_*`` documents the
in-loop lag the base has), green with the offload.
"""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.managers import mm_tokenize_offload as mto
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
)
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.weg2 import vision_stage_service as vss

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

STALL_S = 5.5  # longer than the front's health period (5 s)
HEALTH_PERIOD_S = 5.0
TICK_S = 0.01


class _StallingQwenVL(QwenVLImageProcessor):
    """The real ``process_mm_data_async``; the CPU chain replaced by a GIL-free stall."""

    def process_and_combine_mm_data(self, base_output, mm_tokens, **kwargs):
        self.seen.append(
            dict(
                thread=threading.current_thread().name,
                processor=self._processor,
                tokenizer=self._tokenizer,
                vision_rid=vss._REQUEST_RID.get(),
            )
        )
        time.sleep(self.stall_s)
        ids = torch.tensor([1, 2, 3], dtype=torch.long)
        ret = {
            "mrope_positions": torch.zeros(3, 3, dtype=torch.long),
            "mrope_position_delta": torch.tensor([[0]]),
        }
        return [], ids, ret


def _make_proc(stall_s: float) -> _StallingQwenVL:
    proc = _StallingQwenVL.__new__(_StallingQwenVL)
    proc.hf_config = SimpleNamespace(model_type="qwen3_5")
    proc.model_type = "qwen3_5"
    proc.mm_tokens = SimpleNamespace(image_token_id=7, video_token_id=8, audio_token_id=None)
    proc.vision_start_token_id = 5
    proc.vision_end_token_id = 6
    hf = SimpleNamespace(tokenizer=SimpleNamespace(name="shared-rust-tokenizer"))
    proc._processor = hf
    proc._tokenizer = hf.tokenizer
    proc._offload_twin = None
    proc.seen = []
    proc.stall_s = stall_s

    async def load_mm_data(**kwargs):
        return BaseMultiModalProcessorOutput(input_text="<img>", images=["decoded"])

    proc.load_mm_data = load_mm_data
    return proc


def _req():
    return SimpleNamespace(rid="img-1", video_data=None, audio_data=None)


async def _tokenize_under_ticker(proc, rid="img-1"):
    """Worst loop wake-up lag while one image request tokenizes in its dispatch section."""
    order = mto.MmDispatchOrder()
    stop = asyncio.Event()
    worst = [0.0]

    async def ticker():
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            t0 = loop.time()
            await asyncio.sleep(TICK_S)
            worst[0] = max(worst[0], loop.time() - t0 - TICK_S)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)
    async with order.request(rid=rid, is_mm=True) as turn:
        out = await proc.process_mm_data_async(
            image_data=["x"], input_text="<img>", request_obj=_req()
        )
    stop.set()
    await task
    return worst[0], out, turn


class TestLoopAnswersDuringImagePreprocessing(CustomTestCase):
    def test_loop_lag_stays_below_health_period(self):
        proc = _make_proc(STALL_S)
        lag, out, turn = asyncio.run(_tokenize_under_ticker(proc))
        self.assertTrue(turn.offloaded)
        self.assertLess(lag, 1.0, f"loop stood {lag:.2f} s during the image")
        self.assertLess(lag, HEALTH_PERIOD_S)
        self.assertEqual(out.input_ids, [1, 2, 3])

    def test_off_is_the_old_path(self):
        """Offload not permitted = the base: the loop stands for the whole chain."""
        proc = _make_proc(STALL_S)
        with mock.patch.object(mto, "offload_permitted", return_value=False):
            lag, _, turn = asyncio.run(_tokenize_under_ticker(proc))
        self.assertFalse(turn.offloaded)
        self.assertGreaterEqual(lag, STALL_S * 0.9)
        self.assertEqual(proc.seen[0]["thread"], threading.main_thread().name)


class TestOffloadedCallIsolation(CustomTestCase):
    def test_private_processor_and_context_in_the_worker(self):
        """Two threads on one fast tokenizer raise 'Already borrowed' (tokenizers
        0.22.2, reproduced): the worker must run on a private HF processor. The
        transient vision stage's rid (a ContextVar) must reach the worker."""
        proc = _make_proc(0.0)

        async def go():
            vss.set_request_rid("img-ctx")
            return await _tokenize_under_ticker(proc)

        asyncio.run(go())
        seen = proc.seen[0]
        self.assertNotEqual(seen["thread"], threading.main_thread().name)
        self.assertIsNot(seen["processor"], proc._processor)
        self.assertIsNot(seen["tokenizer"], proc._tokenizer)
        self.assertEqual(seen["vision_rid"], "img-ctx")

    def test_tower_in_process_stays_on_the_loop(self):
        """A vision tower service runs inside process_and_combine_mm_data: not offloaded."""
        proc = _make_proc(0.0)
        with mock.patch.object(vss, "installed", return_value=object()):
            _, _, turn = asyncio.run(_tokenize_under_ticker(proc))
        self.assertFalse(turn.offloaded)
        self.assertEqual(proc.seen[0]["thread"], threading.main_thread().name)


class TestDispatchOrder(CustomTestCase):
    """Before: nothing reached the scheduler while an image tokenized, and the
    image went first. Kept: FIFO behind open image requests; /health exempt."""

    def _run(self, scenario):
        with mock.patch(
            "sglang.srt.managers.mm_utils.wrap_shm_features", side_effect=lambda o: o
        ), mock.patch.object(mto, "offload_permitted", return_value=True):
            return asyncio.run(scenario())

    def test_fifo_behind_open_image_and_health_exempt(self):
        sent = []
        release_a = asyncio.Event

        async def scenario():
            order = mto.MmDispatchOrder()
            gate = release_a()

            async def req(name, rid, is_mm, wait=None):
                async with order.request(rid=rid, is_mm=is_mm) as turn:
                    if wait is not None:
                        await wait.wait()
                    await turn.dispatch(sent.append, name)

            a = asyncio.create_task(req("A", "a", True, gate))
            await asyncio.sleep(0)
            others = [
                asyncio.create_task(req("T", "t", False)),
                asyncio.create_task(req("H", f"{HEALTH_CHECK_RID_PREFIX}_x", False)),
                asyncio.create_task(req("B", "b", True)),
            ]
            for _ in range(5):
                await asyncio.sleep(0)
            before_release = list(sent)
            gate.set()
            await asyncio.wait_for(asyncio.gather(a, *others), 5)
            return before_release

        before_release = self._run(scenario)
        self.assertEqual(before_release, ["H"])
        self.assertLess(sent.index("A"), sent.index("T"))
        self.assertLess(sent.index("A"), sent.index("B"))

    def test_dispatch_does_not_queue_behind_the_next_image(self):
        """Image A finished processing while image B processes: A's shm wrap must not wait for B's
        processing (on one shared worker it did -- A's dispatch then waited a whole second image)."""
        sent = []
        a_go, b_go = threading.Event(), threading.Event()

        async def scenario():
            order = mto.MmDispatchOrder()

            async def image(name, go):
                async with order.request(rid=name, is_mm=True) as turn:
                    await mto.run_offloaded(go.wait, 5)
                    await turn.dispatch(sent.append, name)

            a = asyncio.create_task(image("A", a_go))
            await asyncio.sleep(0.05)
            b = asyncio.create_task(image("B", b_go))
            await asyncio.sleep(0.05)  # B's processing is queued behind A's
            a_go.set()
            t_end = time.monotonic() + 2
            while "A" not in sent and time.monotonic() < t_end:
                await asyncio.sleep(0.01)
            a_first = list(sent)
            b_go.set()
            await asyncio.wait_for(asyncio.gather(a, b), 10)
            return a_first

        a_first = self._run(scenario)
        self.assertEqual(a_first, ["A"], "A's dispatch waited for B's processing")
        self.assertEqual(sent, ["A", "B"])

    def test_failed_image_request_releases_the_order(self):
        sent = []

        async def scenario():
            order = mto.MmDispatchOrder()

            async def failing():
                async with order.request(rid="a", is_mm=True):
                    await asyncio.sleep(0.01)
                    raise ValueError("image refused")

            async def text():
                async with order.request(rid="t", is_mm=False) as turn:
                    await turn.dispatch(sent.append, "T")

            f = asyncio.create_task(failing())
            await asyncio.sleep(0)
            t = asyncio.create_task(text())
            with self.assertRaises(ValueError):
                await f
            await asyncio.wait_for(t, 2)

        self._run(scenario)
        self.assertEqual(sent, ["T"])


class TestShmCleanupBetweenWrapAndSend(CustomTestCase):
    """A request cancelled (client gone) while or after its features were copied to /dev/shm but
    before the send: nobody materializes -- and so unlinks -- those segments. Each leak is a whole
    feature tensor (400 MB for a 4096^2 image) in /dev/shm until reboot."""

    def _scenario(self, *, cancel: bool, send_raises: bool = False):
        from sglang.srt.managers.mm_utils import ShmPointerMMData

        wrap_go = threading.Event()
        names = []

        def fake_wrap(obj):
            wrap_go.wait(5)
            item = obj.mm_inputs.mm_items[0]
            item.feature = ShmPointerMMData(item.feature)
            names.append(item.feature.shm_name)
            return obj

        def send(_obj):
            if send_raises:
                raise RuntimeError("zmq gone")

        obj = SimpleNamespace(
            rid="img-shm",
            mm_inputs=SimpleNamespace(
                mm_items=[SimpleNamespace(feature=torch.ones(1024), precomputed_embeddings=None)]
            ),
        )

        async def go():
            order = mto.MmDispatchOrder()

            async def req():
                async with order.request(rid="img-shm", is_mm=True) as turn:
                    await turn.dispatch(send, obj)

            task = asyncio.create_task(req())
            await asyncio.sleep(0.05)
            if cancel:
                task.cancel()
            wrap_go.set()
            try:
                await task
            except (asyncio.CancelledError, RuntimeError):
                pass
            t_end = time.monotonic() + 2
            while not names and time.monotonic() < t_end:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)  # the done-callback runs on the loop

        with mock.patch(
            "sglang.srt.managers.mm_utils.wrap_shm_features", side_effect=fake_wrap
        ), mock.patch.object(mto, "offload_permitted", return_value=True):
            asyncio.run(go())
        self.assertEqual(len(names), 1)
        return names[0]

    @staticmethod
    def _exists(name):
        from multiprocessing import shared_memory

        try:
            seg = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            return False
        seg.close()
        return True

    def test_cancelled_between_wrap_and_send_unlinks(self):
        name = self._scenario(cancel=True)
        self.assertFalse(self._exists(name), "the cancelled request's segment stayed in /dev/shm")

    def test_sent_request_keeps_its_segment(self):
        """Negative branch: a sent request's segment belongs to the scheduler (materialize unlinks it)."""
        name = self._scenario(cancel=False)
        try:
            self.assertTrue(self._exists(name))
        finally:
            _unlink(name)

    def test_send_failure_keeps_its_segment(self):
        """send() raising counts as 'reached send': the message may already be out, and a scheduler
        whose segment vanished fails in unwrap_shm_features -- worse than a leak. Kept on purpose."""
        name = self._scenario(cancel=False, send_raises=True)
        try:
            self.assertTrue(self._exists(name))
        finally:
            _unlink(name)


def _unlink(name):
    from multiprocessing import shared_memory

    try:
        seg = shared_memory.SharedMemory(name=name)
        seg.close()
        seg.unlink()
    except FileNotFoundError:
        pass


class TestInstrumentLines(CustomTestCase):
    """(I) The lines that are to prove or refute the memory-stall suspicion at the metal."""

    def test_mm_tokenize_line_carries_phases_noted_in_the_worker(self):
        """The worker gets a COPY of the context: the stats must be the same object, or every phase
        noted off the loop (queue, HF processor, hash, shm wrap) would print -1."""
        proc = _make_proc(0.0)
        with mock.patch.object(mto, "MM_TOKENIZE_REPORT_S", 0.0), self.assertLogs(
            mto.logger, level="INFO"
        ) as logs:
            asyncio.run(_tokenize_under_ticker(proc))
        line = next(m for m in logs.output if "WEG2-MM-TOKENIZE rid=img-1 " in m)
        self.assertIn("offloaded=1", line)
        self.assertNotIn("queue_ms=-1", line)  # noted on the worker thread
        self.assertNotIn("load_ms=-1", line)  # noted on the loop
        self.assertIn("pixels=0", line)  # the stub's image is no PIL image

    def test_loop_lag_sampler_reports_a_stall(self):
        async def scenario():
            task = asyncio.create_task(mto.loop_lag_sampler(period_s=0.01, report_s=0.05))
            await asyncio.sleep(0.03)
            time.sleep(0.2)  # the loop stands
            await asyncio.sleep(0.05)
            task.cancel()

        with self.assertLogs(mto.logger, level="INFO") as logs:
            asyncio.run(scenario())
        self.assertTrue(any("TOKENIZER-LOOP-LAG max_ms=" in m for m in logs.output), logs.output)


if __name__ == "__main__":
    unittest.main()
