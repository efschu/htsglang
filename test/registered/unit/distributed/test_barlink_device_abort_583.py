"""#583: a tripped barlink DEVICE spin kernel must not destroy the CUDA context.

The 2026-08-05 production boot (CRASH_20260805_boot5_barlink_full.log) died
after 7 minutes with "RuntimeError: Triton Error [CUDA]: unspecified launch
failure" raised out of Triton's ``load_binary`` -- a cuModuleLoadData call,
not even a kernel launch. That is the signature of a STICKY context error: the
context was already dead, and the next CUDA call of any kind reported it. The
Triton kernel named in the traceback was the victim, not the cause.

The cause is that the device transport's three spin kernels called
``__trap()`` on deadline expiry. A device trap destroys the CUDA context. The
sibling BAR1 transport had already been given the correct treatment by #431
fix 2 -- write a status word, return, and let the host raise a structured
error from ``check_aborted`` -- but the device transport, which is the one
production actually achieves ("ACHIEVED=device" for world:0, tp:0 and dcp:0 in
that log), never got it and was never registered with ``barlink_abort_gate``.

These tests pin the fix. They need no GPU.
"""

import re
import unittest
from pathlib import Path

import torch

from sglang.srt.distributed.device_communicators import barlink_abort_gate
from sglang.srt.distributed.device_communicators.barlink_device import (
    _ABORT_KERNELS,
    BarlinkDeviceTransport,
    DeviceCollectiveAborted,
)

_MODULE = Path(
    "python/sglang/srt/distributed/device_communicators/barlink_device.py"
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _MODULE).is_file():
            return parent
    raise AssertionError(f"could not locate {_MODULE} above {here}")


def _cuda_source() -> str:
    text = (_repo_root() / _MODULE).read_text()
    match = re.search(r'_CUDA_SRC = r"""(.*?)"""', text, re.S)
    assert match is not None, "the inline CUDA source moved"
    return match.group(1)


def _strip_line_comments(text: str) -> str:
    """Drop ``//`` comments.

    The bans below are about generated INSTRUCTIONS, not prose. The kernels
    carry comments that name ``__trap()`` precisely to explain why it must not
    be called, and matching those would make the pin unmaintainable.
    """
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def _spin_kernel_bodies() -> dict:
    """The three kernels that contain a deadline check, by name."""
    src = _cuda_source()
    bodies = {}
    for name in (
        "barlink_begin_kernel",
        "barlink_wait_kernel",
        "barlink_wait_one_kernel",
    ):
        start = src.index(f"__global__ void {name}(")
        # The kernel ends at the next __global__ / template at column 0.
        rest = src[start + 1 :]
        stops = [
            rest.index(marker)
            for marker in ("\n__global__ ", "\ntemplate ", "\n#define ")
            if marker in rest
        ]
        bodies[name] = _strip_line_comments(rest[: min(stops)] if stops else rest)
    return bodies


class TestNoDeviceTrapInSpinLoops(unittest.TestCase):
    """The regression pin. A trap here is what killed the boot."""

    def test_no_spin_kernel_calls_barlink_trap(self):
        for name, body in _spin_kernel_bodies().items():
            self.assertNotIn(
                "BARLINK_TRAP()",
                body,
                f"{name} calls BARLINK_TRAP(). A device trap destroys the "
                f"CUDA context and every later CUDA call in the process fails "
                f"with a sticky cudaErrorLaunchFailure at an unrelated call "
                f"site -- that is #583. Write the abort word into seq_dev[1] "
                f"and return instead.",
            )

    def test_no_spin_kernel_calls_raw_trap_either(self):
        for name, body in _spin_kernel_bodies().items():
            self.assertNotIn("__trap()", body, f"{name} calls __trap()")
            self.assertNotIn("assert(", body, f"{name} calls device assert()")

    def test_every_spin_loop_still_has_a_deadline(self):
        """Removing the trap must not have removed the deadline with it."""
        for name, body in _spin_kernel_bodies().items():
            self.assertIn(
                "clock64() - start > timeout",
                body,
                f"{name} lost its deadline check -- a starved peer would now "
                f"hang the GPU forever instead of reporting.",
            )

    def test_every_spin_loop_writes_a_distinct_abort_code(self):
        codes = set()
        for name, body in _spin_kernel_bodies().items():
            found = re.findall(r"seq_dev\[1\] = (\d+)ull;", body)
            self.assertEqual(
                len(found),
                1,
                f"{name} must write exactly one abort code into seq_dev[1]",
            )
            codes.add(int(found[0]))
        self.assertEqual(
            len(codes), 3, "the three spin kernels must be distinguishable"
        )
        self.assertEqual(
            codes,
            set(_ABORT_KERNELS),
            "_ABORT_KERNELS must describe exactly the codes the kernels write",
        )

    def test_abort_word_is_followed_by_return_not_by_falling_through(self):
        for name, body in _spin_kernel_bodies().items():
            self.assertRegex(
                body,
                r"seq_dev\[1\] = \d+ull; return;",
                f"{name} must RETURN after writing the abort word; falling "
                f"through would keep spinning past the deadline.",
            )


class TestSeqDevCarriesTheAbortWord(unittest.TestCase):
    def test_seq_dev_is_allocated_with_two_elements(self):
        text = (_repo_root() / _MODULE).read_text()
        self.assertIn(
            "self._seq_dev = torch.zeros(2, dtype=torch.int64, device=device)",
            text,
            "seq_dev[1] is the abort word, so the tensor needs two elements. "
            "A one-element tensor makes every kernel's abort write an "
            "out-of-bounds store.",
        )


def _fake_transport(code: int = 0, *, launches: int = 1):
    """A transport shaped for check_aborted, without a card or a segment."""
    t = object.__new__(BarlinkDeviceTransport)
    t._seq_dev = torch.zeros(2, dtype=torch.int64)
    t._seq_dev[1] = code
    t.rank = 1
    t.world_size = 3
    t._unchecked_launches = launches
    t._captured_launches = False
    t._boundary_checks = 0
    t._registered_in_gate = False
    return t


class TestCheckAbortedRaises(unittest.TestCase):
    def setUp(self):
        barlink_abort_gate.reset_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()

    def test_clean_word_does_not_raise(self):
        _fake_transport(code=0).check_aborted("unit")

    def test_tripped_word_raises_with_structured_context(self):
        for code, description in _ABORT_KERNELS.items():
            with self.subTest(code=code):
                t = _fake_transport(code=code)
                with self.assertRaises(DeviceCollectiveAborted) as caught:
                    t.check_aborted("unit-where")
                err = caught.exception
                self.assertEqual(err.rank, 1)
                self.assertEqual(err.world, 3)
                self.assertEqual(err.code, code)
                self.assertEqual(err.where, "unit-where")
                # The message must name the kernel, or the reader is back to
                # guessing which wait expired -- the whole point of #583.
                self.assertIn(description.split(" (")[0], str(err))

    def test_nothing_launched_means_no_device_read(self):
        """The free path: no launches since the last check, nothing to read."""
        t = _fake_transport(code=2, launches=0)
        t.check_aborted("unit")  # must not raise despite the dirty word

    def test_gate_disabled_restores_the_old_silence(self):
        import os

        t = _fake_transport(code=2)
        old = os.environ.get(barlink_abort_gate.ENV_ENABLE)
        os.environ[barlink_abort_gate.ENV_ENABLE] = "0"
        try:
            t.check_aborted("unit")
        finally:
            if old is None:
                os.environ.pop(barlink_abort_gate.ENV_ENABLE, None)
            else:
                os.environ[barlink_abort_gate.ENV_ENABLE] = old

    def test_capture_suppresses_the_device_read(self):
        """Reading the word inside a stream capture is illegal, not just slow."""
        import sglang.srt.distributed.device_communicators.barlink as barlink

        t = _fake_transport(code=2)
        original = barlink.graph_capture_running
        barlink.graph_capture_running = lambda: True
        try:
            t.check_aborted("unit")  # must not raise, must not read
        finally:
            barlink.graph_capture_running = original


class TestTransportJoinsTheAbortGate(unittest.TestCase):
    """Without registration the status word is written and read by nobody --
    which is exactly the defect #431 fix 2 named for the BAR1 transport."""

    def setUp(self):
        barlink_abort_gate.reset_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()

    def test_bring_up_registers_with_the_gate(self):
        text = (_repo_root() / _MODULE).read_text()
        self.assertIn("barlink_abort_gate.register(self)", text)
        self.assertIn("barlink_abort_gate.unregister(self)", text)

    def test_transport_exposes_check_aborted_so_the_gate_finds_it(self):
        # barlink_abort_gate.check_aborts() dispatches via getattr(t,
        # "check_aborted", None) -- a transport without it is silently skipped.
        self.assertTrue(callable(BarlinkDeviceTransport.check_aborted))

    def test_gate_raises_for_a_registered_tripped_transport(self):
        t = _fake_transport(code=2)
        barlink_abort_gate.register(t)
        with self.assertRaises(DeviceCollectiveAborted):
            barlink_abort_gate.check_aborts("gate-unit")

    def test_unregistered_transport_is_not_checked(self):
        _fake_transport(code=2)  # tripped, but never registered
        barlink_abort_gate.check_aborts("gate-unit")


if __name__ == "__main__":
    unittest.main()
