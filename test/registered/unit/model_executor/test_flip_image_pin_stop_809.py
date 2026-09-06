"""#809 G1: STOPPING the read-ahead never stalls the scheduler round.

THE HAZARD, and where it was found. ``FlipImagePin.stop_prefetch`` ends in
``reader.join(_PIN_STOP_JOIN_S)`` -- a BLOCKING join whose 5 s bound is argued
in the constant's own comment for the CUTOVER (``weights_arena.py:1579-1583``
verbatim: *"This runs inside the no-return window, so it is a BOUND and not a
join"*). Two other callers reach the same primitive from the SCHEDULER's own
round, where that argument does not hold:

* ``PhaseFlipRuntime._abandon_parked_flip`` -- reached from ``on_round`` via
  ``_round_as_decider``/``_round_as_follower``, i.e. the scheduler event loop;
* ``FlipImagePin.start_prefetch``'s busy branch -- reached from
  ``_enter_armed_state``, the ARM, on that same loop.

A round that waits for a reader to leave its chunk is a stall the flip does not
need: nothing after either call site depends on the reader being GONE, only on
it having been TOLD to go. The cutover is the one caller that genuinely may not
proceed while a writer is live, because it is about to hand the buffer's prefix
to a DMA, and it keeps the joining form.

WHAT IS PINNED HERE:

* the signal-only ``request_stop`` returns while the reader is still inside its
  chunk (no join) and the reader nevertheless stops (the event IS set);
* an arm for the OTHER layout while a reader is in flight starts NO second
  writer and does not join -- the DANGER direction, because two loops on one
  buffer would advance ``bytes_valid`` under the NEW identity while the bytes
  below it are the OLD layout's, which is silent weight corruption that the
  file's own trailer cannot catch (the trailer travels with the image);
* a re-arm for the SAME identity WITHDRAWS a stop the abandon just requested,
  so the read-ahead the abandon interrupted continues instead of freezing --
  the resume property ``start_prefetch``'s own comment argues for, which a
  non-blocking abandon would otherwise silently lose;
* the read loop RELEASES the reader slot when it leaves the buffer, so an arm
  that follows a completed read is not refused by a thread that is merely
  still being torn down.
"""

import os
import tempfile
import threading
import time
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import pinned_host_budget
from sglang.srt.model_executor import weights_arena
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.model_executor.weights_arena"

#: One reader chunk under the env override below (the accessor's floor).
_CHUNK = 1 << 20

#: How long a call that must not join is allowed to take. The joining form
#: waits ``_PIN_STOP_JOIN_S`` = 5.0 s on a reader this fixture holds for 30 s,
#: so the two are an order of magnitude apart and this bound is not a race:
#: it separates "returned immediately" from "waited out the whole join".
_NO_JOIN_S = 0.5


class _PinStopBase(CustomTestCase):
    """The file-backed image fixture, as ``test_flip_image_pin_809`` builds it."""

    def setUp(self):
        super().setUp()
        self._fds = []
        self._pins = []
        # 32 MiB (the default) swallows a test image in one read, and a
        # reader that cannot be caught INSIDE a chunk cannot be held there.
        self._chunk = envs.SGLANG_PHASE_FLIP_REFILL_CHUNK_MIB.override(1)
        self._chunk.__enter__()
        weights_arena.release_flip_image_pin()
        weights_arena._FILE_BACKED_IMAGES.clear()
        weights_arena._LAYOUT_IMAGE_PHASE.clear()
        pinned_host_budget.clear_registered_posts()

    def tearDown(self):
        # Release every held reader BEFORE the fixture goes away, or a parked
        # thread outlives the test and reads into a freed buffer.
        for pin in self._pins:
            try:
                pin.release_for_teardown()
            except Exception:  # noqa: BLE001 - teardown never masks a failure
                pass
        weights_arena.release_flip_image_pin()
        weights_arena._FILE_BACKED_IMAGES.clear()
        weights_arena._LAYOUT_IMAGE_PHASE.clear()
        pinned_host_budget.clear_registered_posts()
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._chunk.__exit__(None, None, None)
        super().tearDown()

    def _image(self, d, name, payload: bytes, phase: str):
        body = bytearray(payload)
        csum = weights_arena.uint8_checksum(torch.frombuffer(body, dtype=torch.uint8))
        blob = bytes(body) + int(csum).to_bytes(8, "little", signed=True)
        total = len(blob)
        path = os.path.join(d, name)
        with open(path, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        image = torch.from_file(path, shared=True, size=total, dtype=torch.uint8)
        fd = os.open(path, os.O_RDONLY)
        self._fds.append(fd)
        weights_arena._register_file_backed_image(image, fd, total, path, None)
        weights_arena.tag_layout_image(image, phase)
        return image

    def _gated(self, nbytes, gate_after=1):
        """A pin whose reader parks INSIDE the chunk after ``gate_after`` reads.

        The gate is opened only by the test, never by a stop: that is the whole
        point here, because a fixture that released the reader on the stop
        signal could not tell a call that JOINED from one that did not.
        """

        class _Gated(weights_arena.FlipImagePin):
            def __init__(self, buf):
                super().__init__(buf)
                self.gate = threading.Event()
                self.parked = threading.Event()
                self.gate_after = gate_after
                self.chunks_read = 0

            def _read_chunk(self, meta, view, offset, length):
                if self.chunks_read >= self.gate_after:
                    self.parked.set()
                    self.gate.wait(30.0)
                n = super()._read_chunk(meta, view, offset, length)
                self.chunks_read += 1
                return n

            def release_for_teardown(self):
                self._stop.set()
                self.gate.set()
                reader = self.reader_thread()
                if reader is not None:
                    reader.join(30.0)

        pin = _Gated(torch.zeros(int(nbytes), dtype=torch.uint8))
        self._pins.append(pin)
        weights_arena.install_flip_image_pin(pin)
        return pin

    def _park(self, pin):
        self.assertTrue(
            pin.parked.wait(30.0), "the fixture never got a reader into a chunk"
        )
        self.assertTrue(pin.reader_alive(), "the held reader is not alive")

    def _wait_gone(self, pin, timeout=10.0):
        deadline = time.monotonic() + timeout
        while pin.reader_alive() and time.monotonic() < deadline:
            time.sleep(0.005)


class TestTheStopSignalDoesNotJoin(_PinStopBase):
    """``request_stop``: tell the reader to go, do not wait for it."""

    def test_request_stop_returns_while_the_reader_is_still_in_its_chunk(self):
        """The scheduler round may not wait out a read it does not need.

        The reader is held inside its second chunk for up to 30 s. The joining
        form would spend ``_PIN_STOP_JOIN_S`` (5.0 s) of the round here; the
        signal-only form returns at once and leaves the reader alive, which is
        the observable that says no join happened.
        """
        payload = os.urandom(4 * _CHUNK)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image = self._image(d, "tp.img", payload, "tp")
            pin = self._gated(image.numel())
            self.assertTrue(pin.start_prefetch(image))
            self._park(pin)

            started = time.monotonic()
            pin.request_stop()
            elapsed = time.monotonic() - started

            self.assertLess(
                elapsed,
                _NO_JOIN_S,
                f"the abandon path waited {elapsed:.3f}s for the reader; that "
                f"is a stall of the scheduler round, which is what the "
                f"signal-only stop exists to remove",
            )
            self.assertTrue(
                pin.reader_alive(),
                "the reader was gone the instant the stop returned, so this "
                "fixture did not actually exercise a non-blocking stop",
            )

            # THE SIGNAL IS REAL. Opening the gate lets the held reader finish
            # the chunk it was in and then leave: a stop that set nothing
            # would read the image to the end instead.
            pin.gate.set()
            self._wait_gone(pin)
            self.assertFalse(pin.reader_alive())
            self.assertEqual(
                pin.bytes_valid,
                2 * _CHUNK,
                "the reader ran past the chunk it was parked in, so the stop "
                "signal never reached the loop",
            )

    def test_the_stop_keeps_the_prefix_and_the_identity(self):
        """An abandon is not a discard: what was read stays the truth."""
        payload = os.urandom(4 * _CHUNK)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image = self._image(d, "tp.img", payload, "tp")
            pin = self._gated(image.numel())
            self.assertTrue(pin.start_prefetch(image))
            self._park(pin)
            identity = pin.identity
            self.assertIsNotNone(identity)

            pin.request_stop()
            pin.gate.set()
            self._wait_gone(pin)

            self.assertEqual(pin.identity, identity)
            self.assertEqual(pin.bytes_valid, 2 * _CHUNK)
            self.assertEqual(
                bytes(pin.buffer[: 2 * _CHUNK].numpy()), payload[: 2 * _CHUNK]
            )

    def test_the_read_loop_releases_the_reader_slot_when_it_leaves(self):
        """A finished loop must not look like a writer to the next arm.

        ``Thread.is_alive()`` stays True after the loop has left the buffer,
        for as long as the interpreter takes to tear the thread down. The fact
        an arm needs is whether a LOOP is still writing, so the loop releases
        the slot itself -- otherwise an arm landing in that window is refused
        (or, on the joining form, waits) for a reader that is already done.
        """
        payload = os.urandom(2 * _CHUNK)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image = self._image(d, "tp.img", payload, "tp")
            pin = self._gated(image.numel(), gate_after=64)  # never parks
            self.assertTrue(pin.start_prefetch(image))
            deadline = time.monotonic() + 30.0
            while pin.bytes_valid < image.numel() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(pin.bytes_valid, image.numel())

            deadline = time.monotonic() + 5.0
            while pin.reader_thread() is not None and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIsNone(
                pin.reader_thread(),
                "the finished read loop still holds the reader slot, so the "
                "next arm cannot tell it from a writer still filling the "
                "buffer",
            )
            self.assertFalse(pin.reader_alive())


class TestTwoWritersNeverShareOneBuffer(_PinStopBase):
    """The DANGER direction: one buffer, one writer, or the bytes are a lie."""

    def test_an_arm_for_the_other_layout_starts_no_second_writer(self):
        """RB-B-X2. Two loops on one buffer serve the wrong layout's bytes.

        ``_read_loop`` re-reads ``self._stop`` from the instance on every
        iteration and ``start_prefetch`` REPLACES that event, so a second
        reader started beside a live one does not stop the first: both write
        the buffer while ``bytes_valid`` is advanced by whichever is further,
        under the NEW identity. The refill then serves that prefix as the new
        layout's bytes and the image's own trailer cannot catch it, because
        the trailer travels with the image.

        The arm must also not JOIN the live reader: it runs on the scheduler's
        round, exactly like the abandon.
        """
        payload_tp = b"\xcd" * (4 * _CHUNK)
        payload_pp = b"\xab" * (4 * _CHUNK)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image_tp = self._image(d, "tp.img", payload_tp, "tp")
            image_pp = self._image(d, "pp.img", payload_pp, "pp")
            pin = self._gated(max(image_tp.numel(), image_pp.numel()))

            self.assertTrue(pin.start_prefetch(image_pp))
            self._park(pin)
            first = pin.reader_thread()

            started = time.monotonic()
            armed = pin.start_prefetch(image_tp)
            elapsed = time.monotonic() - started

            self.assertFalse(
                armed,
                "the arm started a read-ahead for the tp layout while the pp "
                "reader was still writing the same buffer",
            )
            self.assertLess(
                elapsed,
                _NO_JOIN_S,
                f"the arm waited {elapsed:.3f}s for the previous reader; the "
                f"arm runs on the scheduler round too",
            )
            self.assertIs(
                pin.reader_thread(),
                first,
                "a second writer was installed on the pin's one buffer",
            )
            self.assertEqual(
                len([t for t in threading.enumerate() if t.name == "flip-image-pin"]),
                1,
                "two readers are alive on one pin buffer",
            )

            # THE CORRUPTION ITSELF. Whatever the pin offers for the tp leg
            # must BE the tp image's bytes; with a second writer it is the pp
            # image's, published under the tp identity.
            served, stale = pin.consume(image_tp)
            self.assertEqual(
                bytes(pin.buffer[:served].numpy()),
                payload_tp[:served],
                f"the pin served {served} bytes for the tp leg that are not "
                f"the tp image's bytes",
            )
            self.assertEqual(served, 0)
            self.assertIsNotNone(
                stale,
                "the pin still holds the pp layout, so the tp leg's refusal "
                "of it is a STALE identity and must say so by name",
            )

    def test_a_re_arm_of_the_same_image_withdraws_a_requested_stop(self):
        """The resume the non-blocking abandon must not cost.

        The abandon signals the reader and returns; the re-arm for the SAME
        direction arrives while that reader is still in its chunk. Waiting for
        it would be the stall again, and letting the stop stand would freeze
        the prefix at whatever the abandon caught -- so the re-arm withdraws
        the request and the reader that is already filling this very identity
        carries on. ``start_prefetch``'s own comment argues for exactly this
        (*"the #834 park deadline abandons and re-arms the same direction
        repeatedly, and a restart there would leave the pin holding strictly
        LESS than it held a moment before the re-arm"*).
        """
        payload = os.urandom(4 * _CHUNK)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image = self._image(d, "tp.img", payload, "tp")
            pin = self._gated(image.numel())
            self.assertTrue(pin.start_prefetch(image))
            self._park(pin)
            first = pin.reader_thread()

            pin.request_stop()  # the abandon
            self.assertFalse(
                pin.start_prefetch(image), "a re-arm started a second reader"
            )
            self.assertIs(pin.reader_thread(), first)

            pin.gate.set()
            deadline = time.monotonic() + 30.0
            while pin.bytes_valid < image.numel() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(
                pin.bytes_valid,
                image.numel(),
                "the re-armed direction stopped where the abandon caught it, "
                "so the read-ahead the arm asked for never resumed",
            )
            self.assertEqual(
                bytes(pin.buffer[: image.numel()].numpy()), bytes(image.numpy())
            )


class TestTheCutoverKeepsItsJoin(_PinStopBase):
    """``consume`` is the one caller that may not proceed beside a writer."""

    def test_consume_refuses_a_prefix_a_writer_is_still_filling(self):
        """The bound is spent HERE, inside the no-return window, by design.

        If ``consume`` took the signal-only form it would answer while the
        reader is still writing, and the leg would DMA a prefix that is being
        overwritten under it. The reader here never leaves its chunk, so the
        join times out and the pin is forfeited BY NAME -- 0 bytes, and the
        leg reads the file exactly as it did before #809.
        """
        payload = os.urandom(4 * _CHUNK)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image = self._image(d, "tp.img", payload, "tp")
            pin = self._gated(image.numel())
            self.assertTrue(pin.start_prefetch(image))
            self._park(pin)

            with self.assertLogs(_LOGGER, level="WARNING") as caught:
                started = time.monotonic()
                served, stale = pin.consume(image)
                elapsed = time.monotonic() - started

            self.assertEqual(served, 0)
            self.assertIsNone(stale, "a reader that will not leave is not STALE")
            self.assertTrue(
                any("NOT USED" in line for line in caught.output),
                f"the forfeited pin was not named in the log: {caught.output}",
            )
            self.assertGreaterEqual(
                elapsed,
                weights_arena._PIN_STOP_JOIN_S - 0.5,
                f"consume answered in {elapsed:.3f}s without waiting out its "
                f"bound, so it can answer beside a live writer",
            )


if __name__ == "__main__":
    unittest.main()
