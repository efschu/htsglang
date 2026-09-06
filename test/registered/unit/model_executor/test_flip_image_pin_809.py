"""#809: the INCOMING layout's flip image is pinned and read ahead at ARM time.

THE DEFECT THIS CLOSES, measured on Boot 10/11. The two-file flip image lives
on the ZFS pool (``environ.py:335``/``:352``) and the cutover reads it with
``preadv`` at the moment the arena has to be rewritten: refill legs of
11.0-12.7 s at 1.3-1.4 GB/s, i.e. STORAGE-BOUND, while the x4 PCIe link alone
would move the same 7 GiB in ~1.5 s (#690: 4.93 GB/s H2D on rank 1). The read
happens inside the no-return window, and the whole drain before it -- seconds
of it -- is spent waiting with the storage idle.

WHAT THIS ADDS, and what it deliberately does NOT. It does not add a second
carrier of the weights: the file stays the durable image and the pin is a
READ-AHEAD of it. One page-locked host buffer per rank, sized to the LARGER of
the rank's two layout images, is filled from the incoming layout's image file
while the flip drains; at the cutover the refill takes the leading
``[0, bytes_valid)`` bytes from that buffer as a real DMA and falls through to
the existing file path for ``[bytes_valid, size)``. Content is identical by
construction on every arm -- which is what T-G1-2 and T-G1-3 assert -- so the
change is TIMING only.

WHY THE IDENTITY RECORD IS THE LOAD-BEARING PART. A pin that is read without
checking WHICH image filled it is a silent weight-corruption path: the trailer
travels with the image, so a leg served the other layout's bytes out of the pin
would checksum GREEN against its own trailer whenever the two layouts happen to
be the same size (the same hazard ``_LAYOUT_IMAGE_PHASE`` exists for). T-G1-3
is that danger direction, and it fills the pin with the OTHER layout's bytes so
that skipping the check produces wrong bytes rather than merely a wrong label.

WHAT IS PINNED HERE:

* T-G1-1 a COMPLETE pin is the source -- the file leg is not entered at all;
* T-G1-2 a PARTIAL pin serves its prefix and the file serves the remainder,
  and the concatenation is byte-identical to the file;
* T-G1-3 an identity mismatch (other layout, or an older registration
  generation) REFUSES THE PIN by name and refills from the file;
* T-G1-4 the #721 pinned-host gate is asked BEFORE the buffer is allocated,
  and a failed allocation takes its post back with it (#729/#550);
* T-G1-5 with the env unset there is no buffer, no post and no change to the
  refill path;
* T-G1-6 a second arm for the same image starts no second reader, and an
  abandoned arm stops the reader while keeping what it already read.
"""

import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import pinned_host_budget
from sglang.srt.model_executor import weights_arena
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.model_executor.weights_arena"

#: One reader chunk under the test env below. The reader reads in
#: ``_refill_chunk_bytes()`` units, whose floor is 1 MiB, so a payload that is
#: to be filled PARTIALLY has to be several of them.
_CHUNK = 1 << 20


def _make_layout(nbytes: int):
    """A one-slot layout of ``nbytes``, which is all a refill needs."""
    from sglang.srt.model_executor.weights_arena import ArenaLayout, ArenaSlot

    slot = ArenaSlot(
        name="w",
        offset=0,
        nbytes=nbytes,
        dtype=torch.uint8,
        shape=(nbytes,),
        stride=(1,),
    )
    return ArenaLayout(slots=[slot], total_bytes=nbytes, aliases=[])


class _FileLegSpy:
    """Stands in for ``_staged_file_refill``, which stages through CUDA.

    Records the range it was asked for and moves exactly that range, so a test
    can tell "the pin served these bytes" from "the file did" by looking at
    what the spy was NOT asked to write.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, dst, meta, nbytes, timing=None, start=0):
        self.calls.append((int(start), int(nbytes)))
        view = dst.numpy()
        got = int(start)
        while got < int(nbytes):
            r = os.preadv(meta.fd, [memoryview(view)[got : int(nbytes)]], got)
            if r == 0:
                raise AssertionError("short read in the test file leg")
            got += r


def _GatedPin(buffer, gate_after=1):
    """A pin whose reader blocks after ``gate_after`` chunks.

    The partial-fill tests need a DETERMINISTIC ``bytes_valid``, and a race
    against the pool's read rate is not one. Blocking inside the chunk read is
    the only place a test can hold the reader without reaching into its state.

    Built inside a function rather than at module scope so that a tree WITHOUT
    ``FlipImagePin`` fails each test on its own line instead of collapsing the
    whole module into one collection error -- a red tally has to say which
    behaviour is missing.
    """

    class _Gated(weights_arena.FlipImagePin):
        def __init__(self, buf):
            super().__init__(buf)
            self.gate = threading.Event()
            self.gate_after = gate_after
            self.chunks_read = 0

        def _read_chunk(self, meta, view, offset, length):
            if self.chunks_read >= self.gate_after:
                self.gate.wait(30.0)
                if self._stop.is_set():
                    return 0
            n = super()._read_chunk(meta, view, offset, length)
            self.chunks_read += 1
            return n

        def stop_prefetch(self):
            # STOP BEFORE THE GATE, or the woken reader races the stop flag
            # and reads one more chunk than the test asked for.
            self._stop.set()
            self.gate.set()
            return super().stop_prefetch()

    return _Gated(buffer)


class _FlipImagePinBase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._fds = []
        # The reader's chunk defaults to 32 MiB (``environ.py:416``), which
        # swallows a test-sized image in ONE read -- and a partial fill that
        # cannot be produced cannot be asserted. 1 MiB is the floor the
        # accessor allows and keeps every offset block-aligned.
        self._chunk = envs.SGLANG_PHASE_FLIP_REFILL_CHUNK_MIB.override(1)
        self._chunk.__enter__()
        weights_arena.release_flip_image_pin()
        weights_arena._FILE_BACKED_IMAGES.clear()
        weights_arena._LAYOUT_IMAGE_PHASE.clear()
        pinned_host_budget.clear_registered_posts()

    def tearDown(self):
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
        """A file-backed layout image on disk, registered and tagged as at boot."""
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
        return image, _make_layout(len(payload))

    def _install(self, pin):
        weights_arena.install_flip_image_pin(pin)
        return pin

    def _wait_complete(self, pin, nbytes, timeout=30.0):
        deadline = time.monotonic() + timeout
        while pin.bytes_valid < nbytes and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(
            pin.bytes_valid, nbytes, "the read-ahead never finished the image"
        )


class TestFlipImagePinRefill(_FlipImagePinBase):
    """T-G1-1/2/3/5: which buffer the refill actually reads from."""

    def test_t_g1_1_complete_pin_is_the_only_source(self):
        """T-G1-1: a complete pin serves the leg; the file leg is not entered.

        Can-fail: a refill that ignores the pin calls the spy, and a refill
        that copies nothing leaves the arena zeroed. Both are asserted, so the
        only way through is "the pin wrote every byte".
        """
        payload = os.urandom(2 * _CHUNK)
        spy = _FileLegSpy()
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, layout = self._image(d, "tp.img", payload, "tp")
            pin = self._install(
                weights_arena.FlipImagePin(
                    torch.zeros(image.numel(), dtype=torch.uint8)
                )
            )
            with self.assertLogs(_LOGGER, level="INFO") as cap:
                self.assertTrue(pin.start_prefetch(image))
                self._wait_complete(pin, image.numel())
            self.assertTrue(
                any("#809 FLIP IMAGE PREFETCH start" in m for m in cap.output),
                cap.output,
            )
            dst = torch.zeros(layout.total_bytes, dtype=torch.uint8)
            with mock.patch.object(weights_arena, "_staged_file_refill", spy):
                with self.assertLogs(_LOGGER, level="INFO") as cap:
                    source = weights_arena._refill_from_pin_and_file(
                        dst,
                        weights_arena._file_backed_meta(image),
                        layout.total_bytes,
                        image,
                    )
            self.assertEqual(source, "pin")
            self.assertEqual(spy.calls, [], "the file leg was entered anyway")
            self.assertEqual(bytes(dst.numpy()), payload)
            self.assertTrue(
                any(
                    "#809 FLIP IMAGE PREFETCH complete=" in m and "source=pin" in m
                    for m in cap.output
                ),
                cap.output,
            )

    def test_t_g1_2_partial_pin_then_file_is_byte_identical(self):
        """T-G1-2: prefix from the pin, remainder from the file, one image.

        Can-fail: copying ``size`` instead of ``bytes_valid`` from the pin
        leaves the un-read tail as whatever the buffer held (zeros here), and
        the concatenation stops matching the file.
        """
        payload = os.urandom(3 * _CHUNK)
        spy = _FileLegSpy()
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, layout = self._image(d, "tp.img", payload, "tp")
            pin = self._install(
                _GatedPin(torch.zeros(image.numel(), dtype=torch.uint8), gate_after=1)
            )
            self.assertTrue(pin.start_prefetch(image))
            deadline = time.monotonic() + 30.0
            while pin.bytes_valid < _CHUNK and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(pin.bytes_valid, _CHUNK)
            dst = torch.zeros(layout.total_bytes, dtype=torch.uint8)
            with mock.patch.object(weights_arena, "_staged_file_refill", spy):
                source = weights_arena._refill_from_pin_and_file(
                    dst,
                    weights_arena._file_backed_meta(image),
                    layout.total_bytes,
                    image,
                )
        self.assertEqual(source, "pin+file")
        self.assertEqual(spy.calls, [(_CHUNK, layout.total_bytes)])
        self.assertEqual(bytes(dst.numpy()), payload)

    def test_t_g1_3_other_layout_in_the_pin_is_refused_by_name(self):
        """T-G1-3, THE DANGER DIRECTION: the pin holds the OTHER layout.

        The two payloads differ, so a refill that skips the identity check
        serves ``payload_pp`` bytes for the ``tp`` layout -- silent weight
        corruption, which is what this test exists to make loud.
        """
        payload_pp = bytes([0xAB]) * (2 * _CHUNK)
        payload_tp = os.urandom(2 * _CHUNK)
        spy = _FileLegSpy()
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image_pp, _ = self._image(d, "pp.img", payload_pp, "pp")
            image_tp, layout_tp = self._image(d, "tp.img", payload_tp, "tp")
            pin = self._install(
                weights_arena.FlipImagePin(
                    torch.zeros(
                        max(image_pp.numel(), image_tp.numel()), dtype=torch.uint8
                    )
                )
            )
            self.assertTrue(pin.start_prefetch(image_pp))
            self._wait_complete(pin, image_pp.numel())
            dst = torch.zeros(layout_tp.total_bytes, dtype=torch.uint8)
            with mock.patch.object(weights_arena, "_staged_file_refill", spy):
                with self.assertLogs(_LOGGER, level="INFO") as cap:
                    source = weights_arena._refill_from_pin_and_file(
                        dst,
                        weights_arena._file_backed_meta(image_tp),
                        layout_tp.total_bytes,
                        image_tp,
                    )
        self.assertEqual(source, "file")
        self.assertEqual(spy.calls, [(0, layout_tp.total_bytes)])
        self.assertEqual(bytes(dst.numpy()), payload_tp)
        self.assertTrue(
            any("#809 FLIP IMAGE PIN STALE" in m for m in cap.output), cap.output
        )

    def test_t_g1_3b_older_generation_of_the_same_path_is_refused(self):
        """T-G1-3, second half: the SAME path, an older registration.

        The path and the layout alone cannot see an image that was released
        and re-registered underneath the pin. The generation term can, and it
        has a real writer -- every ``_register_file_backed_image`` call.
        """
        payload = os.urandom(2 * _CHUNK)
        spy = _FileLegSpy()
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, layout = self._image(d, "tp.img", payload, "tp")
            pin = self._install(
                weights_arena.FlipImagePin(
                    torch.zeros(image.numel(), dtype=torch.uint8)
                )
            )
            self.assertTrue(pin.start_prefetch(image))
            self._wait_complete(pin, image.numel())
            meta = weights_arena._file_backed_meta(image)
            # The image is re-registered: a new generation for the same path.
            weights_arena._register_file_backed_image(
                image, meta.fd, meta.nbytes, meta.path, None
            )
            dst = torch.zeros(layout.total_bytes, dtype=torch.uint8)
            with mock.patch.object(weights_arena, "_staged_file_refill", spy):
                with self.assertLogs(_LOGGER, level="INFO") as cap:
                    source = weights_arena._refill_from_pin_and_file(
                        dst,
                        weights_arena._file_backed_meta(image),
                        layout.total_bytes,
                        image,
                    )
        self.assertEqual(source, "file")
        self.assertEqual(spy.calls, [(0, layout.total_bytes)])
        self.assertEqual(bytes(dst.numpy()), payload)
        self.assertTrue(
            any("#809 FLIP IMAGE PIN STALE" in m for m in cap.output), cap.output
        )

    def test_t_g1_5_no_pin_is_the_untouched_path(self):
        """T-G1-5 (GREEN pin): with no pin installed nothing changes.

        The file leg is asked for the WHOLE image from offset 0, exactly as it
        is today, and the #809 line is not emitted at all -- an inert default
        that still logged would be a boot-log regression of its own.
        """
        payload = os.urandom(2 * _CHUNK)
        spy = _FileLegSpy()
        self.assertIsNone(weights_arena.flip_image_pin())
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, layout = self._image(d, "tp.img", payload, "tp")
            dst = torch.zeros(layout.total_bytes, dtype=torch.uint8)
            with mock.patch.object(weights_arena, "_staged_file_refill", spy):
                with self.assertNoLogs(_LOGGER, level="INFO"):
                    source = weights_arena._refill_from_pin_and_file(
                        dst,
                        weights_arena._file_backed_meta(image),
                        layout.total_bytes,
                        image,
                    )
        self.assertEqual(source, "file")
        self.assertEqual(spy.calls, [(0, layout.total_bytes)])
        self.assertEqual(bytes(dst.numpy()), payload)
        self.assertEqual(
            [
                p
                for p in pinned_host_budget.registered_posts()
                if p.name == weights_arena.FLIP_IMAGE_PIN_POST_NAME
            ],
            [],
        )

    def test_t_g1_5b_default_env_is_off(self):
        """T-G1-5 (GREEN pin): the env default is OFF, and it switches."""
        self.assertFalse(weights_arena.flip_image_pin_enabled())
        with envs.SGLANG_PHASE_FLIP_IMAGE_PIN_INCOMING.override(True):
            self.assertTrue(weights_arena.flip_image_pin_enabled())

    def test_g1_c1_pin_without_the_two_file_arm_is_refused(self):
        """G1-C1: the pin is valid only with FILE_BACKED + TWO_FILE.

        Same refusal shape as ``require_two_file_preconditions``
        (``weights_arena.py:1367``): the pin reads the incoming layout's OWN
        file, and under one rotating image there is no such file.
        """
        with envs.SGLANG_PHASE_FLIP_IMAGE_PIN_INCOMING.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
                with envs.SGLANG_PHASE_FLIP_IMAGE_TWO_FILE.override(False):
                    with self.assertRaises(weights_arena.WeightsArenaError):
                        weights_arena.require_pin_preconditions()
            with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(False):
                with envs.SGLANG_PHASE_FLIP_IMAGE_TWO_FILE.override(True):
                    with self.assertRaises(weights_arena.WeightsArenaError):
                        weights_arena.require_pin_preconditions()
            with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
                with envs.SGLANG_PHASE_FLIP_IMAGE_TWO_FILE.override(True):
                    weights_arena.require_pin_preconditions()

    def test_g1_c4_arena_refill_cpu_routing_is_unchanged(self):
        """The CPU arena still takes the plain copy, pin or no pin.

        The staged path exists because the destination is a device; routing a
        host arena into it would be a new failure mode, so this pins the
        condition my branch sits behind.
        """
        payload = os.urandom(4096)
        spy = _FileLegSpy()
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, layout = self._image(d, "tp.img", payload, "tp")
            arena = torch.zeros(layout.total_bytes, dtype=torch.uint8)
            with envs.SGLANG_PHASE_FLIP_REFILL_STAGED.override(True):
                with mock.patch.object(weights_arena, "_staged_file_refill", spy):
                    weights_arena.arena_refill(arena, layout, image)
        self.assertEqual(spy.calls, [])
        self.assertEqual(bytes(arena.numpy()), payload)


class TestFlipImagePinBudget(_FlipImagePinBase):
    """T-G1-4: the #721 gate, and the #729 revert."""

    def _budget(self, total_gb: float, available_gb: float):
        return mock.patch.object(
            pinned_host_budget,
            "pinned_host_memory_bytes",
            lambda: (int(total_gb * 1e9), int(available_gb * 1e9)),
        )

    def test_t_g1_4_over_commit_refuses_before_allocating(self):
        """T-G1-4: the gate is asked FIRST, and it names what it saw.

        Can-fail two ways, both asserted: registering the post AFTER the
        allocation would let a refused configuration allocate first (the
        allocator counter), and a refusal without the numbers is not
        actionable (the message).
        """
        calls = []

        def _alloc(nbytes):
            calls.append(int(nbytes))
            return torch.zeros(int(nbytes), dtype=torch.uint8)

        with self._budget(64.0, 12.0):
            with mock.patch.object(weights_arena, "_alloc_pinned_pin_buffer", _alloc):
                with self.assertRaises(weights_arena.WeightsArenaError) as ctx:
                    weights_arena.create_flip_image_pin(3 * (1024**3))
        self.assertEqual(calls, [], "the buffer was allocated before the gate ran")
        msg = str(ctx.exception)
        self.assertIn("#809 FLIP IMAGE PIN REFUSED", msg)
        self.assertIn(weights_arena.FLIP_IMAGE_PIN_POST_NAME, msg)
        self.assertIn("reserve", msg)
        self.assertIn("available", msg)
        self.assertIsNone(weights_arena.flip_image_pin())
        self.assertEqual([p.name for p in pinned_host_budget.registered_posts()], [])

    def test_t_g1_4_fits_registers_the_post(self):
        """T-G1-4 (GREEN pin): a pin that fits is a NAMED post in the ledger."""
        want = 1 << 20
        with self._budget(64.0, 12.0):
            with mock.patch.object(
                weights_arena,
                "_alloc_pinned_pin_buffer",
                lambda n: torch.zeros(int(n), dtype=torch.uint8),
            ):
                pin = weights_arena.create_flip_image_pin(want)
        self.assertIs(pin, weights_arena.flip_image_pin())
        self.assertEqual(pin.nbytes, want)
        posts = {p.name: p for p in pinned_host_budget.registered_posts()}
        self.assertIn(weights_arena.FLIP_IMAGE_PIN_POST_NAME, posts)
        self.assertEqual(posts[weights_arena.FLIP_IMAGE_PIN_POST_NAME].nbytes, want)

    def test_t_g1_4_failed_allocation_takes_its_post_back(self):
        """#729/#550: a post that never allocated must not stay in the ledger.

        Left behind, it is credited back to the NEXT admission as though its
        bytes were resident, so the registry waves through the over-commit it
        exists to refuse.
        """

        def _boom(nbytes):
            raise RuntimeError("cudaHostRegister refused")

        with self._budget(64.0, 40.0):
            with mock.patch.object(weights_arena, "_alloc_pinned_pin_buffer", _boom):
                with self.assertRaises(RuntimeError):
                    weights_arena.create_flip_image_pin(1 << 20)
        self.assertEqual([p.name for p in pinned_host_budget.registered_posts()], [])
        self.assertIsNone(weights_arena.flip_image_pin())


class TestFlipImagePinReader(_FlipImagePinBase):
    """T-G1-6: one reader per arm, and an abandoned arm keeps what it read."""

    def test_t_g1_6_second_arm_starts_no_second_reader(self):
        payload = os.urandom(3 * _CHUNK)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, _ = self._image(d, "tp.img", payload, "tp")
            pin = self._install(
                _GatedPin(torch.zeros(image.numel(), dtype=torch.uint8), gate_after=1)
            )
            self.assertTrue(pin.start_prefetch(image))
            deadline = time.monotonic() + 30.0
            while pin.bytes_valid < _CHUNK and time.monotonic() < deadline:
                time.sleep(0.005)
            first = pin.reader_thread()
            self.assertTrue(pin.reader_alive())
            self.assertFalse(
                pin.start_prefetch(image), "a second arm started a second read"
            )
            self.assertIs(pin.reader_thread(), first)
            pin.stop_prefetch()
        self.assertFalse(pin.reader_alive())

    def test_t_g1_6_abandoned_arm_keeps_identity_and_bytes(self):
        payload = os.urandom(3 * _CHUNK)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, _ = self._image(d, "tp.img", payload, "tp")
            pin = self._install(
                _GatedPin(torch.zeros(image.numel(), dtype=torch.uint8), gate_after=1)
            )
            self.assertTrue(pin.start_prefetch(image))
            deadline = time.monotonic() + 30.0
            while pin.bytes_valid < _CHUNK and time.monotonic() < deadline:
                time.sleep(0.005)
            identity = pin.identity
            self.assertIsNotNone(identity)
            pin.stop_prefetch()
            self.assertFalse(pin.reader_alive())
            self.assertEqual(pin.identity, identity)
            self.assertEqual(pin.bytes_valid, _CHUNK)
            self.assertEqual(bytes(pin.buffer[:_CHUNK].numpy()), payload[:_CHUNK])

    def test_t_g1_6_the_buffer_is_allocated_once_not_per_arm(self):
        """Mutant (d): allocating at ARM puts a multi-GiB cudaHostAlloc inside
        the drain window. The buffer address is the observable that says the
        allocation did not move there."""
        payload_pp = os.urandom(2 * _CHUNK)
        payload_tp = os.urandom(2 * _CHUNK)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image_pp, _ = self._image(d, "pp.img", payload_pp, "pp")
            image_tp, _ = self._image(d, "tp.img", payload_tp, "tp")
            size = max(image_pp.numel(), image_tp.numel())
            pin = self._install(
                weights_arena.FlipImagePin(torch.zeros(size, dtype=torch.uint8))
            )
            before = pin.buffer.data_ptr()
            self.assertTrue(pin.start_prefetch(image_pp))
            self._wait_complete(pin, image_pp.numel())
            self.assertEqual(pin.buffer.data_ptr(), before)
            self.assertTrue(pin.start_prefetch(image_tp))
            self._wait_complete(pin, image_tp.numel())
            self.assertEqual(pin.buffer.data_ptr(), before)
            self.assertEqual(pin.identity.layout, "tp")


if __name__ == "__main__":
    unittest.main()
