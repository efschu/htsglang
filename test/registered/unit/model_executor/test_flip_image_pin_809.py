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
  abandoned arm stops the reader while keeping what it already read;
* T-G1-7 the ``refill_s`` term of the ``#809 FLIP IMAGE PREFETCH complete=``
  line COVERS the transfer, on the ``pin`` arm as well as the ``pin+file``
  one -- see ``TestFlipImagePinLegClock`` for the defect that motivates it;
* T-G1-8 a re-arm publishes the new identity with a ZEROED count, never with
  the previous fill's;
* T-G1-9 ``consume`` stops the reader before the remainder leg pulls the same
  file, and a reader that will not leave its chunk forfeits the pin by name;
* T-G1-10 an unaligned prefix hands off to the file leg on a BLOCK boundary,
  which is the only observable the flooring has -- the content is identical
  either way and the cost is the O_DIRECT fd;
* T-G1-11 an abandoned arm's prefix survives the NEXT arm for the same image
  and the resumed reader starts at it, which is what the abandon site and
  ``stop_prefetch`` both promise in prose;
* T-G1-12 the #721 admission is MIN-reduced across the world group, so the
  ranks cannot disagree about a boot at a site that is past the first
  collective.
"""

import os
import re
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
                with self.assertRaises(weights_arena.WeightsArenaError) as ctx:
                    weights_arena.create_flip_image_pin(1 << 20)
        # The refusal is the GROUP-uniform one (T-G1-12f), not a bare
        # allocator raise: this site is past the distributed init, so a lone
        # raise here parks the peers in the next collective.
        self.assertIn("#809 FLIP IMAGE PIN REFUSED", str(ctx.exception))
        self.assertIn("cudaHostRegister refused", str(ctx.exception))
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


class TestFlipImagePinLegClock(_FlipImagePinBase):
    """T-G1-7: ``refill_s`` measures the TRANSFER, not the issue of it.

    THE DEFECT THIS CLOSES, and it is a measurement defect with a boot-log
    consequence rather than a content one. The pinned prefix is copied with
    ``non_blocking=True`` onto the current stream; on the ``source=pin`` arm
    the file leg is not entered at all, so nothing between that copy and the
    clock read blocks the HOST. The leg's own bytes would then be billed to
    the ``uint8_checksum`` one frame up in ``arena_refill`` -- the first host
    synchronisation after it -- and this line would print ``refill_s=0.00x``
    for a transfer that took seconds. The ``pin+file`` arm does NOT have the
    defect (``_staged_file_refill`` host-blocks on its own drain events), so
    without the wait ONE field means two different things depending on the
    arm, which is exactly the hazard ``refill_bound_phrase``'s #851/#1082
    docstring in this module exists to keep out.
    """

    _REFILL_S = re.compile(r"refill_s=([0-9.]+)")

    def _logged_refill_s(self, records):
        lines = [r for r in records if "#809 FLIP IMAGE PREFETCH complete=" in r]
        self.assertEqual(len(lines), 1, records)
        match = self._REFILL_S.search(lines[0])
        self.assertIsNotNone(match, lines[0])
        return float(match.group(1))

    def test_t_g1_7_refill_s_covers_the_pinned_copy(self):
        """The wait for the pinned copy is INSIDE the interval this reports.

        Can-fail three ways, all asserted: no wait at all (the call count),
        a wait placed after the clock is read (the reported seconds), and a
        wait handed a different buffer than the leg wrote (the identity of
        the argument). The stand-in sleeps, so the seconds are a real
        measurement of the interval rather than a mock's say-so.
        """
        payload = os.urandom(2 * _CHUNK)
        spy = _FileLegSpy()
        waited = []
        held_s = 0.05

        def _slow_await(dst):
            waited.append(dst)
            time.sleep(held_s)

        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, layout = self._image(d, "tp.img", payload, "tp")
            pin = self._install(
                weights_arena.FlipImagePin(
                    torch.zeros(image.numel(), dtype=torch.uint8)
                )
            )
            self.assertTrue(pin.start_prefetch(image))
            self._wait_complete(pin, image.numel())
            dst = torch.zeros(layout.total_bytes, dtype=torch.uint8)
            with mock.patch.object(weights_arena, "_staged_file_refill", spy):
                with mock.patch.object(
                    weights_arena, "_await_pin_copy", _slow_await, create=True
                ):
                    with self.assertLogs(_LOGGER, level="INFO") as cap:
                        source = weights_arena._refill_from_pin_and_file(
                            dst,
                            weights_arena._file_backed_meta(image),
                            layout.total_bytes,
                            image,
                        )
        self.assertEqual(source, "pin")
        self.assertEqual(
            len(waited),
            1,
            "the leg never waited for the pinned copy, so refill_s is the "
            "time to ISSUE the DMA and not the time to move the bytes",
        )
        self.assertIs(waited[0], dst)
        self.assertGreaterEqual(
            self._logged_refill_s(cap.output),
            held_s,
            "the wait for the pinned copy fell OUTSIDE the interval refill_s "
            "reports; the acceptance would read a win that did not happen",
        )

    def test_t_g1_7_the_pin_and_file_arm_waits_too(self):
        """The remainder leg's own drain does not cover the pinned prefix.

        ``_staged_file_refill`` host-blocks on ITS events; the pinned copy
        rides the current stream and is not one of them. An arm that skipped
        the wait here would leave the same field measuring two things.
        """
        payload = os.urandom(3 * _CHUNK)
        spy = _FileLegSpy()
        waited = []
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
                with mock.patch.object(
                    weights_arena, "_await_pin_copy", waited.append, create=True
                ):
                    source = weights_arena._refill_from_pin_and_file(
                        dst,
                        weights_arena._file_backed_meta(image),
                        layout.total_bytes,
                        image,
                    )
        self.assertEqual(source, "pin+file")
        self.assertEqual(len(waited), 1)
        self.assertEqual(bytes(dst.numpy()), payload)

    def test_t_g1_7_a_leg_the_pin_did_not_serve_never_waits(self):
        """No pinned bytes, no pinned copy, so nothing to wait for.

        The pre-#809 path stays exactly what it was: the file leg blocks on
        its own events and this leg adds no second synchronisation to a
        stream it never wrote to. Kills the "wait unconditionally" mutant.
        """
        payload_pp = bytes([0xAB]) * (2 * _CHUNK)
        payload_tp = os.urandom(2 * _CHUNK)
        spy = _FileLegSpy()
        waited = []
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
                with mock.patch.object(
                    weights_arena, "_await_pin_copy", waited.append, create=True
                ):
                    source = weights_arena._refill_from_pin_and_file(
                        dst,
                        weights_arena._file_backed_meta(image_tp),
                        layout_tp.total_bytes,
                        image_tp,
                    )
        self.assertEqual(source, "file")
        self.assertEqual(waited, [], "a leg with no pinned prefix waited anyway")

    def test_t_g1_7_the_wait_is_a_host_block_on_the_current_stream(self):
        """The seam's BODY, so patching it in the tests above proves something.

        A device destination blocks the host on the stream the copy was
        issued on; a host destination has no stream to wait for and must not
        reach into ``torch.cuda`` at all (this module is imported on CPU-only
        ranks and by every test in this file).
        """
        seen = []

        class _Stream:
            def synchronize(self):
                seen.append("synchronize")

        class _Dst:
            def __init__(self, is_cuda):
                self.is_cuda = is_cuda

        with mock.patch.object(torch.cuda, "current_stream", lambda: _Stream()):
            weights_arena._await_pin_copy(_Dst(True))
            self.assertEqual(seen, ["synchronize"])
            weights_arena._await_pin_copy(_Dst(False))
        self.assertEqual(
            seen,
            ["synchronize"],
            "a host destination reached for a CUDA stream that need not exist",
        )


class TestFlipImagePinReArm(_FlipImagePinBase):
    """T-G1-8: a new identity is never published with the old fill's count.

    THE HAZARD, named by ``start_prefetch``'s own comment. A reader that sees
    ``identity=B`` beside ``bytes_valid`` still carrying A's count is told
    that A's bytes are B's prefix. ``consume`` would hand that prefix out,
    ``_refill_from_pin_and_file`` would DMA the wrong layout into the arena,
    and ``arena_refill``'s post-copy checksum would abort the flip INSIDE
    the no-return window -- loud, so not silent corruption, but a flip
    killer.

    Why the suite could not see it before: the existing re-arm test uses two
    payloads of the SAME size, so a retained count equals the correct one.
    This one holds the second reader before its first chunk, which makes the
    window the comment describes observable rather than incidental.
    """

    def _holdable(self, buffer):
        class _Held(weights_arena.FlipImagePin):
            def __init__(self, buf):
                super().__init__(buf)
                self.hold = threading.Event()
                self.entered = threading.Event()
                self.arm_gate = False

            def _read_chunk(self, meta, view, offset, length):
                if self.arm_gate and offset == 0:
                    self.entered.set()
                    self.hold.wait(30.0)
                return super()._read_chunk(meta, view, offset, length)

            def stop_prefetch(self):
                # Release the held reader BEFORE joining it, or the join
                # burns its whole bound on a thread parked in this test.
                self._stop.set()
                self.hold.set()
                return super().stop_prefetch()

        return _Held(buffer)

    def test_t_g1_8_a_re_arm_publishes_a_zeroed_count(self):
        payload_pp = os.urandom(2 * _CHUNK)
        payload_tp = os.urandom(2 * _CHUNK)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image_pp, _ = self._image(d, "pp.img", payload_pp, "pp")
            image_tp, _ = self._image(d, "tp.img", payload_tp, "tp")
            pin = self._install(
                self._holdable(
                    torch.zeros(
                        max(image_pp.numel(), image_tp.numel()), dtype=torch.uint8
                    )
                )
            )
            self.assertTrue(pin.start_prefetch(image_pp))
            self._wait_complete(pin, image_pp.numel())
            self.assertEqual(pin.identity.layout, "pp")
            filled = pin.bytes_valid
            self.assertGreater(filled, 0)

            pin.arm_gate = True
            self.assertTrue(pin.start_prefetch(image_tp))
            self.assertTrue(
                pin.entered.wait(30.0),
                "the second reader never reached its first chunk",
            )
            # The window the comment names: the identity already says tp.
            self.assertEqual(pin.identity.layout, "tp")
            self.assertEqual(
                pin.bytes_valid,
                0,
                "the new identity was published beside the PREVIOUS fill's "
                "count, which offers the pp layout's bytes as the tp "
                "layout's prefix",
            )
            pin.hold.set()
            self._wait_complete(pin, image_tp.numel())
            self.assertEqual(
                bytes(pin.buffer[: image_tp.numel()].numpy()),
                bytes(image_tp.numpy()),
            )


class TestFlipImagePinConsumeStopsTheReader(_FlipImagePinBase):
    """T-G1-9: ``consume`` leaves no reader behind for the remainder leg.

    THE HAZARD, named by ``consume``'s own docstring: *"A reader still going
    would be pulling the same file the remainder leg is about to pull -- two
    streams competing for one device for bytes that only have to arrive
    once."* That is a decision about the cutover's storage device, taken
    inside the no-return window, and nothing in this suite could see it:
    replacing ``stopped = self.stop_prefetch()`` with ``stopped = True`` left
    the whole slice green, and it also makes the ``if not stopped`` refusal
    below structurally unreachable.
    """

    def test_t_g1_9_consume_stops_the_reader_before_serving(self):
        """A live reader is gone by the time ``consume`` answers."""
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
            self.assertTrue(
                pin.reader_alive(), "the fixture never got a reader to stop"
            )
            served, stale = pin.consume(image)
            self.assertIsNone(stale)
            self.assertEqual(served, _CHUNK)
            self.assertFalse(
                pin.reader_alive(),
                "the read-ahead was still pulling the image file the "
                "remainder leg is about to pull",
            )

    def test_t_g1_9b_a_reader_that_will_not_leave_forfeits_the_pin(self):
        """The ``if not stopped`` arm: no bytes served, and it says so.

        A reader that has not left its chunk is still WRITING the prefix, so
        serving that prefix would hand the leg a buffer under active
        modification. The pin is forfeited for this leg and the file carries
        it -- the same content, one leg slower.
        """
        payload = os.urandom(3 * _CHUNK)

        class _Stubborn(weights_arena.FlipImagePin):
            """Ignores the stop flag, which is what makes the join expire."""

            def __init__(self, buf):
                super().__init__(buf)
                self.wedged = threading.Event()
                self.release = threading.Event()

            def _read_chunk(self, meta, view, offset, length):
                if offset >= _CHUNK:
                    self.wedged.set()
                    self.release.wait(30.0)
                    return 0
                return super()._read_chunk(meta, view, offset, length)

        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, _ = self._image(d, "tp.img", payload, "tp")
            pin = self._install(
                _Stubborn(torch.zeros(image.numel(), dtype=torch.uint8))
            )
            self.assertTrue(pin.start_prefetch(image))
            self.assertTrue(pin.wedged.wait(30.0), "the fixture never wedged")
            try:
                with mock.patch.object(weights_arena, "_PIN_STOP_JOIN_S", 0.05):
                    with self.assertLogs(_LOGGER, level="WARNING") as cap:
                        served, stale = pin.consume(image)
            finally:
                pin.release.set()
            self.assertEqual(served, 0, "a prefix still being written was served")
            self.assertIsNone(stale, "a live reader is not a provenance failure")
            self.assertTrue(
                any("#809 flip image pin NOT USED" in m for m in cap.output),
                cap.output,
            )


class TestFlipImagePinHandOff(_FlipImagePinBase):
    """T-G1-10: the pin/file hand-off point is floored to the block size.

    THE PROPERTY, and why content assertions cannot see it. An unaligned
    resume offset is CONTENT-SAFE -- the remainder leg reads the same bytes
    either way -- so it costs only speed: ``_staged_file_refill`` picks
    O_DIRECT per ABSOLUTE offset (``:777`` ``at % _DIRECT_ALIGN == 0``), and
    an unaligned start pushes every one of its reads onto the buffered fd.
    The suite could not see the flooring because every ``bytes_valid`` it
    produces is a whole number of 1 MiB chunks, i.e. already block-aligned:
    ``from_pin % _DIRECT_ALIGN`` was 0 in every case and the statement a
    no-op under test.
    """

    def test_t_g1_10_an_unaligned_prefix_hands_off_on_a_block_boundary(self):
        payload = os.urandom(3 * _CHUNK)
        spy = _FileLegSpy()
        unaligned = _CHUNK + 3
        self.assertNotEqual(unaligned % weights_arena._DIRECT_ALIGN, 0)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, layout = self._image(d, "tp.img", payload, "tp")
            pin = self._install(
                weights_arena.FlipImagePin(
                    torch.zeros(image.numel(), dtype=torch.uint8)
                )
            )
            self.assertTrue(pin.start_prefetch(image))
            self._wait_complete(pin, image.numel())
            # THE BUFFER HOLDS THE WHOLE IMAGE and only the published count is
            # moved back: the pin's bytes are therefore CORRECT at every offset
            # this test touches, so a refill that skipped the flooring would
            # still produce identical content. That is the point -- the only
            # observable left is the offset the remainder leg is asked for.
            with pin._lock:
                pin._bytes_valid = unaligned
            dst = torch.zeros(layout.total_bytes, dtype=torch.uint8)
            with mock.patch.object(weights_arena, "_staged_file_refill", spy):
                source = weights_arena._refill_from_pin_and_file(
                    dst,
                    weights_arena._file_backed_meta(image),
                    layout.total_bytes,
                    image,
                )
        self.assertEqual(source, "pin+file")
        self.assertEqual(len(spy.calls), 1, spy.calls)
        start = spy.calls[0][0]
        self.assertEqual(
            start % weights_arena._DIRECT_ALIGN,
            0,
            "the remainder leg was resumed off a block boundary, so every "
            "one of its reads falls back to the buffered fd",
        )
        self.assertEqual(start, unaligned - unaligned % weights_arena._DIRECT_ALIGN)
        self.assertEqual(bytes(dst.numpy()), payload)


class TestFlipImagePinResume(_FlipImagePinBase):
    """T-G1-11: an abandoned arm's prefix survives the next arm for it.

    THE CLAIM THIS PINS, made verbatim at two live sites --
    ``stop_prefetch``'s docstring (*"what was read is still the truth about
    that image, so the next arm for the same direction finds it"*) and
    ``phase_flip_runtime._abandon_parked_flip`` (*"the next arm for the same
    direction finds the prefix and its identity intact"*). The early return
    in ``start_prefetch`` fires only while a reader is ALIVE, so after an
    abandon the re-arm fell through to the zeroing: the #834 park-deadline
    abandon/re-arm loop accumulated nothing, and a re-arm issued shortly
    before a cutover left the pin with strictly LESS than it had a moment
    earlier -- worse than not re-arming at all.

    THE ZEROING ITSELF IS NOT WEAKENED. It is the identity CHANGE that must
    zero, and T-G1-8 is the arm that proves it still does.
    """

    def _resumable(self, buffer, gate_after=1):
        class _Resumable(weights_arena.FlipImagePin):
            def __init__(self, buf):
                super().__init__(buf)
                self.gate = threading.Event()
                self.gate_after = gate_after
                self.chunks_read = 0
                self.offsets = []
                self.hold = threading.Event()
                self.entered = threading.Event()
                self.arm_gate = False

            def _read_chunk(self, meta, view, offset, length):
                if self.arm_gate:
                    self.offsets.append(int(offset))
                    self.entered.set()
                    self.hold.wait(30.0)
                    if self._stop.is_set():
                        return 0
                    return super()._read_chunk(meta, view, offset, length)
                if self.chunks_read >= self.gate_after:
                    self.gate.wait(30.0)
                    if self._stop.is_set():
                        return 0
                self.offsets.append(int(offset))
                n = super()._read_chunk(meta, view, offset, length)
                self.chunks_read += 1
                return n

            def stop_prefetch(self):
                # Release both gates BEFORE joining, or the join burns its
                # whole bound on a thread parked in this test.
                self._stop.set()
                self.gate.set()
                self.hold.set()
                return super().stop_prefetch()

        return _Resumable(buffer)

    def test_t_g1_11_a_same_direction_re_arm_keeps_and_resumes(self):
        payload = os.urandom(3 * _CHUNK)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, _ = self._image(d, "tp.img", payload, "tp")
            pin = self._install(
                self._resumable(torch.zeros(image.numel(), dtype=torch.uint8))
            )
            self.assertTrue(pin.start_prefetch(image))
            deadline = time.monotonic() + 30.0
            while pin.bytes_valid < _CHUNK and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(pin.bytes_valid, _CHUNK)
            identity = pin.identity
            self.assertIsNotNone(identity)

            # THE ABANDON (phase_flip_runtime._abandon_parked_flip:7865-7867).
            pin.stop_prefetch()
            self.assertFalse(pin.reader_alive())
            self.assertEqual(pin.bytes_valid, _CHUNK)
            self.assertEqual(pin.identity, identity)

            # THE NEXT ARM FOR THE SAME DIRECTION, held before its first read
            # so the count is observed in the window the comments describe.
            pin.hold.clear()
            pin.entered.clear()
            pin.arm_gate = True
            self.assertTrue(pin.start_prefetch(image))
            self.assertTrue(
                pin.entered.wait(30.0), "the second reader never reached a chunk"
            )
            self.assertEqual(pin.identity, identity)
            self.assertEqual(
                pin.bytes_valid,
                _CHUNK,
                "the re-arm discarded the prefix the abandon kept, so the "
                "read starts again from zero and the pin is worse than it "
                "was a moment earlier",
            )
            self.assertEqual(
                pin.offsets[-1],
                _CHUNK,
                "the resumed reader re-read bytes the pin already held",
            )
            pin.hold.set()
            self._wait_complete(pin, image.numel())
            self.assertEqual(
                bytes(pin.buffer[: image.numel()].numpy()), bytes(image.numpy())
            )

    def test_t_g1_11b_the_resume_is_block_floored_and_never_shrinks(self):
        """The two properties a hand-waved resume gets wrong.

        A ``preadv`` may return SHORT, which leaves the published count off a
        block boundary. Resuming there would push every remaining read onto
        the buffered fd (``_read_chunk`` takes the O_DIRECT fd only for an
        aligned offset AND length), so the resume is floored -- and the few
        bytes below the floor are re-read, which is free because they are
        bytes the pin already holds correctly. The count itself must then
        never follow the cursor DOWNWARDS: the resumed reader starts below
        what is valid, and a short read there would otherwise publish a
        prefix smaller than the buffer actually carries.
        """
        payload = os.urandom(3 * _CHUNK)
        short = 3000
        shorter = 100
        self.assertNotEqual(short % weights_arena._DIRECT_ALIGN, 0)

        class _ShortReads(weights_arena.FlipImagePin):
            def __init__(self, buf):
                super().__init__(buf)
                self.takes = [short, shorter]
                self.reads = 0
                self.park_at = 1
                self.parked = threading.Event()
                self.gate = threading.Event()
                self.offsets = []

            def _read_chunk(self, meta, view, offset, length):
                if self.reads >= self.park_at:
                    self.parked.set()
                    self.gate.wait(30.0)
                    if self._stop.is_set():
                        return 0
                take = (
                    self.takes[self.reads] if self.reads < len(self.takes) else length
                )
                self.offsets.append(int(offset))
                self.reads += 1
                return super()._read_chunk(
                    meta, view, offset, min(int(take), int(length))
                )

            def stop_prefetch(self):
                self._stop.set()
                self.gate.set()
                return super().stop_prefetch()

        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, _ = self._image(d, "tp.img", payload, "tp")
            pin = self._install(
                _ShortReads(torch.zeros(image.numel(), dtype=torch.uint8))
            )
            self.assertTrue(pin.start_prefetch(image))
            self.assertTrue(pin.parked.wait(30.0), "the first reader never parked")
            self.assertEqual(pin.bytes_valid, short)

            pin.stop_prefetch()
            self.assertFalse(pin.reader_alive())
            self.assertEqual(pin.bytes_valid, short)

            pin.gate.clear()
            pin.parked.clear()
            pin.park_at = 2
            self.assertTrue(pin.start_prefetch(image))
            self.assertTrue(pin.parked.wait(30.0), "the second reader never parked")
            self.assertEqual(
                pin.offsets[-1],
                short - short % weights_arena._DIRECT_ALIGN,
                "the resumed read started off a block boundary, so every "
                "read after it falls back to the buffered fd",
            )
            self.assertEqual(
                pin.bytes_valid,
                short,
                "a short read at the resume offset published a prefix "
                "SMALLER than the buffer holds",
            )
            pin.gate.set()
            self._wait_complete(pin, image.numel())
            self.assertEqual(
                bytes(pin.buffer[: image.numel()].numpy()), bytes(image.numpy())
            )


class TestFlipImagePinGroupAdmission(_FlipImagePinBase):
    """T-G1-12: the #721 admission is ONE verdict for the whole group.

    THE LAW AND THE SITE. ``check_and_register_pinned_post`` weighs the post
    against LIVE availability, and ``pinned_host_budget``'s module docstring
    states what that means across ranks, verbatim: *"``available`` shrinks as
    each rank pins its share, so a check against live availability run inside
    every TP worker can pass on rank 0 and raise on rank 2: a rank-divergent
    boot decision, which is an NCCL hang rather than an error."* This pin is
    created from ``Scheduler.init_model_worker`` AFTER ``init_tp_model_worker``
    (``scheduler.py:1595`` before ``:1616``), i.e. long past the first
    collective, so the rank-local raise the slice shipped could park its peers
    in the next collective for ever. The verdict is therefore MIN-reduced
    across the world group before anyone raises: every rank refuses, or none
    does.
    """

    def _budget(self, total_gb: float, available_gb: float):
        return mock.patch.object(
            pinned_host_budget,
            "pinned_host_memory_bytes",
            lambda: (int(total_gb * 1e9), int(available_gb * 1e9)),
        )

    def _allocator(self, calls):
        def _alloc(nbytes):
            calls.append(int(nbytes))
            return torch.zeros(int(nbytes), dtype=torch.uint8)

        return mock.patch.object(weights_arena, "_alloc_pinned_pin_buffer", _alloc)

    def _seam(self, seen, answer):
        def _reduce(local_ok):
            seen.append(bool(local_ok))
            return answer

        return mock.patch.object(weights_arena, "_pin_admission_is_group_wide", _reduce)

    def _scripted_seam(self, seen, answers):
        """A seam that answers a SEQUENCE, so a second verdict is visible.

        A single-answer stub cannot tell "the pin votes once" from "the pin
        votes twice"; this one records every local verdict it was handed and
        refuses to invent an answer the script does not have, so a vote that
        should not exist fails loudly instead of reading the previous one.
        """
        queue = list(answers)

        def _reduce(local_ok):
            seen.append(bool(local_ok))
            if not queue:
                raise AssertionError(
                    f"the pin took more group verdicts than the script has: {seen}"
                )
            return queue.pop(0)

        return mock.patch.object(weights_arena, "_pin_admission_is_group_wide", _reduce)

    def _unregister_spy(self, freed):
        """Watch the real seam a refused rank must use to hand its pages back.

        ``cudaHostRegister`` is a process-wide fact about an ADDRESS RANGE
        (``release_host_image``), so a buffer abandoned still registered is
        not merely leaked host RAM -- it is the rc=712 the next large host
        allocation dies on.
        """
        from sglang.srt.mem_cache.pool_host import common as pool_host_common

        def _spy(buffer):
            freed.append(int(buffer.numel()))

        return mock.patch.object(pool_host_common, "_cuda_host_unregister", _spy)

    def test_t_g1_12_a_peer_refusal_refuses_this_rank_too(self):
        """THE DANGER DIRECTION: this rank fits, a peer does not.

        Under a rank-local raise this rank would allocate, pin 8 GiB and walk
        into the next collective alone. Here it refuses with the group.
        """
        calls = []
        seen = []
        with self._budget(64.0, 40.0):
            with self._seam(seen, False):
                with self._allocator(calls):
                    with self.assertRaises(weights_arena.WeightsArenaError) as ctx:
                        weights_arena.create_flip_image_pin(1 << 20)
        self.assertEqual(seen, [True], "this rank's own verdict was not reduced")
        self.assertEqual(calls, [], "a refused group still allocated the buffer")
        msg = str(ctx.exception)
        self.assertIn("#809 FLIP IMAGE PIN REFUSED", msg)
        self.assertIn("rank", msg, "the refusal does not say the group refused")
        self.assertIsNone(weights_arena.flip_image_pin())
        self.assertEqual(
            [p.name for p in pinned_host_budget.registered_posts()],
            [],
            "a post with no allocation behind it is credited back to the "
            "next admission as though its bytes were resident (#729/#550)",
        )

    def test_t_g1_12b_a_local_refusal_reaches_the_group_first(self):
        """A rank that cannot fit must still arrive at the collective.

        Raising before the reduce is the divergence itself: this rank dies
        while its peers wait in a reduce that will never complete.
        """
        calls = []
        seen = []
        with self._budget(64.0, 12.0):
            with self._seam(seen, False):
                with self._allocator(calls):
                    with self.assertRaises(weights_arena.WeightsArenaError):
                        weights_arena.create_flip_image_pin(3 * (1024**3))
        self.assertEqual(
            seen, [False], "this rank raised without reaching the group verdict"
        )
        self.assertEqual(calls, [])
        self.assertEqual([p.name for p in pinned_host_budget.registered_posts()], [])

    def test_t_g1_12c_a_group_that_agrees_admits_the_pin(self):
        """GREEN arm: BOTH reduces are consulted on the success path too.

        Two verdicts and not one, because the two stages decide different
        facts: the ledger says the bytes were available, the allocator says
        this rank got them. A success that consulted only one of them would
        mean the other stage still has an unreduced exit somewhere.
        """
        calls = []
        seen = []
        want = 1 << 20
        with self._budget(64.0, 40.0):
            with self._seam(seen, True):
                with self._allocator(calls):
                    pin = weights_arena.create_flip_image_pin(want)
        self.assertEqual(
            seen,
            [True, True],
            "the ledger verdict and the allocation verdict are both reduced, "
            "so a rank that got through says so twice",
        )
        self.assertEqual(calls, [want])
        self.assertIs(pin, weights_arena.flip_image_pin())
        posts = {p.name: p for p in pinned_host_budget.registered_posts()}
        self.assertIn(weights_arena.FLIP_IMAGE_PIN_POST_NAME, posts)

    def test_t_g1_12d_the_seam_mins_over_the_world_group(self):
        """The seam's BODY, so patching it above proves something.

        MIN and not MAX: MAX would let a group whose weakest rank cannot hold
        the buffer proceed, and that rank then allocates the very bytes the
        ledger refused it.
        """
        from sglang.srt.distributed import parallel_state

        class _Group:
            world_size = 3
            cpu_group = object()

        recorded = {}

        def _all_reduce(tensor, op=None, group=None):
            recorded["op"] = op
            recorded["group"] = group
            recorded["sent"] = int(tensor.item())
            tensor.fill_(0)

        with mock.patch.object(torch.distributed, "is_available", lambda: True):
            with mock.patch.object(torch.distributed, "is_initialized", lambda: True):
                with mock.patch.object(torch.distributed, "all_reduce", _all_reduce):
                    with mock.patch.object(
                        parallel_state, "get_world_group", lambda: _Group()
                    ):
                        self.assertFalse(
                            weights_arena._pin_admission_is_group_wide(True),
                            "a peer's refusal did not reach this rank",
                        )
        self.assertEqual(recorded["sent"], 1)
        self.assertIs(recorded["group"], _Group.cpu_group)
        self.assertEqual(
            recorded["op"],
            torch.distributed.ReduceOp.MIN,
            "the admission is ANDed across the ranks, so the reduction is a "
            "MIN; a MAX admits a configuration the weakest rank refused",
        )

    def test_t_g1_12e_one_rank_and_no_process_group_keep_the_local_verdict(self):
        """Nothing to disagree with: the pre-#809 shape, unchanged.

        Both arms matter. Without an initialised process group there is no
        collective to enter -- every unit test in this file lives here -- and
        with a world of one there is no peer whose answer could differ.
        """
        from sglang.srt.distributed import parallel_state

        class _Alone:
            world_size = 1
            cpu_group = object()

        def _never(*args, **kwargs):
            raise AssertionError("a lone rank entered a collective")

        with mock.patch.object(torch.distributed, "is_initialized", lambda: False):
            with mock.patch.object(torch.distributed, "all_reduce", _never):
                self.assertTrue(weights_arena._pin_admission_is_group_wide(True))
                self.assertFalse(weights_arena._pin_admission_is_group_wide(False))
        with mock.patch.object(torch.distributed, "is_available", lambda: True):
            with mock.patch.object(torch.distributed, "is_initialized", lambda: True):
                with mock.patch.object(torch.distributed, "all_reduce", _never):
                    with mock.patch.object(
                        parallel_state, "get_world_group", lambda: _Alone()
                    ):
                        self.assertTrue(
                            weights_arena._pin_admission_is_group_wide(True)
                        )
                        self.assertFalse(
                            weights_arena._pin_admission_is_group_wide(False)
                        )

    def test_t_g1_12f_a_failed_allocation_refuses_with_the_group(self):
        """The ALLOCATOR's failure rides the same bus as the ledger's.

        ``_alloc_pinned_pin_buffer`` states its own design verbatim: *"NO
        FALLBACK, unlike ``_alloc_host_image_inner``."* It reaches
        ``pool_host/common.py:65-71``, which RAISES on a non-zero
        ``cudaHostRegister`` return, and the mmap under it can raise ENOMEM on
        a swapless box. That failure is MORE rank-divergent than the ledger's,
        not less: the ledger is weighed against LIVE availability BEFORE any
        rank has pinned its share, so three ranks can all read the same
        pre-pin figure and all admit -- and it is the LAST rank's multi-GiB
        registration that then fails, alone, past the ledger vote. A raise
        there parks the peers in the next collective for ever, which is a
        hang and not a refusal.
        """
        seen = []
        freed = []

        def _boom(nbytes):
            raise RuntimeError("cudaHostRegister failed (rc=2)")

        with self._budget(64.0, 40.0):
            with self._scripted_seam(seen, [True, False]):
                with mock.patch.object(
                    weights_arena, "_alloc_pinned_pin_buffer", _boom
                ):
                    with self._unregister_spy(freed):
                        with self.assertRaises(weights_arena.WeightsArenaError) as ctx:
                            weights_arena.create_flip_image_pin(1 << 20)
        self.assertEqual(
            seen,
            [True, False],
            "the allocation failure never reached a group verdict; this rank "
            "raised alone past the reduce",
        )
        msg = str(ctx.exception)
        self.assertIn("#809 FLIP IMAGE PIN REFUSED", msg)
        self.assertIn(
            "cudaHostRegister failed (rc=2)",
            msg,
            "the group-uniform refusal dropped this rank's own reason",
        )
        self.assertIn(
            "allocator",
            msg,
            "the refusal does not say which of the two stages refused",
        )
        self.assertEqual([p.name for p in pinned_host_budget.registered_posts()], [])
        self.assertIsNone(weights_arena.flip_image_pin())
        self.assertEqual(
            freed, [], "nothing was allocated, so there are no pages to hand back"
        )

    def test_t_g1_12g_a_peer_allocation_failure_refuses_this_fitting_rank(self):
        """THE DANGER DIRECTION: this rank allocated, a peer could not.

        This is the arm a rank-local raise cannot express at all. This rank
        succeeded, so nothing local tells it to stop; only the reduced verdict
        does. It must refuse anyway, and it must hand back BOTH the post
        (#729/#550: a post with no allocation behind it is credited to the
        next admission) and the pages (the registration is process-wide, so a
        buffer left registered is the next allocation's rc=712).
        """
        seen = []
        freed = []
        calls = []
        want = 1 << 20
        with self._budget(64.0, 40.0):
            with self._scripted_seam(seen, [True, False]):
                with self._allocator(calls):
                    with self._unregister_spy(freed):
                        with self.assertRaises(weights_arena.WeightsArenaError) as ctx:
                            weights_arena.create_flip_image_pin(want)
        self.assertEqual(
            seen,
            [True, True],
            "this rank's own allocation verdict was never reduced, so a peer "
            "that failed to allocate could not stop it",
        )
        self.assertEqual(calls, [want])
        msg = str(ctx.exception)
        self.assertIn("#809 FLIP IMAGE PIN REFUSED", msg)
        self.assertIn(
            "a peer rank",
            msg,
            "the refusal does not say the group, not this rank, refused",
        )
        self.assertEqual(
            freed,
            [want],
            "this rank sailed on holding pages the group had just refused",
        )
        self.assertEqual([p.name for p in pinned_host_budget.registered_posts()], [])
        self.assertIsNone(weights_arena.flip_image_pin())


if __name__ == "__main__":
    unittest.main()
