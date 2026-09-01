"""#1078: TWO flip-image files per rank, so the flip leg never copies back.

THE DEFECT THIS CLOSES, measured on boot_855_1078spec_1677a9d463_0901_155207.
Under the file-backed image arm the refill leg is 41.065-67.246 s per rank per
leg, and 94.7-95.2 % of it is ONE term::

    PP0 pp_to_tp  63.911s = save 2.805 + d2h-issue 60.692 + h2d-issue 0.098
                           + wait 0.016 + checksum 0.271 + ring 0.001

``d2h-issue`` is a timer around what is nominally an ENQUEUE
(rotation_executor.py:666-672). It holds 60.692 s because the copy-back's
destination is the file-backed host image -- a ZFS ``MAP_SHARED`` mapping that
is never ``cudaHostRegister``'d (weights_arena.py:752-865 allocates it and
calls no register), so ``copy_(non_blocking=True)`` cannot be asynchronous and
the bytes go out through the mmap write path at 153-226 MiB/s. The same file on
the same pool READS at 2 595 MiB/s buffered and 8 304 MiB/s O_DIRECT
(weights_arena.py:438-445). The H2D lane in the same loop, whose source IS
pinned, costs 0.236 ms per chunk against the D2H's 141.5 ms.

THE FIX IS NOT A FASTER COPY-BACK. The copy-back moves IMMUTABLE bytes:
``rotation_plan.py:16-19`` says so itself -- "nothing is being saved; it is
residency PLACEMENT for the next flip". Placement is only necessary because
there is ONE image file per rank holding whichever layout rests. With TWO
files, a leg reads the incoming layout from ITS OWN file and DISCARDS the
outgoing arena content, and there is no copy-back to make fast.

WHY THIS IS NOT THE DUAL PIN W26 KILLED. ``phase_flip_boot.py:1861-1865``
rejects keeping both images -- correctly, for PINNED images: two lifetime pins
are 55.99 GiB across the three ranks on this rig, the same class as W26's
68.7 GiB, against #721's measured cgroup peak of 111.3 of 118 GiB. File-backed
images are reclaimable page cache, not a pinned post, so the same two images
cost DISK (+27.15 GiB against 501 GiB free) and no locked RAM. The scheme is
therefore VALID ONLY under ``SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED``, and F4
below makes that a refusal in code rather than a sentence in a docstring.

THE INVARIANT MOVES, and naming the new one is half the change. Today's
trailer contract is SELF-REFERENTIAL: leg N writes the trailer that leg N+1
verifies (rotation_executor.py:604, :707-709, :712-719), so it certifies "the
bytes that left the arena came back intact" and NOT "these are the boot
weights". An in-place mutation of an arena-backed weight page during serving is
copied out by leg N, covered by leg N's own trailer, and verified GREEN by leg
N+1 -- forever, undetected. Under two files the trailer is a BOOT CONSTANT
describing its layout, written once and never rewritten, and every leg checks
BOTH directions against it. F2 is therefore not only a test: on metal it is the
first instrument this corpus has ever had for that mutation.

WHAT IS PINNED HERE:

* F1 a leg reads the file of the layout it is streaming in -- and the wrong
  file is refused STRUCTURALLY, not by a size coincidence;
* F2 a leg that discards a MUTATED arena refuses instead of silently reverting
  it to the boot snapshot;
* F3 the default (single-image) arm is untouched and still writes its trailer;
* F4 two files without file-backed images is refused, because that is the pin.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor import weights_arena
from sglang.srt.model_executor.weights_arena import (
    ArenaLayout,
    ArenaSlot,
    WeightsArenaError,
    image_from_tensors,
    uint8_checksum,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _layout(nbytes: int) -> ArenaLayout:
    """A one-slot uint8 layout, which is all the refill contract needs."""
    slot = ArenaSlot(
        name="w",
        offset=0,
        nbytes=nbytes,
        dtype=torch.uint8,
        shape=(nbytes,),
        stride=(1,),
    )
    return ArenaLayout(slots=[slot], total_bytes=nbytes, aliases=[])


def _image(layout: ArenaLayout, fill: int) -> torch.Tensor:
    """A boot-shaped image: payload + the 8-byte trailer, built once."""
    named = {"w": torch.full((layout.total_bytes,), fill, dtype=torch.uint8)}
    return image_from_tensors(named, layout, pin=False)


class TestTwoFilePreconditions(CustomTestCase):
    """F4: the arm may not exist without the file-backed images."""

    def test_two_file_without_file_backed_is_refused(self):
        """Two PINNED lifetime images are the dual pin W26 OOM-killed.

        Can-fail: drop the refusal and this returns None instead of raising,
        which is exactly how the scheme would reach a boot as a silent pin.
        """
        with envs.SGLANG_PHASE_FLIP_IMAGE_TWO_FILE.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(False):
                with self.assertRaises(WeightsArenaError) as caught:
                    weights_arena.require_two_file_preconditions()
        self.assertIn("SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED", str(caught.exception))

    def test_two_file_with_file_backed_is_allowed(self):
        """The other direction, or the refusal above would prove nothing."""
        with envs.SGLANG_PHASE_FLIP_IMAGE_TWO_FILE.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
                weights_arena.require_two_file_preconditions()

    def test_gate_is_off_by_default(self):
        """F3's precondition: the new arm is opt-in, so the default is the old
        path byte for byte."""
        self.assertFalse(envs.SGLANG_PHASE_FLIP_IMAGE_TWO_FILE.get())
        with envs.SGLANG_PHASE_FLIP_IMAGE_TWO_FILE.override(False):
            self.assertFalse(weights_arena.two_file_images_enabled())


class TestBootAnchor(CustomTestCase):
    """F2: the mutation detector -- the half that has no equivalent today."""

    def test_unmutated_arena_passes(self):
        layout = _layout(4096)
        image = _image(layout, 7)
        arena = torch.zeros(8192, dtype=torch.uint8)
        arena[: layout.total_bytes] = 7
        weights_arena.verify_boot_anchor(arena, layout, image, "pp")

    def test_mutated_arena_is_refused(self):
        """ONE byte is enough, and the message must name the mutation.

        Can-fail: remove the comparison and a mutated layout is silently
        reverted to its boot snapshot on the next leg. Today's scheme has the
        opposite failure -- it CARRIES the mutation forward under a trailer it
        wrote itself -- and neither is visible without this check.
        """
        layout = _layout(4096)
        image = _image(layout, 7)
        arena = torch.zeros(8192, dtype=torch.uint8)
        arena[: layout.total_bytes] = 7
        arena[1234] = 8  # a weight page written while the phase was serving
        with self.assertRaises(WeightsArenaError) as caught:
            weights_arena.verify_boot_anchor(arena, layout, image, "pp")
        msg = str(caught.exception)
        # Case-insensitive: the refusal shouts the verdict on purpose, and the
        # test pins the CONTENT, not the typography.
        self.assertIn("mutated while serving", msg.lower())
        self.assertIn("'pp'", msg)
        self.assertIn("4096", msg)  # the offset range it covers

    def test_anchor_is_the_boot_trailer_not_a_recomputed_one(self):
        """The anchor must come from the IMAGE, not from the arena.

        Can-fail: recompute `want` from the arena and the check becomes a
        tautology that passes on every mutation -- which is precisely the
        self-referential shape of today's contract.
        """
        layout = _layout(4096)
        image = _image(layout, 7)
        stored = int(image[layout.total_bytes :].clone().view(torch.int64).item())
        self.assertEqual(
            stored, uint8_checksum(torch.full((4096,), 7, dtype=torch.uint8))
        )


class TestTwoFileLeg(CustomTestCase):
    """F1: a leg reads its OWN file, and the wrong one is refused."""

    def setUp(self):
        self.pp = _layout(4096)
        self.tp = _layout(6144)  # the layouts differ in size, as they do on metal
        self.image_pp = _image(self.pp, 3)
        self.image_tp = _image(self.tp, 9)
        weights_arena.tag_layout_image(self.image_pp, "pp")
        weights_arena.tag_layout_image(self.image_tp, "tp")
        self.arena = torch.zeros(8192, dtype=torch.uint8)
        self.arena[: self.pp.total_bytes] = 3  # the boot primed PP

    def tearDown(self):
        weights_arena._LAYOUT_IMAGE_PHASE.clear()

    def test_leg_streams_the_incoming_layout_in(self):
        weights_arena.two_file_leg(
            arena=self.arena,
            incoming_layout=self.tp,
            incoming_image=self.image_tp,
            incoming_phase="tp",
            outgoing_layout=self.pp,
            outgoing_image=self.image_pp,
            outgoing_phase="pp",
        )
        self.assertTrue(bool((self.arena[: self.tp.total_bytes] == 9).all()))

    def test_leg_does_not_write_the_outgoing_image(self):
        """The whole point: no copy-back, so the outgoing image is untouched.

        Can-fail: reinstate a copy-back and the image's bytes change, which is
        the 60.692 s this ticket removes.
        """
        before = self.image_pp.clone()
        weights_arena.two_file_leg(
            arena=self.arena,
            incoming_layout=self.tp,
            incoming_image=self.image_tp,
            incoming_phase="tp",
            outgoing_layout=self.pp,
            outgoing_image=self.image_pp,
            outgoing_phase="pp",
        )
        self.assertTrue(bool(torch.equal(before, self.image_pp)))

    def test_wrong_file_is_refused_structurally(self):
        """THE SILENT CORRUPTION FORM. A leg handed the other phase's image
        must refuse.

        The size check in ``arena_refill`` catches this ONLY because the two
        layouts happen to differ in size; on a rank where they did not, the
        wrong layout would load and its own trailer would verify GREEN,
        because the trailer travels with the image it belongs to. The phase
        tag is what makes the refusal structural.

        Can-fail: drop ``require_layout_image`` and this test loads the PP
        image under the TP layout.
        """
        with self.assertRaises(WeightsArenaError) as caught:
            weights_arena.two_file_leg(
                arena=self.arena,
                incoming_layout=self.tp,
                incoming_image=self.image_pp,  # <-- the other phase's file
                incoming_phase="tp",
                outgoing_layout=self.pp,
                outgoing_image=self.image_pp,
                outgoing_phase="pp",
            )
        self.assertIn("tagged 'pp'", str(caught.exception))

    def test_wrong_file_is_refused_even_at_equal_sizes(self):
        """The size coincidence removed, so only the tag can answer.

        This is the case ``arena_refill``'s size check CANNOT see, and it is
        the reason the tag exists rather than a comment saying sizes differ.
        """
        same_pp = _layout(4096)
        same_tp = _layout(4096)
        img_pp = _image(same_pp, 3)
        img_tp = _image(same_tp, 9)
        weights_arena.tag_layout_image(img_pp, "pp")
        weights_arena.tag_layout_image(img_tp, "tp")
        arena = torch.zeros(4096, dtype=torch.uint8)
        arena[:] = 3
        with self.assertRaises(WeightsArenaError):
            weights_arena.two_file_leg(
                arena=arena,
                incoming_layout=same_tp,
                incoming_image=img_pp,  # same size, wrong phase
                incoming_phase="tp",
                outgoing_layout=same_pp,
                outgoing_image=img_pp,
                outgoing_phase="pp",
            )

    def test_mutated_outgoing_arena_stops_the_leg_before_it_reads(self):
        """F2 wired into the leg: the detector runs BEFORE the arena is
        overwritten, or the evidence of the mutation is destroyed by the fix."""
        self.arena[99] = 4
        with self.assertRaises(WeightsArenaError) as caught:
            weights_arena.two_file_leg(
                arena=self.arena,
                incoming_layout=self.tp,
                incoming_image=self.image_tp,
                incoming_phase="tp",
                outgoing_layout=self.pp,
                outgoing_image=self.image_pp,
                outgoing_phase="pp",
            )
        self.assertIn("mutated while serving", str(caught.exception).lower())
        # The arena still holds the EVIDENCE, not the incoming layout.
        self.assertEqual(int(self.arena[99]), 4)


class TestDefaultArmUntouched(CustomTestCase):
    """F3: with the gate off, the rotation path is byte-identical."""

    def test_rotation_still_writes_its_trailer(self):
        """The single-image contract must survive this commit unchanged.

        Can-fail: route the default arm through the two-file leg and the
        outgoing trailer stops being written, which breaks the NEXT flip.
        """
        from sglang.srt.model_executor.rotation_executor import rotate_arena

        pp = _layout(4096)
        image = torch.zeros(6144 + 8, dtype=torch.uint8)
        arena = torch.zeros(6144, dtype=torch.uint8)
        arena[: pp.total_bytes] = 5
        with envs.SGLANG_PHASE_FLIP_IMAGE_TWO_FILE.override(False):
            rotate_arena(
                arena=arena,
                host_image=image,
                incoming_bytes=0,
                outgoing_bytes=pp.total_bytes,
                chunk_bytes=1024,
                depth=2,
                ring=None,
                verify_incoming=False,
            )
        want = uint8_checksum(torch.full((4096,), 5, dtype=torch.uint8))
        got = int(image[pp.total_bytes : pp.total_bytes + 8].view(torch.int64).item())
        self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
