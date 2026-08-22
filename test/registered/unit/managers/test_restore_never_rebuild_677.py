"""#677: RESTORE, NEVER REBUILD — no build may happen inside the cutover.

THE INVARIANT. A flip may pay COPY time. It may never pay BUILD time. No
CUDA-graph re-capture, no JIT, no arena construction on the cutover path.
Everything a flip needs must be created once at boot and thereafter only
spilled and restored.

WHY IT IS A FLOOR QUESTION, not a latency one. If a component can only be
recovered by rebuilding it, it must stay RESIDENT across the flip and its
bytes are an irreducible term in the arming floor. If it can be restored by
copying, it may live in host RAM between flips and its bytes leave the floor
entirely, at a price in PCIe time that is measurable and payable
(NOTE_677_floor_components.md section 4). So this invariant is what moves
capture-moment workspace from "per-flip floor" to "boot-time only", and the
floor cannot be shrunk toward the irreducible transient without it.

WHY IT NEEDS A PIN RATHER THAN A CONVENTION. The cutover is the no-return
region: after the pre-cutover movers the source pool's pages are gone. A
build there does not merely cost time -- it allocates, inside a window that
was entered precisely because memory was tight, and its failure mode is the
one with no way back. Nothing in the type system distinguishes
``arena_refill`` (a copy, correct here) from ``allocate_arena`` (a build,
never correct here); they sit in the same module and differ by one word.

WHAT THIS PINS. The real production mover -- ``PhaseFlipStacks.refill``, the
``weights_refill`` leg named in ``phase_flip_runtime.py:1843-1846`` -- run
against a real arena with the build entry points fenced. Not a
re-implementation of the loop: a re-implementation would keep passing while
production drifted, which is the #624 failure this corpus already knows.
"""

import unittest

import torch

from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP
from sglang.srt.model_executor import weights_arena as wa
from sglang.srt.managers.phase_flip_boot import PhaseFlipStacks

PAYLOAD = 4096


class BuildInsideCutover(AssertionError):
    """Raised by the fence. A distinct type so a test cannot pass by
    catching an unrelated failure and calling it a detection."""


def _image(fill: int, nbytes: int = PAYLOAD):
    """A valid arena image: payload plus its 8-byte checksum trailer."""
    payload = torch.full((nbytes,), fill, dtype=torch.uint8)
    checksum = torch.tensor([wa.uint8_checksum(payload)], dtype=torch.int64).view(
        torch.uint8
    )
    return torch.cat([payload, checksum])


def _layout(nbytes: int = PAYLOAD):
    return wa.ArenaLayout(slots=(), aliases=(), total_bytes=nbytes)


def _stacks():
    """A PhaseFlipStacks bound to a real arena, real layouts, real images.

    ``refill`` is the genuine production method; only the surrounding object
    is a shell, and it carries exactly the attributes that method reads.
    """
    stacks = PhaseFlipStacks.__new__(PhaseFlipStacks)
    stacks.arena = torch.zeros(PAYLOAD, dtype=torch.uint8)
    stacks.layout_pp = _layout()
    stacks.layout_tp = _layout()
    stacks.image_pp = _image(0xAB)
    stacks.image_tp = _image(0xCD)
    stacks.arena_carrier = None
    return stacks


class _Fence:
    """Patches every BUILD entry point to raise, leaving copies alone.

    The list is the point of this class, so it is written out rather than
    discovered: arena CONSTRUCTION and PACKING are builds; ``arena_refill``
    is a copy and is deliberately absent. CUDA graph capture is fenced at
    ``torch.cuda.CUDAGraph`` because that is the constructor every capture
    path bottoms out in, whichever backend wraps it.
    """

    TARGETS = (
        (wa, "allocate_arena"),
        (wa, "pack_into_arena"),
        (torch.cuda, "CUDAGraph"),
    )

    def __enter__(self):
        self._saved = []
        for mod, name in self.TARGETS:
            original = getattr(mod, name, None)
            if original is None:
                continue
            self._saved.append((mod, name, original))

            def _refuse(*_a, _name=name, **_kw):
                raise BuildInsideCutover(
                    f"{_name} was called on the cutover path: a flip may pay "
                    f"COPY time, never BUILD time"
                )

            setattr(mod, name, _refuse)
        return self

    def __exit__(self, *exc):
        for mod, name, original in self._saved:
            setattr(mod, name, original)
        return False


class TestTheFenceCanFail(unittest.TestCase):
    """The detector must be shown to work before it is used as evidence.

    A fence that silently patched nothing would make every pin below pass
    vacuously -- the #380/#585 test-honesty class.
    """

    def test_each_build_entry_point_is_actually_fenced(self):
        with _Fence():
            for mod, name in _Fence.TARGETS:
                with self.subTest(entry=name):
                    with self.assertRaises(BuildInsideCutover):
                        getattr(mod, name)()

    def test_the_fence_restores_the_originals(self):
        before = [getattr(mod, name, None) for mod, name in _Fence.TARGETS]
        with _Fence():
            pass
        after = [getattr(mod, name, None) for mod, name in _Fence.TARGETS]
        self.assertEqual(before, after)

    def test_a_planted_build_is_detected(self):
        """CAN-FAIL ARM, and the one the directive asks for. A mover that
        constructs an arena instead of refilling it must fail the pin."""

        def _mover_that_rebuilds(direction):
            wa.allocate_arena(PAYLOAD, "cpu")

        with _Fence():
            with self.assertRaises(BuildInsideCutover):
                _mover_that_rebuilds(PP_TO_TP)


class TestTheRealMoverOnlyRestores(unittest.TestCase):
    """The production ``weights_refill`` leg, both directions."""

    def test_pp_to_tp_refill_builds_nothing(self):
        stacks = _stacks()
        with _Fence():
            stacks.refill(PP_TO_TP)
        self.assertEqual(
            int(stacks.arena[0]), 0xCD, "the TP image must have been copied in"
        )

    def test_tp_to_pp_refill_builds_nothing(self):
        stacks = _stacks()
        with _Fence():
            stacks.refill(TP_TO_PP)
        self.assertEqual(int(stacks.arena[0]), 0xAB)

    def test_the_restore_arm_also_builds_nothing(self):
        """The checksum-mismatch path rewrites the current layout -- the one
        branch that touches the arena twice, and the one most likely to reach
        for a rebuild. It must still only copy."""
        stacks = _stacks()
        corrupt = _image(0xCD)
        corrupt[0] = 0x01  # payload no longer matches its trailer
        stacks.image_tp = corrupt

        with _Fence():
            with self.assertRaises(wa.WeightsArenaError):
                stacks.refill(PP_TO_TP)

        self.assertEqual(
            int(stacks.arena[0]),
            0xAB,
            "the restore arm must have rewritten the PP layout by copy",
        )


class TestTheAllowedCopyIsNotFenced(unittest.TestCase):
    """Guards the fence against over-reach.

    If ``arena_refill`` were ever added to TARGETS the pins above would
    still pass -- by refusing the very operation the flip is supposed to
    perform. This states the boundary the fence draws.
    """

    def test_arena_refill_remains_callable_under_the_fence(self):
        arena = torch.zeros(PAYLOAD, dtype=torch.uint8)
        with _Fence():
            wa.arena_refill(arena, _layout(), _image(0x7F))
        self.assertEqual(int(arena[0]), 0x7F)


if __name__ == "__main__":
    unittest.main()
