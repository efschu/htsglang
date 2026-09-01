"""#809/W28 slice 3: the flip's refill leg IS the rotation, and RAM holds one image.

Slice 2 built and proved the executor. This pins the two wiring facts that make
it the real path rather than a tested module nothing calls:

  * **BOOT ALLOCATES ONE HOST IMAGE, NOT TWO.** The two lifetime images are the
    dual pin, and W26 OOM-killed BOTH its arms in the LAUNCH phase, before any
    flip ran. `PhaseFlipStacks` therefore carries a SINGLE max-sized buffer
    plus a marker of which layout is currently resting in it.
  * **`refill()` ROTATES.** The incoming layout streams RAM -> VRAM while the
    outgoing one is placed back into the pages it frees.

ONE PATH, NOT TWO. The priming fill is not a separate code path: it is the same
rotation with `outgoing_bytes=0` -- nothing in the arena worth keeping yet, so
the copy-back has zero length and the call degenerates to the plain H2D it
always was, with its own `priming` record so it can never be averaged into a
warm number (P4). A buffer holding the WRONG layout is not a fallback either;
it is an invariant violation and it refuses, because under a single-image
budget there is no other source to fall back TO.

THE ARM THIS REMOVES, said plainly because it is a real loss and not an
oversight: `arena_refill`'s `restore=(other_layout, other_image)` rewrote the
OTHER layout on a checksum mismatch so an abort left both layouts byte-exact.
That arm needs a second image to read from. With one buffer it cannot exist,
and the rotation refuses loudly with the arena declared undefined instead. That
is the price of the single-layout RAM budget the design is built on.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import dataclasses
import unittest

import torch

from sglang.srt.model_executor.rotation_executor import (
    RotationHazard,
    allocate_rotation_image,
)
from sglang.srt.model_executor.weights_arena import (
    _CHECKSUM_BYTES,
    image_from_tensors,
    plan_arena_layout,
    uint8_checksum,
)
from sglang.test.test_utils import CustomTestCase

PP_TO_TP = "pp_to_tp"
TP_TO_PP = "tp_to_pp"


def _named(sizes, seed):
    g = torch.Generator().manual_seed(seed)
    return {
        f"w{i}": torch.randint(0, 100, (n,), generator=g, dtype=torch.uint8).view(
            torch.uint8
        )
        for i, n in enumerate(sizes)
    }


def _layouts():
    """Two layouts of DIFFERENT size, because equal ones hide every asymmetry
    bug this scheme has."""
    pp_named = _named([2048, 1024, 512], seed=1)
    tp_named = _named([2048, 2048, 1024], seed=2)
    return pp_named, plan_arena_layout(pp_named), tp_named, plan_arena_layout(tp_named)


class TestBootHoldsOneImageNotTwo(CustomTestCase):
    def test_PhaseFlipStacks_has_no_second_lifetime_image(self):
        # #1078 UPDATED THE PREMISE, NOT THE GUARD. This assertion used to read
        # "`image_pp`/`image_tp` must not be FIELDS", because while both were
        # fields RAM held two layouts for the process life -- the dual pin by
        # another name. That inference is sound for PINNED images and only for
        # those: two pinned lifetime images are 55.99 GiB across this rig's
        # three ranks. A FILE-BACKED image is reclaimable page cache and not a
        # pinned post at all, so the same two images cost disk and no locked
        # RAM (#1078, weights_arena.require_two_file_preconditions).
        #
        # So the guard now pins what it always MEANT: the default stack holds
        # ONE image. Field absence was a proxy for that, and the proxy stopped
        # tracking the property. Asserting the fields are gone would now forbid
        # a scheme that satisfies the memory rule the guard exists to enforce.
        import sglang.srt.model_executor.weights_arena as _wa
        from sglang.srt.managers.phase_flip_boot import PhaseFlipStacks

        names = {f.name for f in dataclasses.fields(PhaseFlipStacks)}
        self.assertIn("rotation_image", names)
        self.assertIn("image_holds", names)
        # The two-file fields exist but are OPT-IN and default to absent, so a
        # stack that nobody armed still holds exactly one image.
        defaults = {
            f.name: f.default
            for f in dataclasses.fields(PhaseFlipStacks)
            if f.name in ("image_pp", "image_tp")
        }
        self.assertEqual(defaults, {"image_pp": None, "image_tp": None})
        # And the second image is unreachable under the allocator that would
        # make it a pin. This is the can-fail half: drop the refusal in
        # `require_two_file_preconditions` and the dual pin becomes reachable
        # again, which is the whole content of the original assertion.
        from sglang.srt.environ import envs

        with envs.SGLANG_PHASE_FLIP_IMAGE_TWO_FILE.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(False):
                with self.assertRaises(_wa.WeightsArenaError):
                    _wa.require_two_file_preconditions()

    def test_the_buffer_is_sized_from_the_LARGER_layout(self):
        _pp_named, pp, _tp_named, tp = _layouts()
        buf = allocate_rotation_image(pp.total_bytes, tp.total_bytes, pin=False)
        self.assertEqual(
            int(buf.numel()),
            max(pp.total_bytes, tp.total_bytes) + _CHECKSUM_BYTES,
        )
        # ...and that is strictly less than the two images it replaces.
        two_images = pp.total_bytes + tp.total_bytes + 2 * _CHECKSUM_BYTES
        self.assertLess(int(buf.numel()), two_images)

    def test_sizing_it_from_the_resting_layout_would_be_short(self):
        # THE CAN-FAIL PARTNER, and it is the OOM this sizing rule exists to
        # prevent: a buffer sized for the smaller layout cannot receive the
        # larger one on the copy-back.
        _pp_named, pp, _tp_named, tp = _layouts()
        self.assertNotEqual(pp.total_bytes, tp.total_bytes)
        smaller = min(pp.total_bytes, tp.total_bytes)
        buf = allocate_rotation_image(smaller, smaller, pin=False)
        self.assertLess(int(buf.numel()), max(pp.total_bytes, tp.total_bytes))


class TestImageFromTensorsCanFillTheSharedBuffer(CustomTestCase):
    """The boot needs to snapshot INTO the one buffer, not beside it."""

    def test_out_writes_into_the_caller_buffer_and_returns_it(self):
        pp_named, pp, _tp_named, tp = _layouts()
        buf = allocate_rotation_image(pp.total_bytes, tp.total_bytes, pin=False)
        got = image_from_tensors(pp_named, pp, pin=False, out=buf)
        self.assertIs(got, buf)
        # The trailer sits after THIS layout's payload, not at the buffer end.
        stored = int(
            buf[pp.total_bytes : pp.total_bytes + _CHECKSUM_BYTES]
            .clone()
            .view(torch.int64)
            .item()
        )
        self.assertEqual(stored, uint8_checksum(buf[: pp.total_bytes]))

    def test_it_matches_the_image_the_allocating_form_builds(self):
        # Byte-for-byte, or the `out` form is a second format wearing the same
        # name -- exactly the kind of divergence a checksum would only catch
        # one flip later.
        pp_named, pp, _t, tp = _layouts()
        alone = image_from_tensors(pp_named, pp, pin=False)
        buf = allocate_rotation_image(pp.total_bytes, tp.total_bytes, pin=False)
        shared = image_from_tensors(pp_named, pp, pin=False, out=buf)
        n = pp.total_bytes + _CHECKSUM_BYTES
        self.assertTrue(torch.equal(alone[:n], shared[:n]))

    def test_a_buffer_too_small_is_REFUSED(self):
        pp_named, pp, _t, _tp = _layouts()
        with self.assertRaises(ValueError):
            image_from_tensors(
                pp_named, pp, pin=False, out=torch.zeros(16, dtype=torch.uint8)
            )


def _stacks(direction_holds="tp"):
    """A PhaseFlipStacks carrying only what refill() touches, on CPU."""
    from sglang.srt.managers.phase_flip_boot import PhaseFlipStacks

    pp_named, pp, tp_named, tp = _layouts()
    arena = torch.zeros(max(pp.total_bytes, tp.total_bytes), dtype=torch.uint8)
    buf = allocate_rotation_image(pp.total_bytes, tp.total_bytes, pin=False)
    s = object.__new__(PhaseFlipStacks)
    s.tp_worker = None
    s.arena = arena
    s.layout_pp, s.layout_tp = pp, tp
    s.rotation_image = buf
    s.image_holds = direction_holds
    s.arena_carrier = None
    s.draft_worker = None
    s.vector = s.token_vector = (1,)
    return s, pp_named, pp, tp_named, tp


class TestRefillRotates(CustomTestCase):
    """The wiring fact: the flip's weights leg goes through the executor."""

    def _prime(self, s, named, layout, tag):
        image_from_tensors(named, layout, pin=False, out=s.rotation_image)
        s.image_holds = tag

    def test_a_warm_flip_leaves_the_arena_and_the_buffer_swapped(self):
        s, pp_named, pp, tp_named, tp = _stacks()
        # RAM rests on TP, arena holds PP: the state boot leaves behind.
        self._prime(s, tp_named, tp, "tp")
        image_from_tensors(pp_named, pp, pin=False)  # reference bytes
        pp_img = image_from_tensors(pp_named, pp, pin=False)
        s.arena[: pp.total_bytes].copy_(pp_img[: pp.total_bytes])
        tp_img = image_from_tensors(tp_named, tp, pin=False)

        s.refill(PP_TO_TP)

        self.assertTrue(
            torch.equal(s.arena[: tp.total_bytes], tp_img[: tp.total_bytes])
        )
        self.assertEqual(s.image_holds, "pp")
        self.assertTrue(
            torch.equal(s.rotation_image[: pp.total_bytes], pp_img[: pp.total_bytes])
        )

    def test_the_reverse_direction_works_the_same_way(self):
        s, pp_named, pp, tp_named, tp = _stacks()
        self._prime(s, pp_named, pp, "pp")
        tp_img = image_from_tensors(tp_named, tp, pin=False)
        s.arena[: tp.total_bytes].copy_(tp_img[: tp.total_bytes])
        pp_img = image_from_tensors(pp_named, pp, pin=False)

        s.refill(TP_TO_PP)

        self.assertTrue(
            torch.equal(s.arena[: pp.total_bytes], pp_img[: pp.total_bytes])
        )
        self.assertEqual(s.image_holds, "tp")
        self.assertTrue(
            torch.equal(s.rotation_image[: tp.total_bytes], tp_img[: tp.total_bytes])
        )

    def test_a_full_cycle_returns_both_sides_to_their_start(self):
        s, pp_named, pp, tp_named, tp = _stacks()
        self._prime(s, tp_named, tp, "tp")
        pp_img = image_from_tensors(pp_named, pp, pin=False)
        tp_img = image_from_tensors(tp_named, tp, pin=False)
        s.arena[: pp.total_bytes].copy_(pp_img[: pp.total_bytes])
        for _ in range(3):
            s.refill(PP_TO_TP)
            s.refill(TP_TO_PP)
        self.assertEqual(s.image_holds, "tp")
        self.assertTrue(
            torch.equal(s.arena[: pp.total_bytes], pp_img[: pp.total_bytes])
        )
        self.assertTrue(
            torch.equal(s.rotation_image[: tp.total_bytes], tp_img[: tp.total_bytes])
        )

    def test_a_buffer_holding_the_WRONG_layout_REFUSES(self):
        # NOT A FALLBACK. Under a single-image budget there is no other source
        # to read the incoming layout from, so proceeding would stream whatever
        # happens to be in the buffer into the arena and serve it.
        s, _pp_named, _pp, tp_named, tp = _stacks()
        self._prime(s, tp_named, tp, "tp")
        s.image_holds = "pp"  # the marker now lies
        with self.assertRaises(RotationHazard):
            s.refill(TP_TO_PP)

    def test_an_unknown_direction_is_still_refused(self):
        s, _a, _b, _c, _d = _stacks()
        with self.assertRaises(Exception):
            s.refill("sideways")


class TestThePrimingFillIsTheSameCallWithNothingToKeep(CustomTestCase):
    """One path. The priming fill is `outgoing_bytes=0`, not a second branch."""

    def test_priming_fills_the_arena_and_keeps_its_own_record(self):
        from sglang.srt.managers.phase_flip_boot import prime_arena_from_image

        s, pp_named, pp, _tp_named, tp = _stacks()
        pp_img = image_from_tensors(pp_named, pp, pin=False, out=s.rotation_image)
        s.image_holds = "pp"
        reference = image_from_tensors(pp_named, pp, pin=False)

        stats = prime_arena_from_image(s.arena, pp, pp_img)

        self.assertTrue(stats.priming)
        self.assertEqual(stats.d2h_bytes, 0, "priming has nothing to copy back")
        self.assertTrue(
            torch.equal(s.arena[: pp.total_bytes], reference[: pp.total_bytes])
        )

    def test_priming_and_a_warm_rotation_do_not_share_a_record(self):
        from sglang.srt.managers.phase_flip_boot import prime_arena_from_image

        s, pp_named, pp, tp_named, tp = _stacks()
        image_from_tensors(pp_named, pp, pin=False, out=s.rotation_image)
        s.image_holds = "pp"
        primed = prime_arena_from_image(s.arena, pp, s.rotation_image)
        self.assertTrue(primed.priming)
        self.assertGreater(primed.h2d_bytes, 0)


if __name__ == "__main__":
    unittest.main()
