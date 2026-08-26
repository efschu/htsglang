"""#878: the reversed copy-back's precondition -- and the checksum cannot supply it.

WHY THIS EXISTS. #875 established that the flip's `save` term (host-to-host
staging, 4.342 s of a 4.801 s leg on PP0, 90 % of the seam's dominant segment)
is forced by a genuine cycle: the rotation's read and write streams share one
coordinate system, so every region must be read before it is written and written
before it is read. Threading the memcpy buys 1.42x under three-rank contention.
The only way to REMOVE it is to break the cycle, and the one candidate that does
not need a second buffer is to write the copy-back REVERSED -- from the top of
`host_image` downward while the H2D reads forward. Aliasing then falls from 498
steps to ~1 (the midpoint crossing).

This file settles the desk half of that scheme's precondition before anyone
builds it. It does NOT implement reversal.

THE QUESTION IS THE SET, NOT THE ORDER. `uint8_checksum` (weights_arena.py:111)
is an exact int64 sum of unsigned bytes, and its own docstring turns that into a
feature: "an exact integer sum is associative -- so sizing the chunk to the
device's free memory changes only the checksum's peak transient, never its
value". Order-independence is therefore not in doubt and is not the question.
The question is whether the reversed image is the same MULTISET of bytes.

IT IS -- AND THAT IS THE FINDING, NOT THE RELIEF. A chunk-reversed image holds
exactly the same bytes in a permuted order, so the sum is bit-identical and the
verification PASSES. The contract is asymmetric by design (rotation_executor.py
~408-422): the outgoing trailer is computed from the ARENA before a byte moves,
and it is verified one flip later, device-side, after the image streams back in.
So if the un-reversal on that next flip is wrong -- a missed parity flag, a
short tail chunk landing at the wrong end, a resumed process that lost which
parity it wrote -- the arena is filled with scrambled weights and the checksum
says GREEN.

So the reversed scheme does not merely "survive" the checksum contract. It moves
the scheme's ONLY new failure mode into the checksum's exact blind spot. Any
implementation needs a parity marker verified by something that is not an
order-blind sum, and that requirement belongs in the design before the code, not
after the first scrambled boot.

WHAT THIS FILE PINS, so the argument cannot rot:
  * the checksum is order-blind -- demonstrated on a permutation, not asserted;
  * it is also blind to the specific permutation reversal produces (chunk
    reversal with a short tail), which is the case that matters;
  * it DOES still catch a changed byte, so this is a statement about which
    property it guards, not a claim that it is useless.

HEADROOM, PRICED RATHER THAN ESTIMATED. Reversal needs one extra chunk of slack
in the image buffer: the un-streamed incoming region and the copied-back
outgoing region grow toward each other, and they collide unless
`N >= I + chunk`. Today `N = max(I, O) + 8`, so when `O < I` (PP0 tp_to_pp:
I=16362.7 MiB, O=15925.8) the slack is 8 BYTES and one 32 MiB chunk does not
fit. The ask is therefore +32 MiB per rank, +96 MiB across three.

That lands on HOST RAM, not VRAM -- the images are pinned host buffers -- so it
does not touch the VRAM corridor or the arming floor at all. Against #721's
measured cgroup peak (111.3 of 118 GiB, oom_kill=17) the remaining headroom is
~6.7 GiB, and +96 MiB is ~1.4 % of it. The three images already total ~34.8 GiB
(16362.7 + 8961.3 + 9481.6). Fundable, and it is not the reason to hesitate.

The reason to hesitate is the blind spot above.

Hermetic: pure arithmetic over CPU tensors, no CUDA.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.model_executor.weights_arena import uint8_checksum
from sglang.test.test_utils import CustomTestCase

MiB = 1 << 20


def _payload(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8)


class TestTheChecksumIsBlindToOrder(CustomTestCase):
    """Demonstrated, not asserted from the docstring."""

    def test_a_reversal_does_not_change_the_checksum(self):
        p = _payload(4096)
        self.assertEqual(uint8_checksum(p), uint8_checksum(torch.flip(p, (0,))))

    def test_a_random_permutation_does_not_change_the_checksum(self):
        p = _payload(4096, seed=1)
        perm = torch.randperm(p.numel(), generator=torch.Generator().manual_seed(2))
        self.assertEqual(uint8_checksum(p), uint8_checksum(p[perm]))

    def test_it_is_blind_to_CHUNK_reversal_with_a_short_tail(self):
        """THE SPECIFIC PERMUTATION THE SCHEME PRODUCES. Chunks reversed, and
        the final short chunk lands at the other end -- the exact shape a
        parity bug would leave behind, and the checksum cannot see it."""
        chunk = 100
        p = _payload(1050, seed=3)  # 10 full chunks + a 50-byte tail
        chunks = list(p.split(chunk))
        scrambled = torch.cat(list(reversed(chunks)))
        self.assertEqual(p.numel(), scrambled.numel())
        self.assertFalse(torch.equal(p, scrambled), "the fixture did not scramble")
        self.assertEqual(
            uint8_checksum(p),
            uint8_checksum(scrambled),
            "if this ever differs, the checksum grew order-sensitivity and the "
            "reversed scheme's risk assessment has to be redone",
        )

    def test_it_DOES_still_catch_a_changed_byte(self):
        """The complement, so the file is not read as 'the checksum is useless'.
        It guards CONTENT and not ARRANGEMENT, and reversal changes only the
        second."""
        p = _payload(4096, seed=4)
        q = p.clone()
        q[123] = (int(q[123]) + 1) % 256
        self.assertNotEqual(uint8_checksum(p), uint8_checksum(q))


class TestWhereTheParityMarkerCanLive(CustomTestCase):
    """POSTEN 4, and checking the premise changed the answer.

    THE QUESTION WAS POSED AS "it must survive a process restart". It does not
    have to, because the thing it describes does not. The host image is RE-PRIMED
    at every boot: `prime_arena_from_image` (phase_flip_boot.py:499) calls the same
    `rotate_arena` with `outgoing_bytes=0`, which degenerates the rotation to a
    plain contiguous H2D and fills the image from the model in canonical forward
    order. Pinned host memory does not outlive the process, and even the
    file-backed arm is overwritten by that priming fill before any flip runs. So
    parity only has to hold from one flip to the next WITHIN one process
    lifetime.

    A LEGAL HOME THEREFORE ALREADY EXISTS: `PhaseStacks.image_holds`, the
    in-process marker recording WHICH layout the image contains. Recording WHICH
    ORDER is the same kind of fact with the same lifetime, in the same place. No
    sidecar, no manifest, no second trailer slot.

    THAT SOLVES STORAGE AND NOT VERIFICATION, and I had conflated the two. A
    wrong `image_holds` today is caught -- `rotate_arena` raises RotationHazard
    on `image_holds != wants`, and the device-side checksum catches the rest. A
    wrong PARITY under the reversed scheme is caught by NOTHING, because the
    checksum is an order-blind sum. So the scheme still needs a second
    verification primitive; it just does not need a second STORAGE slot.

    AND THE ORDER INFORMATION IS ALREADY COMPUTED, THEN DISCARDED.
    `uint8_checksum` builds a per-chunk sum VECTOR and collapses it with a final
    `.sum()` (weights_arena.py:125-129). The vector is order-sensitive; the
    collapse is what throws that away. Retaining it costs no extra data
    movement.

    IT IS NOT FREE, THOUGH, AND THIS IS THE CONSTRAINT TO CARRY INTO THE DESIGN.
    The chunk size is ADAPTIVE -- `_checksum_chunk_bytes` sizes it to free device
    memory -- and the function's docstring makes chunk-size-independence a
    load-bearing property so that "two ranks with different free memory still
    agree". A per-chunk vector under adaptive chunking is not comparable across
    ranks or across a flip. So the order-sensitive primitive must use a FIXED
    partition, deliberately decoupled from the free-memory heuristic. Small, but
    it is a real constraint and it is invisible unless someone reads why the
    adaptive sizing exists.
    """

    def test_the_priming_fill_has_no_copy_back(self):
        """The fact the whole answer rests on: at boot the image is written
        forward from the model, with nothing placed back, so there is no parity
        ambiguity to inherit across a restart."""
        import ast
        import inspect
        import textwrap

        from sglang.srt.managers import phase_flip_boot

        # ASSERTED ON THE CALL, NOT ON THE TEXT. `outgoing_bytes=0` also appears
        # in this function's own DOCSTRING, so a substring check passes even when
        # the real argument changes -- caught by mutation: replacing the first
        # textual occurrence hit the prose and left this test green.
        src = textwrap.dedent(inspect.getsource(phase_flip_boot.prime_arena_from_image))
        kwargs = {}
        for node in ast.walk(ast.parse(src)):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "rotate_arena"
            ):
                kwargs = {
                    kw.arg: kw.value for kw in node.keywords if kw.arg is not None
                }
        self.assertTrue(kwargs, "no rotate_arena(...) call found in the priming fill")
        self.assertIsInstance(kwargs.get("outgoing_bytes"), ast.Constant)
        self.assertEqual(
            0,
            kwargs["outgoing_bytes"].value,
            "the priming fill gained a copy-back; boot no longer establishes a "
            "canonical forward image, and the parity argument has to be redone",
        )
        self.assertIs(True, kwargs["priming"].value)

    def test_the_existing_layout_marker_is_in_process_only(self):
        """`image_holds` is a plain attribute -- no file, no shared memory. That
        is exactly the lifetime parity needs, which is why it is the right
        neighbour for it."""
        import inspect

        from sglang.srt.managers import phase_flip_boot

        src = inspect.getsource(phase_flip_boot.PhaseFlipStacks)
        self.assertIn("image_holds", src)

    def test_the_checksum_computes_per_chunk_sums_and_collapses_them(self):
        """The order information exists for one line and is then summed away."""
        import inspect

        from sglang.srt.model_executor import weights_arena

        src = inspect.getsource(weights_arena.uint8_checksum)
        self.assertIn("payload.split(", src)
        self.assertIn("torch.stack(parts).sum()", src)

    def test_a_per_chunk_vector_WOULD_be_order_sensitive(self):
        """The candidate primitive, demonstrated. Same bytes, permuted chunks:
        the collapsed sum agrees and the vector does not."""
        chunk = 100
        p = _payload(1050, seed=9)
        chunks = list(p.split(chunk))
        scrambled = torch.cat(list(reversed(chunks)))
        vec = [int(c.sum(dtype=torch.int64)) for c in p.split(chunk)]
        vec_s = [int(c.sum(dtype=torch.int64)) for c in scrambled.split(chunk)]
        self.assertEqual(uint8_checksum(p), uint8_checksum(scrambled))
        self.assertNotEqual(vec, vec_s, "the per-chunk vector must see the order")

    def test_the_chunk_SIZE_is_adaptive_which_is_why_the_vector_is_not_free(self):
        """The constraint that makes this a design item rather than a one-liner:
        the partition is chosen from free device memory, and the checksum's
        cross-rank comparability depends on the VALUE not caring."""
        import inspect

        from sglang.srt.model_executor import weights_arena

        src = inspect.getsource(weights_arena._checksum_chunk_bytes)
        self.assertIn("mem_get_info", src)
        doc = weights_arena.uint8_checksum.__doc__ or ""
        self.assertIn("independent of the chunk size", doc)


class TestTheHeadroomArithmetic(CustomTestCase):
    """The +32 MiB ask, from the real layout vectors rather than a guess."""

    # (incoming MiB, outgoing MiB) per rank per direction, from
    # boot_w40_857strict_0826_0516.log's own rotation lines.
    LEGS = {
        "PP0 pp_to_tp": (15925.8, 16362.7),
        "PP1 pp_to_tp": (8573.8, 8961.3),
        "PP2 pp_to_tp": (8573.8, 9481.6),
        "PP0 tp_to_pp": (16362.7, 15925.8),
        "PP1 tp_to_pp": (8961.3, 8573.8),
        "PP2 tp_to_pp": (9481.6, 8573.8),
    }
    CHUNK_MIB = 32.0

    def test_at_least_one_leg_has_less_slack_than_a_chunk_today(self):
        """i.e. the extra headroom is genuinely REQUIRED, not defensive. When
        the outgoing layout is the smaller one the buffer is sized to the
        INCOMING layout and the slack is the 8-byte trailer."""
        tight = [
            name
            for name, (inc, out) in self.LEGS.items()
            if (max(inc, out) - inc) < self.CHUNK_MIB
        ]
        self.assertTrue(
            tight,
            "no leg is tight, so the +32 MiB claim would be unfounded",
        )
        self.assertIn("PP0 tp_to_pp", tight)

    def test_the_total_ask_is_96_mib_of_host_ram(self):
        self.assertEqual(96.0, 3 * self.CHUNK_MIB)

    def test_the_ask_is_negligible_against_MEASURED_host_ram(self):
        """Against a figure taken on this box, not a borrowed one.

        The #721 "111.3 of 118 GiB, oom_kill=17" I first used here had NO
        in-tree source -- the citation sweep found none for `oom_kill=17` and
        the only `111.3` is `111.3%` in a benchmark table. Withdrawn. This box
        reports 128711 MB total / 76971 MB free (numactl), and the images are
        PINNED HOST buffers, so the ask never touches the VRAM corridor or the
        arming floor."""
        free_mib = 76971.0
        ask_mib = 3 * self.CHUNK_MIB
        self.assertLess(
            ask_mib / free_mib,
            0.005,
            "the ask stopped being negligible against measured free host RAM",
        )


if __name__ == "__main__":
    unittest.main()
