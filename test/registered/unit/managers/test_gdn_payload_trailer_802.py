"""#802: an unwritten trailer is not a data corruption.

THE METAL FACT. 2026-08-22 17:22:44, under load, during a pp_to_tp flip, PP0
died and took the instance with it:

    KvReshardError: PHASE-FLIP-GDN payload checksum mismatch from peer 1
    (stored -4664535355886616603, computed 15255426930); refusing to scatter

`stored` is NEGATIVE. `uint8_checksum` is an exact int64 sum of UNSIGNED bytes,
so its value lies in [0, 255 * nbytes] and nowhere else -- a negative value is
not a checksum that disagrees, it is not a checksum at all. The trailer was
never written. `computed`, by contrast, is entirely ordinary: 15255426930 needs
a payload of at least 57 MiB, which a GDN leg is.

So the instance was killed for a data corruption that had not happened, and the
log line sent every reader to look for one. This is the SECOND time: the
discriminator's own docstring records #656 register C22 as the same misreport,
with a negative "sender" checksum of -4450328002521349435.

THE DISCRIMINATOR ALREADY EXISTED AND WAS NEVER CALLED.
`checksum_is_representable(value, nbytes)` sits directly beside
`uint8_checksum` in weights_arena.py and exists for exactly this question.
`GdnFlipMover._verify` compared first and never asked. Mechanism present,
actuator missing -- for the third time in this defect family.

WHY THE LENGTH CHECK CANNOT CATCH IT. The receive buffer in
`kv_reshard._dist_exchange` is `torch.empty` and is never pre-zeroed, and the
byte counts are derived independently on each rank and never handshaked. An
under-filled receive therefore keeps its allocated `numel` -- the length check
passes -- while the tail, which is exactly where the trailer is read from,
still holds the original allocator garbage.

WHAT THIS FILE PINS, in both directions:
  * an out-of-range trailer (negative, and absurdly large) is reported as a
    MISSING TRAILER, naming the transport and the framing, and explicitly NOT
    as data corruption;
  * an in-range but different trailer is STILL reported as a checksum
    mismatch, because that is the case the guard exists for and weakening it
    would be the real regression;
  * a correct trailer still verifies and returns the payload body.

CPU-only: `_verify` is pure tensor arithmetic and needs no GPU, no process
group and no scheduler.
"""

import types
import unittest

import torch

from sglang.srt.layers.dcp.reshard_plan import KvReshardError
from sglang.srt.managers.gdn_flip_mover import GdnFlipMover
from sglang.srt.managers.kv_reshard import _CHECKSUM_BYTES
from sglang.srt.model_executor.weights_arena import uint8_checksum
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15)

#: The specimen's own trailer value. Red-first means red on THIS number.
SPECIMEN_STORED = -4664535355886616603

#: The specimen's computed value, for the arithmetic this file asserts about.
SPECIMEN_COMPUTED = 15255426930


def _verify(payload, want, peer=1):
    """The SHIPPED `_verify`, bound to a bare holder.

    It touches nothing on `self`; binding it here keeps the test against the
    shipped function rather than a copy of its logic.
    """
    holder = types.SimpleNamespace()
    return types.MethodType(GdnFlipMover._verify, holder)(payload, want, peer)


def _payload(body: torch.Tensor, trailer_value: int) -> torch.Tensor:
    """A GDN payload: body bytes followed by an int64 trailer."""
    trailer = torch.tensor([int(trailer_value)], dtype=torch.int64).view(torch.uint8)
    assert trailer.numel() == _CHECKSUM_BYTES
    return torch.cat([body, trailer])


def _body(n: int = 4096) -> torch.Tensor:
    # Sampled on CPU deliberately; this suite must not need a device.
    return (torch.arange(n, dtype=torch.int64) % 251).to(torch.uint8)


class GdnPayloadTrailer(unittest.TestCase):
    def test_a_correct_trailer_still_verifies(self):
        """The control. Without it the two guards below could be blanket refusals."""
        body = _body()
        payload = _payload(body, uint8_checksum(body))
        out = _verify(payload, payload.numel())
        self.assertEqual(out.numel(), body.numel())
        self.assertTrue(torch.equal(out, body))

    def test_the_specimen_trailer_is_reported_as_missing_not_corrupt(self):
        """THE DEFECT, with the specimen's own number. RED before the fix.

        A negative trailer cannot be uint8_checksum of any payload, so the
        conclusion "the DATA differs" was never available from this evidence.
        """
        body = _body()
        payload = _payload(body, SPECIMEN_STORED)
        with self.assertRaises(KvReshardError) as caught:
            _verify(payload, payload.numel())
        msg = str(caught.exception)
        self.assertIn("NO CHECKSUM", msg, f"still reported as a mismatch: {msg}")
        self.assertIn(str(SPECIMEN_STORED), msg, "the impossible value is not named")
        self.assertIn(
            "NOT EVIDENCE OF DATA CORRUPTION",
            msg,
            "the message still lets a reader conclude the data was corrupt",
        )
        self.assertNotIn(
            "checksum mismatch",
            msg,
            "the missing-trailer case still carries the mismatch wording, which "
            "is what sent the last two investigations after a corruption that "
            "had not happened",
        )

    def test_an_absurdly_large_trailer_is_also_reported_as_missing(self):
        """The other side of the range, and #656's own example was this one.

        255 * nbytes is a hard ceiling; a value above it needs a payload larger
        than the one in hand.
        """
        body = _body()
        payload = _payload(body, 255 * body.numel() + 1)
        with self.assertRaises(KvReshardError) as caught:
            _verify(payload, payload.numel())
        self.assertIn("NO CHECKSUM", str(caught.exception))

    def test_can_fail_a_real_mismatch_is_still_a_real_mismatch(self):
        """THE GUARD MUST STILL FIRE.

        An in-range trailer that disagrees IS the case this guard exists for.
        If the representability test ever swallowed it, #802 would have traded
        a misdiagnosis for a silent scatter of wrong GDN state -- which is
        strictly worse than the crash it replaces.
        """
        body = _body()
        good = uint8_checksum(body)
        payload = _payload(body, good - 1)  # in range, wrong by one
        with self.assertRaises(KvReshardError) as caught:
            _verify(payload, payload.numel())
        msg = str(caught.exception)
        self.assertIn("checksum mismatch", msg, f"the real guard stopped firing: {msg}")
        self.assertNotIn("NO CHECKSUM", msg)

    def test_can_fail_the_boundary_values_are_in_range(self):
        """0 and 255*nbytes are REPRESENTABLE and must reach the comparison.

        An off-by-one in the range test would push a legitimate all-zero or
        all-0xff payload into the missing-trailer branch and hide a genuine
        mismatch behind a transport story.
        """
        for body, edge in (
            (torch.zeros(64, dtype=torch.uint8), 0),
            (torch.full((64,), 255, dtype=torch.uint8), 255 * 64),
        ):
            payload = _payload(body, edge)
            out = _verify(payload, payload.numel())
            self.assertTrue(torch.equal(out, body))

    def test_the_wrong_length_still_wins_first(self):
        """Length is checked before the trailer, and must stay that way.

        A short buffer has no meaningful trailer at all, so reporting its
        length is strictly more informative than reporting its tail.
        """
        body = _body()
        payload = _payload(body, uint8_checksum(body))
        with self.assertRaises(KvReshardError) as caught:
            _verify(payload, payload.numel() + 8)
        self.assertIn("expected", str(caught.exception))

    def test_the_specimen_numbers_say_what_this_file_claims(self):
        """The arithmetic, pinned, so the docstring cannot drift from it.

        `computed` is an ordinary checksum for a GDN-sized payload; `stored` is
        impossible for any payload at all. That asymmetry IS the diagnosis.
        """
        from sglang.srt.model_executor.weights_arena import checksum_is_representable

        self.assertLess(SPECIMEN_STORED, 0)
        self.assertFalse(checksum_is_representable(SPECIMEN_STORED, 1 << 40))
        # The smallest payload that could produce the computed value.
        self.assertGreater(SPECIMEN_COMPUTED / 255 / 1024 / 1024, 50.0)
        self.assertTrue(checksum_is_representable(SPECIMEN_COMPUTED, 1 << 30))


if __name__ == "__main__":
    unittest.main()
