# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#852 class fix: EVERY provider answers "promised vs delivered".

#852 fixed ONE funder. `allocator-cache` promised `memory_reserved -
memory_allocated`, an `empty_cache()` draw delivered 0, and 43 of 45 binding
refusals on W24 read `cause=phantom_capacity`. The instance fix priced that
post at what a draw CAN return, and #852 R3 named the fourth term (CUDA-graph
private pools). Both are on the train.

THE CLASS WAS LEFT STANDING, and it is one level up from the funder.

``CorridorGuard.register`` states the law in its own docstring:

    ``free_up_to(nbytes)`` must free AT MOST ``nbytes``, synchronously, and
    return the bytes it actually gave back to the DRIVER -- not to torch's
    cache.

Nothing enforced it, nothing measured it, and nothing recorded it. The ladder
summed what providers CLAIMED:

    got = int(provider.free_up_to(deficit))
    ...
    reclaimed += got

while re-probing the driver only to decide when to stop. So the driver delta
per provider -- the number the law is written in, already sampled on every
iteration -- was computed and thrown away.

THAT IS NOT COSMETIC, because one branch decides on it. ``must_reclaim``
exists to answer "did this ladder RELEASE more", and its own refusal text says
it "judges the DELTA and nothing else". It judged the sum of claims. A
provider claiming ``want`` while the driver's free column never moved turned
that refusal OFF -- the exact false success #689 built the branch to end, one
layer up, and reachable by any provider rather than by one known funder.

The family is already three deep and named in the tree:

* ``lend_to_level`` fixed it for the LENDER only, with the comment "this chain
  has three times credited bytes that went to an allocator free-list instead
  of to the driver", and reasoned the gate "can afford to credit claims
  because its verdict is re-probed anyway". The ``must_reclaim`` branch is
  where that reasoning does not hold.
* ``docs/dev/631/HANDOFF_677.md`` §3a audited the providers and found
  ``draft-weights`` returning ``carrier.spill() * _MIB`` -- an arena decommit
  count, not a driver delta -- filed as "a latent member of the 'freed nothing
  the driver could see' family this chain has shipped three times". It was
  recorded as "not yet a bug" and never fixed.

So: measure the delivery of every provider, judge on it, and keep the record
so a phantom funder is visible on its FIRST pass instead of after a GPU window
and a ticket. The ledger is the structural half -- a hand-tuned factor for one
funder would leave the next one exactly as invisible as this one was.

Hermetic: an injected free-memory probe, no CUDA.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers import corridor_guard as cg

MIB = 1024 * 1024


class _Card:
    """A device whose free column moves only when a provider really pays."""

    def __init__(self, free_mib: int):
        self.free = free_mib * MIB

    def probe(self) -> int:
        return self.free

    def provider(self, pool_mib: int, *, to_driver: bool = True):
        """A payload of ``pool_mib``. ``to_driver=False`` is the phantom: it
        REPORTS the bytes and the driver never sees them."""
        state = {"left": pool_mib * MIB}

        def free_up_to(nbytes: int) -> int:
            take = min(state["left"], max(0, int(nbytes)))
            state["left"] -= take
            if to_driver:
                self.free += take
            return take

        return free_up_to


def _guard(card, floor=0, delta=0):
    return cg.CorridorGuard(
        0, floor_mib=floor, delta_mib=delta, probe=card.probe, law_floor_mib=0
    )


class APhantomProviderMayNotSatisfyMustReclaim(unittest.TestCase):
    """The live decision. Everything else in this file is the record."""

    def test_a_claim_the_driver_never_saw_does_not_fund_an_incremental_ask(self):
        card = _Card(4096)
        g = _guard(card)
        g.register("phantom", 10, card.provider(8000, to_driver=False))
        res = g.ensure_headroom(178 * MIB, reason="seam staging", must_reclaim=True)
        self.assertFalse(
            res.ok,
            "the free column never moved, so the ladder released nothing the "
            "caller can spend; crediting the claim is the #689 false success "
            "reached through a provider instead of through free memory.",
        )

    def test_the_same_ask_is_funded_when_the_driver_actually_sees_it(self):
        """The GUARD DIRECTION. A real payer must still pay -- an
        under-reporting ledger would suppress relief and make the flip
        stickier, which is the defect #852 exists to remove, inverted."""
        card = _Card(4096)
        g = _guard(card)
        g.register("real", 10, card.provider(8000, to_driver=True))
        res = g.ensure_headroom(178 * MIB, reason="seam staging", must_reclaim=True)
        self.assertTrue(res.ok)
        self.assertGreaterEqual(res.reclaimed, 178 * MIB)

    def test_a_partial_payer_funds_exactly_what_it_delivered(self):
        card = _Card(4096)
        g = _guard(card)
        g.register("phantom", 10, card.provider(500, to_driver=False))
        g.register("real", 20, card.provider(300, to_driver=True))
        res = g.ensure_headroom(178 * MIB, reason="seam staging", must_reclaim=True)
        self.assertTrue(res.ok)
        # The deficit is the 178 MiB ask. The phantom claims all of it and
        # moves nothing, so the real payer is still asked for the full
        # deficit and delivers it -- claims 356 MiB, delivered 178.
        self.assertEqual(178 * MIB, res.reclaimed)


class ReclaimedIsDeliveredNotClaimed(unittest.TestCase):
    def test_the_headline_figure_is_the_driver_delta(self):
        card = _Card(4096)
        g = _guard(card)
        g.register("phantom", 10, card.provider(8000, to_driver=False))
        res = g.ensure_headroom(4500 * MIB)
        self.assertEqual(
            0,
            res.reclaimed,
            "reclaimed is the corridor law's own unit -- bytes the DRIVER "
            "gave back. A figure that counts an allocator free-list is the "
            "phantom promise with a different name.",
        )

    def test_the_running_total_is_delivered_too(self):
        card = _Card(4096)
        g = _guard(card)
        g.register("phantom", 10, card.provider(8000, to_driver=False))
        g.ensure_headroom(4500 * MIB)
        self.assertEqual(0, g.reclaimed_total)

    def test_a_real_payer_still_counts_byte_for_byte(self):
        card = _Card(4096)
        g = _guard(card)
        # A pool SMALLER than the 404 MiB deficit, so the figure under test is
        # the payload and not the deficit arithmetic.
        g.register("real", 10, card.provider(300, to_driver=True))
        res = g.ensure_headroom(4500 * MIB)
        self.assertEqual(300 * MIB, res.reclaimed)
        self.assertEqual(300 * MIB, g.reclaimed_total)


class TheLedgerAnswersPromisedVersusDelivered(unittest.TestCase):
    """The structural half: the question is answerable FOR EVERY PROVIDER,
    including the ones nobody has suspected yet."""

    def test_a_registered_provider_starts_unobserved_not_at_zero(self):
        """``None`` means NEVER MEASURED and must never collapse into 0 --
        "no probe has recorded this" and "recorded and delivered nothing" are
        the two readings this whole ticket family exists to keep apart."""
        card = _Card(4096)
        g = _guard(card)
        g.register("untested", 10, card.provider(100))
        rec = g.delivery_report()["untested"]
        self.assertEqual(0, rec.observations)
        self.assertIsNone(rec.delivery_ratio)

    def test_the_phantom_is_recorded_with_both_halves(self):
        card = _Card(4096)
        g = _guard(card)
        g.register("phantom", 10, card.provider(8000, to_driver=False))
        g.ensure_headroom(4500 * MIB)
        rec = g.delivery_report()["phantom"]
        self.assertGreater(rec.claimed_bytes, 0)
        self.assertEqual(0, rec.delivered_bytes)
        self.assertEqual(0.0, rec.delivery_ratio)
        self.assertTrue(rec.is_phantom)

    def test_an_honest_provider_is_not_flagged(self):
        card = _Card(4096)
        g = _guard(card)
        g.register("real", 10, card.provider(600, to_driver=True))
        g.ensure_headroom(4500 * MIB)
        rec = g.delivery_report()["real"]
        self.assertEqual(1.0, rec.delivery_ratio)
        self.assertFalse(rec.is_phantom)

    def test_the_record_accumulates_across_passes(self):
        """W24's shape: the same post re-promising the same bytes every 60-75
        s for 21.6 min. One pass is an anecdote; the count is the finding."""
        card = _Card(4096)
        g = _guard(card)
        g.register("phantom", 10, card.provider(80000, to_driver=False))
        for _ in range(3):
            g.ensure_headroom(4500 * MIB)
        rec = g.delivery_report()["phantom"]
        self.assertEqual(3, rec.observations)
        self.assertEqual(3, rec.phantom_passes)

    def test_a_provider_that_starts_delivering_stops_being_phantom(self):
        """Law 2 recovers. A derate that could not be earned back would
        strand a funder that came good, which is the stickiness defect with
        the sign flipped."""
        card = _Card(4096)
        g = _guard(card)
        pays = {"driver": False}

        def free_up_to(nbytes):
            take = min(600 * MIB, max(0, int(nbytes)))
            if pays["driver"]:
                card.free += take
            return take

        g.register("moody", 10, free_up_to)
        g.ensure_headroom(4500 * MIB)
        self.assertTrue(g.delivery_report()["moody"].is_phantom)
        pays["driver"] = True
        g.ensure_headroom(4500 * MIB)
        self.assertFalse(
            g.delivery_report()["moody"].is_phantom,
            "the last observation delivered, so the standing verdict must "
            "clear; law 2 trusts a post that pays.",
        )

    def test_a_provider_that_stops_delivering_is_flagged_at_once(self):
        """THE DIRECTION A GOOD HISTORY WOULD HIDE, and the one W24 was in.

        The standing verdict is the LAST observation, not a rate over the
        record. A funder that paid for an hour and then went phantom must read
        phantom on the very next pass -- a verdict diluted by its own history
        would take as many passes to turn as it took to earn, which on W24's
        60-75 s cadence is the difference between a line an operator reads and
        a window spent finding out.
        """
        card = _Card(4096)
        g = _guard(card)
        pays = {"driver": True}

        def free_up_to(nbytes):
            take = min(300 * MIB, max(0, int(nbytes)))
            if pays["driver"]:
                card.free += take
            return take

        g.register("was-good", 10, free_up_to)
        for _ in range(5):
            g.ensure_headroom(int(card.free + 300 * MIB))
        rec = g.delivery_report()["was-good"]
        self.assertFalse(rec.is_phantom)
        self.assertGreater(rec.delivered_bytes, 0)
        pays["driver"] = False
        g.ensure_headroom(int(card.free + 300 * MIB))
        self.assertTrue(
            g.delivery_report()["was-good"].is_phantom,
            "five delivering passes must not buy the sixth one credit it did "
            "not earn; the question is whether to believe the NEXT claim.",
        )


class TheRefusalNamesThePhantom(unittest.TestCase):
    def test_a_divergence_is_stated_in_the_detail_line(self):
        """W24 cost a GPU window to discover that a funder promised 320 MiB
        and delivered 0. The line that would have said so costs nothing."""
        card = _Card(4096)
        g = _guard(card)
        g.register("phantom", 10, card.provider(8000, to_driver=False))
        res = g.ensure_headroom(4500 * MIB)
        self.assertIn("phantom", res.detail)
        self.assertIn("claimed", res.detail)
        self.assertIn("delivered", res.detail)

    def test_an_honest_ladder_says_nothing_extra(self):
        """The clause is edge-triggered on divergence. A line that fires on
        every pass is a line nobody reads."""
        card = _Card(4096)
        g = _guard(card)
        g.register("real", 10, card.provider(600, to_driver=True))
        res = g.ensure_headroom(4500 * MIB)
        self.assertNotIn("delivered only", res.detail)


class TheLenderKeepsItsOwnAccounting(unittest.TestCase):
    """``lend_to_level`` already measured its own delta and must keep doing
    so; this fix generalises it, it does not replace it."""

    def test_a_lend_still_reports_the_measured_delta(self):
        card = _Card(4096)
        g = _guard(card)
        g.register("real", 10, card.provider(600, to_driver=True))
        res = g.lend_to_level(600 * MIB, column=[4096 * MIB, 4096 * MIB])
        self.assertEqual(600 * MIB, res.reclaimed)
        self.assertEqual(600 * MIB, g.lent_total)

    def test_a_lend_does_not_pollute_the_gates_counter(self):
        card = _Card(4096)
        g = _guard(card)
        g.register("real", 10, card.provider(600, to_driver=True))
        g.lend_to_level(600 * MIB, column=[4096 * MIB, 4096 * MIB])
        self.assertEqual(0, g.reclaimed_total)

    def test_a_lend_records_delivery_in_the_ledger_as_well(self):
        """The ledger is a property of the PROVIDER, not of the caller that
        happened to spend it."""
        card = _Card(4096)
        g = _guard(card)
        g.register("phantom", 10, card.provider(600, to_driver=False))
        g.lend_to_level(600 * MIB, column=[4096 * MIB, 4096 * MIB])
        self.assertTrue(g.delivery_report()["phantom"].is_phantom)


class ConcurrentTakingReadsAsZeroNotNegative(unittest.TestCase):
    def test_a_falling_free_column_does_not_credit_a_negative(self):
        """The lender already named this case. A negative delivery would make
        the running total lie in the other direction."""
        card = _Card(4096)
        g = _guard(card)

        def thief(nbytes):
            card.free -= 100 * MIB  # another process took memory mid-ladder
            return int(nbytes)

        g.register("thief", 10, thief)
        res = g.ensure_headroom(4500 * MIB)
        self.assertEqual(0, res.reclaimed)
        self.assertEqual(0, g.delivery_report()["thief"].delivered_bytes)


if __name__ == "__main__":
    unittest.main()
