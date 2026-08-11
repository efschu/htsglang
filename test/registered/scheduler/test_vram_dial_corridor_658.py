# SPDX-License-Identifier: Apache-2.0
"""#658 / register C18: the VRAM budget dial answers to the CORRIDOR LAW.

WHY THIS FILE EXISTS
--------------------
Register entry C18 recorded two floor models that had never met. The dial
(#330) derives its OWN card-level floor in ``_measure_local_floor_bytes``
(NVML used minus the VMM-backed KV bytes) and spends against
``budget_bytes - floor_bytes``, with no reference to ``CorridorGuard`` or to
``law_floor_bytes``. It was inert only because ``--enable-vram-dial`` was
absent from the ship config; a physically-growing allocator answering to a
different floor than the law is the same shape as C17, and it diverges the
moment the dial is switched on.

The fix is deliberately NOT a second check bolted onto the dial's commit
path. It is a TERM in the floor the dial already spends against, which is
the same shape the C20 seam-entry margin took: one floor, one arithmetic,
one refusal path. Every consumer of ``floor_bytes`` -- the unit cap, the
budget rows, the minimum viable budget, the effective ceiling -- then
reserves the corridor law without knowing it exists.

THE BOOT STATE IS UNCHANGED, AND THAT IS THE POINT. The natural boot budget
is computed as ``floor + backed``, so raising the floor by the law raises the
natural budget by exactly the same amount and ``budget - floor`` (the bytes
the KV pool may spend) is bit-identical. The term bites only when a caller
names an ABSOLUTE budget -- which is precisely when an external tenant is
taking bytes off this card and the corridor law is what protects serving.

WHAT IS DELIBERATELY NOT TESTED HERE. That the dial can serve the guard as a
relief ACTUATOR: it cannot, and the reason is structural rather than missing
glue. A shrink flushes the radix cache and commits only at a fully-idle
group boundary (``ready_fn=scheduler.is_fully_idle``), so it offers no
bounded-latency release to a synchronous gate. The direction wired here is
therefore the other one: an external budget REDUCTION is treated as corridor
pressure and the guard's relief ladder decides what yields, BEFORE the dial's
own capacity arithmetic is asked for the residual.
"""

import unittest

from sglang.srt.managers.vram_dial import (
    MIB,
    KvCapacityRuntime,
    RankState,
    corridor_law_floor_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _mk_ranks(floors, budgets, trbs, drbs, reserves, draft_reserves, boot_rows):
    return [
        RankState(
            device_index=r,
            card_uuid=f"GPU-{r}",
            budget_bytes=budgets[r],
            floor_bytes=floors[r],
            target_row_bytes=trbs[r],
            draft_token_bytes=drbs[r],
            target_reserve_rows=reserves[r],
            draft_reserve_tokens=draft_reserves[r],
            boot_backed_rows=boot_rows[r],
            backed_rows=boot_rows[r],
        )
        for r in range(len(floors))
    ]


class _Relief:
    """A stand-in for ``CorridorGuard.ensure_headroom``'s relief ladder.

    Records every ask, and returns a fixed number of bytes. A ladder that
    is never called leaves ``asks`` empty, which is the red state this file
    was written against.
    """

    def __init__(self, gives=0):
        self.asks = []
        self.gives = gives

    def __call__(self, want_bytes, reason=""):
        self.asks.append((int(want_bytes), reason))
        return int(self.gives)


def _mk_runtime(
    *, relief=None, floors=None, budgets=None, current_c=4000, n=1, trb=MIB
):
    # Rows are sized in MiB so that a dial expressed in MiB is a meaningful
    # fraction of the pool. A byte-scale row makes every reduction fall below
    # the minimum viable budget and every test in this file vacuous.
    floors = floors if floors is not None else [0]
    trbs, drbs = [trb] * n, [0] * n
    reserves, draft_reserves = [10**9] * n, [0] * n
    boot_rows = [current_c + 1] * n
    if budgets is None:
        budgets = [floors[r] + boot_rows[r] * trbs[r] for r in range(n)]
    return KvCapacityRuntime(
        n_ranks=n,
        my_rank=0,
        ranks=_mk_ranks(
            floors, budgets, trbs, drbs, reserves, draft_reserves, boot_rows
        ),
        page_size=1,
        current_c=current_c,
        user_cap=None,
        consensus_interval=1,
        collective_min=None,
        ready_fn=lambda: True,
        current_vector_fn=lambda: (1,) * n,
        pending_reshard_fn=lambda: None,
        commit_fn=lambda c, b, m: 0,
        relief_fn=relief,
    )


class TestCorridorLawIsInTheDialFloor(CustomTestCase):
    """C18: the law is a term in the floor, not a second opinion."""

    def test_the_law_floor_is_a_positive_byte_figure(self):
        # The corridor law is 1024 MiB by user order. A zero here would make
        # every assertion below vacuously true.
        self.assertGreaterEqual(corridor_law_floor_bytes(), 1024 * MIB)

    def test_an_absolute_budget_reserves_the_law(self):
        """The whole of C18 in one number.

        Two runtimes, identical except that one's floor carries the law. Given
        the SAME absolute budget, the corridor-aware one must plan a strictly
        smaller KV ceiling -- by at least the law, converted to rows.
        """
        law = corridor_law_floor_bytes()
        budget = 8000 * MIB
        trb = MIB
        naive = _mk_runtime(floors=[100 * MIB], budgets=[budget], trb=trb)
        lawful = _mk_runtime(floors=[100 * MIB + law], budgets=[budget], trb=trb)
        self.assertLess(lawful.compute_target_c(), naive.compute_target_c())
        # And by the right amount: the law divided by the row size, since the
        # only difference between the two is that many unspendable bytes.
        withheld = (naive.compute_target_c() - lawful.compute_target_c()) * trb
        self.assertGreaterEqual(withheld, law * 0.9)

    def test_a_dial_that_would_breach_the_law_is_refused_with_numbers(self):
        """A reduction into the law's 1024 MiB is rejected, not clamped.

        The refusal must NAME the arithmetic -- fail fast with the numbers, so
        an external tenant's controller learns why rather than retrying.
        """
        law = corridor_law_floor_bytes()
        rt = _mk_runtime(floors=[100 * MIB + law])
        ok, msg = rt.apply_budget_request(device="all", budget_mib=150)
        self.assertFalse(ok)
        self.assertIn("floor", msg.lower())
        # Nothing changed: a rejected dial is a no-op.
        self.assertEqual(rt.op_seq, 0)


class TestReductionIsCorridorPressure(CustomTestCase):
    """An external budget REDUCTION is pressure, and the ladder decides."""

    def test_a_reduction_asks_the_relief_ladder_first(self):
        relief = _Relief(gives=0)
        rt = _mk_runtime(relief=relief)
        before = rt._ranks[0].budget_bytes
        ok, _ = rt.apply_budget_request(device="all", release_mib=64)
        self.assertTrue(ok)
        self.assertEqual(len(relief.asks), 1, "the ladder was never consulted")
        want, reason = relief.asks[0]
        # It is asked for exactly the bytes the external tenant is taking.
        self.assertEqual(want, 64 * MIB)
        self.assertIn("budget", reason.lower())
        self.assertEqual(rt._ranks[0].budget_bytes, before - 64 * MIB)

    def test_an_increase_is_not_pressure(self):
        """Growth must not spend the relief ladder.

        A budget INCREASE returns residency; asking a spill ladder for bytes
        at that moment would pay a restore cost to fund memory that was just
        handed back.
        """
        relief = _Relief(gives=0)
        rt = _mk_runtime(relief=relief)
        rt.apply_budget_request(device="all", release_mib=64)
        relief.asks.clear()
        rt.apply_budget_request(device="all", release_mib=-64)
        self.assertEqual(relief.asks, [])

    def test_a_rejected_reduction_does_not_spend_the_ladder(self):
        """No relief for a dial that changes nothing.

        The rejection path returns before any budget is committed, so a
        ladder spent there would have paid a real restore cost for a request
        that was refused.
        """
        law = corridor_law_floor_bytes()
        relief = _Relief(gives=0)
        rt = _mk_runtime(relief=relief, floors=[100 * MIB + law])
        ok, _ = rt.apply_budget_request(device="all", budget_mib=150)
        self.assertFalse(ok)
        self.assertEqual(relief.asks, [])

    def test_the_ladder_is_optional(self):
        """No guard wired -> the dial still works, unchanged.

        The relief hook must degrade to the shipped behaviour rather than
        raise: a rank whose guard failed to build is a lost optimisation,
        never a failed dial.
        """
        rt = _mk_runtime(relief=None)
        ok, _ = rt.apply_budget_request(device="all", release_mib=64)
        self.assertTrue(ok)

    def test_a_raising_ladder_never_takes_the_dial_down(self):
        """A provider that raises is absorbed, exactly as in the guard."""

        def _boom(want_bytes, reason=""):
            raise RuntimeError("provider exploded")

        rt = _mk_runtime(relief=_boom)
        ok, _ = rt.apply_budget_request(device="all", release_mib=64)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
