"""#816 -- the allocator must never expose an id with no page behind it.

THE CRASH THIS CLOSES, measured on metal 2026-08-23 00:33:54 (boot
boot_816_probe_0823_0022, rank PP1)::

    #788 PP-ADMISSION verdict=ADMIT ... avail=97385 evictable=320465
    Assertion `index >= 105414 (out of range): set_kv_buffer (MHA)' failed.

105414 is ``self.size + page_size``, i.e. 105413 COMMITTED rows, while
admission was pricing against 97385 + 320465 = 417850 REACHABLE ones. 312437
rows of pure exposure. The first chunked prefill whose tail landed above the
backing handed ``masked_set_kv_buffer_kernel`` a ``loc`` its buffer could not
address and the device assert at memory_pool.py:4978 killed the rank. Four
boots died this way on 2026-08-22/23; a CUDA coredump named the kernel.

THE INVARIANT, and it is the MIRROR of one this module already enforces::

    highest live row  <=  committed backing  >=  exposed id space
    \\________ #717/#722 ________/            \\________ #816 ________/

#717/#722 (revert b7868580a9, rebuild 675793cdc8) is the LEFT half: a cap
below rows that were still live, and the next READ was an illegal address.
``_shrink_to`` has enforced it since the rebuild by re-reading
``_max_live_row()`` AFTER the eviction. The RIGHT half had no enforcement at
all -- the release paths compared against a remembered target or the id-space
span, never against the committed backing.

WHY A CLAMP AND NOT A WIDER BOUND. ``graph_safe_store_bound``
(memory_pool.py:131) is deliberately graph-stable and argues its case at
length; widening it re-admits the silent-corruption band it exists to exclude.
The id space is what is wrong, so the id space is what is corrected.

THE #684 LESSON IS OBSERVED: the clamp targets ``_current_rows()``, a MEASURED
committed count, never ``_rows_at_boot``, the remembered one whose staleness
was the whole of #684.

Hermetic, no GPU: a fake VMM pool and a fake allocator, following
``test_kv_backing_recovery_clamp_684.py``.
"""

import unittest
from typing import Optional

import torch

from sglang.srt.managers.kv_backing_relief import exposure_over_backing
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8)

BYTES_PER_ROW = 4096
LAW_FLOOR = 1024 * 1024 * 1024


class _FakeAlloc:
    """Just enough allocator for ``KvRowCap`` to hold ids back.

    ``free_pages`` is the list the cap filters; ``residency_withheld_slots`` is
    the term #814 D1 taught the census to read. No listener hook, so the cap
    logs once and works on the list it is given -- which is what this file
    measures.
    """

    def __init__(self, size: int):
        self.size = int(size)
        self.free_pages = torch.arange(1, int(size) + 1, dtype=torch.int64)
        self.residency_withheld_slots = 0


class _FakeVmmPool:
    """A VMM-backed pool whose committed rows and id space can diverge.

    That divergence IS the defect, so the fixture has to be able to express it:
    ``size`` is the id space the allocator spans, ``full_pool_backed_rows`` is
    what is physically committed.
    """

    def __init__(self, backed_rows: int, reserved_rows: int, page_size: int = 1):
        self.full_pool_backed_rows = int(backed_rows)
        self.reserved = int(reserved_rows)
        self.reserved_backing_rows = int(reserved_rows)
        #: the allocator's id space -- what ``_reservation_rows()`` reads
        self.size = int(reserved_rows)
        self.page_size = int(page_size)
        self.attempts = []

    def runtime_set_backing_rows(self, rows: int) -> None:
        self.attempts.append(int(rows))
        if not (self.page_size <= int(rows) <= self.reserved):
            raise ValueError(
                f"final_num_tokens={int(rows)} must satisfy "
                f"page_size={self.page_size} <= final <= reserved={self.reserved}"
            )
        self.full_pool_backed_rows = int(rows)


def _relief(
    pool: _FakeVmmPool,
    *,
    alloc: Optional[_FakeAlloc] = None,
    free_mib: int = 8192,
    live_rows=(),
):
    from sglang.srt.managers.kv_backing_relief import KvBackingRelief

    relief = KvBackingRelief(
        pool,
        allocator=alloc,
        # a TENSOR, not a list: ``_max_live_row`` calls ``.max()`` on it and
        # treats an exception as 'unreadable' (-1), which would make this
        # fixture silently measure nothing.
        live_slots_fn=lambda: (
            torch.tensor(list(live_rows), dtype=torch.int64)
            if live_rows
            else torch.empty((0,), dtype=torch.int64)
        ),
        bytes_per_row=BYTES_PER_ROW,
        probe=lambda: free_mib * (1 << 20),
        device_index=0,
        buffers=1,
        law_floor_bytes=LAW_FLOOR,
        pool_fn=lambda: None,
    )
    return relief


class TheArithmeticIsItsOwnDefinition(unittest.TestCase):
    """``exposure_over_backing`` -- one definition, so callers cannot drift."""

    def test_metal_numbers_reproduce(self):
        # the 00:33:54 crash, to the row
        self.assertEqual(exposure_over_backing(417850, 105413), 312437)

    def test_sound_state_is_zero(self):
        self.assertEqual(exposure_over_backing(105413, 105413), 0)

    def test_more_backing_than_exposure_is_not_negative(self):
        """Over-BACKED is legal and must not read as negative exposure."""
        self.assertEqual(exposure_over_backing(1000, 4000), 0)


class TheClampClosesEveryReleasePath(unittest.TestCase):
    """One case per release site, each red without its own call."""

    def test_recovery_else_leg_no_longer_exposes_the_whole_id_space(self):
        """SITE 1 -- ``recover``'s else leg. THE METAL DEFECT.

        ``recover`` ends with ``if now < boot_rows: engage(now)``. That
        compares the committed rows against the REMEMBERED recovery target,
        not against the id space. With ``boot_rows <= now < reservation`` it
        takes the else leg, engages nothing, and the release just above has
        exposed every id over ``now`` committed rows.

        Here: 105413 committed, 417850 id space, boot target already reached.
        Without the clamp ``exposed_rows()`` is 417850 -- 312437 rows with no
        page behind them, which is exactly the crash.
        """
        pool = _FakeVmmPool(backed_rows=105413, reserved_rows=417850)
        alloc = _FakeAlloc(417850)
        relief = _relief(pool, alloc=alloc)
        relief._rows_at_boot = 105413  # target already met -> the else leg

        relief.recover()

        self.assertLessEqual(
            relief.exposed_rows(),
            relief.backed_rows(),
            "recover() left ids exposed above the committed backing (#816)",
        )
        self.assertEqual(relief.exposed_rows(), 105413)

    def test_cap_agreement_no_cap_leg_is_clamped(self):
        """SITE 2 -- the levelling's ``if level < reservation`` leg.

        When the agreed group level is not below the id-space span the old
        code engaged nothing at all, exposing this rank's whole id space
        regardless of what it had committed.
        """
        pool = _FakeVmmPool(backed_rows=118784, reserved_rows=446522)
        alloc = _FakeAlloc(446522)
        relief = _relief(pool, alloc=alloc)
        relief._cap.engage(200000)

        relief.clamp_exposure_to_backing("test")

        self.assertEqual(relief.exposed_rows(), 118784)

    def test_a_sound_state_is_left_alone(self):
        """The other direction. Without this, 'it clamps' proves nothing.

        Committed backing >= id space, so there is nothing to withdraw and the
        clamp must not cap anything -- a clamp that always fires would pass
        every case above while destroying admission capacity.
        """
        pool = _FakeVmmPool(backed_rows=417850, reserved_rows=417850)
        alloc = _FakeAlloc(417850)
        relief = _relief(pool, alloc=alloc)

        withdrawn = relief.clamp_exposure_to_backing("test")

        self.assertEqual(withdrawn, 0)
        self.assertFalse(
            relief._cap.engaged, "clamped a state that was already sound"
        )


class ItCannotBecomeTheDefectItMirrors(unittest.TestCase):
    """#722 is the failure mode a careless version of this fix would cause."""

    def test_the_clamp_never_lowers_the_backing(self):
        """It corrects the ID SPACE, never the pages.

        #717 attempt-1 (c4e557963e, reverted b7868580a9) lowered backing to a
        target validated only against an INTENDED eviction and put 69054 rows
        of backing under a highest live row of 233289 -- live rows unmapped,
        illegal address. This fix must never touch ``runtime_set_backing_rows``
        at all, and the fixture records every call it would make.
        """
        pool = _FakeVmmPool(backed_rows=105413, reserved_rows=417850)
        alloc = _FakeAlloc(417850)
        relief = _relief(pool, alloc=alloc)

        relief.clamp_exposure_to_backing("test")

        self.assertEqual(
            pool.attempts, [], "the clamp changed the backing; that is #722"
        )
        self.assertEqual(pool.full_pool_backed_rows, 105413)

    def test_backing_already_below_the_live_set_is_reported_not_hidden(self):
        """The pre-existing #722 state must stay visible THROUGH this fix.

        Clamping to the backing is still right -- it stops NEW ids escaping --
        but it does not make already-handed-out live rows addressable, and a
        clamp that silently made the state look tidy would bury exactly the
        condition #722 exists to catch.
        """
        pool = _FakeVmmPool(backed_rows=69054, reserved_rows=417850)
        alloc = _FakeAlloc(417850)
        relief = _relief(pool, alloc=alloc, live_rows=[233289])

        with self.assertLogs(
            "sglang.srt.managers.kv_backing_relief", level="ERROR"
        ) as caught:
            relief.clamp_exposure_to_backing("test")

        joined = " ".join(caught.output)
        self.assertIn("233289", joined)
        self.assertIn("69054", joined)
        # and it still did the one thing it can do
        self.assertEqual(relief.exposed_rows(), 69054)


if __name__ == "__main__":
    unittest.main()


class EveryReleaseSiteIsWired(unittest.TestCase):
    """A UNITY test, in the spirit of ``test_kv_store_bound_unity.py``.

    WHY THIS EXISTS, and it is not decoration. The behavioural cases above
    drive ``recover()`` for real, but the other two release sites
    (``_shrink_to``'s failure path and the cap agreement) need a scheduler and
    a collective to reach, so they were exercised by calling the helper
    directly -- which tests the HELPER, not the WIRING. Measured: deleting the
    clamp call in ``apply_cap_agreement`` left all eight cases green. A call
    edge no test can kill is not covered, so this closes it structurally.

    THE RULE: every ``self._cap.release()`` must be followed, within the same
    handful of lines, by a ``clamp_exposure_to_backing`` call. Releasing
    re-exposes the allocator's whole id space; a release that does not then
    re-check it against the committed backing is #816 by construction. A
    FUTURE release site inherits the requirement automatically, which is the
    "one consumer never got the treatment" lesson #345, #352 and #355 each
    paid for separately.
    """

    def _functions_releasing_the_cap(self):
        """Every function in the module that calls ``self._cap.release()``.

        AST rather than a line window, and the difference is not cosmetic: the
        first version of this test used a 12-line window and flagged
        ``recover`` -- whose clamp is 21 lines below its release, with a
        legitimate if/else in between. A window measures PROXIMITY; the rule is
        actually "the same function must not return having released without
        re-clamping", which is a scope question, so it is asked of the scope.
        """
        import ast
        import inspect
        import textwrap

        from sglang.srt.managers import kv_backing_relief

        tree = ast.parse(textwrap.dedent(inspect.getsource(kv_backing_relief)))
        out = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.dump(node)
            if "attr='release'" in body and "attr='_cap'" in body:
                out[node.name] = "clamp_exposure_to_backing" in body
        return out

    def test_every_function_that_releases_the_cap_also_clamps(self):
        found = self._functions_releasing_the_cap()
        self.assertGreaterEqual(
            len(found), 3, f"expected the three known release sites, got {found}"
        )
        unwired = sorted(name for name, wired in found.items() if not wired)
        self.assertEqual(
            unwired,
            [],
            "these functions release the cap -- re-exposing the allocator's "
            "whole id space -- without re-checking it against the committed "
            f"backing (#816): {unwired}",
        )
