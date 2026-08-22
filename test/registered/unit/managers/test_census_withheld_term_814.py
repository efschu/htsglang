"""#814: the pool census must not read WITHHELD CAPACITY as a row leak.

THE SPECIMEN. Serving on this rig, one boot, three ranks, identical on each:

    POOL CENSUS at-arm tp_to_pp: size=465190 free=1680 cached=123248
                                 unaccounted=340262
    UNACCOUNTED: n=340262 min=124929 max=465190 runs=1 longest_run=340262

Read as a leak, that is 73% of the pool surviving its requests. It is not a
leak. `KvRowCap` had the allocator capped at 124928 rows, so
465190 - 124928 = 340262 ids were WITHHELD -- exactly the reported number, in
one contiguous block at the top of the id space, which is the shape a cap
makes and the shape a leak never makes (a leak is fragmented).

The tree already learned this once. `KvRowCap._publish`
(kv_backing_relief.py:530-549) exists solely because the SCHEDULER's idle
invariant made the identical mistake:

    "Withheld capacity is in none of those buckets, so without a term of its
     own it reads as a LEAK -- and it is a fatal one: the first boot that
     exercised the cap died at the first idle check with
     'pool memory leak detected! [full] total=500000, available=419745'."

It published `allocator.residency_withheld_slots` for that check. The FLIP
CENSUS never got the term: `_pool_census` computes
`range(1, size+1) - free - cached` and consults the field nowhere, while
`alloc.size` keeps reporting the full pre-shrink id space (allocator/base.py
:38 is never rewritten by the shrink).

BOTH DIRECTIONS ARE TESTED HERE, and that is the point rather than a courtesy.
Subtracting a term is only a fix if the instrument can still say "leak"
afterwards; otherwise it is a blindfold. So: withheld capacity must stop
reading as a leak, AND a genuine unexplained row must still read as one, AND
the two must be distinguishable when they occur together.

No GPU, no flip, no scheduler -- the shipped method is driven directly.
"""

import logging
import unittest
import unittest.mock
from types import SimpleNamespace

from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5)

# The live specimen, kept as named constants so the numbers in the docstring
# and the numbers under test cannot drift apart.
POOL_SIZE = 465190
CAP = 124928
FREE = 1680
CACHED = 123248
WITHHELD = POOL_SIZE - CAP  # 340262


class _Pages:
    """Stands in for the allocator's id tensors; only `.tolist()` is used."""

    def __init__(self, ids):
        self._ids = list(ids)

    def tolist(self):
        return list(self._ids)


def _census(
    *,
    size,
    free_ids,
    cached_ids,
    withheld_slots,
    page_size=1,
):
    """Drive the real `_pool_census` and return the line it logged."""
    alloc = SimpleNamespace(
        size=size,
        free_pages=_Pages(free_ids),
        release_pages=_Pages([]),
        page_size=page_size,
        residency_withheld_slots=withheld_slots,
        available_size=lambda: len(free_ids),
    )
    tree = SimpleNamespace(all_values_flatten=lambda: _Pages(cached_ids))
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=alloc,
        tree_cache=tree,
        running_mbs=[],
        running_batch=None,
        last_batch=None,
        phase_flip_stacks=None,
        tp_worker=None,
    )
    stub = SimpleNamespace(_census_scheduler=scheduler)
    stub._owner_ident = PhaseFlipRuntime._owner_ident
    stub._owner_pool_of = PhaseFlipRuntime._owner_pool_of
    stub._census_owner_probe = lambda *a, **k: None
    stub._pool_census = PhaseFlipRuntime._pool_census.__get__(stub, SimpleNamespace)

    logger = logging.getLogger("sglang.srt.managers.phase_flip_runtime")
    with unittest.mock.patch.object(logger, "warning") as warn:
        stub._pool_census("at-arm", "tp_to_pp")
    assert warn.called, "the census must always emit"
    args = warn.call_args[0]
    return args[0] % tuple(args[1:])


def _specimen(withheld_slots=WITHHELD, extra_leak=()):
    """The live specimen: ids 1..CAP in circulation, CAP+1..size withheld."""
    free_ids = list(range(1, FREE + 1))
    cached_ids = list(range(FREE + 1, FREE + 1 + CACHED))
    # Anything in `extra_leak` is a row that is genuinely unexplained: below
    # the cap, so not withheld, yet neither free nor in the tree.
    free_ids = [i for i in free_ids if i not in set(extra_leak)]
    cached_ids = [i for i in cached_ids if i not in set(extra_leak)]
    return _census(
        size=POOL_SIZE,
        free_ids=free_ids,
        cached_ids=cached_ids,
        withheld_slots=withheld_slots,
    )


class TestWithheldCapacityIsNotALeak(CustomTestCase):
    """Direction 1: the cap must stop reading as 340262 leaked rows."""

    def test_the_specimen_reports_zero_unaccounted(self):
        text = _specimen()
        self.assertIn("unaccounted=0", text)

    def test_the_withheld_rows_are_reported_as_their_own_term(self):
        """Subtracted is not enough -- the capacity must stay VISIBLE.

        340262 ids out of circulation is the single most important fact about
        this pool. A fix that merely subtracted them would trade a false leak
        for a silent 3.7x capacity loss, which is the worse of the two.
        """
        text = _specimen()
        self.assertIn(f"withheld={WITHHELD}", text)


class TestAGenuineLeakStillReadsAsOne(CustomTestCase):
    """Direction 2: the instrument must still be able to say "leak"."""

    def test_an_uncapped_pool_with_an_unexplained_row_reports_it(self):
        text = _census(
            size=1000,
            free_ids=list(range(1, 500)),
            cached_ids=list(range(500, 1000)),  # id 1000 is unexplained
            withheld_slots=0,
        )
        self.assertIn("unaccounted=1", text)
        self.assertIn("withheld=0", text)

    def test_a_leak_UNDER_the_cap_is_not_masked_by_the_withheld_term(self):
        """THE DISCRIMINATOR, and the reason both terms are reported.

        A cap withholds the TOP of the id space. A row leaking below the cap
        is a different event entirely, and subtracting a raw count must not
        swallow it.

        THE ASSERTION IS ON THE IDS, NOT THE COUNT, and that distinction is
        load-bearing rather than pedantic. A count-subtracting implementation
        (`sorted(unexplained)[:withheld_n]`) reports unaccounted=3 here too --
        it just confiscates the three HIGHEST ids instead of the three real
        ones, and a count assertion passes it. Measured: that mutant survived
        this test until it named the rows. The census prints
        `sorted(leaked)[:12]`, so the identities are on the line and can be
        demanded.
        """
        leaked_rows = (7, 9, 11)
        text = _specimen(extra_leak=leaked_rows)
        self.assertIn(f"unaccounted={len(leaked_rows)}", text)
        self.assertIn(f"withheld={WITHHELD}", text)
        self.assertIn("[7, 9, 11]", text)


class TestTheTermIsReadDefensively(CustomTestCase):
    """An allocator without the field must not break or lie."""

    def test_a_missing_field_reads_as_zero_withheld(self):
        alloc_free = list(range(1, 500))
        alloc_cached = list(range(500, 1000))
        text = _census(
            size=1000,
            free_ids=alloc_free,
            cached_ids=alloc_cached,
            withheld_slots=None,  # absent / not yet published
        )
        self.assertIn("withheld=0", text)
        self.assertIn("unaccounted=1", text)

    def test_withheld_is_converted_from_tokens_to_page_ids(self):
        """`residency_withheld_slots` is published in TOKENS, not page ids.

        `_publish` multiplies by `page_size` on purpose ("Published in the
        unit `available_size()` reports, which is TOKENS"). The census works
        in page ids, so it must divide back, or every paged lane misreports
        by exactly that factor.
        """
        text = _census(
            size=1000,
            free_ids=list(range(1, 501)),
            cached_ids=[],
            withheld_slots=500 * 4,  # 500 page ids withheld, page_size 4
            page_size=4,
        )
        self.assertIn("withheld=500", text)
        self.assertIn("unaccounted=0", text)


if __name__ == "__main__":
    unittest.main()
