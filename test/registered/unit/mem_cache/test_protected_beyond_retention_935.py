"""#935 -- rows between the retention and the protected length belong to nobody.

THE INTERVAL. `UnifiedRadixCache.cache_finished_req` truncates and frees like
this::

    if effective_cache_len < len(token_ids):
        free_start = max(effective_cache_len, req.cache_protected_len)
        self.token_to_kv_pool_allocator.free(kv_indices[free_start:])
        token_ids  = token_ids[:effective_cache_len]
        kv_indices = kv_indices[:effective_cache_len]

and then inserts only the truncated key. So rows in
**[effective_cache_len, cache_protected_len)** are neither INSERTED (the key
stops at the retention) nor FREED (the free starts at the protected length).
After the request finishes they are owned by nobody -- which is exactly the
census verdict "belong to no enumerated owner", and exactly the per-request
deficit #935 measures.

WHY THE max() IS NOT ITSELF THE BUG, and this is what decides the fix. Both
free-sites on this path start at `cache_protected_len` on one stated premise --
`retention_shrinks_protected`: "that length is COMMITTED: the tree owns the KV
below it". While that holds, the max() is a correct optimisation and nothing
leaks: the rows below cpl really are the tree's. The defect is that the
FINISHED path never checks the premise, while the UNFINISHED path asserts on it
(`assert req.cache_protected_len <= len(new_indices) + page_size - 1`,
unified_radix_cache.py:1231 -- the #824 guard). One path guards, the other
trusts.

TWO INDEPENDENT PRODUCERS of a cpl above the retention are known, and this test
uses NEITHER: #930's PP-admission truncation (which does not update cpl) and
#928's refusal-driven re-prefill (which carries the old high-water in). Both are
being repaired elsewhere. The request here is constructed with `cpl > ecl`
DIRECTLY, so the test pins the interval itself rather than a route to it, and
stays meaningful once both producers are fixed. The gap is the root; a producer
is an occasion.

SIZE CROSS-CHECK against the measured deficit. The 2i probe was 9573 prompt
tokens with a 120-token completion: 9573 + 120 = 9693 rows, against a measured
per-request deficit of +9694 (the extra row is the one-cell width #922 fixed on
a different reader -- seqlen counts one past the allocation). A whole request's
KV, not a fragment, which is what a cpl at full length against a retention near
zero produces.

NAMED, NOT FREED, and the direction is the safety argument. Freeing the range
would be correct if the tree does not own it and a DOUBLE FREE if it does, and
this site cannot tell the two apart -- that is precisely why the premise is
trusted here rather than verified. A wrong free is a use-after-free; a named
leak is a number in a log. So the guard converts a SILENT loss into an attributed
one, with the interval and a running total, and leaves the repair to whoever
lets cpl go stale.

WHAT EACH TEST HOLDS DOWN
  1. cpl above the retention is named, with the exact row count -- the defect;
  2. cpl at or below the retention is silent  -- mutant guard: a warning on
     every finished request is noise, and noise is not a finding;
  3. the running total accumulates across requests -- it is a per-request loss,
     so the boot-visible quantity is the sum;
  4. the guard never raises -- it runs at request finish and must not kill a
     rank over an accounting fault.
"""

import logging
import unittest

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

_LOGGER = "sglang.srt.mem_cache.unified_radix_cache"


class _Req:
    def __init__(self, cache_protected_len, rid="probe"):
        self.cache_protected_len = cache_protected_len
        self.rid = rid


class _Cache:
    """Only the guard is under test, so the instance carries only what it
    touches -- constructing a real UnifiedRadixCache would drag in pools that
    have nothing to do with this arithmetic."""

    _note_protected_beyond_retention = (
        UnifiedRadixCache._note_protected_beyond_retention
    )


class TestProtectedBeyondRetention935(unittest.TestCase):
    def test_a_protected_length_above_the_retention_is_named(self):
        """THE DEFECT, at the measured magnitude."""
        cache = _Cache()
        req = _Req(cache_protected_len=9693)
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            cache._note_protected_beyond_retention(req, 0)
        blob = "\n".join(cm.output)
        self.assertIn("#935 PROTECTED-BEYOND-RETENTION", blob)
        self.assertIn("9693", blob)
        self.assertIn("probe", blob)
        self.assertEqual(cache._protected_beyond_retention_rows, 9693)

    def test_a_partial_gap_counts_only_the_gap(self):
        cache = _Cache()
        cache._note_protected_beyond_retention(_Req(9000), 8000)
        self.assertEqual(cache._protected_beyond_retention_rows, 1000)

    def test_a_protected_length_within_the_retention_is_silent(self):
        """MUTANT GUARD. Warning on every finished request is noise, and the
        premise HOLDS in the ordinary case -- the tree really does own the rows
        below cpl. Firing there would train readers to skip the line."""
        cache = _Cache()
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            cache._note_protected_beyond_retention(_Req(4096), 9447)
            cache._note_protected_beyond_retention(_Req(4096), 4096)
            cache._note_protected_beyond_retention(_Req(0), 0)
        self.assertEqual(getattr(cache, "_protected_beyond_retention_rows", 0), 0)

    def test_the_total_accumulates_across_requests(self):
        """It is a PER-REQUEST loss, so the boot-visible quantity is the sum --
        which is how #935 was seen at all (a deficit that grew with traffic)."""
        cache = _Cache()
        for i in range(5):
            cache._note_protected_beyond_retention(_Req(9693, rid=f"r{i}"), 0)
        self.assertEqual(cache._protected_beyond_retention_rows, 5 * 9693)
        self.assertEqual(cache._protected_beyond_retention_count, 5)

    def test_the_guard_never_raises(self):
        """It runs at request finish; an exception here kills a rank over an
        accounting fault."""
        cache = _Cache()

        class _Odd:
            rid = "odd"
            cache_protected_len = None

        logging.getLogger(_LOGGER).setLevel(logging.WARNING)
        cache._note_protected_beyond_retention(_Odd(), 0)
        self.assertEqual(getattr(cache, "_protected_beyond_retention_rows", 0), 0)


if __name__ == "__main__":
    unittest.main()
