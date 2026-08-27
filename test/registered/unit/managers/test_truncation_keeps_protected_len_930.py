# Copyright 2023-2024 SGLang Team
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
"""#930: the PP admission truncation must move cache_protected_len with it.

WHAT THE FIELD MEANS: ``cache_protected_len`` is how many LEADING rows of this
request's KV the TREE owns. ``init_next_round_input`` sets it equal to
``len(prefix_indices)``. The #791 admission-uniformity block in
``_get_new_batch_prefill_raw`` then truncates ``prefix_indices`` to the PP-agreed
``told`` -- on PP0 from the guard's clamped candidate, downstream from PP0's
decision -- and NEITHER branch touched the protected length. The request was
left claiming more tree-owned rows than it holds.

WHY THE SURPLUS IS NOT HARMLESS, and why it stayed invisible. It is the SAFE
direction for ``_insert_helper``'s duplicate free -- a larger ``dup_start``
frees less -- so it never showed up as a double claim. It is the DANGEROUS
direction for ``cache_finished_req``'s truncate branch
(``unified_radix_cache.py:1111-1116``)::

    free_start = max(effective_cache_len, req.cache_protected_len)
    free(kv_indices[free_start:])    # starts ABOVE the interval
    ...                              # the insert covers only up to ecl

With ``cache_protected_len > effective_cache_len`` the rows in
``[effective_cache_len, cache_protected_len)`` are neither freed nor inserted
and belong to nobody afterwards. That interval is #935's per-request row leak.

SCOPE, stated so the two tickets do not blur: the GAP is the root and is #935's
-- it must not be able to leak whatever the value is. This closes one of the
two PRODUCERS that make it reachable (the other is the #928 refusal
re-prefill). Closing a producer does not close the gap; closing the gap makes
the producers harmless. Both are owed.

Hermetic: the real ``Req.truncate_prefix_to`` on a stand-in carrying only the
two fields it touches, plus an arithmetic model of the finished-req interval.
No CUDA, no pools.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import types
import unittest

from sglang.srt.managers.schedule_batch import Req
from sglang.test.test_utils import CustomTestCase

MATCHED = 9447
TOLD = 8192


def _req(prefix_len, protected):
    """Only the two fields the helper touches. The REAL method is bound off the
    class, so a change to it is caught here."""
    r = types.SimpleNamespace(
        prefix_indices=list(range(prefix_len)),
        cache_protected_len=protected,
    )
    r.truncate_prefix_to = types.MethodType(Req.truncate_prefix_to, r)
    return r


def _orphaned_rows(effective_cache_len, cache_protected_len):
    """The interval cache_finished_req leaves to nobody.

    Mirrors unified_radix_cache.py:1112-1116 exactly: the free starts at
    max(ecl, cpl) and the insert covers [0, ecl)."""
    free_start = max(effective_cache_len, cache_protected_len)
    return max(0, free_start - effective_cache_len)


class TruncationKeepsProtectedLen930(CustomTestCase):
    def test_truncation_lowers_the_protected_length_with_it(self):
        """RED BEFORE THE FIX: both branches sliced prefix_indices and left
        cache_protected_len at the pre-truncation value."""
        r = _req(MATCHED, MATCHED)

        r.truncate_prefix_to(TOLD)

        self.assertEqual(len(r.prefix_indices), TOLD)
        self.assertEqual(
            r.cache_protected_len,
            TOLD,
            "the request still claims more tree-owned rows than it holds",
        )

    def test_the_orphaned_interval_is_empty_after_truncation(self):
        """The consequence, as arithmetic rather than as an argument.

        A request truncated to `told` finishes with effective_cache_len == told
        in the ordinary case; the interval cache_finished_req abandons must be
        empty."""
        r = _req(MATCHED, MATCHED)
        stale = _orphaned_rows(TOLD, r.cache_protected_len)
        self.assertEqual(
            stale,
            MATCHED - TOLD,
            "the characterisation no longer models the leak",
        )

        r.truncate_prefix_to(TOLD)

        self.assertEqual(
            _orphaned_rows(TOLD, r.cache_protected_len),
            0,
            "rows are still left to nobody after the truncation",
        )


class TheHelperMayOnlyLower930(CustomTestCase):
    """The direction that must NOT move: raising the claim would invent
    protection the tree never granted, and that IS the dangerous direction for
    the duplicate free."""

    def test_a_lower_protected_length_is_left_alone(self):
        r = _req(MATCHED, 512)
        r.truncate_prefix_to(TOLD)
        self.assertEqual(r.cache_protected_len, 512)

    def test_a_told_at_or_above_the_match_changes_nothing(self):
        r = _req(TOLD, TOLD)
        r.truncate_prefix_to(MATCHED)
        self.assertEqual(len(r.prefix_indices), TOLD)
        self.assertEqual(r.cache_protected_len, TOLD)

    def test_truncating_to_zero_clears_the_claim(self):
        r = _req(MATCHED, MATCHED)
        r.truncate_prefix_to(0)
        self.assertEqual(len(r.prefix_indices), 0)
        self.assertEqual(r.cache_protected_len, 0)


class BothBranchesUseTheHelper930(CustomTestCase):
    """The two truncation sites are siblings of each other and drifted
    identically; one helper is what stops them drifting again."""

    def test_neither_branch_slices_prefix_indices_by_hand(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        self.assertNotIn("req.prefix_indices = req.prefix_indices[:told]", src)
        self.assertEqual(src.count("req.truncate_prefix_to(told)"), 2)


if __name__ == "__main__":
    unittest.main()
