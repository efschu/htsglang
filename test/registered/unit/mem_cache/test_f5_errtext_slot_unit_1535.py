"""1535 F5: the residency-cap clause of the under-delivery error names its unit.

NF K2 20:49:25 printed ``holding 336064 slot ids`` beside ``EFFECTIVE
max_total_num_tokens 262144``, and the number looked impossible. It is not:
``KvRowCap`` publishes ``len(withheld page ids) * page_size`` -- TOKEN slot ids,
over the allocator's whole id space (8192 pages x 64 = 524288 tokens, D.log
line 11256), of which 336064 = 5251 pages sat above the cap (page 2560 =
163840 tokens). The 262144 is the uneven-DCP projection, a different figure.

The clause now says which unit it counts and prints the id space, the cap and
the tokens still available under the cap. TEXT ONLY: the allocation decision,
the delivered count and the raise are unchanged (the old tests in
``test_residency_cap_eviction_790.py`` keep passing unmodified).

The double is the production paged allocator on the CPU with the real
``KvRowCap``, scaled 1:16 from the specimen (512 pages x 64 = 32768 tokens,
cap at page 160 = 10240 tokens).
"""

import logging
import unittest

import torch

from sglang.srt.managers.kv_backing_relief import KvRowCap
from sglang.srt.mem_cache import common as mc
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

PAGE = 64
NUM_PAGES = 512
CAP_PAGES = 160


def _capped_allocator():
    alloc = PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE,
        page_size=PAGE,
        dtype=torch.int64,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    cap = KvRowCap(alloc)
    cap.engage(CAP_PAGES)
    return alloc, cap


class _Tree:
    token_to_kv_pool_allocator = None

    def __init__(self, allocator):
        self.token_to_kv_pool_allocator = allocator

    def full_evictable_size(self):
        return 1000


class TheClauseNamesItsUnit(unittest.TestCase):
    def setUp(self):
        self.alloc, self.cap = _capped_allocator()
        self.msg = mc._eviction_shortfall_note(_Tree(self.alloc), 766, 64)

    def test_the_withheld_count_is_tokens_and_the_pages_behind_them(self):
        pages = self.cap.withheld
        self.assertEqual(pages, NUM_PAGES - CAP_PAGES)
        self.assertEqual(self.alloc.residency_withheld_slots, pages * PAGE)
        self.assertIn(f"holding {pages * PAGE} token slot ids", self.msg)
        self.assertIn(f"({pages} pages x page_size {PAGE};", self.msg)

    def test_it_says_these_are_not_references(self):
        self.assertIn("ids, not references or tree nodes", self.msg)

    def test_it_prints_the_id_space_the_number_lives_in(self):
        self.assertIn(
            f"id space {NUM_PAGES * PAGE} tokens = {NUM_PAGES} pages", self.msg
        )

    def test_it_prints_the_cap_and_what_is_available_under_it(self):
        self.assertIn(f"cap {CAP_PAGES * PAGE} tokens = page {CAP_PAGES}", self.msg)
        available = int(self.alloc.available_size())
        self.assertEqual(available, CAP_PAGES * PAGE)
        self.assertIn(f"available under the cap {available} tokens", self.msg)

    def test_the_old_diagnosis_is_kept(self):
        self.assertIn("EVICTION UNDER-DELIVERED", self.msg)
        self.assertIn("RESIDENCY CAP IS ENGAGED", self.msg)
        self.assertIn("pays the POOL nothing", self.msg)

    def test_no_cap_no_clause(self):
        self.cap.release()
        msg = mc._eviction_shortfall_note(_Tree(self.alloc), 766, 64)
        self.assertNotIn("RESIDENCY CAP", msg)

    def test_the_note_does_not_change_the_allocator(self):
        before = (int(self.alloc.available_size()), self.cap.withheld, self.cap.cap)
        mc._eviction_shortfall_note(_Tree(self.alloc), 766, 64)
        after = (int(self.alloc.available_size()), self.cap.withheld, self.cap.cap)
        self.assertEqual(before, after)

    def test_a_double_without_the_figures_prints_question_marks_not_zeros(self):
        class Bare:
            residency_withheld_slots = 640

        note = mc._residency_withheld_note(Bare())
        self.assertIn("holding 640 token slot ids", note)
        self.assertIn("id space ? tokens", note)
        self.assertIn("cap ? tokens = page ?", note)
        self.assertIn("available under the cap ? tokens", note)


class TheAdmissionGuardTextIsAccurate(unittest.TestCase):
    def test_the_empty_net_line_names_both_admission_paths(self):
        mc.clear_extend_relief_providers()
        try:
            with self.assertLogs(mc.logger, level=logging.WARNING) as logged:
                mc._attempt_extend_relief(766)
        finally:
            mc.clear_extend_relief_providers()
        text = "\n".join(logged.output)
        self.assertIn("chunk_tokens_the_pool_can_fund on the _rem_tokens <= 0", text)
        self.assertIn("PrefillAdder.rem_total_tokens", text)
        # nf-next-1006-13: the admission guard (nf-next-1006-01) subtracts the
        # evictable tokens above an engaged cap, Option C (nf-next-1006-07c) is
        # the alloc-site over-ask; the text must say both and must not claim
        # the old state ("does not subtract a residency cap") again.
        self.assertIn("subtracts the evictable tokens above an engaged residency cap", text)
        self.assertIn("counts evictable only below the cap", text)
        self.assertIn("num_tokens + gap", text)
        self.assertIn("deliverable_evictable_cap_aware_or", text)
        self.assertIn("_cap_overask", text)
        self.assertIn("not a provider", text)
        self.assertIn("That is not 'no net'", text)
        self.assertNotIn("does not subtract", text)
        self.assertNotIn("alloc-site net is empty", text)
        self.assertIn("NO relief provider is registered", text)
        self.assertIn("admission let through work the pool could not fund", text)


if __name__ == "__main__":
    unittest.main()
