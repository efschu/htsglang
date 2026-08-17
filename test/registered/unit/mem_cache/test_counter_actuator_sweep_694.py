# SPDX-License-Identifier: Apache-2.0
"""#694 remainder -- the counter-vs-actuator sweep, and its one unreachable find.

THE FAMILY SIGNATURE, as the four fixed members established it:

    a relief/eviction/recovery path COUNTS capacity, a SEPARATE ACTUATOR
    delivers it, the actuator delivers LESS than the count, and the caller
    TREATS THE COUNT AS A PROMISE -- so it raises or wedges instead of
    degrading or retrying.

    #679  alloc retry                 remedy: retry = identical call
    #681  paged twin / third root     staged frees invisible to available_size
    #684  recover-over-ceiling
    #715  evictable-not-deliverable   remedy: flush staged frees, then retry,
                                      BEFORE the relief ladder

SWEEP RESULT: **no REACHABLE fifth member.** Everything live in a default or
commonly-enabled config either traces to one of the four, or re-checks the real
post-delivery state instead of trusting a count -- `corridor_guard._spend_ladder`
re-probes `free_bytes()` after every provider call rather than trusting its
return, `pin_ledger.reserve` and `lru_file_evictor` re-probe real free space
citing "the #715 lesson" by name, `schedule_batch` discards
`evict_from_tree_cache()`'s return and gates on a fresh `decode_mem_avail()`,
and the radix family computes its count and its delivery in the SAME call.

ONE UNREACHABLE SUSPECT, pinned here rather than fixed, per the #487 rule
(reachable+wrong = fix; unreachable = pin so it is found by search, not by a
crash after someone enables the flag).
"""

import ast
import inspect
import unittest

from sglang.srt.mem_cache import multi_ended_allocator as MEA


def _fn_src(name: str, cls: str = "MultiEndedAllocator") -> str:
    tree = ast.parse(inspect.getsource(MEA))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return ast.get_source_segment(inspect.getsource(MEA), f) or ""
    raise AssertionError(f"{cls}.{name} not found")


class TestTheSuspectIsUnreachable(unittest.TestCase):
    """#487 pattern. The moment this stops being true, the pin below stops
    being a pin and becomes a bug report."""

    def test_unified_memory_is_off_by_default(self):
        from sglang.srt.server_args import ServerArgs

        self.assertFalse(
            ServerArgs.enable_unified_memory,
            "--enable-unified-memory is now on by default, which makes the "
            "counter-vs-actuator divergence pinned in this file REACHABLE. It "
            "must be fixed (family discipline: re-check after delivery, or "
            "degrade instead of asserting) before the default flips.",
        )

    def test_it_is_still_refused_alongside_spec_and_PD(self):
        """Its blast radius stays bounded by these refusals; if they are
        relaxed, the suspect's reach grows with them."""
        import inspect as _i

        from sglang.srt import server_args

        src = " ".join(_i.getsource(server_args).split())
        self.assertIn("--enable-unified-memory is not yet compatible with", src)


class TestTheSuspectShape(unittest.TestCase):
    """The divergence itself, pinned as SHAPE rather than as a number.

    COUNT:    ``UnifiedSWATokenToKVPoolAllocator.available_size()`` solves a
              closed-form joint budget over BOTH sub-pools at once.
    ACTUATOR: ``MultiEndedAllocator._alloc_bind_fast_or_slow`` runs
              SEQUENTIALLY -- the full-attention side binds first and consumes
              its share of the shared byte gap, and only then does the SWA side
              bind against whatever remains, using a virtual-page snapshot
              taken BEFORE the first bind ran.

    A joint closed form and an order-dependent two-step delivery are not the
    same structure, which is the family's precondition for divergence.
    """

    def test_the_actuator_HAS_a_graceful_failure_path(self):
        """`alloc` restores its virtual ids and returns None on shortfall --
        so a non-crashing answer already exists at this site."""
        src = _fn_src("alloc")
        self.assertIn("if phys_pages is None:", src)
        self.assertIn("return None", src)
        self.assertIn("free_virtual_ids = torch.cat", src)

    def test_but_the_composite_TREATS_THE_COUNT_AS_A_PROMISE(self):
        """The signature, in the code's own words: the assert's message says it
        is trusting the counter -- "the composite's byte-budget check should
        have caught this"."""
        src = _fn_src("alloc_with_virtual")
        self.assertIn("assert phys_pages is not None", src)
        self.assertIn("byte-budget check should have caught this", src)
        self.assertNotIn(
            "return None",
            src,
            "alloc_with_virtual has no graceful-shortfall path at all: it "
            "always asserts, so a count that over-promises crashes here",
        )

    def test_the_composite_converts_a_graceful_None_into_an_assert(self):
        """The other half: even where the actuator DOES degrade, the composite
        wraps the safe return in `assert ... is not None`, so a designed
        degradation is turned back into a crash."""
        whole = inspect.getsource(MEA)
        self.assertIn("internal-state inconsistency", whole)
        self.assertGreaterEqual(
            whole.count("internal-state inconsistency"),
            3,
            "the alloc / alloc_extend / alloc_decode composites each assert on "
            "the pre-check having been right",
        )

    def test_the_count_and_the_delivery_are_different_structures(self):
        """Pins WHY this is a suspect rather than provably safe: if these two
        ever became one call, the divergence would be structurally impossible
        and this whole entry could be closed."""
        whole = inspect.getsource(MEA)
        self.assertIn("def available_size", whole)
        self.assertIn("_alloc_bind_fast_or_slow", whole)
        avail = _fn_src("available_size", cls="UnifiedSWATokenToKVPoolAllocator")
        self.assertNotIn(
            "_alloc_bind_fast_or_slow",
            avail,
            "available_size now calls the actuator, so count and delivery share "
            "a path -- re-classify this entry as provably safe and close it",
        )


class TestTheFamilyRecordStaysFindable(unittest.TestCase):
    """Step 4's purpose: a fifth member must be found by search, not by crash."""

    def test_the_design_note_carries_the_register(self):
        import os

        here = os.path.dirname(__file__)
        doc = os.path.join(
            here, "..", "..", "..", "..", "docs", "dev",
            "DESIGN_679_admission_relief_ladder.md",
        )
        with open(doc) as fh:
            text = fh.read()
        self.assertIn("counter-vs-actuator", text.lower())
        self.assertIn("multi_ended_allocator", text)


if __name__ == "__main__":
    unittest.main()
