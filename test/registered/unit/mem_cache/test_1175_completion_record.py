"""#1175 -- the producer half: a completion is RECORDED and readable, the
field named `synced` says whether it synced, and the census names its own
population.

THE SPECIMEN (boot_855_weg1b5). `HiCache prefetch success req=0c34259f
completed_local=12288 completed_synced=12288` printed on PP0 alone.
`completed_synced` is `min_completed_tokens`, overwritten from a packed MIN
all_reduce ONLY under `if self.tp_world_size > 1:` -- and `tp_world_size` is
the ATTENTION-TP cache group, which is 1 on this `--tp-size 1 --pp-size 3`
boot (`attn_reduce_world=1` on 307/307 #1028 lines). The field claimed an
agreement it never asked for: Instrument-Text-luegt Klasse A.

WHAT THIS PINS:

 (a) `completed_prefetch_tokens` returns None for a rid nobody has spoken
     about and an int for one that terminated -- None IS NOT ZERO;
 (b) the record is BOUNDED, so the instrument cannot become the leak
     (#1048 class);
 (c) an aborted request's record is dropped with the rest of its state;
 (d) `prefetch_progress_is_collective_free` reads `_attn_reduce_world` --
     the honest measure built by #1028 precisely so a world of 1 could not
     be read as an agreement;
 (e) the #1157 reaper docstring states WHERE IT CANNOT FIRE instead of
     leaving 0 lines to be read as "nothing was reaped";
 (f) E4b: the #939 double-prefill census line names the population it speaks
     for, because queue-occupant re-issues are never stamped and were
     therefore invisible (measured: two 13225-token re-admissions produced
     not one census line).

MUTANTS (each red): return 0 instead of None for an unknown rid; drop the
FIFO trim; print `synced=yes` unconditionally; drop the population field.
"""

import unittest

from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.producer_phase_census import DoublePrefillCensus


def _bare():
    """A cache shell with only the fields these readers touch. The class's
    __init__ needs a device; the readers do not."""
    c = urc.UnifiedRadixCache.__new__(urc.UnifiedRadixCache)
    c._prefetch_completed_tokens = {}
    c.ongoing_prefetch = {}
    return c


class TestSilenceAndZeroAreDifferentFacts(unittest.TestCase):
    def test_an_unknown_rid_reads_none_not_zero(self):
        c = _bare()
        self.assertIsNone(c.completed_prefetch_tokens("nobody"))

    def test_a_recorded_completion_reads_its_span(self):
        c = _bare()
        c._prefetch_completed_tokens["0c34259f"] = 12288
        self.assertEqual(c.completed_prefetch_tokens("0c34259f"), 12288)

    def test_a_registered_unterminated_prefetch_reads_ongoing(self):
        c = _bare()
        c.ongoing_prefetch["running"] = object()
        self.assertTrue(c.prefetch_is_ongoing("running"))
        self.assertFalse(c.prefetch_is_ongoing("other"))
        self.assertIsNone(
            c.completed_prefetch_tokens("running"),
            "running is not a completion of zero",
        )


class TestTheInstrumentIsBounded(unittest.TestCase):
    def test_the_record_dict_has_a_named_slot_bound(self):
        self.assertIsInstance(urc._PREFETCH_COMPLETION_SLOTS, int)
        self.assertGreater(urc._PREFETCH_COMPLETION_SLOTS, 0)

    def test_the_write_site_trims_oldest_first(self):
        import inspect

        src = inspect.getsource(urc.UnifiedRadixCache.check_prefetch_progress)
        self.assertIn("_PREFETCH_COMPLETION_SLOTS", src)
        self.assertIn("while len(self._prefetch_completed_tokens) >", src)
        self.assertIn("_prefetch_completed_tokens.pop(", src)

    def test_an_aborted_request_drops_its_record(self):
        import inspect

        src = inspect.getsource(urc.UnifiedRadixCache.release_aborted_request)
        self.assertIn("_prefetch_completed_tokens.pop(", src)


class TestTheFieldNamedSyncedSaysWhetherItSynced(unittest.TestCase):
    def test_the_success_line_prints_synced_and_the_reduce_world(self):
        import inspect

        src = inspect.getsource(urc.UnifiedRadixCache.check_prefetch_progress)
        self.assertIn("synced=%s", src)
        self.assertIn("attn_reduce_world=%d", src)
        self.assertIn('"yes" if _synced_world > 1 else "no"', src)

    def test_the_collective_free_predicate_reads_attn_reduce_world(self):
        import inspect

        src = inspect.getsource(
            urc.UnifiedRadixCache.prefetch_progress_is_collective_free
        )
        self.assertIn("_attn_reduce_world", src)


class TestTheReaperStatesWhereItCannotFire(unittest.TestCase):
    def test_the_comment_names_the_structural_silence(self):
        import pathlib

        text = pathlib.Path(urc.__file__).read_text()
        self.assertIn("#1175 (E2), WHERE THIS LINE CANNOT FIRE", text)
        self.assertIn("0 REAPED lines is", text)


class TestTheCensusNamesItsPopulation(unittest.TestCase):
    def test_the_line_declares_the_retract_closure_population(self):
        c = DoublePrefillCensus()
        fields = c.log_fields()
        self.assertEqual(fields["population"], "retract_closure_only")
        self.assertIn("population=retract_closure_only", c.format_line())

    def test_the_fence_term_is_still_the_tail(self):
        # #1068's pin: the fence term is appended last. The population clause
        # is inserted BEFORE it rather than after, so that contract holds.
        c = DoublePrefillCensus()
        self.assertTrue(c.format_line().endswith("fence_proceeds=0"))


if __name__ == "__main__":
    unittest.main()
