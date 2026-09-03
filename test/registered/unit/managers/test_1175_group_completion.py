"""#1175 (E1) -- the admission fact is the GROUP's, formed without a collective.

THE SPECIMEN (boot_855_weg1b5_cd5bb69607_0903_115008, rid 0c34259f, log
107778/113727-113737/113834). After the generation-4 `tp_to_pp` cutover all
three ranks issued the same re-admission prefetch (`#1028B FETCH CAP ...
keys=13224` byte-identical). PP0's completed in 3 s and it ADMITted
`prefix_lens=12288` on `completed_synced=12288` -- a field whose MIN
all_reduce is gated on `if self.tp_world_size > 1:` and therefore never ran on
this `--tp-size 1 --pp-size 3` boot (`attn_reduce_world=1` on 307/307 #1028
lines). PP1 then died 14 s later on `#968 PREFIX MATERIALISATION SHORTFALL`:
the designed group STOP for detected rank divergence.

WHAT THIS PINS:

 (a) `group_completion_verdict` is PURE and treats SILENCE as absent, never
     as zero -- a peer that has not reported contributes NO number to the
     floor, and the verdict defers instead of admitting;
 (b) the bound-expiry outcome is a CLAMP to the group floor, never a raise
     and never `want` -- the #631 bulletin's own prescription;
 (c) `want <= 0` admits unconditionally, so a boot with no storage span is
     byte-identical to the pre-#1175 path;
 (d) `format_group_fact` names EVERY peer including the silent one -- PP2
     printed nothing at all about this rid, and a silent follower must be
     visible AS silent at the decider (#1153 no-statement form);
 (e) the carrier is the #791 ring lap: a follower stamps its own readings
     onto the home-bound payload, a relay unions them, PP0 absorbs them, and
     a lap that would otherwise be void still carries them.

MUTANTS (each red): treat a missing peer as 0 in the floor; admit at `want`
on expiry instead of clamping; drop the early-return guard so a void lap
loses the reports; let a relay overwrite another rank's entry.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.pp_prefetch_completion import (
    PENDING,
    format_group_fact,
    group_completion_enabled,
    group_completion_verdict,
)


class TestTheVerdictIsPureAndSilenceIsNotZero(unittest.TestCase):
    def test_a_silent_peer_defers_instead_of_flooring_to_zero(self):
        # PP1 and PP2 said nothing at all about this rid.
        v = group_completion_verdict({}, "0c34259f", 12288, pp_size=3)
        self.assertFalse(v.admit)
        self.assertIsNone(v.floor, "silence must contribute NO number")
        self.assertEqual(v.missing, (1, 2))
        self.assertEqual(v.reason, "peer_report_absent")

    def test_a_pending_peer_defers_and_is_named_pending(self):
        table = {("r", 1): 12288, ("r", 2): PENDING}
        v = group_completion_verdict(table, "r", 12288, pp_size=3)
        self.assertFalse(v.admit)
        self.assertEqual(v.pending, (2,))
        self.assertEqual(v.missing, ())
        self.assertEqual(v.reason, "peer_prefetch_pending")

    def test_a_covering_group_floor_admits(self):
        table = {("r", 1): 12288, ("r", 2): 13000}
        v = group_completion_verdict(table, "r", 12288, pp_size=3)
        self.assertTrue(v.admit)
        self.assertEqual(v.floor, 12288)
        self.assertIsNone(v.clamp_to)
        self.assertEqual(v.reason, "group_floor_covers")

    def test_a_short_peer_defers_and_is_named_with_its_number(self):
        table = {("r", 1): 4096, ("r", 2): 12288}
        v = group_completion_verdict(table, "r", 12288, pp_size=3)
        self.assertFalse(v.admit)
        self.assertEqual(v.floor, 4096)
        self.assertEqual(v.short, ((1, 4096),))
        self.assertEqual(v.reason, "peer_coverage_short")

    def test_no_store_span_admits_byte_identically_to_the_old_path(self):
        v = group_completion_verdict({}, "r", 0, pp_size=3)
        self.assertTrue(v.admit)
        self.assertEqual(v.reason, "no_store_span")
        self.assertIsNone(v.clamp_to)

    def test_a_single_rank_group_has_no_peers_and_admits(self):
        v = group_completion_verdict({}, "r", 12288, pp_size=1)
        self.assertTrue(v.admit)
        self.assertEqual(v.reason, "no_peers")


class TestTheExpiredBoundClampsInsteadOfRaisingOrOverTelling(unittest.TestCase):
    def test_expiry_clamps_to_the_group_floor(self):
        table = {("r", 1): 4096, ("r", 2): 8192}
        v = group_completion_verdict(
            table, "r", 12288, pp_size=3, deadline_expired=True
        )
        self.assertTrue(v.admit)
        self.assertEqual(v.clamp_to, 4096, "told must be <= every rank's coverage")
        self.assertEqual(v.reason, "bound_expired_clamped_to_group_floor")

    def test_expiry_with_no_number_at_all_clamps_to_zero_not_to_want(self):
        v = group_completion_verdict({}, "r", 12288, pp_size=3, deadline_expired=True)
        self.assertTrue(v.admit)
        self.assertEqual(v.clamp_to, 0)
        self.assertNotEqual(v.clamp_to, 12288)


class TestEveryPeerIsNamedIncludingTheSilentOne(unittest.TestCase):
    def test_the_group_fact_prints_absent_pending_and_numbers(self):
        table = {("0c34259f", 2): PENDING}
        v = group_completion_verdict(table, "0c34259f", 12288, pp_size=3)
        line = format_group_fact("0c34259f", 12288, v)
        self.assertIn("r0=12288", line)
        self.assertIn("r1=absent", line, "PP1 said nothing and must show as absent")
        self.assertIn("r2=pending", line)
        self.assertIn("reason=", line)

    def test_a_covering_peer_is_printed_too_not_only_the_failing_ones(self):
        table = {("r", 1): 12288, ("r", 2): 4096}
        v = group_completion_verdict(table, "r", 12288, pp_size=3)
        line = format_group_fact("r", 12288, v)
        self.assertIn("r1=12288", line)
        self.assertIn("r2=4096", line)


class TestTheKillSwitchShipsOn(unittest.TestCase):
    def test_default_is_on_and_zero_restores_the_old_path(self):
        import os

        prior = os.environ.pop("SGLANG_PP_GROUP_COMPLETION", None)
        try:
            self.assertTrue(group_completion_enabled())
            os.environ["SGLANG_PP_GROUP_COMPLETION"] = "0"
            self.assertFalse(group_completion_enabled())
            os.environ["SGLANG_PP_GROUP_COMPLETION"] = "1"
            self.assertTrue(group_completion_enabled())
        finally:
            os.environ.pop("SGLANG_PP_GROUP_COMPLETION", None)
            if prior is not None:
                os.environ["SGLANG_PP_GROUP_COMPLETION"] = prior


class _FakeTree:
    def __init__(self, completed=None, ongoing=()):
        self._completed = dict(completed or {})
        self._ongoing = set(ongoing)

    def completed_prefetch_tokens(self, rid):
        return self._completed.get(str(rid))

    def prefetch_is_ongoing(self, rid):
        return str(rid) in self._ongoing


def _holder(rank, queue_rids, tree):
    return SimpleNamespace(
        ps=SimpleNamespace(pp_rank=rank, pp_size=3),
        tree_cache=tree,
        waiting_queue=[SimpleNamespace(rid=r) for r in queue_rids],
    )


class TestTheFactRidesTheRingLap(unittest.TestCase):
    def test_pp0_is_a_consumer_not_a_producer(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_prefetch_completion_own

        h = _holder(0, ["r"], _FakeTree({"r": 12288}))
        self.assertEqual(pp_prefetch_completion_own(h), ())

    def test_a_follower_reports_int_pending_and_omits_silence(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_prefetch_completion_own

        tree = _FakeTree({"done": 12288}, ongoing=["running"])
        h = _holder(1, ["done", "running", "silent"], tree)
        own = pp_prefetch_completion_own(h)
        self.assertIn(("done", 12288, 1), own)
        self.assertIn(("running", PENDING, 1), own)
        self.assertNotIn("silent", [rid for rid, _c, _r in own])

    def test_a_relay_unions_and_never_overwrites_another_rank(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_prefetch_completion_facts_from_wire,
            pp_prefetch_completion_stamp,
        )

        incoming = {"__pp_prefetch_completion__": (("r", 4096, 1),)}
        h = _holder(2, ["r"], _FakeTree({"r": 12288}))
        out = {}
        pp_prefetch_completion_stamp(h, incoming, out)
        facts = set(pp_prefetch_completion_facts_from_wire(out))
        self.assertIn(("r", 4096, 1), facts, "PP1's report must be relayed")
        self.assertIn(("r", 12288, 2), facts, "PP2 adds its own")

    def test_pp0_absorbs_and_the_table_keys_by_rid_and_rank(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_note_prefetch_completion,
            pp_prefetch_completion_table,
        )

        h = _holder(0, [], _FakeTree())
        n = pp_note_prefetch_completion(
            h, {"__pp_prefetch_completion__": (("r", 4096, 1), ("r", PENDING, 2))}
        )
        self.assertEqual(n, 2)
        table = pp_prefetch_completion_table(h)
        self.assertEqual(table[("r", 1)], 4096)
        self.assertEqual(table[("r", 2)], PENDING)

    def test_a_message_without_the_key_leaves_the_table_alone(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_note_prefetch_completion,
            pp_prefetch_completion_table,
        )

        h = _holder(0, [], _FakeTree())
        pp_note_prefetch_completion(
            h, {"__pp_prefetch_completion__": (("r", 4096, 1),)}
        )
        pp_note_prefetch_completion(h, {"something_else": 1})
        self.assertEqual(pp_prefetch_completion_table(h)[("r", 1)], 4096)

    def test_a_void_lap_still_carries_the_reports(self):
        # pp_output_payload_with_return_trip returns the payload UNCHANGED when
        # there is no decision and no chain. Before #1175 that early return
        # would have dropped the completion reports on exactly the laps the
        # decider is waiting for.
        import inspect

        from sglang.srt.managers import scheduler_pp_mixin as m

        src = inspect.getsource(m.pp_output_payload_with_return_trip)
        self.assertIn("completion_out", src)
        self.assertIn("not completion_out", src)


if __name__ == "__main__":
    unittest.main()
