"""#1302: the decode split survives the LAG between the replay and the read.

Boot ``weg2dec2c`` (21c46b1876) proved #1241b's identities hold and then
reported the split as unavailable on **13,893 of 13,950 rounds** with reason
``graph-replay-nodes-overwritten`` (+57 ``graph-replay-nodes-unread``). The
instrument was correct; the READ lost the race.

THE ROOT, IN ONE SENTENCE: the generation is bumped on the HOST timeline (at
the launch site, before ``backend.replay``), while the timestamps in the
nodes are destroyed on the DEVICE timeline (when that replay EXECUTES). Under
the overlap scheduler the host runs a batch ahead, so by the time the
scheduler flush reaches round N the next replay has been *issued* -- and a
refusal keyed on "issued" throws away a reading that is still perfectly
valid, because the device has not reached that replay yet.

Two things follow, and this file is the red-first form of both:

1. **The predicate must be the device's, not the host's.** A replay that has
   been issued but not executed has destroyed nothing. The witness is an
   EAGER fence event recorded on the launch stream immediately before the
   replay is issued: a COMPLETED fence of generation G+1 proves the device
   has reached that launch point, and only then are generation G's nodes
   gone.
2. **A reading must be able to outlive the round it was taken in.** When the
   flush finds a round not yet readable, the graph it declared is still
   valid AT THAT MOMENT and will not be after the next replay executes. So
   the reading is taken then and kept in a per-key ring, and the round is
   emitted from the ring one or more rounds later.

Hermetic: the fake device/capture/replay layer of
``test_decode_graph_event_nodes_1241b`` is reused verbatim rather than
re-modelled, so the two files cannot drift apart in what they claim CUDA
does. Run with ``CUDA_VISIBLE_DEVICES=''``.

Named red-first: every test name states the wrong behaviour it exists to
catch, and the file as a whole is the mutant proof for the shipped
single-reading form -- restore the launch-generation check and the first
three tests go red.
"""

from __future__ import annotations

import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sglang.srt.debug_utils.rank_phase_summary import (
    parse_rank_batch_line,
    parse_unsplit_line,
)
from test_decode_graph_event_nodes_1241b import Harness, _Capture

#: The refusal token every overwritten round still starts with. The LAG is
#: appended (``...-by-4``) so a log line can never say "overwritten" without
#: saying how far behind the reader was -- the future check of #1302.
OVERWRITTEN = "graph-replay-nodes-overwritten"


class ReplayLagTest(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness()
        self.cap = _Capture()
        self.logger = logging.getLogger(
            "sglang.srt.managers.scheduler_components.decode_round_log"
        )
        self.logger.addHandler(self.cap)
        self.logger.setLevel(logging.INFO)
        self.addCleanup(self.logger.removeHandler, self.cap)

    # -- helpers ---------------------------------------------------------

    def lines(self):
        return [ln for ln in self.cap.lines if ln.startswith("Decode rank batch")]

    def parse(self, line):
        """Split line or unsplit line, whichever this one is."""
        split = parse_rank_batch_line("[2026-09-09 00:00:00 TP1] " + line)
        if split is not None:
            return split, None
        return None, parse_unsplit_line("[2026-09-09 00:00:00 TP1] " + line)

    def run_and_execute_round(self, round_id, key, per_region_ms, tail_ms=6.0):
        """One graphed round the fake DEVICE also executes immediately."""
        self.h.log.begin_round(round_id=round_id, bs=6, rows=24)
        with self.h.log.segment("decode", graphed=True):
            self.h.clock.note_graph_replay(key)
            self.h.replay(key, per_region_ms)
            self.h.state.advance(tail_ms)

    # -- the tests -------------------------------------------------------

    def test_a_round_read_after_the_next_replay_was_only_ISSUED_is_still_split(self):
        """THE 13,893. The next replay is ISSUED -- the host bumped the
        generation and handed the graph to the driver -- but the device has
        not reached it, so round 1's timestamps are still sitting in the
        nodes, untouched. Refusing here throws away a valid reading and
        reports ``split unavailable`` on a round that could be split.

        MUTANT: compare ``nodes.generation`` to the round's declaration (the
        shipped form). This test is exactly that mutant's red."""
        self.h.capture("k8", ["tp.all_reduce"])
        self.run_and_execute_round(1, "k8", [1.0])

        # The device has caught up to the end of round 1 and no further.
        horizon = self.h.state.now
        self.h.state.readable_from = horizon
        self.h.state.advance(1.0)

        # THE FORWARD THREAD, one batch ahead: round 2's replay is issued.
        # Nothing of it has executed -- no ``replay`` call follows.
        self.h.clock.note_graph_replay("k8")

        self.h.log.end_round()

        got = self.lines()
        self.assertEqual(len(got), 1, got)
        split, un = self.parse(got[0])
        self.assertIsNotNone(
            split,
            "round 1 was refused although the replay that would overwrite it "
            f"has not executed: {un!r}",
        )
        self.assertAlmostEqual(split["wait_ms"], 1.0, places=1)
        self.assertAlmostEqual(split["gpu_ms"], 7.0, places=1)
        self.assertAlmostEqual(split["compute_ms"], 6.0, places=1)
        self.assertEqual(self.h.state.synchronize_calls, 0)

    def test_a_reading_taken_while_valid_survives_the_replay_that_overwrites_it(self):
        """THE RING. The flush finds round 1 unreadable (its bracket end has
        not completed) while the graph it declared IS complete. That is the
        last instant the reading exists; taking it then and keeping it is
        the only thing that lets round 1 be emitted with a split at all.

        MUTANT: keep no ring (read only inside ``harvest_round``). Round 1
        is then refused and round 2's 9.0 ms is the only wait the boot ever
        sees -- a sample biased toward whichever rounds happened to win."""
        self.h.capture("k8", ["tp.all_reduce"])

        self.h.log.begin_round(round_id=1, bs=6, rows=24)
        with self.h.log.segment("decode", graphed=True):
            self.h.clock.note_graph_replay("k8")
            self.h.replay("k8", [1.0])
            nodes_done = self.h.state.now
            self.h.state.advance(6.0)

        # The device has executed the graph but not yet the bracket's end.
        self.h.state.readable_from = nodes_done

        # Round 2 both issues AND executes: generation 1's nodes are gone
        # from here on, and only a reading taken above can still report it.
        self.run_and_execute_round(2, "k8", [9.0])
        self.h.ready_now()
        self.h.log.end_round()

        got = self.lines()
        self.assertEqual(len(got), 2, got)
        first, un = self.parse(got[0])
        self.assertIsNotNone(first, f"the ring did not keep round 1's reading: {un!r}")
        self.assertAlmostEqual(first["wait_ms"], 1.0, places=1)
        self.assertAlmostEqual(first["gpu_ms"], 7.0, places=1)
        second, _ = self.parse(got[1])
        self.assertIsNotNone(second, got[1])
        self.assertAlmostEqual(second["wait_ms"], 9.0, places=1)
        self.assertEqual(self.h.state.synchronize_calls, 0)

    def test_a_reader_four_rounds_late_names_the_lag_and_never_prints_a_zero(self):
        """THE FUTURE CHECK. A reading that is genuinely gone must still say
        HOW FAR behind the reader was -- a bare ``overwritten`` reads as a
        property of the mechanism, and a wait of 0.0 would read as a
        measurement. Four replays execute over round 1's nodes while nothing
        is readable, so no ring entry can exist for it."""
        self.h.capture("k8", ["tp.all_reduce"])
        self.run_and_execute_round(1, "k8", [1.0])

        # The device has reached NOTHING: no flush below can take a reading.
        self.h.state.readable_from = -1.0
        for gen in range(2, 6):
            self.run_and_execute_round(gen, "k8", [float(gen)])
        self.h.ready_now()
        self.h.log.end_round()

        got = self.lines()
        self.assertEqual(len(got), 5, got)
        for line, lag in zip(got[:4], (4, 3, 2, 1)):
            split, un = self.parse(line)
            self.assertIsNone(split, line)
            self.assertIsNotNone(un, line)
            self.assertEqual(un["reason"], f"{OVERWRITTEN}-by-{lag}", line)
            self.assertFalse(un["split_known"])
            self.assertNotIn("wait 0.0", line)
        last, _ = self.parse(got[4])
        self.assertIsNotNone(last, got[4])
        self.assertAlmostEqual(last["wait_ms"], 5.0, places=1)
        self.assertEqual(self.h.state.synchronize_calls, 0)

    def test_a_replay_landing_inside_the_RING_read_is_discarded_not_stored(self):
        """The snapshot read is no more atomic than the round read was. A
        replay that lands between two pairs of the ring's own read loop would
        put a MIXTURE of two replays into the ring, where it would then be
        served to a round as a measurement -- the quietest possible form of
        this defect, because the ring makes it look deliberate.

        MUTANT: check the fence only BEFORE the snapshot read."""
        self.h.capture("k8", ["tp.all_reduce", "dcp.all_gather"])

        self.h.log.begin_round(round_id=1, bs=6, rows=24)
        with self.h.log.segment("decode", graphed=True):
            self.h.clock.note_graph_replay("k8")
            self.h.replay("k8", [1.0, 2.0])
            nodes_done = self.h.state.now
            self.h.state.advance(6.0)
        self.h.state.readable_from = nodes_done

        def land_a_replay(n):
            # The ring's read queries pair 1's post, then pair 2's post.
            # Fire between them: pair 1 has been read, pair 2 has not.
            if n != 2:
                return
            self.h.state.on_query = None
            self.h.state.readable_from = float("inf")
            self.h.clock.note_graph_replay("k8")
            self.h.replay("k8", [50.0, 60.0])

        self.h.state.on_query = land_a_replay
        self.h.log.begin_round(round_id=2, bs=6, rows=24)
        self.h.state.on_query = None
        self.h.ready_now()
        self.h.log.end_round()

        got = self.lines()
        self.assertGreaterEqual(len(got), 1, got)
        split, un = self.parse(got[0])
        self.assertIsNone(
            split,
            "a mixture of two replays was stored in the ring and served as "
            f"round 1's split: {split!r}",
        )
        self.assertIsNotNone(un, got[0])
        self.assertTrue(un["reason"].startswith(OVERWRITTEN), got[0])
        self.assertEqual(self.h.state.synchronize_calls, 0)

    def test_the_two_identities_hold_over_a_mixed_sequence(self):
        """The gates the boot reads, as tests. ``split + withheld == rounds``
        and ``graphed_split == graphed_rounds - (no-nodes + overwritten +
        unread + twice)``. Both are partitions: a round that falls out of
        both sides is a round the log stopped accounting for."""
        self.h.capture("k8", ["tp.all_reduce"])

        # (i) a round whose reading survives, (ii) a round with no nodes at
        # all, (iii) a round read four replays late.
        self.run_and_execute_round(1, "k8", [1.0])
        self.h.log.begin_round(round_id=2, bs=1, rows=1)
        with self.h.log.segment("decode", graphed=True):
            self.h.clock.note_graph_replay("never-captured")
            self.h.state.advance(3.0)
        self.h.state.readable_from = -1.0
        for gen in range(3, 6):
            self.run_and_execute_round(gen, "k8", [float(gen)])
        self.h.ready_now()
        self.h.log.end_round()

        got = self.lines()
        rounds = len(got)
        self.assertEqual(rounds, 5, got)
        split_lines = [ln for ln in got if "split unavailable" not in ln]
        withheld = [ln for ln in got if "split unavailable" in ln]
        self.assertEqual(len(split_lines) + len(withheld), rounds)

        reasons = [self.parse(ln)[1]["reason"] for ln in withheld]
        no_nodes = sum(1 for r in reasons if r == "graph-replay-no-event-nodes")
        overwritten = sum(1 for r in reasons if r.startswith(OVERWRITTEN))
        unread = sum(1 for r in reasons if r == "graph-replay-nodes-unread")
        twice = sum(1 for r in reasons if r == "graph-replay-key-replayed-twice")
        self.assertEqual(
            len(split_lines),
            rounds - (no_nodes + overwritten + unread + twice),
            f"the withheld reasons do not partition the graphed rounds: {reasons!r}",
        )
        self.assertEqual(no_nodes, 1, reasons)
        self.assertGreaterEqual(len(split_lines), 1, got)

    def test_the_overhead_line_labels_the_ring_and_the_round_denominators_apart(self):
        """The boot record of dec2c read ``196 overwritten`` off this line as
        NODES and set it against 1552 -- two orders of magnitude out. The
        counter is FORWARDS, and the line is emitted ONCE after the first
        ``OVERHEAD_ROUNDS`` rounds, so it is a running total as of that
        round and not a boot total. Both facts belong on the line itself,
        next to a ring depth that is a third denominator again."""
        self.h.capture("k8", ["tp.all_reduce"])
        self.h.log.OVERHEAD_ROUNDS = 2
        for gen in range(1, 5):
            self.run_and_execute_round(gen, "k8", [1.0])
        self.h.ready_now()
        self.h.log.end_round()

        over = [
            ln for ln in self.cap.lines if ln.startswith("Decode rank clock overhead")
        ]
        self.assertEqual(len(over), 1, over)
        self.assertIn("FORWARDS", over[0])
        self.assertIn("as of this line", over[0])
        self.assertIn("Reading ring (#1302)", over[0])
        self.assertIn("depth", over[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
