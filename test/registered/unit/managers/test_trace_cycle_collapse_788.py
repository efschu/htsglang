# Copyright 2026 SGLang Team
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
"""#788: a per-pass instrument that has started repeating itself goes quiet.

THE SPECIMEN. Boot instr14 wrote 51212468 bytes in 9m59s, 13.03 MB/min at
its peak, and two per-pass instruments were 89% of that::

    141513 lines  #788 PP-ADMISSION verdict=...   (20367381 B)
    140184 lines  PARKED-DECODE carriers ...      (25373304 B)
      9600 lines  everything else                 ( 5471783 B)

#788 already suppresses the VACUOUS admission verdicts (d59537cd9d), but
that predicate only fires when nothing is queued, running or chunked. Under
burst load every pass has work on it, so every pass counted as informative
and printed -- thousands per second, saying the same handful of things.

WHY THE COLLAPSE IS A CYCLE DETECTOR AND NOT ``key == last_key``. Measured
on instr14, consecutive lines are almost never identical; the scheduler
cycles through a few states and the instrument reports each faithfully::

    PARKED-DECODE carriers 4 parked (+4 -2) of 4 resident; ...
    PARKED-DECODE carriers 2 parked (+2 -4) of 2 resident; ...
    PARKED-DECODE carriers 2 parked (+2 -2) of 2 resident; ...   <- period 3

Replaying instr14's 5 MB tail through a ``max_period=1`` collapser removes
nothing at all: 13545 parked receipts become 13542 plus 3 roll-ups. Admitting
period up to 3 turns the same tail into 39 lines plus 21 roll-ups.

WHAT THIS FILE PINS, and the third item is the one that matters most:

  (i)   a run of identical payloads collapses to ONE line plus a roll-up
        naming the suppressed COUNT;
  (ii)  a CHANGED payload -- anything the cycle does not already contain --
        prints immediately, and flushes the roll-up in front of itself;
  (iii) two ranks fed identical CONGRUENT payload sequences emit identical
        output even when their rank-local values differ. The #788 trace
        exists to prove rank congruence: the acceptance gate
        (evidence-665-f1/verdict_790.sh, step 3) diffs the emitted payloads
        across PP0/PP1/PP2 and requires one group per event. If rank 0
        suppressed a line rank 1 emitted, the gate would report a divergence
        that never happened;
  (iv)  a very long run still reports periodically, so silence is never
        ambiguous -- a reader can always separate "nothing changed" from
        "the instrument died".

ON THE SECOND EMITTER. ``PARKED-DECODE carriers`` is RANK-LOCAL by
construction: ``slot_pool`` comes from this rank's mamba allocator and
``running_bs`` from this rank's ``running_batch``. No acceptance gate diffs
it across ranks -- prove_park_677.sh only COUNTS the receipts and tails the
last twelve -- so it is collapsed on its own terms and must not be used as
congruence evidence. What is pinned for it here is determinism: the same
receipt sequence in, the same emission pattern out.

SINGLE PROCESS, PURE CPU, NO TORCH. Every object here is ordinary Python.
"""

import logging
import re
import types
import unittest

SCHEDULER_LOGGER = "sglang.srt.managers.scheduler"
PARKED_LOGGER = "sglang.srt.managers.parked_decode_set"


class _Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _capture(logger_name, fn):
    logger = logging.getLogger(logger_name)
    handler = _Grab()
    logger.addHandler(handler)
    prior = logger.level
    logger.setLevel(logging.INFO)
    try:
        fn()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior)
    unavailable = [m for m in handler.messages if "trace unavailable" in m]
    assert not unavailable, f"the instrument swallowed its own failure: {unavailable}"
    return handler.messages


def _verdicts(messages):
    return [m for m in messages if "PP-ADMISSION verdict=" in m]


def _admission_rollups(messages):
    return [m for m in messages if "PP-ADMISSION suppressed=" in m]


def _receipts(messages):
    return [m for m in messages if "PARKED-DECODE carriers" in m]


def _parked_rollups(messages):
    return [m for m in messages if "PARKED-DECODE suppressed=" in m]


class _Rank:
    """One PP rank's scheduler, reduced to exactly what the instrument reads.

    ``avail`` and ``evictable`` are settable per rank on purpose: they are the
    rank-local perturbation test (iii) uses to prove the collapse key does not
    consult them.
    """

    def __init__(self, avail=40830, evictable=130):
        from sglang.srt.managers.scheduler import Scheduler

        self.holder = types.SimpleNamespace(
            token_to_kv_pool_allocator=types.SimpleNamespace(
                available_size=lambda: avail
            ),
            tree_cache=types.SimpleNamespace(evictable_size=lambda: evictable),
            waiting_queue=[],
            running_batch=None,
            chunked_req=None,
        )
        self.holder._trace_pp_admission_verdict = types.MethodType(
            Scheduler._trace_pp_admission_verdict, self.holder
        )

    def pass_(self, *, n_reqs=0, queue=0, running=0, chunked=0, rid="r0", prefix=0):
        self.holder.waiting_queue = [object()] * queue
        self.holder.running_batch = (
            types.SimpleNamespace(reqs=[object()] * running) if running else None
        )
        self.holder.chunked_req = object() if chunked else None
        ret = None
        if n_reqs:
            ret = types.SimpleNamespace(
                reqs=[
                    types.SimpleNamespace(rid=rid, prefix_indices=[0] * prefix)
                    for _ in range(n_reqs)
                ]
            )
        self.holder._trace_pp_admission_verdict(ret)


class CycleCollapseUnit788(unittest.TestCase):
    """The predicate itself, away from either emitter."""

    def test_a_constant_stream_collapses_to_one_line(self):
        from sglang.srt.managers.log_cycle_collapse import CycleCollapse

        c = CycleCollapse(rollup_every=1000)
        verdicts = [c.observe("A") for _ in range(50)]
        self.assertEqual(sum(1 for v in verdicts if v.emit), 1)
        self.assertEqual(verdicts[0].period, 0)

    def test_the_measured_period_three_cycle_collapses(self):
        """instr14's actual parked-receipt shape: A B C A B C A B C ..."""
        from sglang.srt.managers.log_cycle_collapse import CycleCollapse

        c = CycleCollapse(rollup_every=1000)
        cycle = ["A", "B", "C"]
        emitted = sum(1 for i in range(300) if c.observe(cycle[i % 3]).emit)
        # 2p-1: every state is printed at least once (A B C), the repetition
        # is shown (A B), and the pass that COMPLETES the second cycle is the
        # first one suppressed -- it is also the first pass at which the
        # doubled window exists to be recognised.
        self.assertEqual(emitted, 5, "a period-3 cycle did not collapse")

    def test_a_break_in_the_cycle_prints_immediately_and_flushes(self):
        from sglang.srt.managers.log_cycle_collapse import CycleCollapse

        c = CycleCollapse(rollup_every=1000)
        for _ in range(40):
            c.observe("A")
        v = c.observe("B")
        self.assertTrue(v.emit, "a changed payload was swallowed")
        self.assertEqual(v.rollup, 39)
        self.assertEqual(v.period, 1)

    def test_a_long_run_reports_periodically(self):
        from sglang.srt.managers.log_cycle_collapse import CycleCollapse

        every = 16
        c = CycleCollapse(rollup_every=every)
        rollups = [c.observe("A").rollup for _ in range(2 * every + 1)]
        self.assertEqual([r for r in rollups if r], [every, every])

    def test_the_window_is_bounded(self):
        from sglang.srt.managers.log_cycle_collapse import CycleCollapse

        c = CycleCollapse(max_period=4)
        for i in range(10000):
            c.observe(f"k{i % 3}")
        self.assertLessEqual(len(c._history), 8)


class PPAdmissionCycleCollapse788(unittest.TestCase):
    def test_a_run_of_identical_payloads_collapses_to_one_line(self):
        """(i) The burst-load shape #788's vacuous predicate cannot reach:
        every pass has work on it, and every pass says the same thing."""
        rank = _Rank()

        def drive():
            for _ in range(500):
                rank.pass_(queue=1, running=2)

        messages = _capture(SCHEDULER_LOGGER, drive)
        self.assertEqual(
            len(_verdicts(messages)),
            1,
            f"500 identical DECLINEs produced {len(_verdicts(messages))} lines",
        )

    def test_the_flushed_rollup_names_the_count(self):
        """(i)+(ii) Silence must not be ambiguous: the reader has to see that
        N passes were swallowed, not that the instrument stopped."""
        rank = _Rank()

        def drive():
            for _ in range(10):
                rank.pass_(queue=1, running=2)
            rank.pass_(queue=3, running=2)

        messages = _capture(SCHEDULER_LOGGER, drive)
        rollups = _admission_rollups(messages)
        self.assertEqual(len(rollups), 1, f"no roll-up line: {messages}")
        self.assertIn("suppressed=9", rollups[0])
        self.assertLess(
            messages.index(rollups[0]),
            len(messages) - 1,
            "the roll-up must precede the line that flushed it",
        )

    def test_the_rollup_is_invisible_to_the_payload_gate(self):
        """verdict_790.sh step 3 collects payloads with `grep -o
        '#788 PP-ADMISSION verdict=.*'`. A roll-up that spelled that token
        would be grouped as if it were a payload and the congruence groups
        would stop meaning anything. It must still match the per-rank COUNT
        grep, which keys on '#788 PP-ADMISSION' alone."""
        rank = _Rank()

        def drive():
            for _ in range(10):
                rank.pass_(queue=1, running=2)
            rank.pass_(queue=3, running=2)

        rollups = _admission_rollups(_capture(SCHEDULER_LOGGER, drive))
        self.assertTrue(rollups)
        for line in rollups:
            self.assertNotIn("verdict=", line)
            self.assertIn("#788 PP-ADMISSION", line)

    def test_a_changed_payload_always_prints(self):
        """(ii) A DECLINE taken while the queue grew is the divergence signal
        this instrument exists to catch. It must never be swallowed."""
        rank = _Rank()

        def drive():
            for _ in range(50):
                rank.pass_(queue=1, running=2)
            rank.pass_(queue=1, running=4)
            rank.pass_(n_reqs=1, queue=1, running=4, rid="deadbeef")
            for _ in range(50):
                rank.pass_(queue=1, running=2)
            rank.pass_(queue=1, running=2, chunked=1)

        messages = _capture(SCHEDULER_LOGGER, drive)
        verdicts = _verdicts(messages)
        self.assertTrue(any("running=4" in v and "ADMIT" not in v for v in verdicts))
        self.assertTrue(any("rids=deadbeef" in v for v in verdicts))
        self.assertTrue(any("chunked=1" in v for v in verdicts))

    def test_a_chunked_prefill_walk_is_never_collapsed(self):
        """prefix_lens climbing 0, 512, 1024 ... is the chunked-prefill
        progress the gate reads. It never repeats, so it must never be
        suppressed."""
        rank = _Rank()

        def drive():
            for step in range(40):
                rank.pass_(n_reqs=1, queue=9, chunked=1, rid="abc", prefix=512 * step)

        verdicts = _verdicts(_capture(SCHEDULER_LOGGER, drive))
        self.assertEqual(len(verdicts), 40, "chunked-prefill progress was lost")

    def test_a_long_run_still_reports_periodically(self):
        """(iv) Under a sustained burst there is no changed payload to flush
        against, so without a periodic roll-up the log would be
        indistinguishable from a dead instrument."""
        from sglang.srt.managers.log_cycle_collapse import (
            CYCLE_COLLAPSE_ROLLUP_EVERY as every,
        )

        rank = _Rank()

        def drive():
            for _ in range(2 * every + 1):
                rank.pass_(queue=1, running=2)

        messages = _capture(SCHEDULER_LOGGER, drive)
        rollups = _admission_rollups(messages)
        self.assertEqual(len(rollups), 2, f"expected two roll-ups, got {len(rollups)}")
        for line in rollups:
            self.assertIn(f"suppressed={every}", line)

    def test_two_ranks_fed_identical_payloads_emit_identically(self):
        """(iii) THE PROPERTY THE GATE RESTS ON.

        The script below is the measured instr14 mix: a constant stretch, a
        period-2 alternation, a chunked-prefill walk, an admit. Two ranks see
        identical CONGRUENT fields and wildly different rank-local ones.
        """
        script = [{"queue": 1, "running": 2}] * 5
        script += [{"queue": 1, "running": r} for r in (2, 4) * 8]
        script += [
            {"n_reqs": 1, "queue": 9, "chunked": 1, "rid": "abc", "prefix": 512 * i}
            for i in range(4)
        ]
        script += [{"queue": 1, "running": 2}] * 5
        script += [{"n_reqs": 1, "queue": 2, "rid": "ffff"}]

        def emission_pattern(avail, evictable):
            rank = _Rank(avail=avail, evictable=evictable)

            def drive():
                for step in script:
                    rank.pass_(**step)

            messages = _capture(SCHEDULER_LOGGER, drive)
            # Blank the rank-local numbers: what must match is WHICH passes
            # spoke and what they said about the congruent fields.
            return [re.sub(r"(avail|evictable)=\d+", r"\1=X", m) for m in messages]

        pp0 = emission_pattern(avail=40830, evictable=130)
        pp1 = emission_pattern(avail=161378, evictable=138412)
        self.assertEqual(pp0, pp1)
        # And the pattern is the useful one, not "everything suppressed".
        self.assertGreaterEqual(
            sum(1 for m in pp0 if "verdict=" in m),
            7,
            f"the informative passes did not all speak: {pp0}",
        )


class ParkedDecodeCycleCollapse788(unittest.TestCase):
    """``PARKED-DECODE carriers``: 25373304 bytes of instr14, 49% of the log.

    RANK-LOCAL, see this module's header -- collapsed on its own terms.
    """

    @staticmethod
    def _set(**kwargs):
        from sglang.srt.managers.parked_decode_set import ParkedDecodeSet

        kwargs.setdefault("slot_pool", 24)
        kwargs.setdefault("max_running", 8)
        return ParkedDecodeSet(**kwargs)

    @staticmethod
    def _drive_measured_cycle(s, rounds):
        """The instr14 shape, reproduced: park 4, drop to 2, hold 2, repeat.

        Yields receipts '4 parked (+4 -2) of 4', '2 parked (+2 -4) of 2',
        '2 parked (+2 -2) of 2' -- a period-3 cycle, no two consecutive lines
        identical.
        """
        for _ in range(rounds):
            s.sync_carriers([f"c{i}" for i in range(4)], running_bs=4)
            s.sync_carriers(["d0", "d1"], running_bs=2)
            s.sync_carriers(["e0", "e1"], running_bs=2)

    def test_the_measured_cycle_collapses(self):
        s = self._set()
        messages = _capture(PARKED_LOGGER, lambda: self._drive_measured_cycle(s, 200))
        # 6, not 2p-1=5: the very first reconcile starts from an EMPTY set,
        # so it reads "(+4 -0)" where every later one reads "(+4 -2)". That
        # warm-up line is genuinely different and is printed; the cycle is
        # recognised one pass later than it would be from a cold identical
        # start. Printing a line too many is the right direction to err.
        self.assertEqual(
            len(_receipts(messages)),
            6,
            f"600 cycling receipts produced {len(_receipts(messages))} lines",
        )

    def test_a_run_of_identical_receipts_collapses_to_one_line(self):
        """(i) The degenerate period-1 case."""
        s = self._set()

        def drive():
            for _ in range(200):
                s.sync_carriers(["a", "b"], running_bs=2)
                s.sync_carriers([], running_bs=8)

        messages = _capture(PARKED_LOGGER, drive)
        self.assertEqual(len(_receipts(messages)), 3, f"{messages}")

    def test_a_changed_receipt_prints_immediately_with_a_rollup(self):
        """(ii) A new parked count is exactly what a reader is watching for."""
        s = self._set()

        def drive():
            self._drive_measured_cycle(s, 100)
            s.sync_carriers([f"z{i}" for i in range(7)], running_bs=7)

        messages = _capture(PARKED_LOGGER, drive)
        self.assertIn("7 parked", messages[-1])
        rollups = _parked_rollups(messages)
        self.assertEqual(len(rollups), 1, f"no roll-up flushed: {messages}")
        self.assertIn("suppressed=", rollups[0])
        self.assertLess(messages.index(rollups[0]), len(messages) - 1)

    def test_the_rollup_is_invisible_to_the_receipt_count_grep(self):
        """prove_park_677.sh counts lines matching 'PARKED-DECODE carriers'.
        A roll-up that spelled that would inflate the receipt count it is
        reporting a reduction in."""
        s = self._set()

        def drive():
            self._drive_measured_cycle(s, 100)
            s.sync_carriers([f"z{i}" for i in range(7)], running_bs=7)

        rollups = _parked_rollups(_capture(PARKED_LOGGER, drive))
        self.assertTrue(rollups)
        for line in rollups:
            self.assertNotIn("PARKED-DECODE carriers", line)

    def test_a_long_run_still_reports_periodically(self):
        """(iv)"""
        from sglang.srt.managers.log_cycle_collapse import (
            CYCLE_COLLAPSE_ROLLUP_EVERY as every,
        )

        s = self._set()

        def drive():
            for _ in range(2 * every + 4):
                s.sync_carriers(["a", "b"], running_bs=2)
                s.sync_carriers([], running_bs=8)

        rollups = _parked_rollups(_capture(PARKED_LOGGER, drive))
        self.assertGreaterEqual(len(rollups), 2, "a long run went silent")
        for line in rollups:
            self.assertIn(f"suppressed={every}", line)

    def test_the_last_receipt_field_is_never_suppressed(self):
        """The suppression is about the LOG, not the state. Callers that read
        ``last_receipt`` -- summaries, the crash path -- must still see the
        current one."""
        s = self._set()
        _capture(PARKED_LOGGER, lambda: self._drive_measured_cycle(s, 50))
        self.assertIn("2 parked", s.last_receipt)

    def test_two_instances_fed_identical_sequences_emit_identically(self):
        """(iii) for the rank-local emitter: determinism. The decision is a
        pure function of the receipt sequence, so it cannot depend on
        anything the sequence does not carry."""

        def pattern():
            s = self._set()

            def drive():
                self._drive_measured_cycle(s, 40)
                s.sync_carriers(["q"], running_bs=1)
                self._drive_measured_cycle(s, 40)

            return _capture(PARKED_LOGGER, drive)

        self.assertEqual(pattern(), pattern())


if __name__ == "__main__":
    unittest.main()
