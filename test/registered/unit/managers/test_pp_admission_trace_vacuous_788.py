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
"""#788: the admission trace must not drown its own evidence at idle.

THE SPECIMEN. Boot instr11 ran ``SGLANG_PP_ADMISSION_TRACE=1`` for about
three hours against an IDLE server and wrote a 5.9 GB log. The payload census
of the previous boot (instr10, 146023 trace lines per rank) shows what was in
it::

    86121 x  #788 PP-ADMISSION verdict=DECLINE n_reqs=0 rids= prefix_lens= \
avail=40830 evictable=130 queue=0 running=0 chunked=0
    60840 x  #788 PP-ADMISSION verdict=DECLINE n_reqs=0 ... \
queue=0 running=0 chunked=0

Every one of those lines says the same thing: nothing was queued, nothing was
running, nothing was chunked, nothing was admitted. A verdict taken over an
empty scheduler cannot show two ranks disagreeing, because there is nothing
for them to disagree about. The lines that CAN -- the ADMITs, and the
DECLINEs taken while work actually existed -- were a rounding error inside
the flood, and the flood is what forced the instrument off.

WHAT THIS FILE PINS, and the fourth item is the one that matters most:

  (i)   a VACUOUS verdict (n_reqs=0, queue=0, running=0, chunked=0) is
        suppressed;
  (ii)  an INFORMATIVE verdict is ALWAYS emitted -- an ADMIT, or a DECLINE
        taken with anything queued, running, or chunked;
  (iii) silence is never ambiguous: a roll-up line naming the suppressed
        COUNT is emitted, both periodically during a long idle stretch and
        as a flush in front of the next informative verdict, so a reader can
        always separate "nothing was happening" from "the instrument died";
  (iv)  the suppression predicate is a PURE FUNCTION OF THE CONGRUENT
        PAYLOAD FIELDS. The whole point of this instrument is that the
        acceptance gate (evidence-665-f1/verdict_790.sh, step 3) diffs the
        trace payloads across PP0/PP1/PP2 and requires one group per event.
        If rank 0 suppresses a line rank 1 emits, the diff shows a phantom
        divergence and the gate is worthless. So two "ranks" fed identical
        congruent fields must produce an identical emission pattern even
        when their rank-local values (avail, evictable) differ.
"""

import logging
import re
import types
import unittest

LOGGER_NAME = "sglang.srt.managers.scheduler"


class _Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _Rank:
    """One PP rank's scheduler, reduced to exactly what the instrument reads.

    `avail` and `evictable` are settable per rank on purpose: they are the
    rank-local perturbation that test (iv) uses to prove the suppression
    predicate does not consult them.
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

    def pass_(self, *, n_reqs=0, queue=0, running=0, chunked=0):
        """Run one admission pass with the given congruent payload fields."""
        self.holder.waiting_queue = [object()] * queue
        self.holder.running_batch = (
            types.SimpleNamespace(reqs=[object()] * running) if running else None
        )
        self.holder.chunked_req = object() if chunked else None
        ret = None
        if n_reqs:
            ret = types.SimpleNamespace(
                reqs=[
                    types.SimpleNamespace(rid=f"r{i}", prefix_indices=[])
                    for i in range(n_reqs)
                ]
            )
        self.holder._trace_pp_admission_verdict(ret)


def _capture(fn):
    logger = logging.getLogger(LOGGER_NAME)
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


def _rollups(messages):
    return [m for m in messages if "PP-ADMISSION suppressed=" in m]


class PPAdmissionTraceVacuous788(unittest.TestCase):
    def test_a_vacuous_verdict_is_suppressed(self):
        """The specimen line itself: DECLINE over an empty scheduler."""
        rank = _Rank()
        messages = _capture(lambda: rank.pass_())
        self.assertEqual(
            _verdicts(messages),
            [],
            f"the vacuous verdict was emitted: {messages}",
        )

    def test_a_long_idle_stretch_costs_a_bounded_number_of_lines(self):
        """Instr11's actual shape: thousands of consecutive idle passes. The
        flood, not the single line, is what took the instrument out."""
        rank = _Rank()

        def drive():
            for _ in range(4096):
                rank.pass_()

        messages = _capture(drive)
        self.assertEqual(_verdicts(messages), [], "idle passes still emit verdicts")
        self.assertLess(
            len(messages),
            64,
            f"4096 idle passes produced {len(messages)} lines",
        )

    def test_an_admit_is_always_emitted(self):
        rank = _Rank()
        messages = _capture(lambda: rank.pass_(n_reqs=2, queue=2))
        admits = [m for m in _verdicts(messages) if "verdict=ADMIT" in m]
        self.assertEqual(len(admits), 1, f"expected one ADMIT line: {messages}")
        self.assertIn("n_reqs=2", admits[0])

    def test_a_decline_with_work_present_is_always_emitted(self):
        """A DECLINE taken while requests were WAITING is the divergence
        signal this instrument exists to catch -- rank 0 declining a queue
        rank 1 admits. It must never be suppressed."""
        for kwargs in ({"queue": 1}, {"running": 1}, {"chunked": 1}):
            with self.subTest(**kwargs):
                rank = _Rank()
                messages = _capture(lambda: rank.pass_(**kwargs))
                declines = [m for m in _verdicts(messages) if "verdict=DECLINE" in m]
                self.assertEqual(
                    len(declines), 1, f"informative DECLINE lost: {messages}"
                )

    def test_an_informative_verdict_flushes_a_rollup_naming_the_count(self):
        """Silence must not be ambiguous. Three suppressed passes, then real
        work: the reader has to be able to see that three verdicts were
        swallowed and not that the instrument stopped."""
        rank = _Rank()

        def drive():
            for _ in range(3):
                rank.pass_()
            rank.pass_(n_reqs=1, queue=1)

        messages = _capture(drive)
        rollups = _rollups(messages)
        self.assertEqual(len(rollups), 1, f"no roll-up line: {messages}")
        self.assertIn("suppressed=3", rollups[0])
        self.assertLess(
            messages.index(rollups[0]),
            messages.index(_verdicts(messages)[0]),
            "the roll-up must precede the verdict it was flushed by",
        )

    def test_a_rollup_appears_periodically_without_any_work_at_all(self):
        """The idle server never produces an informative verdict to flush
        against. Without a periodic roll-up the log would be indistinguishable
        from a dead instrument for hours -- which is exactly the state boot
        instr11 could not rule out."""
        # Imported here rather than at module scope so the rest of this file
        # still reports real assertion failures against an unfixed tree.
        from sglang.srt.managers.pp_admission_congruence import (
            PP_ADMISSION_VACUOUS_ROLLUP_EVERY as every,
        )

        rank = _Rank()

        def drive():
            for _ in range(2 * every):
                rank.pass_()

        messages = _capture(drive)
        rollups = _rollups(messages)
        self.assertEqual(len(rollups), 2, f"expected two roll-ups: {messages}")
        for line in rollups:
            self.assertIn(f"suppressed={every}", line)

    def test_the_predicate_is_pure_in_the_congruent_fields(self):
        """THE PROPERTY THE GATE RESTS ON.

        Two ranks are fed an identical sequence of congruent payload fields
        while their rank-local values differ. Emission must be decided
        identically on both, or verdict_790.sh's payload diff reports a
        divergence that never happened.
        """
        script = [
            {},
            {},
            {},
            {"queue": 2},
            {"n_reqs": 1, "queue": 2},
            {},
            {},
            {"running": 1},
            {},
        ]

        def emission_pattern(avail, evictable):
            rank = _Rank(avail=avail, evictable=evictable)

            def drive():
                for step in script:
                    rank.pass_(**step)

            messages = _capture(drive)
            # Blank the rank-local numbers: what must match is WHICH passes
            # spoke and what they said about the congruent fields.
            return [re.sub(r"(avail|evictable)=\d+", r"\1=X", m) for m in messages]

        pp0 = emission_pattern(avail=40830, evictable=130)
        pp1 = emission_pattern(avail=161378, evictable=138412)
        self.assertEqual(pp0, pp1)
        # And the pattern is the useful one, not "everything suppressed".
        self.assertEqual(
            sum(1 for m in pp0 if "verdict=" in m),
            3,
            f"the three informative passes did not all speak: {pp0}",
        )


if __name__ == "__main__":
    unittest.main()
