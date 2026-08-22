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
"""#796: the verdict line must report what was APPLIED, not what was proposed.

MEASURED DEFECT, boot_798_0822_0646.log, seam at 06:50:59Z. The group granted
76.1% and every rank converted it against its own cap:

    PP0 155853 instead of 204800 (released 1344 MiB)
    PP1  91954 instead of 120832 (released  702 MiB)
    PP2  96629 instead of 126976 (released  704 MiB)

In the SAME round, the verdict line's per-rank clause for PP1 read:

    deficit=-199 MiB -> no change (the cheaper tier covered the gap)

PP1 unmapped 702 MiB while its own diagnostic said it did nothing. The clause
came from ``last_proposal_summary()``, which is honest about itself -- its
docstring scopes it to "the caller that REFUSES" and it compares ``desire``
against ``current``, i.e. the PROPOSAL. Under #796 the proposal and the applied
action diverge by design: the group agrees a proportion, so a rank that asked
for nothing still pays its share. Reusing the refusal summary on the GRANTED
path therefore prints the one number that is guaranteed wrong.

This is not cosmetic. The line was written for #796 precisely so that a seam's
funding story could be read without re-deriving it, and within ten minutes of
its first boot it convinced its own author that unpressed ranks do not pay --
the exact opposite of what the mechanism does, and it took the applied
"KV-BACKING released" lines to overturn it. A diagnostic that misstates the
applied action is worse than no diagnostic, because it is trusted.

The fix shares ONE code path between the preview and the applier, so the line
cannot drift from the behaviour again. These tests pin that:

1. a satisfied rank (proposing the neutral element) still reports the SHRINK
   it is about to perform, and never the words "no change";
2. the reported row target equals what ``apply_shrink_ppm`` actually applies;
3. the declined path still reports the proposal terms, which is what that
   caller needs and what ``last_proposal_summary`` is for.

Hermetic: no CUDA, no scheduler, no distributed.
"""

from __future__ import annotations

import logging
import unittest

from sglang.srt.managers import kv_backing_relief as kbr
from sglang.srt.managers import phase_flip_spill as pfs

MIB = 1024 * 1024

#: PP1's metal shape: the rank whose deficit was already covered (-199 MiB).
CURRENT = 120_832
#: The proportion the PRESSED peer set on that seam.
GRANTED_PPM = 761_000
#: A floor far below the granted proportion, so the floor is not the binding
#: term and the shrink genuinely lands.
MY_FLOOR = 4_228


class _Sched:
    pass


class _Guard:
    floor_bytes = 1024 * MIB
    delta_bytes = 256 * MIB
    device_index = 0


class _StubRung:
    """A rank under NO pressure: it proposes the neutral element."""

    def __init__(self):
        self.applied = []

    def fundable_bytes(self):
        return (CURRENT - MY_FLOOR) * 1024

    def propose(self, **_kw):
        # Satisfied: desire == current, which _shrink_ppm maps to the neutral
        # 1000000 -- it cannot lower the group's MIN.
        return (
            kbr._shrink_ppm(CURRENT, CURRENT),
            -kbr._floor_ppm(MY_FLOOR, CURRENT),
            CURRENT,
            -CURRENT,
        )

    def cap_proposal(self):
        return kbr.CAP_ABSTAIN

    def backed_rows(self):
        return CURRENT

    def normalize_free_lists(self):
        return None

    def last_proposal_summary(self):
        # Verbatim shape of the misleading clause, so the test fails loudly if
        # the granted path ever falls back to it again.
        return (
            f"KV rung: current={CURRENT} rows, floor={MY_FLOOR}, "
            "deficit=-199 MiB -> no change (the cheaper tier covered the gap)"
        )

    def apply_target(self, target):
        self.applied.append(int(target))
        return 0

    def apply_shrink_ppm(self, ppm):
        current, target = self.preview_shrink_ppm(ppm)
        if target >= current:
            return 0
        return self.apply_target(target)

    # Internals the REAL preview/explain need, so the test exercises the
    # shipped formatting rather than a stub's paraphrase of it.
    _bytes_per_row = 1024

    def _current_rows(self):
        return CURRENT

    def _max_live_row(self):
        return 131

    def _evict_floor_rows(self, _max_live):
        return MY_FLOOR, 0

    preview_shrink_ppm = kbr.KvBackingRelief.preview_shrink_ppm
    explain_shrink_ppm = kbr.KvBackingRelief.explain_shrink_ppm


class _Capture:
    def __init__(self):
        self.records = []

    def __enter__(self):
        self._logger = logging.getLogger(pfs.__name__)
        self._prev = self._logger.level
        self._logger.setLevel(logging.INFO)
        self._handler = logging.Handler()
        self._handler.emit = lambda r: self.records.append(r.getMessage())
        self._logger.addHandler(self._handler)
        return self.records

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._prev)
        return False


def _reduce_with_pressed_peer(ppm):
    """Element-wise MIN against one peer that IS under pressure."""

    def reduce(vals, **_kw):
        out = list(vals)
        out[0] = min(int(out[0]), int(ppm))
        return out

    return reduce


def _run(reduce):
    sched = _Sched()
    rung = _StubRung()
    setattr(sched, pfs.KV_BACKING_RELIEF_ATTR, rung)
    with _Capture() as records:
        pfs.collective_kv_backing_relief(
            sched,
            reduce,
            want_bytes=0,
            guard=_Guard(),
            direction="tp_to_pp",
        )
    verdict = [r for r in records if "KV shrink verdict" in r]
    return rung, verdict


class TestVerdictReportsApplied(unittest.TestCase):
    def test_satisfied_rank_does_not_claim_no_change_while_shrinking(self):
        """The measured defect, stated as a test."""
        rung, verdict = _run(_reduce_with_pressed_peer(GRANTED_PPM))
        self.assertEqual(len(verdict), 1, "the verdict line must still be emitted")
        line = verdict[0]
        expected = max(MY_FLOOR, kbr._rows_for_ppm(GRANTED_PPM, CURRENT))
        self.assertEqual(
            rung.applied,
            [expected],
            "precondition: this rank really does shrink despite having no deficit",
        )
        self.assertNotIn(
            "no change",
            line,
            "the rank unmapped pages; the line must not say it did nothing",
        )

    def test_verdict_names_the_applied_row_target(self):
        rung, verdict = _run(_reduce_with_pressed_peer(GRANTED_PPM))
        expected = max(MY_FLOOR, kbr._rows_for_ppm(GRANTED_PPM, CURRENT))
        self.assertIn(
            str(expected),
            verdict[0],
            "the reader must be able to see the row count that was applied",
        )

    def test_preview_agrees_with_the_applier(self):
        """One code path, so the line cannot drift from the behaviour."""
        rung = _StubRung()
        _current, previewed = rung.preview_shrink_ppm(GRANTED_PPM)
        rung.apply_shrink_ppm(GRANTED_PPM)
        self.assertEqual([previewed], rung.applied)

    def test_declined_path_still_reports_the_proposal(self):
        """``last_proposal_summary`` remains correct for the refusing caller."""

        def no_shrink(vals, **_kw):
            return list(vals)

        rung, verdict = _run(no_shrink)
        self.assertEqual(rung.applied, [], "nothing may be applied when declined")
        self.assertEqual(len(verdict), 1)


if __name__ == "__main__":
    unittest.main()
