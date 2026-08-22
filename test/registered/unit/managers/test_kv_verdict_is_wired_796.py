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
"""#796: the shrink verdict must actually REACH the log, from the real gate.

A fix wired into nothing is the defect class this repo keeps finding, and a
diagnostic wired into nothing is the same defect wearing a friendlier face --
it is indistinguishable from a healthy mechanism, which is precisely the
confusion #796 exists to end. The helper existing and being unit-tested proves
nothing about whether a boot will carry the line.

So these tests call the SHIPPED entry point,
``phase_flip_spill.collective_kv_backing_relief``, with a peer injected into
the reduction, and assert on captured log records:

1. the group's verdict is emitted, naming the peer floor as the binding term;
2. THIS rank's own proposal terms ride the same line -- the whole point being
   that a rank which fits must now report the floor that vetoed its peer;
3. it is emitted on the rank that FITS, not only on a rank that refuses.

Hermetic: no CUDA, no scheduler, no distributed. The rung is a stub carrying
exactly the methods the gate calls, so the test pins the CALL SITE rather than
the rung's arithmetic (which its own tests cover).
"""

from __future__ import annotations

import logging
import unittest

from sglang.srt.managers import kv_backing_relief as kbr
from sglang.srt.managers import phase_flip_spill as pfs

MIB = 1024 * 1024

CURRENT = 204_800
MY_FLOOR = 115_681
MY_DESIRE = 149_126
#: A peer whose floor sits at its own cap -- the metal shape of PP2, whose
#: ``fundable_bytes()`` was 0 on every ask.
PEER_FLOOR = 204_800


class _Sched:
    pass


class _Guard:
    floor_bytes = 1024 * MIB
    delta_bytes = 256 * MIB
    device_index = 0


class _StubRung:
    """Exactly the surface ``collective_kv_backing_relief`` calls."""

    def __init__(self, summary: str):
        self._summary = summary
        self.applied = []

    def fundable_bytes(self):
        return (CURRENT - MY_FLOOR) * 1024

    def propose(self, **_kw):
        # #796: proportions of this rank's own cap, as the real propose emits.
        return (
            kbr._shrink_ppm(MY_DESIRE, CURRENT),
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
        return self._summary

    def apply_target(self, target):
        self.applied.append(int(target))
        return 0

    def apply_shrink_ppm(self, ppm):
        # #796: the gate now hands down a PROPORTION and each rank converts it
        # against its own cap. The stub records the row target that produces.
        return self.apply_target(kbr._rows_for_ppm(int(ppm), CURRENT))


def _reduce_against_peer(peer_floor):
    """Element-wise MIN against one peer, which is what the real channel does.

    Only the floor field is moved: the peer agrees about the cap and asks for
    no shrink of its own, which is exactly the rank this ticket is about -- one
    under no memory pressure at all.
    """

    def reduce(vals, **_kw):
        out = list(vals)
        # field 1 is ``-floor``; MIN over negated floors yields the MAX floor.
        out[1] = min(int(out[1]), -kbr._floor_ppm(int(peer_floor), CURRENT))
        return out

    return reduce


def _run(summary="KV rung: current=204800 rows, floor=115681", peer=PEER_FLOOR):
    sched = _Sched()
    rung = _StubRung(summary)
    setattr(sched, pfs.KV_BACKING_RELIEF_ATTR, rung)
    with self_capture() as records:
        freed = pfs.collective_kv_backing_relief(
            sched,
            _reduce_against_peer(peer),
            want_bytes=500 * MIB,
            guard=_Guard(),
            direction="tp_to_pp",
        )
    return freed, rung, records


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


def self_capture():
    return _Capture()


class TestTheVerdictIsWired(unittest.TestCase):
    def test_the_gate_emits_a_verdict_line(self):
        _freed, _rung, records = _run()
        line = [r for r in records if "KV shrink verdict" in r]
        self.assertTrue(
            line,
            "the shipped gate must emit the verdict; a diagnostic that only "
            f"exists in its own unit test is not wired. Got: {records!r}",
        )

    def test_the_verdict_names_the_binding_term(self):
        """Whatever the outcome, the line must carry the deciding quantity."""
        _freed, _rung, records = _run()
        line = "\n".join(r for r in records if "KV shrink verdict" in r)
        self.assertTrue(
            "GRANTED" in line or "DECLINED" in line,
            f"the verdict must state itself; got {line!r}",
        )
        self.assertIn("%", line, "#796: the currency is a proportion")

    def test_this_ranks_own_terms_ride_the_same_line(self):
        """The rank that FITS is the one whose floor is the veto, so it reports."""
        _freed, _rung, records = _run(summary="MY-OWN-TERMS-MARKER")
        line = "\n".join(r for r in records if "KV shrink verdict" in r)
        self.assertIn("MY-OWN-TERMS-MARKER", line)

    def test_a_peer_at_its_own_cap_still_declines(self):
        """A peer whose floor IS its cap genuinely cannot shrink, and must not be
        unmapped through. #796 removed the CURRENCY defect, not this safety law."""
        freed, rung, _records = _run()
        self.assertEqual(rung.applied, [])
        self.assertEqual(freed, 0)

    def test_a_lower_peer_floor_funds_the_pressed_rank(self):
        """The can-fail proof: same call, peer floor lowered, shrink happens.

        Without this the veto test above would pass just as well against a
        mechanism that never shrinks under any circumstances.
        """
        _freed, rung, records = _run(peer=100_000)
        self.assertEqual(
            rung.applied,
            [MY_DESIRE],
            "with no peer veto the group must reach apply_target",
        )
        line = "\n".join(r for r in records if "KV shrink verdict" in r)
        self.assertIn("GRANTED", line)


if __name__ == "__main__":
    unittest.main()
