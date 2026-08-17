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
"""#679 rung 1-3: the admission relief ladder, built to DESIGN_679.

WHAT THIS IS FOR. #679 shipped a PARK: a chunked prefill that the pool cannot
fund is not scheduled, so the instance degrades instead of dying. A park is not
free -- it is a request that made no progress this round -- and the design note
costed the four reliefs that could have been spent instead:

    rung 0  radix eviction        already spent by the caller. Baseline.
    rung 1  kvso.try_spill        bounded, chosen, costs no request's progress
    rung 2  throttle              frees NOTHING now; stops rung 3 repeating
    rung 3  retract_decode        most tokens, loudest -- the victim re-prefills
    rung 4  PARK                  the floor, never the replacement

THREE PROPERTIES ARE LOAD-BEARING AND EACH HAS ITS OWN CLASS BELOW.

1. THE PARK STAYS FINAL. The ladder changes how much the pool can fund; it
   never admits and never refuses. If it ever gained an admit path there would
   be two admission authorities and DESIGN_679 rule 1 would be false.

2. EXHAUSTION IS AN OUTCOME, NOT AN ERROR. ``try_spill`` returns False when no
   host region is free -- a reachable state whose bound has never been measured
   under the 5-lane load that produced the crash. The ladder falls through to
   the next rung. Nothing here raises, ever: a relief bug must not become an
   instance death, which is the failure mode this whole ticket is about.

3. EVERY DECISION IS GROUP-UNIFORM. Rungs mutate the batch. A rung entered on a
   rank-local size splits the group -- the binding rank retracts, its peers do
   not, the batches diverge, and the ranks stop agreeing on which collectives
   run. That is a hang. #603 paid for the decision, #583 paid for the loop
   BOUND (ranks entered together and popped DIFFERENT numbers of victims), and
   rung 3 needs both.

OFF BY DEFAULT. Without SGLANG_ADMISSION_RELIEF_LADDER the ladder returns 0
before touching anything, and admission behaves exactly as c4b88e1923 -- the
boot currently serving.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler

CHUNK = 512


class _Env:
    def __init__(self, **env):
        self.env = {k: (None if v is None else str(v)) for k, v in env.items()}

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.env}
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


class _Batch:
    def __init__(self, empty=False, bs=4):
        self._empty = empty
        self._bs = bs
        self.uniform_avail_floor = None

    def is_empty(self):
        return self._empty

    def batch_size(self):
        return self._bs


class _Kvso:
    """The spill rung. ``regions`` is the host-region supply -- the bound the
    design note flags as unmeasured."""

    def __init__(self, regions=1, frees=0):
        self.regions = regions
        self.frees = frees
        self.calls = []

    def try_spill(self, batch, need=None):
        self.calls.append(need)
        if self.regions <= 0:
            return False  # exhausted: an OUTCOME, not an error
        self.regions -= 1
        return True


def _sched(
    *,
    avail,
    kvso=None,
    chunked=True,
    running=None,
    retract_gain=0,
    avail_after=None,
):
    """A Scheduler stub carrying only what the ladder touches.

    Built with __new__ and pinned field by field, the idiom this suite already
    uses -- and the reason the repaired stubs in #679 needed repairing is that
    such a stub silently rots. Every field here is one the ladder reads.
    """
    s = Scheduler.__new__(Scheduler)
    # available_size the GROUP agreed on. A list is walked so a test can make
    # the reduced value change as rungs free tokens.
    seq = list(avail_after) if avail_after is not None else None
    s._avail_seq = seq
    s._avail = avail

    def _uniform_min_avail():
        if s._avail_seq:
            return s._avail_seq.pop(0)
        return s._avail

    s.uniform_min_avail = _uniform_min_avail
    # A RANK-LOCAL availability that DISAGREES with the group value, so a rung
    # that read the wrong one produces a different verdict instead of an
    # AttributeError. Catching the mutation by accident is not catching it.
    s.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 10 * CHUNK)
    s.kv_session_offload = kvso
    s.admission_limiter = None
    s.server_args = SimpleNamespace(chunked_prefill_size=CHUNK)
    s.chunked_req = object() if chunked else None
    s.retract_calls = []

    def _retract(batch, *, kv_full_retract_flag):
        s.retract_calls.append((batch, kv_full_retract_flag))
        return retract_gain

    s._retract_decode_and_requeue = _retract
    return s


class TheLadderIsOffByDefaultTest(unittest.TestCase):
    """Guard (c): default behaviour must be byte-identical to c4b88e1923."""

    def test_unset_means_no_rung_is_entered(self):
        kvso = _Kvso()
        s = _sched(avail=0, kvso=kvso)
        with _Env(
            SGLANG_ADMISSION_RELIEF_LADDER=None, SGLANG_ADMISSION_RELIEF_RETRACT=None
        ):
            freed = s._maybe_spend_admission_relief(_Batch())
        self.assertEqual(freed, 0)
        self.assertEqual(kvso.calls, [], "the spill rung must not be reached")
        self.assertEqual(s.retract_calls, [], "the retract rung must not be reached")

    def test_rung_3_needs_its_OWN_flag_as_well(self):
        """Retraction destroys progress, so it is opt-in separately from the
        rungs that only cost bandwidth and latency."""
        s = _sched(avail=0, kvso=_Kvso(regions=0))
        with _Env(
            SGLANG_ADMISSION_RELIEF_LADDER=1, SGLANG_ADMISSION_RELIEF_RETRACT=None
        ):
            s._maybe_spend_admission_relief(_Batch())
        self.assertEqual(
            s.retract_calls, [], "rung 3 ran without being asked for separately"
        )

    def test_rung_3_cannot_run_while_the_ladder_is_off(self):
        s = _sched(avail=0, kvso=None)
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=0, SGLANG_ADMISSION_RELIEF_RETRACT=1):
            s._maybe_spend_admission_relief(_Batch())
        self.assertEqual(s.retract_calls, [])


class TheTriggerSpendsNothingWhenComfortableTest(unittest.TestCase):
    def test_a_pool_that_can_fund_the_chunk_enters_no_rung(self):
        """The common case on a healthy instance: one reduced read, one
        comparison, nothing spent."""
        kvso = _Kvso()
        s = _sched(avail=CHUNK, kvso=kvso)
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1, SGLANG_ADMISSION_RELIEF_RETRACT=1):
            self.assertEqual(s._maybe_spend_admission_relief(_Batch()), 0)
        self.assertEqual(kvso.calls, [])
        self.assertEqual(s.retract_calls, [])

    def test_the_shortfall_asked_for_is_chunk_minus_available(self):
        """Sized from the REDUCED value, so every rank asks its rungs for the
        same number of tokens. Sizing it locally is #583 one layer up."""
        kvso = _Kvso()
        s = _sched(avail=100, kvso=kvso)
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1):
            s._maybe_spend_admission_relief(_Batch())
        self.assertEqual(kvso.calls, [CHUNK - 100])

    def test_no_chunked_request_means_nothing_to_relieve_for(self):
        kvso = _Kvso()
        s = _sched(avail=0, kvso=kvso, chunked=False)
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1):
            self.assertEqual(s._maybe_spend_admission_relief(_Batch()), 0)
        self.assertEqual(kvso.calls, [])


class TheDecisionsAreGroupUniformTest(unittest.TestCase):
    """Guard (b) of the build brief, and the constraint the whole ladder is
    shaped by: a rung entered on a RANK-LOCAL size splits the group -- the
    binding rank relieves, its peers do not, the batches diverge, and the ranks
    stop agreeing on which collectives run. That is a hang.

    The stub deliberately carries a rank-local availability that DISAGREES with
    the reduced one (10x the chunk, i.e. comfortable) while the group value is
    starved. A ladder reading the local number would conclude there is nothing
    to do and spend no rung at all.
    """

    def test_a_starved_GROUP_spends_rungs_even_when_this_rank_looks_fine(self):
        kvso = _Kvso(regions=1)
        s = _sched(avail=0, kvso=kvso)
        self.assertGreater(
            s.token_to_kv_pool_allocator.available_size(),
            CHUNK,
            "fixture precondition: this rank must look comfortable locally",
        )
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1, SGLANG_ADMISSION_RELIEF_RETRACT=1):
            s._maybe_spend_admission_relief(_Batch())
        self.assertEqual(
            kvso.calls,
            [CHUNK],
            "the ladder read this rank's own pool instead of the group's, so "
            "a rank whose peers are starved would spend nothing while they do",
        )

    def test_the_shortfall_is_sized_from_the_GROUP_value(self):
        """#583 one layer up: ranks that size the ask differently retract
        different numbers of victims."""
        kvso = _Kvso(regions=1)
        s = _sched(avail=200, kvso=kvso)
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1):
            s._maybe_spend_admission_relief(_Batch())
        self.assertEqual(kvso.calls, [CHUNK - 200])


class TheRungsRunInOrderTest(unittest.TestCase):
    def test_a_successful_spill_stops_the_ladder_before_retraction(self):
        """Rung 1 costs no request's progress, so a ladder that reached rung 3
        anyway would be discarding a session it did not need to."""
        kvso = _Kvso(regions=1)
        # available climbs past the need after the spill.
        s = _sched(
            avail=0,
            kvso=kvso,
            avail_after=[0, 0, CHUNK, CHUNK, CHUNK],
        )
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1, SGLANG_ADMISSION_RELIEF_RETRACT=1):
            s._maybe_spend_admission_relief(_Batch())
        self.assertEqual(len(kvso.calls), 1, "rung 1 ran")
        self.assertEqual(s.retract_calls, [], "rung 3 must not run after a spill")

    def test_an_EXHAUSTED_spill_falls_through_and_is_not_an_error(self):
        """Guard (a). No host region free is a reachable state whose bound is
        unmeasured on this rig -- the ladder continues to the next rung."""
        kvso = _Kvso(regions=0)
        s = _sched(avail=0, kvso=kvso, retract_gain=900)
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1, SGLANG_ADMISSION_RELIEF_RETRACT=1):
            freed = s._maybe_spend_admission_relief(_Batch())
        self.assertEqual(len(kvso.calls), 1, "rung 1 was asked and declined")
        self.assertEqual(len(s.retract_calls), 1, "rung 3 must pick it up")
        self.assertIsInstance(freed, int)

    def test_no_kvso_at_all_still_reaches_retraction(self):
        s = _sched(avail=0, kvso=None)
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1, SGLANG_ADMISSION_RELIEF_RETRACT=1):
            s._maybe_spend_admission_relief(_Batch())
        self.assertEqual(len(s.retract_calls), 1)

    def test_an_empty_running_batch_has_nothing_to_take(self):
        """Every rung acts on the RUNNING batch. Empty means the pressure is
        not coming from resident work."""
        kvso = _Kvso()
        s = _sched(avail=0, kvso=kvso)
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1, SGLANG_ADMISSION_RELIEF_RETRACT=1):
            self.assertEqual(s._maybe_spend_admission_relief(_Batch(empty=True)), 0)
        self.assertEqual(kvso.calls, [])
        self.assertEqual(s.retract_calls, [])


class Rung3CarriesItsPreconditionTest(unittest.TestCase):
    """#583: the loop bound must be the reduced value too, not just the entry
    decision. Ranks that enter together and pop different numbers of victims
    diverge exactly as surely as ranks that enter differently."""

    def test_uniform_avail_floor_is_set_before_retracting(self):
        s = _sched(avail=0, kvso=None, retract_gain=10)
        batch = _Batch()
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1, SGLANG_ADMISSION_RELIEF_RETRACT=1):
            s._maybe_spend_admission_relief(batch)
        self.assertIsNotNone(
            batch.uniform_avail_floor,
            "retract_decode's loop bound would read a rank-local value",
        )
        self.assertEqual(batch.uniform_avail_floor, 0)

    def test_the_retraction_is_flagged_as_a_real_pool_shortage(self):
        s = _sched(avail=0, kvso=None)
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1, SGLANG_ADMISSION_RELIEF_RETRACT=1):
            s._maybe_spend_admission_relief(_Batch())
        self.assertEqual(s.retract_calls[0][1], True)


class NoRungMayRaiseTest(unittest.TestCase):
    """A relief bug must not become an instance death. That is the entire
    subject of #679, and it applies to the relief as much as to the alloc."""

    def test_a_spill_that_raises_is_survived_and_the_ladder_continues(self):
        class _Boom(_Kvso):
            def try_spill(self, batch, need=None):
                raise RuntimeError("spill exploded")

        s = _sched(avail=0, kvso=_Boom(), retract_gain=5)
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1, SGLANG_ADMISSION_RELIEF_RETRACT=1):
            freed = s._maybe_spend_admission_relief(_Batch())
        self.assertIsInstance(freed, int)
        self.assertEqual(len(s.retract_calls), 1, "the next rung still ran")

    def test_a_retraction_that_raises_is_survived(self):
        s = _sched(avail=0, kvso=None)

        def _boom(batch, *, kv_full_retract_flag):
            raise RuntimeError("retract exploded")

        s._retract_decode_and_requeue = _boom
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1, SGLANG_ADMISSION_RELIEF_RETRACT=1):
            self.assertIsInstance(s._maybe_spend_admission_relief(_Batch()), int)

    def test_a_broken_trigger_lets_admission_proceed_unaided(self):
        s = _sched(avail=0, kvso=None)

        def _boom():
            raise RuntimeError("reduce unavailable")

        s.uniform_min_avail = _boom
        with _Env(SGLANG_ADMISSION_RELIEF_LADDER=1):
            self.assertEqual(s._maybe_spend_admission_relief(_Batch()), 0)


class TheParkStaysFinalTest(unittest.TestCase):
    """DESIGN_679 rule 1, pinned on the source: the ladder may change what
    there is to decide from, and may not decide."""

    def test_the_ladder_never_admits(self):
        import inspect

        src = inspect.getsource(Scheduler._admission_relief_ladder)
        for forbidden in ("can_run_list", "add_chunked_req", "set_extend_range"):
            self.assertNotIn(
                forbidden,
                src,
                f"the ladder touches {forbidden}: it has become a second "
                "admission authority and the park guard is no longer final",
            )

    def test_the_ladder_runs_BEFORE_the_admission_decision(self):
        """Read the frame that actually holds the call.

        ``get_new_batch_prefill`` delegates to ``_get_new_batch_prefill_raw``,
        so a pin reading only the outer frame sees neither the ladder nor the
        adder and would pass vacuously -- the same stale-pin failure repaired
        in the PP loop tests, caught here on the first run of this file.
        """
        import inspect

        src = inspect.getsource(
            Scheduler._get_new_batch_prefill_raw
        ) + inspect.getsource(Scheduler.get_new_batch_prefill)
        self.assertIn("_maybe_spend_admission_relief", src)
        self.assertLess(
            src.index("_maybe_spend_admission_relief"),
            src.index("adder.add_chunked_req"),
            "relief after the decision is relief that changed nothing",
        )

    def test_rung_3_reaches_the_SHARED_actuator_not_a_copy(self):
        """A second retraction implementation that forgot to requeue its
        victims would leak every one of them -- worse than the crash."""
        import inspect

        src = inspect.getsource(Scheduler._admission_relief_ladder)
        self.assertIn("_retract_decode_and_requeue", src)
        shared = inspect.getsource(Scheduler._retract_decode_and_requeue)
        self.assertIn("_add_request_to_queue(req, is_retracted=True)", shared)


if __name__ == "__main__":
    unittest.main()
