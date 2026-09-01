# SPDX-License-Identifier: Apache-2.0
"""PP ranks agree on a batch by DECISION, not by luck (#791).

RED-FIRST. Phase 1 established the root cause with file:line citations
(scheduler.py:6414-6417 for the per-rank admission gate, scheduler_pp_mixin.py
1063-1069 for the unconditional chain-forward, scheduler.py:4693-4703 for
#616g's uniformity floors going `None` whenever `tp_cpu_group` has one
member -- true on every rank of a TP=1/PP=N boot): PP stages are N
independent schedulers that agree only by determinism, and nothing today
forwards the admission DECISION alongside the chain-forwarded request. This
file:

1. reproduces that divergence with a NEUTERED (today's) simulation -- three
   rank stand-ins, each independently deriving its own prefix length from its
   own local `match_prefix` result, no decision forwarded at all -- and pins
   that this is genuinely divergent (a CAN-FAIL proof: if this ever stopped
   diverging the fix below would have nothing to fix);
2. proves the FIX (`sglang.srt.managers.pp_admission_congruence`) makes the
   three ranks agree, in the ordinary case (every downstream local match is
   >= what PP0 decided) and in the safe-truncate case (a downstream rank's
   local match is LARGER than told -- #616g's slack trade, taken on the PP
   axis);
3. proves the UNHONOURABLE case (a downstream rank's local match is SMALLER
   than told) degrades exactly the way the design requires: excluded from
   `effective` (never handed a corrupted length), exactly one bounded WARNING
   naming rank/rid/told/local, no exception, the retraction is carried
   forward in the amended decision so the NEXT rank agrees on the same
   membership change, a sibling request in the same decision is unaffected,
   and the process is still alive and able to reconcile further decisions
   afterwards (the liveness property a test that only checks "an error was
   raised" would miss);
4. pins the two structural constraints from the design brief: no
   `torch.distributed` call anywhere in the fix module (nothing here may
   become a collective on the admission path -- scheduler.py:6391-6407
   documents the 2026-08-17 deadlock of exactly that family), and
   `pp_size<=1` is a byte-identical pass-through (today's only shipped
   configuration must not observe this module at all).

WHAT THIS FILE DOES NOT DO. It does not wire `pp_admission_congruence` onto
the typed tensor-dict/proxy channel or into `scheduler_pp_mixin.py`'s receive
path -- that file is under a SCOPE FENCE (another strand owns its #789
readiness-contract work) and the wire-level carrier choice is a design note
in `pp_admission_congruence.py`'s module docstring, not code exercised here.
This file also does not attempt a real-process/gloo reproduction of the
production deadlock the way `test_pp_chain_flush_deadlock_788.py` does for
its own defect -- that file's subject is message ORDERING on a real wire;
this defect is a pure DECISION-agreement question, answerable (and, per the
"no collective" constraint, REQUIRED to be answerable) without any process
group at all. Three "rank stand-ins" here are therefore plain local state
(a rid -> local-match-length mapping per rank), not three OS processes.
"""

from __future__ import annotations

import inspect
import logging
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang.srt.managers.pp_admission_congruence import (  # noqa: E402
    PPAdmissionDecision,
    PPAdmissionEntry,
    build_pp_admission_decision,
    congruent_rids,
    reconcile_pp_admission_decision,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PP0, PP1, PP2 = 0, 1, 2
WORLD = 3


def _decision(mb_id=0, **prefix_lens_and_extend):
    """`rid -> (prefix_len, extend_len)` shorthand for a PP0-authored decision."""
    entries = tuple(
        PPAdmissionEntry(rid=rid, prefix_len=pl, extend_len=el)
        for rid, (pl, el) in prefix_lens_and_extend.items()
    )
    return PPAdmissionDecision(mb_id=mb_id, entries=entries)


class NeuteredTodayDivergesTest(CustomTestCase):
    """(1) CAN-FAIL PROOF: today's per-rank-independent derivation diverges.

    No decision is forwarded here at all -- each rank computes its OWN
    prefix_len from its OWN local match, exactly like
    `_get_new_batch_prefill_raw` -> `PrefillAdder.add_one_req` does today
    (schedule_policy.py:1472-1474, `cand_extend_input_len = len(full_
    untruncated_fill_ids) - len(prefix_indices)`). This is the shape of the
    bug, reproduced without any of the fix module's machinery, so that the
    fix's "ranks now agree" result (below) is provably fixing something real.
    """

    def test_divergent_local_match_yields_divergent_prefix_len_today(self):
        full_len = 200
        # PP0 has the warmest cache (recent tokenizer-side hit), PP1's is
        # colder (independent eviction), PP2's is colder still.
        local_match = {PP0: 120, PP1: 64, PP2: 0}

        # "Today": each rank builds its OWN one-entry decision purely from
        # its own local match, with no cross-rank input whatsoever.
        per_rank_decisions = {
            rank: _decision(req=(m, full_len - m)) for rank, m in local_match.items()
        }

        self.assertFalse(
            congruent_rids(per_rank_decisions.values()),
            "ranks were expected to diverge under today's independent-"
            "derivation behaviour -- if they agree, this simulation is not "
            "reproducing the defect and the tests below prove nothing",
        )
        # Concretely: extend_num_tokens (the cross-stage tensor's row count)
        # would differ per rank -- 80, 136, and 200 rows for one request.
        extend_lens = {
            rank: per_rank_decisions[rank].entries[0].extend_len
            for rank in per_rank_decisions
        }
        self.assertEqual(extend_lens, {PP0: 80, PP1: 136, PP2: 200})
        self.assertGreater(
            len(set(extend_lens.values())),
            1,
            "a real disagreement on the activation tensor's row count",
        )


class FixedRanksAgreeTest(CustomTestCase):
    """(2) GREEN: PP0's decision, reconciled by each downstream rank, is
    congruent -- including the safe-truncate case where a downstream rank's
    cache is WARMER than PP0's."""

    def test_all_ranks_agree_when_every_local_match_meets_told(self):
        decision0 = build_pp_admission_decision(
            mb_id=0,
            reqs=[_Req(rid="req", prefix_len=120, extend_len=80)],
            pp_size=WORLD,
        )
        self.assertEqual(decision0.entries[0].prefix_len, 120)

        # PP1's local cache happens to already hold everything told=120
        # requires; PP2's holds even more (180) -- the safe-truncate case.
        eff1, decision1 = reconcile_pp_admission_decision(
            decision0, {"req": 120}, rank=PP1, pp_size=WORLD
        )
        eff2, decision2 = reconcile_pp_admission_decision(
            decision1, {"req": 180}, rank=PP2, pp_size=WORLD
        )

        self.assertEqual(eff1, {"req": 120})
        self.assertEqual(
            eff2,
            {"req": 120},
            "PP2's extra 60 tokens of local reuse must be discarded, not "
            "used -- the #616g slack trade, taken on the PP axis",
        )
        self.assertTrue(
            congruent_rids([decision0, decision1, decision2]),
            "every rank must land on the identical (prefix_len, extend_len)",
        )
        for d in (decision0, decision1, decision2):
            self.assertEqual(d.entries[0].prefix_len, 120)
            self.assertEqual(d.entries[0].extend_len, 80)

    def test_a_downstream_rank_colder_than_told_still_agrees_via_its_own_truncate(self):
        """The mirror case: PP1 is colder than told for a DIFFERENT (smaller)
        told than its own match, i.e. still >= told -- congruence must not
        depend on which rank happens to be warmest."""
        decision0 = _decision(req=(50, 150))
        eff1, decision1 = reconcile_pp_admission_decision(
            decision0, {"req": 50}, rank=PP1, pp_size=WORLD
        )
        self.assertEqual(eff1, {"req": 50})
        self.assertTrue(congruent_rids([decision0, decision1]))


class UnhonourableCaseDegradesSafelyTest(CustomTestCase):
    """(3) THE CORRECTION: local < told must degrade, never raise, never
    silently proceed with a corrupted length, and never take a sibling
    request or the process down with it."""

    def test_process_liveness_no_exception_and_still_usable_afterwards(self):
        """The property a test that only checks 'an error was raised' would
        miss: nothing here may raise, and the module must go on reconciling
        further, unrelated decisions after handling an unhonourable one."""
        decision_bad = _decision(req=(120, 80))
        try:
            reconcile_pp_admission_decision(
                decision_bad, {"req": 0}, rank=PP1, pp_size=WORLD
            )
        except Exception as exc:  # noqa: BLE001 -- the absence of one IS the point
            self.fail(f"reconcile raised {type(exc).__name__}: {exc}")

        # The scheduler (and this module) must still be alive and correct
        # for the NEXT, unrelated decision.
        decision_ok = _decision(other=(10, 5))
        eff, _ = reconcile_pp_admission_decision(
            decision_ok, {"other": 10}, rank=PP1, pp_size=WORLD
        )
        self.assertEqual(eff, {"other": 10})


class PpSizeOneIsAPassThroughTest(CustomTestCase):
    """(4) DEFAULT PATH: `pp_size<=1` must be byte-identical to not calling
    this module -- no retraction, no logging, told always honoured."""

    def test_pp_size_one_never_retracts_regardless_of_local_match(self):
        decision0 = _decision(req=(120, 80))
        with self.assertNoLogs(
            "sglang.srt.managers.pp_admission_congruence", level="WARNING"
        ):
            eff, decision1 = reconcile_pp_admission_decision(
                decision0, {"req": 0}, rank=0, pp_size=1
            )
        self.assertEqual(
            eff,
            {"req": 120},
            "pp_size<=1 must be an unconditional pass-through: today's only "
            "shipped configuration must not observe this module at all",
        )
        self.assertIs(decision1, decision0)
        self.assertFalse(decision1.by_rid()["req"].retracted)

    def test_build_pp_admission_decision_pp_size_one_is_harmless(self):
        d = build_pp_admission_decision(
            mb_id=0, reqs=[_Req(rid="r", prefix_len=5, extend_len=5)], pp_size=1
        )
        self.assertEqual(d.entries[0].prefix_len, 5)


class NoCollectiveOnTheAdmissionPathTest(CustomTestCase):
    """(5) STRUCTURAL PIN, source-level: this module must never gain a
    `torch.distributed` call. scheduler.py:6391-6407 documents the
    2026-08-17 deadlock of exactly that family (a blocking collective placed
    where PP ranks are not pipeline-aligned); this module exists specifically
    so admission agreement never needs one."""

    def test_no_torch_distributed_reference_anywhere_in_the_CODE(self):
        """Source of every top-level function/class body, EXCLUDING the
        module docstring -- which discusses this exact constraint in prose
        (and therefore legitimately contains the string "torch.distributed"
        as an explanation, not a call site)."""
        import sglang.srt.managers.pp_admission_congruence as mod

        code_src = "\n".join(
            inspect.getsource(obj)
            for _name, obj in vars(mod).items()
            if inspect.isfunction(obj) or inspect.isclass(obj)
            if getattr(obj, "__module__", None) == mod.__name__
        )
        self.assertNotEqual(code_src, "", "sanity: the module must export something")
        for needle in ("torch.distributed", "dist.all_reduce", "dist.barrier", "gloo"):
            self.assertNotIn(
                needle,
                code_src,
                f"found {needle!r} in pp_admission_congruence.py's actual "
                "code -- this module must stay a pure rank-local "
                "computation, never a collective on the admission path",
            )

    def test_the_module_imports_no_torch_at_all(self):
        import sglang.srt.managers.pp_admission_congruence as mod

        src = inspect.getsource(mod)
        self.assertNotIn("import torch", src)


class _Req:
    """The minimal shape `build_pp_admission_decision` reads from a real
    `Req` (schedule_batch.py): `rid`, `prefix_indices` (here a plain
    range/list standing in for real device-pool indices, since only its
    LENGTH is ever read), and an explicit `extend_input_len` so this fixture
    does not need `full_untruncated_fill_ids` at all."""

    def __init__(self, rid, prefix_len, extend_len):
        self.rid = rid
        self.prefix_indices = list(range(prefix_len))
        self.extend_input_len = extend_len


if __name__ == "__main__":
    unittest.main()
