# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""#1289: `WEG2 X RE-SOLVE` fired ZERO times on a boot with 14 completed flips.

MEASURED on boot weg2sb5f (`4f762260ba`), front log
`boot_weg2_weg2sb5f_4f762260ba_0909_050638.front.log`::

    WEG2-FLIP done   14
    WEG2 P-DRAIN      7
    X RE-SOLVE        0     (any form, including the REFUSED/held lines)

So the shipped `--flip-min-work-tokens 12944` stood for the whole run, and its
own provenance line said `flip_s=3.79 (median of 30)` -- thirty flips of the
PREVIOUS boot on a different layout. This boot's own legs measured 3906 ms
(D->P median of 4) and 4175 ms (P->D median of 4), which re-solve X to
13,333-13,792: flips are MORE expensive on the makespan form than the carried
-in constant assumes, so fewer prompts are worth one, not more.

TWO INDEPENDENT ROOTS, each on its own sufficient to produce the zero. Both
are fixed here, because fixing either alone leaves a boot shape that still
never re-solves.

ROOT 1 -- THE SAMPLE WAS NEVER RECORDED (`front.py`, the flip recorder)::

    self.note_x_sample("flip_s", float(rec.get("flip_total_ms", 0)) / 1000.0)

`rec` has no `flip_total_ms` and never did; the key is `flip_ms`. The name
`flip_total` exists only in the LOG LINE two statements below, which is fed
from `rec["flip_ms"]`. `dict.get` with a default of 0 turned that mismatch
into a silent `0.0`, and `note_x_sample`'s own `value <= 0` guard then dropped
it. The `flip_s` deque was therefore EMPTY for all 14 flips, and
`resolve_x_live` returned None at its first line every time it was called.

ROOT 2 -- THE ONLY TRIGGER WAS THE WRONG COUNTER. `resolve_x_live` was called
from `note_x_sample` on the EIGHTH `r_p` sample only, i.e. the eighth P-DRAIN
window. sb5f had SEVEN drains -- because the 22-min load's ~9k-token prompts
all priced below X=12,944 and went to D alone -- so the trigger could not fire
even once, whatever the flip cost did. The legs are the measurement this
ticket is about, so a completed leg pair now re-solves on its own. The r_p
trigger is kept beside it, unchanged.

WHY THE EXISTING SUITE WAS GREEN THROUGH ALL OF THIS. `test_weg2_flip_pricing_
1271.py::RollingResolve` calls `f.note_x_sample("flip_s", 3.29)` DIRECTLY --
it stubs the producer and so never touches the `rec.get("flip_total_ms")` line
that is the defect. That is the #1285 green-by-vacancy shape again, and it is
why the red-first test below drives the REAL producer (`Front._record_flip`'s
statement, exercised through a recorder harness) instead of the seam beneath
it.
"""

import ast
import collections
import inspect
import textwrap
import unittest

from sglang.srt.weg2.front import Front
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

#: The four D->P and four P->D legs weg2sb5f actually measured, ms.
SB5F_LEGS_MS = [6126, 3836, 3906, 3843, 4175, 4365, 4068, 4062]
#: The front's own wall-clock leg rates on that boot (both group_throughput).
SB5F_R_D = 1130.0
SB5F_R_P = 3344.0
#: What shipped, carried in from sb5e's thirty flips on another layout.
SB5F_SHIPPED_X = 12944


def _front(x=SB5F_SHIPPED_X):
    """A Front with only the X machinery wired -- no server, no session."""
    f = Front.__new__(Front)
    f.tp_prefill_max_tokens = x
    f.flip_min_work_tokens = x
    f._x_min_work_follows = True
    f.x_floor_tokens = 4096
    f._x_samples = {
        "r_d": collections.deque(maxlen=Front.X_SAMPLE_WINDOW),
        "r_p": collections.deque(maxlen=Front.X_SAMPLE_WINDOW),
        "flip_s": collections.deque(maxlen=Front.X_SAMPLE_WINDOW),
    }
    f._x_since_resolve = 0
    f._x_seed_note = f"launcher solve X={x}"
    f._x_last_missing = []
    f.counters = collections.Counter()
    # TOLERANT ON PURPOSE, and only here: this helper must also construct
    # against the PARENT commit `4f762260ba`, where `x_flip_s_provenance` does
    # not exist yet. Binding it unconditionally would make the red-first run
    # an AttributeError at construction -- red for the wrong reason, and it
    # would hide WHICH assertion the fix is for. Production code must never
    # take this shape; a test that has to be red on two trees may.
    for name in ("note_x_sample", "resolve_x_live", "x_flip_s_provenance"):
        fn = getattr(Front, name, None)
        if fn is not None:
            setattr(f, name, fn.__get__(f))
    return f


def _flip_rec(flip_ms: int, epoch: int = 1, src: str = "D", dst: str = "P") -> dict:
    """The record the flip recorder builds, with the keys it really builds."""
    return {"epoch": epoch, "sleep": src, "wake": dst,
            "drain_quiesce_ms": 23, "sleep_ms": flip_ms // 2,
            "wake_ms": flip_ms // 2, "flip_ms": flip_ms,
            "interleave_ms": 0, "chunks": [], "legs_wall_ms": flip_ms,
            "sleep_leg_ms": flip_ms // 2, "wake_leg_ms": flip_ms // 2,
            "overlap_ms": 0, "overlap_pct": 0.0,
            "critical_path": "sleep/D", "dc_mib": 0, "t": 0.0}


def _producer_statement() -> ast.stmt:
    """THE REAL PRODUCER, lifted out of the flip recorder by AST.

    Not a copy: the statement is read from the shipped source, so a future
    edit that reintroduces a wrong key fails here rather than passing against
    a duplicate the test carries. Finds the `note_x_sample("flip_s", ...)`
    call in whichever method records a completed flip.
    """
    for name, fn in inspect.getmembers(Front, predicate=inspect.isfunction):
        try:
            src = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):
            continue
        if 'note_x_sample("flip_s"' not in src:
            continue
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if (isinstance(call.func, ast.Attribute)
                    and call.func.attr == "note_x_sample"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == "flip_s"):
                return node
    raise AssertionError("no note_x_sample('flip_s', ...) producer found")


def run_producer(f, rec: dict) -> None:
    """Execute the SHIPPED producer statement against `rec`, as the flip does."""
    stmt = _producer_statement()
    mod = ast.Module(body=[stmt], type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "<producer>", "exec"), {}, {"self": f, "rec": rec})


class TheProducerRecordsTheLeg(CustomTestCase):
    """RED-FIRST at `4f762260ba`: the producer reads a key `rec` does not have."""

    def test_red_first_a_completed_flip_yields_a_flip_s_sample(self):
        """THE defect. At the parent commit this records nothing at all."""
        f = _front()
        run_producer(f, _flip_rec(3906))
        self.assertEqual(len(f._x_samples["flip_s"]), 1,
                         "a completed flip recorded no flip_s sample -- the "
                         "producer is reading a key the record does not carry")
        self.assertAlmostEqual(f._x_samples["flip_s"][0], 3.906, places=3)

    def test_the_producer_reads_the_same_key_the_log_line_reads(self):
        """MUTANT (wrong key). `flip_total_ms` was invented by the log line's
        FORMAT NAME; the dict key is `flip_ms`, and the two must not diverge
        again. A `.get` with a default here is the silent-zero shape."""
        stmt = _producer_statement()
        dumped = ast.dump(stmt)
        self.assertIn("flip_ms", dumped)
        self.assertNotIn("flip_total_ms", dumped)
        self.assertNotIn("attr='get'", dumped,
                         "a dict.get default turns a key typo into a silent 0")

    def test_mutant_a_missing_key_raises_instead_of_recording_zero(self):
        """MUTANT (silent zero). Subscript, not `.get`: if the record ever
        loses `flip_ms` this must BREAK, not quietly price flips at 0 s."""
        f = _front()
        rec = _flip_rec(3906)
        del rec["flip_ms"]
        with self.assertRaises(KeyError):
            run_producer(f, rec)

    def test_mutant_a_zero_length_flip_is_still_refused(self):
        """The `value <= 0` guard stays: 0 s is not a measurement."""
        f = _front()
        run_producer(f, _flip_rec(0))
        self.assertEqual(len(f._x_samples["flip_s"]), 0)


class EveryLegPairReSolves(CustomTestCase):
    """ROOT 2: the trigger was the eighth drain, on a boot with seven."""

    def _seed_rates(self, f):
        f.note_x_sample("r_d", SB5F_R_D)
        f.note_x_sample("r_p", SB5F_R_P)

    def test_red_first_seven_drains_and_fourteen_flips_still_re_solve(self):
        """SB5F'S EXACT SHAPE. At the parent commit: 0 re-solves."""
        f = _front()
        f.note_x_sample("r_d", SB5F_R_D)
        for _ in range(7):  # seven P-DRAIN windows -- one short of the trigger
            f.note_x_sample("r_p", SB5F_R_P)
        self.assertEqual(f.counters["x_resolves"], 0, "premise: no drain trigger")
        for i, ms in enumerate(SB5F_LEGS_MS):
            run_producer(f, _flip_rec(ms, epoch=i + 1))
        self.assertGreater(f.counters["x_resolves"], 0,
                           "14 completed flips and still no re-solve")

    def test_the_re_solved_x_matches_this_boots_own_legs(self):
        """The number the boot record computed by hand, now computed by code:
        flip_s ~3.9-4.0 s against r_D=1130 / r_P=3344 gives ~13.3-13.8k, ABOVE
        the shipped 12,944."""
        f = _front()
        self._seed_rates(f)
        for i, ms in enumerate(SB5F_LEGS_MS):
            run_producer(f, _flip_rec(ms, epoch=i + 1))
        self.assertGreater(f.tp_prefill_max_tokens, SB5F_SHIPPED_X,
                           "flips are more expensive on this form, so X rises")
        self.assertGreater(f.tp_prefill_max_tokens, 13000)
        self.assertLess(f.tp_prefill_max_tokens, 14500)

    def test_one_leg_pair_is_enough_when_the_rates_are_there(self):
        """The legs are the source. No count of legs is withheld from the
        solver -- the WINDOW median is what smooths."""
        f = _front()
        self._seed_rates(f)
        run_producer(f, _flip_rec(3906))
        self.assertEqual(f.counters["x_resolves"], 1)

    def test_mutant_a_counter_gated_trigger_never_fires_on_this_boot(self):
        """MUTANT (the shipped defect, as a property). Any rule that needs N>=8
        SAMPLES OF ONE KIND before it may solve is dead on sb5f's shape: it had
        7 drains. Asserted against the code so a future 'resolve every 8 flips'
        rewrite is caught -- flips are rarer than drains, not commoner."""
        src = inspect.getsource(Front.note_x_sample)
        tree = ast.parse(textwrap.dedent(src))
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_dump = ast.dump(node.test)
            if "X_RESOLVE_EVERY" not in test_dump:
                continue
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            self.assertNotIn("flip_s", body,
                             "the leg-pair path must not sit behind the "
                             "sample-count gate that produced the zero")

    def test_the_drain_trigger_is_unchanged(self):
        """Not a replacement: two ways in, one arithmetic."""
        f = _front()
        f.note_x_sample("r_d", SB5F_R_D)
        run_producer(f, _flip_rec(3906))   # flip_s present, r_p still empty
        self.assertEqual(f.counters["x_resolves"], 0, "no r_P yet: no solve")
        for _ in range(Front.X_RESOLVE_EVERY):
            f.note_x_sample("r_p", SB5F_R_P)
        self.assertGreaterEqual(f.counters["x_resolves"], 1)

    def test_the_smoothing_window_is_kept(self):
        f = _front()
        self._seed_rates(f)
        for i in range(Front.X_SAMPLE_WINDOW + 20):
            run_producer(f, _flip_rec(3906, epoch=i + 1))
        self.assertEqual(len(f._x_samples["flip_s"]), Front.X_SAMPLE_WINDOW)


class TheProvenanceIsOnEveryDecision(CustomTestCase):
    """`flip_s source=live|seed n=` -- never a hand number without a label."""

    def test_seed_before_any_leg(self):
        f = _front()
        self.assertEqual(f.x_flip_s_provenance(), "flip_s source=seed n=0")

    def test_live_after_the_first_leg(self):
        f = _front()
        run_producer(f, _flip_rec(3906))
        self.assertEqual(f.x_flip_s_provenance(), "flip_s source=live n=1")

    def test_it_counts_the_legs_it_actually_holds(self):
        f = _front()
        for i, ms in enumerate(SB5F_LEGS_MS):
            run_producer(f, _flip_rec(ms, epoch=i + 1))
        self.assertEqual(f.x_flip_s_provenance(),
                         f"flip_s source=live n={len(SB5F_LEGS_MS)}")

    def test_all_three_x_decision_lines_carry_it(self):
        """The re-solve, the held-no-break-even, and the no-solve."""
        src = inspect.getsource(Front.resolve_x_live)
        self.assertEqual(src.count("x_flip_s_provenance()"), 3,
                         "an X decision that does not print its flip_s "
                         "provenance is a threshold nobody can account for")

    def test_the_no_solve_state_is_no_longer_silent(self):
        """sb5f's whole defect was invisible: the early return said nothing."""
        src = inspect.getsource(Front.resolve_x_live)
        self.assertIn("WEG2 X NO-SOLVE", src)
        self.assertIn("was SILENT on", src)

    def test_state_dict_carries_x_with_its_provenance(self):
        src = inspect.getsource(Front.state_dict)
        for key in ('"x_tokens"', '"flip_min_work_tokens"', '"x_flip_s"'):
            self.assertIn(key, src)


class TheLoadMustReachTheThreshold(CustomTestCase):
    """FINDING 1 of the sb5f record, as an assertion rather than a note.

    The 22-min driver's ~9k-token prompts priced `SHORT -> D est_prompt=8973`
    and produced ZERO natural flips over 15 rounds and 464 requests. Any
    acceptance that wants natural flips must send prompts whose UNCACHED
    REMAINDER exceeds X -- 12,944 as shipped, ~13.3-13.8k once re-solved from
    this boot's own legs -- or drive a backlog whose uncached SUM does.
    """

    SB5F_LOAD_UNCACHED = 8973

    def test_the_standard_load_is_below_the_shipped_threshold(self):
        self.assertLess(self.SB5F_LOAD_UNCACHED, SB5F_SHIPPED_X)

    def test_and_further_below_the_re_solved_one(self):
        f = _front()
        f.note_x_sample("r_d", SB5F_R_D)
        f.note_x_sample("r_p", SB5F_R_P)
        for i, ms in enumerate(SB5F_LEGS_MS):
            run_producer(f, _flip_rec(ms, epoch=i + 1))
        self.assertLess(self.SB5F_LOAD_UNCACHED, f.tp_prefill_max_tokens,
                        "a re-solve that RAISES X makes the coverage gap "
                        "wider, not narrower -- the driver needs a long arm")


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
