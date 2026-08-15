"""#363 gate 3 -- a constant may only BLOCK the gate if the runtime enforces it.

WHY THIS FILE EXISTS
--------------------
``spread_veto_pct = 25`` blocked gate 3 for four consecutive recordings with
the verdict ``UNREACHED``, and the reason it is unreachable is not that the rig
is quiet. It is that **no runtime code reads it**. The interlock the constant
was standing in for vetoes on a MISSING measurement -- ``rank_ms_spread_pct is
None`` -- and never on the measurement's magnitude, so no value of the signal,
however large, can ever trip a threshold that nothing compares against.

Evidence, measured not asserted:

* ``docs/dev/363/TICKET_363_WINDOW_VERDICT.md`` section 3 -- ``grep -rn
  'spread_veto_pct' python/`` returns nothing, and the live interlock is shown
  vetoing on ``None``.
* Its four recorded verdicts, ``UNREACHED`` every time: peak ``0.68 %`` on the
  card-gates runsheet, peak ``0.407 %`` in the window verdict -- 61x below the
  constant.

A constant in that position is not a failing check. It is a check that was
never wired, and leaving it in the blocking set means the gate reports a
verdict about the rig when what it has measured is a gap in itself. Retired
here: it is still JUDGED and still REPORTED, so the reachability evidence keeps
accumulating, but it can no longer block.

THE RULE, STRUCTURAL RATHER THAN PER-NAME
-----------------------------------------
Retiring one name by hand would leave the next orphan to be found by the next
shift. So the property is made structural: a constant that can block must NAME
the runtime symbol it was read from, and this file resolves that symbol in the
runtime and checks the VALUE still agrees. That makes two failures impossible
to ship silently -- a constant nothing enforces, and a constant whose gate copy
has drifted from the runtime's.

Hermetic; no GPU, no boot.
"""

import importlib
import json
import os
import sys
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_SCRIPTS = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "regime_gates"
    )
)
sys.path.insert(0, _SCRIPTS)

import bands  # noqa: E402


def _write(path, rows):
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "header", "mode": "observe", "rank": 0}) + "\n")
        for i, r in enumerate(rows):
            rec = {"kind": "verdict", "rank": 0, "round": (i + 1) * 8}
            rec.update(r)
            f.write(json.dumps(rec) + "\n")


def _active(spread):
    """A boundary that RAN a forward, carrying the spread.

    ``bands.is_active`` reads the shares -- an idle window carries ``None`` for
    both -- so a row without a share is restricted away and the signal reports
    NO_DATA instead of the UNREACHED these tests are about.
    """
    return {"prefill_share": 0.1, "decode_share": 0.9, "rank_ms_spread_pct": spread}


def _pair(tmp, rows_a, rows_b):
    a, b = os.path.join(tmp, "a.jsonl"), os.path.join(tmp, "b.jsonl")
    _write(a, rows_a)
    _write(b, rows_b)
    return a, b


class TestBlockingSetCarriesNoRuntimeOrphan(CustomTestCase):
    """The pin the mandate asks for, in its general form."""

    def test_every_blocking_eligible_constant_names_a_live_runtime_symbol(self):
        checked = []
        for c in bands.CONSTANTS:
            if not c.blocking_eligible:
                continue
            self.assertTrue(
                c.runtime_site,
                f"{c.name} can block gate 3 but names no runtime site",
            )
            mod_name, _, sym = c.runtime_site.partition(":")
            self.assertTrue(sym, f"{c.name}: runtime_site needs 'module:SYMBOL'")
            mod = importlib.import_module(mod_name)
            self.assertTrue(
                hasattr(mod, sym),
                f"{c.name} claims {c.runtime_site}, which does not exist",
            )
            checked.append(c.name)
        # The rule is worth nothing if it vacuously passes on an empty set.
        self.assertGreaterEqual(len(checked), 3, checked)

    def test_a_blocking_constants_value_still_agrees_with_the_runtime(self):
        """Drift between the gate's copy and the runtime's is the second way
        this gate can report about the rig while measuring itself."""
        for c in bands.CONSTANTS:
            if not c.blocking_eligible:
                continue
            mod_name, _, sym = c.runtime_site.partition(":")
            live = getattr(importlib.import_module(mod_name), sym)
            self.assertEqual(
                float(live),
                float(c.value),
                f"{c.name}: gate holds {c.value}, runtime {c.runtime_site} "
                f"holds {live}",
            )

    def test_the_orphan_detector_can_fail_on_a_missing_symbol(self):
        """Proven able to fail, per the register's own rule -- a detector that
        has never been seen to fail has not been shown to detect anything."""
        bogus = bands.Constant(
            "bogus",
            1.0,
            "decode_share",
            runtime_site="sglang.srt.managers.regime_classifier:NO_SUCH_SYMBOL_363",
        )
        self.assertTrue(bogus.blocking_eligible)
        mod_name, _, sym = bogus.runtime_site.partition(":")
        self.assertFalse(hasattr(importlib.import_module(mod_name), sym))

    def test_the_drift_detector_can_fail_on_a_changed_value(self):
        drifted = bands.Constant(
            "drifted",
            0.123456,
            "occupancy",
            runtime_site="sglang.srt.managers.regime_classifier:KV_ASCEND_MARK",
        )
        mod_name, _, sym = drifted.runtime_site.partition(":")
        live = getattr(importlib.import_module(mod_name), sym)
        self.assertNotEqual(float(live), float(drifted.value))


class TestSpreadVetoIsRetired(CustomTestCase):
    def test_spread_veto_pct_is_not_blocking_eligible(self):
        c = next(c for c in bands.CONSTANTS if c.name == "spread_veto_pct")
        self.assertFalse(c.blocking_eligible)
        self.assertIsNone(c.runtime_site)

    def test_it_is_still_judged_and_still_reported(self):
        """Retired from blocking, NOT deleted: the reachability evidence has to
        keep accumulating, or the next shift re-derives the same 25."""
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [_active(0.4) for _ in range(20)],
                [_active(0.4) for _ in range(20)],
            )
            rep = bands.report(a, b)
            named = {c["constant"] for c in rep["constants"]}
            self.assertIn("spread_veto_pct", named)
            v = next(c for c in rep["constants"] if c["constant"] == "spread_veto_pct")
            self.assertEqual(v["verdict"], "UNREACHED")
            self.assertIs(v["blocking_eligible"], False)

    def test_an_unreached_orphan_no_longer_blocks_the_gate(self):
        """The behaviour change, stated as an outcome rather than a flag: the
        exact recorded situation -- spread peaking two orders of magnitude
        below 25 -- must no longer appear in ``blocking``."""
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [_active(0.407) for _ in range(20)],
                [_active(0.401) for _ in range(20)],
            )
            rep = bands.report(a, b)
            self.assertNotIn(
                "spread_veto_pct",
                " ".join(rep["blocking"]),
                rep["blocking"],
            )

    def test_the_retirement_is_carried_in_the_report_for_the_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [_active(0.4) for _ in range(20)],
                [_active(0.4) for _ in range(20)],
            )
            rep = bands.report(a, b)
            self.assertIn("spread_veto_pct", rep["retired"])
            entry = bands.evidence_entry(rep)
            self.assertIn("retired", entry["f3_bands_measured"]["source"])


class TestRetirementDoesNotWeakenTheGate(CustomTestCase):
    def test_a_wired_constant_inside_its_band_still_blocks(self):
        """The one failure mode of this change: retiring the orphan must not
        become retiring the gate."""
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [{"occupancy": 0.5} for _ in range(20)],
                [{"occupancy": 0.5} for _ in range(20)],
            )
            rep = bands.report(a, b)
            self.assertFalse(rep["passed"])
            self.assertTrue(rep["blocking"])
            # and what blocks is a constant the runtime actually reads
            for item in rep["blocking"]:
                name = item.split(":")[0]
                c = next(c for c in bands.CONSTANTS if c.name == name)
                self.assertTrue(c.blocking_eligible, name)


if __name__ == "__main__":
    unittest.main()
