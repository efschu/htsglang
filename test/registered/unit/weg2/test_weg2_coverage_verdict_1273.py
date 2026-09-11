# SPDX-License-Identifier: Apache-2.0
"""B4g wall 1: the coverage verdict REACHES a fence, so W84 can fire.

Boot weg2xsn14 printed, on 39 of its 40 ``WEG2-XCHG-COVER`` lines::

    WEG2-XCHG-COVER rank=... uncovered=12 short=0 missing=0
    params=114 buffers=21 attrs=6 tms_answered=yes mode=...

``TagCoverage.ok`` is therefore False on all 39, and ``W84
Weg2XchgCoverageRefused`` read **genuine 0**.  That is not a wrong predicate:
``arm_coverage`` is DELIBERATELY non-raising (refuter F5 -- at the end of
weight loading there is no group fence in scope, so a rank that raised there
would die while the other five walked into a collective with five members),
and the raiser it hands the verdict to, :func:`refuse_if_not_ok`, is documented
*"ONLY where a group fence covers it"* and HAS NO PRODUCTION CALLER.  The vote
is computed, recorded by ``_record_boot_vote``, readable through
``boot_vote()`` -- and consumed by nobody.  A success value without an action,
which is one of this fork's catalogued defect classes.

So the fix is the missing CONSUMER, at the fenced site the vote's own docstring
names (the wake RPC's preamble), and the two modes differ exactly as ruled:

* ``inject=shadow`` -- the ring is the authority and the exchange lane is
  report-only, so the SHADOW LEG is refused BY NAME and the flip proceeds.  A
  boot that cannot account for a tag must not also lose its flip.
* ``inject=authoritative`` -- the exchange owns the bytes, so it is a STOP, and
  group-uniform: the reason rides the wake fence's per-rank dict and every rank
  raises ``Weg2FlipRankDisagree`` (W29) naming W84's own text, rather than one
  rank raising W84 alone.  Same carrier B4d uses, same reason.

Hermetic: fabricated votes, no torch.distributed, no model.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _live(name, mib=1.0):
    return wx.LiveTensor(name=name, module_path="model.layers.0",
                         kind="ATTRIBUTE", tag="weights_0",
                         nbytes=int(mib * wx.MIB), storage_key=(id(name), 0),
                         dtype="torch.int8", shape=(1024, 1024))


def _row(uncovered=12, short=(), missing=(), tag="weights_0", rank=0):
    """XSN14's own line: uncovered=12 short=0 missing=0 params=114 buffers=21."""
    return wx.TagCoverage(
        rank=rank, tag=tag, mode="exchange",
        planned_bytes=2988 * wx.MIB, buffers_bytes=21 * wx.MIB,
        tms_bytes=3009 * wx.MIB,
        uncovered=tuple(_live(f"uncov_{i}") for i in range(uncovered)),
        short=tuple(short), missing=tuple(missing),
        n_parameters=114, n_buffers=21, n_attributes=6,
    )


def _vote(ok, rows=None):
    rows = rows if rows is not None else {"weights_0": _row()}
    return wx.CoverageVote(
        rank=0, mode="exchange", region_tag=wx.GPU_MEMORY_TYPE_WEIGHTS,
        rows=rows, ok=ok,
        reason="" if ok else wx.coverage_refusal_message(rows),
    )


class _Boot:
    """Install a boot vote for the duration of a test."""

    def __init__(self, vote, inject):
        self.vote, self.inject = vote, inject

    def __enter__(self):
        self.old_inject = os.environ.get(wx.INJECT_ENV)
        os.environ[wx.INJECT_ENV] = self.inject
        wx._record_boot_vote(self.vote)
        return self

    def __exit__(self, *a):
        wx._record_boot_vote(None)
        if self.old_inject is None:
            os.environ.pop(wx.INJECT_ENV, None)
        else:
            os.environ[wx.INJECT_ENV] = self.old_inject


class TheVerdictIsNoLongerComputedAndDropped(CustomTestCase):
    def test_xsn14s_own_row_is_not_ok(self):
        """The premise, asserted rather than assumed."""
        row = _row()
        self.assertFalse(row.ok)
        self.assertIn("uncovered=12 short=0 missing=0", row.cover_line())
        self.assertIn("params=114 buffers=21 attrs=6", row.cover_line())

    def test_a_clean_vote_arms_the_leg_and_says_nothing(self):
        with _Boot(_vote(True), wx.INJECT_SHADOW):
            armed, reason, stop = wx.coverage_leg_decision()
            self.assertTrue(armed)
            self.assertEqual(reason, "")
            self.assertFalse(stop)

    def test_no_vote_at_all_arms_nothing_and_does_not_stop(self):
        """A boot that never armed coverage (ring) must be untouched."""
        with _Boot(None, wx.INJECT_SHADOW):
            armed, reason, stop = wx.coverage_leg_decision()
            self.assertTrue(armed)
            self.assertFalse(stop)

    def test_under_shadow_the_LEG_is_refused_by_name_and_the_flip_goes_on(self):
        with _Boot(_vote(False), wx.INJECT_SHADOW):
            armed, reason, stop = wx.coverage_leg_decision()
            self.assertFalse(armed)          # the shadow leg does not run
            self.assertFalse(stop)           # but the flip is not stopped
            self.assertIn(wx.COVERAGE_REFUSAL_MARKER, reason)
            self.assertIn("SHADOW LEG REFUSED", reason)
            self.assertIn("uncov_0", reason)   # the tensors are NAMED

    def test_under_authoritative_it_is_a_stop(self):
        with _Boot(_vote(False), wx.INJECT_AUTHORITATIVE):
            armed, reason, stop = wx.coverage_leg_decision()
            self.assertFalse(armed)
            self.assertTrue(stop)
            self.assertIn(wx.COVERAGE_REFUSAL_MARKER, reason)
            self.assertIn("STOP", reason)

    def test_short_and_missing_refuse_too_not_only_uncovered(self):
        rows = {"weights_1": _row(uncovered=0, missing=("mlp.down_proj.weight",))}
        with _Boot(_vote(False, rows), wx.INJECT_AUTHORITATIVE):
            _armed, reason, stop = wx.coverage_leg_decision()
            self.assertTrue(stop)
            self.assertIn("MISSING", reason)

    def test_the_marker_is_W84_and_not_a_new_code(self):
        self.assertEqual(wx.COVERAGE_REFUSAL_MARKER, "W84 Weg2XchgCoverageRefused")


class TheStopIsGroupUniformThroughTheWakeFence(CustomTestCase):
    """Not one rank raising W84 alone -- that is the F5 defect the vote exists
    to avoid.  The reason rides the fence's per-rank dict, so every rank
    raises W29 naming W84's text."""

    def _fence_pairs(self):
        import ast
        import inspect
        import textwrap

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = textwrap.dedent(inspect.getsource(
            wu.SchedulerWeightUpdaterManager._weg2_group_fence_impl))
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "mine" for t in node.targets):
                return {k.value: ast.unparse(v)
                        for k, v in zip(node.value.keys, node.value.values)}
        self.fail("the fence no longer builds a per-rank dict called `mine`")

    def test_the_coverage_stop_votes_not_ok(self):
        pairs = self._fence_pairs()
        self.assertIn("coverage_stop", pairs["ok"])

    def test_the_coverage_reason_rides_the_failure_field(self):
        pairs = self._fence_pairs()
        self.assertIn("coverage_reason", pairs["failure"])

    def test_the_raise_is_still_W29(self):
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu.SchedulerWeightUpdaterManager._weg2_group_fence_impl)
        self.assertIn("W29 Weg2FlipRankDisagree", src)

    def test_refuse_if_not_ok_stays_unwired(self):
        """The operator's ruling: it keeps its zero production callers.  The
        fence is the carrier, so this rank-local raiser is not revived."""
        import subprocess

        out = subprocess.run(
            ["grep", "-rn", "refuse_if_not_ok(", "python/sglang/srt/"],
            capture_output=True, text=True).stdout
        callers = [ln for ln in out.splitlines() if "def refuse_if_not_ok" not in ln]
        self.assertEqual(callers, [], f"revived: {callers}")


if __name__ == "__main__":
    unittest.main()


class TheUncoveredPopulationIsClassifiedNotRelabelled(CustomTestCase):
    """B4g wall 1, second half: WHY 12 tensors were uncovered, and what a real
    ``uncovered=0`` requires.

    STRUCTURAL FINDING FIRST, from the census's own arithmetic and xsn14's own
    line, with no model: a tensor is uncovered only if it is (a) a PARAMETER
    whose name is not in the plan map, or (b) an ATTRIBUTE aliasing no covered
    storage.  The line reads ``uncovered=12 ... attrs=6``, so at most 6 can be
    attributes and therefore **at least 6 are unplanned PARAMETERS**.  That is
    a plan-coverage gap, not stray attribute aliases.

    AND THE MECHANISM IS NAMED: ``plan_bytes_from_descs`` SKIPS ZEROFILL
    descriptors -- correctly, because a zerofill piece moves nothing over the
    link and counting it would claim a transfer that never happens.  But a
    parameter whose descriptors are ALL zerofill then vanishes from the map
    entirely, and the census cannot tell "absent BY DESIGN" (the 128 padded
    vocabulary rows that exist on no card and in no checkpoint, spec section
    2.2) from "absent because the plan forgot it".  Both read as uncovered.

    So the plan DECLARES the zerofill-by-design parameters with 0 planned
    bytes instead of omitting them, and the census classifies them as EXEMPT
    with a printed reason.  Everything else stays uncovered AND IS NAMED on its
    own line, so the table comes out of one boot instead of a second
    investigation.
    """

    def test_a_fully_zerofill_parameter_is_declared_not_omitted(self):
        class D:
            def __init__(self, kind, name, nbytes):
                self.kind, self.param_name, self.nbytes = kind, name, nbytes
                self.tag = "weights_0"

        descs = [D(wx.ZEROFILL, "embed.pad", 4096),
                 D("COPY", "mlp.gate", 1024), D(wx.ZEROFILL, "mlp.gate", 16)]
        out = wx.plan_bytes_from_descs(descs)
        # the copied parameter keeps its non-zerofill bytes ONLY
        self.assertEqual(out["weights_0"]["mlp.gate"], 1024)
        # and the all-zerofill one is PRESENT with zero, not missing
        self.assertIn("embed.pad", out["weights_0"])
        self.assertEqual(out["weights_0"]["embed.pad"], 0)

    def test_a_zero_claim_is_exempt_and_never_short(self):
        """A 0-byte claim against live bytes is the zerofill case, not a
        partially-tiled parameter -- calling it SHORT would refuse every boot
        that pads a vocabulary."""
        self.assertTrue(wx.is_zerofill_by_design(0, 4096))
        self.assertFalse(wx.is_zerofill_by_design(1024, 4096))
        self.assertFalse(wx.is_zerofill_by_design(0, 0))

    def test_the_cover_line_prints_exempt_with_its_reason(self):
        row = _row(uncovered=0)
        row = wx.TagCoverage(**{**row.__dict__, "exempt": ("embed.pad",)})
        line = row.cover_line()
        self.assertIn("exempt=1", line)
        self.assertIn("reason=", line)
        self.assertIn("zerofill-by-design", line)

    def test_an_exempt_tensor_does_not_make_the_row_not_ok(self):
        row = wx.TagCoverage(**{**_row(uncovered=0).__dict__,
                                "exempt": ("embed.pad",)})
        self.assertTrue(row.ok)

    def test_but_an_uncovered_one_still_does(self):
        row = wx.TagCoverage(**{**_row(uncovered=3).__dict__,
                                "exempt": ("embed.pad",)})
        self.assertFalse(row.ok)

    def test_the_uncovered_tensors_are_NAMED_on_their_own_line(self):
        """So the table is a one-boot deliverable.  xsn14 could not produce it:
        the names live only in ``coverage_refusal_message``, which the refusal
        that never fired was the only caller of."""
        row = _row(uncovered=2)
        lines = row.uncovered_lines()
        self.assertEqual(len(lines), 2)
        for i, ln in enumerate(lines):
            self.assertIn(wx.UNCOVERED_LINE_PREFIX, ln)
            self.assertIn(f"name=uncov_{i}", ln)
            self.assertIn("tag=weights_0", ln)
            self.assertIn("kind=ATTRIBUTE", ln)
            self.assertIn("mib=", ln)

    def test_a_clean_row_names_nothing(self):
        self.assertEqual(_row(uncovered=0).uncovered_lines(), [])

    def test_arm_coverage_ACTUALLY_EMITS_them(self):
        """A MUTANT bought this: the test above drives the helper, and nothing
        pinned that the emitter calls it -- so replacing the emit loop with
        `pass` scored GREEN-SURVIVED and the table would silently not exist.

        Third instance of one class in this campaign (B4f's M5, B4d's M9, this):
        a pin on a helper is not a pin on the site that uses it.  Structural,
        via AST, so it survives re-wrapping.
        """
        import ast
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(wx.arm_coverage))
        tree = ast.parse(src)
        calls = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertTrue(
            any("uncovered_lines()" in c for c in calls),
            "arm_coverage must CALL uncovered_lines(), not merely be able to")
        # and the result must reach `emit`, not a local
        emits = [c for c in calls if c.startswith("emit(")]
        self.assertTrue(any("ln" == c[len("emit("):-1] for c in emits),
                        f"the named lines must be emitted; emits={emits}")
