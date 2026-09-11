# SPDX-License-Identifier: Apache-2.0
"""#1329 (B4m, spec AMENDMENT 7 point 4): THE LAUNCH REPLAY AT THE DESK.

THE MEASURED REASON THIS FILE EXISTS.  Three of the four walls of the night
of 2026-09-11 were LAUNCHER ARM ARITHMETIC over inputs that already existed
in recorded form, and each one cost a whole boot cycle -- a gpuq window, a
teardown, a record, a re-stamp -- to discover:

* **XSN13** (``48f55fb393``) never started a rank: ``launcher.py:4625``
  declared ``ring_absent_by_design=(weight_source == "exchange")`` while the
  same run armed a 42.96 GiB ring, and ``host_ledger.py:1832`` refused the
  contradiction BY DESIGN ("a contradiction resolved silently is how a
  charged term becomes invisible").  Inputs: two integers and two arm
  strings.
* **XSN14** (``692b1e8698``) died at the first flip on
  ``W19 DormantResidueRefused``: measured dormant residue 2588/3084/2588 MiB
  against a reserve of 1986/2292/1986 built from constants MEASURED ON A
  SERVING BOOT.  Inputs: three per-card measurements and two constants.
* **W71** (before B4c) refused ``--weg2-weight-source exchange`` for want of
  a census file, because nothing produced one.  Input: a path.

OPERATOR_HANDOVER_S6_0911.md section 5 names the class in one line: all of
them live in the arm layer, which was built for the SERVING form and does
not know the exchange form -- and "drei der vier waren Dry-Run-Arithmetik
ueber aufgezeichnete Inputs und haetten am Desk feuern koennen.  Das ist die
teuerste Lehre."  This file is that lesson as a test: the arithmetic runs
against RECORDED inputs, hermetically, with no boot, no NVML and no CUDA,
and each of the three walls FIRES here on the inputs of the boot that hit
it while the fixed tip is GREEN.

THE FIXTURE (``fixtures/xchg_launch_replay_0911/``), 151 KB, every number
sourced in ``recorded.json``:

* ``ring_evidence/`` -- ONLY the lines ``ring_table.solve`` parses out of the
  ring source boot's three logs, plus that boot's own samples from the
  host-ledger sidecar.  48 MB of logs reduced to 151 KB, and the reduction
  is PROVEN lossless rather than assumed: ``solve`` returns byte-identical
  totals from this directory and from ``/spinning/evidence-665-f1``
  (:meth:`TestTheFixtureIsFaithful.test_the_extract_equals_the_full_evidence_dir`,
  which skips when the real evidence dir is not on the box).  NEVER a ring
  file: the ring itself is 43 GiB of /dev/shm and this replay allocates
  nothing.
* ``census_weg2sn5b_48f55fb393.json`` -- the census B4c's producer wrote,
  the same file XSN13/14/15 armed from.
* ``p_argv_xsn14.txt`` -- group P's argv, which ARMS THE FORM GATE inside
  ``solve`` and is therefore an INPUT, not decoration.
* ``recorded.json`` -- the recorded inputs AND the recorded verdicts.

ONE DEPENDENCY THAT IS NOT A RECORDED INPUT, found by running this file on the
REMOTE desk and not on the rig: arming the W48 form gate (i.e. handing ``solve``
this boot's own P argv) makes it price that argv against the CHECKPOINT HEADERS
of ``--model-path`` -- ``planner/pp_cut.checkpoint_weight_terms`` globs
``*.safetensors`` and reads each shard's header -- and that 27 GB directory
exists only on the rig.  On cachyllama the three form-gate tests failed with
``W48 Weg2RingFormMismatch: ... the checkpoint terms of '<model path>' could not
be read``.  So the replay is hermetic with respect to NVML, CUDA, /dev/shm and
the boot, and NOT with respect to the model directory.  That is now DATA
(``recorded.json`` -> ``checkpoint_dependency``, with the path, the reader, the
call that reaches it and the consequence), the three tests that need it SKIP by
name where it is absent, and a second ring pin -- the same solve with the gate
UNARMED, its own answer recorded separately -- runs EVERYWHERE and still guards
the whole parse surface of the extracted evidence.  Making the armed pin
hermetic too means recording ``checkpoint_weight_terms``'s OUTPUT; that is owed
and named in the fixture.

WHAT THIS REPLAY DOES NOT COVER, named rather than implied: the W20/W21
LADDER.  ``choose_host_ledger`` already has the seam for it (its
``meminfo_path`` / ``cgroup_root`` parameters exist "so a test can hand this
seam a fake box whose readings decide a DIFFERENT arm"), but no boot log
records the box's readings in BYTES -- only rounded GiB prose
("non-reclaimable reading 8.22 GiB").  Replaying the ladder off rounded
prose would grade an arm against inputs the boot did not have, so it is
left out and reported as an open item: it needs one emitter that dumps the
raw meminfo/cgroup readings the ledger already holds.  The contradiction
wall below is unaffected -- ``price()`` raises it BEFORE it touches any host
number, which this file pins (:meth:`test_the_contradiction_precedes_every
_host_reading`).

RED-FIRST.  Each wall class asserts BOTH halves -- the broken input shape
raises, the fixed one does not -- so the file's red-first evidence is
internal and re-runnable rather than a claim about a checkout that no
longer exists.  Where a pre-fix TREE is needed to make the point (W19's
reserve came from a constant family the fix re-sourced), the replay drives
the SAME function with the pre-fix ARM STRING, which is what selects the
pre-fix constants on this tip.
"""

import ast
import copy
import glob
import json
import os
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger, ring_table
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import xchg_residency as xr
from sglang.test.test_utils import CustomTestCase

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "xchg_launch_replay_0911")
RING_EVIDENCE = os.path.join(FIXTURES, "ring_evidence")
CENSUS = os.path.join(FIXTURES, "census_weg2sn5b_48f55fb393.json")
FULL_EVIDENCE_DIR = "/spinning/evidence-665-f1"

with open(os.path.join(FIXTURES, "recorded.json"), encoding="utf-8") as _fh:
    RECORDED = json.load(_fh)

#: The box readings for the ledger call.  APPROXIMATE, from XSN13's preflight
#: prose ("MemAvailable 109.3 GiB") plus this rig's MemTotal, and they are
#: allowed to be approximate for exactly one reason, pinned by
#: ``test_the_contradiction_precedes_every_host_reading``: the contradiction
#: guard runs before ``price`` reads any host term, so no host number can
#: change that verdict.  Any test that GRADED an arm would need recorded
#: bytes instead -- see the module docstring's open item.
GIB = 1024 ** 3
APPROX_MEMTOTAL_BYTES = 135 * GIB
APPROX_MEMAVAIL_BYTES = 109 * GIB


def recorded_cards() -> list[L.Card]:
    """This boot's live cards, IN ORDINAL ORDER, from the recorded NVML map.

    Order is load-bearing (``recorded.json`` says why and quotes the measured
    consequence): ``ring_table.solve`` maps list position to PP stage.
    """
    return [
        L.Card(nvml_index=int(c["nvml_index"]), uuid=c["uuid"],
               name=c["name"], total_mib=int(c["total_mib"]))
        for c in RECORDED["cards_ordinal_order"]
    ]


def recorded_p_argv() -> list[str]:
    with open(os.path.join(FIXTURES, "p_argv_xsn14.txt"), encoding="utf-8") as fh:
        return fh.read().split()


def solve_recorded_ring(evidence_dir: str = RING_EVIDENCE, *, form_gate: bool = True):
    """``ring_table.solve`` over the recorded evidence.

    ``form_gate=True`` hands it this boot's own P argv, which is what makes the
    answer byte-identical to the boot's -- and what makes it NON-HERMETIC: the
    W48 form gate prices the argv against the CHECKPOINT HEADERS of
    ``--model-path`` (``planner/pp_cut.checkpoint_weight_terms`` globs
    ``*.safetensors`` and reads each shard's header), and that 27 GB directory
    exists only on the rig.  ``form_gate=False`` is the hermetic form: no
    checkpoint is touched, and its answer is recorded separately.
    """
    return ring_table.solve(
        recorded_cards(), evidence_dir, RECORDED["ring_table"]["boot_stem"],
        p_argv=recorded_p_argv() if form_gate else None,
    )


def checkpoint_present() -> bool:
    """Is the checkpoint the recorded P argv names on THIS box?

    FOUND ON cachyllama, not on the rig (2026-09-11): the three form-gate
    tests failed there with ``W48 Weg2RingFormMismatch: ... the checkpoint
    terms of '<model path>' could not be read``.  The replay is hermetic with
    respect to NVML, CUDA, /dev/shm and the boot -- but not with respect to the
    model directory, so the tests that need it say so and SKIP instead of
    failing for a reason that is not about the code they grade.
    """
    path = RECORDED["checkpoint_dependency"]["model_path"]
    return bool(glob.glob(os.path.join(path, "*.safetensors")))


NEEDS_CHECKPOINT = unittest.skipUnless(
    checkpoint_present(),
    "the checkpoint named by the recorded P argv is not on this box -- the W48 "
    "form gate cannot be armed (see recorded.json checkpoint_dependency)",
)


class _NoNvml:
    """Any attribute touch is a boot-only dependency reaching into the desk."""

    def __getattr__(self, name):  # pragma: no cover - the point is to raise
        raise AssertionError(
            f"the desk replay reached NVML (pynvml.{name}) -- it must run on "
            "recorded inputs only"
        )


class TestTheFixtureIsFaithful(CustomTestCase):
    """A replay is worth exactly what its inputs are worth.  These are the
    fidelity pins; every wall below rests on them."""

    def test_the_fixture_is_small_and_holds_no_ring(self):
        biggest = max(
            (os.path.getsize(os.path.join(dp, fn)), os.path.join(dp, fn))
            for dp, _dn, fns in os.walk(FIXTURES) for fn in fns
        )
        total = sum(os.path.getsize(os.path.join(dp, fn))
                    for dp, _dn, fns in os.walk(FIXTURES) for fn in fns)
        self.assertLess(biggest[0], 1 << 20, f"fixture file too big: {biggest}")
        self.assertLess(total, 1 << 21, f"fixture tree grew to {total} bytes")

    def test_every_fixture_file_is_CARRIED_BY_THE_REPO(self):
        """A fixture the repo does not carry is a test that works only on the
        box that built it.

        This is not hypothetical: the repo root ignores ``*.log``
        (``.gitignore:62``), and the first ``git add`` of this tree committed
        the sidecar and SILENTLY SKIPPED all three recorded boot logs -- the
        exact three files ``ring_table.solve`` discovers its source boot by.
        On a fresh clone the replay would have found no table and half of this
        file would have died for a reason having nothing to do with the code
        it grades.  ``ring_evidence/.gitignore`` un-ignores them; this test is
        what makes that structural rather than remembered.
        """
        tracked = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "ls-files", "--", FIXTURES],
            capture_output=True, text=True, check=False,
        )
        if tracked.returncode != 0:
            self.skipTest(f"not a git checkout: {tracked.stderr.strip()[:120]}")
        listed = {os.path.basename(p) for p in tracked.stdout.split()}
        on_disk = {fn for _dp, _dn, fns in os.walk(FIXTURES) for fn in fns
                   if fn != ".gitignore"}
        missing = sorted(on_disk - listed)
        self.assertEqual(
            missing, [],
            f"fixture file(s) present on this box but NOT tracked: {missing} "
            "-- most likely the root *.log ignore; see ring_evidence/.gitignore",
        )

    @NEEDS_CHECKPOINT
    def test_solve_reproduces_the_boots_own_ring_bytes(self):
        table, reason = solve_recorded_ring()
        self.assertIsNotNone(table, f"no table from the fixture: {reason}")
        rec = RECORDED["ring_table"]
        self.assertEqual(table.total_h_bytes, rec["ring_bytes"])
        self.assertEqual(table.total_span1_bytes, rec["ring_span1_bytes"])
        self.assertIn(rec["boot_stem"], reason.split("\n")[0])

    @NEEDS_CHECKPOINT
    def test_solve_reproduces_the_boots_own_per_card_L6_rows(self):
        """Not only the total: the per-card H and span1 of XSN14's three
        ``WEG2-HOST-LEDGER RING`` lines.  A total can match while two cards
        swap, which is precisely the nvml-order-vs-ordinal-order defect this
        fixture was built through."""
        table, _reason = solve_recorded_ring()
        h = {c.uuid: c.h_mib for c in table.cards}
        span1 = {c.uuid: c.span1_mib for c in table.cards}
        self.assertEqual(h, {k: int(v) for k, v in
                             RECORDED["ring_table"]["per_card_h_mib"].items()})
        self.assertEqual(span1, {k: int(v) for k, v in
                                 RECORDED["ring_table"]["per_card_span1_mib"].items()})

    def test_the_hermetic_solve_reproduces_its_own_recorded_answer(self):
        """THE RING-SOLVE PIN THAT RUNS EVERYWHERE.  With the W48 form gate
        UNARMED no checkpoint is read, so this one holds on the remote desk
        too -- and it still guards the whole parse surface of the extracted
        evidence (chunk bytes, flip tags, the ordinal map, the sidecar), which
        is what a regression in ``solve`` or a lossy extraction would move.

        Its number is NOT the boot's: the boot armed the gate.  Both are
        recorded, separately and with the reason, so neither can be quoted as
        the other.
        """
        table, reason = solve_recorded_ring(form_gate=False)
        rec = RECORDED["ring_table"]
        self.assertIsNotNone(table, f"no table from the fixture: {reason}")
        self.assertEqual(table.total_h_bytes, rec["no_form_gate_h_bytes"])
        self.assertEqual(table.total_span1_bytes, rec["no_form_gate_span1_bytes"])
        self.assertIn(rec["boot_stem"], reason.split("\n")[0])
        self.assertNotEqual(rec["no_form_gate_h_bytes"], rec["ring_bytes"],
                            "the two recorded answers must stay distinguishable")

    def test_the_checkpoint_dependency_is_declared_not_discovered(self):
        """The dependency that cost a remote run is now DATA: the path, the
        function that reads it, the call that reaches it, and what happens
        without it.  A box fact that is not written down is a test that passes
        only where it was written."""
        dep = RECORDED["checkpoint_dependency"]
        for key in ("model_path", "read_by", "reached_from", "consequence", "owed"):
            self.assertIn(key, dep)
        self.assertIn("--model-path", " ".join(recorded_p_argv()))
        self.assertIn(dep["model_path"], " ".join(recorded_p_argv()),
                      "the declared checkpoint path is not the one the recorded "
                      "argv actually names")

    @NEEDS_CHECKPOINT
    def test_the_p_argv_is_an_input_and_not_decoration(self):
        """Dropping it changes the answer, so it belongs in the fixture.
        Measured while building this: 46133149696 with it, 46888124416
        without (a 720 MiB error that would have read as a solver drift)."""
        with_argv, _ = solve_recorded_ring()
        without, _ = ring_table.solve(
            recorded_cards(), RING_EVIDENCE,
            RECORDED["ring_table"]["boot_stem"], p_argv=None)
        self.assertIsNotNone(without)
        self.assertNotEqual(with_argv.total_h_bytes, without.total_h_bytes)

    @NEEDS_CHECKPOINT
    @unittest.skipUnless(os.path.isdir(FULL_EVIDENCE_DIR),
                         "the real evidence dir is not on this box")
    def test_the_extract_equals_the_full_evidence_dir(self):
        """THE LOSSLESSNESS PROOF for the 48 MB -> 151 KB reduction."""
        small, _ = solve_recorded_ring()
        full, _ = solve_recorded_ring(FULL_EVIDENCE_DIR)
        self.assertIsNotNone(full, "the full evidence dir yielded no table")
        self.assertEqual(
            (small.total_h_bytes, small.total_span1_bytes),
            (full.total_h_bytes, full.total_span1_bytes),
            "the extracted fixture and the full evidence dir disagree -- the "
            "extraction dropped a line the solver reads",
        )

    def test_the_census_fixture_names_the_recorded_source_boot(self):
        census = xr.load_census(CENSUS)
        self.assertIn(RECORDED["ring_table"]["boot_stem"], census.provenance)
        self.assertEqual(
            sorted(census.cards),
            sorted(c["uuid"] for c in RECORDED["cards_ordinal_order"]),
        )


class TestWallXsn13RingAbsentContradiction(CustomTestCase):
    """WALL 1 -- XSN13's dry-run STOP, at the desk, on XSN13's own numbers."""

    def _price(self, *, ring_absent: bool):
        return host_ledger.price(
            memtotal_bytes=APPROX_MEMTOTAL_BYTES,
            memavail_bytes=APPROX_MEMAVAIL_BYTES,
            s_gb=1, m_mib=600,
            ring_bytes=RECORDED["ring_table"]["ring_bytes"],
            ring_span1_bytes=RECORDED["ring_table"]["ring_span1_bytes"],
            ring_absent_by_design=ring_absent,
        )

    def test_the_wall_fires_on_the_recorded_inputs(self):
        """The pre-B4f DECLARATION (ring absent because the weight source is
        ``exchange``) against the ring the same run armed."""
        wall = RECORDED["wall_xsn13_ring_absent_contradiction"]
        with self.assertRaises(ValueError) as ctx:
            self._price(ring_absent=True)
        msg = str(ctx.exception)
        self.assertIn(wall["message_head"], msg)
        self.assertIn(str(RECORDED["ring_table"]["ring_bytes"]), msg)
        self.assertIn(str(RECORDED["ring_table"]["ring_span1_bytes"]), msg)
        self.assertNotIsInstance(ctx.exception, host_ledger.Weg2HostLedgerRefused)

    def test_the_fixed_predicate_does_not_declare_the_ring_absent(self):
        """B4f, and the reason the wall is closed on ``dfceb7004e``: ring
        absence is a property of the INJECT ARM."""
        self.assertFalse(L.ring_absent_by_design(
            L.WEIGHT_SOURCE_EXCHANGE, wx.INJECT_SHADOW))
        self.assertTrue(L.ring_absent_by_design(
            L.WEIGHT_SOURCE_EXCHANGE, wx.INJECT_AUTHORITATIVE))
        for source in (L.WEIGHT_SOURCE_DEFAULT, "shadow"):
            for inject in (wx.INJECT_SHADOW, wx.INJECT_AUTHORITATIVE):
                self.assertFalse(L.ring_absent_by_design(source, inject),
                                 f"{source}/{inject} must keep its ring")

    def test_the_tip_prices_xsn13s_inputs_without_the_contradiction(self):
        """The GREEN half: the same recorded ring, the arm strings XSN13
        actually ran, and the predicate the tip owns -> no contradiction.

        ``price`` may still refuse by NAME (W20/W21) on the approximate box
        readings; that is a different verdict and the assertion says so.
        """
        wall = RECORDED["wall_xsn13_ring_absent_contradiction"]
        absent = L.ring_absent_by_design(wall["weight_source"], wall["inject_mode"])
        self.assertFalse(absent)
        try:
            self._price(ring_absent=absent)
        except ValueError as exc:
            self.assertNotIsInstance(exc, host_ledger.Weg2HostLedgerRefused)
            self.fail(f"the tip still hits a contradiction: {exc}")
        except host_ledger.Weg2HostLedgerRefused:
            pass          # a NAMED host refusal, not the contradiction

    def test_the_contradiction_precedes_every_host_reading(self):
        """Why approximate box readings are sound here: the guard raises with
        host numbers that could not fund anything at all."""
        with self.assertRaises(ValueError) as ctx:
            host_ledger.price(
                memtotal_bytes=1, memavail_bytes=1, s_gb=1, m_mib=600,
                ring_bytes=RECORDED["ring_table"]["ring_bytes"],
                ring_span1_bytes=RECORDED["ring_table"]["ring_span1_bytes"],
                ring_absent_by_design=True,
            )
        self.assertIn(
            RECORDED["wall_xsn13_ring_absent_contradiction"]["message_head"],
            str(ctx.exception))

    def test_the_contradiction_guard_is_where_the_record_says_it_is(self):
        """``host_ledger.py:1832 price`` -- the record's call chain, pinned
        structurally so a move renames this test's reason rather than
        silently changing what the replay drives."""
        src = os.path.join(os.path.dirname(os.path.abspath(host_ledger.__file__)),
                           "host_ledger.py")
        with open(src, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "price"), None)
        self.assertIsNotNone(fn, "host_ledger.price is gone")
        raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
        head = RECORDED["wall_xsn13_ring_absent_contradiction"]["message_head"]
        self.assertTrue(
            any(head in ast.unparse(n) for n in raises),
            "the contradiction raise left price(); the replay's wall 1 no "
            "longer drives the site the XSN13 traceback names",
        )


class TestWallXsn14W19DormantResidue(CustomTestCase):
    """WALL 2 -- W19 at the desk: the launcher's reserve against XSN14's own
    per-card measurement, under both arm strings."""

    def _reserve(self, weight_source: str, transport: str = "bar1") -> dict[str, int]:
        slack = L.reserve_slack_mib(transport)
        return {c.uuid: L.dc_measured_d_mib(c, weight_source) + slack
                for c in recorded_cards()}

    @staticmethod
    def _over(measured: dict[str, int], reserve: dict[str, int]) -> dict[str, int]:
        """The front's own predicate (``front.py:4071``), one line, pinned
        against the front's source by
        :meth:`test_the_predicate_is_the_fronts_own`."""
        return {u: m - reserve[u] for u, m in measured.items()
                if reserve.get(u) is not None and m > reserve[u]}

    def _measured(self) -> dict[str, int]:
        return {r["uuid"]: int(r["measured_mib"])
                for r in RECORDED["wall_xsn14_w19_dormant_residue"]["rows"]}

    def test_the_wall_fires_on_the_recorded_measurement(self):
        rows = RECORDED["wall_xsn14_w19_dormant_residue"]["rows"]
        reserve = self._reserve(L.WEIGHT_SOURCE_DEFAULT)
        self.assertEqual(reserve, {r["uuid"]: int(r["reserve_mib"]) for r in rows},
                         "the serving-form reserve no longer reproduces XSN14's")
        over = self._over(self._measured(), reserve)
        self.assertEqual(over, {r["uuid"]: int(r["excess_mib"]) for r in rows},
                         "the recorded excesses 602/792/602 no longer fall out")
        self.assertEqual(len(over), 3, "W19 fired on all three cards")

    def test_the_exchange_form_is_green_on_the_same_measurement(self):
        """B4h on the tip: the reserve follows the FORM, so XSN14's own
        measurement no longer exceeds it -- which is what XSN15 read as W19
        genuine 0 on both groups."""
        reserve = self._reserve(L.WEIGHT_SOURCE_EXCHANGE)
        over = self._over(self._measured(), reserve)
        self.assertEqual(over, {}, f"W19 would still fire on the exchange form: {over}")

    def test_the_form_selector_is_the_only_difference(self):
        """Same cards, same measurement, one arm string -- the two verdicts
        differ.  A replay that could not tell the forms apart would prove
        nothing about either."""
        serving = self._reserve(L.WEIGHT_SOURCE_DEFAULT)
        exchange = self._reserve(L.WEIGHT_SOURCE_EXCHANGE)
        self.assertNotEqual(serving, exchange)
        for uuid, mib in exchange.items():
            self.assertGreater(mib, serving[uuid])

    def test_the_reserve_is_the_measurement_plus_the_named_slack(self):
        slack = L.reserve_slack_mib("bar1")
        self.assertEqual(slack, L.DC_RESERVE_SLACK_MIB)
        for c in recorded_cards():
            self.assertEqual(
                self._reserve(L.WEIGHT_SOURCE_DEFAULT)[c.uuid],
                L.dc_measured_d_mib(c, L.WEIGHT_SOURCE_DEFAULT) + slack)

    def test_an_unnamed_board_refuses_rather_than_borrowing_a_number(self):
        other = L.Card(nvml_index=3, uuid="GPU-ffffffff", name="NVIDIA L4", total_mib=24564)
        with self.assertRaises(L.Weg2LaunchRefused) as ctx:
            L.dc_measured_d_mib(other, L.WEIGHT_SOURCE_EXCHANGE)
        self.assertIn("W19", str(ctx.exception))

    def test_the_predicate_is_the_fronts_own(self):
        """Structural pin: the front's W19 site compares the MEASURED value
        against ``self.dc_reserve[u]`` with ``>``.  If that changes, this
        replay's one-line predicate is no longer the front's and must be
        re-read rather than trusted."""
        src = os.path.join(os.path.dirname(os.path.abspath(L.__file__)), "front.py")
        with open(src, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        owners = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(isinstance(c, ast.Call)
                    and getattr(c.func, "attr", "") == "do_stop"
                    and c.args
                    and isinstance(c.args[0], ast.Constant)
                    and str(c.args[0].value).startswith("W19")
                    for c in ast.walk(n))
        ]
        self.assertTrue(owners, "front.py no longer raises W19 through do_stop")
        text = "\n".join(ast.unparse(n) for n in owners)
        self.assertIn("self.dc_reserve[u]", text)
        self.assertIn("m > self.dc_reserve[u]", text)


class TestWallW71NoCensus(CustomTestCase):
    """WALL 3 -- W71 at the desk: the arming refusal with no census, and the
    full ARMED line with the recorded one."""

    def test_no_census_path_refuses_by_name(self):
        with self.assertRaises(xr.Weg2XchgResidencyUnarmable) as ctx:
            xr.load_census("")
        self.assertIn(RECORDED["wall_w71_no_census"]["message_head"],
                      str(ctx.exception))

    def test_a_missing_census_file_refuses_by_name(self):
        with self.assertRaises(xr.Weg2XchgResidencyUnarmable) as ctx:
            xr.load_census(os.path.join(FIXTURES, "does-not-exist.json"))
        self.assertIn("W71 Weg2XchgResidencyUnarmable", str(ctx.exception))

    def test_the_arm_reproduces_the_recorded_armed_line(self):
        """``prepare_weight_exchange`` is pure (census + the cards' NVML
        totals in, one ARMED line out), so the boot's own acceptance line is
        re-derivable at the desk -- peaks, frees, wave1_ok, floor, region and
        ring_H together."""
        lines: list[str] = []
        res = L.prepare_weight_exchange(
            recorded_cards(), lines.append, L.WEIGHT_SOURCE_EXCHANGE, CENSUS,
            RECORDED["epoch_xsn14"], int(RECORDED["ring_table"]["ring_h_mib"]),
            floor_mib=float(RECORDED["arming_floor_mib"]),
        )
        self.assertIsNotNone(res)
        armed = [ln for ln in lines if ln.startswith("WEG2-XCHG-ARMED")]
        self.assertEqual(len(armed), 1, "expected exactly one ARMED line")
        got = armed[0].split(" -- provenance:")[0].strip()
        self.assertEqual(got, RECORDED["armed_line_xsn14"].strip())

    def test_the_ring_arm_has_no_subject(self):
        lines: list[str] = []
        self.assertIsNone(L.prepare_weight_exchange(
            recorded_cards(), lines.append, L.WEIGHT_SOURCE_DEFAULT, "",
            RECORDED["epoch_xsn14"], 0))
        self.assertEqual(lines, [])

    def _arm_with_dormant_delta(self, delta: int):
        """Run the arm on the recorded census with every card's dormant
        reading shifted by ``delta`` MiB.  Returns ``(armed_line, refusal)``,
        exactly one of which is ``None``."""
        with open(CENSUS, encoding="utf-8") as fh:
            blob = json.load(fh)
        mutated = copy.deepcopy(blob)
        for entry in mutated["cards"].values():
            entry["dormant_proc_used_mib"] = int(entry["dormant_proc_used_mib"]) + delta
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"census_delta_{delta}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(mutated, fh)
            lines: list[str] = []
            try:
                L.prepare_weight_exchange(
                    recorded_cards(), lines.append, L.WEIGHT_SOURCE_EXCHANGE,
                    path, RECORDED["epoch_xsn14"],
                    int(RECORDED["ring_table"]["ring_h_mib"]),
                    floor_mib=float(RECORDED["arming_floor_mib"]))
            except xr.Weg2XchgResidencyUnarmable as exc:
                return None, str(exc)
        armed = [ln for ln in lines if ln.startswith("WEG2-XCHG-ARMED")]
        self.assertEqual(len(armed), 1, "expected exactly one ARMED line")
        return armed[0], None

    def test_a_mutated_census_refuses_by_name_CAN_FAIL(self):
        """CAN-FAIL on the danger direction: the replay must not be a
        constant.  The operator's own suggested mutation -- the census's
        dormant reading +1000 MiB per card -- must produce a NAMED refusal,
        and it does: W71, with the number of (card x direction) cases that
        no longer fit.  A fixture that answered ARMED either way would prove
        nothing about the boot it claims to replay.

        The GRADED escalation is the second half, and it is why this test
        asserts the case COUNT rather than merely "W71 appeared": +1000 MiB
        breaks ONE of the six cases, +8000 MiB breaks all SIX.  A refusal
        whose text did not move with the input would be a constant string
        dressed as arithmetic.
        """
        armed, refusal = self._arm_with_dormant_delta(1000)
        self.assertIsNone(armed, "a +1000 MiB census must not still arm")
        self.assertIn("W71 Weg2XchgResidencyUnarmable", refusal)
        self.assertIn("1 (card x direction) case(s)", refusal)

        armed, refusal_big = self._arm_with_dormant_delta(8000)
        self.assertIsNone(armed, "a +8000 MiB census must not still arm")
        self.assertIn("6 (card x direction) case(s)", refusal_big)

    def test_a_downward_census_drift_moves_the_armed_line_CAN_FAIL(self):
        """The other direction of the same proof: shift the census DOWN and
        the arm still succeeds, but the peaks the ARMED line prints must
        MOVE -- they are a function of the census, not constants baked into
        the line."""
        armed, refusal = self._arm_with_dormant_delta(-500)
        self.assertIsNone(refusal, f"a -500 MiB census must still arm: {refusal}")
        self.assertNotEqual(armed.split(" -- provenance:")[0].strip(),
                            RECORDED["armed_line_xsn14"].strip())
        self.assertIn("wave1_ok=6/6", armed)


class TestTheReplayNeedsNoBootAndNoNvml(CustomTestCase):
    """The hermeticity claim, proven rather than asserted: the whole replay
    runs with a ``pynvml`` that explodes on any attribute touch."""

    def test_all_three_walls_run_without_nvml(self):
        saved = sys.modules.get("pynvml")
        sys.modules["pynvml"] = _NoNvml()
        try:
            # form_gate=False: the armed gate reads the checkpoint, which is a
            # box fact, not a recorded input -- and this test is about NVML.
            table, _ = solve_recorded_ring(form_gate=False)
            self.assertEqual(table.total_h_bytes,
                             RECORDED["ring_table"]["no_form_gate_h_bytes"])
            with self.assertRaises(ValueError):
                host_ledger.price(
                    memtotal_bytes=APPROX_MEMTOTAL_BYTES,
                    memavail_bytes=APPROX_MEMAVAIL_BYTES, s_gb=1, m_mib=600,
                    ring_bytes=RECORDED["ring_table"]["ring_bytes"],
                    ring_span1_bytes=RECORDED["ring_table"]["ring_span1_bytes"],
                    ring_absent_by_design=True)
            self.assertTrue(L.dc_measured_d_mib(recorded_cards()[0],
                                                L.WEIGHT_SOURCE_EXCHANGE) > 0)
            lines: list[str] = []
            self.assertIsNotNone(L.prepare_weight_exchange(
                recorded_cards(), lines.append, L.WEIGHT_SOURCE_EXCHANGE,
                CENSUS, RECORDED["epoch_xsn14"],
                int(RECORDED["ring_table"]["ring_h_mib"]),
                floor_mib=float(RECORDED["arming_floor_mib"])))
        finally:
            if saved is None:
                sys.modules.pop("pynvml", None)
            else:
                sys.modules["pynvml"] = saved

    def test_the_replay_writes_nothing_into_dev_shm(self):
        before = set(os.listdir("/dev/shm")) if os.path.isdir("/dev/shm") else set()
        solve_recorded_ring(form_gate=False)
        lines: list[str] = []
        L.prepare_weight_exchange(
            recorded_cards(), lines.append, L.WEIGHT_SOURCE_EXCHANGE, CENSUS,
            RECORDED["epoch_xsn14"], int(RECORDED["ring_table"]["ring_h_mib"]),
            floor_mib=float(RECORDED["arming_floor_mib"]))
        after = set(os.listdir("/dev/shm")) if os.path.isdir("/dev/shm") else set()
        self.assertEqual(sorted(after - before), [],
                         "the replay created /dev/shm entries -- it is no "
                         "longer a desk replay")


if __name__ == "__main__":
    unittest.main()
