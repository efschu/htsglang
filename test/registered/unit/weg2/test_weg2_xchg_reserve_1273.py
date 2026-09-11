# SPDX-License-Identifier: Apache-2.0
"""B4i: the `WEG2-XCHG-RESERVE` line -- EMITTED, and on ONE INSTRUMENT.

**THIS FILE'S B4g/B4h ARITHMETIC IS RETRACTED.  Read this paragraph before any
older text that quotes 773 / 935 / 773.**  B4g priced the exchange form's
dormant reserve as *census dormant + named terms* and reported the remainder
against the measured residue as an UNATTRIBUTED residual of 773 / 935 / 773 MiB
(162 on the 5090 alone).  That residual does not exist.  It was manufactured by
subtracting HOST populations from a DEVICE reading: the staging region
(`xchg.bin`, 385 MiB) and the on-card deposit slots (3x32 = 96 MiB) are both
`/dev/shm` allocations, while `WEG2-DC` is per-process NVML at sleep.  They were
never inside `measured`, so removing them invented the gap exactly:
1254 - 481 = 773 and 1416 - 481 = 935.

SAME-INSTRUMENT ARITHMETIC, which is the only kind this line may print
(boot weg2xsn15's per-card term table):

    XSN15  D @ sleep   3084 / 2590 / 2588      WEG2-DC, exchange form
    sn5b   D @ sleep   1668 / 1334 / 1334      WEG2-DC, serving form (the census)
    delta              1416 / 1256 / 1254
    weights_draft tag  1440 / 1280 / 1280      tms_tag_bytes, resident, in_family=no
    unexplained          -24 /  -24 /   -26    instrument margin

So the difference between the two readings IS the resident draft tag, and the
answer is B4k (the draft head joins the weights family and is exchanged like
every other layer), not a wider reserve and not a residual field.

**AND THE LINE WAS NEVER EMITTED.**  Boot weg2xsn15 read `WEG2-XCHG-RESERVE`
**0 times in all four logs** while `W19` was genuine 0 -- so the constant was in
force and the line that was to report it reached no log at all.  The root is the
same class as W84's (a guard that computes and never raises) one level down:
`xchg_form_dormant_reserve` had **zero production call sites**.  It was called
only from this file and from `test_weg2_w19_form_residue_1273.py`, so every
suite was green over a function no boot ever ran.  The fix wires it into
`main()` BEFORE the dry-return, so a dry-run prints it too, and the reachability
is pinned STRUCTURALLY below -- an execution test on the function alone is
exactly the proof that was already green while the boot printed nothing.

RED ON `dfceb7004e`: the line carries `dormant_census_mib=`/`region_mib=`/
`oncard_slots_mib=`/`priced_mib=`/`residual_unattributed_mib=` instead of
`measured_mib=`/`reserved_mib=`, the function returns a three-tuple whose third
element is the invented residual, and `main()` does not call it.
"""

import ast
import inspect
import json
import os
import tempfile
import textwrap
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import xchg_residency
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
SM1 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
SM2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"

#: weg2xsn14's own epoch-0 readings (the constant), and the census's own
#: dormant numbers (the serving form).  BOTH are `WEG2-DC` at sleep.
MEASURED_XCHG = {SM1: 2588, BIG: 3084, SM2: 2588}
CENSUS_DORMANT = {SM1: 1334, BIG: 1668, SM2: 1334}
SERVING_RESERVE = {SM1: 1986, BIG: 2292, SM2: 1986}

CARDS = [
    launcher.Card(1, BIG, "NVIDIA GeForce RTX 5090", 32607),
    launcher.Card(0, SM1, "NVIDIA GeForce RTX 3080", 20480),
    launcher.Card(2, SM2, "NVIDIA GeForce RTX 3080", 20480),
]
FAMILY = [f"weights_{k}" for k in range(8)] + ["weights"]


def _census(path, dormant=None, drop=()):
    dormant = dormant or CENSUS_DORMANT
    cards = {}
    for u in (BIG, SM1, SM2):
        if u in drop:
            continue
        cards[u] = {
            "tags": {g: {t: 100 for t in FAMILY} for g in ("P", "D")},
            "dormant_proc_used_mib": dormant[u],
            "dormant_source": f"READING: boot weg2sn5b front WEG2-DC peak P/D for {u[:12]}",
        }
    blob = {"cards": cards, "waves": [FAMILY], "provenance": "test"}
    with open(path, "w") as fh:
        json.dump(blob, fh)
    return path


class _Collect:
    def __init__(self):
        self.lines = []

    def __call__(self, line):
        self.lines.append(str(line))


# ---------------------------------------------------------------------------
# AST helpers, the #1294 shape (a pin on a helper is not a pin on its caller).
# ---------------------------------------------------------------------------


def _fn_ast_with_offset(fn):
    src, start = inspect.getsourcelines(fn)
    return ast.parse(textwrap.dedent("".join(src))), start - 1


def _callee_name(node):
    f = node.func
    return f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)


def _calls_in(tree, name):
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _callee_name(n) == name
    ]


def _dry_return_line(tree):
    fn = tree.body[0]
    for n in fn.body:
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "dry":
            for stmt in reversed(n.body):
                if isinstance(stmt, ast.Return):
                    return stmt.lineno
    raise AssertionError("main() no longer has a top-level `if dry:` branch")


# ===========================================================================
# THE LINE IS ON ONE INSTRUMENT.
# ===========================================================================


class TheReserveLineReportsTwoReadingsOfOneInstrument(CustomTestCase):
    def _run(self, **kw):
        p = _census(os.path.join(tempfile.mkdtemp(), "c.json"), **kw)
        log = _Collect()
        out, lines = launcher.xchg_form_dormant_reserve(CARDS, p, log=log)
        self.assertEqual(lines, log.lines, "the lines returned must be the "
                                           "lines logged, or one of the two "
                                           "is a second bookkeeping")
        return out, lines

    def test_reserved_is_the_forms_own_constant_through_the_one_selector(self):
        """`reserved_mib` must be what W19 actually grades against, read
        through `dc_measured_d_mib` -- a second copy of the triple here is the
        parallel-object defect the selector exists to prevent."""
        out, _lines = self._run()
        for u, card in ((BIG, CARDS[0]), (SM1, CARDS[1]), (SM2, CARDS[2])):
            self.assertEqual(
                out[u],
                launcher.dc_measured_d_mib(card, launcher.WEIGHT_SOURCE_EXCHANGE),
            )
            self.assertEqual(out[u], MEASURED_XCHG[u])

    def test_both_numbers_on_the_line_are_WEG2_DC_AT_SLEEP(self):
        """THE FIX FOR THE RETRACTED RESIDUAL, as a property of the line.

        The line may carry only readings of ONE instrument, so that the
        difference a reader takes between them is a difference of like for
        like.  `measured_mib` is the census's own `WEG2-DC` dormant reading
        (the serving form), `reserved_mib` is the same instrument on the
        exchange form (weg2xsn14).  Their delta is the resident draft tag.
        """
        _out, lines = self._run()
        self.assertEqual(len(lines), 3)
        for ln in lines:
            self.assertIn("WEG2-XCHG-RESERVE", ln)
            self.assertIn("reserved_mib=", ln)
            self.assertIn("source=measured:weg2xsn14", ln)
            self.assertIn("measured_mib=", ln)
            self.assertIn("instrument=WEG2-DC-at-sleep", ln)
            self.assertIn("census_source=READING", ln)

    def test_the_host_terms_and_the_invented_residual_are_GONE(self):
        """The retraction, asserted so it cannot come back by copy-paste.

        `region_mib` and `oncard_slots_mib` are `/dev/shm` bytes.  Printing
        them on a line whose other numbers are per-process NVML is what
        produced a 773/935/773 MiB finding out of nothing, and a reader who
        subtracts them again gets the same non-number.
        """
        _out, lines = self._run()
        for ln in lines:
            for banned in (
                "region_mib=",
                "oncard_slots_mib=",
                "residual_unattributed_mib=",
                "priced_mib=",
                "dormant_census_mib=",
            ):
                self.assertNotIn(banned, ln, f"{banned} is retracted: {ln}")

    def test_the_census_reading_is_reported_not_subtracted(self):
        """The census stays on the line as the SERVING form's reading of the
        same instrument -- it is the comparison's other end, not a term."""
        _out, lines = self._run()
        for u in (BIG, SM1, SM2):
            self.assertTrue(
                any(f"measured_mib={CENSUS_DORMANT[u]}" in ln for ln in lines),
                f"{u}: the census's own WEG2-DC reading is not on any line",
            )

    def test_a_card_missing_from_the_census_refuses(self):
        """The serving constants must NOT stand in -- they contain none of the
        exchange lane's residency, so borrowing them is how weg2xsn14 got its
        reserve."""
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            self._run(drop=(BIG,))
        self.assertIn("31d7ef41", str(caught.exception))
        self.assertIn("serving constants must not stand in", str(caught.exception))

    def test_the_serving_constants_are_not_hand_raised(self):
        """Operator ruling + the VRAM-corridor law: launcher.py:216 untouched."""
        self.assertEqual(launcher.DC_MEASURED_D_5090_MIB, 2228)
        self.assertEqual(launcher.DC_MEASURED_D_3080_MIB, 1922)
        self.assertEqual(launcher.DC_RESERVE_SLACK_MIB, 64)

    def test_the_exchange_reserve_is_ABOVE_the_serving_reserve_on_every_card(self):
        """The direction one expects and the direction the constant has: the
        exchange form holds MORE dormant residency, so its reserve is larger.
        (B4g's opposite claim came from comparing the CENSUS to the constant,
        which is two forms of one instrument, not two reserves.)"""
        out, _lines = self._run()
        for u in (BIG, SM1, SM2):
            self.assertGreater(out[u], SERVING_RESERVE[u], u)


class TheDeltaIsTheResidentDraftTag(CustomTestCase):
    """The attribution, asserted so the next reader inherits it rather than
    the retracted residual.  Numbers from boot weg2xsn15's term table."""

    #: `tms_tag_bytes` for `weights_draft` on group D, XSN15, per card.
    DRAFT_TAG_MIB = {SM1: 1280, BIG: 1440, SM2: 1280}

    def test_the_same_instrument_delta_is_the_draft_tag_within_the_margin(self):
        for u in (BIG, SM1, SM2):
            delta = MEASURED_XCHG[u] - CENSUS_DORMANT[u]
            self.assertLessEqual(
                abs(delta - self.DRAFT_TAG_MIB[u]), 32,
                f"{u}: WEG2-DC delta {delta} vs resident weights_draft "
                f"{self.DRAFT_TAG_MIB[u]} -- if this drifts, the attribution "
                f"is no longer the draft tag and B4k's premise moved",
            )

    def test_the_5090_asymmetry_is_the_draft_tags_own_asymmetry(self):
        """162 MiB was reported as 'surviving every uniform subtraction'.  On
        one instrument the asymmetry is 1416-1254 = 162 and the draft tag's own
        asymmetry is 1440-1280 = 160: the same number, not a remainder."""
        dc = MEASURED_XCHG[BIG] - CENSUS_DORMANT[BIG]
        dc -= MEASURED_XCHG[SM2] - CENSUS_DORMANT[SM2]
        tag = self.DRAFT_TAG_MIB[BIG] - self.DRAFT_TAG_MIB[SM2]
        self.assertLessEqual(abs(dc - tag), 8, f"dc={dc} tag={tag}")


# ===========================================================================
# THE EMISSION DEFECT: a function with no production caller.
# ===========================================================================


class TheLineReachesTheLog(CustomTestCase):
    """Boot weg2xsn15: 0 `WEG2-XCHG-RESERVE` lines in four logs.  The
    execution test above was green throughout, which is the whole point of
    pinning the CALL SITE structurally as well."""

    def test_main_calls_the_emitter(self):
        tree, offset = _fn_ast_with_offset(L.main)
        calls = _calls_in(tree, "xchg_form_dormant_reserve")
        self.assertTrue(
            calls,
            "main() does not call xchg_form_dormant_reserve() -- that is the "
            "weg2xsn15 defect: the pricer had ZERO production call sites and "
            "every suite was green over it",
        )
        self.assertEqual(
            len(calls), 1, "one call site, or the line prints twice per card"
        )

    def test_the_call_passes_log_so_the_line_is_PRINTED(self):
        """A call that computes and discards is the W84 shape: the value is
        recorded, nothing reaches a log, and no test can tell."""
        tree, _offset = _fn_ast_with_offset(L.main)
        call = _calls_in(tree, "xchg_form_dormant_reserve")[0]
        kwargs = {k.arg for k in call.keywords}
        self.assertIn("log", kwargs, f"log= not passed: {kwargs}")

    def test_the_call_is_BEFORE_the_dry_return_so_a_dry_run_prints_it(self):
        """The dry-run is the desk's only pre-boot sight of this arithmetic
        (operator's proposed change 5).  A call after the dry-return would make
        every dry-run silent about the reserve it is about to grade against."""
        tree, _offset = _fn_ast_with_offset(L.main)
        call = _calls_in(tree, "xchg_form_dormant_reserve")[0]
        self.assertLess(
            call.lineno, _dry_return_line(tree),
            "the emitter must be called before main()'s dry-return",
        )

    def test_the_guard_does_not_truthy_check_the_census_path(self):
        """THE #872 SHAPE, and the one that would re-create this very defect.

        ``and ns.weg2_xchg_census`` would make a census-less exchange launch
        print nothing instead of refusing.  The census is mandatory under this
        arm and ``xchg_residency.load_census`` is its ONE authority (W71 by
        name); a second, silent check here is how a line goes missing.
        """
        tree, _offset = _fn_ast_with_offset(L.main)
        call = _calls_in(tree, "xchg_form_dormant_reserve")[0]
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if not any(c is call for stmt in node.body for c in ast.walk(stmt)):
                continue
            attrs = {
                n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)
            }
            self.assertNotIn(
                "weg2_xchg_census", attrs,
                "the guard truthy-checks the census path, so a census-less "
                "exchange launch would print nothing instead of raising W71",
            )

    def test_the_emitter_is_armed_on_the_exchange_form_only(self):
        """The ring form allocates none of this and must pay nothing: the call
        sits under the form predicate, not beside it."""
        tree, _offset = _fn_ast_with_offset(L.main)
        call = _calls_in(tree, "xchg_form_dormant_reserve")[0]
        guarded = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if any(c is call for c in ast.walk(node.test)):
                continue
            body_calls = [c for stmt in node.body for c in ast.walk(stmt)]
            if any(c is call for c in body_calls):
                guarded = True
                names = {
                    n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
                }
                attrs = {
                    n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)
                }
                self.assertTrue(
                    {"_xchg_form"} & names
                    or {"weg2_weight_source"} & attrs,
                    f"the guard does not read the weight-source form: "
                    f"names={names} attrs={attrs}",
                )
        self.assertTrue(guarded, "the call is not under any `if` at all")


if __name__ == "__main__":
    unittest.main()
