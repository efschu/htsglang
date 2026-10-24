# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-ADMIT-BOOK 24.10. (Auftrag #212; review-vote-line-single M2 / SOLL S1): the D extend's admission peak.

Every D extend pass runs ``width vote (trim, read post) -> admission (post-wake handbacks: Mamba restore + KV
load-back) -> forward trim (card_free_before = driver free after the admission) -> forward``; the near-OOM law asks
``post - A >= 400`` at the admission instant.

Measured on the six int4ar serving boots (2x 3080, D TP2), chunk-window census (budget_rest.admission_passes; the
vote line is logged only when the cap CHANGES, so the vote-log pairing of extend_passes sees a fraction of the passes
and reported 376 / 390): 33 admission instants under 400, down to 216 MiB (TP1). The admission peak A grows with the
handback count and saturates at the capture set (bs <= 6): max 512 int4ar / 516 INT8 drei / 522 NVFP4 duo.

No trim fixes it (S1 measured a no-op): the int4ar vote trim (static 1104 > 400 + 522) ran before 32 of the 33 low
instants and returned <= 64 MiB (the remaining cache sits in segments with live blocks); over every pass of all three
forms there is no admission pass where the vote did not trim and post < 400 + A. The budget priced no admission term.

The fix prices it: ``D_ADMIT_NEED_MIB = near-OOM 400 + A_peak 522 - post_floor 608 = 314`` (derived from two
measured maxima), booked per card as ``max(vote-line delta, ceil8(need))`` -- the admission and the forward never
coexist, so the larger post, never the sum. Form gate: int4ar only. INT8 drei derives 118 (carried by its vote-line
delta 344 / 400..472), NVFP4 duo 0, NVFP4 dual has no width vote.

Danger directions pinned here:
- OOM: the admission post forgotten, the peak of one form only, the floor from the vote-logged passes only, the
  in-sample shortfall instead of the bound, rounded down, the form gate letting nothing through.
- KV / narrow: the posts summed instead of the larger one, the post booked on a form that does not need it.
"""

import contextlib
import gzip
import json
import os
import sys
import unittest.mock as mock

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import corridor_guard as CG  # noqa: E402
from flliper.srt.pdflip import budget_rest as BR  # noqa: E402
from flliper.srt.pdflip import card_identity as CI  # noqa: E402
from flliper.srt.pdflip import corridor_budget as CB  # noqa: E402
from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import inventory_view as IV  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_27b_int8_tp1corridor_1024 as PRE  # noqa: E402  (same directory: the predecessor's live-boot rig)

FIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "admit_trim_1024")
NEED = 314             # D_ADMIT_NEED_MIB (this file recomputes it from the boots)
BOOKED = 320           # ceil8(314)
A_PEAK = 522           # NVFP4 duo, 5 handbacks
POST_FLOOR = 608       # int4ar TP0, lowest vote post of an admission pass
SINGLE_DELTA = 344     # INT8 drei vote-line delta (VOTE-LINE-SINGLE)
FLOOR_DELTAS = [472, 408, 400]
INT4AR_MODEL = "Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound-ct"
DREI_MODEL = "Qwen3.8-27B-INT8-gdncov-vocabembed"
NVFP4_MODEL = "Qwen3.8-27B-NVFP4-RadixArk"
#: the int4ar boots' D budgets (front.log ``budget D`` lines of fe572f3224 / b9cddabda1): 17480 / 17264
INT4AR_HEAD = [17480, 17264]
INT4AR_DORMANT = {PRE.U_TP1: 494, PRE.U_TP2: 706}
CALIBRATED = ("RTX5090", "RTX3080", "RTX3080")


def _load():
    with gzip.open(os.path.join(FIX_DIR, "admit_passes.json.gz"), "rt") as fh:
        doc = json.load(fh)
    cols = doc["columns"]
    out = {}
    for tag, b in doc["boots"].items():
        per = {}
        for row in b["rows"]:
            d = dict(zip(cols, row))
            per.setdefault(int(d["rank"]), []).append(BR.AdmitPass(
                rank=int(d["rank"]), ts=d["ts"], post=d["post"], post_src=d["post_src"], fwd_before=d["fwd_before"],
                released=float(d["released"]), rows=int(d["rows"]), end=float(d["end"]),
                vote_released=d["vote_released"]))
        out[tag] = (b["form"], per)
    return out


ALL = _load()
FORM = {f: [per for (ff, per) in ALL.values() if ff == f] for f in ("int4ar", "drei", "duo")}
INT4AR_TAGS = tuple(t for t, (f, _) in ALL.items() if f == "int4ar")


def _passes(form):
    return [p for b in FORM[form] for v in b.values() for p in v]


def _form_env(model):
    return F.PdFlipForm(arch="dense", experts="none", draft="dflash", p_draft="cold", kv="paged_dcp", flip="family",
                        vision="transient", profile="qwen27b", model=model).env_value()


@contextlib.contextmanager
def _published(model):
    with mock.patch.dict(os.environ, {F.FORM_ENV: _form_env(model) if model else ""}):
        yield


def _int4ar_cards():
    return L.order_cards([
        L.Card(nvml_index=1, uuid=PRE.U_TP1, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
        L.Card(nvml_index=2, uuid=PRE.U_TP2, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425)])


def _int4ar_pass(lines, model=INT4AR_MODEL, admit="1", single="1"):
    """The int4ar D budget pass as the launcher runs it: the 2-card subset view of the calibrated inventory
    installed, the floors UNMEASURED (UNMEASURED-FALLBACK:1024 on the int4ar boots), the published form's checkpoint."""
    cards = _int4ar_cards()
    env = {BR.VOTE_LINE_SINGLE_ENV: single, BR.ADMIT_ENV: admit}
    floors = PRE._floors(source=CG.FLOOR_SOURCE_FALLBACK, values=PRE.FALLBACK)
    IV.set_active(F.PROFILE_QWEN27B, CALIBRATED, ("RTX3080", "RTX3080"))
    try:
        with mock.patch.object(CB, "floors_for_cards", floors), mock.patch.dict(os.environ, env), _published(model):
            kw = dict(L.booked_rest_kwargs(cards, F.PROFILE_QWEN27B, "D", lines.append))
            budgets = L.budgets_from_dc(
                cards, dict(INT4AR_DORMANT), lines.append, "D",
                charge_driver_carve=L.budget_charges_driver_carve(F.PROFILE_QWEN27B),
                driver_carve_min_total_mib=L.driver_carve_min_total_mib(F.PROFILE_QWEN27B), **kw)
    finally:
        IV.clear_active()
    return budgets, kw


class TheBootsViolateTheEdge(CustomTestCase):
    def test_the_chunk_window_census_sees_every_pass(self):
        ps = _passes("int4ar")
        self.assertEqual(len(ps), 7604)
        low = [p for p in ps if p.fwd_before is not None and p.fwd_before < CG.NEAR_OOM_MIB]
        self.assertEqual(len(low), 33)
        self.assertEqual(sum(1 for p in low if p.rank == 1), 26)
        self.assertEqual(min(p.fwd_before for p in low), 216)
        # INT8 drei and NVFP4 duo hold the edge at their boots' budgets
        for form in ("drei", "duo"):
            self.assertFalse([p for p in _passes(form) if p.fwd_before is not None and p.fwd_before < 400], form)

    def test_the_parser_reads_the_real_lines(self):
        with open(os.path.join(FIX_DIR, "excerpt_223241_D.txt")) as fh:
            text = fh.read()
        got = BR.admission_passes(text)
        tp1 = [(p.post, p.post_src, p.fwd_before, p.admission, p.vote_released, p.rows) for p in got[1]]
        self.assertEqual(tp1, [(874.0, "trim", 446.0, 428.0, 0.0, 256), (704.0, "trim", 218.0, 486.0, 42.0, 128),
                               (880.0, "trim", 448.0, 432.0, 0.0, 256), (708.0, "trim", 224.0, 484.0, 42.0, 59)])
        tp0 = [(p.post, p.post_src, p.fwd_before) for p in got[0]]
        self.assertEqual(tp0, [(1058.0, "vote", 630.0), (888.0, "vote", 402.0), (1032.0, "vote", 600.0),
                               (860.0, "vote", 376.0)])

    def test_the_vote_log_pairing_misses_the_low_rank(self):
        """The instrument trap of the predecessor's figure: the vote line is logged only when the cap changes -- TP1
        logged none in these two post-wake windows, extend_passes pairs nothing there and its minimum is TP0 376,
        while TP1 went to 218 / 224."""
        with open(os.path.join(FIX_DIR, "excerpt_223241_D.txt")) as fh:
            text = fh.read()
        ext = BR.extend_passes(text)
        self.assertNotIn(1, ext)
        self.assertEqual(min(p.fwd_before for p in ext[0]), 376.0)
        self.assertEqual(min(p.fwd_before for p in BR.admission_passes(text)[1]), 218.0)


class NoTrimFixesIt(CustomTestCase):
    def test_no_admission_pass_where_the_vote_did_not_trim_below_its_need(self):
        """S1 (trim before the admission under 400 + A) would act only on such a pass: there is none on any form."""
        for form in ("int4ar", "drei", "duo"):
            s1 = [p for p in _passes(form) if p.admission is not None and p.admission >= BR.ADMIT_MIN_MIB
                  and p.vote_released is None and p.post < CG.NEAR_OOM_MIB + p.admission]
            self.assertEqual(s1, [], form)

    def test_the_vote_trim_ran_before_the_low_instants_and_returned_little(self):
        low = [p for p in _passes("int4ar") if p.fwd_before is not None and p.fwd_before < 400]
        trimmed = [p for p in low if p.vote_released is not None]
        self.assertEqual(len(trimmed), 32)
        self.assertLessEqual(max(p.vote_released for p in trimmed), 64)
        # the one without: first pass after a resume, the vote's trim still armed from before the flip (post unknown)
        self.assertEqual([(p.rank, p.post) for p in low if p.vote_released is None], [(1, None)])


class TheRecordIsTheDerivation(CustomTestCase):
    def test_the_peak_saturates_and_is_cross_form(self):
        self.assertEqual([BR.admission_peak(FORM[f]) for f in ("int4ar", "drei", "duo")], [512.0, 516.0, 522.0])
        self.assertEqual(BR.admission_peak(FORM["int4ar"] + FORM["drei"] + FORM["duo"]), A_PEAK)

    def test_the_record_recomputes_from_the_boots(self):
        need, lines = BR.admit_need_from_passes(FORM["int4ar"], float(CG.NEAR_OOM_MIB), A_PEAK)
        self.assertEqual(need, NEED)
        self.assertEqual(NEED, CG.NEAR_OOM_MIB + A_PEAK - POST_FLOOR)
        self.assertTrue(any("rank=0 post_floor 608" in ln for ln in lines), lines)
        self.assertTrue(any("rank=1 post_floor 658" in ln and "26 admission instant(s) under 400" in ln
                            for ln in lines), lines)
        rec = F.PROFILES[F.PROFILE_QWEN27B].constants[BR.ADMIT_RECORD]
        self.assertEqual(list(rec.value), [NEED] * 3)
        self.assertEqual(tuple(rec.boots), BR.BOOTS_ADMIT_NEED)
        self.assertEqual(tuple(rec.boots), INT4AR_TAGS)

    def test_one_forms_peak_the_vote_logged_floor_or_the_in_sample_short_book_less(self):
        """Every shortcut lands under the bound -- the OOM direction."""
        own, _ = BR.admit_need_from_passes(FORM["int4ar"], 400.0)            # A_peak 512 (int4ar only)
        self.assertEqual(own, 304)
        voted = [{r: [p for p in v if p.post_src == "vote"] for r, v in b.items()} for b in FORM["int4ar"]]
        vote_only, _ = BR.admit_need_from_passes(voted, 400.0, A_PEAK)
        self.assertLess(vote_only, NEED)
        in_sample = 400 - min(p.fwd_before for p in _passes("int4ar") if p.fwd_before is not None)
        self.assertEqual(in_sample, 184)
        for short in (own, vote_only, in_sample):
            self.assertLess(-(-short // 8) * 8, BOOKED)

    def test_the_booked_post_holds_every_measured_instant_and_the_bound(self):
        for p in _passes("int4ar"):
            if p.fwd_before is not None:
                self.assertGreaterEqual(p.fwd_before + BOOKED, CG.NEAR_OOM_MIB, p)
        self.assertGreaterEqual(POST_FLOOR + BOOKED - A_PEAK, CG.NEAR_OOM_MIB)

    def test_int8_drei_and_duo_are_carried(self):
        drei, _ = BR.admit_need_from_passes(FORM["drei"], 400.0, A_PEAK)
        duo, _ = BR.admit_need_from_passes(FORM["duo"], 400.0, A_PEAK)
        self.assertEqual((drei, duo), (118, 0))
        self.assertLessEqual(-(-drei // 8) * 8, SINGLE_DELTA)
        self.assertLessEqual(-(-drei // 8) * 8, min(FLOOR_DELTAS))


class TheBudgetBooksTheAdmission(CustomTestCase):
    def test_the_int4ar_boots_budget_is_reproduced_without_it(self):
        budgets, _ = _int4ar_pass([], admit="0")
        self.assertEqual(budgets, INT4AR_HEAD)

    def test_int4ar_books_the_need(self):
        lines = []
        budgets, kw = _int4ar_pass(lines)
        self.assertEqual([h - b for h, b in zip(INT4AR_HEAD, budgets)], [BOOKED, BOOKED])
        self.assertEqual(budgets, [17160, 16944])
        self.assertEqual(kw["booked_rest_mib"], [2079 + BOOKED, 2079 + BOOKED])
        self.assertIn(BR.ADMIT_MARKER, kw["booked_rest_provenance"])
        self.assertIn("not a reserve", kw["booked_rest_provenance"])
        named = [ln for ln in lines if BR.ADMIT_MARKER in ln and f"= {BOOKED} MiB: books {BOOKED}" in ln]
        self.assertEqual(len(named), 2, lines)

    def test_the_torch_cap_is_byte_identical(self):
        b, kw = _int4ar_pass([])
        hb, hkw = _int4ar_pass([], admit="0")
        self.assertEqual([x + r for x, r in zip(b, kw["booked_rest_mib"])],
                         [x + r for x, r in zip(hb, hkw.get("booked_rest_mib") or [2079, 2079])])

    def test_switch_off_no_record_and_foreign_forms_book_nothing(self):
        for model in (DREI_MODEL, NVFP4_MODEL, None):
            lines = []
            budgets, _ = _int4ar_pass(lines, model=model)
            self.assertEqual(budgets, INT4AR_HEAD, model)
            self.assertTrue(any(BR.ADMIT_MARKER in ln and "off" in ln for ln in lines), (model, lines))
        real = L._pconst

        def no_rec(name, profile=None):
            if name == BR.ADMIT_RECORD:
                raise KeyError(name)
            return real(name, profile)

        with mock.patch.object(L, "_pconst", side_effect=no_rec):
            budgets, _ = _int4ar_pass([])
        self.assertEqual(budgets, INT4AR_HEAD)
        self.assertTrue(BR.admit_book_on({}))
        self.assertFalse(BR.admit_book_on({BR.ADMIT_ENV: "0"}))

    def test_int8_drei_budgets_are_unchanged(self):
        """The INT8 drei rig (5090 + 2x 3080): not the record's form; its vote-line delta 344 carries its 118."""
        lines = []
        with _published(DREI_MODEL), mock.patch.dict(os.environ, {BR.ADMIT_ENV: "1"}):
            _, budgets, _ = PRE._d_pass(lines, PRE._floors(), single="1")
        self.assertEqual(budgets, [27464, 17040, 16824])
        self.assertTrue(any(BR.ADMIT_MARKER in ln and "admission need unmeasured for form" in ln for ln in lines))

    def test_the_reader_and_the_form_gate(self):
        cards = _int4ar_cards()
        self.assertEqual(CI.inventory_signature(cards), ("RTX3080", "RTX3080"))
        with mock.patch.dict(os.environ, {BR.ADMIT_ENV: "1"}):
            self.assertEqual(L.d_admit_need_record(F.PROFILE_QWEN27B, 2, cards, INT4AR_MODEL)[0], [float(NEED)] * 2)
            self.assertEqual(L.d_admit_need_record(F.PROFILE_QWEN27B, 2, cards, "/m/" + INT4AR_MODEL + "/")[0],
                             [float(NEED)] * 2)
            for model, cs in ((DREI_MODEL, cards), (NVFP4_MODEL, cards), (INT4AR_MODEL, PRE._cards()),
                              (INT4AR_MODEL, None), ("", cards)):
                vals, why = L.d_admit_need_record(F.PROFILE_QWEN27B, None, cs, model)
                self.assertIsNone(vals, (model, cs))
                self.assertIn("admission need unmeasured for form", why)
            self.assertIsNone(L.d_admit_need_record("nextflash", 2, cards, INT4AR_MODEL)[0])
        self.assertIn("name no form", BR.admit_form_mismatch(("other",), INT4AR_MODEL, ("RTX3080", "RTX3080")))

    def test_group_p_books_nothing(self):
        cards = _int4ar_cards()
        self.assertEqual(L.booked_rest_admit_deltas(cards, [2079, 2079], [0, 0], F.PROFILE_QWEN27B, "P"), [0, 0])


class TheLargerPostNeverTheSum(CustomTestCase):
    def test_pure_deltas(self):
        self.assertEqual(BR.admit_deltas([2079, 2079], [0, 0], [314.0, 314.0]), [320, 320])
        self.assertEqual(BR.admit_deltas([2079, 2079], [344, 344], [314.0, 314.0]), [344, 344])
        self.assertEqual(BR.admit_deltas([2079, 2079], [0, 400], [314.0, 314.0]), [320, 400])
        self.assertEqual(BR.admit_deltas([None, 2079], [0, 0], [314.0, 314.0]), [0, 320])
        self.assertEqual(BR.admit_deltas([2079], [8], None), [8])
        self.assertEqual(BR.admit_deltas([1, 1, 1], [0, 0, 0], [320.0, 321.0, -5.0]), [320, 328, 0])
        with self.assertRaises(ValueError):
            BR.admit_deltas([1, 1], [0], [1.0, 1.0])
