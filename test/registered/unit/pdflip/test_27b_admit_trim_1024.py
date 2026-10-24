# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-ADMIT-BOOK 24.10. (Auftrag #212; review-vote-line-single M2 / SOLL S1; review admit-trim M1-M4): the D
extend's admission peak.

Every D extend pass runs ``width vote (trim, read post) -> admission (post-wake handbacks: Mamba restore + KV
load-back) -> forward trim (card_free_before = driver free after the admission) -> forward``; the near-OOM law asks
``post - A >= 400`` at the admission instant.

Census (budget_rest.admission_passes): every chunk window, vote-log independent; a round window closed by the extend
itself at its forward's end (round t_unix_ms == chunk t0_unix_ms) is NOT a pass boundary (review M1 -- reading it as
one dropped 1980 of the 7604 int4ar windows, the serving case). Six int4ar serving boots (2x 3080, D TP2): 995
admission passes, 57 admission instants under 400 (TP1 45), minimum 32 MiB (_1023_223241 23:19:34 TP1). The peak A
saturates at 504..522 MiB from 3-4 handbacks on all three forms.

Two classes: 47 instants after a vote trim (it returned <= 204 MiB; the rest of the cache is not returnable) -- only
the budget helps; 10 without one (``armed`` left over from refused admissions, 568 in 174 s before the 32) --
``extend_trim.before_admission`` trims there now (review M3).

The price: ``D_ADMIT_NEED_MIB = near-OOM 400 + A_peak 522 - post_floor 508 = 414`` (post_floor = lowest post of ANY
pass, review M2), booked per card as ``max(vote-line delta, ceil8(need)) = 416``. Form gate: int4ar only; INT8 drei
derives 300 on its current trim regime (carried by its vote-line delta 344 / 400..472), NVFP4 duo 52 (not booked,
measured minimum 690), NVFP4 dual has no width vote.

Danger directions pinned here:
- OOM: the admission post forgotten, the peak of one form only, the floor from admission passes or vote-logged passes
  only, the in-sample shortfall instead of the bound, rounded down, the form gate shut, the pass boundary at the
  extend-closed round, the admission trim gone or never trimming.
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
from flliper.srt.pdflip import extend_trim as ET  # noqa: E402
from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import inventory_view as IV  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_27b_int8_tp1corridor_1024 as PRE  # noqa: E402  (same directory: the predecessor's live-boot rig)

FIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "admit_trim_1024")
NEED = 414             # D_ADMIT_NEED_MIB (this file recomputes it from the boots)
BOOKED = 416           # ceil8(414)
A_PEAK = 522           # NVFP4 duo, 5 handbacks
POST_FLOOR = 508       # int4ar TP1, lowest post of any pass
ADM_FLOOR = 540        # int4ar TP1, lowest post of an admission pass (not the floor: review M2)
#: the INT8 drei boots of 21.10. ran TP1/TP2 without a static trim (retired regime, named, not this record's form)
DREI_RETIRED = ("dkr27bbar1drei10210018", "dkr27bbar1drei10210043")
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




def _ceil8(x):
    return -(-int(x) // 8) * 8


class TheBootsViolateTheEdge(CustomTestCase):
    def test_the_census_carries_the_trims_of_every_extend(self):
        ps = _passes("int4ar")
        self.assertEqual(len(ps), 7604)
        # review M1: 1980 windows came out empty when the extend-closed round line ended the pass; 13 are left
        # (passes whose trims were not logged at all)
        self.assertEqual(sum(1 for p in ps if p.post is None and p.fwd_before is None), 13)
        self.assertEqual(sum(1 for p in ps if p.admission is not None and p.admission >= BR.ADMIT_MIN_MIB), 995)
        low = [p for p in ps if p.fwd_before is not None and p.fwd_before < CG.NEAR_OOM_MIB]
        self.assertEqual(len(low), 57)
        self.assertEqual(sum(1 for p in low if p.rank == 1), 45)
        self.assertEqual(min(p.fwd_before for p in low), 32)
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

    def test_only_the_round_closed_by_the_extend_is_no_boundary(self):
        """Review M1, the boundary itself: a round whose t_unix_ms is the next chunk's t0_unix_ms (closed at the
        extend's forward end) keeps the pass; a self-closed round (other t), a trim after the round, a flip or an
        idle window end it."""
        def lines(*rows):
            return "\n".join(f"[2026-10-23 23:19:3{i} TP1] {r}" for i, r in enumerate(rows))
        trim_v = "PDFLIP-EXTEND-CACHE-TRIM rank=1 ms=1 released=60 card_free_before=620 card_free_after=680 threshold=1104"
        trim_f = "PDFLIP-EXTEND-CACHE-TRIM rank=1 ms=9 released=630 card_free_before=32 card_free_after=662 threshold=1104"
        chunk = "PDFLIP-VRAM-PEAK rank=1 phase=chunk rows=247 n=1 t0_unix_ms=200 t_unix_ms=201 card_free_mib=600"

        def rnd(t):
            return f"PDFLIP-VRAM-PEAK rank=1 phase=round n=26 t0_unix_ms=100 t_unix_ms={t} card_free_mib=430"
        kept = BR.admission_passes(lines(trim_v, trim_f, rnd(200), chunk))[1][0]
        self.assertEqual((kept.post, kept.fwd_before, kept.admission), (680.0, 32.0, 648.0))
        for text in (lines(trim_v, trim_f, rnd(199), chunk),
                     lines(trim_v, trim_f, "PDFLIP-VRAM-PEAK rank=1 phase=idle n=0 t0_unix_ms=1 t_unix_ms=200 "
                                           "card_free_mib=430", chunk),
                     lines(trim_v, trim_f, "PDFLIP-VRAM-PEAK rank=1 phase=flip leg=resume t0_unix_ms=1 "
                                           "t_unix_ms=200 card_free_mib=430", chunk)):
            p = BR.admission_passes(text)[1][0]
            self.assertEqual((p.post, p.fwd_before), (None, None), text)
        p = BR.admission_passes(lines(trim_v, rnd(200), trim_f, chunk))[1][0]   # a trim after the round: boundary
        self.assertEqual((p.post, p.fwd_before), (None, 32.0))
        # the vote-log pairing obeys the same boundary
        vote = ("PDFLIP-EXTEND-STUECKELUNG rank=1 geplant=4096 cap=1 post=680 cache_free=600 funds=driver "
                "rate=0.3091 floor=1024")
        ext = BR.extend_passes(lines(trim_v, vote, trim_f, rnd(200), chunk))
        self.assertEqual([(q.post, q.fwd_before) for q in ext[1]], [(680.0, 32.0)])
        self.assertEqual(BR.extend_passes(lines(trim_v, vote, trim_f, rnd(199), chunk)), {})

    def test_an_admission_trim_marks_the_post_before_the_admission(self):
        text = "\n".join([
            "[2026-10-23 23:19:30 TP1] PDFLIP-EXTEND-CACHE-TRIM rank=1 ms=50 released=600 card_free_before=60 "
            "card_free_after=660 threshold=1104 site=admit",
            "[2026-10-23 23:19:31 TP1] PDFLIP-EXTEND-CACHE-TRIM rank=1 ms=9 released=300 card_free_before=170 "
            "card_free_after=470 threshold=1104",
            "[2026-10-23 23:19:31 TP1] PDFLIP-VRAM-PEAK rank=1 phase=chunk rows=247 n=1 t0_unix_ms=1 t_unix_ms=2 "
            "card_free_mib=600"])
        p = BR.admission_passes(text)[1][0]
        self.assertEqual((p.post, p.post_src, p.fwd_before, p.admission, p.vote_released),
                         (660.0, "admit", 170.0, 490.0, 600.0))

    def test_the_vote_log_pairing_misses_the_low_rank(self):
        """The instrument trap of the predecessor's figure: the vote line is logged only when the cap changes."""
        with open(os.path.join(FIX_DIR, "excerpt_223241_D.txt")) as fh:
            text = fh.read()
        ext = BR.extend_passes(text)
        self.assertNotIn(1, ext)
        self.assertEqual(min(p.fwd_before for p in ext[0]), 376.0)
        self.assertEqual(min(p.fwd_before for p in BR.admission_passes(text)[1]), 218.0)


class _Cuda:
    """torch.cuda in MiB: ``free`` card bytes, ``cache`` returnable by empty_cache."""

    def __init__(self, free, cache):
        self.free, self.cache, self.syncs, self.empties = float(free), float(cache), 0, 0

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self):
        return int(self.free * ET.MIB), int(20055 * ET.MIB)

    def memory_reserved(self):
        return int((10000 + self.cache) * ET.MIB)

    def synchronize(self):
        self.syncs += 1

    def empty_cache(self):
        self.empties += 1
        self.free += self.cache
        self.cache = 0.0


class TheTwoClasses(CustomTestCase):
    def test_the_vote_trimmed_class_and_the_armed_class(self):
        ps = _passes("int4ar")
        low = [p for p in ps if p.fwd_before is not None and p.fwd_before < 400]
        trimmed = [p for p in low if p.vote_released is not None]
        self.assertEqual(len(trimmed), 47)
        self.assertEqual(max(p.vote_released for p in trimmed), 204)
        armed = [p for p in ps if p.fwd_before is not None and p.vote_released is None]
        self.assertEqual(len(armed), 148)
        self.assertEqual(sorted(p.fwd_before for p in armed if p.fwd_before < 400)[:2], [32.0, 222.0])
        self.assertEqual(sum(1 for p in armed if p.fwd_before < 400), 10)

    def _rig(self, thresholds="1104,1104"):
        ET.reset_for_tests()
        ET._CACHE["thresholds"] = ET.parse_thresholds(thresholds)
        ET._CACHE["rates"] = ET.parse_thresholds("0.3091,0.3091")
        ET._CACHE["measure"] = False
        ET._CACHE["measure_ranks"] = None
        ET._CACHE["cap_floor"] = False
        ET._CACHE["vote_fixed"] = None
        ET._CACHE["line"] = 1024.0

    def tearDown(self):
        ET.reset_for_tests()

    def test_the_admission_trims_when_the_armed_vote_did_not(self):
        """The 32 MiB pass: the vote of an earlier pass trimmed and stayed armed (its admission was refused, no
        extend ran), decode refilled the cache; this pass's vote does not trim -- the admission does."""
        self._rig()
        cuda = _Cuda(free=700, cache=50)
        with mock.patch.object(ET, "_torch_cap_mib", return_value=None):
            ET.width_vote(cuda, 1, 4096, 1, True)           # pass 1: vote trims, armed
            self.assertEqual(cuda.empties, 1)
            cuda.cache, cuda.free = 630.0, 60.0                # refused admission, decode rounds refill the cache
            ET.width_vote(cuda, 1, 4096, 1, True)           # pass 2: armed -> the vote does not trim
            self.assertEqual(cuda.empties, 1)
            line = ET.before_admission(cuda, 1)
        self.assertIsNotNone(line)
        self.assertIn("site=admit", line)
        self.assertIn("card_free_before=60", line)
        self.assertEqual((cuda.empties, cuda.free), (2, 690.0))
        self.assertIsNone(ET.before_admission(cuda, 1))     # once per pass
        self.assertEqual(cuda.empties, 2)

    def test_no_second_trim_when_the_vote_trimmed_this_pass(self):
        self._rig()
        cuda = _Cuda(free=700, cache=50)
        with mock.patch.object(ET, "_torch_cap_mib", return_value=None):
            ET.width_vote(cuda, 1, 4096, 1, True)
        cuda.cache, cuda.free = 300.0, 400.0
        self.assertIsNone(ET.before_admission(cuda, 1))
        self.assertEqual(cuda.empties, 1)

    def test_no_threshold_no_rank_no_trim(self):
        self._rig(thresholds="")
        cuda = _Cuda(free=60, cache=600)
        with mock.patch.object(ET, "_torch_cap_mib", return_value=None):
            ET.width_vote(cuda, 1, 4096, 1, True)
        self.assertIsNone(ET.before_admission(cuda, 1))
        self._rig()
        self.assertIsNone(ET.before_admission(cuda, None))
        self.assertEqual(cuda.empties, 0)

    def test_the_scheduler_trims_before_the_admission_allocates(self):
        """The hook sits where a request has passed the X gate (note_admit) and before add_one_req (Mamba restore and
        KV load-back allocate there): rank-local memory care, no verdict moves (SOLL R11)."""
        import inspect

        from flliper.srt.managers import scheduler as S

        src = inspect.getsource(S)
        i = src.index("_pdflip_hbd.note_admit(req)")
        j = src.index("_pdflip_extend_trim.before_admission(torch.cuda", i)
        k = src.index("adder.add_one_req(", i)
        self.assertLess(j, k)
        self.assertLess(j - i, 600)


class TheRecordIsTheDerivation(CustomTestCase):
    def test_the_peak_saturates_and_is_cross_form(self):
        self.assertEqual([BR.admission_peak(FORM[f]) for f in ("int4ar", "drei", "duo")], [512.0, 516.0, 522.0])
        self.assertEqual(BR.admission_peak(FORM["int4ar"] + FORM["drei"] + FORM["duo"]), A_PEAK)

    def test_the_record_recomputes_from_the_boots(self):
        need, lines = BR.admit_need_from_passes(FORM["int4ar"], float(CG.NEAR_OOM_MIB), A_PEAK)
        self.assertEqual(need, NEED)
        self.assertEqual(NEED, CG.NEAR_OOM_MIB + A_PEAK - POST_FLOOR)
        self.assertTrue(any("rank=1 post_floor 508" in ln and "lowest admission-pass post 540" in ln
                            and "45 admission instant(s) under 400" in ln for ln in lines), lines)
        rec = F.PROFILES[F.PROFILE_QWEN27B].constants[BR.ADMIT_RECORD]
        self.assertEqual(list(rec.value), [NEED] * 3)
        self.assertEqual(tuple(rec.boots), BR.BOOTS_ADMIT_NEED)
        self.assertEqual(tuple(rec.boots), INT4AR_TAGS)

    def test_every_shortcut_books_less(self):
        """One form's peak, the admission-pass floor, the vote-logged floor, the in-sample shortfall: all under the
        bound -- the OOM direction."""
        own, _ = BR.admit_need_from_passes(FORM["int4ar"], 400.0)            # A_peak 512 (int4ar only)
        self.assertEqual(own, 404)
        ps = _passes("int4ar")
        adm_floor = min(p.post for p in ps if p.admission is not None and p.admission >= BR.ADMIT_MIN_MIB)
        self.assertEqual(adm_floor, ADM_FLOOR)
        vote_floor = min(p.post for p in ps if p.post_src == "vote")
        in_sample = 400 - min(p.fwd_before for p in ps if p.fwd_before is not None)
        self.assertEqual(in_sample, 368)
        for short in (own, 400 + A_PEAK - adm_floor, 400 + A_PEAK - vote_floor, in_sample):
            self.assertLess(_ceil8(short), BOOKED, short)

    def test_the_booked_post_holds_every_measured_instant_and_the_bound(self):
        for p in _passes("int4ar"):
            if p.fwd_before is not None:
                self.assertGreaterEqual(p.fwd_before + BOOKED, CG.NEAR_OOM_MIB, p)
        self.assertGreaterEqual(POST_FLOOR + BOOKED - A_PEAK, CG.NEAR_OOM_MIB)

    def test_int8_drei_and_duo(self):
        cur = [per for t, (f, per) in ALL.items() if f == "drei" and t not in DREI_RETIRED]
        retired = [per for t, (f, per) in ALL.items() if t in DREI_RETIRED]
        drei, _ = BR.admit_need_from_passes(cur, 400.0, A_PEAK)
        self.assertEqual(drei, 300)
        self.assertLessEqual(_ceil8(drei), SINGLE_DELTA)
        self.assertLessEqual(_ceil8(drei), min(FLOOR_DELTAS))
        self.assertEqual(BR.admit_need_from_passes(retired, 400.0, A_PEAK)[0], 872)   # named, not this form
        duo, _ = BR.admit_need_from_passes(FORM["duo"], 400.0, A_PEAK)
        self.assertEqual(duo, 52)
        self.assertGreaterEqual(min(p.fwd_before for p in _passes("duo") if p.fwd_before is not None), 690)


class TheBudgetBooksTheAdmission(CustomTestCase):
    def test_the_int4ar_boots_budget_is_reproduced_without_it(self):
        budgets, _ = _int4ar_pass([], admit="0")
        self.assertEqual(budgets, INT4AR_HEAD)

    def test_int4ar_books_the_need(self):
        lines = []
        budgets, kw = _int4ar_pass(lines)
        self.assertEqual([h - b for h, b in zip(INT4AR_HEAD, budgets)], [BOOKED, BOOKED])
        self.assertEqual(budgets, [17064, 16848])
        self.assertEqual(kw["booked_rest_mib"], [2079 + BOOKED, 2079 + BOOKED])
        self.assertIn(BR.ADMIT_MARKER, kw["booked_rest_provenance"])
        self.assertIn("not a reserve", kw["booked_rest_provenance"])
        named = [ln for ln in lines if BR.ADMIT_MARKER in ln and f"= {BOOKED} MiB: books {BOOKED}" in ln]
        self.assertEqual(len(named), 2, lines)
        self.assertFalse([ln for ln in lines if "REST-MISMATCH" in ln], lines)

    def test_another_rest_is_named_and_still_booked(self):
        lines = []
        with _published(INT4AR_MODEL), mock.patch.dict(os.environ, {BR.ADMIT_ENV: "1"}):
            got = L.booked_rest_admit_deltas(_int4ar_cards(), [2300, 2300], [0, 0], F.PROFILE_QWEN27B, "D",
                                             lines.append, rest_name="D_AWAKE_REST_BOOKED_MIB")
        self.assertEqual(got, [BOOKED, BOOKED])
        self.assertTrue(any("REST-MISMATCH" in ln and "re-measure" in ln for ln in lines), lines)
        self.assertEqual(BR.ADMIT_REST_BASIS, ("D_AWAKE_REST_BOOKED_MIB", 2079))

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
        """The INT8 drei rig (5090 + 2x 3080): not the record's form; its vote-line delta 344 carries its 300."""
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
        self.assertEqual(BR.admit_deltas([2079, 2079], [0, 0], [414.0, 414.0]), [416, 416])
        self.assertEqual(BR.admit_deltas([2079, 2079], [472, 472], [414.0, 414.0]), [472, 472])
        self.assertEqual(BR.admit_deltas([2079, 2079], [0, 400], [414.0, 414.0]), [416, 416])
        self.assertEqual(BR.admit_deltas([None, 2079], [0, 0], [414.0, 414.0]), [0, 416])
        self.assertEqual(BR.admit_deltas([2079], [8], None), [8])
        self.assertEqual(BR.admit_deltas([1, 1, 1], [0, 0, 0], [320.0, 321.0, -5.0]), [320, 328, 0])
        with self.assertRaises(ValueError):
            BR.admit_deltas([1, 1], [0], [1.0, 1.0])
