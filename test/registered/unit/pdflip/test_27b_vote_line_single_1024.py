# SPDX-License-Identifier: Apache-2.0
"""VOTE-LINE-SINGLE 24.10. (root ticket S4 of the VOTE-LINE-BOOK review): ONE honest width-vote line.

The double count. On a MEASURED card the width vote kept the corridor floor as its line (767 / 701 / 700 MiB on the
27B D), and that floor is the ``probe_activation`` peak of a 4096-token prefill -- the extend activation that
``rate x rows`` (D_EXTEND_CAP_PER_ROW_MIB 0.3091, 1266 MiB at 4096 rows) prices already. VOTE-LINE-BOOK (b130e9dab4)
booked the floor line beside the rest, so the KV pool paid the second copy (delta 472 / 408 / 400 MiB).

The single line: ``near-OOM 400 (corridor_guard.NEAR_OOM_MIB) + D_EXTEND_VOTE_FIXED_MIB`` -- the fixed part measured
on the six INT8-drei boots whose vote logged driver free as post (fixtures: every logged vote paired with its chunk,
budget_rest.extend_passes): post - chunk-end free - rate x rows over the 618 passes whose forward ran after a trim,
max 238 MiB (_1022_003738, a 6-request handback) -> line 638, delta 344 per card.

IN-SAMPLE (review M4): the record IS the max over the passes the truth test checks, so the truth test is a
REGRESSION LOCK (a lowered record goes red), not evidence of a bound. What the boots do show out of sample: the
first boot alone (_080355, 155) would have failed on _1022_003738 (238) -- pinned below. The A1 boot is the next
out-of-sample check.

The admission PEAK (post-wake handbacks, ~510 MiB) is released by the forward trim before the forward and is not a
line post; its condition post - peak >= 400 is NOT structurally secured (measured min 542 on INT8 drei; int4ar
violates it, pre-existing). The forward-need trim (extend_trim.need_trim_threshold) makes the release before the
forward hold by construction. FORM GATE (review M1): only the record's forms (INT8 drei) get the single line; int4ar
and NVFP4 duo measured 223-252 and keep the floor line.

Danger directions pinned here:
- OOM: a term forgotten (no near-OOM edge, no fixed part, the forward-need trim gone or on the start rate, the
  single line on an unmeasured card or a foreign form, the record from one boot or one rank).
- Narrow / KV (double): the floor kept beside the single line, the fixed part as the sum of two separate maxima,
  the budget booking a line the ranks do not keep.
"""

import contextlib
import json
import os
import sys
import types
import unittest.mock as mock

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import corridor_guard as CG  # noqa: E402
from flliper.srt.pdflip import budget_rest as BR  # noqa: E402
from flliper.srt.pdflip import card_identity as CI  # noqa: E402
from flliper.srt.pdflip import extend_trim as ET  # noqa: E402
from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_27b_int8_tp1corridor_1024 as PRE  # noqa: E402  (same directory: the predecessor's live-boot rig)

FIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "vote_line_single_1024")
FIXED = 238            # D_EXTEND_VOTE_FIXED_MIB, measured (this file recomputes it from the boots)
LINE = 400 + FIXED     # near-OOM + fixed
DELTA = 344            # ceil8(638 - 300)
FLOOR_DELTAS = [472, 408, 400]   # b130e9dab4: ceil8(767 / 701 / 700 - 300)
#: the six INT8-drei boots with driver-free post (fixture stamp -> boot tag), and F per boot
BOOTS = {"001910": "dkr27bbar1drei10210018", "004350": "dkr27bbar1drei10210043", "003738": "dkr27bbar1drei10220037",
         "010424": "dkr27bbar1drei10220104", "015348": "dkr27bbar1drei10220153", "080355": "dkr27bbar1drei10240803"}
F_PER_BOOT = {"001910": 0, "004350": 191, "003738": 238, "010424": 197, "015348": 153, "080355": 155}
DREI_MODEL = "Qwen3.8-27B-INT8-gdncov-vocabembed"
INT4AR_MODEL = "Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound-ct"
NVFP4_MODEL = "Qwen3.8-27B-NVFP4-RadixArk"


def _form_env(model):
    return F.PdFlipForm(arch="dense", experts="none", draft="dflash", p_draft="cold", kv="paged_dcp", flip="family",
                        vision="transient", profile="qwen27b", model=model).env_value()


@contextlib.contextmanager
def _published(model=DREI_MODEL):
    """The launcher's published form (the budget reads the checkpoint from it)."""
    with mock.patch.dict(os.environ, {F.FORM_ENV: _form_env(model)}):
        yield


def _load(stamp):
    with open(os.path.join(FIX_DIR, f"passes_{stamp}.json")) as fh:
        doc = json.load(fh)
    cols = doc["columns"]
    out = {}
    for row in doc["rows"]:
        d = dict(zip(cols, row))
        out.setdefault(int(d["rank"]), []).append(BR.ExtendPass(
            rank=int(d["rank"]), vote_ts=d["vote_ts"], chunk_ts=d["chunk_ts"], post=float(d["post"]),
            rate=float(d["rate"]), line=float(d["line"]), cap=int(d["cap"]), rows=int(d["rows"]), end=float(d["end"]),
            fwd_before=None if d["fwd_before"] is None else float(d["fwd_before"]), released=float(d["released"])))
    return out


ALL = {s: _load(s) for s in BOOTS}
PASSES = ALL["080355"]


def _line(rank):
    """The single line of a drei rank (measured floor 767 / 701 / 700) with the SHIPPED record."""
    rec = F.PROFILES[F.PROFILE_QWEN27B].constants[BR.VOTE_FIXED_RECORD].value
    return ET.line_for_floor_mib({0: 767, 1: 701, 2: 700}[rank], rec[rank])


def _truth_holds(p, line):
    """line + rate x rows >= (post - forward-phase reading) + near-OOM: the vote funding ``rows`` from the smallest
    post that funds them would leave the forward phase at or above the edge."""
    return line + p.rate * p.rows >= (p.post - p.end) + CG.NEAR_OOM_MIB - 1e-6


class TheRecordIsTheMeasurement(CustomTestCase):
    def test_the_record_recomputes_from_the_boots(self):
        per = {s: BR.vote_fixed_from_passes(ps)[0] for s, ps in ALL.items()}
        self.assertEqual(per, F_PER_BOOT)
        self.assertEqual(max(per.values()), FIXED)
        rec = F.PROFILES[F.PROFILE_QWEN27B].constants[BR.VOTE_FIXED_RECORD]
        self.assertEqual(list(rec.value), [FIXED] * 3)
        self.assertEqual(tuple(rec.boots), tuple(BOOTS.values()))
        self.assertEqual(tuple(rec.boots), BR.BOOTS_VOTE_FIXED)
        self.assertEqual(rec.measured_on, F.PROFILE_QWEN27B)
        self.assertEqual(sum(1 for ps in ALL.values() for v in ps.values() for p in v[1:] if p.trimmed), 618)

    def test_one_boot_is_not_a_bound(self):
        """Review M4: out of sample, the first record (_080355 alone, 155) fails on _1022_003738 -- the record is the
        max over every comparable boot and stays a regression lock until A1 measures out of sample again."""
        first = BR.vote_fixed_from_passes(ALL["080355"])[0]
        oos = [p for v in ALL["003738"].values() for p in v[1:] if p.trimmed]
        self.assertTrue(any(not _truth_holds(p, 400.0 + first) for p in oos))
        self.assertTrue(all(_truth_holds(p, float(LINE)) for p in oos))

    def test_the_parser_reads_the_real_lines(self):
        with open(os.path.join(FIX_DIR, "excerpt_080355_D.txt")) as fh:
            got = BR.extend_passes(fh.read())
        tp1 = {p.vote_ts[11:]: p for p in got[1]}
        p = tp1["08:18:40"]
        self.assertEqual((p.post, p.rows, p.end, p.fwd_before, p.released), (1078.0, 1218, 568.0, 608.0, 282.0))
        self.assertAlmostEqual(p.excess, 133.5, places=1)    # the review's "admission rest 134"
        self.assertEqual(p.admission, 470.0)
        q = [x for x in got[2] if x.vote_ts.endswith("08:17:42")][0]
        self.assertFalse(q.trimmed)
        self.assertAlmostEqual(q.excess, 336.0, places=1)

    def test_a_window_between_vote_and_chunk_unpairs(self):
        txt = "\n".join([
            "[2026-10-24 08:00:00 TP1] PDFLIP-EXTEND-STUECKELUNG rank=1 geplant=4096 cap=100 post=800 rate=0.3091 floor=701",
            "[2026-10-24 08:00:01 TP1] PDFLIP-VRAM-PEAK rank=1 phase=round n=64 card_free_mib=700",
            "[2026-10-24 08:00:02 TP1] PDFLIP-VRAM-PEAK rank=1 phase=chunk rows=50 n=1 card_free_mib=600",
        ])
        self.assertEqual(BR.extend_passes(txt), {})


class TheSingleLineIsTrueAtEveryMeasuredWidth(CustomTestCase):
    """Regression lock (IN-SAMPLE): line + rate x rows >= measured forward phase + 400, per width, every boot."""

    BUCKETS = ((0, 64), (64, 256), (256, 512), (512, 1024), (1024, 1 << 20))

    def test_every_trimmed_pass_at_every_width(self):
        seen = {b: 0 for b in self.BUCKETS}
        for s, passes in ALL.items():
            for r, ps in passes.items():
                for p in ps[1:]:
                    if not p.trimmed:
                        continue
                    self.assertTrue(_truth_holds(p, _line(r)), (s, r, p))
                    for lo, hi in self.BUCKETS:
                        if lo <= p.rows < hi:
                            seen[(lo, hi)] += 1
        self.assertTrue(all(n > 0 for n in seen.values()), seen)

    def test_the_vote_bound_passes_hold(self):
        bound = [(s, r, p) for s, passes in ALL.items() for r, ps in passes.items() for p in ps if p.rows >= p.cap]
        self.assertEqual(sum(1 for s, _, _ in bound if s == "080355"), 156)   # TP1 62, TP2 94
        for s, r, p in bound:
            if p.trimmed:
                self.assertTrue(_truth_holds(p, _line(r)), (s, r, p))

    def test_the_admission_peak_is_not_structurally_secured(self):
        """Review M2: post - peak >= 400 held on INT8 drei (min 542 over six boots) -- measured, not secured: the
        admission runs before the forward-need trim, only the vote's static trim precedes it (TP2 800)."""
        mins = min(p.fwd_before for passes in ALL.values() for v in passes.values() for p in v if p.trimmed)
        self.assertEqual(mins, 542.0)
        self.assertGreaterEqual(mins, CG.NEAR_OOM_MIB)
        peak = max(p.admission for passes in ALL.values() for v in passes.values() for p in v if p.trimmed)
        self.assertGreater(800.0 - peak, 0)
        self.assertLess(800.0 - peak, CG.NEAR_OOM_MIB)   # structurally the static 800 alone leaves < 400

    def test_untrimmed_passes_are_trimmed_by_the_forward_need_at_the_bound(self):
        """An untrimmed pass above the line carried an admission (excess > FIXED >= the forward's own part), so at
        the bound (post = line + rate x rows) the card is under the forward's need -> the trim fires; the SAME pass
        on a rank that trimmed it stays <= FIXED (TP-symmetric shards)."""
        over = without_peer = 0
        for s, passes in ALL.items():
            by_chunk = {}
            for r, ps in passes.items():
                for p in ps:
                    by_chunk.setdefault(p.chunk_ts, {})[r] = p
            for r, ps in passes.items():
                for p in ps[1:]:
                    if p.trimmed or _truth_holds(p, _line(r)):
                        continue
                    over += 1
                    self.assertGreater(p.excess, FIXED)
                    peers = [q for rr, q in by_chunk[p.chunk_ts].items() if rr != r and q.trimmed]
                    without_peer += 0 if peers else 1
                    for q in peers:
                        self.assertLessEqual(q.excess, FIXED, (s, r, p, q))
        self.assertEqual((over, without_peer), (45, 6))

    def test_the_first_pass_is_resident_growth(self):
        """_080355: the first pass's growth stays (next post lower by >= 200); on the trimmed TP1 the rest is under the
        record. TP0 / TP2 ran it untrimmed (612 / 572, 360 not resident) -- the forward-need trim covers that case."""
        for r, ps in PASSES.items():
            first, nxt = ps[0], ps[1]
            resident = first.post - nxt.post
            self.assertGreaterEqual(resident, 200, r)
            if first.trimmed:
                self.assertLessEqual(first.excess - resident, FIXED, r)


class TheLineIsOneFormula(CustomTestCase):
    def setUp(self):
        ET.reset_for_tests()

    def tearDown(self):
        ET.reset_for_tests()

    def test_pure_line(self):
        self.assertEqual(ET.near_oom_mib(), float(CG.NEAR_OOM_MIB))
        self.assertEqual(ET.line_for_floor_mib(701), 701.0)            # no record: the floor line, byte for byte
        self.assertEqual(ET.line_for_floor_mib(120), 300.0)
        for floor in (767, 701, 700, 120, 2000):
            self.assertEqual(ET.line_for_floor_mib(floor, FIXED), float(LINE))   # the floor is NOT in it
        self.assertEqual(ET.line_for_floor_mib(701, -5), 400.0)

    def _rank_line(self, src, fixed_env, rank=1):
        env = {} if fixed_env is None else {BR.VOTE_FIXED_ENV: fixed_env}
        with mock.patch.dict(os.environ, env), mock.patch.object(ET, "_read_corridor_line", lambda: src):
            if fixed_env is None:
                os.environ.pop(BR.VOTE_FIXED_ENV, None)
            ET.reset_for_tests()
            return ET.corridor_line_mib(rank), ET.single_line_active(rank)

    def test_the_rank_votes_with_the_single_line_on_a_measured_floor(self):
        self.assertEqual(self._rank_line((701.0, "MEASURED-D:701"), "238,238,238"), (float(LINE), True))
        self.assertEqual(self._rank_line((767.0, "MEASURED-D:767"), "238", rank=0), (float(LINE), True))

    def test_an_unmeasured_card_keeps_its_law_and_no_record_keeps_the_floor(self):
        self.assertEqual(self._rank_line((1024.0, "stated-law:1024"), "238"), (1024.0, False))
        self.assertEqual(self._rank_line((1024.0, "UNMEASURED-FALLBACK:1024"), "238"), (1024.0, False))
        self.assertEqual(self._rank_line((701.0, "MEASURED-D:701"), None), (701.0, False))

    def test_the_rank_and_the_budget_book_the_same_line(self):
        rank_line, _ = self._rank_line((700.0, "MEASURED-D:700"), "238", rank=2)
        with _published():
            _, budgets, kw = PRE._d_pass([], PRE._floors(), single="1")
        self.assertEqual(PRE.HEAD_BUDGETS[2] - budgets[2], -(-int(rank_line - 300) // 8) * 8)


class TheBudgetBooksTheSingleLine(CustomTestCase):
    def test_the_rig_cards_are_the_records_form(self):
        self.assertEqual(CI.inventory_signature(PRE._cards()), ("RTX5090", "RTX3080", "RTX3080"))

    def test_kv_comes_back(self):
        lines = []
        with _published():
            _, budgets, kw = PRE._d_pass(lines, PRE._floors(), single="1")
        self.assertEqual([h - b for h, b in zip(PRE.HEAD_BUDGETS, budgets)], [DELTA] * 3)
        self.assertEqual(budgets, [27464, 17040, 16824])
        self.assertEqual(kw["booked_rest_mib"], [3191 + DELTA, 2079 + DELTA, 2067 + DELTA])
        # against b130e9dab4 (floor line booked): 128 / 64 / 56 MiB back to the KV pool
        self.assertEqual([f - DELTA for f in FLOOR_DELTAS], [128, 64, 56])
        tp1 = [ln for ln in lines if BR.VOTE_LINE_MARKER in ln and f"ordinal=1 nvml1: delta {DELTA} MiB" in ln]
        self.assertEqual(len(tp1), 1, lines)
        self.assertIn(BR.VOTE_FIXED_MARKER, tp1[0])
        self.assertIn(f"near-OOM 400 + fixed {FIXED}", tp1[0])

    def test_the_torch_cap_is_byte_identical(self):
        with _published():
            _, b, kw = PRE._d_pass([], PRE._floors(), single="1")
            _, hb, hkw = PRE._d_pass([], PRE._floors(source=CG.FLOOR_SOURCE_FALLBACK, values=PRE.FALLBACK),
                                     single="1")
        self.assertEqual([x + r for x, r in zip(b, kw["booked_rest_mib"])],
                         [x + r for x, r in zip(hb, hkw["booked_rest_mib"])])

    def test_no_reserve_and_no_unmeasured_card_books_it(self):
        with _published():
            _, budgets, _ = PRE._d_pass([], PRE._floors(source=CG.FLOOR_SOURCE_FALLBACK, values=PRE.FALLBACK),
                                        single="1")
            self.assertEqual(budgets, PRE.HEAD_BUDGETS)
            _, budgets, _ = PRE._d_pass([], PRE._floors(source=CG.FLOOR_SOURCE_ENV, values=PRE.FALLBACK),
                                        single="1")
        self.assertEqual(budgets, PRE.HEAD_BUDGETS)

    def test_a_foreign_form_books_the_floor_line_and_says_so(self):
        """Review M1: the same cards with an NVFP4 / int4ar checkpoint (F measured 232 / 223-252 there) or no
        published form at all -> the floor line, byte for byte, named."""
        for model in (NVFP4_MODEL, INT4AR_MODEL, None):
            lines = []
            ctx = _published(model) if model else mock.patch.dict(os.environ, {F.FORM_ENV: ""})
            with ctx:
                _, budgets, _ = PRE._d_pass(lines, PRE._floors(), single="1")
            self.assertEqual([h - b for h, b in zip(PRE.HEAD_BUDGETS, budgets)], FLOOR_DELTAS, model)
            self.assertTrue(any("F unmeasured for form" in ln and "floor line" in ln for ln in lines), (model, lines))

    def test_switch_off_and_no_record_are_the_floor_line(self):
        with _published():
            _, budgets, _ = PRE._d_pass([], PRE._floors(), single="0")
            self.assertEqual([h - b for h, b in zip(PRE.HEAD_BUDGETS, budgets)], FLOOR_DELTAS)
            real = L._pconst

            def no_rec(name, profile=None):
                if name == BR.VOTE_FIXED_RECORD:
                    raise KeyError(name)
                return real(name, profile)

            with mock.patch.object(L, "_pconst", side_effect=no_rec):
                _, budgets, _ = PRE._d_pass([], PRE._floors(), single="1")
        self.assertEqual([h - b for h, b in zip(PRE.HEAD_BUDGETS, budgets)], FLOOR_DELTAS)
        self.assertTrue(BR.vote_line_single_on({}))
        self.assertFalse(BR.vote_line_single_on({BR.VOTE_LINE_SINGLE_ENV: "0"}))

    def test_the_record_reader_and_the_form_gate(self):
        drei = PRE._cards()
        with mock.patch.dict(os.environ, {BR.VOTE_LINE_SINGLE_ENV: "1"}):
            self.assertEqual(L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, 3, drei, DREI_MODEL)[0], [238.0] * 3)
            self.assertEqual(L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, 3, drei, "/m/" + DREI_MODEL + "/")[0],
                             [238.0] * 3)
            for model, cards in ((NVFP4_MODEL, drei), (INT4AR_MODEL, drei), (DREI_MODEL, drei[1:]),
                                 (DREI_MODEL, None), ("", drei)):
                vals, why = L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, None, cards, model)
                self.assertIsNone(vals, (model, cards))
                self.assertIn("F unmeasured for form", why)
            self.assertIsNone(L.d_extend_vote_fixed_record("nextflash", 3, drei, DREI_MODEL)[0])
        self.assertIsNone(BR.vote_fixed_form_mismatch(BR.BOOTS_VOTE_FIXED, DREI_MODEL, ("RTX5090", "RTX3080", "RTX3080")))
        self.assertIn("name no form", BR.vote_fixed_form_mismatch(("other",), DREI_MODEL, ("RTX5090",)))


def _ns(profile="qwen27b", env_d="FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,1100,800", model="/m/" + DREI_MODEL, **kw):
    base = dict(profile=profile, model=model, extra_d="", env_d=env_d, dual_layout=False, dual_share=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _duo_cards():
    return L.order_cards([
        L.Card(nvml_index=0, uuid=PRE.U5090, name="NVIDIA GeForce RTX 5090", total_mib=32607, reserved_mib=499),
        L.Card(nvml_index=1, uuid=PRE.U_TP1, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425)])


def _int4ar_cards():
    return L.order_cards([
        L.Card(nvml_index=1, uuid=PRE.U_TP1, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
        L.Card(nvml_index=2, uuid=PRE.U_TP2, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425)])


class TheRanksGetTheSameRecord(CustomTestCase):
    def test_the_flip_arm_writes_the_fixed_part_for_the_records_form(self):
        ns, lines = _ns(), []
        with mock.patch.dict(os.environ, {BR.VOTE_LINE_SINGLE_ENV: "1"}):
            L._d_extend_flip_rate_env(ns, lines.append, "D", PRE._cards())
            first = ns.env_d
            L._d_extend_flip_rate_env(ns, lines.append, "D", PRE._cards())
        self.assertEqual(L.parse_group_env(ns.env_d)[BR.VOTE_FIXED_ENV], "238,238,238")
        self.assertEqual(ns.env_d, first)   # idempotent
        self.assertTrue(any(BR.VOTE_FIXED_MARKER in ln for ln in lines), lines)

    def test_int4ar_and_duo_get_no_single_line(self):
        """Review M1 / MX1: int4ar (2x 3080, INT4 AutoRound, F measured 223-252) and NVFP4 duo (5090 + 3080, F 232):
        no env, the ranks keep the floor line; named once."""
        for model, cards in ((INT4AR_MODEL, _int4ar_cards()), (NVFP4_MODEL, _duo_cards()),
                             (DREI_MODEL, _duo_cards()), (NVFP4_MODEL, PRE._cards())):
            ns, lines = _ns(model="/m/" + model), []
            with mock.patch.dict(os.environ, {BR.VOTE_LINE_SINGLE_ENV: "1"}):
                L._d_extend_flip_rate_env(ns, lines.append, "D", cards)
            self.assertNotIn(BR.VOTE_FIXED_ENV, ns.env_d, model)
            self.assertTrue(any("VOTE-LINE-SINGLE off" in ln and "F unmeasured for form" in ln for ln in lines),
                            (model, lines))
        ns = _ns()
        L._d_extend_flip_rate_env(ns, [].append, "D")   # no cards: fail closed
        self.assertNotIn(BR.VOTE_FIXED_ENV, ns.env_d)

    def test_operator_wins_switch_off_and_dual_write_nothing(self):
        ns = _ns(env_d=f"FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,1100,800;{BR.VOTE_FIXED_ENV}=200")
        L._d_extend_flip_rate_env(ns, [].append, "D", PRE._cards())
        self.assertEqual(L.parse_group_env(ns.env_d)[BR.VOTE_FIXED_ENV], "200")
        with mock.patch.dict(os.environ, {BR.VOTE_LINE_SINGLE_ENV: "0"}):
            ns = _ns()
            L._d_extend_flip_rate_env(ns, [].append, "D", PRE._cards())
        self.assertNotIn(BR.VOTE_FIXED_ENV, ns.env_d)
        ns = _ns(dual_share=True)
        L._d_extend_flip_rate_env(ns, [].append, "D", PRE._cards())
        self.assertNotIn(BR.VOTE_FIXED_ENV, ns.env_d)


class _Batch:
    def __init__(self, rows):
        self.extend_num_tokens = rows
        self.forward_mode = types.SimpleNamespace(is_extend=lambda: True, is_target_verify=lambda: False)


class TheForwardTrimsUnderItsNeed(CustomTestCase):
    """The single line prices only what the trim cannot return: the forward trims under line + rate x rows."""

    def setUp(self):
        ET.reset_for_tests()

    def tearDown(self):
        ET.reset_for_tests()

    def _thr(self, rows, fixed_env="238", src=(700.0, "MEASURED-D:700"), static="1200,1100,800", measured=None):
        env = {"FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB": "0.3091,0.3091,0.3091"}
        if static is not None:
            env["FLLIPER_PDFLIP_EXTEND_TRIM_MIB"] = static
        if fixed_env is not None:
            env[BR.VOTE_FIXED_ENV] = fixed_env
        got = []
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(ET, "_read_corridor_line", lambda: src), \
                mock.patch.object(ET, "maybe_trim", side_effect=lambda cuda, rank, thr, *a: got.append(thr)), \
                mock.patch.object(ET, "measure_open", lambda *a, **k: None):
            for k in ("FLLIPER_PDFLIP_EXTEND_TRIM_MIB", BR.VOTE_FIXED_ENV):
                if k not in env:
                    os.environ.pop(k, None)
            ET.reset_for_tests()
            if measured is not None:   # Q-694b: the rank measured a higher rate (x RATE_SAFETY in the vote)
                ET._CACHE["measure"] = True
                ET._CACHE["measure_ranks"] = None
                ET._CACHE["measured"] = measured
            ET.before_extend(types.SimpleNamespace(tp_rank=2), _Batch(rows))
        return got

    def test_the_need_raises_the_threshold(self):
        # TP2 static 800; 1000 rows need 638 + 309.1
        self.assertAlmostEqual(self._thr(1000)[0], LINE + 0.3091 * 1000, places=3)
        # the static threshold stays where it is higher
        self.assertEqual(self._thr(1)[0], 800.0)

    def test_the_need_prices_the_votes_effective_rate(self):
        """Review MX2: the need uses the rate the vote uses (start or measured x 1.15), never the start alone."""
        eff = ET._ceil4(0.4 * ET.RATE_SAFETY)
        self.assertAlmostEqual(self._thr(1000, measured=0.4)[0], LINE + eff * 1000, places=3)

    def test_without_the_record_or_on_an_unmeasured_card_only_the_static(self):
        self.assertEqual(self._thr(1000, fixed_env=None), [800.0])
        self.assertEqual(self._thr(1000, src=(1024.0, "stated-law:1024")), [800.0])
        self.assertEqual(self._thr(1000, fixed_env=None, static=None), [])

    def test_the_need_trims_without_a_static_threshold(self):
        self.assertAlmostEqual(self._thr(1000, static=None)[0], LINE + 0.3091 * 1000, places=3)


if __name__ == "__main__":
    import unittest

    unittest.main()
