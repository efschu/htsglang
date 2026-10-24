# SPDX-License-Identifier: Apache-2.0
"""VOTE-LINE-SINGLE 24.10. (root ticket S4 of the VOTE-LINE-BOOK review): ONE honest width-vote line.

The double count. On a MEASURED card the width vote kept the corridor floor as its line (767 / 701 / 700 MiB on the
27B D), and that floor is the ``probe_activation`` peak of a 4096-token prefill -- the extend activation that
``rate x rows`` (D_EXTEND_CAP_PER_ROW_MIB 0.3091, 1266 MiB at 4096 rows) prices already. VOTE-LINE-BOOK (b130e9dab4)
booked the floor line beside the rest, so the KV pool paid the second copy (delta 472 / 408 / 400 MiB).

The single line: ``near-OOM 400 (corridor_guard.NEAR_OOM_MIB) + D_EXTEND_VOTE_FIXED_MIB`` -- the fixed part measured
from boot ``boot_weg2_dkr27bbar1drei10240803_b9cddabda1_1024_080355`` (fixture: every logged vote paired with its
chunk, budget_rest.extend_passes): post - chunk-end free - rate x rows over the 481 passes whose forward ran after a
trim, max 155 MiB -> line 555, delta 256 per card. The admission PEAK (post-wake handbacks, <= 508 MiB) is released by
the forward trim before the forward and is the budget's condition (post - peak >= 400), not a line post; the
forward-need trim (extend_trim.need_trim_threshold) makes that release hold by construction.

Danger directions pinned here:
- OOM: a term forgotten (no near-OOM edge, no fixed part, the forward-need trim gone, the fixed part taken from one
  rank's minimum) -> the truth test (line + rate x rows >= measured forward phase + 400 at every measured width) fails.
- Narrow / KV (double): the floor kept beside the single line, or the fixed part as the sum of two separate maxima ->
  the line / delta / record identities fail.
"""

import json
import os
import sys
import types
import unittest.mock as mock

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import corridor_guard as CG  # noqa: E402
from flliper.srt.pdflip import budget_rest as BR  # noqa: E402
from flliper.srt.pdflip import extend_trim as ET  # noqa: E402
from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_27b_int8_tp1corridor_1024 as PRE  # noqa: E402  (same directory: the predecessor's live-boot rig)

FIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "vote_line_single_1024")
FIXED = 155            # D_EXTEND_VOTE_FIXED_MIB, measured (this file recomputes it from the boot)
LINE = 400 + FIXED     # near-OOM + fixed
DELTA = 256            # ceil8(555 - 300)
FLOOR_DELTAS = [472, 408, 400]   # b130e9dab4: ceil8(767 / 701 / 700 - 300)


def _passes():
    with open(os.path.join(FIX_DIR, "passes_080355.json")) as fh:
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


PASSES = _passes()


def _line(rank):
    """The single line of this boot's rank (its measured floor: 767 / 701 / 700)."""
    return ET.line_for_floor_mib({0: 767, 1: 701, 2: 700}[rank], FIXED)


def _truth_holds(p, line):
    """line + rate x rows >= (post - forward-phase minimum) + near-OOM: the vote funding ``rows`` from the smallest
    post that funds them would leave the forward phase at or above the edge."""
    return line + p.rate * p.rows >= (p.post - p.end) + CG.NEAR_OOM_MIB - 1e-6


class TheRecordIsTheMeasurement(CustomTestCase):
    def test_the_record_recomputes_from_the_boot(self):
        val, lines = BR.vote_fixed_from_passes(PASSES)
        self.assertEqual(val, FIXED)
        self.assertTrue(any("rank=1 trimmed=441 max excess 155.0" in ln for ln in lines), lines)
        self.assertTrue(any("rank=0 trimmed=27 max excess 154.9" in ln for ln in lines), lines)
        rec = F.PROFILES[F.PROFILE_QWEN27B].constants[BR.VOTE_FIXED_RECORD]
        self.assertEqual(list(rec.value), [FIXED] * 3)
        self.assertEqual(tuple(rec.boots), ("dkr27bbar1drei10240803",))
        self.assertEqual(rec.measured_on, F.PROFILE_QWEN27B)
        self.assertEqual(sum(1 for ps in PASSES.values() for p in ps[1:] if p.trimmed), 481)

    def test_the_parser_reads_the_real_lines(self):
        with open(os.path.join(FIX_DIR, "excerpt_080355_D.txt")) as fh:
            got = BR.extend_passes(fh.read())
        tp1 = {p.vote_ts[11:]: p for p in got[1]}
        # 08:18:40: 5 new sequences, 209k cached: admission 1078 -> 608, trim 282 back, 1218 rows end at 568
        p = tp1["08:18:40"]
        self.assertEqual((p.post, p.rows, p.end, p.fwd_before, p.released), (1078.0, 1218, 568.0, 608.0, 282.0))
        self.assertAlmostEqual(p.excess, 133.5, places=1)    # the review's "admission rest 134"
        self.assertEqual(p.admission, 470.0)
        # 08:17:42 TP2: post-wake handback under the static 800 -- no trim, admission held over the forward
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
    """Wahrheitstest: line + rate x rows >= measured forward phase + 400, per width, from the boot."""

    BUCKETS = ((0, 64), (64, 256), (256, 512), (512, 1024), (1024, 1 << 20))

    def test_every_trimmed_pass_at_every_width(self):
        seen = {b: 0 for b in self.BUCKETS}
        for r, ps in PASSES.items():
            for p in ps[1:]:
                if not p.trimmed:
                    continue
                self.assertTrue(_truth_holds(p, _line(r)), (r, p))
                for lo, hi in self.BUCKETS:
                    if lo <= p.rows < hi:
                        seen[(lo, hi)] += 1
        self.assertTrue(all(n > 0 for n in seen.values()), seen)

    def test_the_vote_bound_passes_hold(self):
        bound = [(r, p) for r, ps in PASSES.items() for p in ps if p.rows >= p.cap]
        self.assertEqual(len(bound), 156)   # TP1 62, TP2 94: where the vote really cut
        for r, p in bound:
            self.assertTrue(_truth_holds(p, _line(r)), (r, p))

    def test_the_admission_peak_is_the_budgets_condition(self):
        """Released before the forward, no width removes it: post - peak >= 400 on every trimmed pass (the budget's
        headroom; measured min 716 / 568 / 580 under the boot's budget WITHOUT any line delta)."""
        mins = {}
        for r, ps in PASSES.items():
            for p in ps:
                if p.trimmed:
                    self.assertGreaterEqual(p.fwd_before, CG.NEAR_OOM_MIB, (r, p))
                    mins[r] = min(mins.get(r, 1e9), p.fwd_before)
        self.assertEqual(mins, {0: 716.0, 1: 568.0, 2: 580.0})

    def test_untrimmed_passes_are_trimmed_by_the_forward_need_at_the_bound(self):
        """An untrimmed pass above the line carried an admission (excess > FIXED >= the forward's own part), so at
        the bound (post = line + rate x rows) the card is under the forward's need -> the trim fires; the SAME pass
        on a rank that trimmed it shows <= FIXED (TP-symmetric shards)."""
        by_chunk = {}
        for r, ps in PASSES.items():
            for p in ps:
                by_chunk.setdefault(p.chunk_ts, {})[r] = p
        over = 0
        for r, ps in PASSES.items():
            for p in ps[1:]:
                if p.trimmed or _truth_holds(p, _line(r)):
                    continue
                over += 1
                self.assertGreater(p.excess, FIXED)
                peers = [q for rr, q in by_chunk[p.chunk_ts].items() if rr != r and q.trimmed]
                self.assertTrue(peers, (r, p))
                for q in peers:
                    self.assertLessEqual(q.excess, FIXED, (r, p, q))
        self.assertEqual(over, 6)

    def test_the_first_pass_is_resident_growth(self):
        for r, ps in PASSES.items():
            first, nxt = ps[0], ps[1]
            resident = first.post - nxt.post
            self.assertGreaterEqual(resident, 200, r)
            if first.trimmed:   # TP1: what is not resident is inside the fixed part
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
        self.assertEqual(self._rank_line((701.0, "MEASURED-D:701"), "155,155,155"), (float(LINE), True))
        self.assertEqual(self._rank_line((767.0, "MEASURED-D:767"), "155", rank=0), (float(LINE), True))

    def test_an_unmeasured_card_keeps_its_law_and_no_record_keeps_the_floor(self):
        self.assertEqual(self._rank_line((1024.0, "stated-law:1024"), "155"), (1024.0, False))
        self.assertEqual(self._rank_line((1024.0, "UNMEASURED-FALLBACK:1024"), "155"), (1024.0, False))
        self.assertEqual(self._rank_line((701.0, "MEASURED-D:701"), None), (701.0, False))

    def test_the_rank_and_the_budget_book_the_same_line(self):
        rank_line, _ = self._rank_line((700.0, "MEASURED-D:700"), "155", rank=2)
        lines = []
        _, budgets, kw = PRE._d_pass(lines, PRE._floors(), single="1")
        self.assertEqual(PRE.HEAD_BUDGETS[2] - budgets[2], -(-int(rank_line - 300) // 8) * 8)


class TheBudgetBooksTheSingleLine(CustomTestCase):
    def test_kv_comes_back(self):
        lines = []
        _, budgets, kw = PRE._d_pass(lines, PRE._floors(), single="1")
        self.assertEqual([h - b for h, b in zip(PRE.HEAD_BUDGETS, budgets)], [DELTA] * 3)
        self.assertEqual(budgets, [27552, 17128, 16912])
        self.assertEqual(kw["booked_rest_mib"], [3191 + DELTA, 2079 + DELTA, 2067 + DELTA])
        # against b130e9dab4 (floor line booked): 216 / 152 / 144 MiB back to the KV pool
        self.assertEqual([f - DELTA for f in FLOOR_DELTAS], [216, 152, 144])
        tp1 = [ln for ln in lines if BR.VOTE_LINE_MARKER in ln and "ordinal=1 nvml1: delta 256 MiB" in ln]
        self.assertEqual(len(tp1), 1, lines)
        self.assertIn(BR.VOTE_FIXED_MARKER, tp1[0])
        self.assertIn("near-OOM 400 + fixed 155", tp1[0])

    def test_the_torch_cap_is_byte_identical(self):
        _, b, kw = PRE._d_pass([], PRE._floors(), single="1")
        _, hb, hkw = PRE._d_pass([], PRE._floors(source=CG.FLOOR_SOURCE_FALLBACK, values=PRE.FALLBACK), single="1")
        self.assertEqual([x + r for x, r in zip(b, kw["booked_rest_mib"])],
                         [x + r for x, r in zip(hb, hkw["booked_rest_mib"])])

    def test_no_reserve_and_no_unmeasured_card_books_it(self):
        _, budgets, _ = PRE._d_pass([], PRE._floors(source=CG.FLOOR_SOURCE_FALLBACK, values=PRE.FALLBACK),
                                    single="1")
        self.assertEqual(budgets, PRE.HEAD_BUDGETS)
        _, budgets, _ = PRE._d_pass([], PRE._floors(source=CG.FLOOR_SOURCE_ENV, values=PRE.FALLBACK), single="1")
        self.assertEqual(budgets, PRE.HEAD_BUDGETS)

    def test_switch_off_and_no_record_are_the_floor_line(self):
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

    def test_the_record_reader(self):
        with mock.patch.dict(os.environ, {BR.VOTE_LINE_SINGLE_ENV: "1"}):
            self.assertEqual(L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, 3)[0], [155.0] * 3)
            self.assertIsNone(L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, 2)[0])  # no view installed: 3 != 2
            self.assertIsNone(L.d_extend_vote_fixed_record("nextflash", 3)[0])


def _ns(profile="qwen27b", env_d="FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,1100,800", **kw):
    base = dict(profile=profile, model="/nonexistent/Qwen3.8-27B-INT8", extra_d="", env_d=env_d,
                dual_layout=False, dual_share=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


class TheRanksGetTheSameRecord(CustomTestCase):
    def test_the_flip_arm_writes_the_fixed_part(self):
        ns, lines = _ns(), []
        with mock.patch.dict(os.environ, {BR.VOTE_LINE_SINGLE_ENV: "1"}):
            L._d_extend_flip_rate_env(ns, lines.append, "D")
            first = ns.env_d
            L._d_extend_flip_rate_env(ns, lines.append, "D")
        self.assertEqual(L.parse_group_env(ns.env_d)[BR.VOTE_FIXED_ENV], "155,155,155")
        self.assertEqual(ns.env_d, first)   # idempotent
        self.assertTrue(any(BR.VOTE_FIXED_MARKER in ln for ln in lines), lines)

    def test_operator_wins_switch_off_and_dual_write_nothing(self):
        ns = _ns(env_d=f"FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,1100,800;{BR.VOTE_FIXED_ENV}=200")
        L._d_extend_flip_rate_env(ns, [].append, "D")
        self.assertEqual(L.parse_group_env(ns.env_d)[BR.VOTE_FIXED_ENV], "200")
        with mock.patch.dict(os.environ, {BR.VOTE_LINE_SINGLE_ENV: "0"}):
            ns = _ns()
            L._d_extend_flip_rate_env(ns, [].append, "D")
        self.assertNotIn(BR.VOTE_FIXED_ENV, ns.env_d)
        ns = _ns(dual_share=True)
        L._d_extend_flip_rate_env(ns, [].append, "D")
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

    def _thr(self, rows, fixed_env="155", src=(700.0, "MEASURED-D:700"), static="1200,1100,800"):
        env = {"FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB": "0.3091,0.3091,0.3091"}
        if static is not None:
            env["FLLIPER_PDFLIP_EXTEND_TRIM_MIB"] = static
        if fixed_env is not None:
            env[BR.VOTE_FIXED_ENV] = fixed_env
        got = []
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(ET, "_read_corridor_line", lambda: src), \
                mock.patch.object(ET, "maybe_trim", side_effect=lambda cuda, rank, thr, *a: got.append(thr)):
            for k in ("FLLIPER_PDFLIP_EXTEND_TRIM_MIB", BR.VOTE_FIXED_ENV):
                if k not in env:
                    os.environ.pop(k, None)
            ET.reset_for_tests()
            ET.before_extend(types.SimpleNamespace(tp_rank=2), _Batch(rows))
        return got

    def test_the_need_raises_the_threshold(self):
        # TP2 static 800; 1000 rows need 555 + 309.1
        self.assertAlmostEqual(self._thr(1000)[0], LINE + 0.3091 * 1000, places=3)
        # the static threshold stays where it is higher
        self.assertEqual(self._thr(1)[0], 800.0)

    def test_without_the_record_or_on_an_unmeasured_card_only_the_static(self):
        self.assertEqual(self._thr(1000, fixed_env=None), [800.0])
        self.assertEqual(self._thr(1000, src=(1024.0, "stated-law:1024")), [800.0])
        self.assertEqual(self._thr(1000, fixed_env=None, static=None), [])

    def test_the_need_trims_without_a_static_threshold(self):
        self.assertAlmostEqual(self._thr(1000, static=None)[0], LINE + 0.3091 * 1000, places=3)


if __name__ == "__main__":
    import unittest

    unittest.main()
