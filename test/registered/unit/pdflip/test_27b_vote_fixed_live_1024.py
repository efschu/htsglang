# SPDX-License-Identifier: Apache-2.0
"""VOTE-FIXED-LIVE 24.10. (follow-up S4 of the VOTE-LINE-SINGLE review): the rank measures F itself.

VOTE-LINE-SINGLE (9a25171a59) priced the 27B width-vote line ``near-OOM 400 + F`` with F a RECORD: 238 MiB, the max
of ``post - end - rate x rows`` over six INT8-drei boots, measured offline (budget_rest.vote_fixed_from_passes) and
gated to that one form -- int4ar / NVFP4 duo (measured 223-252) keep the floor line until someone writes their record.

VOTE-FIXED-LIVE moves the SAME instrument into the rank (pdflip/vote_fixed_live.Meter, fed by width_vote / the
forward trim / vram_peak_window / the draft forward), persists every reading per form (the H87 footprint -- quant
format included -- the D inventory, KV dtype, draft) and seeds the next boot from the form's store.

What this file pins, against the boots (fixtures: the six INT8-drei boots of the record, the out-of-sample INT8 boot
_1024_110644 (rc41, 221), NVFP4 duo _1024_100906 (232), int4ar _091424 (252, 11 passes), _223241 (223), _003732 (226)):

* the meter IS the record's instrument: replaying each boot's paired passes through the meter's events gives the
  record's F per boot (at the start rate; _004350 voted partly with a measured rate, at the start rate F is 216);
* rule 1 raise only, rule 2 every reading counts upward, rule 3 the gate counts boots (3 x >= 20 passes) -- the
  _080355 case (one boot, 481 passes, 155; failed on _003738 238) never opens it; rule 4 one group F over the packed
  reduce; rule 5 the form knows the quant format;
* OOM direction: for EVERY three-boot store of INT8 drei the worst out-of-store pass still ends at ~360 MiB (> 0, under
  the 400 edge by 40 -- named: the gate is an empirical rule, not a bound; the in-boot ratchet raises F after it).
"""

import contextlib
import itertools
import json
import math
import os
import sys
import tempfile
import types
import unittest
import unittest.mock as mock

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import corridor_guard as CG  # noqa: E402
from flliper.srt.pdflip import budget_rest as BR  # noqa: E402
from flliper.srt.pdflip import extend_trim as ET  # noqa: E402
from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402
from flliper.srt.pdflip import vote_fixed_live as V  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_27b_int8_tp1corridor_1024 as PRE  # noqa: E402  (the predecessor's live-boot rig)

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "fixtures", "vote_line_single_1024")
NEW = os.path.join(HERE, "fixtures", "vote_fixed_live_1024")
START = 0.3091     # D_EXTEND_CAP_PER_ROW_MIB, the vote's start rate on every D rank
RECORD = 238       # D_EXTEND_VOTE_FIXED_MIB (static record, INT8 drei)
DREI_MODEL = "Qwen3.8-27B-INT8-gdncov-vocabembed"
NVFP4_MODEL = "Qwen3.8-27B-NVFP4-RadixArk"

#: chronological INT8-drei boots (fixture stamp -> dir), then the duo / int4ar boots
INT8 = (("001910", OLD), ("004350", OLD), ("003738", OLD), ("010424", OLD), ("015348", OLD), ("080355", OLD),
        ("110644", NEW))
OTHER = (("duo100906", NEW), ("int4ar091424", NEW), ("int4ar223241", NEW), ("int4ar003732", NEW))
#: F per boot at the START rate (the meter's pricing); equals the offline record where the vote kept the start rate
F_START = {"001910": 0, "004350": 216, "003738": 238, "010424": 197, "015348": 153, "080355": 155, "110644": 221,
           "duo100906": 232, "int4ar091424": 252, "int4ar223241": 223, "int4ar003732": 226}
#: counted (trimmed, non-first) passes per boot over its ranks
COUNTED = {"001910": 5, "004350": 41, "003738": 21, "010424": 44, "015348": 26, "080355": 481, "110644": 253,
           "duo100906": 833, "int4ar091424": 11, "int4ar223241": 47, "int4ar003732": 27}


def _load(stamp, d):
    with open(os.path.join(d, f"passes_{stamp}.json")) as fh:
        doc = json.load(fh)
    cols = doc["columns"]
    out = {}
    for row in doc["rows"]:
        x = dict(zip(cols, row))
        out.setdefault(int(x["rank"]), []).append(BR.ExtendPass(
            rank=int(x["rank"]), vote_ts=x["vote_ts"], chunk_ts=x["chunk_ts"], post=float(x["post"]),
            rate=float(x["rate"]), line=float(x["line"]), cap=int(x["cap"]), rows=int(x["rows"]), end=float(x["end"]),
            fwd_before=None if x["fwd_before"] is None else float(x["fwd_before"]), released=float(x["released"])))
    return out


ALL = {s: _load(s, d) for s, d in INT8 + OTHER}


def _meters(stamp):
    return {r: V.feed_passes(V.Meter(rank=r), ps) for r, ps in ALL[stamp].items()}


def _group_f(stamp, rate=START):
    fs = [m.fixed_mib(rate) for m in _meters(stamp).values()]
    fs = [f for f in fs if f is not None]
    return None if not fs else int(math.ceil(max(fs) - 1e-9))


def _counted_passes(stamp):
    return [p for ps in ALL[stamp].values() for p in ps[1:] if p.trimmed]


def _import(root, form, stamps):
    for s in stamps:
        V.import_passes(root, form, s, ALL[s], rate=START)


def _verdict(root, form):
    return V.aggregate(V.load_docs(root, form), START)


class TheMeterIsTheRecordsInstrument(CustomTestCase):
    def test_replay_gives_the_record_per_boot(self):
        got = {s: _group_f(s) for s in ALL}
        self.assertEqual(got, F_START)
        # where the vote kept the start rate the meter IS the offline record (byte for byte, ceil)
        for s in ALL:
            if {p.rate for ps in ALL[s].values() for p in ps} == {START}:
                self.assertEqual(got[s], BR.vote_fixed_from_passes(ALL[s])[0], s)
        # _004350 voted partly with a measured rate (0.4458 / 0.3204): at the start rate F is larger (216 vs 191)
        self.assertEqual(BR.vote_fixed_from_passes(ALL["004350"])[0], 191)
        self.assertEqual(max(got[s] for s, _ in INT8[:6]), RECORD)

    def test_the_meter_counts_the_records_denominator(self):
        got = {s: sum(m.counted for m in _meters(s).values()) for s in ALL}
        self.assertEqual(got, COUNTED)
        self.assertEqual(sum(got[s] for s, _ in INT8[:6]), 618)   # the record's 618 passes

    def test_the_pareto_front_is_exact_at_every_rate(self):
        for s in ("080355", "004350", "duo100906"):
            for m in _meters(s).values():
                front = V.pareto(m.readings)
                self.assertLessEqual(len(front), len(m.readings))
                for rate in (0.0, 0.1, START, 0.4458, 1.0, 3.0):
                    a, b = V.fixed_at(front, rate), m.fixed_mib(rate)
                    self.assertTrue((a is None and b is None) or abs(a - b) < 1e-9, (s, rate, a, b))

    def test_the_event_order(self):
        m = V.Meter(rank=1)
        m.on_vote(1000, START), m.on_trim(), m.on_chunk(10, 700)
        self.assertIsNone(m.on_vote(1000, START))           # the first paired pass: resident growth, not counted
        self.assertEqual((m.paired, m.counted), (1, 0))
        m.on_trim(), m.on_chunk(10, 700)
        m.on_draft(650)                                     # the draft extend of the same pass lowers the end
        self.assertIsNotNone(m.on_target_decode())
        self.assertEqual(m.readings, {10: 350.0})
        m.on_vote(1000, START), m.on_target_decode(), m.on_chunk(20, 100)    # decode between vote and chunk: unpaired
        m.on_vote(1000, START), m.on_trim(), m.on_window("flip"), m.on_chunk(30, 100)   # a flip unpairs
        m.on_vote(1000, START), m.on_chunk(40, 100), m.commit()             # untrimmed: not counted
        self.assertEqual((m.counted, m.untrimmed, m.readings), (1, 1, {10: 350.0}))
        m.on_vote(1000, START), m.on_trim(), m.on_window("round"), m.on_chunk(50, 600), m.commit()  # round: paired
        m.on_vote(2000, START), m.on_vote(900, START), m.on_trim(), m.on_chunk(60, 600), m.commit()  # newest vote
        self.assertEqual(m.readings, {10: 350.0, 50: 400.0, 60: 300.0})
        self.assertEqual(m.counted, 3)


class TheGateCountsBoots(CustomTestCase):
    """Rule 2 (every reading counts upward) and rule 3 (MIN_BOOTS boots of MIN_PASSES_PER_BOOT passes)."""

    def test_int8_chronologically_opens_at_the_max_of_all_boots(self):
        with tempfile.TemporaryDirectory() as root:
            states = []
            for s, _ in INT8:
                _import(root, "drei", [s])
                v = _verdict(root, "drei")
                states.append((s, v.gate_open, v.fixed_mib, len(v.qualifying)))
        self.assertEqual(states, [
            ("001910", False, 0, 0),       # 5 passes: no qualifying boot
            ("004350", False, 216, 1),
            ("003738", False, 238, 2),
            ("010424", True, 238, 3),      # opens with 238 = the max of every boot, the out-of-sample one included
            ("015348", True, 238, 4),
            ("080355", True, 238, 5),
            ("110644", True, 238, 6),      # rc41 INT8 drei, out of sample: 221 <= 238
        ])

    def test_one_boot_of_many_passes_never_opens(self):
        """_080355: 481 counted passes, F 155 -- the record's own counter-example (failed on _003738, 238)."""
        with tempfile.TemporaryDirectory() as root:
            _import(root, "drei", ["080355"])
            v = _verdict(root, "drei")
        self.assertEqual((v.gate_open, v.fixed_mib, v.passes), (False, 155, 481))
        self.assertTrue(any(p.excess > 155 for p in _counted_passes("003738")))

    def test_a_short_boots_reading_still_counts_upward(self):
        """int4ar: _091424 counts 11 passes (not qualifying) but holds the form's max 252 -- the store keeps it."""
        with tempfile.TemporaryDirectory() as root:
            _import(root, "int4ar", ["int4ar091424", "int4ar223241", "int4ar003732"])
            v = _verdict(root, "int4ar")
            self.assertEqual((v.gate_open, v.fixed_mib, v.qualifying), (False, 252, ("int4ar003732", "int4ar223241")))
            V.import_passes(root, "int4ar", "int4ar003732-again", ALL["int4ar003732"], rate=START)
            v = _verdict(root, "int4ar")
        self.assertEqual((v.gate_open, v.fixed_mib), (True, 252))
        self.assertIn("int4ar091424", v.worst)

    def test_duo_one_boot_stays_closed(self):
        with tempfile.TemporaryDirectory() as root:
            _import(root, "duo", ["duo100906"])
            v = _verdict(root, "duo")
        self.assertEqual((v.gate_open, v.fixed_mib, v.passes), (False, 232, 833))

    def test_no_store_of_three_boots_lets_a_pass_reach_oom(self):
        """OOM direction, measured: for every three qualifying INT8 boots as the store, every counted pass of every
        INT8 boot at the bound (post = line + rate x rows) ends at 400 + F_store - excess > 0. Worst: 359.9 MiB
        (store 010424 / 015348 / 080355 = 197 against _003738's 237.1) -- under the 400 edge, named, not OOM."""
        qual = [s for s, _ in INT8 if COUNTED[s] >= V.MIN_PASSES_PER_BOOT]
        self.assertEqual(len(qual), 6)
        worst = None
        for sub in itertools.combinations(qual, V.MIN_BOOTS):
            f = max(F_START[s] for s in sub)
            for s, _ in INT8:
                for p in _counted_passes(s):
                    end = CG.NEAR_OOM_MIB + f - (p.post - p.end - START * p.rows)
                    if worst is None or end < worst[0]:
                        worst = (end, sub, s)
        self.assertGreater(worst[0], 0)
        self.assertEqual(round(worst[0]), 360)
        self.assertEqual(worst[1:], (("010424", "015348", "080355"), "003738"))

    def test_a_lower_boot_never_lowers_the_store(self):
        with tempfile.TemporaryDirectory() as root:
            prev = 0
            for s in ("003738", "080355", "015348", "001910"):
                _import(root, "drei", [s])
                v = _verdict(root, "drei")
                self.assertGreaterEqual(v.fixed_mib, prev)
                prev = v.fixed_mib
        self.assertEqual(prev, 238)

    def test_the_store_survives_a_broken_file(self):
        with tempfile.TemporaryDirectory() as root:
            _import(root, "drei", ["003738"])
            with open(os.path.join(V.store_dir(root, "drei"), "junk.tp9.json"), "w") as fh:
                fh.write("{not json")
            v = _verdict(root, "drei")
            self.assertEqual(v.fixed_mib, 238)
            self.assertEqual(V.aggregate(V.load_docs(root, "absent"), START).fixed_mib, None)


class TheFormKnowsTheQuant(CustomTestCase):
    """Rule 5: the form digest carries the H87 footprint (quant sidecars, dtypes, shapes) -- the corridor footprint
    digest did not (review VOTE-LINE-SINGLE)."""

    def test_material(self):
        inv = ("RTX5090", "RTX3080", "RTX3080")
        base = V.form_material("qwen27b", "fpINT8", DREI_MODEL, inv, "fp8_e4m3", "dflash:0")
        self.assertIsNone(V.form_material("qwen27b", None, DREI_MODEL, inv))
        self.assertIsNone(V.form_material("qwen27b", "fp", DREI_MODEL, ()))
        d = V.form_digest(base)
        self.assertEqual(d, V.form_digest(dict(base)))
        for other in (V.form_material("qwen27b", "fpNVFP4", DREI_MODEL, inv, "fp8_e4m3", "dflash:0"),
                      V.form_material("qwen27b", "fpINT8", DREI_MODEL, inv[:2], "fp8_e4m3", "dflash:0"),
                      V.form_material("qwen27b", "fpINT8", DREI_MODEL, inv, "auto", "dflash:0"),
                      V.form_material("qwen27b", "fpINT8", DREI_MODEL, inv, "fp8_e4m3", "mtp:3")):
            self.assertNotEqual(V.form_digest(other), d)

    def test_the_launcher_form_reads_the_footprint(self):
        fps = {DREI_MODEL: "fpINT8", NVFP4_MODEL: "fpNVFP4"}
        with mock.patch.object(F, "footprint_key", lambda m: (fps.get(F.model_key(m)), "fake")):
            a = L.d_extend_vote_fixed_live_form(F.PROFILE_QWEN27B, PRE._cards(), "/m/" + DREI_MODEL)[0]
            b = L.d_extend_vote_fixed_live_form(F.PROFILE_QWEN27B, PRE._cards(), "/m/" + NVFP4_MODEL)[0]
            c = L.d_extend_vote_fixed_live_form(F.PROFILE_QWEN27B, PRE._cards(), "/m/unknown")
            d = L.d_extend_vote_fixed_live_form(F.PROFILE_QWEN27B, None, "/m/" + DREI_MODEL)
        self.assertTrue(a and b and a != b)
        self.assertIsNone(c[0])
        self.assertIn("no footprint", c[1])
        self.assertIsNone(d[0])
        with mock.patch.dict(os.environ, {V.ENV_ON: "0"}):
            self.assertIsNone(L.d_extend_vote_fixed_live_form(F.PROFILE_QWEN27B, PRE._cards(), "/m/" + DREI_MODEL)[0])


@contextlib.contextmanager
def _store(fps=None):
    fps = {DREI_MODEL: "fpINT8", NVFP4_MODEL: "fpNVFP4"} if fps is None else fps
    with tempfile.TemporaryDirectory() as root, \
            mock.patch.dict(os.environ, {V.ENV_DIR: root, BR.VOTE_LINE_SINGLE_ENV: "1", V.ENV_ON: "1"}), \
            mock.patch.object(F, "footprint_key", lambda m: (fps.get(F.model_key(m)), "fake")):
        yield root


def _digest(model, cards):
    return L.d_extend_vote_fixed_live_form(F.PROFILE_QWEN27B, cards, "/m/" + model)[0]


def _doc(root, form, boot, rank, rows_cost, counted=30, card="RTX3080"):
    m = V.Meter(rank=rank)
    m.readings = dict(rows_cost)
    m.counted = counted
    V.write_doc(V.doc_path(root, form, boot, rank), m.snapshot(START, form=form, boot=boot, card=card))


class TheLauncherSeedsFromTheStore(CustomTestCase):
    def test_a_higher_store_reading_raises_the_static_record(self):
        with _store() as root:
            d = _digest(DREI_MODEL, PRE._cards())
            _doc(root, d, "b1", 1, {61: 280.0})                # 280 - 0.3091 x 61 = 261.1 -> 262
            vals, src = L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, 3, PRE._cards(), "/m/" + DREI_MODEL)
        self.assertEqual(vals, [262.0] * 3)
        self.assertIn(V.MARKER, src)

    def test_a_lower_store_keeps_the_record(self):
        with _store() as root:
            d = _digest(DREI_MODEL, PRE._cards())
            _import(root, d, ["110644"])                       # 221 out of sample
            vals, src = L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, 3, PRE._cards(), "/m/" + DREI_MODEL)
        self.assertEqual(vals, [float(RECORD)] * 3)
        self.assertNotIn(V.MARKER, src)

    def test_a_form_without_record_waits_for_the_gate(self):
        duo = PRE.U5090, PRE.U_TP1
        cards = L.order_cards([c for c in PRE._cards() if c.uuid in duo])
        with _store() as root:
            d = _digest(NVFP4_MODEL, cards)
            _import(root, d, ["duo100906"])
            vals, why = L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, 2, cards, "/m/" + NVFP4_MODEL)
            self.assertIsNone(vals)
            self.assertTrue(why.startswith("F unmeasured for form"), why)
            self.assertIn("gate closed", why)
            V.import_passes(root, d, "duo-b2", ALL["duo100906"], rate=START)
            V.import_passes(root, d, "duo-b3", ALL["int4ar223241"], rate=START)
            vals, src = L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, 2, cards, "/m/" + NVFP4_MODEL)
        self.assertEqual(vals, [232.0, 232.0])
        self.assertIn("gate OPEN", src)

    def test_the_budget_books_the_raised_line(self):
        with _store() as root, mock.patch.dict(os.environ, {F.FORM_ENV: _form_env(DREI_MODEL)}):
            d = _digest(DREI_MODEL, PRE._cards())
            _doc(root, d, "b1", 1, {61: 280.0})
            _, budgets, kw = PRE._d_pass([], PRE._floors(), single="1")
        delta = -(-int(400 + 262 - 300) // 8) * 8          # ceil8(662 - 300) = 368
        self.assertEqual([h - b for h, b in zip(PRE.HEAD_BUDGETS, budgets)], [delta] * 3)

    def test_the_env_hands_the_ranks_the_store(self):
        with _store() as root:
            ns = _ns()
            lines = []
            L._d_extend_flip_rate_env(ns, lines.append, "D", PRE._cards())
            env = L.parse_group_env(ns.env_d)
            self.assertEqual(env[V.ENV_FORM], _digest(DREI_MODEL, PRE._cards()))
            self.assertEqual(env[V.ENV_CARDS], "RTX5090,RTX3080,RTX3080")
            self.assertEqual(env[V.ENV_BOOT], "boot_t")
            self.assertEqual(env[V.ENV_DIR], root)
            self.assertEqual(env[BR.VOTE_FIXED_ENV], "238,238,238")
            # a form in cold start: the meter env, no seed -- the ranks keep the floor line and measure
            duo = L.order_cards([c for c in PRE._cards() if c.uuid in (PRE.U5090, PRE.U_TP1)])
            ns = _ns(model="/m/" + NVFP4_MODEL)
            L._d_extend_flip_rate_env(ns, lines.append, "D", duo)
            env = L.parse_group_env(ns.env_d)
            self.assertIn(V.ENV_FORM, env)
            self.assertNotIn(BR.VOTE_FIXED_ENV, env)

    def test_switch_off_is_the_static_record(self):
        with _store() as root, mock.patch.dict(os.environ, {V.ENV_ON: "0"}):
            d = V.form_digest(V.form_material(F.PROFILE_QWEN27B, "fpINT8", DREI_MODEL,
                                              ("RTX5090", "RTX3080", "RTX3080")))
            _doc(root, d or "x", "b1", 1, {61: 280.0})
            vals, _ = L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, 3, PRE._cards(), "/m/" + DREI_MODEL)
            ns = _ns()
            L._d_extend_flip_rate_env(ns, [].append, "D", PRE._cards())
        self.assertEqual(vals, [float(RECORD)] * 3)
        self.assertNotIn(V.ENV_FORM, ns.env_d)

    def test_no_footprint_is_no_store(self):
        with _store(fps={}) as root:
            _doc(root, "anything", "b1", 1, {61: 900.0})
            vals, _ = L.d_extend_vote_fixed_record(F.PROFILE_QWEN27B, 3, PRE._cards(), "/m/" + DREI_MODEL)
            ns = _ns()
            lines = []
            L._d_extend_flip_rate_env(ns, lines.append, "D", PRE._cards())
        self.assertEqual(vals, [float(RECORD)] * 3)
        self.assertNotIn(V.ENV_FORM, ns.env_d)
        self.assertTrue(any("store off" in ln for ln in lines), lines)


def _form_env(model):
    return F.PdFlipForm(arch="dense", experts="none", draft="dflash", p_draft="cold", kv="paged_dcp", flip="family",
                        vision="transient", profile="qwen27b", model=model).env_value()


def _ns(model="/m/" + DREI_MODEL, **kw):
    base = dict(profile="qwen27b", model=model, extra_d="", env_d="FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,1100,800",
                dual_layout=False, dual_share=False, tag="boot_t")
    base.update(kw)
    return types.SimpleNamespace(**base)


class _Cuda:
    def __init__(self, free):
        self.free = free

    def mem_get_info(self, *a):
        return self.free * float(1 << 20), 20480 * float(1 << 20)


@contextlib.contextmanager
def _rank(seed="238", src=(701.0, "MEASURED-D:701"), live="1", extra=None):
    env = {"FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB": "0.3091,0.3091,0.3091", V.ENV_ON: live}
    if seed is not None:
        env[BR.VOTE_FIXED_ENV] = seed
    env.update(extra or {})
    with mock.patch.dict(os.environ, env), mock.patch.object(ET, "_read_corridor_line", lambda: src):
        if seed is None:
            os.environ.pop(BR.VOTE_FIXED_ENV, None)
        for k in (V.ENV_FORM, V.ENV_DIR):
            if k not in env:
                os.environ.pop(k, None)
        ET.reset_for_tests()
        try:
            yield
        finally:
            ET.reset_for_tests()


def _pass(rank, post, rows, end, trimmed=True, draft=None):
    ET.live_note_vote(rank, post, START)
    if trimmed:
        ET.live_note_trim()
    ET.live_note_window("chunk", rows, end)
    if draft is not None:
        ET.live_note_draft(_Cuda(draft))
    ET.live_note_target_decode()


class TheRankRaisesTheLine(CustomTestCase):
    """Rule 1 (raise only) and rule 4 (group max) on the rank."""

    def test_a_reading_above_the_seed_raises_the_line_at_the_next_vote(self):
        with _rank():
            self.assertEqual(ET.corridor_line_mib(1), 638.0)
            _pass(1, 1000, 10, 700)                          # first pass: not counted
            _pass(1, 1000, 10, 600)                          # cost 400 -> F 396.9 -> 397
            self.assertEqual(ET.corridor_line_mib(1), 400.0 + 397)
            self.assertEqual(ET.live_fixed_payload(), -397)
            self.assertAlmostEqual(ET.need_trim_threshold(1, 100), 797 + START * 100, places=6)
            _pass(1, 1000, 10, 900)                          # a cheaper pass never lowers it
            self.assertEqual(ET.corridor_line_mib(1), 797.0)

    def test_the_draft_extend_is_in_the_reading(self):
        with _rank():
            _pass(1, 1000, 10, 700)
            _pass(1, 1000, 10, 700, draft=500)               # cost 500 with the draft, 300 without
            self.assertEqual(ET.corridor_line_mib(1), 400.0 + math.ceil(500 - START * 10))

    def test_the_group_max_raises_every_rank_and_never_falls(self):
        with _rank():
            ET.live_fixed_group(-300)
            self.assertEqual(ET.corridor_line_mib(2), 700.0)
            ET.live_fixed_group(-100)
            self.assertEqual(ET.corridor_line_mib(2), 700.0)
            ET.live_fixed_group(0)
            self.assertEqual(ET.live_fixed_payload(), 0)     # no reading of its own: neutral in the MIN

    def test_cold_start_keeps_the_floor_line_and_measures(self):
        with _rank(seed=None):
            _pass(1, 1000, 10, 700)
            _pass(1, 1000, 10, 500)
            ET.live_fixed_group(-600)
            self.assertEqual(ET.corridor_line_mib(1), 701.0)                 # floor line, byte for byte
            self.assertEqual(ET._CACHE["live"].counted, 1)
            self.assertEqual(ET.live_fixed_payload(), -math.ceil(500 - START * 10))

    def test_an_unmeasured_card_keeps_its_law(self):
        with _rank(src=(1024.0, "UNMEASURED-FALLBACK:1024")):
            _pass(1, 1500, 10, 1000)
            _pass(1, 1500, 10, 900)
            self.assertEqual(ET.corridor_line_mib(1), 1024.0)

    def test_switch_off_is_the_seed(self):
        with _rank(live="0"):
            _pass(1, 1000, 10, 700)
            _pass(1, 1000, 10, 500)
            ET.live_fixed_group(-600)
            self.assertEqual(ET.corridor_line_mib(1), 638.0)
            self.assertEqual(ET.live_fixed_payload(), 0)

    def test_the_rank_persists_its_readings(self):
        with tempfile.TemporaryDirectory() as root:
            extra = {V.ENV_FORM: "f1", V.ENV_DIR: root, V.ENV_BOOT: "bootX", V.ENV_CARDS: "RTX5090,RTX3080,RTX3080"}
            with _rank(extra=extra):
                _pass(1, 1000, 10, 700)
                _pass(1, 1000, 61, 700)
                docs = V.load_docs(root, "f1")
            self.assertEqual(len(docs), 1)
            self.assertEqual((docs[0]["boot"], docs[0]["card"], docs[0]["rank"], docs[0]["counted"]),
                             ("bootX", "RTX3080", 1, 1))
            self.assertEqual(V.aggregate(docs, START).fixed_mib, math.ceil(300 - START * 61))

    def test_an_unwritable_store_is_named_once_and_never_raises(self):
        with tempfile.TemporaryDirectory() as root:
            blocker = os.path.join(root, "file")
            with open(blocker, "w") as fh:
                fh.write("x")
            extra = {V.ENV_FORM: "f1", V.ENV_DIR: blocker, V.ENV_BOOT: "b"}
            with _rank(extra=extra), self.assertLogs(ET.logger, level="WARNING") as cm:
                _pass(1, 1000, 10, 700)
                _pass(1, 1000, 10, 600)
                _pass(1, 1000, 10, 500)
                self.assertEqual(ET.corridor_line_mib(1), 400.0 + math.ceil(500 - START * 10))
            self.assertEqual(sum("store write failed" in m for m in cm.output), 1)


class TheHooksAreWired(CustomTestCase):
    def setUp(self):
        ET.reset_for_tests()

    def tearDown(self):
        ET.reset_for_tests()

    def test_the_window_close_feeds_the_meter(self):
        from flliper.srt.model_executor import vram_peak_window as VPW

        calls = []
        cuda = types.SimpleNamespace(
            memory_stats=lambda: {}, reset_peak_memory_stats=lambda: None,
            mem_get_info=lambda: (700 * float(1 << 20), 20480 * float(1 << 20)))
        with mock.patch.object(ET, "live_note_window", side_effect=lambda *a: calls.append(a)), \
                mock.patch("flliper.srt.pdflip.vram_actual.on_window", lambda *a, **k: None):
            VPW._close(cuda, 1, "chunk", f" rows=61", rows=61)
            VPW._close(cuda, 1, "flip", " leg=release")
        self.assertEqual(calls, [("chunk", 61, 700.0), ("flip", 0, 700.0)])

    def test_the_reduce_carries_one_slot(self):
        import inspect

        from flliper.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._update_uniform_pool_budget)
        a = src.index("_vfix_at = len(vals)")
        self.assertLess(src.index("_corridor_at = len(vals)"), a)
        self.assertLess(a, src.index("_limit_at = len(vals)"))
        self.assertIn("_pdflip_extend_trim.live_fixed_payload()", src)
        self.assertIn("_pdflip_extend_trim.live_fixed_group(int(t[_vfix_at]))", src)


if __name__ == "__main__":
    unittest.main()
