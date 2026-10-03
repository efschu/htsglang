"""Resplit je Flip, Schritt 1 (27B, 29.09.): the D preset of the coming epoch
is read from MEASURED FormMeasures v3 cells, and the 27B-INT8 matrix record.

Pinned without a GPU (design: /spinning/gpu-arb/docs/DYN_D_RESHARD.md, Nachtrag
29.09. "Resplit je Flip"):
  * THE RECORD -- qwen27b carries FORM_MATRIX_AXES for fmt int8 (the FM-v3
    import refused 27B-INT8 before): the C forms name their MLP vector in the
    weights axis, A/A_KV/B77/B88 are "nicht startbar" with a reason, the
    capacity is the one computed from EFFECTIVE 759616;
  * MEASURED, NOT MODELLED -- with the 26.09. A/B cells (drq vs R+T, bar1,
    dkr27bint8drqbar1i8drq109262005 / dkr27bint8drtbar1i8rt109261955) the
    leader leaves R+T for drq where drq is >= min_gain faster and stays where
    it is not; from drq it never goes back;
  * NEVER INTERPOLATED -- a missing cell keeps the preset in force and is
    named; an "unmoeglich" cell (KV capacity) is never picked; an unknown text
    asks every text (worst case); D-prefill keeps the preset in force;
  * ONE ROW -- the measured decision still produces the one ReshardRow the
    followers adopt; without a measured source the model path is unchanged.
"""

from __future__ import annotations

import unittest

from sglang.srt.weg2 import d_reshard as D
from sglang.srt.weg2 import form_measures as fm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IDENT = fm.Identity(model="qwen27b", precision=("int8", "int8", "int8"),
                    cards=("GPU-31d7ef41", "GPU-5c648f96", "GPU-62dbbae1"),
                    links=("x8", "x8", "x4"), image_rev="2a7f992c14",
                    power_limit_w=(("0", 400.0), ("1", 230.0), ("2", 230.0)))
OTHER = fm.Identity(model="qwen27b", precision=("int8", "int8", "int8"), image_rev="424346f693")

DRQ = D.Preset("drq", (652, 218, 218))
RT = D.Preset("rt", (707, 191, 190))
H = D.Preset("h", (584, 252, 252))

#: v2 SCHRITT, ms per D round = max over ranks gpu-ms, warm, mean
#: (/spinning/docker-acceptance/27b/measure27b_dwin/compare_09261949.txt)
#: (text, depth, bs) -> (R+T, drq). 128k is left out: v3 has no 128k point and
#: buckets it UP to 240k, where its own cell already stands.
AB_0926 = {
    ("code", "10k", 1): (28.92, 28.08), ("code", "10k", 2): (33.84, 33.30),
    ("prose", "10k", 1): (28.98, 28.04), ("prose", "10k", 2): (33.90, 32.83),
    ("thinking", "10k", 1): (29.20, 28.04), ("thinking", "10k", 2): (34.03, 33.09),
    ("code", "32k", 1): (29.88, 28.90), ("code", "32k", 2): (35.33, 34.48),
    ("code", "240k", 1): (35.86, 35.08), ("code", "240k", 2): (46.99, 45.59),
}


def _geom():
    return D.rc9_geometry("int8")


def _spec(boot="drq", presets=(DRQ, RT), min_gain=0.02):
    s = D.ReshardSpec(policy=D.POLICY_WAKE, base=D.RC9_BASE, presets=tuple(presets), boot=boot,
                      min_gain=min_gain, fmt="int8", objective=D.OBJECTIVE_SEGMENT)
    s.validate(_geom())
    return s


def _matrix():
    return fm.matrix_spec("qwen27b", "int8")


def _store(cells=AB_0926, ident=IDENT):
    m = _matrix()
    forms = {f.name: f for f in m.forms}
    st = fm.FormMeasuresV3()
    for (text, depth, bs), (rt_ms, drq_ms) in cells.items():
        for name, ms in (("C_RT", rt_ms), ("C", drq_ms)):
            ax = dict(forms[name].axes)
            ax.update({"form": name, "bs": bs, "depth": depth, "text": text, "temp": "warm"})
            st.upsert(ident, fm.Cell(axes=ax, state=fm.STATE_MEASURED, round_ms_median=ms, n=300,
                                     at="2026-09-26T20:05:00Z", image_rev="2a7f992c14",
                                     source="compare_09261949.txt"))
    return st


def _measured(boot="drq", presets=(DRQ, RT), boot_form=None, store=None, ident=IDENT, min_gain=0.02):
    spec = _spec(boot, presets, min_gain)
    boot_form = boot_form or {"drq": "C", "rt": "C_RT"}[boot]
    return spec, D.MeasuredForms(spec, _matrix(), store if store is not None else _store(), ident, boot_form)


class TestInt8Record(unittest.TestCase):
    def test_record_exists_and_names_its_forms(self):
        m = _matrix()
        forms = {f.name: f for f in m.forms}
        self.assertEqual(set(forms), {"C", "C_RT", "C_T", "C_H", "C_SEG", "A", "A_KV", "B77", "B88"})
        self.assertEqual(D.mlp_of_weights_axis(forms["C"].axes["weights"]), (652, 218, 218))
        self.assertEqual(forms["C"].arm, "drq")
        for n in ("A", "A_KV", "B77", "B88"):
            self.assertFalse(forms[n].startable, n)
            self.assertTrue(forms[n].why_not, n)
        for n in ("C", "C_RT", "C_T", "C_H", "C_SEG"):
            self.assertTrue(forms[n].startable, n)
            self.assertTrue(all(x == "bar1" for x in [forms[n].axes["transport"]]))
        self.assertEqual(m.texts, ("code", "prose", "thinking"))

    def test_capacity_from_effective_759616(self):
        m = _matrix()
        for d, tok in fm.DEPTH_POINTS:
            self.assertEqual(m.capacity["C"][d], min(6, 759616 // tok), d)
        f = {x.name: x for x in m.forms}["C"]
        ax = dict(f.axes, form="C", bs=4, depth="240k", text="code", temp="warm")
        self.assertEqual(fm.structural_state(m, f, ax)[0], fm.STATE_IMPOSSIBLE_CAPACITY)

    def test_nvfp4_record_untouched(self):
        self.assertEqual({f.name for f in fm.matrix_spec("qwen27b", "nvfp4").forms},
                         {"A", "B77", "B88", "A_KV", "C"})


class TestMeasuredChoice(unittest.TestCase):
    def test_preset_to_form_map_only_mlp_differs(self):
        # h=584 is carried by C_T (same placement) -- not by C_H (other token placement)
        _, mf = _measured(presets=(DRQ, RT, H))
        self.assertEqual({k: v.name for k, v in mf.form_of.items()}, {"drq": "C", "rt": "C_RT", "h": "C_T"})

    def test_boot_form_must_carry_the_boot_preset(self):
        with self.assertRaises(D.ReshardError):
            _measured(boot_form="C_RT")
        with self.assertRaises(D.ReshardError):
            _measured(boot_form="Z")

    def test_from_rt_switches_only_over_min_gain(self):
        _, mf = _measured(boot="rt")
        ch = mf.choose(D.LoadClass("decode", 1, 10000, text="code"), "rt")
        self.assertEqual((ch.preset, ch.form), ("drq", "C"))
        self.assertAlmostEqual(ch.gain, (28.92 - 28.08) / 28.92, places=6)
        # code@10k bs2: drq only 1.6 % faster -> R+T stays
        ch = mf.choose(D.LoadClass("decode", 2, 10000, text="code"), "rt")
        self.assertEqual(ch.preset, "rt")
        self.assertIn("< min_gain", ch.reason)
        ch = mf.choose(D.LoadClass("decode", 1, 200000, text="code"), "rt")  # -> 240k point
        self.assertEqual(ch.preset, "drq")

    def test_from_drq_never_back(self):
        _, mf = _measured()
        for (text, depth, bs) in AB_0926:
            ctx = dict(fm.DEPTH_POINTS)[depth]
            ch = mf.choose(D.LoadClass("decode", bs, ctx, text=text), "drq")
            self.assertEqual(ch.preset, "drq", (text, depth, bs))
            self.assertLess(ch.gain, 0.0)

    def test_unknown_text_is_worst_case_over_texts(self):
        _, mf = _measured(boot="rt")
        # 10k bs1: gains code 2.9 %, prose 3.2 %, thinking 4.0 % -> worst 2.9 % -> switch
        ch = mf.choose(D.LoadClass("decode", 1, 10000), "rt")
        self.assertEqual(ch.preset, "drq")
        self.assertAlmostEqual(ch.gain, (28.92 - 28.08) / 28.92, places=6)
        # 32k: only code measured -> prose/thinking missing -> stays, gap named
        ch = mf.choose(D.LoadClass("decode", 1, 30000), "rt")
        self.assertEqual(ch.preset, "rt")
        self.assertIn("C_RT@prose@32k/bs1", ch.missing)

    def test_missing_cell_never_interpolated(self):
        _, mf = _measured(boot="rt")
        # 97k has no cell (neighbours 32k and 240k do): stays, names both forms' gap
        ch = mf.choose(D.LoadClass("decode", 1, 90000, text="code"), "rt")
        self.assertEqual(ch.preset, "rt")
        self.assertEqual(ch.missing, ("C_RT@code@97k/bs1",))
        # bs3 was never measured
        ch = mf.choose(D.LoadClass("decode", 3, 10000, text="code"), "rt")
        self.assertEqual(ch.preset, "rt")

    def test_impossible_never_picked(self):
        cells = dict(AB_0926)
        cells[("code", "240k", 4)] = (60.0, 30.0)  # beyond the KV of every C form (cap 3)
        _, mf = _measured(boot="rt", store=_store(cells))
        ch = mf.choose(D.LoadClass("decode", 4, 245760, text="code"), "rt")
        self.assertEqual(ch.preset, "rt")
        self.assertIn("impossible", ch.reason)

    def test_other_identity_is_a_miss(self):
        _, mf = _measured(boot="rt", ident=OTHER)
        ch = mf.choose(D.LoadClass("decode", 1, 10000, text="code"), "rt")
        self.assertEqual(ch.preset, "rt")
        self.assertTrue(ch.missing)

    def test_d_prefill_keeps_preset(self):
        _, mf = _measured(boot="rt")
        ch = mf.choose(D.LoadClass("prefill", 1, 4096), "rt")
        self.assertEqual(ch.preset, "rt")
        self.assertIn("D-prefill has no cell", ch.reason)

    def test_too_deep_keeps_preset(self):
        _, mf = _measured(boot="rt")
        self.assertEqual(mf.choose(D.LoadClass("decode", 1, 300000, text="code"), "rt").preset, "rt")


class TestLeaderWithMeasured(unittest.TestCase):
    def test_one_row_every_follower(self):
        spec, mf = _measured(boot="rt")
        g = _geom()
        lead = D.LeaderCursor(g, D.rc9_calib("int8"), spec, D.token_share(D.RC9_TOKEN_VECTOR["int8"]), measured=mf)
        fol = [D.FollowerCursor(g, spec, r) for r in range(3)]
        seq = [(1, 10000, "code", "drq"), (2, 10000, "code", "drq"), (1, 90000, "code", "drq")]
        for ep, (bs, ctx, text, want) in enumerate(seq):
            row = lead.decide(ep, D.LoadClass("decode", bs, ctx, text=text))
            self.assertEqual(row.preset, want)
            self.assertTrue(row.reason.startswith("measured:"))
            tiles = [f.adopt(row) for f in fol]
            self.assertEqual(sum(s for _, s in tiles), g.inter)
        # once on drq: code@10k bs2 would not have justified the switch, and from drq nothing goes back
        self.assertEqual(lead.current, "drq")

    def test_without_measured_model_path_unchanged(self):
        spec = _spec(boot="rt")
        g = _geom()
        ts = D.token_share(D.RC9_TOKEN_VECTOR["int8"])
        a = D.LeaderCursor(g, D.rc9_calib("int8"), spec, ts)
        load = D.LoadClass("decode", 1, 10000)
        want = D.choose(g, D.rc9_calib("int8"), spec.base, spec.presets, load, ts, "rt", spec.min_gain)
        row = a.decide(0, load)
        self.assertEqual((row.preset, row.reason), (want, load.key()))

    def test_measured_for_other_spec_refused(self):
        _, mf = _measured(boot="rt")
        other = _spec(boot="rt", min_gain=0.05)
        with self.assertRaises(D.ReshardError):
            D.LeaderCursor(_geom(), D.rc9_calib("int8"), other, (0.4, 0.3, 0.3), measured=mf)

    def test_loadclass_key_unchanged_by_text(self):
        self.assertEqual(D.LoadClass("decode", 2, 32768, text="code").key(), D.LoadClass("decode", 2, 32768).key())


if __name__ == "__main__":
    unittest.main()
