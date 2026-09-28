# SPDX-License-Identifier: Apache-2.0
"""#242r: P-KARTE and both wake-credit gates recompute ANY P cut from the
measured 29,11,8 reference -- or refuse it by name (W167). None of them may
drop out silently any more.

THE BEFUND (cut comparison 28.09., SCHNITTVERGLEICH-NF-0928.md). For every cut
other than the measured 29,11,8 the three planner gates fell away: the P-KARTE
raised ValueError on the stage mismatch and the launcher printed ENTFAELLT; the
two wake-credit gates (D->P H14, P->D H34) compared the reference key and
printed ENTFAELLT (fremde Geometrie). A dry run of 27,11,10 therefore checked
neither the chunk forward nor the flip.

THE FIX.
* P-KARTE (``p_card_chunk.recut_reference``): K0 per stage moves by the dense
  weights of every layer that changes stage and by the Mamba state of every
  LINEAR layer that does (reference slots x MiB per slot); KV and expert rows
  come from the plan as before; the line ``PP-CUT RECUT ref=.. -> .. P-KARTE``
  names the ceiling per stage.
* D->P (``wake_credit.recut_reference``): P tags per band are redistributed onto
  the new stages (layers of the band on the stage x the stage's P demand per
  layer), the on-card staging follows the measured ratio per D rank.
* P->D (``wake_credit_pd.recut_reference_pd``): releases, pauses, lanes and
  deposits per band rebuilt from the per-layer measurements of each stage;
  ``free`` at flip start moves by P's holding per layer (dense + rows, Mamba,
  KV per attention layer) from the PP-cut solve's geometry.
* No geometry, another stage count or layer sum, an unknown reference split:
  W167 Weg2PCutRecutRefused, by name.

The measured cut stays byte-identical (no RECUT line, same numbers).

Hermetic: no GPU, no torch device.
"""

import json
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import msgspec
import pytest

from sglang.srt.planner import p_card_chunk as pc
from sglang.srt.weg2 import launcher as lc
from sglang.srt.weg2 import wake_credit as wc
from sglang.srt.weg2 import wake_credit_pd as wpd

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
ROW = 2.4172
# Qwen3.8-Flash-Next layer_types: every 4th layer full attention
KINDS = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 12
TODAY = (29, 11, 8)
CUT = (27, 11, 10)
GEO = {"layer_kinds": KINDS, "dense_mib_per_layer": 62.0,
       "mamba_mib_per_linear_layer": 32 * 1.5602, "kv_mib_per_attn_layer": 272.0}


def _ref():
    with open(os.path.join(FIX, "pcut_recut_242r", "p_card_reference_h91_0928.json"), "rb") as fh:
        return msgspec.json.decode(fh.read(), type=pc.PCardReference)


def _co():
    with open(os.path.join(FIX, "pcut_recut_242r", "co_tenant_span_h91_0928.json"), "rb") as fh:
        return msgspec.json.decode(fh.read(), type=pc.CoTenantSpan)


def _solve(ref, cut):
    attn, lin = pc.stage_kind_counts(KINDS, cut)
    return pc.solve_p_card(
        reference=ref, support=lc.p_transient_support("nextflash", card=True),
        fractions=[0.324, 0.637, 0.733887], lru_rows=[32] * 3, stage_layers=list(cut), chunk=16384,
        kv_mib=[272.0 * a for a in attn], num_experts=512, row_mib=ROW, draft_on_p=False,
        draft_mib_last_stage=0.0, cards=["nvml1", "nvml0", "nvml2"], near_oom_mib=400.0,
        prompt_tokens=262144, lmem_fixed=pc.arena_write_lmem_fixed(), seats=6, co_tenant=_co(),
        mamba_slots=32, mamba_mib_per_slot=tuple(1.5602 * x for x in lin))


def _recut(ref, cut, **kw):
    args = dict(layer_kinds=KINDS, dense_mib_per_layer=62.0, mamba_mib_per_linear_layer_per_slot=1.5602)
    args.update(kw)
    return pc.recut_reference(ref, cut, **args)


class TestPCard:
    def test_27_11_10_ceiling_is_the_table(self):
        fits = _solve(_recut(_ref(), CUT), CUT)
        assert [(round(f.ceiling_fraction, 3), f.ceiling_max_rows) for f in fits] == [
            (0.339, 206), (0.646, 363), (0.773, 428)]

    def test_recut_moves_k0_by_dense_and_mamba_only(self):
        ref = _ref()
        new = _recut(ref, CUT)
        # PP0 gives 2 layers (1 attention, 1 linear), PP2 takes them
        assert new.headroom0_mib[0] == pytest.approx(ref.headroom0_mib[0] + 2 * 62.0 + 32 * 1.5602, abs=0.1)
        assert new.headroom0_mib[1] == ref.headroom0_mib[1]
        assert new.headroom0_mib[2] == pytest.approx(ref.headroom0_mib[2] - 2 * 62.0 - 32 * 1.5602, abs=0.1)
        assert new.stage_layers == CUT and "RECUT 29,11,8->27,11,10" in new.source

    def test_the_measured_cut_recuts_to_itself(self):
        ref = _ref()
        same = _recut(ref, TODAY)
        assert same.headroom0_mib == tuple(round(h, 1) for h in ref.headroom0_mib)
        assert [f.ceiling_max_rows for f in _solve(same, TODAY)] == [185, 363, 558]

    @pytest.mark.parametrize("cut, kw, why", [
        ((27, 11, 10), {"layer_kinds": None}, "Layer-Typen des Checkpoints unbekannt"),
        ((27, 21), {}, "Stufenzahl 2 gegen 3"),
        ((27, 11, 11), {}, "Layersumme 49 gegen 48"),
        ((27, 11, 10), {"dense_mib_per_layer": None}, "Dense je Layer unbekannt"),
    ])
    def test_an_unknown_geometry_is_refused_by_name(self, cut, kw, why):
        with pytest.raises(pc.PCutRecutRefused, match="W167 Weg2PCutRecutRefused") as exc:
            _recut(_ref(), cut, **kw)
        assert why in str(exc.value)


class TestLauncherPCard:
    def _run(self, cut, **kw):
        lines = []
        attn, lin = pc.stage_kind_counts(KINDS, cut)
        ns = types.SimpleNamespace(p_card_reference_logs="", draft_kv_on_p="off", p_card_prompt_tokens=0,
                                   extra_p="--max-running-requests 1", extra_d="", p_bs=1)
        lc.p_card_verdict(
            ns, [types.SimpleNamespace(nvml_index=i) for i in (1, 0, 2)], lines.append,
            model="/m/" + MODEL, chunk_tokens=16384, fracs=[0.2, 0.5, 0.5], lru_rows=[32, 32, 32],
            stage_layers=list(cut), kv_mib=[272.0 * a for a in attn], num_experts=512, row_mib=ROW,
            mamba_slots=24, mamba_mib_per_slot=tuple(1.5602 * x for x in lin), **kw)
        return lines

    def test_another_cut_is_recut_and_named(self):
        lines = self._run(CUT, layer_kinds=KINDS, dense_mib_per_layer=62.0,
                          mamba_mib_per_linear_layer_per_slot=1.5602)
        assert not any("ENTFAELLT" in ln for ln in lines)
        recut = [ln for ln in lines if ln.startswith("PP-CUT RECUT ref=29,11,8 -> 27,11,10 P-KARTE: Decke")]
        assert len(recut) == 1 and "stage2 f" in recut[0]

    def test_another_cut_without_geometry_refuses_w167(self):
        with pytest.raises(lc.Weg2LaunchRefused, match="W167 Weg2PCutRecutRefused"):
            self._run(CUT)

    def test_the_measured_cut_is_unchanged(self):
        plain = self._run(TODAY)
        with_geo = self._run(TODAY, layer_kinds=KINDS, dense_mib_per_layer=62.0,
                             mamba_mib_per_linear_layer_per_slot=1.5602)
        assert plain == with_geo and not any("RECUT" in ln for ln in plain)


def _wref():
    base = os.path.join(FIX, "wake_credit_h14", "fnFL2x162.")
    texts = []
    for x in ("P", "D", "front"):
        with open(base + x + ".lines") as fh:
            texts.append(fh.read())
    return wc.reference_from_logs(*texts, source="fnFL2x162", p_card=[1, 0, 2])


def _dp(cut, rows, ref=None, key_split="measured"):
    ref = ref or _wref()
    split = wc.reference_p_split(ref, 48) if key_split == "measured" else key_split
    return wc.plan_wake_credit(
        model=MODEL, p_split=cut, chunk_layers=3, n_layers=48, p_card=[1, 0, 2], d_ratio="183,137,168",
        p_rows=rows, d_rows=[103, 122, 133], slot_mib=ROW, label="D", reorder=True, double_staging=False,
        reference=ref, reference_key={"model": MODEL, "p_card": (1, 0, 2), "p_split": split,
                                      "chunk_layers": 3})


def _air(lines):
    import re
    out = {}
    for ln in lines:
        m = re.search(r" (card\d) D TP\d/P PP\d: .*engste Luft (-?\d+) MiB", ln)
        if m and "P->D" not in ln:
            out[m.group(1)] = int(m.group(2))
    return out


class TestWakeCreditDtoP:
    def test_27_11_10_credit_is_the_table(self):
        plan = _dp(CUT, [206, 363, 428])
        assert plan.refusal is None
        assert _air(plan.lines) == {"card1": 5320, "card0": 224, "card2": 586}
        assert any(ln.startswith("PP-CUT RECUT ref=29,11,8 -> 27,11,10 WAKE-CREDIT D->P D:") and
                   ln.endswith("-> PASST") for ln in plan.lines)
        # the front gets the tags of the NEW cut: weights_9 lies on PP1 alone
        dem = {c["card"]: c["demand"] for c in plan.front_plan["D->P"]}
        assert "weights_9" in dem[0] and "weights_9" not in dem[1]

    def test_the_measured_cut_is_the_live_launcher_line(self):
        plan = _dp(TODAY, [198, 359, 408])
        assert _air(plan.lines) == {"card1": 4247, "card0": 133, "card2": 876}
        assert not any("RECUT" in ln for ln in plan.lines)

    def test_a_reference_without_its_split_refuses_another_cut(self):
        plan = _dp(CUT, [206, 363, 428], key_split="ungemessen: ohne Layerzeilen")
        assert plan.refusal is not None and plan.refusal.startswith("W167 Weg2PCutRecutRefused")
        assert not any("ENTFAELLT" in ln for ln in plan.lines)


def _pd(cut, rows, geometry):
    return wpd.plan_wake_credit_pd(
        model=MODEL, p_split=cut, chunk_layers=3, n_layers=48, p_card=[1, 0, 2], d_ratio="183,137,168",
        draft_on_p=False, p_rows=rows, d_rows=[103, 122, 133], slot_mib=ROW, label="D", apply=False,
        p_resident=[min(510, r - 32) for r in rows], d_resident=[12, 74, 85], dense_repack=True,
        d_seats=6, free0_records=(), recut_geometry=geometry)


class TestWakeCreditPtoD:
    def test_27_11_10_is_timed_and_named(self):
        plan = _pd(CUT, [206, 363, 428], GEO)
        assert plan.refusal is None
        line = [ln for ln in plan.lines if ln.startswith("PP-CUT RECUT ref=29,11,8 -> 27,11,10 WAKE-CREDIT P->D")]
        assert len(line) == 1 and "Leg 1661 ms (vollstaendig)" in line[0] and line[0].endswith("-> PASST")
        assert "card2 D TP2 free 5817 Kreditwarten 546 ms" in line[0]

    def test_without_geometry_another_cut_refuses_w167(self):
        plan = _pd(CUT, [206, 363, 428], None)
        assert plan.refusal is not None and plan.refusal.startswith("W167 Weg2PCutRecutRefused")
        assert "layer_kinds" in plan.refusal

    def test_the_measured_cut_ignores_the_geometry(self):
        a = _pd(TODAY, [198, 359, 408], GEO)
        b = _pd(TODAY, [198, 359, 408], None)
        assert a.lines == b.lines and a.front_plan == b.front_plan
        assert not any("RECUT" in ln for ln in a.lines)
