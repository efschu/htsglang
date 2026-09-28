"""YaRN x2 (28.09.): FR_P ist die OBERGRENZE, der Launcher kappt.

Der YaRN-x2-Trockenlauf (Baum 40abc4d9ad, Profil -st-yarn2, Prompt 524288,
Chunk 16384) verweigerte W132: Stufe 0 (5090) bei f 0.324 -> 198 Zeilen,
Kopfraum am letzten Chunk -1259 MiB < near-OOM 400, also 1659 MiB zu wenig;
die groesste tragbare Fraction je Stufe nannte W132 selbst (0.277,0.642,0.996).
Entscheid Koordinator: kein Hand-Pin -- ``NF_FR_P`` ist die Obergrenze, der
Launcher kappt NUR die reissende Stufe auf ihre Karten-Decke und schreibt sie
an die drei Stellen, an denen FR_P lebt (wie der H25-Post). Verweigert bleibt
W132, wenn eine Stufe unter ihrer Mindestresidenz laege (nicht einmal ein
residenter Experte plus Scratch) oder die Karte auch gekappt reisst. Formen,
die heute nicht reissen, bleiben byte-gleich.
"""

from __future__ import annotations

import inspect
import re
import types

import pytest

from sglang.srt.planner import p_card_chunk as pc
from sglang.srt.weg2 import launcher as lc

MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
ROW_MIB = 2.4170379638671875
CUT = (29, 11, 8)
LRU = (32, 32, 32)
KV_H25 = (1904.0, 816.0, 544.0)
H160 = (0.332, 0.605, 0.733887)
YARN = (0.324, 0.637, 0.733887)


def _fit(stage, card, fraction, rows, headroom, ceiling, max_rows):
    return pc.PCardFit(
        stage=stage, card=card, fraction=fraction, buffer_rows=rows, layer_row_mib=ROW_MIB,
        row_card_mib=70.1, expert_mib=0.0, kv_mib=0.0, transient_mib=0.0,
        transient_ref_mib=0.0, transient_source="test", draft_mib=0.0,
        prompt_tokens=524288, last_chunk_index=31, growth_mib=1005.0,
        growth_extrapolated=True, lmem_mib=0.0, lmem_source="test", headroom_mib=headroom,
        near_oom_mib=400.0, ceiling_fraction=ceiling, ceiling_max_rows=max_rows)


def _yarn_fits():
    # die Zahlen des Trockenlaufs dry_z25_yarn2.log (PP-CUT P-KARTE stage0..2)
    return (
        _fit(0, "nvml1", 0.324, 198, -1259.0, 0.277, 174),
        _fit(1, "nvml0", 0.637, 359, 458.0, 0.642, 361),
        _fit(2, "nvml2", 0.733887, 408, 3039.0, 0.996, 628),
    )


def test_only_the_tearing_stage_is_capped_to_its_ceiling():
    cap = lc.p_card_fr_cap(_yarn_fits(), YARN, 512, LRU)
    assert cap.refusal == ""
    # (a) stage 0 auf die Decke, stage 1/2 EXAKT wie gegeben (auch die
    # hoehere Decke von stage 1 wird nicht genommen)
    assert cap.fractions == (0.277, 0.637, 0.733887)
    assert cap.stages == (0,)
    (line,) = cap.lines
    assert "stage0 (nvml1): f 0.3240 -> 0.277 (198 -> 174 Zeilen" in line
    assert "Kopfraum am letzten Chunk -1259 MiB < near-OOM 400, es fehlen 1659 MiB" in line


def test_no_ceiling_keeps_the_refusal_by_name():
    # (b) die bestehende Mindestresidenz: largest_fraction_for_rows = None,
    # wenn nicht einmal ein residenter Experte plus Scratch passt
    fits = (_fit(0, "nvml1", 0.324, 198, -25000.0, None, 20),) + _yarn_fits()[1:]
    cap = lc.p_card_fr_cap(fits, YARN, 512, LRU)
    assert cap.fractions == YARN and cap.stages == ()
    assert "stage0 (nvml1): keine Kappung" in cap.refusal
    assert "Mindestresidenz, largest_fraction_for_rows" in cap.refusal
    assert "es fehlen 25400 MiB" in cap.refusal


def test_a_ceiling_not_below_the_fraction_keeps_the_refusal():
    fits = (_fit(0, "nvml1", 0.324, 198, 100.0, 0.330, 201),) + _yarn_fits()[1:]
    cap = lc.p_card_fr_cap(fits, YARN, 512, LRU)
    assert cap.fractions == YARN
    assert "liegt nicht unter der gegebenen f 0.3240" in cap.refusal


def test_publish_writes_the_three_places():
    ns = types.SimpleNamespace(
        pp_cut_expert_device_fraction="0.324,0.637,0.39",
        extra_p="--max-total-tokens 524288 --rank-moe-resident-fraction 0.324,0.637,0.733887",
        env_p="SGLANG_MOE_SCRATCH_SLOTS=32;SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.324,0.637,0.39;X=1")
    lc.publish_p_fractions(ns, (0.277, 0.637, 0.733887))
    assert ns.pp_cut_expert_device_fraction == "0.277,0.637,0.733887"
    assert "--rank-moe-resident-fraction 0.277,0.637,0.733887" in ns.extra_p
    assert ns.env_p == ("SGLANG_MOE_SCRATCH_SLOTS=32;"
                        "SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.277,0.637,0.733887;X=1")


def _ns(**kw):
    ns = types.SimpleNamespace(
        p_card_reference_logs="", draft_kv_on_p="off", p_card_prompt_tokens=0,
        extra_p="--speculative-draft-model-path /nonexistent/draft "
                "--rank-moe-resident-fraction 0.412,0.605,0.733887",
        extra_d="", env_p="SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.412,0.605,0.733887")
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _verdict(ns, fr):
    lines = []
    out = lc.p_card_verdict(
        ns, [types.SimpleNamespace(nvml_index=i) for i in (1, 0, 2)], lines.append,
        model="/m/" + MODEL, chunk_tokens=16384, fracs=fr, lru_rows=LRU, stage_layers=CUT,
        kv_mib=KV_H25, num_experts=512, row_mib=ROW_MIB)
    return out, lines


def test_a_form_that_fits_stays_byte_identical():
    # (c) -st bei 262k reisst nicht: nichts wird geschrieben, keine Zeile
    ns = _ns()
    before = dict(vars(ns))
    out, lines = _verdict(ns, H160)
    assert out is None
    assert vars(ns) == before
    assert not any("KAPPUNG" in ln for ln in lines)


def test_the_launcher_caps_publishes_and_records_it():
    ns = _ns()
    out, lines = _verdict(ns, (0.412, 0.605, 0.733887))
    assert out == [0.402, 0.605, 0.733887]
    # (d) die Kappung steht im argv/env (D und Planer lesen dieselbe Zahl)
    assert ns.pp_cut_expert_device_fraction == "0.402,0.605,0.733887"
    assert "--rank-moe-resident-fraction 0.402,0.605,0.733887" in ns.extra_p
    assert ns.env_p == "SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.402,0.605,0.733887"
    # ... im Record (BootState.p_fr_cap liest ns._w132_fr_cap) ...
    rec = ns._w132_fr_cap
    assert rec["fr_p_before"] == [0.412, 0.605, 0.733887]
    assert rec["fr_p_after"] == [0.402, 0.605, 0.733887]
    assert rec["stages"] == [0] and rec["chunk"] == 16384 and rec["prompt"] == 262144
    # ... und im Dry-Run: die Karte wird mit der Kappung neu gerechnet und passt
    card = [ln for ln in lines if ln.startswith("PP-CUT P-KARTE stage")]
    assert len(card) == 6 and "STIRBT" in card[0] and all("PASST" in c for c in card[3:])
    assert any(ln.startswith("PP-CUT W132-FR_P-KAPPUNG stage0 (nvml1): f 0.4120 -> 0.402")
               for ln in lines)


def test_the_cap_line_is_not_counted_as_a_refusal_by_the_arm():
    # der Arm zaehlt nach dem Dry-Run 'W[0-9]+ ' als W-Zeile/Verweigerung
    ns = _ns()
    _, lines = _verdict(ns, (0.412, 0.605, 0.733887))
    cap = [ln for ln in lines if "KAPPUNG" in ln]
    assert cap and not any(re.search(r"W[0-9]+ ", ln) for ln in cap)


def test_the_pool_model_and_the_record_read_the_capped_value():
    src = inspect.getsource(lc.solve_p_cut)
    i = src.index("_capped = p_card_verdict(")
    j = src.index("if _capped is not None:", i)
    k = src.index("fracs = list(_capped)", j)
    assert src.index("layer_mib_by_stage = tuple(", k) < src.index("model_pool = ", k)
    main = inspect.getsource(lc.main)
    a = main.index("cut = solve_p_cut(")
    assert main.index('state.p_fr_cap = dict(getattr(ns, "_w132_fr_cap", None) or {})', a) > a
    assert "p_fr_cap" in {f for f in lc.BootState.__dataclass_fields__}
