# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H50 -- der D-Planer traegt den H39-Zustand, gemessen, nicht gepinnt.

DER BEFUND, schwarz-weiss: H39 (d6b7d4a1d3, Dense-Marlin-Repack ausserhalb der
Tag-Pools) hat auf D-TP0 (5090) am Metall 4,3 GiB freigemacht -- fnFL2x150 (vor
H39) 'weights + runtime state' 21.996 GiB, private_free 4993 MiB; fnFL2x151 /
fnFL2x158 (H39) 17.744 GiB, private_free 625 MiB. Der Dry-Run von x155
(SCRATCH_D 130) rechnete TP0 trotzdem mit den Referenzen VOR H39 (fester
Rang-Posten x98-x100 13732 MiB, Karten-Kopfraum x141/x144 mit privat_frei
4993) und verweigerte mit W122 "3591 MiB zuviel" und W130 "Kopfraum -4961".

Der Fix: je Baum-Zustand eine GEMESSENE Referenz (x151 + x158 im H39-Zustand),
gewaehlt nach SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL der D-Gruppe; der Zustand
eines Logs ist die WIRKUNG ("checkpoint-format pool RELEASED"), nicht der
Schalter. Dasselbe fuer den dritten Riegel mit demselben Posten (W126 P->D,
Referenz fnFL2x158).

Die physische Kante ist dabei 133 Zeilen (SCRATCH_D 121 bei f 0.06), nicht
142: eine Zeile ist 48 Layer x 2.417 MiB = 116 MiB (nicht ~75), und im
x158-Decode sind 5399 MiB frei -- 48 Zeilen mehr (5569 MiB) passen nicht auf
die Karte, auch ohne jede Grenze. SCRATCH_D 130 bleibt deshalb verweigert
(W130), jetzt aus dem richtigen Posten.

Alle Zahlen sind Logzeilen: ``fixtures/d_h39_h50/`` sind woertliche Zeilen aus
/spinning/evidence-665-f1/boot_weg2_fnFL2x{150,151,158}_*.D.log.
"""

import json
import os
import types

import msgspec
import pytest

from sglang.srt.planner import expert_residency as er
from sglang.srt.weg2 import wake_credit_pd as pd
from sglang.srt.weg2 import wake_credit_pd_refs as refs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "d_h39_h50")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
SLOT_BYTES = 1297637376 / 512
N_LAYERS = 48
LAYER_MIB = N_LAYERS * SLOT_BYTES / (1 << 20)
RATIOS = (183, 137, 168)
DRY_BUDGETS = (29368, 18184, 17784)  # dry_fnFL2x155.log "budget D(dry, expectation)"
EDGE_K = (0.06, 0.51, 0.48)
VOCAB_MIB = er.draft_vocab_mib(vocab_size=248320, hidden_size=2560)
NEAR_OOM = 400.0
BAND_FLOOR = 819.0
KW = dict(n_ranks=3, n_layers=N_LAYERS, slot_bytes=SLOT_BYTES, model=MODEL,
          rank_tp_ratio="1,0,0")


def _text(name):
    with open(os.path.join(FIX, name + ".D.lines")) as fh:
        return fh.read()


def _boots(*names):
    return [(n, _text(n)) for n in names]


def _solve(scratch0, rank_ref, card_ref, *, fr=EDGE_K):
    fits = er.solve_d_rank_residency(
        budgets_mib=DRY_BUDGETS, fractions=fr, ratios=RATIOS,
        scratch_rows=(scratch0, 48, 48), staging_rows=12, num_experts=512, pad_rows=1,
        n_layers=N_LAYERS, slot_bytes=SLOT_BYTES, reference=rank_ref,
        vocab_mib=VOCAB_MIB, share_embed=True, kv_tokens=262144)
    cards = er.solve_d_card(fits=fits, reference=card_ref, vocab_mib=VOCAB_MIB,
                            share_embed=True, near_oom_mib=NEAR_OOM,
                            band_floor_mib=BAND_FLOOR)
    return fits, cards


OLD = (er.D_RESIDENCY_REFERENCE_FNFL2, er.D_CARD_REFERENCE_FNFL2)
NEW = (er.D_RESIDENCY_REFERENCE_FNFL2_H39, er.D_CARD_REFERENCE_FNFL2_H39)


# ---------------------------------------------------------------------------
# 1. die Messung: welcher Posten den H39-Gewinn nicht kannte
# ---------------------------------------------------------------------------


def test_the_h39_marker_is_the_effect_in_the_log():
    assert not er.boot_dense_repack_outside_pool(_text("fnFL2x150"))
    assert er.boot_dense_repack_outside_pool(_text("fnFL2x151"))
    assert er.boot_dense_repack_outside_pool(_text("fnFL2x158"))
    with pytest.raises(ValueError, match="mischt Baeume vor und nach H39"):
        er.d_card_reference_from_logs(_boots("fnFL2x150", "fnFL2x158"), **KW)
    with pytest.raises(ValueError, match="mischt Baeume vor und nach H39"):
        er.d_rank_reference_from_logs(_boots("fnFL2x150", "fnFL2x158"), **KW)


def test_the_shipped_h39_references_are_the_logs_own_measurement():
    assert er.d_rank_reference_from_logs(
        _boots("fnFL2x151", "fnFL2x158"), **KW) == er.D_RESIDENCY_REFERENCE_FNFL2_H39
    assert er.d_card_reference_from_logs(
        _boots("fnFL2x151", "fnFL2x158"), **KW) == er.D_CARD_REFERENCE_FNFL2_H39
    assert er.D_RESIDENCY_REFERENCE_FNFL2_H39.dense_repack_outside_pool
    assert not er.D_RESIDENCY_REFERENCE_FNFL2.dense_repack_outside_pool
    assert not er.D_CARD_REFERENCE_FNFL2.dense_repack_outside_pool


def _weight_tags_tp0(text):
    """WEG2-DC-BREAKDOWN (kv_cache/cuda_graph abgegeben) TP0: Gewichts-Tags ohne Draft."""
    import ast
    import re

    from sglang.srt.name_compat import has_marker

    for ln in text.splitlines():
        # fixture logs are evidence from before the rename: either spelling (name_compat 1a)
        if has_marker(ln, "TP0] WEG2-DC-BREAKDOWN stage=release tags=['kv_cache', 'cuda_graph']"):
            d = ast.literal_eval(re.search(r"tms_resident \d+ (\{[^}]*\})", ln).group(1))
            return sum(v for k, v in d.items() if k.startswith("weights") and k != "weights_draft")
    raise AssertionError("keine DC-BREAKDOWN-Zeile")


def test_the_posten_is_the_same_4354_mib_in_two_independent_instruments():
    """Indikator in beide Richtungen: der feste Rang-Posten (KV budget posts)
    und die Gewichts-Tags (WEG2-DC-BREAKDOWN) sinken auf TP0 um DIESELBEN 4354
    MiB; TP1/TP2 (keine Dense-Marlin-Linears) bleiben gleich. Der Karten-Posten
    privat_frei sinkt um 4368 MiB (4993 -> 625)."""
    old = er.d_rank_reference_from_logs(_boots("fnFL2x150"), **KW)
    new = er.D_RESIDENCY_REFERENCE_FNFL2_H39
    assert old.fixed_mib[0] - new.fixed_mib[0] == pytest.approx(4354.0, abs=1.0)
    assert old.fixed_mib[1:] == new.fixed_mib[1:]
    tags = _weight_tags_tp0(_text("fnFL2x150")) - _weight_tags_tp0(_text("fnFL2x158"))
    assert tags == pytest.approx(old.fixed_mib[0] - new.fixed_mib[0], abs=2.0)
    oc = er.d_card_reference_from_logs(_boots("fnFL2x150"), **KW)
    nc = er.D_CARD_REFERENCE_FNFL2_H39
    assert (oc.private_free_mib[0], nc.private_free_mib[0]) == (4993.0, 625.0)
    assert nc.headroom0_mib[0] - oc.headroom0_mib[0] == pytest.approx(4326.0, abs=1.0)
    # die eingebauten Vor-H39-Referenzen tragen denselben alten Posten
    assert er.D_CARD_REFERENCE_FNFL2.private_free_mib[0] == pytest.approx(4993.1)
    assert (er.D_RESIDENCY_REFERENCE_FNFL2.fixed_mib[0] - VOCAB_MIB
            == pytest.approx(old.fixed_mib[0], abs=320.0))


# ---------------------------------------------------------------------------
# 2. die Fixtures: x150 verweigert, x158 passt an der Kante
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scratch0", [130, 121])
def test_x150_state_still_refuses(scratch0):
    for rank_ref, card_ref in (
        OLD,
        (er.d_rank_reference_from_logs(_boots("fnFL2x150"), **KW),
         er.d_card_reference_from_logs(_boots("fnFL2x150"), **KW)),
    ):
        fits, cards = _solve(scratch0, rank_ref, card_ref)
        assert fits[0].refused  # W122
        assert cards[0].refused  # W130
        assert cards[0].ceiling_max_rows == 95  # die alte Kante: SCRATCH <= 83


def test_x158_state_passes_the_edge_and_names_it():
    fits, cards = _solve(121, *NEW)
    assert fits[0].buffer_rows == 133
    assert not any(f.refused for f in fits) and not any(c.refused for c in cards)
    assert fits[0].ceiling_max_rows == 145  # Budget-Kante
    assert cards[0].ceiling_max_rows == 133  # Karten-Kante, bindend
    assert er.scratch_edge(local_experts=193, fraction=0.06, max_rows=133) == (133, 121)
    assert cards[0].headroom_mib == pytest.approx(4944.0 - 39 * LAYER_MIB, abs=1.0)
    # Decode-frei ist Befund, kein Stopper: x151 hielt 3485 MiB allgemeinen
    # Cache (card_free 2429 bei 94 Zeilen) -- bei 133 Zeilen muss der Allokator
    # ihn abgeben (alloc retry); x158 (488 MiB Cache) laege bei 874 MiB frei.
    assert cards[0].verdict.startswith("PASST")
    assert 5399.0 - 39 * LAYER_MIB == pytest.approx(874.0, abs=1.0)
    line = er.describe_card(cards[0], NEW[1])
    assert "KARTEN-KANTE bei f 0.060: <= 133 Zeilen = SCRATCH <= 121" in line


@pytest.mark.parametrize("scratch0", [122, 130])
def test_x158_state_refuses_beyond_the_card(scratch0):
    """130 ist nicht tragbar: 48 Zeilen x 116 MiB = 5569 MiB > 5399 MiB, die im
    x158-Decode auf der Karte frei sind. Das Budget (W122) laesst es jetzt
    durch -- verweigert wird aus dem Karten-Posten, nicht aus dem alten."""
    fits, cards = _solve(scratch0, *NEW)
    assert not fits[0].refused
    assert cards[0].refused and cards[0].verdict == "STIRBT AN DER KARTE"
    if scratch0 == 130:
        assert 48 * LAYER_MIB > 5399.0


def test_mutant_without_the_h39_posten_refuses_x158_and_with_it_passes_x150():
    """Der Posten ist tragend, in beide Richtungen."""
    rank_new, card_new = NEW
    gone_card = msgspec.structs.replace(
        card_new, headroom0_mib=(card_new.headroom0_mib[0] - 4368.0,) + card_new.headroom0_mib[1:])
    gone_rank = msgspec.structs.replace(
        rank_new, fixed_mib=(rank_new.fixed_mib[0] + 4354.0,) + rank_new.fixed_mib[1:])
    fits, cards = _solve(121, gone_rank, gone_card)
    assert fits[0].refused and cards[0].refused
    rank_old, card_old = OLD
    given_card = msgspec.structs.replace(
        card_old, headroom0_mib=(card_old.headroom0_mib[0] + 4368.0,) + card_old.headroom0_mib[1:])
    given_rank = msgspec.structs.replace(
        rank_old, fixed_mib=(rank_old.fixed_mib[0] - 4354.0,) + rank_old.fixed_mib[1:])
    fits, cards = _solve(121, given_rank, given_card)
    assert not fits[0].refused and not cards[0].refused


# ---------------------------------------------------------------------------
# 3. die Wahl: Schalter der D-Gruppe, explizite Logs, P->D-Riegel
# ---------------------------------------------------------------------------


def test_the_env_mirror_matches_environ(monkeypatch):
    from sglang.srt.environ import envs

    # UNIFY S2: environ's default is resolved per profile (weg2/form.py); with
    # no published form it is the NF default this mirror names.
    monkeypatch.delenv("SGLANG_WEG2_FORM", raising=False)
    assert er.DENSE_REPACK_OUTSIDE_POOL_DEFAULT is envs.SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL._resolve_default()
    assert er.dense_repack_outside_pool({}) is True
    assert er.dense_repack_outside_pool({er.DENSE_REPACK_OUTSIDE_POOL_ENV: "0"}) is False
    with pytest.raises(ValueError):
        er.dense_repack_outside_pool({er.DENSE_REPACK_OUTSIDE_POOL_ENV: "vielleicht"})


def test_the_state_picks_the_builtin_and_foreign_logs_are_named(tmp_path):
    for state, want in ((True, NEW), (False, OLD)):
        r, _ = er._reference_for(model_path="/m/" + MODEL, rank_tp_ratio="1,0,0",
                                 reference_logs="", dense_repack=state, **{
                                     k: KW[k] for k in ("n_ranks", "n_layers", "slot_bytes")})
        c, _ = er._card_reference_for(model_path="/m/" + MODEL, rank_tp_ratio="1,0,0",
                                      card_reference_logs="", dense_repack=state, **{
                                          k: KW[k] for k in ("n_ranks", "n_layers", "slot_bytes")})
        assert (r, c) == want
    log = os.path.join(FIX, "fnFL2x150.D.lines")
    c, why = er._card_reference_for(model_path="/m/" + MODEL, rank_tp_ratio="1,0,0",
                                    card_reference_logs=log, dense_repack=True,
                                    n_ranks=3, n_layers=N_LAYERS, slot_bytes=SLOT_BYTES)
    assert c is None and "H39 aus" in why and "H39 an" in why


def _p_to_d(scratch0, dense_repack, draft_on_p=False):
    E_D = (193, 145, 177)
    S = (scratch0, 48, 48)
    return pd.plan_wake_credit_pd(
        model=MODEL, p_split=(29, 11, 8), chunk_layers=3, n_layers=N_LAYERS, p_card=(1, 0, 2),
        d_ratio="183,137,168", draft_on_p=draft_on_p, p_rows=(166, 263, 408),
        d_rows=[er.buffer_rows(local_experts=e, fraction=f, scratch_rows=s)
                for e, f, s in zip(E_D, EDGE_K, S)],
        slot_mib=LAYER_MIB / N_LAYERS, label="D(dry)", apply=False,
        p_resident=[er.resident_rows(512, f) for f in (0.26, 0.45, 0.733887)],
        d_resident=[er.resident_rows(e, f) for e, f in zip(E_D, EDGE_K)],
        dense_repack=dense_repack)


def test_the_p_to_d_riegel_carries_the_same_posten():
    """W126 P->D: mit der Vor-H39-Referenz x144 (D TP0 bucht 20720 MiB Tags bei
    94 Zeilen) steht der Wake bei SCRATCH 121 -- mit x158 (16366 MiB) nicht."""
    old, new = _p_to_d(121, False), _p_to_d(121, True)
    assert old.refusal and old.refusal.startswith("W126") and "D TP0" in old.refusal
    assert new.refusal is None
    assert any("fnFL2x158/1" in ln for ln in new.lines[:1])
    assert _p_to_d(82, True).refusal is None
    ent = _p_to_d(121, True, draft_on_p=True)
    assert ent.refusal is None and "ENTFAELLT" in ent.lines[0] and "dense_repack" in ent.lines[0]
    assert refs.FORM_KEYS["fnFL2x158"]["dense_repack"] is True


# ---------------------------------------------------------------------------
# 4. die Launcher-Naht: der Dry-Run der x155-Form
# ---------------------------------------------------------------------------


def _launcher_ns(tmp_path, scratch0, h39=None):
    cfg = {"text_config": {"num_hidden_layers": 48, "vocab_size": 248320, "hidden_size": 2560}}
    (tmp_path / MODEL).mkdir(parents=True)
    (tmp_path / MODEL / "config.json").write_text(json.dumps(cfg))
    env = ("SGLANG_MOE_POOL_STAGING=12;SGLANG_MOE_SCRATCH_SLOTS=%d,48,48;"
           "SGLANG_UNEVEN_MOE_EXPERT_SHARD=1;SGLANG_WEG2_DRAFT_SHARE_EMBED=1" % scratch0)
    if h39 is not None:
        env += ";SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL=%d" % int(h39)
    return types.SimpleNamespace(
        model=str(tmp_path / MODEL),
        extra_d="--rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
                "--rank-moe-resident-fraction 0.06,0.51,0.48",
        env_d=env, extra_p="", env_p="SGLANG_MOE_SCRATCH_SLOTS=32",
        pp_cut_expert_device_fraction="0.26,0.45,0.733887", pp_cut_expert_lru_rows="32",
        d_foreign_context_mib="1446,896,894", d_nontorch_mib="1981,528,524",
        d_reserve_mib="", d_residency_reference_logs="", d_card_reference_logs="",
        wake_credit_reference_logs="")


def test_the_dry_run_of_the_x155_form(tmp_path, monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import launcher

    monkeypatch.setattr(
        pp_cut, "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(expert_layer_weight_bytes=1297637376.0,
                                         num_experts=512, n_layers=48))
    cards = [launcher.Card(1, "u1", "RTX 5090", 32607),
             launcher.Card(0, "u0", "RTX 3080", 20480),
             launcher.Card(2, "u2", "RTX 3080", 20480)]

    def run(name, scratch0, h39=None):
        lines = []
        with envs.SGLANG_WEG2_DRAFT_ON_P.override(False):
            launcher.log_d_rank_vram_solve(
                _launcher_ns(tmp_path / name, scratch0, h39), cards, DRY_BUDGETS,
                lines.append, "D(dry, expectation)", p_split=[29, 11, 8], chunk_layers=3)
        return lines

    lines = run("edge", 121)  # PASST: kein W122/W130/W126
    head = [ln for ln in lines if "KARTE D(dry, expectation) (H33" in ln]
    assert head and "Zeilen [133, 137, 134] = SCRATCH <= [121, 63, 49]" in head[0], lines
    assert "fnFL2x151 + fnFL2x158" in head[0] and "H39 an" in head[0]
    assert any("P->D D(dry, expectation) card1" in ln and "FERTIG" in ln for ln in lines)
    with pytest.raises(launcher.Weg2LaunchRefused, match=r"^W130 Weg2DCardNearOom"):
        run("x155", 130)
    with pytest.raises(launcher.Weg2LaunchRefused, match=r"^W122 .*W130 "):
        run("old", 121, h39=False)
