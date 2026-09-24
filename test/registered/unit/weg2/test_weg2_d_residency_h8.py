# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H8 -- der D-FRACTION-SOLVE kennt die Metallregeln.

DER BUG, schwarz-weiss: der Launcher nannte fuer D-TP1 (Ratio 137, Scratch 48,
Budget 18512 MiB) eine Residenz-Decke 0.807 und fuer D-TP2 0.661. Gefahren
wurden 0.72/0.75 auf TP1 -- beide Boots starben am KV-Pool (fnFL2x98/x99,
"The per-rank budget leaves no GPU memory for the KV cache ... 288 MiB more
than the budget") -- und 0.52 auf TP2 lief, servierte aber 117376 statt der
Pflicht-262144 Token (fnFL2x100). Die Decke kannte (a) die Pufferregel
``min(ceil(f*E) + Scratch, E)`` mit E = Spanne + Pad nicht, (b) rechnete gegen
die Karte statt gegen das Budget, ohne KV und ohne Aktivierung.

Alle Zahlen hier sind Logzeilen: die Fixtures unter
``fixtures/d_residency_h8/`` sind die woertlichen Zeilen aus
/spinning/evidence-665-f1/boot_weg2_fnFL2x{98,99,100}_*.D.log.
"""

import json
import os
import re
import types

import msgspec
import pytest

from sglang.srt.planner import expert_residency as er
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "d_residency_h8")
BOOTS = ("fnFL2x98", "fnFL2x99", "fnFL2x100")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
# Checkpoint-Header des Next-Flash-INT4 (pp_cut.checkpoint_weight_terms):
# expert_layer_weight_bytes 1297637376 auf 512 Experten, 48 Layer.
SLOT_BYTES = 1297637376 / 512
N_LAYERS = 48
RATIOS = (183, 137, 168)
SCRATCH = (70, 48, 48)
LIVE_BUDGETS = (29312, 18512, 18504)  # fnFL2x98..x100 "budget D group=D"
VOCAB_MIB = er.draft_vocab_mib(vocab_size=248320, hidden_size=2560)


def _boot_text(name):
    with open(os.path.join(FIX, name + ".D.lines")) as fh:
        return fh.read()


def _solve(fractions, *, budgets=LIVE_BUDGETS, share=False, reference=None):
    return er.solve_d_rank_residency(
        budgets_mib=budgets,
        fractions=fractions,
        ratios=RATIOS,
        scratch_rows=SCRATCH,
        staging_rows=8,
        num_experts=512,
        pad_rows=1,
        n_layers=N_LAYERS,
        slot_bytes=SLOT_BYTES,
        reference=reference or er.D_RESIDENCY_REFERENCE_FNFL2,
        vocab_mib=VOCAB_MIB,
        share_embed=share,
        kv_tokens=262144,
    )


_RX = re.compile(
    r"TP(\d)\] MoE expert-offload active on layer 47: (\d+)/(\d+) experts resident "
    r"\+ (\d+) scratch \(buffer=(\d+), fraction=([0-9.]+)\)"
)


def test_the_buffer_rule_reproduces_every_measured_rank():
    """Derived property: E = Largest-Remainder-Spanne + 1 Pad (193/145/177 --
    NICHT Ratio + Staging, das gaebe 176 auf TP2) und Puffer = min(R+S, E),
    an allen gemessenen Punkten (TP1 0.75/0.72/0.62, TP2 0.54/0.52, TP0 0.006)."""
    spans = er.expert_span_by_rank(num_experts=512, ratios=RATIOS)
    seen = set()
    for boot in BOOTS:
        for m in _RX.finditer(_boot_text(boot)):
            r, R, E, buf = (int(m.group(i)) for i in (1, 2, 3, 5))
            f = float(m.group(6))
            assert spans[r] + 1 == E, (boot, r)
            assert er.resident_rows(E, f) == R, (boot, r)
            assert (
                er.buffer_rows(local_experts=E, fraction=f, scratch_rows=SCRATCH[r])
                == buf
            )
            seen.add((r, f))
    assert {(1, 0.75), (1, 0.72), (1, 0.62), (2, 0.54), (2, 0.52), (0, 0.006)} <= seen


def test_the_mirrors_equal_their_runtime_sources(monkeypatch):
    """Bookkeeping: ``resident_rows``/``buffer_rows``/``expert_span_by_rank``
    sind Spiegel der Runtime (der Launcher darf kein torch importieren). Wer
    ``plan_load_time_staging`` oder ``partition_units`` aendert, ohne den
    Spiegel nachzuziehen, faellt hier."""
    from sglang.srt.distributed.utils import partition_units
    from sglang.srt.layers.moe.expert_offload import plan_load_time_staging

    for vec in ((183, 137, 168), (60, 226, 226), (1, 1, 1), (10, 7, 7), (5, 0.5, 3)):
        want = (
            partition_units(512, list(vec))
            if all(float(v).is_integer() for v in vec)
            else None
        )
        if want is not None:
            assert list(er.expert_span_by_rank(num_experts=512, ratios=vec)) == list(
                want
            )
    for S in (8, 36, 48, 70):
        monkeypatch.setenv("SGLANG_MOE_SCRATCH_SLOTS", str(S))
        for E in (145, 177, 193, 512):
            for f in (0.006, 0.3, 0.52, 0.62, 0.72, 0.75, 0.95, (E - 2) / E, 1.0):
                plan = plan_load_time_staging(E, fraction=f)
                rows = er.buffer_rows(local_experts=E, fraction=f, scratch_rows=S)
                assert rows == (E if plan is None else plan.buffer_slots), (S, E, f)


def test_the_shipped_reference_is_the_logs_own_measurement():
    """Bookkeeping: die eingebaute Referenz IST die Messung aus den drei Logs
    (fester Posten = 'weights + runtime state' minus Puffer desselben Boots,
    Maximum je Rang). Eine handgepflegte Zahl daneben faellt hier."""
    measured = er.d_rank_reference_from_logs(
        [(b, _boot_text(b)) for b in BOOTS],
        n_ranks=3,
        n_layers=N_LAYERS,
        slot_bytes=SLOT_BYTES,
        model=MODEL,
        rank_tp_ratio="1,0,0",
    )
    shipped = er.D_RESIDENCY_REFERENCE_FNFL2
    assert msgspec.structs.replace(measured, source=shipped.source) == shipped


def test_072_on_tp1_dies_at_the_kv_pool_and_062_does_not():
    """Bug regression x98/x99: TP1 bei 0.72 (Puffer 145) -- die Runtime sagte
    288 MiB zuviel VOR dem ersten KV-Token. 0.62 (Puffer 138, x100) lief."""
    bad = _solve((0.006, 0.72, 0.40))[1]
    assert bad.buffer_rows == 145
    assert bad.verdict == "STIRBT AM KV-POOL"
    assert abs(-bad.pre_kv_rest_mib - 288) <= 6
    good = _solve((0.006, 0.62, 0.40))[1]
    assert good.buffer_rows == 138
    assert good.verdict == "PASST"


def test_052_on_tp2_misses_the_262k_it_served_117376():
    """Bug regression x100: TP2 bei 0.52 ueberlebte den KV-Pool mit 0.084 GiB
    Rest und servierte 117376 Token -- die Rechnung muss denselben Rest nennen
    und die Fraction wegen der 262k-Pflicht verweigern."""
    fit = _solve((0.006, 0.62, 0.52))[2]
    assert abs(fit.pre_kv_rest_mib - 0.084 * 1024) <= 3
    assert fit.verdict == "262K VERFEHLT"
    text = er.refusal_text(_solve((0.006, 0.62, 0.52)), label="D")
    assert text is not None and text.startswith("W122 ") and "rang2" in text


@pytest.mark.parametrize("budgets", [LIVE_BUDGETS, (29424, 18184, 17784)])
@pytest.mark.parametrize("share", [False, True])
def test_the_ceiling_is_the_largest_fraction_that_fits(budgets, share):
    """Derived property: die Decke je Rang passt, und ein residenter Experte
    mehr passt nicht mehr (sonst ist sie nicht die GROESSTE)."""
    ceil = [
        f.ceiling_fraction
        for f in _solve((0.006, 0.5, 0.5), budgets=budgets, share=share)
    ]
    fits = _solve(tuple(ceil), budgets=budgets, share=share)
    assert all(not f.refused for f in fits), [er.describe_rank(f) for f in fits]
    for f in fits:
        E = f.local_experts
        one_more = (f.resident_rows + 1) / E
        if f.resident_rows + 1 > E - 2:
            continue
        nxt = _solve(
            tuple(one_more if g.rank == f.rank else c for g, c in zip(fits, ceil)),
            budgets=budgets,
            share=share,
        )[f.rank]
        assert nxt.refused, er.describe_rank(nxt)


def test_the_shared_draft_vocab_only_moves_the_draft_host():
    """Negative branch: SGLANG_WEG2_DRAFT_SHARE_EMBED zieht die eigenen
    BF16-Tabellen (2 x 248320 x 2560 x 2 B = 2425 MiB) NUR auf dem Draft-Host
    ab; die reinen Experten-Raenge bleiben byte-gleich."""
    assert VOCAB_MIB == pytest.approx(2425.0)
    assert er.draft_share_embed({}) is True
    assert er.draft_share_embed({"SGLANG_WEG2_DRAFT_SHARE_EMBED": "0"}) is False
    own = _solve((0.006, 0.62, 0.5), share=False)
    shared = _solve((0.006, 0.62, 0.5), share=True)
    assert shared[0].pre_kv_rest_mib - own[0].pre_kv_rest_mib == pytest.approx(
        VOCAB_MIB
    )
    assert [f.pre_kv_rest_mib for f in shared[1:]] == [
        f.pre_kv_rest_mib for f in own[1:]
    ]


def test_the_p_ceiling_never_names_a_fraction_without_a_platztausch_buffer():
    """Bug regression (P-Seite, dieselbe Luecke): die lineare Form nannte 1.000,
    sobald ``f*E + LRU`` Zeilen passten -- bei f >= 1 baut die Runtime keinen
    Offload-Puffer (H5, W120), und oberhalb (E-LRU)/E haelt sie nur E Zeilen.
    Die groesste Fraction mit Puffer ist floor((E-2)/E) = 0.996."""
    from sglang.srt.planner import pp_cut

    got = pp_cut.solve_expert_fraction_per_stage(
        budgets_mib=[28264, 17840, 17168],
        stage_layers=[29, 11, 8],
        mean_layer_mib=61.6,
        expert_layer_mib=1297637376 / (1 << 20),
        num_experts=512,
        lru_rows=[32, 32, 32],
        reserve_mib_by_stage=[2176, 1088, 816],
    )
    assert got[1] == got[2] == 0.996
    # Stufe 0 bindet: 346 Zeilen passen -> R = 346 - 32 = 314 -> 314/512.
    assert got[0] == 0.613


def _launcher_ns(tmp_path, fractions):
    cfg = {
        "text_config": {
            "num_hidden_layers": 48,
            "vocab_size": 248320,
            "hidden_size": 2560,
        }
    }
    (tmp_path / MODEL).mkdir(parents=True)
    (tmp_path / MODEL / "config.json").write_text(json.dumps(cfg))
    return types.SimpleNamespace(
        model=str(tmp_path / MODEL),
        extra_d=(
            "--rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
            "--rank-moe-resident-fraction " + fractions
        ),
        env_d=(
            "SGLANG_MOE_POOL_STAGING=8;SGLANG_MOE_SCRATCH_SLOTS=70,48,48;"
            "SGLANG_UNEVEN_MOE_EXPERT_SHARD=1;SGLANG_WEG2_DRAFT_SHARE_EMBED=0"
        ),
        d_foreign_context_mib="",
        d_nontorch_mib="",
        d_reserve_mib="",
        d_residency_reference_logs="",
    )


def test_the_launcher_refuses_x98_before_a_rank_loads(tmp_path, monkeypatch):
    """Die Naht: ``log_d_rank_vram_solve`` -- an BEIDEN budgets_d-Stellen
    aufgerufen, auch im Dry-Run -- verweigert x98s argv (0.006,0.75,0.54) mit
    W122, und laesst die gemessene Decke durch."""
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import launcher

    monkeypatch.setattr(
        pp_cut,
        "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(
            expert_layer_weight_bytes=1297637376.0, num_experts=512, n_layers=48
        ),
    )
    cards = [
        launcher.Card(1, "u1", "RTX 5090", 32607),
        launcher.Card(0, "u0", "RTX 3080", 20480),
        launcher.Card(2, "u2", "RTX 3080", 20480),
    ]
    lines = []
    with pytest.raises(launcher.Weg2LaunchRefused, match=r"^W122 .*rang1.*rang2"):
        launcher.log_d_rank_vram_solve(
            _launcher_ns(tmp_path / "a", "0.006,0.75,0.54"),
            cards,
            LIVE_BUDGETS,
            lines.append,
            "D(dry, expectation)",
        )
    assert any("STIRBT AM KV-POOL" in ln for ln in lines)
    lines.clear()
    # H33: die BUDGET-Decke laesst W122 durch; seit der Karten-Bilanz sagt der
    # Launcher daneben, dass 0.098 bei eigenem Draft-Vokabular (+2425 MiB auf
    # dem Draft-Host) nicht auf die 5090 passt (W130 rang0) -- die Budget-
    # Decke war nie eine Karten-Decke (H30 R1).
    with pytest.raises(launcher.Weg2LaunchRefused, match=r"^W130 .*rang0"):
        launcher.log_d_rank_vram_solve(
            _launcher_ns(tmp_path / "b", "0.098,0.634,0.519"),
            cards,
            LIVE_BUDGETS,
            lines.append,
            "D",
        )
    assert any("DECKE je Rang ['0.098', '0.634', '0.519']" in ln for ln in lines), lines
    assert not any(ln.startswith("W122") for ln in lines)
