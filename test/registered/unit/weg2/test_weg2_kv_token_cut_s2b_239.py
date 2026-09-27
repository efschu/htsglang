# SPDX-License-Identifier: Apache-2.0
"""#239 S2b -- Token-Schnitt UND FR_D in einem Solve.

S2 waehlte die Anteile gegen die GEFAHRENE FR_D: bei x1 blieb der Schnitt
Form A ([64,0,0]), bei x2 verweigerte W122 alle drei Raenge (die Worker-FR
passte nicht zu ihrem neuen KV-Anteil). FR setzt der Planer (Nutzer-Ruege
27.09.): ``--d-kv-token-cut joint`` waehlt den Schnitt als max-min des
residenten Experten-Anteils je Karte an der Kante und setzt FR_D auf genau
diese Kante. Die Decke eines Rangs haengt nicht an seiner FR, also ist das
ein Solve, keine Schleife.

Referenz wie test_weg2_d_kv_token_cut_239: die rc12r-Buchung (launcher.log
docker_dkrnfh91dprbar1dauer09271632 Z.220-228).
"""

import inspect
from types import SimpleNamespace

import msgspec
import pytest

from sglang.srt.planner import expert_residency as er
from sglang.srt.weg2 import launcher as L

SLOT_BYTES = 1297637376 / 512
BUDGETS = (26328, 17664, 17864)
FR_DRIVEN = (0.06, 0.51, 0.48)
RC12R = msgspec.structs.replace(
    er.D_RESIDENCY_REFERENCE_FNFL2_H39,
    source="rc12r launcher.log Z.220-228",
    fixed_mib=(7481.0, 1075.0, 953.0),
    mamba_mib=(2134.5, 0.0, 0.0),
    spec_mib=(133.9, 0.0, 0.0),
    activation_mib=(1104.0, 1024.0, 1024.0),
)


def _solver(kv_tokens, fractions=FR_DRIVEN):
    def solve(shares, fr=None):
        return er.solve_d_rank_residency(
            budgets_mib=BUDGETS,
            fractions=fractions if fr is None else fr,
            ratios=(183, 137, 168),
            scratch_rows=(90, 48, 48),
            staging_rows=12,
            num_experts=512,
            pad_rows=1,
            n_layers=48,
            slot_bytes=SLOT_BYTES,
            reference=RC12R,
            vocab_mib=er.draft_vocab_mib(vocab_size=248320, hidden_size=2560),
            share_embed=True,
            kv_tokens=kv_tokens,
            kv_token_shares=shares,
            kv_dcp_cell_bytes=12288,
        )

    return solve


def _brute_force(solve, grid):
    best = None
    for a in range(grid + 1):
        for b in range(grid + 1 - a):
            s = min(er.resident_edge_share(f) for f in solve((a, b, grid - a - b)))
            best = s if best is None else max(best, s)
    return best


def test_the_resident_edge_share_is_the_edge_of_the_card():
    fit = _solver(262144)((1, 0, 0))[0]
    # TP0 x1 Form A: Decke 102, Scratch 90 -> 12 residente von 193 Experten
    assert fit.ceiling_max_rows == 102
    assert er.resident_edge_share(fit) == pytest.approx(12 / fit.local_experts)
    full = SimpleNamespace(local_experts=50, ceiling_max_rows=500, scratch_rows=10)
    assert er.resident_edge_share(full) == pytest.approx(48 / 50)
    none = SimpleNamespace(local_experts=50, ceiling_max_rows=5, scratch_rows=10)
    assert er.resident_edge_share(none) < 0


@pytest.mark.parametrize("mult", [1, 2, 4])
def test_the_joint_cut_is_the_optimum_of_the_grid(mult):
    solve = _solver(262144 * mult)
    cut, low, fr, edge = er.solve_joint_cut(solve, FR_DRIVEN, 3)
    assert sum(cut) == er.KV_TOKEN_SHARE_GRID
    assert low == pytest.approx(_brute_force(solve, er.KV_TOKEN_SHARE_GRID))
    assert min(er.resident_edge_share(f) for f in edge) == pytest.approx(low)


@pytest.mark.parametrize("mult", [1, 2, 4])
def test_fr_d_at_the_edge_is_never_refused(mult):
    """Die FR, die der Planer setzt, passt auf jedem Rang (kein W122)."""
    solve = _solver(262144 * mult)
    cut, _, fr, edge = er.solve_joint_cut(solve, FR_DRIVEN, 3)
    assert all(f.ceiling_fraction is not None for f in edge)
    assert fr == tuple(f.ceiling_fraction for f in edge)
    fits = solve(cut, fr)
    assert er.refusal_text(fits, label="x%d" % mult) is None


def test_x2_the_joint_form_boots_where_form_a_and_s2_were_refused():
    solve = _solver(524288)
    assert er.refusal_text(solve((1, 0, 0)), label="x2") is not None
    cut, _, fr, _ = er.solve_joint_cut(solve, FR_DRIVEN, 3)
    fits = solve(cut, fr)
    assert er.refusal_text(fits, label="x2") is None
    # die 5090 haelt mehr eigene Experten als Form A an seiner Kante (0 bei x2)
    assert fits[0].resident_rows > 12


def test_x1_moves_experts_onto_the_5090():
    """Release-Ziel: mehr Experten auf der 5090. Form A an der Kante haelt dort
    12 (FR 0.06 gefahren) bzw. 102-90 = 12 residente Zeilen."""
    solve = _solver(262144)
    cut, _, fr, _ = er.solve_joint_cut(solve, FR_DRIVEN, 3)
    fits = solve(cut, fr)
    assert fits[0].resident_rows > 12
    assert cut[0] < er.KV_TOKEN_SHARE_GRID


def test_a_rank_without_edge_keeps_its_driven_fraction():
    def solve(shares):
        return [
            SimpleNamespace(rank=r, local_experts=50, ceiling_max_rows=(5 if r == 0 else 40),
                            scratch_rows=10, ceiling_fraction=(None if r == 0 else 0.5),
                            buffer_rows=20)
            for r in range(2)
        ]

    _, _, fr, _ = er.solve_joint_cut(solve, (0.3, 0.4), 2, grid=4)
    assert fr == (0.3, 0.5)


def test_the_launcher_reads_joint_and_publishes_fr_d():
    assert L.d_kv_token_cut(SimpleNamespace(d_kv_token_cut="joint")) == "joint"
    src = inspect.getsource(L.log_d_rank_vram_solve)
    assert "plan.solved_fractions" in src
    assert '"--rank-moe-resident-fraction"' in src
    assert "SGLANG_MOE_RESIDENT_EXPERT_FRACTION" in src


def test_the_plan_carries_the_solved_form():
    fields = {f.name for f in msgspec.structs.fields(er.DResidencyPlan)}
    assert {"solved_fractions", "kv_token_cut"} <= fields
    plan = er.DResidencyPlan(lines=(), refusal=None)
    assert plan.solved_fractions == () and plan.kv_token_cut == ()
