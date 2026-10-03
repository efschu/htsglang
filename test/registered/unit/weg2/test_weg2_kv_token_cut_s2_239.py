# SPDX-License-Identifier: Apache-2.0
"""#239 S2 -- der Token-Schnitt der Voll-Attention-KV als Formwert.

S1 lehrte den D-Planer, einen GEGEBENEN Anteilsvektor zu bepreisen. S2 macht
daraus eine Form: ``--d-kv-token-cut off|maxmin|<Vektor>`` setzt die Achse
``kv=qsa_forma_dcp``, der Planer waehlt die Anteile selbst (max-min der
relativen Zeilen-Decke, in 64steln -- kein Handwert), und ein ECHTER Boot wird
benannt verweigert, solange die Worker keine Attention ueber ihren KV-Anteil
rechnen (S3, Naht F5). ``off`` ist byte-gleich zu heute.

Referenz: die Buchung des rc12r-Boots (launcher.log
docker_dkrnfh91dprbar1dauer09271632 Z.220-228), dieselbe wie in
test_weg2_d_kv_token_cut_239.
"""

from types import SimpleNamespace

import msgspec
import pytest

from sglang.srt.planner import expert_residency as er
from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import launcher as L

SLOT_BYTES = 1297637376 / 512
BUDGETS = (26328, 17664, 17864)
RC12R = msgspec.structs.replace(
    er.D_RESIDENCY_REFERENCE_FNFL2_H39,
    source="rc12r launcher.log Z.220-228",
    fixed_mib=(7481.0, 1075.0, 953.0),
    mamba_mib=(2134.5, 0.0, 0.0),
    spec_mib=(133.9, 0.0, 0.0),
    activation_mib=(1104.0, 1024.0, 1024.0),
)
NF_TEXT_CFG = {
    "num_hidden_layers": 48,
    "full_attention_interval": 4,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "layer_types": ["linear_attention"] * 3 * 12 + ["full_attention"] * 12,
}
ROLES = "--rank-role host,worker,worker --rank-tp-ratio 1,0,0"


def _solver(kv_tokens):
    def solve(shares):
        return er.solve_d_rank_residency(
            budgets_mib=BUDGETS,
            fractions=(0.06, 0.51, 0.48),
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
            fits = solve((a, b, grid - a - b))
            s = min(f.ceiling_max_rows / max(1, f.buffer_rows) for f in fits)
            best = s if best is None else max(best, s)
    return best


# --------------------------------------------------------------------------
# the planner: max-min over the shares, from the checkpoint's own cell
# --------------------------------------------------------------------------


def test_the_nf_full_attention_cell_comes_from_the_config():
    assert er.fa_kv_token_cell_bytes(NF_TEXT_CFG, 1) == 12288
    no_types = {k: v for k, v in NF_TEXT_CFG.items() if k != "layer_types"}
    assert er.fa_kv_token_cell_bytes(no_types, 1) == 12288
    assert er.fa_kv_token_cell_bytes(NF_TEXT_CFG, 2) == 24576
    with pytest.raises(ValueError, match="no full-attention KV geometry"):
        er.fa_kv_token_cell_bytes({"num_hidden_layers": 48}, 1)


@pytest.mark.parametrize("mult", [1, 2, 4])
def test_the_maxmin_cut_is_the_optimum_of_the_grid(mult):
    """Der Loeser trifft das Optimum der vollen Suche ueber alle 64stel."""
    solve = _solver(262144 * mult)
    shares, low = er.solve_kv_token_shares(solve, 3)
    assert sum(shares) == er.KV_TOKEN_SHARE_GRID
    assert low == pytest.approx(_brute_force(solve, er.KV_TOKEN_SHARE_GRID))
    fits = solve(shares)
    assert min(f.ceiling_max_rows / max(1, f.buffer_rows) for f in fits) == pytest.approx(low)


def test_the_cut_lifts_the_host_ceiling_where_form_a_collapses():
    """x2: Form A laesst TP0 72 Zeilen; der Schnitt hebt die kleinste
    relative Decke und mit ihr TP0 deutlich. x4: Form A laesst TP0 11 Zeilen
    (weniger als die 12 Staging-Zeilen -- keine Form); mit dem Schnitt 77,
    unter dem heutigen Scratch 90 -- der rc12c-Deckel (d_scratch_cap) senkt
    den Scratch dort auf die Kante, die Form bleibt fahrbar."""
    solve2 = _solver(524288)
    shares, _ = er.solve_kv_token_shares(solve2, 3)
    assert solve2((1, 0, 0))[0].ceiling_max_rows == 72
    assert solve2(shares)[0].ceiling_max_rows >= 96
    solve4 = _solver(1048576)
    shares4, _ = er.solve_kv_token_shares(solve4, 3)
    host = solve4(shares4)[0]
    assert solve4((1, 0, 0))[0].ceiling_max_rows == 11
    assert host.ceiling_max_rows >= 77
    assert host.ceiling_max_rows < host.scratch_rows


def test_a_non_monotone_solve_is_refused():
    def solve(shares):
        k = shares[0]
        rows = 50 + (10 if k == 3 else 0) - k
        return [SimpleNamespace(ceiling_max_rows=rows, buffer_rows=50)] * 3

    with pytest.raises(ValueError, match="not monotone"):
        er.solve_kv_token_shares(solve, 3, grid=8)


# --------------------------------------------------------------------------
# the form axis
# --------------------------------------------------------------------------


def test_off_keeps_the_form_a_value():
    words = ROLES.split()
    assert F.derive_kv(words) == F.derive_kv(words, "off")
    assert F.derive_kv(words)[0] == "qsa_forma"
    assert F.derive_kv([])[0] == "paged_dcp"


def test_the_cut_is_its_own_kv_value_of_the_nf_profile():
    kv, src = F.derive_kv(ROLES.split(), "maxmin")
    assert kv == "qsa_forma_dcp"
    assert "--d-kv-token-cut maxmin" in src
    assert "qsa_forma_dcp" in F.AXIS_VALUES["kv"]
    assert "qsa_forma_dcp" in F.PROFILE_EXPECT[F.PROFILE_NEXTFLASH]["kv"]
    assert "qsa_forma_dcp" not in F.PROFILE_EXPECT[F.PROFILE_QWEN27B]["kv"]


def test_a_cut_without_form_a_is_a_contradiction():
    with pytest.raises(F.Weg2FormContradiction, match="no worker rank role"):
        F.derive_kv([], "maxmin")


# --------------------------------------------------------------------------
# the launcher: switch, dtype, refusal of the real boot
# --------------------------------------------------------------------------


def test_the_switch_defaults_to_off():
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
    assert ns.d_kv_token_cut == "off"
    assert L.d_kv_token_cut(ns) is None


def test_the_switch_reads_maxmin_and_ratio_vectors():
    assert L.d_kv_token_cut(SimpleNamespace(d_kv_token_cut="maxmin")) == "maxmin"
    assert L.d_kv_token_cut(SimpleNamespace(d_kv_token_cut="2,1,1")) == (2.0, 1.0, 1.0)
    for bad in ("x", "1,-1,1", "0,0,0"):
        with pytest.raises(L.Weg2LaunchRefused, match="#239 KV-TOKEN-SCHNITT"):
            L.d_kv_token_cut(SimpleNamespace(d_kv_token_cut=bad))


def test_the_kv_dtype_follows_the_d_argv_else_the_profile():
    nf = SimpleNamespace(extra_d="", profile=F.PROFILE_NEXTFLASH)
    assert L.d_kv_dtype_bytes(nf) == 1
    bf16 = SimpleNamespace(extra_d="--kv-cache-dtype bfloat16", profile=F.PROFILE_NEXTFLASH)
    assert L.d_kv_dtype_bytes(bf16) == 2


def _form(kv):
    return SimpleNamespace(kv=kv)


def test_a_real_boot_of_the_cut_is_refused_by_name_while_a_seam_is_open(monkeypatch):
    import dataclasses

    from sglang.srt import rank_role

    ns = SimpleNamespace(dry_run=False, d_kv_token_cut="maxmin")
    L.refuse_unwired_token_cut(ns, _form("qsa_forma_dcp"))  # #239 S4b part 7: every seam wired
    open_f14 = dict(rank_role.SEAMS)
    open_f14["F14"] = dataclasses.replace(open_f14["F14"], wired=False)
    monkeypatch.setattr(rank_role, "SEAMS", open_f14)
    with pytest.raises(L.Weg2TokenCutNotWired, match="#239 S3"):
        L.refuse_unwired_token_cut(ns, _form("qsa_forma_dcp"))


def test_the_dry_run_and_every_other_form_pass():
    L.refuse_unwired_token_cut(SimpleNamespace(dry_run=True, d_kv_token_cut="maxmin"),
                               _form("qsa_forma_dcp"))
    for kv in ("qsa_forma", "paged_dcp"):
        L.refuse_unwired_token_cut(SimpleNamespace(dry_run=False, d_kv_token_cut="off"), _form(kv))
    L.refuse_unwired_token_cut(SimpleNamespace(dry_run=False), None)


def test_every_d_solve_of_the_launcher_carries_the_cut():
    """Die D-Rechnung laeuft an drei Stellen (Solve, Scratch-Deckel-Neusolve,
    Sitz-Tabelle) -- eine Stelle ohne Schnitt druckte eine zweite Wahrheit."""
    import inspect

    src = inspect.getsource(L.log_d_rank_vram_solve)
    assert src.count("_er.plan_d_residency(") == 2
    # #239 S3f: the seat table carries the cut through d_seat_table_form (the
    # SOLVED vector under 'owned', else the stated kwargs unchanged)
    assert src.count("**_kv_cut_kw") == 2
    assert "d_seat_table_form(plan, _kv_cut_kw, fr_d)" in src
    assert src.count("**_seat_cut_kw") == 1
