# SPDX-License-Identifier: Apache-2.0
"""#239 -- der D-Planer bucht den Token-Schnitt der Voll-Attention-KV.

Unter Form A haelt der Attention-Host (TP0, 5090) die ganze KV, die Worker
(3080) keine; der Planer bepreist jeden Rang mit der Zelle seiner Referenz
(``kv_tokens x cell_r``, expert_residency.solve_d_rank_residency). Das
Release-Feature uneven-DCP-KV schneidet die Voll-Attention-KV (12 FA-Layer x
2 kv-Heads x 256 x K+V x fp8 = 12288 B/Token) nach TOKENS ueber alle Raenge;
Indexer und MTP-Draft-KV bleiben auf dem Host. Der Planer muss das rechnen,
bevor die Runtime es faehrt -- sonst setzt er FR fuer eine Karte, die es
nicht gibt (Nutzer-Rüge 27.09.: FR setzt der Planer, keine Handwerte).

Die Zahlen sind die Buchung des rc12r-Boots (launcher.log
docker_dkrnfh91dprbar1dauer09271632 Z.220-228): Budget [26328, 17664, 17864],
fest [7481, 1075, 953] (Record + Gather-Ring), mamba 2134.5 / spec 133.9 auf
TP0, Aktivierung [1104, 1024, 1024], Zelle [14143, 768, 768], Decke
[102, 132, 135] Zeilen.
"""

import msgspec
import pytest

from sglang.srt.planner import expert_residency as er

SLOT_BYTES = 1297637376 / 512
FA_CELL = 12288
BUDGETS = (26328, 17664, 17864)
RC12R = msgspec.structs.replace(
    er.D_RESIDENCY_REFERENCE_FNFL2_H39,
    source="rc12r launcher.log Z.220-228",
    fixed_mib=(7481.0, 1075.0, 953.0),
    mamba_mib=(2134.5, 0.0, 0.0),
    spec_mib=(133.9, 0.0, 0.0),
    activation_mib=(1104.0, 1024.0, 1024.0),
)


def _solve(kv_tokens=262144, shares=None, reference=RC12R, dcp=FA_CELL):
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
        reference=reference,
        vocab_mib=er.draft_vocab_mib(vocab_size=248320, hidden_size=2560),
        share_embed=True,
        kv_tokens=kv_tokens,
        kv_token_shares=shares,
        kv_dcp_cell_bytes=dcp if shares is not None else 0,
    )


def test_without_shares_the_booking_is_the_rc12r_log():
    fits = _solve()
    # #239 S0: Worker ohne QSA-Schluessel (768 B/Token), +192 MiB je Worker (rc12r buchte sie noch)
    assert [f.ceiling_max_rows for f in fits] == [102, 134, 136]
    assert [f.kv_cell_bytes for f in fits] == [14143, 0, 0]
    assert all(f.kv_token_share == -1.0 for f in fits)
    assert all("Token-Schnitt" not in er.describe_rank(f) for f in fits)


def test_the_form_a_cut_is_the_identity():
    """Anteil (1,0,0) ist genau die heutige Form A -- gleiche Bytes."""
    plain, cut = _solve(), _solve(shares=(1, 0, 0))
    assert [f.kv_mib for f in cut] == pytest.approx([f.kv_mib for f in plain])
    assert [f.ceiling_max_rows for f in cut] == [f.ceiling_max_rows for f in plain]


def test_a_third_each_moves_the_full_attention_kv_off_the_host():
    fits = _solve(shares=(1, 1, 1))
    # Host: 14143 - 12288 + 4096; Worker: 4096 (seit S0 ohne 768 B QSA-Schluessel)
    assert [f.kv_cell_bytes for f in fits] == [5951, 4096, 4096]
    assert [round(f.kv_token_share, 6) for f in fits] == [round(1 / 3, 6)] * 3
    assert [f.ceiling_max_rows for f in fits] == [120, 125, 128]
    assert "Token-Schnitt, Anteil 0.333" in er.describe_rank(fits[0])


def test_x2_is_where_the_cut_decides_the_host():
    """YaRN x2: Form A laesst TP0 72 Zeilen, der Drittelschnitt 107."""
    assert [f.ceiling_max_rows for f in _solve(kv_tokens=524288)] == [72, 134, 136]
    assert [f.ceiling_max_rows for f in _solve(kv_tokens=524288, shares=(1, 1, 1))] == [
        107,
        116,
        119,
    ]


def test_x4_does_not_run_on_a_single_host():
    """1M: Form A laesst TP0 11 Zeilen -- weniger als sein eigener Scratch."""
    form_a = _solve(kv_tokens=1048576)
    assert form_a[0].ceiling_max_rows == 11
    assert form_a[0].ceiling_max_rows < form_a[0].scratch_rows
    assert _solve(kv_tokens=1048576, shares=(1, 1, 1))[0].ceiling_max_rows == 82


def test_shares_are_a_ratio_vector():
    a = _solve(shares=(2, 1, 1))
    b = _solve(shares=(0.5, 0.25, 0.25))
    assert [f.kv_mib for f in a] == pytest.approx([f.kv_mib for f in b])


@pytest.mark.parametrize(
    "shares, dcp, match",
    [
        ((1, 1), FA_CELL, "3 D ranks, but 2"),
        ((1, -1, 1), FA_CELL, "not a ratio vector"),
        ((0, 0, 0), FA_CELL, "not a ratio vector"),
        ((1, 1, 1), 14144, "not part of the host cell"),
        ((1, 1, 1), 0, "not part of the host cell"),
    ],
)
def test_a_cut_that_cannot_be_priced_is_refused(shares, dcp, match):
    with pytest.raises(ValueError, match=match):
        _solve(shares=shares, dcp=dcp)


def test_a_reference_that_is_already_a_dcp_slice_is_refused():
    ref = msgspec.structs.replace(RC12R, rank_tp_ratio="2,1,1")
    with pytest.raises(ValueError, match="ONE attention host"):
        _solve(shares=(1, 1, 1), reference=ref)
