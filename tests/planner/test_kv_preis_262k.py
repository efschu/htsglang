"""Der KV-Preis gegen die am Metall emittierten Zellen von fnFL2w123."""
import pytest
from sglang.srt.planner.pp_cut import (
    kv_cell_bytes_per_attention_layer,
    kv_reserve_mib_per_stage,
)

NF = dict(kv_heads=2, head_dim=256, v_head_dim=256, kv_dtype_bytes=1)  # fp8_e4m3


def test_zelle_je_attention_layer_ist_1088():
    assert kv_cell_bytes_per_attention_layer(**NF) == 1088.0


def test_zellen_treffen_die_drei_gemessenen_von_w123():
    """boot_weg2_fnFL2w123...P.log: 'cell_size=8704' / '=4352' / '=3264'.

    Stufen 29/11/8 Layer, davon 7/3/2 full_attention (config layer_types),
    plus je EIN Draft-Layer (--draft-kv-on-p on).
    """
    cell = kv_cell_bytes_per_attention_layer(**NF)
    gemessen = [8704, 4352, 3264]
    for attn, soll in zip([7, 3, 2], gemessen):
        assert (attn + 1) * cell == soll, f"{attn=} -> {(attn+1)*cell} != {soll}"


def test_ohne_draft_trifft_das_design_dokument():
    """DESIGN_FLIP_NEXTFLASH_0920.md: PP0 braucht '7616 x 262144' -- ohne Draft."""
    assert 7 * kv_cell_bytes_per_attention_layer(**NF) == 7616.0


def test_262k_preis_je_stufe():
    mib = kv_reserve_mib_per_stage(
        tokens=262144, attn_layers_by_stage=[7, 3, 2],
        draft_attn_layers_by_stage=[1, 1, 1], **NF
    )
    assert [round(x) for x in mib] == [2176, 1088, 816]


def test_ohne_draft_angabe_ist_der_preis_kleiner_also_eine_untergrenze():
    mit = kv_reserve_mib_per_stage(
        tokens=262144, attn_layers_by_stage=[7, 3, 2],
        draft_attn_layers_by_stage=[1, 1, 1], **NF)
    ohne = kv_reserve_mib_per_stage(
        tokens=262144, attn_layers_by_stage=[7, 3, 2], **NF)
    assert all(o < m for o, m in zip(ohne, mit))


def test_halbe_geometrie_loest_nichts():
    with pytest.raises(ValueError):
        kv_reserve_mib_per_stage(
            tokens=262144, attn_layers_by_stage=[7, 3, 2],
            draft_attn_layers_by_stage=[1, 1], **NF)


def test_der_preis_fliesst_als_reserve_in_die_fraction_decke():
    """Die Verdrahtung: mit KV-Posten muss die Decke SINKEN."""
    from sglang.srt.planner.pp_cut import solve_expert_fraction_per_stage
    kw = dict(budgets_mib=[27080, 16328, 16048], stage_layers=[29, 11, 8],
              mean_layer_mib=62.0, expert_layer_mib=1238.0, num_experts=512,
              lru_rows=[32, 32, 32])
    ohne = solve_expert_fraction_per_stage(**kw)
    kv = kv_reserve_mib_per_stage(
        tokens=262144, attn_layers_by_stage=[7, 3, 2],
        draft_attn_layers_by_stage=[1, 1, 1], **NF)
    mit = solve_expert_fraction_per_stage(**kw, reserve_mib_by_stage=kv)
    # Nur wo die Decke nicht schon bei 1.0 klemmt, kann der Posten sie senken.
    # Dass PP1/PP2 bei 1.0 stehen, waehrend PP2 am Metall nur 380 MiB fuer KV
    # hatte, ist SELBST ein Befund: die Launcher-Decke kennt die Posten aus
    # design_bytes-first-lawful.md 1.1 nicht (Korridor, Mamba, Spec-Zwischen-
    # zustand, Prefill-Aktivierung, Draft, Graph-Pools, CUDA-Kontext).
    assert mit[0] < ohne[0], f"{mit=} {ohne=}"
    assert all(m <= o for m, o in zip(mit, ohne))
    print(f"\nDecke OHNE KV-Posten: {[round(f,3) for f in ohne]}")
    print(f"Decke MIT  KV-Posten: {[round(f,3) for f in mit]}")
    print(f"KV-Preis je Stufe MiB: {[round(x) for x in kv]}")
