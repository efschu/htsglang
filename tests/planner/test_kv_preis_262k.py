"""Der KV-Preis gegen die am Metall emittierten Zellen von fnFL2w123."""
import pytest
from flliper.srt.planner.pp_cut import (
    kv_cell_bytes_per_attention_layer,
    kv_reserve_mib_per_stage,
)

NF = dict(kv_heads=2, head_dim=256, v_head_dim=256, kv_dtype_bytes=1)  # fp8_e4m3


def test_cell_per_attention_layer_is_1088():
    assert kv_cell_bytes_per_attention_layer(**NF) == 1088.0


def test_cells_match_the_three_measured_of_w123():
    """boot_weg2_fnFL2w123...P.log: 'cell_size=8704' / '=4352' / '=3264'.

    Stufen 29/11/8 Layer, davon 7/3/2 full_attention (config layer_types),
    plus je EIN Draft-Layer (--draft-kv-on-p on).
    """
    cell = kv_cell_bytes_per_attention_layer(**NF)
    measured = [8704, 4352, 3264]
    for attn, want in zip([7, 3, 2], measured):
        assert (attn + 1) * cell == want, f"{attn=} -> {(attn+1)*cell} != {want}"


def test_without_draft_matches_design_doc():
    """DESIGN_FLIP_NEXTFLASH_0920.md: PP0 braucht '7616 x 262144' -- ohne Draft."""
    assert 7 * kv_cell_bytes_per_attention_layer(**NF) == 7616.0


def test_262k_price_per_stage():
    mib = kv_reserve_mib_per_stage(
        tokens=262144, attn_layers_by_stage=[7, 3, 2],
        draft_attn_layers_by_stage=[1, 1, 1], **NF
    )
    assert [round(x) for x in mib] == [2176, 1088, 816]


def test_without_draft_price_is_lower_bound():
    mit = kv_reserve_mib_per_stage(
        tokens=262144, attn_layers_by_stage=[7, 3, 2],
        draft_attn_layers_by_stage=[1, 1, 1], **NF)
    without = kv_reserve_mib_per_stage(
        tokens=262144, attn_layers_by_stage=[7, 3, 2], **NF)
    assert all(o < m for o, m in zip(without, mit))


def test_half_geometry_solves_nothing():
    with pytest.raises(ValueError):
        kv_reserve_mib_per_stage(
            tokens=262144, attn_layers_by_stage=[7, 3, 2],
            draft_attn_layers_by_stage=[1, 1], **NF)


def test_price_enters_fraction_ceiling_as_reserve():
    """Die Verdrahtung: mit KV-Posten muss die Decke SINKEN."""
    from flliper.srt.planner.pp_cut import solve_expert_fraction_per_stage
    kw = dict(budgets_mib=[27080, 16328, 16048], stage_layers=[29, 11, 8],
              mean_layer_mib=62.0, expert_layer_mib=1238.0, num_experts=512,
              lru_rows=[32, 32, 32])
    without = solve_expert_fraction_per_stage(**kw)
    kv = kv_reserve_mib_per_stage(
        tokens=262144, attn_layers_by_stage=[7, 3, 2],
        draft_attn_layers_by_stage=[1, 1, 1], **NF)
    mit = solve_expert_fraction_per_stage(**kw, reserve_mib_by_stage=kv)
    # Nur wo die Decke nicht schon bei 1.0 klemmt, kann der Posten sie senken.
    # Dass PP1/PP2 bei 1.0 stehen, waehrend PP2 am Metall nur 380 MiB fuer KV
    # hatte, ist SELBST ein Befund: die Launcher-Decke kennt die Posten aus
    # design_bytes-first-lawful.md 1.1 nicht (Korridor, Mamba, Spec-Zwischen-
    # zustand, Prefill-Aktivierung, Draft, Graph-Pools, CUDA-Kontext).
    assert mit[0] < without[0], f"{mit=} {without=}"
    assert all(m <= o for m, o in zip(mit, without))
    print(f"\nDecke OHNE KV-Posten: {[round(f,3) for f in without]}")
    print(f"Decke MIT  KV-Posten: {[round(f,3) for f in mit]}")
    print(f"KV-Preis je Stufe MiB: {[round(x) for x in kv]}")
