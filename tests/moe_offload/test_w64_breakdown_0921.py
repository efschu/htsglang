"""#62 (21.09.): a W64 feasible=False refusal must NAME the rank it refuses
for and the four terms that ate its budget.

Three fnFL2 boots read "does not leave a positive KV pool on at least one
rank" and had to re-derive which rank, and by how much, by hand.  The clause
is built from the same model that produced the verdict, so the test grades
BOTH that the arithmetic is printed and that it reconciles with the budget.
"""

import types

from sglang.srt.weg2.launcher import _infeasible_breakdown
from sglang.srt.uneven_perf import (
    _PREDICT_MAMBA_ACT_RESERVE_MIB,
    _PREDICT_MIN_RANK_TOKENS,
    _PREDICT_OVERHEAD_MIB,
)

MI = 2 ** 20
RESERVES = _PREDICT_OVERHEAD_MIB + _PREDICT_MAMBA_ACT_RESERVE_MIB


def _pcm(weights_mib, mamba_mib, cell_bytes, offloaded_mib=None):
    off = offloaded_mib or [0] * len(weights_mib)
    return types.SimpleNamespace(
        per_rank_weight_bytes=lambda mlp, attn: [w * MI for w in weights_mib],
        per_rank_offloaded_weight_bytes=lambda mlp: [o * MI for o in off],
        mamba_pool_bytes_for=lambda attn: [m * MI for m in mamba_mib],
        kv_cell_bytes=cell_bytes,
    )


def test_the_refused_rank_is_named_and_the_terms_add_up():
    budgets = [28240, 16672, 16672]
    weights = [24000, 12000, 12000]
    mamba = [900, 400, 400]
    pcm = _pcm(weights, mamba, 14143.0)
    free0 = budgets[0] - weights[0] - mamba[0] - RESERVES
    p = [free0 * MI / 14143.0, 100000.0, 100000.0]
    out = _infeasible_breakdown(pcm, [58, 25, 25], [58, 25, 25], budgets, {"p": p})
    assert "group=D r0 budget=28240" in out
    assert "weights=24000" in out and "mamba=900" in out
    assert f"reserves={RESERVES}" in out
    assert f"free={free0}" in out
    assert "kv_cell=14143" in out
    assert str(_PREDICT_MIN_RANK_TOKENS) in out
    # the refused rank carries the marker, the funded ones do not
    assert out.count("<-- BELOW the minimum") == 0  # this rank is fine


def test_a_rank_below_the_minimum_is_marked_and_only_that_one():
    budgets = [28240, 16672, 16672]
    pcm = _pcm([28000, 12000, 12000], [900, 400, 400], 14143.0)
    p = [10.0, 100000.0, 100000.0]
    out = _infeasible_breakdown(pcm, [58, 25, 25], [58, 25, 25], budgets, {"p": p})
    assert out.count("<-- BELOW the minimum") == 1
    assert "group=D r0 budget=28240" in out and "= 10 tokens  <-- BELOW" in out


def test_every_rank_appears_even_when_all_are_below():
    budgets = [100, 100, 100]
    pcm = _pcm([90, 90, 90], [5, 5, 5], 1024.0)
    out = _infeasible_breakdown(
        pcm, [1, 1, 1], [1, 1, 1], budgets, {"p": [0.0, 0.0, 0.0]}
    )
    assert out.count("<-- BELOW the minimum") == 3
    for r in range(3):
        assert f"group=D r{r} budget=100" in out


def test_a_broken_model_degrades_to_a_named_note_not_an_exception():
    class Boom:
        def per_rank_weight_bytes(self, mlp, attn):
            raise RuntimeError("no gemm scores")

        def per_rank_offloaded_weight_bytes(self, mlp):
            return [0.0]

    out = _infeasible_breakdown(Boom(), [1], [1], [100], {"p": [0.0]})
    assert "per-rank breakdown unavailable" in out and "no gemm scores" in out


def test_the_refusal_text_carries_the_clause():
    import inspect

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher.d_operating_point_rows)
    assert "_infeasible_breakdown(pcm, mlp, attn_units, budgets, cap, cards)" in src


def test_the_host_store_share_is_named_as_its_own_term():
    """Die Zeile darf nie wieder so lesen, als traege die Karte alles: der
    Anteil im Host-Expertenspeicher steht daneben (#62, fnFL2v80)."""
    pcm = _pcm([89398], [619], 13312.0, offloaded_mib=[80000])
    out = _infeasible_breakdown(pcm, [58], [58], [28240], {"p": [100.0]})
    assert "weights=9398 (of which 80000 in the host expert store)" in out


def test_the_line_carries_the_keys_a_reader_needs_to_join_it():
    """Ein Peer-Werkzeug konnte diese Zeile nicht zuordnen: der Zensus spricht
    pp<PP>tp<TP>, die Budgetzeilen card=/nvml_idx=, und diese Zeile sprach
    keines von beidem (gemeldet 21.09.). Gruppe, Rang, nvml-Index und
    Kartenname stehen jetzt drin."""
    import types as _t

    cards = [
        _t.SimpleNamespace(nvml_index=1, name="NVIDIA GeForce RTX 5090"),
        _t.SimpleNamespace(nvml_index=0, name="NVIDIA GeForce RTX 3080"),
    ]
    pcm = _pcm([100, 100], [1, 1], 1024.0)
    out = _infeasible_breakdown(
        pcm, [1, 1], [1, 1], [200, 200], {"p": [0.0, 0.0]}, cards
    )
    assert "group=D r0 nvml1 NVIDIA GeForce RTX 5090 budget=200" in out
    assert "group=D r1 nvml0 NVIDIA GeForce RTX 3080 budget=200" in out


def test_without_cards_it_still_names_group_and_rank():
    pcm = _pcm([100], [1], 1024.0)
    out = _infeasible_breakdown(pcm, [1], [1], [200], {"p": [0.0]})
    assert "group=D r0 budget=200" in out
