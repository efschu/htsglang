"""#62 (21.09.): das Kapazitaetsmodell muss wissen, dass die Experten im
Host-Store liegen.

fnFL2v80 verweigerte W64 mit weights=89398 MiB auf der 5090 gegen ein Budget
von 28240 -- das ganze Routed-Expert-Gebirge als kartenresident gepreist,
waehrend der Lauf es aus /mnt/nf-experts bedient. Ohne Residenzangabe bleibt
das Modell byte-identisch zu vorher; mit Residenzangabe zieht es genau den
Teil ab, den die Karte nicht haelt.
"""

import math
import types

import pytest

from sglang.srt.weg2.launcher import d_moe_residency


class _Fam:
    def __init__(self, shard, byts):
        self.shard, self.bytes, self.params = shard, byts, 1
        self.bytes_per_param = 0.5


def _model(tp=3, experts=512, offloadable=(90.0, 40.0, 40.0)):
    """Ein PerfCostModel-Stellvertreter mit genau den Naehten, die der neue
    Term anfasst -- kein Checkpoint noetig."""
    from sglang.srt.uneven_perf import PerfCostModel

    m = PerfCostModel.__new__(PerfCostModel)
    m.tp_size = tp
    m.num_experts = experts
    m.families = {"mlp": _Fam("mlp", 1.0)}
    m._shard_fractions = lambda shard, vec: [
        v / sum(vec) for v in vec
    ]
    m.per_rank_offloadable_weight_bytes = lambda vec: list(offloadable)
    m.moe_resident_fraction = None
    m.moe_scratch_slots = None
    return m


def test_without_a_residency_nothing_is_subtracted():
    m = _model()
    assert m.per_rank_offloaded_weight_bytes([1, 1, 1]) == [0.0, 0.0, 0.0]


def test_the_fraction_leaves_the_rest_in_the_host_store():
    m = _model(experts=300, offloadable=(90.0, 90.0, 90.0))
    m.moe_resident_fraction = [0.5, 0.0, 1.0]
    out = m.per_rank_offloaded_weight_bytes([1, 1, 1])
    # 100 Experten je Rang: 50 resident -> die Haelfte im Store; 0 -> alles;
    # 1.0 -> nichts
    assert out[0] == pytest.approx(45.0)
    assert out[1] == pytest.approx(90.0)
    assert out[2] == pytest.approx(0.0)


def test_scratch_rows_count_as_resident_experts():
    m = _model(experts=300, offloadable=(90.0, 90.0, 90.0))
    m.moe_resident_fraction = [0.0, 0.0, 0.0]
    m.moe_scratch_slots = [60, 0, 100]
    out = m.per_rank_offloaded_weight_bytes([1, 1, 1])
    assert out[0] == pytest.approx(90.0 * (1 - 60 / 100))
    assert out[1] == pytest.approx(90.0)
    assert out[2] == pytest.approx(0.0)   # 100 Zeilen decken alle 100 Experten


def test_a_dense_checkpoint_offloads_nothing():
    m = _model(experts=0)
    m.moe_resident_fraction = [0.0, 0.0, 0.0]
    assert m.per_rank_offloaded_weight_bytes([1, 1, 1]) == [0.0, 0.0, 0.0]


def test_predict_capacity_reads_the_same_term():
    import inspect

    from sglang.srt.uneven_perf import PerfCostModel

    src = inspect.getsource(PerfCostModel.predict_capacity)
    assert "offloaded = self.per_rank_offloaded_weight_bytes(mlp_vector)" in src


# -- die Quelle der Zahlen: die --env-d-Zeichenkette des Arms, nie os.environ --

def test_the_residency_is_read_from_the_arms_env_d_string():
    frac, scratch = d_moe_residency(
        "SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.006,0.12,0.12;"
        "SGLANG_MOE_SCRATCH_SLOTS=60,48,48;SGLANG_UNEVEN_MOE_EXPERT_SHARD=1",
        3,
    )
    assert frac == [0.006, 0.12, 0.12]
    assert scratch == [60, 48, 48]


def test_one_entry_is_broadcast_and_a_missing_key_is_none():
    frac, scratch = d_moe_residency("SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.25", 3)
    assert frac == [0.25, 0.25, 0.25]
    assert scratch is None
    assert d_moe_residency("", 3) == (None, None)


def test_a_mis_sized_vector_is_ignored_not_guessed():
    frac, _ = d_moe_residency("SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.1,0.2", 3)
    assert frac is None
    frac, _ = d_moe_residency("SGLANG_MOE_RESIDENT_EXPERT_FRACTION=nonsense", 3)
    assert frac is None


def test_the_decision_hands_env_d_to_the_rows():
    import inspect

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher.d_tp_ratio_decision)
    assert "_moe_frac, _moe_scratch = d_moe_residency(env_d, len(budgets))" in src
    assert "moe_resident_fraction=_moe_frac" in src
    # und beide Aufrufstellen im main reichen die Zeichenkette wirklich durch
    main_src = inspect.getsource(launcher)
    assert main_src.count('d_bs, getattr(ns, "env_d", "") or "",') == 2


# -- die Asymmetrie: der Abzug oeffnet das Gate, waehlt aber keinen Vektor ---

def test_the_offload_opens_the_gate_but_leaves_p_and_the_vector_alone():
    """fnFL2v84 hat den Preis bezahlt: der freigewordene Platz verschob den
    DCP-Token-Vektor auf attn [10,7,7], die KV-Zelle der 5090 wuchs von 0,13
    auf 1,12 GB, der Pool FIEL von 262144 auf 196224 Token, und die Karte ging
    beim Graph-Capture mit 0,01 GiB frei OOM. Die Richtung des Abzugs ist
    bewiesen, seine GROESSE nicht (~2x gegen den Zensus) -- also darf er
    verweigern lassen, aber nicht waehlen."""
    import inspect

    from sglang.srt.uneven_perf import PerfCostModel

    src = inspect.getsource(PerfCostModel.predict_capacity)
    # p wird NICHT gekuerzt
    assert "weights = [w - o for w, o in zip(weights, offloaded)]" not in src
    # das Gate schon
    assert "gate_free = [f + o for f, o in zip(free_bytes, offloaded)]" in src
    assert "feasible = all(x >= _PREDICT_MIN_RANK_TOKENS for x in gate_p)" in src
    # und ohne Offload ist gate_p buchstaeblich p
    assert "gate_p = p" in src
