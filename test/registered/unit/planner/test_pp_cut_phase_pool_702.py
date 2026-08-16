"""#702 revision 3: the KV pool is priced PER PHASE, and PP uses the MIN rule.

Arm B `[42,11,11]` was armed on this solver's advice and OOM'd on rank0 at KV
pool commit, twice. The defect was mine and it was a phase confusion:

* Revision 2 computed the world pool as the SUM of per-rank token capacities
  with a free-proportional vector. That is the **TP-phase** rule -- under
  tensor parallelism the model is width-sharded, every rank holds a slice of
  all 64 layers, per-token cost is uniform, tokens are sharded across ranks,
  so the pool is a sum and the vector CAN relieve a tight rank.
* Under **PP prefill the pool is layer-sharded**: every rank stores KV for ALL
  tokens, for its OWN layers. So

      pool = min_i( free_i / (layers_i * kv_cost_per_token_per_layer) )

  and the token vector does not enter at all.

The metal proof that the vector cannot relieve the deep rank: rank0's vector
share was cut 4.3x (13 -> 3 of 64) between the two attempts and rank0's memory
moved by *zero* -- identical `reserved 30.60 GiB` both times.

So revision 2's headline result, "world pool conservation is exact", was an
artifact of applying the wrong phase's rule. Under the min-rule the pool
COLLAPSES as the cut deepens, because the deepest rank's capacity is divided by
its own layer count. Prefill speed and pool capacity are in direct opposition
along the cut axis, and revision 2 priced only the first.

`stage_kv_capacities` in this same module already implemented the min-rule
machinery. The co-solve did not use it.

Calibration is F4-r4's census, reproduced here to ~0.02%: the model is ~14%
pessimistic in absolute terms (it puts the incumbent at 384,203 where the live
boot solves 434,878), so the RANKING is load-bearing and the absolutes are not.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest

from sglang.srt.planner.pp_cut import (
    PhasePoolModel,
    pp_phase_pool,
    stage_pp_capacities,
    tp_phase_pool,
)

INCUMBENT = (28, 20, 16)
ARM_B = (42, 11, 11)
LIVE_INCUMBENT_POOL = 434_878.0

# Census-calibrated (FINDING_702_weight_term.md): the hybrid checkpoint measures
# 374.2 MiB/layer full-attention and 476.2 linear, a flat equivalent of 450.7 --
# NOT the 724.3 revision 2 used (+61%). Real, but not the cause of the failure.
CENSUS_WEIGHT_MIB_PER_LAYER = 450.7


def _model():
    return PhasePoolModel(
        free_mib=(26721.2, 16051.3, 13984.3),
        weight_mib_per_layer=CENSUS_WEIGHT_MIB_PER_LAYER,
        kv_mib_per_token_per_layer=450.7 / 492129.0,
    )


def test_pp_pool_is_the_min_not_the_sum():
    """The whole defect in one assertion."""
    m = _model()
    caps = stage_pp_capacities(INCUMBENT, m)
    assert pp_phase_pool(INCUMBENT, m) == pytest.approx(min(caps), rel=1e-12)
    # And it is emphatically NOT the sum revision 2 reported.
    assert pp_phase_pool(INCUMBENT, m) < sum(caps) / 2


def test_backtest_reproduces_arm_b_failure():
    """CAN-FAIL PROOF: the solver must now predict the boot that OOM'd.

    A model that cannot reproduce the observed failure has not been fixed, it
    has been re-parameterized.
    """
    m = _model()
    bound = pp_phase_pool(ARM_B, m)
    assert bound == pytest.approx(201_298.0, rel=0.01)
    # rank0 is the binding rank, and it is bound by its own layer count.
    caps = stage_pp_capacities(ARM_B, m)
    assert caps[0] == min(caps)
    # Against the incumbent's live pin the arm cannot hold the context: this is
    # the corridor violation that OOM'd on metal, not a near miss.
    assert bound < LIVE_INCUMBENT_POOL / 2


def test_sanity_anchor_32_16_16():
    """F4-r4's recommended calibration arm: ~3% under the incumbent."""
    m = _model()
    bound = pp_phase_pool((32, 16, 16), m)
    assert bound == pytest.approx(419_734.0, rel=1e-3)
    shortfall = 1.0 - bound / LIVE_INCUMBENT_POOL
    assert 0.02 < shortfall < 0.05


def test_incumbent_matches_the_censused_table():
    m = _model()
    caps = stage_pp_capacities(INCUMBENT, m)
    assert caps[0] == pytest.approx(550_000.0, rel=5e-3)
    assert caps[1] == pytest.approx(384_203.0, rel=5e-3)
    assert caps[2] == pytest.approx(462_223.0, rel=5e-3)
    assert pp_phase_pool(INCUMBENT, m) == pytest.approx(384_203.0, rel=5e-3)


def test_the_pool_peaks_and_then_collapses():
    """The real shape: NOT conservation (rev 2) and NOT monotone decline either.

    The binding rank switches. At the incumbent rank1 binds -- it carries 20
    layers -- so moving layers OFF rank1 RAISES the pool. Past the crossover
    rank0 binds and the pool collapses as its layer count grows. The incumbent
    is therefore not pool-optimal, which no previous revision noticed.
    """
    m = _model()
    inc = pp_phase_pool(INCUMBENT, m)
    peak = pp_phase_pool((30, 18, 16), m)
    assert peak > inc  # the incumbent is beatable on pool
    assert peak == pytest.approx(462_231.0, rel=5e-3)
    # Past the crossover it is monotone collapse.
    tail = [
        pp_phase_pool(c, m)
        for c in [(32, 16, 16), (34, 15, 15), (38, 13, 13), (42, 11, 11), (46, 9, 9)]
    ]
    assert tail == sorted(tail, reverse=True)
    # [46,9,9] looked like the dominating arm on speed; it is the WORST on pool.
    assert tail[-1] == pytest.approx(141_000.0, rel=0.02)


def test_the_binding_rank_switches_across_the_cut():
    """Which rank binds is the whole structure, so pin it."""
    m = _model()
    assert (
        stage_pp_capacities(INCUMBENT, m).index(min(stage_pp_capacities(INCUMBENT, m)))
        == 1
    )
    caps = stage_pp_capacities((42, 11, 11), m)
    assert caps.index(min(caps)) == 0


def test_the_vector_cannot_relieve_the_deep_rank():
    """Encodes the metal fact as a refusal, not a comment.

    Cutting rank0's vector share 4.3x moved its memory by zero. Under PP the
    token vector is not an input to the pool bound, so passing one is a
    category error and is refused loudly rather than silently ignored.
    """
    m = _model()
    with pytest.raises(TypeError):
        pp_phase_pool(ARM_B, m, kv_token_vector=(3, 30, 31))  # noqa


def test_tp_phase_pool_is_a_separate_column_and_cut_independent():
    """Under TP the weights are width-sharded, so the PP cut does not enter."""
    m = _model()
    a = tp_phase_pool(INCUMBENT, m)
    b = tp_phase_pool(ARM_B, m)
    assert a == pytest.approx(b, rel=1e-12)
    # And it is a SUM, so it exceeds any single rank's share.
    assert a > pp_phase_pool(INCUMBENT, m)


def test_census_weight_term_changes_the_capacities():
    """450.7 vs the 724.3 revision 2 used: real, and not the cause."""
    census = _model()
    inflated = PhasePoolModel(
        free_mib=census.free_mib,
        weight_mib_per_layer=724.3,
        kv_mib_per_token_per_layer=census.kv_mib_per_token_per_layer,
    )
    # The inflated term makes every rank look poorer.
    assert pp_phase_pool(INCUMBENT, inflated) < pp_phase_pool(INCUMBENT, census)
    # And it encodes the two competing diagnoses of the SAME failure, which is
    # why the first write-up blamed the weight term and had to be retracted:
    #  * at 724.3, arm B looks like a WEIGHT OVERFLOW -- rank0 cannot even hold
    #    42 layers of weights, so the cut never gets as far as a pool number;
    with pytest.raises(ValueError, match="rank0"):
        pp_phase_pool(ARM_B, inflated)
    #  * at the census 450.7 the weights fit comfortably and the arm still
    #    fails, on the pool min-rule. That is the actual cause.
    assert pp_phase_pool(ARM_B, census) == pytest.approx(201_298.0, rel=0.01)
    assert pp_phase_pool(ARM_B, census) < pp_phase_pool(INCUMBENT, census)


def test_a_rank_whose_weights_exceed_its_card_is_refused():
    m = _model()
    with pytest.raises(ValueError, match="rank0"):
        pp_phase_pool((60, 2, 2), m)
