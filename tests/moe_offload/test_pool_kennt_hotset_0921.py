"""#93: der Device-Pool muss wissen, WELCHE Experten resident sind.

fnFL2w22: "IndexError: index 370 is out of bounds for dimension 0 with
size 32" -- size 32 ist der SCRATCH-Pool. Seit #92 sind die Residenten die
Hotset-Ids (u.a. 320..369), der Planner nahm aber weiter
`resident == [0, R) at slot==id` an (expert_offload.py:524) und schob
einen residenten Experten in den Scratch.

Der Planner hat den richtigen Zweig ("Hot residency: resident == frozen id
set at its assigned slot") -- er bekam die Menge nur nie.
"""

from sglang.srt.layers.moe.expert_offload import ExpertResidencyPlanner


def test_statisch_ohne_ids_bleibt_wie_bisher():
    p = ExpertResidencyPlanner(num_local_experts=512, resident_count=188, scratch=32)
    hot, cold = p.split_needed(list(range(180, 200)))
    assert hot == list(range(180, 188)) and cold == list(range(188, 200))


def test_mit_hotset_zaehlen_die_ids_nicht_die_ersten_R():
    """Id 370 ist resident, obwohl sie weit ueber resident_count liegt --
    genau der Fall, an dem w22 starb."""
    ids = frozenset(list(range(0, 92)) + list(range(183, 229)) + list(range(320, 370)))
    p = ExpertResidencyPlanner(
        num_local_experts=512, resident_count=188, scratch=32, resident_ids=ids
    )
    hot, cold = p.split_needed([5, 200, 250, 369, 370, 400])
    assert 369 in hot, "369 steht im Hotset und ist resident"
    assert 5 in hot and 200 in hot, "200 liegt in 183..228, also resident"
    assert 250 in cold and 370 in cold and 400 in cold


def test_kein_residenter_landet_im_scratch():
    """Die Eigenschaft, die der IndexError verletzt hat: was resident ist,
    geht NIE in den Scratch -- egal wie gross seine Id ist."""
    ids = frozenset({0, 91, 183, 369})
    p = ExpertResidencyPlanner(
        num_local_experts=512, resident_count=4, scratch=8, resident_ids=ids
    )
    hot, cold = p.split_needed(sorted(ids) + [1, 2, 500])
    assert set(hot) == ids
    assert not (set(cold) & ids)
