"""Task #48 (19.09.): --rank-moe-ratio time -- the expert split by streaming
TIME with the VRAM-sized hot rows held fixed. Hermetic (CUDA_VISIBLE_DEVICES="")."""
import os

import pytest

from sglang.srt.layers.moe.expert_compute_placement import (
    COMPUTE_PLACEMENT_SYMBOLS,
    COMPUTE_PLACEMENT_TIME,
    NoComputeLever,
    miss_exponent_from_measurement,
    solve_time_equalised_expert_vector,
)

# fn7p 19.09.: base plan 312/104/96, hot rows 2+136 / 39+28 / 43+28, links
# 13.4/13.4/6.6 GB/s, miss rows 2988/461/180
HOT = [138, 67, 71]
LINK = [13.4, 13.4, 6.6]
BASE = [312, 104, 96]
MISSES = [2988, 461, 180]


def test_symbol_is_registered():
    assert COMPUTE_PLACEMENT_TIME in COMPUTE_PLACEMENT_SYMBOLS


def test_first_order_solve_equalises_cold_over_link():
    p = solve_time_equalised_expert_vector(512, HOT, LINK, 1.0, base_weights=BASE)
    assert sum(p.weights) == 512
    # gamma 1: cold_r = owned_r - hot_r proportional to the link -> the two x8
    # ranks carry the same cold mass, the x4 rank about half
    cold = [o - h for o, h in zip(p.weights, HOT)]
    assert cold[0] == cold[1]
    assert abs(cold[2] * 2 - cold[0]) <= 3
    assert p.weights == (233, 162, 117)
    shares = p.predicted_time_shares()
    assert max(shares) - min(shares) < 0.02
    assert p.clock_speedup() > 1.5


def test_calibrated_exponent_fits_the_measured_misses():
    g = miss_exponent_from_measurement(BASE, HOT, MISSES)
    assert 1.8 < g < 2.5
    p = solve_time_equalised_expert_vector(512, HOT, LINK, g, base_weights=BASE)
    assert sum(p.weights) == 512
    # the 5090 sheds experts, both 3080s take some; the x4 rank less than x8
    assert p.weights[0] < BASE[0]
    assert p.weights[1] > BASE[1] and p.weights[2] > BASE[2]
    assert p.weights[1] - BASE[1] > p.weights[2] - BASE[2]
    assert p.clock_speedup() > 2.0
    assert "time-equalised" in p.describe()


def test_exponent_defaults_to_one_without_information():
    assert miss_exponent_from_measurement([100, 100], [100, 100], [0, 0]) == 1.0
    assert miss_exponent_from_measurement([100], [10], [5]) == 1.0


def test_fully_resident_group_has_no_lever():
    with pytest.raises(NoComputeLever):
        solve_time_equalised_expert_vector(200, [100, 100], [1.0, 1.0])


def test_uniform_links_and_hot_rows_give_the_even_split():
    p = solve_time_equalised_expert_vector(300, [20, 20, 20], [1, 1, 1], 1.7)
    assert p.weights == (100, 100, 100)
