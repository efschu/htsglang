# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The reconciliation's MAPPING, fixed against boot 1464299 (#605).

RED-FIRST. Each test here encodes a defect the first reconcile run exposed:

* the load transient read one phase and measured 0 while the boot's true peak
  was 13392 MiB three marks later;
* the weights row SUMMED two weight-load episodes separated by a free, and was
  blind to the KV arena, producing 27800 MiB on a card holding 28436 MiB in
  total;
* GDN scratch and NCCL buffers had no boundary at all and read UNMEASURED
  without saying WHICH boundary was missing;
* terms the ledger REFUSED to price never appeared in the table, so a refusal
  and a term that does not exist looked identical.

The fixtures are the real marks of boot 1464299-1786612548, transcribed.
"""

import pytest

from sglang.srt.mem_ledger.engine import (
    TERM_GDN_SCRATCH,
    TERM_LOAD_TRANSIENT,
    TERM_NCCL_BUFFERS,
    TERM_WEIGHTS,
)
from sglang.srt.mem_ledger.reconcile import TERM_TO_POST, reconcile_card

MIB = 1 << 20


def _m(
    phase, *, draft=None, alloc=0, resv=0, arena=0, transient=0, non_torch=0, mono=0.0
):
    mark = {
        "phase": phase,
        "pid": 1464746,
        "rank": 0,
        "monotonic": float(mono),
        "allocated_bytes": alloc * MIB,
        "reserved_bytes": resv * MIB,
        "kv_arena_backed_bytes": arena * MIB,
        "allocator_transient_bytes": transient * MIB,
        "non_torch_bytes": non_torch * MIB,
        "nvml_self_bytes": 26364 * MIB,
        "nvml_carve_out_bytes": 518 * MIB,
    }
    if draft is not None:
        mark["extra"] = {"draft_worker": draft}
    return mark


#: The 5090's real timeline on boot 1464299, in order.
MARKS_5090 = [
    _m("process_start", mono=0),
    _m(
        "pre_weight_load",
        draft=False,
        alloc=0,
        resv=2,
        non_torch=886,
        transient=20,
        mono=1,
    ),
    _m("weights_loaded", draft=False, alloc=13674, resv=14004, non_torch=886, mono=2),
    _m(
        "kv_pool_sized",
        draft=False,
        alloc=21669,
        resv=21724,
        arena=6916,
        non_torch=886,
        mono=3,
    ),
    _m(
        "pre_weight_load",
        draft=True,
        alloc=8596,
        resv=8758,
        arena=6916,
        non_torch=896,
        transient=13392,
        mono=4,
    ),
    _m(
        "weights_loaded",
        draft=True,
        alloc=22446,
        resv=22556,
        arena=6916,
        non_torch=896,
        mono=5,
    ),
    _m(
        "pre_weight_load",
        draft=True,
        alloc=22293,
        resv=22460,
        arena=20610,
        non_torch=896,
        transient=128,
        mono=6,
    ),
    _m(
        "weights_loaded",
        draft=True,
        alloc=24367,
        resv=24506,
        arena=20610,
        non_torch=896,
        mono=7,
    ),
    _m("kv_pool_sized", draft=True, alloc=32494, resv=32616, arena=22690, mono=8),
    _m("capture_begin", draft=True, alloc=33221, resv=33324, arena=22690, mono=9),
    _m(
        "capture_end",
        draft=True,
        alloc=33344,
        resv=33496,
        arena=22690,
        transient=2,
        mono=10,
    ),
    _m("boot_complete", alloc=33505, resv=33876, arena=21130, mono=11),
    _m("first_forward", alloc=33528, resv=34150, arena=22690, mono=12),
]


def _ledger(terms, unbounded=()):
    return {
        "gpu_id": 0,
        "card": "GPU 0 (NVIDIA GeForce RTX 5090, NVML total 32607 MiB)",
        "ranks": [0],
        "demand_mib": 1656,
        "kv_pool_mib": 29927,
        "unbounded": list(unbounded),
        "terms": [{"name": n, "mib": v, "provenance": "modeled"} for n, v in terms],
    }


def _row(card, name):
    for c in card.comparisons:
        if c.term == name:
            return c
    raise AssertionError(f"no row for {name!r} in {[c.term for c in card.comparisons]}")


# ---------------------------------------------------------------------------
# 1. The load transient must span the boot's true peak
# ---------------------------------------------------------------------------


def test_load_transient_measures_the_boot_peak_not_one_phase():
    card = reconcile_card(_ledger([(TERM_LOAD_TRANSIENT, 70)]), MARKS_5090)
    row = _row(card, TERM_LOAD_TRANSIENT)
    # 13392 is the real peak, on the draft runner's pre_weight_load. The first
    # run read the target runner's weights_loaded and got 0.
    assert row.measured_mib == 13392, row.measured_mib
    assert row.error_mib == 70 - 13392


def test_load_transient_mapping_is_a_peak_kind():
    (kind, *_), _basis = TERM_TO_POST[TERM_LOAD_TRANSIENT]
    assert kind == "peak"


# ---------------------------------------------------------------------------
# 2. Weights: per episode, arena-aware, MAX and never the sum
# ---------------------------------------------------------------------------


def test_weights_are_measured_per_episode_and_never_summed():
    card = reconcile_card(_ledger([(TERM_WEIGHTS, 13850)]), MARKS_5090)
    row = _row(card, TERM_WEIGHTS)
    # Episodes: 13674 (PP layout), 13850 (TP layout), 2074 (NEXTN draft).
    # Each is freed before the next loads, so the card's weight demand is the
    # LARGEST, not the total. The first run reported 27800 = 14002 + 13798.
    assert row.measured_mib == 13850, row.measured_mib
    assert row.measured_mib != 27800
    assert "13674" in row.note and "2074" in row.note


def test_weights_measurement_excludes_the_kv_arena():
    """Episode 2 grows allocated by 13850 while the arena is untouched."""
    card = reconcile_card(_ledger([(TERM_WEIGHTS, 13850)]), MARKS_5090)
    assert _row(card, TERM_WEIGHTS).measured_mib == 13850
    # If the arena were counted, episode 3 would read 2074 + 13694 = 15768 and
    # become the max.
    assert _row(card, TERM_WEIGHTS).measured_mib < 15768


def test_weights_row_closes_within_one_percent_when_modelled_right():
    card = reconcile_card(_ledger([(TERM_WEIGHTS, 13850)]), MARKS_5090)
    row = _row(card, TERM_WEIGHTS)
    assert abs(row.error_mib) <= 0.01 * row.measured_mib


# ---------------------------------------------------------------------------
# 3. GDN scratch and NCCL buffers: a boundary exists, and its absence is NAMED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("term", [TERM_GDN_SCRATCH, TERM_NCCL_BUFFERS])
def test_missing_boundary_is_named_not_merely_unmeasured(term):
    card = reconcile_card(_ledger([(term, 20)]), MARKS_5090)
    row = _row(card, term)
    assert row.measured_mib is None
    (kind, *args), _ = TERM_TO_POST[term]
    for phase in args:
        assert phase in row.note, f"{phase!r} not named in {row.note!r}"


@pytest.mark.parametrize(
    "term,begin,end",
    [
        (TERM_GDN_SCRATCH, "gdn_scratch_begin", "gdn_scratch_end"),
        (TERM_NCCL_BUFFERS, "nccl_init_begin", "nccl_init_end"),
    ],
)
def test_boundary_measures_when_the_boot_records_it(term, begin, end):
    marks = list(MARKS_5090)
    marks.insert(4, _m(begin, draft=False, alloc=13674, resv=14004, mono=2.4))
    marks.insert(5, _m(end, draft=False, alloc=13674, resv=14104, mono=2.6))
    card = reconcile_card(_ledger([(term, 20)]), marks)
    assert _row(card, term).measured_mib == 100


# ---------------------------------------------------------------------------
# 4. A term the ledger REFUSED to price is a row, not an absence
# ---------------------------------------------------------------------------


def test_refused_terms_appear_as_rows_with_their_refusal():
    refusal = (
        "runtime activation + metadata on NVIDIA GeForce RTX 5090 (rank(s) 0): "
        "no phase footprint is calibrated"
    )
    card = reconcile_card(
        _ledger([(TERM_WEIGHTS, 13850)], unbounded=[refusal]), MARKS_5090
    )
    names = [c.term for c in card.comparisons]
    assert any("runtime activation" in n for n in names), names
    row = next(c for c in card.comparisons if "runtime activation" in c.term)
    assert row.modeled_mib is None
    assert "REFUSED" in row.note or "refus" in row.note.lower()


def test_render_lists_refusals_separately_from_unmeasured():
    refusal = "CUDA graph capture on X: no phase footprint is calibrated"
    card = reconcile_card(
        _ledger([(TERM_GDN_SCRATCH, 20)], unbounded=[refusal]), MARKS_5090
    )
    text = card.render()
    assert "REFUSED BY THE MODEL" in text
    assert "gdn_scratch_begin" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--color=no"]))
