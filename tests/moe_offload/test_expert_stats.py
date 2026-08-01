# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the #390 router-distribution / hit-rate instrument.

No CUDA, no model, no server. Feeds synthetic routing decisions -- the same
``[T][k]`` host-side lists the offload path produces with ``topk_ids.tolist()``
-- through the counter class and round-trips the JSON dump.

Covered:
  * activation histogram and both hit-rate grains against a static ``[0, R)``
    resident set and against a frozen hot set;
  * ``-1`` routing padding is ignored;
  * peakedness on a uniform vs a collapsed router;
  * the opt-in gate: no env, no counters, no collector;
  * dump -> file -> ``json.load`` round-trip with the totals preserved;
  * ``ResidencyStats``-shaped objects surface in the dump.

Run:  python -m pytest tests/moe_offload/test_expert_stats.py -q
  or: python tests/moe_offload/test_expert_stats.py
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "python"),
)

from sglang.srt.layers.moe.expert_stats import (  # noqa: E402
    DEFAULT_STATS_PATH,
    ExpertStatsCollector,
    LayerExpertStats,
    get_collector,
    maybe_layer_stats,
    peakedness,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_collector():
    reset_for_tests()
    for name in (
        "SGLANG_EXPERT_STATS",
        "SGLANG_EXPERT_STATS_PATH",
        "SGLANG_EXPERT_STATS_INTERVAL_SEC",
    ):
        os.environ.pop(name, None)
    yield
    reset_for_tests()
    for name in (
        "SGLANG_EXPERT_STATS",
        "SGLANG_EXPERT_STATS_PATH",
        "SGLANG_EXPERT_STATS_INTERVAL_SEC",
    ):
        os.environ.pop(name, None)


# --------------------------------------------------------------------------- #
# counters
# --------------------------------------------------------------------------- #
def test_static_residency_hits_and_misses():
    """R=4 of E=8: experts 0..3 resident, 4..7 stream from the host tier."""
    stats = LayerExpertStats(layer_id=3, num_experts=8, resident_count=4)
    rows = [
        [0, 5],  # 1 hit, 1 miss
        [1, 5],  # 1 hit, 1 miss (expert 5 unique-counted once for the forward)
        [6, 7],  # 2 misses
    ]
    stats.record(rows)

    assert stats.forwards == 1
    assert stats.tokens == 3
    assert stats.activations == 6
    assert stats.hit_activations == 2
    assert stats.miss_activations == 4
    assert stats.hit_rate == pytest.approx(2 / 6)
    # Unique grain: {0,1} resident, {5,6,7} spilled -> what the fetch pays for.
    assert (stats.unique_hits, stats.unique_misses) == (2, 3)
    assert stats.unique_hit_rate == pytest.approx(2 / 5)
    assert stats.expert_activations == [1, 1, 0, 0, 0, 2, 1, 1]


def test_frozen_hot_set_residency():
    """With a frozen hot set the resident ids are arbitrary, not ``[0, R)``."""
    stats = LayerExpertStats(layer_id=0, num_experts=8, resident_count=2)
    stats.record([[7, 0], [7, 1]], resident_ids=frozenset({6, 7}), resident_count=2)
    # 7 is resident twice; 0 and 1 are below R but NOT in the hot set -> misses.
    assert stats.hit_activations == 2
    assert stats.miss_activations == 2
    assert (stats.unique_hits, stats.unique_misses) == (1, 2)


def test_padding_is_ignored():
    stats = LayerExpertStats(layer_id=1, num_experts=4, resident_count=2)
    stats.record([[0, -1], [-1, -1]])
    assert stats.activations == 1
    assert stats.hit_activations == 1
    assert stats.tokens == 2
    assert stats.forwards == 1


def test_all_padding_forward_still_counts_as_a_forward():
    stats = LayerExpertStats(layer_id=1, num_experts=4, resident_count=2)
    stats.record([[-1, -1]])
    assert (stats.forwards, stats.tokens, stats.activations) == (1, 1, 0)
    assert stats.hit_rate == 1.0


def test_accumulates_across_forwards():
    stats = LayerExpertStats(layer_id=2, num_experts=4, resident_count=1)
    for _ in range(5):
        stats.record([[0, 3]])
    assert stats.forwards == 5
    assert stats.hit_activations == 5
    assert stats.miss_activations == 5
    assert stats.expert_activations == [5, 0, 0, 5]
    assert stats.snapshot()["mean_unique_experts_per_forward"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# peakedness
# --------------------------------------------------------------------------- #
def test_peakedness_uniform_vs_collapsed():
    flat = peakedness([10] * 16)
    spiky = peakedness([160] + [0] * 15)
    assert flat["normalized_entropy"] == pytest.approx(1.0)
    assert flat["experts_used"] == 16
    assert flat["top1_share"] == pytest.approx(1 / 16)
    assert spiky["normalized_entropy"] == pytest.approx(0.0)
    assert spiky["experts_used"] == 1
    assert spiky["top1_share"] == pytest.approx(1.0)
    # The question the residency ladder actually asks: what would R catch?
    assert flat["top8_share"] == pytest.approx(0.5)


def test_peakedness_of_empty_histogram():
    empty = peakedness([0, 0, 0])
    assert empty["experts_used"] == 0
    assert empty["top1_share"] == 0.0


# --------------------------------------------------------------------------- #
# opt-in gate
# --------------------------------------------------------------------------- #
def test_disabled_by_default():
    assert get_collector() is None
    assert maybe_layer_stats(layer_id=0, num_experts=8, resident_count=2) is None


def test_enabled_returns_shared_per_layer_counter():
    os.environ["SGLANG_EXPERT_STATS"] = "1"
    first = maybe_layer_stats(layer_id=5, num_experts=8, resident_count=2)
    second = maybe_layer_stats(layer_id=5, num_experts=8, resident_count=2)
    assert first is not None and first is second
    collector = get_collector()
    assert collector is not None
    assert collector.output_path().startswith(DEFAULT_STATS_PATH)


def test_path_override():
    os.environ["SGLANG_EXPERT_STATS"] = "1"
    os.environ["SGLANG_EXPERT_STATS_PATH"] = "/tmp/somewhere/else"
    collector = get_collector(rank_tag="tp1ep0")
    assert collector is not None
    assert collector.output_path() == "/tmp/somewhere/else.tp1ep0.json"


# --------------------------------------------------------------------------- #
# JSON round-trip
# --------------------------------------------------------------------------- #
class _FakeResidencyStats:
    """Shaped like ``ResidencyStats`` without importing the CUDA-side module."""

    fetches = 7
    hits = 11
    misses = 7
    evictions = 0
    forwards = 3
    overflow_forwards = 1
    waves = 4
    h2d_bytes = 1234567


def test_dump_round_trip(tmp_path):
    collector = ExpertStatsCollector(
        path=str(tmp_path / "stats"), rank_tag="tp0ep0", interval_sec=0.0
    )
    layer0 = collector.layer(layer_id=0, num_experts=8, resident_count=4)
    layer1 = collector.layer(layer_id=1, num_experts=8, resident_count=4)
    layer0.residency = _FakeResidencyStats()
    layer0.record([[0, 4], [1, 4]])  # 2 hits, 2 misses
    layer1.record([[5, 6]])  # 0 hits, 2 misses

    target = collector.dump(reason="smoke")
    assert target == str(tmp_path / "stats.tp0ep0.json")

    with open(target) as handle:
        payload = json.load(handle)

    assert payload["schema"] == "sglang.expert_stats/1"
    assert payload["reason"] == "smoke"
    assert payload["rank_tag"] == "tp0ep0"
    totals = payload["totals"]
    assert totals["layers"] == 2
    assert totals["hit_activations"] == 2
    assert totals["miss_activations"] == 4
    assert totals["hit_rate"] == pytest.approx(2 / 6)

    entry0, entry1 = payload["layers"]
    assert entry0["layer_id"] == 0
    assert entry0["expert_activations"] == [1, 1, 0, 0, 2, 0, 0, 0]
    assert entry0["top_experts"][0] == [4, 2]
    assert entry0["residency"]["h2d_bytes"] == 1234567
    assert entry0["residency"]["fetches"] == 7
    assert "residency" not in entry1
    assert entry1["hit_rate"] == 0.0
    assert entry0["peakedness"]["experts_used"] == 3
    # No temp file left behind by the atomic replace.
    assert not os.path.exists(target + ".tmp")


def test_dump_creates_missing_parent_directory(tmp_path):
    collector = ExpertStatsCollector(
        path=str(tmp_path / "deep" / "nested" / "stats"), rank_tag="tp0ep0"
    )
    collector.layer(layer_id=0, num_experts=4, resident_count=2).record([[0, 3]])
    target = collector.dump(reason="mkdir")
    assert target is not None and os.path.exists(target)


def test_dump_failure_is_not_fatal():
    """An unwritable path must not take down a process that only asked for a
    measurement."""
    collector = ExpertStatsCollector(path="/proc/cannot/write/here", rank_tag="tp0ep0")
    collector.layer(layer_id=0, num_experts=4, resident_count=2).record([[0, 3]])
    assert collector.dump(reason="unwritable") is None


def test_periodic_dump_off_by_default(tmp_path):
    collector = ExpertStatsCollector(path=str(tmp_path / "s"), rank_tag="r")
    assert collector.maybe_dump_periodic() is None
    assert not os.path.exists(collector.output_path())


def test_periodic_dump_fires_when_interval_elapsed(tmp_path):
    collector = ExpertStatsCollector(
        path=str(tmp_path / "s"), rank_tag="r", interval_sec=60.0
    )
    collector.layer(layer_id=0, num_experts=4, resident_count=2).record([[0, 3]])
    assert collector.maybe_dump_periodic() is None  # deadline not reached
    collector._next_dump = 0.0  # pretend 60 s went by
    assert collector.maybe_dump_periodic() == collector.output_path()
    assert collector.maybe_dump_periodic() is None  # deadline re-armed


# --------------------------------------------------------------------------- #
# signal-driven dump (subprocess: installing handlers in the pytest process
# would take over its own SIGTERM)
# --------------------------------------------------------------------------- #
_SIGNAL_CHILD = """
import json, os, signal, sys
sys.path.insert(0, {python_dir!r})
os.environ["SGLANG_EXPERT_STATS"] = "1"
os.environ["SGLANG_EXPERT_STATS_PATH"] = {prefix!r}
from sglang.srt.layers.moe.expert_stats import get_collector, maybe_layer_stats

stats = maybe_layer_stats(layer_id=0, num_experts=8, resident_count=4)
stats.record([[0, 4], [1, 5]])
os.kill(os.getpid(), signal.SIGUSR2)          # dump and CONTINUE
with open(get_collector().output_path()) as handle:
    print(json.load(handle)["reason"])
stats.record([[2, 6]])                        # process is still alive
print("alive")
"""


def test_sigusr2_dumps_and_process_continues(tmp_path):
    import subprocess

    prefix = str(tmp_path / "sig")
    script = _SIGNAL_CHILD.format(
        python_dir=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "python")
        ),
        prefix=prefix,
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["sigusr2", "alive"]

    # The exiting process also dumps via atexit, overwriting with the later
    # state -- so the file on disk now carries the second record too.
    with open(f"{prefix}.rank0.json") as handle:
        payload = json.load(handle)
    assert payload["reason"] == "atexit"
    assert payload["totals"]["activations"] == 6  # 2 + 1 tokens x top-2


# --------------------------------------------------------------------------- #
# wiring into the real offload cache (CPU-constructible; install() is not called)
# --------------------------------------------------------------------------- #
class _FakeMoELayer:
    layer_id = 9
    num_local_experts = 32
    moe_tp_rank = 1
    moe_ep_rank = 0


def test_offload_cache_wires_the_counter_when_enabled():
    from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

    os.environ["SGLANG_EXPERT_STATS"] = "1"
    cache = MoEExpertOffloadCache(_FakeMoELayer(), fraction=0.25)
    assert cache._router_stats is not None
    assert cache._stats_collector is not None
    assert cache._router_stats.residency is cache.planner.stats
    assert get_collector().output_path().endswith(".tp1ep0.json")

    # Exactly the call run_waves makes, with the planner's own residency view.
    cache._router_stats.record(
        [[0, 31], [1, 31]], cache.planner.resident_ids, cache.resident_count
    )
    assert cache.planner.resident_ids is None  # static [0, R) residency
    assert cache._router_stats.hit_activations == 2  # experts 0 and 1 < R
    assert cache._router_stats.miss_activations == 2  # expert 31 twice
    assert cache._router_stats.snapshot()["residency"]["fetches"] == 0


def test_offload_cache_has_no_counter_when_disabled():
    from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

    cache = MoEExpertOffloadCache(_FakeMoELayer(), fraction=0.25)
    assert cache._router_stats is None
    assert cache._stats_collector is None
    assert get_collector() is None


# --------------------------------------------------------------------------- #
# end-to-end smoke: synthetic routing -> counters -> JSON on disk
# --------------------------------------------------------------------------- #
def _synthetic_smoke(out_dir: str) -> dict:
    """The whole instrument driven the way the offload path drives it."""
    os.environ["SGLANG_EXPERT_STATS"] = "1"
    os.environ["SGLANG_EXPERT_STATS_PATH"] = os.path.join(out_dir, "expert_stats")
    reset_for_tests()

    num_experts, top_k, resident = 16, 4, 6
    layers = [
        maybe_layer_stats(
            layer_id=lid, num_experts=num_experts, resident_count=resident
        )
        for lid in range(3)
    ]
    assert all(layer is not None for layer in layers)

    # A deliberately skewed router: expert (t % 4) is hot (and resident),
    # the rest of each token's top-k walks the cold tail.
    for step in range(8):
        rows = [
            [t % 4] + [(t * 3 + step + j) % num_experts for j in range(1, top_k)]
            for t in range(32)
        ]
        for layer in layers:
            layer.record(rows)

    collector = get_collector()
    path = collector.dump(reason="smoke")
    with open(path) as handle:
        return json.load(handle)


def test_end_to_end_smoke(tmp_path):
    payload = _synthetic_smoke(str(tmp_path))
    assert payload["totals"]["layers"] == 3
    assert payload["totals"]["forwards"] == 24
    assert payload["totals"]["tokens"] == 3 * 8 * 32
    assert payload["totals"]["activations"] == 3 * 8 * 32 * 4
    assert 0.0 < payload["totals"]["hit_rate"] < 1.0
    for entry in payload["layers"]:
        assert sum(entry["expert_activations"]) == entry["activations"]
        assert 0.0 < entry["peakedness"]["normalized_entropy"] <= 1.0


if __name__ == "__main__":  # hermetic smoke, no pytest required
    with tempfile.TemporaryDirectory() as out_dir:
        result = _synthetic_smoke(out_dir)
        totals = result["totals"]
        print(f"schema           : {result['schema']}")
        print(f"rank_tag         : {result['rank_tag']}")
        print(
            "totals           : "
            f"layers={totals['layers']} forwards={totals['forwards']} "
            f"tokens={totals['tokens']} activations={totals['activations']}"
        )
        print(
            "hit rate         : "
            f"activation={totals['hit_rate']:.4f} "
            f"unique={totals['unique_hit_rate']:.4f}"
        )
        for layer_entry in result["layers"]:
            peak = layer_entry["peakedness"]
            print(
                f"layer {layer_entry['layer_id']:<2}         : "
                f"hit={layer_entry['hit_rate']:.4f} "
                f"norm_entropy={peak['normalized_entropy']:.4f} "
                f"top8_share={peak['top8_share']:.4f} "
                f"top_experts={layer_entry['top_experts'][:3]}"
            )
        print("json round-trip  : OK")
