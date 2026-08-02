# SPDX-License-Identifier: Apache-2.0
"""Hermetic falsifier for the cold-tier FETCH PATH (#394 slice 2).

Slice 1 proved the storage: a peer either gets the exact bytes its manifest
promised, or a named exception. This file proves the ROUTING built on top --
who owns which cold expert, how a rank reaches a row it does not own, and what
happens when it cannot.

No CUDA, no driver, no model. ``/dev/shm`` is redirected to a tmp path, the
segments are real mmaps and the fetch runs through the real
``MoEExpertOffloadCache._fetch`` on CPU tensors (the cache's ``_stream is
None`` branch exists exactly so this chain is testable without a card). What
the GPU window adds is bandwidth, not a different code path.

The four properties, each with a committed CAN-FAIL arm:

  * **the owner map is rank-uniform.** Every rank computes it from the same
    (cold pool, weights) with no collective, so a plan digest taken on three
    simulated ranks must be ONE value. A branch inside a collective family
    that came from local state is the recurring hang shape in this fork; the
    digest is what turns "we believe it is uniform" into a pin.
  * **proportional assignment comes from a synthetic provenance fixture.** No
    rig constants (#434): the bandwidths are invented, and the ASSERTION is
    about the direction and the arithmetic, not about 6.4 vs 13.
  * **absent provenance refuses.** An assignment weighted by nothing is not an
    assignment.
  * **the read path is read-only and correct.** The bytes a peer fetches equal
    the bytes the owner wrote, and a write through the peer view dies rather
    than corrupting the owner (slice-1 SIGSEGV pattern, in a forked child).

Run:
  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \
    python -m pytest tests/moe_offload/test_cold_tier_fetch.py -q
"""

import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe import cold_tier_fetch as ctf  # noqa: E402
from sglang.srt.layers.moe import cold_tier_shm as cts  # noqa: E402
from sglang.srt.layers.moe import expert_offload as eo  # noqa: E402
from sglang.srt.layers.moe.cold_tier_fetch import (  # noqa: E402
    COLD_TIER_INSTANCE_ENV,
    ColdTierAssignment,
    ColdTierOwner,
    ColdTierResolver,
    ColdTierUnavailable,
    assign_cold_experts,
    cold_tier_enabled,
    instance_id,
    publish_cold_tier_instance,
)
from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    HOST_SHARD_SOURCE_EQUAL,
    HOST_SHARD_SOURCE_PROBE,
    ExpertResidencyPlanner,
    HostShardRatio,
    plan_load_time_staging,
    resolve_host_shard_ratio,
)

INSTANCE = "s2test0001"
WORLD = 3
CARDS = ("CARD-A", "CARD-B", "CARD-C")

#: SYNTHETIC provenance (#434). These are not this rig's numbers and nothing
#: below asserts a rig value; they exist so the ratio has a measured source and
#: an unmistakable direction: rank 1 is the weak link, 1 : 4 : 4.
SYNTH_H2D = {"CARD-A": 8.0, "CARD-B": 2.0, "CARD-C": 8.0}
WEAK_RANK = 1


@pytest.fixture(autouse=True)
def _tier(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_MOE_COLD_TIER_SHM_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_MOE_COLD_TIER_SHM", "1")
    monkeypatch.setenv(COLD_TIER_INSTANCE_ENV, INSTANCE)
    monkeypatch.delenv("SGLANG_MOE_HOST_SHARD_RATIO", raising=False)
    monkeypatch.delenv("SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE", raising=False)
    # Verify the redirection instead of trusting it. This file creates real
    # segments with real mmaps; the ONE path on which that is not hermetic is
    # the module default, ``/dev/shm``, which on a shared box is a live
    # resource other sessions are measuring against. A test added later
    # without this fixture, or a rename of the env var, must fail here rather
    # than in somebody else's tmpfs budget.
    assert cts.shm_dir() == str(tmp_path), (
        f"cold-tier tests would write {cts.shm_dir()!r}, not the tmp path. "
        "Hermetic tests never touch a shared-resource default."
    )
    ctf.reset_for_tests()
    yield
    ctf.reset_for_tests()
    assert not list(
        pathlib.Path("/dev/shm").glob("sgl-cold-*")
    ), "a cold-tier test leaked a segment into the real /dev/shm"


def _ratio(h2d=None):
    """A MEASURED ratio over the synthetic profile, through the real chain."""
    table = SYNTH_H2D if h2d is None else h2d
    return resolve_host_shard_ratio(
        WORLD, CARDS, link_gbps=lambda _u: None, probe_gbps=lambda u: table.get(u)
    )


def _write_segment(rank, layer_key, attr, expert_ids, row_shape, dtype=torch.uint8):
    """Stage one rank's cold rows into its segment and seal it.

    Rows are filled with a per-expert constant so a wrong row is visible as a
    value, not as a crash.
    """
    owner = ColdTierOwner(INSTANCE, rank, layer_key, CARDS[rank])
    pool = owner.allocate_spill_pool(attr, expert_ids, row_shape, dtype)
    for row, expert_id in enumerate(expert_ids):
        pool[row].fill_(expert_id)
    owner.seal()
    return owner, pool


# --------------------------------------------------------------------------
# 1. the plan: rank-uniform, proportional, and derived from provenance
# --------------------------------------------------------------------------


def test_the_owner_map_is_proportional_and_names_the_weak_rank():
    ratio = _ratio()
    assert ratio.source == HOST_SHARD_SOURCE_PROBE
    assert ratio.provenance == "measured"

    pool = tuple(range(18))
    plan = assign_cold_experts(pool, ratio, rank=0, world_size=WORLD)

    counts = [plan.owners.count(r) for r in range(WORLD)]
    assert sum(counts) == 18
    # 8 : 2 : 8 over 18 experts -> 8, 2, 8. The weak link owns the fewest.
    assert counts == [8, 2, 8]
    assert counts[WEAK_RANK] == min(counts)


def test_the_plan_is_identical_on_every_rank_without_a_collective():
    """Rank-uniformity as a digest, not as a belief.

    Any branch that feeds a collective must come from the shared plan. Three
    simulated ranks resolve the ratio and the owner map independently -- no
    message passes between them -- and the digest must be one value.
    """
    digests = {
        assign_cold_experts(
            tuple(range(18)), _ratio(), rank=r, world_size=WORLD
        ).digest()
        for r in range(WORLD)
    }

    assert len(digests) == 1, f"the plan is not rank-uniform: {digests}"


def test_the_rank_uniformity_pin_can_fail():
    """CAN-FAIL ARM.

    Planted defect: rank 2's ratio is weighted by a locally observed number
    instead of the group's -- the exact "decided from local state" shape. The
    digest must split.
    """
    local = dict(SYNTH_H2D)
    local["CARD-C"] = 3.0  # only rank 2 sees this
    digests = {
        assign_cold_experts(
            tuple(range(18)),
            _ratio(local if r == 2 else None),
            rank=r,
            world_size=WORLD,
        ).digest()
        for r in range(WORLD)
    }

    assert len(digests) == 2
    with pytest.raises(AssertionError):
        assert len(digests) == 1, "the plan is not rank-uniform"


def test_an_absent_provenance_cannot_weight_an_assignment():
    equal = HostShardRatio((1 / 3, 1 / 3, 1 / 3), HOST_SHARD_SOURCE_EQUAL)
    assert equal.provenance == "absent"

    with pytest.raises(ValueError, match="absent provenance"):
        assign_cold_experts((0, 1, 2), equal, rank=0, world_size=WORLD)


def test_the_absent_provenance_refusal_pin_can_fail():
    """CAN-FAIL ARM for the test above.

    Planted defect: the same equal, unmeasured split, relabelled with the probe
    source -- the "it now LIES about itself" shape. The gate reads the label,
    so a guess dressed as a measurement walks straight through and produces a
    perfectly ordinary-looking assignment.
    """
    lying = HostShardRatio((1 / 3, 1 / 3, 1 / 3), HOST_SHARD_SOURCE_PROBE)
    assert lying.provenance == "measured"

    plan = assign_cold_experts((0, 1, 2), lying, rank=0, world_size=WORLD)

    assert plan.owners == (0, 1, 2)
    with pytest.raises(AssertionError):
        assert plan.ratio_provenance == "absent"


def test_an_expert_outside_the_pool_is_named_not_guessed():
    plan = assign_cold_experts((4, 5, 6), _ratio(), rank=0, world_size=WORLD)

    with pytest.raises(ColdTierUnavailable, match="different pools"):
        plan.owner_of(99)


def test_the_digest_excludes_the_rank_so_it_cannot_pass_trivially():
    a = assign_cold_experts(tuple(range(9)), _ratio(), rank=0, world_size=WORLD)
    b = assign_cold_experts(tuple(range(9)), _ratio(), rank=2, world_size=WORLD)

    assert a.rank != b.rank
    assert a.digest() == b.digest()
    # ... but a different PLAN must move it.
    c = assign_cold_experts(tuple(range(10)), _ratio(), rank=0, world_size=WORLD)
    assert c.digest() != a.digest()


# --------------------------------------------------------------------------
# 2. manifest -> segment resolution
# --------------------------------------------------------------------------


def test_a_peer_resolves_a_delegated_expert_to_the_owners_row():
    _write_segment(
        rank=2, layer_key="L7", attr="w13_qweight", expert_ids=(10, 11), row_shape=(8,)
    )
    plan = ColdTierAssignment(
        rank=0,
        world_size=WORLD,
        cold_ids=(10, 11),
        owners=(2, 2),
        ratio_source=HOST_SHARD_SOURCE_PROBE,
        ratio_provenance="measured",
        weights=(0.4, 0.2, 0.4),
    )
    resolver = ColdTierResolver(INSTANCE, plan, "L7")

    row = resolver.row("w13_qweight", 11)

    assert row.shape == (8,)
    assert torch.equal(row, torch.full((8,), 11, dtype=torch.uint8))


def test_the_local_pool_is_not_reached_through_the_peer_path():
    plan = ColdTierAssignment(
        rank=0,
        world_size=WORLD,
        cold_ids=(10, 11),
        owners=(0, 2),
        ratio_source=HOST_SHARD_SOURCE_PROBE,
        ratio_provenance="measured",
        weights=(0.4, 0.2, 0.4),
    )
    resolver = ColdTierResolver(INSTANCE, plan, "L7")

    with pytest.raises(ColdTierUnavailable, match="owned by this rank"):
        resolver.row("w13_qweight", 10)


def test_a_missing_peer_layout_names_what_the_owner_actually_published():
    _write_segment(
        rank=2, layer_key="L7", attr="w13_qweight", expert_ids=(10,), row_shape=(4,)
    )
    plan = ColdTierAssignment(
        rank=0,
        world_size=WORLD,
        cold_ids=(10,),
        owners=(2,),
        ratio_source=HOST_SHARD_SOURCE_PROBE,
        ratio_provenance="measured",
        weights=(0.4, 0.2, 0.4),
    )
    resolver = ColdTierResolver(INSTANCE, plan, "L7")

    with pytest.raises(ColdTierUnavailable, match="no layout for"):
        resolver.row("w2", 10)


def test_a_manifest_that_never_arrives_expires_rather_than_hanging():
    plan = ColdTierAssignment(
        rank=0,
        world_size=WORLD,
        cold_ids=(10,),
        owners=(1,),
        ratio_source=HOST_SHARD_SOURCE_PROBE,
        ratio_provenance="measured",
        weights=(0.4, 0.2, 0.4),
    )
    resolver = ColdTierResolver(INSTANCE, plan, "L7", manifest_timeout_s=0.05)

    with pytest.raises(cts.ManifestUnavailable):
        resolver.row("w13_qweight", 10)


def test_the_owner_publishes_every_layer_it_has_sealed_so_far():
    """One manifest per rank, cumulative, atomic. A peer that reads it early
    sees a prefix of the layers, never a half-written file."""
    _write_segment(
        rank=1, layer_key="L0", attr="w13_qweight", expert_ids=(3,), row_shape=(4,)
    )
    _write_segment(
        rank=1, layer_key="L1", attr="w13_qweight", expert_ids=(4,), row_shape=(4,)
    )

    layouts = cts.read_peer_manifest(INSTANCE, 1)

    assert set(layouts) == {("L0", "w13_qweight"), ("L1", "w13_qweight")}


# --------------------------------------------------------------------------
# 3. the read path: zero copy, read-only, one copy of the bytes
# --------------------------------------------------------------------------


def test_the_owners_pool_is_the_segment_and_not_a_copy_of_it():
    """The whole point of sharing: writing a row through the owner's tensor is
    visible to a peer's view WITHOUT any flush, because there is one buffer."""
    owner, pool = _write_segment(
        rank=2, layer_key="L3", attr="w2", expert_ids=(5, 6), row_shape=(4,)
    )
    layouts = cts.read_peer_manifest(INSTANCE, 2)
    layout = layouts[("L3", "w2")]

    pool[1].fill_(200)  # after sealing, through the owner's own tensor

    assert torch.equal(
        cts.peer_row_tensor(layout, 6), torch.full((4,), 200, dtype=torch.uint8)
    )


def test_writing_through_a_peer_row_dies_rather_than_corrupting_the_owner():
    """The slice-1 SIGSEGV pattern, re-run against the ROUTED view.

    ``torch.frombuffer`` does not model a non-writable tensor and PyTorch warns
    that one "can write to the underlying buffer". The kernel disagrees, and
    the kernel is what enforces it -- so the proof has to be a real write in a
    forked child, not a type assertion.
    """
    if not hasattr(os, "fork"):  # pragma: no cover - platform guard
        pytest.skip("needs fork")
    _write_segment(rank=2, layer_key="L3", attr="w2", expert_ids=(5,), row_shape=(4,))
    plan = ColdTierAssignment(
        rank=0,
        world_size=WORLD,
        cold_ids=(5,),
        owners=(2,),
        ratio_source=HOST_SHARD_SOURCE_PROBE,
        ratio_provenance="measured",
        weights=(0.4, 0.2, 0.4),
    )
    resolver = ColdTierResolver(INSTANCE, plan, "L3")

    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns
        try:
            resolver.row("w2", 5).fill_(0xFF)
        except BaseException:
            os._exit(0)
        os._exit(1)  # the write SUCCEEDED: the mapping was not read-only
    _, status = os.waitpid(pid, 0)

    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == 11, (
        "a write through the peer view neither raised nor took SIGSEGV; the "
        "PROT_READ guarantee is gone"
    )


def test_the_read_only_pin_can_fail():
    """CAN-FAIL ARM for the test above.

    Planted defect: the same routed row, but taken from a WRITABLE mapping of
    the owner's own pool. The write lands, and the property must notice.
    """
    _owner, pool = _write_segment(
        rank=2, layer_key="L4", attr="w2", expert_ids=(5,), row_shape=(4,)
    )

    pool[0].fill_(0xFF)  # a writable view of the same bytes

    assert int(pool[0][0]) == 0xFF
    with pytest.raises(AssertionError):
        assert int(pool[0][0]) != 0xFF, "the mapping was writable"


# --------------------------------------------------------------------------
# 4. the wired fetch: planner + cache, end to end on CPU
# --------------------------------------------------------------------------


class _Layer:
    """The surface ``MoEExpertOffloadCache`` reads off a staged layer."""

    def __init__(self, plan, resident, spill, assignment, layer_key):
        self.layer_id = 0
        self.num_local_experts = plan.num_experts
        self._moe_offload_full_experts = plan.num_experts
        self._moe_offload_presplit = {"w13_qweight": (resident, spill)}
        self._moe_offload_frozen_layout = (
            list(plan.resident_ids),
            list(plan.spill_ids),
        )
        self._moe_offload_delegated_experts = list(plan.delegated_ids)
        self._moe_cold_tier_assignment = assignment
        self._moe_cold_tier_layer_key = layer_key
        self._moe_offload_host_shard = eo.host_shard_row(plan)


def _staged_cache(monkeypatch, layer_key="L0"):
    """A rank-0 cache whose delegated rows live in rank 2's segment."""
    ratio = _ratio()
    plan = plan_load_time_staging(
        12, fraction=0.25, cold_shard=eo.ColdShardContext(0, WORLD, ratio)
    )
    pool = tuple(sorted(set(plan.spill_ids) | set(plan.delegated_ids)))
    assignment = assign_cold_experts(pool, ratio, rank=0, world_size=WORLD)

    # Every peer stages the rows the shared plan gave it.
    for peer in (1, 2):
        ids = tuple(e for e in pool if assignment.owners[pool.index(e)] == peer)
        if ids:
            _write_segment(peer, layer_key, "w13_qweight", ids, (4,))

    resident = torch.zeros((plan.buffer_slots, 4), dtype=torch.uint8)
    for slot, expert_id in enumerate(plan.resident_ids):
        resident[slot].fill_(expert_id)
    # rank 0's OWN cold rows, in the shape the cache expects.
    spill = torch.zeros((len(plan.spill_ids), 4), dtype=torch.uint8)
    for row, expert_id in enumerate(plan.spill_ids):
        spill[row].fill_(expert_id)

    layer = _Layer(plan, resident, spill, assignment, layer_key)
    cache = eo.MoEExpertOffloadCache(layer, 0.25)
    cache.install()
    return cache, plan, assignment


def test_a_delegated_expert_is_fetched_from_the_peer_instead_of_refused(monkeypatch):
    cache, plan, assignment = _staged_cache(monkeypatch)
    assert plan.delegated_ids, "the fixture must actually delegate something"
    remote = plan.delegated_ids[0]

    slot_of_needed, fetch_plan = cache.planner.resolve([remote])
    cache._fetch(fetch_plan)

    slot = slot_of_needed[remote]
    assert torch.equal(
        cache._resident["w13_qweight"][slot],
        torch.full((4,), remote, dtype=torch.uint8),
    )
    assert cache.planner.stats.remote_fetches == 1
    assert cache.planner.stats.remote_h2d_bytes == 4


def test_a_local_cold_expert_still_comes_from_the_local_pool(monkeypatch):
    cache, plan, _ = _staged_cache(monkeypatch)
    local = plan.spill_ids[0]

    slot_of_needed, fetch_plan = cache.planner.resolve([local])
    cache._fetch(fetch_plan)

    assert torch.equal(
        cache._resident["w13_qweight"][slot_of_needed[local]],
        torch.full((4,), local, dtype=torch.uint8),
    )
    assert cache.planner.stats.remote_fetches == 0
    assert cache.planner.stats.remote_h2d_bytes == 0


def test_a_mixed_wave_places_local_and_remote_rows_in_the_right_slots(monkeypatch):
    cache, plan, _ = _staged_cache(monkeypatch)
    needed = [plan.spill_ids[0], plan.delegated_ids[0], plan.resident_ids[0]]

    slot_of_needed, fetch_plan = cache.planner.resolve(needed)
    cache._fetch(fetch_plan)

    for expert_id in needed:
        assert torch.equal(
            cache._resident["w13_qweight"][slot_of_needed[expert_id]],
            torch.full((4,), expert_id, dtype=torch.uint8),
        ), expert_id


def test_the_wired_fetch_pin_can_fail(monkeypatch):
    """CAN-FAIL ARM for the three tests above.

    Planted defect: the resolver is dropped after construction -- "wired but
    inert", the #421 shape this whole slice exists to remove. The planner then
    reverts to the slice-1 refusal, which is exactly what the property must
    see.
    """
    cache, plan, _ = _staged_cache(monkeypatch)
    remote = plan.delegated_ids[0]

    cache._cold_tier = None
    cache._remote_ids = frozenset()
    cache.planner.delegated_reachable = False

    with pytest.raises(RuntimeError, match="no shared cold tier is attached"):
        cache.planner.resolve([remote])


def test_the_dump_row_says_how_a_delegated_expert_is_reached(monkeypatch):
    """A proportional arm and a proportional arm whose tier never attached must
    not look the same in the #390 dump."""
    monkeypatch.setenv("SGLANG_EXPERT_STATS", "1")
    from sglang.srt.layers.moe import expert_stats as es

    es.reset_for_tests()
    try:
        cache, _plan, _ = _staged_cache(monkeypatch, layer_key="L0")
        assert cache._router_stats is not None
        assert cache._router_stats.host_shard["reachability"] == "shared-cold-tier"
        assert cache._router_stats.host_shard["policy"] == "link-proportional"
    finally:
        es.reset_for_tests()


def test_the_dump_carries_the_fields_the_arm_readout_reads(tmp_path, monkeypatch):
    """``scripts/dev/394_s2_proof/read_arm.py`` reads four keys out of
    ``totals``. Pin them here, or the proof window discovers at 3 a.m. that the
    readout was written against fields nobody emits."""
    monkeypatch.setenv("SGLANG_EXPERT_STATS", "1")
    monkeypatch.setenv("SGLANG_EXPERT_STATS_PATH", str(tmp_path / "stats"))
    from sglang.srt.layers.moe import expert_stats as es

    es.reset_for_tests()
    try:
        cache, plan, _ = _staged_cache(monkeypatch)
        _slots, fetch_plan = cache.planner.resolve([plan.delegated_ids[0]])
        cache._fetch(fetch_plan)
        payload = es.get_collector().snapshot(reason="test")
    finally:
        es.reset_for_tests()

    totals = payload["totals"]
    assert totals["host_shard_policy"] == "link-proportional"
    assert totals["host_shard_reachability"] == "shared-cold-tier"
    assert totals["h2d_bytes"] > 0
    assert totals["remote_h2d_bytes"] > 0


def test_a_plan_that_disagrees_with_the_assignment_is_refused_at_install():
    """The two must describe ONE pool. A cache that fetched under the
    disagreement would read a peer row for an expert the peer never staged."""
    ratio = _ratio()
    plan = plan_load_time_staging(
        12, fraction=0.25, cold_shard=eo.ColdShardContext(0, WORLD, ratio)
    )
    # An assignment built from a DIFFERENT pool (one expert short).
    truncated = tuple(sorted(set(plan.spill_ids) | set(plan.delegated_ids)))[1:]
    assignment = assign_cold_experts(truncated, ratio, rank=0, world_size=WORLD)
    resident = torch.zeros((plan.buffer_slots, 4), dtype=torch.uint8)
    spill = torch.zeros((max(1, len(plan.spill_ids)), 4), dtype=torch.uint8)
    layer = _Layer(plan, resident, spill, assignment, "L0")

    with pytest.raises(RuntimeError, match="different cold pools"):
        eo.MoEExpertOffloadCache(layer, 0.25)


# --------------------------------------------------------------------------
# 5. the default path, and the enablement boundary
# --------------------------------------------------------------------------


def test_with_the_tier_off_nothing_here_is_reachable(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_COLD_TIER_SHM", raising=False)

    assert cold_tier_enabled() is False
    assert ctf.owner_for_layer("L0", 0, WORLD, CARDS) is None
    assert ctf.resolver_for_layer(object()) is None


def test_a_worker_never_mints_its_own_launch_id(monkeypatch):
    monkeypatch.delenv(COLD_TIER_INSTANCE_ENV, raising=False)

    with pytest.raises(ColdTierUnavailable, match="launcher publishes it"):
        instance_id()


def test_the_launcher_mints_once_and_never_overwrites(monkeypatch):
    monkeypatch.delenv(COLD_TIER_INSTANCE_ENV, raising=False)
    first = publish_cold_tier_instance()

    assert first and publish_cold_tier_instance() == first
    assert instance_id() == first


def test_a_segment_needs_the_owners_card_not_just_a_rank_index():
    with pytest.raises(ColdTierUnavailable, match="card UUID"):
        ColdTierOwner(INSTANCE, 0, "L0", card_uuid="")


def test_a_rank_card_vector_of_the_wrong_length_refuses_the_tier():
    with pytest.raises(ColdTierUnavailable, match="rank->card vector"):
        ctf.owner_for_layer("L0", 0, WORLD, ("CARD-A", "CARD-B"))


def test_a_rank_that_owns_no_cold_rows_gets_no_segment():
    """Legal under a lopsided ratio, and it must not be an error: the caller
    already handles a ``None`` spill pool (a fully resident layer makes one)."""
    owner = ColdTierOwner(INSTANCE, 1, "L9", CARDS[1])

    assert owner.allocate_spill_pool("w13_qweight", (), (4,), torch.uint8) is None


def test_the_planner_refusal_is_unchanged_without_a_tier():
    """Slice-1 behaviour, verbatim: no tier, no reachability, named error."""
    planner = ExpertResidencyPlanner(
        num_local_experts=8, resident_count=2, scratch=4, delegated_ids=frozenset({5})
    )

    with pytest.raises(RuntimeError, match="delegated to a peer rank's host tier"):
        planner.resolve([5])


def test_the_capturable_installer_refuses_a_peer_backed_pool(monkeypatch):
    """Graph seam, BOOT-PENDING: the refusal must name the missing pointer and
    the flag that develops it, not merely fail."""
    monkeypatch.delenv("SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE", raising=False)

    with pytest.raises(RuntimeError, match="cudaHostRegister"):
        eo.refuse_capturable_cold_tier(64)


def test_the_graph_seam_can_be_opened_for_a_card_window(monkeypatch):
    monkeypatch.setenv("SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE", "1")

    assert eo.refuse_capturable_cold_tier(64) is None
