# SPDX-License-Identifier: Apache-2.0
"""Hermetic falsifier for link-proportional cold-expert sharding (#394).

No CUDA, no driver, no model. Everything the policy consumes -- the ratio, the
per-rank card identity, the PCIe link geometry -- is injected, so what is
pinned here is the CONTRACT, not a table of numbers:

  * the ratio-derivation chain and its preference order: explicit env vector >
    NVML PCIe link width x generation > equal, with the source named in the
    object rather than inferred by the caller;
  * card identity resolved by UUID through the #331 IdentityMap and never by
    position -- the rig used below is the real one, where CUDA ordinal 0 and
    NVML index 0 are DIFFERENT cards, so a positional implementation produces a
    different answer and fails (the #392 lesson, as a test);
  * the split itself: whole experts on dim 0, shares within one expert of the
    exact proportional share, summing to the pool exactly;
  * DIRECTION: the weak link is handed FEWER cold experts, not more. This is
    the deliberate inverse of the capacity-split rule and is worth a test of
    its own, because a future reader "fixing" it back is the failure mode;
  * default-unchanged: with no ratio the plan is the pre-#394 plan, field for
    field -- plus a companion test proving that pin CAN fail;
  * VRAM invariance: resident_count / buffer_slots / resident_ids are the same
    numbers with and without a ratio, so no per-card budget and no #400 ledger
    figure moves;
  * composition with the frozen-layout path (the #82 pad expert pinned to slot
    0 is resident, therefore never in the cold pool, therefore never
    delegated);
  * both load-time halves: the #123-GGUF stager and the fp8/GPTQ/AWQ presplit
    produce the SAME split from the same plan object.

Run:
  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \
    python -m pytest tests/moe_offload/test_link_proportional_shards.py -q
"""

import contextlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    HOST_SHARD_RATIO_ENV,
    ColdShardContext,
    ExpertResidencyPlanner,
    HostShardRatio,
    MoEExpertOffloadCache,
    cold_shard_context,
    derive_link_weights,
    partition_cold_experts,
    plan_load_time_staging,
    plan_proportional_shares,
    presplit_expert_offload_after_repack,
    resolve_host_shard_ratio,
    reset_expert_offload_release,
    reset_host_shard_log_latch,
    resident_slot_count,
    scratch_slot_count,
    stage_experts_into_tiers,
)

# This rig's measured H2D out of pinned host memory, GB/s (ANALYSE_393 §7.2).
# gen4 x4 feeds a 20 GB 3080; the two x8 slots feed the 5090 and the other
# 3080. Used as the explicit-env arm, which is the arm an operator would set.
MEASURED = (6.4, 13.0, 13.0)

E = 64  # local experts per rank in the toy geometry
ROW = 12  # opaque bytes per expert row (stands in for ggml blocks)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(HOST_SHARD_RATIO_ENV, raising=False)
    monkeypatch.setenv("SGLANG_MOE_RESIDENT_EXPERT_FRACTION", "0.25")
    monkeypatch.delenv("SGLANG_MOE_SCRATCH_SLOTS", raising=False)
    reset_host_shard_log_latch()
    reset_expert_offload_release()
    yield
    reset_host_shard_log_latch()
    reset_expert_offload_release()


# --------------------------------------------------------------------------
# the rig: CUDA order and NVML order deliberately disagree
# --------------------------------------------------------------------------

# NVML enumerates in bus order, CUDA defaults to FASTEST_FIRST, so on this box
# the 5090 is CUDA ordinal 0 and NVML index 1. Ranks are launched in CUDA
# order. Every positional shortcut breaks right here.
CARDS = [
    # (nvml_index, uuid, name, bdf, pcie gen, pcie width, cuda ordinal)
    (0, "GPU-3080-a", "NVIDIA GeForce RTX 3080", "00000000:0B:00.0", 4, 4, 1),
    (1, "GPU-5090", "NVIDIA GeForce RTX 5090", "00000000:2D:00.0", 4, 8, 0),
    (2, "GPU-3080-b", "NVIDIA GeForce RTX 3080", "00000000:41:00.0", 4, 8, 2),
]
UUID_BY_RANK = ("GPU-5090", "GPU-3080-a", "GPU-3080-b")  # rank == CUDA ordinal


class _FakePynvml:
    """Just enough NVML to answer the two link queries, keyed by NVML index."""

    def __init__(self, cards):
        self._gen = {c[0]: c[4] for c in cards}
        self._width = {c[0]: c[5] for c in cards}
        self.queried = []

    def nvmlDeviceGetHandleByIndex(self, index):  # noqa: N802 - NVML spelling
        if index not in self._gen:
            raise RuntimeError(f"no NVML device at index {index}")
        self.queried.append(index)
        return index

    def nvmlDeviceGetMaxPcieLinkGeneration(self, handle):  # noqa: N802
        return self._gen[handle]

    def nvmlDeviceGetMaxPcieLinkWidth(self, handle):  # noqa: N802
        return self._width[handle]


@pytest.fixture
def nvml_rig(monkeypatch):
    """Patch the registry so the real derivation runs against a fake driver."""
    from sglang.srt.registry import nvml as nvml_mod

    cards = [
        nvml_mod.CardIdentity(
            uuid=uuid,
            nvml_index=idx,
            pci_bus_id=bdf,
            name=name,
            total_bytes=(32 if "5090" in name else 20) * (1024**3),
            cuda_ordinal=ordinal,
        )
        for idx, uuid, name, bdf, _gen, _width, ordinal in CARDS
    ]
    fake = _FakePynvml(CARDS)

    @contextlib.contextmanager
    def _session():
        yield fake

    monkeypatch.setattr(
        nvml_mod, "identity_map", lambda *a, **k: nvml_mod.IdentityMap(cards)
    )
    monkeypatch.setattr(nvml_mod, "nvml_session", _session)
    return fake


# --------------------------------------------------------------------------
# 1. the derivation chain
# --------------------------------------------------------------------------


def test_explicit_env_vector_wins_and_names_itself(monkeypatch):
    monkeypatch.setenv(HOST_SHARD_RATIO_ENV, "6.4,13,13")

    ratio = resolve_host_shard_ratio(3, UUID_BY_RANK, link_gbps=lambda u: 1.0)

    assert ratio.source == "env"
    assert not ratio.is_equal
    # It outranks a derivation that would have said "all three links equal".
    assert ratio.weights[1] == pytest.approx(ratio.weights[2])
    assert ratio.weights[0] < ratio.weights[1]
    assert HOST_SHARD_RATIO_ENV in ratio.describe()


@pytest.mark.parametrize(
    "raw", ["6.4,13", "6.4,13,13,13", "6.4,x,13", "6.4,0,13", "6.4,-1,13"]
)
def test_a_malformed_env_vector_is_a_hard_error_not_a_fallback(monkeypatch, raw):
    """An operator who typed a ratio meant it. Silently running a different
    split than the one that was asked for turns a measurement arm into a lie."""
    monkeypatch.setenv(HOST_SHARD_RATIO_ENV, raw)

    with pytest.raises(ValueError):
        resolve_host_shard_ratio(3, UUID_BY_RANK)


def test_nvml_derivation_resolves_the_card_by_uuid_not_by_position(nvml_rig):
    """The #392 falsifier. Rank order is CUDA order; NVML order is bus order;
    on this rig they differ. The right answer follows the UUID."""
    ratio = resolve_host_shard_ratio(3, UUID_BY_RANK)

    assert ratio.source == "nvml-pcie"
    # rank 0 = 5090 = NVML 1 = x8, rank 1 = 3080-a = NVML 0 = x4, rank 2 = x8.
    assert ratio.weights == pytest.approx((0.4, 0.2, 0.4))
    # A positional implementation (rank r -> NVML index r) would answer this,
    # and it is a different answer -- so the assertion above has teeth.
    assert ratio.weights != pytest.approx((0.2, 0.4, 0.4))
    assert sorted(nvml_rig.queried) == [0, 1, 2]


def test_no_card_identity_and_no_env_falls_back_to_equal():
    ratio = resolve_host_shard_ratio(3)

    assert ratio.source == "equal"
    assert ratio.is_equal
    assert ratio.weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_an_unanswerable_link_query_falls_back_to_equal_not_to_a_guess():
    """Partial knowledge is worse than none: two ranks deriving different
    weights would partition the same pool two different ways."""
    ratio = resolve_host_shard_ratio(
        3, UUID_BY_RANK, link_gbps=lambda u: None if u == "GPU-5090" else 13.0
    )

    assert (ratio.source, ratio.is_equal) == ("equal", True)


def test_co_located_ranks_divide_the_link_they_share():
    """Two ranks behind one x8 slot have an x4's worth of link seconds each --
    the quantity being apportioned is link time, not cards (#82 co-location)."""
    weights = derive_link_weights(
        ["GPU-5090", "GPU-5090", "GPU-3080-a"],
        link_gbps=lambda u: 15.75 if u == "GPU-5090" else 7.88,
    )

    assert weights[0] == pytest.approx(weights[1])
    assert weights[0] == pytest.approx(weights[2], rel=1e-2)


def test_context_is_none_when_nothing_is_known():
    """`cold_shard=None` and "#394 absent" have to be the same code path."""
    assert cold_shard_context(0, 1) is None
    assert cold_shard_context(0, 3) is None  # no env, no card identity


# --------------------------------------------------------------------------
# 2. the split
# --------------------------------------------------------------------------


def test_shares_track_the_ratio_within_whole_expert_rounding():
    ratio = HostShardRatio(
        tuple(w / sum(MEASURED) for w in MEASURED), "test", "measured H2D"
    )
    pool = 48

    shares = plan_proportional_shares(pool, ratio.weights)

    assert sum(shares) == pool  # whole experts, nothing lost or invented
    for got, want in zip(shares, (pool * w for w in ratio.weights)):
        assert abs(got - want) < 1.0


def test_shares_sum_exactly_for_every_pool_size():
    ratio = tuple(w / sum(MEASURED) for w in MEASURED)
    for pool in range(0, 200):
        shares = plan_proportional_shares(pool, ratio)
        assert sum(shares) == pool, pool


def test_the_partition_is_a_pure_function_so_ranks_agree_without_talking():
    ids = list(range(7, 7 + 40))
    first = partition_cold_experts(ids, MEASURED)

    assert partition_cold_experts(ids, MEASURED) == first
    assert [e for block in first for e in block] == ids  # covers, no overlap


def test_the_weak_link_gets_fewer_cold_experts_not_more():
    """DIRECTION, the deliberate inverse of the capacity-split rule.

    A capacity split hands the weakest participant the largest relative load
    and the slowest rank still sets the clock. Here the split is of BYTES THAT
    MUST CROSS A LINK, and a fetch wave is over when the slowest link finishes
    its share -- so the narrow link is given LESS, and all three finish
    together. A future reader "restoring" the usual direction breaks this.
    """
    shares = plan_proportional_shares(48, MEASURED)

    assert shares[0] < shares[1] == shares[2]
    # And the finish times equalize, which is the point of the whole exercise.
    # Whole experts are the grain, so the residual spread is bounded by one
    # expert's transfer time on the narrowest link and by nothing tighter.
    finish = [n / bw for n, bw in zip(shares, MEASURED)]
    assert max(finish) - min(finish) < 1.0 / min(MEASURED)
    # Equal shares over the same links would NOT converge: the x4 rank takes
    # 2.03x as long as either x8 rank, and the whole wave waits for it.
    equal = [16 / bw for bw in MEASURED]
    assert max(equal) / min(equal) == pytest.approx(13.0 / 6.4)
    assert max(finish) < max(equal)


# --------------------------------------------------------------------------
# 3. default unchanged -- with its can-fail companion
# --------------------------------------------------------------------------


def _default_plan():
    return plan_load_time_staging(E, fraction=0.25)


def test_without_a_ratio_the_plan_is_the_pre_394_plan_field_for_field():
    plan = _default_plan()
    R = resident_slot_count(E, 0.25)

    assert plan.resident_count == R
    assert plan.buffer_slots == min(R + scratch_slot_count(R), E)
    assert plan.resident_ids == tuple(range(R))
    assert plan.spill_ids == tuple(range(R, E))
    assert plan.delegated_ids == ()
    assert plan.host_shard is None
    assert plan.is_static_layout


def test_the_default_unchanged_pin_can_fail():
    """The companion the pin above needs: the same three assertions applied to
    a plan built WITH a ratio must fail, or the pin is testing nothing."""
    ctx = ColdShardContext(0, 3, HostShardRatio(_norm(MEASURED), "test", ""))
    plan = plan_load_time_staging(E, fraction=0.25, cold_shard=ctx)
    default = _default_plan()

    assert plan.spill_ids != default.spill_ids
    assert plan.delegated_ids != ()
    assert not plan.is_static_layout
    assert plan.host_shard is not None


def test_an_equal_ratio_is_indistinguishable_from_no_ratio():
    ctx = ColdShardContext(0, 3, HostShardRatio((1 / 3, 1 / 3, 1 / 3), "equal", ""))

    assert not ctx.active
    assert plan_load_time_staging(E, fraction=0.25, cold_shard=ctx) == _default_plan()


def _norm(values):
    total = float(sum(values))
    return tuple(v / total for v in values)


# --------------------------------------------------------------------------
# 4. what must NOT move: residency, VRAM, the ledger
# --------------------------------------------------------------------------


def test_the_ratio_moves_host_ownership_and_nothing_on_the_card():
    default = _default_plan()
    shifted = [
        plan_load_time_staging(
            E,
            fraction=0.25,
            cold_shard=ColdShardContext(
                r, 3, HostShardRatio(_norm(MEASURED), "test", "")
            ),
        )
        for r in range(3)
    ]

    for plan in shifted:
        # Every number a per-card VRAM budget or a #400 ledger row is computed
        # from is untouched -- only the host tier's membership changed.
        assert plan.resident_count == default.resident_count
        assert plan.buffer_slots == default.buffer_slots
        assert plan.resident_ids == default.resident_ids
        assert plan.num_experts == default.num_experts

    # The three ranks partition the SAME cold pool the default plan had.
    assert sum(len(p.spill_ids) for p in shifted) == len(default.spill_ids)
    union = sorted(e for p in shifted for e in p.spill_ids)
    assert tuple(union) == default.spill_ids
    # ...and each rank's delegated set is exactly what the peers own.
    for r, plan in enumerate(shifted):
        peers = sorted(e for i, p in enumerate(shifted) if i != r for e in p.spill_ids)
        assert list(plan.delegated_ids) == peers
    assert len(shifted[0].spill_ids) < len(shifted[1].spill_ids)  # x4 rank owns least


def test_the_pinned_pad_expert_is_resident_and_can_never_be_delegated():
    """#82's trailing all-zero padding expert sits at id E-1 and EVERY foreign
    token routes to it. It is pinned resident, so it is not in the cold pool,
    so the #394 partition cannot reach it -- the two features compose without
    either one knowing about the other."""
    for r in range(3):
        plan = plan_load_time_staging(
            E,
            fraction=0.25,
            pinned_experts=(E - 1,),
            cold_shard=ColdShardContext(
                r, 3, HostShardRatio(_norm(MEASURED), "test", "")
            ),
        )
        assert plan.resident_ids[0] == E - 1
        assert E - 1 not in plan.spill_ids
        assert E - 1 not in plan.delegated_ids


def test_a_delegated_expert_reaching_the_router_is_refused_by_name():
    planner = ExpertResidencyPlanner(
        num_local_experts=E, resident_count=8, scratch=8, delegated_ids=frozenset({40})
    )

    planner.resolve([1, 2, 30])  # owned cold experts resolve normally
    with pytest.raises(RuntimeError, match="delegated"):
        planner.resolve([1, 40])


# --------------------------------------------------------------------------
# 5. both load-time halves produce the same split
# --------------------------------------------------------------------------


@pytest.fixture
def desk_pinning(monkeypatch):
    """``pin_memory()`` needs a CUDA context; the presplit half has no
    ``torch.cuda.is_available()`` guard around it the way the GGUF stager does.
    What these tests check is ADDRESSING -- which expert row lands in which
    slot and which pool row -- so the pinning itself is stubbed out and the
    same code path runs on the desk. Pinning is what makes the later H2D fetch
    async; it has no bearing on the layout under test."""
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self, raising=True)


def _opaque_experts(num=E):
    """One row of opaque bytes per expert, stamped with the expert id."""
    return torch.stack(
        [torch.full((ROW,), e, dtype=torch.uint8) for e in range(num)], dim=0
    )


def test_gguf_staging_half_stages_exactly_the_owned_cold_experts():
    experts = _opaque_experts()
    ctx = ColdShardContext(0, 3, HostShardRatio(_norm(MEASURED), "test", "measured"))
    plan = plan_load_time_staging(E, fraction=0.25, cold_shard=ctx)

    asked, released = [], []

    def source(e):
        asked.append(e)
        return experts[e]

    out = torch.zeros((plan.buffer_slots, ROW), dtype=torch.uint8)
    spill = stage_experts_into_tiers(plan, source, out, release=released.append)

    # Host tier holds this rank's share, byte for byte, in plan order.
    assert spill.shape[0] == len(plan.spill_ids)
    for row, e in enumerate(plan.spill_ids):
        assert torch.equal(spill[row], experts[e])
    # A delegated expert's bytes are never read -- but they ARE released, or
    # the VRAM this saves would be paid back in host RAM.
    assert set(asked).isdisjoint(plan.delegated_ids)
    assert sorted(released) == list(range(E))
    # Device side is the default one: resident rows, scratch left alone.
    for slot, e in enumerate(plan.resident_ids):
        assert torch.equal(out[slot], experts[e])


def test_presplit_half_produces_the_same_split_from_the_same_plan(desk_pinning):
    """The fp8 / GPTQ / AWQ door and the GGUF door must not drift apart: both
    take their layout from ``plan_load_time_staging``."""
    ctx = ColdShardContext(0, 3, HostShardRatio(_norm(MEASURED), "test", "measured"))
    plan = plan_load_time_staging(E, fraction=0.25, cold_shard=ctx)

    layer = torch.nn.Module()
    layer.num_local_experts = E
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(_opaque_experts().to(torch.float32), requires_grad=False),
    )
    reference = layer.w13_weight.data.clone()

    presplit_expert_offload_after_repack(layer, cold_shard=ctx)

    buf, spill = layer._moe_offload_presplit["w13_weight"]
    assert buf.shape[0] == plan.buffer_slots
    assert spill.shape[0] == len(plan.spill_ids)
    for slot, e in enumerate(plan.resident_ids):
        assert torch.equal(buf[slot], reference[e])
    for row, e in enumerate(plan.spill_ids):
        assert torch.equal(spill[row], reference[e])
    # Non-static layout is published, so the cache adopts it verbatim.
    assert layer._moe_offload_frozen_layout == (
        list(plan.resident_ids),
        list(plan.spill_ids),
    )
    assert layer._moe_offload_delegated_experts == list(plan.delegated_ids)


def test_presplit_half_is_byte_identical_without_a_ratio(desk_pinning):
    """Default-unchanged for the half that three shipped quant paths call."""
    layer = torch.nn.Module()
    layer.num_local_experts = E
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(_opaque_experts().to(torch.float32), requires_grad=False),
    )
    reference = layer.w13_weight.data.clone()
    R = resident_slot_count(E, 0.25)

    presplit_expert_offload_after_repack(layer)

    buf, spill = layer._moe_offload_presplit["w13_weight"]
    assert torch.equal(buf[:R], reference[:R])
    assert torch.equal(spill, reference[R:])
    assert not hasattr(layer, "_moe_offload_frozen_layout")
    assert not hasattr(layer, "_moe_offload_delegated_experts")


def test_the_cache_adopts_the_delegated_set_as_a_guard():
    layer = torch.nn.Module()
    layer.num_local_experts = E
    ctx = ColdShardContext(0, 3, HostShardRatio(_norm(MEASURED), "test", ""))
    plan = plan_load_time_staging(E, fraction=0.25, cold_shard=ctx)
    layer._moe_offload_full_experts = E
    layer._moe_offload_frozen_layout = (list(plan.resident_ids), list(plan.spill_ids))
    layer._moe_offload_delegated_experts = list(plan.delegated_ids)

    cache = MoEExpertOffloadCache(layer, 0.25)

    assert cache.planner.delegated_ids == frozenset(plan.delegated_ids)
    with pytest.raises(RuntimeError, match="delegated"):
        cache.planner.resolve([plan.delegated_ids[0]])


def test_the_chosen_ratio_and_its_source_are_logged_at_staging_time(caplog):
    ctx = ColdShardContext(1, 3, HostShardRatio(_norm(MEASURED), "env", "measured H2D"))

    with caplog.at_level("INFO", logger="sglang.srt.layers.moe.expert_offload"):
        plan_load_time_staging(E, fraction=0.25, cold_shard=ctx)

    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "#394" in line and "source=env" in line and "measured H2D" in line
