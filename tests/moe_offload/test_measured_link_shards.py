# SPDX-License-Identifier: Apache-2.0
"""Hermetic falsifier: MEASURED link weights and the rank->card vector (#394).

``test_link_proportional_shards.py`` pins the split itself -- apportionment,
direction, default-unchanged, both staging doors. This file pins the two things
that decide whether that split ever runs on real hardware, and with what
authority:

  * **provenance.** A ratio is weighted only by a number somebody has. The
    chain is env > MEASURED pinned H2D (rigmon card probe) > ESTIMATE (the NVML
    PCIe width x generation nameplate) > refusal, the label is derived from the
    source so the two cannot be set to disagree, and
    ``SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE=measured`` makes the nameplate
    inadmissible. An ABSENT link yields an equal split in every setting.
  * **the rank -> card vector**, published by the launcher without a collective
    (#407 cut 2). Resolved through the #331 IdentityMap by CUDA ordinal, carried
    as UUIDs, all-or-nothing, length-checked by the consuming group, and refused
    outright multi-node or without a CUDA context in the launcher.

Both falsifiers the brief demands are committed as CAN-FAIL arms, next to the
property they guard: a planted equal split on unequal declared links, and a
planted absent-provenance bandwidth. A test that cannot fail is not evidence.

Run:
  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \
    python -m pytest tests/moe_offload/test_measured_link_shards.py -q
"""

import contextlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe import expert_offload as eo  # noqa: E402
from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    HOST_SHARD_MIN_PROVENANCE_ENV,
    HOST_SHARD_RATIO_ENV,
    HOST_SHARD_SOURCE_ENV,
    HOST_SHARD_SOURCE_EQUAL,
    HOST_SHARD_SOURCE_NVML,
    HOST_SHARD_SOURCE_PROBE,
    cold_shard_context,
    host_shard_row,
    plan_load_time_staging,
    reset_card_probe_memo,
    reset_host_shard_log_latch,
    resolve_host_shard_ratio,
)
from sglang.srt.registry import rank_cards as rc  # noqa: E402

# This rig, ANALYSE_393 §7.2. NVML enumerates in bus order, CUDA defaults to
# FASTEST_FIRST, so the 5090 is CUDA ordinal 0 and NVML index 1. The x4 slot --
# the one the whole feature exists for -- holds a 3080 at BDF 0B:00.0, which is
# NVML index 0 and CUDA ordinal 1. Every positional shortcut breaks right here.
CARDS = [
    # (nvml_index, uuid, name, bdf, gen, width, cuda_ordinal, measured h2d)
    (0, "GPU-3080-a", "NVIDIA GeForce RTX 3080", "00000000:0B:00.0", 4, 4, 1, 6.4),
    (1, "GPU-5090", "NVIDIA GeForce RTX 5090", "00000000:2D:00.0", 4, 8, 0, 13.0),
    (2, "GPU-3080-b", "NVIDIA GeForce RTX 3080", "00000000:41:00.0", 4, 8, 2, 13.0),
]
UUID_BY_RANK = ("GPU-5090", "GPU-3080-a", "GPU-3080-b")  # rank == CUDA ordinal
MEASURED_H2D = {c[1]: c[7] for c in CARDS}
X4_RANK = 1  # the rank behind the x4 slot, by BDF -- never assumed by index


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(HOST_SHARD_RATIO_ENV, raising=False)
    monkeypatch.delenv(HOST_SHARD_MIN_PROVENANCE_ENV, raising=False)
    monkeypatch.setenv("SGLANG_MOE_RESIDENT_EXPERT_FRACTION", "0.25")
    rc.clear_published_rank_cards()
    reset_card_probe_memo()
    reset_host_shard_log_latch()
    yield
    rc.clear_published_rank_cards()
    reset_card_probe_memo()
    reset_host_shard_log_latch()


class _FakePynvml:
    def __init__(self, cards):
        self._gen = {c[0]: c[4] for c in cards}
        self._width = {c[0]: c[5] for c in cards}

    def nvmlDeviceGetHandleByIndex(self, index):  # noqa: N802 - NVML spelling
        if index not in self._gen:
            raise RuntimeError(f"no NVML device at index {index}")
        return index

    def nvmlDeviceGetMaxPcieLinkGeneration(self, handle):  # noqa: N802
        return self._gen[handle]

    def nvmlDeviceGetMaxPcieLinkWidth(self, handle):  # noqa: N802
        # What the CARD can do. On the reference rig all three answer x16,
        # which is why the derivation must not use it -- see the current-width
        # test below.
        return 16

    def nvmlDeviceGetCurrPcieLinkWidth(self, handle):  # noqa: N802
        return self._width[handle]


def _identity_map():
    from sglang.srt.registry import nvml as nvml_mod

    return nvml_mod.IdentityMap(
        [
            nvml_mod.CardIdentity(
                uuid=uuid,
                nvml_index=idx,
                pci_bus_id=bdf,
                name=name,
                total_bytes=(32 if "5090" in name else 20) * (1024**3),
                cuda_ordinal=ordinal,
            )
            for idx, uuid, name, bdf, _g, _w, ordinal, _h in CARDS
        ]
    )


@pytest.fixture
def nvml_rig(monkeypatch):
    """The real derivation, against a fake driver."""
    from sglang.srt.registry import nvml as nvml_mod

    fake = _FakePynvml(CARDS)

    @contextlib.contextmanager
    def _session():
        yield fake

    monkeypatch.setattr(nvml_mod, "identity_map", lambda *a, **k: _identity_map())
    monkeypatch.setattr(nvml_mod, "nvml_session", _session)
    return fake


def _probe_profile(h2d_by_uuid):
    """A CardProbeProfile carrying exactly these measured H2D rates."""
    from sglang.srt.rigmon.card_probe import CardProbeMeasurement, CardProbeProfile

    return CardProbeProfile(
        cards=[
            CardProbeMeasurement(
                uuid=uuid,
                name="fake",
                cuda_index=i,
                h2d_gbs=h2d_by_uuid.get(uuid),
            )
            for i, uuid in enumerate(MEASURED_H2D)
        ]
    )


@pytest.fixture
def card_probe(monkeypatch):
    """Install a probe cache and count how often it is read."""
    from sglang.srt.rigmon import card_probe as probe_mod

    calls = {"load": 0}

    def _install(h2d_by_uuid):
        def _load(path=None):
            calls["load"] += 1
            return _probe_profile(h2d_by_uuid)

        monkeypatch.setattr(probe_mod, "load_card_probe", _load)
        reset_card_probe_memo()
        return calls

    return _install


# --------------------------------------------------------------------------
# 1. provenance: which number may weight a split
# --------------------------------------------------------------------------


def test_measured_h2d_outranks_the_nameplate_and_names_itself(nvml_rig, card_probe):
    card_probe(MEASURED_H2D)

    ratio = resolve_host_shard_ratio(3, UUID_BY_RANK)

    assert ratio.source == HOST_SHARD_SOURCE_PROBE
    assert ratio.provenance == "measured"
    # 13 : 6.4 : 13 normalized -- the measured 1.00 : 2.03, not the nameplate
    # 1 : 2. The two are close on purpose; the test reads the DIFFERENCE.
    assert ratio.weights[X4_RANK] == pytest.approx(6.4 / 32.4, abs=1e-6)
    nameplate = resolve_host_shard_ratio(3, UUID_BY_RANK, probe_gbps=lambda _u: None)
    assert nameplate.source == HOST_SHARD_SOURCE_NVML
    assert nameplate.provenance == "estimate"
    assert ratio.weights != nameplate.weights


def test_the_explicit_vector_still_outranks_the_probe(monkeypatch, card_probe):
    card_probe(MEASURED_H2D)
    monkeypatch.setenv(HOST_SHARD_RATIO_ENV, "1,1,2")

    ratio = resolve_host_shard_ratio(3, UUID_BY_RANK)

    assert ratio.source == HOST_SHARD_SOURCE_ENV
    assert ratio.provenance == "measured"
    assert ratio.weights == pytest.approx((0.25, 0.25, 0.5))


def test_a_partly_measured_rig_falls_to_the_estimate_never_to_a_mixture(
    nvml_rig, card_probe
):
    """One unmeasured card must not be filled in from the others.

    A ratio half measured and half nameplate is a fourth provenance nobody
    declared, and the ranks would not even agree on it if the probe cache
    differed between them.
    """
    partial = dict(MEASURED_H2D)
    partial["GPU-3080-a"] = None
    card_probe(partial)

    ratio = resolve_host_shard_ratio(3, UUID_BY_RANK)

    assert ratio.source == HOST_SHARD_SOURCE_NVML
    assert ratio.provenance == "estimate"
    # gen4 widths x8 / x4 / x8 -> 8 : 4 : 8, the whole vector from one source.
    assert ratio.weights == pytest.approx((0.4, 0.2, 0.4))


def test_min_provenance_measured_refuses_the_nameplate(monkeypatch, nvml_rig):
    monkeypatch.setenv(HOST_SHARD_MIN_PROVENANCE_ENV, "measured")

    ratio = resolve_host_shard_ratio(3, UUID_BY_RANK, probe_gbps=lambda _u: None)

    assert ratio.source == HOST_SHARD_SOURCE_EQUAL
    assert ratio.provenance == "absent"
    assert ratio.is_equal
    # The refusal has to say what would lift it, or it is just a failure.
    assert "card probe" in ratio.detail


def test_min_provenance_measured_still_accepts_a_measurement(
    nvml_rig, card_probe, monkeypatch
):
    monkeypatch.setenv(HOST_SHARD_MIN_PROVENANCE_ENV, "measured")
    card_probe(MEASURED_H2D)

    ratio = resolve_host_shard_ratio(3, UUID_BY_RANK)

    assert ratio.source == HOST_SHARD_SOURCE_PROBE


@pytest.mark.parametrize("raw", ["absent", "maybe", "0.5"])
def test_an_unselectable_min_provenance_is_a_hard_error(monkeypatch, raw):
    """Including 'absent': there is no setting in which a guess is a ratio."""
    monkeypatch.setenv(HOST_SHARD_MIN_PROVENANCE_ENV, raw)

    with pytest.raises(ValueError, match="measured.*estimate"):
        resolve_host_shard_ratio(3, UUID_BY_RANK)


def test_provenance_is_derived_from_the_source_not_settable_beside_it():
    from sglang.srt.layers.moe.expert_offload import HostShardRatio

    assert HostShardRatio((0.5, 0.5), HOST_SHARD_SOURCE_PROBE).provenance == "measured"
    assert HostShardRatio((0.5, 0.5), HOST_SHARD_SOURCE_NVML).provenance == "estimate"
    assert HostShardRatio((0.5, 0.5), HOST_SHARD_SOURCE_EQUAL).provenance == "absent"
    # An unknown source is ABSENT, not "probably fine".
    assert HostShardRatio((0.5, 0.5), "invented").provenance == "absent"


def test_the_probe_is_read_once_per_process_and_never_triggers_a_measurement(
    nvml_rig, card_probe
):
    """40+ MoE layers must not each parse the cache, and none may probe."""
    from sglang.srt.rigmon import card_probe as probe_mod

    calls = card_probe(MEASURED_H2D)
    ran = {"probe": 0}
    original = probe_mod.run_card_probe
    probe_mod.run_card_probe = lambda *a, **k: ran.__setitem__("probe", 1)
    try:
        for _ in range(40):
            resolve_host_shard_ratio(3, UUID_BY_RANK)
    finally:
        probe_mod.run_card_probe = original

    assert calls["load"] == 1
    assert ran["probe"] == 0


# --------------------------------------------------------------------------
# 2. FALSIFIER: an absent bandwidth is refused, never guessed
# --------------------------------------------------------------------------


def test_absent_bandwidth_is_refused_and_the_context_disappears():
    """No probe, no NVML: equal, absent, and no ColdShardContext at all.

    ``None`` rather than an equal context on purpose -- that is what makes
    "nothing is known" and "#394 is not in this build" the same code path.
    """
    ratio = resolve_host_shard_ratio(
        3, UUID_BY_RANK, link_gbps=lambda _u: None, probe_gbps=lambda _u: None
    )

    assert ratio.source == HOST_SHARD_SOURCE_EQUAL
    assert ratio.provenance == "absent"
    assert ratio.is_equal
    assert (
        cold_shard_context(
            0, 3, UUID_BY_RANK, link_gbps=lambda _u: None, probe_gbps=lambda _u: None
        )
        is None
    )


def test_a_zero_or_negative_bandwidth_counts_as_absent_not_as_a_weight():
    """A 0.0 GB/s row is the classic almost-valid divisor. It is an absence."""
    for planted in (0.0, -1.0, float("nan")):
        ratio = resolve_host_shard_ratio(
            3,
            UUID_BY_RANK,
            link_gbps=lambda _u: None,
            probe_gbps=lambda u, v=planted: v if u == "GPU-3080-a" else 13.0,
        )
        assert ratio.provenance == "absent", planted


def test_the_absent_refusal_pin_can_fail():
    """CAN-FAIL ARM for the test above.

    Planted defect: a lookup that answers an unmeasured card with a plausible
    default instead of ``None`` -- the exact "fill it in with something" bug the
    refusal exists to prevent. The property must notice.
    """
    planted_guess = resolve_host_shard_ratio(
        3,
        UUID_BY_RANK,
        link_gbps=lambda _u: None,
        # the defect: absent becomes 8.0 GB/s, a number nobody measured
        probe_gbps=lambda u: 8.0 if u == "GPU-3080-a" else 13.0,
    )

    assert not planted_guess.is_equal
    assert planted_guess.provenance == "measured"  # it now LIES about itself
    with pytest.raises(AssertionError):
        assert planted_guess.provenance == "absent"


# --------------------------------------------------------------------------
# 3. FALSIFIER: unequal declared links must not produce an equal split
# --------------------------------------------------------------------------


def _shares(ratio, pool=48):
    plan = plan_load_time_staging(
        64,
        fraction=0.25,
        cold_shard=eo.ColdShardContext(0, 3, ratio),
    )
    return len(plan.spill_ids), len(plan.delegated_ids)


def test_unequal_declared_links_produce_unequal_shares(nvml_rig, card_probe):
    card_probe(MEASURED_H2D)
    ratio = resolve_host_shard_ratio(3, UUID_BY_RANK)

    from sglang.srt.layers.moe.expert_offload import partition_cold_experts

    shares = partition_cold_experts(tuple(range(48)), ratio.weights)
    counts = [len(s) for s in shares]

    assert sum(counts) == 48
    assert len(set(counts)) > 1, "unequal links produced an equal split"
    # the x4 rank gets the FEWEST -- direction, not just inequality
    assert counts[X4_RANK] == min(counts)
    assert counts[X4_RANK] < 48 // 3


def test_the_unequal_split_pin_can_fail():
    """CAN-FAIL ARM for the test above.

    Planted defect: the ratio is resolved correctly and then IGNORED, the
    classic "wired but inert" regression -- the weights arrive and the
    apportionment splits evenly anyway. The property must notice.
    """
    from sglang.srt.layers.moe.expert_offload import partition_cold_experts

    planted_equal = (1 / 3, 1 / 3, 1 / 3)
    counts = [len(s) for s in partition_cold_experts(tuple(range(48)), planted_equal)]

    assert counts == [16, 16, 16]
    with pytest.raises(AssertionError):
        assert len(set(counts)) > 1, "unequal links produced an equal split"


def test_the_x4_rank_is_identified_by_bdf_not_by_index(nvml_rig, card_probe):
    """#392 as a test: the weak link is NOT at NVML index 1 nor at CUDA 0.

    A positional implementation weights rank 0 (which is the 5090) as the weak
    one and produces a different, wrong vector.
    """
    card_probe(MEASURED_H2D)
    imap = _identity_map()

    weak_uuid = UUID_BY_RANK[X4_RANK]
    weak_card = imap.require(weak_uuid)
    assert weak_card.pci_bus_id == "00000000:0B:00.0"
    assert weak_card.nvml_index == 0 and weak_card.cuda_ordinal == 1

    ratio = resolve_host_shard_ratio(3, UUID_BY_RANK)
    assert ratio.weights.index(min(ratio.weights)) == X4_RANK
    assert ratio.weights.index(min(ratio.weights)) != weak_card.nvml_index


# --------------------------------------------------------------------------
# 4. the rank -> card vector, published without a collective
# --------------------------------------------------------------------------


class _Args:
    """Just the ServerArgs surface the publisher reads."""

    def __init__(self, rank_gpu_id=None, tp_size=3, pp_size=1, nnodes=1):
        self.rank_gpu_id = rank_gpu_id
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.nnodes = nnodes
        self.base_gpu_id = 0
        self.gpu_id_step = 1

    def world_rank(self, pp_rank, tp_rank):
        return pp_rank * self.tp_size + tp_rank

    def gpu_id_for_rank(self, pp_rank, tp_rank, pp_size_per_node, tp_size_per_node):
        if self.rank_gpu_id is not None:
            return self.rank_gpu_id[self.world_rank(pp_rank, tp_rank)]
        return (
            self.base_gpu_id
            + ((pp_rank % pp_size_per_node) * tp_size_per_node)
            + (tp_rank % tp_size_per_node) * self.gpu_id_step
        )


def test_the_launcher_publishes_uuids_in_world_rank_order():
    vector = rc.resolve_rank_card_vector(
        _Args(rank_gpu_id=[0, 1, 2]), identity_map=_identity_map()
    )

    assert vector.present
    assert vector.uuids == UUID_BY_RANK
    # BDFs travel with it, so a human can see which rank is on the x4 slot.
    assert vector.bdfs[X4_RANK] == "00000000:0B:00.0"


def test_publication_is_an_environment_write_and_nothing_else(monkeypatch):
    """The channel must be inert: no collective, no torch, no CUDA."""
    import sglang.srt.registry.nvml as nvml_mod

    monkeypatch.setattr(nvml_mod, "identity_map", lambda *a, **k: _identity_map())

    vector = rc.publish_rank_card_uuids(_Args(rank_gpu_id=[0, 1, 2]))

    assert vector.present
    assert os.environ[rc.RANK_CARD_UUIDS_ENV] == ",".join(UUID_BY_RANK)
    assert rc.rank_card_uuids(3) == UUID_BY_RANK


def test_a_hand_set_vector_is_never_overwritten(monkeypatch):
    monkeypatch.setenv(rc.RANK_CARD_UUIDS_ENV, "GPU-a,GPU-b")

    vector = rc.publish_rank_card_uuids(
        _Args(rank_gpu_id=[0, 1, 2]), identity_map=_identity_map()
    )

    assert vector.uuids == ("GPU-a", "GPU-b")
    assert os.environ[rc.RANK_CARD_UUIDS_ENV] == "GPU-a,GPU-b"


def test_an_unresolvable_ordinal_publishes_nothing_at_all():
    """All-or-nothing (#397): a partial vector would be completed by the
    consumer with the index it already has, which is the substitution the whole
    module exists to remove."""
    vector = rc.resolve_rank_card_vector(
        _Args(rank_gpu_id=[0, 1, 7]), identity_map=_identity_map()
    )

    assert not vector.present
    assert "CUDA device 7" in vector.reason
    assert rc.rank_card_uuids(3) is None


def test_a_multi_node_group_is_refused_by_name():
    vector = rc.resolve_rank_card_vector(
        _Args(rank_gpu_id=[0, 1, 2], nnodes=2), identity_map=_identity_map()
    )

    assert not vector.present
    assert "nodes" in vector.reason


def test_without_a_cuda_context_the_launcher_refuses_to_build_one(monkeypatch):
    """The context costs VRAM on every card in the process about to spawn
    workers onto them. Not paid silently."""
    monkeypatch.setattr(rc, "_cuda_side_already_paid", lambda _args: False)

    vector = rc.resolve_rank_card_vector(_Args())

    assert not vector.present
    assert "SGLANG_RANK_CARD_PROBE_CUDA" in vector.reason
    assert "--rank-gpu-id" in vector.reason


def test_rank_gpu_id_counts_as_already_paid():
    assert rc._cuda_side_already_paid(_Args(rank_gpu_id=[0, 1, 2])) is True


def test_a_vector_of_the_wrong_length_describes_a_different_group(monkeypatch):
    monkeypatch.setenv(rc.RANK_CARD_UUIDS_ENV, ",".join(UUID_BY_RANK))

    assert rc.rank_card_uuids(3) == UUID_BY_RANK
    # a MoE-TP subgroup of 2 must NOT read ranks 0..1 of a 3-rank vector
    assert rc.rank_card_uuids(2) is None
    assert "different group" in rc.rank_card_vector(2).reason


def test_a_vector_carries_either_uuids_or_a_reason_never_both():
    with pytest.raises(ValueError):
        rc.RankCardVector(uuids=("a",), reason="also absent")
    with pytest.raises(ValueError):
        rc.RankCardVector()


def test_the_published_vector_drives_the_ratio_end_to_end(
    monkeypatch, nvml_rig, card_probe
):
    """Publish in the launcher, read in the worker, get a measured ratio."""
    card_probe(MEASURED_H2D)
    monkeypatch.setattr(
        rc,
        "resolve_rank_card_vector",
        lambda *a, **k: rc.RankCardVector(uuids=UUID_BY_RANK, bdfs=(), source="test"),
    )
    rc.publish_rank_card_uuids(_Args(rank_gpu_id=[0, 1, 2]))

    context = cold_shard_context(X4_RANK, 3, rc.rank_card_uuids(3))

    assert context is not None
    assert context.ratio.source == HOST_SHARD_SOURCE_PROBE
    assert context.ratio.weights[X4_RANK] == min(context.ratio.weights)


# --------------------------------------------------------------------------
# 5. the #390 instrument names its own arm
# --------------------------------------------------------------------------


def test_the_host_shard_row_describes_the_arm_that_produced_it(nvml_rig, card_probe):
    card_probe(MEASURED_H2D)
    ratio = resolve_host_shard_ratio(3, UUID_BY_RANK)
    plan = plan_load_time_staging(
        64, fraction=0.25, cold_shard=eo.ColdShardContext(X4_RANK, 3, ratio)
    )

    row = host_shard_row(plan)

    assert row["policy"] == "link-proportional"
    assert "card-probe-h2d" in row["ratio"] and "measured" in row["ratio"]
    assert row["owned_cold_experts"] + row["delegated_cold_experts"] == row["cold_pool"]
    assert row["owned_share"] < 1.0 / 3.0  # the x4 rank owns less than its third


def test_the_baseline_arm_says_so_rather_than_staying_silent():
    """An equal/default plan still emits a row. A missing row and a baseline
    row are different findings and must not look alike in a dump."""
    plan = plan_load_time_staging(64, fraction=0.25)

    row = host_shard_row(plan)

    assert row["policy"] == "equal"
    assert row["delegated_cold_experts"] == 0
    assert row["owned_share"] == 1.0


def test_the_row_reaches_the_expert_stats_dump(
    tmp_path, monkeypatch, nvml_rig, card_probe
):
    from sglang.srt.layers.moe import expert_stats as es

    card_probe(MEASURED_H2D)
    monkeypatch.setenv("SGLANG_EXPERT_STATS", "1")
    monkeypatch.setenv("SGLANG_EXPERT_STATS_PATH", str(tmp_path / "stats"))
    es.reset_for_tests()
    try:
        ratio = resolve_host_shard_ratio(3, UUID_BY_RANK)
        plan = plan_load_time_staging(
            64, fraction=0.25, cold_shard=eo.ColdShardContext(X4_RANK, 3, ratio)
        )
        stats = es.maybe_layer_stats(
            layer_id=0, num_experts=64, resident_count=plan.resident_count
        )
        assert stats is not None
        stats.host_shard = host_shard_row(plan)
        stats.record([[0, 1], [2, 3]], resident_count=plan.resident_count)

        target = es.get_collector().dump(reason="test")
        payload = json.loads(open(target).read())
    finally:
        es.reset_for_tests()

    assert payload["layers"][0]["host_shard"]["policy"] == "link-proportional"
    assert payload["totals"]["host_shard_policy"] == "link-proportional"
    assert payload["totals"]["delegated_cold_experts"] > 0
    # the hit-rate row ANALYSE_389 wants, in the same file as the arm label
    assert "hit_rate" in payload["totals"]


def test_layers_staged_under_different_ratios_stay_visible_as_mixed(
    tmp_path, monkeypatch
):
    from sglang.srt.layers.moe import expert_stats as es

    monkeypatch.setenv("SGLANG_EXPERT_STATS", "1")
    monkeypatch.setenv("SGLANG_EXPERT_STATS_PATH", str(tmp_path / "stats"))
    es.reset_for_tests()
    try:
        for layer_id, policy in ((0, "equal"), (1, "link-proportional")):
            stats = es.maybe_layer_stats(
                layer_id=layer_id, num_experts=8, resident_count=2
            )
            stats.host_shard = {
                "policy": policy,
                "ratio": policy,
                "cold_pool": 6,
                "owned_cold_experts": 6 if policy == "equal" else 2,
                "delegated_cold_experts": 0 if policy == "equal" else 4,
                "owned_share": 1.0,
                "resident_count": 2,
                "num_experts": 8,
            }
        payload = es.get_collector().snapshot(reason="test")
    finally:
        es.reset_for_tests()

    assert payload["totals"]["host_shard_policy"] == "mixed"


# --------------------------------------------------------------------------
# 6. the nameplate must read the SLOT, not the card
# --------------------------------------------------------------------------


def test_the_estimate_reads_the_negotiated_width_not_the_card_maximum(nvml_rig):
    """Measured on the reference rig 2026-08-02, and it had disabled #394.

    ``nvmlDeviceGetMaxPcieLinkWidth`` answers x16 for all three cards while the
    slots are wired x4 / x8 / x8. Deriving from it yields an EQUAL ratio, an
    equal ratio yields no ``ColdShardContext``, and the whole feature is a
    silent no-op on exactly the box it exists for. The fake driver above
    reproduces that shape, so this test fails if the derivation regresses to
    the card maximum.
    """
    ratio = resolve_host_shard_ratio(3, UUID_BY_RANK, probe_gbps=lambda _u: None)

    assert ratio.source == HOST_SHARD_SOURCE_NVML
    assert not ratio.is_equal, "the derivation read the card maximum, not the slot"
    assert ratio.weights == pytest.approx((0.4, 0.2, 0.4))


def test_generation_still_comes_from_the_maximum_not_the_idle_link_state(nvml_rig):
    """The link idles down (gen 1 at rest on all three cards here).

    A current-generation read taken at weight-load time would describe the
    power state, not the slot, and would scale every rank by the same wrong
    constant -- harmless for the ratio, but the reason the asymmetry with width
    is deliberate rather than an oversight. Pinned so a future "make both
    current" tidy-up has to argue with a test.
    """
    from sglang.srt.layers.moe.expert_offload import _pcie_link_gbps_by_uuid

    # gen 4 lane rate x the x4 slot: 1.969 * 4
    assert _pcie_link_gbps_by_uuid("GPU-3080-a") == pytest.approx(1.969 * 4)
    assert _pcie_link_gbps_by_uuid("GPU-5090") == pytest.approx(1.969 * 8)
