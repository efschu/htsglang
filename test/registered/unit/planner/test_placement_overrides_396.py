"""#396(b): declarative regex placement overrides -- hermetic, planner-level.

No GPU, no NVML, no model load. The identity map used for the device-name
checks is built from injected records (``identity_map`` takes both sides
precisely so a rig whose CUDA and NVML orders disagree can be constructed
without a driver), and the planner is exercised through
``compute_placement_struct`` on a config dict.

Three claims:

* **parse** -- the grammar is what the help text says it is, including the
  split-on-last-'=' rule and the card check;
* **conflict-refusal** -- an override the solve cannot satisfy raises
  :class:`PlacementOverrideConflict`, naming the arithmetic. It is never
  relaxed, dropped, or turned into a note;
* **constraint-respected** -- a satisfiable override changes WHICH experts are
  resident, and an empty override set leaves the plan field-for-field what it
  was.
"""

import pytest

from sglang.srt.planner.placement import compute_placement_struct
from sglang.srt.planner.placement_overrides import (
    PlacementOverrideConflict,
    PlacementOverrideError,
    apply_expert_constraints,
    expert_tensor_names,
    first_match,
    parse_placement_overrides,
    resolve_expert_constraints,
)

# ---------------------------------------------------------------------------
# a rig whose CUDA and NVML orders deliberately disagree (the reference shape)
# ---------------------------------------------------------------------------

UUID_5090 = "GPU-00000000-0000-0000-0000-000000005090"
UUID_3080A = "GPU-00000000-0000-0000-0000-000000003080"
UUID_3080B = "GPU-00000000-0000-0000-0000-000000003081"


def _identity():
    from sglang.srt.registry.nvml import DeviceInfo, identity_map

    devices = [
        DeviceInfo(
            index=0,
            uuid=UUID_3080A,
            name="3080",
            total_bytes=20 << 30,
            pci_bus_id="0000:01:00.0",
        ),
        DeviceInfo(
            index=1,
            uuid=UUID_5090,
            name="5090",
            total_bytes=32 << 30,
            pci_bus_id="0000:02:00.0",
        ),
        DeviceInfo(
            index=2,
            uuid=UUID_3080B,
            name="3080",
            total_bytes=20 << 30,
            pci_bus_id="0000:03:00.0",
        ),
    ]
    # CUDA order is NOT NVML order: the 5090 is nvml 1 but cuda 0.
    ordinals = {"0000:02:00.0": 0, "0000:01:00.0": 1, "0000:03:00.0": 2}
    return identity_map(devices=devices, cuda_ordinals_by_bus=ordinals)


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def test_parse_none_and_empty_yield_no_overrides():
    assert parse_placement_overrides(None) == ()
    assert parse_placement_overrides([]) == ()
    assert parse_placement_overrides(["", "   "]) == ()


def test_parse_cpu_and_host_are_the_same_target():
    (a, b) = parse_placement_overrides([r"experts\.1\.=cpu", r"experts\.2\.=host"])
    assert a.target.kind == "cpu"
    assert b.target.kind == "cpu"
    assert a.order == 0 and b.order == 1


def test_parse_splits_on_the_last_equals_so_a_regex_may_contain_one():
    (o,) = parse_placement_overrides([r"experts\.[0-9]{1,2}=x\.weight=cpu"])
    assert o.pattern == r"experts\.[0-9]{1,2}=x\.weight"
    assert o.target.kind == "cpu"


def test_parse_rejects_a_spec_with_no_equals():
    with pytest.raises(PlacementOverrideError, match="not 'regex=target'"):
        parse_placement_overrides(["experts"])


def test_parse_rejects_an_empty_regex():
    with pytest.raises(PlacementOverrideError, match="empty regex"):
        parse_placement_overrides(["=cpu"])


def test_parse_rejects_an_invalid_regex():
    with pytest.raises(PlacementOverrideError, match="invalid regex"):
        parse_placement_overrides(["experts[=cpu"])


def test_parse_rejects_an_unknown_target():
    with pytest.raises(PlacementOverrideError, match="unknown target"):
        parse_placement_overrides(["experts=nvme"])


def test_gpu_index_resolves_through_the_identity_map_not_torch_order():
    (o,) = parse_placement_overrides(["experts=gpu:0"], identity=_identity())
    assert o.target.kind == "gpu"
    assert o.target.cuda_ordinal == 0
    # cuda 0 is the 5090, which NVML enumerates at index 1. Resolving through
    # the map is the entire reason this check exists.
    assert o.target.uuid == UUID_5090


def test_gpu_uuid_resolves_to_its_cuda_ordinal():
    (o,) = parse_placement_overrides(
        [f"experts=gpu:{UUID_3080B}"], identity=_identity()
    )
    assert o.target.uuid == UUID_3080B
    assert o.target.cuda_ordinal == 2


def test_absent_card_is_refused_by_name_not_matched_to_a_neighbour():
    with pytest.raises(PlacementOverrideError, match="cannot resolve"):
        parse_placement_overrides(
            ["experts=gpu:GPU-deadbeef-0000-0000-0000-000000000000"],
            identity=_identity(),
        )


def test_absent_cuda_ordinal_is_refused():
    with pytest.raises(PlacementOverrideError, match="no card on this host"):
        parse_placement_overrides(["experts=gpu:7"], identity=_identity())


def test_without_an_identity_map_the_card_check_is_skipped_not_faked():
    (o,) = parse_placement_overrides(["experts=gpu:7"], identity=None)
    assert o.target.cuda_ordinal == 7
    assert o.target.uuid is None


def test_disk_target_must_be_a_valid_fs_or_blob_tier():
    (o,) = parse_placement_overrides(["experts=disk:fs:rig0:/nvme"])
    assert o.target.kind == "disk"
    assert o.target.tier_id == "fs:rig0:/nvme"


def test_disk_target_refuses_a_vram_tier():
    with pytest.raises(PlacementOverrideError, match="must be an fs: or blob:"):
        parse_placement_overrides([f"experts=disk:vram:{UUID_5090}"])


def test_disk_target_refuses_a_malformed_tier_id():
    with pytest.raises(PlacementOverrideError, match="not a valid"):
        parse_placement_overrides(["experts=disk:nvme0"])


def test_first_match_wins_in_command_line_order():
    overrides = parse_placement_overrides(
        [r"experts\.3\.=gpu:0", r"experts\.=cpu"], identity=None
    )
    name = "model.layers.0.mlp.experts.3.w1.weight"
    assert first_match(overrides, name).target.kind == "gpu"
    other = "model.layers.0.mlp.experts.4.w1.weight"
    assert first_match(overrides, other).target.kind == "cpu"


def test_expert_names_cover_both_the_hf_and_gguf_spellings():
    """HF names are real checkpoint keys; the GGUF ones are synthetic.

    GGUF stores ONE expert-major tensor per projection, so no per-expert name
    exists on disk for an operator to match. The fork splices the expert id in
    to give the individual expert -- which is the placement unit -- an address.
    Pinned here so nobody later reads a blk.* form back as a checkpoint key.
    """
    names = expert_tensor_names(2, 5)
    assert "model.layers.2.mlp.experts.5.w1.weight" in names
    assert "blk.2.ffn_gate_exps.5.weight" in names


# ---------------------------------------------------------------------------
# constraint resolution + conflict refusal
# ---------------------------------------------------------------------------


def _resolve(specs, gpu_index=0, start=0, end=8, layers=2, rank=0):
    return resolve_expert_constraints(
        parse_placement_overrides(specs, identity=None),
        rank=rank,
        gpu_index=gpu_index,
        expert_start=start,
        expert_end=end,
        num_layers=layers,
    )


def test_no_overrides_is_an_empty_constraint_set():
    assert _resolve([]).is_empty


def test_cpu_rule_collects_the_matching_experts():
    c = _resolve([r"experts\.[01]\.=cpu"])
    assert c.host == (0, 1)
    assert c.resident == ()


def test_gpu_rule_on_the_ranks_own_card_pins_residency():
    c = _resolve([r"experts\.[67]\.=gpu:0"], gpu_index=0)
    assert c.resident == (6, 7)


def test_gpu_rule_naming_another_card_is_refused_not_silently_moved():
    with pytest.raises(PlacementOverrideConflict, match="cannot be moved to another"):
        _resolve([r"experts\.6\.=gpu:1"], gpu_index=0)


def test_one_expert_matched_to_two_targets_is_refused():
    # w1 to the gpu, w2 to the host: the same expert, two tiers.
    with pytest.raises(PlacementOverrideConflict, match="different targets"):
        _resolve([r"experts\.3\.w1=gpu:0", r"experts\.3\.w2=cpu"], gpu_index=0)


def test_disk_rule_lands_on_the_named_tier():
    c = _resolve([r"experts\.7\.=disk:fs:rig0:/nvme"])
    assert c.disk == ((7, "fs:rig0:/nvme"),)


def test_disk_rule_is_refused_by_the_solve_not_folded_into_host_ram():
    """A disk target parses, and is then refused rather than degraded.

    Folding it into the host set would make the plan report host RAM for mass
    the operator asked to keep on an NVMe tier -- a different answer wearing
    the same shape. The refusal is the falsifier: if a disk residency class is
    ever added, this test is what has to change with it.
    """
    c = _resolve([r"experts\.7\.=disk:fs:rig0:/nvme"])
    with pytest.raises(PlacementOverrideConflict, match="no disk rung"):
        apply_expert_constraints(c, 0, 8, resident_slots=3)


# ---------------------------------------------------------------------------
# feasibility arithmetic
# ---------------------------------------------------------------------------


def test_pinning_more_experts_than_slots_is_refused_with_both_numbers():
    c = _resolve([r"experts\.[0-4]\.=gpu:0"], gpu_index=0)
    with pytest.raises(PlacementOverrideConflict) as excinfo:
        apply_expert_constraints(c, 0, 8, resident_slots=2)
    msg = str(excinfo.value)
    assert "pin 5 experts resident" in msg
    assert "only 2 resident slots" in msg


def test_empty_constraints_reproduce_the_ascending_prefix_split():
    c = _resolve([])
    resident, host = apply_expert_constraints(c, 0, 8, resident_slots=3)
    assert resident == (0, 1, 2)
    assert host == (3, 4, 5, 6, 7)


def test_a_host_pin_pushes_an_unconstrained_expert_into_the_resident_set():
    c = _resolve([r"experts\.0\.=cpu"])
    resident, host = apply_expert_constraints(c, 0, 8, resident_slots=3)
    # 0 is forced off the card, so the slots go to the next ascending ids.
    assert resident == (1, 2, 3)
    assert host == (0, 4, 5, 6, 7)


def test_a_gpu_pin_keeps_a_high_id_expert_resident():
    c = _resolve([r"experts\.7\.=gpu:0"], gpu_index=0)
    resident, host = apply_expert_constraints(c, 0, 8, resident_slots=3)
    assert 7 in resident
    assert len(resident) == 3
    assert 7 not in host


# ---------------------------------------------------------------------------
# end to end through the planner solve
# ---------------------------------------------------------------------------

MOE_CFG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_hidden_layers": 4,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
    "torch_dtype": "bfloat16",
}


@pytest.fixture(autouse=True)
def _no_live_nvml(monkeypatch):
    """The planner tests must not read THIS box's cards.

    ``_identity_map_or_none`` builds a live map when NVML is present, which
    would make every assertion below a statement about the machine the suite
    happens to run on. The parse-level tests above inject their own rig; these
    exercise the SOLVE, for which the card check is not the subject.
    """
    from sglang.srt.planner import placement

    monkeypatch.setattr(placement, "_identity_map_or_none", lambda: None)


def _flags(**kw):
    base = {
        "tp_size": 1,
        "moe_resident_expert_fraction": 0.5,
    }
    base.update(kw)
    return base


def test_planner_without_overrides_leaves_the_rule_unchanged():
    plan = compute_placement_struct(MOE_CFG, _flags())
    rule = plan.offload.per_rank[0]
    assert rule.resident_ids is None
    assert rule.host_ids is None
    assert rule.resident_expert_count == 4
    assert (rule.host_expert_start, rule.host_expert_end) == (4, 8)


def test_planner_respects_a_satisfiable_override():
    plan = compute_placement_struct(
        MOE_CFG, _flags(expert_placement_override=[r"experts\.7\.=gpu:0"])
    )
    rule = plan.offload.per_rank[0]
    assert rule.resident_ids is not None
    assert 7 in rule.resident_ids
    assert rule.resident_expert_count == 4
    assert 7 not in rule.host_ids
    # The host set is no longer a tail, so the range is withheld rather than
    # reported wrongly.
    assert rule.host_expert_start is None
    assert any("placement overrides" in n for n in plan.notes)


def test_planner_refuses_an_unsatisfiable_override():
    with pytest.raises(PlacementOverrideConflict, match="only 4 resident slots"):
        compute_placement_struct(
            MOE_CFG, _flags(expert_placement_override=[r"experts\.[0-6]\.=gpu:0"])
        )


def test_planner_refuses_overrides_with_no_residency_split_to_constrain(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_RESIDENT_EXPERT_FRACTION", raising=False)
    flags = _flags(expert_placement_override=[r"experts\.0\.=cpu"])
    flags.pop("moe_resident_expert_fraction")
    with pytest.raises(PlacementOverrideConflict, match="no resident expert fraction"):
        compute_placement_struct(MOE_CFG, flags)


def test_planner_accepts_a_single_override_string_as_well_as_a_list():
    plan = compute_placement_struct(
        MOE_CFG, _flags(expert_placement_override=r"experts\.7\.=gpu:0")
    )
    assert 7 in plan.offload.per_rank[0].resident_ids
