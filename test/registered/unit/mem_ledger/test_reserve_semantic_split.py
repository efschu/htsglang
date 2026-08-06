"""The semantic split of the reserve flags, and the calibration cache.

Hermetic. No GPU, no NVML: every test here is about what the FLAGS mean, which
is decided at parse time and must therefore be decidable without hardware.
"""

import json
import os

import pytest

from sglang.srt.mem_ledger.calibration import (
    VRAM_CALIBRATION_VERSION,
    CalibrationProfile,
    CardResidual,
    calibration_cache_path,
    calibration_fingerprint,
    load_calibration,
    save_calibration,
)
from sglang.srt.mem_ledger.terms import DEFAULT_USER_RESERVE_MIB
from sglang.srt.server_args import ServerArgs

MODEL = "/nonexistent/model"


def make_args(**kwargs):
    """A ServerArgs shell for flag-semantics checks only.

    ``__post_init__`` is bypassed deliberately: this file tests the reserve
    flags' MEANING, and dragging a full arg resolution (which reads a
    checkpoint) into that would make the test about the checkpoint.
    """
    args = ServerArgs.__new__(ServerArgs)
    defaults = dict(
        model_path=MODEL,
        rank_auto_reserve_mib="auto",
        rank_user_reserve_mib=DEFAULT_USER_RESERVE_MIB,
        enable_vram_ledger=False,
    )
    defaults.update(kwargs)
    for key, value in defaults.items():
        setattr(args, key, value)
    return args


# --- the decreed default ----------------------------------------------------


def test_user_reserve_defaults_to_1024_mib_per_card():
    assert DEFAULT_USER_RESERVE_MIB == 1024
    field = ServerArgs.__dataclass_fields__["rank_user_reserve_mib"]
    assert field.default == DEFAULT_USER_RESERVE_MIB


def test_ledger_is_off_by_default_so_existing_recipes_are_unchanged():
    field = ServerArgs.__dataclass_fields__["enable_vram_ledger"]
    assert field.default is False


# --- the split is enforced, never silently merged --------------------------


def test_user_reserve_without_the_ledger_is_refused_not_ignored():
    args = make_args(rank_user_reserve_mib=4096, enable_vram_ledger=False)
    with pytest.raises(ValueError) as excinfo:
        args._check_vram_ledger_flags()
    assert "does nothing without --enable-vram-ledger" in str(excinfo.value)


def test_the_two_reserve_semantics_cannot_be_combined():
    args = make_args(rank_auto_reserve_mib="5500,3800,3800", enable_vram_ledger=True)
    with pytest.raises(ValueError) as excinfo:
        args._check_vram_ledger_flags()
    message = str(excinfo.value)
    assert "mean different things" in message
    # The message must teach the migration, since the runbook value is exactly
    # the conflated vector that motivated the split.
    assert "--rank-user-reserve-mib" in message
    assert "5500,3800,3800" in message


def test_pinned_legacy_reserve_still_boots_but_is_deprecated(caplog):
    args = make_args(rank_auto_reserve_mib="5500,3800,3800")
    with caplog.at_level("WARNING"):
        args._check_vram_ledger_flags()  # must NOT raise: recipes keep booting
    assert any("DEPRECATED" in r.message for r in caplog.records)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "conflates two different quantities" in text
    assert "This boot proceeds unchanged" in text


def test_default_path_emits_no_deprecation_and_no_error(caplog):
    args = make_args()
    with caplog.at_level("WARNING"):
        args._check_vram_ledger_flags()
    assert not [r for r in caplog.records if "DEPRECATED" in r.message]


# --- per-card resolution ----------------------------------------------------


def test_scalar_user_reserve_applies_to_every_card():
    args = make_args(rank_user_reserve_mib=2048, enable_vram_ledger=True)
    assert args.user_reserve_mib_per_gpu([0, 1, 2]) == {0: 2048, 1: 2048, 2: 2048}


def test_per_rank_user_reserve_collapses_to_the_largest_per_card():
    """The headroom belongs to the card: two co-located ranks asking for
    different amounts of free memory are asking for the same bytes twice."""
    args = make_args(rank_user_reserve_mib="512,2048,1024", enable_vram_ledger=True)
    assert args.user_reserve_mib_per_gpu([0, 0, 1]) == {0: 2048, 1: 1024}


def test_user_reserve_length_mismatch_is_refused():
    args = make_args(rank_user_reserve_mib="512,2048", enable_vram_ledger=True)
    with pytest.raises(ValueError) as excinfo:
        args.user_reserve_mib_per_gpu([0, 1, 2])
    assert "2 entries but 3 ranks" in str(excinfo.value)


def test_negative_user_reserve_is_refused():
    args = make_args(rank_user_reserve_mib="-1", enable_vram_ledger=True)
    with pytest.raises(ValueError):
        args.user_reserve_mib_per_gpu([0])


# --- calibration cache: keyed, versioned, never silently reused -------------


def _profile(fingerprint):
    return CalibrationProfile(
        fingerprint=fingerprint,
        driver="580.00",
        build="torch2.9+cuda13",
        cards=(
            CardResidual(
                uuid="GPU-a",
                name="RTX 3080",
                cuda_context_bytes=300 << 20,
                allocator_granularity_bytes=8 << 20,
                lazy_workspace_bytes=100 << 20,
            ),
        ),
    )


def _inventory():
    return ([{"uuid": "GPU-a", "cuda_index": 0, "name": "RTX 3080"}], "580.00")


def test_fingerprint_changes_with_card_set_driver_and_build():
    base = calibration_fingerprint(["a", "b"], "580.00", "torch2.9")
    assert calibration_fingerprint(["a", "b", "c"], "580.00", "torch2.9") != base
    assert calibration_fingerprint(["a", "b"], "581.00", "torch2.9") != base
    assert calibration_fingerprint(["a", "b"], "580.00", "torch3.0") != base
    # Order must not matter: the card SET is the input, not its enumeration.
    assert calibration_fingerprint(["b", "a"], "580.00", "torch2.9") == base


def test_calibration_roundtrips_under_its_own_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.mem_ledger.calibration._build_id", lambda: "torch2.9+cuda13"
    )
    fingerprint, _gpus, _driver = __import__(
        "sglang.srt.mem_ledger.calibration", fromlist=["live_fingerprint"]
    ).live_fingerprint(inventory=_inventory())
    save_calibration(_profile(fingerprint), cache_dir=str(tmp_path))
    loaded = load_calibration(cache_dir=str(tmp_path), inventory=_inventory())
    assert loaded is not None
    assert loaded.residual("GPU-a").total_mib == 408


def test_a_calibration_from_another_rig_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.mem_ledger.calibration._build_id", lambda: "torch2.9+cuda13"
    )
    save_calibration(_profile("someoneelse"), cache_dir=str(tmp_path))
    assert load_calibration(cache_dir=str(tmp_path), inventory=_inventory()) is None


def test_a_version_bump_invalidates_rather_than_reinterprets(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.mem_ledger.calibration._build_id", lambda: "torch2.9+cuda13"
    )
    calib = __import__(
        "sglang.srt.mem_ledger.calibration", fromlist=["live_fingerprint"]
    )
    fingerprint, _gpus, _driver = calib.live_fingerprint(inventory=_inventory())
    path = calibration_cache_path(fingerprint, str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = _profile(fingerprint).to_json()
    payload["version"] = VRAM_CALIBRATION_VERSION + 1
    with open(path, "w") as f:
        json.dump(payload, f)
    assert load_calibration(cache_dir=str(tmp_path), inventory=_inventory()) is None


# --- the boot path actually runs -------------------------------------------


def _install_footprints(monkeypatch, *, activation_mib, capture_mib):
    """Make the phase-footprint lookup succeed with stated numbers.

    The boot path resolves activation and capture from the calibration store,
    not from the inherited heuristics, so a hermetic boot test has to supply
    them. Injecting here rather than monkeypatching the old ServerArgs methods
    is the point: those methods are no longer on the ledger's path at all.
    """
    from sglang.srt.mem_ledger.activation import FootprintProvenance, PhaseFootprint

    def fake(card_uuid, *, hw_fingerprint, profile, cache_dir=None):
        return PhaseFootprint(
            activation_mib=activation_mib,
            capture_mib=capture_mib,
            provenance=FootprintProvenance.MEASURED_PEAK,
            source="test-injected footprint",
            card_uuid=card_uuid,
        )

    monkeypatch.setattr(
        "sglang.srt.mem_ledger.activation.resolve_phase_footprint", fake
    )
    monkeypatch.setattr(
        "sglang.srt.mem_ledger.calibration.live_fingerprint",
        lambda **kw: ("testfp000000", [], "drv"),
    )


def test_the_boot_path_forms_budgets_from_the_ledger(monkeypatch, caplog):
    """EXECUTION SMOKE for the wiring, not just for the ledger.

    ``_vram_ledger_non_kv_per_gpu`` is the one function that joins the ledger to
    the boot, and a joint that is only ever read is not a joint that works. NVML
    and the calibration are injected; everything between them is production
    code, including the itemization the boot logs.

    THE LAUNCH RUNS ON BARLINK, and that is load-bearing rather than incidental
    (#598). #595 gave the NCCL communicator buffers a term that is UNBOUNDED
    until somebody measures them, and no seam populates that measurement from
    ServerArgs -- so from that point on this production path REFUSED for every
    launch, and this test went red with it. It is resolvable in exactly one
    way today: on a barlink-owned group no PyNccl communicator is constructed
    at all, so the term prices 0 with the skip condition as its derivation.
    Setting the switch here therefore also pins that the group description
    reaches the ledger through the real ``DemandInputs.from_server_args``,
    which nothing else exercises end to end.
    """
    import sglang.srt.server_args as sa
    from sglang.srt.mem_ledger.calibration import CalibrationProfile, CardResidual

    monkeypatch.setenv("SGLANG_BARLINK", "1")

    class Card:
        def __init__(self, ordinal, uuid, name, total):
            self.cuda_ordinal = ordinal
            self.nvml_index = ordinal
            self.uuid = uuid
            self.pci_bus_id = f"0000:0{ordinal}:00.0"
            self.name = name
            self.total_mib = total
            self.free_mib = total
            # #602: the boot path prices the NVML carve-out from this field.
            # A realistic non-zero value, so the budget this test forms is the
            # one the boot would form.
            self.reserved_mib = 425

    cards = {
        0: Card(0, "GPU-5090", "RTX 5090", 32768),
        1: Card(1, "GPU-3080-a", "RTX 3080", 20480),
    }
    monkeypatch.setattr(
        sa, "_resolve_rank_gpu_cards", lambda ids: {i: cards[i] for i in set(ids)}
    )
    monkeypatch.setattr(
        "sglang.srt.mem_ledger.calibration.load_calibration",
        lambda **kw: CalibrationProfile(
            fingerprint="fp0",
            driver="580",
            build="torch2.9",
            cards=tuple(
                CardResidual(
                    uuid=c.uuid,
                    name=c.name,
                    cuda_context_bytes=300 << 20,
                    allocator_granularity_bytes=8 << 20,
                    lazy_workspace_bytes=100 << 20,
                )
                for c in cards.values()
            ),
        ),
    )

    args = make_args(
        enable_vram_ledger=True,
        rank_gpu_id=[0, 1],
        chunked_prefill_size=2048,
        context_length=None,
        max_running_requests=4,
        tp_size=2,
        pp_size=1,
        max_prefill_tokens=16384,
        disaggregation_mode="null",
        speculative_num_draft_tokens=None,
    )
    # The demand derivations that need a checkpoint are the ones a hermetic
    # test cannot supply; stub exactly those and let the rest run for real.
    monkeypatch.setattr(type(args), "speculative_capture_tokens", lambda s, n=None: 96)
    _install_footprints(monkeypatch, activation_mib=1766, capture_mib=640)
    monkeypatch.setattr(type(args), "gdn_prefill_scratch_mib", lambda s, share: None)
    monkeypatch.setattr(type(args), "dsv4_indexer_prefill_scratch_mib", lambda s: None)
    monkeypatch.setattr(type(args), "ladder_reserve_gpu_id", lambda s: None)

    from collections import Counter

    with caplog.at_level("INFO"):
        non_kv = args._vram_ledger_non_kv_per_gpu(Counter([0, 1]))

    # 1024 user reserve + 1766 activation + 640 capture + 384 flashinfer
    # workspace + 408 hardware residual + 425 NVML carve-out + 70 load
    # transient = 4717.
    #
    # Was 5976 before the phase footprints landed: activation 3968 (the
    # falsified heuristic) and capture 192 (the token estimate the same window
    # measured 3.3-3.8x low). The net goes back to the KV pool, and that is
    # the whole point of the term fix.
    #
    # #602 added the last term, +425: the MiB the driver reserves out of the
    # card's nominal total and never allocates. It is charged ONCE per card,
    # which is why both cards move by 425 and not by 425 x ranks. Unlike the
    # corrections above this one makes the budget SMALLER on purpose -- the
    # KV pool had been sized against memory that does not exist.
    #
    # #612 added the load transient, +70 per RANK (one rank per card here, so
    # +70 per card): the allocator peak above the resident set that the
    # 2026-08-06 corridor window saw the free-memory floor dip into and that no
    # term charged. Same direction as the carve-out and for the same reason --
    # the budget was being formed against memory the boot does not keep.
    assert non_kv == {0: 4717, 1: 4717}
    text = caplog.text
    assert "attention workspaces (capped)" in text
    assert "SGLANG_FLASHINFER_WORKSPACE_SIZE" in text
    assert "VRAM ledger for GPU 0 (RTX 5090" in text
    assert "user reserve (external)" in text
    assert "calibrated@fp0" in text
    assert "VRAM ledger totals" in text


def test_the_boot_path_refuses_an_overcommitted_card(monkeypatch):
    """The refusal is reachable from the boot, not only from the unit.

    Barlink-owned for the same reason as the test above: the refusal under
    test is an OVERCOMMIT (the card is too small), and it can only be observed
    once the unrelated #595 NCCL refusal is out of the way. Verdicts of
    different kinds must not stand in for each other.
    """
    import sglang.srt.server_args as sa
    from sglang.srt.mem_ledger.calibration import CalibrationProfile, CardResidual
    from sglang.srt.mem_ledger.terms import LedgerOvercommit

    monkeypatch.setenv("SGLANG_BARLINK", "1")

    class Card:
        cuda_ordinal = 1
        nvml_index = 1
        uuid = "GPU-3080-a"
        pci_bus_id = "0000:01:00.0"
        name = "RTX 3080"
        total_mib = 20480
        free_mib = 20480
        reserved_mib = 425  # #602: see the sibling stub above

    monkeypatch.setattr(sa, "_resolve_rank_gpu_cards", lambda ids: {1: Card()})
    monkeypatch.setattr(
        "sglang.srt.mem_ledger.calibration.load_calibration",
        lambda **kw: CalibrationProfile(
            fingerprint="fp0",
            driver="580",
            build="torch2.9",
            cards=(
                CardResidual(
                    uuid="GPU-3080-a",
                    name="RTX 3080",
                    cuda_context_bytes=300 << 20,
                    allocator_granularity_bytes=8 << 20,
                    lazy_workspace_bytes=100 << 20,
                ),
            ),
        ),
    )
    args = make_args(
        enable_vram_ledger=True,
        rank_gpu_id=[1, 1, 1],
        chunked_prefill_size=8192,
        context_length=None,
        max_running_requests=4,
        tp_size=3,
        pp_size=1,
        max_prefill_tokens=16384,
        disaggregation_mode="null",
        speculative_num_draft_tokens=None,
    )
    monkeypatch.setattr(type(args), "speculative_capture_tokens", lambda s, n=None: 512)
    _install_footprints(monkeypatch, activation_mib=8000, capture_mib=640)
    monkeypatch.setattr(type(args), "gdn_prefill_scratch_mib", lambda s, share: None)
    monkeypatch.setattr(type(args), "dsv4_indexer_prefill_scratch_mib", lambda s: None)
    monkeypatch.setattr(type(args), "ladder_reserve_gpu_id", lambda s: None)

    from collections import Counter

    with pytest.raises(LedgerOvercommit) as excinfo:
        args._vram_ledger_non_kv_per_gpu(Counter([1, 1, 1]))
    assert "OVERCOMMITTED by" in str(excinfo.value)
    assert "runtime activation + metadata" in str(excinfo.value)
