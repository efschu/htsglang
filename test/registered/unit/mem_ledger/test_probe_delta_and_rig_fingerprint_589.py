# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#589: the probe dump must carry a DELTA and the RIG's fingerprint.

Two defects from the 2026-08-05 window 5, both of which made the dumps
unusable rather than merely imprecise.

**The peak was absolute.** ``torch.cuda.reset_peak_memory_stats`` does not zero
the peak counter, it re-bases it at whatever is allocated at that moment. After
model load, KV sizing and graph capture that is the rank's whole resident
footprint, so the "activation peak" the window recorded -- 26555 / 17306 /
16368 MiB across the three ranks -- was weights + KV + graphs, not the prefill
transient. Reserving those numbers as an activation term would multiply the
real cost by an order of magnitude. The fix records the floor at every re-base
and dumps ``peak - floor`` alongside the raw figure.

**The fingerprint was per-rank.** Each rank is pinned to one card by
``CUDA_VISIBLE_DEVICES``, and ``live_fingerprint`` hashes the cards the calling
process can see. So the three ranks stamped three different fingerprints
(``fad5762191c9`` / ``9ce0ed6a79fc`` / ``b0c936f23170``) and ingest correctly
refused every one against the rig's ``a191a0712717``. A dump has to name the
rig it was measured on. NVML ignores the mask, so its device list is the rig.

Hermetic: no driver, no CUDA, no torch device. The NVML list and the driver
string are injected, and the memory-stats layer is faked.
"""

import importlib.util
import json
import os

import pytest

from sglang.srt.mem_ledger import activation_probe as ap
from sglang.srt.mem_ledger import calibration as cal
from sglang.srt.mem_ledger.activation import ActivationProfile
from sglang.srt.registry.nvml import DeviceInfo

_HERE = os.path.abspath(__file__)
_ROOT = _HERE
for _ in range(5):
    _ROOT = os.path.dirname(_ROOT)
SCRIPT = os.path.join(_ROOT, "scripts", "vram_ledger", "probe_activation.py")

MIB = 1024**2

UUID_3080_A = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
UUID_5090 = "GPU-bbbbbbbb-0000-0000-0000-000000000002"
UUID_3080_B = "GPU-cccccccc-0000-0000-0000-000000000003"

RIG = [
    DeviceInfo(
        0, UUID_3080_A, "NVIDIA GeForce RTX 3080", 20480 * MIB, "00000000:01:00.0"
    ),
    DeviceInfo(
        1, UUID_5090, "NVIDIA GeForce RTX 5090", 32768 * MIB, "00000000:2D:00.0"
    ),
    DeviceInfo(
        2, UUID_3080_B, "NVIDIA GeForce RTX 3080", 20480 * MIB, "00000000:41:00.0"
    ),
]
DRIVER = "580.65.06"

PROFILE = ActivationProfile(
    architectures=("Qwen3_5ForConditionalGeneration",),
    chunked_prefill_size=2048,
    tp_size=3,
    pp_size=1,
    kv_cache_dtype="fp8_e4m3",
    speculative_num_draft_tokens=4,
    decode_max_bs=24,
)


def load_script():
    spec = importlib.util.spec_from_file_location("probe_activation_589", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_dump(d, rank=0):
    with open(os.path.join(str(d), f"phase_footprint_rank{rank}.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Defect 2: the peak is absolute, the delta is what the ledger reserves for
# ---------------------------------------------------------------------------


@pytest.fixture
def probe(monkeypatch):
    ap._baseline_allocated = None
    ap._capture_bytes_total = 0
    ap._activation_peak_bytes = 0
    ap._identity = None
    ap._peak_floor_bytes = None
    monkeypatch.setattr(ap, "_device_index", lambda: 0)
    yield ap


def _fake_torch_stats(monkeypatch, state):
    """Model the counter's real semantics: reset RE-BASES at current."""
    calls = {"reset": 0}

    def fake_read(_i=0):
        return dict(state)

    def fake_reset(_i=0):
        # This is the defect in one line: the peak becomes the current
        # allocation, not zero.
        calls["reset"] += 1
        state["allocated_peak_bytes"] = state["allocated_bytes"]

    monkeypatch.setattr(ap, "read_peaks", fake_read)
    monkeypatch.setattr(
        ap,
        "reset_peaks",
        lambda i=0: (
            (
                fake_reset(i),
                setattr(ap, "_peak_floor_bytes", int(state["allocated_bytes"])),
            )[0]
            if ap.is_armed()
            else None
        ),
    )
    return calls


def test_the_dump_reports_the_transient_not_the_resident_footprint(
    probe, monkeypatch, tmp_path
):
    """The window-5 shape: a rank holding 16368 MiB whose real prefill
    transient is 368 MiB must dump 368, not 16368."""
    monkeypatch.setenv(ap.DUMP_ENV, str(tmp_path))
    monkeypatch.setattr(
        ap,
        "_resolve_identity",
        lambda _r: {
            "card_uuid": UUID_5090,
            "hw_fingerprint": "a191a0712717",
            "profile_canonical": PROFILE.canonical(),
            "rank": 0,
        },
    )
    state = {"allocated_bytes": 15000 * MIB, "allocated_peak_bytes": 0}
    _fake_torch_stats(monkeypatch, state)

    ap.note_capture_begin()
    state["allocated_bytes"] = 16000 * MIB  # graphs captured
    ap.note_capture_end()

    # The prefill transient: 368 MiB on top of a 16000 MiB resident floor.
    state["allocated_peak_bytes"] = 16368 * MIB
    state["reserved_peak_bytes"] = 16800 * MIB
    ap.record_prefill_peak(object(), 70018)

    d = read_dump(tmp_path)
    assert d["activation_delta_bytes"] is not None, (
        "the dump carries no delta, so the ledger would be handed the "
        f"{d['activation_peak_bytes'] // MIB} MiB absolute figure"
    )
    assert d["activation_delta_bytes"] == 368 * MIB, (
        "the dump must carry the prefill transient; got "
        f"{d['activation_delta_bytes'] // MIB} MiB"
    )
    assert d["peak_floor_bytes"] == 16000 * MIB
    # The raw figure is kept, labelled, so the two can be reconciled against
    # what nvidia-smi showed during the window.
    assert d["activation_peak_bytes"] == 16368 * MIB


def test_a_missing_floor_is_an_absent_delta_not_a_silent_raw_peak(
    probe, monkeypatch, tmp_path
):
    """Falling back to the raw peak would re-introduce the exact over-charge
    the delta exists to prevent, so the absence stays visible."""
    monkeypatch.setenv(ap.DUMP_ENV, str(tmp_path))
    monkeypatch.setattr(
        ap,
        "_resolve_identity",
        lambda _r: {
            "card_uuid": UUID_5090,
            "hw_fingerprint": "a191a0712717",
            "profile_canonical": PROFILE.canonical(),
            "rank": 0,
        },
    )
    monkeypatch.setattr(
        ap, "read_peaks", lambda _i=0: {"allocated_peak_bytes": 26555 * MIB}
    )
    # No capture bracket ran, so no floor was ever recorded.
    ap.record_prefill_peak(object(), 70018)

    d = read_dump(tmp_path)
    assert d["activation_delta_bytes"] is None
    assert d["peak_floor_bytes"] is None
    assert d["activation_peak_bytes"] == 26555 * MIB


def test_reset_peaks_records_the_floor_it_rebased_to(probe, monkeypatch):
    """The invariant the comment claims, pinned: a re-base is only safe if the
    number it re-based to is written down."""
    monkeypatch.setenv(ap.DUMP_ENV, "/tmp/does-not-need-to-exist-589")
    monkeypatch.setattr(ap, "read_peaks", lambda _i=0: {"allocated_bytes": 4242 * MIB})

    class _FakeCuda:
        @staticmethod
        def reset_peak_memory_stats(_i):
            return None

    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = _FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    ap._peak_floor_bytes = None
    ap.reset_peaks(0)
    assert ap._peak_floor_bytes == 4242 * MIB


def test_ingest_refuses_a_dump_that_carries_no_delta(tmp_path, capsys):
    """A pre-fix dump cannot be repaired at ingest -- the floor it would need
    was never recorded -- so it must be refused rather than folded in."""
    dumps = tmp_path / "d"
    dumps.mkdir()
    payload = {
        "rank": 0,
        "card_uuid": UUID_5090,
        "hw_fingerprint": "a191a0712717",
        "profile": PROFILE.canonical(),
        "activation_peak_bytes": 26555 * MIB,  # the window-5 absolute figure
        "capture_bytes": 640 * MIB,
        "reserved_peak_bytes": 27000 * MIB,
        "prefill_tokens": 70018,
    }
    with open(dumps / "phase_footprint_rank0.json", "w") as f:
        json.dump(payload, f)

    m = load_script()
    assert m.ingest(str(dumps), str(tmp_path / "c")) == 1
    out = capsys.readouterr().out
    assert "activation_delta_bytes" in out
    assert "26555" in out


# ---------------------------------------------------------------------------
# Defect 3: the fingerprint must name the rig, not the rank's slice of it
# ---------------------------------------------------------------------------


def _patch_nvml(monkeypatch, devices):
    from sglang.srt.registry import nvml as registry_nvml

    monkeypatch.setattr(registry_nvml, "list_devices", lambda: list(devices))
    monkeypatch.setattr(registry_nvml, "driver_version", lambda: DRIVER)


def test_the_rig_fingerprint_is_identical_in_every_pinned_rank(monkeypatch):
    """The falsifier. NVML sees all three cards regardless of the mask, so
    three pinned ranks must agree -- and must agree with the launcher."""
    _patch_nvml(monkeypatch, RIG)
    answers = set()
    for pin in ("0", "1", "2"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", pin)
        got = cal.rig_fingerprint()
        assert got is not None
        answers.add(got[0])
    assert len(answers) == 1, f"ranks disagreed about the rig: {answers}"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    unmasked = cal.rig_fingerprint()
    assert unmasked[0] == next(iter(answers)), (
        "a pinned rank and the unmasked launcher must compute the same rig "
        "fingerprint, or ingest refuses every dump"
    )


def test_the_per_rank_fingerprint_is_actually_different(monkeypatch):
    """Proves the test above can fail: hashing one card really does produce a
    different fingerprint, which is why the window's dumps were refused."""
    _patch_nvml(monkeypatch, RIG)
    whole = cal.rig_fingerprint()[0]
    per_rank = {
        cal.live_fingerprint(
            inventory=(
                [{"uuid": d.uuid, "name": d.name, "total_mib": d.total_mib}],
                DRIVER,
            )
        )[0]
        for d in RIG
    }
    assert len(per_rank) == 3
    assert whole not in per_rank


def test_the_rig_fingerprint_reuses_the_ledger_hash(monkeypatch):
    """No second hashing implementation: the answer must equal what
    calibration_fingerprint gives for the same card set and driver."""
    _patch_nvml(monkeypatch, RIG)
    expected = cal.calibration_fingerprint([d.uuid for d in RIG], DRIVER)
    assert cal.rig_fingerprint()[0] == expected


def test_enumeration_order_cannot_change_the_rig_fingerprint(monkeypatch):
    """The UUIDs are sorted before hashing, so an NVML reorder across a boot
    does not invalidate a calibration."""
    _patch_nvml(monkeypatch, RIG)
    forward = cal.rig_fingerprint()[0]
    shuffled = [
        DeviceInfo(0, RIG[2].uuid, RIG[2].name, RIG[2].total_bytes, RIG[2].pci_bus_id),
        DeviceInfo(1, RIG[0].uuid, RIG[0].name, RIG[0].total_bytes, RIG[0].pci_bus_id),
        DeviceInfo(2, RIG[1].uuid, RIG[1].name, RIG[1].total_bytes, RIG[1].pci_bus_id),
    ]
    _patch_nvml(monkeypatch, shuffled)
    assert cal.rig_fingerprint()[0] == forward


def test_no_nvml_is_an_honest_none(monkeypatch):
    from sglang.srt.registry import nvml as registry_nvml

    def boom():
        raise registry_nvml.NvmlUnavailableError("no driver here")

    monkeypatch.setattr(registry_nvml, "list_devices", boom)
    assert cal.rig_fingerprint() is None


def test_the_probe_stamps_the_rig_fingerprint_into_the_dump(monkeypatch, tmp_path):
    """End of the chain: what _resolve_identity puts in the dump is the rig's
    fingerprint, so ingest accepts it."""
    _patch_nvml(monkeypatch, RIG)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv(ap.DUMP_ENV, str(tmp_path))
    ap._identity = None

    from sglang.srt.registry import nvml as registry_nvml

    monkeypatch.setattr(registry_nvml, "current_device_uuid", lambda: UUID_5090)
    monkeypatch.setattr(
        "sglang.srt.mem_ledger.engine._model_architectures", lambda _sa: ("X",)
    )
    monkeypatch.setattr(
        "sglang.srt.mem_ledger.activation.profile_from_server_args",
        lambda _sa, _arch: PROFILE,
    )

    class _Runner:
        server_args = object()
        tp_rank = 0

    identity = ap._resolve_identity(_Runner())
    assert identity is not None
    assert identity["hw_fingerprint"] == cal.calibration_fingerprint(
        [d.uuid for d in RIG], DRIVER
    )
    assert identity["card_uuid"] == UUID_5090
