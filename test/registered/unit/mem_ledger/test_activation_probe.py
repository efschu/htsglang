"""Hermetic tests for the activation probe and its ingest half.

The probe itself needs an instrumented boot, which is a GPU window. Everything
around it -- arming, dump format, ingest, refusal on inconsistent input -- is
provable at the desk and is proved here, so the window is spent on the
measurement and not on discovering a typo in the ingest path.
"""

import importlib.util
import json
import os


from sglang.srt.mem_ledger import activation_probe as ap
from sglang.srt.mem_ledger.activation import (
    ActivationProfile,
    FootprintProvenance,
    load_footprints,
)

_HERE = os.path.abspath(__file__)
_ROOT = _HERE
for _ in range(5):
    _ROOT = os.path.dirname(_ROOT)
SCRIPT = os.path.join(_ROOT, "scripts", "vram_ledger", "probe_activation.py")
assert os.path.isfile(SCRIPT), SCRIPT

PROFILE = ActivationProfile(
    architectures=("Qwen3_5ForConditionalGeneration",),
    chunked_prefill_size=2048,
    tp_size=3,
    pp_size=1,
    kv_cache_dtype="fp8_e4m3",
    speculative_num_draft_tokens=4,
    decode_max_bs=24,
)
FP = "a191a0712717"


def load_script():
    spec = importlib.util.spec_from_file_location("probe_activation", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write_dump(d, rank, uuid, activation_mib, capture_mib, profile=PROFILE, fp=FP):
    payload = {
        "rank": rank,
        "card_uuid": uuid,
        "hw_fingerprint": fp,
        "profile": profile.canonical(),
        "activation_peak_bytes": activation_mib << 20,
        "capture_bytes": capture_mib << 20,
        "reserved_peak_bytes": (activation_mib + 300) << 20,
        "prefill_tokens": 70018,
    }
    with open(os.path.join(str(d), f"phase_footprint_rank{rank}.json"), "w") as f:
        json.dump(payload, f)


# --- arming -----------------------------------------------------------------


def test_probe_is_inert_unless_armed(monkeypatch):
    monkeypatch.delenv(ap.DUMP_ENV, raising=False)
    assert ap.is_armed() is False
    assert (
        ap.write_footprint_dump(
            rank=0,
            card_uuid="x",
            hw_fingerprint=FP,
            profile_canonical=PROFILE.canonical(),
            activation_peak_bytes=1,
            capture_bytes=1,
        )
        is None
    )


def test_arming_is_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(ap.DUMP_ENV, str(tmp_path))
    assert ap.is_armed() is True
    path = ap.write_footprint_dump(
        rank=2,
        card_uuid="GPU-x",
        hw_fingerprint=FP,
        profile_canonical=PROFILE.canonical(),
        activation_peak_bytes=900 << 20,
        capture_bytes=640 << 20,
        prefill_tokens=70018,
    )
    assert path and os.path.exists(path)
    with open(path) as f:
        d = json.load(f)
    assert d["rank"] == 2
    assert d["activation_peak_bytes"] == 900 << 20


def test_reset_peaks_is_a_noop_when_unarmed(monkeypatch):
    monkeypatch.delenv(ap.DUMP_ENV, raising=False)
    ap.reset_peaks(0)  # must not raise even with no CUDA


# --- ingest -----------------------------------------------------------------


def test_ingest_folds_dumps_into_a_measured_calibration(tmp_path):
    dumps = tmp_path / "dumps"
    cache = tmp_path / "cache"
    dumps.mkdir()
    write_dump(dumps, 0, "GPU-a", 900, 730)
    write_dump(dumps, 1, "GPU-b", 850, 640)
    m = load_script()
    assert m.ingest(str(dumps), str(cache)) == 0

    got = load_footprints(hw_fingerprint=FP, profile=PROFILE, cache_dir=str(cache))
    assert set(got) == {"GPU-a", "GPU-b"}
    assert got["GPU-a"].activation_mib == 900
    assert got["GPU-a"].capture_mib == 730
    assert got["GPU-a"].provenance is FootprintProvenance.MEASURED_PEAK
    # The source must name the instrument, since the whole point is that it is
    # not the instrument the window used.
    assert "memory_stats" in got["GPU-a"].source


def test_ingest_refuses_mixed_profiles(tmp_path, capsys):
    import dataclasses as dc

    dumps = tmp_path / "d"
    dumps.mkdir()
    write_dump(dumps, 0, "GPU-a", 900, 730)
    write_dump(
        dumps,
        1,
        "GPU-b",
        850,
        640,
        profile=dc.replace(PROFILE, chunked_prefill_size=4096),
    )
    m = load_script()
    assert m.ingest(str(dumps), str(tmp_path / "c")) == 1
    assert "more than one activation profile" in capsys.readouterr().out


def test_ingest_refuses_mixed_fingerprints(tmp_path, capsys):
    dumps = tmp_path / "d"
    dumps.mkdir()
    write_dump(dumps, 0, "GPU-a", 900, 730)
    write_dump(dumps, 1, "GPU-b", 850, 640, fp="otherrig")
    m = load_script()
    assert m.ingest(str(dumps), str(tmp_path / "c")) == 1
    assert "inconsistent hardware fingerprints" in capsys.readouterr().out


def test_ingest_refuses_a_nonpositive_activation_peak(tmp_path, capsys):
    """A zero is a FAILED measurement, not a small one -- the hook did not run
    or ran before the workload."""
    dumps = tmp_path / "d"
    dumps.mkdir()
    write_dump(dumps, 0, "GPU-a", 0, 730)
    m = load_script()
    assert m.ingest(str(dumps), str(tmp_path / "c")) == 1
    out = capsys.readouterr().out
    assert "failed measurement, not a small one" in out


def test_ingest_with_no_dumps_explains_how_to_produce_them(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    m = load_script()
    assert m.ingest(str(empty), str(tmp_path / "c")) == 1
    assert "SGLANG_PHASE_FOOTPRINT_DUMP" in capsys.readouterr().out


def test_show_prints_the_shipped_bounds(capsys):
    m = load_script()
    assert m.show() == 0
    out = capsys.readouterr().out
    assert "1766" in out
    assert "UPPER BOUND" in out
    assert "refuses until this probe runs" in out


def test_script_answers_help_as_a_subprocess():
    import subprocess
    import sys

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "99"
    p = subprocess.run(
        [sys.executable, SCRIPT, "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert p.returncode == 0, p.stderr
    assert "usage:" in p.stdout
