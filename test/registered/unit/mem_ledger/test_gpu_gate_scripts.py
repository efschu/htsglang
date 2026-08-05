"""Mock smokes for the two GPU-gate scripts.

Neither script may be executed for the first time inside a GPU window: a window
is expensive and a NameError found there costs the whole slot. Everything below
runs with CUDA_VISIBLE_DEVICES=99 and injected data, so what the window
exercises is the measurement, not the plumbing.

What these tests deliberately do NOT prove: the residual numbers themselves,
and whether a card can be measured in the time box. Those are the window's job
and are listed as open in the ticket.
"""

import csv
import importlib.util
import os
import subprocess
import sys

import pytest

_HERE = os.path.abspath(__file__)
# test/registered/unit/mem_ledger/<file> -> repo root is five levels up.
_REPO_ROOT = _HERE
for _ in range(5):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
SCRIPTS = os.path.join(_REPO_ROOT, "scripts", "vram_ledger")
assert os.path.isdir(SCRIPTS), SCRIPTS


def load(name):
    path = os.path.join(SCRIPTS, name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def calibrate():
    return load("calibrate_cards.py")


@pytest.fixture(scope="module")
def compare_mod():
    return load("compare_boot_peaks.py")


# --- both scripts at least import and expose a CLI --------------------------


def test_both_scripts_import_without_a_gpu(calibrate, compare_mod):
    assert callable(calibrate.main)
    assert callable(compare_mod.main)


@pytest.mark.parametrize("script", ["calibrate_cards.py", "compare_boot_peaks.py"])
def test_scripts_answer_help_as_a_subprocess(script):
    """The real entry point, really executed. --help exercises argparse
    construction, which is where a typo in a default hides."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "99"
    proc = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, script), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout


# --- calibrate_cards ---------------------------------------------------------


def test_calibration_fingerprint_matches_what_the_boot_will_look_up(monkeypatch):
    """THE TRAP THIS PINS: the script assembles the cache key itself. If it
    read the card set or the driver from a different source than
    live_fingerprint does, it would write a perfectly good calibration under a
    key the boot never looks up -- the cache would miss forever while a
    valid-looking file sat next to it."""
    from sglang.srt.mem_ledger import calibration as calib

    inventory = (
        [
            {"uuid": "GPU-aaa", "cuda_index": 0, "name": "RTX 5090"},
            {"uuid": "GPU-bbb", "cuda_index": 1, "name": "RTX 3080"},
        ],
        "580.00",
    )
    monkeypatch.setattr(calib, "_build_id", lambda: "torch2.9+cuda13")
    monkeypatch.setattr("sglang.srt.rigmon.card_probe._inventory", lambda: inventory)

    script = load("calibrate_cards.py")
    cards, driver = script.resolve_cards(allow_cuda_init=False)
    script_fp = calib.calibration_fingerprint([c["uuid"] for c in cards], driver)

    boot_fp, _gpus, _driver = calib.live_fingerprint()
    assert script_fp == boot_fp


def test_resolve_cards_survives_a_missing_identity_map(monkeypatch):
    """The BDF is how a human recognises a card, not how the script addresses
    it; losing the map must degrade the display, not the run."""
    from sglang.srt.mem_ledger import calibration as calib  # noqa: F401

    monkeypatch.setattr(
        "sglang.srt.rigmon.card_probe._inventory",
        lambda: ([{"uuid": "GPU-aaa", "cuda_index": 0, "name": "RTX 5090"}], "580"),
    )
    monkeypatch.setattr(
        "sglang.srt.registry.nvml.identity_map",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("no nvml")),
    )
    script = load("calibrate_cards.py")
    cards, driver = script.resolve_cards(allow_cuda_init=False)
    assert cards[0]["pci_bus_id"] == "?"
    assert cards[0]["uuid"] == "GPU-aaa"
    assert driver == "580"


def test_no_cards_is_a_named_failure_not_an_empty_success(monkeypatch):
    monkeypatch.setattr("sglang.srt.rigmon.card_probe._inventory", lambda: ([], None))
    script = load("calibrate_cards.py")
    with pytest.raises(RuntimeError) as excinfo:
        script.resolve_cards(allow_cuda_init=False)
    assert "unbounded at the next boot" in str(excinfo.value)


def test_child_is_pinned_by_uuid_never_by_index(calibrate, monkeypatch):
    """The device-order trap, pinned. CUDA_VISIBLE_DEVICES must carry the UUID:
    an index here is #349 sweep-3 arm L, where a budget was accepted against
    one card and the rank then bound another."""
    seen = {}

    class Result:
        returncode = 0
        stdout = (
            '{"uuid": "GPU-aaa", "name": "RTX 5090", "cuda_context_bytes": 1, '
            '"allocator_granularity_bytes": 2, "lazy_workspace_bytes": 3, '
            '"note": ""}'
        )
        stderr = ""

    def fake_run(cmd, **kwargs):
        seen["env"] = kwargs.get("env", {})
        seen["cmd"] = cmd
        seen["timeout"] = kwargs.get("timeout")
        return Result()

    monkeypatch.setattr(calibrate.subprocess, "run", fake_run)
    card = {
        "uuid": "GPU-aaa",
        "name": "RTX 5090",
        "pci_bus_id": "0000:01:00.0",
        "nvml_index": 1,
        "cuda_ordinal": 0,
    }
    payload, error = calibrate.measure_one_in_subprocess(card, timeout_s=42)
    assert error is None
    assert payload["cuda_context_bytes"] == 1
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-aaa"
    assert not seen["env"]["CUDA_VISIBLE_DEVICES"].isdigit()
    assert seen["timeout"] == 42


def test_a_hung_card_is_a_named_timeout_not_a_hang(calibrate, monkeypatch):
    def fake_run(cmd, **kwargs):
        raise calibrate.subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

    monkeypatch.setattr(calibrate.subprocess, "run", fake_run)
    payload, error = calibrate.measure_one_in_subprocess(
        {
            "uuid": "GPU-aaa",
            "name": "RTX 5090",
            "pci_bus_id": "0000:01:00.0",
            "nvml_index": 1,
            "cuda_ordinal": 0,
        },
        timeout_s=7,
    )
    assert payload is None
    assert "TIMEOUT after 7s" in error
    assert "RTX 5090" in error and "0000:01:00.0" in error
    # It must also say that a longer timeout is not the fix.
    assert "does not unwedge a context" in error


def test_a_failing_child_is_a_named_failure(calibrate, monkeypatch):
    class Result:
        returncode = 3
        stdout = '{"uuid": "GPU-aaa", "error": "CUDA out of memory"}'
        stderr = ""

    monkeypatch.setattr(calibrate.subprocess, "run", lambda cmd, **kw: Result())
    payload, error = calibrate.measure_one_in_subprocess(
        {
            "uuid": "GPU-aaa",
            "name": "RTX 5090",
            "pci_bus_id": "0000:01:00.0",
            "nvml_index": 1,
            "cuda_ordinal": 0,
        },
        timeout_s=5,
    )
    assert payload is None
    assert "FAILED on RTX 5090" in error


def test_a_partial_measurement_writes_nothing(calibrate, monkeypatch, tmp_path, capsys):
    """One bad card must not produce a cache file: a profile missing a card
    makes that card's term unbounded later, which reads as a ledger bug rather
    than as this incomplete run."""
    monkeypatch.setattr(
        "sglang.srt.rigmon.card_probe._inventory",
        lambda: (
            [
                {"uuid": "GPU-aaa", "cuda_index": 0, "name": "RTX 5090"},
                {"uuid": "GPU-bbb", "cuda_index": 1, "name": "RTX 3080"},
            ],
            "580",
        ),
    )
    monkeypatch.setattr(
        "sglang.srt.registry.nvml.identity_map",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("no map")),
    )

    def one_ok_one_bad(card, timeout_s, python=None):
        if card["uuid"] == "GPU-aaa":
            return (
                {
                    "uuid": "GPU-aaa",
                    "name": "RTX 5090",
                    "cuda_context_bytes": 300 << 20,
                    "allocator_granularity_bytes": 8 << 20,
                    "lazy_workspace_bytes": 100 << 20,
                    "note": "",
                    "_elapsed_s": 1.0,
                },
                None,
            )
        return None, "TIMEOUT after 5s on RTX 3080"

    monkeypatch.setattr(calibrate, "measure_one_in_subprocess", one_ok_one_bad)
    rc = calibrate.main(["--cache-dir", str(tmp_path), "--timeout", "5"])
    assert rc == 1
    assert "REFUSING to write a partial calibration" in capsys.readouterr().out
    assert not list(tmp_path.glob("vram_calibration-*.json"))


def test_dry_run_touches_no_gpu(calibrate, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "sglang.srt.rigmon.card_probe._inventory",
        lambda: ([{"uuid": "GPU-aaa", "cuda_index": 0, "name": "RTX 5090"}], "580"),
    )
    monkeypatch.setattr(
        "sglang.srt.registry.nvml.identity_map",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("no map")),
    )

    def explode(*a, **k):
        raise AssertionError("--dry-run must not measure")

    monkeypatch.setattr(calibrate, "measure_one_in_subprocess", explode)
    assert calibrate.main(["--dry-run", "--cache-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "no GPU was touched" in out
    assert "Fingerprint" in out


# --- compare_boot_peaks ------------------------------------------------------


LEGACY_LOG = """\
2026-08-05 09:00:00 INFO --rank-tp-ratio auto: derived memory budgets [26000, 15000, 15000] MiB from NVML totals (reserve per GPU: {0: 5500, 1: 4200, 2: 4200}).
2026-08-05 09:00:01 WARNING --rank-auto-reserve-mib pins 3800 MiB on GPU 1, below the 4160 MiB that 'auto' would derive for it (short by 360 MiB). A pinned value replaces the derived demand model outright.
2026-08-05 09:00:02 INFO chunked_prefill_size=2048
2026-08-05 09:02:00 INFO max_total_num_tokens=90624
"""

LEDGER_LOG = """\
2026-08-05 10:00:00 INFO VRAM ledger for GPU 1 (RTX 3080, NVML total 20480 MiB) (ranks: 1): 20480 MiB total -- FITS, KV pool 14504 MiB
2026-08-05 10:00:00 INFO   user reserve (external)             1024 MiB  operator        external headroom
2026-08-05 10:00:00 INFO   runtime activation + metadata       3968 MiB  modeled         mamba_pre_capture_reserve_mb() x 1 rank
2026-08-05 10:00:00 INFO   CUDA graph capture                   192 MiB  modeled         96 captured tokens
2026-08-05 10:00:00 INFO   hardware residual (per process)      408 MiB  calibrated@fp0  measured on RTX 3080

2026-08-05 10:02:00 INFO max_total_num_tokens=90624
"""


def write_samples(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "uuid", "name", "used_mib", "total_mib"])
        writer.writerows(rows)


def test_parses_a_legacy_boot_log_including_the_shortfall_warning(
    compare_mod, tmp_path
):
    log = tmp_path / "boot.log"
    log.write_text(LEGACY_LOG)
    facts = compare_mod.parse_boot_log(str(log))
    assert facts["budgets_mib"] == [26000, 15000, 15000]
    assert facts["reserve_per_gpu"] == {0: 5500, 1: 4200, 2: 4200}
    assert facts["chunked_prefill_size"] == 2048
    assert facts["max_total_num_tokens"] == 90624
    assert len(facts["shortfall_warnings"]) == 1
    assert "short by 360 MiB" in facts["shortfall_warnings"][0]
    assert facts["ledger_cards"] == []


def test_parses_a_ledger_boot_log_itemization(compare_mod, tmp_path):
    log = tmp_path / "boot.log"
    log.write_text(LEDGER_LOG)
    facts = compare_mod.parse_boot_log(str(log))
    assert len(facts["ledger_cards"]) == 1
    card = facts["ledger_cards"][0]
    assert card["gpu_id"] == 1
    assert card["total_mib"] == 20480
    names = {r["name"]: r["mib"] for r in card["rows"]}
    assert names["runtime activation + metadata"] == 3968
    assert names["hardware residual (per process)"] == 408
    assert not facts["shortfall_warnings"]


def test_parses_the_production_recipe_reserve_vector(compare_mod, tmp_path):
    recipe = tmp_path / "start.sh"
    recipe.write_text(
        'RESERVE="${RESERVE:-5500,4200,4200}"\n'
        "  --tp-size 3 --rank-gpu-id 0,1,2 \\\n"
        "  --context-length 262144 \\\n"
        "  --max-mamba-cache-size 96 \\\n"
    )
    facts = compare_mod.parse_recipe(str(recipe))
    assert facts["reserve"] == "5500,4200,4200"
    assert facts["flags"]["--tp-size"] == "3"
    assert facts["flags"]["--rank-gpu-id"] == "0,1,2"
    assert facts["flags"]["--max-mamba-cache-size"] == "96"


def test_a_missing_recipe_is_absent_not_invented(compare_mod):
    facts = compare_mod.parse_recipe("/nonexistent/start.sh")
    assert facts["reserve"] is None
    assert facts["flags"] == {}


def test_phase_peaks_reports_a_transient_peak_and_a_steady_level(compare_mod):
    """The transient must not be smoothed into the steady level, and the steady
    level must not be read off a sample that landed inside a transient."""
    series = [(float(i), 10000, 20480) for i in range(80)]
    series += [(80.0, 19900, 20480)]  # the #493-shaped transient
    series += [(float(i), 10000, 20480) for i in range(81, 100)]
    stats = compare_mod.phase_peaks(series)
    assert stats["peak_mib"] == 19900
    assert stats["steady_mib"] == 10000
    assert stats["total_mib"] == 20480


def test_compare_runs_end_to_end_on_a_legacy_boot(compare_mod, tmp_path, capsys):
    log = tmp_path / "boot.log"
    log.write_text(LEGACY_LOG)
    samples = tmp_path / "s.csv"
    write_samples(
        samples,
        [
            (0.0, "GPU-bbb", "RTX 3080", 9000, 20480),
            (0.1, "GPU-bbb", "RTX 3080", 19900, 20480),
            (0.2, "GPU-bbb", "RTX 3080", 12000, 20480),
        ],
    )
    rc = compare_mod.main(
        [
            "compare",
            "--boot-log",
            str(log),
            "--samples",
            str(samples),
            "--recipe",
            "/nonexistent",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "19900" in out
    assert "short by 360 MiB" in out
    assert "THE DEFECT CLASS THIS WORK REMOVES" in out
    assert "NO LEDGER ITEMIZATION IN THIS LOG" in out


def test_compare_emits_a_per_term_table_on_a_ledger_boot(compare_mod, tmp_path, capsys):
    log = tmp_path / "boot.log"
    log.write_text(LEDGER_LOG)
    samples = tmp_path / "s.csv"
    write_samples(
        samples,
        [
            (0.0, "GPU-bbb", "RTX 3080", 9000, 20480),
            (0.1, "GPU-bbb", "RTX 3080", 19100, 20480),
        ],
    )
    rc = compare_mod.main(
        ["compare", "--boot-log", str(log), "--samples", str(samples), "--recipe", "/x"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "PER-TERM DELTA" in out
    assert "runtime activation + metadata" in out
    # user reserve is external and must NOT be inside the internal demand sum
    assert "predicted internal demand" in out
    assert "4568 MiB" in out  # 3968 + 192 + 408, excluding the 1024 reserve
    assert "unexplained" in out


def test_compare_reports_missing_samples_rather_than_inventing_them(
    compare_mod, tmp_path, capsys
):
    log = tmp_path / "boot.log"
    log.write_text(LEDGER_LOG)
    samples = tmp_path / "s.csv"
    write_samples(samples, [(0.0, "GPU-zzz", "Some Other Card", 100, 8192)])
    rc = compare_mod.main(
        ["compare", "--boot-log", str(log), "--samples", str(samples), "--recipe", "/x"]
    )
    out = capsys.readouterr().out
    assert rc == 2
    assert "no samples matched" in out


def test_sample_loop_survives_a_failing_poll(compare_mod, tmp_path, monkeypatch):
    """A driver hiccup during a boot is exactly when the samples matter."""
    calls = {"n": 0}

    def flaky(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("nvidia-smi transient")

        class R:
            stdout = "GPU-aaa, RTX 5090, 1234, 32768\n"

        return R()

    monkeypatch.setattr(compare_mod.subprocess, "run", flaky)
    monkeypatch.setattr(compare_mod.time, "sleep", lambda s: None)
    out = tmp_path / "s.csv"
    compare_mod.sample_loop(str(out), interval=0.0, duration=0.0)
    assert calls["n"] >= 1


def test_parser_round_trips_the_REAL_ledger_renderer(compare_mod, tmp_path):
    """The strongest form of this test: parse what the ledger actually prints,
    not a hand-written imitation of it.

    A hand-written sample can drift from the renderer silently, and then the
    harness reports "no itemization" on a real ledger boot -- inside the GPU
    window, where there is no time to debug a regex.
    """
    from sglang.srt.mem_ledger.engine import (
        TERM_ACTIVATION,
        TERM_HARDWARE_RESIDUAL,
        CardFacts,
        DemandInputs,
        build_card_ledgers,
    )
    from sglang.srt.mem_ledger.terms import render_all

    class Residual:
        uuid = "GPU-bbb"
        name = "RTX 3080"
        cuda_context_bytes = 300 << 20
        allocator_granularity_bytes = 8 << 20
        lazy_workspace_bytes = 100 << 20
        total_bytes = (300 + 8 + 100) << 20
        total_mib = 408

    class Calib:
        fingerprint = "abc123def456"

        def by_uuid(self):
            return {"GPU-bbb": Residual()}

    card = CardFacts(gpu_id=1, uuid="GPU-bbb", name="RTX 3080", total_mib=20480)
    ledgers = build_card_ledgers(
        DemandInputs(
            weight_mib_per_rank=[0],
            activation_mib_per_rank=[3968.0],
            capture_tokens_per_rank=[96],
            mamba_pool_mib_per_rank=[900.0],
            chunked_prefill_size=2048,
            phase_footprint_fingerprint="abc123def456",
        ),
        cards=[card],
        rank_gpu_id=[1],
        user_reserve_mib={1: 1024},
        calibration=Calib(),
    )

    # Prefix every line the way a production logger does.
    log = tmp_path / "boot.log"
    log.write_text(
        "\n".join(
            f"2026-08-05 10:00:00 INFO {line}"
            for line in render_all(ledgers).splitlines()
        )
        + "\n2026-08-05 10:00:00 INFO VRAM ledger totals: ...\n"
    )

    facts = compare_mod.parse_boot_log(str(log))
    assert len(facts["ledger_cards"]) == 1
    parsed = facts["ledger_cards"][0]
    assert parsed["gpu_id"] == 1
    assert parsed["total_mib"] == 20480
    names = {r["name"]: r["mib"] for r in parsed["rows"]}
    # Every term the renderer emitted must survive the round trip.
    assert names[TERM_ACTIVATION] == ledgers[0].term(TERM_ACTIVATION).mib
    assert names[TERM_HARDWARE_RESIDUAL] == 408
    assert names["user reserve (external)"] == 1024
    provenances = {r["name"]: r["provenance"] for r in parsed["rows"]}
    assert provenances[TERM_HARDWARE_RESIDUAL].startswith("calibrated@")


def test_unresolvable_cards_print_a_message_not_a_traceback(
    calibrate, monkeypatch, tmp_path, capsys
):
    """An operator in a hermetic shell (CUDA_VISIBLE_DEVICES=99) must be told
    why, not handed a stack trace."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "99")
    monkeypatch.setattr("sglang.srt.rigmon.card_probe._inventory", lambda: ([], None))
    rc = calibrate.main(["--dry-run", "--cache-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Cannot resolve the cards" in out
    assert "CUDA_VISIBLE_DEVICES is set to '99'" in out
    assert "Traceback" not in out
