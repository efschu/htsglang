"""NS 26.09.: the flip's residue reading may never kill the flip.

Boot dkr27bbar1i8h109261950 (image rc11b, tree 2a7f992c14, 26.09. 19:53:22Z)
died at its first flip with ``W4 Weg2WakeRefused ... stage 'wake-kv': not
enough values to unpack (expected 3, got 1)`` in ``_nvml_process_mib`` --
``nvidia-smi --query-compute-apps`` printed a row without commas (first flip
~40 min after a host driver reload). ``parse_compute_apps`` skips and counts
every row that is not ``pid, used_memory, gpu_uuid`` with numeric pid/used.
"""

import logging

from sglang.srt.weg2 import front as F

UA = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
UB = "GPU-bbbbbbbb-0000-0000-0000-000000000002"

MIXED = "\n".join([
    f"101, 1200, {UA}",
    "No running processes found",
    f"102, [N/A], {UA}",
    "",
    "   ",
    f"103, 800, {UB}",
    "WARNING: infoROM is corrupted at gpu 0000:01:00.0",
    f"[Insufficient Permissions], 300, {UB}",
    f"104, 50, {UA}",
    f"999, 7000, {UA}",  # not one of ours
    "Failed to initialize NVML: Driver/library version mismatch, " + "x" * 300,
])


def test_mixed_rows_sum_only_valid_rows_and_never_raise(caplog):
    F._NVSMI_SKIP_WARNED = False
    before = F.NVSMI_SKIPPED_LINES
    with caplog.at_level(logging.WARNING, logger="weg2.front"):
        got = F.parse_compute_apps(MIXED, {101, 102, 103, 104})
    assert got == {UA: 1200 + 50, UB: 800}
    # "No running processes found", "[N/A]", the warning, the permissions row,
    # the driver-mismatch row (one comma, two fields); blank lines are not counted.
    assert F.NVSMI_SKIPPED_LINES - before == 5
    warns = [r for r in caplog.records if "NVSMI-PARSE" in r.getMessage()]
    assert len(warns) == 1, "one WARNING per process, not one per row"
    assert "No running processes found" in warns[0].getMessage()


def test_warning_is_once_per_process_and_row_text_is_capped(caplog):
    F._NVSMI_SKIP_WARNED = False
    long_row = "E" * 500
    with caplog.at_level(logging.WARNING, logger="weg2.front"):
        F.parse_compute_apps(long_row, {1})
        F.parse_compute_apps("No running processes found", {1})
    warns = [r for r in caplog.records if "NVSMI-PARSE" in r.getMessage()]
    assert len(warns) == 1
    assert "E" * 120 in warns[0].getMessage()
    assert "E" * 121 not in warns[0].getMessage()


def test_only_garbage_gives_empty_dict():
    assert F.parse_compute_apps("No running processes found\n", {1, 2}) == {}
    assert F.parse_compute_apps("", {1}) == {}
    assert F.parse_compute_apps(None, {1}) == {}


def test_nvml_process_mib_uses_the_parser(monkeypatch):
    class _R:
        stdout = f"7, 10, {UA}\nNo running processes found\n"

    monkeypatch.setattr(F.subprocess, "run", lambda *a, **k: _R())
    assert F._nvml_process_mib({7}) == {UA: 10}


def test_launcher_reader_uses_the_same_parser(monkeypatch):
    from sglang.srt.weg2 import launcher as L

    class _R:
        stdout = f"7, 10, {UA}\n8, [N/A], {UA}\nNo running processes found\n"

    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: _R())
    assert L.nvml_process_mib({7, 8}) == {UA: 10}
