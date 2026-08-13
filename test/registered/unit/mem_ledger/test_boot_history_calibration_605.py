# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Boot-history calibration: measured posts beat inherited constants (#605).

RED-FIRST. Every test here failed before ``boot_history.py`` existed, and the
two that matter most are the REFUSALS: a post whose history is wide must not
be averaged into a tidy number, because an average of a bimodal or
config-straddling distribution is a figure nobody measured and everybody can
quote.
"""

import json
import os
import tempfile

import pytest

from sglang.srt.mem_ledger.boot_history import (
    POST_HARDWARE_RESIDUAL,
    POST_LOAD_TRANSIENT,
    WIDE_SPREAD_FRACTION,
    BootHistory,
    bands_from_marks,
    load_boot_history,
    read_marks,
)

MIB = 1 << 20
UUID_A = "GPU-aaaa"
UUID_B = "GPU-bbbb"


#: Monotonic must be unique per mark: a boot writes ``pre_weight_load`` and
#: ``weights_loaded`` once per RUNNER, so a fixture that reuses the timestamp
#: collides on the recorder's own dedupe key and silently drops marks.
_CLOCK = [0.0]


def _mark(boot, pid, uuid, phase, *, draft=None, non_torch=0, transient=0, rank=0):
    _CLOCK[0] += 1.0
    m = {
        "boot_id": boot,
        "pid": pid,
        "rank": rank,
        "card_uuid": uuid,
        "phase": phase,
        "monotonic": _CLOCK[0],
        "non_torch_bytes": int(non_torch) * MIB,
        "allocator_transient_bytes": int(transient) * MIB,
        "reserved_bytes": 0,
        "allocated_bytes": 0,
    }
    if draft is not None:
        m["extra"] = {"draft_worker": draft}
    return m


def _boot(boot, pid, uuid, *, residual, transient):
    """One boot's marks for one process: a target weight load plus a peak."""
    return [
        _mark(boot, pid, uuid, "process_start"),
        _mark(boot, pid, uuid, "pre_weight_load", draft=False, non_torch=residual),
        _mark(
            boot,
            pid,
            uuid,
            "weights_loaded",
            draft=False,
            non_torch=residual,
            transient=0,
        ),
        _mark(
            boot,
            pid,
            uuid,
            "pre_weight_load",
            draft=True,
            non_torch=residual,
            transient=transient,
        ),
        _mark(boot, pid, uuid, "weights_loaded", draft=True, non_torch=residual),
    ]


def _write(directory, marks, name="flight_marks_rank0.jsonl"):
    path = os.path.join(directory, name)
    with open(path, "a") as f:
        for m in marks:
            f.write(json.dumps(m) + "\n")
    return path


# ---------------------------------------------------------------------------
# The narrow post is calibrated, and it is charged at the band's HIGH
# ---------------------------------------------------------------------------


def test_narrow_residual_history_produces_a_band():
    """A post that repeats within a narrow band is measured, not inherited."""
    marks = []
    for i, resid in enumerate([880, 886, 886, 886, 902, 886]):
        marks += _boot(f"boot{i}", 1000 + i, UUID_A, residual=resid, transient=5)
    history = bands_from_marks(marks)
    band = history.band(UUID_A, POST_HARDWARE_RESIDUAL)
    assert band is not None
    assert not band.refused, band.reason
    assert band.low_mib == 880
    assert band.high_mib == 902
    assert band.n_boots == 6
    # Charged at the HIGH of the band, never the mean: the ledger has to be
    # able to fund the worst boot it has actually seen.
    assert band.charge_mib == 902


def test_charge_is_the_band_high_not_the_mean():
    marks = []
    for i, resid in enumerate([400, 400, 400, 400, 400, 480]):
        marks += _boot(f"b{i}", 2000 + i, UUID_A, residual=resid, transient=1)
    band = bands_from_marks(marks).band(UUID_A, POST_HARDWARE_RESIDUAL)
    assert band.charge_mib == 480
    assert band.charge_mib != 413  # the mean, which nobody measured


# ---------------------------------------------------------------------------
# REFUSE-WIDE. The load transient is the falsified 70 MiB constant's post.
# ---------------------------------------------------------------------------


def test_wide_history_is_refused_and_not_averaged():
    """A post spanning 0..18486 MiB is not a constant and must not become one."""
    marks = []
    for i, tr in enumerate([0, 154, 6690, 13392, 14972, 18486]):
        marks += _boot(f"w{i}", 3000 + i, UUID_A, residual=886, transient=tr)
    band = bands_from_marks(marks).band(UUID_A, POST_LOAD_TRANSIENT)
    assert band is not None
    assert band.refused
    assert band.charge_mib is None
    # The refusal has to NAME the distribution it refused, or the reader
    # cannot tell a refusal from a missing measurement.
    assert "18486" in band.reason
    assert "6 boot" in band.reason


def test_refusal_threshold_is_stated_and_enforced_at_the_boundary():
    """Just inside the rule calibrates; just outside refuses."""
    high = 1000
    just_in = int(high * (1.0 - WIDE_SPREAD_FRACTION)) + 1  # spread just under
    marks = []
    for i, v in enumerate([just_in, high]):
        marks += _boot(f"n{i}", 4000 + i, UUID_A, residual=886, transient=v)
    assert not bands_from_marks(marks).band(UUID_A, POST_LOAD_TRANSIENT).refused

    marks = []
    for i, v in enumerate([just_in - 2, high]):
        marks += _boot(f"m{i}", 4100 + i, UUID_A, residual=886, transient=v)
    assert bands_from_marks(marks).band(UUID_A, POST_LOAD_TRANSIENT).refused


def test_single_boot_is_not_a_distribution():
    """One boot cannot establish a band; it is refused, not calibrated."""
    marks = _boot("solo", 5000, UUID_A, residual=886, transient=13392)
    band = bands_from_marks(marks).band(UUID_A, POST_HARDWARE_RESIDUAL)
    assert band.refused
    assert "1 boot" in band.reason


# ---------------------------------------------------------------------------
# Cards are kept apart. Pooling two cards is the R2 sample-breadth error.
# ---------------------------------------------------------------------------


def test_cards_are_never_pooled():
    marks = []
    for i in range(4):
        marks += _boot(f"c{i}", 6000 + i, UUID_A, residual=886, transient=5)
        marks += _boot(f"c{i}", 7000 + i, UUID_B, residual=480, transient=5)
    history = bands_from_marks(marks)
    assert history.band(UUID_A, POST_HARDWARE_RESIDUAL).charge_mib == 886
    assert history.band(UUID_B, POST_HARDWARE_RESIDUAL).charge_mib == 480


def test_duplicate_mark_lines_do_not_inflate_the_boot_count():
    """The rank0 file carries every pid's marks; reading all files double-counts."""
    marks = []
    for i in range(4):
        marks += _boot(f"d{i}", 8000 + i, UUID_A, residual=886, transient=5)
    doubled = marks + list(marks)
    assert bands_from_marks(doubled).band(UUID_A, POST_HARDWARE_RESIDUAL).n_boots == 4


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


def test_read_marks_dedupes_across_files_and_load_returns_none_when_empty():
    with tempfile.TemporaryDirectory() as d:
        assert load_boot_history(d) is None
        marks = []
        for i in range(3):
            marks += _boot(f"r{i}", 9000 + i, UUID_A, residual=886, transient=5)
        _write(d, marks, "flight_marks_rank0.jsonl")
        _write(d, marks, "flight_marks_rank1.jsonl")  # the duplication defect
        got = read_marks(d)
        assert len(got) == len(marks)
        history = load_boot_history(d)
        assert isinstance(history, BootHistory)
        assert history.band(UUID_A, POST_HARDWARE_RESIDUAL).n_boots == 3


def test_unknown_card_has_no_band_rather_than_a_zero():
    marks = []
    for i in range(3):
        marks += _boot(f"u{i}", 9100 + i, UUID_A, residual=886, transient=5)
    assert bands_from_marks(marks).band("GPU-nope", POST_HARDWARE_RESIDUAL) is None


def test_corrupt_lines_are_skipped_not_fatal():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "flight_marks_rank0.jsonl")
        with open(path, "w") as f:
            f.write("{not json\n")
            for m in _boot("k0", 9200, UUID_A, residual=886, transient=5):
                f.write(json.dumps(m) + "\n")
            f.write("\n")
        assert len(read_marks(d)) == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--color=no"]))
