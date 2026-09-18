"""weg2xsn296: after a sleep ~1.0 GB (P) / ~1.5 GB (D) per card stay that no
tag covers (xsn295). The sleep path names the residue in three terms and
dumps the allocator snapshot when SGLANG_WEG2_MEMHIST=1 armed the history."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_memory_saver as wms  # noqa: E402

MIB = 1 << 20


def test_residue_terms_split_untagged_and_outside_torch():
    t = wms.sleep_residue_terms(active_bytes=25_000 * MIB, reserved_bytes=26_000 * MIB,
                                tagged_bytes=24_100 * MIB, nvml_used_bytes=1168 * MIB)
    assert t["untagged_live"] == 900 * MIB
    assert t["outside_torch"] == 268 * MIB           # context + comm windows
    assert t["torch_reserved"] == 26_000 * MIB
    # unreadable NVML: named absence, never 0
    t2 = wms.sleep_residue_terms(active_bytes=10, reserved_bytes=10, tagged_bytes=20, nvml_used_bytes=None)
    assert t2["untagged_live"] == 0 and t2["outside_torch"] == -1


def test_memhist_arms_only_by_env(monkeypatch):
    monkeypatch.delenv(wms.MEMHIST_ENV, raising=False)
    monkeypatch.setattr(wms, "_MEMHIST_ARMED", False)
    wms._arm_memory_history()
    assert wms._MEMHIST_ARMED is False


def test_sleep_path_names_the_residue_and_dumps_the_snapshot():
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    src = open(wu.__file__).read()
    i = src.index('logger.warning("%s", census.format_line())')
    assert "self._weg2_log_sleep_residue(census, tags)" in src[i:i + 200]
    j = src.index("def _weg2_log_sleep_residue")
    body = src[j:j + 4000]
    assert "WEG2-SLEEP-RESIDUE" in body and "_dump_snapshot(path)" in body
    assert "memsnap_" in body and "SGLANG_WEG2_RANKDUMP_DIR" in body
    assert "_weg2_sleep_count: int = 0" in src        # slots dataclass field
