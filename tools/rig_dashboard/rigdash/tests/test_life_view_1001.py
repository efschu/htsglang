"""Nutzer 01.10.: "Letzte Boots" zeigt Dauer und Bootzeit je Boot."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import ipcboot  # noqa: E402


def test_terminal_boot_runs_to_lifecycle_since():
    ipc = {"serving_since_ts": 1194.0, "terminal": True, "lifecycle_since": 4600.0}
    assert ipcboot.life_view(ipc, 1000.0, 9999.0) == {"boot_s": 194.0, "dur_s": 3600.0}


def test_live_boot_runs_to_last_sign_of_life():
    ipc = {"serving_since_ts": 1194.0, "terminal": False, "lifecycle_since": 1194.0}
    assert ipcboot.life_view(ipc, 1000.0, 1300.0) == {"boot_s": 194.0, "dur_s": 300.0}


def test_never_served_has_no_boot_time():
    ipc = {"terminal": True, "lifecycle_since": 1040.0}
    assert ipcboot.life_view(ipc, 1000.0, 1040.0) == {"boot_s": None, "dur_s": 40.0}
    assert ipcboot.life_view({}, None, 1040.0) == {"boot_s": None, "dur_s": None}
