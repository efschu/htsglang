"""F0-F NF (08.10.): the editor package of the RENAMED tree is a fixed point of the editor rename (rename_rigdash.py).

The kit run (F0-E, tools/** with the web ident-fix) already renamed tools/rig_dashboard.  The rule of rename_rigdash.py is the same engine, so on
the renamed package it must come out as: exit 0, 0 replacements, 0 moved paths, no residue (the engine's own check: no old env prefix, no old
package path).  Two places are LOCKED on purpose and copied back byte-identical: the boot recordings (rigdash/kartenplan_data: their plan_id is a
sha256 over the content) and the Grafana board (it reads every metric family under the old AND the new name; a rename run would turn that into
(new|new)).  The scenario tests of the pre-rename tree (test_profile_catalog_union_1005 RenameRoundtrip, the six F0-C scenarios) stay skipped here on
purpose: their synthetic trees spell the old names.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))          # tools/rig_dashboard
REPO = os.path.abspath(os.path.join(PKG, "..", ".."))
RENAME = os.environ.get("RENAME_RIGDASH") or os.path.join(REPO, "docker", "pdflip-release", "rename_rigdash.py")
KIT = os.environ.get("RELEASE_KIT_TOOLS") or os.path.join(REPO, "tools", "release")

pytestmark = pytest.mark.skipif(not (os.path.isfile(RENAME) and os.path.isfile(os.path.join(KIT, "rename_to_flliper.py"))),
                                reason="rename_rigdash.py or the release kit is not in this tree")


def test_the_renamed_editor_package_is_a_fixed_point_of_the_editor_rename():
    with tempfile.TemporaryDirectory() as d:
        pkg = os.path.join(d, "pkg")
        shutil.copytree(PKG, pkg, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        res = subprocess.run([sys.executable, RENAME, pkg, "--dry-run", "--kit", KIT], capture_output=True, text=True, timeout=300)
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    line = [ln for ln in res.stdout.splitlines() if ln.startswith("rename_rigdash: DRY-RUN ok ")]
    assert line, out
    info = json.loads(line[0].split("DRY-RUN ok ", 1)[1])
    assert info["replacements"] == 0 and info["paths_moved"] == 0, info
    assert info["collisions_allowed"] == 0, info
    locked = info["locked_kept"]
    assert any(p.startswith("rigdash/kartenplan_data/") for p in locked), info
    assert "rigdash/deploy/grafana/dashboards/rig-verlauf.json" in locked, info
    assert all(p.startswith(("rigdash/kartenplan_data/", "rigdash/deploy/grafana/dashboards/")) for p in locked), info


def test_a_real_run_leaves_the_dual_board_and_the_recordings_byte_identical():
    with tempfile.TemporaryDirectory() as d:
        pkg = os.path.join(d, "pkg")
        out = os.path.join(d, "out")
        shutil.copytree(PKG, pkg, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        res = subprocess.run([sys.executable, RENAME, pkg, "--out", out, "--kit", KIT], capture_output=True, text=True, timeout=300)
        assert res.returncode == 0, res.stdout + res.stderr
        for rel in ("rigdash/deploy/grafana/dashboards/rig-verlauf.json", "rigdash/kartenplan_data/27b-int8.json"):
            assert open(os.path.join(out, rel), "rb").read() == open(os.path.join(PKG, rel), "rb").read(), rel
