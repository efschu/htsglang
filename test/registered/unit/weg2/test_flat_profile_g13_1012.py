"""G13 (NF-GGUF, stage A of 12.10.2026): the flat, bootable rig profile ``docker/profiles_release/nf-gguf-d.env``.

The stage-A boot could not start from the tree as it stood: PROFILE_MOUNT=1 mounts ONE file but the profile sources two
siblings (the image carried the 25.09. placeholder ``nf-gguf.env``); the tree's ``profconv/`` is the fLLiper spelling the
rig's ``acc_nf_m1_donly.sh`` refuses; and ``$NF_STORE_DIR`` sat in ``PROFILE_REQUIRED_PATHS`` although
``preflight_model`` (checks) runs before ``preflight_state`` (creates) on a fresh tmpfs.

``tools/release/flat_profile.py`` generates the flat file from the source chain.  These tests pin that the committed file IS
what the generator gives, that it carries none of the refused names and no ``source`` line, and -- the real check -- that
SOURCING it in bash gives the same ``PROFILE_*`` / ``NF_*`` variables and arrays as sourcing the chain in the rig's spelling.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-weg2-unit")

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "tools" / "release" / "profconv"
FLAT = ROOT / "docker" / "profiles_release" / "nf-gguf-d.env"
GEN = ROOT / "tools" / "release" / "flat_profile.py"
RIG_NF = Path("/spinning/gpu-arb/docker/profiles_release/nf.env")


def _gen():
    spec = importlib.util.spec_from_file_location("flat_profile_g13", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FP = _gen()

_DUMP = r"""
set -u
source "$1" >/dev/null 2>&1 || { echo "SOURCE-FAILED rc=$?"; exit 3; }
for v in $(compgen -v | grep -E '^(PROFILE_|NF_)' | sort); do declare -p "$v"; done
"""


def _vars_after_sourcing(path: Path) -> str:
    r = subprocess.run(["bash", "-c", _DUMP, "dump", str(path)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (r.stdout[-500:], r.stderr[-500:])
    assert "PROFILE_NAME=" in r.stdout or 'PROFILE_NAME="' in r.stdout, r.stdout[:300]
    return r.stdout


def _rig_chain(tmp: Path) -> Path:
    """The source chain, each file in the rig's spelling, sourcing its sibling by ``$(dirname BASH_SOURCE)``."""
    for name in ("nf-gguf-d.env", "nf-gguf.env", "nf.env"):
        (tmp / name).write_text(FP.to_rig_spelling((SRC / name).read_text()))
    return tmp / "nf-gguf-d.env"


def test_the_committed_flat_file_is_what_the_generator_gives():
    assert FLAT.is_file(), "run: python3 tools/release/flat_profile.py nf-gguf-d"
    assert FLAT.read_text() == FP.flatten(str(SRC), "nf-gguf-d")
    r = subprocess.run(["python3", str(GEN), "nf-gguf-d", "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_flat_file_has_no_source_line_and_none_of_the_names_the_acceptance_script_refuses():
    text = FLAT.read_text()
    body = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in body if re.match(r"\s*(source|\.)\s", ln)], "a sibling source line survived"
    assert not [ln for ln in body if "BASH_SOURCE" in ln]
    # acc_nf_m1_donly.sh / host_acceptance.sh: ``grep -q 'FLLIPER_' $D/profiles/$PROF.env`` -> die (case-sensitive, comments too)
    assert "FLLIPER_" not in text
    assert "--pdflip-" not in text and "PDFLIP" not in text
    assert "--weg2-vision" in text and "SGLANG_HC_MIXER_INT8" in text
    # three source files inlined in order, later after earlier
    marks = [m.group(1) for m in re.finditer(r"^# >>> inlined from (\S+)", text, re.M)]
    assert marks == ["nf-gguf.env", "nf.env"]


def test_sourcing_the_flat_file_equals_sourcing_the_chain_in_the_rig_spelling():
    with tempfile.TemporaryDirectory() as td:
        chain = _vars_after_sourcing(_rig_chain(Path(td)))
    flat = _vars_after_sourcing(FLAT)
    assert flat == chain
    # and it is the GGUF D-only profile, not the placeholder the image carried next to the mounted file
    assert 'PROFILE_NAME="nf-gguf-d"' in flat or "PROFILE_NAME=nf-gguf-d" in flat
    assert "PROFILE_PLACEHOLDER=\"0\"" in flat or "PROFILE_PLACEHOLDER=0" in flat
    assert "PROFILE_D_ONLY" in flat


def test_the_store_dir_is_not_a_required_path_but_stays_in_the_argv():
    flat = _vars_after_sourcing(FLAT)
    paths = re.search(r"declare -a PROFILE_REQUIRED_PATHS=\((.*)\)\n", flat).group(1)
    assert "/mnt/nf-experts" not in paths
    assert "Qwen3.8-Flash-Next-GGUF-unsloth-sibling" in paths and "-draft" in paths
    assert "SGLANG_MOE_EXPERT_STORE_DIR=/mnt/nf-experts/gguf" in flat  # the env pair of --env-p / --env-d is untouched
    assert "NF_STORE_DIR=" in flat and "/mnt/nf-experts/gguf" in flat


def test_no_other_profile_requires_its_store_dir():
    """The change is scoped to the GGUF D profile: no NF profile of the chain family lists the store in its required paths."""
    for f in sorted(SRC.glob("nf*.env")):
        text = f.read_text()
        m = re.search(r"PROFILE_REQUIRED_PATHS=\((.*?)\n\)", text, re.S)
        if m:
            assert "NF_STORE_DIR" not in m.group(1), f.name


def test_a_changed_source_makes_the_check_fail(tmp_path):
    src = tmp_path / "profconv"
    shutil.copytree(SRC, src, ignore=shutil.ignore_patterns("*.json", "*.bak*"))
    dst = tmp_path / "out"
    assert FP.main(["nf-gguf-d", "--src", str(src), "--dst", str(dst)]) == 0
    assert FP.main(["nf-gguf-d", "--src", str(src), "--dst", str(dst), "--check"]) == 0
    p = src / "nf-gguf.env"
    p.write_text(p.read_text() + "\nNF_G13_PROBE=1\n")
    assert FP.main(["nf-gguf-d", "--src", str(src), "--dst", str(dst), "--check"]) == 1
    assert "NF_G13_PROBE=1" in FP.flatten(str(src), "nf-gguf-d")


def test_a_source_cycle_and_a_foreign_bash_source_use_are_refused(tmp_path):
    (tmp_path / "a.env").write_text('source "$(dirname "${BASH_SOURCE[0]}")/b.env"\n')
    (tmp_path / "b.env").write_text('source "$(dirname "${BASH_SOURCE[0]}")/a.env"\n')
    with pytest.raises(ValueError, match="cycle"):
        FP.flatten(str(tmp_path), "a")
    (tmp_path / "c.env").write_text('DIR="$(dirname "${BASH_SOURCE[0]}")"\n')
    with pytest.raises(ValueError, match="BASH_SOURCE"):
        FP.flatten(str(tmp_path), "c")


def test_the_spelling_map_is_the_inverse_of_the_forward_rename():
    assert FP.to_rig_spelling("FLLIPER_PDFLIP_QUIESCE_FAST=1 FLLIPER_OPT_PDFLIP_D_SEAT_VRAM=1 FLLIPER_HC_MIXER_INT8=0") == (
        "SGLANG_WEG2_QUIESCE_FAST=1 SGLANG_OPT_WEG2_D_SEAT_VRAM=1 SGLANG_HC_MIXER_INT8=0"
    )
    assert FP.to_rig_spelling("--pdflip-vision off ARB/pdflip/x.sh") == "--weg2-vision off ARB/weg2/x.sh"


@pytest.mark.skipif(not RIG_NF.exists(), reason="the rig's profiles_release/nf.env is not mounted")
def test_the_spelling_map_reproduces_the_rigs_nf_env_byte_for_byte():
    assert FP.to_rig_spelling((SRC / "nf.env").read_text()) == RIG_NF.read_text()
