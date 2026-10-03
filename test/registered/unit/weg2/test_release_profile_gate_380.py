# SPDX-License-Identifier: Apache-2.0
"""Item 380: scripts/weg2/devtools/release_profile_gate.py against FAKE release profiles.

The gate sources a profile in a clean bash and compares the EFFECTIVE values with the user
orders of 03.10. (NF = abl, 27B = INT8 gdncov + explicit --model + L15 on, Dual = NVFP4 without
L15 and with the MPS opt-in, every release profile 3 cards + inventory + not experimental).
Here a green set of fakes is built once, then ONE mutation at a time must turn exactly the
named rule red AND name the mutated line.  No rig file is read, nothing is launched.
"""

import importlib.util
import io
import pathlib
import sys

import pytest

TREE = pathlib.Path(__file__).resolve().parents[4]
GATE_PATH = TREE / "scripts" / "weg2" / "devtools" / "release_profile_gate.py"
_spec = importlib.util.spec_from_file_location("release_profile_gate_380", GATE_PATH)
G = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = G
_spec.loader.exec_module(G)

MC = "/models"
BASE = """\
PROFILE_NAME=base
PROFILE_STATUS=experimentell
PROFILE_MODEL=%(mc)s/Qwen3.8-27B-INT8-gdncov-vocabembed
PROFILE_DRAFT=%(mc)s/Qwen3.8-27B-DFlash2-W8-lued
PROFILE_CARD_COUNT=3
PROFILE_INVENTORY="RTX5090,RTX3080,RTX3080"
PROFILE_ARGS=(--p-bs 1)
profile_form_env() { _form SGLANG_WEG2_BASE_SWITCH 1; }
""" % {"mc": MC}

NF = """\
PROFILE_NAME=nf-int4
PROFILE_STATUS=abgenommen
PROFILE_MODEL=%(mc)s/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp
PROFILE_DRAFT=%(mc)s/Qwen3.8-Flash-Next-MTP-INT4-g32-albucino-abl-wxp
PROFILE_CARD_COUNT=3
PROFILE_INVENTORY="RTX5090,RTX3080,RTX3080"
PROFILE_ARGS=(
  --profile nextflash --model "$PROFILE_MODEL"
)
profile_form_env() { _form SGLANG_WEG2_X 1; }
""" % {"mc": MC}

B27 = """\
source "$(dirname "${BASH_SOURCE[0]}")/base.env"
PROFILE_NAME=27b
PROFILE_STATUS=abgenommen   # release
PROFILE_ARGS+=(--model "$PROFILE_MODEL")
profile_form_env() {
  _form SGLANG_WEG2_L15 1
  _form SGLANG_WEG2_L15_REFILL 1
  _form SGLANG_WEG2_L15_HOT_SHARE 1
  _form SGLANG_WEG2_VMM_EXPORTABLE 1
}
"""

DUAL = """\
# L15 steht hier nur im Kommentar: SGLANG_WEG2_L15=1 waere W-L15-DUAL
source "$(dirname "${BASH_SOURCE[0]}")/base.env"
PROFILE_NAME=27b-nvfp4-dual
PROFILE_STATUS=abgenommen
PROFILE_MODEL=%(mc)s/Qwen3.8-27B-NVFP4-RadixArk
PROFILE_DRAFT=%(mc)s/Qwen3.8-27B-DFlash2-NVFP4-RTNcal
PROFILE_ARGS=(--model "$PROFILE_MODEL" --dual-mps on)
profile_form_env() {
  _form SGLANG_WEG2_DUAL_MPS_OPT_IN 1
}
""" % {"mc": MC}

FILES = {"base.env": BASE, "nf-int4.env": NF, "nf.env": NF, "27b.env": B27, "27b-nvfp4-dual.env": DUAL}


def _write(tmp_path, **mut):
    files = dict(FILES)
    for key, (old, new) in mut.items():
        fname = {"nf": "nf-int4.env", "b27": "27b.env", "dual": "27b-nvfp4-dual.env", "base": "base.env"}[key]
        assert old in files[fname], (fname, old)
        files[fname] = files[fname].replace(old, new)
    for fname, text in files.items():
        (tmp_path / fname).write_text(text)
    return tmp_path


def _red(tmp_path, profile, role):
    fs = G.check_profile(str(tmp_path / profile), role)
    return [f for f in fs if not f.ok]


def test_green_set_is_green(tmp_path):
    _write(tmp_path)
    out = io.StringIO()
    assert G.run(str(tmp_path), ref="", out=out) == 0, out.getvalue()
    assert "0 rote Befunde" in out.getvalue()
    assert "SKIP base.env" in out.getvalue()


def test_missing_release_profile_is_red(tmp_path):
    _write(tmp_path)
    (tmp_path / "nf.env").unlink()
    out = io.StringIO()
    assert G.run(str(tmp_path), ref="", out=out) == 1
    assert "nf.env" in out.getvalue() and "fehlt" in out.getvalue()


def test_missing_directory_is_usage_error(tmp_path):
    assert G.run(str(tmp_path / "nope"), ref="") == 2


def test_nf_without_abl_model_names_line(tmp_path):
    _write(tmp_path, nf=("Minachist-abl-wxp", "Minachist"))
    red = _red(tmp_path, "nf-int4.env", "nf")
    assert [f.rule for f in red] == ["NF-MODEL", "NF-MODEL"] or [f.rule for f in red] == ["NF-MODEL"]
    f = red[0]
    assert f.where == "nf-int4.env:3"
    assert "Minachist" in f.detail and "PROFILE_MODEL=" in f.detail


def test_nf_without_abl_draft_names_line(tmp_path):
    _write(tmp_path, nf=("albucino-abl-wxp", "albucino"))
    red = _red(tmp_path, "nf-int4.env", "nf")
    assert len(red) == 1 and red[0].rule == "NF-MODEL" and red[0].where == "nf-int4.env:4"


def test_nf_model_flag_must_match_profile_model(tmp_path):
    _write(tmp_path, nf=('--model "$PROFILE_MODEL"', '--model /other/model'))
    red = _red(tmp_path, "nf-int4.env", "nf")
    assert [f.rule for f in red] == ["NF-MODEL"] and red[0].where == "nf-int4.env:8"


def test_nf_with_l15_is_red(tmp_path):
    _write(tmp_path, nf=("_form SGLANG_WEG2_X 1", "_form SGLANG_WEG2_L15 1"))
    red = _red(tmp_path, "nf-int4.env", "nf")
    assert [f.rule for f in red] == ["NF-NO-L15"]


def test_27b_abl_model_is_red(tmp_path):
    _write(tmp_path, base=("27B-INT8-gdncov-vocabembed", "27B-abl-INT8-gdncov-vocabembed"))
    rules = {f.rule for f in _red(tmp_path, "27b.env", "27b")}
    assert rules == {"27B-MODEL"}


def test_27b_without_model_flag_is_red_with_anchor(tmp_path):
    _write(tmp_path, b27=('PROFILE_ARGS+=(--model "$PROFILE_MODEL")', ""))
    red = _red(tmp_path, "27b.env", "27b")
    assert [f.rule for f in red] == ["27B-ARGV"]
    assert red[0].where == "base.env:7" and "kein '--model'" in red[0].detail


def test_27b_model_flag_with_other_checkpoint_is_red(tmp_path):
    _write(tmp_path, b27=('--model "$PROFILE_MODEL"', "--model /models/other"))
    red = _red(tmp_path, "27b.env", "27b")
    assert [f.rule for f in red] == ["27B-ARGV"] and red[0].where == "27b.env:4"


@pytest.mark.parametrize("key", ["SGLANG_WEG2_L15", "SGLANG_WEG2_L15_REFILL",
                                 "SGLANG_WEG2_L15_HOT_SHARE", "SGLANG_WEG2_VMM_EXPORTABLE"])
def test_27b_each_l15_switch_is_required(tmp_path, key):
    line = "  _form %s 1\n" % key
    _write(tmp_path, b27=(line, ""))
    red = _red(tmp_path, "27b.env", "27b")
    assert [f.rule for f in red] == ["27B-L15"]
    assert key + "=''" in red[0].detail and "Zeile fehlt" in red[0].detail


def test_27b_l15_off_value_is_red_with_its_line(tmp_path):
    _write(tmp_path, b27=("_form SGLANG_WEG2_L15_REFILL 1", "_form SGLANG_WEG2_L15_REFILL 0"))
    red = _red(tmp_path, "27b.env", "27b")
    assert [f.rule for f in red] == ["27B-L15"] and red[0].where == "27b.env:7"


def test_dual_green_ignores_l15_in_comments(tmp_path):
    _write(tmp_path)
    assert _red(tmp_path, "27b-nvfp4-dual.env", "dual") == []


def test_dual_with_l15_line_is_red_and_names_it(tmp_path):
    _write(tmp_path, dual=("  _form SGLANG_WEG2_DUAL_MPS_OPT_IN 1\n",
                           "  _form SGLANG_WEG2_DUAL_MPS_OPT_IN 1\n  _form SGLANG_WEG2_L15 1\n"))
    red = _red(tmp_path, "27b-nvfp4-dual.env", "dual")
    assert [f.rule for f in red] == ["DUAL-NO-L15"] and red[0].where == "27b-nvfp4-dual.env:10"


def test_dual_without_mps_opt_in_is_red(tmp_path):
    _write(tmp_path, dual=("_form SGLANG_WEG2_DUAL_MPS_OPT_IN 1", "_form SGLANG_WEG2_DUAL_MPS_OPT_IN 0"))
    red = _red(tmp_path, "27b-nvfp4-dual.env", "dual")
    assert [f.rule for f in red] == ["DUAL-MPS"] and red[0].where == "27b-nvfp4-dual.env:9"


def test_dual_abl_is_red(tmp_path):
    _write(tmp_path, dual=("NVFP4-RadixArk", "NVFP4-abl-RadixArk"))
    assert {f.rule for f in _red(tmp_path, "27b-nvfp4-dual.env", "dual")} == {"DUAL-MODEL"}


def test_dual_without_model_flag_is_red(tmp_path):
    _write(tmp_path, dual=('--model "$PROFILE_MODEL" ', ""))
    assert [f.rule for f in _red(tmp_path, "27b-nvfp4-dual.env", "dual")] == ["DUAL-MODEL"]


def test_card_count_two_is_red_in_base_file(tmp_path):
    _write(tmp_path, base=("PROFILE_CARD_COUNT=3", "PROFILE_CARD_COUNT=2"))
    red = _red(tmp_path, "27b.env", "27b")
    assert [f.rule for f in red] == ["CARD-COUNT"] and red[0].where == "base.env:5"


def test_missing_inventory_is_red(tmp_path):
    _write(tmp_path, nf=('PROFILE_INVENTORY="RTX5090,RTX3080,RTX3080"', ""))
    red = _red(tmp_path, "nf-int4.env", "nf")
    assert [f.rule for f in red] == ["INVENTORY"]


def test_experimental_status_is_red(tmp_path):
    _write(tmp_path, dual=("PROFILE_STATUS=abgenommen", "PROFILE_STATUS=experimentell"))
    red = _red(tmp_path, "27b-nvfp4-dual.env", "dual")
    assert [f.rule for f in red] == ["STATUS"] and red[0].where == "27b-nvfp4-dual.env:4"


def test_profile_that_exits_while_sourcing_is_red_not_a_crash(tmp_path):
    _write(tmp_path, nf=("PROFILE_NAME=nf-int4", "PROFILE_NAME=nf-int4\nexit 3"))
    red = _red(tmp_path, "nf-int4.env", "nf")
    assert [f.rule for f in red] == ["EVALUATE"] and "rc=3" in red[0].detail


def test_reference_models_come_from_the_abl_template(tmp_path):
    ref = tmp_path / "ref.env"
    ref.write_text(NF.replace("abl-wxp", "abl-wxp2"))
    assert G.reference_models(str(ref)) == (
        "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp2",
        "Qwen3.8-Flash-Next-MTP-INT4-g32-albucino-abl-wxp2")
    assert G.reference_models(str(tmp_path / "missing.env")) == (G.NF_MODEL_BASE, G.NF_DRAFT_BASE)


def test_all_flag_checks_generic_rules_on_other_files(tmp_path):
    _write(tmp_path)
    (tmp_path / "other.env").write_text(BASE.replace("CARD_COUNT=3", "CARD_COUNT=2"))
    out = io.StringIO()
    assert G.run(str(tmp_path), ref="", include_all=True, out=out) == 1
    assert "other.env" in out.getvalue() and "CARD-COUNT" in out.getvalue()
    out2 = io.StringIO()
    assert G.run(str(tmp_path), ref="", out=out2) == 0


def test_gate_is_a_pure_reader():
    src = GATE_PATH.read_text()
    assert "import sglang" not in src and "from sglang" not in src
    assert "nvidia-smi" not in src and "docker run" not in src
    assert src.count("subprocess.run(") == 1   # the one clean-bash source
