"""UNIFY S3: the model-profile registry (weg2/form.py ModelProfile / PROFILES).

* BLOCKER from S2: the NF tail switches (SGLANG_WEG2_TAIL_HANDOFF/ADOPT/VERIFY/
  SKIP_EXTEND, EnvBool(True)) and the NF mamba-anchor displacement
  (SGLANG_WEG2_MAMBA_ARENA_RID_ANCHORS, EnvInt(-1)) ran UNSWITCHED on the 27B.
  Now their defaults are the profile fields ``end_anchor`` / ``mamba_anchor``:
  nextflash on, qwen27b off; no published form = the NF code default; an
  explicitly set env var always wins.
* ``if profile == ...`` in the launcher became table rows (early_read_flags,
  group_env); the measured constants moved into the rows unchanged, the NF row
  carries its own P_DRAFT_RESIDENT_BUDGET_MIB and BORROWS the rest by name.
* ONE calibration identity: checkpoint AND form, AND the 27B line term
  (line_identity 76e87ac3b2) where the profile's records row names it.
"""

import argparse
import ast
import dataclasses
import os
import pathlib
import subprocess
import tempfile

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import form as F

WEG2_DIR = pathlib.Path(F.__file__).resolve().parent
SRT_DIR = WEG2_DIR.parent

TAILS = ("SGLANG_WEG2_TAIL_HANDOFF", "SGLANG_WEG2_TAIL_ADOPT",
         "SGLANG_WEG2_TAIL_VERIFY", "SGLANG_WEG2_TAIL_SKIP_EXTEND")
RID = "SGLANG_WEG2_MAMBA_ARENA_RID_ANCHORS"
SHARE = "SGLANG_WEG2_DRAFT_SHARE_EMBED"


def _form_env(profile, model="m"):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile != "nextflash"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return F.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                      flip="family", vision="off", profile=profile, model=model).env_value()


@pytest.fixture
def clean(monkeypatch):
    for k in TAILS + (RID, SHARE, F.FORM_ENV):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ---------------------------------------------------------------- the blocker


@pytest.mark.parametrize("profile,want", [("qwen27b", False), ("nextflash", True), (None, True)])
def test_the_profile_switches_the_tail_mechanics(clean, profile, want):
    from sglang.srt.weg2 import tail_adopt, tail_handoff

    if profile is not None:
        clean.setenv(F.FORM_ENV, _form_env(profile))
    for name in TAILS:
        assert getattr(envs, name).get() is want, name
    assert tail_handoff.enabled() is want
    assert tail_handoff.skip_extend_enabled() is want
    assert tail_adopt.adopt_enabled() is want
    assert tail_adopt.verify_enabled() is want


@pytest.mark.parametrize("profile", ["qwen27b", "nextflash", None])
@pytest.mark.parametrize("explicit,want", [("1", True), ("0", False)])
def test_an_explicit_tail_env_wins_over_the_profile(clean, profile, explicit, want):
    from sglang.srt.weg2 import tail_handoff

    if profile is not None:
        clean.setenv(F.FORM_ENV, _form_env(profile))
    for name in TAILS:
        clean.setenv(name, explicit)
    assert tail_handoff.enabled() is want
    assert tail_handoff.skip_extend_enabled() is want


@pytest.mark.parametrize("profile,want", [("qwen27b", 0), ("nextflash", -1), (None, -1)])
def test_the_profile_switches_the_mamba_anchor_displacement(clean, profile, want):
    from sglang.srt.weg2 import mamba_arena_displace as mad

    if profile is not None:
        clean.setenv(F.FORM_ENV, _form_env(profile))
    got = envs.SGLANG_WEG2_MAMBA_ARENA_RID_ANCHORS.get()
    assert got == want and type(got) is int
    # 0 = cap 0 = first-come: _weg2_mamba_claim never reaches the displacement
    # nor arena.c arena_drop_unreferenced; -1 = auto max(2, slots // 4).
    assert mad.rid_anchor_cap(configured=got, arena_slots=32) == (0 if want == 0 else 8)


@pytest.mark.parametrize("profile", ["qwen27b", "nextflash"])
@pytest.mark.parametrize("explicit", ["0", "-1", "5"])
def test_an_explicit_anchor_env_wins_over_the_profile(clean, profile, explicit):
    clean.setenv(F.FORM_ENV, _form_env(profile))
    clean.setenv(RID, explicit)
    assert envs.SGLANG_WEG2_MAMBA_ARENA_RID_ANCHORS.get() == int(explicit)


def test_the_rank_readers_ask_the_one_env_entry():
    """No second reader that could skip the profile default."""
    import inspect

    from sglang.srt.mem_cache import unified_radix_cache as urc
    from sglang.srt.weg2 import tail_adopt, tail_handoff

    assert "envs.SGLANG_WEG2_TAIL_HANDOFF.get()" in inspect.getsource(tail_handoff.enabled)
    assert "envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.get()" in inspect.getsource(tail_handoff.skip_extend_enabled)
    assert "envs.SGLANG_WEG2_TAIL_ADOPT.get()" in inspect.getsource(tail_adopt.adopt_enabled)
    assert "envs.SGLANG_WEG2_MAMBA_ARENA_RID_ANCHORS.get()" in inspect.getsource(urc)
    for py in list(SRT_DIR.rglob("*.py")):
        src = py.read_text(errors="replace")
        for name in TAILS + (RID,):
            for bad in (f'os.environ.get("{name}"', f'os.getenv("{name}"', f"os.environ['{name}']"):
                assert bad not in src, (py, bad)


@pytest.mark.parametrize("profile,want", [("qwen27b", False), ("nextflash", True), (None, True)])
def test_draft_share_embed_follows_the_profile_on_rank_and_planner(clean, profile, want):
    from sglang.srt.planner.expert_residency import draft_share_embed

    env = {}
    if profile is not None:
        clean.setenv(F.FORM_ENV, _form_env(profile))
        env[F.FORM_ENV] = _form_env(profile)
    assert envs.SGLANG_WEG2_DRAFT_SHARE_EMBED.get() is want
    assert draft_share_embed(env) is want
    assert draft_share_embed({**env, SHARE: "0"}) is False
    assert draft_share_embed({**env, SHARE: "1"}) is True


# ---------------------------------------------------------------- the table


def test_switch_defaults_are_derived_from_the_rows_not_typed():
    for pid, prof in F.PROFILES.items():
        assert F.PROFILE_SWITCH_DEFAULTS[pid] == prof.switch_defaults()
        assert prof.end_anchor in F.END_ANCHOR_VALUES
        assert prof.mamba_anchor in F.MAMBA_ANCHOR_VALUES
        assert prof.draft.kind in F.DRAFT_KINDS
        assert prof.d_layout in F.D_LAYOUT_VALUES
        assert prof.p_draft in F.AXIS_VALUES["p_draft"]
        assert set(prof.records.fields) <= set(F.RECORD_KEY_FIELDS)
        assert prof.expect == F.PROFILE_EXPECT[pid]
    assert set(F.END_ANCHOR_SWITCHES) == set(F.END_ANCHOR_VALUES)
    assert set(F.MAMBA_ANCHOR_SWITCHES) == set(F.MAMBA_ANCHOR_VALUES)
    q, n = F.PROFILES["qwen27b"], F.PROFILES["nextflash"]
    assert (q.end_anchor, q.mamba_anchor) == ("trim", "grid4096")
    assert (n.end_anchor, n.mamba_anchor) == ("tail_handoff", "deepest")


def test_every_profile_switch_has_exactly_one_environ_entry_with_a_profile_default():
    import inspect

    from sglang.srt import environ as env_mod

    tree = ast.parse(inspect.getsource(env_mod))
    names = {n for d in F.PROFILE_SWITCH_DEFAULTS.values() for n in d}
    for d in F.PROFILE_SWITCH_DEFAULTS.values():
        assert set(d) == names
    for name in names:
        hits = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert len(hits) == 1, name
        assert callable(getattr(env_mod.Envs, name).default), name


def test_the_27b_constants_moved_unchanged_and_the_nf_row_is_separate():
    q = F.PROFILES["qwen27b"].constants
    want = {
        "DC_MEASURED_D_5090_MIB": 2228, "DC_MEASURED_D_3080_MIB": 1922,
        "DC_MEASURED_D_XCHG_MIB": (2588, 3084, 2588),
        "P_OVERSHOOT_MIB": (920, 0, 512), "D_OVERSHOOT_MIB": (489, 0, 0),
        "P_DRAFT_RESIDENT_BUDGET_MIB": 405.2 + 1213.0, "CALIBRATION_LAYERS": 64,
        "MEASURED_MS_PER_LAYER": "8.10,35.16,33.59",
        "P_PP_STAGE_FIXED_MIB": "2342.0,1105.5,3518.0",
        "P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT": 1.5588,
        "X_RECORDED_R_D_TOKS": 690.0, "X_RECORDED_R_P_TOKS": 3640.0, "X_RECORDED_FLIP_S": 13.247,
        "STORE_CENSUS_KV_PAGES": 50651, "STORE_CENSUS_MAMBA_BLOBS": 42,
        "STORE_CENSUS_DRAFT_PAGES": 26040, "STORE_CENSUS_KV_PAGE_BYTES": 32768,
    }
    for k, v in want.items():
        assert q[k].value == v, k
        assert q[k].measured_on == "qwen27b", k
    n = F.PROFILES["nextflash"].constants
    assert n["P_DRAFT_RESIDENT_BUDGET_MIB"].value == pytest.approx(615.7 + 1522.7)
    assert n["P_DRAFT_RESIDENT_BUDGET_MIB"].measured_on == "nextflash"
    borrowed = dict(F.borrowed_constants("nextflash"))
    assert set(borrowed) == set(n) - {"P_DRAFT_RESIDENT_BUDGET_MIB"}
    assert set(borrowed.values()) == {"qwen27b"}
    assert F.borrowed_constants("qwen27b") == ()
    assert "P_OVERSHOOT_MIB" in F.borrowed_constants_line("nextflash")
    assert F.borrowed_constants_line("qwen27b") is None


def test_profile_constant_reads_the_published_profile(clean):
    assert F.profile_constant("P_DRAFT_RESIDENT_BUDGET_MIB") == pytest.approx(1618.2)
    clean.setenv(F.FORM_ENV, _form_env("nextflash"))
    assert F.profile_constant("P_DRAFT_RESIDENT_BUDGET_MIB") == pytest.approx(2138.4)
    assert F.profile_constant("P_DRAFT_RESIDENT_BUDGET_MIB", "qwen27b") == pytest.approx(1618.2)
    with pytest.raises(KeyError):
        F.profile_constant("NOPE", "qwen27b")
    with pytest.raises(KeyError):
        F.profile_constant("CALIBRATION_LAYERS", "no-such-model")
