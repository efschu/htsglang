"""UNIFY S2: switches whose default differs by model profile (weg2/form.py
PROFILE_SWITCH_DEFAULTS), one environ.py entry each.

* SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL (H39): 27B port 3c9bfeff95 shipped off,
  the qwen27b row is on since the operator decision of 26.09. (all 27B profiles
  set 1), NF d6b7d4a1d3 default on (pinned in
  test_weg2_dense_repack_outside_pool_27b.py).
* SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD (H81): the 27B line held the END anchors
  only with SGLANG_WEG2_MAMBA_INNER_ANCHOR_RELEASE=1 (default off, its arms set
  1), the NF line holds by default. The 27B switch is read as an alias.

Explicit value > 27B alias > profile default > the NF default (no form).
"""

import ast
import inspect

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2.form import PROFILE_SWITCH_DEFAULTS, Weg2Form

HOLD = "SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD"
ALIAS = "SGLANG_WEG2_MAMBA_INNER_ANCHOR_RELEASE"


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                    flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    for k in (HOLD, ALIAS, "SGLANG_WEG2_FORM"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.mark.parametrize("profile,want", [("qwen27b", False), ("nextflash", True), (None, True)])
def test_carrier_hold_default_per_profile(clean, profile, want):
    if profile is not None:
        clean.setenv("SGLANG_WEG2_FORM", _form_env(profile))
    assert envs.SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD.get() is want


@pytest.mark.parametrize("profile", ["qwen27b", "nextflash", None])
@pytest.mark.parametrize("alias,want", [("1", True), ("on", True), ("0", False)])
def test_the_27b_switch_is_the_alias(clean, profile, alias, want):
    """A 27B arm (docker/profiles/27b.env: INNER_ANCHOR_RELEASE 1) keeps its hold."""
    if profile is not None:
        clean.setenv("SGLANG_WEG2_FORM", _form_env(profile))
    clean.setenv(ALIAS, alias)
    assert envs.SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD.get() is want


@pytest.mark.parametrize("explicit,want", [("1", True), ("0", False)])
def test_the_explicit_switch_wins_over_the_alias(clean, explicit, want):
    clean.setenv("SGLANG_WEG2_FORM", _form_env("qwen27b"))
    clean.setenv(ALIAS, "0" if want else "1")
    clean.setenv(HOLD, explicit)
    assert envs.SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD.get() is want


def test_the_rank_gate_reads_the_resolved_switch(clean):
    """unified_radix_cache._weg2_carrier_hold_on asks the one env entry."""
    from sglang.srt.mem_cache import unified_radix_cache as urc

    src = inspect.getsource(urc._weg2_carrier_hold_on)
    assert "envs.SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD.get()" in src


def test_every_profile_names_every_switch_and_each_has_one_environ_entry():
    from sglang.srt import environ as env_mod

    names = {n for d in PROFILE_SWITCH_DEFAULTS.values() for n in d}
    for prof, d in PROFILE_SWITCH_DEFAULTS.items():
        assert set(d) == names, prof
    tree = ast.parse(inspect.getsource(env_mod))
    for name in names:
        hits = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert len(hits) == 1, (name, hits)
