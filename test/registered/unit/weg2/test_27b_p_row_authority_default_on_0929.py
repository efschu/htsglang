"""27B default on: SGLANG_WEG2_P_ROW_AUTHORITY (Fix B, #631) -- registry
qwen27b True after the agent-load proof w109290020 (user rule 29.09. ~10:15Z).
Kept in its own file and its own commit (the LAST of the series) so
it can be held back alone: under row authority z30x2 died at the P->D flip
(write-through drain); the fix 4e15b21564 is not yet proven on metal (z30x3).
"""

import pytest

from sglang.srt.weg2 import form as FM
from sglang.srt.weg2 import p_row_authority as PR


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


def test_rows():
    assert FM.PROFILES["qwen27b"].p_row_authority is True
    assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"][PR.ENV] is True
    assert FM.PROFILES["nextflash"].p_row_authority is False        # NF unchanged
    assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][PR.ENV] is False


@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, True),     # default ON on the 27B row
    ("qwen27b", "0", False),     # explicit off wins
    ("nextflash", None, False),  # NF unchanged
    (None, None, False),         # no form: off, unchanged
])
def test_enabled_follows_the_row(profile, explicit, want):
    env = {}
    if profile is not None:
        env[FM.FORM_ENV] = _form_env(profile)
    if explicit is not None:
        env[PR.ENV] = explicit
    assert PR.enabled(env) is want


@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, True), ("qwen27b", "0", False), ("nextflash", None, False), (None, None, False)])
def test_environ_entry_follows_the_row(monkeypatch, profile, explicit, want):
    from sglang.srt.environ import envs

    monkeypatch.delenv(PR.ENV, raising=False)
    monkeypatch.delenv(FM.FORM_ENV, raising=False)
    if profile is not None:
        monkeypatch.setenv(FM.FORM_ENV, _form_env(profile))
    if explicit is not None:
        monkeypatch.setenv(PR.ENV, explicit)
    assert envs.SGLANG_WEG2_P_ROW_AUTHORITY.get() is want
