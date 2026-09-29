"""27B performance switches ON BY DEFAULT after their metal proof (user rule
29.09. ~10:15Z: a switch proven on metal goes default-on IN THE CODE, not only
in the docker profile). Source: gpu-arb/docs/LEISTUNGSSCHALTER-INVENTAR-0929.md,
section "27B", list (a).

Per switch, three facts:
  * the qwen27b row turns it ON without any env / flag,
  * an explicit off (env ``0`` / the CLI flag) wins,
  * the nextflash row stays what it was (NF byte-identical).
"""

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import form as FM


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    for k in (FM.FORM_ENV,):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _as(clean, profile):
    if profile is None:
        clean.delenv(FM.FORM_ENV, raising=False)
    else:
        clean.setenv(FM.FORM_ENV, _form_env(profile))


# ---------------------------------------------------------------------------
# 1. SGLANG_WEG2_FRONT_EXACT_TOKENS -- w109290020 X-EXACT-TOKENS 108x match=1
# ---------------------------------------------------------------------------

XE = "SGLANG_WEG2_FRONT_EXACT_TOKENS"


@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, True),      # default ON on the 27B row
    ("qwen27b", "0", False),      # explicit off wins
    ("nextflash", None, True),    # NF unchanged (on since V1 27.09.)
    ("nextflash", "0", False),
    (None, None, False),          # no form: the code fallback, unchanged
])
def test_front_exact_tokens_default(clean, profile, explicit, want):
    clean.delenv(XE, raising=False)
    _as(clean, profile)
    if explicit is not None:
        clean.setenv(XE, explicit)
    assert envs.SGLANG_WEG2_FRONT_EXACT_TOKENS.get() is want
    assert FM.PROFILES["qwen27b"].front_exact_tokens is True
    assert FM.PROFILES["nextflash"].front_exact_tokens is True
