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


# ---------------------------------------------------------------------------
# 2. HG bundle (row 24h): SGLANG_WEG2_D_EARLY_DRAFT + SGLANG_DFLASH_ACCEPT_SYNC_
#    FUSED + SGLANG_BARLINK_BAR1_CANON_ORDER, measured only together --
#    dkr27bint8dhgbar1dhg109261456 vs dkr27bbar1i8h109261444: step time better
#    at all 24 points (10k bs1 code -2.9 %, prose -4.4 %, 240k -3.4/-2.3 %).
#    ONE registry field (d_hostgap_levers); nextflash off (CANON_ORDER would
#    change NF's bar1 reduction order -- the inventory does not prove it on NF).
# ---------------------------------------------------------------------------

HG = ("SGLANG_WEG2_D_EARLY_DRAFT", "SGLANG_DFLASH_ACCEPT_SYNC_FUSED",
      "SGLANG_BARLINK_BAR1_CANON_ORDER")


def _hg_readers():
    """The three rank-side readers, each asked the way its caller asks."""
    from sglang.srt.distributed.device_communicators import barlink_bar1 as B
    from sglang.srt.managers import weg2_d_hostgap as H
    from sglang.srt.speculative import dflash_worker_v2 as W

    return {
        "SGLANG_WEG2_D_EARLY_DRAFT": H.early_draft_on,
        "SGLANG_DFLASH_ACCEPT_SYNC_FUSED": W.accept_sync_fused_on,
        "SGLANG_BARLINK_BAR1_CANON_ORDER": B.canon_order_on,
    }


def test_hg_bundle_is_one_registry_field():
    assert FM.PROFILES["qwen27b"].d_hostgap_levers is True
    assert FM.PROFILES["nextflash"].d_hostgap_levers is False
    for name in HG:
        assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"][name] is True
        assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][name] is False


@pytest.mark.parametrize("name", HG)
@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, True),      # default ON on the 27B row
    ("qwen27b", "0", False),      # explicit off wins
    ("qwen27b", "1", True),
    ("nextflash", None, False),   # NF unchanged: off
    ("nextflash", "1", True),     # an explicit on still reaches NF
    (None, None, False),          # no form: the code default (off), unchanged
])
def test_hg_bundle_default(clean, name, profile, explicit, want):
    for n in HG:
        clean.delenv(n, raising=False)
    _as(clean, profile)
    if explicit is not None:
        clean.setenv(name, explicit)
    assert _hg_readers()[name]() is want
    assert getattr(envs, name).get() is want
