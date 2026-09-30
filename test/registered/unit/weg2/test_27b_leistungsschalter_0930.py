"""27B Leistungsschalter, second pass (30.09.): the switches every 27B profile
still had to set although their metal proof exists. User rule 29.09. ~10:15Z
(memory leistungsschalter-nach-nachweis-default-an-0929): a switch proven on
metal goes default-on IN THE CODE; the docker profile line becomes redundant.

Group 1 -- THE HG BASE (D host gap, 27B row 24h): the four switches the dhg
measurement ran UNDER in both arms (profiles/27b-int8-dhg.env: "Form =
27b.env (u.a. SGLANG_WEG2_D_DEFER_SEQ_LENS_CPU=1 + SGLANG_WEG2_D_DEFER_REBUILD=1,
SGLANG_DFLASH_PLAN_SYNC_FREE=1, SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE=1) plus
genau die Schalter unten"). The HG levers are default-on for qwen27b since
df762c4bbc, but SGLANG_WEG2_D_EARLY_DRAFT is stage 3 of the deferred length
read and INERT without SGLANG_WEG2_D_DEFER_SEQ_LENS_CPU -- the registry named
an on-switch that could not act without the profile.

Per switch: qwen27b ON without env, explicit 0 wins, nextflash unchanged (off),
no form unchanged (off).
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
    monkeypatch.delenv(FM.FORM_ENV, raising=False)
    return monkeypatch


def _as(clean, profile):
    if profile is None:
        clean.delenv(FM.FORM_ENV, raising=False)
    else:
        clean.setenv(FM.FORM_ENV, _form_env(profile))


# ---------------------------------------------------------------------------
# Group 1: the HG base
# ---------------------------------------------------------------------------

HG_BASE = (
    "SGLANG_WEG2_D_DEFER_SEQ_LENS_CPU",
    "SGLANG_WEG2_D_DEFER_REBUILD",
    "SGLANG_DFLASH_PLAN_SYNC_FREE",
    "SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE",
)


def _base_readers():
    """Each switch asked the way its rank-side caller asks it."""
    from sglang.srt.managers import weg2_d_hostgap as H

    return {
        "SGLANG_WEG2_D_DEFER_SEQ_LENS_CPU": H.defer_seq_lens_cpu_on,
        "SGLANG_WEG2_D_DEFER_REBUILD": H.defer_rebuild_on,
        "SGLANG_DFLASH_PLAN_SYNC_FREE": lambda: bool(envs.SGLANG_DFLASH_PLAN_SYNC_FREE.get()),
        "SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE":
            lambda: bool(envs.SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE.get()),
    }


def test_hg_base_is_one_registry_field():
    assert tuple(FM.HG_BASE_SWITCHES) == HG_BASE
    assert FM.PROFILES["qwen27b"].d_hostgap_base is True
    assert FM.PROFILES["nextflash"].d_hostgap_base is False
    for name in HG_BASE:
        assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"][name] is True
        assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][name] is False


def test_every_row_with_hg_levers_carries_their_base():
    """EARLY_DRAFT is inert without the deferred read: a row that turns the
    levers on without the base names an on-switch that cannot act."""
    for pid, prof in FM.PROFILES.items():
        if prof.d_hostgap_levers:
            assert prof.d_hostgap_base, pid


@pytest.mark.parametrize("name", HG_BASE)
@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, True),      # default ON on the 27B row
    ("qwen27b", "", True),        # blank = unset
    ("qwen27b", "0", False),      # explicit off wins
    ("qwen27b", "1", True),
    ("nextflash", None, False),   # NF unchanged: off (its D is MTP)
    ("nextflash", "1", True),     # an explicit on still reaches NF
    (None, None, False),          # no form: the code default (off), unchanged
])
def test_hg_base_default(clean, name, profile, explicit, want):
    for n in HG_BASE:
        clean.delenv(n, raising=False)
    _as(clean, profile)
    if explicit is not None:
        clean.setenv(name, explicit)
    assert _base_readers()[name]() is want


def test_early_draft_acts_on_the_27b_row_without_any_env(clean):
    """The point of group 1: with nothing set, the 27B row arms the deferred
    read AND the early draft on top of it (before: early draft armed, deferral
    off -> inert)."""
    from sglang.srt.managers import weg2_d_hostgap as H

    for n in HG_BASE + ("SGLANG_WEG2_D_EARLY_DRAFT",):
        clean.delenv(n, raising=False)
    _as(clean, "qwen27b")
    assert H.early_draft_on() is True
    assert H.defer_seq_lens_cpu_on() is True
    assert H.defer_rebuild_on() is True
