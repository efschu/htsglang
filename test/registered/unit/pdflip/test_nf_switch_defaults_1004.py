# SPDX-License-Identifier: Apache-2.0
"""SWITCH-DEFAULTS 1004 (user order 04.10. ~07:15Z "alle die gefunden werden ANSCHALTEN";
rule 29.09.: proven on metal -> default ON in the code, a profile line is no substitute).

The NF profile nf-int4-h6-abl.env sets these to 1; the code had them off. They become
nextflash defaults WITHOUT any profile line, the NF line's way (pdflip/form.py):

* front + rank switches every NF process carries (``_form`` / ``export``):
  STATED_SWITCHES -> PROFILE_SWITCH_DEFAULTS, read through the published form by
  ``envs.<NAME>.get()`` on P, D and the front: PARK_COLLECT_WINDOW,
  DEPOSIT_LANE_LOOKAHEAD, D_SEAT_REWAKE;
* group-only lines (NF_ENV_P / NF_ENV_D): NEXTFLASH_GROUP_SWITCH_DEFAULTS, written by
  the launcher into that group's env only: P PREFILL_FETCH_OVERLAP, TARGETED_PREWARM;
  D TAIL_STAGE_EARLY, RESUME_WARM_FINISH, TAIL_STAGE_WORKER (CUT_WORKER_END and
  D_PARK_END were already there).

Per switch: on without a profile line under nextflash, ``0`` still turns it off, qwen27b
and no form keep the code default. RED on 1b14d1a876, GREEN after. Hermetic.
"""

import argparse
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import form as FM  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402

NF, Q = "nextflash", "qwen27b"

STATED = ("FLLIPER_PDFLIP_ENABLE_PARK_COLLECT_WINDOW", "FLLIPER_PDFLIP_DEPOSIT_LANE_LOOKAHEAD",
          "FLLIPER_PDFLIP_D_SEAT_REWAKE")
GROUP = {
    "P": ("FLLIPER_PDFLIP_ENABLE_PREFILL_FETCH_OVERLAP", "FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM"),
    "D": ("FLLIPER_PDFLIP_ENABLE_TAIL_STAGE_EARLY", "FLLIPER_PDFLIP_RESUME_WARM_FINISH",
          "FLLIPER_PDFLIP_TAIL_STAGE_WORKER", "FLLIPER_PDFLIP_ENABLE_CUT_WORKER_END",
          "FLLIPER_PDFLIP_ENABLE_D_PARK_END"),
}


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == Q
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.PdFlipForm(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    for k in list(STATED) + [n for ns in GROUP.values() for n in ns] + [FM.FORM_ENV]:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.mark.parametrize("name", STATED)
@pytest.mark.parametrize("profile,explicit,want", [
    (NF, None, True), (NF, "0", False), (Q, None, False), (None, None, False)])
def test_stated_switch_default(clean, name, profile, explicit, want):
    if profile is not None:
        clean.setenv(FM.FORM_ENV, _form_env(profile))
    if explicit is not None:
        clean.setenv(name, explicit)
    assert bool(getattr(envs, name).get()) is want


def test_qwen27b_states_none_of_them():
    q = FM.PROFILE_SWITCH_DEFAULTS.get(Q, {})
    for name in STATED:
        assert name not in q, name
    row = FM.profile_row(Q)
    flat = str(dict(getattr(row, "group_switch_defaults", {}) or {}))
    for names in GROUP.values():
        for name in names:
            assert name not in flat, name


@pytest.mark.parametrize("group", ["P", "D"])
def test_group_switches_written_into_that_group_only(group):
    ns = argparse.Namespace(profile=NF, env_p="", env_d="")
    line = L.apply_profile_group_switch_defaults(ns, environ={})
    mine = L.parse_group_env(ns.env_p if group == "P" else ns.env_d)
    other = L.parse_group_env(ns.env_d if group == "P" else ns.env_p)
    for name in GROUP[group]:
        assert mine.get(name) == "1", (name, line)
        assert f"{group}:{name}=1" in line
        assert name not in other, name


def test_group_switch_explicit_zero_wins():
    ns = argparse.Namespace(profile=NF, env_p="FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM=0",
                            env_d="FLLIPER_PDFLIP_RESUME_WARM_FINISH=0")
    L.apply_profile_group_switch_defaults(ns, environ={})
    assert L.parse_group_env(ns.env_p)["FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM"] == "0"
    assert L.parse_group_env(ns.env_d)["FLLIPER_PDFLIP_RESUME_WARM_FINISH"] == "0"
