# SPDX-License-Identifier: Apache-2.0
"""build_env publishes the Weg-2 group identity for EVERY profile.

weg2xsn412 (24.09., the first 27B boot on the NF line): the #107 expert-map
block had been inserted between ``if group:`` and its ``else:``, so the
``env.pop("SGLANG_WEG2_GROUP")`` fired whenever NO expert map was given --
the whole 27B profile. All three P ranks then refused the exchange plan at
the first release (W68 "this rank has no Weg-2 group identity" -> W84 -> W29).
"""

import pytest

from sglang.srt.weg2 import launcher


@pytest.fixture(autouse=True)
def scrub_env(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_GROUP", raising=False)
    monkeypatch.delenv("SGLANG_MOE_EXPERT_MAP", raising=False)


def _build(**kw):
    return launcher.build_env(
        tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s",
        debug_hold=False, tag="probe", **kw,
    )


def test_the_group_survives_a_boot_without_expert_map():
    # the 27B form: a group, no expert map
    for group in ("P", "D"):
        env = _build(group=group)
        assert env.get("SGLANG_WEG2_GROUP") == group
        assert "SGLANG_MOE_EXPERT_MAP" not in env


def test_the_expert_map_form_keeps_both():
    # the Next Flash form: a group and the map
    env = _build(group="D", expert_map_path="/tmp/karte.json")
    assert env.get("SGLANG_WEG2_GROUP") == "D"
    assert env.get("SGLANG_MOE_EXPERT_MAP") == "/tmp/karte.json"


def test_no_group_pops_an_inherited_value(monkeypatch):
    # launcher output, never operator input: a value from the shell never arms
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert "SGLANG_WEG2_GROUP" not in _build(group="")
    assert "SGLANG_WEG2_GROUP" not in _build(group="", expert_map_path="/tmp/karte.json")
