"""Operator decision 26.09. (RM): #49 (27B 196f6a8f57, in the unified tree
unswitched since S7c) runs behind SGLANG_WEG2_ENABLE_AGENT_SPAN (NF P49
c1988ff84f), and the switch is a MODEL PROFILE field (weg2/form.py
ModelProfile.agent_span): qwen27b on (the 27B line's behaviour since RC9),
nextflash off until the NF seat releases it with a boot tag, off without a
form (the NF code default). An explicit env value always wins. The #49 rest
(SGLANG_WEG2_FRONT_SPAN_INFLIGHT) credits only when the span is on.
"""

import asyncio
import importlib.util
import logging
import os

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import form as FM
from sglang.srt.weg2 import front as F

SWITCH = "SGLANG_WEG2_ENABLE_AGENT_SPAN"


def _mid_stream_price():
    """The FS test's D-leg-2 harness (a fake D that decodes until released)."""
    spec = importlib.util.spec_from_file_location(
        "_agent_span_fs", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "test_weg2_front_span_inflight_fs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._mid_stream_price()


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    for k in (SWITCH, FM.FORM_ENV, "SGLANG_WEG2_FRONT_SPAN_INFLIGHT"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_the_registry_rows():
    assert FM.PROFILES["qwen27b"].agent_span is True
    assert FM.PROFILES["nextflash"].agent_span is False
    assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"][SWITCH] is True
    assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][SWITCH] is False


@pytest.mark.parametrize("profile,want", [("qwen27b", True), ("nextflash", False), (None, False)])
def test_the_default_is_per_profile(clean, profile, want):
    if profile is not None:
        clean.setenv(FM.FORM_ENV, _form_env(profile))
    assert envs.SGLANG_WEG2_ENABLE_AGENT_SPAN.get() is want
    assert F.SpanLRU().agent_span is want
    body = {"system": "s", "messages": [{"role": "user", "content": "u"}],
            "tools": [{"name": "Bash"}]}
    assert F.request_text(body).startswith("tools:") is want


@pytest.mark.parametrize("profile", ["qwen27b", "nextflash", None])
@pytest.mark.parametrize("explicit,want", [("1", True), ("0", False)])
def test_an_explicit_value_wins(clean, profile, explicit, want):
    if profile is not None:
        clean.setenv(FM.FORM_ENV, _form_env(profile))
    clean.setenv(SWITCH, explicit)
    assert envs.SGLANG_WEG2_ENABLE_AGENT_SPAN.get() is want
    assert F.SpanLRU().agent_span is want


def test_environ_holds_exactly_one_entry():
    import inspect

    from sglang.srt import environ

    src = inspect.getsource(environ)
    assert src.count(f"    {SWITCH} = ") == 1


def test_nextflash_drops_the_held_epoch_and_prices_pre_49(clean):
    clean.setenv(FM.FORM_ENV, _form_env("nextflash"))
    spans = F.SpanLRU()
    text = "x" * 3000
    spans.record_presence(text, 10, prompt_tokens=1000, held_epoch=4)
    assert list(spans.entries.values()) == [(text, 10, 0, None)]
    spans.record_presence("y" * 30, 0, prompt_tokens=10, held_epoch=4)
    assert len(spans.entries) == 1  # no held credit without a measured share


def test_inflight_credits_only_under_the_span(clean, caplog):
    """FS on, profile nextflash (span off): the first-content instrument line
    stays, the in-flight credit does not happen (it would bring #49's held
    price back through record_inflight)."""
    clean.setenv(FM.FORM_ENV, _form_env("nextflash"))
    clean.setenv("SGLANG_WEG2_FRONT_SPAN_INFLIGHT", "1")
    with caplog.at_level(logging.INFO):
        front, mid, _end = asyncio.run(_mid_stream_price())
    assert front.spans.agent_span is False
    assert any("LEG2-FIRST-CONTENT rid=r1" in r.getMessage() for r in caplog.records)
    assert front.counters["span_inflight_credited"] == 0
    assert mid[2] is False and mid[0] == mid[1], mid


def test_inflight_credits_under_the_27b_profile(clean):
    clean.setenv(FM.FORM_ENV, _form_env("qwen27b"))
    clean.setenv("SGLANG_WEG2_FRONT_SPAN_INFLIGHT", "1")
    front, mid, _end = asyncio.run(_mid_stream_price())
    assert front.spans.agent_span is True
    assert front.counters["span_inflight_credited"] == 1
    assert mid[2] is True
