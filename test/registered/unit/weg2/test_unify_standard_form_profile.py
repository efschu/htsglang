"""Operator decision 26.09. (RM, H91 base into the unified tree): the NF H91
STANDARD FORM -- the front's phase policy (H91c/c2), the D park (H91b/d), D's
seats per phase from the wake's handoff_n (H95 B/c) -- is a MODEL PROFILE
field, ``ModelProfile.standard_form``: nextflash on, qwen27b off (the 27B
byte-identical). No form = on (the NF code default). An explicit env value or
front flag always wins. R12's Form A host shadow follows ``d_layout``.
"""

import asyncio
import importlib.util
import logging
import os
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import d_seats, phase_policy  # noqa: E402
from sglang.srt.weg2 import form as FM  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SWITCHES = ("SGLANG_WEG2_STANDARD_FORM", "SGLANG_WEG2_D_PARK",
            "SGLANG_WEG2_ENABLE_D_PARK_DRAFT_KV", "SGLANG_WEG2_ENABLE_FORM_A_HOST_SHADOW")


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    for k in SWITCHES + (FM.FORM_ENV, "SGLANG_WEG2_GROUP"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _h91c():
    spec = importlib.util.spec_from_file_location(
        "_std_form_h91c", os.path.join(HERE, "test_weg2_phase_policy_h91c.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_registry_rows():
    assert FM.PROFILES["qwen27b"].standard_form is False
    assert FM.PROFILES["nextflash"].standard_form is True
    for s in SWITCHES:
        assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"][s] is False, s
        assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][s] is True, s


@pytest.mark.parametrize("profile,want", [("qwen27b", False), ("nextflash", True), (None, True)])
def test_the_env_defaults_follow_the_profile(clean, profile, want):
    if profile is not None:
        clean.setenv(FM.FORM_ENV, _form_env(profile))
    assert envs.SGLANG_WEG2_STANDARD_FORM.get() is want
    assert envs.SGLANG_WEG2_ENABLE_D_PARK_DRAFT_KV.get() is want
    assert envs.SGLANG_WEG2_ENABLE_FORM_A_HOST_SHADOW.get() is want


@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, False), ("nextflash", None, True), (None, None, True),
    ("qwen27b", "1", True), ("nextflash", "0", False), (None, "off", False)])
def test_the_d_park_follows_the_profile_on_group_d(clean, profile, explicit, want):
    env = {"SGLANG_WEG2_GROUP": "D"}
    if profile is not None:
        env[FM.FORM_ENV] = _form_env(profile)
    if explicit is not None:
        env["SGLANG_WEG2_D_PARK"] = explicit
    assert d_seats.d_park_active(env) is want
    env["SGLANG_WEG2_GROUP"] = "P"
    assert d_seats.d_park_active(env) is False  # group P never parks


@pytest.mark.parametrize("std", [True, False])
def test_the_front_defaults(std):
    ns = types.SimpleNamespace(p_phase_max_requests=None, p_pool_tokens=None,
                               d_wait_bound_s=None, p_leg1_stall_s=None)
    phase_policy.resolve_front_defaults(ns, std)
    got = (ns.p_phase_max_requests, ns.p_pool_tokens, ns.d_wait_bound_s, ns.p_leg1_stall_s)
    assert got == ((6, 262144, 60.0, 180.0) if std else (0, 0, 0.0, 0.0))
    ns = types.SimpleNamespace(p_phase_max_requests=3, p_pool_tokens=None,
                               d_wait_bound_s=0.0, p_leg1_stall_s=None)
    phase_policy.resolve_front_defaults(ns, std)
    assert ns.p_phase_max_requests == 3 and ns.d_wait_bound_s == 0.0  # written values win


def test_the_launcher_seat_defaults_are_standard_form_only():
    for profile, want in (("qwen27b", False), ("nextflash", True)):
        ns = types.SimpleNamespace(profile=profile, d_bs=L.DEFAULT_D_BS, env_d="")
        assert L.profile_standard_form(ns) is want
        L.apply_profile_d_bs_default(ns, ["--profile", profile])
        assert ns.d_bs == (L.DEFAULT_D_BS_NEXTFLASH if want else L.DEFAULT_D_BS)
        assert (L.apply_profile_d_seat_vram_default(ns) is not None) is want
        assert ("SGLANG_OPT_WEG2_D_SEAT_VRAM" in ns.env_d) is want


def _front_kwargs():
    return dict(awake="D", tag="t", store_dir="", prefill_sid=0, decode_sid=0,
                dc_reserve={}, w_s=0.0, tp_prefill_max_tokens=10, min_dwell_ms=0.0,
                idle_layout="D")


@pytest.mark.parametrize("profile,want", [("qwen27b", False), ("nextflash", True)])
def test_the_front_names_the_policy_only_under_the_standard_form(clean, caplog, profile, want):
    from sglang.srt.weg2.front import Front

    clean.setenv(FM.FORM_ENV, _form_env(profile))
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        f = Front("http://p", "http://d", **_front_kwargs())
    assert f.standard_form is want
    assert any("WEG2-PHASE-POLICY" in r.getMessage() for r in caplog.records) is want


def test_qwen27b_wake_of_d_carries_no_seat_counts(clean):
    """27B: the kv_cache resume to D is the pre-H91 body (no handoff_n)."""
    clean.setenv(FM.FORM_ENV, _form_env("qwen27b"))
    T = _h91c()

    async def body():
        async with T.Harness(awake="P", p_concurrency=8, d_bs=8) as h:
            h.p.delay = 0.05
            tasks = [h.post(f"q{i}") for i in range(3)]
            results = await asyncio.wait_for(asyncio.gather(*tasks), 40)
            assert [s for s, _ in results] == [200] * 3
            kv = [b for b in h.d.resume_bodies if b.get("tags") == ["kv_cache"]]
            assert kv and all("handoff_n" not in b and "parked_n" not in b for b in kv), kv

    asyncio.run(body())


def test_qwen27b_d_exhaustion_is_the_pre_h91_rule(clean):
    """27B: D is exhausted when its ledger is empty and nothing is handed off
    -- a prefilled request still in _ready_for_d does not hold the flip
    (rule 2 is the NF standard form's)."""
    clean.setenv(FM.FORM_ENV, _form_env("qwen27b"))
    T = _h91c()
    from sglang.srt.weg2.front import Pending

    async def body():
        async with T.Harness(admitter=False, awake="D", d_bs=1) as h:
            loop = asyncio.get_running_loop()
            ready = Pending("weg2-1-1", "/generate", {}, "x", time.time(), loop.create_future(),
                            est_prompt=5000, est_uncached=5000)
            h.front._ready_for_d.append(ready)
            h.front._sync_batch_gate()
            h.front.queue.append(Pending("weg2-1-2", "/generate", {}, "y", time.time(),
                                         loop.create_future(), est_prompt=5000, est_uncached=5000))
            assert await T._until(lambda: h.front.awake == "P", 10)
            ready.fut.cancel()

    asyncio.run(body())
