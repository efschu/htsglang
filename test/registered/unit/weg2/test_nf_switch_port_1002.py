# SPDX-License-Identifier: Apache-2.0
"""SCHALTER-HALBPORT 1002 (NF seat; audit /spinning/gpu-arb/docs/SCHALTER-HALBPORT-AUDIT-1002.md):
the half-ported switches become nextflash REGISTRY defaults (weg2/form.py ModelProfile ->
PROFILE_SWITCH_DEFAULTS, read through the published form on P, D and the front), not
profile-env lines. Per switch: the nextflash row turns it on without any env, an explicit
value wins, the qwen27b row states nothing (its switch_defaults stay byte-identical), no
form keeps the code default -- and the switch leaves a metal marker when it acts.

Plus the VISION-SYNC LAW (user 02.10. ~08:00Z, both lines): vision runs only synchronously;
the launcher refuses SGLANG_WEG2_VISION_ASYNC / SGLANG_WEG2_P_ROW_VISION_ASYNC on.

Red on 0e1967fd36 (the NF y6y head), green after. Hermetic: no GPU, no boot.
"""

import argparse
import asyncio
import collections
import logging
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import form as FM  # noqa: E402

NF, Q = "nextflash", "qwen27b"

#: the switches the nextflash row now states, with the value it states
NF_STATED = {
    "SGLANG_WEG2_ENABLE_P_ANCHOR_PRESENCE": True,
    "SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS": 2.0,
    "SGLANG_HICACHE_LOAD_ASYNC_INDEX": True,
    "SGLANG_WEG2_VISION_FLIP_URGENT": True,
    "SGLANG_BARLINK_BAR1_CANON_ORDER": True,
    "SGLANG_VRAM_PEAK_FAST_READ": True,
    "SGLANG_WEG2_DC_OFF_PATH": True,
    "SGLANG_WEG2_QUIESCE_FAST": True,
    "SGLANG_WEG2_CTL_KICK_ARRIVAL": True,
    "SGLANG_WEG2_CTL_KICK_AFTER_FLIP": True,
    "SGLANG_WEG2_CENSUS_O1_EVICT": True,
}


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == Q
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    for k in list(NF_STATED) + [FM.FORM_ENV, "SGLANG_WEG2_STORE_SHORT_TAIL",
                                "SGLANG_WEG2_VISION_ASYNC", "SGLANG_WEG2_P_ROW_VISION_ASYNC"]:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _as(clean, profile):
    if profile is None:
        clean.delenv(FM.FORM_ENV, raising=False)
    else:
        clean.setenv(FM.FORM_ENV, _form_env(profile))


# ---------------------------------------------------------------------------
# the registry rows
# ---------------------------------------------------------------------------


def test_nextflash_row_states_the_ported_switches():
    got = FM.PROFILE_SWITCH_DEFAULTS[NF]
    for name, want in NF_STATED.items():
        assert got.get(name) == want and type(got.get(name)) is type(want), name
    # the two DFLASH-only HG switches stay off on NF (only CANON_ORDER is ported)
    assert got["SGLANG_WEG2_D_EARLY_DRAFT"] is False
    assert got["SGLANG_DFLASH_ACCEPT_SYNC_FUSED"] is False


def test_qwen27b_row_states_none_of_them():
    """The qwen27b row stays byte-identical: no STATED field set, no new key."""
    row = FM.PROFILES[Q]
    for fld, _ in FM.STATED_SWITCHES:
        assert getattr(row, fld) is None, fld
    assert row.vision_arg_default is False
    got = FM.PROFILE_SWITCH_DEFAULTS[Q]
    for name in NF_STATED:
        if name == "SGLANG_BARLINK_BAR1_CANON_ORDER":
            assert got[name] is True        # the HG bundle (d_hostgap_levers), unchanged
        else:
            assert name not in got, name
    assert "SGLANG_WEG2_STORE_SHORT_TAIL" not in str(dict(row.group_switch_defaults))


@pytest.mark.parametrize("profile,explicit,want", [
    (NF, None, True), (NF, "0", False), (Q, None, False), (None, None, False)])
def test_p_anchor_presence_default(clean, profile, explicit, want):
    _as(clean, profile)
    if explicit is not None:
        clean.setenv("SGLANG_WEG2_ENABLE_P_ANCHOR_PRESENCE", explicit)
    assert envs.SGLANG_WEG2_ENABLE_P_ANCHOR_PRESENCE.get() is want


@pytest.mark.parametrize("profile,explicit,want", [
    (NF, None, 2.0), (NF, "60", 60.0), (Q, None, -1), (None, None, -1)])
def test_admission_wedge_recovery_default(clean, profile, explicit, want):
    from sglang.srt.managers.scheduler_components import invariant_checker as IC

    _as(clean, profile)
    if explicit is not None:
        clean.setenv("SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS", explicit)
    got = envs.SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS.get()
    assert got == want and type(got) is type(want)          # qwen27b/no form: the int -1 as before
    assert IC._admission_wedge_recovery_threshold() == (want if want > 0 else 60.0)


@pytest.mark.parametrize("profile,explicit,want", [
    (NF, None, True), (NF, "0", False), (Q, None, False), (None, None, False)])
def test_quiesce_fast_and_vram_peak_fast_read_default(clean, profile, explicit, want):
    _as(clean, profile)
    for name in ("SGLANG_WEG2_QUIESCE_FAST", "SGLANG_VRAM_PEAK_FAST_READ"):
        if explicit is not None:
            clean.setenv(name, explicit)
        assert getattr(envs, name).get() is want, name


@pytest.mark.parametrize("profile,explicit,want", [
    (NF, None, True), (NF, "0", False), (Q, None, False), (None, None, False)])
def test_load_async_index_default_on_p_and_d(clean, profile, explicit, want, caplog):
    """NF-P 10020634: WEG2-START-LOADING mamba.idx p90 795 ms -- P runs it too now."""
    from sglang.srt.mem_cache.pool_host import arena_pool as AP

    clean.setattr(AP, "_LOAD_ASYNC_INDEX_SEEN", [])
    _as(clean, profile)
    if explicit is not None:
        clean.setenv("SGLANG_HICACHE_LOAD_ASYNC_INDEX", explicit)
    with caplog.at_level(logging.INFO, logger=AP.logger.name):
        assert AP.load_index_async() is want
        assert AP.load_index_async() is want
    armed = [r for r in caplog.records if "HICACHE-LOAD-ASYNC-INDEX armed" in r.getMessage()]
    assert len(armed) == (1 if want else 0)          # the metal marker, once per process


@pytest.mark.parametrize("profile,explicit,want", [
    (NF, None, True), (NF, "0", False), (Q, None, False), (None, None, False)])
def test_census_o1_evict_default(clean, profile, explicit, want, caplog):
    from sglang.srt.mem_cache import producer_phase_census as PPC

    clean.setattr(PPC, "_o1_evict", None)
    _as(clean, profile)
    if explicit is not None:
        clean.setenv("SGLANG_WEG2_CENSUS_O1_EVICT", explicit)
    with caplog.at_level(logging.INFO, logger=PPC.__name__):
        assert PPC.census_o1_evict_armed() is want
    armed = [r for r in caplog.records if "KR CENSUS-O1-EVICT armed" in r.getMessage()]
    assert len(armed) == (1 if want else 0)
    clean.setattr(PPC, "_o1_evict", None)


@pytest.mark.parametrize("profile,explicit,want", [
    (NF, None, True), (NF, "0", False), (Q, None, False), (None, None, False)])
def test_front_switches_default(clean, profile, explicit, want):
    from sglang.srt.weg2 import front as FR

    _as(clean, profile)
    for name in (FR.DC_OFF_PATH_ENV, FR.CTL_KICK_ARRIVAL_ENV, FR.CTL_KICK_AFTER_FLIP_ENV,
                 FR.VISION_FLIP_URGENT_ENV):
        if explicit is not None:
            clean.setenv(name, explicit)
        assert FR._env_switch_on_or_profile(name) is want, name
    assert FR.vision_flip_urgent() is want


def test_front_reads_its_switches_through_the_profile():
    import inspect

    from sglang.srt.weg2 import front as FR

    src = inspect.getsource(FR.Front.__init__)
    assert "_env_switch_on_or_profile(env) for why, env in CTL_KICK_REASONS.items()" in src
    assert "self._dc_off_path = _env_switch_on_or_profile(DC_OFF_PATH_ENV)" in src
    assert "self.vision_flip_urgent = vision_flip_urgent()" in src


@pytest.mark.parametrize("profile,explicit,want", [
    (NF, None, True), (NF, "0", False), (Q, None, True), (None, None, False)])
def test_bar1_canon_order_default(clean, profile, explicit, want):
    from sglang.srt.distributed.device_communicators import barlink_bar1 as B

    _as(clean, profile)
    if explicit is not None:
        clean.setenv("SGLANG_BARLINK_BAR1_CANON_ORDER", explicit)
    assert B.canon_order_on() is want


@pytest.mark.parametrize("profile", [NF, Q, None])
def test_seq_sync_batch_default_is_the_proven_value(clean, profile):
    """Item 290 (03.10.): 256 MiB / 128 units is a CODE default for every
    form (proven at the 27B metal, xsn123); it used to be 64/32 and NF ran the
    unproven-for-it value. No registry row states it (a code default, not a
    profile switch); see test_seq_sync_batch_default_290.py for the cost."""
    from sglang.srt.weg2 import weight_exchange_bounce as WX

    _as(clean, profile)
    clean.delenv(WX.SEQ_SYNC_BATCH_MIB_ENV, raising=False)
    clean.delenv(WX.SEQ_SYNC_BATCH_UNITS_ENV, raising=False)
    assert WX.seq_sync_batch() == (256 << 20, 128)
    assert WX.SEQ_SYNC_BATCH_MIB_ENV not in FM.PROFILE_SWITCH_DEFAULTS[NF]


# ---------------------------------------------------------------------------
# launcher: per-group STORE_SHORT_TAIL, the vision axis default
# ---------------------------------------------------------------------------


def test_store_short_tail_on_d_only_through_the_group_defaults():
    from sglang.srt.weg2 import launcher as L

    ns = argparse.Namespace(profile=NF, env_p="", env_d="")
    line = L.apply_profile_group_switch_defaults(ns, environ={})
    assert "D:SGLANG_WEG2_STORE_SHORT_TAIL=1" in line
    assert L.parse_group_env(ns.env_d)["SGLANG_WEG2_STORE_SHORT_TAIL"] == "1"
    assert "SGLANG_WEG2_STORE_SHORT_TAIL" not in L.parse_group_env(ns.env_p)
    ns_q = argparse.Namespace(profile=Q, env_p="", env_d="")
    assert L.apply_profile_group_switch_defaults(ns_q, environ={}) is None
    assert ns_q.env_d == "" and ns_q.env_p == ""


def _vis_ns(profile, vision="off"):
    return argparse.Namespace(profile=profile, weg2_vision=vision, teardown=False)


def test_vision_axis_default_nextflash_transient():
    from sglang.srt.weg2 import launcher as L

    assert FM.PROFILES[NF].vision == "transient" and FM.PROFILES[NF].vision_arg_default
    ns = _vis_ns(NF)
    line = L.apply_profile_vision_default(ns, ["--profile", NF])
    assert ns.weg2_vision == "transient" and line.startswith(L.VISION_DEFAULT_MARKER)
    ns = _vis_ns(NF)                                           # a given flag wins
    assert L.apply_profile_vision_default(ns, ["--weg2-vision", "off"]) is None
    assert ns.weg2_vision == "off"
    ns = _vis_ns(Q)                                            # qwen27b: parser default stays
    assert L.apply_profile_vision_default(ns, ["--profile", Q]) is None
    assert ns.weg2_vision == "off"


def test_vision_axis_default_runs_before_the_form_is_resolved():
    import inspect

    from sglang.srt.weg2 import launcher as L

    src = inspect.getsource(L.main)
    assert src.index("apply_profile_vision_default(") < src.index("weg2_form.resolve_form(")


# ---------------------------------------------------------------------------
# VISION-SYNC LAW: the launcher refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("where,name,val,refused", [
    ("os", "SGLANG_WEG2_VISION_ASYNC", "1", True),
    ("os", "SGLANG_WEG2_P_ROW_VISION_ASYNC", "1", True),
    ("p", "SGLANG_WEG2_VISION_ASYNC", "true", True),
    ("d", "SGLANG_WEG2_P_ROW_VISION_ASYNC", "on", True),
    ("os", "SGLANG_WEG2_VISION_ASYNC", "0", False),          # the law's own value passes
    ("p", "SGLANG_WEG2_VISION_ASYNC", "0", False),
    (None, None, None, False),
])
def test_vision_async_is_refused_at_launch(where, name, val, refused):
    from sglang.srt.weg2 import launcher as L

    env, env_p, env_d = {}, "", ""
    if where == "os":
        env[name] = val
    elif where == "p":
        env_p = f"{name}={val}"
    elif where == "d":
        env_d = f"{name}={val}"
    ns = argparse.Namespace(profile=Q, env_p=env_p, env_d=env_d)
    line = L.vision_async_refusal(ns, environ=env)
    if refused:
        assert line.startswith("WEG2 VISION-ASYNC refused") and name in line
        assert "VISION-SYNC LAW" in line
    else:
        assert line is None


def test_vision_async_refusal_is_wired_into_main():
    import inspect

    from sglang.srt.weg2 import launcher as L

    src = inspect.getsource(L.main)
    assert "vision_async_refusal(ns)" in src and "raise SystemExit(_vis_async)" in src


def test_vision_async_code_default_off_for_every_profile(clean):
    from sglang.srt.weg2 import vision_rank_runner as vrr

    for profile in (NF, Q, None):
        _as(clean, profile)
        assert vrr.vision_async_on() is False, profile


# ---------------------------------------------------------------------------
# metal markers
# ---------------------------------------------------------------------------


def test_vram_peak_fast_read_marker(clean, caplog):
    from sglang.srt.model_executor import vram_family_census as VFC

    _as(clean, NF)
    clean.setattr(VFC, "_FAST_READ_SEEN", set())
    clean.setattr(torch._C, "_cuda_memoryStats",
                  lambda dev: {"allocated_bytes": {"all": {"peak": 3 << 20}}}, raising=False)
    clean.setattr(torch.cuda, "current_device", lambda: 0)
    with caplog.at_level(logging.INFO, logger=VFC.logger.name):
        assert VFC._max_allocated_bytes(torch.cuda) == 3 << 20
        assert VFC._max_allocated_bytes(torch.cuda) == 3 << 20
    armed = [r.getMessage() for r in caplog.records if "VRAM-PEAK-FAST-READ armed" in r.getMessage()]
    assert len(armed) == 1 and "first read 3 MiB" in armed[0]


def test_admission_wedge_recovery_marker(clean, caplog):
    import threading

    from sglang.srt.managers.scheduler_components import invariant_checker as IC

    _as(clean, NF)
    clean.setattr(IC, "make_admission_wedge_poller", lambda s: (lambda: None))
    stop = threading.Event()
    stop.set()
    with caplog.at_level(logging.INFO, logger=IC.logger.name):
        t = IC.create_admission_wedge_watchdog(object(), poll_interval=0.001, stop=stop)
        t.join(timeout=5.0)
    said = [r.getMessage() for r in caplog.records if "recovery armed after" in r.getMessage()]
    assert len(said) == 1 and "ADMISSION-WEDGE recovery armed after 2.0s" in said[0]
    assert "profile" in said[0]


def test_admission_wedge_no_marker_without_override(clean, caplog):
    import threading

    from sglang.srt.managers.scheduler_components import invariant_checker as IC

    _as(clean, Q)
    clean.setattr(IC, "make_admission_wedge_poller", lambda s: (lambda: None))
    stop = threading.Event()
    stop.set()
    with caplog.at_level(logging.INFO, logger=IC.logger.name):
        IC.create_admission_wedge_watchdog(object(), poll_interval=0.001, stop=stop).join(5.0)
    assert not [r for r in caplog.records if "recovery armed after" in r.getMessage()]


def test_ctl_kick_marker_at_powers_of_two(caplog):
    from sglang.srt.weg2 import front as FR

    f = object.__new__(FR.Front)
    f._kick_on = {"arrival": True, "after_flip": True}
    f.counters = collections.Counter()
    f._ready_for_d = None

    async def go():
        for _ in range(5):
            f._kick_controller("arrival")

    with caplog.at_level(logging.INFO, logger=FR.logger.name):
        asyncio.run(go())
    said = [r.getMessage() for r in caplog.records if "WEG2-FLIPFAST kick why=arrival" in r.getMessage()]
    assert [s.split(" n=")[1].split()[0] for s in said] == ["1", "2", "4"]
