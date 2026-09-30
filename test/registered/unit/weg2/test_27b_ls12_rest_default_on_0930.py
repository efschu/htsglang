"""LS12 rest (30.09.): the six remaining 27B Leistungsschalter default-on IN THE CODE after their metal proof.

Metal: hauenh dkr27browauthorityls12bar1fs09301342 (image z30y5r @ b43c19ef1a with the LS12 markers,
profile 27b-row-authority-ls12: 15/15, needle MATCH, group death 0, watchdog 0, 0 tracebacks);
/spinning/gpu-arb/docker/pending/ls12-0930/check_ls12.py -> ALL 12 GRUEN. The six proven by their new
markers (da8464b9f0):
  (4) SGLANG_DFLASH_VERIFY_VOCAB_ARGMAX       'DFLASH-VERIFY-VOCAB-ARGMAX armed' 3, argmax rounds 512
  (5) SGLANG_VRAM_PEAK_FAST_READ              'VRAM-PEAK-FAST-READ armed' P 3 / D 3, fallback 0
  (7) SGLANG_HICACHE_LOAD_ASYNC_INDEX (+ SGLANG_HICACHE_DRAIN_AGREE_EVERY 8)
                                              'HICACHE-LOAD-ASYNC-INDEX armed' 3, 'HICACHE-DRAIN-GATE every=8' 3
  (8) SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS 2  'ADMISSION-WEDGE recovery armed after 2.0s' 6, failed 0
  (9) SGLANG_WEG2_CTL_KICK_ARRIVAL            'WEG2-FLIPFAST kick why=arrival' 4
  (10) SGLANG_WEG2_CTL_KICK_AFTER_FLIP        'WEG2-FLIPFAST kick why=after_flip' 3
Per switch, through the reader its rank uses: qwen27b on without env, blank = unset, explicit value wins,
nextflash and "no form" keep the code default.
"""

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import form as FM

NAMES = ("SGLANG_DFLASH_VERIFY_VOCAB_ARGMAX", "SGLANG_VRAM_PEAK_FAST_READ", "SGLANG_HICACHE_LOAD_ASYNC_INDEX",
         "SGLANG_HICACHE_DRAIN_AGREE_EVERY", "SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS",
         "SGLANG_WEG2_CTL_KICK_ARRIVAL", "SGLANG_WEG2_CTL_KICK_AFTER_FLIP")


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv(FM.FORM_ENV, raising=False)
    for n in NAMES:
        monkeypatch.delenv(n, raising=False)
    return monkeypatch


def _as(clean, profile, name=None, explicit=None):
    if profile is not None:
        clean.setenv(FM.FORM_ENV, _form_env(profile))
    if explicit is not None:
        clean.setenv(name, explicit)


def test_rows():
    q, n = FM.PROFILES["qwen27b"], FM.PROFILES["nextflash"]
    assert (q.d_verify_vocab_argmax, n.d_verify_vocab_argmax) == (True, False)
    assert (q.vram_peak_fast_read, n.vram_peak_fast_read) == (True, False)
    assert (q.hicache_load_async_index, n.hicache_load_async_index) == (True, False)
    assert (q.hicache_drain_agree_every, n.hicache_drain_agree_every) == (8, 1)
    assert (q.admission_wedge_recovery_s, n.admission_wedge_recovery_s) == (2.0, -1.0)
    assert (q.front_ctl_kick, n.front_ctl_kick) == (True, False)


CASES = [("qwen27b", None), ("qwen27b", ""), ("qwen27b", "0"), ("nextflash", None), (None, None)]


@pytest.mark.parametrize("profile,explicit", CASES)
def test_4_verify_vocab_argmax(clean, profile, explicit):
    from sglang.srt.layers.logits_processor import verify_local_vocab_requested

    _as(clean, profile, "SGLANG_DFLASH_VERIFY_VOCAB_ARGMAX", explicit)
    want = profile == "qwen27b" and explicit in (None, "")
    assert verify_local_vocab_requested() is want


@pytest.mark.parametrize("profile,explicit", CASES)
def test_5_vram_peak_fast_read(clean, profile, explicit):
    _as(clean, profile, "SGLANG_VRAM_PEAK_FAST_READ", explicit)
    want = profile == "qwen27b" and explicit in (None, "")
    assert envs.SGLANG_VRAM_PEAK_FAST_READ.get() is want


@pytest.mark.parametrize("profile,explicit", CASES)
def test_7_load_async_index(clean, profile, explicit):
    from sglang.srt.mem_cache.pool_host import arena_pool as A

    _as(clean, profile, "SGLANG_HICACHE_LOAD_ASYNC_INDEX", explicit)
    want = profile == "qwen27b" and explicit in (None, "")
    assert A.load_index_async() is want


@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, 8), ("qwen27b", "", 8), ("qwen27b", "1", 1), ("qwen27b", "4", 4),
    ("nextflash", None, 1), (None, None, 1)])
def test_7b_drain_agree_every(clean, profile, explicit, want):
    from sglang.srt.mem_cache import unified_radix_cache as U

    _as(clean, profile, "SGLANG_HICACHE_DRAIN_AGREE_EVERY", explicit)
    assert U._hicache_drain_agree_every() == want


@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, 2.0), ("qwen27b", "30", 30.0), ("nextflash", None, 60.0), (None, None, 60.0)])
def test_8_wedge_recovery_threshold(clean, profile, explicit, want):
    from sglang.srt.managers.scheduler_components import invariant_checker as IC

    _as(clean, profile, "SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS", explicit)
    assert IC._admission_wedge_recovery_threshold() == want


@pytest.mark.parametrize("profile,explicit", CASES)
def test_9_10_ctl_kicks(clean, profile, explicit):
    from sglang.srt.weg2 import front as F

    if explicit is not None:
        clean.setenv("SGLANG_WEG2_CTL_KICK_AFTER_FLIP", explicit)
    _as(clean, profile, "SGLANG_WEG2_CTL_KICK_ARRIVAL", explicit)
    f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="ls12", store_dir="/tmp",
                prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                flip_min_work_tokens=1)
    want = profile == "qwen27b" and explicit in (None, "")
    assert f._kick_on == {"arrival": want, "after_flip": want}
