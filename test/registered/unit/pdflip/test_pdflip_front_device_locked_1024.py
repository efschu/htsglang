"""#215b FRONT-DEVICE-LOCKED (SOLL v2 R7, R11; follow-up of d-prefix-price-1024 section 5.1 on the KVD heads).

D's lifetime price (#215) credits the device rows the load-back MAPS below a Mamba cut (``locked_device_gap``: device
rows another holder keeps FULL-locked). The front priced the same request from D's probe, which answers ``device=0``
from the same Mamba cut -- so with KVD the head never got its AGE PLAN and never reached D's corrected gate.

Metal specimen: 27B int4ar ``_1024_003732`` (/spinning/docker-acceptance/27b/evidence), pdflip-18-188 / 18-189:
  front 01:02:39,192  D-PROBE rid=pdflip-18-188 n=100380 device=0 host=0 store=96704 covered=96704 uncached_d=3676
  front 01:02:43,727  ARRIVAL-SEAT ... kv need=104476 > free=66163 [need = prompt 100380 - shared 0 + decode 4096 ...]
                      SEAT-AGE-PLAN deficit=38313
  D     01:02:45      #823 HEAD-VOTE ANCHOR matched=78567 (device 0 + host 78567) anchor_depth=96704
                      -> 18137 rows of the shared system prompt on the device, locked by three running decodes
  front 01:03:21      USAGE-DETAILS rid=pdflip-18-188 cached_device=18137 cached_l2=78567
  #215: D's price 104477 -> 86340 (= 100380 - 18137 + 4096 + 1 page)

Danger directions:
  * Ueberbuchung -- o1 (unlocked device rows are not reported), o2 (locked rows past the priced covered depth are not
                    credited), o3 (the credit is max(session, locked), never their sum), o4 (ranks that disagree give
                    no number)
  * Verhungern   -- v1 (specimen: the front prices 86340 and plans the displacement), v2 (the probe walk reports every
                    locked node of the path, a partial child's slice too), v3 (an old D without the field: as before),
                    v4 (the answer carries the field end to end: parse -> price)
"""
from __future__ import annotations

import asyncio
import os
import time
from array import array

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip.front import Front  # noqa: E402

HEAD, VICTIM = "pdflip-18-188", "pdflip-18-189"
N, STORE, KEY = 100380, 96704, 96774
SHARED = 18137
CLIP = 4096
FREE = 66163                  # the front's room at 01:02:43,727 (need 104476 > free 66163, deficit 38313)
LEAVING, RESERVE = 54190 - SHARED, 401      # KVD: the victim's guaranteed release (36053 + 401)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", "0")
    for k in ("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "FLLIPER_PDFLIP_KV_FRESH_MS", "FLLIPER_PDFLIP_EXACT_CLASS",
              "FLLIPER_PDFLIP_DUAL_LAYOUT"):
        monkeypatch.delenv(k, raising=False)


# ------------------------------------------------------------------------------------------------ the D answer
def _body(spans=None):
    b = {"success": True, "ids": dmp.ids_to_b64(np.arange(N, dtype=np.int64)), "device_tokens": 0, "host_tokens": 0,
         "key_match_depth": KEY, "mm_spans": []}
    if spans is not None:
        b["device_locked_spans"] = [list(s) for s in spans]
    return b


def _price(spans=None, store=STORE):
    ans = dmp.parse_answer(_body(spans))
    assert ans is not None
    return dmp.exact_price(ans, store)


def test_v4_the_answer_carries_the_locked_rows_end_to_end():
    xp = _price([(0, SHARED)])
    assert (xp.covered, xp.uncached, xp.device_locked) == (STORE, N - STORE, SHARED)


def test_v3_an_old_d_without_the_field_prices_as_before():
    assert _price(None).device_locked is None


def test_o2_locked_rows_past_the_priced_covered_depth_are_not_credited():
    # a locked node reaching past what the store/tree covers: no continuation past it, D computes those rows (review
    # SOLL 2: only spans that END before covered are credited -- there the load-back maps them)
    xp = _price([(0, SHARED), (STORE - 100, STORE + 50)], store=STORE)
    assert xp.device_locked == SHARED


def test_o2b_first_turn_of_a_new_session_whose_cache_is_just_the_locked_prompt_gets_no_credit():
    """Review SOLL 2: the cached part is only the shared prompt (store depth = its end): D has no state at the split
    node and no host tail to load back -- it recomputes the prompt (#215 credits 0 without a load-back extent)."""
    assert _price([(0, SHARED)], store=SHARED).device_locked == 0
    assert _price([(0, SHARED)], store=SHARED + 64).device_locked == SHARED


# ------------------------------------------------------------------------------------------------ front harness
def _front(reading):
    f = Front("http://p", "http://d", "D", "fdl", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=12288)
    f.awake, f.state, f.admit_d = "D", "serving", True

    async def _kv():
        return reading

    async def _no_pool():
        return None

    f._arrival_seat_kv_reading = _kv
    f._d_reading_if_armed = _no_pool
    now = time.time()
    f.__dict__.setdefault("_asr_arrive", {}).update({HEAD: now - 30.0, VICTIM: now - 10.0})

    class _S:
        rid, tokens = VICTIM, 54133

    f._d_seats_live = [_S()]
    f.groups["D"].outstanding[VICTIM] = now
    return f, now


def _reading():
    return {"available": FREE, "evictable": 0, "decode_clip": CLIP, "price_extra": 1,
            "running_rids": [VICTIM], "displaceable_rids": [VICTIM], "admitted_rids": [], "ledger_rids": [],
            "decode_reserve_by_rid": {VICTIM: RESERVE}, "leaving_rows_by_rid": {VICTIM: LEAVING}}


def test_v1_specimen_the_front_prices_86340_and_plans_the_displacement():
    f, now = _front(_reading())
    f.__dict__["_probe_price"] = {HEAD: _price([(0, SHARED)])}
    fits, why, deficit = asyncio.run(f._arrival_seat_kv(N, 64000, HEAD, N - STORE))
    basis = Front._asr_kv_basis(f)
    assert basis["need"] == N - SHARED + CLIP + 1 == 86340
    assert not fits and deficit == 86340 - FREE
    # KVD: the victim's guaranteed release 36053 + 401 covers the deficit 20177 -- the head goes to D
    assert Front._arrival_seat_age_verdict(f, HEAD, now - 30.0, True, deficit, basis=basis) == [VICTIM]


def test_v3b_an_old_d_keeps_the_old_need_and_no_plan():
    f, now = _front(_reading())
    f.__dict__["_probe_price"] = {HEAD: _price(None)}
    fits, why, deficit = asyncio.run(f._arrival_seat_kv(N, 64000, HEAD, N - STORE))
    basis = Front._asr_kv_basis(f)
    assert basis["need"] == N + CLIP + 1
    assert Front._arrival_seat_age_verdict(f, HEAD, now - 30.0, True, deficit, basis=basis) is None


def test_o3_the_credit_is_the_larger_of_session_and_locked_never_their_sum():
    f, _now = _front(_reading())
    f.__dict__["_probe_price"] = {HEAD: _price([(0, SHARED)])}
    f._asr_shared_credit = lambda rid, est, unc, d_has=None: 20000      # a running session predecessor holds 20000
    asyncio.run(f._arrival_seat_kv(N, 64000, HEAD, N - STORE))
    assert Front._asr_kv_basis(f)["need"] == N - 20000 + CLIP + 1


# ------------------------------------------------------------------------------------------------ D's probe walk
def _real_cache():
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "mem_cache", "test_unified_radix_cache_unittest.py")
    spec = importlib.util.spec_from_file_location("_fdl_unified_fixture", os.path.abspath(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.get_device = lambda: "cpu"
    cache, alloc, _ = m.build_fixture(m.CacheConfig(page_size=1, components=(ComponentType.FULL,), kv_size=256,
                                                    max_context_len=256))
    return cache, alloc


def _ins(cache, alloc, seq):
    from flliper.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
    from flliper.srt.mem_cache.radix_cache import RadixKey

    cache.insert(InsertParams(key=RadixKey(array("q", seq)), value=alloc.alloc(len(seq))))
    return cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq)))).last_device_node


PROMPT = list(range(1, 21))            # 20 rows: the shared system prompt, scaled


def test_v2_the_probe_reports_the_locked_prompt_of_another_session():
    cache, alloc = _real_cache()
    other = _ins(cache, alloc, PROMPT + list(range(100, 110)))
    cache.inc_lock_ref(other)                                    # a running decode of ANOTHER session holds it
    head = PROMPT + list(range(300, 340))
    spans = dmp.peek_locked_device_spans(cache, array("q", head), None, len(head) - 1)
    assert spans == [(0, 20)]
    # a partial match inside a locked node: the would-be split parent carries the slice and the lock
    spans = dmp.peek_locked_device_spans(cache, array("q", PROMPT[:12] + [999] * 5), None, 16)
    assert spans == [(0, 12)]
    # two locked nodes along the path, merged
    deeper = _ins(cache, alloc, PROMPT + list(range(100, 110)) + list(range(500, 504)))
    cache.inc_lock_ref(deeper)
    seq = PROMPT + list(range(100, 110)) + list(range(500, 504)) + [7, 7]
    assert dmp.peek_locked_device_spans(cache, array("q", seq), None, len(seq) - 1) == [(0, 34)]


def test_o1_unlocked_device_rows_are_not_reported():
    cache, alloc = _real_cache()
    _ins(cache, alloc, PROMPT + list(range(100, 110)))         # on the device, nobody holds it: evictable
    head = PROMPT + list(range(300, 340))
    assert dmp.peek_locked_device_spans(cache, array("q", head), None, len(head) - 1) == []


def test_o4_ranks_that_disagree_give_no_number_and_are_named(caplog):
    agreed = dmp.locked_agreed
    assert agreed([(0, 20)], 1, 20, 20, HEAD, 20, 20) == [[0, 20]]
    n0 = dmp.LOCKED_FALLBACKS["disagree"]
    dmp._locked_last_log.clear()
    with caplog.at_level("WARNING"):
        assert agreed([(0, 20)], 1, 12, 20, HEAD, 20, 20) is None
    assert dmp.LOCKED_FALLBACKS["disagree"] == n0 + 1 and "D-PROBE-LOCKED-FALLBACK kind=disagree" in caplog.text
    assert agreed([(0, 20)], 0, 20, 20, HEAD, 20, 20) is None     # a rank could not read it
    # review SOLL 3: the same row count at different places ([0,20) vs [10,30)) -- the span-end fingerprint differs
    assert agreed([(0, 20)], 1, 20, 20, HEAD, 20, 30) is None


def test_o5_the_locked_credit_never_exceeds_the_prompt():
    """Review NIT 2: a prompt shorter than the locked rows (the probe's ids against a shorter front estimate) is
    credited at most its own length."""
    f, _now = _front(_reading())
    f.__dict__["_probe_price"] = {HEAD: _price([(0, SHARED)])}
    asyncio.run(f._arrival_seat_kv(10000, 64000, HEAD, 0))
    assert Front._asr_kv_basis(f)["locked_credit"] == 10000
