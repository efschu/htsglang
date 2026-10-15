"""ZERO-ANCHOR 1015, part 2: the store is complete to the last anchor when P reads.

(A) Typ A -- D's tail reaches the store in the flip's flush, after P's read
(boot 133855, epoch 127): weg2-126-628 served on D (``#59 FINISH-TRACK
cache_len=45824``, front ``credited`` 45568), D runs ``write_back`` and its 50
un-backed nodes were published only at the flip (``PUBLISH-SWEEP unbacked=50
issued=37`` 15:17:09.1, ``WRITE-BACK DRAIN n=38``). P's dormant read of the
follow-up weg2-126-651 completed in the same second at the store's 38656
(``READ-STAGES pages=604``); the front had hinted 45568 (``PLE-HINT
start_hint=45568``). PP0 now holds its told while dormant and re-reads once
awake when its read ended a page or more below the hint (epochs 103 / 127 / 149
numbers below); no progress -> the told goes out (the existing cap).

(B) D-park resume -- the kept span loses its L3 copies to the LRU (boot 133855,
weg2-40-179): ``#248 PARK-DEMOTE pool=kv pages=1832 written=8 l3_pages=1832``
(1824 pages relied on older copies), stage ii freed the kept slots against
them, ``EVICTION-BG ... toward cap 150 GB`` ran at 14:12/14:20, P's claim at the
resume: ``#1028B kv=865`` (the shared fork 55360 survived, the private tail
did not). Now a copy a kept page relies on -- and its QSA twin -- is made MRU.

RED on 525c7345c3: no hint trigger (A), the relied-on copies stay at the LRU
head (B)."""
from __future__ import annotations

import importlib.util
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import weg2_store_told as T  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import PoolName  # noqa: E402
from sglang.srt.weg2 import short_read as SR  # noqa: E402

PAGE = 64


class _Tree:
    def __init__(self, have):
        self.have = have
        self.prefetch_loaded_tokens_by_reqid = {}

    def completed_prefetch_tokens(self, rid):
        return self.have


class _Sched:
    def __init__(self, dormant=False):
        self.calls = []
        self.weg2_dormant = dormant
        self.page_size = PAGE

    def _prefetch_kvcache(self, req, *a, **k):
        self.calls.append(req.rid)
        return "issued"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in (T.P_HELD_REREAD_ENV, T.P_HINT_REREAD_ENV):
        monkeypatch.delenv(env, raising=False)
    for d in (SR._HELD, SR._MORE, SR._HINT):
        d.clear()
    yield
    for d in (SR._HELD, SR._MORE, SR._HINT):
        d.clear()


def _reap_whole(rid, have):
    """The store held exactly what the read landed (hit == completed): the old
    trigger (store_holds_more) stays silent."""
    SR.note_reap(rid, requested_pages=have // PAGE + 100, hit_pages=have // PAGE,
                 completed_tokens=have, page_size=PAGE)


def test_epoch_127_weg2_126_651_rereads_after_the_flush():
    rid = "weg2-126-651"
    _reap_whole(rid, 38656)
    assert not SR.store_holds_more(rid)
    SR.note_hint(rid, 45568)
    req = types.SimpleNamespace(rid=rid)
    asleep = _Sched(dormant=True)
    # dormant: D's flush is still landing -- hold the told, issue nothing
    assert T._p_held_short_reread(asleep, _Tree(38656), req, rid) is True
    assert asleep.calls == []
    awake = _Sched(dormant=False)
    assert T._p_held_short_reread(awake, _Tree(38656), req, rid) is True
    assert awake.calls == [rid]
    # the re-read reached 45568: nothing left below the hint -> the told
    _reap_whole(rid, 45568)
    assert T._p_held_short_reread(awake, _Tree(45568), req, rid) is False
    assert awake.calls == [rid]


def test_no_progress_after_one_reread_publishes_the_told():
    """Epoch 103 weg2-102-517: hint 50176, the store never held more than
    45312 anchored -- one re-read, unmoved, then the told (no livelock)."""
    rid = "weg2-102-517"
    _reap_whole(rid, 45312)
    SR.note_hint(rid, 50176)
    s, req, tree = _Sched(), types.SimpleNamespace(rid=rid), _Tree(45312)
    assert T._p_held_short_reread(s, tree, req, rid) is True
    _reap_whole(rid, 45312)
    assert T._p_held_short_reread(s, tree, req, rid) is False
    assert s.calls == [rid]


@pytest.mark.parametrize("rid,hint,have,reread", [
    ("weg2-102-516", 67840, 65792, True),     # epoch 103: 2048 above the anchor
    ("weg2-148-1036", 93184, 92608, True),    # epoch 149: 576
    ("weg2-148-1038", 103680, 99008, True),   # epoch 149: 4672
    ("weg2-148-1035", 73216, 75200, False),   # epoch 149: P read past the hint
    ("weg2-148-1037", 3008, 3008, False),     # epoch 149: nothing hinted beyond
    ("weg2-x-1", 38656 + 63, 38656, False),   # less than a page beyond
])
def test_epoch_103_149_trigger(rid, hint, have, reread):
    _reap_whole(rid, have)
    SR.note_hint(rid, hint)
    s = _Sched()
    assert T._p_held_short_reread(s, _Tree(have), types.SimpleNamespace(rid=rid), rid) is reread
    assert s.calls == ([rid] if reread else [])


def test_switch_off_keeps_the_old_trigger_only(monkeypatch):
    monkeypatch.setenv(T.P_HINT_REREAD_ENV, "0")
    rid = "weg2-126-651"
    _reap_whole(rid, 38656)
    SR.note_hint(rid, 45568)
    s = _Sched()
    assert T._p_held_short_reread(s, _Tree(38656), types.SimpleNamespace(rid=rid), rid) is False
    assert s.calls == []


def test_the_scheduler_records_the_front_hint(monkeypatch):
    """The real ``Scheduler.handle_ple_prefetch_hint`` keeps the hint for
    PP0's told (and still hands it to the PLE admit as before)."""
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.models import qwen4_exp_ple_admit as ple

    seen = []
    monkeypatch.setattr(ple, "admit_ple_hint", lambda rid, ids, *a, **k: seen.append((rid, k.get("start_hint"))))
    sched = types.SimpleNamespace(chunked_prefill_size=16384, weg2_dormant=True, page_size=PAGE)
    Scheduler.handle_ple_prefetch_hint(
        sched, types.SimpleNamespace(rid="weg2-126-651", input_ids=[1, 2, 3], start_hint=45568))
    assert SR.hint_tokens("weg2-126-651") == 45568
    assert seen == [("weg2-126-651", 45568)]


# --- (B) relied-on L3 copies become MRU ------------------------------------

pytestmark_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")


def _qs():
    p = os.path.join(os.path.dirname(__file__), "..", "mem_cache", "test_l3_qsa_pair_writers_0930.py")
    spec = importlib.util.spec_from_file_location("_t_qs_pair_za2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def _awake_gate():
    try:
        from sglang.srt.mem_cache import l3_write_behind as gate
    except ImportError:
        yield
        return
    gate._reset_for_tests()
    yield
    gate._reset_for_tests()


def _parked_on_old_copies(tmp_path, monkeypatch):
    """weg2-40-179 in miniature: the kept span's KV + QSA pages went to L3
    long ago; younger files were written after them (the LRU tail)."""
    qs = _qs()
    monkeypatch.setenv("SGLANG_WEG2_L3_WRITE_BEHIND_S", "0")
    be = qs._boot(tmp_path, monkeypatch, "park")
    be._evictor.set_limits(max_size_bytes=1 << 30)
    kv, q = be._arenas[qs.KV_TOTAL], be._arenas[qs.Q_TOTAL]
    kept = qs._hashes("k")
    for i, h in enumerate(kept):
        qs._put(kv, be._get_suffixed_key(h), qs.KV_TOTAL, 0x40 + i)
        qs._put(q, be._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}"), qs.Q_TOTAL, 0x80 + i)
    be._arena_copy_pages_to_disk(q, [be._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}") for h in kept])
    be._arena_copy_pages_to_disk(kv, [be._get_suffixed_key(h) for h in kept])
    young = qs._hashes("y")
    for i, h in enumerate(young):
        qs._put(kv, be._get_suffixed_key(h), qs.KV_TOTAL, 0x10 + i)
    be._arena_copy_pages_to_disk(kv, [be._get_suffixed_key(h) for h in young])
    stems = [be._get_suffixed_key(h) for h in kept]
    twins = [be._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}") for h in kept]
    return qs, be, stems, twins, [be._get_suffixed_key(h) for h in young]


def _oldest(be, n):
    return list(be._evictor._lru.keys())[:n]


@pytestmark_gcc
def test_park_demote_makes_the_relied_on_copies_mru(tmp_path, monkeypatch, _awake_gate):
    qs, be, stems, twins, young = _parked_on_old_copies(tmp_path, monkeypatch)
    assert set(_oldest(be, 2 * qs.PAGES)) == set(stems) | set(twins)   # the LRU head
    out = be.arena_copy_to_disk(be._arenas[qs.KV_TOTAL], stems)       # the #248 demote
    assert out["on_disk"] == qs.PAGES and out.get("refreshed") == 2 * qs.PAGES
    head = _oldest(be, qs.PAGES)
    assert set(head) == set(young), "the kept span is no longer the next victim"


@pytestmark_gcc
def test_claim_room_stage_ii_refreshes_the_copy_it_frees_against(tmp_path, monkeypatch, _awake_gate):
    qs, be, stems, twins, young = _parked_on_old_copies(tmp_path, monkeypatch)
    kv = be._arenas[qs.KV_TOTAL]
    cands = [c for c in kv.evict_candidates(64) if kv.slot_stem(c[0]) in set(stems)]
    assert len(cands) == qs.PAGES
    sec = be.arena_secure_to_disk(kv, cands)
    assert sec["on_disk"] == qs.PAGES and sec.get("refreshed") == 2 * qs.PAGES
    assert set(_oldest(be, qs.PAGES)) == set(young)


@pytestmark_gcc
def test_refresh_switch_off(tmp_path, monkeypatch, _awake_gate):
    monkeypatch.setenv("SGLANG_WEG2_L3_REFRESH_KEPT", "0")
    qs, be, stems, twins, young = _parked_on_old_copies(tmp_path, monkeypatch)
    out = be.arena_copy_to_disk(be._arenas[qs.KV_TOTAL], stems)
    assert out.get("refreshed") == 0
    assert set(_oldest(be, 2 * qs.PAGES)) == set(stems) | set(twins)
