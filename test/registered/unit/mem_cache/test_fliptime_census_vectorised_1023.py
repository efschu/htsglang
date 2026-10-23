"""FLIPTIME-1023 (boot weg2_dkr27bint4arbar1int4ar10231400_a45112428e_1023_140032, 27B AutoRound, D TP2 / P PP2).

THE METAL. The flip body grew with the L2 fill: D>P total 1.6 s at 14:05Z -> 6.4 s at 14:13Z -> 9.9 s at 14:21Z,
while the weight legs stayed ~1.0 s. ``PDFLIP-SLEEP-SUB`` on D TP0 names the term: rpc_flush ``sweep`` 6 / 116 / 411 /
961 / 2161 ms and release_flush ``sweep`` 6 / 108 / 406 / 736 / 2132 ms against 1k / 30k / 94k / 193k / 260k COMPLETE
slots of the KV arena (``L3-REUSE WRITE-BEHIND complete=``); ``#1470 FLUSH-PUBLISH waited_ms`` 0-237 of it; TP1
waits the same time in ``verdict``. The term is the S109-A7 quiesce census (``l3_secured.queue_unsecured_all`` ->
``unsecured_keys_lo``): one ``slot_stem`` ctypes call + decode + encode per COMPLETE slot on the scheduler thread of
the write-behind's rank, twice per flip. Measured here (idle box, 262144 slots): 625-680 ms; in the boot's process,
under the write-behind's GIL, 1.9-2.2 s. S86 SLEEP-L3-RELEASE adds 0.5-0.9 s per D sleep from ``tolist`` /
``torch.tensor`` round trips of its 720896-slot veto vector and a ``sorted(set(...))`` digest (nodes=0, slots=0).

THE FIX. ``ShmArena.slot_stem_ptrs`` hands the slot headers' stem addresses to the same #1459 C lookup
(``HiCacheFile.l3_listed_ptrs`` -> ``L3Index.has_ptrs``) -- the same bytes, the same index, the same fail-closed
answers; ``FLLIPER_L3SEC_STEM_PTRS=0`` keeps the per-slot form. S86 reduces the veto vector with the tree's
``hicache_group_max_tensor`` (the same all-reduce, no list round trip) and digests with ``np.unique``
(byte-identical crc input).

These tests hold the EQUALITY of the two forms (the danger direction: a page reported secured that is not, which
would let a spill free a page without an L3 copy) and the cost ratio as a characterisation."""
from __future__ import annotations

import ctypes
import os
import shutil
import sys
import time
import types
import zlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache import l3_secured as l3s  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")

_REAL_UNSEC = l3s.unsecured_keys_lo   # bound at import, before the per-test conftest seam replaces it


def _arena_with_index(tmp_path, n, missing=(), stale_gen=(), sb=256, cap=1 << 16):
    """A real C arena of ``n`` COMPLETE slots, its shared record published, and a real #1459 index listing every
    stem except ``missing``; ``stale_gen`` slots get a record generation that does not match (re-completed)."""
    from flliper.srt.mem_cache import hicache_storage as hs
    from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena
    from flliper.srt.mem_cache.storage.file.l3_index import L3Index

    a = ShmArena(str(tmp_path / f"arena-{sb}.bin"), sb, n)
    stems = ["k%058x_%d" % (i * 7919, i) for i in range(n)]
    for i in range(0, n, 1024):
        part = stems[i:i + 1024]
        got = a.claim_slots(part, [sb] * len(part))
        a.complete_slots([g[0] for g in got], [g[2] for g in got], [(0, sb)])
    sl, gens, _klo, _khi = a.complete_census()
    assert len(sl) == n
    gens = np.asarray(gens, dtype=np.int64).copy()
    for s in stale_gen:
        gens[np.nonzero(np.asarray(sl) == s)[0]] += 1000
    l3s.publish(a, sl, gens)
    idx = L3Index(str(tmp_path / "l3idx.bin"), cap=cap)
    by_slot = {int(s): a.slot_stem(int(s)) for s in sl}
    miss_stems = {by_slot[s] for s in missing}
    idx.add([st for st in stems if st not in miss_stems])
    be = types.SimpleNamespace(_l3idx=idx)
    be.l3_listed = types.MethodType(hs.HiCacheFile.l3_listed, be)
    be.l3_listed_ptrs = types.MethodType(hs.HiCacheFile.l3_listed_ptrs, be)
    return a, be, np.asarray(sl, dtype=np.int64)


@pytest.fixture
def stem_ptrs(monkeypatch):
    def _set(on):
        monkeypatch.setenv(l3s.STEM_PTRS_ENV, "1" if on else "0")
    return _set


def test_slot_stem_ptrs_point_at_the_bytes_slot_stem_reads(tmp_path):
    a, _be, sl = _arena_with_index(tmp_path, 300)
    probe = np.concatenate([sl[:50], np.array([-1, a.slots, a.slots + 5], dtype=np.int64)])
    ptrs, nonempty = a.slot_stem_ptrs(probe)
    assert ptrs.dtype == np.uint64 and ptrs.shape == probe.shape
    for i, s in enumerate(probe.tolist()):
        if 0 <= s < a.slots:
            assert bool(nonempty[i])
            assert ctypes.string_at(int(ptrs[i])).decode() == a.slot_stem(s)
        else:
            assert not bool(nonempty[i]), "an out-of-range slot must read as no stem (never listed)"


def test_free_slot_with_empty_stem_reads_empty(tmp_path):
    from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

    a = ShmArena(str(tmp_path / "arena-e.bin"), 256, 16)
    ptrs, nonempty = a.slot_stem_ptrs(np.arange(16))
    for i in range(16):
        assert a.slot_stem(i) == "" and not bool(nonempty[i])


@pytest.mark.parametrize("n,missing,stale", [(512, (), ()), (2048, (3, 17, 900, 2047), (5, 6)),
                                             (4096, tuple(range(0, 4096, 97)), tuple(range(1, 4096, 389)))])
def test_unsecured_keys_lo_fast_form_equals_per_slot_form(tmp_path, stem_ptrs, monkeypatch, n, missing, stale):
    a, be, sl = _arena_with_index(tmp_path, n, missing=missing, stale_gen=stale)
    seen = {}

    def _req(arena, slots):
        seen.setdefault(os.environ[l3s.STEM_PTRS_ENV], []).append(sorted(int(x) for x in np.asarray(slots)))
        return len(slots)

    monkeypatch.setattr(l3s, "request", _req)
    stem_ptrs(False)
    old_keys = _REAL_UNSEC(a, be)
    old_n = _REAL_UNSEC(a, be, keys=False)
    f0 = dict(l3s.FAST_STATS)
    stem_ptrs(True)
    new_keys = _REAL_UNSEC(a, be)
    new_n = _REAL_UNSEC(a, be, keys=False)
    assert l3s.FAST_STATS["fast"] >= f0["fast"] + 2, "the array form did not run"
    assert np.array_equal(old_keys, new_keys) and old_keys.dtype == new_keys.dtype
    assert old_n == new_n == len(set(missing) | set(stale))
    assert seen.get("0") == seen.get("1"), "the write-behind was asked for different slots"


def test_secured_fast_form_equals_per_slot_form(tmp_path, stem_ptrs):
    missing, stale = (1, 2, 50, 777), (3, 900)
    a, be, sl = _arena_with_index(tmp_path, 1024, missing=missing, stale_gen=stale)
    probe = np.concatenate([sl, np.array([-1, a.slots, 10 ** 6], dtype=np.int64), sl[::7]])
    stem_ptrs(False)
    old = l3s.secured(a, probe, be)
    stem_ptrs(True)
    new = l3s.secured(a, probe, be)
    assert np.array_equal(old, new)
    bad = set(missing) | set(stale)
    for i, s in enumerate(probe.tolist()):
        assert bool(new[i]) == (0 <= s < a.slots and s in set(sl.tolist()) and s not in bad)


def test_no_index_opened_lists_nothing_on_both_forms(tmp_path, stem_ptrs):
    """Fail closed: a backend whose index is not open lists no stem -- every COMPLETE page reads unsecured."""
    a, be, sl = _arena_with_index(tmp_path, 256)
    be._l3idx = None
    stem_ptrs(True)
    assert not l3s.secured(a, sl, be).any()
    assert _REAL_UNSEC(a, be, keys=False) == 256


def test_backend_without_the_array_form_keeps_the_per_slot_form(tmp_path, stem_ptrs):
    a, be, sl = _arena_with_index(tmp_path, 256, missing=(4,))
    plain = types.SimpleNamespace(_l3idx=be._l3idx, l3_listed=be.l3_listed)   # no l3_listed_ptrs
    stem_ptrs(True)
    f0 = l3s.FAST_STATS["fallback"]
    assert _REAL_UNSEC(a, plain, keys=False) == 1
    assert l3s.FAST_STATS["fallback"] > f0


def test_census_cost_characterisation(tmp_path, stem_ptrs):
    """Characterisation (not a verdict): 65536 COMPLETE slots, per-slot vs array form. Measured on the desk box
    (262144 slots): 625-680 ms against 128-143 ms -- the C index lookup is the rest, and it releases the GIL. Bound
    here: the array form at most half the per-slot time."""
    a, be, _sl = _arena_with_index(tmp_path, 65536, missing=(9,), cap=1 << 18)

    def best(on):
        stem_ptrs(on)
        b = 1e9
        for _ in range(3):
            t = time.perf_counter()
            _REAL_UNSEC(a, be, keys=False)
            b = min(b, time.perf_counter() - t)
        return b * 1e3

    old_ms, new_ms = best(False), best(True)
    print("FLIPTIME-1023 census ms per-slot=%.1f array=%.1f slots=65536" % (old_ms, new_ms))
    assert new_ms < old_ms / 2.0


# --- S86 -----------------------------------------------------------------------------------------------------------


def _old_digest(slots):
    arr = torch.tensor(sorted(set(int(s) for s in slots)), dtype=torch.int64).numpy().tobytes()
    return int(zlib.crc32(arr)) & 0x7FFFFFFF


@pytest.mark.parametrize("vals", [[], [5], [3, 1, 2, 3, 3, 0], list(range(1000, 0, -3)) + [7, 7, 7],
                                  [720895, 0, 12345]])
def test_s86_digest_is_byte_identical_to_the_previous_form(vals):
    from flliper.srt.pdflip import sleep_l3_release as S86

    want = _old_digest(vals)
    assert S86._digest(vals) == want
    assert S86._digest(torch.tensor(vals, dtype=torch.int64)) == want
    assert S86._digest(np.asarray(vals, dtype=np.int64)) == want


def test_s86_digest_of_the_agreed_set_tensor():
    from flliper.srt.pdflip import sleep_l3_release as S86

    agreed = torch.zeros(720896, dtype=torch.int64)
    agreed[torch.arange(0, 720896, 13)] = 1
    want = _old_digest(torch.nonzero(agreed == 0).reshape(-1).tolist())
    t = time.perf_counter()
    got = S86._digest(torch.nonzero(agreed == 0).reshape(-1))
    ms = (time.perf_counter() - t) * 1e3
    print("FLIPTIME-1023 S86 digest ms=%.1f (720896 slots)" % ms)
    assert got == want


def test_s86_veto_reduce_takes_the_tensor_and_a_list_reducer_still_works():
    """The tree's ``hicache_group_max`` answers a tensor with a tensor (no list round trip); an injected reducer that
    answers a list (the rank threads of the S86 tests, a cache without HiCache collectives) is converted as before."""
    from flliper.srt.pdflip import sleep_l3_release as S86

    v = torch.tensor([0, 1, 0, 0, 1], dtype=torch.int64)
    a = S86._vec(v.clone())
    b = S86._vec([int(x) for x in v.tolist()])
    c = S86._vec(list(v))                      # a list of 0-d tensors (``list(values)`` of a tensor)
    assert a.dtype == b.dtype == c.dtype == torch.int64
    assert a.tolist() == b.tolist() == c.tolist() == v.tolist()


def test_unified_tree_group_max_tensor_in_is_the_same_all_reduce(monkeypatch):
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    calls = []
    t = object.__new__(UnifiedRadixCache)

    def ar(tensor, op, label="hicache"):
        calls.append((tensor.dtype, tuple(tensor.shape), str(op), label, tensor.device.type))
        tensor.mul_(2)

    monkeypatch.setattr(t, "_all_reduce_attn_groups", ar, raising=False)
    vals = [0, 3, 1, 0]
    out_l = UnifiedRadixCache.hicache_group_max(t, vals, label="s86_sleep_l3_release/veto")
    src = torch.tensor(vals, dtype=torch.int64)
    out_t = UnifiedRadixCache.hicache_group_max(t, src, label="s86_sleep_l3_release/veto")
    assert calls[0] == calls[1]
    assert isinstance(out_l, list) and isinstance(out_t, torch.Tensor)
    assert out_l == out_t.tolist() == [0, 6, 2, 0]
    assert src.tolist() == vals, "the caller's vector must not be reduced in place"


def test_a_stand_in_index_without_the_pointer_form_keeps_the_per_slot_form(tmp_path, stem_ptrs):
    """A stand-in ``_l3idx`` (tests, an older index) without ``has_ptrs``: ``l3_listed_ptrs`` says so (None) and the
    per-slot ``l3_listed`` answers -- never a silent "nothing listed"."""
    a, be, sl = _arena_with_index(tmp_path, 128, missing=(3,))
    real = be._l3idx

    class _Idx:
        def has(self, stems):
            return real.has(stems)

    be._l3idx = _Idx()
    stem_ptrs(True)
    f0 = l3s.FAST_STATS["fallback"]
    assert _REAL_UNSEC(a, be, keys=False) == 1
    assert l3s.FAST_STATS["fallback"] > f0


def test_an_l3_listed_replaced_on_the_instance_is_honoured(tmp_path, stem_ptrs):
    a, be, sl = _arena_with_index(tmp_path, 64)
    be.l3_listed = lambda stems: [False] * len(stems)        # a stand-in answer: nothing listed
    stem_ptrs(True)
    assert _REAL_UNSEC(a, be, keys=False) == 64


# --- S83 VICTIM-CENSUS at the flush (``#1470 FLUSH-PUBLISH issued=0 unbacked_left=1 waited_ms=1343 / 1753``) -------------


def test_slot_refs_np_equals_the_per_slot_read(tmp_path):
    a, _be, sl = _arena_with_index(tmp_path, 512)
    a.ref_slots([int(s) for s in sl[:100]], +1)
    a.ref_slots([int(s) for s in sl[50:60]], +2)
    got = a.slot_refs_np()
    assert got.dtype == np.int64 and got.shape == (a.slots,)
    assert got.tolist() == a.slot_refs(list(range(a.slots)))


def test_s83_victim_census_other_only_is_the_same_count_as_before(tmp_path):
    """The census line (an instrument) prints the same ``other_only`` from the arrays as from the per-slot loop: slots some
    process references that this process's ledger does not hold."""
    import threading

    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    a, _be, sl = _arena_with_index(tmp_path, 512)
    a.ref_slots([int(s) for s in sl[:200]], +1)               # 200 slots referenced (by "everyone")
    held = torch.zeros(300, dtype=torch.int32)                 # this process holds 120 of them; the ledger is SHORTER
    held[torch.arange(0, 240, 2)] = 1                          # than the arena (the old form's ``s >= numel`` branch)
    a._ledger = types.SimpleNamespace(lock=threading.Lock(), held=held)

    class _NoArr:
        def __init__(self, inner):
            self._a = inner

        def __getattr__(self, k):
            if k == "slot_refs_np":
                raise AttributeError(k)
            return getattr(self._a, k)

    t = object.__new__(UnifiedRadixCache)
    t.root_node = types.SimpleNamespace(children={})
    t.cache_controller = None
    lines = []
    for arena in (_NoArr(a), a):
        t.__dict__.pop("_s83_census_last", None)
        lines.append(UnifiedRadixCache.s83_victim_census(t, "flush_end", pool=types.SimpleNamespace(arena=arena)))
    oo = [ln.split("other_only=")[1].split()[0] for ln in lines]
    refs = np.asarray(a.slot_refs(list(range(a.slots))))
    hh = np.zeros(a.slots, dtype=np.int64)
    hh[:300] = held.numpy()
    want = int(((refs > 0) & (hh == 0)).sum())
    assert oo[0] == oo[1] == str(want) and want > 0, (oo, want)


# --- D-DECODE-GAP: the store presence probes get a name (boot 1023_144422 14:49:12-36Z) -------------------------------


def test_decode_gap_names_store_probes_that_ate_the_rest(monkeypatch, caplog):
    """34 gaps of 450-580 ms ``cause=other`` with ``rest`` 400-500 ms: D re-probed two refused hand-backs (keys=118709 /
    107619) every round. The probe's thread CPU now rides the gap (``probe=n:keys:wall:cpu``) and names it
    ``store_probe`` when it is at least half of the unnamed rest; a probe-free gap stays ``other``."""
    import logging
    from types import SimpleNamespace

    from flliper.srt.pdflip import d_decode_gap as G

    class _Mode:
        def is_decode(self):
            return True

        def is_extend(self):
            return False

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    for e in (G.ENV, G.ENV_MS, G.ENV_CENSUS_S):
        monkeypatch.delenv(e, raising=False)
    dec = SimpleNamespace(forward_mode=_Mode(), batch_size=lambda: 2)
    sched = SimpleNamespace(tp_rank=0, _pdflip_wake_seq=1)
    with caplog.at_level(logging.INFO, logger=G.__name__):
        G.on_batch(sched, dec, now=100.0)
        G.note_probe(118709, 230.0, 215.0)
        G.note_probe(107619, 210.0, 200.0)
        G.on_batch(sched, dec, now=100.5)                     # 500 ms gap, ~415 ms of probe CPU
        G.on_batch(sched, dec, now=101.0)                     # 500 ms gap, no probe
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(G.MARK + " gap_ms")]
    assert len(lines) == 2, lines
    assert "cause=store_probe" in lines[0] and "probe=2:226328:440:415" in lines[0], lines[0]
    assert "cause=other" in lines[1] and "probe=-" in lines[1], lines[1]


def test_batch_exists_v2_reports_its_probe_to_the_gap_instrument(monkeypatch):
    from flliper.srt.mem_cache import hicache_storage as hs
    from flliper.srt.pdflip import d_decode_gap as G

    got = []
    monkeypatch.setattr(G, "note_probe", lambda k, w, c: got.append((k, w, c)))
    assert getattr(hs.HiCacheFile.batch_exists_v2, "__wrapped__", None) is not None
    fake = types.SimpleNamespace(_canonical_probe_mismatch=lambda: "stand-in")
    hs.HiCacheFile._probe_mismatch_count = 10 ** 6 + 1   # keep the refusal's log quiet
    res = hs.HiCacheFile.batch_exists_v2(fake, ["a", "b", "c"])
    assert got and got[0][0] == 3 and got[0][1] >= 0.0 and got[0][2] >= 0.0
    assert int(getattr(res, "kv_hit_pages", 0) or 0) == 0
