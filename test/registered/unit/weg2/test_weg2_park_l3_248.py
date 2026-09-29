"""#248 (rc12s dkrnfh91dprbar1dauer09271719, D-TP0 17:32:40): a sleeping D
pinned 5213 of the 5461 KV arena slots by reference.

THE METAL (tmp/r989/befund_248_park_l2_pinnt.md):
``ARENA-REF-HOLDERS n=7 ... tree=1231 tree_in_use=3982 sum=5213 own_held=5213``
while D slept (sleep 17:31:11, wake 17:32:59). Nothing ran on D. The
references were the dormant-hold read (#1443/#1455) of 2 parked requests
(1292 pages) and 3 requests that arrived in the sleep (3921 pages). P's claims
found "no free slot" (17:32:21-58 and 17:35:18-25), BACKUP-REFUSED
arena_claim, a parent_unbacked cascade, four empty 200s (W50 re-route
impossible). There was no path "kept -> L3 -> free": the arena eviction only
takes unreferenced pages, the claim frees without I/O, L3 held 96 MiB.

THE FIX (#248):
* the dormant hold intake only LOOKS UP (P's hand-off chain onto the request);
  the read -- reference, pin, host tree -- is issued at DORMANT-RELEASE, in
  hold order, and the #1471 settle releases the request once it is complete;
* the flip park marks every parked request (role ``park``, the chain of its
  retained span) -- kept by ORDER, like #243's hand-off;
* D's attention rank 0 copies every kept page to the L3 disk store in the
  background without freeing it (PARK-DEMOTE);
* a claim evicts (i) unkept, (ii) kept WITH an L3 copy (no I/O: the copy is the
  page, the read takes it back via arena_fill_from_disk), (iii) kept without a
  copy -- named PARK-LOST / HANDOFF-LOST.

Hermetic, CPU: real C arenas (gcc) on temp files, the real KV pool claim, the
real HiCacheFile copy / fill (pageio) against a temp directory, the real
Scheduler intake / wake release bound to a stand-in.
"""
from __future__ import annotations

import ctypes
import functools
import logging
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.managers import cache_controller as _cc  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from sglang.srt.weg2 import handoff as ho  # noqa: E402
from sglang.srt.weg2 import handoff_pending as hp  # noqa: E402

try:  # absent on the base: the tests then show the pinned hold itself
    from sglang.srt.weg2 import park_demote, park_l3
except ImportError:  # pragma: no cover - the base
    park_demote = park_l3 = None

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")

P = 4          # tokens per page
SLOTS = 8
SB = 256
S = 5          # staging rows
PARKED = "weg2-0-7"        # parked at epoch 4 (1292 pages on the metal, 3 here)
HELD = "weg2-1-14"         # arrived in the sleep (3921 pages on the metal)


class _Store:
    """The HiCacheFile surface the demoter, the claim's stage ii and the L3
    fill touch, on a temp directory (one file per stem)."""

    def __init__(self, root):
        self.root = root
        self.evictor = types.SimpleNamespace(reserve=lambda *a, **k: True,
                                             commit=lambda stem: None, abort=lambda stem: None)

    # arena pool side
    def _get_suffixed_key(self, k):
        return f"{k}.sfx"

    # HiCacheFile side
    def _path(self, stem):
        return os.path.join(self.root, stem + ".bin")

    def _stat_stems(self, stems):
        return {s: os.path.getsize(self._path(s)) for s in stems if os.path.exists(self._path(s))}


def _hicache(store):
    be = object.__new__(HiCacheFile)
    be._stat_stems = store._stat_stems
    be._sharded_path = store._path
    be._existing_path = store._path
    be._ensure_shard_dir = lambda path: os.makedirs(os.path.dirname(path), exist_ok=True)
    be._evictor = store.evictor
    be._key_geom = {"is_mla_model": False}
    be._arena_evict_to_disk = lambda arena, want: 0
    return be


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_QUEUE_REFS", "1")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path / "arena"))
    monkeypatch.setenv("SGLANG_WEG2_HANDOFF", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    (tmp_path / "l3").mkdir()
    kv = ShmArena(str(tmp_path / "arena-kv.bin"), SB, SLOTS)
    store = _Store(str(tmp_path / "l3"))
    yield types.SimpleNamespace(kv=kv, mp=monkeypatch, tmp=tmp_path, store=store, be=_hicache(store))
    kv.close()


def _kv_pool(arena, store):
    pool = object.__new__(ap.ArenaMHAHostPool)
    pool._arena_page_tokens = P
    pool.staging_rows = S
    pool.arena = arena
    pool.arena_tokens = SLOTS * P
    pool.arena_slots = SLOTS
    pool.row_slot = None
    pool._page_bytes = SB
    pool._backend = store
    pool._pending = {}
    pool._pending_mask = torch.zeros(SLOTS, dtype=torch.bool)
    pool._pending_gen = torch.zeros(SLOTS, dtype=torch.int64)
    pool._pending_fresh = torch.zeros(SLOTS, dtype=torch.bool)
    return pool


def _publish(arena, stems, fill=None):
    out = {}
    for i, stem in enumerate(stems):
        (s, st, g), = arena.claim_slots([stem], [SB])
        assert st == 0, (stem, st)
        if fill is not None:
            arena.slot_view(s, SB)[:] = bytes([fill + i]) * SB
        assert arena.complete_slots([s], [g], [(0, SB)]) == [1]
        out[stem] = s
    return out


def _foreign_ref(arena, slot, delta=1):
    c = (ctypes.c_int64 * 1)(slot)
    assert arena._lib.arena_ref_slots(arena._base, 1, c, delta) == 1


def _parked_arena(e, pool, chain=("pA", "pB", "pC")):
    """The metal after the flip park and the sleep's reset: the parked span's
    pages COMPLETE and unreferenced (RESET-RELEASE), the rest of the arena
    held by others (P's own tree) -- every other slot referenced."""
    parked = _publish(e.kv, pool._stems(list(chain)), fill=0x40)
    others = _publish(e.kv, [f"other{i}.sfx" for i in range(SLOTS - len(chain))])
    for s in others.values():
        _foreign_ref(e.kv, s)
    assert hp.mark_park(PARKED, list(chain), P)
    return parked, others


# -- (a) the hold takes no reference over the flip -----------------------------
class _Req(types.SimpleNamespace):
    pass


def _req(rid, n=12):
    return _Req(rid=rid, origin_input_ids=list(range(n)), output_ids=[], kv_arrival_seq=None,
                time_stats=types.SimpleNamespace(set_wait_queue_entry_time=lambda: None),
                extra_key=None)


def _intake_sched(reads):
    from sglang.srt.disaggregation.utils import DisaggregationMode

    s = types.SimpleNamespace()
    s._set_or_validate_priority = lambda req: True
    s._kv_arrival_ct = 0
    s.disaggregation_mode = DisaggregationMode.NULL
    s._abort_on_queued_limit = lambda req: False
    s._host_pool_available_size = lambda: 0
    s._prefetch_kvcache = lambda req, **kw: reads.append(req.rid) or "issued"
    s._apply_prefetch_deferral = lambda req, verdict, site: None
    s._ple_admit_on_intake = lambda req: None
    s.weg2_dormant = True
    s.weg2_dormant_hold = []
    s.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace())
    s.forward_ct = 0
    s.ps = types.SimpleNamespace(pp_rank=0)
    s._weg2_park_l3_defer_hold_read = functools.partial(Scheduler._weg2_park_l3_defer_hold_read, s)
    s._weg2_d_park_hold_late = lambda req: False
    return s


def test_metal_shape_the_dormant_hold_reads_nothing_in_the_sleep(env):
    """RED on the base (27e1d90738): the hold intake issued the store read
    (reference + pin for the whole sleep: own_held 5213 on the metal). Fixed:
    the held request is only looked up -- P's chain resolved onto it -- and
    no read is issued until the wake."""
    from sglang.srt.weg2.handoff_keys import CHAIN_ATTR

    assert ho.write(HELD, list(range(12)), ["hA", "hB", "hC"])
    reads = []
    s = _intake_sched(reads)
    req = _req(HELD)
    Scheduler._add_request_to_queue(s, req)
    assert s.weg2_dormant_hold == [req], "held, as before"
    assert reads == [], "no store read (no reference, no pin) while D sleeps"
    assert getattr(req, CHAIN_ATTR, None) == ["hA", "hB", "hC"], "P's chain looked up for the wake read"


def test_switch_off_is_the_pre_248_hold_read(env):
    env.mp.setenv("SGLANG_WEG2_ENABLE_PARK_L3", "0")
    reads = []
    s = _intake_sched(reads)
    Scheduler._add_request_to_queue(s, _req(HELD))
    assert reads == [HELD]


def test_group_p_is_untouched(env):
    env.mp.setenv("SGLANG_WEG2_GROUP", "P")
    reads = []
    s = _intake_sched(reads)
    Scheduler._add_request_to_queue(s, _req(HELD))
    assert reads == [HELD]


class _Tree:
    """check_prefetch_progress: a registered read is 'reading' until done."""

    def __init__(self):
        self.ongoing_prefetch = {}
        self.prefetch_loaded_tokens_by_reqid = {}
        self.cache_controller = types.SimpleNamespace(weg2_hold_rids=set())

    def check_prefetch_progress(self, rid):
        return rid not in self.ongoing_prefetch


def _wake_sched(tree, reads):
    s = types.SimpleNamespace(tree_cache=tree, waiting_queue=[], weg2_post_wake_settle=[],
                              ps=types.SimpleNamespace(tp_size=1), server_args=types.SimpleNamespace())

    def _read(req, **kw):
        reads.append(req.rid)
        tree.ongoing_prefetch[req.rid] = object()
        return "issued"

    s._prefetch_kvcache = _read
    s._apply_prefetch_deferral = lambda req, verdict, site: None
    s._weg2_refetch_one = functools.partial(Scheduler._weg2_refetch_one, s)
    s._weg2_group_min_flags = functools.partial(Scheduler._weg2_group_min_flags, s)
    s._weg2_note_store_shortfall = lambda req: None
    return s


def test_the_wake_issues_the_read_in_hold_order_and_settles_until_complete(env):
    """DORMANT-RELEASE: every looked-up request is read NOW, in hold order
    (the collectives line up on every rank), and waits in the #1471 settle
    until its read is complete. Its hand-off keys stay for that read."""
    assert ho.write(HELD, list(range(12)), ["hA", "hB", "hC"])
    reads = []
    tree = _Tree()
    s = _wake_sched(tree, reads)
    a, b = _req(PARKED), _req(HELD)
    for r in (a, b):
        park_l3.defer_hold_read(s, r)
    _cc.WEG2_HANDOFF_PAGE_KEYS[HELD] = ["hA"]
    s.weg2_dormant_hold = [a, b]
    n = Scheduler._weg2_release_dormant_hold(s)
    assert reads == [PARKED, HELD], "one read per held request, hold order"
    assert n == 0 and s.waiting_queue == [], "no request joins before its read completes"
    assert s.weg2_post_wake_settle == [a, b]
    assert os.path.exists(ho.path(HELD)) and _cc.WEG2_HANDOFF_PAGE_KEYS.get(HELD) == ["hA"], \
        "the wake read still needs P's keys"
    # the reads complete: the settle releases, the keys go
    tree.ongoing_prefetch.clear()
    s.weg2_dormant = False
    s.WEG2_POST_WAKE_SETTLE_S = 20.0
    s.WEG2_TAIL_RECOMPUTE_TOKENS = 256
    assert Scheduler._weg2_post_wake_settle_tick(s) == 2
    assert s.waiting_queue == [a, b]
    assert not os.path.exists(ho.path(HELD)) and HELD not in _cc.WEG2_HANDOFF_PAGE_KEYS


def test_the_sleep_top_up_skips_a_looked_up_request(env):
    tree = _Tree()
    s = _wake_sched(tree, [])
    r = _req(HELD)
    park_l3.defer_hold_read(s, r)
    tree.check_prefetch_progress = lambda rid: (_ for _ in ()).throw(AssertionError("no read in the sleep"))
    assert Scheduler._weg2_refetch_one(s, r, 0.0) == "complete"


# -- the park is kept by order -------------------------------------------------
def test_the_flip_park_marks_the_retained_chain_once(env):
    """park_running -> mark_parked: the tree's page keys of the retained span,
    written by the attention rank 0 only (every rank would write the same)."""
    root = types.SimpleNamespace(parent=None)
    n1 = types.SimpleNamespace(parent=root, hash_value=["pA", "pB"])
    n2 = types.SimpleNamespace(parent=n1, hash_value=["pC"])

    class _T:
        page_size = P
        is_eagle = False
        root_node = root

        def match_prefix(self, params):
            return types.SimpleNamespace(last_device_node=n2, last_host_node=None)

    s0 = types.SimpleNamespace(tp_rank=0, tree_cache=_T())
    s1 = types.SimpleNamespace(tp_rank=1, tree_cache=_T())
    req = _req(PARKED)
    assert park_l3.mark_parked(s1, [req]) == 0
    assert hp.status(PARKED)["state"] == "none"
    assert park_l3.mark_parked(s0, [req]) == 1
    st = hp.status(PARKED)
    assert st["state"] == "pending" and st["pages"] == 3 and st["page_size"] == P


# -- (b) (c) (d) the claim's stages -------------------------------------------
def test_metal_shape_a_claim_in_a_full_arena_frees_kept_pages_with_a_copy(env, caplog):
    """(b) RED on the base: the arena is full -- the parked span unreferenced
    but kept, everything else referenced. The demoter copied the park to L3;
    the claim frees two of its pages WITHOUT I/O (stage ii), names nothing
    lost, and the claim succeeds."""
    pool = _kv_pool(env.kv, env.store)
    parked, _ = _parked_arena(env, pool)
    recs = park_demote.demote_once(env.be, [pool])
    assert [(r["rid"], r["role"], r["written"], r["l3_pages"]) for r in recs] == [(PARKED, "park", 3, 3)]
    with caplog.at_level(logging.INFO):
        ids = pool.alloc_write(["new0", "new1"])
    assert ids is not None, "P's claim found room"
    drop = [r.getMessage() for r in caplog.records if "ARENA-DROP" in r.getMessage()]
    assert drop and "stages=i:0,ii:2,iii:0" in drop[-1]
    assert not [r for r in caplog.records if "LOST" in r.getMessage()]
    assert hp.status(PARKED)["state"] == "pending"


def test_a_park_before_its_copy_is_evicted_last_and_named(env, caplog):
    """(c) no copy yet: the claim takes the kept pages only as stage iii, and
    names them (PARK-LOST, status lost with the first lost page)."""
    pool = _kv_pool(env.kv, env.store)
    _parked_arena(env, pool)
    with caplog.at_level(logging.INFO):
        ids = pool.alloc_write(["new0"])
    assert ids is not None
    msgs = [r.getMessage() for r in caplog.records]
    assert any("PARK-LOST" in m and PARKED in m for m in msgs)
    assert any("stages=i:0,ii:0,iii:1" in m for m in msgs if "ARENA-DROP" in m)
    st = hp.status(PARKED)
    assert st["state"] == "lost" and st["reason"] == "evicted" and st["first_lost_page"] is not None


def test_the_wake_reads_a_demoted_page_back_byte_for_byte(env):
    """(d) the page a claim freed at stage ii comes back from L3 identical."""
    pool = _kv_pool(env.kv, env.store)
    parked, others = _parked_arena(env, pool)
    want = {st: bytes(env.kv.slot_view(s, SB)) for st, s in parked.items()}
    park_demote.demote_once(env.be, [pool])
    assert pool.alloc_write(["new0", "new1", "new2"]) is not None  # takes all three
    stems = list(parked)
    assert all(s < 0 or st != 2 for s, st in env.kv.find_slots(stems)), "evicted from L2"
    # the wake read needs room: P's own tree gives three slots back
    back = list(others.values())[:3]
    for s in back:
        _foreign_ref(env.kv, s, -1)
    env.kv.free_slots(back)
    out = HiCacheFile.arena_fill_from_disk(env.be, env.kv, stems, SB)
    assert all(o is not None for o in out), out
    for st, slot in zip(stems, out):
        assert bytes(env.kv.slot_view(slot, SB)) == want[st], st


def test_demote_copies_without_freeing_and_is_idempotent(env):
    pool = _kv_pool(env.kv, env.store)
    parked, _ = _parked_arena(env, pool)
    first = park_demote.demote_once(env.be, [pool], state={})
    assert first and first[0]["written"] == 3
    assert [st for _, st in env.kv.find_slots(list(parked))] == [2, 2, 2], "still COMPLETE in L2"
    assert env.kv.ref_census()[0] == SLOTS - 3, "the transient pins are gone"
    state = {}
    park_demote.demote_once(env.be, [pool], state=state)
    again = park_demote.demote_once(env.be, [pool], state=state)
    assert again == [], "a span on disk is not asked again"


# -- (e) every rank keeps the same ----------------------------------------------
def test_every_d_rank_keeps_the_same_park(env):
    """(e) one mark, read by every rank's pool: the same kept keys on all of
    them (Form A workers follow TP0 -- they read, TP0 writes)."""
    pools = [_kv_pool(env.kv, env.store) for _ in range(3)]
    _parked_arena(env, pools[0])
    keeps = [hp.keep_for(p) for p in pools]
    assert all(np.array_equal(k.keys, keeps[0].keys) for k in keeps)
    assert keeps[0].roles == ["park"] and keeps[0].rids == [PARKED]


def test_census_splits_handoff_and_park(env):
    pool = _kv_pool(env.kv, env.store)
    _parked_arena(env, pool)
    line = hp.census(pool)
    assert "park_kept=3/3" in line and "park_rids=1" in line and "handoff_kept=0/0" in line


# -- PA's partial park seam (TEILPARKEN_notiz_0927.md §5) -----------------------
def test_keep_role_range_puts_the_tail_first_and_is_the_same_on_three_ranks(env):
    """keep_role(rid, "park", (a, b)) on every D rank with the same inputs:
    the same return, the same mark; the demotion order takes [a, b) from the
    back first, then [0, a)."""
    pool = _kv_pool(env.kv, env.store)
    chain = ["pA", "pB", "pC", "pD"]
    outs = [hp.keep_role(PARKED, "park", (2, 4), page_keys=chain, page_size=P) for _rank in range(3)]
    assert outs == [2, 2, 2]
    rec = hp._read(os.path.join(hp._sub(hp.PARK), PARKED))
    assert rec["range"] == [2, 4] and rec["page_keys"] == chain
    (role, rid, stems), = hp.rid_spans(pool)
    assert (role, rid) == ("park", PARKED)
    assert stems == pool._stems(["pD", "pC", "pA", "pB"])


def test_keep_role_empty_window_is_a_pure_pause_and_none_drops(env):
    chain = ["pA", "pB"]
    assert hp.keep_role(PARKED, "park", (2, 2), page_keys=chain) == 0   # shortfall 0
    assert hp.status(PARKED)["state"] == "pending"                       # kept by order
    assert hp.keep_role(PARKED, None) == 0
    assert hp.status(PARKED)["state"] == "none"
    assert hp.keep_role(PARKED, None) == 0                               # idempotent
    assert hp.keep_role("", "park", (0, 1), page_keys=chain) == 0        # never raises


def test_keep_role_off_group_d_keeps_nothing(env):
    env.mp.setenv("SGLANG_WEG2_GROUP", "P")
    assert hp.keep_role(PARKED, "park", (0, 2), page_keys=["pA", "pB"]) == 0
    assert hp.status(PARKED)["state"] == "none"


def test_keep_state_counts_l2_and_l3(env):
    """keep_state: COMPLETE in the arena (l2) and with a disk copy (l3)."""
    pool = _kv_pool(env.kv, env.store)
    hp._POOLS.clear()
    hp.register_pool(pool)
    try:
        _parked_arena(env, pool)
        assert hp.keep_state(PARKED) == {"device": None, "l2": 3, "l3": 0, "pages": 3}
        park_demote.demote_once(env.be, [pool])
        assert hp.keep_state(PARKED) == {"device": None, "l2": 3, "l3": 3, "pages": 3}
    finally:
        hp._POOLS.clear()


def test_a_partial_park_frees_its_tail_first(env):
    """With [2, 3) demoted first, a claim that needs one slot frees exactly the
    tail page (stage ii) and keeps [0, 2) in the arena."""
    pool = _kv_pool(env.kv, env.store)
    chain = ["pA", "pB", "pC"]
    parked = _publish(env.kv, pool._stems(chain), fill=0x40)
    others = _publish(env.kv, [f"other{i}.sfx" for i in range(SLOTS - len(chain))])
    for s in others.values():
        _foreign_ref(env.kv, s)
    assert hp.keep_role(PARKED, "park", (2, 3), page_keys=chain, page_size=P) == 1
    park_demote.demote_once(env.be, [pool], batch=1)
    # the demoter wrote the tail first; forget the head's copies to show the order matters
    head = pool._stems(chain[:2])
    for st in head:
        os.remove(env.store._path(st))
    assert pool.alloc_write(["new0"]) is not None
    states = dict(zip(pool._stems(chain), (int(st) for _, st in env.kv.find_slots(pool._stems(chain)))))
    assert states[head[0]] == 2 and states[head[1]] == 2
    assert states[pool._stems(["pC"])[0]] != 2


# ---------------------------------------------------------------- F22 read early
# F22 (29.09.): the #248 hold read at the weight legs' START, beside the legs.
# Measured without it (#1471 SETTLE held_after_wake_s): z30w-park median 0.60 s,
# z30x2-kvdemand 0.35 s; x178 (read during the flip) 0.


def test_f22_early_read_runs_before_the_release_and_the_release_queues_it(env):
    env.mp.setenv("SGLANG_WEG2_ENABLE_WAKE_READ_EARLY", "1")
    assert ho.write(HELD, list(range(12)), ["hA", "hB", "hC"])
    reads = []
    tree = _Tree()
    s = _wake_sched(tree, reads)
    a, b = _req(PARKED), _req(HELD)
    for r in (a, b):
        park_l3.defer_hold_read(s, r)
    _cc.WEG2_HANDOFF_PAGE_KEYS[HELD] = ["hA"]
    s.weg2_dormant_hold = [a, b]
    # the legs' start: one read per held request, hold order, nothing released yet
    assert park_l3.issue_reads_at_wake_begin(s) == [a, b]
    assert reads == [PARKED, HELD] and s.waiting_queue == []
    # the legs run (~1.5 s on the metal): the reads complete meanwhile
    tree.ongoing_prefetch.clear()
    n = Scheduler._weg2_release_dormant_hold(s)
    assert reads == [PARKED, HELD], "the release issues no second read"
    assert n == 2 and s.waiting_queue == [a, b] and s.weg2_post_wake_settle == [], \
        "complete at the release: no #1471 settle"
    assert not os.path.exists(ho.path(HELD)) and HELD not in _cc.WEG2_HANDOFF_PAGE_KEYS


def test_f22_early_read_keeps_the_keys_while_the_read_is_still_short(env):
    """A read the legs did not outlast parks in the settle as before -- and the
    release must not drop P's hand-off keys the read still registers with
    (RED on 895559fed2's release: an issued read was not skipped there)."""
    env.mp.setenv("SGLANG_WEG2_ENABLE_WAKE_READ_EARLY", "1")
    assert ho.write(HELD, list(range(12)), ["hA", "hB", "hC"])
    tree = _Tree()
    s = _wake_sched(tree, [])
    b = _req(HELD)
    park_l3.defer_hold_read(s, b)
    _cc.WEG2_HANDOFF_PAGE_KEYS[HELD] = ["hA"]
    s.weg2_dormant_hold = [b]
    park_l3.issue_reads_at_wake_begin(s)
    assert Scheduler._weg2_release_dormant_hold(s) == 0
    assert s.weg2_post_wake_settle == [b]
    assert os.path.exists(ho.path(HELD)) and _cc.WEG2_HANDOFF_PAGE_KEYS.get(HELD) == ["hA"]


def test_f22_switch_off_reads_at_the_release_as_before(env):
    """Off (the default until the first series), and the 27B's D the same:
    the legs' start issues nothing, the release reads as on 895559fed2."""
    env.mp.delenv("SGLANG_WEG2_ENABLE_WAKE_READ_EARLY", raising=False)
    reads = []
    tree = _Tree()
    s = _wake_sched(tree, reads)
    a = _req(PARKED)
    park_l3.defer_hold_read(s, a)
    s.weg2_dormant_hold = [a]
    assert park_l3.issue_reads_at_wake_begin(s) == [] and reads == []
    Scheduler._weg2_release_dormant_hold(s)
    assert reads == [PARKED] and s.weg2_post_wake_settle == [a]


def test_f22_early_read_is_group_d_only(env):
    env.mp.setenv("SGLANG_WEG2_ENABLE_WAKE_READ_EARLY", "1")
    env.mp.setenv("SGLANG_WEG2_GROUP", "P")
    reads = []
    s = _wake_sched(_Tree(), reads)
    r = _req(PARKED)
    r._weg2_248_read_at_wake = True
    s.weg2_dormant_hold = [r]
    assert park_l3.issue_reads_at_wake_begin(s) == [] and reads == []
