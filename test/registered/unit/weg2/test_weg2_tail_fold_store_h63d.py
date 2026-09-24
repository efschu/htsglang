"""fnFL2 H63d: the folded finish on the metal path -- what D can read against what P wrote.

Hermetic (no CUDA). Metal x169 (tree c286615929, arm TAIL_FOLD=1, TAIL_KEEP_MIB=512),
the 97841-token probe, its last chunk [81920, 97841) folded:

* P, all three ranks: ``WEG2-TAIL-PUBLISH refused rid=weg2-0-4: kv rows 97792 <
  cut 97840`` -- no END part, D waits ``no_parts``;
* D: ``HiCache prefetch INCOMPLETE ... completed 86016 ... deliverable=97792
  shortfall=11776`` and ``#1028B FETCH CAP kv=1528 claimed=1344 caps={mamba: 1528,
  qsa_indexer: 1344}``, re-read 2757 times -- the KV arena held all 1528 pages,
  the store's QSA-index sidecar only 1344 (the first 4096-token piece of the
  folded chunk, nothing after it).

Two defects, both on paths the H63/H63b cases never ran:

(a) THE FINISH TRUNCATES BEFORE IT PUBLISHES. The cut form ended the prompt with
    a 1-4 token forward; the extra_buffer track of that forward stays unset
    (extend < 64), and the unfinished insert at c had consumed the track of the
    chunk before (``MambaComponent.cleanup_after_caching_req``), so its finish
    saw ``mamba_last_track_seqlen = None`` and kept every row. Under the fold the
    last chunk IS the finishing forward: its track (floor_page(N) = 97792) is
    still pending at ``cache_finished_req``, the mamba component returns it as
    ``cache_len``, and the truncation freed [97792, N) and sliced ``kv_indices``
    BEFORE ``tail_handoff.publish_rows`` read them. The H63 cases called
    ``publish_rows`` directly with all N rows; the H63b edge walk asserted that
    the track VALUE is the same with and without the fold -- it is -- and never
    who consumes it before the finish.

(b) THE FLUSH RESETS THE STORE PIPELINE UNDER QUEUED WRITES. The fold moves the
    whole last chunk's publish to the finish: the retain publish issues its
    write-throughs, their acks queue the plain-sidecar store writes (#106S, the
    QSA index), and PP0's ``/flush_cache`` 50 ms later resets the tree -- the
    controller reset stops the backup thread after the operation in flight and
    starts a fresh queue, the rest is gone without a trace (#1068 RESET JOIN
    counts prefetch operations only). #1470 joins the write-throughs, not the
    store writes their acks issue. PP0 was not stopped by its own
    ``hicache_backup(n)`` blocker because it decides the flush on the idle lap
    that landed at the PREVIOUS sleep (epoch 1, 22:00:34, consumed 22:04:26 --
    PP1/PP2, deciding rank-locally, refused with ``hicache_backup(5)`` in the
    same second and drained 20 ms). Under the cut the last chunk's writes were
    issued one tail forward (>= 4 s) before the flush. No H63 case ran a store
    write at all.

Each metal-path case below is RED on c286615929 and GREEN with H63d; the
default-path cases (the cut form, a flush with nothing queued) are green on both.
"""

import logging
import threading
import time
from array import array
from queue import Empty
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_handoff as th

RID = "weg2-0-4"
PAGE, RATIO, CHUNK = 64, 4, 16384
N = 97841  # the x169 probe
ANCHOR = 97792  # floor_page(N - 1): the page prefix, the last chunk's track under the fold
C = 97840  # the cut (grain 4)
FIRST = 19  # P's sampled token (x166 "first_token=19")
SLOTS, SLOT = 3, 1
REQ_SLOTS, P_RPI = 3, 2
KV_ROWS = N + 2 * PAGE


def _join(name):
    for t in threading.enumerate():
        if t.name == name:
            t.join(10)


@pytest.fixture
def p_group(tmp_path, monkeypatch):
    """Group P: END-ANCHOR on, E2 armed, the NF extra_buffer track form."""
    import sglang.srt.managers.schedule_batch as sb
    import sglang.srt.managers.schedule_policy as sp
    import sglang.srt.mem_cache.unified_radix_cache as urc

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", True)
    monkeypatch.setattr(urc, "_WEG2_END_ANCHOR", True)
    monkeypatch.setattr(sb, "get_server_args", lambda: SimpleNamespace(
        mamba_cache_chunk_size=PAGE, mamba_checkpoint_interval=None,
        enable_mamba_extra_buffer_lazy=lambda: False))
    th._CAPTURES.clear()
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_ADOPT.override(True), \
            envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.override(True):
        yield sp, sb, urc
    th._CAPTURES.clear()


# ------------------------------------------------------------------ (a) the finish
def _p_pools(seed=63):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    g = torch.Generator().manual_seed(seed)
    fa = {3: 0, 7: 1}
    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    kv.full_attention_layer_id_mapping = dict(fa)
    kv.full_kv_pool = SimpleNamespace(
        k_buffer=[torch.randn(KV_ROWS, 2, 8, generator=g) for _ in fa],
        v_buffer=[torch.randn(KV_ROWS, 2, 8, generator=g) for _ in fa],
    )
    kv.qsa_compressed_k_buffer_pool = [torch.randn(KV_ROWS // RATIO + 1, 1, 4, generator=g) for _ in fa]
    kv.qsa_key_state_buffer_pool = [torch.randn(REQ_SLOTS * RATIO, 1, 4, generator=g) for _ in fa]
    kv.qsa_rope_position_buffer = torch.arange(REQ_SLOTS * RATIO * 3, dtype=torch.int64).view(-1, 3)
    rp = object.__new__(HybridReqToTokenPool)
    rp.mamba_map = {0: 0, 1: 1, 2: 2}
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=torch.randn(3, SLOTS, 2, 4, 4, generator=g),
        conv=[torch.randn(3, SLOTS, 6, 3, generator=g)],
    ))
    rp.req_to_token = torch.zeros(REQ_SLOTS, N + PAGE, dtype=torch.int32)
    rp.req_to_token[P_RPI, :N] = torch.arange(N, dtype=torch.int32) + PAGE  # row of token i = i + 64
    return kv, rp


class _Allocator:
    """The finish's allocator: ``free`` hands rows back -- and the next
    forward may write them, so a freed row reads as NaN from here on (a
    publish behind the free reads garbage, not the prompt's KV)."""

    def __init__(self, kv):
        self.kv = kv
        self.freed = []

    def get_kvcache(self):
        return self.kv

    def free(self, idx):
        idx = idx.to(torch.int64)
        if idx.numel():
            self.freed.append((int(idx.min()), int(idx.max()) + 1))
            for buf in self.kv.full_kv_pool.k_buffer + self.kv.full_kv_pool.v_buffer:
                buf[idx] = float("nan")


class _TrackComp:
    """The mamba component's two answers on this path, nothing else:
    ``prepare_for_caching_req`` returns the pending extra_buffer track as the
    retention (mamba_component.py ``cache_len = req.mamba_last_track_seqlen``,
    on-grid at interval None), the unfinished cleanup consumes it
    (``cleanup_after_caching_req(is_finished=False)``, the real method)."""

    def prepare_for_caching_req(self, req, insert_params, token_ids_len, is_finished):
        return req.mamba_last_track_seqlen if is_finished else None

    def cleanup_after_caching_req(self, req, is_finished, insert_result=None, insert_params=None):
        if not is_finished:
            from sglang.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent

            MambaComponent.cleanup_after_caching_req(self, req, False, None, SimpleNamespace(mamba_value=None))


class _Tree:
    """The real ``UnifiedRadixCache.cache_finished_req`` runs on this: its
    collaborators, recorded."""

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as _U

    cache_finished_req = _U.cache_finished_req
    _note_protected_beyond_retention = _U._note_protected_beyond_retention
    disable = False
    is_eagle = True  # NF: MTP, bigram keys
    bigram_anchor_exact = True  # bigram keys + a recurrent component
    page_size = PAGE
    pp_rank, pp_size = 0, 3

    def __init__(self, rp, alloc):
        self.session = SimpleNamespace(try_cache_finished_req=lambda req, **kw: False)
        self.req_to_token_pool = rp
        self.token_to_kv_pool_allocator = alloc
        self._components_tuple = (_TrackComp(),)
        self.inserted_units = None
        self.probe_tokens = None
        self.handoff_units = None

    def insert(self, params):
        self.inserted_units = len(params.key)
        return SimpleNamespace(prefix_len=0, mamba_exist=False)

    def _weg2_note_end_anchor(self, req, token_ids):
        self.probe_tokens = len(token_ids)

    def _weg2_handoff_write(self, req, radix_key):
        self.handoff_units = len(radix_key)

    def _weg2_publish_at_retain(self, req, radix_key):
        pass

    def _anchor_dec_skip(self, req, params):
        pass

    def dec_lock_ref(self, node, params, skip_swa=False):
        pass


def _p_req():
    ids = array("q", [(7 * i + 3) % 50000 for i in range(N)])
    return SimpleNamespace(
        rid=RID, origin_input_ids=ids, output_ids=array("q", [FIRST]), full_untruncated_fill_ids=ids,
        extra_key=None, prefix_indices=[], mamba_pool_idx=torch.tensor(SLOT), req_pool_idx=P_RPI,
        mamba_ping_pong_track_buffer=torch.tensor([0, 1]), mamba_next_track_idx=0, mamba_branching_seqlen=None,
        mamba_last_track_seqlen=None, cache_protected_len=0, priority=None, swa_uuid_for_lock=None, last_node=None,
        return_logprob=False, return_hidden_states=False, pop_committed_kv_cache=lambda: N,
    )


def _prefill_and_finish(p_group, fold: bool):
    """P's chunk loop for the x169 prompt on the REAL decision/track code,
    the scheduler's per-chunk hooks in its order (split, track, fold arm,
    forward; at a stash the H18 capture and the unfinished insert that
    consumes the track), then the REAL finish."""
    sp, sb, _urc = p_group
    kv, rp = _p_pools()
    alloc = _Allocator(kv)
    req = _p_req()
    adder = SimpleNamespace(rem_chunk_tokens=CHUNK, page_size=PAGE, token_to_kv_pool_allocator=alloc)
    batch = SimpleNamespace(req_to_token_pool=SimpleNamespace(get_mamba_ping_pong_other_idx=lambda i: 1 - i))
    comp = _TrackComp()
    extents, start = [], 0
    with envs.SGLANG_WEG2_ENABLE_P_TAIL_FOLD.override(fold):
        while start < N:
            length, _forced = sp.PrefillAdder._weg2_end_anchor_split(adder, req, start, min(N - start, CHUNK))
            req.prefix_indices = range(start)
            req.extend_range = SimpleNamespace(start=start, end=start + length, length=length)
            sb.ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(batch, req)
            th.arm_fold([req], alloc, PAGE, None)  # _run_batch_forward, before the forward (H63)
            rp.mamba_pool.mamba_cache.temporal[:, SLOT] += 1.0  # the forward advances the state
            extents.append((start, start + length))
            start += length
            if start < N:  # the stash: E1 capture at c (H18), the unfinished insert
                th.capture_state(req, rp, alloc, PAGE, None)
                if req.mamba_last_track_seqlen:
                    req.cache_protected_len = req.mamba_last_track_seqlen
                comp.cleanup_after_caching_req(req, is_finished=False)
        pending = req.mamba_last_track_seqlen
        prompt_kv = [b.clone() for b in kv.full_kv_pool.k_buffer]
        tree = _Tree(rp, alloc)
        tree.cache_finished_req(req)
        _join("weg2-tail-publish")
    return SimpleNamespace(kv=kv, rp=rp, alloc=alloc, tree=tree, extents=extents, pending=pending,
                           prompt_kv=prompt_kv, headers=th.headers_for(RID))


def test_fold_finish_publishes_the_end_rows_before_the_retention_frees_them(p_group):
    """x169 (a), RED on c286615929: the finish of the folded chunk truncates to
    the pending track and publishes nothing."""
    run = _prefill_and_finish(p_group, fold=True)
    assert run.extents[-1] == (81920, N)  # the folded last chunk, no own tail forward
    assert run.pending == ANCHOR  # its track is still pending at the finish
    assert len(run.headers) == 1, (
        "no END part: the finish freed [97792, 97841) before the publish read it "
        "(x169: 'WEG2-TAIL-PUBLISH refused rid=weg2-0-4: kv rows 97792 < cut 97840')")
    (h,) = run.headers
    assert h.e1 is False and h.n_parts == 3
    e = h.end
    assert (e.rows, e.groups, e.ring_rows, e.first_token) == (N - ANCHOR, (N - ANCHOR) // RATIO, N % RATIO, FIRST)
    sec = th.verify_part(h)["end"]
    rows = torch.arange(ANCHOR, N) + PAGE
    for gid, local in run.kv.full_attention_layer_id_mapping.items():
        k = sec["fa"][gid][0]
        assert not torch.isnan(k).any(), "the END rows were read after the retention freed them"
        assert torch.equal(k, run.prompt_kv[local][rows])
    # the retention still frees the unaligned rows -- after the publish
    assert (ANCHOR + PAGE, N + PAGE) in run.alloc.freed
    # the tree and D's page keys cover the page prefix (1528 pages) ...
    assert run.tree.inserted_units == run.tree.handoff_units == ANCHOR
    # ... and the END-ANCHOR probe asks about the prompt, not the retained key
    assert run.tree.probe_tokens == N


def test_cut_finish_publishes_e1_and_end_unchanged(p_group):
    """Default (fold off): the cut form's finish, green before and after H63d."""
    run = _prefill_and_finish(p_group, fold=False)
    assert run.extents[-2:] == [(81920, C), (C, N)]  # the tail as its own forward
    assert run.pending is None  # the unfinished insert at c consumed the track
    (h,) = run.headers
    assert h.e1 is True and h.spec.cut == C and h.spec.page_prefix == ANCHOR
    e = h.end
    assert (e.rows, e.ring_rows, e.first_token) == (N - ANCHOR, N % RATIO, FIRST)
    bundle = th.verify_part(h)
    rows = torch.arange(ANCHOR, N) + PAGE
    for gid, local in run.kv.full_attention_layer_id_mapping.items():
        assert torch.equal(bundle["end"]["fa"][gid][0], run.prompt_kv[local][rows])
        assert torch.equal(bundle["fa"][gid][0], run.prompt_kv[local][rows[: C - ANCHOR]])
    assert run.tree.inserted_units == ANCHOR and run.tree.probe_tokens == N


# ------------------------------------------------------------------ (b) the store
#: x169 PP0 at the finish: pages [0, 1280) were stored at the chunk publishes;
#: the retain publish split the folded node (PUBLISH-SPLIT pieces=4 window=4096,
#: the END-ANCHOR probe had cut off page 1527) -- five write-throughs, whose
#: acks (WT-ACK n=21-25, 18-28 ms) each queued one sidecar store write
STORED_BEFORE = 1280
LAST_CHUNK_WRITES = [(1280, 1344), (1344, 1408), (1408, 1472), (1472, 1527), (1527, 1528)]
PAGES = ANCHOR // PAGE  # 1528


def _controller(persisted: set, entered: threading.Event, first_write_s: float):
    """The serving controller's REAL storage pipeline (backup thread, stop,
    reset, restart); its persist step records the pages it wrote."""
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController

    cc = object.__new__(HybridCacheController)
    cc.enable_storage = True
    cc.storage_stop_event = threading.Event()
    cc.backup_skip = False
    cc.storage_backend = SimpleNamespace(check_disk_space=lambda: None)
    cc.mem_pool_host = SimpleNamespace(entries=[], anchor_entry=None, entry_map={})
    cc.write_queue, cc.load_queue, cc.ack_write_queue, cc.ack_load_queue = [], [], [], []
    cc.prefetch_tokens_occupied = 0

    def prefetch_idle():  # nothing prefetches at the flip: the loop only waits for its stop
        while not cc.storage_stop_event.is_set():
            try:
                cc.prefetch_queue.get(timeout=0.05)
            except Empty:
                pass

    def page_backup(op):  # one store write (#106S: the QSA-index sidecar of a node)
        first = not entered.is_set()
        entered.set()
        time.sleep(first_write_s if first else 0.005)
        persisted.update(int(k) for k in op.hash_value)

    cc.prefetch_thread_func = prefetch_idle
    cc._page_backup = page_backup
    cc._start_storage_threads()
    return cc


def _stop(cc):
    cc.storage_stop_event.set()
    for q in (cc.backup_queue, cc.prefetch_queue):
        q.put(None)
    for t in (cc.backup_thread, cc.prefetch_thread):
        t.join(5)


class _FlushTree:
    """PP0's tree at the flush: everything published, the write-throughs
    joined (#1470: issued=0 unbacked_left=0), the store writes queued."""

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as _U

    join_storage_backups = getattr(_U, "join_storage_backups", None)
    enable_storage = True

    def __init__(self, cc):
        self.cache_controller = cc
        self.ongoing_backup = {}
        self.ongoing_write_through = {}
        self.resets = 0

    def publish_unbacked_sweep(self, max_issue=256, **_kw):
        return {"unbacked": 0, "issued": 0, "in_flight_after": None}

    def writing_check(self, write_back=False):
        pass

    def queue_store_write(self, first_page, end_page):
        """What ``_weg2_write_plain_sidecars`` does at a write-through ack."""
        pages = list(range(first_page, end_page))
        op_id = self.cache_controller.write_storage(
            torch.arange(len(pages) * PAGE), [0] * (len(pages) * PAGE), [str(p) for p in pages],
            None, extra_pools=[], sidecar_only=True)
        self.ongoing_backup[op_id] = (SimpleNamespace(id=first_page), None)

    def reset(self):  # UnifiedRadixCache.reset: the tree's records, then the controller
        self.resets += 1
        self.ongoing_backup.clear()
        self.cache_controller.reset()


class _P0:
    """PP0 at /flush_cache: nothing runs, and the verdict is the lap that
    landed at the previous sleep (x169: epoch=1 landed 22:00:34, consumed at
    22:04:26; x166: the same at every flip)."""

    from sglang.srt.managers.scheduler import Scheduler as _S

    flush_cache = _S.flush_cache
    _weg2_join_store_writes_before_reset = getattr(_S, "_weg2_join_store_writes_before_reset", None)
    enable_hierarchical_cache = True
    chunked_req = None
    anchor_tails = None
    draft_worker = None

    def __init__(self, tree):
        self.tree_cache = tree
        self.running_batch = SimpleNamespace(is_empty=lambda: True, reqs=[])
        self.waiting_queue = []
        self.req_to_token_pool = SimpleNamespace(clear=lambda: None)
        self.token_to_kv_pool_allocator = SimpleNamespace(clear=lambda: None)
        self.grammar_manager = SimpleNamespace(clear=lambda: None, grammar_queue=[])
        self.metrics_reporter = SimpleNamespace(reset_metrics=lambda: None, is_stats_logging_rank=True)

    def group_idle_verdict(self, tp_group_verdict=False):
        return True, "group idle (epoch=1, participation=3/3, every rank agreed)"

    def _flush_zero_kv_wanted(self, zero_kv):
        return False


def _deliverable(persisted: set) -> int:
    """D's claim (#1028B, ALL_PAGES): the leading run of stored pages."""
    k = 0
    while k in persisted:
        k += 1
    return k


def _flush_after_folded_finish(writes, first_write_s=0.5):
    persisted = set(range(STORED_BEFORE))
    entered = threading.Event()
    cc = _controller(persisted, entered, first_write_s)
    tree = _FlushTree(cc)
    try:
        for first_page, end_page in writes:
            tree.queue_store_write(first_page, end_page)
        if writes:
            assert entered.wait(5)  # the backup thread is inside the first write when the flush arrives
        assert _P0(tree).flush_cache(empty_cache=False) is True
        assert tree.resets == 1
    finally:
        _stop(cc)
    return persisted


def test_flush_keeps_the_folded_chunks_store_writes(monkeypatch, caplog):
    """x169 (b), RED on c286615929: D's store read stops at 1344 of 1528 pages."""
    monkeypatch.delenv("SGLANG_HICACHE_FLUSH_PUBLISH_SWEEP", raising=False)
    with caplog.at_level(logging.INFO, logger="sglang.srt.managers.scheduler"):
        persisted = _flush_after_folded_finish(LAST_CHUNK_WRITES)
    got = _deliverable(persisted)
    assert got == PAGES, (
        f"D's store read stops at {got} of {PAGES} pages = {got * PAGE} tokens, shortfall "
        f"{(PAGES - got) * PAGE} (x169: completed 86016 deliverable=97792 shortfall=11776): the reset "
        f"dropped the store writes queued behind the one in flight")
    # the first write was in flight (the reset's own join finishes it), four were queued
    assert "#1470b FLUSH-STORE-JOIN drained=4 left=0" in caplog.text


def test_flush_with_nothing_queued_is_unchanged(monkeypatch, caplog):
    """Default path: no store write in flight -- the flush resets as before, no join line."""
    monkeypatch.delenv("SGLANG_HICACHE_FLUSH_PUBLISH_SWEEP", raising=False)
    with caplog.at_level(logging.INFO, logger="sglang.srt.managers.scheduler"):
        persisted = _flush_after_folded_finish([])
    assert _deliverable(persisted) == STORED_BEFORE
    assert "#1470b" not in caplog.text and "#1470 FLUSH-PUBLISH" in caplog.text


def test_join_waits_for_the_queue_only_and_never_hangs():
    """The join is bounded and never hangs the flush: a dead backup thread
    answers at once, a queue nobody drains is named at the bound, and a
    record whose operation is in no queue (in flight -- the reset's own join
    finishes it -- or lost) does not hold the flush at all."""
    from queue import Queue

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    q = Queue()
    for op_id in (1, 2):
        q.put(SimpleNamespace(id=op_id))
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    tree = SimpleNamespace(cache_controller=SimpleNamespace(backup_queue=q, backup_thread=dead),
                           enable_storage=True, ongoing_backup={1: None, 2: None, 3: None})
    t0 = time.perf_counter()
    assert UnifiedRadixCache.join_storage_backups(tree, 10.0)[:2] == (0, 2)
    assert time.perf_counter() - t0 < 1.0
    release = threading.Event()
    stuck = threading.Thread(target=release.wait, daemon=True)  # alive, never drains
    stuck.start()
    tree.cache_controller.backup_thread = stuck
    drained, left, waited_ms = UnifiedRadixCache.join_storage_backups(tree, 0.1)
    assert (drained, left) == (0, 2) and 90 <= waited_ms < 2000
    q.get()  # the thread took one before this join: the other stays named
    drained, left, _ms = UnifiedRadixCache.join_storage_backups(tree, 0.1)
    assert (drained, left) == (0, 1)
    q.get()
    t0 = time.perf_counter()
    assert UnifiedRadixCache.join_storage_backups(tree, 10.0)[:2] == (0, 0)  # op 3: in no queue
    assert time.perf_counter() - t0 < 1.0
    release.set()
    tree.ongoing_backup = {}
    assert UnifiedRadixCache.join_storage_backups(tree, 10.0) == (0, 0, 0.0)


def test_join_switch_is_the_1470_switch(monkeypatch):
    """SGLANG_HICACHE_FLUSH_PUBLISH_SWEEP=0 restores the old flush form whole
    (the loss comes back -- named here so the switch cannot hide it)."""
    monkeypatch.setenv("SGLANG_HICACHE_FLUSH_PUBLISH_SWEEP", "0")
    persisted = _flush_after_folded_finish(LAST_CHUNK_WRITES)
    assert _deliverable(persisted) == 1344  # the metal number: the first piece, nothing after it
