"""rc12z17-s0 (D, 28.09. 11:22:57-11:23:14): the flush before the D->P sleep
stood 16.4 s on TP0 (`#1476 DISPATCH kind=FlushCacheReqInput dispatch_ms=16372`,
TIMELINE drain+quiesce=16552 ms), between `R12 HOST-VERDICT reset drops` and
`#1068 RESET JOIN ... joined_s=0.00` -- the reset walk over the host release
queue (`_pdflip_release_queued_refs_before_reset`), no store wait, no join.

What filled the queue: the W88 retry storm of pdflip-10-50. On every pass the
group trim cut TP0's prefetch to the workers' span (`#915 PREFETCH TRUNCATED
need=69952 got=5760 lost=64192`, 1299 lines on TP0, lost=0 on TP1/TP2) and
queued the cut tail -- read placeholders from `alloc_read`, no slot, no
reference -- one entry per 64-id page, ~1000 per pass, on TP0 only. The drain
agrees on the group MIN of the queue sizes, so a surplus one rank alone has
is never drained; the reset walked ~1.26M entries one at a time (the census
thread 23 s on the same queue). The tail has nothing to give back: it is not
queued.

RED on 1bab093912: the trim queues the placeholder tail.
"""
from __future__ import annotations

import os
from types import MethodType, SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

PAGE = 64
THRESHOLD = 256
CHUNK = 4096
#: D TP0: staging rows, arena tokens (786432 slots x 64), then placeholders
STAGING = 4096
ARENA_TOKENS = 786432 * 64


class _ArenaReadPool:
    """The calls ``prefetch_from_storage`` makes on an arena-bound host pool:
    ``alloc_read`` hands out placeholders (no budget), ``alloc`` real rows."""

    def __init__(self):
        self.read_calls = []
        self.alloc_calls = []
        self._next = 0

    def ensure_bound(self, storage_backend, role="kv"):
        return True

    def available_size(self):
        return 1 << 30

    def alloc_read(self, n):
        self.read_calls.append(n)
        base = STAGING + ARENA_TOKENS + self._next
        self._next += n
        return torch.arange(base, base + n, dtype=torch.int64)

    def alloc(self, n):
        self.alloc_calls.append(n)
        return torch.arange(n, dtype=torch.int64)


class _StagingPool(_ArenaReadPool):
    """No arena binding: the registration allocates real staging rows."""

    def ensure_bound(self, storage_backend, role="kv"):
        return False


def _stub(pool, peer_votes):
    queued = []

    def _reduce(t, op, label):
        if label == "prefetch_cut_rank":
            t[0] = 1  # a worker's span set the group MIN (Form A)
            return t
        assert label == "prefetch_participation_vote"
        t[2] = min([int(t[2].item())] + [int(v) for v in peer_votes])
        if t.numel() > 4:
            t[4] = -int(t[3].item())
        return t

    def _prefetch(req_id, host_indices, prefetch_key, last_hash, prefix_keys,
                  extra_pools=None):
        registered[req_id] = SimpleNamespace(host_indices=host_indices,
                                             key_len=len(prefetch_key))
        return SimpleNamespace()

    def _release(host_indices=None, extra_pools=None):
        if host_indices is not None and host_indices.numel():
            queued.append(host_indices)

    registered = {}
    controller = SimpleNamespace(
        mem_pool_host=pool,
        storage_backend=object(),
        prefetch_rate_limited=lambda: False,
        prefetch_tokens_occupied=0,
        append_host_mem_release=_release,
        prefetch=_prefetch,
    )
    stub = SimpleNamespace(
        enable_storage=True,
        cache_controller=controller,
        page_size=PAGE,
        prefetch_threshold=THRESHOLD,
        _prefetch_chunk_tokens=CHUNK,
        is_eagle=False,
        _components_tuple=(),
        sidecar_pool_specs=(),
        ongoing_prefetch={},
        registered=registered,
        queued=queued,
        _hicache_prefetch_symmetric=lambda: True,
        _all_reduce_attn_groups=_reduce,
        evict_host=lambda *a, **k: None,
        inc_host_lock_ref=lambda node: SimpleNamespace(to_dec_params=lambda: "l"),
        dec_host_lock_ref=lambda node, params: None,
        _retire_ongoing_prefetch=lambda rid: None,
    )
    for name in (
        "prefetch_from_storage",
        "_build_sidecar_transfers",
        "_log_prefetch_refused",
        "_log_prefetch_truncated",
        "_prefetch_line_terms",
        "_pdflip_extent_topup",
    ):
        setattr(stub, name, MethodType(getattr(UnifiedRadixCache, name), stub))
    return stub


def _issue(stub, rid, tokens):
    stub.prefetch_from_storage(rid, SimpleNamespace(key=None), list(range(tokens)))


def test_the_placeholder_tail_of_a_group_trim_is_not_queued():
    """The rc12z17-s0 form: TP0 needs 69952, the group span is 5760."""
    pool = _ArenaReadPool()
    tree = _stub(pool, peer_votes=[5760])
    _issue(tree, "pdflip-10-50", 69952)
    assert pool.read_calls == [69952]
    assert tree.registered["pdflip-10-50"].key_len == 5760
    assert tree.queued == [], (
        f"{sum(int(q.numel()) for q in tree.queued)} placeholder ids queued for release"
    )
    assert tree._pdflip_trim_placeholders_dropped == 69952 - 5760


def test_a_retry_storm_leaves_the_release_queue_empty():
    """1299 passes on the boot; 64 here -- the queue must not grow per pass."""
    pool = _ArenaReadPool()
    tree = _stub(pool, peer_votes=[5760])
    for _ in range(64):
        _issue(tree, "pdflip-10-50", 69952)
    assert sum(int(q.numel()) for q in tree.queued) == 0
    assert tree._pdflip_trim_placeholders_dropped == 64 * (69952 - 5760)


def test_a_staging_tail_is_still_released():
    """Real rows (``alloc``, no arena binding) are allocations: freed as before."""
    pool = _StagingPool()
    tree = _stub(pool, peer_votes=[5760])
    _issue(tree, "rid-staging", 69952)
    assert pool.alloc_calls == [69952]
    assert sum(int(q.numel()) for q in tree.queued) == 69952 - 5760


def test_no_trim_no_release():
    pool = _ArenaReadPool()
    tree = _stub(pool, peer_votes=[])
    _issue(tree, "rid-whole", 5760)
    assert tree.registered["rid-whole"].key_len == 5760
    assert tree.queued == []
