"""DFlash draft-KV producer for Weg 2 group P (speculative/dflash_draft_kv_producer).

Hermetic. Pins the three things the producer must get right without a card:
the page hashes it publishes under are the radix cache's own position-aware
chain (continued across chunks), the chunk ring is bounded by the prefill
batch bound, and the arena's direct publish claims/copies/completes exactly
the pending slots while a producer holds no reader reference.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.utils import get_hash_str
from sglang.srt.speculative.dflash_draft_kv_producer import (
    DFlashDraftKvProduceError,
    batch_page_hashes,
    chunk_page_hashes,
    ring_capacity,
)


def _req(tokens, rid="r"):
    return SimpleNamespace(rid=rid, fill_ids=list(tokens))


def test_chain_is_position_global_and_continues_across_chunks():
    tokens = list(range(100, 140))
    whole = get_hash_str(tokens, None, page_size=1)
    assert isinstance(whole, list) and len(whole) == len(tokens)
    r = _req(tokens)
    got = chunk_page_hashes(r, 0, 16) + chunk_page_hashes(r, 16, 16) + chunk_page_hashes(r, 32, 8)
    assert got == whole
    # the radix cache chains node by node from the parent's last hash: same result
    assert get_hash_str(tokens[:16], None, page_size=1) + get_hash_str(
        tokens[16:], whole[15], page_size=1
    ) == whole


def test_prefix_hit_rebuilds_the_chain_from_zero():
    tokens = list(range(1, 33))
    whole = get_hash_str(tokens, None, page_size=1)
    r = _req(tokens)  # no cached chain, chunk starts at a prefix hit of 20
    assert chunk_page_hashes(r, 20, 12) == whole[20:]


def test_batch_hashes_follow_batch_order_and_refuse_short_fill():
    a, b = _req(list(range(10)), "a"), _req(list(range(50, 60)), "b")
    batch = SimpleNamespace(reqs=[a, b], prefix_lens=[0, 4], extend_lens=[10, 6])
    got = batch_page_hashes(batch)
    assert got == get_hash_str(a.fill_ids, None, page_size=1) + get_hash_str(
        b.fill_ids, None, page_size=1
    )[4:]
    bad = SimpleNamespace(reqs=[a], prefix_lens=[0], extend_lens=[11])
    with pytest.raises(DFlashDraftKvProduceError, match="reaches past"):
        batch_page_hashes(bad)


def test_ring_capacity_is_the_prefill_batch_bound():
    assert ring_capacity(SimpleNamespace(max_prefill_tokens=16384, chunked_prefill_size=8192)) == 16384
    assert ring_capacity(SimpleNamespace(max_prefill_tokens=0, chunked_prefill_size=8192)) == 8192
    with pytest.raises(ValueError, match="cannot be sized"):
        ring_capacity(SimpleNamespace(max_prefill_tokens=0, chunked_prefill_size=0))


# --- arena publish_direct ----------------------------------------------------


class _Arena:
    def __init__(self, states):
        self.states = states  # stem -> claim status (0 fresh, 2 complete)
        self.calls = []
        self.next_slot = 10

    def claim_slots(self, stems, totals):
        out = []
        for s in stems:
            st = self.states.get(s, 0)
            slot = self.next_slot
            self.next_slot += 1
            out.append((slot, st, 7))
        self.calls.append(("claim", list(stems)))
        return out

    def ref_slots(self, slots, delta):
        self.calls.append(("ref", list(slots), delta))
        return 1

    def complete_slots(self, slots, gens, extents):
        self.calls.append(("complete", list(slots), list(gens), extents))
        return [1] * len(slots)

    def free_slots(self, slots):
        self.calls.append(("free", list(slots)))


class _Backend:
    def _get_suffixed_key(self, key):
        return key + "|id"


def _bound_pool(monkeypatch, arena):
    from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool

    pool = ArenaMHAHostPool.__new__(ArenaMHAHostPool)
    pool.size = 0  # staging rows: none (the fixture never stages)
    pool._arena_init_fields()
    pool.arena = arena
    pool._backend = _Backend()
    pool.row_slot = {}
    pool._page_bytes = 20480
    pool._own_extents = [(0, 10240), (10240, 10240)]
    pool.ensure_bound = lambda backend, role="kv": True
    copied = []
    monkeypatch.setattr(
        pool, "_backup_arena", lambda dev, slots, idx: copied.append((slots.tolist(), idx.tolist()))
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    return pool, copied


def test_publish_direct_copies_pending_only_and_holds_no_reference(monkeypatch):
    arena = _Arena({"h2.draft-x|id": 2})  # h2 is already complete in the arena
    pool, copied = _bound_pool(monkeypatch, arena)
    n = pool.publish_direct(
        ["h1", "h2", "h3"], "draft-x", object(), torch.tensor([0, 1, 2]), _Backend()
    )
    assert n == 3
    # claimed under the drafter's component suffix, through the backend's key suffix
    assert arena.calls[0] == ("claim", ["h1.draft-x|id", "h2.draft-x|id", "h3.draft-x|id"])
    # h1 (slot 10) and h3 (slot 12) copied from ring rows 0 and 2; h2 skipped
    assert copied == [([10, 12], [0, 2])]
    completes = [c for c in arena.calls if c[0] == "complete"]
    assert completes == [("complete", [10, 12], [7, 7], pool._own_extents)]
    # the reader reference _claim took on the complete page is released again
    assert ("ref", [11], -1) in arena.calls
    assert pool._pending == {}


def test_publish_direct_counts_a_refused_claim_as_zero(monkeypatch):
    arena = _Arena({})
    pool, copied = _bound_pool(monkeypatch, arena)
    monkeypatch.setattr(pool, "_claim", lambda stems: None)
    assert pool.publish_direct(["h1"], "draft-x", object(), torch.tensor([0]), _Backend()) == 0
    assert copied == []


def test_cache_controller_direct_publish_disarms_row_transfers():
    from sglang.srt.managers.cache_controller import HiCacheController

    cc = HiCacheController.__new__(HiCacheController)
    cc.has_draft = True
    cc.draft_owner_phase = None
    cc.draft_binding_generation = None
    cc.mem_pool_device_draft = SimpleNamespace(weg2_direct_publish=True)
    assert cc.draft_tier_armed("write") is False
    assert cc.draft_tier_armed("load") is False
    # the hash-keyed L3 points stay armed for this pool
    assert cc.draft_tier_armed("l3-write") is True


def test_cache_controller_publish_routes_to_the_host_pool():
    from sglang.srt.managers.cache_controller import HiCacheController

    seen = {}

    def publish(hashes, comp, device_pool, device_indices, backend):
        seen.update(hashes=hashes, comp=comp, n=len(hashes))
        return len(hashes) - 1

    cc = HiCacheController.__new__(HiCacheController)
    cc.has_draft = True
    cc.draft_identity = "abc"
    cc.mem_pool_host_draft = SimpleNamespace(publish_direct=publish)
    cc.storage_backend = object()
    cc._draft_l3_write_issued = 0
    cc._draft_l3_write_refused = 0
    n = cc.publish_draft_rows_direct(["h1", "h2"], object(), torch.tensor([0, 1]))
    assert n == 1 and seen["comp"] == "draft-abc" and seen["hashes"] == ["h1", "h2"]
    assert (cc._draft_l3_write_issued, cc._draft_l3_write_refused) == (1, 1)
    cc.mem_pool_host_draft = SimpleNamespace()  # a non-arena host pool
    with pytest.raises(RuntimeError, match="cannot publish directly"):
        cc.publish_draft_rows_direct(["h1"], object(), torch.tensor([0]))


def test_xsn262_the_token_stream_comes_from_the_forks_req_get_fill_ids():
    """weg2xsn262: this fork's Req has no `fill_ids`; PP2 died on the first P
    prefill. The stream is read through `get_fill_ids()` (cut at the extend
    range), then a plain `fill_ids`, then origin + output."""
    from types import SimpleNamespace
    from sglang.srt.mem_cache.utils import get_hash_str
    from sglang.srt.speculative.dflash_draft_kv_producer import (
        _token_stream, chunk_page_hashes,
    )
    toks = list(range(100, 112))

    class _ForkReq:
        rid = "fork"
        origin_input_ids = toks[:8]
        output_ids = toks[8:]

        def get_fill_ids(self):
            return toks[:10]          # extend_range.end = 10

    assert _token_stream(_ForkReq()) == toks[:10]
    assert _token_stream(SimpleNamespace(fill_ids=toks[:5])) == toks[:5]
    assert _token_stream(SimpleNamespace(origin_input_ids=toks[:3], output_ids=toks[3:6])) == toks[:6]
    r = _ForkReq()
    got = chunk_page_hashes(r, 0, 10)
    assert got == get_hash_str(toks[:10], None, page_size=1)
