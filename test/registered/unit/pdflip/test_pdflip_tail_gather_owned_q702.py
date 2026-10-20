"""Q-702 TAIL-GATHER-OWNED: the forward-stream tail gathers read their own indices.

Hermetic (no CUDA). Metal NF y9nf3 (boot 10040106 @ 458851283d, 9-way burst):
P PP1 (pid 707) published pdflip-34-183's END state at its finish (01:27:59Z);
``_capture_end`` queued the GDN slot gather on the FORWARD stream behind a
2.4-s forward, with ``phys`` a VIEW of the allocator's ``free_slots`` block
(``req.mamba_pool_idx = free_slots[:1][0]``). The finish then freed the slot
(``req.mamba_pool_idx = None``), the caching allocator handed the block back to
the schedule stream, the next stash wrote 78208 into it, and the gather read
78208 against the 33-slot pool: CUDA coredump ``indexSelectSmallIndex
<BFloat16, long>`` assert (srcSelectDimSize=33, index 78208), ~100 s dump,
SIGABRT; PP2 hung in the NCCL recv, PP0 in PP-RECV-OBJ, group dead.

The replay below is that stream order on the CPU: every gather issued inside
the forward-stream context is DEFERRED (its index is read when "the stream
gets there"), and between the capture and that moment the caller's storage is
recycled exactly as the allocator did. On the base the deferred index reads
the recycled value (red); with the owned copies it still reads the slot.
"""

import contextlib
import threading
from types import SimpleNamespace

import pytest
import torch

from flliper.srt.environ import envs
from flliper.srt.pdflip import tail_handoff as th

RID = "pdflip-34-183"
PAGE, RATIO = 64, 4
N = 241  # c = 240, page prefix 192
C = 240
FIRST = 151645
SLOTS = 33  # --mamba-cache-size 32 + dummy slot 0, the metal pool
SLOT = 7
REQ_SLOTS, P_RPI = 6, 4
RECYCLED = 78208  # what the next schedule-stream tensor wrote into the block


def _pools(seed=702):
    from flliper.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from flliper.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    g = torch.Generator().manual_seed(seed)
    rows = N + 2 * PAGE
    fa = {3: 0, 7: 1}
    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    kv.full_attention_layer_id_mapping = dict(fa)
    kv.full_kv_pool = SimpleNamespace(
        k_buffer=[torch.randn(rows, 2, 8, generator=g) for _ in fa],
        v_buffer=[torch.randn(rows, 2, 8, generator=g) for _ in fa],
    )
    kv.qsa_compressed_k_buffer_pool = [torch.randn(rows // RATIO + 1, 1, 4, generator=g) for _ in fa]
    kv.qsa_key_state_buffer_pool = [torch.randn(REQ_SLOTS * RATIO, 1, 4, generator=g) for _ in fa]
    kv.qsa_rope_position_buffer = torch.arange(REQ_SLOTS * RATIO * 3, dtype=torch.int64).view(-1, 3)
    rp = object.__new__(HybridReqToTokenPool)
    rp.mamba_map = {0: 0, 1: 1, 2: 2}
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=torch.randn(3, SLOTS, 2, 4, 4, generator=g).to(torch.bfloat16),
        conv=[torch.randn(3, SLOTS, 6, 3, generator=g).to(torch.bfloat16)],
    ))
    return kv, rp, SimpleNamespace(get_kvcache=lambda: kv)


class _ForwardStream:
    """The scheduler's forward stream: work issued under it runs LATER."""

    def wait_stream(self, other):  # device-only in the helper; never reached on CPU
        raise AssertionError("a CPU tensor needs no device-side wait")


@pytest.fixture
def deferred(tmp_path, monkeypatch):
    """Record every gather issued inside ``torch.cuda.stream(<forward>)`` with
    its index tensor OBJECT; ``replay()`` reads those indices afterwards, the
    way the device reads them when the forward stream reaches the kernel."""
    import flliper.srt.managers.schedule_policy as sp

    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(sp, "_PDFLIP_END_ANCHOR", True)  # group P
    th._CAPTURES.clear()
    on_fwd = [False]
    issued = []

    @contextlib.contextmanager
    def _stream(s):
        prev = on_fwd[0]
        on_fwd[0] = isinstance(s, _ForwardStream)
        try:
            yield s
        finally:
            on_fwd[0] = prev

    real = torch.Tensor.index_select

    def _index_select(self, dim, index):
        if on_fwd[0]:
            issued.append((int(self.shape[dim]), index))
        return real(self, dim, index)

    monkeypatch.setattr(torch.cuda, "stream", _stream)
    monkeypatch.setattr(th, "_record", lambda s: None)  # no device events in the desk replay
    monkeypatch.setattr(torch.Tensor, "index_select", _index_select)

    def replay():
        """(rows of the source, index values) as the deferred kernels see them."""
        return [(n, idx.reshape(-1).tolist()) for n, idx in issued]

    with envs.FLLIPER_PDFLIP_TAIL_HANDOFF.override(True), envs.FLLIPER_PDFLIP_TAIL_ADOPT.override(True), \
            envs.FLLIPER_PDFLIP_TAIL_SKIP_EXTEND.override(True):
        yield SimpleNamespace(issued=issued, replay=replay)


def _join(name):
    for t in threading.enumerate():
        if t.name == name:
            t.join(10)


def _req(free_slots, req_to_token):
    """A P request as the scheduler holds it: the mamba slot is a VIEW of the
    allocator's free list (allocator/mamba.py ``_do_alloc``: ``free_slots[:1]``,
    memory_pool.py ``req.mamba_pool_idx = mid[0]``)."""
    mid = free_slots[:1]
    ids = list(range(N))
    return SimpleNamespace(rid=RID, origin_input_ids=ids, full_untruncated_fill_ids=ids, extra_key=None,
                           extend_range=SimpleNamespace(end=C), mamba_pool_idx=mid[0],
                           req_pool_idx=P_RPI, output_ids=[FIRST], return_logprob=False,
                           return_hidden_states=False)


def test_end_gather_survives_the_finish_recycling_its_slot_block(deferred):
    kv, rp, alloc = _pools()
    stream = _ForwardStream()
    free_slots = torch.tensor([SLOT, 9, 11], dtype=torch.int64)  # the allocator's block
    req_to_token = torch.arange(REQ_SLOTS * (N + 8), dtype=torch.int64).view(REQ_SLOTS, -1) % (N + PAGE)
    req = _req(free_slots, req_to_token)
    kv_indices = req_to_token[P_RPI, :N]  # the finish's view of the request's row

    assert th.capture_state(req, rp, alloc, PAGE, stream)  # E1 at the stash of the chunk ending at c
    th.publish_rows(req, kv_indices, alloc, "pp1-707", req_to_token_pool=rp, n_parts=3)  # E2 at the finish
    _join("pdflip-tail-publish")
    assert deferred.issued, "no gather was issued on the forward stream"

    # the finish frees the slot and the request's row; the caching allocator
    # hands both blocks to the schedule stream, whose next tensors overwrite them
    req.mamba_pool_idx = None
    free_slots.fill_(RECYCLED)
    req_to_token[P_RPI].fill_(RECYCLED)

    # ... and only now does the forward stream reach the gathers
    bad = [(n, vals) for n, vals in deferred.replay() if any(v < 0 or v >= n for v in vals)]
    assert not bad, f"a forward-stream gather reads recycled indices: {bad[:3]}"
    slot_reads = [vals for n, vals in deferred.replay() if n == SLOTS]
    assert slot_reads and all(vals == [SLOT] for vals in slot_reads)


def test_owned_copy_is_a_copy_never_a_view(deferred):
    stream = _ForwardStream()
    src = torch.tensor([SLOT], dtype=torch.int64)
    own = th._stream_owned(src, stream, "unit")
    assert own.dtype == torch.int64 and own.tolist() == [SLOT]
    assert own.data_ptr() != src.data_ptr()  # never a view of the caller's storage
    src.fill_(RECYCLED)
    assert own.tolist() == [SLOT]
    # int32 (the unified pool's translate) comes back int64 and owned too
    own32 = th._stream_owned(torch.tensor([3], dtype=torch.int32), stream, "unit")
    assert own32.dtype == torch.int64 and own32.tolist() == [3]


def test_no_stream_keeps_the_stream_ordered_path_unchanged():
    src = torch.tensor([SLOT], dtype=torch.int64)
    assert th._stream_owned(src, None, "park") is src  # same stream: stream order already protects it
