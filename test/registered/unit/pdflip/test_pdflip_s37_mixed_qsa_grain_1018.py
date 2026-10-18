"""S37 (18.10.): rc10 death = MIXED-D (S32, PR #36933) x QSA compression (K11 grain).

fLLiper rc10 (NF d655d8f617), boot ...10181121_d655d8f617_1018_112119, D-TP1 11:24:32Z:
``RuntimeError: QSA-PREFIX-GRAIN rid=pdflip-0-1 prefix=142791 ratio=4 row=1 rows_bad=1`` right after the first
``MIXED-D mixed n=2 n_prefill_tok=23 n_decode_rows=1``. ``mix_with_running`` turns each running decode into a
1-token extend whose prefix is seq_len - 1 (any value); the QSA extend write plan reads a compressed group's members
from this forward's packed rows and so demands group-aligned prefixes (K11). PR #36933 does not know QSA.

The root, not a plaster (user order 18.10.: mixed chunk for short D prefills stays ON): a tail row IS a decode step.
Its token always goes to its pending-ring slot and the group its length completes is compressed from the ring --
the decode rule of ``_qsa_build_write_plan`` (start = end - (len % ratio == 0)), the source the decode path and
the E2 tail hand-off (tail_adopt) already use. The prefill rows of the same forward keep the extend plan.

Hermetic: the real ``QwenSparseAttnBackend.init_forward_metadata`` on CPU with a fake QSA pool, the real
``QSAIndexer.update_key_state_and_compress`` control flow (the compress kernels recorded).
* repro (red before S37): a MIXED batch whose tail prefix % 4 != 0 builds its metadata (no QSA-PREFIX-GRAIN);
* the tail plan of a MIXED forward == the DECODE plan of the same request at the same length (write slot, group
  end, ring members), for a tail that closes a group and one that does not;
* the tail token goes to its ring slot even when its length closes the group (decode store, never the dump rows);
* the prefill rows' extend plan is unchanged by the tails; a pure EXTEND with a split prefix still stops by name (K11);
* the indexer compresses the extend groups from the packed rows and THEN the tail groups from the ring.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.layers.attention.qsa.qsa_indexer import QSAIndexer  # noqa: E402
from flliper.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend  # noqa: E402
from flliper.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

R = 4            # compress ratio
CTX = 64


class _Pool:
    qsa_index_kv_heads = 1
    qsa_index_head_dim = 8
    qsa_compressed_page_size = 16
    qsa_compress_ratio = R
    qsa_token_topk = 16
    qsa_block_topk = 4

    def __init__(self, capacity=512):
        self.token_k = torch.zeros(capacity, 1, 8)
        self.rope_positions = torch.zeros(capacity, 3, dtype=torch.int64)
        self.qsa_rope_position_buffer = self.rope_positions
        self.stored = []

    def set_qsa_key_state_buffer(self, layer_id, loc, token_k):
        self.stored.append(loc.clone())
        self.token_k[loc.long()] = token_k

    def get_qsa_key_state_buffer(self, layer_id):
        return self.token_k

    def set_qsa_rope_position_buffer(self, loc, positions):
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        self.rope_positions[loc.long()] = positions.transpose(0, 1)


def _backend(num_reqs=4):
    pool = _Pool()
    req_to_token = torch.stack([torch.arange(i * CTX, (i + 1) * CTX, dtype=torch.int32) for i in range(num_reqs)])
    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        model_config=SimpleNamespace(context_len=CTX, hf_config=SimpleNamespace(indexer_compress_ratio=R)),
    )
    return QwenSparseAttnBackend(runner), pool, runner.req_to_token_pool


def _mixed_batch(pool, req_pool, prefill, tails, mode=ForwardMode.MIXED):
    """``prefill`` = [(req, prefix, extend)], ``tails`` = [(req, seq_len)] -- the running decodes appended by
    mix_with_running as 1-token extends (last rows)."""
    reqs, seqs, exts, pos = [], [], [], []
    for r, p, e in prefill:
        reqs.append(r); seqs.append(p + e); exts.append(e); pos += list(range(p, p + e))
    for r, n in tails:
        reqs.append(r); seqs.append(n); exts.append(1); pos.append(n - 1)
    ntok = len(pos)
    return SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=req_pool,
        req_pool_indices=torch.tensor(reqs, dtype=torch.int32),
        seq_lens=torch.tensor(seqs, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(seqs, dtype=torch.int64),
        forward_mode=mode,
        extend_seq_lens=torch.tensor(exts, dtype=torch.int32),
        extend_seq_lens_cpu=list(exts),
        extend_prefix_lens=torch.tensor([s - e for s, e in zip(seqs, exts)], dtype=torch.int32),
        extend_prefix_lens_cpu=[s - e for s, e in zip(seqs, exts)],
        positions=torch.tensor(pos, dtype=torch.int64),
        mrope_positions=None,
        input_ids=torch.zeros(ntok, dtype=torch.int32),
        out_cache_loc=torch.arange(ntok, dtype=torch.int32),
        rids=[f"pdflip-{i}" for i in range(len(reqs))],
        _original_forward_mode=None,
    )


def _decode_batch(pool, req_pool, req, n):
    return SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=req_pool,
        req_pool_indices=torch.tensor([req], dtype=torch.int32),
        seq_lens=torch.tensor([n], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([n], dtype=torch.int64),
        positions=torch.tensor([n - 1], dtype=torch.int64),
        out_cache_loc=torch.arange(1, dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        _original_forward_mode=None,
    )


def _meta(backend, fb):
    backend.init_forward_metadata(fb)
    return backend.forward_metadata.indexer_metadata


def test_repro_a_mixed_tail_off_the_grain_builds_its_metadata():
    """RED before S37: QSA-PREFIX-GRAIN on the tail (prefix 142791 % 4 == 3 on the metal; here 10 % 4 == 2)."""
    backend, pool, rp = _backend()
    fb = _mixed_batch(pool, rp, prefill=[(1, 8, 4)], tails=[(2, 11)])
    im = _meta(backend, fb)
    assert im.write_locs is not None


@pytest.mark.parametrize("n", [12, 11, 9, 16])
def test_the_tail_plan_is_the_decode_plan_of_the_same_request(n):
    backend, pool, rp = _backend()
    im = _meta(backend, _mixed_batch(pool, rp, prefill=[(1, 8, 4)], tails=[(2, n)]))
    backend2, pool2, rp2 = _backend()
    dm = _meta(backend2, _decode_batch(pool2, rp2, 2, n))
    closes = n % R == 0
    d_locs = dm.write_locs
    d_ring = dm.compress_group_ring_locs
    if closes:
        assert im.tail_write_locs is not None and im.tail_write_locs.numel() >= 1
        # padded plan: the tail row's entry is the valid one (rows == the tail's batch row)
        assert int(im.tail_write_locs[0]) == int(d_locs[0])
        assert torch.equal(im.tail_group_ring_locs[0], d_ring[0])
        assert im.tail_group_ring_locs[0].tolist() == [2 * R + (p % R) for p in range(n - R, n)]
    else:
        # a tail that closes no group writes only the inert padding slot 0 -- exactly as its decode step
        assert im.tail_write_locs is not None and set(im.tail_write_locs.tolist()) == {0}
        assert set(d_locs.tolist()) <= {0}


def test_the_tail_token_goes_to_its_ring_slot_even_when_it_closes_the_group():
    backend, pool, rp = _backend()
    fb = _mixed_batch(pool, rp, prefill=[(1, 8, 4)], tails=[(2, 12), (3, 10)])
    im = _meta(backend, fb)
    slots = im.pending_ring_slots.tolist()
    # tokens: 4 prefill (positions 8..11 of req 1: 12 % 4 == 0 -> no pending tail, dumped), then the two tails
    assert slots[4] == 2 * R + (11 % R), "tail closing its group: stored like a decode token, not dumped"
    assert slots[5] == 3 * R + (9 % R)
    assert slots[:4] == [p % R for p in range(8, 12)], "the prefill rows' dump rule is unchanged"


def test_the_prefill_rows_extend_plan_is_unchanged_by_tails():
    backend, pool, rp = _backend()
    im = _meta(backend, _mixed_batch(pool, rp, prefill=[(1, 8, 8)], tails=[(2, 12)]))
    backend2, pool2, rp2 = _backend()
    pm = _meta(backend2, _mixed_batch(pool2, rp2, prefill=[(1, 8, 8)], tails=[], mode=ForwardMode.EXTEND))
    k = int(pm.write_locs.numel())
    valid = int((pm.compress_member_rows >= 0).sum()) if pm.compress_member_rows is not None else 0
    assert valid >= 2
    # the extend groups (blocks 2, 3 of req 1) with the same slots and the same packed member rows
    assert im.write_locs[:2].tolist() == pm.write_locs[:2].tolist()
    assert im.compress_member_rows[:2].tolist() == pm.compress_member_rows[:2].tolist()
    assert k >= 2
    # and NO extend entry for the tail row (batch row 1): its members are not in this forward's packed rows
    tail_entries = ((im.compress_sequence_ids == 1) & (im.write_locs != 0)).nonzero().flatten().tolist()
    assert tail_entries == [], f"the tail leaked into the extend plan at entries {tail_entries}"


def test_a_pure_extend_with_a_split_prefix_still_stops_by_name():
    backend, pool, rp = _backend()
    fb = _mixed_batch(pool, rp, prefill=[(1, 10, 4)], tails=[], mode=ForwardMode.EXTEND)
    with pytest.raises(RuntimeError, match="QSA-PREFIX-GRAIN rid=pdflip-0 prefix=10"):
        _meta(backend, fb)


def test_a_mixed_prefill_row_with_a_split_prefix_still_stops_by_name():
    """Only the 1-token rows are decode steps; a longer prefill row off the grain is still K11's named stop."""
    backend, pool, rp = _backend()
    fb = _mixed_batch(pool, rp, prefill=[(1, 10, 4)], tails=[(2, 11)])
    with pytest.raises(RuntimeError, match="QSA-PREFIX-GRAIN rid=pdflip-0 prefix=10"):   # the HOST stop, with rid
        _meta(backend, fb)


def test_the_indexer_compresses_extend_groups_then_tail_groups_from_the_ring(monkeypatch):
    backend, pool, rp = _backend()
    im = _meta(backend, _mixed_batch(pool, rp, prefill=[(1, 8, 8)], tails=[(2, 12)]))
    calls = []
    ix = object.__new__(QSAIndexer)
    ix.layer_id = 0
    ix.compress_ratio = R
    monkeypatch.setattr(QSAIndexer, "_use_fused_compress", lambda self, pool: True)

    def rec(self, pool_, group_locs, compressed_locs, source_keys=None, source_rope=None):
        src = "ring" if source_keys is pool_.token_k else "packed"
        calls.append((src, group_locs.clone(), compressed_locs.clone()))

    monkeypatch.setattr(QSAIndexer, "_fused_compress_store", rec)
    ntok = 9
    token_k = torch.randn(ntok, 1, 8)
    positions = torch.tensor(list(range(8, 16)) + [11], dtype=torch.int64)
    ix.update_key_state_and_compress(token_k, positions, positions, im)
    assert [c[0] for c in calls] == ["packed", "ring"], calls
    assert calls[1][1][0].tolist() == [2 * R + (p % R) for p in range(8, 12)]
    assert int(calls[1][2][0]) == int(im.tail_write_locs[0])
    # the tail's own token is in the ring before the ring compress (the store comes first)
    assert torch.equal(pool.token_k[2 * R + 3], token_k[8])
