# SPDX-License-Identifier: Apache-2.0
"""#1243 SLICE 2q -- the precision tail on Next Flash's QSA rows reader.

Next Flash (Qwen4-Exp) reads its full-attention KV one pool ROW per selected
token (``qwen_sparse_attn_backend._attend_rows`` -> ``_sparse_attn_rows_fwd``),
on a paged pool (page a multiple of the QSA compress ratio). The tail therefore
has no plan to split: the rows kernel looks every row up in the ring's mapping
and reads the bf16 ring row where one exists. What is pinned here:

* the fused kernel (Triton INTERPRETER, in a subprocess so the JIT is built in
  interpreter mode no matter what the rest of the session imported) reads
  exactly the mapped lanes from the ring, everything else from the fp8 pool,
  counts each tail lane once, and is bit-identical to the tail-less kernel on
  an empty mapping;
* the sliding window: every written token at request index p ages out the
  token at p - N; age-out before precommit; an extend chunk longer than N keeps
  its head body-only; the padding slot never gets a ring row;
* the paged free path releases whole pages;
* a finished request's rows are released before the tree takes its slots;
* the W141 wiring gate on the rows reader, and the device-side READ fact
  (a graph replay counts);
* the backend: token indices from lengths (decode / verify / extend, a padded
  replay view), the ring written BEFORE the body, trtllm bypassed with the tail
  on, the CPU reference reading tail rows;
* the ring cell charged without the QSA index bytes.

Hermetic: CUDA_VISIBLE_DEVICES="" and CPU tensors throughout.
"""

import json
import os
import subprocess
import sys
import types
import unittest

import torch

from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.kv_tail import (
    KV_TAIL_NULL,
    KvTailKnobs,
    KvTailRing,
    Weg2KvTailFormRefused,
    Weg2KvTailNoOp,
    install_kv_tail_ring,
    release_request_ring_rows,
)
from sglang.test.test_utils import CustomTestCase

DEV = "cpu"
LAYERS = 3
HEADS = 2
HEAD_DIM = 8
PAGE = 4  # QSA: a multiple of the compress ratio


def _body_pool(rows=64, page=PAGE, dtype=torch.float8_e4m3fn, heads=HEADS):
    return memory_pool.MHATokenToKVPool(
        size=rows,
        page_size=page,
        dtype=dtype,
        head_num=heads,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
        device=DEV,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


def _rows_ring(n=4, ring_rows=32, body_rows=64, page=PAGE, max_tokens="fixed"):
    """``max_tokens="fixed"`` pins max == min: the FIXED sliding window the
    mechanics tests pin; the elastic tests pass -1 (open) or a ceiling."""
    mx = n if max_tokens == "fixed" else max_tokens
    return KvTailRing(
        _body_pool(body_rows, page),
        KvTailKnobs(min_tokens=n, max_tokens=mx),
        ring_rows=ring_rows,
        reader="rows",
    )


class _Layer:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.k_scale = None
        self.v_scale = None
        self.scaling = 0.5


class _TorchMaskedWrite:
    """``masked_set_kv_buffer_kernel`` in torch, so the ring's write runs on a
    CPU desk with its real data flow (the Triton kernel needs a device)."""

    def __getitem__(self, grid):
        def run(cache_k, cache_v, k_buf, v_buf, loc, mask, *args, **kwargs):
            m = mask.to(torch.bool)
            idx = loc.to(torch.int64)[m]
            k_buf[idx] = cache_k[m].to(k_buf.dtype)
            v_buf[idx] = cache_v[m].to(v_buf.dtype)

        return run


def _mapped(ring):
    return set(torch.nonzero(ring.mapping >= 0).flatten().tolist())


def _step(ring, r2t, req_rows, written, site="decode", expect_read=True, forward=True):
    """One plan step: ``written`` = [(batch_idx, token_index), ...]. With
    ``forward`` the step's attention is simulated as the rows kernel would
    run it: one tail-enabled launch per read step (the device counter the
    backend's ``_kv_tail_operands`` bumps)."""
    tok_req = torch.tensor([b for b, _ in written], dtype=torch.int64)
    tok_index = torch.tensor([p for _, p in written], dtype=torch.int64)
    rpi = torch.tensor(req_rows, dtype=torch.int64)
    loc = r2t[rpi[tok_req], tok_index].to(torch.int64)
    fresh = ring.plan_rows_step(site, loc, tok_req, tok_index, rpi, r2t, expect_read)
    if forward and expect_read:
        ring.note_rows_launch()
    return fresh


# ---------------------------------------------------------------------------
# The fused kernel, in the Triton interpreter.
# ---------------------------------------------------------------------------

_KERNEL_PROBE = r"""
import json, torch
torch.manual_seed(0)
import sglang.srt.layers.attention.qsa.sparse_attn as sa
sa._get_best_config = lambda total_q: (16, 4, 1)   # no device to name on a desk
Tq, Hq, Hkv, D, N, K, R = 5, 4, 2, 32, 60, 16, 30
q = torch.randn(Tq, Hq, D)
kp = torch.randn(N, Hkv, D).to(torch.float8_e4m3fn)
vp = torch.randn(N, Hkv, D).to(torch.float8_e4m3fn)
ring_k = torch.randn(R, Hkv, D); ring_v = torch.randn(R, Hkv, D)
mapping = torch.full((N + 1,), -1, dtype=torch.int32)
tail_slots = torch.tensor([3, 7, 11, 20, 41, 42, 43])
mapping[tail_slots] = torch.arange(tail_slots.numel(), dtype=torch.int32) + 5
rows = torch.randint(1, N, (Tq, K), dtype=torch.int32)
rows[0, :4] = torch.tensor([3, 7, 42, 11], dtype=torch.int32)
rows[1, 6:] = -1
rows[2, :] = -1
res = {}
def ref(rows):
    kf, vf = kp.float(), vp.float()
    out = torch.zeros(Tq, Hq, D); g = Hq // Hkv
    for i in range(Tq):
        r = rows[i][rows[i] >= 0].long()
        if r.numel() == 0:
            continue
        kk = torch.stack([ring_k[mapping[s]] if mapping[s] >= 0 else kf[s] for s in r.tolist()])
        vv = torch.stack([ring_v[mapping[s]] if mapping[s] >= 0 else vf[s] for s in r.tolist()])
        for h in range(Hq):
            s = (q[i, h] @ kk[:, h // g].T) * 0.125
            out[i, h] = torch.softmax(s, 0) @ vv[:, h // g]
    return out
read = torch.zeros((), dtype=torch.int64)
out, lse = sa.sparse_attn_rows_triton(q, kp, vp, rows, 0.125, tail=(ring_k, ring_v, mapping, read))
res["max_err"] = float((out - ref(rows)).abs().max())
res["read"] = int(read)
res["expect_read"] = int(((rows >= 0) & (mapping[rows.clamp(min=0).long()] >= 0)).sum())
res["empty_row_lse_inf"] = bool(torch.isinf(lse[2]).all())
# compacted rows (the DCP / fused-resolve form): holes inside the counted range
rows_c, counts = sa.compact_owned_rows(rows)
read_c = torch.zeros((), dtype=torch.int64)
out_c, _ = sa.sparse_attn_rows_triton(q, kp, vp, rows_c, 0.125, row_counts=counts, tail=(ring_k, ring_v, mapping, read_c))
res["compact_err"] = float((out_c - ref(rows)).abs().max())
res["compact_read"] = int(read_c)
# empty mapping == the tail-less kernel, bit for bit
out0, lse0 = sa.sparse_attn_rows_triton(q, kp, vp, rows, 0.125)
read0 = torch.zeros((), dtype=torch.int64)
m0 = torch.full_like(mapping, -1)
out1, lse1 = sa.sparse_attn_rows_triton(q, kp, vp, rows, 0.125, tail=(ring_k, ring_v, m0, read0))
res["empty_equal_out"] = bool(torch.equal(out0, out1))
res["empty_equal_lse"] = bool(torch.equal(lse0.nan_to_num(), lse1.nan_to_num()))
res["empty_read"] = int(read0)
# a lane mapped to the ring but NOT selected is never read
read_ns = torch.zeros((), dtype=torch.int64)
rows_ns = rows.clone(); rows_ns[rows_ns == 20] = 21
sa.sparse_attn_rows_triton(q, kp, vp, rows_ns, 0.125, tail=(ring_k, ring_v, mapping, read_ns))
res["unselected_read"] = int(read_ns)
res["unselected_expect"] = int(((rows_ns >= 0) & (mapping[rows_ns.clamp(min=0).long()] >= 0)).sum())
print("PROBE " + json.dumps(res))
"""


def _run_kernel_probe():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TRITON_INTERPRET"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _KERNEL_PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[len("PROBE ") :])
    raise AssertionError(
        f"kernel probe produced no result (rc={proc.returncode}):\n{proc.stderr[-3000:]}"
    )


class TestTheFusedRowsKernelReadsTheRing(CustomTestCase):
    """The READ half: one kernel, each selected lane from exactly one source."""

    @classmethod
    def setUpClass(cls):
        cls.res = _run_kernel_probe()

    def test_mapped_lanes_read_bf16_and_the_rest_read_the_pool(self):
        self.assertLess(self.res["max_err"], 1e-5, self.res)

    def test_the_read_fact_counts_each_tail_lane_once_across_head_groups(self):
        # Hkv = 2 programs per query read the same rows; group 0 counts.
        self.assertEqual(self.res["read"], self.res["expect_read"])
        self.assertGreater(self.res["read"], 0)

    def test_a_query_with_no_rows_still_reads_nothing(self):
        self.assertTrue(self.res["empty_row_lse_inf"])

    def test_compacted_rows_with_holes_read_the_same(self):
        self.assertLess(self.res["compact_err"], 1e-5, self.res)
        self.assertEqual(self.res["compact_read"], self.res["expect_read"])

    def test_an_empty_mapping_is_the_tailless_kernel_bit_for_bit(self):
        self.assertTrue(self.res["empty_equal_out"])
        self.assertTrue(self.res["empty_equal_lse"])
        self.assertEqual(self.res["empty_read"], 0)

    def test_a_ring_row_the_indexer_did_not_select_is_not_read(self):
        self.assertEqual(self.res["unselected_read"], self.res["unselected_expect"])
        self.assertLess(self.res["unselected_read"], self.res["read"])


# ---------------------------------------------------------------------------
# The ring on the rows reader: form, window, frees, finish, W141.
# ---------------------------------------------------------------------------


class TestTheRowsReaderForm(CustomTestCase):
    def test_a_paged_body_is_legal_for_the_rows_reader(self):
        ring = _rows_ring(page=64, body_rows=256)
        self.assertEqual(ring.reader, "rows")
        self.assertEqual(ring.page_size, 64)
        self.assertEqual(ring.mapping.numel(), 256 + 64 + 1)

    def test_the_paged_reader_still_refuses_a_paged_body(self):
        with self.assertRaises(Weg2KvTailFormRefused) as cm:
            KvTailRing(_body_pool(64, 4), KvTailKnobs(min_tokens=4), ring_rows=16, reader="paged")
        self.assertIn("--page-size 1", str(cm.exception))

    def test_an_unknown_reader_is_refused_by_name(self):
        with self.assertRaises(Weg2KvTailFormRefused):
            KvTailRing(_body_pool(64, 1), KvTailKnobs(min_tokens=4), ring_rows=16, reader="fa3")

    def test_a_form_a_expert_worker_without_kv_gets_no_ring(self):
        pool = types.SimpleNamespace(page_size=64, head_num=0, size=128)
        self.assertIsNone(
            install_kv_tail_ring(
                pool, KvTailKnobs(min_tokens=16), max_running_requests=1,
                owned_share_num=1, owned_share_den=1, reader="rows",
            )
        )
        self.assertFalse(hasattr(pool, "kv_tail"))


class TestTheSlidingWindow(CustomTestCase):
    """Every written token at request index p pushes p - N out."""

    def _r2t(self, rows=2, width=40, base=4):
        # Request r's token j lives in slot base + r*width + j (page 0 reserved).
        return (base + torch.arange(rows * width).reshape(rows, width)).to(torch.int32)

    def test_decode_holds_exactly_the_newest_n_tokens(self):
        ring, r2t = _rows_ring(n=4), self._r2t()
        for p in range(10):
            _step(ring, r2t, [0], [(0, p)])
            L = p + 1
            want = {int(r2t[0, j]) for j in range(max(0, L - 4), L)}
            self.assertEqual(_mapped(ring), want, f"after token {p}")
            self.assertEqual(ring.rows_held, min(L, 4))
        self.assertEqual(ring.counters.demoted_by_age, 6)
        self.assertEqual(ring.counters.claimed_total, 10)

    def test_an_extend_chunk_longer_than_the_window_keeps_its_head_body_only(self):
        ring, r2t = _rows_ring(n=4), self._r2t()
        fresh = _step(ring, r2t, [0], [(0, p) for p in range(10)], site="extend", expect_read=False)
        self.assertEqual(fresh, 4)
        self.assertEqual(_mapped(ring), {int(r2t[0, j]) for j in range(6, 10)})
        # the next chunk slides the window over the chunk boundary
        _step(ring, r2t, [0], [(0, p) for p in range(10, 13)], site="extend")
        self.assertEqual(_mapped(ring), {int(r2t[0, j]) for j in range(9, 13)})
        self.assertEqual(ring.counters.extend_steps, 2)

    def test_a_long_prefill_ends_with_its_newest_prompt_tokens_in_bf16(self):
        """Slice 2d on this reader: chunked prefill, then decode."""
        ring, r2t = _rows_ring(n=6), self._r2t()
        for start in range(0, 30, 8):
            _step(ring, r2t, [0], [(0, p) for p in range(start, min(start + 8, 30))], site="extend")
        self.assertEqual(_mapped(ring), {int(r2t[0, j]) for j in range(24, 30)})
        _step(ring, r2t, [0], [(0, 30)])
        self.assertEqual(_mapped(ring), {int(r2t[0, j]) for j in range(25, 31)})

    def test_a_verify_chain_slides_by_its_draft_width(self):
        ring, r2t = _rows_ring(n=4), self._r2t()
        _step(ring, r2t, [0], [(0, p) for p in range(8)], site="extend", expect_read=False)
        _step(ring, r2t, [0], [(0, 8 + i) for i in range(3)], site="verify")
        self.assertEqual(_mapped(ring), {int(r2t[0, j]) for j in range(7, 11)})
        self.assertEqual(ring.counters.verify_steps, 1)

    def test_two_requests_keep_two_independent_windows(self):
        ring, r2t = _rows_ring(n=3), self._r2t()
        _step(ring, r2t, [0, 1], [(0, p) for p in range(5)] + [(1, p) for p in range(2)], site="extend",
              expect_read=False)
        self.assertEqual(
            _mapped(ring),
            {int(r2t[0, j]) for j in range(2, 5)} | {int(r2t[1, j]) for j in range(0, 2)},
        )
        _step(ring, r2t, [0, 1], [(0, 5), (1, 2)])
        self.assertEqual(
            _mapped(ring),
            {int(r2t[0, j]) for j in range(3, 6)} | {int(r2t[1, j]) for j in range(0, 3)},
        )

    def test_the_padding_slot_never_gets_a_ring_row(self):
        ring = _rows_ring(n=4)
        r2t = torch.zeros((1, 8), dtype=torch.int32)  # every token -> slot 0
        _step(ring, r2t, [0], [(0, 0), (0, 1)], site="extend", expect_read=False)
        self.assertEqual(int(ring.mapping[0]), KV_TAIL_NULL)
        self.assertEqual(ring.rows_held, 0)

    def test_a_short_ring_clamps_instead_of_failing(self):
        ring, r2t = _rows_ring(n=8, ring_rows=8), self._r2t()
        _step(ring, r2t, [0, 1], [(0, p) for p in range(6)] + [(1, p) for p in range(6)], site="extend",
              expect_read=False)
        self.assertEqual(ring.rows_held, 8)
        self.assertEqual(ring.counters.clamped_alloc, 4)
        # partial, newest first: request 1's six tokens and request 0's last two
        self.assertEqual(
            _mapped(ring), {int(r2t[0, j]) for j in (4, 5)} | {int(r2t[1, j]) for j in range(6)}
        )


class TestTheElasticWindow(CustomTestCase):
    """Slice 2e (basis 2.2 / 7.1 / 7.7 / 7.8): above the guaranteed minimum the
    16-bit portion uses whatever the ring has free; under pressure the largest
    holdings shrink from the back, water-levelled, never below the minimum."""

    def _r2t(self, rows=3, width=64, base=4):
        return (base + torch.arange(rows * width).reshape(rows, width)).to(torch.int32)

    def _held(self, ring, r2t, row):
        return len(_mapped(ring) & {int(x) for x in r2t[row].tolist()})

    def test_one_request_grows_into_the_whole_ring_then_slides(self):
        ring, r2t = _rows_ring(n=4, ring_rows=16, body_rows=256, max_tokens=-1), self._r2t()
        for p in range(20):
            _step(ring, r2t, [0], [(0, p)])
        self.assertEqual(ring.rows_held, 16)
        self.assertEqual(_mapped(ring), {int(r2t[0, j]) for j in range(4, 20)})
        self.assertEqual(ring.counters.clamped_alloc, 0)
        self.assertEqual(ring.counters.demoted_by_pressure, 4)

    def test_the_ceiling_bounds_the_growth(self):
        ring, r2t = _rows_ring(n=4, ring_rows=32, body_rows=256, max_tokens=6), self._r2t()
        for p in range(20):
            _step(ring, r2t, [0], [(0, p)])
        self.assertEqual(_mapped(ring), {int(r2t[0, j]) for j in range(14, 20)})
        self.assertEqual(ring.counters.demoted_by_pressure, 0)

    def test_a_newcomer_takes_its_rows_from_the_largest_holding(self):
        ring, r2t = _rows_ring(n=4, ring_rows=16, body_rows=256, max_tokens=-1), self._r2t()
        _step(ring, r2t, [0], [(0, p) for p in range(12)], site="extend", expect_read=False)
        self.assertEqual(self._held(ring, r2t, 0), 12)
        _step(ring, r2t, [1], [(0, p) for p in range(6)], site="extend", expect_read=False)
        self.assertEqual(self._held(ring, r2t, 1), 6)
        self.assertEqual(self._held(ring, r2t, 0), 10)  # shrunk from the back
        self.assertEqual(_mapped(ring) & {int(x) for x in r2t[0].tolist()},
                         {int(r2t[0, j]) for j in range(2, 12)})
        # both decode: water-levelling converges to equal holdings
        for step in range(8):
            _step(ring, r2t, [0, 1], [(0, 12 + step), (1, 6 + step)])
        a, b = self._held(ring, r2t, 0), self._held(ring, r2t, 1)
        self.assertEqual(a + b, 16)
        self.assertLessEqual(abs(a - b), 1)
        self.assertEqual(ring.counters.clamped_alloc, 0)

    def test_requests_at_the_minimum_keep_it_by_sliding(self):
        ring, r2t = _rows_ring(n=4, ring_rows=8, body_rows=256, max_tokens=-1), self._r2t()
        _step(ring, r2t, [0, 1], [(0, p) for p in range(4)] + [(1, p) for p in range(4)],
              site="extend", expect_read=False)
        for step in range(5):
            _step(ring, r2t, [0, 1], [(0, 4 + step), (1, 4 + step)])
        self.assertEqual(_mapped(ring),
                         {int(r2t[0, j]) for j in range(5, 9)} | {int(r2t[1, j]) for j in range(5, 9)})
        self.assertEqual(ring.counters.clamped_alloc, 0)
        self.assertEqual(ring.counters.demoted_by_pressure, 0)  # sliding, not a cut

    def test_the_minimum_is_never_cut_for_a_newcomer(self):
        ring, r2t = _rows_ring(n=4, ring_rows=8, body_rows=256, max_tokens=-1), self._r2t()
        _step(ring, r2t, [0, 1], [(0, p) for p in range(4)] + [(1, p) for p in range(4)],
              site="extend", expect_read=False)
        _step(ring, r2t, [2], [(0, 0), (0, 1)], site="extend", expect_read=False)
        self.assertEqual(self._held(ring, r2t, 0), 4)
        self.assertEqual(self._held(ring, r2t, 1), 4)
        self.assertEqual(self._held(ring, r2t, 2), 0)  # clamped, counted
        self.assertEqual(ring.counters.clamped_alloc, 2)

    def test_a_holding_above_the_minimum_is_cut_only_down_to_it(self):
        """The partial case: one holding above the minimum cannot cover the
        newcomer. It is cut to the minimum and NOT below; the newcomer's rest
        is clamped (and counted)."""
        ring, r2t = _rows_ring(n=4, ring_rows=10, body_rows=256, max_tokens=-1), self._r2t()
        _step(ring, r2t, [0, 1], [(0, p) for p in range(6)] + [(1, p) for p in range(4)],
              site="extend", expect_read=False)
        self.assertEqual((self._held(ring, r2t, 0), self._held(ring, r2t, 1)), (6, 4))
        _step(ring, r2t, [2], [(0, p) for p in range(4)], site="extend", expect_read=False)
        self.assertEqual(self._held(ring, r2t, 0), 4)
        self.assertEqual(self._held(ring, r2t, 1), 4)
        self.assertEqual(self._held(ring, r2t, 2), 2)
        self.assertEqual(ring.counters.clamped_alloc, 2)

    def test_a_finished_request_frees_its_window_for_the_next_one(self):
        ring, r2t = _rows_ring(n=4, ring_rows=8, body_rows=256, max_tokens=-1), self._r2t()
        _step(ring, r2t, [0], [(0, p) for p in range(8)], site="extend", expect_read=False)
        ring.release_request(0, r2t[0, :8])
        self.assertEqual(ring.rows_held, 0)
        self.assertNotIn(0, ring._win)
        # the row is reused by a new request: its window starts afresh
        _step(ring, r2t, [0], [(0, p) for p in range(3)], site="extend", expect_read=False)
        self.assertEqual(ring._win[0], [0, 3])

    def test_a_request_restarted_behind_its_window_starts_afresh(self):
        ring, r2t = _rows_ring(n=2, ring_rows=8, body_rows=256, max_tokens=2), self._r2t()
        _step(ring, r2t, [0], [(0, p) for p in range(6)], site="extend", expect_read=False)
        self.assertEqual(ring._win[0][0], 4)
        _step(ring, r2t, [0], [(0, 0)], site="extend", expect_read=False)  # a new occupant
        self.assertEqual(ring._win[0], [0, 1])

    def test_a_reset_forgets_every_window_and_the_wiring_expectation(self):
        ring, r2t = _rows_ring(n=4, ring_rows=8, body_rows=256, max_tokens=-1), self._r2t()
        _step(ring, r2t, [0], [(0, 0)], forward=False)
        ring.reset()
        self.assertEqual(ring._win, {})
        _step(ring, r2t, [0], [(0, 1)])  # no W141: the flush took the expectation along


class TestThePagedFreePath(CustomTestCase):
    def test_a_page_free_releases_every_token_of_the_page(self):
        ring = _rows_ring(n=8, page=4)
        r2t = torch.arange(4, 20, dtype=torch.int32).reshape(1, 16)
        _step(ring, r2t, [0], [(0, p) for p in range(6)], site="extend", expect_read=False)
        self.assertEqual(_mapped(ring), set(range(4, 10)))
        # the allocator frees page 1 (slots 4..7) with ONE listed token
        ring._on_body_free(torch.tensor([5]))
        self.assertEqual(_mapped(ring), {8, 9})
        self.assertEqual(ring.counters.demoted_by_free, 4)


class TestAFinishedRequestReleasesItsRows(CustomTestCase):
    def _tree(self, ring, r2t):
        pool = types.SimpleNamespace(kv_tail=ring)
        return types.SimpleNamespace(
            token_to_kv_pool_allocator=types.SimpleNamespace(get_kvcache=lambda: pool),
            req_to_token_pool=types.SimpleNamespace(req_to_token=r2t),
        )

    def test_the_rows_are_released_before_the_tree_takes_the_slots(self):
        ring = _rows_ring(n=8)
        r2t = torch.arange(4, 36, dtype=torch.int32).reshape(2, 16)
        _step(ring, r2t, [0, 1], [(0, p) for p in range(5)] + [(1, p) for p in range(3)], site="extend",
              expect_read=False)
        req = types.SimpleNamespace(req_pool_idx=0, kv_allocated_len=5)
        n = release_request_ring_rows(req, self._tree(ring, r2t))
        self.assertEqual(n, 5)
        self.assertEqual(_mapped(ring), {int(r2t[1, j]) for j in range(3)})
        self.assertEqual(ring.counters.demoted_by_finish, 5)

    def test_no_ring_no_work(self):
        tree = types.SimpleNamespace(
            token_to_kv_pool_allocator=types.SimpleNamespace(get_kvcache=lambda: object())
        )
        self.assertEqual(
            release_request_ring_rows(types.SimpleNamespace(req_pool_idx=0, kv_allocated_len=3), tree), 0
        )

    def test_release_kv_cache_releases_before_cache_finished_req(self):
        from sglang.srt.mem_cache import common

        ring = _rows_ring(n=8)
        r2t = torch.arange(4, 20, dtype=torch.int32).reshape(1, 16)
        _step(ring, r2t, [0], [(0, p) for p in range(4)], site="extend", expect_read=False)
        seen = {}
        tree = self._tree(ring, r2t)

        def cache_finished_req(req, is_insert=True, **kw):
            seen["held_at_insert"] = ring.rows_held
            req.req_pool_idx = None  # the streaming-session exit: nothing after

        tree.cache_finished_req = cache_finished_req
        tree.supports_mamba = lambda: False
        req = types.SimpleNamespace(
            req_pool_idx=0, kv_allocated_len=4, kv_spill_state=None, skip_radix_cache_insert=False
        )
        common.release_kv_cache(req, tree)
        self.assertEqual(seen["held_at_insert"], 0)


class TestTheRowsWiringGate(CustomTestCase):
    """W141 on the rows reader, and the READ fact from the device."""

    def _warm(self):
        ring = _rows_ring(n=4)
        r2t = torch.arange(4, 36, dtype=torch.int32).reshape(1, 32)
        _step(ring, r2t, [0], [(0, p) for p in range(6)], site="extend", expect_read=False)
        return ring, r2t

    def test_a_read_step_with_rows_held_and_no_launch_is_w56(self):
        ring, r2t = self._warm()
        _step(ring, r2t, [0], [(0, 6)], forward=False)  # decode: the kernel MUST read
        with self.assertRaises(Weg2KvTailNoOp) as cm:
            _step(ring, r2t, [0], [(0, 7)])
        self.assertIn("W141", str(cm.exception))

    def test_a_launch_that_selected_no_tail_lane_is_not_w56(self):
        ring, r2t = self._warm()
        _step(ring, r2t, [0], [(0, 6)])  # the kernel ran; it just selected no tail lane
        _step(ring, r2t, [0], [(0, 7)])
        self.assertEqual(ring.counters.rows_launches, 1)
        self.assertEqual(ring.counters.tail_rows_read_step, 0)

    def test_an_extend_without_prefix_expects_no_read(self):
        ring, r2t = self._warm()
        _step(ring, r2t, [0], [(0, 6), (0, 7)], site="extend", expect_read=False)
        _step(ring, r2t, [0], [(0, 8)], site="extend", expect_read=False)  # no raise

    def test_a_graph_replay_is_counted_on_the_device(self):
        ring, r2t = self._warm()
        _step(ring, r2t, [0], [(0, 6)], forward=False)
        # what a replayed graph does: device adds, no Python
        ring._merge_dev += 3
        ring._read_dev += 11
        _step(ring, r2t, [0], [(0, 7)])
        self.assertEqual(ring.counters.rows_launches, 3)
        self.assertEqual(ring.counters.tail_rows_read, 11)
        self.assertEqual(ring.counters.attended_rows, 11)

    def test_the_counter_line_names_the_reader_and_its_read_fact(self):
        ring, r2t = self._warm()
        line = ring.counter_line("extend")
        for field in ("reader=rows", "tail_rows_read=", "rows_launches=", "demoted_by_finish=",
                      "suppressed=", "instrument=kernel-read-counts", "window=4"):
            self.assertIn(field, line)


# ---------------------------------------------------------------------------
# The backend.
# ---------------------------------------------------------------------------


def _qsa_backend(ring_on=True, rows=64, page=PAGE, n=4, r2t=None):
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend

    b = QwenSparseAttnBackend(runner=None)
    pool = _body_pool(rows, page)
    if ring_on:
        install_kv_tail_ring(
            pool, KvTailKnobs(min_tokens=n, max_tokens=n), max_running_requests=2,
            owned_share_num=1, owned_share_den=1, reader="rows",
        )
    b.token_to_kv_pool = pool
    b.req_to_token = r2t if r2t is not None else torch.arange(4, 4 + 2 * 24, dtype=torch.int32).reshape(2, 24)
    return b


class _Mode:
    def __init__(self, name):
        self.name = name

    def is_target_verify(self):
        return self.name == "verify"

    def is_decode(self):
        return self.name == "decode"

    def is_split_prefill(self):
        return False

    def is_idle(self):
        return self.name == "idle"

    def __eq__(self, other):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        return self.name == "extend" and other is ForwardMode.EXTEND


class TestTheBackendPlan(CustomTestCase):
    def test_decode_verify_and_extend_index_the_request_by_length_not_rope(self):
        b = _qsa_backend(n=4)
        r2t = b.req_to_token
        ring = b.token_to_kv_pool.kv_tail
        # extend: request 0 prefix 0 len 6, request 1 prefix 0 len 3
        fb = types.SimpleNamespace(
            forward_mode=_Mode("extend"),
            out_cache_loc=torch.cat([r2t[0, :6], r2t[1, :3]]).to(torch.int64),
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([6, 3]),
            extend_seq_lens=torch.tensor([6, 3]),
            extend_seq_lens_cpu=[6, 3],
            extend_prefix_lens_cpu=[0, 0],
            positions=torch.full((3, 9), 999),  # M-RoPE: must NOT be read
            spec_info=None,
        )
        b._kv_tail_plan(fb)
        self.assertEqual(
            _mapped(ring), {int(r2t[0, j]) for j in range(2, 6)} | {int(r2t[1, j]) for j in range(3)}
        )
        # verify: d = 2 at seq_lens + [0, 2)
        fb_v = types.SimpleNamespace(
            forward_mode=_Mode("verify"),
            out_cache_loc=torch.stack([r2t[0, 6:8], r2t[1, 3:5]]).reshape(-1).to(torch.int64),
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([6, 3]),
            spec_info=types.SimpleNamespace(draft_token_num=2),
        )
        ring.note_rows_launch()  # the extend's prefix-free step read nothing; keep the gate honest
        b._kv_tail_plan(fb_v)
        self.assertEqual(
            _mapped(ring), {int(r2t[0, j]) for j in range(4, 8)} | {int(r2t[1, j]) for j in range(1, 5)}
        )

    def test_a_padded_replay_view_plans_the_real_requests_only(self):
        b = _qsa_backend(n=4)
        r2t = b.req_to_token
        ring = b.token_to_kv_pool.kv_tail
        fb = types.SimpleNamespace(
            forward_mode=_Mode("verify"),
            out_cache_loc=r2t[0, 0:3].to(torch.int64),  # real tokens only
            req_pool_indices=torch.tensor([0, 1]),  # padded to bs 2
            seq_lens=torch.tensor([0, 1]),
            num_padding=1,
            spec_info=types.SimpleNamespace(draft_token_num=3),
        )
        b._kv_tail_plan(fb)
        self.assertEqual(_mapped(ring), {int(r2t[0, j]) for j in range(3)})

    def test_a_capture_hands_out_no_rows(self):
        b = _qsa_backend()
        ring = b.token_to_kv_pool.kv_tail
        fb = types.SimpleNamespace(forward_mode=_Mode("decode"))
        b._kv_tail_plan(fb, in_capture=True)
        self.assertTrue(ring._precommitted)
        self.assertEqual(ring.rows_held, 0)

    def test_idle_and_draft_forwards_disarm(self):
        b = _qsa_backend()
        ring = b.token_to_kv_pool.kv_tail
        ring.begin_step("decode")
        b._kv_tail_plan(types.SimpleNamespace(forward_mode=_Mode("idle")))
        self.assertFalse(ring._armed)

    def test_the_tail_off_backend_plans_nothing(self):
        b = _qsa_backend(ring_on=False)
        b._kv_tail_plan(types.SimpleNamespace(forward_mode=_Mode("decode")))  # no attribute touched


class TestTheBackendWriteAndRead(CustomTestCase):
    def setUp(self):
        self._real = memory_pool.masked_set_kv_buffer_kernel
        memory_pool.masked_set_kv_buffer_kernel = _TorchMaskedWrite()

    def tearDown(self):
        memory_pool.masked_set_kv_buffer_kernel = self._real

    def test_the_ring_is_written_in_bf16_before_the_body(self):
        b = _qsa_backend(n=4)
        ring = b.token_to_kv_pool.kv_tail
        r2t = b.req_to_token
        loc = r2t[0, :3].to(torch.int64)
        fb = types.SimpleNamespace(
            forward_mode=_Mode("extend"), out_cache_loc=loc, req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([3]), extend_seq_lens=torch.tensor([3]), extend_seq_lens_cpu=[3],
            extend_prefix_lens_cpu=[0], spec_info=None,
        )
        b._kv_tail_plan(fb)
        order = []
        body_set = b.token_to_kv_pool.set_kv_buffer
        ring_write = ring.write

        def rec_body(*a, **kw):
            order.append("body")
            return body_set(*a, **kw)

        def rec_ring(*a, **kw):
            order.append("ring")
            return ring_write(*a, **kw)

        b.token_to_kv_pool.set_kv_buffer = rec_body
        ring.write = rec_ring
        k = torch.randn(3, HEADS, HEAD_DIM, dtype=torch.bfloat16)
        v = torch.randn(3, HEADS, HEAD_DIM, dtype=torch.bfloat16)
        b._set_kv_buffer(types.SimpleNamespace(out_cache_loc=loc), _Layer(1), k, v)
        self.assertEqual(order, ["ring", "body"])
        rows = ring.mapping[loc].to(torch.int64)
        self.assertTrue(bool((rows >= 0).all()))
        k_ring = ring.pool.get_key_buffer(1)[rows]
        self.assertEqual(k_ring.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(k_ring, k))

    def test_the_cpu_reference_reads_the_tail_rows(self):
        from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention

        b = _qsa_backend(n=4)
        ring = b.token_to_kv_pool.kv_tail
        r2t = b.req_to_token
        loc = r2t[0, :3].to(torch.int64)
        b._kv_tail_plan(types.SimpleNamespace(
            forward_mode=_Mode("extend"), out_cache_loc=loc, req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([3]), extend_seq_lens=torch.tensor([3]), extend_seq_lens_cpu=[3],
            extend_prefix_lens_cpu=[0], spec_info=None,
        ))
        k = torch.randn(3, HEADS, HEAD_DIM, dtype=torch.bfloat16)
        v = torch.randn(3, HEADS, HEAD_DIM, dtype=torch.bfloat16)
        b._set_kv_buffer(types.SimpleNamespace(out_cache_loc=loc), _Layer(1), k, v)
        q = torch.randn(1, 4, HEAD_DIM)
        slots = loc.view(1, 3)
        before = int(ring._read_dev)
        tail = b._kv_tail_operands(_Layer(1))
        out_tail = qsa_sparse_attention(
            q, b.token_to_kv_pool.get_key_buffer(1), b.token_to_kv_pool.get_value_buffer(1),
            slots, 0.5, tail=tail,
        )
        exact = qsa_sparse_attention(q, k.float(), v.float(), torch.arange(3).view(1, 3), 0.5)
        body_only = qsa_sparse_attention(
            q, b.token_to_kv_pool.get_key_buffer(1), b.token_to_kv_pool.get_value_buffer(1), slots, 0.5
        )
        self.assertTrue(torch.allclose(out_tail, exact, atol=1e-5))
        self.assertFalse(torch.allclose(body_only, exact, atol=1e-5))  # the fp8 body is not 16 bit
        self.assertEqual(int(ring._read_dev) - before, 3)
        self.assertEqual(int(ring._merge_dev), 1)  # the launch was noted


class TestTheRoutesWithTheTailOn(CustomTestCase):
    def test_trtllm_is_bypassed_and_the_rows_route_armed(self):
        import inspect

        from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb

        src = inspect.getsource(qb.QwenSparseAttnBackend._forward_paged_attention)
        self.assertIn("if self.dcp_size <= 1 and not tail_on", src)
        self.assertIn("_qsa_rows_path_armed() or tail_on", src)
        dec = inspect.getsource(qb.QwenSparseAttnBackend.forward_decode)
        self.assertIn("(tail_on and q.is_cuda)", dec)
        ext = inspect.getsource(qb.QwenSparseAttnBackend._forward_extend_impl)
        self.assertIn("(tail_on and q.is_cuda)", ext)

    def test_the_tail_off_rows_call_is_the_pre_1243_call(self):
        """No ``tail=`` keyword at all with the tail off: a patched or older
        kernel signature must see exactly the call it saw before."""
        from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb

        b = _qsa_backend(ring_on=False)
        seen = {}

        def fake(q, k, v, rows, s, **kw):
            seen.update(kw)
            return q.float(), torch.zeros(q.shape[:2])

        real = qb.sparse_attn_rows_triton
        qb.sparse_attn_rows_triton = fake
        try:
            b._attend_rows(torch.randn(2, 4, HEAD_DIM), _Layer(1), torch.zeros(2, 3, dtype=torch.int32))
        finally:
            qb.sparse_attn_rows_triton = real
        self.assertNotIn("tail", seen)


class TestTheRingCellIsTheFullAttentionKvOnly(CustomTestCase):
    def test_the_qsa_index_bytes_are_not_charged_to_the_ring(self):
        from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator

        cfg = DefaultPoolConfigurator.__new__(DefaultPoolConfigurator)
        cfg._cell_size = 12288 + 768
        cfg._kv_tail_target_cell_size = 12288 + 768
        cfg._kv_tail_qsa_cell = 768
        cfg._kv_tail_body_itemsize = 1
        cfg._kv_tail_mr = types.SimpleNamespace(
            server_args=types.SimpleNamespace(
                kv_tail_min_tokens=16, kv_tail_max_tokens=-1, kv_tail_ring_rows=None,
                kv_tail_host_max_tokens=None, max_running_requests=1, kv_cache_dtype="fp8_e4m3",
            )
        )
        import sglang.srt.model_executor.pool_configurator as pc

        real = pc.get_parallel
        pc.get_parallel = lambda: types.SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0)
        try:
            post, terms = cfg._kv_tail_ring_post(page_size=64)
        finally:
            pc.get_parallel = real
        # Next Flash: 12 full-attention layers x 2 kv heads x 256 x (K+V) x 2 B
        self.assertEqual(terms["cell"], 12288 * 2)
        self.assertEqual(post, 16 * 12288 * 2)


if __name__ == "__main__":
    unittest.main()
