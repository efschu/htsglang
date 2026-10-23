"""27b-ddeath-1023: a long chunked D prefill must not exhaust the DFLASH
window pool.

Boot boot_weg2_dkr27bint4ardcurvebar1int4ar10231804 (27B AutoRound, D = TP2
on 2x3080, window pool W=2048, 24673 draft slots = 1 + (2048 + 8) x 6 x 2)
died at 18:10:03Z on a 32k-token D prefill at depth 0 with no decode running:

    RuntimeError: DFLASH small solo pool exhausted: need 87 more draft slots
    but only 0 reclaimable of 24673 total (ctx cap 2048)

at prefix 22608 + chunk 264. The mapper's LRU reclaim frees only slots "not
touched in the current round", and the round advanced on the decode prep
alone -- every prefill chunk of the request was written at one epoch.

These tests drive the PRODUCTION call site (DFlashWorkerV2.prefill_after_target)
chunk by chunk against the real mapper at the boot's geometry, and bind the
RESULT: the full prompt goes through, and the request's last W rows (its next
decode window) stay mapped. CPU only.
"""

import types
import unittest

import torch


def _t(vals):
    return torch.tensor(list(vals), dtype=torch.int64)

from flliper.srt.speculative.dflash_solo_pool import DraftKVSlotMapper
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# The boot's geometry (D.log 18:07:01 'DFLASH small solo draft pool ACTIVE').
W = 2048
BLOCK = 8
MRR = 6
FACTOR = 2
DRAFT_SLOTS = 1 + (W + BLOCK) * MRR * FACTOR  # 24673
GLOBAL_SLOTS = 278852


# Target req_to_token of the fake: row r, position p -> global slot
# base(r) + p, the same addressing _chunk uses for out_cache_loc.
REQ_ROWS = 8
REQ_WIDTH = 140000


def _base(r):
    return 1 + r * 10000


def _req_to_token():
    pos = torch.arange(REQ_WIDTH, dtype=torch.int64).unsqueeze(0)
    base = torch.tensor([_base(r) for r in range(REQ_ROWS)], dtype=torch.int64)
    return (base.unsqueeze(1) + pos).clamp_(max=GLOBAL_SLOTS)


def _fake_worker(mapper, window_pool=True):
    from flliper.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    def _append(target_hidden, cache_loc, positions):
        # The real _append_target_hidden_to_draft_kv_by_loc translates the
        # 1-D write addressing through the mapper first (allocating draft
        # slots); that translation is the call that raised in the boot.
        mapper.translate_write(cache_loc)

    fake = types.SimpleNamespace(
        draft_window_size=W,
        _solo_pool_filter_prefill=lambda batch, h, loc, pos: (h, loc, pos),
        _target_context_hidden=lambda lo: lo.hidden_states,
        _tp_sync=types.SimpleNamespace(sync=lambda site, t: None),
        _solo_is_shadow=False,
        model_runner=types.SimpleNamespace(
            server_args=types.SimpleNamespace(attention_backend="torch_native"),
            req_to_token_pool=types.SimpleNamespace(req_to_token=_req_to_token()),
        ),
        _solo_pool_mapper=mapper,
        _window_pool=window_pool,
        _append_target_hidden_to_draft_kv_by_loc=_append,
        _release_target_context_hidden=lambda lo: None,
        _make_next_draft_input_prefill=lambda bonus_tokens, seq_lens: None,
    )
    fake._window_pool_mark_prefill_tails = (
        lambda batch, m: DFlashWorkerV2._window_pool_mark_prefill_tails(fake, batch, m)
    )
    return lambda batch, out: DFlashWorkerV2.prefill_after_target(fake, batch, out)


def _chunk(prefix, n, row):
    base_slot = _base(row)
    batch = types.SimpleNamespace(
        req_pool_indices=torch.tensor([row], dtype=torch.int64),
        extend_lens=[n],
        prefix_lens=[prefix],
        out_cache_loc=torch.arange(
            base_slot + prefix, base_slot + prefix + n, dtype=torch.int64
        ),
        seq_lens=torch.tensor([prefix + n], dtype=torch.int64),
    )
    out = types.SimpleNamespace(
        logits_output=types.SimpleNamespace(hidden_states=torch.zeros(n, 1)),
        next_token_ids=torch.zeros(1, dtype=torch.int64),
    )
    return batch, out


def _mapper(sync_free):
    return DraftKVSlotMapper(
        num_global_slots=GLOBAL_SLOTS,
        num_draft_slots=DRAFT_SLOTS,
        ctx_cap=W,
        device="cpu",
        sync_free=sync_free,
    )


def _warm_last_decode_round(m, rows=1800):
    """An earlier request's window rows, written and then READ in the last
    decode round (globals in, globals in -- never draft slots fed back)."""
    m.begin_round()
    g = torch.arange(200001, 200001 + rows, dtype=torch.int64)
    m.translate_write(g)
    m.translate_read(g)
    return g


class TestLongChunkedPrefillWindowPool(CustomTestCase):
    def _run(self, sync_free, prompt=32768, chunk=264, warm_rows=1800):
        m = _mapper(sync_free)
        # The boot's state when the 32k sample arrived: an earlier request's
        # window rows touched in the last decode round.
        g = _warm_last_decode_round(m, warm_rows)
        self.assertTrue(bool((m.map[g] > 0).all()))
        prefill = _fake_worker(m)
        base = _base(0)
        done = 0
        while done < prompt:
            n = min(chunk, prompt - done)
            prefill(*_chunk(done, n, 0))
            done += n
        # Exactly the last W rows carry the window-tail mark.
        self.assertEqual(int(m._slot_tail.sum()), W)
        # The request's next decode window: its last W global slots.
        last_w = torch.arange(base + prompt - W, base + prompt, dtype=torch.int64)
        self.assertTrue(bool((m.map[last_w] > 0).all()))
        return m

    def test_32k_prefill_at_boot_geometry_survives_legacy_mapper(self):
        m = self._run(sync_free=False)
        self.assertGreater(m.reclaim_events, 0)

    def test_32k_prefill_at_boot_geometry_survives_sync_free_mapper(self):
        # The boot ran FLLIPER_DFLASH_WINDOW_POOL_SYNC_FREE=1 (the raising
        # frame was _translate_write_sync_free -> _exact_section).
        m = self._run(sync_free=True)
        self.assertGreater(m.reclaim_events, 0)

    def test_prompt_far_beyond_pool_survives(self):
        # X ceiling of the profile is 131072: a D prefill up to X is legal
        # (SCHEDULER-SOLL R7), so the pool must carry it, not die on it.
        self._run(sync_free=True, prompt=131072, chunk=4096)

    def test_production_burst_under_small_x_survives(self):
        # Reachable under the production X (4096 / 12288) too: prefill has
        # priority over decode in the scheduler, so a burst of queued
        # requests prefills back to back with no decode prep in between.
        # Six seats x 4000 uncached tokens (each < X=4096) + the last decode
        # round's window rows exceed the 24672 slots at one frozen epoch.
        for sync_free in (False, True):
            m = _mapper(sync_free)
            _warm_last_decode_round(m)
            prefill = _fake_worker(m)
            for r in range(MRR):
                prefill(*_chunk(0, 4000, r))
            self.assertGreater(m.reclaim_events, 0)
            for r in range(MRR):
                tail = torch.arange(_base(r) + 4000 - W, _base(r) + 4000)
                self.assertTrue(
                    bool((m.map[tail] > 0).all()),
                    f"sync_free={sync_free}: window of request {r} holed",
                )

    def test_round_advances_per_prefill_forward(self):
        m = _mapper(sync_free=False)
        prefill = _fake_worker(m)
        e0 = m._epoch
        prefill(*_chunk(0, 16, 0))
        prefill(*_chunk(16, 16, 0))
        self.assertEqual(m._epoch, e0 + 2)

    def test_non_window_solo_pool_keeps_decode_rounds_only(self):
        # The full-context solo pool must not drop early chunks of a live
        # request (permanent mid-context hole): no round advance, no marks.
        m = _mapper(sync_free=False)
        prefill = _fake_worker(m, window_pool=False)
        e0 = m._epoch
        prefill(*_chunk(0, 16, 0))
        prefill(*_chunk(16, 16, 0))
        self.assertEqual(m._epoch, e0)
        self.assertEqual(int(m._slot_tail.sum()), 0)

    def test_reclaim_takes_marked_tail_last(self):
        m = DraftKVSlotMapper(
            num_global_slots=1000, num_draft_slots=9, ctx_cap=4, device="cpu"
        )
        old = _t(range(1, 5))
        newer = _t(range(11, 15))
        m.begin_round()
        m.translate_write(old)
        m.mark_window_tail(old)  # older but a live window: survives
        m.begin_round()
        m.translate_write(newer)
        m.begin_round()
        m.translate_write(_t([21, 22]))  # forces a reclaim
        self.assertTrue(bool((m.map[old] > 0).all()))

    def test_decode_clear_drops_marks(self):
        m = _mapper(sync_free=False)
        g = _t(range(1, 9))
        m.translate_write(g)
        m.mark_window_tail(g)
        self.assertEqual(int(m._slot_tail.sum()), 8)
        m.clear_window_tail()
        self.assertEqual(int(m._slot_tail.sum()), 0)

    def test_current_forward_rows_are_still_protected(self):
        # A single forward larger than the whole pool still raises the
        # readable error (nothing older to reclaim) -- the fix does not make
        # the mapper evict rows of the forward it is writing.
        m = DraftKVSlotMapper(
            num_global_slots=1000, num_draft_slots=17, ctx_cap=8, device="cpu"
        )
        prefill = _fake_worker(m)
        with self.assertRaisesRegex(RuntimeError, "small solo pool exhausted"):
            prefill(*_chunk(0, 32, 0))


if __name__ == "__main__":
    unittest.main()
