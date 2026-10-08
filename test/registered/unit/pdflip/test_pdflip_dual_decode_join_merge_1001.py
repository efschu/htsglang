# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 stage 2 step 2: decode-join wiring -- split, conversion, spec-state merge/filter.

DANGER DIRECTIONS guarded here:
* split: switch off -> nothing joins (old path byte-identical); the chunked request and
  anchor tails never join; non-joinable requests fall back to extend AND are named;
  outside the dual layout nothing joins;
* conversion (no forward): committed KV = N-1 on the request and in seq_lens (cpu/gpu/orig),
  DECODE mode, no input_ids/out_cache_loc, pending token prompt[N-1] as the DFlash bonus,
  no output token; under overlap the state is published/stashed under the pool indices;
* merge/filter of DFlashDraftInputV2 with joined requests keeps bonus/seq_len rows aligned
  with the requests for every decode-graph batch size 1..8, in tensor AND future-index form;
* the scheduler wires split -> _pdflip_build_join_batch -> merge at get_next_batch_to_run.
"""
from __future__ import annotations

import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from flliper.srt.speculative.draft_worker_common import make_draft_input_v2
from flliper.srt.pdflip import dual_decode_join as J
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ON = {J.JOIN_ENV: "1", J.DUAL_LAYOUT_ENV: "1"}


def _req(rid, n=10, prefix=None, **kw):
    r = types.SimpleNamespace(rid=rid, origin_input_ids=list(range(1000 + 10 * n, 1000 + 11 * n)),
                              output_ids=[], multimodal_inputs=None, return_logprob=False,
                              logprob_start_len=-1, cached_tokens=0, cached_tokens_device=0,
                              already_computed=0, kv_committed_len=n, kv_allocated_len=n)
    r.prefix_indices = torch.arange(n - 1 if prefix is None else prefix)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


class _FM:
    def __init__(self):
        self.published, self.stashed = [], []

    def publish(self, idx, lens):
        self.published.append((idx.clone(), lens.clone()))

    def stash(self, idx, payload):
        self.stashed.append((idx.clone(), payload.bonus_tokens.clone(), payload.topk_p))


def _batch(reqs, pool0=40):
    return types.SimpleNamespace(reqs=reqs, device=torch.device("cpu"),
                                 req_pool_indices=torch.arange(pool0, pool0 + len(reqs)),
                                 seq_lens=torch.tensor([len(r.origin_input_ids) for r in reqs]),
                                 input_ids=torch.tensor([1]), out_cache_loc=torch.tensor([5]),
                                 forward_mode=None, spec_info=None)


class DualDecodeJoinMerge(CustomTestCase):
    def test_split_switch_off_is_the_old_path(self):
        reqs = [_req("a"), _req("b")]
        self.assertEqual(J.split_join_reqs(reqs, spec_is_dflash=True, env={}), ([], reqs, []))

    def test_split_outside_dual(self):
        j, rest, fb = J.split_join_reqs([_req("a")], spec_is_dflash=True, env={J.JOIN_ENV: "1"})
        self.assertEqual(j, [])
        self.assertEqual(len(rest), 1)

    def test_split_joins_and_names_fallbacks(self):
        a, b, c, d = _req("a"), _req("b", prefix=7), _req("c", output_ids=[3]), _req("d")
        j, rest, fb = J.split_join_reqs([a, b, c, d], spec_is_dflash=True, exclude=[d], env=ON)
        self.assertEqual(j, [a])
        self.assertEqual(rest, [b, c, d])
        self.assertEqual([r.rid for r, _ in fb], ["b", "c"])   # d excluded silently (chunked/tail)

    def test_convert_overlap(self):
        reqs = [_req("a", n=10), _req("b", n=20)]
        b, fm = _batch(reqs), _FM()
        J.convert_to_joined(b, fm, enable_overlap=True)
        self.assertEqual([r.kv_committed_len for r in reqs], [9, 19])
        self.assertEqual([r.output_ids for r in reqs], [[], []])
        self.assertTrue(all(r._pdflip_decode_joined for r in reqs))
        self.assertEqual(b.seq_lens.tolist(), [9, 19])
        self.assertEqual(b.seq_lens_cpu.tolist(), [9, 19])
        self.assertEqual(b.orig_seq_lens.tolist(), [9, 19])
        self.assertEqual(b.seq_lens_sum, 28)
        self.assertTrue(b.forward_mode.is_decode())
        self.assertIsNone(b.input_ids)
        self.assertIsNone(b.out_cache_loc)
        self.assertEqual(b.spec_info.bonus_tokens.tolist(),
                         [reqs[0].origin_input_ids[-1], reqs[1].origin_input_ids[-1]])
        self.assertEqual(b.spec_info.new_seq_lens.tolist(), [9, 19])
        self.assertEqual(b.spec_info.future_indices.tolist(), [40, 41])
        (pi, pl), = fm.published
        self.assertEqual((pi.tolist(), pl.tolist()), ([40, 41], [9, 19]))
        (si, sb, stopk), = fm.stashed
        self.assertEqual(si.tolist(), [40, 41])
        self.assertEqual(sb.tolist(), b.spec_info.bonus_tokens.tolist())
        self.assertIsNone(stopk)   # DFlash's zero-width Eagle placeholders relay as absent

    def test_convert_no_overlap(self):
        reqs = [_req("a")]
        b, fm = _batch(reqs), _FM()
        J.convert_to_joined(b, fm, enable_overlap=False)
        self.assertIsNone(b.spec_info.future_indices)
        self.assertEqual((fm.published, fm.stashed), ([], []))

    def test_merge_filter_alignment_all_graph_bs(self):
        for bs_run in range(0, 8):
            for bs_join in range(1, 9 - bs_run):
                run = make_draft_input_v2(bonus_tokens=torch.arange(100, 100 + bs_run),
                                          new_seq_lens=torch.arange(500, 500 + bs_run))
                join = make_draft_input_v2(bonus_tokens=torch.arange(900, 900 + bs_join),
                                           new_seq_lens=torch.arange(50, 50 + bs_join))
                run.merge_batch(join)
                n = bs_run + bs_join
                self.assertEqual(run.bonus_tokens.tolist(),
                                 list(range(100, 100 + bs_run)) + list(range(900, 900 + bs_join)))
                self.assertEqual(run.new_seq_lens.numel(), n)
                self.assertEqual(tuple(run.topk_p.shape), (n, 0))
                keep = torch.tensor([i for i in range(n) if i % 2 == 0])
                run.filter_batch(keep, keep.tolist())
                self.assertEqual(run.bonus_tokens.tolist(),
                                 [([*range(100, 100 + bs_run), *range(900, 900 + bs_join)])[i] for i in keep.tolist()])

    def test_merge_filter_future_index_form(self):
        run = make_draft_input_v2(bonus_tokens=torch.arange(3), new_seq_lens=torch.arange(3))
        run.future_indices = torch.tensor([1, 2, 3])
        join = make_draft_input_v2(bonus_tokens=torch.tensor([7]), new_seq_lens=torch.tensor([9]))
        join.future_indices = torch.tensor([40])
        run.merge_batch(join)
        self.assertEqual(run.future_indices.tolist(), [1, 2, 3, 40])
        run.filter_batch(torch.tensor([3, 0]), [3, 0])
        self.assertEqual(run.future_indices.tolist(), [40, 1])

    def test_merge_joined_into_empty_and_full(self):
        a, b = types.SimpleNamespace(is_empty=lambda: True), object()
        self.assertIs(J.merge_joined(a, [b]), b)
        got = []
        full = types.SimpleNamespace(is_empty=lambda: False, merge_batch=got.append)
        self.assertIs(J.merge_joined(full, [b]), full)
        self.assertEqual(got, [b])

    def test_scheduler_wiring(self):
        from flliper.srt.managers import scheduler as S

        raw = inspect.getsource(S.Scheduler._get_new_batch_prefill_raw)
        self.assertIn("_ddj.split_join_reqs(", raw)
        self.assertIn("if not _ddj.join_enabled():", raw)   # switch off: list untouched
        self.assertIn("self._pdflip_build_join_batch(_join)", raw)
        self.assertLess(raw.index("_ddj.split_join_reqs("), raw.index("new_batch.prepare_for_extend()"))
        build = inspect.getsource(S.Scheduler._pdflip_build_join_batch)
        self.assertLess(build.index("jb.prepare_for_extend()"), build.index("arm_draft_cold_for_admission(self, jb)"))
        self.assertLess(build.index("arm_draft_cold_for_admission(self, jb)"), build.index("_ddj.convert_to_joined("))
        nxt = inspect.getsource(S.Scheduler.get_next_batch_to_run)
        self.assertIn("_ddj.merge_joined(running_batch, _jr)", nxt)
