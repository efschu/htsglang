# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#274 lane prefill chunking (DESIGN_121 §13.10), hermetic falsifiers.

The chunked lane prefill replaces the one whole-prompt forward with a loop of
extend forwards over the same request.  Three properties carry the whole
posten, and each has a way to be silently wrong that these tests make loud:

* **Tiling.**  The chunk plan must cover ``[0, n)`` exactly once, in order.
  An overlapping or gapped plan writes KV twice or never for some position
  and nothing downstream would raise -- the guard in ``_prefill_chunked``
  must, and the corrupt-plan test proves that guard can fire.

* **The head's boundary token (§13.10 point 3, "the part with the real
  risk").**  The NEXTN head's input is shifted one position left, so the
  LAST row of every chunk needs the token one past the chunk end.  For a
  middle chunk that token is the PROMPT's; only the final chunk may use the
  target's prediction.  The single-forward code appends the target argmax
  unconditionally -- an implementation that keeps doing that per chunk
  primes the head's KV against a token the prompt never contained.  The mock
  target's argmax is deliberately disjoint from the prompt alphabet so that
  exact mistake changes an assertion, not a probability.

* **The counter grain.**  ``work_total["prefill_tokens"]`` moving from the
  job boundary to the chunk boundary is the declared yield of the posten
  (finer grains for the pairing decider).  The tests read the counter from
  INSIDE the mocked forwards: a counter still advanced once at the end shows
  up as three identical mid-forward readings, not as a passing sum.

Everything runs on CPU driving the real ``_prefill`` / ``_prefill_chunked``
bodies with the batch machinery mocked (the ``test_lane_hidden_view_399``
pattern): ``_make_batch``, the timed forwards and ``_draft_forward`` are
recording stubs, the loop logic under test is the production code.
"""

import logging
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor import dual_group_lane as dgl
from sglang.srt.model_executor.dual_group_lane import (
    DualGroupLane,
    plan_prefill_chunks,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

HIDDEN = 4
# The mock target's argmax lives at prompt-token + ARGMAX_OFFSET, so a head
# input that ever contains a value >= ARGMAX_OFFSET at a middle-chunk
# boundary is the exact §13.10-point-3 defect, not noise.
ARGMAX_OFFSET = 1000


class _Out:
    def __init__(self, hidden_states, next_token_logits):
        self.hidden_states = hidden_states
        self.next_token_logits = next_token_logits


class _FakeReq:
    def __init__(self, ids):
        self.origin_input_ids = list(ids)
        self.full_untruncated_fill_ids = list(ids)
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.req_pool_idx = 7
        self.extend_range = None

    def set_extend_range(self, start, end):
        self.extend_range = (start, end)


class _FakeBatch:
    """The four fields and one method of ``ScheduleBatch`` the chunked path
    touches, with the falsifier built into ``prepare_for_extend``: the
    request's prefix must equal the chunk start, or the KV the forward would
    attend over does not exist yet."""

    def __init__(self, req, tag, log, next_slot):
        self.reqs = [req]
        self.device = "cpu"
        self.input_ids = None
        self.prefill_input_ids_cpu = None
        self.out_cache_loc = None
        self._tag = tag
        self._log = log
        self._next_slot = next_slot

    def prepare_for_extend(self):
        req = self.reqs[0]
        start, end = req.extend_range
        if len(req.prefix_indices) != start:
            raise AssertionError(
                f"{self._tag}: chunk starts at {start} but the request's "
                f"prefix holds {len(req.prefix_indices)} slots -- the chunk "
                "would attend over KV that was never written"
            )
        self.out_cache_loc = torch.arange(
            self._next_slot[0], self._next_slot[0] + (end - start)
        )
        self._next_slot[0] += end - start
        self.input_ids = torch.tensor(
            req.full_untruncated_fill_ids[start:end], dtype=torch.int64
        )
        self._log.append(("prepare", self._tag, start, end))


def _make_lane(prompt, spec, chunk_flag=None, ladder=(16, 32, 64, 128)):
    """A ``DualGroupLane`` shell around the real prefill bodies.

    Instance attributes shadow the class methods for exactly the helpers the
    production loop calls out to; the loop itself is the code under test.
    """
    lane = DualGroupLane.__new__(DualGroupLane)
    lane.lane_id = 0
    lane.draft_runner = object() if spec else None
    lane.work_total = {"prefill_tokens": 0, "decode_tokens": 0}
    lane._last_wall_ms = None
    lane._last_margin = None
    lane._chunk_ladder_warned = set()
    lane.runner = SimpleNamespace(
        server_args=SimpleNamespace(
            dual_group_lane_prefill_chunk=chunk_flag,
            cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(bs=list(ladder))),
        )
    )

    log = []
    next_slot = [100]
    next_slot_d = [500]

    def _make_batch(job, runner=None, req=None):
        tag = "head" if runner is not None else "target"
        if req is None:
            req = _FakeReq(job["input_ids"])
            req.set_extend_range(0, len(job["input_ids"]))
        return _FakeBatch(
            req, tag, log, next_slot_d if runner is not None else next_slot
        )

    def _timed_forward_raw(batch, capture_mode=None):
        start, end = batch.reqs[0].extend_range
        rows = end - start
        log.append(("target_fwd", start, end, lane.work_total["prefill_tokens"]))
        hidden = torch.zeros(rows, HIDDEN)
        for i in range(rows):
            hidden[i].fill_(float(start + i))
        logits = torch.zeros(1, ARGMAX_OFFSET + end + 1)
        logits[0, prompt[end - 1] + ARGMAX_OFFSET] = 10.0
        return _Out(hidden, logits), 1.0

    def _timed_forward(batch):
        start, end = batch.reqs[0].extend_range
        log.append(("target_fwd", start, end, lane.work_total["prefill_tokens"]))
        return torch.tensor([prompt[end - 1] + ARGMAX_OFFSET]), 1.0

    def _draft_forward(batch_d, hidden_states):
        log.append(
            (
                "head_fwd",
                batch_d.reqs[0].extend_range,
                [int(t) for t in batch_d.input_ids],
                int(hidden_states.shape[0]),
            )
        )
        return None

    lane._make_batch = _make_batch
    lane._timed_forward_raw = _timed_forward_raw
    lane._timed_forward = _timed_forward
    lane._draft_forward = _draft_forward
    lane._record_pool_checksum = lambda job, path=None: log.append(("checksum", path))
    lane._record_margin = lambda job, value: None
    lane._dbg_on = lambda: False
    lane._log = log
    return lane


def _job(prompt, **kw):
    job = {"input_ids": list(prompt), "max_new_tokens": 4, "output_ids": []}
    job.update(kw)
    return job


class TestChunkPlan(CustomTestCase):
    def test_plan_tiles_exactly(self):
        self.assertEqual(plan_prefill_chunks(0, 10, 4), [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(plan_prefill_chunks(0, 8, 4), [(0, 4), (4, 8)])
        self.assertEqual(plan_prefill_chunks(0, 3, 8), [(0, 3)])
        self.assertEqual(plan_prefill_chunks(2, 7, 2), [(2, 4), (4, 6), (6, 7)])
        for n in range(1, 40):
            for chunk in range(1, 12):
                spans = plan_prefill_chunks(0, n, chunk)
                self.assertEqual(spans[0][0], 0)
                self.assertEqual(spans[-1][1], n)
                for (a, b), (c, d) in zip(spans, spans[1:]):
                    self.assertEqual(b, c)
                    self.assertLess(a, b)
                    self.assertLess(c, d)
                self.assertTrue(all(b - a <= chunk for a, b in spans))

    def test_plan_rejects_invalid(self):
        with self.assertRaises(ValueError):
            plan_prefill_chunks(0, 10, 0)
        with self.assertRaises(ValueError):
            plan_prefill_chunks(0, 10, -3)
        with self.assertRaises(ValueError):
            plan_prefill_chunks(10, 10, 4)
        with self.assertRaises(ValueError):
            plan_prefill_chunks(-1, 10, 4)


class TestChunkedPrefill(CustomTestCase):
    def test_nonspec_counter_moves_on_chunk_boundary(self):
        prompt = list(range(10))
        lane = _make_lane(prompt, spec=False)
        job = _job(prompt, prefill_chunk=4)
        DualGroupLane._prefill(lane, job)

        fwds = [e for e in lane._log if e[0] == "target_fwd"]
        self.assertEqual([(s, e) for _, s, e, _ in fwds], [(0, 4), (4, 8), (8, 10)])
        # The counter read INSIDE forward k must already hold chunks 0..k-1;
        # a job-boundary counter reads 0 three times and fails here even
        # though its final sum would be right.
        self.assertEqual([c for *_, c in fwds], [0, 4, 8])
        self.assertEqual(lane.work_total["prefill_tokens"], 10)
        # Exactly one emission, from the final chunk.
        self.assertEqual(job["output_ids"], [prompt[-1] + ARGMAX_OFFSET])
        self.assertEqual(job["prefill_ms"], 3.0)
        self.assertEqual(len(job["prefill_chunk_ms"]), 3)

    def test_spec_head_gets_prompt_token_at_middle_boundaries(self):
        prompt = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        lane = _make_lane(prompt, spec=True)
        job = _job(prompt, prefill_chunk=4)
        DualGroupLane._prefill(lane, job)

        heads = [e for e in lane._log if e[0] == "head_fwd"]
        self.assertEqual(len(heads), 3)
        # Middle chunks: shifted prompt slice ending in the NEXT PROMPT
        # token. Any ARGMAX_OFFSET-range value here is the point-3 defect.
        self.assertEqual(heads[0][2], prompt[1:5])
        self.assertEqual(heads[1][2], prompt[5:9])
        for _, _, ids, _ in heads[:-1]:
            self.assertTrue(
                all(t < ARGMAX_OFFSET for t in ids),
                f"head primed with a target prediction at a middle chunk: {ids}",
            )
        # Final chunk: shifted tail plus the target's own next token.
        self.assertEqual(heads[2][2], prompt[9:] + [prompt[-1] + ARGMAX_OFFSET])
        # Hidden rows handed to the head match the chunk row count.
        self.assertEqual([h[3] for h in heads], [4, 4, 2])

    def test_spec_interleave_and_prefix_growth(self):
        prompt = list(range(12))
        lane = _make_lane(prompt, spec=True)
        job = _job(prompt, prefill_chunk=4)
        DualGroupLane._prefill(lane, job)

        # Per chunk: target prepare, target forward, head prepare, head
        # forward -- the head of chunk k must never run before its target
        # chunk produced the hidden states it consumes.
        kinds = [
            e[0] if e[0] != "prepare" else f"prepare_{e[1]}"
            for e in lane._log
            if e[0] in ("prepare", "target_fwd", "head_fwd")
        ]
        self.assertEqual(
            kinds,
            ["prepare_target", "target_fwd", "prepare_head", "head_fwd"] * 3,
        )
        # prepare_for_extend's built-in falsifier already enforced
        # prefix == chunk start for every chunk on both runners; check the
        # final lengths landed at n.
        prepares = [e for e in lane._log if e[0] == "prepare"]
        self.assertEqual([p[2:] for p in prepares[::2]], [(0, 4), (4, 8), (8, 12)])
        self.assertEqual([p[2:] for p in prepares[1::2]], [(0, 4), (4, 8), (8, 12)])

    def test_spec_bookkeeping_parity(self):
        prompt = list(range(9))
        lane = _make_lane(prompt, spec=True)
        job = _job(prompt, prefill_chunk=4)
        DualGroupLane._prefill(lane, job)

        self.assertEqual(job["_kv_len"], 9)
        self.assertEqual(job["_kv_len_draft"], 9)
        self.assertIn("_batch_d", job)
        self.assertIn("_batch", job)
        self.assertIn("_req", job)
        self.assertEqual(job["_req_pool_idx"], 7)
        # _hidden is the last row of the last chunk, cloned; the mock writes
        # position index into every hidden row, so the value pins the row.
        self.assertEqual(float(job["_hidden"][0, 0]), 8.0)
        self.assertEqual(int(job["_next"][0]), prompt[-1] + ARGMAX_OFFSET)
        self.assertEqual(job["output_ids"], [prompt[-1] + ARGMAX_OFFSET])
        self.assertAlmostEqual(
            job["prefill_ms"], sum(job["prefill_chunk_ms"]), places=9
        )
        # The #404 anchor still lands after a chunked prefill.
        self.assertIn(("checksum", "prefill"), lane._log)

    def test_degenerate_single_chunk(self):
        prompt = list(range(5))
        lane = _make_lane(prompt, spec=False)
        job = _job(prompt, prefill_chunk=64)
        DualGroupLane._prefill(lane, job)
        fwds = [e for e in lane._log if e[0] == "target_fwd"]
        self.assertEqual([(s, e) for _, s, e, _ in fwds], [(0, 5)])
        self.assertEqual(job["output_ids"], [prompt[-1] + ARGMAX_OFFSET])


class TestDispatch(CustomTestCase):
    def test_default_stays_single_forward(self):
        prompt = list(range(10))
        lane = _make_lane(prompt, spec=False, chunk_flag=None)

        def _boom(job, chunk):
            raise AssertionError("chunked path reached without a chunk size")

        lane._prefill_chunked = _boom
        job = _job(prompt)
        DualGroupLane._prefill(lane, job)
        prepares = [e for e in lane._log if e[0] == "prepare"]
        self.assertEqual(prepares, [("prepare", "target", 0, 10)])
        self.assertEqual(lane.work_total["prefill_tokens"], 10)

    def test_flag_engages_and_job_override_wins(self):
        prompt = list(range(10))
        # Flag on, no override: chunked.
        lane = _make_lane(prompt, spec=False, chunk_flag=4)
        job = _job(prompt)
        DualGroupLane._prefill(lane, job)
        fwds = [e for e in lane._log if e[0] == "target_fwd"]
        self.assertEqual(len(fwds), 3)
        # Flag on, job says 0: single forward.
        lane = _make_lane(prompt, spec=False, chunk_flag=4)
        job = _job(prompt, prefill_chunk=0)
        DualGroupLane._prefill(lane, job)
        fwds = [e for e in lane._log if e[0] == "target_fwd"]
        self.assertEqual(len(fwds), 1)

    def test_off_ladder_chunk_warns_once(self):
        prompt = list(range(10))
        lane = _make_lane(prompt, spec=False, ladder=(16, 32))
        with self.assertLogs(dgl.logger, level=logging.WARNING) as cm:
            DualGroupLane._prefill(lane, _job(prompt, prefill_chunk=5))
            DualGroupLane._prefill(lane, _job(prompt, prefill_chunk=5))
        ladder_warnings = [m for m in cm.output if "prefill tier ladder" in m]
        self.assertEqual(len(ladder_warnings), 1)

    def test_on_ladder_chunk_is_silent(self):
        prompt = list(range(40))
        lane = _make_lane(prompt, spec=False, ladder=(16, 32))
        with self.assertNoLogs(dgl.logger, level=logging.WARNING):
            DualGroupLane._prefill(lane, _job(prompt, prefill_chunk=16))


class TestPlanGuardCanFail(CustomTestCase):
    """The tiling guard's can-fail proof: a corrupted plan is refused by
    ``_prefill_chunked`` itself, before any forward runs."""

    def _run_with_plan(self, spans):
        prompt = list(range(10))
        lane = _make_lane(prompt, spec=False)
        job = _job(prompt, prefill_chunk=4)
        original = dgl.plan_prefill_chunks
        dgl.plan_prefill_chunks = lambda *a, **k: spans
        try:
            with self.assertRaises(RuntimeError):
                DualGroupLane._prefill(lane, job)
        finally:
            dgl.plan_prefill_chunks = original
        # Refused BEFORE the first forward: nothing was executed.
        self.assertEqual([e for e in lane._log if e[0] == "target_fwd"], [])

    def test_overlapping_plan_refused(self):
        self._run_with_plan([(0, 4), (3, 8), (8, 10)])

    def test_gapped_plan_refused(self):
        self._run_with_plan([(0, 4), (5, 10)])

    def test_short_plan_refused(self):
        self._run_with_plan([(0, 4), (4, 8)])


if __name__ == "__main__":
    unittest.main()
