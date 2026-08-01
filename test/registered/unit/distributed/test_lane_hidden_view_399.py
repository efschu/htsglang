"""#399: ``job["_hidden"]`` as a view into the verify forward's output.

The lane's speculative chain stores two things past the forward that produced
them. ``_verify_rows`` is ``.clone()``d, with a comment naming the exact
hazard -- a later forward writing through a view of a buffer the caller still
holds (Geteilte-Puffer-Familie). ``_hidden``, fifteen lines above it in the
same function, was not. This file settles what that asymmetry was worth, on
CPU tensors, by driving the real ``_propose`` -> ``_verify`` ->
``_rollback_draft`` sequence with the two model forwards mocked.

The mock is faithful in the one property that decides the question: a captured
replay does NOT return a fresh tensor. ``DecodeCudaGraphRunner`` hands back
``output.hidden_states[: raw_num_token]`` of the tensor it retained at capture
time, so every replay of that shape overwrites the rows a previous caller may
still be holding, and a Python reference does not prevent it.
``_StaticGraphOutputs`` below is exactly that contract and nothing else.

Two results come out of it, and they are separate:

* the view IS written through when a same-shape replay lands between the
  verify and the next round's ``_propose``, and the corrupted rows arrive as
  the head's input -- the head then proposes from the wrong hidden state and
  every proposal is rejected;
* the COMMITTED tokens are byte-identical anyway. ``_hidden`` is read by
  ``_draft_forward`` and by nothing else; the target's verify forward takes
  ``input_ids`` and the pool, never a hidden state; and under the greedy
  accept rule every emitted token is a target prediction whose conditioning
  tokens are the committed prefix. A corrupted ``_hidden`` moves the accept
  length and leaves the trajectory alone.

The second result is the desk argument of the #399 diagnosis, made
executable. It is why the clone is a latent-hazard fix and not the carrier of
the ``squares`` content divergence.
"""

import os
import unittest

import torch

from sglang.srt.model_executor.dual_group_lane import DualGroupLane
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


HIDDEN = 4
VOCAB = 512
POOL_SLOTS = 512
MAX_POS = 128


class _Out:
    """The two fields of ``LogitsProcessorOutput`` this path reads."""

    def __init__(self, hidden_states, next_token_logits):
        self.hidden_states = hidden_states
        self.next_token_logits = next_token_logits


class _StaticGraphOutputs:
    """One retained buffer per captured shape, overwritten by every replay.

    The production contract, reduced to its aliasing behaviour: the caller
    receives a SLICE of a tensor the runner keeps, so holding the slice does
    not protect its contents from the next replay of the same shape.
    """

    def __init__(self):
        self.hidden = {}
        self.logits = {}
        self.replays = 0

    def replay(self, rows, tokens):
        """Write ``tokens`` into the retained buffers and hand back slices.

        Row ``i`` carries token ``tokens[i]`` in its hidden state and an
        argmax at ``tokens[i] + 1`` in its logits -- a counting target, so
        "what the target predicts after token t" is a single readable number
        and a corrupted hidden row shows up as a wrong proposal rather than
        as noise.
        """
        self.replays += 1
        h = self.hidden.get(rows)
        if h is None:
            h = self.hidden[rows] = torch.zeros(rows, HIDDEN)
        lg = self.logits.get(rows)
        if lg is None:
            lg = self.logits[rows] = torch.zeros(rows, VOCAB)
        lg.zero_()
        for i, tok in enumerate(tokens):
            h[i].fill_(float(tok))
            lg[i, (int(tok) + 1) % VOCAB] = 10.0
        return _Out(h[:rows], lg[:rows])


class _Allocator:
    def __init__(self):
        self.next = 1
        self.freed = []

    def alloc(self, n):
        out = torch.arange(self.next, self.next + n, dtype=torch.int64)
        self.next += n
        return out

    def free(self, slots):
        self.freed.append(slots.clone() if torch.is_tensor(slots) else slots)


class _ReqPool:
    def __init__(self):
        self.req_to_token = torch.zeros(2, MAX_POS, dtype=torch.int32)
        self.mamba_pool = None


class _Req:
    def __init__(self, idx, n_cached):
        self.req_pool_idx = idx
        self.decode_batch_idx = n_cached
        self.kv_committed_len = n_cached
        self.kv_allocated_len = n_cached


class _Batch:
    """Enough ``ScheduleBatch`` for the lane's verify and head bookkeeping."""

    def __init__(self, idx, n_cached, pool, allocator):
        self.device = torch.device("cpu")
        self.req_to_token_pool = pool
        self.token_to_kv_pool_allocator = allocator
        self.reqs = [_Req(idx, n_cached)]
        self.seq_lens = torch.tensor([n_cached], dtype=torch.int64)
        self.seq_lens_cpu = torch.tensor([n_cached], dtype=torch.int64)
        self.orig_seq_lens = torch.tensor([n_cached], dtype=torch.int64)
        self.seq_lens_sum = n_cached
        self.input_ids = None
        self.out_cache_loc = None
        self.spec_info = None
        self.forward_mode = None
        self.capture_hidden_mode = None
        self._idx = idx

    def prepare_for_decode(self):
        pos = int(self.seq_lens[0].item())
        slot = self.token_to_kv_pool_allocator.alloc(1)
        self.req_to_token_pool.req_to_token[self._idx, pos] = slot.to(torch.int32)[0]
        self.seq_lens.fill_(pos + 1)
        self.seq_lens_cpu.fill_(pos + 1)
        self.orig_seq_lens.fill_(pos + 1)
        self.reqs[0].decode_batch_idx = pos + 1


class _ServerArgs:
    page_size = 1


class _Runner:
    def __init__(self, pool):
        self.server_args = _ServerArgs()
        self.req_to_token_pool = pool
        self.attn_backend = object()
        self.decode_cuda_graph_runner = None
        self.model = None


class _Harness:
    """The real lane methods over mocked forwards.

    ``_timed_forward_raw`` and ``_draft_forward`` are the two model-forward
    boundaries of the speculative round; everything between them --
    ``_propose``, ``_verify``, ``_verify_by_target_verify``,
    ``_verify_predictions``, ``_rollback_draft``, ``_truncate_draft`` -- is
    the shipped code.
    """

    def __init__(self, first_token=100, n_cached=8, rung=1):
        self.graph = _StaticGraphOutputs()
        self.pool = _ReqPool()
        self.draft_pool = _ReqPool()
        self.allocator = _Allocator()
        self.draft_allocator = _Allocator()
        self.rung = rung
        #: what ``_propose`` handed to the head, per head forward
        self.head_inputs = []

        lane = DualGroupLane.__new__(DualGroupLane)
        lane.runner = _Runner(self.pool)
        lane.draft_runner = object()
        lane.spec_steps = rung
        lane.work_total = {"prefill_tokens": 0, "decode_tokens": 0}
        lane._timed_forward_raw = self._target_forward
        lane._draft_forward = self._head_forward
        self.lane = lane

        batch = _Batch(0, n_cached, self.pool, self.allocator)
        batch_d = _Batch(0, n_cached, self.draft_pool, self.draft_allocator)
        for pos in range(n_cached):
            self.pool.req_to_token[0, pos] = self.allocator.alloc(1).to(torch.int32)[0]
            self.draft_pool.req_to_token[0, pos] = self.draft_allocator.alloc(1).to(
                torch.int32
            )[0]
        self.job = {
            "_batch": batch,
            "_batch_d": batch_d,
            "_req": batch.reqs[0],
            "_req_pool_idx": 0,
            "_kv_len": n_cached,
            "_next": torch.tensor([first_token], dtype=torch.int64),
            "_rung": rung,
            "output_ids": [],
        }
        # The state ``_prefill`` would have left. The head's two inputs are
        # SHIFTED against each other exactly as they are in the real chain:
        # ``_next`` is the token at the newest position and ``_hidden`` is the
        # target's hidden state of the position BEFORE it, so the invariant
        # this toy model keeps is ``hidden == _next - 1``.
        self.job["_hidden"] = self.graph.replay(
            1, [first_token - 1]
        ).hidden_states.clone()

    # -- the two mocked forwards ----------------------------------------
    def _target_forward(self, batch, capture_mode=None):
        tokens = [int(t) for t in batch.input_ids.tolist()]
        return self.graph.replay(len(tokens), tokens), 1.0

    def _head_forward(self, batch_d, hidden_states):
        """The counting head: it proposes ``hidden + 2``.

        Two, not one, because of the shift: the head's hidden input belongs to
        the position before its token input, so continuing the count from the
        hidden state means skipping the token that already sits between them.
        Reading the proposal out of the HIDDEN state rather than out of the
        input token is what makes a corrupted ``_hidden`` observable at all --
        it is the head's other input, and in the real NEXTN head it is the one
        that carries the target's state.
        """
        self.head_inputs.append(hidden_states.clone())
        proposal = int(hidden_states[0, 0].item()) + 2
        logits = torch.zeros(1, VOCAB)
        logits[0, proposal % VOCAB] = 10.0
        return _Out(torch.full((1, HIDDEN), float(proposal)), logits)

    # -- the real round --------------------------------------------------
    def round(self, intervening_replay=False):
        """One speculative round, in ``_spec_step``'s order.

        ``intervening_replay`` fires one further replay of the verify shape
        between the verify and the next ``_propose`` -- the hazard the sibling
        clone comment names, applied to the sibling that was not cloned.
        """
        proposals = self.lane._propose(self.job)
        emitted, n_accept, _ = self.lane._verify(self.job, proposals)
        self.lane._rollback_draft(self.job, n_accept)
        self.job["output_ids"].extend(emitted)
        if intervening_replay:
            self.graph.replay(self.rung + 1, [777] * (self.rung + 1))
        return emitted, n_accept


class TestHiddenIsNotAViewIntoTheForwardOutput(CustomTestCase):
    """The storage question, asked of the shipped assignment."""

    def setUp(self):
        for var in (
            "SGLANG_LANE_MARGIN_PROBE",
            "SGLANG_LANE_SPEC_DEBUG",
            "SGLANG_LANE_SPEC_ROW_ORACLE",
            "SGLANG_LANE_SPEC_TV_MAX_ACCEPT",
            "SGLANG_LANE_SPEC_VERIFY",
        ):
            os.environ.pop(var, None)

    def test_the_stored_hidden_does_not_share_storage_with_the_verify_output(self):
        h = _Harness()
        h.round()
        stored = h.job["_hidden"]
        produced = h.graph.hidden[h.rung + 1]
        self.assertFalse(
            stored.untyped_storage().data_ptr()
            == produced.untyped_storage().data_ptr(),
            "job['_hidden'] still aliases the verify forward's retained output "
            "buffer; the next replay of that shape writes through it",
        )

    def test_a_later_replay_of_the_same_shape_does_not_change_the_stored_row(self):
        h = _Harness()
        h.round()
        before = h.job["_hidden"].clone()
        h.graph.replay(h.rung + 1, [777] * (h.rung + 1))
        self.assertTrue(
            torch.equal(before, h.job["_hidden"]),
            "the stored hidden row moved when an unrelated forward replayed "
            "the verify shape",
        )

    def test_the_verify_row_block_and_its_two_derived_views_are_safe_too(self):
        """The siblings the same function creates, checked as a family.

        ``_verify_rows`` and ``_verify_tokens`` are cloned; ``_verify_hidden``
        and ``_verify_last_token`` are views INTO those clones and inherit the
        guarantee. Pinned so a later edit cannot quietly re-point one of the
        derived views at the forward's own output.
        """
        h = _Harness(rung=2)
        h.round()
        rows = h.job["_verify_rows"]
        produced = h.graph.hidden[3]
        self.assertNotEqual(
            rows.untyped_storage().data_ptr(), produced.untyped_storage().data_ptr()
        )
        self.assertEqual(
            h.job["_verify_hidden"].untyped_storage().data_ptr(),
            rows.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            h.job["_verify_last_token"].untyped_storage().data_ptr(),
            h.job["_verify_tokens"].untyped_storage().data_ptr(),
        )

    def test_next_is_a_view_into_the_argmax_and_not_into_a_replayed_buffer(self):
        """``_next`` is the other row stored past its forward.

        It comes off ``argmax``, which allocates, so it is safe for a reason
        the code does not have to state -- but it is stored in the same dict
        and read in the same place a round later, so it is checked here rather
        than assumed.
        """
        h = _Harness()
        h.round()
        nxt = h.job["_next"]
        for buf in h.graph.logits.values():
            self.assertNotEqual(
                nxt.untyped_storage().data_ptr(), buf.untyped_storage().data_ptr()
            )


class TestWhatAWrittenThroughHiddenCosts(CustomTestCase):
    """The constraint the #399 window put on the hypothesis.

    A corrupted ``_hidden`` has to be shown to do something, and what it does
    decides whether it can be the carrier of a CONTENT divergence.
    """

    def setUp(self):
        for var in ("SGLANG_LANE_MARGIN_PROBE", "SGLANG_LANE_SPEC_DEBUG"):
            os.environ.pop(var, None)

    def _run(self, rounds, intervening_replay):
        h = _Harness()
        accepts = []
        for _ in range(rounds):
            _, n_accept = h.round(intervening_replay=intervening_replay)
            accepts.append(n_accept)
        return h, accepts

    def test_the_clean_chain_accepts_every_proposal(self):
        _, accepts = self._run(4, intervening_replay=False)
        self.assertEqual(accepts, [1, 1, 1, 1])

    def test_the_head_is_conditioned_on_the_row_the_verify_produced(self):
        """Every hidden row the head is handed, in order, over three rounds.

        Two per round on a fully accepted chain, and they come from different
        places: ``_propose`` hands it ``_hidden`` (the row stored past the
        previous verify -- 99, 101, 103), and ``_rollback_draft``'s re-seed
        hands it ``_verify_rows[0]`` of the round that just ran (100, 102,
        104), which is the cloned sibling. Pinning both in one list is what
        makes the asymmetry visible: they are consecutive rows of the same
        forward output, one cloned and one -- until this change -- not.
        """
        h, _ = self._run(3, intervening_replay=False)
        self.assertEqual(
            [float(t[0, 0]) for t in h.head_inputs],
            [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
        )

    def test_an_intervening_same_shape_replay_reaches_the_head_and_costs_accept(self):
        """With the clone in place this is a no-op; without it, it is the bug.

        Kept as an assertion on the CLEAN behaviour, because that is what the
        fix guarantees: the head still sees the verify's rows and the chain
        still accepts, no matter what replayed the shape in between.
        """
        h, accepts = self._run(4, intervening_replay=True)
        self.assertEqual(accepts, [1, 1, 1, 1])
        self.assertNotIn(777.0, [float(t[0, 0]) for t in h.head_inputs])

    def test_the_committed_trajectory_does_not_depend_on_the_hidden_row(self):
        """The desk argument of #399, executable.

        Deliberately NOT a can-fail test: it is an invariant that holds on
        both sides of the fix, and that is the finding. The target's verify
        forward never reads a hidden state, and every emitted token is one of
        its own predictions over the committed prefix, so a corrupted
        ``_hidden`` moves the accept length and the round count and leaves the
        trajectory alone. Run against the pre-fix assignment the two arms
        differ only in HOW FAR they got in four rounds -- eight tokens against
        five -- and agree on every token they share.
        """
        clean, _ = self._run(4, intervening_replay=False)
        dirty, _ = self._run(4, intervening_replay=True)
        shared = min(len(clean.job["output_ids"]), len(dirty.job["output_ids"]))
        self.assertGreaterEqual(shared, 4)
        self.assertEqual(
            clean.job["output_ids"][:shared], dirty.job["output_ids"][:shared]
        )
        self.assertEqual(
            clean.job["output_ids"], [101, 102, 103, 104, 105, 106, 107, 108]
        )

    def test_a_head_that_proposes_from_a_wrong_row_loses_the_accept_not_the_tokens(
        self,
    ):
        """The same statement, driven from the other end.

        Rather than corrupting the buffer, hand the head a row it cannot use.
        The chain then accepts nothing -- and commits the identical tokens,
        one per round instead of two.
        """
        clean = _Harness()
        for _ in range(6):
            clean.round()

        broken = _Harness()
        broken.job["_hidden"] = torch.full((1, HIDDEN), 777.0)
        original = broken.lane._propose

        def _propose_from_a_stale_row(job):
            job["_hidden"] = torch.full((1, HIDDEN), 777.0)
            return original(job)

        broken.lane._propose = _propose_from_a_stale_row
        accepts = []
        for _ in range(6):
            _, n_accept = broken.round()
            accepts.append(n_accept)

        self.assertEqual(accepts, [0, 0, 0, 0, 0, 0])
        self.assertEqual(
            broken.job["output_ids"],
            clean.job["output_ids"][: len(broken.job["output_ids"])],
        )


class TestThePlainDecodeStepKeepsTheCommittedLength(CustomTestCase):
    """The K = 0 rung, which routes a SPECULATIVE job through ``_decode_step``.

    ``_verify`` reads ``n_cached`` from ``job["_kv_len"]``. A plain step that
    advances the batch and leaves that field behind puts the next verify round
    one position too early: it overwrites the slot pointers of tokens already
    committed, places the chain at the wrong absolute positions and masks a
    prefix one token short. That is committed content, not a lost proposal.
    """

    def test_a_plain_step_advances_the_committed_length_with_the_batch(self):
        h = _Harness()
        lane = h.lane
        job = h.job
        lane._timed_forward = lambda batch: (
            torch.tensor([int(job["_next"][0].item()) + 1], dtype=torch.int64),
            1.0,
        )
        lane._last_margin = None
        job["decode_ms"] = []
        before = int(job["_kv_len"])
        lane._decode_step(job)
        self.assertEqual(int(job["_batch"].seq_lens[0].item()), before + 1)
        self.assertEqual(int(job["_kv_len"]), before + 1)

    def test_the_next_verify_round_starts_where_the_plain_step_stopped(self):
        h = _Harness()
        lane = h.lane
        job = h.job
        lane._timed_forward = lambda batch: (
            torch.tensor([int(job["_next"][0].item()) + 1], dtype=torch.int64),
            1.0,
        )
        lane._last_margin = None
        job["decode_ms"] = []
        lane._decode_step(job)
        committed_slot = int(job["_batch"].req_to_token_pool.req_to_token[0, 8].item())
        h.round()
        self.assertEqual(
            int(job["_batch"].req_to_token_pool.req_to_token[0, 8].item()),
            committed_slot,
            "the verify round wrote its candidate slots over a position the "
            "plain decode step had already committed",
        )


if __name__ == "__main__":
    unittest.main()
