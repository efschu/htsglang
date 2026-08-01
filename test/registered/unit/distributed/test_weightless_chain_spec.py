"""#143: chain speculation (topk == 1) on the weightless-KV fast lane, in its
pure parts.

The lane runs an ASYMMETRIC forward: the head rank holds every layer weight, the
other ranks hold only a KV token-shard and mirror the per-attention-layer DCP
dispatch with an empty [T,0,D] head slice. Its #1 failure mode is a
collective-sequence divergence between the two sides, which is a SILENT NCCL
HANG rather than an error.

Speculation adds the one input that could make the sequence data-dependent: an
accept length. This file pins that it does not, plus the three decisions that
made the composition possible:

1. THE LOCKSTEP INVARIANT. The number, order and op-tags of DCP collectives in a
   step are a function of (forward_mode, has_prefix, layer count) only --
   `accept_len` is not an input. Tested as a trace comparison between the state
   reached after accepting 0 drafts and after accepting all k.
2. VERIFY EQUALS DECODE in that schedule. This is what makes a symmetric
   head+worker verify CUDA-graph capture legal on a communicator that pairs
   collectives by ISSUE ORDER.
3. FORWARD-MODE FIRST for the prefix gate. A target-verify batch carries no
   extend_prefix_lens, so any length- or slot-count-based test falls through to
   a rank-local quantity -- the D5 defect family
   ([[rank-lokaler-test-vor-kollektiv]]), whose third door this would be.

CPU only: no device, no process group, no model. The op-tags are pinned against
`layers/dcp/comm.py`'s source so a rename there fails here instead of drifting
into a runtime hang.
"""

import ast
import pathlib
import types
import unittest

from sglang.srt.layers.dcp.lockstep import (
    AG_HEADS_TAG_PREFIX,
    LSE_MERGE_TAG,
    chain_spec_verify_rows,
    spec_accept_broadcast_shapes,
    weightless_has_prefix,
    weightless_layer_op_tags,
    weightless_step_op_tags,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_SRT = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"
_COMM = _SRT / "layers" / "dcp" / "comm.py"
_FLASHINFER = _SRT / "layers" / "attention" / "flashinfer_backend.py"
_MODEL_RUNNER = _SRT / "model_executor" / "model_runner.py"
_GRAPH_RUNNER = _SRT / "model_executor" / "runner" / "decode_cuda_graph_runner.py"
_TP_WORKER = _SRT / "managers" / "tp_worker.py"

# The lane's geometry on the reference rig: head rank owns every q/kv head, the
# weightless workers own none -- the [H, 0, 0] head vector.
_Q_HEADS = 24
_KV_HEADS = 4
_LAYERS = 16


def _step_after_accept(accept_len: int, num_draft_tokens: int):
    """The op-tag trace of the generation step that FOLLOWS a verify which
    accepted ``accept_len`` tokens.

    The point of routing through this helper is that it takes ``accept_len`` at
    all: under chain spec the next step is again a TARGET_VERIFY over
    bs*(k+1) rows with a forced prefix, whatever was accepted. If some future
    change made the schedule accept-dependent, it would have to enter here.
    """
    assert 1 <= accept_len <= num_draft_tokens
    return weightless_step_op_tags(
        is_idle=False,
        is_decode=False,
        has_prefix=weightless_has_prefix(
            forces_prefix=True, extend_prefix_lens_cpu=None
        ),
        num_full_attention_layers=_LAYERS,
        kv_head_total=_KV_HEADS,
        q_head_total=_Q_HEADS,
    )


class TestLockstepInvariance(CustomTestCase):
    """(1) The collective schedule does not depend on the accept length."""

    def test_trace_identical_for_accept_zero_and_accept_k(self):
        for num_draft_tokens in (2, 4, 5, 8):
            # accept_len == 1 is "zero drafts accepted, bonus token only";
            # accept_len == num_draft_tokens is "every draft accepted".
            lo = _step_after_accept(1, num_draft_tokens)
            hi = _step_after_accept(num_draft_tokens, num_draft_tokens)
            self.assertEqual(lo, hi, f"k+1={num_draft_tokens}")
            # And every value in between.
            for a in range(1, num_draft_tokens + 1):
                self.assertEqual(_step_after_accept(a, num_draft_tokens), lo)

    def test_trace_length_is_layers_times_four_ops(self):
        trace = _step_after_accept(1, 4)
        # 3 guarded steps per layer (fused KV all-gather, Q all-gather, LSE
        # merge); the LSE merge is ONE guarded step carrying TWO NCCL ops.
        self.assertEqual(len(trace), 3 * _LAYERS)

    def test_verify_rows_are_accept_independent(self):
        for bs in (1, 2, 7):
            for k1 in (2, 4, 8):
                self.assertEqual(chain_spec_verify_rows(bs, k1), bs * k1)
        with self.assertRaises(ValueError):
            chain_spec_verify_rows(1, 0)

    def test_idle_step_issues_nothing(self):
        # An IDLE batch: the head's decoder layers skip attention, so the
        # worker must issue no collective either. Rank-uniform predicate.
        self.assertEqual(
            weightless_step_op_tags(
                is_idle=True,
                is_decode=True,
                has_prefix=True,
                num_full_attention_layers=_LAYERS,
                kv_head_total=_KV_HEADS,
                q_head_total=_Q_HEADS,
            ),
            (),
        )


class TestVerifyEqualsDecode(CustomTestCase):
    """(2) Verify and decode issue the same schedule -- the capture premise."""

    def test_layer_tuples_match(self):
        decode = weightless_layer_op_tags(
            is_decode=True,
            has_prefix=True,
            kv_head_total=_KV_HEADS,
            q_head_total=_Q_HEADS,
        )
        verify = weightless_layer_op_tags(
            is_decode=False,
            has_prefix=weightless_has_prefix(True, None),
            kv_head_total=_KV_HEADS,
            q_head_total=_Q_HEADS,
        )
        self.assertEqual(decode, verify)
        self.assertEqual(
            decode,
            (
                f"{AG_HEADS_TAG_PREFIX}{_KV_HEADS}",
                f"{AG_HEADS_TAG_PREFIX}{_Q_HEADS}",
                LSE_MERGE_TAG,
            ),
        )

    def test_first_chunk_extend_is_the_one_short_schedule(self):
        # A prefix-free extend (first chunked-prefill chunk) issues the KV
        # all-gather only. That is the ONE data-dependent branch on the lane,
        # and it is driven by a replicated length vector, not by a rank-local
        # slot count.
        short = weightless_layer_op_tags(
            is_decode=False,
            has_prefix=False,
            kv_head_total=_KV_HEADS,
            q_head_total=_Q_HEADS,
        )
        self.assertEqual(short, (f"{AG_HEADS_TAG_PREFIX}{_KV_HEADS}",))


class TestPrefixGateIsForwardModeFirst(CustomTestCase):
    """(3) The D5 class: verify answers True before any length is consulted."""

    def test_verify_true_without_any_prefix_lengths(self):
        self.assertTrue(weightless_has_prefix(True, None))
        self.assertTrue(weightless_has_prefix(True, []))
        self.assertTrue(weightless_has_prefix(True, [0, 0, 0]))

    def test_extend_follows_the_replicated_length_vector(self):
        self.assertFalse(weightless_has_prefix(False, None))
        self.assertFalse(weightless_has_prefix(False, []))
        self.assertFalse(weightless_has_prefix(False, [0, 0]))
        self.assertTrue(weightless_has_prefix(False, [0, 5]))
        self.assertTrue(weightless_has_prefix(False, [12]))

    def test_both_flashinfer_call_sites_go_through_the_helper(self):
        """One expression, two callers.

        The head (`_forward_extend_dcp`) and the weightless worker
        (`forward_extend_weightless_worker`) used to carry verbatim copies of
        this rule. Two copies of a rule whose two answers must agree is the
        drift surface; pin that neither re-derives it.
        """
        src = _FLASHINFER.read_text()
        self.assertIn("from sglang.srt.layers.dcp.lockstep import", src)
        self.assertEqual(
            src.count("weightless_has_prefix("),
            2,  # exactly the two call sites (the import has no parenthesis)
            "flashinfer_backend must reach the prefix rule through "
            "layers/dcp/lockstep.weightless_has_prefix, not re-derive it",
        )
        for fn in ("_forward_extend_dcp", "forward_extend_weightless_worker"):
            body = _function_source(src, fn)
            self.assertIn("weightless_has_prefix(", body, fn)
            # ... and does not keep a second, local derivation alongside it.
            self.assertNotIn("any(prefix_cpu)", body, fn)


class TestAcceptBroadcastGeometry(CustomTestCase):
    """The receive-only accept path allocates what the head broadcasts."""

    def test_shapes_match_eagle_sample_allocations(self):
        bs, num_draft_tokens, max_tree_depth = 3, 4, 4
        predict, accept_index, num_correct = spec_accept_broadcast_shapes(
            bs, num_draft_tokens, max_tree_depth
        )
        # eagle_sample: predict is flat over all query rows, accept_index is
        # [bs, max_tree_depth], num_correct_drafts is [bs].
        self.assertEqual(predict, (bs * num_draft_tokens,))
        self.assertEqual(accept_index, (bs, max_tree_depth))
        self.assertEqual(num_correct, (bs,))

    def test_shapes_are_boot_fixed_not_accept_dependent(self):
        a = spec_accept_broadcast_shapes(2, 4, 4)
        b = spec_accept_broadcast_shapes(2, 4, 4)
        self.assertEqual(a, b)

    def test_rejects_degenerate_geometry(self):
        with self.assertRaises(ValueError):
            spec_accept_broadcast_shapes(2, 0, 4)
        with self.assertRaises(ValueError):
            spec_accept_broadcast_shapes(2, 4, 0)


class TestSourcePins(CustomTestCase):
    """Pins against the real code, so a rename fails here not at runtime."""

    def test_op_tags_match_comm_py(self):
        src = _COMM.read_text()
        self.assertIn(f'guard_dcp_step(f"{AG_HEADS_TAG_PREFIX}', src)
        self.assertIn(f'guard_dcp_step("{LSE_MERGE_TAG}"', src)

    def test_guard_rule_covers_target_verify(self):
        """graphs-enabled => DCP guard OFF on both sides, for every mode the
        lane can enter a captured region in.

        A gloo handshake is not capturable, so a mode that replays a graph while
        the guard is on would put one rank in a barrier the other baked out.
        TARGET_VERIFY is such a mode under chain spec (it IS the generation
        step), so the rule must name it.
        """
        body = _function_source(_MODEL_RUNNER.read_text(), "_forward_raw")
        self.assertIn("is_target_verify()", body)
        self.assertIn("is_decode_or_idle()", body)

    def test_worker_graph_body_follows_capture_mode(self):
        """The worker must record the dispatch its capture mode implies.

        Recording the DECODE dispatch for a TARGET_VERIFY capture reads the
        wrong flashinfer wrapper family (decode_wrappers vs prefill_wrappers) --
        a wrong-answer bug that still emits the right collective count, i.e. one
        that shows up as a collapsed accept rate rather than a hang.
        """
        body = _function_source(
            _GRAPH_RUNNER.read_text(), "_capture_one_shape_weightless"
        )
        self.assertIn("forward_extend_weightless_worker", body)
        self.assertIn("forward_decode_weightless_worker", body)
        self.assertIn("capture_forward_mode", body)

    def test_verify_returns_before_the_lane_token_broadcast(self):
        """tp_worker ordering: the is_verify early return must precede the
        weightless gloo recv.

        The head skips sampling on a verify step and never reaches the matching
        send, so a worker that took the recv branch would block in gloo forever.
        Under spec the tokens come from the accept broadcast instead.
        """
        body = _function_source(
            _TP_WORKER.read_text(),
            "forward_batch_generation",
            class_name="TpModelWorker",
        )
        i_verify = body.index("if is_verify:")
        i_recv = body.index('"is_weightless_worker", False')
        self.assertLess(
            i_verify,
            i_recv,
            "the is_verify early return must come BEFORE the weightless-worker "
            "gloo receive, or every verify step deadlocks the lane",
        )


class TestWeightlessWorkerPredicateSurvivesTheSpecWorker(CustomTestCase):
    """The scheduler-side weightless-worker guard must resolve through a spec worker.

    `SchedulerBatchResultProcessor._is_weightless_worker` decides whether this
    rank skips every logprob / hidden-state dereference, because a weightless
    worker has no lm_head and therefore `result.logits_output is None`.

    Without speculation `self.model_worker` is a TpModelWorker and carries
    `.model_runner` directly. With speculation it is a BaseSpecWorker, which has
    no `model_runner` attribute at all -- the target runner sits behind its
    `target_worker` property. A plain `getattr(model_worker, "model_runner")`
    therefore returns None on EXACTLY the configuration #143 adds, the predicate
    degrades to False, and the worker rank walks into
    `logits_output.next_token_logprobs` on a None the moment a request asks for
    logprobs.

    That is not a hang and not a wrong number: it is an AttributeError that
    takes the rank down and SIGQUITs the server, and it only fires under
    `return_logprob`, so a plain generate smoke test misses it entirely. Found
    on hardware (Llama-3.1-8B TP=2, lane + EAGLE3 solo) at the first probe
    request; pinned here so the guard cannot silently lose the spec path again.
    """

    @staticmethod
    def _predicate(model_worker):
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )

        stub = types.SimpleNamespace(model_worker=model_worker)
        return SchedulerBatchResultProcessor._is_weightless_worker(stub)

    def test_plain_tp_worker_weightless_is_true(self):
        runner = types.SimpleNamespace(is_weightless_worker=True)
        worker = types.SimpleNamespace(model_runner=runner)
        self.assertTrue(self._predicate(worker))

    def test_plain_tp_worker_head_is_false(self):
        runner = types.SimpleNamespace(is_weightless_worker=False)
        worker = types.SimpleNamespace(model_runner=runner)
        self.assertFalse(self._predicate(worker))

    def test_spec_worker_resolves_through_target_worker(self):
        # The regression: no `model_runner` on the spec worker itself.
        runner = types.SimpleNamespace(is_weightless_worker=True)
        spec_worker = types.SimpleNamespace(
            target_worker=types.SimpleNamespace(model_runner=runner)
        )
        self.assertFalse(hasattr(spec_worker, "model_runner"))
        self.assertTrue(
            self._predicate(spec_worker),
            "the guard must see the weightless role through a spec worker, or "
            "the worker rank dereferences logits_output=None under "
            "return_logprob",
        )

    def test_spec_worker_on_the_head_rank_is_false(self):
        runner = types.SimpleNamespace(is_weightless_worker=False)
        spec_worker = types.SimpleNamespace(
            target_worker=types.SimpleNamespace(model_runner=runner)
        )
        self.assertFalse(self._predicate(spec_worker))

    def test_default_path_worker_without_either_attribute_is_false(self):
        # No lane, no spec: the predicate must not raise and must stay False.
        self.assertFalse(self._predicate(types.SimpleNamespace()))
        self.assertFalse(self._predicate(types.SimpleNamespace(target_worker=None)))

    def test_base_spec_worker_really_has_no_model_runner(self):
        """The premise of the fix, pinned against the class itself.

        If BaseSpecWorker ever grows a `model_runner` property the fallback
        becomes dead code -- and, worse, a property returning the DRAFT runner
        would flip the predicate to False again, since draft runners are
        deliberately never weightless.
        """
        from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

        self.assertFalse(
            hasattr(BaseSpecWorker, "model_runner"),
            "BaseSpecWorker grew a model_runner attribute; re-check "
            "SchedulerBatchResultProcessor._is_weightless_worker resolves the "
            "TARGET runner, not the draft one",
        )
        self.assertTrue(hasattr(BaseSpecWorker, "target_worker"))


def _function_source(module_src: str, func_name: str, class_name: str = None) -> str:
    """Source of the def named ``func_name``, optionally inside ``class_name``.

    ``class_name`` matters wherever a name is reused across classes (base +
    concrete worker), because pinning the wrong one silently passes.
    """
    tree = ast.parse(module_src)
    lines = module_src.splitlines(keepends=True)
    scopes = [tree]
    if class_name is not None:
        scopes = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == class_name
        ]
        if not scopes:
            raise AssertionError(f"class {class_name!r} not found")
    for scope in scopes:
        for node in ast.walk(scope):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name == func_name
            ):
                return "".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"function {func_name!r} not found")


if __name__ == "__main__":
    unittest.main()
