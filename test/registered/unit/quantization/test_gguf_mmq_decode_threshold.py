"""Task #163: bucket-coupled MMVQ->MMQ decode threshold for GGUF.

WHAT IS UNDER TEST
------------------
`fused_mul_mat_gguf` picks between three paths. The first two are the pair this
change touches:

    M <= mmvq_safe               -> ggml_mul_mat_vec_a8   (MMVQ, matrix-vector)
    M <= _MMQ_MAX_TOKENS         -> ggml_mul_mat_a8       (MMQ, tiled GEMM)
    otherwise                    -> dequantize + cuBLAS   (OTHER precision class)

`_batched_mmvq_enabled()` pins `mmvq_safe = 8` and `_MMQ_MAX_TOKENS` is also 8,
so the MMQ branch is unreachable: every batch up to 8 tokens takes the
matrix-VECTOR kernel. Measured on the real per-rank Q8_0 decode shapes, 8
tokens is exactly where that is wrong on an RTX 5090 (MMQ 1.17-1.75x faster)
and still right on an RTX 3080 (MMQ loses). `_mmq_threshold_prefers_mmq` adds
the opt-in reroute.

These tests are CPU-only: they exercise the DECISION, whose inputs are the
device capability, the weight shape, the token count and the registered
CUDA-graph buckets. The device capability is substituted rather than probed
(`_device_cap` is patched), which is also what makes the sm120 / sm86 /
unmeasured cases testable on one machine.

THE THREE PROPERTIES THAT MUST HOLD
-----------------------------------
1. Flag OFF is byte-identical: the decision is False for every input, so the
   dispatch expression reduces to what it was before.
2. Bucket coupling: the decision is constant across a CUDA-graph decode
   bucket, and is the identity on a bucket itself. Without this a captured
   graph could replay a kernel other than the one it was captured with.
3. `dequant_cublas` is never reachable through this path at any tolerance --
   it does not quantize activations, i.e. a different PRECISION class.

Every property is also verified in its PRE-FIX RED form: the old semantics are
fed into the SAME assertion helpers and must fail them (see
`PreFixSemanticsAreRedTest`). An ImportError is not accepted as evidence.
"""

import unittest
from unittest import mock

import gguf as gguf_lib

from sglang.srt.layers.quantization import gguf as G
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

Q8_0 = int(gguf_lib.GGMLQuantizationType.Q8_0)
Q6_K = int(gguf_lib.GGMLQuantizationType.Q6_K)
# An I-matrix type: in DEQUANT/MMVQ types but NOT in MMQ_QUANT_TYPES, so no
# MMQ kernel exists for it and the threshold must never select one.
IQ4_XS = int(gguf_lib.GGMLQuantizationType.IQ4_XS)

SM120 = (12, 0)  # RTX 5090 -- measured: MMQ wins from bucket 8
SM86 = (8, 6)  # RTX 3080 -- measured: MMQ never wins inside the MMQ band
SM90 = (9, 0)  # not in the table -- unmeasured, must get no rerouting

# The real per-rank decode shapes the thresholds were measured on (N, K).
SHAPES = {
    "qkv_proj": (5120, 5120),  # square
    "o_proj": (5120, 5120),  # square
    "mlp_gate_up": (5120, 2688),  # tall
    "mlp_down": (2688, 5120),  # wide_k
}
# Typical decode-bucket ladder without speculative decoding.
BUCKETS = (1, 2, 4, 8)


class _FakeQWeight:
    """Just enough of a GGUF weight tensor for the decision: its shape.

    `_gguf_shape_class` reads `shape[0]` (N) and reconstructs K from
    `shape[1]` via the type's block/type size, exactly as the dispatch does.
    """

    def __init__(self, n: int, k: int, qtype: int = Q8_0):
        block_size, type_size = gguf_lib.GGML_QUANT_SIZES[qtype]
        assert k % block_size == 0, (k, block_size)
        self.shape = (n, k // block_size * type_size)


def _w(name: str, qtype: int = Q8_0) -> _FakeQWeight:
    n, k = SHAPES[name]
    return _FakeQWeight(n, k, qtype)


class _Env:
    """Threshold ON/OFF with the module caches reset around the block."""

    def __init__(self, enabled: bool, cap=SM120, buckets=BUCKETS):
        self.enabled, self.cap, self.buckets = enabled, cap, buckets

    def __enter__(self):
        self._stack = []
        G._reset_mmq_threshold_cache()
        p = mock.patch.dict(
            "os.environ", {G._MMQ_THRESHOLD_ENV: "1" if self.enabled else "0"}
        )
        p.start()
        self._stack.append(p)
        # _is_cuda is a module-level constant captured at import; force it so
        # the test result does not depend on whether the runner has a GPU.
        p = mock.patch.object(G, "_is_cuda", True)
        p.start()
        self._stack.append(p)
        p = mock.patch.object(G, "_device_cap", lambda: self.cap)
        p.start()
        self._stack.append(p)
        if self.buckets is not None:
            G.set_decode_token_buckets(self.buckets)
        return self

    def __exit__(self, *exc):
        for p in reversed(self._stack):
            p.stop()
        G._reset_mmq_threshold_cache()
        return False


# ---------------------------------------------------------------------------
# Assertion helpers. Each takes the DECISION FUNCTION as a parameter so the
# identical body can be run against the pre-fix semantics further down; that
# is what makes the red verification the strong form rather than "the import
# used to fail".
# ---------------------------------------------------------------------------


def assert_bucket_constant(tc, decide):
    """Property 2: one kernel decision per CUDA-graph decode bucket.

    Every raw token count that a graph would pad to bucket B must get B's
    decision, and B itself must map to B.
    """
    for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 8)):
        at_bucket = decide(hi, _w("qkv_proj"), Q8_0)
        for m in range(lo, hi + 1):
            tc.assertEqual(
                decide(m, _w("qkv_proj"), Q8_0),
                at_bucket,
                f"m={m} disagrees with its bucket {hi}: a replay of the "
                f"graph captured at {hi} would run a different kernel",
            )


def assert_mmq_selected_at_measured_threshold(tc, decide):
    """Property: on sm120 the measured crossover (bucket 8) is acted on."""
    for name in SHAPES:
        tc.assertTrue(
            decide(8, _w(name), Q8_0),
            f"{name}: MMQ measured 1.17-1.75x faster at bucket 8 on sm120, "
            f"but the decision keeps MMVQ",
        )


def assert_never_reaches_dequant_band(tc, decide):
    """Property 3: the reroute stays inside the MMVQ<->MMQ pair.

    Whenever the decision is True the dispatch must land in the MMQ branch,
    which requires the token count to be within `_MMQ_MAX_TOKENS`. A True
    above that band would mean rerouting into dequant+cuBLAS, a different
    precision class.
    """
    for m in (1, 2, 4, 8, 9, 12, 16, 32, 64):
        for name in SHAPES:
            if decide(m, _w(name), Q8_0):
                tc.assertLessEqual(
                    m,
                    G._MMQ_MAX_TOKENS,
                    f"{name} m={m}: decision True above _MMQ_MAX_TOKENS "
                    f"would route into dequant+cuBLAS (other precision class)",
                )


class BucketMathTest(CustomTestCase):
    def test_bucket_is_identity_on_a_registered_bucket(self):
        """A replay always runs at a bucket, so rounding must be a no-op there."""
        with _Env(enabled=True):
            for b in BUCKETS:
                self.assertEqual(G._decode_bucket_for(b), b)

    def test_rounds_up_to_the_enclosing_bucket(self):
        with _Env(enabled=True):
            self.assertEqual(G._decode_bucket_for(3), 4)
            self.assertEqual(G._decode_bucket_for(5), 8)
            self.assertEqual(G._decode_bucket_for(7), 8)

    def test_above_the_ladder_falls_back_to_the_raw_count(self):
        with _Env(enabled=True):
            self.assertEqual(G._decode_bucket_for(9), 9)
            self.assertEqual(G._decode_bucket_for(64), 64)

    def test_no_buckets_registered_is_the_identity(self):
        """CUDA graphs disabled -> threshold on the raw token count."""
        with _Env(enabled=True, buckets=None):
            for m in (1, 3, 5, 8, 13):
                self.assertEqual(G._decode_bucket_for(m), m)

    def test_registration_is_union_merging_and_idempotent(self):
        """Target and draft runners both register; every replayed count must
        remain a bucket, so the union is kept rather than the last writer."""
        with _Env(enabled=True, buckets=None):
            G.set_decode_token_buckets([1, 2, 4, 8])  # draft: 1 token per bs
            G.set_decode_token_buckets([4, 8, 16, 32])  # target: 4 tokens per bs
            G.set_decode_token_buckets([1, 2, 4, 8])  # again
            self.assertEqual(G._decode_token_buckets, (1, 2, 4, 8, 16, 32))
            for b in (1, 2, 4, 8, 16, 32):
                self.assertEqual(G._decode_bucket_for(b), b)


class ShapeClassTest(CustomTestCase):
    def test_real_decode_shapes_classify_as_measured(self):
        self.assertEqual(G._gguf_shape_class(_w("qkv_proj"), Q8_0), "square")
        self.assertEqual(G._gguf_shape_class(_w("o_proj"), Q8_0), "square")
        self.assertEqual(G._gguf_shape_class(_w("mlp_gate_up"), Q8_0), "tall")
        self.assertEqual(G._gguf_shape_class(_w("mlp_down"), Q8_0), "wide_k")

    def test_every_class_has_a_measured_entry_on_measured_devices(self):
        """A shape whose class is missing from a device's table would silently
        fall through to MMVQ; the table must cover all three classes."""
        for cap in (SM120, SM86):
            table = G._MMQ_BUCKET_MIN[cap]
            self.assertEqual(set(table), {"square", "tall", "wide_k"}, cap)


class FlagOffIsByteIdenticalTest(CustomTestCase):
    def test_decision_is_false_for_every_input(self):
        with _Env(enabled=False):
            for cap in (SM120, SM86, SM90):
                with mock.patch.object(G, "_device_cap", lambda c=cap: c):
                    for m in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16):
                        for name in SHAPES:
                            self.assertFalse(
                                G._mmq_threshold_prefers_mmq(m, _w(name), Q8_0),
                                f"flag OFF must not reroute: {cap} {name} m={m}",
                            )

    def test_flag_off_also_holds_for_kquants(self):
        with _Env(enabled=False):
            for m in (4, 8):
                self.assertFalse(
                    G._mmq_threshold_prefers_mmq(m, _w("mlp_down", Q6_K), Q6_K)
                )


class MeasuredThresholdTest(CustomTestCase):
    def test_sm120_takes_mmq_from_bucket_8(self):
        with _Env(enabled=True, cap=SM120):
            assert_mmq_selected_at_measured_threshold(self, G._mmq_threshold_prefers_mmq)

    def test_sm120_keeps_mmvq_below_the_measured_crossover(self):
        """MMVQ wins at every M<=7 on sm120 (gain 0.39-0.99), so no BUCKET
        below 8 may be rerouted.

        The property is stated on buckets, not on raw token counts, and that
        distinction is the whole design: with the ladder (1,2,4,8) a batch of
        5..7 real tokens replays the graph captured at bucket 8 and therefore
        MUST take bucket 8's kernel (asserted separately in
        BucketCouplingTest). Only 1..4 sit in a sub-8 bucket.
        """
        with _Env(enabled=True, cap=SM120):
            for m in (1, 2, 3, 4):
                self.assertLess(G._decode_bucket_for(m), 8)
                for name in SHAPES:
                    self.assertFalse(
                        G._mmq_threshold_prefers_mmq(m, _w(name), Q8_0),
                        f"{name} m={m} (bucket {G._decode_bucket_for(m)}) "
                        f"must stay on MMVQ below the crossover",
                    )

    def test_sm120_with_a_finer_ladder_keeps_mmvq_at_every_sub_8_bucket(self):
        """Same property where bucket == raw M, so nothing is hidden by
        rounding: with every count captured, 1..7 each have their own bucket
        and none of them may take MMQ."""
        with _Env(enabled=True, cap=SM120, buckets=tuple(range(1, 9))):
            for m in range(1, 8):
                self.assertEqual(G._decode_bucket_for(m), m)
                for name in SHAPES:
                    self.assertFalse(
                        G._mmq_threshold_prefers_mmq(m, _w(name), Q8_0),
                        f"{name} bucket={m} must stay on MMVQ",
                    )
            for name in SHAPES:
                self.assertTrue(G._mmq_threshold_prefers_mmq(8, _w(name), Q8_0))

    def test_sm86_never_reroutes_it_was_measured_and_mmq_loses(self):
        with _Env(enabled=True, cap=SM86):
            for m in (1, 2, 4, 7, 8):
                for name in SHAPES:
                    self.assertFalse(
                        G._mmq_threshold_prefers_mmq(m, _w(name), Q8_0),
                        f"sm86 {name} m={m}: MMQ measured SLOWER, must not fire",
                    )

    def test_unmeasured_device_gets_no_rerouting(self):
        """Never extrapolate a crossover onto a device nobody measured."""
        self.assertNotIn(SM90, G._MMQ_BUCKET_MIN)
        with _Env(enabled=True, cap=SM90):
            for m in (1, 4, 8):
                for name in SHAPES:
                    self.assertFalse(
                        G._mmq_threshold_prefers_mmq(m, _w(name), Q8_0)
                    )

    def test_type_without_an_mmq_kernel_is_never_rerouted(self):
        self.assertNotIn(IQ4_XS, G.MMQ_QUANT_TYPES)
        with _Env(enabled=True, cap=SM120):
            for m in (4, 8):
                self.assertFalse(
                    G._mmq_threshold_prefers_mmq(m, _w("qkv_proj", IQ4_XS), IQ4_XS)
                )


class BucketCouplingTest(CustomTestCase):
    def test_decision_is_constant_within_a_bucket(self):
        with _Env(enabled=True, cap=SM120):
            assert_bucket_constant(self, G._mmq_threshold_prefers_mmq)

    def test_a_partially_filled_bucket_8_gets_the_bucket_8_kernel(self):
        """The concrete graph-replay case: a batch of 5 real tokens padded to
        the captured bucket 8 must run MMQ, the kernel bucket 8 was captured
        with -- otherwise the replay is numerically inconsistent with capture."""
        with _Env(enabled=True, cap=SM120):
            self.assertTrue(G._mmq_threshold_prefers_mmq(5, _w("qkv_proj"), Q8_0))
            self.assertTrue(G._mmq_threshold_prefers_mmq(8, _w("qkv_proj"), Q8_0))


class DequantIsNeverATargetTest(CustomTestCase):
    def test_reroute_stays_inside_the_mmvq_mmq_pair(self):
        for cap in (SM120, SM86, SM90):
            with _Env(enabled=True, cap=cap):
                assert_never_reaches_dequant_band(self, G._mmq_threshold_prefers_mmq)

    def test_large_buckets_are_left_to_the_existing_dequant_branch(self):
        """Even where MMQ is measured faster than dequant+cuBLAS at M=12/16,
        this path must not claim that range: crossing the precision class has
        to be a declared decision, not a side effect of a speed threshold."""
        with _Env(enabled=True, cap=SM120, buckets=(1, 2, 4, 8, 16, 32)):
            for m in (9, 12, 16, 32):
                for name in SHAPES:
                    self.assertFalse(
                        G._mmq_threshold_prefers_mmq(m, _w(name), Q8_0),
                        f"{name} m={m} is above _MMQ_MAX_TOKENS="
                        f"{G._MMQ_MAX_TOKENS} and belongs to dequant+cuBLAS",
                    )


class EnableGateTest(CustomTestCase):
    def test_env_override_wins_over_server_args(self):
        fake_args = mock.Mock(gguf_mmq_decode_threshold=True)
        with mock.patch(
            "sglang.srt.runtime_context.get_server_args", return_value=fake_args
        ):
            G._reset_mmq_threshold_cache()
            with mock.patch.dict("os.environ", {G._MMQ_THRESHOLD_ENV: "0"}):
                self.assertFalse(G._mmq_decode_threshold_enabled())
            G._reset_mmq_threshold_cache()
            with mock.patch.dict("os.environ", {G._MMQ_THRESHOLD_ENV: "1"}):
                self.assertTrue(G._mmq_decode_threshold_enabled())
        G._reset_mmq_threshold_cache()

    def test_server_args_flag_is_honoured_when_env_is_unset(self):
        G._reset_mmq_threshold_cache()
        env = {k: v for k, v in __import__("os").environ.items()}
        env.pop(G._MMQ_THRESHOLD_ENV, None)
        fake_args = mock.Mock(gguf_mmq_decode_threshold=True)
        with mock.patch.dict("os.environ", env, clear=True):
            with mock.patch(
                "sglang.srt.runtime_context.get_server_args", return_value=fake_args
            ):
                self.assertTrue(G._mmq_decode_threshold_enabled())
        G._reset_mmq_threshold_cache()

    def test_a_pre_publish_call_does_not_latch_the_answer_off(self):
        """The first GGUF matmul can run before ModelRunner publishes
        ServerArgs. Answering False then is fine; CACHING it would silently
        disable the flag for the whole process."""
        G._reset_mmq_threshold_cache()
        env = {k: v for k, v in __import__("os").environ.items()}
        env.pop(G._MMQ_THRESHOLD_ENV, None)
        with mock.patch.dict("os.environ", env, clear=True):
            with mock.patch(
                "sglang.srt.runtime_context.get_server_args",
                side_effect=RuntimeError("not published yet"),
            ):
                self.assertFalse(G._mmq_decode_threshold_enabled())
            self.assertIsNone(G._mmq_threshold_cached)  # not latched
            fake_args = mock.Mock(gguf_mmq_decode_threshold=True)
            with mock.patch(
                "sglang.srt.runtime_context.get_server_args", return_value=fake_args
            ):
                self.assertTrue(G._mmq_decode_threshold_enabled())
        G._reset_mmq_threshold_cache()


class ServerArgsSurfaceTest(CustomTestCase):
    def test_flag_exists_defaults_off_and_documents_the_determinism_trade(self):
        """The trade-off must live where the flag is OFFERED, not only in a
        design note: anyone reading --help has to see what they give up."""
        import dataclasses
        from typing import get_type_hints

        from sglang.srt.server_args import ServerArgs

        hints = get_type_hints(ServerArgs, include_extras=True)
        field = {f.name: f for f in dataclasses.fields(ServerArgs)}[
            "gguf_mmq_decode_threshold"
        ]
        self.assertIs(field.default, False, "the flag must default to OFF")
        help_text = " ".join(
            m.help
            for m in getattr(hints["gguf_mmq_decode_threshold"], "__metadata__", ())
            if getattr(m, "help", None)
        ).lower()
        self.assertTrue(help_text, "the flag must carry help text")
        for token in ("bit-identical", "run-to-run", "bucket", "precision class"):
            self.assertIn(
                token,
                help_text,
                f"--gguf-mmq-decode-threshold help must state '{token}'",
            )


class PreFixSemanticsAreRedTest(CustomTestCase):
    """PRE-FIX RED, strong form.

    Not "the symbol did not exist" -- the OLD semantics are implemented here
    and pushed through the SAME assertion helpers the green tests use. If a
    helper cannot tell old from new, it is not testing the fix.
    """

    @staticmethod
    def _pre_fix_decide(m, qweight, qweight_type):
        """Dispatch before this change: MMVQ owns everything up to mmvq_safe,
        so there is no reroute at all."""
        return False

    @staticmethod
    def _raw_m_decide(m, qweight, qweight_type):
        """The obvious WRONG implementation: threshold on the raw token count
        instead of the CUDA-graph bucket. Fast, and it breaks graph replays."""
        if not G._mmq_decode_threshold_enabled():
            return False
        if qweight_type not in G.MMQ_QUANT_TYPES:
            return False
        if m > G._MMQ_MAX_TOKENS:
            return False
        table = G._MMQ_BUCKET_MIN.get(G._device_cap())
        if not table:
            return False
        min_m = table.get(G._gguf_shape_class(qweight, qweight_type))
        return min_m is not None and m >= min_m

    def test_pre_fix_dispatch_fails_the_threshold_assertion(self):
        with _Env(enabled=True, cap=SM120):
            with self.assertRaises(AssertionError):
                assert_mmq_selected_at_measured_threshold(self, self._pre_fix_decide)

    def test_raw_m_threshold_fails_the_bucket_constancy_assertion(self):
        """The red that matters for CUDA graphs: a raw-M threshold gives m=5
        and m=8 different kernels although both replay the SAME captured
        graph at bucket 8."""
        with _Env(enabled=True, cap=SM120):
            self.assertTrue(self._raw_m_decide(8, _w("qkv_proj"), Q8_0))
            self.assertFalse(self._raw_m_decide(5, _w("qkv_proj"), Q8_0))
            with self.assertRaises(AssertionError):
                assert_bucket_constant(self, self._raw_m_decide)

    def test_the_bucket_coupled_implementation_passes_that_same_assertion(self):
        with _Env(enabled=True, cap=SM120):
            assert_bucket_constant(self, G._mmq_threshold_prefers_mmq)

    def test_an_unbounded_threshold_fails_the_precision_class_assertion(self):
        """A threshold without the `bucket > _MMQ_MAX_TOKENS` guard claims the
        range that belongs to dequant+cuBLAS."""

        def unbounded(m, qweight, qweight_type):
            if not G._mmq_decode_threshold_enabled():
                return False
            table = G._MMQ_BUCKET_MIN.get(G._device_cap())
            if not table:
                return False
            min_m = table.get(G._gguf_shape_class(qweight, qweight_type))
            return min_m is not None and G._decode_bucket_for(m) >= min_m

        with _Env(enabled=True, cap=SM120, buckets=(1, 2, 4, 8, 16, 32)):
            with self.assertRaises(AssertionError):
                assert_never_reaches_dequant_band(self, unbounded)

    def test_pre_fix_dispatch_trivially_passes_the_flag_off_property(self):
        """Sanity on the red harness itself: the flag-OFF property must NOT
        separate old from new -- that is the whole point of the default path
        being byte-identical. A helper that failed here would be testing
        something other than the fix."""
        with _Env(enabled=False, cap=SM120):
            for m in (1, 4, 8):
                self.assertEqual(
                    self._pre_fix_decide(m, _w("qkv_proj"), Q8_0),
                    G._mmq_threshold_prefers_mmq(m, _w("qkv_proj"), Q8_0),
                )


if __name__ == "__main__":
    unittest.main()
