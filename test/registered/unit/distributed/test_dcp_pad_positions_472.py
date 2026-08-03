"""#472: graph-padded ``positions`` must never reach a DCP ownership decision.

Upstream sgl-project/sglang#33253 reports the pattern for the EVEN DCP owner
rule: the breakable / piecewise CUDA-graph attention wrapper narrows Q/K/V and
``out_cache_loc`` to the real token count but leaves ``forward_batch.positions``
at the padded bucket length, and the even rule derives KV ownership from those
positions.  This file falsifies the pattern against OUR tree, which carries two
owner rules side by side:

  * the WEIGHTED rule (#173, ours) derives ownership from ``out_cache_loc``
    -- the tensor the wrapper already narrows -- so it is structurally immune;
    the tests below are its positive pin, not a bug report.
  * the EVEN modulo rule (upstream, still reachable here whenever no non-uniform
    token vector is installed) derives ownership from ``positions``.  With
    ``out_cache_loc`` narrowed and ``positions`` padded the two lengths disagree,
    the length guard rejects the mask, and the fallback
    (``forward_batch.dcp_kv_mask``, populated on HIP only) is ``None`` on CUDA --
    so the write went out UNMASKED and every rank claimed every token.

Hermetic: pure CPU tensor math plus unbound calls into the two production
write paths with a recording pool.  No device, no collectives, no model.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.distributed.utils import get_cp_token_ratios, set_cp_token_ratios
from sglang.srt.layers.dcp.owner import (
    dcp_even_write_mask,
    dcp_weighted_owner_bounds,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REAL_TOKENS = 6
PADDED_TOKENS = 10
DCP_SIZE = 3


class _RecordingPool:
    """Stands in for the KV pool: records what the owner rule asked it to write."""

    def __init__(self):
        self.calls = []

    def set_kv_buffer(self, layer, loc, k, v, *args, **kwargs):
        self.calls.append(
            {
                "loc": loc.clone(),
                "dcp_kv_mask": (
                    None
                    if kwargs.get("dcp_kv_mask") is None
                    else kwargs["dcp_kv_mask"].clone()
                ),
            }
        )


def _padded_batch(dcp_kv_mask=None):
    """A graph-padded prefill batch: ZERO-padded positions and out_cache_loc.

    Mirrors ``cuda_graph_buffer_registry``'s slot policy for both tensors
    (``PaddingPolicy.ZERO``) and the wrapper's narrowing of ``out_cache_loc``
    alone -- i.e. the state the attention backend actually observes today.
    """
    positions = torch.zeros(PADDED_TOKENS, dtype=torch.int64)
    positions[:REAL_TOKENS] = torch.arange(100, 100 + REAL_TOKENS)
    out_cache_loc = torch.zeros(PADDED_TOKENS, dtype=torch.int64)
    out_cache_loc[:REAL_TOKENS] = torch.arange(100, 100 + REAL_TOKENS)
    return SimpleNamespace(
        positions=positions,
        # already narrowed by radix_attention's wrapper
        out_cache_loc=out_cache_loc[:REAL_TOKENS],
        dcp_kv_mask=dcp_kv_mask,
        num_token_non_padded_cpu=REAL_TOKENS,
    )


def _written_rows(call, num_rows):
    """Row indices the pool was actually told to write (mask ``None`` == all)."""
    if call["dcp_kv_mask"] is None:
        return set(range(num_rows))
    return {i for i in range(num_rows) if bool(call["dcp_kv_mask"][i])}


class TestDcpPadPositionsEvenLane(CustomTestCase):
    """The EVEN modulo lane -- the one upstream #33253 fixed."""

    def _drive_triton(self, forward_batch):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        pool = _RecordingPool()
        fake = SimpleNamespace(
            dcp_size=DCP_SIZE,
            dcp_rank=1,
            uneven_dcp=False,
            uneven_dcp_weighted=False,
            token_to_kv_pool=pool,
            _dcp_layer_token_sharded=lambda layer: True,
        )
        layer = SimpleNamespace(layer_id=0, k_scale=None, v_scale=None)
        k = torch.zeros(REAL_TOKENS, 1, 4)
        TritonAttnBackend._set_kv_buffer(fake, forward_batch, layer, None, k, k)
        return pool

    def _drive_flashinfer(self, forward_batch):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        pool = _RecordingPool()
        fake = SimpleNamespace(
            dcp_size=DCP_SIZE,
            dcp_rank=1,
            uneven_dcp_weighted=False,
            token_to_kv_pool=pool,
            _sess_spill=None,
            _sess_prefill_spill=None,
            _sess_verify_active=lambda: False,
            _wl_spill_active=False,
        )
        layer = SimpleNamespace(layer_id=0, k_scale=None, v_scale=None)
        k = torch.zeros(REAL_TOKENS, 1, 4)
        FlashInferAttnBackend._dcp_write_scatter(
            fake, layer, forward_batch, forward_batch.out_cache_loc, k, k
        )
        return pool

    def _assert_only_owned_rows(self, pool, forward_batch):
        self.assertEqual(len(pool.calls), 1)
        call = pool.calls[0]
        written = _written_rows(call, REAL_TOKENS)
        owned = {
            i
            for i in range(REAL_TOKENS)
            if int(forward_batch.positions[i]) % DCP_SIZE == 1
        }
        self.assertEqual(
            written,
            owned,
            "a graph-padded batch made this rank write rows it does not own -- "
            "an unmasked or wrongly-masked DCP write corrupts foreign KV slots",
        )

    def test_triton_even_lane_padded_positions_refused(self):
        """Padded positions + narrowed loc has NO owner mask -- refuse, never write."""
        with self.assertRaises(ValueError):
            self._drive_triton(_padded_batch())

    def test_flashinfer_even_lane_padded_positions_refused(self):
        with self.assertRaises(ValueError):
            self._drive_flashinfer(_padded_batch())

    def test_even_lane_after_wrapper_narrowing(self):
        """End-to-end pin: the PCG wrapper's narrowing feeds a correct mask."""
        from sglang.srt.layers.radix_attention import narrow_pcg_token_views

        for drive in (self._drive_triton, self._drive_flashinfer):
            with self.subTest(drive=drive.__name__):
                fb = _padded_batch()
                # undo the pre-narrowing the fixture bakes in, so the wrapper
                # sees exactly what the graph runner hands it
                fb.out_cache_loc = torch.zeros(PADDED_TOKENS, dtype=torch.int64)
                fb.out_cache_loc[:REAL_TOKENS] = torch.arange(100, 100 + REAL_TOKENS)
                narrow_pcg_token_views(fb, REAL_TOKENS)
                self._assert_only_owned_rows(drive(fb), fb)

    def test_even_lane_unpadded_batch_is_unchanged(self):
        """The non-graph path (positions and loc both full length) is untouched."""
        positions = torch.arange(100, 100 + REAL_TOKENS)
        fb = SimpleNamespace(
            positions=positions,
            out_cache_loc=positions.clone(),
            dcp_kv_mask=None,
            num_token_non_padded_cpu=REAL_TOKENS,
        )
        self._assert_only_owned_rows(self._drive_triton(fb), fb)
        self._assert_only_owned_rows(self._drive_flashinfer(fb), fb)

    def test_precomputed_mask_fallback_still_honoured(self):
        """HIP precomputes ``forward_batch.dcp_kv_mask``; that path must survive."""
        fb = _padded_batch()
        precomputed = fb.positions[:REAL_TOKENS] % DCP_SIZE == 1
        fb.positions = None
        fb.dcp_kv_mask = precomputed
        pool = self._drive_triton(fb)
        self.assertEqual(len(pool.calls), 1)
        torch.testing.assert_close(pool.calls[0]["dcp_kv_mask"], precomputed)


class TestDcpEvenWriteMaskHelper(CustomTestCase):
    """The shared even-lane mask helper refuses instead of writing unmasked."""

    def test_positions_of_matching_length(self):
        positions = torch.arange(100, 100 + REAL_TOKENS)
        mask = dcp_even_write_mask(positions, REAL_TOKENS, DCP_SIZE, 1, None)
        torch.testing.assert_close(mask, positions % DCP_SIZE == 1)

    def test_padded_positions_without_fallback_is_refused(self):
        positions = torch.zeros(PADDED_TOKENS, dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            dcp_even_write_mask(positions, REAL_TOKENS, DCP_SIZE, 1, None)
        self.assertIn("owner", str(ctx.exception).lower())

    def test_no_source_at_all_is_refused(self):
        with self.assertRaises(ValueError):
            dcp_even_write_mask(None, REAL_TOKENS, DCP_SIZE, 1, None)

    def test_fallback_of_wrong_length_is_refused(self):
        with self.assertRaises(ValueError):
            dcp_even_write_mask(
                None, REAL_TOKENS, DCP_SIZE, 1, torch.ones(PADDED_TOKENS, dtype=bool)
            )


class TestDcpPadPositionsWeightedLane(CustomTestCase):
    """The WEIGHTED lane (#173, ours): positive pin on its structural immunity."""

    PLAN = [2, 1, 1]

    def setUp(self):
        self._saved = get_cp_token_ratios()
        set_cp_token_ratios(self.PLAN)

    def tearDown(self):
        set_cp_token_ratios(self._saved)

    def _drive_triton_weighted(self, forward_batch, rank):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        cp_S, cp_lo, cp_hi, cp_ratio = dcp_weighted_owner_bounds(len(self.PLAN), rank)
        pool = _RecordingPool()
        fake = SimpleNamespace(
            dcp_size=len(self.PLAN),
            dcp_rank=rank,
            uneven_dcp=True,
            uneven_dcp_weighted=True,
            cp_S=cp_S,
            cp_lo=cp_lo,
            cp_hi=cp_hi,
            cp_ratio=cp_ratio,
            token_to_kv_pool=pool,
            _dcp_layer_token_sharded=lambda layer: True,
            _dcp_write_gather=lambda layer, k, v: (k, v),
        )
        layer = SimpleNamespace(layer_id=0, k_scale=None, v_scale=None)
        k = torch.zeros(REAL_TOKENS, 1, 4)
        TritonAttnBackend._set_kv_buffer(fake, forward_batch, layer, None, k, k)
        return pool, (cp_S, cp_lo, cp_hi, cp_ratio)

    def test_weighted_lane_ignores_padded_positions(self):
        """Padding ``positions`` arbitrarily must not move a single write."""
        for rank in range(len(self.PLAN)):
            with self.subTest(rank=rank):
                fb = _padded_batch()
                pool_a, bounds = self._drive_triton_weighted(fb, rank)

                poisoned = _padded_batch()
                # Loud, adversarial padded tail AND scrambled real positions:
                # the weighted rule must read none of it.
                poisoned.positions = torch.full((PADDED_TOKENS,), -7, dtype=torch.int64)
                pool_b, _ = self._drive_triton_weighted(poisoned, rank)

                torch.testing.assert_close(
                    pool_a.calls[0]["loc"], pool_b.calls[0]["loc"]
                )
                torch.testing.assert_close(
                    pool_a.calls[0]["dcp_kv_mask"], pool_b.calls[0]["dcp_kv_mask"]
                )

                cp_S, cp_lo, cp_hi, _ = bounds
                owned = {
                    i
                    for i in range(REAL_TOKENS)
                    if cp_lo <= int(fb.out_cache_loc[i]) % cp_S < cp_hi
                }
                self.assertEqual(_written_rows(pool_a.calls[0], REAL_TOKENS), owned)

    def test_weighted_lane_partitions_the_batch(self):
        """Across the group every real row is written exactly once."""
        fb = _padded_batch()
        seen = []
        for rank in range(len(self.PLAN)):
            pool, _ = self._drive_triton_weighted(_padded_batch(), rank)
            seen.append(_written_rows(pool.calls[0], REAL_TOKENS))
        flat = [i for s in seen for i in s]
        self.assertEqual(sorted(flat), list(range(REAL_TOKENS)))
        self.assertEqual(len(fb.out_cache_loc), REAL_TOKENS)


class TestPiecewiseWrapperNarrowsPositions(CustomTestCase):
    """The wrapper's token-axis narrowing: what the backend is handed under PCG.

    The custom ops themselves (``sglang::unified_attention_with_output`` and
    friends) are registered for the CUDA dispatch key only, so they cannot be
    driven on CPU. What IS hermetically drivable is the pair the ops delegate
    to; the ratchet below pins that every op under the piecewise context keeps
    delegating to it rather than re-inlining a partial narrowing.
    """

    def test_narrow_covers_positions_and_out_cache_loc(self):
        from sglang.srt.layers.radix_attention import (
            narrow_pcg_token_views,
            restore_pcg_token_views,
        )

        fb = _padded_batch()
        fb.out_cache_loc = torch.arange(PADDED_TOKENS)
        originals = narrow_pcg_token_views(fb, REAL_TOKENS)
        self.assertEqual(fb.positions.numel(), REAL_TOKENS)
        self.assertEqual(fb.out_cache_loc.numel(), REAL_TOKENS)

        restore_pcg_token_views(fb, originals)
        self.assertEqual(fb.positions.numel(), PADDED_TOKENS)
        self.assertEqual(fb.out_cache_loc.numel(), PADDED_TOKENS)

    def test_narrow_tolerates_absent_positions(self):
        from sglang.srt.layers.radix_attention import (
            narrow_pcg_token_views,
            restore_pcg_token_views,
        )

        fb = _padded_batch()
        fb.positions = None
        originals = narrow_pcg_token_views(fb, REAL_TOKENS)
        self.assertIsNone(fb.positions)
        restore_pcg_token_views(fb, originals)
        self.assertIsNone(fb.positions)

    def test_every_piecewise_op_delegates_the_narrowing(self):
        """Ratchet: no op may hand-roll a narrowing that forgets ``positions``."""
        import ast
        import inspect

        from sglang.srt.layers import radix_attention, radix_linear_attention

        owners = {"narrow_pcg_token_views", "restore_pcg_token_views"}
        for module in (radix_attention, radix_linear_attention):
            src = inspect.getsource(module)
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef) or node.name in owners:
                    continue
                body = ast.get_source_segment(src, node) or ""
                self.assertNotIn(
                    "forward_batch.out_cache_loc =",
                    body,
                    f"{module.__name__}.{node.name} assigns out_cache_loc "
                    "directly; use narrow_pcg_token_views / "
                    "restore_pcg_token_views so positions is narrowed with "
                    "it (#472)",
                )
            self.assertEqual(
                src.count("narrow_pcg_token_views("),
                src.count("restore_pcg_token_views("),
                f"{module.__name__}: unbalanced narrow/restore",
            )


if __name__ == "__main__":
    unittest.main()
