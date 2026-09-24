"""P prefill graph (27B line, --p-prefill-graph): the runtime half, on CPU.

The full prefill CUDA graph on group P (PP3, Qwen3.8-27B GDN hybrid, DFlash
draft-KV form) needed five things the tree did not have:

1. **GDN plain-EXTEND graph metadata.** ``MambaAttnBackendBase._replay_metadata``
   serves decode / target-verify only and RAISES for EXTEND (upstream too).
   ``_extend_graph_metadata`` refreshes a static ``query_start_loc`` and state
   index buffer in place; sentinel rows get PAD_SLOT_ID.
2. **FLA chunk tables pinned to the static cu_seqlens.** Without the pin,
   FLA's 4-entry ``tensor_cache`` rotates the capture-time table out and the
   address the captured graph reads is freed (pinned here: the eviction is
   real, the pin survives it). Kernel-level proof of the baked grid:
   test_p_prefill_graph_gdn_baked_grid_0924.py.
3. **PP stage input** (upstream #35451 plus the fork's DFlash aux carry):
   static proxy buffers, registry slots with ZERO padding, the key set derived
   from the stage's start layer and the capture layers, a loud refusal of a
   drifted key set, and the tuple-aware output slice.
4. **Replay eligibility** of the body-only full backend: captured >= requested
   hidden mode (P's batches carry NULL on every stage but the producer's),
   only a plain EXTEND, never multimodal inputs, never a body captured on
   mrope positions without them; every refusal NAMED by a rate-limited line.
5. **Global layer ids** under PP for the captured attention ops.
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.fla import index as fidx
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
    is_plain_extend_graph_mode,
)
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_prefill_registry,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.model_runner import align_pipeline_layers
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    PREFILL_GRAPH_POOL_ENV,
    prefill_transient_mib_for_rank,
)
from sglang.srt.model_executor.runner import prefill_cuda_graph_runner as pcgr
from sglang.srt.model_executor.runner_utils.buffers import PrefillInputBuffers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# 1. GDN plain-EXTEND graph metadata
# --------------------------------------------------------------------------
class _Pool:
    mapping = torch.tensor([7, 3, 5, 9], dtype=torch.int32)

    def get_mamba_indices(self, rpi):
        return self.mapping[rpi.long()]

    def translate_mamba_indices(self, x):
        return x


def _gdn_backend():
    be = object.__new__(MambaAttnBackendBase)
    be.device = CPU
    be.pad_slot_id = -1
    be.replayssm_write_pos_list = None
    be._extend_graph_static = {}
    be.req_to_token_pool = _Pool()
    return be


def _slot_view(lens, rpi, *, mode=ForwardMode.EXTEND, track=None):
    """The runner's slot-padded view: sentinels are zero-length rows whose
    start is the real token count (load_batch / _prepare_forward_metadata_
    for_replay)."""
    lens_t = torch.tensor(lens, dtype=torch.int64)
    real = sum(lens)
    starts, acc = [], 0
    for n in lens:
        starts.append(acc if n > 0 else real)
        acc += n
    return types.SimpleNamespace(
        batch_size=len(lens),
        forward_mode=mode,
        extend_start_loc=torch.tensor(starts, dtype=torch.int64),
        extend_seq_lens=lens_t,
        req_pool_indices=torch.tensor(rpi, dtype=torch.int64),
        seq_lens_cpu=lens_t.clone(),
        mamba_track_mask=track,
        spec_info=None,
    )


class TestGdnExtendGraphMetadata(CustomTestCase):
    def test_only_plain_extend_takes_the_graph_path(self):
        self.assertTrue(is_plain_extend_graph_mode(ForwardMode.EXTEND))
        for mode in (
            ForwardMode.MIXED,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DECODE,
            ForwardMode.DRAFT_EXTEND_V2,
            ForwardMode.SPLIT_PREFILL,
        ):
            self.assertFalse(is_plain_extend_graph_mode(mode), mode)

    def test_capture_then_a_short_replay_refresh_the_same_buffers(self):
        be = _gdn_backend()
        be.init_forward_metadata_out_graph(_slot_view([512], [2]), in_capture=True)
        qsl = be.forward_metadata.query_start_loc
        idx = be.forward_metadata.mamba_cache_indices
        self.assertEqual(qsl.tolist(), [0, 512])
        self.assertEqual(idx.tolist(), [5])
        self.assertEqual(qsl.dtype, torch.int32)
        # the capture pins the static tensor for FLA's chunk tables
        self.assertIn(id(qsl), fidx._GRAPH_STATIC_CU_SEQLENS)

        be.init_forward_metadata_out_graph(_slot_view([300], [1]), in_capture=False)
        # SAME objects (the captured kernels hold their addresses) ...
        self.assertIs(be.forward_metadata.query_start_loc, qsl)
        self.assertIs(be.forward_metadata.mamba_cache_indices, idx)
        # ... refreshed in place with the live values
        self.assertEqual(qsl.tolist(), [0, 300])
        self.assertEqual(idx.tolist(), [3])

    def test_sentinel_rows_are_zero_length_and_pad_slot(self):
        be = _gdn_backend()
        # a previous replay armed BOTH rows with live slots ...
        be.init_forward_metadata_out_graph(_slot_view([300, 200], [1, 2]))
        self.assertEqual(be.forward_metadata.mamba_cache_indices.tolist(), [3, 5])
        # ... so the sentinel row of the next one must be re-poisoned, not
        # left holding slot 5 (a zero-length lane still reads and writes back
        # its state slot in the h-kernel unless it carries PAD_SLOT_ID).
        be.init_forward_metadata_out_graph(_slot_view([300, 0], [1, 0]))
        self.assertEqual(be.forward_metadata.query_start_loc.tolist(), [0, 300, 300])
        # req_pool_indices 0 maps to slot 7 -- it must NOT be read-then-written
        self.assertEqual(be.forward_metadata.mamba_cache_indices.tolist(), [3, -1])

    def test_track_and_replayssm_are_refused_by_name(self):
        be = _gdn_backend()
        with self.assertRaises(NotImplementedError):
            be.init_forward_metadata_out_graph(
                _slot_view([8], [1], track=torch.ones(1, dtype=torch.bool))
            )
        be.replayssm_write_pos_list = []
        with self.assertRaises(NotImplementedError):
            be.init_forward_metadata_out_graph(_slot_view([8], [1]))


# --------------------------------------------------------------------------
# 2. FLA chunk tables: tensor_cache evicts, the pin does not
# --------------------------------------------------------------------------
class TestFlaGraphStaticPin(CustomTestCase):
    def _churn(self):
        for i in range(6):
            fidx.prepare_chunk_indices(
                torch.tensor([0, 64 * (i + 1)], dtype=torch.int32), 64
            )
            fidx.prepare_chunk_offsets(
                torch.tensor([0, 64 * (i + 1)], dtype=torch.int32), 64
            )

    def test_unpinned_tables_are_evicted_by_later_cu_seqlens(self):
        """The hazard the pin exists for: the capture-time object is gone."""
        cu = torch.tensor([0, 192], dtype=torch.int32)
        first = fidx.prepare_chunk_indices(cu, 64)
        self._churn()
        self.assertIsNot(fidx.prepare_chunk_indices(cu, 64), first)

    def test_pinned_tables_survive_and_keep_the_capture_content(self):
        cu = torch.tensor([0, 192], dtype=torch.int32)
        fidx.pin_graph_static_cu_seqlens(cu)
        ci = fidx.prepare_chunk_indices(cu, 64)
        co = fidx.prepare_chunk_offsets(cu, 64)
        self.assertEqual(ci.tolist(), [[0, 0], [0, 1], [0, 2]])
        self.assertEqual(co.tolist(), [0, 3])
        cu[1] = 70  # a replay refreshes the content in place
        self._churn()
        self.assertIs(fidx.prepare_chunk_indices(cu, 64), ci)
        self.assertIs(fidx.prepare_chunk_offsets(cu, 64), co)
        self.assertEqual(ci.tolist(), [[0, 0], [0, 1], [0, 2]])
        self.assertIn(("chunk_indices", 64), fidx.graph_static_tables(cu))

    def test_pin_is_idempotent_and_identity_keyed(self):
        cu = torch.tensor([0, 64], dtype=torch.int32)
        fidx.pin_graph_static_cu_seqlens(cu)
        ci = fidx.prepare_chunk_indices(cu, 64)
        fidx.pin_graph_static_cu_seqlens(cu)  # no reset
        self.assertIs(fidx.prepare_chunk_indices(cu, 64), ci)
        twin = torch.tensor([0, 64], dtype=torch.int32)  # equal, not the same
        self.assertEqual(fidx.graph_static_tables(twin), {})


# --------------------------------------------------------------------------
# 3. PP stage input: buffers, registry slots, keys, drift refusal, slicing
# --------------------------------------------------------------------------
KEYS = ("hidden_states", "residual", "aux_layer_6")
H = 8
MAX_TOKENS = 16


def _buffers(keys=KEYS):
    return PrefillInputBuffers.create(
        device=CPU,
        max_bs=2,
        max_num_tokens=MAX_TOKENS,
        cache_loc_dtype=torch.int64,
        is_multimodal=False,
        hidden_size=H,
        dtype=torch.float32,
        enable_mamba_track=False,
        pp_proxy_keys=keys,
    )


def _registry(bufs):
    return build_prefill_registry(
        device=CPU,
        max_bs=2,
        max_num_token=MAX_TOKENS,
        cache_loc_dtype=torch.int64,
        share_pool=False,
        source=bufs,
    )


def _extend_batch(n, bs=1):
    per = n // bs
    ext = torch.full((bs,), per, dtype=torch.int64)
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        input_ids=torch.arange(n, dtype=torch.int64),
        req_pool_indices=torch.arange(bs, dtype=torch.int64) + 1,
        seq_lens=ext.clone(),
        out_cache_loc=torch.arange(n, dtype=torch.int64) + 100,
        seq_lens_sum=n,
        orig_seq_lens=ext.clone(),
        seq_lens_cpu=ext.clone(),
        positions=torch.arange(n, dtype=torch.int64),
        extend_num_tokens=n,
        extend_seq_lens=ext.clone(),
        extend_prefix_lens=torch.zeros((bs,), dtype=torch.int64),
        extend_start_loc=torch.arange(bs, dtype=torch.int64) * per,
        extend_prefix_lens_cpu=[0] * bs,
        extend_seq_lens_cpu=[per] * bs,
        extend_logprob_start_lens_cpu=[per] * bs,
        global_forward_mode=ForwardMode.EXTEND,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def _live_proxy(n, keys=KEYS, value=1.0):
    return PPProxyTensors({k: torch.full((n, H), value + i) for i, k in enumerate(keys)})


class TestPpStageInput(CustomTestCase):
    def test_buffers_carry_one_tokens_by_hidden_tensor_per_key(self):
        bufs = _buffers()
        self.assertEqual(sorted(bufs.pp_proxy_tensors), sorted(KEYS))
        for t in bufs.pp_proxy_tensors.values():
            self.assertEqual(tuple(t.shape), (MAX_TOKENS, H))
        self.assertIsNone(_buffers(keys=None).pp_proxy_tensors)

    def test_registry_copies_the_head_and_zeroes_the_padded_tail(self):
        bufs = _buffers()
        reg = _registry(bufs)
        for key in KEYS:
            self.assertTrue(reg.has_slot(f"pp_proxy_tensors.{key}"))
            bufs.pp_proxy_tensors[key].fill_(7.0)  # stale previous replay
        reg.fill_from(
            _extend_batch(5),
            raw_bs=1,
            padded_bs=1,
            raw_num_tokens=5,
            padded_num_tokens=MAX_TOKENS,
            pp_proxy_tensors=_live_proxy(5),
        )
        for i, key in enumerate(KEYS):
            buf = bufs.pp_proxy_tensors[key]
            self.assertTrue(torch.equal(buf[:5], torch.full((5, H), 1.0 + i)), key)
            self.assertTrue(torch.equal(buf[5:], torch.zeros(MAX_TOKENS - 5, H)), key)

    def test_no_pp_slots_without_pp(self):
        reg = _registry(_buffers(keys=None))
        self.assertFalse(
            any(n.startswith("pp_proxy_tensors.") for n in reg.slot_names())
        )

    def _keys_for(self, start_layer, first=False, capture=(6, 20, 34, 48, 62), backend="full"):
        runner = object.__new__(pcgr.PrefillCudaGraphRunner)
        text = types.SimpleNamespace(
            layers=[None] * 64,
            start_layer=start_layer,
            layers_to_capture=list(capture),
        )
        runner.model_runner = types.SimpleNamespace(
            pp_group=types.SimpleNamespace(world_size=3, is_first_rank=first),
            model=types.SimpleNamespace(model=text),
        )
        return runner._resolve_pp_proxy_keys(backend)

    def test_the_received_aux_set_is_the_capture_ids_below_the_stage(self):
        # the 27B cut 42,11,11; capture ids = DFlash target ids + 1
        self.assertEqual(
            self._keys_for(42),
            ("hidden_states", "residual", "aux_layer_6", "aux_layer_20", "aux_layer_34"),
        )
        self.assertEqual(
            self._keys_for(53),
            (
                "hidden_states",
                "residual",
                "aux_layer_6",
                "aux_layer_20",
                "aux_layer_34",
                "aux_layer_48",
            ),
        )
        # producer off: no capture layer is marked on any stage
        self.assertEqual(self._keys_for(42, capture=()), ("hidden_states", "residual"))
        self.assertIsNone(self._keys_for(0, first=True))

    def test_a_non_full_backend_cannot_build_a_pp_stage(self):
        with self.assertRaises(RuntimeError):
            self._keys_for(42, backend="breakable")

    def test_output_slice_keeps_structure(self):
        hs = torch.arange(10.0).reshape(5, 2)
        aux = [torch.ones(5, 2), torch.zeros(5, 2)]
        cut = pcgr._slice_rows((hs, aux), 3)
        self.assertIsInstance(cut, tuple)
        self.assertEqual(tuple(cut[0].shape), (3, 2))
        self.assertEqual([tuple(a.shape) for a in cut[1]], [(3, 2), (3, 2)])
        ppx = pcgr._slice_rows(PPProxyTensors({"hidden_states": hs}), 2)
        self.assertEqual(tuple(ppx["hidden_states"].shape), (2, 2))
        self.assertIsNone(pcgr._slice_rows(None, 2))


def _full_runner(*, static_keys=KEYS, slots=1, captured_mode=CaptureHiddenMode.FULL):
    runner = object.__new__(pcgr.PrefillCudaGraphRunner)
    bufs = _buffers(keys=static_keys)
    runner.buffers = bufs
    runner.buffer_registry = _registry(bufs)
    runner._static_pp_proxy_tensors = (
        PPProxyTensors(bufs.pp_proxy_tensors) if bufs.pp_proxy_tensors else None
    )
    runner.capture_num_tokens = [MAX_TOKENS]
    runner.max_num_tokens = MAX_TOKENS
    runner.capture_hidden_mode = captured_mode
    runner.prefill_backend_name = "full"
    runner.backend = object()
    runner._is_full_backend = True
    runner._capture_req_slots = slots
    runner._prefill_static_buffers = {
        name: torch.zeros((2,), dtype=torch.int64) for name in pcgr._PREFILL_STATIC_FIELDS
    }
    runner.static_draft_hidden_states = None
    runner.capture_return_pooled_hidden_states = False
    runner._prepare_forward_metadata_for_replay = lambda *a, **k: None
    runner._next_token_logits_buffer = lambda rows: None
    runner._prefill_logits_buffer_rows = lambda fb: fb.batch_size
    return runner


class TestFullGraphEligibility(CustomTestCase):
    def test_a_null_or_full_extend_replays_on_a_full_capture(self):
        runner = _full_runner()
        for mode in (CaptureHiddenMode.NULL, CaptureHiddenMode.LAST, CaptureHiddenMode.FULL):
            fb = _extend_batch(9)
            fb.capture_hidden_mode = mode
            self.assertTrue(runner.can_run_graph(fb), mode)

    def test_a_request_above_the_captured_mode_is_refused(self):
        runner = _full_runner(captured_mode=CaptureHiddenMode.NULL)
        fb = _extend_batch(9)
        fb.capture_hidden_mode = CaptureHiddenMode.FULL
        self.assertEqual(runner._full_graph_ineligible_reason(fb), "capture_hidden_mode")

    def test_each_refusal_has_its_name(self):
        runner = _full_runner()
        self.assertEqual(runner._full_graph_ineligible_reason(_extend_batch(8, bs=2)), "bs>slots")
        fb = _extend_batch(9)
        fb.forward_mode = ForwardMode.MIXED
        self.assertEqual(runner._full_graph_ineligible_reason(fb), "mode=MIXED")
        fb = _extend_batch(MAX_TOKENS + 1)
        self.assertEqual(runner._full_graph_ineligible_reason(fb), "tokens>bucket")
        fb = _extend_batch(9)
        fb.contains_mm_inputs = lambda: True
        self.assertEqual(runner._full_graph_ineligible_reason(fb), "mm_inputs")
        fb = _extend_batch(9)
        fb.return_logprob = True
        fb.extend_logprob_start_lens_cpu = [0]
        self.assertEqual(runner._full_graph_ineligible_reason(fb), "input_logprob")
        fb = _extend_batch(9)
        fb.input_embeds = torch.zeros(9, H)
        self.assertEqual(runner._full_graph_ineligible_reason(fb), "input_embeds")

    def test_a_body_captured_on_mrope_needs_mrope_positions(self):
        runner = _full_runner()
        runner.__dict__["_captured_mrope"] = True
        self.assertEqual(runner._full_graph_ineligible_reason(_extend_batch(9)), "mrope_missing")
        fb = _extend_batch(9)
        fb.mrope_positions = torch.zeros(3, 9, dtype=torch.int64)
        self.assertIsNone(runner._full_graph_ineligible_reason(fb))

    def test_eager_fallbacks_are_counted_per_reason(self):
        runner = _full_runner()
        for _ in range(5):
            self.assertFalse(runner.can_run_graph(_extend_batch(8, bs=2)))
        self.assertEqual(runner._eager_reasons, {"bs>slots": 5})


class TestTypedChannelMetadataIsNotAStageKey(CustomTestCase):
    """Boot weg2xsn427 (2026-09-24 18:05:37Z, PP1, first replay): the live
    proxy off the typed channel carried ``__msg_type__`` beside
    hidden_states/residual, and load_batch refused it as a key drift:
    ``captured ['hidden_states', 'residual'], live ['__msg_type__',
    'hidden_states', 'residual']``. The channel metadata is left out by its
    EXPLICIT name list -- a real stage tensor with a dunder name is not."""

    def _live_with_meta(self, n, keys=KEYS, **meta):
        live = _live_proxy(n, keys=keys, value=3.0)
        for k, v in meta.items():
            live.tensors[k] = v
        return live

    def test_the_xsn427_message_replays(self):
        runner = _full_runner(static_keys=("hidden_states", "residual"))
        live = self._live_with_meta(
            5, keys=("hidden_states", "residual"), __msg_type__="proxy"
        )
        runner.load_batch(_extend_batch(5), pp_proxy_tensors=live)
        hs = runner.buffers.pp_proxy_tensors["hidden_states"]
        self.assertTrue(torch.equal(hs[:5], torch.full((5, H), 3.0)))
        self.assertTrue(torch.equal(hs[5:], torch.zeros(MAX_TOKENS - 5, H)))
        # the caller's message is not mutated: the eager tail gets it as sent
        self.assertEqual(live.tensors["__msg_type__"], "proxy")

    def test_every_listed_metadata_key_is_left_out(self):
        from sglang.srt.distributed.pp_typed_channel import CHANNEL_META_KEYS

        runner = _full_runner()
        live = self._live_with_meta(
            5,
            __msg_type__="proxy",
            __stamp__=(2, 1, 512, -1, 1, ("r", 0, 512)),
            __admission_decision__=((1, 2),),
        )
        self.assertTrue({"__msg_type__", "__stamp__", "__admission_decision__"} <= set(live.tensors))
        self.assertEqual(
            CHANNEL_META_KEYS,
            frozenset({"__msg_type__", "__stamp__", "__admission_decision__"}),
        )
        runner.load_batch(_extend_batch(5), pp_proxy_tensors=live)

    def test_a_missing_stage_tensor_is_still_refused(self):
        runner = _full_runner(static_keys=("hidden_states", "residual"))
        live = self._live_with_meta(5, keys=("hidden_states",), __msg_type__="proxy")
        with self.assertRaisesRegex(RuntimeError, r"live stage keys \['hidden_states'\]"):
            runner.load_batch(_extend_batch(5), pp_proxy_tensors=live)

    def test_a_dunder_named_stage_tensor_is_not_swallowed(self):
        """The list is explicit, not a prefix rule: an unknown dunder-named
        TENSOR stays a stage key, so a body that never captured it refuses."""
        runner = _full_runner(static_keys=("hidden_states", "residual"))
        live = self._live_with_meta(
            5,
            keys=("hidden_states", "residual"),
            __msg_type__="proxy",
            __extra_stage__=torch.zeros(5, H),
        )
        with self.assertRaisesRegex(RuntimeError, "__extra_stage__"):
            runner.load_batch(_extend_batch(5), pp_proxy_tensors=live)

    def test_the_names_are_the_senders_names(self):
        """The list mirrors the writer: Scheduler._pp_send_dict_to_next_stage
        writes these three, _pp_recv_proxy_tensors pops two of them."""
        import os

        from sglang.srt.distributed import pp_typed_channel as ch

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(ch.__file__))),
            "managers",
            "scheduler_pp_mixin.py",
        )
        src = open(path).read()
        self.assertIn(f'tensor_dict["{ch.MSG_TYPE_KEY}"] = msg_type', src)
        self.assertIn(f'tensor_dict["{ch.STAMP_KEY}"] = stamp', src)
        self.assertIn(f'_ADMISSION_DECISION_PAYLOAD_KEY = "{ch.ADMISSION_DECISION_KEY}"', src)


class TestNonFirstStageLoadBatch(CustomTestCase):
    def test_the_live_proxy_lands_in_the_static_buffers(self):
        runner = _full_runner()
        runner.load_batch(_extend_batch(5), pp_proxy_tensors=_live_proxy(5, value=3.0))
        hs = runner.buffers.pp_proxy_tensors["hidden_states"]
        self.assertTrue(torch.equal(hs[:5], torch.full((5, H), 3.0)))
        self.assertTrue(torch.equal(hs[5:], torch.zeros(MAX_TOKENS - 5, H)))

    def test_a_drifted_key_set_is_refused_by_name(self):
        runner = _full_runner()
        with self.assertRaisesRegex(RuntimeError, "pp proxy key mismatch"):
            runner.load_batch(
                _extend_batch(5),
                pp_proxy_tensors=_live_proxy(5, keys=("hidden_states", "residual")),
            )
        with self.assertRaisesRegex(RuntimeError, "pp proxy key mismatch"):
            runner.load_batch(_extend_batch(5))

    def test_the_padded_view_carries_the_slot_padded_start_locations(self):
        runner = _full_runner(slots=2)
        seen = {}
        runner._full_cg_seq_lens_cpu = torch.zeros((2,), dtype=torch.int64)
        runner.model_runner = types.SimpleNamespace(
            attn_backend=types.SimpleNamespace(
                init_forward_metadata_out_graph=lambda view: seen.setdefault("v", view)
            )
        )
        s = runner._prefill_static_buffers
        s["extend_start_loc"][:2] = torch.tensor([0, 5])
        pcgr.PrefillCudaGraphRunner._prepare_forward_metadata_for_replay(
            runner, _extend_batch(5), None, MAX_TOKENS
        )
        view = seen["v"]
        self.assertEqual(view.batch_size, 2)
        self.assertEqual(view.extend_start_loc.tolist(), [0, 5])
        self.assertEqual(view.extend_start_loc.data_ptr(), s["extend_start_loc"].data_ptr())


# --------------------------------------------------------------------------
# 5. Global layer ids under PP; the runtime post's env parse
# --------------------------------------------------------------------------
class TestAlignPipelineLayers(CustomTestCase):
    def _lm(self, start, end, total=64):
        return types.SimpleNamespace(layers=[None] * total, start_layer=start, end_layer=end)

    def test_a_stage_is_indexed_by_global_layer_id(self):
        owned = [f"L{i}" for i in range(42, 53)]
        out = align_pipeline_layers(owned, self._lm(42, 53))
        self.assertEqual(len(out), 64)
        self.assertEqual(out[42], "L42")
        self.assertEqual(out[52], "L52")
        self.assertIsNone(out[41])
        self.assertIsNone(out[53])

    def test_a_gap_comes_out_short_so_the_gqa_check_keeps_the_graph_off(self):
        out = align_pipeline_layers(["a"] * 10, self._lm(42, 53))
        self.assertLess(len(out), 64)

    def test_more_layers_than_owned_is_refused(self):
        with self.assertRaises(AssertionError):
            align_pipeline_layers(["a"] * 12, self._lm(42, 53))


class _FakeFullBackend:
    """Stands in for FullCudaGraphBackend: a replay returns the output the
    'captured' body produced at the bucket, and records what the static
    buffers held at that moment."""

    def __init__(self, runner, output):
        self.runner = runner
        self.output = output
        self.seen = {}

    def replay_session(self):
        import contextlib

        return contextlib.nullcontext()

    def replay(self, shape_key, static_forward_batch, **kw):
        reg = self.runner.buffer_registry
        if reg.has_slot("input_embeds"):
            self.seen["embeds"] = reg.get_slot("input_embeds").buffer.clone()
        if reg.has_slot("mrope_positions"):
            self.seen["mrope"] = reg.get_slot("mrope_positions").buffer.clone()
        if self.runner._static_pp_proxy_tensors is not None:
            self.seen["hidden"] = (
                self.runner._static_pp_proxy_tensors["hidden_states"].clone()
            )
        return self.output


class _LayerModel(torch.nn.Module):
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        raise AssertionError("the eager body ran: the replay closure was not installed")


class _Outer:
    """The multimodal outer forward's shape (qwen3_vl.forward via
    general_mm_embed_routine): embeddings computed eagerly on the first stage,
    the body called with input_ids=None and input_embeds / pp_proxy_tensors."""

    def __init__(self, layer_model, first, last):
        self.layer_model = layer_model
        self.first, self.last = first, last

    def get_input_embeddings(self):
        return lambda ids: ids.to(torch.float32)[:, None].repeat(1, H) + 0.5

    def forward(self, input_ids, positions, forward_batch, pp_proxy_tensors=None):
        ie = self.get_input_embeddings()(input_ids) if self.first else None
        out = self.layer_model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=ie,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.last:
            hidden, aux = out
            # what the eager tail (the logits processor) is handed
            self.body_rows = (int(hidden.shape[0]), [int(a.shape[0]) for a in aux])
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            return LogitsProcessorOutput(
                next_token_logits=hidden[-1:].clone(),
                hidden_states=torch.cat([hidden] + list(aux), dim=-1),
            )
        return out


def _exec_runner(*, first, last, static_keys):
    runner = object.__new__(pcgr.PrefillCudaGraphRunner)
    bufs = PrefillInputBuffers.create(
        device=CPU,
        max_bs=2,
        max_num_tokens=MAX_TOKENS,
        cache_loc_dtype=torch.int64,
        is_multimodal=True,
        hidden_size=H,
        dtype=torch.float32,
        enable_mamba_track=False,
        pp_proxy_keys=static_keys,
    )
    runner.buffers = bufs
    runner.buffer_registry = build_prefill_registry(
        device=CPU,
        max_bs=2,
        max_num_token=MAX_TOKENS,
        cache_loc_dtype=torch.int64,
        is_multimodal=True,
        hidden_size=H,
        embed_dtype=torch.float32,
        share_pool=False,
        source=bufs,
    )
    runner._static_pp_proxy_tensors = (
        PPProxyTensors(bufs.pp_proxy_tensors) if bufs.pp_proxy_tensors else None
    )
    runner.capture_num_tokens = [MAX_TOKENS]
    runner.max_num_tokens = MAX_TOKENS
    runner.capture_hidden_mode = CaptureHiddenMode.FULL
    runner.prefill_backend_name = "full"
    runner._is_full_backend = True
    runner._capture_req_slots = 1
    runner._prefill_static_buffers = {
        name: torch.zeros((2,), dtype=torch.int64) for name in pcgr._PREFILL_STATIC_FIELDS
    }
    runner._full_cg_seq_lens_cpu = torch.zeros((1,), dtype=torch.int64)
    runner.static_draft_hidden_states = None
    runner.capture_return_pooled_hidden_states = False
    runner._prepare_forward_metadata_for_replay = lambda *a, **k: None
    runner._next_token_logits_buffer = lambda rows: None
    runner._prefill_logits_buffer_rows = lambda fb: fb.batch_size
    runner.layer_model = _LayerModel()
    runner._input_embeds_arg_idx = 3
    runner.attention_layers = []
    runner.quant_config = None
    runner.moe_layers = []
    runner.moe_fusions = []
    runner.dsa_indexers = None
    outer = _Outer(runner.layer_model, first, last)
    runner.model_runner = types.SimpleNamespace(
        model=outer,
        attn_backend=object(),
        pp_group=types.SimpleNamespace(is_first_rank=first, is_last_rank=last),
        spec_algorithm=types.SimpleNamespace(is_speculative=lambda: False),
    )
    return runner


class TestExecuteFullPath(CustomTestCase):
    """execute() of the full backend, end to end around a fake replay: the
    text chunk of the MULTIMODAL P (is_multimodal registry: mrope + embeds
    slots) shorter than the bucket."""

    def _batch(self, n):
        fb = _extend_batch(n)
        fb.mrope_positions = torch.arange(3 * n, dtype=torch.int64).reshape(3, n) + 1
        return fb

    def test_first_stage_embeds_head_copy_and_padded_tail(self):
        n = 5
        runner = _exec_runner(first=True, last=False, static_keys=None)
        # stale previous replay in both slots
        runner.buffer_registry.get_slot("input_embeds").buffer.fill_(9.0)
        runner.buffer_registry.get_slot("mrope_positions").buffer.fill_(7)
        out_rows = PPProxyTensors(
            {"hidden_states": torch.ones(MAX_TOKENS, H), "residual": torch.ones(MAX_TOKENS, H)}
        )
        runner.backend = _FakeFullBackend(runner, out_rows)
        fb = self._batch(n)
        out = runner.execute(fb)
        seen = runner.backend.seen
        want = torch.arange(n, dtype=torch.float32)[:, None].repeat(1, H) + 0.5
        self.assertTrue(torch.equal(seen["embeds"][:n], want))
        self.assertTrue(torch.equal(seen["embeds"][n:], torch.zeros(MAX_TOKENS - n, H)))
        self.assertTrue(torch.equal(seen["mrope"][:, :n], fb.mrope_positions))
        self.assertTrue(torch.equal(seen["mrope"][:, n:], torch.zeros(3, MAX_TOKENS - n, dtype=torch.int64)))
        # non-last stage: the stage output, raw rows only
        self.assertIsInstance(out, PPProxyTensors)
        self.assertEqual(tuple(out["hidden_states"].shape), (n, H))
        # the body's own forward is back after the replay
        self.assertIs(runner.layer_model.forward.__func__, _LayerModel.forward)

    def test_last_stage_with_aux_capture_slices_every_tensor(self):
        n = 5
        keys = ("hidden_states", "residual", "aux_layer_6")
        runner = _exec_runner(first=False, last=True, static_keys=keys)
        body_out = (
            torch.full((MAX_TOKENS, H), 2.0),
            [torch.full((MAX_TOKENS, H), 3.0), torch.full((MAX_TOKENS, H), 4.0)],
        )
        runner.backend = _FakeFullBackend(runner, body_out)
        live = _live_proxy(n, keys=keys, value=6.0)
        live.tensors["__msg_type__"] = "proxy"  # as the typed channel delivers it
        out = runner.execute(self._batch(n), pp_proxy_tensors=live)
        self.assertTrue(
            torch.equal(runner.backend.seen["hidden"][:n], torch.full((n, H), 6.0))
        )
        self.assertTrue(
            torch.equal(runner.backend.seen["hidden"][n:], torch.zeros(MAX_TOKENS - n, H))
        )
        # (hidden, [aux, aux]) cut to n rows EACH before the tail sees them --
        # a bare hs[:n] slices the tuple and hands the tail bucket-row tensors
        self.assertEqual(runner.model_runner.model.body_rows, (n, [n, n]))
        self.assertEqual(tuple(out.hidden_states.shape), (n, 3 * H))


class _NoHostRead(torch.Tensor):
    """A cu_seqlens stand-in whose host reads are the defect under capture."""

    def item(self):  # noqa: D401
        raise AssertionError("item() read under capture")

    def tolist(self):
        raise AssertionError("tolist() read under capture")


class TestGdnTraceIsSkippedUnderCapture(CustomTestCase):
    """The #631b trace syncs (item / tolist) for its first 40 calls; one sync
    inside a stream capture invalidates the capture even though the trace
    swallows the exception. Under capture it must not touch the tensor."""

    def _extend(self, capturing):
        from sglang.srt.layers.attention.linear.kernels import gdn_triton

        calls = {}

        def fake_chunk(**kw):
            calls["kw"] = kw
            return "OUT"

        saved = (
            gdn_triton.__dict__.get("chunk_gated_delta_rule"),
            torch.cuda.is_available,
            torch.cuda.is_current_stream_capturing,
        )
        gdn_triton.chunk_gated_delta_rule = fake_chunk
        torch.cuda.is_available = lambda: True
        torch.cuda.is_current_stream_capturing = lambda: capturing
        seen = []
        try:
            gdn_triton.TritonGDNKernel._631b_n = 0
            qsl = torch.tensor([0, 8], dtype=torch.int32).as_subclass(_NoHostRead)
            out = gdn_triton.TritonGDNKernel().extend(
                torch.zeros(1, 8, 1, 4),
                torch.zeros(1, 8, 1, 4),
                torch.zeros(1, 8, 1, 4),
                torch.zeros(1, 8, 1),
                torch.zeros(1, 8, 1),
                ssm_states=torch.zeros(2, 1, 4, 4),
                cache_indices=torch.tensor([1], dtype=torch.int32),
                query_start_loc=qsl,
            )
            seen.append(getattr(gdn_triton.TritonGDNKernel, "_631b_n", 0))
        finally:
            if saved[0] is None:
                gdn_triton.__dict__.pop("chunk_gated_delta_rule", None)
            else:
                gdn_triton.chunk_gated_delta_rule = saved[0]
            torch.cuda.is_available = saved[1]
            torch.cuda.is_current_stream_capturing = saved[2]
        return out, calls, seen[0]

    def test_under_capture_the_trace_never_reads_the_device(self):
        out, calls, n = self._extend(capturing=True)
        self.assertEqual(out, "OUT")
        self.assertIs(calls["kw"]["cu_seqlens"].__class__, _NoHostRead)
        self.assertEqual(n, 0, "the trace counter moved: the trace body ran")

    def test_outside_capture_the_trace_still_runs(self):
        """Control: the same tensor IS read when no capture is active (the
        read raises, the trace swallows it, the forward goes on)."""
        out, _calls, n = self._extend(capturing=False)
        self.assertEqual(out, "OUT")
        self.assertEqual(n, 1)


class TestGraphPoolPostEnv(CustomTestCase):
    def test_one_entry_per_rank(self):
        self.assertEqual(PREFILL_GRAPH_POOL_ENV, "SGLANG_KV_BUDGET_PREFILL_GRAPH_MIB")
        self.assertEqual(prefill_transient_mib_for_rank("160.0,150.5,140.0", 1), 150.5)
        self.assertEqual(prefill_transient_mib_for_rank("", 0), 0.0)


if __name__ == "__main__":
    unittest.main()
