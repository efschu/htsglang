# SPDX-License-Identifier: Apache-2.0
"""#444: the TARGET_VERIFY conv must not publish an uncommitted window.

THE DEFECT, as it stood
``_causal_conv1d_update_kernel`` is intentionally in-place. STEP 2 stores the
shifted sliding window back into whatever ``conv_state`` it was handed
(``layers/attention/mamba/causal_conv1d_triton.py:752``) and that store is NOT
gated on ``SAVE_INTERMEDIATE`` -- the speculative branch only moves the source
offset (``:715``). ``GDNAttnBackend.forward_extend`` handed it the PERSISTENT
pool row, so a verify forward advanced that row over every drafted step,
accepted or not. The row only became correct again afterwards, when
``update_mamba_state_after_mtp_verify``
(``layers/attention/hybrid_linear_attn_backend.py:1098``) scattered the
accepted step back out of the intermediate cache.

That is mutate-then-repair. Between the two points the pool row holds the LAST
CANDIDATE's window -- a value that is neither the pre-verify state nor the
committed one. ``TestTheWindowIsRealAndObservable`` measures exactly that: it
reads the pool at the seam and shows the reading differs from both endpoints.

Single-stream the window is unobservable: nothing between the verify forward
and the commit reads the recurrent pool (``speculative/eagle_worker_v2.py``
runs ``eagle_sample`` and KV bookkeeping in between, then commits at
``:2716``). It is observable for a second worker reading the same pool
concurrently, which is what the dual-group lane does -- its batches are built
on ``runner.req_to_token_pool`` itself (``model_executor/dual_group_lane.py``
``:2590``) and it reads ``mamba_cache.conv`` directly (``:2932``, ``:3398``).
Exposed intermediate state is correct only for as long as no such reader
exists.

THE FIX
``GDNAttnBackend._target_verify_conv`` runs the verify conv on a
request-private copy of the window (the same shape as the upstream DVR
adapter's ``private_conv_state``). The persistent row is then written exactly
once per verify, by the commit.

WHAT FAILS WITHOUT THE FIX
``TestPersistentRowIsNotPublishedDuringVerify`` -- with ``_target_verify_conv``
reverted to handing ``conv_states`` / ``cache_indices`` to the kernel, the pool
row read at the seam is not the pre-verify row.

WHAT MUST NOT MOVE
``TestCommittedBytesAreUnchanged`` pins byte-neutrality against the legacy
mutate-then-repair sequence: the committed pool and the conv OUTPUT are
element-identical, because the commit's scatter overwrites every element of
the row it touches and the kernel sees the same prior state either way.
``TestRefusalWhenNoSpeculativeCache`` pins that the non-speculative path never
silently acquires the private buffer.

FIDELITY OF THE MODEL
No CUDA here, so the Triton kernel is replaced by ``_reference_conv_update``,
transcribed from the kernel's own index arithmetic:

  * STEP 2 store -- ``new_state[:, i] = old_state[:, i + seqlen]`` while
    ``i + seqlen < state_len``, else ``x[:, i + seqlen - state_len]``
    (``causal_conv1d_triton.py:705-752``), i.e. the last ``state_len`` columns
    of ``cat([state, x])``.
  * ``SAVE_INTERMEDIATE`` -- step ``t`` stores the ``K-1`` most recently
    consumed inputs, oldest first, into ``[slot, step, dim, win]``
    (``:930-960``), i.e. ``cat([state, x])[:, t+1 : t+K]``.

``_reference_commit`` transcribes
``_fused_conv_window_scatter_with_mask_kernel`` (``:305-372``): destination row
from ``dst_indices_raw``, source row from the request POSITION, whole
``dim x (K-1)`` window copied, entries with ``step < 0`` skipped.

THE GRAPH-PADDED VERIFY (#444b)
``TestGraphPaddedVerify`` covers the case the first revision of this file
missed. When the live batch is smaller than the CUDA graph it replays into,
the tail of the state-index buffer holds ``PAD_SLOT_ID`` --
``hybrid_linear_attn_backend.py:378`` and ``:459`` fill the per-bs buffers with
it, ``:555`` re-stamps it over the padding requests on every replay, and
``:97-99`` does the same on the eager path. The Triton kernel tolerates that by
construction: ``USE_PAD_SLOT`` makes program ``idx_seq`` return before it reads
or writes anything (``causal_conv1d_triton.py:659-662``). Torch's
``index_select`` / ``index_copy_`` do not -- a ``-1`` row is an out-of-bounds
index, which is the device-side assert the #444a boot hit.

Because of that the reference here is pad-aware too, per the reference-twin
drift rule (FEATURE_CATALOG §12): a transcription that silently accepts a row
index the kernel refuses to touch cannot validate the kernel. ``-1`` is a legal
Python index, so without the explicit skip the reference would quietly seed and
mutate the LAST pool row on every padded slot.
"""

import unittest
from types import SimpleNamespace
from typing import List, Optional
from unittest import mock

import torch
from sglang.srt.layers.attention.linear import gdn_backend as gdn_backend_mod
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Geometry of the modelled verify: two requests, a width-4 conv (so the window
# is three columns wide, the Qwen3-Next GDN shape), four drafted tokens.
DIM = 6
KERNEL_WIDTH = 4
WIN = KERNEL_WIDTH - 1
DRAFT_TOKENS = 4
BATCH = 2
# The captured CUDA-graph batch the live BATCH replays into, so GRAPH_BATCH -
# BATCH lanes are padding. Any live size that is not exactly a captured size
# lands here; bs=3 into the bs=4 graph is what killed the server on the boot.
GRAPH_BATCH = 4
POOL_ROWS = 7
SPEC_ROWS = 5
DTYPE = torch.float32


class _ConvKernelSpy:
    """CPU stand-in for ``causal_conv1d_update`` with the kernel's contract.

    ``calls`` records, per invocation, the ``conv_state`` object it was handed
    -- which is the whole question this file asks. ``row_indices`` records the
    ``conv_state_indices`` it was handed, which is how the padded case checks
    that padding still reaches the kernel as ``PAD_SLOT_ID``.
    """

    def __init__(self):
        self.calls: List[torch.Tensor] = []
        self.row_indices: List[torch.Tensor] = []
        self.seam_reader = None

    def __call__(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        activation=None,
        *,
        conv_state_indices: torch.Tensor,
        intermediate_conv_window: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        **_ignored,
    ) -> torch.Tensor:
        self.calls.append(conv_state)
        self.row_indices.append(conv_state_indices.clone())
        out = _reference_conv_update(
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices=conv_state_indices,
            intermediate_conv_window=intermediate_conv_window,
            intermediate_state_indices=intermediate_state_indices,
        )
        if self.seam_reader is not None:
            self.seam_reader()
        return out


def _reference_conv_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation,
    *,
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
) -> torch.Tensor:
    """Transcription of the Triton kernel's contract; see the module docstring.

    ``x``: ``[bs, dim, seqlen]``. ``conv_state``: ``[rows, dim, K-1]``, mutated
    IN PLACE. ``intermediate_conv_window``: ``[slots, steps, dim, K-1]``.
    """
    bs, dim, seqlen = x.shape
    win = conv_state.shape[-1]
    width = weight.shape[-1]
    out = torch.empty_like(x)
    for b in range(bs):
        row = int(conv_state_indices[b])
        if row == PAD_SLOT_ID:
            # USE_PAD_SLOT (causal_conv1d_triton.py:659-662): the program for
            # this sequence returns before reading the state, before the STEP 2
            # in-place store, before SAVE_INTERMEDIATE and before writing `out`.
            # `out` is torch.empty_like (causal_conv1d_triton.py:1070), so a
            # padded lane's output stays uninitialised on purpose.
            continue
        slot = int(intermediate_state_indices[b])
        full = torch.cat([conv_state[row], x[b]], dim=-1)  # [dim, win + seqlen]
        for t in range(seqlen):
            acc = torch.zeros(dim, dtype=x.dtype)
            for j in range(width):
                acc = acc + weight[:, j] * full[:, t + j]
            if bias is not None:
                acc = acc + bias
            if activation in ("silu", "swish"):
                acc = torch.nn.functional.silu(acc)
            out[b, :, t] = acc
            # STEP 4 / SAVE_INTERMEDIATE: window after consuming token t.
            intermediate_conv_window[slot, t] = full[:, t + 1 : t + 1 + win]
        # STEP 2: the in-place sliding-window store.
        conv_state[row] = full[:, seqlen:]
    return out


def _reference_commit(
    conv_states: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    dst_indices: torch.Tensor,
    step_indices: torch.Tensor,
) -> None:
    """Transcription of ``_fused_conv_window_scatter_with_mask_kernel``."""
    for pos in range(step_indices.shape[0]):
        step = int(step_indices[pos])
        if step < 0:
            continue
        dst = int(dst_indices[pos])
        if dst < 0:
            # The kernel guards the destination row too
            # (``mamba_state_scatter_triton.py:344-350``), so a padded lane
            # commits nothing even if its step were non-negative.
            continue
        conv_states[dst] = intermediate_conv_window[pos, step]


def _fixture(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    conv_states = torch.randn(POOL_ROWS, DIM, WIN, generator=gen, dtype=DTYPE)
    intermediate = torch.zeros(SPEC_ROWS, DRAFT_TOKENS, DIM, WIN, dtype=DTYPE)
    x = torch.randn(BATCH, DIM, DRAFT_TOKENS, generator=gen, dtype=DTYPE)
    layer = SimpleNamespace(
        conv_weights=torch.randn(DIM, KERNEL_WIDTH, generator=gen, dtype=DTYPE),
        bias=torch.randn(DIM, generator=gen, dtype=DTYPE),
        activation="silu",
    )
    # Deliberately not 0..BATCH-1: the pool rows a request owns and the spec
    # rows the intermediates use are different index spaces, and the fix must
    # not conflate them.
    cache_indices = torch.tensor([4, 1], dtype=torch.int32)
    spec_indices = torch.arange(POOL_ROWS, dtype=torch.int32)
    return conv_states, intermediate, x, layer, cache_indices, spec_indices


def _padded_fixture(seed: int = 0):
    """A verify whose live batch is smaller than the graph it replays into.

    Two live requests replaying the bs=4 graph, so ``cache_indices`` carries two
    trailing ``PAD_SLOT_ID`` entries -- byte for byte what
    ``_replay_metadata`` produces (``hybrid_linear_attn_backend.py:555``,
    ``mamba_indices[bs - num_padding:] = -1``) and what the eager path produces
    at ``:97-99``. ``x`` is full graph width because the graph replays at its
    captured batch size regardless of how many lanes are live.
    """
    conv_states, intermediate, x, layer, live_indices, spec_indices = _fixture(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    x = torch.randn(GRAPH_BATCH, DIM, DRAFT_TOKENS, generator=gen, dtype=DTYPE)
    cache_indices = torch.full((GRAPH_BATCH,), PAD_SLOT_ID, dtype=torch.int32)
    cache_indices[:BATCH] = live_indices
    return conv_states, intermediate, x, layer, cache_indices, spec_indices


def _backend_with_scratch() -> GDNAttnBackend:
    """A ``GDNAttnBackend`` carrying only what ``_target_verify_conv`` reads.

    ``__init__`` needs a ModelRunner, i.e. a booted engine; the unit under test
    is the buffer decision, so the instance is built without it.
    """
    backend = object.__new__(GDNAttnBackend)
    backend._verify_conv_scratch = torch.zeros(SPEC_ROWS, DIM, WIN, dtype=DTYPE)
    return backend


class TestTheWindowIsRealAndObservable(CustomTestCase):
    """The defect, measured: mutate-then-repair publishes a third state.

    Always green -- it describes the legacy sequence rather than the fix, and
    it is what makes the fix's assertion mean something. If this ever goes red
    the kernel contract changed and the rest of the file is stale.
    """

    def test_seam_read_differs_from_both_endpoints(self):
        conv_states, intermediate, x, layer, cache_indices, spec_indices = _fixture()
        before = conv_states.clone()

        # Legacy call shape: the kernel is handed the persistent pool.
        _reference_conv_update(
            x,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
            intermediate_conv_window=intermediate,
            intermediate_state_indices=spec_indices[:BATCH],
        )
        at_seam = conv_states.clone()

        # Accept one drafted token for req 0, three for req 1.
        accepted = torch.tensor([0, 2], dtype=torch.int64)
        _reference_commit(conv_states, intermediate, cache_indices, accepted)
        after = conv_states.clone()

        for pos in range(BATCH):
            row = int(cache_indices[pos])
            self.assertFalse(
                torch.equal(at_seam[row], before[row]),
                f"row {row}: the seam read equals the pre-verify state, so the "
                "kernel did not mutate the pool and this file models the wrong "
                "kernel",
            )
            self.assertFalse(
                torch.equal(at_seam[row], after[row]),
                f"row {row}: the seam read equals the committed state, so there "
                "is no window to close",
            )
        # And the published value is specifically the LAST candidate's window,
        # which is the accepted one only by accident.
        self.assertTrue(
            torch.equal(
                at_seam[int(cache_indices[0])], intermediate[0, DRAFT_TOKENS - 1]
            ),
            "the published row is not the last candidate's window",
        )

    def test_rows_the_verify_does_not_own_are_untouched(self):
        conv_states, intermediate, x, layer, cache_indices, spec_indices = _fixture()
        before = conv_states.clone()
        _reference_conv_update(
            x,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
            intermediate_conv_window=intermediate,
            intermediate_state_indices=spec_indices[:BATCH],
        )
        owned = {int(i) for i in cache_indices}
        for row in range(POOL_ROWS):
            if row in owned:
                continue
            self.assertTrue(torch.equal(conv_states[row], before[row]))


class TestPersistentRowIsNotPublishedDuringVerify(CustomTestCase):
    """The falsifier. Red with ``_target_verify_conv`` reverted to the pool row."""

    def test_pool_is_byte_identical_at_the_seam(self):
        conv_states, intermediate, x, layer, cache_indices, spec_indices = _fixture()
        before = conv_states.clone()
        backend = _backend_with_scratch()

        spy = _ConvKernelSpy()
        seen: List[torch.Tensor] = []
        spy.seam_reader = lambda: seen.append(conv_states.clone())

        with mock.patch.object(gdn_backend_mod, "causal_conv1d_update", spy):
            backend._target_verify_conv(
                layer,
                x,
                conv_states,
                cache_indices,
                intermediate,
                spec_indices,
                None,
                None,
                None,
            )

        self.assertEqual(len(seen), 1)
        self.assertTrue(
            torch.equal(seen[0], before),
            "a reader between the verify conv and the commit saw a pool row "
            "that no accept decision had authorised",
        )
        self.assertTrue(torch.equal(conv_states, before))

    def test_the_kernel_is_handed_the_private_buffer(self):
        conv_states, intermediate, x, layer, cache_indices, spec_indices = _fixture()
        backend = _backend_with_scratch()
        spy = _ConvKernelSpy()
        with mock.patch.object(gdn_backend_mod, "causal_conv1d_update", spy):
            backend._target_verify_conv(
                layer,
                x,
                conv_states,
                cache_indices,
                intermediate,
                spec_indices,
                None,
                None,
                None,
            )
        self.assertEqual(len(spy.calls), 1)
        self.assertIs(spy.calls[0], backend._verify_conv_scratch)
        self.assertIsNot(spy.calls[0], conv_states)

    def test_private_rows_are_seeded_with_the_persistent_window(self):
        """The prior state is an INPUT to the conv; a private copy that is not
        seeded would silently change the arithmetic rather than the exposure."""
        conv_states, intermediate, x, layer, cache_indices, spec_indices = _fixture()
        backend = _backend_with_scratch()
        seeded: List[torch.Tensor] = []

        spy = _ConvKernelSpy()
        original = spy.__call__

        def capture(*args, **kwargs):
            seeded.append(args[1].clone())
            return original(*args, **kwargs)

        with mock.patch.object(gdn_backend_mod, "causal_conv1d_update", capture):
            backend._target_verify_conv(
                layer,
                x,
                conv_states,
                cache_indices,
                intermediate,
                spec_indices,
                None,
                None,
                None,
            )

        for pos in range(BATCH):
            self.assertTrue(
                torch.equal(seeded[0][pos], conv_states[int(cache_indices[pos])]),
                f"private row {pos} was not seeded from pool row "
                f"{int(cache_indices[pos])}",
            )


class TestCommittedBytesAreUnchanged(CustomTestCase):
    """Byte-neutrality: only the publication of the window changed."""

    def _legacy(self, accepted: torch.Tensor):
        conv_states, intermediate, x, layer, cache_indices, spec_indices = _fixture()
        out = _reference_conv_update(
            x,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
            intermediate_conv_window=intermediate,
            intermediate_state_indices=spec_indices[:BATCH],
        )
        _reference_commit(conv_states, intermediate, cache_indices, accepted)
        return conv_states, out

    def _fixed(self, accepted: torch.Tensor):
        conv_states, intermediate, x, layer, cache_indices, spec_indices = _fixture()
        backend = _backend_with_scratch()
        spy = _ConvKernelSpy()
        with mock.patch.object(gdn_backend_mod, "causal_conv1d_update", spy):
            out = backend._target_verify_conv(
                layer,
                x,
                conv_states,
                cache_indices,
                intermediate,
                spec_indices,
                None,
                None,
                None,
            )
        _reference_commit(conv_states, intermediate, cache_indices, accepted)
        return conv_states, out

    def test_committed_pool_is_element_identical(self):
        for accepted in (
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([0, 2], dtype=torch.int64),
            torch.tensor([DRAFT_TOKENS - 1, DRAFT_TOKENS - 1], dtype=torch.int64),
        ):
            with self.subTest(accepted=accepted.tolist()):
                legacy_pool, legacy_out = self._legacy(accepted)
                fixed_pool, fixed_out = self._fixed(accepted)
                self.assertTrue(torch.equal(legacy_pool, fixed_pool))
                self.assertTrue(torch.equal(legacy_out, fixed_out))

    def test_commit_overwrites_the_whole_row(self):
        """The load-bearing premise: nothing of the verify's in-place write
        survived the commit, which is why dropping it changes no committed
        byte."""
        conv_states, intermediate, x, layer, cache_indices, spec_indices = _fixture()
        _reference_conv_update(
            x,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
            intermediate_conv_window=intermediate,
            intermediate_state_indices=spec_indices[:BATCH],
        )
        accepted = torch.tensor([1, 3], dtype=torch.int64)
        _reference_commit(conv_states, intermediate, cache_indices, accepted)
        for pos in range(BATCH):
            self.assertTrue(
                torch.equal(
                    conv_states[int(cache_indices[pos])],
                    intermediate[pos, int(accepted[pos])],
                )
            )


class TestGraphPaddedVerify(CustomTestCase):
    """#444b: a verify replayed into a larger captured graph.

    Red on the tree that shipped #444a. ``conv_states.index_select(0,
    cache_indices)`` is handed a ``-1`` row, which on CUDA is
    ``Indexing.cu:1515: indexSelectSmallIndex: Assertion 'srcIndex <
    srcSelectDimSize' failed`` and on CPU an ``IndexError`` from the same bounds
    check. The kernel this call replaced skipped those lanes.
    """

    def _run_fixed(self, conv_states, intermediate, x, layer, cache_indices, spec):
        backend = _backend_with_scratch()
        spy = _ConvKernelSpy()
        with mock.patch.object(gdn_backend_mod, "causal_conv1d_update", spy):
            out = backend._target_verify_conv(
                layer, x, conv_states, cache_indices, intermediate, spec,
                None, None, None,
            )  # fmt: skip
        return backend, spy, out

    def test_a_padded_verify_does_not_index_out_of_bounds(self):
        """The crash, reduced: the seed gather must tolerate PAD_SLOT_ID."""
        fixture = _padded_fixture()
        _, spy, out = self._run_fixed(*fixture)
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(out.shape, (GRAPH_BATCH, DIM, DRAFT_TOKENS))

    def test_padding_reaches_the_kernel_as_pad_slot_id(self):
        """Masking the gather is not enough: the kernel must still SKIP the
        padded lanes. Handing it a valid scratch row for a padded lane would
        run the conv on an unseeded row instead of not running it."""
        conv_states, intermediate, x, layer, cache_indices, spec = _padded_fixture()
        _, spy, _ = self._run_fixed(
            conv_states, intermediate, x, layer, cache_indices, spec
        )
        handed = spy.row_indices[0]
        self.assertEqual(tuple(handed.shape), (GRAPH_BATCH,))
        for pos in range(BATCH):
            self.assertEqual(
                int(handed[pos]),
                int(spec[pos]),
                f"live lane {pos} lost its private row",
            )
        for pos in range(BATCH, GRAPH_BATCH):
            self.assertEqual(
                int(handed[pos]),
                PAD_SLOT_ID,
                f"padded lane {pos} was handed a real row, so the kernel will "
                "run the conv on it instead of skipping it",
            )

    def test_padded_lanes_touch_neither_pool_nor_intermediate(self):
        conv_states, intermediate, x, layer, cache_indices, spec = _padded_fixture()
        before_pool = conv_states.clone()
        before_inter = intermediate.clone()
        self._run_fixed(conv_states, intermediate, x, layer, cache_indices, spec)

        self.assertTrue(
            torch.equal(conv_states, before_pool),
            "the padded verify published a persistent pool row",
        )
        for slot in range(SPEC_ROWS):
            if slot < BATCH:
                continue
            self.assertTrue(
                torch.equal(intermediate[slot], before_inter[slot]),
                f"padded lane wrote intermediate slot {slot}",
            )

    def test_padded_verify_is_element_identical_to_the_legacy_path(self):
        """Byte-neutrality, extended to the padded shape: what the pre-#444a
        tree served at these batch sizes is what the fixed tree serves."""
        for accepted_live in ([0, 0], [0, 2], [DRAFT_TOKENS - 1, 1]):
            with self.subTest(accepted=accepted_live):
                accepted = torch.tensor(
                    accepted_live + [-1] * (GRAPH_BATCH - BATCH), dtype=torch.int64
                )

                # Legacy: the pre-#444a call shape, pool row + raw cache_indices.
                c_l, inter_l, x_l, layer_l, idx_l, spec_l = _padded_fixture()
                out_l = _reference_conv_update(
                    x_l,
                    c_l,
                    layer_l.conv_weights,
                    layer_l.bias,
                    layer_l.activation,
                    conv_state_indices=idx_l,
                    intermediate_conv_window=inter_l,
                    intermediate_state_indices=spec_l[:GRAPH_BATCH],
                )
                _reference_commit(c_l, inter_l, idx_l, accepted)

                c_f, inter_f, x_f, layer_f, idx_f, spec_f = _padded_fixture()
                _, _, out_f = self._run_fixed(c_f, inter_f, x_f, layer_f, idx_f, spec_f)
                _reference_commit(c_f, inter_f, idx_f, accepted)

                self.assertTrue(torch.equal(c_l, c_f), "committed pool moved")
                self.assertTrue(
                    torch.equal(inter_l, inter_f), "intermediate window moved"
                )
                # Padded lanes are torch.empty_like in both paths -- comparing
                # them would compare uninitialised memory.
                self.assertTrue(
                    torch.equal(out_l[:BATCH], out_f[:BATCH]),
                    "the conv output of a live lane moved",
                )

    def test_the_reference_itself_refuses_the_pad_row(self):
        """Control for the reference-twin rule: -1 is a legal Python index, so
        this pins that the pad skip above is real and not a coincidence of the
        fixture. Without the skip the reference would seed and mutate
        ``conv_states[-1]``, i.e. pool row POOL_ROWS-1, on every padded lane."""
        conv_states, intermediate, x, layer, cache_indices, spec = _padded_fixture()
        before = conv_states.clone()
        _reference_conv_update(
            x,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
            intermediate_conv_window=intermediate,
            intermediate_state_indices=spec[:GRAPH_BATCH],
        )
        self.assertTrue(
            torch.equal(conv_states[POOL_ROWS - 1], before[POOL_ROWS - 1]),
            "the reference wrapped PAD_SLOT_ID onto the last pool row",
        )


class TestRefusalWhenNoSpeculativeCache(CustomTestCase):
    def test_missing_private_buffer_is_a_named_refusal(self):
        conv_states, intermediate, x, layer, cache_indices, spec_indices = _fixture()
        backend = object.__new__(GDNAttnBackend)
        backend._verify_conv_scratch = None
        with self.assertRaises(AssertionError) as ctx:
            backend._target_verify_conv(
                layer,
                x,
                conv_states,
                cache_indices,
                intermediate,
                spec_indices,
                None,
                None,
                None,
            )
        self.assertIn("private conv window", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
