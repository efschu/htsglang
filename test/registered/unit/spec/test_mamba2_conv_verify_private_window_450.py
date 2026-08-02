# SPDX-License-Identifier: Apache-2.0
"""#450a: the Mamba2 mixer's TARGET_VERIFY conv must not publish an
uncommitted window.

THE TWIN
#444 removed mutate-then-repair from ``GDNAttnBackend.forward_extend`` and
named the site it did NOT treat: ``MambaMixer2.forward``'s own TARGET_VERIFY
conv, which handed the SAME kernel the SAME persistent pool row with the same
intermediate cache. This file is the falsifier for that twin, mirroring
``test_conv_verify_private_window_444.py`` deliberately: same defect, same
remedy, same shape of proof.

THE DEFECT, as it stood
``_causal_conv1d_update_kernel`` is intentionally in-place. STEP 2 stores the
shifted sliding window back into whatever ``conv_state`` it was handed
(``layers/attention/mamba/causal_conv1d_triton.py:752``) and that store is NOT
gated on ``SAVE_INTERMEDIATE`` -- the speculative branch only moves the source
offset (``:715``). ``MambaMixer2.forward`` passed ``layer_cache.conv[0]``, the
PERSISTENT pool, so a verify forward advanced that row over every drafted step,
accepted or not. The row only became correct again afterwards, when
``update_mamba_state_after_mtp_verify``
(``layers/attention/hybrid_linear_attn_backend.py``) scattered the accepted
step back out of the intermediate cache.

That is mutate-then-repair. Between the two points the pool row holds the LAST
CANDIDATE's window -- a value that is neither the pre-verify state nor the
committed one. ``TestTheWindowIsRealAndObservable`` measures exactly that.

THE FIX
``MambaMixer2._target_verify_conv`` runs the verify conv on a request-private
copy of the window. The buffer is owned by ``Mamba2AttnBackend`` (one object
for all mamba layers, allocated in ``__init__``) and handed down through
``MambaMixer2.forward(verify_conv_window=...)``, because the mixers themselves
are per-layer modules and must not each hold a copy.

WHAT FAILS WITHOUT THE FIX
``TestPersistentRowIsNotPublishedDuringVerify`` -- with ``_target_verify_conv``
reverted to handing ``conv_state`` / ``state_indices_tensor_d`` to the kernel,
the pool row read at the seam is not the pre-verify row.

WHAT MUST NOT MOVE
``TestCommittedBytesAreUnchanged`` pins byte-neutrality against the legacy
mutate-then-repair sequence: the committed pool and the conv OUTPUT are
element-identical, because the commit's scatter overwrites every element of the
row it touches and the kernel sees the same prior state either way.
``TestRefusalWhenNoPrivateWindow`` pins that a caller which never allocated the
buffer gets a named refusal rather than the pool row.

FIDELITY OF THE MODEL
No CUDA here, so the Triton kernel is replaced by ``_reference_conv_update``,
transcribed INDEPENDENTLY from the kernel's own index arithmetic (a second
transcription that agrees with #444's is worth more than a shared one):

  * STEP 2 store -- ``new_state[:, i] = old_state[:, i + seqlen]`` while
    ``i + seqlen < state_len``, else ``x[:, i + seqlen - state_len]``
    (``causal_conv1d_triton.py:705-752``), i.e. the last ``state_len`` columns
    of ``cat([state, x])``.
  * ``SAVE_INTERMEDIATE`` -- step ``t`` stores the ``K-1`` most recently
    consumed inputs, oldest first, into ``[slot, step, dim, win]``
    (``:930-960``), i.e. ``cat([state, x])[:, t+1 : t+K]``.

``_reference_commit`` transcribes
``_fused_conv_window_scatter_with_mask_kernel``
(``mamba_state_scatter_triton.py:305-372``): destination row from
``dst_indices_raw``, source row from the request POSITION, whole
``dim x (K-1)`` window copied, entries with ``step < 0`` skipped.

THE ROWS ARE BATCH POSITIONS, ON PURPOSE
``MambaMixer2`` builds ``intermediate_state_indices = arange(num_decodes)``, so
the private rows are batch positions 0..bs-1, exactly the rows the commit reads
back (``src_idx = pid_req``). The private buffer therefore inherits the
intermediate cache's ownership rule instead of introducing a second one; see
``TestPrivateRowsFollowTheIntermediateCache``. What that rule is worth across
concurrently verifying runners is #450b, answered in
``test_verify_intermediate_row_ownership_450.py``.
"""

import unittest
from types import SimpleNamespace
from typing import List, Optional
from unittest import mock

import torch

from sglang.srt.layers.attention.mamba import mamba as mamba_mod
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Geometry of the modelled verify: two decode requests, a width-4 conv (window
# three columns wide), four drafted tokens.
DIM = 6
KERNEL_WIDTH = 4
WIN = KERNEL_WIDTH - 1
DRAFT_TOKENS = 4
BATCH = 2
POOL_ROWS = 7
SPEC_ROWS = 5
DTYPE = torch.float32


class _ConvKernelSpy:
    """CPU stand-in for ``causal_conv1d_update_triton`` with the kernel's
    contract.

    ``calls`` records, per invocation, the ``conv_state`` object it was handed
    -- which is the whole question this file asks.
    """

    def __init__(self):
        self.calls: List[torch.Tensor] = []
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
        conv_states[dst] = intermediate_conv_window[pos, step]


def _fixture(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    conv_state = torch.randn(POOL_ROWS, DIM, WIN, generator=gen, dtype=DTYPE)
    intermediate = torch.zeros(SPEC_ROWS, DRAFT_TOKENS, DIM, WIN, dtype=DTYPE)
    x = torch.randn(BATCH, DIM, DRAFT_TOKENS, generator=gen, dtype=DTYPE)
    mixer = SimpleNamespace(
        conv1d=SimpleNamespace(bias=torch.randn(DIM, generator=gen, dtype=DTYPE)),
        activation="silu",
    )
    conv_weights = torch.randn(DIM, KERNEL_WIDTH, generator=gen, dtype=DTYPE)
    # Deliberately not 0..BATCH-1: the mamba slots a request owns and the spec
    # rows the intermediates use are different index spaces (the slot pool is
    # sized `max_mamba_cache_size`, the spec rows `max_num_reqs`), and the fix
    # must not conflate them.
    state_indices_d = torch.tensor([4, 1], dtype=torch.int32)
    # Exactly what MambaMixer2.forward builds: arange(num_decodes).
    spec_indices = torch.arange(BATCH, dtype=torch.int32)
    return (
        conv_state,
        intermediate,
        x,
        mixer,
        conv_weights,
        state_indices_d,
        spec_indices,
    )


def _run_fixed(
    scratch,
    mixer,
    x,
    conv_state,
    conv_weights,
    state_indices_d,
    intermediate,
    spec_indices,
    spy,
):
    """Drive the production method with the kernel replaced by ``spy``."""
    with mock.patch.object(mamba_mod, "causal_conv1d_update_triton", spy, create=True):
        return MambaMixer2._target_verify_conv(
            mixer,
            x,
            conv_state,
            conv_weights,
            state_indices_d,
            intermediate,
            spec_indices,
            None,
            None,
            None,
            scratch,
        )


class TestTheWindowIsRealAndObservable(CustomTestCase):
    """The defect, measured: mutate-then-repair publishes a third state.

    Always green -- it describes the legacy sequence rather than the fix, and
    it is what makes the fix's assertion mean something. If this ever goes red
    the kernel contract changed and the rest of the file is stale.
    """

    def test_seam_read_differs_from_both_endpoints(self):
        conv_state, intermediate, x, _, weights, state_d, spec = _fixture()
        before = conv_state.clone()

        # Legacy call shape: the kernel is handed the persistent pool.
        _reference_conv_update(
            x,
            conv_state,
            weights,
            None,
            "silu",
            conv_state_indices=state_d,
            intermediate_conv_window=intermediate,
            intermediate_state_indices=spec,
        )
        at_seam = conv_state.clone()

        # Accept one drafted token for req 0, three for req 1.
        accepted = torch.tensor([0, 2], dtype=torch.int64)
        _reference_commit(conv_state, intermediate, state_d, accepted)
        after = conv_state.clone()

        for pos in range(BATCH):
            row = int(state_d[pos])
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
            torch.equal(at_seam[int(state_d[0])], intermediate[0, DRAFT_TOKENS - 1]),
            "the published row is not the last candidate's window",
        )

    def test_rows_the_verify_does_not_own_are_untouched(self):
        conv_state, intermediate, x, _, weights, state_d, spec = _fixture()
        before = conv_state.clone()
        _reference_conv_update(
            x,
            conv_state,
            weights,
            None,
            "silu",
            conv_state_indices=state_d,
            intermediate_conv_window=intermediate,
            intermediate_state_indices=spec,
        )
        owned = {int(i) for i in state_d}
        for row in range(POOL_ROWS):
            if row in owned:
                continue
            self.assertTrue(torch.equal(conv_state[row], before[row]))


class TestPersistentRowIsNotPublishedDuringVerify(CustomTestCase):
    """The falsifier. Red with ``_target_verify_conv`` reverted to the pool
    row."""

    def test_pool_is_byte_identical_at_the_seam(self):
        conv_state, intermediate, x, mixer, weights, state_d, spec = _fixture()
        before = conv_state.clone()
        scratch = torch.zeros(SPEC_ROWS, DIM, WIN, dtype=DTYPE)

        spy = _ConvKernelSpy()
        seen: List[torch.Tensor] = []
        spy.seam_reader = lambda: seen.append(conv_state.clone())

        _run_fixed(
            scratch, mixer, x, conv_state, weights, state_d, intermediate, spec, spy
        )

        self.assertEqual(len(seen), 1)
        self.assertTrue(
            torch.equal(seen[0], before),
            "a reader between the verify conv and the commit saw a pool row "
            "that no accept decision had authorised",
        )
        self.assertTrue(torch.equal(conv_state, before))

    def test_the_kernel_is_handed_the_private_buffer(self):
        conv_state, intermediate, x, mixer, weights, state_d, spec = _fixture()
        scratch = torch.zeros(SPEC_ROWS, DIM, WIN, dtype=DTYPE)
        spy = _ConvKernelSpy()
        _run_fixed(
            scratch, mixer, x, conv_state, weights, state_d, intermediate, spec, spy
        )
        self.assertEqual(len(spy.calls), 1)
        self.assertIs(spy.calls[0], scratch)
        self.assertIsNot(spy.calls[0], conv_state)

    def test_private_rows_are_seeded_with_the_persistent_window(self):
        """The prior state is an INPUT to the conv; a private copy that is not
        seeded would silently change the arithmetic rather than the
        exposure."""
        conv_state, intermediate, x, mixer, weights, state_d, spec = _fixture()
        scratch = torch.zeros(SPEC_ROWS, DIM, WIN, dtype=DTYPE)
        seeded: List[torch.Tensor] = []

        spy = _ConvKernelSpy()
        inner = spy.__call__

        def capture(*args, **kwargs):
            seeded.append(args[1].clone())
            return inner(*args, **kwargs)

        _run_fixed(
            scratch, mixer, x, conv_state, weights, state_d, intermediate, spec, capture
        )

        for pos in range(BATCH):
            row = int(spec[pos])
            self.assertTrue(
                torch.equal(seeded[0][row], conv_state[int(state_d[pos])]),
                f"private row {row} was not seeded from pool row "
                f"{int(state_d[pos])}",
            )


class TestPrivateRowsFollowTheIntermediateCache(CustomTestCase):
    """The private buffer must be indexed by the SAME rows as the intermediate
    conv window, or the commit reads a window the conv never wrote."""

    def test_kernel_gets_one_row_space_not_two(self):
        conv_state, intermediate, x, mixer, weights, state_d, spec = _fixture()
        scratch = torch.zeros(SPEC_ROWS, DIM, WIN, dtype=DTYPE)
        recorded = {}

        def capture(*args, **kwargs):
            recorded["conv_state_indices"] = kwargs["conv_state_indices"]
            recorded["intermediate_state_indices"] = kwargs[
                "intermediate_state_indices"
            ]
            return _ConvKernelSpy()(*args, **kwargs)

        _run_fixed(
            scratch, mixer, x, conv_state, weights, state_d, intermediate, spec, capture
        )
        self.assertIs(
            recorded["conv_state_indices"], recorded["intermediate_state_indices"]
        )
        self.assertTrue(torch.equal(recorded["conv_state_indices"], spec))
        # And the mamba SLOT indices are NOT what the private buffer is keyed
        # by -- conflating them would index a buffer sized by spec rows with a
        # slot id.
        self.assertFalse(torch.equal(recorded["conv_state_indices"], state_d))


class TestCommittedBytesAreUnchanged(CustomTestCase):
    """Byte-neutrality: only the publication of the window changed."""

    def _legacy(self, accepted: torch.Tensor):
        conv_state, intermediate, x, mixer, weights, state_d, spec = _fixture()
        out = _reference_conv_update(
            x,
            conv_state,
            weights,
            mixer.conv1d.bias,
            mixer.activation,
            conv_state_indices=state_d,
            intermediate_conv_window=intermediate,
            intermediate_state_indices=spec,
        )
        _reference_commit(conv_state, intermediate, state_d, accepted)
        return conv_state, out

    def _fixed(self, accepted: torch.Tensor):
        conv_state, intermediate, x, mixer, weights, state_d, spec = _fixture()
        scratch = torch.zeros(SPEC_ROWS, DIM, WIN, dtype=DTYPE)
        spy = _ConvKernelSpy()
        out = _run_fixed(
            scratch, mixer, x, conv_state, weights, state_d, intermediate, spec, spy
        )
        _reference_commit(conv_state, intermediate, state_d, accepted)
        return conv_state, out

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
        conv_state, intermediate, x, mixer, weights, state_d, spec = _fixture()
        _reference_conv_update(
            x,
            conv_state,
            weights,
            mixer.conv1d.bias,
            mixer.activation,
            conv_state_indices=state_d,
            intermediate_conv_window=intermediate,
            intermediate_state_indices=spec,
        )
        accepted = torch.tensor([1, 3], dtype=torch.int64)
        _reference_commit(conv_state, intermediate, state_d, accepted)
        for pos in range(BATCH):
            self.assertTrue(
                torch.equal(
                    conv_state[int(state_d[pos])],
                    intermediate[pos, int(accepted[pos])],
                )
            )


class TestRefusalWhenNoPrivateWindow(CustomTestCase):
    def test_missing_private_buffer_is_a_named_refusal(self):
        conv_state, intermediate, x, mixer, weights, state_d, spec = _fixture()
        with self.assertRaises(AssertionError) as ctx:
            _run_fixed(
                None,
                mixer,
                x,
                conv_state,
                weights,
                state_d,
                intermediate,
                spec,
                _ConvKernelSpy(),
            )
        self.assertIn("private conv window", str(ctx.exception))


class TestBackendOwnsOneBufferForAllLayers(CustomTestCase):
    """The buffer lives on ``Mamba2AttnBackend`` (one per model runner), not on
    the per-layer mixers: a per-layer copy would multiply the allocation by the
    mamba layer count for no correctness gain, since the layers run in sequence
    on one stream."""

    def test_backend_declares_the_window_and_forward_passes_it(self):
        import inspect

        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            Mamba2AttnBackend,
        )

        init_src = inspect.getsource(Mamba2AttnBackend.__init__)
        self.assertIn("self.verify_conv_window", init_src)
        fwd_src = inspect.getsource(Mamba2AttnBackend.forward)
        self.assertIn("verify_conv_window=self.verify_conv_window", fwd_src)
        self.assertIn(
            "verify_conv_window",
            inspect.signature(MambaMixer2.forward).parameters,
        )


if __name__ == "__main__":
    unittest.main()
