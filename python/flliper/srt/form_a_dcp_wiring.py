# SPDX-License-Identifier: Apache-2.0
"""#239 S3c: the Form A x token-cut wiring ModelRunner delegates to.

model_runner.py is a frozen core file (orchestration only). The three
decisions the worker half of the token cut needs live here:

* which attention backend a Form A worker builds (``form_a_worker_attn_backend``);
* which LSE merge the boot gate declares for this rank (``form_a_dcp_merge_of``);
* the per-layer attention step the worker route runs (``form_a_worker_attention_step``);
* the eager forward's attention metadata (``prepare_form_a_worker_eager_forward``).

Each is decided from the object that RUNS the attention (the backend), so a
rank whose backend did not take the cut disagrees at the boot gate instead of
parking in a collective the host issues alone.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

FORM_A_WORKER = "form_a_worker"
FORM_A_WORKER_QSA_DCP = "form_a_worker_qsa_dcp"


@dataclass(frozen=True)
class FormAWorkerAttn:
    """The backend a Form A worker attends through, and its per-mode name
    stamp (normally resolved inside ``_get_attention_backend``, which a worker
    never calls -- fnFA9)."""

    name: str
    backend: object


def form_a_worker_owns_token_cut(*, is_draft_worker: bool, server_args) -> bool:
    """Under Form A x the token cut (``--form-a-dcp-vector``) a worker OWNS a
    token range of the full-attention KV. The draft is the host's alone
    (solo placement), so a draft runner never does."""
    return not is_draft_worker and bool(server_args.form_a_dcp_vector())


def form_a_worker_attn_backend(model_runner) -> FormAWorkerAttn:
    """The attention backend of a Form A worker rank.

    Without the token cut (FORM A fnFA7 20.09.): a worker has no attention
    layers and a head share of 0 -- a real backend cannot even be constructed
    (flashinfer's should_use_tensor_core divides by the kv-head count). The
    worker backend answers the per-forward bookkeeping as no-ops and refuses
    any real attention by name.

    Under the token cut (#239 S3c): the worker builds the QSA full-attention
    backend alone -- not the hybrid wrapper: the GDN layers are the host's, a
    worker has no linear-attention state to plan for. Its Form A branch (S3b,
    ``_init_form_a_dcp``) takes the worker's zero-head geometry; the worker
    route drives it per layer. A backend that did not take the geometry dies
    here by name: the host would issue A/T/Q/M per full-attention layer that
    this rank never joins.

    Takes the runner because both backends' constructor contract does; it
    only reads it and returns the choice, the runner assigns it.
    """
    from flliper.srt.form_a_construction import FormAWorkerAttnBackend

    if not form_a_worker_owns_token_cut(
        is_draft_worker=model_runner.is_draft_worker,
        server_args=model_runner.server_args,
    ):
        return FormAWorkerAttn(FORM_A_WORKER, FormAWorkerAttnBackend(model_runner))

    from flliper.srt.layers.attention.qsa.config import is_qwen_qsa
    from flliper.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnBackend,
    )

    if not is_qwen_qsa(model_runner.model_config.hf_config):
        raise RuntimeError(
            "#239 S3c: the Form A token cut is wired for QSA full "
            "attention only; this model has none."
        )
    backend = QwenSparseAttnBackend(model_runner)
    if backend.form_a_dcp is None:
        raise RuntimeError(
            "#239 S3c: a Form A worker under the token cut built an "
            f"attention backend ({type(backend).__name__}) that "
            "did not take the Form A DCP geometry; the host would issue "
            "A/T/Q/M per full-attention layer that this rank never joins."
        )
    return FormAWorkerAttn(FORM_A_WORKER_QSA_DCP, backend)


def form_a_dcp_merge_of(attn_backend) -> Optional[str]:
    """The LSE merge ('ar' | 'a2a') this rank's full-attention layers declare
    at the boot gate, or None when its backend did not take the token cut."""
    from flliper.srt.layers.attention.qsa.form_a_dcp import form_a_dcp_of

    if form_a_dcp_of(attn_backend) is None:
        return None
    from flliper.srt.layers.dcp.comm import lse_merge_mode

    return lse_merge_mode()


def prepare_form_a_worker_eager_forward(attn_backend, forward_batch) -> None:
    """#239 M1s (rc12z30g D-only, D log 22:07:33): build THIS forward's
    attention metadata on a Form A worker's EAGER forward.

    The host builds it per forward in its eager runner
    (``runner/eager_runner.py`` -> ``attn_backend.init_forward_metadata``);
    the worker's eager entry (``ModelRunner._forward_form_a_worker``) bypasses
    that runner, and the backend's lazy ``_resolve_metadata`` only builds when
    ``forward_metadata is None``. So the first extend with a prefix (the second
    4096-token chunk of the S1 probe) resolved the host's 4096 top-k rows
    against the LAST graph capture/replay metadata -- ``QSA top-k rows do not
    match query rows`` on TP1/TP2 (``rows_resolve.py`` Tq-check). A stale table
    of the same row count (an eager decode after a graph replay) attends the
    wrong KV rows without any error; that is the same bug, silent.

    Only forwards whose worker joins the rows path need it (``form_a_attends``:
    decode, target verify, an extend with a prefix); the graph path gets its
    metadata from the decode-graph runner's capture/replay, never from here."""
    if getattr(attn_backend, "form_a_dcp", None) is None:
        return
    from flliper.srt.layers.attention.qsa.form_a_dcp import form_a_attends

    if form_a_attends(forward_batch):
        attn_backend.init_forward_metadata(forward_batch)


def form_a_worker_attention_step(
    attn_backend, forward_batch, token_to_kv_pool
) -> Tuple[Optional[Callable[[int], None]], Tuple[int, ...]]:
    """The worker route's per-layer attention step and the layers it runs on.

    Under the token cut the worker joins every full-attention layer's
    A [, T, Q, M] through the QSA backend's worker step. On every other Form A
    boot ``form_a_dcp`` is None (the worker backend refuses attention by name)
    and the route runs no attention: (None, ())."""
    if attn_backend.form_a_dcp is None:
        return None, ()

    def attention(layer_id: int) -> None:
        # #239 S4b part 5: this worker's owned rows of an adopted tail page
        # land before its first attention read of the extend (no-op otherwise)
        from flliper.srt.pdflip import tail_adopt

        if tail_adopt.PENDING_INSTALLS:
            tail_adopt.install_worker_rows(layer_id)
        attn_backend.form_a_worker_attention(forward_batch, layer_id)

    return attention, tuple(token_to_kv_pool.full_attention_layer_id_mapping)
