# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================

"""Pure lockstep decisions for the weightless-KV fast lane (#121) and for chain
speculation on top of it (#143).

THE LANE'S ONE INVARIANT, stated in one place:

    For a weightless worker, the NUMBER, ORDER and OP-TAGS of DCP-group
    collectives in a forward step are a function of
    ``(forward_mode, has_prefix, num_full_attention_layers)`` only.
    ``accept_len`` is NOT an input.

Everything in this module is a pure function of replicated (rank-uniform)
quantities: no device, no process group, no ModelRunner, no server_args. That
is deliberate -- the decisions that make the lane hang or not hang are then
pinnable on CPU, in the style of ``layers/dcp/owner.py``.

Why a module and not comments: the verify-first prefix rule used to live as two
verbatim copies inside ``flashinfer_backend.py`` (head and weightless worker).
That duplication is precisely the drift surface of the D5 defect family
([[rank-lokaler-test-vor-kollektiv]], five sightings) -- a rank-local condition
deciding whether a group collective is entered. One expression, two callers.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

__all__ = [
    "AG_HEADS_TAG_PREFIX",
    "LSE_MERGE_TAG",
    "weightless_has_prefix",
    "weightless_layer_op_tags",
    "weightless_step_op_tags",
    "chain_spec_verify_rows",
    "spec_accept_broadcast_shapes",
]


# The two op-tags ``layers/dcp/comm.py`` hands to ``guard_dcp_step``. They are
# restated here (not imported) because comm.py builds them inline inside the
# collective wrappers; ``test_weightless_chain_spec.py`` pins these constants
# against comm.py's source so a rename there fails loudly instead of drifting.
AG_HEADS_TAG_PREFIX = "ag_heads:"
LSE_MERGE_TAG = "lse_merge"


def weightless_has_prefix(
    is_target_verify: bool,
    extend_prefix_lens_cpu: Optional[Sequence[int]],
) -> bool:
    """Does this extend-class step read a COMMITTED prefix from the KV pool?

    The answer decides whether the Q all-gather and the LSE merge are issued at
    all, so it must be identical on the head rank and on every weightless
    worker. Both inputs are replicated: ``forward_mode`` is part of the batch
    every rank builds from the same scheduler state, and
    ``extend_prefix_lens_cpu`` is a host-side length vector, not a per-rank
    tensor.

    FORWARD-MODE FIRST, unconditionally. A target-verify batch carries NO
    prefix lengths -- ``forward_batch_info`` fills them from
    ``batch.prefix_lens``, which a verify batch built out of a decode batch
    never sets -- so a length-based test would fall through to whatever comes
    after it. Under the owner rule "whatever comes after it" has historically
    been a rank-local slot count, and a short committed prefix over an uneven
    token vector is routinely owned entirely by one rank: the others return
    early and the owner sits alone in an all-gather nobody joins. Verify ALWAYS
    has a committed prefix (``seq_lens >= 1``), and even if it did not, forcing
    the branch costs two collectives over an empty read, which the LSE merge
    already handles as ``out=0 / lse=-inf``.

    This is #180's ``force_prefix`` rule and the Triton twin's
    ``_dcp_batch_has_prefix`` ordering, expressed once for the flashinfer lane.
    """
    if is_target_verify:
        return True
    if extend_prefix_lens_cpu is None:
        return False
    return any(extend_prefix_lens_cpu)


def weightless_layer_op_tags(
    is_decode: bool,
    has_prefix: bool,
    kv_head_total: int,
    q_head_total: int,
) -> Tuple[str, ...]:
    """The guarded DCP step tags one full-attention layer issues, in order.

    Mirrors ``FlashInferAttnBackend.forward_decode_weightless_worker`` /
    ``forward_extend_weightless_worker`` and their head-side counterparts
    ``_forward_decode_dcp`` / ``_forward_extend_dcp``:

      * fused K+V all-gather (#132 stacks k and v into ONE collective)
      * Q all-gather                       -- prefix stages only
      * LSE merge (all-gather + all-reduce) -- prefix stages only

    ``lse_merge`` is ONE guarded step carrying TWO NCCL ops; the guard counts
    steps, NCCL counts ops, and both must agree between ranks. The head-local
    ragged current-chunk attention issues nothing and does not appear here.

    Note the identity this function makes visible: a DECODE layer and a
    prefix-carrying EXTEND/VERIFY layer emit the SAME tuple. That is what makes
    a captured verify graph and a captured decode graph pair up on the
    communicator, and it is the reason #143 needs no new collective kind.
    """
    ag_kv = f"{AG_HEADS_TAG_PREFIX}{kv_head_total}"
    ag_q = f"{AG_HEADS_TAG_PREFIX}{q_head_total}"
    if is_decode:
        return (ag_kv, ag_q, LSE_MERGE_TAG)
    if not has_prefix:
        return (ag_kv,)
    return (ag_kv, ag_q, LSE_MERGE_TAG)


def weightless_step_op_tags(
    is_idle: bool,
    is_decode: bool,
    has_prefix: bool,
    num_full_attention_layers: int,
    kv_head_total: int,
    q_head_total: int,
) -> Tuple[str, ...]:
    """The whole step's guarded DCP step sequence.

    ``is_idle`` short-circuits to the empty tuple: on an IDLE (padding) batch
    the head's decoder layers skip attention entirely, so the worker must issue
    nothing either (``ModelRunner._forward_weightless_worker``'s early return).

    ACCEPT-INDEPENDENCE: none of the parameters is derivable from an accept
    length. ``is_decode`` / ``is_idle`` come from the forward mode, which under
    chain spec is TARGET_VERIFY on every generation step regardless of what was
    accepted; ``has_prefix`` is forced True for verify; the head counts and the
    layer count are boot geometry. An accept length changes ``seq_lens`` and
    ``kv_committed_len`` -- payload VALUES -- and nothing else.
    """
    if is_idle:
        return ()
    per_layer = weightless_layer_op_tags(
        is_decode=is_decode,
        has_prefix=has_prefix,
        kv_head_total=kv_head_total,
        q_head_total=q_head_total,
    )
    return per_layer * int(num_full_attention_layers)


def chain_spec_verify_rows(batch_size: int, num_draft_tokens: int) -> int:
    """Query rows a chain (topk == 1) verify step attends: ``bs * (k + 1)``.

    ``num_draft_tokens`` is already ``k + 1`` -- ``build_tree_kernel_efficient``
    prepends the bonus token to the k-token chain -- so this is a plain product,
    stated as a function only so the accept-independence is testable: the row
    count is the same after accepting 0 drafts as after accepting all k.
    """
    if batch_size < 0:
        raise ValueError(f"batch_size must be >= 0, got {batch_size}")
    if num_draft_tokens < 1:
        raise ValueError(
            f"a verify step needs num_draft_tokens >= 1, got {num_draft_tokens}"
        )
    return batch_size * num_draft_tokens


def spec_accept_broadcast_shapes(
    batch_size: int,
    num_draft_tokens: int,
    max_tree_depth: int,
):
    """Shapes of the three tensors the rank-0 accept broadcast carries.

    Returns ``(predict_shape, accept_index_shape, num_correct_drafts_shape)``,
    matching what ``eagle_sample`` allocates before the accept kernel:
    ``predict`` is flat over all query rows, ``accept_index`` is
    ``[bs, max_tree_depth]`` (``spec_steps + 1`` for a chain), and
    ``num_correct_drafts`` is ``[bs]``.

    A weightless worker has no logits and therefore cannot allocate these by
    running the accept kernel -- it allocates them from this pure geometry and
    enters the same broadcast. Every input is boot-fixed or scheduler state, so
    head and worker agree without communicating; a shape disagreement here
    would be an unrecoverable mismatch inside the broadcast rather than a hang.
    """
    rows = chain_spec_verify_rows(batch_size, num_draft_tokens)
    if max_tree_depth < 1:
        raise ValueError(f"max_tree_depth must be >= 1, got {max_tree_depth}")
    return ((rows,), (batch_size, max_tree_depth), (batch_size,))
