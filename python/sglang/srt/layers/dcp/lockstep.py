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
    "PrefixLensRankDivergence",
    "dcp_forces_prefix",
    "draft_extend_prefix_lens",
    "format_prefix_lens_divergence",
    "prefix_lens_ballot",
    "prefix_lens_ballot_agrees",
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


def dcp_forces_prefix(is_target_verify: bool, is_draft_extend: bool) -> bool:
    """Which extend-class forward modes ALWAYS read a committed prefix (#180).

    Both of these carry a committed context in the token-sharded pool that a
    length vector does not describe:

    * target-VERIFY -- built out of a decode batch, so ``batch.prefix_lens`` is
      never set and ``extend_prefix_lens_cpu`` is empty or None (#180).
    * draft-EXTEND (#108 slice 2) -- ``seq_lens`` already counts the newly
      accepted tokens this step appends, and the request's earlier draft
      context is in the pool. The batch likewise carries no prefix vector for
      it: ``ForwardBatch`` fills ``extend_prefix_lens`` from
      ``batch.prefix_lens``, which the draft-extend batch does not populate.

    Returning False for either would not read a short prefix -- it would skip
    the prefix stage ENTIRELY, so the draft would attend only the tokens of the
    current step and silently ignore everything before them. And because the
    prefix stage is where the Q all-gather and the LSE merge live, a
    length-based test that answers differently per rank leaves the owner of a
    short prefix sitting alone in an all-gather nobody joins -- the
    rank-local-condition-before-a-group-collective family (#94), five sightings.

    Forward-mode first, unconditionally. Both inputs are replicated: the
    forward mode is part of the batch every rank builds from the same scheduler
    state.
    """
    return bool(is_target_verify) or bool(is_draft_extend)


def draft_extend_prefix_lens(seq_lens, num_tokens_per_req: int):
    """The COMMITTED prefix length per request on a draft-extend step (#108).

    ``seq_lens`` on a draft-extend already includes the ``num_tokens_per_req``
    tokens this very step is appending -- they are written into the pool by the
    owner-rule masked write at the top of the forward. The paged prefix read
    must therefore cover ``seq_len - num_tokens_per_req``; reading the full
    ``seq_len`` would let a query attend its OWN key through the paged
    (non-causal) stage as well as through the ragged causal stage, i.e. count
    it twice in the LSE merge.

    ``num_tokens_per_req`` is constant across the batch by construction (the
    draft-extend qo layout is a fixed stride so it can be cuda-graph captured),
    which is what makes this a vector op and not a per-request rebuild.

    Clamped at zero: a request whose whole sequence is this step's tokens has
    no committed prefix, and a negative length would index backwards.
    """
    k = int(num_tokens_per_req)
    if k <= 0:
        return seq_lens
    return (seq_lens - k).clamp_(min=0)


def weightless_has_prefix(
    forces_prefix: bool,
    extend_prefix_lens_cpu: Optional[Sequence[int]],
) -> bool:
    """Does this extend-class step read a COMMITTED prefix from the KV pool?

    ``forces_prefix`` is :func:`dcp_forces_prefix` -- the forward modes whose
    committed prefix no length vector describes (target-verify, and since #108
    slice 2 draft-extend). It is named for what it MEANS rather than for the
    one mode that originally set it, because a second mode now does.

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
    if forces_prefix:
        return True
    if extend_prefix_lens_cpu is None:
        return False
    return any(extend_prefix_lens_cpu)


# ---------------------------------------------------------------------------
# #639: the DETECTOR for the premise the docstring above ASSERTS.
#
# `weightless_has_prefix` requires its inputs to be replicated and says so:
# "The answer decides whether the Q all-gather and the LSE merge are issued at
# all, so it must be identical on the head rank and on every weightless
# worker. Both inputs are replicated." The REQUIREMENT is stated; the PREMISE
# was never checked. `extend_prefix_lens_cpu` traces to
# `schedule_batch.py:2239` -- `prefix_lens = [len(r.prefix_indices) for r in
# reqs]` -- i.e. to the content of this rank's radix cache, which is the
# quantity #616B had to install a group-MIN floor to keep uniform. The same
# line at :2235 builds `extend_num_tokens`, so one vector decides both the
# SHAPE of every per-layer collective and WHICH collectives run at all.
#
# Caught live 2026-08-07 08:26 with all three ranks alive: TP0 past attention
# in `prepare_mlp`'s all_reduce, TP1 and TP2 inside `_ag_lse`'s all_gather,
# seven lines apart in one layer body. TP0 had taken `_forward_extend_dcp`'s
# `if not has_prefix: ... return` and left an LSE all-gather its peers were
# still waiting in.
#
# THIS IS A DETECTOR, NOT A CORRECTION, and that distinction must survive in
# the code. Once the prefix vector diverges the attention result is already
# wrong; making the branch uniform only makes the failure uniform and loud. It
# is deliberately NOT an OR-ballot: OR-ing the derived boolean would quietly
# adopt the position that a divergent prefix vector is legitimate and should
# be absorbed, which is the OPPOSITE of the position #616B took when it made
# that vector uniform on the paths it reached. Refusing on disagreement is
# neutral between "extend #616B's floor to this path" and "divergence is
# legal here", and it forecloses neither -- while turning a 60-second silent
# stall plus stack archaeology into one line naming the rank and its vector.
# ---------------------------------------------------------------------------


class PrefixLensRankDivergence(RuntimeError):
    """The extend prefix-length vector is not the same on every rank (#639)."""


def _prefix_lens_digest(prefix_lens: Sequence[int]) -> int:
    """A deterministic 62-bit digest of the vector.

    NOT :func:`hash`: that is salted for ``str``/``bytes`` and is a per-process
    value in exactly the situation this must compare ACROSS processes. A plain
    integer polynomial is deterministic everywhere, needs no import, and stays
    inside int64 after the negation the MIN-pair trick performs.
    """
    h = 1469598103934665603
    for n in prefix_lens:
        h = (h * 1099511628211 + (int(n) + 1)) & ((1 << 62) - 1)
    return h


def prefix_lens_ballot(prefix_lens: Sequence[int]) -> Tuple[int, int, int, int]:
    """This rank's packed ballot: ``(n, -n, digest, -digest)``.

    Packed as (x, -x) pairs so ONE MIN all_reduce yields both the group min and
    the group max of each field, which is what makes disagreement detectable
    without a second collective. The LENGTH rides alongside the digest because
    the batch size can diverge too -- a rank that admitted a different number
    of requests is the same defect one step earlier, and a digest alone would
    report it as an opaque mismatch instead of naming the count.
    """
    n = len(prefix_lens)
    d = _prefix_lens_digest(prefix_lens)
    return (n, -n, d, -d)


def prefix_lens_ballot_agrees(reduced: Sequence[int]) -> bool:
    """True when the MIN-reduced ballot shows every rank sent the same vector.

    ``reduced`` is the elementwise MIN of every rank's :func:`prefix_lens_ballot`
    output, so ``reduced[0] == -reduced[1]`` says the group agreed on the
    length and ``reduced[2] == -reduced[3]`` that it agreed on the contents.
    """
    return reduced[0] == -reduced[1] and reduced[2] == -reduced[3]


def format_prefix_lens_divergence(per_rank: Sequence[Sequence[int]]) -> str:
    """The message the detector raises with, given every rank's vector.

    Prints the vectors rather than a digest: the whole point of paying for the
    check is that the next occurrence self-diagnoses instead of costing a live
    py-spy capture of three processes inside a 60-second stall.
    """
    lines = [
        "DCP extend prefix-length vector is NOT rank-uniform (#639). "
        "`prefix_lens = [len(r.prefix_indices) for r in reqs]` "
        "(schedule_batch.py) feeds BOTH `extend_num_tokens` and "
        "`weightless_has_prefix`, so a divergent vector makes the ranks enter "
        "the per-layer TP collectives with different shapes AND makes some "
        "ranks skip `_forward_extend_dcp`'s LSE all-gather entirely "
        "(`if not has_prefix: return`). Refusing here instead of stalling.",
        "Per-rank vectors:",
    ]
    for rank, lens in enumerate(per_rank):
        vec = list(lens) if lens is not None else None
        has_prefix = bool(vec) and any(vec)
        lines.append(
            f"  rank {rank}: n={0 if vec is None else len(vec)} "
            f"sum={0 if not vec else sum(vec)} has_prefix={has_prefix} {vec}"
        )
    lines.append(
        "This is a DETECTOR, not a correction: once the vector diverges the "
        "attention result is already wrong. The open question it exists to "
        "answer is whether a divergent vector is legitimate under DCP token "
        "ownership or a gap in #616B's uniformity floor -- the vectors above "
        "are the evidence for that call."
    )
    return "\n".join(lines)


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
