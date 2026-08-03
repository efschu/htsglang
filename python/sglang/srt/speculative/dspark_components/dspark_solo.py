"""DSpark under ``--speculative-draft-placement solo``: the round contract.

Solo placement runs an ASYMMETRIC draft phase -- one rank (the HOST) owns the
whole unsharded draft, every other rank (a SHADOW) builds the draft on the
``meta`` device and never runs a draft forward. What keeps the group in step is
the ONE-BROADCAST-PER-ROUND contract: the host hands the shadows a payload that
is only integers, and every rank derives the rest of the round from it.

DFLASH satisfies that contract with the token ids alone, because its block
length is a constant (``block_size``). DSpark does not: its confidence head
truncates the block early (per-request, ``p_min``-driven), so the number of
positions the target will verify is a RUNTIME value that only the host can
compute -- it is a function of the draft hidden states, which live on the host.
Two ranks that disagree about that length disagree about the verify shape, the
graph tier and the committed prefix; the divergence is silent.

So the DSpark payload is the DFLASH payload plus the block length:

    [ graph_num_tokens | flags | bs | draft ids (bs*gamma) | verify lens (bs) ]

packed into ONE int64 tensor and sent in ONE broadcast. Packing rather than a
tuple of tensors is deliberate: ``capture_safe_tp_broadcast`` issues one
collective per tensor, and the contract the solo refusal protects is literally
"one broadcast per round".

The payload LENGTH is a pure function of rank-uniform state (``bs`` from the
batch, ``gamma`` from the draft config), so a shadow sizes its receive buffer
without hearing from the host first -- the same property DFLASH relies on for
``num_sample_tokens``.

``verify_lens`` is the confidence head's truncation length expressed the way
the rest of the DSpark path already expresses it: anchor + kept draft tokens,
so ``1 <= verify_lens[i] <= gamma + 1``. A request whose confidence collapses
at the first position still verifies its anchor.

TEMPERATURE CAVEAT (property of DSpark, not of solo placement): DSpark accepts
by confidence threshold, not by rejection sampling. At temperature 0 the arm is
output-identical to the target; above it, it is not distribution-preserving.
Solo placement neither adds nor removes that property -- it is named here
because the solo review is where a reader will look for it.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Header slots of the packed payload, in order.
_HDR_GRAPH_NUM_TOKENS = 0
_HDR_FLAGS = 1
_HDR_BS = 2
SOLO_HEADER_SLOTS = 3

FLAG_LAYOUT_PRESENT = 1 << 0
FLAG_RUN_COMPACT = 1 << 1


class DsparkSoloRound:
    """One decoded round payload. Plain class (not a msgspec Struct): it is
    built once per round on a hot path and carries live device tensors."""

    __slots__ = (
        "draft_tokens",
        "verify_lens",
        "verify_lens_cpu",
        "graph_num_tokens",
        "run_compact",
    )

    def __init__(
        self,
        *,
        draft_tokens: torch.Tensor,
        verify_lens: Optional[torch.Tensor],
        verify_lens_cpu: Optional[list[int]],
        graph_num_tokens: Optional[int],
        run_compact: bool,
    ) -> None:
        self.draft_tokens = draft_tokens
        self.verify_lens = verify_lens
        self.verify_lens_cpu = verify_lens_cpu
        self.graph_num_tokens = graph_num_tokens
        self.run_compact = run_compact

    @property
    def has_layout(self) -> bool:
        return self.verify_lens is not None


class DsparkSoloRoundCodec:
    """Packs / unpacks the per-round solo payload.

    One grow-only int64 staging buffer, reused every round. The host encodes
    into it, the collective overwrites it on the shadows, and both sides decode
    the same way -- so an encoding bug shows up as an assertion on the host,
    not as a shape mismatch three collectives later.
    """

    def __init__(self, *, gamma: int, max_bs: int, device) -> None:
        if int(gamma) < 1:
            raise ValueError(f"DSpark solo codec needs gamma >= 1, got {gamma}.")
        self.gamma = int(gamma)
        self.device = device
        self._buf: Optional[torch.Tensor] = None
        self._cap_bs = 0
        if int(max_bs) > 0:
            self._grow(int(max_bs))

    # -- buffer ----------------------------------------------------------
    def payload_numel(self, bs: int) -> int:
        return SOLO_HEADER_SLOTS + int(bs) * (self.gamma + 1)

    def _grow(self, bs: int) -> None:
        self._buf = torch.empty(
            (self.payload_numel(bs),), dtype=torch.int64, device=self.device
        )
        self._cap_bs = int(bs)

    def buffer(self, bs: int) -> torch.Tensor:
        """The rank-uniform staging view for a batch of ``bs`` requests.

        Rank-uniform by construction: ``bs`` comes from the batch (identical on
        every rank) and ``gamma`` from the draft config, so a shadow calls this
        before the broadcast and gets exactly the buffer the host filled.
        """
        bs = int(bs)
        if bs < 0:
            raise ValueError(f"DSpark solo payload needs bs >= 0, got {bs}.")
        if self._buf is None or bs > self._cap_bs:
            self._grow(max(bs, 1))
        return self._buf[: self.payload_numel(bs)]

    # -- host side -------------------------------------------------------
    def encode(
        self,
        *,
        bs: int,
        draft_tokens: torch.Tensor,
        verify_lens: Optional[torch.Tensor],
        graph_num_tokens: Optional[int],
        run_compact: bool,
    ) -> torch.Tensor:
        """Fill the staging buffer with this round's payload (host only)."""
        bs = int(bs)
        expected = (bs, self.gamma)
        if tuple(draft_tokens.shape) != expected:
            raise ValueError(
                "DSpark solo payload expects draft tokens of shape "
                f"{expected} (bs x gamma), got {tuple(draft_tokens.shape)}. "
                "The block the host broadcasts must be the block the shadows "
                "size their buffer for."
            )
        buf = self.buffer(bs)
        flags = 0
        if verify_lens is not None:
            if tuple(verify_lens.shape) != (bs,):
                raise ValueError(
                    "DSpark solo payload expects one verify length per request "
                    f"(shape ({bs},)), got {tuple(verify_lens.shape)}."
                )
            flags |= FLAG_LAYOUT_PRESENT
        if run_compact:
            flags |= FLAG_RUN_COMPACT
        buf[_HDR_GRAPH_NUM_TOKENS] = (
            -1 if graph_num_tokens is None else int(graph_num_tokens)
        )
        buf[_HDR_FLAGS] = flags
        buf[_HDR_BS] = bs
        ids_end = SOLO_HEADER_SLOTS + bs * self.gamma
        buf[SOLO_HEADER_SLOTS:ids_end].copy_(draft_tokens.reshape(-1))
        if verify_lens is not None:
            buf[ids_end:].copy_(verify_lens.reshape(-1))
        else:
            # No ragged layout this round: the block is verified whole, which
            # is exactly gamma + 1 (anchor + gamma drafts). Filling the slots
            # instead of leaving them stale keeps the payload self-describing.
            buf[ids_end:].fill_(self.gamma + 1)
        return buf

    # -- both sides ------------------------------------------------------
    def decode(self, buf: torch.Tensor, *, bs: int) -> DsparkSoloRound:
        """Unpack a payload. Validates before use: a payload the shadows
        cannot trust is a silent divergence, so it fails loudly here."""
        bs = int(bs)
        if int(buf.numel()) != self.payload_numel(bs):
            raise ValueError(
                "DSpark solo payload length mismatch: buffer has "
                f"{int(buf.numel())} slots, bs={bs} gamma={self.gamma} needs "
                f"{self.payload_numel(bs)}."
            )
        header = buf[:SOLO_HEADER_SLOTS].tolist()
        graph_num_tokens = int(header[_HDR_GRAPH_NUM_TOKENS])
        flags = int(header[_HDR_FLAGS])
        echoed_bs = int(header[_HDR_BS])
        if echoed_bs != bs:
            raise ValueError(
                "DSpark solo payload batch-size echo mismatch: the host sent a "
                f"round for bs={echoed_bs} while this rank prepared bs={bs}. "
                "The ranks disagree about the batch -- refusing rather than "
                "verifying a block that belongs to other requests."
            )
        ids_end = SOLO_HEADER_SLOTS + bs * self.gamma
        draft_tokens = buf[SOLO_HEADER_SLOTS:ids_end].view(bs, self.gamma)
        run_compact = bool(flags & FLAG_RUN_COMPACT)
        if not (flags & FLAG_LAYOUT_PRESENT):
            return DsparkSoloRound(
                draft_tokens=draft_tokens,
                verify_lens=None,
                verify_lens_cpu=None,
                graph_num_tokens=None,
                run_compact=run_compact,
            )
        verify_lens = buf[ids_end:].to(torch.int32)
        verify_lens_cpu = [int(v) for v in verify_lens.tolist()]
        validate_verify_lens(
            verify_lens_cpu,
            gamma=self.gamma,
            graph_num_tokens=graph_num_tokens if graph_num_tokens >= 0 else None,
        )
        return DsparkSoloRound(
            draft_tokens=draft_tokens,
            verify_lens=verify_lens,
            verify_lens_cpu=verify_lens_cpu,
            graph_num_tokens=graph_num_tokens if graph_num_tokens >= 0 else None,
            run_compact=run_compact,
        )


def validate_verify_lens(
    verify_lens_cpu: Sequence[int],
    *,
    gamma: int,
    graph_num_tokens: Optional[int],
) -> None:
    """The block-length invariants, checked on every rank that decodes.

    ``RaggedVerifyLayout.__post_init__`` checks the same lower bound, but only
    when a CPU list is attached and only after the layout has been assembled.
    Under solo the length arrives over the wire, so it is checked at the wire.
    """
    if min(verify_lens_cpu, default=1) < 1:
        raise ValueError(
            "DSpark solo payload carries a verify length below 1: every "
            "request verifies at least its anchor token. Got "
            f"{list(verify_lens_cpu)}."
        )
    if max(verify_lens_cpu, default=1) > gamma + 1:
        raise ValueError(
            "DSpark solo payload carries a verify length above gamma + 1 = "
            f"{gamma + 1} (anchor + drafted block). Got "
            f"{list(verify_lens_cpu)}."
        )
    if graph_num_tokens is not None and sum(verify_lens_cpu) > graph_num_tokens:
        raise ValueError(
            f"DSpark solo payload sums to {sum(verify_lens_cpu)} verify tokens "
            f"but declares graph_num_tokens={graph_num_tokens}; the shadows "
            "would rebuild a layout that does not fit its own graph tier."
        )


def committed_prefix(
    *,
    anchor_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    verify_lens: torch.Tensor,
) -> list[list[int]]:
    """The token prefix each request can commit this round, per rank.

    This is the observable the solo contract has to make rank-invariant: given
    the same payload, every rank must name the same tokens. Kept as a small
    pure function so the hermetic falsifier can assert on it without a
    scheduler, a model or a collective.
    """
    ids = torch.cat([anchor_tokens.view(-1, 1), draft_tokens], dim=1)
    lens = [int(v) for v in verify_lens.reshape(-1).tolist()]
    return [[int(t) for t in ids[i, : lens[i]].tolist()] for i in range(len(lens))]


class DsparkSoloDraftMirror:
    """The ordered collective mirror for the DSpark draft phase under solo.

    The host runs the draft; the shadows run none of it. But the draft reaches
    into the TARGET's vocab-parallel modules -- the embedding and the lm_head
    shard -- and both of those are collectives that every rank has to enter.
    This class is the ONE place that fixes their order, and both roles drive it
    through the same three calls so the sequence cannot drift apart:

        1. ``embed_block``   -- the vocab-parallel embedding all_reduce
        2. ``publish_normed``-- the host's post-norm hidden states, broadcast
        3. ``vocab_logits``  -- the vocab all_gather over each rank's lm_head
                                shard (the shadows discard the result; the
                                host's block is authoritative)

    followed by the single round payload broadcast (``DsparkSoloRoundCodec``).

    CAPTURE SAFETY: every one of these runs EAGER, outside any captured region.
    The shadows capture no draft graphs at all, so a collective recorded inside
    the host's draft graph would have no matching participant. This is why the
    graph-folded greedy proposal is disabled under solo -- its
    ``compute_base_logits`` call carries the vocab gather into the capture.
    """

    def __init__(self, *, is_host: bool, solo_rank: int, lm_head, device) -> None:
        self.is_host = bool(is_host)
        self.solo_rank = int(solo_rank)
        self.device = device
        weight = getattr(lm_head, "weight", None)
        if weight is None:
            raise RuntimeError(
                "--speculative-draft-placement solo with DSPARK requires the "
                "target lm_head to expose `weight`: both the host and the "
                "shadows compute their own vocab shard's logits from the "
                "broadcast hidden states."
            )
        self.hidden_dim = int(weight.shape[1])
        self.hs_dtype = weight.dtype
        self._hs_buf: Optional[torch.Tensor] = None
        self._hs_cap = 0

    # -- 1) embedding ----------------------------------------------------
    def embed_block(self, embed_module, block_ids: torch.Tensor):
        """Join the vocab-parallel embedding all_reduce.

        ``block_ids`` is rank-uniform (mask fill + the previous round's bonus
        token, which every rank already holds), so the shadows can make this
        call without hearing from the host.
        """
        return embed_module(block_ids)

    # -- 2) hidden-state broadcast ---------------------------------------
    def _hs_view(self, num_tokens: int) -> torch.Tensor:
        if self._hs_buf is None or self._hs_cap < num_tokens:
            cap = max(int(num_tokens), 1)
            # NOTE: no device comparison -- ``self.device`` may be the
            # indexless "cuda" string while a tensor reports "cuda:0", which
            # would reallocate every round. The worker never migrates devices.
            self._hs_buf = torch.empty(
                (cap, self.hidden_dim), dtype=self.hs_dtype, device=self.device
            )
            self._hs_cap = cap
        return self._hs_buf[:num_tokens]

    def publish_normed(
        self, tp_group, num_tokens: int, hidden_states: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Publish (host) or receive (shadow) the post-norm draft hidden
        states -- the last value produced by a DRAFT weight and the first
        consumed by a TARGET vocab shard, which is exactly where the two roles
        have to meet."""
        if num_tokens == 0 or getattr(tp_group, "world_size", 1) == 1:
            return hidden_states if hidden_states is not None else self._hs_view(0)
        buf = self._hs_view(num_tokens)
        if hidden_states is not None:
            if int(hidden_states.shape[-1]) != self.hidden_dim:
                raise RuntimeError(
                    "--speculative-draft-placement solo with DSPARK needs the "
                    "draft trunk's hidden width to match the target lm_head's "
                    f"input width, but got {int(hidden_states.shape[-1])} vs "
                    f"{self.hidden_dim}."
                )
            buf.copy_(hidden_states)
        from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

        capture_safe_tp_broadcast(tp_group, (buf,), src=self.solo_rank)
        return buf

    # -- 3) vocab gather -------------------------------------------------
    def vocab_logits(self, x: torch.Tensor, *, lm_head, use_fp32: bool):
        """Every rank contributes its lm_head shard to the vocab all_gather.
        The host keeps the result; a shadow throws it away."""
        from sglang.srt.models.dspark import logits_from_normed_hidden

        return logits_from_normed_hidden(
            x, lm_head=lm_head, use_fp32_lm_head=use_fp32, skip_vocab_gather=False
        )

    # -- 4) the round payload -------------------------------------------
    def broadcast_round(self, tp_group, buf: torch.Tensor) -> torch.Tensor:
        from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

        if getattr(tp_group, "world_size", 1) == 1:
            return buf
        capture_safe_tp_broadcast(tp_group, (buf,), src=self.solo_rank)
        return buf


def refuse_solo_nongreedy_round(sampling_info) -> None:
    """v1 limit of DSpark solo, checked per round.

    Under non-greedy acceptance ``accept_draft_tokens`` needs the draft's
    ``corrected_logits`` -- ``[bs, gamma, vocab]`` -- on every verifying rank.
    At the shipped geometry that is ~2.5 GiB per round per request, which is
    not a payload, it is a second model forward's worth of traffic. The solo
    contract exists precisely to avoid that, so the combination is refused by
    name rather than approximated.

    This is the DSpark analogue of the EAGLE ``--speculative-use-rejection-
    sampling`` refusal in ``_handle_speculative_draft_placement``; it cannot be
    a startup check because greediness is a per-request sampling property.
    """
    if sampling_info is None or sampling_info.is_all_greedy:
        return
    raise ValueError(
        "--speculative-draft-placement solo with DSPARK supports greedy "
        "acceptance only. A non-greedy request needs the draft's corrected "
        "logits [bs, gamma, vocab] on every verifying rank, while solo "
        "broadcasts only token ids and block lengths. Serve non-greedy "
        "requests with placement 'split'."
    )


def apply_solo_dspark_overrides(draft_model, *, tp_rank: int) -> None:
    """Turn off the draft-model optimizations the solo round contract cannot
    carry. Called once, before the first collective.

    ``SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD`` defaults to ON: it shards
    ``markov_w2`` over the lm_head TP group, skips the vocab all_gather in
    ``_logits_from_x_post_hc`` and all-gathers per markov STEP instead. Under
    solo that is not a slower path, it is a hang: the markov head exists only
    on the host, the shadows have no shard to contribute, and the host's
    skipped gather is the one the shadows are sitting in.

    Disabled rather than refused, because it is an optimization the user did
    not ask for by name -- refusing a default-on flag would make solo look
    unsupported. The reason is logged once from rank 0.
    """
    if not getattr(draft_model, "_opt_markov_w2_tp_shard", False):
        return
    draft_model._opt_markov_w2_tp_shard = False
    markov_head = getattr(draft_model, "markov_head", None)
    if markov_head is not None:
        markov_head._opt_markov_w2_tp_shard = False
        markov_head._tp_shard = None
    if tp_rank == 0:
        logger.warning(
            "DSpark draft-solo placement: disabling "
            "SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD. It shards markov_w2 over "
            "the lm_head TP group and replaces the single vocab all_gather "
            "with a per-step one, but under solo the markov head lives only "
            "on the host rank -- the shadows hold no shard, so the per-step "
            "gather has no participant. The draft falls back to the single "
            "gather the shadows mirror."
        )
