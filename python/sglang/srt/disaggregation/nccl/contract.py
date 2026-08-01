# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""What the two instances must agree on before a byte moves (#111).

Three separable decisions, all pure functions of replicated facts, all made
ABOVE the link so no link member can redecide them:

* :class:`TransportIdentity` -- the compatibility handshake. Model identity,
  KV byte format and parallel geometry, compared field by field with a named
  diff. This is where #241's lesson lives: a key or a handshake that omits the
  KV dtype lets a decode instance accept pages written in another byte format,
  and the symptom is wrong tokens, not an error.
* :attr:`MessageClass` -- which class a payload belongs to, so a deployment can
  pin classes to different nets the way ``--collective-net-small`` /
  ``--collective-net-bulk`` already do for collectives (#240/#244/#263).
* :func:`resolve_route` -- direct versus store. #212 paid for the finding that
  the store route is USELESS for hybrid GDN models; encoding that here means a
  hybrid deployment cannot be configured into the route that silently does
  nothing.

Nothing in this module imports torch or touches a device, so all of it is
decidable and testable on a card-less host -- which is the point: these are
exactly the decisions whose failure mode is silent.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, List, Optional, Sequence, Tuple


class MessageClass(str, enum.Enum):
    """What kind of payload a transfer carries.

    The split is by SIZE REGIME and by CRITICALITY, matching the existing
    collective-net vocabulary rather than inventing a second one: ``--collective
    -net-small`` and ``--collective-net-bulk`` already divide traffic that way,
    and #240 pins nets per class. A deployment that has a fast dedicated link
    and a slow management LAN wants the same division here.
    """

    #: The KV pages themselves: the bulk of the payload, latency-tolerant,
    #: throughput-bound. Belongs on the fast wire.
    KV_BULK = "kv_bulk"
    #: Component state (mamba/GDN slot, SWA, DSA...). Small per request but
    #: MANDATORY for hybrid models -- see resolve_route().
    STATE = "state"
    #: Aux metadata (indices, lengths, the completion word). Small, latency-
    #: sensitive, and the thing that goes first.
    AUX_SMALL = "aux_small"

    @property
    def is_bulk(self) -> bool:
        """Whether this class should follow ``--collective-net-bulk``."""
        return self in (MessageClass.KV_BULK, MessageClass.STATE)


def net_for_class(
    message_class: MessageClass,
    *,
    net_small: Optional[str],
    net_bulk: Optional[str],
) -> Optional[str]:
    """Which net a class rides, following the existing per-class pinning.

    Returns ``None`` when the deployment pinned nothing, which means "let the
    transport choose" -- the unchanged default. Deliberately the same two knobs
    as the collective path so an operator does not have to learn a second
    vocabulary for the same physical decision (#240).
    """
    return net_bulk if message_class.is_bulk else net_small


class Route(str, enum.Enum):
    """How the KV reaches the decode instance."""

    #: Instance to instance, over a :class:`~.link.KvLink`. Carries KV AND the
    #: state components in one plan.
    DIRECT = "direct"
    #: Through a HiCache storage tier: prefill writes, decode reads.
    STORE = "store"


class RouteUnavailable(RuntimeError):
    """A route was requested that cannot work for this model."""


def resolve_route(
    requested: Route,
    *,
    is_hybrid_gdn: bool,
    has_state_components: bool,
) -> Route:
    """Pick the route, refusing the one that silently does nothing (#212).

    THE PAID-FOR FINDING, in one sentence: for a hybrid GDN model a store round
    trip carries KV pages only, while the recurrent state lives in a separate
    pool, and ``MambaRadixCache._match_post_processor`` truncates any prefix
    match to the deepest node that owns a mamba checkpoint -- so a KV-only
    import matches ZERO tokens and the decode side recomputes the whole prompt.
    It looks like it works right up to the point where it silently does
    nothing, which is why this is a hard refusal and not a warning.

    The direct route is what PD already does correctly: ``setup_state_kv_args``
    appends a ``StateType.MAMBA`` component and the state moves WITH the KV.

    For a dense (non-hybrid) model the store route is viable and is returned
    unchanged.
    """
    if requested is Route.DIRECT:
        return Route.DIRECT
    if is_hybrid_gdn:
        raise RouteUnavailable(
            "route 'store' cannot carry a hybrid GDN model: the store round "
            "trip moves KV pages only, while the recurrent state lives in a "
            "separate pool, and MambaRadixCache truncates any prefix match to "
            "the deepest node owning a mamba checkpoint -- so the decode side "
            "matches zero tokens and recomputes the whole prompt while "
            "appearing to work (#212). Use route 'direct', which moves the "
            "StateType.MAMBA component together with the KV."
        )
    if has_state_components:
        raise RouteUnavailable(
            "route 'store' cannot carry a model with state components "
            f"({has_state_components}): the store tier is KV-page addressed "
            "and would drop them silently. Use route 'direct'."
        )
    return Route.STORE


@dataclasses.dataclass(frozen=True)
class TransportIdentity:
    """Everything two instances must agree on for a transfer to mean anything.

    Built on BOTH sides from local facts only -- no collective decides it, so
    a mismatch is discovered by comparison at handshake time rather than by a
    group that half-joined (the #94 family). The bootstrap exchange carries
    the remote copy; :meth:`assert_compatible` does the comparing.
    """

    #: sha256 over (model_path, revision, dtype, quantization, kv_cache_dtype),
    #: from mem_cache.hicache_storage.compute_model_identity_hash. REUSED, not
    #: re-derived: #241 established that recipe and upstream is converging on
    #: it, so a second hash here would be a second thing to keep in step.
    model_identity_hash: str
    #: Spelled out ALONGSIDE the hash, not instead of it. The hash makes a
    #: mismatch a clean refusal; these make the error message say WHICH field
    #: differs, which is the difference between a two-minute fix and a bisect.
    kv_dtype: str
    page_size: int
    #: Geometry that changes the row layout on the wire.
    tp_size: int
    pp_size: int
    total_kv_head_num: int
    head_dim: int
    #: Per-component state layout, parallel to KVArgs.state_types. A hybrid
    #: model whose peer has no mamba component is the #212 failure in advance.
    state_types: Tuple[str, ...] = ()
    #: Fork geometry: token-sharded KV changes which rows a rank owns.
    dcp_size: int = 1
    #: THE ROW-OWNERSHIP DESCRIPTOR (S, lo, hi) from the weighted owner rule:
    #: this rank owns global slot L iff ``(L % S) in [lo, hi)``, and stores it
    #: at ``(L // S) * (hi - lo) + (L % S - lo)``. ``None`` off the
    #: token-sharded lane, where a row is just its slot.
    #:
    #: This is the quantity a transfer actually has to agree on, which is why
    #: it is compared rather than inferred. See the note on COMPARED.
    row_ownership: Optional[Tuple[int, int, int]] = None

    #: Fields compared for compatibility, in the order they are reported.
    #:
    #: ``dcp_size`` and ``row_ownership`` ARE compared, and the review that
    #: put them here is worth restating: the earlier exclusion reasoned that
    #: PD already supports differing prefill/decode TP because ``KVArgs``
    #: carries ``state_dim_per_tensor`` / ``state_dim_offsets`` so the sender
    #: can re-slice. That argument is true and does not extend to DCP. Those
    #: tables describe the STATE/HEAD axis; DCP shards the TOKEN axis -- slot
    #: L lives on rank ``L % S`` at row ``L // S`` -- and no offset table
    #: re-derives row ownership. A ``dcp_size=1`` prefill and a ``dcp_size=3``
    #: decode have entirely different row->token mappings, so a transfer
    #: between them moves bytes to rows that mean something else. It could not
    #: bite while ``transfer()`` raised; the wire slice is exactly what makes
    #: it bite.
    #:
    #: ``row_ownership`` is the precise invariant and ``dcp_size`` the coarse
    #: guard that still refuses when ownership could not be computed on one
    #: side (``None`` vs a tuple). Keeping both means a future differing-DCP
    #: PD is legal exactly when the row mappings genuinely agree, rather than
    #: being blocked by a proxy.
    COMPARED: Tuple[str, ...] = (
        "model_identity_hash",
        "kv_dtype",
        "page_size",
        "total_kv_head_num",
        "head_dim",
        "state_types",
        "dcp_size",
        "row_ownership",
    )

    def diff(self, other: TransportIdentity) -> List[str]:
        """Field-by-field disagreement, human-readable, empty when compatible.

        ``tp_size`` / ``pp_size`` are deliberately NOT compared: PD already
        supports differing prefill/decode TP (KVArgs carries
        ``state_dim_per_tensor`` and ``state_dim_offsets`` precisely so the
        sender can re-slice), so demanding equality would refuse working
        configurations. They travel in the identity for diagnostics.

        ``dcp_size`` and ``row_ownership`` ARE compared -- the head-axis
        re-slicing argument above does not reach the token axis. See COMPARED.
        """
        out: List[str] = []
        for field in self.COMPARED:
            mine = getattr(self, field)
            theirs = getattr(other, field)
            if mine != theirs:
                out.append(f"{field}: local={mine!r} peer={theirs!r}")
        return out

    def assert_compatible(self, other: TransportIdentity, *, peer: str) -> None:
        """Raise unless the peer can be transferred with. Loud and specific."""
        problems = self.diff(other)
        if not problems:
            return
        raise IncompatiblePeer(
            f"KV transport refusing peer {peer}: "
            + "; ".join(problems)
            + ". A transfer between these two would move bytes that the "
            "receiver reads in a different format -- wrong tokens, not an "
            "error, which is why this is checked before the first transfer "
            "rather than after the first bad answer."
        )

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["state_types"] = list(self.state_types)
        d.pop("COMPARED", None)
        return d

    @classmethod
    def from_json(cls, data: dict) -> TransportIdentity:
        data = dict(data)
        data.pop("COMPARED", None)
        data["state_types"] = tuple(data.get("state_types") or ())
        return cls(**data)


class IncompatiblePeer(RuntimeError):
    """The two instances cannot exchange KV meaningfully."""


def identity_from_args(server_args: Any, kv_args: Any) -> TransportIdentity:
    """Build the local identity from the resolved server args and KVArgs.

    Reuses ``compute_model_identity_hash`` so the transport handshake and the
    HiCache storage key answer "is this the same model and byte format" with
    the SAME function. Two answers to one question is how they drift.
    """
    from sglang.srt.mem_cache.hicache_storage import compute_model_identity_hash

    state_types = tuple(
        str(getattr(s, "value", s)) for s in (getattr(kv_args, "state_types", ()) or ())
    )
    dcp_size = int(getattr(server_args, "dcp_size", 1) or 1)
    # The row-ownership descriptor, taken from the SAME owner rule the pools
    # and the attention path use -- never re-derived here, because a second
    # derivation of who owns a row is how the two drift apart. None off the
    # token-sharded lane (and whenever no plan is installed), which the
    # comparison then treats as "must also be None on the peer".
    row_ownership = None
    try:
        from sglang.srt.layers.dcp.owner import dcp_weighted_owner_bounds

        bounds = dcp_weighted_owner_bounds(
            dcp_size, int(getattr(kv_args, "engine_rank", 0) or 0)
        )
        if bounds is not None:
            row_ownership = (int(bounds[0]), int(bounds[1]), int(bounds[2]))
    except Exception:  # noqa: BLE001 - absent plan/module is "not sharded"
        row_ownership = None
    return TransportIdentity(
        model_identity_hash=compute_model_identity_hash(server_args),
        kv_dtype=str(getattr(server_args, "kv_cache_dtype", "auto") or "auto").lower(),
        page_size=int(getattr(server_args, "page_size", 1) or 1),
        tp_size=int(getattr(server_args, "tp_size", 1) or 1),
        pp_size=int(getattr(server_args, "pp_size", 1) or 1),
        total_kv_head_num=int(getattr(kv_args, "total_kv_head_num", 0) or 0),
        head_dim=int(getattr(kv_args, "head_dim", 0) or 0),
        state_types=state_types,
        dcp_size=dcp_size,
        row_ownership=row_ownership,
    )


def plan_blocks(
    *,
    region_index: int,
    src_rows: Sequence[int],
    dst_rows: Sequence[int],
    row_bytes: int,
):
    """Turn a row mapping into a coalesced block list.

    Row-addressed because every PD payload is (KV pages, mamba slots, aux
    rows), and coalesced because a contiguous run of rows is one transfer
    rather than N -- the difference between a bulk wire and a ping-pong.

    Src and dst row lists must be the same length: they are a MAPPING, and a
    length mismatch means the caller does not know which row goes where. That
    is refused rather than truncated to the common prefix -- the mooncake path
    already learned (conn.py, the state-index mismatch note) that silent
    truncation misaligns rows and corrupts KV.
    """
    from sglang.srt.disaggregation.nccl.link import TransferBlock

    if len(src_rows) != len(dst_rows):
        raise ValueError(
            f"row mapping length mismatch: {len(src_rows)} source rows against "
            f"{len(dst_rows)} destination rows. Truncating to the common "
            "prefix would misalign every row after the first gap."
        )
    if row_bytes <= 0:
        raise ValueError("row_bytes must be positive")

    blocks: List[TransferBlock] = []
    run_start = 0
    n = len(src_rows)
    for i in range(1, n + 1):
        contiguous = (
            i < n
            and src_rows[i] == src_rows[i - 1] + 1
            and dst_rows[i] == dst_rows[i - 1] + 1
        )
        if contiguous:
            continue
        length = i - run_start
        blocks.append(
            TransferBlock(
                region_index=region_index,
                src_offset_bytes=src_rows[run_start] * row_bytes,
                dst_offset_bytes=dst_rows[run_start] * row_bytes,
                length_bytes=length * row_bytes,
            )
        )
        run_start = i
    return blocks
