"""#410 slice 1: the SESSION-level manifest for conversation checkpoints.

The payload format already exists. #706 made the store a per-token-content
snapshot of exactly the two things a conversation checkpoint needs -- ``{hash}``
for a token's KV (all attention layers, geometry-free, cut at read time) and
``{hash}.mamba`` for the GDN/recurrent state -- both keyed by content alone, so
both already survive a phase flip and a reboot. What was missing is the layer
above: WHICH pages, WHICH blob, at which token position, under which sampling
and template state. That is this module.

THE #212 LESSON IS THE CONSTRAINT, not a caveat. A KV-only prefix is worth ZERO
usable tokens on a hybrid checkpoint: ``batch_exists_v2`` takes the MINIMUM
across pools with the mamba pool registered ``TRAILING_PAGES``, and device-side
``MambaRadixCache._match_prefix_helper`` advances only at nodes carrying
``mamba_value`` (re-derived and pinned in ``test_mamba_gates_the_hit_706.py``).
Three rules follow, and the module enforces all three rather than documenting
them:

1. the GDN anchor is REQUIRED for a hybrid model and is refused at WRITE time,
   because a manifest describing a prefix that resolves to nothing is a bug
   best caught where it is created;
2. a checkpoint position must be one where state exists -- with
   ``--mamba-checkpoint-interval`` that is an absolute grid, so the position is
   SNAPPED DOWN and both the requested and the effective position are recorded
   (a checkpoint that quietly covers fewer tokens than asked for is a wrong
   answer, not a rounding);
3. completeness is VERIFIED against the store before a branch seeds anything,
   because pages are cache entries and the tiers evict -- a manifest holds
   references, not bytes.

The format is versioned and content-only (no geometry, no placement, no rank),
which is what lets a manifest written by the PP prefill phase be applied by the
TP decode phase, by another rig, or after a reboot -- and what makes the
portable-session contract (#411) a later serialisation question rather than a
redesign.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Callable, Optional, Sequence

FORMAT_VERSION = 1

# Pool suffix of the GDN blob, mirroring HiCacheStorage's component convention.
MAMBA_SUFFIX = ".mamba"


class ManifestError(ValueError):
    """A manifest that cannot be trusted. Never downgraded to a warning."""


class ManifestIncomplete(ManifestError):
    """References named by the manifest are no longer in the store."""

    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(
            f"{len(self.missing)} reference(s) named by this checkpoint are no "
            f"longer in the store (first: {', '.join(self.missing[:4])}). "
            "Branching would seed a PARTIAL prefix, which on a hybrid model "
            "resolves to zero usable tokens rather than to fewer -- refusing "
            "at branch time instead."
        )


@dataclasses.dataclass(frozen=True)
class GdnAnchor:
    """Where the recurrent state for this checkpoint lives, and for which token.

    ``token_position`` is the EFFECTIVE position -- the grid point the
    checkpoint was snapped down to -- so a reader never has to re-derive it
    from the interval.
    """

    blob_key: str
    token_position: int
    checkpoint_interval: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "blob_key": self.blob_key,
            "token_position": int(self.token_position),
            "checkpoint_interval": self.checkpoint_interval,
        }


@dataclasses.dataclass(frozen=True)
class SessionManifest:
    """One conversation checkpoint, as references plus replay state."""

    model_identity: str
    session_id: str
    token_count: int
    page_hashes: tuple[str, ...]
    gdn: Optional[GdnAnchor] = None
    sampling: dict = dataclasses.field(default_factory=dict)
    template: dict = dataclasses.field(default_factory=dict)
    parent: Optional[dict] = None
    requested_token_count: Optional[int] = None
    created_at: float = dataclasses.field(default_factory=time.time)
    format_version: int = FORMAT_VERSION

    @property
    def snapped(self) -> bool:
        """True when the effective position is behind the requested one."""
        return self.requested_token_count is not None and int(
            self.requested_token_count
        ) != int(self.token_count)

    def references(self) -> tuple[str, ...]:
        """Every store key this checkpoint depends on, in seed order."""
        keys = list(self.page_hashes)
        if self.gdn is not None:
            keys.append(self.gdn.blob_key)
        return tuple(keys)

    def as_dict(self) -> dict:
        return {
            "format_version": int(self.format_version),
            "model_identity": self.model_identity,
            "session_id": self.session_id,
            "token_count": int(self.token_count),
            "requested_token_count": self.requested_token_count,
            "page_hashes": list(self.page_hashes),
            "gdn": self.gdn.as_dict() if self.gdn is not None else None,
            "sampling": dict(self.sampling),
            "template": dict(self.template),
            "parent": dict(self.parent) if self.parent else None,
            "created_at": float(self.created_at),
        }


def build_manifest(
    *,
    model_identity: str,
    session_id: str,
    page_hashes: Sequence[str],
    requested_token_count: int,
    gdn_blob_key: Optional[str],
    checkpoint_interval: Optional[int],
    is_hybrid_model: bool,
    sampling: Optional[dict] = None,
    template: Optional[dict] = None,
    parent: Optional[dict] = None,
) -> SessionManifest:
    """Build a checkpoint, snapping to the grid and refusing a stateless one.

    Snapping happens HERE rather than at the caller so that every producer gets
    the same rule: a checkpoint may only claim a position at which recurrent
    state exists.
    """
    from sglang.srt.mem_cache.mamba_ckpt_utils import is_on_interval

    requested = int(requested_token_count)
    if requested < 0:
        raise ManifestError(f"token count cannot be negative, got {requested}")
    if requested > len(page_hashes):
        raise ManifestError(
            f"checkpoint claims {requested} tokens but only "
            f"{len(page_hashes)} page hashes were supplied; a checkpoint may "
            "not name tokens it has no pages for."
        )

    effective = requested
    if checkpoint_interval:
        # Snap DOWN to the grid: the state upstream of this position exists,
        # the state at an off-grid position does not.
        effective = (requested // int(checkpoint_interval)) * int(checkpoint_interval)
    if not is_on_interval(effective, checkpoint_interval):
        raise ManifestError(
            f"effective position {effective} is not on the checkpoint grid "
            f"(interval {checkpoint_interval})."
        )

    if is_hybrid_model and not gdn_blob_key:
        raise ManifestError(
            "this model has linear/GDN layers, so a checkpoint without a "
            "recurrent-state anchor describes a prefix that resolves to ZERO "
            "usable tokens (the storage hit is the MINIMUM across pools, and "
            "the device-side match advances only at nodes carrying mamba "
            "state). Refused where it is created rather than at branch time."
        )
    if effective == 0 and requested > 0:
        raise ManifestError(
            f"a checkpoint at {requested} tokens snaps to position 0 under "
            f"interval {checkpoint_interval}: there is no state to anchor it "
            "to. Take the checkpoint at or after the first grid point."
        )

    gdn = (
        GdnAnchor(
            blob_key=gdn_blob_key,
            token_position=effective,
            checkpoint_interval=checkpoint_interval,
        )
        if gdn_blob_key
        else None
    )
    return SessionManifest(
        model_identity=model_identity,
        session_id=session_id,
        token_count=effective,
        requested_token_count=requested,
        page_hashes=tuple(page_hashes[:effective]),
        gdn=gdn,
        sampling=dict(sampling or {}),
        template=dict(template or {}),
        parent=dict(parent) if parent else None,
    )


def dumps(manifest: SessionManifest) -> str:
    """Deterministic serialisation, so a manifest hashes reproducibly."""
    return json.dumps(manifest.as_dict(), sort_keys=True, separators=(",", ":"))


def loads(blob: str) -> SessionManifest:
    """Parse a manifest, refusing anything this version cannot vouch for."""
    try:
        raw = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ManifestError(f"manifest is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ManifestError("manifest must be a JSON object")
    version = raw.get("format_version")
    if version != FORMAT_VERSION:
        # Refused, never best-effort parsed: a checkpoint is replayed into a
        # live session, and guessing at an unknown layout is how a future
        # field silently stops being honoured.
        raise ManifestError(
            f"manifest format_version {version!r} is not {FORMAT_VERSION}; this "
            "server cannot vouch for its contents. Portability across versions "
            "is the #411 contract and needs an explicit converter, not a "
            "tolerant parser."
        )
    gdn_raw = raw.get("gdn")
    gdn = (
        GdnAnchor(
            blob_key=gdn_raw["blob_key"],
            token_position=int(gdn_raw["token_position"]),
            checkpoint_interval=gdn_raw.get("checkpoint_interval"),
        )
        if gdn_raw
        else None
    )
    return SessionManifest(
        model_identity=raw["model_identity"],
        session_id=raw["session_id"],
        token_count=int(raw["token_count"]),
        requested_token_count=raw.get("requested_token_count"),
        page_hashes=tuple(raw.get("page_hashes") or ()),
        gdn=gdn,
        sampling=dict(raw.get("sampling") or {}),
        template=dict(raw.get("template") or {}),
        parent=raw.get("parent"),
        created_at=float(raw.get("created_at") or 0.0),
    )


def verify_against_store(
    manifest: SessionManifest, exists: Callable[[str], bool]
) -> tuple[str, ...]:
    """References the store no longer holds, in seed order. ``()`` when whole.

    ``exists`` is the backend's own existence check, so this asks the question
    the way a read would answer it -- including #706's rule that an incomplete
    page is INVISIBLE rather than partially readable.
    """
    return tuple(key for key in manifest.references() if not exists(key))


def verify_model_identity(manifest: SessionManifest, model_identity: str) -> None:
    """Refuse a checkpoint from a different model or KV byte format."""
    if manifest.model_identity != model_identity:
        raise ManifestError(
            f"this checkpoint was taken under model identity "
            f"{manifest.model_identity!r} and this server runs "
            f"{model_identity!r}. The identity covers weights, dtype, "
            "quantization and kv-cache-dtype, so the stored pages are a "
            "different byte format -- a mismatch is a refusal, not a miss."
        )


@dataclasses.dataclass(frozen=True)
class SeedStep:
    """One fetch a branch performs, in order."""

    key: str
    kind: str  # "kv" | "gdn"


def seed_plan(
    manifest: SessionManifest,
    *,
    exists: Optional[Callable[[str], bool]] = None,
    model_identity: Optional[str] = None,
) -> tuple[SeedStep, ...]:
    """The ordered fetch list a branch executes, verified before it starts.

    Verification up front is the point: a traversal that discovers a gap
    halfway has already seeded a partial prefix into a live session, and on a
    hybrid model a partial prefix does not degrade -- it resolves to nothing
    while looking like a hit.
    """
    if model_identity is not None:
        verify_model_identity(manifest, model_identity)
    if exists is not None:
        missing = verify_against_store(manifest, exists)
        if missing:
            raise ManifestIncomplete(missing)
    steps = [SeedStep(key=h, kind="kv") for h in manifest.page_hashes]
    if manifest.gdn is not None:
        # LAST, deliberately: the recurrent state is what makes the prefix
        # usable, so it is the final step of a branch and its absence can never
        # look like a completed seed.
        steps.append(SeedStep(key=manifest.gdn.blob_key, kind="gdn"))
    return tuple(steps)
