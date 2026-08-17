# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Server-side conversation checkpoints, branching and rewind (#410).

A checkpoint freezes one session -- its KV pages AND its GDN/Mamba state --
so the conversation can later be branched or rewound WITHOUT re-prefilling.

The whole feature is a re-aim of machinery that already exists and is already
byte-proven, and the reuse is deliberate rather than incidental:

*   the SNAPSHOT is #261's, unchanged and shared as code
    (``session_handover.export_session_snapshot``). A checkpoint is a handover
    export whose destination is a storage TIER instead of a peer group. There
    is exactly ONE session serialization in this fork -- the manifest is
    versioned, and the task ledger names it as #411's portable-session format
    too, so a second one would be a second thing to keep byte-correct;
*   the #212 lesson rides along for free: the GDN blob is named explicitly in
    the manifest and ``validate_manifest_completeness`` refuses a hybrid-GDN
    export without it. The store route's prefix matching would otherwise
    truncate the recurrent state silently, and a truncated recurrent state is
    a WRONG conversation, not a slow one;
*   the RESTORE half is #261's ``verify_import`` (the #241 identity key:
    model, dtype, quantization, kv-cache dtype) plus one gate this module
    adds, :func:`verify_geometry`. Rewind and branch land on the SAME group,
    so no umsharder is needed for same-geometry restore -- but a checkpoint
    read back from a persistent tier written by an earlier boot may not
    match, and that is refused BY NAME. There is no silent conversion;
*   the tier the checkpoint rests on comes from the #407 registry through
    ``memtier.consumers.checkpoint_tier_targets``, VRAM -> RAM -> Disk by age
    and durability, provenance-labelled. A rig where nothing is admissible
    produces the itemised refusal, never a guess. This is the registry's
    first PRODUCTION consumer;
*   BRANCHING copies nothing. Two sessions that share a token prefix already
    share radix nodes -- ``UnifiedRadixCache._split_node`` splits at the
    divergence point and the common ancestor keeps the SAME device page
    indices -- so a branch is a lock reference on the checkpoint's chain plus
    a new session that starts at the checkpoint's token offset. Copy-on-
    restore is what the radix tree does anyway; the only thing this module
    adds is the pin that stops the shared prefix being evicted underneath a
    branch that has not run its first turn yet.

Rewind and branch are both expressed through ``SessionParams.offset``, the
token-level splice point the session controller already implements
(``context[:offset] + new_tokens``). Nothing new rewrites history.

v1 limits, refused by name rather than worked around:

*   TP=1 / PP=1 source, inherited from #261: the manifest is rank-local, and
    a TP>1 checkpoint needs the same per-rank manifest merging #261 named as
    its follow-up;
*   ``page_size == 1``, inherited from ``dcp_owner_mode``;
*   the ``file`` HiCache storage backend only.

The pure pieces (ids, ledger, manifest envelope, gates, accounting) are
separated from the scheduler adapter so the falsifiers run hermetically.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.managers.session_handover import (
    SessionHandoverError,
    backend_exists_fn,
    export_session_snapshot,
    handover_id_for,
    match_prefix_result,
    storage_preconditions,
    verify_import,
)

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import (
        SessionCheckpointReqInput,
        SessionCheckpointReqOutput,
    )

logger = logging.getLogger(__name__)

#: Envelope schema version. Independent of the #261 manifest version: the
#: checkpoint envelope is an ADDITIVE key on a v1 handover manifest, so a
#: reader that only knows #261 still validates and imports a checkpoint, and
#: a reader that knows #410 can tell how the extra keys are shaped.
CHECKPOINT_ENVELOPE_VERSION = 1

#: Geometry fields a restore must match exactly. These are the fields the
#: #261 manifest already records under ``source``; the identity fields
#: (model, dtype, quantization, kv-cache dtype) live in the #241 hash and are
#: checked by ``verify_import``, not here.
GEOMETRY_FIELDS = ("tp_size", "page_size", "dcp_owner_mode")

#: Host memory layouts a checkpoint refuses. ``page_head``'s host write-back
#: dispatches to ``transfer_kv_all_layer_lf_ph`` (``pool_host/mha.py:401``),
#: which segfaults (#441a); the layouts that remain reach either
#: ``transfer_kv_all_layer_direct_lf_pf`` (#436) or the layer-first route. A
#: checkpoint writes the WHOLE session to the host tier, so this is not an
#: edge case for it. Same refusal family as ``draft_migrate.REFUSED_LAYOUTS``.
REFUSED_HOST_LAYOUTS = ("page_head",)


class SessionCheckpointError(SessionHandoverError):
    """A checkpoint/branch/rewind step that cannot proceed. The message always
    names why, and where retrying can help it says so."""


# ---------------------------------------------------------------------------
# Ids and manifest envelope
# ---------------------------------------------------------------------------


def checkpoint_id_for(token_ids: Sequence[int], identity_hash: str) -> str:
    """CONTENT-addressed, via #261's derivation.

    Checkpointing the same prefix of the same session twice therefore yields
    the same id and is idempotent -- which is the correct behaviour, because
    the two checkpoints would name byte-identical blobs in a
    content-addressed store. Distinct turns diverge in their token ids and so
    get distinct ids without any counter to keep.
    """
    return handover_id_for(token_ids, identity_hash)


def build_checkpoint_manifest(
    *,
    checkpoint_id: str,
    session_id: str,
    base_manifest: Dict,
    tier_id: Optional[str],
    tier_provenance: str,
    tier_alternatives: Sequence[str] = (),
    durable: bool = False,
    parent_checkpoint_id: Optional[str] = None,
    label: str = "",
) -> Dict:
    """A #261 manifest plus the ``checkpoint`` envelope.

    Additive by construction: every key #261 writes is left exactly as
    ``build_manifest`` produced it, so ``verify_import`` reads a checkpoint
    manifest without knowing this module exists. The envelope carries only
    what a checkpoint has and a handover does not -- which session it came
    from, where it was placed and with what provenance, and how it relates to
    the checkpoint it was branched off.
    """
    manifest = dict(base_manifest)
    manifest["checkpoint"] = {
        "envelope_version": CHECKPOINT_ENVELOPE_VERSION,
        "checkpoint_id": checkpoint_id,
        "session_id": session_id,
        "parent_checkpoint_id": parent_checkpoint_id,
        "label": label,
        "durable": bool(durable),
        "tier_id": tier_id,
        "tier_provenance": tier_provenance,
        "tier_alternatives": list(tier_alternatives),
        "created_unix": time.time(),
    }
    return manifest


def geometry_of(manifest: Dict) -> Dict[str, Any]:
    source = manifest.get("source") or {}
    return {field_name: source.get(field_name) for field_name in GEOMETRY_FIELDS}


def verify_geometry(manifest: Dict, local_geometry: Dict[str, Any]) -> Tuple[bool, str]:
    """The #241-keyed cross-geometry gate: same shape, or a named refusal.

    Rewind and branch land on the SAME running group, so in the common case
    this is a tautology and costs one dict comparison. It exists for the case
    that is not common and is silent when it goes wrong: a checkpoint read
    back from a PERSISTENT tier that a previous boot wrote under a different
    tensor-parallel size, page size or DCP owner mode. The pages are then laid
    out differently and nothing in the read path would notice.

    Converting is a real operation with a real implementation -- the
    manifest-scoped umsharder, run offline -- so this refuses and names it
    instead of attempting a reshape in-process. A checkpoint is never
    silently converted.
    """
    got = geometry_of(manifest)
    mismatched = {
        field_name: (got.get(field_name), local_geometry.get(field_name))
        for field_name in GEOMETRY_FIELDS
        if got.get(field_name) != local_geometry.get(field_name)
    }
    if mismatched:
        detail = "; ".join(
            f"{name}: checkpoint {have!r} vs this server {want!r}"
            for name, (have, want) in sorted(mismatched.items())
        )
        return False, (
            f"checkpoint geometry does not match this server ({detail}). A "
            "same-geometry rewind or branch needs no reshape; a different "
            "geometry needs the manifest-scoped umsharder "
            "(hicache_migrate --manifest) run offline first. Refusing to "
            "convert silently."
        )
    return True, "geometry matches: " + ", ".join(
        f"{name}={got.get(name)!r}" for name in GEOMETRY_FIELDS
    )


def verify_restore(
    manifest: Dict,
    exists_fn: Callable[[str], bool],
    local_identity_hash: str,
    local_geometry: Dict[str, Any],
) -> Tuple[bool, str]:
    """Both restore gates in the order a reader wants them.

    Identity first (#261's ``verify_import``: manifest version, the #241
    model/dtype/kv-format hash, and every named blob present in this rank's
    store, GDN blob included), then geometry. Identity failing is the more
    fundamental problem, so it is the message the caller sees.
    """
    ok, message = verify_import(manifest, exists_fn, local_identity_hash)
    if not ok:
        return False, message
    geometry_ok, geometry_message = verify_geometry(manifest, local_geometry)
    if not geometry_ok:
        return False, geometry_message
    return True, f"{message}; {geometry_message}"


# ---------------------------------------------------------------------------
# Page-sharing accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchAccounting:
    """What a branch actually costs, in pages.

    ``copied`` is structurally zero and the test asserts it: a branch never
    duplicates a page. The radix tree shares the common prefix by splitting at
    the divergence point, and this module's only contribution is the lock
    reference that stops the shared chain being evicted before the branch runs
    its first turn.

    ``prefetch`` is the honest remainder -- pages the tree no longer holds on
    device or host, which the next turn pulls back from the tier the
    checkpoint was written to. That is a storage read, not a re-prefill, so it
    is reported separately from both.
    """

    total: int
    shared: int
    prefetch: int
    copied: int = 0

    @property
    def shared_fraction(self) -> float:
        return 0.0 if self.total == 0 else self.shared / self.total

    def to_json(self) -> Dict[str, Any]:
        return {
            "total_pages": self.total,
            "shared_pages": self.shared,
            "prefetch_pages": self.prefetch,
            "copied_pages": self.copied,
            "shared_fraction": round(self.shared_fraction, 4),
        }


def account_branch(total_tokens: int, matched_tokens: int) -> BranchAccounting:
    if matched_tokens > total_tokens:
        raise SessionCheckpointError(
            f"radix tree reports {matched_tokens} matched tokens for a "
            f"{total_tokens}-token checkpoint; a prefix cannot be longer than "
            "the sequence it is a prefix of -- refusing to report a negative "
            "prefetch count"
        )
    return BranchAccounting(
        total=total_tokens,
        shared=matched_tokens,
        prefetch=total_tokens - matched_tokens,
    )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@dataclass
class CheckpointRecord:
    checkpoint_id: str
    session_id: str
    token_ids: List[int]
    manifest: Dict
    tier_id: Optional[str] = None
    tier_provenance: str = "absent"
    label: str = ""
    durable: bool = False
    created_unix: float = field(default_factory=time.time)
    #: Sessions currently derived from this checkpoint (branches that have not
    #: been dropped, plus a rewound session that is sitting on it). While this
    #: is non-empty the checkpoint's radix chain is pinned.
    derived_sessions: List[str] = field(default_factory=list)

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.created_unix)

    def to_json(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "tokens": len(self.token_ids),
            "tier_id": self.tier_id,
            "tier_provenance": self.tier_provenance,
            "label": self.label,
            "durable": self.durable,
            "created_unix": self.created_unix,
            "age_s": round(self.age_s, 3),
            "derived_sessions": list(self.derived_sessions),
            "hybrid_gdn": bool(self.manifest.get("hybrid_gdn")),
            "kv_pages": len(self.manifest.get("kv_keys") or []),
        }


class PinCoverageIncomplete(SessionCheckpointError):
    """A checkpoint could not pin every reference it depends on.

    Raised at CHECKPOINT time, which is the point. ``stems_with_sizes`` drops a
    stem whose file is absent -- correct for the budget, silent to the caller --
    so without this a checkpoint could pin four of its six pages and report
    success. The other two would then be evicted and the failure would surface
    at the BRANCH, later and further from its cause.

    Nothing here can make an absent page exist. This moves the refusal to where
    the caller can still act on it, and names what is missing.
    """

    def __init__(self, checkpoint_id: str, unpinned: Sequence[str]):
        self.checkpoint_id = checkpoint_id
        self.unpinned = tuple(unpinned)
        super().__init__(
            f"checkpoint {checkpoint_id!r} could not pin {len(self.unpinned)} of "
            f"its references on the file tier, so it would protect less than it "
            f"claims: {list(self.unpinned)!r}. Nothing was pinned."
        )


@dataclasses.dataclass(frozen=True)
class FileTierPins:
    """What the file tier actually protects for one checkpoint."""

    requested: int
    pinned: int
    unpinned: Tuple[str, ...] = ()
    protected: bool = True
    reason: str = ""


def take_file_tier_pins(store, checkpoint_id: str, keys: Sequence[str]) -> FileTierPins:
    """Pin a checkpoint's references on the FILE tier, or say why not.

    THIS IS A DIFFERENT TIER FROM ``_pin``. That one takes
    ``tree.inc_lock_ref`` on the radix chain: in memory, for the life of the
    process. This one protects the on-disk stems from the file evictor and
    across a restart. A checkpoint needs both, and this lineage had only the
    first -- an A checkpoint survives radix eviction and does not survive
    file-tier eviction.

    NOT EVERY CHECKPOINT HAS A FILE TIER. #407 may place one on vram or host,
    where there are no stems to pin. That is reported as unprotected rather
    than raised: refusing a placement the tier policy chose would be this layer
    overruling the one whose decision it is.
    """
    keys = list(keys)
    if not keys:
        return FileTierPins(0, 0, (), True, "no references to pin")
    if store is None:
        return FileTierPins(
            len(keys), 0, (), False, "no file tier holds this checkpoint"
        )
    if not hasattr(store, "pin_checkpoint"):
        return FileTierPins(
            len(keys),
            0,
            (),
            False,
            "storage backend does not support pinning (pre-#410 wheel)",
        )

    result = store.pin_checkpoint(checkpoint_id, keys)
    unpinned = tuple(getattr(result, "unpinned", ()) or ())
    if unpinned:
        # Roll back: a partially pinned checkpoint holds bytes for a promise it
        # cannot keep, and the orphan reaper would only collect them after its
        # age gate.
        try:
            store.unpin_checkpoint(checkpoint_id)
        except Exception:  # pragma: no cover - best-effort rollback
            logger.exception(
                "SESSION-CHECKPOINT could not release pins after refusing %s",
                checkpoint_id,
            )
        raise PinCoverageIncomplete(checkpoint_id, unpinned)
    return FileTierPins(len(keys), len(getattr(result, "stems", ()) or ()), (), True, "")


def pin_imported_pages(store, checkpoint_id: str, manifest: Dict) -> FileTierPins:
    """Pin an imported session's pages on the target's file tier (#411).

    THE GDN BLOB IS PART OF THE SET, not an extra. #212: a KV-only prefix is
    worth zero usable tokens on a hybrid model, so a session whose mamba blob
    was evicted is not a partially usable session, it is an unusable one. It is
    pinned with the KV pages or the protection is decorative.

    Called AFTER ``verify_restore`` -- there is nothing to pin until the pages
    are in the store -- and BEFORE the session is seeded. Verification proves
    the pages are there NOW; it does not stop the evictor reclaiming them a
    moment later, and a freshly imported session is unreferenced by any running
    request until it is used, so a store under pressure can drop it before it
    is ever read.

    This is the capability the export refusal already advised and could not
    name (`NOTE_411_portable_sessions.md` §C5): the pin ledger did not exist on
    that base. It does here.
    """
    keys = list(manifest.get("kv_keys") or [])
    mamba_key = manifest.get("mamba_key")
    if mamba_key:
        keys.append(mamba_key)
    return take_file_tier_pins(store, checkpoint_id, keys)


class CheckpointLedger:
    """Every live checkpoint and what derives from it. Pure, hermetic.

    Deliberately NOT #261's ``HandoverLedger``: that one is a transfer-of-
    ownership state machine whose whole purpose is that exactly one server
    owns a session, and its prefix-conflict rule refuses a second active
    record on an overlapping prefix. A checkpoint is the opposite -- an
    immutable snapshot that MANY sessions may derive from at once, and
    overlapping prefixes are the normal case (every checkpoint of one
    conversation is a prefix of the next). Reusing the handover ledger would
    have made branching refuse itself.
    """

    def __init__(self) -> None:
        self._records: Dict[str, CheckpointRecord] = {}
        #: session id -> checkpoint id it currently derives from.
        self._derivations: Dict[str, str] = {}

    # -- checkpoints --------------------------------------------------------

    def add(self, record: CheckpointRecord) -> CheckpointRecord:
        existing = self._records.get(record.checkpoint_id)
        if existing is not None:
            # Content-addressed ids: re-checkpointing an unchanged prefix is
            # idempotent, not an error. Keep the original record (and its
            # derivations) rather than resetting the age the tier policy reads.
            return existing
        self._records[record.checkpoint_id] = record
        return record

    def get(self, checkpoint_id: str) -> CheckpointRecord:
        record = self._records.get(checkpoint_id)
        if record is None:
            known = ", ".join(sorted(self._records)) or "none"
            raise SessionCheckpointError(
                f"unknown checkpoint id {checkpoint_id!r}; known: {known}"
            )
        return record

    def list_for(self, session_id: Optional[str] = None) -> List[CheckpointRecord]:
        records = list(self._records.values())
        if session_id is not None:
            records = [r for r in records if r.session_id == session_id]
        return sorted(records, key=lambda r: r.created_unix)

    def drop(self, checkpoint_id: str) -> CheckpointRecord:
        record = self.get(checkpoint_id)
        if record.derived_sessions:
            raise SessionCheckpointError(
                f"checkpoint {checkpoint_id} still has "
                f"{len(record.derived_sessions)} derived session(s) "
                f"({', '.join(record.derived_sessions)}); dropping it would "
                "unpin pages they are sharing. Close or re-point them first."
            )
        del self._records[checkpoint_id]
        return record

    # -- derivations --------------------------------------------------------

    def derive(self, session_id: str, checkpoint_id: str) -> CheckpointRecord:
        """Point ``session_id`` at ``checkpoint_id``, releasing any previous.

        Returns the checkpoint that is now pinned FOR this session. A session
        derives from at most one checkpoint at a time: a second rewind
        supersedes the first, and the superseded checkpoint loses this
        session's reference.
        """
        record = self.get(checkpoint_id)
        self.release(session_id)
        record.derived_sessions.append(session_id)
        self._derivations[session_id] = checkpoint_id
        return record

    def release(self, session_id: str) -> Optional[CheckpointRecord]:
        """Drop ``session_id``'s reference, if it holds one."""
        previous_id = self._derivations.pop(session_id, None)
        if previous_id is None:
            return None
        previous = self._records.get(previous_id)
        if previous is not None and session_id in previous.derived_sessions:
            previous.derived_sessions.remove(session_id)
        return previous

    def derivation_of(self, session_id: str) -> Optional[str]:
        return self._derivations.get(session_id)

    def pinned_checkpoint_ids(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                r.checkpoint_id for r in self._records.values() if r.derived_sessions
            )
        )


# ---------------------------------------------------------------------------
# Scheduler adapter
# ---------------------------------------------------------------------------


class SessionCheckpointRuntime:
    """Thin adapter between the scheduler and the pure pieces above.

    Runs entirely on the scheduler thread (control requests are processed
    between scheduling iterations), so the radix tree cannot mutate under a
    snapshot, a lock reference or a rewind -- exactly the property #261's
    runtime relies on, and the reason none of this needs a lock of its own.
    No group collective is issued anywhere: every operation is rank-local.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.ledger = CheckpointLedger()
        self._identity_hash: Optional[str] = None
        self._registry = None
        self._registry_note: str = ""
        #: checkpoint id -> the radix node its chain ends at, while pinned.
        self._pinned_nodes: Dict[str, Any] = {}

    # -- control dispatch ---------------------------------------------------

    def handle(self, recv_req: SessionCheckpointReqInput) -> SessionCheckpointReqOutput:
        from sglang.srt.managers.io_struct import SessionCheckpointReqOutput

        try:
            action = recv_req.action
            if action == "checkpoint":
                record = self._checkpoint(
                    session_id=recv_req.session_id or "",
                    token_ids=recv_req.token_ids,
                    durable=bool(recv_req.durable),
                    label=recv_req.label or "",
                    deadline_s=recv_req.deadline_s,
                )
                return SessionCheckpointReqOutput(
                    success=True,
                    message=(
                        f"checkpoint {record.checkpoint_id}: "
                        f"{len(record.token_ids)} token(s), tier "
                        f"{record.tier_id or '-'} ({record.tier_provenance})"
                    ),
                    checkpoint_id=record.checkpoint_id,
                    manifest_json=json.dumps(record.manifest),
                    info=record.to_json(),
                )
            if action == "branch":
                return self._branch_output(recv_req)
            if action == "rewind":
                return self._rewind_output(recv_req)
            if action == "list":
                records = self.ledger.list_for(recv_req.session_id)
                return SessionCheckpointReqOutput(
                    success=True,
                    message=f"{len(records)} checkpoint(s)",
                    info={"checkpoints": [r.to_json() for r in records]},
                )
            if action == "drop":
                record = self.ledger.drop(recv_req.checkpoint_id or "")
                self._unpin(record.checkpoint_id)
                return SessionCheckpointReqOutput(
                    success=True, message=f"dropped {record.checkpoint_id}"
                )
            return SessionCheckpointReqOutput(
                success=False,
                message=(
                    f"unknown action {action!r}; known: checkpoint, branch, "
                    "rewind, list, drop"
                ),
            )
        except SessionHandoverError as e:
            return SessionCheckpointReqOutput(success=False, message=str(e))
        except Exception as e:  # never crash the scheduler over control plane
            logger.exception("SESSION-CHECKPOINT %s failed", recv_req.action)
            return SessionCheckpointReqOutput(
                success=False, message=f"{type(e).__name__}: {e}"
            )

    # -- shared bits --------------------------------------------------------

    def _identity(self) -> str:
        if self._identity_hash is None:
            from sglang.srt.mem_cache.hicache_storage import (
                compute_model_identity_hash,
            )

            self._identity_hash = compute_model_identity_hash(
                self.scheduler.server_args
            )
        return self._identity_hash

    def _local_geometry(self) -> Dict[str, Any]:
        args = self.scheduler.server_args
        tree = self.scheduler.tree_cache
        controller = getattr(tree, "cache_controller", None)
        storage_config = getattr(controller, "storage_config", None)
        return {
            "tp_size": args.tp_size,
            "page_size": args.page_size,
            "dcp_owner_mode": bool(getattr(storage_config, "dcp_owner_mode", False)),
        }

    def _preconditions(self):
        args = self.scheduler.server_args
        if not getattr(args, "enable_session_checkpoints", False):
            raise SessionCheckpointError(
                "session checkpoints are off; start the server with "
                "--enable-session-checkpoints"
            )
        tree = storage_preconditions(self.scheduler)
        if args.tp_size != 1 or args.pp_size != 1:
            raise SessionCheckpointError(
                f"session checkpoints are built for a TP=1, PP=1 server "
                f"(got tp_size={args.tp_size}, pp_size={args.pp_size}); the "
                "manifest is rank-local and a TP>1 checkpoint needs the "
                "per-rank manifest merging named as #261's follow-up"
            )
        if args.page_size != 1:
            raise SessionCheckpointError(
                f"page_size == 1 is required (got {args.page_size}); the "
                "manifest-scoped umsharder inherits this limit from "
                "dcp_owner_mode, and a checkpoint has to stay restorable by it"
            )
        if args.hicache_mem_layout in REFUSED_HOST_LAYOUTS:
            raise SessionCheckpointError(
                "--hicache-mem-layout page_head is refused for session "
                "checkpoints (#441a): its host write-back takes the "
                "transfer_kv_all_layer_lf_ph route, which segfaults. A "
                "checkpoint is a host-tier WRITE of the whole session, so it "
                "would take that route for every page. Use page_first / "
                "page_first_direct (direct IO reaches "
                "transfer_kv_all_layer_direct_lf_pf, #436) or layer_first."
            )
        return tree

    def _session_token_ids(self, session_id: str) -> List[int]:
        """The session's current token prefix: input plus generated output.

        Explicit ``token_ids`` on the request win, because a caller that wants
        a checkpoint at an earlier point in the conversation has to be able to
        say so -- that is the whole rewind story.
        """
        controller = getattr(self.scheduler, "session_controller", None)
        session = None if controller is None else controller.get(session_id)
        if session is None:
            raise SessionCheckpointError(
                f"unknown session {session_id!r}; open it with /open_session "
                "first (a checkpoint is taken of a server-side session, not "
                "of a stateless request)"
            )
        if not session.req_nodes:
            raise SessionCheckpointError(
                f"session {session_id!r} has no completed turn yet; there is "
                "nothing to checkpoint"
            )
        if len(session.req_nodes) > 1:
            raise SessionCheckpointError(
                f"session {session_id!r} holds {len(session.req_nodes)} "
                "request nodes (a non-streaming session tree); pass explicit "
                "token_ids to say which branch point to checkpoint"
            )
        [node] = session.req_nodes.values()
        req = node.req
        return list(req.origin_input_ids) + list(req.output_ids or [])

    def _in_flight_conflict(self, session_id: str) -> Optional[str]:
        """A request of THIS session that is still running or queued.

        #261 compares token prefixes because a handover parks a prefix
        globally. A checkpoint is scoped to one session, so the cheaper and
        more precise question is whether that session has work in flight --
        another conversation extending an overlapping prefix is none of this
        operation's business.
        """
        reqs = []
        running = getattr(self.scheduler, "running_batch", None)
        if running is not None and getattr(running, "reqs", None):
            reqs.extend(running.reqs)
        reqs.extend(getattr(self.scheduler, "waiting_queue", []) or [])
        chunked = getattr(self.scheduler, "chunked_req", None)
        if chunked is not None:
            reqs.append(chunked)
        for req in reqs:
            session = getattr(req, "session", None)
            if session is not None and session.session_id == session_id:
                return req.rid
        return None

    # -- tier selection (#407 consumer) -------------------------------------

    def _file_tier_store(self):
        """The HiCache file backend, or None when this boot has no file tier.

        Reached through the cache controller because that is who owns it;
        ``storage_backend`` is None when no storage tier is configured, and a
        checkpoint on vram or host legitimately has none.
        """
        tree = getattr(self.scheduler, "tree_cache", None)
        controller = getattr(tree, "cache_controller", None)
        return getattr(controller, "storage_backend", None)

    def _tier_registry(self):
        if self._registry is None:
            from sglang.srt.memtier.profile import collect_local_facts, nvml_card_facts
            from sglang.srt.memtier.registry import TierRegistry

            try:
                cards = nvml_card_facts()
            except Exception as e:  # a driverless box still gets host/fs tiers
                cards = ()
                self._registry_note = f"NVML enumeration failed ({e}); no device tiers"
            mounts = []
            storage_dir = getattr(
                self.scheduler.server_args, "hicache_storage_file_path", None
            )
            if storage_dir:
                mounts.append(storage_dir)
            facts = collect_local_facts(mounts=mounts, cards=cards)
            self._registry, selection = TierRegistry.for_machine(facts)
            logger.info(
                "SESSION-CHECKPOINT tier registry: profile=%s %s%s",
                self._registry.profile_id,
                getattr(selection, "reason", ""),
                f" ({self._registry_note})" if self._registry_note else "",
            )
        return self._registry

    def _select_tier(self, *, bytes_needed: int, durable: bool, age_s: float):
        from sglang.srt.memtier.consumers import (
            CheckpointTierPolicy,
            checkpoint_tier_targets,
        )

        args = self.scheduler.server_args
        policy = CheckpointTierPolicy(
            vram_max_age_s=float(
                getattr(args, "session_checkpoint_vram_max_age_s", 60.0)
            ),
            host_max_age_s=float(
                getattr(args, "session_checkpoint_host_max_age_s", 900.0)
            ),
        )
        answer = checkpoint_tier_targets(
            self._tier_registry(),
            bytes_needed=bytes_needed,
            age_s=age_s,
            durable=durable,
            policy=policy,
        )
        if not answer.ok:
            raise SessionCheckpointError(
                "no memory tier can hold this checkpoint. The #407 registry "
                "refuses rather than guessing a target:\n" + answer.refusal
            )
        return answer

    @staticmethod
    def _provenance_of(answer) -> str:
        selection = answer.selection
        if selection is None or not selection.candidates:
            return "absent"
        return selection.candidates[0].bandwidth_gbs.provenance.value

    def _checkpoint_bytes(self, tree, manifest: Dict) -> int:
        """Bytes this checkpoint occupies on whatever tier holds it.

        Page byte size comes from the pool when the pool exposes it and is
        ABSENT otherwise -- in which case the query asks about admissibility
        only (``bytes_needed=0``) rather than inventing a page size. A capacity
        check against a made-up number is worse than no capacity check,
        because it reads like one.
        """
        pages = len(manifest.get("kv_keys") or [])
        for owner in (getattr(tree, "token_to_kv_pool_allocator", None), tree):
            pool = getattr(owner, "_kvcache", None) or getattr(
                owner, "token_to_kv_pool", None
            )
            size = getattr(pool, "get_kv_size_bytes", None)
            if callable(size):
                try:
                    total = size()
                except Exception:
                    continue
                total = sum(total) if isinstance(total, (tuple, list)) else total
                capacity = getattr(pool, "size", 0) or 0
                if capacity:
                    return int(pages * (int(total) / int(capacity)))
        return 0

    # -- checkpoint ---------------------------------------------------------

    def _checkpoint(
        self,
        *,
        session_id: str,
        token_ids: Optional[Sequence[int]],
        durable: bool,
        label: str,
        deadline_s: float,
    ) -> CheckpointRecord:
        tree = self._preconditions()
        if not session_id:
            raise SessionCheckpointError("checkpoint needs a session id")
        resolved = list(token_ids) if token_ids else self._session_token_ids(session_id)
        if not resolved:
            raise SessionCheckpointError(
                f"session {session_id!r} resolved to an empty token prefix; "
                "there is nothing to checkpoint"
            )

        rid = self._in_flight_conflict(session_id)
        if rid is not None:
            raise SessionCheckpointError(
                f"session {session_id!r} has an in-flight request ({rid}); "
                "checkpoint refused. Retry after the turn finishes -- the "
                "natural checkpoint point is between turns."
            )

        checkpoint_id = checkpoint_id_for(resolved, self._identity())
        existing = self.ledger.list_for(session_id)
        parent = existing[-1].checkpoint_id if existing else None

        base_manifest = export_session_snapshot(
            self.scheduler,
            tree,
            snapshot_id=checkpoint_id,
            token_ids=resolved,
            identity_hash=self._identity(),
            deadline_s=deadline_s,
        )
        answer = self._select_tier(
            bytes_needed=self._checkpoint_bytes(tree, base_manifest),
            durable=durable,
            age_s=0.0,
        )
        manifest = build_checkpoint_manifest(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            base_manifest=base_manifest,
            tier_id=answer.tier_id,
            tier_provenance=self._provenance_of(answer),
            tier_alternatives=answer.alternatives,
            durable=durable,
            parent_checkpoint_id=parent,
            label=label,
        )
        # FILE-TIER PINS, before the record exists. _pin below locks the radix
        # chain, which protects this checkpoint in memory for the life of the
        # process; this protects the on-disk stems from the file evictor and
        # across a restart. Taken FIRST so a checkpoint that cannot be
        # protected never becomes a record: the alternative is a ledger entry
        # promising a branch whose pages are already reclaimable.
        file_pins = take_file_tier_pins(
            self._file_tier_store(), checkpoint_id, list(base_manifest["kv_keys"])
        )
        if not file_pins.protected:
            logger.info(
                "SESSION-CHECKPOINT %s has NO file-tier protection: %s. It "
                "survives radix eviction only, not a restart.",
                checkpoint_id,
                file_pins.reason,
            )

        record = self.ledger.add(
            CheckpointRecord(
                checkpoint_id=checkpoint_id,
                session_id=session_id,
                token_ids=resolved,
                manifest=manifest,
                tier_id=answer.tier_id,
                tier_provenance=self._provenance_of(answer),
                label=label,
                durable=durable,
            )
        )
        logger.info(
            "SESSION-CHECKPOINT %s session=%s tokens=%d pages=%d mamba=%s tier=%s (%s)",
            record.checkpoint_id,
            session_id,
            len(resolved),
            len(base_manifest["kv_keys"]),
            base_manifest["mamba_key"] or "-",
            record.tier_id,
            record.tier_provenance,
        )
        return record

    # -- restore gate + pinning ---------------------------------------------

    def _verified_record(self, checkpoint_id: str, tree) -> CheckpointRecord:
        record = self.ledger.get(checkpoint_id)
        ok, message = verify_restore(
            record.manifest,
            backend_exists_fn(tree),
            self._identity(),
            self._local_geometry(),
        )
        if not ok:
            raise SessionCheckpointError(
                f"cannot restore checkpoint {checkpoint_id}: {message}"
            )
        return record

    def _pin(self, record: CheckpointRecord, tree) -> BranchAccounting:
        """Lock the checkpoint's radix chain and report what is shared.

        ``inc_lock_ref`` on the matched node pins the whole ancestor chain
        against eviction -- the same reference the scheduler takes for a
        running request. Nothing is copied: a branch that later diverges makes
        the tree SPLIT at the divergence point, leaving this chain intact and
        shared. That is copy-on-restore, and the radix tree already does it.
        """
        result = match_prefix_result(tree, record.token_ids)
        matched = len(result.device_indices) + (result.host_hit_length or 0)
        accounting = account_branch(len(record.token_ids), matched)
        node = result.last_host_node
        if node is not None and node is not getattr(tree, "root_node", None):
            if record.checkpoint_id not in self._pinned_nodes:
                tree.inc_lock_ref(node)
                self._pinned_nodes[record.checkpoint_id] = node
        return accounting

    def _unpin(self, checkpoint_id: str) -> None:
        node = self._pinned_nodes.pop(checkpoint_id, None)
        if node is None:
            return
        tree = self.scheduler.tree_cache
        try:
            tree.dec_lock_ref(node)
        except Exception:
            logger.exception(
                "SESSION-CHECKPOINT failed to release the lock reference for "
                "%s; the pages stay pinned rather than being freed twice",
                checkpoint_id,
            )

    # -- branch -------------------------------------------------------------

    def import_bundle_and_seed(
        self,
        bundle_path: str,
        *,
        new_session_id: Optional[str] = None,
        store_put_fn: Optional[Callable[[str, bytes], None]] = None,
    ) -> Tuple[str, str]:
        """#411 Cut 3: seed a session from a portable bundle. Returns
        ``(new_session_id, detail)``.

        THE SAME ENTRY branch/rewind use, deliberately not a parallel one:
        ``_preconditions`` -> gate -> ``controller.branch_from`` -> ``_pin``
        -> ``ledger.derive``. A second seeding path would be a second place
        for the mid-generation rule, the TP=1 limit and the page_head refusal
        to be got wrong.

        ORDER IS THE GUARANTEE. Every refusal happens BEFORE
        ``branch_from``, which is the only step that creates a session. So a
        failure anywhere in this method leaves NO partial session -- there is
        nothing to roll back, because nothing was seeded. Blobs written to the
        store before the gate are content-addressed cache entries, not session
        state: they are harmless if the import then refuses, and they are what
        makes the store-backed verification possible at all.

        The bundle's own gate (identity, manifest version, blob presence, the
        #212 GDN clause, geometry) runs in ``session_portable.import_bundle``
        against the bundle's member names. It is re-run here through
        ``verify_restore`` against the STORE, because between those two
        moments the blobs move: passing the first proves the bundle is
        complete, passing the second proves this rank can actually read what
        it was handed.
        """
        from sglang.srt.managers.session_portable import import_bundle

        tree = self._preconditions()

        # 1. Gate the file. Refuses before any payload is extracted.
        manifest, blobs = import_bundle(
            bundle_path,
            local_identity=self._identity(),
            local_geometry=self._local_geometry(),
        )

        # 2. Mid-generation target: refuse by default (#410's rule). Checked
        # before anything is written, because a refusal that has already
        # spent store IO is a worse refusal.
        session_id = (manifest.get("checkpoint") or {}).get("session_id") or ""
        if session_id:
            conflict = self._in_flight_conflict(session_id)
            if conflict is not None:
                raise SessionCheckpointError(
                    f"session {session_id!r} has request {conflict} in flight; "
                    f"importing into a mid-generation session is refused. Wait "
                    f"for it to finish, or import into a new session."
                )

        # 3. Materialise the payloads into this rank's store.
        if store_put_fn is not None:
            for key, payload in blobs.items():
                store_put_fn(key, payload)

        # 4. The SAME verification the branch path runs, now against the
        # store rather than the bundle.
        ok, message = verify_restore(
            manifest,
            backend_exists_fn(tree),
            self._identity(),
            self._local_geometry(),
        )
        if not ok:
            raise SessionCheckpointError(
                f"imported bundle did not verify against this rank's store: "
                f"{message}"
            )

        # 4b. PIN what was just verified, before anything is created. An
        # imported session's pages are the youngest entries in the LRU and are
        # referenced by no running request until the seeded session uses them,
        # so a store under pressure can evict them before they are ever read.
        # Refuses by name if a page did not land; a session whose prefix is
        # already reclaimable must not be created.
        imported_id = (
            (manifest.get("checkpoint") or {}).get("checkpoint_id") or session_id or ""
        )
        import_pins = pin_imported_pages(self._file_tier_store(), imported_id, manifest)
        if not import_pins.protected:
            logger.info(
                "SESSION-IMPORT %s has NO file-tier protection: %s. The seeded "
                "session survives radix eviction only, not a restart.",
                imported_id,
                import_pins.reason,
            )

        controller = getattr(self.scheduler, "session_controller", None)
        if controller is None:
            raise SessionCheckpointError(
                "this scheduler has no session controller; seeding an "
                "imported session needs server-side sessions (/open_session)"
            )

        # 5. ONLY NOW is anything created.
        seeded = controller.branch_from(
            parent_session_id=session_id,
            checkpoint_tokens=list(manifest.get("token_ids") or []),
            new_session_id=new_session_id,
        )
        logger.info(
            "SESSION-CHECKPOINT imported %s -> session %s: %s",
            bundle_path,
            seeded,
            message,
        )
        return seeded, message

    def _branch_output(self, recv_req) -> SessionCheckpointReqOutput:
        from sglang.srt.managers.io_struct import SessionCheckpointReqOutput

        tree = self._preconditions()
        record = self._verified_record(recv_req.checkpoint_id or "", tree)
        controller = getattr(self.scheduler, "session_controller", None)
        if controller is None:
            raise SessionCheckpointError(
                "this scheduler has no session controller; branching needs "
                "server-side sessions (/open_session)"
            )
        new_session_id = controller.branch_from(
            parent_session_id=record.session_id,
            checkpoint_tokens=record.token_ids,
            new_session_id=recv_req.new_session_id,
        )
        accounting = self._pin(record, tree)
        self.ledger.derive(new_session_id, record.checkpoint_id)
        logger.info(
            "SESSION-CHECKPOINT branch %s -> session %s: %s",
            record.checkpoint_id,
            new_session_id,
            accounting.to_json(),
        )
        return SessionCheckpointReqOutput(
            success=True,
            message=(
                f"branched checkpoint {record.checkpoint_id} into session "
                f"{new_session_id}: {accounting.shared} of {accounting.total} "
                f"page(s) shared, {accounting.prefetch} to prefetch, "
                f"{accounting.copied} copied"
            ),
            checkpoint_id=record.checkpoint_id,
            session_id=new_session_id,
            info={"accounting": accounting.to_json(), **record.to_json()},
        )

    # -- rewind -------------------------------------------------------------

    def _rewind_output(self, recv_req) -> SessionCheckpointReqOutput:
        from sglang.srt.managers.io_struct import SessionCheckpointReqOutput

        tree = self._preconditions()
        record = self._verified_record(recv_req.checkpoint_id or "", tree)
        session_id = recv_req.session_id or record.session_id
        rid = self._in_flight_conflict(session_id)
        if rid is not None:
            raise SessionCheckpointError(
                f"session {session_id!r} has an in-flight request ({rid}); "
                "rewind refused -- it would cut the context out from under a "
                "running turn. Retry after it finishes."
            )
        controller = getattr(self.scheduler, "session_controller", None)
        if controller is None:
            raise SessionCheckpointError(
                "this scheduler has no session controller; rewind needs "
                "server-side sessions (/open_session)"
            )
        controller.rewind_to(session_id, record.token_ids)
        accounting = self._pin(record, tree)
        self.ledger.derive(session_id, record.checkpoint_id)
        logger.info(
            "SESSION-CHECKPOINT rewind session %s -> %s: %s",
            session_id,
            record.checkpoint_id,
            accounting.to_json(),
        )
        return SessionCheckpointReqOutput(
            success=True,
            message=(
                f"session {session_id} rewound to checkpoint "
                f"{record.checkpoint_id} ({len(record.token_ids)} tokens); "
                f"{accounting.shared} of {accounting.total} page(s) resident, "
                f"{accounting.prefetch} to prefetch -- no re-prefill"
            ),
            checkpoint_id=record.checkpoint_id,
            session_id=session_id,
            info={"accounting": accounting.to_json(), **record.to_json()},
        )
