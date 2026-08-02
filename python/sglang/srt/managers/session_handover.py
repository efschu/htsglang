"""Live session handover without server stop (#261, second half).

The merged #261 flow moves a session between geometries through the HiCache
``file`` store, but requires the source server to be DOWN before the
umsharder runs -- the only reason being that the migration reads raw store
files with no coordination against a live writer. This module adds that
coordination at SESSION scope, reusing the #329 five-phase vocabulary:

* QUIESCE  -- refuse if the session has an in-flight request; PARK the token
  prefix (new requests extending it are refused with a named error).
* SNAPSHOT -- force write-through + storage backup of the session's radix
  chain (KV pages AND the GDN/Mamba blob -- the #212 lesson: the recurrent
  state must travel explicitly, the store's prefix matching would silently
  truncate without it), drain bounded, then emit a MANIFEST naming every
  blob that constitutes the session.
* RE-FORM  -- the manifest-scoped umsharder
  (``hicache_migrate --manifest``) runs OUTSIDE this process; the manifest
  files are complete and immutable (parked session, content-addressed
  store), so a live source store is safe to read.
* RESTORE  -- the destination's ``verify_import``: every manifest key is
  present in ITS store and the model identity hash matches (#241 gate
  crossed visibly; geometry is the only thing allowed to differ).
* RESUME   -- the destination serves the next turn (storage prefetch hit);
  the source ``commit`` releases the parked prefix.

Rollback rule (mirrors #329): ``abort`` is legal at any point BEFORE the
source receives ``commit`` -- the snapshot only ever ADDED files to the
store, nothing on the source was mutated. After ``commit`` there is
deliberately no rollback: two servers both believing they own a session is
the failure mode this state machine exists to prevent.

Collective discipline: this path issues NO group collective. The source side
is a single scheduler process (live export is declared TP=1-source-only, see
``_export_preconditions``); the destination side answers from rank-local
``exists()`` checks fanned out over the existing ZMQ control plane. All
waits are bounded deadlines (#259/#312), and a drain timeout rolls the
session back (unpark) before raising -- never a wedged session.

The pure pieces (ledger, manifest, gates) are separated from the scheduler
adapter so the falsifiers run hermetically: a planted GDN-state omission
MUST fail the completeness gate, and the gate must also be able to pass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import (
        SessionHandoverReqInput,
        SessionHandoverReqOutput,
    )

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1

# Ledger states. PARKED and EXPORTED are the ACTIVE states (prefix refusal
# in force, rollback legal); COMMITTED and ABORTED are terminal.
STATE_PARKED = "parked"
STATE_EXPORTED = "exported"
STATE_COMMITTED = "committed"
STATE_ABORTED = "aborted"
_ACTIVE_STATES = (STATE_PARKED, STATE_EXPORTED)


class SessionHandoverError(Exception):
    """A handover step that cannot proceed. The message always names why and,
    where retrying can help, says so."""


def prefixes_conflict(a: Sequence[int], b: Sequence[int]) -> bool:
    """True when one token sequence is a prefix of the other (equal counts
    as both). Distinct conversations diverge early, so this conservative
    rule costs nothing in practice."""
    n = min(len(a), len(b))
    if n == 0:
        return False
    return list(a[:n]) == list(b[:n])


def handover_id_for(token_ids: Sequence[int], identity_hash: str) -> str:
    payload = identity_hash + ":" + ",".join(str(t) for t in token_ids)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class HandoverRecord:
    handover_id: str
    token_ids: List[int]
    state: str = STATE_PARKED
    manifest: Optional[Dict] = None
    created_unix: float = field(default_factory=time.time)


class HandoverLedger:
    """State machine over active handovers. Pure, hermetically testable."""

    def __init__(self):
        self._records: Dict[str, HandoverRecord] = {}

    def park(self, token_ids: Sequence[int], identity_hash: str) -> HandoverRecord:
        conflict = self.conflicts(token_ids)
        if conflict is not None:
            raise SessionHandoverError(
                f"prefix conflicts with active handover {conflict}; commit or "
                "abort it first"
            )
        hid = handover_id_for(token_ids, identity_hash)
        record = HandoverRecord(handover_id=hid, token_ids=list(token_ids))
        self._records[hid] = record
        return record

    def get(self, handover_id: str) -> HandoverRecord:
        record = self._records.get(handover_id)
        if record is None:
            raise SessionHandoverError(f"unknown handover id {handover_id!r}")
        return record

    def mark_exported(self, handover_id: str, manifest: Dict) -> None:
        record = self.get(handover_id)
        if record.state != STATE_PARKED:
            raise SessionHandoverError(
                f"handover {handover_id} is {record.state}, not {STATE_PARKED}"
            )
        record.manifest = manifest
        record.state = STATE_EXPORTED

    def commit(self, handover_id: str) -> HandoverRecord:
        record = self.get(handover_id)
        if record.state != STATE_EXPORTED:
            raise SessionHandoverError(
                f"cannot commit handover {handover_id}: state is "
                f"{record.state}, commit requires {STATE_EXPORTED} (the "
                "destination must have verified its import first)"
            )
        record.state = STATE_COMMITTED
        return record

    def abort(self, handover_id: str) -> HandoverRecord:
        record = self.get(handover_id)
        if record.state == STATE_COMMITTED:
            raise SessionHandoverError(
                f"cannot abort handover {handover_id}: already committed. "
                "There is no rollback after commit -- the destination owns "
                "the session now."
            )
        if record.state == STATE_ABORTED:
            raise SessionHandoverError(f"handover {handover_id} already aborted")
        record.state = STATE_ABORTED
        return record

    def conflicts(self, token_ids: Sequence[int]) -> Optional[str]:
        """Id of an ACTIVE handover whose prefix conflicts, else None."""
        for record in self._records.values():
            if record.state in _ACTIVE_STATES and prefixes_conflict(
                record.token_ids, token_ids
            ):
                return record.handover_id
        return None

    def active_count(self) -> int:
        return sum(1 for r in self._records.values() if r.state in _ACTIVE_STATES)


# ---------------------------------------------------------------------------
# Manifest: build, completeness gate (source), import gate (destination)
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    handover_id: str,
    model_identity_hash: str,
    token_ids: Sequence[int],
    kv_keys: Sequence[str],
    mamba_key: Optional[str],
    hybrid_gdn: bool,
    draft_keys: Sequence[str],
    tp_size: int,
    tp_rank: int,
    dcp_owner_mode: bool,
    page_size: int,
) -> Dict:
    return {
        "version": MANIFEST_VERSION,
        "handover_id": handover_id,
        "model_identity_hash": model_identity_hash,
        "source": {
            "tp_size": tp_size,
            "tp_rank": tp_rank,
            "dcp_owner_mode": dcp_owner_mode,
            "page_size": page_size,
        },
        "token_ids": list(token_ids),
        "kv_keys": list(kv_keys),
        "mamba_key": mamba_key,
        "hybrid_gdn": hybrid_gdn,
        "draft_keys": list(draft_keys),
        "created_unix": time.time(),
    }


def validate_manifest_completeness(
    manifest: Dict, exists_fn: Callable[[str], bool]
) -> Dict[str, int]:
    """The source-side gate: every blob the manifest names must be IN the
    store before the manifest may leave this process.

    The GDN clause is the #212 falsifier hook: for a hybrid model a missing
    (or absent-from-store) mamba blob fails LOUDLY here -- the store route
    would otherwise truncate the prefix match at the destination and
    silently re-prefill, which for a recurrent state means a wrong session,
    not a slow one.
    """
    if not manifest["kv_keys"]:
        raise SessionHandoverError("manifest has no KV keys; nothing to hand over")
    missing = [k for k in manifest["kv_keys"] if not exists_fn(k)]
    if missing:
        raise SessionHandoverError(
            f"{len(missing)} of {len(manifest['kv_keys'])} KV page(s) absent "
            f"from the store after the drain: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''} -- refusing to emit a "
            "partial manifest"
        )
    if manifest["hybrid_gdn"]:
        mamba_key = manifest.get("mamba_key")
        if not mamba_key:
            raise SessionHandoverError(
                "hybrid-GDN model but the manifest carries no mamba key: the "
                "recurrent state would not travel (#212) and the destination "
                "would silently re-prefill a WRONG session. Export refused."
            )
        if not exists_fn(mamba_key):
            raise SessionHandoverError(
                f"hybrid-GDN model but the GDN state blob {mamba_key!r} is "
                "absent from the store after the drain (#212: the recurrent "
                "state must travel explicitly). Export refused."
            )
    checked = {
        "kv_keys": len(manifest["kv_keys"]),
        "mamba": 1 if manifest.get("mamba_key") else 0,
        "draft_keys": len(manifest.get("draft_keys") or []),
    }
    return checked


def verify_import(
    manifest: Dict,
    exists_fn: Callable[[str], bool],
    local_identity_hash: str,
) -> tuple:
    """The destination-side (request-less) gate, rank-local.

    Presence + identity only, by design: byte sizes are gated inside the
    umsharder (hard errors naming both numbers) and at read time (a short
    read is an error). Identity must MATCH -- the #241 storage keys carry
    model identity so a different model/dtype/kv-format is a clean miss, and
    a handover crosses geometry only, never identity.
    """
    if manifest.get("version") != MANIFEST_VERSION:
        return False, (
            f"manifest version {manifest.get('version')!r} != "
            f"{MANIFEST_VERSION}; refusing to guess the schema"
        )
    if manifest["model_identity_hash"] != local_identity_hash:
        return False, (
            f"model identity mismatch: manifest "
            f"{manifest['model_identity_hash']} vs local {local_identity_hash} "
            "-- a handover crosses geometry, never model/dtype/kv-format"
        )
    missing = [k for k in manifest["kv_keys"] if not exists_fn(k)]
    mamba_key = manifest.get("mamba_key")
    if manifest.get("hybrid_gdn"):
        if not mamba_key:
            return False, (
                "hybrid-GDN manifest without a mamba key -- the source gate "
                "should have refused this export (#212)"
            )
        if not exists_fn(mamba_key):
            missing.append(mamba_key)
    if missing:
        return False, (
            f"{len(missing)} manifest blob(s) absent from this rank's store: "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''} -- run the "
            "manifest-scoped umsharder for this geometry first"
        )
    return True, (
        f"import verified: {len(manifest['kv_keys'])} KV page(s)"
        + (", GDN state present" if mamba_key else "")
        + (
            f", {len(manifest['draft_keys'])} draft blob(s) declared"
            if manifest.get("draft_keys")
            else ""
        )
    )


# ---------------------------------------------------------------------------
# Snapshot: the reusable half
#
# #410 (server-side checkpoints) is the same export with a different
# destination -- a storage TIER instead of a peer group -- so the snapshot
# lives here as module-level functions rather than inside the handover
# runtime. There is exactly ONE session serialization in the fork; a second
# one would be a second thing to keep byte-correct, and the #212 GDN gate
# below is the reason that matters.
# ---------------------------------------------------------------------------


def storage_preconditions(scheduler):
    """The tree, if this server can serialize a session at all."""
    tree = scheduler.tree_cache
    if not getattr(tree, "enable_storage", False):
        raise SessionHandoverError(
            "hierarchical cache with a storage backend is required "
            "(--enable-hierarchical-cache --hicache-storage-backend file)"
        )
    if scheduler.server_args.hicache_storage_backend != "file":
        raise SessionHandoverError(
            "session handover supports the 'file' storage backend only "
            "(the same limit the offline flow carries)"
        )
    return tree


def backend_exists_fn(tree) -> Callable[[str], bool]:
    backend = tree.cache_controller.storage_backend
    return lambda key: bool(backend.exists(key))


def drain_storage(tree, deadline_s: float) -> None:
    """Bounded drain of the write-through and storage-backup pipelines
    (#259/#312 discipline: a deadline, then a loud error -- and the caller
    rolls the park back, never a wedged session)."""
    controller = tree.cache_controller
    deadline = time.monotonic() + max(1.0, float(deadline_s))
    while time.monotonic() < deadline:
        tree.writing_check()
        tree.drain_storage_control_queues()
        if (
            len(tree.ongoing_write_through) == 0
            and len(tree.ongoing_backup) == 0
            and controller.backup_queue.qsize() == 0
            and controller.ack_backup_queue.qsize() == 0
        ):
            return
        time.sleep(0.02)
    raise SessionHandoverError(
        "storage drain did not complete within "
        f"{deadline_s:.1f}s: ongoing_write={len(tree.ongoing_write_through)} "
        f"ongoing_backup={len(tree.ongoing_backup)} "
        f"backup_queue={controller.backup_queue.qsize()} "
        f"ack_backup_queue={controller.ack_backup_queue.qsize()} -- "
        "export rolled back; retry with a larger deadline_s"
    )


def match_prefix_result(tree, token_ids: Sequence[int]):
    """``tree.match_prefix`` with the sequence-type discipline applied.

    The ``array("q")`` conversion is load-bearing: ``RadixKey.match`` asserts
    both sides carry the SAME sequence type, and every key already in the tree
    was built from the scheduler's ``array("q")`` fill ids. A plain JSON list
    off the control plane would abort the match with a bare AssertionError
    instead of matching the prefix. Every control-plane caller goes through
    this function so the rule lives in one place.

    Returns the raw result, including a PARTIAL match -- #410's branch
    accounting needs to know how much of a checkpoint the tree still holds,
    which is not an error condition there.
    """
    from array import array

    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    return tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", token_ids))))


def match_session_chain(tree, token_ids: Sequence[int]) -> List:
    """The radix chain root -> leaf that holds ``token_ids``, or a refusal.

    Shared by the handover snapshot and by #410's checkpoint export, which
    need the FULL prefix resident: a snapshot carries what the tree holds and
    nothing less.
    """
    result = match_prefix_result(tree, token_ids)
    matched = len(result.device_indices) + (result.host_hit_length or 0)
    if matched < len(token_ids):
        raise SessionHandoverError(
            f"only {matched} of {len(token_ids)} tokens are resident in "
            "the radix tree; a snapshot carries what the tree holds, "
            "nothing less -- refusing a partial session. (For a hybrid "
            "GDN model the resumable length is the deepest GDN "
            "checkpoint; snapshot at that boundary.)"
        )

    chain = []
    node = result.last_host_node
    while node is not tree.root_node and node is not None:
        chain.append(node)
        node = node.parent
    chain.reverse()
    if not chain:
        raise SessionHandoverError("prefix matched no radix node")
    return chain


def export_session_snapshot(
    scheduler,
    tree,
    *,
    snapshot_id: str,
    token_ids: Sequence[int],
    identity_hash: str,
    deadline_s: float,
) -> Dict:
    """Flush one session's KV pages AND its GDN blob to the store, then emit
    the manifest that names every blob.

    This is the whole of #261's SNAPSHOT phase, callable by anything that
    needs a session on stable storage. ``snapshot_id`` lands in the
    manifest's ``handover_id`` field: the field is the snapshot's identity,
    and #410 checkpoints reuse the same content-addressed derivation
    (:func:`handover_id_for`) rather than adding a parallel id space.
    """
    token_ids = list(token_ids)
    chain = match_session_chain(tree, token_ids)

    # SNAPSHOT: force host backup for nodes that never crossed the
    # write-through threshold; storage backup follows automatically on
    # the ack (write-through -> _finish_write_through_ack ->
    # write_backup_storage). Already host-backed nodes are pushed to
    # storage directly -- the store is content-addressed, so re-writing
    # an existing key is a no-op-equivalent, never a corruption.
    for n in chain:
        if not n.backuped:
            if n.evicted:
                raise SessionHandoverError(
                    f"radix node {n.id} is device-evicted but not "
                    "host-backed; this cannot happen under write_through "
                    "-- refusing to guess the state"
                )
            tree.write_backup(n)
        else:
            tree.write_backup_storage(n)
    drain_storage(tree, deadline_s)

    kv_keys: List[str] = []
    for n in chain:
        if not n.hash_value:
            raise SessionHandoverError(
                f"radix node {n.id} carries no page hashes after the "
                "drain; storage keying is incomplete -- export refused"
            )
        kv_keys.extend(n.hash_value)
    if len(kv_keys) != len(token_ids):
        raise SessionHandoverError(
            f"page-hash count ({len(kv_keys)}) != token count "
            f"({len(token_ids)}) at page_size == 1 -- bigram/page "
            "alignment mismatch, refusing to guess the mapping"
        )

    # ONE source for "does this model carry recurrent state": the
    # BasePrefixCache capability, which every tree implements. Sniffing a
    # concrete class's helper (``mamba_archive_transfers`` lives only on
    # HiMambaRadixCache) silently reports False on the UnifiedRadixCache
    # that --enable-hierarchical-cache actually builds for a hybrid SSM
    # model -- and a False here disables the #212 gate on BOTH sides, so
    # the recurrent state never travels and the destination re-prefills a
    # WRONG session. That is the exact failure this gate exists to catch.
    hybrid_gdn = bool(tree.supports_mamba())
    mamba_key = f"{kv_keys[-1]}.mamba" if hybrid_gdn else None

    exists = backend_exists_fn(tree)
    controller = tree.cache_controller
    draft_keys = (
        [k for k in (f"{h}.draft" for h in kv_keys) if exists(k)]
        if getattr(controller, "has_draft", False)
        else []
    )

    args = scheduler.server_args
    storage_config = getattr(controller, "storage_config", None)
    manifest = build_manifest(
        handover_id=snapshot_id,
        model_identity_hash=identity_hash,
        token_ids=token_ids,
        kv_keys=kv_keys,
        mamba_key=mamba_key,
        hybrid_gdn=hybrid_gdn,
        draft_keys=draft_keys,
        tp_size=args.tp_size,
        tp_rank=0,
        dcp_owner_mode=bool(getattr(storage_config, "dcp_owner_mode", False)),
        page_size=args.page_size,
    )
    validate_manifest_completeness(manifest, exists)
    return manifest


# ---------------------------------------------------------------------------
# Scheduler adapter
# ---------------------------------------------------------------------------


class SessionHandoverRuntime:
    """Thin adapter between the scheduler and the pure pieces above.

    Runs entirely on the scheduler thread (control requests are processed
    between scheduling iterations), so the tree cannot mutate under the
    snapshot -- the in-flight check, the chain walk and the flush are atomic
    with respect to the radix tree. The cost is honest and bounded: OTHER
    sessions' scheduling stalls for at most the drain deadline during the
    SNAPSHOT phase; they are never stopped, never restarted.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.ledger = HandoverLedger()
        self._identity_hash: Optional[str] = None

    # -- admission hook -----------------------------------------------------

    def parked_conflict(self, token_ids: Sequence[int]) -> Optional[str]:
        if self.ledger.active_count() == 0:
            return None
        hid = self.ledger.conflicts(token_ids)
        if hid is None:
            return None
        return (
            f"the requested prefix is parked for live session handover "
            f"{hid}; retry after the handover commits or aborts"
        )

    # -- control dispatch ----------------------------------------------------

    def handle(self, recv_req: SessionHandoverReqInput) -> SessionHandoverReqOutput:
        from sglang.srt.managers.io_struct import SessionHandoverReqOutput

        try:
            action = recv_req.action
            if action == "export":
                manifest = self._export(recv_req.token_ids or [], recv_req.deadline_s)
                return SessionHandoverReqOutput(
                    success=True,
                    message=f"exported handover {manifest['handover_id']}",
                    manifest_json=json.dumps(manifest),
                )
            if action == "commit":
                record = self.ledger.commit(recv_req.handover_id or "")
                logger.info("SESSION-HANDOVER commit %s", record.handover_id)
                return SessionHandoverReqOutput(
                    success=True, message=f"committed {record.handover_id}"
                )
            if action == "abort":
                record = self.ledger.abort(recv_req.handover_id or "")
                logger.info("SESSION-HANDOVER abort %s", record.handover_id)
                return SessionHandoverReqOutput(
                    success=True,
                    message=f"aborted {record.handover_id}; session resumes here",
                )
            if action == "verify_import":
                ok, msg = self._verify_import(recv_req.manifest_json or "")
                return SessionHandoverReqOutput(success=ok, message=msg)
            return SessionHandoverReqOutput(
                success=False,
                message=f"unknown action {action!r}; known: export, commit, "
                "abort, verify_import",
            )
        except SessionHandoverError as e:
            return SessionHandoverReqOutput(success=False, message=str(e))
        except Exception as e:  # never crash the scheduler over control plane
            logger.exception("SESSION-HANDOVER %s failed", recv_req.action)
            return SessionHandoverReqOutput(
                success=False, message=f"{type(e).__name__}: {e}"
            )

    # -- shared bits ----------------------------------------------------------

    def _storage_preconditions(self):
        return storage_preconditions(self.scheduler)

    def _identity(self) -> str:
        if self._identity_hash is None:
            from sglang.srt.mem_cache.hicache_storage import (
                compute_model_identity_hash,
            )

            self._identity_hash = compute_model_identity_hash(
                self.scheduler.server_args
            )
        return self._identity_hash

    def _backend_exists(self, tree) -> Callable[[str], bool]:
        return backend_exists_fn(tree)

    # -- export (QUIESCE + SNAPSHOT) ------------------------------------------

    def _export_preconditions(self, tree):
        args = self.scheduler.server_args
        if args.tp_size != 1 or args.pp_size != 1:
            raise SessionHandoverError(
                "live export is built for a TP=1, PP=1 source only (the "
                "fast-card -> group direction); a TP>1 live export needs "
                "per-rank manifest merging and is a named follow-up -- use "
                "the stop-based flow there"
            )
        if args.page_size != 1:
            raise SessionHandoverError(
                f"page_size == 1 is required (got {args.page_size}); the "
                "umsharder inherits this limit from dcp_owner_mode"
            )

    def _in_flight_conflict(self, token_ids: Sequence[int]) -> Optional[str]:
        reqs = []
        running = getattr(self.scheduler, "running_batch", None)
        if running is not None and getattr(running, "reqs", None):
            reqs.extend(running.reqs)
        reqs.extend(getattr(self.scheduler, "waiting_queue", []) or [])
        chunked = getattr(self.scheduler, "chunked_req", None)
        if chunked is not None:
            reqs.append(chunked)
        for req in reqs:
            seq = list(req.origin_input_ids) + list(req.output_ids or [])
            if prefixes_conflict(seq, token_ids):
                return req.rid
        return None

    def _export(self, token_ids: List[int], deadline_s: float) -> Dict:
        if not token_ids:
            raise SessionHandoverError("export needs the session's token_ids")
        tree = self._storage_preconditions()
        self._export_preconditions(tree)

        rid = self._in_flight_conflict(token_ids)
        if rid is not None:
            raise SessionHandoverError(
                f"session has an in-flight request ({rid}); quiesce refused. "
                "Retry after the request finishes -- the natural handover "
                "point is between turns."
            )

        # QUIESCE: park FIRST, so nothing can start extending the prefix
        # while we snapshot. Every failure below unparks (rollback).
        record = self.ledger.park(token_ids, self._identity())
        try:
            manifest = self._snapshot(tree, record, token_ids, deadline_s)
        except BaseException:
            self.ledger.abort(record.handover_id)
            raise
        self.ledger.mark_exported(record.handover_id, manifest)
        logger.info(
            "SESSION-HANDOVER exported %s: %d tokens, %d KV pages, mamba=%s, "
            "%d draft blob(s)",
            record.handover_id,
            len(token_ids),
            len(manifest["kv_keys"]),
            manifest["mamba_key"] or "-",
            len(manifest["draft_keys"]),
        )
        return manifest

    def _snapshot(self, tree, record, token_ids, deadline_s: float) -> Dict:
        return export_session_snapshot(
            self.scheduler,
            tree,
            snapshot_id=record.handover_id,
            token_ids=token_ids,
            identity_hash=self._identity(),
            deadline_s=deadline_s,
        )

    def _drain_storage(self, tree, deadline_s: float) -> None:
        drain_storage(tree, deadline_s)

    # -- destination side (RESTORE gate) --------------------------------------

    def _verify_import(self, manifest_json: str):
        if not manifest_json:
            raise SessionHandoverError("verify_import needs manifest_json")
        tree = self._storage_preconditions()
        try:
            manifest = json.loads(manifest_json)
        except json.JSONDecodeError as e:
            raise SessionHandoverError(f"manifest_json is not valid JSON: {e}")
        return verify_import(manifest, self._backend_exists(tree), self._identity())
