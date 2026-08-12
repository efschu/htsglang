"""kv-session-offload spill destinations (#224).

Makes the spill TARGET selectable as an ordered destination list
(``--kv-session-offload-destinations``), including targets across rig
boundaries. Unset (default), nothing in this module runs and the offload
path is byte-identical to today.

Semantics
---------
``local`` is today's pinned host-RAM pool (live spill: the session keeps
decoding through the host tick). Every FURTHER entry is a PARK tier named
after a HiCache storage backend: when the local tier is exhausted (no free
host region under fresh spill pressure), the oldest eligible spilled
session's host tail is moved out of its local region into the park tier as
rank-local blobs, the region is freed for the next spill, and the session
is SUSPENDED -- it stops ticking but keeps ALL of its work and identity.
Unparking (when local pressure is gone and a region is free) reads the tail
back into a free region; from there the unchanged wave-back/restore
machinery returns it to the device. No re-prefill, no output cap. This is
the extension of the #236 budget exhaustion case: instead of demoting
(work -> HiCache, generation capped), remote capacity takes over -- see
``park_instead_of_demote``, the named seam the budget branch calls at its
demotion sites.

Order of the destination list is the user's choice, with ONE hard
constraint: ``local`` must be first. This is a physical boundary, not a
preference -- the device's D2H DMA lands in locally pinned host RAM, so
every further hop stages through the local tier; and the measured
cross-rig path (3.43 GB/s RDMA write / 1.47 us latency, PCIe-3.0-x4-bound,
2026-07-27) is orders of magnitude below the local host-RAM path, so a
network tier is a tier BELOW local host RAM, never above it.

What deliberately stays local (named limits, not omissions)
-----------------------------------------------------------
* GDN/Mamba recurrent state: device-resident, exactly as in the base
  offload (module invariant). It never travels, is never quantized, never
  crosses the wire: it is ~75 MB/session length-INDEPENDENT bf16 on the
  27B (the park tier exists for the KV tail's VOLUME, not this constant),
  and it is the most corruption-sensitive payload in the system (recurrent
  error accumulates). Parking therefore relieves the local host-RAM
  region, NOT the mamba pool and NOT the device head.
* The draft-KV bundle stash and tick hiddens (small pinned-CPU tensors on
  the SpillSlot) stay in process memory inside the ParkedSession record.

Rank uniformity
---------------
Per-rank network I/O completes at different times and must never feed a
decision directly. The I/O worker thread only sets rank-local (done, ok)
flags; the manager's single unconditional per-iteration MIN all-reduce
(``update_dcp_admission_state``) carries two extra elements [done, ok]
when this feature is armed (flag off -> today's 2-element payload,
byte-identical). Every state transition (commit / failover / timeout) is a
pure function of the REDUCED flags plus the replicated iteration counter
(``transfer_verdict``). Park/unpark DECISIONS are pure over replicated
scheduler state; blob sizes are rank-local (owner-compact shards) but the
choice of session and tier is identical on every rank.

Producer identity
-----------------
A remote entry is only a hit if producer == consumer on every axis that
shapes the bytes: model, quantization, KV dtype, TP size / rank-tp-ratio,
DCP geometry, head geometry, layer count. The fingerprint is stored in the
per-rank meta blob and verified explicitly before an unpark commits; a
mismatch is a miss (hard request error, never silent corruption). Keys are
rank-suffixed BY CONSTRUCTION: unlike HiCache's ``dcp_owner_mode`` shared
pages (file-backend-only, see cache_controller), these blobs are
rank-PRIVATE owner-compact region rows, for which rank-scoped keys are the
correct contract.

Supported park tiers (V1)
-------------------------
``file`` (validated end-to-end on CPU), ``mooncake`` (registered-buffer
pointer I/O straight from the pinned spill pool; the cross-rig remote
host-RAM tier), and ``dynamic`` (operator-supplied class implementing the
generic tensor ``set``/``get`` contract). Other HiCache backends (nixl,
eic, simm, mori, hf3fs, aibrix) are rejected by name at validation: their
arbitrary-blob ``set``/``get`` semantics are unverified (page-geometry- or
pointer-registration-bound interfaces); they remain reachable through
``dynamic`` with an adapter class. This is a named, liftable exclusion.

No delete API exists on the storage ABC; blobs of finished/unparked
sessions are left to the backend's own eviction (keys are rid-unique and
can never be re-matched). Counted in ``counters['blobs_orphaned']``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

#: Iterations between two #659 spill-ladder reports. A region shortfall can
#: repeat every iteration under sustained pressure; the ladder is an
#: accounting read, so it is reported once per episode rather than per decline.
_LADDER_REPORT_EVERY_ITERS: int = 512


def _local_host_name() -> str:
    """This machine's name, as the tier ids spell it.

    One spelling for the whole process: a tier id that disagrees with
    ``TierRegistry.local_host`` would make every local tier read as remote and
    silently invert the ladder's ``role()``.
    """
    import socket

    return socket.gethostname()

LOCAL_DESTINATION = "local"

# Park tiers accepted in V1 (see module docstring for why the other
# HiCache backend names are rejected). Kept in sync with the storage
# factory registry by a unit test.
SUPPORTED_PARK_BACKENDS = ("file", "mooncake", "dynamic")
# Full HiCache storage-backend namespace (mirror of
# StorageBackendFactory._registry plus "dynamic"; sync-tested) -- used to
# tell "known but unsupported" from "unknown" in error messages.
ALL_STORAGE_BACKENDS = (
    "file",
    "nixl",
    "mooncake",
    "hf3fs",
    "aibrix",
    "eic",
    "simm",
    "mori",
    "dynamic",
)

# A region shortfall (try_spill declined for lack of a free region) counts
# as "fresh pressure" for this many scheduler iterations: parking starts
# only under fresh pressure, unparking only without it. Together with the
# FIFO park order this is the anti-pendulum rule: a freed region is not
# immediately re-filled by unparking the session that just vacated it.
PARK_PRESSURE_WINDOW_ITERS = 64

# Meta-blob format version.
_META_VERSION = 1


# ---------------------------------------------------------------------------
# Pure helpers (CPU-tested)
# ---------------------------------------------------------------------------


def parse_destinations(spec: Optional[str]) -> List[str]:
    """Split the ``--kv-session-offload-destinations`` value into an ordered
    list of lower-cased tokens. Empty segments are dropped."""
    if spec is None:
        return []
    return [tok.strip().lower() for tok in str(spec).split(",") if tok.strip()]


def destinations_error(dests: List[str]) -> Optional[str]:
    """Validate the parsed destination list; None = valid.

    Rules (each names its reason):
    * non-empty;
    * ``local`` exactly once and FIRST (hard physical boundary: the D2H DMA
      lands in locally pinned host RAM, every further hop stages through
      it; measured cross-rig RDMA is 3.43 GB/s vs the far wider local
      host-RAM path, so a network tier is always BELOW local);
    * at least one park tier (a bare ``local`` is today's behaviour -- ask
      for it by NOT setting the flag);
    * park tiers must be supported backend names, no duplicates.
    """
    if not dests:
        return "destination list is empty."
    if dests[0] != LOCAL_DESTINATION:
        return (
            f"first destination must be '{LOCAL_DESTINATION}' (got "
            f"'{dests[0]}'). The device D2H copy lands in locally pinned "
            "host RAM, so every further tier stages through the local one; "
            "and the measured cross-rig path (3.43 GB/s RDMA write, 1.47 us "
            "latency) is orders of magnitude below local host RAM -- a "
            "remote tier is a tier BELOW local, never above it."
        )
    tail = dests[1:]
    if LOCAL_DESTINATION in tail:
        return "'local' may appear only once (first)."
    if not tail:
        return (
            "no park tier named after 'local'. A bare 'local' list is "
            "today's behaviour -- request it by not setting "
            "--kv-session-offload-destinations."
        )
    seen = set()
    for tok in tail:
        if tok in seen:
            return f"duplicate destination '{tok}'."
        seen.add(tok)
        if tok in SUPPORTED_PARK_BACKENDS:
            continue
        if tok in ALL_STORAGE_BACKENDS:
            return (
                f"storage backend '{tok}' is not a supported park tier in "
                f"V1 (supported: {', '.join(SUPPORTED_PARK_BACKENDS)}). Its "
                "arbitrary-blob set/get semantics are unverified "
                "(page-geometry- or registration-bound interface). It stays "
                "reachable via 'dynamic' with an adapter class implementing "
                "the generic tensor set/get contract."
            )
        return (
            f"unknown destination '{tok}' (supported park tiers: "
            f"{', '.join(SUPPORTED_PARK_BACKENDS)})."
        )
    return None


def producer_fingerprint(
    *,
    model_path: str,
    quantization: Optional[str],
    kv_cache_dtype: str,
    dtype: str,
    tp_size: int,
    rank_tp_ratio: Any,
    mode: str,
    split_factor: int,
    cp_prefix: List[int],
    layer_num: int,
    head_num: int,
    head_dim: int,
    store_dtype: str,
) -> str:
    """Canonical producer-identity string. Every axis that shapes the KV
    bytes or their owner split is included; equality of the WHOLE string is
    the hit condition for a remote entry (checked explicitly at unpark).
    The KV dtype IS part of this fingerprint -- unlike today's HiCache
    storage keys (owned by a parallel work item)."""
    payload = {
        "v": _META_VERSION,
        "model_path": str(model_path),
        "quantization": str(quantization),
        "kv_cache_dtype": str(kv_cache_dtype),
        "dtype": str(dtype),
        "tp_size": int(tp_size),
        "rank_tp_ratio": str(rank_tp_ratio),
        "mode": str(mode),
        "S": int(split_factor),
        "cp_prefix": [int(x) for x in cp_prefix],
        "layer_num": int(layer_num),
        "head_num": int(head_num),
        "head_dim": int(head_dim),
        "store_dtype": str(store_dtype),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def fingerprint_hash(fingerprint: str) -> str:
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def blob_key(fp_hash: str, rid: str, rank: int, generation: int, item: str) -> str:
    """Key of one parked blob. Rank-suffixed by construction: the payload
    is this rank's PRIVATE owner-compact shard (not a shared page). The
    park GENERATION makes every park episode's keys fresh: backends
    legitimately dedupe by key (HiCacheFile skips rewrites of an existing
    key), so re-parking a session that grew after an unpark must never
    reuse the previous episode's keys. The generation counter advances
    rank-uniformly (transfers begin at rank-uniform decisions)."""
    return f"kvso.{fp_hash}.{rid}.r{rank}.g{generation}.{item}"


def owned_tail_rows(residues: torch.Tensor, lo: int, hi: int) -> int:
    """Number of host-region rows THIS rank holds for a spilled tail whose
    per-position owner residues are ``residues``: exactly the positions
    whose residue falls in this rank's class ``[lo, hi)``. Matches the
    sequential owner-compact write order of the region (initial spill +
    per-tick appends), so it is also the parked blob's row count."""
    if residues.numel() == 0:
        return 0
    r = residues.to(torch.int64)
    return int(((r >= lo) & (r < hi)).sum().item())


def meta_blob(meta: Dict[str, Any]) -> torch.Tensor:
    """Serialize the per-rank park metadata as a uint8 tensor (transportable
    through the generic tensor set/get contract)."""
    raw = json.dumps(meta, sort_keys=True).encode("utf-8")
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


def parse_meta_blob(t: torch.Tensor) -> Optional[Dict[str, Any]]:
    """Inverse of meta_blob / padded_meta_blob: tolerates the fixed-size
    zero padding by trimming at the JSON object boundary."""
    try:
        raw = bytes(t.to(torch.uint8).contiguous().numpy().tobytes())
        end = raw.rfind(b"}")
        if end < 0:
            return None
        meta = json.loads(raw[: end + 1].decode("utf-8"))
        return meta if isinstance(meta, dict) else None
    except Exception:  # noqa: BLE001 -- any malformed blob is a miss
        return None


def meta_matches(
    meta: Optional[Dict[str, Any]], fingerprint: str, rid: str, rows: int
) -> Optional[str]:
    """Explicit producer-identity check before an unpark counts as a hit.
    Returns None on match, else the mismatch reason (a miss, surfaced as a
    hard request error -- never silently restored)."""
    if meta is None:
        return "meta blob missing or unparsable"
    if meta.get("v") != _META_VERSION:
        return f"meta version {meta.get('v')} != {_META_VERSION}"
    if meta.get("fingerprint") != fingerprint:
        return "producer fingerprint mismatch (model/quant/kv-dtype/geometry)"
    if meta.get("rid") != rid:
        return f"rid mismatch ({meta.get('rid')} != {rid})"
    if int(meta.get("rows", -1)) != int(rows):
        return f"row count mismatch ({meta.get('rows')} != {rows})"
    return None


def transfer_verdict(
    min_done: int, min_ok: int, abandoned: bool
) -> str:
    """Pure verdict over the RANK-UNIFORM (min-reduced) transfer flags:
    * "wait"   -- some rank's I/O still running. A transfer NEVER resolves
      before every rank's worker has stopped touching the pinned pool rows
      (releasing the region earlier would let a new spill D2H into memory a
      late worker still reads/writes);
    * "commit" -- every rank done and ok, and the transfer was not
      abandoned;
    * "failed" -- every rank done, and either some rank failed or the
      transfer was abandoned past the timeout (an abandoned transfer fails
      DETERMINISTICALLY even if the late I/O happened to succeed, so all
      ranks agree regardless of which side of the timeout their bytes
      landed on).
    All inputs are rank-uniform (reduced flags + the replicated abandoned
    flag), so every rank reaches the same verdict in the same iteration."""
    if min_done < 1:
        return "wait"
    if abandoned or min_ok < 1:
        return "failed"
    return "commit"


def should_abandon(min_done: int, iters_waited: int, timeout_iters: int) -> bool:
    """Rank-uniform timeout: past ``timeout_iters`` scheduler iterations
    (the uniform clock) an unfinished transfer is marked abandoned. This
    bounds how long the outcome stays open for SUCCESS -- the physical I/O
    is still awaited (see transfer_verdict) so pool rows are never handed
    out under a live worker."""
    return min_done < 1 and iters_waited > max(1, int(timeout_iters))


def park_decision(
    *,
    free_regions: int,
    pressure_iter: int,
    now_iter: int,
    inflight: bool,
    candidates: List[Tuple[int, int, int]],
    window: int = PARK_PRESSURE_WINDOW_ITERS,
) -> Optional[int]:
    """Pick the session to park, or None. Pure over replicated state.

    Preconditions: no free region, FRESH spill pressure (a try_spill
    declined for lack of a region within ``window`` iterations), no
    transfer already in flight. Candidates are ``(spill_iter,
    arrival_seq, rpi)`` of park-ELIGIBLE slots; the OLDEST spill wins
    (deterministic tiebreak by arrival seq) -- the session least likely to
    restore soon."""
    if inflight or free_regions > 0 or not candidates:
        return None
    if now_iter - pressure_iter > window:
        return None
    return min(candidates)[2]


def unpark_decision(
    *,
    free_regions: int,
    pressure_iter: int,
    now_iter: int,
    inflight: bool,
    have_parked: bool,
    window: int = PARK_PRESSURE_WINDOW_ITERS,
) -> bool:
    """Whether to start unparking the FIFO-oldest parked session. Pure.
    Requires a free region, NO fresh spill pressure (anti-pendulum mirror
    of park_decision), and no transfer in flight."""
    if inflight or not have_parked or free_regions <= 0:
        return False
    return now_iter - pressure_iter > window


# ---------------------------------------------------------------------------
# Blob-store tier adapters
# ---------------------------------------------------------------------------


class DestinationTier:
    """One park tier: a named blob store with a put/get-into contract.

    ``pointer_io`` tiers (mooncake) move bytes with (data_ptr, nbytes)
    pairs into memory REGISTERED with the store at attach (the pinned spill
    pool); tensor tiers (file, dynamic) move materialized tensors through
    the generic HiCacheStorage set/get. Neither path is a new transport --
    the backend carries the bytes."""

    def __init__(self, name: str, storage: Any, pointer_io: bool):
        self.name = name
        self.storage = storage
        self.pointer_io = pointer_io

    def put(self, key: str, tensor: torch.Tensor) -> bool:
        try:
            if self.pointer_io:
                t = tensor.contiguous()
                nbytes = t.numel() * t.element_size()
                return bool(
                    self.storage.set(
                        key, target_location=t.data_ptr(), target_sizes=nbytes
                    )
                )
            return bool(self.storage.set(key, value=tensor.contiguous()))
        except Exception as e:  # noqa: BLE001 -- I/O failure = tier failover
            logger.warning(
                "kv-session-offload destinations: put(%s) on tier '%s' "
                "failed: %r",
                key,
                self.name,
                e,
            )
            return False

    def get_into(self, key: str, tensor: torch.Tensor) -> bool:
        try:
            if self.pointer_io:
                nbytes = tensor.numel() * tensor.element_size()
                return bool(
                    self.storage.get(
                        key, target_location=tensor.data_ptr(), target_sizes=nbytes
                    )
                )
            return self.storage.get(key, target_location=tensor) is not None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "kv-session-offload destinations: get(%s) on tier '%s' "
                "failed: %r",
                key,
                self.name,
                e,
            )
            return False

    def get_meta(self, key: str, max_bytes: int = 1 << 20) -> Optional[Dict]:
        """Fetch + parse a meta blob. The generic get contract needs the
        size up front; metas are small, so probe with a fixed buffer and
        trim at the JSON boundary."""
        try:
            if self.pointer_io:
                buf = torch.zeros(max_bytes, dtype=torch.uint8, pin_memory=True)
                ok = self.storage.get(
                    key, target_location=buf.data_ptr(), target_sizes=max_bytes
                )
                if not ok:
                    return None
                raw = bytes(buf.numpy().tobytes())
            else:
                got = self.storage.get(
                    key, target_location=torch.zeros(max_bytes, dtype=torch.uint8)
                )
                if got is None:
                    return None
                raw = bytes(got.to(torch.uint8).numpy().tobytes())
            end = raw.rfind(b"}")
            if end < 0:
                return None
            meta = json.loads(raw[: end + 1].decode("utf-8", errors="strict"))
            return meta if isinstance(meta, dict) else None
        except Exception:  # noqa: BLE001
            return None


def make_tier(
    name: str,
    *,
    storage_config: Any,
    mem_pool_host: Any,
    extra_config: Optional[dict],
) -> DestinationTier:
    """Build one park tier through the HiCache storage factory (fail fast:
    called at manager init, so a missing package / unreachable master is a
    boot error naming the tier, not a runtime surprise)."""
    from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory

    storage = StorageBackendFactory.create_backend(
        name, storage_config, mem_pool_host
    )
    pointer_io = name == "mooncake"
    if name == "dynamic" and extra_config is not None:
        pointer_io = bool(extra_config.get("pointer_io", False))
    return DestinationTier(name, storage, pointer_io)


# Meta-blob get() on the file backend short-reads (meta is smaller than the
# probe buffer): HiCacheFile.get raises on short read. Read metas through
# exists()+direct sizing instead is backend-private; instead metas are
# PADDED to a fixed size at put time so the generic contract holds.
META_BLOB_BYTES = 8192


def padded_meta_blob(meta: Dict[str, Any]) -> torch.Tensor:
    t = meta_blob(meta)
    if t.numel() > META_BLOB_BYTES:
        raise ValueError(
            f"park meta blob {t.numel()} B exceeds fixed size {META_BLOB_BYTES} B"
        )
    out = torch.zeros(META_BLOB_BYTES, dtype=torch.uint8)
    out[: t.numel()] = t
    return out


# ---------------------------------------------------------------------------
# Transfer records + controller
# ---------------------------------------------------------------------------


@dataclass
class _Transfer:
    """Replicated decision state of the ONE in-flight transfer (serialized
    deliberately: the cross-rig path is 3.43 GB/s -- overlapping transfers
    would only hide the queue, not the cost). done/ok are RANK-LOCAL until
    min-reduced."""

    kind: str  # "park" | "unpark"
    rid: str
    rpi: int
    tier_index: int  # index into chain.tiers
    start_iter: int
    seq: int
    # unpark only: the region claimed for the read-back.
    region: int = -1
    payload: List[Tuple[str, torch.Tensor]] = field(default_factory=list)
    meta: Optional[Dict[str, Any]] = None
    expected_rows: int = 0
    generation: int = 0
    meta_key: str = ""
    # Set (rank-uniformly) when the timeout expired before every rank
    # finished; the transfer then resolves as "failed" once the I/O is
    # really over on every rank.
    abandoned: bool = False


@dataclass
class ParkedSession:
    """A session whose host tail lives in a park tier. The SpillSlot is
    kept WHOLE (draft-KV stash, tick hiddens, batch object) so unparking
    rebuilds nothing -- it just reattaches a region."""

    rid: str
    rpi: int
    slot: Any
    tier_index: int
    park_iter: int
    meta: Dict[str, Any]
    boundary: int
    length: int
    rows: int
    generation: int = 0


class SpillDestinationController:
    """Owns the destination chain, the parked set, the single in-flight
    transfer and its I/O worker. All DECISIONS are made by the manager-side
    flow (pure functions above) -- this class only executes and accounts."""

    def __init__(
        self,
        destinations: List[str],
        tiers: List[DestinationTier],
        fingerprint: str,
        rank: int,
        timeout_iters: int,
    ):
        self.destinations = destinations
        self.tiers = tiers  # park tiers only (destinations[1:])
        self.fingerprint = fingerprint
        self.fp_hash = fingerprint_hash(fingerprint)
        self.rank = int(rank)
        self.timeout_iters = int(timeout_iters)
        self.parked: OrderedDict[str, ParkedSession] = OrderedDict()
        self.inflight: Optional[_Transfer] = None
        self.pressure_iter = -(1 << 30)
        self.counters: Dict[str, int] = {
            "parks_started": 0,
            "parks_committed": 0,
            "parks_failed": 0,
            "parks_timeout": 0,
            "unparks_started": 0,
            "unparks_committed": 0,
            "unparks_failed": 0,
            "unparks_timeout": 0,
            "unpark_identity_miss": 0,
            "park_bytes_out": 0,
            "unpark_bytes_in": 0,
            "blobs_orphaned": 0,
            "parked_aborted": 0,
        }
        self._seq = 0
        # Park-episode generation: advances at every _start_park (a
        # rank-uniform decision), making each episode's blob keys fresh --
        # backends dedupe by key, so a re-park after growth must never
        # reuse the previous episode's keys.
        self.park_generation = 0
        # Rank-local completion flags of the current transfer, set by the
        # worker; read only through the min-reduce (reduce_extra ->
        # consume_reduced).
        self._io_done = 0
        self._io_ok = 0
        self._reduced_done = 0
        self._reduced_ok = 0
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="kvso-dest-io",
            daemon=True,
        )
        self._worker.start()

    # -- reduce plumbing ---------------------------------------------------

    def reduce_extra(self) -> List[int]:
        """Two extra MIN-reduce elements [done, ok] piggybacked on the
        manager's per-iteration collective. Only called when armed, so the
        flag-off payload stays byte-identical."""
        with self._lock:
            return [int(self._io_done), int(self._io_ok)]

    def consume_reduced(self, min_done: int, min_ok: int) -> None:
        self._reduced_done = int(min_done)
        self._reduced_ok = int(min_ok)

    def verdict(self, now_iter: int) -> Optional[str]:
        if self.inflight is None:
            return None
        t = self.inflight
        if not t.abandoned and should_abandon(
            self._reduced_done, now_iter - t.start_iter, self.timeout_iters
        ):
            t.abandoned = True
            key = "parks_timeout" if t.kind == "park" else "unparks_timeout"
            self.counters[key] += 1
            logger.error(
                "kv-session-offload destinations: %s of rid=%s on tier "
                "'%s' exceeded %d iterations; abandoned (will resolve as "
                "failed once every rank's I/O has stopped -- pool rows are "
                "never released under a live worker).",
                t.kind,
                t.rid,
                self.tiers[t.tier_index].name,
                self.timeout_iters,
            )
        return transfer_verdict(
            self._reduced_done, self._reduced_ok, t.abandoned
        )

    # -- pressure ----------------------------------------------------------

    def note_region_shortfall(self, now_iter: int, manager=None) -> None:
        self.pressure_iter = int(now_iter)
        if manager is not None:
            self._report_spill_ladder(manager)

    def _report_spill_ladder(self, manager) -> None:
        """#659: say WHICH tier declined and why, once per shortfall episode.

        A region shortfall is the exact moment the local host tier is full,
        and until now it was recorded only as a pressure timestamp -- the
        operator saw a spill decline with no statement of what was full or how
        full. The #407 registry answers that in the vocabulary the rest of the
        fleet already uses (headroom with a provenance, one named refusal per
        tier), and it renders ABSENCES rather than hiding them.

        Read-only: this reports the ladder, it does not pick the destination.
        The #224 chain still decides, so the spill path is byte-identical --
        the registry is under the rung as an observer in this cut, which is
        what makes its verdict comparable against the hardcoded law before
        anything is allowed to depend on it.

        Rate-limited to one report per episode, and every failure is swallowed:
        an accounting read may never turn a spill decline into a crash.
        """
        if self.pressure_iter - getattr(self, "_ladder_reported_iter", -(10**9)) < (
            _LADDER_REPORT_EVERY_ITERS
        ):
            return
        self._ladder_reported_iter = self.pressure_iter
        try:
            from sglang.srt.managers.kv_spill_tier_selection import (
                ladder_rows,
                live_kv_spill_ladder,
                local_first_disagreement,
            )

            registry = live_kv_spill_ladder(manager, local_host=_local_host_name())
            if registry is None:
                return
            for tier_id, role, headroom, bandwidth in ladder_rows(registry):
                logger.info(
                    "kv-spill ladder: %s (%s) headroom=%s bandwidth=%s",
                    tier_id,
                    role,
                    headroom,
                    bandwidth,
                )
            disagreement = local_first_disagreement(
                registry, 0, local_host=_local_host_name()
            )
            if disagreement:
                logger.warning("kv-spill ladder: %s", disagreement)
        except Exception as exc:  # noqa: BLE001 - report, never escalate
            logger.debug("kv-spill ladder report skipped: %r", exc)

    # -- transfer lifecycle ------------------------------------------------

    def begin(self, transfer: _Transfer, sync_events: List[Any]) -> None:
        assert self.inflight is None, "one transfer at a time"
        self._seq += 1
        transfer.seq = self._seq
        self.inflight = transfer
        with self._lock:
            self._io_done = 0
            self._io_ok = 0
        self._reduced_done = 0
        self._reduced_ok = 0
        self._jobs.put((transfer, sync_events))

    def finish(self) -> _Transfer:
        t = self.inflight
        assert t is not None
        self.inflight = None
        with self._lock:
            self._io_done = 0
            self._io_ok = 0
        self._reduced_done = 0
        self._reduced_ok = 0
        return t

    # -- worker ------------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            item = self._jobs.get()
            if item is None:
                return
            transfer, sync_events = item
            ok = True
            try:
                for evt in sync_events:
                    # CUDA events recorded on the copy/forward streams at
                    # enqueue: the pinned rows must not be read (park) or
                    # written (unpark) while a queued GPU copy still touches
                    # them. cudaEventSynchronize is thread-safe; absent on
                    # the CPU test path.
                    evt.synchronize()
                tier = self.tiers[transfer.tier_index]
                if transfer.kind == "park":
                    for key, tensor in transfer.payload:
                        if not tier.put(key, tensor):
                            ok = False
                            break
                        self.counters["park_bytes_out"] += (
                            tensor.numel() * tensor.element_size()
                        )
                else:
                    meta = tier.get_meta(transfer.meta_key, META_BLOB_BYTES)
                    err = meta_matches(
                        meta,
                        self.fingerprint,
                        transfer.rid,
                        transfer.expected_rows,
                    )
                    if err is not None:
                        self.counters["unpark_identity_miss"] += 1
                        logger.error(
                            "kv-session-offload destinations: unpark of "
                            "rid=%s from tier '%s' is a MISS: %s. The "
                            "parked entry does not verify against this "
                            "producer -- refusing to restore it.",
                            transfer.rid,
                            tier.name,
                            err,
                        )
                        ok = False
                    else:
                        for key, tensor in transfer.payload:
                            if not tier.get_into(key, tensor):
                                ok = False
                                break
                            self.counters["unpark_bytes_in"] += (
                                tensor.numel() * tensor.element_size()
                            )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "kv-session-offload destinations: %s transfer of rid=%s "
                    "failed on rank %d: %r",
                    transfer.kind,
                    transfer.rid,
                    self.rank,
                    e,
                )
                ok = False
            with self._lock:
                # Publish only if this is still the current transfer (a
                # timeout may already have abandoned it; its late result
                # must not leak into the next transfer's flags).
                if self.inflight is not None and self.inflight.seq == transfer.seq:
                    self._io_done = 1
                    self._io_ok = 1 if ok else 0


# ---------------------------------------------------------------------------
# Manager-facing glue (called from KVSessionOffloadManager, all gated on
# manager._dest being non-None)
# ---------------------------------------------------------------------------


def attach_destinations(mgr) -> Optional[SpillDestinationController]:
    """Build the controller at manager init. Returns None when the flag is
    unset (the byte-identical default). Fails FAST on an invalid list or an
    unreachable/unimportable tier, naming the tier."""
    sa = mgr.scheduler.server_args
    spec = getattr(sa, "kv_session_offload_destinations", None)
    if not spec:
        return None
    dests = parse_destinations(spec)
    err = destinations_error(dests)
    if err is not None:
        raise ValueError(f"--kv-session-offload-destinations: {err}")

    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

    extra = getattr(sa, "kv_session_offload_destination_extra_config", None)
    extra_cfg = json.loads(extra) if extra else None

    host_pool = mgr.host_pool
    fingerprint = producer_fingerprint(
        model_path=str(getattr(sa, "model_path", "")),
        quantization=getattr(sa, "quantization", None),
        kv_cache_dtype=str(getattr(sa, "kv_cache_dtype", "auto")),
        dtype=str(getattr(sa, "dtype", "auto")),
        tp_size=int(getattr(sa, "tp_size", 1)),
        rank_tp_ratio=getattr(sa, "rank_tp_ratio", None),
        mode=mgr.mode,
        split_factor=mgr.S,
        cp_prefix=list(mgr.cp_prefix),
        layer_num=int(getattr(host_pool, "layer_num", 0)),
        head_num=int(getattr(host_pool, "head_num", 0)),
        head_dim=int(getattr(host_pool, "head_dim", 0)),
        store_dtype=str(getattr(host_pool, "dtype", "")),
    )
    storage_config = HiCacheStorageConfig(
        tp_rank=int(getattr(mgr.scheduler, "tp_rank", 0)),
        tp_size=int(getattr(sa, "tp_size", 1)),
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=str(getattr(sa, "served_model_name", None) or ""),
        extra_config=extra_cfg,
        dcp_owner_mode=False,  # blobs are rank-private; rank-suffixed keys
    )
    tiers = []
    for name in dests[1:]:
        tiers.append(
            make_tier(
                name,
                storage_config=storage_config,
                mem_pool_host=host_pool,
                extra_config=extra_cfg,
            )
        )
    ctl = SpillDestinationController(
        destinations=dests,
        tiers=tiers,
        fingerprint=fingerprint,
        rank=int(getattr(mgr.scheduler, "tp_rank", 0)),
        timeout_iters=int(
            getattr(sa, "kv_session_offload_park_timeout_iters", 512)
        ),
    )
    logger.info(
        "kv-session-offload destinations ARMED: %s (park tiers: %s; "
        "fingerprint %s). Local host RAM stays the first tier; further "
        "tiers PARK suspended sessions (work kept, no re-prefill). GDN "
        "state stays device-resident and never crosses the wire.",
        ",".join(dests),
        ",".join(t.name for t in tiers),
        ctl.fp_hash,
    )
    return ctl


def _tail_geometry(mgr, slot) -> Optional[Tuple[int, int, torch.Tensor, int]]:
    """(L, boundary, residues, rows) of the slot's spilled tail, derived
    exactly like _restore does. None when the tail cannot be parked (e.g.
    rank owns rows but the row scan is inconsistent)."""
    req = slot.req
    if slot.batch is not None:
        L = int(slot.batch.seq_lens_cpu[0].item())
    else:
        L = len(req.origin_input_ids) + len(req.output_ids)
        if len(req.output_ids) > 0:
            L -= 1
    row = mgr.req_to_token_pool.req_to_token[req.req_pool_idx, :L]
    boundary = int((row < mgr.host_base).sum().item())
    seg = row[boundary:L]
    if seg.numel() == 0:
        return None
    if mgr.mode == "weighted":
        residues = (seg.to(torch.int64) % mgr.S).contiguous()
    else:
        residues = (
            torch.arange(boundary, L, dtype=torch.int64, device=row.device)
            % mgr.S
        )
    rows = owned_tail_rows(residues, mgr.lo, mgr.hi)
    return L, boundary, residues, rows


def park_eligible(mgr, slot, last_batch) -> bool:
    """Rank-uniform park eligibility (every input is replicated state):
    quiescent (its last host tick is settled), no wave-back drain yet
    (host_row_base == 0 -- the region must hold the FULL tail contiguously
    from row 0), not spec-in-tick (device-suffix bookkeeping stays local),
    not already pending, not finished."""
    if getattr(slot, "park_pending", False) or slot.req.finished():
        return False
    if getattr(slot, "spec_in_tick", False):
        return False
    if not getattr(slot, "adopted", True):
        return False  # born-spilled handover not completed yet
    bslot = mgr.backend._sess_slots.get(slot.req.req_pool_idx)
    if bslot is None or int(getattr(bslot, "host_row_base", 0)) != 0:
        return False
    if slot.batch is not None and last_batch is slot.batch:
        return False  # tick result not settled yet
    if slot.batch is None and mgr._iter_ct <= slot.spill_iter:
        return False
    return True


def _region_row_views(mgr, region_base: int, rows: int):
    """Per-layer contiguous K/V row views of a region's first ``rows``
    host rows (layer_first layout). These views ARE the pinned pool memory
    -- pointer-io tiers transfer straight out of registered RAM."""
    pool = mgr.host_pool
    out = []
    layer_num = int(getattr(pool, "layer_num", 0))
    for fl in range(layer_num):
        out.append(("k%d" % fl, pool.k_data_refs[fl][region_base : region_base + rows]))
        out.append(("v%d" % fl, pool.v_data_refs[fl][region_base : region_base + rows]))
    return out


def _record_sync_events(mgr) -> List[Any]:
    """CUDA events ordering the I/O thread after any queued GPU copy that
    still touches the pool (tick prefetch H2D reads / per-tick D2H
    appends). CPU/unit path: no real streams -> no events."""
    events = []
    copy_stream = getattr(mgr.backend, "_sess_copy_stream", None)
    if isinstance(copy_stream, torch.cuda.Stream):
        evt = torch.cuda.Event()
        evt.record(copy_stream)
        events.append(evt)
        evt2 = torch.cuda.Event()
        evt2.record(torch.cuda.current_stream())
        events.append(evt2)
    return events


def _start_park(mgr, ctl: SpillDestinationController, rpi: int) -> None:
    slot = mgr.spills.get(rpi)
    if slot is None:
        return
    geo = _tail_geometry(mgr, slot)
    if geo is None:
        slot.park_pending = False
        return
    L, boundary, _residues, rows = geo
    region_base = slot.region * mgr.region_tokens
    ctl.park_generation += 1
    gen = ctl.park_generation
    rid = str(slot.req.rid)
    meta = {
        "v": _META_VERSION,
        "fingerprint": ctl.fingerprint,
        "rid": rid,
        "rows": int(rows),
        "L": int(L),
        "boundary": int(boundary),
        "rank": ctl.rank,
        "generation": gen,
    }
    meta_key = blob_key(ctl.fp_hash, rid, ctl.rank, gen, "meta")
    payload: List[Tuple[str, torch.Tensor]] = [(meta_key, padded_meta_blob(meta))]
    for item, view in _region_row_views(mgr, region_base, rows):
        payload.append((blob_key(ctl.fp_hash, rid, ctl.rank, gen, item), view))
    ctl.counters["parks_started"] += 1
    ctl.begin(
        _Transfer(
            kind="park",
            rid=rid,
            rpi=rpi,
            tier_index=0,
            start_iter=mgr._iter_ct,
            seq=0,
            payload=payload,
            meta=meta,
            expected_rows=rows,
            generation=gen,
            meta_key=meta_key,
        ),
        _record_sync_events(mgr),
    )
    mgr._log(
        "kv-session-offload destinations: PARK start rid=%s tier=%s L=%d "
        "boundary=%d rows(rank)=%d region=%d",
        slot.req.rid,
        ctl.tiers[0].name,
        L,
        boundary,
        rows,
        slot.region,
    )


def _commit_park(mgr, ctl: SpillDestinationController, t: _Transfer) -> None:
    slot = mgr.spills.pop(t.rpi, None)
    if slot is None:  # finished/aborted while transferring: blobs orphan
        ctl.counters["blobs_orphaned"] += 1
        return
    mgr._free_regions.append(slot.region)
    mgr.backend._sess_close_slot(t.rpi)
    slot.park_pending = False
    region = slot.region
    slot.region = -1
    slot.req.kv_spill_state = "parked"
    ctl.parked[t.rid] = ParkedSession(
        rid=t.rid,
        rpi=t.rpi,
        slot=slot,
        tier_index=t.tier_index,
        park_iter=mgr._iter_ct,
        meta=t.meta or {},
        boundary=int((t.meta or {}).get("boundary", 0)),
        length=int((t.meta or {}).get("L", 0)),
        rows=t.expected_rows,
        generation=t.generation,
    )
    ctl.counters["parks_committed"] += 1
    mgr._log(
        "kv-session-offload destinations: PARK commit rid=%s tier=%s "
        "region %d freed (parked=%d)",
        t.rid,
        ctl.tiers[t.tier_index].name,
        region,
        len(ctl.parked),
    )


def _start_unpark(mgr, ctl: SpillDestinationController) -> None:
    rid, rec = next(iter(ctl.parked.items()))
    if not mgr._free_regions:
        return
    region = mgr._free_regions.pop(0)
    region_base = region * mgr.region_tokens
    payload: List[Tuple[str, torch.Tensor]] = []
    for item, view in _region_row_views(mgr, region_base, rec.rows):
        payload.append(
            (blob_key(ctl.fp_hash, rid, ctl.rank, rec.generation, item), view)
        )
    ctl.counters["unparks_started"] += 1
    ctl.begin(
        _Transfer(
            kind="unpark",
            rid=rid,
            rpi=rec.rpi,
            tier_index=rec.tier_index,
            start_iter=mgr._iter_ct,
            seq=0,
            region=region,
            payload=payload,
            expected_rows=rec.rows,
            generation=rec.generation,
            meta_key=blob_key(ctl.fp_hash, rid, ctl.rank, rec.generation, "meta"),
        ),
        _record_sync_events(mgr),
    )
    mgr._log(
        "kv-session-offload destinations: UNPARK start rid=%s tier=%s "
        "rows(rank)=%d -> region %d",
        rid,
        ctl.tiers[rec.tier_index].name,
        rec.rows,
        region,
    )


def _commit_unpark(mgr, ctl: SpillDestinationController, t: _Transfer) -> None:
    rec = ctl.parked.pop(t.rid, None)
    if rec is None:
        mgr._free_regions.append(t.region)
        return
    slot = rec.slot
    slot.region = t.region
    mgr.spills[rec.rpi] = slot
    mgr.backend._sess_open_slot(rec.rpi, t.region * mgr.region_tokens)
    slot.req.kv_spill_state = "host"
    slot.suppress_tick = False
    ctl.counters["unparks_committed"] += 1
    ctl.counters["blobs_orphaned"] += 1  # remote copy now stale, rid-unique
    mgr._log(
        "kv-session-offload destinations: UNPARK commit rid=%s -> region %d "
        "(host-resident again; wave-back/restore unchanged from here)",
        t.rid,
        t.region,
    )


def _fail_unpark(mgr, ctl: SpillDestinationController, t: _Transfer, why: str) -> None:
    """An unpark that cannot complete (identity miss, tier loss, timeout)
    is a hard, NAMED request error -- the parked KV is unrecoverable, and
    silently resuming would corrupt. The region goes back to the pool."""
    mgr._free_regions.append(t.region)
    rec = ctl.parked.pop(t.rid, None)
    ctl.counters["unparks_failed"] += 1
    logger.error(
        "kv-session-offload destinations: UNPARK of rid=%s FAILED (%s); "
        "aborting the request (its parked KV cannot be restored).",
        t.rid,
        why,
    )
    if rec is not None:
        req = rec.slot.req
        req.to_abort = True
        req.to_abort_message = (
            f"kv-session-offload: parked session could not be restored from "
            f"tier '{ctl.tiers[rec.tier_index].name}' ({why})."
        )
        _release_parked_req(mgr, rec)


def parked_session_ended(req) -> bool:
    """Whether a PARKED session has ended and must be reaped.

    ``req.finished()`` alone is not the signal here. ``Scheduler.abort_request``
    reaches parked sessions on purpose (``inflight_batches`` extends with
    ``parked_inflight_entries``) and marks them the same way it marks every
    other in-flight request -- "abort method 3", ``req.to_finish =
    FINISH_ABORT()`` -- leaving the promotion to ``finished_reason`` to
    ``Req.update_finish_state``, which runs only while processing a batch
    result. A parked session is in no batch and runs no forward (``_commit_park``
    popped it out of ``mgr.spills`` and its tick is suppressed), so that
    promotion never happens: keying the reap on ``finished()`` dropped the abort
    silently and leaked the device head, the req-pool slot, the tree lock and
    the remote blob, while the client waited forever.

    Pure function of replicated request state (``abort_request`` marks the same
    rid on every rank in the same iteration), so the reap set is rank-uniform.
    """
    return req.finished() or getattr(req, "to_finish", None) is not None


def _stream_terminal(mgr, req) -> None:
    """Emit one terminal output for a request that ends outside any batch.

    Tolerant by design: the CPU unit path drives this module with a stub
    scheduler that carries no output streamer, and a missing streamer must not
    turn a memory-release path into a crash."""
    streamer = getattr(getattr(mgr, "scheduler", None), "output_streamer", None)
    if streamer is None:
        return
    streamer.stream_output([req], getattr(req, "return_logprob", False))


def _release_parked_req(mgr, rec: ParkedSession) -> None:
    """Cleanup of a parked session that ends without unparking (abort or
    unrecoverable unpark). Mirrors release_finished_spilled_req minus the
    region/backend-slot part (already released at park commit) and minus
    the spec-suffix part (spec-in-tick sessions are park-ineligible)."""
    req = rec.slot.req
    pool = mgr.req_to_token_pool
    rpi = req.req_pool_idx
    # Promote a pending abort BEFORE releasing: nothing else will. The normal
    # promotion site (Req.update_finish_state) only runs on a batch result, and
    # this session never produces one. Without it every later `finished()`
    # check -- reap loops, abort re-scans, the tokenizer's terminal handling --
    # would keep seeing an unfinished request whose memory is already gone.
    if not req.finished() and getattr(req, "to_finish", None) is not None:
        req.finished_reason = req.to_finish
        req.to_finish = None
        # ...and TELL the client. A device-resident or host-spilled abort gets
        # its terminal chunk from the one decode forward the request still runs
        # ("the request will still run one decode forward pass. Then we reuse
        # all existing code to clean up"). A parked session runs no forward, so
        # this is the only place that can emit it -- same single-request emit
        # the scheduler uses for requests that terminate before ever entering a
        # batch. Without it the abort frees the memory and the caller hangs.
        _stream_terminal(mgr, req)
    if rpi is None:
        return
    boundary = rec.boundary
    protected = int(req.cache_protected_len or 0)
    if boundary > protected:
        head = pool.req_to_token[rpi, protected:boundary]
        mgr.allocator.free(head.to(torch.int64))
    if req.last_node is not None:
        mgr.tree_cache.dec_lock_ref(req.last_node)
    if req.mamba_pool_idx is not None and hasattr(pool, "free_mamba_cache"):
        pool.free_mamba_cache(req)
    if rec.slot.batch is not None:
        rec.slot.batch.filter_batch()
    pool.free(req)
    req.kv_spill_state = None
    req.kv_spill_boundary = 0
    rb = getattr(mgr.scheduler, "running_batch", None)
    if rb is not None:
        rb.batch_is_full = False


def maybe_park_flow(mgr, running_batch, last_batch):
    """Per-iteration destination step, called from pre_schedule (after the
    reap loop, before the fast-lane spill). No-op shape when the controller
    is unarmed is proven by a unit test. Order:

    1. reap parked sessions whose request finished (abort);
    2. advance the in-flight transfer by the RANK-UNIFORM verdict;
    3. else start an unpark (free region, pressure gone, FIFO oldest);
    4. else start a park (no free region, fresh pressure, oldest eligible
       spilled session; two-phase: mark park_pending first so the tick
       stops, enqueue the I/O one settled iteration later).
    """
    ctl: Optional[SpillDestinationController] = mgr._dest
    if ctl is None:
        return running_batch
    now = mgr._iter_ct

    # 1. Reap aborted parked sessions (see parked_session_ended: a parked
    #    session's abort arrives as `to_finish`, which nothing else promotes).
    for rid in [
        r for r, rec in ctl.parked.items() if parked_session_ended(rec.slot.req)
    ]:
        rec = ctl.parked.pop(rid)
        ctl.counters["parked_aborted"] += 1
        ctl.counters["blobs_orphaned"] += 1
        _release_parked_req(mgr, rec)

    # 2. In-flight transfer: rank-uniform verdict.
    v = ctl.verdict(now)
    if v is not None and v != "wait":
        t = ctl.finish()
        if t.kind == "park":
            if v == "commit":
                _commit_park(mgr, ctl, t)
            else:
                slot = mgr.spills.get(t.rpi)
                if slot is not None:
                    slot.park_pending = False
                ctl.counters["parks_failed"] += 1
                if t.tier_index + 1 < len(ctl.tiers):
                    # Failover: the next tier in the user's order takes
                    # over. Restart from the pending mark next iteration.
                    if slot is not None:
                        slot.park_pending = True
                        ctl.begin(
                            _Transfer(
                                kind="park",
                                rid=t.rid,
                                rpi=t.rpi,
                                tier_index=t.tier_index + 1,
                                start_iter=now,
                                seq=0,
                                payload=t.payload,
                                meta=t.meta,
                                expected_rows=t.expected_rows,
                                generation=t.generation,
                                meta_key=t.meta_key,
                            ),
                            _record_sync_events(mgr),
                        )
                        ctl.counters["parks_started"] += 1
                        mgr._log(
                            "kv-session-offload destinations: PARK failover "
                            "rid=%s -> tier=%s",
                            t.rid,
                            ctl.tiers[t.tier_index + 1].name,
                        )
                else:
                    mgr._log(
                        "kv-session-offload destinations: PARK of rid=%s "
                        "abandoned (%s, no further tier); session stays "
                        "host-resident (today's behaviour).",
                        t.rid,
                        v,
                    )
        else:  # unpark
            if v == "commit":
                _commit_unpark(mgr, ctl, t)
            else:
                _fail_unpark(
                    mgr,
                    ctl,
                    t,
                    "abandoned past timeout" if t.abandoned else "read/verify failed",
                )
        return running_batch

    if ctl.inflight is not None:
        return running_batch

    # 3. Unpark: FIFO oldest, only when pressure is gone (anti-pendulum).
    if unpark_decision(
        free_regions=len(mgr._free_regions),
        pressure_iter=ctl.pressure_iter,
        now_iter=now,
        inflight=False,
        have_parked=bool(ctl.parked),
    ):
        _start_unpark(mgr, ctl)
        return running_batch

    # 4. Park: fresh pressure, no free region, oldest eligible session.
    #    Two-phase: an already-pending candidate whose tick has settled is
    #    enqueued; otherwise the chosen candidate is marked pending (its
    #    tick stops this iteration, the transfer starts on a later one).
    pending = [
        s for s in mgr.spills.values() if getattr(s, "park_pending", False)
    ]
    if pending:
        slot = min(
            pending, key=lambda s: (s.spill_iter, s.req.kv_arrival_seq)
        )
        if slot.req.finished():
            slot.park_pending = False
            return running_batch
        # park_pending stopped its tick; enqueue once the last tick result
        # is settled (same quiescence rule as the restore path).
        settled = slot.batch is None or last_batch is not slot.batch
        if settled:
            _start_park(mgr, ctl, slot.req.req_pool_idx)
        return running_batch
    candidates = [
        (s.spill_iter, s.req.kv_arrival_seq, rpi)
        for rpi, s in mgr.spills.items()
        if park_eligible(mgr, s, last_batch)
    ]
    rpi = park_decision(
        free_regions=len(mgr._free_regions),
        pressure_iter=ctl.pressure_iter,
        now_iter=now,
        inflight=False,
        candidates=candidates,
    )
    if rpi is not None:
        mgr.spills[rpi].park_pending = True
        mgr._log(
            "kv-session-offload destinations: park PENDING rid=%s "
            "(tick stops; transfer starts once settled)",
            mgr.spills[rpi].req.rid,
        )
    return running_batch


def park_instead_of_demote(mgr, slot) -> bool:
    """SEAM for the #236 spill-budget branch: called at its demotion sites
    (BEFORE _budget_demote). When the destination chain is armed and the
    slot is park-eligible, the session is marked park-pending -- the budget
    exhaustion extends into the next tier instead of demoting (work kept
    AND liveness kept, merely suspended). Returns True when parking takes
    over, False -> demote as before. On the base branch the region-pressure
    trigger in maybe_park_flow is the only caller-equivalent.

    Budget note (for the merge): a committed park moves the tail OUT of the
    local host volume -- the budget branch discounts it at its
    _budget_note_* sites when this returns True."""
    ctl: Optional[SpillDestinationController] = mgr._dest
    if ctl is None or ctl.inflight is not None:
        return False
    if getattr(slot, "park_pending", False):
        return True  # already on its way out
    if not park_eligible(mgr, slot, None):
        return False
    slot.park_pending = True
    return True


def parked_reqs(mgr) -> List[Any]:
    """The requests of currently PARKED sessions.

    A parked session is live and still owns device-side bookkeeping -- its
    req-pool slot, its radix tree lock and, notably, its GDN/Mamba state slot
    -- but ``_commit_park`` popped it out of ``mgr.spills``, so anything that
    enumerates live offload sessions through ``spills`` alone does not see it.
    Every such consumer must go through here (or through
    :meth:`KVSessionOffloadManager.live_offload_reqs`) instead.
    """
    ctl: Optional[SpillDestinationController] = mgr._dest
    if ctl is None:
        return []
    return [rec.slot.req for rec in ctl.parked.values()]


def parked_inflight_entries(mgr) -> List[Any]:
    """Parked sessions for the manager's abort scanning (they are in no
    real batch while parked)."""
    from types import SimpleNamespace

    return [SimpleNamespace(reqs=[req]) for req in parked_reqs(mgr)]
