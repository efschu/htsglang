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
"""The cold expert tier in SHARED host memory, so a delegated expert stays
reachable (#394).

**The wall this removes.** #394 apportions a rank's cold experts across the TP
group in proportion to each rank's PCIe bandwidth. Measured 2026-08-02 on
V4-Flash TP=3, that refused to serve: under the #82 GGUF expert-dim shard the
ranks hold DISJOINT expert ranges, so "delegating" rank 0's expert 80 to rank 1
does not relocate it -- rank 1 never had it, and rank 0 just dropped it. The
first token routed to expert 80 died in ``ExpertResidencyPlanner.resolve``.

**Why shared memory is the right answer and not a new tier.** All three ranks'
cold pools were already host DRAM in the same machine. Nothing needs to move to
a different medium; what was missing is that each pool lived in one process's
private anonymous pages, so a peer could not address it. A named POSIX shared
segment is the same DRAM with a name, and ``cudaHostRegister`` makes it DMA
source for whichever process registers it. So this is a MAPPING change:

* the bytes stay where they were, in host DRAM, one copy;
* the owner still writes its own experts at load time, exactly as before;
* a peer that needs a delegated expert maps the owner's segment once and DMAs
  the row straight to its own device -- over its OWN PCIe link, which is the
  entire point of apportioning by link bandwidth.

**What this module is NOT.** It is not a transport, not a cache and not a
tier. It hands out validated, read-only host views of a peer's cold rows. The
existing fetch path issues the same ``copy_`` it always did.

Layout and identity
-------------------

A segment is described by a :class:`ColdTierLayout` and every layout is
identity-keyed by the owner's card UUID and PCI BDF (#397 canon), never by rank
index alone: rank indices are a launch artifact, and a manifest that outlived
its launch would otherwise silently name a different card. The manifest is
published as a file, which is the same collective-free channel
``registry.rank_cards`` uses -- and it is read LAZILY, at first fetch, long
after every rank has finished loading, so nothing here can turn into a barrier
inside the weight-load loop.

Torn mappings fail loudly
-------------------------

Every segment carries a fixed header: magic, format version, instance id, owner
rank, row count, row bytes, and a digest of the layout that produced it. A
reader validates all of it against the layout it was handed before returning a
single byte. A stale segment from a previous launch, a segment written by a
different build, a manifest that disagrees with the segment it points at, or a
row index out of range are each a named exception. **Serving plausible-looking
wrong bytes is the failure this design is most exposed to, so it is the one
thing the header exists to make impossible.**
"""

from __future__ import annotations

import hashlib
import json
import logging
import mmap
import os
import struct
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "COLD_TIER_FORMAT",
    "HEADER_BYTES",
    "ColdTierLayout",
    "ColdTierError",
    "SegmentMismatch",
    "ManifestUnavailable",
    "shm_dir",
    "manifest_path",
    "segment_path",
    "create_owned_segment",
    "publish_manifest",
    "read_peer_manifest",
    "attach_peer_segment",
    "detach_all",
    "shm_capacity_bytes",
]

#: Bump when the HEADER or the manifest schema changes meaning. A segment
#: written by another version is rejected, never reinterpreted.
COLD_TIER_FORMAT = 1

_MAGIC = b"SGLCOLD1"
#: magic(8) fmt(4) rank(4) rows(8) row_bytes(8) instance(16) digest(16) = 64,
#: padded to 128 so the payload starts on a page-friendly boundary and a future
#: field does not move every row.
_HEADER_STRUCT = struct.Struct("<8sIIQQ16s16s")
HEADER_BYTES = 128


class ColdTierError(RuntimeError):
    """Base for every refusal in this module."""


class SegmentMismatch(ColdTierError):
    """A segment does not describe what the layout says it should.

    Raised in preference to returning bytes, always. This is the class that
    stands between a torn mapping and silently wrong expert weights.
    """


class ManifestUnavailable(ColdTierError):
    """A peer's manifest is not there (yet, or at all), with the reason."""


def shm_dir() -> str:
    """Where segments and manifests live. Overridable for tests."""
    return os.environ.get("SGLANG_MOE_COLD_TIER_SHM_DIR", "/dev/shm")


def shm_capacity_bytes(path: Optional[str] = None) -> Optional[int]:
    """Total bytes the shm filesystem can hold, or ``None`` if unknowable.

    Worth checking BEFORE staging: ``/dev/shm`` is a tmpfs with its own size
    cap (63 GiB by default on this box) while the cold tier can want more. The
    pages are RAM either way, so this is an accounting limit rather than a real
    memory constraint -- but hitting it mid-load is a hard failure at the worst
    possible moment, and the operator can raise it with a remount.
    """
    try:
        st = os.statvfs(path or shm_dir())
        return int(st.f_blocks) * int(st.f_frsize)
    except OSError:
        return None


@dataclass(frozen=True)
class ColdTierLayout:
    """Where one rank's cold rows live, and whose card they belong to.

    ``expert_ids[j]`` is the global expert id occupying row ``j``. Ascending and
    contiguous, matching the spill-pool convention the rest of the offload uses
    (``_spill_pool_index``), so a peer resolves an expert to a row by lookup
    rather than by arithmetic it could get wrong.
    """

    instance_id: str
    owner_rank: int
    owner_card_uuid: str
    owner_pci_bdf: str
    layer_key: str
    param_attr: str
    rows: int
    row_bytes: int
    dtype: str
    row_shape: Tuple[int, ...]
    expert_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.rows < 0 or self.row_bytes < 0:
            raise ValueError("rows and row_bytes must be non-negative")
        if self.expert_ids and len(self.expert_ids) != self.rows:
            raise ValueError(
                f"layout claims {self.rows} rows but names "
                f"{len(self.expert_ids)} expert ids"
            )
        if not self.owner_card_uuid:
            raise ValueError(
                "a cold-tier layout must name the owner's card UUID; a rank "
                "index alone is a launch artifact and cannot identify a card "
                "across process boundaries (#397)"
            )

    @property
    def nbytes(self) -> int:
        return HEADER_BYTES + self.rows * self.row_bytes

    @property
    def segment_name(self) -> str:
        return (
            f"sgl-cold-{self.instance_id}-r{self.owner_rank}-"
            f"{self.layer_key}-{self.param_attr}"
        )

    def row_of(self, expert_id: int) -> int:
        """Row holding this expert, or a named error. Never a guess."""
        try:
            return self.expert_ids.index(int(expert_id))
        except ValueError:
            raise SegmentMismatch(
                f"expert {expert_id} is not in rank {self.owner_rank}'s cold "
                f"tier for {self.layer_key}/{self.param_attr}; it holds "
                f"{list(self.expert_ids)[:8]}{'...' if self.rows > 8 else ''}"
            ) from None

    def digest(self) -> bytes:
        """16-byte digest of everything a reader must agree with."""
        payload = json.dumps(
            {
                "format": COLD_TIER_FORMAT,
                "instance_id": self.instance_id,
                "owner_rank": self.owner_rank,
                "owner_card_uuid": self.owner_card_uuid,
                "owner_pci_bdf": self.owner_pci_bdf,
                "layer_key": self.layer_key,
                "param_attr": self.param_attr,
                "rows": self.rows,
                "row_bytes": self.row_bytes,
                "dtype": self.dtype,
                "row_shape": list(self.row_shape),
                "expert_ids": list(self.expert_ids),
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).digest()[:16]

    def to_json(self) -> dict:
        d = {
            "instance_id": self.instance_id,
            "owner_rank": self.owner_rank,
            "owner_card_uuid": self.owner_card_uuid,
            "owner_pci_bdf": self.owner_pci_bdf,
            "layer_key": self.layer_key,
            "param_attr": self.param_attr,
            "rows": self.rows,
            "row_bytes": self.row_bytes,
            "dtype": self.dtype,
            "row_shape": list(self.row_shape),
            "expert_ids": list(self.expert_ids),
        }
        return d

    @classmethod
    def from_json(cls, d: dict) -> "ColdTierLayout":
        return cls(
            instance_id=str(d["instance_id"]),
            owner_rank=int(d["owner_rank"]),
            owner_card_uuid=str(d["owner_card_uuid"]),
            owner_pci_bdf=str(d.get("owner_pci_bdf", "")),
            layer_key=str(d["layer_key"]),
            param_attr=str(d["param_attr"]),
            rows=int(d["rows"]),
            row_bytes=int(d["row_bytes"]),
            dtype=str(d["dtype"]),
            row_shape=tuple(int(x) for x in d.get("row_shape", ())),
            expert_ids=tuple(int(e) for e in d.get("expert_ids", ())),
        )


def segment_path(layout: ColdTierLayout) -> str:
    return os.path.join(shm_dir(), layout.segment_name)


def manifest_path(instance_id: str, rank: int) -> str:
    return os.path.join(shm_dir(), f"sgl-cold-{instance_id}-r{rank}.manifest.json")


def _write_header(buf: mmap.mmap, layout: ColdTierLayout) -> None:
    buf[: _HEADER_STRUCT.size] = _HEADER_STRUCT.pack(
        _MAGIC,
        COLD_TIER_FORMAT,
        layout.owner_rank,
        layout.rows,
        layout.row_bytes,
        layout.instance_id.encode()[:16].ljust(16, b"\0"),
        layout.digest(),
    )


def _validate_header(buf, layout: ColdTierLayout, where: str) -> None:
    raw = bytes(buf[: _HEADER_STRUCT.size])
    try:
        magic, fmt, rank, rows, row_bytes, instance, digest = _HEADER_STRUCT.unpack(raw)
    except struct.error as exc:
        raise SegmentMismatch(f"{where}: header is unreadable ({exc})") from exc
    if magic != _MAGIC:
        raise SegmentMismatch(
            f"{where}: not a cold-tier segment (magic {magic!r}); refusing to "
            "read bytes out of something this process did not write"
        )
    if fmt != COLD_TIER_FORMAT:
        raise SegmentMismatch(
            f"{where}: written by cold-tier format {fmt}, this build speaks "
            f"{COLD_TIER_FORMAT}. Rejected rather than reinterpreted"
        )
    want = layout.instance_id.encode()[:16].ljust(16, b"\0")
    if instance != want:
        raise SegmentMismatch(
            f"{where}: segment belongs to instance "
            f"{instance.rstrip(bytes(1))!r}, "
            f"this launch is {layout.instance_id!r}. A leftover segment from a "
            "previous run is the classic source of plausible wrong weights"
        )
    if (
        rank != layout.owner_rank
        or rows != layout.rows
        or row_bytes != layout.row_bytes
    ):
        raise SegmentMismatch(
            f"{where}: segment says (rank={rank}, rows={rows}, "
            f"row_bytes={row_bytes}) but the layout says (rank="
            f"{layout.owner_rank}, rows={layout.rows}, "
            f"row_bytes={layout.row_bytes})"
        )
    if digest != layout.digest():
        raise SegmentMismatch(
            f"{where}: the manifest and the segment describe different layouts "
            "(digest mismatch). One of them is stale"
        )


def create_owned_segment(layout: ColdTierLayout):
    """Create (or truncate) this rank's segment and return an mmap over it.

    The owner writes rows into ``memoryview(mm)[HEADER_BYTES:]``. The header is
    written LAST by :func:`seal_owned_segment` so a peer that maps the segment
    early sees an invalid header rather than a half-filled pool.
    """
    path = segment_path(layout)
    size = layout.nbytes
    capacity = shm_capacity_bytes()
    if capacity is not None and size > capacity:
        raise ColdTierError(
            f"cold-tier segment for {layout.layer_key}/{layout.param_attr} "
            f"needs {size / 2**30:.2f} GiB but {shm_dir()} holds at most "
            f"{capacity / 2**30:.2f} GiB. These pages are RAM either way -- "
            f"this is the tmpfs size cap, not a memory shortage. Remount with "
            f"a larger size= to proceed"
        )
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    # Deliberately zeroed magic until sealed.
    mm[:HEADER_BYTES] = b"\0" * HEADER_BYTES
    return mm


def seal_owned_segment(mm, layout: ColdTierLayout) -> None:
    """Stamp the header. Call ONLY after every row is written.

    Sealing last is what makes "mapped but not ready" distinguishable from
    "ready": before this, the magic is zero and every reader refuses.
    """
    _write_header(mm, layout)
    mm.flush()


def publish_manifest(
    instance_id: str, rank: int, layouts: Sequence[ColdTierLayout]
) -> str:
    """Write this rank's layouts where peers can find them, atomically.

    Atomic rename, so a peer either sees the complete manifest or no manifest.
    A partially written JSON that happens to parse is exactly the kind of thing
    that produces a wrong row index.
    """
    path = manifest_path(instance_id, rank)
    payload = {
        "format": COLD_TIER_FORMAT,
        "instance_id": instance_id,
        "rank": rank,
        "written": time.time(),
        "layouts": [layout.to_json() for layout in layouts],
    }
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    logger.info(
        "cold-tier manifest published: rank %d, %d layouts -> %s",
        rank,
        len(layouts),
        path,
    )
    return path


def read_peer_manifest(
    instance_id: str, rank: int, timeout_s: float = 0.0
) -> Dict[Tuple[str, str], ColdTierLayout]:
    """A peer's layouts, keyed by ``(layer_key, param_attr)``.

    ``timeout_s`` is a BOUNDED wait, and it exists only because a peer may be a
    few milliseconds behind at the very first fetch. It is not a barrier: it
    expires with a named error instead of hanging the group, and every caller
    reaches it after load, not inside it.
    """
    path = manifest_path(instance_id, rank)
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        try:
            with open(path) as f:
                payload = json.load(f)
            break
        except (OSError, ValueError) as exc:
            if time.monotonic() >= deadline:
                raise ManifestUnavailable(
                    f"rank {rank}'s cold-tier manifest is not readable at "
                    f"{path} after {timeout_s:.1f}s ({exc}). Without it this "
                    f"rank cannot locate the experts it delegated to rank "
                    f"{rank}, and guessing a location is not available"
                ) from exc
            time.sleep(0.02)
    if int(payload.get("format", -1)) != COLD_TIER_FORMAT:
        raise ManifestUnavailable(
            f"rank {rank}'s manifest is format {payload.get('format')}, this "
            f"build speaks {COLD_TIER_FORMAT}"
        )
    if str(payload.get("instance_id")) != instance_id:
        raise ManifestUnavailable(
            f"manifest at {path} belongs to instance "
            f"{payload.get('instance_id')!r}, not {instance_id!r} -- a "
            "leftover from a previous launch"
        )
    out = {}
    for entry in payload.get("layouts", []):
        layout = ColdTierLayout.from_json(entry)
        out[(layout.layer_key, layout.param_attr)] = layout
    return out


#: Peer segments mapped by this process, so a 43-layer model maps each peer
#: segment once rather than once per fetch.
_ATTACHED: Dict[str, object] = {}


def attach_peer_segment(layout: ColdTierLayout, register_with_cuda: bool = True):
    """Map a peer's segment read-only and validate it. Returns the mmap.

    ``register_with_cuda`` pins the mapping in THIS process
    (``cudaHostRegister``) so the local DMA engine can read it directly -- that
    is what lets a rank pull a delegated expert over its own PCIe link. It is
    skipped when torch/CUDA is absent, which is the desk case; the mapping is
    still valid, the copy is simply pageable.
    """
    path = segment_path(layout)
    cached = _ATTACHED.get(path)
    if cached is not None:
        return cached
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise SegmentMismatch(
            f"rank {layout.owner_rank}'s cold-tier segment {path} cannot be "
            f"opened ({exc}); its manifest names it but it is not there"
        ) from exc
    try:
        size = os.fstat(fd).st_size
        if size < layout.nbytes:
            raise SegmentMismatch(
                f"{path}: segment is {size} B but the layout needs "
                f"{layout.nbytes} B ({layout.rows} rows x {layout.row_bytes} B "
                f"+ {HEADER_BYTES} B header). A truncated segment cannot be "
                "read past its end and must not be read before it"
            )
        mm = mmap.mmap(fd, layout.nbytes, mmap.MAP_SHARED, mmap.PROT_READ)
    finally:
        os.close(fd)
    try:
        _validate_header(mm, layout, path)
    except Exception:
        mm.close()
        raise
    if register_with_cuda:
        _register_host_memory(mm, path)
    _ATTACHED[path] = mm
    return mm


def _register_host_memory(mm, path: str) -> None:
    """Pin a mapped peer segment for DMA, or leave it pageable and say so."""
    try:
        import ctypes

        import torch

        if not torch.cuda.is_available():
            return
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        cudart = torch.cuda.cudart()
        # cudaHostRegisterReadOnly(0x08) | cudaHostRegisterPortable(0x01)
        err = cudart.cudaHostRegister(addr, len(mm), 0x09)
        if int(err) != 0:
            logger.warning(
                "cold tier: could not pin peer segment %s (cudaHostRegister -> "
                "%s); fetches from it will be pageable and slower, but correct.",
                path,
                err,
            )
    except Exception as exc:  # noqa: BLE001 - pinning is an optimization
        logger.debug("cold tier: host registration skipped for %s (%s)", path, exc)


def detach_all() -> None:
    """Drop every mapped peer segment (tests; and engine teardown)."""
    for mm in list(_ATTACHED.values()):
        try:
            mm.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass
    _ATTACHED.clear()


def peer_row_view(layout: ColdTierLayout, expert_id: int) -> memoryview:
    """Read-only host bytes of one delegated expert. Validated, never guessed."""
    row = layout.row_of(expert_id)
    if not (0 <= row < layout.rows):
        raise SegmentMismatch(
            f"row {row} for expert {expert_id} is outside rank "
            f"{layout.owner_rank}'s {layout.rows}-row cold tier"
        )
    mm = attach_peer_segment(layout)
    start = HEADER_BYTES + row * layout.row_bytes
    return memoryview(mm)[start : start + layout.row_bytes]
