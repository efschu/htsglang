"""#411: a session as a FILE -- export here, import on another server or rig.

STATUS ON THIS BRANCH (recorded 2026-08-17 by the #121 determination). This
module has NO importer anywhere in ``python/`` or ``test/`` -- not a production
caller, not even a test. It is the paused WIP that ``#411`` superseded with
``managers/session_portable.py``, and that successor is NOT merged here; it
lives on ``feat/411-portable-sessions``. So on this tree ``export_bundle`` /
``import_bundle`` are reachable from nothing.

This paragraph exists because the docstring above reads like a shipped
feature, and a reader who finds ``export_bundle``/``import_bundle`` while
looking for portable sessions would reasonably either wire this file or build
a second one beside it. Both are wrong: the live migration path on this branch
is ``POST /session_handover`` (#261, ``managers/session_handover.py``) plus the
external ``hicache_migrate`` CLI, and the portable-bundle successor arrives
with #411. Do not wire this module; take #411's, or take the handover path.

The design below is still the design -- it is the reasoning that was ported
forward, not stale thinking. Only its shipped-ness was overstated.

#410's manifest is already the format core: content-only references, no
geometry, versioned, deterministic. This module is the container around it --
the manifest plus the bytes it points at, so the receiving rig needs nothing
from the sending one.

WHY THE BUNDLE CARRIES THE CANONICAL WHOLE FORM. The store's read path cuts a
page to whatever geometry asks for it, and that is exactly what an export must
NOT do: a slice is a serving convenience, while a bundle is the thing another
rig rebuilds from. So export copies the stored object VERBATIM -- the full
16-slot page, the full-layer full-head GDN blob -- and import writes it back
through the same protocol. The receiving rig then cuts it at read time for its
own geometry, whatever that is. That is the whole dividend of #706 collected in
one place: the sender does not need to know the receiver's parallelism, and the
receiver does not need to know the sender's.

WHAT THE COMPAT GATE CHECKS, and the discipline behind the list (#241). Only
what genuinely binds the BYTES:

* ``model_identity`` -- weights revision, dtype, quantization, kv-cache-dtype,
  all folded into the one hash the store already keys by. Different identity
  means the pages are a different byte format.
* the canonical page spec -- attention-layer count and bytes per layer per
  token. Spec equality IS the layout contract (#706 slice 1); a page that
  disagrees means something else by the same bytes.
* ``page_size`` -- a canonical page is ONE token.
* the canonical GDN blob layout -- layer count, head count, head dim, state
  size, conv dim/width and the two itemsizes.

And what it deliberately does NOT check, because checking it would betray the
point of the format: tp_rank, tp_size, pp_rank, pp_size, the token vector, the
layer cut, ``gdn_tp_units`` or any ratio. Those decide how bytes are CUT, never
what they mean. A bundle from a TP=3 rig imports onto a PP=3 rig, and onto a
single-GPU rig, unchanged -- that is the feature.

A mismatch is a NAMED refusal listing the fields, never a silent conversion. A
converter is a deliberate, auditable act (the same rule ``hicache_migrate``
states for geometry handover); guessing at import time would produce a session
that looks restored and answers wrong.

TRUNCATION LEAVES NO RESIDUE. Import verifies the WHOLE bundle -- every entry's
length and digest -- before writing a single byte. That costs a second pass
over the file and buys the property that matters: a bundle that turns out to be
short refuses having changed nothing. A one-pass streaming import would have
written half a session before discovering the problem, and half a session is
exactly the silent-partial class this lane keeps closing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import struct
from typing import Callable, Optional, Sequence

from sglang.srt.mem_cache.session_manifest import (
    SessionManifest,
    dumps as manifest_dumps,
    loads as manifest_loads,
)

BUNDLE_VERSION = 1
MAGIC = b"SGLSESS\x00"
_U32 = struct.Struct("<I")
_HEADER_MAX = 64 * 1024 * 1024  # a header larger than this is not a header


class BundleError(ValueError):
    """A bundle that cannot be trusted."""


class BundleTruncated(BundleError):
    """The file ends before the bundle does."""


class BundleIntegrityError(BundleError):
    """An entry's bytes do not match the digest recorded for them."""


class BundleIncompatible(BundleError):
    """The receiving server cannot mean the same thing by these bytes."""

    def __init__(self, mismatches: Sequence[str]) -> None:
        self.mismatches = tuple(mismatches)
        super().__init__(
            "this bundle was written under "
            + "; ".join(self.mismatches)
            + ". These fields bind the MEANING of the stored bytes, so the "
            "session is refused rather than converted: a converter is a "
            "deliberate, auditable act, and guessing here would produce a "
            "session that looks restored and answers wrong. Parallel geometry "
            "(tp/pp ranks and sizes, token vectors, layer cuts, gdn units) is "
            "NOT checked and never will be -- geometry freedom is what the "
            "canonical format is for."
        )


@dataclasses.dataclass(frozen=True)
class CompatKey:
    """Everything that binds the meaning of a canonical payload. No geometry."""

    model_identity: str
    num_attn_layers: int
    kv_cell_bytes: int
    page_size: int = 1
    mamba: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "model_identity": self.model_identity,
            "num_attn_layers": int(self.num_attn_layers),
            "kv_cell_bytes": int(self.kv_cell_bytes),
            "page_size": int(self.page_size),
            "mamba": dict(self.mamba) if self.mamba else None,
        }

    @staticmethod
    def from_dict(raw: dict) -> "CompatKey":
        return CompatKey(
            model_identity=raw["model_identity"],
            num_attn_layers=int(raw["num_attn_layers"]),
            kv_cell_bytes=int(raw["kv_cell_bytes"]),
            page_size=int(raw.get("page_size", 1)),
            mamba=raw.get("mamba"),
        )

    def mismatches(self, other: "CompatKey") -> tuple[str, ...]:
        """Field-by-field differences, phrased for a human reading a refusal."""
        out = []
        if self.model_identity != other.model_identity:
            out.append(
                f"model identity {self.model_identity!r} (this server: "
                f"{other.model_identity!r}) -- weights, dtype, quantization or "
                "kv-cache-dtype differ"
            )
        if int(self.num_attn_layers) != int(other.num_attn_layers):
            out.append(
                f"{self.num_attn_layers} attention layers per page (this "
                f"server: {other.num_attn_layers})"
            )
        if int(self.kv_cell_bytes) != int(other.kv_cell_bytes):
            out.append(
                f"{self.kv_cell_bytes} KV bytes per layer per token (this "
                f"server: {other.kv_cell_bytes})"
            )
        if int(self.page_size) != int(other.page_size):
            out.append(
                f"page_size {self.page_size} (this server: {other.page_size})"
            )
        if (self.mamba or None) != (other.mamba or None):
            out.append(
                f"a different canonical GDN blob layout ({self.mamba} vs this "
                f"server's {other.mamba})"
            )
        return tuple(out)


@dataclasses.dataclass(frozen=True)
class BundleEntry:
    key: str
    kind: str  # "kv" | "gdn"
    nbytes: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class BundleStats:
    entries: int
    payload_bytes: int
    path: str


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def export_bundle(
    path: str,
    manifest: SessionManifest,
    compat: CompatKey,
    read_payload: Callable[[str], Optional[bytes]],
) -> BundleStats:
    """Write a session bundle. Deterministic: same inputs, same bytes.

    ``read_payload`` returns the STORED OBJECT verbatim for a content key --
    the whole canonical page or blob, never a geometry slice. Returning None
    means the store no longer holds it, which fails the export rather than
    producing a bundle that cannot be imported.
    """
    entries: list[BundleEntry] = []
    payloads: list[bytes] = []
    for key in manifest.references():
        kind = "gdn" if manifest.gdn is not None and key == manifest.gdn.blob_key else "kv"
        payload = read_payload(key)
        if payload is None:
            raise BundleError(
                f"cannot export {key!r}: the store no longer holds it. A bundle "
                "missing a reference is a session that cannot be imported, so "
                "the export fails here rather than at the far end. Pin the "
                "checkpoint (#410 slice 2) to keep its references alive."
            )
        entries.append(
            BundleEntry(
                key=key, kind=kind, nbytes=len(payload), sha256=_digest(payload)
            )
        )
        payloads.append(payload)

    header = {
        "bundle_version": BUNDLE_VERSION,
        "compat": compat.as_dict(),
        "manifest": json.loads(manifest_dumps(manifest)),
        "entries": [dataclasses.asdict(e) for e in entries],
    }
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()

    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(_U32.pack(len(blob)))
        f.write(blob)
        for payload in payloads:
            f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return BundleStats(
        entries=len(entries), payload_bytes=sum(e.nbytes for e in entries), path=path
    )


def read_header(path: str) -> dict:
    """Parse and version-check the header. Raises on anything unusable."""
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise BundleError(
                f"{path!r} is not a session bundle (bad magic {magic!r})."
            )
        raw_len = f.read(_U32.size)
        if len(raw_len) != _U32.size:
            raise BundleTruncated("bundle ends inside its header length")
        (length,) = _U32.unpack(raw_len)
        if length > _HEADER_MAX:
            raise BundleError(f"bundle header claims {length} bytes; refusing")
        blob = f.read(length)
        if len(blob) != length:
            raise BundleTruncated("bundle ends inside its header")
    try:
        header = json.loads(blob)
    except json.JSONDecodeError as e:
        raise BundleError(f"bundle header is not valid JSON: {e}") from e
    version = header.get("bundle_version")
    if version != BUNDLE_VERSION:
        raise BundleError(
            f"bundle_version {version!r} is not {BUNDLE_VERSION}. Refused, not "
            "best-effort parsed: a bundle is replayed into a live session, and "
            "a tolerant parser is how a field silently stops being honoured. "
            "Upgrades go through an explicit #411 converter that reads the old "
            "version and writes this one."
        )
    return header


def verify_bundle(path: str) -> tuple[BundleEntry, ...]:
    """Check every entry's length and digest. Reads, writes nothing.

    This is the whole reason import is two-pass: a bundle that turns out to be
    short or corrupt must be refused having changed nothing on the receiving
    store.
    """
    header = read_header(path)
    entries = tuple(
        BundleEntry(
            key=e["key"],
            kind=e["kind"],
            nbytes=int(e["nbytes"]),
            sha256=e["sha256"],
        )
        for e in header.get("entries", [])
    )
    with open(path, "rb") as f:
        f.seek(len(MAGIC) + _U32.size)
        (length,) = _U32.unpack_from(
            open(path, "rb").read(len(MAGIC) + _U32.size)[len(MAGIC) :]
        )
        f.seek(len(MAGIC) + _U32.size + length)
        for entry in entries:
            payload = f.read(entry.nbytes)
            if len(payload) != entry.nbytes:
                raise BundleTruncated(
                    f"bundle ends inside entry {entry.key!r}: expected "
                    f"{entry.nbytes} bytes, found {len(payload)}. Nothing has "
                    "been written to this store."
                )
            if _digest(payload) != entry.sha256:
                raise BundleIntegrityError(
                    f"entry {entry.key!r} does not match its recorded digest. "
                    "The bundle is corrupt; importing it would put bytes into "
                    "a content-addressed store under a key that does not "
                    "describe them."
                )
        trailing = f.read(1)
    if trailing:
        raise BundleError(
            "bundle has trailing bytes after its last entry; refusing rather "
            "than guessing what they were meant to be."
        )
    return entries


def iter_payloads(path: str):
    """(entry, payload) in bundle order. Call only after ``verify_bundle``."""
    header = read_header(path)
    with open(path, "rb") as f:
        f.seek(len(MAGIC))
        (length,) = _U32.unpack(f.read(_U32.size))
        f.seek(len(MAGIC) + _U32.size + length)
        for raw in header.get("entries", []):
            entry = BundleEntry(
                key=raw["key"],
                kind=raw["kind"],
                nbytes=int(raw["nbytes"]),
                sha256=raw["sha256"],
            )
            yield entry, f.read(entry.nbytes)


@dataclasses.dataclass(frozen=True)
class ImportResult:
    manifest: SessionManifest
    written: tuple[str, ...]
    skipped: tuple[str, ...]  # already present, content-addressed


def import_bundle(
    path: str,
    local_compat: CompatKey,
    write_payload: Callable[[str, bytes], bool],
    *,
    exists: Optional[Callable[[str], bool]] = None,
    register: Optional[Callable[[SessionManifest, tuple[str, ...]], None]] = None,
) -> ImportResult:
    """Verify a bundle completely, then write it, then register the session.

    Ordering is the contract: verify-all, write-all, register. A manifest that
    became visible before its payloads landed would be a checkpoint that
    branches into a partial prefix -- the exact failure #410 slice 1 refuses at
    the other end, and cheaper to prevent here.
    """
    header = read_header(path)
    bundle_compat = CompatKey.from_dict(header["compat"])
    mismatches = bundle_compat.mismatches(local_compat)
    if mismatches:
        raise BundleIncompatible(mismatches)

    verify_bundle(path)  # nothing is written before this returns

    manifest = manifest_loads(json.dumps(header["manifest"]))
    written: list[str] = []
    skipped: list[str] = []
    for entry, payload in iter_payloads(path):
        if exists is not None and exists(entry.key):
            # Content-addressed: the same key already names these bytes.
            skipped.append(entry.key)
            continue
        write_payload(entry.key, payload)
        written.append(entry.key)

    if register is not None:
        register(manifest, tuple(written) + tuple(skipped))
    return ImportResult(
        manifest=manifest, written=tuple(written), skipped=tuple(skipped)
    )
