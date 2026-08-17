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
"""#411 — a session checkpoint as a FILE: export here, import there.

#410 built the checkpoint and versioned its manifest "for #411" so this would
be a CONVERTER rather than a silent drop. This is that layer, and it adds
exactly two things: a container, and a gate that runs before any bytes move.

WHY A TAR, not a bespoke container or a single blob:

  * ``tarfile`` is stdlib, so a portability format acquires no dependency --
    which matters most on the machine that has to READ it, possibly one that
    does not run this server at all;
  * page payloads are opaque blobs of varying size, and tar streams them
    without anyone inventing a length-prefix format to get wrong;
  * it is inspectable with tools every machine already has. A format people
    debug on a second rig should not need this repo to open;
  * and the property that shapes the whole design: **the manifest is written
    FIRST**, so ``read_manifest`` reads ONE member and the compatibility gate
    can refuse a bundle without extracting a single payload. Refusing after
    unpacking 40 GiB is a worse refusal than the same one taken up front.

THE GATE IS THE POINT, not the container. Three checks, in this order, each a
NAMED refusal and never a conversion:

  1. **format version** -- unknown versions are refused, never best-effort
     parsed. #410 chose that rule; this inherits it rather than restating it.
  2. **model identity** -- ``compute_model_identity_hash``: weights, dtype,
     quantization and KV byte format. A mismatch is a refusal, not a miss.
  3. **geometry** -- delegated to ``session_checkpoint.verify_geometry``,
     which already owns this rule and already names the offline converter
     (``hicache_migrate --manifest``). Re-deriving it here would be a second
     authority for a correctness rule.

Geometry is BINDING, not advisory. The #410 design says the manifest is
"content only, no placement", but the implementation carries
``GEOMETRY_FIELDS`` and refuses on mismatch -- and the implementation is
right: pages written under a different tp_size / page_size / dcp_owner_mode
are laid out differently and nothing in the read path would notice. Cross-rig
import across geometries therefore runs the umsharder OFFLINE first; this
module refuses and says so, exactly as #545 found the umsharder to be
load-bearing rather than a harness convenience.

KNOWN GAP, carried forward from #726 and NOT closed here: the identity hash
covers the kv-cache dtype STRING, not the layout WITHIN it. Two builds that
both call themselves ``int8`` but differ in group size or scale dtype produce
the same identity and would pass this gate. That is a real hole in the
compatibility key, it predates this module, and it is named in
``IDENTITY_LAYOUT_GAP`` so a reader of a refusal message can see what the gate
does NOT cover.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "IDENTITY_LAYOUT_GAP",
    "PortableSessionError",
    "export_bundle",
    "read_manifest",
    "check_compatibility",
    "import_bundle",
]

#: Bumped when the CONTAINER changes. Independent of the manifest's own
#: version: a reader must be able to reject a container it cannot open before
#: it has any opinion about the manifest inside.
BUNDLE_FORMAT_VERSION = 1

MANIFEST_MEMBER = "manifest.json"
BUNDLE_MEMBER = "bundle.json"
PAGE_PREFIX = "pages/"

#: What the identity hash does NOT cover (#726). Quoted into refusals so the
#: gate never implies more coverage than it has.
IDENTITY_LAYOUT_GAP = (
    "note: the identity hash covers the kv-cache dtype NAME, not the byte "
    "layout within it (group size, scale dtype) -- two builds that both call "
    "themselves the same dtype are indistinguishable to this gate (#726)"
)


class PortableSessionError(RuntimeError):
    """Export or import refused. Always names what mismatched."""


def export_bundle(
    manifest: Dict[str, Any],
    page_bytes_for: Callable[[str], Optional[bytes]],
    path: str,
    *,
    page_hashes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Write ``manifest`` and every page it references into one tar at ``path``.

    ``page_bytes_for`` returns a page's bytes, or None when the page is gone.
    A None is a REFUSAL, not a hole: #410's rule that a branch refuses on an
    evicted page applies just as much at export, and more so -- an export that
    silently omitted a page would produce a bundle that imports cleanly and
    decodes wrong on a machine that cannot tell. Nothing is written when a
    page is missing; the partial file is removed.

    Returns a summary dict (also stored as ``bundle.json``).
    """
    hashes = list(page_hashes if page_hashes is not None else manifest.get("page_hashes") or [])
    payloads: list[Tuple[str, bytes]] = []
    missing: list[str] = []
    for page_hash in hashes:
        blob = page_bytes_for(page_hash)
        if blob is None:
            missing.append(page_hash)
            continue
        payloads.append((page_hash, bytes(blob)))
    if missing:
        raise PortableSessionError(
            f"export refused: {len(missing)} of {len(hashes)} referenced pages "
            f"are not retrievable (first: {missing[0]!r}). An export that "
            f"omitted them would import cleanly and decode wrong. Re-pin the "
            f"checkpoint (#410 slice 2) or re-fetch the pages, then export."
        )

    summary = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "page_count": len(payloads),
        "total_page_bytes": sum(len(b) for _h, b in payloads),
    }

    tmp_path = f"{path}.part"
    try:
        with tarfile.open(tmp_path, "w") as tar:
            # MANIFEST FIRST, and this ordering is load-bearing: a reader must
            # be able to gate on it without streaming past the payloads.
            _add_bytes(tar, MANIFEST_MEMBER, json.dumps(manifest, sort_keys=True).encode())
            _add_bytes(tar, BUNDLE_MEMBER, json.dumps(summary, sort_keys=True).encode())
            for page_hash, blob in payloads:
                _add_bytes(tar, f"{PAGE_PREFIX}{page_hash}", blob)
        os.replace(tmp_path, path)
    except Exception:
        # Same discipline as the canonical store's write: the file becomes
        # visible only at the rename, so a failure leaves no half-bundle.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return summary


def _add_bytes(tar: tarfile.TarFile, name: str, blob: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(blob)
    tar.addfile(info, io.BytesIO(blob))


def read_manifest(path: str) -> Dict[str, Any]:
    """Read ONLY the manifest member. No payload is extracted.

    This is what lets the gate refuse a 40 GiB bundle without unpacking it.
    """
    with tarfile.open(path, "r") as tar:
        try:
            member = tar.extractfile(MANIFEST_MEMBER)
        except KeyError:
            member = None
        if member is None:
            raise PortableSessionError(
                f"not a session bundle: {MANIFEST_MEMBER!r} is missing from {path!r}"
            )
        return json.loads(member.read().decode())


def check_compatibility(
    manifest: Dict[str, Any],
    *,
    local_identity: str,
    local_geometry: Dict[str, Any],
    accepted_envelope_versions: Optional[Iterable[int]] = None,
) -> Tuple[bool, str]:
    """Version, then identity, then geometry. First failure wins, by name.

    Order is deliberate: a bundle from an unknown version may not even have
    the fields the later checks read, so asking about identity first would
    produce a confusing refusal about a field that means something else.
    """
    envelope = manifest.get("checkpoint") or {}
    version = envelope.get("envelope_version")
    accepted = (
        set(accepted_envelope_versions)
        if accepted_envelope_versions is not None
        else _default_accepted_versions()
    )
    if version not in accepted:
        return False, (
            f"unknown checkpoint envelope version {version!r}; this server "
            f"accepts {sorted(accepted)}. Refusing to best-effort parse a "
            f"format it does not know."
        )

    got_identity = manifest.get("model_identity") or manifest.get("model_identity_hash")
    if got_identity != local_identity:
        return False, (
            f"model identity mismatch: bundle {got_identity!r} vs this server "
            f"{local_identity!r}. The hash covers weights, dtype, quantization "
            f"and KV byte format, so this bundle was not written by a server "
            f"this one can replay. {IDENTITY_LAYOUT_GAP}"
        )

    from sglang.srt.managers.session_checkpoint import verify_geometry

    ok, detail = verify_geometry(manifest, local_geometry)
    if not ok:
        return False, detail
    return True, "compatible: " + detail


def _default_accepted_versions() -> set:
    from sglang.srt.managers.session_checkpoint import CHECKPOINT_ENVELOPE_VERSION

    return {int(CHECKPOINT_ENVELOPE_VERSION)}


def import_bundle(
    path: str,
    *,
    local_identity: str,
    local_geometry: Dict[str, Any],
    accepted_envelope_versions: Optional[Iterable[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """Gate FIRST, then read payloads. Returns ``(manifest, pages)``.

    Nothing is extracted until the gate passes -- so a rejected bundle costs
    one member read, and an incompatible bundle can never leave a partially
    seeded session behind, because seeding has not begun.
    """
    manifest = read_manifest(path)
    ok, detail = check_compatibility(
        manifest,
        local_identity=local_identity,
        local_geometry=local_geometry,
        accepted_envelope_versions=accepted_envelope_versions,
    )
    if not ok:
        raise PortableSessionError(f"import refused: {detail}")

    pages: Dict[str, bytes] = {}
    with tarfile.open(path, "r") as tar:
        for member in tar.getmembers():
            if not member.name.startswith(PAGE_PREFIX):
                continue
            handle = tar.extractfile(member)
            if handle is None:  # pragma: no cover - directory entry
                continue
            pages[member.name[len(PAGE_PREFIX) :]] = handle.read()

    expected = list(manifest.get("page_hashes") or [])
    absent = [h for h in expected if h not in pages]
    if absent:
        # Completeness BEFORE seeding, mirroring #410's own rule.
        raise PortableSessionError(
            f"import refused: bundle is missing {len(absent)} of "
            f"{len(expected)} pages the manifest references (first: "
            f"{absent[0]!r}). Seeding a partial chain would produce a session "
            f"that decodes wrong rather than one that fails."
        )
    return manifest, pages
