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

import hashlib
import io
import json
import os
import tarfile
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

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
DIGESTS_MEMBER = "digests.json"
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


def referenced_blobs(manifest: Dict[str, Any]) -> list:
    """Every blob key the manifest references, in a stable order.

    #411 CUT 2 CORRECTION. Cut 1 read ``page_hashes`` -- a field name taken
    from DESIGN_410's prose. The manifest the code actually builds
    (``session_handover.build_manifest``) carries ``kv_keys``, ``mamba_key``
    and ``draft_keys``, and the difference is not cosmetic: exporting a hybrid
    session without its ``mamba_key`` produces a bundle that imports cleanly
    and replays a WRONG session, because a missing recurrent state truncates
    the prefix match at the destination and silently re-prefills. That is the
    #212 failure verbatim, and the source gate exists to make it loud.

    ``page_hashes`` is still honoured when present so a manifest written to
    the design's prose shape is not silently dropped.
    """
    keys: list = []
    keys.extend(manifest.get("kv_keys") or manifest.get("page_hashes") or [])
    mamba_key = manifest.get("mamba_key")
    if mamba_key:
        keys.append(mamba_key)
    keys.extend(manifest.get("draft_keys") or [])
    seen = set()
    ordered = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


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
    hashes = list(page_hashes if page_hashes is not None else referenced_blobs(manifest))
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

    # HARVESTED from the paused #411 WIP (0dc48c92d8), which had per-payload
    # digests this module lacked. A bundle crosses machines: a truncated or
    # corrupted transfer must be caught here, not discovered as a wrong
    # session later. Content-addressed keys do NOT give this for free -- the
    # key names what the bytes SHOULD be, and nothing re-checks that they are.
    digests = {key: hashlib.sha256(blob).hexdigest() for key, blob in payloads}

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
            _add_bytes(tar, DIGESTS_MEMBER, json.dumps(digests, sort_keys=True).encode())
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
    exists_fn: Optional[Callable[[str], bool]] = None,
) -> Tuple[bool, str]:
    """Delegate to ``session_checkpoint.verify_restore``. Nothing is re-decided.

    #411 CUT 2, SECOND CORRECTION. Cut 1 hand-rolled version and identity
    checks. The first pass of Cut 2 replaced them with a hand-rolled
    COMPOSITION of ``verify_import`` + ``verify_geometry`` -- which is itself
    already a function: ``verify_restore``, in the same order, with the same
    reasoning ("Identity failing is the more fundamental problem, so it is the
    message the caller sees"). DESIGN_410 section 7 specifies exactly that
    composition. Re-composing it here was a second authority one level up from
    the one I had just avoided.

    So this now calls ``verify_restore`` and adds exactly one thing of its
    own: the #726 note on an identity refusal, because the identity hash
    covers the kv-cache dtype NAME and not the byte layout within it, and a
    reader of a refusal should learn where the gate stops.

    ``exists_fn`` answers "is this blob in the bundle". Omitted means the
    presence half is skipped, which is what a caller inspecting a manifest
    without its payloads wants.
    """
    from sglang.srt.managers.session_checkpoint import verify_restore

    probe = exists_fn if exists_fn is not None else (lambda _key: True)
    try:
        ok, detail = verify_restore(manifest, probe, local_identity, local_geometry)
    except KeyError as e:
        return False, (
            f"malformed manifest: required field {e} is absent, so this "
            f"bundle cannot be judged rather than judged leniently"
        )
    if not ok and "model identity mismatch" in detail:
        return False, f"{detail}. {IDENTITY_LAYOUT_GAP}"
    return ok, detail


def import_bundle(
    path: str,
    *,
    local_identity: str,
    local_geometry: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """Gate FIRST, on NAMES, then read payloads. Returns ``(manifest, pages)``.

    #411 CUT 2: the completeness check runs against the bundle's member NAMES,
    which the tar directory yields without extracting a byte. So an incomplete
    or incompatible bundle is refused before ANY payload is read -- and, since
    nothing is returned, before any caller could have begun seeding from it.

    THE FAILURE DIRECTION THIS MAKES IMPOSSIBLE is partial-seed-then-fail. A
    gate that ran after extraction could still be recovered from; one that ran
    after seeding could not, because a half-seeded session decodes wrong
    rather than failing. The completeness rule is #410's own
    (``validate_manifest_completeness`` on the write side, ``verify_import``
    on the read side); this only extends the boundary it applies at.
    """
    manifest = read_manifest(path)

    with tarfile.open(path, "r") as tar:
        present = {
            name[len(PAGE_PREFIX) :]
            for name in tar.getnames()
            if name.startswith(PAGE_PREFIX)
        }

        ok, detail = check_compatibility(
            manifest,
            local_identity=local_identity,
            local_geometry=local_geometry,
            exists_fn=lambda key: key in present,
        )
        if not ok:
            raise PortableSessionError(f"import refused: {detail}")

        pages: Dict[str, bytes] = {}
        for member in tar.getmembers():
            if not member.name.startswith(PAGE_PREFIX):
                continue
            handle = tar.extractfile(member)
            if handle is None:  # pragma: no cover - directory entry
                continue
            pages[member.name[len(PAGE_PREFIX) :]] = handle.read()

        digests = {}
        try:
            handle = tar.extractfile(DIGESTS_MEMBER)
            if handle is not None:
                digests = json.loads(handle.read().decode())
        except KeyError:
            digests = {}

    # Integrity AFTER extraction and BEFORE returning: a corrupted payload
    # must not reach a caller that would seed from it. Absent digests are
    # tolerated (a bundle from an older writer), and that tolerance is
    # deliberate rather than silent -- an older bundle is readable, it simply
    # carries no integrity claim to check.
    corrupt = [
        key
        for key, want in digests.items()
        if key in pages and hashlib.sha256(pages[key]).hexdigest() != want
    ]
    if corrupt:
        raise PortableSessionError(
            f"import refused: {len(corrupt)} payload(s) failed their digest "
            f"(first: {corrupt[0]!r}). The bundle was truncated or corrupted "
            f"in transfer; seeding from it would produce a wrong session, not "
            f"a failed one."
        )
    return manifest, pages
