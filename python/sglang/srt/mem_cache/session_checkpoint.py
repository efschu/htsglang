# Copyright 2023-2024 SGLang Team
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
"""#410 slice 2, wiring: the checkpoint lifecycle that actually takes the pins.

Slice 1 built the manifest (a pure value layer: references, verification, seed
plan). Slice 2 built the pin ledger (ref-counted, budgeted, durable, honoured
by the evictor). Between them sat NOTHING: ``pin_checkpoint`` had no caller, so
no checkpoint ever pinned anything and slice 1's "refused, a reference was
evicted" governed every checkpoint in practice.

That gap is the #698 lesson from the other side. #698 was a trigger that never
fired; this was a protection never taken. Both look finished in the diff and do
nothing at runtime, which is why the tests here assert through the REAL store
and the REAL evictor rather than through the ledger's own bookkeeping.

Two orderings carry the whole design, and both are chosen against the failure
that follows from reversing them:

* CREATE pins BEFORE it writes the manifest. Between "the references exist" and
  "the checkpoint is durable" the pages are ordinary LRU entries; anything that
  evicts in that window makes the manifest stillborn. Pinning first closes the
  window, and it is the order the ledger's orphan reaper was already built for
  -- it age-gates precisely because "pins exist, manifest does not yet" is a
  legitimate transient state, not a leak.
* DELETE removes the manifest BEFORE it unpins. The reverse strips protection
  from a checkpoint that still exists and is still branchable, so a crash in
  the window leaves a live checkpoint whose pages can evaporate. In this order
  a crash leaves an orphan pin instead, which the age-gated reaper collects.
  One direction fails into a reapable leak; the other into a broken promise.

ONE AUTHORITY, checked rather than assumed. Nothing here re-implements what
exists: ``session_manifest`` stays the pure value layer (it gains no store
coupling), ``PinLedger`` stays the sole owner of pin accounting and the budget
refusal, and ``LRUFileEvictor`` stays the sole place eviction skips a pin. This
module owns exactly one thing nobody owned: the ORDER in which those are called
and the store-resident record that says a checkpoint exists.

It is also not ``session_bundle``. That is a portable single-file export
(magic header, embedded payloads, digests) for moving a checkpoint BETWEEN
machines; the pages travel inside it. This is the in-store lifecycle, where the
pages stay in the store and a pin is what keeps them there. Different problems:
a bundle needs no pin because it carries its own bytes.

Unpinned checkpoints stay a first-class option (``pin=False``). Slice 1's
refusal is not superseded by pinning, it is the correct behaviour for a
checkpoint nobody paid to keep -- and a caller that cannot afford the budget
should get a cheap best-effort checkpoint, not a hard failure at create time.
Both behaviours are pinned by tests; neither silently replaces the other.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.mem_cache.session_manifest import (
    ManifestError,
    ManifestIncomplete,
    SessionManifest,
    dumps,
    loads,
    seed_plan,
    verify_against_store,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sglang.srt.mem_cache.session_manifest import SeedStep

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#410 checkpoint]"

# Manifests live beside the pins (``pins706/``) rather than among the pages:
# both are checkpoint metadata with a lifetime the page LRU must never touch,
# and keeping them out of the page tree keeps the evictor's walk over .bin
# files exactly as it was.
CHECKPOINT_DIRNAME = "checkpoints410"

MANIFEST_SUFFIX = ".manifest.json"


class CheckpointNotFound(ManifestError):
    """No checkpoint by that id in this store."""


class CheckpointExists(ManifestError):
    """Refusing to overwrite an existing checkpoint id.

    Creation is not idempotent on purpose: silently replacing a checkpoint
    would unpin the old references as a side effect of a call that reads like
    a create, and the caller would learn about it only when a branch refused.
    """


@dataclasses.dataclass(frozen=True)
class CheckpointRecord:
    """What a create actually did -- including that it pinned nothing."""

    checkpoint_id: str
    manifest: SessionManifest
    pinned: bool
    bytes_added: int = 0
    bytes_shared: int = 0


def checkpoint_dir(store: Any) -> str:
    return os.path.join(str(store.file_path), CHECKPOINT_DIRNAME)


def _manifest_path(store: Any, checkpoint_id: str) -> str:
    return os.path.join(checkpoint_dir(store), f"{checkpoint_id}{MANIFEST_SUFFIX}")


def _write_atomic(path: str, blob: str) -> None:
    """Write via a temp file in the same directory, then replace.

    Same discipline as the pin ledger: a reader must never observe a manifest
    that is half-written, because a truncated manifest deserialises into a
    SHORTER reference list -- a checkpoint that looks whole and restores a
    prefix of itself.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def create_checkpoint(
    store: Any,
    checkpoint_id: str,
    manifest: SessionManifest,
    *,
    pin: bool = True,
) -> CheckpointRecord:
    """Persist a checkpoint, pinning what it references.

    Order is load-bearing (see the module docstring): verify, then PIN, then
    write the manifest, and unpin if the write fails so a refused create leaves
    no pinned bytes behind.

    Raises ``ManifestIncomplete`` when the store does not hold every reference
    -- pinning a key that is not there protects nothing while still looking
    like success, so a checkpoint of an already-incomplete prefix is refused at
    creation rather than at the branch that needed it.

    Raises ``PinBudgetExceeded`` (unchanged, with its four numbers) when the
    pins would cross the budget. Nothing is created and nothing stays pinned:
    the ledger refuses before it installs.
    """
    if not checkpoint_id:
        raise ManifestError("checkpoint_id must be a non-empty string")
    path = _manifest_path(store, checkpoint_id)
    if os.path.exists(path):
        raise CheckpointExists(
            f"{LOG_PREFIX} checkpoint {checkpoint_id!r} already exists in this "
            "store. Delete it first: overwriting would drop the old "
            "checkpoint's pins as a side effect of a create."
        )

    missing = verify_against_store(manifest, store.exists)
    if missing:
        raise ManifestIncomplete(missing)

    pinned = False
    added = shared = 0
    if pin:
        # Budget refusal propagates verbatim -- it already names the checkpoint
        # and every number (want, held, budget, overshoot), and this layer has
        # nothing truer to add.
        result = store.pin_checkpoint(checkpoint_id, list(manifest.references()))
        pinned = True
        added, shared = int(result.bytes_added), int(result.bytes_shared)

    try:
        _write_atomic(path, dumps(manifest))
    except BaseException:
        # The manifest is what makes the pins meaningful. Without it they are
        # orphans the reaper would eventually collect -- but only after the age
        # gate, so release them now rather than leaving the store protecting
        # something that will never exist.
        if pinned:
            try:
                store.unpin_checkpoint(checkpoint_id)
            except Exception:  # pragma: no cover - best-effort rollback
                logger.exception(
                    "%s could not release pins for the failed create of %s; "
                    "the orphan reaper will collect them.",
                    LOG_PREFIX,
                    checkpoint_id,
                )
        raise

    logger.info(
        "%s created %s: %d reference(s), pinned=%s (+%d B new, %d B shared).",
        LOG_PREFIX,
        checkpoint_id,
        len(manifest.references()),
        pinned,
        added,
        shared,
    )
    return CheckpointRecord(
        checkpoint_id=checkpoint_id,
        manifest=manifest,
        pinned=pinned,
        bytes_added=added,
        bytes_shared=shared,
    )


def load_checkpoint(store: Any, checkpoint_id: str) -> SessionManifest:
    path = _manifest_path(store, checkpoint_id)
    try:
        with open(path) as handle:
            blob = handle.read()
    except FileNotFoundError as e:
        raise CheckpointNotFound(
            f"{LOG_PREFIX} no checkpoint {checkpoint_id!r} in this store."
        ) from e
    return loads(blob)


def delete_checkpoint(store: Any, checkpoint_id: str) -> int:
    """Remove a checkpoint and release its pins. Returns bytes actually freed.

    Manifest first, pins second (module docstring). Shared references stay
    pinned for their other holders: the ledger ref-counts, so deleting a branch
    never strips the parent it forked from.
    """
    path = _manifest_path(store, checkpoint_id)
    existed = True
    try:
        os.unlink(path)
    except FileNotFoundError:
        existed = False

    freed = int(store.unpin_checkpoint(checkpoint_id))
    if not existed and freed == 0:
        raise CheckpointNotFound(
            f"{LOG_PREFIX} no checkpoint {checkpoint_id!r} in this store."
        )
    logger.info(
        "%s deleted %s: %d B released (shared references stay pinned).",
        LOG_PREFIX,
        checkpoint_id,
        freed,
    )
    return freed


def list_checkpoints(store: Any) -> tuple[str, ...]:
    directory = checkpoint_dir(store)
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return ()
    return tuple(
        sorted(n[: -len(MANIFEST_SUFFIX)] for n in names if n.endswith(MANIFEST_SUFFIX))
    )


def branch_plan(
    store: Any,
    checkpoint_id: str,
    *,
    model_identity: Optional[str] = None,
) -> tuple["SeedStep", ...]:
    """The ordered seed plan for branching from a stored checkpoint.

    Verification still runs against the live store even for a pinned
    checkpoint. A pin makes eviction skip the page; it does not make the page
    indestructible (an operator can delete the directory, a disk can lose it),
    and a protection that is TRUSTED instead of checked is how a partial prefix
    reaches a live session.
    """
    manifest = load_checkpoint(store, checkpoint_id)
    return seed_plan(manifest, exists=store.exists, model_identity=model_identity)
