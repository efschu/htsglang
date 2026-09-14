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
"""In-process hook that records one rank's phase footprint.

Armed only by ``SGLANG_PHASE_FOOTPRINT_DUMP``; absent that variable every
function here returns immediately and allocates nothing, so an unarmed boot is
byte-identical to one without this module.

The counters are ``torch.cuda.memory_stats()``, not ``nvidia-smi``, and the
difference is the entire point (see ``scripts/vram_ledger/probe_activation.py``):
``allocated_bytes.all.peak`` tracks LIVE allocations, so it sees a prefill
transient that fits inside a segment the caching allocator already holds.
``nvidia-smi`` does not, which is why the 2026-08-05 window could only bound the
activation peak instead of measuring it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DUMP_ENV",
    "is_armed",
    "reset_peaks",
    "read_peaks",
    "dump_filename",
    "boot_token",
    "write_footprint_dump",
    "note_capture_begin",
    "note_capture_end",
    "record_prefill_peak",
]

DUMP_ENV = "SGLANG_PHASE_FOOTPRINT_DUMP"


def is_armed() -> bool:
    return bool(os.environ.get(DUMP_ENV))


def reset_peaks(device_index: int = 0) -> None:
    """Re-base the peak counters, and record the floor they were re-based to.

    ``torch.cuda.reset_peak_memory_stats`` does not zero the peak: it re-bases
    it at whatever is allocated *right now*, which after model load and KV
    sizing is weights + KV pool + captured graphs. The counter read back later
    is therefore an ABSOLUTE resident figure, not the activation cost. That is
    what the window-5 dumps recorded -- 26555 / 17306 / 16368 MiB, the whole
    footprint of each rank rather than its prefill transient (#589).

    So every re-base records its floor here, and the activation number the
    ledger ingests is peak MINUS that floor. The raw peak is still dumped
    alongside it, labelled, because it is the figure the operator sees in
    ``nvidia-smi`` and dropping it would make the two impossible to reconcile.
    """
    global _peak_floor_bytes
    if not is_armed():
        return
    try:
        import torch

        torch.cuda.reset_peak_memory_stats(device_index)
    except Exception as e:  # pragma: no cover - torch shape differences
        logger.debug("could not reset peak memory stats: %s", e)
        return
    _peak_floor_bytes = int(read_peaks(device_index).get("allocated_bytes", 0))


def read_peaks(device_index: int = 0) -> dict:
    """``{allocated_peak_bytes, reserved_peak_bytes, allocated_bytes}``."""
    try:
        import torch

        stats = torch.cuda.memory_stats(device_index)
        return {
            "allocated_peak_bytes": int(stats.get("allocated_bytes.all.peak", 0)),
            "reserved_peak_bytes": int(stats.get("reserved_bytes.all.peak", 0)),
            "allocated_bytes": int(stats.get("allocated_bytes.all.current", 0)),
        }
    except Exception as e:  # pragma: no cover - torch shape differences
        logger.debug("could not read memory stats: %s", e)
        return {}


def dump_filename(rank: int, group: str = "") -> str:
    """The dump filename for this rank, qualified by Weg-2 group if any.

    FIX #1292: P and D are two independently-launched process groups that
    can share one dump directory (the existing recipe arms both with the
    identical ``SGLANG_PHASE_FOOTPRINT_DUMP``). ``rank`` alone is only
    unique WITHIN one group's ``torch.distributed`` job -- see
    :func:`_global_rank` -- so P's rank 0 and D's rank 0 collided on the
    same filename and P, booted second, silently overwrote D's dump.

    The group tag is not a new identity: it is ``SGLANG_WEG2_GROUP``, the
    one thing in the tree that already tells a rank which Weg-2 group it is
    in (``weg2/launcher.py``'s ``build_env(group=...)``, read back via
    ``weg2_memory_saver.weg2_group_name()``). Outside Weg-2 that group is
    ``""`` and the filename is byte-identical to the pre-fix shape, so a
    single-regime boot is unaffected.

    NAMES ONLY GROUP+RANK, DELIBERATELY -- NOT THE BOOT. #1292 fixed the P-vs-D
    collision within one boot; it never gave the boot itself an identity, so
    two boots of the SAME form (P and D unchanged, same rank, same group)
    still collide on this exact filename and the later boot silently destroys
    the earlier one's dump -- boot weg2xsn31/8's own dump, proven written by
    its boot log (03:21:36Z, PP0, legs=0) and gone from disk by the time
    #1389 needed it. See :func:`boot_token` / :func:`_boot_subdir`: the boot
    goes into the DIRECTORY, this filename stays exactly as it was so every
    existing reader of ONE boot's own dumps keeps working unchanged.
    """
    tag = f"{group}_" if group else ""
    return f"phase_footprint_{tag}rank{rank}.json"


#: Captured once, at import -- i.e. once per REAL PROCESS, which for Weg-2 is
#: once per real boot attempt of this rank (boots never run concurrently on
#: this rig, per the standing arbitration rule, so two attempts of the
#: identical ``--tag`` are two SEPARATE, SEQUENTIAL processes with two
#: different import times). This needs no cross-rank agreement: the
#: collision this fixes is "same rank+group, different BOOT", never "two
#: ranks of the same boot disagreeing" -- #1292's own group tag already
#: settled that half.
#:
#: NOT PER-PROCESS, DELIBERATELY: an earlier draft of this function derived
#: its own token from THIS process's own start time. That is wrong -- P and D
#: are separate OS processes with different PIDs and different start times
#: for the SAME real boot, so two independently-derived per-process tokens
#: would scatter one boot's own dumps across two boot-subdirectories, which
#: breaks the #1292 "P and D share one dump directory, ingest reads both"
#: reading this exact module's ``ingest`` groups by profile digest for. The
#: token has to be computed ONCE, by the ONE process that spawns both groups,
#: and handed down -- see :data:`BOOT_TOKEN_ENV`.
_PROCESS_START_EPOCH: int = int(time.time())
_boot_token_cache: Optional[str] = None

#: #1395: published UNCONDITIONALLY by ``weg2/launcher.py build_env``
#: (``weg2_boot_token(ns)``, memoised on the launcher's own namespace so
#: BOTH groups' ranks inherit the identical value). Safe to always publish,
#: unlike ``SGLANG_WEG2_LANE_COVERAGE_TOKEN``: this value decides nothing
#: and arms nothing by existing, it only names which boot a rank belongs to.
BOOT_TOKEN_ENV = "SGLANG_WEG2_BOOT_TOKEN"


def _boot_tag_from_env() -> str:
    """The human-readable ``--tag`` this rank's boot ran under, or ``""``.

    Used only by the FALLBACK path in :func:`boot_token` (no
    :data:`BOOT_TOKEN_ENV` -- outside Weg-2, or a tree older than #1395).
    ``SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR`` is published to every Weg-2
    rank UNCONDITIONALLY (``weg2/launcher.py build_env``, "R19 shape...
    written unconditionally"), and its value is
    ``f"{STORE_ROOT}/{tag}"`` (``weg2/launcher.py plan_store``) -- the
    basename IS the tag. Empty outside Weg-2, where this probe's own module
    docstring says the group is also empty.
    """
    store_dir = os.environ.get("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", "")
    return os.path.basename(store_dir.rstrip("/")) if store_dir else ""


def boot_token() -> str:
    """``"<tag>:<epoch>:<pid>"`` -- THE #1348 SHAPE, reused rather than a
    second one invented for this module.

    Memoised per PROCESS (module-level cache): every call within one rank's
    lifetime -- ``note_capture_begin``, ``note_capture_end``,
    ``record_prefill_peak``'s (potentially many) rewrites -- must return the
    SAME token, or the rank's own dumps would scatter across several
    boot-subdirectories of their own single boot.

    READS :data:`BOOT_TOKEN_ENV` FIRST -- the launcher's own, ONE token,
    computed once and shared by every rank of BOTH groups (see the module
    note above :data:`_PROCESS_START_EPOCH` for why a per-process fallback
    alone would be wrong). Only when that variable is absent (outside
    Weg-2, or a launcher older than #1395) does this derive its own
    per-process token from :func:`_boot_tag_from_env` -- correct there
    because there is no sibling process to disagree with.
    """
    global _boot_token_cache
    if _boot_token_cache is None:
        _boot_token_cache = os.environ.get(BOOT_TOKEN_ENV, "") or (
            f"{_boot_tag_from_env() or 'notag'}:{_PROCESS_START_EPOCH}:"
            f"{os.getpid()}"
        )
    return _boot_token_cache


def _boot_subdir(token: str) -> str:
    """A filesystem-safe, per-boot directory name from :func:`boot_token`.

    IDENTICAL SHAPE to ``weg2.lane_coverage._boot_subdir`` (#1395) --
    extended here rather than reinvented: ``:`` becomes ``_`` (unsafe on some
    filesystems, reads as a path separator to some tools), and an empty
    token -- a caller that could not resolve one at all -- gets its own
    named fallback rather than collapsing to the un-namespaced directory,
    which would reintroduce exactly the collision this function removes for
    the one caller that never got a token.
    """
    if not token:
        return "no-boot-token"
    return token.replace(":", "_").replace("/", "_").replace(os.sep, "_")


def _read_boot_token(path: str) -> Optional[str]:
    """The ``boot_token`` field of an existing dump at ``path``, or ``None``
    when it cannot be read -- an unreadable existing file is reported as
    such, never silently treated as "no collision"."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("boot_token")
    except (OSError, ValueError):
        return None


BOOT_COLLISION_CODE = "W109 Weg2PhaseFootprintBootCollision"


def write_footprint_dump(
    *,
    rank: int,
    card_uuid: str,
    hw_fingerprint: str,
    profile_canonical: list,
    activation_peak_bytes: int,
    capture_bytes: int,
    reserved_peak_bytes: int = 0,
    prefill_tokens: Optional[int] = None,
    dump_dir: Optional[str] = None,
    peak_floor_bytes: Optional[int] = None,
    group: str = "",
    boot_token_override: str = "",
) -> Optional[str]:
    """Write this rank's dump. One file per rank, so no collective is needed
    and a rank that dies mid-run simply contributes nothing.

    Three activation numbers are written, not one, because they answer
    different questions and window 5 proved they get confused otherwise:

    ``activation_peak_bytes``   the raw counter -- ABSOLUTE, weights and KV
                                included, because the reset re-bases rather
                                than zeroes (see :func:`reset_peaks`).
    ``peak_floor_bytes``        what was resident when the bracket opened.
    ``activation_delta_bytes``  peak minus floor: the prefill transient, and
                                the only one of the three the ledger reserves
                                for. ``None`` when no floor was recorded --
                                an honest absence, never a silent fallback to
                                the raw peak, which would re-introduce the
                                exact over-charge this field exists to fix.

    ``group`` is this rank's Weg-2 group ("P", "D", or "" outside Weg-2);
    see :func:`dump_filename`. FIX #1292 also refuses to overwrite an
    existing dump under this filename when its profile digest differs from
    this write's: same filename, different profile is exactly the P-over-D
    collision, and a same-group, same-profile rewrite (the normal
    keep-the-running-peak path in :func:`record_prefill_peak`) is
    unaffected because the digest then matches.

    #1395 (the sibling of #1292's own gap): the write now lands under
    ``<directory>/<_boot_subdir(token)>/<dump_filename(rank, group)>``, never
    under ``directory`` directly, so a SECOND boot of the identical form
    (same group, same rank, same profile digest -- W18 above cannot see this
    axis at all) writes to a DIFFERENT subdirectory instead of destroying the
    first boot's dump. ``boot_token_override`` lets a caller that already
    knows the boot's own token (or a test) supply it directly; the default
    (empty) resolves through :func:`boot_token`, memoised per process so
    every rewrite of THIS rank's dump across one boot's lifetime lands in the
    SAME subdirectory. Defense in depth, matching #1292's own W18 shape one
    axis over (:data:`BOOT_COLLISION_CODE`): even within one subdirectory, a
    write that would overwrite an EXISTING file carrying a DIFFERENT
    ``boot_token`` is refused by name rather than silently replaced -- this
    can only fire if two distinct tokens sanitise to the identical directory
    name, or a caller passes an ``boot_token_override`` that collides with
    another boot's by construction.
    """
    directory = dump_dir or os.environ.get(DUMP_ENV)
    if not directory:
        return None
    delta: Optional[int] = None
    if peak_floor_bytes is not None:
        delta = max(0, int(activation_peak_bytes) - int(peak_floor_bytes))
    from sglang.srt.mem_ledger.activation import profile_digest_from_canonical

    digest = profile_digest_from_canonical(profile_canonical)
    token = boot_token_override or boot_token()
    try:
        boot_dir = os.path.join(directory, _boot_subdir(token))
        os.makedirs(boot_dir, exist_ok=True)
        path = os.path.join(boot_dir, dump_filename(rank, group))
        if os.path.exists(path):
            existing_token = _read_boot_token(path)
            if existing_token is not None and existing_token != token:
                # DEFENSE IN DEPTH (#1395): with per-boot subdirectories this
                # can only fire if two DIFFERENT tokens sanitise to the SAME
                # directory name -- named and refused rather than silently
                # overwritten, the same shape #1292's own W18 uses for a
                # differing profile digest, except on the axis W18 cannot see
                # (two boots of the SAME form share one digest).
                logger.warning(
                    "%s PHASE-FOOTPRINT REFUSED to overwrite %s (group=%r "
                    "rank=%d): existing boot_token %r does not match this "
                    "write's %r. Two boot tokens collided on one directory "
                    "name instead of writing to distinct subdirectories -- "
                    "refusing rather than destroying the earlier boot's "
                    "evidence.",
                    BOOT_COLLISION_CODE, path, group, rank,
                    existing_token, token,
                )
                return None
            existing_digest = None
            try:
                with open(path) as f:
                    existing = json.load(f)
                existing_digest = existing.get("profile_digest") or (
                    profile_digest_from_canonical(existing.get("profile") or [])
                )
            except (OSError, ValueError) as e:
                logger.warning(
                    "phase footprint %s unreadable while checking for a "
                    "collision (%s); refusing rather than guessing.", path, e,
                )
                return None
            if existing_digest != digest:
                logger.warning(
                    "W18 Weg2PhaseFootprintCollision PHASE-FOOTPRINT REFUSED "
                    "to overwrite %s (group=%r rank=%d): existing profile "
                    "digest %s does not match this write's %s. Two Weg-2 "
                    "groups collided on one filename instead of writing "
                    "distinct dumps -- fix the group tag, do not force the "
                    "write.",
                    path, group, rank, existing_digest, digest,
                )
                return None
        payload = {
            "rank": rank,
            "group": group,
            "boot_token": token,
            "card_uuid": card_uuid,
            "hw_fingerprint": hw_fingerprint,
            "profile": profile_canonical,
            "profile_digest": digest,
            "activation_peak_bytes": int(activation_peak_bytes),
            "peak_floor_bytes": (
                None if peak_floor_bytes is None else int(peak_floor_bytes)
            ),
            "activation_delta_bytes": delta,
            "capture_bytes": int(capture_bytes),
            "reserved_peak_bytes": int(reserved_peak_bytes),
            "prefill_tokens": prefill_tokens,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, path)
        logger.info("phase footprint written to %s", path)
        return path
    except OSError as e:  # pragma: no cover - filesystem differences
        logger.warning("could not write the phase footprint dump: %s", e)
        return None


# ---------------------------------------------------------------------------
# Serving-path wiring
# ---------------------------------------------------------------------------
#
# The three call sites below are what turn this module from a design into a
# measurement. Before them the probe had unit tests and no callers, so
# ``SGLANG_PHASE_FOOTPRINT_DUMP`` armed nothing and ``probe_activation.py
# ingest`` could only ever report "No rank dumps" -- the ledger stayed on its
# shipped UPPER_BOUNDs with no way to reach MEASURED_PEAK.
#
# State is module-level (process-wide) ON PURPOSE. ``torch.cuda.memory_stats``
# is per-device and process-wide, and a rank IS a process, so the number the
# ledger reserves for is the process peak -- not one model runner's. This also
# makes the speculative case right for free: the target runner and the NEXTN
# draft runner both capture graphs in the same process, and their capture costs
# must ADD rather than overwrite.

_baseline_allocated: Optional[int] = None
_capture_bytes_total: int = 0
_activation_peak_bytes: int = 0
_identity: Optional[dict] = None
#: What was resident at the last peak re-base; see :func:`reset_peaks`. None
#: until a bracket is opened, which is why the delta can be honestly absent.
_peak_floor_bytes: Optional[int] = None


def _device_index() -> int:
    try:
        import torch

        return int(torch.cuda.current_device())
    except Exception:  # pragma: no cover - non-CUDA platforms
        return 0


def note_capture_begin() -> None:
    """Baseline immediately before graph capture (KV pool already sized)."""
    global _baseline_allocated
    if not is_armed():
        return
    peaks = read_peaks(_device_index())
    _baseline_allocated = int(peaks.get("allocated_bytes", 0))
    reset_peaks(_device_index())


def note_capture_end() -> None:
    """Fold this capture's PERSISTENT cost in, then re-baseline for prefill.

    Capture is charged as the delta in LIVE allocations across the capture,
    not as the peak: the ledger reserves for the memory the captured graphs go
    on holding, and a transient spike during capture is already gone by the
    time the first prefill runs. The peak counters are reset afterwards so the
    activation measurement starts from the post-capture steady state.
    """
    global _capture_bytes_total, _baseline_allocated
    if not is_armed():
        return
    if _baseline_allocated is not None:
        current = int(read_peaks(_device_index()).get("allocated_bytes", 0))
        _capture_bytes_total += max(0, current - _baseline_allocated)
        _baseline_allocated = None
    reset_peaks(_device_index())


def _resolve_identity(model_runner) -> Optional[dict]:
    """Card UUID, rig fingerprint and activation profile for the dump."""
    global _identity
    # (see _global_rank below for why this is not tp_rank)
    if _identity is not None:
        return _identity or None
    try:
        from sglang.srt.managers.weg2_memory_saver import weg2_group_name
        from sglang.srt.mem_ledger.activation import profile_from_server_args
        from sglang.srt.mem_ledger.calibration import rig_fingerprint
        from sglang.srt.mem_ledger.engine import _model_architectures
        from sglang.srt.registry import nvml as registry_nvml

        server_args = model_runner.server_args
        # The RIG fingerprint, not this process's. Every rank here is pinned
        # to one card by CUDA_VISIBLE_DEVICES, and ``live_fingerprint`` would
        # hash only that card -- three ranks, three different fingerprints,
        # none of them the rig's, and ingest rightly refuses all three (#589).
        live = rig_fingerprint()
        profile = profile_from_server_args(
            server_args, _model_architectures(server_args)
        )
        _identity = {
            "card_uuid": registry_nvml.current_device_uuid(),
            "hw_fingerprint": live[0] if live else "",
            "profile_canonical": profile.canonical(),
            "rank": _global_rank(model_runner),
            # FIX #1292: the Weg-2 group this rank belongs to ("P", "D", or
            # "" outside Weg-2) -- see dump_filename(). weg2_group_name()
            # reads SGLANG_WEG2_GROUP, set only by weg2/launcher.py's
            # build_env(); it never raises, so this import cannot turn a
            # non-Weg2 boot into a probe failure.
            "group": weg2_group_name(),
        }
    except Exception as e:  # pragma: no cover - NVML/config availability
        logger.warning("phase footprint probe cannot identify this rank: %s", e)
        _identity = {}
    return _identity or None


def _global_rank(model_runner) -> int:
    """The rank that makes this dump's FILENAME unique.

    NOT ``tp_rank``. Under pure pipeline parallelism -- pp_size 3, tp_size 1 --
    every rank's tp_rank is 0, so all three wrote
    ``phase_footprint_rank0.json`` over one another and the ingest saw a single
    card. The dumps were never wrong, only two of the three were destroyed, and
    the ledger then went on refusing the terms it had in fact measured: boot
    v7pp5 left exactly one file behind for three cards.

    The distributed rank is unique across the job by definition, which is the
    property the filename needs. tp_rank is kept as the fallback for a run with
    no process group, where it is 0 and correct because there is one process.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001 - identity may never break the probe
        pass
    return int(getattr(model_runner, "tp_rank", 0) or 0)


def record_prefill_peak(model_runner, num_tokens: int) -> None:
    """After a prefill: keep the running peak and rewrite this rank's dump.

    Rewritten on every new high rather than once at exit, because there is no
    reliable "last prefill" to hook and a rank that is killed mid-run should
    still leave its best measurement behind. The deepest prefill the rank is
    driven through is the one that sets the number, which is why the operator
    drives a representative deep prefill before ingesting.
    """
    global _activation_peak_bytes
    if not is_armed():
        return
    peaks = read_peaks(_device_index())
    peak = int(peaks.get("allocated_peak_bytes", 0))
    if peak <= _activation_peak_bytes:
        return
    _activation_peak_bytes = peak
    identity = _resolve_identity(model_runner)
    if not identity:
        return
    write_footprint_dump(
        rank=identity["rank"],
        card_uuid=identity["card_uuid"],
        hw_fingerprint=identity["hw_fingerprint"],
        profile_canonical=identity["profile_canonical"],
        activation_peak_bytes=peak,
        capture_bytes=_capture_bytes_total,
        reserved_peak_bytes=int(peaks.get("reserved_peak_bytes", 0)),
        prefill_tokens=int(num_tokens),
        peak_floor_bytes=_peak_floor_bytes,
        group=identity.get("group", ""),
    )
