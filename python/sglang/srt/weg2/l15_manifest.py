"""L1.5 hold manifest + group agreement (AP L15-04, pure stdlib, no torch).

One rank publishes its L1.5 hold state (which requests hold which slots,
down to which L2 generations) as a small JSON record in shared memory; the
group reads every rank's record and agrees on a single fingerprint to decide
whether the common step is a "hold" or the "fallback" of plan 2.3/2 (every
rank drops H together).

Storage pattern reused from ``weg2/card_kv_ledger.py`` (verified by the lead):
one small record in /dev/shm, updated under ``fcntl.flock``, atomic write via
``<path>.tmp`` + ``os.replace``, and a pid liveness reap -- a manifest whose
owning process died describes memory the driver already freed, so ``read``
removes it and reports absence.

The fingerprint deliberately ignores ``pid``: the same hold content from two
different processes must hash identically, otherwise the group could never
agree. It canonicalizes span order (``to_json`` sorts spans by rid), so a
permutation of the same spans cannot split the group, while any changed slot,
L2 generation, depth or rows_by_rank value must.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
from typing import Optional, Tuple


@dataclasses.dataclass(frozen=True)
class HoldSpan:
    """One held context: its request id, the hold depth, the L1 slots it
    occupies, the anchor slot it is pinned to, and where its suffix lives in
    L2 (slots plus the generation each L2 slot was written at)."""

    rid: str
    depth: int
    slots: Tuple[int, ...]
    anchor_slot: int
    l2_slots: Tuple[int, ...]
    l2_gens: Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Manifest:
    """A rank's whole L1.5 hold state at one epoch, as published for the
    group. ``rows_by_rank`` is the per-rank row budget and ``anchor_slots``
    the anchor region size -- both part of the group agreement.
    ``rows_by_rank`` is the per-rank KEEP capacity (blocks*ratio_r from the
    compact plan), NOT the admitted rows -- ``l15_policy.HoldSet.rows_by_rank``
    carries the latter, and the L15-RETAIN/L15-RESTORE log lines print this
    field as ``keep_rows_by_rank``."""

    epoch: int
    pid: int
    spans: Tuple[HoldSpan, ...]
    rows_by_rank: Tuple[int, ...]
    anchor_slots: int


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def manifest_path(group: str, rank: int, env) -> str:
    """The per-(group, rank) manifest file.

    Co-located ranks on one host must each own a SEPARATE file: with one
    shared default they overwrite each other's record at sleep, and at
    wake they all read the SAME file -> identical fingerprint -> a FALSE
    "hold" agreement (the group votes on one rank's content). When
    ``SGLANG_WEG2_L15_MANIFEST`` is set it is a directory (when it ends
    with a path separator) or a path prefix; otherwise the default is
    ``/tmp/weg2_l15_manifest.<group>.<rank>.json``.
    """
    override = ""
    if env:
        try:
            override = env.get("SGLANG_WEG2_L15_MANIFEST", "") or ""
        except AttributeError:  # pragma: no cover - non-mapping guard
            override = ""
    if not override:
        return "/tmp/weg2_l15_manifest.%s.%d.json" % (group, rank)
    if override.endswith(os.sep):
        return override + "weg2_l15_manifest.%s.%d.json" % (group, rank)
    return "%s.%s.%d.json" % (override, group, rank)


def _span_to_dict(s: HoldSpan) -> dict:
    return {
        "rid": s.rid,
        "depth": s.depth,
        "slots": list(s.slots),
        "anchor_slot": s.anchor_slot,
        "l2_slots": list(s.l2_slots),
        "l2_gens": list(s.l2_gens),
    }


def to_json(m: Manifest) -> str:
    """Canonical JSON: keys sorted, spans sorted by rid (span order must not
    affect the fingerprint, so it must not affect the serialization)."""
    obj = {
        "epoch": m.epoch,
        "pid": m.pid,
        "spans": [_span_to_dict(s) for s in sorted(m.spans, key=lambda s: s.rid)],
        "rows_by_rank": list(m.rows_by_rank),
        "anchor_slots": m.anchor_slots,
    }
    return json.dumps(obj, sort_keys=True)


def _require(obj: dict, field: str, where: str) -> None:
    if field not in obj:
        raise ValueError(f"malformed L1.5 manifest: {where} missing field {field!r}")


def _as_int(obj: dict, field: str, where: str) -> int:
    """Return ``int(obj[field])``; on a non-integer value raise ValueError
    naming where and the field -- the from_json docstring promise. For a
    list element the caller passes the element name (``'slots[0]'``) with a
    single-element view of the list."""
    value = obj[field]
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: field {field!r} is not an integer: {value!r}") from None


def _int_list(obj: dict, field: str, where: str) -> Tuple[int, ...]:
    """Int-coerce every element of the list obj[field]; a non-integer
    element raises naming it as ``field[i]``."""
    raw = obj[field]
    if not isinstance(raw, list):
        raise ValueError(f"malformed L1.5 manifest: {where} field {field!r} is not a list")
    return tuple(
        _as_int({f"{field}[{i}]": x}, f"{field}[{i}]", where) for i, x in enumerate(raw)
    )


def from_json(s: str) -> Manifest:
    """Inverse of :func:`to_json`. A malformed record raises ValueError
    naming the offending field."""
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed L1.5 manifest: not valid json: {exc}") from None
    if not isinstance(obj, dict):
        raise ValueError("malformed L1.5 manifest: top level is not an object")
    for field in ("epoch", "pid", "spans", "rows_by_rank", "anchor_slots"):
        _require(obj, field, "record")
    if not isinstance(obj["spans"], list):
        raise ValueError("malformed L1.5 manifest: 'spans' is not a list")
    spans = []
    for i, sp in enumerate(obj["spans"]):
        where = f"spans[{i}]"
        if not isinstance(sp, dict):
            raise ValueError(f"malformed L1.5 manifest: {where} is not an object")
        for field in ("rid", "depth", "slots", "anchor_slot", "l2_slots", "l2_gens"):
            _require(sp, field, where)
        spans.append(
            HoldSpan(
                rid=str(sp["rid"]),
                depth=_as_int(sp, "depth", where),
                slots=_int_list(sp, "slots", where),
                anchor_slot=_as_int(sp, "anchor_slot", where),
                l2_slots=_int_list(sp, "l2_slots", where),
                l2_gens=_int_list(sp, "l2_gens", where),
            )
        )
    return Manifest(
        epoch=_as_int(obj, "epoch", "record"),
        pid=_as_int(obj, "pid", "record"),
        spans=tuple(spans),
        rows_by_rank=_int_list(obj, "rows_by_rank", "record"),
        anchor_slots=_as_int(obj, "anchor_slots", "record"),
    )


def write(path: str, m: Manifest) -> None:
    """Publish under ``flock`` on ``<path>.lock``; land atomically with a
    tmp file + ``os.replace`` so a reader never sees a half record."""
    with open(path + ".lock", "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                fh.write(to_json(m))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def read(path: str, pid_alive=_pid_alive) -> Optional[Manifest]:
    """Return the manifest, or None when absent. A record whose owning pid
    is dead is reaped here (removed and reported absent): the driver freed
    that process's memory, so its hold description is stale by definition."""
    try:
        with open(path, "r") as fh:
            raw = fh.read()
    except FileNotFoundError:
        return None
    m = from_json(raw)
    if not pid_alive(m.pid):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return None
    return m


def read_and_clear(path: str, pid_alive=_pid_alive) -> Optional[Manifest]:
    """read() plus the unlink of the consumed record.

    The manifest's lifetime is ONE sleep-wake pair: the wake is the only
    reader, and a record left on disk would let a LATER wake re-vote on a
    stale record from an earlier sleep. That consumption is also what makes
    the missing wake-epoch comparison moot (the wake never compares its own
    epoch against the manifest's): by construction the wake reads only the
    record its own sleep wrote, and that record is gone after the vote.
    ``read`` already unlinks a dead-pid record (and returns None); this
    unlinks the valid one too, so a second read gives None.
    """
    m = read(path, pid_alive)
    if m is not None:
        try:
            os.remove(path)
        except FileNotFoundError:  # pragma: no cover - concurrent consumer
            pass
    return m


def fingerprint(m: Manifest) -> int:
    """Signed int64 over the canonical serialization WITHOUT ``pid`` (the
    same content from two processes must agree). Stable under span order;
    changes when any slot, generation, depth or rows_by_rank changes."""
    obj = json.loads(to_json(m))
    obj.pop("pid", None)
    payload = json.dumps(obj, sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=True)


def agree(min_fp: int, max_fp: int) -> bool:
    """Group agreement: the whole group agrees exactly when the minimum and
    maximum fingerprint collected over the ranks coincide."""
    return min_fp == max_fp


def decide(min_fp: int, max_fp: int) -> str:
    """Agreement -> "hold". Disagreement -> "fallback", the common fallback
    of plan 2.3/2: every rank drops H together (no rank keeps a hold the
    group did not confirm)."""
    return "hold" if agree(min_fp, max_fp) else "fallback"
