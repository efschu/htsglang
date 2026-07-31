# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Cross-session card arbitration for the workbench (DESIGN #347 W5).

Two things can want this rig's cards that cannot talk to each other: the
serving session and whatever else runs on the same hardware (on this rig, a
driver session on the PVE host). They share a directory and the hardware, and
nothing else. The protocol in that directory's ``README.md`` is:

* availability is **published**, not negotiated -- ``free-until`` names a
  window during which the other side may take the cards without asking;
* whoever uses cards writes ``holder`` and **touches it as a heartbeat**;
* a ``holder`` that is stale *and* whose cards are empty is an orphan and may
  be reaped, with a line in ``log``; a stale ``holder`` on **busy** cards is a
  working holder that forgot to touch, and is left alone;
* before every access, regardless of what any file says, the hardware is
  checked -- the files are intentions and can go stale, the hardware cannot.

This module is that protocol as code, for the one side the workbench is on.
It never performs a destructive action (no module reload, no reboot, no
killing anyone else's process) and it never waits: a refused claim returns
immediately with a reason, and the scheduler tries again on a later tick.

The directory path is a flag, not a constant. Baking this rig's path into the
tree is exactly the rig-only assumption ANALYSE #347 excludes: the protocol
generalizes, the path does not.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

#: A card holding more than this is treated as in use. Same number the
#: arbitration README names as the safety net.
DEFAULT_BUSY_BYTES = 500 * MIB

#: A ``holder`` older than this has stopped heartbeating. The README's number.
DEFAULT_STALE_AFTER_S = 20 * 60.0


def _utc_now_iso(ts: Optional[float] = None) -> str:
    return (
        datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_free_until(text: str) -> tuple[Optional[float], set[int]]:
    """``<ISO-UTC>  cards=<list|all>  by=...  note=...`` -> (epoch, indices).

    An unparsable line yields ``(None, set())``, which means "no window", the
    safe direction: a window that cannot be read must not be treated as open.
    """
    line = (text or "").strip().splitlines()
    if not line:
        return None, set()
    parts = line[0].split()
    if not parts:
        return None, set()
    try:
        stamp = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        expiry = stamp.timestamp()
    except ValueError:
        return None, set()
    cards: set[int] = set()
    for token in parts[1:]:
        key, _, value = token.partition("=")
        if key != "cards":
            continue
        if value.strip().lower() in ("all", "*"):
            return expiry, set()
        for item in value.split(","):
            with contextlib.suppress(ValueError):
                cards.add(int(item.strip()))
    return expiry, cards


def parse_holder(text: str) -> dict[str, str]:
    """``key=value  key=value`` -> dict. Unknown keys are kept."""
    out: dict[str, str] = {}
    for token in (text or "").strip().split():
        key, sep, value = token.partition("=")
        if sep:
            out[key] = value
    return out


class ArbRefused(RuntimeError):
    """The window could not be claimed. Carries the reason the operator needs."""


class ArbClaim:
    """A held window. Heartbeat it, then release it."""

    def __init__(self, directory: ArbDirectory, indices: Sequence[int], purpose: str):
        self.directory = directory
        self.indices = tuple(sorted(indices))
        self.purpose = purpose
        self.since = time.time()
        self._released = False

    def heartbeat(self) -> None:
        """Re-stamp ``holder``. Must be called well inside the stale window.

        A workbench window can exceed twenty minutes -- a tuner queue or a long
        training attempt -- so this is called from the supervision loop, not
        once at claim time. A claim that stops heartbeating looks exactly like
        a crashed session, and the other side is entitled to reap it.
        """
        if self._released:
            return
        self.directory._write_holder(self.indices, self.purpose, self.since)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.directory._release_holder(self.indices, self.purpose)

    @property
    def released(self) -> bool:
        return self._released

    def to_json(self) -> dict[str, Any]:
        return {
            "cards": list(self.indices),
            "purpose": self.purpose,
            "since": _utc_now_iso(self.since),
            "released": self._released,
        }


class ArbDirectory:
    """The shared arbitration directory, from the workbench's side."""

    def __init__(
        self,
        root: str | Path,
        *,
        session: str = "operator",
        busy_bytes: int = DEFAULT_BUSY_BYTES,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
        occupancy: Optional[Callable[[Sequence[int]], dict[int, int]]] = None,
        accounted: Optional[Callable[[Sequence[int]], dict[int, int]]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self.session = session
        self.busy_bytes = int(busy_bytes)
        self.stale_after_s = float(stale_after_s)
        self.occupancy = occupancy or _nvml_occupancy
        #: Bytes on each card that a *known* tenant legitimately holds, read
        #: from the VRAM ledger. Subtracted from the NVML reading before the
        #: busy test, because the workbench runs beside a resident serving
        #: engine: that engine's memory is accounted for and is not evidence
        #: of a foreign process. Unaccounted memory still refuses the claim,
        #: which is the case this check exists for.
        self.accounted = accounted or (lambda indices: {})
        self._clock = clock

    # -- paths --------------------------------------------------------------

    @property
    def holder_path(self) -> Path:
        return self.root / "holder"

    @property
    def free_until_path(self) -> Path:
        return self.root / "free-until"

    @property
    def log_path(self) -> Path:
        return self.root / "log"

    def usable(self) -> tuple[bool, str]:
        if not self.root.is_dir():
            return False, f"{self.root} is not a directory"
        if not os.access(self.root, os.W_OK):
            return False, f"{self.root} is not writable by this process"
        return True, ""

    # -- reading ------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        ok, reason = self.usable()
        body: dict[str, Any] = {
            "root": str(self.root),
            "session": self.session,
            "usable": ok,
            "reason": reason,
        }
        if not ok:
            return body
        holder_text = _read(self.holder_path)
        if holder_text:
            age = self._clock() - self.holder_path.stat().st_mtime
            body["holder"] = {
                **parse_holder(holder_text),
                "age_s": round(age, 1),
                "stale": age > self.stale_after_s,
                "mine": parse_holder(holder_text).get("session") == self.session,
            }
        expiry, cards = parse_free_until(_read(self.free_until_path))
        if expiry:
            body["free_until"] = {
                "until": _utc_now_iso(expiry),
                "open": expiry > self._clock(),
                "cards": sorted(cards) or "all",
            }
        return body

    # -- claiming -----------------------------------------------------------

    def claim(self, indices: Sequence[int], purpose: str) -> ArbClaim:
        """Take the window, or raise :class:`ArbRefused` with the reason.

        Never blocks and never waits for the other side. Order of checks is
        deliberate: cheap file reads first, the hardware check last and always,
        so a refusal costs no NVML call when the files already say no.
        """
        ok, reason = self.usable()
        if not ok:
            raise ArbRefused(f"arbitration directory unusable: {reason}")
        wanted = sorted(set(int(i) for i in indices))
        now = self._clock()

        expiry, free_cards = parse_free_until(_read(self.free_until_path))
        if expiry and expiry > now and (not free_cards or free_cards & set(wanted)):
            raise ArbRefused(
                f"a free window is published until {_utc_now_iso(expiry)} for "
                f"cards {sorted(free_cards) or 'all'}; the other session may "
                "take them without asking, so this side stays off them"
            )

        holder_text = _read(self.holder_path)
        if holder_text:
            fields = parse_holder(holder_text)
            age = now - self.holder_path.stat().st_mtime
            if fields.get("session") == self.session and age <= self.stale_after_s:
                # Our own live claim. One window at a time, by construction.
                raise ArbRefused(
                    f"this session already holds cards {fields.get('cards', '?')} "
                    f"for {fields.get('purpose', '?')}"
                )
            if age <= self.stale_after_s:
                raise ArbRefused(
                    f"held by {fields.get('session', '?')} for "
                    f"{fields.get('purpose', '?')} since "
                    f"{fields.get('since', '?')} ({age:.0f}s ago)"
                )
            # Stale. Orphan only if its cards are empty -- a stale holder on
            # busy cards is a working holder that forgot to touch.
            held = _indices_of(fields.get("cards", ""))
            busy = self._busy(held or wanted)
            if busy:
                self._log(
                    f"stale holder ({age:.0f}s) left in place, cards busy: "
                    f"{busy}; content: {holder_text.strip()}"
                )
                raise ArbRefused(
                    f"a stale holder ({age:.0f}s) names cards that are still "
                    f"busy ({busy}); not reaping it"
                )
            self._log(
                f"reaped orphan holder, age {age:.0f}s, cards empty; "
                f"content: {holder_text.strip()}"
            )
            with contextlib.suppress(OSError):
                self.holder_path.unlink()

        busy = self._busy(wanted)
        if busy:
            raise ArbRefused(
                "the hardware says these cards are in use even though no "
                f"holder claims them: {busy}"
            )

        claim = ArbClaim(self, wanted, purpose)
        self._write_holder(claim.indices, purpose, claim.since)
        self._log(f"claimed cards {wanted} for {purpose}")
        return claim

    # -- internals ----------------------------------------------------------

    def _busy(self, indices: Sequence[int]) -> str:
        if not indices:
            return ""
        try:
            used = self.occupancy(list(indices))
        except Exception as exc:  # noqa: BLE001
            # An unreadable NVML must not become permission to proceed: the
            # README's rule is that a file may never make a busy card look
            # free, and the same holds for a failed probe.
            return f"the occupancy check failed ({type(exc).__name__}: {exc})"
        try:
            known = self.accounted(list(indices))
        except (
            Exception
        ) as exc:  # noqa: BLE001 - an unreadable ledger accounts for nothing
            logger.debug("workbench arb: ledger accounting failed: %s", exc)
            known = {}
        offenders = []
        for i in indices:
            unaccounted = int(used.get(i, 0)) - int(known.get(i, 0))
            if unaccounted > self.busy_bytes:
                offenders.append(
                    f"card {i} holds {used.get(i, 0) / MIB:.0f} MiB, of which "
                    f"{unaccounted / MIB:.0f} MiB is not in the VRAM ledger"
                )
        return "; ".join(offenders)

    def _write_holder(self, indices: Sequence[int], purpose: str, since: float) -> None:
        line = (
            f"session={self.session}  cards={','.join(str(i) for i in indices)}  "
            f"purpose={purpose}  since={_utc_now_iso(since)}\n"
        )
        with contextlib.suppress(OSError):
            self.holder_path.write_text(line, encoding="utf-8")

    def _release_holder(self, indices: Sequence[int], purpose: str) -> None:
        text = _read(self.holder_path)
        if text and parse_holder(text).get("session") != self.session:
            # Somebody else's holder appeared while we worked. Do not delete
            # another session's file; say so and leave it.
            self._log(
                f"release skipped: holder now belongs to "
                f"{parse_holder(text).get('session', '?')}"
            )
            return
        with contextlib.suppress(OSError):
            self.holder_path.unlink()
        self._log(f"released cards {list(indices)} after {purpose}")

    def _log(self, message: str) -> None:
        with contextlib.suppress(OSError):
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{_utc_now_iso(self._clock())}  {self.session}  {message}\n"
                )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _indices_of(raw: str) -> list[int]:
    out: list[int] = []
    for token in (raw or "").split(","):
        with contextlib.suppress(ValueError):
            out.append(int(token.strip()))
    return out


def _nvml_occupancy(indices: Sequence[int]) -> dict[int, int]:
    """Used bytes per NVML index. The hardware, not a file."""
    from sglang.srt.registry import nvml

    wanted = set(int(i) for i in indices)
    out: dict[int, int] = {}
    for device in nvml.list_devices():
        if device.index in wanted:
            out[device.index] = nvml.memory_info_for_uuid(device.uuid).used_bytes
    return out


__all__ = [
    "ArbClaim",
    "ArbDirectory",
    "ArbRefused",
    "DEFAULT_BUSY_BYTES",
    "DEFAULT_STALE_AFTER_S",
    "parse_free_until",
    "parse_holder",
]
