"""fnFL2x100: A LEG THAT DIED MUST BE NAMED TO THE WAITERS AT ONCE.

Boot fnFL2x100 (22:12:34Z, D->P sleep leg): D TP1 raised W106 on
``weights_14`` in its pause loop and went straight into its group fence.
Nobody else learned of it for 120 s:

* P PP1 sat in its VRAM-credit wait for ``weights_10`` on card 1
  (``published_mib=2072 need_mib=4210 peer_leg_complete=False``) -- the
  credit TP1 would have posted after pausing ``weights_10``, a tag TP1 never
  reached;
* D TP0 sat in its BAR1 deposit of ``p0/weights_10`` waiting for P PP1's
  ``free`` credit ("no 'free' for batch 0 within 120 s (collector gone or
  stuck)");
* P PP0 sat in ``lane_mode`` for ``p2/weights_1`` (TP1's deposit) and fell
  back to the host path at 22:14:34.

Every one of those waits asks a ``liveness`` probe between its polls, and the
probe only asked "is the process on my card still there" -- TP1 was alive,
it was waiting in its fence. The failure itself was already known (TP1 held
the exception); it just had no channel to the waiters.

This module is that channel: a failing rank posts ONE small file under
``/dev/shm/weg2-legabort-<boot>/`` keyed by the flip index both groups share
(the front's ``<boot>.<flip>`` epoch, the same number the BAR1 flag seq
carries), and every wait that already polls a liveness probe or a lane reader
also reads this directory. A wait that finds another rank's abort for its own
flip stops within one poll and says whose leg died and why.

Stateless on purpose: no process-local cache, no counter -- the directory IS
the state, and it is keyed so that a previous flip's or a previous boot's
abort can never stop this one.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import msgspec

#: THE ONE ROOT, shared with the lanes' ``/dev/shm/weg2-seq-<boot>`` tree.
DEFAULT_ROOT = "/dev/shm"
_PREFIX = "weg2-legabort-"
#: A reason is a log line, not a transcript: the W29 texts run to kilobytes.
REASON_MAX_CHARS = 700


class Weg2FlipPeerLegAborted(RuntimeError):
    """W121: another rank's leg of THIS flip already failed; waiting on is
    waiting for bytes, credits or collects that will never come."""


class LegAbort(msgspec.Struct, frozen=True):
    group: str
    rank: int
    flip: int
    reason: str
    t: float


def abort_dir(*, boot_nonce: str, root: Optional[str] = None) -> str:
    return os.path.join(DEFAULT_ROOT if root is None else root,
                        f"{_PREFIX}{boot_nonce}")


def _file_name(*, flip: int, group: str, rank: int) -> str:
    return f"{int(flip)}.{group}{int(rank)}.json"


def post(*, boot_nonce: str, flip: int, group: str, rank: int, reason: str,
         root: Optional[str] = None) -> Optional[str]:
    """Publish this rank's failed leg; the path, or ``None`` when there is no
    flip to key it by (``flip < 0``: the boot-time sleep is not a flip) or no
    boot nonce. Atomic (tmp + rename), so a reader never sees half a record."""
    if not boot_nonce or int(flip) < 0:
        return None
    d = abort_dir(boot_nonce=boot_nonce, root=root)
    os.makedirs(d, exist_ok=True)
    rec = LegAbort(group=str(group), rank=int(rank), flip=int(flip),
                   reason=str(reason)[:REASON_MAX_CHARS], t=time.time())
    path = os.path.join(d, _file_name(flip=flip, group=group, rank=rank))
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(msgspec.json.encode(rec))
    os.replace(tmp, path)
    return path


def read(*, boot_nonce: str, flip: int,
         root: Optional[str] = None) -> Tuple[LegAbort, ...]:
    """Every abort posted for ``flip`` of this boot, oldest first. An absent
    directory is "nobody aborted", not an error; an unreadable record is
    skipped (the writer renames atomically, so that is a foreign file)."""
    if not boot_nonce or int(flip) < 0:
        return ()
    d = abort_dir(boot_nonce=boot_nonce, root=root)
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return ()
    head = f"{int(flip)}."
    out = []
    for name in names:
        if not name.startswith(head) or not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, name), "rb") as fh:
                out.append(msgspec.json.decode(fh.read(), type=LegAbort))
        except (OSError, msgspec.DecodeError):
            continue
    return tuple(sorted(out, key=lambda a: a.t))


def foreign(*, boot_nonce: str, flip: int, group: str, rank: int,
            root: Optional[str] = None) -> Tuple[LegAbort, ...]:
    """The aborts of this flip posted by any OTHER rank (either group)."""
    return tuple(a for a in read(boot_nonce=boot_nonce, flip=flip, root=root)
                 if not (a.group == str(group) and a.rank == int(rank)))


def describe(aborts: Tuple[LegAbort, ...]) -> str:
    """One clause per aborted rank, for the refusal that stops a waiter."""
    return "; ".join(f"{a.group} rank {a.rank}: {a.reason}" for a in aborts)
