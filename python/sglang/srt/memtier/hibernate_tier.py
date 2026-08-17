# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#407 cut 6: resolve ``--hibernate-dir`` through the tier registry.

THE HOLE THIS CLOSES. ``--enable-weights-disk-backup`` promises weights are
parked on disk and restorable after a restart. ``ServerArgs`` validated that the
flag and ``--hibernate-dir`` were passed together and that the checkpoint was
GGUF, then accepted **any directory** -- including one on a tmpfs. Every write
succeeds, the run reports a successful park, and the backup is gone at the next
reboot. Nothing tells the operator, because from the process's side nothing
failed. ``hibernate.py`` even takes an ``flock`` on that directory to serialise
co-located ranks, which works perfectly on a tmpfs and so proves nothing about
durability.

WHY THIS IS A WIRING CUT AND NOT A NEW MECHANISM. The registry already models
it: :data:`bootstrap.NON_PERSISTENT_FS_TYPES` names the volatile filesystems,
``bootstrap`` derives ``persistent`` from the fs type, and
:func:`bootstrap.collect_fs_types` resolves a directory to its mount by LONGEST
match -- its docstring names ``--hibernate-dir /dev/shm/img`` as exactly the
case that matters, because that path is on a tmpfs while not being a mount
point itself. This module is the consumer side that never asked.

UNKNOWN IS A WARNING, NOT A REFUSAL, and the asymmetry is deliberate.
``bootstrap`` treats an unresolvable fs type as not-persistent, which is right
for RANKING a tier: an unknown medium must not win a placement. It is wrong for
a REFUSAL, because a container without a readable ``/proc/mounts`` would then be
unable to hibernate at all -- a working configuration broken by a gate that
cannot see. So this refuses only what it can positively identify as volatile.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Dict, Optional

from sglang.srt.memtier.bootstrap import NON_PERSISTENT_FS_TYPES, collect_fs_types

logger = logging.getLogger(__name__)


class HibernateDirNotPersistent(ValueError):
    """``--hibernate-dir`` points at a filesystem that does not survive a reboot."""


@dataclasses.dataclass(frozen=True)
class HibernateDirVerdict:
    """What the registry can say about a hibernate directory's medium."""

    path: str
    #: ``None`` when the mount could not be resolved.
    fs_type: Optional[str]
    #: Whether the filesystem type was identified at all.
    known: bool
    #: True only when identified AND not in the volatile set. An unknown medium
    #: is never reported persistent -- it is reported unknown.
    persistent: bool


def hibernate_dir_verdict(
    path: str, fs_types: Optional[Dict[str, str]] = None
) -> HibernateDirVerdict:
    """Resolve ``path`` to its filesystem type and judge its durability.

    ``fs_types`` is injectable so the volatile and persistent cases are both
    testable on a machine that happens to have only one of them mounted.
    """
    table = collect_fs_types([path]) if fs_types is None else dict(fs_types)
    fs_type = table.get(path)
    if fs_type is None:
        return HibernateDirVerdict(path, None, False, False)
    lowered = str(fs_type).lower()
    return HibernateDirVerdict(
        path, lowered, True, lowered not in NON_PERSISTENT_FS_TYPES
    )


def refuse_volatile_hibernate_dir(
    path: str, fs_types: Optional[Dict[str, str]] = None
) -> None:
    """Raise when the hibernate directory is provably volatile.

    Returns None for a persistent medium AND for an unresolvable one; see the
    module docstring for why those two share an outcome.
    """
    verdict = hibernate_dir_verdict(path, fs_types=fs_types)
    if verdict.known and not verdict.persistent:
        raise HibernateDirNotPersistent(
            f"--hibernate-dir {path!r} is on a {verdict.fs_type} filesystem, "
            "which does not survive a reboot. --enable-weights-disk-backup "
            "would report a successful park and lose the backup at the next "
            "reboot, with nothing failing at the time to tell you. Point it at "
            "persistent storage, or drop --enable-weights-disk-backup if the "
            "park was only meant to last for this boot."
        )
    if not verdict.known:
        logger.warning(
            "#407 cut 6: could not resolve the filesystem behind "
            "--hibernate-dir %s, so its durability is unverified. Hibernate "
            "proceeds; if this is a tmpfs the disk backup will not survive a "
            "reboot.",
            path,
        )
