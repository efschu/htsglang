"""Single owner of the question "may this PINNED host buffer be allocated?".

Before #550 every pinned host pool answered it alone, which cost two things.

1.  THE NUMBER WAS NOT HONEST (#549/#551). Six call sites read
    ``psutil.virtual_memory()``, i.e. ``/proc/meminfo``. Inside this LXC
    container that file is synthesised by lxcfs: ``MemAvailable`` can exceed
    ``MemTotal`` (observed on this rig), and with ``memory.max`` unlimited it
    reports the HOST's figures on a box other containers are also spending.
    A pool whose over-commit is the OOM killer rather than a swap cannot be
    validated against a figure that does not denote anything.
    ``memtier.profile.host_memory_bytes_for_pinning`` is the #407 declared
    owner of that number and consults ``/sys/fs/cgroup``; every check here
    goes through it, so the pinned-RAM question has ONE answer.

2.  NOBODY SUMMED THE POSTS (#547). HiCache sized its host pool from
    ``--hicache-ratio``/``--hicache-size`` and kv-session-offload sized its own
    from ``--kv-session-offload-host-ram-gib``, each validating against the
    whole machine as though it were the only claimant. Two independently
    plausible budgets can be jointly impossible, and because both pools are
    PINNED the over-commit is not a swap -- it is the OOM killer picking a
    victim that need not even be this process. That missing sum, not a
    physical conflict, is what kept the two features mutually refused.

A POST is one named claim on pinned host RAM. Two rules govern them.

*   The guard SUMS, it never caps. Nothing in this module shrinks a request.
    Silently capping a pinned pool moves the failure from a boot-time message
    that names both posts to a later allocation whose victim the OOM killer
    chooses.
*   Every post is priced in the refusal. An operator who is told only the
    total cannot tell which flag to lower.

RANK DIVERGENCE -- a bound worth stating, because it decides where the
authoritative guard lives. ``available`` shrinks as each rank pins its share,
so a check against live availability run inside every TP worker can pass on
rank 0 and raise on rank 2: a rank-divergent boot decision, which is an NCCL
hang rather than an error (the reasoning is spelled out at
``kv_session_offload.host_ram_budget_error``). The runtime registry below is
therefore a BACKSTOP that fires on the second allocation in a process; the
rank-invariant guard that actually gates a configuration is
``joint_pinned_host_error`` called ONCE in the launcher, over configured
numbers only.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Host RAM left free for the OS and every non-pool consumer. Pinned memory is
# non-swappable and this box has no swap at all, so the reserve is the only
# thing standing between a tight configuration and the OOM killer.
PINNED_HOST_RESERVE_BYTES: int = 10 * (1024**3)


@dataclass(frozen=True)
class PinnedHostPost:
    """One named claim on pinned host RAM.

    ``flag`` is the CLI argument an operator would lower to shrink this post.
    It is carried separately from ``name`` because the refusal has to be
    actionable, and the pool class name is not a flag.
    """

    name: str
    flag: str
    nbytes: int


def pinned_host_memory_bytes() -> Tuple[Optional[int], Optional[int]]:
    """``(total, available)`` pinnable host bytes, or ``(None, None)``.

    Delegates to the #407 memtier profile rather than ``psutil`` -- see the
    module docstring. ``(None, None)`` means no honest number was available;
    callers must degrade to "no guard" rather than guess, since refusing a
    boot on a fabricated figure is worse than not checking.
    """
    from sglang.srt.memtier.profile import host_memory_bytes_for_pinning

    return host_memory_bytes_for_pinning()


def _format_posts(posts: Sequence[PinnedHostPost]) -> str:
    return "; ".join(
        f"{p.name} {p.nbytes / 1e9:.2f} GB ({p.flag})" for p in posts if p.nbytes > 0
    )


def joint_pinned_host_error(
    posts: Sequence[PinnedHostPost],
    total_bytes: Optional[int],
    available_bytes: Optional[int],
    reserve_bytes: int = PINNED_HOST_RESERVE_BYTES,
) -> Optional[str]:
    """``None`` when every post fits together, else a complete message.

    Checked against BOTH ceilings because they fail for different reasons and
    an operator fixes them differently: exceeding ``total`` is a configuration
    that no machine state could satisfy, exceeding ``available - reserve`` is
    one that this machine cannot satisfy right now.
    """
    live = [p for p in posts if p.nbytes > 0]
    if not live:
        return None
    if total_bytes is None or available_bytes is None:
        return None
    demand = sum(p.nbytes for p in live)
    breakdown = _format_posts(live)
    if demand > int(total_bytes):
        return (
            f"Pinned host RAM over-committed: {demand / 1e9:.2f} GB requested "
            f"across {len(live)} pool(s) [{breakdown}] exceeds the machine's "
            f"TOTAL host RAM ({int(total_bytes) / 1e9:.2f} GB). These pools are "
            "PINNED and cannot be swapped out."
        )
    usable = int(available_bytes) - int(reserve_bytes)
    if demand > usable:
        return (
            f"Pinned host RAM over-committed: {demand / 1e9:.2f} GB requested "
            f"across {len(live)} pool(s) [{breakdown}] does not fit in "
            f"{int(available_bytes) / 1e9:.2f} GB available minus a "
            f"{int(reserve_bytes) / 1e9:.2f} GB OS reserve = "
            f"{max(0, usable) / 1e9:.2f} GB usable. Lower one of the named "
            "flags, or free host memory first: the pools are pinned, so an "
            "over-commit invokes the OOM killer instead of swapping."
        )
    return None


# --- runtime registry -------------------------------------------------------
#
# Process-local, because pinned pools are allocated per worker process and the
# thing being protected (this container's RAM) is shared by all of them. The
# registry makes the SECOND allocation in a process see the first, which is
# the allocation that would over-commit. It does not and cannot see the other
# ranks' pools -- that is what the launcher-side guard is for.

_registry_lock = threading.Lock()
_registered: Dict[str, PinnedHostPost] = {}


def register_pinned_post(post: PinnedHostPost) -> None:
    with _registry_lock:
        _registered[post.name] = post


def unregister_pinned_post(name: str) -> None:
    """Release a post whose buffer is gone.

    Without this the registry would keep charging for a freed pool, so a
    re-init inside one process (or a test that builds several pools in turn)
    would refuse an allocation that in fact fits -- the mirror image of the
    over-commit this module exists to prevent.
    """
    with _registry_lock:
        _registered.pop(name, None)


def registered_posts() -> Tuple[PinnedHostPost, ...]:
    with _registry_lock:
        return tuple(_registered.values())


def clear_registered_posts() -> None:
    """Drop all posts. For tests and for a re-init inside one process."""
    with _registry_lock:
        _registered.clear()


def check_and_register_pinned_post(
    name: str,
    flag: str,
    requested_bytes: int,
    reserve_bytes: int = PINNED_HOST_RESERVE_BYTES,
) -> None:
    """Admit ``requested_bytes`` for ``name``, or raise naming every post.

    Replaces the per-pool ``psutil.virtual_memory()`` checks. Two changes over
    what each site did alone: the availability figure is the lxcfs-safe one,
    and the demand is this pool PLUS every pool already registered in this
    process instead of this pool by itself.
    """
    total, available = pinned_host_memory_bytes()
    post = PinnedHostPost(name=name, flag=flag, nbytes=int(requested_bytes))
    if total is None or available is None:
        # No honest number -> no guard, and say so once. Refusing to guess
        # beats refusing (or admitting) a boot on a fabricated figure.
        logger.warning(
            "%s: host RAM could not be established honestly (neither "
            "/sys/fs/cgroup nor /proc/meminfo gave a usable pair); allocating "
            "%.2f GB of PINNED host memory unchecked. Size it against the "
            "machine yourself.",
            name,
            int(requested_bytes) / 1e9,
        )
        register_pinned_post(post)
        return
    others = [p for p in registered_posts() if p.name != name]
    err = joint_pinned_host_error(
        list(others) + [post], total, available, reserve_bytes
    )
    if err is not None:
        raise ValueError(err)
    register_pinned_post(post)


def hicache_configured_host_bytes(
    hicache_size_gb: float, hicache_ratio: float
) -> Optional[int]:
    """HiCache's pinned host bytes as far as ARGUMENTS alone determine them.

    ``--hicache-size`` is an absolute GB figure, so it is exact here.
    ``--hicache-ratio`` is a multiple of the DEVICE KV pool, whose size is not
    known until the model is loaded and the memory profile has run; there is
    no honest parse-time number for it, and ``None`` says so rather than
    inventing one. The ratio case is covered at allocation time by
    :func:`check_and_register_pinned_post`.
    """
    if hicache_size_gb and float(hicache_size_gb) > 0:
        return int(float(hicache_size_gb) * 1e9)
    return None
