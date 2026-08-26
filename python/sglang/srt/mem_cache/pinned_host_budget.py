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
import functools
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


def revert_pinned_posts_on_failure(fn):
    """Decorator: undo posts a FAILED call registered, and only those (#729).

    THE WINDOW. Every producer declares its post BEFORE allocating -- on
    purpose, so an over-commitment is refused at the declaration rather than
    discovered at the allocation. If the allocation then raises, the post
    describes bytes that never existed, and #706's credit-back subtracts
    already-allocated posts from the next admission's demand. A post that never
    allocated is credited back anyway, so the next admission is charged too
    little: the registry waves through the very over-commitment it exists to
    refuse.

    WHY A DECORATOR AND NOT A TRY BLOCK PER SITE. All six remaining producers
    are ``__init__`` bodies whose allocation runs to the end of the
    constructor; wrapping each would mean re-indenting six constructors, which
    is a large diff for a small property. This changes one line per site and
    leaves every success path byte-identical -- on success the wrapper does
    nothing at all.

    WHAT IT UNDOES: exactly the posts that appeared during the call, computed
    as a set difference, so a post registered by someone else is never touched.
    Nesting is correct by construction: a subclass ``__init__`` that fails
    after ``super().__init__()`` registered undoes BOTH, which is right,
    because the object as a whole failed.

    #386: the original exception is re-raised UNTOUCHED. Cleanup never
    substitutes the diagnosis.

    LIMIT, stated rather than hidden: the set difference is not safe against a
    CONCURRENT registration from another thread inside the same extent -- that
    post would be undone with the failing one. Every producer this wraps is a
    boot-time constructor on one thread. A future producer that registers
    off-thread must not use this.
    """

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        before = {p.name for p in registered_posts()}
        try:
            return fn(*args, **kwargs)
        except BaseException:
            for post in registered_posts():
                if post.name not in before:
                    unregister_pinned_post(post.name)
            raise

    return _wrapped


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
    # THE ALREADY-ALLOCATED POSTS MUST BE CREDITED BACK, or they are charged
    # twice. `available` is read LIVE, a moment where every post in `others`
    # has ALREADY been allocated -- their bytes are therefore already missing
    # from it. Adding them to the demand as well bills them a second time.
    #
    # `joint_pinned_host_error` is right where it was designed to be used: the
    # launcher calls it once over CONFIGURED numbers, before anything is
    # pinned, so there `available` is untouched by any post and summing them
    # all is exact. Reusing that same comparison as a RUNTIME backstop is what
    # introduces the error, because by then the sum and the availability
    # figure disagree about what has happened.
    #
    # MEASURED, 2026-08-17. The Flip+HiCache boot refused on PP0 with
    # "40.42 GB requested ... does not fit in 33.97 GB available minus a
    # 10.74 GB OS reserve": 35.18 GB of that demand was the three phase-flip
    # weight images, which #695 registers AFTER allocating them
    # (weights_arena.py:428). Sampled against the live serving process the same
    # day: MemAvailable 33.62 GB while the three schedulers held 86.51 GB
    # resident on a 126.75 GB box -- i.e. the images were in RSS and already
    # absent from `available`. The true marginal cost of that boot was the 5.24
    # GB tier, which fits in 23.23 GB with 18 GB to spare.
    #
    # THE PRECONDITION, CORRECTED TWICE (#550, then #871c).
    #
    # #550 corrected an earlier claim that crediting is sound "because
    # registration FOLLOWS allocation for every producer". That is FALSE:
    # `_register_image_post` declares the post BEFORE `_alloc_host_image`
    # allocates. So far so good.
    #
    # BUT #550'S OWN REPLACEMENT CLAUSE WAS ALSO WRONG, and it is the more
    # dangerous of the two because it describes a guard that does not exist. It
    # said the early declaration happens "so the registry refuses an
    # over-commitment at the DECLARATION rather than discovering it at the
    # allocation". THE REGISTRY REFUSES NOTHING THERE. `register_pinned_post`
    # above is a bare dict write with no comparison in it, `_register_image_post`
    # says "Never raises" in its own first line, and `weights_arena` does not
    # call `check_and_register_pinned_post` anywhere -- checked with
    # `grep -c check_and_register_pinned_post weights_arena.py` -> 0.
    #
    # The early declaration is deliberate for a DIFFERENT reason, stated at the
    # call site (weights_arena.py, `_alloc_host_image`): "Registered, not
    # CHECKED: a new refusal path here could break a boot that works today, and
    # the diagnosis this is for is served by the number being present, not by a
    # veto." That is a reasoned #695 decision about the largest post in the
    # system -- the phase-flip host weight images -- and it stands. What must
    # not stand is a comment HERE promising a veto THERE: a later reader sizing
    # against this module would believe the biggest claimant is admitted when
    # it is only counted.
    #
    # What actually makes the credit sound is weaker and sufficient: by the time
    # a LATER post is weighed, the EARLIER posts' allocations have COMPLETED, so
    # their bytes are already absent from `available`. That is what was measured
    # -- MemAvailable 33.62 GB while the schedulers held 86.51 GB resident.
    #
    # The real hazard is therefore a post that is registered and then NEVER
    # allocates: it would be credited back without ever having been resident,
    # and the next admission would be charged too little -- the registry waving
    # through the over-commitment it exists to refuse. #550 closes exactly that
    # window by taking the post back when the allocation raises
    # (`_unregister_image_post`). A future producer that can leave a post
    # registered with no allocation behind it must do the same.
    #
    # Other ranks' pins are correctly still charged: they are absent from
    # `available` and absent from `others`, which is this process's registry
    # only -- the rank-divergence bound in the module docstring is unchanged.
    already_allocated = sum(int(p.nbytes) for p in others)
    err = joint_pinned_host_error(
        list(others) + [post],
        total,
        int(available) + already_allocated,
        reserve_bytes,
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
