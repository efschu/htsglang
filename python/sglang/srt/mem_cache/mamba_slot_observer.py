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
"""#743: make mamba/GDN slot pressure observable per event.

THE GAP THIS CLOSES, in the words of the note that could not answer its own
question (``NOTE_743_mamba_slot_hitrate.md`` §4.1-4.2): a successful mamba slot
eviction is SILENT, and a prefix match cut short by a mamba tombstone is
indistinguishable in the logs from a genuine cache miss. Both are the same
event seen from two ends -- slot pressure destroying a reusable prefix -- and
neither end emitted anything, so "did the 12-slot pool cost us prefix reuse"
was unanswerable from a boot log. The note's §5 states the ordering
explicitly: the instrument comes BEFORE the agent-shaped soak, because without
it the soak produces the same unanswerable logs.

WHAT ALREADY EXISTED, and why it does not cover this:

* ``MambaRadixCache._log_mamba_slot_starvation`` (mamba_radix_cache.py) fires
  on the pool being exhausted with NOTHING evictable. That is the failure
  case. The case #743 asks about is the pool SUCCEEDING -- yielding a slot by
  destroying a cached anchor -- which that emitter is silent on by
  construction.
* the ``SGLANG_MAMBA_CKPT_DEBUG`` per-request line reports full_match vs
  resume length, but only under an off-by-default debug flag and only inside
  the ``--mamba-checkpoint-interval`` lineage. The specimen boot ran neither.

ONE OBSERVER, BOTH LINEAGES. ``MambaRadixCache`` (device-only) and
``HiMambaRadixCache`` (host-tier) reach the same two events by different code,
and #747's rule -- one anchor rule, both lineages -- applies to the instrument
as much as to the policy: two spellings of one emitter is how the two lineages
drift apart and how a future reader compares numbers that do not mean the same
thing. Both call into this module.

RATE CONTROL IS A TOKEN BUCKET, NOT "FIRST N THEN EVERY 1000". The starvation
emitter's cadence is right for its job -- a starved pool hits it every
scheduler step and the log must not flood -- but it is wrong for an
instrument: three lines and then silence for a thousand events cannot answer
"how often, and how bad". A bucket gives PER-EVENT detail while pressure is
normal and degrades to a ROLLUP under thrash, and the rollup carries the
totals, so no event is ever lost from the arithmetic even when its line is.

PURE, AND SEPARATE FROM THE LOGGER. Every method returns the line(s) it wants
emitted rather than emitting them, so the cadence, the arithmetic and the
wording are all testable against a fake clock without capturing log output and
without building a radix tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

#: Unbracketed and uppercase, matching ``gdn_slot_executor.py``'s
#: ``GDN-SLOT-LADDER`` -- the nearest sibling in this subsystem -- and the
#: boot-log families a reader greps alongside it (PHASE-FLIP, CORRIDOR-GUARD,
#: PARKED-DECODE). The bracketed ``[#NNN name]`` form used by the hicache
#: modules is the other convention in this directory; it is not used here
#: because these lines are read from a boot log next to those families, not
#: from a feature's own trace.
LOG_PREFIX = "MAMBA-SLOT"

#: SUSTAINED lines per second. Overridable per boot
#: (``SGLANG_MAMBA_SLOT_LOG_RATE``); 0 disables the instrument entirely, which
#: is the only way to get the pre-#743 silence back.
#:
#: 2/s, not 20/s. This is a PERMANENTLY ON instrument at WARNING level, so its
#: sustained cost is paid by every boot forever: 20/s is 1200 lines a minute on
#: a thrashing rig, which buries the very log the reader came to search. 2/s
#: still prints every event on a rig where slot eviction is occasional -- which
#: is the healthy case #743 wants confirmed -- and degrades to the rollup
#: exactly when the answer has stopped being "which node" and become "how
#: often", where the totals are the useful form anyway.
DEFAULT_LOG_RATE_PER_S = 2.0

#: Bucket capacity, DECOUPLED from the drain rate. A slot shortage does not
#: arrive at a steady 2/s -- it arrives as several evictions inside one
#: scheduler step, and a bucket whose capacity equals its rate would print the
#: first and suppress the rest of exactly the burst that matters. Capacity 8
#: covers a full pool turnover on this rig's 12-slot pool in one line group,
#: then drains at the sustained rate.
DEFAULT_LOG_BURST = 8.0

#: ``(node_id, anchor_tokens)`` for one node whose mamba state was freed.
#: ``anchor_tokens`` is the prefix depth in tokens that stops being resumable
#: when that state goes -- the quantity #743 §1 calls the amplification, and
#: the reason a node id alone is not enough.
EvictedNode = Tuple[int, int]


def anchor_depth_tokens(node) -> int:
    """Token depth of ``node`` from the root, i.e. the resumable prefix it anchors.

    Walks parents rather than reading a cached depth because no node carries
    one, and a cached depth maintained at every split/insert would be a second
    invariant to keep -- the #743 instrument must not be able to make the tree
    wrong. Called ONLY on a call that is about to emit (see ``would_emit``), so
    the walk is off the hot path.

    The root contributes nothing: it holds no tokens, and counting it would
    make an anchor at depth 0 report a nonzero prefix.
    """
    depth = 0
    cur = node
    while cur is not None and getattr(cur, "parent", None) is not None:
        value = getattr(cur, "value", None)
        if value is not None:
            depth += len(value)
        cur = cur.parent
    return depth


@dataclass
class MambaSlotObserver:
    """Cadence, running totals and wording for the two #743 events.

    Held per cache instance. All timestamps are the caller's clock, in
    seconds; the observer never reads one itself so a test can drive it.
    """

    rate_per_s: float = DEFAULT_LOG_RATE_PER_S
    #: Bucket capacity. None -> ``max(rate_per_s, DEFAULT_LOG_BURST)``, so a
    #: boot that only turns the rate DOWN still keeps burst detail, and a boot
    #: that turns it up gets a capacity at least as large as its rate.
    burst: Optional[float] = None

    # -- running totals, cumulative for the life of the cache ---------------
    #: Successful ``evict_mamba`` calls that freed at least one slot.
    evictions: int = 0
    #: Slots freed by those calls.
    slots_evicted: int = 0
    #: Anchor tokens that stopped being resumable because of them. This is the
    #: number #743 wanted and could not get: the PREFIX cost of slot pressure,
    #: as distinct from the slot count.
    anchor_tokens_lost: int = 0
    #: Prefix matches cut short by a missing mamba state.
    truncations: int = 0
    #: Tokens the radix matched and the mamba state could not back.
    truncated_tokens: int = 0

    # -- bucket state -------------------------------------------------------
    _tokens: float = field(default=-1.0, repr=False)
    _last_refill: Optional[float] = field(default=None, repr=False)
    # -- suppression rollup -------------------------------------------------
    _held_events: int = field(default=0, repr=False)
    _held_slots: int = field(default=0, repr=False)
    _held_anchor_tokens: int = field(default=0, repr=False)
    _held_truncations: int = field(default=0, repr=False)
    _held_truncated_tokens: int = field(default=0, repr=False)
    _held_since: Optional[float] = field(default=None, repr=False)

    @property
    def enabled(self) -> bool:
        return self.rate_per_s > 0.0

    @property
    def capacity(self) -> float:
        if self.burst is not None:
            return max(0.0, float(self.burst))
        return max(self.rate_per_s, DEFAULT_LOG_BURST)

    # ------------------------------------------------------------------ bucket
    def _refill(self, now: float) -> None:
        if self._tokens < 0.0:
            # First sight: a full bucket, so the first events of a boot are
            # always emitted. An empty start would silently swallow exactly
            # the events an instrument exists to catch -- the first ones.
            self._tokens = self.capacity
            self._last_refill = now
            return
        last = self._last_refill if self._last_refill is not None else now
        elapsed = now - last
        if elapsed > 0.0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_s)
        self._last_refill = now

    def would_emit(self, now: float) -> bool:
        """Peek: will the next event get a line? Does NOT consume a token.

        The caller uses this to decide whether to pay for the per-node anchor
        depths, which cost a walk to the root each. A peek that lies in the
        pessimistic direction only costs a line's detail, never correctness.
        """
        if not self.enabled:
            return False
        self._refill(now)
        return self._tokens >= 1.0

    def _take(self, now: float) -> bool:
        if not self.enabled:
            return False
        self._refill(now)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _rollup(self, now: float) -> Optional[str]:
        """The line that repays the suppressed events, or None if there were none."""
        if self._held_events <= 0 and self._held_truncations <= 0:
            return None
        span = 0.0 if self._held_since is None else max(0.0, now - self._held_since)
        line = (
            f"{LOG_PREFIX} SUPPRESSED {self._held_events} eviction(s) "
            f"({self._held_slots} slot(s), {self._held_anchor_tokens} anchor tok) "
            f"and {self._held_truncations} truncation(s) "
            f"({self._held_truncated_tokens} tok) over {span:.1f}s -- the rate "
            f"limit hid the lines, not the arithmetic; the cumulative totals "
            f"on the next event include these"
        )
        self._held_events = 0
        self._held_slots = 0
        self._held_anchor_tokens = 0
        self._held_truncations = 0
        self._held_truncated_tokens = 0
        self._held_since = None
        return line

    def _hold(self, now: float) -> None:
        if self._held_since is None:
            self._held_since = now

    # ---------------------------------------------------------------- events
    def note_eviction(
        self,
        *,
        now: float,
        requested: int,
        evicted: int,
        nodes: Sequence[EvictedNode] = (),
        evictable: Optional[int] = None,
        protected: Optional[int] = None,
        available: Optional[int] = None,
        lineage: str = "device",
    ) -> List[str]:
        """Record one ``evict_mamba`` that freed slots; return lines to emit.

        ``protected`` is the observer's stand-in for "how many slots the
        RUNNING set holds": the cache has no scheduler handle, and
        ``mamba_protected_size()`` is exactly the states locked by running
        requests. Named as such in the line so nobody reads it as a pool
        constant.

        An eviction that freed NOTHING is not recorded -- it is the starvation
        case, which ``_log_mamba_slot_starvation`` already owns, and counting
        it here would inflate the very rate #743 wants measured.
        """
        if evicted <= 0:
            return []
        anchor_tokens = sum(int(t) for _, t in nodes)
        self.evictions += 1
        self.slots_evicted += int(evicted)
        self.anchor_tokens_lost += anchor_tokens
        if not self.enabled:
            return []
        if not self._take(now):
            self._held_events += 1
            self._held_slots += int(evicted)
            self._held_anchor_tokens += anchor_tokens
            self._hold(now)
            return []
        lines: List[str] = []
        rollup = self._rollup(now)
        if rollup is not None:
            lines.append(rollup)
        if nodes:
            detail = ", ".join(f"node {nid}@{tok}tok" for nid, tok in nodes)
        else:
            detail = "per-node detail not collected on this call"
        lines.append(
            f"{LOG_PREFIX} EVICT ({lineage}) freed {evicted} of {requested} "
            f"requested slot(s), dropping {anchor_tokens} anchor tok of "
            f"resumable prefix: {detail}. pool now "
            f"available={_num(available)} evictable={_num(evictable)} "
            f"held-by-running={_num(protected)}. cumulative: "
            f"{self.evictions} eviction(s), {self.slots_evicted} slot(s), "
            f"{self.anchor_tokens_lost} anchor tok lost"
        )
        return lines

    def note_truncation(
        self,
        *,
        now: float,
        rid: Optional[str],
        matched_tokens: int,
        usable_tokens: int,
        node_id: Optional[int] = None,
        lineage: str = "device",
    ) -> List[str]:
        """Record a prefix match cut short by a missing mamba state.

        THE DISTINCTION THIS RESTORES: ``matched_tokens`` is what the radix
        found, ``usable_tokens`` is what a surviving mamba anchor can back.
        The difference is recomputed from scratch by the model, and until now
        it looked in the logs exactly like a prefix that was never cached at
        all -- so a cache doing its job behind a slot pool that is too small
        read as a cache that was not working.
        """
        lost = int(matched_tokens) - int(usable_tokens)
        if lost <= 0:
            return []
        self.truncations += 1
        self.truncated_tokens += lost
        if not self.enabled:
            return []
        if not self._take(now):
            self._held_truncations += 1
            self._held_truncated_tokens += lost
            self._hold(now)
            return []
        lines: List[str] = []
        rollup = self._rollup(now)
        if rollup is not None:
            lines.append(rollup)
        lines.append(
            f"{LOG_PREFIX} TRUNCATED ({lineage}) rid={rid}: the radix matched "
            f"{matched_tokens} tok but only {usable_tokens} tok are backed by a "
            f"surviving mamba state, so {lost} tok are re-prefilled. This is "
            f"SLOT PRESSURE, not a cache miss -- the prefix was cached and its "
            f"state was evicted. anchor node={_num(node_id)}. cumulative: "
            f"{self.truncations} truncation(s), {self.truncated_tokens} tok"
        )
        return lines


def probe_available(allocator) -> Optional[int]:
    """Free slots in ``allocator``, or None if it could not be asked.

    ARMOURED, and not decoratively: this runs inside an eviction that the
    scheduler is waiting on, and an instrument that can raise is an
    instrument that takes the server down to report a statistic. None (not 0)
    on failure, so ``_num`` prints ``?`` and nobody reads a swallowed probe as
    an empty pool.
    """
    if allocator is None:
        return None
    try:
        return int(allocator.available_size())
    except Exception:  # noqa: BLE001 - an instrument never breaks the cache
        return None


def clock() -> float:
    """The instrument's clock, in one place.

    The observer itself never calls this -- every method takes ``now`` -- so
    the cadence and the rollup spans are testable against a fake clock. This
    exists only so the two call sites do not each pick a clock and drift.
    """
    import time

    return time.perf_counter()


def emit_lines(logger, lines: Sequence[str]) -> None:
    """Write what an observer returned. WARNING level on purpose: these lines
    describe cache capacity being destroyed, which is what the reader is
    looking for when serving got slower, and an INFO instrument is one
    ``--log-level`` away from not existing."""
    for line in lines:
        logger.warning("%s", line)


def _num(value: Optional[int]) -> str:
    """``?`` for a quantity this call could not measure, never a plausible 0.

    A swallowed probe that yields a legal number is the shape #698 and #714
    were both caused by: nothing downstream can tell "the pool holds none"
    from "the pool was never asked".
    """
    return "?" if value is None else str(int(value))


def observer_of(owner) -> MambaSlotObserver:
    """The cache's observer, created on first use.

    Lazily attached rather than constructed in ``__init__`` so a cache built
    by an older test -- or by a path that bypasses the constructor -- still
    gets one, matching the ``getattr(self, "_mamba_starvation_count", 0)``
    idiom already used for the starvation counter in the same file.
    """
    obs = getattr(owner, "_mamba_slot_observer", None)
    if obs is None:
        from sglang.srt.environ import envs

        try:
            rate = float(envs.SGLANG_MAMBA_SLOT_LOG_RATE.get())
        except Exception:  # noqa: BLE001 - an instrument never breaks the cache
            rate = DEFAULT_LOG_RATE_PER_S
        obs = MambaSlotObserver(rate_per_s=rate)
        owner._mamba_slot_observer = obs
    return obs
