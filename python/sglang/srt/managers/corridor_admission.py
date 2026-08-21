# SPDX-License-Identifier: Apache-2.0
"""#656 item 15a at the PREFILL ADMISSION site (register C17).

WHY THIS MODULE EXISTS
----------------------
``CorridorGuard.ensure_headroom`` is item 15a's spill-before-alloc check, it
works, and until this module it was called from exactly ONE place: the flip
seam. Successor 33's acceptance run breached the corridor at a different
place, and the log says so in two lines:

    11:20:47Z  gpu0 = 1001 MiB free, and stays there for 12 samples (~1.6 s)
    11:20:49Z  CORRIDOR-GUARD cleared on device 0: want 681 MiB,
               free 1186 -> 2150, reclaimed 964 MiB from [allocator-cache]

The gate armed TWO SECONDS AFTER the dip and pulled the card back at once.
So the ladder is not weak and the seam did not breach: a 272k-token bs1
prefill walked the binding card under the floor at a site with no gate, and
the corridor law -- which is a CONTINUOUS minimum, broken by the trough and
not by the average -- was already broken by the time the seam looked.

This module is the second caller. It is deliberately small.

WHAT IT CHARGES FOR, AND WHY IT IS NOT THE KV BYTES
---------------------------------------------------
The obvious ``want`` for a prefill is "the KV this chunk will occupy". It is
the WRONG number here, and getting this backwards would make the gate arm on
every admission for memory that is never allocated.

In this fork the KV pool is a VA reservation whose physical pages are
committed once at ``KvVmmBufferOwner.finalize()`` after capture, and
``KvRowCap`` holds ``available_size()`` at or below
``full_pool_backed_rows``. ``alloc_extend`` therefore hands out slots that
are ALREADY COMMITTED: admitting a prefill moves no physical page. The two
things that do grow VRAM around a prefill are

  * the transient activation working set of the forward, linear in the chunk
    width, taken from torch's caching allocator, and
  * KV backing RE-COMMIT (``KvBackingRelief.recover``), which is already
    bounded against the law floor at ``kv_backing_relief.py:617``.

So this gate prices the first one and leaves the second to its own bound.

THE SLOPE IS A MOVEMENT PROXY, AND IS LABELLED AS ONE. The fork's only
per-token activation figures are the metrics reporter's
``_qkv_act_bytes_per_token`` / ``_ffn_act_bytes_per_token``, which are
movement bytes rather than a peak-residency model. They are used here
because they are MEASURED FROM THE MODEL'S OWN GEOMETRY and biased small,
and because the alternative -- a peak model invented at this call site --
is exactly the second-source-of-truth mistake register entry C10 is still
open about. An underestimate is absorbed by the gate's upper watermark
(``delta``, 256 MiB), which frees past the floor rather than to it. When the
slope cannot be read, ``want`` is 0 and the gate degrades to enforcing the
floor itself at admission time, which is still strictly more than the one
call site this chain shipped with.

IT SPILLS. IT NEVER REFUSES. THIS IS THE LOAD-BEARING DECISION
---------------------------------------------------------------
``ensure_headroom`` returns a verdict, and at the seam a false verdict means
ABANDON THE FLIP -- a decision every rank reaches unanimously, because the
seam already carries a reduction.

Prefill admission carries no such reduction, and the guard is RANK-LOCAL by
construction: it reads this rank's NVML free column and nothing else. A
refusal here would therefore be a rank-local answer to a question that
decides how much work the GROUP takes on, which is the precise shape that
desynced this instance before -- 449039/451037/175225/145734 rows across
ranks, followed by a scheduler that stopped heartbeating while every rank
was alive (HANDOFF_675 1a, and the reason
``collective_kv_backing_relief`` exists at all).

The user's own wording for item 15a is not "refuse", it is:

    "check AT the allocation (free-X >= 1024, otherwise spill synchronously
     first)"

-- check at the allocation, and otherwise SPILL FIRST, synchronously. That
is what this does. The verdict is recorded, logged and counted, and it does
not change whether the batch is admitted, so no rank can take a scheduling
decision its peers did not take. A shallow dip that the ladder could not
fund is a corridor problem; a rank-local admission split is a hang, and a
hang is strictly worse than the thing it would be protecting against.

If a successor wants a REFUSING prefill gate, the prerequisite is a
reduction on this path, not a change to this function.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from sglang.srt.managers.corridor_width import (
    MIN_CHUNK_TOKENS,
    fundable_chunk_tokens,
    width_was_cut,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#656 CORRIDOR-ADMISSION]"

#: Attribute the per-scheduler gate is memoised under.
ADMISSION_GATE_ATTR = "phase_flip_corridor_admission"

#: Minimum seconds between two RELIEF attempts on this path.
#:
#: The check is cheap (one ``cudaMemGetInfo``) and runs on every admission.
#: RELIEF is not: the first provider is ``torch.cuda.empty_cache()``, which
#: synchronises the device. A card parked just under the floor would otherwise
#: pay that on every scheduling iteration -- roughly 11 times a second on the
#: measured mix -- and the cure would cost more than the dip.
#:
#: THE COOLDOWN SUPPRESSES THE SPEND, NEVER THE CHECK, and it cannot make the
#: gate laxer: during a cooldown the gate still evaluates and still reports.
DEFAULT_ARM_COOLDOWN_S = 0.25

#: Upper bound on what one prefill chunk may be priced at, in MiB.
#:
#: This is a SAFETY BOUND on a proxy, not an estimate. The guard frees towards
#: ``floor + delta + want``; a ``want`` bigger than the card can hold makes
#: that target unreachable, and an unreachable target spends every provider on
#: every call. 256 MiB is the guard's own delta -- i.e. one watermark's worth
#: of slack -- which is enough to fund a chunk's transient and small enough
#: that mispricing costs a little extra headroom instead of the drafter.
WANT_CAP_MIB = 256


def _activation_bytes_per_token(scheduler: Any) -> int:
    """Per-token transient activation bytes for ONE layer, or 0 if unreadable.

    Read from the metrics reporter's own geometry rather than recomputed: a
    second derivation of the same quantity is a second thing to keep in
    agreement, and this chain has shipped that defect twice.

    TWO CORRECTIONS THAT A REVIEW CAUGHT BEFORE THEY REACHED A RUN, both of
    which matter more than the number itself:

    1. **The reporter's figures are summed over EVERY layer** (each term in
       `metrics_reporter._init_estimated_perf_constants` carries a
       `* num_layers`). Transient activation residency is not: the buffers
       are reused down the stack, so the resident peak is about ONE layer's
       working set. Taking the sum would have priced a 512-token chunk at
       ~4.7 GiB on this model, which is larger than any card's free column --
       the gate would then have armed on EVERY admission, found its target
       unreachable every time, and spent the whole ladder each round,
       including the rebalance provider that evacuates the drafter and with
       it speculative decoding. An overstatement here is not conservative;
       it is a self-inflicted outage.
    2. **The figures only EXIST under `--enable-mfu-metrics`**, which is not
       set on this rig's boot. So 0 is the normal answer, not the corner
       case, and the gate must degrade to plain floor enforcement without
       pretending otherwise. The announcement line reports the slope it
       resolved precisely so this is visible rather than inferred.

    The result is still only a movement proxy. It is bounded at the call site
    (see `WANT_CAP_MIB`) so that no error in it can destabilise the ladder.
    """
    reporter = getattr(scheduler, "metrics_reporter", None)
    if reporter is None:
        return 0
    total = 0
    for name in ("_qkv_act_bytes_per_token", "_ffn_act_bytes_per_token"):
        try:
            total += float(getattr(reporter, name, 0) or 0)
        except (TypeError, ValueError):
            return 0
    layers = 0
    for owner in (getattr(scheduler, "model_config", None), reporter):
        for name in ("num_hidden_layers", "num_layers", "_num_layers"):
            try:
                layers = int(getattr(owner, name, 0) or 0)
            except (TypeError, ValueError):
                layers = 0
            if layers > 0:
                break
        if layers > 0:
            break
    if layers <= 0:
        # An unknown layer count means the per-layer share cannot be taken,
        # and the SUM is known to be wrong by that factor. Refuse to price
        # rather than price it wrongly: want=0 still enforces the floor.
        return 0
    return max(0, int(total // layers))


#: The forward-peak probe's name for a prefill forward. Its rows are keyed by
#: phase, and "extend" is what the model runner passes for every non-decode
#: batch (``model_runner.py``, the ``peak_probe.begin`` call).
PREFILL_PHASE = "extend"


def _measured_bytes_per_token(scheduler: Any, tokens: int) -> Optional[float]:
    """The MEASURED per-token transient for a chunk this size, or None.

    Read from ``ForwardPeakTracker``, which brackets every forward with
    ``reset_peak_memory_stats`` / ``max_memory_allocated`` and now records the
    peak MINUS the baseline -- i.e. what the forward ADDED, which is the only
    part an admission may be charged for.

    WHY THIS IS BETTER THAN THE GEOMETRY SLOPE and not merely different: the
    geometry slope is assembled from the metrics reporter's per-token terms,
    exists only under ``--enable-mfu-metrics``, and is a movement proxy that
    was wrong by the layer count until a review caught it. The tracker's figure
    is what this rank's allocator actually reached on this bucket, on this
    model, at this chunk size. HANDOFF_678 §4.1 asks for a peak-residency
    figure so the gate can PREEMPT a crossing rather than recover from it; this
    is that figure.

    None when the probe is off (it is off unless ``SGLANG_FORWARD_PEAK_PATH``
    is set), when the bucket has not been measured, or when anything at all
    goes wrong. None means NOT PRICED and the caller falls back -- never a
    substituted literal, which is the rule ``measured_capture_mib_per_token``
    exists to enforce and which this path inherits.
    """
    try:
        runner = getattr(getattr(scheduler, "tp_worker", None), "model_runner", None)
        tracker = getattr(runner, "_forward_peak", None)
        if tracker is None:
            return None
        value = tracker.transient_bytes_per_token(PREFILL_PHASE, int(tokens))
    except Exception:  # noqa: BLE001 -- a probe must never break admission
        return None
    if value is None or value <= 0:
        return None
    return float(value)


class PrefillAdmissionGate:
    """Item 15a's spill-before-alloc, at the prefill admission site.

    One instance per scheduler, memoised, because the guard it spends through
    is itself memoised and holds providers whose host images must survive
    every flip.
    """

    def __init__(
        self,
        scheduler: Any,
        *,
        cooldown_s: float = DEFAULT_ARM_COOLDOWN_S,
        clock=time.monotonic,
    ) -> None:
        self._scheduler = scheduler
        self._cooldown_s = float(cooldown_s)
        self._clock = clock
        self._last_arm = float("-inf")
        #: Item 16's lender: None = not resolved yet, False = this rank has
        #: none (disabled, or its NVML column could not be proven).
        self._lender = None
        self._slope_logged = False
        self._measured_logged = False
        self._announced = False
        # Counters are the evidence this gate ever did anything. A mechanism
        # that cannot be shown to have fired is indistinguishable from one
        # that is never reached, and this chain has shipped several.
        self.checks = 0
        self.armed = 0
        self.cleared = 0
        self.short = 0
        self.cooldown_skips = 0
        self.reclaimed_bytes = 0
        # #794: the ACTUATOR's own evidence. `short` counts a corridor the
        # ladder could not restore; `cuts` counts the times something was done
        # about it. The 2026-08-21 crash is precisely the state where the first
        # counter moves and the second cannot.
        self.cuts = 0
        self.tokens_withheld = 0
        self._last_cut_logged = None
        self._unpriced_logged = False

    # -- the gate ---------------------------------------------------------

    def want_bytes(self, tokens: int) -> int:
        """Transient bytes the forward for ``tokens`` is expected to take.

        HARD-CAPPED at ``WANT_CAP_MIB``. The cap is the safety property, not
        a tuning knob: ``ensure_headroom`` frees towards
        ``floor + delta + want``, so a ``want`` larger than the card can ever
        hold makes that target unreachable, and an unreachable target spends
        EVERY provider on EVERY call -- up to and including evacuating the
        drafter and killing speculative decoding. With the cap, the worst a
        wrong slope can do is free a little more than necessary.
        """
        tokens = max(0, int(tokens or 0))
        cap = WANT_CAP_MIB * (1024 * 1024)
        measured = (
            _measured_bytes_per_token(self._scheduler, tokens) if tokens else None
        )
        if measured is not None:
            raw = int(tokens * measured)
            # NET OF THE ALLOCATOR CACHE, and only on this branch.
            #
            # The measured figure is ALLOCATOR-SIDE: it is torch's peak
            # allocated bytes. The corridor is DRIVER-SIDE: NVML's free column.
            # Those two move together only when the allocator has to grow its
            # reservation -- bytes served out of its own cache are already
            # reserved and already absent from the free column, so charging
            # them again arms the gate for an allocation the driver never sees.
            # This is the same subtraction the KV rung makes with
            # ``cheap_relief_bytes``, for the same tier-law reason.
            #
            # The geometry branch below does NOT subtract it, and that is not
            # an oversight: that slope is a movement proxy for driver-side
            # growth, already biased small, and netting it against the cache
            # would zero it on every admission -- removing what little an
            # underpriced gate can still do (§1a-bis).
            want = max(0, raw - self._allocator_cache_bytes())
            if not self._measured_logged:
                self._measured_logged = True
                logger.info(
                    "%s pricing prefill admission from a MEASURED transient: "
                    "%.0f bytes/token on the %s/%d bucket, net of the "
                    "allocator cache. This figure is the forward-peak probe's "
                    "peak-minus-baseline, not the metrics-reporter geometry.",
                    LOG_PREFIX,
                    measured,
                    PREFILL_PHASE,
                    tokens,
                )
            return min(want, cap)
        slope = _activation_bytes_per_token(self._scheduler)
        if slope > 0 and not self._slope_logged:
            self._slope_logged = True
            logger.info(
                "%s pricing prefill admission at %d bytes/token of transient "
                "activation (movement proxy from the model geometry; KV rows "
                "are pre-committed and are deliberately NOT charged here)",
                LOG_PREFIX,
                slope,
            )
        return min(tokens * slope, cap)

    # -- #794: the actuator ------------------------------------------------

    def _config_transient_bytes(self, tokens: int) -> Optional[float]:
        """The GDN prefill transient for ``tokens``, from the model config.

        THE ONLY PRICE THAT EXISTS ON A DEFAULT BOOT. The measured probe needs
        ``SGLANG_FORWARD_PEAK_PATH`` and the geometry proxy needs
        ``--enable-mfu-metrics``; the rig's own boot log answers "pricing
        source NONE -- want is 0" on all three ranks, which is why the gate
        that measured the 2026-08-21 OOM to the MiB was pricing the chunk it
        died on at zero. ``ServerArgs.gdn_prefill_scratch_mib`` is derived from
        the allocation sites of the chunked gated-delta-rule prefill and needs
        nothing but the checkpoint config, so it is always available.

        It is an UPPER BOUND on residency (every intermediate of the layer
        counted alive at once), which is the right direction for a width
        decision and the wrong magnitude for the relief ladder -- hence it
        prices the cut here and never ``want``.
        """
        server_args = getattr(self._scheduler, "server_args", None)
        estimator = getattr(server_args, "gdn_prefill_scratch_mib", None)
        if estimator is None:
            return None
        try:
            mib = estimator(self._head_share(), int(tokens))
        except Exception:  # noqa: BLE001 -- an estimator must never break admission
            return None
        if mib is None or mib <= 0:
            return None
        return float(mib) * (1024 * 1024)

    def _head_share(self) -> float:
        """This rank's share of the sharded head dimension, 1.0 when unsharded.

        Under pure PP (this rig: ``--tp-size 1 --pp-size 3``) no head is split,
        so every rank carries the full width and the share is 1. Read from the
        parallel state rather than assumed, because the flip's TP phase does
        shard it.
        """
        try:
            tp_size = int(getattr(self._scheduler.server_args, "tp_size", 1) or 1)
        except Exception:  # noqa: BLE001
            return 1.0
        return 1.0 / max(1, tp_size)

    def _measured_calibration(self, tokens: int) -> float:
        """Scale the config upper bound onto what the forward MEASURABLY took.

        The two prices answer different questions. ``_config_transient_bytes``
        counts every intermediate of a GDN layer as simultaneously resident;
        the probe reports ``max_memory_allocated`` minus the pre-forward
        baseline, i.e. what the allocator actually grew by. The first is an
        upper bound on the second, and on this rig the gap is large enough to
        matter: a 4096-token chunk prices at ~635 MiB of config transient on a
        card whose free column sits near 350 MiB, and the instance serves that
        chunk all day -- so the bound is not what the driver is asked for.

        Where the probe has measured THIS width's bucket, its magnitude is
        therefore used and the config formula supplies only the SHAPE (the
        curvature a narrower width has, including the ``ceil(T/64)`` state
        term the probe cannot interpolate across buckets). Where it has not,
        the scale is 1.0 and the bound prices the cut unmodified -- earlier
        cuts, never later ones, which is the safe direction for a mechanism
        whose failure mode is a dead worker.

        Clamped to 1.0: a measurement ABOVE the bound means the bound is not
        one, and trusting it to raise the price would let a probe artefact
        widen the chunk past what the config says is possible.
        """
        measured = _measured_bytes_per_token(self._scheduler, tokens)
        if measured is None:
            return 1.0
        config = self._config_transient_bytes(tokens)
        if config is None or config <= 0:
            return 1.0
        scale = (float(measured) * float(tokens)) / config
        if not (scale > 0):
            return 1.0
        return min(1.0, scale)

    def transient_price(self, requested_tokens: int):
        """A price callable for widths <= ``requested_tokens``, or None.

        None means NOT PRICED, and an unpriced chunk is never cut.
        """
        if self._config_transient_bytes(int(requested_tokens)) is None:
            return None
        scale = self._measured_calibration(int(requested_tokens))

        def price(width: int) -> Optional[float]:
            raw = self._config_transient_bytes(int(width))
            return None if raw is None else raw * scale

        return price

    def spendable_bytes(self, guard) -> Optional[float]:
        """What the forward may grow by without crossing into an OOM.

        THE LINE IS THE OOM LINE, NOT THE CORRIDOR LAW, and that distinction is
        a user decision this actuator may not quietly reverse. The 1024 MiB
        corridor law is a FILL-QUALITY target: "it may not block, delay or
        refuse anything on its own" (corridor_guard.py:949-953, 2026-08-16,
        after a law-gated seam refused 76 times in a row while 727004 tokens
        waited on an idle GPU). Sizing the cut against the law would make the
        law block prefill, which is exactly what that decision forbids -- and
        on this rig, where the free column sits far below 1024 MiB at
        admission time as a matter of course, it would narrow every chunk to
        the floor and stall the instance.

        What stays hard is the thing that was ever unsurvivable: an allocation
        larger than what can be served. So the budget is the driver's free
        column plus the allocator's own cache (bytes already reserved from the
        driver, which a forward can take without the free column moving),
        minus the guard's existing ``delta`` watermark as the margin for what
        this estimate does not model -- fragmentation, and the concurrent
        allocations of the rest of the round.
        """
        try:
            free = float(guard.free_bytes())
        except Exception:  # noqa: BLE001
            return None
        delta = float(getattr(guard, "delta_mib", 0) or 0) * (1024 * 1024)
        return free + float(self._allocator_cache_bytes()) - delta

    def granted_width(self, requested_tokens: int) -> int:
        """The width this rank's corridor can fund, <= ``requested_tokens``.

        Called AFTER ``before_admission``, deliberately: the relief ladder runs
        first, so the actuator narrows only what relief could not fund, and the
        free column it reads is the post-relief one.
        """
        requested = int(requested_tokens or 0)
        if requested <= 0:
            return requested
        guard = self._guard()
        if guard is None:
            return requested
        budget = self.spendable_bytes(guard)
        if budget is None:
            return requested
        price = self.transient_price(requested)
        if price is None:
            if not self._unpriced_logged:
                self._unpriced_logged = True
                logger.warning(
                    "%s the prefill width is NOT PRICED on this rank (no GDN "
                    "layers in the config, or the config is unreadable), so "
                    "the corridor cannot narrow a chunk it cannot cost. The "
                    "gate keeps measuring and the ladder keeps spending; the "
                    "width is whatever was configured.",
                    LOG_PREFIX,
                )
            return requested
        granted = fundable_chunk_tokens(
            requested_tokens=requested,
            budget_bytes=budget,
            price_bytes=price,
            # FLA_CHUNK_SIZE. The GDN prefill walks the chunk in 64-token
            # blocks, so a width off that grid buys a partial block at the
            # price of a whole one.
            granularity_tokens=64,
            min_tokens=MIN_CHUNK_TOKENS,
        )
        if not width_was_cut(requested, granted):
            return requested
        self.cuts += 1
        self.tokens_withheld += requested - granted
        if granted != self._last_cut_logged:
            self._last_cut_logged = granted
            logger.warning(
                "%s NARROWED this prefill chunk from %d to %d tokens: the "
                "card can fund %.0f MiB of forward transient after relief and "
                "the full width prices at %.0f MiB. The tokens are not "
                "refused -- they are admitted over more passes. This is the "
                "actuator the 2026-08-21 OOM proved missing, when the same "
                "shortfall was measured, logged and then run into.",
                LOG_PREFIX,
                requested,
                granted,
                budget / (1024 * 1024),
                (price(requested) or 0) / (1024 * 1024),
            )
        return granted

    def _price_source(self) -> str:
        """Which estimator is pricing this gate, for the announcement.

        The two differ by orders of magnitude and by kind -- one is a measured
        peak, the other a geometry proxy -- so a bytes/token figure in a log
        that does not say which one it is cannot be checked by the next reader.
        This chain lost an acceptance run to exactly that ambiguity.
        """
        measured = _measured_bytes_per_token(self._scheduler, 512)
        if measured is not None:
            return f"MEASURED forward-peak transient (~{measured:.0f} B/token at 512)"
        if _activation_bytes_per_token(self._scheduler) > 0:
            return "metrics-reporter geometry (a movement proxy)"
        return (
            "NONE -- want is 0 and the gate enforces the floor without "
            "preempting (set SGLANG_FORWARD_PEAK_PATH for a measured price)"
        )

    def _allocator_cache_bytes(self) -> int:
        """Bytes torch holds reserved but not allocated: the cheap tier.

        Deliberately the same quantity the KV rung calls ``cheap_relief`` --
        ``reserved - allocated``. It overstates what ``empty_cache`` could
        return, because it counts intra-segment fragmentation, and that is the
        safe direction here: overstating the cache understates ``want``, and an
        understated want costs a late arm, while an overstated one spends the
        ladder for memory the driver was never going to be asked for.
        """
        try:
            import torch

            return max(
                0,
                int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated()),
            )
        except Exception:  # noqa: BLE001
            return 0

    def before_admission(self, tokens: int) -> Optional[Any]:
        """Make room for a ``tokens``-wide prefill chunk BEFORE it is built.

        Returns the guard's verdict, or None when there is nothing to guard.
        THE RETURN VALUE IS EVIDENCE, NOT A DECISION: see the module
        docstring. Callers must admit the batch either way.
        """
        guard = self._guard()
        self._announce_once(guard)
        if guard is None:
            return None
        # SPEC ITEM 16, FIRST RELIEF STAGE, BEFORE THE GATE IS EVEN PRICED.
        #
        # The lender is consulted on EVERY admission, not only the ones that
        # arm: its whole purpose is to act before the trough, and the trough
        # by definition happens between arms. It has its own (wider) pressure
        # band and its own rate limiter, so the common path costs one
        # monotonic clock read. Running it FIRST also means the gate below
        # prices its want against a card the lender has already levelled --
        # the two must not compete for the same bytes in the same round.
        self._maybe_lend(guard, tokens)
        self.checks += 1
        want = self.want_bytes(tokens)
        try:
            free = guard.free_bytes()
        except Exception as e:  # noqa: BLE001
            # A probe that cannot read the card must not take prefill down.
            logger.warning("%s free probe failed: %s", LOG_PREFIX, e)
            return None
        if free - want >= guard.floor_bytes:
            return None
        now = self._clock()
        if now - self._last_arm < self._cooldown_s:
            self.cooldown_skips += 1
            return None
        self._last_arm = now
        self.armed += 1
        try:
            verdict = guard.ensure_headroom(
                want,
                reason=f"prefill admission, {int(tokens)} tokens",
                # NEVER fatal here. ``refusal_is_fatal`` opens the HOST tier
                # against item 16's levelling preference, and it is justified
                # at the pp->tp seam because a refusal there starves decode
                # outright. A prefill chunk has no such claim: it can be built
                # one round later, so it must not be the reason host RAM is
                # spent while a peer card still has headroom.
                refusal_is_fatal=False,
            )
        except Exception as e:  # noqa: BLE001
            # The gate is a safety net. A net that tears must not take down
            # the thing it was protecting.
            logger.error("%s gate failed to evaluate: %s", LOG_PREFIX, e)
            return None
        self.reclaimed_bytes += int(getattr(verdict, "reclaimed", 0) or 0)
        # THE DIP IS THE EVENT HERE, not the verdict. This gate never refused
        # anything -- it admits regardless and returns evidence -- so its
        # shortfall counter is a FILL-QUALITY signal. Since the corridor law
        # stopped gating (user decision 2026-08-16) a dip comes back as
        # ok=True with `law_breached` set, and a counter keyed on `ok` alone
        # would have quietly reported a perfect corridor forever.
        if verdict.ok and not getattr(verdict, "law_breached", False):
            self.cleared += 1
            logger.info("%s cleared before prefill: %s", LOG_PREFIX, verdict.detail)
        else:
            self.short += 1
            logger.error(
                "%s SHORT before prefill: %s. Every provider is exhausted, so "
                "the corridor cannot be restored ahead of this chunk. The "
                "batch is still admitted -- a rank-local refusal here would "
                "split the group's admission decision, which is a hang, and a "
                "hang is worse than the dip it would prevent. This counter is "
                "the cost of the missing reduction on this path.",
                LOG_PREFIX,
                verdict.detail,
            )
        return verdict

    def _announce_once(self, guard) -> None:
        """Say, ONCE, that this gate is installed and what it can see.

        WHY THIS IS NOT OPTIONAL NOISE. Every other line this module can emit
        is conditional on the gate ARMING. A gate that is installed but whose
        guard lookup returns None therefore logs exactly nothing -- and so
        does a gate that is working perfectly on a card that never approaches
        the floor. Those two states are opposite in value and identical in
        the log, which is the failure this very module's docstring accuses
        the KV rung of, and it caught this shift out on its own first run:
        6066 prefill admissions, zero lines, and no way to tell whether the
        mechanism was silent or absent.

        So the first admission reports what the gate resolved, at INFO when
        it is armed and at WARNING when it is inert. One line per process.
        """
        if self._announced:
            return
        self._announced = True
        if guard is None:
            logger.warning(
                "%s INERT: no corridor guard is reachable from this "
                "scheduler, so prefill admissions are NOT gated on this rank. "
                "The corridor law is then enforced at the flip seam only, "
                "which is the state register C17 describes.",
                LOG_PREFIX,
            )
            return
        logger.info(
            "%s ARMED on device %s: arming floor %d MiB, law floor %d MiB, "
            "delta %d MiB, activation slope %d bytes/token, providers %s, "
            "pricing source %s",
            LOG_PREFIX,
            getattr(guard, "device_index", "?"),
            getattr(guard, "floor_mib", -1),
            getattr(guard, "law_floor_mib", -1),
            getattr(guard, "delta_mib", -1),
            _activation_bytes_per_token(self._scheduler),
            list(getattr(guard, "providers", ())) or "[] (nothing to spend yet)",
            self._price_source(),
        )

    # -- internals --------------------------------------------------------

    def _maybe_lend(self, guard, tokens: int) -> None:
        """Consult item 16's rebalance lender, if this rank has one.

        Resolved lazily and cached, for the same reason the guard is: this
        object is built before the flip runtime has finished wiring, and a
        None captured at construction would be a lender that never runs while
        looking exactly like a lender that never needed to.

        Failures are swallowed on purpose. A levelling optimisation that can
        take prefill down is a worse bargain than an unlevel column.
        """
        if self._lender is not None:
            # An injected lender (tests) keeps the direct path.
            try:
                self._lender.maybe_lend(f"prefill admission, {int(tokens)} tokens")
            except Exception as e:  # noqa: BLE001
                logger.warning("%s rebalance lender raised: %s", LOG_PREFIX, e)
            return
        from sglang.srt.managers.corridor_rebalance import lend_for_guard

        lend_for_guard(guard, f"prefill admission, {int(tokens)} tokens")

    def _guard(self):
        try:
            from sglang.srt.managers.phase_flip_spill import get_corridor_guard

            guard = get_corridor_guard(self._scheduler)
            if guard is not None:
                return guard
            # FALLBACK: the flip runtime holds its own handle to "the
            # scheduler" (``_census_scheduler``) and caches the guard on THAT
            # object. If the two are ever not the same object, the builder
            # above returns None here while a perfectly good guard exists one
            # reference away -- and the gate would be silently inert, which is
            # exactly what this module accuses other mechanisms of being.
            # Cheap to check, and it names the hazard rather than assuming it
            # cannot happen.
            return self._guard_via_flip_runtime()
        except Exception as e:  # noqa: BLE001
            logger.warning("%s corridor guard unavailable: %s", LOG_PREFIX, e)
            return None

    def _guard_via_flip_runtime(self):
        from sglang.srt.managers.phase_flip_spill import CORRIDOR_GUARD_ATTR

        runtime = getattr(self._scheduler, "phase_flip_runtime", None)
        census = getattr(runtime, "_census_scheduler", None) if runtime else None
        if census is None or census is self._scheduler:
            return None
        guard = getattr(census, CORRIDOR_GUARD_ATTR, None)
        if guard is not None:
            logger.warning(
                "%s the corridor guard was reachable only through the flip "
                "runtime's scheduler handle, not through the scheduler this "
                "gate was built on. The two are different objects; using the "
                "existing guard rather than building a second one.",
                LOG_PREFIX,
            )
        return guard

    def stats(self) -> dict:
        return {
            "checks": self.checks,
            "armed": self.armed,
            "cleared": self.cleared,
            "short": self.short,
            "cooldown_skips": self.cooldown_skips,
            "reclaimed_mib": int(self.reclaimed_bytes // (1024 * 1024)),
        }


def get_prefill_admission_gate(scheduler: Any) -> Optional[PrefillAdmissionGate]:
    """The scheduler's memoised admission gate, built on first use."""
    if scheduler is None:
        return None
    gate = getattr(scheduler, ADMISSION_GATE_ATTR, None)
    if gate is None:
        gate = PrefillAdmissionGate(scheduler)
        setattr(scheduler, ADMISSION_GATE_ATTR, gate)
    return gate


def guard_prefill_admission(scheduler: Any, tokens: int) -> Optional[Any]:
    """Module-level entry point for the scheduler's one call.

    OFF UNLESS THE PHASE FLIP IS ON, and that gate is deliberate. Before this
    module, ``get_corridor_guard`` had exactly one caller and it lived on the
    flip seam, so the guard -- with its provider ladder, its NVML fleet probe
    and its ``KvBackingRelief`` -- was only ever constructed on a phase-flip
    boot. Calling it unconditionally from the prefill path would build all of
    that on EVERY boot of this fork, which is new behaviour on the default
    path for users who never asked for #656. The corridor law is a property
    of this feature's regime; it does not get to change everyone else's
    allocator behaviour as a side effect.
    """
    server_args = getattr(scheduler, "server_args", None)
    if not getattr(server_args, "enable_phase_flip", False):
        return None
    gate = get_prefill_admission_gate(scheduler)
    if gate is None:
        return None
    return gate.before_admission(tokens)
