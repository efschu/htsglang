"""#701 — chunked-prefill admission and the cross-pass commitment ledger.

Design: ``DESIGN_701_chunked_admission.md``. Reworked after the review gate
BLOCKED the first slice-3 premise; both of its central claims were verified
in-code before this rewrite:

* ``schedule_policy.py:1464`` already gates the FULL lifetime
  (``total_tokens >= self.rem_total_tokens -> NO_TOKEN``), so there is no
  missing full-length gate and the original "admitted on a 512-token check"
  story was wrong at first admission.
* ``schedule_policy.py:734-737`` documents the real suspect: ``rem_total_tokens``
  includes ``full_evictable_size()`` while the allocator can only recover
  MAMBA-recoverable bytes. Paper-evictable funds an admission the evictor
  cannot honour -- the gate passes and relief later frees 0, which is the
  specimen's signature.

What was genuinely missing is a RESERVATION across passes. ``PrefillAdder`` is
rebuilt every pass, so a resident chunked request's remaining PREFILL is
represented nowhere later; only remaining DECODE is reserved, and only for
requests that appear in ``running_batch.reqs`` at all -- which a
resident-but-batchless chunked request need not (#631 defect O). Later
admissions then spend the chunked request's committed future.

Generality clause: pool ARITHMETIC only. No rig threshold, no tuned fraction,
no hardware or model name, and deliberately NO chunk-size parameter -- the
chunk is what the original code substituted for the commitment.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PoolState:
    """Measured pool arithmetic at the admission decision point.

    ``recoverable_evictable_tokens`` is what eviction can ACTUALLY honour, not
    what it can count. On the hybrid-SSM path these differ: the caller must
    pass ``min(full_evictable, mamba_recoverable)``, never
    ``full_evictable_size()`` alone. The previous field name
    (``evictable_unlocked_tokens``) described only the locked-chain exclusion
    and would have funded the specimen through the new rule; it is removed
    rather than renamed so a stale caller fails loudly.

    ``permanent_reserve_tokens`` are offsets that never come back within a
    request's lifetime (mixed-decode seed, mamba gap reserves, per-layout
    arming floors). They set the ACHIEVABLE ceiling, which is below raw
    capacity -- see :func:`decide_chunked_admission`.

    ``spillable_tokens`` stays zero until a spill/retract path for a request's
    own prefix exists; it is present so that capability raises the fundable
    total without this rule being re-derived.
    """

    free_tokens: float
    recoverable_evictable_tokens: float
    locked_tokens: float
    total_capacity_tokens: float
    permanent_reserve_tokens: float = 0.0
    spillable_tokens: float = 0.0

    @property
    def fundable_tokens(self) -> float:
        return (
            float(self.free_tokens)
            + float(self.recoverable_evictable_tokens)
            + float(self.spillable_tokens)
        )

    @property
    def achievable_ceiling_tokens(self) -> float:
        """The most this pool can EVER fund for one request."""
        return float(self.total_capacity_tokens) - float(self.permanent_reserve_tokens)


@dataclasses.dataclass(frozen=True)
class AdmissionDecision:
    verdict: str  # "admit" | "defer" | "refuse"
    reason: str
    required_tokens: float
    fundable_tokens: float

    @property
    def admitted(self) -> bool:
        return self.verdict == "admit"


def decide_chunked_admission(
    remaining_tokens: int, pool: PoolState
) -> AdmissionDecision:
    """Decide admission from the request's FULL remaining length.

    * ``refuse`` -- above the ACHIEVABLE ceiling (capacity minus permanent
      reserves). Refusing at RAW capacity instead leaves a band in which a
      request defers on every pass forever; because any non-CONTINUE verdict
      breaks the FCFS admission loop, the whole queue then wedges behind it
      with no usage-1.00 tell -- the same syndrome this ticket exists to fix.
    * ``defer`` -- fits the ceiling but not today's fundable total.
    * ``admit`` -- fits now.
    """
    required = float(remaining_tokens)
    fundable = pool.fundable_tokens
    ceiling = pool.achievable_ceiling_tokens

    if required > ceiling:
        return AdmissionDecision(
            verdict="refuse",
            reason=(
                f"remaining length {required:,.0f} tokens exceeds the achievable "
                f"ceiling {ceiling:,.0f} (capacity {pool.total_capacity_tokens:,.0f} "
                f"minus {pool.permanent_reserve_tokens:,.0f} of permanent "
                "reserves). No future pass can fund it, so deferring would wedge "
                "the queue behind it silently."
            ),
            required_tokens=required,
            fundable_tokens=fundable,
        )

    if required > fundable:
        return AdmissionDecision(
            verdict="defer",
            reason=(
                f"remaining length {required:,.0f} tokens exceeds the fundable "
                f"{fundable:,.0f} (free {pool.free_tokens:,.0f} + RECOVERABLE "
                f"evictable {pool.recoverable_evictable_tokens:,.0f} + spillable "
                f"{pool.spillable_tokens:,.0f}); {pool.locked_tokens:,.0f} locked "
                "tokens cannot fund it. Deferring rather than admitting into a "
                "self-deadlock."
            ),
            required_tokens=required,
            fundable_tokens=fundable,
        )

    return AdmissionDecision(
        verdict="admit",
        reason=(
            f"remaining length {required:,.0f} tokens is fundable from {fundable:,.0f}."
        ),
        required_tokens=required,
        fundable_tokens=fundable,
    )


class ChunkedCommitmentLedger:
    """Cross-pass reservation for live chunked requests -- the actual fix.

    A chunked request's remaining PREFILL is a commitment against the pool that
    outlives the ``PrefillAdder`` which admitted it. This ledger is the only
    thing that carries it between passes, and it is deliberately keyed by
    request id rather than by membership in ``running_batch.reqs``, because a
    resident-but-batchless request is absent from that list (#631 defect O)
    while still holding its prefix.
    """

    #: Monotone across the ledger's life; never rewound by release(), because a
    #: progress counter that goes backwards reads as a restart to any watcher.
    committed_chunks: int

    def __init__(self) -> None:
        self.committed_chunks = 0
        self._outstanding: dict[str, int] = {}
        self._first_deferred_pass: dict[str, int] = {}
        self._last_deferred_pass: dict[str, int] = {}

    def commit(self, request_id: str, remaining_tokens: int) -> None:
        if request_id in self._outstanding:
            raise ValueError(
                f"request {request_id!r} already holds a commitment of "
                f"{self._outstanding[request_id]} tokens; committing again would "
                "double-count it. Use spend() per chunk and release() on "
                "finish/abort/retract."
            )
        if int(remaining_tokens) < 0:
            raise ValueError(f"negative commitment for {request_id!r}.")
        self._outstanding[request_id] = int(remaining_tokens)

    def spend(self, request_id: str, chunk_tokens: int) -> None:
        held = self._outstanding.get(request_id)
        if held is None:
            raise ValueError(f"request {request_id!r} holds no commitment to spend.")
        if int(chunk_tokens) > held:
            raise ValueError(
                f"request {request_id!r} would spend {int(chunk_tokens)} against a "
                f"remaining commitment of {held}. A chunk cannot exceed what is "
                "left; this means the commitment was mis-sized at admission."
            )
        self._outstanding[request_id] = held - int(chunk_tokens)
        # #699 ride-along: a MONOTONE count of chunks that actually committed.
        # The liveness detector needs this to separate a retry loop (batch
        # attempts advancing, nothing committing) from real progress. forward_ct
        # counts attempts and cannot tell those apart; this can, and it is one
        # line because spend() is already the single commit path.
        self.committed_chunks += 1

    def release(self, request_id: str) -> None:
        """On finish, abort or retract. Idempotent: a double release is not an
        error, because the three release paths are not mutually exclusive."""
        self._outstanding.pop(request_id, None)
        self._first_deferred_pass.pop(request_id, None)
        self._last_deferred_pass.pop(request_id, None)

    def outstanding_for(self, request_id: str) -> int:
        return int(self._outstanding.get(request_id, 0))

    def outstanding_tokens(self) -> int:
        return int(sum(self._outstanding.values()))

    def note_deferred(self, request_id: str, pass_index: int) -> None:
        """Telemetry for the defer band, so a wedge is observable rather than
        silent -- the failure mode defect 3 names."""
        self._first_deferred_pass.setdefault(request_id, int(pass_index))
        self._last_deferred_pass[request_id] = int(pass_index)

    def defer_age(self, request_id: str) -> int:
        first = self._first_deferred_pass.get(request_id)
        last = self._last_deferred_pass.get(request_id)
        if first is None or last is None:
            return 0
        return int(last) - int(first)


def effective_rem_total_tokens(
    rem_total_tokens: float, ledger: ChunkedCommitmentLedger | None
) -> float:
    """The budget a later pass may actually spend.

    ``PrefillAdder.rem_total_tokens`` is computed fresh each pass and knows
    nothing about commitments made in earlier ones. Subtracting the ledger is
    what stops a second request consuming a live chunked request's committed
    future -- the two-actor return of the deadlock.
    """
    if ledger is None:
        return float(rem_total_tokens)
    return float(rem_total_tokens) - float(ledger.outstanding_tokens())


def deferred_head_blocks_idle_flip(deferred_head_count: int) -> bool:
    """#701 defect 5: a deferred head is pending work, not idleness.

    ``effective_running_bs`` counts resident chunked requests but not
    deferred-waiting ones, so without this an idle/flip detector sees an idle
    instance while the head-of-line is deferred and the queue is blocked, and
    can park or flip an instance with work it will not serve.
    """
    return int(deferred_head_count) > 0


def effective_running_bs(running_bs: int, resident_chunked: int) -> int:
    """#631 defect O, as a counting truth.

    A chunked request is resident-but-batchless: it holds a locked prefix while
    appearing in no batch, so a raw ``running_bs`` reads 0 and every consumer
    concludes the instance is idle while it is wedged.
    """
    return int(running_bs) + int(resident_chunked)
