# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding data model for the #124 harness (#143 follow-up).

A speculative run does not produce one logits row per emitted token the way
:class:`Trajectory` assumes. It produces, per *verify round*, a ``[D, V]``
target logits matrix with ``D = k + 1`` rows -- one per draft slot -- of
which the first ``accept_len`` rows are the parents of the tokens the round
actually commits.

This module carries that shape (:class:`VerifyRound`, :class:`SpecRun`) and
PROJECTS it back into emitted-token space (:meth:`SpecRun.to_trajectory`).
After the projection a spec run is the same object as a non-spec run and
every existing #124 primitive applies verbatim -- there is deliberately no
parallel comparison machinery for speculation.

Why the projection is exactly ``logits[i] -> emitted[i]``
--------------------------------------------------------
For chain speculation (``--speculative-eagle-topk 1``, the only mode the
weightless lane admits) the greedy verify kernel walks slot 0, 1, 2, ...
and writes ``predicts[slot_i] = target_predict[slot_i]`` for every accepted
step plus the trailing bonus (``verify_tree_greedy_kernel_triton``, and the
identical sgl_kernel op). So the round's emitted sequence is

    target_predict[slot_0], target_predict[slot_1], ..., target_predict[slot_{n-1}]

with ``n = accept_len``, and ``target_predict = argmax(next_token_logits)``
(``eagle_utils.py:1029``). Row ``i`` of the verify matrix is therefore the
parent of emitted token ``i`` of that round.

What the projection deliberately does NOT hide
----------------------------------------------
Token-index alignment across two arms survives different accept lengths --
that is the point -- but the two arms' rows for the same token index may
come from different *slot positions* within a round once their accept
lengths diverge (token 5 can be row 2 of round 1 in one arm and row 0 of
round 2 in the other). That is itself a different forward shape, hence a
different fp reduction order. The consequence is stated once, in
:data:`~.classes.ByteIdentityClass.SPEC_NEAR_TIE`: the tolerated relation
between matched-spec arms is near-tie-gated, never "0 flips".

The first token of a request comes from the PREFILL forward, not from a
verify round, in both a spec and a non-spec arm. It is not part of a
``SpecRun``; the runner records it separately (this is why Window 5's
lane-vs-plain comparison saw a common prefix of exactly 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import torch

from .trajectory import Trajectory

__all__ = ["VerifyRound", "SpecRun"]


@dataclass
class VerifyRound:
    """One target-verify step of a chain-speculative round.

    ``logits``     -- ``[D, V]`` target next-token logits, D = draft_token_num
                      (= k + 1), captured BEFORE any sampling preprocessing
                      (the byte-identity classes are dtype-strict).
    ``candidates`` -- ``[D]`` the draft tokens fed into the round;
                      ``candidates[0]`` is the already-committed root and
                      ``candidates[1:]`` the k proposals.
    ``emitted``    -- the tokens this round committed, in order. Its length is
                      the round's ``accept_len``, in ``1..D``.
    """

    logits: torch.Tensor
    candidates: Sequence[int]
    emitted: Sequence[int]

    def __post_init__(self) -> None:
        if self.logits.dim() != 2:
            raise ValueError(
                f"verify logits must be [D, V], got {tuple(self.logits.shape)}"
            )
        d = int(self.logits.shape[0])
        if len(self.candidates) != d:
            raise ValueError(
                f"candidates length {len(self.candidates)} != verify rows {d}"
            )
        if not self.emitted:
            raise ValueError(
                "a verify round always commits at least the bonus token; "
                "accept_len must be >= 1"
            )
        if len(self.emitted) > d:
            raise ValueError(
                f"accept_len {len(self.emitted)} exceeds the round's "
                f"{d} verify rows -- a round cannot commit more tokens than "
                "it scored"
            )

    @property
    def accept_len(self) -> int:
        return len(self.emitted)

    @property
    def draft_token_num(self) -> int:
        return int(self.logits.shape[0])

    @classmethod
    def from_dump_record(cls, record: dict, request: int = 0) -> "VerifyRound":
        """Build a round from one record written by
        ``sglang.srt.speculative.spec_verify_dump``.

        The rows are REORDERED into emitted order via the record's
        ``accepted_rows`` (which come from ``accept_index``, i.e. GLOBAL flat
        row indices) rather than sliced positionally. For a chain those happen
        to be ``b*D, b*D+1, ...``, but relying on that would silently produce a
        plausible-looking wrong trajectory the day a tree layout is dumped --
        and "plausible but wrong" is the failure mode this whole harness
        exists to make impossible.

        Only the ACCEPTED rows survive into the round. The record keeps the
        full matrix for accept-length forensics; the oracle only ever compares
        rows that produced a token.
        """
        if record.get("mode") != "target_verify":
            raise ValueError(
                f"expected a target_verify record, got mode " f"{record.get('mode')!r}"
            )
        rows = record["accepted_rows"][request]
        logits = record["logits"]
        return cls(
            logits=torch.stack([logits[r] for r in rows]),
            # Candidates are carried per surviving row, so the round stays
            # self-consistent (candidates width == logits rows).
            candidates=[int(c) for c in record["candidates"][request][: len(rows)]],
            emitted=[int(t) for t in record["emitted"][request]],
        )


@dataclass
class SpecRun:
    """A speculative generation captured as its sequence of verify rounds."""

    rounds: List[VerifyRound]
    seed: int
    label: str = ""
    #: Optional per-round provenance (boot id, request id, ...) for gate logs.
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rounds)

    def accept_lengths(self) -> List[int]:
        return [r.accept_len for r in self.rounds]

    def mean_accept_length(self) -> float:
        """The Gate-2 instrument. Read this, never ``spec_ema_accept_len``
        (memory ``spec-acceptance-messfalle``); server-side the equivalent is
        ``meta_info.spec_accept_length``."""
        if not self.rounds:
            return 0.0
        lens = self.accept_lengths()
        return sum(lens) / len(lens)

    def emitted_tokens(self) -> List[int]:
        return [int(t) for r in self.rounds for t in r.emitted]

    @classmethod
    def from_dump_records(
        cls,
        records: Sequence[dict],
        seed: int,
        request: int = 0,
        label: str = "",
    ) -> "SpecRun":
        """Assemble a run from dump records, ORDERED BY THEIR OWN ``step``.

        Never by directory listing or by argument order: the writer names files
        ``rank{r}_verify{step:07d}.pt`` and a reader that trusts glob order
        would silently reorder the trajectory on any filesystem that does not
        sort, which reads as a token flip rather than as a reader bug.
        """
        ordered = sorted(records, key=lambda r: int(r["step"]))
        rounds = [VerifyRound.from_dump_record(r, request=request) for r in ordered]
        return cls(
            rounds=rounds,
            seed=seed,
            label=label,
            meta={"steps": [int(r["step"]) for r in ordered], "request": request},
        )

    def to_trajectory(self) -> Trajectory:
        """Project into emitted-token space: one logits row per emitted token.

        Round boundaries and accept lengths vanish here BY DESIGN -- they must
        not be an index in the oracle, or two arms with different accept
        lengths would be incomparable even when they emit the same tokens.
        """
        if not self.rounds:
            raise ValueError("cannot project an empty SpecRun")
        rows = torch.cat([r.logits[: r.accept_len] for r in self.rounds], dim=0)
        return Trajectory(
            token_ids=self.emitted_tokens(),
            logits=rows,
            seed=self.seed,
            label=self.label,
        )
