# SPDX-License-Identifier: Apache-2.0
"""Data model for the #124 determinism harness.

A :class:`Trajectory` is the atomic observable of every byte-identity gate:
one greedy decode run (or single-shot prefill, T == 1) captured as the
per-step next-token logits row plus the token that was actually emitted.

All comparison primitives (:mod:`.primitives`) are pure functions over pairs
of trajectories and return a structured :class:`Verdict` -- never a bare
bool -- so a failing gate always names the first diverging token, the flip
classification (genuine near-tie vs. real corruption) and the observed
magnitudes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch


class FlipKind(enum.Enum):
    """Classification of an argmax flip between a reference and a test run.

    NEAR_TIE:   the reference's own logits put the test-chosen token within
                ``near_tie_margin`` of the reference winner -- a genuine
                coin-flip near-tie (the only divergence the
                SELF_DET_NEAR_TIE class tolerates; pre-existing cross-boot
                sensitivity, see memory ``hetero-spec-determinismus``).
    CORRUPTION: the reference was confident (margin > ``near_tie_margin``)
                and the test still picked a different token -- a real
                regression. No byte-identity class ever tolerates this.
    """

    NONE = "none"
    NEAR_TIE = "near_tie"
    CORRUPTION = "corruption"


@dataclass
class Verdict:
    """Structured result of one comparison check (or a bundle of checks)."""

    ok: bool
    check: str
    message: str = ""
    #: First decode step (0-based) at which the pair diverged, if any.
    first_divergence_step: Optional[int] = None
    flip_kind: FlipKind = FlipKind.NONE
    #: max |ref - test| over the compared logits region (float64 math).
    max_abs_delta: Optional[float] = None
    #: Reference-side margin ``ref_logits[t, ref_token] - ref_logits[t, test_token]``
    #: at the divergence step (the near-tie discriminator).
    margin_at_divergence: Optional[float] = None
    sub_verdicts: List["Verdict"] = field(default_factory=list)

    def summary(self, indent: int = 0) -> str:
        pad = "  " * indent
        head = f"{pad}[{'PASS' if self.ok else 'FAIL'}] {self.check}"
        bits = []
        if self.first_divergence_step is not None:
            bits.append(f"diverged@step={self.first_divergence_step}")
        if self.flip_kind is not FlipKind.NONE:
            bits.append(f"flip={self.flip_kind.value}")
        if self.max_abs_delta is not None:
            bits.append(f"max|d|={self.max_abs_delta:.3e}")
        if self.margin_at_divergence is not None:
            bits.append(f"margin={self.margin_at_divergence:.3e}")
        if bits:
            head += " (" + ", ".join(bits) + ")"
        if self.message:
            head += f": {self.message}"
        lines = [head]
        for sub in self.sub_verdicts:
            lines.append(sub.summary(indent + 1))
        return "\n".join(lines)


def combine(check: str, subs: Sequence[Verdict], message: str = "") -> Verdict:
    """Bundle sub-verdicts; the composite fails iff any sub-check fails and
    inherits the first failing sub's divergence step / flip / magnitudes.

    A composite that PASSES still inherits the diagnostics of the first sub
    that recorded a divergence. The near-tie classes pass *with* a fork -- a
    legitimate one -- and a gate log that reports "passed" without naming the
    fork token and its margin hides the one number a reader needs to judge
    whether the near-tie verdict was earned or lucky.
    """
    ok = all(s.ok for s in subs)
    v = Verdict(ok=ok, check=check, message=message, sub_verdicts=list(subs))
    donor = next((s for s in subs if not s.ok), None)
    if donor is None:
        donor = next((s for s in subs if s.first_divergence_step is not None), None)
    if donor is not None:
        v.first_divergence_step = donor.first_divergence_step
        v.flip_kind = donor.flip_kind
        v.max_abs_delta = donor.max_abs_delta
        v.margin_at_divergence = donor.margin_at_divergence
        if not message:
            v.message = (
                f"sub-check '{donor.check}' failed"
                if not donor.ok
                else donor.message
            )
    return v


@dataclass
class Trajectory:
    """One captured run: per-step emitted token + per-step next-token logits.

    ``logits`` is ``[T, V]`` (T decode steps, V vocab -- or top-k slice, see
    the runner docstring for the top-k caveat). Row ``t`` is the model's
    next-token distribution *before* sampling at step ``t``; ``token_ids[t]``
    is the token actually emitted at that step. For greedy gates (all #124
    gates run temperature 0 with the pinned seed) ``token_ids`` equals the
    row-wise argmax.

    A single-shot prefill gate is a ``T == 1`` trajectory (the last-position
    prefill logits row + the first sampled token).
    """

    token_ids: Sequence[int]
    logits: torch.Tensor
    seed: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.logits.dim() != 2:
            raise ValueError(f"logits must be [T, V], got {tuple(self.logits.shape)}")
        if len(self.token_ids) != self.logits.shape[0]:
            raise ValueError(
                f"token_ids length {len(self.token_ids)} != logits rows "
                f"{self.logits.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.logits.shape[0])

    @classmethod
    def from_greedy(
        cls, logits: torch.Tensor, seed: int, label: str = ""
    ) -> "Trajectory":
        """Build a trajectory whose tokens are the row-wise argmax (greedy)."""
        return cls(
            token_ids=[int(t) for t in logits.argmax(dim=-1)],
            logits=logits,
            seed=seed,
            label=label,
        )

    def margin_to(self, step: int, token: int) -> float:
        """Reference-side near-tie discriminator: how far below THIS
        trajectory's step-``step`` winner the given ``token`` scores,
        ``logits[step].max() - logits[step, token]`` in float64.

        <= 0 means ``token`` is (one of) the winner(s); small positive means
        a genuine near-tie candidate; large positive means this trajectory
        was confident against ``token``.
        """
        row = self.logits[step].to(torch.float64)
        return float(row.max() - row[token])
