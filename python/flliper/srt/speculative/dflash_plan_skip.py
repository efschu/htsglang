# SPDX-License-Identifier: Apache-2.0
"""FORMHP-DRAFT-PLAN-SKIP (formhp-devdraft-b2-1028): the DFLASH draft's FlashInfer plan without the wrapper no layer reads.

Why. The DFlash2 draft (5 x ``sliding_attention``) runs on a FlashInfer backend with ``WrapperDispatch.SLIDING_WINDOW``: two prefill
wrappers, 0 = sliding, 1 = full, and ``FlashInferIndicesUpdaterPrefill.update_sliding_window`` plans BOTH every round
(``generate_attn_arg_prefill`` -- arange / zeros / add / cumsum / empty + a Triton launch --, the host lengths, ``HostPlanMeta``, the FlashInfer
``plan()`` with its pageable H2D, ``_assert_plan_indptr_matches``). ``FlashInferAttnBackend._get_wrapper_idx`` maps a layer to wrapper 1 only when
its ``sliding_window_size == -1``; the DFlash2 draft has no such layer, so the second plan is written every round and read by nobody. It sits in
the FORMHP-DSTEP mark ``d_load`` (1.75 ms at bs1, formb156), i.e. in the host stretch before the verify launch while the card idles
(/spinning/gpu-arb/deskq/basalt-vs-formb-1028.md, Posten 1; Basalt plans nothing per window: its kernels read the length from the device).

What. The updater skips the plan of a wrapper id in ``skip_wrapper_ids`` (empty by default: every other model and line is byte-identical).
The DFLASH worker arms it for the draft backend only when BOTH hold:

* static: :func:`unread_wrapper_ids` over the draft model's attention layers says the wrapper is read by no layer, and
* numeric (the boot self-test, FORMHP-DRAFT-PLAN-SKIP self-test PASSED/FAILED): on the first eligible draft round (solo host or TP=1 draft, graph replay,
  all-greedy, no collective inside the draft, no hole mask) the draft forward runs once with that wrapper planned on a deliberately DIFFERENT
  valid layout (one key per row, ``probe_wrapper_ids``) and then once the old way; the old forward is the round's result. Hidden rows and the
  folded greedy tokens must be bitwise equal -- a layer that read the wrapper would see 1 key instead of the full context and differ. A
  difference or an error keeps the old plan (both wrappers) for the process.

Switch ``FLLIPER_FORMHP_DRAFT_SKIP_UNREAD_PLAN`` (default on; 0/false/no/off = both wrappers planned, no self-test). Read per call.

Rank-uniform (27b-rank-uniform-1028). The 27B runs D with TP2/TP3. With solo placement only the solo HOST runs a draft (the shadows
never reach the self-test), with split placement the draft is TP-sharded and never probes -- both decide on one rank, ``reduce_min``
is None. Any placement in which SEVERAL ranks probe (a TP=1 draft replicated on a TP>1 group) gets the worker's
``SpecTpSync.verdict_reducer``: the probe participation (:meth:`DraftPlanSkip.agree_candidate`) and every probe round's outcome
(:meth:`DraftPlanSkip.settle`, FAIL < UNDECIDED < PASS) are MIN-reduced over the TP CPU group, so a rank arms the skip only when
EVERY rank passed, and one failing rank keeps the old plan on all of them. Per probing round and rank exactly two reductions
(participation, outcome), none once the verdict is taken.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

ENV = "FLLIPER_FORMHP_DRAFT_SKIP_UNREAD_PLAN"
TAG = "FORMHP-DRAFT-PLAN-SKIP"
# a probe round's outcome, ordered so that the MIN over the ranks is the group's outcome
FAIL, UNDECIDED, PASS = 0, 1, 2
_OUTCOME = {FAIL: "FAILED", UNDECIDED: "undecided", PASS: "PASSED"}
_NUM_WRAPPERS = 2  # the SLIDING_WINDOW dispatch (unread_wrapper_ids)


def switch_on() -> bool:
    return str(os.environ.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _attention_layers(model: Any) -> Iterable[Any]:
    """The model's attention layers: modules carrying both ``layer_id`` and ``sliding_window_size`` (RadixAttention's fields)."""
    if not isinstance(model, torch.nn.Module):
        return ()
    # duck scan on purpose: any module with both fields is an attention layer (RadixAttention and the CPU test stubs alike)
    return [m for m in model.modules() if hasattr(m, "sliding_window_size") and hasattr(m, "layer_id")]


def unread_wrapper_ids(model: Any, num_wrappers: int, sliding_dispatch: bool) -> Tuple[int, ...]:
    """Wrapper ids of a SLIDING_WINDOW FlashInfer backend that no attention layer of ``model`` maps to
    (``_get_wrapper_idx``: ``sliding_window_size == -1`` -> 1, else 0). ``()`` when unsure: not the sliding dispatch, not two wrappers, or no
    attention layer found (an empty scan proves nothing)."""
    if not sliding_dispatch or int(num_wrappers) != 2:
        return ()
    layers = list(_attention_layers(model))
    if not layers:
        return ()
    used = set()
    for m in layers:
        sws = m.sliding_window_size
        if sws is None:
            return ()  # a layer whose window is unknown: the full wrapper might be read
        used.add(1 if int(sws) == -1 else 0)
    return tuple(i for i in range(2) if i not in used)


def outputs_equal(a: Sequence[Optional[Any]], b: Sequence[Optional[Any]]) -> Tuple[bool, str]:
    """Bitwise equality of two snapshots (tensors or None, pairwise): shape, dtype and every byte. Returns (equal, why)."""
    if len(a) != len(b):
        return False, f"{len(a)} vs {len(b)} outputs"
    for i, (x, y) in enumerate(zip(a, b)):
        if x is None and y is None:
            continue
        if x is None or y is None:
            return False, f"output {i}: one side is None"
        if tuple(x.shape) != tuple(y.shape) or x.dtype != y.dtype:
            return False, f"output {i}: {tuple(x.shape)}/{x.dtype} vs {tuple(y.shape)}/{y.dtype}"
        xb = x.contiguous().view(-1).view(torch.uint8) if x.numel() else x
        yb = y.contiguous().view(-1).view(torch.uint8) if y.numel() else y
        if x.numel() and not bool((xb == yb).all()):
            n = int((xb != yb).sum())
            return False, f"output {i}: {n} of {int(xb.numel())} bytes differ"
    return True, "bitwise equal"


class DraftPlanSkip:
    """Per worker: the verdict of the self-test and the arming of the draft backend's prefill updater.

    ``verdict``: None = not decided yet, True = armed (the updater skips ``ids``), False = old plan for the process.
    ``reduce_min``: None = this rank decides alone; else the TP CPU-group MIN (``SpecTpSync.verdict_reducer``) every probing rank
    calls exactly once per probing round (in :meth:`agree_candidate` and in :meth:`settle`)."""

    def __init__(self, *, reduce_min: Optional[Callable[[Sequence[int]], List[int]]] = None) -> None:
        self.verdict: Optional[bool] = None
        self.ids: Tuple[int, ...] = ()
        self.why = ""
        self.undecided = 0  # probe rounds that ran eagerly (no graph replay): they decide nothing
        self.max_undecided = 4
        self.reduce_min = reduce_min

    def agree_candidate(self, updater: Any, ids: Tuple[int, ...]) -> Tuple[int, ...]:
        """The probe ids every rank agrees on, or ``()`` -- then the verdict is FAILED on every rank (a rank-local switch, updater or
        static scan differs; probing on some ranks only would leave the others out of the outcome reduction). Identity without
        ``reduce_min``."""
        if self.reduce_min is None:
            return ids
        flags = [1 if ids else 0] + [1 if i in ids else 0 for i in range(_NUM_WRAPPERS)]
        red = self.reduce_min(flags + [1 - f for f in flags])  # MIN of a flag and of its complement: both 0 <=> the ranks differ
        uniform = all(red[i] or red[i + len(flags)] for i in range(len(flags)))
        if uniform and ids:
            return ids
        if self.verdict is None:
            why = ("no rank has a probe candidate (switch off or static scan)" if uniform
                   else f"rank-uniform: the ranks' probe candidates differ (local {list(ids)})")
            self.decide(updater, False, why)
        return ()

    def settle(self, updater: Any, *, code: int, why: str, ids: Tuple[int, ...] = ()) -> None:
        """One probe round's outcome (FAIL / UNDECIDED / PASS) -> the verdict; with ``reduce_min`` the MIN over the ranks first.
        UNDECIDED counts toward ``max_undecided`` (then FAILED)."""
        if self.reduce_min is not None:
            group = self.reduce_min([int(code)])[0]
            if group != code:
                why = f"{why}; rank-uniform: another rank {_OUTCOME[group]} -> {_OUTCOME[group]} on every rank"
            code = group
        if code == UNDECIDED:
            self.undecided += 1
            if self.undecided >= self.max_undecided:
                self.decide(updater, False, f"{self.undecided} probe rounds without a draft graph replay")
            return
        self.decide(updater, code == PASS, why, ids)

    def candidate(self, updater: Any, model: Any) -> Tuple[int, ...]:
        """The wrapper ids the self-test may probe, or ``()`` (switch off, verdict taken, not a sliding FlashInfer updater, all read)."""
        if self.verdict is not None or not switch_on() or updater is None:
            return ()
        sliding = updater.update == updater.update_sliding_window
        ids = unread_wrapper_ids(model, int(updater.attn_backend.num_wrappers), sliding)
        if not ids:
            self.decide(updater, False, "static scan: every wrapper is read by some draft attention layer (or no layer found)")
        return ids

    def decide(self, updater: Any, ok: bool, why: str, ids: Tuple[int, ...] = ()) -> None:
        self.verdict = bool(ok)
        self.why = why
        self.ids = tuple(ids) if ok else ()
        if updater is not None:
            updater.probe_wrapper_ids = frozenset()
            updater.skip_wrapper_ids = frozenset(self.ids)
        (logger.info if ok else logger.warning)(
            "%s self-test %s: %s (%s=0 = off)", TAG,
            f"PASSED -> the draft plans without wrapper(s) {list(self.ids)} (no layer reads them)" if ok
            else "FAILED -> both wrappers planned as before", why, ENV)
