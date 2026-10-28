# SPDX-License-Identifier: Apache-2.0
"""FORMHP-VPREP-EARLY (formhp-devdraft-vprep-1028): the DFLASH verify prep of an early-draft round runs BEFORE the length wait.

Why. In an early-draft round (FLLIPER_PDFLIP_D_EARLY_DRAFT, Form B steady state) the host waits for round N's committed lengths
(``seq_lens_cpu_ready``, FORMHP-DSTEP ``d_wait``) after the draft and BEFORE the verify prep. Everything after that wait up to the
verify launch is on the critical path whenever the card is ahead of the host: formb164 (0aa71fa3b) v_prep 0.70 ms bs1 / 1.12 ms bs6
of a post-wait stretch of 1.56 / 2.11 ms, while behind the wait the card only has round N's draft-KV append and the draft of N+1
(~1.7 ms at bs6, H58). The prep (DFlashVerifyInput, mamba track indices, seq_lens clone, ForwardBatch.init_new, can_run_graph and
the full load_batch: fill_from + the QSA / GDN replay metadata) reads round N's EXACT host lengths at one place only -- the verify's
host bound ``seq_lens_cpu + block`` -- and the replay path consumes none of it: the QSA GPU replay reads the device seq_lens, the
GDN replay takes num_padding, the buffers are sized by the bucket. The reservation bound (``nxt_kv_lens_cpu``, the branch that was
already there for a deferred read) is enough. Basalt's ``stage_inputs`` is the model (verify.cpp:3794): the inputs stand before
the launch, only flags run at it.

What. In an early-draft graph round the prep runs before the wait with the reservation bound; after the wait :func:`fix_after_wait`
puts the exact values where the old order put them (the forward batch's ``seq_lens_cpu`` / ``seq_lens_sum`` and the graph
registry's host ``seq_lens_cpu`` slot), so the host state equals the old order too. A round whose prep did not choose the graph
(eager) prepares again after the wait, the old way.

Self-test (boot, FORMHP-VPREP-EARLY self-test PASSED/FAILED): the first ``SELFTEST_ROUNDS`` eligible graph rounds run the early
prep + fix, snapshot every tensor and scalar the replay can read (the graph registry's slots, the runner's replay state, the attention
backends' replay metadata, the verify forward batch's fields), then the OLD prep after the wait (the round's result) and compare
bitwise. Any difference, an error, or an unmatched snapshot keeps the old order for the process. Switch ``FLLIPER_FORMHP_VPREP_EARLY``
(default on; 0/false/no/off = the old order). Read per call.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

ENV = "FLLIPER_FORMHP_VPREP_EARLY"
TAG = "FORMHP-VPREP-EARLY"
SELFTEST_ROUNDS = 3
LOG_EVERY = 2000  # one counter line per this many rounds
_MAX_TENSOR_BYTES = 64 << 20  # a replay metadata tensor is KB..MB; anything bigger (a KV pool) is not written by the prep
_MAX_TOTAL_BYTES = 256 << 20  # a snapshot that would clone more is refused (the self-test then keeps the old order)


class SnapshotTooBig(RuntimeError):
    pass
_RUNNER_STATE = ("raw_bs", "raw_num_token", "bs", "_replay_graph_key")
_SCALARS = (int, float, bool, str, type(None))


def switch_on() -> bool:
    return str(os.environ.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _walk(obj: Any, prefix: str, out: Dict[str, Any], depth: int, seen: set) -> None:
    """Every tensor (cloned) and scalar reachable from ``obj`` through dataclass fields, SimpleNamespace attributes and dict / list /
    tuple items, by path. Other objects are recorded by type only (never walked: a metadata object may hold a back reference to a
    runner or a pool)."""
    if isinstance(obj, torch.Tensor):
        nbytes = obj.numel() * obj.element_size()
        if nbytes > _MAX_TENSOR_BYTES:
            out[prefix] = ("big", tuple(obj.shape), obj.dtype, obj.data_ptr())
        else:
            total = out.get("__bytes__", 0) + nbytes
            if total > _MAX_TOTAL_BYTES:
                raise SnapshotTooBig(f"snapshot above {_MAX_TOTAL_BYTES >> 20} MiB at {prefix}")
            out["__bytes__"] = total
            out[prefix] = obj.detach().clone()
        return
    if isinstance(obj, _SCALARS):
        out[prefix] = obj
        return
    if depth <= 0 or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, dict):
        items = [(str(k), v) for k, v in obj.items()]
    elif isinstance(obj, (list, tuple)):
        items = [(str(i), v) for i, v in enumerate(obj)]
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        items = [(f.name, getattr(obj, f.name, None)) for f in dataclasses.fields(obj)]
    elif type(obj).__name__ == "SimpleNamespace":
        items = list(vars(obj).items())
    else:
        out[prefix] = f"<{type(obj).__module__}.{type(obj).__qualname__}>"
        return
    for k, v in items:
        if isinstance(v, (torch.nn.Module,)) or callable(v) and not isinstance(v, torch.Tensor):
            continue
        _walk(v, f"{prefix}.{k}", out, depth - 1, seen)


def _backends(attn_backend: Any):
    bl = getattr(attn_backend, "attn_backend_list", None)
    if isinstance(bl, (list, tuple)) and bl:
        return [(f"attn{i}", b) for i, b in enumerate(bl)]
    return [("attn", attn_backend)]


def snapshot(runner: Any, forward_batch: Any) -> Dict[str, Any]:
    """What the verify replay (and the code after it) can read from the prep: registry slots, runner replay state, the attention
    backends' ``forward_metadata``, the forward batch's own fields (tensors and scalars, one level)."""
    out: Dict[str, Any] = {}
    reg = getattr(runner, "buffer_registry", None)
    for name, slot in (getattr(reg, "_slots", None) or {}).items():
        buf = getattr(slot, "buffer", None)
        if buf is not None and getattr(slot, "enabled", True):
            _walk(buf, f"slot.{name}", out, 1, set())
    for k in _RUNNER_STATE:
        v = getattr(runner, k, None)
        out[f"runner.{k}"] = v if isinstance(v, _SCALARS) else repr(v)
    for label, be in _backends(getattr(runner, "attn_backend", None)):
        _walk(getattr(be, "forward_metadata", None), f"{label}.forward_metadata", out, 4, set())
    if forward_batch is not None:
        fields = dataclasses.fields(forward_batch) if dataclasses.is_dataclass(forward_batch) else ()
        for f in fields:
            v = getattr(forward_batch, f.name, None)
            if isinstance(v, (torch.Tensor,) + _SCALARS):
                _walk(v, f"fb.{f.name}", out, 1, set())
    return out


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, str]:
    """Bitwise: same keys, equal scalars, tensors equal in shape / dtype / device / every byte. Returns (equal, first difference)."""
    if a.keys() != b.keys():
        only = sorted(set(a) ^ set(b))
        return False, f"different paths: {only[:4]}"
    for k in a:
        if k == "__bytes__":
            continue
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
                return False, f"{k}: tensor vs {type(y).__name__ if isinstance(x, torch.Tensor) else type(x).__name__}"
            if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
                return False, f"{k}: {tuple(x.shape)}/{x.dtype}/{x.device} vs {tuple(y.shape)}/{y.dtype}/{y.device}"
            if x.numel() and not torch.equal(x.contiguous().view(-1).view(torch.uint8), y.contiguous().view(-1).view(torch.uint8)):
                return False, f"{k}: bytes differ"
        elif x != y:
            return False, f"{k}: {x!r} vs {y!r}"
    return True, f"{len(a)} paths bitwise equal"


def fix_after_wait(runner: Any, forward_batch: Any, exact_seq_lens_cpu: torch.Tensor, block_size: int) -> None:
    """After the length wait: the exact host values where the old order put them -- the verify forward batch's host bound
    (``seq_lens_cpu`` = committed + block, ``seq_lens_sum``) and the graph registry's host ``seq_lens_cpu`` slot (its first raw_bs
    rows; the padding rows hold the fill value in both orders)."""
    exact = exact_seq_lens_cpu + int(block_size)
    forward_batch.seq_lens_cpu = exact
    forward_batch.seq_lens_sum = int(exact.sum())
    reg = getattr(runner, "buffer_registry", None)
    slot = (getattr(reg, "_slots", None) or {}).get("seq_lens_cpu")
    buf = getattr(slot, "buffer", None)
    if buf is not None and getattr(slot, "enabled", True):
        n = int(exact.shape[0])
        buf[:n].copy_(exact.to(buf.dtype))


class VprepEarly:
    """Per worker: ``verdict`` None = self-test running, True = armed, False = old order for the process."""

    def __init__(self) -> None:
        self.verdict: Optional[bool] = None
        self.passes = 0
        self.why = ""
        # round census (27B port, 28.10.: the wall-clock effect is open on the metal, the counters say how many rounds took which path):
        # old = old order (switch off / not eligible / verdict False), early = armed early prep, selftest = self-test round,
        # redo = early prep done but the round prepared again after the wait (eager graph choice, error)
        self.counts: Dict[str, int] = {"old": 0, "early": 0, "selftest": 0, "redo": 0}

    def tick(self, kind: str) -> None:
        self.counts[kind] += 1
        n = sum(self.counts.values())
        if n % LOG_EVERY == 0:
            logger.info("%s rounds=%d %s verdict=%s (%s=0 = old order)", TAG, n, self.counts, self.verdict, ENV)

    def mode(self) -> Optional[str]:
        """``None`` (old order), ``"armed"`` or ``"selftest"``."""
        if self.verdict is False or not switch_on():
            return None
        return "armed" if self.verdict else "selftest"

    def record(self, ok: bool, why: str) -> None:
        if not ok:
            self.verdict, self.why = False, why
            logger.warning("%s self-test FAILED -> the verify prep stays behind the length wait (%s; %s=0 = off)", TAG, why, ENV)
            return
        self.passes += 1
        self.why = why
        if self.passes >= SELFTEST_ROUNDS:
            self.verdict = True
            logger.info("%s self-test PASSED (%d graph rounds, %s): the verify prep runs before the length wait with the reservation "
                        "bound (%s=0 = off)", TAG, self.passes, why, ENV)


def staged_prep(*, state: "VprepEarly", eligible: bool, prep: Any, marks: Any, wait: Any, exact_lens: Any, runner: Any,
                block_size: int, bs: int):
    """The round's verify prep around the length wait. ``prep()`` -> (verify_input, forward_batch, seq_lens_pre_verify, can_graph);
    ``marks()`` the prep's instrument marks (once per round); ``wait()`` the length wait (+ its hooks and marks); ``exact_lens()`` round
    N's exact host lengths after the wait (None = not available).

    old order (switch off, verdict False, not eligible): wait, prep. armed: prep (reservation bound), wait, fix -- a prep that chose
    eager prepares again after the wait. self-test: prep, wait, fix, snapshot, prep again (the round's result), compare, record."""
    mode = state.mode() if eligible else None
    early = None
    snap = None
    if mode is not None:
        try:
            early = prep()
            marks()
        except Exception as e:  # noqa: BLE001 -- the old order is always correct
            state.record(False, f"early prep raised {type(e).__name__}: {e}")
            early = None
    wait()
    vp = None
    lens = exact_lens() if early is not None else None
    if early is not None and early[3] and lens is not None:
        try:
            fix_after_wait(runner, early[1], lens, block_size)
            if mode == "armed":
                vp = early
            else:
                snap = snapshot(runner, early[1])
        except Exception as e:  # noqa: BLE001
            state.record(False, f"fix / snapshot raised {type(e).__name__}: {e}")
    if vp is not None:
        state.tick("early")
    else:
        kind = "old" if mode is None else ("selftest" if snap is not None else "redo")
        vp = prep()
        if snap is not None:
            try:
                ok, why = compare(snap, snapshot(runner, vp[1]))
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"compare raised {type(e).__name__}: {e}"
            state.record(ok, f"bs {bs}: {why}")
        if early is None:
            marks()
        state.tick(kind)
    return vp
