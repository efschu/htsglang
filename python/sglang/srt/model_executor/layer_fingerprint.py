"""Per-layer forward fingerprints for the determinism harness (task #343).

WHY THIS EXISTS. ``ModelRunner._determinism_dump_logits`` (the #124 tap) only
records the FINAL next-token logits row. That answers "do two runs diverge",
never "where". When an uneven-TP arm (``--rank-tp-ratio``) produces a correct
first token and a wrong second one, the interesting question is which layer's
output first stops matching the TP=1 reference on the first decode step, and
that needs a tap inside the forward pass.

WHAT IT RECORDS. One JSONL row per (step, tensor):

    step, mode, name, shape, dtype, sha256 of the fp32-cast bytes, the leading
    values, and max|x|.

The hash is the match/mismatch verdict; the leading values and max|x| make a
mismatching row readable without loading anything. For the steps named in
``full_steps`` the whole fp32 tensor is additionally written as a ``.pt``, so
a comparison can report an exact max-abs delta rather than only "differs".

WHAT IT HOOKS. Module selection is by structural role, not by model-specific
names, so the same tap works on any decoder stack:

    embed             token embedding output
    L<i>.attn_shard   attention output BEFORE o_proj -- this rank's HEAD SLICE,
                      the tensor whose width changes under --rank-tp-ratio
    L<i>.o_proj       after o_proj, i.e. after the row-parallel all-reduce;
                      every rank and the TP=1 reference must agree here
    L<i>.mlp          MLP block output (also post all-reduce)
    L<i>.out          decoder layer output (hidden_states, residual)
    final_norm        the stack's output norm
    logits            the logits processor's next-token logits, post vocab
                      gather

STEP ALIGNMENT. A raw forward counter is useless across two arms: boot warmup
and memory profiling burn a different number of forwards on TP=1 than on TP=2,
so "step 3" means different things in the two traces. The tap is therefore
ARMED from outside -- the harness creates ``<dump_dir>/ARM`` once the server is
up and before it sends the first probe request. Every row carries ``astep``,
the forward index counted from the arming point, and a comparison joins on
that. ``astep == 0`` is the probe's prefill and ``astep == 1`` its first decode
step, in both arms, by construction rather than by hope. Rows before arming
carry ``astep: null`` and are ignored by the comparison.

GATING. Two conditions, both off by default, so neither the serving path nor
the existing #124 harness changes: ``--determinism-logits-dump-dir`` must be
set AND ``SGLANG_DETERMINISM_LAYER_FINGERPRINT=1`` must be in the environment.
The second gate is separate because these hooks force a device-to-host sync per
layer, which the logits-only #124 harness must not pay.

CUDA GRAPHS. A ``.cpu()`` inside a graph capture is illegal, and a hook that
ran only at capture time would silently record nothing at replay. Capture is
therefore skipped explicitly (with a one-time warning); run the probe with
``--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled``
to get decode-step rows at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

ENV_ENABLE = "SGLANG_DETERMINISM_LAYER_FINGERPRINT"
ENV_FULL_STEPS = "SGLANG_DETERMINISM_LAYER_FULL_STEPS"
ENV_HEAD_VALUES = "SGLANG_DETERMINISM_LAYER_HEAD_VALUES"

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)$")

DEFAULT_FULL_STEPS = "0,1"
DEFAULT_HEAD_VALUES = 8


def layer_fingerprint_enabled(dump_dir: Optional[str]) -> bool:
    """Both gates, evaluated in one place."""
    return bool(dump_dir) and os.environ.get(ENV_ENABLE, "0") == "1"


def _parse_steps(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _tensors_of(value: Any) -> List[Tuple[str, torch.Tensor]]:
    """Flatten a module output into named tensors.

    Tuples are the common decoder-layer shape ``(hidden_states, residual)``;
    the index suffix keeps the two apart. Anything that merely carries tensors
    as attributes is handled by the caller, which knows what it asked for.
    """
    if isinstance(value, torch.Tensor):
        return [("", value)]
    if isinstance(value, (tuple, list)):
        out: List[Tuple[str, torch.Tensor]] = []
        for i, item in enumerate(value):
            if isinstance(item, torch.Tensor):
                out.append((f"[{i}]", item))
        return out
    logits = getattr(value, "next_token_logits", None)
    if isinstance(logits, torch.Tensor):
        return [("", logits)]
    return []


class LayerFingerprintDumper:
    """Registers the hooks and owns the step counter and the output files."""

    def __init__(
        self,
        dump_dir: str,
        tp_rank: int,
        full_steps: Optional[Sequence[int]] = None,
        head_values: Optional[int] = None,
    ) -> None:
        self.dump_dir = dump_dir
        self.tp_rank = tp_rank
        self.step = -1
        self.astep: Optional[int] = None
        self.mode = "unknown"
        self._handles: List[Any] = []
        self._warned_capture = False
        self._armed_at: Optional[int] = None
        self.arm_path = os.path.join(dump_dir, "ARM")
        if full_steps is None:
            full_steps = _parse_steps(
                os.environ.get(ENV_FULL_STEPS, DEFAULT_FULL_STEPS)
            )
        self.full_steps = set(full_steps)
        if head_values is None:
            head_values = int(os.environ.get(ENV_HEAD_VALUES, str(DEFAULT_HEAD_VALUES)))
        self.head_values = head_values
        os.makedirs(self.dump_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.dump_dir, f"layers_rank{tp_rank}.jsonl")
        # Truncate: a rank that restarts must not append to a stale trace.
        with open(self.jsonl_path, "w"):
            pass

    # -- recording -----------------------------------------------------------

    def _skip(self) -> bool:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            if not self._warned_capture:
                logger.warning(
                    "layer fingerprint: skipping a CUDA-graph capture; run with "
                    "--cuda-graph-backend-decode disabled "
                    "--cuda-graph-backend-prefill disabled to record decode steps."
                )
                self._warned_capture = True
            return True
        return False

    def record(self, name: str, tensor: torch.Tensor) -> None:
        if self._skip() or self.step < 0:
            return
        cpu = tensor.detach().to(torch.float32).cpu().contiguous()
        flat = cpu.reshape(-1)
        row = {
            "step": self.step,
            "astep": self.astep,
            "mode": self.mode,
            "rank": self.tp_rank,
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(flat.numpy().tobytes()).hexdigest(),
            "head": [float(v) for v in flat[: self.head_values].tolist()],
            "absmax": float(flat.abs().max().item()) if flat.numel() else 0.0,
        }
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if self.astep is not None and self.astep in self.full_steps:
            safe = name.replace("/", "_")
            path = os.path.join(
                self.dump_dir,
                f"full_rank{self.tp_rank}_astep{self.astep:03d}_{safe}.pt",
            )
            torch.save(cpu, path + ".tmp")
            os.replace(path + ".tmp", path)

    def record_output(self, name: str, value: Any) -> None:
        for suffix, tensor in _tensors_of(value):
            self.record(name + suffix, tensor)

    # -- attachment ----------------------------------------------------------

    def _role_of(self, name: str, module: torch.nn.Module) -> Optional[str]:
        """Map a module path to a fingerprint role, or None to leave it alone."""
        cls = type(module).__name__
        if cls == "LogitsProcessor":
            return "logits"
        match = _LAYER_INDEX_RE.search(name)
        if match:
            return f"L{int(match.group(1)):02d}.out"
        parts = name.split(".")
        parent = ".".join(parts[:-1])
        leaf = parts[-1]
        layer_match = _LAYER_INDEX_RE.search(parent) or _LAYER_INDEX_RE.search(
            ".".join(parts[:-2])
        )
        if layer_match:
            idx = int(layer_match.group(1))
            if leaf == "attn":
                return f"L{idx:02d}.attn_shard"
            if leaf == "o_proj":
                return f"L{idx:02d}.o_proj"
            if leaf == "mlp":
                return f"L{idx:02d}.mlp"
            return None
        if leaf == "embed_tokens":
            return "embed"
        if leaf == "norm" and parts[:-1] and parts[-2] != "layers":
            return "final_norm"
        return None

    def begin_step(self, forward_batch: Any = None) -> None:
        """Open a new forward step. Called by ``ModelRunner._forward_raw``.

        Deliberately not a forward pre-hook on the model: sglang enters the
        model as ``model.forward(...)``, and torch runs hooks only on
        ``__call__``. A root pre-hook would never fire while every submodule
        hook still would -- recording against a step that was never opened,
        which is silently an empty trace.
        """
        self.step += 1
        if self._armed_at is None and os.path.exists(self.arm_path):
            self._armed_at = self.step
        self.astep = None if self._armed_at is None else self.step - self._armed_at
        mode = getattr(forward_batch, "forward_mode", None)
        try:
            self.mode = "decode" if mode.is_decode() else "extend"
        except AttributeError:
            self.mode = "unknown"
        input_ids = getattr(forward_batch, "input_ids", None)
        if self.astep is not None and isinstance(input_ids, torch.Tensor):
            self.record("input_ids", input_ids)

    def attach(self, model: torch.nn.Module) -> int:
        """Register the hooks. Returns how many tensors are being watched."""
        count = 0
        for name, module in model.named_modules():
            if not name:
                continue
            role = self._role_of(name, module)
            if role is None:
                continue

            def make_hook(role_name: str):
                def hook(_module, _args, output):
                    self.record_output(role_name, output)
                    return None

                return hook

            self._handles.append(module.register_forward_hook(make_hook(role)))
            count += 1
        logger.warning(
            "layer fingerprint: %d hooks attached, trace -> %s (full tensors for "
            "steps %s)",
            count,
            self.jsonl_path,
            sorted(self.full_steps),
        )
        return count

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def maybe_attach(
    model: torch.nn.Module, dump_dir: Optional[str], tp_rank: int
) -> Optional[LayerFingerprintDumper]:
    """Attach the tap when both gates are set; otherwise do nothing at all."""
    if not layer_fingerprint_enabled(dump_dir):
        return None
    dumper = LayerFingerprintDumper(dump_dir, tp_rank)
    dumper.attach(model)
    return dumper


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
