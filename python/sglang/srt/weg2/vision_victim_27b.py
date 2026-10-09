# SPDX-License-Identifier: Apache-2.0
"""VISION-WEIGHTS AP2 (plan PLAN-VISION-GEWICHTE-VERDRAENGEN-1009 §2/§4): the
27B line's victim sources for the transient tower.

* ``dense`` (flip, INT8 and NVFP4): the MLP storages of P's PP0 stage
  (``model.layers.<i>.mlp.*``) -- a whole layer band is 143-255 MiB, the fused
  gate_up alone 85-170 MiB, so the largest tower tensor (45 MiB) fits.
* ``pp_only`` (dual, ``SGLANG_WEG2_DUAL_SHARE=1``): only the hull PARTS that
  are not D's shard on this card. P's stage is a hull over D's three TP3
  shards; the part of the D rank on this card is ONE set of bytes with D
  (union bind) and D reads it while it decodes -- borrowing it would put the
  tower into D's weights. The other two parts D never reads.
  Their largest tensor is ~12 MiB (mlp 19/136, mixer 25/108), so the merger
  linears (40.5 / 45.0 MiB) are ROW-SPLIT (:func:`row_split_map`): weight
  ``[out, in]`` cut along dim 0 into pieces that each fit one victim, forward
  = one GEMM per piece + cat. Only tensors larger than the largest run.

Never a victim (R2): any storage the local part reaches (params, buffers,
plain tensor attributes), any storage inside an attached union arena (D's
VMM image), ``embed_tokens`` / ``lm_head`` (draft/tied), the tower itself,
an ``mtp.`` head, any storage the draft reaches. A source with nothing
eligible is a named arming refusal (W111b), never a fallback to KV.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.weg2 import vision_victim as vv

logger = logging.getLogger(__name__)

KIND_DENSE = "dense"
KIND_PP_ONLY = "pp_only"

#: name parts that never make a victim (R2)
EXCLUDED_NAME_PARTS = ("embed_tokens", "lm_head", "visual.", "mtp.")


# ---------------------------------------------------------------------------
# 1. eligibility (pure over tensors)
# ---------------------------------------------------------------------------


def storage_key(t: torch.Tensor) -> Optional[Tuple[int, int]]:
    """(base address, bytes) of ``t``'s storage, or None (meta / no bytes)."""
    if t.device.type == "meta":
        return None
    try:
        st = t.untyped_storage()
    except (RuntimeError, NotImplementedError):
        return None
    n = int(st.nbytes())
    return (int(st.data_ptr()), n) if n > 0 else None


def reachable_tensors(model: Any) -> Iterable[Tuple[str, torch.Tensor]]:
    """Every tensor a module reaches: parameters, buffers, plain tensor
    attributes (a kernel workspace, a captured view)."""
    for name, p in model.named_parameters():
        yield name, p
    for name, b in model.named_buffers():
        if b is not None:
            yield name, b
    for path, mod in model.named_modules():
        for attr, v in list(vars(mod).items()):
            if isinstance(v, torch.Tensor):
                yield (f"{path}.{attr}" if path else attr), v


def reachable_keys(models: Iterable[Any]) -> Set[Tuple[int, int]]:
    out = set()
    for m in models:
        for _n, t in reachable_tensors(m):
            k = storage_key(t)
            if k is not None:
                out.add(k)
    return out


def _in_ranges(key: Tuple[int, int], ranges: Sequence[Tuple[int, int]]) -> bool:
    lo, hi = key[0], key[0] + key[1]
    return any(lo < r_hi and r_lo < hi for r_lo, r_hi in ranges)


def name_is_victim(name: str, *, mlp_only: bool) -> bool:
    if any(p in name for p in EXCLUDED_NAME_PARTS):
        return False
    return ".mlp." in name if mlp_only else True


def eligible_storages(named: Iterable[Tuple[str, torch.Tensor]], *, shared: Set[Tuple[int, int]],
                      arena_ranges: Sequence[Tuple[int, int]], mlp_only: bool,
                      device_types: Tuple[str, ...] = ("cuda",)) -> List[Tuple[str, torch.Tensor]]:
    """(name, uint8 whole-storage view) of every storage that may be a victim,
    deduplicated by storage, in the given order."""
    seen: Set[Tuple[int, int]] = set()
    out = []
    for name, t in named:
        if t.device.type not in device_types or not name_is_victim(name, mlp_only=mlp_only):
            continue
        k = storage_key(t)
        if k is None or k in seen or k in shared or _in_ranges(k, arena_ranges):
            continue
        seen.add(k)
        out.append((name, vv.storage_view(t)))
    return out


# ---------------------------------------------------------------------------
# 2. the row split of a tower tensor larger than every victim run
# ---------------------------------------------------------------------------


def row_split_map(tower: Sequence[Tuple[str, Tuple[int, ...], int]], largest_run: int) -> vv.SplitMap:
    """name -> ((piece name, rows), ...) for every tensor larger than
    ``largest_run``: the fewest equal row pieces that each fit. Only a 2-D
    ``.weight`` can be split; anything else larger than every run is W105b."""
    out: vv.SplitMap = {}
    for name, shape, n in tower:
        if n <= largest_run:
            continue
        if len(shape) != 2 or not name.endswith(".weight") or largest_run <= 0:
            raise vv.VisionVictimShort(
                f"{vv.W_VICTIM_SHORT}: tower tensor {name} {tuple(shape)} needs {n / vv.MIB:.1f} MiB in one "
                f"piece and cannot be row-split; the largest victim run is {largest_run / vv.MIB:.1f} MiB")
        rows, row_bytes = int(shape[0]), n // int(shape[0])
        if row_bytes > largest_run:
            raise vv.VisionVictimShort(
                f"{vv.W_VICTIM_SHORT}: one row of {name} ({row_bytes} B) exceeds the largest victim run")
        k = max(2, math.ceil(n / largest_run))
        while math.ceil(rows / k) * row_bytes > largest_run:
            k += 1
        per = math.ceil(rows / k)
        cuts = [per] * (rows // per) + ([rows % per] if rows % per else [])
        owner = name[: -len(".weight")]
        out[name] = tuple((f"{owner}.weight_parts.{i}", r) for i, r in enumerate(cuts))
    return out


class RowSplitLinear(torch.nn.Module):
    """A linear whose weight lives in row pieces (one victim each). Forward:
    one GEMM per piece with its slice of the bias, concatenated -- the
    unsplit linear's numbers up to the GEMM's summation order (each output
    element is its own dot product; CPU fp32 measured <= 1 ulp apart).
    Mirrors the owner's return convention: a ``torch.nn.Linear`` returns the
    tensor, an sglang ``LinearBase`` ``(out, bias or None)``. Has no
    ``.weight``: a reader of the full matrix fails by name (W107), never on
    a wrong tensor."""

    def __init__(self, owner: torch.nn.Module, rows: Sequence[int]):
        super().__init__()
        w = owner.weight
        self.weight_parts = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty((int(r), int(w.shape[1])), dtype=w.dtype, device=w.device),
                                requires_grad=False) for r in rows])
        self.bias = owner.bias
        self._rows = tuple(int(r) for r in rows)
        self.returns_tuple = not isinstance(owner, torch.nn.Linear)
        self.skip_bias_add = bool(getattr(owner, "skip_bias_add", False))
        self.out_features = int(w.shape[0])
        self.in_features = int(w.shape[1])

    def forward(self, x: torch.Tensor):
        fuse = self.bias is not None and not self.skip_bias_add
        outs, r0 = [], 0
        for w, r in zip(self.weight_parts, self._rows):
            outs.append(F.linear(x, w, self.bias[r0:r0 + r] if fuse else None))
            r0 += r
        out = torch.cat(outs, dim=-1)
        if not self.returns_tuple:
            return out
        return out, (self.bias if self.skip_bias_add else None)


def _splittable(owner: torch.nn.Module) -> str:
    """'' when ``owner``'s forward is ``F.linear(x, weight, bias)``, else why not."""
    if isinstance(owner, torch.nn.Linear):
        return ""
    try:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    except ImportError as exc:  # pragma: no cover
        return f"sglang linear layer not importable ({exc})"
    if not isinstance(owner, LinearBase):
        return f"{type(owner).__name__} is not a linear layer"
    if not isinstance(getattr(owner, "quant_method", None), UnquantizedLinearMethod):
        return f"{type(owner).__name__} is quantized"
    if int(getattr(owner, "tp_size", 1) or 1) != 1:
        return f"{type(owner).__name__} runs tp_size={owner.tp_size}"
    return ""


def apply_row_split(module: torch.nn.Module, split: vv.SplitMap) -> None:
    """Replace each split tensor's owner by a :class:`RowSplitLinear` (on
    meta, before the placement)."""
    for name, pieces in split.items():
        owner_path = name[: -len(".weight")]
        owner = module.get_submodule(owner_path)
        why = _splittable(owner)
        if why:
            raise vv.VisionVictimShort(f"{vv.W_VICTIM_SHORT}: {owner_path} cannot be row-split: {why}")
        parent_path, _, attr = owner_path.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        setattr(parent, attr, RowSplitLinear(owner, [r for _p, r in pieces]))


# ---------------------------------------------------------------------------
# 3. the two sources
# ---------------------------------------------------------------------------


class _Line27BVictims(vv.HostImageVictims):
    """Shared shape of both 27B sources: a host-image return and the row
    split; a subclass names the eligible storages."""

    def shape_tower(self, module, tower, largest_run) -> vv.SplitMap:
        split = row_split_map(tower, largest_run)
        if split and module is not None:
            apply_row_split(module, split)
        return split


class DenseMlpVictims(_Line27BVictims):
    """Flip: the MLP storages of this rank's model (PP0's layers only)."""

    kind = KIND_DENSE

    def __init__(self, model: Any, *, exclude: Sequence[Any] = (), arena_ranges: Sequence[Tuple[int, int]] = (),
                 device_types: Tuple[str, ...] = ("cuda",)):
        super().__init__()
        self._model = model
        self._exclude = list(exclude)
        self._arena = list(arena_ranges)
        self._devices = device_types

    def _named_storages(self):
        return eligible_storages(self._model.named_parameters(), shared=reachable_keys(self._exclude),
                                 arena_ranges=self._arena, mlp_only=True, device_types=self._devices)


class PpOnlyVictims(_Line27BVictims):
    """Dual: the storages of every hull part EXCEPT the one D shares on this
    card (``local``)."""

    kind = KIND_PP_ONLY

    def __init__(self, parts: Sequence[Any], local: int, *, exclude: Sequence[Any] = (),
                 arena_ranges: Sequence[Tuple[int, int]] = (), device_types: Tuple[str, ...] = ("cuda",)):
        super().__init__()
        if not 0 <= int(local) < len(parts):
            raise vv.VisionVictimPlanRefused(
                f"{vv.W_VICTIM_PLAN_REFUSED}: shared part {local} is not one of the {len(parts)} hull parts")
        self._parts = list(parts)
        self._local = int(local)
        self._exclude = list(exclude)
        self._arena = list(arena_ranges)
        self._devices = device_types

    def _named_storages(self):
        shared = reachable_keys([self._parts[self._local]] + self._exclude)
        named = ((f"part{r}.{n}", p) for r, part in enumerate(self._parts) if r != self._local
                 for n, p in part.named_parameters())
        return eligible_storages(named, shared=shared, arena_ranges=self._arena, mlp_only=False,
                                 device_types=self._devices)


# ---------------------------------------------------------------------------
# 4. registration: which source a rank gets
# ---------------------------------------------------------------------------


def _runner(scheduler):
    return scheduler.tp_worker.model_runner


def _draft_models(scheduler) -> List[Any]:
    """Every draft model this process holds (R2: never a victim). A spec
    worker whose ``model_runner`` IS the target's (ngram) names no draft."""
    target = _runner(scheduler).model
    out = []
    dw = getattr(scheduler, "draft_worker", None)
    for path in (("model_runner", "model"), ("draft_model_runner", "model"), ("draft_runner", "model")):
        obj = dw
        for attr in path:
            obj = getattr(obj, attr, None) if obj is not None else None
        if obj is not None and obj is not target and all(obj is not o for o in out):
            out.append(obj)
    return out


def _arena_ranges() -> List[Tuple[int, int]]:
    try:
        from sglang.srt.model_executor.dual_stage_hull import _arena_ranges as ranges

        return list(ranges())
    except Exception:  # noqa: BLE001 -- no union module: nothing attached
        return []


def _dual(scheduler) -> bool:
    from sglang.srt.model_executor.dual_stage_hull import dual_share_armed

    return dual_share_armed() and getattr(_runner(scheduler), "dual_share_part_models", None) is not None


def _build_pp_only(scheduler) -> PpOnlyVictims:
    r = _runner(scheduler)
    return PpOnlyVictims(r.dual_share_part_models, int(r.dual_share_local_part),
                         exclude=_draft_models(scheduler), arena_ranges=_arena_ranges())


def _build_dense(scheduler) -> DenseMlpVictims:
    return DenseMlpVictims(_runner(scheduler).model, exclude=_draft_models(scheduler),
                           arena_ranges=_arena_ranges())


vv.register_source(KIND_PP_ONLY, _dual, _build_pp_only)
vv.register_source(KIND_DENSE, lambda s: not _dual(s), _build_dense)
