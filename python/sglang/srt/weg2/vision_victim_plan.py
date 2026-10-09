# SPDX-License-Identifier: Apache-2.0
"""VISION-WEIGHTS AP4 (plan PLAN-VISION-GEWICHTE-VERDRAENGEN-1009 section 8): the PLANNER side of ``--weg2-vision-place weights``.

PURE (stdlib only).  The runtime (``vision_victim.py``, ``vision_victim_27b.py``, AP1/AP2) borrows weight memory of PP0 for the
transient vision tower and gives it back after every image.  This module states, before any boot, what that means in numbers:

* ``Vision transient`` -- a sub-item of the WEIGHTS segment of the PP0 card: the victim bytes the tower sits on (at least the tower
  bytes).  It is a CHECK item, not a cost: the victims are weights that are already counted, so the resident part is 0 and the sum of
  the weights does not grow (plan section 8, "Pruefposten").  Only with ``--weg2-vision transient`` AND ``--weg2-vision-place weights``.
* ``Displaced weights, temporary`` -- the HOST item: the victim bytes that wait in a host image while the tower is up (27B: the tower
  bytes, anonymous pageable memory, allocated per image and freed after it; NF: 0, the victim rows already have their copy in the expert
  store).  After the give-back it is 0.  Host RAM is a WARNING only (user rule 2026-10-06), never a refusal.
* the verdict ``W105b Weg2VisionVictimShort`` -- the victims on PP0 hold fewer bytes than the tower.  A VERDICT with the code and the
  numbers, never a lock (oracle rule of the plan): the runtime would refuse the image request by name, the rig stays intact.

The victim kind is derived from the form exactly as the runtime does (``vision_victim_27b.KIND_*``, plan section 5): 27B flip =
``dense`` (MLP storages of PP0), 27B dual = ``pp_only`` (the hull parts that are not D's shard on this card), NF = ``experts``
(resident expert rows of PP0; zero extra host RAM).

Numbers and their source:

* 27B tower = 921 460 192 B = 878.77 MiB = "878.8 MiB": 333 tensors, sum of the safetensors headers of ``Qwen3.8-27B-NVFP4-RadixArk``
  and ``Qwen3.8-27B-INT8-gdncov`` (measured on this box 2026-10-09, identical in both), the largest tensor ``merger.linear_fc2`` 45.0 MiB.
  It is the number of the AP2 commit c685baaf8f ("333 Tensoren 878,8 MiB") and of the plan (section 0), and the 27B tower of
  ``test_weg2_vision_victim_27b.py`` (``_merger_tower``, line 213: the real merger shapes) is built on its two largest tensors.
  The model profile reads the same bytes (``weights.visual_bytes``, source "Index"); the constant is only the fallback and the reference
  of the tests.
* NF tower = 897 862 112 B = 856.3 MiB (plan section 0; header of the NF checkpoint, NOT on this box: unverified here).

Nothing here guesses a size: a missing input is ``None`` with the reason, the verdict is then ``ungeprueft``.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

SCHEMA = "flliper.vision-victim/1"
MIB = 1024.0 * 1024.0

PLACE_FLAG = "--weg2-vision-place"
VISION_FLAG = "--weg2-vision"
PLACE_ENV = "SGLANG_WEG2_VISION_PLACE"
PLACE_WEIGHTS = "weights"
VISION_TRANSIENT = "transient"

KIND_DENSE = "dense"
KIND_PP_ONLY = "pp_only"
KIND_EXPERTS = "experts"

#: codes of the runtime (weg2/vision_victim.py:77-79) -- one spelling
W_SHORT = "W105b Weg2VisionVictimShort"
W_NOT_RESTORED = "W110c Weg2VisionVictimNotRestored"
W_PLAN_REFUSED = "W111b Weg2VisionVictimPlanRefused"

#: 27B tower (see the module docstring): 333 tensors, sum of the safetensors headers
TOWER_BYTES_27B = 921460192
TOWER_MIB_27B = 878.8          # = round(TOWER_BYTES_27B / MIB, 1)

#: planner labels (English like the rest of the planner; the German names of the plan are "Vision transient" and "verdraengte Gewichte,
#: temporaer")
LABEL_TRANSIENT = "Vision transient"
LABEL_HOST = "Displaced weights, temporary (host)"
NAME_TRANSIENT = "vision_transient"
NAME_HOST = "vision_victim_host"


def active(vision: Any, place: Any) -> bool:
    """The planner items exist only for the transient tower on displaced weights."""
    return str(vision or "") == VISION_TRANSIENT and str(place or "") == PLACE_WEIGHTS


def victim_kind(*, is_moe: bool, dual: bool) -> str:
    """The victim kind of the runtime for this form (plan section 5): NF experts, 27B dual pp_only, 27B flip dense."""
    if is_moe:
        return KIND_EXPERTS
    return KIND_PP_ONLY if dual else KIND_DENSE


def tower_bytes_from_checkpoint(model_dir: str) -> Tuple[Optional[int], str]:
    """``(bytes, source)`` of the tower tensors of a sharded safetensors checkpoint (index + header of the one shard); ``(None, reason)``
    when it cannot be read.  Header only, no weight byte is read (stdlib; the same selector as ``planner.vision_stage_load``)."""
    try:
        from sglang.srt.planner.vision_stage_load import find_tower_shard, is_vision_weight

        shard = find_tower_shard(model_dir)           # already joined with model_dir
        with open(shard, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(hlen))
        total, n = 0, 0
        for name, meta in header.items():
            if name == "__metadata__" or not is_vision_weight(name):
                continue
            s, e = (int(x) for x in meta["data_offsets"])
            total += e - s
            n += 1
        if n == 0:
            return None, "no tower tensor in the header of %s" % shard
        return total, "safetensors header %s (%d tensors)" % (os.path.basename(shard), n)
    except Exception as exc:  # noqa: BLE001 -- unmeasured is said, never guessed
        return None, "tower size unread (%s: %s)" % (type(exc).__name__, exc)


def available_dense_mib(modell: Mapping[str, Any], stage0_layers: Optional[int]) -> Tuple[Optional[float], str]:
    """27B flip: the MLP bytes of the PP0 stage = (MLP bytes of the model / layers) x layers of stage 0.  The mean per layer is an
    approximation (every layer of the dense 27B has the same MLP shapes; the model profile keeps only the sum per role)."""
    try:
        w = modell["weights"]
        role = w["per_role_bytes"]
        role = role["v"] if "v" in role and isinstance(role["v"], dict) else role
        mlp = float(role["mlp"])
        n_layers = int(modell["arch"]["n_layers"]["v"] if isinstance(modell["arch"]["n_layers"], dict) else modell["arch"]["n_layers"])
    except (KeyError, TypeError, ValueError):
        return None, "the model profile carries no MLP bytes per role"
    if not stage0_layers or n_layers <= 0:
        return None, "the layer count of stage 0 is not known"
    return mlp / n_layers * int(stage0_layers) / MIB, ("MLP bytes of the model profile (per_role_bytes.mlp) / %d layers x %d layers of stage 0 "
                                                       "(mean per layer: approximation)" % (n_layers, int(stage0_layers)))


def available_experts_mib(layer_expert_mib: Sequence[float], stage0_layers: Optional[int], fraction0: Optional[float]) -> Tuple[Optional[float], str]:
    """NF: resident expert rows of PP0 = expert bytes of the stage 0 layers x the resident fraction of stage 0 (approximation: the launcher
    rounds to whole rows)."""
    if not layer_expert_mib or not stage0_layers or fraction0 is None:
        return None, "expert bytes / resident fraction of stage 0 not known"
    mib = float(sum(float(x) for x in list(layer_expert_mib)[: int(stage0_layers)])) * float(fraction0)
    return mib, "expert bytes of the first %d layers x resident fraction %.3f of stage 0 (approximation: whole rows at the launcher)" % (int(stage0_layers), float(fraction0))


def section(*, form: str, vision: Any, place: Any, is_moe: bool, dual: bool,
            tower_bytes: Optional[float], tower_src: str,
            available_mib: Optional[float], available_src: str) -> Dict[str, Any]:
    """The ``vision`` section of a proposal / bar: items, host item and the W105b verdict.  Inactive (not transient+weights) -> only the
    switch state, nothing else (the default profiles carry no new item)."""
    if not active(vision, place):
        return {"schema": SCHEMA, "aktiv": False, "vision": str(vision or ""), "place": str(place or "")}
    kind = victim_kind(is_moe=is_moe, dual=dual)
    out: Dict[str, Any] = {"schema": SCHEMA, "aktiv": True, "form": form, "vision": VISION_TRANSIENT, "place": PLACE_WEIGHTS, "opferart": kind}
    if form == "tp":
        out["aktiv"] = False
        out["hinweis"] = "--d-only has no P group: the vision stage is not part of this form"
        return out
    if kind == KIND_PP_ONLY:
        # AP2 desk arithmetic (test_weg2_vision_victim_27b.py:226-228, parametrized on 12 MiB pp_only runs): the two merger linears (40.5 / 45.0 MiB)
        # exceed the largest pp_only tensor (about 12 MiB) and are row-split. An expectation of the plan, NOT a measurement: the arming line of the
        # first boot (M0) logs the real ``split_tensors`` and ``largest_run_mib``.
        out["split_tensors_erwartet"] = 2
        out["split_tensors_quelle"] = ("AP2 desk arithmetic with 12 MiB runs (test_weg2_vision_victim_27b.py:226-228); unverified on the metal: "
                                       "the M0 arming line logs the real split_tensors")
    tower_mib = None if tower_bytes is None else round(float(tower_bytes) / MIB, 1)
    out["turm_mib"] = tower_mib
    out["turm_quelle"] = tower_src
    out["resident_mib"] = 0.0
    out["vision_transient"] = {
        "name": NAME_TRANSIENT, "label": LABEL_TRANSIENT, "mib": tower_mib, "resident_mib": 0.0, "in_summe": False, "transient": True,
        "detail": ("Victim weights of PP0 (%s) the tower is mapped onto while an image is encoded; the victims are weights that are already "
                   "counted, so this adds 0 MiB resident and nothing to the sum of the weights. At least the tower bytes: moved bytes = tower "
                   "bytes (plan section 2)." % kind),
        "herkunft": tower_src, "gerechnet": tower_mib is not None}
    host_mib = 0.0 if kind == KIND_EXPERTS else tower_mib
    out["host_mib"] = host_mib
    out["host_posten"] = {
        "name": NAME_HOST, "label": LABEL_HOST, "mib": host_mib, "transient": True, "nach_rueckholung_mib": 0.0,
        "detail": ("NF: 0, the victim expert rows have their copy in the expert store (already booked); the give-back is load_refill_rows."
                   if kind == KIND_EXPERTS else
                   "The victim bytes (= the tower bytes) wait in an anonymous pageable host image while the tower is up; allocated per image, "
                   "freed after the give-back (0 afterwards). Host RAM is a warning only, never a refusal (user rule 2026-10-06)."),
        "gerechnet": host_mib is not None}
    out["verfuegbar_mib"] = None if available_mib is None else round(float(available_mib), 1)
    out["verfuegbar_quelle"] = available_src
    out["verdikt"] = verdict(kind, tower_mib, out["verfuegbar_mib"], available_src)
    return out


def verdict(kind: str, tower_mib: Optional[float], avail_mib: Optional[float], avail_src: str) -> Dict[str, Any]:
    """``W105b`` as a verdict with code and numbers: ``ja`` (victims >= tower), ``nein`` (short), ``ungeprueft`` (an input is unknown)."""
    what = {KIND_DENSE: "MLP storages of the PP0 stage (27B flip)", KIND_PP_ONLY: "pp_only hull parts of the PP0 card (27B dual: not D's shard)",
            KIND_EXPERTS: "resident expert rows of PP0 (NF)"}[kind]
    if tower_mib is None or avail_mib is None:
        why = "tower size" if tower_mib is None else "victim bytes"
        return {"code": "W105b", "stufe": "ungeprueft",
                "text": "%s: not calculated, %s unknown (%s)" % (W_SHORT, why, avail_src if tower_mib is not None else "tower size unread"),
                "turm_mib": tower_mib, "verfuegbar_mib": avail_mib}
    if avail_mib + 1e-6 < tower_mib:
        return {"code": "W105b", "stufe": "nein", "turm_mib": tower_mib, "verfuegbar_mib": avail_mib, "fehlt_mib": round(tower_mib - avail_mib, 1),
                "text": ("%s: the victims (%s) hold %.1f MiB, the tower needs %.1f MiB (%.1f MiB short). The image request would be refused "
                         "by name with 503, text keeps running, no fallback to KV or free VRAM." % (W_SHORT, what, avail_mib, tower_mib, tower_mib - avail_mib))}
    return {"code": "W105b", "stufe": "ja", "turm_mib": tower_mib, "verfuegbar_mib": avail_mib, "rest_mib": round(avail_mib - tower_mib, 1),
            "text": ("%s: not triggered, the victims (%s) hold %.1f MiB, the tower needs %.1f MiB (rest %.1f MiB)" % (
                W_SHORT, what, avail_mib, tower_mib, avail_mib - tower_mib))}
