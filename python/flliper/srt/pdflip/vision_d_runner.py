# SPDX-License-Identifier: Apache-2.0
"""The D group's transient vision stage in the flip layouts (user order 27.10.: "vision natürlich
in D und in P, für die flip layouts, vision nur auf P in dual layout"; "die 5090er encoded").

The wiring half of ``vision_d_pass`` (the pure half): arming and the scheduler pass.

* Arming (``Scheduler.__init__``): group D, and the launcher published ``FLLIPER_PDFLIP_VISION_D=1`` (flip layouts only;
  ``launcher.vision_on_d_of`` is the ONE decision the front reads too) and the model has a vision config. Every TP rank
  takes part in the pass; TP0 (the 5090) is the leader and alone arms the victim source (the MLP storages of its own D
  shard) and the standing pinned tower copy -- exactly as PP0 does on P.
* WHO is a candidate (review 27.10. H1/H3): only a request the D verdict marked (``STAGE_WANTED_ATTR``): the GROUP's covered
  prefix -- after the store read of P's leg and after the VISION-ROWS attempt -- does not reach its image, the request has
  sent no byte yet (so a refusal can go back through P) and this group has a stage. The verdict is group-uniform by
  construction, so every rank plans the same items; no rank-local prefix match decides anything here.
* The pass runs right before the admission, on every rank, like ``vision_rank_pass`` on P: the leader runs the in-rank
  stage with the weights lease on its card and the tower built UNSHARDED (``use_data_parallel=True``: the group is TP3,
  the tower must not be), and the rows reach the other ranks over the TP cpu group before anything is admitted. D's decode
  stands for the duration of the pass (VISION-SYNC LAW: the lease lives only inside one synchronous pass), which is the
  price of an image instead of a flip to P.
* A refused pass holds the requests on every rank (``_pdflip_vision_refused``) and the leader sends
  the abort to the origin (``_pdflip_vision_origin_aborts``), as on P.

Kill switch: ``FLLIPER_PDFLIP_VISION_D=0`` (default on).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from flliper.srt.pdflip import vision_d_pass as vdp
from flliper.srt.pdflip import vision_rank_runner as vrr
from flliper.srt.pdflip import vision_rank_stage as vrs

logger = logging.getLogger(__name__)

D_ENV = "FLLIPER_PDFLIP_VISION_D"


def _on(value: str) -> bool:
    return (value or "").strip().lower() not in ("0", "false", "no", "off")


def d_stage_wanted(hf_config: Any, env: Optional[Dict[str, str]] = None) -> bool:
    """Group D, the launcher published the decision (``launcher.vision_on_d_of`` -> ``FLLIPER_PDFLIP_VISION_D=1``, flip
    layouts only, kill switch ``FLLIPER_PDFLIP_VISION_D=0``), and the model has a vision config. Nothing is inferred from
    other variables: the launcher pops FLLIPER_PDFLIP_VISION for D and publishes the dual layout only in the dual env."""
    e = os.environ if env is None else env
    if (e.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper() != "D":
        return False
    if (e.get(D_ENV, "") or "").strip() != "1":
        return False
    return getattr(hf_config, "vision_config", None) is not None


def d_leg_without_pixels(model_config: Any, env: Optional[Dict[str, str]] = None) -> bool:
    """A group D whose legs need no pixels: tower-less (``vision_d_guard.d_guard_armed``) AND no stage of its own. With the
    stage (:func:`d_stage_wanted`) the leg's items must keep their features -- the stage encodes from them; feature-less
    items there are planned by nobody and the request waits for ever (27.10. boot formb98v, pdflip-2-2)."""
    from flliper.srt.pdflip.vision_d_guard import d_guard_armed

    return d_guard_armed(model_config, env) and not d_stage_wanted(getattr(model_config, "hf_config", None), env)


def arm_d_stage(scheduler: Any, env: Optional[Dict[str, str]] = None) -> bool:
    """Arm this TP rank of group D. Returns whether the rank takes part in the pass. An arming
    refusal of the leader still arms the pass: it then refuses every image by name."""
    hf_config = getattr(getattr(scheduler, "model_config", None), "hf_config", None)
    if not d_stage_wanted(hf_config, env):
        return False
    e = os.environ if env is None else env
    leader = int(scheduler.ps.tp_rank) == 0
    scheduler._pdflip_vision_d_leader = leader
    scheduler._pdflip_vision_place = vrs.PLACE_WEIGHTS
    scheduler._pdflip_vision_source = None
    scheduler._pdflip_vision_victims = None
    scheduler._pdflip_vision_arm_refusal = ""
    scheduler._pdflip_vision_origin_aborts = []
    scheduler._pdflip_vision_refused = set()
    scheduler._pdflip_vision_runs = 0
    if leader:
        scheduler._pdflip_vision_victims, refusal = vrr.arm_victims(scheduler, e)
        if not refusal and vrs.pinned_on(e):
            scheduler._pdflip_vision_source = vrr.arm_pinned_source(vrr.tower_dir(scheduler.server_args))
        scheduler._pdflip_vision_arm_refusal = refusal
        if refusal:
            logger.error("%s -- the D vision stage on TP0 is NOT armed: %s. Image requests to this "
                         "group are aborted by name (%s).", vrr.W_ARM_REFUSED, refusal, vrr.W_NOT_ARMED)
        else:
            logger.info("%s ARMED on D: TP0 pid=%d leads (the card's own MLP victims, tower unsharded, "
                        "rows broadcast to the %d TP ranks before the admission)", vrr.W_STAGE_OK,
                        os.getpid(), int(scheduler.ps.tp_size))
    return True


class TpComm:
    """The two collectives over group D's TP cpu group (gloo); the leader is the group's first rank."""

    def __init__(self, scheduler: Any):
        self._group = scheduler.tp_cpu_group
        self._src = int(scheduler.tp_group.first_rank)
        self._rank = int(torch.distributed.get_rank())

    def broadcast_obj(self, obj: Any) -> Any:
        from flliper.srt.utils import broadcast_pyobj

        return broadcast_pyobj([obj], self._rank, self._group, src=self._src)[0]

    def broadcast_rows(self, rows: torch.Tensor) -> torch.Tensor:
        torch.distributed.broadcast(rows, src=self._src, group=self._group)
        return rows


def _leader_stage(scheduler: Any, reqs: List[Any]) -> Tuple[bool, str, str]:
    """(ok, code, detail) of the leader's stage for ``reqs``; the rows are attached to their items.
    A W110c (victims not restored) is raised, as on P: the group stops here."""
    refusal = scheduler._pdflip_vision_arm_refusal
    if refusal:
        logger.error("%s rids=%s -- %s", vrr.W_NOT_ARMED, [r.rid for r in reqs], refusal)
        return False, vrr.W_NOT_ARMED, refusal
    scheduler._pdflip_vision_runs += 1
    dev = vrr._rank_device()
    try:
        out = vrr.run_rank_stage(
            scheduler, reqs, model_dir=vrr.tower_dir(scheduler.server_args),
            hf_config=scheduler.model_config.hf_config, device=dev,
            build=lambda hf, d: vrr.build_tower_meta(hf, d, use_data_parallel=True),
            place=vrs.PLACE_WEIGHTS, source=scheduler._pdflip_vision_source,
            victims=scheduler._pdflip_vision_victims)
        if dev.type == "cuda":
            out.card = vrr._nvml_index_of(dev.index if dev.index is not None else 0)
    except Exception as exc:  # noqa: BLE001 -- a stage never kills the group
        logger.exception("%s: the D stage raised outside its own verdict", vrr.W_LOAD)
        return False, vrr.W_LOAD, f"stage: {type(exc).__name__}: {exc}"
    vrr.log_outcome(out, [r.rid for r in reqs], scheduler._pdflip_vision_runs)
    if out.fatal:
        from flliper.srt.pdflip.vision_victim import VisionVictimNotRestored

        raise VisionVictimNotRestored(
            f"{out.fatal} -- rids={[r.rid for r in reqs]}; the D group stops here, the watchdog/launcher "
            "restarts it")
    return out.ok, out.code, out.detail


def _requests(entries) -> List[Any]:
    seen, out = set(), []
    for r, _i, _it in entries:
        if r.rid not in seen:
            seen.add(r.rid)
            out.append(r)
    return out


#: the request attribute the scheduler's D verdict sets (``Scheduler._pdflip_vision_d_verdict``): the GROUP's covered prefix
#: (after the store read and the VISION-ROWS attempt) does not reach the image and D has a stage -- the only candidates
STAGE_WANTED_ATTR = "_pdflip_vision_d_stage_wanted"


def _wanted_items(req: Any) -> List[Any]:
    """The unstaged items of a request the D verdict marked. Review H1/H3: the candidate set comes from the group-uniform
    verdict AFTER the store read -- never from a rank-local prefix match -- so every rank plans the same items and P's
    already-prefilled images are not encoded a second time."""
    return vrr.unstaged_items(req) if getattr(req, STAGE_WANTED_ATTR, False) else []


def _mark_staged(reqs: Sequence[Any]) -> None:
    """The rows are attached: the request is no longer a candidate, and the verdict's bookkeeping of the refusal it just
    avoided goes (review M2: a stale ``need``/``tail_lost`` would keep the admission loop from ever admitting it)."""
    for r in reqs:
        setattr(r, STAGE_WANTED_ATTR, False)
        r._pdflip_vision_need = 0
        r._pdflip_vision_tail_lost = False


def d_pass(scheduler: Any, *, comm: Any = None,
           stage: Callable[[Any, List[Any]], Tuple[bool, str, str]] = _leader_stage) -> List[Tuple[int, Any]]:
    """Right before D's admission, on EVERY TP rank. Returns the requests held out of THIS pass as
    (index, req) -- the refused ones, until their abort lands -- the caller puts them back with
    ``vision_unpark``. Collectives happen only when a marked candidate exists (the marks come from the
    group-uniform verdict, so that is a rank-uniform condition)."""
    if getattr(scheduler, "pdflip_dormant", False):
        return []
    wq = scheduler.waiting_queue
    refused = scheduler._pdflip_vision_refused
    refused.intersection_update({r.rid for r in wq})
    held = set(refused)
    entries = vdp.plan(wq, unstaged_items=_wanted_items, refused=refused)
    if entries:
        held |= _run_pass(scheduler, entries, comm if comm is not None else TpComm(scheduler), stage)
    if not held:
        return []
    parked = [(i, r) for i, r in enumerate(wq) if r.rid in held]
    for i, _ in reversed(parked):
        wq.pop(i)
    return parked


def _run_pass(scheduler: Any, entries, comm: Any, stage) -> set:
    from flliper.srt.planner.vision_stage_load import attach_precomputed_embeddings

    leader = scheduler._pdflip_vision_d_leader
    sel = comm.broadcast_obj(vdp.selection_of(entries) if leader else None)
    chosen = vdp.resolve(entries, sel)
    held: set = set()
    reqs = _requests(chosen)
    packet, rows = (None, [])
    if leader:
        ok, code, detail = stage(scheduler, reqs)
        packet, rows = vdp.leader_packet(chosen, ok=ok, code=code, detail=detail)
    got, rows = vdp.exchange(chosen, packet, rows, is_leader=leader, comm=comm)
    if got.ok:
        if not leader:
            vdp.attach_followers(chosen, rows, attach=attach_precomputed_embeddings, width=got.width)
        _mark_staged(reqs)
        return held
    for r in reqs:
        scheduler._pdflip_vision_refused.add(r.rid)
        held.add(r.rid)
        if leader:
            scheduler._pdflip_vision_origin_aborts.append((r.rid, _refusal_message(r, got.code, got.detail)))
    return held


def _refusal_message(req: Any, code: str, detail: str) -> str:
    """A request that has sent no byte yet goes back through P (the front's X-REQUEUE, keyed on the
    W50 text); one that has streamed keeps the stage's named code (RESUME-VIA-P carries no image)."""
    from flliper.srt.pdflip import vision_d_guard as vdg

    if vdg.reroute_eligible(req):
        return vdg.stage_reroute_message(req, code, detail)
    return f"{code}: {detail}"
