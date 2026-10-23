# SPDX-License-Identifier: Apache-2.0
"""S35 (18.10.): an ADMITTED request's result is the group's on a Form A group (R11, raenge-nie-uneins).

THE DEATH (fLLiper rc9 = h16v7 f1011c1a90, boot ...dauer10180915, D 09:51:22Z -> DEADMAN_PROGRESS_STALL 09:53:19Z).
pdflip-29-190 (an E2 END-state hand-off) was admitted on all three D ranks: the attention host's H105 verdict ADMIT
reached every rank, HANDBACK path=skip and TAIL-READY on TP0, TP1, TP2. ``PrefillAdder.add_one_req`` then returns
``self.budget_state()`` -- THIS rank's own budget (``rem_total_tokens`` = its available + evictable, ``cur_rem_tokens``,
``rem_mamba_slots``). On a Form A group those are rank-local by design (H105: "the budget gates are RANK-LOCAL ...
the attention host's verdict is taken"), and the host's pool is the binding one: the host alone read NO_TOKEN for the
request it had just admitted, the workers read CONTINUE. The admission loop treats a NO_TOKEN as the pass's D-HOL head
(``hol_overtake.HolPass.may_overtake``) and the SEAT-AGE note (``_pdflip_sa_no_token``); after the loop the HOL head
enters PW-R's group MIN (``d_wall_head.pick_unservable_head`` -> ``Scheduler._pdflip_group_min_flags``). TP0 entered
that all_reduce alone: it logged X-GATE for the next request (it scanned on, ``_hol_go_on``), then nothing -- its
waiting queue still held pdflip-29-190 ("3 queued" against the workers' "2 queued"), i.e. it never reached the
queue removal behind the PW-R site, while the workers passed the extend-set riegel, ran the skip extend and stood at
their next collective. Nobody named it: the riegel compares the extend SETS, which were equal.

THE RULE: on a Form A follow group (``PrefillAdder.form_a_admission_follow`` set) an admitted request (it is the
last entry of ``can_run_list``) never reads NO_TOKEN in the loop -- the admit was the group's verdict, and whether
the NEXT request fits is that request's own H105 verdict (the host's gate, broadcast / gathered, identical on every
rank). So it reads CONTINUE on every rank; a refusal (not added) and every other result stay as they are. Off a Form
A group the budgets are the group's already (``_update_uniform_pool_budget``): unchanged.

CORRECTION (RA 23.10., 27B duo boot ..._114625): off a Form A group only the FLOOR term of ``rem_total_tokens`` is
the group's; the local term (available + deliverable evictable) can bind on one rank alone -- TP0 admitted
pdflip-36-127, TP1 refused it, PW-R's MIN was entered by TP1 alone. The plain-TP group now has its own group verdict,
riegel and this same post-admit rule: ``managers/tp_admission_agree.py``.
"""
from __future__ import annotations

from typing import Any

MARK = "S35 FORM-A POST-ADMIT"


def uniform_result(res: Any, added: bool, form_a_follow: Any) -> Any:
    """The loop's result for a request ``add_one_req`` returned ``res`` for. ``added`` = the request is the last entry
    of ``can_run_list`` (the loop's own ``added`` test); ``form_a_follow`` = the adder's H105 follow callable (None
    off a Form A group). NO_TOKEN of an ADDED request on a Form A group -> CONTINUE; everything else unchanged."""
    if form_a_follow is None or not added:
        return res
    if getattr(res, "name", None) != "NO_TOKEN":
        return res
    return type(res).CONTINUE
