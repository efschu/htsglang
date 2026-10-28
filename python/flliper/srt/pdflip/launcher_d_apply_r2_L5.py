"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from flliper.srt.pdflip import budget_rest as _budget_rest
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip.launcher_d_leaf import d_card_vote_lines
from flliper.srt.pdflip.launcher_d_mid import D_RANK_SOLVE_MARKER
from flliper.srt.pdflip.launcher_d_p_r2_L2 import argv_p
from flliper.srt.pdflip.launcher_d_p_r2_L4 import _d_extend_vote_fixed_live_env
from flliper.srt.pdflip.launcher_d_p_r2_L4 import _max_running_requests
from flliper.srt.pdflip.launcher_d_p_r2_L4 import d_extend_vote_fixed_record
from flliper.srt.pdflip.launcher_log_plan import Log
from flliper.srt.pdflip.launcher_p_d_r2_L3 import RING_FORM_SENTINEL_STORE_CFG
from flliper.srt.pdflip.launcher_p_l3_leaf import parse_group_env
from flliper.srt.pdflip.launcher_p_mid import P_MAMBA_AUTO_SAFETY_MARGIN
from flliper.srt.pdflip.launcher_refuse_proc_leaf import set_group_env
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_xchg_vision_leaf import _without_budget_flag
import math
import shlex

#: Mamba state SLOTS one running request can hold at once -- the runtime's
#: ``mamba_slots_per_running_req`` (mem_cache/mamba_pool_floor.py), which is
#: also the ceiling on ``_calculate_mamba_ratio`` (:2684 takes the ``min`` of
#: the two). Boot weg2sb5f emits both: ``[auto-mamba] target_concurrency=8
#: ratio=2 safety=1.25 -> max_mamba_cache_size=20 slots``.
#:
#: WHY THIS REPLACED A `2.5` FUDGE FACTOR (#1286 F4). The old constant folded
#: "ratio 2" and "safety 1.25" into one number and multiplied ``--p-bs`` by it,
#: which reproduced weg2sb5f and nothing else: ``_auto_mamba_demand_size``
#: (:2085-2106) is ``max(ceil(target * ratio * safety), ratio,
#: mamba_hard_floor(sa, target))`` and the third term -- ``target *
#: slots_per_running_req`` -- was simply not in the model. A boot whose ratio
#: branch differs (no overlap schedule, no ack-release, radix cache off) or
#: whose hard floor binds takes MORE slots than 2.5x, the model then charges
#: LESS mamba than the boot, and the pool is OVER-priced: #1286's own defect
#: recurring on a different term.
#:
#: The expression below is a PROVEN upper bound on the runtime's slot count
#: rather than a re-derivation of its branches, which is why it needs one
#: number and not the whole ServerArgs surface. With ``p = slots per running
#: request`` and ``t = target``: ``ratio = min(base, p) <= p``, so
#: ``ceil(t*ratio*1.25) <= ceil(t*p*1.25)``; the hard floor is exactly ``t*p``,
#: and ``t*p <= ceil(t*p*1.25)``; ``ratio <= t*p`` likewise. Hence
#: ``_auto_mamba_demand_size <= ceil(t * p * MAMBA_AUTO_SAFETY_MARGIN)``. An
#: upper bound on SLOTS is an over-charge of mamba MiB, i.e. an UNDER-price of
#: the pool -- the conservative direction, and the opposite of the one that
#: cost weg2sb5f its floor. At p=2, t=8 it is ceil(20.0) = 20, the boot's own
#: number, so the bound is TIGHT on the form weg2sb5f ran and not merely safe.
#:
#: THAT ANCHOR IS A RECORD, NOT TODAY'S FORM: weg2sb5f booted at t=8, and the
#: shipped default moved to ``DEFAULT_P_BS`` by the user order of 2026-09-09.
#: The bound is proven for every t (the derivation above uses no value of t),
#: so nothing here needed changing -- but the tightness claim is about that
#: boot's t, not about the default, and re-reading it as "tight at the shipped
#: bs" would be the instrument-text trap. The number the launcher actually
#: charges is ``ceil(--p-bs x this x safety)`` and is printed with its inputs
#: in the PP-CUT budget posts line, where --p-bs appears as a resolved value.
P_MAMBA_SLOTS_PER_RUNNING_REQUEST = 2

def booked_rest_vote_line_deltas(cards: List[Card], rests: Sequence[Optional[int]], name: str,
                                 profile: Optional[str] = None, group: str = "D",
                                 log: Optional[Log] = None) -> List[int]:
    """PDFLIP-VOTE-LINE-BOOK (desk/27b-int8-tp1corridor-1024, pdflip/budget_rest.py):
    per card the MiB the extend width vote now keeps free beyond the line the booked
    rest ``name`` was measured under -- ``ceil8(max(0, line_for_floor_mib(floor) -
    record_line))`` where the card's corridor floor is MEASURED and its footprint is of
    this form (reserve 0, as the booked path charges none; an unmeasured, hand-set or
    foreign-form floor books nothing), 0 elsewhere. DERIVED, not measured: the record
    under-books the current line, the caller books the delta beside it, so the vote's
    line comes out of the KV pool, not out of the chunk."""
    n = len(cards)
    if not _budget_rest.vote_line_switch_on():
        if log is not None:
            log(f"{_budget_rest.VOTE_LINE_MARKER} group={group}: {_budget_rest.VOTE_LINE_ENV}=0 -- no delta "
                f"booked (the KV pool keeps the vote's room, the vote narrows the chunk)")
        return [0] * n
    prof = profile if profile is not None else pdflip_form.DEFAULT_PROFILE
    row = pdflip_form.profile_row(prof)
    rec = row.constants.get(name) if row is not None else None
    boots = tuple(getattr(rec, "boots", ()) or ()) if rec is not None else ()
    rec_line = _budget_rest.record_vote_line(name, boots)
    if rec_line is None:
        if log is not None:
            log(f"{_budget_rest.VOTE_LINE_MARKER} group={group}: the vote line {name} was measured "
                f"under is unknown (boots {','.join(boots) or '-'}) -- no delta booked (named, not "
                f"guessed); re-measured under the current line the rest already contains it")
        return [0] * n
    from flliper.srt.pdflip.extend_trim import near_oom_mib

    # VOTE-LINE-SINGLE: on a measured floor the line is near-OOM + the measured fixed part (the same record the
    # ranks vote with, d_extend_vote_fixed_env) -- the floor is the activation rate x rows already prices
    fixed, fixed_src = d_extend_vote_fixed_record(profile, n, cards)
    if fixed is None and log is not None and (fixed_src.startswith("F unmeasured") or "name no form" in fixed_src):
        log(f"{_budget_rest.VOTE_LINE_MARKER} group={group}: {_budget_rest.VOTE_FIXED_MARKER} off -- {fixed_src}; "
            f"the delta books the floor line (review M1 form gate)")
    floors, lines, act, form_why = d_card_vote_lines(cards, group, fixed)
    deltas = _budget_rest.vote_line_deltas(rests, lines, act, rec_line)
    if log is not None:
        for i, c in enumerate(cards):
            cf = floors[c.uuid]
            reason = ""
            if rests[i] is None:
                reason = " -> 0: no booked rest on this card"
            elif form_why[i] is not None:
                reason = f" -> 0: the floor is not of this form ({form_why[i]})"
            elif not act[i]:
                reason = " -> 0: the floor is not measured (an unmeasured or hand-set law cuts no KV for the vote line)"
            if fixed is not None and act[i]:
                what = (f"{_budget_rest.VOTE_FIXED_MARKER}: near-OOM {near_oom_mib():.0f} + fixed "
                        f"{fixed[i]:.0f} ({_budget_rest.VOTE_FIXED_RECORD}, {fixed_src}); floor {cf.mib} "
                        f"source={cf.source} is the activation rate x rows prices, not held twice")
            else:
                what = f"max(300, floor {cf.mib} source={cf.source})"
            log(f"{_budget_rest.VOTE_LINE_MARKER} group={group} ordinal={i} nvml{c.nvml_index}: "
                f"delta {deltas[i]} MiB {_budget_rest.VOTE_LINE_BASIS} = ceil8(vote line {lines[i]:.0f} "
                f"({what}) - record line {rec_line} ({name}, rc12g halt line of its boots))" + reason)
    return deltas

def _d_extend_vote_fixed_env(ns, log, label: str, cards: Optional[Sequence[Card]] = None) -> Optional[str]:
    """VOTE-LINE-SINGLE: wherever the launcher armed the D width vote, hand the ranks the measured fixed part
    (``FLLIPER_PDFLIP_EXTEND_VOTE_FIXED_MIB`` from ``D_EXTEND_VOTE_FIXED_MIB``): a rank whose corridor floor is
    MEASURED then votes with ``near-OOM 400 + fixed`` and trims before the forward under ``line + rate x rows``
    (extend_trim.corridor_line_mib / need_trim_threshold) -- the line the budget books
    (:func:`booked_rest_vote_line_deltas`). A value named in --env-d wins; no record / switch off = nothing written."""
    vals, src = d_extend_vote_fixed_record(getattr(ns, "profile", None), None if cards is None else len(cards),
                                           cards, getattr(ns, "model", None))
    _d_extend_vote_fixed_live_env(ns, log, label, cards, src)
    if vals is None:
        if src.startswith("F unmeasured") or "name no form" in src:
            log(f"{D_RANK_SOLVE_MARKER} {label} {_budget_rest.VOTE_FIXED_MARKER} off: {src} -- no "
                f"{_budget_rest.VOTE_FIXED_ENV}, the ranks keep the floor line (review M1 form gate)")
        return None
    text = ",".join(f"{v:g}" for v in vals)
    _env_d = getattr(ns, "env_d", "") or ""
    _have = parse_group_env(_env_d)
    _given = _budget_rest.VOTE_FIXED_ENV in _have and str(_have.get(_budget_rest.VOTE_FIXED_ENV, "")).strip() != text
    if not _given:
        ns.env_d = set_group_env(_env_d, _budget_rest.VOTE_FIXED_ENV, text)
    _now = str(parse_group_env(getattr(ns, "env_d", "") or "").get(_budget_rest.VOTE_FIXED_ENV, "")).strip()
    log(f"{D_RANK_SOLVE_MARKER} {label} {_budget_rest.VOTE_FIXED_MARKER} {_budget_rest.VOTE_FIXED_ENV}={_now}"
        f"{' (aus --env-d, Vorrang)' if _given else ''} ({_budget_rest.VOTE_FIXED_RECORD} {vals}, {src}; "
        f"{_budget_rest.VOTE_FIXED_BASIS}): ein Rang mit GEMESSENEM Boden stimmt mit der Linie near-OOM + fixed statt "
        f"mit dem Boden ab (der Boden ist die Aktivierung, die Rate x Zeilen schon bepreist) und trimmt vor dem "
        f"Forward unter Linie + Rate x Zeilen")
    return _now

def p_mamba_slots(ns, model: str, p_bs: int) -> Tuple[int, str]:
    """Group P's device mamba SLOTS as the boot allocates them, and the source.

    H92c. The pool model charged ``ceil(--p-bs x 2 x 1.25)`` slots -- the
    DEMAND-DRIVEN branch of the runtime's sizer -- but argv_p never lets P reach
    that branch: it always states ``--max-mamba-cache-size`` (its own
    ``P_MAX_MAMBA_CACHE_SIZE`` = 24, or the operator's value in ``--extra-p``,
    which wins), and ``model_runner_kv_cache_mixin`` takes an explicit value as
    the pool size (``Mamba Cache is allocated. max_mamba_cache_size: N``; later
    passes only LOWER it: the world MIN-sync and ``--gdn-resident-state-slots``).
    fnFL2h91bb2 (P_BS 8, ``--max-mamba-cache-size 32``) was charged 20 slots and
    allocated 32 -- 411/150/112 MiB missing from the pool budget per stage;
    x178 (P_BS 4, the default 24) was charged 10 and allocated 24.

    Read off P's argv AS THIS LAUNCHER BUILDS IT, with the boot's own
    ``--extra-p`` (the last occurrence, argparse's rule), not restated; only a
    P argv without the flag falls back to the demand formula, which then IS the
    runtime's branch (an upper bound on ``_auto_mamba_demand_size``, #1286 F4).

    UNIFY (operator rule, 27B byte-identical): only a profile whose registry
    row sets ``p_mamba_slots_from_argv`` (nextflash) reads the argv; any other
    (qwen27b) keeps the demand formula as before H92c.
    """
    _row = pdflip_form.profile_row(getattr(ns, "profile", None))
    # The probe's budgets [1, 1, 1] are sentinels, not the launcher's card
    # calculation: W100 (``_refuse_if_extra_raises_budget``) must not judge an
    # arm's --rank-gpu-memory-mib against them -- dual19 D2 (dual1g, --extra-p
    # '--rank-gpu-memory-mib 8830,3400,2900 ...'): "Launcher 1 MiB -> extra
    # 8830 MiB". The probe reads --max-mamba-cache-size only; the budget flag
    # is dropped from ITS extra, and the real argv_p still meets W100.
    _probe_extra = _without_budget_flag(shlex.split(str(getattr(ns, "extra_p", "") or "")))
    flags = (argv_p("py", model, [1, 1, 1], 1, 1, RING_FORM_SENTINEL_STORE_CFG,
                    _probe_extra, p_bs=int(p_bs),
                    stage_ratio="", attn_stage_ratio="")
             if _row is not None and _row.p_mamba_slots_from_argv else [])
    value = None
    for i, tok in enumerate(flags):
        tok = str(tok)
        if tok == "--max-mamba-cache-size" and i + 1 < len(flags):
            value = str(flags[i + 1])
        elif tok.startswith("--max-mamba-cache-size="):
            value = tok[len("--max-mamba-cache-size="):]
    if value is not None:
        return int(value), "--max-mamba-cache-size %d on P's argv" % int(value)
    target = _max_running_requests(model, "P", int(p_bs))
    per_req = int(getattr(ns, "pp_cut_mamba_slots_per_running_request",
                          P_MAMBA_SLOTS_PER_RUNNING_REQUEST))
    slots = math.ceil(target * per_req * P_MAMBA_AUTO_SAFETY_MARGIN)
    return slots, (
        "ceil(--p-bs %d x %d slots/running-request x safety %.2f), an UPPER BOUND on "
        "_auto_mamba_demand_size incl. its hard floor (P's argv states no "
        "--max-mamba-cache-size)" % (target, per_req, P_MAMBA_AUTO_SAFETY_MARGIN)
    )
