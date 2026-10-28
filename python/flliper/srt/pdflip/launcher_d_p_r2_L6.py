"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Optional

from flliper.srt.pdflip import budget_rest as _budget_rest
from flliper.srt.pdflip.launcher_d_apply_r2_L5 import booked_rest_vote_line_deltas
from flliper.srt.pdflip.launcher_d_leaf import budget_rest_from_records
from flliper.srt.pdflip.launcher_d_p_r2_L1 import booked_rest_admit_deltas
from flliper.srt.pdflip.launcher_log_plan import Log
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst_boots
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused

#: The corridor law is 819-1229 MiB NVML-free per card under the awake
#: group's load.  MEASURED 2026-09-07 boot weg2onebackup2 with this constant
#: at 1024: the 5090's continuous minimum under group D was 620-684 MiB
#: (P log "CORRIDOR LAW BREACHED", 4 samples) -- the awake group lands
#: ~400 MiB below the budget line (CUDA context + BAR1 windows sit outside
#: the --rank-gpu-memory-mib fraction).  The measured overshoot is charged
#: here so the minimum lands mid-band; R5 grades the result.
#:
#: SPLIT 2026-09-09 (#1257c). The 404 stays -- it is a measurement of THIS
#: form (D's awake overshoot outside the --rank-gpu-memory-mib fraction) and
#: nothing about the user decision touches it. The 1024 does not: it is now
#: the DERIVED per-card floor from
#: ``managers.corridor_guard.corridor_floor_mib`` = measured transient peak of
#: the group awake on that card + that card's user reserve. Where nothing
#: measured the transient the derivation returns the same 1024, stamped
#: ``UNMEASURED-FALLBACK``, so an unmeasured rig at reserve 0 subtracts
#: exactly 1428 here as before -- byte-identical, deliberately.
D_AWAKE_OVERSHOOT_MIB = 404

def booked_rest_kwargs(cards: List[Card], profile: Optional[str] = None,
                       group: str = "D", log: Optional[Log] = None,
                       capped: bool = False) -> Dict[str, object]:
    """The ``booked_rest_mib`` / ``booked_rest_provenance`` kwargs of
    :func:`budgets_from_dc` for ``group`` -- ``{}`` (budgets byte-identical)
    when the switch is off or the profile carries no record for the group
    (then ``log`` says so once, by name: UNMEASURED, legacy terms stand).

    ``capped`` (PDFLIP-ALLOC-OVERHANG, P0 torch cache cap armed for the group,
    :func:`torch_cache_cap_armed`): the CAPPED rest
    ``<G>_AWAKE_REST_CAPPED_MIB`` (non-torch + allocation overhang + kept
    cache) where the profile carries it -- the general allocator cache is
    then torch's to release under the cap, not a post; without that record
    the uncapped rest stands (named)."""
    if not budget_rest_from_records(profile):
        return {}
    name = _budget_rest.record_name(group)
    if capped:
        cname = _budget_rest.capped_record_name(group)
        try:
            _pconst(cname, profile)
            name = cname
        except KeyError:
            if log is not None:
                log(f"{_budget_rest.OVERHANG_MARKER} group={group}: torch cache cap armed but the "
                    f"profile carries no {cname} -- the uncapped rest {name} stands (named); "
                    f"measure with python -m flliper.srt.pdflip.budget_rest --capped")
    try:
        vals = list(_pconst(name, profile))
    except KeyError:
        if log is not None:
            log(f"{_budget_rest.MARKER} {_budget_rest.SOURCE_UNMEASURED} group={group}: the "
                f"profile carries no {name} -- floor + user reserve + awake_overshoot "
                f"{D_AWAKE_OVERSHOOT_MIB} + overshoot record stand (named, not measured); "
                f"measure with python -m flliper.srt.pdflip.budget_rest")
        return {}
    try:
        res = _budget_rest.resolve(vals, _pconst_boots(name, profile), len(cards), group, name)
    except ValueError as exc:
        raise PdFlipLaunchRefused(f"{_budget_rest.MARKER}: {exc}") from None
    rests = list(res.values)
    prov = res.provenance
    deltas = booked_rest_vote_line_deltas(cards, rests, name, profile, group, log)
    vote_deltas = list(deltas)
    # PDFLIP-ADMIT-BOOK (desk/27b-admit-trim-1024): the admission peak priced beside the vote line -- per card the
    # larger of the two posts (they never coexist), only where the record's form runs
    deltas = booked_rest_admit_deltas(cards, rests, vote_deltas, profile, group, log, rest_name=name)
    if any(deltas):
        rests = [None if v is None else int(v) + int(d) for v, d in zip(rests, deltas)]
        if deltas == vote_deltas:
            prov = (f"{prov} + vote-line delta {','.join(str(d) for d in deltas)} "
                    f"{_budget_rest.VOTE_LINE_BASIS} ({_budget_rest.VOTE_LINE_MARKER})")
        else:
            prov = (f"{prov} + delta {','.join(str(d) for d in deltas)} = max(vote-line delta "
                    f"{','.join(str(d) for d in vote_deltas)} {_budget_rest.VOTE_LINE_BASIS} "
                    f"({_budget_rest.VOTE_LINE_MARKER}), admission need {_budget_rest.ADMIT_RECORD} "
                    f"{_budget_rest.ADMIT_BASIS} ({_budget_rest.ADMIT_MARKER}))")
    return {"booked_rest_mib": rests, "booked_rest_provenance": prov}
