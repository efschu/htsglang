"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

from flliper.srt.managers import corridor_guard
from flliper.srt.pdflip import budget_rest as _budget_rest
from flliper.srt.pdflip.launcher_d_leaf import effective_reserve_by_card
from flliper.srt.pdflip.launcher_d_p_r2_L6 import booked_rest_kwargs
from flliper.srt.pdflip.launcher_log_plan import Log
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst
from flliper.srt.pdflip.launcher_p_l3_leaf import parse_group_env
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_vision_spec_mid import torch_cache_cap_armed

D_ONLY_LABEL = "D(d-only, expectation)"

def d_record_torch_caps(ns, cards: List[Card], budgets_d: Sequence[int], log,
                        label: str) -> Optional[List[int]]:
    """PDFLIP-ALLOC-OVERHANG: the P0 cap vector from the RECORDS, for a D solve
    whose verdict books no foreign/non-torch term (the 27B: its verdict's
    ``verfuegbar`` is the NVML total, so ``verfuegbar - floor`` never binds --
    31788 / 19661 / 19661 MiB against a torch reach of ~29.4 / 18.6 / 18.4 GiB).

    Per card ``budget + rest - OTHER`` (:func:`budget_rest.torch_caps`), where
    ``rest`` is what this pass's budget booked (the capped rest while the cap
    is armed) and OTHER is ``D_TORCH_CAP_OTHER_MIB``. ``None`` (the caller
    keeps the verdict's cap) for the d-only pass (no P: the record is the flip
    form's), a profile without OTHER, or a card the rest record does not price."""
    if label == D_ONLY_LABEL:
        return None
    try:
        others = list(_pconst(_budget_rest.other_record_name("D"), ns.profile))
    except KeyError:
        return None
    kw = booked_rest_kwargs(cards, ns.profile, "D", capped=torch_cache_cap_armed(ns))
    rests = kw.get("booked_rest_mib")
    if rests is None or len(others) != len(cards):
        return None
    caps = _budget_rest.torch_caps([int(b) for b in budgets_d], list(rests),
                                   [None if o is None else int(o) for o in others])
    if caps is not None:
        log(f"{_budget_rest.OVERHANG_MARKER} {label} caps "
            + ", ".join(f"rang{i} {b} + {r} - {o} = {c}"
                        for i, (b, r, o, c) in enumerate(zip(budgets_d, rests, others, caps)))
            + f" MiB ({kw.get('booked_rest_provenance', '')})")
    return caps

def rank_reserve_kwargs(group: str, ns, cards: Sequence["Card"],
                        user_reserve_by_card: Optional[Dict[str, int]], log: Optional[Log] = None) -> Dict[str, object]:
    """The ``build_env`` kwargs that hand ``group``'s ranks the reserve the launcher books for it.

    #236 review MUSS-1: D's awake rest is a booked record on every card of the standard
    form (``budget_rest_from_records``), where ``budgets_from_dc`` charges NO user reserve;
    the ranks of D therefore get 0 there (named, with the stated value, in the rank's
    ``PDFLIP-USER-RESERVE-READ`` line). P is unbooked: the operator's value. The booked
    vector is the one ``budgets_from_dc`` is called with (``booked_rest_kwargs``), so the
    two read one record and one function (:func:`effective_reserve_by_card`).
    Logs the masking once per call when ``log`` is given, and names an operator
    ``--env-<g>`` value of the variable that stands instead (#236 review SOLL-3).
    """
    booked = None
    # --d-only: ``d_only_solve`` calls ``budgets_from_dc`` WITHOUT ``booked_rest_mib`` (the
    # D-solo pass keeps floor + reserve), so the launcher charges the reserve on every card
    # there and the ranks must carry it too -- one truth (#236 nachpruefung BUG 1).
    if group == "D" and not bool(getattr(ns, "d_only", False)):
        booked = booked_rest_kwargs(cards, ns.profile, "D", capped=torch_cache_cap_armed(ns)).get("booked_rest_mib")
    if log is not None:
        eff, masked = effective_reserve_by_card(cards, user_reserve_by_card, booked)
        if any(int(v) > 0 for v in (user_reserve_by_card or {}).values()):
            log(f"PDFLIP-USER-RESERVE-RANK-ENV group={group}: "
                + ", ".join(f"nvml{c.nvml_index} stated={int((user_reserve_by_card or {}).get(c.uuid, 0))} "
                            f"effective={eff[c.uuid]}"
                            + (" (masked: booked awake rest, the launcher charges no reserve there)"
                               if c.uuid in masked else "")
                            for c in cards))
        _op = parse_group_env(str(getattr(ns, "env_d" if group == "D" else "env_p", "") or ""))
        _named = [k for k in (corridor_guard.USER_RESERVE_ENV, corridor_guard.USER_RESERVE_MASKED_ENV) if k in _op]
        if _named:
            log(f"PDFLIP-USER-RESERVE-RANK-ENV group={group}: --env-{group.lower()} states {', '.join(_named)}; "
                "the operator value STANDS and the launcher's effective mapping is NOT published over it "
                "(rank floor may then differ from the launcher budget -- operator's choice)")
    return dict(user_reserve_by_card=user_reserve_by_card, reserve_cards=cards, reserve_booked_rest_mib=booked)
