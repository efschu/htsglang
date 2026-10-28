"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip.launcher_d_leaf import _csv
from flliper.srt.pdflip.launcher_d_leaf import _csv_ints
from flliper.srt.pdflip.launcher_d_mid import attn_anchor_stage
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PP_ATTN_STAGE_RATIO_SCORES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PP_STAGE_RATIO_SCORES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import derived_family_cost
from flliper.srt.pdflip.launcher_d_p_r2_L2 import stage_fit_family_cost
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst
from flliper.srt.pdflip.launcher_p_l3_leaf import floor_source_for_shipped_line
from flliper.srt.pdflip.launcher_p_l3_leaf import n_stage_incumbent
from flliper.srt.pdflip.launcher_refuse_proc_leaf import read_mean_prefill_prefix
from flliper.srt.pdflip.launcher_refuse_proc_leaf import shipped_line
from flliper.srt.pdflip.launcher_solve_main_r6_L0_1 import _solve_p_cut_step07_design_src
from flliper.srt.pdflip.launcher_solve_main_r6_L0_1 import _solve_p_cut_step08__power_now
from flliper.srt.pdflip.launcher_solve_main_r6_L0_1 import _solve_p_cut_step09__depth_profile
from flliper.srt.pdflip.launcher_solve_main_r6_L0_1 import _solve_p_cut_step19_incumbent_row
from flliper.srt.pdflip.launcher_solve_main_r6_L0_1 import _solve_p_cut_step20__cell_bytes_per_attn_lay

def _solve_p_cut_step22_incumbent(*, budgets_p, model, ms, n_layers, ns):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # FIX 2: the incumbent is the cost model's MEASUREMENT BASIS -- the cut
    # MEASURED_MS_PER_LAYER was taken under -- and that is what
    # P_PP_STAGE_RATIO_SCORES is defined to be. It was restated here as a bare
    # "32,18,14", a fourth copy of a vector whose whole point is one
    # definition; an operator pin still overrides it.
    #
    # #1362 [22-fix follow, DESK6 #1378]: THE SAME MODEL-IDENTITY GUARD AS
    # argv_p(), not a second unguarded reader of "the incumbent ratio".
    # Measured live: without this, a 24-layer model with its OWN complete
    # calibration record still crashed here with the bare
    # pp_cut.family_costs_from_measurement ValueError
    # (refuse_foreign_calibration's own message already quotes that exact
    # text as the failure #1362 exists to name) -- because this line ran
    # BEFORE argv_p ever got a chance to consult the calibration record.
    if ns.pp_stage_ratio:
        incumbent = _csv_ints(ns.pp_stage_ratio)
    elif len(budgets_p) != len(P_PP_STAGE_RATIO_SCORES):
        # HW-P1c 1003: the measurement basis is a THREE-stage cut; a P group of
        # another size starts the solver from a cut sized to its stage rates
        incumbent = n_stage_incumbent(len(budgets_p), n_layers, ms)
    else:
        incumbent = (
            host_ledger.resolve_calibrated_stage_layer_counts(
                model, ns.pp_stage_ratio,
                _pconst("CALIBRATION_LAYERS", getattr(ns, "profile", None)))
            or list(P_PP_STAGE_RATIO_SCORES)
        )
    design_src = _solve_p_cut_step07_design_src(ns=ns)
    census = read_mean_prefill_prefix(design_src) if design_src else None
    return census, design_src, incumbent

def _solve_p_cut_step24_anchor_stage(*, _pp_cut, budgets_p, cards, chunk_tokens, families, incumbent, log, ms, ns, text_cfg):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    if len(budgets_p) != len(P_PP_STAGE_RATIO_SCORES):
        # HW-P1c 1003: the family split is fitted on the REFERENCE basis and
        # mapped to the live stages by card class (see derived_family_cost)
        family_cost, family_prov = derived_family_cost(ns, cards, families, int(chunk_tokens))
    else:
        measured_attn = _pp_cut.attention_counts(families, incumbent)
        # HW-GENERIC 1002: the stage whose card is of the class the deep
        # attention anchor (ATTN_ANCHOR_MS, "a 3080 ... 0.4 s per chunk") was
        # measured on -- by calibration class, not "the first card that is not a
        # 5090". On the reference rig that is stage 1, as before.
        anchor_stage = attn_anchor_stage(cards, log)
        family_cost, family_prov = _pp_cut.family_costs_from_measurement(
            measured_ms_per_layer=ms,
            measured_counts=incumbent,
            measured_attn_counts=measured_attn,
            chunk_tokens=int(chunk_tokens),
            ref_prefix_tokens=float(ns.pp_cut_calibration_prefix_tokens),
            anchor_stage=anchor_stage,
            anchor_attn_ms_per_layer=float(ns.pp_cut_attn_anchor_ms),
            anchor_prefix_tokens=float(ns.pp_cut_attn_anchor_prefix_tokens),
        )
    _fitted_log, _power_now = _solve_p_cut_step08__power_now(cards=cards, log=log)
    if str(getattr(ns, "pp_cut_stage_fit", "") or "").strip():
        family_cost, family_prov, _fitted_log = stage_fit_family_cost(
            ns, cards, int(chunk_tokens), _power_now, log)
    _depth_profile, lanes, pair_ms, payload_bytes = _solve_p_cut_step09__depth_profile(_fitted_log=_fitted_log, cards=cards, chunk_tokens=chunk_tokens, log=log, ns=ns, text_cfg=text_cfg)
    return _depth_profile, _fitted_log, _power_now, family_cost, family_prov, lanes, pair_ms, payload_bytes

def _solve_p_cut_step28_incumbent_row(*, chosen, decision, log, ship_why):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    incumbent_row, makespan_row = _solve_p_cut_step19_incumbent_row(decision=decision)
    # #1286b: THE FLOOR APPLIES TO WHAT SHIPS, not only to what the objective
    # ranked. `incumbent` names a candidate and looks it up in the ranked field,
    # so it never passes through the set --pp-solve-pool-floor narrowed; without
    # this call an operator could set a floor, take that arm, and boot below it
    # with every line saying the floor was honoured. Same writer as the solver's
    # own refusals (CutDecision.refuse_shipped_below_floors -> refuse_below_floors).
    decision.refuse_shipped_below_floors(
        chosen, "the SHIPPED cut (%s)" % (ship_why.split(" (")[0],)
    )
    log(
        shipped_line(
            decision,
            chosen,
            ship_why,
            incumbent_row,
            makespan_row,
            incumbent_fallback="%s / %s"
            % (_csv(P_PP_STAGE_RATIO_SCORES), _csv(P_PP_ATTN_STAGE_RATIO_SCORES)),
            floor_source=floor_source_for_shipped_line(decision),
        )
    )

def _solve_p_cut_step29_per_rank_priced(*, _pp_cut, chosen, kv_mib, model_pool):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # #1286 -- THE JOIN LINE. `PP-CUT SHIPPED` prices the three objectives, but
    # nothing on it could be JOINED against what group P then sized: the two
    # halves are emitted by different processes, in different units, with no
    # shared key, so the +64.1 % of weg2sb5f sat in plain sight across two logs
    # for a whole boot. `cell_bytes` is that key -- it is the per-rank
    # `cell_size=` of P's own `KV pool sizing:` line, byte for byte -- and the
    # per-rank priced capacities line up against P's per-rank
    # `KV token sizing: rank r local capacity N tokens`.
    #
    # ONE GREP FINDS BOTH HALVES, and that is the whole point of the marker:
    #   grep -h 'PP-POOL-JOIN\|KV pool sizing\|KV token sizing' front.log P.log
    per_rank_priced = _pp_cut.stage_pp_capacities(
        chosen.layers, chosen.attn, model_pool
    ) if chosen.kind == "contiguous" else ()
    _cell_bytes_per_attn_layer = _solve_p_cut_step20__cell_bytes_per_attn_lay(_pp_cut=_pp_cut, kv_mib=kv_mib)
    return _cell_bytes_per_attn_layer, per_rank_priced
