"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Tuple
import os

from flliper.srt.pdflip import DEFAULT_P_BS
from flliper.srt.pdflip import card_identity
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import ring_table
from flliper.srt.pdflip.launcher_add_r5_L0 import DESIGN_PREFIX_FALLBACK_TOKENS
from flliper.srt.pdflip.launcher_d_apply_r2_L5 import p_mamba_slots
from flliper.srt.pdflip.launcher_d_leaf import _csv
from flliper.srt.pdflip.launcher_d_leaf import _csv_floats
from flliper.srt.pdflip.launcher_d_leaf import _csv_ints
from flliper.srt.pdflip.launcher_d_leaf import card_power_limits_w
from flliper.srt.pdflip.launcher_d_leaf import card_power_line
from flliper.srt.pdflip.launcher_d_leaf import count_marker
from flliper.srt.pdflip.launcher_d_mid import CONTEXT_LENGTH_TOKENS
from flliper.srt.pdflip.launcher_d_mid import ONCARD_MODE_DEFAULT
from flliper.srt.pdflip.launcher_d_mid import gate_w10
from flliper.srt.pdflip.launcher_d_p_build import p_rope_draft_delta_mib
from flliper.srt.pdflip.launcher_d_p_r2_L0 import EVIDENCE_DIR
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_ACTIVATION_RECORD
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PP_ATTN_STAGE_RATIO_SCORES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PP_STAGE_RATIO_SCORES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PREFILL_TRANSIENT_SUPPORT
from flliper.srt.pdflip.launcher_d_p_r2_L0 import p_activation_record
from flliper.srt.pdflip.launcher_d_p_r2_L0 import p_draft_kv_cell_bytes_by_stage
from flliper.srt.pdflip.launcher_d_p_r2_L0 import p_produces_draft_pages
from flliper.srt.pdflip.launcher_d_p_r2_L0 import p_stage_fixed_ctx_delta_record
from flliper.srt.pdflip.launcher_d_p_r2_L0 import p_stage_fixed_for_checkpoint
from flliper.srt.pdflip.launcher_d_p_r2_L0 import p_stage_layers
from flliper.srt.pdflip.launcher_d_p_r2_L1 import P_ACTIVATION_RECORD_CHUNK
from flliper.srt.pdflip.launcher_d_p_r2_L1 import calib_log_accept_of
from flliper.srt.pdflip.launcher_d_p_r2_L1 import gate_w11
from flliper.srt.pdflip.launcher_d_p_r2_L1 import p_transient_support
from flliper.srt.pdflip.launcher_d_p_r2_L2 import p_prefill_transient_vector_mib
from flliper.srt.pdflip.launcher_d_p_r2_L2 import w10_skip_no_producer_line
from flliper.srt.pdflip.launcher_p_choose import p_rope_context_delta_mib
from flliper.srt.pdflip.launcher_p_d_r2_L3 import P_PREFILL_TRANSIENT_ENV
from flliper.srt.pdflip.launcher_p_l3_leaf import forced_lane_borrow
from flliper.srt.pdflip.launcher_p_l3_leaf import incumbent_candidate
from flliper.srt.pdflip.launcher_p_l3_leaf import model_config_path
from flliper.srt.pdflip.launcher_p_l3_leaf import p_mamba_floor_text
from flliper.srt.pdflip.launcher_p_l3_leaf import p_pool_posts_as_booked
from flliper.srt.pdflip.launcher_p_l3_leaf import p_prefill_graph_bucket
from flliper.srt.pdflip.launcher_p_l3_leaf import pcie_lanes
from flliper.srt.pdflip.launcher_p_l3_leaf import per_pair_crossing_ms
from flliper.srt.pdflip.launcher_p_mid import newest_prefill_census_log
from flliper.srt.pdflip.launcher_p_mid import p_seats
from flliper.srt.pdflip.launcher_refuse_proc_leaf import WEIGHT_SOURCE_DEFAULT
from flliper.srt.pdflip.launcher_refuse_proc_leaf import pp_cut_chunk_mix
from flliper.srt.pdflip.launcher_refuse_proc_leaf import resolve_pool_floor
from flliper.srt.pdflip.launcher_types import PCutFacts
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_xchg_vision_leaf import xchg_bounce_arm_pins_host
from flliper.srt.planner import power_limit as _power
import json

#: WHICH OBJECTIVE THE SOLVER IS ASKED, per arm of ``--pp-solve-objective``.
#: ``incumbent`` is not a solver objective and must never be passed as one:
#: it names a CANDIDATE, not a ranking, and it is answered here by looking
#: that candidate up in the solver's own ranked field
#: (:func:`pick_shipped_cut`).  The ranking it is looked up in stays the
#: standing law's ``maxkv``, so the priced alternatives beside it are the
#: same rows every other arm prints.
P_SOLVER_OBJECTIVE_OF = {
    "maxkv": "maxkv",
    "makespan": "makespan",
    "incumbent": "maxkv",
}

def _solve_p_cut_step01__pp_cut(*, cards, chunk_tokens, log, model, ns, power_reading):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    from flliper.srt.planner import pp_cut as _pp_cut
    from flliper.srt.planner import pp_cut_launch as _cut

    cfg_path = model_config_path(model)
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    text_cfg = cfg.get("text_config") or cfg
    n_layers = int(text_cfg["num_hidden_layers"])
    kinds = text_cfg.get("layer_types") or []
    n_attn = sum(1 for k in kinds if str(k) == "full_attention")
    if n_attn <= 0:
        raise PdFlipLaunchRefused(
            f"W40 PdFlipPPCutRefused: {cfg_path} states no full_attention layer "
            f"in layer_types, so the attention axis cannot be solved and the "
            f"KV pool cannot be priced. Refusing rather than defaulting."
        )
    # fnFL2 H57: THE CUT THE MEASURED STAGE RATES RECOMMEND FOR THIS POWER
    # LIMIT, printed before anything below can refuse (x164 died at W40 inside
    # the solve) -- and ONLY printed: the shipped cut stays --pp-stage-ratio
    # (the arm's PP_RATIO), nothing here reads the line back.
    log(_power.rate_cut_line(
        power_reading,
        stage_cards=cards,
        model_name=os.path.basename(os.path.normpath(str(model))),
        is_full_attention=[str(k) == "full_attention" for k in kinds],
        pinned=_csv_ints(ns.pp_stage_ratio) if getattr(ns, "pp_stage_ratio", None) else None,
        chunk_tokens=int(chunk_tokens),
        fr_p=str(getattr(ns, "pp_cut_expert_device_fraction", "") or ""),
    ))
    return _cut, _pp_cut, cfg, kinds, n_attn, n_layers, text_cfg

def _solve_p_cut_step02_kv_mib(*, _pp_cut, cfg, log, model, n_layers, ns):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # The KV cell is CONSUMED from config, never fitted (#704 D1).
    kv_mib = _pp_cut.kv_mib_per_token_per_attn_layer_from_config(
        cfg, "fp8_e4m3", n_layers
    )
    # Weights from the safetensors HEADERS, not from a parameter formula.
    terms = _pp_cut.checkpoint_weight_terms(model)
    n_attn_ckpt = len(terms.attention_layer_indices)
    mean_layer_mib = (
        terms.attn_layer_weight_bytes * n_attn_ckpt
        + terms.linear_layer_weight_bytes * (terms.n_layers - n_attn_ckpt)
    ) / max(1, terms.n_layers) / _pp_cut.MIB
    # Task #47/#48 (Next Flash): a checkpoint with MoE experts prices its
    # layers PER STAGE -- dense mean + experts at the stage's resident
    # fraction + the stage's LRU rows -- and PLE (an mmap on disk) at zero.
    # No default: a checkpoint with experts and no fractions is refused by
    # name, because the 27B constants priced 8 layers on rank 0 and refused
    # every cut (W40, 19.09. dry run nfdry4).
    layer_mib_by_stage: Tuple[float, ...] = ()
    # H92c: P's mamba slots, read ONCE off P's argv (with --extra-p): the pool
    # model's mamba post and the P card's mamba term are the same number.
    _p_mamba_slots, _p_mamba_src = p_mamba_slots(
        ns, model, int(getattr(ns, "p_bs", DEFAULT_P_BS) or DEFAULT_P_BS))
    _skip_140 = pdflip_form.gate_skip_line(
        "#140 PP-CUT FRACTION-SOLVE", getattr(ns, "pdflip_boot_form", None))
    if _skip_140 and terms.expert_layer_weight_bytes <= 0.0:
        log(_skip_140)
    return _p_mamba_slots, _p_mamba_src, kv_mib, layer_mib_by_stage, mean_layer_mib, terms

def _solve_p_cut_step04_ms(*, budgets_p, log, model, ns, terms, text_cfg):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    ms = _csv_floats(ns.pp_cut_measured_ms_per_layer)
    # YaRN x2 27B (29.09.): the posts were read at 262144; a longer P context
    # grows the eager RoPE cache before the pool is sized (p_rope_context_delta_mib).
    _stage_fixed = tuple(_csv_floats(ns.pp_cut_stage_fixed_mib))
    _rope_delta, _rope_line = p_rope_context_delta_mib(ns, len(_stage_fixed))
    if _rope_line is not None:
        log(_rope_line)
        _ctx_delta, _ctx_line = p_stage_fixed_ctx_delta_record(ns, len(_stage_fixed))
        if _ctx_delta is not None:
            # measured beats the RoPE estimate (the posts carry more than the cos/sin cache)
            log(_ctx_line)
            _rope_delta = _ctx_delta
        _stage_fixed = tuple(float(v) + float(d) for v, d in zip(_stage_fixed, _rope_delta))
    # PPCUT-INT4AR 24.10.: the record's non-layer half is its reference
    # checkpoint's embedding/lm_head; move it onto THIS checkpoint's bytes.
    _stage_fixed, _sf_line = p_stage_fixed_for_checkpoint(
        ns, _stage_fixed, terms, text_cfg=text_cfg, model=model)
    log(_sf_line)
    _booked = p_pool_posts_as_booked(ns)
    # the draft cell rides the same per-model gate: a model whose row keeps the old posts keeps the old cell
    _p_draft_cell, _p_draft_cell_line = (p_draft_kv_cell_bytes_by_stage(ns, len(budgets_p), text_cfg)
                                         if _booked else ((), None))
    if _p_draft_cell_line is not None:
        log(_p_draft_cell_line)
    return _booked, _p_draft_cell, _stage_fixed, ms

def _solve_p_cut_step06_if__booked(*, _booked, _cut, _pp_cut, budgets_p, kinds, log, model_pool, ns):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    if _booked:
        log("PP-CUT POSTEN (p_pool_posts_as_booked, %s): corridor holdback %s, prefill activation %s, "
            "mamba slots from P's argv -- what group P books (KV budget holdback=0.0 on every P rank; the P budget "
            "line carries the corridor)" % (
                getattr(ns, "profile", None),
                "0" if ns.pp_cut_corridor_holdback_mib is None else "%s (pinned)" % ns.pp_cut_corridor_holdback_mib,
                "0" if ns.pp_cut_activation_reserve_mib is None else "%s (pinned)" % ns.pp_cut_activation_reserve_mib))
    # #1286: a LAUNCH may not price with an unfunded post. The desk paths keep
    # their upper bound; this seam publishes a number that is compared against
    # --max-kv-per-request and read as the boot's pool, so it refuses instead.
    _cut.refuse_unfunded_posts(model_pool)
    # #1286 F5: and a per-stage vector of the wrong LENGTH is a launch input
    # error, not an unpriceable layout. Checked here, before the candidate
    # loop, because the loop's own ValueError catch turns it into "not one cut
    # is priceable" -- true of every candidate, and about the wrong thing.
    _cut.refuse_pool_model_geometry(model_pool, len(budgets_p))
    families = tuple(
        _pp_cut.LAYER_FAMILY_ATTENTION
        if str(k) == "full_attention"
        else _pp_cut.LAYER_FAMILY_LINEAR
        for k in kinds
    )
    return families

def _solve_p_cut_step07_design_src(*, ns):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # THE GAPPED DEFAULT IS NOT TAKEN, AND THE HOOK IS NAMED RATHER THAN LEFT
    # OPEN (#753 / boot weg2gp1, 2026-09-08, /spinning/gpu-arb/weg2/
    # BOOT_weg2gp1_0908.md). The user's 0,8,8 layout was probed on metal for
    # exactly this decision and the verdict is STAYS, twice over:
    #   * CORRECTNESS. The gate's PREMISE is refuted -- the gapped forward is
    #     not the '\n\n' garbage its docstring describes, it answers
    #     byte-identically on the #753 probe verbatim -- but its VERDICT
    #     survives: 3 of 6 determined-answer probes DIVERGE from the contiguous
    #     control, in both graph modes, against an A/A floor of 0 divergences.
    #     A forward that silently changes half the determined answers is the
    #     'confidently wrong' class the gate names.
    #   * MOTIVE. The gapped map loses on both axes it was chosen for: world
    #     pool 391,904 vs the contiguous control's 714,788 (-45.2 %) and
    #     1097.0 vs 577.9 ms per full chunk on the binding stage (+89.8 %).
    #     The 1.0M-token aim is CLOSER on the contiguous axis, not further.
    #   * 0,8,8 itself never reached the forward: stage 0 owns zero attention
    #     layers, so its KV cell is 0 and the HiCache host-pool constructor
    #     divides by it -- a third wall, new, and upstream of the gate.
    # THE HOOK, recorded and deliberately NOT applied (the record spells out
    # that it is unsupported by this boot): when a forward passes the six-probe
    # comparison, the predicate to narrow is not `is_gapped` but the composite
    # `is_gapped AND (flip vector OR speculative decoding)`, warning otherwise.
    # Until then the default is the CONTIGUOUS kv-floor cut, gapped maps stay
    # reachable only through --pp-layer-set, and that path still meets the
    # gate.

    # -- THE DEPTH AXIS (#1240) ------------------------------------------
    # The design prefix is a MEASUREMENT of this rig's own traffic when one
    # exists, and a printed fallback when it does not. It is not cosmetic: the
    # attention/linear cost ratio moves by more than an order of magnitude
    # between 4,096 and 262,144 tokens of prefix, so "the optimal cut" is a
    # function of it and a table without it stated is a table about nothing.
    design_src = ns.pp_cut_design_prefix_from or newest_prefill_census_log(
        EVIDENCE_DIR, accept=calib_log_accept_of(ns))
    return design_src

def _solve_p_cut_step08__power_now(*, cards, log):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # 27B line (user 24.09.): every boot prints its P cards' power limits --
    # the record a later --pp-cut-stage-fit is checked against.
    _power_now = card_power_limits_w(cards)
    log(card_power_line(cards, _power_now))
    # --pp-cut-stage-fit (default off): the stage cost FITTED from a boot's own
    # #PGAP lines replaces the bsscale family split above.
    _fitted_log = None
    return _fitted_log, _power_now

def _solve_p_cut_step09__depth_profile(*, _fitted_log, cards, chunk_tokens, log, ns, text_cfg):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # --pp-cut-depth-profile (default off): price each candidate over a depth
    # profile instead of the single design prefix.
    _depth_profile = None
    _prof_arg = str(getattr(ns, "pp_cut_depth_profile", "") or "").strip()
    if _prof_arg:
        from flliper.srt.planner import pgap_stage_fit as _fit

        try:
            _depth_profile, _prof_prov = _fit.depth_profile(
                _prof_arg, int(chunk_tokens), _fitted_log)
        except ValueError as exc:
            raise SystemExit(str(exc))
        log("PP-CUT DEPTH PROFILE (--pp-cut-depth-profile): %s -- every "
            "candidate's makespan below is the weighted mean over it, and its "
            "depth_tokens the profile's mean" % _prof_prov)
    # THE CROSSING FRAME is the same object the depth price already charges:
    # one PPProxyTensors hidden-states frame, [chunk, hidden] in the model
    # dtype. Derived here from the same three numbers rather than restated.
    payload_bytes = int(chunk_tokens) * int(text_cfg["hidden_size"]) * 2
    lanes = forced_lane_borrow(cards, pcie_lanes(cards), log)
    pair_ms = per_pair_crossing_ms(lanes, payload_bytes)
    return _depth_profile, lanes, pair_ms, payload_bytes

def _solve_p_cut_step10_log_pp_cut(*, _pp_cut, chunk_tokens, lanes, log, ns, pair_ms, payload_bytes, text_cfg):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    log(
        "PP-CUT crossing prices: PCIe lanes per ordinal %s, frame %.1f MiB "
        "(chunk %d x hidden %d x 2 B), measured pair ms %s (pairs absent from "
        "the measured lane table are OMITTED, and every candidate using one is "
        "reported UNPRICED rather than ranked at a guess)"
        % (
            lanes,
            payload_bytes / _pp_cut.MIB,
            int(chunk_tokens),
            int(text_cfg["hidden_size"]),
            ", ".join(
                "%d->%d %.2f" % (a, b, v) for (a, b), v in sorted(pair_ms.items())
            )
            or "NONE",
        )
    )
    # THE FLOOR AND ITS SOURCE ARE RESOLVED BEFORE THE SOLVE, and the line is
    # logged whether the solve then succeeds or REFUSES -- a W40 that does not
    # say which floor it was measured against, and whether that floor was the
    # shipped default or something the operator typed, sends the reader to the
    # wrong flag.
    _floor_flag = ns.pp_solve_pool_floor
    return _floor_flag

def _solve_p_cut_step11__cap_floor_default(*, _floor_flag, budgets_p, chunk_tokens, log, ns):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # #1441b (xsn204): the cap-floor rule shipped 40,12,12 and the launch was
    # refused at the ledger -- the exchange bounce term for that form is 27.75
    # GiB against 15.75 for 39,13,12 (+12 GiB of pinned host lanes: the 8-layer
    # tag windows fall differently across the stage boundaries), run peak
    # 105 GiB over the 95.9 GiB watermark. The rule stays, opt-in
    # (FLLIPER_PDFLIP_PCUT_CAP_FLOOR=1), until the solver prices the exchange
    # bounce of a candidate cut beside its makespan.
    # #1447: the cap-floor rule is the DEFAULT again on the exchange arm, now
    # that the solver prices each candidate's host bounce beside its makespan
    # (host_priced_pick); FLLIPER_PDFLIP_PCUT_CAP_FLOOR=0 turns it off.
    _cap_floor_default = "1" if xchg_bounce_arm_pins_host(
        str(getattr(ns, "pdflip_weight_source", WEIGHT_SOURCE_DEFAULT)),
        str(getattr(ns, "pdflip_xchg_oncard", ONCARD_MODE_DEFAULT))) else "0"
    if (_floor_flag is None and int(getattr(ns, "p_bs", DEFAULT_P_BS)) == 1
            and os.environ.get("FLLIPER_PDFLIP_PCUT_CAP_FLOOR", _cap_floor_default) == "1"):
        # #1441 (user 16.09.): with ONE chunked prefill at a time on P (flliper's
        # single chunked_req; --p-bs 1) the pool only has to hold the largest
        # request plus one chunk, so the floor is the cap, not the ordered
        # cut's pool -- 'immer das schnellste Layout, in das der Prefill
        # reinpasst'. The ordered cut of 2026-09-09 stays the rule for p_bs > 1.
        _cap = int(ns.max_kv_per_request or CONTEXT_LENGTH_TOKENS)
        _floor_flag = _cap + int(chunk_tokens)
        log("PP-CUT POOL FLOOR RULE: source=cap+chunk (p_bs=1, user order 2026-09-16) floor=%d = "
            "max_kv_per_request %d + chunk %d -- the solver ships the fastest cut at or above it"
            % (_floor_flag, _cap, int(chunk_tokens)))
    pool_floor, pool_floor_from_cut, pool_floor_rule = resolve_pool_floor(
        _floor_flag, n_stages=len(budgets_p))
    log("PP-CUT POOL FLOOR RULE: " + pool_floor_rule)
    return pool_floor, pool_floor_from_cut

def _solve_p_cut_step14_log_pp_cut(*, chunk_tokens, log, model_pool, ns):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # #114: the post's OWN provenance, next to the posts line it feeds.
    log(
        "PP-CUT activation post (#114): chunk %d -> measured prefill transient "
        "per stage %s MiB (support points %s, linear between, H41 -- see the "
        "PP-CUT ACTIVATION lines); post = %.1f MiB/rank (%s)"
        % (
            int(chunk_tokens),
            list(p_prefill_transient_vector_mib(int(chunk_tokens))),
            list(P_PREFILL_TRANSIENT_SUPPORT.chunks),
            float(model_pool.activation_reserve_mib),
            "pinned by --pp-cut-activation-reserve-mib"
            if ns.pp_cut_activation_reserve_mib is not None
            else "max(reference 1024, binding stage)",
        )
    )

def _solve_p_cut_step15__rec(*, log, model_pool, ns):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    if model_pool.activation_reserve_mib_by_stage:
        _rec, _rec_boots = p_activation_record(getattr(ns, "profile", None) or None)
        log(
            "PP-CUT activation post (#242): PER STAGE %s MiB (record %s %s at chunk "
            "%d, %s) -- charged in place of the one scalar %.1f above, the same "
            "vector group P's ranks book as %s; the P-KARTE takes max(builtin, "
            "record) %s"
            % (
                ["%.1f" % v for v in model_pool.activation_reserve_mib_by_stage],
                P_ACTIVATION_RECORD,
                list(_rec or ()),
                P_ACTIVATION_RECORD_CHUNK,
                _rec_boots or "no boots named",
                float(model_pool.activation_reserve_mib),
                P_PREFILL_TRANSIENT_ENV,
                [list(pt.mib) for pt in p_transient_support(
                    getattr(ns, "profile", None) or None, card=True).points
                 if pt.chunk == P_ACTIVATION_RECORD_CHUNK][0],
            )
        )

def _solve_p_cut_step16_log_pp_cut(*, _p_mamba_src, log, mean_layer_mib, model_pool, ns):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    log(
        "PP-CUT budget posts (the boot's own list, MiB/rank): "
        "weights+runtime = %.1f/layer x n + stage_fixed %s; corridor holdback "
        "%.1f; mamba %.4f/linear-layer/slot x %d slots (%s; %s); speculative "
        "intermediate %.1f; prefill activation reserve %.1f; mamba pre-capture "
        "reserve %.1f; page_size %d. Zero posts ACKNOWLEDGED (absent from both "
        "reference boots' emitted lists, not merely unmeasured): %s. Every one "
        "of these was UNFUNDED or wrong before #1286 and each omission "
        "over-prices; measured on weg2sb5f, the same cut published 499967 "
        "tokens and group P sized 304655 (+64.1%%)."
        % (
            mean_layer_mib,
            ",".join("%.1f" % v for v in model_pool.stage_fixed_mib),
            float(model_pool.corridor_holdback_mib or 0.0),
            float(model_pool.mamba_mib_per_linear_layer_per_slot),
            int(model_pool.mamba_slots),
            _p_mamba_src,
            p_mamba_floor_text(int(model_pool.mamba_slots), p_seats(ns)),
            float(model_pool.speculative_intermediate_mib),
            float(model_pool.activation_reserve_mib),
            float(model_pool.mamba_precapture_reserve_mib),
            int(model_pool.page_size),
            ", ".join(model_pool.zero_posts_acknowledged) or "none",
        )
    )

def _solve_p_cut_step17_log_pp_cut(*, _act_heuristic_mib, _act_provenance, decision, kv_mib, log, model_pool):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # #1286 F2: the ONE post in that list whose value is not a runtime
    # constant. Named with its exposure and its detection line, because an
    # unpriced risk reads as no risk -- the same lesson one level up.
    log(
        "PP-CUT activation reserve RISK: charging %.0f MiB/rank, %s. The gap "
        "is %+.0f MiB/rank, i.e. %.0f tokens per ATTENTION LAYER on whichever "
        "stage binds -- OVER-pricing, silently, in the direction that clears a "
        "pool floor the boot cannot hold. NOT PRICED IN, and NOT auto-resolved "
        "here on purpose: building the boot's ActivationProfile in this "
        "process would restate profile_from_server_args, and a restatement "
        "wrong by one field returns None = 'uncalibrated' = a false refusal of "
        "a healthy launch. DETECTION, one grep of group P's log: "
        "`grep -c 'Using the INHERITED activation heuristic' P.log` -- 0 means "
        "the calibrated branch was taken and the prices above stand; >=1 means "
        "every price above is stale by the gap and only the boot's own sizing "
        "line counts."
        % (
            float(model_pool.activation_reserve_mib),
            _act_provenance,
            _act_heuristic_mib - float(model_pool.activation_reserve_mib),
            (_act_heuristic_mib - float(model_pool.activation_reserve_mib))
            / max(1e-9, kv_mib),
        )
    )
    log(decision.provenance_line())
    log(decision.trade_line())

def _solve_p_cut_step18_log_decision_frontie(*, decision, log):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # #1286b -- THE CURVE. provenance_line prices TWO points of it (the two
    # objectives' cuts); this line prices every point that is not beaten on
    # both axes at once, so "what would a floor of N cost me in ms" is read off
    # one boot log instead of re-solved. It is also how a value for
    # --pp-solve-pool-floor is chosen.
    log(decision.frontier_line())
    for row in decision.table_lines():
        log(row)

def _solve_p_cut_step19_incumbent_row(*, decision):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # ALL THREE ARMS PRICED ON EVERY BOOT, the shipped one named.  The two the
    # boot did NOT take are the whole reason this line exists: the makespan
    # default was paid for weeks because its alternative was never printed
    # beside it, and a trade nobody can see is a trade nobody selected.
    incumbent_row = incumbent_candidate(
        decision, P_PP_STAGE_RATIO_SCORES, P_PP_ATTN_STAGE_RATIO_SCORES
    )
    makespan_row = decision.makespan or decision.chosen
    return incumbent_row, makespan_row

def _solve_p_cut_step20__cell_bytes_per_attn_lay(*, _pp_cut, kv_mib):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # #1286 F1: the join KEY comes from the pool model's own cell, never from a
    # typed 2048. The price above already uses
    # `kv_mib_per_token_per_attn_layer` (consumed from config by
    # kv_mib_per_token_per_attn_layer_from_config), and the inputs line one
    # screen up already prints it correctly -- so a hardcoded 2048 here would
    # publish a cell_bytes that does not equal P's own `cell_size=` on ANY
    # config with a different KV dtype or head geometry, while the price
    # itself stayed right. That is the one failure this line exists to
    # prevent: a join key that silently stops joining.
    _cell_bytes_per_attn_layer = int(round(kv_mib * _pp_cut.MIB))
    return _cell_bytes_per_attn_layer

def _solve_p_cut_step23_design_prefix(*, census, design_src, model, ns):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    if ns.pp_cut_design_prefix_tokens is not None:
        design_prefix = int(ns.pp_cut_design_prefix_tokens)
        design_prov = "PINNED by --pp-cut-design-prefix-tokens"
    elif census is not None:
        design_prefix = int(round(census[0]))
        design_prov = (
            "MEASURED mean prefix over %d prefill-batch chunks of %s (rank %s, "
            "one rank only: every stage prints the same batch)"
            % (census[1], design_src, census[2])
        )
    else:
        design_prefix = int(DESIGN_PREFIX_FALLBACK_TOKENS)
        design_prov = (
            "FALLBACK: no P log %sin %s carries a prefill census, so the design "
            "prefix is one chunk -- the shallowest depth this boot can run. "
            "This is an absence of the instrument, not a measured shallow rig."
            % (
                ("of " + pdflip_form.model_key(model) + " (PDFLIP-FORM calibration identity) ")
                if calib_log_accept_of(ns) is not None else "",
                EVIDENCE_DIR,
            )
        )
    return design_prefix, design_prov

def _solve_p_cut_step26_decision(*, _chunk_mix, _cut, _depth_profile, _fitted_log, _stage_model, _stage_prov, cards, design_prefix, families, family_cost, incumbent, model_pool, ms, ns, pair_ms, pool_floor, pool_floor_from_cut):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    decision = _cut.solve_launch_cut(
        stage_model=_stage_model,
        chunk_mix=_chunk_mix,
        stage_model_provenance=_stage_prov,
        layer_families=families,
        incumbent_layers=incumbent,
        measured_ms_per_layer=ms,
        family_cost=family_cost,
        design_prefix_tokens=design_prefix,
        per_pair_crossing_ms=pair_ms,
        pinned_layer_set=ns.pp_layer_set or None,
        measured_provenance=(
            ("makespan priced by the STAGE FIT (--pp-cut-stage-fit, line 'PP-CUT "
             "depth axis'); the terms that follow only order the enumeration: "
             if _fitted_log is not None else "")
            + "MEASURED per-layer ms %s (boot bsscale, BSSCALE_0907.md tip "
            "37c884b0b0: PP0 259.1 ms/32 layers, PP1 632.9/18, PP2 470.2/14, "
            "per full 4096-token chunk at bs6)" % ns.pp_cut_measured_ms_per_layer
        ),
        card_names=[c.name for c in cards],
        pool_model=model_pool,
        cap_tokens=int(ns.max_kv_per_request or CONTEXT_LENGTH_TOKENS),
        pinned_layers=_csv_ints(ns.pp_stage_ratio) if ns.pp_stage_ratio else None,
        pinned_attn=_csv_ints(ns.pp_attn_stage_ratio) if ns.pp_attn_stage_ratio else None,
        # #1254: the DEFAULT is the kv-floor row. Under the previous makespan
        # default the solver's own dry run picked 44,10,10 attn 11,2,3 at a
        # 499,967-token pool -- +33.5 % prefill for -47.7 % pool on metal --
        # which is a trade nobody selected, taken by a flag nobody passed.
        objective=P_SOLVER_OBJECTIVE_OF[str(ns.pp_solve_objective)],
        # #1286b: the floor under the SHIPPED pool. It constrains the set the
        # objective ranks over; it does not rank, and it never degrades to the
        # fastest cut below itself. #1305: under the SHIPPED DEFAULT (user
        # order 2026-09-09, DEFAULT_PP_ORDERED_CUT) pool_floor is None here
        # and pool_floor_from_cut names the ordered cut, and the solver reads
        # the floor off its own frontier; both None = the operator passed 0.
        pool_floor=pool_floor,
        pool_floor_from_cut=pool_floor_from_cut,
        depth_profile=_depth_profile,
    )
    return decision

def _solve_p_cut_step30_log_pp_poo(*, _cell_bytes_per_attn_layer, chosen, log, model_pool, ns, per_rank_priced):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    log(
        "PP-POOL-JOIN: objective=%s layers=%s attn=%s PRICED pool_tokens=%d "
        "priced_per_rank=%s cell_bytes=%s page_size=%d mamba_slots=%d -- "
        "REALISED is group P's own sizing: "
        "join on cell_bytes against 'KV pool sizing: available_bytes=... "
        "cell_size=...' per rank, and compare the world against 'KV token "
        "sizing: ... min-reduced across ranks to N'. PRICED minus REALISED is "
        "this pool model's error FOR THIS CUT and belongs in the boot record; "
        "it was +64.1%% before #1286 and the residual the posts carry now is "
        "the two-boot spread of --pp-cut-stage-fixed-mib (<=12.7 MiB/rank). "
        "SECOND JOIN, same discipline: mamba_slots is P's --max-mamba-cache-size "
        "(H92c; the demand bound only for an argv without it) against the boot's "
        "'Mamba Cache is allocated. max_mamba_cache_size: N', so N <= %d must "
        "hold; N above it means the mamba post is under-charged and the pool "
        "over-priced (#1286 F4)."
        % (
            str(ns.pp_solve_objective),
            ",".join(str(n) for n in chosen.layers),
            ",".join(str(a) for a in chosen.attn),
            int(chosen.pool_tokens),
            ",".join("%d" % int(c) for c in per_rank_priced) or "n/a (gapped)",
            ",".join(
                str(int(a) * _cell_bytes_per_attn_layer) for a in chosen.attn
            ),
            int(model_pool.page_size),
            int(model_pool.mamba_slots),
            int(model_pool.mamba_slots),
        )
    )

def _solve_p_cut_step31_log_pp_cut(*, chosen, decision, log):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # #1305 item 2 (boot weg2sn5pre): THE CAP AND THE FLOOR, SIDE BY SIDE.
    # Group P's `--max-total-tokens` was a constant 428,000 from boot weg2zr2
    # and cut the realised pool 7.7 % below the priced one -- and below the
    # floor the boot claimed to hold. It is now the shipped cut's priced pool
    # (argv_p, p_max_total_tokens), so realised = min(P's own sizing, priced)
    # and the floor binds the REALISED pool: a cap below the floor is
    # impossible by construction (the shipped pool clears the floor or the
    # solve refused above), which is why this line prints both numbers rather
    # than carrying a second refusal.
    log(
        "PP-CUT P-CAP: --max-total-tokens=%d (= the SHIPPED cut's priced pool; "
        "was the boot-weg2zr2 constant 428000 until 2026-09-09, which undercut "
        "the priced pool by 7.7 %% on weg2sn5pre) pool_floor=%s -- the floor "
        "binds the REALISED pool by construction: realised = min(group P's own "
        "'KV token sizing' min-reduce, this cap), and PRICED minus REALISED "
        "reads off PP-POOL-JOIN"
        % (
            int(chosen.pool_tokens),
            "none" if decision.pool_floor is None else str(int(decision.pool_floor)),
        )
    )

def _solve_p_cut_step32_stage_ratio(*, chosen, kinds):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    # ROUND TRIP AGAINST THE RUNTIME AUTHORITY, not against our own model.
    # --pp-stage-ratio entries are SCORES: server_args hands them to
    # derive_pp_layer_split, which SNAPS the boundary into the window that
    # realizes the attention target -- and snaps SILENTLY when an attention
    # vector is given (the #505(a) warning in _handle_pp_stage_ratio only
    # fires when attn_scores is None). MEASURED: 43,10,11 with attn 5,6,5
    # comes back as 23,24,17. A solved cut that does not survive this call is
    # a cut the boot would not run, so it is refused here rather than logged
    # and departed from.
    # THE SHIPPED candidate, not the ranking's winner: train 2 reconciled the
    # objective knob into ONE flag, so ``chosen`` and ``decision.chosen`` are
    # the same row only when the arm is the solver's own objective. Everything
    # below -- the round trip, the argv strings, PCutFacts -- must describe the
    # cut the boot RUNS.
    stage_ratio = _csv(chosen.layers)
    attn_ratio = _csv(chosen.attn)
    # FIX 2: through ``p_stage_layers``, the launcher's ONE score-pair ->
    # split seam, rather than a second direct call to the same upstream
    # function. Same authority, one caller of it.
    realized = p_stage_layers(
        [str(k) == "full_attention" for k in kinds],
        scores=list(chosen.layers),
        attn_scores=list(chosen.attn),
    )
    return attn_ratio, realized, stage_ratio

def _solve_p_cut_step33_if_list_realized(*, attn_ratio, chosen, log, realized, stage_ratio):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.)."""
    if list(realized) != list(chosen.layers):
        raise PdFlipLaunchRefused(
            f"W40 PdFlipPPCutRefused: the cut this launcher solved "
            f"({stage_ratio} / attn {attn_ratio}) does not survive "
            f"derive_pp_layer_split -- it would run as "
            f"{','.join(str(c) for c in realized)}. Refusing rather than "
            f"booting a layout the provenance line misdescribes."
        )
    log(
        f"PP-CUT round trip: derive_pp_layer_split({stage_ratio}, attn "
        f"{attn_ratio}) = {','.join(str(c) for c in realized)} -- the argv "
        f"states what the boot will run."
    )

def _solve_p_cut_step34_stage_model_mix(*, _chunk_mix, _power_now, _stage_model, _stage_prov, cards, chunk_tokens, design_prefix, design_src, log):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.): check the stage model against the live power limits, build the chunk mix."""
    if _stage_model is not None:
        from flliper.srt.pdflip import p_stage_model as _psm
        from flliper.srt.pdflip import power_limit as _plim

        _current = {}
        for _c, _w in zip(cards, _power_now):
            if _w is not None:
                _current.setdefault(card_identity.class_label(_c), float(_w))
        _stage_model, _plines = _psm.check_power(_stage_model, _current)
        for _ln in _plines:
            log("PP-CUT STAGE MODEL " + _ln)
        _chunk_mix, _mix_prov = pp_cut_chunk_mix(
            design_src, int(chunk_tokens), int(design_prefix), int(p_prefill_graph_bucket()))
        log("PP-CUT STAGE MODEL (--pp-cut-stage-model): %s; %s -- every candidate's makespan below is the "
            "mean ms per chunk over that mix (per chunk the bottleneck stage)" % (_stage_prov, _mix_prov))
    else:
        log("PP-CUT STAGE MODEL off: %s" % _stage_prov)
    return _chunk_mix, _stage_model

def _solve_p_cut_step36_gapped_facts(*, budgets_p, chosen, kv_mib, log, n_layers, ns, text_cfg):
    """Step of :func:`solve_p_cut` (split behaviour-neutrally, refactor 28.10.): the facts of a GAPPED map (layer set, round trip through the runtime's parser)."""
    # A GAPPED map is not expressible as --pp-stage-ratio and does not go
    # through derive_pp_layer_split at all: it is published as the layer
    # SET and executed by the #753 mid-loop crossing wire. The round trip
    # below is therefore skipped by KIND, not by accident -- and the map is
    # instead round-tripped through the runtime's OWN parser, which is the
    # matching authority for this form.
    from flliper.srt.distributed.utils import parse_pp_layer_sets

    back = parse_pp_layer_sets(
        chosen.layer_set, n_layers, len(budgets_p), allow_gapped=True
    )
    realized_counts = tuple(len(x) for x in back)
    if realized_counts != chosen.layers:
        raise PdFlipLaunchRefused(
            "W40 PdFlipPPCutRefused: the gapped map this launcher solved "
            "(%s) does not survive parse_pp_layer_sets -- it comes back as "
            "%s layers per stage. Refusing rather than booting a layout the "
            "provenance line misdescribes."
            % (chosen.layer_set, realized_counts)
        )
    log(
        "PP-CUT round trip: parse_pp_layer_sets(%s) = %s layers per stage, "
        "attn %s, %d crossings per chunk -- the --pp-layer-set argv states "
        "what the boot will run. --pp-stage-ratio/--pp-attn-stage-ratio are "
        "OMITTED from group P's argv for this kind: the count form cannot "
        "state a gapped map (derive_pp_layer_split refuses an attention "
        "count of 0 and silently re-derives a contiguous split from any "
        "other), so the map travels once, on the wire."
        % (
            chosen.layer_set,
            realized_counts,
            ",".join(str(a) for a in chosen.attn),
            chosen.crossings,
        )
    )
    return PCutFacts(
        # EMPTY, DELIBERATELY (FOLLOW FIX 1). A gapped map's layout is the
        # SET, and the count form cannot state it: `48,8,8 / 0,8,8` is
        # refused by derive_pp_layer_split outright, and a positive-count
        # gapped map is accepted and derives a DIFFERENT contiguous split.
        # argv_p omits both flags when these are empty, so the argv states
        # exactly one layout -- the one on the wire.
        stage_ratio="",
        attn_stage_ratio="",
        pool_tokens=float(chosen.pool_tokens),
        attn_counts=tuple(int(a) for a in chosen.attn),
        kv_mib_per_token_per_attn_layer=float(kv_mib),
        hidden_size=int(text_cfg["hidden_size"]),
        cap_tokens=int(ns.max_kv_per_request or CONTEXT_LENGTH_TOKENS),
        layer_set=chosen.layer_set,
        gapped=True,
        layer_counts=tuple(int(c) for c in realized_counts),
    )

def _main_step42_if_hicache_disabled(*, _draft_kv_on_p_requested, cards, chunk_count, chunk_layers, chunk_tag_cards, hicache_disabled, log, ns, p_split, shipped_argv_p, spec_d, spec_p):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # #1233 draft KV across the flip: W10 DRAFTER-IDENTITY GATE. Both groups
    # register a drafter (P: the producer on its last stage, D: the NEXTN
    # worker on every rank); the identity in their `HiCache draft KV
    # registered` lines and the canonical draft layout in their `#706
    # canonical DRAFT page active` lines must agree, or D fetches
    # `{hash}.draft-<Pid>` pages P never wrote (the weg2zr2 shape, 194,088
    # failed draft fetches). Refused before the front opens.
    # #1264: BOTH GATES GRADE A PRODUCER, so both are asked of THE ONE
    # PREDICATE first -- read off the argv this boot actually shipped, not off
    # ns.draft_kv_on_p, so a future change to argv_p carries the gates with it.
    # Under `off` group P registers no drafter and emits no DRAFT-KV-PRODUCER
    # line at all, which is the pre-#1233 shape these two gates were written to
    # refuse; running them there would refuse the rg6 baseline for being the
    # rg6 baseline. They are SKIPPED with a named line, never silently.
    # THE ROUTE CONSEQUENCE IS STATED IN THE SAME BREATH: D keeps its own NEXTN
    # head under `off`, so it still ASKS the carrier for `{hash}.draft-<id>`
    # pages -- and P writes none, so every one of those reads misses and D's
    # draft state is cold after each flip. That is the cost of this arm; it is
    # not a defect of the store and must not be triaged as one.
    # fnFL2x48b: under --pdflip-disable-hicache no carrier exists, so NEITHER
    # group prints a `HiCache draft KV registered` or `#706 canonical DRAFT
    # page` line -- the gate read P=None D=None and refused a boot whose D was
    # already READY, although the route it guards (draft pages through the
    # carrier) is absent by construction. Same fact as the W7/W9/W45 skips.
    if hicache_disabled:
        log("W10/W11 SKIPPED (--pdflip-disable-hicache): no HiCache carrier exists, so neither group registers "
            "a drafter identity or a canonical draft layout to match, and no draft pages cross a flip -- the "
            "#1233 route these gates grade is absent by construction (same fact as the W7/W9/W45 skips).")
    elif not ring_table.p_carries_drafter(shipped_argv_p):
        log(w10_skip_no_producer_line())
    else:
        # UNIFY S6 (27B c60a5d387f): under p_draft=cold P builds the producer
        # but writes NO draft pages -- W10 is skipped by name (identities still
        # logged); W11 still grades the cold-resident draft bytes.
        # IPC-LOG-EXCEPTION: W10-drafter / W11 still read drafter identity and draft-build
        # lines (Phase 2 rest: RankState schema 2 `drafter`, `draft_build`).
        gate_w10(spec_p.log, spec_d.log, log,
                 p_produces_draft_pages=p_produces_draft_pages())
        # #1233 fix 2: W11 DRAFT-RESIDENT gate. The last stage's pool is priced
        # against a budgeted draft residue (P_DRAFT_RESIDENT_BUDGET_MIB); the L2
        # line carries the MEASURED one. Over budget = the corridor derivation is refuted by this
        # boot -> refuse before the front opens (boot weg2dk2's 3994 MiB build).
        # IPC-LOG-EXCEPTION: W11 (see W10 above).
        gate_w11(spec_p.log, log, rope_ctx_mib=p_rope_draft_delta_mib(ns))
    # IPC-LOG-EXCEPTION: display only (class A), decides nothing; metric bar1_window_clip_total later.
    clips = count_marker(spec_d.log, "window clip") + count_marker(spec_d.log, "Bar1WindowRefused")
    log(f"BAR1 fit (deviation: transports open): D log 'window clip'/'Bar1WindowRefused' lines = {clips} (0 = both groups fit the aperture)")

    # 6. front
    # #1233 boot weg2dk4 -- WHICH CARD DOES A CHUNK TAG LIVE ON.  Group P is
    # PP: a weights_<k> tag is a layer band and sits on ONE stage's card (two
    # when the band straddles a stage boundary).  Group D is TP: every card
    # holds a shard of every layer, so D has NO map and pauses in the natural
    # order, exactly as before.  Derived here from the SAME stage ratio argv_p
    # passes and the SAME chunk geometry both groups were built with; the front
    # picks the pause order from it per flip, against a live NVML free sample.
    # fix 5: from the DERIVED layer split (section 1b'), never from the score
    # vector.  FIX 2: and from the SOLVED cut's split, never from the
    # incumbent's -- the two are the same vector only when the solve happens
    # to land back on the incumbent, and on this tip they do not.  No split ->
    # no map -> the front pauses in the identity order and prints why, which is
    # the honest degradation; a complete-but-wrong map is not, because
    # interleave_pause_order only refuses an INCOMPLETE one.
    p_chunk_cards = chunk_tag_cards(
        p_split, chunk_layers, chunk_count, card_of_stage=[c.nvml_index for c in cards]
    ) if p_split else {}
    src_chunk_cards = {"P": {t: list(v) for t, v in p_chunk_cards.items()}}
    # 23.09. (fnFL2x30): THE DRAFT TAG IS A BAND OF THE LAST STAGE'S CARD.
    # With --draft-kv-on-p the MTP head lives on P's last stage only (W11
    # DRAFT-RESIDENT prices exactly that pool), so its pause refunds that
    # card -- 2770 MiB on the Next-Flash 3080. Absent from this map it sat
    # in the front's `rest` behind every chunk, which was harmless while
    # lane p4 was a SEQ host buffer (x22) and a deadlock over a BAR1 ring:
    # the collector reaches the draft last, D TP2 ran dry at weights_4.
    # `front.orderable_band` places it with its card; the base tag still
    # closes the sleep.
    if p_split and _draft_kv_on_p_requested and cards:
        src_chunk_cards["P"]["weights_draft"] = [int(cards[-1].nvml_index)]
    return src_chunk_cards

def _main_step43_log_f_pdflip_flip_or(*, _launch_guard, cards, chunk_layers, cut, log, n_layers, p_split, p_split_note, src_chunk_cards):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    log(f"PDFLIP-FLIP-ORDER MAP group=P (SOLVED cut {cut.stage_ratio or '(gapped)'}/"
        f"{cut.attn_stage_ratio or '(gapped)'} -> REALIZED layer split "
        f"{p_split if p_split else 'REFUSED (' + p_split_note + ') -> NO MAP, identity order'} "
        f"over {n_layers} layers, {chunk_layers} layers per chunk, "
        f"nvml {[c.nvml_index for c in cards]} in stage order): {src_chunk_cards['P']}; "
        f"group=D TP -> no map (uniform across cards). Boot weg2dk4 died because the interleave paused P's tag k "
        f"against D's tag k while P's bytes for k=0..5 were on OTHER cards than the one D was allocating on.")
    # #1361a: the front is up from here on and carries the same latch on the
    # same terms, so the launcher-side guard hands over rather than doubling.
    if _launch_guard is not None:
        # 12.10.2026 (NF W98 HOST-WARN): hand over only. The guard WARNS (marker
        # HOST-RAM-WARN W98) and never returns a verdict that stops the boot, so
        # the old `PdFlipLaunchHostLatched` raise that stood here is gone (the class
        # stays defined: it is part of the refusal tuple and of old test imports).
        _launch_guard.stop("front takes over")
