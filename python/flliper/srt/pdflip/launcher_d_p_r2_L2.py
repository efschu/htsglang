"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import os
import re
import sys
import time

from flliper.srt.pdflip import DEFAULT_P_BS
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import rank_state as rank_state_mod
from flliper.srt.pdflip import state_file as state_file_mod
from flliper.srt.pdflip import topology as _topo
from flliper.srt.pdflip.launcher_d_leaf import CARD_POWER_MARKER
from flliper.srt.pdflip.launcher_d_leaf import P_BARLINK_BAR1_WINDOW_MIB
from flliper.srt.pdflip.launcher_d_leaf import VISION_OFF
from flliper.srt.pdflip.launcher_d_leaf import _SPEC_FORM
from flliper.srt.pdflip.launcher_d_leaf import _argv_scalar
from flliper.srt.pdflip.launcher_d_leaf import _csv
from flliper.srt.pdflip.launcher_d_leaf import _csv_floats
from flliper.srt.pdflip.launcher_d_leaf import admin_key_flag
from flliper.srt.pdflip.launcher_d_leaf import d_replayssm_spec
from flliper.srt.pdflip.launcher_d_leaf import early_read_flags
from flliper.srt.pdflip.launcher_d_mid import CONTEXT_LENGTH_TOKENS
from flliper.srt.pdflip.launcher_d_mid import canonical_marker_counts
from flliper.srt.pdflip.launcher_d_mid import d_effective_seats
from flliper.srt.pdflip.launcher_d_p_r2_L0 import DEVTOOLS_DIR
from flliper.srt.pdflip.launcher_d_p_r2_L0 import EVIDENCE_DIR
from flliper.srt.pdflip.launcher_d_p_r2_L0 import GPU_ARB
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PP_ATTN_STAGE_RATIO_SCORES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PP_STAGE_RATIO_SCORES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import SPEC_NUM_DRAFT_TOKENS
from flliper.srt.pdflip.launcher_d_p_r2_L0 import boot_state_write
from flliper.srt.pdflip.launcher_d_p_r2_L0 import d_verify_window
from flliper.srt.pdflip.launcher_d_p_r2_L0 import memts_csv_path
from flliper.srt.pdflip.launcher_d_p_r2_L1 import BARLINK_BAR1_CAP_CYCLES
from flliper.srt.pdflip.launcher_d_p_r2_L1 import COLLECTIVE_CENSUS_INTERVAL
from flliper.srt.pdflip.launcher_d_p_r2_L1 import MEMTS
from flliper.srt.pdflip.launcher_d_p_r2_L1 import RANDOM_SEED
from flliper.srt.pdflip.launcher_d_p_r2_L1 import calib_log_accept_of
from flliper.srt.pdflip.launcher_d_p_r2_L1 import common_flags
from flliper.srt.pdflip.launcher_d_p_r2_L1 import d_extend_vote_fixed_live_material
from flliper.srt.pdflip.launcher_d_p_r2_L1 import d_replayssm_spec_ring_len
from flliper.srt.pdflip.launcher_d_p_r2_L1 import memts_pgrep_pattern
from flliper.srt.pdflip.launcher_d_p_r2_L1 import p_draft_kv_flags
from flliper.srt.pdflip.launcher_d_p_r2_L1 import p_transient_support
from flliper.srt.pdflip.launcher_log_plan import Log
from flliper.srt.pdflip.launcher_p_l3_leaf import _group_env_value
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst
from flliper.srt.pdflip.launcher_p_l3_leaf import memts_owner_pattern
from flliper.srt.pdflip.launcher_p_l3_leaf import p_log_model_path
from flliper.srt.pdflip.launcher_p_mid import _model_config
from flliper.srt.pdflip.launcher_p_mid import p_micro_batch_flags
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import SPEC_ALGORITHM
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import rank_state_dir_for
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import state_path
from flliper.srt.pdflip.launcher_refuse_proc_leaf import PROFILE_QWEN27B
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _proc_cmdline
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _refuse_if_extra_drops_transient_lmo
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _refuse_if_extra_raises_budget
from flliper.srt.pdflip.launcher_refuse_proc_leaf import read_card_power_record
from flliper.srt.pdflip.launcher_refuse_proc_leaf import spec_form_is_dflash
from flliper.srt.pdflip.launcher_resolve_stop import p_prefill_graph_flags
from flliper.srt.pdflip.launcher_resolve_xchg_r2_L1_1 import HICACHE_DRAFT_TIER_ENV
from flliper.srt.pdflip.launcher_resolve_xchg_r2_L1_1 import _write_state
from flliper.srt.pdflip.launcher_resolve_xchg_r2_L1_1 import rank_vram_display
from flliper.srt.pdflip.launcher_resolve_xchg_r2_L1_1 import register_helper
from flliper.srt.pdflip.launcher_resolve_xchg_r2_L1_1 import resolve_hicache_draft_tier
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import GroupSpec
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_xchg_vision_leaf import strip_speculative_flags
from flliper.srt.planner import p_card_chunk as _p_card
import json
import shlex
import signal
import subprocess

PORT_P = 30031
DEADMAN = f"{DEVTOOLS_DIR}/boot_deadman.sh"
#: Group P's device mamba pool (--max-mamba-cache-size); see argv_p.
P_MAX_MAMBA_CACHE_SIZE = 24
#: deadmen that never probe their own port (boot_deadman.sh TIER2=0): the two groups, whose ports sit behind the front.
GROUP_DEADMEN_TIER1_ONLY = ("P", "D")
#: Seconds between two owner-liveness polls of the memts supervisor.
MEMTS_OWNER_POLL_S = 5
#: The supervisor: logger as a background child (same pgid), then poll the
#: owner pattern; "$4" empty = no owner watch (only teardown stops it).
_MEMTS_SUPERVISOR_SH = (
    '"$1" "$2" "$3" & '
    'if [ -n "$4" ]; then '
    'while sleep "$5"; pgrep -f "$4" > /dev/null; do :; done; '
    'kill -TERM 0; '
    'else wait; fi'
)
#: model -> the arena page widths `_pdflip_arena_ledger_terms` last priced
_ARENA_BOOKED_WIDTHS: Dict[str, List[int]] = {}
#: Two limits within this many watts are the same limit.
CARD_POWER_TOLERANCE_W = 5.0

def hicache_draft_tier_env() -> Dict[str, str]:
    """What BOTH groups' environment carries for the tier: ``off`` only."""
    tier, _ = resolve_hicache_draft_tier()
    return {HICACHE_DRAFT_TIER_ENV: "off"} if tier == "off" else {}

def hicache_draft_tier_line() -> str:
    """The one launcher line naming the tier (never silent)."""
    tier, why = resolve_hicache_draft_tier()
    if tier == "off":
        return (
            f"PDFLIP HICACHE-DRAFT-TIER: off ({why}) -- the draft gets no HiCache space: "
            "no draft host pool on any rank, no draft arena, no draft write-back, no "
            "draft lookup/read at a restore or at admission; D builds its draft context "
            f"cold (#993 zeros + 1 bootstrap round); {HICACHE_DRAFT_TIER_ENV}=off in both "
            "groups' environment, the ledger prices no draft arena"
        )
    return (
        f"PDFLIP HICACHE-DRAFT-TIER: on ({why}) -- the draft tier as before (draft host "
        "pools, draft arena, write-back and restore of draft pages)"
    )

def w10_skip_no_producer_line() -> str:
    """The W10/W11 skip line when group P carries no drafter. Its ROUTE
    sentence follows the RESOLVED tier (H53): the old text ("D still asks the
    carrier for draft pages ... its draft reads miss") is false once the draft
    gets no HiCache space -- D asks for none, so the line must not tell the
    reader to expect misses that no longer happen."""
    head = (
        "W10/W11 SKIPPED (--draft-kv-on-p off): group P's argv carries no --speculative-* flag, so there is "
        "no draft-KV producer to grade -- no drafter identity to match against D's and no last-stage draft "
        "residue to hold against a budget. "
    )
    if resolve_hicache_draft_tier()[0] == "off":
        route = (
            "ROUTE UNDER THIS ARM: HICACHE-DRAFT-TIER off -- group D keeps its own NEXTN head but the draft "
            "gets no HiCache space, so D asks the carrier for NO draft pages (no draft lookup, no DRAFT-PRESENCE "
            "probe, no '#993 draft L3 READ') and builds its draft context cold after every flip (#993 zeros + "
            "1 bootstrap round). Expect draft_pages=0 on the front's served lines; that is this arm, not a "
            "carrier fault. "
        )
    else:
        route = (
            "ROUTE UNDER THIS ARM: group D keeps its own NEXTN head and still "
            "asks the carrier for draft pages, but group P writes none, so D's draft reads miss and its draft "
            "state is COLD after every flip. Expect draft_pages=0 on the front's served lines; that is this "
            "arm, not a carrier fault. "
        )
    return head + route + "KV and Mamba across the flip are untouched."

def d_replayssm_spec_flags() -> List[str]:
    """Group D's ReplaySSM spec-ring flags -- empty unless --d-replayssm-spec on.

    ONE FACTORY with :func:`spec_plan_fields`: the argv this ships and the
    PlanInputs the launcher prices D with read the same switch, so the D pool
    the boot sizes and the one the launcher publishes cannot disagree about
    the "speculative intermediate state" post."""
    if not d_replayssm_spec():
        return []
    return [
        "--enable-linear-replayssm-spec",
        "--linear-replayssm-cache-len", str(d_replayssm_spec_ring_len()),
    ]

def d_replayssm_spec_line() -> str:
    """The one launcher line naming group D's verify form, with the metal gate
    the ON arm must pass before it may become the default."""
    if not d_replayssm_spec():
        return (
            "PDFLIP D-REPLAYSSM-SPEC: off -- group D's GDN target verify writes "
            "per-draft intermediate states (the 'speculative intermediate state' "
            "post); explicit --d-replayssm-spec off, or a registry row without "
            "replayssm"
        )
    accept = "DFLASH" if spec_form_is_dflash() else "MTP (NEXTN)"
    return (
        "PDFLIP D-REPLAYSSM-SPEC: on -- METAL GATE PASSED (27B xsn436 D-KV +21/+13 %, "
        "NF x172/x174), the registry row's form (replayssm) unless the flag says "
        "off: group D runs "
        f"--enable-linear-replayssm-spec (ring L={d_replayssm_spec_ring_len()}, "
        f"verify window {d_verify_window()}, fold every commit, bf16 checkpoint "
        "with hi/lo compensation). Expect in D's log: 'GDN ReplaySSM SPEC ring "
        "allocated' and 'ReplaySSM spec ring: 'speculative intermediate state' "
        "priced at ...' per rank, the KV budget posts line with the smaller post, "
        "and a smaller 'Mamba Cache is allocated' line. The gate it passed: "
        f"{accept} acceptance length vs the off arm, needle MATCH at 262k, greedy "
        "A/B on fixed prompts (first tokens identical, logprob drift), a 4-8k "
        "token generation without degeneration."
    )

def d_replayssm_spec_plan_form(ns):
    """H64 (NF line): group D's verify form for the D planner
    (``expert_residency.plan_d_residency``: the #145 FRACTION-SOLVE budget and
    KARTE D) -- ``None`` unless --d-replayssm-spec on, so the default form is
    priced exactly as before.

    The planner's references (fnFL2x151 + x158, x141 + x144) were measured on
    the recurrent verify: their 'speculative intermediate state' post and the
    peak of their card carry the per-draft intermediate states. Without this
    form the planner would keep booking them under the ring and the freed
    bytes would never reach the expert-row edge (D's KV is capped at the
    context on the Next-Flash form, so the edge is where they go).

    Read off the argv D runs: the window (:func:`d_verify_window`),
    --max-running-requests and --mamba-ssm-dtype as --extra-d ships them (the
    last value wins, as in argparse), else D's own ``--d-bs`` and the SSM dtype
    the launcher's early-read flags give the 27B profile; None = the
    checkpoint's.
    """
    if not d_replayssm_spec():
        return None
    from flliper.srt.planner.expert_residency import ReplaySSMSpecForm

    extra = str(getattr(ns, "extra_d", "") or "")
    mrr = d_effective_seats(ns)
    ssm = _argv_scalar(extra, "--mamba-ssm-dtype")
    _row = pdflip_form.profile_row(getattr(ns, "profile", None))
    if ssm is None and _row is not None and _row.early_read_flags:
        ssm = _argv_scalar(" ".join(early_read_flags("D")), "--mamba-ssm-dtype")
    return ReplaySSMSpecForm(
        ring_len=d_replayssm_spec_ring_len(),
        draft_tokens=d_verify_window(),
        max_running=max(1, mrr),
        ssm_dtype=ssm,
    )

def spec_plan_fields() -> Dict[str, object]:
    """The PlanInputs fields of the form (d_plan_inputs)."""
    ring = (
        {"linear_replayssm_spec_ring_len": d_replayssm_spec_ring_len()}
        if d_replayssm_spec()
        else {}
    )
    if spec_form_is_dflash():
        return {
            "speculative_algorithm": "DFLASH",
            "speculative_num_draft_tokens": int(_SPEC_FORM["block"]),
            "speculative_draft_model_path": str(_SPEC_FORM["draft_path"]),
            **ring,
        }
    return {
        "speculative_algorithm": SPEC_ALGORITHM,
        "speculative_num_draft_tokens": SPEC_NUM_DRAFT_TOKENS,
        **ring,
    }

def p_prefill_transient_vector_mib(chunk_tokens: int, profile: Optional[str] = None) -> Tuple[float, ...]:
    """Group P's MEASURED prefill transient for ``chunk_tokens``, MiB per
    stage (#114, H41) -- linear between the measured support points; a chunk
    outside them is REFUSED by name (W131), never extrapolated. #242: the
    profile's ``P_ACTIVATION_MIB`` record replaces its support point."""
    try:
        return _p_card.transient_vector_mib(p_transient_support(profile), int(chunk_tokens))
    except _p_card.PChunkUnmeasured as exc:
        raise PdFlipLaunchRefused(str(exc)) from None

def argv_p(
    py: str,
    model: str,
    budgets: List[int],
    s_gb: int,
    m_mib: int,
    store_cfg: str,
    extra: List[str],
    # #1356 [fix] EVERYTHING OPTIONAL IS KEYWORD-ONLY FROM HERE.
    #
    # 2cc618b819 inserted `vision` in the middle of this list while BOTH
    # production callers pass positionally (:10147, the form-key build, and
    # :10578, shipped_argv_p). SEVEN parameters shifted one place: vision got
    # depth, depth got window_mib, window_mib got random_seed, random_seed got
    # barlink_cap_cycles, census_interval got draft_kv_on_p, and so on.
    #
    # WHAT SAVED THIS BOOT WAS A TYPE ACCIDENT, not a guard: `window_mib` is
    # the string "24,PP_0=96", so `str(int(depth))` raised and every boot died
    # loudly. Had that value been purely numeric the launch would have SUCCEEDED
    # with the seed, the BAR1 window and the census interval silently swapped,
    # and the form key hashed over an argv nobody ever meant -- a wrong ring
    # table inherited on a form that never existed. A crash was the good case.
    #
    # So the repair is not "move the new parameter": it is that a positional
    # caller can no longer reach ANY optional term. Appending fixes one
    # instance; the star fixes the class, for whoever adds the next one.
    *,
    p_bs: int = DEFAULT_P_BS,
    max_kv_per_request: int = CONTEXT_LENGTH_TOKENS,
    stage_ratio: Optional[str] = None,
    attn_stage_ratio: Optional[str] = None,
    write_policy: str = "write_back",
    depth: int = 0,
    window_mib: str = P_BARLINK_BAR1_WINDOW_MIB,
    random_seed: int = RANDOM_SEED,
    barlink_cap_cycles: int = BARLINK_BAR1_CAP_CYCLES,
    census_interval: int = COLLECTIVE_CENSUS_INTERVAL,
    draft_kv_on_p: bool = True,
    admin_api_key: Optional[str] = None,
    p_max_total_tokens: Optional[int] = None,
    vision: str = VISION_OFF,
    profile: str = PROFILE_QWEN27B,
    # #1386: forwarded to `common_flags` unchanged; see that function.
    hicache_disabled: bool = False,
    # #1369: forwarded to `common_flags` unchanged; see that function.
    weights_cpu_backup: bool = True,
    # #1264 follow-up (fnFL2 v39): the producer's four speculative flags, read
    # off group D's argv by :func:`p_draft_kv_flags`.  Default = the module
    # constants, which is what every pre-Next-Flash arm shipped.
    spec_flags: Optional[Sequence[str]] = None,
) -> List[str]:
    _refuse_if_extra_raises_budget(budgets, list(extra or ()), "P")
    _refuse_if_extra_drops_transient_lmo(vision, extra, "P")
    # THE COUNT FLAGS ARE THE CONTIGUOUS FORM, AND ONLY THAT (#1240 FOLLOW FIX
    # 1). --pp-stage-ratio/--pp-attn-stage-ratio are per-stage COUNTS that
    # server_args hands to derive_pp_layer_split, which builds a CONTIGUOUS
    # split from them. A gapped map is not expressible that way at all:
    # * the user's own layout (48 GDN on the 5090, 8+8 attention on the 3080s)
    #   is `48,8,8 / 0,8,8`, and derive_pp_layer_split raises
    #   "--pp-attn-stage-ratio entries must be positive integers" before a
    #   single weight loads -- boot weg2pp2 arm P_G1 died exactly there;
    # * a gapped map whose attention counts are all positive is worse than
    #   refused, it is ACCEPTED and derives something else (52,6,6 / 4,6,6 ->
    #   [19,24,21]), so the argv would contradict FLLIPER_PP_LAYER_SET.
    # Under a gapped map the SET published in env_p IS the layout and
    # make_layers/model_runner resolve ownership from get_pp_layer_set, never
    # from the count form -- so the count copy is the second set of books and
    # it is dropped, rather than kept and asserted against.
    # FIX 2: THE INCUMBENT IS READ, NOT RESTATED. ``None`` (nobody solved or
    # pinned a cut) renders the incumbent score vectors from their ONE
    # definition, at call time -- so patching the constant moves the flag, and
    # a bare "32,18,14" here cannot drift away from it again. ``""`` still
    # means OMIT BOTH FLAGS (the gapped kind, FOLLOW FIX 1) and is left alone.
    if stage_ratio is None:
        # #1362 [22-fix] THE CALIBRATION IS CHECKED AGAINST THE MODEL, here,
        # where the incumbent vector is otherwise handed over unasked. The
        # constants below are a 64-LAYER measurement from boot bsscale and
        # carry no model reference; on a 32-layer checkpoint the chain ends in
        # a bare ValueError from the cut solver ("the anchor stage 1 holds 18
        # linear and 0 attention layers in the calibration cut"), which is the
        # only unnamed refusal in the whole launch path. Reading the model's own
        # record first turns that into a verdict that names the missing
        # measuring run -- and, when the run HAPPENED and refused, the reason it
        # refused (the 4B-FP8 record carries KERNEL-ARCH-ABBRUCH verbatim, which
        # no amount of booking another window fixes).
        _dg, _dg_why = host_ledger.checkpoint_digest(model)
        if _dg:
            _calib, _calib_why = host_ledger.read_pp_calibration(_dg)
            _depth = host_ledger.checkpoint_layers(model)
            if _calib is None and _depth is not None and _depth != _pconst("CALIBRATION_LAYERS", profile):
                raise host_ledger.refuse_foreign_calibration(_dg, _calib_why)
            if _calib is not None:
                stage_ratio = _csv(tuple(_calib["measured_counts"]))
        stage_ratio = stage_ratio or _csv(P_PP_STAGE_RATIO_SCORES)
    if attn_stage_ratio is None:
        attn_stage_ratio = _csv(P_PP_ATTN_STAGE_RATIO_SCORES)
    if bool(stage_ratio) != bool(attn_stage_ratio):
        raise PdFlipLaunchRefused(
            "W40 PdFlipPPCutRefused: --pp-stage-ratio %r and "
            "--pp-attn-stage-ratio %r must be given together or omitted "
            "together. They are one statement of one layout; half of it "
            "would let derive_pp_layer_split snap the other half silently "
            "(the #505(a) class)." % (stage_ratio, attn_stage_ratio)
        )
    # SOLVED unless the operator pinned them; the launcher prints the PP-CUT
    # provenance line either way. Never a hand constant reaching this line
    # unannounced -- and, since FOLLOW FIX 1, never a SECOND statement of a
    # layout the --pp-layer-set wire already carries.
    ratio_flags = (
        ["--pp-stage-ratio", stage_ratio, "--pp-attn-stage-ratio", attn_stage_ratio]
        if stage_ratio
        else []
    )
    return [py, "-m", "flliper.launch_server"] + common_flags(
        model, s_gb, m_mib, store_cfg, max_kv_per_request, write_policy, "P",
        random_seed, barlink_cap_cycles, census_interval,
        hicache_disabled=hicache_disabled,
        # Task #58 xsn403: argv_p accepted `vision` but never handed it on, so
        # P was launched with vision_model_flags(VISION_OFF) = --no-enable-multimodal
        # and the transient stage refused to arm (W111). The P group is the
        # ONLY place the vision form reaches the model argv; D stays text-only.
        vision=vision,
        profile=profile,
        weights_cpu_backup=weights_cpu_backup,
        n_cards=len(budgets),
    ) + [
        # C1/K1: P's own bs. Concurrency for the front's leg-1 fan-out AND
        # the size of P's req_to_token_pool (R-13), which is why it is
        # resolved before the budget solve and printed with it.
        "--max-running-requests", str(p_bs),
        # HW-GENERIC 1002 S2 (enabling): P = PP<N> over every card (TP1);
        # from the topology, "--pp-size 3" on the N=3 release.
        "--tp-size", "1", "--pp-size", str(_topo.release_topology(len(budgets)).p_pp),
        # #692 MICROBATCH DEPTH, group P only -- group D runs pp_size=1 and a
        # pipeline depth is meaningless there. STATED even at 0 so the argv is
        # an honest statement of what the boot runs, and published ONCE as a
        # group constant: every rank reads the same token off the same argv,
        # so no rank can derive a different depth (the ring size is a
        # collective property -- two ranks disagreeing about pp_loop_size is
        # the v7pp12 starvation, not a tuning difference). Solved by
        # solve_p_depth from the previous boot's own PP-BUBBLE line; the
        # launcher prints the PDFLIP P-DEPTH provenance line either way.
        "--pp-async-batch-depth", str(int(depth)),
        # P is the pp_size>1 group, so the overlap schedule is refused HERE
        # and only here (server_args.py:19507). Passing it explicitly rather
        # than letting the post-process pass force it keeps the argv an
        # honest statement of what this group runs.
        "--disable-overlap-schedule",
    ] + ratio_flags + [
        "--rank-gpu-memory-mib", ",".join(str(b) for b in budgets),
        # #1235: P's window had no comment while D's carried the whole #1234 C1
        # derivation. Its provenance is the SAME arithmetic, from the other
        # side: the two groups' windows are sized to fit TOGETHER, P 24+96 and
        # D 16+32+40 = 208 of the 224 MiB usable per 3080. That sentence lived
        # only in state.deviations; it is cited at P_BARLINK_BAR1_WINDOW_MIB now.
        "--barlink-bar1-window-mib", window_mib,
    # #1233 draft KV across the flip (C15): P carries D's four speculative
    # flags BYTE-FOR-BYTE (they hash into the drafter identity, W5) and is
    # silenced by the producer flag, which is deliberately not hashed.
    # #1264: AND IT IS A SWITCH.  This is the ONLY place the two forms differ,
    # and it is a list rather than five branches, so `off` cannot half-apply.
    # Everything downstream reads the fact back off this argv through
    # ring_table.p_carries_drafter -- including the form key, which is a
    # blacklist and therefore hashes the two forms apart with no code at all.
    # Group D is NOT touched by this switch: D keeps its own NEXTN head in
    # both forms (argv_d, below), because `off` removes the PRODUCER, not
    # speculative decode.
    ] + (list(p_draft_kv_flags(()) if spec_flags is None else spec_flags)
         if draft_kv_on_p else []) + (
        # #1305 item 2: the cap is the SHIPPED cut's priced pool, handed in by
        # the caller that holds PCutFacts; never a constant here.  Head-scoped
        # (see P_DRAFT_KV_FLAGS); absent when no cut was solved (the sentinel
        # helpers that only read flags off this argv).
        ["--max-total-tokens", str(int(p_max_total_tokens))]
        if draft_kv_on_p and p_max_total_tokens is not None
        else []
    ) + [
        "--port", str(PORT_P),
    ] + admin_key_flag(admin_api_key) + (
        # Boots xsn127-134 (2026-09-15): P's device mamba pool of 5 left a
        # pin budget of 1 for the write_back anchors ("skipping the host
        # backup"), so the store held KV without anchors and D's leg-2 reads
        # came back short. 24 slots (pin budget 20) carried three parallel
        # 11k requests. An operator value in --extra-p wins.
        []
        if any(str(a).startswith("--max-mamba-cache-size") for a in extra)
        else ["--max-mamba-cache-size", str(P_MAX_MAMBA_CACHE_SIZE)]
        # H25: no draft on P means ZERO --speculative-* tokens on P, including
        # the ones an arm passes through --extra-p for both forms (the
        # Next-Flash arm's --speculative-draft-model-path).
    ) + p_micro_batch_flags(p_bs, list(extra or ())) + (
        # 27B --p-prefill-graph: [] when off (the default argv is unchanged).
        p_prefill_graph_flags()
    ) + (
        list(extra) if draft_kv_on_p else strip_speculative_flags(extra)[0])

def canonical_state_gate(spec: GroupSpec, expected: int, log: Log,
                         kv_owner_ranks: Sequence[int] = ()) -> None:
    """IPC Phase 1: the launcher half of W7/W10 decided on the ranks'
    RankState records (pdflip/rank_state.py), not on counted log lines.

    rc12z29d (28.09. 20:29Z) booted D clean and was refused as 'kv x1 blob
    x1': under the uneven-DCP cut the Form A workers print '#239 F14
    KV-WORKER-WINDOW' and the count knew only the older worker line. The log
    count still runs beside the records and a disagreement is REPORTED
    ('IPC MISMATCH'), but it no longer decides."""
    state_dir = rank_state_dir_for(spec)
    states, bad = rank_state_mod.read_group_states(state_dir)
    verdict = rank_state_mod.grade_canonical(states, expected, group=spec.name)
    for b in bad:
        verdict.ok = False
        verdict.reasons.append(f"unreadable record {b}")
    # #239 F14 (a33ee80394), kept under the records: a worker the launcher's own
    # argv gives token rows must REPORT the KV page as applicable and active --
    # a rank that under-reports applicability would otherwise pass as 'n/a'.
    _by_tp = {s.tp_rank: s for s in states if s.pp_rank == 0}
    for r in kv_owner_ranks:
        s = _by_tp.get(int(r))
        if s is None or not (s.kv_page_applicable and s.kv_page_active):
            verdict.ok = False
            verdict.reasons.append(
                f"tp{r} owns token rows under the cut but reports kv "
                f"{'missing' if s is None else f'applicable={s.kv_page_applicable} active={s.kv_page_active}'}")
    log(f"W7/W10 launcher half, group {spec.name} RankState ({state_dir}): {verdict.line()}")
    rank_vram_display(spec, log, states=states)
    # IPC-LOG-EXCEPTION: reported, never decides (see canonical_marker_counts).
    n_kv, n_blob, n_worker = canonical_marker_counts(spec.log)
    if (n_kv, n_blob) != (verdict.n_kv, verdict.n_blob):
        log(f"IPC MISMATCH W7/W10 group {spec.name}: log count kv x{n_kv} blob x{n_blob} "
            f"(worker lines x{n_worker}) vs RankState kv x{verdict.n_kv} blob x{verdict.n_blob} "
            f"-- the records decide, the log count is reported only")
    boot_state_write(log, fields={
        f"groups.{spec.name}.ranks": [json.loads(s.to_json()) for s in states],
        f"groups.{spec.name}.rankstate_dir": state_dir,
        f"invariants.W7_W10_CanonicalWindow.{spec.name}": {
            "verdict": "pass" if verdict.ok else "refuse",
            "per_rank": [{"rank": k, "state": v} for k, v in sorted(verdict.by_rank.items())],
            "why": verdict.line()},
    })
    if not verdict.ok:
        raise PdFlipLaunchRefused(
            f"W7 PdFlipMambaBlobAbsent / W10 PdFlipCanonicalPageMissing (launcher half, RankState): "
            f"group {spec.name} {verdict.line()}")

def arm_deadman(log: Log, boot_log: str, port: int, pattern: str, probe_s: int, tag: str, name: str, dry: bool) -> int:
    out = f"{GPU_ARB}/deadman_{tag}_{name}.out"
    # IPC Phase 2 (IPC-VERBRAUCHER-27B A4/N4): the deadman writes its verdict into the
    # boot's state (event deadman_verdict + stop_request.json) through the ONE writer,
    # and reads READY from groups.<G>.state -- PDFLIP_STATE_DIR is inherited; these name
    # the group, the interpreter and the writer file (the image has no flliper on PATH).
    ipc = (f"PDFLIP_DEADMAN_GROUP={shlex.quote(name)} PDFLIP_PY={shlex.quote(sys.executable)} "
           f"PDFLIP_STATE_FILE_PY={shlex.quote(state_file_mod.__file__)} ")
    # #1158c root: the GROUP deadmen run tier 1 only, said explicitly -- PROBE_S spaced past the boot alone did not hold
    # (boot_deadman.sh started its probe clock at 0, so ONE /health_generate hit the group port at grace end and ran as D's
    # first forward inside a flip's quiesce window, metal 10.10. 01:07:32Z -> W120). TIER2=0 keeps every synthetic generate
    # off the group ports; the front's deadman keeps tier 2 (it reaches the awake group through the front).
    tier1_only = name in GROUP_DEADMEN_TIER1_ONLY
    if tier1_only:
        ipc += "TIER2=0 "
    cmd = f"{ipc}GRACE_S=600 PROBE_S={probe_s} setsid {DEADMAN} {shlex.quote(boot_log)} {port} {shlex.quote(pattern)} > {shlex.quote(out)} 2>&1 & echo $!"
    if dry:
        log(f"DRY-RUN: would arm deadman: {cmd}")
        return 0
    pid = int(subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip() or 0)
    if pid:
        register_helper(tag, f"deadman_{name}", pid, log)
    time.sleep(1)
    proof = subprocess.run(["pgrep", "-af", f"boot_deadman.sh {boot_log}"], capture_output=True, text=True).stdout.strip()
    n = len(proof.splitlines())
    log(f"deadman {name}: pid {pid} GRACE_S=600 PROBE_S={probe_s}{' TIER2=0' if tier1_only else ''} pattern={pattern!r} verdict -> {out}; pgrep proof: {n} process(es) whose argv carries THIS log ({proof or 'NONE -- UNKNOWN, never alive'})")
    return pid

def register_boot_group(tag: str, name: str, pid: int, log: Optional[Log] = None) -> str:
    return register_helper(tag, name, pid, log, suffix="grp")

def start_memts(state: "BootState", log: Log) -> int:
    """Start the logger under its owner-watching supervisor, record the pid in
    the boot state AND persist the state before returning: a crash or refusal
    between here and the next _write_state() must still leave the pid on file
    for `--teardown PATH`."""
    csv = memts_csv_path(state.tag)
    owner = memts_owner_pattern(state.tag)
    if not re.search(owner, _proc_cmdline(os.getpid())):
        log(f"memts owner watch OFF: this launcher's own argv does not match {owner!r} "
            f"-- only teardown() stops the logger")
        owner = ""
    p = subprocess.Popen(
        ["bash", "-c", _MEMTS_SUPERVISOR_SH, "memts-guard", MEMTS, csv, "5", owner,
         str(MEMTS_OWNER_POLL_S)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    state.memts_pid = p.pid
    state.helper_pids.append(p.pid)
    _write_state(state)
    register_helper(state.tag, "memts", p.pid, log)
    log(f"mem time series pid {p.pid} (pgid, supervisor; owner watch {owner!r}, "
        f"poll {MEMTS_OWNER_POLL_S} s) csv {csv} (started before group P: launch AND run "
        f"moments sampled; state {state_path(state)} carries the pid)")
    return p.pid

def stop_memts(tag: str, pid: int) -> str:
    """TERM the logger's process group (only if its leader is still OURS, by
    argv -- a recycled pid is never signalled), then TERM whatever the
    anchored pgrep net still finds for THIS tag. Returns the teardown line."""
    killed: List[int] = []
    csv = memts_csv_path(tag)
    if pid and csv in _proc_cmdline(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
            killed.append(pid)
        except OSError:
            pass
    net: List[int] = []
    if tag:
        out = subprocess.run(["pgrep", "-f", memts_pgrep_pattern(tag)],
                             capture_output=True, text=True).stdout.split()
        for s in out:
            q = int(s)
            if q == os.getpid() or q in killed:
                continue
            try:
                os.kill(q, signal.SIGTERM)
                net.append(q)
            except OSError:
                pass
    verdict = "killed" if killed or net else "already-gone"
    return f"PDFLIP-TEARDOWN memts pid={pid} {verdict} (killpg={killed} pgrep-net={net} csv={csv})"

def _pdflip_arena_ledger_terms(model: str, group_envs=None) -> dict:
    """#1432: the L2 arena's host bytes and the REAL fallback pool sizes, for
    the ledger. Sizes come from the same environment the child processes get
    (FLLIPER_HICACHE_ARENA_GIB / _MAMBA_SLOTS / _STAGING_GB, FLLIPER_PDFLIP_MAMBA_
    ANCHOR_MIB) and the same geometry the store derives: one KV slot per token
    (all attention layers), one draft slot per KV slot (one layer cell), one
    mamba slot per canonical GDN blob (hicache_migrate.qwen3_5_mamba_spec,
    bf16 states -- 78,446,592 B on Qwen3.8-27B, checked against the arena
    file). Empty dict when the arena host tier is off.

    29.09. (z30w-park): the sizes are read through ``_group_env_value`` --
    the NF arm sets ARENA_GIB=4 / MAMBA_SLOTS=32 in the group env only, and
    the launcher env's defaults (22 / 140) priced the arena at 23.75 GiB
    against 6.27 GiB measured in /dev/shm."""
    import logging as _logging
    if os.environ.get("FLLIPER_HICACHE_ARENA_HOST", "1") != "1":
        return {}
    try:
        cfg = _model_config(model)
        text_cfg = cfg.get("text_config") or cfg
        n_layers = int(text_cfg["num_hidden_layers"])
        kinds = text_cfg.get("layer_types") or []
        n_attn = sum(1 for k in kinds if str(k) == "full_attention")
        n_lin = len(kinds) - n_attn
        from flliper.srt.planner import pp_cut as _pp_cut
        cell = int(_pp_cut.kv_mib_per_token_per_attn_layer_from_config(cfg, "fp8_e4m3", n_layers) * (1 << 20))
        kv_page = cell * n_attn
        from flliper.srt.mem_cache.pool_host.arena_pool import arena_slots_for
        kv_gib = float(_group_env_value("FLLIPER_HICACHE_ARENA_GIB", group_envs, "22"))
        kv_slots = arena_slots_for(kv_gib, kv_page)
        # 29.09.: the ranks cut the arena in CANONICAL pages when the env
        # names one (HiCacheFile._arena_for: slots = GIB // canonical KV
        # extents; z30w wrote arena-786432.bin = 5461 x 64-token pages). The
        # count follows that page; the terms below stay in tokens (one KV
        # slot = one token, the draft 1:1), so the bytes are the same.
        _canon_page = int(float(_group_env_value("FLLIPER_HICACHE_ARENA_KV_PAGE_BYTES", group_envs, "0") or 0))
        if _canon_page > 0 and kv_page > 0 and _canon_page % kv_page == 0:
            kv_slots = arena_slots_for(kv_gib, _canon_page) * (_canon_page // kv_page)
        from flliper.srt.mem_cache.hicache_migrate import qwen3_5_mamba_spec
        blob = qwen3_5_mamba_spec(text_cfg, num_linear_layers=n_lin, units=1,
                                  temporal_itemsize=2, conv_itemsize=2).total_bytes if n_lin else 0
        mamba_slots = int(float(_group_env_value("FLLIPER_HICACHE_ARENA_MAMBA_SLOTS", group_envs, "140")))
        # xsn267 (17.09.): under --spec-form DFLASH the draft slot is the
        # DFlash draft's own page -- draft_layers x (K+V) x draft kv heads x
        # head_dim x kv itemsize (5 x 2 x 8 x 128 x 1 = 10240 B, the
        # arena-10240.bin the boot actually created: 7.76 GiB) -- not one
        # target attention cell (the NEXTN form's draft). Priced with the
        # target cell the ledger was 7 GiB short and W98 latched at cushion
        # 1.07 GiB (xsn267) with the lanes already released per leg.
        draft_cell = cell
        try:
            if str(_SPEC_FORM.get("form") or "").upper() == "DFLASH":
                _dp = str(_SPEC_FORM.get("draft_path") or "")
                if _dp:
                    _dc = json.load(open(os.path.join(_dp, "config.json")))
                    _dc = _dc.get("text_config") or _dc
                    _kv_item = 1 if "fp8" in "fp8_e4m3" else 2
                    draft_cell = (int(_dc["num_hidden_layers"]) * 2
                                  * int(_dc["num_key_value_heads"]) * int(_dc["head_dim"]) * _kv_item)
        except Exception as _dexc:  # noqa: BLE001 -- unreadable draft config keeps the target cell
            _logging.getLogger("pdflip.launcher").info("PDFLIP-ARENA-LEDGER draft page not derived: %r", _dexc)
        # HICACHE-DRAFT-TIER off: no rank registers a draft host pool, so no
        # draft arena file exists -- the ledger prices none (same resolver as
        # the environment the ranks get, hicache_draft_tier_env).
        draft_slots = kv_slots if resolve_hicache_draft_tier()[0] == "on" else 0
        arena_bytes = kv_slots * kv_page + draft_slots * draft_cell + mamba_slots * blob
        out = dict(
            arena_gib=arena_bytes / (1 << 30),
            staging_gb=float(_group_env_value("FLLIPER_HICACHE_ARENA_STAGING_GB", group_envs, "0.05")),
            anchor_mib=int(float(_group_env_value("FLLIPER_PDFLIP_MAMBA_ANCHOR_MIB", group_envs, "100"))),
        )
        _logging.getLogger("pdflip.launcher").info(
            "PDFLIP-ARENA-LEDGER kv=%d slots x %d B + draft %d x %d B + mamba %d x %d B = %.2f GiB "
            "(term arena_gib); fallback pools staging=%s GB anchor=%s MiB",
            kv_slots, kv_page, draft_slots, draft_cell, mamba_slots, blob, out["arena_gib"],
            out["staging_gb"], out["anchor_mib"])
        # 29.09. (host census): the arena FILE widths this term prices, so the
        # census can tell a sidecar width (the QSA index page) from them
        _pt = (_canon_page // kv_page) if (_canon_page > 0 and kv_page > 0
                                           and _canon_page % kv_page == 0) else 1
        _ARENA_BOOKED_WIDTHS[str(model)] = [w for w in (
            kv_page * _pt, (draft_cell * _pt) if draft_slots else 0, blob) if w]
        return out
    except Exception as exc:  # noqa: BLE001 - a mispriced arena is refused, never guessed
        raise PdFlipLaunchRefused(f"W108 PdFlipArenaLedgerRefused: the arena term could not be derived: {exc!r}")

def d_extend_vote_fixed_live_form(profile: Optional[str], cards: Optional[Sequence[Card]],
                                  model: Optional[str]) -> Tuple[Optional[str], str]:
    """``(digest, why)`` -- :func:`d_extend_vote_fixed_live_material` without the material."""
    dig, _mat, why = d_extend_vote_fixed_live_material(profile, cards, model)
    return dig, why

def stage_fit_family_cost(ns, cards: Sequence[Card], chunk_tokens: int,
                          limits_now: Sequence[Optional[float]], log):
    """--pp-cut-stage-fit: the P cut's stage cost FITTED from a boot's #PGAP
    lines (planner/pgap_stage_fit.py), or a W40 refusal.

    Returns ``(cost, provenance, fitted_log)``. THE POWER RECORD (user order
    24.09.): the fitted boot's per-stage card and power limit come from its
    own front log (:data:`CARD_POWER_MARKER`), else from the operator's
    ``--pp-cut-stage-fit-power`` assertion; a record whose card or limit
    differs from this boot's metal is STALE and refused, unless
    ``--pp-cut-stage-fit-stale-ok`` -- then it is used and the provenance says
    STALE in capitals. No record at all is refused: never reused silently."""
    from flliper.srt.planner import pgap_stage_fit as _fit

    arg = str(getattr(ns, "pp_cut_stage_fit", "") or "").strip()
    if arg == "auto":
        # UNIFY S7: the form's calibration identity (27B line_identity merged, S3)
        path, seen = _fit.newest_pgap_log(
            EVIDENCE_DIR, int(chunk_tokens), len(cards),
            accept=calib_log_accept_of(ns))
        for s in seen:
            log("PP-CUT STAGE FIT auto: " + s)
        if path is None:
            raise PdFlipLaunchRefused(
                "W40 PdFlipPPCutRefused: --pp-cut-stage-fit auto found no P log in "
                f"{EVIDENCE_DIR} with #PGAP lines at chunk {int(chunk_tokens)} and "
                f"{len(cards)} stages (run a boot with --p-hostgap first, or pass a path)")
    else:
        path = arg
    try:
        fitted = _fit.read_pgap_log(path)
        if fitted.chunk_tokens != int(chunk_tokens):
            raise _fit.StageFitRefused(
                f"it ran {fitted.chunk_tokens}-token P chunks, this boot runs "
                f"{int(chunk_tokens)}; the attention cost per chunk is chunk-specific "
                "(192 CTAs at 512 on the 5090), so a fit does not transfer")
        _fit_model = p_log_model_path(path)
        _want = str(getattr(ns, "model", "") or "")
        # Checked where both checkpoints exist on disk (a log naming a path that
        # is gone cannot be compared, and says so in the provenance below).
        if (_fit_model and _want and os.path.isdir(_fit_model) and os.path.isdir(_want)
                and os.path.realpath(_fit_model) != os.path.realpath(_want)):
            raise _fit.StageFitRefused(
                f"it ran checkpoint {_fit_model}, this boot runs {_want}; per-layer "
                "costs are a property of the checkpoint's kernels (INT8 W8A8 vs FP8 "
                "Marlin W8A16), so a fit does not transfer between checkpoints")
        cost, prov = _fit.fit_stage_cost(fitted, [c.name for c in cards])
    except (_fit.StageFitRefused, OSError, ValueError, ZeroDivisionError) as exc:
        raise PdFlipLaunchRefused(f"W40 PdFlipPPCutRefused: --pp-cut-stage-fit {path}: {exc}")
    front = path[: -len(".P.log")] + ".front.log" if path.endswith(".P.log") else ""
    record = read_card_power_record(front) if front else None
    source = "record in %s" % os.path.basename(front)
    asserted = str(getattr(ns, "pp_cut_stage_fit_power", "") or "").strip()
    if record is None and asserted:
        vals = _csv_floats(asserted)
        if len(vals) != len(cards):
            raise SystemExit(
                f"--pp-cut-stage-fit-power {asserted!r}: {len(cards)} per-stage watts expected")
        record = [(c.name, float(v)) for c, v in zip(cards, vals)]
        source = "ASSERTED by --pp-cut-stage-fit-power (the fitted boot printed no record)"
    if record is None:
        raise PdFlipLaunchRefused(
            "W40 PdFlipPPCutRefused: --pp-cut-stage-fit %s carries no power record "
            "(its boot printed no '%s' line) and none was asserted. The per-layer "
            "cost moves with the power limit, so an unrecorded fit is not reused "
            "silently: pass --pp-cut-stage-fit-power W,W,W with the limits that "
            "boot ran at (24.09. user measurement: 400,230,230)"
            % (path, CARD_POWER_MARKER))
    diffs = []
    for r, ((name, lim), c, now) in enumerate(zip(record, cards, limits_now)):
        if name != c.name:
            diffs.append(f"stage {r} card {name!r} -> {c.name!r}")
        elif lim is None or now is None:
            diffs.append(f"stage {r} limit {lim} W -> {now} W (unknown)")
        elif abs(float(lim) - float(now)) > CARD_POWER_TOLERANCE_W:
            diffs.append(f"stage {r} {c.name} {lim:.0f} W -> {now:.0f} W")
    verdict = (
        "power MATCH (%s): %s" % (
            source, ",".join("%.0f" % float(l) for _n, l in record if l is not None))
        if not diffs else "power STALE (%s): %s" % (source, "; ".join(diffs))
    )
    log("PP-CUT STAGE FIT POWER: " + verdict)
    if diffs and not getattr(ns, "pp_cut_stage_fit_stale_ok", False):
        raise PdFlipLaunchRefused(
            "W40 PdFlipPPCutRefused: --pp-cut-stage-fit %s is STALE against this "
            "boot's metal -- %s. Re-measure (a boot with --p-hostgap at the new "
            "limits), or pass --pp-cut-stage-fit-stale-ok to price with it anyway "
            "(the provenance then says STALE)." % (path, "; ".join(diffs)))
    return cost, prov + " | " + verdict, fitted
