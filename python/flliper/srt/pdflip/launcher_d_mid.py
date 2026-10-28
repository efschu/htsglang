"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
import os
import re

from flliper.srt.environ import envs
from flliper.srt.managers import corridor_guard
from flliper.srt.name_compat import canonical_env
from flliper.srt.pdflip import DEFAULT_D_BS
from flliper.srt.pdflip import budget_rest as _budget_rest
from flliper.srt.pdflip import card_identity
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import l15_plan
from flliper.srt.pdflip import weight_exchange_transport
from flliper.srt.pdflip.launcher_d_leaf import DC_RECORD_ENV
from flliper.srt.pdflip.launcher_d_leaf import DC_RECORD_MARGIN_MIB
from flliper.srt.pdflip.launcher_d_leaf import DFLASH_BLOCK_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import DFLASH_DRAFT_PATH_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import DFLASH_PRODUCE_ON_P_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import DFLASH_WINDOW_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import D_REPLAYSSM_SPEC_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import SPEC_FORM_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import TORCH_CACHE_CAP_ENV
from flliper.srt.pdflip.launcher_d_leaf import _D_TOKEN_PLACEMENT
from flliper.srt.pdflip.launcher_d_leaf import _SPEC_FORM
from flliper.srt.pdflip.launcher_d_leaf import _argv_scalar
from flliper.srt.pdflip.launcher_d_leaf import _argv_vector
from flliper.srt.pdflip.launcher_d_leaf import _axis_floor_ranks
from flliper.srt.pdflip.launcher_d_leaf import _csv
from flliper.srt.pdflip.launcher_d_leaf import _csv_floats
from flliper.srt.pdflip.launcher_d_leaf import check_drafter_identity
from flliper.srt.pdflip.launcher_d_leaf import count_marker
from flliper.srt.pdflip.launcher_d_leaf import d_card_vote_lines
from flliper.srt.pdflip.launcher_d_leaf import d_expect_from_p_records
from flliper.srt.pdflip.launcher_d_leaf import d_kv_token_cut
from flliper.srt.pdflip.launcher_d_leaf import d_residue_capture_bs
from flliper.srt.pdflip.launcher_d_leaf import d_seat_vram_armed
from flliper.srt.pdflip.launcher_d_leaf import dc_residue_from_record
from flliper.srt.pdflip.launcher_d_leaf import dc_residue_max_from_records
from flliper.srt.pdflip.launcher_d_leaf import dflash_produce_on_p
from flliper.srt.pdflip.launcher_d_leaf import er_trim_cell
from flliper.srt.pdflip.launcher_p_l3_leaf import D_KV_STAGE_KEYS
from flliper.srt.pdflip.launcher_p_l3_leaf import _last_flag_value
from flliper.srt.pdflip.launcher_p_l3_leaf import model_config_path
from flliper.srt.pdflip.launcher_p_l3_leaf import model_is_gguf_file
from flliper.srt.pdflip.launcher_p_l3_leaf import p_log_model_path
from flliper.srt.pdflip.launcher_p_l3_leaf import parse_group_env
from flliper.srt.pdflip.launcher_refuse_proc_leaf import WEIGHT_SOURCE_DEFAULT
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _published_profile
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _refuse_value_once
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _running_power_by_class
from flliper.srt.pdflip.launcher_refuse_proc_leaf import pick_d_residue_record
from flliper.srt.pdflip.launcher_refuse_proc_leaf import rank_wave_floor
from flliper.srt.pdflip.launcher_refuse_proc_leaf import records_inventory
from flliper.srt.pdflip.launcher_refuse_proc_leaf import set_group_env
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import PdFlipDKvStageWavesRefused
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_xchg_vision_leaf import xchg_bounce_arm_pins_host
from flliper.srt.pdflip.launcher_xchg_vision_leaf import yarn_context_tokens
from flliper.srt.registry import nvml as nvml_registry
import json

DFLASH_PRODUCE_ENV = "FLLIPER_PDFLIP_DFLASH_PRODUCE"
#: --d-token-placement (27B, user 26.09. ~09:40Z/09:45Z: "es muss dynamisch
#: sein ... den kv so zu verteilen wie er optimal schnell ist zur jeweiligen
#: fuelle"). pdflip/d_token_placement.py: NEW D tokens pick the card through the
#: allocator's slot id (weighted owner rule), by effective bandwidth while the
#: pool has room, by capacity when it is tight; stored KV never moves.
#: 'capacity' (default) = today: no env, argv byte-identical.
D_TOKEN_PLACEMENT_DEFAULT = "capacity"
W10_SKIPPED_NO_DRAFT_PAGES = (
    "W10 SKIPPED: P produces no draft pages (--dflash-produce-on-p off)"
)
#: The model context both groups are launched with.  Named once because K9's
#: --max-kv-per-request default IS this number (the as-built cap, decoupled
#: from the model context so it can be lowered without touching the context).
CONTEXT_LENGTH_TOKENS = 262144
ONCARD_MODE_DEFAULT = weight_exchange_transport.ONCARD_MODE_IPC
#: DEEP ANCHOR. The user's physics note of 2026-09-07 (recorded verbatim in
#: the GAPPED block of WEG2_BUILD_DECISIONS_0906.md section 1r): at deep
#: prefixes one full-attention layer costs a 3080 about 0.4 s per chunk at a
#: 262,144-token prefix. That is the second depth the family split needs; it
#: is what makes the optimum a FUNCTION of the design prefix rather than a
#: constant, and it is cited to its record rather than written as a literal.
ATTN_ANCHOR_MS = 400.0
#: HW-GENERIC 1002: the calibration class ATTN_ANCHOR_MS was measured on.
ATTN_ANCHOR_CARD_CLASS = "RTX3080"
#: The worker's own #706 line (cache_controller._generate_storage_config):
#: a Form A expert worker holds no attention layer, builds no page window
#: and rides the null storage tier -- for W7/W10 it stands in for BOTH the
#: KV-page and the GDN-blob marker of that rank (fnFL2 v18, 21.09.: D READY
#: with 1 host + 2 workers was refused "kv x1 blob x1, need 3 each").
FORM_A_WORKER_CANONICAL_MARKER = "this rank is a Form A expert worker (no attention layer) -- no page window"
#: #239 S3h/F14 (rc12z29d, 28.09. 20:34:28Z): under the KV token cut a Form A
#: worker that owns token rows builds a whole-page KV window
#: (cache_controller, F14 branch) and says so with THIS line instead of the
#: no-window one. It is that rank's canonical KV page; it holds no GDN blob
#: (the recurrent state stays the attention host's), so like the no-window
#: line it stands in for the blob marker of that rank. W7/W10 counted only the
#: old line and refused the first cut boot that reached READY:
#: "D logged kv x1 blob x1, need 3 each" (TP0 #706 lines + 2 F14 workers).
FORM_A_KV_WORKER_CANONICAL_MARKER = "#239 F14 KV-WORKER-WINDOW: Form A worker owns token rows"
#: LEISTUNGSSCHALTER (user rule 29.09.): the marker of the line naming the
#: registry row's per-group switch defaults written into --env-p/--env-d.
GROUP_SWITCH_DEFAULTS_MARKER = "PDFLIP-LEISTUNGSSCHALTER"
#: NF port 1003: the arming variables of the per-rank run-time measurement of the extend rate
EXTEND_RATE_MEASURE_ENV = "FLLIPER_PDFLIP_EXTEND_RATE_MEASURE"
EXTEND_RATE_MEASURE_RANKS_ENV = "FLLIPER_PDFLIP_EXTEND_RATE_MEASURE_RANKS"
EXTEND_RATE_MARKER = "EXTEND-RATE"
#: The source boot of a census produced by pdflip/xchg_census.py, as its
#: provenance names it ("ring-table boot boot_weg2_<tag>_<tree>_<date>_<time>").
_CENSUS_SOURCE_BOOT_RE = re.compile(r"ring-table boot (boot_weg2_[A-Za-z0-9_]+)")
#: #145: das Praefix, an dem der Gate-Lauf und der Verdrahtungs-Check die
#: D-Seiten-Zeile finden. EIN Marker, beide Zeilen, damit `grep -c` eine
#: Zahl liefert und nicht zwei Muster braucht.
D_RANK_SOLVE_MARKER = "D-RANK VRAM (#145)"
#: Q-710: the arming variable of the chunk-cap floor (the vote never crawls)
EXTEND_CAP_FLOOR_ENV = "FLLIPER_PDFLIP_EXTEND_CAP_FLOOR"
EXTEND_CAP_FLOOR_MARKER = "EXTEND-CAP-FLOOR"

def d_residue_record_accept(line_id, d_bs: int) -> Callable[[dict], bool]:
    """#1444 + RC1 (24.09.): which measured-record samples may price THIS
    boot's group-D dormant residue. The line's own samples (line identity),
    and for group D only those of the SAME capture set: D's residue is
    mostly the CUDA graphs that stay resident through its sleep, and their
    set follows --max-running-requests. Measured: xsn439 at 32 left
    3206/2742/2740 MiB, xsn438 and RC1 at 6 left 2096/1512/1512 and
    2072/1490/1490. xsn439's was the newest sample when RC1 (6) launched,
    priced dormant_other 3526/3062/3060, took 1112-1232 MiB off every P rank
    and pushed the 42,11,11 pool from 269,805 to 212,871 tokens, under the
    262,656 floor -- the cut fell to 39,13,12 (ranked #57). A sample whose
    capture set cannot be proven is not this boot's residue either."""
    want = int(d_bs)

    def _accept(sample: dict) -> bool:
        if not line_id.accepts_sample(sample):
            return False
        if str((sample or {}).get("group", "")).upper() != "D":
            return True
        return d_residue_capture_bs(sample, line_id) == want

    return _accept

def d_residue_record_for(path: str, accept: Optional[Callable[[dict], bool]], cards: Sequence[Card]
                         ) -> Tuple[Optional[Dict[str, object]], str]:
    """#227: the group-D dormant-image sample of the sidecar at ``path`` that prices this boot's D residue
    (:func:`pick_d_residue_record` over every D sample through ``accept``), and the provenance note -- empty
    when the pick is the plain newest sample, so the pre-#227 DC-RESIDUE line stays byte-identical."""
    rows = host_ledger.read_measured_records(path, "D", accept=accept)
    rec, how = pick_d_residue_record(rows, cards)
    if rec is None or rec is max(rows, key=lambda e: str(e.get("at", ""))):
        return rec, ""
    return rec, f" [#227 card-set pick, {how}]"

def dc_residue_for(path: str, accept: Optional[Callable[[dict], bool]], cards: Sequence[Card],
                   weight_source: str) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, int]], str, str]:
    """#227: ``(sample for the RoPE context term, per-uuid residue or None, provenance, pick note)`` of the
    sidecar at ``path`` for this boot's group-D reserve. Order: (1) the measured maximum of exact-card-set
    boots (:func:`dc_residue_max_from_records`); (2) otherwise the pre-#227 form -- newest sample of exactly
    this card set, else of a COVERING card set (another D topology: named, keeps the #1444 margin), through
    :func:`dc_residue_from_record`."""
    rows = host_ledger.read_measured_records(path, "D", accept=accept)
    if os.environ.get(DC_RECORD_ENV, "1").strip() != "0":
        vals, prov = dc_residue_max_from_records(rows, cards, weight_source)
        if vals is not None:
            want = {c.uuid for c in cards}
            exact = [r for r in rows if isinstance(r.get("vram_residue_mib"), dict) and set(r["vram_residue_mib"]) == want]
            return max(exact, key=lambda e: str(e.get("at", ""))), vals, prov, ""
    rec, how = pick_d_residue_record(rows, cards)
    vals, prov = dc_residue_from_record(rec, cards, weight_source)
    note = ("" if rec is None or rec is max(rows, key=lambda e: str(e.get("at", "")))
            else f" [#227 card-set pick, {how}]")
    if vals is not None and how.startswith("covering"):
        prov = (f"COVERING card set (a sample of another D topology, pre-#227 form with the "
                f"{DC_RECORD_MARGIN_MIB} MiB margin; this card set has no sample of its own yet): " + prov)
    return rec, vals, prov, note

def apply_spec_form(ns) -> None:
    """Install the CLI's speculative form once, before any argv is built."""
    _SPEC_FORM["form"] = str(getattr(ns, "spec_form", SPEC_FORM_DEFAULT) or SPEC_FORM_DEFAULT).upper()
    _SPEC_FORM["draft_path"] = str(getattr(ns, "dflash_draft_path", DFLASH_DRAFT_PATH_DEFAULT) or DFLASH_DRAFT_PATH_DEFAULT)
    _SPEC_FORM["block"] = int(getattr(ns, "dflash_block", DFLASH_BLOCK_DEFAULT) or DFLASH_BLOCK_DEFAULT)
    _SPEC_FORM["window"] = int(getattr(ns, "dflash_window", DFLASH_WINDOW_DEFAULT) or DFLASH_WINDOW_DEFAULT)
    _SPEC_FORM["produce_on_p"] = (
        str(getattr(ns, "dflash_produce_on_p", DFLASH_PRODUCE_ON_P_DEFAULT) or DFLASH_PRODUCE_ON_P_DEFAULT).lower()
        == "on"
    )
    # NF: the CLI value of --draft-kv-on-p is not the answer (H25 overrules
    # it); main installs the resolved value beside resolve_draft_on_p.
    _SPEC_FORM["draft_kv_on_p"] = None
    _SPEC_FORM["d_replayssm_spec"] = (
        str(getattr(ns, "d_replayssm_spec", D_REPLAYSSM_SPEC_DEFAULT) or D_REPLAYSSM_SPEC_DEFAULT).lower()
        == "on"
    )
    # NF line (H64): group D's MTP depth as --extra-d ships it (read by
    # d_verify_window; the last value wins, as in argparse). None when --extra-d
    # names none -- the constants' depth then stands.
    _SPEC_FORM["d_extra_draft_tokens"] = _argv_scalar(
        str(getattr(ns, "extra_d", "") or ""), "--speculative-num-draft-tokens"
    )
    if _SPEC_FORM["form"] == "DFLASH" and not os.path.isdir(_SPEC_FORM["draft_path"]):
        raise SystemExit(
            f"--spec-form DFLASH: draft checkpoint {_SPEC_FORM['draft_path']} is not a directory"
        )

def apply_p_gc_warn_default(ns) -> Optional[str]:
    """30.09. (NF y3z ep52: PP0 stood still 5.4 s process-wide at the first
    tag of P's sleep; a gen-2 GC is one candidate): group P's schedulers arm
    the GC warning at 0.5 s by default -- ``FLLIPER_PDFLIP_GC_WARN_SECS`` into
    ``ns.env_p`` unless --env-p states it (=0 turns it off). Returns the line
    naming it, or None."""
    from flliper.srt.pdflip import gc_instrument as _gci

    env_p = str(getattr(ns, "env_p", "") or "")
    if _gci.WARN_ENV in parse_group_env(env_p):
        return None
    ns.env_p = set_group_env(env_p, _gci.WARN_ENV, f"{_gci.P_GC_WARN_DEFAULT_S:g}")
    return (f"PDFLIP-GC P warn: --env-p {_gci.WARN_ENV}={_gci.P_GC_WARN_DEFAULT_S:g} (every P "
            f"scheduler logs a GC collection over {_gci.P_GC_WARN_DEFAULT_S:g} s after boot; "
            f"--env-p {_gci.WARN_ENV}=0 = off)")

def apply_d_token_placement(ns) -> None:
    from flliper.srt.pdflip import d_reshard as _dr
    from flliper.srt.pdflip import d_token_placement as _dtp

    policy = str(getattr(ns, "d_token_placement", D_TOKEN_PLACEMENT_DEFAULT) or D_TOKEN_PLACEMENT_DEFAULT)
    _D_TOKEN_PLACEMENT["spec"] = None
    if policy not in _dtp.POLICIES:
        raise SystemExit(f"--d-token-placement {policy!r}: one of {list(_dtp.POLICIES)}")
    if policy == _dtp.POLICY_CAPACITY:
        return
    model = str(getattr(ns, "model", "") or "")
    try:
        with open(model_config_path(model)) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--d-token-placement {policy}: cannot read the model config of {model!r}: {exc}")
    try:
        _dr.check_profile(_dr.profile_of_config(cfg), "live")
    except _dr.ReshardProfileRefused as exc:
        raise SystemExit(str(exc).replace("--d-reshard live", f"--d-token-placement {policy}"))
    raw = str(getattr(ns, "d_token_shares", "auto") or "auto")
    try:
        shares = _dtp.RC9_EFF_BW_GBS if raw == "auto" else tuple(float(x) for x in raw.split(","))
        spec = _dtp.PlacementSpec(policy, tuple(shares),
                                  float(getattr(ns, "d_token_fill_switch", _dtp.DEFAULT_FILL_SWITCH)))
        spec.validate()
    except (ValueError, _dtp.PlacementError) as exc:
        raise SystemExit(f"--d-token-placement {policy}: {exc}")
    _D_TOKEN_PLACEMENT["spec"] = spec

def dflash_produce_line() -> str:
    """The one launcher line naming group P's DFlash producer form (the
    draft_kv_off_line pattern: an A/B arm that reads like the default is how a
    measurement gets attributed to the wrong tree)."""
    if dflash_produce_on_p():
        return (
            "PDFLIP DFLASH-PRODUCE-ON-P: on (p_draft=compute) -- group P's DFlash draft-KV producer "
            "COMPUTES (aux capture on every PP stage, produce() per prefill chunk, "
            "hash-keyed publish into the draft arena); the old producer form, A/B only "
            f"({DFLASH_PRODUCE_ENV}=1 in P's environment)"
        )
    return (
        "PDFLIP DFLASH-PRODUCE-ON-P: off (p_draft=cold) -- STANDARD FORM (27B user decision "
        "2026-09-24): group P loads the DFlash2 draft on its last PP stage (cold-resident bytes, "
        "same VRAM, same planner cut, same exchange census and flip) and computes "
        "nothing with it: no aux capture on any PP stage, no produce(), no "
        "publish_draft_rows_direct. Group D is unchanged (DFLASH sharded, window "
        f"{int(_SPEC_FORM['window'])}) and builds its draft context cold "
        "(#993 zeros + 1 bootstrap round); under the default HICACHE-DRAFT-TIER auto "
        "the draft gets no HiCache space at all, so expect no DRAFT-PRESENCE line and "
        f"draft_pages=0 on the front's served lines ({DFLASH_PRODUCE_ENV}=0 in P's "
        "environment; argv_p and the P FORM key are identical under on/off)"
    )

def d_effective_seats(ns) -> int:
    """H91b: the seats group D actually runs -- ``--max-running-requests`` as
    --extra-d ships it (the LAST value wins, as in argparse), else D's own
    ``--d-bs``. ONE reader for every D post that scales with the seats (the
    H64 verify form, the H91b seat rebook), so the price and the argv cannot
    describe two different seat counts."""
    extra = str(getattr(ns, "extra_d", "") or "")
    d_bs = int(getattr(ns, "d_bs", DEFAULT_D_BS) or DEFAULT_D_BS)
    raw_mrr = _argv_scalar(extra, "--max-running-requests")
    try:
        mrr = int(str(raw_mrr)) if raw_mrr is not None else d_bs
    except ValueError:
        mrr = d_bs
    return max(1, mrr)

def d_seat_vram_plan_form(ns, er, plan_kwargs, cfg, form):
    """H95c: the seat-post geometry for the seat table's runtime columns
    (``expert_residency.seat_vram_form``), or None when the switch is off in
    --env-d or the checkpoint has no GDN geometry."""
    env_d = plan_kwargs.get("env_d") or {}
    if not d_seat_vram_armed(env_d):
        return None
    from flliper.srt.planner import pp_cut as _pp

    terms = _pp.checkpoint_weight_terms(ns.model)
    text_cfg = cfg.get("text_config") or cfg
    ratios = str(plan_kwargs.get("rank_tp_ratio") or "")
    return er.seat_vram_form(
        text_cfg, ssm_dtype=getattr(form, "ssm_dtype", None), rank_tp_ratio=ratios,
        n_ranks=len(plan_kwargs.get("budgets_mib") or ()) or 1,
        expert_row_bytes=float(terms.expert_layer_weight_bytes) / int(terms.num_experts),
        moe_layers=int(terms.n_layers))

def apply_d_seat_expert_rows(ns, er, rows, seat_vram, label) -> List[str]:
    """H95c: write FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS (the seat rows' virtual
    reservation per rank, from the table) into --env-d -- before build_env
    reads it -- unless the operator stated it. Returns the lines to log."""
    env_d_raw = str(getattr(ns, "env_d", "") or "")
    stated = parse_group_env(env_d_raw).get("FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS")
    value = er.seat_expert_rows_value(rows)
    fixed = er.seat_fixed_mib(rows, seat_vram, value)
    head = "%s FRACTION-SOLVE %s D-SITZ-VRAM (H95c)" % (D_RANK_SOLVE_MARKER, label)
    fixed_txt = (
        "; fest an JEDER Sitzzahl (auch der Kappe): die Seat-Zeilen der Skalen-Tensoren, "
        "zu klein zum Entmappen, je Rang %s MiB GESCHAETZT -- NICHT im Kappen-Riegel "
        "gebucht" % ["%.1f" % f for f in fixed] if fixed else "")
    if stated is not None:
        return ["%s: --env-d nennt FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS=%s selbst (Tabelle: %s)%s"
                % (head, stated, value, fixed_txt)]
    if not value or all(v == "0" for v in value.split(",")):
        return ["%s: keine Seat-Zeilen (die Tabelle finanziert bei n=1 keine Zeile)" % head]
    item = "FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS=%s" % value
    ns.env_d = (env_d_raw.rstrip(";") + ";" + item) if env_d_raw.strip() else item
    return ["%s: --env-d %s (virtuelle Reserve je Rang = k(n=1) + 1; gemappt wird je "
            "Phase nur, was die Laufzeit exakt aus den ungemappten GDN-Slots finanziert)%s"
            % (head, item, fixed_txt)]

def d_kv_stage_tokens_named(ns) -> Optional[Tuple[int, ...]]:
    """#239 S3h: the KV stage tokens the operator names in --env-d
    (``FLLIPER_PDFLIP_D_KV_STAGE_TOKENS``), or None when the D group's env does
    not name them (the launcher would then write its own stage form)."""
    raw = parse_group_env(str(getattr(ns, "env_d", "") or "")).get(D_KV_STAGE_KEYS[0])
    if raw is None:
        return None
    try:
        return tuple(int(float(x)) for x in str(raw).split(",") if x.strip())
    except ValueError:
        return ()

def kv_stage_wave_floor(group, rows, fits, max_by: Sequence[int], waves_cap: int,
                        head: str) -> List[str]:
    """#239 S3g floor (rc12z30b 28.09. 20:09:07): ``rank_wave_floor`` over the
    form the launcher writes. D_r from the seat table at the cap (ids of the
    cap's seats, E - R of the rank); the scratch after the move (a KV rank
    gives its stage rows to its seat rows); capture_on[r] = a KV rank's
    capture floor at the cap's batch (rows + seat extra - the stage rows of
    the highest stage the cap may take, as ``d_seat_vram.capture_floors``),
    0 on every rank without a stage table (no KV, no cells, no floor). A
    rank given stage rows the plan stages no KV for is refused as well: its
    runtime would build no cells and count them OFF. Refuses by name (W171)
    before any rank is launched; the lines name each rank's air."""
    last = rows[-1]
    n = len(last.scratch_given)
    staged = {int(t.host_rank): t for t in group.tables}
    fit_by = {int(f.rank): f for f in fits}
    cap = len(rows)
    demand, after, on = [], [], []
    for r in range(n):
        c = int(last.scratch_given[r])
        f = fit_by.get(r)
        E = int(f.local_experts) if f is not None else 0
        demand.append(min(int(last.ids_per_step), max(E - (int(last.max_rows[r]) - c), 0)))
        t = staged.get(r)
        if t is None:
            after.append(c)
            on.append(0)
            continue
        # the scratch the rank RUNS (FLLIPER_MOE_SCRATCH_SLOTS as written:
        # the low rows a floor ladder funds are NEW bank rows, only the rest
        # leaves the scratch) = its 'C=LRU+staging'; on = its capture floor,
        # the rows of the cap's top stage -- the two numbers the rank's H95
        # line prints (y6n TP2: 46 + 0, where the old split read 38 + 9)
        a = c - (int(t.rows) - int(getattr(t, "low_rows", 0) or 0))
        after.append(a)
        j = min(int(max_by[cap - 1]), len(t.tokens) - 1)
        on.append(int(t.capacity[cap - 1][j]) - a)
    lines, bad = rank_wave_floor(demand, after, on, int(waves_cap))
    # the smaller batches of a KV rank: its floor can sit lower (a stage the
    # cap never takes) -- the table's per-batch waves, the same floor rule
    bad += [r for r, t in sorted(staged.items())
            if r not in bad and max(t.capture_waves(max_by)) > int(waves_cap)]
    host = int(group.host.host_rank)
    orphan = [r for r in staged if r != host and er_trim_cell(fit_by.get(r), host) <= 0]
    if bad or orphan:
        raise PdFlipDKvStageWavesRefused(
            "W171 PdFlipDKvStageWavesRefused: Wellenboden je D-Rang gebrochen %s%s -- %s; "
            "keine Stufenform geschrieben"
            % (["rang%d" % r for r in bad],
               " / Stufenzeilen ohne KV auf %s" % (["rang%d" % r for r in orphan],)
               if orphan else "", "; ".join(lines)))
    return ["%s: WELLENBODEN JE RANG (#239 S3g-Boden, D <= W x (LRU+Staging+Capture-Boden)) %s"
            % (head, "; ".join(lines))]

def attn_anchor_stage(cards: Sequence["Card"], log: Optional["Log"] = None) -> int:
    """The first P stage whose card is of :data:`ATTN_ANCHOR_CARD_CLASS`.
    No such card: refused BY NAME (HW-UNCALIBRATED) -- the anchor is a
    measurement of that class, and pricing another card's attention layer
    with it would be a borrow.

    AP1 1006 (Q-710, flags instead of code): that refusal is the VALUE refusal
    HW-UNCALIBRATED (``refusals.refuse_value``); without ``--force`` it reads and
    raises as before. With ``--force`` the anchor is BORROWED onto the stage the
    reference rig measured it on (the position of the anchor class in
    ``card_identity.REFERENCE_INVENTORY``, clamped to the stage count): the
    positional ``MEASURED_MS_PER_LAYER`` entry of that stage is the one the anchor
    is combined with, so the pair stays the reference's pair. UNMEASURED on the
    live card; the FORCED-PAST line says so."""
    for i, c in enumerate(cards):
        if card_identity.calibration_class(c) == ATTN_ANCHOR_CARD_CLASS:
            return i
    ref_stage = min(max(0, len(cards) - 1), card_identity.REFERENCE_INVENTORY.index(ATTN_ANCHOR_CARD_CLASS))
    _refuse_value_once(
        card_identity.CODE_UNCALIBRATED,
        f"{card_identity.CODE_UNCALIBRATED}: the P-cut deep attention anchor (--pp-cut-attn-anchor-ms, "
        f"{ATTN_ANCHOR_MS:g} ms/layer) was measured on class {ATTN_ANCHOR_CARD_CLASS}; no P stage "
        "carries that class (" + ", ".join(card_identity.class_label(c) for c in cards)
        + "). Measure the anchor on this rig (a 262k-prefix chunk on the slowest stage) -- not borrowed."
        f"  --force BORROWS it onto P stage {ref_stage} ({card_identity.class_label(cards[ref_stage])}), "
        f"the stage the reference rig measured it on: UNMEASURED on this card (HW-UNCALIBRATED).",
        log=log)
    return ref_stage

def d_kv_dtype_bytes(ns) -> int:
    """#239: bytes per KV element on D -- ``--kv-cache-dtype`` of extra_d,
    else the profile row's ``kv_dtype`` (fp8 = 1, anything else 2)."""
    dtype = _argv_scalar(getattr(ns, "extra_d", ""), "--kv-cache-dtype") or ""
    if not dtype:
        row = pdflip_form.profile_row(getattr(ns, "profile", "") or "")
        dtype = str(getattr(row, "kv_dtype", "") or "") if row is not None else ""
    return 1 if dtype.lower().startswith("fp8") else 2

def d_form_solved_by_planner(ns) -> bool:
    """#239 rc12z29c: the D solve sets D's ownership and/or FR_D itself (an
    owned cut, S2b joint, S2 maxmin: ``--d-kv-token-cut`` names a mode, not a
    vector). Then that form must be solved BEFORE the Platztausch map is
    built, because the map's D phase (window per rank, resident ids) and its P
    phase (the common prefix, the store slots) are both functions of it."""
    return isinstance(d_kv_token_cut(ns), str)

def calibrated_inventory(ns) -> Tuple[Tuple[str, ...], str]:
    """(the inventory the profile's positional VECTORS (argv/env) were
    measured on, its source): ``--profile-inventory``, else the records'."""
    explicit = card_identity.parse_inventory(getattr(ns, "profile_inventory", ""))
    if explicit is not None:
        return explicit, "--profile-inventory"
    prof = getattr(ns, "profile", None) or pdflip_form.DEFAULT_PROFILE
    return records_inventory(str(prof))

def d_session_tokens(extra_d: str, max_kv_per_request: int) -> int:
    """D's maximum decode session: the ``--max-kv-per-request`` group D runs
    with (its ``--extra-d`` value wins, argparse keeps the last)."""
    raw = _argv_scalar(extra_d, "--max-kv-per-request")
    try:
        return int(raw) if raw is not None else int(max_kv_per_request)
    except (TypeError, ValueError):
        return int(max_kv_per_request)

def group_context_tokens(ns, group: str) -> Tuple[int, str]:
    """YaRN x2 (27.09.): the context group ``group`` ('p'/'d') is launched
    with, and where it comes from. argparse keeps the LAST ``--context-length``
    and the group's ``--extra-*`` follows the common flags
    (``CONTEXT_LENGTH_TOKENS``), so an extra value is the group's; without one
    a YaRN override in the extra stretches it; else the rig's common value."""
    extra = getattr(ns, f"extra_{group}", "") or ""
    raw = _argv_scalar(extra, "--context-length")
    if raw is not None:
        try:
            return int(raw), f"--extra-{group} --context-length"
        except ValueError:
            pass
    yarn = yarn_context_tokens(_argv_scalar(extra, "--json-model-override-args"))
    if yarn:
        return yarn, f"--extra-{group} YaRN override (original x factor)"
    return CONTEXT_LENGTH_TOKENS, "CONTEXT_LENGTH_TOKENS"

def gate_w10(log_p: str, log_d: str, log, *, p_produces_draft_pages: bool) -> Dict[str, object]:
    """W10 (#1233) as a function so its skip is pinnable (27B c60a5d387f).

    Under --dflash-produce-on-p off (p_draft=cold) group P registers its
    drafter (the producer is built) but writes no page under it, so D's draft
    reads miss whatever the identities say -- a mismatch there refuses a boot
    for a route that does not exist (metal: weg2xsn414). The identities are
    still read and logged, never graded. Otherwise the W10 gate as before."""
    w10 = check_drafter_identity(log_p, log_d)
    ids = (f"P={w10['P']} D={w10['D']} layout_P={w10['layout_P']} layout_D={w10['layout_D']} "
           f"match={w10['match']} (P lines {w10['n_P']}, D lines {w10['n_D']})")
    if not p_produces_draft_pages:
        log(f"{W10_SKIPPED_NO_DRAFT_PAGES} -- D's draft reads miss by construction and take the cold path "
            f"(#993 zeros + 1 bootstrap round); identities read, NOT graded: {ids}")
        return w10
    log(f"W10 DRAFTER-IDENTITY {ids}")
    if not w10["match"]:
        raise PdFlipLaunchRefused(f"W10 PdFlipDrafterIdentityMismatch: P={w10['P']} D={w10['D']} layout_P={w10['layout_P']} "
                                f"layout_D={w10['layout_D']} -- the decode group would ask the carrier for draft pages under "
                                f"an identity the prefill group never writes")
    return w10

def canonical_marker_counts(path: str) -> tuple:
    """(kv, blob, workers): the #706 KV-page and GDN-blob markers of a group
    log, each worker line (no-window or F14 KV window) counted once into both."""
    # IPC-LOG-EXCEPTION: comparison number only -- canonical_state_gate decides on the
    # RankState records and merely REPORTS a disagreement ('IPC MISMATCH').
    n_kv = count_marker(path, "#706 canonical KV page active")
    n_blob = count_marker(path, "canonical GDN blob active")
    n_worker = (count_marker(path, FORM_A_WORKER_CANONICAL_MARKER)
                + count_marker(path, FORM_A_KV_WORKER_CANONICAL_MARKER))
    return n_kv + n_worker, n_blob + n_worker, n_worker

def d_kv_worker_ranks(extra_d: str) -> List[int]:
    """#239 F14: the Form A workers (``--rank-tp-ratio`` 0) that own token rows
    under the cut D runs (``--uneven-token-vector``) -- exactly the ranks that
    must build a canonical KV window. Empty without a cut or outside Form A."""
    toks = _argv_vector(extra_d or "", "--uneven-token-vector")
    tp = _argv_vector(extra_d or "", "--rank-tp-ratio")
    if not toks or not tp or len(toks) != len(tp):
        return []
    try:
        return [r for r, (t, w) in enumerate(zip(toks, tp))
                if float(w) == 0.0 and float(t) > 0.0]
    except ValueError:
        return []

def arena_gib_into_groups(ns, group_envs, gib: float) -> List[str]:
    """29.09.: write a ledger-derived arena size into every group env that
    names FLLIPER_HICACHE_ARENA_GIB (``ns.env_p``/``ns.env_d`` and the parsed
    dicts the ledger prices from), so ranks and ledger read the same number.
    Groups that do not name it inherit the launcher env. Returns the groups
    rewritten."""
    key, val, done = "FLLIPER_HICACHE_ARENA_GIB", f"{float(gib):g}", []
    for attr in ("env_d", "env_p"):
        spec = str(getattr(ns, attr, "") or "")
        if key in parse_group_env(spec):
            setattr(ns, attr, set_group_env(spec, key, val))
            done.append(attr)
    for e in group_envs or ():
        if e is not None and key in e:
            e[key] = val
    return done

def apply_profile_torch_cache_cap_default(ns) -> Optional[str]:
    """PDFLIP-ALLOC-OVERHANG: the registry row's ``torch_cache_cap`` into
    ``ns.env_d`` (FLLIPER_PDFLIP_TORCH_CACHE_CAP=1) unless --env-d states the
    switch itself (=0 turns it off). Written INTO ``ns.env_d`` so every
    reader (the D solve's cap vector, the budget's capped rest, P1c, the
    rank) sees ONE value. Returns the line naming it, or None."""
    row = pdflip_form.profile_row(getattr(ns, "profile", None))
    if row is None or not bool(getattr(row, "torch_cache_cap", False)):
        return None
    env_d = str(getattr(ns, "env_d", "") or "")
    if TORCH_CACHE_CAP_ENV in parse_group_env(env_d):
        return None
    ns.env_d = set_group_env(env_d, TORCH_CACHE_CAP_ENV, "1")
    return (f"{_budget_rest.OVERHANG_MARKER} registry {row.id}: --env-d {TORCH_CACHE_CAP_ENV}=1 "
            f"(P0 torch cache cap on the physical line, D books the capped rest; --env-d "
            f"{TORCH_CACHE_CAP_ENV}=0 = the uncapped form)")

def apply_profile_group_switch_defaults(ns, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """LEISTUNGSSCHALTER (user rule 29.09. ~10:15Z: proven on metal -> default
    on in the code): the registry row's ``group_switch_defaults`` (pdflip/form.py)
    into ``ns.env_p`` / ``ns.env_d``. A key that group's env already states,
    or that the launcher's own environment carries (an operator export, which
    build_env hands every group), is left alone -- an explicit value always
    wins, =0 turns the switch off. Written INTO the group env so every reader
    (the D solve, the ranks; build_env applies --env-* last) sees ONE value.
    Returns the line naming what was written, or None (qwen27b: GC_FREEZE only)."""
    row = pdflip_form.profile_row(getattr(ns, "profile", None))
    rows = dict(getattr(row, "group_switch_defaults", None) or {}) if row is not None else {}
    if not rows:
        return None
    env_os = canonical_env(dict(os.environ if environ is None else environ))
    wrote: List[str] = []
    for group, attr in (("P", "env_p"), ("D", "env_d")):
        spec = str(getattr(ns, attr, "") or "")
        stated = parse_group_env(spec)
        for key, val in rows.get(group, {}).items():
            if key in stated or key in env_os:
                continue
            spec = set_group_env(spec, key, str(val))
            wrote.append(f"{group}:{key}={val}")
        setattr(ns, attr, spec)
    if not wrote:
        return None
    return (f"{GROUP_SWITCH_DEFAULTS_MARKER} registry {row.id} (default on after the metal proof; "
            f"a stated --env-p/--env-d or exported value wins): " + " ".join(wrote))

def d_extend_rate_measure_ranks_env(ns, log, label: str, fill: dict, rtext: str) -> None:
    """NF port 1003: arm the run-time measurement on the filled ranks ONLY (a recorded rank keeps
    its recorded rate) and name every rank's start source: ``EXTEND-RATE source=record+derived``.
    A value named in --env-d wins; a rank that stays without any rate is named as NOT armed."""
    from flliper.srt.pdflip import extend_trim as _et

    _env_d = getattr(ns, "env_d", "") or ""
    _have = parse_group_env(_env_d)
    _ranks = ",".join(str(r) for r in fill["ranks"])
    _given = (str(_have.get(EXTEND_RATE_MEASURE_ENV, "1")).strip() != "1"
              or str(_have.get(EXTEND_RATE_MEASURE_RANKS_ENV, _ranks)).strip() != _ranks)
    if _ranks and not _given:
        _env_d = set_group_env(_env_d, EXTEND_RATE_MEASURE_ENV, "1")
        ns.env_d = set_group_env(_env_d, EXTEND_RATE_MEASURE_RANKS_ENV, _ranks)
    _vals = [x.strip() for x in str(rtext).split(",")]
    _per_rank = "; ".join(f"rank{i}={src} {_vals[i] if i < len(_vals) else '-'}"
                          for i, src in enumerate(fill["sources"]))
    _unarmed = [i for i, src in enumerate(fill["sources"]) if src == "none"]
    log(f"{D_RANK_SOLVE_MARKER} {label} {EXTEND_RATE_MARKER} source=record+derived start={rtext} "
        f"measure={'on' if _ranks and not _given else 'off'} ranks={_ranks or '-'}"
        f"{' (aus --env-d, Vorrang)' if _given else ''} safety={_et.RATE_SAFETY} ({_per_rank}; "
        f"Start = max(record-floor, ratio, geometry): record-floor = hoechste erfasste Rate des Profils "
        f"(kein Rang startet unter dem gemessenen Wert), ratio = Rate des erfassten Rangs x "
        f"D_EXTEND_GROWTH_MIB[Rang] / D_EXTEND_GROWTH_MIB[erfasster Rang], "
        f"geometry = {fill['geometry']} aus config.json (Q-694b-Formel, auf NF unter dem gemessenen Wert)"
        f"{'' if not fill['geo_err'] else ' (' + fill['geo_err'] + ')'}; je Startrang gilt "
        f"max(Start, gemessen x {_et.RATE_SAFETY}), Ratsche nur nach oben)"
        + (f" -- Rang {','.join(str(i) for i in _unarmed)}: weder Wachstums-Record noch Geometrie, "
           f"der Extend-Chunk-Deckel ist dort NICHT scharf" if _unarmed else ""))

def _group_rank_count(argv: Sequence[str]) -> int:
    """Scheduler ranks one group spawns: --tp-size x --pp-size of its argv."""
    toks = list(argv)
    return int(_last_flag_value(toks, "--tp-size") or 1) * int(_last_flag_value(toks, "--pp-size") or 1)

def _d_layout_is_tp_symmetric() -> bool:
    """UNIFY S7: is the published profile's D TP-symmetric (d_layout
    paged_dcp, the 27B)? No form / unknown profile: False (the NF rule)."""
    row = pdflip_form.profile_row(_published_profile())
    return bool(row is not None and row.d_layout == "paged_dcp")

def _axis_renormalisation(
    family: str,
    axis: str,
    units: int,
    weights: Sequence[int],
    realised: Sequence[int],
) -> str:
    """Floor-aware re-normalisation, as a printable statement (#1293).

    When a grid binds, ``partition_units`` floors the starved ranks to one unit
    and takes the shortfall back off the largest shares, so the realised
    fractions are NOT the requested ratio. Printing the requested ratio there
    would be the instrument-text-lies shape -- the number reported would not be
    the number shipped. This prints BOTH plus the drift between them, with the
    family and the axis named, so a floored grid is READABLE instead of either
    silently wrong or automatically fatal.
    """
    floored = _axis_floor_ranks(units, weights)
    if not floored or not realised:
        return ""
    w_total = float(sum(int(w) for w in weights)) or 1.0
    r_total = float(sum(int(u) for u in realised)) or 1.0
    want = [float(int(w)) / w_total for w in weights]
    got = [float(int(u)) / r_total for u in realised]
    return (
        "%s/%s grid BINDS: rank(s) %s below one of %d units, pinned at the "
        "floor; realised shares %s vs requested %s (max drift %.3f)"
        % (
            family,
            axis,
            list(floored),
            int(units),
            [round(g, 3) for g in got],
            [round(w, 3) for w in want],
            max(abs(g - w) for g, w in zip(got, want)),
        )
    )

def d_moe_residency(env_d: str, tp_size: int):
    """Group D's expert residency, as the capacity model needs it (#62).

    ``(resident_fraction, scratch_slots)`` per rank, or ``(None, None)`` when
    the arm pins neither -- then the model keeps pricing every expert as
    card-resident, which is what a non-offloading boot really does.

    READ FROM THE ARM'S OWN ``--env-d`` STRING, not from this process's
    environment: the launcher never has D's env set (the keys are handed to
    the group at spawn), and three fnFL2 boots were spent on exactly that
    confusion -- a term "read from the env" that no env in the reading process
    ever carried.  One entry is broadcast to every rank, ``tp_size`` entries
    are taken per rank, and any other count is IGNORED rather than guessed at:
    a mis-sized vector must not silently become a capacity claim.
    """
    def _vec(raw, cast):
        if not raw:
            return None
        try:
            parsed = [cast(x) for x in str(raw).split(",") if x.strip()]
        except ValueError:
            return None
        if len(parsed) == 1:
            return parsed * tp_size
        return parsed if len(parsed) == tp_size else None

    env = parse_group_env(env_d or "")
    return (
        _vec(env.get("FLLIPER_MOE_RESIDENT_EXPERT_FRACTION"), float),
        _vec(env.get("FLLIPER_MOE_SCRATCH_SLOTS"), int),
    )

def checkpoint_quant_method(model_path: str) -> str:
    """``quantization_config.quant_method`` of the checkpoint's config.json (top
    level or ``text_config``), lower case; for an export that states its
    quantization only in ``hf_quant_config.json`` (older ModelOpt exports)
    ``"modelopt"``; ``""`` for none or unreadable.

    A GGUF file is ``"gguf"``: the server's own override for it
    (arg_groups/overrides._gguf_quantization), and its sibling config.json is an
    upstream config that says nothing about the file's quantization (27B line
    G2)."""
    if model_is_gguf_file(str(model_path)):
        return "gguf"
    try:
        with open(model_config_path(str(model_path))) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return ""
    for holder in (cfg, cfg.get("text_config") or {}):
        qc = holder.get("quantization_config") if isinstance(holder, dict) else None
        if isinstance(qc, dict) and qc.get("quant_method"):
            return str(qc["quant_method"]).strip().lower()
    try:
        with open(os.path.join(str(model_path), "hf_quant_config.json")) as f:
            hf = json.load(f)
    except (OSError, ValueError):
        return ""
    section = hf.get("quantization") if isinstance(hf, dict) else None
    if isinstance(section, dict) and section.get("quant_algo"):
        return "modelopt"
    return ""

def calib_power_limits(form) -> Tuple[Tuple[str, Tuple[float, ...]], ...]:
    """UNIFY S9 (release table row 27, Agent PL): the limits the cards run now,
    for the ``power_limit`` term of the calibration identity -- one NVML read
    per launcher, and only when the profile's records row keys the term. ()
    = unknown (NVML silent): the term is then not applied (never a refusal)."""
    row = pdflip_form.profile_row(getattr(form, "profile", ""))
    if row is None or "power_limit" not in row.records.fields:
        return ()
    return _running_power_by_class()

def census_checkpoint(census_path: str,
                      evidence_dirs: Sequence[str]) -> Tuple[str, str]:
    """``(checkpoint path, how it was read)`` of the checkpoint a census was
    MEASURED on; ``("", why)`` when it cannot be named. The census's own
    ``checkpoint`` field first (xchg_census.py writes it since weg2xsn441), else
    the ``server_args`` model_path of its source boot's P log."""
    try:
        with open(census_path) as fh:
            blob = json.load(fh)
    except (OSError, ValueError) as exc:
        return "", "the census is unreadable (%s)" % exc
    ck = blob.get("checkpoint") if isinstance(blob, dict) else None
    if isinstance(ck, str) and ck:
        return ck, "the census's own checkpoint field"
    m = _CENSUS_SOURCE_BOOT_RE.search(str(blob.get("provenance", "")))
    if not m:
        return "", "the census names no source boot in its provenance"
    stem = m.group(1)
    for d in evidence_dirs:
        if not d:
            continue
        path = os.path.join(d, stem + ".P.log")
        if os.path.isfile(path):
            model = p_log_model_path(path)
            if model:
                return model, "%s.P.log server_args" % stem
    return "", ("its source boot %s has no readable P log with a server_args line "
                "under %s" % (stem, ", ".join(d for d in evidence_dirs if d) or "-"))

def host_price_is_region_form(ns) -> bool:
    """#1464: does this arm's exchange run its legs through the boot's
    ``xchg.bin`` region (hook + semaphores) rather than the per-lane file
    bounce?  True for every exchange+host arm: ``--pdflip-xchg-inject`` has no
    "off" choice, and both leg call sites pass ``hook=``/``sems=``.  Kept as
    its own predicate so a future arm that really runs ``run_bounce_leg``
    (no hook) can say so in ONE place; ``FLLIPER_PDFLIP_PCUT_LANE_PRICE=1``
    restores the lane price for the cut pick (measurement/diagnosis only).
    """
    if os.environ.get("FLLIPER_PDFLIP_PCUT_LANE_PRICE", "") == "1":
        return False
    return xchg_bounce_arm_pins_host(
        str(getattr(ns, "pdflip_weight_source", WEIGHT_SOURCE_DEFAULT)),
        str(getattr(ns, "pdflip_xchg_oncard", ONCARD_MODE_DEFAULT)))

def host_priced_pick(rows, price, incumbent_gib: float, slack_gib: float):
    """#1447, pure: ``rows`` are frontier candidates FASTEST FIRST, already at
    or above the pool floor; ``price(row) -> (gib, lanes, source)``.  Returns
    ``(first fundable row or None, log lines)`` -- fundable = its bounce is
    within ``incumbent_gib + slack_gib``.  Every row is priced and printed,
    the ones the boot did not take included."""
    picked = None
    lines = []
    for row in rows:
        gib, lanes, source = price(row)
        ok = float(gib) <= float(incumbent_gib) + float(slack_gib) + 1e-9
        lines.append(
            "PP-CUT HOST-PRICE cut=%s attn=%s makespan_ms=%.1f pool=%d xchg_bounce=%.2f GiB "
            "lanes=%d (%s) vs incumbent %.2f + slack %.2f -> %s"
            % (_csv(row.layers), _csv(row.attn), float(row.makespan_ms), int(row.pool_tokens),
               float(gib), int(lanes), source, float(incumbent_gib), float(slack_gib),
               "FUNDABLE" if ok else "over the host budget"))
        if ok and picked is None:
            picked = row
    return picked, lines

def _d_extend_rate_measure_env(ns, log, label: str, source: str, rtext: str, why: str) -> None:
    """Q-694b: arm the per-rank run-time measurement in the D group env (a value named in
    --env-d wins) and name the start rate's source: ``EXTEND-RATE source=record|derived``."""
    from flliper.srt.pdflip import extend_trim as _et

    _env_d = getattr(ns, "env_d", "") or ""
    # a value other than our own came from the user (a second solve pass sees its own write)
    _given = str(parse_group_env(_env_d).get(EXTEND_RATE_MEASURE_ENV, "1")).strip() != "1"
    if not _given:
        ns.env_d = set_group_env(_env_d, EXTEND_RATE_MEASURE_ENV, "1")
    _on = str(parse_group_env(getattr(ns, "env_d", "") or "").get(EXTEND_RATE_MEASURE_ENV, "")
              ).strip().lower() in ("1", "true", "yes", "on")
    log(f"{D_RANK_SOLVE_MARKER} {label} {EXTEND_RATE_MARKER} source={source} start={rtext} "
        f"measure={'on' if _on else 'off'}{' (aus --env-d, Vorrang)' if _given else ''} "
        f"safety={_et.RATE_SAFETY} ({why}; je Rang gilt max(start, gemessen x {_et.RATE_SAFETY}), "
        f"gemessen = max(Allokator-Peak, reserved-Zuwachs) / Zeilen ueber jeden Ziel-Extend "
        f">= {_et.GROWTH_PER_ROW_MIN_ROWS} Zeilen, Ratsche nur nach oben)")

def d_flip_trim_lines(ns, cards: Optional[Sequence[Card]], n: int) -> Tuple[List[float], List[str]]:
    """#201 TRIM-LAW (Unify-RISK (c)): the corridor line per D rank the flip arm's trim threshold is priced on, with
    its source -- the line of the rank's OWN card (:func:`d_card_vote_lines`, the one launcher reader of the card law,
    the same line PDFLIP-VOTE-LINE-BOOK books and the rank's vote keeps): ``cards[r]`` is D rank ``r``'s card by NVML
    UUID. Under VOTE-LINE-SINGLE the fixed part is the one in ``ns.env_d`` (what the ranks read). Where no line of
    this card is usable -- no D cards at this call, a rank count that does not match, an unmeasured / hand-set floor,
    a footprint of another form -- the rank gets the STATED LAW ``corridor_law_mib()`` (the pre-#201 value), and the
    source says why (``Rueckfall Korridor-Gesetz``). The rank lifts a threshold under its own line itself
    (``extend_trim.lift_threshold_to_law``), so a launcher read that names another line than the rank's can only
    trim earlier, never later."""
    law = float(corridor_guard.corridor_law_mib())
    if cards is None or len(list(cards)) != int(n):
        why = ("keine D-Karten an dieser Stelle" if cards is None
               else f"{len(list(cards))} D-Karten fuer {int(n)} Raten")
        return [law] * int(n), [f"Rueckfall Korridor-Gesetz {law:.0f} ({why})"] * int(n)
    cards = list(cards)
    fixed = None
    raw = str(parse_group_env(getattr(ns, "env_d", "") or "").get(_budget_rest.VOTE_FIXED_ENV, "") or "").strip()
    if raw:
        try:
            fv = [float(x) for x in raw.split(",") if x.strip()]
            fixed = (fv * int(n)) if len(fv) == 1 else (fv if len(fv) == int(n) else None)
        except ValueError:
            fixed = None
    try:
        floors, lines, usable, form_why = d_card_vote_lines(cards, "D", fixed)
    except Exception as exc:  # noqa: BLE001 -- an unreadable card law: the stated law, named
        why = f"Kartengesetz unlesbar ({type(exc).__name__}: {exc})"
        return [law] * int(n), [f"Rueckfall Korridor-Gesetz {law:.0f} ({why})"] * int(n)
    out: List[float] = []
    srcs: List[str] = []
    for r, c in enumerate(cards):
        cf = floors[c.uuid]
        if usable[r]:
            out.append(float(lines[r]))
            single = fixed is not None  # usable implies measured, and a measured floor takes the fixed part
            srcs.append(f"Kartengesetz {float(lines[r]):.0f} ({cf.source}:{cf.mib}"
                        + (", VOTE-LINE-SINGLE near-OOM + fixed" if single else "") + ")")
        else:
            why = form_why[r] if form_why[r] is not None else f"Boden {cf.source}:{cf.mib} nicht gemessen"
            out.append(law)
            srcs.append(f"Rueckfall Korridor-Gesetz {law:.0f} ({why})")
    return out, srcs

def _d_extend_cap_floor_env(ns, log, label: str) -> None:
    """Q-710 (INT8 y8vb 03.10. 19:34-19:51Z): arm the vote floor wherever the flip arm armed
    the Q-694 chunk cap -- and only there. On a card with < 300 MiB free in the D phase the
    rc12g vote was ONE row, the group cut to 1 token per forward (3905 extends of ~152 ms in 17
    min, 133 s without a decode step). With the flag the vote keeps a floor chunk derived from
    the configured chunk width and what the free card funds at the priced rate, and a fresh
    D-direct request longer than a starved group width is refused at the X gate so the front
    routes it through P (``EXTEND-CAP-FLOOR route=P``). A value named in --env-d wins. The P0
    arm, Next Flash (ledger rate) and the dual layout never reach this."""
    _env_d = getattr(ns, "env_d", "") or ""
    _given = EXTEND_CAP_FLOOR_ENV in parse_group_env(_env_d) and \
        str(parse_group_env(_env_d).get(EXTEND_CAP_FLOOR_ENV, "1")).strip() != "1"
    if not _given:
        ns.env_d = set_group_env(_env_d, EXTEND_CAP_FLOOR_ENV, "1")
    _on = str(parse_group_env(getattr(ns, "env_d", "") or "").get(EXTEND_CAP_FLOOR_ENV, "")
              ).strip().lower() in ("1", "true", "yes", "on")
    log(f"{D_RANK_SOLVE_MARKER} {label} {EXTEND_CAP_FLOOR_MARKER} floor={'on' if _on else 'off'}"
        f"{' (aus --env-d, Vorrang)' if _given else ''}: der Chunk-Deckel faellt nie unter "
        f"chunked_prefill_size/16 (begrenzt auf floor(card_free_post / Rate)); ein frischer "
        f"D-Direktprefill laenger als eine Gruppenbreite unter dem Boden geht ueber P")

def expert_map_vectors(ns) -> Tuple[List[str], List[str], List[float]]:
    """(--rank-moe-ratio, FR_D, FR_P) exactly as :func:`publish_expert_map`
    reads them -- one reader, so the host ledger's store post
    (:func:`planned_expert_store_bytes`) sizes the store the map will build."""
    ratios = _argv_vector(getattr(ns, "extra_d", ""), "--rank-moe-ratio")
    fr_tp = _argv_vector(getattr(ns, "extra_d", ""), "--rank-moe-resident-fraction")
    _fp_text = str(getattr(ns, "pp_cut_expert_device_fraction", "") or "")
    fr_pp = (_csv_floats(_fp_text) if _fp_text else
             [float(x) for x in (_argv_vector(
                 getattr(ns, "extra_p", ""), "--rank-moe-resident-fraction") or [])])
    return ratios, fr_tp, fr_pp

def expert_store_dir_of(ns) -> str:
    """The expert store directory the RANKS get: --env-d, then --env-p, then
    the launcher's own environment (ENV-LAUNCHER != RANG -- the NF arm sets it
    in the group env only)."""
    from flliper.srt.layers.moe.expert_store import STORE_DIR_ENV

    for spec in (getattr(ns, "env_d", ""), getattr(ns, "env_p", "")):
        v = parse_group_env(spec or "").get(STORE_DIR_ENV)
        if v:
            return str(v)
    return str(os.environ.get(STORE_DIR_ENV, "") or "")

def d_draft_share_embed(ns) -> bool:
    """Group D's FLLIPER_PDFLIP_DRAFT_SHARE_EMBED (H1b): --env-d wins over the
    launcher's own environment, which the ranks inherit."""
    raw = parse_group_env(getattr(ns, "env_d", "") or "").get("FLLIPER_PDFLIP_DRAFT_SHARE_EMBED")
    if raw is None:
        return bool(envs.FLLIPER_PDFLIP_DRAFT_SHARE_EMBED.get())
    return str(raw).strip().lower() in ("1", "true", "yes", "y")

def apply_p_draft_post(ns, cards, fracs, stage_layers, row_mib: float,
                       num_experts: int, log) -> List[float]:
    """H25: raise the last P stage's expert fraction by the draft's post and
    publish it to ``--pp-cut-expert-device-fraction``, ``--extra-p
    --rank-moe-resident-fraction`` and ``--env-p
    FLLIPER_MOE_RESIDENT_EXPERT_FRACTION`` -- the three places FR_P lives
    (arm_fnFL2.sh: "FR_P steht an DREI Stellen"); a reader left on the old
    value would be the argv-doppelung class. Returns the fractions the pool
    model prices."""
    from flliper.srt.pdflip import draft_post as _dp

    path = (_argv_scalar(getattr(ns, "extra_p", ""), "--speculative-draft-model-path")
            or _argv_scalar(getattr(ns, "extra_d", ""), "--speculative-draft-model-path")
            or "")
    new, post, why = _dp.raise_for_draft_post(
        fracs=fracs, stage_layers=stage_layers, row_mib=row_mib,
        num_experts=num_experts, draft_path=str(path), cards=cards)
    if post is None:
        log(f"PP-CUT draft post (H25) ENTFAELLT: {why}")
        return list(fracs)
    log(post.line())
    ns._h25_draft_post = _dp.summary(post)
    ns._h25_fr_p_before = [float(x) for x in fracs]
    ns.pp_cut_expert_device_fraction = ",".join(_dp._fmt(x) for x in new)
    ns.extra_p = _dp.replace_vector_flag(ns.extra_p, "--rank-moe-resident-fraction", new)
    ns.env_p = _dp.replace_env_vector(getattr(ns, "env_p", ""),
                                      "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION", new)
    log(f"PP-CUT draft post (H25) FR_P {[round(float(x), 4) for x in fracs]} -> "
        f"{ns.pp_cut_expert_device_fraction} (published to --pp-cut-expert-device-"
        f"fraction, --extra-p --rank-moe-resident-fraction, --env-p "
        f"FLLIPER_MOE_RESIDENT_EXPERT_FRACTION)")
    return new

def d_park_split_refusal(ns, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """27B park (review RV B5): the front fires the immediate park from the
    launcher's env (FLLIPER_PDFLIP_D_PARK_IMMEDIATE / the profile row), D parks
    only with its FLIP park on (d_seats.d_flip_park_active over D's env = the
    launcher env + --env-d). An explicit FLLIPER_PDFLIP_D_PARK=0 decides alone on
    D -- profile_docker.render writes exactly that line for qwen27b -- so the
    front would park and D answer 409 every phase (PARK-FAILED, the pre-park
    fallback, K0). Refused at launch instead, by name. None = consistent."""
    env = dict(os.environ if environ is None else environ)
    prof = getattr(ns, "profile", None)
    front_on, front_src = pdflip_form.d_park_immediate_state(env, prof)
    if not front_on:
        return None
    d_env = dict(env)
    d_env.update(parse_group_env(str(getattr(ns, "env_d", "") or "")))
    raw = str(d_env.get("FLLIPER_PDFLIP_D_PARK", "") or "").strip()
    if raw:
        d_on, d_src = raw.lower() not in ("0", "false", "no", "off"), "FLLIPER_PDFLIP_D_PARK=" + raw
    else:
        d_imm, d_imm_src = pdflip_form.d_park_immediate_state(d_env, prof)
        d_sf, d_sf_src = pdflip_form.standard_form_state(d_env, prof)
        d_on, d_src = d_imm or d_sf, "immediate %s, standard form %s" % (d_imm_src, d_sf_src)
    if d_on:
        return None
    return ("PDFLIP D-PARK-SPLIT refused: the front's immediate park is ON (%s) but group D's flip "
            "park is OFF (%s) -- the front would park every D phase and D answer 409 "
            "(PARK-FAILED). Unset FLLIPER_PDFLIP_D_PARK (or set it 1) in the profile/--env-d, or turn "
            "FLLIPER_PDFLIP_D_PARK_IMMEDIATE off" % (front_src, d_src))

def l15_mib_by_card_identity(ns, cards: List[Card], log,
                             environ: Optional[Dict[str, str]] = None) -> None:
    """HW-GENERIC 1002 (L15): FLLIPER_PDFLIP_L15_MIB keyed by CARD IDENTITY.

    Resolves the value's card keys (``GPU-<uuid>=``, ``pci:<bus id>=``,
    ``nvml<i>=``, or the legacy budget ordinal ``c<i>=``) ONCE against the
    ordered ``cards`` (``l15_plan.resolve_l15_mib``) and hands the ranks the
    canonical ordinal form: rewritten in ``environ`` (default ``os.environ``,
    which ``build_env`` copies for P and D and ``resolve_posts`` reads) and in
    ``--env-p`` / ``--env-d`` (applied last by ``build_env``) when they carry
    the key. An ordinal-only value is left byte-identical. One
    ``L15-MIB-CARDS`` line per source names key -> ordinal -> card; a bad key
    is refused BY NAME (W-L15-CARDKEY). With the master on, a cap-0 rank
    count other than one prints ``L15-CAP0`` (a warning: D rank i sits on
    ordinal i, ``--rank-gpu-id 0,1,2``)."""
    env = os.environ if environ is None else environ
    key = l15_plan.L15_MIB_ENV
    sources = [("env", env.get(key))]
    groups = {a: parse_group_env(getattr(ns, a, "") or "") for a in ("env_p", "env_d")}
    sources += [(a, g[key]) for a, g in groups.items() if key in g]
    for src, raw in sources:
        if raw is None:
            continue
        pci = {}
        if "pci:" in str(raw).lower():
            # the launcher Card carries no bus id; the registry's NVML reader does
            pci = {d.uuid: d.pci_bus_id for d in nvml_registry.list_devices()}
        try:
            res = l15_plan.resolve_l15_mib(raw, cards, pci)
        except ValueError as exc:
            raise PdFlipLaunchRefused(f"{exc} (source {src})") from exc
        if res.mode == "auto":
            continue
        log(l15_plan.mib_cards_line(res, cards, src))
        if res.rewritten and src == "env":
            env[key] = res.value
        elif res.rewritten:
            setattr(ns, src, set_group_env(getattr(ns, src, ""), key, res.value))
    # the value the D ranks run: --env-d (applied last) over the launcher env
    d_env = dict(env)
    d_env.update({k: v for k, v in parse_group_env(getattr(ns, "env_d", "") or "").items()
                  if k in (key, l15_plan.L15_MASTER_ENV)})
    if l15_plan.master_on(d_env):
        from flliper.srt.pdflip import l15_shadow

        line = l15_shadow.cap0_line(l15_shadow.cap0_ranks(d_env, len(cards)))
        if line is not None:
            log(line)

def d_early_start_proven(profile: Optional[str]) -> bool:
    """``--pdflip-d-early-start auto`` = on for this profile: the registry row
    carries a metal proof (``ModelProfile.d_early_start_proven``) AND prices
    the early plan from P's records. No/unknown profile = the launcher
    default (qwen27b)."""
    row = pdflip_form.profile_row(profile or pdflip_form.DEFAULT_PROFILE)
    return (bool(getattr(row, "d_early_start_proven", False))
            and d_expect_from_p_records(profile))
