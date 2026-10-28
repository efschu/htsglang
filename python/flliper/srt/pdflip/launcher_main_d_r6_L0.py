"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

import os
import sys
import time

from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import refusals
from flliper.srt.pdflip import ring_table
from flliper.srt.pdflip import weight_exchange
from flliper.srt.pdflip import xchg_bounce
from flliper.srt.pdflip.launcher_d_leaf import DC_EXPECT_3080_MIB
from flliper.srt.pdflip.launcher_d_leaf import DC_EXPECT_5090_MIB
from flliper.srt.pdflip.launcher_d_leaf import DC_MAX_MIN_SLEEPS
from flliper.srt.pdflip.launcher_d_leaf import DC_ONE_SLEEP_MARK
from flliper.srt.pdflip.launcher_d_leaf import DC_RECORD_MARGIN_MIB
from flliper.srt.pdflip.launcher_d_leaf import VISION_OFF
from flliper.srt.pdflip.launcher_d_leaf import VISION_TRANSIENT
from flliper.srt.pdflip.launcher_d_leaf import _argv_scalar
from flliper.srt.pdflip.launcher_d_leaf import _argv_vector
from flliper.srt.pdflip.launcher_d_leaf import _arm_run_peak_gib
from flliper.srt.pdflip.launcher_d_leaf import _d_adopt_armed
from flliper.srt.pdflip.launcher_d_leaf import arena_from_ledger
from flliper.srt.pdflip.launcher_d_leaf import bs_source
from flliper.srt.pdflip.launcher_d_leaf import chunked_prefill_size_of
from flliper.srt.pdflip.launcher_d_leaf import d_reserve_from_records_or_census
from flliper.srt.pdflip.launcher_d_leaf import d_reshard_lines
from flliper.srt.pdflip.launcher_d_leaf import d_token_vector_decision
from flliper.srt.pdflip.launcher_d_leaf import dormant_max_from_records
from flliper.srt.pdflip.launcher_d_leaf import early_read_provenance
from flliper.srt.pdflip.launcher_d_mid import CONTEXT_LENGTH_TOKENS
from flliper.srt.pdflip.launcher_d_mid import arena_gib_into_groups
from flliper.srt.pdflip.launcher_d_mid import d_residue_record_accept
from flliper.srt.pdflip.launcher_d_mid import d_session_tokens
from flliper.srt.pdflip.launcher_d_mid import dc_residue_for
from flliper.srt.pdflip.launcher_d_mid import expert_store_dir_of
from flliper.srt.pdflip.launcher_d_p_apply import census_checkpoint_decision
from flliper.srt.pdflip.launcher_d_p_apply import mamba_arena_slots_into_groups
from flliper.srt.pdflip.launcher_d_p_apply import p_attn_head_split_cfg
from flliper.srt.pdflip.launcher_d_p_r2_L0 import EVIDENCE_DIR
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_CENSUS_DRAFT_PAGES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_CENSUS_KV_PAGES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_CENSUS_KV_PAGE_BYTES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_CENSUS_MAMBA_BLOBS
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_CENSUS_PROVENANCE
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_MAMBA_BLOB_BYTES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_ROOT
from flliper.srt.pdflip.launcher_d_p_r2_L0 import arc_preflight_line
from flliper.srt.pdflip.launcher_d_p_r2_L0 import boot_state_write
from flliper.srt.pdflip.launcher_d_p_r2_L0 import cold_tier_shm_post
from flliper.srt.pdflip.launcher_d_p_r2_L0 import dc_measured_d_mib
from flliper.srt.pdflip.launcher_d_p_r2_L0 import l3_persist_disk_sum
from flliper.srt.pdflip.launcher_d_p_r2_L0 import lane_coverage_dump_dir
from flliper.srt.pdflip.launcher_d_p_r2_L0 import measured_record_path
from flliper.srt.pdflip.launcher_d_p_r2_L0 import plan_store
from flliper.srt.pdflip.launcher_d_p_r2_L1 import _VFL_ARGV
from flliper.srt.pdflip.launcher_d_p_r2_L1 import _env_knobs
from flliper.srt.pdflip.launcher_d_p_r2_L1 import common_flags
from flliper.srt.pdflip.launcher_d_p_r2_L1 import l3_persist_attach
from flliper.srt.pdflip.launcher_d_p_r2_L1 import p_chunk_policy_install
from flliper.srt.pdflip.launcher_d_p_r2_L1 import p_dormant_records
from flliper.srt.pdflip.launcher_d_p_r2_L1 import p_prefill_graph_line
from flliper.srt.pdflip.launcher_d_p_r2_L1 import prepare_weight_exchange
from flliper.srt.pdflip.launcher_d_p_r2_L2 import PORT_P
from flliper.srt.pdflip.launcher_d_p_r2_L2 import _ARENA_BOOKED_WIDTHS
from flliper.srt.pdflip.launcher_d_p_r2_L2 import d_replayssm_spec_line
from flliper.srt.pdflip.launcher_d_p_r2_L2 import hicache_draft_tier_line
from flliper.srt.pdflip.launcher_d_p_r2_L2 import start_memts
from flliper.srt.pdflip.launcher_d_p_r2_L4 import build_env
from flliper.srt.pdflip.launcher_d_p_r2_L4 import d_overhead_calibration
from flliper.srt.pdflip.launcher_d_rank_r2_L7 import rank_reserve_kwargs
from flliper.srt.pdflip.launcher_hicache_calib import d_rope_context_delta_mib
from flliper.srt.pdflip.launcher_hicache_calib import model_layer_kinds
from flliper.srt.pdflip.launcher_log_plan import Log
from flliper.srt.pdflip.launcher_p_d_r2_L3 import p_prefill_transient_for
from flliper.srt.pdflip.launcher_p_l3_leaf import WEIGHT_SOURCE_EXCHANGE
from flliper.srt.pdflip.launcher_p_l3_leaf import _fmt_gib
from flliper.srt.pdflip.launcher_p_l3_leaf import _group_env_value
from flliper.srt.pdflip.launcher_p_l3_leaf import expert_map_dir
from flliper.srt.pdflip.launcher_p_l3_leaf import fork_anchor_env
from flliper.srt.pdflip.launcher_p_l3_leaf import hicache_disabled_line
from flliper.srt.pdflip.launcher_p_l3_leaf import l3_persist_check_identity
from flliper.srt.pdflip.launcher_p_l3_leaf import l3_persist_dir_name
from flliper.srt.pdflip.launcher_p_l3_leaf import l3_persist_enabled
from flliper.srt.pdflip.launcher_p_l3_leaf import p_attn_head_split_env
from flliper.srt.pdflip.launcher_p_l3_leaf import p_attn_head_split_line
from flliper.srt.pdflip.launcher_p_l3_leaf import p_chunk_power_lines
from flliper.srt.pdflip.launcher_p_l3_leaf import p_prefill_graph_bucket
from flliper.srt.pdflip.launcher_p_l3_leaf import p_prefill_graph_policy_lines
from flliper.srt.pdflip.launcher_p_l3_leaf import p_trim_end_anchor_env
from flliper.srt.pdflip.launcher_p_l3_leaf import parse_group_env
from flliper.srt.pdflip.launcher_p_mid import l3_persist_check_rope
from flliper.srt.pdflip.launcher_p_mid import l3_persist_identity
from flliper.srt.pdflip.launcher_p_mid import newest_bubble_log
from flliper.srt.pdflip.launcher_p_mid import p_covers_d_session
from flliper.srt.pdflip.launcher_p_mid import p_micro_batch_line
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import prepare_store
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import store_read_cost_line
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import sweep_store_residue
from flliper.srt.pdflip.launcher_refuse_proc_leaf import DC_RESERVE_SLACK_MIB
from flliper.srt.pdflip.launcher_refuse_proc_leaf import PP_LAYER_SET_ENV
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _PROFILE_27B
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _split_fraction_text
from flliper.srt.pdflip.launcher_refuse_proc_leaf import pinned_reserve_line
from flliper.srt.pdflip.launcher_refuse_proc_leaf import publish_store_identity
from flliper.srt.pdflip.launcher_refuse_proc_leaf import read_arc_state
from flliper.srt.pdflip.launcher_refuse_proc_leaf import read_pp_bubble
from flliper.srt.pdflip.launcher_refuse_proc_leaf import refuse_inherited_layer_set
from flliper.srt.pdflip.launcher_refuse_proc_leaf import refuse_unless_same_form_source
from flliper.srt.pdflip.launcher_refuse_proc_leaf import set_xchg_census_map
from flliper.srt.pdflip.launcher_resolve_stop import vision_gguf_tower_refusal
from flliper.srt.pdflip.launcher_resolve_stop import vision_victim_host_term
from flliper.srt.pdflip.launcher_resolve_xchg_r2_L1_1 import prepare_xchg_env
from flliper.srt.pdflip.launcher_resolve_xchg_r2_L1_1 import xchg_form_dormant_reserve
from flliper.srt.pdflip.launcher_types import BootState
from flliper.srt.pdflip.launcher_types import GroupSpec
from flliper.srt.pdflip.launcher_types import LaunchGuard
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_vision_spec_mid import shipped_layer_split
from flliper.srt.pdflip.launcher_vision_spec_mid import transport_argv
from flliper.srt.pdflip.launcher_vision_spec_mid import vision_place_dual_refusal
from flliper.srt.pdflip.launcher_vision_spec_mid import vision_tower_dir
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_SOURCE_DISK
from flliper.srt.pdflip.launcher_xchg_vision_leaf import turn_anchor_env
from flliper.srt.pdflip.launcher_xchg_vision_leaf import vision_source_host_line
from flliper.srt.pdflip.launcher_xchg_vision_leaf import w38_armed_line
from flliper.srt.pdflip.launcher_xchg_vision_leaf import xchg_census_is_reserve
from flliper.srt.pdflip.launcher_xchg_vision_leaf import xchg_form_argv
import json
import math
import shlex
import subprocess

#: #1273 B4h -- THE EXCHANGE FORM'S OWN DORMANT RESIDUE, behind the switch and
#: not in the constant above, which is exactly the precedent the NCCL slack set
#: twelve lines up ("The extra slack lives HERE, behind the switch, and not in
#: the constant").  One level MORE honest than that one, because this form was
#: measured WHOLE rather than as a delta:
#:
#: MEASURED on boot weg2xsn14 (BOOT_weg2xsn14_0911.md, the W19 line at EPOCH 0
#: on tip 692b1e8698, under `--pdflip-weight-source exchange` with
#: `--pdflip-xchg-inject shadow` and `--pdflip-xchg-oncard ipc`): D's dormant
#: residue is 2588 / 3084 / 2588 MiB on nvml0 / nvml1 / nvml2, against the
#: serving reserve's 1986 / 2292 / 1986 -- so every card tripped W19
#: DormantResidueRefused before the first flip.
#:
#: SELECTED IFF `weight_source == "exchange"`, under ANY inject mode: the
#: residency is PRESENT under `shadow`, which is where it was measured.  That
#: makes this the one predicate in this slice family that does NOT conjoin the
#: inject arm -- B4f's conjunction answers a different question (who OWNS the
#: bytes), not what is RESIDENT.
#:
#: KEYED BY CARD IDENTITY, the same way `DC_MEASURED_D_*` is keyed -- by the
#: board's NAME, never by NVML index, which is not stable across boots on this
#: rig (#589).  The two 3080s measured IDENTICALLY at 2588, so the name is
#: exact here rather than a convenient collapse of two different numbers.
#:
#: NOT FINAL, AND NOW ATTRIBUTED (B4i, boot weg2xsn15's per-card term table).
#: B4g reported 773 / 935 / 773 MiB of this measurement as UNATTRIBUTED; THAT
#: RESIDUAL IS RETRACTED -- it was produced by subtracting /dev/shm byte counts
#: (the staging region 385 + on-card slots 96 = 481) from a PER-PROCESS NVML
#: reading that never contained them, which is where 1254-481 and 1416-481 came
#: from.  On ONE instrument (`PDFLIP-DC` at sleep) the arithmetic closes:
#:
#:     XSN15  D @ sleep   3084 / 2590 / 2588   this form (this constant)
#:     sn5b   D @ sleep   1668 / 1334 / 1334   the serving form (the census)
#:     delta              1416 / 1256 / 1254
#:     weights_draft tag  1440 / 1280 / 1280   tms_tag_bytes, RESIDENT on D
#:     unexplained          -24 /  -24 /   -26 instrument margin
#:
#: So the whole excess is group D's resident NEXTN/MTP draft tag, which the
#: exchange never moves because `weights_draft` is out of the weights family,
#: and which no flip pauses (`in_family=no`, absent from pause_order, ms=-1).
#:
#: B4k (spec AMENDMENT 6) puts the draft head IN the family, and THIS TRIPLE IS
#: THEN EXPECTED TO EXPIRE RATHER THAN MOVE.  The prediction, stated here so the
#: next boot GRADES it instead of re-deriving it by hand -- exchange-form
#: reading minus the measured resident draft tag:
#:
#:     5090   3084 - 1440 = 1644   against the serving form's census 1668 (-24)
#:     3080   2588 - 1280 = 1308   against                        1334 (-26)
#:
#: i.e. the exchange form's dormant residue should fall to the SERVING form's
#: own reading, which is already BELOW the serving constants
#: (`DC_MEASURED_D_5090_MIB` 2228 + 64 slack = 2292; 1922 + 64 = 1986).  If the
#: B4k boot confirms that, this triple has no reason left to exist and W19 can
#: grade the exchange arm against the serving constants again -- which is a
#: RETIREMENT, decided by that measurement and not here.
#:
#: UNTIL THEN IT STAYS, deliberately, and it is NOT hand-lowered: it
#: OVER-reserves, which is the safe direction under the host-threshold law, and
#: the cost of being wrong the other way is a W19 refusal at epoch 0 (three
#: boots have paid it).  The price of keeping it is named rather than hidden:
#: group P's budget loses 666 MiB per 3080 and 856 MiB on the 5090 for residency
#: that B4k removes.  A PER-FORM MEASURED RECORD REPLACES THIS CONSTANT
#: WHOLESALE (AMENDMENT 7 item 2 = B4o, keyed on B4e's form token) -- nobody
#: should read this triple as final, and nobody should re-derive a second
#: constant that B4o then deletes.
DC_MEASURED_D_XCHG_MIB = _PROFILE_27B.constant("DC_MEASURED_D_XCHG_MIB")
PP_CROSSING_WIRE_ENV = "FLLIPER_PP_CROSSING_WIRE"

def _main_step03__vis_place_dual(*, argv, ns):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # VISION-WEIGHTS AP4: a place that is guaranteed broken in the dual layout is refused by name, here at the verdict site (parse side; no
    # computing core). After resolve_dual_layout (--dual-share implies --dual-layout), after the teardown branch.
    _vis_place_dual = vision_place_dual_refusal(ns, list(sys.argv[1:] if argv is None else argv))
    if _vis_place_dual:
        print(_vis_place_dual, flush=True)
        raise SystemExit(_vis_place_dual)
    _vis_gguf = vision_gguf_tower_refusal(ns)  # VISION-GGUF: the tower of a GGUF comes from --tokenizer-path
    if _vis_gguf:
        print(_vis_gguf, flush=True)
        raise SystemExit(_vis_gguf)
    # Arm the complement instrument in the LAUNCHER's own process too. The
    # rank processes never import this module, so without this the ingest
    # could only ever print NO-OBSERVATION for `launcher.py` -- an honest
    # answer and a useless one, because the arm/ledger half of this file is
    # exactly the set of decisions an operator wants the complement of ("which
    # arm branch did this boot NOT take"). Group `L` keeps its dump out of
    # every rank's filename space. No-op and no import when the flag is off.
    _lane_cov_dir = lane_coverage_dump_dir(ns)
    return _lane_cov_dir

def _main_step04_refuse_inherited_lay(*, ns):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # BEFORE build_env(), which starts from os.environ: an inherited stage map
    # would reach group P's ranks without passing through the solver at all.
    refuse_inherited_layer_set(os.environ)
    # #1273 B4k: THE ARM IS PUBLISHED INTO THIS PROCESS'S OWN ENVIRONMENT, and
    # it has to be, because the flag and the env var were two readings of one
    # fact.  `weight_exchange.weight_source()` is the ONE reader of the mode and
    # it reads the ENVIRONMENT; the launcher, however, took the mode from ARGV
    # and only ever wrote the env var into the per-group `build_env` -- so in the
    # launcher's own process, and in every process that inherits it, the arm
    # read as `ring`.
    #
    # Harmless while the arm changed nothing the launcher computes; a NO-OP
    # BOOT the moment `weights_draft` joined the family (AMENDMENT 6).  The
    # front's environment is `dict(os.environ)` (:10056) and the front builds
    # the pause order from `weights_family_tags` (front.py:1609): without this
    # line the ranks would carry a 10-tag family while the launcher and the
    # front carried 9, the draft tag would never appear in any pause order, its
    # bytes would stay resident through every flip, and B4k would grade green at
    # the desk while achieving NOTHING at the boot.  It is also the
    # ranks-never-disagree law: a family the launcher and the ranks read
    # differently is a disagreement, not a compensation.
    #
    # Written HERE, next to the other two pre-launch environment decisions, and
    # with the value the launcher publishes to the groups -- so the mode a rank
    # enforces, the mode the front schedules and the mode the launcher prices
    # cannot be three readings.
    os.environ[weight_exchange.WEIGHT_SOURCE_ENV] = str(ns.pdflip_weight_source)
    # H25 (Nutzer-Order 24.09.): "does group P carry the draft" -- resolved
    # ONCE from --draft-kv-on-p and FLLIPER_PDFLIP_DRAFT_ON_P, written back into
    # `ns` (every later `ns.draft_kv_on_p` reader agrees: the KV price, the
    # PP-CUT draft post, argv_p) and published into this process's
    # environment for the same reason as the arm above: the weights family
    # (`draft_tag_in_family`) is read here, by the front and by every rank.
    # UNIFY S6: resolved ONCE by the p_draft axis (pdflip/form.resolve_form ran
    # the same rule with the profile's default and wrote ns.pdflip_draft_on_p);
    # read here, never re-derived. The direct call stays only for a caller
    # without a resolved form.
    _resolved_draft_on_p = getattr(ns, "pdflip_draft_on_p", None)
    return _resolved_draft_on_p

def _main_step05_vision_victim_host_mib(*, ns):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # VISION-WEIGHTS AP4: the host image of the displaced weights (ledger post vision_victim_host); a WARNING only, never a refusal
    vision_victim_host_mib, vision_victim_host_prov = vision_victim_host_term(ns)
    # #1032 RESOLVED HERE, BEFORE THE SWEEPS, THE STORE AND ANY LAUNCH: a
    # retracted token vector is a desk fact, and the whole point of W46 is that
    # it must not cost a boot window -- nor a mount, nor an shm sweep -- to
    # discover. Same reason refuse_inherited_layer_set sits one line above.
    d_tokvec = d_token_vector_decision(
        ns.d_uneven_token_vector,
        ns.d_uneven_token_vector_role,
        ns.d_uneven_token_vector_provenance,
    )

    tree = os.path.abspath(ns.tree)
    tip = subprocess.run(["git", "-C", tree, "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", tree, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    stamp = time.strftime("%m%d_%H%M%S", time.gmtime())
    base = f"{EVIDENCE_DIR}/boot_weg2_{ns.tag}_{tip}_{stamp}"
    # VOTE-FIXED-LIVE (review M1/S3): the FULL boot name (the store's boot identity, also what --import reads off a
    # log name) and D's effective KV dtype for the form
    ns.boot_name = os.path.basename(base)
    _VFL_ARGV["kv"] = _argv_scalar(str(getattr(ns, "extra_d", "") or ""), "--kv-cache-dtype") or ""
    dry = ns.dry_run
    return base, d_tokvec, dirty, dry, stamp, tip, tree, vision_victim_host_mib, vision_victim_host_prov

def _main_step06_os_makedirs_evidence(*, _h87_alias_line, base, boot_form, calib_identity, dirty, dry, ns, stamp, tip, tree):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    front_log = None if dry else f"{base}.front.log"
    log = Log(front_log)
    log(f"=== PDFLIP BOOT tag={ns.tag} tree={tree} @ {tip} ({'DIRTY: ' + dirty[:200] if dirty else 'clean'}) stamp={stamp} dry={dry}")
    set_xchg_census_map(getattr(ns, "pdflip_xchg_census_map", ""))
    if getattr(ns, "force", False):
        # PROFIL-EDITOR S1: one switch for all VALUE refusals (pdflip/refusals.py); the sink writes each one to state.json events
        refusals.arm(True, sink=lambda code, text: boot_state_write(log, event=("forced_past", {"code": code, "text": text})))
        log("FORCED BOOT (--force): value refusals are lifted and listed as FORCED-PAST <CODE>; this boot writes NO records "
            "unless every refusal it passed is a proof gap of its own form (#227, verdict line FORCED-RECORDS before the "
            "front starts; records held until then); not lifted: card occupation, missing/broken files, unsupported architecture")
        _ep_forced = refusals.entrypoint_forced()
        log("FORCED-RECORDS entrypoint list (" + refusals.ENV_FORCED_PAST_EP + "): "
            + ("ABSENT under " + refusals.ENV_FORCE_REQUEST + "=1 -> no records" if _ep_forced is None
               else ("; ".join(f"{c} {' '.join(t.split())[:160]}" for c, t in _ep_forced) or "nothing passed")))
    # THE PIN RESOLVER'S CONTRACT LINE, and it must stay in exactly this shape:
    # `line_gate.BOOT_TREE_RE` is `^tree\s*:\s*\S+\s+@\s+([0-9a-f]{7,40})\b`, so
    # the header above -- which carries the same two facts -- does NOT match it.
    #
    # WHY THIS LINE EXISTS AT ALL (measured 2026-09-21).  `resolve_pin` step 3
    # reads this stamp out of `/root/current_boot.log` to learn which tree the
    # standing boot runs.  No boot log since early September carried it: the
    # emitter lived in the retired `boot_855_*` scripts and was never ported
    # here.  `boot_commit()` therefore returned None on EVERY call, resolution
    # fell through to step 4 -- the newest merge/integ/train NAME -- and the
    # codegraph answered every query against `merge/train-0901` (14.09.) while
    # a three-weeks-newer tree was booting.  Nobody noticed for a fortnight
    # because step 4 returns a PLAUSIBLE pin instead of refusing.
    #
    # The filename is deliberately NOT the source of this fact: line_gate's own
    # header records a launcher whose LOG= template hard-coded a stale SHA into
    # the name while the header read correctly.
    log(f"tree: {tree} @ {tip}")
    if boot_form is not None:
        log(boot_form.line())
        _borrowed = pdflip_form.borrowed_constants_line(boot_form.profile) \
            if boot_form.profile in pdflip_form.PROFILES else None
        if _borrowed:
            log(_borrowed)
        if calib_identity is not None:
            log("PDFLIP-PROFILE calibration identity (" + boot_form.profile + "): a measured "
                "source counts only when its boot ran " + calib_identity.describe())
            _head_gap = calib_identity.head_unresolvable_line()
            if _head_gap:
                log(_head_gap)
        if _h87_alias_line:
            log(_h87_alias_line)
        log("#114 P-PREFILL-TRANSIENT (form " + boot_form.describe() + "): "
            + p_prefill_transient_for(boot_form)[1])
    # 27B line (24.09., 159333c14d): group D's planner overhead, MEASURED on the
    # newest D log that passes this boot's calibration identity (UNIFY S7: the
    # form's CalibrationIdentity replaces the 27B LineIdentity).
    # None = the flat default, named on the line.
    d_overhead_mib, _d_overhead_prov = d_overhead_calibration(str(ns.model), calib_identity)
    log("PDFLIP D-PLANNER-CALIBRATION " + ("overhead " if d_overhead_mib is not None
                                          else "UNCALIBRATED (flat 2304 MiB/rank): ")
        + _d_overhead_prov)
    if dirty and not dry:
        raise PdFlipLaunchRefused("tree is not clean -- boot from a COMMITTED tip only")
    py = f"{ns.venv}/bin/python"
    state = BootState(tag=ns.tag, tip=tip, tree=tree, stamp=stamp)
    return d_overhead_mib, front_log, log, py, state

def _main_step11__dc_accept(*, boot_form, calib_identity, calib_sample_accept, cards, d_bs, log, ns, slack_mib, state):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # B4h: THE ONE CONSUMER, form-switched through the one selector, so P's
    # per-card budget follows the arm automatically and no parallel reserve
    # object exists.
    # #1444: the MEASURED residue of the previous boot on this form wins over
    # the xsn14 constant; the constant is the fallback with a printed reason.
    # PDFLIP-FORM (24.09.): only a record MEASURED ON THIS CHECKPOINT -- xsn417
    # (Qwen3.8-27B) priced the Next-Flash boot fnFL2x142's D residue here.
    # 27B RC1 (f1c9a43964) / UNIFY S7: with the profile's ``d_capture_set``
    # record term, only a sample of THIS capture set (D --max-running-requests
    # d_bs) prices D's dormant residue (d_residue_record_accept).
    _dc_accept = (d_residue_record_accept(calib_identity, d_bs)
                  if calib_identity is not None and calib_identity.uses_d_capture_set
                  else calib_sample_accept)
    # #227: with forced boots writing records again, forms of ONE checkpoint and form on DIFFERENT card sets
    # write into the same identity (NVFP4: duo 5090+3080 writes D samples, dual 5090+3080+3080 reads them). The
    # newest sample alone would hand dual a 2-card duo sample, which dc_residue_from_record refuses card by card
    # (-> the constant, ~1 GiB per card off P). Prefer the newest sample of EXACTLY this boot's card set, then the
    # newest that covers it (the pre-#227 answer), then the newest at all (for the refusal's wording).
    _dc_rec_d, _dc_from_record, _dc_record_prov, _dc_pick_note = dc_residue_for(
        measured_record_path(), _dc_accept, cards, ns.pdflip_weight_source)
    # D-EXPECT: group P's measured dormant residue, same calibration identity,
    # read only where the expectation uses it (27B: not read at all).
    _p_dormant_recs = p_dormant_records(ns.profile, calib_sample_accept)
    if _dc_rec_d is None and calib_sample_accept is not None:
        _dc_record_prov = (
            f"no group-D dormant-image record MEASURED ON {boot_form.model} under "
            f"this form ({' '.join(a + '=' + getattr(boot_form, a) for a in pdflip_form.RESIDUE_AXES)}) "
            f"in the sidecar -- samples of another checkpoint or form are not this "
            f"boot's residue (PDFLIP-FORM; xsn418 died W19 on fnFL2x144's) -> constant")
    elif calib_sample_accept is not None:
        _dc_record_prov += _dc_pick_note
        _dc_record_prov += (
            f" [PDFLIP-FORM calibration identity: {boot_form.model}, "
            f"{' '.join(a + '=' + getattr(boot_form, a) for a in pdflip_form.RESIDUE_AXES)}]")
    if calib_identity is not None and calib_identity.uses_d_capture_set:
        _dc_record_prov += f" [capture set --max-running-requests {int(d_bs)}]"
    # YaRN x2 27B: D's RoPE context stays resident while D sleeps; the records/constants were measured at
    # 262144 -> the same reading's D share on every card (0 at x1: byte-identical).
    _d_rope_from = (int((_dc_rec_d or {}).get("vram_residue_context_tokens") or CONTEXT_LENGTH_TOKENS)
                    if _dc_from_record is not None else CONTEXT_LENGTH_TOKENS)
    _d_rope_mib, _d_rope_line = d_rope_context_delta_mib(ns, _d_rope_from)
    _d_rope_add = int(math.ceil(_d_rope_mib)) if _d_rope_mib != 0 else 0
    dc_expect_d = {
        c.uuid: (
            _dc_from_record[c.uuid] if _dc_from_record is not None
            else dc_measured_d_mib(c, ns.pdflip_weight_source)
        ) + slack_mib + _d_rope_add
        for c in cards
    }
    if _d_rope_line:
        log(_d_rope_line + f" -> dormant reserve += {_d_rope_add} MiB per card")
    log(f"#1444 DC-RESIDUE group=D source="
        f"{'RECORD' if _dc_from_record is not None else 'CONSTANT'}: {_dc_record_prov}; "
        f"reserve incl. {slack_mib} MiB slack = "
        + ", ".join(f"nvml{c.nvml_index} {dc_expect_d[c.uuid]}" for c in cards) + " MiB")
    if _dc_from_record is not None and DC_ONE_SLEEP_MARK in _dc_record_prov:
        log(f"#227 R1 {DC_ONE_SLEEP_MARK}: at least one of the boots behind this D reserve saw fewer than "
            f"{DC_MAX_MIN_SLEEPS} D sleeps (its maximum is not reached yet) -> its first sleep is booked with "
            f"the {DC_RECORD_MARGIN_MIB} MiB margin, not bare (boots listed in the DC-RESIDUE line above)")
    refusals.flush(log)         # AP1 1006: the W19 twin borrow (dc_measured_d_mib has no logger); nothing without --force
    if ns.transport == "nccl":
        log(
            f"TRANSPORT=nccl (development mode, user order 2026-09-07): barlink flags dropped from both groups; "
            f"group D's dormant-residue slack raised {DC_RESERVE_SLACK_MIB} -> {slack_mib} MiB because libnccl's buffers "
            f"are not memory-saver-tagged and survive the sleep (MEASURED 2230 -> 2310 MiB on the 5090, boot weg2ab0 ARM 0, "
            f"where the unchanged 2228+64 reserve tripped W19 DormantResidueRefused). Under --transport bar1 this number "
            f"and the whole flip path are untouched."
        )
    state.dc_expect_d = dc_expect_d
    _xchg_form = ns.pdflip_weight_source == WEIGHT_SOURCE_EXCHANGE
    return _dc_accept, _dc_from_record, _p_dormant_recs, _xchg_form, dc_expect_d

def _main_step12_if__dc_from_record_i(*, _dc_accept, _dc_from_record, _xchg_form, cards, dc_expect_d, log, ns, slack_mib, state):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    if _dc_from_record is None:  # #1444: the constant's provenance only when it was priced
        log(("dormant residue RESERVE for group D = MEASURED D_c(D) of boot "
             + (f"weg2xsn14 ({'/'.join(str(v) for v in DC_MEASURED_D_XCHG_MIB)} MiB, "
                "the EXCHANGE form's own residue at epoch 0; its whole excess over "
                "the serving form is the RESIDENT weights_draft tag -- see "
                "DC_MEASURED_D_XCHG_MIB)"
                if _xchg_form else
                "weg2ls1b2 (2228 / 1922 / 1922 MiB, NVML per-process, windows included)"))
            + f" + {slack_mib} MiB slack; spec 1.6 expectation was "
            f"{DC_EXPECT_5090_MIB}/{DC_EXPECT_3080_MIB} (exceeded); "
            f"form={ns.pdflip_weight_source}; graded by W19 at D's first sleep: "
            + ", ".join(f"nvml{c.nvml_index}={dc_expect_d[c.uuid]}" for c in cards))
    if _xchg_form:
        # B4i: THE LINE, ON EVERY LAUNCH OF THIS FORM, DRY-RUN INCLUDED.
        #
        # Boot weg2xsn15 read `PDFLIP-XCHG-RESERVE` ZERO times in all four logs
        # while W19 was genuine 0 -- the constant was in force and the line that
        # was to report it existed only as a function with no production caller
        # (called from two test files and from nowhere else, so every suite was
        # green over code no boot ever ran).  That is W84's class one level down
        # and it is closed HERE, at the one place the reserve is built, which is
        # also why the call sits beside `dc_expect_d` rather than in a helper:
        # the reserve and its report cannot then drift apart.
        #
        # BEFORE the dry-return, so the desk sees the arithmetic without a boot.
        #
        # NO `and ns.pdflip_xchg_census` HERE, and that is deliberate.  A truthy
        # check on the path would make a census-less exchange launch print
        # nothing instead of refusing, which is the #872 shape (a naive truthy
        # check with no else) and exactly how this line came to be missing in
        # the first place.  The census is MANDATORY under this arm and there is
        # ONE authority for that: `xchg_residency.load_census`, which raises
        # W71 by name with its own full sentence for an absent, non-existent or
        # malformed file.  Reaching it here rather than at
        # `prepare_weight_exchange` (200 lines down) is the same named refusal
        # with the same text, only sooner; the richer W71 of an UNFUNDABLE peak
        # still comes from that gate, which is the only place that solves one.
        # weg2xsn441: the census must be THIS checkpoint's (W161 in the dry run).
        _census_line = census_checkpoint_decision(
            ns.pdflip_xchg_census, str(ns.model),
            (ns.evidence_dir, EVIDENCE_DIR),
            bool(getattr(ns, "pdflip_xchg_census_foreign", False)))
        if _census_line:
            log(_census_line)
        _xr, _ = xchg_form_dormant_reserve(cards, ns.pdflip_xchg_census, log=log, profile=ns.profile)
        # UNIFY redfix: only on the profile whose census IS the reserve
        # (xchg_census_is_reserve: Next Flash, cb1575e94e). The 27B line keeps
        # dc_expect_d as priced above (record, else the weg2xsn14 constant);
        # its census is only reported (the line just printed).
        if xchg_census_is_reserve(ns.profile):
            # Die Census-Messung ersetzt die Konstante auch im BUDGET von P
            # (`budgets_from_dc` zieht dc_expect_d ab) und im W19-Riegel -- EINE
            # Zahl je Karte, nicht Messung im Log und Konstante in der Rechnung.
            # fnFL2x111 (24.09.): die Census-Zeile stammt aus Boot fnFL2x86 (1896 MiB
            # auf der 5090), der #1444-Record des VORIGEN Boots (x110) mass 1950 und
            # preiste 1950+256+64 = 2270. Diese Schleife ueberschrieb den Record mit
            # der aelteren Census (1896+64 = 1960); mit P-Chunk 8192 hielt D-TP0 beim
            # ersten Sleep 2000 MiB (torch_reserved +50 MiB) -> W19 vor dem ersten
            # Flip. Eine aeltere Census unterbietet keine neuere Messung derselben
            # Form: je Karte gewinnt die GROESSERE der beiden gepreisten Reserven,
            # und die Zeile darunter nennt, welche.
            # D-RESERVE 29.09.: the maximum over the newest D records of this
            # identity and form wins over the (older) census when records exist.
            _d_recs = (host_ledger.read_measured_records(
                measured_record_path(), "D", accept=_dc_accept)
                if _dc_accept is not None else [])
            _d_max, _d_max_why = dormant_max_from_records(
                _d_recs, cards, ns.pdflip_weight_source, group="D")
            log(f"D-RESERVE RECORD-MAX: {_d_max_why}")
            dc_expect_d, _won = d_reserve_from_records_or_census(
                cards, dc_expect_d,
                {u: int(v) + slack_mib for u, v in _xr.items()},
                _d_max, slack_mib, _dc_from_record is not None)
            state.dc_expect_d = dc_expect_d
            log("dormant residue RESERVE for group D (census-korrigiert, neuere Messung "
                "unterbietet die aeltere nicht): "
                + ", ".join(f"nvml{c.nvml_index}={dc_expect_d[c.uuid]}"
                            for c in cards) + " -- " + ", ".join(_won))
    return dc_expect_d

def _main_step15_shipped_split(*, attn_stage_ratio, budgets_p, cut, log, ns, p_split, stage_ratio):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # W46 (#1254): THE MAP AND THE ARGV MUST BE THE SAME SPLIT -- kept as a
    # REFUSAL on train 2 even though the two halves now come from one source.
    # It was written when they did not: the map was derived from the incumbent
    # score constants while argv_p shipped whatever solve_p_cut returned, and a
    # map built for one split while another runs is COMPLETE BUT WRONG PER CARD
    # (a weights tag charged to one card while its bytes straddle two) -- the
    # accounting class that killed boot weg2dk4, which interleave_pause_order
    # cannot catch because it refuses only an INCOMPLETE map.
    #
    # It passes BY CONSTRUCTION now, and that is exactly why it stays: the two
    # sides are still derived along INDEPENDENT paths -- p_split off
    # cut.layer_counts (solve_p_cut's round trip through the runtime authority),
    # shipped_split by RE-PARSING the argv strings this launcher is about to
    # hand group P, through the same runtime parser the boot will use.  A guard
    # that can only pass is not the same thing as a guard whose two inputs
    # cannot disagree; deleting it would remove the only place that reads P's
    # argv back and compares it to the map, and the next time those paths part
    # (a pin, a gapped map, a flag that stops being forwarded) nothing would
    # say so.  Checked before either group starts.
    shipped_split = shipped_layer_split(
        stage_ratio, attn_stage_ratio, cut.layer_set,
        model_layer_kinds(ns.model), len(budgets_p),
    )
    if p_split and shipped_split and list(shipped_split) != list(p_split):
        raise PdFlipLaunchRefused(
            "W46 PdFlipPPSplitMapMismatch: group P's argv would run the layer "
            f"split {','.join(str(n) for n in shipped_split)} (argv "
            f"--pp-stage-ratio {stage_ratio or '(none)'} --pp-attn-stage-ratio "
            f"{attn_stage_ratio or '(none)'}"
            + (f" --pp-layer-set {cut.layer_set}" if cut.layer_set else "")
            + f"), while the PDFLIP-FLIP-ORDER MAP was built from "
            f"{','.join(str(n) for n in p_split)} (flip_order_split over "
            "solve_p_cut's own round-tripped layer counts). Those two are the "
            "same cut by construction, so a mismatch here means one of the two "
            "paths stopped describing the shipped cut -- the argv strings are "
            "no longer what the solve returned, or the counts are no longer "
            "what the argv parses to. A map built for one split and an argv "
            "running another is COMPLETE BUT WRONG PER CARD -- the weg2dk4 "
            "accounting class -- and interleave_pause_order refuses only an "
            "INCOMPLETE map, so it would pause the wrong card first with a "
            "confident reason. Refusing rather than booting that."
        )
    log(
        "PDFLIP-PP-SPLIT CHECK: argv split "
        f"{','.join(str(n) for n in shipped_split) if shipped_split else 'NOT STATED'} "
        f"== flip-order map split "
        f"{','.join(str(n) for n in p_split) if p_split else 'MAP REFUSED (see above)'} "
        "-- W46, the two independent derivations of one cut: P's argv re-parsed "
        "through the runtime's own parser against the counts the map was built "
        "from"
    )

def _main_step16__draft_kv_on_p_requested(*, argv, hicache_disabled, ns):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # 1b'. THE FORM KEY (W48).  The argv group P will run, built with SENTINEL
    # host-ledger terms because the ledger is priced FROM the ring and cannot
    # exist yet -- and those three flags are excluded from the key BY NAME
    # (ring_table.FORM_KEY_EXCLUDED_FLAGS), so the sentinel cannot reach the
    # hash.  Re-derived from the REAL argv further down and refused on any
    # difference, so the exclusion is proven on every boot, not trusted.
    # TRAIN 2 MERGE FIX: the four TRAILING arguments are passed EXPLICITLY, and
    # that is not tidiness.  Fix 5 was written against an ``argv_p`` that ended
    # at ``depth``; the #1235 argv slice appended --barlink-bar1-window-mib,
    # --random-seed, --barlink-bar1-cap-cycles and --collective-census-interval,
    # NONE of which is in FORM_KEY_EXCLUDED_FLAGS.  Letting them fall to their
    # module defaults here would hash a form that differs from the shipped one
    # the moment an operator passes any of the four -- and fix 5's own W48
    # CONFIRMED check below would then refuse a boot whose ring was in fact
    # gated correctly, for a difference that has nothing to do with the ledger
    # sentinels.  Only the three ledger terms and the depth are sentinels; every
    # other flag must be the one that ships.
    # #1264: THE SWITCH IS APPLIED HERE, ONCE, AND NEVER READ AGAIN.  From this
    # line on the fact lives in group P's argv and every consumer asks
    # ring_table.p_carries_drafter of it -- so the form key that gates the ring
    # is hashed over the form that ships, which is the whole point of the
    # sentinel round trip below.  The OFF line is printed before the ring is
    # solved, because the ring's PP2 span is one of the things it changes.
    #
    # #1386 FOLLOW-UP (xsn31/7 Versuch 7 wall, server_args.py:8432): P's
    # draft-KV PRODUCER writes into the canonical page store
    # (--hicache-canonical-kv-page), and that store has NO existence when
    # HiCache is disabled -- registry.create_tree_cache builds MambaRadixCache
    # for this hybrid-SSM model whenever `enable_hierarchical_cache` is False,
    # and MambaRadixCache carries no cache_controller/canonical_page_store at
    # all (only HiRadixCache / UnifiedRadixCache.init_hicache do, gated
    # SOLELY on enable_hierarchical_cache, per scheduler.py's own
    # `derive_enable_hicache_storage` docstring: "which tree cache gets built
    # is decided by enable_hierarchical_cache alone... never by the storage
    # backend"). So the producer flag belongs in the SAME falls-together group
    # as the ten --hicache-* flags below (P_DRAFT_KV_FLAGS has to leave AS A
    # WHOLE, not just its --speculative-draft-kv-only member: server_args.py's
    # own `_refuse_proposing_drafter_under_pp` asserts pp_size>1 implies
    # speculative_algorithm is None OR speculative_draft_kv_only, so P cannot
    # keep the other four NEXTN flags without the producer flag either).  MTP
    # itself is untouched -- group D carries its own, independent NEXTN head
    # in BOTH forms (see the comment on P_DRAFT_KV_FLAGS's use in argv_p); only
    # P's CO-PRODUCTION into the now-nonexistent store drops, which is exactly
    # the pre-#1264 / "draft_kv_on_p=off" serving-base form already named and
    # logged below.  ONE local, computed here where its #1264 twin already is,
    # so no downstream reader (the sentinel `common_flags` call, the ledger
    # call, the form-key build, the shipped `argv_p`) can price the producer
    # one way and ship a different argv -- the exact #1385/#1386 shape three
    # boots already cost, now guarded against for this flag too.
    _draft_kv_on_p_requested = ns.draft_kv_on_p == "on"
    if hicache_disabled and _draft_kv_on_p_requested and (
        bs_source("--draft-kv-on-p", argv) == "flag"
    ):
        # Named refusal, not a silent downgrade and not a 9s crash inside the
        # child's server_args.py: an EXPLICIT --draft-kv-on-p on is a stated
        # intent this boot cannot honor while HiCache is disabled, and
        # overriding a stated intent without a word is the swallow class this
        # whole switch exists to prevent. Left at its (on) DEFAULT, no operator
        # stated an intent, so the auto-off fallback below applies silently
        # and correctly -- only an explicit, contradicted request refuses.
        raise PdFlipLaunchRefused(
            "W104 PdFlipHicacheDraftKvProducerConflict: --pdflip-disable-hicache "
            "was combined with an EXPLICIT --draft-kv-on-p on. Group P's "
            "draft-KV producer needs --hicache-canonical-kv-page, and that "
            "page format has no store to write into once HiCache is disabled "
            "(server_args.py:8432, root-caused on boot weg2xsn31 Versuch 7: "
            "MambaRadixCache carries no cache_controller/canonical_page_store "
            "-- only the HiCache tree caches do, gated solely on "
            "enable_hierarchical_cache). Drop --pdflip-disable-hicache, or drop "
            "the explicit --draft-kv-on-p on (the default already falls to "
            "off automatically whenever HiCache is disabled -- MTP/NEXTN "
            "itself stays on via group D's own independent drafter; only P's "
            "co-production into the store drops), or pass --draft-kv-on-p "
            "off yourself to say the same thing without relying on the "
            "auto-fallback."
        )
    draft_kv_on_p = _draft_kv_on_p_requested and not hicache_disabled
    return _draft_kv_on_p_requested, draft_kv_on_p

def _main_step17_log_f_pdflip_host_d(*, d_draft_host_mib, d_draft_host_prov, log, vision_victim_host_mib, vision_victim_host_prov):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    log(f"PDFLIP-HOST d_draft_host={d_draft_host_mib:.0f} MiB -- {d_draft_host_prov}")
    if vision_victim_host_prov and not vision_victim_host_prov.startswith("none (--pdflip-vision-place"):
        # only with --pdflip-vision-place weights: the default boots print nothing new (golden dry runs byte-equal)
        log(f"PDFLIP-HOST vision_victim_host={vision_victim_host_mib:.0f} MiB -- {vision_victim_host_prov}")
        if vision_victim_host_mib > 0:
            log("PDFLIP VISION-VICTIM host-RAM WARNING (never a refusal, user rule 2026-10-06): --pdflip-vision-place weights holds "
                f"{vision_victim_host_mib:.0f} MiB of displaced weights in host RAM while one image is encoded (transient, 0 afterwards)")
    if p_prefill_graph_bucket():
        # Only when on: the default boot's front log stays byte-identical.
        log(p_prefill_graph_line())
        for _pgl in p_prefill_graph_policy_lines():
            log("PDFLIP " + _pgl)

def _main_step18__pwl(*, hicache_disabled, log):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # Row 27: the chunk model's power-limit check ([] under 'fixed').
    for _pwl in p_chunk_power_lines():
        log(_pwl)
    # --d-reshard: () under 'off' (front log byte-identical).
    for _drl in d_reshard_lines():
        log(_drl)
    if not hicache_disabled:
        log(hicache_draft_tier_line())
    # --d-replayssm-spec (27B ReplaySSM S6): D's verify form, both states named.
    log(d_replayssm_spec_line())
    # #1386: `hicache_disabled` was already resolved once, at the top of
    # `main`, before `log` even existed (the first `common_flags` sentinel
    # call needs it long before this point) -- this is only the LOG SIDE of
    # the same fact, placed here beside its `draft_kv_on_p` twin now that a
    # logger is available.
    if hicache_disabled:
        log(hicache_disabled_line())

def _main_step19_form_key(*, form_argv_p, log):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    form_key, form_norm = ring_table.p_form_key(form_argv_p)
    log(
        f"PDFLIP-P-FORM key={form_key} -- the identity of what group P LOADS, "
        f"hashed over its own argv with these flags excluded by name: "
        f"{', '.join(ring_table.FORM_KEY_EXCLUDED_FLAGS)} "
        f"(policy={ring_table.FORM_KEY_POLICY}: everything else is IN, because a "
        f"whitelist stops discriminating the day a new form-changing flag is "
        f"added -- nobody would have thought to whitelist "
        f"--speculative-draft-kv-only, and that is the flag family boot weg2tr2 "
        f"died of). A ring table solved from a boot of another form is never "
        f"MEASURED for this one. FORM: {form_norm}"
    )
    return form_key, form_norm

def _main_step20_bootstrap_w48(*, log, ns, ring_plan, state):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # B4e (ii): with the token on, an absent same-form source is no longer an
    # EXCLUSION but a silent fall-through to the closest twin -- refuse here,
    # before either group starts, with the twin printed.
    # #1367: the bootstrap is a LINE, not a silence -- it goes into the ring
    # evidence, so a reader of this boot's state can tell "first boot of this
    # form" from "inherited a same-form table" without re-deriving either.
    bootstrap_w48 = refuse_unless_same_form_source(ring_plan.table, ns.pdflip_weight_source)
    if bootstrap_w48:
        log(bootstrap_w48)
        ring_plan.lines.append(bootstrap_w48)
    state.ring_lines = ring_plan.lines
    state.ring_epoch = str(ring_plan.epoch)

def _main_step21_state_ring_dir_rin(*, cards, log, ns, ring_plan, state):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    state.ring_dir = ring_plan.dir if (ring_plan.armed and ring_plan.form == "MAP_SHARED") else ""

    # 1c. #1273 S7: W71, the exchange's VRAM residency, per card per direction
    # per wave -- HERE, beside the ring's own inequalities, because it grades
    # the same kind of claim (a peak nobody can observe once the flip is
    # running) and must refuse in the same place (before either group starts).
    # On the default `ring` arm this returns None and logs nothing at all.
    #
    # ``ring_H_mib`` is reported from THIS boot's ring plan rather than assumed
    # to be 0: S6 is what stops publishing TMS_HOST_RING_* under `exchange`,
    # and until it lands a non-zero value on the arming line is the truth about
    # this boot, not a formatting default.  An unarmed gate must never read as a
    # passed one, and neither may an un-disarmed ring read as a disarmed one.
    #
    # TODO(S7->S6), round-2 refuter F12: ``epoch`` is the RING's epoch because
    # today the ring is what publishes one.  When S6 stops publishing
    # TMS_HOST_RING_* under `exchange`, this token goes empty unless the
    # exchange mints its own epoch -- the arming line must not lose its boot
    # identity in the same commit that makes the arm real.
    xchg_res = prepare_weight_exchange(
        cards, log, ns.pdflip_weight_source, ns.pdflip_xchg_census,
        ring_plan.epoch,
        (ring_plan.table.total_h_bytes // ring_table.MIB) if ring_plan.table else 0,
    )
    state.xchg_lines = list(xchg_res.lines) if xchg_res is not None else []
    # #1273 S5: the SHADOW arm's region, armed here and published to both
    # groups by build_env below.  Empty on every other arm.
    # #1273 S6 step 5: PRICE THE BOUNCE TERM HERE so the ranks can be given it.
    #
    # Through the SAME glue `xchg_bounce_terms_for_arm` uses
    # (`checkpoint_census.widest_layer_terms`, which is itself the one caller of
    # the one sizing function `xchg_bounce.bounce_terms`) and behind the SAME
    # arm predicate -- so this is ONE AUTHORITY CALLED TWICE, never a second
    # formula.  The census is a safetensors HEADER read and `bounce_terms` is
    # pure, so `choose_host_ledger`'s copy below and this one are identical by
    # construction.
    #
    # Deliberately NOT by widening that function's return: its source is graded
    # by another seat's guard this cycle
    # (`test_the_under_coverage_GUARD_is_present_and_currently_unreachable`
    # greps it for `covers_widest_layer`), and `choose_host_ledger`'s signature
    # is pinned keyword-for-keyword by `test_pdflip_store_priced_x_1317`.  A
    # publication need is not a reason to edit either.
    bounce_terms_for_ranks = None
    # #1385 (Wand 11b): resolved ONCE, here, before either the ranks'
    # publication or the ledger's own call (below) prices anything -- the same
    # value reaches both through `xchg_bounce.resolve_lanes_concurrent`, so a
    # boot that refuses on one side and passes on the other cannot happen.
    _lanes_concurrent = xchg_bounce.resolve_lanes_concurrent(
        getattr(ns, "xchg_lanes_concurrent", None))
    # #1397 (Option 3): resolved ONCE, here, beside `_lanes_concurrent` and
    # for the identical reason -- the ranks' publication call and the
    # ledger's own call below must read the IDENTICAL bool, or a cross
    # lane's smaller priced buffer and a rank's larger allocated one could
    # disagree. A plain bool, not a validated int: unlike
    # `--xchg-lanes-concurrent` there is no "argv value that cannot mean
    # anything" for a `store_true` flag, so there is nothing for a
    # `resolve_*` function to refuse here -- `bounce_terms` itself is what
    # refuses the one real danger (band_credit armed without max_tag_bytes).
    _band_credit = bool(getattr(ns, "xchg_band_credit", False))
    return _band_credit, _lanes_concurrent, bounce_terms_for_ranks, xchg_res

def _main_step23_xchg_env(*, boot_form, bounce_terms_for_ranks, dry, log, ns, ring_plan, xchg_res):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    xchg_env = prepare_xchg_env(log, str(ring_plan.epoch),
                                  ns.pdflip_weight_source, dry=dry,
                                  hop_bound_ms=ns.pdflip_shadow_hop_bound_ms,
                                  oncard_mode=ns.pdflip_xchg_oncard,
                                  oncard_slot_mib=ns.pdflip_xchg_oncard_slot_mib,
                                  bounce_terms=bounce_terms_for_ranks,
                                  inject_mode=ns.pdflip_xchg_inject,
                                  legs=ns.pdflip_xchg_legs,
                                  manifest_dir=getattr(ns, "evidence_dir",
                                                       EVIDENCE_DIR),
                                  # B4d: the partition W71 PRICED, not a second
                                  # derivation of it -- `xchg_res` is the solved
                                  # table and carries the census's own list.
                                  waves=(xchg_res.partition if xchg_res is not None
                                         else None))
    # #106 (fnFL2w42/w43): DIE GEMEINSAME STORE-GEOMETRIE, ENDLICH GESETZT.
    #
    # `expert_store.shared_geometry()` existiert seit #91 und sein Docstring
    # sagt "Gesetzt vom Launcher ueber STORE_GEOMETRY_ENV" -- der Launcher
    # setzte sie NIE. Im ganzen Baum stand ein Leser ohne Schreiber, also
    # rechnete jede Gruppe ihre eigene Abbildung globale Id -> Slot auf
    # DERSELBEN Datei. Was #91 als Gefahr beschreibt, war der Normalfall.
    #
    # Am Metall: w42 und w43 starben beim Laden von D mit
    #   ValueError: shared store .../L0-w13_weight_scale.bin
    #               has 23091200 bytes, this layout wants 26214400
    # P rechnete aus seinen Vektoren 451 Plaetze, D aus seinen 512. w41 lief
    # davor nur, weil FR_P zufaellig auf dieselbe Zahl fuehrte -- kein
    # Beweis, ein Zufall, und er verdeckte den fehlenden Schreiber.
    #
    # D's Vektoren sind die Autoritaet: die PP-Gruppe haelt jeden Experten
    # ungeshardet (`num_local == num_global`, lo=0, pad=False) und hat gar
    # keine eigene Aufteilung, waehrend D genau die Aufteilung IST, um die
    # es geht. Fehlt einer der beiden Vektoren, wird NICHTS publiziert --
    # dann rechnen beide wie bisher, statt dass eine halbe Geometrie eine
    # falsche Zuordnung festschreibt (#91: "eine falsche gemeinsame
    # Zuordnung waere schlimmer als gar keine").
    _geom_ratios = _argv_vector(getattr(ns, "extra_d", ""), "--rank-moe-ratio")
    _geom_fracs = _argv_vector(getattr(ns, "extra_d", ""),
                               "--rank-moe-resident-fraction")
    _skip_106 = pdflip_form.gate_skip_line("#106 PDFLIP-STORE-GEOMETRY", boot_form)
    if _skip_106:
        log(_skip_106)
    elif _geom_ratios and _geom_fracs and len(_geom_ratios) == len(_geom_fracs):
        xchg_env["FLLIPER_MOE_EXPERT_STORE_GEOMETRY"] = (
            f"{','.join(_geom_ratios)}|{','.join(_geom_fracs)}")
        log(f"PDFLIP-STORE-GEOMETRY shared={xchg_env['FLLIPER_MOE_EXPERT_STORE_GEOMETRY']}"
            f" source=extra_d -- BEIDE Gruppen rechnen dieselbe Abbildung"
            f" globale Id -> Slot (#106; ohne sie rechnete P 451 und D 512"
            f" Plaetze auf derselben Datei und D starb beim Laden)")
    else:
        log(f"PDFLIP-STORE-GEOMETRY none ratios={_geom_ratios} fracs={_geom_fracs}"
            f" -- keine gemeinsame Geometrie publiziert; jede Gruppe rechnet"
            f" wie vor #106. Das ist der KONSERVATIVE Fall, nicht der"
            f" gewuenschte: passen die Vektoren nicht zusammen, kollidiert"
            f" der geteilte Store weiterhin beim Oeffnen der Datei")
    return _geom_fracs, _geom_ratios, xchg_env

def _main_step24__adopt(*, boot_form, log, ns, xchg_env):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # #108 ERSTBOOT-ADOPTION: die Entscheidung an BEIDE Gruppen.
    # An P, weil P wissen muss, dass seine Bytes gebraucht werden (und es
    # nicht schlafen legen darf, bevor der Erstflip lief); an D, weil dort
    # die zwei Riegel daran haengen. Ein Kanal, wie bei #106/#107 -- zwei
    # Kanaele fuer eine Entscheidung sind der Weg, auf dem zwei Antworten
    # entstehen.
    try:
        from flliper.srt.pdflip import adopt as _adopt

        _adopt_on = _adopt.ADOPT_ON if _d_adopt_armed(ns) else "off"
        xchg_env[_adopt.ADOPT_ENV] = _adopt_on
        if _d_adopt_armed(ns):
            # #113: OHNE DIESE ENV KANN DIE UNION-ARENA DIE GEWICHTE NICHT
            # ERREICHEN. tms_csrc/utils.h erzeugt jede Allokation ohne
            # `requestedHandleTypes`; `cuMemExportToShareableHandle`
            # scheitert dann per Definition, und die metallbewiesene
            # Union-Arena (500c984795) bleibt bei ihren eigenen 1,50 GiB
            # statt an die ~27 GB der Gewichte zu kommen. Die Env geht an
            # BEIDE Gruppen -- der Besitzer muss exportierbar anlegen, der
            # Peer muss importieren koennen, und eine Entscheidung, die nur
            # eine Seite erreicht, ist keine.
            xchg_env["FLLIPER_PDFLIP_VMM_EXPORTABLE"] = "1"
            # #68: der Datei-Pool gibt EINEN Task je Shard; auf diesem
            # Checkpoint sind das ~16 000 Tensoren in EINEM Thread. Diese Env
            # teilt die Keys einer Datei auf mehrere Threads -- an BEIDE
            # Gruppen, weil beide denselben Loader fahren. Ohne Schreiber waere
            # sie wieder eine Env, die nur gelesen wird (heute elfmal).
            xchg_env["FLLIPER_LOAD_KEY_WORKERS"] = str(
                max(1, int(os.environ.get("PDFLIP_LOAD_KEY_WORKERS", "4") or 4))
            )
            log("PDFLIP-VMM-EXPORTABLE on -- each TMS allocation gets "
                "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, damit die "
                "Union-Arena die Gewichtsbytes zwischen den Phasen teilen "
                "kann statt sie zweimal zu halten (#113).")
        if _adopt_on == _adopt.ADOPT_ON:
            log("PDFLIP-D-ADOPT on -- D starts with --load-format dummy and "
                "holt seine Gewichte vom ERSTFLIP aus P's Karten. Riegel: "
                "kein Store-Schreiben und kein Forward auf D, bis der Flip "
                "vollstaendig gedeckt hat (#108).")
    except BaseException as _exc:  # noqa: BLE001 -- kippt nie den Boot
        log(f"PDFLIP-D-ADOPT failed: {type(_exc).__name__}: {_exc}")

    # #107: DIE EXPERTEN-KARTE. Nutzer-Gesetz 22.09.: "alles was geshardet
    # wird braucht ne karte". #106 publiziert die zwei VEKTOREN, aus denen
    # jede Gruppe ihre Aufteilung ableitet -- das reicht nicht, denn eine
    # Ableitung kann von der anderen abweichen (acht tote Boots am 22.09.,
    # zuletzt an funf Ids, weil die rohen Ratios 183,137,168 statt der
    # skalierten 192,144,176 als Bereichsgrenzen dienten).
    #
    # Die Karte schreibt die Aufteilung HIN, einmal, hier: wer haelt welche
    # globale Id, und wo liegt der Rest. Beide Gruppen LESEN sie.
    #
    # Sie ist der TAUSCH, nicht die Vereinigung (Nutzer 07:08Z): "die vram
    # layer zurueck in den moe cache und die benoetigten experten in den
    # vram, kein zusaetzlicher systemram". Der Store haelt darum
    # `total - resident` Zeilen -- 324 bei 188 residenten, exakt die in
    # fnFL2w24/w30 gemessene Zahl -- und nicht die Vereinigung (420).
    # PDFLIP-FORM: bands and map are declared for arch=moe (FORM_GATES).
    _moe_skips = [x for x in (
        pdflip_form.gate_skip_line("#134 PDFLIP-EXPERT-BAND", boot_form),
        pdflip_form.gate_skip_line("#107 PDFLIP-EXPERT-MAP", boot_form)) if x]
    for _x in _moe_skips:
        log(_x)
    return _moe_skips

def _main_step25__bands_on(*, _geom_fracs, _geom_ratios, _moe_skips, log, ns, xchg_env):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    try:
        from flliper.srt.layers.moe import expert_map as _em

        # #130 DIESELBE QUELLE WIE DER PP-CUT.
        #
        # Hier stand `_argv_vector(ns.extra_p, "--pp-cut-expert-device-
        # fraction")` -- aber dieses Flag ist ein LAUNCHER-Flag und steht
        # nie in `--extra-p`. Der Ausdruck war IMMER leer, `_pp_frac`
        # IMMER None, und die Karte fiel auf ihre Default-Residenz 188
        # zurueck: slots = 512-188 = 324, in JEDEM Boot, unabhaengig von
        # jeder Fraction (gemessen fnFL2w60 FR 0.367 und fnFL2w62 FR
        # 0.367,0.75,0.95 -- beide slots=324, shared_resident=92).
        # Der PP-Cut selbst liest `ns.pp_cut_expert_device_fraction`
        # (launcher.py:9629) und sah die Fractions korrekt -- zwei Leser,
        # zwei Quellen, eine davon leer.
        _fr_p = _split_fraction_text(
            getattr(ns, "pp_cut_expert_device_fraction", "")
        ) or _argv_vector(getattr(ns, "extra_p", ""),
                          "--rank-moe-resident-fraction")
        # #131 DIE EINE FRAKTION MUSS DAS LAYER-GEWICHTETE MITTEL SEIN.
        #
        # `resident_unsharded` modelliert die PP-Gruppe als EINEN Rang
        # ("sie haelt jeden Experten und residiert die ersten
        # round(total*fraction)") -- konzeptionell richtig, denn PP teilt
        # nach LAYERN, und jede Stufe haelt alle Experten IHRER Layer.
        # Genau deshalb ist `_fr_p[0]` die falsche Zahl: sie ist die
        # Residenz der ERSTEN Stufe, nicht die der Gruppe.
        #
        # GEMESSEN fnFL2w62: FR_P 0.367,0.75,0.95 auf Stufen 29,11,8
        # Layer -> `_fr_p[0]`=0,367 -> resident 188 -> slots 324 ->
        # 37 GB tmpfs-Store, waehrend die Karten 71/83/78 % tragen.
        # Gewichtet: (29x0,367 + 11x0,75 + 8x0,95)/48 = 0,5519
        # -> resident 283 -> slots 229 -> Store ~26 GB. 11 GB weniger,
        # und der Host stand bei 2 GB frei.
        # Gewichtet wird mit den PP-STUFEN (Layer je Rang), nicht mit
        # `_geom_ratios` -- das sind D's MoE-Ratios und hier die falsche
        # Groesse.
        _ne = _argv_vector(getattr(ns, "extra_d", ""), "--num-experts")
        _total = int(_ne[0]) if _ne else 512

        # #134: DIE BANDBREITE, aus derselben Geometrie wie die Karte und
        # UNABHAENGIG davon, ob die Karte publiziert wird -- sie braucht nur
        # die Ratios, nicht die Fractions, und eine refuesende Karte darf
        # nicht stillschweigend auch die Bandteilung abschalten.
        # Die Breite ist der ggT der D-Spans (expert_map.band_geometry), also
        # faellt jede Besitzgrenze von D auf eine Bandgrenze: ein Band hat
        # genau EINEN Halter je Phase. Im Arm gepinnt waere sie die naechste
        # Instanz von `arm-defaults-die-den-boot-toeten` -- andere Ratios,
        # andere Breite, und niemand denkt daran.
        # #134, NACH fnFL2w67: DIE SCHEIBE GEHT GANZ AN ODER GANZ AUS, und
        # das ist nicht Vorsicht, sondern das, was der Plan selbst verlangt:
        # `build_plan` wirft W74 fuer JEDEN Wellen-Tag ohne Deskriptor
        # ("a wave that moves nothing is an agreement between six ranks that
        # no byte has to arrive"). Ein Band-Tag in der Familie, dessen Bytes
        # der Plan nicht traegt, ist also kein halber Gewinn -- er ist ein
        # toter Boot. Solange die Baender nicht im xchg_manifest stehen,
        # bleibt die Teilung AUS und der Flip faehrt je Layer-Chunk wie
        # vorher. Der Schalter ist EXPLIZIT: ein Default, der halb
        # einschaltet, waere die Falle, vor der der Arm-Kopf warnt.
        _bands_on = os.environ.get("FLLIPER_PDFLIP_EXPERT_BANDS_ARM", "0") == "1"
        if _geom_ratios and not _bands_on:
            log("PDFLIP-EXPERT-BAND aus (FLLIPER_PDFLIP_EXPERT_BANDS_ARM!=1): die "
                "Tag-Ebene steht (#134), die Bytes stehen noch nicht im "
                "xchg_manifest -- ein Band-Tag ohne Deskriptor waere W74")
        if _geom_ratios and _bands_on:
            _bs, _bc = _em.band_geometry([int(x) for x in _geom_ratios], _total)
            if _bs > 0:
                xchg_env["FLLIPER_PDFLIP_EXPERT_BAND_SIZE"] = str(_bs)
                xchg_env["FLLIPER_PDFLIP_EXPERT_BANDS"] = str(_bc)
                log(f"PDFLIP-EXPERT-BAND size={_bs} count={_bc} "
                    f"spans={_em.scaled_spans([int(x) for x in _geom_ratios], _total)} "
                    f"-- die Verschiebeeinheit des Flips ist jetzt das "
                    f"EXPERTEN-BAND je Layer-Chunk (#134), nicht der ganze "
                    f"Layer-Chunk")
            else:
                log(f"PDFLIP-EXPERT-BAND aus: ratios={_geom_ratios} total={_total} "
                    f"ergeben keine Breite, die jede D-Grenze trifft und unter "
                    f"{_em.MAX_BANDS} Baendern bleibt -- der Flip transportiert "
                    f"je Layer-Chunk wie vorher (langsamer, nicht falsch)")

        # #132: DER VEKTOR GEHT DURCH, NICHT SEIN MITTEL.
        #
        # #131 hatte hier das layer-gewichtete Mittel gebildet -- das war
        # die falsche Antwort auf die richtige Frage: es gaukelt der Karte
        # eine Residenz vor, die KEINE Stufe hat, und fnFL2w64 starb an
        # "95 eigene kalte Experten haben in der KARTE keinen Platz".
        # `resident_unsharded` nimmt seit #132 einen Vektor; die Karte
        # rechnet die drei Stufen einzeln, und `kalt_p` ist, was
        # MINDESTENS EINE Stufe nicht haelt.
        _pp_frac = list(_fr_p) if _fr_p else None
        if _geom_ratios and _geom_fracs and _pp_frac is not None:
            _karte = _em.build(
                total=_total,
                ratios=[int(x) for x in _geom_ratios],
                fr_pp=_pp_frac,
                fr_tp=[float(x) for x in _geom_fracs])
            _reason = _em.refuse_if_inconsistent(_karte)
            if _reason:
                log(f"PDFLIP-EXPERT-MAP REFUSED: {_reason} -- keine Karte "
                    f"publiziert, beide Gruppen rechnen wie vor #107")
            else:
                import json as _json

                _mp = os.path.join(expert_map_dir(getattr(ns, "evidence_dir", EVIDENCE_DIR), log),
                                   f"expert_map_{ns.tag}.json")
                with open(_mp, "w") as _fh:
                    _json.dump(_karte, _fh)
                xchg_env[_em.MAP_ENV] = _mp
                log(f"PDFLIP-EXPERT-MAP {_mp} total={_karte['total']} "
                    f"slots={_karte['slots']} spans={_karte['spans']} "
                    f"bounds={_karte['bounds']} moves={_karte['moves']} "
                    f"shared_resident={_karte['shared_resident']} -- BEIDE "
                    f"Gruppen lesen diese Karte statt zu rechnen (#107); "
                    f"slots ist der TAUSCH (total-resident), nicht die "
                    f"Vereinigung")
        elif not _moe_skips:
            log(f"PDFLIP-EXPERT-MAP none ratios={_geom_ratios} "
                f"fracs={_geom_fracs} pp_frac={_pp_frac} -- ohne alle drei "
                f"Angaben wird KEINE Karte publiziert (eine halbe waere "
                f"schlimmer als keine)")
    except BaseException as _exc:  # noqa: BLE001 -- eine Karte kippt nie den Boot
        log(f"PDFLIP-EXPERT-MAP failed: {type(_exc).__name__}: {_exc}")

def _main_step26_xchg_env_weight_exch(*, hicache_disabled, log, ns, ring_plan, state, xchg_env):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # #1369: PUBLISHED UNCONDITIONALLY, unlike everything else in `xchg_env`
    # above -- `prepare_xchg_env` returns `{}` outright under an un-armed
    # exchange (`weight_source not in WEIGHT_SOURCE_ARMED`), which is correct
    # for the arm-specific terms above (there is nothing to tell a rank about
    # a bounce that will not run) but would be wrong here: an operator who
    # explicitly typed `--pdflip-weights-cpu-backup off` under `ring` (the hard
    # kill switch, named in that flag's own help text as working "including
    # under ring") must have every rank see that override, not just the
    # launcher's own process. The raw mode string travels, exactly as
    # `--pdflip-xchg-inject` does through `INJECT_ENV`: a rank with no
    # `explicit` re-derives `auto` from its OWN `exchange_armed()` read,
    # which is published to it separately and independently correct either
    # way.
    xchg_env[weight_exchange.WEIGHTS_CPU_BACKUP_ENV] = str(
        getattr(ns, "pdflip_weights_cpu_backup",
                weight_exchange.WEIGHTS_CPU_BACKUP_AUTO))

    # 2. host ledger
    # #1362 [22-fix2]: the form verdict lives on the SOLVED TABLE, not on the
    # plan that wraps it -- reading `form_same` off the PLAN with a False
    # default reads False on every boot and makes the transition path dead code
    # that no dry run could distinguish from a working one (measured: gdncov2
    # still refused W99 with the form key printed as MATCHES four lines above).
    # `None` = the gate was never armed, and an unarmed gate must never read as
    # a passed one: it stays False, i.e. a legacy record stays refused.
    _rt = getattr(ring_plan, "table", None)
    _form_key_matches = bool(getattr(_rt, "form_same", None)) if _rt is not None else False
    # rc12d: the allocation-history post, from the SAME environments the
    # groups will see (the launcher's own = container env, overlaid by
    # --env-p / --env-d), priced per armed rank.
    from flliper.srt.pdflip import memhist as _memhist

    _memhist_gib, _memhist_run_only, _memhist_prov = _memhist.host_charge(
        os.environ,
        {"P": parse_group_env(getattr(ns, "env_p", "") or ""),
         "D": parse_group_env(getattr(ns, "env_d", "") or "")},
        {"P": 3, "D": 3},
    )
    if _memhist_prov:
        log(f"PDFLIP-MEMHIST ledger post memhist {_memhist_gib:.2f} GiB "
            f"({'run moment only' if _memhist_run_only else 'both moments'}): {_memhist_prov}")

    # 28.09. (user order): the L3 store's RAM index is a host post -- entries
    # (snapshot header, else the file count of the ONE walk the first boot
    # makes) x RAM_BYTES_PER_ENTRY, once per owner (one per group).
    from flliper.srt.mem_cache.storage.file import store_journal as _store_journal

    _l3_idx_ident = (l3_persist_identity(ns.model, getattr(ns, "profile", ""),
                                         getattr(ns, "form_kv", "") or "",
                                         extra_p=getattr(ns, "extra_p", "") or "",
                                         extra_d=getattr(ns, "extra_d", "") or "",
                                         vision=str(getattr(ns, "pdflip_vision", "") or ""))
                     if l3_persist_enabled() else None)
    _l3_idx_owners = 1 if getattr(ns, "d_only", False) else 2
    _l3_idx_n, _l3_idx_src = (
        _store_journal.index_entries(f"{STORE_ROOT}/{l3_persist_dir_name(_l3_idx_ident)}")
        if _l3_idx_ident and _store_journal.enabled() else (0, "no persistent L3 (per-boot store)"))
    _l3_index_gib = _l3_idx_n * _store_journal.RAM_BYTES_PER_ENTRY * _l3_idx_owners / (1 << 30)
    log(f"PDFLIP-HOST-LEDGER L3 INDEX post l3_index {_l3_index_gib:.2f} GiB = {_l3_idx_n} entries "
        f"({_l3_idx_src}) x {_store_journal.RAM_BYTES_PER_ENTRY} B x {_l3_idx_owners} owner(s), "
        f"both moments; grows with the store up to its cap between boots")

    # 29.09. (Planer post COLD_TIER_SHM): the tmpfs expert store, sized from the
    # map the ranks will load (dry-built here, re-priced below once written).
    _ledger_group_envs = [parse_group_env(getattr(ns, "env_d", "") or ""),
                          parse_group_env(getattr(ns, "env_p", "") or "")]; _ = hicache_disabled or log(mamba_arena_slots_into_groups(ns, _ledger_group_envs))  # MAMBA-ARENA-FLOOR 16.10.: raised before the ledger prices the arena (line-neutral: the catalog and tests cite launcher lines)
    _cold_tier_shm_gib, _cold_tier_line = cold_tier_shm_post(
        ns, ns.model, getattr(state, "p_stage_layers", None))
    log(_cold_tier_line)
    # 29.09. (host census): what this model|form was MEASURED to hold outside
    # the ranks and outside store/arena -- a record, never a constant; the
    # front keeps it current (sample_live every 60 s, max-merged).
    from flliper.srt.pdflip import host_census as _hc

    _census_key = _hc.census_key(
        host_ledger.checkpoint_digest(ns.model)[0] or "",
        (getattr(ns, "pdflip_boot_form", None).env_value()
         if getattr(ns, "pdflip_boot_form", None) is not None else ""))
    return _census_key, _cold_tier_shm_gib, _form_key_matches, _hc, _l3_index_gib, _ledger_group_envs, _memhist_gib, _memhist_run_only

def _main_step27_arm(*, _ledger_group_envs, _price_host_ledger, hicache_disabled, log, ns, state):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    arm, reap_headroom_gib, lines, cg = _price_host_ledger()
    # #1453 (user 16.09.: 'L2-Groesse aus Ledger-Spielraum'): the arena is the
    # largest host term (33.6 GiB with draft+mamba); when the priced run peak
    # sits ABOVE the riegel, the KV arena gives that excess back and the
    # ledger prices once more -- the boot then funds itself without the
    # --host-ledger-deviation it needed before (weg2xsn207: 93.15 vs 93.00).
    _arena_new, _arena_why = arena_from_ledger(
        float(_group_env_value("FLLIPER_HICACHE_ARENA_GIB", _ledger_group_envs, "22")),
        # 29.09.: the latch the ledger derived (a flag only tightens it)
        _arm_run_peak_gib(arm), getattr(arm, "riegel_effective", getattr(ns, "host_riegel_gib", None)),
        enabled=(os.environ.get("FLLIPER_PDFLIP_ARENA_FROM_LEDGER", "1") == "1" and not hicache_disabled))
    log("#1453 L2-ARENA FROM LEDGER: " + _arena_why)
    if _arena_new is not None:
        os.environ["FLLIPER_HICACHE_ARENA_GIB"] = f"{_arena_new:g}"
        # 29.09. (arena8 09291303): a group that names the arena overrides the
        # launcher env in build_env, so the new size goes where the ranks and
        # the ledger read it -- otherwise the ledger re-prices a number no rank gets.
        arena_gib_into_groups(ns, _ledger_group_envs, _arena_new)
        arm, reap_headroom_gib, lines, cg = _price_host_ledger()
        log("#1453 L2-ARENA FROM LEDGER: re-priced with arena %.2f GiB -> run_peak %s GiB"
            % (_arena_new, _fmt_gib(_arm_run_peak_gib(arm))))
    state.cgroup = dict(cg)
    return arm, cg, lines, reap_headroom_gib

def _main_step28_state_ledger_lines(*, arm, cg, cut, dry, lines, log, ns, reap_headroom_gib, state):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    state.ledger_lines = lines
    # 29.09. (z30x2-yarn2): ONE pinned-host reserve for the ledger and the
    # ranks. The ledger bounded its run peak by memory.max minus this number
    # (PINNED WALL); the ranks' HiCache pools now read it too, through
    # pinned_host_reserve(). An explicit FLLIPER_PINNED_HOST_RESERVE_GIB wins
    # and is left untouched; without a finite cgroup nothing is exported.
    _pin_gib, _pin_src = getattr(arm, "pinned_reserve", (None, ""))
    from flliper.srt.mem_cache import pinned_host_budget as _phb
    if _pin_gib is not None and not (os.environ.get(_phb.PINNED_HOST_RESERVE_ENV) or "").strip():
        os.environ[_phb.PINNED_HOST_RESERVE_LEDGER_ENV] = f"{float(_pin_gib):.4f}"
        os.environ[_phb.PINNED_HOST_RESERVE_LEDGER_SOURCE_ENV] = str(_pin_src or "")
    log(pinned_reserve_line(_pin_gib, _pin_src))
    state.reap_headroom_gib = reap_headroom_gib

    # #1269 fix 4 follow-up: KEEP THE PREFLIGHT'S OWN ANON READING. `cg` is the
    # cgroup snapshot the ledger just took, BEFORE any group process exists, so
    # `cg["anon"]` is exactly the baseline W22's `flliper=/foreign=` split has to
    # subtract. It was already measured and already returned here, and was being
    # dropped -- sb5c's split was reconstructed by hand afterwards.
    #
    # ONE READ, CARRIED -- not a second `read_cgroup()` at front-launch time.
    # A baseline taken after the groups start is not a baseline, and two reads
    # are two quantities; the fix-2 lesson, on a measurement instead of a key.
    anon_preboot_bytes = int(cg.get("anon") or 0)
    log(f"PDFLIP-LAUNCH ANON-BASELINE preboot_anon={anon_preboot_bytes / 1073741824:.2f} GiB "
        f"(cgroup memory.stat anon at PREFLIGHT, before any group process exists; "
        f"handed to the front as --anon-preboot-bytes so W22 can split its reading into "
        f"flliper= and foreign=. 0 = unreadable -> the front prints NO split rather than "
        f"an invented one; sb5c had to be split by hand because this value was dropped)")

    # 2b. host memory time series from BEFORE the first group, so the launch
    # moment (P image resident, D loading) is measured this time, not only
    # the run moment (record 1h: the ls1b2 sampler started after D READY).
    if not dry:
        start_memts(state, log)

    # 3. store -- ON DISK, SIZED FROM THE P KV POOL (#1236).
    #
    # ORDERING, and why it is safe: `cut` (solve_p_cut) is already solved above,
    # before the ring, so the pool this store must cover is known here without
    # moving anything.  The ledger ran first only because the ARC line prices
    # itself against the reap headroom the ledger computes.
    _l3_ident = (l3_persist_identity(ns.model, getattr(ns, "profile", ""),
                                     getattr(ns, "form_kv", "") or "",
                                     extra_p=getattr(ns, "extra_p", "") or "",
                                     extra_d=getattr(ns, "extra_d", "") or "",
                                     vision=str(getattr(ns, "pdflip_vision", "") or ""))
                 if l3_persist_enabled() else None)
    store_plan = plan_store(
        ns.tag,
        cut.pool_tokens,
        cut.attn_counts,
        cut.kv_mib_per_token_per_attn_layer,
        sidecar_factor=ns.store_sidecar_factor,
        store_max_gb=ns.store_max_gb,
        min_free_gib=ns.store_disk_min_free_gib,
        directory_name=(l3_persist_dir_name(_l3_ident) if _l3_ident else None),
        dry=dry,
    )
    store_cfg = store_plan.extra_config()
    log(store_plan.disk_line(
        "WARN store on %s, not the told disk (dry run: not refused, the space check "
        "is skipped)" % store_plan.fs_type if store_plan.fs_refused else "ok"))
    return _l3_ident, anon_preboot_bytes, store_cfg, store_plan

def _main_step29_log_f_pdfli(*, _l3_ident, dry, log, max_kv_per_request, ns, reap_headroom_gib, state, store_cfg, store_plan):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    log(
        f"PDFLIP-STORE: dir={store_plan.directory} ON DISK (ZFS dataset, plain directory "
        f"-- #1236 user ruling 2026-09-09: 'normales hicaching mit lvl2 und lvl3', the "
        f"upstream HiCacheFile backend, no new tier and no overflow mechanism; the "
        f"tmpfs form and its mount/umount are DELETED, not flagged off). "
        f"max_size={store_plan.max_size_bytes} B "
        f"({store_plan.max_size_bytes / host_ledger.GIB:.2f} GiB, scope=shared over the "
        f"writer ranks) = P POOL {store_plan.p_pool_tokens} tokens x "
        f"{store_plan.cell_bytes} B/token = "
        f"{store_plan.p_pool_bytes / host_ledger.GIB:.2f} GiB x sidecar factor "
        f"{store_plan.sidecar_factor:.2f} ({STORE_CENSUS_PROVENANCE}: kv={STORE_CENSUS_KV_PAGES} "
        f"pages x {STORE_CENSUS_KV_PAGE_BYTES} B, mamba={STORE_CENSUS_MAMBA_BLOBS} blobs x "
        f"{STORE_MAMBA_BLOB_BYTES} B measured on /spinning/hicache-l3, "
        f"draft={STORE_CENSUS_DRAFT_PAGES} pages x {host_ledger.DRAFT_PAGE_BYTES} B). "
        f"min_free={store_plan.min_free_bytes} B "
        f"({store_plan.min_free_bytes / host_ledger.GIB:.2f} GiB) AGAINST THIS DISK, "
        f"which has {store_plan.fs_free_bytes / host_ledger.GIB:.2f} GiB free of "
        f"{store_plan.fs_total_bytes / host_ledger.GIB:.2f} GiB -- not the old 1 GiB on a "
        f"6 GiB tmpfs that latched HiCacheFile's write stop after 5 GiB and refused 1,324 "
        f"group-P writes on boot weg2sb5g. "
        f"#1236 LAW CHECK: max_size >= P pool is "
        f"{'HELD' if store_plan.max_size_bytes >= store_plan.p_pool_bytes else 'VIOLATED'} "
        f"({store_plan.max_size_bytes / max(1, store_plan.p_pool_bytes):.2f}x; weg2sb5g ran "
        f"at 0.54x). extra_config={store_cfg}"
    )
    log(
        f"PDFLIP-STORE LEDGER DELTA: {store_plan.max_size_bytes / host_ledger.GIB:.2f} GiB of "
        f"store budget left the HOST RAM ledger entirely (max_size; min_free is free "
        f"space the store never writes into and was never a RAM post), because "
        f"this store is a directory on disk and its host cost is page cache plus the ZFS "
        f"ARC -- both reclaimable, neither in cgroup anon, neither what the reaper kills "
        f"for. The reap headroom the ledger now reports is "
        + ("unreadable (no cgroup sample)" if reap_headroom_gib is None
           else f"{reap_headroom_gib:.2f} GiB")
        + ", and it is larger than the pre-#1236 form's by exactly the store term that "
        "form charged (6 GiB on boot weg2sb5g). NOT a claim that the box gained RAM: the "
        "bytes moved to a device, they did not disappear."
    )
    log(arc_preflight_line(read_arc_state(), store_plan, reap_headroom_gib))
    log(store_read_cost_line(store_plan, max_kv_per_request))
    sweep_store_residue(log, store_plan.directory, dry)
    if _l3_ident:
        # identity FIRST: a store of another identity is refused before this
        # boot touches a single file in it.
        l3_persist_check_rope(getattr(ns, "extra_p", "") or "", getattr(ns, "extra_d", "") or "")
        _l3_state = l3_persist_check_identity(store_plan.directory, _l3_ident, dry)
        log(f"L3-PERSIST identity={_l3_state} dir={store_plan.directory} "
            f"{json.dumps(_l3_ident, sort_keys=True)}")
        _l3_files, _l3_bytes, _ = l3_persist_attach(log, store_plan.directory, dry)
        l3_persist_disk_sum(log, os.path.dirname(store_plan.directory), store_plan.directory,
                            store_plan.max_size_bytes, store_plan.min_free_bytes, _l3_bytes, dry)
    store_dir = prepare_store(log, store_plan, dry)
    state.store_dir = store_dir
    state.store_max_size_bytes = store_plan.max_size_bytes
    state.store_min_free_bytes = store_plan.min_free_bytes
    return store_dir

def _main_step31_ns__host_census(*, _census_key, _census_record, _emap, budgets_p, cards, chunk_count, chunk_layers, chunk_tokens, cvd, log, ns, ring_plan, store_dir, tms_so, tree, user_reserve_by_card, xchg_env):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # 29.09.: the front samples the census live (widths known once the arena
    # term has run inside the pricing above)
    ns._host_census = {"record": _census_record, "key": _census_key,
                       "store_dir": expert_store_dir_of(ns),
                       "widths": list(_ARENA_BOOKED_WIDTHS.get(str(ns.model), []))}
    _estore_id = publish_store_identity(ns.model, _emap, log)
    env_p = build_env(tree, ns.venv, cvd, store_dir, ns.debug_hold in ("P", "both"), ns.tag, chunk_layers, chunk_count, tms_so, ns.transport, ring_plan, group="P", xchg_env=xchg_env, group_env_extra=parse_group_env(getattr(ns, "env_p", "")), **_env_knobs(ns), **rank_reserve_kwargs("P", ns, cards, user_reserve_by_card, log), expert_map_path=_emap, expert_store_identity=_estore_id)
    # H125: the host-RAM price of `--pdflip-vision-source ram`, named where the
    # P env is built (the line is empty, and nothing is logged, for `disk`).
    # VISION-GGUF: only a transient boot resolves the tower's directory (a GGUF reads it from --tokenizer-path)
    _vis_line = vision_source_host_line(
        vision_tower_dir(ns), str(getattr(ns, "pdflip_vision_source", VISION_SOURCE_DISK))
    ) if str(getattr(ns, "pdflip_vision", VISION_OFF)) == VISION_TRANSIENT else ""
    if _vis_line:
        log(_vis_line)
    # #1269 PUBLICATION: build_env's own comment says the decision is
    # "published explicitly so the boot log names the decision" -- but it only
    # SET the variables and logged nothing, so no boot log ever carried them.
    # A knob that shapes host memory and is never named is exactly the
    # instrument-that-does-not-report shape; the value is a CEILING derived
    # from the reset path's thread names, not a live count, and
    # PDFLIP-IDLE-CENSUS `threads=` is what confirms or corrects it.
    log(f"PDFLIP MALLOC-ARENAS MALLOC_ARENA_MAX={env_p.get('MALLOC_ARENA_MAX')} "
        f"FLLIPER_IDLE_BLOCKING_POLL={env_p.get('FLLIPER_IDLE_BLOCKING_POLL')} "
        f"(both groups) -- provenance: glibc defaults to 8 x ncores arenas and this "
        f"box reports affinity 0-31, so the default ceiling is 256; measured growth was "
        f"64 MiB-aligned [anon] arenas extended in place plus NEW arenas appearing "
        f"(+5.4 MiB/min on P PP0/PP1, +2.8 on each D rank, +19.0 MiB/min over six ranks "
        f"against an independently observed +21.3). 4 = the scheduler thread plus the "
        f"three the reset path names (prefetch, backup, prefetch_io_aux), each with its "
        f"own arena. A CEILING, not a target. DERIVED FROM THREAD NAMES, NOT A LIVE "
        f"COUNT -- the rank thread census was never captured before the host OOM took "
        f"the base down; PDFLIP-IDLE-CENSUS threads= confirms or corrects it (#1269).")
    # #1233 zero-remainder: group P ends every prefill's last chunk at N-1 and
    # publishes the recurrent anchor there (schedule_policy END-OF-PREFILL
    # ANCHOR); D can claim at most N-1 tokens of a prompt, so this is the
    # anchor it resumes from. P only: D's finish anchors serve the NEXT turn.
    env_p["FLLIPER_PDFLIP_END_ANCHOR"] = "1"
    # P-TRIM-END-ANCHOR (pdflip/p_trim_end_anchor.py): {} when off, so the
    # default env stays byte-identical; on, P takes a leg-1 prompt as N-1 and
    # the 1-token END-ANCHOR forward above never runs for it.
    _p_trim_env = p_trim_end_anchor_env(bool(getattr(ns, "p_trim_end_anchor", False)))
    env_p.update(_p_trim_env)
    # FORK ANCHOR (pdflip/fork_anchor.py): {} when off; the same env goes to D.
    _fork_env = fork_anchor_env(getattr(ns, "fork_anchor_token", None),
                                bool(getattr(ns, "p_trim_end_anchor", False)))
    env_p.update(_fork_env)
    # TURN ANCHOR (pdflip/turn_anchor.py): {} when off; group P only.
    _turn_env = turn_anchor_env(getattr(ns, "turn_anchor_token", None))
    env_p.update(_turn_env)
    if _turn_env:
        log("PDFLIP TURN-ANCHOR: on (--turn-anchor-token %s) -- group P snapshots the "
            "recurrent state where a prompt's last message starts as a second extend "
            "track and inserts it as its own anchor (rank lines 'PDFLIP TURN-ANCHOR "
            "TRACK', 'PDFLIP TURN-ANCHOR INSERT')" % ns.turn_anchor_token)
    if _fork_env:
        log("PDFLIP FORK-ANCHOR: on (--fork-anchor-token %s) -- group P cuts a leg-1 "
            "prompt before its generation prompt, group D reads the store up to the "
            "same cut and keeps its prefill anchor at or below it (rank lines 'PDFLIP "
            "P-TRIM-END-ANCHOR ... fork', 'PDFLIP FORK-ANCHOR TRACK')"
            % ns.fork_anchor_token)
    if _p_trim_env:
        log("PDFLIP P-TRIM-END-ANCHOR: on (--p-trim-end-anchor) -- group P takes every "
            "front leg-1 prompt of N tokens as N-1 (token level, at its intake): the "
            "last regular chunk ends at N-1, the finish insert is the N-1 anchor D "
            "claims, no 1-token END-ANCHOR forward; prompt_tokens and the #1442 "
            "hand-off still carry N; N<2, read outputs, sessions, embeds and "
            "multimodal prompts stay on the split path (rank line 'PDFLIP "
            "P-TRIM-END-ANCHOR kept(<reason>)')")
    # --p-attn-head-split: {} when off (env byte-identical). Same builder and
    # inputs as the pool model's post in solve_p_cut.
    _ah_cfg = p_attn_head_split_cfg(ns, len(budgets_p), chunk_tokens, ns.model)
    env_p.update(p_attn_head_split_env(_ah_cfg))
    if _ah_cfg is not None:
        log(p_attn_head_split_line(_ah_cfg, len(budgets_p)))
    # H92 --p-chunk-policy: ({}, []) under 'fixed' -- env and front log
    # byte-identical. FR_P is read from THIS env_p (the vector that ships).
    _pcp_env, _pcp_lines = p_chunk_policy_install(env_p)
    env_p.update(_pcp_env)
    return _estore_id, _pcp_lines, env_p

def _main_step33_chunk_tokens_armed(*, arm, calib_log_accept, chunk_tokens, cut, env_p, hicache_disabled, log, max_kv_per_request, ns, store_cfg):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # TRAIN FIX 5: the chunk size the cut solver was given BEFORE the ring is
    # re-read here against the arm this boot actually chose.  The hoist above
    # rests on --chunked-prefill-size being a CONSTANT of common_flags rather
    # than a function of the ledger arm; this line proves that premise on every
    # boot instead of assuming it, and a divergence is named rather than
    # silently shipping a cut solved for a chunk size the argv does not carry.
    chunk_tokens_armed = chunked_prefill_size_of(
        common_flags(ns.model, arm.s_gb, arm.m_mib, store_cfg, max_kv_per_request,
                     ns.p_hicache_write_policy, "P", ns.random_seed,
                     ns.barlink_bar1_cap_cycles, ns.collective_census_interval,
                     hicache_disabled=hicache_disabled)
    )
    if chunk_tokens_armed != chunk_tokens:
        raise PdFlipLaunchRefused(
            f"W40 PdFlipPPCutRefused: the PP cut was solved for "
            f"--chunked-prefill-size {chunk_tokens}, read before the host-ledger "
            f"arm existed, but the argv this boot ships carries "
            f"{chunk_tokens_armed}. The cut solve has to run BEFORE the ring "
            f"(the ring must be sized for the form that ships, and the cut is "
            f"part of that form), which is sound only while the chunk size does "
            f"not depend on the arm. It now does. Refusing rather than shipping "
            f"a cut solved against the wrong chunk."
        )
    # #1254 W46: THE MAP AND THE ARGV MUST BE THE SAME SPLIT.  The PDFLIP-FLIP-
    # ORDER MAP above is derived from the incumbent score constants
    # (p_stage_layers); argv_p ships whatever solve_p_cut returned.  While the
    # solver's winner differs from the incumbent those two halves describe
    # DIFFERENT layouts, and the map is then complete but WRONG PER CARD -- a
    # weights tag charged to one card while its bytes straddle two, which is
    # the accounting class that killed boot weg2dk4.  Neither half can see the
    # other, and interleave_pause_order refuses only an INCOMPLETE map, so a
    # confident wrong pause order is exactly what a silent divergence buys.
    # Checked here, by name, before either group starts.
    # #1240 THE LAUNCHER IS THE ONLY WRITER. The solved (or pinned) map is
    # published here, into the environment group P will actually get -- the
    # flag is the interface, the variable is the wire. A GAPPED map also arms
    # the #753 mid-loop crossing wire, because that is the executor: without
    # it a stage runs its own layers back to back and silently skips its
    # peer's, and get_pp_layer_set refuses a gapped set with the wire off.
    if cut.layer_set:
        env_p[PP_LAYER_SET_ENV] = cut.layer_set
        if cut.gapped:
            env_p[PP_CROSSING_WIRE_ENV] = "1"
        log(
            "PDFLIP LAYER-SET published for group P: %s=%s%s (source: %s). The "
            "#753 mid-loop crossing wire is the executor; nothing is "
            "duplicated here."
            % (
                PP_LAYER_SET_ENV,
                cut.layer_set,
                " " + PP_CROSSING_WIRE_ENV + "=1" if cut.gapped else "",
                "--pp-layer-set (PINNED)" if ns.pp_layer_set else "the solver",
            )
        )
    # #692 MICROBATCH DEPTH. The share is READ from a previous boot's own
    # PP-BUBBLE line, never written here: an operator-supplied path wins, else
    # the newest log in EVIDENCE_DIR that actually carries the instrument, else
    # no measurement at all and the depth is today's 0 -- printed, so the
    # absence is visible rather than inferred from a missing line.
    bubble_src = ns.p_bubble_measured_from or newest_bubble_log(
        EVIDENCE_DIR, accept=calib_log_accept)
    if ns.p_bubble_measured_from and read_pp_bubble(ns.p_bubble_measured_from) is None:
        raise PdFlipLaunchRefused(
            f"W42 PdFlipDepthUnfunded: --p-bubble-measured-from "
            f"{ns.p_bubble_measured_from} carries no PP-BUBBLE line, so it "
            f"states nothing about the bubble. An explicitly named measurement "
            f"file that turns out to be empty is refused rather than silently "
            f"treated as 'no measurement' -- that would read as depth 0 for a "
            f"reason the operator did not intend."
        )
    return bubble_src

def _main_step36_shipped_key(*, chunk_count, dry, env_p, form_key, form_norm, log, max_kv_per_request, ns, shipped_argv_p, state):
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # TRAIN FIX 5: THE SENTINEL PREMISE, PROVEN ON EVERY BOOT.  The form key that
    # gated the ring table was hashed over an argv built with sentinel ledger
    # terms, which is sound only while every flag those sentinels reach is
    # excluded from the key by name.  Here the REAL argv exists, so the key is
    # re-derived from it and any difference is a refusal -- the alternative is a
    # ring gated on a form that is not the one launched, which is a subtler
    # version of the very defect this gate exists to catch.
    # #1325c: THE SAME TRANSFORM ON BOTH SIDES OF THE COMPARISON.  `form_key`
    # above was hashed over `xchg_form_argv(...)`, which APPENDS the SYNTHETIC
    # `ring_table.XCHG_FORM_TOKEN` when the arm is armed (B4e).  That token is
    # a marker, never a flag `argv_p` emits, so a shipped argv can never carry
    # it and it is not in FORM_KEY_EXCLUDED_FLAGS either: comparing the raw
    # shipped key against the gated key refuses EVERY armed boot by
    # construction, and equals for every unarmed one -- which is why it was
    # invisible until the #1325b netting correction let an armed boot past the
    # host ledger for the first time (weg2xsn21, 2026-09-12: W21 at the ledger
    # on the head, W48 here the moment the ledger funded).  A check that
    # compares a TRANSFORMED value against an UNTRANSFORMED one tests the
    # transform, not its subject.  TRAIN FIX 5's subject -- "are the sentinel
    # ledger terms inert?" -- is untouched: the token is appended identically
    # to both sides and cancels, and a REAL form drift still refuses.
    shipped_key, shipped_norm = ring_table.p_form_key(
        xchg_form_argv(shipped_argv_p, ns.pdflip_weight_source)
    )
    if shipped_key != form_key:
        mine, theirs = set(shipped_norm.split(" ")), set(form_norm.split(" "))
        raise ring_table.PdFlipRingFormMismatch(
            f"W48 PdFlipRingFormMismatch: the host ring was gated on group-P form "
            f"key {form_key}, but the argv this boot ships hashes to "
            f"{shipped_key}. The gate ran against a form that is not the one "
            f"being launched, so the ring's size is not a statement about this "
            f"boot. Shipped-only: {sorted(mine - theirs)}; gated-only: "
            f"{sorted(theirs - mine)}. Either a flag reached by the sentinel "
            f"ledger terms is NOT in ring_table.FORM_KEY_EXCLUDED_FLAGS "
            f"({', '.join(ring_table.FORM_KEY_EXCLUDED_FLAGS)}), or a "
            f"form-changing flag is added to argv_p after the ring is armed. "
            f"Refusing before either group starts."
        )
    log(f"PDFLIP-P-FORM CONFIRMED key={shipped_key} on the argv actually shipped "
        f"(the ring was gated on this same key before the host-ledger arm "
        f"existed; the sentinel terms are proven inert, not assumed to be)")
    # P-COVERS-D-SESSION (law 02.10.): a named refusal, never a D prefill
    _cov_ok, _cov_line = p_covers_d_session(
        shipped_argv_p, d_session_tokens(ns.extra_d, max_kv_per_request))
    log(_cov_line)
    if not _cov_ok:
        raise PdFlipLaunchRefused(_cov_line)
    spec_p = GroupSpec("P", PORT_P, transport_argv(shipped_argv_p, ns.transport), state.logs["P"], env_p)
    state.argv["P"] = " ".join(shlex.quote(a) for a in spec_p.argv)
    log(w38_armed_line(spec_p.argv))
    # fnFL2 H37: how many requests one P forward may carry, off the SHIPPED argv.
    log(p_micro_batch_line(spec_p.argv))
    state.deviations = [
        "transports stay OPEN across sleep (barlink_reopen() unwired this round; BAR1 windows sized to fit both groups: P 24+96, D 16+32+40 MiB "
        "= 208 of 224 usable, measured Used 224/256 incl. RM carve-out; #1234 C1 raised dcp:0 from 24 so the 96-MiB dcp all_reduce plans to 10 rounds "
        "instead of 17 and stops falling to the host-staged plane)",
        "FLLIPER_ENABLE_HEALTH_ENDPOINT_GENERATION=0 on both groups (K2); /health_generate only via the front to the awake group",
        "group deadmen tier 1 only (TIER2=0 keeps every synthetic generate off the group ports; PROBE_S spaced past the boot as a second guard); tier 2 on the front's deadman",
        "no disk tier for the store (file backend has no second tier); tmpfs sized by the ledger",
        "the epoch is carried in the front's ledger only (no group echo/checksum on the RPC)",
        "torch_memory_saver 0.0.9.post1 preload hook rebuilt from the vendored csrc with ONE patch (cpu backup freed after resume, "
        "python/flliper/srt/pdflip/tms_csrc/PATCH.md); the stock wheel is untouched and used when FLLIPER_PDFLIP_TMS_PRELOAD_SO is unset",
        f"weights paused/resumed as {chunk_count} weights_<k> chunk tags + the base tag (#1233 one-backup flip); cuda_graph stays resident (not in the sleep tag set, as in round 1)",
        "zero-remainder: group P holds the last token of every prefill back into its own chunk (FLLIPER_PDFLIP_END_ANCHOR=1) so the GDN anchor D resumes from sits at N-1; "
        "costs P one 1-token pass per request and serialises whole-fit prompts behind the one chunked request per pass",
        "zero-remainder: /flush_cache publishes un-backed nodes before the idle witness on both groups (FLLIPER_HICACHE_FLUSH_PUBLISH_SWEEP=1); the quiesce then waits for those backups",
        # y7d (9bdfe50185) still declared the deleted CARRIER-EXCEEDS route here
        # ("served by ONE prefill on D, no leg 1"); 9bdfe50185 removed it.
        "zero-remainder: no D single-prefill route -- every BATCH prompt takes P's leg 1 (CARRIER-EXCEEDS deleted, P-COVERS-D-SESSION law 02.10.); "
        "a boot whose P prefill stages do not cover D's session is refused by name above, never served by a D prefill",
        "zero-remainder: BATCH streams are priced post hoc via stream_options.include_usage (one standard trailing usage chunk reaches the client)",
    ]
    log(early_read_provenance())
    for d in state.deviations:
        log(f"DEVIATION (declared): {d}")
    # #1361a: ARM BEFORE THE FIRST GROUP STARTS. Everything after this line --
    # P's load, its first sleep, D's load, D's dormant-image write -- is the
    # window the front cannot see, because the front is started further down.
    _launch_guard = None if dry else LaunchGuard(
        log, host_ledger.OBSERVED_REAP_NONRECLAIM_BYTES / host_ledger.GIB).start()
    return _launch_guard, spec_p

def _main_step41__cc():
    """Step of :func:`main` (split behaviour-neutrally, refactor 28.10.)."""
    # #1246 CARRIER CENSUS.  The bound the front routes by is READ from group D's
    # own prefetch-limit instrument ('#915 PREFETCH LIMIT ... site=<one of the two
    # launch-time cache-init sites>', carrier_census.CENSUS_SITES),
    # which names its component and carries the number the runtime ENFORCES.  It
    # is no longer re-derived as 0.9 x min(<a warning that fires for any host
    # pool below its device pool, mamba included>) -- that census had no named
    # population, duplicated the runtime's own fraction, and went silent when the
    # warning did not fire.  Boot weg2rg5 read 19 mamba slots as the KV carrier
    # and shipped a bound of 17: every request took CARRIER-EXCEEDS, group P ran
    # zero prefill passes, exit 0.  Full derivation, provenance and the two
    # rejected alternatives: flliper/srt/pdflip/carrier_census.py module docstring.
    #
    # THE LAUNCHER NEVER SHIPS 0, and no flag makes it.  0 is not an off switch
    # for the round trip: the front's two carrier guards read
    # `carrier_max_tokens > 0` (Front.handle_generate CARRIER-EXCEEDS, Front.leg1
    # the post-leg-1 correction), so 0 removes the BYPASS and routes every prompt
    # above the SHORT grant through leg 1 + a leg-2 store read with NO bound on
    # what the store is asked to carry -- the '#915 PREFETCH REFUSED' / W16
    # shape of boot weg2ls4b2 (84,027 tokens against a 30,518-token host pool).
    # front.main's own help says it in one line.  Fix 1 shipped a
    # --no-carrier-route flag whose help, log line and refusal-remedy sentence
    # all asserted the opposite of that; it is gone.  What replaces it is
    # --carrier-max-tokens N, and FIX 3 gives it the second guard it was missing:
    # N may only LOWER a measured bound (floor < N <= measured), never raise one.
    # Checked against the floor ALONE it could ship a bound ABOVE what D's
    # carrier can read -- 262144, this rig's --max-model-len, was accepted while
    # the launcher held the measured 27466 on the line above -- which is the same
    # W16 the bound exists to prevent, reached through the flag a W45 recommends.
    # With a missing/disagree census there is no measured number to lower, so the
    # override is refused there too: an operator who wants a boot whose census
    # cannot be read fixes the census, not the number.  ONE MECHANISM, in
    # carrier_census.decide_bound, so the two sources of the number cannot drift
    # apart again; every way of failing to MEASURE the bound stays a W45 refusal
    # by name, never a silent zero.
    from flliper.srt.pdflip import carrier_census as _cc
    return _cc
