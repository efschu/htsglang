"""Weg-2 group launcher (spec section 1.5, record sections 1c-1g).

Two STOCK ``python -m flliper.launch_server`` launches per rig, sequenced so
that exactly one group is awake at the end:

    1. NVML resolve (the 5090 by name, the two 3080s), CUDA_VISIBLE_DEVICES
       pinned by UUID: ordinal 0 = 5090, 1 and 2 = the 3080s.
    2. host ledger (#721) for six processes at BOTH moments -> S, M and the
       tmpfs store size; W20 refuses.
    3. the canonical page store on a RAM-backed tmpfs, sized by the ledger.
    4. group P (PP=3 prefill, :30031) launches with per-card budgets derived
       from NVML total minus the corridor minus the EXPECTED dormant residue
       of group D; R1 on :30031; W7/W10 checked from its log; then the
       launcher puts P to sleep and MEASURES D_c(P) per card.
    5. group D (TP=3 + NEXTN decode, :30032) launches with budgets derived
       from the MEASURED D_c(P); R1 on :30032; D stays awake.
    6. the front (:30030) comes up; deadmen armed for all three logs.

Inherited from /spinning/gpu-arb/boot_855_train0901.sh (named per commit):
the cu13 LD_LIBRARY_PATH line (:152-153, plus S1 killer K1's reason), the
#1217 presence sweep (:107-150), the host-ledger preflight call (:315-319),
the env block (:251-255, :283, :329-330, :333, :358), the argv block
(:689-765 minus the removed flip family), the stale-deadman sweep and the
pgrep proof (:782-821), the mem time series (:829-832) and the
current_boot.log symlink (:865).  The Weg-1 phase-flip family, the seam
flags and the flip image env are NOT inherited (S0 removed the flags).

Declared V1 deviations (all printed at launch and listed in the postmortem):
* transports stay OPEN across sleep -- ``barlink_reopen()`` is not wired on
  the wake path in this round; D's BAR1 windows are sized so both groups fit
  the 3080 aperture (24+96 for P, 16+32+40 for D = 208 of 224 MiB usable;
  #1234 C1 raised dcp:0 from 24, measured BAR1 Used 224/256 per 3080).
* FLLIPER_ENABLE_HEALTH_ENDPOINT_GENERATION=0 for both groups (K2): /health
  is a pure liveness probe; the front's /health_generate reaches the AWAKE
  group only.
* the deadmen of the two groups run tier 1 only (TIER2=0, #1158c; PROBE_S
  spaced past the boot); tier 2 (/health_generate) runs on the FRONT's deadman, which the
  front routes to the awake group.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from flliper.srt.environ import envs
from flliper.srt.managers import corridor_guard
from flliper.srt.name_compat import canonical_env, tolerant_compile
from flliper.srt.planner import p_card_chunk as _p_card
# fnFL2 H57: das Power-Limit je Karte (Startzeile, Referenz-Datierung, Raten-Schnitt).
from flliper.srt.planner import power_limit as _power
from flliper.srt.registry import nvml as nvml_registry
from flliper.srt.pdflip import (
    DEFAULT_D_BS,
    DEFAULT_D_BS_NEXTFLASH,
    DEFAULT_D_POOL_WAVES_NEXTFLASH,
    DEFAULT_D_SEAT_VRAM_NEXTFLASH,
    DEFAULT_P_BS,
    DEFAULT_PP_ORDERED_CUT,
)
from flliper.srt.pdflip import pp_ordered_cut_for as pdflip_pp_ordered_cut_for
from flliper.srt.pdflip import admin_key as admin_key_mod
from flliper.srt.pdflip import bar1_windows as bar1_windows_mod
from flliper.srt.pdflip import inventory_view as inventory_view_mod
from flliper.srt.pdflip import budget_rest as _budget_rest
from flliper.srt.pdflip import vote_fixed_live as _vfl
from flliper.srt.pdflip import rank_state as rank_state_mod
from flliper.srt.pdflip import shared_cache_release as shared_cache_release_mod
from flliper.srt.pdflip import state_file as state_file_mod
# PDFLIP-FORM (24.09.): the boot's form axes -- ONE resolver, ONE line, ONE env.
from flliper.srt.pdflip import form as pdflip_form
# X-CURVES 1006: --x-mode / --x-curves (X per request from profile curves).
from flliper.srt.pdflip import x_curves as pdflip_x_curves
# L15 (1.5-stage KV post) is a 27B-line feature: that tree ships
# pdflip/l15_plan.py. On the NF release it lives in front.py and is off by
# default, so this import is absent here; the guards below degrade to the
# L15-off base behaviour (no posts, no L15-POSTS blocker).
try:
    from flliper.srt.pdflip import l15_plan
except ImportError:  # NF line: no L1.5 planner module
    l15_plan = None

from flliper.srt.pdflip.launcher_refuse_proc_leaf import (  # noqa: F401 -- re-export, behaviour-neutral move
    BOOT_TOKEN_ENV_KEYS,
    DC_RESERVE_SLACK_MIB,
    DC_RESERVE_SLACK_NCCL_MIB,
    DRAFT_ON_P_CONFLICT,
    DRAFT_ON_P_ENV,
    DUPLEX_GATE,
    D_TOKEN_CUT_FLAGS,
    D_TOKEN_CUT_PROVENANCE,
    MM_SKIP_HASH_ENV,
    PP_LAYER_SET_ENV,
    PROFILE_QWEN27B,
    PdFlipXchgFormSourceMissing,
    SHORT_DRAIN_ABOVE_X_NAME,
    VISION_TRANSIENT_OVERRIDE,
    WEIGHT_SOURCE_ARMED,
    WEIGHT_SOURCE_CHOICES,
    WEIGHT_SOURCE_DEFAULT,
    X_BUSY_NEGATIVE_NAME,
    _BUBBLE_RE,
    _CARD_POWER_RE,
    _KEY_PROBE,
    _PREFILL_RE,
    _PROFILE_27B,
    _XCHG_CENSUS_MAP,
    _proc_argv,
    _proc_boot_tag,
    _proc_cmdline,
    _proc_has_cuda_context,
    _proc_ppid,
    _proc_start_ticks,
    _profile_early_read,
    _profile_row_or_27b,
    _published_profile,
    _rank_vec,
    _read_driver_text,
    _refuse_if_extra_drops_transient_lmo,
    _refuse_if_extra_raises_budget,
    _refuse_if_transient_mm_hash_unstable,
    _refuse_unbuilt_platztausch_buffers,
    _refuse_value_once,
    _resolve_keys_in_rank_tree,
    _rope_cache_cols,
    _row_format_default,
    _running_power_by_class,
    _same_vector,
    _split_decisions,
    _split_fraction_text,
    pick_d_residue_record,
    pinned_reserve_line,
    pool_floor_line,
    pp_cut_chunk_mix,
    profile_arg_defaults_line,
    publish_d_token_cut,
    publish_p_fractions,
    publish_store_identity,
    rank_wave_floor,
    read_arc_state,
    read_card_power_record,
    read_mean_prefill_prefix,
    read_pp_bubble,
    record_card_power,
    records_inventory,
    refuse_double_coverage_arm,
    refuse_inherited_layer_set,
    refuse_short_drain_above_x,
    refuse_unless_same_form_source,
    refuse_unwired_token_cut,
    repoint_store_geometry_at_pinned_form,
    reserve_slack_mib,
    resolve_draft_on_p,
    resolve_pool_floor,
    resolve_x_ceiling,
    resolve_x_mode_launch,
    ring_absent_by_design,
    select_cards,
    served_model_name,
    session_pids,
    set_group_env,
    set_xchg_census_map,
    shipped_line,
    shm_holder_pids,
    spec_form_is_dflash,
    split_preflight_refusals,
    standard_form_resolved,
    store_extra_config,
)


from flliper.srt.pdflip.launcher_p_l3_leaf import (  # noqa: F401 -- re-export, behaviour-neutral move
    CHUNKED_PREFILL_TOKENS,
    CODE_CARDS,
    D_KV_STAGE_KEYS,
    D_KV_STAGE_LRU_FLOOR_MARKER,
    EXPERT_MAP_DIR_ENV,
    FP4_NATIVE_MIXED_ARGV,
    FP4_UNIFORM_MARLIN_ARGV,
    FP8_UNIFORM_MARLIN_ENV,
    L3_IDENTITY_FILE,
    L3_PERSIST_PREFIX,
    L3_REVOKED_FILE,
    L3_REVOKED_SUFFIX,
    L3_SUFFIX_REGISTRY,
    LAUNCH_SERVER_MODULE,
    LAUNCH_SERVER_MODULES,
    P_CHUNK_BUILTIN_FP8,
    P_CHUNK_BUILTIN_INT8,
    P_CHUNK_BUILTIN_SOURCES,
    P_CHUNK_MAX_DEFAULT,
    P_CHUNK_MODEL_DEFAULT,
    P_CHUNK_NF_MODEL_DEFAULT,
    P_CHUNK_POLICY_DEFAULT,
    P_DEEP_SPLIT_FROM_DEFAULT,
    P_GC_FREEZE_DEFAULT,
    P_LAYER_SPLIT_DEFAULT,
    P_MAMBA_FLOOR_SLOTS_PER_SEAT,
    P_PREFILL_GRAPH_DEFAULT,
    P_PREFILL_GRAPH_POLICY_DEFAULT,
    P_PREFILL_TRANSIENT_CALIBRATION_MODELS,
    P_STAGE_CARD_CLASSES,
    UNIFORM_MARLIN_QUANT_METHODS,
    WEIGHT_SOURCE_EXCHANGE,
    _CALIBRATION_FOOTPRINT_ALIAS,
    _P_CHUNK,
    _P_CHUNK_BUILTIN_RECORD,
    _P_CHUNK_BUILTIN_RECORD_FP8,
    _P_LAYER_SPLIT,
    _P_LOG_MODEL_RE,
    _P_PREFILL_GRAPH,
    _fmt_gib,
    _forced_record_borrow,
    _free_instrument,
    _front_leg_form,
    _gcd_reduce,
    _gib,
    _group_env_value,
    _group_fact_value,
    _import_duplex_gate,
    _infeasible_breakdown,
    _int_list,
    _l3_extra_flag,
    _l3_move_revoked,
    _l3_revoked_path,
    _l3_rope_view,
    _last_flag_value,
    _live_stage_support,
    _median,
    _mps_pids_on_pipe,
    _now,
    _owned_miss_min_paired,
    _p_chunk_dynamic_runnable,
    _p_chunk_interp,
    _p_chunk_nf_args,
    _pconst,
    _pconst_boots,
    expert_map_dir,
    flip_order_split,
    flip_weights_tags,
    floor_source_for_shipped_line,
    forced_lane_borrow,
    fork_anchor_env,
    fp4_native_mixed_env,
    fp4_native_mixed_refusal,
    fp8_layout_decision,
    front_launch_fields,
    fs_of,
    gguf_backbone_depth,
    group_form_block,
    group_rank_count,
    hicache_disabled_line,
    incumbent_candidate,
    is_launch_server_argv,
    kv_stage_cut_workers,
    kv_stage_lru_floor,
    kv_stage_wave_cap,
    l3_persist_check_identity,
    l3_persist_dir_name,
    l3_persist_enabled,
    l3_persistent_store,
    l3_registered_suffixes,
    l3_weights_fingerprint,
    log_power_limits,
    memts_owner_pattern,
    model_config_path,
    model_is_gguf_file,
    n_stage_incumbent,
    note_calibration_footprint_alias,
    nvml_memory,
    nvml_process_mib,
    operator_kv_stage_max,
    order_cards,
    p_attn_head_split_env,
    p_attn_head_split_line,
    p_card_fr_cap,
    p_chunk_model_cut,
    p_chunk_model_power_limits,
    p_chunk_policy_spec,
    p_chunk_power_lines,
    p_corridor_holdback_default_mib,
    p_deep_split_env,
    p_gc_env,
    p_host_overlap_env,
    p_layer_split_env,
    p_layer_split_slab_lines,
    p_layer_split_swing_by_stage,
    p_log_model_path,
    p_mamba_floor_text,
    p_mamba_mib_per_slot_by_stage,
    p_pool_posts_as_booked,
    p_prefill_graph_bucket,
    p_prefill_graph_calibration_of,
    p_prefill_graph_captured,
    p_prefill_graph_policy_lines,
    p_prefill_graph_tiny_of,
    p_recv_stream_env,
    p_trim_end_anchor_env,
    parse_card_selection,
    parse_group_env,
    parse_user_reserve,
    pcie_lanes,
    pdflip_boot_token,
    pdflip_weights_cpu_backup_ring_kw,
    per_pair_crossing_ms,
)
# Task #58: the arming variable's name comes from the module that READS it, so
# the publisher and the reader cannot drift into two spellings of one key.
from flliper.srt.pdflip.vision_stage_boot import VISION_ENV as VISION_STAGE_ENV
from flliper.srt.pdflip import topology as _topo
from flliper.srt.pdflip import (
    checkpoint_census,
    corridor_budget,
    host_ledger,
    l15_plan,
    l15_pool_peak,
    card_identity,
    refusals,
    ring_table,
    seam_digest,
    weight_exchange,
    weight_exchange_region,
    weight_exchange_shadow,
    weight_exchange_transport,
    xchg_bounce,
    xchg_manifest,
    xchg_residency,
)

from flliper.srt.pdflip.launcher_d_p_r2_L0 import (  # noqa: F401 -- re-export, behaviour-neutral move
    COLD_TIER_MARKER,
    COLD_TIER_SHM_FS,
    CORE_CWD_ENV,
    DC_MEASURED_D_BY_CLASS,
    DEVTOOLS_DIR,
    EVIDENCE_DIR,
    GPU_ARB,
    HOST_PREFLIGHT,
    L3_CAP_FILE,
    MIB,
    P_ACTIVATION_RECORD,
    P_CHUNKED_PREFILL_TOKENS,
    P_DRAFT_BUILD_ACCOUNTING_TOL_MIB,
    P_DRAFT_RESIDENT_TOL_MIB,
    P_PP_ATTN_STAGE_RATIO_SCORES,
    P_PP_STAGE_FIXED_REF_NONLAYER_MIB,
    P_PP_STAGE_RATIO_SCORES,
    P_PREFILL_TRANSIENT_SUPPORT,
    P_STAGE_FIXED_YARN2_DELTA_RECORD,
    SPEC_NUM_DRAFT_TOKENS,
    STORE_CENSUS_DRAFT_PAGES,
    STORE_CENSUS_KV_PAGES,
    STORE_CENSUS_KV_PAGE_BYTES,
    STORE_CENSUS_MAMBA_BLOBS,
    STORE_CENSUS_PROVENANCE,
    STORE_DISK_MIN_FREE_GIB,
    STORE_MAMBA_BLOB_BYTES,
    STORE_ROOT,
    STORE_ROOT_TOLD,
    STORE_SIDECAR_FACTOR,
    _CARD_FREE_RE,
    _CENSUS_DRAFT_BYTES,
    _CENSUS_KV_BYTES,
    _CENSUS_MAMBA_BYTES,
    _DEFAULT_POOL_RE,
    _HEAD_RELEASED_RE,
    _NVML_DELTA_RE,
    _OTHER_LIVE_RE,
    _OUTSIDE_TORCH_RE,
    _RESIDENT_RE,
    _STATE_MISSING_LOGGED,
    _TAG_POOL_INACTIVE_RE,
    _alive_not_zombie,
    _l3_attach_from_index,
    _p_chunk_nf_limits,
    apply_profile_d_bs_default,
    apply_profile_d_pool_waves_default,
    apply_profile_d_seat_vram_default,
    arc_preflight_line,
    boot_state_write,
    build_tms_preload,
    calib_identity_of,
    cards_free_check,
    check_draft_resident,
    cold_tier_shm_post,
    core_cwd_for,
    d_activation_record,
    d_admit_need_record,
    d_awake_rest,
    d_extend_growth_per_row_record,
    d_extend_growth_record,
    d_fixed_record,
    d_overshoot_record,
    d_pool_peak_record,
    d_verify_window,
    dc_measured_d_mib,
    derived_family_cost,
    helper_registry_dir,
    host_preflight,
    hostsample_csv_path,
    l3_persist_disk_sum,
    l3_revoke_window,
    lane_coverage_dump_dir,
    load_xchg_census_for_cards,
    log_boot_state,
    measured_record_path,
    memts_csv_path,
    owned_miss_record_root,
    p_activation_record,
    p_chunked_prefill_tokens,
    p_cut_calibration_line,
    p_draft_kv_cell_bytes_by_stage,
    p_group_has_draft_producer,
    p_produces_draft_pages,
    p_stage_fixed_ctx_delta_record,
    p_stage_fixed_for_checkpoint,
    p_stage_layers,
    plan_store,
)
PORT_FRONT = 30030
# User order 2026-09-10: the front serves on the LAN, permanently. 0.0.0.0
# covers loopback (router, deadmen, agents) AND the LAN address. The groups
# stay on 127.0.0.1 -- only the front is a client surface. Loopback trust in
# utils/auth.py keys on the PEER address, so an admin route reached from the
# LAN still needs the bearer; the API routes need nothing.
DEFAULT_FRONT_HOST = "0.0.0.0"
from flliper.srt.pdflip.launcher_d_p_r2_L2 import (  # noqa: F401 -- re-export, behaviour-neutral move
    CARD_POWER_TOLERANCE_W,
    DEADMAN,
    GROUP_DEADMEN_TIER1_ONLY,
    MEMTS_OWNER_POLL_S,
    PORT_P,
    P_MAX_MAMBA_CACHE_SIZE,
    _ARENA_BOOKED_WIDTHS,
    _MEMTS_SUPERVISOR_SH,
    _pdflip_arena_ledger_terms,
    argv_p,
    arm_deadman,
    canonical_state_gate,
    d_extend_vote_fixed_live_form,
    d_replayssm_spec_flags,
    d_replayssm_spec_line,
    d_replayssm_spec_plan_form,
    hicache_draft_tier_env,
    hicache_draft_tier_line,
    p_prefill_transient_vector_mib,
    register_boot_group,
    spec_plan_fields,
    stage_fit_family_cost,
    start_memts,
    stop_memts,
    w10_skip_no_producer_line,
)
from flliper.srt.pdflip.launcher_p_d_r2_L3 import (  # noqa: F401 -- re-export, behaviour-neutral move
    D_EXTEND_CAP_RATE_RECORD,
    PORT_D,
    P_PREFILL_ACTIVATION_RESERVE_MIB,
    P_PREFILL_TRANSIENT_ENV,
    RING_FORM_SENTINEL_STORE_CFG,
    _p_page_size,
    argv_d,
    d_extend_vote_fixed_live_verdict,
    d_plan_inputs,
    p_prefill_activation_reserve_by_stage_mib,
    p_prefill_activation_reserve_mib,
    p_prefill_transient_for,
)
from flliper.srt.compat_shims import canonical_flags as _canonical_flags, env_name_variants as _env_name_variants, name_counterparts as _name_counterparts, name_variants as _name_variants, operator_dir as _operator_dir, register_flag_aliases as _register_flag_aliases  # noqa: E402  rename transition
#: The step-0 metal probe's RECORD (C0, WEG2_BUILD_DECISIONS_0906 section 1p).
#: A FILE, not a number: the per-card duplex ratios C12/C13 gate on are parsed
#: out of its own measured rows and printed with this path beside them.
DUPLEX_PROBE_DEFAULT = _operator_dir(GPU_ARB, "PROBE_RING_0907.md")
from flliper.srt.pdflip.launcher_d_p_r2_L1 import (  # noqa: F401 -- re-export, behaviour-neutral move
    ARMING_FLOOR_MIB,
    BARLINK_BAR1_CAP_CYCLES,
    COLLECTIVE_CENSUS_INTERVAL,
    MEMTS,
    OWNED_MISS_RECORD_ENV,
    PDFLIP_CARD_COUNT,
    P_ACTIVATION_RECORD_CHUNK,
    P_CORRIDOR_TOP_MIB,
    RANDOM_SEED,
    W11B_FREE_OVER_UNACCOUNTED,
    XCHG_RANK_BEHAVIOUR_WIRED,
    XCHG_REGION_HEADER_MIB,
    XCHG_REGION_SLOTS_PER_PAIR,
    XCHG_REGION_SLOT_MIB,
    XCHG_UNWIRED_REASON,
    _VFL_ARGV,
    _apply_p_chunk_policy_nf,
    _env_knobs,
    _installed_max_share_from_record,
    apply_owned_miss_record_default,
    booked_rest_admit_deltas,
    calib_log_accept_of,
    common_flags,
    d_early_stage0_watch,
    d_extend_rate_fill_null,
    d_extend_vote_fixed_live_material,
    d_owned_miss_ms,
    d_replayssm_spec_ring_len,
    d_window_pool_fields,
    gate_w11,
    l3_persist_attach,
    launch_group,
    memts_pgrep_pattern,
    p_chunk_policy_install,
    p_dormant_records,
    p_draft_kv_flags,
    p_prefill_graph_line,
    p_transient_support,
    prepare_weight_exchange,
)
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import (  # noqa: F401 -- re-export, behaviour-neutral move
    DEPTH_DERIVATION_GROUNDED,
    HOST_RING_DIR,
    S104_SCHED_KILL_S,
    SHM_DIR,
    SPEC_ALGORITHM,
    SPEC_EAGLE_TOPK,
    SPEC_NUM_STEPS,
    STORE_DISK_READ_MBPS,
    STORE_DISK_WRITE_MBPS,
    _stop_helper_pid,
    _vfl_effective_draft,
    prepare_host_ring,
    prepare_store,
    prepare_xchg_region,
    rank_state_dir_for,
    record_line_spec,
    refuse_if_front_unbindable,
    resolve_x,
    s104_escalate_schedulers,
    s104_scheduler_pids,
    served_dormant_growth,
    solve_p_depth,
    spec_flags,
    spec_form_env,
    stale_deadman_sweep,
    state_path,
    store_read_cost_line,
    store_sidecar_factor_of,
    sweep_dead_credit_counters,
    sweep_store_residue,
    sweep_xchg_semaphores,
    teardown_xchg_region,
    user_reserve_env_entries,
    write_p_respawn_spec,
)
PRESENCE_DIR = f"{SHM_DIR}/flliper-phase-flip-presence"
#: #1390: named exactly like `SHM_DIR` above, and for the same reason --
#: `main`'s ledger call must reference these BY NAME at its own call site
#: (`meminfo_path=MEMINFO_PATH`, not the literal default `choose_host_ledger`
#: already carries) so a test's `mock.patch.object(launcher, "MEMINFO_PATH",
#: fake)` is actually read: a default parameter value is bound ONCE, at
#: `choose_host_ledger`'s own def time, and a call with no keyword at all
#: would keep reading the ORIGINAL literal forever, exactly as `shm_dir`
#: silently did until this ticket.
MEMINFO_PATH = "/proc/meminfo"
CGROUP_ROOT = "/sys/fs/cgroup"
SHM_ARCHIVE_ROOT = f"{GPU_ARB}/shm_residue"
#: #1233 fix 8, boot weg2dk7: the /dev/shm NAME FAMILIES THIS LINE'S OWN CODE
#: CREATES, each traced to the file that writes it.  A name outside this tuple
#: is FOREIGN and is never touched, whatever it looks like -- /dev/shm on this
#: box carries hundreds of `sem.mp-*` files belonging to other people's Python
#: processes, and a sweep that guessed would eat them.
#:
#: WHY THE SWEEP EXISTS: weg2dk7 measured 3.23 GiB of host RAM held by
#: `/dev/shm/hicache-weg2-fix973`, the page store of a boot that had been dead
#: for ~16 h (BOOT_weg2fix973_0907.md, whose own teardown line claims the
#: directory was removed).  Nothing mapped it, and every host reading of that
#: boot -- the ledger's `memory.current`, the shmem attribution, the corridor
#: -- counted it as occupied.  The #1217 check saw none of it: it looked only
#: inside the presence directory.
SHM_OWN_PREFIXES = (
    "flliper-phase-flip-presence",   # managers/phase_flip_presence.py, #1217
    "hicache-weg2-",                # the canonical page store when it is put on /dev/shm
    ".pdflip-pcie-serialize-",        # model_loader/hibernate.py:504
    "flliper_loads_",                # managers/load_snapshot.py:285
    # #1273 S3: the weight-exchange staging region (385 MiB) and the 24 named
    # POSIX semaphores glibc materialises as `/dev/shm/sem.<name>`. Both
    # families are created by pdflip/weight_exchange_region.py. Without these two
    # entries a crashed exchange boot leaves 385 MiB of tmpfs resident and
    # INVISIBLE to this sweep -- exactly the weg2dk7 shape (3.23 GiB of a boot
    # dead for ~16 h, counted by the ledger as occupied) that the sweep exists
    # for. `sem.` is a SEPARATE entry because the semaphore names do not begin
    # with the region prefix; a prefix list that assumed they did would sweep
    # the region and leave the handshake behind.
    weight_exchange_region.REGION_PREFIX,           # "pdflip-xchg-"
    # #1378 xsn72: the sequential transport's per-boot lane buffers
    # (`/dev/shm/pdflip-seq-<nonce>/<lane>_unit_buffer.bin`, up to ~2 GiB each,
    # weight_exchange_bounce.run_sequential_units). The collect side unlinks
    # them when a tag is drained; a boot that dies mid-leg leaves the whole
    # set. MEASURED 2026-09-15 after six such boots: 20 GiB of tmpfs resident,
    # 55 GB "in use" on a box running nothing -- the user saw it before the
    # sweep could, because the prefix was not here. An orphan directory tree
    # is archived as a manifest and removed (see shm_residue_sweep); a live
    # holder refuses the boot as for every other entry.
    "pdflip-seq-",
    f"sem.{weight_exchange_region.REGION_PREFIX}",  # "sem.pdflip-xchg-"
    # rc30 visiontail (A4): the tower rows P publishes for D (`/dev/shm/pdflip-vrows-<tower id>`,
    # pdflip/vision_row_store.py, bounded at FLLIPER_PDFLIP_VISION_ROWS_MB). Content-keyed and valid across boots,
    # but a dead line's residue is swept and archived as a manifest like every other family here.
    "pdflip-vrows-",
    # fnFL2x16 (2026-09-23): the shared HiCache page arena of every boot
    # (`/dev/shm/pdflip-arena-<tag>`, FLLIPER_HICACHE_ARENA_DIR below, one
    # directory per boot tag). MEASURED after a 259k needle: 4 GiB of tmpfs,
    # left behind by the dead boot and charged by the next boot's W22 as
    # occupied. The tag is new every boot, so an arena is never reopened.
    "pdflip-arena-",
)
#: Rename transition (compat_shims): the residue of a boot that ran under the other package /
#: subsystem name is this line's own too, so the sweep also lists the other spelling of every family.
SHM_OWN_PREFIXES = SHM_OWN_PREFIXES + _name_counterparts(SHM_OWN_PREFIXES)
from flliper.srt.pdflip.launcher_d_p_r2_L6 import (  # noqa: F401 -- re-export, behaviour-neutral move
    D_AWAKE_OVERSHOOT_MIB,
    booked_rest_kwargs,
)
#: Kept as the name the pre-#1257c budget line printed, and as the value the
#: fallback still produces. NEVER read as a floor: the floor is per card now.
CORRIDOR_MIB = 1024 + D_AWAKE_OVERSHOOT_MIB
from flliper.srt.pdflip.launcher_d_leaf import (  # noqa: F401 -- re-export, behaviour-neutral move
    ARENA_FROM_LEDGER_FLOOR_GIB,
    CARD_POWER_MARKER,
    DC_EXPECT_3080_MIB,
    DC_EXPECT_5090_MIB,
    DC_EXPECT_MIB_BY_CLASS,
    DC_MAX_MIN_SLEEPS,
    DC_ONE_SLEEP_MARK,
    DC_RECORD_ENV,
    DC_RECORD_MARGIN_MIB,
    DFLASH_BLOCK_DEFAULT,
    DFLASH_DRAFT_PATH_DEFAULT,
    DFLASH_PRODUCE_ON_P_DEFAULT,
    DFLASH_WINDOW_DEFAULT,
    DUAL_P_BARLINK_BAR1_WINDOW_MIB,
    D_BARLINK_BAR1_WINDOW_MIB,
    D_GC_FREEZE_DEFAULT,
    D_KV_EVICT_DEFAULT,
    D_KV_STAGE_BY_DEMAND_KEY,
    D_KV_STAGE_FLOOR_KEY,
    D_REPLAYSSM_SPEC_DEFAULT,
    D_RESERVE_RECORD_ENV,
    D_RESHARD_DEFAULT,
    D_TP_OBJECTIVE_DEFAULT,
    EARLY_READ_FACTS,
    ENV_DFLASH_PLACEMENT,
    FR_D_CAP_ENV,
    FR_D_CEILING_ENV,
    FR_D_CEILING_MARKER,
    KV_TOKEN_CUT_MARKER,
    P_BARLINK_BAR1_WINDOW_MIB,
    P_COVERS_D_SESSION_MARK,
    P_DORMANT_EXPECT_N,
    RATE_UNIT_GROUP_THROUGHPUT,
    RETRACTED_SEED_VECTOR_1032,
    SPEC_FORM_DEFAULT,
    TORCH_CACHE_CAP_ENV,
    VISION_CHOICES,
    VISION_DEFAULT_MARKER,
    VISION_OFF,
    VISION_RESIDENT,
    VISION_TRANSIENT,
    _DRAFTER_RE,
    _D_KV_EVICT,
    _D_RESHARD,
    _D_TOKEN_PLACEMENT,
    _LAYOUT_RE,
    _SPEC_FORM,
    _adopt_load_format_flag,
    _alive,
    _argv_int_list,
    _argv_scalar,
    _argv_vector,
    _arm_run_peak_gib,
    _attn_axis_for,
    _axis_floor_ranks,
    _cards_empty,
    _credit_counter_rows,
    _csv,
    _csv_floats,
    _csv_ints,
    _d_adopt_armed,
    _d_argv_of_front_log,
    _d_reshard_seg_placement,
    _derive_csv,
    _drop_group_env,
    admin_key_flag,
    apply_bar1_windows,
    apply_d_kv_evict,
    apply_profile_vision_default,
    arena_from_ledger,
    boot_state_dir,
    bs_source,
    budget_charges_driver_carve,
    budget_rest_from_records,
    build_expert_karte,
    card_power_limits_w,
    card_power_line,
    carrier_covers_d_session,
    census_entry_for_pricing,
    check_drafter_identity,
    chunked_prefill_size_of,
    configure_xchg_region,
    count_marker,
    d_card_ledger,
    d_card_vote_lines,
    d_expect_check_lines,
    d_expect_from_p_records,
    d_extend_trim_env,
    d_fr_ceiling_adopt,
    d_gc_env,
    d_hold_active,
    d_kv_evict_env,
    d_kv_stage_by_demand,
    d_kv_stage_floor_tokens,
    d_kv_token_cut,
    d_lend_floor_env,
    d_only_bind_argv,
    d_operating_point_line,
    d_owned_extend_h2d,
    d_owned_heat_records,
    d_owned_miss_ms_rank,
    d_replayssm_spec,
    d_reserve_from_records_or_census,
    d_reshard_argv,
    d_reshard_depth,
    d_reshard_env,
    d_reshard_lines,
    d_residue_capture_bs,
    d_row_attn_units,
    d_scratch_cap,
    d_seat_graph_mib,
    d_seat_table_form,
    d_seat_vram_armed,
    d_token_cut_vector,
    d_token_placement_env,
    d_token_vector_decision,
    d_verdict_measured_terms,
    dc_expect_mib,
    dc_residue_from_record,
    dc_residue_max_from_records,
    derive_x_star,
    dflash_placement,
    dflash_produce_on_p,
    dormant_max_from_records,
    draft_kv_off_line,
    draft_on_p_line,
    driver_carve_charged,
    driver_carve_min_total_mib,
    early_read_flags,
    early_read_provenance,
    effective_reserve_by_card,
    er_trim_cell,
)


#: MEASURED on boot weg2ls1b2 (2026-09-07 07:11:55-58Z, NVML per-process at
#: group D's first sleep, tags kv_cache+weights, NEXTN draft + TP decode
#: graphs resident): D_c(D) = 2228 MiB on the 5090, 1922 MiB on each 3080 --
#: ABOVE the spec 1.6 expectations by 380 / 480 MiB.  These carry the
#: provenance into P's budget until a boot measures them again; the front's
#: W19 still grades the live measurement against the reserve actually used.
#: SCOPE OF ALL FOUR NUMBERS ABOVE (FIX 1r/1): every boot behind them -- the
#: spec 1.6 expectations and these two measurements -- ran group D with
#: `--disable-overlap-schedule`, hence `no_buffer`, hence ZERO ping-pong mamba
#: state slots. D now runs the overlap schedule, so its device residue carries
#: `d_mamba_ping_pong_cost` extra state slots per rank that no boot behind these
#: constants contained. They are NOT corrected here: a MiB conversion would be a
#: second accounting of the runtime's own sizing (see `d_overlap_cost_line`).
#: The term is printed at boot beside the SCHEDULER line, and W19 grades the
#: live residue rather than these expectations.
DC_MEASURED_D_5090_MIB = _PROFILE_27B.constant("DC_MEASURED_D_5090_MIB")
DC_MEASURED_D_3080_MIB = _PROFILE_27B.constant("DC_MEASURED_D_3080_MIB")
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
DC_MEASURED_D_XCHG_3080_MIB = DC_MEASURED_D_XCHG_MIB[0]
DC_MEASURED_D_XCHG_5090_MIB = DC_MEASURED_D_XCHG_MIB[1]






from flliper.srt.pdflip.launcher_d_mid import (  # noqa: F401 -- re-export, behaviour-neutral move
    ATTN_ANCHOR_CARD_CLASS,
    ATTN_ANCHOR_MS,
    CONTEXT_LENGTH_TOKENS,
    DFLASH_PRODUCE_ENV,
    D_RANK_SOLVE_MARKER,
    D_TOKEN_PLACEMENT_DEFAULT,
    EXTEND_CAP_FLOOR_ENV,
    EXTEND_CAP_FLOOR_MARKER,
    EXTEND_RATE_MARKER,
    EXTEND_RATE_MEASURE_ENV,
    EXTEND_RATE_MEASURE_RANKS_ENV,
    FORM_A_KV_WORKER_CANONICAL_MARKER,
    FORM_A_WORKER_CANONICAL_MARKER,
    GROUP_SWITCH_DEFAULTS_MARKER,
    ONCARD_MODE_DEFAULT,
    W10_SKIPPED_NO_DRAFT_PAGES,
    _CENSUS_SOURCE_BOOT_RE,
    _axis_renormalisation,
    _d_extend_cap_floor_env,
    _d_extend_rate_measure_env,
    _d_layout_is_tp_symmetric,
    _group_rank_count,
    apply_d_seat_expert_rows,
    apply_d_token_placement,
    apply_p_draft_post,
    apply_p_gc_warn_default,
    apply_profile_group_switch_defaults,
    apply_profile_torch_cache_cap_default,
    apply_spec_form,
    arena_gib_into_groups,
    attn_anchor_stage,
    calib_power_limits,
    calibrated_inventory,
    canonical_marker_counts,
    census_checkpoint,
    checkpoint_quant_method,
    d_draft_share_embed,
    d_early_start_proven,
    d_effective_seats,
    d_extend_rate_measure_ranks_env,
    d_flip_trim_lines,
    d_form_solved_by_planner,
    d_kv_dtype_bytes,
    d_kv_stage_tokens_named,
    d_kv_worker_ranks,
    d_moe_residency,
    d_park_split_refusal,
    d_residue_record_accept,
    d_residue_record_for,
    d_seat_vram_plan_form,
    d_session_tokens,
    dc_residue_for,
    dflash_produce_line,
    expert_map_vectors,
    expert_store_dir_of,
    gate_w10,
    group_context_tokens,
    host_price_is_region_form,
    host_priced_pick,
    kv_stage_wave_floor,
    l15_mib_by_card_identity,
)












P_DORMANT_EXPECT_ENV = "FLLIPER_PDFLIP_D_EXPECT_P_RECORD"






from flliper.srt.pdflip.launcher_p_mid import (  # noqa: F401 -- re-export, behaviour-neutral move
    D_FR_ENV,
    D_OWNER_MARKER,
    GGUF_DENSE_VOCAB_ENV,
    KV_CACHE_DTYPE,
    L3_ANCHOR_KEYING_PROFILES,
    L3_PERSIST_GENERATION,
    L3_ROPE_APPLY,
    MAMBA_ARENA_SLOTS_PER_REQUEST_UPPER,
    P_CHUNK_CUT_TAG,
    P_CHUNK_DRY_RUN_TOKENS,
    P_CHUNK_GRAPH_FIXED_MS,
    P_MAMBA_AUTO_SAFETY_MARGIN,
    P_MICRO_BATCH_FLAG,
    _CAUSE_RE,
    _RE_DRAIN,
    _RE_FLIP,
    _RE_LEG1,
    _RE_LEG2,
    _model_config,
    _p_chunk_stage,
    _p_gguf_dense_vocab,
    _proc_ancestors,
    _restage_p_card_reference,
    _serialised_cards,
    l3_persist_check_rope,
    l3_persist_identity,
    l3_revoked_windows,
    lane_coverage_boot_token,
    mamba_arena_group_floor,
    measure_x_inputs,
    newest_bubble_log,
    newest_prefill_census_log,
    p_chunk_model_cut_check,
    p_chunk_policy_env,
    p_chunk_policy_lines,
    p_covers_d_session,
    p_dormant_from_records,
    p_host_overlap_lines,
    p_micro_batch_flags,
    p_micro_batch_line,
    p_prefill_graph_buckets,
    p_recv_stream_lines,
    p_seats,
    pick_shipped_cut,
    pin_d_form_for_map,
    prefix_switches_announce,
    profile_standard_form,
    publish_d_owner_ratio,
    refusal_cause,
    refuse_d_early_start_unreviewed,
    refuse_d_form_off_the_map,
    region_form_lane_price_cap,
    remove_vram_credit_counters,
)






def d_expect_dormant_other(
    cards: Sequence[Card], dc_expect_d: Mapping[str, int],
    p_recs: Optional[Sequence[Dict[str, object]]], weight_source: str, profile: Optional[str],
) -> Tuple[Dict[str, int], str]:
    """``dormant_other`` of group D's EXPECTATION budget (the map pass and the
    dry pass, both before P has slept): what group P leaves on each card
    while D runs.

    Before (and still on every profile whose D residue is not census-priced,
    i.e. the 27B, whose dormant_other IS its resident draft): D's own dormant
    reserve moved by the window difference, ``dc_expect_d + P_WINDOWS_MIB -
    D_WINDOWS_MIB`` -- D's residue standing in for P's. rc12z30r3 (29.09.,
    Next Flash, token cut): that booked 2054/1188/1586 MiB (nvml1/0/2) where
    P's measured residue was 1252-1326/646-718/630-980; the map pass solved
    the D form (S3f ownership, FR_D, scratch) 540-960 MiB per card short, and
    the real pass could not raise it any more (the map is built from it), so
    the gap sat free all run.

    A profile whose row sets ``d_expect_from_p_records`` (Next Flash; the
    27B since 29.09. -- its legacy term booked ~2.1 GiB on the 5090 against
    a measured P residue of 1104-1240 MiB) with same-identity group-P
    records: the maximum of P's measured residue over the newest
    ``P_DORMANT_EXPECT_N`` boots (:func:`p_dormant_from_records`). Anything
    else: the legacy term, with the reason named. The switch is its own
    registry field, not ``d_residue_census`` (the reserve question).
    ``FLLIPER_PDFLIP_D_EXPECT_P_RECORD=0`` forces the legacy term.

    DANGER DIRECTION (P holds more after serving than the records say): the
    real pass prices the launcher's own reading after P's sleep, and the
    pinned form is CHECKED against that budget in ``log_d_rank_vram_solve``
    -- the scratch is capped to the planner's edge (named), and a form that
    still does not fit is refused by name (W122 ``PdFlipLaunchRefused``), never
    started into an OOM. ``D-EXPECT CHECK`` names each card's difference."""
    legacy = {c.uuid: int(dc_expect_d[c.uuid]) + P_WINDOWS_MIB - D_WINDOWS_MIB for c in cards}
    if not d_expect_from_p_records(profile):
        why = "legacy (profile's dormant_other is D's reserve + windows)"
    elif os.environ.get(P_DORMANT_EXPECT_ENV, "1").strip() == "0":
        why = f"legacy ({P_DORMANT_EXPECT_ENV}=0)"
    else:
        meas, why = p_dormant_from_records(p_recs, cards, weight_source)
        if meas is not None:
            vram_view().note_asleep("P", cards, meas, f"RECORD({why})")
            return meas, why
        why = f"legacy (UNMEASURED: {why})"
    vram_view().note_asleep("P", cards, legacy, f"MODEL(dc_expect_d + windows; {why})")
    return legacy, why








from flliper.srt.pdflip.launcher_xchg_vision_leaf import (  # noqa: F401 -- re-export, behaviour-neutral move
    BARLINK_FLAGS_BARE,
    BARLINK_FLAGS_WITH_VALUE,
    STORE_SHORT_TAIL_X_ENV,
    VISION_PLACE_AUTO,
    VISION_SOURCE_DISK,
    VISION_SOURCE_PINNED,
    VISION_SOURCE_RAM,
    XCHG_D_VECTOR_DEFAULT,
    _TOKENIZER_PATH,
    _W_CODE_RE,
    _tree_bytes,
    _vector_len,
    _w_code_of,
    _without_budget_flag,
    store_short_tail_env,
    strip_barlink_flags,
    strip_speculative_flags,
    tokenizer_path_flags,
    turn_anchor_env,
    uniform_marlin_argv,
    user_reserve_env_value,
    vision_model_flags,
    vision_on_d_of,
    vision_source_host_line,
    vision_stage_knobs_refusal,
    vote_fixed_live_root,
    w38_armed_line,
    with_uniform_marlin_argv,
    xchg_bounce_arm_pins_host,
    xchg_census_is_reserve,
    xchg_form_argv,
    xchg_lane_count,
    xchg_region_pairs,
    yarn_context_tokens,
)
from flliper.srt.pdflip.launcher_d_p_r2_L4 import (  # noqa: F401 -- re-export, behaviour-neutral move
    BARLINK_BUILD_WINDOW_CAP_S,
    BARLINK_ENV_KEYS,
    L2NORM_RUNTIME_T_ENV,
    MATCH_REFUSAL_CENSUS_EVERY,
    PP_CHAIN_RECV_STALL_S,
    PP_OCCUPANT_HORIZON_S,
    VISION_D_ENV,
    VISION_PLACE_ENV,
    VISION_SOURCE_ENV,
    _KV_POOL_SIZING_RE,
    _d_extend_vote_fixed_live_env,
    _max_running_requests,
    build_env,
    corridor_recipe_probe,
    d_extend_vote_fixed_record,
    d_overhead_calibration,
)


from flliper.srt.pdflip.launcher_vision_spec_mid import (  # noqa: F401 -- re-export, behaviour-neutral move
    VISION_ASYNC_SWITCHES,
    VISION_PLACES_BROKEN_IN_DUAL,
    VISION_SYNC_LAW,
    W111_VISION_PLACE_DUAL,
    _vision_knob,
    _weights_from_scores,
    shipped_layer_split,
    tokenizer_path_decision,
    torch_cache_cap_armed,
    transport_argv,
    user_reserve_rank_env,
    vision_async_refusal,
    vision_place_dual_refusal,
    vision_tower_dir,
)




#: #1233 boot weg2ls2b2 (2026-09-07 08:58Z, front corridor sampler with group
#: P awake, epoch 1, idle after the wake): NVML free 785 / 1806 / 990 MiB on
#: the 5090 (PP0) / nvml0 (PP1) / nvml2 (PP2) against the budget line's own
#: expectation total - budget - D_c(D) measured = 1501 / 1496 / 1498 MiB, i.e.
#: P lands 716 MiB OVER its line on the 5090 and 508 over on PP2 (PP1 310
#: under).  The 5090 shortfall armed PP0's rank-local #656 narrowing (4096 ->
#: 448) and killed the group (b1/b2 killer).  Charged here PER P ORDINAL so
#: that P's idle free lands ~1.7 GiB on the 5090 (a full 4096-token GDN chunk
#: prices ~635 MiB of transient there; the #656 floor is 1024) and ~1.5 GiB on
#: PP2; PP1 keeps its budget.  R5 grades the result; a boot that measures a
#: different overshoot replaces these numbers, it does not add to them.
P_OVERSHOOT_MIB = list(_PROFILE_27B.constant("P_OVERSHOOT_MIB"))
#: #1233 boot weg2ls4b1 (2026-09-07 11:24-11:50Z, front corridor sampler with
#: group D awake under three back-to-back agent-load rounds, 31 requests
#: each): NVML free continuous minimum 535 / 1260 / 1724 MiB on the 5090
#: (TP0, token share 29/64) / nvml0 / nvml2 against the budget line's own
#: expectation of CORRIDOR_MIB = 1428, i.e. D lands 893 MiB OVER its line on
#: the 5090 (NEXTN draft + verify-tree transients and the decode graphs sit
#: outside the --rank-gpu-memory-mib fraction, CAMPAIGN_d0: the capture term
#: is a lower bound ~4x below the driver cost) -- BELOW the 819-1229 band
#: (R5 corridor: UNDER = investigate).  Charged here on ordinal 0 so the
#: 5090's minimum lands mid-band (~1024).  The two 3080 ranks land 168 over /
#: 296 UNDER the line and are NOT charged: their residue is structural (the
#: pool size follows the min-over-ranks token count, so the cheaper ranks
#: allocate less), a planner item (uneven token vector), not a budget one.
#: A boot that measures a different overshoot replaces this number.
D_OVERSHOOT_MIB = list(_PROFILE_27B.constant("D_OVERSHOOT_MIB"))
D_WINDOWS_MIB = 16 + 32 + 24
P_WINDOWS_MIB = 24 + 96
#: TRAIN FIX 5.  The group-P FORM KEY has to be computed BEFORE the host ring,
#: because the ring is sized for the form that will run -- and the host-ledger
#: arm (S, M, store) is priced FROM the ring, so it does not exist yet.  These
#: four sentinels stand in its place while the form argv is assembled.  They are
#: SAFE ONLY BECAUSE every flag they reach is in
#: ``ring_table.FORM_KEY_EXCLUDED_FLAGS`` and is dropped before the hash, and
#: that is not left as a claim: the key is re-derived from the REAL argv once the
#: arm exists and any difference is a named refusal (W48). Values are
#: deliberately impossible ones, so a sentinel that ever escaped into a real
#: flag would be loud rather than plausible.
RING_FORM_SENTINEL_S_GB = -1
RING_FORM_SENTINEL_M_MIB = -1
RING_FORM_SENTINEL_DEPTH = -1

MODEL_DEFAULT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
#: #1233 draft KV across the flip (spec section 6, Q17): group P's token
#: capacity is EXPLICIT, never left to the profiler, because the last stage
#: now carries the MTP head (405.2 MiB) plus a resident INT8 embed_tokens
#: (1213.0 MiB) = 1618.2 MiB RESIDENT, and a draft device pool at
#: target-token parity (2048 B/token beside the 8192 B/token target cell).
#: Measured weg2zr2 on the last stage: KV pool 5584.4 MiB (714,788 tokens),
#: NVML free minimum 1450 MiB. Target: idle NVML free at the corridor TOP
#: (1229 MiB) so the draft-extend transient lands inside 819-1229:
#:   pool_new = 1450 + 5584.4 - 1618.2 - 1229 = 4187 MiB
#:   T_P      = 4187 x 2^20 / (8192 + 2048) = 428,769 -> 428,000 (uniform over
#:              the stages; demand bound 8 x 27,466 = 219,728, 1.95x slack).
#: BOOT weg2dk2 (35bdc9e310, 2026-09-07 18:22:15, PP2 log): the MTP head's
#: "Load weight end" delta was 3.90 GB = 3994 MiB -- 2376 MiB more than the
#: resident post. That figure is the BUILD, not the residue: it holds the
#: head's OWN never-loaded lm_head, a bf16 [248320 x 5120] table (2426 MiB;
#: "lm_head" is in the checkpoint's quantization ignore list, so it is not
#: int8), which `Qwen3_5ForCausalLMMTP.__init__` materialises on the scope's
#: single-rank is_last_rank and which the producer drops when it shares the
#: target's head (405.2 + 1213.0 + 2426.0 = 4044, 50 MiB from the reading).
#: No second embedding is budgeted: `load_resident_embedding` loads INTO the
#: built int8 tensors. The W11 gate below refuses the boot when the measured
#: residue exceeds this budget by more than the tolerance, so the corridor
#: claim of this derivation is checked at readiness, never assumed.
#: BOOT weg2dk3 (bc31554f90, PP2 log :260) measured `resident_mib=3998.0`
#: against this line and W11 refused. TWO defects, both fixed in fix 3, and
#: the budget stands unchanged:
#:  1. fix 2 released the table by swapping the MODULE and calling
#:     empty_cache(); the upstream release form is `del lm_head.weight`
#:     (qwen3_5_mtp.py:185-193), because a module swap frees nothing while
#:     any other holder remains. Fix 3 deletes the parameters.
#:  2. the instrument could not measure the residue at all: with
#:     --enable-memory-saver the weights are loaded inside
#:     `memory_saver_adapter.region(GPU_MEMORY_TYPE_WEIGHTS)`, which is
#:     `torch.cuda.use_mem_pool(...)` (torch_memory_saver/entrypoint.py:89-91),
#:     and `empty_cache()` does not return a MemPool block to the driver
#:     (maps/memsaver.md N4). The NVML free delta therefore reads the BUILD
#:     under this boot form no matter what is released. It is not a loss --
#:     the KV pools allocate from the SAME primary pool and reuse the block --
#:     so `resident_mib` is now the draft model's live weight bytes (shared
#:     target lm_head excluded) and the NVML delta rides beside it as its own
#:     named term `nvml_delta_mib`.
#: The next boot's L2 `resident_mib` replaces 1618.2 here, with its tag.
#:
#: fnFL2v71 (21.09.) is that next boot, and it REFUSED against the old pair:
#: measured resident_mib=2202.6 vs budget 1618.2+256. The build changed under
#: two fixes of the same day, both of which this budget has to follow:
#:   embed 405.2 -> 615.7 MiB  the vocab is now loaded PACKED, under the
#:                             TARGET's quantization (d84f1394fe). The old
#:                             number priced a table the producer refused to
#:                             load at all -- it is not a saving that was lost.
#:   mtp  1213.0 -> 1522.7 MiB the mtp layers now follow the DRAFT checkpoint
#:                             (a905902f47), which is INT4 g32, not the bf16
#:                             stump the target's index implied.
#: Sum 2138.4 MiB; the measured 2202.6 sits 64.2 MiB above it, inside the 256
#: tolerance -- the gate keeps its grip, it is only no longer aimed at a build
#: that no longer exists.
#: UNIFY S3: TWO measurements, one per model -- 27B 405.2 + 1213.0 (the NEXTN
#: head of the 27B line), NF 615.7 + 1522.7 (fnFL2v71 above) -- each in its
#: registry row (pdflip/form.py). This alias is the 27B row; check_draft_resident
#: reads the published profile's row.
P_DRAFT_RESIDENT_BUDGET_MIB = _PROFILE_27B.constant("P_DRAFT_RESIDENT_BUDGET_MIB")

#: #1362 [22-fix]: the layer count the vector above was MEASURED on. A model
#: with this depth may keep using it (that is the 27B, and every boot this line
#: has run); any other depth must bring its own calibration or be refused by
#: name. The constant existed with no model reference at all, which is how a
#: 64-layer measurement reached a 32-layer checkpoint.
CALIBRATION_LAYERS = _PROFILE_27B.constant("CALIBRATION_LAYERS")




#: The incumbent cut RENDERED for help text, DERIVED from the one definition
#: above rather than retyped.  It exists so ``build_parser`` can name the
#: numbers without becoming a fourth reader of the score constants -- the
#: registered guard in ``test_pdflip_host_budget_1233`` pins that reader set to
#: ``{argv_p, p_stage_layers, solve_p_cut}`` and walks FUNCTION bodies, so a
#: module-level derivation is the honest way to keep both properties: one
#: statement of the vector, and a help text that prints it.
P_PP_INCUMBENT_FMT = "%s / attention %s" % (
    _csv(P_PP_STAGE_RATIO_SCORES),
    _csv(P_PP_ATTN_STAGE_RATIO_SCORES),
)



P_DRAFT_KV_FLAGS: Tuple[str, ...] = (
    "--speculative-algorithm", SPEC_ALGORITHM,
    "--speculative-num-steps", str(SPEC_NUM_STEPS),
    "--speculative-eagle-topk", str(SPEC_EAGLE_TOPK),
    "--speculative-num-draft-tokens", str(SPEC_NUM_DRAFT_TOKENS),
    "--speculative-draft-kv-only",
)


#: The standing user order of 2026-09-07 is draft KV ACROSS THE FLIP, so the
#: producer is the default.  ``off`` is the serving-base / A-B form and is
#: never silent -- see :func:`draft_kv_off_line`.
DRAFT_KV_ON_P_DEFAULT = "on"







def apply_p_prefill_graph(ns) -> None:
    """Install --p-prefill-graph once, before any argv is built (the same
    install-once shape as apply_spec_form: every common_flags("P") call --
    the cut solve's chunk read, its armed re-check, the form key, the shipped
    argv -- reads this ONE value, so they cannot disagree)."""
    from flliper.srt.pdflip import p_graph_policy as _pgp

    bucket = int(getattr(ns, "p_prefill_graph", P_PREFILL_GRAPH_DEFAULT) or 0)
    if bucket < 0:
        raise SystemExit(f"--p-prefill-graph must be >= 0, got {bucket}")
    _P_PREFILL_GRAPH["bucket"] = bucket
    _P_PREFILL_GRAPH["tiny"] = p_prefill_graph_tiny_of(ns, bucket)
    policy = str(getattr(ns, "p_prefill_graph_policy", P_PREFILL_GRAPH_POLICY_DEFAULT)
                 or P_PREFILL_GRAPH_POLICY_DEFAULT)
    raw_extra = str(getattr(ns, "p_prefill_graph_buckets", "") or "").strip()
    try:
        extras = tuple(sorted({int(x) for x in raw_extra.split(",") if x.strip()}))
    except ValueError:
        raise SystemExit(f"--p-prefill-graph-buckets {raw_extra!r}: comma list of token counts expected")
    cal = p_prefill_graph_calibration_of(ns)
    min_gain = float(getattr(ns, "p_prefill_graph_min_gain", _pgp.DEFAULT_MIN_GAIN))
    legacy = tuple(sorted(set(_P_PREFILL_GRAPH["tiny"]) | {bucket})) if bucket else ()
    _P_PREFILL_GRAPH.update(policy=policy, legacy=legacy, extras=extras, min_gain=min_gain,
                            cut_check=None)
    _p_prefill_graph_decide(cal)


def _p_prefill_graph_decide(cal) -> None:
    """The policy's answer for the installed switch and ``cal`` (None = no
    table); shared by the install and the cut check's fallback."""
    from flliper.srt.pdflip import p_graph_policy as _pgp

    g = _P_PREFILL_GRAPH
    policy, bucket = str(g["policy"]), int(g["bucket"])
    try:
        captured, verdicts = _pgp.decide(policy, bucket, g["tiny"], g["extras"], cal,
                                         p_group_stages(), float(g["min_gain"]))
    except _pgp.GraphPolicyError as exc:
        raise SystemExit(str(exc))
    g.update(
        captured=tuple(captured), verdicts=tuple(verdicts), calibration=cal,
        lines=tuple(_pgp.decision_lines(policy, captured, verdicts, cal)) if bucket else (),
    )


def p_prefill_graph_cut_check(ns, cut, log) -> bool:
    """The calibration table's P cut against the cut this boot SHIPS (Agent
    PG2 26.09.; graphcal_int8 was measured with G 43,11,10 against E
    44,10,10 and was therefore no measurement). Called ONCE in the launcher
    right after solve_p_cut, before any rank starts, so every P rank runs
    the same verdict (raenge-nie-uneins): the ranks only ever see the
    captured set and the chunk spec this function leaves behind.

    No table or the switch off: nothing, no line (byte-identical). The table
    names no cut: a WARNING line, the table stays. A different cut: the loud
    ``PREFILL-GRAPH-CALIBRATION CUT-MISMATCH table=... boot=... -> fallback``
    line, the policy is decided again WITHOUT the table and the chunk policy
    re-installed; returns True -- the capture set, and with it the capture
    pool post the solve priced, may have changed, so the caller solves the
    cut again (that solve has no table, so it is final)."""
    from flliper.srt.pdflip import p_graph_policy as _pgp

    cal = _P_PREFILL_GRAPH.get("calibration")
    if cal is None or not p_prefill_graph_bucket():
        return False
    verdict, line = _pgp.cut_check(
        cal, tuple(getattr(cut, "layer_counts", ()) or ()),
        gapped=bool(getattr(cut, "gapped", False)), layer_set=str(getattr(cut, "layer_set", "") or ""))
    log("PDFLIP " + line)
    if verdict != _pgp.CUT_MISMATCH:
        _P_PREFILL_GRAPH["cut_check"] = verdict
        _P_PREFILL_GRAPH["lines"] = tuple(p_prefill_graph_policy_lines()) + (line,)
        return False
    _p_prefill_graph_decide(None)
    _P_PREFILL_GRAPH["cut_check"] = verdict
    _P_PREFILL_GRAPH["lines"] = tuple(p_prefill_graph_policy_lines()) + (line,)
    # The chunk plan priced the widths off the same table (overlay): without it.
    apply_p_chunk_policy(ns)
    return True








def p_prefill_graph_vram_gate(ns, model_pool, pinned_layers, families, pool_floor, log):
    """Extra buckets (above the main one) that 'auto' admitted must also fit:
    the P cut's pool WITH their capture pool has to clear the pool floor.
    Returns the (possibly rebuilt) pool model; drops the extras -- and says so
    -- when they do not fit or when the cut is not pinned (the solved field
    has no single cut to check against, so the safe side is taken)."""
    from flliper.srt.planner import pp_cut as _pp_cut

    captured = tuple(p_prefill_graph_captured())
    legacy = tuple(_P_PREFILL_GRAPH.get("legacy", ()) or ())
    extras = tuple(b for b in captured if b not in legacy)
    if str(_P_PREFILL_GRAPH.get("policy")) != "auto" or not extras:
        return model_pool
    reason = None
    counts = tuple(int(x) for x in (pinned_layers or ()))
    if not counts or sum(counts) != len(families):
        reason = "the cut is not pinned by --pp-stage-ratio"
    else:
        attn, lo = [], 0
        for c in counts:
            attn.append(sum(1 for f in families[lo:lo + c] if f == _pp_cut.LAYER_FAMILY_ATTENTION))
            lo += c
        try:
            pool = _pp_cut.pp_phase_pool(counts, attn, model_pool)
        except ValueError as exc:
            pool, reason = None, f"the pinned cut is not priceable with them ({exc})"
        if pool is not None:
            if pool_floor is None or pool >= float(pool_floor):
                log("P-PREFILL-GRAPH-POLICY VRAM gate: extra bucket(s) %s CARRIED -- pinned cut %s "
                    "pool %d >= floor %s with prefill graph pool %s MiB/stage"
                    % (list(extras), ",".join(map(str, counts)), int(pool), pool_floor,
                       ",".join("%.1f" % v for v in model_pool.prefill_graph_pool_mib)))
                return model_pool
            reason = "pinned cut %s pool %d < floor %d with their capture pool" % (
                ",".join(map(str, counts)), int(pool), int(pool_floor))
    kept = tuple(b for b in captured if b in legacy)
    _P_PREFILL_GRAPH["captured"] = kept
    _P_PREFILL_GRAPH["lines"] = tuple(p_prefill_graph_policy_lines()) + (
        "P-PREFILL-GRAPH-POLICY VRAM gate: extra bucket(s) %s DROPPED (%s); captured=%s"
        % (list(extras), reason, list(kept) or "none"),)
    log(_P_PREFILL_GRAPH["lines"][-1])
    # The chunk policy prices the widths against the captured set: re-install.
    apply_p_chunk_policy(ns)
    return replace(model_pool, prefill_graph_pool_mib=p_prefill_graph_pool_mib(ns))










from flliper.srt.pdflip.launcher_resolve_stop import (  # noqa: F401 -- re-export, behaviour-neutral move
    KV_TOKEN_CUT_FLIP_MARKER,
    _rope_cols,
    p_prefill_graph_env,
    p_prefill_graph_flags,
    planned_expert_store_bytes,
    refuse_flip_stage_form_without_trim_cells,
    refuse_flip_under_token_cut,
    vision_gguf_tower_refusal,
    vision_victim_host_term,
)


#: The `prefill graph pool` post, MiB per 512 graph tokens per P stage. AN
#: ESTIMATE, NOT A MEASUREMENT (desk, 2026-09-24) -- the capture runs after
#: KV sizing, so the first boot can only book a forecast, and every later
#: boot should book the MEASURED rank line 'PREFILL-GRAPH captured ...
#: capture_mib=' via --p-prefill-graph-pool-mib. Derivation for one 512-token
#: chunk of the 27B (hidden 5120, intermediate 17408, bf16): the capture pool
#: reuses freed blocks inside the capture, so it holds ONE layer's working
#: set -- MLP gate_up 34 MiB + activation 17 + int8 activation 8.5 + the GDN
#: chunk intermediates ~30 + the residual stream 10 -- plus the stage's
#: outputs (hidden + residual 10 MiB, + 5 MiB per DFlash capture when the
#: producer computes) and the static inputs (embeddings / proxy 5-35 MiB).
#: The flashinfer full-CG split-kv workspace (~167 MiB on the 5090, ~94 MiB
#: on a 3080) is NOT in it: under the switch P shares the float workspace
#: that is already in the cuda_graph tag (FLLIPER_FULL_CG_PREFILL_SHARED_
#: WORKSPACE). The task's own forecast was ~0.1 GiB per stage.
P_PREFILL_GRAPH_POOL_MIB_PER_512 = 160.0
#: Group P's stage count on the reference rig (argv_p ships --pp-size <N>; N = 3).
P_PREFILL_GRAPH_STAGES = 3


def p_group_stages() -> int:
    """HW-P1c 1003: group P's stage count = the launch's card count (one stage
    per card). :data:`P_PREFILL_GRAPH_STAGES` (3) wherever NVML cannot say, so
    the reference rig and every desk test read what they always read."""
    live = _live_inventory_or_none()
    if not live:
        return P_PREFILL_GRAPH_STAGES
    # only a launch the inventory check will DERIVE (a subset of the calibrated
    # cards) changes the table; a foreign inventory keeps the three-stage tables
    # and meets the named HW-COUNT / HW-UNCALIBRATED refusal in main first
    cal, _ = records_inventory(pdflip_form.DEFAULT_PROFILE)
    return len(live) if inventory_view_mod.is_subset(live, cal) else P_PREFILL_GRAPH_STAGES


def p_prefill_graph_pool_mib(ns) -> Tuple[float, ...]:
    """The per-stage `prefill graph pool` post, MiB; () when the switch is
    off. ONE vector for both sides of the seam: the pool model's
    PhasePoolModel.prefill_graph_pool_mib and the ranks' FLLIPER_KV_BUDGET_
    PREFILL_GRAPH_MIB (p_prefill_graph_env) are both built from this call."""
    if not p_prefill_graph_buckets():
        return ()
    raw = str(getattr(ns, "p_prefill_graph_pool_mib", "") or "").strip()
    if raw:
        vals = tuple(float(x) for x in raw.split(",") if x.strip())
        if len(vals) == 1:
            vals = vals * p_group_stages()
        if len(vals) != p_group_stages() or any(v < 0 for v in vals):
            raise SystemExit(
                f"--p-prefill-graph-pool-mib {raw!r}: one value or "
                f"{p_group_stages()} non-negative per-stage values expected"
            )
        return vals
    # Tiny buckets (--p-prefill-graph-tiny) are priced proportionally on top:
    # an UPPER estimate, the runner captures every bucket into ONE pool (the
    # full backend's global graph pool, largest first) and the smaller ones
    # reuse its blocks; the rank line 'PREFILL-GRAPH captured ... capture_mib='
    # measures all buckets together and replaces this via the flag above.
    est = P_PREFILL_GRAPH_POOL_MIB_PER_512 * float(sum(p_prefill_graph_buckets())) / 512.0
    return (est,) * p_group_stages()


#: The crossover prefix read off the xsn426/xsn428 #PGAP fits: the graph's
#: PP0 chunk costs 41.9 + 1.432 * (prefix + 256) / 1000 ms, the eager chunk is
#: bounded below by the ~57 ms host launch of 42 layers, and the split lowers
#: the eager device time under that line -- so eager + split wins from
#: (57 - 41.9) / 1.432 * 1000 - 256 ~ 10,300 tokens on. Rounded to 20 chunks
#: of 512. A RECOMMENDATION printed in the help, never applied by default.
P_DEEP_SPLIT_FROM_FIT = 10240








P_CHUNK_MSCALE_DEFAULT = "int8"
#: Short-prompt bypass (rc9j metal 26.09., Operator): at 2k/8k the plan is
#: 512xn anyway, yet B measured 10-30 ms slower than A. At or below this rest
#: the cursor does not plan and every forward gets the fixed width.
P_CHUNK_DYNAMIC_MIN_TOKENS_DEFAULT = 8192
#: Per-token cost relative to M=512, per stage, as a function of the chunk.
#: 'int8' PP0 (5090): kernel bench n4b_bench_0925_1239 -- the INT8 GEMMs
#: (gate_up/down/qkvz/o) cost 12.3 % less per token at M=4096 than at 512
#: (470 -> 535 TOPS gate_up), GEMMs are ~87 % of PP0's 512 chunk (35.8 of
#: ~41 ms by the same bench), log-interpolated between 512 and 4096 -- a
#: HOCHRECHNUNG, not a measurement. 256 is a guess (+15 %). The 3080 stages
#: have NO bench above M=512, so they are FLAT (no gain assumed) until the
#: metal ladder measures them. 'nvfp4' PP0: NVFP4 GEMM 5090 -21 % (gate_up)
#: / -28 % (down) per token at 4096, same share -- also a HOCHRECHNUNG.
P_CHUNK_MSCALES = {
    "flat": ({512: 1.0}, {512: 1.0}, {512: 1.0}),
    "int8": (
        {256: 1.15, 512: 1.0, 1024: 0.964, 2048: 0.929, 4096: 0.893},
        {256: 1.10, 512: 1.0},
        {256: 1.10, 512: 1.0},
    ),
    "nvfp4": (
        {256: 1.10, 512: 1.0, 1024: 0.93, 2048: 0.87, 4096: 0.80},
        {256: 1.10, 512: 1.0},
        {256: 1.10, 512: 1.0},
    ),
}
#: --p-power-scale (release table row 27, pdflip/power_limit.py). 'off' = the
#: model is used as calibrated (spec byte-identical); only warning lines added.
P_POWER_SCALE_DEFAULT = "off"
P_POWER_SCALE_LAWS = ("off", "linear", "power")


def p_stage_live_classes(stages: int) -> List[Optional[str]]:
    """The live calibration-class label of each P stage (``order_cards``
    order); ``None`` per stage when NVML cannot answer (never refuses)."""
    try:
        cards = order_cards(resolve_cards())
    except Exception:  # noqa: BLE001 - a label, never a boot blocker
        return [None] * stages
    return [card_identity.class_label(cards[r]) if r < len(cards) else None for r in range(stages)]


def p_stage_class_mismatch_lines(stages: int, live: Optional[Sequence[Optional[str]]] = None) -> List[str]:
    """One line per P stage whose live card class is not the class the
    builtin/fitted chunk model was calibrated on (the power comparison of
    row 27 is then between different cards: UNCALIBRATED, named)."""
    live = list(p_stage_live_classes(stages) if live is None else live)
    cal = list(P_STAGE_CARD_CLASSES[:stages]) + [P_STAGE_CARD_CLASSES[-1]] * max(0, stages - 3)
    return [f"P-STAGE-CLASS PP{r}: live {lv} vs calibration {cv} -- the chunk model's timings and "
            f"power limit are a measurement of {cv}, not of this card ({card_identity.CODE_UNCALIBRATED})"
            for r, (lv, cv) in enumerate(zip(live, cal)) if lv is not None and lv != cv]




def p_stage_power_current(stages: int) -> Tuple[List[Optional[float]], List[str]]:
    """(running power limit per P stage, stage labels): PP<r> is the r-th card
    of order_cards (5090 first), read from NVML by UUID. Unknown (NVML
    unavailable) = None per stage, labels without a card."""
    from flliper.srt.pdflip import power_limit as _pl

    try:
        cards = order_cards(resolve_cards())
        cur = _pl.read_current()
    except Exception:  # noqa: BLE001 - the check degrades to 'unknown', it never blocks a boot
        return [None] * stages, [f"PP{r}" for r in range(stages)]
    vals, labels = [], []
    for r in range(stages):
        c = cards[r] if r < len(cards) else None
        w = cur.get(c.uuid, (None, ""))[0] if c is not None else None
        vals.append(None if w is None else float(w))
        labels.append(f"PP{r}" + (f"(nvml{c.nvml_index} {card_identity.class_label(c)})" if c is not None else ""))
    return vals, labels


def apply_p_power_check(ns, models, source: str, src: str):
    """Row 27 on the installed chunk model: compare, warn, optionally rescale.
    Returns (models, source); both unchanged under --p-power-scale off."""
    from flliper.srt.pdflip import power_limit as _pl

    law = str(getattr(ns, "p_power_scale", P_POWER_SCALE_DEFAULT) or P_POWER_SCALE_DEFAULT)
    alpha = getattr(ns, "p_power_scale_exponent", None)
    try:
        _pl.exponent_of(law, alpha)
    except _pl.PowerLimitError as exc:
        raise SystemExit(f"--p-power-scale: {exc}")
    n = len(models)
    cal, cal_src = p_chunk_model_power_limits(src, n)
    cur, labels = p_stage_power_current(n)
    verdicts = _pl.check(labels, cal or [None] * n, cur, law, alpha)
    lines = _pl.verdict_lines("p-chunk-model", verdicts, law, alpha, cal_src)
    # HW-GENERIC 1002: a stage on another card class than the model's
    # calibration stage is named beside the power verdicts (empty on the
    # reference rig).
    lines = list(lines) + p_stage_class_mismatch_lines(n)
    _P_CHUNK["power_lines"] = tuple(lines)
    if not any(v.factor != 1.0 for v in verdicts):
        return models, source
    models = tuple(_pl.scale_stage_model(m, v.factor) for m, v in zip(models, verdicts))
    tag = ",".join(f"PP{r}x{v.factor:.4f}" for r, v in enumerate(verdicts) if v.factor != 1.0)
    return models, f"{source} +powerscale:law={law},{tag} ({_pl.MODEL_LABEL})"






def _p_chunk_by_class(items, key, stages: int, profile: Optional[str]):
    """HW-P1c 1003: the reference stage tables (one entry per calibrated card)
    for a P group of ``stages`` stages on a SUBSET of those cards: each live
    stage takes the SLOWEST (by ``key``) calibrated stage of its card class.
    ``(mapped items, the reference index each live stage took)``; ``None`` when
    there is nothing to map (same count) or no mapping exists."""
    items = tuple(items)
    if int(stages) == len(items):
        return None
    prof = str(profile or pdflip_form.DEFAULT_PROFILE)
    cal, _ = records_inventory(prof)
    live = _live_inventory_or_none()
    if not live or len(live) != int(stages) or len(cal) != len(items):
        return None
    if not inventory_view_mod.is_subset(live, cal):
        return None
    idx = tuple(max((i for i, k in enumerate(cal) if k == c), key=lambda i: key(items[i])) for c in live)
    return tuple(items[i] for i in idx), idx


def p_chunk_stage_model(src: str, mscale: str, stages: int = 3, profile: Optional[str] = None):
    """``(stage models, source text)`` for --p-chunk-model / --p-chunk-mscale.
    ``stages`` != the reference table's three: the table is mapped to the live
    stages by card class (:func:`_p_chunk_by_class`), else refused as before."""
    from flliper.srt.pdflip import p_chunk_policy as _pcp

    src = str(src or P_CHUNK_MODEL_DEFAULT).strip()
    if mscale not in P_CHUNK_MSCALES:
        raise SystemExit(f"--p-chunk-mscale {mscale!r}: one of {sorted(P_CHUNK_MSCALES)}")
    curves = P_CHUNK_MSCALES[mscale]
    if src in P_CHUNK_BUILTIN_SOURCES:
        rows, _record, label = P_CHUNK_BUILTIN_SOURCES[src]
        source = f"{src} ({label}, mscale={mscale})"
    elif src.startswith("fit:"):
        from flliper.srt.planner import pgap_stage_fit as _psf

        path = src[4:]
        try:
            lines = _psf.fit_rank_lines(_psf.read_pgap_log(path), lo=0, hi=32768)
        except (OSError, _psf.StageFitRefused) as exc:
            raise SystemExit(f"--p-chunk-model {src}: {exc}")
        rows = tuple(
            {"a_ms": ln.a_ms, "b_ms_per_1k": ln.b_ms_per_1k,
             "fwd_overhead_ms": P_CHUNK_BUILTIN_INT8[min(i, 2)]["fwd_overhead_ms"],
             "eager_floor_ms": P_CHUNK_BUILTIN_INT8[min(i, 2)]["eager_floor_ms"]}
            for i, ln in enumerate(lines)
        )
        source = f"fit:{os.path.basename(path)} [0,32k) mscale={mscale}"
    else:
        try:
            with open(src) as fh:
                data = json.load(fh)
            models = tuple(_pcp.StageModel.from_json(s) for s in data["stages"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SystemExit(f"--p-chunk-model {src}: not {'/'.join(P_CHUNK_BUILTIN_SOURCES)}, fit:<P.log> or a "
                             f"readable JSON with 'stages': {exc}")
        mapped = _p_chunk_by_class(models, lambda m: float(m.points[-1][1]), stages, profile)
        if mapped is not None:
            import dataclasses

            return tuple(dataclasses.replace(m, name=f"PP{j}") for j, m in enumerate(mapped[0])), (
                f"json:{os.path.basename(src)} (HW-DERIVE: {len(models)} -> {stages} stages by card class)")
        if len(models) != stages:
            raise SystemExit(f"--p-chunk-model {src}: {len(models)} stages, group P has {stages}")
        return models, f"json:{os.path.basename(src)}"
    curve_of = list(range(len(rows)))   # the chunk-scale curve follows the CARD, not the stage index
    mapped_rows = _p_chunk_by_class(
        rows, lambda r: float(r["a_ms"]) + 4.0 * float(r["b_ms_per_1k"]), stages, profile)
    if mapped_rows is not None:
        rows, curve_of = mapped_rows[0], list(mapped_rows[1])
        source += f" (HW-DERIVE: stage table mapped to {stages} stages by card class)"
    if len(rows) != stages:
        raise SystemExit(f"--p-chunk-model {src}: {len(rows)} stages, group P has {stages}")
    models = tuple(
        _p_chunk_stage(r["a_ms"], r["b_ms_per_1k"], r["fwd_overhead_ms"], r["eager_floor_ms"],
                       curves[min(curve_of[i], len(curves) - 1)], f"PP{i}")
        for i, r in enumerate(rows)
    )
    return models, source


def apply_p_chunk_policy(ns, boot_form=None, argv_words: Sequence[str] = ()) -> None:
    """Install --p-chunk-policy once, AFTER apply_p_prefill_graph (it reads the
    graph buckets) and before any argv is built: p_chunked_prefill_tokens()
    reads the ceiling from here, so the cut solve, the ledger, the form key
    and the shipped argv see ONE value.

    H92 (NF): a profile whose chunk row prices 'dynamic' by model key
    (``Chunk.dynamic_source``, nextflash) resolves the NF profile of
    this checkpoint instead (:func:`_apply_p_chunk_policy_nf`); its spec is
    built with env_p (``p_chunk_policy_install``). Every other profile takes
    the path below, unchanged."""
    from flliper.srt.pdflip import p_chunk_policy as _pcp

    policy = str(getattr(ns, "p_chunk_policy", P_CHUNK_POLICY_DEFAULT) or P_CHUNK_POLICY_DEFAULT)
    if policy not in _pcp.POLICIES:
        raise SystemExit(f"--p-chunk-policy {policy!r}: one of {list(_pcp.POLICIES)}")
    _P_CHUNK["policy"] = policy
    _P_CHUNK["spec"] = None
    _P_CHUNK["power_lines"] = ()
    _P_CHUNK["profile"] = None
    _P_CHUNK["ns"] = None
    if policy == _pcp.POLICY_FIXED:
        return
    _row = pdflip_form.profile_row((getattr(boot_form, "profile", "") if boot_form is not None
                                  else getattr(ns, "profile", "")) or "")
    if _row is not None and _row.chunk.dynamic_source == "model-key":
        _apply_p_chunk_policy_nf(ns, boot_form, argv_words)
        return
    bucket = p_prefill_graph_bucket()
    fixed = int(getattr(ns, "p_chunk_fixed", 0) or 0) or bucket or CHUNKED_PREFILL_TOKENS
    cap = int(getattr(ns, "p_chunk_max", P_CHUNK_MAX_DEFAULT) or P_CHUNK_MAX_DEFAULT)
    low = int(getattr(ns, "p_chunk_min", 0) or 0) or fixed
    buckets = tuple(b for b in p_prefill_graph_buckets())
    if bucket and cap < bucket:
        raise SystemExit(f"--p-chunk-max {cap} is below the P prefill graph bucket {bucket}")
    models, source = p_chunk_stage_model(
        str(getattr(ns, "p_chunk_model", P_CHUNK_MODEL_DEFAULT) or P_CHUNK_MODEL_DEFAULT),
        str(getattr(ns, "p_chunk_mscale", P_CHUNK_MSCALE_DEFAULT) or P_CHUNK_MSCALE_DEFAULT),
        p_group_stages(),
        profile=getattr(ns, "profile", None),
    )
    # --p-prefill-graph-policy: the plan prices every width in the mode the
    # policy decided (graph or eager), from the SAME calibration table. No
    # table = the models are untouched (spec byte-identical).
    _cal = _P_PREFILL_GRAPH.get("calibration")
    if _cal is not None and bucket:
        from flliper.srt.pdflip import p_graph_policy as _pgp

        models = _pgp.overlay_stage_models(models, _cal, buckets)
        source = f"{source} +graphcal:{_cal.source or 'table'}"
    # Release table row 27: the model's calibration power limit against the
    # running one (after the graph overlay, so a rescale covers every point
    # the plan prices). --p-power-scale off: models/source untouched.
    models, source = apply_p_power_check(
        ns, models, source,
        str(getattr(ns, "p_chunk_model", P_CHUNK_MODEL_DEFAULT) or P_CHUNK_MODEL_DEFAULT))
    _sweep_raw = str(getattr(ns, "p_chunk_sweep", "") or "").strip()
    try:
        _sweep = tuple(int(x) for x in _sweep_raw.split(",") if x.strip())
    except ValueError:
        raise SystemExit(f"--p-chunk-sweep {_sweep_raw!r}: comma list of token counts expected")
    try:
        limits = _pcp.ChunkLimits(
            max_tokens=cap, min_tokens=low, fixed_tokens=fixed, page=1,
            grid=int(getattr(ns, "p_chunk_grid", 0) or 0),
            graph_buckets=buckets, eager=True,
            min_gain=float(getattr(ns, "p_chunk_min_gain", _pcp.DEFAULT_MIN_GAIN)),
            dynamic_min_tokens=int(getattr(ns, "p_chunk_dynamic_min_tokens",
                                           P_CHUNK_DYNAMIC_MIN_TOKENS_DEFAULT) or 0),
            sweep=_sweep,
        )
    except _pcp.ChunkPolicyError as exc:
        raise SystemExit(f"--p-chunk-policy dynamic: {exc}")
    _P_CHUNK["spec"] = _pcp.PolicySpec(tuple(models), limits, source)
















from flliper.srt.pdflip.launcher_d_p_apply import (  # noqa: F401 -- re-export, behaviour-neutral move
    DRAFT_PARK_UNPRICED,
    EXTEND_TRIM_ENV,
    EXTEND_TRIM_FLIP_MARKER,
    MAMBA_ARENA_ENV,
    P_ATTN_HEAD_SPLIT_DEFAULT,
    P_ATTN_HEAD_SPLIT_MIN_W_DEFAULT,
    P_LAYER_SPLIT_PULL_GBPS_DEFAULT,
    P_PREFILL_GRAPH_SPLIT_DEFAULT,
    TRIM_LAW_MARKER,
    _d_early_start_armed,
    _d_extend_flip_trim_env,
    apply_profile_d_kv_token_cut_default,
    apply_tokenizer_path,
    census_checkpoint_decision,
    d_draft_park_term,
    d_fr_ceiling_before_map,
    d_reshard_format,
    d_seat_lines,
    d_stated_seats,
    group_kv_tokens,
    live_launch_servers,
    log_draft_swap,
    mamba_arena_slots_into_groups,
    model_num_layers,
    p_attn_head_split_cfg,
    p_graph_split_env,
    p_layer_split_spec_from_ns,
)
from flliper.srt.pdflip.launcher_hicache_calib import (  # noqa: F401 -- re-export, behaviour-neutral move
    D_RESHARD_DRY_RUN_LOADS,
    P_LAYER_SPLIT_DRY_RUN_TOKENS,
    _p_ah_post_vector,
    _rope_context_terms,
    apply_d_reshard,
    apply_p_layer_split,
    d_rope_context_delta_mib,
    model_layer_kinds,
    publish_expert_map,
)







































































from flliper.srt.pdflip.launcher_resolve_xchg_r2_L1_1 import (  # noqa: F401 -- re-export, behaviour-neutral move
    HICACHE_DRAFT_TIER_DEFAULT,
    HICACHE_DRAFT_TIER_ENV,
    HICACHE_DRAFT_TIER_VALUES,
    _write_state,
    prepare_xchg_env,
    rank_vram_display,
    register_helper,
    release_shared_cache,
    resolve_hicache_draft_tier,
    unregister_helper,
    xchg_form_dormant_reserve,
)










































#: #239 S3g: the stage rows per D rank under the token cut (every KV rank)
D_KV_STAGE_ROWS_BY_RANK_KEY = "FLLIPER_PDFLIP_D_KV_STAGE_ROWS_BY_RANK"
















D_KV_STAGE_LRU_FLOOR_ENV = "FLLIPER_PDFLIP_D_KV_STAGE_LRU_FLOOR"
#: re-solves the floor may ask for in one pass (the second one checks the first)
D_KV_STAGE_LRU_FLOOR_ROUNDS = 2
















def apply_d_kv_stage_form(ns, er, rows, seat_vram, plan, label, *, verify_tokens: int,
                          top_k: int, host_rank: int = 0, kv_token_shares=None) -> List[str]:
    """#251c: D's KV stage form into --env-d (before build_env reads it) --
    the stage tokens, the stage rows (TP0 moves them from its scratch into its
    seat rows: FLLIPER_MOE_SCRATCH_SLOTS[0] - rows, FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS
    [0] + rows, so the boot maps the same bank) and the highest stage per seat
    count (``expert_residency.kv_stage_table``, default: no extra wave).
    An operator's FLLIPER_PDFLIP_D_KV_STAGE_TOKENS wins (a single stage = off);
    an operator's FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS replaces the default
    highest stage per seat count (the A/B arm), checked against the table.
    Returns the lines to log; ``d_kv_stage_undo`` reverses it for a second
    solve pass."""
    head = "%s FRACTION-SOLVE %s D-KV-STUFEN (#251c)" % (D_RANK_SOLVE_MARKER, label)
    ns._d_kv_stage_lru_raise = None
    env_raw = str(getattr(ns, "env_d", "") or "")
    env = parse_group_env(env_raw)
    if D_KV_STAGE_KEYS[0] in env:
        return ["%s: --env-d nennt %s=%s selbst -- keine Stufenform vom Launcher"
                % (head, D_KV_STAGE_KEYS[0], env[D_KV_STAGE_KEYS[0]])]
    fits = list(getattr(plan, "fits", None) or ())
    cut_workers = kv_stage_cut_workers(kv_token_shares, host_rank)
    staged = [f.rank for f in fits
              if f.rank != host_rank and er.kv_stage_trim_cell(f, host_rank) > 0]
    missing = [r for r in cut_workers if r not in staged]
    if missing:
        # 28.09. (M1 prep): the stage form sizes EVERY rank's KV pool at the top
        # stage (kv_stage_pool_tokens); a worker holding FA KV under the cut
        # trims it only with its own stage rows (S3g), i.e. with its trim cell
        # in the plan. A cut the plan does not carry per rank: no form.
        return ["%s: entfaellt unter dem Token-Schnitt %s -- KV-haltende Worker %s ohne "
                "Trim-Zelle im Plan bekaemen ihren FA-KV-Pool auf der obersten Stufe "
                "ungetrimmt (Stufen je KV-Rang: S3g requires den geloesten Schnitt); "
                "D faehrt fest %s Token"
                % (head, kv_token_shares, "(noch ungeloest)" if cut_workers == [-1]
                   else "Rang %s" % ",".join(str(r) for r in cut_workers),
                   next((int(f.kv_tokens) for f in fits if f.rank == host_rank),
                        "die gebuchten"))]
    why: List[str] = []
    floor_tokens = d_kv_stage_floor_tokens(env)
    group = er.kv_stage_group(rows, seat_vram, fits, verify_tokens=int(verify_tokens),
                              top_k=int(top_k), host_rank=host_rank, why=why,
                              floor_tokens=floor_tokens)
    if group is None:
        return ["%s: entfaellt -- %s" % (head, "; ".join(why) or "ohne Grund")]
    tab = group.host
    op_raw = env.get(D_KV_STAGE_KEYS[2])
    op_max = operator_kv_stage_max(op_raw, len(tab.max_by_seats), len(tab.tokens))
    max_by = group.max_by_seats if op_max is None else op_max
    by_demand = d_kv_stage_by_demand(env)
    if by_demand:
        # #251d: no table -- every stage at every seat count (the rank does not
        # read the row either; it is written so the capture line names it)
        max_by = tuple(len(tab.tokens) - 1 for _ in tab.max_by_seats)
    waves_to, waves_lines = kv_stage_wave_cap(ns, er, group, max_by, env, head)
    # the cap the captured steps run with: the launcher's raise, else the
    # H95 cap the seat table was solved at (the env the launcher derives)
    waves_lines = waves_lines + kv_stage_wave_floor(
        group, rows, fits, max_by,
        int(waves_to) if waves_to is not None
        else max(er.pool_overflow_waves(env), int(rows[-1].waves)), head)
    # (3) 29.09.: the measured peak as the LRU floor of every allowed stage;
    # a short rank gets its scratch raised and the solve runs again
    # (log_d_rank_vram_solve), so FR_D moves the rows from resident to scratch
    _peak, _span, _peak_src = d_pool_peak_record(getattr(ns, "profile", None))
    if _peak is not None:
        if os.environ.get(D_KV_STAGE_LRU_FLOOR_ENV, "1").strip() == "0":
            waves_lines = waves_lines + ["%s: %s aus (%s=0)" % (
                head, D_KV_STAGE_LRU_FLOOR_MARKER, D_KV_STAGE_LRU_FLOOR_ENV)]
        else:
            _add, _floor_lines = kv_stage_lru_floor(group, rows, fits, max_by, _peak, _span,
                                                    _peak_src, head)
            waves_lines = waves_lines + _floor_lines
            ns._d_kv_stage_lru_raise = _add or None
    n = len(rows[-1].scratch_given)
    scratch = _rank_vec(env.get("FLLIPER_MOE_SCRATCH_SLOTS"), n)
    seat_raw = env.get("FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS")
    seat = _rank_vec(seat_raw, n)
    # every KV rank moves its stage rows from its scratch into its seat rows;
    # the rows a floor below S0 funds (low_rows) are NEW rows of the bank (the
    # KV they replace is born unmapped), so only the rest leaves the scratch
    for t in group.tables:
        scratch[t.host_rank] -= t.rows - int(getattr(t, "low_rows", 0) or 0)
        seat[t.host_rank] += t.rows
    items = {
        "FLLIPER_MOE_SCRATCH_SLOTS": ",".join(str(x) for x in scratch),
        "FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS": ",".join(str(x) for x in seat),
        D_KV_STAGE_KEYS[0]: ",".join(str(t) for t in tab.tokens),
        D_KV_STAGE_KEYS[1]: str(tab.rows),
        D_KV_STAGE_KEYS[2]: ",".join(str(j) for j in max_by),
    }
    worker_tabs = group.tables[1:]
    if worker_tabs:
        items[D_KV_STAGE_ROWS_BY_RANK_KEY] = ",".join(str(x) for x in group.rows_by_rank)
    if waves_to is not None:
        items[er.POOL_OVERFLOW_WAVES_ENV] = str(int(waves_to))
    for k, v in items.items():
        env_raw = set_group_env(env_raw, k, v)
    ns.env_d = env_raw
    vram_view().note_d_stages(group)
    ns._d_kv_stage_written = {"host": host_rank, "rows": tab.rows, "seat_raw": seat_raw,
                              "op_max": op_raw if op_max is not None else None,
                              "worker_rows": {int(t.host_rank): int(t.rows)
                                              for t in worker_tabs},
                              "cut_workers": [int(r) for r in cut_workers]}
    if waves_to is not None:
        ns._d_kv_stage_written["waves_raw"] = env.get(er.POOL_OVERFLOW_WAVES_ENV)
    caps = " ".join("n=%d:%s" % (i + 1, "/".join(str(c) for c in cs))
                    for i, cs in enumerate(tab.capacity))
    # the A/B candidate (main 28.09.): S1 wherever the default keeps S0 -- the
    # waves it adds per bs, priced by the capture floor (not written)
    cand = tuple(max(1, int(j)) if len(tab.tokens) > 1 else 0 for j in group.max_by_seats)
    cand_extra = group.extra_waves(cand)
    op_line = [] if op_max is None else [
        "%s: OPERATOR %s=%s statt der Voreinstellung %s -- Zusatzwellen je bs %s "
        "(Voreinstellung %s); im Rang: '#251 WAKE-RESHARD CAPTURE-FLOOR ... max_by_seats %s'"
        % (head, D_KV_STAGE_KEYS[2], ",".join(str(j) for j in op_max),
           ",".join(str(j) for j in group.max_by_seats), list(group.extra_waves(op_max)),
           list(group.extra_waves(group.max_by_seats)), list(op_max))]
    if by_demand:
        op_line.append(
            "%s: BEDARF (#251d) %s=1 -- keine Tabelle je Sitzzahl: jede Stufe %s bei jeder "
            "Sitzzahl, der Wake nimmt die kleinste, die die KV-Tokens seiner Sitze haelt "
            "(%s geschrieben, im Rang nicht gelesen%s); Zusatzwellen je bs %s in JEDER Phase "
            "(auch S0) -- der Preis, den der A/B-Boot misst"
            % (head, D_KV_STAGE_BY_DEMAND_KEY, list(tab.tokens), D_KV_STAGE_KEYS[2],
               "" if op_max is None else ", Operator-Wert %s ersetzt" % op_raw,
               list(group.extra_waves(max_by))))
    op_line.extend(waves_lines)
    if worker_tabs:
        # #239 S3g: the stage rows of every KV rank, booked as residency in S0
        op_line.extend(er.describe_kv_stage_residency(group, marker=D_RANK_SOLVE_MARKER,
                                                      label=label))
        op_line.append(
            "%s: JE KV-RANG (#239 S3g) %s=%s -- hoechste Stufe je Sitzzahl = MIN ueber die "
            "KV-Raenge %s (Host %s, %s); jeder Rang trimmt seine Voll-Attention-KV auf S0 und "
            "schneidet am Wake seine eigenen Stufenzeilen"
            % (head, D_KV_STAGE_ROWS_BY_RANK_KEY, items[D_KV_STAGE_ROWS_BY_RANK_KEY],
               list(group.max_by_seats), list(tab.max_by_seats),
               ", ".join("Rang %d %s" % (t.host_rank, list(t.max_by_seats))
                         for t in worker_tabs)))
    return [
        "%s: --env-d %s (Stufen %s Tok, KV-Zelle %d B/Tok, Stufenzeile %.1f MiB: "
        "S1..S%d nehmen %s Zeilen; TP0 Scratch %d -> %d, Sitzzeilen %s -> %s -- die "
        "Stufenzeilen sind in S0 LRU-Zeilen, der Boot mappt dieselbe Bank)"
        % (head, ";".join("%s=%s" % kv for kv in items.items()), list(tab.tokens),
           tab.kv_cell_bytes, tab.row_mib, len(tab.tokens) - 1, list(tab.stage_rows[1:]),
           tab.scratch, tab.scratch - tab.rows, seat_raw or "0",
           items["FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS"]),
        "%s: hoechste Stufe je Sitzzahl %s (Voreinstellung ohne Zusatzwelle -- Variante B, "
        "bis die Wellenmessung eine Zusatzwelle preist); Zeilen je (n, S0/S1/S2) %s gegen "
        "den Bedarf %s; der Capture zaehlt je bs die wenigsten Zeilen, die eine erlaubte "
        "Phase hat (#251 CAPTURE-FLOOR)" % (head, list(group.max_by_seats), caps,
                                            list(tab.need)),
        "%s: A/B-Kandidat Zwischenform (S1 bis zur Kappe) FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS=%s "
        "-- Zusatzwellen je bs %s (Default: %s); nicht geschrieben, fuer --env-d im A/B"
        % (head, ",".join(str(j) for j in cand), list(cand_extra),
           list(group.extra_waves(group.max_by_seats))),
    ] + op_line


def d_kv_stage_undo(ns) -> None:
    """#251c: take back this launcher's own stage form (a second solve pass
    prices the bank as priced and writes the form again)."""
    w = getattr(ns, "_d_kv_stage_written", None)
    if not w:
        return
    vram_view().note_d_stages(None)
    env_raw = str(getattr(ns, "env_d", "") or "")
    env = parse_group_env(env_raw)
    h, rows_ = int(w["host"]), int(w["rows"])
    scratch = [int(float(x)) for x in env.get("FLLIPER_MOE_SCRATCH_SLOTS", "").split(",")
               if x.strip()]
    given = {h: rows_}
    given.update({int(r): int(x) for r, x in (w.get("worker_rows") or {}).items()})
    if any(r < len(scratch) for r in given):
        for r, x in given.items():
            if r < len(scratch):
                scratch[r] += x
        env_raw = set_group_env(env_raw, "FLLIPER_MOE_SCRATCH_SLOTS",
                                ",".join(str(x) for x in scratch))
    if w.get("seat_raw") is None:
        env_raw = _drop_group_env(env_raw, "FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS")
    else:
        env_raw = set_group_env(env_raw, "FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS", str(w["seat_raw"]))
    for k in D_KV_STAGE_KEYS + (D_KV_STAGE_ROWS_BY_RANK_KEY,):
        env_raw = _drop_group_env(env_raw, k)
    if w.get("op_max") is not None:  # the operator's value is not the launcher's to take back
        env_raw = set_group_env(env_raw, D_KV_STAGE_KEYS[2], str(w["op_max"]))
    if "waves_raw" in w:  # #251d: the wave cap this form raised goes back too
        from flliper.srt.planner.expert_residency import POOL_OVERFLOW_WAVES_ENV

        env_raw = (_drop_group_env(env_raw, POOL_OVERFLOW_WAVES_ENV) if w["waves_raw"] is None
                   else set_group_env(env_raw, POOL_OVERFLOW_WAVES_ENV, str(w["waves_raw"])))
    ns.env_d = env_raw
    ns._d_kv_stage_written = None


def d_seat_table_lines(ns, er, plan_kwargs, label) -> List[str]:
    """H95: the D-FRACTION-SOLVE once per seat count n = 1..seats (the SAME
    ``plan_d_residency`` with ``seats=n`` -- seat_rebook re-books the posts;
    no second pricing), rendered by ``expert_residency.describe_seat_table``.
    Informational: the boot is priced (and refused) at the upper bound by the
    solve above. Empty below two seats or without the pool mode / verify
    form; never raises."""
    try:
        seats = d_stated_seats(ns)
        form = d_replayssm_spec_plan_form(ns)
        env_d = plan_kwargs.get("env_d") or {}
        pool = str(env_d.get(er.POOL_GRAPH_MODE_ENV, "")).strip().lower() == "pool"
        if seats is None or int(seats) < 2 or form is None or not pool:
            return []
        import json
        import os

        import msgspec

        with open(model_config_path(ns.model)) as fh:  # a GGUF FILE: its sibling config.json
            cfg = json.load(fh)
        top_k = int((cfg.get("text_config") or cfg).get("num_experts_per_tok") or 0)
        if top_k <= 0:
            return []

        plans = {}

        def plan_for(n):
            if n not in plans:
                plans[n] = er.plan_d_residency(
                    **plan_kwargs, seats=n,
                    replayssm_spec=msgspec.structs.replace(form, max_running=n))
            return plans[n]

        seat_vram = d_seat_vram_plan_form(ns, er, plan_kwargs, cfg, form)
        rows = er.seat_table(
            plan_for, seats_max=int(seats), verify_tokens=int(form.draft_tokens),
            top_k=top_k, waves=er.pool_overflow_waves(env_d), seat_vram=seat_vram)
        out = list(er.describe_seat_table(rows, marker=D_RANK_SOLVE_MARKER, label=label))
        if seat_vram is not None:
            out.extend(apply_d_seat_expert_rows(ns, er, rows, seat_vram, label))
            out.extend(apply_d_kv_stage_form(
                ns, er, rows, seat_vram, plan_for(int(seats)), label,
                verify_tokens=int(form.draft_tokens), top_k=top_k,
                kv_token_shares=plan_kwargs.get("kv_token_shares")))
        return out
    except PdFlipDKvStageMaxRefused:
        raise  # the operator's A/B arm does not fit: named stop, not a quiet default
    except Exception as exc:  # noqa: BLE001 -- an informational table never kills a launch
        return ["%s FRACTION-SOLVE %s D-SITZE (H95) Tabelle entfaellt: %s: %s"
                % (D_RANK_SOLVE_MARKER, label, type(exc).__name__, exc)]






















#: FLLIPER_PDFLIP_VENV: see the rig-path block at the top (--venv still wins).
VENV_DEFAULT = os.environ.get("FLLIPER_PDFLIP_VENV") or "/spinning/htsglang-gpu/.venv"
#: WEG2_SCHEDULING_SPEC_0907 C10/K5 -- the RECORDED break-even inputs, used
#: only when this rig's own logs carry none.  Measured pair from record
#: 1l/1o (boot weg2zr2, tip 7e3a9150b4): D's realised single-prefill rate at
#: the fast end of 610-690 tok/s, P's leg-1 rate ~3,640 tok/s, and the
#: SHORTER of the two measured flips (13.247 s of 13.2-16.1) -- the
#: conservative corner of the range in all three terms, which is why the
#: value it produces is ~22,000 rather than the 27,200 the other corner
#: gives.  STAMPED PRE-BARLINK: record 1n measured D at 1,354 tok/s after
#: the barlink fix, which pushes X to ~69,000, and the flipcost build pulls
#: it back; the launcher recomputes from this boot's lines and this pair is
#: only the fallback (O4 -- do not freeze a table).
X_RECORDED_R_D_TOKS = _PROFILE_27B.constant("X_RECORDED_R_D_TOKS")
X_RECORDED_R_P_TOKS = _PROFILE_27B.constant("X_RECORDED_R_P_TOKS")
X_RECORDED_FLIP_S = _PROFILE_27B.constant("X_RECORDED_FLIP_S")


RATE_UNIT_REQUEST_LATENCY = "request_latency"
RATE_UNIT_COMPUTE_HONEST = "compute_honest"


from flliper.srt.pdflip.launcher_types import (  # noqa: F401 -- re-export, behaviour-neutral move
    BootState,
    BubbleMeasurement,
    Card,
    DEPTH_RETRACTION_RECORD,
    DOperatingPointRow,
    DTokenVectorDecision,
    DTpRatioDecision,
    DepthDecision,
    EarlyReadFact,
    GroupSpec,
    HostRingPlan,
    LaunchGuard,
    MixedRateUnits,
    PCutFacts,
    PFrCap,
    PdFlipCarrierFloorUnreachable,
    PdFlipChunkCardMismatch,
    PdFlipDEarlyStartUnreviewed,
    PdFlipDFormOffTheMap,
    PdFlipDKvStageMaxRefused,
    PdFlipDKvStageWavesRefused,
    PdFlipDualLayoutRefused,
    PdFlipDualPSleepShareRefused,
    PdFlipLaunchHostLatched,
    PdFlipLaunchRefused,
    PdFlipRingFormUnproven,
    PdFlipStoreArcRefused,
    PdFlipStoreDiskRefused,
    PdFlipTokenCutFlipNotWired,
    PdFlipTokenCutNotWired,
    XSeed,
)






















def dual_x_tokens(ns) -> Optional[int]:
    """DUAL-TP3PP3 (user decision 01.10.): D's X in the dual layout, or None outside it.

    Metal dual1m (...10011343, 13:47Z): the front still routed SHORT (uncached <= X=12288)
    straight to D in the dual layout -- 39 "SHORT -> D", D X-GATE admitted 2..3957 uncached
    tokens and a 2185-token extend stalled D's decode ~2.5 s. In the dual layout P prefills
    everything: X = 1 + --dual-d-prefill-tokens (default 0), the 1 being the N-1 anchor
    token D's first step computes. Bypasses resolve_x's floor (--chunked-prefill-size) on
    purpose: that floor is a flip-design break-even, and the dual layout never flips."""
    if not getattr(ns, "dual_layout", False):
        return None
    allowance = int(getattr(ns, "dual_d_prefill_tokens", 0) or 0)
    if allowance < 0:
        raise SystemExit(f"--dual-d-prefill-tokens {allowance} < 0: the D prefill allowance "
                         f"counts tokens beyond the N-1 anchor token; 0 = none")
    return 1 + allowance















#: The operating point both groups are launched at (`--context-length`), and
#: therefore the longest prompt group P can be asked to prefill. It is the
#: default floor for P's KV pool: P frees a request's rows once its prefill
#: completes, but DURING that prefill the whole prefix must be device-resident
#: for the stage's attention layers, so a pool below this cannot serve the
#: boot's own admitted maximum.
CONTEXT_LENGTH = CONTEXT_LENGTH_TOKENS

from flliper.srt.pdflip.launcher_d_set_r3_L0 import (  # noqa: F401 -- re-export, behaviour-neutral move
    MEASURED_MS_PER_LAYER,
    PROFILE_ARG_DEFAULTS,
    P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT,
    P_PP_STAGE_FIXED_MIB,
    apply_profile_arg_defaults,
)


# ---------------------------------------------------------------------------
# #1286 -- THE POSTS THE BOOT CHARGES, so the ranking prices what the boot
# sizes. Every constant below is READ OFF THE RUNTIME'S OWN SUCCESS-PATH LINE
# (model_runner_kv_cache_mixin.py:1174):
#
#   [world_rank 0] KV budget posts (GiB): weights + runtime state=17.908,
#     gapped corridor holdback=1.000, mamba state pool=1.005,
#     prefill activation reserve=1.000, GGUF dequant scratch=0.000 | rest=6.392
#
# TWO BOOTS, TWO DIFFERENT CUTS, so these are measurements and not a fit:
#   weg2sb5f  44,10,10 / attn 11,2,3  (2026-09-09, tip 11df059318)
#   weg2rg6   32,18,14 / attn 8,4,4   (2026-09-08, tip 7f88b1c75d; weg2sb4
#                                      reproduced it byte-identically)
# Recovering the same per-stage constant from both is the falsifiable part: a
# term that tracked layer count would move by thousands of MiB between them.
# ---------------------------------------------------------------------------







































#: The `gapped corridor holdback` post, MiB per rank. 1.000 GiB on every rank
#: of both boots. This is the term ARMING_FLOOR_MIB was standing in for, and
#: they are NOT the same thing: the runtime charges this holdback and no
#: arming floor, so the 1229 MiB stand-in over-charged by 205 MiB per rank.
#: That was the only one of the four errors pointing the SAFE way.
P_CORRIDOR_HOLDBACK_MIB = 1024.0





from flliper.srt.pdflip.launcher_d_apply_r2_L5 import (  # noqa: F401 -- re-export, behaviour-neutral move
    P_MAMBA_SLOTS_PER_RUNNING_REQUEST,
    _d_extend_vote_fixed_env,
    booked_rest_vote_line_deltas,
    p_mamba_slots,
)

#: #1273 S6: THE ON-CARD LANE'S ARM, spec section 3.7 degrade 3's own flag
#: (``--pdflip-xchg-oncard {ipc|host}``).  It had no producer at all: the ranks
#: read ``FLLIPER_PDFLIP_XCHG_ONCARD`` from their inherited environment and the
#: launcher never wrote it, so the ``host`` arm -- the only arm on which a
#: store-and-forward deposit can exist -- was reachable only by an operator
#: exporting a variable by hand, while the #1269 ledger charged its bytes on
#: every ``shadow`` boot (S6 refuter, must_fix 4).  One name, one producer, and
#: the SAME value decides the charge and the allocation.
ONCARD_MODE_CHOICES = (weight_exchange_transport.ONCARD_MODE_IPC,
                       weight_exchange_transport.ONCARD_MODE_HOST)






#: #1240 -- the DEPTH axis of the cost model, and the two records that pin it.
#:
#: A full-attention layer's cost for one chunk grows with the prefix it
#: attends over; a GDN layer's does not. One number per stage cannot carry
#: that, so the family split is solved from TWO measurements at TWO depths
#: (pp_cut.family_costs_from_measurement) and both are named here.
#:
#: CALIBRATION DEPTH. MEASURED_MS_PER_LAYER above was taken on boot bsscale,
#: whose driver sends a ~12,000-token prompt at --chunked-prefill-size 4096,
#: i.e. THREE full chunks (BSSCALE_0907.md: "A ~12 000-token prompt is 3
#: chunks"). Those chunks enter at prefix 0, 4096 and 8192, so the mean prefix
#: of the full chunks the ms/layer figures average over is 4096. Derived, not
#: chosen: it is the arithmetic mean of the census the calibration ran on.
CALIBRATION_PREFIX_TOKENS = 4096



#: PP-COST (01.10.): the stage model the 27B INT8 P cut is priced on under
#: --pp-cut-stage-model auto (rank-line fit of 09290020 cut 43,11,10 and four
#: 44,10,10 boots of 01.10., pdflip/p_stage_model_data/)
PP_CUT_STAGE_MODEL_27B_INT8 = "27b_int8_rc12z30"
#: FP8 (27.10., desk/fp8-optima-1027): rank-line fit (``p_stage_model fit``) of boot dkr27bfp8bar1drei10271317 (cut 44/11/9, 512
#: graph, W8A16 Marlin on every card). Per layer and 512-token forward the 5090 costs 2.55 ms and a 3080 7.27 ms (INT8: ~1.0 / ~3.1),
#: so the INT8-priced solve left PP0 the bottleneck (44,11,9: PP0 177 ms against PP1 116 / PP2 137 ms at prefix 65k, berechnet).
PP_CUT_STAGE_MODEL_27B_FP8 = "27b_fp8_drei1027"
#: HW-GENERIC 1002: the stage models --pp-cut-stage-model auto may pick,
#: keyed by (profile, weight format, the card inventory the fit was measured
#: on, in P-stage order). Replaces the profile-NAME branch: a model is
#: selected only where profile, format AND cards are the ones it measured.
PP_CUT_STAGE_MODELS = {
    ("qwen27b", "int8", card_identity.REFERENCE_INVENTORY): PP_CUT_STAGE_MODEL_27B_INT8,
    ("qwen27b", "fp8", card_identity.REFERENCE_INVENTORY): PP_CUT_STAGE_MODEL_27B_FP8,
}


def _live_inventory_or_none() -> Optional[Tuple[str, ...]]:
    """The live P-stage inventory signature, or None (NVML cannot answer)."""
    try:
        return card_identity.inventory_signature(order_cards(resolve_cards()))
    except Exception:  # noqa: BLE001 - selection degrades to 'no model', never a refusal here
        return None


def resolve_pp_cut_stage_model(arg: str, profile: Optional[str], model: str,
                               inventory: Optional[Sequence[str]] = None):
    """(LayerCostModel, provenance) for --pp-cut-stage-model, or (None, why).

    ``inventory``: the live card-class signature in P-stage order (default:
    read from NVML; ``None`` when NVML cannot answer -- then only a model
    whose key does not depend on the cards could match, i.e. none)."""
    from flliper.srt.pdflip import p_stage_model as _psm

    arg = str(arg or "").strip()
    if arg in ("", "off"):
        return None, "--pp-cut-stage-model off"
    if arg == "auto":
        fmt = pdflip_form.format_of(profile, model)
        inv = tuple(inventory) if inventory is not None else _live_inventory_or_none()
        key = (str(profile or ""), fmt, inv)
        picked = PP_CUT_STAGE_MODELS.get(key)
        if picked is None:
            known = "; ".join("%s/%s on [%s]" % (k[0], k[1], ",".join(k[2])) for k in PP_CUT_STAGE_MODELS)
            return None, ("--pp-cut-stage-model auto: no stage model for profile %r format %r on cards "
                          "[%s] (calibrated: %s) -- previous pricing" % (
                              profile, fmt or "?", ",".join(inv) if inv else "unknown", known))
        arg = picked
    path = arg if arg.endswith(".json") or os.sep in arg else os.path.join(
        os.path.dirname(os.path.abspath(_psm.__file__)), "p_stage_model_data", arg + ".json")
    model_obj = _psm.load_model(path)
    return model_obj, "%s (%s)" % (os.path.basename(path), model_obj.source[:160])


ATTN_ANCHOR_PREFIX_TOKENS = 262144

#: DESIGN DEPTH FALLBACK. When no boot log carries a prefill census the design
#: prefix is one chunk -- the shallowest depth the boot can actually run -- and
#: the launcher PRINTS that it fell back, so a table read at 4096 is never
#: mistaken for a table read at this rig's real mean prefix.
DESIGN_PREFIX_FALLBACK_TOKENS = 4096

#: #1240/#1030 decode defaults, MEASURED on boot weg2pp1.
D_NUM_CONTINUOUS_DECODE_STEPS = 2

#: THE PROVENANCE OF THAT 2, WRITTEN ONCE (#1030).
#:
#: It used to be written three times -- in this constant's comment, in
#: --num-continuous-decode-steps' help, and in the boot's SCHEDULER: line --
#: and the three copies had already drifted: only the comment carried the
#: CONFOUND that the measuring boot recorded about its own headline number.
#: That is the shape the operator's own briefing rule names (a paragraph
#: retyped into N places degrades by the third); a provenance sentence that
#: degrades is worse than none, because it reads as evidence.
#:
#: So there is one sentence and three readers. The help escapes its percent
#: signs (argparse evaluates every help string as `help % params`, so one bare
#: '%' makes --help raise for ALL flags); the log line and this comment take it
#: verbatim.
#:
#: WHAT THE RECORD /spinning/gpu-arb/weg2/BOOT_weg2pp1_0907.md ACTUALLY
#: SUPPORTS, arms table rows D0-D3:
#:   * D2 (overlap ON, steps 2, extra_buffer) 75.5 tok/s bs1 / 305.0 bs6
#:     against the D0 control (overlap OFF, steps 1, no_buffer) 66.2 / 284.9 =
#:     +14.0 % / +7.1 %. That is the PAIR, and it is what the default buys.
#:   * It is NOT an overlap-alone number and the record says so in its own
#:     words: "The overlap delta is CONFOUNDED and must not be quoted as an
#:     overlap-alone number" -- D0 runs --mamba-radix-cache-strategy no_buffer
#:     and D1/D2/D3 run extra_buffer, so D1-D0 is overlap+buffer jointly, and
#:     it measured NEGATIVE: 61.4 vs 66.2 = -7.35 % at bs1, -1.68 % at bs6.
#:   * The UNCONFOUNDED half is the steps knob alone (D1 -> D2 -> D3, all
#:     overlap ON, all extra_buffer): D2 vs D1 = +23.1 % bs1 / +8.9 % bs6, and
#:     D3 (steps 4) regresses to 73.6 / 295.6. So 2 is an optimum, not a
#:     direction, and the steps knob is the half carrying the pair's gain.
D_DECODE_STEPS_PROVENANCE = (
    "MEASURED, not chosen (boot weg2pp1, /spinning/gpu-arb/weg2/"
    "BOOT_weg2pp1_0907.md arms table). THE PAIR overlap ON + "
    "--num-continuous-decode-steps 2 + the paired extra_buffer mamba strategy "
    "(row D2) reads 75.5 tok/s at bs1 and 305.0 at bs6 against the D0 control "
    "(overlap OFF, steps 1, no_buffer) 66.2 / 284.9 = +14.0 % / +7.1 %. THAT "
    "IS A PAIR AND NOT AN OVERLAP NUMBER, in the record's own words: 'The "
    "overlap delta is CONFOUNDED and must not be quoted as an overlap-alone "
    "number' -- D1 vs D0 moves the buffer strategy with the schedule and "
    "measured NEGATIVE, 61.4 vs 66.2 = -7.35 % at bs1. The UNCONFOUNDED half "
    "is the steps knob alone (D1 -> D2 -> D3, all overlap ON, all "
    "extra_buffer): D2 vs D1 = +23.1 % bs1 / +8.9 % bs6, and row D3 (steps 4) "
    "REGRESSES to 73.6 / 295.6 -- so 2 is an optimum, not a direction, and the "
    "steps knob is the half that carries the pair's gain."
)

#: #1017 GROUP D'S WEIGHT-VECTOR OBJECTIVE -- the knob, not a second solver.
#:
#: ``--rank-tp-ratio auto`` is the CAPACITY-FIRST split (server_args.py:838
#: _CAPACITY_FIRST_DEFAULT_NOTICE: "the weights are proportional to each
#: rank's VRAM budget ... deliberately independent of how fast the cards
#: are"). Under the maxkv law that is the right DEFAULT, but it was reaching
#: group D's argv as a bare literal, so no boot ever stated which objective it
#: had chosen or what the other one costs. This is the knob that states it.
#:
#: The launcher does NOT solve the vector. Both arms are upstream mechanisms
#: and stay upstream: 'maxkv' emits ``--rank-tp-ratio auto`` (resolved by
#: ServerArgs._resolve_auto_rank_tp_ratio) and 'speed' emits
#: ``--rank-tp-ratio auto-performance --rank-perf-tune <target>`` (resolved by
#: uneven_perf.apply_auto_performance). What the launcher adds is the CHOICE,
#: priced, on one line -- ``d_tp_ratio_decision``.
#: #1241 slice (2). THE OPERATING-POINT AXIS OF THE SAME #1017 SOLVE.
#:
#: ``maxkv`` and ``speed`` answer "what does this boot optimise for". They do
#: NOT answer "at which decode operating point", and the two answers are not
#: the same vector: a bs=1 decode round is BANDWIDTH-bound (every rank streams
#: its weight shard once per token, so the barrier is
#: ``max_r bytes_r / bandwidth_r``) while a bs=6 round is COMPUTE-bound (the
#: same shard is read once for six tokens, so the barrier is
#: ``max_r flops_r / gemm_r``). The 5090's measured advantage over a 3080 is
#: 2.32x on bandwidth and 3.99x on GEMM (this rig's own card-rate library), so
#: the two operating points do not want the same split, and until now no boot
#: could state which of them the shipped vector serves.
#:
#: NO SECOND SOLVER, exactly as the two older arms. Both new positions are
#: priced with the runtime's OWN cost model -- ``PerfCostModel.
#: decode_round_time`` / ``per_rank_decode_times`` for the bandwidth arm,
#: ``prefill_lockstep_compute_time`` over the #324 per-(rank, family) GEMM
#: scores for the compute arm, ``predict_capacity`` for the pool -- and what
#: the launcher adds is the CHOICE plus a line that prints all three vectors
#: beside each other so the trade is readable per boot instead of per
#: archaeology.
D_TP_OBJECTIVE_CHOICES = ("maxkv", "speed", "decode-bs1", "decode-bs6")


#: The two positions that emit an EXPLICIT weight vector rather than one of
#: the runtime's two symbolic resolvers.
D_OPERATING_POINTS = ("decode-bs1", "decode-bs6")

#: #705, MEASURED, quoted with the commits that measured it. The desk half of
#: #705 (``05baa99213897ed9a5ae987e67938a92c648baf0``) priced a family-split
#: TP decode to a single threshold: the net is positive if and only if a
#: BLOCKING TP all-reduce costs more than 14.3 us on this rig. The verdict
#: commit (``d937d5f76b03a0f3c5ecb916649a97245acf0148``) then measured that
#: all-reduce at 31.0-33.7 us for the 10 KB bs=1 payload (INTEGRATION_R3
#: __group__ rows) -- i.e. above the threshold -- and REFUSED the family split
#: anyway, because the gate is a desk net and the split would need a
#: per-family vector that has no flag.
#:
#: Carried here as the collective term of the bs=1 row: at bs=1 the round is
#: bandwidth-bound and this all-reduce is charged ONCE per layer per token, so
#: it is the part of the round that a weight vector cannot move. Printing it
#: beside the barrier is what keeps a reader from reading the whole barrier
#: delta as available.
D705_AR_BS1_US = (31.0, 33.7)
D705_FAMILY_SPLIT_BREAKEVEN_US = 14.3
D705_PROVENANCE = (
    "#705 desk price 05baa99213 (family split net positive iff a blocking TP "
    "all-reduce > %.1f us) against the measured %.1f-%.1f us at the 10 KB bs=1 "
    "payload, d937d5f76b (INTEGRATION_R3 __group__ rows) -- which is why the "
    "family split is REFUSED and only the head vector moves here."
    % (D705_FAMILY_SPLIT_BREAKEVEN_US, D705_AR_BS1_US[0], D705_AR_BS1_US[1])
)








































































from flliper.srt.pdflip.launcher_log_plan import (  # noqa: F401 -- re-export, behaviour-neutral move
    Log,
    StoreDiskPlan,
)


# --------------------------------------------------------------------------
# NVML
# --------------------------------------------------------------------------


#: HW-P1a 1003: the ``--cards`` selection of the running launch (NVML
#: indices), set ONCE at the top of :func:`main` and read by every
#: :func:`resolve_cards` call of that launch (the boot path and the helpers
#: that re-read the card list: p_stage_live_classes, p_stage_power_current,
#: _live_inventory_or_none). None = every NVML device (today's behaviour).
_CARD_SELECTION: Optional[Tuple[int, ...]] = None




def set_card_selection(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Install the launch's ``--cards`` selection (None clears it)."""
    global _CARD_SELECTION
    _CARD_SELECTION = parse_card_selection(text)
    return _CARD_SELECTION




def selected_nvml_cards() -> List[Card]:
    """The launch's cards from the registry's ONE NVML reader, restricted to
    the ``--cards`` selection (:data:`_CARD_SELECTION`) -- WITHOUT the arch
    gate (:func:`resolve_cards` adds it; the simulation harness reads the
    ungated list to name every blocker of a refused inventory)."""
    cards = [
        Card(d.index, d.uuid, d.name, d.total_mib, reserved_mib=d.reserved_mib,
             cc=getattr(d, "compute_capability", None),
             bar1_total_mib=(None if getattr(d, "bar1_total_bytes", None) is None
                             else int(d.bar1_total_bytes) // (1 << 20)),
             pcie_max_gen=getattr(d, "pcie_max_gen", None),
             pcie_max_width=getattr(d, "pcie_max_width", None),
             mem_bus_width_bits=getattr(d, "mem_bus_width_bits", None),
             mem_clock_max_mhz=getattr(d, "mem_clock_max_mhz", None))
        for d in nvml_registry.list_devices()
    ]
    return select_cards(cards, _CARD_SELECTION)


def resolve_cards() -> List[Card]:
    """The rig's cards, from the registry's ONE NVML reader.

    Was a second pynvml transcript here (init / getCount / getHandle / decode /
    shutdown).  ``registry.nvml.list_devices`` is that transcript plus the v2
    carve-out term, which this launcher now needs, so the copy is gone rather
    than grown.

    HW-P1a 1003: restricted to the launch's ``--cards`` selection
    (:data:`_CARD_SELECTION`), BEFORE the arch gate -- a card outside the
    selection is not this launch's card and is not judged.
    """
    cards = selected_nvml_cards()
    # HW-GENERIC 1002: THE arch gate, once, at the one card-list producer --
    # every boot path (main, p_stage_power_current, xchg_census) reads its
    # cards here. sm_86 / sm_89 / sm_120 pass (sm_89 UNCALIBRATED: it reaches
    # the named HW-UNCALIBRATED path in inventory_check_line, not a borrow);
    # anything else refused BY NAME (SM89-DURCHSPIEL-1002).
    try:
        card_identity.arch_gate(cards)
    except card_identity.CardInventoryRefused as exc:
        raise PdFlipLaunchRefused(str(exc)) from exc
    return cards




#: HW-GENERIC 1002: the launcher flags (dest names) and the --extra-p/-d,
#: --env-p/-d tokens that carry a POSITIONAL per-card vector (ordinal i =
#: the i-th card of order_cards). Each is a measurement or a pin of the
#: inventory the profile was written for (HW-KOPPLUNG-AUDIT-1002.md 2).
POSITIONAL_VECTOR_FLAGS = (
    "d_foreign_context_mib", "d_nontorch_mib", "pp_stage_ratio", "pp_attn_stage_ratio",
    "pp_cut_expert_device_fraction", "pp_cut_expert_lru_rows", "user_reserve_mib",
    "d_reserve_mib", "pp_cut_reserve_mib", "d_reshard_presets", "p_barlink_bar1_window_mib",
)
POSITIONAL_VECTOR_TOKENS = (
    "--rank-role", "--rank-tp-ratio", "--rank-moe-ratio", "--rank-moe-resident-fraction",
    "--rank-user-reserve-mib", "--rank-gpu-memory-mib", "--pp-stage-ratio",
    "--pp-attn-stage-ratio", "FLLIPER_MOE_SCRATCH_SLOTS=", "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION=",
    "FLLIPER_PDFLIP_L15_MIB=", "FLLIPER_PDFLIP_EXTEND_TRIM_MIB=",
)


def positional_inputs_present(ns) -> List[str]:
    """The positional per-card inputs this launch carries (flag dests set to
    something other than their parser default + vector tokens found in
    --extra-p/-d and --env-p/-d)."""
    out: List[str] = []
    defaults = build_parser().parse_args(["--tree", "/", "--tag", "x"])
    for dest in POSITIONAL_VECTOR_FLAGS:
        v = getattr(ns, dest, None)
        if v not in (None, "", [], ()) and v != getattr(defaults, dest, None):
            out.append("--" + dest.replace("_", "-"))
    blob = " ".join(str(getattr(ns, k, "") or "") for k in ("extra_p", "extra_d", "env_p", "env_d"))
    for tok in POSITIONAL_VECTOR_TOKENS:
        if tok in blob and tok.rstrip("=") not in out:
            out.append(tok.rstrip("="))
    env_l15 = os.environ.get("FLLIPER_PDFLIP_L15_MIB", "")
    if env_l15 and env_l15.strip().lower() != "auto" and "FLLIPER_PDFLIP_L15_MIB" not in out:
        out.append("FLLIPER_PDFLIP_L15_MIB")
    return out






def derivation_assessment(ns, cards: Sequence["Card"],
                          environ: Optional[Mapping[str, str]] = None):
    """HW-P1c 1003: can the profile's positional records and vectors be DERIVED
    for the live inventory (pdflip/inventory_view.py)? ``None`` when there is
    nothing to derive (the live inventory IS the calibrated one, or the
    operator named the vectors' inventory with ``--profile-inventory``);
    else ``(subset, records_ok, underivable_records, underivable_vectors,
    l15_underivable)``."""
    from flliper.srt.pdflip import inventory_view as _iv
    from flliper.srt.pdflip import profile_records as _pr

    env = os.environ if environ is None else environ
    prof = str(getattr(ns, "profile", None) or pdflip_form.DEFAULT_PROFILE)
    rec_inv, _ = records_inventory(prof)
    vec_inv, vec_src = calibrated_inventory(ns)
    live = tuple(card_identity.inventory_signature(cards))
    if vec_src == "--profile-inventory" or live == tuple(rec_inv):
        return None
    pos = [(r.name, r.value) for r in _pr.records(prof) if _pr.is_positional(r.value, len(rec_inv))]
    asm = _iv.assess_records(pos, tuple(rec_inv), live)
    vec_bad = sorted(k for k, c in positional_vector_lengths(ns).items()
                     if int(c) != len(live) and not _iv.vector_derivable(k, int(c), tuple(vec_inv), live))
    l15_bad = []
    if l15_plan is not None and l15_plan.master_on(env):
        mib = str(env.get(l15_plan.L15_MIB_ENV, "") or "")
        try:
            mode, posts = l15_plan.parse_l15_mib(mib)
        except ValueError:
            mode, posts = "auto", {}
        if mode == "override" and not _iv.l15_derivable(mib, tuple(rec_inv), live):
            l15_bad = [l15_plan.L15_MIB_ENV]
    return asm.subset, asm.derivable + asm.exempt, list(asm.underivable), vec_bad, l15_bad


def inventory_check_line(ns, cards: Sequence["Card"],
                         environ: Optional[Mapping[str, str]] = None) -> str:
    """The HW-INVENTORY line; raises PdFlipLaunchRefused (HW-UNCALIBRATED) when
    the live ordered inventory is not the one the profile's positional
    measured RECORDS were taken on, or not the one its positional VECTORS
    (argv/env, ``--profile-inventory``) were written for. Two checks: a
    profile that names a foreign inventory does not make the reference
    rig's records hold there.

    HW-P1c 1003: a live inventory that is a SUBSET of the calibrated one (every
    live card has its own measured twin of its class, e.g. 5090 + 3080 or
    3080 + 3080 of [RTX5090, RTX3080, RTX3080]) is not refused when every
    positional value has a derivation (pdflip/inventory_view.py): the line says
    DERIVED and :func:`apply_inventory_derivation` installs the values. Only
    what cannot be derived is refused, and only that is named."""
    from flliper.srt.pdflip import profile_records as _pr

    prof = str(getattr(ns, "profile", None) or pdflip_form.DEFAULT_PROFILE)
    rec_inv, rec_src = records_inventory(prof)
    vec_inv, vec_src = calibrated_inventory(ns)
    rec_what = sorted({r.name for r in _pr.records(prof) if _pr.is_positional(r.value, len(rec_inv))})
    vec_what = positional_inputs_present(ns)
    msgs = []
    if vec_src == "--profile-inventory":
        m = card_identity.uncalibrated_message(cards, vec_inv, vec_what,
                                               f"profile {prof!r} vectors ({vec_src})")
        if m is not None:
            msgs.append(m)
        m = card_identity.uncalibrated_message(cards, rec_inv, rec_what,
                                               f"profile {prof!r} measured records ({rec_src})")
        if m is not None:
            msgs.append(m)
    else:
        da = derivation_assessment(ns, cards, environ)
        what = rec_what + vec_what
        if da is not None:
            subset, ok, bad_rec, bad_vec, bad_l15 = da
            if subset and not (bad_rec or bad_vec or bad_l15):
                return ("HW-INVENTORY " + "; ".join(f"ordinal {i}: {card_identity.describe(c)}"
                                                    for i, c in enumerate(cards))
                        + f" -- calibrated inventory [{','.join(rec_inv)}] ({rec_src}): the live cards "
                        f"are a SUBSET of it (each has a measured twin of its class), positional "
                        f"values DERIVED not measured: records {', '.join(ok) or '(none)'}; vectors "
                        f"{', '.join(sorted(positional_vector_lengths(ns))) or '(none)'}"
                        + ("; L15 posts" if l15_plan is not None and l15_plan.master_on(os.environ if environ is None else environ) else "")
                        + " (pdflip/inventory_view.py; the first boot on these cards is the "
                        "measurement) DERIVED")
            if subset:
                what = bad_rec + bad_vec + bad_l15
        m = card_identity.uncalibrated_message(cards, rec_inv, what,
                                               f"profile {prof!r} ({rec_src})")
        if m is not None:
            msgs.append(m)
    if msgs:
        # PROFIL-EDITOR S1: --force lifts HW-UNCALIBRATED (a value refusal); the line then says MISMATCH
        refusals.refuse_value("HW-UNCALIBRATED", " || ".join(msgs), PdFlipLaunchRefused)
    return ("HW-INVENTORY " + "; ".join(f"ordinal {i}: {card_identity.describe(c)}"
                                        for i, c in enumerate(cards))
            + f" -- calibrated inventory [{','.join(rec_inv)}] ({rec_src}"
            + (f"; vectors: {vec_src}" if vec_src == "--profile-inventory" else "")
            + (") MISMATCH (--force)" if msgs else ") MATCH"))








_derive_csv_strict = _derive_csv




def apply_inventory_derivation(ns, cards: Sequence["Card"], log=None,
                               environ: Optional[Dict[str, str]] = None) -> List[str]:
    """HW-P1c 1003 (PROFILE-VECTORS / RECORDS-NVEC / L15-POSTS / PP-CUT-PIN):
    install the values :func:`inventory_check_line` called DERIVED, for a live
    inventory that is a subset of the calibrated one. ONE ``HW-DERIVE`` line
    per value (the old text, the new text, the rule); the profile's records
    are read derived from here on (``form.profile_constant``). Returns the
    lines (also ``log``-ged). No-op for the calibrated inventory itself (the
    reference rig: nothing changes) and for an underivable one (the check
    already refused)."""
    from flliper.srt.pdflip import inventory_view as _iv

    env = os.environ if environ is None else environ
    lines: List[str] = []

    def note(msg: str) -> None:
        lines.append(msg)
        if log is not None:
            log(msg)

    da = derivation_assessment(ns, cards, env)
    if da is None:
        _iv.clear_active()
        return lines
    subset, _ok, bad_rec, bad_vec, bad_l15 = da
    prof = str(getattr(ns, "profile", None) or pdflip_form.DEFAULT_PROFILE)
    cal, _ = records_inventory(prof)
    vcal, _ = calibrated_inventory(ns)
    live = tuple(card_identity.inventory_signature(cards))
    borrow = None
    if not subset or bad_rec or bad_vec or bad_l15:
        if not refusals.forced_boot():
            _iv.clear_active()
            return lines
        # AP2 1006 (Q-710, flags instead of code): a boot FORCED past HW-UNCALIBRATED
        # used to keep the reference rig's 3-vectors (N = 2 read the first two
        # entries silently, N = 4 died at the 4th card: AP1 (c)). It now BORROWS
        # every positional record for the live cards (inventory_view.borrow_vector:
        # class-max by class or AP1 arch twin, else the stage/rank ROLE) and names
        # each one UNMEASURED; what has no borrow stays as it was.
        borrow, twins = _forced_record_borrow(prof, cal, live, cards)
        _iv.set_active(prof, tuple(cal), live, borrow=sorted(borrow), twins=twins)
    else:
        twins = {}
        _iv.set_active(prof, tuple(cal), live)
    n = len(live)
    defaults = build_parser().parse_args(["--tree", "/", "--tag", "x"])
    head = f"HW-DERIVE [{','.join(cal)}] -> [{','.join(live)}]"

    def _derive_csv(policy, text, calibrated, live_):   # noqa: F811 - the forced borrow shadows the module helper
        if borrow is None:
            return _derive_csv_strict(policy, text, calibrated, live_)
        got = _iv.borrow_vector(policy, str(text), tuple(calibrated), tuple(live_), twins)
        return None if got is None else (got if isinstance(got, str) else ",".join(str(x) for x in got))

    if borrow is not None:
        for name in sorted(borrow):
            note(f"{head} record {name} {borrow[name][0]} -> {borrow[name][1]} "
                 f"(FORCED BORROW, {_iv.RECORD_POLICY.get(name) or 'role'}; UNMEASURED on these cards)")
        refusals.refuse_value(
            card_identity.CODE_UNCALIBRATED,
            f"{card_identity.CODE_UNCALIBRATED}: profile {prof!r} positional records measured on "
            f"[{', '.join(cal)}] are BORROWED for [{', '.join(live)}] (--force): "
            + (", ".join(sorted(borrow)) or "(none)")
            + " -- class-max by card class (a class without a calibrated card: its arch twin), otherwise "
            "by stage/rank role (first, middle, last); UNMEASURED here, the first boot on these cards is "
            "the measurement (pdflip/inventory_view.py borrow_vector)",
            PdFlipLaunchRefused, log=log)
    # 1. launcher flags that carry a per-card vector or a pinned P cut
    for dest in _TOPOLOGY_VECTOR_FLAGS:
        flag = "--" + dest.replace("_", "-")
        pol = _iv.FLAG_POLICY.get(flag)
        v = getattr(ns, dest, None)
        if pol is None or v in (None, "", [], ()) or v == getattr(defaults, dest, None):
            continue
        cnt = _vector_len(v if isinstance(v, str) else ",".join(str(x) for x in v))
        if cnt is None or cnt == n:
            continue
        if pol == _iv.CUT_PIN:
            setattr(ns, dest, getattr(defaults, dest, None))
            note(f"{head} {flag} {v}: a P cut pinned for {cnt} stages does not hold for P = PP{n} "
                 "-> DROPPED, the planner's cut solver (planner/pp_cut.py) derives the cut")
            continue
        got = _derive_csv(pol, v, vcal, live)
        if got is not None:
            setattr(ns, dest, got)
            note(f"{head} {flag} {v} -> {got} ({pol} by card class)")
    # 2. record-fed launcher flags (the profile default put the 3-vector there)
    for dest, rec in _iv.RECORD_FLAGS.items():
        v = getattr(ns, dest, None)
        pol = _iv.RECORD_POLICY.get(rec)
        if v in (None, "") or pol is None:
            continue
        cnt = _vector_len(str(v))
        if cnt is None or cnt == n:
            continue
        got = _derive_csv(pol, str(v), cal, live)
        if got is not None:
            setattr(ns, dest, got)
            note(f"{head} --{dest.replace('_', '-')} {v} -> {got} (record {rec}, {pol})")
    # 3. --env-p/-d tokens (K=V;K=V): per-card vectors with a policy
    for key in ("env_p", "env_d"):
        spec = str(getattr(ns, key, "") or "")
        if not spec:
            continue
        envd = parse_group_env(spec)
        for tok, pol in _iv.FLAG_POLICY.items():
            if tok.startswith("--") or tok not in envd:
                continue
            cnt = _vector_len(envd[tok])
            if cnt is None or cnt == n:
                continue
            got = _derive_csv(pol, envd[tok], vcal, live)
            if got is not None:
                setattr(ns, key, set_group_env(getattr(ns, key, ""), tok, got))
                note(f"{head} --{key.replace('_', '-')} {tok}={envd[tok]} -> {got} ({pol} by card class)")
    # 4. cut pins inside --extra-p/-d words
    for key in ("extra_p", "extra_d"):
        blob = str(getattr(ns, key, "") or "")
        if not blob:
            continue
        try:
            words = shlex.split(blob)
        except ValueError:
            continue
        out: List[str] = []
        i = 0
        dropped = []
        while i < len(words):
            w = words[i]
            if w in ("--pp-stage-ratio", "--pp-attn-stage-ratio") and i + 1 < len(words):
                cnt = _vector_len(words[i + 1])
                if cnt is not None and cnt != n:
                    dropped.append(f"{w} {words[i + 1]}")
                    i += 2
                    continue
            out.append(w)
            i += 1
        if dropped:
            setattr(ns, key, " ".join(shlex.quote(x) for x in out))
            note(f"{head} --{key.replace('_', '-')}: {', '.join(dropped)} DROPPED (a cut pinned for another "
                 f"stage count; the cut solver derives it)")
    # 5. L15 ordinal posts (L1.5 is a 27B-line feature; the NF tree has no
    # l15_plan module, so the step is a no-op there)
    if l15_plan is not None:
        mib = str(env.get(l15_plan.L15_MIB_ENV, "") or "")
        if l15_plan.master_on(env) and mib:
            new, why = _iv.derive_l15_override(mib, cal, live)
            if why and new != mib:
                env[l15_plan.L15_MIB_ENV] = new
                note(f"{head} {l15_plan.L15_MIB_ENV} {mib!r} -> {new!r} ({why})")
    return lines






#: HW-P1a: launcher flag dests whose value is a per-card vector the topology
#: probe counts (POSITIONAL_VECTOR_FLAGS minus the BAR1 window, which is a
#: window spec "24,PP_0=96" -- BAR1-WINDOW names it -- and the d_reshard
#: presets, which are not per card).
_TOPOLOGY_VECTOR_FLAGS = tuple(f for f in POSITIONAL_VECTOR_FLAGS
                               if f not in ("p_barlink_bar1_window_mib", "d_reshard_presets"))
#: the --extra-*/--env-* tokens of POSITIONAL_VECTOR_TOKENS the probe counts
#: (L1.5 has its own probe; --rank-role/-ratio are vectors like the rest)
_TOPOLOGY_VECTOR_TOKENS = tuple(t for t in POSITIONAL_VECTOR_TOKENS if t != "FLLIPER_PDFLIP_L15_MIB=")




def positional_vector_lengths(ns) -> Dict[str, int]:
    """HW-P1a: every positional per-card vector this launch carries -> its
    entry count (launcher flags set off their default, and the vector tokens
    inside --extra-p/-d, --env-p/-d). Only a COUNT -- the values are judged
    by the inventory check."""
    out: Dict[str, int] = {}
    defaults = build_parser().parse_args(["--tree", "/", "--tag", "x"])
    for dest in _TOPOLOGY_VECTOR_FLAGS:
        v = getattr(ns, dest, None)
        if v in (None, "", [], ()) or v == getattr(defaults, dest, None):
            continue
        n = _vector_len(v if isinstance(v, str) else ",".join(str(x) for x in v))
        if n is not None:
            out["--" + dest.replace("_", "-")] = n
    for key in ("extra_p", "extra_d", "env_p", "env_d"):
        blob = str(getattr(ns, key, "") or "")
        if not blob:
            continue
        try:
            words = shlex.split(blob)
        except ValueError:
            words = blob.split()
        # --env-p/-d carry 'K=V;K=V' -- one assignment per word
        words = [x for w in words for x in w.split(";") if x]
        for i, w in enumerate(words):
            for tok in _TOPOLOGY_VECTOR_TOKENS:
                if tok.endswith("="):
                    if not w.startswith(tok):
                        continue
                    val = w[len(tok):]
                elif w == tok:
                    val = words[i + 1] if i + 1 < len(words) else ""
                elif w.startswith(tok + "="):
                    val = w[len(tok) + 1:]
                else:
                    continue
                n = _vector_len(val)
                if n is not None:
                    out.setdefault(tok.rstrip("="), n)
    return out


def topology_context(ns, environ: Optional[Mapping[str, str]] = None,
                     cards: Optional[Sequence["Card"]] = None) -> "_topo.TopologyContext":
    """HW-P1a: what of THIS launch the topology blockers depend on. ``cards``
    (card order): the live inventory and the BAR1 of each card (HW-P1c)."""
    env = os.environ if environ is None else environ
    prof = str(getattr(ns, "profile", None) or pdflip_form.DEFAULT_PROFILE)
    try:
        fmt = pdflip_form.format_of(prof, str(getattr(ns, "model", "") or ""))
    except Exception:  # noqa: BLE001 - an unknown format only drops the cut-pin probe
        fmt = ""
    return _topo.TopologyContext(
        profile=prof,
        weight_format=str(fmt or ""),
        dual=bool(getattr(ns, "dual_layout", False) or getattr(ns, "dual_share", False)),
        weight_source=str(getattr(ns, "pdflip_weight_source", "") or ""),
        l15=bool(l15_plan is not None and l15_plan.master_on(env)),
        l15_mib=(str(env.get(l15_plan.L15_MIB_ENV, "") or "") if l15_plan is not None else ""),
        vectors=positional_vector_lengths(ns),
        live_inventory=(tuple(card_identity.inventory_signature(cards)) if cards is not None else ()),
        vector_inventory=tuple(calibrated_inventory(ns)[0]),
        record_inventory=tuple(records_inventory(prof)[0]),
        bar1_mib=(tuple(getattr(c, "bar1_total_mib", None) for c in cards) if cards is not None else ()),
    )


def topology_check_line(ns, cards: Sequence["Card"],
                        environ: Optional[Mapping[str, str]] = None) -> str:
    """HW-P1a 1003: the HW-TOPOLOGY line of the launch's card count (from the
    inventory, not a constant); raises PdFlipLaunchRefused BY NAME --
    HW-TOPOLOGY (N outside 2..8) or HW-COUNT with the CONCRETE blockers of
    this launch -- when the count is not runnable. N = 3 passes as before."""
    try:
        t = _topo.plan_topology(len(cards), topology_context(ns, environ, cards))
    except _topo.TopologyRefused as exc:
        text = f"{exc} || visible: " + "; ".join(card_identity.describe(c) for c in cards)
        if str(exc).startswith(_topo.CODE_COUNT):
            # PROFIL-EDITOR S1: an N inside the range that is not proven / N-capable is a VALUE refusal (--force lifts it);
            # HW-TOPOLOGY (N outside the range: no flip topology exists at all) is never lifted
            refusals.refuse_value("HW-COUNT", text, PdFlipLaunchRefused, cause=exc)
            return f"HW-TOPOLOGY N={len(cards)}: not proven, started with --force: {str(exc)[:240]}"
        raise PdFlipLaunchRefused(text) from exc
    return _topo.topology_line(t)










# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------


#: #1275 FIX 2: THE LAUNCHER'S OWN KEY, set ONCE at mint time and read from
#: here by every RPC. Never re-read from the file per call: two reads are two
#: sources, and the second one is the one that goes stale.
_ADMIN_KEY: Optional[str] = None
#: The path, so teardown (and the refusal handler) can unlink it.
_ADMIN_KEY_FILE: str = ""

#: #1248: the LIVE BootState of whatever this process has spawned so far,
#: read back by cli()'s refusal handler AFTER main() has already unwound.
#: Same reason _ADMIN_KEY is a module global rather than a return value: the
#: handler runs outside main()'s frame, in the except clause of its caller,
#: so this is the only channel that survives the raise. Reset to None at the
#: top of every main() call and pointed at the real BootState the moment one
#: exists (before any group is spawned) -- from then on its .pids dict is the
#: SAME dict main() mutates, so the handler sees pids even if the refusal hit
#: before the next _write_state() flushed them to disk (front's pid has that
#: gap: set at one line, written a dozen lines later).
_ACTIVE_BOOT_STATE: BootState | None = None


def set_admin_key(key: Optional[str], path: str = "") -> None:
    """Arm every launcher RPC with this boot's key. Idempotent, one source."""
    global _ADMIN_KEY, _ADMIN_KEY_FILE
    _ADMIN_KEY = key or None
    _ADMIN_KEY_FILE = path or ""


def drop_admin_key_file() -> str:
    """Unlink this boot's key file. Returns what happened, never raises.

    #1275 FIX 2 (3): THE KEY DIES WITH THE BOOT. A per-boot key whose file
    outlives the boot is a stale secret lying in a shared directory naming a
    server that no longer exists -- and the next boot mints its own, so nothing
    ever reads it again. Called from BOTH exits: the `--teardown` path and the
    refusal/killer path.
    """
    path = _ADMIN_KEY_FILE
    if not path:
        return "no key file for this boot"
    try:
        os.unlink(path)
        return f"removed {path}"
    except FileNotFoundError:
        return f"already gone {path}"
    except OSError as e:
        return f"could NOT remove {path}: {e}"


def rpc_site_count() -> int:
    """How many call sites go through :func:`http`, counted from the source.

    #1275 FIX 2: THE CALLER ENUMERATION, MADE A NUMBER THE BOOT LOG PRINTS.
    Boot weg2sb5 died because #1275 secured the ROUTES and taught exactly ONE
    of the two clients to authenticate; the launcher's own `sleep_group` was
    never enumerated. Counting the sites here -- rather than trusting a
    hand-kept list -- is what makes a NEW call site visible, and the desk test
    asserts this count against the AST so a site added without the helper fails
    before a boot does.
    """
    import ast as _ast

    try:
        tree = _ast.parse(open(__file__, encoding="utf-8").read())
    except OSError:  # pragma: no cover - source always readable in practice
        return -1
    return sum(
        1
        for n in _ast.walk(tree)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id == "http"
    )


def http(method: str, url: str, body: Optional[dict] = None, timeout: float = 25.0):
    """THE ONE HTTP DOOR OUT OF THIS PROCESS, and the only place auth is added.

    #1275 FIX 2, and the class is worth naming: access control was introduced at
    the SERVER and wired into ONE of its TWO clients. The front authenticates
    its flip RPCs; the launcher sleeps group P during the START SEQUENCE, before
    the front process exists, and got 401 on
    `POST /release_memory_occupation` -- boot weg2sb5, 39 s after P READY.
    `/release_memory_occupation`, `/flush_cache`, `/abort_request` and
    `/resume_memory_occupation` are all `ADMIN_OPTIONAL`, which REQUIRES the
    admin key once one is configured.

    So the header is attached HERE, at the single door, rather than at each call
    site: a new RPC anywhere in this module is authenticated by construction,
    which is the only version of this fix that a future call site cannot
    silently miss. `/health` gets it too -- a bearer on a NORMAL route with no
    api_key configured is ignored, and a per-path allowlist here would be a
    second copy of http_server.py's decorators that drifts.
    """
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if _ADMIN_KEY:
        req.add_header("Authorization", f"Bearer {_ADMIN_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def wait_ready(port: int, pid: int, deadline_s: float, log: Log, name: str, proc: Optional[subprocess.Popen] = None) -> float:
    t0 = time.time()
    last = ""
    while time.time() - t0 < deadline_s:
        # proc.poll() reaps: a dead child is a ZOMBIE until reaped and
        # os.kill(pid, 0) still succeeds on it (measured 07:00Z: D died at
        # parse and the launcher kept waiting).
        if proc is not None and proc.poll() is not None:
            raise PdFlipLaunchRefused(f"group {name} pid {pid} died before READY (exit {proc.returncode}); see its log")
        if pid and not _alive(pid):
            raise PdFlipLaunchRefused(f"group {name} pid {pid} died before READY")
        code, body = http("GET", f"http://127.0.0.1:{port}/health", timeout=25)
        if code == 200:
            dt = time.time() - t0
            log(f"R1 READY group={name} port={port} after {dt:.1f} s")
            if name in ("P", "D"):
                boot_state_write(log, fields={f"groups.{name}.state": "ready",
                                              f"groups.{name}.ready_ts": round(time.time(), 3)},
                                 event=("group_ready", {"group": name, "port": port, "after_s": round(dt, 1)}))
            return dt
        last = f"{code} {body[:80]!r}"
        time.sleep(3)
    raise PdFlipLaunchRefused(f"group {name} not READY on :{port} after {deadline_s:.0f} s (last {last})")




# --------------------------------------------------------------------------
# Preflight (inherited from boot_855_train0901.sh)
# --------------------------------------------------------------------------






















def shm_residue_sweep(
    log: Log,
    tag: str,
    stamp: str,
    dry: bool,
    shm_dir: str = SHM_DIR,
    proc_root: str = "/proc",
    archive_root: str = SHM_ARCHIVE_ROOT,
    sem_unlink: Optional[Callable[[str], int]] = None,
) -> Dict[str, object]:
    """#1217 + #1233 fix 8: sweep THIS LINE'S dead /dev/shm residue, and only that.

    ONE authority for /dev/shm residue, replacing the #1217 presence-directory
    check it grew out of: that check swept one directory, called `fuser` once on
    its files, and answered "none" for a box holding 3.23 GiB of a dead boot's
    page store two entries away (weg2dk7).  Same defect class, one function.

    The rules, in order:

    * a name outside :data:`SHM_OWN_PREFIXES` is FOREIGN and is never listed,
      never stat'ed for size, never moved -- /dev/shm here is full of other
      people's `sem.mp-*`;
    * any live ``flliper.launch_server`` REFUSES the boot before anything is
      touched (another boot is up; this is not the moment to tidy /dev/shm).
      "Live server" is decided by :func:`live_launch_servers`, never by a
      substring of some command line (fnFL2 H26: the substring caught the
      operator's own start shell, an ancestor of this launcher, three times);
    * an entry with a live holder (:func:`shm_holder_pids`) REFUSES the boot,
      naming the entry and the pids.  Nothing is killed and nothing is swept;
    * an orphan REGULAR FILE is moved into the archive WITH ITS CONTENT: the
      #1217 presence flags are the evidence and they are bytes long;
    * an orphan DIRECTORY TREE is archived as a MANIFEST (name, bytes, entry
      count, mtime) and then removed.  Copying a dead boot's 3.2 GiB page store
      of ~100k small files onto spinning disk at preflight would cost minutes
      and buy nothing -- the evidence of a leaked store is that it existed and
      how big it was, not the pages inside it.
    * #1273 S3: an orphan POSIX NAMED SEMAPHORE (`sem.pdflip-xchg-*`) is
      `sem_unlink`ed rather than moved -- see the call to
      :func:`sweep_xchg_semaphores` below for why the order inside this
      function is the whole point.
    """
    try:
        names = sorted(os.listdir(shm_dir))
    except OSError:
        log(f"#1217/#1233 shm residue: {shm_dir} unreadable -- NOT swept, and not read as empty")
        return {"swept": [], "bytes_freed": 0, "refused": {}, "archive": ""}
    own = [n for n in names if any(n.startswith(p) for p in SHM_OWN_PREFIXES)]
    foreign = len(names) - len(own)
    if not own:
        log(f"#1217/#1233 shm residue: none of ours in {shm_dir} ({foreign} foreign entries untouched)")
        return {"swept": [], "bytes_freed": 0, "refused": {}, "archive": ""}
    live, ignored = live_launch_servers(tag, proc_root)
    if ignored:
        log(
            f"#1217 shm residue (fnFL2 H26): {len(ignored)} process(es) mention "
            f"{'|'.join(LAUNCH_SERVER_MODULES)} but are NOT a live foreign server, not counted: {ignored}"
        )
    if live:
        raise PdFlipLaunchRefused(
            f"#1217 shm residue: LIVE launch_server pid(s)=[{' '.join(str(s['pid']) for s in live)}] "
            f"{live} -- refusing this boot, not sweeping, not killing anything "
            f"(our /dev/shm entries: {own})"
        )
    held = {n: shm_holder_pids(os.path.join(shm_dir, n), proc_root) for n in own}
    refused = {n: pids for n, pids in held.items() if pids}
    if refused:
        raise PdFlipLaunchRefused(
            f"#1217/#1233 shm residue: LIVE HOLDER on our own /dev/shm entries "
            f"{ {n: pids for n, pids in sorted(refused.items())} } -- refusing this boot, not "
            f"sweeping, not killing anything"
        )
    # #1273 S3, review round 2 F1 -- THIS IS THE ONLY PLACE THE SEMAPHORE SWEEP
    # CAN RUN.  It has to be AFTER the two refusals above (a live boot is never
    # swept, and unlinking a live boot's handshake would be strictly worse than
    # leaving a dead one's behind) and BEFORE the move loop below, which would
    # otherwise `shutil.move` every `sem.pdflip-xchg-*` file into the archive and
    # leave the kernel object alive for a later `sem_open` to collide with.
    # Called from outside this function it can only ever find an empty
    # /dev/shm and log "residue: none" -- which is what it did.
    sem_gone = sweep_xchg_semaphores(log, shm_dir, dry, unlink=sem_unlink)
    if sem_gone:
        # sem_unlink removed the backing files; only survivors get archived.
        own = [n for n in own if os.path.exists(os.path.join(shm_dir, n))]
        if not own:
            log(f"#1217/#1233 shm residue: {len(sem_gone)} exchange semaphore(s) "
                f"sem_unlink'ed, nothing else of ours left in {shm_dir}")
            return {"swept": [], "bytes_freed": 0, "bytes_apparent": 0,
                    "refused": {}, "archive": "", "sems_unlinked": sem_gone}
    archive = f"{archive_root}/{tag}_{stamp}"
    sizes = {n: _tree_bytes(os.path.join(shm_dir, n)) for n in own}
    total = sum(a for a, _b, _n in sizes.values())
    apparent = sum(b for _a, b, _n in sizes.values())
    if dry:
        log(
            f"#1217/#1233 DRY-RUN: would sweep {len(own)} orphan entr(ies) from {shm_dir} -> "
            f"{archive}, freeing {total} allocated bytes = {total / host_ledger.GIB:.2f} GiB "
            f"(apparent {apparent / host_ledger.GIB:.2f} GiB -- allocated is what the tmpfs "
            f"occupies and what cgroup shmem counts); per entry (allocated): "
            f"{ {n: sizes[n][0] for n in own} }; {foreign} foreign entries untouched"
        )
        return {"swept": own, "bytes_freed": total, "bytes_apparent": apparent,
                "refused": {}, "archive": archive, "sems_unlinked": sem_gone}
    os.makedirs(archive, exist_ok=True)
    manifest: List[dict] = []
    for n in own:
        src = os.path.join(shm_dir, n)
        nbytes, napparent, count = sizes[n]
        try:
            mtime = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.lstat(src).st_mtime))
        except OSError:
            mtime = "?"
        is_dir = os.path.isdir(src) and not os.path.islink(src)
        manifest.append(
            {"name": n, "bytes_allocated": nbytes, "bytes_apparent": napparent,
             "entries": count, "mtime": mtime,
             "kind": "dir" if is_dir else "file", "action": "manifest+removed" if is_dir else "moved"}
        )
        if is_dir:
            shutil.rmtree(src, ignore_errors=True)
        else:
            shutil.move(src, archive)
    with open(os.path.join(archive, "MANIFEST.json"), "w") as f:
        json.dump({"tag": tag, "stamp": stamp, "shm_dir": shm_dir, "entries": manifest}, f, indent=1)
    log(
        f"#1217/#1233 shm residue swept: {len(own)} orphan entr(ies), "
        f"{total} allocated bytes = {total / host_ledger.GIB:.2f} GiB freed "
        f"(apparent {apparent / host_ledger.GIB:.2f} GiB) -> {archive} "
        f"({manifest}; {foreign} foreign entries untouched)"
    )
    return {"swept": own, "bytes_freed": total, "bytes_apparent": apparent,
            "refused": {}, "archive": archive, "sems_unlinked": sem_gone}
















# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------



































































# --------------------------------------------------------------------------
# argv composition
# --------------------------------------------------------------------------




VISION_SOURCES = (VISION_SOURCE_DISK, VISION_SOURCE_RAM, VISION_SOURCE_PINNED)
#: VISION-WEIGHTS (09.10.): `weights` = the tower borrows victim weight memory
#: of P's PP0 (pdflip.vision_victim); off until the metal proof
VISION_PLACES = (VISION_PLACE_AUTO, "kvtail", "free", "weights")







PROFILE_NEXTFLASH = pdflip_form.PROFILE_NEXTFLASH
PROFILES = tuple(pdflip_form.PROFILES)























































































from flliper.srt.pdflip.launcher_p_choose import (  # noqa: F401 -- re-export, behaviour-neutral move
    p_rope_context_delta_mib,
)








from flliper.srt.pdflip.launcher_d_p_build import (  # noqa: F401 -- re-export, behaviour-neutral move
    p_rope_draft_delta_mib,
)































































def _derive_d_l2_budget(
    ns, max_kv_per_request: int
) -> Tuple[int, Dict[str, float], List[str]]:
    """Group D's ``--hicache-size`` in GB, plus the lines that make it auditable.

    A HELPER AND NOT INLINE IN ``main()``, for two PINNED reasons rather than
    taste: ``main()`` must contain zero try/except (the one refusal funnel is
    ``cli()``, #1248) and its inline-raise sites are counted exactly. Both
    refusals below therefore live here, where they are ordinary raises that the
    single funnel catches. The first draft put them in ``main`` and the gate
    caught it -- `test_main_has_zero_try_blocks` read
    ``[<ast.Try>] != []`` and `test_mains_own_raise_sites_are_pinned_pre_and_post_spawn`
    read ``6 != 4``.

    WHY D NEEDS ITS OWN BUDGET AT ALL: upstream stages every L3 read through
    the host pool and sizes that pool so residency cannot bind a read. A fixed
    1 GB on D gave 30,518 rows and a #915 limit of 27,466, and the #1317
    window/chain/anchor layer was the compensation for the shortfall. Derived
    from the cap, one cap-sized read fits in ONE prefetch on every rank.
    """
    cap_tokens = int(max_kv_per_request or 0)
    rec_share = _installed_max_share_from_record()
    # #1317n THE RECORD IS THE SHARE SOURCE WHEN IT HAS ONE; the flag is an
    # OVERRIDE/FLOOR, not the primary. A measured share is a reading and the
    # flag is a margin, and preferring the margin over a reading is how the
    # 1 GB constant survived.
    #
    # AND IT BUYS NOTHING ON THIS BOX, which is worth stating here so nobody
    # re-derives it: `hicache_size` is an INT in upstream server_args
    # (gigabytes), so S_D is ceil()ed either way -- 262,144 x 0.3750 x 32,768
    # / 0.9 = 3.58 GB and x 0.4000 = 3.82 GB both land on **4**. The share
    # affects `rows_needed` (98,304 vs 104,858), i.e. how much of the pool the
    # invariant demands, never the pool's price. A decimal S_D would save
    # ~1.18 GiB of rings and needs an upstream int->float change on that flag.
    share = float(ns.d_cap_rank_share)
    # #1361 [23b] THE SOURCE, NOT ONLY THE VALUE. The record silently overrides
    # the flag here, and the boot seat read `S_D=1` off its own sizing table
    # while the ARM line said `S_D=2` -- effective-and-unprinted, the #896
    # class. Whoever reads the arm has to be able to see WHICH of the two
    # inputs produced it without re-deriving the precedence from this comment.
    share_source = "flag"
    # #1361 [23c] THE COMPARISON HAPPENS BEFORE THE OVERWRITE, and that is the
    # whole fix. The guard below read
    #
    #     share = rec_share                      # <- the overwrite
    #     if rec_share > share + 1e-9: raise ...  # <- compares y against y
    #
    # so W89x could not fire on any input: `share` had just been set to
    # `rec_share` one line above. A refusal that is named, documented and
    # unreachable is worse than no refusal, because every reader downstream
    # (including this seat, twice) takes it as proof that the case is covered.
    # Same family as the reader that was never called and the comment that
    # promised "priced rather than borrowed" -- here the condition is killed by
    # its own precedent rather than by a missing consumer.
    _flag_share = float(ns.d_cap_rank_share)
    if rec_share is not None and float(rec_share) > _flag_share + 1e-9:
        raise PdFlipLaunchRefused(
            f"W89x PdFlipL2ShareBelowInstalled: the sizing record's installed "
            f"KV-token ownership vector has max(v)/sum(v) = {rec_share:.4f} "
            f"(source=record), above --d-cap-rank-share {_flag_share:.4f} "
            f"(source=flag). BOTH NUMBERS AND BOTH SOURCES are named because "
            f"the record SILENTLY OVERRIDES the flag a few lines below this "
            f"guard: without the pair, a reader cannot tell which value the "
            f"arm was actually sized on -- the boot seat read S_D=1 off its "
            f"own table while the ARM line said S_D=2, and this refusal was "
            f"the only thing that could have said so. Sizing D's L2 on the smaller "
            f"share ships a pool that cannot hold one cap-sized read on the "
            f"largest rank -- the 27,466-row wall this change removes -- so it "
            f"is refused by name rather than discovered as a 413. Raise "
            f"--d-cap-rank-share to at least {rec_share:.4f}."
        )
    # THE RECORD IS THE SHARE SOURCE WHEN IT HAS ONE -- unchanged behaviour for
    # every input that was not refused above (#1317n: a measured share is a
    # reading and the flag is a margin).
    if rec_share is not None:
        share = rec_share
        share_source = "record"
    try:
        terms = host_ledger.derive_d_hicache_size_gb(
            cap_tokens, share, host_ledger.CELL_BYTES_D_PER_RANK[0],
            HICACHE_LOAD_POOL_USAGE_FRACTION_1317N,
        )
        terms["max_rank_share_source"] = share_source
    except ValueError as exc:
        raise PdFlipLaunchRefused(
            f"W88x PdFlipL2Unsizable: group D's L2 budget could not be derived "
            f"from cap={cap_tokens} share={share}: {exc}"
        ) from exc
    src = (
        f"record {rec_share:.4f} (sizing record; flag {ns.d_cap_rank_share} "
        f"is override/floor)"
        if rec_share is not None else
        f"flag-default {ns.d_cap_rank_share} vs installed unknown at launch "
        f"(#1032: no vector shipped)"
    )
    return int(terms["s_gb"]), terms, [
        host_ledger.d_hicache_provenance(terms, src),
        f"PDFLIP-L2 SHARE PROVENANCE: the launcher ships NO ownership vector "
        f"(#1032), so the INSTALLED vector is not knowable here -- the runtime "
        f"derives an estimate from this boot's budgets and supersedes it after "
        f"profiling. `--d-cap-rank-share` is a NAMED MARGIN, not a reading: "
        f"boot weg2sn6p measured installed [17,24,23] -> max/sum = 0.3750 and "
        f"its own pre-boot estimate [2,3,3] -> 0.3750, i.e. the SHARE was "
        f"stable across the supersession even though the vector was not. "
        f"Default {share} covers that with margin. IF the installed share "
        f"exceeds it, D's #915 limit lands BELOW cap x share and the #1246 "
        f"carrier bound says so by name -- a readable outcome, never a silent "
        f"one.",
    ]




















#: #1374 OPTION 1: the weight-chunk length this boot ships, published by `main`
#: to the ONE producer below. A module global rather than a new keyword on
#: `choose_host_ledger`, whose signature is pinned keyword-for-keyword by
#: `test_pdflip_store_priced_x_1317`: the number is a property of the BOOT, set
#: once before either consumer runs, and an unset value REFUSES rather than
#: defaulting -- a silent 0 would size the buffer for the old geometry and
#: re-enter weg2xsn30's deadlock.
_WEIGHT_CHUNK_LAYERS: Optional[int] = None
_TAGMAX_CACHE: Dict[tuple, int] = {}


def publish_weight_chunk_layers(chunk_layers: int) -> None:
    """`main` states the chunking once; both term consumers read it."""
    global _WEIGHT_CHUNK_LAYERS
    _WEIGHT_CHUNK_LAYERS = int(chunk_layers)


def xchg_max_tag_bytes(model_dir: str) -> int:
    """The largest WEIGHT TAG of any source rank, in bytes -- ONE producer.

    Option 1 sizes the assemble buffer so a tag's deposit completes without its
    collector; that bound is the largest tag, and the tags are the language
    model's layer chunks. Cached per (model_dir, chunk_layers) because it reads
    every safetensors header and both consumers ask the same question.

    VALIDATED against boot weg2xsn30's own manifests before it was wired
    (operator rule: deviation > 5 % per rank = report, do not wire):
        stage 0 (layers 0:39)  derived 2907 MiB  manifest 2988 MiB  -2.7 %
        stage 1 (layers 39:52) derived 2907 MiB  manifest 2856 MiB  +1.8 %
        stage 2 (layers 52:64) derived 2907 MiB  manifest 2856 MiB  +1.8 %
    The first draft overshot stage 0 by +9.2 %, and the cause was enumerated
    rather than scaled away: `LAYER_RE` matches the substring `layers.<k>.`,
    and this checkpoint carries TWO trees under it -- the language model's and
    the MTP draft head's `mtp.layers.<k>`. Layer 0 is 366.2 MiB of model plus
    355.1 MiB of mtp, and 355 MiB was the entire overshoot. The MTP head is its
    own module with its own tag, so `MTP_TREE_PREFIXES` excludes it BY NAME.

    THE SCALAR MAX OVER SOURCE RANKS, not per lane: it never under-sizes a
    lane, and on this checkpoint every stage derives the same 2907 MiB, so a
    per-lane refinement would change no number here. Named as a follow-up
    rather than built blind.
    """
    if _WEIGHT_CHUNK_LAYERS is None:
        raise checkpoint_census.PdFlipXchgWidestLayerUnreadable(
            "W14 PdFlipXchgWidestLayerUnreadable: the weight-chunk length was "
            "never published (`publish_weight_chunk_layers`), so the largest "
            "weight tag cannot be derived. REFUSING at launch: a default of 0 "
            "would size the assemble buffer for the old depth-based geometry, "
            "and a tag that does not fit it cannot be deposited without its "
            "collector -- boot weg2xsn30's deadlock, re-entered silently.")
    key = (str(model_dir), int(_WEIGHT_CHUNK_LAYERS))
    if key not in _TAGMAX_CACHE:
        _TAGMAX_CACHE[key] = int(checkpoint_census.max_tag_bytes_from_census(
            str(model_dir), int(_WEIGHT_CHUNK_LAYERS)))
    return _TAGMAX_CACHE[key]


def xchg_bounce_terms_for_arm(weight_source: str, oncard_mode: str,
                              model_dir: str,
                              oncard_slot_mib: Optional[int] = None,
                              bounce_depth: int = xchg_bounce.ASSEMBLE_DEPTH_DEFAULT,
                              n_lanes: int = 1,
                              lanes_concurrent: int = 0,
                              # #1397: THE SINGLE BOOLEAN (design doc section
                              # 9.2) -- `n_cross_lanes` stays `None` at every
                              # call site in this file, on purpose, so it
                              # ALWAYS auto-derives through
                              # `xchg_bounce.resolve_cross_lanes` rather than
                              # becoming a second knob nobody asked for.
                              band_credit: bool = False,
                              price_lane_cap: int = 0):
    """``(charged_bytes, lines)`` for the host bounce. #1332 B1b.

    THE SECOND PRODUCER OF THE SAME PREDICATE AS
    :func:`xchg_bounce_arm_pins_host`, and deliberately its neighbour rather
    than its replacement: that one answers "does this arm pin host bytes at
    all" for every arm and stays the authority for `ring` and `ipc`, which
    charge 0.  This one answers "how many, on the arm that does", and it is the
    call site where seat 3's sizing function finally receives a MEASURED
    ``widest_layer_bytes`` instead of the missing producer WEG2_REUSE_SPEC
    §10.8 named as open decide 1.

    Nothing here is arithmetic: the census MEASURES
    (``checkpoint_census.layer_census_from_headers``, safetensors headers,
    group-wide, per layer, MAX), ``bounce_terms`` SIZES, ``arm_line`` PRINTS
    and ``under_coverage_refusal`` REFUSES.  The launcher only decides which
    arm asks.

    REFUSES rather than defaults, twice: an arm that pins host bytes with no
    checkpoint to measure, and a buffer whose depth-slot cannot hold the widest
    layer (W71 form).  Both at ARM time, before either group starts -- a layer
    that cannot be assembled whole cannot be sliced by its destinations, and
    finding that out mid-flip means finding it out after VRAM was mutated.
    """
    # ONE AUTHORITY FOR THE PREDICATE, and it is the OLDER function.
    # `xchg_bounce_arm_pins_host` already answers "does this arm pin host bytes
    # at all" and is pinned by two tests in test_pdflip_xchg_shadow_1273; asking
    # it here instead of re-deciding the arm strings keeps that single reader
    # (upstream-minimal: a second copy of a predicate is the defect class this
    # fork keeps paying for).  What B1b adds is the SIZE, not the predicate.
    if not xchg_bounce_arm_pins_host(weight_source, oncard_mode):
        return 0, []
    if not str(model_dir or "").strip():
        # THE EXCEPTION MATCHES THE CODE IT PRINTS.  An earlier draft raised
        # `PdFlipXchgBounceUnderCovered` while naming the other class in the
        # text, which is the "instrument text lies" shape one level down: a
        # reader greps the name and lands on a class that was never raised.
        raise checkpoint_census.PdFlipXchgWidestLayerUnreadable(
            "W14 PdFlipXchgWidestLayerUnreadable: the host on-card arm pins a "
            "host bounce whose size is the WIDEST layer of this boot's "
            "checkpoint, and no checkpoint path reached this call site. "
            "Refused rather than sized against a default"
        )
    slot_bytes = 0
    if oncard_slot_mib is not None:
        slot_bytes = (weight_exchange_transport.validate_oncard_slot_mib(
            oncard_slot_mib) * weight_exchange_region.MIB)
    terms, widest, widest_name = checkpoint_census.widest_layer_terms(
        str(model_dir), pairs=int(weight_exchange_region.N_CARDS),
        depth=int(bounce_depth), slot_bytes=slot_bytes, n_lanes=int(n_lanes),
        max_tag_bytes=xchg_max_tag_bytes(str(model_dir)),
        lanes_concurrent=int(lanes_concurrent),
        band_credit=bool(band_credit),
        price_lane_cap=int(price_lane_cap or 0))
    lines = [widest, xchg_bounce.arm_line(terms)]
    # PROVENANCE, ADDITIVE: the depth-slot is derived HERE, from this
    # checkpoint's census, and never from a knob -- but no line said so, and a
    # size whose origin is not printed is a size a later reader will assume
    # came from a default. The desk replay carried its own 4 MiB constant and
    # met W71 at full width (`widest unit ... 252086022 B against a depth-slot
    # of 4194304 B`); that was the replay's constant, not this path, and this
    # line is what makes the difference readable from a boot log.
    lines.append(
        f"PDFLIP-XCHG DEPTH-SLOT bytes={int(terms.widest_layer_bytes)} "
        f"source=census widest={widest_name} "
        f"(depth={int(terms.depth)} buffer_bytes={int(terms.buffer_bytes)} "
        f"staging_slot_bytes={int(terms.slot_bytes)} "
        f"n_layers={int(terms.n_layers)}) -- the assemble buffer's depth-slot "
        f"holds the WIDEST layer of THIS checkpoint; a boot sized on the mean "
        f"dies on whichever layer is above it")
    # #1385 (Wand 11b): APPENDED, not inserted -- existing callers index
    # `lines[0..2]` by position (WIDEST, PDFLIP-XCHG-BOUNCE, PDFLIP-XCHG
    # DEPTH-SLOT) and an insertion in the middle would silently shift every
    # one of them. Named and counted, never silent: a cap that only showed
    # up as a smaller `bounce_total_mib` would itself be the printed
    # advisory nobody could grep for the flip-time cost of.
    lines.append(xchg_bounce.lanes_concurrent_line(terms))
    if terms.lanes_serialised:
        # #1385 STEP 2 OF 2: THE CAVEAT NOW SAYS WIRED, BECAUSE IT IS. Step 1
        # priced `lanes_priced` without touching the per-rank leg driver, and
        # this line said NOT_WIRED for exactly as long as that was true. Now
        # `weight_updater.py`'s lane loop opens a boot-wide POSIX permit
        # (`weight_exchange_transport.LanePermit`, created here by
        # `weight_exchange_region.create_lane_permit_semaphore` a few lines
        # below) and acquires it before each lane's `LayerBounce` is created,
        # releasing it only after `run_bounce_leg`'s own `finally` has closed
        # (unpinned) that buffer -- so at most `lanes_priced` may be pinned at
        # once, the SAME number this ARM just charged. THE LINE MUST NEVER
        # SAY WIRED WHILE IT IS NOT, so it stays a mechanical consequence of
        # this class's own code (see `test_the_caveat_says_wired_iff_the_
        # permit_is_really_acquired`) and not a claim typed once and left
        # behind the next time the runtime changes.
        lines.append(
            f"PDFLIP-XCHG-LANES-CONCURRENT-CAVEAT runtime_enforcement=WIRED "
            f"(#1385 step 2) -- lanes_priced={int(terms.lanes_priced)} is "
            f"enforced by a boot-wide permit the lane loop acquires before "
            f"each buffer and releases right after it closes; a wait beyond "
            f"{weight_exchange_transport.LANE_PERMIT_TIMEOUT_S:.0f}s refuses "
            f"by name (W69 PdFlipXchgGateTimeout) rather than hanging. NOT YET "
            f"BOOT-PROVEN: confirm with the #1358 3-way measurement (tmpfs "
            f"holders, cgroup d_shmem, host-slot lines) that the real peak "
            f"falls to this number on the metal, not only at the desk")
    if not terms.covers_widest_layer:
        raise xchg_bounce.PdFlipXchgBounceUnderCovered(
            xchg_bounce.under_coverage_refusal(terms,
                                               widest_layer_name=widest_name))
    return int(terms.total_bytes), lines









def choose_host_ledger(
    ring_bytes: int,
    ring_span1_bytes: int,
    ring_provenance: str = "",
    d_draft_host_gib: float = 0.0,
    memhist_gib: float = 0.0,
    memhist_run_only: bool = False,
    l3_index_gib: float = 0.0,
    # 29.09. (Planer post COLD_TIER_SHM): the tmpfs expert store, GiB, both
    # moments -- `cold_tier_shm_post` sizes it from the map, never from df.
    cold_tier_shm_gib: float = 0.0,
    # 29.09.: the groups' own env (--env-d, --env-p); the arena term reads
    # its sizes through it, not through the launcher env alone.
    group_envs: Optional[Sequence[Dict[str, str]]] = None,
    # 29.09.: the host census posts of this model|form (host_census.ledger_terms)
    census: Optional[Dict[str, object]] = None,
    meminfo_path: str = "/proc/meminfo",
    cgroup_root: str = "/sys/fs/cgroup",
    record_path: Optional[str] = None,
    pin_m_mib: int = 0,
    pin_s_gb: int = 0,
    deviation_reason: str = "",
    guard_gib: Optional[float] = None,
    model_digest_want: str = "",
    form_key_matches: bool = False,
    s_gb_d: Optional[int] = None,
    d_cap_terms: Optional[Dict[str, float]] = None,
    weight_source: str = WEIGHT_SOURCE_DEFAULT,
    # B4f: the OTHER half of the ring-absence predicate.  Defaulted to the
    # flag's own default so an existing caller that names only the weight
    # source gets the arm the launcher actually runs -- which is the arm that
    # HAS a ring, i.e. the safe side of the contradiction weg2xsn13 hit.
    inject_mode: str = weight_exchange.INJECT_SHADOW,
    oncard_mode: str = ONCARD_MODE_DEFAULT,
    oncard_slot_mib: Optional[int] = None,
    model_dir: str = "",    # #1358: ONE source for the depth -- the ledger term, `assemble_slots`
    # and the published terms all price the same number. A second spelling is
    # how `2.16 GiB` came to have three causes (#1361 [23a]).
    bounce_depth: int = xchg_bounce.ASSEMBLE_DEPTH_DEFAULT,
    # #1358: THE CUT, for the measured lane count. Not the form key -- the lane
    # set follows from which card sends to which, and text-only (#1356) changes
    # the form and not the cut.
    stage_ratio: str = "",
    d_vector: str = XCHG_D_VECTOR_DEFAULT,
    legs: str = "both",
    # #1385 (Wand 11b): THE VALIDATED CAP, never the raw argv value -- the
    # caller resolves it once through `xchg_bounce.resolve_lanes_concurrent`
    # (the same function the ranks' publication call resolves it through) and
    # hands this function the number, exactly as `bounce_depth` already does.
    # 0 is "not stated": byte-identical, every lane prices as before #1385.
    lanes_concurrent: int = 0,
    # #1397 (Option 3, design doc section 9): resolved ONCE by `main`,
    # exactly as `lanes_concurrent` two lines up -- the ranks' publication
    # call and this ledger call must read the identical bool, or a cross
    # lane's smaller price and a rank's larger allocation could disagree.
    # False is byte-identical: `n_cross_lanes_priced` is 0 whenever
    # `band_credit` is unset, so `total_bytes` takes its pre-#1397 branch.
    band_credit: bool = False,
    # #1386 (Minimalform HiCache switch): resolved ONCE by `main`, beside
    # `draft_kv_on_p`, and handed to `host_ledger.choose` unchanged -- the
    # ledger term and the argv this function's caller builds from `arm.s_gb`/
    # `arm.m_mib` must read the identical bool, never a second copy of it.
    hicache_disabled: bool = False,
    # #1369: resolved ONCE by `main`, beside `hicache_disabled`, from the
    # SAME predicate that decided whether `--enable-weights-cpu-backup`
    # ships on either group's argv (`ring_absent_by_design`'s negation, or
    # the operator's explicit on/off) -- never re-derived here, which is
    # exactly the #1358 defect class ("2.16 GiB" had three causes) this
    # order named by name. True (default) is byte-identical to every boot
    # before this flag existed: the ring prices exactly as it always did.
    weights_cpu_backup_armed: bool = True,
    # #1378 Stage 2 (W105): the PRIOR boot's own cited numbers. FLAGS ARE AN
    # OVERRIDE ONLY (coordinator order): when either is given explicitly it
    # wins outright; otherwise this function SELF-READS the sidecar via
    # `flip_ratchet_form_key` below -- the same "record, not operator typing"
    # shape `record`/`measured_record` already carry for every other term.
    prior_cushion_min_gib: Optional[float] = None,
    prior_bounce_gib: Optional[float] = None,
    # #1378 Stage 2: THIS boot's own form, for the auto-resolve's exact-match
    # filter. "" (the default) auto-resolves nothing -- a caller that does
    # not know its own form gets the byte-identical pre-Stage-2 behaviour,
    # never a guess at what "this form" means.
    flip_ratchet_form_key: str = "",
    # PDFLIP-FORM (24.09.): the calibration identity of the record read below
    # (``pdflip_form.same_model_sample``). None = every sample, as before --
    # xsn417's RUN-PEAK ADVISORY then priced the Next-Flash boot fnFL2x142's
    # run sample (-90.18 GiB, a death sample) for a Qwen3.8-27B arm.
    record_accept: Optional[Callable[[dict], bool]] = None,
    # H87 (rc2.1c2 container, NVFP4 --d-only, W87/W20): a D-only boot has NO
    # flip arm -- no flip ratchet, no dormant group, no P host pools. False is
    # byte-identical to every flip boot.
    d_only: bool = False,
    vision_victim_host_gib: float = 0.0,  # VISION-WEIGHTS AP4: transient host image of the displaced weights (printed post, in no ledger sum, warning only); LAST
) -> Tuple[host_ledger.Arm, Optional[float], List[str], Dict[str, Optional[int]]]:
    """THE LAUNCHER'S ONE LEDGER CALL SITE: read the host, price the ladder.

    Returns ``(arm, reap headroom GiB, printed lines, the cgroup reading)`` -- ``main`` only logs
    the lines and carries the reading into the boot state. HOSTWARN2 (13.10.2026, user rule
    06.10.): the host-RAM VERDICTS (W20 no arm funds, W21 run peak, W87 pinned arm, W97
    deviation) are ``HOST-RAM-WARN`` lines among ``lines`` and the most frugal (or the pinned)
    arm comes back; what still raises :class:`host_ledger.PdFlipHostLedgerRefused` is a number
    that does not exist (unmeasured term, record identity, half-cited prior boot).

    #1233 fix 5: the reaper watches the CGROUP, so the ledger reads it.  The
    ``oom_kill`` value goes in as the boot's PRE-BOOT BASELINE of a cumulative,
    timestamp-free counter -- no boot starts without one.

    #1233 fix 7: this is a FUNCTION and not four lines inside ``main`` because
    it is the only wire that carries fix 6 into a boot, and inside ``main`` --
    behind NVML, the store preparation and two servers -- nothing could reach it.  The
    fix-6 review measured exactly that: replacing ``cg["reclaimable"]`` below
    with ``None`` reverts the whole boot to fix 5's denominator and 152 tests
    stayed green.  The two paths are the same code from here down, and the
    ``meminfo_path`` / ``cgroup_root`` arguments exist so a test can hand this
    seam a fake box whose readings decide a DIFFERENT arm.

    C19 (the ring): the host WEIGHTS term arrives as ``ring_bytes`` /
    ``ring_span1_bytes`` from :func:`prepare_host_ring`, not as a
    ``weight_chunks`` count.  That parameter is gone because the term it fed --
    one image plus one chunk in flight -- is gone: the shared region is the
    whole host weights cost and it is charged once.  There is deliberately no
    default: a caller with no measured table must reach the W20 refusal, never
    a zero that prices the flip as free.
    """
    mi = host_ledger.read_meminfo(meminfo_path)
    # #1361 (2) THE PEAK BASELINE, and it is the same lesson as `cg_oom_kill`.
    # `memory.peak` is a LIFETIME high-water mark of this cgroup with no reset
    # anywhere in this tree, so a boot record that quotes it as "this boot's
    # peak" is quoting every boot since the container started. Measured on
    # weg2xsn26b: the mark already stood at ~95.1 GiB when it launched --
    # inherited from weg2xsn25's death -- and a reader who diffs nothing reads
    # 95.86 as xsn26b's own. Baselined here and printed, exactly as the
    # cumulative `oom_kill` counter is, so the NEXT record can state a DELTA.
    # Reset is not attempted: the file is `nobody:nogroup` inside this LXC and
    # a failed write would be a silent no-op, which is worse than a delta.
    _cg0 = host_ledger.read_cgroup(cgroup_root)
    _baseline_lines = [
        (
            f"PDFLIP-HOST PEAK BASELINE memory.peak={_gib(_cg0.get('peak'))} "
        f"memory.current={_gib(_cg0.get('current'))} -- CUMULATIVE marks with no "
        f"reset in this tree (the file is read-only in this container). Any "
        f"'peak' in this boot's record is a DELTA against this line, never an "
        f"absolute: weg2xsn26b inherited ~95.1 GiB of mark from weg2xsn25's death "
        f"and its record read the inherited figure as its own"
        ),
        host_ledger.reap_model_line(
            mi["MemTotal"], mi.get("MemAvailable"), _cg0.get("current")),
    ]
    # #1392: the wall-clock THIS box read happened at, captured at the read
    # itself -- not reconstructed later from a log timestamp that could be
    # seconds or minutes stale by the time the ARM line prints.
    _box_state_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cg = host_ledger.read_cgroup(cgroup_root)
    cg_ceiling, cg_ceiling_source = host_ledger.resolve_cg_ceiling(cg, mi["MemTotal"])
    # fix 8: this line's OWN previous measurements of the dormant image and of
    # the run-moment residual.  Absent (first boot, or a wiped sidecar) the
    # ledger prices the named dk7 reading and prints that it is another boot's.
    _record_path_resolved = (
        measured_record_path() if record_path is None else record_path
    )
    record = host_ledger.read_measured_record(_record_path_resolved, accept=record_accept)
    # H87: IS THIS BOOT'S CHECKPOINT THE ONE THE BUILT-IN REFERENCES WERE
    # MEASURED ON? Decided once, here, by memory footprint (an abliterated
    # derivative of the reference counts, a different checkpoint under any
    # name does not). No model named (a desk caller) = None = pre-H87 exactly.
    if model_dir:
        _ref_ok, _ref_why = pdflip_form.reference_model_verdict(
            model_dir, host_ledger.REFERENCE_MODEL)
    else:
        _ref_ok, _ref_why = None, ""
    # #1378 Stage 2 (order): SELF-READ the prior boot's cushion from the same
    # sidecar `record` came from -- flags stay an OVERRIDE, never the only
    # path, so a boot that names nobody by hand still gets the gate. The
    # provenance line is ALWAYS appended (even on a flag override, even on a
    # miss) -- the order's "nie stumm": an absent auto-resolution must say it
    # could not measure, not just leave the field blank.
    if prior_cushion_min_gib is not None or prior_bounce_gib is not None:
        _auto_cushion, _auto_bounce, _auto_prov = None, None, (
            "auto-resolve skipped: --prior-cushion-min-gib/--prior-bounce-gib "
            "override it")
    elif flip_ratchet_form_key:
        _auto_cushion, _auto_bounce, _auto_prov = host_ledger.resolve_prior_cushion(
            _record_path_resolved, flip_ratchet_form_key, accept=record_accept)
    else:
        _auto_cushion, _auto_bounce, _auto_prov = None, None, (
            "auto-resolve skipped: caller passed no flip_ratchet_form_key")
    _resolved_prior_cushion_min_gib = (
        prior_cushion_min_gib if prior_cushion_min_gib is not None else _auto_cushion)
    _resolved_prior_bounce_gib = (
        prior_bounce_gib if prior_bounce_gib is not None else _auto_bounce)
    _baseline_lines.append(f"PDFLIP-CUSHION-HEADROOM-SOURCE {_auto_prov}")
    # ONE kwargs block for the ladder AND the pin. Boot weg2sn6a (Q6, 07:33Z)
    # died in this function with `price() got an unexpected keyword argument
    # 'ring_provenance'`: the pinned arm had been priced by a hand-copied
    # price() call that carried choose()'s keywords (ring_provenance,
    # slab_reclaimable_bytes, cg_ceiling_source, cg_oom_kill), none of which
    # price() takes -- and no test exercised the call. The pin now goes THROUGH
    # choose() with the ladder restricted to the one arm, so it is scored by
    # the identical model (slab term, reap bound, W20/W21 refusals) and cannot
    # drift from the ladder call again. test_pdflip_store_priced_x_1317 pins the
    # keyword/signature conformance of every host_ledger call in this module.
    # #1332 B1b -- MEASURED BEFORE THE LEDGER IS ASKED, so the two refusals
    # (unreadable checkpoint, under-covered buffer) land before any host number
    # is priced against them.
    # #1358 THE LANE COUNT, MEASURED AND KEYED ON THE CUT. One assemble buffer
    # per lane, and the lane set is built at the rank at runtime -- so the arm
    # reads the number a previous boot of THIS CUT allocated, exactly as it
    # reads Sigma H. W102 by name when no boot of this cut has measured one;
    # never a default, because pricing one buffer for a five-lane boot is the
    # 5.64 GiB under-charge that latched W98 on weg2xsn28.
    # #1358 [fix] ONLY WHERE THERE ARE LANES. `xchg_bounce_arm_pins_host` is
    # the ONE producer of "does this arm pin host bytes for the exchange", and
    # an arm that pins none creates no assemble buffer at all -- so demanding a
    # lane record from it refused the DEFAULT arm, which is the very boot that
    # would have produced the record the exchange arm needs. A producer that
    # locks out its own bootstrap is worse than the under-charge it fixes.
    _price_cap = 0
    if xchg_bounce_arm_pins_host(weight_source, oncard_mode):
        _lane_n, _lane_prov = xchg_lane_count(stage_ratio, legs, d_vector)
        # #1464b: REGION FORM -- the lane term is a coverage constant, not
        # this cut's buffers.  xsn220 (32a8193805) shipped the solver's
        # 43,11,10 and the ledger charged its WORST-CASE 9 lanes = 27.75 GiB
        # -> run peak 103.00 GiB -> W97 at launch, for buffers the region
        # form never allocates (see host_price_for_cut, #1464).  Charge every
        # cut what the measured incumbent was charged (5 lanes -> 15.75 GiB,
        # the term that has covered an unattributed ~10 GiB since xsn31), so
        # the ledger's prediction does not move with the cut.  The count
        # published to the ranks stays `_lane_n` (W102 compares counts).
        if os.environ.get("FLLIPER_PDFLIP_PCUT_LANE_PRICE", "") != "1":
            _price_cap = region_form_lane_price_cap(
                d_vector, legs, n_stages=len([x for x in str(stage_ratio or "").split(",") if x.strip()]) or None)
            if _price_cap:
                _lane_prov += (
                    f" | #1464b REGION-FORM: lanes charged min(n_lanes={_lane_n}, "
                    f"{_price_cap}) -- the incumbent {_csv(DEFAULT_PP_ORDERED_CUT)}'s "
                    f"measured count as a coverage constant; no per-lane buffer "
                    f"exists in this form (xsn218: 0 HOST-SLOT lines, shmem "
                    f"+0.09 GiB per flip)")
    else:
        _lane_n = 1
        _lane_prov = (
            f"PDFLIP-XCHG-LANES cut={host_ledger.xchg_cut_key(str(stage_ratio or ''), d_vector, legs)} "
            f"lanes=n/a (no exchange) source=arm weight_source={weight_source} "
            f"oncard={oncard_mode} -- this arm pins no host bounce, so there "
            f"is no lane to count and no record to demand. The term is 0 and "
            f"says so; W102 is for an arm that WOULD allocate buffers.")
    _bounce_charge_bytes, _bounce_lines = xchg_bounce_terms_for_arm(
        weight_source, oncard_mode, model_dir, oncard_slot_mib, bounce_depth,
        _lane_n, lanes_concurrent, band_credit, price_lane_cap=_price_cap)
    _bounce_lines = list(_bounce_lines) + [_lane_prov]
    # #1361 [23a] THE PRICED STATE, NOT ONLY THE PRICE. B4n measured THREE
    # distinct states that all cost 2.16 GiB -- (depth=1, comparing=True),
    # (depth=2, comparing=False) and (depth=2, the OLD expression that was just
    # fixed) -- so the figure alone cannot tell a correctly priced shallow arm
    # from the underpricing. The ARM line prints the state beside the term.
    #
    # NOT a second derivation of the size: `depth` is the SAME VALUE the helper
    # hands to `widest_layer_terms` -- since #1358 that is the `bounce_depth`
    # flag rather than the constant, and it is READ FROM THE SAME VARIABLE.
    # Leaving the constant here while the arm is priced at depth=1 would make
    # the ARM line print `bounce_depth=2` for a depth=1 arm: the field [23a]
    # introduced to stop a number having three readings would itself be the
    # lie. Caught at the merge by the train seat, who refused to resolve it
    # mechanically and was right to.
    # and `slots` comes from `assemble_slots`, which is the one authority
    # (#1335, B4n). `inject_mode` is resolved by that function from this
    # process's own `weight_exchange.inject_mode()`, which is exactly the term
    # a desk run and a boot can disagree about -- printing it is what makes the
    # disagreement readable instead of hidden inside an equal number.
    _bounce_prov = {
        "depth": int(bounce_depth),
        # #1361 [23e] `comparing` IS PASSED, NOT LEFT TO THE PROCESS DEFAULT.
        # `assemble_slots(depth)` with `comparing=None` asks
        # `weight_exchange.inject_mode()`, which reads INJECT_ENV -- and the
        # launcher NEVER sets that for itself: `prepare_xchg_env` puts it only
        # into the dict handed to the ranks (launcher.py:3594), while the
        # launcher sets WEIGHT_SOURCE_ENV for its own process (:9631) and not
        # this one. So a boot armed `authoritative` still got `slots` computed
        # as if it were shadow, while `bounce_inject` printed `authoritative`
        # from the argv -- the two fields could contradict each other on the
        # same line, and the one that disagreed with the ranks was the one
        # nobody would have checked. Found by B4n on the env publication, not
        # by me on my own field.
        "slots": int(xchg_bounce.assemble_slots(
            int(bounce_depth),
            comparing=(str(inject_mode) == weight_exchange.INJECT_SHADOW))),
        "inject_mode": str(inject_mode),
    } if _bounce_charge_bytes else None
    # #1369 BOOT7 FIX (2026-09-14): NOT `ring_absent_by_design(weight_source,
    # inject_mode)`. That call reads ONLY the weight-source/inject axis and
    # is correct EXCLUSIVELY under `auto` (where it is this predicate's own
    # definition -- see `main`'s resolution site, `weights_cpu_backup_armed
    # = not ring_absent_by_design(...)`). Under an EXPLICIT
    # `--pdflip-weights-cpu-backup on`, `weights_cpu_backup_armed` is forced
    # True regardless of weight_source/inject_mode (the whole point of an
    # A/B comparison instrument that keeps the ring under exchange+
    # authoritative) -- but the bare call above still returned True (ring
    # "absent by design") for exactly that arm, because it never learned
    # about the flag at all. `pdflip_weights_cpu_backup_ring_kw`'s own `if
    # weights_cpu_backup_armed:` branch then passed that stale True through
    # UNCHANGED beside the real, non-zero ring bytes `on` built -- the exact
    # contradiction `price()`'s own guard exists to catch (boot xsn32,
    # 6/6 dry runs, ValueError "ring_absent_by_design with a non-zero ring
    # (ring_bytes=51036291072, ring_span1_bytes=51036291072)",
    # host_ledger.py:3835).
    #
    # `weights_cpu_backup_armed` IS ALREADY THE FULL AUTHORITY over all
    # three modes (`weight_exchange.weights_cpu_backup_armed`'s own
    # docstring: on=True always, off=False always, auto=`not
    # ring_absent_by_design(...)`) -- so "is the ring absent" is simply its
    # negation in EVERY mode, not only under auto. Verified identity for
    # auto: `weights_cpu_backup_armed = not ring_absent_by_design(...)`
    # there means `not weights_cpu_backup_armed == ring_absent_by_design(...)`
    # exactly, so this is BYTE-IDENTICAL to the old call for every pre-#1369
    # boot (which only ever ran `auto`). Under `on`/`off` it is now correct
    # where the old call was not, and it needs no new parameter, no new
    # function, and no second authority: the SAME resolved bool already
    # reaches this call site.
    #
    # `ring_absent_by_design(weight_source, inject_mode)` ITSELF is
    # untouched -- it stays exactly what `main`'s own `auto` branch (the
    # ONE remaining caller of the bare two-argument form) needs, and that
    # branch already only ever calls it when the mode IS auto.
    (_ring_bytes_kw, _ring_span1_bytes_kw,
     _ring_absent_by_design_kw) = pdflip_weights_cpu_backup_ring_kw(
        ring_bytes, ring_span1_bytes,
        not weights_cpu_backup_armed,
        weights_cpu_backup_armed,
    )
    ledger_kw = dict(
        # H25: the parked D draft's pinned host image (ledger post
        # 'd_draft_host'), priced by `d_draft_park_term` -- one kwargs block,
        # so the pinned arm and the ladder price the same post.
        d_draft_host_gib=float(d_draft_host_gib), vision_victim_host_gib=float(vision_victim_host_gib),  # AP4: displaced weights (printed post only)
        # rc12d: the torch allocation history (pdflip/memhist.py) -- 5.4 GiB of
        # host nobody booked when it was armed from rank start in P and D.
        memhist_gib=float(memhist_gib), memhist_run_only=bool(memhist_run_only),
        # 28.09.: the L3 store's RAM index (ledger post 'l3_index'), every owner.
        l3_index_gib=float(l3_index_gib),
        # 29.09.: the tmpfs expert store (Planer post COLD_TIER_SHM).
        cold_tier_shm_gib=float(cold_tier_shm_gib),
        # 29.09.: measured non-rank anon / lane ring / arena sidecar / hand-off
        census=census,
        # #1451: no arena on a --pdflip-disable-hicache boot -- the term charged
        # 33.6 GiB there too (test_pdflip_hicache_disabled_1386 M=2400 unfundable)
        **(_pdflip_arena_ledger_terms(model_dir, group_envs) if (model_dir and not hicache_disabled) else {}),  # #1432
        # #1317n D's L2 IS PRICED SEPARATELY FROM P'S. It rides the ONE kwargs
        # block for exactly the reason the block exists (boot weg2sn6a died of
        # a price() call that had drifted from choose()'s keywords): a term
        # that reaches only one of the two pricing paths is how an arm gets
        # scored against a budget the boot does not run. None = "same as P",
        # which is every recorded arm.
        s_gb_d=s_gb_d,
        d_cap_terms=d_cap_terms,
        # #1392: PRINT-ONLY, like `xchg_bounce_prov` -- the box state THIS
        # ladder was read against, so the ARM line can say which reading its
        # verdict is measured on instead of leaving a quiet-box FUNDABLE and
        # a noisy-box FUNDABLE to print identically.
        box_state_at=_box_state_at,
        cg_anon_bytes=cg.get("anon"),
        cg_shmem_bytes=cg.get("shmem"),
        # #1386: the ONE bool `choose_host_ledger`'s own caller resolved,
        # forwarded unchanged into BOTH `host_ledger.choose` calls below
        # (`arms=[(1, pin_m_mib)]` pinned, and the DEFAULT_ARMS ladder) --
        # `**ledger_kw` is exactly the mechanism the #1317n comment above
        # names to stop a term reaching only one of the two.
        hicache_disabled=hicache_disabled,
        # #1369: computed together with the declaration below, by
        # `pdflip_weights_cpu_backup_ring_kw` -- see its docstring for why
        # zeroing the bytes here and declaring the absence there must never
        # be two separate call sites again.
        ring_bytes=_ring_bytes_kw,
        ring_span1_bytes=_ring_span1_bytes_kw,
        ring_provenance=ring_provenance,
        cg_current_bytes=cg["current"],
        # fix 6: only the NON-reclaimable part of that reading is charged --
        # page cache is what the kernel hands back instead of killing for.
        reclaimable_bytes=cg["reclaimable"],
        # train fix 3: the LIVE slab term the reap watermark's own row lacked.
        # It is subtracted wherever the watermark bounds; passing None here
        # reverts that bound to the optimistic form and says so on every ARM
        # line. #1236: it no longer sizes a store -- there is no RAM store.
        slab_reclaimable_bytes=cg["slab_reclaimable"],
        cg_ceiling_bytes=cg_ceiling,
        cg_ceiling_source=cg_ceiling_source,
        cg_oom_kill=cg["oom_kill"],
        measured_record=record,
        # #1360: handed to the ONE place that computes the verdict, exactly as
        # every other arm property is. The launcher does not re-implement the
        # decision; it passes the operator's declaration down to it.
        deviation_reason=deviation_reason,
        guard_gib=guard_gib,
        # #1378 Stage 2 (W105): the RESOLVED pair -- the operator's override
        # when given, else this function's own auto-read of the sidecar
        # (`_resolved_prior_*` above), never the raw flag values: a caller
        # that passed neither flag must still reach the gate when the
        # sidecar has a candidate.
        prior_cushion_min_gib=_resolved_prior_cushion_min_gib,
        prior_bounce_gib=_resolved_prior_bounce_gib,
        # #1362 [22-fix]: the model this boot actually loads, by CONTENT. Every
        # recorded image now has to prove it belongs to it -- the fossil was a
        # 47 GiB ring solved from weg2xsn25's census for a 7.48 GiB model.
        model_digest_want=model_digest_want,
        form_key_matches=form_key_matches,
        # #1350 THE FLIP RATCHET, RESOLVED AT THE ONE CALL SITE THAT ALREADY
        # KNOWS THE BOOT -- the same placement as `xchg_bounce_host_bytes` and
        # `ring_absent_by_design` above, and for the same reason: the ledger
        # never opens the sidecar behind its caller's back, and the REFUSAL
        # (W94 PdFlipHostFlipRatchetUnmeasured) has to be reachable from here or
        # an absent measurement becomes a silent 0 in the run peak. `record` is
        # the dict this function already read; the front writes the field into
        # it at `PDFLIP-FLIP done epoch=2`.
        # H87: --d-only has no flip -- a DECLARED zero-flip ratchet, never the
        # series of another model's flip boots.
        flip_ratchet=(host_ledger.d_only_flip_ratchet() if d_only else
                      host_ledger.resolve_flip_ratchet_gib(
                          record, reference_model_ok=_ref_ok,
                          reference_model_why=_ref_why)),
        d_only=bool(d_only),
        reference_model_ok=_ref_ok,
        reference_model_why=_ref_why,
        # 29.09. NF1c (H87): the footprint keys the margin's OWN-MODEL
        # residual rows; only read when the checkpoint is not the reference
        # (form.footprint_key is cached per checkpoint stamp).
        model_footprint=(
            (pdflip_form.footprint_key(model_dir)[0] or "")
            if (model_dir and _ref_ok is False) else ""
        ),
        # #1273 S6: the exchange's own pinned host carrier.  The ARM STRINGS
        # decide it here, at the one ledger call site, and not inside the
        # ledger -- `WEIGHT_SOURCE_CHOICES` is this module's, and a ledger that
        # knew about arm names would be a second reader of a decision that
        # already has one.  `ring` charges 0 and every existing boot number is
        # unchanged.
        #
        # BOTH ARMS, NOT ONE (S6 refuter, must_fix 4).  This charged on every
        # non-ring boot, but the bounce FILE exists only on the `host` on-card
        # arm -- on `ipc` the verdict is `DEPOSIT_REASON_IPC`, the lane logs
        # the blameless not-drainable line and no host byte moves.  So a
        # `shadow` + `ipc` boot (the default, and the only one the launcher
        # could produce before the flag above existed) shrank the store and the
        # run-peak headroom by 0.75 GiB for bytes that arm cannot allocate: a
        # charge for a thing that does not happen is the mirror of the omission
        # this term was added to close.  The predicate is now the SAME one the
        # deposit allocates under, and it is the launcher's own published
        # value rather than a second reading of the environment.
        # #1332 B1b: ON THE HOST ARM THE CHARGE IS THE MEASURED BOUNCE.
        # `xchg_bounce_arm_pins_host` stays the authority for `ring` and `ipc`
        # (both 0); where bytes are actually pinned, the size is derived from
        # THIS checkpoint's WIDEST layer by `xchg_bounce_terms_for_arm` ->
        # `bounce_terms`, and its two lines join the ledger's printed lines.
        xchg_bounce_host_bytes=_bounce_charge_bytes,
        xchg_bounce_prov=_bounce_prov,
        # #1327 (S6 slice 2): SIGMA H IS 0 BY DESIGN ON THE AUTHORITATIVE
        # EXCHANGE ARM, and the ledger is TOLD so rather than left to infer it
        # from a zero. There the launcher publishes no `TMS_HOST_RING_*` and the
        # arming line prints `ring_H_mib=0`; `price()` would otherwise raise
        # W20, because `ring_bytes == 0` is its signal for "the predecessor
        # logged no ring table", i.e. a number it refuses to guess. Two facts,
        # one spelling -- separated here, at the ONE call site that already
        # knows the arms, exactly as the bounce is. The ledger never reads an
        # arm string.
        #
        # B4f CORRECTS THE PREDICATE AND THIS COMMENT WITH IT: #1327 wrote
        # "ON THE EXCHANGE ARM" and read the weight source alone, which is
        # right only when the exchange is the AUTHORITY. Under
        # `--pdflip-xchg-inject shadow` the ring is still armed and still the
        # authority, so boot weg2xsn13's dry-run handed the ledger 42.97 GiB of
        # ring beside a declaration that there was none. See
        # `ring_absent_by_design`.
        # #1369: computed together with the byte counts above, by
        # `pdflip_weights_cpu_backup_ring_kw` -- under `auto` (the default)
        # `weights_cpu_backup_armed` is exactly
        # `not ring_absent_by_design(weight_source, inject_mode)` (see
        # `main`'s resolution site), so this is unchanged there; it only
        # forces True for an explicit `off`, which must zero the ring even
        # under an arm `ring_absent_by_design` itself would call present
        # (e.g. plain `ring`, the hard-kill-switch case named in that
        # flag's help text).
        ring_absent_by_design=_ring_absent_by_design_kw,
    )
    # HOSTWARN2 (13.10.2026, user rule 06.10.): the ledger WARNS, it does not refuse.
    # `warn_only=True` turns W20/W21/W97 into HOST-RAM-WARN lines and returns the
    # priced arm; the other host-ledger refusals (an unmeasured term, a record
    # identity mismatch, a half-cited prior boot) are about a number that does not
    # exist, not about how much RAM the box has, and still raise.
    ledger_kw["warn_only"] = True
    if pin_m_mib and int(pin_m_mib) > 0:
        # #1317/#1318 THE PINNED ARM IS PRICED, NOT ASSUMED. The ladder is run
        # with exactly one rung, so a pinned arm is scored by the identical
        # model choose() applies to DEFAULT_ARMS. HOSTWARN2: an unfundable pinned
        # arm used to be re-raised here as W87 PdFlipPinnedArmRefused; it is now the
        # same price with a HOST-RAM-WARN W87 line in front of it, and the pin boots.
        arm, reap_headroom_gib, lines = host_ledger.choose(
            mi["MemTotal"], mi["MemAvailable"],
            arms=[(int(pin_s_gb) if pin_s_gb else 1, int(pin_m_mib))], **ledger_kw,
        )
        _unfunded_pin = [ln for ln in lines
                         if ln.startswith(f"{host_ledger.HOST_WARN_MARKER} W2")
                         and "Unfunded" in ln.split(" -- ", 1)[0]]
        if _unfunded_pin:
            lines = list(lines) + [
                f"{host_ledger.HOST_WARN_MARKER} W87 PinnedArmUnfunded -- --pin-ledger-arm-m {int(pin_m_mib)} "
                f"(S={int(pin_s_gb) if pin_s_gb else 1} M={int(pin_m_mib)}) was priced by the ladder itself and is "
                f"not fundable ({len(_unfunded_pin)} ledger warning line(s) above name the binding terms) -- "
                "WARNING ONLY: the pinned arm boots (user rule 06.10.: no refusal on host RAM; an OOM kill is "
                "the accepted risk)."
            ]
        lines = list(lines) + [
            f"PDFLIP-LEDGER ARM PINNED by --pin-ledger-arm-m/-s: S={int(pin_s_gb) if pin_s_gb else 1} M={int(pin_m_mib)} "
            f"launch={arm.launch_leftover_gib:.2f} GiB "
            f"run={arm.run_leftover_gib:.2f} GiB "
            f"run_peak={arm.predicted_run_peak_gib():.2f} GiB "
            f"host_ring_gib={arm.terms.get('host_ring_gib', float('nan')):.2f} "
            f"(#1318 rings derived: P={host_ledger.RING_P_MULT_GB_PER_S:.3f} + "
            f"D={host_ledger.RING_D_MULT_GB_PER_S:.3f} = "
            f"{host_ledger.RING_P_MULT_GB_PER_S + host_ledger.RING_D_MULT_GB_PER_S:.3f} xS "
            f"against the b0 reading {host_ledger.RING_B0_TOTAL_MULT_GB_PER_S:.3f} xS). "
            "The ladder was restricted to the pinned arm by the operator.",
        ]
        return arm, reap_headroom_gib, _baseline_lines + _bounce_lines + lines, cg
    arm, reap_headroom_gib, lines = host_ledger.choose(
        mi["MemTotal"], mi["MemAvailable"], **ledger_kw,
    )
    return arm, reap_headroom_gib, _baseline_lines + _bounce_lines + lines, cg
































# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------




def p_planned_need(cards, budgets_p, ns, log: Log) -> Optional[Dict[int, int]]:
    """S7b review MUSS 1: P's PLANNED peak per NVML card -- budget + overhead, exactly the number
    ``dual_share_planned_dc`` sizes D against (specimen 10240650: 9110/7550/7700 MiB, the argv budget alone
    6610/5050/5200). None = not derivable (the front's restart then refuses to spawn, by name)."""
    if not getattr(ns, "dual_layout", False):
        return None
    try:
        plan = dual_share_planned_dc(cards, list(budgets_p or []), getattr(ns, "extra_p", ""),
                                     int(getattr(ns, "dual_p_overhead_mib", 1500)))
        return {int(c.nvml_index): int(plan[c.uuid]) for c in cards}
    except Exception as e:  # noqa: BLE001 -- named; no plan = no spawn
        log(f"S7B P-RESPAWN-SPEC: P's planned peak not derivable ({type(e).__name__}: {e})")
        return None














# --------------------------------------------------------------------------
# H52: the host memory time series, bound to the boot's lifetime
# --------------------------------------------------------------------------
#
# MEASURED 24.09. 14:00Z (27B session, sar): 107 mem_timeseries.sh loggers of
# fnFL2x33 .. x153b were still sampling, 3397 process starts/s instead of 258,
# %sys 15.7 instead of 4.0. Root (H52): the logger was stopped ONLY by
# teardown(), and teardown() runs only for `--teardown PATH` and for a refusal
# AFTER a spawn (26 of 157 fnFL2 boots). The normal end -- LAUNCHED, the arm
# probes, the arm's own teardown TERMs front/launcher and kill -9s the
# schedulers -- never runs it, and the logger lives in its OWN session
# (setsid), so nothing the arm kills takes it along. The pid was right
# (`$!` == the logger, state json carried it: x157 3358575 in both) and the
# script dies on SIGTERM; nobody sent one.
#
# So the logger now watches its OWNER itself: a supervisor shell (session
# leader, pgid == its pid) runs the logger as a child and polls for a live
# launcher or front of THIS tag; when neither is left it TERMs its whole
# process group. teardown() still stops it explicitly (killpg + an anchored
# pgrep net on this tag's csv) and says so in one line.




# --------------------------------------------------------------------------
# #135: one process-group registry per boot, shared with the arm
# --------------------------------------------------------------------------
#
# MEASURED 26.09. (fnFL2h91v1): the launcher-armed front deadman stayed alive
# five minutes after the arm's teardown (teardown 08:00:50Z, its CRASH verdict
# 08:05:58Z) -- nothing but its own grace ceiling ended it, because the arm
# never learnt its pid (state json only, read by `--teardown PATH`, which the
# normal end does not run). Every helper this launcher starts is a session
# leader already (setsid / start_new_session, pgid == pid), so the missing
# piece is only the ENTRY: the arm's boot_helpers.sh reads
#   {EVIDENCE_DIR}/helpers_{tag}/{name}.pid = "PGID START OWNER OWNER_START NAME"
# and stops each group in its teardown (and its owner guard does it when the
# arm dies without one). OWNER is this launcher's parent -- the arm, which
# starts it with `setsid nohup python -m ...` (setsid execs in place, the arm
# stays the parent). START is /proc/<pid>/stat field 22, so a recycled pid is
# never signalled. The format is a contract with boot_helpers.sh.










# --------------------------------------------------------------------------
# H135b: the BOOT GROUPS in the same registry -- no headless boot
# --------------------------------------------------------------------------
#
# P, D and the front are session leaders (start_new_session, pgid == pid) and
# this launcher RETURNS after LAUNCHED, so nothing but the arm's teardown ever
# ended them. An arm killed by SIGKILL, the OOM killer or `timeout -k` runs no
# teardown: the boot went on headless, holding the cards past the gpuq window.
# PR_SET_PDEATHSIG is no answer -- the parent of the groups is THIS process,
# which exits by design after LAUNCHED, so it would kill every healthy boot.
# The arm's owner guard (boot_helpers.sh, setsid + `tail --pid`) already
# outlives the arm; it needs the boot's groups by identity. Each group is
# written as ``{name}.grp`` -- the SAME line as a helper's ``.pid`` entry
# (PGID START OWNER OWNER_START NAME, OWNER = the arm, this launcher's parent),
# but its own suffix, so the helper paths (bh_stop_all, bh_sweep_stale) never
# touch a serving group. The guard acts on them only when its owner armed the
# boot (``boot.arm``) and died without a teardown: ``--teardown`` first, then
# TERM/KILL per group. ``--teardown`` removes the entries.

BOOT_GROUP_NAMES = ("launcher", "P", "D", "front")






















def sleep_group(port: int, log: Log, name: str, weights_tags: List[str]) -> float:
    tags = ["kv_cache"] + list(weights_tags)
    t0 = time.time()
    code, body = http("POST", f"http://127.0.0.1:{port}/release_memory_occupation", {"tags": tags}, timeout=900)
    dt = (time.time() - t0) * 1000
    if code != 200:
        raise PdFlipLaunchRefused(f"sleep({name}) failed: HTTP {code} {body[:300]!r}")
    log(f"sleep({name}) OK in {dt:.0f} ms (tags {','.join(tags)}; flush BEFORE pause per record 1d MUST_FIX)")
    return dt




































#: The prefetch rate bound the carrier route is derived with: a prompt may be
#: at most this fraction of the host staging pool that must carry it (it
#: refused the 84k prompt on boot weg2ls4b2 at limit=27466 of pool 30518).
CARRIER_PREFETCH_FRACTION = 0.9































































































from flliper.srt.pdflip.launcher_d_rank_r2_L7 import (  # noqa: F401 -- re-export, behaviour-neutral move
    D_ONLY_LABEL,
    d_record_torch_caps,
    rank_reserve_kwargs,
)


def d_only_solve(ns, cards, dc_mib, log, *, user_reserve_by_card, p_split, chunk_layers):
    """The --d-only pass priced with the SAME terms as the map pass (dormant
    growth, driver carve, awake rest) and handed its card terms, so the #145
    card ledger exists and ``log_d_rank_vram_solve`` writes EXTEND-TRIM and
    EXTEND-STUECKELUNG here too. Without them the map pass wrote both, restored
    env_d, and this pass wrote neither: kvstage-donly (29.09. 11:02Z) ran with
    0 trims and its allocator cache climbed to 3.4-3.8 GiB per rank -- a D-solo
    boot measured a card the flip boot never has."""
    terms: List[Dict[str, object]] = []
    rest = d_awake_rest(cards, ns.profile)
    budgets = budgets_from_dc(
        cards, dc_mib, log, D_ONLY_LABEL,
        corridor_sample_path=ns.corridor_budget_sample, corridor_constrain=True,
        user_reserve_by_card=user_reserve_by_card,
        **dict(zip(("dormant_growth_mib", "dormant_growth_provenance"),
                   served_dormant_growth(cards, ns.profile))),
        charge_driver_carve=budget_charges_driver_carve(ns.profile),
        driver_carve_min_total_mib=driver_carve_min_total_mib(ns.profile),
        awake_rest_mib=rest[0], awake_rest_provenance=rest[1], terms_out=terms)
    log_d_rank_vram_solve(ns, cards, budgets, log, D_ONLY_LABEL,
                          p_split=p_split, chunk_layers=chunk_layers,
                          card_terms=(terms if rest[0] is not None else None))
    return budgets














#: DUAL-TP3PP3: --dual-mps on only with this env set to 1. MEASURED 30.09.:
#: repro v2 (gpuq scjhru, scripts/dual_layout/run_dualstress.sh) arm S1, where
#: both groups are clients of ONE MPS daemon and run extend-sized barlink
#: collectives (D 8 MiB AR, P 1 MiB AR + GEMM), WEDGED: P rc=124, D managed 31
#: rounds in 45 s (max 62 s). The same arm without MPS (S3) was clean. Metal
#: boots kw6pft and ndktv4 died the same way: both groups froze in the same
#: second while D ran an eager extend and P prefilled. The device-spin
#: collectives of the two groups couple through the shared MPS server context.
DUAL_MPS_OPT_IN_ENV = "FLLIPER_PDFLIP_DUAL_MPS_OPT_IN"


def resolve_dual_layout(ns) -> None:
    """DUAL-TP3PP3 (F26): the ONE place --dual-layout changes other flags.

    See docs/dev/dual-layout.md (user explanation of the dual layout: weights in the D layout, P loads only
    the byte difference, no all-reduce in prefill, GEMM share by load via MPS) before changing anything dual.

    Both groups stay awake, so their weights never sleep: --flip-weights is
    forced to 'resident' and --idle-layout pp (an idle flip rule) to 'tp'.
    Refused by name: --pdflip-d-adopt on (D would hold placeholders that only a
    flip fills -- and there is no flip), --dual-mps on without --dual-layout.
    Off: no-op."""
    if str(getattr(ns, "dual_unified_kv", "off")) == "on" and not getattr(ns, "dual_share", False):
        raise PdFlipDualLayoutRefused(
            "DUAL-TP3PP3: --dual-unified-kv on needs --dual-share (the card KV ledger lives next to "
            "the union image; P's pages are D's card)")
    if getattr(ns, "dual_share", False) and not getattr(ns, "dual_layout", False):
        print("PDFLIP-DUAL --dual-share implies --dual-layout", flush=True)
        ns.dual_layout = True
    dual = bool(getattr(ns, "dual_layout", False))
    refuse_dual_priority(ns)  # DUAL-SHARE: no-op unless one of its switches is set
    if not dual:
        if str(getattr(ns, "dual_mps", "off")) == "on":
            raise PdFlipDualLayoutRefused(
                "DUAL-TP3PP3: --dual-mps on needs --dual-layout (MPS only pays when both groups "
                "run kernels at the same time)")
        return
    # L15-01b: L1.5 (and the hot handover) need a SLEEPING P group; the dual
    # layout keeps both groups awake, so refuse the combination BY NAME before
    # any launch. refuse_dual names the armed switch; a silent off would let a
    # boot believe L1.5 holds what it does not.
    l15_msg = l15_plan.refuse_dual(["--dual-layout"], os.environ)
    if l15_msg is not None:
        raise PdFlipLaunchRefused(l15_msg)
    if str(getattr(ns, "dual_mps", "off")) == "on" and os.environ.get(DUAL_MPS_OPT_IN_ENV, "").strip() != "1":
        raise PdFlipDualLayoutRefused(
            "DUAL-TP3PP3: --dual-mps on needs the opt-in -- on 30.09. MPS wedged both groups: repro v2 "
            "(gpuq scjhru 30.09., run_dualstress.sh arm S1: MPS + extend-sized barlink collectives of "
            "D and P, P rc=124, D 31 rounds in 45 s; without MPS, arm S3, clean) and metal boots "
            "kw6pft/ndktv4 (both groups frozen in the same second); the root cause is fixed by b855aecfd3 "
            "and the release profile 27b-nvfp4-dual sets the opt-in. Set "
            f"{DUAL_MPS_OPT_IN_ENV}=1 to use MPS, or run --dual-mps off (time-slicing, latency guard "
            "--dual-p-duty, no green ladder).")
    if _d_adopt_armed(ns):
        raise PdFlipDualLayoutRefused(
            "DUAL-TP3PP3: --dual-layout with --pdflip-d-adopt on -- D would hold placeholder "
            "weights that only the first flip fills, and the dual layout never flips")
    if str(getattr(ns, "idle_layout", "tp")) == "pp":
        # The idle layout is a flip rule; the dual layout never flips, so it is
        # moot -- but the front would still read it, hence forced, not kept.
        print("PDFLIP-DUAL --idle-layout pp -> tp (the idle layout is a flip rule; "
              "the dual layout never flips)", flush=True)
        ns.idle_layout = "tp"
    if getattr(ns, "flip_weights", "family") != "resident":
        print(f"PDFLIP-DUAL --flip-weights {getattr(ns, 'flip_weights', 'family')} -> resident "
              "(both groups stay awake, no weight ever sleeps)", flush=True)
        ns.flip_weights = "resident"
    if str(getattr(ns, "p_barlink_bar1_window_mib", P_BARLINK_BAR1_WINDOW_MIB)) == P_BARLINK_BAR1_WINDOW_MIB:
        # Only the untouched default is replaced; an explicit operator value is kept.
        print(f"PDFLIP-DUAL --p-barlink-bar1-window-mib {P_BARLINK_BAR1_WINDOW_MIB} -> "
              f"{DUAL_P_BARLINK_BAR1_WINDOW_MIB} (P never sleeps, so both groups' BAR1 windows "
              f"share the 3080's 256 MiB aperture)", flush=True)
        ns.p_barlink_bar1_window_mib = DUAL_P_BARLINK_BAR1_WINDOW_MIB


#: DUAL-TP3PP3: group P's BAR1 windows while BOTH groups are awake. Metal
#: 30.09. boot f9fch3: with P at the flip default (24 + pp 96) D's dcp window
#: was REFUSED on both 3080s (40 MiB wanted, 39 free). The 3080 has 256 MiB
#: BAR1. The RM base, measured from that refusal line, is 19 MiB (NVML free 71
#: with P 120 + D 46 held). So 19 + P 120 + D 88 = 227 left 29 free, under the
#: 32 MiB reserve. The flip default only fits because P sleeps before D builds.
#: With P at 16 + 64: 19 + 80 + 88 = 187, 69 free, 37 above the reserve. The
#: price is more rounds per large pp send (per-round ~16 instead of ~24 MiB at
#: ~323 us/round), about a millisecond on an 80 MiB hidden-state send.
DUAL_P_BARLINK_BAR1_WINDOW_MIB = "16,PP_0=64"


def dual_kv_ledger_paths(ns, cards) -> List[str]:
    """DUAL-TP3PP3 unified KV: the card KV ledgers (pdflip/card_kv_ledger.py) of
    this boot, one per card; the ranks derive the same path from the tag and
    their card uuid. Empty unless --dual-share."""
    if not getattr(ns, "dual_share", False):
        return []
    from flliper.srt.pdflip.card_kv_ledger import ledger_path

    return [ledger_path(str(ns.tag), str(getattr(c, "uuid", ""))) for c in cards if getattr(c, "uuid", "")]


#: DUAL-TP3PP3 --dual-share: written by the launcher once D is READY; the P
#: stage waits for it before loading (model_executor/dual_stage_hull.py).
DUAL_D_READY_FILE = "d_ready"


def dual_share_env(ns, group: str) -> Dict[str, str]:
    """DUAL-TP3PP3 --dual-share: the union env of one group. D OWNS the card's
    image (and publishes its installed TP vectors with it), P BINDS the part of
    its stage that is D's shard on the same card. Off: {}."""
    if not getattr(ns, "dual_share", False):
        return {}
    import hashlib

    # SHORT on purpose: the card sockets live under it and a unix socket path
    # takes at most ~107 bytes; boot tags on this rig run past 60 characters.
    _h = hashlib.sha1(str(ns.tag).encode()).hexdigest()[:10]
    env = {"FLLIPER_PDFLIP_UNION_DIR": f"/dev/shm/wu-{_h}",
           "FLLIPER_PDFLIP_UNION_MODE": "own" if group == "D" else "bind",
           # Only the main model is shared: the draft is TP3-sharded in D and
           # TP1 in P, so none of its tensors has one form in both groups
           # (union_arena_bind.UNION_ROLES_ENV; metal 30.09. ...09300226).
           "FLLIPER_PDFLIP_UNION_ROLES": "main",
           # Both groups stay awake: rank-side rules that assume the other
           # group sleeps read this (scheduler._pdflip_store_short_max_cycles).
           "FLLIPER_PDFLIP_DUAL_LAYOUT": "1",
           # DUAL ANCHOR N-1 (pdflip/dual_anchor_claim.py): the exact bigram
           # keying in BOTH groups, so P's N-1 anchor node has N-1 units and
           # D claims it whole (uncached 1 after every hand-back, not 2)
           "FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT": "1",
           # the card KV ledgers' name space: the ranks derive the same path
           # as the front (dual_kv_ledger_paths) from this tag
           "FLLIPER_PDFLIP_DUAL_KV_TAG": str(ns.tag)}
    if group == "P" and str(getattr(ns, "dual_unified_kv", "off")) == "on":
        # unified KV (user orders 30.09. 07:10Z/07:25Z): P's pool is virtually
        # this big; its pages come from the card pool (pdflip/dual_p_kv_stage.py)
        env["FLLIPER_PDFLIP_DUAL_P_KV_MAX_TOKENS"] = str(int(getattr(ns, "dual_p_kv_max_tokens", 0) or 0))
    if group == "D" and str(getattr(ns, "dual_unified_kv", "off")) == "on":
        env["FLLIPER_PDFLIP_DUAL_D_KV_MAX_TOKENS"] = str(int(getattr(ns, "dual_d_kv_max_tokens", 0) or 0))
    if group == "P":
        env["FLLIPER_PDFLIP_DUAL_SHARE"] = "1"
    if group == "P" and dual_p_sleep_armed(ns):
        # D PRIORITY stage 2: P's weights are NOT resident (the resident arm
        # refuses any weights release by name), backed up by TMS at the pause,
        # and outside the exchange; D stays resident, D never sleeps
        env.update(DUAL_P_SLEEP_GROUP_ENV)
    return env


def dual_alloc_cache_book_env(ns, cards: List[Card], log: Log) -> Dict[str, str]:
    """PDFLIP-ALLOC-CACHE-BOOK (item 170): P's env carrying the MiB per card
    ordinal that the card ledger books at P's join for D's allocator-cache
    growth (pdflip/budget_rest.py, record ``D_DUAL_ALLOC_CACHE_BOOK_MIB`` of the
    profile, fmt = the checkpoint's format). RECORD > BUILTIN > UNMEASURED:
    a profile / format without the record books nothing and the line says
    UNMEASURED; so does any non-dual-share boot ({} and no line)."""
    if not getattr(ns, "dual_share", False) or str(getattr(ns, "dual_unified_kv", "off")) != "on":
        return {}
    from flliper.srt.pdflip import dual_p_kv_stage as _dual_p_kv

    name = _budget_rest.alloc_cache_record_name("D")
    fmt = d_reshard_format(str(getattr(ns, "model", "") or ""))
    sel = pdflip_form.profile_record(name, getattr(ns, "profile", None), fmt=fmt)
    if sel is None:
        log(f"{_budget_rest.ALLOC_CACHE_MARKER} {_budget_rest.SOURCE_UNMEASURED} profile="
            f"{getattr(ns, 'profile', None)} fmt={fmt or '-'}: no record {name} of this checkpoint and "
            f"form -- the card ledger books no allocator-cache growth at P's join (named, not measured); "
            f"measure with python -m flliper.srt.pdflip.budget_rest --dual-d-log")
        return {}
    vals = list(sel.record.value)
    if len(vals) != len(cards):
        raise PdFlipLaunchRefused(
            f"{_budget_rest.ALLOC_CACHE_MARKER}: {name} has {len(vals)} entries for {len(cards)} cards; "
            f"re-measure with python -m flliper.srt.pdflip.budget_rest --dual-d-log")
    log(f"{_budget_rest.ALLOC_CACHE_MARKER} {_budget_rest.SOURCE_RECORD} {name} fmt={fmt} "
        f"boot {','.join(sel.record.boots)}: " + ", ".join(
            f"ordinal {i} nvml{c.nvml_index} " + ("UNMEASURED" if v is None else f"{int(v)} MiB")
            for i, (c, v) in enumerate(zip(cards, vals))) + " -- booked into the card ledger at P's join")
    return {_dual_p_kv.ALLOC_CACHE_BOOK_ENV: ",".join("" if v is None else str(int(v)) for v in vals)}


def dual_outside_free_env(ns, cards: List[Card], user_reserve_by_card: Optional[Dict[str, int]],
                          log: Log) -> Dict[str, str]:
    """PDFLIP-USER-RESERVE-BOOK (user rule 24.10.): P's env carrying the MiB per card ordinal that the card keeps
    free BESIDE its KV ledger (record ``D_DUAL_OUTSIDE_FREE_MIB`` of the profile, fmt = the checkpoint's format,
    pdflip/budget_rest.py). P's join books ``max(0, reserve - outside_free)`` out of the card budget so that the
    user's reserve is the floor of the card's free memory (dual_p_kv_stage.book_user_reserve). RECORD >
    UNMEASURED: a profile / format without the record books nothing and the line says so loudly. {} and no line
    when the layout is not dual-share + unified KV or the user stated no reserve (byte-identical to before);
    ``FLLIPER_PDFLIP_DUAL_USER_RESERVE_BOOK=0`` is the named off switch (default ON)."""
    if not getattr(ns, "dual_share", False) or str(getattr(ns, "dual_unified_kv", "off")) != "on":
        return {}
    if not any(int(v) > 0 for v in (user_reserve_by_card or {}).values()):
        return {}
    from flliper.srt.pdflip import dual_p_kv_stage as _dual_p_kv

    marker = _budget_rest.RESERVE_BOOK_MARKER
    if not _dual_p_kv.reserve_book_armed():
        log(f"{marker} OFF ({_dual_p_kv.RESERVE_BOOK_SWITCH_ENV}=0): the user's reserve is NOT booked in the card "
            f"ledger on this boot (switch off, named)")
        return {_dual_p_kv.RESERVE_BOOK_SWITCH_ENV: "0"}
    name = _budget_rest.outside_free_record_name("D")
    fmt = d_reshard_format(str(getattr(ns, "model", "") or ""))
    sel = pdflip_form.profile_record(name, getattr(ns, "profile", None), fmt=fmt)
    if sel is None:
        log(f"{marker} {_budget_rest.SOURCE_UNMEASURED} profile={getattr(ns, 'profile', None)} fmt={fmt or '-'}: no "
            f"record {name} of this checkpoint and form -- the user's reserve is NOT booked in the card ledger "
            f"on this boot (named, not measured); measure the tightest 'LEDGER-PHYS' phys_free - ledger_free of a "
            f"serving boot per card")
        return {}
    vals = list(sel.record.value)
    if len(vals) != len(cards):
        raise PdFlipLaunchRefused(
            f"{marker}: {name} has {len(vals)} entries for {len(cards)} cards; re-measure the tightest "
            f"'LEDGER-PHYS' phys_free - ledger_free per card")
    parts = []
    for i, (c, v) in enumerate(zip(cards, vals)):
        res = int((user_reserve_by_card or {}).get(c.uuid, 0))
        parts.append(f"ordinal {i} nvml{c.nvml_index} reserve {res} MiB, outside_free "
                     + ("UNMEASURED" if v is None else f"{int(v)} MiB -> book {_dual_p_kv.reserve_deficit_mib(res, int(v))} MiB"))
    log(f"{marker} {_budget_rest.SOURCE_RECORD} {name} fmt={fmt} boot {','.join(sel.record.boots)}: "
        + ", ".join(parts) + " -- booked out of the card ledger at P's join")
    return {_dual_p_kv.OUTSIDE_FREE_ENV: ",".join("" if v is None else str(int(v)) for v in vals)}




def dual_p_sleep_share_supported() -> bool:
    """Steps 1-3 of P's sleep under --dual-share are in this tree: the shared part
    loads in a transient scope (pdflip_memory_saver.transient_load_scope) and the
    untagged-live riegel guards it."""
    from flliper.srt.managers import pdflip_memory_saver as _ms

    return hasattr(_ms, "transient_load_scope") and hasattr(_ms, "assert_no_untagged_live")


#: DUAL P SLEEP: the env group P sleeps under (dual layout, unified KV, --dual-p-sleep
#: on). The physical TMS backup of the weights is gated by
#: weight_exchange.weights_cpu_backup_armed() -- the ENV, not the argv bit -- and the
#: dual boots ship 'off' (gmps7 PDFLIP-GROUP-ENV P: FLLIPER_PDFLIP_WEIGHTS_CPU_BACKUP=off):
#: a sleep would wake to undefined weights. And with the exchange armed
#: (FLLIPER_PDFLIP_WEIGHT_SOURCE=exchange, inject authoritative) the sleep leg would
#: deposit for a peer flip and the wake inject from one; the dual layout never flips.
DUAL_P_SLEEP_GROUP_ENV = {
    "FLLIPER_PDFLIP_WEIGHTS_RESIDENT": "0",
    "FLLIPER_PDFLIP_WEIGHTS_CPU_BACKUP": "on",
    "FLLIPER_PDFLIP_WEIGHT_SOURCE": "ring",
}


def dual_p_sleep_choice(ns) -> str:
    """'on' / 'off': --dual-p-sleep as given, or its DEFAULT -- on when
    --dual-share is set (the dual-share sleep is metal-proven: gmps12, SLEEP-LEND
    7958691840 / 3219128320 / 3508535296 B and WAKE-RECLAIM on all three P
    stages), off otherwise. The metal probe (FLLIPER_PDFLIP_DUAL_P_SLEEP_PROBE_S) is
    never part of the default."""
    v = getattr(ns, "dual_p_sleep", None)
    if v is None:
        return "on" if bool(getattr(ns, "dual_share", False)) else "off"
    return str(v)


def dual_p_sleep_armed(ns) -> bool:
    """D PRIORITY stage 2 is armed: dual layout + unified KV + --dual-p-sleep on
    (explicit, or the default under --dual-share: :func:`dual_p_sleep_choice`).
    An explicit 'on' under --dual-share in a tree without the transient load is
    REFUSED by name (:class:`PdFlipDualPSleepShareRefused`); the DEFAULT there
    stays off instead."""
    on = (bool(getattr(ns, "dual_layout", False))
          and str(getattr(ns, "dual_unified_kv", "off")) == "on"
          and dual_p_sleep_choice(ns) == "on")
    if on and bool(getattr(ns, "dual_share", False)) and not dual_p_sleep_share_supported():
        if getattr(ns, "dual_p_sleep", None) is None:
            return False
        raise PdFlipDualPSleepShareRefused(
            "W-DUAL-P-SLEEP-SHARE: --dual-p-sleep on with --dual-share -- P's weights would load inside the "
            "memory-saver's private MemPool and the union bind's freed copies would stay reserved there "
            "(gmps9: 'card free 3.35 -> 3.35 GiB', P-PP0 KV budget refused). Use --dual-p-sleep off (default).")
    return on


def apply_dual_p_sleep(ns, spec_p, log) -> bool:
    """D PRIORITY stage 2 on P's spec: the weights CPU backup, never through the
    host ring (a resting pinned pool would be a permanent host image). The
    not-resident env comes with dual_share_env(ns, "P")."""
    if not dual_p_sleep_armed(ns):
        return False
    spec_p.argv = dual_p_sleep_argv(ns, spec_p.argv)
    for _k in ("TMS_HOST_RING_DIR", "TMS_HOST_RING_MAP", "TMS_HOST_RING_EPOCH", "TMS_HOST_RING_FORM"):
        spec_p.env.pop(_k, None)
    log("PDFLIP-DUAL-P-SLEEP armed: P --enable-weights-cpu-backup, weights not resident, no host ring "
        "(image allocated at the pause, freed after the restore)")
    return True


def dual_p_sleep_argv(ns, argv_p) -> List[str]:
    """P's argv with the weights CPU backup when stage 2 is armed (P only; the
    host ring stays absent for P -- dual_p_sleep_env pops it -- so the image is
    the stock per-pause pinned block, freed after the restore)."""
    argv_p = list(argv_p)
    if dual_p_sleep_armed(ns) and "--enable-weights-cpu-backup" not in argv_p:
        argv_p.append("--enable-weights-cpu-backup")
    return argv_p


#: D's verify tokens per seat and round on the 27B dual (DFlash: the draft's
#: block plus the bonus token); the look-ahead's per-seat term
DUAL_D_VERIFY_TOKENS = 4


def dual_d_air_tokens(d_bs: int, chunk: int = CHUNKED_PREFILL_TOKENS,
                      verify: int = DUAL_D_VERIFY_TOKENS) -> int:
    """D's look-ahead in tokens (d_mem_sched.air_tokens): one extend chunk plus
    one decode round of every seat with the draft's tokens."""
    return int(chunk) + max(1, int(d_bs)) * max(1, int(verify))


def dual_front_kv_tag_env(ns, environ=None) -> dict:
    """#1495: the front env carries the boot's --tag as FLLIPER_PDFLIP_DUAL_KV_TAG so
    its dual P stage-file reads (grant/air/weights/card_room/p_lent, d_signal,
    ANCHOR-OWED) key the files like the ranks do. Only in the dual layout; DEFAULT ON
    (user order 14.10.: new features on; 27B dual 21.10. 22:57:17Z the blind front
    resumed P past PP0's refused reclaim and the 5090 ran to driver_free 2.7 MiB) --
    off only with FLLIPER_PDFLIP_DUAL_FRONT_KV_TAG_FIX=0 (then {} = the old front env);
    an operator-set FLLIPER_PDFLIP_DUAL_KV_TAG is never overwritten."""
    environ = os.environ if environ is None else environ
    if str(environ.get("FLLIPER_PDFLIP_DUAL_FRONT_KV_TAG_FIX", "")).strip().lower() in ("0", "false", "off", "no"):
        return {}
    if not getattr(ns, "dual_layout", False) or not getattr(ns, "tag", None):
        return {}
    if environ.get("FLLIPER_PDFLIP_DUAL_KV_TAG"):
        return {}
    return {"FLLIPER_PDFLIP_DUAL_KV_TAG": str(ns.tag)}


def dual_p_sleep_front_env(ns, d_air_tokens: int) -> Dict[str, str]:
    """The front's capability switch and D's look-ahead in tokens."""
    if not dual_p_sleep_armed(ns):
        return {}
    from flliper.srt.pdflip import dual_d_priority as _ddp

    return {_ddp.P_SLEEP_ENV: "1", "FLLIPER_PDFLIP_DUAL_D_AIR_TOKENS": str(int(d_air_tokens))}


def dual_p_cut_from_argv(argv) -> str:
    """The --pp-stage-ratio group P was launched with ('' if absent)."""
    argv = list(argv or ())
    for i, t in enumerate(argv):
        if t == "--pp-stage-ratio" and i + 1 < len(argv):
            return str(argv[i + 1])
        if str(t).startswith("--pp-stage-ratio="):
            return str(t).split("=", 1)[1]
    return ""


def dual_share_planned_dc(cards, budgets_p: List[int], extra_p: str, overhead_mib: int) -> Dict[str, int]:
    """What P will hold per card, from its PLAN: the effective P budget (an
    --extra-p '--rank-gpu-memory-mib' lowers the launcher's, W100) plus what P
    holds outside it. Indexed like ``cards`` (P rank i on cards[i])."""
    eff = [int(b) for b in budgets_p]
    toks = shlex.split(extra_p or "")
    for i, t in enumerate(toks):
        raw = None
        if t == "--rank-gpu-memory-mib" and i + 1 < len(toks):
            raw = toks[i + 1]
        elif t.startswith("--rank-gpu-memory-mib="):
            raw = t.split("=", 1)[1]
        if raw is not None:
            vals = [int(x) for x in raw.split(",") if x]
            if len(vals) == len(eff):
                eff = [min(a, b) for a, b in zip(eff, vals)]
    return {c.uuid: int(eff[i]) + int(overhead_mib) for i, c in enumerate(cards)}


def dual_duty_env(ns) -> Dict[str, str]:
    """DUAL-TP3PP3 --dual-p-duty below 1: P's throttle env (the front gets the
    same signal file on its argv). {} when off."""
    duty = float(getattr(ns, "dual_p_duty", 1.0))
    if not (0.0 < duty <= 1.0):
        raise PdFlipDualLayoutRefused(f"DUAL-TP3PP3: --dual-p-duty {duty} outside (0, 1]")
    if duty >= 1.0 or not getattr(ns, "dual_layout", False):
        return {}
    from flliper.srt.pdflip.dual_duty import dbusy_path

    return {"FLLIPER_PDFLIP_DUAL_P_DUTY": f"{duty:.3f}",
            "FLLIPER_PDFLIP_DUAL_DBUSY_FILE": dbusy_path(ns.tag)}


def dual_p_sm_env(ns) -> Dict[str, str]:
    """DUAL-TP3PP3: P's MPS SM share (only with --dual-mps on, only below 100)."""
    pct = int(getattr(ns, "dual_p_sm_pct", 100))
    if not (1 <= pct <= 100):
        raise PdFlipDualLayoutRefused(f"DUAL-TP3PP3: --dual-p-sm-pct {pct} outside 1..100")
    if pct >= 100 or not getattr(ns, "dual_layout", False) or str(getattr(ns, "dual_mps", "off")) != "on":
        return {}
    return {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(pct)}


#: DUAL-SHARE --dual-priority modes (pdflip/dual_share.MODES).
DUAL_PRIORITY_MODES = ("p", "balanced", "d", "dynamic")


def dual_priority_armed(ns) -> bool:
    """Any DUAL-SHARE switch set (each default off)."""
    return (getattr(ns, "dual_priority", None) is not None
            or str(getattr(ns, "dual_d_capture_prio", "off")) == "on"
            or str(getattr(ns, "dual_p_mps_low_prio", "off")) == "on"
            or str(getattr(ns, "dual_green_ladder", "off")) != "off")




def dual_priority_env(ns, group: str, log=None) -> Dict[str, str]:
    """DUAL-SHARE (item 800): one group's env of the P/D share switches. {} for
    every switch left off, and {} outside --dual-layout (resolve_dual_layout
    refuses the switches there by name)."""
    if not getattr(ns, "dual_layout", False):
        return {}
    from flliper.srt.pdflip import dual_share as _dsh

    env: Dict[str, str] = {}
    if group == "D":
        if str(getattr(ns, "dual_d_capture_prio", "off")) == "on":
            env[_dsh.D_CAPTURE_PRIO_ENV] = "1"
        return env
    mode = getattr(ns, "dual_priority", None)
    if mode is not None:
        acts = _dsh.parse_actuators(str(getattr(ns, "dual_share_actuators", "chunk") or "chunk"))
        env[_dsh.CTL_ENV] = _dsh.ctl_path(ns.tag)
        env[_dsh.ACT_ENV] = ",".join(acts)
        if p_prefill_graph_buckets():
            env[_dsh.BUCKETS_ENV] = ",".join(str(b) for b in p_prefill_graph_buckets())
        if "duty" in acts:
            env[_dsh.DUTY_ENV] = "1"
        if str(getattr(ns, "dual_green_ladder", "off")) != "off":
            # STAGE 3 (pdflip/dual_green.py): the switch is the P rank's env, NOT inherited from the launcher's own
            env[_dsh.GREEN_LADDER_ENV] = "1"
            if str(getattr(ns, "dual_green_ladder", "off")) == "hold":
                env["FLLIPER_PDFLIP_DUAL_GREEN_HOLD"] = "1"
    if str(getattr(ns, "dual_p_mps_low_prio", "off")) == "on":
        penv, line = _dsh.mps_client_priority_env(
            mps_on=str(getattr(ns, "dual_mps", "off")) == "on", driver_text=_read_driver_text())
        env.update(penv)
        if log is not None:
            log(line)
    return env


def dual_priority_front_argv(ns) -> List[str]:
    """DUAL-SHARE: the front's controller terms ([] unless --dual-priority)."""
    mode = getattr(ns, "dual_priority", None)
    if mode is None or not getattr(ns, "dual_layout", False):
        return []
    from flliper.srt.pdflip import dual_share as _dsh

    return ["--dual-priority", str(mode), "--dual-share-ctl", _dsh.ctl_path(ns.tag),
            "--dual-d-min-rate-tps", f"{float(getattr(ns, 'dual_d_min_rate_tps', 0.0)):g}",
            "--dual-p-min-share", f"{float(getattr(ns, 'dual_p_min_share', 0.25)):g}",
            "--dual-share-actuators",
            ",".join(_dsh.parse_actuators(str(getattr(ns, "dual_share_actuators", "chunk") or "chunk")))] + (
        ["--dual-green-ladder", str(getattr(ns, "dual_green_ladder", "off"))]
        if str(getattr(ns, "dual_green_ladder", "off")) != "off" else [])


def refuse_dual_priority(ns) -> None:
    """DUAL-SHARE: named refusals before any launch."""
    if not dual_priority_armed(ns):
        return
    if not getattr(ns, "dual_layout", False):
        raise PdFlipDualLayoutRefused(
            "DUAL-SHARE: --dual-priority / --dual-d-capture-prio / --dual-p-mps-low-prio need --dual-layout "
            "(the P/D share exists only while both groups run at once; the flip form stays untouched)")
    from flliper.srt.pdflip import dual_share as _dsh

    try:
        acts = _dsh.parse_actuators(str(getattr(ns, "dual_share_actuators", "chunk") or "chunk"))
    except ValueError as e:
        raise PdFlipDualLayoutRefused(str(e))
    share = float(getattr(ns, "dual_p_min_share", 0.25))
    if not (0.0 < share <= 1.0):
        raise PdFlipDualLayoutRefused(f"DUAL-SHARE: --dual-p-min-share {share} outside (0, 1]")
    if float(getattr(ns, "dual_d_min_rate_tps", 0.0)) < 0.0:
        raise PdFlipDualLayoutRefused("DUAL-SHARE: --dual-d-min-rate-tps must be >= 0")
    if str(getattr(ns, "dual_green_ladder", "off")) != "off":
        if getattr(ns, "dual_priority", None) is None:
            raise PdFlipDualLayoutRefused(
                "DUAL-SHARE: --dual-green-ladder needs --dual-priority (the front's controller writes the stage "
                "the green streams follow)")
        if "green" not in acts:
            raise PdFlipDualLayoutRefused(
                "DUAL-SHARE: --dual-green-ladder needs 'green' in --dual-share-actuators (e.g. green,duty: duty is "
                "the named fallback for rungs the ladder cannot serve)")
        if str(getattr(ns, "dual_mps", "off")) != "on":
            raise PdFlipDualLayoutRefused(
                "DUAL-SHARE: --dual-green-ladder needs --dual-mps on (the green streams share the card with D "
                "through MPS; the metal probe ran under MPS)")
        if int(getattr(ns, "dual_p_kv_max_tokens", 0) or 0) <= 0:
            raise PdFlipDualLayoutRefused(
                "DUAL-SHARE: --dual-green-ladder needs --dual-p-kv-max-tokens > 0 (the P-side dual gate "
                "dual_p_kv_stage.armed() reads it: without it the ladder would silently never arm on the P ranks)")
        if int(getattr(ns, "dual_p_sm_pct", 100)) < 100:
            raise PdFlipDualLayoutRefused(
                "DUAL-SHARE: --dual-green-ladder and --dual-p-sm-pct below 100 exclude each other (a static MPS "
                "percentage caps P at its CLIENT START and the ladder could never lift it: P's "
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE must stay unset)")
    if (getattr(ns, "dual_priority", None) is not None and "duty" in acts
            and float(getattr(ns, "dual_p_duty", 1.0)) < 1.0):
        raise PdFlipDualLayoutRefused(
            "DUAL-SHARE: --dual-share-actuators duty with --dual-p-duty below 1 -- two duty throttles on "
            "PP0; take one (the share duty follows the rung, --dual-p-duty is constant)")


def start_dual_mps(ns, log, dry: bool) -> Dict[str, str]:
    """DUAL-TP3PP3 --dual-mps on: one PRIVATE MPS control daemon for this
    boot (pipe dir under /tmp/pdflip-dual-mps-<tag>, so nothing outside the boot
    can attach), started before the groups; returns the client env both
    groups get. The daemon lives until teardown() (or the early-exit paths of cli())
    calls :func:`stop_dual_mps`. Off: {}.

    Why MPS: P and D are two MPS clients per card (docs/dev/dual-layout.md); the static SM percentage is
    dual_p_sm_env, the dynamic share is the green ladder (pdflip/dual_green.py), not this daemon."""
    if not getattr(ns, "dual_layout", False) or str(getattr(ns, "dual_mps", "off")) != "on":
        return {}
    root = dual_mps_root(ns.tag)
    env = {"CUDA_MPS_PIPE_DIRECTORY": root + "/pipe", "CUDA_MPS_LOG_DIRECTORY": root + "/log"}
    if not dry:
        os.makedirs(env["CUDA_MPS_PIPE_DIRECTORY"], exist_ok=True)
        os.makedirs(env["CUDA_MPS_LOG_DIRECTORY"], exist_ok=True)
        r = subprocess.run(["nvidia-cuda-mps-control", "-d"], env={**os.environ, **env},
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise PdFlipDualLayoutRefused(
                f"DUAL-TP3PP3: nvidia-cuda-mps-control -d failed rc={r.returncode}: "
                f"{(r.stderr or r.stdout).strip()[:300]}")
    log(f"PDFLIP-DUAL MPS daemon {'(dry, not started)' if dry else 'started'}: pipe "
        f"{env['CUDA_MPS_PIPE_DIRECTORY']} -- P and D are MPS clients on every card")
    return env


#: Root of one boot's private MPS daemon (pipe + log dir); only paths under this prefix are ever removed.
DUAL_MPS_ROOT_PREFIX = "/tmp/pdflip-dual-mps-"
DUAL_MPS_CONTROL = "nvidia-cuda-mps-control"
#: How long the daemon's own ``quit`` may take to end the daemon and its server before TERM, then 2 s before KILL.
DUAL_MPS_STOP_BOUND_S = 8.0


def dual_mps_root(tag: str) -> str:
    return f"{DUAL_MPS_ROOT_PREFIX}{tag}"






def stop_dual_mps(tag: str, root: str = "", *, armed: bool = False, run=subprocess.run, scan=_mps_pids_on_pipe,
                  alive=_alive_not_zombie, kill=os.kill, sleep=time.sleep, clock=time.monotonic,
                  bound_s: float = DUAL_MPS_STOP_BOUND_S, rmtree=shutil.rmtree) -> str:
    """Stop this boot's private MPS daemon (``start_dual_mps``) and return the teardown line.

    The MPS daemon and its server are not children of any group: no TERM of the group pids reaches them, and
    a server that outlives the groups keeps a context on every card (boot dkr27bnvfp4dualbar1drei10271237:
    ``cards_empty=False`` after a refusal, the teardown lines named no MPS process). Order: ``quit`` on the
    daemon's own control pipe (with ``get_server_list`` first, to name the server pids), wait, TERM and then KILL
    what is left (daemon and server found by argv0 + this boot's pipe dir in their environment), then remove the
    pipe/log dir. Nothing to do and nothing armed -> ``""`` (a boot without --dual-mps prints nothing)."""
    root = root or (dual_mps_root(tag) if tag else "")
    if not root or not root.startswith(DUAL_MPS_ROOT_PREFIX):
        return ""
    pipe, logd = root + "/pipe", root + "/log"
    if not os.path.isdir(root):
        return (f"PDFLIP-TEARDOWN MPS daemon absent: {root} does not exist (nothing to stop)" if armed else "")
    before = scan(pipe)
    pids = set(before["daemon"]) | set(before["server"])
    env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe, "CUDA_MPS_LOG_DIRECTORY": logd}
    quit_rc = "not-run"
    if pids or os.path.exists(pipe + "/control"):
        # only against a daemon that is there: nvidia-cuda-mps-control without -d would START one in the foreground
        try:
            r = run([DUAL_MPS_CONTROL], input="get_server_list\nquit\n", env=env, capture_output=True, text=True,
                    timeout=15)
            quit_rc = str(r.returncode)
            pids |= {int(x) for x in (r.stdout or "").split() if x.isdigit()}
        except FileNotFoundError:
            quit_rc = "no-control-binary"
        except subprocess.TimeoutExpired:
            quit_rc = "timeout"
        except OSError as e:
            quit_rc = f"oserror:{e.__class__.__name__}"
    t0 = clock()
    while clock() - t0 < bound_s and any(alive(p) for p in pids):
        sleep(0.1)
    termed = sorted(p for p in pids if alive(p))
    for p in termed:
        try:
            kill(p, signal.SIGTERM)
        except OSError:
            pass
    t1 = clock()
    while termed and clock() - t1 < 2.0 and any(alive(p) for p in termed):
        sleep(0.1)
    killed = sorted(p for p in termed if alive(p))
    for p in killed:
        try:
            kill(p, signal.SIGKILL)
        except OSError:
            pass
    rest = scan(pipe)  # a server the daemon spawned while it was quitting
    for p in sorted(set(rest["daemon"]) | set(rest["server"])):
        if alive(p) and p not in pids:
            try:
                kill(p, signal.SIGKILL)
            except OSError:
                pass
            killed.append(p)
            pids.add(p)
    still = sorted(p for p in pids if alive(p))
    rmtree(root, ignore_errors=True)
    verdict = "stopped" if not still else "NOT stopped (process(es) survive KILL)"
    servers = sorted(pids - set(before["daemon"]))
    return (f"PDFLIP-TEARDOWN MPS daemon {verdict} pipe={pipe} quit_rc={quit_rc} "
            f"daemon_pids={sorted(before['daemon'])} server_pids={servers} term={termed} kill={killed} "
            f"alive={still} dir_removed={not os.path.isdir(root)}")


def stop_dual_mps_of_state(st, log=print) -> str:
    """The early-exit paths of cli() (refusal or crash BEFORE any group spawned): teardown() does not run
    there, so the daemon ``start_dual_mps`` started is stopped here. ``st``: the live BootState (or None)."""
    if st is None:
        return ""
    root = getattr(st, "dual_mps_root", "") or ""
    line = stop_dual_mps(getattr(st, "tag", ""), root, armed=bool(root))
    if line:
        log(line)
    return line












def budgets_from_dc(
    cards: List[Card],
    dc_mib: Dict[str, int],
    log: Log,
    label: str,
    overshoot_mib: Optional[List[int]] = None,
    overshoot_provenance: str = "",
    corridor_sample_path: Optional[str] = None,
    corridor_constrain: bool = False,
    user_reserve_by_card: Optional[Dict[str, int]] = None,
    dormant_growth_mib: Optional[List[int]] = None,
    dormant_growth_provenance: str = "",
    charge_driver_carve: bool = False,
    driver_carve_min_total_mib: int = 0,
    awake_rest_mib: Optional[List[Optional[int]]] = None,
    awake_rest_provenance: str = "",
    terms_out: Optional[List[Dict[str, object]]] = None,
    booked_rest_mib: Optional[Sequence[Optional[int]]] = None,
    booked_rest_provenance: str = "",
    l15_mib: Optional[Sequence[int]] = None,
) -> List[int]:
    """Per card the budget of group ``label``; ``terms_out`` (a list) receives
    the terms per card in order, for the #145 card ledger (rc12c: the card
    side prices the SAME terms, so budget and card cannot disagree).

    ``booked_rest_mib`` (VRAM-GRUNDGESETZ 29.09., pdflip/budget_rest.py): per
    card the MEASURED awake rest beyond the budget line (record
    ``<G>_AWAKE_REST_BOOKED_MIB``). Where it is given the card's budget is
    ``total - driver carve - dormant_other - growth - rest`` and NOTHING else:
    the rest contains the floor's transient, and the user reserve, the
    builtin awake overshoot, a budget-relative overshoot and the awake rest of
    the form are what it replaces -- no reserve is charged beside it. A
    ``None`` entry is UNMEASURED: that card keeps the terms below, and its
    line says so by name.

    ``l15_mib`` (L15-01b, pdflip/l15_plan.py): the L1.5 post per card, in the
    SAME order as ``cards``. Where given, each card's budget subtracts its
    post BEFORE the ``// 8 * 8`` rounding, the term dict gains an ``l15`` key,
    and the budget line carries a ``- l15 {mib} (L1.5 post)`` suffix. ``None``
    (the L1.5 master switch off) leaves the pass byte-identical: no term, no
    suffix.
    """
    out = []
    if overshoot_mib is not None and len(overshoot_mib) < len(cards):
        # AP1 1006 (c): a record vector written for fewer cards than the inventory
        # (P_OVERSHOOT_MIB holds three entries) used to die here as a bare
        # IndexError at ``overshoot_mib[i]``. Same refusal, now by name; a longer
        # vector is read as before (the first N entries).
        raise PdFlipLaunchRefused(
            f"{card_identity.CODE_UNCALIBRATED}: the awake-overshoot vector of group {label} "
            f"({overshoot_provenance or 'record'}) has {len(overshoot_mib)} entries for {len(cards)} cards; "
            "a positional measurement is not stretched over cards it was not taken on "
            "(pdflip/inventory_view.py derives it by card class for a subset of the calibrated inventory) "
            "-- measure this inventory or name the vector.")
    l15_given = l15_mib is not None
    l15_vec: List[int] = [None] * len(cards) if not l15_given else [int(v) for v in l15_mib]
    if l15_given and len(l15_vec) != len(cards):
        raise PdFlipLaunchRefused(
            f"L15-01b: {len(l15_vec)} L1.5 posts for {len(cards)} cards -- a partial "
            "vector never prices a card it never saw")
    # #1257c: ONE derivation, per card, for the group this budget is for.
    # ``user_reserve_by_card`` is the operator's external headroom (default 0);
    # it RAISES the floor and therefore lowers the budget by exactly as much,
    # which is what the knob is for.
    # #1257c REFUTER FIX 2: the LABEL names the pass ("D(dry, expectation)"),
    # the GROUP names the floor it keys. They were the same string, so a
    # decorated dry pass and the real pass would have read different floors the
    # day D is measured. Both are printed from here on, because an operator
    # who greps ``group=D`` must find the dry pass too.
    group = corridor_guard.normalise_group(label)
    booked = [None] * len(cards) if booked_rest_mib is None else list(booked_rest_mib)
    if len(booked) != len(cards):
        raise PdFlipLaunchRefused(
            f"{_budget_rest.MARKER}: {len(booked)} booked rests for {len(cards)} cards "
            f"({booked_rest_provenance}) -- a partial vector never prices a card it never saw")
    _reserve_eff = user_reserve_by_card
    if any(v is not None for v in booked):
        # the reserve is not charged on a card whose measured rest is booked
        # (the rest is what consumed it); the floor lines still print it.
        # #236 review MUSS-1: THE function the rank env uses too.
        _reserve_eff, _ = effective_reserve_by_card(cards, user_reserve_by_card, booked)
    floors = corridor_budget.floors_for_cards(cards, label, _reserve_eff)
    _terms_seen: List[Dict[str, object]] = []  # VRAM-VERTRAG M1: the pass's view
    for i, c in enumerate(cards):
        over = int(overshoot_mib[i]) if overshoot_mib is not None else 0
        cf = floors[c.uuid]
        if booked[i] is not None:
            rest_b = int(booked[i])
            grow = int(dormant_growth_mib[i]) if dormant_growth_mib is not None else 0
            carve = int(getattr(c, "reserved_mib", 0) or 0)
            l15_i = 0 if not l15_given else int(l15_vec[i])
            b = ((c.total_mib - carve - dc_mib[c.uuid] - grow - rest_b - l15_i) // 8) * 8
            out.append(b)
            _terms_seen.append(dict(
                total=int(c.total_mib), carve=carve, floor=0,
                dormant=int(dc_mib[c.uuid]) + grow, growth=grow, awake=rest_b,
                awake_source=f"{_budget_rest.SOURCE_RECORD} {booked_rest_provenance}".strip()))
            if l15_given:
                _terms_seen[-1]["l15"] = l15_i
            if terms_out is not None:
                terms_out.append({k: v for k, v in _terms_seen[-1].items() if k != "growth"})
            _res = int((user_reserve_by_card or {}).get(c.uuid, 0))
            log(
                f"budget {label} group={group} ordinal={i} "
                f"nvml_idx={c.nvml_index} {c.name}: "
                f"{b} MiB = total {c.total_mib} - driver_carve {carve} (NVML reserved) "
                f"- dormant_other {dc_mib[c.uuid]}"
                + (f" - served_dormant_growth {grow} ({dormant_growth_provenance})" if grow else "")
                + (f" - l15 {l15_i} (L1.5 post)" if l15_given else "")
                + f" - awake_rest_booked {rest_b} ({_budget_rest.SOURCE_RECORD} "
                f"{booked_rest_provenance}; replaces corridor floor transient "
                f"{int(cf.mib)} (source={cf.source}), user reserve {_res}, awake_overshoot "
                f"{D_AWAKE_OVERSHOOT_MIB}"
                + (f", measured_awake_overshoot {over}" if over else "")
                + ") MiB"
            )
            log(cf.line)
            continue
        if booked_rest_mib is not None:
            log(f"{_budget_rest.MARKER} {_budget_rest.SOURCE_UNMEASURED} group={group} "
                f"pass={label} ordinal={i} nvml{c.nvml_index}: no measured awake rest in "
                f"{booked_rest_provenance or 'the record'} -- this card keeps floor + reserve "
                f"{int((user_reserve_by_card or {}).get(c.uuid, 0))} + awake_overshoot "
                f"{D_AWAKE_OVERSHOOT_MIB} + overshoot {over} (named, not measured)")
        # rc12c (NF Dauerlauf, D TP0 died at 29.2 GiB under budgets 29624 /
        # 29144 / 28288): where the awake rest is MEASURED against the booked
        # form (D_AWAKE_REST_MIB, pdflip/d_awake_rest.py) it is the whole
        # awake excess -- charged instead of the builtin 404 and instead of a
        # budget-relative overshoot, which would count it twice.
        rest = None
        if awake_rest_mib is not None and i < len(awake_rest_mib):
            rest = awake_rest_mib[i]
        if rest is not None:
            over = 0
        awake_builtin = D_AWAKE_OVERSHOOT_MIB if rest is None else 0
        corridor = int(cf.mib) + awake_builtin
        # rc12 OOM (D TP0, 22:59:07Z): ``dc_mib`` is the other group's residue
        # at its FIRST sleep; a group that has served sleeps heavier
        # (P PP0 1320 -> 1802 MiB). The growth is a measured record
        # (pdflip/dormant_residue.py), charged here as its own term.
        grow = int(dormant_growth_mib[i]) if dormant_growth_mib is not None else 0
        # rc12b OOM: total_mib is the NVML board; the driver's carve
        # (Card.reserved_mib) is never allocatable. The corridor pass below
        # subtracts it from predicted_free on its own, so charging it here
        # does not count it twice.
        # 27B b1 death: a profile may scope the carve to its big cards
        # (driver_carve_min_total_mib; 0 = every card, the NF row).
        carve = (int(getattr(c, "reserved_mib", 0) or 0)
                 if charge_driver_carve and driver_carve_charged(c, driver_carve_min_total_mib) else 0)
        awake = int(rest) if rest is not None else 0
        l15_i = 0 if not l15_given else int(l15_vec[i])
        b = c.total_mib - corridor - dc_mib[c.uuid] - grow - over - carve - awake - l15_i
        b = (b // 8) * 8
        out.append(b)
        _terms_seen.append(dict(
            total=int(c.total_mib), carve=carve, floor=int(cf.mib),
            dormant=int(dc_mib[c.uuid]) + grow, growth=grow,
            awake=(awake if rest is not None else awake_builtin + over),
            awake_source=(
                f"D_AWAKE_REST_MIB {awake_rest_provenance}".strip() if rest is not None
                else f"awake_overshoot {awake_builtin}"
                + (f" + measured_awake_overshoot {over}" if over else "")),
        ))
        if l15_given:
            _terms_seen[-1]["l15"] = l15_i
        if terms_out is not None:
            terms_out.append({k: v for k, v in _terms_seen[-1].items() if k != "growth"})
        log(
            f"budget {label} group={group} ordinal={i} "
            f"nvml_idx={c.nvml_index} {c.name}: "
            f"{b} MiB = total {c.total_mib} - corridor {corridor} "
            f"(floor {cf.mib} source={cf.source} reserve={cf.reserve_mib}"
            + (f" + awake_overshoot {D_AWAKE_OVERSHOOT_MIB}" if rest is None else "")
            + f") - dormant_other {dc_mib[c.uuid]}"
            + (f" - served_dormant_growth {grow} ({dormant_growth_provenance})" if grow else "")
            + (f" - driver_carve {carve} (NVML reserved)" if carve else "")
            + (f" - measured_awake_overshoot {over} ({overshoot_provenance})" if over else "")
            + (f" - awake_rest {awake} (D_AWAKE_REST_MIB, measured against the booked "
               f"form, not the budget; {awake_rest_provenance})" if rest is not None else "")
            + (f" - l15 {l15_i} (L1.5 post)" if l15_given else "")
            + " MiB"
        )
        log(cf.line)
    if not corridor_constrain:
        vram_view().note_budget(label, cards, out, out, _terms_seen)
        return out
    # #1257c THE INSTALL GATE. The corridor pass may lower a budget only where
    # SOMEBODY PRICED THE FLOOR -- a measured transient, an explicit reserve,
    # or a hand-set law. Where no card's floor actuates the pass is
    # VERDICT-ONLY: it says so by name and returns the budgets byte-identical.
    # (User decision 2026-09-09, consequences 3 and 4.)
    if not any(f.actuates for f in floors.values()):
        log(
            f"PDFLIP-BUDGET corridor-constrained group={group} pass={label} "
            f"{corridor_budget.UNMEASURED_FLOOR_NAME}: no card's corridor floor "
            f"is priced (every source is UNMEASURED-FALLBACK and every user "
            f"reserve is 0), so the corridor pass is VERDICT-ONLY on this boot "
            f"and every budget stands byte-identical "
            f"({','.join(str(b) for b in out)} MiB). Measure the awake group's "
            f"transient (scripts/vram_ledger/probe_activation.py ingest) or "
            f"pass --rank-user-reserve-mib to make it actuate."
        )
        for cf in floors.values():
            log(cf.line)
        vram_view().note_budget(label, cards, out, out, _terms_seen)
        return out
    # #1257 -- THE CORRIDOR LAW AS A HARD CONSTRAINT, applied here and nowhere
    # else so this function stays the one producer of a budget number.  The
    # subtraction above is a per-card EXPECTATION (CORRIDOR_MIB = 1024 + 404, a
    # constant plus one cross-boot overshoot); the pass below is the same law
    # checked against what the front's corridor sampler actually MEASURED under
    # D-awake load, and it may only lower a budget or leave it untouched.  It
    # refuses -- loudly, keeping today's budget byte-identical -- both when the
    # inputs to price it are missing and when the only card that needs the
    # margin is the one the world pool is bound by.
    sample, why = corridor_budget.load_sample(corridor_sample_path)
    if dormant_growth_mib is not None:
        dc_mib = {c.uuid: dc_mib[c.uuid] + int(dormant_growth_mib[i]) for i, c in enumerate(cards)}
    solve = corridor_budget.solve_corridor_budgets(
        cards, out, dc_mib, sample, why, floors=floors, group=f"{group} pass={label}"
    )
    for line in solve.lines:
        log(line)
    vram_view().note_budget(label, cards, out, list(solve.budgets), _terms_seen)
    return list(solve.budgets)


#: VRAM-VERTRAG M1 (29.09.): the budget passes' view (pdflip/vram_plan_view.py),
#: one per launcher run; ``main`` resets it with the profile's pins.
_VRAM_VIEW = None
#: the known-token sets the VRAM-PLAN-PFLICHT line evaluates (one short
#: request, bs2 x 128k, the kvs2 load bs2 x 240k that filled the P arena)
VRAM_DEMAND_PROBES = (("1x32k", (32768,)), ("2x128k", (131072, 131072)),
                      ("2x240k", (245760, 245760)))


def vram_view():
    global _VRAM_VIEW
    if _VRAM_VIEW is None:
        from flliper.srt.pdflip import vram_plan_view as _vpv

        _VRAM_VIEW = _vpv.PlanView()
    return _VRAM_VIEW


def vram_plan_emit(ns, cards: Sequence[Card], pass_name: str, log: Log, *,
                   d_label: Optional[str] = None, group: str = "D") -> Optional[dict]:
    """M1: lay the pass's solver outputs into ``pdflip.vram_plan/1``, write it to
    the boot's state directory (vram_plan-<pass>.json + vram_plan.json) and log
    the one human line plus the terms that moved since the previous pass. A
    VIEW: it changes no budget, argv or bolt, and a view that fails is named,
    never raised into the boot."""
    from flliper.srt.pdflip import vram_plan as _vp

    view = vram_view()
    try:
        view.note_inputs(ns)
        plan = view.build(pass_name, cards, boot_id=str(getattr(ns, "tag", "") or ""),
                          rev=str(os.environ.get("HTSGLANG_REV", "") or ""), d_label=d_label)
    except Exception as exc:  # noqa: BLE001 -- the view never kills a launch
        log(f"VRAM-PLAN pass={pass_name} failed: {type(exc).__name__}: {exc}")
        return None
    log(_vp.plan_line(plan))
    if plan.get("demand"):
        log(_vp.demand_line(plan, VRAM_DEMAND_PROBES))
    prev = view.last
    if prev is not None:
        moved = _vp.diff_plans(prev, plan)
        if moved:
            log("VRAM-PLAN-DIFF %s -> %s: %d Terme%s %s" % (
                prev["pass"], pass_name, len(moved), ":" if moved else "",
                "; ".join("%s %s->%s" % m for m in moved[:16])
                + (" ..." if len(moved) > 16 else "")))
    view.last = plan
    d = boot_state_dir()
    if d:
        try:
            _vp.write_plan(d, plan, state_file_mod.write_json_atomic)
        except OSError as exc:
            log(f"VRAM-PLAN pass={pass_name} write failed ({type(exc).__name__}): {exc}")
        boot_state_write(log, fields={f"groups.{group}.launch.vram_plan_id": plan["plan_id"],
                                      f"groups.{group}.launch.vram_plan_pass": pass_name})
    return plan
















def p_activation_reserve_provenance(model: str, p_bs: int = DEFAULT_P_BS) -> Tuple[float, str]:
    """What group P's boot would charge for the prefill activation reserve if
    NOTHING on this rig were calibrated -- and the line that tells you which
    branch it actually took.

    #1286 F2. ``--pp-cut-activation-reserve-mib`` defaults to the 1024 MiB both
    reference boots emitted, and that number is NOT a runtime constant: the
    boot resolves it per card through ``ServerArgs.activation_reserve_mb``
    (server_args.py:14176), which asks ``mem_ledger.resolve_phase_footprint``
    for a MEASURED footprint keyed by (card UUID, hardware fingerprint,
    activation profile) and falls back to the inherited
    ``mamba_pre_capture_reserve_mb`` heuristic (:14150) when nothing matches.
    On this rig the calibrated branch was taken -- the sb5f P log carries no
    ``Using the INHERITED activation heuristic`` line -- but nothing in the
    pool model depends on that having been true, and the fallback is several
    times larger. Unpriced, it is the #1286 defect recurring by a different
    road, so the launcher prices the RISK explicitly even though it does not
    charge it.

    WHY THE LAUNCHER DOES NOT SIMPLY RESOLVE THE FOOTPRINT ITSELF (a named
    refusal, not an oversight). ``resolve_phase_footprint`` needs the
    ``ActivationProfile`` the BOOT will build, and building it here means
    restating ``profile_from_server_args`` -- seven fields, including
    ``cuda_graph_config.decode.max_bs`` -- against a ServerArgs this process
    never constructs. A restatement that is wrong by one field does not fail
    loudly: it returns ``None``, which reads as "uncalibrated", which would
    refuse a launch that is perfectly fine. A second set of books whose error
    mode is a false refusal is worse than the risk it removes, and the risk it
    removes is DETECTABLE in one grep of the boot's own log. So: measure the
    exposure, print it, name the grep.

    Returns ``(heuristic_mib, one-line provenance)``. The heuristic comes from
    the runtime's own method against a view carrying exactly the attributes it
    reads -- never a typed number.
    """
    from flliper.srt.server_args import ServerArgs

    # H68c: no cut on the probe argv (see ``_max_running_requests``); the view
    # below reads none of the ratio flags
    flags = argv_p("py", model, [1, 1, 1], 1, 1, RING_FORM_SENTINEL_STORE_CFG, [], p_bs=int(p_bs),
                   stage_ratio="", attn_stage_ratio="")

    def _flag(name: str, default):
        try:
            return type(default)(flags[flags.index(name) + 1])
        except (ValueError, IndexError):
            return default

    class _ReserveView:
        """Exactly the ServerArgs surface ``mamba_pre_capture_reserve_mb``
        reads, filled from the argv this launcher builds for group P."""

        disaggregation_mode = "null"
        max_running_requests = _flag("--max-running-requests", DEFAULT_P_BS)
        chunked_prefill_size = _flag("--chunked-prefill-size", 0)
        max_prefill_tokens = _flag("--max-prefill-tokens", 0)
        speculative_num_draft_tokens = _flag("--speculative-num-draft-tokens", 0)
        tp_size = _flag("--tp-size", 1)
        pp_size = _flag("--pp-size", 1)

    heuristic = float(
        ServerArgs.mamba_pre_capture_reserve_mb(_ReserveView(), None)
    )
    return heuristic, (
        "resolved per card by ServerArgs.activation_reserve_mb "
        "(server_args.py:14176) from a MEASURED phase footprint; its fallback "
        "when no footprint matches this rig's (hardware fingerprint, "
        "activation profile) is the inherited heuristic "
        "mamba_pre_capture_reserve_mb = %.0f MiB/rank"
    ) % heuristic


def d_mamba_ping_pong_cost(model: str, disable_overlap: bool, d_bs: int = DEFAULT_D_BS) -> Tuple[str, int, int, int]:
    """What D's overlap choice costs in DEVICE mamba state slots (FIX 1r/1).

    Returns ``(strategy, ping_pong_slots_per_running_request, extra_slots_per
    _rank, max_running_requests)``.

    THE TERM. ``arg_groups/overrides._mamba_radix_cache_resolution`` reads
    ``wants_overlap = not view.disable_overlap_schedule``, so turning the
    overlap schedule on for group D ALSO turns its mamba radix-cache strategy
    from ``no_buffer`` to ``extra_buffer`` -- and
    ``mem_cache/mamba_pool_floor.mamba_ping_pong_slots`` then charges 2 slots
    per running request where it charged 0. That is device residency out of
    D's FIXED ``--rank-gpu-memory-mib`` budgets on a rig whose corridor law is
    819-1229 MiB NVML-free per card, and the commit that switched the overlap
    schedule on priced only the overlap benefit. An unpriced term does not
    read as unknown, it reads as free (#1009).

    NOT A SECOND ACCOUNTING. The slot count is taken from the runtime's own
    ``mamba_ping_pong_slots`` against a view carrying exactly the three fields
    it reads, with ``ServerArgs.enable_mamba_extra_buffer`` itself as the
    predicate -- so a change to either function moves this number too. Only
    the PING-PONG term is charged, because it is the only term of
    ``mamba_slots_per_running_req`` that depends on the overlap choice; the
    active slot and the donation/pin term are identical in both arms and
    cancel in the delta. The remaining terms are deliberately NOT reproduced
    here: ``mamba_slot_reorder_active`` reads an environment variable, and the
    launcher's environment is not the group's (``build_env``), so evaluating
    it here would answer about the wrong process.
    """
    from flliper.srt.mem_cache.mamba_pool_floor import mamba_ping_pong_slots
    from flliper.srt.server_args import ServerArgs

    # Group D's own argv, for the same reason and with the same defect history
    # as _max_running_requests above: this list is read for the presence of a
    # flag, so it must be the list the group actually gets.
    flags = argv_d("py", model, [1, 1, 1], 1, 1, RING_FORM_SENTINEL_STORE_CFG, [], d_bs=int(d_bs))

    class _StrategyView:
        """Exactly the ServerArgs surface ``mamba_ping_pong_slots`` reads."""

        # The upstream predicate itself, not a restatement of it.
        enable_mamba_extra_buffer = ServerArgs.enable_mamba_extra_buffer

        def __init__(self, strategy: str, disable_overlap_schedule: bool) -> None:
            self.mamba_radix_cache_strategy = strategy
            self.disable_overlap_schedule = disable_overlap_schedule
            # Read off the argv this launcher builds, never asserted: the day
            # --disable-radix-cache appears in common_flags the price changes
            # to 0 and this follows it.
            self.disable_radix_cache = "--disable-radix-cache" in flags

    strategy = "no_buffer" if disable_overlap else "extra_buffer"
    per_req = mamba_ping_pong_slots(_StrategyView(strategy, disable_overlap))
    # The arm this replaces: group D as it booted before the overlap schedule
    # was turned on, i.e. the arm every DC_MEASURED_D_* number was taken on.
    baseline = mamba_ping_pong_slots(_StrategyView("no_buffer", True))
    mrr = _max_running_requests(model, "D", d_bs)
    return strategy, per_req, (per_req - baseline) * mrr, mrr


def d_overlap_cost_line(model: str, disable_overlap: bool, d_bs: int = DEFAULT_D_BS) -> str:
    """The one line that must appear wherever D's overlap choice is announced.

    The launcher must not be able to change D's device residency without the
    number appearing (FIX 1r/1), so the price is built from
    :func:`d_mamba_ping_pong_cost` rather than typed, and the MiB conversion
    it does NOT make is named rather than left as a silent omission.
    """
    strategy, per_req, extra, mrr = d_mamba_ping_pong_cost(model, disable_overlap, d_bs)
    baseline_per_req = per_req - (extra // max(1, mrr))
    return (
        f"DEVICE PRICE OF THAT CHOICE: --mamba-radix-cache-strategy "
        f"{strategy} is now STATED on D's argv instead of falling out of the "
        f"absent disable flag (arg_groups/overrides._mamba_radix_cache_"
        f"resolution reads `wants_overlap = not view.disable_overlap_"
        f"schedule`). Derived from the runtime's own mamba_pool_floor."
        f"mamba_ping_pong_slots against the no_buffer arm this replaces: "
        f"{per_req} - {baseline_per_req} = {per_req - baseline_per_req} extra "
        f"ping-pong state slots per running request x --max-running-requests "
        f"{mrr} = {extra} extra device mamba state slots on EVERY D rank, out "
        f"of the FIXED --rank-gpu-memory-mib budgets and inside the "
        f"819-1229 MiB corridor. This term is not converted to MiB here and "
        f"the refusal is named: per-rank slot bytes follow Mamba2StateShape "
        f"under --rank-tp-ratio auto, and re-deriving that shape in the "
        f"launcher would be a second accounting of the runtime's own sizing; "
        f"the boot prints it as 'mamba_cache_per_req=<x> MB' "
        f"(model_runner_kv_cache_mixin.py:2529) and the front's W19 grades "
        f"the live residue. DC_EXPECT_*/DC_MEASURED_D_* were measured on "
        f"no_buffer boots and do NOT contain this term."
    )




































#: Prefix of the one line that says where group D's capacity model got its
#: numbers.  It rides in the refusal list because that is what
#: ``d_operating_point_line`` prints, but it is PROVENANCE, never a refusal:
#: ``d_tp_ratio_decision`` filters it out of the fatal set explicitly, so a
#: note whose text happens to contain an objective name can never kill a
#: launch the way a real W61/W63 does.
ANCHOR_NOTE_PREFIX = "W65 PdFlipMeasuredAnchor: "

#: S68 review (19.10.): may the W65 scan WRITE its log index?  ``main`` sets
#: ``rw`` for a real launch and ``ro`` for ``--dry-run`` (the oracle runs the
#: real ``main`` in-process with ``--dry-run``); anything that reaches the scan
#: without ``main`` (unit tests, sizing calls) stays ``ro``, so no test or dry
#: run writes into the shared evidence dir.  ``FLLIPER_MEASURED_ANCHOR_INDEX``
#: (rw|ro|off) overrides both -- writing from a dry run is allowed only that way.
_ANCHOR_INDEX_MODE = "ro"


def _d_measured_anchor(
    cards: Sequence[Card],
    budgets: Sequence[int],
    model: str,
    maxkv_weights: Sequence[int],
    user_reserve_by_card: Optional[Dict[str, int]],
    injected: Optional[object],
    evidence_dirs: Sequence[str],
) -> Tuple[Optional[object], str]:
    """Group D's measured anchor, or a NAMED reason there is none.

    Returns ``(anchor_or_None, note)``; the note always goes to the log, so a
    boot that plans unanchored says so in one line instead of looking
    identical to a boot that planned on a measurement.

    Two inputs are the caller's own configuration and are never invented
    here.  ``ranks_on_gpu`` is counted off the rank->card map this launcher is
    building (two ranks on one uuid = 2).  ``required_free_bytes`` is the
    ``--user-reserve-mib`` this boot ships; without it the anchor is REFUSED
    rather than anchored against an assumed zero reserve, because a zero that
    was never configured reads exactly like one that was.
    """
    if injected is not None:
        return injected, ANCHOR_NOTE_PREFIX + getattr(
            injected, "provenance", "injected anchor"
        )
    if user_reserve_by_card is None:
        return None, (
            ANCHOR_NOTE_PREFIX
            + "no anchor: --user-reserve-mib was not handed to the operating"
            " point pass, so the reserve the runtime subtracts is unknown."
            " The heuristic path stands; it is not anchored against an"
            " assumed zero."
        )
    # Imported in its own step: naming MeasuredAnchorRefused in an `except`
    # whose `try` also performs the import turns an ImportError into a
    # NameError and loses the real reason.
    try:
        from flliper.srt.planner.measured_anchor import (
            MeasuredAnchorRefused,
            read_measured_anchor,
            resolve_index_mode,
        )
    except Exception as exc:  # noqa: BLE001 - an anchor never kills a launch
        return None, ANCHOR_NOTE_PREFIX + (
            "anchor module unavailable (%s: %s); heuristic path stands"
            % (type(exc).__name__, exc)
        )
    try:
        uuids = [getattr(c, "uuid", None) for c in cards]
        on_gpu = [uuids.count(u) if u is not None else 1 for u in uuids]
        reserve_b = [
            int(user_reserve_by_card.get(u, 0)) << 20 for u in uuids
        ]
        anchor = read_measured_anchor(
            # Group D is pp=1, tp=len(budgets), so the census tag's TP
            # ordinal is this group's rank. Derived from the geometry this
            # function is called with, not wired in: a group whose ranks are
            # pipeline stages would name "pp" here.
            group="D",
            tp_size=len(budgets),
            rank_axis="tp",
            ranks_on_gpu=on_gpu,
            required_free_bytes=reserve_b,
            mlp_vector=list(maxkv_weights),
            budgets_mib=list(budgets),
            model_path=model,
            evidence_dirs=tuple(evidence_dirs),
            index_mode=resolve_index_mode(_ANCHOR_INDEX_MODE),
        )
    except MeasuredAnchorRefused as exc:
        # The EXPECTED absence: no log of this form, or one that does not
        # state every post. It names itself, so it is passed through verbatim.
        return None, ANCHOR_NOTE_PREFIX + (
            "no anchor, heuristic path stands: %s" % exc
        )
    except Exception as exc:  # noqa: BLE001 - an anchor never kills a launch
        # The UNEXPECTED one (unreadable tree, import failure). Kept separate
        # so a bug here cannot read as "nothing was measured yet".
        return None, ANCHOR_NOTE_PREFIX + (
            "anchor lookup FAILED (%s: %s); heuristic path stands"
            % (type(exc).__name__, exc)
        )
    # S68: the scan note (index hits / logs read / wall) rides behind the
    # provenance, which itself is unchanged
    return anchor, ANCHOR_NOTE_PREFIX + anchor.provenance + getattr(anchor, "scan_note", "")


def d_operating_point_rows(
    cards: Sequence[Card],
    budgets: Sequence[int],
    model: str,
    d_bs: int,
    facts: Sequence[EarlyReadFact] = EARLY_READ_FACTS,
    moe_resident_fraction: Optional[Sequence[float]] = None,
    moe_scratch_slots: Optional[Sequence[int]] = None,
    user_reserve_by_card: Optional[Dict[str, int]] = None,
    measured_anchor: Optional[object] = None,
    anchor_evidence_dirs: Sequence[str] = (EVIDENCE_DIR,),
    provenance_out: Optional[List[str]] = None,
    overhead_mib_by_rank: Optional[Sequence[float]] = None,
) -> Tuple[List[DOperatingPointRow], List[str]]:
    """Price the three D weight vectors side by side. Returns (rows, refusals).

    Row 0 is always the SHIPPED maxkv vector (the gcd-reduced budget vector
    ``ServerArgs._resolve_auto_rank_tp_ratio`` derives), so the two operating
    points are always read against what this boot actually ships rather than
    against each other.

    REFUSALS ARE RETURNED, NOT RAISED. This function runs on every boot to
    build the provenance line, including the boots that ship ``maxkv`` and
    must not be killed by a card-rate library that cannot price an arm nobody
    selected. ``d_tp_ratio_decision`` raises the matching refusal only when
    the refused position is the one being SHIPPED.
    """
    refusals: List[str] = []
    budgets = [int(b) for b in budgets]
    maxkv_weights = _gcd_reduce(budgets)

    # -- the geometry and the cost model, both the runtime's own -------------
    # UNIFY S7 (27B 159333c14d (d)): does this D install the capacity-matched
    # token vector on EVERY position? Only a TP-symmetric D (profile d_layout
    # paged_dcp: the 27B, --rank-tp-ratio only); Form A / no form: no.
    _tv_every_position = _d_layout_is_tp_symmetric()
    pcm = None
    plan = None
    try:
        from flliper.srt.planner.measured_anchor import (
            anchor_has_uniform_kv_cell,
        )
        from flliper.srt.uneven_perf import PerfCostModel

        # BEFORE the checkpoint is read: where the capacity numbers come from
        # is a fact about this boot's evidence, not about whether the model
        # config happens to parse, so the provenance line must survive a W61.
        anchor, anchor_note = _d_measured_anchor(
            cards,
            budgets,
            model,
            maxkv_weights,
            user_reserve_by_card,
            measured_anchor,
            anchor_evidence_dirs,
        )
        # The note is PROVENANCE, not a refusal, and it deliberately does not
        # ride in ``refusals``: that list is a contract -- its exact contents
        # decide which positions are fatal (``d_tp_ratio_decision``) and six
        # existing cases assert on it. A caller that wants the line passes a
        # sink; one that does not is byte-identical to before.
        if provenance_out is not None:
            provenance_out.append(anchor_note)
        plan = d_plan_inputs(model, len(budgets), d_bs, overhead_mib_by_rank)
        pcm = PerfCostModel(
            plan,
            list(maxkv_weights),
            list(budgets),
            moe_resident_fraction=moe_resident_fraction,
            moe_scratch_slots=moe_scratch_slots,
            measured=(list(anchor.components) if anchor is not None else None),
            measured_mlp_vector=(
                list(anchor.mlp_vector) if anchor is not None else None
            ),
        )
    except Exception as exc:  # pragma: no cover - geometry is diagnostic
        refusals.append(
            "W61 PdFlipTpOperatingPointUnpriced: the cost model could not be "
            "built for %r (%s), so neither operating point can be derived. The "
            "shipped maxkv vector is unaffected -- it is the budget vector and "
            "needs no model." % (model, exc)
        )
        return ([], refusals)

    # -- the two measured rate vectors, by card NAME -------------------------
    gemm: List[float] = []
    membw: List[float] = []
    try:
        from flliper.srt.planner.card_rate_pass import load_measured_library

        library = load_measured_library()
        if library is not None:
            for c in cards:
                variant = next(
                    (
                        v
                        for v in (library.variants(c.name) or ())
                        if getattr(v, "gemm_tflops", None)
                        and getattr(v, "membw_gbs", None)
                    ),
                    None,
                )
                if variant is None:
                    gemm, membw = [], []
                    break
                gemm.append(float(variant.gemm_tflops))
                membw.append(float(variant.membw_gbs))
    except Exception:
        gemm, membw = [], []

    if len(gemm) != len(budgets) or len(membw) != len(budgets):
        refusals.append(
            "W61 PdFlipTpOperatingPointUnpriced: this rig's card-rate library "
            "carries no measured (gemm_tflops, membw_gbs) pair for every card "
            "of group D (%s). An operating-point vector IS the measured rate "
            "ratio, so without the measurement there is no vector to ship -- "
            "it is refused, never approximated from a nameplate peak. Write "
            "one with `python -m flliper.srt.planner.card_rate_pass --run`."
            % ", ".join(c.name for c in cards)
        )
        return ([], refusals)

    # -- W63: an explicit vector needs the uneven-TP early-read env fact ------
    # `_handle_uneven_tp` (server_args.py:12025) reads FLLIPER_UNEVEN_DCP* off
    # os.environ BEFORE the flag half publishes itself (:7186). An explicit
    # ratio shipped on a boot whose EARLY_READ_FACTS no longer carries that row
    # for group D would be silently evened out -- a wrong vector that boots.
    have = {
        f.env_key for f in facts if f.groups in ("both", "D") and f.env_key
    }
    missing = [
        k
        for k in ("FLLIPER_UNEVEN_DCP", "FLLIPER_UNEVEN_DCP_WEIGHTED")
        if k not in have
    ]
    if missing:
        refusals.append(
            "W63 PdFlipTpOperatingPointNeedsEnvPin: an explicit --rank-tp-ratio "
            "vector is resolved by _handle_uneven_tp (server_args.py:12025), "
            "which reads %s off os.environ BEFORE the flag half publishes "
            "itself (:7186). EARLY_READ_FACTS no longer states %s for group D, "
            "so shipping the vector would need an env pin in the launcher's own "
            "shell -- the R19 shape #1235 removed. Refused instead."
            % (" and ".join(missing), " and ".join(missing))
        )
        return ([], refusals)

    # -- the three rows ------------------------------------------------------
    from flliper.srt.distributed.utils import _CP_TOKEN_UNITS, partition_units

    n_ranks = len(budgets)

    # #1293 -- the plan-time flags the attention-axis predicate reads. This
    # launcher's group D always ships ``--uneven-dcp`` + ``--uneven-dcp-
    # weighted`` (``early_read_flags("D")``), and the W63 gate immediately
    # above has just PROVEN both facts present -- under a non-uniform
    # ``--rank-tp-ratio`` the runtime then auto-sets ``dcp_size = tp_size``
    # (server_args.py:12112-12128, "Uneven DCP: auto-set dcp_size"). Mirroring
    # that auto-set here is gated on the SAME fact the W63 check read, so a
    # future form that stops shipping the env fact falls back to the head
    # axis instead of silently keeping the token one. The non-uniformity half
    # of the predicate stays inside ``plan_uneven_dcp_kv_replicated`` itself.
    uneven_dcp_armed = "FLLIPER_UNEVEN_DCP" in have
    dcp_flags = replace(
        plan, dcp_size=(n_ranks if uneven_dcp_armed else plan.dcp_size)
    )

    def _fam_partition(units: int, weights: Sequence[int]) -> Tuple[int, ...]:
        """Partition ``units`` over ``weights``, or ``()`` if the runtime
        would not shard that family by a vector at all.

        THE PREDICATE IS THE RUNTIME'S, NOT A WEAKER ONE OF OUR OWN (review
        R2). ``_partition_units_raw`` RAISES when ``units < len(weights)``
        (distributed/utils.py), and ``PerfCostModel`` answers exactly this
        question with ``gdn_units >= tp_size`` (uneven_perf.py:4553, :4596)
        before it shards. The first version of this row guarded GDN with
        ``> 1``, which is true and still below the world size on any model
        with 2 GDN units and 3 ranks -- and the call sat OUTSIDE every
        try/except in a function reached from the DEFAULT maxkv boot. A
        diagnostic line that runs on every boot must not be able to kill one.
        """
        if int(units) < int(n_ranks):
            return tuple()
        return tuple(partition_units(int(units), list(weights)))

    def _row(
        position: str, score_name: str, scores, weights
    ) -> Tuple[DOperatingPointRow, Optional[str]]:
        """One priced row, and the refusal it earned (or ``None``).

        NOTHING IN HERE RAISES. Every arm is priced on every boot so the
        trade is readable per boot, which means this runs on maxkv boots that
        selected none of the arms it prices.
        """
        try:
            # Attention. The runtime widens the grid to the q-heads when the
            # o-group count is below the world size rather than crashing in
            # partition_units (uneven_perf.py:4520-4528); mirrored here so the
            # row quotes the geometry the model would actually shard on.
            a_units = int(pcm.attn_units)
            grid = a_units if a_units >= n_ranks else max(int(pcm.q_heads), n_ranks)
            attn_units = d_row_attn_units(pcm, weights)
            scale = int(pcm.q_heads) // max(1, grid)
            attn = tuple(u * scale for u in attn_units)
            # #1293 THE AXIS THE ATTENTION COMPUTE RIDES. Under replicated-KV
            # uneven DCP (the serving form: dcp_size=3, replicated kv heads,
            # token-sharded, weighted owner rule #173) the attention COMPUTE
            # follows the TOKEN vector, not the kv-head unit grid -- #492's
            # correction verbatim: "attention grid-PINNED was FALSE -- kv
            # heads are replicated, attention compute follows the token
            # vector (continuous)". The head partition above stays what the
            # PROJECTIONS really do (that split is real and prices the weight
            # terms below); the token partition is what the family's per-rank
            # work follows, on the SAME 64-unit largest-remainder grid the
            # runtime's own "Uneven DCP ... installed" vector uses
            # (_CP_TOKEN_UNITS, distributed/utils.py). This is #503's escape
            # extended to this solve: #503 fixed the phase-prefill enumerator
            # gridding attention on kv-heads (placement.py:813 models it
            # right); the same class sat here.
            axis = _attn_axis_for(weights, dcp_flags)
            # 27B line (24.09.): NO position pins the token vector. The
            # launcher ships only --rank-tp-ratio for every position (never
            # --rank-kv-ratio), so the runtime installs the CAPACITY-matched
            # vector after profiling on every one of them (xsn420 D.log:
            # 'installed measured KV-token ownership vector [32, 15, 17]').
            # The rate-proportional vector this used to pin priced a funded
            # context no boot of these positions can have (decode-bs6:
            # 116416 vs ~625k). The derived vector is read off
            # predict_capacity below, for every position as for maxkv.
            # UNIFY S7: that is the TP-symmetric D's rule (_tv_every_position,
            # profile d_layout paged_dcp); any other D keeps the NF line's
            # #1293 pin -- token ownership proportional to the measured rates,
            # integerised the way the runtime integerises its token vector.
            token_units: Tuple[int, ...] = ()
            if axis == "token" and position != "maxkv" and not _tv_every_position:
                token_units = tuple(
                    partition_units(_CP_TOKEN_UNITS, list(weights))
                )
            # GDN through the cost model's OWN partitioner, so its predicate
            # (and its `[0] * tp_size` answer for a model it will not shard)
            # is inherited rather than re-implemented.
            gdn_raw = pcm.gdn_unit_partition(list(weights))
            gdn = tuple(int(x) for x in gdn_raw) if any(gdn_raw) else tuple()
            mlp_part = _fam_partition(int(pcm.mlp_units), weights)
            if not mlp_part:
                return (
                    DOperatingPointRow(
                        position=position,
                        score_name=score_name,
                        scores=tuple(float(x) for x in scores),
                        weights=tuple(int(w) for w in weights),
                        attn_heads=attn,
                        gdn_heads=gdn,
                        round_ms=None,
                        round_unit="unpriced (%d MLP units < %d ranks: the "
                        "runtime does not shard this family by a vector)"
                        % (int(pcm.mlp_units), n_ranks),
                        world_pool_tokens=None,
                        feasible=None,
                        funded_ctx_tokens=None,
                        attn_axis=axis,
                        attn_token_units=token_units,
                    ),
                    None,
                )
            mlp = list(mlp_part)
            # #1293 floor-aware re-normalisation, per family, per axis. When a
            # unit grid floors a rank, the realised shares drift off the
            # requested ratio; the drift is PRINTED with the family and the
            # axis instead of either passing silently (instrument-text-lies
            # shape) or being fatal for a grid that no longer carries the
            # compute. Families checked on the grid they actually ride: the
            # attention COMPUTE on its axis (token grid when uneven DCP, the
            # head grid otherwise), the attention PROJECTIONS always on the
            # head grid (their split is real either way), GDN and MLP on
            # their own unit grids.
            _notes = []
            if axis == "token":
                if token_units:
                    _notes.append(
                        _axis_renormalisation(
                            "attention", "token", _CP_TOKEN_UNITS, weights,
                            token_units,
                        )
                    )
                _notes.append(
                    _axis_renormalisation(
                        "attention-projections", "head", grid, weights,
                        attn_units,
                    )
                )
            else:
                _notes.append(
                    _axis_renormalisation(
                        "attention", "head", grid, weights, attn_units
                    )
                )
            if gdn:
                _notes.append(
                    _axis_renormalisation(
                        "GDN", "head", int(pcm.gdn_units), weights, list(gdn)
                    )
                )
            _notes.append(
                _axis_renormalisation(
                    "MLP", "unit", int(pcm.mlp_units), weights, mlp
                )
            )
            axis_note = "; ".join(n for n in _notes if n)
        except Exception as exc:  # pragma: no cover - geometry is diagnostic
            return (
                DOperatingPointRow(
                    position=position,
                    score_name=score_name,
                    scores=tuple(float(x) for x in scores),
                    weights=tuple(int(w) for w in weights),
                    attn_heads=tuple(),
                    gdn_heads=tuple(),
                    round_ms=None,
                    round_unit="unpriced (geometry: %s)" % exc,
                    world_pool_tokens=None,
                    feasible=None,
                    funded_ctx_tokens=None,
                ),
                None,
            )

        round_ms: Optional[float] = None
        unit = "unpriced"
        note = ""
        try:
            if position == "decode-bs1":
                # THE RUNTIME'S OWN bs=1 ROOFLINE. Relative units by its own
                # docstring ("only ratios between candidates are consumed"),
                # so it is reported as a ratio and never as milliseconds.
                round_ms = float(
                    pcm.decode_round_time(mlp, membw, None, list(attn_units))
                )
                unit = "relative bs1 round (PerfCostModel.decode_round_time)"
                note = D705_PROVENANCE
            else:
                # The lockstep sum-of-per-family-maxima (#475).
                #
                # PER-CARD SCALAR RATES, NOT PER-(RANK, FAMILY) SCORES, AND
                # THE LINE SAYS SO (refuter MF-3). ``family_tflops`` is passed
                # as None because this launcher has no per-family score
                # source: ``family_prefill_tflops`` needs a ``GemmScores``
                # from the planner's gemm pass and returns None for a
                # single-scheme checkpoint anyway. With None, one rank is
                # slowest in EVERY family, so #475's ``sum_fam max_rank``
                # degenerates to ``max_rank sum_fam`` -- the pre-#475 lower
                # bound. Building a second per-family score here to dress the
                # number up would be exactly the parallel bookkeeping
                # UPSTREAM-MINIMAL forbids, so the claim is dropped instead of
                # the input invented, and the unit carries the limitation to
                # the log line.
                round_ms = (
                    float(
                        pcm.prefill_lockstep_compute_time(
                            mlp, gemm, None, list(attn_units)
                        )
                    )
                    * 1000.0
                )
                unit = (
                    "ms lockstep GEMM over PER-CARD scalar rates "
                    "(PerfCostModel.prefill_lockstep_compute_time, "
                    "family_tflops=None, so sum_fam max_rank collapses to "
                    "max_rank sum_fam); the COMPUTE-BOUND premise of this arm "
                    "is BOOT-UNPROVEN -- it is the very quantity #1241 slice "
                    "(1) measures and no boot has run it yet"
                )
        except Exception as exc:
            round_ms = None
            unit = "unpriced (%s)" % exc

        # THE POOL, AND THE TWO FIELDS THE FIRST VERSION THREW AWAY
        # (refuter MF-4). ``predict_capacity`` returns `feasible` -- the
        # model's own verdict that the vector fits at all -- and `ctx`, the
        # FUNDED context ``min(sum_r P_r, 64 * min_r P_r)``. Reading only
        # ``sum(p)`` prints a barely-moved world pool for a vector that
        # collapses one rank, because the funded context is bounded by the
        # SMALLEST rank and the sum is not. Both are carried on the row now,
        # and an infeasible vector is refused rather than shipped.
        pool: Optional[int] = None
        feasible: Optional[bool] = None
        funded_ctx: Optional[int] = None
        try:
            # #1293: a token-axis POSITION pins its own token vector, so the
            # funded context priced is the pinned vector's own budget
            # (predict_capacity's #492 arm: cp_token_context_budget of the pin,
            # strictly the weaker of the two) -- the price of taking token
            # ownership proportional to a rate instead of to capacity reaches
            # the gate rather than being rounded away. maxkv (auto) keeps the
            # byte-identical derived-matched call, and its token vector is
            # read BACK off the model's own answer below.
            cap = (
                pcm.predict_capacity(
                    mlp, list(attn_units), token_vector=list(token_units)
                )
                if token_units
                else pcm.predict_capacity(mlp, list(attn_units))
            )
            pool = int(sum(cap["p"]))
            if "feasible" in cap:
                feasible = bool(cap["feasible"])
            if cap.get("ctx") is not None:
                funded_ctx = int(cap["ctx"])
            # #62 FOLLOW-UP: READ THE VECTOR BACK ONLY FROM AN ANCHORED
            # MODEL.
            #
            # This line takes the capacity model's own token vector and ships
            # it. That is right when the model's absolute bytes are anchored
            # to a measurement (`measured is not None`, where
            # `measured_weight_bias` removes the family model's absolute
            # error) and wrong when they are not -- and until #62 it never
            # fired on this form at all, because the unanchored model declared
            # every vector infeasible and `predict_capacity` returns
            # `token_vector=None` in that branch.
            #
            # So fixing `feasible` (which WAS wrong: the routed experts live
            # in the host store) had a second, unasked-for effect: it started
            # feeding a vector computed from weights that read ~2x the
            # measured census. Boots fnFL2v84 and v85 both died of it -- the
            # 5090's KV cell grew from 0,5 to 5,7 KB per token, the pool fell
            # from 262144 to 195008-196224 tokens, and the card ran out during
            # graph capture.
            #
            # Without an anchor the RUNTIME's own derivation stands, which is
            # what carried 262144 tokens on this form (fnFL2v72). The gate
            # still sees the offload; only the vector waits for #48.
            # #48 MEASURED 2026-09-21, and the reason this guard grew a second
            # clause instead of opening: anchoring the model (see
            # planner/measured_anchor.py) makes `measured is not None` TRUE on
            # this form, and the vector it then derives is still wrong here.
            # Against fnFL2v72's own posts the anchored model answers
            # token_vector [19,21,24] and a funded context of 1016030 tokens,
            # i.e. it spreads KV across ranks 1 and 2 -- which own NO attention
            # heads under `--rank-tp-ratio 1,0,0` and sized their pools at a
            # 768-byte placeholder cell against rank 0's 14143. Their
            # `available_bytes` is therefore not "bytes fundable for group KV",
            # and re-priced at the group cell it reads as ~340k phantom tokens
            # per rank. Shipping that is the fnFL2v84/v85 failure in a new
            # costume.
            #
            # So the readback additionally requires that every rank sized its
            # pool at the SAME cell -- the one condition under which an
            # anchored per-rank budget means the same thing on every rank. On
            # a uniform-cell form it opens by itself; on this one the runtime's
            # own derivation keeps carrying the vector, which is what ran
            # 262144 tokens on fnFL2v72.
            #
            # UNIFY S7 (27B 159333c14d (d)): on a TP-symmetric D
            # (profile d_layout paged_dcp, the 27B) the launcher ships only
            # --rank-tp-ratio, so the runtime installs the capacity-matched
            # vector on EVERY position; the row reads it back for all of them.
            if axis == "token" and cap.get("token_vector") and (
                _tv_every_position
                or (
                    position == "maxkv"
                    and getattr(pcm, "measured", None) is not None
                    and anchor_has_uniform_kv_cell(getattr(pcm, "measured", None))
                )
            ):
                token_units = tuple(int(v) for v in cap["token_vector"])
                # THE RUNTIME'S OWN INSTALLED POOL: the owner rule over the
                # vector it installs (distributed/utils.cp_token_context_budget,
                # the same arithmetic as 'max_total_num_tokens X -> ~Y'), i.e.
                # the funded context after the 64-unit quantisation.
                from flliper.srt.distributed.utils import cp_token_context_budget

                funded_ctx = int(cp_token_context_budget(
                    [max(int(v), 1) for v in token_units],
                    [max(int(x), 1) for x in cap["p"]]))
        except Exception:
            pool = None
        refusal = None
        if feasible is False:
            refusal = (
                "W64 PdFlipTpOperatingPointInfeasible: position %s derives "
                "weights %s, which PerfCostModel.predict_capacity marks "
                "feasible=False against this boot's budgets %s -- the weight "
                "shards plus the mamba pool plus the reserves do not leave a "
                "positive KV pool on at least one rank. Refused. This is the "
                "model's own verdict, not a margin chosen here: no safety "
                "factor is applied on top of it."
                % (position, list(int(w) for w in weights), list(budgets))
                + _infeasible_breakdown(pcm, mlp, attn_units, budgets, cap, cards)
            )
        return (
            DOperatingPointRow(
                position=position,
                score_name=score_name,
                scores=tuple(float(x) for x in scores),
                weights=tuple(int(w) for w in weights),
                attn_heads=attn,
                gdn_heads=gdn,
                round_ms=round_ms,
                round_unit=unit,
                world_pool_tokens=pool,
                feasible=feasible,
                funded_ctx_tokens=funded_ctx,
                note=note,
                attn_axis=axis,
                attn_token_units=token_units,
                axis_note=axis_note,
            ),
            refusal,
        )

    rows = []
    for _pos, _score_name, _scores, _weights in (
        ("maxkv", "budget MiB (capacity-first)", tuple(budgets), maxkv_weights),
        ("decode-bs1", "measured membw_gbs", membw, _weights_from_scores(membw)),
        ("decode-bs6", "measured gemm_tflops", gemm, _weights_from_scores(gemm)),
    ):
        _row_obj, _refusal = _row(_pos, _score_name, _scores, _weights)
        rows.append(_row_obj)
        # A refusal on the SHIPPED maxkv vector is not this axis's to make:
        # that vector is what `--rank-tp-ratio auto` resolves to with or
        # without #1241, so refusing it here would turn a diagnostic into a
        # boot blocker for the default path.
        if _refusal is not None and _pos != "maxkv":
            refusals.append(_refusal)

    # -- W62: a position that would turn an uneven axis OFF -------------------
    #
    # THE OBVIOUS CHECK WAS UNREACHABLE AND IS STILL NOT THE ONE MADE HERE --
    # but the reason has CHANGED, and the old wording is now false. It used to
    # read: "a rank owning zero heads cannot happen: `partition_units`
    # guarantees every rank >= 1 unit by construction". Form A (the
    # attention-host layout, --rank-role host,worker,worker) makes zero heads
    # a deliberate LAYOUT: partition_units(allow_zero=True) hands a worker
    # rank nothing at all, on purpose, because it holds only experts.
    #
    # So the zero case is now reachable and is EXCLUDED rather than asserted
    # against: a weight of 0 is not a saturated axis, it is a rank that was
    # never on the axis. Without this exclusion _saturated fires for every
    # Form A worker (0 / total * units = 0.0 < 1.0 always) and the refusal
    # below would reject every Form A vector as "axis switched off" -- a
    # guard that was correct becoming a boot blocker.
    #
    # The reachable failure is SATURATION. When a rank's proportional share of
    # a family's units falls below one unit, the partitioner floors it to 1 and
    # the shipped partition stops representing the measured ratio: the axis is
    # still nominally uneven, but it is pinned at its floor and every further
    # difference in the measurement is invisible to it. That is an axis
    # disabled in the only sense that matters to a boot -- it no longer carries
    # the quantity it exists to carry -- and it is what an extreme rate ratio
    # actually produces.
    def _saturated(weights: Sequence[int], units: int) -> Optional[int]:
        total = sum(int(w) for w in weights)
        for r, w in enumerate(weights):
            if int(w) == 0:
                continue  # Form A worker: not on this axis, not saturated
            if total > 0 and float(w) / total * float(units) < 1.0:
                return r
        return None

    for row in rows[1:]:
        sat_family = None
        sat_rank = None
        sat_units = None
        sat_axis = None
        # #1293: THE GRID THE VERDICT IS TAKEN AGAINST IS THE GRID THE COMPUTE
        # RIDES. Under replicated-KV uneven DCP the attention compute follows
        # the TOKEN vector (row.attn_axis == "token"), so saturation for that
        # family is judged on the 64-unit token grid -- the 4-kv-head-unit
        # grid stopped carrying the family's compute the moment the KV heads
        # were replicated (#492), and refusing a measured ratio against a grid
        # it does not ride was this seam's defect (boot weg2dec1 arms 2/3,
        # BOOT_weg2dec1_0909.md). The refusal itself STAYS for a vector that
        # genuinely cannot be represented on its riding axis (a rank's token
        # share below 1/64), and for every head-axis form.
        for fam_name, fam_units, fam_axis in (
            (
                "attention",
                _CP_TOKEN_UNITS
                if row.attn_axis == "token"
                else int(pcm.attn_units),
                row.attn_axis,
            ),
            ("GDN", int(pcm.gdn_units), "head"),
        ):
            # SAME PREDICATE AS THE PARTITION (review R2). A family the
            # runtime does not shard by the vector has no axis to disable, so
            # a saturation refusal against it would be a FALSE W62 -- the
            # `<= 1` form fired for a GDN family of 2 units on 3 ranks, which
            # `gdn_unit_partition` answers with [0, 0, 0].
            if fam_units < n_ranks:
                continue
            r = _saturated(row.weights, fam_units)
            if r is not None:
                sat_family, sat_rank, sat_units, sat_axis = (
                    fam_name, r, fam_units, fam_axis,
                )
                break
        if sat_family is not None:
            if sat_family == "attention":
                sat_part = list(row.attn_token_units or row.attn_heads)
            else:
                sat_part = list(row.gdn_heads)
            refusals.append(
                "W62 PdFlipTpOperatingPointDisablesUnevenAxis: position %s "
                "derives weights %s; rank %d's share of the %d %s units "
                "(axis=%s) is below ONE unit, so partition_units floors "
                "it to 1 (it guarantees >= 1 per rank) and the shipped "
                "partition %s no "
                "longer represents the measured ratio -- the axis is pinned at "
                "its floor, which is the axis disabled in the sense a boot can "
                "observe. Refused rather than shipped."
                % (
                    row.position,
                    list(row.weights),
                    int(sat_rank),
                    int(sat_units),
                    sat_family,
                    sat_axis,
                    sat_part,
                )
            )
        elif len(set(row.weights)) == 1 and len(row.weights) > 1:
            refusals.append(
                "W62 PdFlipTpOperatingPointDisablesUnevenAxis: position %s "
                "derives the FLAT vector %s, which is even TP wearing an "
                "uneven flag. Refused: an axis that resolves to equality is "
                "the axis disabled." % (row.position, list(row.weights))
            )
    return (rows, refusals)






def dual_w64_override(refusal: str, budgets: Sequence[int], model: str, *, dual_layout: bool,
                      evidence_dirs: Sequence[str] = (EVIDENCE_DIR,)) -> Tuple[bool, str]:
    """DUAL-TP3PP3: may a W64 refusal of the SHIPPED D position be lifted under
    --dual-layout? Only when the newest dual-share D log of this model with the
    same installed weight vector shows, on its OWN budget posts (the runtime's
    ``KV = budget - posts``), that every rank keeps >= the model's minimum
    tokens at these budgets (pdflip/dual_w64.py; metal a3t5js: model 17188 vs
    measured ~14853 MiB on D r0). ``(lifted, note)``; never lifts off-dual,
    never a non-W64 refusal, never without a measurement."""
    if not dual_layout or not str(refusal).startswith("W64 "):
        return False, ""
    m = re.search(r"derives weights \[([\d, ]+)\]", str(refusal))
    if not m:
        return False, "W64-DUAL: the refusal names no weight vector; the model verdict stands"
    weights = [int(x) for x in m.group(1).split(",") if x.strip()]
    from flliper.srt.uneven_perf import _PREDICT_MIN_RANK_TOKENS
    from flliper.srt.pdflip import dual_w64 as _dw

    meas = _dw.find_dual_d_measurement(list(evidence_dirs), model, weights)
    if meas is None:
        return False, ("W64-DUAL: no measured dual-share D log of %s with weights %s under %s; "
                       "the model verdict stands" % (model, weights, ", ".join(evidence_dirs)))
    v = _dw.judge(meas, list(budgets), int(_PREDICT_MIN_RANK_TOKENS))
    return v.feasible, v.line


def d_tp_ratio_decision(
    objective: str,
    tune: str,
    cards: Sequence[Card],
    budgets: Sequence[int],
    model: str,
    d_bs: int,
    env_d: str = "",
    user_reserve_by_card: Optional[Dict[str, int]] = None,
    overhead_mib_by_rank: Optional[Sequence[float]] = None,
    dual_layout: bool = False,
) -> DTpRatioDecision:
    """Choose group D's weight objective and PRICE the choice on one line.

    NO SECOND SOLVER (upstream-minimal). Both arms are the runtime's own
    mechanisms and the launcher only names which one this boot took:

      * ``maxkv``  -> ``--rank-tp-ratio auto``              (capacity-first)
      * ``speed``  -> ``--rank-tp-ratio auto-performance --rank-perf-tune T``

    What is computed here is the COST LINE, and every term in it comes from a
    source that already exists in the tree and is already used by this
    launcher:

      * the weight vector is the gcd-reduced budget vector, which is exactly
        what ``ServerArgs._resolve_auto_rank_tp_ratio`` (server_args.py:11239)
        derives from the same ``--rank-gpu-memory-mib`` list this launcher is
        about to pass -- recomputed to be PRINTED, never passed, so there is
        one writer of the vector and it is the runtime;
      * the per-rank head partitions come from ``PerfCostModel`` +
        ``partition_units``, the same pair ``_log_derived_plan_vectors``
        (server_args.py:11316) uses, so the line quotes the geometry the model
        actually shards on rather than a second reading of config.json;
      * the per-card compute score is the MEASURED card-rate library
        (``planner/card_rate_pass.load_measured_library``), which is already
        the preferred score source of the PP cut on the P side
        (``pp_cut_launch.ms_per_layer_from_card_library``). It is keyed by
        card NAME, so it survives a change of NVML/CUDA ordering, and reading
        it touches no GPU: a missing library prints UNPRICED and the arm is
        still chosen, because the objective is the operator's decision and not
        the estimate's.

    THE SLOWEST-RANK LAW is what the estimate states. Under uneven TP every
    rank runs the attention of its own q-heads and the barrier waits for the
    last one, so the cost of the capacity split is
    ``max_r(q_heads_r / score_r)`` against the compute-proportional split's
    ``max_r`` -- one number, in relative units, and labelled an ESTIMATE
    because the launcher has no per-layer measurement of group D.

    AND THE HONEST LIMIT OF THE SPEED ARM, which is the reason this line
    exists rather than a promise: ``auto-performance`` does NOT move the
    attention/GDN split. Its docstring says so (uneven_perf.py:6571, "Never
    touches the base (attention/GDN/DCP) split ... the only vector this
    function writes to server_args is rank_mlp_ratio"), so the attention
    barrier priced below is IDENTICAL under both objectives. What the speed
    arm moves is the dense-MLP family vector, and -- with
    ``--rank-perf-tune dec`` -- the KV-TOKEN split via ``--rank-kv-ratio
    speed``, which is the lever the flag's own help calls the larger one at
    depth (server_args.py:2694, measured -24.5 % of the context-dependent part
    of the decode step at 120k resident tokens, #210). A reader who takes the
    attention delta below as the speed arm's yield would be wrong, so the line
    says which lever each number belongs to.
    """
    if objective not in D_TP_OBJECTIVE_CHOICES:
        raise PdFlipLaunchRefused(
            "W47 PdFlipTpObjectiveRefused: --d-tp-objective %r is not one of %s."
            % (objective, "|".join(D_TP_OBJECTIVE_CHOICES))
        )

    # #1241 slice (2). Built on EVERY boot, including the maxkv ones, because
    # the point of the line is that the trade is visible per boot. Refusals
    # only KILL the launch when the refused position is the one being shipped.
    _moe_frac, _moe_scratch = d_moe_residency(env_d, len(budgets))
    _anchor_notes: List[str] = []
    op_rows, op_refusals = d_operating_point_rows(
        cards, budgets, model, d_bs,
        moe_resident_fraction=_moe_frac, moe_scratch_slots=_moe_scratch,
        user_reserve_by_card=user_reserve_by_card,
        provenance_out=_anchor_notes,
        overhead_mib_by_rank=overhead_mib_by_rank,
    )
    op_line = d_operating_point_line(op_rows, op_refusals, objective)
    # Where group D's capacity numbers came from, on every boot -- an
    # unanchored boot must not look identical to an anchored one.
    for _note in _anchor_notes:
        op_line = op_line + "\n  " + _note
    if objective in D_OPERATING_POINTS:
        mine = [
            r
            for r in op_refusals
            if objective in r or r.startswith(("W61", "W63"))
        ]
        if mine and dual_layout:
            _lifted, _note = dual_w64_override(mine[0], budgets, model, dual_layout=True)
            if _note:
                op_line = op_line + "\n  " + _note
            if _lifted:
                mine = []
            else:
                mine = [mine[0] + (" | " + _note if _note else "")]
        if mine:
            raise PdFlipLaunchRefused(mine[0] + " (position %s was SHIPPED, so "
                                    "the refusal is fatal here; on a maxkv boot "
                                    "the same finding is printed and the launch "
                                    "continues)." % objective)

    if objective == "speed":
        flags = ("--rank-tp-ratio", "auto-performance", "--rank-perf-tune", str(tune))
    elif objective in D_OPERATING_POINTS:
        row = next(r for r in op_rows if r.position == objective)
        # THE ONE PLACE THE LAUNCHER WRITES A VECTOR INSTEAD OF NAMING A
        # RESOLVER. Deliberate and narrow: neither `auto` nor
        # `auto-performance` moves the attention/GDN split (uneven_perf.py:6571
        # says so in its own docstring), and the attention barrier IS the
        # operating-point question. The vector is the measured rate ratio and
        # nothing else -- no hand numbers, no tuning constant.
        flags = ("--rank-tp-ratio", ",".join(str(w) for w in row.weights))
    else:
        flags = ("--rank-tp-ratio", "auto")

    budgets = [int(b) for b in budgets]
    g = math.gcd(*budgets) if len(budgets) > 1 else budgets[0]
    weights = [b // max(1, g) for b in budgets]

    # -- the geometry, from the runtime's own cost model ---------------------
    heads_note = "geometry UNPRICED"
    q_heads: List[int] = []
    gdn_heads: List[int] = []
    n_q = 0
    try:
        from flliper.srt.distributed.utils import partition_units
        from flliper.srt.uneven_perf import PerfCostModel

        pcm = PerfCostModel(
            d_plan_inputs(model, len(budgets), d_bs, overhead_mib_by_rank), weights, list(budgets)
        )
        n_q = int(pcm.q_heads)
        scale = n_q // max(1, int(pcm.attn_units))
        q_heads = [u * scale for u in partition_units(int(pcm.attn_units), weights)]
        # Same predicate as the operating-point rows (review R2): `> 1` is
        # true and still below the world size for a 2-unit GDN family on 3
        # ranks, and partition_units RAISES there. `gdn_unit_partition`
        # carries the runtime's own `gdn_units >= tp_size` and answers
        # [0]*tp_size instead of raising.
        _gdn_raw = pcm.gdn_unit_partition(list(weights))
        if any(_gdn_raw):
            gdn_heads = [int(x) for x in _gdn_raw]
        heads_note = "attention %s of %d q-heads%s" % (
            q_heads,
            n_q,
            "; GDN %s of %d linear-attention heads" % (gdn_heads, int(pcm.gdn_units))
            if gdn_heads
            else "",
        )
    except Exception as exc:  # pragma: no cover - geometry is diagnostic
        heads_note = (
            "geometry UNPRICED (%s): the per-rank head partition could not be "
            "derived, so the barrier estimate below is omitted rather than "
            "guessed" % (exc,)
        )

    # -- the measured per-card score, or an honest absence -------------------
    scores: Optional[List[float]] = None
    try:
        from flliper.srt.planner.card_rate_pass import load_measured_library

        library = load_measured_library()
        if library is not None:
            got: List[float] = []
            for c in cards:
                variant = next(
                    (
                        v
                        for v in (library.variants(c.name) or ())
                        if getattr(v, "gemm_tflops", None)
                    ),
                    None,
                )
                if variant is None:
                    got = []
                    break
                got.append(float(variant.gemm_tflops))
            scores = got or None
    except Exception:
        scores = None

    if scores and q_heads and len(scores) == len(q_heads):
        cap_bar = max(h / s for h, s in zip(q_heads, scores))
        ideal_units = partition_units(
            int(n_q), [max(1, int(round(s * 1000.0))) for s in scores]
        )
        ideal_bar = max(h / s for h, s in zip(ideal_units, scores))
        idle = [100.0 * (1.0 - (h / s) / cap_bar) for h, s in zip(q_heads, scores)]
        price = (
            "PRICE (ESTIMATE, slowest-rank law, attention barrier only): "
            "measured card GEMM %s TFLOP/s (card-rate library, by card NAME); "
            "the capacity split's barrier is rank %d at %.4f head/TFLOP-s "
            "while the others idle %s of every attention barrier. A "
            "compute-proportional split of the same %d q-heads would be %s at "
            "%.4f = %+.1f %% -- and that delta is NOT what the speed arm buys: "
            "auto-performance never moves the attention split "
            "(uneven_perf.py:6571). Its levers are the dense-MLP family vector "
            "and, under --rank-perf-tune dec, the KV-TOKEN split "
            "(--rank-kv-ratio speed, #210, measured -24.5 %% of the "
            "context-dependent part of the decode step at 120k resident "
            "tokens). Moving the attention split itself needs an explicit "
            "--rank-tp-ratio vector, which this knob deliberately does not "
            "emit."
            % (
                ", ".join(
                    "%s %.1f" % (c.name, s) for c, s in zip(cards, scores)
                ),
                max(range(len(q_heads)), key=lambda r: q_heads[r] / scores[r]),
                cap_bar,
                ", ".join("%.0f %%" % v for v in idle),
                int(n_q),
                ideal_units,
                ideal_bar,
                100.0 * (ideal_bar - cap_bar) / cap_bar,
            )
        )
    else:
        price = (
            "PRICE UNPRICED: no measured card-rate library on this rig for "
            "these card names (`python -m flliper.srt.planner.card_rate_pass "
            "--run` writes one), so the cost of the capacity split at the "
            "attention barrier is NOT estimated here. An absent estimate is "
            "reported as absent; it is never a measured zero."
        )

    line = (
        "PDFLIP D-WEIGHTS objective=%s (default %s since the user order of "
        "2026-09-09, \"der decode bs6 soll mit bs6 (nicht mehr bs4) der "
        "standard werden\": dec2c measured the bs=6 vector [42,11,11] at "
        "+10.3 %% at bs4 for -1.4 %% pool, so the maxkv law is outranked AT "
        "THIS OPERATING POINT, not repealed -- 'maxkv' and 'speed' stay "
        "selectable and never silent) -> argv %s. Weights the runtime will "
        "derive from the SAME --rank-gpu-memory-mib %s: %s (gcd-reduced, "
        "server_args.py:11239 -- printed here, written there). %s. %s"
        % (
            objective,
            D_TP_OBJECTIVE_DEFAULT,
            " ".join(flags),
            budgets,
            weights,
            heads_note,
            price,
        )
    )
    return DTpRatioDecision(
        objective=objective,
        tune=str(tune),
        flags=tuple(flags),
        line=line,
        op_line=op_line,
        rows=tuple(op_rows),
    )





#: #1317n the controller's own load-pool fraction, mirrored here because the
#: launcher must size the pool the controller will only lend 90 % of.
#: Read from the controller rather than typed, so the two cannot drift.
try:  # pragma: no cover - import shape differs in bare launcher tests
    from flliper.srt.managers.cache_controller import (
        HICACHE_LOAD_POOL_USAGE_FRACTION as HICACHE_LOAD_POOL_USAGE_FRACTION_1317N,
    )
except Exception:  # noqa: BLE001
    HICACHE_LOAD_POOL_USAGE_FRACTION_1317N = 0.9




























































PP_CROSSING_WIRE_ENV = "FLLIPER_PP_CROSSING_WIRE"










































































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


#: #1447: how much more pinned host bounce a faster cut may cost than the
#: incumbent (39,13,12), in GiB.  Default 0: the solver ships the fastest cut
#: at or above the floor whose exchange bounce is no larger than the
#: incumbent's -- boot weg2xsn204 shipped 40,12,12 under the cap-floor rule and
#: the ledger refused it (W97, run peak 105 GiB) because that cut's bounce is
#: priced at the WORST-CASE lane count (no boot has measured it).  An operator
#: who has the host room grants it here, by number, and the ledger still has
#: the last word.
PCUT_BOUNCE_SLACK_ENV = "FLLIPER_PDFLIP_PCUT_BOUNCE_SLACK_GIB"






def host_price_for_cut(ns, stage_ratio: str) -> Tuple[float, int, str]:
    """``(xchg_bounce GiB, lanes, lane source)`` the ledger WOULD charge for
    this cut -- the same chain ``choose_host_ledger`` prices (``xchg_lane_count``
    -> ``xchg_bounce_terms_for_arm``), with the same argv-derived inputs, so
    the solver and the ledger cannot disagree on a candidate's host cost.
    ``(0.0, 1, "no exchange")`` on an arm that pins no host bounce."""
    weight_source = str(getattr(ns, "pdflip_weight_source", WEIGHT_SOURCE_DEFAULT))
    oncard_mode = str(getattr(ns, "pdflip_xchg_oncard", ONCARD_MODE_DEFAULT))
    if not xchg_bounce_arm_pins_host(weight_source, oncard_mode):
        return 0.0, 1, "no exchange"
    legs = str(getattr(ns, "pdflip_xchg_legs", "both") or "both")
    lanes, prov = xchg_lane_count(stage_ratio, legs)
    source = ("measured" if "source=measured" in prov
              else "WORST-CASE" if "WORST-CASE" in prov else "n/a")
    if host_price_is_region_form(ns):
        # #1464: THE PER-LANE ASSEMBLE BUFFERS ARE NOT ALLOCATED IN THE
        # REGION FORM, so they must not decide the cut.  Every production
        # exchange boot arms `--pdflip-xchg-inject` (choices: shadow |
        # authoritative -- there is no "off"), and both leg call sites in
        # weight_updater pass `hook=` + `sems=`, so `_pdflip_xchg_bounce_leg`
        # takes the BounceSlots/region branch and never reaches
        # `run_bounce_leg` (the `LayerBounce` that creates
        # bounce.bin.{c0,c1,p1,p2,p4} and emits PDFLIP-XCHG-HOST-SLOT).
        # MEASURED, boot weg2xsn218 (0113846b61, 2026-09-17): 0 HOST-SLOT
        # lines on P and D; /dev/shm/pdflip-xchg-<epoch>/ holds xchg.bin
        # (385 MiB apparent, 1 MiB resident) and a 3.3 KB bnc index; the
        # host sampler read shmem +0.094 GiB across the D->P flip and
        # +0.007 GiB across the P->D flip, anon +0.28 GiB.  The 15.75 GiB
        # lane price (5 lanes x 3.00 GiB + 0.75 staging) was measured on
        # weg2xsn28/31 (2026-09-13) in the FILE bounce form and kept the
        # solver on 39,13,12 for every boot since: every unmeasured cut was
        # charged the WORST-CASE 9 lanes = 27.75 GiB and refused.  Priced
        # by the form that runs, every cut costs the same 385 MiB region,
        # so the fastest frontier cut at/above the pool floor ships.
        # NOT CHANGED HERE: the ledger's ARM term (choose_host_ledger ->
        # xchg_bounce_terms_for_arm) still charges the lane price.  That
        # is deliberate and named: xsn218's ledger predicted the run peak
        # at 91.91 GiB and the sampler measured 85.86 GiB non-reclaimable;
        # without the 15.75 the prediction would read 76.2 GiB, ~10 GiB
        # UNDER the measurement.  The lane term currently covers an
        # unattributed ~10 GiB, and dropping it before that is attributed
        # would let arena_from_ledger grow the arena into the reap mark.
        _region_gib = float(weight_exchange_region.REGION_BYTES) / host_ledger.GIB
        return _region_gib, int(lanes), "region-form %s" % source
    nbytes, _lines = xchg_bounce_terms_for_arm(
        weight_source, oncard_mode, str(getattr(ns, "model", "")),
        getattr(ns, "pdflip_xchg_oncard_slot_mib", None),
        int(getattr(ns, "xchg_bounce_depth", xchg_bounce.ASSEMBLE_DEPTH_DEFAULT)),
        int(lanes),
        int(xchg_bounce.resolve_lanes_concurrent(getattr(ns, "xchg_lanes_concurrent", None))),
        bool(getattr(ns, "xchg_band_credit", False)))
    return float(nbytes) / host_ledger.GIB, int(lanes), source




















#: VRAM-VERTRAG M1: the pass a D solve's label belongs to (anything else, the
#: early start's expectation, is ``d_early``)
_VRAM_PASS_BY_LABEL = {"D(Karte, Erwartung)": "map", "D(dry, expectation)": "dry",
                       "D(d-only, expectation)": "d_only", "D": "d"}






def _d_extend_cap_rate_env(ns, log, label: str, *, capped: bool = True) -> Optional[str]:
    """PDFLIP-EXTEND-CAP (29.09.): with the record caps armed (the 27B, whose verdict books no
    non-torch term and so never reaches the #145 ledger's EXTEND-STUECKELUNG), the D extend
    chunk must follow the CAP, not only the card: p0-nopin (dkr27browauthorityp0nopinbar1fs09291720)
    D-TP1 died 17:24:06Z at a 4096-row extend on the torch line (reserved 18593 + 40 > cap 18609)
    with 134 MiB still free on the card. Writes ``FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB`` from
    the profile's ``D_EXTEND_CAP_PER_ROW_MIB`` -- the ALLOCATED transient per extend row under the
    cap, a different quantity from Next Flash's uncapped reserved growth -- (extend_trim.width_vote then funds the chunk
    from ``min(card_free, cap - reserved)``); a value named in --env-d wins. A profile without
    the record: nothing written, no vote. ``capped=False`` is the Q-694 flip arm
    (:func:`_d_extend_flip_rate_env`): same record, the vote funds the chunk from
    ``card_free`` alone (no torch cap), and the line says so.

    Q-694b (hardware-generic): the record is a START value measured on one rig. The flip arm
    also arms the per-rank run-time measurement (``FLLIPER_PDFLIP_EXTEND_RATE_MEASURE=1``, line
    ``EXTEND-RATE source=record``); a profile WITHOUT the record no longer leaves the vote unarmed
    -- :func:`_d_extend_derived_rate_env` writes a start rate derived from the model geometry
    (``EXTEND-RATE source=derived``). The P0 arm with its record is byte-identical."""
    try:
        _rate = [None if v is None else float(v) for v in _pconst(D_EXTEND_CAP_RATE_RECORD, ns.profile)]
    except KeyError:
        return _d_extend_derived_rate_env(ns, log, label, capped=capped)
    _rate_src = _pconst_boots(D_EXTEND_CAP_RATE_RECORD, ns.profile)
    from flliper.srt.pdflip import extend_trim as _et

    _rtext = _et.launcher_rates(_rate)
    _env_d = getattr(ns, "env_d", "") or ""
    _given = any(x.split("=", 1)[0].strip() == "FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB"
                 and x.split("=", 1)[1].strip() != _rtext
                 for x in _env_d.split(";") if "=" in x)
    if not _rtext:
        return None
    if not _given:
        ns.env_d = set_group_env(_env_d, "FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB", _rtext)
    _what = ("EXTEND-CAP rows_cap aus min(card_free, torch_cap - reserved)" if capped else
             f"{EXTEND_CAP_FLIP_MARKER} rows_cap aus card_free_post (ohne torch cache cap, Flip-Linie)")
    log(f"{D_RANK_SOLVE_MARKER} {label} {_what}: "
        f"FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB={_rtext}{' (aus --env-d, Vorrang)' if _given else ''} "
        f"(Rate aus {D_EXTEND_CAP_RATE_RECORD} {_rate}, boots {_rate_src}; D kappt den Extend-Chunk "
        f"auf floor((post - 300) / Rate), MIN ueber TP)")
    if not capped:
        _d_extend_rate_measure_env(ns, log, label, "record", _rtext,
                                   f"{D_EXTEND_CAP_RATE_RECORD} {_rate}, boots {_rate_src}")
    return _rtext


#: Q-694b: the arming variable of the per-rank run-time measurement of the extend rate
EXTEND_RATE_MEASURE_ENV = "FLLIPER_PDFLIP_EXTEND_RATE_MEASURE"
#: Q-694b: the launcher's line naming where the start rate of the chunk cap came from
EXTEND_RATE_MARKER = "EXTEND-RATE"




def _d_extend_derived_rate_env(ns, log, label: str, *, capped: bool) -> Optional[str]:
    """Q-694b: the chunk-cap start rate of a profile WITHOUT ``D_EXTEND_CAP_PER_ROW_MIB``
    (another model, another card mix): derived from the checkpoint's geometry
    (:func:`extend_trim.derived_rate_mib`, no rig number), the run-time measurement armed.

    Next Flash UNCHANGED: a profile carrying ``D_EXTEND_GROWTH_PER_ROW_MIB`` gets its rate from
    the #145 ledger's EXTEND-STUECKELUNG write -- nothing is written here, byte for byte as
    before. An unreadable geometry writes nothing and says the cap is NOT armed."""
    if d_extend_growth_per_row_record(getattr(ns, "profile", None))[0] is not None:
        return None
    from flliper.srt.pdflip import extend_trim as _et

    try:
        _rate = _et.derived_rate_mib(_model_config(str(getattr(ns, "model", "") or "")))
        _geo_err = "" if _rate is not None else "keine hidden_size"
    except Exception as exc:  # noqa: BLE001 -- an unreadable geometry is NAMED below, never fatal
        _rate, _geo_err = None, f"{type(exc).__name__}: {exc}"
    if _rate is None:
        log(f"{D_RANK_SOLVE_MARKER} {label} {EXTEND_RATE_MARKER} source=none: kein "
            f"{D_EXTEND_CAP_RATE_RECORD} im Profil {getattr(ns, 'profile', None)} und keine "
            f"Modellgeometrie ({_geo_err}) -- der Extend-Chunk-Deckel ist NICHT scharf")
        return None
    _rtext = _et.launcher_rates([_rate])
    _env_d = getattr(ns, "env_d", "") or ""
    _given = any(x.split("=", 1)[0].strip() == "FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB"
                 and x.split("=", 1)[1].strip() != _rtext
                 for x in _env_d.split(";") if "=" in x)
    if not _given:
        ns.env_d = set_group_env(_env_d, "FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB", _rtext)
    _what = ("EXTEND-CAP rows_cap aus min(card_free, torch_cap - reserved)" if capped else
             f"{EXTEND_CAP_FLIP_MARKER} rows_cap aus card_free_post (ohne torch cache cap, Flip-Linie)")
    log(f"{D_RANK_SOLVE_MARKER} {label} {_what}: "
        f"FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB={_rtext}{' (aus --env-d, Vorrang)' if _given else ''} "
        f"(Startrate aus der Modellgeometrie, kein {D_EXTEND_CAP_RATE_RECORD} im Profil "
        f"{getattr(ns, 'profile', None)}; D kappt den Extend-Chunk auf floor((post - 300) / Rate), "
        f"MIN ueber TP)")
    _d_extend_rate_measure_env(
        ns, log, label, "derived", _rtext,
        f"Geometrie {getattr(ns, 'model', '')}: ({_et.DERIVED_HIDDEN_COPIES} x hidden + "
        f"{_et.DERIVED_MLP_COPIES} x intermediate + {_et.DERIVED_MIXER_COPIES} x mixer [+ MoE top-k]) "
        f"x max({_et.DERIVED_MIN_ELEM_BYTES}, dtype) Bytes je Zeile, ohne TP-Teilung")
    return _rtext


#: Q-694: the line of the uncapped (flip-layout) arm of the EXTEND-CAP rate
EXTEND_CAP_FLIP_MARKER = "Q694 EXTEND-CAP-FLIP"


def _d_extend_flip_rate_env(ns, log, label: str, cards: Optional[Sequence[Card]] = None) -> Optional[str]:
    """Q-694 (INT8 y8va, 03.10. 18:39:37Z, D-TP0 OOM): the D extend chunk follows the card on
    the FLIP line too, not only under the P0 torch cache cap.

    rc12g's chunk cap (``extend_trim.width_vote``, a vote in the #794 MIN reduce) is armed by
    ``FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB``, and the launcher wrote that rate on two paths only:
    the #145 ledger (Next Flash's ``D_EXTEND_GROWTH_PER_ROW_MIB``; the 27B is dense, its fraction
    solve is SKIPPED) and the P0 cap branch (:func:`_d_extend_cap_rate_env`, only with
    ``FLLIPER_PDFLIP_TORCH_CACHE_CAP=1``; the qwen27b registry row has it off). So the 27B INT8 flip
    boot ran with the trim threshold (``FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,0,0``) but no chunk cap:
    the trim emptied the cache to 529 MiB free and the extend still went 4096 rows deep (prefix
    70169, pdflip-30-101), whose transient on TP0 measures 518-1001 MiB at 4096 rows -> barlink
    all_reduce ``torch.empty_like`` 20 MiB with 3.44 MiB free. The #656 corridor actuator did not
    cut it: its config price for 4096 rows is 198 MiB (a third of what TP0 takes).

    Same record as the P0 branch (``D_EXTEND_CAP_PER_ROW_MIB`` 0.3091, the deep-prefix maximum of
    the ALLOCATED transient per row; 40 boots up to 03.10. measured <= 0.2690 on every extend of
    >= 2048 rows), so ``rows_cap = floor((card_free_post - 300) / 0.3091)``: 740 rows at the
    death's 529 MiB. DUAL UNCHANGED: the dual layout (``--dual-layout`` / ``--dual-share``)
    writes nothing here, byte for byte as before. Next Flash (its own record
    ``D_EXTEND_GROWTH_PER_ROW_MIB`` through the #145 ledger): nothing written. Q-694b: any other
    profile without the record gets a geometry-derived start rate; the record, where present, is
    only the start value of the per-rank run-time measurement."""
    if getattr(ns, "dual_layout", False) is True or getattr(ns, "dual_share", False) is True:
        return None
    _rtext = _d_extend_cap_rate_env(ns, log, label, capped=False)
    if _rtext:
        _d_extend_cap_floor_env(ns, log, label)
        # #201 TRIM-LAW: the fixed part first -- the trim threshold prices the line the ranks will keep, and under
        # VOTE-LINE-SINGLE that line reads the fixed part from this same env
        _d_extend_vote_fixed_env(ns, log, label, cards)
        _d_extend_flip_trim_env(ns, log, label, _rtext, cards)
    return _rtext
















def log_d_rank_vram_solve(ns, cards: List[Card], budgets_d: Sequence[int], log,
                          label: str, *, p_split=None, chunk_layers=None,
                          card_terms: Optional[Sequence[Dict[str, object]]] = None) -> None:
    """#145: die D-Seite des Planners spuckt ihre Sizing-Zahlen aus.

    ZWEI Zeilen, und die Reihenfolge ist Absicht:

    1. DAS VERDIKT -- rein physisch, ohne jede Modellgeometrie: passt das
       Budget, das dieser Launcher dem Rang GERADE gibt, in
       ``karte - fremd_kontext - nicht_torch - reserve``? Das ist die Zeile,
       die fnFL2w83 haette verhindern muessen (Rang 0: 29680 gefragt,
       32607-1342-3628 = 27637 verfuegbar).
    2. DIE FRACTION-DECKE -- das D-Gegenstueck zu #140/#141, seit H8 gegen
       das BUDGET mit den Metallregeln (Pufferregel ``min(R+S,E)``, fester
       Rang-Posten gemessen, Aktivierung, 262k-KV;
       :func:`expert_residency.plan_d_residency`). Nur wenn die Geometrie
       VOLLSTAENDIG vorliegt; fehlt ein Vektor, wird er BENANNT und die Zeile
       bleibt aus, statt dass eine halbe Geometrie eine Zahl erfindet.
       Passt die GEFAHRENE Fraction auf einem Rang nicht, VERWEIGERT diese
       Funktion den Boot (W122) -- im Dry-Run, bevor irgendein Rang laedt.

    Aufgerufen an BEIDEN Stellen, an denen ``budgets_d`` entsteht (Dry-Run und
    echter Lauf) -- die #114-Falle: eine Fassung, die nur eine der beiden
    trifft, fehlt im Dry-Run genau dann, wenn der Gate-Lauf sie sucht.
    """
    from flliper.srt.planner import pp_cut as _pp_cut

    # #251c: a second pass (dry-run expectation, then the real budgets) prices
    # the bank as priced, not as the first pass's stage form re-split it
    d_kv_stage_undo(ns)
    n = len(list(budgets_d))
    absent: List[str] = []

    def _plan_without_solve() -> None:
        # M1: a pass whose solve does not run (27B dense, half geometry, an
        # unreadable reference) still writes its plan -- the D closure is then
        # named open, never dropped with the pass.
        vram_plan_emit(ns, cards, _VRAM_PASS_BY_LABEL.get(label, "d_early"), log, d_label=label)

    def _vec(attr: str, name: str):
        raw = str(getattr(ns, attr, "") or "").strip()
        if not raw:
            absent.append(name)
            return None
        v = _csv_floats(raw)
        if len(v) != n:
            log(f"{D_RANK_SOLVE_MARKER} IGNORED: {name} has {len(v)} "
                f"Eintraege, die D-Gruppe hat {n} Raenge")
            absent.append(name)
            return None
        return v

    fremd = _vec("d_foreign_context_mib", "--d-foreign-context-mib")
    nicht_torch = _vec("d_nontorch_mib", "--d-nontorch-mib")
    reserve = _vec("d_reserve_mib", "--d-reserve-mib")
    totals = [float(c.total_mib) for c in cards]

    # Ein FEHLENDER Term ist 0 und heisst "nicht gebucht" -- er wird im Druck
    # namentlich genannt, damit niemand die Zeile fuer vollstaendig haelt.
    # Genau das ist der Unterschied zu einem geratenen Default.
    # #158: den Korridor-Floor mitgeben, damit der Verdikt die BETRIEBSREGEL
    # bewerten kann und nicht nur die Physik. fnFL2w130/w131 sagten beide
    # "fits", liefen auf 124 MiB frei und verloren P.
    try:
        from flliper.srt.managers import corridor_guard as _cg

        _floor = float(_cg.corridor_band_floor_mib())
    except BaseException:  # noqa: BLE001 -- ohne Floor wird NICHT geprueft,
        _floor = 0.0       # und der Verdikt sagt das auch ("NICHT GEPRUEFT")
    # 27B port fix 21.10. (rc29b INT8: the verdict printed REST 3096 MiB on D TP1, the card had 1332 MiB free at idle):
    # without the two flags the verdict booked 0 beside torch while the BUDGET of this same pass charged the driver carve
    # and P's dormant footprint (card_terms) -- and the card holds D's measured non-torch part on top. Unset flags are now
    # booked from what is measured, never guessed: fremd = the budget's dormant_other term, nichttorch = the budget's
    # driver_carve term + the profile record D_TORCH_CAP_OTHER_MIB (D's non-torch part, max over the awake VRAM-PEAK lines of
    # 8 boots). A flag given on the command line still wins; without terms or record the term stays "NICHT GEBUCHT".
    try:
        _other = list(_pconst(_budget_rest.other_record_name("D"), ns.profile))
    except KeyError:
        _other = None
    _v_fremd, _v_nt, _booked_from = d_verdict_measured_terms(fremd, nicht_torch, card_terms, _other, n,
                                                            _budget_rest.other_record_name("D"))
    if _v_fremd is not None and fremd is None and "--d-foreign-context-mib" in absent:
        absent.remove("--d-foreign-context-mib")
    if _v_nt is not None and nicht_torch is None and "--d-nontorch-mib" in absent:
        absent.remove("--d-nontorch-mib")
    rank_verdicts = _pp_cut.d_rank_budget_verdict(
        budgets_mib=[float(b) for b in budgets_d],
        card_total_mib=totals,
        foreign_context_mib=_v_fremd if _v_fremd is not None else [0.0] * n,
        nontorch_mib=_v_nt if _v_nt is not None else [0.0] * n,
        reserve_mib_by_rank=reserve,
        corridor_floor_mib=_floor,
    )
    _corridor_riss = [v.rank for v in rank_verdicts if not v.corridor_ok]
    above = [v.rank for v in rank_verdicts if not v.fits]
    # VRAM loop P0 (28.09.): with FLLIPER_PDFLIP_TORCH_CACHE_CAP=1 in --env-d every
    # D rank caps its caching allocator at the verdict's verfuegbar minus the
    # corridor floor -- the booked size, not a reserve (pdflip/torch_cache_cap.py).
    if str(parse_group_env(getattr(ns, "env_d", "") or "").get(TORCH_CACHE_CAP_ENV, "0")
           ).strip() == "1":
        from flliper.srt.pdflip import torch_cache_cap as _tcc

        _rec_caps = (d_record_torch_caps(ns, cards, budgets_d, log, label)
                     if fremd is None and nicht_torch is None else None)
        if _rec_caps is not None:
            _caps = ",".join(str(int(x)) for x in _rec_caps)
            ns.env_d = set_group_env(getattr(ns, "env_d", "") or "", _tcc.MIB_ENV, _caps)
            log(f"{D_RANK_SOLVE_MARKER} {label} P0 {_tcc.MARKER} {_tcc.MIB_ENV}={_caps} "
                f"({_budget_rest.OVERHANG_MARKER}: physische Linie je Rang = Budget + gebuchter "
                f"Rest - {_budget_rest.other_record_name('D')} = total - carve - dormant_other - "
                f"nicht-torch; das Verdikt bucht hier kein fremd/nicht-torch)")
            if _d_extend_cap_rate_env(ns, log, label):
                _d_extend_vote_fixed_env(ns, log, label, cards)
        else:
            _caps = _tcc.launcher_caps(rank_verdicts, _floor)
            ns.env_d = set_group_env(getattr(ns, "env_d", "") or "", _tcc.MIB_ENV, _caps)
            log(f"{D_RANK_SOLVE_MARKER} {label} P0 {_tcc.MARKER} {_tcc.MIB_ENV}={_caps} (verfuegbar "
                f"minus Korridor-Floor {_floor:.0f} MiB je Rang; der torch-Allokator leert seinen "
                f"Cache, bevor er die Linie ueberschreitet)")
    else:
        # Q-694: without the torch cap the flip line's D extend chunk follows
        # card_free_post (the rc12g vote); the dual layout writes nothing.
        _d_extend_flip_rate_env(ns, log, label, cards)
    log(
        "%s VERDIKT %s: %s%s"
        % (
            D_RANK_SOLVE_MARKER,
            label,
            " | ".join(
                "rang%d nvml%d %s: total %.0f - fremd %.0f - nichttorch %.0f "
                "- reserve %.0f = %.0f verfuegbar, gefragt %.0f -> %s %.0f MiB"
                % (v.rank, cards[v.rank].nvml_index, cards[v.rank].name,
                   v.card_total_mib, v.foreign_context_mib, v.nontorch_mib,
                   v.reserve_mib, v.available_mib, v.asked_mib,
                   "REST" if v.fits else "DARUEBER", abs(v.over_mib))
                + " [" + v.corridor_note + "]"
                for v in rank_verdicts
            ),
            ((" -- DARUEBER on rank %s" % above) if above else "")
            + ((" -- KORRIDOR GERISSEN auf Rang %s: die Bytes passen, die "
                "Betriebsregel nicht" % _corridor_riss) if _corridor_riss else "")
            + ((" [GEBUCHT AUS MESSUNG: %s]" % "; ".join(_booked_from)) if _booked_from else "")
            + ((" [NICHT GEBUCHT: %s -- this row is therefore a OBERGRENZE, "
                "keine Bilanz]" % ", ".join(absent)) if absent else ""),
        )
    )

    # ---- 2. die Fraction-Decke, nur bei VOLLSTAENDIGER Geometrie ----------
    # PDFLIP-FORM: the fraction solve and the H14 wake credit it feeds are
    # declared for arch=moe; any other form says so by name instead of
    # listing the MoE vectors it was never meant to carry as "missing".
    _form = getattr(ns, "pdflip_boot_form", None)
    _skips = [x for x in (
        pdflip_form.gate_skip_line("#145 D-RANK FRACTION-SOLVE", _form),
        pdflip_form.gate_skip_line("H14 WAKE-CREDIT", _form)) if x]
    if _skips:
        for _x in _skips:
            log(f"{D_RANK_SOLVE_MARKER} {label}: {_x}")
        _plan_without_solve()
        return
    gaps: List[str] = []
    ratios = _argv_vector(getattr(ns, "extra_d", ""), "--rank-moe-ratio")
    tp_ratio = _argv_vector(getattr(ns, "extra_d", ""), "--rank-tp-ratio")
    fr_d = _argv_vector(getattr(ns, "extra_d", ""), "--rank-moe-resident-fraction")
    _env_d = parse_group_env(getattr(ns, "env_d", "") or "")
    _scratch_raw = _env_d.get("FLLIPER_MOE_SCRATCH_SLOTS", "")
    scratch = _csv_floats(_scratch_raw) if _scratch_raw else []
    if len(scratch) == 1:
        scratch = scratch * n
    if not ratios or len(ratios) != n:
        gaps.append("--rank-moe-ratio (extra_d)")
    if not tp_ratio or len(tp_ratio) != n:
        gaps.append("--rank-tp-ratio (extra_d)")
    if not fr_d or len(fr_d) != n:
        gaps.append("--rank-moe-resident-fraction (extra_d)")
    if len(scratch) != n:
        gaps.append("FLLIPER_MOE_SCRATCH_SLOTS (env_d)")
    if gaps:
        log(f"{D_RANK_SOLVE_MARKER} FRACTION-SOLVE {label} ENTFAELLT: "
            f"{', '.join(gaps)} fehlt/passt nicht zu {n} Raengen. Eine "
            f"halbe Geometrie loest nichts (#91), also rechnet hier nichts.")
        _plan_without_solve()
        return
    # H8: DIE METALLREGELN. Die alte Decke rechnete gegen die KARTE mit
    # ``fraction x Spanne + Scratch`` Zeilen und ohne KV/Aktivierung -- sie
    # nannte TP1 0.807 und TP2 0.661, beide toeten den Boot (x98/x99 KV-Pool-
    # ValueError 288 MiB, x97 OOM beim Laden). Jetzt: Budget je Rang, Puffer
    # min(R+S,E) mit E = Spanne + Pad, fester Rang-Posten gemessen, 262k-KV.
    from flliper.srt.planner import expert_residency as _er

    # rc12c: the fixed post from the profile's record (measured on the form
    # that runs today) and the card from the budget's own terms -- only where
    # the profile measured D's awake rest; otherwise byte-identical.
    _fixed_rec, _fixed_src = (d_fixed_record(ns.profile) if card_terms is not None
                              else (None, ""))
    _ledger = (d_card_ledger(card_terms, budgets_d, label)
               if card_terms is not None and len(card_terms) == n else None)
    # rc12e: the D activation measured on the same form (max extend transient)
    _act_rec, _act_src = (d_activation_record(ns.profile) if card_terms is not None
                          else (None, ""))
    _derive_waves = bool(getattr(ns, "d_pool_waves_derived", False))
    # #239 S3f Miss-Record: measured cost per missed expert row, else the seed
    _miss_ms, _miss_src = d_owned_miss_ms(
        ns, env_d=_env_d,
        host=next((i for i, v in enumerate(tp_ratio) if float(v) > 0), 0))
    if _miss_ms is not None:
        log(f"{D_RANK_SOLVE_MARKER} {label} D-EIGENTUM FEHLGRIFF-KOSTEN (#239 S3f): "
            f"{_miss_ms[0]:g}/{_miss_ms[1]:g} ms je Zeile (Host/Worker) aus {_miss_src} "
            f"statt der Saat")
    elif _miss_src:
        log(f"{D_RANK_SOLVE_MARKER} {label} D-EIGENTUM FEHLGRIFF-KOSTEN (#239 S3f): "
            f"Saat -- {_miss_src}")
    # 01.10.: the same records per rank -- each card at its own link
    _miss_rank, _miss_rank_src = d_owned_miss_ms_rank(ns, env_d=_env_d, n=n)
    _heat_recs, _heat_root = d_owned_heat_records(ns, env_d=_env_d)
    # S54: each D rank's link for the extend term of the owned solve
    _ext_gbps, _ext_src = d_owned_extend_h2d(cards)
    if _miss_rank is not None:
        log(f"{D_RANK_SOLVE_MARKER} {label} D-EIGENTUM FEHLGRIFF-KOSTEN JE RANG (01.10.): "
            f"{'/'.join('%.4f' % x for x in _miss_rank)} ms je Zeile (TP0..TP{n - 1}) "
            f"aus {_miss_rank_src}")
    # #239: the token cut of the full-attention KV (None = off, byte-identical)
    _kv_cut = d_kv_token_cut(ns)
    _pinned = getattr(ns, "_d_map_form", None)
    if _pinned:
        # #239 rc12z29c: the Platztausch map was built from this form (P loaded
        # it) -- check exactly it: fixed cut, ownership and FR_D from --extra-d
        _kv_cut = tuple(_pinned["cut"]) if _pinned.get("cut") else None
        log(f"{D_RANK_SOLVE_MARKER} {label} KARTE-FORM (#239 rc12z29c): geprueft, nicht neu "
            f"geloest -- Eigentum {','.join(_pinned['ratios'])}, FR_D "
            f"{','.join(_pinned['fractions'])}, Schnitt "
            f"{list(_kv_cut) if _kv_cut else 'keiner'} (die Form der Karte)")
    elif _er.owned_cut_request(_kv_cut)[0] and getattr(ns, "_d_owner_stated", None):
        # #239 S3f: every pass solves the ownership from the STATED vector
        ratios = list(ns._d_owner_stated)
    _kv_cut_kw = ({} if _kv_cut is None else
                  dict(kv_token_shares=_kv_cut, kv_dtype_bytes=d_kv_dtype_bytes(ns)))
    # YaRN x2 (27.09.): the KV duty is the D form's own context (#77), not
    # the rig's fixed 262144 -- a 524288 form priced at 262144 "fitted".
    _d_kv, _d_kv_src = group_kv_tokens(ns, "d")
    if _d_kv != CONTEXT_LENGTH_TOKENS:
        log(f"{D_RANK_SOLVE_MARKER} {label} KV-PFLICHT {_d_kv} tokens from {_d_kv_src} "
            f"(statt {CONTEXT_LENGTH_TOKENS}): der KV-Posten jedes Rangs ist damit bepreist")
    try:
        plan = _er.plan_d_residency(
            model_path=ns.model,
            budgets_mib=[float(b) for b in budgets_d],
            ratios=[float(x) for x in ratios],
            fractions=[float(x) for x in fr_d],
            scratch_rows=[int(x) for x in scratch],
            rank_tp_ratio=",".join(tp_ratio),
            env_d=_env_d,
            reference_logs=ns.d_residency_reference_logs,
            kv_tokens=_d_kv,
            label=label,
            marker=D_RANK_SOLVE_MARKER,
            # H33: die KARTEN-Bilanz (Posten ausserhalb des Budgets, W130).
            card_reference_logs=getattr(ns, "d_card_reference_logs", "") or "",
            # H64 (NF): die Verify-Form der D-Gruppe; None unter
            # --d-replayssm-spec off -- die Rechnung bleibt dann byte-gleich.
            replayssm_spec=d_replayssm_spec_plan_form(ns),
            # H91b: die Sitze, die D wirklich faehrt, und was ihr Graph je
            # zusaetzlichem Sitz kostet (gemessen, sonst benannt ungebucht).
            seats=d_stated_seats(ns),
            seat_graph_mib=d_seat_graph_mib(ns),
            reference_seats=int(getattr(ns, "d_residency_reference_seats", 1) or 1),
            fixed_record_mib=_fixed_rec,
            fixed_record_source=_fixed_src,
            card_ledger=_ledger,
            activation_record_mib=_act_rec,
            activation_record_source=_act_src,
            derive_waves=_derive_waves,
            owned_miss_ms=_miss_ms,
            owned_miss_source=_miss_src,
            owned_miss_ms_rank=_miss_rank,
            owned_miss_rank_source=_miss_rank_src,
            owned_heat_records=_heat_recs,
            owned_heat_source=_heat_root,
            owned_extend_h2d_gbps=_ext_gbps,
            owned_extend_source=_ext_src,
            **_kv_cut_kw,
        )
        if _ledger is not None and plan.refusal is not None and plan.fits:
            # THE CONSUMER FOLLOWS THE BOOKED BUDGET (rc12c): lower the
            # expert scratch to the planner's edge, then solve again with it.
            _capped, _cap_lines = d_scratch_cap(plan.fits, [int(x) for x in scratch])
            if _cap_lines:
                for _ln in _cap_lines:
                    log(f"{D_RANK_SOLVE_MARKER} {label} {_ln}")
                scratch = [float(x) for x in _capped]
                ns.env_d = set_group_env(getattr(ns, "env_d", "") or "",
                                         "FLLIPER_MOE_SCRATCH_SLOTS",
                                         ",".join(str(int(x)) for x in _capped))
                _env_d = parse_group_env(ns.env_d)
                plan = _er.plan_d_residency(
                    model_path=ns.model,
                    budgets_mib=[float(b) for b in budgets_d],
                    ratios=[float(x) for x in ratios],
                    fractions=[float(x) for x in fr_d],
                    scratch_rows=[int(x) for x in scratch],
                    rank_tp_ratio=",".join(tp_ratio),
                    env_d=_env_d,
                    reference_logs=ns.d_residency_reference_logs,
                    kv_tokens=_d_kv,
                    label=label,
                    marker=D_RANK_SOLVE_MARKER,
                    card_reference_logs=getattr(ns, "d_card_reference_logs", "") or "",
                    replayssm_spec=d_replayssm_spec_plan_form(ns),
                    seats=d_stated_seats(ns),
                    seat_graph_mib=d_seat_graph_mib(ns),
                    reference_seats=int(getattr(ns, "d_residency_reference_seats", 1) or 1),
                    fixed_record_mib=_fixed_rec,
                    fixed_record_source=_fixed_src,
                    card_ledger=_ledger,
                    activation_record_mib=_act_rec,
                    activation_record_source=_act_src,
                    derive_waves=_derive_waves,
                    owned_miss_ms=_miss_ms,
                    owned_miss_source=_miss_src,
                    owned_miss_ms_rank=_miss_rank,
                    owned_miss_rank_source=_miss_rank_src,
                    owned_heat_records=_heat_recs,
                    owned_heat_source=_heat_root,
                    owned_extend_h2d_gbps=_ext_gbps,
                    owned_extend_source=_ext_src,
                    **_kv_cut_kw,
                )
        if d_stated_seats(ns) is not None:
            for _ln in d_seat_lines(ns):
                log(f"{D_RANK_SOLVE_MARKER} {label} {_ln}")
    except (OSError, KeyError, ValueError, _pp_cut.DraftResidencyUnavailable) as _exc:
        # Eine UNLESBARE Geometrie/Referenz verweigert nicht den Boot, sie wird
        # benannt; die Verweigerung unten kommt nur aus einer GERECHNETEN Bilanz.
        log(f"{D_RANK_SOLVE_MARKER} FRACTION-SOLVE {label} failed: "
            f"{type(_exc).__name__}: {_exc}")
        _plan_without_solve()
        return
    for line in plan.lines:
        log(line)
    if plan.solved_owner_ratio:
        # #239 S3f: the ownership is the planner's -- published before FR_D
        # and the token vector, and every later reader of --rank-moe-ratio
        # (seat table, Platztausch map, geometry) sees the same vector.
        ratios = publish_d_owner_ratio(ns, plan, label, log)
    if plan.solved_fractions:
        # #239 S2b: FR_D is the planner's -- the D group runs the edge it solved
        # together with the token cut (the stated FR was only the start point).
        from flliper.srt.pdflip import draft_post as _dp

        ns.extra_d = _dp.replace_vector_flag(getattr(ns, "extra_d", "") or "",
                                             "--rank-moe-resident-fraction",
                                             plan.solved_fractions)
        ns.env_d = _dp.replace_env_vector(getattr(ns, "env_d", "") or "",
                                          "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION",
                                          plan.solved_fractions)
        log(f"{D_RANK_SOLVE_MARKER} {label} FR-D (#239 S2b) veroeffentlicht: "
            f"--extra-d --rank-moe-resident-fraction "
            f"{','.join(_dp._fmt(x) for x in plan.solved_fractions)} (vorher "
            f"{','.join(str(x) for x in fr_d)}), Token-Schnitt {list(plan.kv_token_cut)}/"
            f"{_er.KV_TOKEN_SHARE_GRID}")
    # VRAM loop P1c (28.09.): the planner's edge wins over the stated FR_D
    # (outside an owned cut and an S2b solve), coupled to P0 in this group.
    _fr_new, _fr_lines = d_fr_ceiling_adopt(
        plan, fr_d, _env_d, bool(_er.owned_cut_request(_kv_cut)[0]),
        map_built=bool(_pinned or getattr(ns, "_expert_map_path", "")))
    for _ln in _fr_lines:
        log(f"{D_RANK_SOLVE_MARKER} {label} {_ln}")
    if _fr_new is not None:
        from flliper.srt.pdflip import draft_post as _dp

        ns.extra_d = _dp.replace_vector_flag(getattr(ns, "extra_d", "") or "",
                                             "--rank-moe-resident-fraction", _fr_new)
        ns.env_d = _dp.replace_env_vector(getattr(ns, "env_d", "") or "",
                                          "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION", _fr_new)
        _env_d = parse_group_env(ns.env_d)
        log(f"{D_RANK_SOLVE_MARKER} {label} {FR_D_CEILING_MARKER} veroeffentlicht: --extra-d "
            f"--rank-moe-resident-fraction {','.join(_dp._fmt(x) for x in _fr_new)} (vorher "
            f"{','.join(str(x) for x in fr_d)}; der Planer setzt FR_D, das Profil ist nur "
            f"noch benannte Obergrenze {FR_D_CAP_ENV})")
        fr_d = [float(x) for x in _fr_new]
    # #239 S3a: the cut itself goes to D as the runtime's token vector.
    ns._d_solved_cut = tuple(getattr(plan, "kv_token_cut", ()) or ()) or None
    _cut_line = publish_d_token_cut(
        ns, d_token_cut_vector(getattr(plan, "kv_token_cut", ()), _kv_cut), label)
    if _cut_line:
        log(f"{D_RANK_SOLVE_MARKER} {_cut_line}")
    if plan.overflow_waves is not None and plan.overflow_waves != _er.pool_overflow_waves(_env_d):
        # rc12e: the D group runs the waves its solved form needs.
        ns.env_d = set_group_env(getattr(ns, "env_d", "") or "",
                                 _er.POOL_OVERFLOW_WAVES_ENV, str(int(plan.overflow_waves)))
        _env_d = parse_group_env(ns.env_d)
    # H95: the seats are dynamic 1..--d-bs per D phase -- the same solve per
    # seat count n (seat_rebook re-books the seat posts), one line each.
    # #239 S3f: under an owned cut the table recomputes the solved form.
    _seat_cut_kw, _seat_fr = d_seat_table_form(plan, _kv_cut_kw, fr_d)
    for _ln in d_seat_table_lines(ns, _er, dict(
            model_path=ns.model,
            budgets_mib=[float(b) for b in budgets_d],
            ratios=[float(x) for x in ratios],
            fractions=_seat_fr,
            scratch_rows=[int(x) for x in scratch],
            rank_tp_ratio=",".join(tp_ratio),
            env_d=_env_d,
            reference_logs=ns.d_residency_reference_logs,
            kv_tokens=_d_kv,
            label=label,
            marker=D_RANK_SOLVE_MARKER,
            card_reference_logs=getattr(ns, "d_card_reference_logs", "") or "",
            seat_graph_mib=d_seat_graph_mib(ns),
            reference_seats=int(getattr(ns, "d_residency_reference_seats", 1) or 1),
            fixed_record_mib=_fixed_rec,
            fixed_record_source=_fixed_src,
            card_ledger=_ledger,
            activation_record_mib=_act_rec,
            activation_record_source=_act_src,
            derive_waves=_derive_waves,
            owned_miss_ms=_miss_ms,
            owned_miss_source=_miss_src,
            owned_miss_ms_rank=_miss_rank,
            owned_miss_rank_source=_miss_rank_src,
            owned_heat_records=_heat_recs,
            owned_heat_source=_heat_root,
            owned_extend_h2d_gbps=_ext_gbps,
            owned_extend_source=_ext_src,
            **_seat_cut_kw,
    ), label):
        log(_ln)
    if plan.refusal is not None:
        log(f"{D_RANK_SOLVE_MARKER} {plan.refusal}")
        refusals.refuse_value("D-BUDGET", plan.refusal, PdFlipLaunchRefused, log)
    _lru_add = getattr(ns, "_d_kv_stage_lru_raise", None)
    ns._d_kv_stage_lru_raise = None
    _own_add = {r: int(k) for r, k in enumerate(getattr(plan, "owned_scratch_raise", ()) or ())
                if int(k) > 0}
    if _own_add:
        # 01.10. (y6n): the owned solve's wave floor needs these scratch rows
        # (rows from resident to scratch, FR_D already solved with them) --
        # the same raise-and-solve-again as the #251c LRU floor
        log(f"{D_RANK_SOLVE_MARKER} {label} D-EIGENTUM GRENZEN: Wellenboden requires Scratch "
            + ", ".join("rang%d +%d" % (r, k) for r, k in sorted(_own_add.items())))
        _lru_add = {r: max(int((_lru_add or {}).get(r, 0)), int(_own_add.get(r, 0)))
                    for r in set(_lru_add or {}) | set(_own_add)}
    if _lru_add:
        # (3) 29.09.: a KV rank's LRU falls below its measured peak at an
        # allowed stage -- raise its scratch and solve again, so the planner
        # moves the rows from resident to scratch (FR_D) before the map pins
        _round = int(getattr(ns, "_d_kv_stage_lru_round", 0) or 0)
        _what = ", ".join("rang%d +%d" % (r, k) for r, k in sorted(_lru_add.items()))
        if _pinned:
            log(f"{D_RANK_SOLVE_MARKER} {label} {D_KV_STAGE_LRU_FLOOR_MARKER}: Scratch {_what} "
                f"waere noetig, die Karte ist gebaut -- nicht neu geloest (die Peak kostet "
                f"eine Zusatzwelle)")
        elif _round >= D_KV_STAGE_LRU_FLOOR_ROUNDS:
            log(f"{D_RANK_SOLVE_MARKER} {label} {D_KV_STAGE_LRU_FLOOR_MARKER}: nach {_round} "
                f"Runden noch {_what} -- nicht reached, die Peak kostet eine Zusatzwelle")
        else:
            d_kv_stage_undo(ns)
            _base = _rank_vec(parse_group_env(getattr(ns, "env_d", "") or "")
                              .get("FLLIPER_MOE_SCRATCH_SLOTS"), n)
            _new = [int(x) + int(_lru_add.get(r, 0)) for r, x in enumerate(_base)]
            ns.env_d = set_group_env(getattr(ns, "env_d", "") or "", "FLLIPER_MOE_SCRATCH_SLOTS",
                                     ",".join(str(x) for x in _new))
            log(f"{D_RANK_SOLVE_MARKER} {label} {D_KV_STAGE_LRU_FLOOR_MARKER}: Scratch {_what} "
                f"-> FLLIPER_MOE_SCRATCH_SLOTS {','.join(str(x) for x in _base)} -> "
                f"{','.join(str(x) for x in _new)}, neu geloest (Runde {_round + 1}; der Planer "
                f"senkt FR_D, die Zeilen gehen von resident nach Scratch)")
            ns._d_kv_stage_lru_round = _round + 1
            # the map pass restores env_d; the raised scratch is part of the
            # form the map pins (FR_D was solved with it), so it outlives it
            ns._d_kv_stage_lru_scratch = list(_new)
            try:
                return log_d_rank_vram_solve(ns, cards, budgets_d, log, label, p_split=p_split,
                                             chunk_layers=chunk_layers, card_terms=card_terms)
            finally:
                ns._d_kv_stage_lru_round = _round
    if _ledger is not None and plan.fits:
        # rc12e: D's extend trims the allocator cache when the card holds
        # less than floor + booked activation (PDFLIP-EXTEND-CACHE-TRIM).
        _growth, _growth_src = d_extend_growth_record(ns.profile)
        _trim = d_extend_trim_env(_ledger, plan.fits, _growth)
        if _trim:
            ns.env_d = set_group_env(getattr(ns, "env_d", "") or "",
                                     "FLLIPER_PDFLIP_EXTEND_TRIM_MIB", _trim)
            _gtext = (f"floor + max(gebuchte Aktivierung, gemessener Extend-Zuwachs "
                      f"D_EXTEND_GROWTH_MIB {_growth} aus {_growth_src}) je Rang"
                      if _growth is not None else "floor + gebuchte Aktivierung je Rang")
            log(f"{D_RANK_SOLVE_MARKER} {label} EXTEND-TRIM FLLIPER_PDFLIP_EXTEND_TRIM_MIB={_trim} "
                f"({_gtext}; darunter leert D vor dem Extend den Allokator-Cache)")
        # D-TRANSIENT-LEND: the near-OOM edge per rank, down to which D lends
        # its booked transient as expert rows between extends; a value named
        # in --env-d wins
        _env_d = getattr(ns, "env_d", "") or ""
        if "FLLIPER_PDFLIP_D_LEND_FLOOR_MIB" not in parse_group_env(_env_d):
            _lend = d_lend_floor_env(_ledger)
            ns.env_d = set_group_env(_env_d, "FLLIPER_PDFLIP_D_LEND_FLOOR_MIB", _lend)
            log(f"{D_RANK_SOLVE_MARKER} {label} TRANSIENT-LEND FLLIPER_PDFLIP_D_LEND_FLOOR_MIB={_lend} "
                f"(Karten-Boden je Rang; zwischen zwei Extends werden die Bytes darueber "
                f"Expertenzeilen, vor dem Extend zurueckgegeben)")
        # rc12g: the extend chunk follows the card after the trim
        # (PDFLIP-EXTEND-STUECKELUNG); a value named in --env-d wins.
        _rate, _rate_src = d_extend_growth_per_row_record(ns.profile)
        if _rate is not None and len(_rate) == len(_ledger.floor_mib):
            from flliper.srt.pdflip import extend_trim as _et

            # NF port 1003 (order 980 item 2): a rank the record leaves null gets a
            # DERIVED start rate (and the run-time measurement) instead of no vote
            _rate_rec = list(_rate)
            _rate, _fill = d_extend_rate_fill_null(ns, _rate)
            _rtext = _et.launcher_rates(_rate)
            _env_d = getattr(ns, "env_d", "") or ""
            # a value that differs from the record's came from the user (a second
            # solve pass sees its own write, which is not a user value)
            _given = any(x.split("=", 1)[0].strip() == "FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB"
                         and x.split("=", 1)[1].strip() != _rtext
                         for x in _env_d.split(";") if "=" in x)
            if _rtext and not _given:
                ns.env_d = set_group_env(_env_d, "FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB", _rtext)
            if _rtext:
                log(f"{D_RANK_SOLVE_MARKER} {label} EXTEND-STUECKELUNG rows_cap from card_free_post: "
                    f"FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB={_rtext}"
                    f"{' (aus --env-d, Vorrang)' if _given else ''} "
                    f"(Rate aus D_EXTEND_GROWTH_PER_ROW_MIB {_rate_rec}, boots {_rate_src}; D kappt den "
                    f"Extend-Chunk auf floor((card_free_post - 300) / Rate), MIN ueber TP)")
            if _fill is not None and _rtext and not _given:
                d_extend_rate_measure_ranks_env(ns, log, label, _fill, _rtext)
    _w = getattr(ns, "_d_kv_stage_written", None) or {}
    _stage_rows = dict((_w.get("worker_rows") or {}))
    if _w:
        _stage_rows[int(_w.get("host", 0))] = int(_w.get("rows", 0))
    vram_view().note_d_solve(label, cards, plan.fits,
                             objective=str(getattr(ns, "d_tp_objective", "") or ""),
                             kv_tokens_max=int(_d_kv), seats=d_stated_seats(ns),
                             stage_rows=_stage_rows, fixed_src=_fixed_src, act_src=_act_src)
    vram_plan_emit(ns, cards, _VRAM_PASS_BY_LABEL.get(label, "d_early"), log, d_label=label)
    log_wake_credit_solve(ns, cards, plan.fits, log, label,
                          p_split=p_split, chunk_layers=chunk_layers)


#: H14: die Tag-Tabelle des ersten Wakes fuer ``front --wake-credit-plan``,
#: gesetzt von :func:`log_wake_credit_solve` (echter Lauf und Dry-Run), gelesen
#: von ``main`` beim Bau des Front-argv. ``None`` = der Riegel entfiel.
_WAKE_CREDIT_FRONT_PLAN: Optional[Dict[str, object]] = None


def log_wake_credit_solve(ns, cards: List[Card], fits, log, label: str, *,
                          p_split=None, chunk_layers=None) -> None:
    """H14: der Kreditbedarf des ersten Wakes (D schlaeft, P wacht), je Karte.

    fnFL2x114c/x114d starben am ersten Wake an W109: der co-lokierte
    Schlaefer gibt den Tag seines Wakers erst frei, nachdem er einen Tag an
    einen Waker auf einer ANDEREN Karte deponiert hat, dessen Kredit am
    Schlaefer DORT haengt. Der D-FRACTION-SOLVE (H8) und P's Puffer (H5)
    passten -- nur der Weg nicht. Dieser Riegel rechnet den Weg: die gemessene
    Referenz (``wake_credit.REFERENCE_FNFL2X114D`` oder
    ``--wake-credit-reference-logs``) plus die Pufferregel als Delta, gegen die
    Gates von ``VramCredit.wait_for``; endet der Wake -- auch nach der
    Kredit-Ordnung der Front (FLLIPER_PDFLIP_FLIP_ORDER_CREDIT) -- im Zyklus,
    VERWEIGERT er (W126) mit der Arithmetik je Karte, bevor ein Rang laedt.
    Eine unvollstaendige Geometrie entfaellt mit Namen, nie eine halbe Zahl.
    """
    global _WAKE_CREDIT_FRONT_PLAN
    from flliper.srt.planner import expert_residency as _er
    from flliper.srt.pdflip import wake_credit as _wc

    gaps: List[str] = []
    if not p_split:
        gaps.append("P-Schnitt (flip_order_split)")
    if not chunk_layers:
        gaps.append("chunk_layers")
    if not fits:
        gaps.append("D-FRACTION-SOLVE (H8)")
    _fp_text = str(getattr(ns, "pp_cut_expert_device_fraction", "") or "")
    fr_pp = (_csv_floats(_fp_text) if _fp_text else
             [float(x) for x in (_argv_vector(getattr(ns, "extra_p", ""),
                                              "--rank-moe-resident-fraction") or [])])
    env_p = parse_group_env(getattr(ns, "env_p", "") or "")
    # P's scratch rows: the group env the ranks read, else the PP-CUT's own
    # --pp-cut-expert-lru-rows (the same number the #140 FRACTION-SOLVE uses)
    _s = (str(env_p.get("FLLIPER_MOE_SCRATCH_SLOTS", "")).strip()
          or str(getattr(ns, "pp_cut_expert_lru_rows", "") or "").strip())
    scratch_p = _csv_floats(_s) if _s else []
    if p_split and len(scratch_p) == 1:
        scratch_p = scratch_p * len(p_split)
    if not p_split or len(fr_pp) != len(p_split):
        gaps.append("P-Fractions (--pp-cut-expert-device-fraction / extra_p)")
    if not p_split or len(scratch_p) != len(p_split):
        gaps.append("FLLIPER_MOE_SCRATCH_SLOTS (env_p)")
    if gaps:
        log(f"{_wc.MARKER} {label} ENTFAELLT: {', '.join(gaps)} fehlt -- "
            f"ohne vollstaendige Geometrie rechnet der Riegel nichts.")
        _WAKE_CREDIT_FRONT_PLAN = None
        return
    ref = None
    _logs = [p.strip() for p in str(getattr(ns, "wake_credit_reference_logs", "") or "").split(",")
             if p.strip()]
    try:
        if _logs:
            if len(_logs) != 3:
                raise ValueError("--wake-credit-reference-logs requires P.log,D.log,front.log")
            texts = []
            for p in _logs:
                with open(p, errors="replace") as fh:
                    texts.append(fh.read())
            ref = _wc.reference_from_logs(*texts, source=",".join(os.path.basename(p) for p in _logs),
                                          p_card=[c.nvml_index for c in cards])
        num_experts = int(sum(int(f.span) for f in fits))
        p_rows = [_er.buffer_rows(local_experts=num_experts, fraction=float(f),
                                  scratch_rows=int(s)) for f, s in zip(fr_pp, scratch_p)]
        d_rows = [int(f.buffer_rows) for f in fits]
        ratios = _argv_vector(getattr(ns, "extra_d", ""), "--rank-moe-ratio") or []
        live = str(env_p.get("FLLIPER_PDFLIP_CREDIT_LIVE_STAGING", "")).strip().lower()
        double = (live in ("0", "false", "no", "off")) if live else (
            not envs.FLLIPER_PDFLIP_CREDIT_LIVE_STAGING.get())
        wplan = _wc.plan_wake_credit(
            model=ns.model, p_split=p_split, chunk_layers=int(chunk_layers),
            n_layers=int(fits[0].n_layers), p_card=[c.nvml_index for c in cards],
            d_ratio=",".join(str(x) for x in ratios), p_rows=p_rows, d_rows=d_rows,
            slot_mib=float(fits[0].slot_mib), label=label,
            reorder=bool(envs.FLLIPER_PDFLIP_FLIP_ORDER_CREDIT.get()), double_staging=double,
            # H54: dieselbe Suche, die die Front faehrt (die Tabelle traegt sie mit)
            search=bool(envs.FLLIPER_PDFLIP_ENABLE_FLIP_ORDER_CREDIT_SEARCH.get()),
            reference=ref,
            # H59: der Schnitt, bei dem die REFERENZ gemessen ist -- nicht der
            # geplante. Bis hier stand p_split des geplanten Boots im Schluessel,
            # die Pruefung "fremde Geometrie" war damit immer erfuellt, und ein
            # anderer Schnitt (26,10,12 gegen die x162-Referenz 29,11,8) rechnete
            # still mit den P-Tags der alten Karten (weights_12 auf PP1 statt
            # PP2). Unbekannt (Log ohne Layerzeilen) -> ENTFAELLT mit Namen.
            reference_key=None if ref is None else {
                # H87: DAS MODELL DER REFERENZ-LOGS. Bis hier fehlte es im
                # Schluessel: die NVFP4-Profile rechneten ihren ersten Wake gegen
                # die INT4-Logs fnFL2x162 (Tags, Zeilen, Freigaben eines anderen
                # Checkpoints), ohne dass ein Vergleich es sah. Verglichen wird
                # nach Speicher-Fussabdruck (wake_credit._model_same_footprint),
                # ein Log ohne Modellnennung ENTFAELLT mit Namen.
                "model": (pdflip_form.log_model(_logs[2]) or pdflip_form.log_model(_logs[0])
                          or "ungemessen: %s nennt kein Modell" % ref.source),
                "p_card": tuple(ref.p_card),
                "p_split": _wc.reference_p_split(ref, int(fits[0].n_layers))
                or ("ungemessen: %s ohne 'MoE expert-offload active on layer'-Zeilen"
                    % ref.source),
                "chunk_layers": int(chunk_layers)},
        )
    except (OSError, KeyError, ValueError) as _exc:
        log(f"{_wc.MARKER} {label} failed: {type(_exc).__name__}: {_exc}")
        _WAKE_CREDIT_FRONT_PLAN = None
        return
    for line in wplan.lines:
        log(line)
    _WAKE_CREDIT_FRONT_PLAN = wplan.front_plan
    # H34: the other direction, P->D (P sleeps, D wakes), timed per card --
    # printed BEFORE either riegel refuses, so the dry run shows both walls.
    pd_refusal = log_wake_credit_solve_pd(
        ns, cards, log, label, p_split=p_split, chunk_layers=int(chunk_layers),
        n_layers=int(fits[0].n_layers), ratios=ratios, p_rows=p_rows, d_rows=d_rows,
        slot_mib=float(fits[0].slot_mib),
        p_resident=[_er.resident_rows(num_experts, float(f)) for f in fr_pp],
        d_resident=[int(f.resident_rows) for f in fits],
        # #239 S3h2: under the KV token cut the kv_cache tag per rank (KV pool
        # from the SOLVED cut + the mamba state pool); None without the cut.
        kv_first_mib=([float(f.kv_mib) + float(f.mamba_mib) for f in fits]
                      if all(float(getattr(f, "kv_token_share", -1.0)) >= 0.0 for f in fits)
                      else None),
        kv_cut_shares=([float(f.kv_token_share) for f in fits]
                       if all(float(getattr(f, "kv_token_share", -1.0)) >= 0.0 for f in fits)
                       else None))
    if wplan.refusal is not None:
        log(f"{_wc.MARKER} {wplan.refusal}")
        refusals.refuse_value("WAKE-CREDIT", wplan.refusal, PdFlipLaunchRefused, log)
    if pd_refusal is not None:
        refusals.refuse_value("WAKE-CREDIT", pd_refusal, PdFlipLaunchRefused, log)


def log_wake_credit_solve_pd(ns, cards: List[Card], log, label: str, *, p_split,
                             chunk_layers: int, n_layers: int, ratios, p_rows, d_rows,
                             slot_mib: float, p_resident=None,
                             d_resident=None, kv_first_mib=None,
                             kv_cut_shares=None) -> Optional[str]:
    """H34: der Wake P->D der geplanten Form, je Karte in Millisekunden.

    fnFL2x141: im P->D-Wake wartet D TP2 (nvml2) 73-94 ms an weights_4 und
    427-472 ms an weights_6 auf Kredit; H14 rechnet nur D->P. Dieser Riegel
    rechnet die gemessene Referenz derselben Form (Draft auf P: fnFL2x141,
    H25: fnFL2x144, ``wake_credit_pd_refs``) plus die Pufferregel als Delta:
    welche P-Freigabe auf welcher Karte wann kommt, welcher D-Tag wann
    Kredit bekommt, engste Luft, Kreditwarten und was es das Leg kostet;
    dazu die Empfehlung einer Band-Umstellung der knappsten Quelle (die Front
    faehrt sie nur mit FLLIPER_PDFLIP_ENABLE_PD_TIMED_ORDER=1). Rueckgabe: die
    Verweigerung (W126, Kreditmangel P->D) oder None; die Front-Tabelle
    haengt an ``_WAKE_CREDIT_FRONT_PLAN``."""
    global _WAKE_CREDIT_FRONT_PLAN
    from flliper.srt.planner import expert_residency as _er
    from flliper.srt.pdflip import wake_credit_pd as _wpd

    # H92d: free beim Flip-Start aus der juengsten Messung derselben Form UND
    # derselben D-Sitzzahl (der schlafende D-Mitbewohner waechst mit den
    # Sitzen: x178 1 Sitz 1698/768/766 MiB, bb2 6 Sitze 1944/872/874) --
    # Sidecar-Records vor den eingebauten; eine Form ohne genannte Sitze
    # (handgebaute Namespaces) rechnet wie vorher.
    _d_seats = d_stated_seats(ns)
    try:
        pplan = _wpd.plan_wake_credit_pd(
            d_seats=_d_seats,
            free0_records=(_wpd.read_free0_records(measured_record_path())
                           if _d_seats is not None else ()),
            model=ns.model, p_split=p_split, chunk_layers=int(chunk_layers),
            n_layers=int(n_layers), p_card=[c.nvml_index for c in cards],
            d_ratio=",".join(str(x) for x in ratios),
            draft_on_p=bool(envs.FLLIPER_PDFLIP_DRAFT_ON_P.get()),
            p_rows=p_rows, d_rows=d_rows, slot_mib=float(slot_mib), label=label,
            apply=bool(envs.FLLIPER_PDFLIP_ENABLE_PD_TIMED_ORDER.get()),
            p_resident=p_resident, d_resident=d_resident,
            # H50: der Baum-Zustand H39 der D-Gruppe waehlt die Referenz mit
            # (dieselbe Env, aus der der D-FRACTION-SOLVE seine liest).
            dense_repack=_er.dense_repack_outside_pool(
                parse_group_env(getattr(ns, "env_d", "") or "")),
            # #242r: die Geometrie des PP-Cut-Solves fuer den Umschnitt
            recut_geometry=getattr(ns, "_pcut_recut_geometry", None),
            # #239 S3h2: the KV-first verdict per rank under the token cut
            kv_first_mib=kv_first_mib, kv_cut_shares=kv_cut_shares,
        )
    except (KeyError, ValueError, IndexError) as _exc:
        log(f"{_wpd.MARKER} {_wpd.DIRECTION} {label} failed: {type(_exc).__name__}: {_exc}")
        return None
    for line in pplan.lines:
        log(line)
    if pplan.front_plan:
        _WAKE_CREDIT_FRONT_PLAN = dict(_WAKE_CREDIT_FRONT_PLAN or {}, **pplan.front_plan)
    if pplan.refusal is None:
        return None
    log(f"{_wpd.MARKER} {pplan.refusal}")
    if not envs.FLLIPER_PDFLIP_ENABLE_PD_CREDIT_REFUSAL.get():
        log(f"{_wpd.MARKER} {_wpd.DIRECTION} {label}: Verweigerung abgeschaltet "
            f"(FLLIPER_PDFLIP_ENABLE_PD_CREDIT_REFUSAL=0) -- der Boot geht weiter")
        return None
    return pplan.refusal














#: The map may move the store after the first pricing (the D form is pinned
#: for the map, #239 rc12z29c); a move above this re-prices the ledger.
COLD_TIER_REPRICE_EPS_GIB = 0.01




























def p_card_verdict(ns, cards, log, *, model: str, chunk_tokens: int,
                   fracs: Sequence[float], lru_rows: Sequence[int],
                   stage_layers: Sequence[int], kv_mib, num_experts: int,
                   row_mib: float, mamba_slots: int = 0,
                   mamba_mib_per_slot: Sequence[float] = (),
                   layer_kinds: Optional[Sequence[str]] = None,
                   dense_mib_per_layer: Optional[float] = None,
                   mamba_mib_per_linear_layer_per_slot: Optional[float] = None,
                   ) -> Optional[List[float]]:
    """H41: ``PP-CUT ACTIVATION`` (T_s(chunk) je Stufe mit Quelle) und
    ``PP-CUT P-KARTE`` (Kopfraum je Stufe im Chunk-Forward); W132, wenn eine
    Stufe unter near-OOM faellt, W131 fuer einen ungemessenen Chunk.

    YaRN x2 (28.09.): FR_P ist die OBERGRENZE, nicht der exakte Wert. Reisst
    eine Stufe die Karte, kappt der Launcher NUR diese Stufe auf ihre Karten-
    Decke (``p_card_fr_cap``), rechnet die Karte mit der Kappung noch einmal
    und schreibt sie an die drei Stellen, an denen FR_P lebt (wie der
    H25-Post). Rueckgabe: die gekappten Fractions, None = unveraendert.
    Verweigert bleibt W132, wenn eine Stufe keine Decke hat (nicht einmal ein
    residenter Experte plus Scratch) oder die Karte auch gekappt reisst.

    Eine Referenz, die nicht zu DIESER Form passt (anderes Modell, anderer
    Schnitt, fehlender KV-Preis oder Draft-Posten), laesst die Karten-Bilanz
    MIT NAMEN entfallen -- verweigert wird nur aus einer GERECHNETEN Bilanz.
    """
    from flliper.srt.pdflip import draft_post as _dp

    # #242: max(builtin, record) per stage -- see p_transient_support(card=True).
    support = p_transient_support(getattr(ns, "profile", None) or None, card=True)
    name = os.path.basename(os.path.normpath(str(model)))
    # H87: a footprint-identical derivative carries the support's measurement.
    if name != support.model and not pdflip_form.reference_model_verdict(
            str(model), support.model)[0]:
        log(f"{_p_card.CARD_MARKER} ENTFAELLT: die Chunk-Transiente ist auf "
            f"{support.model} gemessen, dieser Boot faehrt {name}.")
        return
    try:
        for line in _p_card.activation_lines(support, int(chunk_tokens)):
            log(line)
    except _p_card.PChunkUnmeasured as exc:
        log(f"{_p_card.ACTIVATION_MARKER} {exc}")
        raise PdFlipLaunchRefused(str(exc)) from None
    ref_logs = [x.strip() for x in str(getattr(ns, "p_card_reference_logs", "") or "").split(",")
                if x.strip()]
    # H59: der Mitbewohner (schlafender D-Rang derselben Karte) -- eingebauter
    # gemessener Hochstand, aufgefrischt um die D-Logs neben den Referenz-Logs.
    co_tenant = _p_card.P_CARD_CO_TENANT_FNFL2
    try:
        if ref_logs:
            boots = []
            d_logs = {}
            for path in ref_logs:
                with open(path, errors="replace") as fh:
                    boots.append((os.path.basename(path), fh.read()))
                d_path = path[: -len(".P.log")] + ".D.log" if path.endswith(".P.log") else ""
                if d_path and os.path.isfile(d_path):
                    with open(d_path, errors="replace") as fh:
                        d_logs[os.path.basename(path)] = fh.read()
            over = []
            for path in [x.strip() for x in str(
                    getattr(ns, "p_card_over_logs", "") or "").split(",") if x.strip()]:
                with open(path, errors="replace") as fh:
                    over.append((os.path.basename(path), fh.read()))
            # #242b: Logs der Mehr-Sequenz-Form messen aus den H55-Fenstern
            # (Ein-Sequenz-Chunk-0, Grundstand-Wachstum), sonst mischt die
            # Hochwassermarke die Last in das Wachstum mit der Tiefe.
            reference = _p_card.p_card_reference_from_logs(
                boots, stage_layers=stage_layers, row_mib=row_mib,
                support=support, model=name, over_boots=over, d_logs=d_logs,
                windows=_p_card.logs_carry_seq_windows(boots))
            if d_logs:
                co_tenant = _p_card.merge_co_tenant(
                    co_tenant,
                    _p_card.co_tenant_span_from_logs(
                        sorted(d_logs.items()), n_stages=len(stage_layers)),
                )
        else:
            reference = _p_card.P_CARD_REFERENCE_FNFL2
    except (OSError, ValueError) as exc:
        log(f"{_p_card.CARD_MARKER} ENTFAELLT: reference unreadable: "
            f"{type(exc).__name__}: {exc}")
        return
    # H59 FORM-SCHLUESSEL: die Sitze DIESES Boots = group P's wirksames
    # --max-running-requests (das argparse behaelt: ein --extra-p-Wert schlaegt
    # --p-bs, dieselbe Regel wie p_micro_batch_flags).
    seats = p_seats(ns)
    if kv_mib is None:
        log(f"{_p_card.CARD_MARKER} ENTFAELLT: no KV-price per stage (#156 dropped "
            f"oder --pp-cut-reserve-mib ersetzt ihn); ohne KV ist der Kopfraum "
            f"nicht rechenbar.")
        return
    draft_on_p = str(getattr(ns, "draft_kv_on_p", "off")) == "on"
    draft_mib = 0.0
    if draft_on_p != bool(reference.draft_on_p):
        path = (_argv_scalar(getattr(ns, "extra_p", ""), "--speculative-draft-model-path")
                or _argv_scalar(getattr(ns, "extra_d", ""), "--speculative-draft-model-path")
                or "")
        weights, transient = _dp.p_draft_post_mib(str(path))
        if weights is None:
            log(f"{_p_card.CARD_MARKER} ENTFAELLT: Draft on P "
                f"{'an' if draft_on_p else 'aus'}, die Referenz ({reference.source}) "
                f"{'an' if reference.draft_on_p else 'aus'}, und der Draft-Posten "
                f"ist nicht lesbar ({path!r}).")
            return
        draft_mib = float(weights) + float(transient)
    # H41c/H41d: der Riegel rechnet am LETZTEN Chunk des Prompts, Default der
    # volle Kontext (262144, NF-Pflicht). Das Chunk-Wachstum SAETTIGT
    # (fnFL2x160: nach Chunk 3, gemessen ueber 16 Chunks), also ist die
    # 262k-Kante die 97k-Kante. LMEM des Run-Writes: 0 mit dem H47-Fix.
    prompt = int(getattr(ns, "p_card_prompt_tokens", 0) or 0) or group_context_tokens(ns, "p")[0]
    lmem_fixed = _p_card.arena_write_lmem_fixed()
    # #242r: ein anderer Schnitt als der gemessene wird UMGERECHNET (K0 je
    # Stufe um Dense und Mamba der wandernden Layer); geht das nicht, VERWEIGERT
    # die Karte mit Namen (W167) statt still zu entfallen.
    recut_from = None
    if tuple(int(x) for x in stage_layers) != tuple(reference.stage_layers):
        recut_from = tuple(reference.stage_layers)
        try:
            if len(stage_layers) != len(reference.stage_layers):
                reference, support, co_tenant = _restage_p_card_reference(
                    ns, cards, reference, support, co_tenant, stage_layers, log,
                    layer_kinds=layer_kinds, dense_mib_per_layer=dense_mib_per_layer,
                    mamba_mib_per_linear_layer_per_slot=mamba_mib_per_linear_layer_per_slot)
            else:
                reference = _p_card.recut_reference(
                    reference, stage_layers, layer_kinds=layer_kinds,
                    dense_mib_per_layer=dense_mib_per_layer,
                    mamba_mib_per_linear_layer_per_slot=mamba_mib_per_linear_layer_per_slot)
        except _p_card.PCutRecutRefused as exc:
            log(f"{_p_card.CARD_MARKER} {exc}")
            raise PdFlipLaunchRefused(str(exc)) from None

    def _solve(tokens, fractions=fracs):
        return _p_card.solve_p_card(
            reference=reference, support=support, fractions=fractions,
            lru_rows=lru_rows, stage_layers=stage_layers, chunk=int(chunk_tokens),
            kv_mib=[float(x) for x in kv_mib], num_experts=int(num_experts),
            row_mib=float(row_mib), draft_on_p=draft_on_p,
            draft_mib_last_stage=draft_mib,
            cards=[_dp.stage_card_label(cards, s) for s in range(len(stage_layers))],
            near_oom_mib=float(corridor_guard.NEAR_OOM_MIB),
            prompt_tokens=tokens, lmem_fixed=lmem_fixed,
            seats=seats, co_tenant=co_tenant,
            mamba_slots=int(mamba_slots), mamba_mib_per_slot=tuple(mamba_mib_per_slot),
        )

    try:
        fits = _solve(prompt)
    except _p_card.PChunkUnmeasured as exc:
        raise PdFlipLaunchRefused(str(exc)) from None
    except ValueError as exc:
        log(f"{_p_card.CARD_MARKER} ENTFAELLT: {exc}")
        return
    for fit in fits:
        log(_p_card.describe_p_card(fit, reference))
    vram_view().note_p_card(cards, fits, fracs, lru_rows, reference.source, num_experts=int(num_experts))
    if recut_from is not None:
        log(_p_card.recut_card_line(
            recut_from, stage_layers, fits, dense_mib_per_layer=float(dense_mib_per_layer),
            mamba_mib_per_linear_layer=float(mamba_mib_per_linear_layer_per_slot or 0.0)
            * int(reference.mamba_slots)))
    refusal = _p_card.p_card_refusal_text(fits, reference, chunk=int(chunk_tokens))
    if refusal is None:
        return None
    cap = p_card_fr_cap(fits, fracs, int(num_experts), lru_rows)
    if cap.refusal:
        refusal = "%s -- %s" % (refusal, cap.refusal)
        log(f"{_p_card.CARD_MARKER} {refusal}")
        refusals.refuse_value("P-CARD", refusal, PdFlipLaunchRefused, log)
        return None                     # --force: no cap applied, the fractions stay as given
    capped = _solve(prompt, cap.fractions)
    for fit in capped:
        log(_p_card.describe_p_card(fit, reference))
    vram_view().note_p_card(cards, capped, cap.fractions, lru_rows, reference.source, num_experts=int(num_experts))
    again = _p_card.p_card_refusal_text(capped, reference, chunk=int(chunk_tokens))
    if again is not None:
        again = "%s -- auch nach %s %s" % (again, P_FR_CAP_MARKER, "; ".join(cap.lines))
        log(f"{_p_card.CARD_MARKER} {again}")
        refusals.refuse_value("P-CARD", again, PdFlipLaunchRefused, log)
    publish_p_fractions(ns, cap.fractions)
    ns._w132_fr_cap = {
        "fr_p_before": [float(x) for x in fracs],
        "fr_p_after": [float(x) for x in cap.fractions],
        "stages": list(cap.stages),
        "lines": list(cap.lines),
        "chunk": int(chunk_tokens),
        "prompt": int(capped[0].prompt_tokens) if capped else int(prompt),
    }
    for line in cap.lines:
        log(f"{P_FR_CAP_MARKER} {line}")
    log(f"{P_FR_CAP_MARKER} FR_P {[round(float(x), 4) for x in fracs]} -> "
        f"{ns.pp_cut_expert_device_fraction} (published to --pp-cut-expert-device-"
        f"fraction, --extra-p --rank-moe-resident-fraction, --env-p "
        f"FLLIPER_MOE_RESIDENT_EXPERT_FRACTION; die uebrigen Stufen exakt wie gegeben)")
    return list(cap.fractions)


#: YaRN x2 (28.09.): die Zeile der Kappung. Bewusst 'W132-...' mit Bindestrich:
#: der Arm zaehlt nach dem Dry-Run ``W[0-9]+ `` als Verweigerung
#: (test_no_h57_line_is_counted_as_a_w_line_by_the_arm) -- eine Kappung ist
#: keine.
P_FR_CAP_MARKER = "PP-CUT W132-FR_P-KAPPUNG"












def _drop_cut_pins(ns) -> List[str]:
    """Clear the P-cut pins of ``ns`` (``--pp-stage-ratio`` / ``--pp-attn-stage-ratio``,
    also inside ``--extra-p/-d`` words); the texts dropped, for the log."""
    defaults = build_parser().parse_args(["--tree", "/", "--tag", "x"])
    dropped: List[str] = []
    for dest in ("pp_stage_ratio", "pp_attn_stage_ratio"):
        v = getattr(ns, dest, None)
        if v not in (None, ""):
            dropped.append(f"--{dest.replace('_', '-')} {v}")
            setattr(ns, dest, getattr(defaults, dest, None))
    for key in ("extra_p", "extra_d"):
        blob = str(getattr(ns, key, "") or "")
        try:
            words = shlex.split(blob) if blob else []
        except ValueError:
            continue
        out: List[str] = []
        i = 0
        hit = False
        while i < len(words):
            if words[i] in ("--pp-stage-ratio", "--pp-attn-stage-ratio") and i + 1 < len(words):
                dropped.append(f"--{key.replace('_', '-')}: {words[i]} {words[i + 1]}")
                hit = True
                i += 2
                continue
            out.append(words[i])
            i += 1
        if hit:
            setattr(ns, key, " ".join(shlex.quote(x) for x in out))
    return dropped


def _unpin_foreign_cut(solve):
    """AP1 1006 (d), as a decorator so the call sites (and the tests that pin their text)
    stay as they are: when the cut the profile PINS (NF ``--pp-stage-ratio 29,11,8``) is
    priced infeasible by the planner on a FOREIGN inventory (the live cards are not the
    ones the pin was solved for: 3 x RTX 3090 puts rank 0 1897 MiB over its budget, 1520),
    a FORCED boot drops the pin and lets the cut solver derive the cut from the live
    budgets -- the planner's verdict decides, not a second table. Printed and remembered
    as the value refusal HW-UNCALIBRATED (FORCED-PAST). Everything else is unchanged:
    without ``--force``, on the calibrated inventory, with an inventory the operator named
    (``--profile-inventory``), without a pin, or when the solver finds no cut either, the
    planner's refusal propagates as before."""

    @functools.wraps(solve)
    def wrapper(ns, cards, budgets_p, model, log, *args, **kwargs):
        from flliper.srt.planner import pp_cut_launch as _cut

        try:
            return solve(ns, cards, budgets_p, model, log, *args, **kwargs)
        except _cut.PPCutRefused as exc:
            pinned = any(str(getattr(ns, d, "") or "") for d in ("pp_stage_ratio", "pp_attn_stage_ratio"))
            if not (pinned and refusals.forced_boot() and derivation_assessment(ns, cards) is not None):
                raise
            prof = str(getattr(ns, "profile", None) or pdflip_form.DEFAULT_PROFILE)
            cal, _ = records_inventory(prof)
            live = card_identity.inventory_signature(cards)
            dropped = _drop_cut_pins(ns)
            _refuse_value_once(
                card_identity.CODE_UNCALIBRATED,
                f"{card_identity.CODE_UNCALIBRATED}: the P cut the profile pins ({'; '.join(dropped)}) was solved "
                f"for [{', '.join(cal)}]; on [{', '.join(live)}] the planner prices it infeasible "
                f"({' '.join(str(exc).split())[:260]}) -> the pin is DROPPED, the cut solver (planner/pp_cut.py) "
                "derives the cut from the live budgets and the (borrowed, UNMEASURED) stage costs.",
                log=log)
            return solve(ns, cards, budgets_p, model, log, *args, **kwargs)

    return wrapper


@_unpin_foreign_cut
def solve_p_cut(
    ns,
    cards: List[Card],
    budgets_p: List[int],
    model: str,
    log,
    chunk_tokens: int = 4096,
    user_reserve_by_card: Optional[Dict[str, int]] = None,
    power_reading: Optional["_power.PowerReading"] = None,
) -> PCutFacts:
    """Group P's layer + attention cut, and the ONE provenance line for it.

    Returns the flag strings plus the pool facts the depth solver prices
    against (:class:`PCutFacts`). Everything it feeds the solver is either measured on this box
    (NVML card names, the per-rank budgets this launcher just derived, the
    checkpoint's own weight headers and KV cell) or a flag whose default
    carries its provenance in the help text -- no number is chosen here.
    """
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
    if terms.expert_layer_weight_bytes > 0.0:
        frac_text = str(getattr(ns, "pp_cut_expert_device_fraction", "") or "").strip()
        rows_text = str(getattr(ns, "pp_cut_expert_lru_rows", "") or "").strip()
        n_stages_p = len(budgets_p)
        if not frac_text:
            raise SystemExit(
                f"W40 PdFlipPPCutRefused: {model} carries {terms.num_experts} MoE "
                f"experts per layer ({terms.expert_layer_weight_bytes / _pp_cut.MIB:.0f} "
                f"MiB per layer) that live in a device pool, and no "
                f"--pp-cut-expert-device-fraction was given for the {n_stages_p} P "
                f"stages. Pass the resident fraction per stage (the P group's "
                f"--rank-moe-resident-fraction) and --pp-cut-expert-lru-rows."
            )
        fracs = _csv_floats(frac_text)
        rows = _csv_floats(rows_text) if rows_text else [0.0] * n_stages_p
        if len(fracs) != n_stages_p or len(rows) != n_stages_p:
            raise SystemExit(
                f"--pp-cut-expert-device-fraction has {len(fracs)} entries and "
                f"--pp-cut-expert-lru-rows {len(rows)}; the P group has {n_stages_p} stages."
            )
        row_bytes = terms.expert_layer_weight_bytes / max(1, terms.num_experts)
        layer_mib_by_stage = tuple(
            mean_layer_mib
            + (terms.expert_layer_weight_bytes * float(f) + row_bytes * float(r)) / _pp_cut.MIB
            for f, r in zip(fracs, rows)
        )
        # DIE LAYER JE STUFE, und sie wird NICHT geraten: --pp-stage-ratio ist
        # ein SCORE-Vektor (Memory rank-ratios-sind-verhaeltnis), der nur dann
        # die Layerzahlen IST, wenn seine Summe die Layerzahl des Checkpoints
        # trifft. Fuer Next Flash tut sie das (29+11+8 = 48). Sonst wird
        # proportional skaliert -- derselbe `scaled_spans`-Gedanke wie bei den
        # Experten-Baendern, mit dem Rest auf der letzten Stufe, damit keine
        # Layer zwischen zwei Stufen verschwindet.
        _ratio_p = _csv_floats(str(getattr(ns, "pp_stage_ratio", "") or "")) or []
        if len(_ratio_p) == n_stages_p and sum(_ratio_p) > 0:
            if int(round(sum(_ratio_p))) == int(terms.n_layers):
                _stage_layers_for_solve = [int(round(x)) for x in _ratio_p]
            else:
                _acc = [int(round(x * terms.n_layers / sum(_ratio_p)))
                        for x in _ratio_p[:-1]]
                _stage_layers_for_solve = _acc + [int(terms.n_layers) - sum(_acc)]
        else:
            _stage_layers_for_solve = [
                int(terms.n_layers) // max(1, n_stages_p)] * n_stages_p

        # H25 (Nutzer-Order 24.09.): "Draft auf P streichen. dadurch koennen
        # auch mehr experten im P layout gehalten werden." The draft's post on
        # P's draft card (the last stage) becomes expert rows of THAT stage
        # before anything is priced, and the raised fraction is written back
        # to all three places FR_P lives (pdflip/draft_post.py).
        ns._expert_row_mib = row_bytes / _pp_cut.MIB
        if str(getattr(ns, "draft_kv_on_p", "on")) == "off":
            fracs = apply_p_draft_post(ns, cards, fracs, _stage_layers_for_solve,
                                       row_bytes / _pp_cut.MIB,
                                       int(terms.num_experts), log)
            layer_mib_by_stage = tuple(
                mean_layer_mib
                + (terms.expert_layer_weight_bytes * float(f) + row_bytes * float(r)) / _pp_cut.MIB
                for f, r in zip(fracs, rows)
            )

        # #140 (Nutzer-Order 22.09.: "der planner muss sie ausspucken"):
        # DIE UMKEHRUNG, neben der gegebenen Zahl. Was hier steht, ist bis
        # heute nur Eingabe gewesen -- welche Fraction die Karte TRAEGT,
        # rechnete niemand, und fnFL2w73 hat den Preis gezeigt: PP2 haelt bei
        # 0.95 487 von 512 Experten, danach bleiben 4,93 GB und der
        # Draft-KV-Produzent stirbt still im C++ (cu_mem_create OOM).
        # Gedruckt wird sie mit ihrer Herleitung; wer sie uebernimmt,
        # entscheidet der Operator -- aber raten muss er nicht mehr.
        # #141: DIE RESERVE ALS EXPLIZITER EINGANG, nicht als Schaetzung.
        #
        # Erster Versuch war, sie aus --pp-attn-stage-ratio x kv_mib x
        # Kontext zu rechnen. Der Launcher PARST dieses Flag aber gar nicht
        # (kein add_argument; es ist ein Flag, das argv_p an den SERVER
        # weiterreicht) -- mein Leser haette ins Leere gegriffen und die
        # Bedingung waere still nie wahr geworden. Genau die Klasse, die
        # heute schon zweimal Boots gekostet hat
        # (Memory riegel-hinter-dem-was-er-sichert).
        #
        # Also kommt die Reserve dorther, wo sie gemessen wird: der Operator
        # gibt sie je Stufe, und der Solver rechnet damit. Ohne Angabe bleibt
        # das Ergebnis die DECKE -- als solche benannt, nicht als Empfehlung
        # getarnt.
        _res_text = str(getattr(ns, "pp_cut_reserve_mib", "") or "").strip()
        _reserve_p = None
        if _res_text:
            _r = _csv_floats(_res_text)
            if len(_r) == n_stages_p:
                _reserve_p = _r
            else:
                log("PP-CUT RESERVE (#141) IGNORED: --pp-cut-reserve-mib has "
                    "%d Eintraege, die P-Gruppe hat %d Stufen"
                    % (len(_r), n_stages_p))

        # #156 KV-PREIS: der Posten, den die Sizing-Kette nie gefuehrt hat.
        #
        # Nutzer-Order 22.09., woertlich: "kv muss bepreist werden fuer 262k
        # und danach geht der rest an moe experten" -- und die Diagnose davor:
        # "ein rang kann nicht 441728 kv bekommen und die anderen viel weniger
        # als 262k, da rechnet was absolut grundfalsch".
        #
        # Er hat recht. In design_bytes-first-lawful.md 1.1 stehen alle Posten,
        # die vom Budget abgezogen werden -- KV ist keiner davon, sondern der
        # REST nach allen anderen. Unter PP haelt jede Stufe KV fuer IHRE Layer
        # und ALLE Token, der effektive Kontext ist also das MINIMUM ueber die
        # Stufen. fnFL2w123 rechnete 441728/172672/121920, servierte 121920 und
        # liess 6,75/3,54/3,48 GB nach dem Sizing brach liegen.
        #
        # Ohne --pp-cut-reserve-mib wird der Preis hier GERECHNET statt mit 0
        # angesetzt. Die Zelle kommt aus der Modell-Geometrie und ist am Metall
        # gegengeprueft (fnFL2w123: 8704/4352/3264 B/Token, exakt).
        _kv_p = None
        if _reserve_p is None:
            try:
                _kv_tokens = int(_argv_scalar(
                    getattr(ns, "extra_p", ""), "--max-total-tokens") or 0)
                _attn_text = str(getattr(ns, "pp_attn_stage_ratio", "") or "")
                _attn = [int(x) for x in _csv_floats(_attn_text)] if _attn_text else []
                _kv_dtype = str(_argv_scalar(
                    getattr(ns, "extra_p", ""), "--kv-cache-dtype") or "auto")
                _kv_bytes = 1.0 if "fp8" in _kv_dtype else 2.0
                # --draft-kv-on-p legt EINEN Draft-Attention-Layer an -- auf der
                # LETZTEN Stufe, wo der Produzent lebt
                # (Scheduler._maybe_init_draft_kv_producer: is_last_rank).
                # w123s 8704 auf Stufe 0 war die Zelle der Rang-Planung, die
                # den Draft bis H42c-2 auf JEDER Stufe einrechnete
                # (pool_configurator.draft_kv_pool_on_this_rank); allokiert
                # wurde er nie dort: x161 'KV Cache is allocated' K 0.88/0.38
                # GB auf PP0/PP1 wie x160 ohne Draft, der Draft-Pool (K 0.13 GB)
                # nur auf PP2. Der Phantom-Posten kostete 272 MiB je
                # Nicht-Letzt-Stufe (P-Kante 0.332 -> 0.324).
                _dr = 1 if str(getattr(ns, "draft_kv_on_p", "off")) == "on" else 0
                _tc = text_cfg
                if _kv_tokens > 0 and len(_attn) == n_stages_p:
                    _kv_p = _pp_cut.kv_reserve_mib_per_stage(
                        tokens=_kv_tokens,
                        attn_layers_by_stage=_attn,
                        kv_heads=int(_tc["num_key_value_heads"]),
                        head_dim=int(_tc["head_dim"]),
                        v_head_dim=int(_tc.get("v_head_dim", _tc["head_dim"])),
                        kv_dtype_bytes=_kv_bytes,
                        draft_attn_layers_by_stage=[0] * (n_stages_p - 1) + [_dr],
                    )
                    _reserve_p = _kv_p
                    log(
                        "PP-CUT KV-PREIS (#156): %d Token x cell per Stage "
                        "(Attention-Layer %s + Draft %d auf der letzten Stufe, %s -> %.0f B je "
                        "Attention-Layer und Token) = %s MiB je Stufe. DAS ist "
                        "der Posten, der in der Kette fehlte; er geht jetzt als "
                        "Reserve in die Fraction-Decke, damit der Rest den "
                        "Experten gehoert und nicht umgekehrt."
                        % (_kv_tokens, _attn, _dr, _kv_dtype,
                           _pp_cut.kv_cell_bytes_per_attention_layer(
                               kv_heads=int(_tc["num_key_value_heads"]),
                               head_dim=int(_tc["head_dim"]),
                               v_head_dim=int(_tc.get("v_head_dim", _tc["head_dim"])),
                               kv_dtype_bytes=_kv_bytes),
                           ["%.0f" % m for m in _kv_p])
                    )
                else:
                    log("PP-CUT KV-PREIS (#156) ENTFAELLT: tokens=%d, "
                        "--pp-attn-stage-ratio %r passt nicht zu %d Stufen. "
                        "Ohne beides ist der Preis nicht rechenbar und die "
                        "Decke bleibt die nackte Obergrenze."
                        % (_kv_tokens, _attn_text, n_stages_p))
            except BaseException as _exc:  # noqa: BLE001
                log("PP-CUT KV-PREIS (#156) failed: %s: %s"
                    % (type(_exc).__name__, _exc))

        try:
            _fmax = _pp_cut.solve_expert_fraction_per_stage(
                budgets_mib=budgets_p,
                stage_layers=_stage_layers_for_solve,
                mean_layer_mib=mean_layer_mib,
                expert_layer_mib=terms.expert_layer_weight_bytes / _pp_cut.MIB,
                num_experts=terms.num_experts,
                lru_rows=rows,
            )
            _frec = _pp_cut.solve_expert_fraction_per_stage(
                budgets_mib=budgets_p,
                stage_layers=_stage_layers_for_solve,
                mean_layer_mib=mean_layer_mib,
                expert_layer_mib=terms.expert_layer_weight_bytes / _pp_cut.MIB,
                num_experts=terms.num_experts,
                lru_rows=rows,
                reserve_mib_by_stage=_reserve_p,
            ) if _reserve_p is not None else None
            _stages_over_cap = [i for i, (f, m) in enumerate(
                zip(fracs, _frec if _frec is not None else _fmax)) if f > m]
            log(
                "PP-CUT FRACTION-SOLVE (#140): budgets %s MiB, layers %s, "
                "dense %.0f + experts %.0f MiB/layer, LRU %s, Puffer "
                "min(ceil(f*E)+LRU, E) (H8, groesste Offload-Fraction "
                "(E-2)/E) -> OBERGRENZE je "
                "Stufe %s (gegeben: %s)%s. Die Obergrenze laesst NICHTS fuer "
                "KV, Draft und Aktivierungen -- sie ist die Decke, nicht die "
                "Empfehlung."
                % (
                    list(budgets_p), list(_stage_layers_for_solve), mean_layer_mib,
                    terms.expert_layer_weight_bytes / _pp_cut.MIB, list(rows),
                    ["%.3f" % f for f in _fmax], ["%.3f" % f for f in fracs],
                    ((" | MIT KV-RESERVE: %s" % ["%.3f" % f for f in _frec])
                     if _frec is not None else "")
                    + ((" -- DARUEBER on stage(s) %s" % _stages_over_cap) if _stages_over_cap else ""),
                )
            )
        except BaseException as _exc:  # noqa: BLE001 -- eine Zahl kippt nie den Boot
            log("PP-CUT FRACTION-SOLVE (#140) failed: %s: %s"
                % (type(_exc).__name__, _exc))
        log(
            "PP-CUT POOL TERM (Task #47/#48): dense %.0f MiB/layer + experts %.0f MiB/layer "
            "(%d experts, %.2f MiB/row) at fractions %s + LRU rows %s -> per-stage %s MiB/layer; "
            "PLE %.0f MiB/layer stays on disk (0 on device)"
            % (
                mean_layer_mib, terms.expert_layer_weight_bytes / _pp_cut.MIB, terms.num_experts,
                row_bytes / _pp_cut.MIB, list(fracs), [int(x) for x in rows],
                [round(x) for x in layer_mib_by_stage], terms.ple_layer_weight_bytes / _pp_cut.MIB,
            )
        )
        # H41 (Task #114): DIE P-KARTE ALS FUNKTION DES CHUNKS. Die Decke oben
        # (#140) und das Pool-Modell unten entscheiden das BUDGET; bei
        # --max-total-tokens 262144 bleibt dessen Rest liegen, und was die
        # KARTE im Chunk-Forward fuellt, steht in keinem Posten -- x121/x122
        # (FR_P[0] 0.45/0.40, Chunk 16384) passierten beides und starben im
        # ersten Chunk. Hier: gemessener Kopfraum je Stufe, verschoben um
        # Puffer, KV, die Chunk-Transiente T_s(chunk) und den Draft.
        # H92c: und um die Mamba-Slots dieses Boots gegen die der Referenz
        # (Slots x 1.5588 MiB x lineare Layer der Stufe).
        # #242r: die Geometrie des Umschnitts, EINMAL hier (Layer-Typen des
        # Checkpoints, Dense je Layer, Mamba je linearem Layer, KV je
        # Attention-Layer) -- P-KARTE und der Wake-Kredit P->D lesen dieselbe.
        _kv_attn = None
        if _kv_p is not None:
            _kv_attn = next((float(k) / int(a) for k, a in zip(_kv_p, _attn) if int(a) > 0), None)
        ns._pcut_recut_geometry = {
            "layer_kinds": [str(k) for k in kinds] if kinds else None,
            "dense_mib_per_layer": float(mean_layer_mib),
            "mamba_mib_per_linear_layer": float(ns.pp_cut_mamba_mib_per_linear_layer_per_slot)
            * int(_p_mamba_slots),
            "kv_mib_per_attn_layer": _kv_attn,
        }
        _capped = p_card_verdict(
            ns, cards, log,
            model=model,
            chunk_tokens=int(chunk_tokens),
            fracs=[float(f) for f in fracs],
            lru_rows=[int(round(float(r))) for r in rows],
            stage_layers=[int(x) for x in _stage_layers_for_solve],
            kv_mib=_kv_p,
            num_experts=int(terms.num_experts),
            row_mib=row_bytes / _pp_cut.MIB,
            mamba_slots=int(_p_mamba_slots),
            mamba_mib_per_slot=p_mamba_mib_per_slot_by_stage(
                kinds, _stage_layers_for_solve,
                float(ns.pp_cut_mamba_mib_per_linear_layer_per_slot)),
            layer_kinds=[str(k) for k in kinds] if kinds else None,
            dense_mib_per_layer=float(mean_layer_mib),
            mamba_mib_per_linear_layer_per_slot=float(ns.pp_cut_mamba_mib_per_linear_layer_per_slot),
        )
        if _capped is not None:
            # W132-FR_P-KAPPUNG: das Pool-Modell unten bepreist dieselbe
            # Fraction, die jetzt im argv steht.
            fracs = list(_capped)
            layer_mib_by_stage = tuple(
                mean_layer_mib
                + (terms.expert_layer_weight_bytes * float(f) + row_bytes * float(r)) / _pp_cut.MIB
                for f, r in zip(fracs, rows)
            )
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
    model_pool = _pp_cut.PhasePoolModel(
        free_mib=tuple(float(b) for b in budgets_p),
        # The FAMILY split of weights is deliberately averaged: a stage's
        # divisor in stage_pp_capacities is its ATTENTION COUNT, and moving
        # one attention layer changes that divisor by 1 in 4-8 (12-25 %)
        # while changing weights by the attn/linear difference of a single
        # layer against a ~17-28 GiB free -- two orders of magnitude apart.
        # The total is exact for any cut summing to n_layers.
        weight_mib_per_layer=mean_layer_mib,
        weight_mib_per_layer_by_stage=layer_mib_by_stage,
        kv_mib_per_token_per_attn_layer=kv_mib,
        arming_floor_mib=tuple(float(ns.pp_cut_arming_floor_mib) for _ in budgets_p),
        # -- #1286: THE POSTS THE BOOT CHARGES AND THIS MODEL DID NOT ------
        # Every one of these is read off the runtime's OWN success-path line,
        # `[world_rank r] KV budget posts (GiB): ...`, so the model's
        # subtraction is that line transcribed rather than re-derived. Before
        # this the model charged the per-LAYER weight half plus a 1229 MiB
        # arming floor and nothing else, and boot weg2sb5f published 499,967
        # tokens for the cut group P then sized at 304,655 (+64.1 %).
        stage_fixed_mib=_stage_fixed,
        # #114: priced from the chunk P boots with (see the log line below),
        # never the reference boots' 1024 on a wider chunk.
        # --p-prefill-graph (27B): the SAME vector the ranks book (env_p, see
        # p_prefill_graph_env); () when off -- the pool model is unchanged.
        prefill_graph_pool_mib=p_prefill_graph_pool_mib(ns),
        # --p-layer-split dynamic: the swing slab per stage; () under static.
        swing_layers_by_stage=p_layer_split_swing_by_stage(len(budgets_p)),
        # --p-attn-head-split: the SAME vector the ranks book (env_p, see
        # p_attn_head_split_env); () when off -- the pool model is unchanged.
        attn_head_split_mib=_p_ah_post_vector(ns, len(budgets_p), chunk_tokens, model),
        activation_reserve_mib=(0.0 if _booked and ns.pp_cut_activation_reserve_mib is None
                                else p_prefill_activation_reserve_mib(
                                    ns.pp_cut_activation_reserve_mib, int(chunk_tokens))),
        # #242: per stage where the profile measured it (P_ACTIVATION_MIB).
        activation_reserve_mib_by_stage=p_prefill_activation_reserve_by_stage_mib(
            ns.pp_cut_activation_reserve_mib, int(chunk_tokens),
            getattr(ns, "profile", None) or None,
        ),
        # #1257c: None = follow the reserve (the runtime charges exactly it);
        # a number = the operator pinned it. PP-POSTEN (27B, p_pool_posts_as_booked):
        # the P budget line already carries the corridor -> 0.
        corridor_holdback_mib=(
            (0.0 if _booked else p_corridor_holdback_default_mib(
                max((user_reserve_by_card or {}).values() or [0])
            ))
            if ns.pp_cut_corridor_holdback_mib is None
            else float(ns.pp_cut_corridor_holdback_mib)
        ),
        mamba_mib_per_linear_layer_per_slot=float(
            ns.pp_cut_mamba_mib_per_linear_layer_per_slot
        ),
        # Read off the argv this launcher builds rather than restated: a
        # second copy of --max-running-requests would drift the day the flag
        # moves, and the mamba residency scales linearly with it.
        # P's OWN bs (C1/R-12): this is P's pool model.
        #
        # #1286: --max-running-requests is the CONCURRENCY TARGET, not the slot
        # count. The demand-driven pool allocates
        # `max(ceil(target * ratio * safety), ratio, mamba_hard_floor)` slots
        # (model_runner_kv_cache_mixin.py:2085-2106, and the boot prints the
        # result: `[auto-mamba] ... target_concurrency=8 ratio=2 safety=1.25 ->
        # max_mamba_cache_size=20 slots`). Charging 8 where the allocator takes
        # 20 under-charged the mamba post by 2.5x on every rank.
        #
        # F4: and charging `--p-bs x 2.5` reproduced THAT boot and no other,
        # because the hard-floor term was absent and the `2` was one branch of
        # `_calculate_mamba_ratio`. The bound below dominates all three terms
        # of the max for any branch (see P_MAMBA_SLOTS_PER_RUNNING_REQUEST),
        # so the model can no longer charge LESS mamba than the boot -- and an
        # over-charge here under-prices the pool, which is the safe way to be
        # wrong.
        #
        # The target is READ OFF P's OWN ARGV, and the branch that would clamp
        # it (`_auto_mamba_target_concurrency` caps an AUTO-DEFAULTED
        # --max-running-requests at MAMBA_AUTO_TARGET_CONCURRENCY = 16) is
        # unreachable here: `_max_running_requests` raises unless the flag is
        # literally on the argv this launcher builds, so P's value is always
        # the user-set branch.
        #
        # H92c: and the demand branch is not the one P takes. argv_p ALWAYS
        # states --max-mamba-cache-size (24, or the arm's --extra-p value), and
        # an explicit value IS the pool: fnFL2h91bb2 was charged 20 slots here
        # and allocated 32 ('Mamba Cache is allocated. max_mamba_cache_size:
        # 32'), x178 10 against 24. p_mamba_slots reads P's argv and keeps the
        # bound above only for an argv without the flag.
        mamba_slots=int(_p_mamba_slots),
        extra_cell_bytes_by_stage=_p_draft_cell,
        # #1286 F7: the sizer's second floor. Read off P's argv for the same
        # reason as the target above -- a second copy would drift the day
        # --page-size moves.
        page_size=_p_page_size(model, int(getattr(ns, "p_bs", DEFAULT_P_BS) or DEFAULT_P_BS)),
        # #1286 F3: the two runtime posts that had no field, funded at zero
        # WITH the claim stated. Both are absent from the emitted post list of
        # weg2sb5f AND weg2rg6:
        #   * `mamba pre-capture reserve` is charged only when
        #     `post_capture_kv_active` (model_runner_kv_cache_mixin.py:864), and
        #     group P sizes its pool BEFORE capture on this form;
        #   * `speculative intermediate state` is the `admitted * D * per_req`
        #     half of `_mamba_pool_budget_cost_gb` (:2107) and is zero at D=0 --
        #     which is also why 1.5588 MiB/linear-layer/slot recovers from
        #     `size * per_req` alone.
        # A form that changes either of those must re-measure the post; the
        # acknowledgement is what makes that a decision rather than a silence.
        zero_posts_acknowledged=(
            "mamba pre-capture reserve",
            "speculative intermediate state",
            "GGUF dequant scratch",
        ) + (("prefill activation reserve",)
             if _booked and ns.pp_cut_activation_reserve_mib is None else ()),
    )
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
    census = read_mean_prefill_prefix(design_src) if design_src else None
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
    # 27B line (user 24.09.): every boot prints its P cards' power limits --
    # the record a later --pp-cut-stage-fit is checked against.
    _power_now = card_power_limits_w(cards)
    log(card_power_line(cards, _power_now))
    # --pp-cut-stage-fit (default off): the stage cost FITTED from a boot's own
    # #PGAP lines replaces the bsscale family split above.
    _fitted_log = None
    if str(getattr(ns, "pp_cut_stage_fit", "") or "").strip():
        family_cost, family_prov, _fitted_log = stage_fit_family_cost(
            ns, cards, int(chunk_tokens), _power_now, log)
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
    log(
        "PP-CUT depth axis: design_prefix=%d tokens (%s); calibration prefix "
        "%d (%s); %s"
        % (
            design_prefix,
            design_prov,
            int(ns.pp_cut_calibration_prefix_tokens),
            "boot bsscale, ~12k prompt = 3 full chunks at 4096 -> mean 4096, "
            "BSSCALE_0907.md",
            family_prov,
        )
    )
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
    # --p-prefill-graph-policy auto: extra buckets only where the pool still
    # clears the floor with their capture pool (no-op without extras).
    model_pool = p_prefill_graph_vram_gate(
        ns, model_pool, _csv_ints(ns.pp_stage_ratio) if ns.pp_stage_ratio else None,
        families, pool_floor, log)
    # PP-COST (01.10.): the stage model over the predecessor boot's chunk mix.
    _stage_model, _stage_prov = resolve_pp_cut_stage_model(
        str(getattr(ns, "pp_cut_stage_model", "off")), getattr(ns, "profile", None), model,
        inventory=card_identity.inventory_signature(cards))
    _chunk_mix = None
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
    # #1305: THE NUMBER, now that the frontier exists.  A W67/W40 raised by
    # the solve above carries the cut and the floor in its own text.
    log(pool_floor_line(decision))
    log(
        f"PP-CUT inputs: layers={n_layers} attn={n_attn} "
        f"kv={kv_mib * 1024 * 1024:.0f} B/token/attn-layer (from config, fp8_e4m3) "
        f"weights attn {terms.attn_layer_weight_bytes / _pp_cut.MIB:.1f} / linear "
        f"{terms.linear_layer_weight_bytes / _pp_cut.MIB:.1f} MiB per layer -> mean "
        f"{mean_layer_mib:.1f} used; free={budgets_p} MiB (this launcher's own "
        f"per-rank budgets); arming floor {ns.pp_cut_arming_floor_mib} MiB/rank "
        f"(rank0 layer cap only -- the pool model charges the corridor holdback "
        f"instead, #1286)"
    )
    # #1286: the posts, on their own line, in the runtime's own order and with
    # the runtime's own names, so this line and the boot's `KV budget posts`
    # line can be read side by side without a translation step.
    _act_heuristic_mib, _act_provenance = p_activation_reserve_provenance(
        model, int(getattr(ns, "p_bs", DEFAULT_P_BS) or DEFAULT_P_BS)
    )
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
    # #1286b -- THE CURVE. provenance_line prices TWO points of it (the two
    # objectives' cuts); this line prices every point that is not beaten on
    # both axes at once, so "what would a floor of N cost me in ms" is read off
    # one boot log instead of re-solved. It is also how a value for
    # --pp-solve-pool-floor is chosen.
    log(decision.frontier_line())
    for row in decision.table_lines():
        log(row)
    # WHICH of those priced rows this boot actually SHIPS (#1254 -- see
    # :func:`pick_shipped_cut`).  An operator pin outranks the objective: it IS
    # the chosen candidate, and overriding it here would make --pp-stage-ratio
    # a suggestion.
    if decision.pinned:
        chosen, ship_why = decision.chosen, (
            "PINNED by --pp-stage-ratio (the operator's own cut, which outranks "
            "--pp-solve-objective)"
        )
    else:
        chosen, ship_why = pick_shipped_cut(
            decision,
            P_PP_STAGE_RATIO_SCORES,
            P_PP_ATTN_STAGE_RATIO_SCORES,
            str(ns.pp_solve_objective),
        )
        # #1447: on the exchange arm the makespan pick is re-read against the
        # HOST cost of every frontier row at/above the floor, fastest first:
        # the boot ships the first one whose pinned bounce fits the incumbent's
        # plus the operator's slack.  The ledger keeps the last word.
        if str(ns.pp_solve_objective) == "makespan" and xchg_bounce_arm_pins_host(
                str(getattr(ns, "pdflip_weight_source", WEIGHT_SOURCE_DEFAULT)),
                str(getattr(ns, "pdflip_xchg_oncard", ONCARD_MODE_DEFAULT))) \
                and pdflip_pp_ordered_cut_for(len(budgets_p)) is not None:
            _floor = decision.pool_floor
            _rows = [c for c in decision.frontier
                     if _floor is None or float(c.pool_tokens) >= float(_floor)]
            # the incumbent of the HOST price is the ordered default cut
            # (39,13,12 -- the one measured lane set), not the scores' 32,18,14
            _inc_cut = _csv(DEFAULT_PP_ORDERED_CUT)
            _inc_gib, _inc_lanes, _inc_src = host_price_for_cut(ns, _inc_cut)
            try:
                _slack = float(os.environ.get(PCUT_BOUNCE_SLACK_ENV, "0") or 0.0)
            except ValueError:
                _slack = 0.0
            log("PP-CUT HOST-PRICE incumbent %s: xchg_bounce=%.2f GiB lanes=%d (%s); slack %s=%.2f GiB; "
                "%d frontier rows at/above the floor priced fastest first"
                % (_inc_cut, _inc_gib, _inc_lanes, _inc_src,
                   PCUT_BOUNCE_SLACK_ENV, _slack, len(_rows)))
            _picked, _hp_lines = host_priced_pick(
                _rows, lambda r: host_price_for_cut(ns, _csv(r.layers)), _inc_gib, _slack)
            for _ln in _hp_lines:
                log(_ln)
            if _picked is None:
                log("PP-CUT HOST-PRICE: NO frontier row at/above the floor is fundable within "
                    "the incumbent's bounce + slack -- the makespan pick stays and the ledger decides")
            elif _picked is not chosen:
                chosen, ship_why = _picked, (
                    "the fastest frontier cut at/above the floor whose exchange bounce fits the "
                    "incumbent's + slack (#1447 host-priced; the makespan pick %s/%s costs more host)"
                    % (_csv((decision.makespan or decision.chosen).layers),
                       _csv((decision.makespan or decision.chosen).attn)))
    # ALL THREE ARMS PRICED ON EVERY BOOT, the shipped one named.  The two the
    # boot did NOT take are the whole reason this line exists: the makespan
    # default was paid for weeks because its alternative was never printed
    # beside it, and a trade nobody can see is a trade nobody selected.
    incumbent_row = incumbent_candidate(
        decision, P_PP_STAGE_RATIO_SCORES, P_PP_ATTN_STAGE_RATIO_SCORES
    )
    makespan_row = decision.makespan or decision.chosen
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
    if chosen.kind == "gapped":
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
    return PCutFacts(
        stage_ratio=stage_ratio,
        attn_stage_ratio=attn_ratio,
        pool_tokens=float(chosen.pool_tokens),
        attn_counts=tuple(int(a) for a in chosen.attn),
        kv_mib_per_token_per_attn_layer=float(kv_mib),
        hidden_size=int(text_cfg["hidden_size"]),
        cap_tokens=int(ns.max_kv_per_request or CONTEXT_LENGTH_TOKENS),
        layer_counts=tuple(int(c) for c in realized),
    )


def build_parser() -> argparse.ArgumentParser:
    """THE parser, built apart from :func:`main` so the desk can render it.

    ``argparse`` evaluates every ``help=`` string as ``help % params`` when it
    formats it, so one bare ``%`` in one flag's help makes ``--help`` raise for
    ALL of them. That is not a cosmetic failure: the boot record cites group
    D's measured default as readable "verbatim from its own help path", and a
    channel that cannot be opened is not provenance. Splitting the build out is
    what lets a unit test call ``format_help()`` once per commit instead of an
    operator discovering it in a terminal.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--venv", default=VENV_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--profile", choices=list(PROFILES), default=PROFILE_QWEN27B,
                    help="Task #47 Scheibe 6a: checkpoint profile of the two groups; "
                         "nextflash = P PP3 + D Form A, uneven-DCP env per group, no DCP flag half.")
    # PDFLIP-FORM: --form-arch/--form-experts/--form-draft/--form-p-draft/--form-kv
    pdflip_form.add_form_arguments(ap)
    ap.add_argument(
        "--tokenizer-path", default="",
        help="27B line G2: the tokenizer directory BOTH groups load "
             "(--tokenizer-path on P and D). Default empty: nothing is added and "
             "the server reads the tokenizer from --model -- byte-identical argv. "
             "Needed for a GGUF --model whose directory holds no tokenizer files "
             "(qwen35 GGUFs load their tokenizer only from sibling files); the "
             "launcher refuses such a GGUF without it (W163) before any card is "
             "touched.",
    )
    # #1360: THE NAMED DEVIATION, flag only. The user's standing rule of
    # 2026-09-12 makes the reap mark soft -- crossing it is allowed WITH a
    # reason and a runtime latch, never silently -- and until now the launcher
    # had no way to say so: a refused arm simply stopped the boot. NO ENV TWIN
    # (#781 fossil class): one spelling, on the command line, where it lands in
    # the argv echo and in the form key.
    ap.add_argument("--host-ledger-deviation", default="", metavar="REASON",
                    help="Accept a REFUSED host-ledger fundability verdict under "
                         "a named reason. Requires --host-riegel-gib. Converts "
                         "the verdict only: no term, no ring size and no guard "
                         "is re-priced.")
    ap.add_argument("--host-riegel-gib", type=float, default=None, metavar="GIB",
                    help="The RUNTIME LATCH the deviation is accepted under, in "
                         "GiB non-reclaimable. Must be BELOW the hard bound, or "
                         "it cannot fire before the bound is crossed. Required "
                         "with --host-ledger-deviation.")
    # #1378 Stage 2 (W105): the PRIOR boot's own cited cushion_min/bounce,
    # operator-supplied like --host-riegel-gib -- there is no automatic
    # sidecar for this pair yet, so a boot that wants the cross-boot gate
    # cites its predecessor's own printed numbers (PDFLIP-HOST-LEDGER TERMS /
    # the v3 sampler's cushion_min). BOTH OR NEITHER (W105, checked inside
    # host_ledger.choose, not re-validated here): a half citation is a
    # guessed input printed as a measured one.
    ap.add_argument("--prior-cushion-min-gib", type=float, default=None,
                    metavar="GIB",
                    help="The PRIOR boot's own measured cushion minimum "
                         "(the v3 sampler's cushion_gib column, MINIMUM over "
                         "that boot). Feeds the cross-boot cushion_headroom "
                         "gate (#1378 Stage 2): ONE-TERM MODEL, measured "
                         "non-monotonic against real boots -- RED (negative "
                         "headroom) MAY refuse this boot's arm, GREEN funds "
                         "nothing by itself. Requires --prior-bounce-gib.")
    ap.add_argument("--prior-bounce-gib", type=float, default=None,
                    metavar="GIB",
                    help="The PRIOR boot's own priced xchg_bounce (the "
                         "PDFLIP-HOST-LEDGER ARM line's xchg_bounce= field). "
                         "Required with --prior-cushion-min-gib.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force", action="store_true",
        help="PROFIL-EDITOR S1: start although the planner/launcher REFUSES the VALUES (capacity, VRAM, "
             "card count, calibration of the inventory, disk floor -- the 'value' class of pdflip/refusals.py). Every refusal passed "
             "is printed as 'FORCED-PAST <CODE> <reason>'. A forced boot writes NO records -- unless every refusal it "
             "passed is a proof gap of its own form (#227: HW-COUNT [METAL-UNPROVEN] alone, the PP-CUT crossing-width "
             "HW-UNCALIBRATED); then its records carry forced_past (line FORCED-RECORDS). NOT lifted: the occupation "
             "check (another process or window on a card), a missing or broken model/file, an unsupported "
             "architecture. HOST RAM is not refused at all any more (HOSTWARN2, user rule 06.10.): the host ledger, "
             "the preflight and the pinned-host registry print 'HOST-RAM-WARN ...' and the boot goes on, with or "
             "without --force. Container: FLLIPER_FORCE=1.")
    ap.add_argument(
        "--profile-inventory", default="",
        help="HW-GENERIC 1002: the card inventory (calibration-class labels in card order, "
             "pdflip/card_identity.py, e.g. 'RTX5090,RTX3080,RTX3080') the profile's positional "
             "vectors were measured on. Default: the inventory the profile's measured records "
             "declare (profile_records_data/<profile>.json 'inventory'). A live inventory that "
             "differs is refused BY NAME (HW-UNCALIBRATED) -- positional measurements of other "
             "cards are never borrowed.")
    ap.add_argument(
        "--cards", default="",
        help="HW-P1a 1003: the NVML indices this launch may use, e.g. '1,0' (the 5090 and one "
             "3080 of the reference rig). Default: every NVML device. The launcher reads NVML "
             "directly and ignores CUDA_VISIBLE_DEVICES (it sets that itself per group, by UUID), "
             "so a bare-metal subset is chosen here; the card ORDER still comes from the card "
             "properties (order_cards), never from this list. An index NVML does not report, or "
             "a duplicate, is refused by name (HW-CARDS).")
    ap.add_argument("--debug-hold", choices=["none", "P", "D", "both"], default="none")
    # #1236: --store-min-gib IS DELETED. It was a FLOOR on how much of the host
    # RAM leftover the store tmpfs had to get, and there is no RAM leftover to
    # floor any more -- the store is a directory on the ZFS dataset sized from
    # the P KV pool. Its replacement is not another knob but a LAW checked
    # against the disk (max_size >= P pool bytes, W57). The two knobs below are
    # the only ones the disk form has, and both have a stated default.
    ap.add_argument(
        "--pin-ledger-arm-s", type=int, default=1,
        help="#1422 (boot xsn172): the pinned arm's host KV pool S (GB) for group P. "
             "The ladder's S=1 gave PP0 a 54,254-token staging pool (cell 18,432 B), "
             "so a re-queued 100k prompt could never be read back from the arena and "
             "was recomputed on P (98k tokens, 29-57 s) with every page in L2. S=5 "
             "holds 271k tokens on PP0 = the 262,144 cap plus one window. Back to 1 with "
             "#1424 (Stufe 3): reads are arena slots, the pool is a 1 GB staging ring. 0 = S=1.")
    ap.add_argument(
        # #1451: default 600 (was 2400) -- the number every arm has pinned
        # since the arena entered the ledger (#1432): with arena 33.6 GiB the
        # 2400 rung is unfundable on this box (test_pdflip_hicache_disabled_1386
        # read it against the pinned quiet-box meminfo), 600 is what boots run.
        "--pin-ledger-arm-m", type=int, default=600,
        help="#1317/#1318: PIN the host-ledger arm's mamba-host-pool size M (MiB) "
             "instead of letting `host_ledger.choose` pick it. 0 = choose. Default 2400 "
             "since 2026-09-16 (#1414, boot xsn163: at M=600 = 13 host anchor slots PP2 "
             "answered PP0's told=4095 with 'anchor_pool_exhausted' during 3 x 100k and "
             "the told mismatch stopped P; 600 gave 12-13 slots since 2026-09-15, M=150 "
             "gave 4 and dropped the next prefetch on xsn127). "
             "WHY IT EXISTS: #1318 derived the host ring multipliers from rows x cell "
             "bytes (D 6.0 -> 3.00 GB/S, P 2.0 -> 1.78), which frees 3.12 GiB at every "
             "arm and moves `choose` from M=600 to M=1200 -- the arm boot weg2dk5 was "
             "REAPED on. Until a boot measures the host peak at the derived ring, a "
             "measurement boot pins M to the value its reference boot actually ran. "
             "The pinned arm is still PRICED and is REFUSED by name if it is not "
             "fundable at both moments: this is a pin, never an override of the "
             "ledger's verdict.")
    ap.add_argument(
        "--store-max-gb", type=float, default=0.0,
        help="#1236 / user order 2026-09-10: the store's max_size as an ABSOLUTE size in "
             "GB (10^9), overriding the derived `P pool bytes x --store-sidecar-factor`. "
             "0 = derive as before. The #1236 floor stays enforced: a value below the P "
             "pool is refused by name (W57), never clamped. Use this when the operator has "
             "given the store a disk budget; use --store-sidecar-factor when you mean "
             "'more sidecar headroom per pool token'.")
    ap.add_argument(
        "--store-sidecar-factor", type=float, default=STORE_SIDECAR_FACTOR,
        help="#1236: multiply the P KV pool's bytes by this to size the store, "
             "covering the Mamba blobs and draft pages that ride beside the KV "
             "pages. Default %(default).2f, DERIVED not picked: "
             "(kv + mamba + draft) / kv at " + STORE_CENSUS_PROVENANCE + " "
             f"(kv={STORE_CENSUS_KV_PAGES}x{STORE_CENSUS_KV_PAGE_BYTES} B, "
             f"mamba={STORE_CENSUS_MAMBA_BLOBS}x{STORE_MAMBA_BLOB_BYTES} B measured on "
             f"/spinning/hicache-l3, draft={STORE_CENSUS_DRAFT_PAGES}x2048 B). The mamba "
             "half moves with chunk granularity and prefix reuse, so this is a sizing "
             "input with provenance, not a constant of the model.")
    ap.add_argument(
        "--store-disk-min-free-gib", type=float, default=STORE_DISK_MIN_FREE_GIB,
        help="#1236: HiCacheFile min_free_space, against the DISK the store lives on "
             "(default %(default).0f GiB). This filesystem is the container's own root, "
             "so the floor protects the box -- boot logs, the evidence tree, the model "
             "cache -- not the store. NOT the old 1 GiB, which was 1/6th of a 6 GiB "
             "tmpfs and latched the backend's write stop after 5 GiB.")
    ap.add_argument("--dual-layout", action="store_true", default=False,
                    help="DUAL-TP3PP3 (F26, user 29.09.): BOTH groups stay awake for the whole boot and "
                         "the front never flips. Implies --flip-weights resident; P is NOT put to sleep "
                         "after READY, its AWAKE footprint is measured in its place and D is sized from "
                         "that measurement (the same per-PID reading the dormant residue uses). P's own "
                         "budget is lowered with --extra-p '--rank-gpu-memory-mib ...' (W100 allows "
                         "lowering). Refuses --pdflip-d-adopt on and --idle-layout pp. Default off: the "
                         "launcher is byte-identical.")
    ap.add_argument("--dual-share", action="store_true", default=False,
                    help="DUAL-TP3PP3 stage 1b (implies --dual-layout): P's stage computes on D's TP "
                         "shards (shells over D's three shards of its layers) and binds the shard of the "
                         "D rank on its card to D's bytes via the union image. D boots right after P as "
                         "the union OWNER, sized from P's PLANNED budget (+ --dual-p-overhead-mib); P "
                         "waits for D's image before it loads anything.")
    ap.add_argument("--dual-p-overhead-mib", type=int, default=1500,
                    help="DUAL-TP3PP3 --dual-share: what P holds on a card outside its "
                         "--rank-gpu-memory-mib budget (CUDA context, graphs, activations), charged "
                         "when D is sized from P's plan instead of P's measurement.")
    ap.add_argument("--dual-d-prefill-tokens", type=int, default=0,
                    help="DUAL-TP3PP3 D PREFILL ALLOWANCE, DEFAULT 0 (user decision 01.10.: in the dual "
                         "layout EVERY prefill runs on P, D only decodes). D's X (its W31 riegel and the "
                         "front's route bound) becomes 1 + this value: the 1 is the N-1 anchor convention "
                         "(P holds the last prompt token back, D's first step computes it), anything "
                         "larger goes to P and a larger remainder at D is refused by name (W31 -> P), "
                         "never silently recomputed. Ignored outside --dual-layout; replaces "
                         "--tp-prefill-max-tokens / --x-ceiling-tokens / --d-short-drain-tokens there.")
    ap.add_argument("--dual-p-duty", type=float, default=1.0,
                    help="DUAL-TP3PP3 PAUSE ALLOWANCE (Pausenzulassung), DEFAULT 0 = 1.0 = no pause: P and D "
                         "run at once and share every card, only physics limits them (user decision 01.10.). "
                         "Below 1.0 P MAY pause while D holds decodes: the share of wall time P's first stage "
                         "may compute (pdflip/dual_duty.py). Plain form: PP0 idles t_fwd*(1-duty)/duty after "
                         "each forward. With env FLLIPER_PDFLIP_DUAL_P_GANG_CHUNKS=K (gang window): PP0 runs K "
                         "chunks, waits for the pipeline to drain, then holds so D is alone on all cards.")
    ap.add_argument("--dual-p-sm-pct", type=int, default=100,
                    help="DUAL-TP3PP3 with --dual-mps on: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for group P, "
                         "i.e. the share of SMs P's kernels may occupy while D decodes. MEASURED 29.09. "
                         "(risk-1 bench, 5090): unlimited P takes ~90%% of the card and D's step runs "
                         "~7x slower; 50 splits ~50/50; the sum of both shares stays ~1.0 either way. "
                         "100 = no limit.")
    # DUAL-SHARE (pdflip/dual_share.py, item 800, user order 03.10. ~18:55-19:15Z):
    # P vs D precedence selectable and dynamic. Every switch alone, default off =
    # launcher argv/env byte-identical; refused by name outside --dual-layout.
    ap.add_argument("--dual-priority", choices=DUAL_PRIORITY_MODES, default=None,
                    help="DUAL-SHARE stage 2: the front's P/D share controller writes a rung (P's share "
                         "1.0/0.75/0.5/0.25, profile env FLLIPER_PDFLIP_DUAL_SHARE_RUNGS) to <busy>.ctl; P acts "
                         "through --dual-share-actuators. p = P full (today), balanced/d = a static rung while "
                         "D decodes, dynamic = matrix of tau (queued prefill tokens / P full rate) x D bs. "
                         "Switchable at runtime: POST /pdflip/dual-priority on the front. Unset = off.")
    ap.add_argument("--dual-d-min-rate-tps", type=float, default=0.0,
                    help="DUAL-SHARE: D's minimum decode rate per request (tok/s, measured from the leg-2 "
                         "token stream); below it the rung steps toward D. 0 = off.")
    ap.add_argument("--dual-p-min-share", type=float, default=0.25,
                    help="DUAL-SHARE: floor of P's share (rungs below it are never used).")
    ap.add_argument("--dual-share-actuators", default="chunk",
                    help="DUAL-SHARE: comma list of P actuators: chunk (PP0's chunk cap), duty (P duty throttle "
                         "at the rung's fraction), green (stage 3, NOT built: named fallback to chunk).")
    ap.add_argument("--dual-green-ladder", choices=("off", "on", "hold"), default="off",
                    help="DUAL-SHARE stage 3 (pdflip/dual_green.py): P's SM share as a green-context LADDER "
                         "(100/75/50/25 %% per forward, dynamic, up AND down; PP0 stamps the stage on the request "
                         "wire so all three stages run the same one). 'on' = ladder + hold OBSERVER (the P-STUFE "
                         "log carries would_hold), 'hold' = additionally PP0 really holds (0 %%) while the arena "
                         "is full. Needs --dual-priority, 'green' in --dual-share-actuators, --dual-mps on, "
                         "no --dual-p-sm-pct. Default off = argv/env/launch path byte-identical.")
    ap.add_argument("--dual-d-capture-prio", choices=("off", "on"), default="off",
                    help="DUAL-SHARE stage 1a: D captures its CUDA graphs on the device's highest-priority "
                         "stream (graph nodes keep the capture stream's priority). Group D of the dual layout only.")
    ap.add_argument("--dual-p-mps-low-prio", choices=("off", "on"), default="off",
                    help="DUAL-SHARE stage 1b: P starts with CUDA_MPS_CLIENT_PRIORITY=1 (below normal; read at "
                         "connect, boot only). Needs --dual-mps on, else a named W-DUAL-SHARE-FALLBACK line.")
    ap.add_argument("--dual-p-sleep", choices=("off", "on"), default=None,
                    help="DEFAULT: ON with --dual-share, OFF otherwise (since gmps12, "
                         "dkr27bnvfp4dual1mpsleepbar1fs10020008: P slept and woke under --dual-share on all three "
                         "stages, SLEEP-LEND/WAKE-RECLAIM; the metal probe FLLIPER_PDFLIP_DUAL_P_SLEEP_PROBE_S stays "
                         "off by default). History: "
                         "default off since gmps9 (dkr27bnvfp4dual1mbar1fs10012051, P-PP0 init_memory_pool: 'per-rank "
                         "budget leaves no GPU memory for the KV cache'): 'on' loads P's weights inside the "
                         "memory-saver's private MemPool (not resident), and the --dual-share union bind frees P's "
                         "own copies into that pool, which empty_cache never returns ('PDFLIP-UNION PEER ... card free "
                         "3.35 -> 3.35 GiB', resident gmps7: 3.31 -> 11.74 GiB). 'on' together with --dual-share is "
                         "therefore REFUSED at launch (W-DUAL-P-SLEEP-SHARE) until P's union-bound part loads outside "
                         "the pool. "
                         "DUAL-TP3PP3 with --dual-unified-kv on: D PRIORITY stage 2 (user decision 01.10.) -- "
                         "when D is still short after P stopped and released its KV, P sleeps and parks its "
                         "weights in host RAM (P boots with --enable-weights-cpu-backup and NOT resident; the "
                         "vendored torch_memory_saver allocates the host image at the pause and frees it after "
                         "the restore, tms_csrc/core.cpp -- no resting image, KEIN-DAUER-HOSTRAM). off = stage 1 "
                         "only, the front prints 'stage=2 unavailable (weights resident)'.")
    ap.add_argument("--dual-unified-kv", choices=("off", "on"), default="off",
                    help="DUAL-TP3PP3: one KV pool per card shared by P and D at runtime (user orders "
                         "30.09. 07:10Z/07:25Z; pdflip/card_kv_ledger.py). P maps KV only while it "
                         "prefills and pauses when D is short. Needs --dual-share.")
    ap.add_argument("--dual-p-kv-max-tokens", type=int, default=196608,
                    help="DUAL-TP3PP3 --dual-unified-kv: P's KV pool rows (virtual; pages from the card "
                         "pool). Each K/V buffer is born at this size and trimmed at once, so the boot "
                         "transient is one buffer (tokens x bytes per token per layer).")
    ap.add_argument("--dual-d-kv-max-tokens", type=int, default=1048576,
                    help="DUAL-TP3PP3 --dual-unified-kv: D's KV pool in GLOBAL tokens (virtual; D keeps "
                         "its boot level mapped and grows from the card pool). Must exceed D's boot "
                         "context; each buffer is born at its owner share of this and trimmed at once.")
    ap.add_argument("--dual-mps", choices=("off", "on"), default="off",
                    help="DUAL-TP3PP3: start a private MPS control daemon before the groups (pipe dir "
                         "under the boot's run dir) so P and D kernels run concurrently on a card instead "
                         "of time-slicing. Only with --dual-layout. Needs FLLIPER_PDFLIP_DUAL_MPS_OPT_IN=1 (the "
                         "release profile 27b-nvfp4-dual sets it): on 30.09. MPS wedged both groups under "
                         "extend-sized collectives (repro v2 scjhru S1, boots kw6pft/ndktv4); root cause fixed by "
                         "b855aecfd3 (barlink-BAR1 under MPS without the cooperative full-card grid). MPS is not "
                         "needed to RUN the dual layout (without it: time-slicing, latency guard --dual-p-duty, "
                         "no green ladder) but it is what gives concurrent P/D kernels and the dynamic GEMM share "
                         "(--dual-green-ladder needs --dual-mps on). See docs/dev/dual-layout.md.")
    ap.add_argument("--flip-weights", choices=("family", "resident"), default="family",
                    help="Task #47 Scheibe 6a: 'family' (default) moves the weights family across the flip "
                         "(gathered legs, host ring / exchange); 'resident' keeps BOTH groups' weights mapped "
                         "and flips only kv_cache (the front skips the legs: --weights-resident).")
    ap.add_argument("--weight-chunks", type=int, default=8,
                    help="#1233: number of weights_<k> layer-chunk tags per group (0 = the round-1 single tag / two-backup shape)")
    ap.add_argument("--ready-deadline-s", type=float, default=900.0)
    ap.add_argument("--extra-p", default="", help="extra flags for group P (shell-split)")
    ap.add_argument("--env-p", default="", help="Scheibe 6a: per-group env for P, 'KEY=VAL;KEY=VAL' (values may hold commas; applied last in build_env)")
    ap.add_argument("--env-d", default="", help="Scheibe 6a: per-group env for D, 'KEY=VAL;KEY=VAL' (values may hold commas; applied last in build_env)")
    ap.add_argument("--extra-d", default="", help="extra flags for group D (shell-split)")
    ap.add_argument("--fairness-w-s", type=float, default=45.0,
                    help="A1-1: the only sanctioned pre-emption; 0 disables it. Passed to the front.")
    ap.add_argument("--front-host", default=DEFAULT_FRONT_HOST,
                    help=f"bind address of the front on :{PORT_FRONT}. Default {DEFAULT_FRONT_HOST} = LAN-reachable "
                         f"(user order 2026-09-10); 127.0.0.1 keeps it host-local. Refused before any group "
                         f"is spawned if the address cannot be bound (e.g. the temporary LAN forwarder "
                         f"pdflip-lan-forward.socket still holds the port).")
    ap.add_argument(
        "--p-chunk-raster", type=int, default=0, metavar="TOKENS",
        help="H92, --profile nextflash only, with --p-chunk-policy dynamic: the NF chunk raster (0 = 4096, the only "
             "accepted value; ChunkLimits.page).")
    ap.add_argument("--p-bs", type=int, default=DEFAULT_P_BS,
                    help=f"K1: group P's --max-running-requests AND the front's leg-1 concurrency. "
                         f"Independent of --d-bs (law 2). Default {DEFAULT_P_BS} by the user order of "
                         f"2026-09-09 ('bs2 fuer prefill'), which calls itself PROVISIONAL -- the "
                         f"optimum for this workload is to be re-measured once the strand is finished, "
                         f"and this flag is how you do it. Also sizes P's req_to_token_pool, so it is "
                         f"resolved before the budget solve.")
    ap.add_argument("--d-bs", type=int, default=DEFAULT_D_BS,
                    help=f"K2: group D's --max-running-requests AND the number of front seats, so "
                         f"the front never hands D more concurrent requests than D can run. Default "
                         f"{DEFAULT_D_BS} by the user order of 2026-09-09 ('der decode bs6 soll mit "
                         f"bs6 (nicht mehr bs4) der standard werden'), which superseded that same day's "
                         f"'bs4 fuer decode' before it reached metal. Equally provisional and equally "
                         f"re-measurable.")
    ap.add_argument("--tp-prefill-max-tokens", type=int, default=4096,
                    help="K5 (X): uncached tokens D may prefill itself. Default 4096 since "
                         "2026-09-15: the derivation below priced r_P from a single in-flip "
                         "leg's front wall (1346 tok/s) and routed every request SHORT->D "
                         "(xsn124-126); pass 0 to derive. Derived = "
                         "2*flip_s/(1/r_D - 1/r_P) from this rig's own front-log rate and flip "
                         "lines, else the recorded PRE-BARLINK pair; floor = --chunked-prefill-size. "
                         "The derivation and its three inputs are printed at launch.")
    ap.add_argument("--d-only", action="store_true",
                    help="Nutzer 25.09.: NUR Gruppe D (TP3 auf allen Karten), kein P, kein Flip, keine "
                         "Front. D wird mit derselben Env und demselben argv gebaut wie im Flip-Boot "
                         "(Budgets aus der Erwartung wie im Dry-Run), nimmt aber jede ungecachte Laenge "
                         "selbst an (--tp-prefill-max-tokens = --max-kv-per-request) und wird direkt auf "
                         f":{PORT_D} angesprochen. Fuer Hosts, deren RAM den Flip nicht traegt.")
    ap.add_argument("--x-ceiling-tokens", type=int, default=0,
                    help="H84: group D's --tp-prefill-max-tokens (its W50 riegel) and the ceiling "
                         "of the front's live X re-solve. 0 (default) = off: D and the front both "
                         "keep the start X, argv unchanged. >0: D starts with this riegel, the "
                         "front keeps its start X and gets --x-ceiling-tokens, so the live X may "
                         "rise up to it; a request between the start X and the live X goes to D "
                         "only when nothing else is in flight (PDFLIP X-SOLO). Lifted to the start X "
                         "when below it. Group P is not affected.")
    ap.add_argument("--x-mode", choices=pdflip_x_curves.X_MODES, default=None,
                    help="X-CURVES 1006 (user 06.10.: only fixed and curve): how the front sets X. "
                         "fixed = --tp-prefill-max-tokens for the whole boot, no live re-solve; curve = "
                         "X PER REQUEST from --x-curves (the D/P prefill curves and the flip price of "
                         "this model x form x hardware); --x-ceiling-tokens beside it clamps X from "
                         "above (user 06.10.), D's W50 riegel = the LOWER of the curves' envelope and "
                         "that ceiling. Unset = the front exactly as before the flag, argv "
                         "byte-identical. Refused in the dual layout (W197).")
    ap.add_argument("--x-curves", default=None,
                    help="X-CURVES: the curve file (weg2-x-curves/1, tools/build_x_curves.py). Only "
                         "with --x-mode curve (W194 otherwise); absent/unreadable W190, "
                         "malformed W191, another model/form/hardware W192 -- refused at launch.")
    ap.add_argument("--x-curves-beyond", choices=pdflip_x_curves.BEYOND_POLICIES, default=None,
                    help="X-CURVES: a request deeper than the curves reach: clamp (default, priced at "
                         "the deepest row and named) or refuse (W193 at the front).")
    ap.add_argument("--x-busy-tokens", type=int, default=None,
                    help="27B RC7-X / UNIFY S4: the floor of the front's X-SOLO band -- up to it "
                         "D prefills a SHORT request while it decodes others, above it (up to the "
                         "live X) only as a singleton. Unset (default) = the front's default, the "
                         "start X (the H84 band floor); passed to the front only when set. 0 = no D "
                         "prefill while D decodes others; negative is refused (W155).")
    ap.add_argument("--flip-min-work-tokens", type=int, default=None,
                    help="K6: queued tokens that make a D->P round trip worth its cost. Unset = X.")
    ap.add_argument("--min-dwell-ms", type=float, default=None,
                    help="K7: minimum phase dwell. Unset = derived from the last flip in that direction.")
    ap.add_argument(
        "--pp-cut-reserve-mib", default="",
        help="#141: per P-stage the MiB that on the card NICHT the weights "
             "gehoeren (KV-Pool, Draft-KV-Produzent, Prefill-Transiente). "
             "Macht aus der #140-Decke eine Empfehlung. Ohne Angabe druckt "
             "der Solver nur die Decke -- fnFL2w73 starb in genau dieser "
             "Luecke (PP2: 4,93 GB frei, dann cu_mem_create OOM).")
    # -- #145: DIE D-SEITE. Drei Flags, ALLE ohne Default, alle in RANG-/
    # ORDINAL-Reihenfolge (dieselbe wie --user-reserve-mib und die
    # `budget D ... ordinal=i`-Zeilen: ordinal 0 = die 5090 = nvml1).
    # Wer einen Term weglaesst, bekommt ihn im Druck als FEHLT benannt --
    # kein geratener Default, weil ein geratener Default hier genau die
    # Zahl waere, um die es geht.
    ap.add_argument(
        "--d-kv-token-cut", default="off",
        help="#239 (uneven-DCP-KV, release feature): cut D's full-attention KV "
             "by TOKENS over the Form A ranks (form kv=qsa_forma_dcp). 'off' "
             "(default, byte-identical), 'maxmin' (the D solve picks the shares: "
             "max-min of the relative row ceiling, in 64ths) or a ratio vector "
             "per D rank; 'joint' (S2b) also sets FR_D at every rank's edge "
             "(max-min of the resident expert share per card); 'owned' (S3f, the "
             "target form) holds the attention host at share 0 and solves the "
             "workers' shares, FR_D and the MoE ownership (--rank-moe-ratio) "
             "together (min max miss time, x1 rule). A flip boot under the cut "
             "(S3h) needs the host tier and exactly one D KV stage "
             "(--env-d FLLIPER_PDFLIP_D_KV_STAGE_TOKENS=<tokens>), else it is "
             "refused by name (--d-only boots it either way).")
    ap.add_argument(
        "--d-foreign-context-mib", default="",
        help="#145 Term (b): per D-RANG (ordinal) of the VRAM, which the SLEEPING "
             "Phase auf derselben Karte liegen laesst -- ihr CUDA-Kontext. "
             "Die Layer-Bytes wechseln beim Flip nur den Besitzer (Sharing, "
             "kein Backup), der Kontext nicht; dieser Betrag steht der "
             "aktiven Phase NICHT zur Verfuegung. GEMESSEN in sieben Boots "
             "(vramwatch_fnFL2w80..w89, Plateau nach P's Einschlafen): "
             "nvml0=852, nvml1=1342, nvml2=844 MiB, also in "
             "Ordinal-Reihenfolge (5090 zuerst) 1342,852,844.")
    ap.add_argument(
        "--d-nontorch-mib", default="",
        help="#145 Term (c): per D-RANG (ordinal) of the VRAM, which the rank "
             "AUSSERHALB des Torch-Allokators haelt -- eigener CUDA-Kontext, "
             "TMS-Handles aus cu_mem_create, private VMM-Pools. Messbar als "
             "nvml_used - Fremd-Kontext - torch_reserved. GEMESSEN am Ende "
             "des Ladens: 555 MiB (TP1/nvml0), 522 MiB (TP2/nvml2); fuer "
             "TP0/nvml1 mindestens 3628 MiB (w83) bzw. 3818 (w86) -- dieser "
             "Rang hat das Laden nie beendet, die Zahl ist eine UNTERGRENZE. "
             "Genau dieser Term fehlte in w83: Budget 29680, verfuegbar "
             "32607-1342-3628 = 27637.")
    ap.add_argument(
        "--d-reserve-mib", default="",
        help="#145: das Gegenstueck zu --pp-cut-reserve-mib auf der D-Seite. "
             "Je D-Rang (ordinal) die MiB, die auf der Karte NICHT den "
             "Gewichten gehoeren: KV-Pool, Draft, Aktivierungen. Ohne Angabe "
             "druckt der Solver die DECKE, nicht die Empfehlung.")
    ap.add_argument(
        "--d-residency-reference-seats", type=int, default=1,
        help="H91b: die Sitze (--max-running-requests), mit denen die Boots aus "
             "--d-residency-reference-logs liefen. Die eingebauten NF-Referenzen "
             "tragen 1 (bs1, belegt durch ihre 7 Mamba-Slots). Die Posten 'mamba "
             "state pool' und 'speculative intermediate state' werden von dieser "
             "Zahl auf D's wirksame --max-running-requests umgebucht.")
    ap.add_argument(
        "--d-seat-graph-mib", default="",
        help="H91b: GEMESSENE Mehrkosten des Decode-CUDA-Graphen je zusaetzlichem "
             "D-Sitz, MiB (ein Wert fuer alle Raenge oder einer je Rang). Leer = die "
             "KARTE-Zeile nennt den Posten als NICHT GEBUCHT (Obergrenze).")
    ap.add_argument(
        "--d-residency-reference-logs", default="",
        help="H8: comma-separated list of D-Boot-Logs (boot_weg2_<tag>_*.D.log), from "
             "denen der D-FRACTION-SOLVE den festen Rang-Posten MISST "
             "('weights + runtime state' minus Experten-Puffer, Aktivierung, "
             "mamba/spec, KV-Zelle, Draft-Vokabular; je Term das Maximum ueber "
             "die Boots). Leer = die eingebaute Referenz des Baum-Zustands "
             "(H50, FLLIPER_PDFLIP_DENSE_REPACK_OUTSIDE_POOL der D-Gruppe): an = "
             "expert_residency.D_RESIDENCY_REFERENCE_FNFL2_H39 (fnFL2x151 + x158), "
             "aus = D_RESIDENCY_REFERENCE_FNFL2 (fnFL2x98/x99/x100), beide Next "
             "Flash Form A, --rank-tp-ratio 1,0,0; fuer jede andere Form "
             "entfaellt die Decke mit Namen, bis Logs DIESER Form gegeben sind. "
             "Logs im anderen H39-Zustand als die D-Gruppe werden benannt, nicht "
             "gerechnet.")
    ap.add_argument(
        "--d-card-reference-logs", default="",
        help="H33: Komma-Liste von D-Boot-Logs, aus denen die KARTEN-Bilanz des "
             "D-FRACTION-SOLVE den Posten AUSSERHALB des Budgets misst: je Rang "
             "Kopfraum = cap - peak - privat_frei am bindenden Messpunkt "
             "(PDFLIP-GRAPH-POOL-Zeilen; aeltere Logs: [vram-peak] + #1027 "
             "trapped=), normiert auf den Pufferbytes des Logs; je Rang das "
             "Minimum ueber die Boots. Unter corridor_guard.NEAR_OOM_MIB "
             "verweigert W130. Leer = die eingebaute Referenz des Baum-Zustands "
             "(H50): H39 an = expert_residency.D_CARD_REFERENCE_FNFL2_H39 "
             "(fnFL2x151 + fnFL2x158), aus = D_CARD_REFERENCE_FNFL2 (fnFL2x141 + "
             "fnFL2x144).")
    ap.add_argument(
        "--p-card-reference-logs", default="",
        help="H41: comma list of P-Boot-Logs (PDFLIP-GRAPH-POOL-Zeilen), from which "
             "die P-KARTE je Stufe ihren Kopfraum misst: cap - peak - privat_frei "
             "am bindenden Punkt, normiert auf Puffer, KV-Zelle x Token und die "
             "Chunk-Transiente des Logs; je Stufe das Minimum ueber die Boots. "
             "Unter corridor_guard.NEAR_OOM_MIB verweigert W132. Leer = die "
             "eingebaute Referenz p_card_chunk.P_CARD_REFERENCE_FNFL2 (H59: "
             "fnFL2x163 + fnFL2x164 + fnFL2x165, 4 Sitze, Schnitt 29,11,8, Chunk 16384, "
             "97k-Prompt); ein anderer Schnitt oder MEHR Sitze als die Referenz "
             "lassen die Bilanz mit Namen entfallen. Liegt neben <x>.P.log ein "
             "<x>.D.log, liest die Karte daraus den Mitbewohner (schlafender "
             "D-Rang derselben Karte) am Referenzpunkt und frischt dessen "
             "gemessenen Hochstand auf. Die Referenz ist ein BAUMSTAND "
             "(privat_frei, LMEM) und wird mit ihm aufgefrischt.")
    ap.add_argument(
        "--p-card-over-logs", default="",
        help="H41c: Komma-Liste von P-Boot-Logs DERSELBEN Form mit mehr "
             "Pufferzeilen (lebend oder tot). Ihr CHUNK-0-Punkt (PDFLIP-GRAPH-POOL + "
             "#969 EXTENT) misst, was eine Zeile ueber der Referenz die Karte "
             "kostet (obere Schranke, nie unter dem Zeilenbild). Nur zusammen mit "
             "--p-card-reference-logs; leer = eingebaut (fnFL2x149).")
    ap.add_argument(
        "--p-card-prompt-tokens", type=int, default=0,
        help="H41c/H41d: die Prompt-Laenge, an deren LETZTEM Chunk die P-KARTE "
             "verweigert. Das Chunk-Wachstum der Peak saettigt am letzten "
             "wachsenden Chunk der Referenz (fnFL2x160: Chunk 3, gemessen ueber "
             "16 Chunks); ein Prompt laenger als der laengste der Referenz heisst "
             "HOCHRECHNUNG. 0 = der volle Kontext (262144).")
    ap.add_argument(
        "--wake-credit-reference-logs", default="",
        help="H14: P.log,D.log,front.log of ONE boat, whose first Wake (D->P) "
             "die Referenz des WAKE-CREDIT-Riegels (W126) ist: free beim "
             "Flip-Start, Ordnung, Tags beider Gruppen, P-Floors, Pufferzeilen, "
             "On-card-Staging. Leer = die eingebaute Referenz "
             "wake_credit.REFERENCE_FNFL2X114D (Next Flash Form A, P-Schnitt "
             "29/11/8, Karten 1,0,2); fuer eine andere Geometrie entfaellt der "
             "Riegel mit Namen.")
    ap.add_argument("--idle-layout", choices=["tp", "pp"], default="tp",
                    help="K8: which layout is awake at rest -- tp = group D (today's shape), "
                         "pp = group P. The front's idle guard always counts the requests P has "
                         "just prefilled, so resting on P loses no request.")
    ap.add_argument("--d-short-drain-tokens", type=int, default=0,
                    help="27B idle policy (b), passed to the front: while D is awake, a queued "
                         "backlog made ONLY of SHORT requests (each <= X, law 4 -- D never prefills "
                         "above X) whose uncached tokens sum to at most N is served on D instead of "
                         "waiting for a flip. N > X is refused (W153: law 4 holds for the sum of a "
                         "drain too), and the front caps every drain at min(N, the X in force). "
                         "0 (default) = off = today: such a backlog waits for "
                         "FLIP-ECONOMICS (--flip-min-work-tokens) or the fairness bound.")
    ap.add_argument("--d-hold-s", type=float, default=None,
                    help="27B idle policy (c), passed to the front: D stays awake this many seconds "
                         "after its own work ended before it flips -- at rest under --idle-layout pp, "
                         "and in front of a small backlog FLIP-ECONOMICS holds -- so a SHORT arrival "
                         "inside the hold is served without a flip. Unset (default) or 0 = off = today: "
                         "the idle flip waits for --min-dwell-ms only, a held backlog for "
                         "--fairness-w-s only.")
    ap.add_argument("--d-admit-max-tokens", type=int, default=None,
                    help="FIX 4 (round 4): OPERATOR CEILING on the aggregate store-read budget "
                         "group D may hold in flight. Unset (the default) is NOT a derived number "
                         "any more -- the front reads group D's own #915 host-pool terms off "
                         "/server_info and needs no launcher-side proxy for them. 0 disables the "
                         "gate.")
    ap.add_argument("--drain-deadline-s", type=float, default=120.0,
                    help="K10: seconds before a flip is refused by name (W1 -> W2). Shipped value.")
    ap.add_argument("--max-kv-per-request", type=int, default=None,
                    help="K9: per-request KV ceiling for BOTH groups. Unset = the model context "
                         f"({CONTEXT_LENGTH_TOKENS}), i.e. the as-built cap.")
    ap.add_argument("--d-cap-rank-share", type=float, default=0.40,
                    help="#1317n: the largest share of --max-kv-per-request one "
                         "group-D rank may own, i.e. max(v)/sum(v) of D's "
                         "KV-token ownership vector. Sizes D's --hicache-size so "
                         "ONE cap-sized store read fits in ONE prefetch per rank "
                         "(no window, no chain). A NAMED MARGIN, not a reading: "
                         "the launcher ships no vector (#1032) so the installed "
                         "one is unknowable here; weg2sn6p measured 0.3750 both "
                         "as estimate and as installed, and 0.40 covers it. Too "
                         "small and D's #915 limit lands under cap x share and "
                         "the #1246 bound names it; too large and the host "
                         "ledger pays for rows nobody reads.")
    ap.add_argument("--carrier-max-tokens", type=int, default=None,
                    help="#1246: ship a LOWER carrier bound than the one the census reads from group D's "
                         "own '#915 PREFETCH LIMIT' line. IT CAN ONLY LOWER IT: N is accepted exactly on "
                         "floor < N <= measured, where 'measured' is the budget group D's KV carrier "
                         "reports it will enforce and 'floor' is the route floor derived from the front's "
                         "own two bypass branches. N ABOVE measured is a W45 PdFlipCarrierCensusRefused "
                         "(operator_above_measured): a higher number does not enlarge the carrier, it only "
                         "stops the front bypassing prompts the carrier cannot read back, so everything "
                         "priced between measured and N takes leg 1 on P and a leg-2 store read the store "
                         "must refuse -- the '#915 PREFETCH REFUSED' / W16 shape of boot weg2ls4b2 (84,027 "
                         "tokens against a 30,518-token host pool). N AT OR BELOW the floor is a W45 "
                         "(operator_below_floor): such a bound bypasses the leg-1/leg-2 round trip for "
                         "EVERY prompt length, so group P runs zero prefill passes (boot weg2rg5 shipped "
                         "17 and exited 0). 0 falls in that arm, and 0 is NOT an off switch for the "
                         "round trip -- there is no flag that is, because the front's two carrier guards "
                         "(front.Front.handle_generate and front.Front.leg1) read 'carrier_max_tokens > 0', "
                         "so 0 removes the CARRIER-EXCEEDS BYPASS and leaves the store read unbounded. And "
                         "when the census measured NOTHING (verdict missing or disagree) there is no bound "
                         "to lower, so the flag is refused too (operator_without_measured_bound): a refusal "
                         "about the MEASUREMENT is fixed by fixing the census, not by typing a number. (A "
                         "census whose measured bound is itself at or below the floor leaves the interval "
                         "empty, so there every N is refused by one of the two checks above.) The census is "
                         "taken and logged on both paths, so the log always carries the measured number "
                         "beside the shipped one.")
    ap.add_argument(
        "--draft-kv-on-p", choices=["on", "off"], default=DRAFT_KV_ON_P_DEFAULT,
        help="#1264. Whether group P is the draft-KV-only PRODUCER. 'on' (the "
             "default, and the standing user order of 2026-09-07: draft KV "
             "across the flip) puts the checkpoint's mtp.* head on P's last "
             "stage under " + " ".join(P_DRAFT_KV_FLAGS) + ", so D re-admits "
             "with a warm draft KV after a flip. 'off' boots the rg6-proven "
             "form: group P carries NO speculative flag and no MTP head, and "
             "the launcher says so in one line -- it is the serving-base / "
             "A-B arm, never silent. GROUP D IS UNCHANGED BY EITHER VALUE: it "
             "keeps its own NEXTN head in both forms, because 'off' removes "
             "the producer, not speculative decode. What follows the switch "
             "follows it through ONE predicate read off P's argv "
             "(ring_table.p_carries_drafter), not through a flag re-read at "
             "each consumer: the group-P FORM KEY (a blacklist, so the two "
             "forms hash apart by construction), the ring's per-stage drafter "
             "term (no MTP/embedding/lm_head bytes on the last stage under "
             "'off'), and the W10 drafter-identity / W11 draft-resident gates "
             "(which grade a producer that does not exist under 'off' and are "
             "SKIPPED with a named line rather than refusing the boot).")
    ap.add_argument(
        "--spec-form", choices=["NEXTN", "DFLASH"], default=SPEC_FORM_DEFAULT,
        help="PLAN_DFLASH2_P_0917. The speculative form of BOTH groups. NEXTN "
             "(default) is the shipping form: the checkpoint's mtp.* head, the "
             "constants SPEC_* above. DFLASH puts the external DFlash2 draft on "
             "both groups: P as the draft-KV producer (chunk ring, hash-keyed "
             "direct publish into the draft arena), D as the proposer with the "
             "compact draft cache (--dflash-window) and the window pool "
             "(FLLIPER_DFLASH_WINDOW_POOL=1 in D's environment). The draft's bytes "
             "are priced off its own checkpoint headers (uneven_perf dflash_* "
             "families) so the D budget carries them.")
    ap.add_argument("--dflash-draft-path", default=DFLASH_DRAFT_PATH_DEFAULT,
                    help="DFLASH draft checkpoint directory (both groups, byte-identical flag).")
    ap.add_argument("--dflash-block", type=int, default=DFLASH_BLOCK_DEFAULT,
                    help="DFLASH block size = --speculative-num-draft-tokens (8 for DFlash2).")
    ap.add_argument("--dflash-window", type=int, default=DFLASH_WINDOW_DEFAULT,
                    help="D's compact draft cache window = --speculative-draft-window-size "
                         "(the draft's sliding_window, 2048 for DFlash2).")
    ap.add_argument(
        "--dflash-produce-on-p", choices=["off", "on"], default=DFLASH_PRODUCE_ON_P_DEFAULT,
        help="--spec-form DFLASH only (27B user decision 2026-09-24). 'off' (the default, "
             "the standard form, p_draft=cold): group P keeps loading the DFlash2 draft on its "
             "last PP stage -- the bytes stay cold-resident, so VRAM, the planner cut, the "
             "exchange census and the flip are unchanged -- but computes NOTHING with "
             "it: no aux-hidden capture on any PP stage, no DFlashDraftKvProducer."
             "produce(), no publish_draft_rows_direct. Group D is unchanged and finds "
             "no draft pages (its existing cold path). 'on' (p_draft=compute) is the old "
             "producer form, A/B only. UNIFY S6: an alias of the p_draft axis, resolved by "
             "pdflip/form.py (--form-p-draft cold|compute drives it). The value reaches group "
             "P's ranks through its environment (" + DFLASH_PRODUCE_ENV + "=0/1, "
             "spec_form_env('P')); argv_p and the P FORM key are byte-identical under both "
             "values, and the launcher names the form in one line (PDFLIP DFLASH-PRODUCE-ON-P).")
    ap.add_argument(
        "--d-replayssm-spec", choices=["off", "on"], default=D_REPLAYSSM_SPEC_DEFAULT,
        help="27B ReplaySSM package. DEFAULT: the booted profile's registry row "
             "(pdflip/form.py replayssm: qwen27b and nextflash on, proven on metal); "
             "the parser value 'off' is the code default without a row. "
             "'on' gives group D --enable-linear-replayssm-spec "
             "(+ --linear-replayssm-cache-len, a power of two >= 16 and >= the draft "
             "window): D's GDN target verify runs the compact spec ring instead of "
             "per-draft intermediate states and the freed 'speculative intermediate "
             "state' post goes to D's KV pool; the launcher's D pricing follows the "
             "same switch. 'off' leaves argv_d byte-identical. Group "
             "P never gets it (P runs no target verify). NF line (H64): the draft "
             "window is group D's --speculative-num-draft-tokens as --extra-d ships "
             "it (the Next-Flash MTP runs 4 over the constants' 3); the D planner "
             "(#145 FRACTION-SOLVE and KARTE D) books the ring's post and verify "
             "allocation instead of the recurrent references' -- D's KV is capped at "
             "the context there, so the freed bytes reach the expert-row edge.")
    ap.add_argument(
        "--p-prefill-graph", type=int, default=P_PREFILL_GRAPH_DEFAULT,
        metavar="TOKENS",
        help="Group P prefill CUDA graph (27B line). 0 (the default) = off: "
             "argv_p, the P form key and every ledger input are byte-identical "
             "to a tree without this flag. N > 0 = P captures ONE full prefill "
             "graph of N tokens (--cuda-graph-config prefill backend 'full', "
             "bs [N], one request slot) AND runs its chunks at N tokens "
             "(--chunked-prefill-size N on group P only; D keeps its chunk) -- "
             "a bucket below the chunk would replay only the rest chunk. The "
             "capture pool lives in the memory saver's cuda_graph tag, so P's "
             "sleep releases it and the wake remaps it, like D's decode graphs. "
             "A shorter rest chunk pads up to N. Batches the graph cannot take "
             "(two requests, multimodal inputs, input logprobs) run eager and "
             "are named by the rank line 'PREFILL-GRAPH eager reason='. The "
             "capture cost per stage is the rank line 'PREFILL-GRAPH captured "
             "... capture_mib='. Changes P's form key (chunk + graph config "
             "are device allocations). User goal: 512.")
    ap.add_argument(
        "--p-prefill-graph-policy", choices=["auto", "on", "off"],
        default=P_PREFILL_GRAPH_POLICY_DEFAULT,
        help="Only with --p-prefill-graph (27B line, pdflip/p_graph_policy.py; user order "
             "26.09.: 'cuda graphen im prefill nur anschalten wenn es was bringt'). "
             "'auto' (default): per width (the main bucket and every "
             "--p-prefill-graph-buckets entry) the graph is captured only when the "
             "calibration table (--p-prefill-graph-calibration) puts its slowest P stage "
             "at least --p-prefill-graph-min-gain ahead of eager; a width without a "
             "measurement keeps the safe default -- main bucket graph, extra buckets "
             "eager -- so without a table the argv is byte-identical to before. Extra "
             "buckets must also pass the VRAM gate (the pinned cut's pool with their "
             "capture pool clears the pool floor). 'on': capture every configured bucket "
             "(the metal calibration arm). 'off': the pre-switch form, extra buckets "
             "refused. The chunk policy prices each width in the decided mode.")
    ap.add_argument(
        "--p-prefill-graph-buckets", default="", metavar="TOKENS[,TOKENS]",
        help="Only with --p-prefill-graph and --p-prefill-graph-policy auto|on: extra "
             "prefill graph buckets ABOVE the main one (e.g. 1024,2048 for the dynamic "
             "chunk ladder). Empty (default) = none. Captured into the same pool; the "
             "budget post prices them proportionally (upper estimate) unless "
             "--p-prefill-graph-pool-mib gives the measured capture_mib.")
    ap.add_argument(
        "--p-prefill-graph-calibration", default="", metavar="JSON",
        help="Only with --p-prefill-graph-policy auto: the calibration table "
             "{'ref_prefix': N, 'widths': {'512': {'graph_ms': [..], 'eager_ms': [..]}, ...}} "
             "(per P stage, the prefix-free ms of one forward; written by "
             "docker/graphcal_eval.py from a graph and an eager metal boot). Empty "
             "(default) = the 'graph_calibration' key of a --p-chunk-model JSON, else none. "
             "The table's 'pp_layer_ratio' (the P cut it was measured on) is compared with "
             "the cut this boot ships: different = PREFILL-GRAPH-CALIBRATION CUT-MISMATCH, "
             "the table is dropped and the cut solved again; absent (older tables) = a "
             "warning, the table is used.")
    ap.add_argument(
        "--p-prefill-graph-min-gain", type=float, default=0.01, metavar="FRACTION",
        help="Only with --p-prefill-graph-policy auto: the graph must be at least this "
             "much faster at the slowest stage (default 0.01).")
    ap.add_argument(
        "--p-chunk-sweep", default="", metavar="TOKENS[,TOKENS]",
        help="MEASUREMENT only, with --p-chunk-policy dynamic: no plan; the n-th request "
             "runs every chunk at the n-th width of this list (cyclic), so one boot "
             "measures every width on every prompt length (graph calibration ladder). "
             "Overrides --p-chunk-dynamic-min-tokens. Empty (default) = off.")
    ap.add_argument(
        "--d-token-placement", choices=["capacity", "bandwidth"], default=D_TOKEN_PLACEMENT_DEFAULT,
        help="Group D: where NEW KV tokens land (27B uneven DCP only; pdflip/d_token_placement.py, "
             "DYN_D_RESHARD.md sec. 13). DEFAULT: the booted profile's registry row "
             "(pdflip/form.py d_token_placement on its d_token_placement_formats: qwen27b "
             "bandwidth on the INT8 checkpoint, proven on metal); every other checkpoint and "
             "NF: 'capacity' = today, env byte-identical. "
             "'bandwidth' = the allocator interleaves its free slot ids so new tokens go to the "
             "ranks by effective bandwidth while the pool has room, by free capacity once a "
             "rank's class would pass --d-token-fill-switch; stored KV never moves. Refused "
             "for the NF profile.")
    ap.add_argument(
        "--d-token-shares", default="auto", metavar="W",
        help="Only with --d-token-placement bandwidth: per-rank weights (rank order), "
             "'auto' = the d_reshard fit 937,604,604 GB/s (5090 first).")
    ap.add_argument(
        "--d-token-fill-switch", type=float, default=0.85, metavar="FRACTION",
        help="Only with --d-token-placement bandwidth: fall to capacity placement once the "
             "bandwidth target would fill a rank's class beyond this fraction (default 0.85).")
    ap.add_argument(
        "--d-kv-evict-for-placement", choices=["off", "on"], default=D_KV_EVICT_DEFAULT,
        help="Group D, only with --d-token-placement bandwidth (pdflip/d_kv_evict.py, "
             "DYN_D_RESHARD.md sec. 14): before the fill would push the placement to "
             "capacity, cold radix leaves (no active reader, idle >= --d-kv-evict-min-idle-s) "
             "whose KV is already in L2 on every D rank leave the device copy-free "
             "(the class over its share first); cold un-backed decode tails are published "
             "through the write stream first. One small group collective per 16 ticks; "
             "the victim list is group-uniform, a disagreement is a crash-stop. "
             "'off' (default) = no env, unchanged.")
    ap.add_argument(
        "--d-kv-evict-min-idle-s", type=float, default=20.0, metavar="S",
        help="Only with --d-kv-evict-for-placement on: a radix leaf counts as cold after "
             "this many seconds without access (group-agreed clock). Default 20.")
    ap.add_argument(
        "--d-kv-evict-publish-max", type=int, default=8, metavar="N",
        help="Only with --d-kv-evict-for-placement on: cold un-backed leaves handed to the "
             "publish sweep per pass (0 = evict only what is already in L2). Default 8.")
    ap.add_argument(
        "--d-reshard", choices=["off", "wake", "wake-seg", "live"], default=D_RESHARD_DEFAULT,
        help="Group D weight shards per load class (27B TP>1 uneven TP + uneven DCP only; "
             "pdflip/d_reshard.py, DYN_D_RESHARD.md). 'off' (default) = today, argv and env "
             "byte-identical. 'wake' = the MLP family vector is chosen from "
             "--d-reshard-presets at each P->D wake; with ONE preset that is the static "
             "--rank-mlp-ratio on group D (bootable), with more the boot is refused until the "
             "executor is wired. 'wake-seg' = 'wake' whose 'auto' preset minimises the SEGMENT "
             "round (sum over layer segments of the slowest rank + floor, calibrated on "
             "rc9meas 26.09.) instead of the per-rank sums, over --d-reshard-depth; group D's "
             "D-SPEED table follows (FLLIPER_PDFLIP_DRESHARD_OBJECTIVE=segment). 'live' = refused "
             "(switch costs more than it gains). Refused for the NF profile (Form A has no "
             "shard vector).")
    ap.add_argument(
        "--d-reshard-depth", default="robust", metavar="TOKENS",
        help="Only with --d-reshard wake-seg and presets 'auto': 'robust' (default) = the one "
             "vector with the smallest worst loss against the per-point optimum over the "
             "measured ladder (bs1/bs2 x 10k/32k/128k/240k); a token count ('131072', '128k') "
             "= the best vector for that expected depth (bs1/bs2).")
    ap.add_argument(
        "--d-gc-freeze", choices=["off", "on"], default=D_GC_FREEZE_DEFAULT,
        help="Group D: 'on' = every D scheduler calls gc.freeze() once after boot (all CUDA "
             "graphs captured), so gen-2 collections no longer walk the boot objects (agent FR "
             "26.09.: ~0.5 s single-rank D-round stalls). 'off' (default) = no env from this flag; the profile rows (qwen27b, nextflash) turn the switch on by default (--env-d FLLIPER_PDFLIP_GC_FREEZE=0 = off).")
    ap.add_argument(
        "--p-gc-freeze", choices=["off", "on"], default=P_GC_FREEZE_DEFAULT,
        help="Group P: 'on' = every P scheduler calls gc.freeze() once after boot (all CUDA "
             "graphs captured), so gen-2 collections no longer walk the boot objects (NF "
             "13.10.: 4.4-5.4 s PP0/PP1 collections inside flips). 'off' (default) = no env "
             "from this flag; the profile rows (qwen27b, nextflash) turn the switch on for P and D by default "
             "(--env-p/--env-d FLLIPER_PDFLIP_GC_FREEZE=0 = off there).")
    ap.add_argument(
        "--d-reshard-presets", default="auto", metavar="SPEC",
        help="Only with --d-reshard wake: 'auto' (= dec), desk names 'dec,pf,rc9', or "
             "explicit MLP unit vectors 'name=u0:u1:u2;name2=...' (sum = the MLP units, "
             "1088 INT8 / 136 NVFP4). The first preset is the boot preset.")
    ap.add_argument(
        "--d-reshard-min-gain", type=float, default=0.02, metavar="FRACTION",
        help="Only with --d-reshard wake: a different preset is taken at a wake only when "
             "predicted at least this much faster than the one in force (default 0.02).")
    ap.add_argument(
        "--p-chunk-policy", choices=["fixed", "dynamic"], default=P_CHUNK_POLICY_DEFAULT,
        help="Group P prefill chunk width (27B and NF line, pdflip/p_chunk_policy.py). "
             "DEFAULT: the booted profile's registry row (pdflip/form.py chunk.policy on its "
             "chunk.default_formats: qwen27b dynamic on the INT8 checkpoint, proven on metal "
             "with --p-prefill-graph 512), where the other flags make it runnable (a P "
             "prefill graph bucket <= --p-chunk-max); every other case and NF: 'fixed'. "
             "'fixed' = today: every forward "
             "takes --chunked-prefill-size; argv, env and the P form key are "
             "byte-identical. 'dynamic' = a per-request plan priced on a per-stage "
             "time model as a pipeline (flow shop): a middle size from the ladder "
             "--p-chunk-min * 2^k .. --p-chunk-max, optional doubling head ramp and "
             "halving tail ramp; the fixed plan wins unless another is predicted "
             "--p-chunk-min-gain faster. Group P's --chunked-prefill-size becomes "
             "--p-chunk-max (the planner prices the corridor at it); chunks above the "
             "largest prefill graph bucket run eager. Rank lines 'P-CHUNK-POLICY "
             "armed' / 'P-CHUNK-POLICY plan', launcher lines with the dry-run plans.")
    ap.add_argument(
        "--p-layer-split", choices=["static", "dynamic"], default=P_LAYER_SPLIT_DEFAULT,
        help="Group P stage boundaries (27B, pdflip/p_layer_split.py, DYN_LAYER_SPLIT.md). "
             "'static' (default) = today: the P cut is fixed; argv, env and the P form key "
             "are byte-identical. 'dynamic' = PP0 moves each boundary chunk by chunk "
             "inside [--p-layer-split-home, home + --p-layer-split-window] (upstream "
             "stage executes the head of its neighbour's home span from a mirror and "
             "writes KV/GDN state back; a rise pulls the prefix from home over the PP "
             "link; non-home cuts run eager, the home cut keeps its graphs; the planner "
             "stays on the home cut wherever it predicts < --p-layer-split-min-gain). "
             "Rank lines 'P-LAYER-SPLIT runtime/slab/plan/census', 'PREFILL-GRAPH eager "
             "reason=layer_split'.")
    ap.add_argument(
        "--p-layer-split-home", default="", metavar="CUT",
        help="Only with --p-layer-split dynamic: the home cut as stage ends, e.g. 41,53.")
    ap.add_argument(
        "--p-layer-split-window", default="", metavar="W",
        help="Only with --p-layer-split dynamic: per boundary, how many head layers of "
             "the downstream stage the upstream stage may execute, e.g. 7,3.")
    ap.add_argument(
        "--p-layer-split-max-attn", default="", metavar="A",
        help="Only with --p-layer-split dynamic: per boundary, the most attention "
             "layers inside the window (each costs a pool-shaped KV mirror).")
    ap.add_argument(
        "--p-layer-split-model", default="", metavar="SRC",
        help="Only with --p-layer-split dynamic: SG's per-layer-type stage model "
             "(pdflip/p_stage_model.py), a JSON path or a name under p_stage_model_data/ "
             "(27b_int8_rc9j, 27b_nvfp4_rc9j).")
    ap.add_argument(
        "--p-layer-split-pull-gbps", type=float, default=None, metavar="GBPS",
        help="Only with --p-layer-split dynamic: the link rate the planner prices a "
             "PREFIX PULL at (a cut RISING mid-request pulls the KV prefix / GDN state "
             "of the newly swung layers from home). Default "
             f"{P_LAYER_SPLIT_PULL_GBPS_DEFAULT} (the x4-3080 edge); 0 = V1 (rises only on a "
             "fresh request's first chunk, no pull).")
    ap.add_argument(
        "--p-layer-split-min-gain", type=float, default=0.01, metavar="FRACTION",
        help="Only with --p-layer-split dynamic: a moving cut is taken only when "
             "predicted at least this much faster than the home cut (default 0.01).")
    ap.add_argument(
        "--p-chunk-max", type=int, default=P_CHUNK_MAX_DEFAULT, metavar="TOKENS",
        help="Only with --p-chunk-policy dynamic: the widest chunk (default "
             f"{P_CHUNK_MAX_DEFAULT}); becomes group P's --chunked-prefill-size.")
    ap.add_argument(
        "--p-chunk-min", type=int, default=0, metavar="TOKENS",
        help="Only with --p-chunk-policy dynamic: the smallest planned chunk (a "
             "final rest may be shorter). 0 (default) = --p-chunk-fixed.")
    ap.add_argument(
        "--p-chunk-fixed", type=int, default=0, metavar="TOKENS",
        help="Only with --p-chunk-policy dynamic: the fixed baseline the plan must "
             "beat. 0 (default) = the fixed policy's chunk (the --p-prefill-graph "
             "bucket, else 4096). With --p-chunk-min = --p-chunk-max = this value the "
             "width is FORCED (a metal probe of one chunk size).")
    ap.add_argument(
        "--p-chunk-model", default=P_CHUNK_MODEL_DEFAULT, metavar="SRC",
        help="Only with --p-chunk-policy dynamic: the per-stage time model. "
             "'builtin-int8' (default; the weg2rc7c #PGAP fit, INT8 cut 42/11/11), "
             "'builtin-fp8' (the W8A16-Marlin fp8 checkpoint, boot dkr27bfp8bar1drei10271317 cut 44/11/9, "
             "512-token forwards only: use with --p-chunk-mscale flat), "
             "'fit:<P.log>' (fit the #PGAP lines of that boot's P log, prefixes "
             "[0,32k)), or a JSON file {\"stages\": [StageModel JSON, ...]}.")
    ap.add_argument(
        "--p-chunk-mscale", choices=sorted(P_CHUNK_MSCALES), default=P_CHUNK_MSCALE_DEFAULT,
        help="Only with --p-chunk-policy dynamic and a builtin/fit model: the "
             "per-token cost curve over the chunk width per stage ('int8' / "
             "'nvfp4': the 5090 kernel-bench HOCHRECHNUNG on PP0, 3080 stages flat "
             "until measured; 'flat': no kernel gain anywhere).")
    ap.add_argument(
        "--p-power-scale", choices=list(P_POWER_SCALE_LAWS), default=P_POWER_SCALE_DEFAULT,
        help="Only with --p-chunk-policy dynamic (release table row 27): the chunk model "
             "carries the power limit per card it was CALIBRATED under (power_limit_w; "
             "builtin-int8 = the rig limits 5090 400 W / 3080 230 W; fit:<P.log> = that "
             "log's POWER-LIMIT rank lines). A card whose running limit (NVML) is more "
             "than 5%% away is ALWAYS warned about loudly. 'off' (default) uses the model "
             "as calibrated; 'linear' / 'power' rescale its COMPUTE terms by "
             "(P_cal/P_now)^alpha (alpha 1 / --p-power-scale-exponent) -- a MODEL, not a "
             "measurement; bandwidth and host-launch terms stay unscaled. The only honest "
             "fix for a changed limit is a new measurement.")
    ap.add_argument(
        "--p-power-scale-exponent", type=float, default=None, metavar="ALPHA",
        help="alpha of --p-power-scale power (default 1/3: f ~ P^(1/3) on the DVFS curve).")
    ap.add_argument(
        "--p-chunk-grid", type=int, default=0, metavar="TOKENS",
        help="Only with --p-chunk-policy dynamic: no chunk crosses an absolute "
             "multiple of TOKENS (0 = off, the 27B default: its P anchors are "
             "distance-based). The NF line sets 4096.")
    ap.add_argument(
        "--p-chunk-dynamic-min-tokens", type=int, default=P_CHUNK_DYNAMIC_MIN_TOKENS_DEFAULT,
        metavar="TOKENS",
        help="Only with --p-chunk-policy dynamic: a request whose remaining prompt is at "
             f"most TOKENS (default {P_CHUNK_DYNAMIC_MIN_TOKENS_DEFAULT}) is not planned -- "
             "every forward gets the fixed width (--p-chunk-fixed), as under 'fixed'. "
             "rc9j measured 2k/8k slower under dynamic although the plan was 512xn. "
             "0 = plan every request.")
    ap.add_argument(
        "--p-chunk-min-gain", type=float, default=0.01, metavar="FRACTION",
        help="Only with --p-chunk-policy dynamic: a plan other than fixed is taken "
             "only when predicted at least this much faster (default 0.01).")
    ap.add_argument(
        "--p-attn-head-split", default=P_ATTN_HEAD_SPLIT_DEFAULT, metavar="SPEC",
        help="Group P attention split BY HEADS (27B, release table row 22; "
             "ATTN_HEAD_SPLIT.md phase 2). 'off' (default) = no env, argv and the "
             "pool model byte-identical. SPEC = owner:helper:n_layers:G|hA-B[,...]: "
             "stage OWNER hands the attention of its last G kv groups "
             "(G x 6 q heads), or of q heads A..B (whole groups or inside one "
             "group), of its last n_layers full-attention layers to stage "
             "HELPER. helper < owner = V1 (upstream helper runs the same rule), "
             "e.g. '2:0:2:1,1:0:1:1' = INT8 128k model optimum; helper > owner = "
             "V2 (downstream helper serves the owner's announcements, owner "
             "deadline + self-computing fallback), e.g. '0:1:4:h18-20,0:2:4:h21-23' "
             "= NVFP4 long (P_MICROBATCH_AH2.md). Cold single-request eager "
             "chunks only; KV of all groups stays in the owner's pool.")
    ap.add_argument(
        "--p-attn-head-split-min-w", type=int, default=P_ATTN_HEAD_SPLIT_MIN_W_DEFAULT,
        metavar="TOKENS",
        help="Only with --p-attn-head-split: the narrowest chunk that splits (must be "
             "above every captured prefill bucket, so a split chunk is eager).")
    ap.add_argument(
        "--p-attn-head-split-deadline-ms", type=float, default=20.0, metavar="MS",
        help="Only with a downstream (V2) --p-attn-head-split: how long past its own "
             "heads the owner waits for a helper's output before it computes the range "
             "itself and ends the split for that request (<= 0: wait, named stall).")
    ap.add_argument(
        "--p-attn-head-split-ab", choices=("off", "alt"), default="off",
        help="Only with --p-attn-head-split: 'alt' = every second new request runs "
             "unsplit (rank line 'AH-AB ... arm=split|base'), A/B inside one boot.")
    ap.add_argument(
        "--p-trim-end-anchor", action="store_true",
        help="Group P (27B line): take every front leg-1 prompt of N tokens as "
             "N-1, at the token level in P's intake. P's last regular chunk then "
             "ends at N-1 and its finish insert is the N-1 anchor D resumes from "
             "(D claims at most N-1 tokens and computes the last one itself), so "
             "the 1-token END-OF-PREFILL ANCHOR forward (xsn430/433: 35-40 ms per "
             "stage) never runs. prompt_tokens and the #1442 hand-off still carry "
             "all N; N<2, read outputs, sessions, input embeds and multimodal "
             "prompts keep today's split. Adds FLLIPER_PDFLIP_P_TRIM_END_ANCHOR=1 to "
             "group P only. Code default off (argv and env byte-identical); an UNSET "
             "flag takes the booted row (end_anchor=trim: qwen27b on since LS6 30.09.).")
    ap.add_argument(
        "--no-p-trim-end-anchor", dest="p_trim_end_anchor", action="store_false",
        help="Turn --p-trim-end-anchor off against the registry row's default (A/B arm).")
    ap.add_argument(
        "--fork-anchor-token", type=int, default=None, metavar="ID",
        help="FORK ANCHOR (27B line, pdflip/fork_anchor.py): the chat template's "
             "turn-start token id (<|im_start|> = 248045 on Qwen3.8-27B). Group P "
             "then cuts a leg-1 prompt before its generation prompt (the last such "
             "token among the final 16) instead of at N-1, so its END anchor also "
             "serves a sibling request that forks there (boot "
             "dkr27brc10bar1agent09261821: 9 of 78 D-direct requests fell back "
             "1.2k-4.7k tokens); group D reads the store up to the same cut and keeps "
             "its own prefill anchor at or below it. Needs --p-trim-end-anchor. Adds "
             "FLLIPER_PDFLIP_FORK_ANCHOR_TOKEN to both groups; default off = argv and "
             "env byte-identical.")
    ap.add_argument(
        "--turn-anchor-token", type=int, default=None, metavar="ID",
        help="TURN ANCHOR (NF, pdflip/turn_anchor.py): the chat template's turn-start "
             "token id (<|im_start|> = 248045 on Qwen3.8-27B and NF). Group P then "
             "also snapshots the recurrent state where the prompt's LAST message "
             "starts (a second extend track in the same forward, no extra forward) "
             "and inserts it as an anchor -- where the next turn and a client's side "
             "request fork (y3m pdflip-50-71: FETCH CAP lost=16 pages behind the "
             "END anchor). Adds FLLIPER_PDFLIP_TURN_ANCHOR_TOKEN to group P only; "
             "default off = argv and env byte-identical.")
    ap.add_argument(
        "--fp8-uniform-marlin", action="store_true",
        help="27B line, an FP8 checkpoint (Qwen/Qwen3.8-27B-FP8, block 128x128) on "
             "the flip: force the Marlin FP8 kernel on EVERY rank of both groups "
             "(FLLIPER_FORCE_FP8_MARLIN=1) so the 5090 holds the same byte layout "
             "as the 3080s (Marlin int32 [K/16,4N] + group scales) and the "
             "exchange can move bytes between them; the 5090 gives up its native "
             "block-FP8 GEMM for it. Marlin's per-card lock workspace is booked "
             "local_scratch by the exchange coverage and re-zeroed after every "
             "weights resume (weight_exchange.zero_local_scratch). Also a ModelOpt "
             "checkpoint (quant_method modelopt, e.g. RadixArk Qwen3.8-27B-NVFP4): "
             "its FP8 linears the same way, its NVFP4 linears through "
             "--fp4-gemm-backend marlin on both groups. Without it such a checkpoint "
             "under --pdflip-weight-source exchange is refused (W160). Inert on any "
             "other checkpoint; default off = argv and env byte-identical.")
    ap.add_argument(
        "--fp4-native-mixed", action="store_true",
        help="Backlog #38, together with --fp8-uniform-marlin on a ModelOpt NVFP4 "
             "checkpoint: the NVFP4 linears get --fp4-gemm-backend native-mixed on "
             "both groups instead of marlin -- ONE native NVFP4 byte layout on every "
             "rank (E2M1 [N,K/2] + 128x4-swizzled E4M3 block scales), the kernel "
             "chosen per rank (sm_120 native W4A4, sm_86 W4A8 on the INT8 tensor "
             "cores from the same bytes; FLLIPER_FP4_NATIVE_MIXED_SM8X=marlin opts into "
             "Marlin W4A16 with its content permuted in place around every flip -- "
             "BARRED since 26.09. as broken, diagnosis override "
             "FLLIPER_FP4_ALLOW_BROKEN_SM8X_MARLIN=1). "
             "Sets FLLIPER_FP4_NATIVE_MIXED_BOOT=1 for launcher and ranks (exchange "
             "class leaves L4, planner NVFP4 lane record L7). The FP8 "
             "linears stay on Marlin. Refused (W160) without --fp8-uniform-marlin "
             "or on a non-ModelOpt checkpoint. Default off = argv and env "
             "byte-identical.")
    ap.add_argument(
        "--p-prefill-graph-tiny", default="", metavar="TOKENS[,TOKENS]",
        help="Only with --p-prefill-graph (27B line). Empty (the default) = off: "
             "one captured bucket, argv byte-identical. TOKENS = extra, smaller "
             "prefill graph buckets, e.g. 16: the 1-token END-OF-PREFILL ANCHOR "
             "chunk every P prompt ends with (and rests up to TOKENS) replays "
             "that bucket instead of padding to the main one -- measured "
             "xsn430/xsn433: a padded 1-token replay costs 37/35/38 ms per "
             "stage, a real 512 chunk ~41/34/38. Captured into the same pool "
             "(largest first; smaller buckets reuse its blocks); the budget "
             "post adds the proportional estimate unless "
             "--p-prefill-graph-pool-mib gives the measured capture_mib. "
             "Changes P's form key (graph config).")
    ap.add_argument(
        "--p-prefill-graph-pool-mib", default="", metavar="MIB[,MIB,MIB]",
        help="Only with --p-prefill-graph: the 'prefill graph pool' budget post "
             "per P stage, MiB -- booked by the ranks "
             "(FLLIPER_KV_BUDGET_PREFILL_GRAPH_MIB) AND by the P cut's pool "
             "model (PhasePoolModel.prefill_graph_pool_mib), one vector for "
             "both. Empty (default) = the desk ESTIMATE "
             f"{P_PREFILL_GRAPH_POOL_MIB_PER_512:.0f} MiB per 512 graph tokens "
             "per stage; pass the measured 'PREFILL-GRAPH captured ... "
             "capture_mib=' values of a previous boot instead.")
    ap.add_argument(
        "--p-prefill-graph-max-prefix", type=int, default=None, metavar="TOKENS",
        help="Only with --p-prefill-graph: a P chunk whose prefix is LONGER than "
             "this many tokens runs eager instead of replaying the graph (rank "
             "census reason 'deep_split'), so deep chunks can take the eager "
             "KV-split prefill. Default: no threshold (every eligible chunk "
             "replays). Reaches the ranks as FLLIPER_PREFILL_GRAPH_MAX_PREFIX.")
    ap.add_argument(
        "--p-deep-split-from", type=int, default=P_DEEP_SPLIT_FROM_DEFAULT,
        metavar="PREFIX_TOKENS",
        help="Group P deep-chunk attention form (27B line). 0 (the default) = "
             "off: group P's environment is byte-identical. N > 0 = an EAGER P "
             "chunk whose prefix is >= N tokens plans its flashinfer prefill "
             "with a wave-aware KV split (the extend plan's fixed_split_size; "
             "rank line 'FI-WAVE-SPLIT'). With the prefill graph on, pair it "
             "with Agent H's --p-prefill-graph-max-prefix N (same N): the graph "
             "then declines those chunks (census reason 'deep_split') and they "
             "take this split -- the graph-mode plan cannot split. Reason, "
             "measured on xsn426/xsn428: at 512 tokens head_dim 256 gives 192 "
             "attention CTAs, two rounds on the 5090's 170 SMs. The 3080 stages "
             "find no gain and keep the stock plan. Crossover from the same "
             f"fits: ~{P_DEEP_SPLIT_FROM_FIT} (eager PP0 host launch ~57 ms "
             "against the graph's 41.9 + 1.432 ms per 1k prefix). Env names in "
             "layers/attention/fi_prefill_wave_split.py.")
    ap.add_argument(
        "--transport", choices=["bar1", "nccl"], default="bar1",
        help="Collective transport for BOTH groups. 'bar1' is the shipping "
             "default. 'nccl' is the DEVELOPMENT mode of the user's order of "
             "2026-09-07: it drops the barlink flag family so the stock "
             "flliper dispatch builds a PyNccl communicator, and it raises "
             "group D's dormant-residue slack 64 -> 192 MiB because NCCL's "
             "buffers are not memory-saver-tagged and survive the sleep "
             "(measured 2230 -> 2310 MiB on the 5090, boot weg2ab0). Under "
             "'bar1' nothing about the flip path changes.",
    )
    ap.add_argument(
        "--ring-form", choices=["auto", "none", "MAP_SHARED"], default="auto",
        help="C18/R16: the cross-process registration form for the shared host "
             "granule ring. MAP_SHARED (cudaHostRegister on a /dev/shm MAP_SHARED "
             "file) is the form the step-0 metal probe PROVED on this rig "
             "(WEG2_BUILD_DECISIONS_0906 section 1p); it is the only form built, "
             "and the memfd candidate is deleted because that probe retired it. "
             "'auto' (the default) arms it when the launch checks fund it, and "
             "otherwise prints the failing check by name (W32 / W34) and REFUSES "
             "the boot, exit 2 -- spec Amendment A1-3: the OLD flip form is "
             "infeasible on this host budget, so an un-armed ring has nothing to "
             "fall back to. 'none' never arms and therefore never launches. Never "
             "a size, never a knob: H and both spans are solved from the previous "
             "boot's own lines.",
    )
    ap.add_argument(
        "--xchg-coverage-diff",
        action="store_true",
        help="#1348 DIAGNOSTIC BOOT ONLY -- record, per rank, which lines of "
             "the exchange lane's modules this boot actually executed, and "
             "print the complement afterwards with "
             "scripts/pdflip/lane_coverage_diff.py. The complement is the list "
             "of seams nothing reached, i.e. the walls not yet found. "
             "OVERHEAD: coverage.py line tracing, which its own numbers put "
             "at 2x-10x ON THE TRACED MODULES ONLY (11 files, allowlisted -- "
             "not the whole tree); the measured per-leg cost of the dump "
             "itself is printed on every PDFLIP-COVERAGE DUMP line, never "
             "estimated. Default OFF and byte-identical off: the environment "
             "variable it publishes is POPPED when this flag is absent. Do "
             "NOT combine with FLLIPER_SEAM_COVERAGE_DIR -- two coverage "
             "tracers in one process measure neither, and the arm refuses by "
             "name if it sees it.")
    ap.add_argument("--evidence-dir", default=EVIDENCE_DIR,
                    help="where ring_table reads the previous boot's logs from")
    ap.add_argument("--duplex-probe", default=DUPLEX_PROBE_DEFAULT,
                    help="C12/C13 + A1-4: the step-0 probe RECORD the per-card PCIe "
                         "duplex ratio is solved from. A measured FILE, never a "
                         "number: the launcher parses its 'PROBE granule ... ratio=' "
                         "rows per card UUID, prints one PDFLIP-PCIE-DUPLEX line per "
                         "card with the R17 gate beside it, and publishes the table "
                         "to the ranks as FLLIPER_PDFLIP_PCIE_DUPLEX. An unreadable or "
                         "rowless file means NO card splits its lock key")
    ap.add_argument(
        "--pdflip-seam-digest", action="store_true",
        help="#1350: arm the SEAM GRADER on both groups. It answers one "
             "question -- 'did the round trip bring MY bytes back' -- by "
             "hashing this rank's own pieces at the last instant before the "
             "weights family is paused and again at the first instant after it "
             "has landed, and refusing by name (W90 PdFlipSeamDigestMismatch) "
             "when they differ. Rank-local: no peer, no collective, no byte "
             "moved between cards, and both moments AWAKE -- at flip time the "
             "sleeping group holds no weight bytes in VRAM at all, so a "
             "cutover-time compare is impossible rather than merely expensive. "
             "Keyed by the PLACEMENT identity the plan already publishes "
             "(param_name, class, tag, card, storage extents), so a mismatch "
             "NAMES the pieces that moved -- and so a change of arena, pointer "
             "or pitch, which the exchange makes by design, is not mistaken "
             "for a change of content. "
             "DEFAULT OFF and meant for an S6I INSTRUMENT boot: the hash costs "
             "a full read of the shard on every flip leg, so arming it on a "
             "serving boot is a flip-cost regression. Published to the ranks "
             "as FLLIPER_PDFLIP_SEAM_DIGEST, which is launcher OUTPUT and is "
             "POPPED when this flag is absent, so an inherited shell value "
             "cannot arm it. LIMIT: a green digest says the bytes came back, "
             "never that the layout was verified.")
    ap.add_argument(
        "--pdflip-disable-hicache", action="store_true",
        help="#1386: for the ZIELFRAGE MINIMALFORM (two manual flips + "
             "shadow compare, zero requests served) -- drop the "
             "HiCache/Mamba-anchor host tier that exists to reuse KV pages "
             "ACROSS requests, which a boot that serves none cannot do. "
             "ONE SWITCH, ONE TRUTH: this bool reaches BOTH "
             "`host_ledger.charge_terms` (anchors_gib/rings_gib/overhead_gib "
             "price at 0.00) AND `common_flags` (no --enable-hierarchical-"
             "cache, no --hicache-* flag on either group's argv) -- never "
             "two readers of two copies of the same decision, which is the "
             "class of defect that under-priced #1385's bounce buffer against "
             "the allocator xsn31/4 and xsn31/5 actually ran. "
             "DEFAULT OFF, and OFF is byte-identical to every boot before "
             "this flag existed: it is for the Minimalform's own use, not "
             "for the serving basis, and it is NOT a smaller "
             "--hicache-size/--hicache-mamba-host-mib -- those flags are "
             "omitted outright, not passed as 0, so no HiCache code path is "
             "entered at all.")
    ap.add_argument(
        "--pdflip-weights-cpu-backup",
        choices=list(weight_exchange.WEIGHTS_CPU_BACKUP_CHOICES),
        default=weight_exchange.WEIGHTS_CPU_BACKUP_AUTO,
        help="#1369: user order 2026-09-14 (\"DIE 48GB MUESSEN WEG... wenn es "
             "korrekt implementiert ist requires es NIEMALS einen Rueckfall\"). "
             "The WEIGHTS region's 46.40 GiB host ring "
             "(host_ledger.host_ring_gib) has backed both groups' weights "
             "unconditionally under every arm, including an ARMED exchange, "
             "where the exchange is already the weight source at the wake "
             "seam and the ring refills bytes nobody reads. 'auto' (default) "
             "drops the ring's --enable-weights-cpu-backup flag AND its "
             "host-ledger Sigma H term exactly when the exchange is armed "
             "(weight_exchange.exchange_armed()), and keeps both exactly as "
             "every pre-#1369 boot had them otherwise. 'on' forces the ring "
             "ON regardless of arm -- an A/B COMPARISON INSTRUMENT for one "
             "boot only, never a design claim. 'off' forces it OFF "
             "regardless of arm, the hard kill switch, including under "
             "'ring' -- for a boot that wants to prove the exchange needs no "
             "fallback at all. Published to both P and D exactly as "
             "--pdflip-xchg-inject is (weight_exchange.WEIGHTS_CPU_BACKUP_ENV "
             "via prepare_xchg_env/build_env), so a rank with no argv reads "
             "the same decision the launcher's own process resolved. An "
             "unrecognised value refuses by name (W107) rather than "
             "guessing which side of a 46.40 GiB ring a typo meant.")
    ap.add_argument(
        "--pdflip-weight-source", choices=WEIGHT_SOURCE_CHOICES,
        default=WEIGHT_SOURCE_DEFAULT,
        help="#1273: where a waking group's weight BYTES come from. 'ring' (the "
             "default, and today's path byte for byte) refills them from the "
             "shared host granule ring. 'exchange' moves them card-to-card from "
             "the sleeping group's still-mapped pages, which removes the ring's "
             "whole Sigma H from host RAM and pays for it with a VRAM peak while "
             "both groups' bytes are resident on one card. In THIS slice (S7) "
             "the flag arms one thing only -- the W71 residency check on that "
             "peak, before either group starts. "
             "TODO(S7->S6): S6 owns propagating it into the two groups' argv, "
             "the request structs and the memory saver's region flag; until "
             "then 'exchange' arms the gate and changes no rank behaviour. "
             "'shadow' (S5) is RING-AUTHORITATIVE: the ring refills the "
             "weights exactly as it does today and the exchange runs beside "
             "it into raw cudaMalloc buffers nothing reads, comparing what it "
             "pulled against what the ring restored (PDFLIP-XCHG-SHADOW).  It "
             "arms the region and the six-rank shadow gate; it can refuse "
             "ITSELF (W77, per card, against that card's free column) and it "
             "can refuse NOTHING ELSE.",
    )
    ap.add_argument(
        # #1358 THE ONE HOST LEVER FOR xsn28, AS A FLAG AND NOT A CONSTANT.
        # weg2xsn27 died in `shm,pinned`, and the #1358 reader attributed the
        # WHOLE 10.565 GiB of that region to five slot holders x 2.113 GiB from
        # one emit site. The buffer is `widest x assemble_slots(depth)` and
        # `slots = depth + 1` under shadow, so one step of depth is one widest
        # layer per holder: 721.3 MiB x 5 = 3.52 GiB of non-reclaimable, at the
        # sleep leg, in the region the boot died in.
        #
        # DEFAULT 2 IS BYTE-IDENTICAL to every boot so far -- the flag exposes
        # the constant, it does not move it. The value is PUBLISHED with the
        # rest of the bounce terms, so P, D and the host ledger read ONE
        # source: a second spelling of the depth is how `2.16 GiB` came to
        # have three causes (#1361 [23a]).
        #
        # THE COST IS NOT PRICED HERE, and that is deliberate: depth=1 leaves
        # one pipeline stage fewer, and nothing on this rig has measured what
        # that does to flip duration. Operator order 2026-09-13: measured at
        # the metal, never estimated.
        "--xchg-bounce-depth", type=int,
        default=xchg_bounce.ASSEMBLE_DEPTH_DEFAULT,
        help="#1358: depth-slots of the assemble buffer (default 2, "
             "byte-identical). `assemble_slots` adds ONE more under "
             "inject_mode=shadow, so the buffer is widest_layer x (depth+1) "
             "there. depth=1 saves one widest layer per slot holder -- "
             "measured on weg2xsn27: 721.3 MiB x 5 holders = 3.52 GiB of "
             "non-reclaimable at the sleep leg. The value is published to the "
             "ranks and read back by the host ledger, so all three agree.",
    )
    ap.add_argument(
        # #1385 (Wand 11b): THE LANE-COUNT LEVER, as a flag and not a
        # constant, exactly the way #1358 exposed depth. xsn31/3 measured
        # `lanes=5 source=measured` and `xchg_bounce=15.75` GiB (5 lanes x
        # 3.00 GiB + 0.75 GiB staging) at the sleep leg the boot dies in --
        # 13.50 GiB more non-reclaimable than xsn31/2's 78.90 GiB peak, which
        # is exactly the growth the cushion floor (1.50 GiB) has no room for
        # (cushion measured 0.20, W98 PdFlipHostRateLatched). ALL FIVE lanes are
        # pair/diagonal buffers of the SAME boot, created by different ranks
        # (host_ledger.xchg_lane_count is BOOT-WIDE) -- this cap does not
        # change which lanes exist, only how many of their buffers may be
        # pinned in tmpfs AT ONCE; the rest are SERIALISED (named on
        # `PDFLIP-XCHG-LANES-CONCURRENT`, never a silent flip-time cost).
        #
        # DEFAULT UNSET (None) IS BYTE-IDENTICAL: `None` reaches
        # `xchg_bounce.resolve_lanes_concurrent` as "the flag was never
        # given" and resolves to the 0 sentinel, which prices every measured
        # lane exactly as every boot before this flag did. `0` is NOT the
        # default -- an operator who explicitly types
        # `--xchg-lanes-concurrent 0` is refused (W103) rather than silently
        # handed the "uncapped" behaviour their argv says the opposite of.
        "--xchg-lanes-concurrent", type=int, default=None,
        help="#1385: cap on how many of this boot's measured exchange lanes "
             "(host_ledger.xchg_lane_count, boot-wide) may hold their "
             "assemble buffer in tmpfs at once. Unset (default) is "
             "byte-identical -- every lane prices, exactly as before this "
             "flag. `--xchg-lanes-concurrent 2` on a 5-lane cut charges "
             "min(5, 2) buffers instead of 5 (measured on xsn31/3's "
             "checkpoint: 15.75 GiB falls to 6.75 GiB) and serialises the "
             "remaining 3 -- named and counted on PDFLIP-XCHG-LANES-CONCURRENT, "
             "because a lane that waits for an earlier one's collector to "
             "drain costs flip time the size line alone cannot show. A value "
             "below 1 is refused (W103 PdFlipXchgLanesConcurrentInvalid), "
             "never silently read as 'uncapped'.",
    )
    ap.add_argument(
        # #1397 OPTION 3 (DESIGN_option3_band_credit_0914.md section 9):
        # a CROSS lane's collector gate names a DIFFERENT rank than a
        # DIAGONAL lane's does (the design doc's own "why cross but never
        # diagonal" finding), so a cross lane may reuse a slot WITHIN one
        # tag under a per-band drain credit instead of sizing the whole
        # tag in -- the diagonal share stays at the #1374 floor,
        # unconditionally, because relaxing it would re-enter boot
        # weg2xsn30's circular wait.
        #
        # ARMING REDUCES TO THIS ONE BOOLEAN (design doc section 9.2): the
        # cross-lane COUNT (`n_cross_lanes`) is never a second flag here --
        # `xchg_bounce.resolve_cross_lanes` derives a SAFE, structural lower
        # bound from `n_lanes` (measured, #1358) and
        # `weight_exchange_region.N_CARDS` (a pigeonhole argument: at most
        # N_CARDS of any `n_lanes` active lanes can be diagonal, so the
        # remainder is a hard floor on how many must be cross, regardless of
        # WHICH specific lanes those are). A caller with a real, measured
        # split may still override it -- `bounce_terms(n_cross_lanes=...)` --
        # but no CLI flag exposes that override: this switch alone is the
        # whole arming surface, on purpose.
        #
        # DEFAULT OFF IS BYTE-IDENTICAL: `band_credit=False` makes
        # `n_cross_lanes_priced` 0 by construction (`BounceTerms.
        # n_cross_lanes_priced`), so `total_bytes` takes its pre-#1397
        # branch untouched -- no existing boot's price moves by a byte.
        #
        # NEEDS `max_tag_bytes > 0` (Option 1 / #1374) OR REFUSES: the
        # cross-lane price is the small ONCARD slot unit, which
        # `run_bounce_leg` only ever allocates under Option 1 -- pricing the
        # small number while Option 1 is absent would allocate the WIDEST-
        # LAYER-sized slot instead, the #1385-round-3 mismatch class
        # (measured 5.6x on boot weg2xsn31/4). `bounce_terms` itself refuses
        # this combination at ARM time, before either group starts.
        "--xchg-band-credit", action="store_true", default=False,
        help="#1397 (Option 3): let a CROSS lane (a directed card pair, "
             "never the diagonal) reuse a slot within one tag under a "
             "per-band drain credit, instead of sizing the whole tag into "
             "the buffer -- the diagonal share is unaffected. Default off, "
             "byte-identical. On a 5-lane cut with weight_exchange_region."
             "N_CARDS=3, the safe auto-derived cross count is 2 "
             "(xchg_bounce.resolve_cross_lanes); at slot=128 MiB/depth=2 "
             "this falls the pinned shm from 15.75 GiB (5 x 3.00 diag + "
             "0.75 staging) to 10.5 GiB (3 x 3.00 diag + 2 x 384 MiB cross "
             "+ 0.75 staging). Needs --xchg-bounce-oncard-slot-mib's Option "
             "1 (max_tag_bytes measured > 0) to price safely -- refuses "
             "(ValueError, xchg_bounce.bounce_terms) rather than silently "
             "pricing the small number while the leg allocates the large "
             "widest-layer one.",
    )
    ap.add_argument(
        "--pdflip-vision-source", choices=list(VISION_SOURCES), default=VISION_SOURCE_DISK,
        help="H125: where the TRANSIENT tower's bytes come from (only with "
             "--pdflip-vision transient). `disk` (default): the checkpoint shard, "
             "O_DIRECT, per image request. `ram`: P's PP0 stages the tower extent "
             "once into a tmpfs image (NF 856 MiB, 27B 879 MiB of host RAM for the "
             "life of the boot) and every stage reads it from RAM.",
    )
    ap.add_argument(
        "--pdflip-vision-place", choices=list(VISION_PLACES), default=VISION_PLACE_AUTO,
        help="H125: where the TRANSIENT tower sits on PP0's card (only with "
             "--pdflip-vision transient). `auto` (default): the KV tail when it is "
             "wholly free (the phase start), else the card's free VRAM, else a "
             "named refusal. `kvtail`: the 27B stage exactly. `free`: always the "
             "card's free VRAM. `weights` (VISION-WEIGHTS, not yet a profile default): "
             "the tower's parameters are views on victim WEIGHT memory of PP0 "
             "(27B flip: MLP storages; dual: only the hull parts D does not share); "
             "the victim bytes go to a host image (freed after every image) and come "
             "back with a device checksum before the admission -- a mismatch stops "
             "the group (W110c). No KV page, no free VRAM, always synchronous. "
             "`kvtail` or `auto` TOLD on the command line together with --dual-layout "
             "is refused (W111): the dual P pool is born trimmed to 0 tokens, so the "
             "dual's place is `weights`.",
    )
    ap.add_argument(
        # #1356: see VISION_OFF. Default `off` = P and D boot TEXT-ONLY.
        "--pdflip-vision", choices=list(VISION_CHOICES), default=VISION_OFF,
        help="#1356: `off` (default) boots both groups text-only by passing "
             "--no-enable-multimodal; `resident` keeps the vision tower loaded "
             "on P and D, which is what every boot through weg2xsn27 did "
             "(879 MiB per P card, 2.63 GiB across the host ring). NOTE: `off` "
             "CHANGES THE GROUP-P FORM KEY -- the flag is in the argv and is "
             "not in FORM_KEY_EXCLUDED_FLAGS, because a tower that is absent "
             "IS a different weight statement. The first boot of the text-only "
             "form therefore solves its ring from ITS OWN stem and W48 "
             "PdFlipRingFormMismatch is EXPECTED once, not W20: there is no "
             "same-form predecessor to inherit a census from, and inheriting "
             "the multimodal form's Sigma H would price 2.63 GiB of tower this "
             "form does not hold.",
    )
    ap.add_argument(
        "--pdflip-xchg-oncard-slot-mib", type=int,
        default=weight_exchange_transport.ONCARD_SLOT_MIB_DEFAULT,
        help="#1273 S6 fix E: the CEILING one diagonal slot may reach, in MiB "
             "(default 128 = byte-identical to every boot so far). The #1269 "
             "host ledger charges the WORST CASE, ONCARD_SLOTS_MAX x this "
             "value per card x 3 cards, at both the launch and the run moment "
             "-- 8 x 128 MiB x 3 = 3.00 GiB by default. That charge is what "
             "closed the last fundable rung for the 'host' arm at --d-bs 6 on "
             "this box (boot weg2shadowD D2: W20, every rung under the 6 GiB "
             "store floor). Lowering it to 64 charges 1.50 GiB and re-opens a "
             "rung; the price is more batches per lane and a higher priced "
             "hop, which the boot then MEASURES. The worst-case charge itself "
             "is deliberately NOT relaxed (operator decision 2026-09-09): a "
             "per-flip charge would fund the smallest flip and refuse none. "
             "Must be a whole multiple of the 32 MiB slot floor, else the "
             "launcher refuses by name (W82).",
    )
    ap.add_argument(
        "--pdflip-xchg-oncard", choices=ONCARD_MODE_CHOICES,
        default=ONCARD_MODE_DEFAULT,
        help="#1273 S6 (spec 3.7 degrade 3): how the CO-LOCATED pair of ranks "
             "on one card moves its diagonal. 'ipc' (the default) exports the "
             "producer's VRAM bounce to the consumer's process; 'host' routes "
             "it through a per-card pinned shm file instead, which is the only "
             "arm on which a STORE-AND-FORWARD deposit exists -- an exported "
             "VRAM bounce is freed with the exporting leg and cannot outlive "
             "it, so the source could not deposit and return. The value is "
             "published to both groups as FLLIPER_PDFLIP_XCHG_ONCARD and it is "
             "the SAME value the host ledger charges the deposit's pinned "
             "bytes on, so the arm a rank enforces and the arm that was paid "
             "for cannot diverge. On 'host' the ledger charges the S6-BOUNCE "
             "term -- the assemble buffer (widest layer x depth) PLUS path "
             "(a)'s staging (N_CARDS x 2 x --pdflip-xchg-oncard-slot-mib) -- at "
             "both moments, printed with its expression on the "
             "PDFLIP-XCHG-BOUNCE arm line, and the store shrinks by it. The "
             "retired per-card ceiling (8 x slot) is no longer charged or "
             "graded against: one payload, one term. Ignored on "
             "--pdflip-weight-source ring",
    )
    ap.add_argument(
        "--pdflip-xchg-inject", choices=list(weight_exchange.INJECT_CHOICES),
        default=weight_exchange.INJECT_SHADOW,
        help="Whether the S6-BOUNCE injection OWNS the weight bytes at the "
             "wake seam. 'shadow' (default) runs the bounce legs BESIDE the "
             "unchanged disk refill: the refill stays the authority, the ring "
             "stays, the ledger's host weights term prints unchanged, and the "
             "assembled bytes are compared byte-exact per descriptor against "
             "the refilled weights, one PDFLIP-XCHG-INJECT verdict per leg. That "
             "is what S6I boots first, because a transfer never graded against "
             "a known-correct copy of the same bytes is not evidence. "
             "'authoritative' replaces the refill -- host weights goes to 0.00 "
             "-- and a MISMATCH is a refusal, not a line. Published to both "
             "groups so the mode a rank enforces and the mode the boot was "
             "launched in cannot diverge. Ignored on "
             "--pdflip-weight-source ring",
    )
    ap.add_argument(
        "--pdflip-d-adopt", choices=("on", "off"), default="off",
        help="#108 ERSTBOOT-ADOPTION (user-order 22.09.): 'on' lets the "
             "D-Gruppe mit --load-format dummy starten und ihre Gewichte vom "
             "ERSTFLIP aus P's Karten holen, statt denselben Checkpoint ein "
             "zweites Mal von Platte zu lesen. Gemessen fnFL2w51: P 28,24 s, "
             "D danach nochmal 71,69 s fuer dieselben Bytes. Der groessere "
             "Gewinn ist aber die Rueckkopplung -- wer beim ERSTEN Laden den "
             "Flip-Pfad benutzt, sieht dessen Fehler in der ersten Minute "
             "statt nach der fuenften (acht Boots am 22.09. starben jeweils "
             "am Ende einer ~5-min-Ladephase). Zwei Riegel haengen daran, "
             "beide in pdflip/adopt.py: D schreibt nicht in den GETEILTEN "
             "Store (sonst ueberschreibt sein dummy-Presplit P's echte "
             "Experten) und rechnet keinen Forward, bis der Erstflip "
             "VOLLSTAENDIG gedeckt hat. 'off' ist byte-identisch zu vorher.",
    )
    ap.add_argument(
        "--pdflip-p-free-read-journal", choices=("on", "off"), default="off",
        help="BOOTZEIT 3 (29.09., 27B-Auflage): jede Lesung des freien Kartenspeichers "
             "durch P von 'P kv sized' bis zum ersten Wake als JSONL je Rang "
             "(p_free_reads_<tag>/), fuer den Vergleich frueher gegen seriellen Start "
             "(python -m flliper.srt.pdflip.d_early_start EARLY SERIAL). Mit "
             "--pdflip-d-early-start on immer an. Nur Instrument, aendert keine Entscheidung.",
    )
    ap.add_argument(
        "--pdflip-d-early-start", choices=("auto", "on", "off"), default="auto",
        help="BOOTZEIT 3 (29.09.): 'auto' (Default seit der Serie 29.09., dearly "
             "z30x2: serving 205 statt 226 s) = an, wo die Registry einen "
             "Metallbeleg traegt (d_early_start_proven: Next Flash), sonst aus "
             "(27B bis zu seinem Beleg-Boot). 'on' erlaubt W185 auf jedem Profil "
             "mit d_expect_from_p_records (NF, 27B). 'on' startet Gruppe D ZUSAMMEN mit P statt "
             "nach P-READY + sleep(P) + D-Plan (z30r3: D-Init 30 s, davor 8 s "
             "Schlaf+Plan). D plant aus den Erwartungs-Budgets (Planer-Sitz: "
             "Record statt DC_EXPECT) und wartet VOR seinem Gewichtsladen an "
             "einem Gate (pdflip/d_early_start.py). Nach sleep(P) misst der "
             "Launcher wie bisher und schreibt 'go' nur, wenn jedes geplante "
             "D-Budget <= dem gemessenen ist; sonst 'refuse', D beendet sich und "
             "startet seriell wie bisher. 'off' ist byte-identisch zu vorher.",
    )
    ap.add_argument(
        "--pdflip-xchg-legs", choices=list(weight_exchange.XCHG_LEGS_CHOICES),
        default=weight_exchange.LEGS_BOTH,
        help="#1330 B4n. WHICH DIRECTION's exchange legs run. 'both' (the "
             "default) is byte-identical to the behaviour before this flag "
             "existed. 'pp_to_tp' runs only the legs whose SOURCE is group P "
             "-- TP layers assembled out of the PP layout's card bytes, which "
             "is the first direction to be graded on metal against the host "
             "copy still in system RAM; 'tp_to_pp' is its mirror. A leg of "
             "the other direction is skipped BY NAME: every skip prints "
             "PDFLIP-XCHG-LEGS legs_skipped=<n> reason=direction-knob with both "
             "the leg's direction and the armed one, because a silently "
             "skipped leg reads in a log exactly like a leg that ran clean. "
             "Published to both groups so the directions a rank runs and the "
             "ones the boot was launched for cannot diverge. Ignored on "
             "--pdflip-weight-source ring",
    )
    ap.add_argument(
        "--pdflip-shadow-hop-bound-ms", type=float,
        default=weight_exchange_shadow.SHADOW_HOP_BOUND_MS_DEFAULT,
        help="#1273 S5b: the wall the SHADOW is allowed to price its on-card "
             "hop at before it refuses ITSELF for this leg (W77, scope=hop). "
             "The default is the spec's 20 ms diagonal target for the "
             "AUTHORITATIVE lane times a named factor of "
             f"{weight_exchange_shadow.SHADOW_HOP_BOUND_FACTOR:g} -- the "
             "observer runs on a card that is simultaneously carrying a real "
             "flip leg, so a bound equal to the authoritative target would "
             "refuse it for being an observer. It grades the PRICED hop, "
             "before any wall is spent; the MEASURED one is on the shadow "
             "line as shadow_ms= and is a finding, never a retro-active "
             "refusal. It can refuse only the shadow: the flip proceeds on "
             "the ring either way. Ignored on every arm but "
             "--pdflip-weight-source shadow",
    )
    ap.add_argument(
        "--pdflip-xchg-census", default="",
        help="#1273 S7: the per-card, per-tag, per-group census W71 prices the "
             "exchange's VRAM peak from -- a JSON FILE with its own provenance "
             "string, the same kind of measured input --duplex-probe already is, "
             "never a number on this command line. Required by "
             "--pdflip-weight-source exchange; absent, the launcher REFUSES by "
             "name (W71) rather than invent a table",
    )
    ap.add_argument(
        "--pdflip-xchg-census-map", default="",
        help="AP4 1526: name the census row that prices a live card the census does "
             "not carry by UUID: comma list of <live>=<census uuid>, <live> = a live "
             "card UUID, nvml<N> or a card class label (RTX3080). Accepted as the "
             "operator's assertion (one PDFLIP-XCHG-CENSUS-MAP line, no refusal). A live "
             "card with neither a census row nor a mapping is BORROWED from the "
             "heaviest row of its class (else of the census) and refused as HW-BORROWED "
             "unless --force. Ignored when every live card is in the census",
    )
    ap.add_argument(
        "--pdflip-xchg-census-foreign", action="store_true",
        help="27B line (weg2xsn441): accept a --pdflip-xchg-census that was MEASURED "
             "on another checkpoint than --model (named line PDFLIP XCHG-CENSUS "
             "FOREIGN). Without it such a census is refused in the dry run (W161): "
             "its tag bytes, dormant residue and flip peaks are the other "
             "checkpoint's. For the boot that measures this checkpoint's census. "
             "Default off; a census of the booted checkpoint prints nothing new.",
    )
    ap.add_argument("--ring-table-boot", default="",
                    help="pin the ring table to ONE boot instead of the newest usable "
                         "one. Matched as a SUBSTRING of the log stem, so the boot TAG "
                         "('weg2zr2') is enough; it must match exactly one boot, and a "
                         "pin that matches none or several returns the R22 reason "
                         "rather than raising an OSError -- and since A1-3 leaves no "
                         "feasible form to fall back to, that reason is a named "
                         "refusal (W20) and exit 2, verified on the rig")
    # -- group P: the layer cut ------------------------------------------
    ap.add_argument(
        "--pp-stage-ratio", default=None,
        help="OVERRIDE the solved layer cut for group P (e.g. '32,18,14'). "
             "Passing it is announced as 'PINNED (user override)' in the "
             "PP-CUT provenance line and priced on the same two axes as the "
             "solved cut. Unset = the solver decides. Next-Flash arm: this is "
             "the ONE cut variable, PP_RATIO in arm_fnFL2_long.sh (default "
             "29,11,8), with PP_ATTN_RATIO = the attention count that cut "
             "lands on (fnFL2 H57). The 'PP-CUT RATEN-SCHNITT' line prints the "
             "cut the measured stage rates recommend for the CURRENT power "
             "limit (or 'keine Raten fuer <Limit>, Schnitt bleibt gepinnt'); "
             "it is never applied.",
    )
    ap.add_argument(
        "--pp-attn-stage-ratio", default=None,
        help="OVERRIDE the solved full-attention split for group P (e.g. "
             "'8,4,4'). Same PINNED provenance. Unset = the attention axis is "
             "resolved for the chosen layer cut by pp_cut.best_attention_split.",
    )
    ap.add_argument(
        # THE DEFAULT IS `incumbent`, AND IT IS A MEASURED EXCEPTION TO THE
        # MAXKV LAW RATHER THAN A DISAGREEMENT WITH IT (boot weg2tr1,
        # 2026-09-08, BOOT_weg2tr1_0908.md).  The kv-floor solve 31,17,16 puts
        # SIXTEEN layers on PP2, and PP2 is the binding stage for a reason the
        # pool model cannot see: it carries the per-stage FIXED POSTS -- lm_head,
        # the MTP/draft head and the draft pools -- which are not terms in
        # PhasePoolModel at all.  Measured gap on that boot: the model priced
        # PP2 at 966,544 tokens against a PROFILED 155,164.  A cut chosen to
        # maximise a pool the model over-states by ~6x on the binding stage is
        # not the maxkv law's answer, it is the law applied to the wrong
        # number, and moving 16 layers onto that stage makes it worse.
        #
        # SO THIS IS A DEFAULT WITH AN EXPIRY, not a preference: when the pool
        # model carries the per-stage fixed posts (#1259 draft/MTP posts,
        # #1019 lm_head, #1260 joint draft-pool sizing) the kv-floor row becomes
        # trustworthy on PP2 and this default goes back to 'maxkv'.  Both other
        # arms stay PRICED on the PP-CUT SHIPPED line of every boot meanwhile,
        # so the trade this default declines is visible rather than hidden.
        "--pp-solve-objective", choices=["maxkv", "makespan", "incumbent"],
        default="makespan",
        help="#1254. WHICH PRICED CUT group P's layer split SHIPS. "
             "DEFAULT 'makespan' SINCE 2026-09-08, BY USER ORDER -- verbatim: "
             "'nimm als default ab jetzt makespan'. It is the solver's SPEED "
             "cut (42,11,11 / attn 10,3,3 in the perf boots) and it BUYS TTFT "
             "WITH KV: measured, makespan pays pool for prefill (+33.5 %% "
             "prefill measured in pp1/pp2 only; P pool 587k solver-priced "
             "against the incumbent's 714k measured). That trade is now "
             "SELECTED rather than inherited, and both alternatives stay "
             "priced on the PP-CUT SHIPPED: line of every boot so its cost is "
             "visible per boot instead of inferred later. "
             f"'incumbent' = the rg6-boot-proven contiguous cut "
             f"{P_PP_INCUMBENT_FMT}, looked up BY NAME in the solver's own "
             "ranked field, so its pool is priced by this boot's model and an "
             "incumbent the solver did not rank is a W40 REFUSAL rather than a "
             "silent substitution of the ranking's winner. IT WAS THE DEFAULT "
             "FROM weg2tr1 UNTIL 2026-09-08 as a "
             "MEASURED EXCEPTION WITH AN EXPIRY, not a preference: boot "
             "weg2tr1 showed PP2 is the binding stage because it carries the "
             "per-stage FIXED POSTS (lm_head, the MTP/draft head, the draft "
             "pools) that PhasePoolModel does not price at all -- it priced "
             "PP2 at 966,544 tokens against a PROFILED 155,164 -- and 'maxkv' "
             "would put SIXTEEN layers there. It returns to 'maxkv' once "
             "#1259/#1019/#1260 put those posts in the model. 'maxkv' = the "
             "kv-floor row, the pool-maximal cut that still clears the "
             "one-full-context-prompt floor, i.e. the standing law's answer "
             "against the pool model as it stands. 'makespan' takes the "
             "smallest compute+crossing total. It was the ORIGINAL default and "
             "was taken away in #1254 because it was paid WITHOUT BEING "
             "SELECTED -- the solver's own dry run chose 44,10,10 attn 11,2,3 "
             "at a 499,967-token pool, -47.7 %% of the pool for +33.5 %% of "
             "prefill, and nobody had picked that. It is the default again "
             "now because the user picked it, with both alternatives priced "
             "beside it every boot. ALL THREE arms are priced on the PP-CUT SHIPPED: line of "
             "EVERY boot with their pool AND their ms/chunk whichever is "
             "chosen, and the PP-CUT trade: line does the division, so a large "
             "trade is visible as the defect candidate the law calls it. "
             "TRAIN 2 reconciled this flag with the train's --p-cut-objective, "
             "which is GONE: two flags answering 'which cut ships' is the "
             "second-bookkeeping shape, and the flip-order map no longer needs "
             "the incumbent to agree with argv -- it is derived FROM the "
             "shipped cut (flip_order_split) and W46 checks that by "
             "construction. GAPPED MAPS ARE NOT AN ARM OF THIS FLAG: see "
             "--pp-layer-set and the #753 gate.",
    )
    ap.add_argument(
        "--pp-solve-pool-floor", type=int, default=None,
        help="#1286b. A HARD LOWER BOUND, in WORLD KV TOKENS, on the priced "
             "pool of whatever --pp-solve-objective ships. UNSET = the SHIPPED "
             "DEFAULT, which since the user order of "
             "2026-09-09 -- verbatim: '39,13,12 mit bs2 im pp layout soll "
             "standard werden vorerst' -- is the ORDERED CUT "
             f"{','.join(str(n) for n in DEFAULT_PP_ORDERED_CUT)} "
             "(DEFAULT_PP_ORDERED_CUT): the solver READS THE FLOOR OFF THIS "
             "BOOT'S OWN FRONTIER as that cut's priced pool (#1305), so "
             "'fastest above F' selects it exactly and the floor cannot drift "
             "off a moved frontier the way a typed constant did on boot "
             "weg2sn5pre; a cut that is not on the frontier is a W67 refusal, "
             "never a neighbour -- the cut stays SOLVED, "
             "this only bounds the pool. PASS 0 to turn the floor OFF and get "
             "the pre-#1286b behaviour exactly: the objective then ranks over "
             "the same feasible set it always did and nothing moves. SET to a "
             "positive number = the objective still ranks, but only over the "
             "cuts whose PRICED "
             "world pool clears this number -- so 'makespan' becomes 'the "
             "FASTEST cut that still holds N tokens'. It is a CONSTRAINT ON "
             "the one objective knob, not a second objective: the standing law "
             "is that trades live behind one knob, and 'fastest above N' is "
             "the same ranking over a smaller set. WHY IT EXISTS: #1286 "
             "repriced every candidate against the boot's own sizing formula "
             "and the makespan winner came out at 304,946 tokens against the "
             "incumbent's 715,089 -- 43 %% of the capacity for the speed cut -- "
             "and before this flag the only way to put a bound under that was "
             "to pin a cut by hand, which is the thing the solver replaces. "
             "The pool it is compared against is the SAME number "
             "PP-POOL-JOIN publishes and group P then sizes (within 0.1 %% on "
             "both reference boots), NOT a second estimate. NO SILENT "
             "FALLBACK: if no servable cut clears it the launch is a W40 "
             "REFUSAL that prints the frontier -- the fastest cuts that DO "
             "clear it and the best pool anywhere in the field -- so the next "
             "move is a number, not a re-solve by hand. Distinct from "
             "--max-kv-per-request, which is the PHYSICAL floor (one "
             "full-context prompt must fit); both are checked and each refusal "
             "names its own flag. Read the curve on the PP-CUT FRONTIER: line "
             "of any boot to choose a value.",
    )
    ap.add_argument(
        "--pp-cut-measured-ms-per-layer", default=MEASURED_MS_PER_LAYER,
        help=f"Per-stage prefill ms per layer for the cost model. Default "
             f"'{MEASURED_MS_PER_LAYER}' is MEASURED on boot bsscale "
             f"(/spinning/gpu-arb/weg2/BSSCALE_0907.md, tip 37c884b0b0, "
             f"per full 4096-token chunk at bs6: PP0 259.1 ms/32 layers, PP1 "
             f"632.9/18, PP2 470.2/14). Used as the ANCHOR when the measured "
             f"card-rate library supplies the per-card ratios.",
    )
    ap.add_argument(
        "--pp-cut-arming-floor-mib", type=float, default=ARMING_FLOOR_MIB,
        help=f"Per-rank arming floor subtracted before KV in the pool model. "
             f"Default {ARMING_FLOOR_MIB} = the top of the rig's VRAM "
             f"corridor (819-1229 MiB NVML-free per card under load). "
             f"pp_cut.PhasePoolModel names the gap this stands in for: the "
             f"solved per-layout floor only exists for a layout that has "
             f"booted, and the proxy carries about +-500 MiB.",
    )
    ap.add_argument(
        "--p-microbatch-depth", type=int, default=None,
        help="OVERRIDE group P's solved --pp-async-batch-depth (#692). Unset = "
             "DERIVED from the previous boot's own PP-BUBBLE line: a stage busy "
             "forward/(gap+forward) of the time needs ceil(1/(1-stall_share)) "
             "passes in flight to cover its gap, and the flag counts the passes "
             "BEYOND the one being forwarded. Priced against P's KV pool -- one "
             "extra pass costs one chunk of KV rows plus one crossing "
             "activation frame per stage -- and refused as W42 rather than "
             "lowered when the pool cannot fund it. Passing it is announced as "
             "PINNED and priced on the same axis.",
    )
    ap.add_argument(
        "--p-host-overlap", action="store_true", default=True,
        help="Group P: let each stage's host work for the NEXT prefill chunk run "
             "while the forward it just launched computes (managers/"
             "pdflip_p_overlap.py). Adds FLLIPER_PDFLIP_P_HOST_OVERLAP=1 and "
             "upstream's FLLIPER_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1 to group P "
             "only: a middle chunk exchanges no output (the last rank stops "
             "waiting on its own forward), the last rank fences its schedule "
             "stream device-side instead, and the chunked-prefill HiCache "
             "publish moves from the plan to right after the next launch. "
             "Measured reason (weg2xsn420, 4x98k): PP2 pass 873 ms vs 733 "
             "gpu-ms, ~95-140 ms card idle per 4096 chunk, flat over depth. "
             "DEFAULT ON since S43 (NF rc12 1018: PP0 duty 57 %% in its busy "
             "runs, 80 s of PP0 commit_ms on middle-chunk output exchanges "
             "alone -- the output ring, not L3; user rule 14.10. new features "
             "on); an UNSET flag takes the booted row (p_host_overlap: qwen27b "
             "on since LS6 30.09.). --no-p-host-overlap = the old argv/env, "
             "byte-identical.",
    )
    ap.add_argument(
        "--no-p-host-overlap", dest="p_host_overlap", action="store_false",
        help="Group P: the A/B arm of --p-host-overlap (on by default since "
             "S43; against the registry row's default): every chunk exchanges "
             "its output on the PP ring again, the chunked-prefill publish runs "
             "inline, no P-NOSYNC.",
    )
    ap.add_argument(
        "--p-recv-stream", action="store_true",
        help="Group P: receive each stage's proxy frame (hidden states from the "
             "stage before) on a side stream that does not wait for the running "
             "forward, so the transfer overlaps it; the next forward still waits "
             "for the frame device-side (managers/pdflip_p_overlap.py, "
             "FLLIPER_PDFLIP_P_RECV_STREAM=1). Measured reason (#PGAP 26.09., "
             "i8B): PP1/PP2 idle ~11 ms per 2048 chunk = the frame transfer, "
             "queued behind the fence on the previous forward. Default off = "
             "argv and env byte-identical to before.",
    )
    ap.add_argument(
        "--p-hostgap", action="store_true", default=True,
        help="Group P: the #PGAP instrument (FLLIPER_PDFLIP_P_HOSTGAP=1) -- one "
             "line per forward with the card's idle before it (timing events on "
             "the forward stream, harvested without a sync) and the host phases "
             "since the previous launch. For the --p-host-overlap A/B. DEFAULT "
             "ON since rc14 (NF 1018: the P pauses -- PP0 duty, card idle per "
             "chunk -- are the open class the next boot must measure; S43's "
             "metal check reads '#PGAP ... overlap=1'). Changes no decision. "
             "--no-p-hostgap = no #PGAP lines, env without the variable.",
    )
    ap.add_argument(
        "--no-p-hostgap", dest="p_hostgap", action="store_false",
        help="Group P: turn the #PGAP instrument (--p-hostgap, on by default "
             "since rc14) off -- FLLIPER_PDFLIP_P_HOSTGAP is not set.",
    )
    ap.add_argument(
        "--p-prefill-graph-split", type=int, default=P_PREFILL_GRAPH_SPLIT_DEFAULT,
        metavar="MAX_CHUNKS",
        help="Group P, only with --p-prefill-graph (27B line). 0 (the default) = "
             "off: the full prefill graph keeps flashinfer's own plan, env "
             "byte-identical. N >= 2 = the captured attention grid is sized for "
             "up to N KV chunks per q tile and every replay plans its own chunk "
             "count by the wave rule (rank lines 'FI-GRAPH-SPLIT armed' / "
             "'FI-GRAPH-SPLIT split ...'), in EVERY depth, no eager fallback "
             "needed. Why: flashinfer's graph plan assumes 2 CTAs/SM "
             "(scheduler.cuh:717, real occupancy 1 for fp8 KV hd256) and caps "
             "the grid at 85 work items per KV head on the 5090, so 48 q tiles "
             "are never split: 192 CTAs on 170 SMs, two rounds. 7 = 99 %% wave "
             "fill (192 x 7 on 170 SMs), 44 MB of the float workspace. "
             "layers/attention/fi_graph_split.py.")
    ap.add_argument(
        "--p-bubble-measured-from", default="",
        help="Path of the group-P log whose PP-BUBBLE lines feed "
             f"--p-microbatch-depth. Unset = the newest log in {EVIDENCE_DIR} "
             "that actually carries the instrument (a log without the line is "
             "skipped, never read as a measured zero). No measurement anywhere "
             "= depth 0, i.e. today's behaviour, and the line says so.",
    )
    ap.add_argument(
        "--pp-layer-set", default=None,
        help="OVERRIDE the solved stage map with an explicit per-stage LAYER "
             "SET, e.g. '0-2,4-6,...,60-62;3,7,...,31;35,...,63' (stages "
             "separated by ';'). THE ONLY WAY to configure the map: an "
             "inherited FLLIPER_PP_LAYER_SET in the launcher's environment is "
             "refused by name (W44), never honoured, because the launcher is "
             "the only writer of that variable and the solver's provenance "
             "line must describe the layout the boot actually runs. Unset = "
             "the solver enumerates gapped maps alongside contiguous cuts and "
             "publishes its own choice. Passing it is announced as PINNED and "
             "priced on the same axes.",
    )
    ap.add_argument(
        "--pp-cut-design-prefix-tokens", type=int, default=None,
        help="OVERRIDE the prefix depth the layout is optimised FOR. Unset = "
             "DERIVED from the newest P log carrying a prefill census (the "
             "mean of the accumulated per-chunk prefix on one rank); no census "
             f"anywhere = {DESIGN_PREFIX_FALLBACK_TOKENS}, and the line says "
             "FALLBACK. It matters because a full-attention layer's cost per "
             "chunk grows with the prefix while a GDN layer's does not, so the "
             "optimal placement of the 16 attention layers is a FUNCTION of "
             "this number, not a constant.",
    )
    ap.add_argument(
        "--pp-cut-stage-fit", default="", metavar="P_LOG|auto",
        help="27B line, default off (unset = the bsscale 4096 family split "
             "prices the P cut, as before). A group-P log with #PGAP lines "
             "(--p-hostgap), or 'auto' = the newest such log at THIS boot's P "
             "chunk size: every forward's gpu_fwd_ms is joined to its chunk's "
             "prefix and fitted per stage as a + b*(prefix+C/2)/1000 (device-"
             "bound chunks only), and the solver prices every candidate cut with "
             "the per-layer, per-attention-layer and per-forward costs that fit "
             "implies (planner/pgap_stage_fit.py). The pool floor is untouched. "
             "The fitted boot's power record (its front log's CARD-POWER line, "
             "else --pp-cut-stage-fit-power) must match this boot's metal, else "
             "W40 (--pp-cut-stage-fit-stale-ok prices with it anyway, marked "
             "STALE). The cut itself stays free: --pp-stage-ratio / "
             "--pp-attn-stage-ratio pin any P cut.")
    ap.add_argument(
        "--pp-cut-stage-fit-power", default="", metavar="W,W,W",
        help="Only with --pp-cut-stage-fit, for a fitted boot that printed no "
             "CARD-POWER record: the per-stage power limits it ran at, W "
             "(user measurement 24.09. ~18:55Z: 400,230,230). Compared with "
             "this boot's NVML limits like a printed record.")
    ap.add_argument(
        "--pp-cut-stage-fit-stale-ok", action="store_true",
        help="Only with --pp-cut-stage-fit: price with a fit whose card or "
             "power limit differs from this boot's metal (default: W40).")
    ap.add_argument(
        "--pp-cut-stage-model", default="auto", metavar="auto|off|NAME|PATH",
        help="PP-COST (27B line, 01.10.): the TIME axis of the P cut solver is priced by the "
             "per-layer-type, per-card stage model (pdflip/p_stage_model.py, calibrated from the "
             "'Prefill rank batch' + '#1469 RETAIN' lines) over the CHUNK MIX of the predecessor "
             "boot (the same P log the design prefix comes from, --pp-cut-design-prefix-from), "
             "instead of the card-rate/family split at one design prefix. 'auto' (default) = "
             f"profile qwen27b on its int8 checkpoint -> {PP_CUT_STAGE_MODEL_27B_INT8}, on its fp8 checkpoint -> "
             f"{PP_CUT_STAGE_MODEL_27B_FP8}; any other profile/format -> off. 'off' = the previous pricing, byte-identical. NAME = a file "
             "under pdflip/p_stage_model_data/ (without .json), PATH = a JSON. Why: the old pricing "
             "ranked 44,10,10 ahead of 43,11,10 while the rank lines measure 43,11,10 +4.5 %% faster "
             "with +15 %% P pool.")
    ap.add_argument(
        "--pp-cut-depth-profile", default="", metavar="ladder:N,N,...|fit",
        help="27B line, default off (unset = price every cut at the single "
             "design prefix, as before). 'ladder:2048,8192,32768' = each "
             "candidate's makespan is the mean over every chunk start of those "
             "prompt lengths, rungs weighted equally -- a max of lines is not "
             "the line of the mean, so a prompt ladder is priced on its own "
             "chunks; 'fit' = over the --pp-cut-stage-fit log's own chunks.")
    ap.add_argument(
        "--pp-cut-design-prefix-from", default="",
        help="Path of the P log whose prefill census feeds "
             f"--pp-cut-design-prefix-tokens. Unset = the newest log in "
             f"{EVIDENCE_DIR} that actually carries the census (a log without "
             "it is skipped, never read as a measured prefix of 0).",
    )
    ap.add_argument(
        "--pp-cut-expert-device-fraction", default="",
        help="Task #47/#48 (Next Flash): resident fraction of each P stage's MoE "
             "experts on the device (the P group's --rank-moe-resident-fraction), "
             "comma vector per stage. REQUIRED for a checkpoint with experts; the "
             "cut solver prices expert bytes at this fraction plus "
             "--pp-cut-expert-lru-rows rows per layer, and PLE tables at zero.",
    )
    ap.add_argument(
        "--pp-cut-expert-lru-rows", default="",
        help="Task #47/#48: LRU + staging rows of the expert pool per layer and "
             "stage (the P group's FLLIPER_MOE_SCRATCH_SLOTS), comma vector; "
             "default 0 rows.",
    )
    ap.add_argument(
        "--pp-cut-calibration-prefix-tokens", type=float,
        default=float(CALIBRATION_PREFIX_TOKENS),
        help=f"The prefix depth --pp-cut-measured-ms-per-layer was MEASURED at. "
             f"Default {CALIBRATION_PREFIX_TOKENS} is DERIVED: boot bsscale "
             f"drove a ~12,000-token prompt at chunk 4096, i.e. three full "
             f"chunks entering at prefix 0/4096/8192, whose mean is 4096 "
             f"(BSSCALE_0907.md). Half of the two-depth family split.",
    )
    ap.add_argument(
        "--pp-cut-attn-anchor-ms", type=float, default=ATTN_ANCHOR_MS,
        help=f"MEASURED cost of ONE full-attention layer per chunk at the deep "
             f"anchor prefix, ms. Default {ATTN_ANCHOR_MS} is the user's "
             f"physics note of 2026-09-07, recorded in the GAPPED block of "
             f"WEG2_BUILD_DECISIONS_0906.md section 1r. The other half of the "
             f"two-depth split: one measurement cannot separate a stage's "
             f"attention cost from its GDN cost, two at different depths can.",
    )
    ap.add_argument(
        "--pp-cut-attn-anchor-prefix-tokens", type=float,
        default=float(ATTN_ANCHOR_PREFIX_TOKENS),
        help=f"The prefix the deep anchor was measured at. Default "
             f"{ATTN_ANCHOR_PREFIX_TOKENS} = --context-length.",
    )
    ap.add_argument(
        "--pp-cut-mamba-mib-per-linear-layer-per-slot", type=float,
        default=P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT,
        help=f"Device GDN state per linear layer per sequence slot, MiB. "
             f"Default {P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT} = MEASURED "
             f"(#1286): the boot's own 'mamba state pool' post divided by "
             f"(linear layers x slots) on SIX rank readings across two boots "
             f"that ran different cuts -- 31.185/31.23/31.16 and "
             f"31.19/31.16/31.13 MiB per linear layer at 20 slots. It was 0.0 "
             f"= UNFUNDED until #1286, which inflated every stage's capacity, "
             f"more for stages holding more linear layers. 0.0 restores that "
             f"upper bound and is REFUSED at the launch seam.",
    )
    ap.add_argument(
        "--pp-cut-mamba-slots-per-running-request", type=int,
        default=P_MAMBA_SLOTS_PER_RUNNING_REQUEST,
        help=f"Mamba state slots ONE running request holds at once -- the "
             f"runtime's mamba_pool_floor.mamba_slots_per_running_req, which "
             f"is also the ceiling on _calculate_mamba_ratio "
             f"(model_runner_kv_cache_mixin.py:2684 takes the min of the two). "
             f"Default {P_MAMBA_SLOTS_PER_RUNNING_REQUEST}, the value boot "
             f"weg2sb5f emits as '[auto-mamba] ... ratio=2 ... -> "
             f"max_mamba_cache_size=20 slots'. The pool model charges "
             f"ceil(--p-bs x this x {P_MAMBA_AUTO_SAFETY_MARGIN}), which is a "
             f"PROVEN UPPER BOUND on _auto_mamba_demand_size (:2085-2106) "
             f"including its mamba_hard_floor term -- an upper bound on slots "
             f"over-charges mamba and therefore UNDER-prices the pool, the "
             f"conservative direction. It replaces a 2.5 fudge factor that "
             f"omitted the hard floor entirely (#1286 F4). VERIFY PER BOOT: "
             f"the '[auto-mamba]' line's slot count must be <= the "
             f"'PP-POOL-JOIN: ... mamba_slots=' this launcher publishes.",
    )
    ap.add_argument(
        "--pp-cut-stage-fixed-mib", default=P_PP_STAGE_FIXED_MIB,
        help=f"PER-STAGE non-layer weight and runtime state, MiB, comma list "
             f"(#1286). Default {P_PP_STAGE_FIXED_MIB} = MEASURED: the boot's "
             f"'weights + runtime state' post minus the per-layer half, "
             f"recovered INDEPENDENTLY from two boots that ran different cuts "
             f"(weg2sb5f 44,10,10 and weg2rg6 32,18,14) -- 2348.2/1100.0/"
             f"3515.6 against 2335.5/1111.2/3520.1 MiB, agreeing to <=12.7 "
             f"MiB, which is what makes it a per-STAGE constant and not "
             f"something that tracks layer count. It covers the embedding on "
             f"stage 0, lm_head plus the MTP/draft head on the last stage, the "
             f"replicated payload every stage carries, and the per-rank "
             f"runtime state. Empty = UNFUNDED and REFUSED at the launch seam.",
    )
    ap.add_argument(
        "--pp-cut-activation-reserve-mib", type=float, default=None,
        help=f"The boot's 'prefill activation reserve' post, MiB per rank "
             f"(#1286). Unset = max({P_PREFILL_ACTIVATION_RESERVE_MIB:.0f} MiB, "
             f"the 1.000 GiB both reference boots measured at chunk 4096, and "
             f"the MEASURED prefill transient of the binding P stage for the "
             f"chunk P boots with: support points "
             f"{list(P_PREFILL_TRANSIENT_SUPPORT.chunks)} per stage, linear "
             f"between, W131 above, #114/H41). A number pins the post.",
    )
    ap.add_argument(
        "--pp-cut-corridor-holdback-mib", type=float, default=None,
        help=f"The boot's 'gapped corridor holdback' post, MiB per rank "
             f"(#1286). Default: FOLLOWS --user-reserve-mib, because that post "
             f"IS the user reserve at the runtime end "
             f"(model_runner_kv_cache_mixin._gapped_corridor_holdback) and the "
             f"reserve defaults to 0 since #1257c. Both reference boots "
             f"measured {P_CORRIDOR_HOLDBACK_MIB} = 1.000 GiB per rank, which "
             f"is exactly what this returns at --user-reserve-mib 1024; a "
             f"pinned {P_CORRIDOR_HOLDBACK_MIB} on a reserve-0 boot would "
             f"over-charge group P by a gibibyte PER RANK against a runtime "
             f"that charges 0.000. Pass a number to pin it. This REPLACES "
             f"--pp-cut-arming-floor-mib in the pool model's subtraction: the "
             f"runtime's post list charges this holdback and no arming floor "
             f"(the #676 floor is held back after the profiler and already "
             f"sits inside the recovered reserve), so the old 1229 MiB "
             f"stand-in was over-charging by 205 MiB per rank -- the one term "
             f"of the four that erred the SAFE way.",
    )
    # -- group D: the decode knobs ---------------------------------------
    ap.add_argument(
        "--num-continuous-decode-steps", type=int,
        default=D_NUM_CONTINUOUS_DECODE_STEPS,
        help=f"Group D only. Decode steps run per scheduler visit "
             f"(server_args.py:1424). Default "
             f"{D_NUM_CONTINUOUS_DECODE_STEPS}: "
             # ONE sentence, three readers -- see D_DECODE_STEPS_PROVENANCE.
             # argparse renders every help as `help % params`, so the percent
             # signs are escaped HERE rather than kept doubled in the constant,
             # which the log line prints verbatim.
             + D_DECODE_STEPS_PROVENANCE.replace("%", "%%")
             + " Pass 1 to restore the shipped value.",
    )
    ap.add_argument(
        "--user-reserve-mib", type=str, default="0",
        help="#1257c. EXTERNAL headroom, MiB PER CARD: VRAM left free for YOUR "
             "OTHER PROCESSES while this boot serves. A scalar applied to "
             "every card, or a comma-separated list in RANK/CUDA-ORDINAL "
             "ORDER -- the 5090 FIRST, then the 3080s by NVML index. That is "
             "NOT NVML order: nvml0 is a 3080 while ordinal 0 is the 5090, so "
             "the standing form on this rig is `1800,1400,1400`. Every boot "
             "prints a PDFLIP-USER-RESERVE PROVENANCE line giving the argv "
             "position, card NAME and UUID beside each MiB value, so the "
             "mapping never has to be inferred from this sentence (#1342). "
             "Default 0 (user decision 2026-09-09): the 1024 this used to "
             "default to existed only because the engine could not price its "
             "own transient, and it now can -- the corridor floor is "
             "`measured transient peak of the awake group + THIS value`, per "
             "card, printed with its provenance on every CORRIDOR-FLOOR line. "
             "THE KNOB IS KEPT, deliberately, because the operator sometimes "
             "runs other VRAM consumers beside the rig. A nonzero value here "
             "raises that card's corridor floor by exactly as much, lowers its "
             "--rank-gpu-memory-mib by exactly as much, and MAKES THE CORRIDOR "
             "PASS ACTUATE on that card even when nothing measured the "
             "transient -- an explicit reserve is a priced floor.",
    )
    ap.add_argument(
        "--corridor-budget-sample", default=corridor_budget.DEFAULT_SAMPLE_PATH,
        help="#1257. PATH to the paired corridor/budget measurement group D's "
             "budgets are corridor-constrained against -- not an on/off switch: "
             "the 819-1229 MiB free-per-card corridor is the law for the Weg-2 "
             "form, so the constraint is always on and only its INPUT is "
             "selectable. The file pairs, for one boot, each card's NVML total, "
             "the --rank-gpu-memory-mib it ran with, the dormant image of the "
             "sleeping group, the driver carve-out, the front sampler's free "
             "column idle AND under load, and that rank's profiled token "
             "capacity; from those the pass derives the two terms the budget "
             "line does not book (unbudgeted awake residue, load transient) and "
             "the world pool C = min_r(P_r // v_r) * sum(v). A missing or "
             "incomplete file -- or one whose rows do not pair with this "
             "boot's card ordinals -- is a REFUSAL TO PRICE "
             f"({corridor_budget.UNPRICED_NAME}), not a default: the budgets "
             "then ship byte-identical and the log says so. The code is "
             "interpolated from the module constant on purpose; a hand-typed "
             "one here is exactly how front.py's 'W52 is free' comment "
             "outlived the number it described.",
    )
    ap.add_argument(
        "--d-tp-objective", choices=list(D_TP_OBJECTIVE_CHOICES),
        default=D_TP_OBJECTIVE_DEFAULT,
        help=f"Group D only (#1017). WHICH OBJECTIVE group D's weight vector "
             f"is solved for. Default {D_TP_OBJECTIVE_DEFAULT!r} SINCE THE "
             f"USER ORDER OF 2026-09-09, verbatim \"der decode bs6 soll mit "
             f"bs6 (nicht mehr bs4) der standard werden\": the bs=6 operating "
             f"point, whose vector is the measured gemm_tflops ratio. Boot "
             f"dec2c priced that vector ([42,11,11]) at +10.3%% at bs4 for "
             f"-1.4%% pool, which is the trade the order makes -- the ranking "
             f"a boot follows is the ranking AT THE OPERATING POINT IT RUNS, "
             f"and group D runs decode. The maxkv law is therefore outranked "
             f"here, not repealed: 'maxkv' remains fully selectable and emits "
             f"the same --rank-tp-ratio auto it always did (byte-identical to "
             f"the pre-#1241 form) -- the capacity-first split, weights "
             f"proportional to the per-rank VRAM budgets, which maximizes the "
             f"KV pool and is deliberately independent of how fast the cards "
             f"are (server_args.py:838). 'speed' emits "
             f"--rank-tp-ratio auto-performance --rank-perf-tune "
             f"<--d-rank-perf-tune>, the runtime's own per-task optimizer. "
             f"'decode-bs1' and 'decode-bs6' (#1241) add the OPERATING-POINT "
             f"axis to the same solve and are the only two arms that emit an "
             f"EXPLICIT --rank-tp-ratio vector: the bs=1 arm's vector is the "
             f"measured membw_gbs ratio, the bs=6 arm's is the measured "
             f"gemm_tflops ratio -- both from this rig's card-rate library, by "
             f"card NAME, with no hand number anywhere in the derivation. "
             f"BOTH REGIME PREMISES ARE STILL UNPROVEN ON THIS RIG, AND ONE "
             f"OF THEM IS NOW THE DEFAULT: that a bs=1 round is "
             f"bandwidth-bound and a bs=6 round compute-bound is "
             f"exactly the quantity the #1241 decode compute/wait clock "
             f"measures. Boot dec2c (21c46b1876) DID run that clock and "
             f"withheld the split on 13,893 of 13,950 rounds "
             f"(graph-replay-nodes-overwritten), so it measured the rounds "
             f"and not the regime; #1302 is the fix for that read and the "
             f"next dec boot is where the premise is first testable. What "
             f"the order rests on is the A/B (throughput and pool), not the "
             f"regime claim -- so these two arms are "
             f"hypotheses with a derivation, not measurements, and the "
             f"D-OPERATING-POINTS line says so per row. They are needed "
             f"because neither 'auto' nor 'auto-performance' moves the "
             f"attention/GDN split (uneven_perf.py:6571) and that split IS the "
             f"operating-point question. A position is REFUSED rather than "
             f"shipped when the card-rate library cannot price it (W61), when "
             f"its vector SATURATES a family -- a rank's proportional share "
             f"below one unit, so partition_units floors it and the axis is "
             f"pinned at its floor -- or resolves FLAT, which is even TP "
             f"wearing an uneven flag (W62), when it would need an env pin to "
             f"be honoured (W63), or when PerfCostModel.predict_capacity marks "
             f"it feasible=False (W64). A rank with ZERO heads is NOT among "
             f"them and never was: partition_units guarantees every rank at "
             f"least one unit, so that check could not fire. "
             f"Whatever is chosen, the launcher prints the PDFLIP D-WEIGHTS line "
             f"naming the objective, the weight vector, the per-rank head "
             f"partition and the estimated cost at the attention barrier, PLUS "
             f"the PDFLIP D-OPERATING-POINTS line carrying ALL THREE vectors "
             f"with their priced round cost, world pool, FUNDED CONTEXT "
             f"(min(sum P, 64 min P), the bound the smallest rank sets) and "
             f"the model's feasible verdict -- on every boot, "
             f"including the default one, so the trade is readable per boot. "
             f"No arm is silent and no arm is solved here.",
    )
    ap.add_argument(
        "--d-rank-perf-tune", default="both",
        help="Target handed to --rank-perf-tune when --d-tp-objective is "
             "'speed'. Ignored otherwise (and saying so is the point: a tune "
             "target without the speed arm would read as an active setting). "
             "Choices are the runtime's own "
             "(server_args._RANK_PERF_TUNE_CHOICES: both|dec|enc|maxkv|"
             "phase-prefill|phase-decode); 'dec' is the one that also selects "
             "--rank-kv-ratio speed, i.e. the KV-token lever, which is the "
             "larger one at depth.",
    )
    ap.add_argument(
        "--d-uneven-token-vector", default=None,
        help="#1032. Group D's KV-token ownership vector, e.g. '17,7,8'. "
             "UNSET IS THE DEFAULT AND THE RIGHT ANSWER: group D derives its "
             "own starting vector from this boot's budgets and replaces it "
             "with the measured per-rank optimum after profiling, which is "
             "what the 'Uneven-DCP token sizing' lines are. This launcher used "
             "to ship 29,19,16 as a seed -- the emitted value of RETRACTED "
             "investigation #602 -- and boot weg2rg6 measured the runtime "
             "superseding it to 17,7,8 on all three ranks, so the seed bought "
             "a #797 warning and nothing else. A value passed here is checked "
             "against planner.retracted's register BEFORE the boot: a "
             "withdrawn vector is a W46 refusal naming its ticket, in EITHER "
             "role, which is stricter than the runtime (it refuses only a "
             "retracted 'pin').",
    )
    ap.add_argument(
        "--d-uneven-token-vector-role", choices=["seed", "pin"], default="pin",
        help="Role of --d-uneven-token-vector. 'pin' (default here) asserts "
             "the value against the runtime's own measurement; 'seed' declares "
             "it provisional and promises supersession in-process. The default "
             "is 'pin' deliberately: an operator who types a vector is "
             "asserting it, and a 'seed' default would re-create the shape "
             "#1032 removed -- a number nobody stands behind riding along "
             "because its role made it look harmless.",
    )
    ap.add_argument(
        "--d-uneven-token-vector-provenance", default=None,
        help="WHERE --d-uneven-token-vector came from (#797): the "
             "investigation, task id or tool. Declaring a lineage that is not "
             "retracted does NOT switch off the value match here -- #900's "
             "lesson, and the value match is what caught the shipped 29,19,16 "
             "in the first place.",
    )
    ap.add_argument(
        "--d-disable-cuda-graph", action="store_true",
        help="CONTROL ARM ONLY (#1241b). Put --disable-cuda-graph on group "
             "D, so every decode round runs eager and the compute/wait split "
             "is measured the slice-1 way. THIS CHANGES THE MEASURED FORM: "
             "boot weg2dec1_0909 recorded eager rounds averaging 16.6x the "
             "graphed ones, so a ms/round or a compute/wait number taken "
             "under this flag describes the eager form and must never be "
             "quoted as the form's. The graphed form is split by the event "
             "nodes #1241b lays at capture -- use this arm to CHECK those "
             "numbers, or when the overhead line reports the nodes as "
             "overwritten. Default off, and off is byte-identical argv.",
    )
    ap.add_argument(
        "--d-disable-overlap-schedule", action="store_true",
        help="Put --disable-overlap-schedule back on group D. The escape "
             "hatch for a server_args gate that refuses overlap under D's "
             "combination; taking it logs W41 with the gate named. Not a "
             "tuning knob.",
    )
    # -- measurement arms -------------------------------------------------
    ap.add_argument(
        "--p-hicache-write-policy", default="write_back",
        choices=["write_through", "write_back", "write_through_selective"],
        help="Group P's hicache write policy. Default write_back since "
             "2026-09-15 (user decision: 'die Eviction ist der Flip' -- the "
             "publish sweep in flush_cache before sleeping writes the store, "
             "accepted on boots xsn128-134). 'off' is NOT "
             "offered because this runtime has no such policy "
             "(server_args.py:4318 choices); write_back is the nearest arm "
             "-- it defers the store write rather than removing it, and the "
             "structural removal (a shared ring, async ack) belongs to the "
             "ring slice, not here.",
    )
    # -- #1235: the literals and env gates that used to be nobody's ------
    ap.add_argument(
        "--random-seed", type=int, default=RANDOM_SEED,
        help=f"Sampling seed for BOTH groups. Default {RANDOM_SEED} is "
             f"ARBITRARY AND FIXED, and that is the whole provenance -- it is "
             f"not measured and not solved, and it is a flag so that no reader "
             f"has to wonder which it was. What it buys is that two boots of "
             f"one tip sample identically, so a decode difference between them "
             f"is a difference in the code.",
    )
    ap.add_argument(
        "--barlink-bar1-cap-cycles", type=int, default=BARLINK_BAR1_CAP_CYCLES,
        help=f"Ceiling on barlink's build-time cycle budget, both groups. "
             f"Default {BARLINK_BAR1_CAP_CYCLES} is large BY CONSTRUCTION: it "
             f"is a ceiling, not a target, and the gate that actually binds on "
             f"this rig is the BAR1 aperture priced at "
             f"--barlink-bar1-window-mib (#1234 C1), not this number.",
    )
    ap.add_argument(
        "--collective-census-interval", type=int, default=COLLECTIVE_CENSUS_INTERVAL,
        help=f"How often the collective census prints, both groups. Default "
             f"{COLLECTIVE_CENSUS_INTERVAL} is a STRIDE and not a measurement: "
             f"larger is quieter, smaller costs log volume, and nothing about "
             f"the boot depends on it except how much of it a reader sees.",
    )
    ap.add_argument(
        "--d-barlink-bar1-window-mib", default=D_BARLINK_BAR1_WINDOW_MIB,
        help=f"Group D's BAR1 windows. Default {D_BARLINK_BAR1_WINDOW_MIB!r} is the N = 3 / 256 MiB "
             f"BAR measurement (#1234 C1, see argv_d); on another card count or BAR1 size the "
             f"launcher derives it (HW-P1c, pdflip/bar1_windows.py) unless this flag names one.",
    )
    ap.add_argument(
        "--p-barlink-bar1-window-mib", default=P_BARLINK_BAR1_WINDOW_MIB,
        help=f"Group P's BAR1 windows. Default {P_BARLINK_BAR1_WINDOW_MIB!r} "
             f"shares group D's provenance from the other side (#1234 C1): the "
             f"two groups' windows are sized to fit TOGETHER -- P 24+96 and D "
             f"16+32+40 = 208 of the 224 MiB usable per 3080, measured Used "
             f"224/256 including the RM carve-out. That sentence lived only in "
             f"the declared-deviations list while D's window carried the whole "
             f"derivation in a comment.",
    )
    ap.add_argument(
        "--barlink-build-window-cap-s", type=int, default=BARLINK_BUILD_WINDOW_CAP_S,
        help=f"FLLIPER_BARLINK_BUILD_WINDOW_CAP_S, both groups. Default "
             f"{BARLINK_BUILD_WINDOW_CAP_S} s. A TIMEOUT, not a measurement. "
             f"It is now launcher OUTPUT: it used to be written as "
             f"env.get(KEY, default), so an inherited export from the "
             f"launcher's own shell won silently and the boot ran a number no "
             f"flag, record or log line ever named (the R19 shape).",
    )
    ap.add_argument(
        "--pp-chain-recv-stall-s", type=int, default=PP_CHAIN_RECV_STALL_S,
        help=f"FLLIPER_PP_CHAIN_RECV_STALL_S. Default {PP_CHAIN_RECV_STALL_S} s, "
             f"a timeout, launcher OUTPUT for the same reason as above.",
    )
    ap.add_argument(
        "--pp-occupant-horizon-s", type=int, default=PP_OCCUPANT_HORIZON_S,
        help=f"FLLIPER_PP_OCCUPANT_HORIZON_S. Default {PP_OCCUPANT_HORIZON_S} s, "
             f"a timeout, launcher OUTPUT for the same reason as above.",
    )
    ap.add_argument(
        "--match-refusal-census-every", type=int, default=MATCH_REFUSAL_CENSUS_EVERY,
        help=f"FLLIPER_MATCH_REFUSAL_CENSUS_EVERY. Default "
             f"{MATCH_REFUSAL_CENSUS_EVERY} is a census STRIDE. Launcher "
             f"OUTPUT for the same reason as above -- and note the denominator "
             f"law applies to what it emits: a rate-limited counter's zero is "
             f"not a zero.",
    )
    ap.add_argument(
        "--no-arming-floor-solved", dest="arming_floor_solved",
        action="store_false",
        help="Drop FLLIPER_ARMING_FLOOR_SOLVED=1 (default ON, unchanged "
             "shipped behaviour). A boolean env gate with no flag was "
             "indistinguishable from a hard-coded constant (#1235).",
    )
    ap.add_argument(
        "--no-hicache-bigram-keys", dest="hicache_bigram_keys",
        action="store_false",
        help="Drop FLLIPER_HICACHE_BIGRAM_KEYS=1 (default ON). MEASURED reason "
             "for ON, boot weg2ls3b3 (#1233): group D (NEXTN) keys the store "
             "by BIGRAM page hashes and group P (no spec) by UNIGRAM, so the "
             "two built disjoint chains for the same prompt and D never read "
             "P's pages. One key scheme for both groups; a no-op on D.",
    )
    ap.add_argument(
        "--no-hicache-flush-publish-sweep", dest="hicache_flush_publish_sweep",
        action="store_false",
        help="Drop FLLIPER_HICACHE_FLUSH_PUBLISH_SWEEP=1 (default ON). Reason "
             "for ON (#1233 zero-remainder): /flush_cache publishes every "
             "un-backed device node to the store before the idle witness, so a "
             "chain the write-through pin budget declined mid-prefill is not "
             "lost at the flip.",
    )
    ap.add_argument("--teardown", default="", help="path of a boot state json to tear down")
    _register_flag_aliases(ap)   # rename transition (compat_shims): every flip flag also under the other spelling, same dest
    return ap




































def main(argv: Optional[Sequence[str]] = None) -> int:
    global _ACTIVE_BOOT_STATE
    # #1248: unclaimed until a BootState exists below -- a stale pointer from
    # a PREVIOUS boot in the same process (there is none today; guards the
    # shape anyway) must never make a fast pre-spawn refusal look post-spawn.
    _ACTIVE_BOOT_STATE = None
    # rename transition (compat_shims): old and new flip-flag spellings both reach the parser
    ns = build_parser().parse_args(_canonical_flags(sys.argv[1:] if argv is None else list(argv)))
    # HW-P1a 1003: the launch's --cards selection, installed before ANY card
    # read of this launch (and cleared when absent -- one process, one launch).
    set_card_selection(getattr(ns, "cards", ""))
    if not ns.teardown:   # rename transition: rig-state dir (compat_shims); a teardown boots nothing
        from flliper._compat_boot import link_state_dir

        link_state_dir()
    # H91b/H95: the Next-Flash form's own D seat bound (6, dynamic 1..6 per
    # phase) and its pool waves, before anything reads --d-bs or --env-d.
    apply_profile_d_bs_default(ns, list(sys.argv[1:] if argv is None else argv))
    _p_gc_line = apply_p_gc_warn_default(ns)  # NF y3z ep52: gen-2 GC on P named
    if _p_gc_line:
        print(_p_gc_line, flush=True)
    _h95_waves_line = apply_profile_d_pool_waves_default(ns)
    if _h95_waves_line:
        print(_h95_waves_line, flush=True)
    _h95c_line = apply_profile_d_seat_vram_default(ns)
    if _h95c_line:
        print(_h95c_line, flush=True)
    _p0_line = apply_profile_torch_cache_cap_default(ns)  # PDFLIP-ALLOC-OVERHANG
    if _p0_line:
        print(_p0_line, flush=True)
    if not ns.teardown:
        # LEISTUNGSSCHALTER (user rule 29.09.): the row's per-group switch
        # defaults and its --d-kv-token-cut, before any reader of --env-* or
        # of the kv axis (resolve_form below).
        _ls_line = apply_profile_group_switch_defaults(ns)
        if _ls_line:
            print(_ls_line, flush=True)
        _cut_line = apply_profile_d_kv_token_cut_default(
            ns, list(sys.argv[1:] if argv is None else argv))
        if _cut_line:
            print(_cut_line, flush=True)
        _vis_default_line = apply_profile_vision_default(
            ns, list(sys.argv[1:] if argv is None else argv))
        if _vis_default_line:
            print(_vis_default_line, flush=True)
    _miss_rec_line = apply_owned_miss_record_default(ns)  # PR: paired miss record
    if _miss_rec_line:
        print(_miss_rec_line, flush=True)
    if not ns.teardown:
        _park_split = d_park_split_refusal(ns)  # 27B park (RV B5): front and D must agree
        if _park_split:
            print(_park_split, flush=True)
            raise SystemExit(_park_split)
        _vis_async = vision_async_refusal(ns)  # VISION-SYNC LAW (user 02.10.), ported from the NF line
        if _vis_async:
            print(_vis_async, flush=True)
            raise SystemExit(_vis_async)
    # VRAM-VERTRAG M1: the profile's hand pins, read before any solver
    # publishes into --extra-*/--env-* (every one is planner debt: OVERRIDE)
    from flliper.srt.pdflip import vram_plan_view as _vpv

    vram_view().reset(overrides=_vpv.profile_pins(ns), identity={
        "profile": str(getattr(ns, "profile", "") or ""),
        "model": os.path.basename(str(getattr(ns, "model", "") or "").rstrip("/")),
        "tag": str(getattr(ns, "tag", "") or "")})
    _sf_on, _sf_src = standard_form_resolved(ns)
    if _sf_on or _sf_src.startswith("env "):
        print("PDFLIP-STANDARD-FORM (NF H91, profile field standard_form): %s (%s) -- D seats per "
              "phase, park, front phase policy%s" % ("on" if _sf_on else "off", _sf_src,
              "" if _sf_on else " all OFF (pre-H91 form)"), flush=True)
    # PDFLIP-FORM: resolved ONCE, before apply_spec_form (--form-draft and
    # --form-p-draft may drive --spec-form / --draft-kv-on-p /
    # --dflash-produce-on-p), and PUBLISHED into this process's own environment
    # so build_env (dict(os.environ)) and the front hand the SAME value to every
    # rank -- the #1273 B4k pattern below. A teardown boots nothing: no form.
    # #233 review MUSS 1/2: a frozen corridor pointer or recipe inherited from
    # a shell or an earlier main() in this process never describes THIS boot;
    # they are re-derived below, before the first floor read.
    corridor_guard.clear_boot_pointer_env()
    boot_form = None
    if not ns.teardown:
        boot_form = pdflip_form.resolve_form(
            ns, list(sys.argv[1:] if argv is None else argv),
            parse_group_env=parse_group_env, shlex_split=shlex.split,
            # UNIFY S6: the p_draft axis is the ONE writer of "does P carry
            # a draft producer" -- it applies the H25 rule here, with the
            # profile's default and the env alias, before any reader.
            draft_on_p_rule=resolve_draft_on_p,
            draft_on_p_env=(envs.FLLIPER_PDFLIP_DRAFT_ON_P.is_set(),
                            bool(envs.FLLIPER_PDFLIP_DRAFT_ON_P.get())))
        os.environ[pdflip_form.FORM_ENV] = boot_form.env_value()
        # RG (operator 26.09.): the agent-load prefix switches are registry
        # fields. build_env (P, D) and the front's env publish them from the
        # row of the resolved form's profile (pdflip_form.publish_prefix_switches);
        # an explicitly set env wins; an off row writes nothing (NF byte-equal).
        # Here: the one line naming value and source, and the MZ P==D refusal
        # (--env-p/--env-d are applied after the row).
        print(prefix_switches_announce(ns, boot_form), flush=True)
        # UNIFY S3: argparse defaults that are MEASURED constants follow the
        # profile's registry row (unset flags only).
        _pad_line = profile_arg_defaults_line(
            ns, apply_profile_arg_defaults(ns, list(sys.argv[1:] if argv is None else argv)))
        if _pad_line:
            print(_pad_line, flush=True)
    ns.pdflip_boot_form = boot_form
    # #239 S2: a planned-but-unwired token cut never reaches a rank.
    refuse_unwired_token_cut(ns, boot_form)
    # #239 S3h: the cut runs D-only until the P->D hand-off under it is built.
    refuse_flip_under_token_cut(ns, boot_form)
    # H87: the #114/H41 calibration identity by memory footprint, decided once
    # here where the checkpoint PATH is known (build_env sees only its name).
    _h87_alias_line = (note_calibration_footprint_alias(ns.model)
                       if boot_form is not None else None)
    # PDFLIP-FORM + UNIFY S3: THE CALIBRATION IDENTITY (one acceptor, its terms
    # from the profile's records row: checkpoint AND form, AND the 27B line
    # term where the row names it). Every measured source this boot prices
    # from (the sidecar record, the P logs the cut and the depth read) passes
    # it. None (teardown) = no filter.
    calib_identity = calib_identity_of(ns)
    calib_sample_accept = (
        calib_identity.accepts_sample if calib_identity is not None else None)
    calib_log_accept = calib_log_accept_of(ns)
    apply_spec_form(ns)
    apply_p_prefill_graph(ns)
    # --p-chunk-policy: after the graph (reads its buckets), before any argv.
    # H92 (NF): under --profile nextflash the NF profile of this checkpoint (model key).
    apply_p_chunk_policy(ns, boot_form, list(sys.argv[1:] if argv is None else argv))
    # --p-layer-split: after the chunk policy (the joint plan reads its limits).
    apply_p_layer_split(ns)
    # --d-reshard: before any argv; 'off' installs nothing.
    apply_d_reshard(ns)
    apply_d_token_placement(ns)
    apply_d_kv_evict(ns)  # after the placement it requires
    # 27B line G2: the one tokenizer both groups load, installed before any
    # argv is built; its line is logged with the checkpoint lines under 1a'.
    tokenizer_line = apply_tokenizer_path(ns)
    # #1386: THE SWITCH IS RESOLVED HERE, ONCE, AS EARLY AS `ns` EXISTS --
    # earlier than `draft_kv_on_p` below, because the FIRST `common_flags`
    # call (the sentinel `chunk_tokens` solve, several hundred lines down)
    # already needs it. Every later reader (that call, its post-ledger
    # re-check, the ledger call, the form-key build and the shipped
    # `argv_p`/`argv_d`) takes this ONE local, never `ns` again -- so a boot
    # cannot price the switch one way and allocate the other (the exact
    # #1385/#1386 shape three boots already cost).
    hicache_disabled = bool(getattr(ns, "pdflip_disable_hicache", False))
    # #1369: user order 2026-09-14. ONE resolution, here, before the form-key
    # build and the shipped argvs both need it. Every later reader (the
    # form-key build, the shipped `argv_p`/`argv_d`, `choose_host_ledger`)
    # takes this ONE local.
    #
    # NOT a bare call to `weight_exchange.weights_cpu_backup_armed()` for the
    # `auto` case: that function's `auto` branch reads `exchange_armed()` and
    # `inject_authoritative()`, NEITHER of which take an `explicit` this
    # process could hand them (`exchange_armed()` has no such parameter at
    # all) -- exactly the S6 fix E trap `inject_mode()`'s own docstring
    # warns about, one predicate over: the LAUNCHER's own process never has
    # `WEIGHT_SOURCE_ENV`/`INJECT_ENV` in its bare environment, only the
    # ranks do (`prepare_xchg_env`/`build_env` publish them there, never
    # here). `ring_absent_by_design` already solves exactly this -- it takes
    # `weight_source`/`inject_mode` as plain VALUES, never through the
    # env-reading functions -- so `auto` here is that ONE authority's
    # negation rather than a second, launcher-unsafe spelling of the same
    # conjunction. `on`/`off` need none of this: both short-circuit inside
    # `weights_cpu_backup_armed()` before either sub-predicate is touched.
    if str(ns.pdflip_weights_cpu_backup) == weight_exchange.WEIGHTS_CPU_BACKUP_AUTO:
        weights_cpu_backup_armed = not ring_absent_by_design(
            str(ns.pdflip_weight_source), str(ns.pdflip_xchg_inject))
    else:
        weights_cpu_backup_armed = weight_exchange.weights_cpu_backup_armed(
            explicit=ns.pdflip_weights_cpu_backup)
    resolve_dual_layout(ns)
    if getattr(ns, "flip_weights", "family") == "resident":
        # Task #47 Scheibe 6a: the weights never sleep -> no host ring, no cpu
        # backup, no exchange; the ring planner takes the ABSENT-BY-DESIGN path.
        weights_cpu_backup_armed = False
        print("PDFLIP-LAUNCH --flip-weights resident: weights of BOTH groups stay mapped, "
              "only kv_cache flips; host ring / cpu backup ABSENT-BY-DESIGN", flush=True)
    # #1348: both coverage instruments armed at once is a burnt window, and
    # the cheapest place to say so is here -- before anything is started.
    refuse_double_coverage_arm(os.environ, xchg_coverage_diff=getattr(
        ns, "xchg_coverage_diff", False))
    if ns.teardown:
        # AFTER the teardown branch (review N-6): arming before it meant a
        # `--teardown ... --xchg-coverage-diff` wrote an L-dump into the
        # evidence dir that a LATER boot's ingest would pick up -- the stale
        # dump problem, manufactured by the instrument itself.
        return teardown(ns.teardown)
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
    if _lane_cov_dir:
        from flliper.srt.pdflip import lane_coverage as _wlc

        _wlc.arm_launcher(_lane_cov_dir, boot_token=lane_coverage_boot_token(ns))
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
    if _resolved_draft_on_p is not None:
        draft_on_p, draft_on_p_prov = _resolved_draft_on_p
    else:
        draft_on_p, draft_on_p_prov = resolve_draft_on_p(
            ns.draft_kv_on_p,
            bs_source("--draft-kv-on-p", argv) == "flag",
            envs.FLLIPER_PDFLIP_DRAFT_ON_P.is_set(),
            envs.FLLIPER_PDFLIP_DRAFT_ON_P.get(),
        )
    ns.draft_kv_on_p = "on" if draft_on_p else "off"
    envs.FLLIPER_PDFLIP_DRAFT_ON_P.set(draft_on_p)
    # H25 (C): what D's parked draft holds in pinned host RAM -- a desk fact,
    # refused (W128) before any sweep or launch when it cannot be priced.
    d_draft_host_mib, d_draft_host_prov = d_draft_park_term(ns, draft_on_p)
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
    global _ANCHOR_INDEX_MODE
    _ANCHOR_INDEX_MODE = "ro" if dry else "rw"  # S68 review: a dry run reads the W65 index, never writes it
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
    # #1248: from here on, cli()'s refusal handler can see this boot's pids
    # the instant main() mutates state.pids -- no group has spawned yet
    # (launch_group(spec_p, ...) is still ~500 lines below), so a refusal
    # between here and there still takes today's unchanged fast exit.
    _ACTIVE_BOOT_STATE = state
    state.logs = {"front": front_log or "<dry>", "P": f"{base}.P.log", "D": f"{base}.D.log"}

    # 1. preflight
    # #1273 S3 (review round 2, F1): the exchange semaphores are swept INSIDE
    # shm_residue_sweep, between its refusal phase and its move phase. A second
    # call out here was the defect: the residue sweep had already moved every
    # `sem.pdflip-xchg-*` file into the archive, so the sem_unlink that follows it
    # sees an empty /dev/shm and logs "residue: none" on every boot regardless.
    # #1390: `shm_dir` NAMED HERE, not left to the parameter's own default.
    # `shm_residue_sweep(shm_dir: str = SHM_DIR)` binds that default ONCE, at
    # module-import time -- a test's `mock.patch.object(launcher, "SHM_DIR",
    # tmp)` changes the ATTRIBUTE, but a call with no `shm_dir=` keyword still
    # gets the ORIGINAL `/dev/shm` the function was defined against. Passing
    # `shm_dir=SHM_DIR` here forces THIS call to read the name fresh, at call
    # time, so the mock five #1386 tests already installed actually reaches
    # the sweep instead of silently falling through to the real box -- which
    # is why they read live and turned red under any boot on the same host.
    state.shm_sweep = shm_residue_sweep(log, ns.tag, stamp, dry, shm_dir=SHM_DIR)
    sweep_dead_credit_counters(log, dry=dry)
    stale_deadman_sweep(log, [PORT_FRONT, PORT_P, PORT_D], dry)
    refuse_if_front_unbindable(log, ns.front_host, PORT_FRONT, dry)
    host_preflight(log, ns.tag, dry)
    cards = order_cards(resolve_cards())
    # HW-P1a 1003: the card count is the inventory's; an unrunnable one is
    # refused here BY NAME with its concrete blockers. The line is printed only
    # for a count the metal has NOT proven: on the proven reference count (N = 3)
    # the launch output stays byte-identical to the tree without HW-P1a (the
    # rig-plan gate of desk 27b-int8-hw-1004; the topology is derived either way).
    _topology_line = topology_check_line(ns, cards)
    if len(cards) not in _topo.PROVEN_CARD_COUNTS:
        log(_topology_line)
    # HW-GENERIC 1002: the profile's positional records and vectors hold only
    # for the inventory they were measured on -- checked once, here, before
    # any of them is read. The reference rig passes silently-identically.
    log(inventory_check_line(ns, cards))
    refusals.flush(log)         # --force: refusals passed by sites without a logger (HW-COUNT); nothing otherwise
    # HW-P1c 1003: what the inventory check called DERIVED is installed now (a
    # subset of the calibrated cards); the BAR1 group windows and the exchange
    # region follow the card count and the measured BAR1. N = 3 on the
    # reference rig: nothing changes, no line.
    apply_inventory_derivation(ns, cards, log)
    apply_bar1_windows(ns, cards, log)
    configure_xchg_region(len(cards), log)
    state.cards = [c.__dict__ for c in cards]
    record_card_power(state.cards, cards, log)
    if not dry:
        cards_free_check(cards, log)
    cvd = ",".join(c.uuid for c in cards)
    state.cvd = cvd
    # #1348 (review MF-1): THE BOOT DECLARES WHO MUST REPORT, here, where the
    # rank count is first known. A report derived from the dumps it FOUND can
    # never notice a rank that wrote nothing -- which is the #1329 shape and
    # the whole reason the instrument exists. Both groups run one rank per
    # card; the launcher itself is the single `L` rank.
    if _lane_cov_dir:
        from flliper.srt.pdflip import lane_coverage as _wlc

        _expect_path = _wlc.write_expect_manifest(
            _lane_cov_dir,
            lane_coverage_boot_token(ns),
            {"P": len(cards), "D": len(cards), _wlc.LAUNCHER_GROUP: 1},
        )
        log(f"PDFLIP-COVERAGE ARMED timings=CONFOUNDED token={lane_coverage_boot_token(ns)} "
            f"expect=P={len(cards)},D={len(cards)},L=1 manifest={_expect_path} -- "
            f"coverage.py line tracing is ON inside every exchange leg of this "
            f"boot; it costs 2.6x-4x on EVERY Python call in the traced process "
            f"while a leg is open, so NO ms/round, prefill, decode or flip "
            f"duration from this boot is quotable as a measurement.")
    log("NVML -> CUDA ordinal map: " + ", ".join(f"ordinal {i} = nvml {c.nvml_index} {c.name} {c.uuid} total {c.total_mib} MiB" for i, c in enumerate(cards)))
    # fnFL2 H57: the power limit per card -- one line, the state JSON, and the
    # date check of every time/rate reference (a line, never a refusal).
    power_reading = log_power_limits(state, cards, log)

    # 1a2. WEG2_SCHEDULING_SPEC_0907 slice A -- THE SCHEDULING KNOBS, RESOLVED
    # BEFORE THE BUDGET SOLVE (R-13), and the host ledger is part of that
    # solve, so they are resolved above it. --p-bs / --d-bs are not only
    # concurrency: max_running_requests sizes req_to_token_pool and feeds
    # formation_target, so a bs decided after the budgets would be a
    # post-hoc override of the planner. X is derived here for the same
    # reason -- one number, printed with its inputs, then TOLD to both the
    # front and group D (C2/R-6: no HTTP round trip to a sleeping group).
    p_bs = max(1, int(ns.p_bs))
    d_bs = max(1, int(ns.d_bs))
    max_kv_per_request = int(ns.max_kv_per_request or CONTEXT_LENGTH_TOKENS)

    # #1317n D's L2 budget, DERIVED. The whole guard lives in the helper:
    # `main()` carries a pinned law of ZERO try/except blocks and a pinned
    # inline-raise count (test_pdflip_launcher_teardown_1248 -- the single
    # refusal funnel is `cli()`), so a site-local try here was a second
    # refusal path, which is the class #1248 exists to stop. It cost this
    # cycle a red gate.
    s_gb_d, _l2_terms, _l2_lines = _derive_d_l2_budget(ns, max_kv_per_request)
    for _ln in _l2_lines:
        log(_ln)
    # 0 = derive (the flag's default is the 4096 pin since 2026-09-15).
    x_seed = resolve_x(ns.tp_prefill_max_tokens or None, EVIDENCE_DIR, CHUNKED_PREFILL_TOKENS,
                       accept=calib_log_accept_of(ns))
    x_tokens, x_provenance = x_seed.tokens, x_seed.provenance
    # X-CURVES 1006: the X mode, checked before D's riegel is fixed (curve: the
    # riegel admits the curves' envelope). No --x-mode: the ceiling flag as is.
    x_mode_launch = resolve_x_mode_launch(ns, cards, x_tokens)
    # H84: D's riegel and the front's live-X ceiling, one number (0 = off).
    d_x_tokens, front_x_ceiling, x_ceiling_line = resolve_x_ceiling(
        x_mode_launch.ceiling_for_d, x_tokens, getattr(ns, "x_busy_tokens", None))
    _dual_x = dual_x_tokens(ns)
    if _dual_x is not None:
        x_tokens = d_x_tokens = _dual_x
        front_x_ceiling = 0
        x_provenance = (f"X={_dual_x} source=DUAL (--dual-d-prefill-tokens "
                        f"{int(getattr(ns, 'dual_d_prefill_tokens', 0) or 0)}: every prefill on P, D "
                        f"computes at most the N-1 anchor token; was {x_seed.provenance})")
        x_ceiling_line = (f"X CEILING: off in the dual layout -- group D --tp-prefill-max-tokens "
                          f"{_dual_x}, front --tp-prefill-max-tokens {_dual_x}: any larger remainder "
                          f"routes to P, and D refuses one by name (W31 -> P)")
        if int(ns.d_short_drain_tokens or 0) > 0:
            log(f"PDFLIP-DUAL --d-short-drain-tokens {ns.d_short_drain_tokens} -> 0 (a flip-design "
                f"idle drain onto D; in the dual layout D prefills nothing)")
            ns.d_short_drain_tokens = 0
    flip_min_work_tokens = int(ns.flip_min_work_tokens) if ns.flip_min_work_tokens is not None else x_tokens
    idle_layout_front = "P" if ns.idle_layout == "pp" else "D"
    if int(ns.d_short_drain_tokens or 0) < 0 or (ns.d_hold_s is not None and float(ns.d_hold_s) < 0):
        raise SystemExit(f"--d-short-drain-tokens {ns.d_short_drain_tokens} / --d-hold-s {ns.d_hold_s}: "
                         f"both must be >= 0 (0 / unset = off)")
    # RC2 review (L1): law 4 summed over a drain -- N above the X this launch
    # hands out is refused by name (W153), never quietly cut.
    refuse_short_drain_above_x(int(ns.d_short_drain_tokens or 0), x_tokens, x_provenance)
    # THE OPERATING POINT, WITH ITS PROVENANCE, ON EVERY BOOT RECORD. The pair
    # is provisional by the order that set it (2026-09-09, "vorerst"), so a
    # later re-measurement has to be able to sort past boots into "told" and
    # "inherited" -- the value alone cannot do that, and neither can a
    # comparison against the default (an explicit `--d-bs 4` is still a choice).
    log(f"SCHEDULING BS PROVENANCE: p_bs={p_bs} d_bs={d_bs} "
        f"source={bs_source('--p-bs', argv)}|{bs_source('--d-bs', argv)} "
        f"(shipped defaults --p-bs {DEFAULT_P_BS} / --d-bs {DEFAULT_D_BS}, user order 2026-09-09, "
        f"PROVISIONAL: 'wenn alles fertig ist kann man das immernoch nachmessen')")
    log(f"SCHEDULING KNOBS (spec slice A, resolved before the budget solve): --p-bs {p_bs} "
        f"(group P --max-running-requests + front leg-1 concurrency) --d-bs {d_bs} (group D "
        f"--max-running-requests + front D seats), independent by construction; "
        f"--max-kv-per-request {max_kv_per_request} (K9, = --context-length {CONTEXT_LENGTH_TOKENS} "
        f"unless overridden); --idle-layout {ns.idle_layout} -> front --idle-layout "
        f"{idle_layout_front}; --fairness-w-s {ns.fairness_w_s} "
        f"({'OFF' if float(ns.fairness_w_s) <= 0 else 'the only sanctioned pre-emption, A1-1'}); "
        f"--drain-deadline-s {ns.drain_deadline_s}; --min-dwell-ms "
        f"{'derived from the last flip in that direction' if ns.min_dwell_ms is None else ns.min_dwell_ms}")
    log(f"X PROVENANCE: {x_provenance}; --flip-min-work-tokens {flip_min_work_tokens} "
        f"({'= X, the same break-even at aggregate granularity' if ns.flip_min_work_tokens is None else 'operator override'})"
        f"{x_mode_launch.provenance}")
    log(x_ceiling_line + x_mode_launch.ceiling_note)
    if x_mode_launch.given:
        log(x_mode_launch.line)
    log(f"IDLE POLICY (27B, user order 2026-09-24): (a) --idle-layout {ns.idle_layout} -> the "
        f"front rests on {idle_layout_front}; (b) --d-short-drain-tokens "
        f"{int(ns.d_short_drain_tokens or 0)} ({'off' if not ns.d_short_drain_tokens else 'a queued SHORT-only backlog up to this many tokens is served on D, each request <= X=' + str(x_tokens)}); "
        f"(c) --d-hold-s {ns.d_hold_s if d_hold_active(ns.d_hold_s) else ('unset (off)' if ns.d_hold_s is None else f'{ns.d_hold_s} (off, like unset)')} "
        f"({'D stays this long after its own work ended before it flips' if d_hold_active(ns.d_hold_s) else 'the idle flip waits for min-dwell only, a held backlog for the fairness bound'})")

    log(f"SCHEDULING FLAGS AS EMITTED -- group P: --max-running-requests {p_bs} "
        f"--max-kv-per-request {max_kv_per_request} (no --tp-prefill-max-tokens: the PP prefill "
        f"group is not the one law 4 bounds); group D: --max-running-requests {d_bs} "
        f"--max-kv-per-request {max_kv_per_request} --tp-prefill-max-tokens {d_x_tokens}; "
        f"front: --p-concurrency {p_bs} --d-bs {d_bs} --tp-prefill-max-tokens {x_tokens} "
        + (f"--x-ceiling-tokens {front_x_ceiling} " if front_x_ceiling > 0 else "")
        + ("" if getattr(ns, "x_busy_tokens", None) is None
           else f"--x-busy-tokens {int(ns.x_busy_tokens)} ")
        + f"--flip-min-work-tokens {flip_min_work_tokens} --idle-layout {idle_layout_front} "
        f"--drain-deadline-s {ns.drain_deadline_s} --fairness-w-s {ns.fairness_w_s}"
        + ("" if ns.min_dwell_ms is None else f" --min-dwell-ms {ns.min_dwell_ms}"))

    # 1a'. 27B FP8: one byte layout for the flip (W160 refusal without the
    # flag under the exchange), and a NAMED line when a non-incumbent checkpoint
    # has no P-cut calibration record (argv_p would hand it the INT8 vector).
    _quant = checkpoint_quant_method(str(ns.model))
    _fp8_env, _fp8_line = fp8_layout_decision(
        _quant, str(getattr(ns, "pdflip_weight_source", WEIGHT_SOURCE_DEFAULT)),
        bool(getattr(ns, "fp8_uniform_marlin", False)))
    if _fp8_line:
        log(_fp8_line)
    # RadixArk NVFP4 (ModelOpt): the NVFP4 linears' Marlin backend is a SERVER
    # argument, appended to both groups' extras here, before any argv is built.
    _nm_refused = fp4_native_mixed_refusal(
        _quant, bool(getattr(ns, "fp8_uniform_marlin", False)),
        bool(getattr(ns, "fp4_native_mixed", False)))
    if _nm_refused:
        raise PdFlipLaunchRefused(_nm_refused)
    _um_argv = uniform_marlin_argv(
        _quant, bool(getattr(ns, "fp8_uniform_marlin", False)),
        bool(getattr(ns, "fp4_native_mixed", False)))
    _nm_env = fp4_native_mixed_env(_um_argv)
    if _nm_env:
        # #38 L4: ONE class reading for the launcher's census and every rank
        # (build_env copies os.environ), set before any census is taken.
        os.environ.update(_nm_env)
        log("PDFLIP FP4-NATIVE-MIXED env for launcher and BOTH groups: %s"
            % " ".join("%s=%s" % kv for kv in sorted(_nm_env.items())))
    if _um_argv:
        ns.extra_p = with_uniform_marlin_argv(ns.extra_p, _um_argv)
        ns.extra_d = with_uniform_marlin_argv(ns.extra_d, _um_argv)
        log("PDFLIP FP8-UNIFORM-MARLIN argv for BOTH groups: %s (--extra-p now %r, "
            "--extra-d now %r)" % (" ".join(_um_argv), ns.extra_p, ns.extra_d))
    _calib_line = p_cut_calibration_line(str(ns.model), _quant)
    if _calib_line:
        log(_calib_line)
    if tokenizer_line:
        log(tokenizer_line)

    # 1b. #1233 one-backup flip geometry + the patched saver hook
    n_layers = model_num_layers(ns.model)
    chunk_count = max(0, int(ns.weight_chunks))
    chunk_layers = int(math.ceil(n_layers / chunk_count)) if chunk_count > 0 else 0
    from flliper.srt.managers.pdflip_memory_saver import (
        chunk_tag_cards,
        weights_family_tags,
    )
    weights_tags = flip_weights_tags(ns, weights_family_tags(chunk_count))
    log(f"PDFLIP-WEIGHT-CHUNKS N={chunk_count} tags (layers per chunk {chunk_layers} of {n_layers}; family {weights_tags}); "
        "flip (C9, gathered legs) = src.pause(kv) -> ONE src.release(family) and ONE dst.resume(family) "
        "in flight together -> dst.resume(kv); the host holds ONE image per card (H(c) = max_g image_g(c)) "
        "and dst's per-tag releases fund src's acquires inside it")
    # #1374 OPTION 1: publish the chunking to the ONE producer of the tag
    # bound, BEFORE either term consumer runs, and say the number out loud per
    # lane. `source=derived` because it comes from this checkpoint's own
    # headers; an unpublished chunking REFUSES rather than defaulting to 0.
    publish_weight_chunk_layers(chunk_layers)

    # 1b'. P's PP LAYER SPLIT is NOT derived here any more (FIX 2).  It was,
    # from the two module score constants, at a point in main that runs BEFORE
    # solve_p_cut -- so from the moment the solver was wired in, this line
    # answered for the INCUMBENT while group P launched on the SOLVED cut, and
    # the flip-order map built from it was complete, confident and wrong for
    # every boot whose solve moved the cut (measured on this tip: constants ->
    # [32,18,14], solved maxkv 31,17,16/7,5,4 -> [31,17,16]).  The split now
    # comes off PCutFacts.layer_counts, next to the argv it belongs to.
    tms_so = "" if dry else build_tms_preload(tree, ns.venv, log)
    state.weight_chunks = chunk_count
    state.tms_so = tms_so

    # 1b-. TRAIN FIX 5: THE BUDGET AND CUT SOLVE MOVED IN FRONT OF THE RING.
    # The ring must be sized for the form group P will actually run, and the CUT
    # IS PART OF THAT FORM -- a boot whose stages carry different layers parks
    # different bytes per card.  While the cut was solved after the ring, the
    # ring could only ever be sized from a PREDECESSOR's form, which is boot
    # weg2tr2's killer: rg6's group P carried no MTP head, tr2's carries one on
    # its last stage, and PP2 asked for 2426 MiB with 204 free.  Nothing in this
    # block reads the host-ledger arm or the store (see the chunk-size
    # re-check under "4. group P", which proves that rather than asserting it),
    # so the move is sound; ``env_p`` stays behind because it does need the ring.
    slack_mib = reserve_slack_mib(ns.transport)
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
    # #1257c: the operator's external headroom, resolved ONCE per boot and
    # keyed by CARD UUID -- never by NVML index, which is not stable across
    # boots on this rig. Everything downstream (both budget solves, the
    # corridor floor, the front's sampler) reads THIS dict.
    # #233 review MUSS 1 + 2: ONE corridor pointer state for the whole boot.
    # The recipe of each group (its argv's footprint-relevant flags) is
    # published first, then the pointer is read ONCE and its resolution frozen
    # into this process's environment -- before the first floor of the boot is
    # read (the P budget pass right below). The budget, the lend floor, the
    # front (dict(os.environ)) and every rank (build_env; launch_group
    # re-asserts it) read THIS state; the ranks' publish in alloc_memory_pool
    # is for the next boot.
    os.environ[corridor_guard.RECIPE_ENV] = json.dumps(
        corridor_recipe_probe(ns, cards, p_bs, d_bs, hicache_disabled), sort_keys=True)
    for _frozen_line in corridor_guard.freeze_boot_pointer():
        log_boot_state(log, _frozen_line)
    user_reserve_by_card = parse_user_reserve(ns.user_reserve_mib, cards)
    log(
        "PDFLIP-USER-RESERVE PROVENANCE (#1257c, line shape #1342): "
        + "; ".join(
            f"argv[{i}] -> ordinal={i} nvml{c.nvml_index} "
            f"name={c.name!r} uuid={c.uuid} reserve_mib="
            f"{user_reserve_by_card.get(c.uuid, 0)}"
            for i, c in enumerate(cards)
        )
        + " -- VRAM left free for processes OUTSIDE this engine; it raises "
        "that card's corridor floor and lowers its budget by exactly as much. "
        "THE ARGV POSITION IS PRINTED BESIDE THE CARD NAME AND UUID ON "
        "PURPOSE: `--user-reserve-mib` is a list in RANK/CUDA-ORDINAL order "
        "(`order_cards`: the 5090 first, then the 3080s by NVML index), which "
        "is NOT NVML order on this rig -- nvml0 is a 3080 and ordinal 0 is the "
        "5090. Boot weg2xsn17 had to read this mapping back out of a "
        "CORRIDOR-FLOOR line and infer the ordering from it; after this line "
        "nobody has to."
    )
    vram_view().note_asleep("D", cards, dc_expect_d,
                            "MODEL(dc_expect_d: D reserve = record or census + margin + slack)")
    # L15-01b: the L1.5 post, gated on the master switch; off = byte-identical
    # to today. The post sizes against the PRE-post budget (the pure layer
    # prices budget minus the measured P awake peak), so the first pass is
    # today's pass without l15 and the second pass is what ships.
    # HW-GENERIC 1002: card-identity keys -> the ordinal form, ONCE, before
    # resolve_posts reads it and before build_env hands it to the ranks.
    l15_mib_by_card_identity(ns, cards, log)
    l15_posts = None
    # L15-27B-ONLY (user correction 2026-10-02): no L1.5 on any other line
    _not27b = l15_plan.refuse_not_27b(ns.profile, os.environ)
    if _not27b is not None:
        raise PdFlipLaunchRefused(_not27b)
    # L15-POOL S2: the pooled hold is a part of L1.5 -- no master, no pool
    _poolmaster = l15_plan.refuse_pool_without_master(os.environ)
    if _poolmaster is not None:
        raise PdFlipLaunchRefused(_poolmaster)
    # L15-POOL S3 is a part switch on top of POOL -- alone it would do nothing
    _pools3 = l15_plan.refuse_pool_s3_without_pool(os.environ)
    if _pools3 is not None:
        raise PdFlipLaunchRefused(_pools3)
    _pools4 = l15_plan.refuse_pool_s4_without_s3(os.environ)
    if _pools4 is not None:
        raise PdFlipLaunchRefused(_pools4)
    _pools4b = l15_plan.refuse_pool_s4b_without_s4(os.environ)
    if _pools4b is not None:
        raise PdFlipLaunchRefused(_pools4b)
    from flliper.srt.pdflip import l15_pool_anchor as _l15_pa4b

    for _ln in _l15_pa4b.s4b_boot_lines(os.environ):
        log(_ln)
    _pooldep = l15_plan.refuse_pool_with_deposit(os.environ)
    if _pooldep is not None:
        raise PdFlipLaunchRefused(_pooldep)
    if l15_plan.master_on(os.environ):
        if ns.profile == pdflip_form.PROFILE_QWEN27B:
            l15_base = budgets_from_dc(
                cards, dc_expect_d, log, "P",
                overshoot_mib=list(_pconst("P_OVERSHOOT_MIB", ns.profile)),
                overshoot_provenance=_pconst_boots("P_OVERSHOOT_MIB", ns.profile) or "boot weg2ls2b2",
                user_reserve_by_card=user_reserve_by_card,
            )
            try:
                l15_peaks = list(_pconst("P_AWAKE_PEAK_MIB", ns.profile))
            except KeyError:
                # UNMEASURED: the pure layer prices a missing peak as a 0 post,
                # never an estimate.
                l15_peaks = [None] * len(cards)
            # L15-POOL S1b: the measured per-card record as the planner input,
            # behind FLLIPER_PDFLIP_L15_POOL_PEAK_RECORD (default off: peaks and
            # log untouched). On: a card without a record gets None = no share.
            l15_peaks, _l15_peak_lines = l15_pool_peak.planner_peaks(
                ns.profile, cards, l15_peaks, os.environ)
            for _ln in _l15_peak_lines:
                log(_ln)
            l15_posts = l15_plan.resolve_posts(
                ns.profile, l15_base, l15_peaks, os.environ)
            for post in l15_posts:
                log(l15_plan.post_line(post))
            # L15-POOL S1 (log only): the per-card posts seen as ONE pool.
            from flliper.srt.pdflip import l15_keep_split, l15_pool, l15_pool_anchor

            if l15_pool.pool_on(os.environ):
                log(l15_pool.boot_line(
                    [p.mib for p in l15_posts], [p.src for p in l15_posts],
                    l15_keep_split.anchor_cap(os.environ),
                    mode=l15_pool.POOL_MODE if not l15_pool.pool_s3_on(os.environ)
                    else l15_pool.S3_MODE))
                if l15_pool_anchor.pool_s4_on(os.environ):
                    # S4: the anchors are part of the whole request (second line,
                    # the S3 boot line above stays as it was)
                    log("L15-POOL-S4 mode=%s anchor_cap=%d" % (
                        l15_pool_anchor.S4_MODE, l15_keep_split.anchor_cap(os.environ)))
            elif l15_pool.pool_shadow_on(os.environ):
                log(l15_pool.boot_line(
                    [p.mib for p in l15_posts], [p.src for p in l15_posts],
                    l15_keep_split.anchor_cap(os.environ)))
            # L15-NOCAP (N3f): armed with nothing to hold is a launch error,
            # not a boot window spent on cap=0 (N3c/N3e).
            _nocap = l15_plan.refuse_no_caps(l15_posts, os.environ, cards)
            if _nocap is not None:
                raise PdFlipLaunchRefused(_nocap)
        else:
            log(f"L15-POST skipped profile={ns.profile}")
    budgets_p = budgets_from_dc(
        cards, dc_expect_d, log, "P", overshoot_mib=list(_pconst("P_OVERSHOOT_MIB", ns.profile)),
        # #240: whose measurement the line charges -- the profile's own record
        # (NF: measured at the card), not the 27B boot it used to name for
        # every profile.
        overshoot_provenance=_pconst_boots("P_OVERSHOOT_MIB", ns.profile) or "boot weg2ls2b2",
        user_reserve_by_card=user_reserve_by_card,
        l15_mib=[p.mib for p in l15_posts] if l15_posts is not None else None,
    )
    state.budgets["P"] = budgets_p
    # Same TRAIN 2 MERGE FIX as at ``form_argv_p`` below: everything past the
    # three ledger sentinels is the value that ships.  ``chunk_tokens`` only
    # needs --chunked-prefill-size, but a caller that stops short of a grown
    # signature is the exact drift class train fix 2 closed five times over,
    # and the armed re-check under "4. group P" compares this number against a
    # FULLY specified call -- so a default bound here would read as a real
    # divergence there.
    chunk_tokens = chunked_prefill_size_of(
        common_flags(ns.model, RING_FORM_SENTINEL_S_GB, RING_FORM_SENTINEL_M_MIB,
                     RING_FORM_SENTINEL_STORE_CFG, max_kv_per_request,
                     ns.p_hicache_write_policy, "P", ns.random_seed,
                     ns.barlink_bar1_cap_cycles, ns.collective_census_interval,
                     hicache_disabled=hicache_disabled)
    )
    cut = solve_p_cut(
        ns, cards, budgets_p, ns.model, log, chunk_tokens=chunk_tokens,
        user_reserve_by_card=user_reserve_by_card, power_reading=power_reading,
    )
    state.p_fr_cap = dict(getattr(ns, "_w132_fr_cap", None) or {})
    # --p-prefill-graph-calibration: the table is valid only for the cut it
    # was measured on (Agent PG2 26.09.). Checked HERE, once, for all P ranks;
    # no table = no-op. On a mismatch the table is dropped and the cut solved
    # again for the table-less capture set (chunk_tokens does not depend on
    # the table: it is the graph bucket or the dynamic ceiling).
    if p_prefill_graph_cut_check(ns, cut, log):
        cut = solve_p_cut(
            ns, cards, budgets_p, ns.model, log, chunk_tokens=chunk_tokens,
            user_reserve_by_card=user_reserve_by_card,
        )
        log("PDFLIP PREFILL-GRAPH-CALIBRATION CUT-MISMATCH re-solved without the table: cut %s "
            "(captured=%s)" % (",".join(str(c) for c in cut.layer_counts) or cut.layer_set or "?",
                               p_prefill_graph_captured() or "none"))
    # UNIFY S9: the P chunk model is a per-stage table of ONE cut as well
    # (record pp_layer_ratio); checked on the final cut, after the graph's.
    p_chunk_model_cut_check(ns, cut, log)
    vram_plan_emit(ns, cards, "p_budget", log, group="P")
    stage_ratio, attn_stage_ratio = cut.stage_ratio, cut.attn_stage_ratio
    # 1b' (moved here by the argv slice's FIX 2) -- P's PP LAYER SPLIT, of the
    # cut group P is ACTUALLY LAUNCHED WITH.  Read off PCutFacts, where
    # solve_p_cut has already round-tripped it through the runtime's own
    # authority and refused anything that would not survive it, rather than
    # re-derived from the module constants at a point in main that cannot see
    # the solve yet.  The flip-order map below is the consumer, and a map that
    # is complete but WRONG is the failure mode this whole seam exists for:
    # interleave_pause_order refuses only an INCOMPLETE map, so a confident
    # wrong one pauses the wrong card first -- the weg2dk4 class.
    #
    # The decision itself lives in flip_order_split, which names its three
    # NO-MAP refusals -- and lives OUTSIDE main so a test can call it.
    p_split, p_split_note = flip_order_split(cut, n_layers)
    log(
        f"PDFLIP-PP-SPLIT group=P cut --pp-stage-ratio {cut.stage_ratio or '(omitted: gapped)'} "
        f"--pp-attn-stage-ratio {cut.attn_stage_ratio or '(omitted: gapped)'} "
        f"over {n_layers} layers -> REALIZED layer split "
        f"{p_split if p_split else 'NO MAP (' + p_split_note + ')'} "
        f"(source: solve_p_cut's own round trip against "
        f"{'parse_pp_layer_sets' if cut.gapped else 'derive_pp_layer_split'}, the "
        f"authority the runtime itself uses. FIX 2: this line used to state the "
        f"INCUMBENT score vectors -- main does not name them any more, in code "
        f"or in prose; the PP-CUT solver: line above prints the incumbent and "
        f"what the solve moved)"
    )
    state.p_stage_layers = list(p_split)
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
    # HICACHE-DRAFT-TIER: p_group_has_draft_producer reads the RESOLVED value.
    _SPEC_FORM["draft_kv_on_p"] = bool(draft_kv_on_p)
    if not draft_kv_on_p:
        log(draft_kv_off_line())
    elif spec_form_is_dflash():
        # --dflash-produce-on-p (27B 2026-09-24): P carries the DFlash producer
        # flags; whether it COMPUTES rides P's environment (spec_form_env).
        log(dflash_produce_line())
    log(draft_on_p_line(
        draft_kv_on_p, draft_on_p_prov,
        [] if draft_kv_on_p else strip_speculative_flags(shlex.split(ns.extra_p))[1]))
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
    # --p-chunk-policy: [] under 'fixed' (front log byte-identical).
    for _pcl in p_chunk_policy_lines():
        log(_pcl)
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
    # B4e: ONE form argv, built with the xchg token when the arm is armed, and
    # used for BOTH the key this line prints and the argv `solve` is handed --
    # so the printed provenance IS the form that was selected on, token
    # included, and the two cannot drift.
    # #1356 [fix] BY KEYWORD. This call and :10620 were the two that shifted
    # when `vision` was inserted mid-signature; naming each term is what makes
    # the next insert a no-op here instead of a silent re-binding.
    form_argv_p = xchg_form_argv(argv_p(
        py, ns.model, budgets_p, RING_FORM_SENTINEL_S_GB, RING_FORM_SENTINEL_M_MIB,
        RING_FORM_SENTINEL_STORE_CFG, shlex.split(ns.extra_p),
        p_bs=p_bs, max_kv_per_request=max_kv_per_request,
        stage_ratio=stage_ratio, attn_stage_ratio=attn_stage_ratio,
        write_policy=ns.p_hicache_write_policy,
        depth=RING_FORM_SENTINEL_DEPTH,
        window_mib=ns.p_barlink_bar1_window_mib,
        random_seed=ns.random_seed,
        barlink_cap_cycles=ns.barlink_bar1_cap_cycles,
        census_interval=ns.collective_census_interval,
        draft_kv_on_p=draft_kv_on_p,
        spec_flags=p_draft_kv_flags(shlex.split(ns.extra_d)),
        vision=ns.pdflip_vision,
        profile=ns.profile,
        p_max_total_tokens=int(cut.pool_tokens),
        hicache_disabled=hicache_disabled,
        # #1369: form-key must reflect the ring's real presence/absence --
        # the same reasoning `vision` is threaded here for ("a tower that is
        # absent IS a different weight statement").
        weights_cpu_backup=weights_cpu_backup_armed,
    ), ns.pdflip_weight_source)
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

    # 1b. C20: the R5 corridor inequality per card per direction, and C18: the
    # per-card region, both BEFORE either group starts.  The table it is solved
    # from also carries the ledger's host weights term, so this runs first.
    ring_plan = prepare_host_ring(cards, log, ns.tag, ns.ring_form, ns.evidence_dir,
                                  ns.ring_table_boot, dry, duplex_probe=ns.duplex_probe,
                                  tree=tree, py=py, p_argv=form_argv_p,
                                  # #1369 CLOSEOUT: the SAME resolved local
                                  # `form_argv_p`/`choose_host_ledger` already
                                  # read above and below -- never re-derived
                                  # here, which is the #1358 defect class this
                                  # order named by name.
                                  weights_cpu_backup_armed=weights_cpu_backup_armed)
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
    # A1-3: an un-armed ring has no fallback form to name.  The predecessor
    # named the pre-ring form here instead, which told the reader of a state
    # file that a form the launcher refuses to start had started.  The claim is
    # retracted rather than re-quoted -- reprinting it would hand the guard test
    # a false positive and the next reader a true one.
    state.ring_form = (
        ring_plan.form if ring_plan.armed else
        # #1369 CLOSEOUT: an intentional off-switch is not a refusal -- the
        # A1-3 text below describes a boot that COULD NOT get the ring it
        # needed; this describes one that deliberately does not need one.
        # Conflating the two would make a state file lie about a boot that
        # ran exactly as asked.
        "none -- ABSENT BY DESIGN (--pdflip-weights-cpu-backup resolved to "
        "off for this boot; not a refusal, not a fallback)"
        if not weights_cpu_backup_armed else
        "none -- NOT ARMED (A1-3: no fallback form exists on "
        "this host budget; this boot refuses by name)")
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
    # #77 (fnFL2w3, 21.09.): ZWEI FRAGEN, BISHER EINE ANTWORT.
    # `xchg_bounce_arm_pins_host` beantwortet "pinnt dieser Arm HOST-Bytes?"
    # und ist bei `oncard=ipc` korrekt FALSCH -- eine VRAM-Bounce wird mit
    # ihrem eigenen Leg befreit, kein Host-Byte bleibt liegen. Daran hing aber
    # auch die PUBLIKATION DER TERME, und die braucht jeder Austausch-Arm:
    # der Rang sizet seinen DEPOSIT daraus. Ohne sie ueberspringt die
    # schlafende Gruppe ihre Tags --
    #
    #   PDFLIP-XCHG-DEPOSIT-SKIPPED group=D rank=1: the arm published no bounce
    #   terms (FLLIPER_PDFLIP_XCHG_BOUNCE_TERMS), so this rank cannot size a
    #   deposit
    #
    # -- gemessen an fnFL2w3: von 16 Chunk-Tags wurde GENAU EINER deponiert
    # (weights_0, 643-693 ms je Rang), die anderen 15 uebersprungen. Die
    # wachende Gruppe wartete daraufhin ihr volles Lane-Budget von 120 s auf
    # `unit 0` von `weights_9`, was exakt dem Front-Bound entspricht, also
    # STALL epoch=0 elapsed=129,2 s, dann W29 auf resume_memory_occupation und
    # 0/6 Raenge.
    #
    # Die Ledger-Buchung bleibt am ENGEREN Praedikat (sie zahlt Host-Bytes,
    # die es bei ipc nicht gibt); publiziert wird, sobald ueberhaupt getauscht
    # wird. Ein Term, den niemand allokiert, kostet nichts -- ein fehlender
    # Term kostet den Boot.
    _xchg_armed = str(ns.pdflip_weight_source) != WEIGHT_SOURCE_DEFAULT
    if _xchg_armed:
        # #1368: THE SAME LANE COUNT THE LEDGER CHARGES. The terms published
        # here are the INPUTS the ranks recompute from, and `n_lanes` is one of
        # them (xchg_bounce._TERM_FIELDS) -- so leaving it at the default while
        # `choose_host_ledger` priced the measured count hands every rank a
        # one-lane term and the rank-side guard refuses the boot the moment a
        # rank enumerates two. Measured on this checkpoint: 1 lane charges
        # 1.784 GiB, 5 charge 7.419 GiB. One producer, called twice with the
        # same key, exactly as the glue below it already is.
        _pub_lane_n, _ = xchg_lane_count(
            stage_ratio, str(getattr(ns, "pdflip_xchg_legs", "both") or "both"))
        # #1374 OPTION 1: SAY THE BOUND OUT LOUD, per lane, where both the lane
        # count and the cut exist. (It was first placed beside the
        # PDFLIP-WEIGHT-CHUNKS line, 40 lines before `stage_ratio` is assigned --
        # an F821 that ruff caught and that would have been a NameError on
        # every armed boot. The #1368 bind ratchet does not see bare locals.)
        _tagmax_b = xchg_max_tag_bytes(str(ns.model))
        for _lane in range(int(_pub_lane_n)):
            log(f"PDFLIP-XCHG-TAGMAX lane={_lane} src_rank=* "
                f"max_tag_mib={_tagmax_b // (1 << 20)} source=derived -- the "
                f"widest window of {_WEIGHT_CHUNK_LAYERS} consecutive layers "
                f"of this checkpoint, MTP tree excluded by name "
                f"(checkpoint_census.MTP_TREE_PREFIXES); validated against "
                f"weg2xsn30's own manifests, -0.04% on stage 0")
        bounce_terms_for_ranks, _widest_line, _widest_name = (
            checkpoint_census.widest_layer_terms(
                str(ns.model),
                pairs=int(weight_exchange_region.N_CARDS),
                depth=int(getattr(ns, "xchg_bounce_depth",
                                  xchg_bounce.ASSEMBLE_DEPTH_DEFAULT)),
                slot_bytes=weight_exchange_transport.validate_oncard_slot_mib(
                    ns.pdflip_xchg_oncard_slot_mib) * weight_exchange_region.MIB,
                n_lanes=int(_pub_lane_n),
                max_tag_bytes=xchg_max_tag_bytes(str(ns.model)),
                lanes_concurrent=int(_lanes_concurrent),
                band_credit=_band_credit))
        if int(_lanes_concurrent) > 0:
            # #1385 (Wand 11b step 2): THE PERMIT, CREATED HERE -- before
            # either group starts, exactly where the region's own 24
            # semaphores are created (`prepare_xchg_region`, a few lines
            # below) and for the identical reason: a rank may only OPEN this
            # name (`weight_exchange_transport.LanePermit`), never create it,
            # so an ENOENT there means "the launcher did not arm a cap",
            # never "wait longer". Sized to `lanes_priced` -- the SAME number
            # the ledger just charged with (`bounce_terms_for_ranks`), never
            # a second count: the boot-wide budget of buffers that may be
            # pinned at once is exactly what was priced, no more and no
            # fewer.
            weight_exchange_region.create_lane_permit_semaphore(
                str(ring_plan.epoch), int(bounce_terms_for_ranks.lanes_priced))
            log(f"PDFLIP-XCHG-LANES-CONCURRENT-PERMIT armed permits="
                f"{int(bounce_terms_for_ranks.lanes_priced)} "
                f"name={weight_exchange_region.lane_permit_sem_name(str(ring_plan.epoch))} "
                f"-- a rank blocks here (W69 PdFlipXchgGateTimeout after "
                f"{weight_exchange_transport.LANE_PERMIT_TIMEOUT_S:.0f}s) "
                f"rather than pin a lane buffer past this budget")
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
    _census_record = f"{EVIDENCE_DIR}/{_hc.RECORD_NAME}"
    _census_entry, _bf_line = census_entry_for_pricing(
        _census_record, _census_key, model=ns.model, evidence_dir=EVIDENCE_DIR)
    if _bf_line:
        log(_bf_line)
    _census_terms = _hc.ledger_terms(_census_entry)
    log(_hc.census_line(_census_key, _census_terms))

    def _price_host_ledger():
        """#1453: the ONE ledger pricing, callable twice (see below)."""
        return choose_host_ledger(
            ring_plan.host_weights_bytes,
            ring_plan.host_weights_span1_bytes, ring_plan.provenance,
            d_draft_host_gib=d_draft_host_mib / 1024.0,
            vision_victim_host_gib=vision_victim_host_mib / 1024.0,
            memhist_gib=_memhist_gib, memhist_run_only=_memhist_run_only,
            l3_index_gib=_l3_index_gib,
            cold_tier_shm_gib=_cold_tier_shm_gib,
            group_envs=_ledger_group_envs,
            census=_census_terms,
            # #1390: NAMED, not left to choose_host_ledger's own literal
            # defaults -- see MEMINFO_PATH/CGROUP_ROOT above for why a bare call
            # would keep reading the real box even under a test's mock.
            meminfo_path=MEMINFO_PATH, cgroup_root=CGROUP_ROOT,
            pin_m_mib=int(getattr(ns, "pin_ledger_arm_m", 0) or 0),
            pin_s_gb=int(getattr(ns, "pin_ledger_arm_s", 0) or 0),
            # #1360: the operator's declaration, from argv and nowhere else.
            deviation_reason=str(getattr(ns, "host_ledger_deviation", "") or ""),
            guard_gib=getattr(ns, "host_riegel_gib", None),
            # #1378 Stage 2 (W105): the operator's citation of the PRIOR boot's
            # own numbers, from argv and nowhere else -- same placement, same
            # "None means unmeasured, not zero" rule as the deviation pair above.
            prior_cushion_min_gib=getattr(ns, "prior_cushion_min_gib", None),
            prior_bounce_gib=getattr(ns, "prior_bounce_gib", None),
            # #1378 Stage 2: THIS boot's own form, in the EXACT shape
            # `front.py._write_flip_ratchet` writes it in -- a different spelling
            # here would auto-resolve nothing (silently, since form_key is a
            # plain equality filter) rather than raise, so the two producers are
            # kept to the one literal construction, not re-derived twice.
            flip_ratchet_form_key=f"wtags={len(weights_tags)}",
            # PDFLIP-FORM: only samples measured on THIS checkpoint.
            record_accept=calib_sample_accept,
            # #1362 [22-fix]: content digest, snapshot-independent. `None` (an
            # unreadable checkpoint) stays empty and the arm keeps the pre-#1362
            # behaviour rather than refusing on a digest it could not compute.
            model_digest_want=(host_ledger.checkpoint_digest(ns.model)[0] or ""),
            # #1362 [22-fix2]: the ring solve already decided whether THIS boot's
            # group-P form key matches the source boot's. A legacy record (no
            # digest, written before #1362) is admitted only when it did -- the
            # form key contains --model-path and excludes labels, so a match is
            # evidence of the same checkpoint path. Read from the solve rather than
            # recomputed: a second computation of the same predicate is the second
            # bookkeeping this fork keeps paying for.
            form_key_matches=_form_key_matches,
            # #1358 [fix] THE CUT REACHES THE LEDGER. `choose_host_ledger` grew
            # `stage_ratio` with a DEFAULT of "", and this -- its only caller --
            # never passed it, so the lane record was looked up under
            # 'pp=;d=tp3;legs=both' while the seed is 'pp=39,13,12;...'. W102 on
            # every rung, rc=2, ARM=0: the producer locked out every boot.
            #
            # A DEFAULT PARAMETER IS THE SAME TRAP AS A POSITIONAL ONE, one step
            # quieter: the positional shift crashed, this one silently looked up
            # the wrong key. Desk tests passed `stage_ratio=` explicitly and were
            # blind to it exactly as they were to the argv_p shift.
            stage_ratio=str(stage_ratio or ""),
            legs=str(getattr(ns, "pdflip_xchg_legs", "both") or "both"),
            # #1317n NO BOOT WITH S_D=S_P IN THE LEDGER. D carries 4 GB where P
            # carries 1, which is +8.38 GiB of rings; an arm priced without it is
            # optimistic by that much against a reap mark nobody may touch.
            s_gb_d=s_gb_d, d_cap_terms=_l2_terms,
            weight_source=ns.pdflip_weight_source,
            bounce_depth=int(getattr(ns, "xchg_bounce_depth",
                                     xchg_bounce.ASSEMBLE_DEPTH_DEFAULT)),
            # #1385 (Wand 11b): THE SAME RESOLVED VALUE the ranks' publication
            # call above priced with -- resolved once, before either call, so the
            # ledger and the ranks cannot disagree about the cap.
            lanes_concurrent=int(_lanes_concurrent),
            # #1397 (Option 3): the SAME resolved value the ranks' publication
            # call above priced with -- resolved once, before either call, for
            # the identical reason `lanes_concurrent` is.
            band_credit=_band_credit,
            # B4f: ring absence needs BOTH arms, and the inject mode is already in
            # hand here -- `prepare_xchg_env` published it three statements above.
            inject_mode=ns.pdflip_xchg_inject,
            oncard_mode=ns.pdflip_xchg_oncard,
            oncard_slot_mib=ns.pdflip_xchg_oncard_slot_mib,
            # #1332 B1b: the checkpoint whose WIDEST layer sizes the host bounce.
            model_dir=ns.model,
            # #1386: the SAME local `main` resolved once, above, beside
            # `draft_kv_on_p` -- never re-read from `ns` here.
            hicache_disabled=hicache_disabled,
            # #1369: the SAME local `main` resolved once, above, from the SAME
            # predicate the shipped argvs read -- never re-read from `ns` here.
            weights_cpu_backup_armed=weights_cpu_backup_armed,
            # H87: the D-ONLY branch below starts group D alone -- the ledger
            # prices exactly that, not a flip boot.
            d_only=bool(getattr(ns, "d_only", False)))

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
    for ln in lines:
        log(ln)
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

    # 4. group P
    # TRAIN FIX 5 moved the dc-reserve / budgets_p / cut solve ahead of the
    # ring (block 1b- above); only ``env_p`` stays here, because it is the
    # one thing in this step that genuinely needs the armed ring.
    # #239 rc12z29c: a form the D solve sets itself (owned / joint / maxmin cut)
    # is solved HERE, from the expectation budgets, before the map is built --
    # the map's D windows and P's common prefix are functions of it, and P
    # loads the map before D's measured budgets exist.
    # R2 (28.09.): so is a P1c FR-D ceiling (d_fr_ceiling_before_map) -- its
    # FR_D is published here, the map and STORE-GEOMETRY read exactly it.
    if ((d_form_solved_by_planner(ns) or d_fr_ceiling_before_map(ns))
            and not getattr(ns, "_d_map_form", None)):
        # only the form's vectors outlive this pass; its env writes (scratch
        # cap, seat rows, stage form, waves, trims) come from the real budgets
        _env_d_before = str(getattr(ns, "env_d", "") or "")
        _map_terms: List[Dict[str, object]] = []
        _map_other, _map_other_why = d_expect_dormant_other(cards, dc_expect_d, _p_dormant_recs, ns.pdflip_weight_source, ns.profile)
        ns._d_expect_other = (_map_other if not _map_other_why.startswith("legacy") else None)
        log("D-EXPECT DORMANT-OTHER D(Karte, Erwartung): " + ", ".join(f"nvml{c.nvml_index} {_map_other[c.uuid]}" for c in cards) + f" MiB -- {_map_other_why}")
        _map_budgets = budgets_from_dc(cards, _map_other, log, "D(Karte, Erwartung)", corridor_sample_path=ns.corridor_budget_sample, corridor_constrain=True, user_reserve_by_card=user_reserve_by_card, **dict(zip(("dormant_growth_mib", "dormant_growth_provenance"), served_dormant_growth(cards, ns.profile))), charge_driver_carve=budget_charges_driver_carve(ns.profile), driver_carve_min_total_mib=driver_carve_min_total_mib(ns.profile), **dict(zip(("awake_rest_mib", "awake_rest_provenance"), d_awake_rest(cards, ns.profile))), **booked_rest_kwargs(cards, ns.profile, "D", capped=torch_cache_cap_armed(ns)), terms_out=_map_terms)
        log_d_rank_vram_solve(ns, cards, _map_budgets, log, "D(Karte, Erwartung)",
                              p_split=p_split, chunk_layers=chunk_layers,
                              card_terms=(_map_terms if d_awake_rest(cards, ns.profile)[0] is not None else None))
        _pin = pin_d_form_for_map(ns, log)
        ns.env_d = _env_d_before
        ns._d_kv_stage_written = None
        _lru_scratch = getattr(ns, "_d_kv_stage_lru_scratch", None)
        if _lru_scratch:
            # (3) 29.09.: FR_D of the pinned form was solved with this scratch
            ns.env_d = set_group_env(ns.env_d, "FLLIPER_MOE_SCRATCH_SLOTS",
                                     ",".join(str(int(x)) for x in _lru_scratch))
            log(f"{D_RANK_SOLVE_MARKER} D(Karte, Erwartung) {D_KV_STAGE_LRU_FLOOR_MARKER}: "
                f"FLLIPER_MOE_SCRATCH_SLOTS={','.join(str(int(x)) for x in _lru_scratch)} "
                f"gehoert zur gepinnten Form (FR_D mit ihm geloest) -- gilt fuer alle "
                f"weiteren Passes")
        if _pin:
            from flliper.srt.pdflip import draft_post as _dp

            ns.env_d = _dp.replace_env_vector(ns.env_d, "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION",
                                              [float(x) for x in _pin["fractions"]])
            repoint_store_geometry_at_pinned_form(
                xchg_env, _pin,
                int((_argv_vector(getattr(ns, "extra_d", ""), "--num-experts") or [512])[0]), log)

    # #107: EINMAL bauen, BEIDE Gruppen bekommen denselben Pfad.
    _emap = publish_expert_map(ns, ns.model, ns.evidence_dir, log,
                               p_stage_layers=getattr(state, "p_stage_layers", None),
                               chunk_layers=chunk_layers)
    # rc12z30b: the map read FR_D from --extra-d now; a later D solve must not move it
    ns._expert_map_path = _emap
    # 29.09.: the map is written -- the store post from the map as the ranks
    # load it. The D form pinned for the map (#239) moves the store (z30w:
    # ratios 183,137,168 -> 191,153,144); a move re-prices the ledger here,
    # before any argv reads `arm` and before any rank starts.
    if _emap:
        _ct_after, _ct_line_after = cold_tier_shm_post(
            ns, ns.model, getattr(state, "p_stage_layers", None), karte_path=_emap)
        log(_ct_line_after)
        if abs(_ct_after - _cold_tier_shm_gib) > COLD_TIER_REPRICE_EPS_GIB:
            log(f"{COLD_TIER_MARKER} RE-PRICE: the written map moved the post "
                f"{_cold_tier_shm_gib:.2f} -> {_ct_after:.2f} GiB -- the ledger prices again")
            _cold_tier_shm_gib = _ct_after
            arm, reap_headroom_gib, lines, cg = _price_host_ledger()
            state.cgroup = dict(cg)
            for ln in lines:
                log(ln)
            state.ledger_lines = lines
            state.reap_headroom_gib = reap_headroom_gib
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
    for _pcl in _pcp_lines:
        log(_pcl)
    # #1465: group P's write-through copy kernels at high stream priority, so
    # the backlog measured on weg2xsn219 (P running-req 2: 72 -> 103 un-backed
    # nodes, 2.0-2.8 s flush drain at the flip) does not build behind a
    # prefill that never idles.  P only; D keeps the default (decode graphs).
    env_p.setdefault("FLLIPER_HICACHE_WRITE_STREAM_PRIORITY", "-1")
    # P-HOST-OVERLAP / #PGAP (managers/pdflip_p_overlap.py): group P only; both
    # default ON (S43 / rc14), --no-p-host-overlap / --no-p-hostgap add nothing.
    # 27B port 20.10. (user 20.10.: the dual layout stays as it was): the #PGAP instrument is default ON only
    # on the flip form -- the 27B dual ran without it (its profile drops --p-hostgap, 27B code default off)
    _pgap = bool(getattr(ns, "p_hostgap", False)) and not getattr(ns, "dual_layout", False)
    env_p.update(p_host_overlap_env(getattr(ns, "p_host_overlap", False), _pgap))
    for _pline in p_host_overlap_lines(getattr(ns, "p_host_overlap", False), _pgap):
        log(_pline)
    # P-RECV-STREAM (managers/pdflip_p_overlap.py): group P only, default off --
    # off adds nothing, so argv and env stay byte-identical.
    env_p.update(p_recv_stream_env(getattr(ns, "p_recv_stream", False)))
    for _pline in p_recv_stream_lines(getattr(ns, "p_recv_stream", False)):
        log(_pline)
    # --p-prefill-graph: {} when off (env byte-identical). The pool vector is
    # the SAME call the cut's pool model was built from (solve_p_cut).
    _pg_pool = p_prefill_graph_pool_mib(ns)
    env_p.update(p_prefill_graph_env(
        _pg_pool, max_prefix=getattr(ns, "p_prefill_graph_max_prefix", None)))
    # --p-prefill-graph-split: {} when off (env byte-identical).
    _gs_env = p_graph_split_env(ns)
    env_p.update(_gs_env)
    if _gs_env:
        log(
            "PDFLIP P-GRAPH-SPLIT: on, up to %s KV chunks inside the prefill graph "
            "-- group P gets %s; the captured grid is q_tiles x N, every replay "
            "writes its own work-item arrays (rank lines 'FI-GRAPH-SPLIT')"
            % (int(getattr(ns, "p_prefill_graph_split", 0)),
               " ".join("%s=%s" % kv for kv in sorted(_gs_env.items())))
        )
    elif int(getattr(ns, "p_prefill_graph_split", 0) or 0):
        log("PDFLIP P-GRAPH-SPLIT: requested but INERT -- --p-prefill-graph is off, "
            "there is no captured prefill graph to split inside")
    # --p-deep-split-from: {} when off (env byte-identical).
    _ds_env = p_deep_split_env(ns)
    env_p.update(_ds_env)
    if _ds_env:
        log(
            "PDFLIP P-DEEP-SPLIT: on from prefix %s tokens -- group P gets %s: an "
            "eager chunk that deep plans a wave-aware flashinfer KV split (rank "
            "line 'FI-WAVE-SPLIT')%s"
            % (
                int(getattr(ns, "p_deep_split_from", 0)),
                " ".join("%s=%s" % kv for kv in sorted(_ds_env.items())),
                "; the prefill graph is ON, so only the chunks its own depth "
                "threshold hands to eager (--p-prefill-graph-max-prefix, census "
                "reason 'deep_split') reach the split -- set it to the same N"
                if p_prefill_graph_bucket() else "",
            )
        )
    env_p.update(p_gc_env(ns))  # --p-gc-freeze: {} under 'off'
    # --p-chunk-policy: {} under 'fixed' (env byte-identical).
    env_p.update(p_chunk_policy_env())
    # --p-layer-split: {} under 'static' (env byte-identical).
    env_p.update(p_layer_split_env())
    if _pg_pool:
        log(
            "PDFLIP P-PREFILL-GRAPH post 'prefill graph pool' MiB per stage = %s "
            "(%s; booked by the ranks via FLLIPER_KV_BUDGET_PREFILL_GRAPH_MIB and "
            "by the cut's pool model; full-CG split-kv workspace shared with the "
            "float workspace via FLLIPER_FULL_CG_PREFILL_SHARED_WORKSPACE=1)"
            % (
                ",".join("%.1f" % v for v in _pg_pool),
                "operator value"
                if str(getattr(ns, "p_prefill_graph_pool_mib", "") or "").strip()
                else "DESK ESTIMATE, replace by the measured capture_mib",
            )
        )
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
    depth_decision = solve_p_depth(
        measured=read_pp_bubble(bubble_src) if bubble_src else None,
        pool_tokens=cut.pool_tokens,
        attn_counts=cut.attn_counts,
        kv_mib_per_token_per_attn_layer=cut.kv_mib_per_token_per_attn_layer,
        hidden_size=cut.hidden_size,
        chunk_tokens=chunk_tokens,
        cap_tokens=cut.cap_tokens,
        # Read from the environment group P will actually get -- which since
        # #1240 this launcher is the only writer of (W44 refuses an inherited
        # one), so it is the solver's own decision arriving here rather than
        # an operator's export sneaking past the flags.
        gapped_layer_set=env_p.get(PP_LAYER_SET_ENV, "") if cut.gapped else "",
        pinned_depth=ns.p_microbatch_depth,
    )
    log(depth_decision.line())
    state.p_depth = depth_decision.depth
    if ns.d_disable_cuda_graph:
        log(
            "W66 PdFlipEagerDecodeArm: --d-disable-cuda-graph was passed, so "
            "group D runs --disable-cuda-graph. THIS IS A CONTROL ARM: the "
            "decode form under measurement is now eager, not the graphed "
            "full-perf form, and boot weg2dec1_0909 measured the eager mean "
            "round at 16.6x the graphed one. Every ms/round, compute/wait and "
            "throughput figure from this boot carries that scope; none of "
            "them is a statement about the shipped form."
        )
    if ns.d_disable_overlap_schedule:
        log(
            "W41 PdFlipOverlapRefused: --d-disable-overlap-schedule was passed, "
            "so group D keeps --disable-overlap-schedule. The gate that "
            "refused must be named in this boot's record; the flag is not a "
            "tuning knob and no server_args gate is to be weakened to avoid it. "
            + d_overlap_cost_line(ns.model, True, d_bs)
        )
    else:
        log(
            "SCHEDULER: --disable-overlap-schedule is group P's flag only "
            "(pp_size=3, server_args.py:19507). Group D runs pp_size=1 with "
            "the overlap scheduler ON; every forcing gate was checked against "
            "D's argv (pdmux off, device cuda, no sparse-head/dllm, and the "
            "hybrid-mamba resolution picks extra_buffer for "
            "Qwen3_5ForConditionalGeneration on linear_attn_backend=triton). "
            f"--num-continuous-decode-steps {ns.num_continuous_decode_steps} "
            f"on D (default {D_NUM_CONTINUOUS_DECODE_STEPS}). "
            + D_DECODE_STEPS_PROVENANCE
            + " "
            + d_overlap_cost_line(ns.model, False, d_bs)
        )
    if ns.p_hicache_write_policy != "write_back":
        log(
            f"MEASUREMENT ARM: group P --hicache-write-policy "
            f"{ns.p_hicache_write_policy} (shipped default is write_back since 2026-09-15; "
            f"#1016 measured the write_through store tax at +3.9 % @50k / "
            f"+11.6 % @12k and BSSCALE_0907.md P5 did NOT re-A/B it -- this "
            f"arm exists to, and changes nothing else). Group D is unchanged."
        )
    # #1275: mint THIS boot's admin key and write it 0600 before either group's
    # argv is built, so both carry the same one and the front has a path to read
    # it from. The key never appears in a log line -- only its PATH does (see
    # admin_key.redact) -- and it is excluded from the P form key by name, or a
    # fresh random value per boot would give every boot its own form.
    admin_api_key = admin_key_mod.mint()
    admin_key_file = admin_key_mod.key_path(GPU_ARB, ns.tag)
    # A DRY RUN WRITES NOTHING, and that line is a promise this must not break:
    # `--dry-run` prints "nothing started, mounted, armed or written" a few
    # dozen lines below. The key is still MINTED so the printed argv is faithful
    # in shape (and so a reader sees where the flag lands), but it is not
    # persisted and the log says which of the two happened.
    if not dry:
        admin_key_mod.write(admin_key_file, admin_api_key)
    # #1275 FIX 2: ARM THE LAUNCHER'S OWN RPCs. This is the line whose absence
    # killed weg2sb5 -- the groups demanded the key and this process, their
    # FIRST client, never sent it. Armed in a dry run too, so the auth line
    # below reports the real state rather than a special case.
    set_admin_key(admin_api_key, admin_key_file)
    state.admin_key_file = admin_key_file
    log(f"PDFLIP-LAUNCH RPC auth=bearer sites={rpc_site_count()} "
        f"(every launcher->group HTTP call goes through one door, `http()`, which "
        f"attaches the admin bearer; sites counted from this module's own AST, not "
        f"a hand-kept list. weg2sb5 died with sites=2 and auth on NEITHER: the "
        f"front authenticated, the launcher's own startup sleep(P) did not, and "
        f"/release_memory_occupation is ADMIN_OPTIONAL -> 401 at 39 s. #1275 fix 2)")
    log(f"PDFLIP ADMIN-KEY {'minted (DRY: not written)' if dry else 'minted'} for this boot -> {admin_key_file} (mode 0600); "
        f"both groups get --admin-api-key, the front authenticates its flip RPCs "
        f"with it, and /hicache/storage-backend/resize is LIVE (#1275). "
        f"Direct call: curl -s -X POST http://127.0.0.1:{PORT_D}/hicache/storage-backend/resize "
        f"-H \"Authorization: Bearer $(cat {admin_key_file})\" "
        f"-H 'Content-Type: application/json' -d '{{\"max_size_gb\": 8, \"min_free_gb\": 20}}'")
    # #1356 [fix] BY KEYWORD -- see :10189. This is the argv the ranks RUN, so
    # a silent re-binding here ships a boot with a swapped seed and window.
    shipped_argv_p = argv_p(
        py, ns.model, budgets_p, arm.s_gb, arm.m_mib, store_cfg,
        shlex.split(ns.extra_p),
        p_bs=p_bs, max_kv_per_request=max_kv_per_request,
        stage_ratio=stage_ratio, attn_stage_ratio=attn_stage_ratio,
        write_policy=ns.p_hicache_write_policy,
        depth=depth_decision.depth,
        window_mib=ns.p_barlink_bar1_window_mib,
        random_seed=ns.random_seed,
        barlink_cap_cycles=ns.barlink_bar1_cap_cycles,
        census_interval=ns.collective_census_interval,
        draft_kv_on_p=draft_kv_on_p,
        spec_flags=p_draft_kv_flags(shlex.split(ns.extra_d)),
        vision=ns.pdflip_vision,
        profile=ns.profile,
        admin_api_key=admin_api_key,
        p_max_total_tokens=int(cut.pool_tokens),
        hicache_disabled=hicache_disabled,
        weights_cpu_backup=weights_cpu_backup_armed)
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
    if ns.d_only:
        # D-ONLY (Nutzer 25.09., NVFP4 im Container: der Host traegt den Flip nicht). Nur Gruppe D,
        # gebaut wie im Dry-Run aus der Erwartung -- P's Ruherest bleibt eingeplant, also exakt die
        # D-Form des Flip-Boots. Kein Flip heisst: D muss jede ungecachte Laenge selbst prefillen,
        # der W50-Riegel steht deshalb auf dem Kontext statt auf X.
        budgets_d = d_only_solve(ns, cards, {c.uuid: dc_expect_d[c.uuid] + P_WINDOWS_MIB - D_WINDOWS_MIB for c in cards}, log,
                                 user_reserve_by_card=user_reserve_by_card,
                                 p_split=p_split, chunk_layers=chunk_layers)
        d_ratio = d_tp_ratio_decision(
            ns.d_tp_objective, ns.d_rank_perf_tune, cards, budgets_d, ns.model,
            d_bs, getattr(ns, "env_d", "") or "",
            user_reserve_by_card=user_reserve_by_card,
            dual_layout=bool(getattr(ns, "dual_layout", False)),
            overhead_mib_by_rank=d_overhead_mib,
        )
        log(d_ratio.line)
        log(d_ratio.op_line)
        log(d_tokvec.line)
        refuse_d_form_off_the_map(ns, log)  # W170, #239 rc12z29c
        env_d = build_env(tree, ns.venv, cvd, store_dir, ns.debug_hold in ("D", "both"), ns.tag, chunk_layers, chunk_count, tms_so, ns.transport, ring_plan, group="D", xchg_env=xchg_env, group_env_extra=parse_group_env(getattr(ns, "env_d", "")), **_env_knobs(ns), **rank_reserve_kwargs("D", ns, cards, user_reserve_by_card, log), expert_map_path=_emap, expert_store_identity=_estore_id)
        _sg = ";".join(f"{k}={v}" for k, v in sorted(env_d.items()) if str(k).startswith("FLLIPER_"))
        log(f"PDFLIP-GROUP-ENV D: {_sg or '(leer)'}")
        spec_d = GroupSpec("D", PORT_D, transport_argv(argv_d(py, ns.model, budgets_d, s_gb_d, arm.m_mib, store_cfg, shlex.split(ns.extra_d) + d_only_bind_argv(ns.front_host), d_bs, max_kv_per_request, max_kv_per_request, ns.num_continuous_decode_steps, ns.d_disable_overlap_schedule, d_ratio.flags, d_tokvec.flags, ns.random_seed, ns.barlink_bar1_cap_cycles, ns.collective_census_interval, ns.d_disable_cuda_graph, admin_api_key=admin_api_key, hicache_disabled=hicache_disabled, weights_cpu_backup=weights_cpu_backup_armed, profile=ns.profile, d_adopt=_d_adopt_armed(ns), vision=ns.pdflip_vision, window_mib=ns.d_barlink_bar1_window_mib), ns.transport), state.logs["D"], env_d)
        state.argv["D"] = " ".join(shlex.quote(a) for a in spec_d.argv)
        launch_group(spec_d, tree, log, dry)
        if dry:
            log(f"DRY-RUN complete (d-only): group D alone on :{PORT_D}, no P, no flip, no front; nothing started")
            return 0
        state.pids["D"] = spec_d.pid
        _write_state(state)
        register_boot_group(ns.tag, "D", spec_d.pid, log)  # H135b
        state.t_ready["D"] = wait_ready(PORT_D, spec_d.pid, ns.ready_deadline_s, log, "D", spec_d.proc)
        log(f"PDFLIP-LAUNCH D-ONLY READY group=D port={PORT_D} after {state.t_ready['D']:.1f} s -- no P, no flip, "
            f"no front; D prefills every uncached length itself (W50 = {max_kv_per_request})")
        while spec_d.proc.poll() is None:
            time.sleep(5)
        log(f"PDFLIP-LAUNCH D-ONLY group D exited rc={spec_d.proc.returncode}")
        return int(spec_d.proc.returncode or 0)
    # 5-prep. THE D PLAN AS ONE FUNCTION (BOOTZEIT 3): the serial path calls it
    # with P's measured dormant footprint after sleep(P), the early start with
    # the expectation before P loads. Its body is the former inline block,
    # unchanged but for the two names (dc_src, label) and the extra env.
    def _d_spec_from(dc_src, label, extra_env=None):
        _grow, _grow_prov = served_dormant_growth(cards, ns.profile)
        _rest, _rest_prov = d_awake_rest(cards, ns.profile)
        _over, _over_prov = d_overshoot_record(ns.profile)
        _d_terms: List[Dict[str, object]] = []
        budgets_d = budgets_from_dc(
            cards, dc_src, log, label, overshoot_mib=_over,
            overshoot_provenance=_over_prov,
            corridor_sample_path=ns.corridor_budget_sample, corridor_constrain=True,
            user_reserve_by_card=user_reserve_by_card,
            dormant_growth_mib=_grow, dormant_growth_provenance=_grow_prov,
            charge_driver_carve=budget_charges_driver_carve(ns.profile), driver_carve_min_total_mib=driver_carve_min_total_mib(ns.profile),
            awake_rest_mib=_rest, awake_rest_provenance=_rest_prov, terms_out=_d_terms,
            **booked_rest_kwargs(cards, ns.profile, "D", log, capped=torch_cache_cap_armed(ns)),
        )
        state.budgets["D"] = budgets_d
        if label == "D":  # the real pass: P's residue as measured after sleep(P)
            vram_view().note_asleep("P", cards, dc_src, "RECORD(NVML je PID nach sleep(P))")
        log_d_rank_vram_solve(ns, cards, budgets_d, log, label,
                              p_split=p_split, chunk_layers=chunk_layers,
                              card_terms=(_d_terms if _rest is not None else None))
        state.d_owner_solve = dict(getattr(ns, "_d_owner_solve", None) or {})
        d_ratio = d_tp_ratio_decision(
            ns.d_tp_objective, ns.d_rank_perf_tune, cards, budgets_d, ns.model,
            d_bs, getattr(ns, "env_d", "") or "",
            user_reserve_by_card=user_reserve_by_card,
            dual_layout=bool(getattr(ns, "dual_layout", False)),
            overhead_mib_by_rank=d_overhead_mib,
        )
        log(d_ratio.line)
        log(d_ratio.op_line)
        log(d_tokvec.line)
        refuse_flip_stage_form_without_trim_cells(ns, getattr(ns, "pdflip_boot_form", None), log)
        refuse_d_form_off_the_map(ns, log)  # W170, #239 rc12z29c
        env_d = build_env(tree, ns.venv, cvd, store_dir, ns.debug_hold in ("D", "both"), ns.tag, chunk_layers, chunk_count, tms_so, ns.transport, ring_plan, group="D", xchg_env=xchg_env, group_env_extra=parse_group_env(getattr(ns, "env_d", "")), **_env_knobs(ns), **rank_reserve_kwargs("D", ns, cards, user_reserve_by_card, log), expert_map_path=_emap, expert_store_identity=_estore_id)
        env_d.update(store_short_tail_env(x_tokens, d_x_tokens))  # 27B RC7-X / UNIFY S7
        env_d.update(d_reshard_env())  # --d-reshard: {} under 'off'
        env_d.update(d_gc_env(ns))  # --d-gc-freeze: {} under 'off'
        env_d.update(d_token_placement_env())  # --d-token-placement: {} under 'capacity'
        env_d.update(d_kv_evict_env())  # --d-kv-evict-for-placement: {} under 'off'
        env_d.update(fork_anchor_env(getattr(ns, "fork_anchor_token", None),
                                     bool(getattr(ns, "p_trim_end_anchor", False))))  # {} when off
        # #114: DIE EFFEKTIVE GRUPPEN-ENV GEHOERT INS LOG.
        #
        # Der Verdrahtungs-Check (pdflip/verdrahtung_check.sh) liest das
        # Dry-Run-Log, also das, was der Boot WIRKLICH startet. Er konnte bisher
        # keine einzige Env pruefen, weil der Launcher sie nirgends druckt:
        # `grep FLLIPER_MOE_EXPERT_STORE_DIR dry_*.log` gibt 0 Treffer, obwohl sie
        # gesetzt IST. Ein Pruefer, der am falschen Ort misst, meldet Luecken,
        # die es nicht gibt, und uebersieht die echten.
        env_d.update(extra_env or {})  # BOOTZEIT 3: the early start's gate paths; {} serially
        for _g, _e in (("P", env_p), ("D", env_d)):
            _sg = ";".join(f"{k}={v}" for k, v in sorted((_e or {}).items())
                           if str(k).startswith("FLLIPER_"))
            log(f"PDFLIP-GROUP-ENV {_g}: {_sg or '(leer)'}")
        spec_d = GroupSpec("D", PORT_D, transport_argv(argv_d(py, ns.model, budgets_d, s_gb_d, arm.m_mib, store_cfg, shlex.split(ns.extra_d) + d_reshard_argv(d_ratio.flags, shlex.split(ns.extra_d)), d_bs, max_kv_per_request, d_x_tokens, ns.num_continuous_decode_steps, ns.d_disable_overlap_schedule, d_ratio.flags, d_tokvec.flags, ns.random_seed, ns.barlink_bar1_cap_cycles, ns.collective_census_interval, ns.d_disable_cuda_graph, admin_api_key=admin_api_key, hicache_disabled=hicache_disabled, weights_cpu_backup=weights_cpu_backup_armed, profile=ns.profile, d_adopt=_d_adopt_armed(ns), vision=ns.pdflip_vision, window_mib=ns.d_barlink_bar1_window_mib), ns.transport), state.logs["D"], env_d)
        return spec_d, budgets_d

    refuse_d_early_start_unreviewed(ns)  # W185, before any group starts
    _early_sized_dir = None
    if _d_early_start_armed(ns):
        from flliper.srt.pdflip import d_early_start as _des

        _early_sized_dir = os.path.join(os.path.dirname(os.path.abspath(state.logs["D"])),
                                        f"d_early_p_sized_{ns.tag}")
        os.makedirs(_early_sized_dir, exist_ok=True)
        for _old in os.listdir(_early_sized_dir):
            os.unlink(os.path.join(_early_sized_dir, _old))
        spec_p.env[_des.P_SIZED_DIR_ENV] = _early_sized_dir  # stage 0: P ranks report sized
    if _d_early_start_armed(ns) or str(getattr(ns, "pdflip_p_free_read_journal", "off")) == "on":
        from flliper.srt.pdflip import d_early_start as _des

        _journal_dir = os.path.join(os.path.dirname(os.path.abspath(state.logs["P"])),
                                    f"p_free_reads_{ns.tag}")
        spec_p.env[_des.FREE_READ_JOURNAL_ENV] = _journal_dir
        log(f"{_des.FREE_READ_MARKER} P journals its free-memory reads -> {_journal_dir}")
    apply_dual_p_sleep(ns, spec_p, log)  # D PRIORITY stage 2 (dual unified KV only)
    spec_p.env.update(dual_alloc_cache_book_env(ns, cards, log))  # item 170: D's cache growth, booked at P's join
    spec_p.env.update(dual_outside_free_env(ns, cards, user_reserve_by_card, log))  # user rule 24.10.: the reserve, booked at P's join
    ns._dual_mps_env = start_dual_mps(ns, log, dry)
    if ns._dual_mps_env and not dry:
        # persisted AT ONCE: every exit path (cli() early exits, teardown() in a fresh process) reads it
        state.dual_mps_root = dual_mps_root(ns.tag)
        _write_state(state)
    spec_p.env.update(ns._dual_mps_env)
    spec_p.env.update(dual_p_sm_env(ns))
    spec_p.env.update(dual_duty_env(ns))
    spec_p.env.update(dual_share_env(ns, "P"))
    spec_p.env.update(dual_priority_env(ns, "P", log))  # DUAL-SHARE: {} unless a switch is on
    launch_group(spec_p, tree, log, dry)
    ns._s7b_spec_path = write_p_respawn_spec(spec_p, ns, log, dry, p_planned_need(cards, state.budgets.get("P"), ns, log))
    ns._dual_spec_d = None
    if getattr(ns, "dual_share", False):
        # DUAL-TP3PP3 1b: D boots NOW as the union owner; P holds before its
        # load until D's image (and D's installed vectors) are published.
        _dual_dc = dual_share_planned_dc(cards, state.budgets["P"], getattr(ns, "extra_p", ""),
                                         int(getattr(ns, "dual_p_overhead_mib", 1500)))
        log("PDFLIP-DUAL-SHARE D sized from P's PLAN (budget + overhead, MiB): " + ", ".join(
            f"nvml{c.nvml_index} {_dual_dc[c.uuid]}" for c in cards))
        _denv = dual_share_env(ns, "D")
        _pcut = dual_p_cut_from_argv(spec_p.argv)
        if _pcut:
            # D's union image then holds only what the P stage on each card binds;
            # D's other weights stay in their TMS tags (sleepable, stage 2).
            _denv["FLLIPER_PDFLIP_DUAL_P_CUT"] = _pcut
        log(f"PDFLIP-DUAL-SHARE P cut for D's image filter: {_pcut or '(none -> whole image)'}")
        _sd, _ = _d_spec_from(_dual_dc, "D(dual-share, P-Plan)", _denv)
        _sd.env.update(ns._dual_mps_env)
        _sd.env.update(dual_priority_env(ns, "D", log))  # DUAL-SHARE: {} unless a switch is on
        state.argv["D"] = " ".join(shlex.quote(a) for a in _sd.argv)
        launch_group(_sd, tree, log, dry)
        ns._dual_spec_d = _sd
        if not dry:
            # D must be READY -- its KV sized, its graphs captured -- before P
            # loads a byte: D sizes its KV off the card's free memory, and a P
            # load running alongside would move that reading under it. P
            # waits for this marker next to D's union socket.
            state.t_ready["D"] = wait_ready(PORT_D, _sd.pid, ns.ready_deadline_s, log, "D", _sd.proc)
            _udir = dual_share_env(ns, "D")["FLLIPER_PDFLIP_UNION_DIR"]
            os.makedirs(_udir, exist_ok=True)
            with open(os.path.join(_udir, DUAL_D_READY_FILE), "w") as _f:
                _f.write(f"{time.time():.3f}\n")
            log(f"PDFLIP-DUAL-SHARE D READY -> {_udir}/{DUAL_D_READY_FILE}: P may load now")
    if dry:
        _dry_terms: List[Dict[str, object]] = []
        _dry_other, _dry_other_why = d_expect_dormant_other(cards, dc_expect_d, _p_dormant_recs, ns.pdflip_weight_source, ns.profile)
        log("D-EXPECT DORMANT-OTHER D(dry, expectation): " + ", ".join(f"nvml{c.nvml_index} {_dry_other[c.uuid]}" for c in cards) + f" MiB -- {_dry_other_why}")
        budgets_d = budgets_from_dc(cards, _dry_other, log, "D(dry, expectation)", corridor_sample_path=ns.corridor_budget_sample, corridor_constrain=True, user_reserve_by_card=user_reserve_by_card, **dict(zip(("dormant_growth_mib", "dormant_growth_provenance"), served_dormant_growth(cards, ns.profile))), charge_driver_carve=budget_charges_driver_carve(ns.profile), driver_carve_min_total_mib=driver_carve_min_total_mib(ns.profile), **dict(zip(("awake_rest_mib", "awake_rest_provenance"), d_awake_rest(cards, ns.profile))), **booked_rest_kwargs(cards, ns.profile, "D", capped=torch_cache_cap_armed(ns)), terms_out=_dry_terms)
        # #145 AN BEIDEN STELLEN -- siehe #114 direkt darunter: es gibt ZWEI
        # Stellen, an denen budgets_d entsteht, und eine Fassung, die nur die
        # untere trifft, fehlt genau im Dry-Run, wo das Gate sie sucht.
        log_d_rank_vram_solve(ns, cards, budgets_d, log, "D(dry, expectation)",
                              p_split=p_split, chunk_layers=chunk_layers,
                              card_terms=(_dry_terms if d_awake_rest(cards, ns.profile)[0] is not None else None))
        d_ratio = d_tp_ratio_decision(
            ns.d_tp_objective, ns.d_rank_perf_tune, cards, budgets_d, ns.model,
            d_bs, getattr(ns, "env_d", "") or "",
            user_reserve_by_card=user_reserve_by_card,
            dual_layout=bool(getattr(ns, "dual_layout", False)),
        )
        log(d_ratio.line)
        log(d_ratio.op_line)
        log(d_tokvec.line)
        refuse_flip_stage_form_without_trim_cells(ns, getattr(ns, "pdflip_boot_form", None), log)
        refuse_d_form_off_the_map(ns, log)  # W170, #239 rc12z29c
        env_d = build_env(tree, ns.venv, cvd, store_dir, False, ns.tag, chunk_layers, chunk_count, tms_so, ns.transport, ring_plan, group="D", xchg_env=xchg_env, group_env_extra=parse_group_env(getattr(ns, "env_d", "")), **_env_knobs(ns), **rank_reserve_kwargs("D", ns, cards, user_reserve_by_card, log), expert_map_path=_emap, expert_store_identity=_estore_id)
        env_d.update(store_short_tail_env(x_tokens, d_x_tokens))  # 27B RC7-X / UNIFY S7
        env_d.update(d_reshard_env())  # --d-reshard: {} under 'off'
        env_d.update(d_gc_env(ns))  # --d-gc-freeze: {} under 'off'
        env_d.update(d_token_placement_env())  # --d-token-placement: {} under 'capacity'
        env_d.update(d_kv_evict_env())  # --d-kv-evict-for-placement: {} under 'off'
        env_d.update(fork_anchor_env(getattr(ns, "fork_anchor_token", None),
                                     bool(getattr(ns, "p_trim_end_anchor", False))))  # {} when off
        # #114 auch HIER: es gibt ZWEI spec_d-Stellen, und die erste Fassung
        # traf nur die andere -- der Dry-Run blieb ohne die Zeile, und der
        # Verdrahtungs-Check meldete "#114 fehlt im Baum", obwohl es im Baum
        # war. Der Pruefer hat genau das gefangen, wofuer er gebaut ist.
        for _g, _e in (("P", env_p), ("D", env_d)):
            _sg = ";".join(f"{k}={v}" for k, v in sorted((_e or {}).items())
                           if str(k).startswith("FLLIPER_"))
            log(f"PDFLIP-GROUP-ENV {_g}: {_sg or '(leer)'}")
        spec_d = GroupSpec("D", PORT_D, transport_argv(argv_d(py, ns.model, budgets_d, s_gb_d, arm.m_mib, store_cfg, shlex.split(ns.extra_d) + d_reshard_argv(d_ratio.flags, shlex.split(ns.extra_d)), d_bs, max_kv_per_request, d_x_tokens, ns.num_continuous_decode_steps, ns.d_disable_overlap_schedule, d_ratio.flags, d_tokvec.flags, ns.random_seed, ns.barlink_bar1_cap_cycles, ns.collective_census_interval, ns.d_disable_cuda_graph, admin_api_key=admin_api_key, hicache_disabled=hicache_disabled, weights_cpu_backup=weights_cpu_backup_armed, profile=ns.profile, d_adopt=_d_adopt_armed(ns), vision=ns.pdflip_vision, window_mib=ns.d_barlink_bar1_window_mib), ns.transport), state.logs["D"], env_d)
        launch_group(spec_d, tree, log, dry)
        log("front argv (dry): " + " ".join(shlex.quote(a) for a in front_argv_for(
            py, store_dir, 0, 0, dc_expect_d, cards, ns, chunk_count, 0, p_bs, d_bs, x_tokens,
            flip_min_work_tokens, idle_layout_front, admin_key_file=admin_key_file,
            wake_credit_plan=_WAKE_CREDIT_FRONT_PLAN,
            anon_preboot_bytes=anon_preboot_bytes, front_host=ns.front_host,
            x_ceiling_tokens=front_x_ceiling, x_mode_argv=x_mode_launch.front_argv)))
        log("DRY-RUN complete: nothing started, mounted, armed or written")
        return 0
    # BOOTZEIT 3 (--pdflip-d-early-start): D starts NOW, planned from the
    # expectation, and holds before its weight load until _d_early_verdict
    # below has compared that plan with P's measured dormant footprint.
    _early_d = None
    if _d_early_start_armed(ns):
        from flliper.srt.pdflip import d_early_start as _des

        _early_snap = (getattr(ns, "env_d", ""), getattr(ns, "_d_kv_stage_written", None),
                       getattr(ns, "_d_owner_solve", None))
        _early_gate = os.path.join(os.path.dirname(os.path.abspath(state.logs["D"])),
                                   f"d_early_gate_{ns.tag}.json")
        try:
            os.unlink(_early_gate)
        except FileNotFoundError:
            pass
        # dormant_other = the planner seat's ONE selector (D-EXPECT, 9f74489683):
        # P's measured residue from the record on Next Flash, else the legacy term
        _early_other, _early_other_why = d_expect_dormant_other(
            cards, dc_expect_d, _p_dormant_recs, ns.pdflip_weight_source, ns.profile)
        log("D-EXPECT DORMANT-OTHER D(early, expectation): " + ", ".join(
            f"nvml{c.nvml_index} {_early_other[c.uuid]}" for c in cards) + f" MiB -- {_early_other_why}")
        # the D-EXPECT CHECK after P's sleep compares against the plan D was
        # started with; a map pass (NF) already published the same selector's
        # value, the 27B runs none -- publish it here then (display only)
        if (getattr(ns, "_d_expect_other", None) is None
                and not _early_other_why.startswith("legacy")):
            ns._d_expect_other = _early_other
        _early_stage0 = os.path.join(os.path.dirname(_early_gate), f"d_early_stage0_{ns.tag}.json")
        try:
            os.unlink(_early_stage0)
        except FileNotFoundError:
            pass
        _early_spec, _early_budgets = _d_spec_from(
            _early_other, "D(frueh, Erwartung)",
            {_des.GATE_ENV: _early_gate, _des.STAGE0_ENV: _early_stage0})
        state.argv["D"] = " ".join(shlex.quote(a) for a in _early_spec.argv)
        launch_group(_early_spec, tree, log, dry)
        log(f"{_des.MARKER} LAUNCH group D with P (planned budgets {_early_budgets} MiB, "
            f"gate {_early_gate}); D holds before its CUDA context until P is sized "
            f"({_early_stage0}) and before its load until the verdict after sleep(P)")
        _early_stage0_result: Dict[str, object] = {}
        threading.Thread(
            target=d_early_stage0_watch, name="bz3-d-early-stage0", daemon=True,
            args=(_early_stage0, _early_sized_dir, _group_rank_count(spec_p.argv),
                  lambda: session_pids(_early_spec.pid), lambda: _early_spec.proc.poll() is None,
                  lambda: spec_p.proc.poll() is None, nvml_process_mib,
                  {c.uuid: f"nvml{c.nvml_index}" for c in cards}, _early_stage0_result, log),
        ).start()
        _early_d = (_early_spec, _early_budgets, _early_gate, _early_snap, _early_stage0_result)

    def _d_early_verdict(early, dc_measured):
        """go -> the running early D; refuse -> None (it exits, the caller starts D serially)."""
        from flliper.srt.pdflip import d_early_start as _des

        spec, planned, gate, snap, stage0 = early
        _grow, _grow_prov = served_dormant_growth(cards, ns.profile)
        _rest, _rest_prov = d_awake_rest(cards, ns.profile)
        _over, _over_prov = d_overshoot_record(ns.profile)
        measured = budgets_from_dc(
            cards, dc_measured, log, "D(gemessen, Gate)", overshoot_mib=_over,
            overshoot_provenance=_over_prov,
            corridor_sample_path=ns.corridor_budget_sample, corridor_constrain=True,
            user_reserve_by_card=user_reserve_by_card,
            dormant_growth_mib=_grow, dormant_growth_provenance=_grow_prov,
            charge_driver_carve=budget_charges_driver_carve(ns.profile), driver_carve_min_total_mib=driver_carve_min_total_mib(ns.profile),
            awake_rest_mib=_rest, awake_rest_provenance=_rest_prov,
            **booked_rest_kwargs(cards, ns.profile, "D", capped=torch_cache_cap_armed(ns)),
        )
        uuids = [c.uuid for c in cards]
        names = {c.uuid: f"nvml{c.nvml_index}" for c in cards}
        ok, lines = _des.budget_verdict(dict(zip(uuids, planned)), dict(zip(uuids, measured)), names)
        rc = spec.proc.poll() if spec.proc is not None else None
        if rc is not None:
            ok, lines = False, lines + [f"early D exited before the gate rc={rc}"]
        if stage0.get("verdict") != _des.VERDICT_GO:
            ok, lines = False, lines + [f"stage 0 not go ({stage0.get('verdict', 'no verdict yet')})"]
        for _ln in lines:
            log(f"{_des.MARKER} {_ln}")
        if ok:
            _des.write_gate(gate, _des.VERDICT_GO, "planned <= measured on every card",
                            dict(zip(uuids, planned)), dict(zip(uuids, measured)))
            state.budgets["D"] = planned
            log(f"{_des.MARKER} GO -- D loads now with the plan it was started with")
            return spec
        _des.write_gate(gate, _des.VERDICT_REFUSE, "; ".join(lines),
                        dict(zip(uuids, planned)), dict(zip(uuids, measured)))
        log(f"{_des.MARKER} REFUSE -- early D exits, D starts serially from the measured budgets")
        try:
            spec.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            spec.proc.kill()
            spec.proc.wait(timeout=30)
        # the group leads its own session (launch_group): its scheduler ranks
        # must not outlive it on the cards the serial D is about to take
        try:
            os.killpg(spec.proc.pid, signal.SIGKILL)
        except OSError:
            pass
        ns.env_d, ns._d_kv_stage_written, ns._d_owner_solve = snap
        return None

    state.pids["P"] = spec_p.pid
    _write_state(state)
    register_boot_group(ns.tag, "P", spec_p.pid, log)  # H135b
    state.t_ready["P"] = wait_ready(PORT_P, spec_p.pid, ns.ready_deadline_s, log, "P", spec_p.proc)
    # #1386 FOLLOW-UP 2 (xsn31/7 Versuch 2 wall): both markers this gate greps
    # for are printed by managers/cache_controller.py's HiCacheController
    # alone (lines ~1411/1608, grep-verified: no other emitter in the tree),
    # and that class is instantiated ONLY under `enable_hierarchical_cache`
    # (mem_cache/registry.py's create_tree_cache: `if
    # ctx.enable_hierarchical_cache: ... init_hicache ...`; this hybrid-SSM
    # model falls through to plain MambaRadixCache otherwise, which carries
    # no cache_controller at all). Under
    # `hicache_disabled` there is therefore NO EMITTER for either marker --
    # P booted clean to READY (health 200, no exception, no refusal inside
    # the rank) and was torn down ONLY by this unconditional launcher-side
    # check never having been taught the switch exists, the exact same class
    # the draft-KV-producer gate above was fixed for. SKIPPED with a named
    # line, never silently -- the mirror of "W10/W11 SKIPPED" below.
    if hicache_disabled:
        log("W7/W10 SKIPPED (--pdflip-disable-hicache): group P's cache_controller "
            "never builds (enable_hierarchical_cache is False), so there is no "
            "canonical-page/GDN-blob emitter to grade -- the plain MambaRadixCache "
            "this arm runs under prints neither marker. P's own health check "
            "already confirmed it booted clean to READY.")
    else:
        canonical_state_gate(spec_p, group_rank_count(spec_p.argv), log)

    # 4d. sleep P, measure D_c(P) -- and, fix 8, P's DORMANT HOST IMAGE.
    # This is the one sleep on this box that is NOT interleaved: group D does
    # not exist yet, so nothing is resuming and freeing an image while P writes
    # its own.  The cgroup shmem delta and the per-rank RssShmem sum are
    # therefore comparable here, and the pair is what the next boot's ledger
    # prices its image term from (BOOT_weg2dk7_0907.md: the weight-tag census
    # under-charged the measured image by +9.80 GiB).
    shmem_before = host_ledger.read_cgroup_shmem_bytes()
    if getattr(ns, "dual_layout", False):
        # DUAL-TP3PP3: P stays awake. What is measured below is P's AWAKE
        # footprint, and D is sized from it exactly as it would be from the
        # dormant residue. No dormant image is sampled or recorded -- an awake
        # footprint in the dormant-image record would poison the next boot's
        # ledger.
        state.sleep_p_ms = 0.0
        log("PDFLIP-DUAL P stays AWAKE (--dual-layout): no sleep leg, no dormant image; "
            "the per-PID reading below is P's awake footprint and sizes D")
        time.sleep(2)
        pids_p = session_pids(spec_p.pid)
    else:
        state.sleep_p_ms = sleep_group(PORT_P, log, "P", weights_tags)
        time.sleep(2)
        pids_p = session_pids(spec_p.pid)
    image_rec = None if getattr(ns, "dual_layout", False) else host_ledger.dormant_image_sample(
        group="P",
        shmem_before_bytes=shmem_before,
        shmem_after_bytes=host_ledger.read_cgroup_shmem_bytes(),
        pids=sorted(pids_p),
        weight_tags_gib=host_ledger.WEIGHT_TAGS_P_BYTES / host_ledger.GIB,
        interleaved=False,
        boot_tag=ns.tag,
        commit=tip,
        # #1362: stamp the model. Without it the next boot cannot tell whether
        # this image is its own model's or the previous tenant's, and the
        # measured 27B numbers walk straight into a 4B boot.
        # #1362 [22-fix2]: the CONTENT digest, the same identity the readers
        # compare against. [22] stamped the PATH form here, which no reader
        # keys on -- so every record this boot wrote would have stayed legacy
        # forever and the transition path below would never retire.
        model_digest_=(host_ledger.checkpoint_digest(ns.model)[0] or ""),
    )
    if image_rec is not None:
        log(host_ledger.format_dormant_image(image_rec))
        host_ledger.append_measured_record(measured_record_path(), image_rec)
    state.dormant_image_p = image_rec or {}
    dc_p = nvml_process_mib(pids_p)
    for c in cards:
        dc_p.setdefault(c.uuid, 0)
    mem = nvml_memory(cards)
    for c in cards:
        exp = dc_expect_mib(c)
        _exp_txt = (f"expectation {exp} + P windows {P_WINDOWS_MIB} = {exp + P_WINDOWS_MIB} MiB "
                    f"({'AT OR BELOW' if dc_p[c.uuid] <= exp + P_WINDOWS_MIB else 'ABOVE'} expectation; "
                    if exp is not None else
                    f"expectation UNCALIBRATED ({card_identity.card_key(c)}: no spec-1.6 class; ")
        log(
            f"PDFLIP-DC group=P nvml{c.nvml_index} {c.name}: measured {dc_p[c.uuid]} MiB per-process "
            f"(pids {sorted(pids_p)}) card used {mem[c.uuid].tenant_used_mib} MiB by processes "
            f"(instrument {mem[c.uuid].tenant_used_instrument}; card free {mem[c.uuid].free_mib} MiB allocatable, "
            f"{mem[c.uuid].reserved_mib} MiB driver-reserved); " + _exp_txt
            + "the launcher derives D from the MEASUREMENT, record 1f B6)"
        )
    state.dc_measured_p = dc_p
    # BOOTZEIT 3: the free-read journal's cut (P's first sleep is over here) and,
    # with an early D, D's bytes on each card measured per PID -- the known term
    # a deviating P reader would have to carry (never an estimate).
    _p_sleep_evt: Dict[str, object] = {"t": time.time(), "dc_p_mib": dict(dc_p)}
    if _early_d is not None:
        try:
            _p_sleep_evt["d_early_foreign_mib"] = nvml_process_mib(session_pids(_early_d[0].pid))
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            _p_sleep_evt["d_early_foreign_mib"] = f"unmeasured ({type(e).__name__}: {e})"
        log(f"BZ3 D-EARLY-FOREIGN after sleep(P): D per-PID MiB {_p_sleep_evt['d_early_foreign_mib']}")
    boot_state_write(log, event=("p_first_sleep_done", _p_sleep_evt))
    for _ln in d_expect_check_lines(cards, getattr(ns, "_d_expect_other", None), dc_p):
        log(_ln)

    # 5. group D
    spec_d = getattr(ns, "_dual_spec_d", None)  # DUAL-TP3PP3 1b: launched right after P
    if spec_d is None and _early_d is not None:
        spec_d = _d_early_verdict(_early_d, dc_p)
    if spec_d is None:
        spec_d, _ = _d_spec_from(dc_p, "D")
        state.argv["D"] = " ".join(shlex.quote(a) for a in spec_d.argv)
        spec_d.env.update(getattr(ns, "_dual_mps_env", None) or {})
        spec_d.env.update(dual_priority_env(ns, "D", log))  # DUAL-SHARE: {} unless a switch is on
        launch_group(spec_d, tree, log, dry)
    state.pids["D"] = spec_d.pid
    _write_state(state)
    register_boot_group(ns.tag, "D", spec_d.pid, log)  # H135b
    state.t_ready["D"] = wait_ready(PORT_D, spec_d.pid, ns.ready_deadline_s, log, "D", spec_d.proc)
    if not dry:
        release_shared_cache(spec_d, log)  # W98 (z30u): 2b's kept ranges, after D's take-over
    # #1386 FOLLOW-UP 2: the mirror of group P's same skip above -- D's
    # cache_controller never builds under `hicache_disabled` either (D is a
    # plain MambaRadixCache then, same as P), so it prints neither marker.
    if hicache_disabled:
        log("W7/W10 SKIPPED (--pdflip-disable-hicache): group D's cache_controller "
            "never builds either (enable_hierarchical_cache is False) -- no "
            "canonical-page/GDN-blob emitter to grade, same as group P above.")
    else:
        canonical_state_gate(spec_d, group_rank_count(spec_d.argv), log,
                             kv_owner_ranks=d_kv_worker_ranks(getattr(ns, "extra_d", "") or ""))
        # VRAM-Vertrag M2 (display only): P's blocks again, now carrying its
        # first sleep leg -- the per-PID source of the plan's asleep.P
        rank_vram_display(spec_p, log)
    # #1233 zero-remainder (1j finding 6): W9 LAUNCH-TIME KEY-SCHEME GATE. The
    # store is one carrier; a spec-less group keys pages by unigram unless
    # FLLIPER_HICACHE_BIGRAM_KEYS=1 forced the bigram scheme, a NEXTN/EAGLE
    # group keys by bigram natively. Divergent schemes wrote disjoint chains
    # for the same prompt once (boots weg2ls3b1-b3); the identity-suffix
    # check in the front cannot see it. Refuse the boot before any traffic.
    #
    # #1386 FOLLOW-UP 3 (xsn31/7 Versuch 3 wall): "the store is one carrier"
    # is the premise this whole gate stands on, and under `hicache_disabled`
    # there IS no carrier for either group -- both fall through to plain
    # MambaRadixCache (no cache_controller, no canonical page store at all,
    # same fact the W7/W10 skip above already established). The mismatch
    # this gate exists to catch (P and D writing the SAME store under
    # DIFFERENT hash schemes) cannot occur when neither group writes a store.
    # P is unigram here for an unrelated, correct reason (dropping the WHOLE
    # P_DRAFT_KV_FLAGS bundle drops P's OWN --speculative-algorithm too, not
    # just --speculative-draft-kv-only -- see the fold-in above); D stays
    # bigram because D keeps its own, independent NEXTN head. That is a
    # real divergence in each group's OWN spec config, not a shared-carrier
    # hazard, so it is not this gate's concern once the carrier is gone.
    if hicache_disabled:
        log("W9 SKIPPED (--pdflip-disable-hicache): no shared carrier exists for "
            "either group to key mismatched (same fact as the W7/W10 skip) -- "
            "P/D's own --speculative-algorithm divergence is expected here "
            "(P dropped the whole producer bundle, D keeps its independent "
            "NEXTN head) and immaterial without a store to key.")
    else:
        _forced = "#1233 HICACHE BIGRAM KEYS FORCED"
        # IPC-LOG-EXCEPTION: W9 still reads the forced-bigram LINE (Phase 2 rest, producer
        # side: RankState schema 2 `store_key_scheme`, IPC-STATE-PLAN §2.1).
        _p_bigram = ("--speculative-algorithm" in spec_p.argv) or count_marker(spec_p.log, _forced) >= 1
        _d_bigram = ("--speculative-algorithm" in spec_d.argv) or count_marker(spec_d.log, _forced) >= 1
        log(f"W9 launch-time key-scheme gate: P bigram={_p_bigram} (forced lines {count_marker(spec_p.log, _forced)}) "
            f"D bigram={_d_bigram} (forced lines {count_marker(spec_d.log, _forced)})")
        if _p_bigram != _d_bigram:
            raise PdFlipLaunchRefused(f"W9 PdFlipStoreIdentityMismatch (launch-time key scheme): P bigram={_p_bigram} D bigram={_d_bigram} -- "
                                    f"the two groups would key the sole carrier by different page-hash schemes")
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

    # #1386 FOLLOW-UP 4 (xsn31/7 Versuch 4 wall): the census this whole block
    # runs greps group D's log for cache_controller's OWN '#915 PREFETCH
    # LIMIT ... site=(init_hicache|hiradix_init)' line -- COMPONENT is
    # literally "group D KV carrier host pool (cache_controller.mem_pool_host
    # via prefetch_budget.host_pool_anchor)" (carrier_census.py). Same root
    # fact as the three skips above: under `hicache_disabled` D has no
    # cache_controller and therefore no host-staged carrier pool to overflow
    # in the first place -- there is nothing for the front's CARRIER-EXCEEDS
    # route to protect against, so there is nothing to census. Skipped with a
    # named line; carrier_max_tokens falls back to the context length, which
    # front.py's own routing reads as "never take the CARRIER-EXCEEDS branch"
    # (no prompt this rig accepts exceeds --context-length) -- the ZERO-
    # REQUEST Minimalform never exercises this route regardless, and every
    # non-Minimalform boot still runs the real census untouched.
    if hicache_disabled:
        carrier_max_tokens = CONTEXT_LENGTH_TOKENS
        _bound_src = "hicache_disabled fallback (no carrier host pool exists)"
        log(f"W45 SKIPPED (--pdflip-disable-hicache): group D's cache_controller never "
            f"builds (same fact as the W7/W10/W9 skips above), so there is no "
            f"'#915 PREFETCH LIMIT' emitter to census -- carrier_max_tokens falls "
            f"back to --context-length {CONTEXT_LENGTH_TOKENS} (front never takes "
            f"the CARRIER-EXCEEDS branch above the model's own context length).")
    else:
        # the floor is derived against THIS boot's SHORT bound (slice A's X), not
        # against a literal: the front routes SHORT on `remainder <= X`.
        _floor, _floor_why = _cc.route_floor(x_tokens)
        _expect_ranks = _cc.tp_size_of(spec_d.argv)
        # fnFL2x61: Form A expert workers (no attention layer, null storage
        # tier) hold no carrier; their plain pool is synced to the attention
        # host's staging ring and prints 0.9 x that, while the host prints the
        # arena's capacity -- they abstain, as they do in the prefetch vote.
        # IPC-LOG-EXCEPTION: W45 still reads worker and '#915 PREFETCH LIMIT' lines (Phase 2
        # rest, IPC-STATE-PLAN R1: RankState role + `carrier_prefetch_limit_tokens`).
        _abstain = _cc.form_a_worker_ranks(spec_d.log)
        _cen = _cc.census(spec_d.log, expected_ranks=_expect_ranks, floor=_floor,
                          abstain_ranks=_abstain)
        log(f"CARRIER BOUND: source='{_cc.SOURCE_MARKER}' in {spec_d.log}; component={_cc.COMPONENT}; "
            f"per-rank(TP)={_cen.per_rank} expected_ranks={_cen.expected_ranks} (from group D argv --tp-size) "
            f"abstained(Form A workers)={list(_cen.abstained)}; "
            f"{_cen.terms()} sites={_cen.site}; "
            f"floor={_cen.floor} [{_floor_why}]; verdict={_cen.verdict}: {_cen.detail}")
        for _sl in _cen.lines:
            log(f"CARRIER BOUND source line: {_sl}")

        # #1299: the floor is 1.25x X, so a FALLBACK X must not be allowed to refuse
        # the boot through W45 -- an unmeasured term is a named fallback, never an
        # actuator. The census bound itself is a measurement and is graded as before.
        _dec = _cc.decide_bound(_cen, ns.carrier_max_tokens, log_path=spec_d.log,
                                floor_why=_floor_why, x_measured=x_seed.measured)
        if _dec.refused:
            raise PdFlipLaunchRefused(_dec.detail)
        carrier_max_tokens = _dec.bound
        _bound_src = _dec.source
        if _dec.note:
            log(f"CARRIER BOUND: {_dec.note}")
        log(f"CARRIER BOUND: front --carrier-max-tokens {carrier_max_tokens} (source: {_bound_src}; measured "
            f"by the census: {_cen.measured}); prompts the front prices above it are served by ONE prefill on "
            f"D; prompts it prices between the floor {_cen.floor} and it take the leg-1/leg-2 round trip while "
            f"group D is awake and serving -- SHORT is four conjuncts, so while D sleeps a sub-floor prompt "
            f"queues to BATCH and round-trips as well")
    state.carrier_max_tokens = carrier_max_tokens
    if not hicache_disabled:
        # P-COVERS-D-SESSION (law 02.10.): the measured carrier holds D's session
        # too (no capacity route to a D prefill exists any more)
        _cc_ok, _cc_line = carrier_covers_d_session(
            carrier_max_tokens, d_session_tokens(ns.extra_d, int(ns.max_kv_per_request
                                                                or CONTEXT_LENGTH_TOKENS)))
        log(_cc_line)
        if not _cc_ok:
            raise PdFlipLaunchRefused(_cc_line)
    log(f"D-ADMIT STORE-READ GATE (FIX 4, round 4): the budget is group D's OWN #915 reading "
        f"(available/occupied/limit via /server_info hicache_prefetch), not a launcher-derived "
        f"proxy; front --d-admit-max-tokens "
        f"{'unset (that reading alone)' if ns.d_admit_max_tokens is None else str(ns.d_admit_max_tokens) + ' (operator ceiling)'}"
        f". Per-request the same pool is CARRIER-EXCEEDS at {carrier_max_tokens}; in aggregate "
        f"nothing bounded it, and --d-bs {d_bs} seats opened at once overcommitted it on boot "
        f"weg2sc1 (D read available=5418 occupied=25100 limit=27466 -> #915 vote_negative while "
        f"the round-1 front-local tally read 27466 free -- the quantity, not the bound, was wrong)")
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
    front_argv = front_argv_for(
        py, store_dir, spec_p.pid, spec_d.pid, dc_expect_d, cards, ns, chunk_count,
        carrier_max_tokens, p_bs, d_bs, x_tokens, flip_min_work_tokens, idle_layout_front,
        src_chunk_cards=src_chunk_cards, wake_credit_plan=_WAKE_CREDIT_FRONT_PLAN,
        measured_record=measured_record_path(),
        commit=tip, ledger_arm={"s_gb": arm.s_gb, "s_gb_d": s_gb_d, "m_mib": arm.m_mib,
                    # #1326: the xchg deposit is an ARM CHARGE, so the front's
                    # dormant-image sampler must be able to subtract it from
                    # the run residual exactly as it subtracts the rings.
                    # Without it an armed boot books the deposit twice at the
                    # run moment. 0.0 on every ring-arm boot, so the ring-arm
                    # record is byte-identical.
                    "xchg_bounce_gib": float(arm.terms.get("xchg_bounce_gib", 0.0) or 0.0)},
        admin_key_file=admin_key_file,
        anon_preboot_bytes=anon_preboot_bytes,
        front_host=ns.front_host,
        x_ceiling_tokens=front_x_ceiling,
        x_mode_argv=x_mode_launch.front_argv,
    )
    fenv = dict(os.environ)
    # #227 FORCED-RECORDS: the front is the record writer and spawns LAST, after every refusal site of main --
    # the verdict is final here. Held records (the launcher's own P image) are written or dropped by it, and the
    # front gets it in its environment (no seal = no records in a forced child, as before).
    refusals.seal_records()
    if refusals.forced_boot():
        log(refusals.records_verdict_line())
    host_ledger.flush_held_records(log)
    refusals.export_records_verdict(fenv)
    if refusals.forced_boot():
        boot_state_write(log, event=("forced_records", {"allow": refusals.records_allowed(),
                                                        "line": refusals.records_verdict_line()}))
    fenv["PYTHONPATH"] = f"{tree}/python"
    # L3-INDEX PRICE (02.10.): the front asks the shared L3 stem index with the
    # groups' own page-key scheme and arena dir (group_env's values, so the
    # credit is what P/D read).
    if ns.hicache_bigram_keys:
        fenv["FLLIPER_HICACHE_BIGRAM_KEYS"] = "1"
    else:
        fenv.pop("FLLIPER_HICACHE_BIGRAM_KEYS", None)
    fenv.setdefault("FLLIPER_HICACHE_ARENA_DIR", f"/dev/shm/pdflip-arena-{ns.tag}")
    # D PRIORITY stage 2: the front's capability + D's look-ahead (one extend
    # chunk + one decode round of every seat with the draft's tokens)
    fenv.update(dual_p_sleep_front_env(ns, dual_d_air_tokens(d_bs)))
    fenv.update(dual_front_kv_tag_env(ns))
    # FORK-TAIL on the front (W16, boot ...10271342): the leg-2 price needs the same fork token the groups cut at ({} when off).
    fenv.update(fork_anchor_env(getattr(ns, "fork_anchor_token", None), bool(getattr(ns, "p_trim_end_anchor", False))))
    # #71 (fnFL2v96): DIE FRONT SCHREIBT IN EINE DATEI, ALSO PUFFERT PYTHON
    # BLOCKWEISE -- und ein Tod vor dem ersten vollen Block hinterlaesst NICHTS.
    #
    # Gemessen: v96 erreichte beide Gruppen READY (P 159,0 s, D 222,1 s, W11b
    # gruen), dann verweigerte der Launcher mit "group front not READY on
    # :30030 after 120 s (Connection refused)" -- und der front.log enthielt
    # keine einzige Zeile der Front selbst, nur die des Launchers. Nicht weil
    # sie nichts tat, sondern weil stdout hier ein FILE ist (`stdout=ffh`
    # unten) und nicht ein TTY: Python schaltet dann von zeilen- auf
    # blockgepuffert (8 KiB), und was im Puffer steht, wenn der Prozess
    # abgeraeumt wird, ist verloren.
    #
    # Ein Prozess, dessen letzte 120 s unsichtbar sind, ist nicht
    # diagnostizierbar -- und das ist bei dem Prozess, der den Flip fuehrt,
    # der teuerste Blindflug im ganzen Boot. Die Gruppen selbst sind nicht
    # betroffen (ihre Logger flushen je Zeile); es geht allein um die Front.
    fenv["PYTHONUNBUFFERED"] = "1"
    if getattr(ns, "_s7b_spec_path", ""):
        from flliper.srt.pdflip import dual_p_restart as _s7b

        fenv[_s7b.SPEC_ENV] = ns._s7b_spec_path  # S7b: the front's P restart reads P's exec record here
    # FH (b1): the front's health poller finds a held rank (#1223) by its dump;
    # hand it the dir the groups write to (read only by front_health).
    if env_p.get("FLLIPER_DEBUG_HOLD_DIR") and not fenv.get("FLLIPER_DEBUG_HOLD_DIR"):
        fenv["FLLIPER_DEBUG_HOLD_DIR"] = env_p["FLLIPER_DEBUG_HOLD_DIR"]
    # RG (operator 26.09.): the prefix switches the ranks got from build_env,
    # the same call on the front's env (FLLIPER_PDFLIP_FRONT_SPAN_INFLIGHT is the
    # front's own; all five as on metal: ENV-IM-RANG front = 1).
    if getattr(ns, "pdflip_boot_form", None) is not None:
        pdflip_form.publish_prefix_switches(fenv, ns.pdflip_boot_form.profile)
    # C14 / FIX 2 round 2: the BOOT half of the VRAM credit epoch.  Launcher
    # OUTPUT in exactly the class of TMS_HOST_RING_MAP (R19), never an operator
    # knob: it is this boot's ring epoch, the same nonce every rank already got
    # from build_env, so the front's composed <boot>.<flip> token and the ranks'
    # ring identity name ONE boot.  Absent (no armed ring) the front falls back
    # to its own start time, which is boot-unique for the same reason.
    if ring_plan is not None and ring_plan.armed:
        fenv["TMS_HOST_RING_EPOCH"] = str(ring_plan.epoch)
    # #1257c: DELIVERABLE 6 -- the front's corridor sampler and this launcher
    # read THE SAME derived floor. The measured half already comes from one
    # store (the activation footprint cache, keyed by the digest the boot
    # publishes); the operator half is a launcher argument, so the front is
    # TOLD it here rather than re-deriving it. The front then prints
    # `floor=/source=/reserve=` per card on every PDFLIP-CORRIDOR line, and the
    # boot's corridor VERDICT (`corridor_arm.arm_report`) grades against THAT
    # LINE -- which is the shared surface, in one direction only, so the two
    # cannot disagree.
    if user_reserve_by_card:
        fenv[corridor_guard.USER_RESERVE_ENV] = user_reserve_env_value(user_reserve_by_card)
    log("front argv: " + " ".join(shlex.quote(a) for a in front_argv))
    ffh = open(front_log, "ab")
    _front_cwd = core_cwd_for(front_log, tree, log)
    log(f"front cwd {_front_cwd} (ops #164: crash cores land here, not in the tree)")
    fp = subprocess.Popen(front_argv, env=fenv, stdout=ffh, stderr=subprocess.STDOUT, cwd=_front_cwd, start_new_session=True)
    state.pids["front"] = fp.pid
    boot_state_write(log, fields=front_launch_fields(front_argv, fenv))
    # H135b: the front into the state json AND the registry at once -- until the
    # LAUNCHED write below (behind wait_ready and the deadmen) a guard-run
    # `--teardown` would not know the front by pid.
    _write_state(state)
    register_boot_group(ns.tag, "front", fp.pid, log)
    # PUBLISH THE LOG BEFORE WAITING, NOT AFTER.  This used to sit after the
    # LAUNCHED line, i.e. behind `wait_ready` below -- so a boot that never
    # reached a ready front never updated the symlink, and every reader
    # (`resolve_pin`, the deadman, a seat looking for "the current boot") was
    # pointed at the last boot that SUCCEEDED.  Measured 2026-09-21: seven
    # fnFL2 attempts in 2.5 h all ended `PDFLIP-LAUNCH REFUSED`, and
    # /root/current_boot.log still named the 07:26 boot.  The pointer is needed
    # MOST while a boot is failing, which is exactly when it was not written.
    os.system(f"ln -sfn {shlex.quote(front_log)} /root/current_boot.log")
    state.t_ready["front"] = wait_ready(PORT_FRONT, fp.pid, 120, log, "front", fp)

    # 7. deadmen + helpers
    huge = 10**7
    state.helper_pids.append(arm_deadman(log, spec_p.log, PORT_P, f"launch_server.*--port {PORT_P}", huge, ns.tag, "P", dry))
    state.helper_pids.append(arm_deadman(log, spec_d.log, PORT_D, f"launch_server.*--port {PORT_D}", huge, ns.tag, "D", dry))
    state.helper_pids.append(arm_deadman(log, front_log, PORT_FRONT, "flliper.srt.pdflip.front", 120, ns.tag, "front", dry))
    with open(_operator_dir(GPU_ARB, f"boot_{ns.tag}.logpath"), "w") as f:
        f.write(front_log + "\n")
    _write_state(state)
    log(f"LAUNCHED: P pid {spec_p.pid} (asleep) D pid {spec_d.pid} (awake) front pid {fp.pid}; state {state_path(state)}; /root/current_boot.log -> {front_log}")
    return 0



#: B4i DELETED B4g's two "named residency terms" (`XCHG_RESIDENT_ONCARD_SLOTS`
#: and `xchg_resident_region_mib`) together with the subtraction they served.
#: Both were /dev/shm byte counts standing next to a per-process NVML reading,
#: and subtracting them from it is what invented the 773 / 935 / 773 MiB
#: "unattributed residual" (1254-481, 1416-481).  The region's own size is still
#: derived where it is actually used, at `prepare_region`
#: (`xchg_residency.region_mib_from_layout`, launcher.py:3612) -- that call site
#: is the one `test_pdflip_xchg_instruments_1273`' no-literal-385 rule is about.
#: See `xchg_form_dormant_reserve` for the same-instrument arithmetic that
#: replaced the subtraction.










def front_argv_for(py: str, store_dir: str, p_pid: int, d_pid: int, dc_expect_d: Dict[str, int],
                   cards: List[Card], ns, chunk_count: int, carrier_max_tokens: int,
                   p_bs: int, d_bs: int, x_tokens: int, flip_min_work_tokens: int,
                   idle_layout_front: str,
                   src_chunk_cards: Optional[Dict[str, Dict[str, List[int]]]] = None,
                   wake_credit_plan: Optional[Dict[str, object]] = None,
                   measured_record: str = "", commit: str = "",
                   ledger_arm: Optional[Dict[str, float]] = None,
                   admin_key_file: str = "",
                   anon_preboot_bytes: int = 0,
                   front_host: str = DEFAULT_FRONT_HOST,
                   x_ceiling_tokens: int = 0,
                   x_mode_argv: Sequence[str] = ()) -> List[str]:
    """ONE front argv builder, so --dry-run prints exactly what a real boot runs.

    C2/R-6: the front is TOLD the two bs numbers and X. It never asks a
    group over HTTP -- that round trip would make the front's startup depend
    on a group the launcher has just put to sleep, for numbers the launcher
    itself wrote.
    """
    argv = [
        py, "-m", "flliper.srt.pdflip.front",
        "--prefill", f"http://127.0.0.1:{PORT_P}", "--decode", f"http://127.0.0.1:{PORT_D}",
        "--port", str(PORT_FRONT), "--host", front_host, "--awake", "D", "--tag", ns.tag,
        "--store-dir", store_dir,
        "--prefill-sid", str(p_pid), "--decode-sid", str(d_pid),
        # Task #58: the front decides image requests (route / stage / refuse)
        # and it cannot infer the mode from the groups it talks to -- their
        # tower state is in THEIR argv, not in any response the front sees.
        # Told once, here, so --dry-run prints it too.
        "--vision", str(getattr(ns, "pdflip_vision", VISION_OFF)),
        "--dc-reserve", ",".join(f"{c.uuid}={dc_expect_d.get(c.uuid, 0)}" for c in cards),
        "--weight-form", str(ns.pdflip_weight_source),  # #1444: stamps the record
    ] + (["--d-residue-context-tokens", str(group_context_tokens(ns, "d")[0])]  # YaRN x2: x1 argv unchanged
         if group_context_tokens(ns, "d")[0] > CONTEXT_LENGTH_TOKENS else []) + [
        "--fairness-w-s", str(ns.fairness_w_s),
        "--weight-chunks", str(chunk_count),
    ] + (["--weights-resident"] if getattr(ns, "flip_weights", "family") == "resident" else []) + (
        ["--dual-layout"] if getattr(ns, "dual_layout", False) else []) + (
        ["--dual-dbusy-file", dual_duty_env(ns)["FLLIPER_PDFLIP_DUAL_DBUSY_FILE"]]
        if dual_duty_env(ns) else []) + dual_priority_front_argv(ns) + (
        ["--dual-kv-ledgers", ",".join(dual_kv_ledger_paths(ns, cards))]
        if dual_kv_ledger_paths(ns, cards) else []) + [
        "--carrier-max-tokens", str(carrier_max_tokens),
        "--p-concurrency", str(p_bs),
        "--d-bs", str(d_bs),
        "--tp-prefill-max-tokens", str(x_tokens),
        "--flip-min-work-tokens", str(flip_min_work_tokens),
        "--idle-layout", idle_layout_front,
        "--drain-deadline-s", str(ns.drain_deadline_s),
    ]
    # 27B line (76e87ac3b2) / UNIFY S7: the front's own record reader (the
    # sleep-leg page-cache gate) takes the same calibration identity as the
    # launcher's readers -- handed over where the profile's records row names
    # the LINE term (qwen27b); nextflash keeps its front argv byte-identical.
    _ident = calib_identity_of(ns)
    if _ident is not None and _ident.uses_line:
        argv += ["--record-line", record_line_spec(ns)]
    # #1275: the front is handed the PATH, never the key. Its argv is as
    # world-readable as any other; the groups have no alternative (server_args
    # takes only --admin-api-key) but the front does, so it uses it.
    if admin_key_file:
        argv += ["--admin-key-file", admin_key_file]
    # H84: the live X's ceiling = group D's riegel (resolve_x_ceiling); 0 = off
    # (UNIFY S4: --x-busy-tokens follows the same rule -- only when set)
    # and the argv stays byte-identical to the one before the flag existed.
    if x_ceiling_tokens > 0:
        argv += ["--x-ceiling-tokens", str(int(x_ceiling_tokens))]
    if getattr(ns, "x_busy_tokens", None) is not None:
        argv += ["--x-busy-tokens", str(int(ns.x_busy_tokens))]
    # X-CURVES 1006: --x-mode / --x-curves / --x-curves-beyond, each only when
    # given (resolve_x_mode_launch); none = the argv byte-identical.
    argv += list(x_mode_argv)
    # #1269 fix 4 follow-up: THE PRE-BOOT ANON BASELINE, MEASURED ONCE AND
    # CARRIED. The W22 guard splits its reading into `flliper=` and `foreign=`
    # by subtracting this baseline, and the split is only subtractable because
    # it is THE SAME QUANTITY AT AN EARLIER TIME -- cgroup anon before this
    # boot's processes existed. The preflight already reads it
    # (`choose_host_ledger` -> `cg["anon"]`) and, until now, DROPPED it: boot
    # weg2sb5c's split had to be computed by hand afterwards. 0 stays "unset",
    # and the front then prints no split rather than an invented one.
    if anon_preboot_bytes > 0:
        argv += ["--anon-preboot-bytes", str(int(anon_preboot_bytes))]
    # 29.09.: the host census record the front keeps current (absent = off)
    _hcn = getattr(ns, "_host_census", None)
    if _hcn:
        argv += ["--host-census-record", str(_hcn["record"]), "--host-census-key", str(_hcn["key"])]
        if _hcn.get("store_dir"):
            argv += ["--host-census-store-dir", str(_hcn["store_dir"])]
        if _hcn.get("widths"):
            argv += ["--host-census-arena-widths", ",".join(str(int(w)) for w in _hcn["widths"])]
    if ns.min_dwell_ms is not None:
        argv += ["--min-dwell-ms", str(ns.min_dwell_ms)]
    # 27B idle policy (b)/(c): emitted only when set, so a boot without them
    # ships the front argv byte for byte as before.
    if int(getattr(ns, "d_short_drain_tokens", 0) or 0) > 0:
        argv += ["--d-short-drain-tokens", str(int(ns.d_short_drain_tokens))]
    # RC2 review (L4): --d-hold-s 0 is OFF and ships the argv of unset.
    if d_hold_active(getattr(ns, "d_hold_s", None)):
        argv += ["--d-hold-s", str(float(ns.d_hold_s))]
    if ns.d_admit_max_tokens is not None:
        argv += ["--d-admit-max-tokens", str(ns.d_admit_max_tokens)]
    # #1233 weg2dk4/fix 8 (merged into this ONE builder rather than a second
    # inline argv beside it): the chunk->card map, the measured-record sidecar,
    # the tip stamped into every measurement, and the ledger arm the front
    # derives its run-moment residual against.  The map is derived after group
    # D launches, so the --dry-run print carries it EMPTY and says so here --
    # every other term is byte-identical to the real boot's.
    # #1264: NO TERM OF THIS ARGV FOLLOWS --draft-kv-on-p, and that is a
    # CHECKED fact, not an oversight. Enumerated against the switch:
    # --src-chunk-cards is the WEIGHTS chunk->card map (derived from P's stage
    # ratio, which the switch does not move); --measured-record is the
    # dormant-image sidecar; --commit and --ledger-arm are boot identity and
    # host accounting. The front's draft handling is purely OBSERVATIONAL --
    # Front._draft_terms reads D's own draft_l3_hits/misses counters over HTTP
    # and the W9 census counts '.draft' files by name in the store -- so it
    # needs no flag to be told which arm ran: under `off` it will simply
    # report draft_pages=0 and count zero draft files, which is the honest
    # reading of that arm. The launcher's own W10/W11 SKIPPED line is what
    # names the arm; the front is not told twice.
    if src_chunk_cards:
        argv += ["--src-chunk-cards", json.dumps(src_chunk_cards, sort_keys=True)]
    if wake_credit_plan:
        # H14: the first wake's per-card tag table (W126 riegel); the front
        # reads free/floors live and reorders only an order that cycles.
        argv += ["--wake-credit-plan", json.dumps(wake_credit_plan, sort_keys=True)]
    if measured_record:
        argv += ["--measured-record", measured_record]
    if commit:
        argv += ["--commit", commit]
    if ledger_arm:
        argv += ["--ledger-arm", json.dumps(ledger_arm, sort_keys=True)]
    return argv


















def teardown(path: str, report: dict | None = None) -> int:
    """Tear down a boot from its state json. Unchanged signature for the
    ``--teardown PATH`` CLI mode (main()'s ``if ns.teardown: return
    teardown(ns.teardown)`` passes only ``path``, so ``report`` must stay
    optional). #1248: the refusal handler is the second caller -- it passes
    a dict and reads back the pid/group tallies for its one-line summary,
    rather than this function returning a different type to two callers.
    """
    st = json.load(open(path))
    print(f"[{_now()}] PDFLIP-TEARDOWN {path}")
    # #1275 fix 2 (3): THE KEY DIES WITH THE BOOT. Read from the state json --
    # a teardown runs in a FRESH process that never minted anything, so the
    # module global is empty here and the recorded path is the only source.
    set_admin_key(None, st.get("admin_key_file", ""))
    print(f"PDFLIP-TEARDOWN admin key file: {drop_admin_key_file()}")
    if st.get("tag"):  # S7b review SOLL 7: P's exec record dies with the boot
        from flliper.srt.pdflip import dual_p_restart as _s7b

        print(f"PDFLIP-TEARDOWN S7B respawn record: {_s7b.remove_spec(GPU_ARB, st['tag'])}")
    pids = set()
    for name, pid in st.get("pids", {}).items():
        if pid:
            pids |= session_pids(int(pid))
            pids.add(int(pid))
    out = subprocess.run(["pgrep", "-f", r"launch_server.*--port 3003[12]|flliper\.srt\.pdflip\.front"], capture_output=True, text=True).stdout.split()
    pids |= {int(p) for p in out}
    # H52: the logger first, by process group, with its own verdict line --
    # the helper loop below only TERMs a pid, and this is the helper that
    # outlived 107 boots.
    memts_pid = int(st.get("memts_pid") or 0)
    print(stop_memts(st.get("tag", ""), memts_pid), flush=True)
    # #135: the deadmen are session leaders -- TERM the group, so a deadman
    # caught inside `sleep` or a py-spy dump does not leave that child behind.
    helper_verdicts = []
    for hp in st.get("helper_pids", []):
        if hp and int(hp) != memts_pid:
            helper_verdicts.append(f"{int(hp)}:{_stop_helper_pid(int(hp))}")
    if helper_verdicts:
        print(f"PDFLIP-TEARDOWN helpers {' '.join(helper_verdicts)}", flush=True)
    if st.get("tag"):
        for name in ("memts", "deadman_P", "deadman_D", "deadman_front"):
            unregister_helper(st["tag"], name)
    # S104 review A2: the scheduler processes are named BEFORE the TERM (a dead pid has no cmdline)
    s104_sched = s104_scheduler_pids(pids)
    print(f"TERM {sorted(pids)}")
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    s104_escalate_schedulers(s104_sched)
    t0 = time.time()
    while time.time() - t0 < 60 and any(_alive(p) for p in pids):
        time.sleep(2)
    left = [p for p in pids if _alive(p)]
    for p in left:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    print(f"KILL leftovers {left}")
    # The MPS daemon + server of --dual-mps on are no group's child: stop them after their clients (the
    # groups above), before the cards are read again below and by the refusal's cards_empty.
    _mps_line = stop_dual_mps(st.get("tag", ""), st.get("dual_mps_root", "") or "",
                              armed=bool(st.get("dual_mps_root")))
    if _mps_line:
        print(_mps_line, flush=True)
    # H135b: the groups are gone -- their registry entries go with them, so the
    # arm's owner guard finds nothing left to stop after this teardown.
    if st.get("tag"):
        for name in BOOT_GROUP_NAMES:
            unregister_helper(st["tag"], name, suffix="grp")
    time.sleep(3)
    # #1236: the store is a DIRECTORY ON DISK, so teardown removes it instead
    # of unmounting a tmpfs. The umount pair is deleted, not kept behind a
    # flag: there is no tmpfs form left to unmount. Removing it here is the
    # symmetric act (the umount destroyed the store too); the boot-time
    # `sweep_store_residue` is the belt-and-braces for a teardown that never
    # ran, e.g. after a crash. These are DISK bytes -- nothing freed here was
    # ever charged to the host reap ledger.
    store_dir = st.get("store_dir", "")
    if store_dir and os.path.isdir(store_dir) and l3_persistent_store(store_dir):
        # L3P-TEARDOWN (27B rc12z7b b1 08:01Z: 6.5 GB / 172795 files removed
        # here, b5 found `identity=new files=0`): a persistent L3 store is
        # the one thing a teardown must NOT remove -- it is what the next
        # boot reattaches to. Only the per-boot store of the pre-L3P form
        # (no identity file, no ``l3-`` name) goes.
        alloc, _apparent, count = _tree_bytes(store_dir)
        print(f"store dir {store_dir} KEPT (L3 persistent: {alloc} B allocated, {count} files)")
    elif store_dir and os.path.isdir(store_dir):
        alloc, _apparent, count = _tree_bytes(store_dir)
        shutil.rmtree(store_dir, ignore_errors=True)
        print(f"store dir {store_dir} removed ({alloc} B allocated, {count} files; DISK, "
              f"not host RAM)")
    # C18: the per-card ring files are tmpfs pages charged to this boot's host
    # ledger.  Leaving them behind would carry Sigma H of RAM into the NEXT
    # boot's baseline, where nothing names it.
    ring_dir = st.get("ring_dir", "")
    if ring_dir and os.path.isdir(ring_dir):
        for name in os.listdir(ring_dir):
            try:
                os.unlink(os.path.join(ring_dir, name))
            except OSError:
                pass
        try:
            os.rmdir(ring_dir)
        except OSError:
            pass
        print(f"host ring dir {ring_dir} removed")
    # C14 / FIX 2 round 2, finding 1: the per-card VRAM credit counters are
    # /dev/shm pages of THIS boot, and a terminal one left behind
    # (leg_complete + a whole image of credit) is what the next boot's flip of
    # the same index used to read as its own funding.  The composed epoch makes
    # that harmless; removing the file makes it absent.  Two halves, because
    # teardown alone cannot clean up after a boot that crashed -- the launch
    # sweep is the other half (sweep_dead_credit_counters).
    # SCOPED TO THIS BOOT (FIX 3 round 3): the nonce is the ring epoch this
    # boot's state file recorded, so tearing down boot A cannot unlink a live
    # boot B's counters.
    nonce = str(st.get("ring_epoch", ""))
    who = nonce or ("UNKNOWN -- none removed; a boot whose ring never armed "
                    "wrote no counters, and an unscoped sweep here would take "
                    "another boot's")
    print(f"vram credit counters removed (boot {who}): "
          + str(remove_vram_credit_counters(boot_nonce=nonce)))
    # Same one reader as everywhere else, and the line says which figure it is:
    # a teardown that prints "0 MiB used" from a carve-out-blind subtraction
    # would be the same lie in the other direction.
    try:
        snap = nvml_registry.memory_snapshot()
        inst = (
            nvml_registry.FREE_INSTRUMENT_V2
            if all(m.carve_out_known for _d, m in snap)
            else nvml_registry.FREE_INSTRUMENT_V1
        )
        print(
            f"cards after teardown (instrument: {inst}, allocatable): "
            + ", ".join(
                f"nvml{d.index} {m.tenant_used_mib} MiB used by processes / {m.free_mib} MiB free "
                f"({m.reserved_mib} MiB driver-reserved)"
                for d, m in snap
            )
        )
    except Exception as e:  # noqa: BLE001 - a teardown print never fails a teardown
        print(f"cards after teardown: NVML unreadable ({e})")
    if report is not None:
        # The SAME pid set this function just TERMed/KILLed, and the SAME
        # group names the state json recorded -- not a re-derivation, so the
        # refusal handler's printed tallies cannot drift from what actually
        # happened above.
        report["pids"] = sorted(pids)
        report["groups"] = sorted(name for name, pid in st.get("pids", {}).items() if pid)
    return 0










#: Every refusal class that must leave this launcher as the ONE named line and
#: exit 2, never as a traceback.  ``PdFlipRingRefused`` is a BASE class on purpose
#: (FIX 2): the previous list enumerated its members, ``PdFlipRingNeedsInterleave``
#: was not among them, and an explicitly requested ``--ring-form MAP_SHARED`` on
#: this rig therefore exited 1 with a stack trace -- which any wrapper keying on
#: the exit code reads as a crash rather than as the refusal it is.  A refusal
#: added to ring_table now inherits this handler instead of needing a line here.
#: #1233 fix 8: W21 (PdFlipHostRunPeakRefused) is a refusal of the SAME standing
#: as W20 -- an arm whose predicted RUN PEAK is not below the observed reap
#: point does not boot -- so it joins this tuple instead of growing a second
#: `except` clause beside the one handler.
#: #1273 S7 round-2 review F1: ``xchg_residency.PdFlipXchgRefused`` (W71) joins
#: it for the SAME reason W34 once did not -- it shipped outside this tuple and
#: therefore exited 1 with a traceback and dropped no admin key. It is the
#: module's BASE class, not the leaf, so the FIX 2 lesson holds one module on:
#: a future exchange refusal inherits the handler instead of needing a line.
REFUSALS = (PdFlipLaunchRefused, ring_table.PdFlipRingRefused,
            host_ledger.PdFlipHostLedgerRefused, host_ledger.PdFlipHostRunPeakRefused,
            xchg_residency.PdFlipXchgRefused, pdflip_form.PdFlipFormContradiction)


def cli(argv: Optional[Sequence[str]] = None) -> int:
    """``main`` plus the one refusal handler.  Returns the exit code.

    A function rather than a bare ``__main__`` block so the handler is
    REACHABLE FROM A TEST: the defect it fixes was an except list that had gone
    out of step with the exceptions raised, and nothing could see it.

    #1248: a refusal raised AFTER a group has already spawned (W7/W9/W10/W45,
    W62, ...) used to stop here -- print, drop the key, exit 2 -- leaving
    whatever OS process(es) main() had already started running (weg2dec1
    arms 2/3: W62, which was W53 on that tree and is renumbered here because
    W53 is PdFlipStoreHandbackFailed on this one, left group P serving on
    :30031, ~1.4 GiB held on the 5090,
    cleared only by a MANUAL teardown against the state json --
    BOOT_weg2dec1_0909.md). This is the SAME funnel, not a second one: a
    refusal before any spawn still takes the fast exit below unchanged,
    because ``_ACTIVE_BOOT_STATE.pids`` is still empty at that point.
    """
    def _say(line: str) -> None:
        print(f"[{_now()}] {line}", flush=True)

    try:
        rc = main(argv)
    except REFUSALS as e:
        print(f"[{_now()}] PDFLIP-LAUNCH REFUSED: {e}", flush=True)
        # IPC §2.2: the refusal is the boot's cause, written where the arm reads
        # it -- not only printed for a later grep.
        boot_state_write(_say, "dead", cause=refusal_cause(e))
        # #1275 fix 2 (3): the KILLER path drops the key too. weg2sb5 refused
        # here and left its key file behind -- a boot that never served, whose
        # secret outlived it. Both exits, or the guarantee is only half true.
        print(f"[{_now()}] PDFLIP-LAUNCH admin key file: {drop_admin_key_file()}", flush=True)
        st = _ACTIVE_BOOT_STATE
        spawned = {name: pid for name, pid in (st.pids if st else {}).items() if pid}
        if not spawned:
            # Refused before the first launch_group() call -- nothing to tear
            # down. Today's behavior, unchanged -- except the memts logger,
            # which starts BEFORE group P (H52): stop it here, not teardown().
            if st is not None and st.memts_pid:
                print(f"[{_now()}] {stop_memts(st.tag, st.memts_pid)}", flush=True)
            stop_dual_mps_of_state(st, log=lambda line: print(f"[{_now()}] {line}", flush=True))
            return 2
        # #1248: at least one group is already running. Go through the SAME
        # teardown() the operator's `--teardown PATH` mode and the killer use
        # -- TERM the pid set (+ the pgrep fallback), wait, KILL leftovers,
        # unmount, drop the ring dir and vram credit counters -- not a
        # second, parallel cleanup path. Flush first: state.pids["front"] is
        # set one statement before its next _write_state() (a real gap in
        # main()), and teardown() reads the FILE, not this live object.
        _write_state(st)
        td_report: dict = {}
        teardown(state_path(st), report=td_report)
        cards_empty = _cards_empty(st.cards)
        print(
            f"PDFLIP-LAUNCH REFUSED {_w_code_of(e)} teardown=done "
            f"groups={len(spawned)} pids={len(td_report.get('pids', []))} "
            f"cards_empty={cards_empty}",
            flush=True,
        )
        return 2
    except Exception as e:
        # Not a refusal: a launcher crash. Named in state.json, then re-raised
        # unchanged (traceback and exit code stay what they were).
        boot_state_write(_say, "dead", cause=state_file_mod.make_cause(
            f"LAUNCHER_EXCEPTION_{type(e).__name__}", "launcher", repr(e), exception_type=type(e).__name__))
        _st = _ACTIVE_BOOT_STATE
        if _st is not None and not any(_st.pids.values()):
            # no group spawned: nobody is an MPS client yet, the daemon only holds the pipe dir. With a group
            # running it stays up -- stopping it under live clients is undefined; teardown() ends both then.
            stop_dual_mps_of_state(_st, log=lambda line: print(f"[{_now()}] {line}", flush=True))
        raise
    boot_state_write(_say, event=("launcher_done", {"rc": rc}))
    return rc








# a setattr on this module (tests, propose_oracle, hw_sim) must reach the launcher_* modules that carry the same name
from flliper.srt.pdflip import launcher_facade as _launcher_facade  # noqa: E402

_launcher_facade.install(__name__)


if __name__ == "__main__":
    raise SystemExit(cli())
