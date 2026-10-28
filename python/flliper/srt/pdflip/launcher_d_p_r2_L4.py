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

from flliper.srt.managers import corridor_guard
from flliper.srt.name_compat import canonical_env
from flliper.srt.pdflip import DEFAULT_D_BS
from flliper.srt.pdflip import budget_rest as _budget_rest
from flliper.srt.pdflip import card_identity
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import seam_digest
from flliper.srt.pdflip import vote_fixed_live as _vfl
from flliper.srt.pdflip import weight_exchange
from flliper.srt.pdflip import weight_exchange_region
from flliper.srt.pdflip import weight_exchange_shadow
from flliper.srt.pdflip import weight_exchange_transport
from flliper.srt.pdflip import xchg_manifest
from flliper.srt.pdflip.launcher_d_leaf import EARLY_READ_FACTS
from flliper.srt.pdflip.launcher_d_leaf import VISION_OFF
from flliper.srt.pdflip.launcher_d_leaf import VISION_TRANSIENT
from flliper.srt.pdflip.launcher_d_leaf import _d_argv_of_front_log
from flliper.srt.pdflip.launcher_d_leaf import d_row_attn_units
from flliper.srt.pdflip.launcher_d_mid import D_RANK_SOLVE_MARKER
from flliper.srt.pdflip.launcher_d_p_r2_L0 import EVIDENCE_DIR
from flliper.srt.pdflip.launcher_d_p_r2_L0 import GPU_ARB
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_CHUNKED_PREFILL_TOKENS
from flliper.srt.pdflip.launcher_d_p_r2_L0 import hostsample_csv_path
from flliper.srt.pdflip.launcher_d_p_r2_L1 import d_extend_vote_fixed_live_material
from flliper.srt.pdflip.launcher_d_p_r2_L1 import p_draft_kv_flags
from flliper.srt.pdflip.launcher_d_p_r2_L2 import argv_p
from flliper.srt.pdflip.launcher_d_p_r2_L2 import hicache_draft_tier_env
from flliper.srt.pdflip.launcher_d_p_r2_L2 import p_prefill_transient_vector_mib
from flliper.srt.pdflip.launcher_p_d_r2_L3 import P_PREFILL_TRANSIENT_ENV
from flliper.srt.pdflip.launcher_p_d_r2_L3 import RING_FORM_SENTINEL_STORE_CFG
from flliper.srt.pdflip.launcher_p_d_r2_L3 import argv_d
from flliper.srt.pdflip.launcher_p_d_r2_L3 import d_extend_vote_fixed_live_verdict
from flliper.srt.pdflip.launcher_p_d_r2_L3 import d_plan_inputs
from flliper.srt.pdflip.launcher_p_d_r2_L3 import p_prefill_transient_for
from flliper.srt.pdflip.launcher_p_l3_leaf import FP8_UNIFORM_MARLIN_ENV
from flliper.srt.pdflip.launcher_p_l3_leaf import _gcd_reduce
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst_boots
from flliper.srt.pdflip.launcher_p_l3_leaf import parse_group_env
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import spec_form_env
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import user_reserve_env_entries
from flliper.srt.pdflip.launcher_refuse_proc_leaf import PROFILE_QWEN27B
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _refuse_if_transient_mm_hash_unstable
from flliper.srt.pdflip.launcher_refuse_proc_leaf import set_group_env
from flliper.srt.pdflip.launcher_resolve_xchg_r2_L1_1 import HICACHE_DRAFT_TIER_ENV
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_PLACE_AUTO
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_SOURCE_DISK
from flliper.srt.pdflip.launcher_xchg_vision_leaf import vote_fixed_live_root
from flliper.srt.pdflip.vision_stage_boot import VISION_ENV as VISION_STAGE_ENV
import shlex

#: Environment keys the bar1 build path owns; dropped with the flags so the
#: NCCL arm is not half-configured for a transport it does not run.
BARLINK_ENV_KEYS = ("FLLIPER_BARLINK_BUILD_WINDOW_CAP_S",)
#: #1235 THE UNOWNED ENV LITERALS, PROMOTED TO FLAGS AND TO LAUNCHER OUTPUT.
#:
#: All four used to be written as ``env.get(KEY, "<literal>")``, which is the
#: shape the ring family was already fixed out of (R19): an inherited value
#: from the launcher's own shell wins SILENTLY, and the boot then runs a number
#: no flag, no record and no log line ever mentioned. They are now flags with
#: defaults, written unconditionally, exactly like TMS_HOST_RING_* and
#: FLLIPER_PDFLIP_PCIE_DUPLEX -- launcher OUTPUT, never operator input.
#:
#: Their VALUES are unchanged and none of them is measured; they are timeouts
#: and a census stride, and the help text says so rather than implying a
#: provenance they do not have.
BARLINK_BUILD_WINDOW_CAP_S = 60
PP_CHAIN_RECV_STALL_S = 60
PP_OCCUPANT_HORIZON_S = 90
MATCH_REFUSAL_CENSUS_EVERY = 64
#: H103: the FLA l2norm run-time-T switch (fla/l2norm.py ``L2NORM_RUNTIME_T_ENV``,
#: same name, pinned by the test; spelled here so the launcher never imports
#: triton/torch). build_env sets it for both groups.
L2NORM_RUNTIME_T_ENV = "FLLIPER_FLA_L2NORM_RUNTIME_T"
VISION_SOURCE_ENV = "FLLIPER_PDFLIP_VISION_SOURCE"
#: user order 27.10.: group D's own vision stage. Published by build_env for group D ONLY, from the ONE launcher decision
#: (:func:`vision_on_d_of`, the front's ``vision_on_d``) -- the group never infers it from other variables (the launcher pops
#: FLLIPER_PDFLIP_VISION for D, and the dual layout is published only in the dual env)
VISION_D_ENV = "FLLIPER_PDFLIP_VISION_D"
VISION_PLACE_ENV = "FLLIPER_PDFLIP_VISION_PLACE"
_KV_POOL_SIZING_RE = re.compile(
    r"TP(\d+)\] KV pool sizing: available_bytes=(\d+) .*?cell_size=(\d+)")

def build_env(tree: str, venv: str, cvd: str, store_dir: str, debug_hold: bool, tag: str,
              chunk_layers: int = 0, chunk_count: int = 0, tms_so: str = "",
              transport: str = "bar1", ring: Optional["HostRingPlan"] = None,
              barlink_build_window_cap_s: int = BARLINK_BUILD_WINDOW_CAP_S,
              pp_chain_recv_stall_s: int = PP_CHAIN_RECV_STALL_S,
              pp_occupant_horizon_s: int = PP_OCCUPANT_HORIZON_S,
              match_refusal_census_every: int = MATCH_REFUSAL_CENSUS_EVERY,
              arming_floor_solved: bool = True,
              hicache_bigram_keys: bool = True,
              hicache_flush_publish_sweep: bool = True,
              seam_digest_armed: bool = False,
              group: str = "",
              lane_coverage_dir: str = "",
              lane_coverage_token: str = "",
              # #1395: the generic sibling of `lane_coverage_token` -- see
              # `pdflip_boot_token`'s own docstring for why this one is safe to
              # publish UNCONDITIONALLY where that one may not be.
              pdflip_boot_token: str = "",
              # #107 DIE EXPERTEN-KARTE, fuer BEIDE Gruppen dieselbe Datei.
              # Ohne sie liest `expert_store.expert_map()` nichts, `_karte`
              # bleibt None und `_expert_store_rows_for` faellt auf die
              # GLOBALE Residenzmenge zurueck -- eine Zahl fuer alle Karten,
              # die die Residenz der jeweils anderen Raenge wegwirft
              # (Nutzer 22.09.: "da wird dort vergessen ALLE karten zu
              # zaehlen"). Gemessen w128: Hotset 1 Id -> 511 Store-Slots;
              # die Karte rechnet fuer dieselbe Form 354.
              expert_map_path: str = "",
              # H2c: wessen Checkpoint und Karte die Store-Sentinels dieses
              # Boots belegen (expert_store.compute_identity); BEIDE Gruppen
              # dieselbe Zeichenkette, sonst adoptiert D nichts.
              expert_store_identity: str = "",
              # Task #58: which vision form this boot runs. Published as
              # FLLIPER_PDFLIP_VISION for the P group ONLY (see below).
              vision: str = VISION_OFF,
              profile: str = PROFILE_QWEN27B,
              flip_weights: str = "family",
              # H125: the transient stage's source and place, P only; the
              # defaults publish nothing (byte-identical env).
              vision_source: str = VISION_SOURCE_DISK,
              vision_place: str = VISION_PLACE_AUTO,
              # user order 27.10.: D encodes images in the flip layouts (front.vision_on_d); published for group D only
              vision_on_d: bool = False,
              group_env_extra: Optional[Dict[str, str]] = None,
              xchg_env: Optional[Dict[str, str]] = None,
              # PDFLIP-FORM (24.09.): the ONE resolved form of this boot
              # (``pdflip_form.resolve_form``), gathered by ``_env_knobs``. None
              # = a desk caller: every form-keyed branch below keeps its
              # pre-form behaviour and FLLIPER_PDFLIP_FORM is popped.
              boot_form: Optional["pdflip_form.PdFlipForm"] = None,
              # 27B FP8 (--fp8-uniform-marlin): one byte layout on every rank.
              fp8_uniform_marlin: bool = False,
              # #236: the operator's per-card external headroom
              # (``--user-reserve-mib``, ``{card_uuid: MiB}``), handed to EVERY
              # rank of the group through the group env -- see the block at the
              # end of this function. None = a desk caller / no reserve stated.
              user_reserve_by_card: Optional[Dict[str, int]] = None,
              # #236 review MUSS-1: the group's cards and booked awake-rest vector
              # (None for P and desk callers) -- see ``effective_reserve_by_card``.
              reserve_cards: Optional[Sequence["Card"]] = None,
              reserve_booked_rest_mib: Optional[Sequence[Optional[int]]] = None) -> Dict[str, str]:
    # Rename 1b: ONE spelling per variable from here on -- the one this tree
    # reads (name_compat.canonical_env). Every pop, the PHASE_FLIP prefix strip
    # and the per-group overrides below act on that name; an operator's
    # legacy/renamed spelling is folded onto it first, so none survives a pop
    # to be mirrored back by the rank's own import. Foreign readers
    # (sgl_kernel, JIT C++, TMS) keep their legacy spelling as well.
    env = canonical_env(dict(os.environ))
    # Task #58: THE ARMING SIGNAL for the transient vision stage, and the ONE
    # thing that turns `vision_stage_service`'s seam from a no-op into a stage.
    # Read by `pdflip/vision_stage_boot.vision_mode`, which the P group's
    # tokenizer process calls once.
    #
    # SAME DISCIPLINE AS FLLIPER_PDFLIP_GROUP AND THE RING FAMILY (R19): launcher
    # OUTPUT, published only when this call names the transient form AND the P
    # group, POPPED otherwise -- a value inherited from the operator's shell
    # must never arm a stage nobody asked for, and a D rank that armed one
    # would spawn a CUDA-context probe and load a tower on a group that has no
    # use for either.
    #
    # P ONLY, and that is the design's decision, not a shortcut: the transient
    # stage exists because the P layout has no tower and needs the rows before
    # its prefill (DESIGN_VISION_TRANSIENT_0920 §6). The front forces image
    # requests to the long/P route for the same reason.
    if vision == VISION_TRANSIENT and group == "P":
        env[VISION_STAGE_ENV] = VISION_TRANSIENT
    else:
        env.pop(VISION_STAGE_ENV, None)
    # H125: SAME DISCIPLINE -- launcher output, P of a transient boot only,
    # and only when not the default; popped everywhere else so an operator's
    # shell value can never pick a source the argv does not say.
    if vision == VISION_TRANSIENT and group == "P" and vision_source != VISION_SOURCE_DISK:
        env[VISION_SOURCE_ENV] = vision_source
    else:
        env.pop(VISION_SOURCE_ENV, None)
    if vision == VISION_TRANSIENT and group == "P" and vision_place != VISION_PLACE_AUTO:
        env[VISION_PLACE_ENV] = vision_place
    else:
        env.pop(VISION_PLACE_ENV, None)
    if group == "D" and vision_on_d:
        env[VISION_D_ENV] = "1"
    else:
        env.pop(VISION_D_ENV, None)
    _refuse_if_transient_mm_hash_unstable(vision, env, group)
    # #1348: the exchange lane's unexecuted-line instrument. SAME DISCIPLINE
    # as FLLIPER_PDFLIP_GROUP and the host-ring family below (R19): this is
    # LAUNCHER OUTPUT, published only when `--xchg-coverage-diff` named a
    # directory, and POPPED otherwise -- so a value left in the operator's own
    # shell can never arm a coverage.py line tracer (2x-10x on the traced
    # modules) on an acceptance boot. That popping is the whole reason the
    # switch is a FLAG and not an ambient environment variable, which is where
    # `managers/seam_coverage.py`'s switch differs and must keep differing.
    if lane_coverage_dir:
        env["FLLIPER_PDFLIP_LANE_COVERAGE_DIR"] = lane_coverage_dir
        env["FLLIPER_PDFLIP_LANE_COVERAGE_TOKEN"] = lane_coverage_token
    else:
        env.pop("FLLIPER_PDFLIP_LANE_COVERAGE_DIR", None)
        env.pop("FLLIPER_PDFLIP_LANE_COVERAGE_TOKEN", None)
    # FIX 2, finding 1: WHICH WEG-2 GROUP THIS RANK BELONGS TO, and the only
    # thing in either tree that says so.  Read by
    # `pdflip_memory_saver.pdflip_group_name()`; it is the discriminator the
    # graph-tag coupling gates on, because the alternative it used first
    # (`server_args.enable_memory_saver`) is an UPSTREAM flag that upstream
    # engines set too -- gating a Weg-2-only widening of an upstream RPC on it
    # changed stock `/release_memory_occupation` semantics for everyone.
    #
    # `server_args.pdflip_group` was believed to be this fact and is not: it is
    # read in exactly one place (`weight_updater._pdflip_group_name`) with a
    # `getattr(..., "")` default and is ASSIGNED NOWHERE, so it has answered
    # "?" on every rank of every boot.
    #
    # Same discipline as the ring family above: published only when this call
    # names a group, and POPPED otherwise, so a value inherited from the
    # operator's own shell can never arm a coupling nobody asked for.
    if group:
        env["FLLIPER_PDFLIP_GROUP"] = group
    else:
        env.pop("FLLIPER_PDFLIP_GROUP", None)
    # 24.09. (weg2xsn412, first 27B boot on the NF line): this block used to
    # sit BETWEEN the `if group:` above and its `else:`, so the pop answered
    # "no expert map" instead of "no group" -- every boot without a map (the
    # whole 27B profile) lost FLLIPER_PDFLIP_GROUP and the exchange plan provider
    # refused at the first release (W68 -> W84 -> W29 on all three P ranks).
    if expert_map_path:
        # BEIDE Gruppen, derselbe Pfad -- das ist der ganze Punkt der Karte.
        env["FLLIPER_MOE_EXPERT_MAP"] = expert_map_path
    # H2c: Launcher-Ausgabe wie die Karte -- ohne Karte keine Identitaet, und
    # ein Wert aus der Shell des Operators darf nie einen fremden Store adeln.
    if expert_map_path and expert_store_identity:
        env["FLLIPER_MOE_EXPERT_STORE_IDENTITY"] = expert_store_identity
    else:
        env.pop("FLLIPER_MOE_EXPERT_STORE_IDENTITY", None)
    # C18: the shared host granule ring (spec C1-C8).  These four variables are
    # LAUNCHER OUTPUT, never operator input (R19): every size in them is solved
    # by ring_table from the previous boot's own lines, and the whole family is
    # absent when no form is proven.  THAT ABSENCE IS NOT A FALLBACK (spec
    # Amendment A1-3, and FIX 3 round 3 retracting the sentence that said it
    # was): on this host budget no other flip form is feasible, so a boot that
    # proves no form REFUSES by name and exits 2 -- the popped family is what a
    # refused boot leaves behind, never a configuration anything runs under.
    if ring is not None and ring.armed:
        env["TMS_HOST_RING_DIR"] = ring.dir
        env["TMS_HOST_RING_MAP"] = ring.env_map
        env["TMS_HOST_RING_EPOCH"] = str(ring.epoch)
        env["TMS_HOST_RING_FORM"] = ring.form
    else:
        for key in ("TMS_HOST_RING_DIR", "TMS_HOST_RING_MAP", "TMS_HOST_RING_EPOCH",
                    "TMS_HOST_RING_FORM"):
            env.pop(key, None)
    # C12/C13: the per-card duplex ratio, same class of variable and the same
    # rule -- launcher OUTPUT solved from a measured file, never operator input.
    # Absent means no card splits its PCIe lock key, which is the conservative
    # direction; it is popped rather than left inherited so a stale value from
    # the launcher's own environment can never open a split nobody measured.
    if ring is not None and ring.duplex_env:
        env["FLLIPER_PDFLIP_PCIE_DUPLEX"] = ring.duplex_env
    else:
        env.pop("FLLIPER_PDFLIP_PCIE_DUPLEX", None)
    # #1233 one-backup flip: chunked weights tags (pdflip_memory_saver.py) and
    # the patched torch_memory_saver preload hook (tms_csrc/PATCH.md).
    if chunk_layers > 0 and chunk_count > 0:
        env["FLLIPER_PDFLIP_WEIGHT_CHUNK_LAYERS"] = str(chunk_layers)
        env["FLLIPER_PDFLIP_WEIGHT_CHUNKS"] = str(chunk_count)
    if tms_so:
        env["FLLIPER_PDFLIP_TMS_PRELOAD_SO"] = tms_so
    # #1273 S3/S5: the staging region and the weight-source mode, published to
    # BOTH groups by the same two names the region's own module reads
    # (FLLIPER_PDFLIP_XCHG_REGION / _BOOT / FLLIPER_PDFLIP_WEIGHT_SOURCE).  SAME
    # DISCIPLINE AS THE RING FAMILY ABOVE and for the same measured reason: it
    # is launcher OUTPUT, so it is POPPED when this boot arms no region -- a
    # value inherited from the operator's shell must never arm a coupling
    # nobody asked for, and a stale region path is a rank mapping another
    # boot's shm.
    for key in ("FLLIPER_PDFLIP_XCHG_REGION", "FLLIPER_PDFLIP_XCHG_BOOT",
                weight_exchange_region.ENV_N_CARDS,
                "FLLIPER_PDFLIP_WEIGHT_SOURCE",
                # S5b: the shadow's hop bound is launcher OUTPUT too, so an
                # operator's inherited shell value may not silently regrade a
                # boot that arms no shadow at all.
                weight_exchange_shadow.ENV_HOP_BOUND_MS,
                # S6: the on-card arm is launcher OUTPUT for the same reason --
                # an inherited ``FLLIPER_PDFLIP_XCHG_ONCARD=host`` would put ranks
                # on an arm this boot's ledger charged nothing for.
                weight_exchange_transport.ENV_ONCARD_MODE,
                # S6 fix E: the slot ceiling is launcher OUTPUT too -- an
                # inherited value would size a deposit this boot's ledger
                # charged a different worst case for.
                weight_exchange_transport.ENV_ONCARD_SLOT_MIB,
                # #1350: the seam grader's arm, launcher OUTPUT for the
                # sharpest version of the same reason -- an inherited
                # ``FLLIPER_PDFLIP_SEAM_DIGEST=1`` would put a whole-shard hash on
                # the critical path of every flip leg of a SERVING boot, i.e.
                # it would turn an instrument into a performance regression
                # nobody asked for.  Popped first, set below only when the
                # flag is on.
                seam_digest.ENV_ARM,
                # #1330 B4n: the direction knob is launcher OUTPUT for the
                # sharpest version of this reason -- an inherited
                # ``FLLIPER_PDFLIP_XCHG_LEGS=pp_to_tp`` would SILENTLY REMOVE
                # half the legs of a boot that asked for both, and a skipped
                # leg reads in a log exactly like a leg that ran clean.
                weight_exchange.XCHG_LEGS_ENV,
                # #1330 B4n: launcher OUTPUT, same rule -- an inherited
                # directory would make a ring boot write placement manifests
                # into whatever path the operator's shell happened to name.
                xchg_manifest.DIR_ENV):
        env.pop(key, None)
    # #1350: armed ONLY by ``--pdflip-seam-digest``, and on any weight-source
    # arm -- the grader asks "did the round trip return my bytes", which is a
    # question the ring carrier has to answer too, so tying it to the exchange
    # arms would have made the ring's own round trip ungraded.
    if seam_digest_armed:
        env[seam_digest.ENV_ARM] = "1"
    for key, value in (xchg_env or {}).items():
        env[str(key)] = str(value)
    cu13 = f"{venv}/lib/python3.12/site-packages/nvidia/cu13/lib"
    # boot_855_train0901.sh:152-153 (NVRTC) and S1 killer K1: the memory
    # saver's cu13 preload hook links libcudart.so.13, which must be on the
    # loader path at LD_PRELOAD time or every rank dies exit 127 before main().
    env["LD_LIBRARY_PATH"] = cu13 + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = f"{tree}/python"
    env["CUDA_VISIBLE_DEVICES"] = cvd
    env["FLLIPER_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "0"  # K2: pure liveness /health
    # #1233 boot weg2ls3b3: group D (NEXTN) keys the store by BIGRAM page
    # hashes, group P (no spec) by UNIGRAM -- disjoint chains for the same
    # prompt, D never read P's pages. One key scheme for both groups; a
    # no-op on D (already bigram), forces bigram on P.
    if hicache_bigram_keys:
        env["FLLIPER_HICACHE_BIGRAM_KEYS"] = "1"
    else:
        env.pop("FLLIPER_HICACHE_BIGRAM_KEYS", None)
    # #1233 zero-remainder: /flush_cache (the front's quiesce before a sleep)
    # first publishes every un-backed device node to the store, so a chain
    # the write-through pin budget declined mid-prefill is not lost at the
    # flip (UnifiedRadixCache.publish_unbacked_sweep). Both groups.
    if hicache_flush_publish_sweep:
        env["FLLIPER_HICACHE_FLUSH_PUBLISH_SWEEP"] = "1"
    else:
        env.pop("FLLIPER_HICACHE_FLUSH_PUBLISH_SWEEP", None)
    # 2026-09-15 (boot xsn129, D's leg-2 store reads 100-430 pages/s while the
    # raw file reads run at 15k files/s): with the #720 read-buffer ring OFF
    # every page read allocates a fresh PINNED host tensor (cudaHostAlloc +
    # cudaFreeHost per page, under the driver lock the decode loop shares).
    # The ring is allocated once and reused; 256 page buffers are a few MiB.
    # An operator value in the environment wins.
    env.setdefault("FLLIPER_HICACHE_READ_BUFFERS", "256")
    # Task #3 (17.09.): the arena re-admission probes 128 pages per call;
    # 1024 cuts the call count 8x on the read path's critical section
    # (xsn246: 2048 calls, find_ms=1037). An operator value wins.
    env.setdefault("FLLIPER_HICACHE_STORAGE_BATCH", "1024")
    # Task #3: the arena -> device gather after a wake runs on an idle card;
    # 16 blocks instead of the interference-tuned 2 (measured next boot,
    # PDFLIP-LOAD-DEVICE). An operator value wins.
    env.setdefault("FLLIPER_HICACHE_ARENA_LOAD_BLOCK_QUOTA", "16")
    # #1402: the store read path is bounded by the GIL hand-off after every
    # syscall (see run_scheduler_process); 0.2 ms measured 1.07 ms/page vs
    # 20.6 ms/page at Python's 5 ms default beside a busy main thread.
    env.setdefault("FLLIPER_GIL_SWITCH_INTERVAL_MS", "0.2")
    # BAUPLAN_SHM_ARENA_0916: ONE HiCache for all six ranks -- the shared
    # page arena under /dev/shm, one directory per boot tag, opened by every
    # rank of both groups. The disk store stays the cold tier behind it.
    # FLLIPER_HICACHE_ARENA_DIR="" in the operator's environment disables it.
    if "FLLIPER_HICACHE_ARENA_DIR" not in env:
        env["FLLIPER_HICACHE_ARENA_DIR"] = f"/dev/shm/pdflip-arena-{tag}"
    # 16 GiB = 524k KV pages: three ~100k prefixes fit without eviction
    # (boot xsn153: three 80k prefixes overflowed 8 GiB, evictions dropped
    # pages other ranks had written, followers found "store_absent").
    # #1431 (xsn193): the park test's working set is 6 x 100k tokens x 32 KiB
    # = 19.5 GiB and every page a live tree node references stays in the
    # arena (the host copy IS the slot); 16 GiB ran full. 28 GiB = 917k
    # slots at 28; 24 GiB = 786k slots is enough for the 600k and leaves the
    # mamba arena its share: the ledger's last dry-run had 18.7 GiB of
    # headroom (75.74 of 94.43 GiB) for both.
    env.setdefault("FLLIPER_HICACHE_ARENA_GIB", "22")  # #1432: 720k slots, fits the priced headroom with the mamba arena
    # #1424 Stufe 3: the host tier IS the arena (rows = slots, reads in place);
    # the per-rank pool shrinks to a 1 GB staging ring for the write side.
    env.setdefault("FLLIPER_HICACHE_ARENA_HOST", "1")
    env.setdefault("FLLIPER_HICACHE_ARENA_STAGING_GB", "0.05")  # #1430: fallback range only
    # #1410: 48 thrashed on 3x100k (xsn159). #1431: one anchor state per 4096
    # tokens x 6 x 100k = ~150 live blobs in the park test, 128 ran full;
    # 192 x 46.76 MiB = 8.8 GiB (with 24 GiB KV: 32.8 GiB shm, fundable).
    env.setdefault("FLLIPER_HICACHE_ARENA_MAMBA_SLOTS", "140")  # #1432: 140 x 74.8 MiB = 10.2 GiB, priced; L3 return path covers the rest
    # write_back + bubble publisher (pdflip_bubble_publish, 2026-09-16): the
    # publish sweep runs bounded in the PP loop's bubbles, nothing is left
    # for the flip's flush. "0" in the operator's environment disables it.
    env.setdefault("FLLIPER_PDFLIP_BUBBLE_PUBLISH", "1")
    # H103: FLA l2norm with the row count as a RUN-TIME bound for BOTH groups
    # (fla/l2norm.py L2NORM_RUNTIME_T_ENV, the 27B strand's switch, 6b24aa60da;
    # there --p-host-overlap sets it for P only). The stock kernel takes T as
    # tl.constexpr: rc9p D-TP0 loaded it 15x after READY, 16.7 s of cold-load
    # windows (one 12.7 s), P 78x. Bit-identical output (same body). "0" in
    # the operator's environment keeps the stock kernel.
    env.setdefault(L2NORM_RUNTIME_T_ENV, "1")
    if arming_floor_solved:
        env["FLLIPER_ARMING_FLOOR_SOLVED"] = "1"
    else:
        env.pop("FLLIPER_ARMING_FLOOR_SOLVED", None)
    # Item `dormant` (record [1y]): the ONE switch that puts the CUDA-graph
    # capture pool and the flashinfer FLOAT workspace inside the memory saver's
    # `cuda_graph` region, so the sleeping group hands their physical pages
    # back and the wake remaps them at stable virtual addresses -- no
    # recapture, which RESTORE-NEVER-REBUILD forbids inside a cutover.
    # Upstream's own flag, read at the CAPTURE site
    # (full_cuda_graph_backend.py:78-81) and, since that slice's commit 2, at
    # the SLEEP site (pdflip_memory_saver.pdflip_graph_tag_armed); both groups get
    # it so the two sides can never disagree.  Its fix 2: this flag ALONE no
    # longer arms the sleep-side coupling -- FLLIPER_PDFLIP_GROUP above is the
    # Weg-2 conjunct, so setting this variable on a stock engine gets stock
    # behaviour, which is what it always meant upstream.  Expected release per
    # rank from [1y]: 384 MiB workspace + 92-133 MiB capture pool, measured on
    # the boot by `PDFLIP-SLEEP released tags=['cuda_graph'] mib=`.
    #
    # Compatible with this boot's spec config: adaptive graph memory is the one
    # mechanism declared mutually exclusive with this flag
    # (adaptive_graph_memory.py:390-394), and it resolves through `auto`, which
    # LOGS and degrades to 'offload-scratch' rather than raising -- and is not
    # even reached here, since both groups run speculative_adaptive=False.
    #
    # TRAIN 2 MERGE NOTE, an OPEN item rather than a silent change: this is the
    # last read-THROUGH-the-launcher's-environment default left in this
    # function.  The #1235 R19 rule below turned every sibling into launcher
    # OUTPUT precisely because an inherited export won silently and the boot
    # then ran a value no flag and no log line named.  It is kept in the
    # dormant slice's own form here -- changing it would remove the operator's
    # only way to disable the coupling while its arm is still being graded --
    # and it is named here so the next pass makes it a flag rather than
    # discovering it.
    env["FLLIPER_MEMORY_SAVER_CUDA_GRAPH"] = env.get(
        "FLLIPER_MEMORY_SAVER_CUDA_GRAPH", "1"
    )
    # THE DOUBLY-STATED FACTS, FROM THE SAME TABLE THE ARGV IS BUILT FROM
    # (#1235). Not second bookkeeping: ServerArgs reads these keys off
    # os.environ before the matching flags publish themselves -- see
    # EarlyReadFact for the file:line ordering. Written for BOTH groups, which
    # preserves an existing asymmetry the table records rather than hides.
    # (The dormant slice restated FLLIPER_UNEVEN_DCP / _WEIGHTED /
    # FLLIPER_MAMBA_SSM_DTYPE here by hand; that is the pre-#1235 form of the
    # same three facts and this loop is the one that survives.)
    for fact in EARLY_READ_FACTS:
        env[fact.env_key] = fact.env_value
    if group:
        # Scheibe 6a / UNIFY S3: per-group env rows of the profile (pdflip/form.py
        # ModelProfile.group_env). Next Flash owns the uneven-DCP axis per group
        # (seam D): P (PP3, tp 1) keeps 1/1 (inert), D (Form A) gets 0/0 -- an
        # inherited 1 lands in RankRoleError (rank_role.py
        # resolve_dcp_under_host_kv). The 27B row has none.
        _row = pdflip_form.profile_row(profile)
        if _row is not None:
            env.update(_row.group_env.get(group, {}))
    # RG (operator 26.09.): the agent-load prefix switches (pdflip/form.py
    # PREFIX_SWITCHES) from the row of THIS boot's form -- an explicitly set
    # env already in `env` wins, an off row writes nothing, and --env-p /
    # --env-d below still override. A desk caller (no form) publishes nothing.
    # The front gets the same call on its env (the front spawn below).
    if boot_form is not None:
        pdflip_form.publish_prefix_switches(env, boot_form.profile)
    if flip_weights == "resident":
        from flliper.srt.managers.pdflip_memory_saver import WEIGHTS_RESIDENT_ENV

        env[WEIGHTS_RESIDENT_ENV] = "1"
    if group == "D":
        # 27B b1 death (27.09. 06:57:17Z): D's full graphs upload at capture,
        # not at their first replay under load (pdflip/graph_upload.py). A value
        # already in the env (container / arm) and --env-d below win.
        env.setdefault("FLLIPER_PDFLIP_GRAPH_UPLOAD_AT_CAPTURE", "1")
    if group_env_extra:
        # --env-p / --env-d (Scheibe 6a): per-group variables the two layouts
        # spell differently (e.g. FLLIPER_MOE_SCRATCH_SLOTS: one value per TP
        # rank -- P's stages run tp 1, D's Form A runs tp 3). Applied LAST so
        # an operator value is what the group runs, and printed by the caller.
        env.update(canonical_env({str(k): str(v) for k, v in group_env_extra.items()}))
    # #236: the names the operator stated (user_reserve_env_entries respects them)
    _stated_by_operator = set(group_env_extra or {})
    env["FLLIPER_BARLINK_BUILD_WINDOW_CAP_S"] = str(barlink_build_window_cap_s)
    if str(transport) == "nccl":
        # #1234 C6: half-configuring a transport the group does not run is
        # how a mode switch turns into a mystery. The flags go with
        # strip_barlink_flags(), the env keys go here.
        for key in BARLINK_ENV_KEYS:
            env.pop(key, None)
    # LAUNCHER OUTPUT, NOT env.get (#1235). All four used to read a default
    # THROUGH the launcher's own environment, so an inherited export won
    # silently and the boot ran a number no flag and no log line named -- the
    # R19 shape the ring family was already fixed out of. Now they are flags,
    # written unconditionally.
    env["FLLIPER_PP_CHAIN_RECV_STALL_S"] = str(pp_chain_recv_stall_s)
    env["FLLIPER_PP_OCCUPANT_HORIZON_S"] = str(pp_occupant_horizon_s)
    env["FLLIPER_HICACHE_FILE_BACKEND_STORAGE_DIR"] = store_dir
    env["FLLIPER_MATCH_REFUSAL_CENSUS_EVERY"] = str(match_refusal_census_every)
    # #1395: UNCONDITIONAL, unlike FLLIPER_PDFLIP_LANE_COVERAGE_TOKEN a few
    # lines up -- this value arms nothing by existing, it only names which
    # boot a rank belongs to (mem_ledger.activation_probe.boot_token() reads
    # it so two boots of the identical form never collide on one dump path).
    env["FLLIPER_PDFLIP_BOOT_TOKEN"] = pdflip_boot_token
    # #1378 xsn43 (TEIL 1): the evidence dir for the lanes' own stack dumps
    # (weight_exchange_bounce.dump_rank_stacks) -- every named refusal of
    # the W68/W29 family carries its diagnosis with it, per rank, named.
    # Published like the boot token: an identity, never a secret.
    env["FLLIPER_PDFLIP_RANKDUMP_DIR"] = EVIDENCE_DIR
    # #1378 Posten 4: the cushion sampler's CSV path, published to BOTH
    # groups so the front's flip-ratchet record carries a MEASURED
    # cushion_min instead of the null every FLIP record so far wrote
    # (weg2xsn29/zwerg/ring1 -- the env existed only as a reader,
    # host_ledger.ENV_HOSTSAMPLE_CSV, and nothing ever set it; the numbers
    # sat on disk per boot: xsn29 14.857 / zwerg 5.23 / ring1 4.276 /
    # xsn31 1.082 / xsn32 0.014 GiB, measured 2026-09-14 with the tree's
    # own reader). ONE PRODUCER of the path: hostsample_csv_path(). The
    # file may not exist at launch (the sampler writes during the boot);
    # an absent file reads back as a NAMED absence at the record, never
    # 0.0 -- host_ledger.cushion_min_and_reason.
    env[host_ledger.ENV_HOSTSAMPLE_CSV] = hostsample_csv_path(tag)
    for k in list(env):
        if k.startswith("FLLIPER_PHASE_FLIP"):
            del env[k]
    alloc = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in alloc:
        raise PdFlipLaunchRefused(
            f"PYTORCH_CUDA_ALLOC_CONF={alloc!r} -- torch_memory_saver refuses expandable_segments (entrypoint.py _sanity_checks)"
        )
    env.pop("FLLIPER_KILLPG_ON_SCHEDULER_EXCEPTION", None)
    if debug_hold:
        env["FLLIPER_DEBUG_HOLD"] = "1"
        env["FLLIPER_DEBUG_HOLD_S"] = env.get("FLLIPER_DEBUG_HOLD_S", "1800")
        env["FLLIPER_DEBUG_HOLD_PORT_BASE"] = env.get("FLLIPER_DEBUG_HOLD_PORT_BASE", "5000")
        env["FLLIPER_DEBUG_HOLD_DIR"] = env.get("FLLIPER_DEBUG_HOLD_DIR", f"{GPU_ARB}/debug_hold")
        env["FLLIPER_DEBUG_HOLD_TAG"] = tag
        os.makedirs(env["FLLIPER_DEBUG_HOLD_DIR"], exist_ok=True)
    # #1269/#1276: THE IDLE FAMILY, published to BOTH groups with provenance.
    #
    # FLLIPER_IDLE_BLOCKING_POLL -- boot weg2sb4 set neither this nor
    # --sleep-on-idle, so Scheduler.init_idle_sleeper built no sleeper on any
    # of the six ranks and both loops spun: group P 1,499.8 scheduler
    # rounds/s, group D 303.1/s, 34 idle minutes, empty queue, live front.
    # The serve tree now also arms on FLLIPER_PDFLIP_GROUP, so this is the
    # BELT to that braces -- published explicitly so the boot log names the
    # decision instead of leaving it to a discriminator set for other reasons.
    #
    # MALLOC_ARENA_MAX=4 -- chosen from this rig's own measurement, not a
    # folk value. glibc defaults to 8 x ncores arenas; CPU affinity on this
    # box is 0-31, so the default ceiling is 256, and the measured growth was
    # 64 MiB-aligned [anon] arenas being extended in place plus NEW arenas
    # appearing (+5.4 MiB/min on P PP0/PP1, +2.8 on each D rank, +19.0 MiB/min
    # over the six ranks against an independently observed +21.3). The
    # allocating threads on the steady-state path are the scheduler thread
    # plus the three the reset path names by hand -- "#N RESET JOIN
    # threads=['prefetch', 'backup', 'prefetch_io_aux']" -- so 4 gives each
    # one its own arena without the default's fragmentation surface. This is a
    # CEILING, not a target: a rank that needs fewer uses fewer.
    #
    # THE RANK THREAD COUNT WAS NOT CAPTURED before the host OOM took the base
    # down, so 4 is derived from the reset line's thread names, not from a
    # live count. PDFLIP-IDLE-CENSUS prints `threads=` for exactly this reason;
    # the next boot confirms or corrects the value.
    env.setdefault("FLLIPER_IDLE_BLOCKING_POLL", "1")
    env.setdefault("MALLOC_ARENA_MAX", "4")
    # #114: group P books its MEASURED prefill transient in the sizer's own
    # ledger (one entry per P rank, priced from the chunk P boots with), so
    # the KV pool is sized around it instead of over it -- fnFL2x121/x122
    # sized 262144 tokens into the transient's room and died in the first
    # chunk's forward. The operator's own value wins (setdefault); every other
    # group gets the family POPPED, so D never inherits P's post.
    #
    # PDFLIP-FORM (24.09.): only for the checkpoint the slopes were MEASURED on
    # (p_prefill_transient_for). xsn417 published the Next-Flash slopes to a
    # Qwen3.8-27B group P and its PP0 pool fell to 246155 tokens (< 262144;
    # xsn411 on the 27B line: 277606) for a transient nobody measured there.
    if group == "P":
        if p_prefill_transient_for(boot_form)[0]:
            env.setdefault(
                P_PREFILL_TRANSIENT_ENV,
                ",".join(
                    "%.0f" % v for v in p_prefill_transient_vector_mib(
                        P_CHUNKED_PREFILL_TOKENS, getattr(boot_form, "profile", None) or None)
                ),
            )
    else:
        env.pop(P_PREFILL_TRANSIENT_ENV, None)
    env.update(spec_form_env(group))  # --spec-form: D's window pool under DFLASH
    # PDFLIP-FORM: launcher OUTPUT, the R19 discipline of FLLIPER_PDFLIP_GROUP --
    # written from the resolved form, POPPED when this call names none, so a
    # value inherited from a shell (or from an earlier main() in the same
    # process) can never describe a boot it did not resolve.
    if boot_form is not None:
        env[pdflip_form.FORM_ENV] = boot_form.env_value()
    else:
        env.pop(pdflip_form.FORM_ENV, None)
    # HICACHE-DRAFT-TIER: the RESOLVED value, the same on both groups; an
    # operator's `auto` never reaches a rank (it cannot resolve it there).
    env.pop(HICACHE_DRAFT_TIER_ENV, None)
    env.update(hicache_draft_tier_env())
    # 27B FP8 (--fp8-uniform-marlin, resolved by _env_knobs against the
    # checkpoint): the same two variables on BOTH groups, set only then --
    # every other boot's environment stays exactly what it was.
    if fp8_uniform_marlin:
        env.update(FP8_UNIFORM_MARLIN_ENV)
    # #233 review MUSS 1/2: the frozen corridor pointer state and the recipes are
    # launcher OUTPUT armed per group in launch_group
    # (corridor_guard.arm_group_recipe_env), not here: this env is printed as
    # PDFLIP-GROUP-ENV into the dry-run plan dump, which must not carry box state
    # (the pointer file's resolution, a freeze timestamp).
    for _k in (corridor_guard.POINTER_FROZEN_ENV, corridor_guard.RECIPE_ENV,
               corridor_guard.RECIPE_SELF_ENV):
        env.pop(_k, None)
    # #236 (Auftrag, fund of the #233 second review): THE USER RESERVE REACHES
    # THE RANKS. ``corridor_guard.USER_RESERVE_ENV`` had exactly one setter, the
    # front's ``fenv`` (main(), below); every rank-side reader of the floor
    # (``vram_dial``, ``weight_updater._pdflip_corridor_floor_bytes``,
    # ``corridor_floor_for_current_device`` -> kv_vmm_backing / seam reserve /
    # extend_trim / phase_flip_runtime) reads the reserve ONLY from this env, so
    # with ``--user-reserve-mib`` the launcher priced floor = transient + reserve
    # while every rank priced floor = transient + 0 (the OOM direction).
    # Group env, so P and D, host and worker of every form inherit it through
    # ``Popen(env=spec.env)`` -> ``mp.Process`` exactly as FLLIPER_PDFLIP_GROUP
    # does; the mapping is keyed by card UUID and each rank picks its own card
    # by ``registry.nvml.current_device_uuid()`` -- never by index.
    # Launcher OUTPUT (R19 discipline): published only when the operator stated
    # a reserve (any card > 0), POPPED otherwise, so a value inherited from a
    # shell never describes a boot it did not resolve and a boot without
    # ``--user-reserve-mib`` keeps today's env byte for byte (a rank booted with
    # only a ``--rank-user-reserve-mib`` argv keeps its scalar fallback).
    # NO NEW MARGIN: the value is the operator's, serialised by the same
    # function the front uses.
    user_reserve_env_entries(env, _stated_by_operator, user_reserve_by_card, reserve_cards, reserve_booked_rest_mib)
    return env

def d_extend_vote_fixed_record(profile: Optional[str] = None, n: Optional[int] = None,
                               cards: Optional[Sequence[Card]] = None,
                               model: Optional[str] = None) -> Tuple[Optional[List[float]], str]:
    """VOTE-LINE-SINGLE: the profile's ``D_EXTEND_VOTE_FIXED_MIB`` per card (one value covers every card) and its
    boots, or ``(None, why)`` -- no record (other profiles, Next Flash), the switch off
    (``FLLIPER_PDFLIP_VOTE_LINE_SINGLE=0``), a vector that does not fit ``n`` cards, or (review M1) a FORM the record
    was not measured on: the checkpoint (``model``, else the published form's) and the D cards' inventory must be a
    form of the record's boots (``budget_rest.VOTE_FIXED_FORMS``); unknown checkpoint or no cards = not this form
    (fail closed: the floor line). The ranks' env (:func:`_d_extend_vote_fixed_env`) and the budget's delta read this
    ONE reader."""
    if not _budget_rest.vote_line_single_on():
        return None, f"{_budget_rest.VOTE_LINE_SINGLE_ENV}=0"
    try:
        vals = list(_pconst(_budget_rest.VOTE_FIXED_RECORD, profile))
    except (KeyError, TypeError):
        return None, f"no {_budget_rest.VOTE_FIXED_RECORD} record"
    if not vals or any(v is None for v in vals):
        return None, f"{_budget_rest.VOTE_FIXED_RECORD} {vals!r} has an unmeasured card"
    prof = profile if profile is not None else pdflip_form.DEFAULT_PROFILE
    row = pdflip_form.profile_row(prof)
    rec = row.constants.get(_budget_rest.VOTE_FIXED_RECORD) if row is not None else None
    boots = tuple(getattr(rec, "boots", ()) or ()) if rec is not None else ()
    if model is None:
        form = pdflip_form.current_form()
        model = form.model if form is not None else ""
    try:
        inventory = card_identity.inventory_signature(list(cards)) if cards else ()
    except Exception:  # noqa: BLE001 -- an unreadable inventory is not this form (fail closed)
        inventory = ()
    why = _budget_rest.vote_fixed_form_mismatch(boots, pdflip_form.model_key(model or ""), inventory)
    # VOTE-FIXED-LIVE (24.10., pdflip/vote_fixed_live.py): the form's PERSISTED run-time readings -- they raise a
    # static record (rule 2: every reading counts upward) and, once the gate holds (rule 3: MIN_BOOTS boots of
    # MIN_PASSES_PER_BOOT passes), stand in for a record the form never had. Same reader for ranks and budget.
    _digest, live, live_why = d_extend_vote_fixed_live_verdict(profile, cards, model)
    if why is not None:
        if live is None or not live.gate_open:
            return None, why + ("" if live is None else f"; {_vfl.MARKER} form {_digest}: {live.describe()}")
        k = int(n) if n is not None else max(1, len(inventory))
        return ([float(live.fixed_mib)] * k,  # type: ignore[arg-type]
                f"{_vfl.MARKER} form {_digest}: {live.describe()} (no static record of this form)")
    if n is not None and len(vals) not in (1, int(n)):
        return None, f"{_budget_rest.VOTE_FIXED_RECORD} {vals!r} does not fit {n} cards"
    out = [float(v) for v in vals]
    if n is not None and len(out) == 1:
        out = out * int(n)
    src = _pconst_boots(_budget_rest.VOTE_FIXED_RECORD, profile)
    if live is not None and live.fixed_mib is not None and any(float(live.fixed_mib) > v for v in out):
        out = [max(v, float(live.fixed_mib)) for v in out]
        src = f"{src}; raised by {_vfl.MARKER} form {_digest}: {live.describe()}"
    return out, src

def _d_extend_vote_fixed_live_env(ns, log, label: str, cards: Optional[Sequence[Card]], record_src: str) -> None:
    """VOTE-FIXED-LIVE: wherever the width vote is armed on a profile with the record, hand the ranks the meter's
    store target -- form digest, D inventory, boot tag, store root -- so every boot of the form measures and persists
    (cold start included: the floor line holds until the gate opens, the meter still reads). Unreadable form / switch
    off: nothing written, named. A value named in --env-d wins."""
    try:
        _pconst(_budget_rest.VOTE_FIXED_RECORD, getattr(ns, "profile", None))
    except (KeyError, TypeError):
        return   # a profile without the single line (Next Flash: VOTE-MEASURED-LINE on its own branch)
    dig, mat, why = d_extend_vote_fixed_live_material(getattr(ns, "profile", None), cards, getattr(ns, "model", None))
    if dig is None:
        log(f"{D_RANK_SOLVE_MARKER} {label} {_vfl.MARKER} store off: {why} -- the ranks' meter raises the line "
            f"within the boot only, nothing persists")
        return
    # review M1: the FULL boot name (main: ns.boot_name = boot_weg2_<tag>_<rev>_<date>_<time>), the tag only where
    # main did not run (desk)
    boot = str(getattr(ns, "boot_name", "") or getattr(ns, "tag", "") or "")
    root = vote_fixed_live_root()
    env = {_vfl.ENV_FORM: dig, _vfl.ENV_CARDS: ",".join(card_identity.inventory_signature(list(cards or ()))),
           _vfl.ENV_BOOT: boot, _vfl.ENV_DIR: root}
    if not getattr(ns, "dry_run", False):
        try:   # review M3: the yardstick --import checks old logs against
            # the totals a D rank's torch sees (NVML total - driver reserved), what PDFLIP-VRAM-PEAK logs as
            # card_total_mib (5090 32607 - 499 = 32108, 3080 20480 - 425 = 20055)
            _vfl.write_form(root, dig, mat or {}, [int(getattr(c, "total_mib", 0) or 0)
                                                   - int(getattr(c, "reserved_mib", 0) or 0) for c in (cards or ())])
        except Exception as exc:  # noqa: BLE001 -- no sidecar: --import of old logs stays unverified
            log(f"{D_RANK_SOLVE_MARKER} {label} {_vfl.MARKER} {_vfl.FORM_FILE} not written ({type(exc).__name__})")
    _d, _v, _w = d_extend_vote_fixed_live_verdict(getattr(ns, "profile", None), cards, getattr(ns, "model", None))
    if _v is not None:
        for q in _v.quarantined:   # review M2: loud, never a seed
            log(f"{D_RANK_SOLVE_MARKER} {label} WARN {_vfl.MARKER} QUARANTINED reading (no seed): {q}")
        for k in _v.skipped:
            log(f"{D_RANK_SOLVE_MARKER} {label} {_vfl.MARKER} store doc left out: {k}")
    _have = parse_group_env(getattr(ns, "env_d", "") or "")
    for k, v in env.items():
        if k not in _have and v:
            ns.env_d = set_group_env(getattr(ns, "env_d", "") or "", k, v)
    log(f"{D_RANK_SOLVE_MARKER} {label} {_vfl.MARKER} {_vfl.ENV_FORM}={dig} store={env[_vfl.ENV_DIR]} "
        f"({why}); record: {record_src}")

def corridor_recipe_probe(ns, cards, p_bs: int, d_bs: int, hicache_disabled: bool) -> Dict[str, Dict[str, object]]:
    """#233 review MUSS 2: each group's RECIPE, read off the group argv this
    launcher builds -- BEFORE the first floor of the boot is read.

    The budget needs the floor and the real argv needs the budget, so the real
    argv does not exist yet where the floor is read. The probe builds each
    group's argv with the SAME builders (``argv_p`` / ``argv_d``) and SENTINEL
    budgets of the boot's card count -- the pattern ``_max_running_requests``
    and ``_p_page_size`` already read flags by ("the number is READ OFF THE
    ARGV this launcher builds, never restated") -- with no cut (H68c) and the
    operator's ``--extra-*`` reduced to the recipe flags
    (``corridor_guard.recipe_relevant_tokens``: the items are the same, and the
    W100 budget refusal must not judge an operator budget against a sentinel).
    ``launch_group`` compares this with the REAL argv and names any drift
    (``#233 CORRIDOR-RECIPE-DRIFT``). No ``profile_from_server_args`` here: the
    #1257c refusal to restate it stands.

    A group whose probe raises gets no recipe -> no reader finds an own record
    -> the law, named; never a guessed recipe.
    """
    n = max(1, len(cards))
    sentinel = [1] * n
    extra_p = corridor_guard.recipe_relevant_tokens(shlex.split(getattr(ns, "extra_p", "") or ""))
    extra_d_full = shlex.split(getattr(ns, "extra_d", "") or "")
    extra_d = corridor_guard.recipe_relevant_tokens(extra_d_full)
    vision = getattr(ns, "pdflip_vision", VISION_OFF)
    profile = getattr(ns, "profile", PROFILE_QWEN27B)
    out: Dict[str, Dict[str, object]] = {}
    try:
        out["P"] = corridor_guard.recipe_from_argv(argv_p(
            "py", ns.model, sentinel, 1, 1, RING_FORM_SENTINEL_STORE_CFG, extra_p,
            p_bs=int(p_bs), stage_ratio="", attn_stage_ratio="",
            draft_kv_on_p=(getattr(ns, "draft_kv_on_p", "on") == "on" and not hicache_disabled),
            spec_flags=p_draft_kv_flags(extra_d_full), vision=vision, profile=profile,
            hicache_disabled=hicache_disabled))
    except Exception as exc:  # noqa: BLE001 - named, the law follows
        print(f"{corridor_guard.RECIPE_DRIFT_TAG} group=P: recipe probe failed ({exc!r}); "
              "no own record can be found this boot (the law)", file=sys.stderr, flush=True)
    try:
        out["D"] = corridor_guard.recipe_from_argv(argv_d(
            "py", ns.model, sentinel, 1, 1, RING_FORM_SENTINEL_STORE_CFG, extra_d,
            d_bs=int(d_bs), disable_cuda_graph=bool(getattr(ns, "d_disable_cuda_graph", False)),
            hicache_disabled=hicache_disabled, profile=profile, vision=vision))
    except Exception as exc:  # noqa: BLE001 - named, the law follows
        print(f"{corridor_guard.RECIPE_DRIFT_TAG} group=D: recipe probe failed ({exc!r}); "
              "no own record can be found this boot (the law)", file=sys.stderr, flush=True)
    return out

def _max_running_requests(model: str, group: str = "D", bs: int = DEFAULT_D_BS) -> int:
    """``--max-running-requests`` as this launcher actually passes it, per GROUP.

    PRE-EXISTING BOOT KILLER, found by this slice's own --dry-run and fixed
    here rather than left for the next window (it is two frames below
    ``solve_p_cut``, so no boot from the merge-train tip reached group P's
    launch). It read the flag out of ``common_flags``, and the flag LEFT
    ``common_flags`` when C1/R-12 made P's and D's bs independent -- so this
    reader was wrong twice over, and both raise rather than answer:

      * ``common_flags(model, 1, 1, 1.0)`` is missing ``max_kv_per_request``
        (TypeError), because a fifth required parameter was added ahead of it;
      * even given the arguments, ``flags.index("--max-running-requests")``
        raises ValueError, because common_flags' own docstring says the flag is
        "NOT here any more ... emitted per group from --p-bs / --d-bs instead".

    THE CLASS, so the sweep is on the record: a flag MOVED and one consumer was
    not moved with it. The fix keeps the property the original was reaching for
    -- the number is READ OFF THE ARGV this launcher builds, never restated --
    by reading it off the GROUP's argv, which is where the flag now lives. The
    two callers ask about different groups and now say which: the P pool
    model's mamba slots are P's ``--p-bs``, and D's ping-pong price is D's
    ``--d-bs``. Sharing one value was the very coupling C1/R-12 removed.

    H68c: the probe argv states NO cut (``stage_ratio=""``, the omit-both
    form). With ``None`` argv_p hands over the incumbent ratio and first checks
    the model's PP calibration (#1362 W99) -- a check about a CUT this reader
    never looks at. It refused the first NVFP4 dry run (48 layers, no record
    yet) although that arm pins its cut (--pp-stage-ratio 29,11,8), whose own
    argv never asks for the incumbent.
    """
    if str(group) == "P":
        flags = argv_p("py", model, [1, 1, 1], 1, 1, RING_FORM_SENTINEL_STORE_CFG, [], p_bs=int(bs),
                       stage_ratio="", attn_stage_ratio="")
    else:
        flags = argv_d("py", model, [1, 1, 1], 1, 1, RING_FORM_SENTINEL_STORE_CFG, [], d_bs=int(bs))
    return int(flags[flags.index("--max-running-requests") + 1])

def d_overhead_calibration(
    model: str, line_id, evidence_dir: str = EVIDENCE_DIR,
) -> Tuple[Optional[List[float]], str]:
    """Per-rank MiB of group D's non-KV, non-weight, non-mamba posts, MEASURED
    on the newest D log of THIS line (``line_id``: same checkpoint, a commit
    of this line's own history) -- ``(None, why)`` when no such boot exists.

    Against the planner's own weight and mamba model at THAT boot's vector
    and budgets: ``O_r = budget_r - KV bytes_r - weights_r - mamba_r -
    window reserve``, KV bytes the rank's own final 'KV pool sizing:
    available_bytes' line. The planner then reproduces that boot's per-rank
    capacity exactly at its own vector and moves weights / mamba with any
    other vector. The flat 2304 MiB it replaces priced xsn420's D at 330939
    tokens against 634087 realised. The measured cell must equal the
    planner's cell, or the calibration does not transfer and is refused.
    """
    from flliper.srt.distributed.utils import partition_units
    from flliper.srt.uneven_perf import PerfCostModel

    if line_id is None:
        return None, "no line identity (desk caller)"
    try:
        names = [n for n in os.listdir(evidence_dir) if n.endswith(".D.log")]
    except OSError as exc:
        return None, f"evidence dir unreadable: {exc}"
    paths = sorted((os.path.join(evidence_dir, n) for n in names),
                   key=os.path.getmtime, reverse=True)
    skipped = 0
    for d_log in paths:
        if not line_id.accepts_log(d_log):
            continue
        front = d_log[: -len(".D.log")] + ".front.log"
        argv = _d_argv_of_front_log(front)
        if not argv:
            skipped += 1
            continue
        kv: Dict[int, Tuple[int, int]] = {}
        try:
            with open(d_log, errors="replace") as f:
                for line in f:
                    if "KV pool sizing:" in line:
                        m = _KV_POOL_SIZING_RE.search(line)
                        if m:
                            kv[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
        except OSError:
            skipped += 1
            continue
        fm: Dict[str, str] = {}
        for i, tok in enumerate(argv[:-1]):
            if tok.startswith("--"):
                fm[tok] = argv[i + 1]
        try:
            budgets = [int(x) for x in fm["--rank-gpu-memory-mib"].split(",")]
            mrr = int(fm.get("--max-running-requests", "1"))
        except (KeyError, ValueError):
            skipped += 1
            continue
        n = len(budgets)
        if sorted(kv) != list(range(n)):
            skipped += 1
            continue
        # 28.09. (VRAM loop, P1b): an expert-offload D (--rank-moe-resident-fraction,
        # the NF Form A) is NOT priced by this row planner -- PerfCostModel books
        # every expert resident (TP0 ~164 GiB at ratio 1,0,0) and the KV-less
        # Form A workers report cell 0, so no overhead can come out of it. Its
        # budget is the MEASURED-D budget line + FRACTION-SOLVE D (H8), which
        # already read this model's own records. Named, not a cell mismatch.
        if "--rank-moe-resident-fraction" in fm:
            return None, (
                f"{os.path.basename(d_log)}: expert-offload D (--rank-moe-resident-fraction "
                f"{fm['--rank-moe-resident-fraction']}, --rank-tp-ratio "
                f"{fm.get('--rank-tp-ratio', 'auto')}) -- this row planner prices every expert "
                f"resident, so it does not apply; D's budget comes from the MEASURED-D budget "
                f"line and FRACTION-SOLVE D (this model's records D_FIXED_MIB/D_ACTIVATION_MIB)")
        ratio = fm.get("--rank-tp-ratio", "auto")
        if ratio == "auto":
            weights = list(_gcd_reduce(budgets))
        else:
            try:
                weights = [int(x) for x in ratio.split(",")]
            except ValueError:
                skipped += 1
                continue
            if len(weights) != n:
                skipped += 1
                continue
        cells = {c for _, c in kv.values()}
        try:
            pcm = PerfCostModel(d_plan_inputs(model, n, mrr), weights, budgets)
            mlp = list(partition_units(int(pcm.mlp_units), weights))
            # the SAME joint candidate the row prices (predict_capacity(mlp,
            # attn_units): attention/GDN/mamba follow the attn units), so the
            # row reproduces this boot per rank, not only in the sum
            attn_units = d_row_attn_units(pcm, weights)
            w = pcm.per_rank_weight_bytes(mlp, attn_units)
            mamba = pcm.mamba_pool_bytes_for(attn_units)
        except Exception as exc:  # noqa: BLE001 -- a calibration never kills a boot
            return None, f"planner not buildable for {os.path.basename(d_log)}: {exc}"
        if cells != {int(pcm.kv_cell_bytes)}:
            return None, (f"{os.path.basename(d_log)}: measured cell {sorted(cells)} != planner cell "
                          f"{int(pcm.kv_cell_bytes)} -- the calibration would not transfer")
        mib = float(1 << 20)
        ovh = [
            (budgets[r] * mib - kv[r][0] - w[r] - mamba[r] - pcm.window_pool_reserve_bytes) / mib
            for r in range(n)
        ]
        if any(o < 0 or o > 8192 for o in ovh):
            return None, (f"{os.path.basename(d_log)}: implausible measured overhead "
                          f"{[round(o) for o in ovh]} MiB -- refused, flat default stays")
        return ovh, (
            f"MEASURED on {os.path.basename(d_log)} (this line, {os.path.basename(str(model).rstrip('/'))}): budgets {budgets}, "
            f"--rank-tp-ratio {ratio}, mrr {mrr}, KV {[kv[r][0] for r in range(n)]} B at cell "
            f"{sorted(cells)[0]} -> overhead {[round(o, 1) for o in ovh]} MiB/rank "
            f"(flat default was {1280 + 1024}); skipped {skipped} newer log(s) without the instruments"
        )
    return None, f"no D log of this line carries 'KV pool sizing' and a group D argv (skipped {skipped})"
