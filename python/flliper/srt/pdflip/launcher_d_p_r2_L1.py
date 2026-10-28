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
import time

from flliper.srt.managers import corridor_guard
from flliper.srt.pdflip import admin_key as admin_key_mod
from flliper.srt.pdflip import budget_rest as _budget_rest
from flliper.srt.pdflip import card_identity
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import rank_state as rank_state_mod
from flliper.srt.pdflip import shared_cache_release as shared_cache_release_mod
from flliper.srt.pdflip import state_file as state_file_mod
from flliper.srt.pdflip import topology as _topo
from flliper.srt.pdflip import vote_fixed_live as _vfl
from flliper.srt.pdflip import xchg_residency
from flliper.srt.pdflip.launcher_d_leaf import VISION_OFF
from flliper.srt.pdflip.launcher_d_leaf import _SPEC_FORM
from flliper.srt.pdflip.launcher_d_leaf import d_expect_from_p_records
from flliper.srt.pdflip.launcher_d_leaf import dflash_placement
from flliper.srt.pdflip.launcher_d_leaf import early_read_flags
from flliper.srt.pdflip.launcher_d_mid import CONTEXT_LENGTH_TOKENS
from flliper.srt.pdflip.launcher_d_mid import checkpoint_quant_method
from flliper.srt.pdflip.launcher_d_p_r2_L0 import DEVTOOLS_DIR
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_ACTIVATION_RECORD
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_CHUNKED_PREFILL_TOKENS
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PREFILL_TRANSIENT_SUPPORT
from flliper.srt.pdflip.launcher_d_p_r2_L0 import SPEC_NUM_DRAFT_TOKENS
from flliper.srt.pdflip.launcher_d_p_r2_L0 import _l3_attach_from_index
from flliper.srt.pdflip.launcher_d_p_r2_L0 import _p_chunk_nf_limits
from flliper.srt.pdflip.launcher_d_p_r2_L0 import boot_state_write
from flliper.srt.pdflip.launcher_d_p_r2_L0 import calib_identity_of
from flliper.srt.pdflip.launcher_d_p_r2_L0 import check_draft_resident
from flliper.srt.pdflip.launcher_d_p_r2_L0 import core_cwd_for
from flliper.srt.pdflip.launcher_d_p_r2_L0 import d_admit_need_record
from flliper.srt.pdflip.launcher_d_p_r2_L0 import d_extend_growth_record
from flliper.srt.pdflip.launcher_d_p_r2_L0 import d_verify_window
from flliper.srt.pdflip.launcher_d_p_r2_L0 import lane_coverage_dump_dir
from flliper.srt.pdflip.launcher_d_p_r2_L0 import load_xchg_census_for_cards
from flliper.srt.pdflip.launcher_d_p_r2_L0 import log_boot_state
from flliper.srt.pdflip.launcher_d_p_r2_L0 import measured_record_path
from flliper.srt.pdflip.launcher_d_p_r2_L0 import memts_csv_path
from flliper.srt.pdflip.launcher_d_p_r2_L0 import owned_miss_record_root
from flliper.srt.pdflip.launcher_d_p_r2_L0 import p_activation_record
from flliper.srt.pdflip.launcher_d_p_r2_L0 import p_chunked_prefill_tokens
from flliper.srt.pdflip.launcher_log_plan import Log
from flliper.srt.pdflip.launcher_p_l3_leaf import CHUNKED_PREFILL_TOKENS
from flliper.srt.pdflip.launcher_p_l3_leaf import L3_REVOKED_SUFFIX
from flliper.srt.pdflip.launcher_p_l3_leaf import UNIFORM_MARLIN_QUANT_METHODS
from flliper.srt.pdflip.launcher_p_l3_leaf import _P_CHUNK
from flliper.srt.pdflip.launcher_p_l3_leaf import _P_PREFILL_GRAPH
from flliper.srt.pdflip.launcher_p_l3_leaf import _l3_move_revoked
from flliper.srt.pdflip.launcher_p_l3_leaf import _live_stage_support
from flliper.srt.pdflip.launcher_p_l3_leaf import _now
from flliper.srt.pdflip.launcher_p_l3_leaf import _owned_miss_min_paired
from flliper.srt.pdflip.launcher_p_l3_leaf import _p_chunk_nf_args
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst_boots
from flliper.srt.pdflip.launcher_p_l3_leaf import group_form_block
from flliper.srt.pdflip.launcher_p_l3_leaf import l3_registered_suffixes
from flliper.srt.pdflip.launcher_p_l3_leaf import p_prefill_graph_bucket
from flliper.srt.pdflip.launcher_p_l3_leaf import parse_group_env
from flliper.srt.pdflip.launcher_p_l3_leaf import pdflip_boot_token
from flliper.srt.pdflip.launcher_p_mid import KV_CACHE_DTYPE
from flliper.srt.pdflip.launcher_p_mid import _model_config
from flliper.srt.pdflip.launcher_p_mid import l3_revoked_windows
from flliper.srt.pdflip.launcher_p_mid import lane_coverage_boot_token
from flliper.srt.pdflip.launcher_p_mid import p_prefill_graph_buckets
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import SPEC_ALGORITHM
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import SPEC_EAGLE_TOPK
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import SPEC_NUM_STEPS
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import _vfl_effective_draft
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import rank_state_dir_for
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import spec_flags
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import spec_form_env
from flliper.srt.pdflip.launcher_refuse_proc_leaf import PROFILE_QWEN27B
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _profile_early_read
from flliper.srt.pdflip.launcher_refuse_proc_leaf import served_model_name
from flliper.srt.pdflip.launcher_refuse_proc_leaf import set_group_env
from flliper.srt.pdflip.launcher_refuse_proc_leaf import spec_form_is_dflash
from flliper.srt.pdflip.launcher_resolve_stop import p_prefill_graph_flags
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import GroupSpec
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_vision_spec_mid import _vision_knob
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_PLACE_AUTO
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_SOURCE_DISK
from flliper.srt.pdflip.launcher_xchg_vision_leaf import tokenizer_path_flags
from flliper.srt.pdflip.launcher_xchg_vision_leaf import vision_model_flags
from flliper.srt.pdflip.launcher_xchg_vision_leaf import vision_on_d_of
from flliper.srt.pdflip.launcher_xchg_vision_leaf import xchg_region_pairs
from flliper.srt.planner import p_card_chunk as _p_card
import json
import shlex
import subprocess

MEMTS = f"{DEVTOOLS_DIR}/mem_timeseries.sh"
P_CORRIDOR_TOP_MIB = 1229.0
#: Per-rank arming floor for the pool model, MiB. The rig's VRAM corridor is
#: 819-1229 MiB NVML-free per card under load and the desk pre-flight arming
#: floor is <= 1229; the ceiling is taken because a floor that under-charges
#: inflates the pool, which is the unsafe direction for a capacity FLOOR.
#: pp_cut.PhasePoolModel names the gap this stands in for: the per-layout
#: solved floor exists only for a layout that has booted.
ARMING_FLOOR_MIB = 1229.0
P_ACTIVATION_RECORD_CHUNK = 16384
#: The /dev/shm staging region's layout terms, spec section 6/S3: 2 slots per
#: DIRECTED cross-card pair (double buffering), 32 MiB per slot (evidence E2
#: measured 32/64/128 MiB indistinguishable below 0.6 %, so the smallest is
#: taken and the carve-out halves for free), 1 MiB of header + gate rows + the
#: n x n matrix + the pointer directory. Inputs to
#: ``xchg_residency.region_mib_from_layout``, never a literal 385.
#: TODO(S7->S3): S3 owns the region; when weight_exchange creates it, these move
#: there and the launcher reads the size back off the created file.
XCHG_REGION_SLOTS_PER_PAIR = 2
XCHG_REGION_SLOT_MIB = 32
XCHG_REGION_HEADER_MIB = 1
#: #1273 S7 round-2 review F2: does the ``exchange`` arm change any RANK
#: behaviour yet?  No -- S7 arms the W71 gate and nothing else, so a boot
#: launched with ``--pdflip-weight-source exchange`` still runs the ring path
#: byte for byte.  The arming line SAYS so rather than leaving a later reader
#: of the log to infer it from a non-zero ``ring_H_mib``.
#: TODO(S7->S6): S6 flips this to True in the same commit that propagates the
#: flag into the two groups' argv, the request structs and the saver's region
#: flag.  One name, one place.
XCHG_RANK_BEHAVIOUR_WIRED = False
XCHG_UNWIRED_REASON = "S7-gate-only-no-rank-behaviour-until-S6"
#: #1235 THE ARGV LITERALS THAT WERE NOBODY'S.
#:
#: RANDOM_SEED is arbitrary and FIXED, and that is its whole provenance: it is
#: not measured, not solved, and not to be presented as either. What it buys is
#: that two boots of the same tip sample identically, so a decode difference
#: between them is a difference in the code. Stated as a flag so a boot that
#: WANTS a different sample says so.
RANDOM_SEED = 785500001
#: BAR1 cap cycles: the ceiling on barlink's build-time cycle budget. Large by
#: construction (it is a ceiling, not a target) -- the binding gate on this rig
#: is the aperture, priced at --barlink-bar1-window-mib, not this number.
BARLINK_BAR1_CAP_CYCLES = 300000000000
#: How often the collective census prints. A STRIDE, not a measurement: bigger
#: is quieter, smaller costs log volume, and nothing about the boot depends on
#: the value except how much of it a reader can see.
COLLECTIVE_CENSUS_INTERVAL = 50
#: HW-P1a 1003: the card count of the REFERENCE rig (the proven release
#: topology P = PP3 / D = TP3). NO LONGER A GATE: the launch's count is
#: ``len(order_cards(resolve_cards()))`` and :func:`topology_check_line`
#: decides it (``topology.plan_topology``, named blockers). Kept as the
#: default of the helpers that have no card list (``common_flags`` outside
#: argv_p/argv_d, desk tests).
PDFLIP_CARD_COUNT = _topo.PROVEN_CARD_COUNTS[0]
#: #66: wieviel freie Karte eine unerklaerte W11b-Differenz aufwiegen muss,
#: damit das Gate MELDET statt zu verweigern. Vier auf 1: gemessen trug
#: fnFL2v91 2,24 GiB frei gegen 533,7 MiB unerklaert, also das 4,3-fache.
#: Der Faktor ist bewusst grob -- er trennt "die Karte hat reichlich Luft"
#: von "es wird eng", und nicht zwei Zahlen, die sich um Prozente
#: unterscheiden. Faellt der Rand darunter, verweigert das Gate wieder.
W11B_FREE_OVER_UNACCOUNTED = 4.0
#: VOTE-FIXED-LIVE (review S3): group D's effective --kv-cache-dtype as main resolved it from --extra-d ("" = the
#: profile row's); set once in main, read by the form
_VFL_ARGV: Dict[str, str] = {}
#: PR (30.09.): group D's paired miss record is ON by default -- the records
#: root of the line under the evidence dir (``<root>/<model_id>/owned_miss``,
#: layers.moe.pool_miss_cost.record_dir_for). --env-d
#: FLLIPER_PDFLIP_OWNED_MISS_RECORD=<dir> names another root; an empty value
#: turns it off.
OWNED_MISS_RECORD_ENV = "FLLIPER_PDFLIP_OWNED_MISS_RECORD"

def p_dormant_records(profile: Optional[str],
                      accept: Optional[Callable[[dict], bool]]
                      ) -> Optional[List[Dict[str, object]]]:
    """The group-P dormant-image records the D expectation may price from --
    READ ONLY where they are used. ``None`` without reading on every profile
    whose expectation stays the legacy term (``d_expect_from_p_records``
    False: a broken or foreign sidecar must not touch a boot that never uses
    the value -- 27B review, 'Riegel hinter dem, was er sichert') and without
    a calibration identity (no identity = no foreign model's record). A
    malformed sidecar reads as ``[]`` (UNMEASURED), never as an exception.
    Since 29.09. the 27B reads them too (its row switched the expectation)."""
    if not d_expect_from_p_records(profile) or accept is None:
        return None
    return host_ledger.read_measured_records(measured_record_path(), "P", accept=accept)

def p_draft_kv_flags(extra_d: Sequence[str]) -> Tuple[str, ...]:
    """P's four speculative flags, READ OFF GROUP D's OWN ARGV.

    W10 grades P's and D's drafter lines against each other, so a constant
    here is only correct while every arm runs the constant's depth.  fnFL2 v38
    is the boot that broke it: group D runs the Next-Flash MTP at
    ``--speculative-num-steps 3 --speculative-num-draft-tokens 4`` out of
    ``--extra-d`` while the constants say 2/3.

    NOTE (2026-09-21) that the CORRECTNESS argument this originally carried --
    "the four flags hash into the drafter identity, so a mismatch makes D ask
    the carrier under an identity P never wrote" -- no longer holds, and it
    was the wrong argument to begin with: the chain geometry left
    ``drafter_identity_hash`` because it decides how many tokens a round
    PROPOSES, never what a stored row means, and because
    ``--speculative-adaptive`` changes it per round.  Taking D's depth is now
    a matter of keeping ONE number in the boot record instead of two; P is
    free to run a shorter chain, and the producer never proposes anyway.

    ``--speculative-draft-kv-only`` is P's alone and is never taken from D.

    UNIFY S7 (27B RC9 form): under DFLASH the producer family is the external
    draft's (:func:`spec_flags` ``producer=True``: DFLASH, the draft path, the
    block, ``--speculative-draft-kv-only``), byte-identical to the 27B line's
    argv_p -- the NEXTN constants and D's MTP depth do not apply to it. Before
    this, a 27B boot with a producer on P (profile p_draft cold) would have
    handed P ``--speculative-algorithm NEXTN``.
    """
    if spec_form_is_dflash():
        return tuple(spec_flags(producer=True))
    want = {
        "--speculative-algorithm": SPEC_ALGORITHM,
        "--speculative-num-steps": str(SPEC_NUM_STEPS),
        "--speculative-eagle-topk": str(SPEC_EAGLE_TOPK),
        "--speculative-num-draft-tokens": str(SPEC_NUM_DRAFT_TOKENS),
    }
    for i, tok in enumerate(extra_d):
        if tok in want and i + 1 < len(extra_d):
            want[tok] = extra_d[i + 1]
    out: List[str] = []
    for flag, value in want.items():
        out += [flag, value]
    return tuple(out) + ("--speculative-draft-kv-only",)

def p_prefill_graph_line() -> str:
    """The one launcher line naming group P's prefill-graph form."""
    bucket = p_prefill_graph_bucket()
    if not bucket:
        return (
            "PDFLIP P-PREFILL-GRAPH: off -- group P prefills eager at "
            f"--chunked-prefill-size {CHUNKED_PREFILL_TOKENS} (default form)"
        )
    if not p_prefill_graph_buckets():
        return (
            f"PDFLIP P-PREFILL-GRAPH: bucket={bucket} but NOTHING captured "
            f"(--p-prefill-graph-policy {_P_PREFILL_GRAPH.get('policy')}: the calibration puts "
            f"eager ahead at every width) -- group P runs --chunked-prefill-size {bucket} EAGER"
        )
    if p_prefill_graph_buckets() != list(_P_PREFILL_GRAPH.get("legacy", ()) or ()):
        return (
            f"PDFLIP P-PREFILL-GRAPH: bucket={bucket} captured={p_prefill_graph_buckets()} "
            f"(--p-prefill-graph-policy {_P_PREFILL_GRAPH.get('policy')}) -- group P runs "
            f"--chunked-prefill-size {p_chunked_prefill_tokens()} ({' '.join(p_prefill_graph_flags())}); "
            "a chunk replays the smallest captured bucket that holds it, above the largest it "
            "runs eager ('PREFILL-GRAPH eager reason=')"
        )
    tiny = [b for b in p_prefill_graph_buckets() if b != bucket]
    return (
        f"PDFLIP P-PREFILL-GRAPH: on bucket={bucket} -- group P runs "
        f"--chunked-prefill-size {bucket} and captures ONE full prefill graph "
        f"of {bucket} tokens per PP stage"
        + (f" plus tiny bucket(s) {tiny} (the 1-token END-ANCHOR and short rests "
           f"replay those instead of padding to {bucket})" if tiny else "")
        + f" ({' '.join(p_prefill_graph_flags())}); "
        "capture pool in the memory saver's cuda_graph tag (released by P's "
        "sleep, remapped by its wake); rank lines 'PREFILL-GRAPH captured ... "
        "capture_mib=' (cost) and 'PREFILL-GRAPH eager reason=' (every batch "
        "the graph did not take)"
    )

def d_replayssm_spec_ring_len() -> int:
    """The ring length group D gets: a power of two, >= 16 (the commit
    kernel's tl.dot floor) and >= the widest verify window (the draft block
    under DFLASH, the draft tokens under NEXTN -- :func:`d_verify_window`).
    With fold every commit the ring holds one window, so the floor is also the
    whole cost."""
    window = d_verify_window()
    ring = 16
    while ring < window:
        ring *= 2
    return ring

def p_transient_support(profile: Optional[str] = None, *, card: bool = False):
    """#242: the support table the launcher prices from -- the builtin one,
    with the point at :data:`P_ACTIVATION_RECORD_CHUNK` replaced by the
    profile's ``P_ACTIVATION_MIB`` record where it carries one.

    ``card=True`` (the P-KARTE): the LARGER of the builtin point and the record
    per stage. The card's reference K0 was normalised against the builtin point
    of its own boots (fnFL2x163-165); a record BELOW it would hand the card
    headroom its reference never had (PP1 lands at 86 MiB card free on the
    metal of 28.09., the card predicts 1274), a record ABOVE it is a load state
    the reference never saw (PP0 with two to five sequences in one chunk). Until
    the card reference is re-measured on this form (--p-card-reference-logs),
    only the second direction may reach it.
    """
    base = _live_stage_support(P_PREFILL_TRANSIENT_SUPPORT)
    rec, boots = p_activation_record(profile)
    if rec is None or len(rec) != base.n_stages or P_ACTIVATION_RECORD_CHUNK not in base.chunks:
        return base
    points = []
    for p in base.points:
        if p.chunk != P_ACTIVATION_RECORD_CHUNK:
            points.append(p)
            continue
        mib = tuple(max(a, b) for a, b in zip(p.mib, rec)) if card else tuple(rec)
        source = "record %s (%s)" % (P_ACTIVATION_RECORD, boots or "no boots named")
        if card:
            source += " max builtin %s" % p.source
        points.append(_p_card.TransientPoint(chunk=p.chunk, mib=mib, source=source))
    return _p_card.TransientSupport(model=base.model, points=tuple(points))

def _apply_p_chunk_policy_nf(ns, boot_form, argv_words: Sequence[str]) -> None:
    """H92: resolve --p-chunk-policy dynamic for --profile nextflash: the NF
    profile of THIS checkpoint (model key, never 27B numbers) and the NF hard
    limits against the P chunk the planner prices. The spec itself is built
    at env time (``p_chunk_policy_install``), where FR_P is the effective one
    (after the H25 draft post)."""
    from flliper.srt.pdflip import p_chunk_nf as _nf

    nf_args = _p_chunk_nf_args(ns, argv_words)
    model_key = pdflip_form.model_key(getattr(ns, "model", "") or "")
    try:
        prof = _nf.resolve_profile(str(nf_args["model"]), model_key)
        if boot_form is not None and prof.profile and boot_form.profile and prof.profile != boot_form.profile:
            raise _nf.NfChunkRefused(
                f"NF chunk profile {os.path.basename(prof.path)} is for --profile {prof.profile}, "
                f"this boot runs --profile {boot_form.profile}")
        if prof.ceiling_tokens != P_CHUNKED_PREFILL_TOKENS:
            raise _nf.NfChunkRefused(
                f"NF chunk profile {os.path.basename(prof.path)} was measured at P chunk "
                f"{prof.ceiling_tokens}, this boot prices {P_CHUNKED_PREFILL_TOKENS} "
                f"(FLLIPER_PDFLIP_P_CHUNKED_PREFILL_TOKENS): a_s belongs to its own residency")
        _p_chunk_nf_limits(nf_args, len(prof.stages))
    except _nf.NfChunkRefused as exc:
        raise SystemExit(f"--p-chunk-policy dynamic (NF): {exc}")
    _P_CHUNK.update(profile=prof, ns=nf_args)

def p_chunk_policy_install(env_p: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """``(env additions for group P, launcher lines)``; ``({}, [])`` under
    'fixed' (env and front log byte-identical). FR_P is read from the group
    P environment that ships (the effective vector after the draft post)."""
    from flliper.srt.pdflip import p_chunk_nf as _nf
    from flliper.srt.pdflip import p_chunk_policy as _pcp

    prof = _P_CHUNK.get("profile")
    if prof is None:
        return {}, []
    fr = _nf.parse_fractions(env_p.get("FLLIPER_MOE_RESIDENT_EXPERT_FRACTION", ""))
    try:
        stages, fr_note = _nf.fr_adjusted_stages(prof, fr)
        limits = _p_chunk_nf_limits(_P_CHUNK["ns"], len(stages))
    except _nf.NfChunkRefused as exc:
        raise PdFlipLaunchRefused(f"--p-chunk-policy dynamic (NF): {exc}") from None
    spec = _pcp.PolicySpec(stages, limits, prof.source)
    _P_CHUNK["spec"] = spec
    env = {_pcp.POLICY_ENV: _pcp.POLICY_DYNAMIC, _pcp.SPEC_ENV: spec.to_json(),
           _nf.BUDGET_ENV: _nf.BUDGET_STREAM}
    lines = ["PDFLIP " + _pcp.armed_line(spec, "group=P line=NF budget=stream")
             + f" -- ceiling = group P's --chunked-prefill-size {P_CHUNKED_PREFILL_TOKENS} (unchanged), "
               f"raster {limits.page} = ChunkLimits.page, profile {os.path.basename(prof.path)}, {fr_note}"]
    lines.extend(_nf.dry_run_lines(spec))
    return env, lines

def l3_persist_attach(log: Log, directory: str, dry: bool) -> Tuple[int, int, int]:
    """L3P: reattach this boot to the persistent store; returns
    ``(files, bytes, removed)``.

    Before any rank starts (so nothing here sits inside a rank collective),
    from the store's persistent index (``_l3_attach_from_index``, 28.09.) --
    ONE walk only when the index is unusable, exactly when the ranks walk:
    counts the ``.bin`` pages a previous boot left (the
    ``L3-PERSIST`` line is the acceptance marker) and removes
    - the ``<final>.tmp.<uuid>`` staging files a crashed writer left behind --
      no writer of this store is alive yet, so a staging file here is
      abandoned by construction;
    - ORPHAN pages (N1): a ``.bin`` whose stem ends in a suffix NO group
      scans (``L3_SUFFIXES.*``, the groups' own records from the previous
      boot). No evictor indexes them, so they would sit against the cap as
      foreign bytes forever; the key already made them unreachable (e.g. a
      changed uneven-TP ratio moves the identity hash in every key). Without
      any record yet they are counted, never removed.
    Canonical partials are NOT touched: HiCacheFile's attach sweep keeps the
    resumable ones by marker.

    It also publishes the GROUP-WIDE attach epoch (N2,
    ``FLLIPER_PDFLIP_L3_EPOCH``): every rank of both groups grades "inherited"
    against the same instant, not against its own start time.

    H8, named: ``/clear_hicache_storage_backend`` clears the WHOLE persistent
    store (every page of this identity), exactly as it always cleared the
    per-boot one; the identity and suffix records survive (they are not
    ``.bin`` pages).
    """
    epoch = time.time()
    if not dry:
        os.environ["FLLIPER_PDFLIP_L3_EPOCH"] = f"{epoch:.6f}"
        os.environ.pop("FLLIPER_PDFLIP_L3_INHERITED_SUFFIXES", None)
    files = nbytes = removed = orphans = orphan_bytes = 0
    t0 = time.monotonic()
    if not os.path.isdir(directory):
        log(f"L3-PERSIST fresh dir={directory} (no previous store) epoch={epoch:.3f}")
        return 0, 0, 0
    known = l3_registered_suffixes(directory)
    revoked = l3_revoked_windows(directory)
    rev_n = rev_bytes = rev_failed = 0
    if not dry:
        # N1: the suffixes the previous boot's groups scanned, published ONCE
        # for every rank -- a page under one of them that this rank's own
        # group does not scan is the sibling's (or this group's before an
        # identity-hash drift): inherited. Anything else unindexed still
        # counts against W8b (a blind filter on this boot's own writes).
        if known:
            os.environ["FLLIPER_PDFLIP_L3_INHERITED_SUFFIXES"] = json.dumps(list(known))
        else:
            os.environ.pop("FLLIPER_PDFLIP_L3_INHERITED_SUFFIXES", None)
    # 28.09. (operator order): the same answer from the store's persistent
    # index (snapshot + journals) -- the directory is walked only when the
    # ranks would walk it too (snapshot missing/broken, another kernel boot).
    from flliper.srt.mem_cache.storage.file import store_journal as _sj

    if _sj.enabled():
        got = _l3_attach_from_index(log, directory, dry, epoch, known, revoked)
        if got is not None:
            return got
    # NF metal rc12z30c: this walk is THE one walk of the boot -- the full map
    # it sees is persisted as the snapshot at once, so no rank walks after it
    snap_items: Optional[List[Tuple[float, str, int]]] = (
        [] if _sj.enabled() and not dry else None)

    def _keep(st, name):
        if snap_items is not None:
            snap_items.append((st.st_mtime, name[:-4], max(st.st_blocks * 512, st.st_size)))

    for root, _dirs, names in os.walk(directory):
        for name in names:
            p = os.path.join(root, name)
            if ".tmp." in name:
                if not dry:
                    try:
                        os.unlink(p)
                        removed += 1
                    except OSError:
                        pass
                continue
            if not name.endswith(".bin"):
                continue
            try:
                st = os.lstat(p)
            except OSError:
                continue
            size = st.st_blocks * 512
            if revoked and any(t0 <= st.st_mtime <= t1 for t0, t1, _r in revoked):
                if _l3_move_revoked(directory, root, name, dry):
                    rev_n += 1
                    rev_bytes += size
                else:
                    rev_failed += 1
                    _keep(st, name)  # still on disk: the map must name it
                continue
            if known is not None and not name[:-4].endswith(known):
                orphans += 1
                orphan_bytes += size
                if not dry:
                    try:
                        os.unlink(p)
                        removed += 1
                    except OSError:
                        _keep(st, name)
                continue
            files += 1
            nbytes += size
            _keep(st, name)
    walk_s = time.monotonic() - t0
    if snap_items is not None:
        snap_items.sort(key=lambda it: it[0])
        try:
            with _sj.snapshot_lock(directory):
                _n, _b, _gone = _sj.persist(directory, snap_items, len(snap_items), int(epoch))
            log(f"L3-PERSIST SNAPSHOT written by the one walk: {_n} files, {_b} B, "
                f"{_gone} journal(s) of earlier boots folded in -- the ranks load it, none walks")
        except (OSError, ValueError) as exc:
            log(f"L3-PERSIST snapshot NOT written ({type(exc).__name__}: {exc}) -- the first "
                f"rank walks once under the store lock")
    log(
        f"L3-PERSIST {'reuse' if files else 'fresh'} dir={directory} files={files} bytes={nbytes} "
        f"({nbytes / host_ledger.GIB:.2f} GiB) orphans={orphans} ({orphan_bytes / host_ledger.GIB:.2f} GiB, "
        + ("no suffix record yet: counted, kept" if known is None else
           f"suffix in no group's record: {'would be ' if dry else ''}removed")
        + f") staging+orphans_removed={removed}"
        + (f" revoked={rev_n} ({rev_bytes / host_ledger.GIB:.2f} GiB, "
           f"{'would be ' if dry else ''}moved to {directory.rstrip('/') + L3_REVOKED_SUFFIX}"
           f"{f', {rev_failed} NOT moved (kept out of the count, still on disk)' if rev_failed else ''}"
           f"; windows {[(time.strftime('%H:%M:%SZ', time.gmtime(a)), time.strftime('%H:%M:%SZ', time.gmtime(b)), r) for a, b, r in revoked]})"
           if revoked else "")
        + f" walk_s={walk_s:.1f} epoch={epoch:.3f}"
        + (" (DRY-RUN: nothing removed)" if dry else "")
        + " -- the ranks seed the L3 index and the evictor ledger from it at attach"
    )
    return files, nbytes, removed

def common_flags(
    model: str,
    s_gb: int,
    m_mib: int,
    store_cfg: str,
    max_kv_per_request: int,
    write_policy: str = "write_back",
    group: str = "both",
    random_seed: int = RANDOM_SEED,
    barlink_cap_cycles: int = BARLINK_BAR1_CAP_CYCLES,
    census_interval: int = COLLECTIVE_CENSUS_INTERVAL,
    # #1356 [fix] KEYWORD-ONLY. Inserted positionally between `group` and
    # `random_seed`, this shifted THREE callers at once -- argv_p:2772,
    # argv_d:2888 and the form-key build at :10060 all pass positionally, so
    # `random_seed` landed in `vision` and everything after it moved one place.
    # See argv_p for the failure this produced on the normal start path.
    *,
    vision: str = VISION_OFF,
    profile: str = PROFILE_QWEN27B,
    # #1386 [Minimalform HiCache switch]: OFF (default) is byte-identical to
    # every argv this function built before the flag existed. ON drops the
    # WHOLE hicache block below rather than passing a smaller size through it
    # -- "abschalten heisst abschalten, nicht kleinrechnen" (no
    # `--hicache-size 0`, which would still enter the HiCache code path on
    # both groups). The identical bool also zeroes `anchors_gib`/`rings_gib`
    # in `host_ledger.charge_terms`; `launcher.main` reads ONE local once and
    # hands it to both, never two.
    hicache_disabled: bool = False,
    # #1369: user order 2026-09-14 ("DIE 48GB MUESSEN WEG"). True (default)
    # is byte-identical to every argv this function built before the flag
    # existed -- `--enable-weights-cpu-backup` unconditional, both groups,
    # every arm. False drops it: the ROOT this order names
    # (`weight_exchange.weights_cpu_backup_armed`), resolved ONCE by the one
    # caller that has both `ns.pdflip_weights_cpu_backup` and the arm, and
    # passed down rather than re-derived here -- a second call site for the
    # same predicate is the #1358 defect class ("2.16 GiB" had three causes)
    # one flag over.
    weights_cpu_backup: bool = True,
    # HW-P1a 1003: the launch's card count (argv_p/argv_d pass len(budgets),
    # one budget per card); the default is the reference rig's 3.
    n_cards: int = PDFLIP_CARD_COUNT,
) -> List[str]:
    """Flags BOTH groups share.

    ``--max-running-requests`` is NOT here any more (C1/R-12): sharing it
    pinned P and D to one value by construction, and law 2 says the two are
    independent.  It is emitted per group from --p-bs / --d-bs instead.

    ``--disable-overlap-schedule`` is not here either.  It is group P's flag,
    not a common one: the justification is ``pp_size > 1``
    (server_args.py:19507, "Pipeline parallelism is not compatible with
    overlap schedule", plus the same forcing in
    arg_groups/overrides._pipeline_parallel_overlap_disable), and group D runs
    pp_size=1.

    ``store_cfg`` is the RENDERED backend extra-config JSON
    (:func:`store_extra_config`), not a size.  #1236: it used to be a
    ``store_gib`` float, because the store was a tmpfs the HOST RAM ledger had
    sized; the store is a directory on disk now and its size comes from the P
    KV pool (:func:`plan_store`), so passing a GiB number through here would be
    carrying the old currency past the point where it stopped meaning anything.

    MEASURED consequence of the old placement: D booted with
    disable_overlap_schedule=True inheriting a PP-only reason (BSSCALE_0907.md
    D4), so CPU scheduling of round n+1 could not hide behind GPU work of
    round n on a group that has no pipeline at all.  See argv_p / argv_d.
    """
    return [
        "--model-path", model,
        # 27B line G2: [] unless --tokenizer-path was installed (apply_tokenizer_path).
        *tokenizer_path_flags(),
        "--trust-remote-code",
        # #1356: TEXT-ONLY BY DEFAULT, on BOTH groups, via the ONE decision the
        # model config already makes (`model_config.py:436-447`, read at
        # `server_args.py:14112`). NOT `--language-only`: that flag is the
        # encoder-disagg RECEIVER and refuses without `--encoder-urls`
        # (`server_args.py:17996`), so passing it here would kill the launch
        # rather than drop the tower. And not a model-name list either -- that
        # is the #1362 fossil class. `--no-enable-multimodal` became spellable
        # in this same commit; before it, `False` existed in the tri-state and
        # in the config branch and no caller could say it.
    ] + vision_model_flags(vision) + [
        # #1362 FOSSIL 4: this was the LITERAL "Qwen3.8-27B" on every boot,
        # including the 4B-FP8 transition vehicle -- the served name is what the
        # front, the load drivers and every probe address, so a wrong one makes
        # a boot of a DIFFERENT model answer under the old model's name. Derived
        # from the model path by default and overridable by flag; never a
        # constant, because a constant here is a silent identity swap.
        "--served-model-name", served_model_name(model),
        # HW-GENERIC 1002 S2 (enabling): one rank per card in order_cards
        # order -- the topology's rank map, "0,1,2" on the N=3 release.
        "--rank-gpu-id", _topo.rank_gpu_id_csv(n_cards),
        "--skip-server-warmup",
        "--kv-cache-dtype", KV_CACHE_DTYPE,
        "--context-length", str(CONTEXT_LENGTH_TOKENS),
        # C14/K9: the per-request KV ceiling, decoupled from the model
        # context. A CEILING, not the pressure relief: on D the device pool
        # is far below d_bs x 262,144, so this never binds first.
        "--max-kv-per-request", str(max_kv_per_request),
        "--reasoning-parser", "qwen3",
        "--tool-call-parser", "qwen3_coder",
        "--chat-template-default-kwargs", '{"preserve_thinking": true}',
        "--enable-cache-report",
        "--enable-metrics",
    ] + ([] if hicache_disabled else [
        # #1386: the WHOLE block, never a subset -- `--hicache-size 0` or a
        # bare `--enable-hierarchical-cache` alone would still walk server_args
        # into the HiCache code path (a smaller cache, not an absent one).
        # Dropping every flag here is what makes `hicache_disabled` mean the
        # code path is never entered, matching the ledger side pricing 0.00
        # for the same reason (host_ledger.charge_terms).
        "--enable-hierarchical-cache",
        "--hicache-host-role", "staging",
        "--hicache-size", str(s_gb),
        # #1430: the mamba anchor pool is a fallback id range too (Stufe 4b
        # writes the states into the mamba arena); two slots, not 13. The
        # ledger still prices M (conservative). Override: FLLIPER_PDFLIP_MAMBA_ANCHOR_MIB.
        "--hicache-mamba-host-mib", str(int(os.environ.get("FLLIPER_PDFLIP_MAMBA_ANCHOR_MIB", "100") or m_mib)),
        "--hicache-write-policy", write_policy,
        "--hicache-storage-backend", "file",
        "--hicache-mem-layout", "layer_first",
        "--hicache-io-backend", "direct",
        "--hicache-storage-backend-extra-config", store_cfg,
        "--hicache-canonical-kv-page",
    ]) + [
        "--host", "127.0.0.1",
        # --p-prefill-graph N moves GROUP P's chunk to the graph bucket N
        # (p_chunked_prefill_tokens); D and the default keep the constant.
        "--chunked-prefill-size",
        str(p_chunked_prefill_tokens() if group == "P" else CHUNKED_PREFILL_TOKENS),
        "--scheduler-distributed-teardown",
        "--page-size", "1",
        # #1235: arbitrary and FIXED, which is the whole provenance -- see
        # RANDOM_SEED. Overridable by --random-seed.
        "--random-seed", str(random_seed),
        "--mamba-slot-reorder",
        "--kv-backing-relief",
        "--barlink", "--barlink-transport", "bar1",
        "--barlink-bar1-cap-cycles", str(barlink_cap_cycles),
        "--collective-census-interval", str(census_interval),
    ] + (early_read_flags(group) if _profile_early_read(profile) else []) + [
        # THE DOUBLY-STATED FACTS COME FROM ONE TABLE (#1235). --mamba-ssm-dtype
        # is here for both groups and the two uneven-DCP flags for group D; the
        # matching environment keys are written by build_env from the SAME
        # rows, because ServerArgs reads them from os.environ before the flags
        # publish themselves. See EarlyReadFact for the ordering evidence.
        "--enable-memory-saver",
    ] + (["--enable-weights-cpu-backup"] if weights_cpu_backup else [])

def prepare_weight_exchange(
    cards: List[Card],
    log: Log,
    weight_source: str,
    census_path: str,
    epoch: object,
    ring_h_mib: int,
    floor_mib: float = ARMING_FLOOR_MIB,
) -> Optional["xchg_residency.XchgResidency"]:
    """W71 (#1273 S7): price the exchange's VRAM peak BEFORE either group starts.

    Returns None on the ``ring`` arm, where nothing about this boot changes --
    the check is not merely skipped there, it has no subject: no wave schedule
    runs, so there is no co-residency window to price.

    On the ``exchange`` arm it solves spec section 5's table per card, per
    direction, per wave from a MEASURED census plus this boot's own live NVML
    totals, logs one ``PDFLIP-XCHG-CHECK`` row per (card, direction) and then
    either the ``PDFLIP-XCHG-ARMED`` line or raises
    :class:`xchg_residency.PdFlipXchgResidencyUnarmable` -- exit 2, the same
    contract and the same launch position as W32/W34/W49 above, and for the
    same reason: a refusal that arrives mid-flip arrives after VRAM has already
    been mutated.

    W71 REPLACES W49's ROLE HERE rather than joining it (spec section 0.4,
    finding R1-1): under ``exchange`` the launcher publishes no
    ``TMS_HOST_RING_*``, Sigma H is 0 and every ring inequality reads ``0 > 0``
    -- false, i.e. a gate that passes because it has nothing to grade.  An
    unarmed gate must never be read as a passed one, so the arming line carries
    ``ring_H_mib`` beside the peaks, and (round-2 review F2) a ``wired=`` token
    saying in words that no rank behaviour hangs off this arm until S6.

    THE REFUSAL EXITS 2 BECAUSE IT IS ENROLLED, NOT BECAUSE IT SAYS SO
    (round-2 review F1): :class:`xchg_residency.PdFlipXchgRefused` is in
    ``REFUSALS``, which is what ``cli()`` catches; the first cut of this slice
    raised a bare ``RuntimeError`` and exited 1 with a traceback while three
    docstrings, this one included, promised exit 2.
    """
    if weight_source != "exchange":
        return None
    census = load_xchg_census_for_cards(census_path, cards, log)   # raises W71 by name
    res = xchg_residency.solve(cards, census, floor_mib)
    for ln in res.lines:
        log(ln)
    if not res.armed:
        head = xchg_residency.refusal_head(res)
        log(head)
        for ln in res.refusals:
            log(ln)
        raise xchg_residency.PdFlipXchgResidencyUnarmable(
            head + "\n" + "\n".join(res.refusals)
        )
    log(
        xchg_residency.armed_line(
            res,
            epoch,
            xchg_residency.region_mib_from_layout(
                xchg_region_pairs(len(cards)),
                XCHG_REGION_SLOTS_PER_PAIR,
                XCHG_REGION_SLOT_MIB,
                XCHG_REGION_HEADER_MIB,
            ),
            ring_h_mib,
            XCHG_RANK_BEHAVIOUR_WIRED,
            XCHG_UNWIRED_REASON,
        )
    )
    return res

def _env_knobs(ns) -> Dict[str, object]:
    """The #1235 flags ``build_env`` takes, gathered once.

    Three call sites build an environment and every one of them must pass the
    same seven values; spelling them out three times is how the pair of
    environments would drift the day an eighth arrives.
    """
    return {
        "seam_digest_armed": bool(getattr(ns, "pdflip_seam_digest", False)),
        "barlink_build_window_cap_s": ns.barlink_build_window_cap_s,
        "pp_chain_recv_stall_s": ns.pp_chain_recv_stall_s,
        "pp_occupant_horizon_s": ns.pp_occupant_horizon_s,
        "match_refusal_census_every": ns.match_refusal_census_every,
        "arming_floor_solved": ns.arming_floor_solved,
        "hicache_bigram_keys": ns.hicache_bigram_keys,
        "hicache_flush_publish_sweep": ns.hicache_flush_publish_sweep,
        "lane_coverage_dir": lane_coverage_dump_dir(ns),
        "lane_coverage_token": lane_coverage_boot_token(ns),
        "pdflip_boot_token": pdflip_boot_token(ns),
        "profile": getattr(ns, "profile", PROFILE_QWEN27B),
        "flip_weights": getattr(ns, "flip_weights", "family"),
        # Task #58: the vision form, so `build_env` can publish the arming
        # variable for P. Gathered HERE with the other six for the reason this
        # function exists: three call sites build an environment and a value
        # spelled out at each of them is a value that drifts.
        "vision": str(getattr(ns, "pdflip_vision", VISION_OFF)),
        # H125: source and place of the transient stage (P only, see build_env).
        "vision_source": _vision_knob(ns, "pdflip_vision_source", VISION_SOURCE_DISK),
        "vision_place": _vision_knob(ns, "pdflip_vision_place", VISION_PLACE_AUTO),
        "vision_on_d": vision_on_d_of(ns),
        # PDFLIP-FORM: the one resolved form (launcher.main sets it on ns).
        "boot_form": getattr(ns, "pdflip_boot_form", None),
        # 27B FP8: one byte layout on every rank of BOTH groups (the two
        # groups flip bytes into each other), hence gathered here once.
        "fp8_uniform_marlin": bool(getattr(ns, "fp8_uniform_marlin", False))
        and checkpoint_quant_method(str(getattr(ns, "model", "")))
        in UNIFORM_MARLIN_QUANT_METHODS,
    }

def gate_w11(log_p: str, log: Log, rope_ctx_mib: float = 0.0) -> Dict[str, object]:
    """THE LAUNCHER'S W11 GATE, both instruments and both refusals.

    Grades group P's last-stage draft build from its log (see
    :func:`check_draft_resident`) and REFUSES the boot before the front opens
    when either instrument says no:

    * ``W11 PdFlipDraftResidentOverBudget`` -- the residue is over the budget
      ``T_P`` was derived from, so the corridor derivation is refuted by this
      boot (weg2dk2's 3994 MiB build).
    * ``W11b PdFlipDraftBuildUnaccounted`` -- the BUILD is not explained by
      residue + release, the fix-2 shape ``resident_mib`` is blind to.

    #1233 fix 7: a function, for the same reason as :func:`choose_host_ledger`
    -- the fix-6 review gated the W11b refusal off and the whole slice stayed
    green, because the only caller sits behind two launched servers.  A gate
    nothing can reach is a gate nothing can pin.
    """
    # --spec-form DFLASH: the last stage's drafter is the EXTERNAL DFlash
    # checkpoint, whole (TP1), priced off its headers -- the same term
    # ring_table.checkpoint_stage_weights charges the stage with
    # (external_drafter_mib_from_argv); the NEXTN constant is its own head.
    _w11_budget = None
    if spec_form_is_dflash():
        from flliper.srt.speculative.dflash_pricing import dflash_draft_family_bytes

        _w11_budget = float(
            sum(dflash_draft_family_bytes(str(_SPEC_FORM["draft_path"])).values())
        ) / float(1 << 20)
    w11 = check_draft_resident(log_p, budget_mib=_w11_budget, rope_ctx_mib=rope_ctx_mib)
    log(f"W11 DRAFT-RESIDENT P last stage resident_mib={w11['resident_mib']} budget_mib={w11['budget_mib']:.1f} "
        f"tol_mib={w11['tol_mib']:.0f} over_mib={w11['over_mib']} resident_ok={w11['resident_ok']} "
        f"| W11b BUILD-ACCOUNTING nvml_delta_mib={w11['nvml_delta_mib']} = resident_mib + "
        f"head_released_mib={w11['head_released_mib']} + tag_pool_inactive_mib={w11['tag_pool_inactive_mib']} "
        f"+ outside_torch_mib={w11['outside_torch_mib']} "
        f"+ default_pool_inactive_mib={w11['default_pool_inactive_mib']} "
        f"+ other_live_mib={w11.get('other_live_mib')} "
        + (f"+ rope_ctx_mib={w11['rope_ctx_mib']:.1f} (YaRN draft RoPE context, budget and build) "
           if w11.get("rope_ctx_mib") else "")
        + f"+ unaccounted_mib={w11['unaccounted_mib']} cache_term={w11.get('cache_term')} "
        f"card_free_mib={w11.get('card_free_mib')} "
        f"(tol {w11['accounting_tol_mib']:.0f}) accounted={w11['accounted']} "
        f"ok={w11['ok']}")
    if not w11["resident_ok"]:
        raise PdFlipLaunchRefused(f"W11 PdFlipDraftResidentOverBudget: measured resident_mib={w11['resident_mib']} vs budget "
                                f"{w11['budget_mib']:.1f}+{w11['tol_mib']:.0f} MiB -- the last stage's pool was priced against "
                                f"this budget (#1286 stage_fixed) and its corridor target (idle NVML free at {P_CORRIDOR_TOP_MIB:.0f}) cannot hold")
    if not w11["accounted"]:
        # #1233 fix 6: the SECOND instrument. resident_mib is a model-graph
        # quantity, so a table released only from the graph passes it while
        # still sitting on the card -- the fix-2 failure the W11 gate exists
        # for. The build must be explained by residue + released.
        #
        # #66 (21.09.): DIESES GATE VERHINDERT EIN OOM; ES IST KEINE
        # BUCHPRUEFUNG UM IHRER SELBST WILLEN.
        #
        # Vier Boots (fnFL2v86, v89, v90, v91) haben an ihm geendet, waehrend
        # die Karte nachweislich Luft hatte: v91 erreichte D READY nach 255,1 s
        # und TP0 hielt 2,24 GiB frei (allocated 16,60, reserved 26,82 von
        # 31,34 GiB). Die unerklaerten 533,7 MiB sind REAL belegt -- sie sind
        # Teil der 5334, die der Build kostet, und der Korridor hat sie
        # getragen. Drei Anlaeufe haben sie nicht benannt: nicht der
        # Cache-Term (torch_total, nach empty_cache unveraendert 2597,7),
        # nicht `outside_torch` (0,0, im selben Kartenzustand gemessen), nicht
        # `resident_mib` (zaehlt Parameter UND Buffers).
        #
        # Die Prämisse des Gates ist "der Build koennte mehr belegen, als wir
        # sehen, und die Karte ueberlaufen". Wo die freie Karte die
        # Unerklaerten um ein Vielfaches uebersteigt, ist diese Praemisse
        # widerlegt -- und ein Gate auf einer widerlegten Praemisse verweigert
        # nicht, es MELDET. Laut, mit allen Zahlen, und mit dem Laufzeit-Riegel
        # dahinter (der Korridor prueft jede Phase weiter).
        #
        # KEINE TOLERANZERWEITERUNG: die 256 MiB stehen unveraendert. Was sich
        # aendert, ist allein die FOLGE einer unerklaerten Differenz, die die
        # Karte beweisbar traegt. Faellt der freie Rand unter das Vielfache,
        # verweigert das Gate wieder.
        _free = w11.get("card_free_mib")
        _un = abs(float(w11["unaccounted_mib"] or 0.0))
        if _free is not None and float(_free) >= W11B_FREE_OVER_UNACCOUNTED * _un:
            log(
                "W11b ACCOUNTING-GAP GEMELDET STATT VERWEIGERT: "
                f"unaccounted_mib={w11['unaccounted_mib']} bei card_free_mib={_free} "
                f"auf der engsten Karte -- das ist das {float(_free) / max(_un, 1e-9):.1f}-fache "
                f"(Schwelle {W11B_FREE_OVER_UNACCOUNTED:.1f}x). Die Bytes sind real belegt und "
                "vom Korridor getragen; was sie SIND, ist offen (#66). Das Gate verhindert ein "
                "OOM, und diese Praemisse ist hier widerlegt -- die Toleranz bleibt unveraendert "
                "bei %.0f MiB." % w11["accounting_tol_mib"]
            )
            w11["accounting_gap_reported"] = True
            return w11
        raise PdFlipLaunchRefused(f"W11b PdFlipDraftBuildUnaccounted: nvml_delta_mib={w11['nvml_delta_mib']} is not explained by "
                                f"resident_mib={w11['resident_mib']} + head_released_mib={w11['head_released_mib']} "
                                f"+ tag_pool_inactive_mib={w11['tag_pool_inactive_mib']} "
                                f"+ outside_torch_mib={w11['outside_torch_mib']} "
                                f"+ default_pool_inactive_mib={w11['default_pool_inactive_mib']} "
                                f"+ other_live_mib={w11.get('other_live_mib')} "
                                f"(unaccounted {w11['unaccounted_mib']} MiB, tolerance {w11['accounting_tol_mib']:.0f}) -- either the "
                                f"never-loaded lm_head was unbound from the graph without being freed (fix 2's shape, invisible to "
                                f"resident_mib by construction) or one of the three instruments did not measure")
    return w11

def _installed_max_share_from_record() -> Optional[float]:
    """max(v)/sum(v) of the INSTALLED D ownership vector, if any record has it.

    None today: no sizing record carries the installed vector yet, and this
    returns None rather than guessing. It exists so the refusal above is WIRED
    the day the record does carry it -- the alternative is a TODO, and a TODO
    is what let the 1 GB constant survive unquestioned for a whole campaign.
    """
    try:
        import json

        with open(measured_record_path(), "r") as fh:
            rec = json.load(fh)
    except Exception:  # noqa: BLE001 - no record is not a refusal
        return None
    v = None
    if isinstance(rec, dict):
        for key in ("d_installed_token_vector", "installed_token_vector"):
            cand = rec.get(key)
            if isinstance(cand, (list, tuple)) and cand:
                v = [int(x) for x in cand]
                break
    if not v or sum(v) <= 0:
        return None
    return max(v) / float(sum(v))

def launch_group(spec: GroupSpec, tree: str, log: Log, dry: bool) -> None:
    # #1275: the LOG copy is redacted; spec.argv itself is untouched and is what
    # is actually exec'd. A log file outlives the boot and gets pasted into
    # records -- a strictly worse channel than /proc, which is inherent.
    log(f"group {spec.name} argv: " + " ".join(
        shlex.quote(a) for a in admin_key_mod.redact_argv(spec.argv)))
    if spec.name in ("P", "D"):
        # #233 review MUSS 1 + 2: the group's REAL argv recipe (the key its
        # ranks publish under, for the next boot), the frozen pointer state and
        # the boot's recipes re-asserted; a probe/argv drift is named.
        for _rline in corridor_guard.arm_group_recipe_env(spec.name, spec.argv, spec.env):
            log_boot_state(log, _rline)
    if dry:
        return
    # IPC Phase 1: the ranks write their RankState next to the group log;
    # records of an earlier launch into the same log path are removed first,
    # so a record the launcher grades is always THIS launch's.
    _rs_dir = rank_state_dir_for(spec)
    _rs_old = rank_state_mod.clear_rank_state_dir(_rs_dir)
    spec.env[rank_state_mod.RANK_STATE_ENV] = _rs_dir
    log(f"group {spec.name} rank state -> {_rs_dir}" + (f" ({_rs_old} record(s) of an earlier launch removed)" if _rs_old else ""))
    if spec.name in ("P", "D"):
        # W98 (z30u): Stufe 2b's keeper writes its kept ranges here; the
        # launcher drops exactly those once D is ready (release_shared_cache).
        spec.env[shared_cache_release_mod.MANIFEST_ENV] = shared_cache_release_mod.manifest_dir_for(spec.log)
    fh = open(spec.log, "ab")
    _logged_argv = ' '.join(shlex.quote(a) for a in admin_key_mod.redact_argv(spec.argv))
    fh.write(f"=== PDFLIP group {spec.name} launched {_now()} ===\nargv: {_logged_argv}\n".encode())
    fh.flush()
    _cwd = core_cwd_for(spec.log, tree, log)
    p = subprocess.Popen(spec.argv, env=spec.env, stdout=fh, stderr=subprocess.STDOUT, cwd=_cwd,
                         start_new_session=True)
    spec.pid = p.pid
    spec.proc = p
    spec.cwd = str(_cwd or "")
    log(f"group {spec.name} pid {p.pid} (session id = pid) log {spec.log} cwd {_cwd}")
    if spec.name in ("P", "D"):
        # Nutzer 29.09.: rigdash liest daraus „aktiv“ je Schalter (argv redigiert wie im Log).
        boot_state_write(log, "loading", fields={f"groups.{spec.name}": {
            "state": "loading", "pids": [p.pid], "ready_ts": None, "rankstate_dir": _rs_dir, "ranks": [],
            "launch": state_file_mod.launch_snapshot(admin_key_mod.redact_argv(spec.argv), spec.env),
            "form": group_form_block(spec.env)}})

def memts_pgrep_pattern(tag: str) -> str:
    """Anchored argv pattern of THIS tag's logger and its supervisor, and of
    nothing else: starts with ``bash``, carries the script path, and the csv
    path must END at a space or the line end (``..._fnFL2x15.csv`` never
    matches ``..._fnFL2x157.csv``, and never another line's tag)."""
    return rf"^bash .*{re.escape(MEMTS)} {re.escape(memts_csv_path(tag))}( |$)"

def booked_rest_admit_deltas(cards: List[Card], rests: Sequence[Optional[int]], vote_deltas: Sequence[int],
                             profile: Optional[str] = None, group: str = "D",
                             log: Optional[Log] = None, rest_name: Optional[str] = None) -> List[int]:
    """PDFLIP-ADMIT-BOOK (desk/27b-admit-trim-1024, budget_rest): per card ``max(vote-line delta,
    ceil8(D_ADMIT_NEED_MIB))`` where the card has a booked rest and the record's form runs; the vote-line deltas
    unchanged everywhere else (no record, foreign form, switch off, group P). The admission peak (post-wake handbacks,
    up to 522 MiB) lands before the forward trim and no trim can return it (the vote trim already ran, the remaining
    cache is not returnable); the budget therefore keeps it free -- a priced post, never a reserve. Rank-uniform: a
    launcher term, no width moves rank-locally (SOLL R11)."""
    n = len(cards)
    if str(group).upper() != "D":
        return list(vote_deltas)
    needs, src = d_admit_need_record(profile, n, cards)
    if needs is None:
        if log is not None and (src.startswith("admission need unmeasured") or "name no form" in src
                                or src.endswith("=0")):
            log(f"{_budget_rest.ADMIT_MARKER} group={group}: off -- {src}; the vote-line delta alone")
        return list(vote_deltas)
    out = _budget_rest.admit_deltas(rests, vote_deltas, needs)
    # review S1: the need is relative to the rest it was derived beside -- another rest is named loudly (and still
    # booked: dropping the post would be the OOM direction), never carried on silently
    basis_name, basis_mib = _budget_rest.ADMIT_REST_BASIS
    off = [(i, r) for i, r in enumerate(rests) if r is not None and int(r) != int(basis_mib)]
    if log is not None and (off or (rest_name is not None and str(rest_name) != basis_name)):
        log(f"{_budget_rest.ADMIT_MARKER} REST-MISMATCH group={group}: {_budget_rest.ADMIT_RECORD} was derived beside "
            f"{basis_name} {basis_mib} MiB per card; this pass books {rest_name or '?'} "
            f"{','.join('-' if r is None else str(r) for r in rests)} -- the need still books (dropping it is the OOM "
            f"direction) but is NOT re-derived for this rest: re-measure with python -m flliper.srt.pdflip.budget_rest "
            f"--admit-need")
    if log is not None:
        for i, c in enumerate(cards):
            if rests[i] is None:
                why = "no booked rest on this card -> nothing"
            elif out[i] == int(vote_deltas[i]):
                why = f"the vote-line delta {vote_deltas[i]} already carries it"
            else:
                why = f"books {out[i]} (vote-line delta {vote_deltas[i]})"
            log(f"{_budget_rest.ADMIT_MARKER} group={group} ordinal={i} nvml{c.nvml_index}: admission need "
                f"{needs[i]:.0f} MiB ({_budget_rest.ADMIT_RECORD}, {src}; {_budget_rest.ADMIT_BASIS}) -> "
                f"max(vote-line delta, ceil8(need)) = {out[i]} MiB: {why}")
    return out

def d_extend_vote_fixed_live_material(profile: Optional[str], cards: Optional[Sequence[Card]],
                                      model: Optional[str]) -> Tuple[Optional[str], Optional[dict], str]:
    """VOTE-FIXED-LIVE rule 5: ``(digest, material, why)`` of the D form -- profile, the H87 footprint key of the
    checkpoint (config + quant sidecars + (name, dtype, shape) of every tensor: the quant format is IN it), the D
    inventory, D's effective KV dtype and draft (review S3). NOT in the form (named): the code revision, the chunk
    width, the seats, the torch cache cap, the HiCache configuration. ``(None, None, why)`` with the switch off,
    without cards or without a readable checkpoint (fail closed: no store, the static record alone)."""
    if not _vfl.switch_on():
        return None, None, f"{_vfl.ENV_ON}=0"
    if not cards:
        return None, None, "no D cards"
    prof = profile if profile is not None else pdflip_form.DEFAULT_PROFILE
    if model is None:
        form = pdflip_form.current_form()
        model = form.model if form is not None else ""
    try:
        fp, fp_why = pdflip_form.footprint_key(model or "")
    except Exception as exc:  # noqa: BLE001 -- unreadable checkpoint: not a form
        fp, fp_why = None, f"footprint unreadable ({type(exc).__name__})"
    if not fp:
        return None, None, f"no footprint ({fp_why})"
    row = pdflip_form.profile_row(prof)
    try:
        inventory = card_identity.inventory_signature(list(cards))
    except Exception as exc:  # noqa: BLE001
        return None, None, f"inventory unreadable ({type(exc).__name__})"
    kv = str(_VFL_ARGV.get("kv") or "") or (str(getattr(row, "kv_dtype", "") or "") if row is not None else "")
    draft = _vfl_effective_draft(row)
    mat = _vfl.form_material(prof, fp, pdflip_form.model_key(model or ""), inventory, kv, draft)
    dig = _vfl.form_digest(mat)
    return (dig, mat, f"form {dig} = {mat}") if dig else (None, None, "no form material")

def apply_owned_miss_record_default(ns) -> Optional[str]:
    """PR: ``FLLIPER_PDFLIP_OWNED_MISS_RECORD`` into ``ns.env_d`` unless --env-d
    states it (every reader -- the D ranks, ``d_owned_miss_ms`` -- sees ONE
    value). Returns the line naming it, or None."""
    env_d = str(getattr(ns, "env_d", "") or "")
    if OWNED_MISS_RECORD_ENV in parse_group_env(env_d):
        return None
    root = owned_miss_record_root()
    ns.env_d = set_group_env(env_d, OWNED_MISS_RECORD_ENV, root)
    return (f"D-EIGENTUM MISS-RECORD (#239 S3f, PR): --env-d {OWNED_MISS_RECORD_ENV}={root} "
            f"(D ranks write paired prefill miss records at their sleep; the owned solve reads "
            f"them as RECORD from K={_owned_miss_min_paired()} paired forwards per rank)")

def d_owned_miss_ms(ns, *, env_d: Mapping[str, str], host: int
                    ) -> Tuple[Optional[Tuple[float, float]], str]:
    """#239 S3f Miss-Record (29.09.): the owned solve's cost per missed expert
    row (host, worker). RECORD = the D ranks' own records in group D's
    ``FLLIPER_PDFLIP_OWNED_MISS_RECORD`` directory (``layers.moe.pool_miss_cost``)
    > LOG-BOOTSTRAP (sidecar entry of ``pdflip.tools.owned_miss_record``, "aus
    Log (Uebergang)", dropped once a rank record exists) > BUILTIN (profile
    constant ``OWNED_MISS_MS``) > UNMEASURED. The env names the records root
    of the line; the records of this model lie in its
    ``<model_id>/owned_miss`` (``pool_miss_cost.record_dir_for``). ``(None, "")`` when nothing is
    measured -- the solve then keeps its seed, byte-identical."""
    from flliper.srt.planner import expert_residency as _er

    # getattr like every other ns reader of this module (d_pool_peak_record,
    # profile_row): a namespace without profile/model is the default profile.
    profile = getattr(ns, "profile", None)
    model = str(getattr(ns, "model", "") or "")
    try:
        builtin = list(_pconst("OWNED_MISS_MS", profile))
        builtin_src = _pconst_boots("OWNED_MISS_MS", profile)
    except (KeyError, TypeError):
        builtin, builtin_src = None, ""
    from flliper.srt.layers.moe import pool_miss_cost as _miss_cost

    root = str(env_d.get("FLLIPER_PDFLIP_OWNED_MISS_RECORD", "") or "").strip()
    # F0-F: the records root of the line AND its pre-rename sibling are read (writers: the root alone)
    rank_dirs = _miss_cost.record_dirs_for(root, model) if root else None
    ms, tier, src = _er.resolve_owned_miss_ms(
        _er.read_owned_miss_records(measured_record_path()),
        rank_records=_er.read_owned_miss_rank_records(rank_dirs), host=int(host),
        model=model, builtin=builtin, builtin_source=builtin_src)
    if tier == _er.OWNED_MISS_UNMEASURED:
        # PR: the seed stays -- the line still names a too-young RECORD's count
        return None, src if "RECORD zu jung" in src else ""
    return ms, src

def d_extend_rate_fill_null(ns, rate: Sequence[Optional[float]]):
    """NF port 1003 (order 980 item 2): ``(rates, fill)`` -- ``rate`` (the profile's
    ``D_EXTEND_GROWTH_PER_ROW_MIB``) with every NULL rank given a start rate DERIVED from the
    profile's own per-rank extend growth (``D_EXTEND_GROWTH_MIB``, ratio to the recorded rank) and
    from the checkpoint's geometry (``extend_trim.derived_rate_mib``), and never under the profile's
    highest recorded (measured) rate -- the largest of the three; no rig constant is written here. ``fill`` is ``None`` when the record has no null rank (the
    unchanged path, byte for byte), else a dict ``{"sources", "ranks", "geometry", "geo_err",
    "record"}`` (``ranks`` = the ranks the run-time measurement is armed on)."""
    from flliper.srt.pdflip import extend_trim as _et

    rate = list(rate)
    if not any(r is None for r in rate):
        return rate, None
    growth, _ = d_extend_growth_record(getattr(ns, "profile", None))
    geo_err = ""
    try:
        geo = _et.derived_rate_mib(_model_config(str(getattr(ns, "model", "") or "")))
        if geo is None:
            geo_err = "keine hidden_size"
    except Exception as exc:  # noqa: BLE001 -- an unreadable geometry is NAMED, never fatal
        geo, geo_err = None, f"{type(exc).__name__}: {exc}"
    filled, sources = _et.fill_null_rates(rate, growth, geo)
    ranks = [i for i, r in enumerate(rate) if r is None and filled[i] is not None]
    return filled, {"sources": sources, "ranks": ranks, "geometry": geo, "geo_err": geo_err,
                    "record": rate}

def d_early_stage0_watch(gate: str, sized_dir: str, expected_p_ranks: int, d_pid_fn,
                         d_alive_fn, p_alive_fn, vram_fn, names: Mapping[str, str],
                         result: Dict[str, object], log: Log, poll_s: float = 1.0,
                         clock=time.monotonic, sleep=time.sleep) -> None:
    """BOOTZEIT 3 stage 0, launcher side: sample the early D's VRAM per card
    until every P rank reports sized, then write the stage-0 gate. ``go`` only
    if D held 0 MiB on every card meanwhile (else P's used_by_me holds D's
    context: refuse, D exits, the serial path starts D later)."""
    from flliper.srt.pdflip import d_early_start as _des

    held: Dict[str, int] = {}
    t0 = clock()
    while True:
        try:
            now = vram_fn(d_pid_fn())
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            now = {}
            log(f"{_des.STAGE0_MARKER} VRAM sample failed ({type(e).__name__}: {e}); counted as no reading")
        for u, v in now.items():
            held[u] = max(held.get(u, 0), int(v))
        records = _des.read_p_sized(sized_dir)
        verdict, lines = _des.stage0_verdict(records, expected_p_ranks, held, names)
        if verdict is None and not (d_alive_fn() and p_alive_fn()):
            verdict, lines = False, lines + ["a group exited before P was sized"]
        if verdict is not None:
            for ln in lines:
                log(f"{_des.STAGE0_MARKER} {ln}")
            waited = clock() - t0
            _des.write_gate(gate, _des.VERDICT_GO if verdict else _des.VERDICT_REFUSE,
                            "; ".join(lines), measured=held)
            log(f"{_des.STAGE0_MARKER} {'GO' if verdict else 'REFUSE'} after {waited:.1f} s -- "
                + ("D creates its CUDA contexts now" if verdict else "early D exits; D starts serially"))
            result.update({"verdict": _des.VERDICT_GO if verdict else _des.VERDICT_REFUSE,
                           "waited_s": round(waited, 1), "d_held_mib": dict(held),
                           "p_used_by_me_mib": {f"pp{r.get('pp_rank')}tp{r.get('tp_rank')}": r.get("used_by_me_mib")
                                                for r in records}})
            boot_state_write(log, event=("d_early_stage0", dict(result)))
            return
        sleep(poll_s)

def d_window_pool_fields() -> Dict[str, object]:
    """PlanInputs fields of group D's DFLASH window pool: set exactly when
    spec_form_env("D") publishes FLLIPER_DFLASH_WINDOW_POOL=1 and the draft is
    split (the solo host prices its own pool)."""
    if spec_form_is_dflash() and spec_form_env("D").get("FLLIPER_DFLASH_WINDOW_POOL") == "1" \
            and dflash_placement() != "solo":
        return {"dflash_window_pool": True,
                "speculative_draft_window_size": int(_SPEC_FORM["window"])}
    return {}

def calib_log_accept_of(ns) -> Optional[Callable[[str], bool]]:
    """PDFLIP-FORM: the P-log calibration filter of this boot -- logs of THIS
    checkpoint only (xsn417, a Qwen3.8-27B boot, cut its P stages on the
    Next-Flash boot fnFL2x142's prefill census), and of this line where the
    profile's records row names it (UNIFY S3). None when ``ns`` carries no
    resolved form (a desk caller): every log, as before."""
    ident = calib_identity_of(ns)
    return ident.accepts_log if ident is not None else None
