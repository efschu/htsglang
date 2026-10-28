"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Tuple

from flliper.srt.compat_shims import register_flag_aliases as _register_flag_aliases
from flliper.srt.pdflip import DEFAULT_D_BS
from flliper.srt.pdflip import DEFAULT_P_BS
from flliper.srt.pdflip import corridor_budget
from flliper.srt.pdflip import weight_exchange
from flliper.srt.pdflip import weight_exchange_shadow
from flliper.srt.pdflip import weight_exchange_transport
from flliper.srt.pdflip import x_curves as pdflip_x_curves
from flliper.srt.pdflip.launcher_d_apply_r2_L5 import P_MAMBA_SLOTS_PER_RUNNING_REQUEST
from flliper.srt.pdflip.launcher_d_leaf import DFLASH_BLOCK_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import DFLASH_DRAFT_PATH_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import DFLASH_PRODUCE_ON_P_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import DFLASH_WINDOW_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import D_GC_FREEZE_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import D_KV_EVICT_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import D_REPLAYSSM_SPEC_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import D_RESHARD_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import D_TP_OBJECTIVE_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import SPEC_FORM_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import VISION_CHOICES
from flliper.srt.pdflip.launcher_d_leaf import VISION_OFF
from flliper.srt.pdflip.launcher_d_leaf import _csv
from flliper.srt.pdflip.launcher_d_mid import ATTN_ANCHOR_MS
from flliper.srt.pdflip.launcher_d_mid import CONTEXT_LENGTH_TOKENS
from flliper.srt.pdflip.launcher_d_mid import DFLASH_PRODUCE_ENV
from flliper.srt.pdflip.launcher_d_mid import D_TOKEN_PLACEMENT_DEFAULT
from flliper.srt.pdflip.launcher_d_mid import ONCARD_MODE_DEFAULT
from flliper.srt.pdflip.launcher_d_p_apply import P_ATTN_HEAD_SPLIT_DEFAULT
from flliper.srt.pdflip.launcher_d_p_apply import P_ATTN_HEAD_SPLIT_MIN_W_DEFAULT
from flliper.srt.pdflip.launcher_d_p_apply import P_LAYER_SPLIT_PULL_GBPS_DEFAULT
from flliper.srt.pdflip.launcher_d_p_apply import P_PREFILL_GRAPH_SPLIT_DEFAULT
from flliper.srt.pdflip.launcher_d_p_r2_L0 import EVIDENCE_DIR
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PP_ATTN_STAGE_RATIO_SCORES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PP_STAGE_RATIO_SCORES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PREFILL_TRANSIENT_SUPPORT
from flliper.srt.pdflip.launcher_d_p_r2_L0 import SPEC_NUM_DRAFT_TOKENS
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_CENSUS_DRAFT_PAGES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_CENSUS_KV_PAGES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_CENSUS_KV_PAGE_BYTES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_CENSUS_MAMBA_BLOBS
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_CENSUS_PROVENANCE
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_DISK_MIN_FREE_GIB
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_MAMBA_BLOB_BYTES
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_SIDECAR_FACTOR
from flliper.srt.pdflip.launcher_d_set_r3_L0 import P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT
from flliper.srt.pdflip.launcher_d_set_r3_L0 import P_PP_STAGE_FIXED_MIB
from flliper.srt.pdflip.launcher_p_d_r2_L3 import PORT_D
from flliper.srt.pdflip.launcher_p_d_r2_L3 import P_PREFILL_ACTIVATION_RESERVE_MIB
from flliper.srt.pdflip.launcher_p_l3_leaf import P_CHUNK_MAX_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import P_CHUNK_MODEL_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import P_CHUNK_POLICY_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import P_DEEP_SPLIT_FROM_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import P_GC_FREEZE_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import P_LAYER_SPLIT_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import P_PREFILL_GRAPH_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import P_PREFILL_GRAPH_POLICY_DEFAULT
from flliper.srt.pdflip.launcher_p_mid import P_MAMBA_AUTO_SAFETY_MARGIN
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import SPEC_ALGORITHM
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import SPEC_EAGLE_TOPK
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import SPEC_NUM_STEPS
from flliper.srt.pdflip.launcher_refuse_proc_leaf import WEIGHT_SOURCE_CHOICES
from flliper.srt.pdflip.launcher_refuse_proc_leaf import WEIGHT_SOURCE_DEFAULT
import argparse

PORT_FRONT = 30030
# User order 2026-09-10: the front serves on the LAN, permanently. 0.0.0.0
# covers loopback (router, deadmen, agents) AND the LAN address. The groups
# stay on 127.0.0.1 -- only the front is a client surface. Loopback trust in
# utils/auth.py keys on the PEER address, so an admin route reached from the
# LAN still needs the bearer; the API routes need nothing.
DEFAULT_FRONT_HOST = "0.0.0.0"
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
#: The `gapped corridor holdback` post, MiB per rank. 1.000 GiB on every rank
#: of both boots. This is the term ARMING_FLOOR_MIB was standing in for, and
#: they are NOT the same thing: the runtime charges this holdback and no
#: arming floor, so the 1229 MiB stand-in over-charged by 205 MiB per rank.
#: That was the only one of the four errors pointing the SAFE way.
P_CORRIDOR_HOLDBACK_MIB = 1024.0
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
#: DUAL-SHARE --dual-priority modes (pdflip/dual_share.MODES).
DUAL_PRIORITY_MODES = ("p", "balanced", "d", "dynamic")

def _add_args_pin_ledger_arm_s(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--pin-ledger-arm-s`` on."""
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

def _add_args_dual_p_duty(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--dual-p-duty`` on."""
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

def _add_args_extra_p(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--extra-p`` on."""
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

def _add_args_d_kv_token_cut(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--d-kv-token-cut`` on."""
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

def _add_args_p_card_reference_logs(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--p-card-reference-logs`` on."""
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

def _add_args_carrier_max_tokens(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--carrier-max-tokens`` on."""
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

def _add_args_d_replayssm_spec(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--d-replayssm-spec`` on."""
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

def _add_args_d_token_shares(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--d-token-shares`` on."""
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

def _add_args_p_layer_split(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--p-layer-split`` on."""
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

def _add_args_p_chunk_dynamic_min_tokens(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--p-chunk-dynamic-min-tokens`` on."""
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

def _add_args_fp8_uniform_marlin(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--fp8-uniform-marlin`` on."""
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

def _add_args_pdflip_weights_cpu_backup(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--pdflip-weights-cpu-backup`` on."""
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

def _add_args_pdflip_vision(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--pdflip-vision`` on."""
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

def _add_args_pdflip_p_free_read_journal(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--pdflip-p-free-read-journal`` on."""
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

def _add_args_pp_solve_objective(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--pp-solve-objective`` on."""
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

def _add_args_no_p_host_overlap(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--no-p-host-overlap`` on."""
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

def _add_args_pp_cut_stage_fit(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--pp-cut-stage-fit`` on."""
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

def _add_args_pp_cut_mamba_mib_per_linear_layer_per_slot(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--pp-cut-mamba-mib-per-linear-layer-per-slot`` on."""
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

def _add_args_num_continuous_decode_steps(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--num-continuous-decode-steps`` on."""
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

def _add_args_d_tp_objective(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--d-tp-objective`` on."""
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

def _add_args_d_uneven_token_vector_role(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--d-uneven-token-vector-role`` on."""
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

def _add_args_no_hicache_flush_publish_sweep(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--no-hicache-flush-publish-sweep`` on."""
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
