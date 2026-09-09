"""Weg-2 group launcher (spec section 1.5, record sections 1c-1g).

Two STOCK ``python -m sglang.launch_server`` launches per rig, sequenced so
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
* SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 for both groups (K2): /health
  is a pure liveness probe; the front's /health_generate reaches the AWAKE
  group only.
* the deadmen of the two groups run tier 1 only (PROBE_S spaced past the
  boot); tier 2 (/health_generate) runs on the FRONT's deadman, which the
  front routes to the awake group.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from sglang.srt.weg2 import admin_key as admin_key_mod
from sglang.srt.registry import nvml as nvml_registry
from sglang.srt.weg2 import host_ledger, ring_table

MIB = 1024 * 1024
PORT_FRONT = 30030
PORT_P = 30031
PORT_D = 30032
EVIDENCE_DIR = "/spinning/evidence-665-f1"
GPU_ARB = "/spinning/gpu-arb"
#: The step-0 metal probe's RECORD (C0, WEG2_BUILD_DECISIONS_0906 section 1p).
#: A FILE, not a number: the per-card duplex ratios C12/C13 gate on are parsed
#: out of its own measured rows and printed with this path beside them.
DUPLEX_PROBE_DEFAULT = f"{GPU_ARB}/weg2/PROBE_RING_0907.md"
DEADMAN = f"{GPU_ARB}/devtools/boot_deadman.sh"
MEMTS = f"{GPU_ARB}/devtools/mem_timeseries.sh"
HOST_PREFLIGHT = f"{GPU_ARB}/devtools/host_ledger_preflight.sh"
SHM_DIR = "/dev/shm"
PRESENCE_DIR = f"{SHM_DIR}/sglang-phase-flip-presence"
STORE_MOUNT = "/spinning/hicache-weg2-ram"
#: C18: where the per-card host-ring files live under the MAP_SHARED form.  A
#: tmpfs, because the granules must be shared PAGES (both co-located rank
#: processes map the same file), not a disk-backed file.
HOST_RING_DIR = "/dev/shm/weg2-hostring"
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
    "sglang-phase-flip-presence",   # managers/phase_flip_presence.py, #1217
    "hicache-weg2-",                # the canonical page store when it is put on /dev/shm
    ".weg2-pcie-serialize-",        # model_loader/hibernate.py:504
    "sglang_loads_",                # managers/load_snapshot.py:285
)
#: The corridor law is 819-1229 MiB NVML-free per card under the awake
#: group's load.  MEASURED 2026-09-07 boot weg2onebackup2 with this constant
#: at 1024: the 5090's continuous minimum under group D was 620-684 MiB
#: (P log "CORRIDOR LAW BREACHED", 4 samples) -- the awake group lands
#: ~400 MiB below the budget line (CUDA context + BAR1 windows sit outside
#: the --rank-gpu-memory-mib fraction).  The measured overshoot is charged
#: here so the minimum lands mid-band; R5 grades the result.
CORRIDOR_MIB = 1024 + 404
#: Spec section 1.6 V1 (arm B, graphs resident) derived upper bounds for the
#: dormant residue of a rank: 5090 1,848 MiB, 3080 1,442 MiB -- EXPECTATIONS
#: (record 1d/1f B6), printed beside the measurement, used only for group P's
#: budget (D has not slept yet when P is sized) and graded by W19 at D's
#: first sleep by the front.  Plus D's open BAR1 windows (deviation above).
DC_EXPECT_5090_MIB = 1848
DC_EXPECT_3080_MIB = 1442
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
DC_MEASURED_D_5090_MIB = 2228
DC_MEASURED_D_3080_MIB = 1922
DC_RESERVE_SLACK_MIB = 64
#: #1234 C6 -- the DEVELOPMENT transport switch, and the one number it moves.
#:
#: The user's order of 2026-09-07 put development on NCCL until barlink
#: covers the Weg-2 message classes again. NCCL is a LAUNCHER-level mode, not
#: a tier inside barlink: a barlink-owned group never constructs a PyNccl
#: communicator (parallel_state.should_build_pynccl), and building one would
#: change the flip path, which is out of bounds. Dropping --barlink instead
#: leaves the stock sglang dispatch in place, which does build one.
#:
#: MEASURED (A/B boot weg2ab0, ARM 0): D's dormant residue on the 5090 rises
#: 2230 -> 2310 MiB under NCCL, because libnccl's buffers are not
#: memory-saver-tagged and therefore survive the sleep. That tripped W19
#: DormantResidueRefused against DC_MEASURED_D_5090_MIB 2228 +
#: DC_RESERVE_SLACK_MIB 64 = 2292 and killed the run. The extra slack lives
#: HERE, behind the switch, and not in the constant -- under 'bar1' nothing
#: about the flip path changes, which is the entire point.
DC_RESERVE_SLACK_NCCL_MIB = 192
#: Flags that only make sense while barlink owns the group's collectives.
#: Each takes a value except the bare --barlink itself.
BARLINK_FLAGS_WITH_VALUE = (
    "--barlink-transport",
    "--barlink-bar1-window-mib",
    "--barlink-bar1-cap-cycles",
    "--barlink-uncovered-class",
)
BARLINK_FLAGS_BARE = ("--barlink",)
#: Environment keys the bar1 build path owns; dropped with the flags so the
#: NCCL arm is not half-configured for a transport it does not run.
BARLINK_ENV_KEYS = ("SGLANG_BARLINK_BUILD_WINDOW_CAP_S",)


def transport_argv(argv: List[str], transport: str) -> List[str]:
    """``argv`` as the chosen transport needs it. One seam, both groups."""
    return argv if str(transport) != "nccl" else strip_barlink_flags(argv)


def reserve_slack_mib(transport: str) -> int:
    """Dormant-residue slack for group D under this transport, in MiB.

    ONE definition, so the NCCL arm's +128 MiB cannot drift away from the
    reason it exists. Under 'bar1' this is the unchanged
    :data:`DC_RESERVE_SLACK_MIB`, and the flip path's arithmetic is exactly
    what it was.
    """
    return (DC_RESERVE_SLACK_NCCL_MIB if str(transport) == "nccl"
            else DC_RESERVE_SLACK_MIB)


def strip_barlink_flags(argv: List[str]) -> List[str]:
    """``argv`` without the barlink family -- the NCCL development mode.

    Removal, not substitution: without --barlink the group takes the stock
    sglang dispatch, which builds the PyNccl communicator barlink suppresses.
    Written as a filter over the ONE argv builder rather than as a second
    argv builder, so the two arms cannot drift apart in anything except the
    transport.
    """
    out: List[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in BARLINK_FLAGS_WITH_VALUE:
            skip = True
            continue
        if token in BARLINK_FLAGS_BARE:
            continue
        out.append(token)
    return out
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
P_OVERSHOOT_MIB = [920, 0, 512]
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
D_OVERSHOOT_MIB = [489, 0, 0]
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
RING_FORM_SENTINEL_STORE_GIB = -1.0
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
P_DRAFT_RESIDENT_BUDGET_MIB = 405.2 + 1213.0
P_DRAFT_RESIDENT_TOL_MIB = 256.0
P_WEG2ZR2_LAST_STAGE_NVML_FREE_MIN_MIB = 1450.0
P_WEG2ZR2_LAST_STAGE_KV_POOL_MIB = 5584.4
P_CORRIDOR_TOP_MIB = 1229.0
P_BYTES_PER_TOKEN = 8192 + 2048
#: Group P's per-stage CAPABILITY SCORES, in stage order (ordinal 0 = the
#: 5090) -- what ``--pp-stage-ratio`` takes.  NOT a layer split: fix 5, after
#: the map below was found to restate this vector as one.  ``server_args``
#: hands these to ``derive_pp_layer_split(scores, is_full_attention=kinds,
#: attn_scores=...)`` and its hybrid snap decides the real per-stage layer
#: counts, which agree with the scores for exactly the current (checkpoint,
#: 32/18/14, 8/4/4) triple and diverge for its neighbours -- measured:
#: attn_scores 7,5,4 -> [31,19,14]; scores 31,17,16 -> [32,16,16]; scores
#: 30,20,14 -> [32,18,14].  Anything that needs the SPLIT calls
#: :func:`p_stage_layers`; nothing reads these as layer counts.
#:
#: FIX 2 (#1233 fix 5, second instance).  WHICH cut this vector is, stated so
#: no third consumer can adopt it for the wrong one: it is the INCUMBENT --
#: the cut ``MEASURED_MS_PER_LAYER`` was taken under on boot bsscale (PP0
#: 259.1 ms/32 layers, PP1 632.9/18, PP2 470.2/14), hence the cost model's
#: incumbent in :func:`solve_p_cut` and the default ``argv_p`` renders when
#: nobody solved or pinned one.  It is NOT the cut the boot runs: since
#: ``solve_p_cut`` was wired into ``main`` that is ``PCutFacts.stage_ratio``
#: and its round-tripped ``PCutFacts.layer_counts``, and every consumer that
#: describes THIS boot -- argv, the flip-order map, the provenance lines --
#: reads those.  Exactly three functions may name this vector: ``argv_p``
#: (the flag default), ``solve_p_cut`` (the incumbent) and
#: :func:`p_stage_layers` (what a score pair derives to); the registered
#: guard in test_weg2_host_budget_1233 pins that set.
P_PP_STAGE_RATIO_SCORES = (32, 18, 14)
#: Group P's per-stage FULL-ATTENTION scores (#485 ``--pp-attn-stage-ratio``),
#: the incumbent's other half, defined once for the same three readers.
P_PP_ATTN_STAGE_RATIO_SCORES = (8, 4, 4)


def _csv(values: Sequence[int]) -> str:
    """One vector, one rendering: the CSV form the two count flags take.

    FIX 2.  ``argv_p`` used to carry ``"32,18,14"`` / ``"8,4,4"`` as bare
    default arguments -- a THIRD statement of the constants above, which the
    #1233 fix 5 one-definition property had already closed once and which
    ``test_neither_score_vector_survives_anywhere_as_a_bare_literal`` was
    written to catch.  It caught it; the reading was wrong, not the guard.
    """
    return ",".join(str(int(v)) for v in values)


#: The incumbent cut RENDERED for help text, DERIVED from the one definition
#: above rather than retyped.  It exists so ``build_parser`` can name the
#: numbers without becoming a fourth reader of the score constants -- the
#: registered guard in ``test_weg2_host_budget_1233`` pins that reader set to
#: ``{argv_p, p_stage_layers, solve_p_cut}`` and walks FUNCTION bodies, so a
#: module-level derivation is the honest way to keep both properties: one
#: statement of the vector, and a help text that prints it.
P_PP_INCUMBENT_FMT = "%s / attention %s" % (
    _csv(P_PP_STAGE_RATIO_SCORES),
    _csv(P_PP_ATTN_STAGE_RATIO_SCORES),
)


def derive_p_max_total_tokens() -> int:
    pool_new = (P_WEG2ZR2_LAST_STAGE_NVML_FREE_MIN_MIB + P_WEG2ZR2_LAST_STAGE_KV_POOL_MIB
                - P_DRAFT_RESIDENT_BUDGET_MIB - P_CORRIDOR_TOP_MIB)
    return int(pool_new * 2**20 / P_BYTES_PER_TOKEN) // 1000 * 1000


P_MAX_TOTAL_TOKENS = derive_p_max_total_tokens()
assert P_MAX_TOTAL_TOKENS == 428000, P_MAX_TOTAL_TOKENS
#: #1264 ``--draft-kv-on-p``.  THE WHOLE PRODUCER FORM OF GROUP P, in one
#: place, so that ``on`` and ``off`` differ by the presence of ONE list and
#: not by five ``if``s scattered through ``argv_p``.
#:
#: The four speculative flags are group D's, BYTE-FOR-BYTE (they hash into the
#: drafter identity, W5); ``--speculative-draft-kv-only`` is the silencer that
#: turns that head into a draft-KV PRODUCER instead of a verifier.
#: ``--max-total-tokens`` is in the SAME list and not beside it, because its
#: derivation (:func:`derive_p_max_total_tokens`) subtracts
#: :data:`P_DRAFT_RESIDENT_BUDGET_MIB` -- the MTP head plus the resident
#: embedding -- from the last stage's pool.  With no head on that stage the
#: subtrahend is zero and the number is not merely unnecessary but WRONG (it
#: would hand back 1618 MiB of pool the boot could have used), and the rg6
#: baseline carried no such flag at all.  A term whose premise is the head
#: belongs to the head.
#: #1241 (refuter MF-6). THE BOOT'S SPEC AND KV FACTS, ONE WRITER.
#:
#: These five values were literals in FIVE places: the P draft-KV flag tuple,
#: ``common_flags``, ``argv_d``, and BOTH ``PlanInputs`` blocks that price a
#: weight vector. The argv half and the pricing half agreed by coincidence of
#: typing, not by construction -- so a change to the shipped draft-token count
#: (which #1242 is already moving) would have gone on pricing the OLD
#: configuration, silently, on every boot including the default one. That is a
#: second bookkeeping of the boot's own argv, which is the shape
#: UPSTREAM-MINIMAL forbids. One name each, read by everybody.
SPEC_ALGORITHM = "NEXTN"
SPEC_NUM_STEPS = 2
SPEC_EAGLE_TOPK = 1
SPEC_NUM_DRAFT_TOKENS = 3
KV_CACHE_DTYPE = "fp8_e4m3"

P_DRAFT_KV_FLAGS: Tuple[str, ...] = (
    "--speculative-algorithm", SPEC_ALGORITHM,
    "--speculative-num-steps", str(SPEC_NUM_STEPS),
    "--speculative-eagle-topk", str(SPEC_EAGLE_TOPK),
    "--speculative-num-draft-tokens", str(SPEC_NUM_DRAFT_TOKENS),
    "--speculative-draft-kv-only",
    "--max-total-tokens", str(P_MAX_TOTAL_TOKENS),
)
#: The standing user order of 2026-09-07 is draft KV ACROSS THE FLIP, so the
#: producer is the default.  ``off`` is the serving-base / A-B form and is
#: never silent -- see :func:`draft_kv_off_line`.
DRAFT_KV_ON_P_DEFAULT = "on"


def draft_kv_off_line() -> str:
    """The one line the launcher prints when the producer is switched OFF.

    An A-B arm that reads like the default is how a measurement gets
    attributed to the wrong tree.  This says, in the boot's own log, exactly
    which of the two forms ran and what it costs.
    """
    return (
        "WEG2 DRAFT-KV-ON-P: off -- group P boots without the MTP head; "
        "D reads KV+Mamba from the store, draft state is cold after every "
        "flip (user order 2026-09-07 stands; this form is the rg6 baseline)"
    )
VENV_DEFAULT = "/spinning/htsglang-gpu/.venv"
#: The model context both groups are launched with.  Named once because K9's
#: --max-kv-per-request default IS this number (the as-built cap, decoupled
#: from the model context so it can be lowered without touching the context).
CONTEXT_LENGTH_TOKENS = 262144
#: D's prefill chunk width.  Named once because it is X's FLOOR (K5): a bound
#: below one chunk would refuse work D must be able to do to make progress.
CHUNKED_PREFILL_TOKENS = 4096
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
X_RECORDED_R_D_TOKS = 690.0
X_RECORDED_R_P_TOKS = 3640.0
X_RECORDED_FLIP_S = 13.247


#: #1271: the UNIT a rate is expressed in. X* divides one rate by another, so
#: two rates of different units produce a number with no meaning -- and that is
#: not hypothetical, it is what shipped.
#:
#: ``group_throughput``  -- tokens the GROUP moved / the wall it was busy.
#: ``request_latency``   -- one request's tokens / that request's own wall,
#:                          which under concurrency includes the time it spent
#:                          queued behind its peers and is therefore NOT a rate
#:                          of the group.
#: ``compute_honest``    -- tokens / rank gpu-ms (compute+wait), the per-rank
#:                          instrument both groups carry.
RATE_UNIT_GROUP_THROUGHPUT = "group_throughput"
RATE_UNIT_REQUEST_LATENCY = "request_latency"
RATE_UNIT_COMPUTE_HONEST = "compute_honest"


class MixedRateUnits(ValueError):
    """X* was asked for from two rates that are not the same measurement."""


def derive_x_star(flip_s: float, r_d: float, r_p: float, floor_tokens: int,
                  *, unit_d: str = RATE_UNIT_GROUP_THROUGHPUT,
                  unit_p: str = RATE_UNIT_GROUP_THROUGHPUT) -> int:
    """X* = 2*flip_s / (1/r_D - 1/r_P), floored (spec section 0, K5, A1-4).

    THE BREAK-EVEN of the round trip: below X* it is cheaper for D to
    prefill the request itself than to pay two flips so P can do it faster.
    ONE number, TWO consumers -- the per-request bound of law 4 and C7's
    aggregate departure latch -- so there is no second hand number.

    Refuses (ValueError) rather than guessing when the inputs cannot
    produce a break-even: a D that is not slower than P has none.
    """
    # #1271 (a): REFUSE MIXED UNITS BY NAME, before any arithmetic.
    #
    # THE SHIPPED DEFECT, measured. `r_P` was
    # `(prompt_tokens - cached_tokens) / wall` off the front's own
    # `WEG2-SERVED group=P leg=1` line (launcher.py:445-447, appended at
    # :485-489) -- a PER-REQUEST wall, taken while the front runs the drain at
    # `p_concurrency=8`, so it counts the time a leg sat behind its peers.
    # `r_D` came from `verdict=single_prefill` legs, i.e. concurrency ONE.
    # The formula therefore divided a concurrent-latency rate by a
    # single-request rate and called the quotient a break-even.
    #
    # What that cost: on sb1's front log the SAME boot yields r_P = 1964 tok/s
    # under the shipped `unc >= 4096` filter (23 samples), 528 tok/s over all
    # 79 legs, and ~3.1k tok/s as drain-aggregate. rg6 gave 3096 on 8 samples.
    # r_P did not halve between boots -- the SAMPLE changed: the filter admits
    # only large-uncached legs, which are the cold, least-queued ones, and as
    # more of a drain's later legs qualify the median falls. P's compute-honest
    # rate was ~6594-8325 tok/s across both boots, i.e. flat.
    if unit_d != unit_p:
        raise MixedRateUnits(
            f"X* would divide r_D measured as {unit_d!r} by r_P measured as "
            f"{unit_p!r}. These are not the same quantity: a request_latency "
            f"rate under concurrency counts queueing behind peers, a "
            f"group_throughput rate does not, and a compute_honest rate counts "
            f"neither. 1/r_D - 1/r_P is only a time-per-token difference when "
            f"both sides are the SAME measurement. Refusing rather than "
            f"returning a number with no unit (#1271)."
        )
    if flip_s <= 0 or r_d <= 0 or r_p <= 0:
        raise ValueError(f"non-positive input: flip_s={flip_s} r_d={r_d} r_p={r_p}")
    denom = (1.0 / r_d) - (1.0 / r_p)
    if denom <= 0:
        raise ValueError(
            f"r_D {r_d:.0f} >= r_P {r_p:.0f}: no break-even exists, the round trip never pays"
        )
    return max(int(floor_tokens), int(2.0 * flip_s / denom))


_RE_FLIP = re.compile(r"flip_total=(\d+) ms")
_RE_LEG1 = re.compile(
    r"WEG2-SERVED group=P leg=1 .*?prompt_tokens=(\d+) cached_tokens=(\d+) wall=([0-9.]+)s"
)
#: #1271: the front's own P-drain window -- `prefilled` legs over `drain_s`
#: seconds. This is the denominator r_P was always meant to have.
_RE_DRAIN = re.compile(r"WEG2 P-DRAIN epoch=(\d+) .*?drain_s=([0-9.]+)")
_RE_LEG2 = re.compile(
    r"WEG2-SERVED group=D leg=2 .*?uncached=(\d+) verdict=(\S+) wall=([0-9.]+)s"
)


def _median(xs: List[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def measure_x_inputs(log_path: str, floor_tokens: int) -> Optional[Tuple[float, float, float, int, int, int]]:
    """(flip_s, r_D, r_P, n_flips, n_leg2, n_leg1) from ONE front log, or None.

    Reads the front's OWN instruments -- ``flip_total=`` and the two
    ``WEG2-SERVED`` lines -- so every input to X is a measurement this rig
    printed, not a table.  Only leg-2 lines that actually prefilled on D
    (``verdict=single_prefill|short_mispriced``) and only extents of at
    least one chunk count, because a rate taken over a few hundred tokens
    prices fixed overhead, not throughput.
    """
    flips: List[float] = []
    r_d: List[float] = []
    r_p: List[float] = []
    drain_tokens = 0
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                m = _RE_FLIP.search(line)
                if m:
                    flips.append(int(m.group(1)) / 1000.0)
                    continue
                m = _RE_LEG2.search(line)
                if m:
                    unc, verdict, wall = int(m.group(1)), m.group(2), float(m.group(3))
                    if verdict in ("single_prefill", "short_mispriced") and unc >= floor_tokens and wall > 0:
                        r_d.append(unc / wall)
                    continue
                # #1271 (a): r_P IS A GROUP THROUGHPUT, NOT A REQUEST LATENCY.
                # The leg-1 wall is per-request and the front drains P at
                # `p_concurrency=8`, so `unc/wall` charges one leg for the time
                # it spent queued behind its peers. Accumulate the drain's
                # uncached tokens instead and divide by the drain's OWN wall,
                # which the front already prints -- the same shape as r_D
                # (tokens a group moved / the wall it was busy).
                m = _RE_LEG1.search(line)
                if m:
                    unc = int(m.group(1)) - int(m.group(2))
                    if unc > 0:
                        drain_tokens += unc
                    continue
                m = _RE_DRAIN.search(line)
                if m:
                    drain_s = float(m.group(2))
                    if drain_tokens > 0 and drain_s > 0:
                        r_p.append(drain_tokens / drain_s)
                    drain_tokens = 0
    except OSError:
        return None
    if not (flips and r_d and r_p):
        return None
    return (_median(flips), _median(r_d), _median(r_p), len(flips), len(r_d), len(r_p))


class XSeed(NamedTuple):
    """X, its provenance line, and whether a MEASUREMENT produced it.

    ``measured`` is load-bearing, not decoration.  The carrier floor is 1.25x X
    (:func:`carrier_census.route_floor`), so an X this rig never measured
    decides through W45 whether a boot may start at all -- a fallback acting as
    an actuator, which the standing law forbids.  Carried as a field rather
    than sniffed back out of ``provenance`` so the two cannot drift (#1299).
    """

    tokens: int
    provenance: str
    measured: bool


def resolve_x(override: Optional[int], evidence_dir: str, floor_tokens: int) -> XSeed:
    """X and its PROVENANCE LINE, naming the three inputs and their source.

    Order: an explicit ``--tp-prefill-max-tokens`` wins (A1-4: derived with
    the flag as override), else the newest front log on this rig that
    carries all three instruments, else the recorded PRE-BARLINK pair.

    #1299: THE SCAN READS EVERY LOG UNTIL ONE QUALIFIES -- it used to stop
    after the newest EIGHT, and a log that parsed to ``None`` consumed a slot
    exactly like one that measured.  Measured consequence, four boots of one
    day: dec1 (07:32Z) and both sb5h (10:25Z, 10:31Z) found
    ``weg2sb5f...front.log`` at scan positions 1 and 5 and seeded X=8,742 from
    it; by dec2b (12:50Z) and shadow boot B (12:56Z) the same file had drifted
    to positions 8 and 9 behind eight barren logs -- refused or decode-only
    boots that print a front log but never drain P -- and both fell through to
    the recorded pair at X=22,556.  The seed tripled on two INDEPENDENT lines
    at the same wall-clock while the file it should have read sat unchanged on
    disk, because the cap counts files examined, not files that qualified.

    The two ``.P.log`` scans in this same launcher
    (:func:`newest_p_log_with_bubble` and the mean-prefix scan beside it)
    already walk their whole sorted list and skip what does not carry the line;
    this is that shape, and it deletes a hand number rather than adding one.

    Both provenance lines now carry the scan's own denominator: how many logs
    were skipped before the seed, or how many were examined before the
    fallback.  Without it "no front log on this rig carried all three
    instruments" reads as a fact about the rig when it was only a fact about
    the eight the scan looked at.
    """
    if override is not None:
        return XSeed(max(floor_tokens, int(override)), (
            f"X={max(floor_tokens, int(override))} source=flag "
            f"(--tp-prefill-max-tokens, operator override; floor {floor_tokens})"
        ), True)
    logs: List[str] = []
    try:
        logs = sorted(
            (os.path.join(evidence_dir, n) for n in os.listdir(evidence_dir) if n.endswith(".front.log")),
            key=lambda q: os.path.getmtime(q),
            reverse=True,
        )
    except OSError:
        logs = []
    skipped = 0
    for path in logs:
        got = measure_x_inputs(path, floor_tokens)
        if got is None:
            skipped += 1
            continue
        flip_s, r_d, r_p, n_f, n_d, n_p = got
        try:
            x = derive_x_star(flip_s, r_d, r_p, floor_tokens)
        except ValueError:
            skipped += 1
            continue
        return XSeed(x, (
            f"X={x} source=boot:{os.path.basename(path)} "
            f"X*=2*flip_s/(1/r_D-1/r_P) flip_s={flip_s:.2f} (median of {n_f}) "
            f"r_D={r_d:.0f} tok/s (median of {n_d} D single prefills) "
            f"r_P={r_p:.0f} tok/s (median of {n_p} P leg 1s) floor={floor_tokens}; "
            f"skipped {skipped} newer front log(s) carrying no complete instrument set"
        ), True)
    x = derive_x_star(X_RECORDED_FLIP_S, X_RECORDED_R_D_TOKS, X_RECORDED_R_P_TOKS, floor_tokens)
    return XSeed(x, (
        f"X={x} source=recorded PRE-BARLINK (examined {len(logs)} front log(s) on this rig, "
        f"none carried all three instruments) X*=2*flip_s/(1/r_D-1/r_P) flip_s={X_RECORDED_FLIP_S} "
        f"r_D={X_RECORDED_R_D_TOKS:.0f} tok/s r_P={X_RECORDED_R_P_TOKS:.0f} tok/s "
        f"floor={floor_tokens} -- record 1l/1o weg2zr2, conservative corner of the measured "
        f"range; post-barlink r_D 1354 tok/s pushes X far higher and post-flipcost pulls it "
        f"back, so this is a fallback, never a table (O4)"
    ), False)

#: The operating point both groups are launched at (`--context-length`), and
#: therefore the longest prompt group P can be asked to prefill. It is the
#: default floor for P's KV pool: P frees a request's rows once its prefill
#: completes, but DURING that prefill the whole prefix must be device-resident
#: for the stage's attention layers, so a pool below this cannot serve the
#: boot's own admitted maximum.
CONTEXT_LENGTH = CONTEXT_LENGTH_TOKENS

#: MEASURED per-stage prefill cost, ms per layer, boot `bsscale`
#: (/spinning/gpu-arb/weg2/BSSCALE_0907.md, tip 37c884b0b0, table under
#: "The headline of Table A"): per full 4096-token chunk at bs6 PP0 259.1 ms
#: over 32 layers = 8.10, PP1 632.9 over 18 = 35.16, PP2 470.2 over 14 =
#: 33.59, with stage 0 on the 5090 and stages 1-2 on the two 3080s. The two
#: 3080 figures differ by 4.7 %, which is inside the +-10 % per-rank spread
#: that measurement's own A/A repeat established. Used as the ANCHOR for the
#: card-rate library's ratios, and as the whole cost model when no measured
#: library exists.
MEASURED_MS_PER_LAYER = "8.10,35.16,33.59"

#: Per-rank arming floor for the pool model, MiB. The rig's VRAM corridor is
#: 819-1229 MiB NVML-free per card under load and the desk pre-flight arming
#: floor is <= 1229; the ceiling is taken because a floor that under-charges
#: inflates the pool, which is the unsafe direction for a capacity FLOOR.
#: pp_cut.PhasePoolModel names the gap this stands in for: the per-layout
#: solved floor exists only for a layout that has booted.
ARMING_FLOOR_MIB = 1229.0

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

#: DEEP ANCHOR. The user's physics note of 2026-09-07 (recorded verbatim in
#: the GAPPED block of WEG2_BUILD_DECISIONS_0906.md section 1r): at deep
#: prefixes one full-attention layer costs a 3080 about 0.4 s per chunk at a
#: 262,144-token prefix. That is the second depth the family split needs; it
#: is what makes the optimum a FUNCTION of the design prefix rather than a
#: constant, and it is cited to its record rather than written as a literal.
ATTN_ANCHOR_MS = 400.0
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

#: USER ORDER 2026-09-09, verbatim: "der decode bs6 soll mit bs6 (nicht mehr
#: bs4) der standard werden". Group D's default objective moves from the
#: capacity-first `maxkv` split to the bs=6 OPERATING POINT.
#:
#: WHAT THE ORDER TRADES, from dec2c's own A/B rather than from an argument:
#: the bs=6 vector [42,11,11] measured **+10.3 % at bs4 for -1.4 % pool**. So
#: the maxkv law is not repealed -- it is outranked at this one position,
#: because the ranking a boot should follow is the ranking AT THE OPERATING
#: POINT IT RUNS, and D runs decode.
#:
#: `maxkv` stays fully selectable via --d-tp-objective and its argv is
#: byte-identical to the pre-#1241 form; only which arm you get by saying
#: nothing has changed. --d-bs is NOT touched here (separate branch).
D_TP_OBJECTIVE_DEFAULT = "decode-bs6"

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


@dataclass(frozen=True)
class EarlyReadFact:
    """One fact group D states TWICE, and the reader that makes both necessary.

    #1235's inventory called these "double statements of one fact" and read
    them as second bookkeeping to delete. THEY ARE NOT, and the ordering is the
    proof: ``ServerArgs.__post_init__`` reads ``os.environ`` for these keys in
    ``_handle_dcp_validation`` (server_args.py:9631), ``_handle_uneven_tp``
    (server_args.py:12025) and ``uneven_perf.apply_auto_performance``
    (uneven_perf.py:4648, reached FROM _handle_uneven_tp), while the FLAG only
    publishes itself into the environment later, in
    ``_handle_environment_variables`` -> ``_publish_promoted_781_flags``
    (server_args.py:18271), called at server_args.py:7186 -- after all three.

    So the environment variable is the statement that reaches those readers and
    the flag is the statement that reaches everything downstream plus the argv
    a human reads. Dropping either one changes behaviour or hides it.

    What was WRONG is that they were two independent literals in two functions
    that could drift apart silently. Here they are one row, consumed by both
    ``build_env`` and the argv builders, so the pair cannot disagree.

    ``groups`` records an EXISTING asymmetry rather than quietly fixing it:
    the uneven-DCP keys are exported to BOTH groups' environments while only
    group D carries the flags. Group P runs tp_size=1, where the weighted-DCP
    path has nothing to weight, and no boot has ever been run with the export
    removed -- so it is preserved exactly and named here for the boot that can
    measure it, instead of being changed by a slice that cannot.
    """

    env_key: str
    env_value: str
    flag: Tuple[str, ...]
    groups: str
    reader: str


#: THE THREE FACTS, ONCE. See :class:`EarlyReadFact` for why each is stated in
#: both currencies and why that is not second bookkeeping.
EARLY_READ_FACTS: Tuple[EarlyReadFact, ...] = (
    EarlyReadFact(
        env_key="SGLANG_UNEVEN_DCP",
        env_value="1",
        flag=("--uneven-dcp",),
        groups="D",
        reader="server_args.py:9631 _handle_dcp_validation / :12025 "
               "_handle_uneven_tp, both before :7186 _handle_environment_variables",
    ),
    EarlyReadFact(
        env_key="SGLANG_UNEVEN_DCP_WEIGHTED",
        env_value="1",
        flag=("--uneven-dcp-weighted",),
        groups="D",
        reader="server_args.py:9632 _handle_dcp_validation / :10517 "
               "uneven_weighted_dcp_enabled, both before :7186",
    ),
    EarlyReadFact(
        env_key="SGLANG_MAMBA_SSM_DTYPE",
        env_value="bfloat16",
        flag=("--mamba-ssm-dtype", "bfloat16"),
        groups="both",
        reader="uneven_perf.py:4648, reached from _handle_uneven_tp at "
               "server_args.py:6976 -- before the flag publishes itself at :7186",
    ),
)


def early_read_flags(group: str) -> List[str]:
    """The argv half of :data:`EARLY_READ_FACTS` for one group."""
    out: List[str] = []
    for fact in EARLY_READ_FACTS:
        if fact.groups in ("both", group):
            out.extend(fact.flag)
    return out


def early_read_provenance() -> str:
    """One line naming every doubly-stated fact and the reader that needs it."""
    return "WEG2 EARLY-READ ENV (#1235): " + "; ".join(
        "%s=%s == %s [groups %s, pre-promotion reader %s]"
        % (f.env_key, f.env_value, " ".join(f.flag), f.groups, f.reader)
        for f in EARLY_READ_FACTS
    )


#: #1235 THE UNOWNED ENV LITERALS, PROMOTED TO FLAGS AND TO LAUNCHER OUTPUT.
#:
#: All four used to be written as ``env.get(KEY, "<literal>")``, which is the
#: shape the ring family was already fixed out of (R19): an inherited value
#: from the launcher's own shell wins SILENTLY, and the boot then runs a number
#: no flag, no record and no log line ever mentioned. They are now flags with
#: defaults, written unconditionally, exactly like TMS_HOST_RING_* and
#: SGLANG_WEG2_PCIE_DUPLEX -- launcher OUTPUT, never operator input.
#:
#: Their VALUES are unchanged and none of them is measured; they are timeouts
#: and a census stride, and the help text says so rather than implying a
#: provenance they do not have.
BARLINK_BUILD_WINDOW_CAP_S = 60
PP_CHAIN_RECV_STALL_S = 60
PP_OCCUPANT_HORIZON_S = 90
MATCH_REFUSAL_CENSUS_EVERY = 64

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
#: Group P's BAR1 windows. The provenance is the SAME arithmetic group D's
#: window carries (#1234 C1) and it lived only in state.deviations: the two
#: groups' windows are sized to fit TOGETHER -- P 24+96 and D 16+32+40 = 208 of
#: the 224 MiB usable per 3080, measured Used 224/256 including the RM
#: carve-out. It is cited here so P's window is not the one number in this file
#: a reader has to go looking for.
P_BARLINK_BAR1_WINDOW_MIB = "24,PP_0=96"


class Weg2LaunchRefused(RuntimeError):
    pass


@dataclass
class Card:
    nvml_index: int
    uuid: str
    name: str
    total_mib: int
    #: Bytes the driver holds back out of ``total_mib`` and never hands to any
    #: allocation (425 MiB on this rig's 3080s, 518 on the 5090).  Carried so
    #: the free/used readers below can be carve-out honest.  NOTE: ``total_mib``
    #: is still the full board and the budget arithmetic still spends against
    #: it -- that the awake budget omits this term is a SEPARATE open finding
    #: (weg2/refute/lens2.md sec 3), deliberately not changed here.
    reserved_mib: int = 0


@dataclass
class GroupSpec:
    name: str
    port: int
    argv: List[str]
    log: str
    env: Dict[str, str]
    pid: int = 0
    proc: Optional[subprocess.Popen] = None


@dataclass
class BootState:
    tag: str
    tip: str
    tree: str
    stamp: str
    cards: List[Dict] = field(default_factory=list)
    cvd: str = ""
    logs: Dict[str, str] = field(default_factory=dict)
    pids: Dict[str, int] = field(default_factory=dict)
    helper_pids: List[int] = field(default_factory=list)
    store_mount: str = STORE_MOUNT
    store_gib: float = 0.0
    budgets: Dict[str, List[int]] = field(default_factory=dict)
    dc_measured_p: Dict[str, int] = field(default_factory=dict)
    dc_expect_d: Dict[str, int] = field(default_factory=dict)
    ledger_lines: List[str] = field(default_factory=list)
    ring_lines: List[str] = field(default_factory=list)
    ring_form: str = ""
    ring_dir: str = ""
    #: The boot nonce (``HostRingPlan.epoch``), so teardown can unlink THIS
    #: boot's VRAM credit counters and no other boot's (FIX 3 round 3).
    ring_epoch: str = ""
    #: #1275 fix 2: this boot's admin key FILE (never the key), so `--teardown`
    #: can unlink it. A per-boot secret that outlives its boot is a stale file
    #: naming a server that no longer exists.
    admin_key_file: str = ""
    argv: Dict[str, str] = field(default_factory=dict)
    t_ready: Dict[str, float] = field(default_factory=dict)
    sleep_p_ms: float = 0.0
    deviations: List[str] = field(default_factory=list)
    carrier_max_tokens: int = 0
    weight_chunks: int = 0
    tms_so: str = ""
    p_depth: int = 0
    #: #1233 fix 5: the pre-boot cgroup sample, oom_kill baseline included, so
    #: a later death can be attributed by diff instead of retroactively.
    cgroup: Dict[str, Optional[int]] = field(default_factory=dict)
    #: P's DERIVED PP layer split (empty = refused, reason in the log line).
    p_stage_layers: List[int] = field(default_factory=list)
    #: #1233 fix 8: what the /dev/shm orphan sweep found and freed.
    shm_sweep: Dict[str, object] = field(default_factory=dict)
    #: #1233 fix 8: the DORMANT-IMAGE sample taken at group P's first sleep --
    #: the term the next boot's ledger prices its image from.
    dormant_image_p: Dict[str, object] = field(default_factory=dict)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Log:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.fh = open(path, "a") if path else None

    def __call__(self, msg: str) -> None:
        line = f"[{_now()}] WEG2-LAUNCH {msg}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()


# --------------------------------------------------------------------------
# NVML
# --------------------------------------------------------------------------


def resolve_cards() -> List[Card]:
    """The rig's cards, from the registry's ONE NVML reader.

    Was a second pynvml transcript here (init / getCount / getHandle / decode /
    shutdown).  ``registry.nvml.list_devices`` is that transcript plus the v2
    carve-out term, which this launcher now needs, so the copy is gone rather
    than grown.
    """
    return [
        Card(d.index, d.uuid, d.name, d.total_mib, reserved_mib=d.reserved_mib)
        for d in nvml_registry.list_devices()
    ]


def order_cards(cards: List[Card]) -> List[Card]:
    """CUDA ordinal order: the 5090 first (rank 0 / PP0 / TP0), then the 3080s
    by NVML index.  Never a fixed index: resolved by name per launch."""
    big = [c for c in cards if "5090" in c.name]
    small = sorted([c for c in cards if "3080" in c.name], key=lambda c: c.nvml_index)
    if len(big) != 1 or len(small) != 2:
        raise Weg2LaunchRefused(
            f"NVML inventory is not 1x5090 + 2x3080: {[(c.nvml_index, c.name) for c in cards]}"
        )
    return [big[0], small[0], small[1]]


def nvml_memory(cards: List[Card]) -> Dict[str, "nvml_registry.MemoryInfo"]:
    """Live memory per card UUID, from the registry's ONE NVML reader.

    Replaces a local ``nvidia-smi --query-gpu=uuid,memory.used,memory.total``
    reader.  Two figures the callers below need are not derivable from that
    pair: ALLOCATABLE FREE (the driver's own ``free``, carve-out already
    excluded -- ``total - used`` returns free PLUS the carve-out, the form the
    corridor rule forbids and boot weg2rg6 measured at +424/+518/+424 MiB), and
    the carve-out itself.  ``MemoryInfo.tenant_used_mib`` is the "is another
    tenant on this card" figure and is 0-1 MiB on an idle card; the v1
    ``used_bytes`` is not, and reading it as tenancy is how #539 refused an
    empty machine.
    """
    res = {dev.uuid: mem for dev, mem in nvml_registry.memory_snapshot()}
    missing = [c.uuid for c in cards if c.uuid not in res]
    if missing:
        raise Weg2LaunchRefused(
            f"NVML has no memory row for {missing} -- the card set changed under the "
            f"launcher (present: {sorted(res)})"
        )
    return res


def nvml_process_mib(pids: set) -> Dict[str, int]:
    """Sum of NVML per-process usage per card UUID over ``pids``."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    res: Dict[str, int] = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, used, uuid = [x.strip() for x in line.split(",")]
        if int(pid) in pids:
            res[uuid] = res.get(uuid, 0) + int(used)
    return res


def session_pids(sid: int) -> set:
    out = subprocess.run(["ps", "-eo", "pid,sid"], capture_output=True, text=True).stdout
    pids = set()
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) == sid:
            pids.add(int(parts[0]))
    return pids


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------


#: #1275 FIX 2: THE LAUNCHER'S OWN KEY, set ONCE at mint time and read from
#: here by every RPC. Never re-read from the file per call: two reads are two
#: sources, and the second one is the one that goes stale.
_ADMIN_KEY: Optional[str] = None
#: The path, so teardown (and the refusal handler) can unlink it.
_ADMIN_KEY_FILE: str = ""


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
            raise Weg2LaunchRefused(f"group {name} pid {pid} died before READY (exit {proc.returncode}); see its log")
        if pid and not _alive(pid):
            raise Weg2LaunchRefused(f"group {name} pid {pid} died before READY")
        code, body = http("GET", f"http://127.0.0.1:{port}/health", timeout=25)
        if code == 200:
            dt = time.time() - t0
            log(f"R1 READY group={name} port={port} after {dt:.1f} s")
            return dt
        last = f"{code} {body[:80]!r}"
        time.sleep(3)
    raise Weg2LaunchRefused(f"group {name} not READY on :{port} after {deadline_s:.0f} s (last {last})")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------
# Preflight (inherited from boot_855_train0901.sh)
# --------------------------------------------------------------------------


def shm_holder_pids(path: str, proc_root: str = "/proc") -> List[int]:
    """Every LIVE pid that maps or holds open a file at or under ``path``.

    Read out of ``/proc/<pid>/maps`` and ``/proc/<pid>/fd`` rather than from
    ``fuser``: ``fuser`` is one exit code for a whole argv (the #1217 check
    could not say WHICH file was held, or by whom), it is not always installed,
    and it cannot be handed a fake tree by a test.  The pid list this returns is
    printed in the refusal, so a live holder is actionable instead of a boolean.

    A path that appears with the kernel's ``(deleted)`` suffix still counts: an
    unlinked-but-mapped region still occupies the memory this sweep exists to
    free.  Matching is on the full path with a boundary, never a substring, so
    ``hicache-weg2-fix973`` never matches ``hicache-weg2-fix9730``.
    """
    prefix = os.path.realpath(path)
    holders: List[int] = []

    def _hit(candidate: str) -> bool:
        p = candidate.strip()
        if p.endswith(" (deleted)"):
            p = p[: -len(" (deleted)")]
        return p == prefix or p.startswith(prefix + "/")

    try:
        entries = os.listdir(proc_root)
    except OSError:
        return holders
    for entry in sorted(entries, key=lambda e: int(e) if e.isdigit() else 0):
        if not entry.isdigit():
            continue
        found = False
        try:
            with open(f"{proc_root}/{entry}/maps") as f:
                for line in f:
                    parts = line.split(maxsplit=5)
                    if len(parts) == 6 and _hit(parts[5]):
                        found = True
                        break
        except OSError:
            pass
        if not found:
            fd_dir = f"{proc_root}/{entry}/fd"
            try:
                for fd in os.listdir(fd_dir):
                    try:
                        if _hit(os.readlink(f"{fd_dir}/{fd}")):
                            found = True
                            break
                    except OSError:
                        continue
            except OSError:
                pass
        if found:
            holders.append(int(entry))
    return holders


def _tree_bytes(path: str) -> Tuple[int, int, int]:
    """(allocated bytes, apparent bytes, entry count) of a file or tree.

    WHICH BYTE COUNT, stated because the two differ here by 0.53 GiB and only
    one of them is host RAM: ``st_blocks * 512`` is what the tmpfs actually
    OCCUPIES (and what ``df`` and the cgroup's ``shmem`` count), ``st_size`` is
    the apparent size ``ls`` shows.  Measured on the real orphan
    ``/dev/shm/hicache-weg2-fix973`` (2026-09-08): apparent 4,042,810,432 B =
    3.77 GiB, allocated 3,471,384,576 B = 3.23 GiB -- and 3.23 GiB is the figure
    boot weg2dk7 attributed and the figure a sweep frees.  Reporting the
    apparent size as "freed" would over-claim by half a GiB.
    """
    def _pair(p: str) -> Tuple[int, int]:
        st = os.lstat(p)
        return st.st_blocks * 512, st.st_size

    try:
        alloc, apparent = _pair(path)
    except OSError:
        return 0, 0, 0
    if not os.path.isdir(path) or os.path.islink(path):
        return alloc, apparent, 1
    alloc, apparent, n = 0, 0, 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                a, b = _pair(os.path.join(root, name))
            except OSError:
                continue
            alloc += a
            apparent += b
            n += 1
    return alloc, apparent, n


def shm_residue_sweep(
    log: Log,
    tag: str,
    stamp: str,
    dry: bool,
    shm_dir: str = SHM_DIR,
    proc_root: str = "/proc",
    archive_root: str = SHM_ARCHIVE_ROOT,
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
    * any live ``sglang.launch_server`` REFUSES the boot before anything is
      touched (another boot is up; this is not the moment to tidy /dev/shm);
    * an entry with a live holder (:func:`shm_holder_pids`) REFUSES the boot,
      naming the entry and the pids.  Nothing is killed and nothing is swept;
    * an orphan REGULAR FILE is moved into the archive WITH ITS CONTENT: the
      #1217 presence flags are the evidence and they are bytes long;
    * an orphan DIRECTORY TREE is archived as a MANIFEST (name, bytes, entry
      count, mtime) and then removed.  Copying a dead boot's 3.2 GiB page store
      of ~100k small files onto spinning disk at preflight would cost minutes
      and buy nothing -- the evidence of a leaked store is that it existed and
      how big it was, not the pages inside it.
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
    live = subprocess.run(
        ["pgrep", "-f", "sglang[.]launch_server"], capture_output=True, text=True
    ).stdout.split()
    if live:
        raise Weg2LaunchRefused(
            f"#1217 shm residue: LIVE launch_server pid(s)=[{' '.join(live)}] -- refusing this "
            f"boot, not sweeping, not killing anything (our /dev/shm entries: {own})"
        )
    held = {n: shm_holder_pids(os.path.join(shm_dir, n), proc_root) for n in own}
    refused = {n: pids for n, pids in held.items() if pids}
    if refused:
        raise Weg2LaunchRefused(
            f"#1217/#1233 shm residue: LIVE HOLDER on our own /dev/shm entries "
            f"{ {n: pids for n, pids in sorted(refused.items())} } -- refusing this boot, not "
            f"sweeping, not killing anything"
        )
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
                "refused": {}, "archive": archive}
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
            "refused": {}, "archive": archive}


def stale_deadman_sweep(log: Log, ports: Sequence[int], dry: bool) -> None:
    out = subprocess.run(["pgrep", "-af", "boot_deadman.sh"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, _, argv = line.partition(" ")
        if any(f" {p}" in argv for p in ports) and "boot_deadman.sh" in argv:
            if dry:
                log(f"DRY-RUN: would kill stale deadman pid {pid} ({argv})")
            else:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    log(f"stale deadman pid {pid} killed (argv: {argv}) -- it watched a previous boot")
                except OSError:
                    pass


def host_preflight(log: Log, tag: str, dry: bool) -> None:
    if dry:
        log("DRY-RUN: host_ledger_preflight.sh not run")
        return
    rc = subprocess.run([HOST_PREFLIGHT, f"{GPU_ARB}/preflight_weg2_{tag}.log"]).returncode
    if rc != 0:
        raise Weg2LaunchRefused("BOOT REFUSED by the host-ledger preflight (#721 floor) -- see its log")
    mi = host_ledger.read_meminfo()
    avail_gib = mi["MemAvailable"] / host_ledger.GIB
    if avail_gib < 40:
        top = subprocess.run(["ps", "-eo", "rss,pid,comm", "--sort=-rss"], capture_output=True, text=True).stdout.splitlines()[:8]
        raise Weg2LaunchRefused(f"free -g available {avail_gib:.1f} GiB < 40 GiB; top RSS holders (NOT killed):\n" + "\n".join(top))
    oom = open("/sys/fs/cgroup/memory.events").read()
    log(f"host preflight PASS: MemAvailable {avail_gib:.1f} GiB; cgroup memory.events baseline: {' '.join(oom.split())}")


def _free_instrument(mem: Dict[str, "nvml_registry.MemoryInfo"], cards: List[Card]) -> str:
    """The ``free`` instrument covering EVERY card in one printed line.

    FIX 2, finding 3: the label used to be the literal ``nvml_v2_free`` typed
    into the f-string, while the figure came from whichever struct the registry
    could read.  The weakest card sets the token, because one token stands for
    the whole line.
    """
    rows = [mem[c.uuid] for c in cards if c.uuid in mem]
    if rows and not all(r.carve_out_known for r in rows):
        return nvml_registry.FREE_INSTRUMENT_V1
    return nvml_registry.FREE_INSTRUMENT_V2


def cards_free_check(cards: List[Card], log: Log) -> None:
    mem = nvml_memory(cards)
    for c in cards:
        m = mem[c.uuid]
        if m.tenant_used_mib > 1500:
            # The instrument is the one this MemoryInfo actually read, not a
            # literal typed here (FIX 2, finding 3): with no v2 struct the
            # figure below is the v1 ``used``, which counts the carve-out as
            # tenancy (#539) -- the refusal must not read as carve-out-aware
            # when it is not.
            raise Weg2LaunchRefused(
                f"card {c.nvml_index} ({c.name}) has {m.tenant_used_mib} MiB held by processes "
                f"(> 1500; instrument {m.tenant_used_instrument}, i.e. the driver carve-out of "
                f"{m.reserved_mib} MiB is NOT counted as tenancy) -- not free; not killing anything"
            )
    log(
        f"cards free (instrument: {_free_instrument(mem, cards)}, allocatable): "
        + ", ".join(
            f"idx{c.nvml_index}={mem[c.uuid].tenant_used_mib} MiB used by processes / "
            f"{mem[c.uuid].free_mib} MiB free / {mem[c.uuid].allocatable_mib} MiB allocatable "
            f"({mem[c.uuid].reserved_mib} MiB driver-reserved, never allocatable)"
            for c in cards
        )
    )


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


def mount_store(log: Log, store_gib: float, dry: bool) -> None:
    if dry:
        log(f"DRY-RUN: would mount tmpfs size={store_gib:.0f}G at {STORE_MOUNT}")
        return
    os.makedirs(STORE_MOUNT, exist_ok=True)
    mounts = open("/proc/mounts").read()
    if f" {STORE_MOUNT} " in mounts:
        subprocess.run(["umount", STORE_MOUNT], check=False)
        log(f"store: stale tmpfs at {STORE_MOUNT} unmounted (fresh store per boot)")
    subprocess.run(["mount", "-t", "tmpfs", "-o", f"size={int(store_gib)}G", "tmpfs", STORE_MOUNT], check=True)
    st = os.statvfs(STORE_MOUNT)
    log(
        f"store: tmpfs mounted at {STORE_MOUNT} size={st.f_blocks * st.f_frsize / host_ledger.GIB:.2f} GiB "
        f"(RAM-backed canonical page store, user ruling 2026-09-07; NO disk tier in V1 -- the file "
        f"backend has no second tier, /spinning/hicache-weg2 is not used)"
    )


def store_extra_config(store_gib: float) -> str:
    # max_size + min_free_space must fit the filesystem (W8): the tmpfs IS
    # store_gib, so max_size = store_gib - 1 G and min_free_space = 1 G.
    max_size = max(1, int(store_gib) - 1)
    return json.dumps({"max_size": f"{max_size}G", "min_free_space": "1G", "max_size_scope": "shared"}, separators=(",", ":"))


# --------------------------------------------------------------------------
# argv composition
# --------------------------------------------------------------------------


def common_flags(
    model: str,
    s_gb: int,
    m_mib: int,
    store_gib: float,
    max_kv_per_request: int,
    write_policy: str = "write_through",
    group: str = "both",
    random_seed: int = RANDOM_SEED,
    barlink_cap_cycles: int = BARLINK_BAR1_CAP_CYCLES,
    census_interval: int = COLLECTIVE_CENSUS_INTERVAL,
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
    pp_size=1.  MEASURED consequence of the old placement: D booted with
    disable_overlap_schedule=True inheriting a PP-only reason (BSSCALE_0907.md
    D4), so CPU scheduling of round n+1 could not hide behind GPU work of
    round n on a group that has no pipeline at all.  See argv_p / argv_d.
    """
    return [
        "--model-path", model,
        "--trust-remote-code",
        "--served-model-name", "Qwen3.8-27B",
        "--rank-gpu-id", "0,1,2",
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
        "--enable-hierarchical-cache",
        "--hicache-host-role", "staging",
        "--hicache-size", str(s_gb),
        "--hicache-mamba-host-mib", str(m_mib),
        "--hicache-write-policy", write_policy,
        "--hicache-storage-backend", "file",
        "--hicache-mem-layout", "layer_first",
        "--hicache-io-backend", "direct",
        "--hicache-storage-backend-extra-config", store_extra_config(store_gib),
        "--hicache-canonical-kv-page",
        "--host", "127.0.0.1",
        "--chunked-prefill-size", str(CHUNKED_PREFILL_TOKENS),
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
    ] + early_read_flags(group) + [
        # THE DOUBLY-STATED FACTS COME FROM ONE TABLE (#1235). --mamba-ssm-dtype
        # is here for both groups and the two uneven-DCP flags for group D; the
        # matching environment keys are written by build_env from the SAME
        # rows, because ServerArgs reads them from os.environ before the flags
        # publish themselves. See EarlyReadFact for the ordering evidence.
        "--enable-memory-saver",
        "--enable-weights-cpu-backup",
    ]


def admin_key_flag(admin_api_key: Optional[str]) -> List[str]:
    """#1275: BOTH groups get the key, or the capability is half-present.

    The front talks to P and D with the same header, and the resize lever is
    wanted on whichever group holds the store -- so a key on one group only
    would make the front's flip RPCs succeed against one peer and 401 against
    the other, which is a worse state than having no key at all. One helper,
    two call sites, so the two argv builders cannot drift apart.

    Empty list when unkeyed, which keeps every pre-#1275 argv byte-identical.
    """
    return ["--admin-api-key", admin_api_key] if admin_api_key else []


def argv_p(
    py: str,
    model: str,
    budgets: List[int],
    s_gb: int,
    m_mib: int,
    store_gib: float,
    extra: List[str],
    p_bs: int = 8,
    max_kv_per_request: int = CONTEXT_LENGTH_TOKENS,
    stage_ratio: Optional[str] = None,
    attn_stage_ratio: Optional[str] = None,
    write_policy: str = "write_through",
    depth: int = 0,
    window_mib: str = P_BARLINK_BAR1_WINDOW_MIB,
    random_seed: int = RANDOM_SEED,
    barlink_cap_cycles: int = BARLINK_BAR1_CAP_CYCLES,
    census_interval: int = COLLECTIVE_CENSUS_INTERVAL,
    draft_kv_on_p: bool = True,
    admin_api_key: Optional[str] = None,
) -> List[str]:
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
    #   [19,24,21]), so the argv would contradict SGLANG_PP_LAYER_SET.
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
        stage_ratio = _csv(P_PP_STAGE_RATIO_SCORES)
    if attn_stage_ratio is None:
        attn_stage_ratio = _csv(P_PP_ATTN_STAGE_RATIO_SCORES)
    if bool(stage_ratio) != bool(attn_stage_ratio):
        raise Weg2LaunchRefused(
            "W40 Weg2PPCutRefused: --pp-stage-ratio %r and "
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
    return [py, "-m", "sglang.launch_server"] + common_flags(
        model, s_gb, m_mib, store_gib, max_kv_per_request, write_policy, "P",
        random_seed, barlink_cap_cycles, census_interval,
    ) + [
        # C1/K1: P's own bs. Concurrency for the front's leg-1 fan-out AND
        # the size of P's req_to_token_pool (R-13), which is why it is
        # resolved before the budget solve and printed with it.
        "--max-running-requests", str(p_bs),
        "--tp-size", "1", "--pp-size", "3",
        # #692 MICROBATCH DEPTH, group P only -- group D runs pp_size=1 and a
        # pipeline depth is meaningless there. STATED even at 0 so the argv is
        # an honest statement of what the boot runs, and published ONCE as a
        # group constant: every rank reads the same token off the same argv,
        # so no rank can derive a different depth (the ring size is a
        # collective property -- two ranks disagreeing about pp_loop_size is
        # the v7pp12 starvation, not a tuning difference). Solved by
        # solve_p_depth from the previous boot's own PP-BUBBLE line; the
        # launcher prints the WEG2 P-DEPTH provenance line either way.
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
    ] + (list(P_DRAFT_KV_FLAGS) if draft_kv_on_p else []) + [
        "--port", str(PORT_P),
    ] + admin_key_flag(admin_api_key) + extra


def w38_armed_line(argv_of_p: Sequence[str]) -> str:
    """State group P's carrierless-PP arm AT LAUNCH, once.

    RECONCILED 2026-09-08 (weg2 train, W38 vs #1245): there is ONE arm for the
    carrierless PP fact now, and it is #1245's ``undistributable`` drop --
    ``clear_state_aligned_extent_undistributable`` at scheduler.py:12157, gated
    on ``pp_row_carrier_present(self)`` exactly as W38 was.  W38's own gate at
    ``Scheduler._prefetch_kvcache`` is DELETED: it refused the store READ
    outright and was desk-only, while #1245 is BOOT-PROVEN (weg2rg6, 141 group-P
    prefill passes, 20 leg-1 legs, three real drops, W27 genuine 0 over that
    population).  So group P DOES read the store again; what it may not do is
    ADOPT a host hit that landed on this rank alone.

    Read off the argv this launcher is about to run rather than asserted from
    memory: ``--pp-size N`` with N > 1 is the predicate's own precondition, so
    if that flag ever changes the line changes with it instead of lying.
    """
    pp = 1
    for i, a in enumerate(argv_of_p):
        if a == "--pp-size" and i + 1 < len(argv_of_p):
            try:
                pp = int(argv_of_p[i + 1])
            except ValueError:
                pp = 1
    if pp <= 1:
        return ("WEG2 #1245 NOT ARMED: group P is launched with --pp-size %d, so no rank can hold a "
                "fact it cannot tell its peers and nothing is dropped" % pp)
    return ("WEG2 #1245 ARMED (W38 RETIRED INTO IT): group P is launched --pp-size %d and the no-flip "
            "PP form carries no #631 row carrier, so an asynchronously-completed host hit that landed "
            "on ONE rank is DROPPED at admission (#1245 UNDISTRIBUTABLE LOAD-BACK DROPPED) instead of "
            "moving that rank's prefix_indices alone -- the W27 width divergence that killed boots "
            "weg2sc1 and weg2rg3. THE STORE READ ITSELF IS NOT REFUSED any more: W38 "
            "Weg2CarrierlessPpStoreRead disarmed it entirely and was the desk-only half of the same "
            "fact; the two arms were reconciled to the boot-proven one. WHAT IT COSTS: only the "
            "extents actually dropped, counted by that log line, not P's whole L3 prefix reuse. "
            "Measured per drain epoch by the front's 'WEG2 P-PREFIX-REUSE' line; the write-through "
            "and D's own store read are untouched, and the drop lifts itself the moment a carrier "
            "exists (#968 PP0-authoritative materialisation is the named remedy)." % pp)

def argv_d(
    py: str,
    model: str,
    budgets: List[int],
    s_gb: int,
    m_mib: int,
    store_gib: float,
    extra: List[str],
    d_bs: int = 8,
    max_kv_per_request: int = CONTEXT_LENGTH_TOKENS,
    x_tokens: int = 0,
    num_continuous_decode_steps: int = 1,
    disable_overlap: bool = False,
    tp_ratio_flags: Sequence[str] = ("--rank-tp-ratio", "auto"),
    token_vector_flags: Sequence[str] = (),
    random_seed: int = RANDOM_SEED,
    barlink_cap_cycles: int = BARLINK_BAR1_CAP_CYCLES,
    census_interval: int = COLLECTIVE_CENSUS_INTERVAL,
    disable_cuda_graph: bool = False,
    admin_api_key: Optional[str] = None,
) -> List[str]:
    return [py, "-m", "sglang.launch_server"] + common_flags(
        model, s_gb, m_mib, store_gib, max_kv_per_request, "write_through", "D",
        random_seed, barlink_cap_cycles, census_interval,
    ) + (
        ["--disable-overlap-schedule"] if disable_overlap else []
    ) + (
        # #1241b CONTROL ARM, NOT A TUNING KNOB. Default off, and off means
        # this list is empty -- the shipped argv is byte-identical to the one
        # before this flag existed. Passing it CHANGES THE MEASURED FORM
        # (full-perf validation runs with graphs and spec), so a number taken
        # under it describes the eager form and nothing else.
        ["--disable-cuda-graph"] if disable_cuda_graph else []
    ) + [
        # C1/K2: D's own bs, independent of P's by construction.
        "--max-running-requests", str(d_bs),
        # C11/K5: law 4 enforced where the uncached extent is REAL -- after
        # match_prefix, on extend_input_len, inside get_new_batch_prefill.
        # The front prices with len(text)/3.0 and never re-checks, so a
        # front-side estimate alone let a 19,401-token uncached extent
        # through a 4,096-token grant (weg2zr2 rid weg2-4-8). 0 = off.
        "--tp-prefill-max-tokens", str(x_tokens),
        # THE STRATEGY IS STATED, NOT INHERITED (FIX 1r/1). Leaving this at
        # 'auto' made D's mamba radix-cache strategy a SIDE EFFECT of the
        # absent disable flag: _mamba_radix_cache_resolution reads
        # `wants_overlap = not view.disable_overlap_schedule`
        # (arg_groups/overrides.py:1225-1239), so switching the overlap
        # schedule on also switched no_buffer -> extra_buffer, and
        # mamba_pool_floor.mamba_ping_pong_slots then charges 2 state slots
        # per running request instead of 0 -- DOUBLING D's device mamba floor
        # (16 -> 32 slots at --max-running-requests 8) out of the FIXED
        # --rank-gpu-memory-mib budgets. Stated here so the argv is an honest
        # statement of what D runs, and priced in the SCHEDULER: log line
        # (d_overlap_cost_line) so the launcher cannot move D's device
        # residency without the number appearing. The value follows the
        # overlap choice exactly -- it is not a second knob.
        "--mamba-radix-cache-strategy", "no_buffer" if disable_overlap else "extra_buffer",
        "--tp-size", "3", "--pp-size", "1",
        # OVERLAP SCHEDULE ON for D -- by ABSENCE of the disable flag, which
        # is the only way to have it: there is no --enable-overlap-schedule.
        # Every gate that forces it off was checked against THIS argv and
        # none applies: pp_size>1 (D is 1), --enable-pdmux (not passed,
        # server_args.py:19553), device cpu/mps (cuda), sparse-head
        # embeddings and dllm (neither), and the hybrid-mamba resolution
        # (arg_groups/overrides._mamba_radix_cache_resolution), which picks
        # 'extra_buffer' and LEAVES overlap on for Qwen3_5ForConditional-
        # Generation on linear_attn_backend=triton -- that arch is in
        # _MAMBA_EXTRA_BUFFER_ARCHS. If a gate ever does refuse, it is not to
        # be weakened: the launcher's --d-disable-overlap-schedule puts the
        # flag back and logs W41.
        #
        # #1030's justification is PP-only and does not reach D
        # (BSSCALE_0907.md D4).
        "--num-continuous-decode-steps", str(int(num_continuous_decode_steps)),
        "--rank-gpu-memory-mib", ",".join(str(b) for b in budgets),
    ] + list(tp_ratio_flags) + [
        # THE WEIGHT OBJECTIVE IS STATED, NOT ASSUMED (#1017). A per-rank MiB
        # LIST under TP requires the uneven-TP ratio, and 'auto' derives the
        # weights from that list -- but 'auto' is the CAPACITY-FIRST default
        # (server_args.py:838), so shipping it as a literal meant every Weg-2
        # boot chose an objective without saying so. The choice now arrives
        # from d_tp_ratio_decision as --d-tp-objective, priced on the launch
        # line; the default is still 'auto' because the maxkv law makes
        # capacity the default objective, not because nothing was decided.
        "--speculative-algorithm", SPEC_ALGORITHM,
        "--speculative-num-steps", str(SPEC_NUM_STEPS),
        "--speculative-eagle-topk", str(SPEC_EAGLE_TOPK),
        "--speculative-num-draft-tokens", str(SPEC_NUM_DRAFT_TOKENS),
    ] + list(token_vector_flags) + [
        # NO TOKEN VECTOR BY DEFAULT (#1032). What stood here was
        # `--uneven-token-vector 29,19,16 --uneven-token-vector-role seed`, the
        # emitted value of RETRACTED investigation #602. See
        # RETRACTED_SEED_VECTOR_1032 for what boot weg2rg6 measured the runtime
        # doing with it (superseding it to 17,7,8, every rank, every boot) and
        # for why "the install landed" is a reason to stop shipping it rather
        # than a reason it was harmless. An operator vector arrives through
        # d_token_vector_decision, which refuses a retracted one BY TICKET.
        # #1234 C1 -- dcp:0 goes 24 -> 40 MiB. MEASURED, and this one
        # number is the whole regression fix.
        #
        # WHY 24 WAS WRONG. At a 24-MiB window max_payload yields
        # chunk_max = 2 093 056 B, exactly one 4096-byte page below 2 MiB,
        # so 16 rounds carry 100 466 688 B = 4088 tokens. The dcp
        # attention-out combine at --chunked-prefill-size 4096 sends
        # 4096 x 24576 = 100 663 296 B. Miss: 196 608 B, 0.196 %, EIGHT
        # TOKENS -- and every one of the 16 calls per prefill chunk ran on
        # the host-staged gloo plane at 0.68 GB/s instead of bar1's 3.19.
        #
        # WHY 40 AND NOT MORE. The binding gate is not "sum of declared <=
        # 224" but NVML free minus RESERVE_MIB_DEFAULT (32), evaluated when
        # each group builds (barlink_matrix_transport.py:330, :357-364);
        # measured at dcp:0 build time on boot weg2ab1: 74 - 32 = 42 MiB.
        # DCP_0=48 was REFUSED on metal (Bar1WindowRefused, boot weg2ab1).
        # At 40 MiB chunk_max = 3 493 888, so the 96-MiB all_reduce plans to
        # 10 rounds and the 22-MiB q_full all_gather to 7; measured BAR1
        # Used 224/256 MiB per 3080, i.e. the aperture is EXHAUSTED, which
        # is the argument for the derived round bound rather than for a
        # bigger window next time.
        #
        # WHY NOT 25 (the arithmetic minimum for 16 rounds). Because the
        # window also buys all_gather rounds: 12 -> 7 across the same step.
        # ARM 1 of the A/B boot measured w40 at 2361.4 ms of wait per full
        # chunk against the baseline's 4755.7 and NCCL's 3307.6.
        "--barlink-bar1-window-mib", "16,TP_0=32,DCP_0=40",
        # #1234 C5: group D is the one place where a silent host-staged
        # 4.7x is a known boot killer, and with the window above no declared
        # class is anywhere near the round budget. The library default stays
        # 'warn'; opting in is a deployment decision, made here.
        #
        # WHAT THIS DOES AND DOES NOT STOP (FIX 1). Only the two size-driven
        # refusal kinds -- 'round' and 'oversize', barlink.py's
        # UNCOVERED_REFUSAL_STOPS -- stop the group. A sub-min_bytes or
        # misaligned collective anywhere in D's process still takes the
        # priced warn path and its small, correct gloo answer; killing a boot
        # for one of those would be an abort wider than the reason above.
        # Under the DERIVED bound the 'round' half is currently inert on this
        # rig (the crossover is scale-invariant and has no root here -- see
        # barlink_bar1.round_budget); it binds when a cap is pinned, which is
        # exactly how boot weg2zr2 failed, and 'oversize' is the reachable
        # half that guards the window this line sits next to.
        "--barlink-uncovered-class", "refuse",
        "--port", str(PORT_D),
    ] + admin_key_flag(admin_api_key) + extra


def _import_duplex_gate() -> float:
    from sglang.srt.managers import weg2_memory_saver

    return float(weg2_memory_saver.DUPLEX_SPLIT_MIN_RATIO)


@dataclass
class HostRingPlan:
    """What C18 publishes, and why it may publish nothing.

    ``armed`` False means the ring is NOT in this boot, and THERE IS NO
    FALLBACK -- spec Amendment A1-3, measured, not argued:

        the OLD flip form is INFEASIBLE on this host budget on the default
        table (W20 at every rung: host weights 38.77 GiB run / 33.12 GiB
        launch, boot weg2rg1) -- no Weg-2 boot may pin --ring-table-boot
        weg2zr2 to get past it; the ring with gathered legs is the only route.

    So every un-armed arm below REFUSES BY NAME and the launcher exits 2.  The
    predecessor of this text said "the host ring is NOT armed and the OLD flip
    form runs (spec section 10.4)" in the W32, W33 and W34 arms, and the metal
    had refuted it twenty minutes before that commit was written: those lines
    are printed at ``prepare_host_ring`` (launcher.py step 1b) and the arm that
    decides whether the un-armed form is fundable is ``host_ledger.choose`` one
    step LATER, so the sentence was an assertion made before its own check --
    and the check said no.  A refusal that names a fallback must HAVE one.

    Two reasons remain, and both are printed:

    * ``table`` is not None -- W33 (no proven registration form), W32, or W34
      (the launcher found itself without gathered legs).  The table exists, so
      the un-armed charge CAN be priced; it just cannot be funded.
    * ``table`` is None -- R22, no measured table at all.  Then it cannot even
      be priced, and the ledger refuses by name (W20) for that reason instead.
    """

    form: str = ""
    dir: str = ""
    env_map: str = ""
    epoch: int = 0
    armed: bool = False
    table: Optional["ring_table.RingTable"] = None
    lines: List[str] = field(default_factory=list)
    #: C12/C13: the per-card duplex table and the ``SGLANG_WEG2_PCIE_DUPLEX``
    #: string built from it.  Launcher OUTPUT (spec R19), solved from the
    #: step-0 probe record; empty means no card splits its lock key.
    duplex: Optional["ring_table.DuplexTable"] = None
    duplex_env: str = ""

    @property
    def host_weights_bytes(self) -> int:
        """The RUN-moment host weights charge, from the same measured table for
        both forms.

        Armed: ``Sigma_c H(c)``.  The region is fungible, so the tag in flight
        is drawn from the SAME per-card bytes the dormant image occupies and
        there is nothing to add (spec C19 / section 7).

        Un-armed (the OLD serial per-tag form): one image plus ONE tag in
        flight -- ``Sigma_c H(c) + max_k Sigma_c bytes(k, c)``.  This is the
        parent's ``backup_resident + chunk_gib`` model with the stale constant
        replaced by the measured table and the AVERAGE chunk replaced by the
        measured largest one.  It is ONE tag because the front sends one
        ``/release_memory_occupation`` per tag; the sum is over cards because
        that tag's shards land on every card at once.  Charging ``Sigma_c
        max_k`` instead -- each card's own largest tag, summed -- prices a step
        the flip never takes (the maxima are different tags) and cost the
        DEFAULT path 5.6 GiB it does not spend, which was enough to make every
        arm of the ladder refuse with W20.
        """
        if self.table is None:
            return 0
        total = self.table.total_h_bytes
        if not self.armed:
            total += self.table.max_step_total_mib * ring_table.MIB
        return total

    @property
    def host_weights_span1_bytes(self) -> int:
        return 0 if self.table is None else self.table.total_span1_bytes

    @property
    def provenance(self) -> str:
        if self.table is None:
            return "no measured table"
        # A1-3: un-armed names no form that runs (FIX 3 round 3, same class as
        # the two sentences fix 2 retracted).  The table is still charged and
        # printed, because the refusal's arithmetic is what the operator reads.
        form = (f"ring form {self.form}" if self.armed
                else "NO ARMED FORM -- this boot refuses (A1-3)")
        return f"{self.table.provenance()}; charged for the {form}"


#: R17's gate, read from the module that owns the key so the launcher's printed
#: verdict and the rank's actual key can never be computed from two numbers.
DUPLEX_GATE = _import_duplex_gate()


def _front_leg_form() -> str:
    """``front.FLIP_LEG_FORM`` -- read, never restated.

    The launch check has to know how the flip orders its legs, and the only
    honest source is the module that does the ordering.  A copy here would be a
    second bookkeeping of the same fact and would go stale the moment C9 lands.
    """
    from sglang.srt.weg2 import front

    return front.FLIP_LEG_FORM


def _split_decisions(
    card_uuids: Sequence[str],
    ratios: Optional[Dict[str, float]] = None,
    leg_form: str = "interleave",
    override: Optional[bool] = None,
) -> Dict[str, bool]:
    """Per card: does its PCIe key SPLIT the two directions?  Decided ONCE, here.

    FIX 1 round 1, and boot weg2rg2 is the reason it is a decision at all rather
    than a reading of R17's ratio.  R17 asks whether the split BUYS enough
    (concurrent aggregate >= 1.5x serial); weg2rg2 asked whether the single key
    is SAFE under C9's gathered legs, and the answer is no.  Each leg holds the
    per-card key across its whole tag loop, so on a one-key card the pair is
    serialised again and R5's premise -- W's per-tag releases fund S's acquires
    -- is false there: the sleep leg blocks in ``acquire`` at free = 0 holding
    the key the wake leg needs.  Measured cost of that on nvml0: 120.189 s, W31,
    W29 on three ranks, group-fatal W4.

    So: under GATHERED legs every card splits, whatever its ratio.  That is not
    free-lunch reasoning -- the x4 card's 1.316 is concurrent OVER serial, so
    concurrency is 32 % faster there too; R17's gate only ever said the split is
    not worth a mechanism of its own, and A1-4 says that card gets no BENEFIT
    from the split, not that it must not have one.  Under a front that does NOT
    gather (no state of this tree), the ratio gate stands and a card below it
    keeps one key -- which is correct there, because a serial front never has
    two legs in flight to deadlock.

    ``override`` is the test injection point: ``True`` = force every key to
    split, ``False`` = force every key single (the weg2rg2 state).
    """
    if override is not None:
        return {uuid: bool(override) for uuid in card_uuids}
    if leg_form == "interleave":
        return {uuid: True for uuid in card_uuids}
    table = ratios or {}
    return {
        uuid: table.get(uuid) is not None and table[uuid] >= DUPLEX_GATE
        for uuid in card_uuids
    }


#: The launch check's body, run BY THE TREE THE RANKS IMPORT (FIX 3 round 3,
#: finding 2).  It sets the duplex variable under the NAME THAT TREE USES -- not
#: the launcher's -- so a tree that renamed it is a failed probe rather than a
#: silently unconfigured one, and it asks for the KEY, which is the only artefact
#: the launcher and the rank must agree on.  ``CUDA_VISIBLE_DEVICES=''`` in the
#: caller's env means no context is bought to answer.
_KEY_PROBE = (
    "import json, os, sys\n"
    "from sglang.srt.managers import weg2_memory_saver as s\n"
    "os.environ[s.PCIE_DUPLEX_ENV] = sys.argv[1]\n"
    "out = {}\n"
    "for u in json.loads(sys.argv[2]):\n"
    "    try:\n"
    "        keys = {s.pcie_lock_path(u, direction=d) for d in s.PCIE_DIRECTIONS}\n"
    "    except TypeError:\n"
    "        keys = {s.pcie_lock_path(u)}\n"
    "    out[u] = [len(keys), len(s.PCIE_DIRECTIONS)]\n"
    "sys.stdout.write('WEG2-KEYS ' + json.dumps(out))\n"
)


def _resolve_keys_in_rank_tree(
    card_uuids: Sequence[str], published: str, py: str, tree: str
) -> Tuple[Dict[str, Tuple[int, int]], str]:
    """``{uuid: (keys THAT tree builds, directions THAT tree has)}``, or ``({}, why)``.

    BOTH halves come from the rank tree, never one from here: a tree with a
    different :data:`PCIE_DIRECTIONS` would otherwise be compared against this
    launcher's count and read as split when it is not.

    One interpreter start per boot, in the environment ``build_env`` gives the
    ranks (``PYTHONPATH=<tree>/python``), because a stale tree on that path is
    precisely the divergence this check exists to see and precisely the one an
    in-process import cannot show.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}/python"
    env["CUDA_VISIBLE_DEVICES"] = ""
    from sglang.srt.managers import weg2_memory_saver as _s

    env.pop(_s.PCIE_DUPLEX_ENV, None)  # the probe sets the TREE's own name
    try:
        proc = subprocess.run(
            [py, "-c", _KEY_PROBE, published, json.dumps(list(card_uuids))],
            capture_output=True, text=True, env=env, cwd=tree, timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    out = proc.stdout or ""
    marker = "WEG2-KEYS "
    if proc.returncode != 0 or marker not in out:
        tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
        return {}, f"rc={proc.returncode} stderr={tail[:400]!r}"
    try:
        raw = json.loads(out[out.index(marker) + len(marker):].strip())
        return {str(k): (int(v[0]), int(v[1])) for k, v in raw.items()}, ""
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return {}, f"unparsable probe answer ({type(exc).__name__}: {exc})"


def _serialised_cards(
    card_uuids: Sequence[str],
    published: str = "",
    leg_form: str = "interleave",
    *,
    py: str = "",
    tree: str = "",
    notes: Optional[List[str]] = None,
) -> List[str]:
    """The cards on which the two flip legs are SERIALISED, by UUID.

    FIX 1 (round 1), and boot weg2rg2 is why.  Serialisation is a property of a
    CARD, not of the flip: C9 puts both legs in flight, but each leg takes the
    per-card PCIe key for its WHOLE tag loop (weight_updater sleep-D2H at the
    ``with self._weg2_pcie_lock(...)`` around the pause loop, wake-H2D around
    the resume loop), and on a card whose measured duplex ratio does not reach
    R17's gate both directions resolve to the SAME key.  There the gathered pair
    is serialised again -- W's per-tag releases cannot fund S's acquires,
    because W is queued behind S on the key, or S behind W.

    The predecessor computed the AND over cards, printed it, and gated nothing;
    its justification was that "a card that keeps one key costs time on that
    card, and nothing else".  The metal refuted that 20 minutes later: nvml0
    (1.316, DUPLEX-NULL) passed R5 on both directions, armed, and wedged
    120.189 s into the FIRST flip -- W31 Weg2HostRingExhausted need=24 free=12
    MiB, its peer holding the granules and itself queued on that same single
    key, ending in W29 on all three P ranks and a group-fatal W4.

    IT ASKS THE KEY, NOT ITS OWN DICT (FIX 2 round 2, finding 2).  The
    grandparent passed :func:`_split_decisions`' answer straight back in as
    ``splits=``, so the saver returned it verbatim and the check was
    tautological -- it could not see a rank that would resolve the published
    string differently, which is the ONE thing it exists to see.  Here the
    argument is the PUBLISHED STRING and the answer is read off the KEY ITSELF:
    the two directions' lock paths, compared.

    IT ASKS THE TREE THE RANKS IMPORT (FIX 3 round 3, finding 2), and that is
    the half FIX 2 still owed.  Resolving through the LAUNCHER's own
    ``weg2_memory_saver`` cannot see a stale worktree on ``PYTHONPATH`` or a
    partial rebase -- the exact deployment shape the W34 comment names as the
    reason this arm exists -- because the launcher's import is not the ranks'
    import.  With ``py`` and ``tree`` given, the key is resolved by one
    subprocess per boot under ``PYTHONPATH=<tree>/python`` and
    ``CUDA_VISIBLE_DEVICES=''``; a tree that single-keys a card sends that card
    to W34, and a probe that DIES (no module, unknown wire format, W36) makes
    every card serialised, which is the same refusal one step earlier.

    Without them the resolution is IN-PROCESS and says so in ``notes``: that is
    the desk path, for tests that own both sides of the import themselves.  It is
    never the boot path -- :func:`prepare_host_ring` always passes both.
    """
    if leg_form != "interleave":
        # The front itself does not gather: every card is serialised, whatever
        # its key does.
        if notes is not None:
            notes.append(
                f"WEG2-HOST-RING KEY-PROBE skipped: leg_form={leg_form!r} is not "
                "'interleave', so every card is serialised whatever its key does")
        return list(card_uuids)
    if py and tree:
        counts, why = _resolve_keys_in_rank_tree(card_uuids, published, py, tree)
        if not counts:
            if notes is not None:
                notes.append(
                    f"WEG2-HOST-RING KEY-PROBE FAILED in the RANK TREE {tree} "
                    f"({py}): {why} -- the tree the ranks import cannot resolve "
                    "the key this launcher published, so EVERY card is reported "
                    "SERIALISED and this boot refuses (W34) before either group "
                    "starts, rather than taking one key at the first flip")
            return list(card_uuids)
        # A card the probe did not answer for is SERIALISED, never assumed split.
        serialised = [u for u in card_uuids
                      if counts.get(u, (0, 1))[0] < counts.get(u, (0, 1))[1]]
        if notes is not None:
            notes.append(
                f"WEG2-HOST-RING KEY-PROBE resolver=RANK TREE {tree} ({py}) "
                + ", ".join(
                    f"{u}={counts.get(u, (0, 0))[0]}/{counts.get(u, (0, 0))[1]} "
                    "key(s) per direction" for u in card_uuids)
                + f" -- serialised={serialised or 'none'}")
        return serialised
    from sglang.srt.managers import weg2_memory_saver

    if notes is not None:
        notes.append(
            "WEG2-HOST-RING KEY-PROBE resolver=IN-PROCESS (desk path: no --tree "
            "given) -- this cannot see a stale tree on the ranks' PYTHONPATH")
    serialised: List[str] = []
    saved = os.environ.get(weg2_memory_saver.PCIE_DUPLEX_ENV)
    os.environ[weg2_memory_saver.PCIE_DUPLEX_ENV] = published
    try:
        for uuid in card_uuids:
            try:
                keys = {
                    weg2_memory_saver.pcie_lock_path(uuid, direction=d)
                    for d in weg2_memory_saver.PCIE_DIRECTIONS
                }
            except TypeError:
                # A saver whose pcie_lock_path has no direction parameter at
                # all: one key, both legs, by construction.
                keys = {weg2_memory_saver.pcie_lock_path(uuid)}
            except getattr(weg2_memory_saver, "Weg2DuplexDecisionRefused", ()):
                # W36 raised while resolving.  It is already a named refusal;
                # here it means SERIALISED, so the launcher's own W34 arm gives
                # the operator the whole picture in one refusal instead of a
                # traceback out of a helper.
                keys = {uuid}
            if len(keys) < len(weg2_memory_saver.PCIE_DIRECTIONS):
                serialised.append(uuid)
    finally:
        if saved is None:
            os.environ.pop(weg2_memory_saver.PCIE_DUPLEX_ENV, None)
        else:
            os.environ[weg2_memory_saver.PCIE_DUPLEX_ENV] = saved
    return serialised


def _credit_counter_rows(credit_dir: str = "") -> List[Tuple[str, str, str, int]]:
    """``(path, name, boot half of the epoch, publisher pid)`` per counter file.

    The boot half is everything before the first ``.`` of the stored epoch token
    (:func:`weg2_memory_saver.credit_epoch` composes ``<boot>.<flip>``).  A
    LEGACY file whose stamp is a bare integer has no boot half and gets ``""``,
    which matches no boot -- the conservative direction, since it belongs to a
    boot that predates the token.
    """
    from sglang.srt.managers import weg2_memory_saver

    directory = credit_dir or os.environ.get(
        weg2_memory_saver.VRAM_CREDIT_DIR_ENV,
        os.environ.get(weg2_memory_saver.PCIE_LOCK_DIR_ENV,
                       weg2_memory_saver.DEFAULT_PCIE_LOCK_DIR),
    )
    prefix = "." + weg2_memory_saver.VRAM_CREDIT_PREFIX
    rows: List[Tuple[str, str, str, int]] = []
    if not os.path.isdir(directory):
        return rows
    for name in sorted(os.listdir(directory)):
        if not name.startswith(prefix):
            continue
        path = os.path.join(directory, name)
        epoch, pid = "", 0
        try:
            with open(path, errors="replace") as fh:
                state = json.loads(fh.read() or "{}")
            if isinstance(state, dict):
                epoch = str(state.get("epoch", ""))
                pid = int(state.get("publisher_pid", 0) or 0)
        except (OSError, ValueError, TypeError):
            # A torn or foreign file names no boot and no holder.  It is swept
            # as dead residue at launch, never removed as "this boot's".
            epoch, pid = "", 0
        rows.append((path, name, epoch.split(".")[0] if "." in epoch else "", pid))
    return rows


def remove_vram_credit_counters(credit_dir: str = "", *,
                                boot_nonce: str = "") -> List[str]:
    """Unlink THIS boot's per-card VRAM credit counters.  Returns what went.

    FIX 2 round 2, finding 1.  ``vram_credit_path`` is keyed by card UUID and
    nothing else, so the file outlives the boot that wrote it -- and a TERMINAL
    one (``leg_complete`` plus a whole image of credit) is exactly what a later
    boot's flip of the same index read as its own funding while the epoch was
    only the front's flip counter.  Measured on this rig 2026-09-08: three such
    files from boot weg2rg2, one carrying 13912 MiB of credit.

    SCOPED BY THE BOOT NONCE (FIX 3 round 3).  The predecessor unlinked EVERY
    ``.weg2-vram-credit-*`` in the directory, so a teardown of boot A would
    delete a live boot B's counters mid-flip -- and B then waits on a counter
    whose file has gone.  The same commit that introduced the nonce is the one
    that made a counter identifiable; here it is used.  An EMPTY nonce removes
    nothing: "this boot" is then unknown, and the unscoped sweep is precisely
    the defect.  Crash residue is the LAUNCH sweep's job
    (:func:`sweep_dead_credit_counters`), which can check that no holder lives.
    """
    removed: List[str] = []
    if not boot_nonce:
        return removed
    for path, name, boot, _pid in _credit_counter_rows(credit_dir):
        if boot != str(boot_nonce):
            continue
        try:
            os.unlink(path)
        except OSError:
            continue
        removed.append(name)
    return removed


def sweep_dead_credit_counters(log: Log, credit_dir: str = "",
                               dry: bool = False) -> List[str]:
    """Remove credit counters of DEAD boots at LAUNCH.  Never a live holder.

    FIX 3 round 3.  Teardown is exactly what a CRASHED boot does not run, and a
    crash is how three weg2rg2 counters were still on this rig when weg2rg3
    launched.  The composed epoch makes them harmless; this makes them absent,
    and the printed line is the evidence a reader wants instead of an inference.

    The holder test is the file's own ``publisher_pid``: a counter whose
    publisher is ALIVE belongs to a boot in flight, and is left alone and named
    -- the same rule ``presence_sweep`` applies to its own residue.
    """
    rows = _credit_counter_rows(credit_dir)
    if not rows:
        log("WEG2-VRAM-CREDIT residue: none")
        return []
    live = [n for _p, n, _b, pid in rows if pid and _alive(pid)]
    dead = [(p, n, b) for p, n, b, pid in rows if not (pid and _alive(pid))]
    if live:
        log(f"WEG2-VRAM-CREDIT residue: {len(live)} counter(s) held by a LIVE "
            f"publisher, left untouched: {live}")
    if dry:
        log(f"WEG2-VRAM-CREDIT DRY-RUN: would remove {len(dead)} dead-epoch "
            f"counter(s): {[n for _p, n, _b in dead]}")
        return []
    removed: List[str] = []
    for path, name, boot in dead:
        try:
            os.unlink(path)
        except OSError:
            continue
        removed.append(f"{name} (boot {boot or 'pre-token'})")
    log(f"WEG2-VRAM-CREDIT residue swept: {len(removed)} dead-epoch counter(s) "
        f"{removed} -- a crashed boot runs no teardown, so LAUNCH sweeps too")
    return removed


def prepare_host_ring(cards: List[Card], log: Log, tag: str, form: str,
                      evidence_dir: str, boot_stem: str, dry: bool,
                      leg_form: str = "", pcie_directional: Optional[bool] = None,
                      duplex_probe: str = "", tree: str = "",
                      py: str = "",
                      p_argv: Optional[Sequence[str]] = None) -> HostRingPlan:
    """C20 + C18: solve the table, print L6, REFUSE by name, then arm the region.

    Order is load-bearing: the inequalities are checked and the per-card files
    are created BEFORE either group starts, so a configuration that cannot walk
    the corridor never reaches a rank (W32/W34 rather than a mid-flip W31).

    TWO inequalities, and WHICH ONE A CARD IS CHECKED AGAINST IS THE CARD'S OWN
    PROPERTY (FIX 1 round 1).  R5's corridor is checked on every card (W32).
    The SERIAL form's ``H(c) >= image_W(c) + max_tag_S(c)`` is checked on the
    cards where the two legs cannot actually overlap (W34) -- see
    :func:`_serialised_cards`, which reads the leg form from the front and the
    key from the module that owns it, per card.  ``pcie_directional`` is the
    test override for that second fact: ``True`` = every card's key splits,
    ``False`` = none does, ``None`` = ask per card.

    NO TABLE IS NOT A FALLBACK: with ``table`` None this returns an un-armed,
    unpriceable plan and the ledger refuses the launch (W20).  See
    :class:`HostRingPlan`.
    """
    leg_form = leg_form or _front_leg_form()
    # C12/C13 + A1-4: the per-card duplex ratio, SOLVED from the step-0 probe's
    # own lines and published to the ranks as launcher output.  Printed per card
    # BEFORE either group starts, beside H(c), because the split's benefit is
    # per card and the flip's critical path is the card that does not get it.
    duplex, duplex_why = ring_table.solve_duplex(duplex_probe) if duplex_probe else (None, "no --duplex-probe given")
    plan = HostRingPlan(form=form, duplex=duplex)
    if duplex is None:
        plan.lines.append(
            f"WEG2-PCIE-DUPLEX UNMEASURED: {duplex_why} -- no card has a RATIO "
            "this boot, so none of them has an expectation about what the split "
            "buys.  Under gathered legs every key still SPLITS, and that is a "
            "correctness decision, not an optimisation: with one key the sleep "
            "leg blocks in the ring's acquire holding the key the wake leg needs "
            "(boot weg2rg2, W31 after 120.189 s -> W29 -> W4).  The predecessor "
            "of this line said an unmeasured card is 'slower, never wrong'; the "
            "metal refuted that."
        )
    else:
        plan.lines.extend(duplex.format_lines(cards, DUPLEX_GATE))
    for ln in plan.lines:
        log(ln)
    logged = len(plan.lines)
    table, reason = ring_table.solve(cards, evidence_dir, boot_stem or None, p_argv=p_argv)
    if table is None:
        plan.lines.append(
            "WEG2-HOST-RING R22: no measured per-card byte table -- " + reason +
            ".  The planner REFUSES to guess H, and THIS BOOT REFUSES: the OLD "
            "flip form is not a fallback here, because its own host charge (one "
            "image plus one tag in flight) is solved from the same table, so the "
            "ledger has no host weights term and stops the launch by name (W20 "
            "Weg2HostLedgerRefused, exit 2) before either group starts.  What "
            "makes a boot priceable again is a predecessor boot in "
            + (evidence_dir or "the evidence dir") +
            " that logged WEG2-CHUNK-BYTES / WEG2-FLIP-TAG lines and its own "
            "NVML -> CUDA ordinal map, or --ring-table-boot naming one."
        )
        for ln in plan.lines[logged:]:
            log(ln)
        return plan
    plan.table = table
    # FIX 3 round 3: every NEWER boot the solver skipped, with its reason, one
    # line each -- on SUCCESS, not only on failure.  Without them the log shows
    # a well-formed table and no way to ask why that boot and not the newest.
    # #1264 (B): PRINT THE CHOSEN SOURCE, not only the rejected ones.  The
    # predecessor took `reason.split("\n")[1:]`, i.e. it kept every SKIPPED line
    # and threw away line 0 -- the one that names the boot the table was
    # actually solved from, the form it was measured in, and (since #1264)
    # which boot's dormant sample was joined to it.  So a launch log could say
    # why four boots were NOT used and never say which one was, and the weg2t2b
    # ring drift had to be reconstructed from the ledger's arithmetic instead of
    # read off.  Line 0 first, then the skipped ones.
    _reason_lines = reason.split("\n")
    if _reason_lines and _reason_lines[0].strip():
        plan.lines.append("WEG2-HOST-RING SOURCE " + _reason_lines[0].strip())
    plan.lines.extend(ln for ln in _reason_lines[1:] if ln.strip())
    plan.lines.extend(table.format_l6())          # L6
    for ln in plan.lines[logged:]:
        log(ln)
    if form not in ("auto", "MAP_SHARED"):
        head = (
            f"W33 Weg2RingFormUnproven: --ring-form {form or 'none'} selects no "
            "registration form, so the host ring cannot be armed -- AND THERE IS NO "
            "FALLBACK TO RUN INSTEAD (spec Amendment A1-3: the OLD flip form is "
            "INFEASIBLE on this host budget, W20 at every rung on the default "
            "table, boot weg2rg1).  This boot REFUSES, exit 2, before either group "
            "starts.  MAP_SHARED (cudaHostRegister on a /dev/shm MAP_SHARED file) "
            "is the form the step-0 probe PROVED on this rig "
            "(WEG2_BUILD_DECISIONS_0906 section 1p, 2026-09-07T23:13:23Z: all three "
            "gates pass, verdict BUILD-MAP_SHARED, duplex arm KEEP); it is the only "
            "form built, and the memfd candidate that probe retired is deleted."
        )
        plan.lines.append(head)
        log(head)
        raise Weg2RingFormUnproven(head)
    # TWO launch checks, ONE rule for what a failure does: the ring cannot be
    # armed, and per Amendment A1-3 there is nothing to fall back to, so the
    # launcher REFUSES BY NAME and exits 2 -- whether or not a form was asked
    # for explicitly.  The predecessor downgraded silently to "the OLD flip form
    # runs (spec section 10.4)" unless --ring-form named a form; boot weg2rg1
    # then died of W20 two steps later, having told the operator a fallback had
    # run that never did.
    #
    # W32 is spec R5's corridor inequality -- the one that makes a blocking ring
    # deadlock-free when the legs are GATHERED (spec C9).  It is checked always.
    checks = [
        # L6 AGAINST AN INDEPENDENT SOURCE (train fix 5).  Listed FIRST because
        # it is the only one of the three that does not read the census that
        # sized the ring: it compares span1 against the weight bytes the
        # checkpoint says this boot's stages will load.  boot weg2tr2 passed
        # every check below it and then died at its first sleep with the runtime
        # saying "the launch check (L6) was violated" -- because L6 was checking
        # a premise against itself.  A stale table now stops here, by name,
        # before either group starts.
        ("W49 Weg2RingWeightsUnderSized",
         "span1 is below the checkpoint's own weight bytes for that PP stage on "
         "{n} card(s) -- the ring cannot hold what the stage loads, so its FIRST "
         "pause blocks and the group dies on the acquire budget, which is boot "
         "weg2tr2 exactly",
         table.ckpt_refusals(),
         ring_table.Weg2RingWeightsUnderSized),
        ("W32 Weg2RingCreditRefused",
         "the R5 corridor inequality (H(c) >= image_W(c) - device_credit(c) + "
         "max_tag_S(c) + max_tag_W(c), spec C20) fails on {n} (card x direction) "
         "case(s); a negative slack is a flip that would wedge, not one that "
         "would be slow",
         table.refusals(),
         ring_table.Weg2RingCreditRefused),
    ]
    # W34 IS PER CARD (FIX 1 round 1).  The serial requirement
    # H(c) >= image_W(c) + max_tag_S(c) applies to a card wherever the two legs
    # cannot actually overlap ON THAT CARD, and there are two ways for that to
    # be true:
    #
    #   * the FRONT does not gather at all (``FLIP_LEG_FORM != "interleave"``,
    #     which no state of this tree produces -- a stale worktree on
    #     PYTHONPATH, a partial rebase).  Then every card is serialised.
    #   * the CARD's key does not split (A1-4: 5090 1.759 and nvml2 1.687 reach
    #     R17's gate, nvml0 measured 1.316 and does not).  Each leg holds that
    #     key across its whole tag loop, so on that card the gathered pair is
    #     serialised again and R5's premise -- W's per-tag releases fund S's
    #     acquires -- is false there.
    #
    # The predecessor checked only the first and printed the second, and boot
    # weg2rg2 is what that cost: nvml0 passed R5 on both directions, the L6 line
    # said SERIAL FORM slack=1122/-2986 MiB on the SAME card, the launcher armed
    # anyway, and the first flip wedged for 120.189 s into W31 -> W29 on all
    # three P ranks -> group-fatal W4.  A refusal 120 s into a flip that has
    # already mutated VRAM is not the refusal this gate exists to give.
    card_uuids = [c.uuid for c in table.cards]
    splits = _split_decisions(
        card_uuids, duplex.ratios if duplex else {}, leg_form, pcie_directional
    )
    # THE DECISION IS PUBLISHED HERE AND NOWHERE ELSE (R19: launcher output).
    # The ranks build their key from this same string, so the inequality this
    # module checks and the key the metal takes cannot be two different facts.
    # THE FORMAT NAMES ITS VERSION IN A ROW OF ITS OWN (FIX 3 round 3, finding
    # 1).  A reader that knows the key refuses by name (W36) on an absent or
    # unknown version instead of resolving a single key on every card; a reader
    # that does NOT know it drops that one row on float() and still resolves
    # every real card.  FIX 2's "v2|" PREFIX had neither property -- it mangled
    # exactly the first card's uuid, which on this rig is the card carrying the
    # co-located pair, and was therefore a regression against the un-versioned
    # string it replaced.
    from sglang.srt.managers import weg2_memory_saver as _saver
    plan.duplex_env = ",".join(
        [f"{_saver.PCIE_DUPLEX_VERSION_KEY}={_saver.PCIE_DUPLEX_FORMAT}"]
        + [
            f"{u}={('%.3f' % duplex.ratios[u]) if duplex and u in duplex.ratios else ''}"
            f":{'split' if splits[u] else 'single'}"
            for u in card_uuids
        ]
    )
    probe_notes: List[str] = []
    serialised = _serialised_cards(card_uuids, plan.duplex_env, leg_form,
                                   py=py, tree=tree, notes=probe_notes)
    plan.lines.extend(probe_notes)
    for ln in probe_notes:
        log(ln)
    checks_logged = len(plan.lines)
    for c in table.cards:
        is_serial = c.uuid in serialised
        plan.lines.append(
            f"WEG2-HOST-RING CHECK card={c.uuid} nvml{c.nvml_index} {c.name} "
            f"key={'SINGLE (both legs serialise here)' if is_serial else 'SPLIT per direction'} "
            f"duplex_ratio={(duplex.ratios.get(c.uuid) if duplex else None)} "
            f"gate={DUPLEX_GATE} leg_form={leg_form} -- checked against "
            + ("the R5 corridor (W32) AND the SERIAL requirement "
               "H >= image_W + max_tag_S (W34), because the two legs cannot "
               "overlap on this card"
               if is_serial else
               "the R5 corridor (W32) alone, because both legs can be in flight "
               "on this card at once")
        )
        # #1284: SAY WHAT THE WAIVER COSTS, per card, at launch.
        #
        # boot weg2sb5e armed with `serialised_cards=none` and printed
        # `SERIAL FORM ... slack=-1608/-2509`, `-881/-2986`, `-982/-2622` on
        # its three L6 rows -- the serial requirement failed on EVERY card and
        # the waiver was silent about it, because the CHECK line above says
        # only WHICH inequality was used, never what the unused one would have
        # demanded. Thirty flips later the wake leg did not run, the sleeping
        # leg met exactly that shortfall, and the C++ acquire burned its full
        # 110 s budget before dying with a line that blames L6.
        #
        # The debt is stated, not funded: paying it costs +8117 MiB of shmem on
        # these numbers and the host ledger's chosen arm is already ON its
        # store floor, so funding it would trade this wedge for a W21 refusal.
        # What covers it is the runtime guard named on the line.
        if not is_serial and c.serial_debt_mib > 0:
            plan.lines.append(
                f"WEG2-HOST-RING SERIAL-DEBT card={c.uuid} nvml{c.nvml_index} "
                f"debt_mib={c.serial_debt_mib} H={c.h_mib} "
                f"need_serial_d2p={c.need_serial_d2p_mib} "
                f"need_serial_p2d={c.need_serial_p2d_mib} -- this card is armed "
                "under the INTERLEAVE waiver, so R5's corridor was checked and "
                "the serial requirement H >= image_W + max_tag_S was NOT. If "
                "the two legs ever fail to overlap on this card the ring is "
                f"short by AT LEAST {c.serial_debt_mib} MiB and the sleeping "
                "leg cannot complete however long it waits -- at least, because "
                "image_W + max_tag_S prices a serial order in which S places "
                "only its largest tag, while a peer that releases NOTHING makes "
                "the requirement image_W + image_S, which no ring this box can "
                "fund. R5's premise is a RACE, not an "
                "invariant -- the front's own gather carries no happens-before "
                "between the sleeping leg's first acquire and the waking leg's "
                "first release. COVERED AT RUNTIME by W51 Weg2HostRingUnfunded "
                "(weg2/ring_guard.py), which emits the WEG2-RING NEED series per "
                "tag and refuses in ~2 s when free stops rising, instead of "
                "wedging for the C++ acquire budget (weg2sb5e: 110 s). A "
                "non-zero debt here with no W51 in the sleeping leg is weg2sb5e."
            )
    for ln in plan.lines[checks_logged:]:
        log(ln)
    if serialised:
        why_serial = (
            f"front.FLIP_LEG_FORM is {leg_form!r}, not 'interleave' -- this "
            "launcher is running against a front that does not gather its legs, "
            "which no state of this tree produces (C9), so EVERY card is "
            "serialised"
            if leg_form != "interleave" else
            f"the legs are gathered (C9) but on {len(serialised)} of "
            f"{len(table.cards)} card(s) -- {', '.join(serialised)} -- the "
            "measured duplex ratio does not reach R17's gate, so a sleep-D2H and "
            "the co-located wake-H2D take the SAME per-card PCIe key and each "
            "holds it across its whole tag loop; on those cards the pair is "
            "serialised whatever the front does"
        )
        checks.append(
            ("W34 Weg2RingNeedsInterleave",
             why_serial + "; the serial requirement H(c) >= image_W(c) + "
             "max_tag_S(c) therefore applies THERE and fails on {n} "
             "(card x direction) case(s).  S blocks in acquire holding the "
             "per-card key, W cannot issue the release that would fund it "
             "because it is queued on that same key, and the acquire budget "
             "expires into W31 -> W29 on every rank of the group -> "
             "group-fatal W4.  That is boot weg2rg2, observed, not predicted",
             table.serial_refusals(only=serialised),
             ring_table.Weg2RingNeedsInterleave))
    for name, why, bad, exc in checks:
        if not bad:
            continue
        head = (f"{name}: " + why.format(n=len(bad)) +
                ".  The host ring is NOT armed, and per spec Amendment A1-3 "
                "there is NO fallback flip form to run on this host budget "
                "(the OLD form refuses W20 at every rung on the default table, "
                "boot weg2rg1) -- so this boot REFUSES by name and exits 2, "
                "BEFORE either group starts.")
        plan.lines.append(head)
        plan.lines.extend(bad)
        for ln in plan.lines[-(len(bad) + 1):]:
            log(ln)
        raise exc(head + "\n" + "\n".join(bad))
    plan.epoch = int(time.time())
    plan.dir = f"{HOST_RING_DIR}-{tag}"
    if not dry:
        os.makedirs(plan.dir, exist_ok=True)
    for c in table.cards:
        path = os.path.join(plan.dir, f"{c.uuid}.ring")
        size = ring_table.MIB * 2 + c.h_mib * ring_table.MIB  # header granule + data
        if not dry:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.ftruncate(fd, size)
            finally:
                os.close(fd)
        plan.lines.append(
            f"WEG2-HOST-RING file {path} size={size // ring_table.MIB} MiB "
            f"(2 MiB header granule + H {c.h_mib} MiB) "
            + ("DRY-RUN: would be" if dry else "")
            + " created and ftruncated BEFORE either group starts"
        )
    plan.env_map = table.env_map()
    plan.form = "MAP_SHARED"          # 'auto' resolves to the one proven form
    plan.armed = True
    plan.lines.append(
        f"WEG2-HOST-RING ARMED form={plan.form} leg_form={leg_form} "
        f"serialised_cards={serialised or 'none'} "
        # #1284: this parenthetical used to read "each checked against the
        # SERIAL requirement, not R5's corridor" UNCONDITIONALLY, which is the
        # exact opposite of what the per-card CHECK lines say whenever
        # `serialised` is empty -- and it was empty on weg2sb5e, the boot that
        # then died of the un-checked serial requirement. A summary line that
        # contradicts its own detail lines sends every reader to the wrong
        # half of the arithmetic. It now states which inequality was used and
        # what debt that leaves.
        + (f"(SERIAL requirement H >= image_W + max_tag_S checked on "
           f"{len(serialised)}/{len(table.cards)} card(s); the rest were armed "
           f"on R5's corridor alone, carrying "
           f"{sum(c.serial_debt_mib for c in table.cards if c.uuid not in serialised)}"
           f" MiB of SERIAL-DEBT covered at runtime by W51 -- see the "
           f"WEG2-HOST-RING CHECK and SERIAL-DEBT lines per card) "
           if len(serialised) < len(table.cards) else
           "(SERIAL requirement H >= image_W + max_tag_S checked on every card "
           "-- see the WEG2-HOST-RING CHECK line per card) ")
        + f"epoch={plan.epoch} dir={plan.dir} "
        f"Sigma H={table.total_h_bytes // ring_table.MIB} MiB "
        f"Sigma span1={table.total_span1_bytes // ring_table.MIB} MiB granule=2 MiB "
        f"register_est_ms={int(table.total_span1_bytes / float(2**30) * 44)} "
        "(38-49 ms/GiB measured on the step-0 probe at ONE size point, 4 GiB -- "
        "an extrapolation, and it is a LAUNCH-moment charge: span 1 is "
        "registered at P's first pause, R7) "
        f"-- provenance: {table.provenance()}"
    )
    for ln in plan.lines[-(len(table.cards) + 1):]:
        log(ln)
    return plan


def _env_knobs(ns) -> Dict[str, object]:
    """The #1235 flags ``build_env`` takes, gathered once.

    Three call sites build an environment and every one of them must pass the
    same seven values; spelling them out three times is how the pair of
    environments would drift the day an eighth arrives.
    """
    return {
        "barlink_build_window_cap_s": ns.barlink_build_window_cap_s,
        "pp_chain_recv_stall_s": ns.pp_chain_recv_stall_s,
        "pp_occupant_horizon_s": ns.pp_occupant_horizon_s,
        "match_refusal_census_every": ns.match_refusal_census_every,
        "arming_floor_solved": ns.arming_floor_solved,
        "hicache_bigram_keys": ns.hicache_bigram_keys,
        "hicache_flush_publish_sweep": ns.hicache_flush_publish_sweep,
    }


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
              group: str = "") -> Dict[str, str]:
    env = dict(os.environ)
    # FIX 2, finding 1: WHICH WEG-2 GROUP THIS RANK BELONGS TO, and the only
    # thing in either tree that says so.  Read by
    # `weg2_memory_saver.weg2_group_name()`; it is the discriminator the
    # graph-tag coupling gates on, because the alternative it used first
    # (`server_args.enable_memory_saver`) is an UPSTREAM flag that upstream
    # engines set too -- gating a Weg-2-only widening of an upstream RPC on it
    # changed stock `/release_memory_occupation` semantics for everyone.
    #
    # `server_args.weg2_group` was believed to be this fact and is not: it is
    # read in exactly one place (`weight_updater._weg2_group_name`) with a
    # `getattr(..., "")` default and is ASSIGNED NOWHERE, so it has answered
    # "?" on every rank of every boot.
    #
    # Same discipline as the ring family above: published only when this call
    # names a group, and POPPED otherwise, so a value inherited from the
    # operator's own shell can never arm a coupling nobody asked for.
    if group:
        env["SGLANG_WEG2_GROUP"] = group
    else:
        env.pop("SGLANG_WEG2_GROUP", None)
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
        env["SGLANG_WEG2_PCIE_DUPLEX"] = ring.duplex_env
    else:
        env.pop("SGLANG_WEG2_PCIE_DUPLEX", None)
    # #1233 one-backup flip: chunked weights tags (weg2_memory_saver.py) and
    # the patched torch_memory_saver preload hook (tms_csrc/PATCH.md).
    if chunk_layers > 0 and chunk_count > 0:
        env["SGLANG_WEG2_WEIGHT_CHUNK_LAYERS"] = str(chunk_layers)
        env["SGLANG_WEG2_WEIGHT_CHUNKS"] = str(chunk_count)
    if tms_so:
        env["SGLANG_WEG2_TMS_PRELOAD_SO"] = tms_so
    cu13 = f"{venv}/lib/python3.12/site-packages/nvidia/cu13/lib"
    # boot_855_train0901.sh:152-153 (NVRTC) and S1 killer K1: the memory
    # saver's cu13 preload hook links libcudart.so.13, which must be on the
    # loader path at LD_PRELOAD time or every rank dies exit 127 before main().
    env["LD_LIBRARY_PATH"] = cu13 + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = f"{tree}/python"
    env["CUDA_VISIBLE_DEVICES"] = cvd
    env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "0"  # K2: pure liveness /health
    # #1233 boot weg2ls3b3: group D (NEXTN) keys the store by BIGRAM page
    # hashes, group P (no spec) by UNIGRAM -- disjoint chains for the same
    # prompt, D never read P's pages. One key scheme for both groups; a
    # no-op on D (already bigram), forces bigram on P.
    if hicache_bigram_keys:
        env["SGLANG_HICACHE_BIGRAM_KEYS"] = "1"
    else:
        env.pop("SGLANG_HICACHE_BIGRAM_KEYS", None)
    # #1233 zero-remainder: /flush_cache (the front's quiesce before a sleep)
    # first publishes every un-backed device node to the store, so a chain
    # the write-through pin budget declined mid-prefill is not lost at the
    # flip (UnifiedRadixCache.publish_unbacked_sweep). Both groups.
    if hicache_flush_publish_sweep:
        env["SGLANG_HICACHE_FLUSH_PUBLISH_SWEEP"] = "1"
    else:
        env.pop("SGLANG_HICACHE_FLUSH_PUBLISH_SWEEP", None)
    if arming_floor_solved:
        env["SGLANG_ARMING_FLOOR_SOLVED"] = "1"
    else:
        env.pop("SGLANG_ARMING_FLOOR_SOLVED", None)
    # Item `dormant` (record [1y]): the ONE switch that puts the CUDA-graph
    # capture pool and the flashinfer FLOAT workspace inside the memory saver's
    # `cuda_graph` region, so the sleeping group hands their physical pages
    # back and the wake remaps them at stable virtual addresses -- no
    # recapture, which RESTORE-NEVER-REBUILD forbids inside a cutover.
    # Upstream's own flag, read at the CAPTURE site
    # (full_cuda_graph_backend.py:78-81) and, since that slice's commit 2, at
    # the SLEEP site (weg2_memory_saver.weg2_graph_tag_armed); both groups get
    # it so the two sides can never disagree.  Its fix 2: this flag ALONE no
    # longer arms the sleep-side coupling -- SGLANG_WEG2_GROUP above is the
    # Weg-2 conjunct, so setting this variable on a stock engine gets stock
    # behaviour, which is what it always meant upstream.  Expected release per
    # rank from [1y]: 384 MiB workspace + 92-133 MiB capture pool, measured on
    # the boot by `WEG2-SLEEP released tags=['cuda_graph'] mib=`.
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
    env["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = env.get(
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH", "1"
    )
    # THE DOUBLY-STATED FACTS, FROM THE SAME TABLE THE ARGV IS BUILT FROM
    # (#1235). Not second bookkeeping: ServerArgs reads these keys off
    # os.environ before the matching flags publish themselves -- see
    # EarlyReadFact for the file:line ordering. Written for BOTH groups, which
    # preserves an existing asymmetry the table records rather than hides.
    # (The dormant slice restated SGLANG_UNEVEN_DCP / _WEIGHTED /
    # SGLANG_MAMBA_SSM_DTYPE here by hand; that is the pre-#1235 form of the
    # same three facts and this loop is the one that survives.)
    for fact in EARLY_READ_FACTS:
        env[fact.env_key] = fact.env_value
    env["SGLANG_BARLINK_BUILD_WINDOW_CAP_S"] = str(barlink_build_window_cap_s)
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
    env["SGLANG_PP_CHAIN_RECV_STALL_S"] = str(pp_chain_recv_stall_s)
    env["SGLANG_PP_OCCUPANT_HORIZON_S"] = str(pp_occupant_horizon_s)
    env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = store_dir
    env["SGLANG_MATCH_REFUSAL_CENSUS_EVERY"] = str(match_refusal_census_every)
    for k in list(env):
        if k.startswith("SGLANG_PHASE_FLIP"):
            del env[k]
    alloc = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in alloc:
        raise Weg2LaunchRefused(
            f"PYTORCH_CUDA_ALLOC_CONF={alloc!r} -- torch_memory_saver refuses expandable_segments (entrypoint.py _sanity_checks)"
        )
    env.pop("SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION", None)
    if debug_hold:
        env["SGLANG_DEBUG_HOLD"] = "1"
        env["SGLANG_DEBUG_HOLD_S"] = env.get("SGLANG_DEBUG_HOLD_S", "1800")
        env["SGLANG_DEBUG_HOLD_PORT_BASE"] = env.get("SGLANG_DEBUG_HOLD_PORT_BASE", "5000")
        env["SGLANG_DEBUG_HOLD_DIR"] = env.get("SGLANG_DEBUG_HOLD_DIR", f"{GPU_ARB}/debug_hold")
        env["SGLANG_DEBUG_HOLD_TAG"] = tag
        os.makedirs(env["SGLANG_DEBUG_HOLD_DIR"], exist_ok=True)
    # #1269/#1276: THE IDLE FAMILY, published to BOTH groups with provenance.
    #
    # SGLANG_IDLE_BLOCKING_POLL -- boot weg2sb4 set neither this nor
    # --sleep-on-idle, so Scheduler.init_idle_sleeper built no sleeper on any
    # of the six ranks and both loops spun: group P 1,499.8 scheduler
    # rounds/s, group D 303.1/s, 34 idle minutes, empty queue, live front.
    # The serve tree now also arms on SGLANG_WEG2_GROUP, so this is the
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
    # live count. WEG2-IDLE-CENSUS prints `threads=` for exactly this reason;
    # the next boot confirms or corrects the value.
    env.setdefault("SGLANG_IDLE_BLOCKING_POLL", "1")
    env.setdefault("MALLOC_ARENA_MAX", "4")
    return env


_DRAFTER_RE = re.compile(r"HiCache draft KV registered: .*drafter=([0-9a-f]{16}|None)")
_LAYOUT_RE = re.compile(r"#706 canonical DRAFT page active: .*layout=v(\d+) drafter=([0-9a-f]{16}|None)")


def check_drafter_identity(log_p: str, log_d: str) -> Dict[str, object]:
    """W10 (#1233): the drafter identity and canonical draft layout of both
    groups, read from their logs. ``match`` is False on ANY disagreement or
    absence -- a group that registered no drafter at all is the pre-fix
    shape, not a pass."""
    out: Dict[str, object] = {}
    for name, path in (("P", log_p), ("D", log_d)):
        ids, layouts, n = set(), set(), 0
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    m = _DRAFTER_RE.search(line)
                    if m:
                        ids.add(m.group(1))
                        n += 1
                    m = _LAYOUT_RE.search(line)
                    if m:
                        layouts.add(f"v{m.group(1)}")
        except OSError:
            pass
        out[name] = ",".join(sorted(ids)) if ids else None
        out[f"layout_{name}"] = ",".join(sorted(layouts)) if layouts else None
        out[f"n_{name}"] = n
    out["match"] = bool(
        out["P"] and out["D"] and out["P"] == out["D"] and "," not in str(out["P"])
        and out["layout_P"] and out["layout_P"] == out["layout_D"]
    )
    return out


_RESIDENT_RE = re.compile(r"WEG2 DRAFT-KV-PRODUCER armed .*resident_mib=(-?\d+(?:\.\d+)?)")
_HEAD_RELEASED_RE = re.compile(r"WEG2 DRAFT-KV-PRODUCER armed .*head_released_mib=(-?\d+(?:\.\d+)?)")
_NVML_DELTA_RE = re.compile(r"WEG2 DRAFT-KV-PRODUCER armed .*nvml_delta_mib=(-?\d+(?:\.\d+)?)")
#: W11b (#1233 fix 6): how far the BUILD may stay unexplained by the two terms
#: that claim to explain it, IN EITHER DIRECTION.  MEASURED on boot weg2dk5's
#: own L2 line: nvml_delta 3998.0 against resident 1682.9 + head_released
#: 2425.0 = 4107.9, i.e. -109.9 MiB, not attributable at this granularity (the
#: driver's own allocation rounding, and the fact that the two sides are read
#: at different instants of the build).  The bound is set at the same 256 MiB
#: as the residue tolerance -- the next tighter measurement replaces it.
#:
#: WHICH SIDE HAS BEEN MEASURED, stated rather than implied (#1233 fix 7): the
#: single calibration sample is NEGATIVE, i.e. the accounting OVER-claims by
#: 109.9 MiB, while the shape this gate hunts -- a table freed from the graph
#: only, so the build is larger than residue + release explain -- is POSITIVE
#: and has NEVER been measured on this box.  The gate can still catch it (a
#: whole unfreed lm_head is ~2425 MiB, an order of magnitude over the bound),
#: but the bound's own calibration comes from the other side, and a boot that
#: lands in 256..~2400 MiB positive would be graded by an extrapolation.  Both
#: directions are refusals, and both are pinned.
P_DRAFT_BUILD_ACCOUNTING_TOL_MIB = 256.0


def check_draft_resident(log_p: str, budget_mib: float = None, tol_mib: float = None) -> Dict[str, object]:
    """W11 (#1233 fix 2, second instrument added in fix 6): the producer's
    build on P's last stage, graded by TWO independent readings of the L2 line.

    1. ``resident_mib`` against the budget T_P was derived from.  This is a
       model-GRAPH quantity (fix 3 made it ``_live_weight_mib``), which is what
       it must be for the corridor derivation -- and is exactly why it can no
       longer catch the fix-2 failure it was built for: a table that
       ``_drop_parameters`` unbinds while a loader, quant method or backup path
       still holds a reference LEAVES ``model.parameters()`` while staying on
       the card, and this reading falls even though nothing was freed.
    2. THE BUILD ACCOUNTING: ``nvml_delta_mib`` -- which under
       --enable-memory-saver reads the BUILD, not the residue (the weights live
       in the saver's MemPool, memsaver.md N4) -- must be explained by
       ``resident_mib + head_released_mib`` within
       :data:`P_DRAFT_BUILD_ACCOUNTING_TOL_MIB`.  A release that frees nothing
       shows up here as an unaccounted remainder the size of the table, on a
       quantity no graph edit can move.

    ``ok`` requires both.  It is False on absence and on an unmeasured value
    (-1) in either instrument: silence is not a pass, and a build nobody
    measured is not a build anybody checked.
    """
    budget = P_DRAFT_RESIDENT_BUDGET_MIB if budget_mib is None else float(budget_mib)
    tol = P_DRAFT_RESIDENT_TOL_MIB if tol_mib is None else float(tol_mib)
    out: Dict[str, object] = {
        "resident_mib": None, "budget_mib": budget, "tol_mib": tol, "over_mib": None,
        "head_released_mib": None, "nvml_delta_mib": None, "unaccounted_mib": None,
        "accounting_tol_mib": P_DRAFT_BUILD_ACCOUNTING_TOL_MIB,
        "resident_ok": False, "accounted": False, "ok": False,
    }
    try:
        with open(log_p, errors="replace") as f:
            for line in f:
                for key, rx in (
                    ("resident_mib", _RESIDENT_RE),
                    ("head_released_mib", _HEAD_RELEASED_RE),
                    ("nvml_delta_mib", _NVML_DELTA_RE),
                ):
                    m = rx.search(line)
                    if m:
                        out[key] = float(m.group(1))
    except OSError:
        pass
    r = out["resident_mib"]
    if r is not None:
        out["over_mib"] = r - budget
        out["resident_ok"] = r >= 0 and r <= budget + tol
    released, delta = out["head_released_mib"], out["nvml_delta_mib"]
    if r is not None and r >= 0 and released is not None and released >= 0 and delta is not None and delta >= 0:
        # SIGN, both directions named because both are refused (fix 7):
        #   POSITIVE = the build is LARGER than residue + release explain, i.e.
        #     VRAM is held that nobody accounts for -- the fix-2 shape this
        #     gate exists for (a table unbound from the graph, never freed).
        #   NEGATIVE = the accounting OVER-claims: the two explaining terms
        #     together are larger than the build the driver reports, so at
        #     least one of them is measuring something the build did not do
        #     (double-counted release, or the two sides read at instants far
        #     enough apart to disagree).  weg2dk5's -109.9 is this side.
        # Neither direction is a pass: an explanation that does not add up is
        # not an explanation, whichever way it fails to add up.
        out["unaccounted_mib"] = delta - (r + released)
        out["accounted"] = abs(out["unaccounted_mib"]) <= P_DRAFT_BUILD_ACCOUNTING_TOL_MIB
    out["ok"] = bool(out["resident_ok"] and out["accounted"])
    return out


def gate_w11(log_p: str, log: Log) -> Dict[str, object]:
    """THE LAUNCHER'S W11 GATE, both instruments and both refusals.

    Grades group P's last-stage draft build from its log (see
    :func:`check_draft_resident`) and REFUSES the boot before the front opens
    when either instrument says no:

    * ``W11 Weg2DraftResidentOverBudget`` -- the residue is over the budget
      ``T_P`` was derived from, so the corridor derivation is refuted by this
      boot (weg2dk2's 3994 MiB build).
    * ``W11b Weg2DraftBuildUnaccounted`` -- the BUILD is not explained by
      residue + release, the fix-2 shape ``resident_mib`` is blind to.

    #1233 fix 7: a function, for the same reason as :func:`choose_host_ledger`
    -- the fix-6 review gated the W11b refusal off and the whole slice stayed
    green, because the only caller sits behind two launched servers.  A gate
    nothing can reach is a gate nothing can pin.
    """
    w11 = check_draft_resident(log_p)
    log(f"W11 DRAFT-RESIDENT P last stage resident_mib={w11['resident_mib']} budget_mib={w11['budget_mib']:.1f} "
        f"tol_mib={w11['tol_mib']:.0f} over_mib={w11['over_mib']} resident_ok={w11['resident_ok']} "
        f"| W11b BUILD-ACCOUNTING nvml_delta_mib={w11['nvml_delta_mib']} = resident_mib + "
        f"head_released_mib={w11['head_released_mib']} + unaccounted_mib={w11['unaccounted_mib']} "
        f"(tol {w11['accounting_tol_mib']:.0f}) accounted={w11['accounted']} "
        f"ok={w11['ok']} (T_P={P_MAX_TOTAL_TOKENS})")
    if not w11["resident_ok"]:
        raise Weg2LaunchRefused(f"W11 Weg2DraftResidentOverBudget: measured resident_mib={w11['resident_mib']} vs budget "
                                f"{w11['budget_mib']:.1f}+{w11['tol_mib']:.0f} MiB -- T_P={P_MAX_TOTAL_TOKENS} was derived for the "
                                f"budget and the last stage's corridor target (idle NVML free at {P_CORRIDOR_TOP_MIB:.0f}) cannot hold")
    if not w11["accounted"]:
        # #1233 fix 6: the SECOND instrument. resident_mib is a model-graph
        # quantity, so a table released only from the graph passes it while
        # still sitting on the card -- the fix-2 failure the W11 gate exists
        # for. The build must be explained by residue + released.
        raise Weg2LaunchRefused(f"W11b Weg2DraftBuildUnaccounted: nvml_delta_mib={w11['nvml_delta_mib']} is not explained by "
                                f"resident_mib={w11['resident_mib']} + head_released_mib={w11['head_released_mib']} "
                                f"(unaccounted {w11['unaccounted_mib']} MiB, tolerance {w11['accounting_tol_mib']:.0f}) -- either the "
                                f"never-loaded lm_head was unbound from the graph without being freed (fix 2's shape, invisible to "
                                f"resident_mib by construction) or one of the three instruments did not measure")
    return w11


def measured_record_path() -> str:
    """The sidecar this line writes its own dormant-image measurements into."""
    return f"{EVIDENCE_DIR}/{host_ledger.MEASURED_RECORD_NAME}"


def choose_host_ledger(
    store_min_gib: float,
    ring_bytes: int,
    ring_span1_bytes: int,
    ring_provenance: str = "",
    meminfo_path: str = "/proc/meminfo",
    cgroup_root: str = "/sys/fs/cgroup",
    record_path: Optional[str] = None,
) -> Tuple[host_ledger.Arm, float, List[str], Dict[str, Optional[int]]]:
    """THE LAUNCHER'S ONE LEDGER CALL SITE: read the host, price the ladder.

    Returns ``(arm, store_gib, printed lines, the cgroup reading)`` or raises
    :class:`host_ledger.Weg2HostLedgerRefused` -- ``main`` only logs the lines
    and carries the reading into the boot state.

    #1233 fix 5: the reaper watches the CGROUP, so the ledger reads it.  The
    ``oom_kill`` value goes in as the boot's PRE-BOOT BASELINE of a cumulative,
    timestamp-free counter -- no boot starts without one.

    #1233 fix 7: this is a FUNCTION and not four lines inside ``main`` because
    it is the only wire that carries fix 6 into a boot, and inside ``main`` --
    behind NVML, a tmpfs mount and two servers -- nothing could reach it.  The
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
    cg = host_ledger.read_cgroup(cgroup_root)
    cg_ceiling, cg_ceiling_source = host_ledger.resolve_cg_ceiling(cg, mi["MemTotal"])
    # fix 8: this line's OWN previous measurements of the dormant image and of
    # the run-moment residual.  Absent (first boot, or a wiped sidecar) the
    # ledger prices the named dk7 reading and prints that it is another boot's.
    record = host_ledger.read_measured_record(
        measured_record_path() if record_path is None else record_path
    )
    arm, store_gib, lines = host_ledger.choose(
        mi["MemTotal"],
        mi["MemAvailable"],
        store_min_gib=store_min_gib,
        ring_bytes=ring_bytes,
        ring_span1_bytes=ring_span1_bytes,
        ring_provenance=ring_provenance,
        cg_current_bytes=cg["current"],
        # fix 6: only the NON-reclaimable part of that reading is charged --
        # page cache is what the kernel hands back instead of killing for.
        reclaimable_bytes=cg["reclaimable"],
        # train fix 3: the LIVE slab term the reap watermark's own row lacked.
        # The store's reap bound subtracts it; passing None here reverts the
        # bound to the optimistic form and says so on every ARM line.
        slab_reclaimable_bytes=cg["slab_reclaimable"],
        cg_ceiling_bytes=cg_ceiling,
        cg_ceiling_source=cg_ceiling_source,
        cg_oom_kill=cg["oom_kill"],
        measured_record=record,
    )
    return arm, store_gib, lines, cg


def count_marker(path: str, marker: str) -> int:
    n = 0
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if marker in line:
                    n += 1
    except FileNotFoundError:
        pass
    return n


# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------


def launch_group(spec: GroupSpec, tree: str, log: Log, dry: bool) -> None:
    # #1275: the LOG copy is redacted; spec.argv itself is untouched and is what
    # is actually exec'd. A log file outlives the boot and gets pasted into
    # records -- a strictly worse channel than /proc, which is inherent.
    log(f"group {spec.name} argv: " + " ".join(
        shlex.quote(a) for a in admin_key_mod.redact_argv(spec.argv)))
    if dry:
        return
    fh = open(spec.log, "ab")
    _logged_argv = ' '.join(shlex.quote(a) for a in admin_key_mod.redact_argv(spec.argv))
    fh.write(f"=== WEG2 group {spec.name} launched {_now()} ===\nargv: {_logged_argv}\n".encode())
    fh.flush()
    p = subprocess.Popen(spec.argv, env=spec.env, stdout=fh, stderr=subprocess.STDOUT, cwd=tree,
                         start_new_session=True)
    spec.pid = p.pid
    spec.proc = p
    log(f"group {spec.name} pid {p.pid} (session id = pid) log {spec.log}")


def arm_deadman(log: Log, boot_log: str, port: int, pattern: str, probe_s: int, tag: str, name: str, dry: bool) -> int:
    out = f"{GPU_ARB}/deadman_{tag}_{name}.out"
    cmd = f"GRACE_S=600 PROBE_S={probe_s} setsid {DEADMAN} {shlex.quote(boot_log)} {port} {shlex.quote(pattern)} > {shlex.quote(out)} 2>&1 & echo $!"
    if dry:
        log(f"DRY-RUN: would arm deadman: {cmd}")
        return 0
    pid = int(subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip() or 0)
    time.sleep(1)
    proof = subprocess.run(["pgrep", "-af", f"boot_deadman.sh {boot_log}"], capture_output=True, text=True).stdout.strip()
    n = len(proof.splitlines())
    log(f"deadman {name}: pid {pid} GRACE_S=600 PROBE_S={probe_s} pattern={pattern!r} verdict -> {out}; pgrep proof: {n} process(es) whose argv carries THIS log ({proof or 'NONE -- UNKNOWN, never alive'})")
    return pid


def sleep_group(port: int, log: Log, name: str, weights_tags: List[str]) -> float:
    tags = ["kv_cache"] + list(weights_tags)
    t0 = time.time()
    code, body = http("POST", f"http://127.0.0.1:{port}/release_memory_occupation", {"tags": tags}, timeout=900)
    dt = (time.time() - t0) * 1000
    if code != 200:
        raise Weg2LaunchRefused(f"sleep({name}) failed: HTTP {code} {body[:300]!r}")
    log(f"sleep({name}) OK in {dt:.0f} ms (tags {','.join(tags)}; flush BEFORE pause per record 1d MUST_FIX)")
    return dt


def _model_config(model: str) -> dict:
    with open(os.path.join(model, "config.json")) as f:
        return json.load(f)


def model_num_layers(model: str) -> int:
    """The backbone depth, through the SERVER's own probe order (fix 6)."""
    from sglang.srt.server_args import declared_num_hidden_layers_from_config

    n = declared_num_hidden_layers_from_config(_model_config(model))
    if not n or n <= 0:
        raise Weg2LaunchRefused(f"num_hidden_layers not found in {model}/config.json")
    return int(n)


def model_layer_kinds(model: str) -> List[bool]:
    """One flag per layer: True = full attention, False = linear/GDN.

    THE SERVER'S OWN DERIVATION, called rather than re-implemented
    (``server_args.declared_layer_kinds_from_config``, which
    ``ServerArgs.declared_layer_kinds`` also calls), so the launcher's
    chunk->card map and the server's PP split cannot disagree about the
    checkpoint.

    #1233 fix 6: this used to read ``layer_types`` ONLY and refuse otherwise,
    which is STRICTER than the authority it claimed to mirror -- that one also
    accepts ``layers_block_type`` and ``full_attention_interval`` (its own
    docstring names the latter as the Qwen3.5/3.6 GDN hybrid source).  On such
    a checkpoint the server derived a real hybrid split while this refused and
    published NO map, and no map means ``front.interleave_pause_order`` returns
    the IDENTITY order -- the order that killed boot weg2dk4 with a device OOM
    in ``cu_mem_create``.  A degradation that is honest about its reason is
    still a degradation into a known boot killer.

    The one refusal left is the one that has no authority to defer to: a
    config whose depth cannot be read at all.
    """
    from sglang.srt.server_args import declared_layer_kinds_from_config

    return declared_layer_kinds_from_config(_model_config(model), model_num_layers(model))


def p_stage_layers(
    is_full_attention: Sequence[bool],
    scores: Optional[Sequence[int]] = None,
    attn_scores: Optional[Sequence[int]] = None,
) -> List[int]:
    """The per-stage LAYER COUNTS a score pair derives to -- never restated.

    #1233 fix 5.  The flip-order map used to be built from
    ``P_PP_STAGE_RATIO_SCORES`` directly, with a comment calling that vector
    "the pipeline layer split".  It is not: it is the per-stage capability
    score vector, and the layer counts are what
    ``derive_pp_layer_split`` makes of it under the hybrid snap.  The two
    agree for exactly the current (checkpoint, 32/18/14, 8/4/4) triple and
    diverge for its neighbours, so the old map was right by coincidence and
    unguarded by construction: ``interleave_pause_order`` refuses only an
    INCOMPLETE map, and a map that is complete but wrong yields a confident
    ``why="tightest-card-first"`` order that pauses the wrong card first --
    the wrong-per-card-accounting class that killed weg2dk4, wearing an
    instrument that says it is fine.

    This calls the SAME function ``server_args._handle_pp_stage_ratio`` calls
    (``distributed.utils.derive_pp_layer_split``), so there is one authority
    for the split and the launcher reads it rather than keeping a second copy.

    FIX 2 -- WHICH SCORE PAIR.  Fix 5 hard-wired the module constants here,
    which was right while they were the only cut in the file.  They stopped
    being that when ``solve_p_cut`` was wired into ``main``: the boot then ran
    the SOLVED pair while this function still answered for the INCUMBENT, so
    ``main`` published a flip-order map and a WEG2-PP-SPLIT line for a layout
    group P was not launched with -- complete, confident, and wrong, which is
    precisely the failure mode fix 5 names and the one
    ``interleave_pause_order`` cannot refuse.  The pair is a PARAMETER now and
    every caller states which cut it is asking about; the constants remain the
    default because they are the incumbent, and ``solve_p_cut`` round-trips
    its own candidate through this same seam.
    """
    from sglang.srt.distributed.utils import derive_pp_layer_split

    return derive_pp_layer_split(
        list(P_PP_STAGE_RATIO_SCORES if scores is None else scores),
        is_full_attention=list(is_full_attention),
        attn_scores=list(
            P_PP_ATTN_STAGE_RATIO_SCORES if attn_scores is None else attn_scores
        ),
    )


def shipped_layer_split(
    stage_ratio: str,
    attn_stage_ratio: str,
    layer_set: str,
    is_full_attention: Sequence[bool],
    n_stages: int,
) -> List[int]:
    """The per-stage LAYER COUNTS group P's ARGV will actually run.

    The counterpart of :func:`p_stage_layers`, and the reason it exists: that
    function's own docstring says it reads "the SAME two score vectors
    ``argv_p`` passes", and since the merge train brought :func:`solve_p_cut`
    that premise is no longer true by construction -- argv_p passes whatever
    the solver returned.  A premise that used to hold and now has to be CHECKED
    is a guard, not a comment, so the split is derived from the argv strings
    themselves and compared (``main``, W46).

    Both argv forms are resolved through the runtime's OWN parser for that
    form -- ``parse_pp_layer_sets`` for a layer SET, ``derive_pp_layer_split``
    for the score pair -- never by counting the score vector, which is the
    conflation :func:`p_stage_layers` exists to undo.  ``[]`` when the argv
    states no cut at all; an empty answer is never compared as agreement.
    """
    if layer_set:
        from sglang.srt.distributed.utils import parse_pp_layer_sets

        return [
            len(x)
            for x in parse_pp_layer_sets(
                layer_set, len(list(is_full_attention)), int(n_stages), allow_gapped=True
            )
        ]
    if not stage_ratio:
        return []
    from sglang.srt.distributed.utils import derive_pp_layer_split

    return list(
        derive_pp_layer_split(
            _csv_ints(stage_ratio),
            is_full_attention=list(is_full_attention),
            attn_scores=_csv_ints(attn_stage_ratio) if attn_stage_ratio else None,
        )
    )


def build_tms_preload(tree: str, venv: str, log: Log) -> str:
    """Build (or reuse) the patched torch_memory_saver preload hook."""
    script = os.path.join(tree, "scripts", "weg2", "tms", "build_tms_preload.sh")
    r = subprocess.run([script, "--venv", venv], capture_output=True, text=True)
    if r.returncode != 0:
        raise Weg2LaunchRefused(f"torch_memory_saver preload build failed: {r.stderr[-800:]}")
    so = r.stdout.strip().splitlines()[-1]
    log(f"torch_memory_saver 0.0.9.post1 preload hook REBUILT from python/sglang/srt/weg2/tms_csrc (PATCH.md: cpu backup "
        f"freed after resume) -> {so}; SGLANG_WEG2_TMS_PRELOAD_SO set for both groups")
    return so


#: The prefetch rate bound the carrier route is derived with: a prompt may be
#: at most this fraction of the host staging pool that must carry it (it
#: refused the 84k prompt on boot weg2ls4b2 at limit=27466 of pool 30518).
CARRIER_PREFETCH_FRACTION = 0.9

def budgets_from_dc(
    cards: List[Card],
    dc_mib: Dict[str, int],
    log: Log,
    label: str,
    overshoot_mib: Optional[List[int]] = None,
    overshoot_provenance: str = "",
) -> List[int]:
    out = []
    for i, c in enumerate(cards):
        over = int(overshoot_mib[i]) if overshoot_mib is not None else 0
        b = c.total_mib - CORRIDOR_MIB - dc_mib[c.uuid] - over
        b = (b // 8) * 8
        out.append(b)
        log(
            f"budget {label} ordinal={i} nvml_idx={c.nvml_index} {c.name}: "
            f"{b} MiB = total {c.total_mib} - corridor {CORRIDOR_MIB} - dormant_other {dc_mib[c.uuid]}"
            + (f" - measured_awake_overshoot {over} ({overshoot_provenance})" if over else "")
            + " MiB"
        )
    return out


def _max_running_requests(model: str, group: str = "D", bs: int = 8) -> int:
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
    """
    if str(group) == "P":
        flags = argv_p("py", model, [1, 1, 1], 1, 1, 1.0, [], p_bs=int(bs))
    else:
        flags = argv_d("py", model, [1, 1, 1], 1, 1, 1.0, [], d_bs=int(bs))
    return int(flags[flags.index("--max-running-requests") + 1])


def d_mamba_ping_pong_cost(model: str, disable_overlap: bool, d_bs: int = 8) -> Tuple[str, int, int, int]:
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
    from sglang.srt.mem_cache.mamba_pool_floor import mamba_ping_pong_slots
    from sglang.srt.server_args import ServerArgs

    # Group D's own argv, for the same reason and with the same defect history
    # as _max_running_requests above: this list is read for the presence of a
    # flag, so it must be the list the group actually gets.
    flags = argv_d("py", model, [1, 1, 1], 1, 1, 1.0, [], d_bs=int(d_bs))

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


def d_overlap_cost_line(model: str, disable_overlap: bool, d_bs: int = 8) -> str:
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


@dataclass(frozen=True)
class DOperatingPointRow:
    """One operating point's vector and what the runtime's own model prices it at."""

    position: str
    #: What the vector is proportional to, named so a reader never has to guess
    #: which of the two measured rates produced it.
    score_name: str
    scores: Tuple[float, ...]
    weights: Tuple[int, ...]
    attn_heads: Tuple[int, ...]
    gdn_heads: Tuple[int, ...]
    #: Lockstep round cost from PerfCostModel, in the unit that method returns.
    #: ``None`` = the model could not price it, and the row then says so rather
    #: than carrying a number that looks measured.
    round_ms: Optional[float]
    round_unit: str
    world_pool_tokens: Optional[int]
    #: ``PerfCostModel.predict_capacity``'s own verdict that this vector fits
    #: at all. ``None`` = the capacity call did not run (unpriced geometry).
    feasible: Optional[bool] = None
    #: The FUNDED context ``min(sum_r P_r, 64 * min_r P_r)`` -- bounded by the
    #: SMALLEST rank, which is the whole reason it is carried next to the
    #: world pool: a vector that collapses one rank barely moves ``sum(p)``
    #: and can halve this.
    funded_ctx_tokens: Optional[int] = None
    note: str = ""
    #: #1293. WHICH AXIS THE ATTENTION FAMILY'S COMPUTE RIDES on this boot's
    #: geometry -- ``"token"`` under replicated-KV uneven DCP, ``"head"``
    #: otherwise. It is not decoration: it names the grid the W53 saturation
    #: verdict below was taken against, and the two grids differ by 16x on this
    #: rig (4 kv-head units vs 64 token units), which is the difference between
    #: a position being representable and being refused.
    attn_axis: str = "head"
    #: The DCP token partition of ``_CP_TOKEN_UNITS`` units (largest-remainder,
    #: the same partitioner and the same units the runtime's own "Uneven DCP:
    #: auto-set dcp_size" install uses). Empty tuple on the head axis, where
    #: there is no token vector to report.
    attn_token_units: Tuple[int, ...] = ()
    #: Floor-aware re-normalisation statement: which family/axis had a rank
    #: pinned at the partitioner's one-unit floor, and how far the REALISED
    #: shares then sit from the requested ratio. Empty when no grid binds.
    axis_note: str = ""


def _gcd_reduce(values: Sequence[int]) -> Tuple[int, ...]:
    vals = [max(1, int(v)) for v in values]
    g = math.gcd(*vals) if len(vals) > 1 else vals[0]
    return tuple(v // max(1, g) for v in vals)


def _weights_from_scores(scores: Sequence[float]) -> Tuple[int, ...]:
    """A ratio vector proportional to a measured per-rank rate.

    ``--rank-tp-ratio`` takes a comma-separated integer ratio
    (server_args.py:467 ``_parse_rank_tp_ratio``), so the rates are scaled to
    integers and gcd-reduced. x1000 because this rig's slowest and fastest
    measured rates differ by ~4x and rounding at x1 would quantise a 3.99x
    ratio to 4:1 -- the vector must carry the measurement, not a rounding of it.
    """
    lo = min(float(s) for s in scores)
    return _gcd_reduce([int(round(float(s) / max(lo, 1e-9) * 1000.0)) for s in scores])


def d_plan_inputs(model: str, tp_size: int, d_bs: int):
    """The runtime ``PlanInputs`` describing THIS boot's group D.

    ONE FACTORY (refuter MF-6). Every field is read from the module constant
    the argv builder emits, so the vector this launcher PRICES and the argv it
    SHIPS cannot describe two different configurations. Both callers -- the
    #1017 weight line and the #1241 operating-point rows -- go through here.
    """
    from sglang.srt.uneven_perf import PlanInputs

    return PlanInputs(
        tp_size=int(tp_size),
        model_path=model,
        kv_cache_dtype=KV_CACHE_DTYPE,
        speculative_algorithm=SPEC_ALGORITHM,
        speculative_num_draft_tokens=SPEC_NUM_DRAFT_TOKENS,
        max_running_requests=int(d_bs),
    )


def _attn_axis_for(weights: Sequence[int], plan_flags) -> str:
    """Which axis the attention family's COMPUTE rides for one D vector (#1293).

    ``"token"`` exactly when the boot form is replicated-KV uneven DCP -- the
    serving form: every rank keeps the FULL replicated kv-head set and runs the
    attention core over its own TOKEN shard, so the compute follows the token
    vector, which is continuous (#492: "attention grid-PINNED was FALSE").
    ``"head"`` otherwise (even DCP, no DCP, or a uniform plan), where the
    kv-head unit grid genuinely carries the family.

    THE PREDICATE IS #503's SHARED ONE, NOT A THIRD SPELLING.
    ``plan_uneven_dcp_kv_replicated`` (distributed/utils.py, placed next to the
    runtime's ``uneven_dcp_kv_replicated`` by #503 so the two cannot drift)
    answers this exact question at plan time from the flags about to be
    emitted. A ``plan_flags`` with no DCP evidence (``dcp_size`` unset and no
    KV token vector) answers "head" for ANY weights -- a non-DCP form must
    never silently take the token axis, because there the token vector does
    not exist and the head grid is the real carrier.
    """
    from sglang.srt.distributed.utils import plan_uneven_dcp_kv_replicated

    return (
        "token"
        if plan_uneven_dcp_kv_replicated(plan_flags, list(weights))
        else "head"
    )


def _axis_floor_ranks(units: int, weights: Sequence[int]) -> Tuple[int, ...]:
    """Ranks whose proportional share of ``units`` is BELOW one whole unit.

    ``partition_units`` guarantees every rank >= 1 unit, so each of these is
    pinned at the partitioner's floor and the realised split stops carrying the
    weight ratio on that axis. This is the one REACHABLE failure of a unit grid
    -- a rank owning ZERO units cannot happen -- and naming the ranks rather
    than answering yes/no is what lets the caller re-normalise against the grid
    that binds and say WHICH rank moved.
    """
    total = sum(int(w) for w in weights)
    if total <= 0 or int(units) <= 0:
        return tuple()
    return tuple(
        r
        for r, w in enumerate(weights)
        if float(int(w)) / float(total) * float(int(units)) < 1.0
    )


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


def d_operating_point_rows(
    cards: Sequence[Card],
    budgets: Sequence[int],
    model: str,
    d_bs: int,
    facts: Sequence[EarlyReadFact] = EARLY_READ_FACTS,
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
    pcm = None
    plan = None
    try:
        from sglang.srt.uneven_perf import PerfCostModel

        plan = d_plan_inputs(model, len(budgets), d_bs)
        pcm = PerfCostModel(
            plan,
            list(maxkv_weights),
            list(budgets),
        )
    except Exception as exc:  # pragma: no cover - geometry is diagnostic
        refusals.append(
            "W52 Weg2TpOperatingPointUnpriced: the cost model could not be "
            "built for %r (%s), so neither operating point can be derived. The "
            "shipped maxkv vector is unaffected -- it is the budget vector and "
            "needs no model." % (model, exc)
        )
        return ([], refusals)

    # -- the two measured rate vectors, by card NAME -------------------------
    gemm: List[float] = []
    membw: List[float] = []
    try:
        from sglang.srt.planner.card_rate_pass import load_measured_library

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
            "W52 Weg2TpOperatingPointUnpriced: this rig's card-rate library "
            "carries no measured (gemm_tflops, membw_gbs) pair for every card "
            "of group D (%s). An operating-point vector IS the measured rate "
            "ratio, so without the measurement there is no vector to ship -- "
            "it is refused, never approximated from a nameplate peak. Write "
            "one with `python -m sglang.srt.planner.card_rate_pass --run`."
            % ", ".join(c.name for c in cards)
        )
        return ([], refusals)

    # -- W54: an explicit vector needs the uneven-TP early-read env fact ------
    # `_handle_uneven_tp` (server_args.py:12025) reads SGLANG_UNEVEN_DCP* off
    # os.environ BEFORE the flag half publishes itself (:7186). An explicit
    # ratio shipped on a boot whose EARLY_READ_FACTS no longer carries that row
    # for group D would be silently evened out -- a wrong vector that boots.
    have = {
        f.env_key for f in facts if f.groups in ("both", "D") and f.env_key
    }
    missing = [
        k
        for k in ("SGLANG_UNEVEN_DCP", "SGLANG_UNEVEN_DCP_WEIGHTED")
        if k not in have
    ]
    if missing:
        refusals.append(
            "W54 Weg2TpOperatingPointNeedsEnvPin: an explicit --rank-tp-ratio "
            "vector is resolved by _handle_uneven_tp (server_args.py:12025), "
            "which reads %s off os.environ BEFORE the flag half publishes "
            "itself (:7186). EARLY_READ_FACTS no longer states %s for group D, "
            "so shipping the vector would need an env pin in the launcher's own "
            "shell -- the R19 shape #1235 removed. Refused instead."
            % (" and ".join(missing), " and ".join(missing))
        )
        return ([], refusals)

    # -- the three rows ------------------------------------------------------
    from sglang.srt.distributed.utils import _CP_TOKEN_UNITS, partition_units

    n_ranks = len(budgets)

    # #1293 -- the plan-time flags the attention-axis predicate reads. This
    # launcher's group D always ships ``--uneven-dcp`` + ``--uneven-dcp-
    # weighted`` (``early_read_flags("D")``), and the W54 gate immediately
    # above has just PROVEN both facts present -- under a non-uniform
    # ``--rank-tp-ratio`` the runtime then auto-sets ``dcp_size = tp_size``
    # (server_args.py:12112-12128, "Uneven DCP: auto-set dcp_size"). Mirroring
    # that auto-set here is gated on the SAME fact the W54 check read, so a
    # future form that stops shipping the env fact falls back to the head
    # axis instead of silently keeping the token one. The non-uniformity half
    # of the predicate stays inside ``plan_uneven_dcp_kv_replicated`` itself.
    uneven_dcp_armed = "SGLANG_UNEVEN_DCP" in have
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
            attn_units = list(partition_units(grid, list(weights)))
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
            token_units: Tuple[int, ...] = ()
            if axis == "token" and position != "maxkv":
                # The position's own intent made concrete: token ownership
                # proportional to the measured rates, integerised exactly the
                # way the runtime integerises its token vector. For maxkv
                # (auto) the runtime DERIVES its capacity-matched vector
                # instead; that derived vector is read off predict_capacity
                # below rather than re-spelled here.
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
            if position == "maxkv" and axis == "token" and cap.get("token_vector"):
                token_units = tuple(int(v) for v in cap["token_vector"])
        except Exception:
            pool = None
        refusal = None
        if feasible is False:
            refusal = (
                "W55 Weg2TpOperatingPointInfeasible: position %s derives "
                "weights %s, which PerfCostModel.predict_capacity marks "
                "feasible=False against this boot's budgets %s -- the weight "
                "shards plus the mamba pool plus the reserves do not leave a "
                "positive KV pool on at least one rank. Refused. This is the "
                "model's own verdict, not a margin chosen here: no safety "
                "factor is applied on top of it."
                % (position, list(int(w) for w in weights), list(budgets))
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

    # -- W53: a position that would turn an uneven axis OFF -------------------
    #
    # THE OBVIOUS CHECK IS UNREACHABLE AND IS NOT THE ONE MADE HERE. A rank
    # owning zero heads cannot happen: `partition_units` guarantees every rank
    # >= 1 unit by construction (distributed/utils.py, "largest-remainder
    # rounding, every rank gets >= 1 unit"). Asserting against zero heads would
    # be a guard that can never fire -- written, never executed.
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
            # a saturation refusal against it would be a FALSE W53 -- the
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
                "W53 Weg2TpOperatingPointDisablesUnevenAxis: position %s "
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
                "W53 Weg2TpOperatingPointDisablesUnevenAxis: position %s "
                "derives the FLAT vector %s, which is even TP wearing an "
                "uneven flag. Refused: an axis that resolves to equality is "
                "the axis disabled." % (row.position, list(row.weights))
            )
    return (rows, refusals)


def d_operating_point_line(
    rows: Sequence[DOperatingPointRow], refusals: Sequence[str], shipped: str
) -> str:
    """All three vectors, their price and their pool, on ONE line.

    The trade is only readable if the alternatives are printed next to what
    shipped, in the same units, from the same model, on the same boot.
    """
    if not rows:
        return (
            "WEG2 D-OPERATING-POINTS UNPRICED (shipped=%s): %s"
            % (shipped, " | ".join(refusals) or "no reason given")
        )
    parts = []
    for row in rows:
        priced = (
            "%.4f %s" % (row.round_ms, row.round_unit)
            if row.round_ms is not None
            else row.round_unit
        )
        # #1293: which axis the attention compute rides, named per row, plus
        # the token partition when that axis carries it and the floor-binding
        # statement when any grid binds. `attn_axis=head` is the pre-DCP
        # reading of the attn heads printed before it; `attn_axis=token`
        # says the heads are the PROJECTION split and the compute follows
        # `token_units`.
        axis_bit = " attn_axis=%s" % row.attn_axis
        if row.attn_token_units:
            axis_bit += " token_units %s" % (list(row.attn_token_units),)
        if row.axis_note:
            axis_bit += " [%s]" % row.axis_note
        parts.append(
            "%s%s: weights %s from %s %s -> attn %s%s GDN %s, round %s, world pool "
            "%s, funded ctx %s, feasible %s"
            % (
                row.position,
                " [SHIPPED]" if row.position == shipped else "",
                list(row.weights),
                row.score_name,
                [round(x, 1) for x in row.scores],
                list(row.attn_heads),
                axis_bit,
                list(row.gdn_heads),
                priced,
                row.world_pool_tokens
                if row.world_pool_tokens is not None
                else "UNPRICED",
                # BOTH, because they answer different questions: the world
                # pool is the sum over ranks, the funded context is bounded by
                # the SMALLEST rank (min(sum P, 64 min P)) and is the quantity
                # a carrier bound is about. A vector that collapses one rank
                # moves the second far more than the first.
                row.funded_ctx_tokens
                if row.funded_ctx_tokens is not None
                else "UNPRICED",
                "UNPRICED" if row.feasible is None else row.feasible,
            )
        )
    tail = (" REFUSED: " + " | ".join(refusals)) if refusals else ""
    note = next((r.note for r in rows if r.note), "")
    return (
        "WEG2 D-OPERATING-POINTS (#1241 slice 2, the operating-point axis of "
        "the #1017 solve; shipped=%s, default %s since the user order of "
        "2026-09-09 -- 'maxkv' is still selectable and its argv is still "
        "byte-identical to the pre-#1241 form). %s. %s%s"
        % (shipped, D_TP_OBJECTIVE_DEFAULT, " || ".join(parts), note, tail)
    )


@dataclass(frozen=True)
class DTpRatioDecision:
    """Group D's weight-vector objective, the argv it produces, and its price."""

    objective: str
    tune: str
    flags: Tuple[str, ...]
    line: str
    #: #1241 slice (2). The three-vector provenance line. Empty only when the
    #: rows could not be built at all, which the line itself then says.
    op_line: str = ""
    rows: Tuple[DOperatingPointRow, ...] = ()


def d_tp_ratio_decision(
    objective: str,
    tune: str,
    cards: Sequence[Card],
    budgets: Sequence[int],
    model: str,
    d_bs: int,
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
        raise Weg2LaunchRefused(
            "W47 Weg2TpObjectiveRefused: --d-tp-objective %r is not one of %s."
            % (objective, "|".join(D_TP_OBJECTIVE_CHOICES))
        )

    # #1241 slice (2). Built on EVERY boot, including the maxkv ones, because
    # the point of the line is that the trade is visible per boot. Refusals
    # only KILL the launch when the refused position is the one being shipped.
    op_rows, op_refusals = d_operating_point_rows(cards, budgets, model, d_bs)
    op_line = d_operating_point_line(op_rows, op_refusals, objective)
    if objective in D_OPERATING_POINTS:
        mine = [
            r
            for r in op_refusals
            if objective in r or r.startswith(("W52", "W54"))
        ]
        if mine:
            raise Weg2LaunchRefused(mine[0] + " (position %s was SHIPPED, so "
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
        from sglang.srt.distributed.utils import partition_units
        from sglang.srt.uneven_perf import PerfCostModel

        pcm = PerfCostModel(
            d_plan_inputs(model, len(budgets), d_bs), weights, list(budgets)
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
        from sglang.srt.planner.card_rate_pass import load_measured_library

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
            "these card names (`python -m sglang.srt.planner.card_rate_pass "
            "--run` writes one), so the cost of the capacity split at the "
            "attention barrier is NOT estimated here. An absent estimate is "
            "reported as absent; it is never a measured zero."
        )

    line = (
        "WEG2 D-WEIGHTS objective=%s (default %s since the user order of "
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


@dataclass(frozen=True)
class DTokenVectorDecision:
    """Group D's KV-token ownership vector: what is shipped, and why."""

    flags: Tuple[str, ...]
    line: str


#: #1032 THE VECTOR THIS LAUNCHER MUST NEVER SHIP AGAIN, and the boot that
#: showed what the runtime does with it.
#:
#: ``--uneven-token-vector 29,19,16 --uneven-token-vector-role seed`` sat on
#: group D's argv. 29,19,16 is the emitted value of RETRACTED investigation
#: #602 (planner/retracted.py REGISTER), so every Weg-2 boot printed the #797
#: PROVENANCE warning and then spent a supersession undoing it.
#:
#: WHAT THE LOG ACTUALLY SHOWS, because the fix must not be justified against
#: a claim the evidence refutes (boot weg2rg6, D log
#: /spinning/evidence-665-f1/boot_weg2_weg2rg6_7f88b1c75d_0908_070324.D.log):
#: the seed did NOT reach the pools. All three "Uneven-DCP token sizing" lines
#: read ratios 17 / 7 / 8, i.e. partition_units(64, measured P_r) gcd-reduced
#: from the measured per-rank capacities 364718 / 154980 / 170464; `vector [29`
#: does not appear on a single sizing line. The install LANDED and superseded
#: the seed, exactly as role='seed' promises.
#:
#: SO THE DEFECT IS NOT "D SERVED ON THE RETRACTED SPLIT" -- it is that a
#: retracted lineage was shipped at all, and that the runtime spends a
#: supersession every boot undoing a number the launcher had no business
#: stating. Normalised to one sum, the seed 29,19,16 against the measured
#: 34,14,16 is +17 % on rank 0 and -26 % on rank 1: had the install NOT landed,
#: that is the split D would have served.
#:
#: The fix is therefore to state nothing. With no vector on the argv the
#: runtime derives its own starting point from THIS boot's budgets
#: (distributed/utils.py:1135, partition_units(64, budget-minus-checkpoint))
#: and the measured install replaces it after profiling -- the same install
#: that already ran, now without withdrawn evidence upstream of it.
#:
#: WHAT IS GIVEN UP, named rather than left to be discovered: no seed means
#: nothing arms ``note_seed_awaiting_supersession``, so the
#: assert_seed_superseded gate does not hold this boot to an install. That
#: gate exists because a SEED carries foreign lineage; the budget-derived
#: fallback carries this boot's own numbers, so an install that does not
#: happen leaves D on its own honest estimate rather than on a withdrawn one.
#: (And the gate's disarm keys on the VERDICT boolean, not on a changed vector
#: -- distributed/utils.py:306 says so in its own docstring -- so "the gate did
#: not fire" was never proof of an install either. The sizing lines are.)
RETRACTED_SEED_VECTOR_1032 = (29, 19, 16)


def d_token_vector_decision(
    vector: Optional[str],
    role: str,
    provenance: Optional[str],
) -> DTokenVectorDecision:
    """Group D's token-vector argv, and the refusal that keeps #602 out.

    Default (``vector`` unset): NOTHING is shipped. The line says which branch
    the boot is on and where the vector will come from instead.

    Operator-supplied: the value is checked against the runtime's OWN
    retraction register (``planner.retracted.find_retracted_token_vector``, the
    same authority ``_refuse_retracted_token_vector`` uses) BEFORE the window
    is spent. The runtime refuses a retracted PIN and only warns about a
    retracted SEED, because at that point a seed is still going to be
    superseded; here neither is accepted, because a launcher that types a
    retracted number has already made the mistake the register exists to
    catch, and the desk is the cheap place to say so.
    """
    role = str(role or "pin")
    if not vector:
        return DTokenVectorDecision(
            flags=(),
            line=(
                "WEG2 D-TOKEN-VECTOR: none shipped (default). SOURCE = "
                "estimate(budget): group D derives its own starting vector "
                "from THIS boot's --rank-gpu-memory-mib "
                "(distributed/utils.py resolve_cp_token_ratios, the "
                "partition_units(64, budget - weights - 1536 MiB) rung). "
                "OBJECTIVE = the profiled per-rank capacity optimum: the "
                "runtime supersedes that estimate IN-PROCESS after profiling "
                "and prints the pool it moved -- 'installed measured KV-token "
                "ownership vector [..] (pre-boot estimate was [..]), "
                "max_total_num_tokens X -> ~Y'. That install is armed by the "
                "ROLE, and #1270 is why this sentence can be believed again: "
                "an undeclared vector now reads role='estimate', not 'pin'. "
                "Boot weg2sb1 is the counter-example -- with the #1032 seed "
                "gone nothing declared a vector, the estimate [30,17,17] "
                "inherited 'pin', the install was suppressed and D served "
                "574,336 tokens against rg6's 681,856 (-15.8 %%) while the "
                "measured [17,7,8] was printed as an unusable restart hint. "
                "The retracted #602 seed %s this launcher used to ship stays "
                "GONE (#1032): the seed was never the right way to arm the "
                "install, it was only the way that happened to work, and it "
                "carried a foreign lineage to hold a boot to."
                % (",".join(str(v) for v in RETRACTED_SEED_VECTOR_1032),)
            ),
        )

    try:
        parsed = [int(x) for x in str(vector).split(",") if x.strip() != ""]
    except ValueError:
        raise Weg2LaunchRefused(
            "W46 Weg2TokenVectorRefused: --d-uneven-token-vector %r is not a "
            "comma-separated integer vector." % (vector,)
        )
    if not parsed or any(v <= 0 for v in parsed):
        raise Weg2LaunchRefused(
            "W46 Weg2TokenVectorRefused: --d-uneven-token-vector %r must be "
            "positive integers, one per DCP rank." % (vector,)
        )

    from sglang.srt.planner.retracted import (
        find_retracted_token_vector,
        token_vector_refusal_text,
    )

    entry = find_retracted_token_vector(parsed, provenance)
    if entry is not None:
        raise Weg2LaunchRefused(
            "W46 Weg2TokenVectorRefused: "
            + token_vector_refusal_text(
                entry,
                parsed,
                "the Weg-2 launcher refuses to ship it in EITHER role. The "
                "runtime warns about a retracted role='seed' and refuses only "
                "a 'pin' (distributed/utils.py:704); this launcher refuses "
                "both, because a boot window is the expensive place to learn "
                "that a number was withdrawn",
            )
        )

    flags = (
        "--uneven-token-vector",
        ",".join(str(v) for v in parsed),
        "--uneven-token-vector-role",
        role,
    )
    if provenance:
        flags = flags + ("--uneven-token-vector-provenance", str(provenance))
    return DTokenVectorDecision(
        flags=flags,
        line=(
            "WEG2 D-TOKEN-VECTOR: %s role=%s provenance=%s -- OPERATOR "
            "ADVISORY, checked against planner.retracted's register at the "
            "desk and not found withdrawn. role='seed' still promises "
            "supersession by this boot's measured per-rank capacity; 'pin' "
            "asserts the value against that measurement. Default is to ship "
            "nothing at all."
            % (
                ",".join(str(v) for v in parsed),
                role,
                provenance or "(undeclared -- the value match is what caught "
                "#602, so state it if you know it)",
            )
        ),
    )


#: The one line ``read_pp_bubble`` understands, emitted per window per rank by
#: ``scheduler_components/pp_bubble.py:summary_line``. Anchored on the whole
#: field name including its ``=``: a bare number would match milliseconds
#: elsewhere in the line, which is the bare-ticket-number trap in miniature.
_BUBBLE_RE = re.compile(
    r"PP-BUBBLE rank=(?P<rank>\d+) .*?"
    r" n=(?P<n>\d+) \(n_gaps=(?P<n_gaps>\d+),"
    r" forward_ms=(?P<forward>[0-9.]+),"
    r" bubble_ms=(?P<bubble>[0-9.]+),"
    r" starved_ms=(?P<starved>[0-9.]+);"
)


@dataclass(frozen=True)
class BubbleMeasurement:
    """One stage's PP-BUBBLE totals, summed over every window in one log.

    Summed, never averaged: each window line carries its own numerator AND its
    own denominator, so a mean of the printed shares would weight a 12-forward
    window like a 2-forward one. Sum of numerators over sum of denominators is
    the only aggregation that keeps the denominator honest.
    """

    source: str
    rank: int
    windows: int
    forward_ms: float
    bubble_ms: float
    starved_ms: float
    n_forwards: int

    @property
    def denominator_ms(self) -> float:
        """``gap + forward`` -- the denominator the emitting line names."""
        return self.bubble_ms + self.forward_ms

    @property
    def bubble_share(self) -> float:
        d = self.denominator_ms
        return 0.0 if d <= 0.0 else self.bubble_ms / d

    @property
    def forward_share(self) -> float:
        return 1.0 - self.bubble_share

    @property
    def stall_share(self) -> float:
        """The part depth can move.

        ``starved_ms`` is the part of the gap in which this rank visited the
        loop with NOTHING to launch (fix 1r/2). Depth overlaps a rank's output
        exchange with its next forward; it cannot manufacture a chunk that the
        queue never supplied. Charging starvation to depth would buy in-flight
        slots against a supply problem -- and pay for them out of the KV pool.
        """
        d = self.denominator_ms
        if d <= 0.0:
            return 0.0
        return max(0.0, self.bubble_ms - self.starved_ms) / d


def read_pp_bubble(path: str) -> Optional[BubbleMeasurement]:
    """The BINDING stage's bubble totals from one group-P log, or None.

    The binding stage is the one with the largest forward total: under a
    pipelined prefill the makespan is that stage's time, so its idle is the
    idle that costs throughput. Returns None when the file does not exist or
    carries no PP-BUBBLE line -- absence of the instrument, never a measured
    zero (#892 / the indicator law: a tool says "I found nothing", never
    "there is nothing").
    """
    per_rank: Dict[int, List[float]] = {}
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = _BUBBLE_RE.search(line)
                if m is None:
                    continue
                acc = per_rank.setdefault(int(m.group("rank")), [0.0] * 4 + [0.0])
                acc[0] += float(m.group("forward"))
                acc[1] += float(m.group("bubble"))
                acc[2] += float(m.group("starved"))
                acc[3] += float(m.group("n"))
                acc[4] += 1.0
    except OSError:
        return None
    if not per_rank:
        return None
    rank = max(per_rank, key=lambda r: per_rank[r][0])
    fwd, bub, starved, n_fwd, windows = per_rank[rank]
    return BubbleMeasurement(
        source=path,
        rank=rank,
        windows=int(windows),
        forward_ms=fwd,
        bubble_ms=bub,
        starved_ms=starved,
        n_forwards=int(n_fwd),
    )


def newest_bubble_log(evidence_dir: str) -> Optional[str]:
    """Newest ``*.P.log`` in ``evidence_dir`` that actually CARRIES the line.

    Not simply the newest P log: a boot that died before its first bubble
    window, or one built before the instrument existed, has no measurement,
    and taking its silence as ``share=0`` would derive ``depth=0`` from a file
    rather than from a measurement. Such a log is skipped and an older one
    that carries the line is preferred; the chosen path is printed, so a
    reader can see how old the number is.
    """
    try:
        names = [n for n in os.listdir(evidence_dir) if n.endswith(".P.log")]
    except OSError:
        return None
    paths = [os.path.join(evidence_dir, n) for n in names]
    for path in sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True):
        if read_pp_bubble(path) is not None:
            return path
    return None


#: The runtime's transport for an explicit per-stage layer SET. It is READ by
#: the ranks and WRITTEN by exactly one thing: this launcher, from
#: ``--pp-layer-set`` or from the solver's own chosen map. An INHERITED value
#: is refused (W44), never honoured -- see :func:`refuse_inherited_layer_set`.
PP_LAYER_SET_ENV = "SGLANG_PP_LAYER_SET"
PP_CROSSING_WIRE_ENV = "SGLANG_PP_CROSSING_WIRE"

_PREFILL_RE = re.compile(
    r"\b(?P<rank>PP\d+)\] Prefill batch,.*?#new-token: (?P<new>\d+), "
    r"#cached-token: (?P<cached>\d+)"
)


def refuse_inherited_layer_set(env: Mapping[str, str]) -> None:
    """W44: a layer set inherited from the environment is REFUSED, never used.

    ``build_env`` starts from ``os.environ``, so before this refusal an
    exported ``SGLANG_PP_LAYER_SET`` reached group P's ranks without passing
    through the launcher at all: the solver would rank and print one layout
    while the boot ran another, and the PP-CUT provenance line would describe
    a layout that never existed. That is the #505(a) silent-substitution class
    with a whole stage map as the substituted object.

    ONE KNOB, ONE READER (upstream-minimal). The flag ``--pp-layer-set`` is the
    only way to configure the map, and the launcher is the only writer of the
    variable. Not a twin of the env: its REPLACEMENT as the interface, with the
    variable demoted to the wire that carries the decision to the ranks.
    """
    raw = str(env.get(PP_LAYER_SET_ENV, "") or "").strip()
    if not raw:
        return
    raise Weg2LaunchRefused(
        "W44 Weg2LayerSetEnvRefused: %s=%r is set in this launcher's own "
        "environment. It is no longer an input: --pp-layer-set is, and the "
        "launcher is the only writer of the variable (it carries the solved "
        "or pinned map to group P's ranks). Honouring an inherited value "
        "would let the boot run a stage map the PP-CUT provenance line does "
        "not describe. Unset it and pass --pp-layer-set %s instead."
        % (PP_LAYER_SET_ENV, raw, raw)
    )


def read_mean_prefill_prefix(path: str) -> Optional[Tuple[float, int, str]]:
    """Mean PREFIX DEPTH over one boot's prefill census. ``(mean, n, rank)``.

    THE PREFIX IS ACCUMULATED, NOT READ. No line states it: a ``Prefill batch``
    line carries ``#new-token`` (this chunk) and ``#cached-token`` (the
    prefix-cache hit), never "how much of this request is already computed".
    But group P admits ONE chunked request per pass, so a request is a maximal
    run of consecutive lines on one rank and the prefix at chunk *j* is the sum
    of ``#new-token`` over chunks *0..j-1* plus that run's cache hit. A chunk
    SHORTER than the run's own maximum ends the request -- the last chunk of a
    prompt, or the zero-remainder 1-token end anchor -- and the accumulator
    resets.

    ONE RANK ONLY, and it is named in the answer: every stage prints the same
    batch, so summing across ranks would triple the denominator without adding
    a single measurement.

    ``None`` when the file does not exist or carries no census -- absence of
    the instrument, never a measured zero (#892).
    """
    rows: Dict[str, List[Tuple[int, int]]] = {}
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = _PREFILL_RE.search(line)
                if m is None:
                    continue
                rows.setdefault(m.group("rank"), []).append(
                    (int(m.group("new")), int(m.group("cached")))
                )
    except OSError:
        return None
    if not rows:
        return None
    rank = sorted(rows)[0]
    seq = rows[rank]
    full = max(n for n, _ in seq)
    prefixes: List[int] = []
    acc = 0
    for new, cached in seq:
        prefixes.append(acc + cached)
        if new < full:
            acc = 0
        else:
            acc += new
    return (sum(prefixes) / float(len(prefixes)), len(prefixes), rank)


def newest_prefill_census_log(evidence_dir: str) -> Optional[str]:
    """Newest ``*.P.log`` that actually CARRIES a prefill census.

    Same rule and the same reason as :func:`newest_bubble_log`: a boot that
    died before its first prefill has no census, and reading its silence as
    "mean prefix 0" would derive the design depth from a file rather than from
    a measurement.
    """
    try:
        names = [n for n in os.listdir(evidence_dir) if n.endswith(".P.log")]
    except OSError:
        return None
    for path in sorted(
        (os.path.join(evidence_dir, n) for n in names),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    ):
        if read_mean_prefill_prefix(path) is not None:
            return path
    return None


def pcie_lanes(cards: Sequence[Card]) -> List[Optional[int]]:
    """Current PCIe link width per CUDA ordinal, from NVML. ``None`` if unknown.

    The measured link table this rig owns
    (``pp_crossing_transport.MEASURED_GBPS_BY_LANES``) is keyed by an edge's
    BOTTLENECK lane count, so the lane width is the join key -- read here per
    card rather than assumed, because "GPU0 is the x4 slot" is a fact about
    today's NVML enumeration and this launcher already refuses to hardcode
    that kind of fact (``order_cards``).
    """
    out: List[Optional[int]] = []
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return [None for _ in cards]
    try:
        for c in cards:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(int(c.nvml_index))
                out.append(int(pynvml.nvmlDeviceGetCurrPcieLinkWidth(h)))
            except Exception:
                out.append(None)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out


def per_pair_crossing_ms(
    lanes: Sequence[Optional[int]], payload_bytes: int
) -> Dict[Tuple[int, int], float]:
    """``{(src_stage, dst_stage): ms}`` for one crossing of ``payload_bytes``.

    Priced from the MEASURED point-to-point table, at the edge's bottleneck
    lane count -- ``min`` of the two cards' widths, which is what "bottleneck"
    means and what those numbers were measured at. A pair whose lane count has
    no measured entry is OMITTED, never interpolated: ``pp_cut.crossing_price``
    then refuses to rank the candidates that use it, and an unrankable
    candidate is reported rather than priced at a guess.
    """
    from sglang.srt.distributed.pp_crossing_transport import MEASURED_GBPS_BY_LANES

    out: Dict[Tuple[int, int], float] = {}
    for a in range(len(lanes)):
        for b in range(len(lanes)):
            if a == b or lanes[a] is None or lanes[b] is None:
                continue
            gbps = MEASURED_GBPS_BY_LANES.get(min(int(lanes[a]), int(lanes[b])))
            if gbps is None:
                continue
            out[(a, b)] = float(payload_bytes) / (float(gbps) * 1e9) * 1e3
    return out


def chunked_prefill_size_of(argv: Sequence[str]) -> int:
    """The chunk this launcher's own argv states, read back off it.

    Same rule as ``--max-running-requests`` in the pool model: a second copy of
    the number here would drift the day the flag moves, and the depth price is
    linear in it.
    """
    argv = list(argv)
    if "--chunked-prefill-size" not in argv:
        raise Weg2LaunchRefused(
            "W42 Weg2DepthUnfunded: group P's argv states no "
            "--chunked-prefill-size, so one in-flight microbatch has no "
            "priceable size and the depth cannot be funded against the pool. "
            "Refusing rather than assuming a chunk."
        )
    return int(argv[argv.index("--chunked-prefill-size") + 1])


#: THE DERIVATION IS NOT RE-GROUNDED, so a derived depth is not shipped
#: (#1240 FOLLOW FIX 1, MUST_FIX 3 of BOOT_weg2pp2_0907.md). MEASURED against
#: this workload, cited to that record's DEPTH VERDICT table: depth 1 cost
#: +6.9 % on TTFT p50 (3.215 vs 3.008 s, A/A floor 1.7 %) and depth 2 +9.6 %
#: (3.298), the bubble share on the binding stage did NOT fall (21.8/24.6 % at
#: depth 0, 26.8 % at depth 1, 24.1 % at depth 2, floor 2.8 pts), duty fell 15
#: points, and at the ~60 k deep point all depths sat inside the 3.35 % floor
#: = null. The #692 mechanism overlaps a rank's output exchange with its next
#: forward; the measurement says this workload's gap is not that exchange --
#: consistent with the depth line's own stall_share=0.221 against
#: bubble_share=0.466, i.e. over half the bubble was already queue starvation,
#: which depth cannot touch, and the remainder is evidently not exchange-bound
#: either. Re-grounding means a MEASURED exchange-bound share to derive from;
#: until one exists the arithmetic is printed and the shipped depth is 0. An
#: explicit --p-microbatch-depth still wins, priced exactly as before.
DEPTH_DERIVATION_GROUNDED = False

#: The record line the retraction above is read from. Named so the number and
#: its provenance cannot drift apart.
DEPTH_RETRACTION_RECORD = "/spinning/gpu-arb/weg2/BOOT_weg2pp2_0907.md"


@dataclass(frozen=True)
class DepthDecision:
    """Group P's ``--pp-async-batch-depth`` and the ONE provenance line for it."""

    depth: int
    passes_in_flight: int
    pinned: bool
    measured: Optional[BubbleMeasurement]
    pool_tokens: float
    pool_after: float
    cap_tokens: int
    price_rows: int
    price_tokens: float
    price_mib: Tuple[float, ...]
    act_mib_per_pass: float
    chunk_tokens: int
    #: What the bubble ARITHMETIC asks for, before the retraction. Kept beside
    #: the shipped :attr:`depth` rather than replaced by it: the derivation is
    #: the thing that has to be re-grounded, so it stays visible and the day a
    #: measured exchange-bound share exists, ``DEPTH_DERIVATION_GROUNDED``
    #: flips and this number ships without a second derivation being written.
    derived_depth: int = 0
    #: True when the chosen layout is a GAPPED layer set. The depth is then 0
    #: BY THE LAYOUT, not lowered: a gapped map admits exactly one pass in
    #: flight, and that is the number the PP-CUT solver already priced the
    #: candidate at. Printed so the 0 is read as a derivation, not a default.
    gapped_layout: bool = False
    #: WHICH rule lowered the shipped depth below the derivation, recorded at
    #: the point it acts rather than inferred afterwards from the numbers:
    #: ``""`` (none acted -- the derivation shipped, or there was nothing to
    #: lower), ``"retraction"`` (the #692 derivation is not re-grounded) or
    #: ``"layout"`` (a gapped map admits exactly one pass). The two rules are
    #: branches of ONE ``elif`` in :func:`solve_p_depth`, so at most one can
    #: act, and :meth:`line` prints exactly one cause for a 0. Inferring it
    #: from ``derived_depth > depth`` cannot tell them apart, which is how the
    #: line came to print both.
    lowered_by: str = ""
    #: What ONE extra pass costs in pool tokens at the worst-converting stage.
    #: :attr:`price_tokens` is ``depth * price_tokens_per_pass``; the two are
    #: kept apart so no sentence can price "one extra pass" with the total
    #: (fix 3, review finding 2: the W42 text did exactly that at depth 2).
    price_tokens_per_pass: float = 0.0

    @property
    def retracted(self) -> bool:
        """True when the RETRACTION is the rule that lowered the depth."""
        return self.lowered_by == "retraction"

    @property
    def lowered_by_layout(self) -> bool:
        """True when the GAPPED LAYOUT is the rule that lowered the depth."""
        return self.lowered_by == "layout"

    def line(self) -> str:
        """THE one line. Format is load-bearing: a reader greps ``P-DEPTH solver:``."""
        if self.measured is None:
            src = (
                "no PP-BUBBLE measurement found, so bubble_share is unknown and "
                "the depth stays at today's behaviour"
            )
            shares = "bubble_share=n/a forward_share=n/a"
        else:
            m = self.measured
            src = (
                "from %s rank=%d (binding stage, largest forward total) over %d "
                "window(s), n=%d forwards; starved_ms=%.1f of bubble_ms=%.1f is "
                "queue starvation and is NOT charged to depth, stall_share=%.3f"
                % (
                    m.source,
                    m.rank,
                    m.windows,
                    m.n_forwards,
                    m.starved_ms,
                    m.bubble_ms,
                    m.stall_share,
                )
            )
            shares = "bubble_share=%.3f forward_share=%.3f" % (
                m.bubble_share,
                m.forward_share,
            )
        if self.retracted:
            src = (
                "the #692 DERIVATION IS RETRACTED and NOT re-grounded: it asks "
                "for depth %d, and %s (DEPTH VERDICT) measured that depth "
                "against this workload -- +6.9 %% on TTFT p50 at depth 1, "
                "+9.6 %% at depth 2, the bubble share did NOT fall (21.8/24.6 "
                "-> 26.8 -> 24.1 %%) and the deep point was inside its floor. "
                "The shipped depth is 0 until a MEASURED exchange-bound share "
                "exists to derive from; an explicit --p-microbatch-depth still "
                "wins and is still priced. " % (self.derived_depth, DEPTH_RETRACTION_RECORD)
                + src
            )
        if self.lowered_by_layout:
            src = (
                "the chosen layout is a GAPPED layer set, which admits exactly "
                "ONE pass in flight (scheduler_pp_mixin.init_pp_loop_state "
                "refuses the pair: under a gapped set each stage's next layer "
                "is another's previous one, so every stage must be inside the "
                "same forward). The depth is 0 BY THE LAYOUT -- the same one "
                "pass the PP-CUT solver priced that candidate at, which is why "
                "it is a derivation and not a lowered hand number. " + src
            )
        elif self.gapped_layout:
            # The bound is real and worth printing, but it did NOT act: the
            # depth was already at or below one pass. Saying "BY THE LAYOUT"
            # here would name a rule that never ran, which is the same
            # instrument-text defect as printing two causes for one 0.
            src = (
                "the chosen layout is a GAPPED layer set, which admits exactly "
                "ONE pass in flight; the depth was already 0, so the layout "
                "bound did not have to lower anything. " + src
            )
        # The bracket names TWO pass counts and says which one ships. The
        # shipped count is depth+1 by definition; the derived count is the
        # formula's output and ships only when no rule lowered it (fix 3,
        # review finding 1: printing the formula label over the shipped
        # value read "ceil(1/(1-stall_share))=1" on a retracted depth whose
        # formula gave 2).
        if self.measured is None:
            derived = "derived passes=n/a (no measurement)"
        else:
            derived = "derived passes=ceil(1/(1-stall_share))=%d" % (
                self.derived_depth + 1
            )
            if self.lowered_by:
                derived += " (NOT shipped: %s)" % self.lowered_by
            elif self.pinned and self.depth != self.derived_depth:
                derived += " (NOT shipped: pin overrides)"
        return (
            "WEG2 P-DEPTH solver:%s%s %s depth=%d price_rows=%d/stage "
            "price_mib=%s MiB/stage pool_after=%d (constraint pool >= %d) "
            "[passes_in_flight (shipped)=%d = depth+1; %s; each extra pass "
            "costs %d KV rows plus a %.1f MiB crossing frame per stage = %d "
            "pool tokens at the stage that converts worst, %d in total for "
            "depth %d; %s]"
            % (
                " PINNED (user override)" if self.pinned else "",
                " RETRACTED (derived %d, shipped %d)" % (self.derived_depth, self.depth)
                if self.retracted
                else "",
                shares,
                self.depth,
                self.price_rows,
                ",".join("%.1f" % v for v in self.price_mib),
                int(self.pool_after),
                int(self.cap_tokens),
                self.passes_in_flight,
                derived,
                self.chunk_tokens,
                self.act_mib_per_pass,
                int(self.price_tokens_per_pass),
                int(self.price_tokens),
                self.depth,
                src,
            )
        )


def solve_p_depth(
    measured: Optional[BubbleMeasurement],
    pool_tokens: float,
    attn_counts: Sequence[int],
    kv_mib_per_token_per_attn_layer: float,
    hidden_size: int,
    chunk_tokens: int,
    cap_tokens: int,
    gapped_layer_set: str = "",
    dtype_bytes: int = 2,
    pinned_depth: Optional[int] = None,
) -> DepthDecision:
    """Group P's microbatch depth, DERIVED from the previous boot's own bubble.

    THE MECHANISM. ``init_pp_loop_state`` reads this knob twice: it widens
    ``pp_loop_size = pp_size + depth``, and -- the half that matters for the
    bubble -- ``_event_loop_pp_body`` moves
    ``_pp_commit_send_output_work_and_preprocess_output_tensors`` from AFTER
    ``_pp_launch_batch`` to BEFORE it when the depth is non-zero. At depth 0
    a rank's output exchange for pass *i-1* therefore serialises with its
    launch of pass *i*, and that serialisation is exactly the host time the
    PP-BUBBLE meter measures between two forwards.

    THE DERIVATION. A stage busy ``forward/(gap+forward)`` of the time needs
    ``ceil(1/(1-stall_share))`` passes in flight to stay busy across the gap.
    One of those passes is the one it is forwarding, so the FLAG -- which
    counts passes BEYOND the ring's own -- is that number minus one. Stated
    rather than folded in, because it is where this function departs from the
    briefing's ``depth = ceil(1/(1-bubble_share))``: that form provisions one
    extra in-flight pass beyond the gap it has to cover, and every extra pass
    is charged to the KV pool below. Both terms are printed, so a reader who
    wants the other convention can see the arithmetic rather than infer it.

    THE #692 GATE, RE-READ FOR WEG 2. ``DESIGN_691_bubble_levers.md`` gated
    this lever on "measure it against the seam, not just against throughput",
    for two named costs. They do not survive equally here:

    * *"more live KV on cards already failing seam funding at 179 MiB"* -- does
      NOT apply. That was the one-process flip, where the live set at the seam
      had to be FUNDED to survive a cutover. In Weg 2 group P is its own
      process group and carries nothing across: its sleep releases ``kv_cache``
      and the weight tags wholesale to the memory saver (``sleep_group``), and
      the carrier between the phases is the HiCache store, which D reads. There
      is no funded live set for depth to grow. What depth does cost is priced
      here instead, in the only budget it actually touches: P's own KV pool,
      against ``--max-kv-per-request``.
    * *"a deeper pipeline has more state to quiesce, so seam entry takes
      longer"* -- DOES apply, unchanged, and is not priced here. The front's
      quiesce witness is ``/flush_cache`` returning 200 only when every
      in-flight term is zero (``front.py`` witness B), so ``depth`` extra
      in-flight chunks are ``depth`` extra chunk-forwards of drain before P can
      sleep. At the measured 369.6 ms per full chunk on the binding stage that
      is sub-second per unit of depth against a 15-17 s flip, which is why it
      is named and left to the flip's own measurement rather than converted
      into a second, unmeasured budget here.

    Refuses W42 rather than lowering the depth: a silently reduced depth is a
    hand number wearing a derivation.
    """
    kv_mib_per_token = [
        max(1, int(a)) * float(kv_mib_per_token_per_attn_layer) for a in attn_counts
    ]
    # The activation the extra pass keeps alive: one PPProxyTensors
    # hidden-states frame per stage boundary, [chunk, hidden] in the model
    # dtype. It is NOT in pp_cut.PhasePoolModel (which prices weights, mamba
    # state, the arming floor and KV only), so it is the one genuinely new
    # term -- and it is converted into pool tokens rather than charged against
    # a second MiB budget, because the pool model has already spent every free
    # MiB into tokens and charging both would be two books for one byte.
    act_mib_per_pass = float(chunk_tokens) * float(hidden_size) * float(dtype_bytes)
    act_mib_per_pass /= 1024.0 * 1024.0

    if measured is None:
        passes = 1
    else:
        stall = measured.stall_share
        passes = int(math.ceil(1.0 / (1.0 - stall))) if stall < 1.0 else 0
    depth = max(0, passes - 1)
    derived_depth = depth
    pinned = pinned_depth is not None
    if pinned:
        if int(pinned_depth) < 0:
            raise Weg2LaunchRefused(
                "W42 Weg2DepthUnfunded: --p-microbatch-depth %d is negative; "
                "the flag counts in-flight passes." % int(pinned_depth)
            )
        # A pin replaces the derivation, never the PRICE: it is announced and
        # then funded on exactly the same axis, so an override cannot buy a
        # depth the pool cannot hold.
        depth = int(pinned_depth)
        passes = depth + 1

    gapped_layout = bool(gapped_layer_set)
    # ONE ZERO, ONE CAUSE. Both rules below produce depth 0, so they are
    # branches of one ``elif`` and the acting one is RECORDED: a reader
    # grepping ``P-DEPTH solver:`` for why the depth is 0 gets the rule that
    # ran, not two candidate explanations. The LAYOUT is tested first because
    # it is the structural bound -- a gapped map cannot run more than one pass
    # whatever the measurement later says -- while the retraction is a verdict
    # on the derivation that a re-grounded #692 will lift.
    lowered_by = ""
    if depth > 0 and gapped_layout and not pinned:
        # DERIVED, not lowered. The bubble measurement asks for more passes,
        # the layout admits one, and the PP-CUT solver already RANKED this
        # candidate at one pass (CutCandidate: a gapped map's stages do not
        # overlap, so its makespan was the SUM). Taking the depth the layout
        # admits is therefore consistent with the number that chose it; the
        # line says so rather than printing a bare 0.
        depth, passes = 0, 1
        lowered_by = "layout"
    elif depth > 0 and not pinned and not DEPTH_DERIVATION_GROUNDED:
        # RETRACTED BY MEASUREMENT, not lowered by taste. The arithmetic above
        # is unchanged and is printed; what does not ship is its OUTPUT, because
        # BOOT_weg2pp2_0907.md measured this knob against this workload and it
        # lost on every column that has a floor (see DEPTH_DERIVATION_GROUNDED).
        # A derivation whose premise the metal refuted is not a default, and
        # keeping it would be a hand number wearing a derivation just as much as
        # silently lowering one would be.
        depth, passes = 0, 1
        lowered_by = "retraction"

    # Only a PIN can still be non-zero against a gapped map: the derived path
    # was taken to 0 by the branch above.
    if depth > 0 and gapped_layout:
        raise Weg2LaunchRefused(
            "W43 Weg2DepthGapped: the layer set %r puts group P on a "
            "GAPPED layer set, and the PINNED --pp-async-batch-depth %d would "
            "die at scheduler_pp_mixin.init_pp_loop_state, which refuses the "
            "pair outright: under a gapped set every stage must be inside the "
            "SAME forward, because each stage's next layer is another's "
            "previous one, and async depth lets a rank enter the next forward "
            "while a peer is still in the last one's output exchange. Refused "
            "here so the launcher says it, rather than three ranks discovering "
            "it after the weights are loaded. A DERIVED depth is taken down to "
            "0 by the layout instead (the solver priced this candidate at one "
            "pass); an OVERRIDE is refused, because an override the boot cannot "
            "run is not a tuning choice." % (gapped_layer_set, depth)
        )

    price_rows = depth * int(chunk_tokens)
    price_mib = tuple(
        depth * (float(chunk_tokens) * k + act_mib_per_pass) for k in kv_mib_per_token
    )
    # The stage that converts worst is the one with the FEWEST attention
    # layers: a token costs it less MiB, so the same MiB of crossing frame
    # costs it MORE tokens. The pool is a MIN over stages, so that stage is
    # the one that binds it.
    price_tokens_per_pass = max(
        float(chunk_tokens) + act_mib_per_pass / k for k in kv_mib_per_token
    )
    price_tokens = depth * price_tokens_per_pass if depth > 0 else 0.0
    pool_after = float(pool_tokens) - price_tokens

    if depth > 0 and pool_after < float(cap_tokens):
        # WHO ASKED FOR THIS DEPTH decides the first and last sentence. While
        # DEPTH_DERIVATION_GROUNDED is False a derived depth is always 0, so
        # the only depth that can reach this floor today is a PIN -- and
        # opening with "the measured bubble asks for" then attributed the
        # operator's own override to a measurement, printed that measurement's
        # stall_share as if it had produced the number, and closed with a
        # sentence ("a depth the bubble did not ask for is a hand number")
        # that refutes itself on exactly this path. Both branches below are
        # live: the derived one the day #692 re-grounds.
        if pinned:
            asked = (
                "--p-microbatch-depth %d was PINNED (the measurement derived "
                "%d), and %d passes in flight do not fit group P's pool"
                % (depth, derived_depth, passes)
            )
            closing = (
                "The pin is NOT quietly lowered to what fits: an override "
                "that silently becomes another number is a hand number "
                "wearing a pin. Re-pin the depth the pool can hold (0 is a "
                "pin), or fund this one."
            )
        else:
            asked = (
                "the measured bubble asks for --pp-async-batch-depth %d (%d "
                "passes in flight at stall_share %.3f), but group P's pool "
                "cannot fund it"
                % (
                    depth,
                    passes,
                    0.0 if measured is None else measured.stall_share,
                )
            )
            closing = (
                "The depth is NOT quietly lowered to what fits: a depth the "
                "bubble did not ask for is a hand number. Pin depth 0 to "
                "accept today's behaviour."
            )
        raise Weg2LaunchRefused(
            "W42 Weg2DepthUnfunded: %s. Pool is %d tokens; EACH extra pass "
            "costs %d KV rows plus a %.1f MiB crossing frame per stage = %d "
            "pool tokens at the worst-converting stage, so depth %d costs %d "
            "in total, leaving %d against the %d-token floor "
            "(--max-kv-per-request) -- short by %d. Raise the per-rank "
            "budgets or lower --max-kv-per-request. %s"
            % (
                asked,
                int(pool_tokens),
                int(chunk_tokens),
                act_mib_per_pass,
                int(price_tokens_per_pass),
                depth,
                int(price_tokens),
                int(pool_after),
                int(cap_tokens),
                int(float(cap_tokens) - pool_after),
                closing,
            )
        )

    return DepthDecision(
        depth=depth,
        derived_depth=derived_depth,
        passes_in_flight=passes,
        pinned=pinned,
        measured=measured,
        pool_tokens=float(pool_tokens),
        pool_after=pool_after,
        cap_tokens=int(cap_tokens),
        price_rows=price_rows,
        price_tokens=price_tokens,
        price_mib=price_mib,
        act_mib_per_pass=act_mib_per_pass,
        chunk_tokens=int(chunk_tokens),
        gapped_layout=gapped_layout,
        lowered_by=lowered_by,
        price_tokens_per_pass=price_tokens_per_pass,
    )


def _csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


@dataclass(frozen=True)
class PCutFacts:
    """What the solved cut knows that the depth price also needs.

    Handed on rather than re-derived: the pool, the attention split and the KV
    cell are already solved once here, and a second derivation beside them
    would be a second set of books for one physical fact.
    """

    stage_ratio: str
    attn_stage_ratio: str
    pool_tokens: float
    attn_counts: Tuple[int, ...]
    kv_mib_per_token_per_attn_layer: float
    hidden_size: int
    cap_tokens: int
    #: The solver's chosen ownership map in ``--pp-layer-set`` syntax, and
    #: whether it is GAPPED. Empty string = a contiguous cut, which needs no
    #: map at all: the count form expresses it and every path below stays
    #: byte-identical to what it was.
    layer_set: str = ""
    gapped: bool = False
    #: FIX 2: THE REALIZED PER-STAGE LAYER COUNTS OF THIS CUT -- the one fact
    #: the flip-order map needs and the only one that may not be re-derived
    #: from a score vector somewhere else. Not a convenience field: it is
    #: already ROUND-TRIPPED here against the runtime's own authority
    #: (``derive_pp_layer_split`` for the count form, ``parse_pp_layer_sets``
    #: for a gapped map) and a cut that fails that round trip is refused, so
    #: this is the split the boot provably runs. ``main`` reads it instead of
    #: asking ``p_stage_layers`` about the incumbent again.
    layer_counts: Tuple[int, ...] = ()


def flip_order_split(cut: PCutFacts, n_layers: int) -> Tuple[List[int], str]:
    """The per-stage layer counts the FLIP-ORDER MAP may be built from.

    FIX 2.  ``main`` used to derive this from the module score constants, at a
    point that runs before ``solve_p_cut`` -- so it answered for the INCUMBENT
    while group P was launched on the SOLVED cut, and the front got a map that
    was complete, confident and wrong (measured on this tip: incumbent
    ``[32,18,14]`` against the shipped maxkv cut's ``[31,17,16]``, which moves
    ``weights_3`` and ``weights_6`` onto other cards).  It comes off the cut
    now, and this function is the whole decision -- extracted from ``main`` so
    it can be tested by CALLING it rather than by reading main's AST, which is
    the only reason the previous instance of this defect had to be found by a
    reviewer instead of by a test.

    Returns ``([], reason)`` in the three cases where no honest map exists.
    NO MAP is the honest degradation: ``interleave_pause_order`` refuses an
    INCOMPLETE map and the front falls back to the identity pause order with
    the reason printed.  It does NOT refuse a complete-but-wrong one, which is
    the weg2dk4 death, so every case that cannot be stated exactly is stated
    as nothing.
    """
    split = list(cut.layer_counts)
    if not split:
        return [], "solve_p_cut published no layer counts"
    if sum(split) != n_layers:
        return [], f"solved split {split} sums to {sum(split)}, not {n_layers}"
    if cut.gapped:
        # A gapped cut's stages own NON-CONTIGUOUS layer bands, and
        # chunk_tag_cards walks cumulative per-stage counts -- that walk IS the
        # contiguous assumption, so no count vector can state this map.  Before
        # this fix a gapped boot silently got the incumbent's CONTIGUOUS map
        # here: the same wrong-map class, one layer deeper.
        return [], (
            f"the solved cut is GAPPED ({cut.layer_set}); its stages own "
            f"non-contiguous layer bands and chunk_tag_cards walks cumulative "
            f"contiguous counts, so no count vector can state this map"
        )
    return split, ""


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


def pick_shipped_cut(decision, incumbent_layers, incumbent_attn, objective: str):
    """WHICH priced candidate group P ships, and why.  Pure.

    ONE mechanism, reconciled on train 2 from two that had grown in parallel:
    the argv slice made the OBJECTIVE the solver's own argument (``maxkv`` by
    default, the standing law), while the train added a shipped-candidate
    picker over the solver's ranked field.  Both answered the same question --
    which of the priced rows the boot pays for -- and two answers to one
    question is the second-bookkeeping shape.  The flag name and default are
    the argv slice's; the ranked-field lookup and its refusal are the train's;
    ``incumbent`` becomes a third value of the one flag rather than a second
    flag.

    Why the picker is needed at all, measured, not asserted:

    * the maxkv law (user 2026-09-08, #1254): the pool-maximal cut and the
      makespan cut differ by 47.7 % of the KV pool on this box, and a boot may
      not pay that trade by accident.  That is why `makespan` stopped being the
      default;
    * boot weg2rg6 PROVED one cut on metal (32,18,14 / attention 8,4,4), and
      boot weg2tr1 then showed WHY that matters more than the ranking: the pool
      model does not price PP2's per-stage FIXED POSTS (lm_head, the MTP/draft
      head, the draft pools) and read that stage at 966,544 tokens against a
      PROFILED 155,164, while the kv-floor row wants SIXTEEN layers on it.  So
      `incumbent` is the default until #1259/#1019/#1260 put those posts in the
      model -- an exception WITH AN EXPIRY, stated at the flag, not a standing
      disagreement with the law.

    The candidate is looked up in the solver's OWN ranked field rather than
    constructed here -- a hand-built row would carry a pool figure this
    launcher invented.  An incumbent the solver did not rank is a REFUSAL (W40)
    naming it, never a silent fallback to the ranking's winner.
    """
    if objective == "makespan":
        return (decision.makespan or decision.chosen), (
            "the makespan-optimal cut (--pp-solve-objective makespan, THE "
            "DEFAULT since 2026-09-08 by user order -- verbatim: 'nimm als "
            "default ab jetzt makespan'). This is the solver's SPEED cut "
            "(42,11,11 / attn 10,3,3 in the perf boots) and it BUYS TTFT WITH "
            "KV: the pool it prices is smaller than the incumbent's, and both "
            "alternatives stay priced on this same line so the cost is visible "
            "every boot rather than inferred later"
        )
    if objective == "maxkv":
        return decision.kv_floor, (
            "the pool-maximal kv-floor cut (--pp-solve-objective maxkv): the "
            "standing maxkv law #1254 against the pool model AS IT STANDS -- "
            "which does not yet price PP2's lm_head/MTP/draft posts (#1259, "
            "#1019, #1260), the reason the shipped default is the incumbent"
        )
    cand = incumbent_candidate(decision, incumbent_layers, incumbent_attn)
    if cand is not None:
        return cand, (
            "the INCUMBENT cut (--pp-solve-objective incumbent, THE DEFAULT "
            "from boot weg2tr1 UNTIL 2026-09-08, when the user made makespan "
            "the default): boot-proven on weg2rg6, looked up in the "
            "solver's own ranked field so its pool is priced by this boot's "
            "model and not by that boot's. Default because the pool model does "
            "not price PP2's per-stage fixed posts (lm_head + MTP/draft head + "
            "draft pools; weg2tr1 priced 966,544 against a profiled 155,164), "
            "so the kv-floor row's 16 layers on PP2 maximise a number that is "
            "wrong there. Returns to maxkv with #1259/#1019/#1260"
        )
    want_layers = tuple(int(n) for n in incumbent_layers)
    want_attn = tuple(int(a) for a in incumbent_attn)
    raise Weg2LaunchRefused(
        "W40 Weg2PPCutRefused: --pp-solve-objective incumbent asks for the cut "
        f"{','.join(str(n) for n in want_layers)} / attention "
        f"{','.join(str(a) for a in want_attn)}, and the solver did not rank it "
        f"as a choosable candidate ({len(decision.ranked)} ranked). Refusing "
        "rather than shipping the ranking's winner under the incumbent's name "
        "-- that substitution is exactly what would desynchronise argv from the "
        "flip-order map."
    )


def incumbent_candidate(decision, incumbent_layers, incumbent_attn):
    """The incumbent's row in the solver's ranked field, or ``None``.

    Split out of :func:`pick_shipped_cut` because the PP-CUT SHIPPED line
    prices the incumbent on EVERY boot, including the boots that do not ship
    it -- an alternative that is not priced is an alternative nobody can
    compare, which is how the makespan default came to be paid unnoticed.  A
    missing row is printed as "not ranked"; only ASKING for it and not
    finding it is a refusal.
    """
    want_layers = tuple(int(n) for n in incumbent_layers)
    want_attn = tuple(int(a) for a in incumbent_attn)
    for cand in decision.ranked:
        if (
            cand.kind == "contiguous"
            and tuple(cand.layers) == want_layers
            and tuple(cand.attn) == want_attn
        ):
            return cand
    return None


def solve_p_cut(
    ns,
    cards: List[Card],
    budgets_p: List[int],
    model: str,
    log,
    chunk_tokens: int = 4096,
) -> PCutFacts:
    """Group P's layer + attention cut, and the ONE provenance line for it.

    Returns the flag strings plus the pool facts the depth solver prices
    against (:class:`PCutFacts`). Everything it feeds the solver is either measured on this box
    (NVML card names, the per-rank budgets this launcher just derived, the
    checkpoint's own weight headers and KV cell) or a flag whose default
    carries its provenance in the help text -- no number is chosen here.
    """
    from sglang.srt.planner import pp_cut as _pp_cut
    from sglang.srt.planner import pp_cut_launch as _cut

    cfg_path = os.path.join(model, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    text_cfg = cfg.get("text_config") or cfg
    n_layers = int(text_cfg["num_hidden_layers"])
    kinds = text_cfg.get("layer_types") or []
    n_attn = sum(1 for k in kinds if str(k) == "full_attention")
    if n_attn <= 0:
        raise Weg2LaunchRefused(
            f"W40 Weg2PPCutRefused: {cfg_path} states no full_attention layer "
            f"in layer_types, so the attention axis cannot be solved and the "
            f"KV pool cannot be priced. Refusing rather than defaulting."
        )
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
    ms = _csv_floats(ns.pp_cut_measured_ms_per_layer)
    model_pool = _pp_cut.PhasePoolModel(
        free_mib=tuple(float(b) for b in budgets_p),
        # The FAMILY split of weights is deliberately averaged: a stage's
        # divisor in stage_pp_capacities is its ATTENTION COUNT, and moving
        # one attention layer changes that divisor by 1 in 4-8 (12-25 %)
        # while changing weights by the attn/linear difference of a single
        # layer against a ~17-28 GiB free -- two orders of magnitude apart.
        # The total is exact for any cut summing to n_layers.
        weight_mib_per_layer=mean_layer_mib,
        kv_mib_per_token_per_attn_layer=kv_mib,
        arming_floor_mib=tuple(float(ns.pp_cut_arming_floor_mib) for _ in budgets_p),
        mamba_mib_per_linear_layer_per_slot=float(
            ns.pp_cut_mamba_mib_per_linear_layer_per_slot
        ),
        # Read off the argv this launcher builds rather than restated: a
        # second copy of --max-running-requests would drift the day the flag
        # moves, and the mamba residency scales linearly with it.
        # P's OWN bs (C1/R-12): this is P's pool model.
        mamba_slots=_max_running_requests(model, "P", int(getattr(ns, "p_bs", 8) or 8)),
    )
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
    incumbent = _csv_ints(ns.pp_stage_ratio) if ns.pp_stage_ratio else list(P_PP_STAGE_RATIO_SCORES)
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
    design_src = ns.pp_cut_design_prefix_from or newest_prefill_census_log(EVIDENCE_DIR)
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
            "FALLBACK: no P log in %s carries a prefill census, so the design "
            "prefix is one chunk -- the shallowest depth this boot can run. "
            "This is an absence of the instrument, not a measured shallow rig."
            % EVIDENCE_DIR
        )

    measured_attn = _pp_cut.attention_counts(families, incumbent)
    anchor_stage = next(
        (i for i, c in enumerate(cards) if "5090" not in c.name), len(cards) - 1
    )
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
    # THE CROSSING FRAME is the same object the depth price already charges:
    # one PPProxyTensors hidden-states frame, [chunk, hidden] in the model
    # dtype. Derived here from the same three numbers rather than restated.
    payload_bytes = int(chunk_tokens) * int(text_cfg["hidden_size"]) * 2
    lanes = pcie_lanes(cards)
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
    decision = _cut.solve_launch_cut(
        layer_families=families,
        incumbent_layers=incumbent,
        measured_ms_per_layer=ms,
        family_cost=family_cost,
        design_prefix_tokens=design_prefix,
        per_pair_crossing_ms=pair_ms,
        pinned_layer_set=ns.pp_layer_set or None,
        measured_provenance=(
            "MEASURED per-layer ms %s (boot bsscale, BSSCALE_0907.md tip "
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
    )
    log(
        f"PP-CUT inputs: layers={n_layers} attn={n_attn} "
        f"kv={kv_mib * 1024 * 1024:.0f} B/token/attn-layer (from config, fp8_e4m3) "
        f"weights attn {terms.attn_layer_weight_bytes / _pp_cut.MIB:.1f} / linear "
        f"{terms.linear_layer_weight_bytes / _pp_cut.MIB:.1f} MiB per layer -> mean "
        f"{mean_layer_mib:.1f} used; free={budgets_p} MiB (this launcher's own "
        f"per-rank budgets); arming floor {ns.pp_cut_arming_floor_mib} MiB/rank; "
        f"mamba/linear-layer/slot {ns.pp_cut_mamba_mib_per_linear_layer_per_slot} MiB "
        f"(0.0 = UNFUNDED, pool is an UPPER bound)"
    )
    log(decision.provenance_line())
    log(decision.trade_line())
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
    # ALL THREE ARMS PRICED ON EVERY BOOT, the shipped one named.  The two the
    # boot did NOT take are the whole reason this line exists: the makespan
    # default was paid for weeks because its alternative was never printed
    # beside it, and a trade nobody can see is a trade nobody selected.
    incumbent_row = incumbent_candidate(
        decision, P_PP_STAGE_RATIO_SCORES, P_PP_ATTN_STAGE_RATIO_SCORES
    )
    makespan_row = decision.makespan or decision.chosen
    log(
        "PP-CUT SHIPPED: layers=%s attn=%s pool_tokens=%d makespan_ms=%.1f -- %s. "
        "The two objectives this boot did NOT take stay priced beside it and are "
        "therefore not paid by accident: incumbent %s pool %s makespan %s, "
        "pool-maximal (kv-floor) %s pool %d makespan %.1f, makespan-optimal %s "
        "pool %d makespan %.1f (--pp-solve-objective incumbent|maxkv|makespan "
        "ships them)."
        % (
            ",".join(str(n) for n in chosen.layers),
            ",".join(str(a) for a in chosen.attn),
            int(chosen.pool_tokens),
            chosen.makespan_ms,
            ship_why,
            incumbent_row.fmt() if incumbent_row is not None else "%s / %s" % (
                _csv(P_PP_STAGE_RATIO_SCORES), _csv(P_PP_ATTN_STAGE_RATIO_SCORES)
            ),
            "%d" % int(incumbent_row.pool_tokens) if incumbent_row is not None
            else "n/a (NOT RANKED by this solve)",
            "%.1f" % incumbent_row.makespan_ms if incumbent_row is not None
            else "n/a",
            decision.kv_floor.fmt(),
            int(decision.kv_floor.pool_tokens),
            decision.kv_floor.makespan_ms,
            makespan_row.fmt(),
            int(makespan_row.pool_tokens),
            makespan_row.makespan_ms,
        )
    )
    if chosen.kind == "gapped":
        # A GAPPED map is not expressible as --pp-stage-ratio and does not go
        # through derive_pp_layer_split at all: it is published as the layer
        # SET and executed by the #753 mid-loop crossing wire. The round trip
        # below is therefore skipped by KIND, not by accident -- and the map is
        # instead round-tripped through the runtime's OWN parser, which is the
        # matching authority for this form.
        from sglang.srt.distributed.utils import parse_pp_layer_sets

        back = parse_pp_layer_sets(
            chosen.layer_set, n_layers, len(budgets_p), allow_gapped=True
        )
        realized_counts = tuple(len(x) for x in back)
        if realized_counts != chosen.layers:
            raise Weg2LaunchRefused(
                "W40 Weg2PPCutRefused: the gapped map this launcher solved "
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
        raise Weg2LaunchRefused(
            f"W40 Weg2PPCutRefused: the cut this launcher solved "
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
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug-hold", choices=["none", "P", "D", "both"], default="none")
    ap.add_argument("--store-min-gib", type=float, default=8.0)
    ap.add_argument("--weight-chunks", type=int, default=8,
                    help="#1233: number of weights_<k> layer-chunk tags per group (0 = the round-1 single tag / two-backup shape)")
    ap.add_argument("--ready-deadline-s", type=float, default=900.0)
    ap.add_argument("--extra-p", default="", help="extra flags for group P (shell-split)")
    ap.add_argument("--extra-d", default="", help="extra flags for group D (shell-split)")
    ap.add_argument("--fairness-w-s", type=float, default=45.0,
                    help="A1-1: the only sanctioned pre-emption; 0 disables it. Passed to the front.")
    ap.add_argument("--p-bs", type=int, default=8,
                    help="K1: group P's --max-running-requests AND the front's leg-1 concurrency. "
                         "Independent of --d-bs (law 2). Default 8 = today's shipped value. Also "
                         "sizes P's req_to_token_pool, so it is resolved before the budget solve.")
    ap.add_argument("--d-bs", type=int, default=8,
                    help="K2: group D's --max-running-requests AND the number of front seats, so "
                         "the front never hands D more concurrent requests than D can run.")
    ap.add_argument("--tp-prefill-max-tokens", type=int, default=None,
                    help="K5 (X): uncached tokens D may prefill itself. Unset = DERIVED as "
                         "2*flip_s/(1/r_D - 1/r_P) from this rig's own front-log rate and flip "
                         "lines, else the recorded PRE-BARLINK pair; floor = --chunked-prefill-size. "
                         "The derivation and its three inputs are printed at launch.")
    ap.add_argument("--flip-min-work-tokens", type=int, default=None,
                    help="K6: queued tokens that make a D->P round trip worth its cost. Unset = X.")
    ap.add_argument("--min-dwell-ms", type=float, default=None,
                    help="K7: minimum phase dwell. Unset = derived from the last flip in that direction.")
    ap.add_argument("--idle-layout", choices=["tp", "pp"], default="tp",
                    help="K8: which layout is awake at rest -- tp = group D (today's shape), "
                         "pp = group P. The front's idle guard always counts the requests P has "
                         "just prefilled, so resting on P loses no request.")
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
    ap.add_argument("--carrier-max-tokens", type=int, default=None,
                    help="#1246: ship a LOWER carrier bound than the one the census reads from group D's "
                         "own '#915 PREFETCH LIMIT' line. IT CAN ONLY LOWER IT: N is accepted exactly on "
                         "floor < N <= measured, where 'measured' is the budget group D's KV carrier "
                         "reports it will enforce and 'floor' is the route floor derived from the front's "
                         "own two bypass branches. N ABOVE measured is a W45 Weg2CarrierCensusRefused "
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
        "--transport", choices=["bar1", "nccl"], default="bar1",
        help="Collective transport for BOTH groups. 'bar1' is the shipping "
             "default. 'nccl' is the DEVELOPMENT mode of the user's order of "
             "2026-09-07: it drops the barlink flag family so the stock "
             "sglang dispatch builds a PyNccl communicator, and it raises "
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
    ap.add_argument("--evidence-dir", default=EVIDENCE_DIR,
                    help="where ring_table reads the previous boot's logs from")
    ap.add_argument("--duplex-probe", default=DUPLEX_PROBE_DEFAULT,
                    help="C12/C13 + A1-4: the step-0 probe RECORD the per-card PCIe "
                         "duplex ratio is solved from. A measured FILE, never a "
                         "number: the launcher parses its 'PROBE granule ... ratio=' "
                         "rows per card UUID, prints one WEG2-PCIE-DUPLEX line per "
                         "card with the R17 gate beside it, and publishes the table "
                         "to the ranks as SGLANG_WEG2_PCIE_DUPLEX. An unreadable or "
                         "rowless file means NO card splits its lock key")
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
             "solved cut. Unset = the solver decides.",
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
             "inherited SGLANG_PP_LAYER_SET in the launcher's environment is "
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
        "--pp-cut-design-prefix-from", default="",
        help="Path of the P log whose prefill census feeds "
             f"--pp-cut-design-prefix-tokens. Unset = the newest log in "
             f"{EVIDENCE_DIR} that actually carries the census (a log without "
             "it is skipped, never read as a measured prefix of 0).",
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
        "--pp-cut-mamba-mib-per-linear-layer-per-slot", type=float, default=0.0,
        help="Device GDN state per linear layer per sequence slot, MiB. "
             "Default 0.0 = UNFUNDED, and the direction is named rather than "
             "hidden: omitting it inflates every stage's capacity, more for "
             "stages holding more linear layers, so the printed pool is an "
             "UPPER bound and the pool floor is looser than reality. Pass a "
             "measured value to tighten it.",
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
             f"shipped when the card-rate library cannot price it (W52), when "
             f"its vector SATURATES a family -- a rank's proportional share "
             f"below one unit, so partition_units floors it and the axis is "
             f"pinned at its floor -- or resolves FLAT, which is even TP "
             f"wearing an uneven flag (W53), when it would need an env pin to "
             f"be honoured (W54), or when PerfCostModel.predict_capacity marks "
             f"it feasible=False (W55). A rank with ZERO heads is NOT among "
             f"them and never was: partition_units guarantees every rank at "
             f"least one unit, so that check could not fire. "
             f"Whatever is chosen, the launcher prints the WEG2 D-WEIGHTS line "
             f"naming the objective, the weight vector, the per-rank head "
             f"partition and the estimated cost at the attention barrier, PLUS "
             f"the WEG2 D-OPERATING-POINTS line carrying ALL THREE vectors "
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
        "--p-hicache-write-policy", default="write_through",
        choices=["write_through", "write_back", "write_through_selective"],
        help="MEASUREMENT ONLY. Group P's hicache write policy. Default "
             "write_through = unchanged shipped behaviour. 'off' is NOT "
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
        help=f"SGLANG_BARLINK_BUILD_WINDOW_CAP_S, both groups. Default "
             f"{BARLINK_BUILD_WINDOW_CAP_S} s. A TIMEOUT, not a measurement. "
             f"It is now launcher OUTPUT: it used to be written as "
             f"env.get(KEY, default), so an inherited export from the "
             f"launcher's own shell won silently and the boot ran a number no "
             f"flag, record or log line ever named (the R19 shape).",
    )
    ap.add_argument(
        "--pp-chain-recv-stall-s", type=int, default=PP_CHAIN_RECV_STALL_S,
        help=f"SGLANG_PP_CHAIN_RECV_STALL_S. Default {PP_CHAIN_RECV_STALL_S} s, "
             f"a timeout, launcher OUTPUT for the same reason as above.",
    )
    ap.add_argument(
        "--pp-occupant-horizon-s", type=int, default=PP_OCCUPANT_HORIZON_S,
        help=f"SGLANG_PP_OCCUPANT_HORIZON_S. Default {PP_OCCUPANT_HORIZON_S} s, "
             f"a timeout, launcher OUTPUT for the same reason as above.",
    )
    ap.add_argument(
        "--match-refusal-census-every", type=int, default=MATCH_REFUSAL_CENSUS_EVERY,
        help=f"SGLANG_MATCH_REFUSAL_CENSUS_EVERY. Default "
             f"{MATCH_REFUSAL_CENSUS_EVERY} is a census STRIDE. Launcher "
             f"OUTPUT for the same reason as above -- and note the denominator "
             f"law applies to what it emits: a rate-limited counter's zero is "
             f"not a zero.",
    )
    ap.add_argument(
        "--no-arming-floor-solved", dest="arming_floor_solved",
        action="store_false",
        help="Drop SGLANG_ARMING_FLOOR_SOLVED=1 (default ON, unchanged "
             "shipped behaviour). A boolean env gate with no flag was "
             "indistinguishable from a hard-coded constant (#1235).",
    )
    ap.add_argument(
        "--no-hicache-bigram-keys", dest="hicache_bigram_keys",
        action="store_false",
        help="Drop SGLANG_HICACHE_BIGRAM_KEYS=1 (default ON). MEASURED reason "
             "for ON, boot weg2ls3b3 (#1233): group D (NEXTN) keys the store "
             "by BIGRAM page hashes and group P (no spec) by UNIGRAM, so the "
             "two built disjoint chains for the same prompt and D never read "
             "P's pages. One key scheme for both groups; a no-op on D.",
    )
    ap.add_argument(
        "--no-hicache-flush-publish-sweep", dest="hicache_flush_publish_sweep",
        action="store_false",
        help="Drop SGLANG_HICACHE_FLUSH_PUBLISH_SWEEP=1 (default ON). Reason "
             "for ON (#1233 zero-remainder): /flush_cache publishes every "
             "un-backed device node to the store before the idle witness, so a "
             "chain the write-through pin budget declined mid-prefill is not "
             "lost at the flip.",
    )
    ap.add_argument("--teardown", default="", help="path of a boot state json to tear down")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ns = build_parser().parse_args(argv)
    if ns.teardown:
        return teardown(ns.teardown)
    # BEFORE build_env(), which starts from os.environ: an inherited stage map
    # would reach group P's ranks without passing through the solver at all.
    refuse_inherited_layer_set(os.environ)
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
    dry = ns.dry_run
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    front_log = None if dry else f"{base}.front.log"
    log = Log(front_log)
    log(f"=== WEG2 BOOT tag={ns.tag} tree={tree} @ {tip} ({'DIRTY: ' + dirty[:200] if dirty else 'clean'}) stamp={stamp} dry={dry}")
    if dirty and not dry:
        raise Weg2LaunchRefused("tree is not clean -- boot from a COMMITTED tip only")
    py = f"{ns.venv}/bin/python"
    state = BootState(tag=ns.tag, tip=tip, tree=tree, stamp=stamp)
    state.logs = {"front": front_log or "<dry>", "P": f"{base}.P.log", "D": f"{base}.D.log"}

    # 1. preflight
    state.shm_sweep = shm_residue_sweep(log, ns.tag, stamp, dry)
    sweep_dead_credit_counters(log, dry=dry)
    stale_deadman_sweep(log, [PORT_FRONT, PORT_P, PORT_D], dry)
    host_preflight(log, ns.tag, dry)
    cards = order_cards(resolve_cards())
    state.cards = [c.__dict__ for c in cards]
    if not dry:
        cards_free_check(cards, log)
    cvd = ",".join(c.uuid for c in cards)
    state.cvd = cvd
    log("NVML -> CUDA ordinal map: " + ", ".join(f"ordinal {i} = nvml {c.nvml_index} {c.name} {c.uuid} total {c.total_mib} MiB" for i, c in enumerate(cards)))

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
    x_seed = resolve_x(ns.tp_prefill_max_tokens, EVIDENCE_DIR, CHUNKED_PREFILL_TOKENS)
    x_tokens, x_provenance = x_seed.tokens, x_seed.provenance
    flip_min_work_tokens = int(ns.flip_min_work_tokens) if ns.flip_min_work_tokens is not None else x_tokens
    idle_layout_front = "P" if ns.idle_layout == "pp" else "D"
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
        f"({'= X, the same break-even at aggregate granularity' if ns.flip_min_work_tokens is None else 'operator override'})")

    log(f"SCHEDULING FLAGS AS EMITTED -- group P: --max-running-requests {p_bs} "
        f"--max-kv-per-request {max_kv_per_request} (no --tp-prefill-max-tokens: the PP prefill "
        f"group is not the one law 4 bounds); group D: --max-running-requests {d_bs} "
        f"--max-kv-per-request {max_kv_per_request} --tp-prefill-max-tokens {x_tokens}; "
        f"front: --p-concurrency {p_bs} --d-bs {d_bs} --tp-prefill-max-tokens {x_tokens} "
        f"--flip-min-work-tokens {flip_min_work_tokens} --idle-layout {idle_layout_front} "
        f"--drain-deadline-s {ns.drain_deadline_s} --fairness-w-s {ns.fairness_w_s}"
        + ("" if ns.min_dwell_ms is None else f" --min-dwell-ms {ns.min_dwell_ms}"))

    # 1b. #1233 one-backup flip geometry + the patched saver hook
    n_layers = model_num_layers(ns.model)
    chunk_count = max(0, int(ns.weight_chunks))
    chunk_layers = int(math.ceil(n_layers / chunk_count)) if chunk_count > 0 else 0
    from sglang.srt.managers.weg2_memory_saver import chunk_tag_cards, weights_family_tags
    weights_tags = weights_family_tags(chunk_count)
    log(f"WEG2-WEIGHT-CHUNKS N={chunk_count} tags (layers per chunk {chunk_layers} of {n_layers}; family {weights_tags}); "
        "flip (C9, gathered legs) = src.pause(kv) -> ONE src.release(family) and ONE dst.resume(family) "
        "in flight together -> dst.resume(kv); the host holds ONE image per card (H(c) = max_g image_g(c)) "
        "and dst's per-tag releases fund src's acquires inside it")

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
    dc_expect_d = {
        c.uuid: (DC_MEASURED_D_5090_MIB if "5090" in c.name else DC_MEASURED_D_3080_MIB) + slack_mib
        for c in cards
    }
    if ns.transport == "nccl":
        log(
            f"TRANSPORT=nccl (development mode, user order 2026-09-07): barlink flags dropped from both groups; "
            f"group D's dormant-residue slack raised {DC_RESERVE_SLACK_MIB} -> {slack_mib} MiB because libnccl's buffers "
            f"are not memory-saver-tagged and survive the sleep (MEASURED 2230 -> 2310 MiB on the 5090, boot weg2ab0 ARM 0, "
            f"where the unchanged 2228+64 reserve tripped W19 DormantResidueRefused). Under --transport bar1 this number "
            f"and the whole flip path are untouched."
        )
    state.dc_expect_d = dc_expect_d
    log("dormant residue RESERVE for group D = MEASURED D_c(D) of boot weg2ls1b2 (2228 / 1922 / 1922 MiB, "
        f"NVML per-process, windows included) + {slack_mib} MiB slack; spec 1.6 expectation was "
        f"{DC_EXPECT_5090_MIB}/{DC_EXPECT_3080_MIB} (exceeded); graded by W19 at D's first sleep: "
        + ", ".join(f"nvml{c.nvml_index}={dc_expect_d[c.uuid]}" for c in cards))
    budgets_p = budgets_from_dc(
        cards, dc_expect_d, log, "P", overshoot_mib=P_OVERSHOOT_MIB, overshoot_provenance="boot weg2ls2b2"
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
                     RING_FORM_SENTINEL_STORE_GIB, max_kv_per_request,
                     ns.p_hicache_write_policy, "P", ns.random_seed,
                     ns.barlink_bar1_cap_cycles, ns.collective_census_interval)
    )
    cut = solve_p_cut(ns, cards, budgets_p, ns.model, log, chunk_tokens=chunk_tokens)
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
        f"WEG2-PP-SPLIT group=P cut --pp-stage-ratio {cut.stage_ratio or '(omitted: gapped)'} "
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
        raise Weg2LaunchRefused(
            "W46 Weg2PPSplitMapMismatch: group P's argv would run the layer "
            f"split {','.join(str(n) for n in shipped_split)} (argv "
            f"--pp-stage-ratio {stage_ratio or '(none)'} --pp-attn-stage-ratio "
            f"{attn_stage_ratio or '(none)'}"
            + (f" --pp-layer-set {cut.layer_set}" if cut.layer_set else "")
            + f"), while the WEG2-FLIP-ORDER MAP was built from "
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
        "WEG2-PP-SPLIT CHECK: argv split "
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
    draft_kv_on_p = ns.draft_kv_on_p == "on"
    if not draft_kv_on_p:
        log(draft_kv_off_line())
    form_argv_p = argv_p(
        py, ns.model, budgets_p, RING_FORM_SENTINEL_S_GB, RING_FORM_SENTINEL_M_MIB,
        RING_FORM_SENTINEL_STORE_GIB, shlex.split(ns.extra_p), p_bs, max_kv_per_request,
        stage_ratio, attn_stage_ratio, ns.p_hicache_write_policy,
        RING_FORM_SENTINEL_DEPTH, ns.p_barlink_bar1_window_mib, ns.random_seed,
        ns.barlink_bar1_cap_cycles, ns.collective_census_interval,
        draft_kv_on_p,
    )
    form_key, form_norm = ring_table.p_form_key(form_argv_p)
    log(
        f"WEG2-P-FORM key={form_key} -- the identity of what group P LOADS, "
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
                                  tree=tree, py=py, p_argv=form_argv_p)
    state.ring_lines = ring_plan.lines
    state.ring_epoch = str(ring_plan.epoch)
    # A1-3: an un-armed ring has no fallback form to name.  The predecessor
    # named the pre-ring form here instead, which told the reader of a state
    # file that a form the launcher refuses to start had started.  The claim is
    # retracted rather than re-quoted -- reprinting it would hand the guard test
    # a false positive and the next reader a true one.
    state.ring_form = (ring_plan.form if ring_plan.armed
                       else "none -- NOT ARMED (A1-3: no fallback form exists on "
                            "this host budget; this boot refuses by name)")
    state.ring_dir = ring_plan.dir if (ring_plan.armed and ring_plan.form == "MAP_SHARED") else ""

    # 2. host ledger
    arm, store_gib, lines, cg = choose_host_ledger(
        ns.store_min_gib, ring_plan.host_weights_bytes,
        ring_plan.host_weights_span1_bytes, ring_plan.provenance)
    state.cgroup = dict(cg)
    for ln in lines:
        log(ln)
    state.ledger_lines = lines
    state.store_gib = store_gib

    # #1269 fix 4 follow-up: KEEP THE PREFLIGHT'S OWN ANON READING. `cg` is the
    # cgroup snapshot the ledger just took, BEFORE any group process exists, so
    # `cg["anon"]` is exactly the baseline W22's `sglang=/foreign=` split has to
    # subtract. It was already measured and already returned here, and was being
    # dropped -- sb5c's split was reconstructed by hand afterwards.
    #
    # ONE READ, CARRIED -- not a second `read_cgroup()` at front-launch time.
    # A baseline taken after the groups start is not a baseline, and two reads
    # are two quantities; the fix-2 lesson, on a measurement instead of a key.
    anon_preboot_bytes = int(cg.get("anon") or 0)
    log(f"WEG2-LAUNCH ANON-BASELINE preboot_anon={anon_preboot_bytes / 1073741824:.2f} GiB "
        f"(cgroup memory.stat anon at PREFLIGHT, before any group process exists; "
        f"handed to the front as --anon-preboot-bytes so W22 can split its reading into "
        f"sglang= and foreign=. 0 = unreadable -> the front prints NO split rather than "
        f"an invented one; sb5c had to be split by hand because this value was dropped)")

    # 2b. host memory time series from BEFORE the first group, so the launch
    # moment (P image resident, D loading) is measured this time, not only
    # the run moment (record 1h: the ls1b2 sampler started after D READY).
    if not dry:
        memts = f"{GPU_ARB}/memts_weg2_{ns.tag}.csv"
        mpid = int(subprocess.run(["bash", "-c", f"setsid {MEMTS} {shlex.quote(memts)} 5 > /dev/null 2>&1 & echo $!"], capture_output=True, text=True).stdout.strip() or 0)
        state.helper_pids.append(mpid)
        log(f"mem time series pid {mpid} csv {memts} (started before group P: launch AND run moments sampled)")

    # 3. store
    mount_store(log, store_gib, dry)
    store_dir = f"{STORE_MOUNT}/store"
    if not dry:
        os.makedirs(store_dir, exist_ok=True)

    # 4. group P
    # TRAIN FIX 5 moved the dc-reserve / budgets_p / cut solve ahead of the
    # ring (block 1b- above); only ``env_p`` stays here, because it is the
    # one thing in this step that genuinely needs the armed ring.
    env_p = build_env(tree, ns.venv, cvd, store_dir, ns.debug_hold in ("P", "both"), ns.tag, chunk_layers, chunk_count, tms_so, ns.transport, ring_plan, group="P", **_env_knobs(ns))
    # #1269 PUBLICATION: build_env's own comment says the decision is
    # "published explicitly so the boot log names the decision" -- but it only
    # SET the variables and logged nothing, so no boot log ever carried them.
    # A knob that shapes host memory and is never named is exactly the
    # instrument-that-does-not-report shape; the value is a CEILING derived
    # from the reset path's thread names, not a live count, and
    # WEG2-IDLE-CENSUS `threads=` is what confirms or corrects it.
    log(f"WEG2 MALLOC-ARENAS MALLOC_ARENA_MAX={env_p.get('MALLOC_ARENA_MAX')} "
        f"SGLANG_IDLE_BLOCKING_POLL={env_p.get('SGLANG_IDLE_BLOCKING_POLL')} "
        f"(both groups) -- provenance: glibc defaults to 8 x ncores arenas and this "
        f"box reports affinity 0-31, so the default ceiling is 256; measured growth was "
        f"64 MiB-aligned [anon] arenas extended in place plus NEW arenas appearing "
        f"(+5.4 MiB/min on P PP0/PP1, +2.8 on each D rank, +19.0 MiB/min over six ranks "
        f"against an independently observed +21.3). 4 = the scheduler thread plus the "
        f"three the reset path names (prefetch, backup, prefetch_io_aux), each with its "
        f"own arena. A CEILING, not a target. DERIVED FROM THREAD NAMES, NOT A LIVE "
        f"COUNT -- the rank thread census was never captured before the host OOM took "
        f"the base down; WEG2-IDLE-CENSUS threads= confirms or corrects it (#1269).")
    # #1233 zero-remainder: group P ends every prefill's last chunk at N-1 and
    # publishes the recurrent anchor there (schedule_policy END-OF-PREFILL
    # ANCHOR); D can claim at most N-1 tokens of a prompt, so this is the
    # anchor it resumes from. P only: D's finish anchors serve the NEXT turn.
    env_p["SGLANG_WEG2_END_ANCHOR"] = "1"
    # TRAIN FIX 5: the chunk size the cut solver was given BEFORE the ring is
    # re-read here against the arm this boot actually chose.  The hoist above
    # rests on --chunked-prefill-size being a CONSTANT of common_flags rather
    # than a function of the ledger arm; this line proves that premise on every
    # boot instead of assuming it, and a divergence is named rather than
    # silently shipping a cut solved for a chunk size the argv does not carry.
    chunk_tokens_armed = chunked_prefill_size_of(
        common_flags(ns.model, arm.s_gb, arm.m_mib, store_gib, max_kv_per_request,
                     ns.p_hicache_write_policy, "P", ns.random_seed,
                     ns.barlink_bar1_cap_cycles, ns.collective_census_interval)
    )
    if chunk_tokens_armed != chunk_tokens:
        raise Weg2LaunchRefused(
            f"W40 Weg2PPCutRefused: the PP cut was solved for "
            f"--chunked-prefill-size {chunk_tokens}, read before the host-ledger "
            f"arm existed, but the argv this boot ships carries "
            f"{chunk_tokens_armed}. The cut solve has to run BEFORE the ring "
            f"(the ring must be sized for the form that ships, and the cut is "
            f"part of that form), which is sound only while the chunk size does "
            f"not depend on the arm. It now does. Refusing rather than shipping "
            f"a cut solved against the wrong chunk."
        )
    # #1254 W46: THE MAP AND THE ARGV MUST BE THE SAME SPLIT.  The WEG2-FLIP-
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
            "WEG2 LAYER-SET published for group P: %s=%s%s (source: %s). The "
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
    bubble_src = ns.p_bubble_measured_from or newest_bubble_log(EVIDENCE_DIR)
    if ns.p_bubble_measured_from and read_pp_bubble(ns.p_bubble_measured_from) is None:
        raise Weg2LaunchRefused(
            f"W42 Weg2DepthUnfunded: --p-bubble-measured-from "
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
            "W56 Weg2EagerDecodeArm: --d-disable-cuda-graph was passed, so "
            "group D runs --disable-cuda-graph. THIS IS A CONTROL ARM: the "
            "decode form under measurement is now eager, not the graphed "
            "full-perf form, and boot weg2dec1_0909 measured the eager mean "
            "round at 16.6x the graphed one. Every ms/round, compute/wait and "
            "throughput figure from this boot carries that scope; none of "
            "them is a statement about the shipped form."
        )
    if ns.d_disable_overlap_schedule:
        log(
            "W41 Weg2OverlapRefused: --d-disable-overlap-schedule was passed, "
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
    if ns.p_hicache_write_policy != "write_through":
        log(
            f"MEASUREMENT ARM: group P --hicache-write-policy "
            f"{ns.p_hicache_write_policy} (shipped default is write_through; "
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
    log(f"WEG2-LAUNCH RPC auth=bearer sites={rpc_site_count()} "
        f"(every launcher->group HTTP call goes through one door, `http()`, which "
        f"attaches the admin bearer; sites counted from this module's own AST, not "
        f"a hand-kept list. weg2sb5 died with sites=2 and auth on NEITHER: the "
        f"front authenticated, the launcher's own startup sleep(P) did not, and "
        f"/release_memory_occupation is ADMIN_OPTIONAL -> 401 at 39 s. #1275 fix 2)")
    log(f"WEG2 ADMIN-KEY {'minted (DRY: not written)' if dry else 'minted'} for this boot -> {admin_key_file} (mode 0600); "
        f"both groups get --admin-api-key, the front authenticates its flip RPCs "
        f"with it, and /hicache/storage-backend/resize is LIVE (#1275). "
        f"Direct call: curl -s -X POST http://127.0.0.1:{PORT_D}/hicache/storage-backend/resize "
        f"-H \"Authorization: Bearer $(cat {admin_key_file})\" "
        f"-H 'Content-Type: application/json' -d '{{\"max_size_gb\": 8, \"min_free_gb\": 20}}'")
    shipped_argv_p = argv_p(py, ns.model, budgets_p, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_p), p_bs, max_kv_per_request, stage_ratio, attn_stage_ratio, ns.p_hicache_write_policy, depth_decision.depth, ns.p_barlink_bar1_window_mib, ns.random_seed, ns.barlink_bar1_cap_cycles, ns.collective_census_interval, draft_kv_on_p, admin_api_key=admin_api_key)
    # TRAIN FIX 5: THE SENTINEL PREMISE, PROVEN ON EVERY BOOT.  The form key that
    # gated the ring table was hashed over an argv built with sentinel ledger
    # terms, which is sound only while every flag those sentinels reach is
    # excluded from the key by name.  Here the REAL argv exists, so the key is
    # re-derived from it and any difference is a refusal -- the alternative is a
    # ring gated on a form that is not the one launched, which is a subtler
    # version of the very defect this gate exists to catch.
    shipped_key, shipped_norm = ring_table.p_form_key(shipped_argv_p)
    if shipped_key != form_key:
        mine, theirs = set(shipped_norm.split(" ")), set(form_norm.split(" "))
        raise ring_table.Weg2RingFormMismatch(
            f"W48 Weg2RingFormMismatch: the host ring was gated on group-P form "
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
    log(f"WEG2-P-FORM CONFIRMED key={shipped_key} on the argv actually shipped "
        f"(the ring was gated on this same key before the host-ledger arm "
        f"existed; the sentinel terms are proven inert, not assumed to be)")
    spec_p = GroupSpec("P", PORT_P, transport_argv(shipped_argv_p, ns.transport), state.logs["P"], env_p)
    state.argv["P"] = " ".join(shlex.quote(a) for a in spec_p.argv)
    log(w38_armed_line(spec_p.argv))
    state.deviations = [
        "transports stay OPEN across sleep (barlink_reopen() unwired this round; BAR1 windows sized to fit both groups: P 24+96, D 16+32+40 MiB "
        "= 208 of 224 usable, measured Used 224/256 incl. RM carve-out; #1234 C1 raised dcp:0 from 24 so the 96-MiB dcp all_reduce plans to 10 rounds "
        "instead of 17 and stops falling to the host-staged plane)",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 on both groups (K2); /health_generate only via the front to the awake group",
        "group deadmen tier 1 only (PROBE_S spaced past the boot); tier 2 on the front's deadman",
        "no disk tier for the store (file backend has no second tier); tmpfs sized by the ledger",
        "the epoch is carried in the front's ledger only (no group echo/checksum on the RPC)",
        "torch_memory_saver 0.0.9.post1 preload hook rebuilt from the vendored csrc with ONE patch (cpu backup freed after resume, "
        "python/sglang/srt/weg2/tms_csrc/PATCH.md); the stock wheel is untouched and used when SGLANG_WEG2_TMS_PRELOAD_SO is unset",
        f"weights paused/resumed as {chunk_count} weights_<k> chunk tags + the base tag (#1233 one-backup flip); cuda_graph stays resident (not in the sleep tag set, as in round 1)",
        "zero-remainder: group P holds the last token of every prefill back into its own chunk (SGLANG_WEG2_END_ANCHOR=1) so the GDN anchor D resumes from sits at N-1; "
        "costs P one 1-token pass per request and serialises whole-fit prompts behind the one chunked request per pass",
        "zero-remainder: /flush_cache publishes un-backed nodes before the idle witness on both groups (SGLANG_HICACHE_FLUSH_PUBLISH_SWEEP=1); the quiesce then waits for those backups",
        "zero-remainder: a BATCH prompt longer than group D's host staging pool can carry (launcher-measured from D's log, x0.9 prefetch bound) is served by ONE prefill on D "
        "(front route CARRIER-EXCEEDS, no leg 1) -- served and single-prefill, but NOT a zero-remainder leg 2; the windowed prefetch that would lift it is the next round",
        "zero-remainder: BATCH streams are priced post hoc via stream_options.include_usage (one standard trailing usage chunk reaches the client)",
    ]
    log(early_read_provenance())
    for d in state.deviations:
        log(f"DEVIATION (declared): {d}")
    launch_group(spec_p, tree, log, dry)
    if dry:
        budgets_d = budgets_from_dc(cards, {c.uuid: dc_expect_d[c.uuid] + P_WINDOWS_MIB - D_WINDOWS_MIB for c in cards}, log, "D(dry, expectation)")
        d_ratio = d_tp_ratio_decision(
            ns.d_tp_objective, ns.d_rank_perf_tune, cards, budgets_d, ns.model, d_bs
        )
        log(d_ratio.line)
        log(d_ratio.op_line)
        log(d_tokvec.line)
        env_d = build_env(tree, ns.venv, cvd, store_dir, False, ns.tag, chunk_layers, chunk_count, tms_so, ns.transport, ring_plan, group="D", **_env_knobs(ns))
        spec_d = GroupSpec("D", PORT_D, transport_argv(argv_d(py, ns.model, budgets_d, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_d), d_bs, max_kv_per_request, x_tokens, ns.num_continuous_decode_steps, ns.d_disable_overlap_schedule, d_ratio.flags, d_tokvec.flags, ns.random_seed, ns.barlink_bar1_cap_cycles, ns.collective_census_interval, ns.d_disable_cuda_graph, admin_api_key=admin_api_key), ns.transport), state.logs["D"], env_d)
        launch_group(spec_d, tree, log, dry)
        log("front argv (dry): " + " ".join(shlex.quote(a) for a in front_argv_for(
            py, store_dir, 0, 0, dc_expect_d, cards, ns, chunk_count, 0, p_bs, d_bs, x_tokens,
            flip_min_work_tokens, idle_layout_front, admin_key_file=admin_key_file,
            anon_preboot_bytes=anon_preboot_bytes)))
        log("DRY-RUN complete: nothing started, mounted, armed or written")
        return 0
    state.pids["P"] = spec_p.pid
    _write_state(state)
    state.t_ready["P"] = wait_ready(PORT_P, spec_p.pid, ns.ready_deadline_s, log, "P", spec_p.proc)
    n_kv = count_marker(spec_p.log, "#706 canonical KV page active")
    n_blob = count_marker(spec_p.log, "canonical GDN blob active")
    log(f"W7/W10 launcher half, group P log: '#706 canonical KV page active' x{n_kv}, 'canonical GDN blob active' x{n_blob} (need >= 3 each: three ranks)")
    if n_kv < 3 or n_blob < 3:
        raise Weg2LaunchRefused(f"W7 Weg2MambaBlobAbsent / W10 Weg2CanonicalPageMissing (launcher half): P logged kv x{n_kv} blob x{n_blob}, need 3 each")

    # 4d. sleep P, measure D_c(P) -- and, fix 8, P's DORMANT HOST IMAGE.
    # This is the one sleep on this box that is NOT interleaved: group D does
    # not exist yet, so nothing is resuming and freeing an image while P writes
    # its own.  The cgroup shmem delta and the per-rank RssShmem sum are
    # therefore comparable here, and the pair is what the next boot's ledger
    # prices its image term from (BOOT_weg2dk7_0907.md: the weight-tag census
    # under-charged the measured image by +9.80 GiB).
    shmem_before = host_ledger.read_cgroup_shmem_bytes()
    state.sleep_p_ms = sleep_group(PORT_P, log, "P", weights_tags)
    time.sleep(2)
    pids_p = session_pids(spec_p.pid)
    image_rec = host_ledger.dormant_image_sample(
        group="P",
        shmem_before_bytes=shmem_before,
        shmem_after_bytes=host_ledger.read_cgroup_shmem_bytes(),
        pids=sorted(pids_p),
        weight_tags_gib=host_ledger.WEIGHT_TAGS_P_BYTES / host_ledger.GIB,
        interleaved=False,
        boot_tag=ns.tag,
        commit=tip,
    )
    log(host_ledger.format_dormant_image(image_rec))
    host_ledger.append_measured_record(measured_record_path(), image_rec)
    state.dormant_image_p = image_rec
    dc_p = nvml_process_mib(pids_p)
    for c in cards:
        dc_p.setdefault(c.uuid, 0)
    mem = nvml_memory(cards)
    for c in cards:
        exp = DC_EXPECT_5090_MIB if "5090" in c.name else DC_EXPECT_3080_MIB
        log(
            f"WEG2-DC group=P nvml{c.nvml_index} {c.name}: measured {dc_p[c.uuid]} MiB per-process "
            f"(pids {sorted(pids_p)}) card used {mem[c.uuid].tenant_used_mib} MiB by processes "
            f"(instrument {mem[c.uuid].tenant_used_instrument}; card free {mem[c.uuid].free_mib} MiB allocatable, "
            f"{mem[c.uuid].reserved_mib} MiB driver-reserved); expectation {exp} + P windows {P_WINDOWS_MIB} = {exp + P_WINDOWS_MIB} MiB "
            f"({'AT OR BELOW' if dc_p[c.uuid] <= exp + P_WINDOWS_MIB else 'ABOVE'} expectation; the launcher derives D from the MEASUREMENT, record 1f B6)"
        )
    state.dc_measured_p = dc_p

    # 5. group D
    budgets_d = budgets_from_dc(
        cards, dc_p, log, "D", overshoot_mib=D_OVERSHOOT_MIB, overshoot_provenance="boot weg2ls4b1"
    )
    state.budgets["D"] = budgets_d
    d_ratio = d_tp_ratio_decision(
        ns.d_tp_objective, ns.d_rank_perf_tune, cards, budgets_d, ns.model, d_bs
    )
    log(d_ratio.line)
    log(d_ratio.op_line)
    log(d_tokvec.line)
    env_d = build_env(tree, ns.venv, cvd, store_dir, ns.debug_hold in ("D", "both"), ns.tag, chunk_layers, chunk_count, tms_so, ns.transport, ring_plan, group="D", **_env_knobs(ns))
    spec_d = GroupSpec("D", PORT_D, transport_argv(argv_d(py, ns.model, budgets_d, arm.s_gb, arm.m_mib, store_gib, shlex.split(ns.extra_d), d_bs, max_kv_per_request, x_tokens, ns.num_continuous_decode_steps, ns.d_disable_overlap_schedule, d_ratio.flags, d_tokvec.flags, ns.random_seed, ns.barlink_bar1_cap_cycles, ns.collective_census_interval, ns.d_disable_cuda_graph, admin_api_key=admin_api_key), ns.transport), state.logs["D"], env_d)
    state.argv["D"] = " ".join(shlex.quote(a) for a in spec_d.argv)
    launch_group(spec_d, tree, log, dry)
    state.pids["D"] = spec_d.pid
    _write_state(state)
    state.t_ready["D"] = wait_ready(PORT_D, spec_d.pid, ns.ready_deadline_s, log, "D", spec_d.proc)
    n_kv = count_marker(spec_d.log, "#706 canonical KV page active")
    n_blob = count_marker(spec_d.log, "canonical GDN blob active")
    log(f"W7/W10 launcher half, group D log: '#706 canonical KV page active' x{n_kv}, 'canonical GDN blob active' x{n_blob}")
    if n_kv < 3 or n_blob < 3:
        raise Weg2LaunchRefused(f"W7/W10 (launcher half): D logged kv x{n_kv} blob x{n_blob}, need 3 each")
    # #1233 zero-remainder (1j finding 6): W9 LAUNCH-TIME KEY-SCHEME GATE. The
    # store is one carrier; a spec-less group keys pages by unigram unless
    # SGLANG_HICACHE_BIGRAM_KEYS=1 forced the bigram scheme, a NEXTN/EAGLE
    # group keys by bigram natively. Divergent schemes wrote disjoint chains
    # for the same prompt once (boots weg2ls3b1-b3); the identity-suffix
    # check in the front cannot see it. Refuse the boot before any traffic.
    _forced = "#1233 HICACHE BIGRAM KEYS FORCED"
    _p_bigram = ("--speculative-algorithm" in spec_p.argv) or count_marker(spec_p.log, _forced) >= 1
    _d_bigram = ("--speculative-algorithm" in spec_d.argv) or count_marker(spec_d.log, _forced) >= 1
    log(f"W9 launch-time key-scheme gate: P bigram={_p_bigram} (forced lines {count_marker(spec_p.log, _forced)}) "
        f"D bigram={_d_bigram} (forced lines {count_marker(spec_d.log, _forced)})")
    if _p_bigram != _d_bigram:
        raise Weg2LaunchRefused(f"W9 Weg2StoreIdentityMismatch (launch-time key scheme): P bigram={_p_bigram} D bigram={_d_bigram} -- "
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
    # rejected alternatives: sglang/srt/weg2/carrier_census.py module docstring.
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
    from sglang.srt.weg2 import carrier_census as _cc

    # the floor is derived against THIS boot's SHORT bound (slice A's X), not
    # against a literal: the front routes SHORT on `remainder <= X`.
    _floor, _floor_why = _cc.route_floor(x_tokens)
    _expect_ranks = _cc.tp_size_of(spec_d.argv)
    _cen = _cc.census(spec_d.log, expected_ranks=_expect_ranks, floor=_floor)
    log(f"CARRIER BOUND: source='{_cc.SOURCE_MARKER}' in {spec_d.log}; component={_cc.COMPONENT}; "
        f"per-rank(TP)={_cen.per_rank} expected_ranks={_cen.expected_ranks} (from group D argv --tp-size); "
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
        raise Weg2LaunchRefused(_dec.detail)
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
    if not ring_table.p_carries_drafter(shipped_argv_p):
        log("W10/W11 SKIPPED (--draft-kv-on-p off): group P's argv carries no --speculative-* flag, so there is "
            "no draft-KV producer to grade -- no drafter identity to match against D's and no last-stage draft "
            "residue to hold against a budget. ROUTE UNDER THIS ARM: group D keeps its own NEXTN head and still "
            "asks the carrier for draft pages, but group P writes none, so D's draft reads miss and its draft "
            "state is COLD after every flip. Expect draft_pages=0 on the front's served lines; that is this "
            "arm, not a carrier fault. KV and Mamba across the flip are untouched.")
    else:
        w10 = check_drafter_identity(spec_p.log, spec_d.log)
        log(f"W10 DRAFTER-IDENTITY P={w10['P']} D={w10['D']} layout_P={w10['layout_P']} layout_D={w10['layout_D']} "
            f"match={w10['match']} (P lines {w10['n_P']}, D lines {w10['n_D']})")
        if not w10["match"]:
            raise Weg2LaunchRefused(f"W10 Weg2DrafterIdentityMismatch: P={w10['P']} D={w10['D']} layout_P={w10['layout_P']} "
                                    f"layout_D={w10['layout_D']} -- the decode group would ask the carrier for draft pages under "
                                    f"an identity the prefill group never writes")
        # #1233 fix 2: W11 DRAFT-RESIDENT gate. T_P (P_MAX_TOTAL_TOKENS) is derived
        # from a budgeted residue on P's last stage; the L2 line carries the
        # MEASURED one. Over budget = the corridor derivation is refuted by this
        # boot -> refuse before the front opens (boot weg2dk2's 3994 MiB build).
        gate_w11(spec_p.log, log)
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
    log(f"WEG2-FLIP-ORDER MAP group=P (SOLVED cut {cut.stage_ratio or '(gapped)'}/"
        f"{cut.attn_stage_ratio or '(gapped)'} -> REALIZED layer split "
        f"{p_split if p_split else 'REFUSED (' + p_split_note + ') -> NO MAP, identity order'} "
        f"over {n_layers} layers, {chunk_layers} layers per chunk, "
        f"nvml {[c.nvml_index for c in cards]} in stage order): {src_chunk_cards['P']}; "
        f"group=D TP -> no map (uniform across cards). Boot weg2dk4 died because the interleave paused P's tag k "
        f"against D's tag k while P's bytes for k=0..5 were on OTHER cards than the one D was allocating on.")
    front_argv = front_argv_for(
        py, store_dir, spec_p.pid, spec_d.pid, dc_expect_d, cards, ns, chunk_count,
        carrier_max_tokens, p_bs, d_bs, x_tokens, flip_min_work_tokens, idle_layout_front,
        src_chunk_cards=src_chunk_cards, measured_record=measured_record_path(),
        commit=tip, ledger_arm={"s_gb": arm.s_gb, "m_mib": arm.m_mib, "store_gib": store_gib},
        admin_key_file=admin_key_file,
        anon_preboot_bytes=anon_preboot_bytes,
    )
    fenv = dict(os.environ)
    fenv["PYTHONPATH"] = f"{tree}/python"
    # C14 / FIX 2 round 2: the BOOT half of the VRAM credit epoch.  Launcher
    # OUTPUT in exactly the class of TMS_HOST_RING_MAP (R19), never an operator
    # knob: it is this boot's ring epoch, the same nonce every rank already got
    # from build_env, so the front's composed <boot>.<flip> token and the ranks'
    # ring identity name ONE boot.  Absent (no armed ring) the front falls back
    # to its own start time, which is boot-unique for the same reason.
    if ring_plan is not None and ring_plan.armed:
        fenv["TMS_HOST_RING_EPOCH"] = str(ring_plan.epoch)
    log("front argv: " + " ".join(shlex.quote(a) for a in front_argv))
    ffh = open(front_log, "ab")
    fp = subprocess.Popen(front_argv, env=fenv, stdout=ffh, stderr=subprocess.STDOUT, cwd=tree, start_new_session=True)
    state.pids["front"] = fp.pid
    state.t_ready["front"] = wait_ready(PORT_FRONT, fp.pid, 120, log, "front", fp)

    # 7. deadmen + helpers
    huge = 10**7
    state.helper_pids.append(arm_deadman(log, spec_p.log, PORT_P, f"launch_server.*--port {PORT_P}", huge, ns.tag, "P", dry))
    state.helper_pids.append(arm_deadman(log, spec_d.log, PORT_D, f"launch_server.*--port {PORT_D}", huge, ns.tag, "D", dry))
    state.helper_pids.append(arm_deadman(log, front_log, PORT_FRONT, "sglang.srt.weg2.front", 120, ns.tag, "front", dry))
    os.system(f"ln -sfn {shlex.quote(front_log)} /root/current_boot.log")
    with open(f"{GPU_ARB}/weg2/boot_{ns.tag}.logpath", "w") as f:
        f.write(front_log + "\n")
    _write_state(state)
    log(f"LAUNCHED: P pid {spec_p.pid} (asleep) D pid {spec_d.pid} (awake) front pid {fp.pid}; state {state_path(state)}; /root/current_boot.log -> {front_log}")
    return 0


def front_argv_for(py: str, store_dir: str, p_pid: int, d_pid: int, dc_expect_d: Dict[str, int],
                   cards: List[Card], ns, chunk_count: int, carrier_max_tokens: int,
                   p_bs: int, d_bs: int, x_tokens: int, flip_min_work_tokens: int,
                   idle_layout_front: str,
                   src_chunk_cards: Optional[Dict[str, Dict[str, List[int]]]] = None,
                   measured_record: str = "", commit: str = "",
                   ledger_arm: Optional[Dict[str, float]] = None,
                   admin_key_file: str = "",
                   anon_preboot_bytes: int = 0) -> List[str]:
    """ONE front argv builder, so --dry-run prints exactly what a real boot runs.

    C2/R-6: the front is TOLD the two bs numbers and X. It never asks a
    group over HTTP -- that round trip would make the front's startup depend
    on a group the launcher has just put to sleep, for numbers the launcher
    itself wrote.
    """
    argv = [
        py, "-m", "sglang.srt.weg2.front",
        "--prefill", f"http://127.0.0.1:{PORT_P}", "--decode", f"http://127.0.0.1:{PORT_D}",
        "--port", str(PORT_FRONT), "--awake", "D", "--tag", ns.tag,
        "--store-dir", store_dir,
        "--prefill-sid", str(p_pid), "--decode-sid", str(d_pid),
        "--dc-reserve", ",".join(f"{c.uuid}={dc_expect_d.get(c.uuid, 0)}" for c in cards),
        "--fairness-w-s", str(ns.fairness_w_s),
        "--weight-chunks", str(chunk_count),
        "--carrier-max-tokens", str(carrier_max_tokens),
        "--p-concurrency", str(p_bs),
        "--d-bs", str(d_bs),
        "--tp-prefill-max-tokens", str(x_tokens),
        "--flip-min-work-tokens", str(flip_min_work_tokens),
        "--idle-layout", idle_layout_front,
        "--drain-deadline-s", str(ns.drain_deadline_s),
    ]
    # #1275: the front is handed the PATH, never the key. Its argv is as
    # world-readable as any other; the groups have no alternative (server_args
    # takes only --admin-api-key) but the front does, so it uses it.
    if admin_key_file:
        argv += ["--admin-key-file", admin_key_file]
    # #1269 fix 4 follow-up: THE PRE-BOOT ANON BASELINE, MEASURED ONCE AND
    # CARRIED. The W22 guard splits its reading into `sglang=` and `foreign=`
    # by subtracting this baseline, and the split is only subtractable because
    # it is THE SAME QUANTITY AT AN EARLIER TIME -- cgroup anon before this
    # boot's processes existed. The preflight already reads it
    # (`choose_host_ledger` -> `cg["anon"]`) and, until now, DROPPED it: boot
    # weg2sb5c's split had to be computed by hand afterwards. 0 stays "unset",
    # and the front then prints no split rather than an invented one.
    if anon_preboot_bytes > 0:
        argv += ["--anon-preboot-bytes", str(int(anon_preboot_bytes))]
    if ns.min_dwell_ms is not None:
        argv += ["--min-dwell-ms", str(ns.min_dwell_ms)]
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
    if measured_record:
        argv += ["--measured-record", measured_record]
    if commit:
        argv += ["--commit", commit]
    if ledger_arm:
        argv += ["--ledger-arm", json.dumps(ledger_arm, sort_keys=True)]
    return argv


def state_path(state: BootState) -> str:
    return f"{GPU_ARB}/weg2/boot_{state.tag}.json"


def _write_state(state: BootState) -> None:
    os.makedirs(f"{GPU_ARB}/weg2", exist_ok=True)
    with open(state_path(state), "w") as f:
        json.dump(state.__dict__, f, indent=1, default=str)


def teardown(path: str) -> int:
    st = json.load(open(path))
    print(f"[{_now()}] WEG2-TEARDOWN {path}")
    # #1275 fix 2 (3): THE KEY DIES WITH THE BOOT. Read from the state json --
    # a teardown runs in a FRESH process that never minted anything, so the
    # module global is empty here and the recorded path is the only source.
    set_admin_key(None, st.get("admin_key_file", ""))
    print(f"WEG2-TEARDOWN admin key file: {drop_admin_key_file()}")
    pids = set()
    for name, pid in st.get("pids", {}).items():
        if pid:
            pids |= session_pids(int(pid))
            pids.add(int(pid))
    out = subprocess.run(["pgrep", "-f", r"launch_server.*--port 3003[12]|sglang\.srt\.weg2\.front"], capture_output=True, text=True).stdout.split()
    pids |= {int(p) for p in out}
    for hp in st.get("helper_pids", []):
        if hp:
            try:
                os.kill(int(hp), signal.SIGTERM)
            except OSError:
                pass
    print(f"TERM {sorted(pids)}")
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
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
    time.sleep(3)
    mount = st.get("store_mount", STORE_MOUNT)
    if f" {mount} " in open("/proc/mounts").read():
        subprocess.run(["umount", mount], check=False)
        print(f"store tmpfs {mount} unmounted")
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
    return 0


class Weg2RingFormUnproven(ring_table.Weg2RingRefused):
    """W33: ``--ring-form`` named no form this tree has a proven build for.

    A subclass of the ring's own refusal base, so it inherits ``cli()``'s
    handler -- the one named line and exit 2 -- without being enumerated
    anywhere (FIX 2's lesson, applied to the new member rather than repeated).
    """


#: Every refusal class that must leave this launcher as the ONE named line and
#: exit 2, never as a traceback.  ``Weg2RingRefused`` is a BASE class on purpose
#: (FIX 2): the previous list enumerated its members, ``Weg2RingNeedsInterleave``
#: was not among them, and an explicitly requested ``--ring-form MAP_SHARED`` on
#: this rig therefore exited 1 with a stack trace -- which any wrapper keying on
#: the exit code reads as a crash rather than as the refusal it is.  A refusal
#: added to ring_table now inherits this handler instead of needing a line here.
#: #1233 fix 8: W21 (Weg2HostRunPeakRefused) is a refusal of the SAME standing
#: as W20 -- an arm whose predicted RUN PEAK is not below the observed reap
#: point does not boot -- so it joins this tuple instead of growing a second
#: `except` clause beside the one handler.
REFUSALS = (Weg2LaunchRefused, ring_table.Weg2RingRefused,
            host_ledger.Weg2HostLedgerRefused, host_ledger.Weg2HostRunPeakRefused)


def cli(argv: Optional[Sequence[str]] = None) -> int:
    """``main`` plus the one refusal handler.  Returns the exit code.

    A function rather than a bare ``__main__`` block so the handler is
    REACHABLE FROM A TEST: the defect it fixes was an except list that had gone
    out of step with the exceptions raised, and nothing could see it.
    """
    try:
        return main(argv)
    except REFUSALS as e:
        print(f"[{_now()}] WEG2-LAUNCH REFUSED: {e}", flush=True)
        # #1275 fix 2 (3): the KILLER path drops the key too. weg2sb5 refused
        # here and left its key file behind -- a boot that never served, whose
        # secret outlived it. Both exits, or the guarantee is only half true.
        print(f"[{_now()}] WEG2-LAUNCH admin key file: {drop_admin_key_file()}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
