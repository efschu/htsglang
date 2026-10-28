"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import os
import sys
import time

from flliper.srt.environ import envs
from flliper.srt.name_compat import tolerant_compile
from flliper.srt.pdflip import DEFAULT_D_BS_NEXTFLASH
from flliper.srt.pdflip import DEFAULT_D_POOL_WAVES_NEXTFLASH
from flliper.srt.pdflip import DEFAULT_D_SEAT_VRAM_NEXTFLASH
from flliper.srt.pdflip import budget_rest as _budget_rest
from flliper.srt.pdflip import card_identity
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import refusals
from flliper.srt.pdflip import state_file as state_file_mod
from flliper.srt.pdflip import xchg_residency
from flliper.srt.pdflip.launcher_d_leaf import DFLASH_DRAFT_PATH_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import _SPEC_FORM
from flliper.srt.pdflip.launcher_d_leaf import _alive
from flliper.srt.pdflip.launcher_d_leaf import _csv
from flliper.srt.pdflip.launcher_d_leaf import _csv_floats
from flliper.srt.pdflip.launcher_d_leaf import boot_state_dir
from flliper.srt.pdflip.launcher_d_leaf import bs_source
from flliper.srt.pdflip.launcher_d_leaf import dflash_produce_on_p
from flliper.srt.pdflip.launcher_d_mid import ATTN_ANCHOR_CARD_CLASS
from flliper.srt.pdflip.launcher_d_mid import CONTEXT_LENGTH_TOKENS
from flliper.srt.pdflip.launcher_d_mid import calib_power_limits
from flliper.srt.pdflip.launcher_d_mid import expert_store_dir_of
from flliper.srt.pdflip.launcher_d_mid import group_context_tokens
from flliper.srt.pdflip.launcher_log_plan import Log
from flliper.srt.pdflip.launcher_log_plan import StoreDiskPlan
from flliper.srt.pdflip.launcher_p_l3_leaf import L3_PERSIST_PREFIX
from flliper.srt.pdflip.launcher_p_l3_leaf import L3_REVOKED_SUFFIX
from flliper.srt.pdflip.launcher_p_l3_leaf import WEIGHT_SOURCE_EXCHANGE
from flliper.srt.pdflip.launcher_p_l3_leaf import _P_CHUNK
from flliper.srt.pdflip.launcher_p_l3_leaf import _free_instrument
from flliper.srt.pdflip.launcher_p_l3_leaf import _l3_move_revoked
from flliper.srt.pdflip.launcher_p_l3_leaf import _l3_revoked_path
from flliper.srt.pdflip.launcher_p_l3_leaf import _now
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst_boots
from flliper.srt.pdflip.launcher_p_l3_leaf import fs_of
from flliper.srt.pdflip.launcher_p_l3_leaf import nvml_memory
from flliper.srt.pdflip.launcher_p_l3_leaf import p_prefill_graph_bucket
from flliper.srt.pdflip.launcher_p_l3_leaf import parse_group_env
from flliper.srt.pdflip.launcher_p_mid import GGUF_DENSE_VOCAB_ENV
from flliper.srt.pdflip.launcher_p_mid import _model_config
from flliper.srt.pdflip.launcher_p_mid import _p_gguf_dense_vocab
from flliper.srt.pdflip.launcher_p_mid import l3_revoked_windows
from flliper.srt.pdflip.launcher_p_mid import profile_standard_form
from flliper.srt.pdflip.launcher_refuse_proc_leaf import PROFILE_QWEN27B
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _PROFILE_27B
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _XCHG_CENSUS_MAP
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _refuse_value_once
from flliper.srt.pdflip.launcher_refuse_proc_leaf import records_inventory
from flliper.srt.pdflip.launcher_refuse_proc_leaf import spec_form_is_dflash
from flliper.srt.pdflip.launcher_refuse_proc_leaf import split_preflight_refusals
from flliper.srt.pdflip.launcher_resolve_stop import planned_expert_store_bytes
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_types import PdFlipStoreArcRefused
from flliper.srt.pdflip.launcher_types import PdFlipStoreDiskRefused
from flliper.srt.pdflip.launcher_xchg_vision_leaf import _tree_bytes
from flliper.srt.planner import p_card_chunk as _p_card
import json
import math
import subprocess

MIB = 1024 * 1024
#: RIG PATHS, overridable for a container image (docker plan, stage B,
#: 2026-09-24). Every default is this rig's literal; with no FLLIPER_PDFLIP_* path
#: variable set, every value below is byte-identical to before. An EMPTY
#: variable counts as unset. Most of these have no CLI flag (--evidence-dir only
#: reaches the ring tables), so the environment is the one seam an image has.
EVIDENCE_DIR = os.environ.get("FLLIPER_PDFLIP_EVIDENCE_DIR") or "/spinning/evidence-665-f1"
GPU_ARB = os.environ.get("FLLIPER_PDFLIP_GPU_ARB") or "/spinning/gpu-arb"
#: The three rig helpers main() runs itself. Their own override, so an image can
#: ship them read-only while GPU_ARB is a writable state volume.
DEVTOOLS_DIR = os.environ.get("FLLIPER_PDFLIP_DEVTOOLS_DIR") or f"{GPU_ARB}/devtools"
HOST_PREFLIGHT = f"{DEVTOOLS_DIR}/host_ledger_preflight.sh"
#: #1236, USER RULING 2026-09-09: the Weg-2 page store is "normales hicaching
#: mit lvl2 und lvl3" -- the UPSTREAM HiCache file backend, pointed at a plain
#: directory on the ZFS dataset under /spinning.  NOT a tmpfs, NOT a new tier,
#: NOT an overflow mechanism, NOT a second bookkeeping: the ONLY things this
#: line owns are WHERE the backend's directory is and HOW BIG it is told to be.
#:
#: WHAT THIS REPLACES, and why the replacement is the whole fix: the store was
#: a tmpfs mounted per boot at ``/spinning/hicache-weg2-ram``, sized out of the
#: HOST RAM ledger's leftover.  Boot weg2sb5g got 6 GiB of it (max_size 5 G,
#: min_free 1 G) and REFUSED 1,324 group-P writes with "HiCacheFile ... would
#: fall below min_free" -- 5 GiB of usable store against a P KV pool of 304,950
#: tokens x 32,768 B = 9.30 GiB, i.e. the "#1236 Store >= P pool" law violated
#: by 1.86x before the first flip.  RAM was the wrong currency for a carrier
#: that has to be at least as large as a pool that already lives in RAM.
#: FLLIPER_PDFLIP_STORE_ROOT: see the rig-path block above.
STORE_ROOT = os.environ.get("FLLIPER_PDFLIP_STORE_ROOT") or "/spinning/hicache-weg2"
#: W57 (28.09.): whether the store root was TOLD (FLLIPER_PDFLIP_STORE_ROOT) or is
#: the rig default above. A launcher run without it measured the ZFS pool under
#: /spinning (171 GiB free, 93 % full) instead of the NF store the container
#: mounts from XFS [/l3/nf] (1.3 TB free) and refused the store by W57.
STORE_ROOT_TOLD = bool(os.environ.get("FLLIPER_PDFLIP_STORE_ROOT"))
#: HW-GENERIC 1002: calibration class -> (the record constant of the serving
#: form, the index into DC_MEASURED_D_XCHG_MIB of the exchange form). The
#: xchg triple is in NVML order of the reference rig (nvml0 3080, nvml1 5090,
#: nvml2 3080): the 5090 reads index 1, both 3080s read index 0 -- as before.
DC_MEASURED_D_BY_CLASS = {
    "RTX5090": ("DC_MEASURED_D_5090_MIB", 1),
    "RTX3080": ("DC_MEASURED_D_3080_MIB", 0),
}
P_DRAFT_RESIDENT_TOL_MIB = 256.0
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
#: guard in test_pdflip_host_budget_1233 pins that set.
P_PP_STAGE_RATIO_SCORES = (32, 18, 14)
#: Group P's per-stage FULL-ATTENTION scores (#485 ``--pp-attn-stage-ratio``),
#: the incumbent's other half, defined once for the same three readers.
P_PP_ATTN_STAGE_RATIO_SCORES = (8, 4, 4)
SPEC_NUM_DRAFT_TOKENS = 3
#: PP-POSTEN (27B, 29.09.): the measured stage_fixed increment of the YaRN x2 context over the x1 record
P_STAGE_FIXED_YARN2_DELTA_RECORD = "P_PP_STAGE_FIXED_YARN2_DELTA_MIB"
#: P's prefill chunk width (L2 lever, 24.09.): read ONCE at import from the
#: registered env so the cut solver, the depth funding and argv_p agree; the
#: TRAIN FIX 5 re-check below still proves that on every boot.  D keeps
#: CHUNKED_PREFILL_TOKENS (X's floor).
P_CHUNKED_PREFILL_TOKENS = int(envs.FLLIPER_PDFLIP_P_CHUNKED_PREFILL_TOKENS.get())
#: PPCUT-INT4AR 24.10.: WHICH CHECKPOINT'S non-layer weights the profile's
#: P_PP_STAGE_FIXED_MIB record carries, per profile -- (label, embedding MiB,
#: lm_head MiB), read from that checkpoint's safetensors headers.  The 27B
#: record (PP-POSTEN 29.09.) was fitted on boots 09291750/09291831/09291331,
#: whose P argv names --model-path .../Qwen3.8-27B-INT8-gdncov-vocabembed:
#: embed_tokens I8 [248320, 5120] 1212.5 + weight_scale 0.47 MiB, lm_head BF16
#: [248320, 5120] 2425.0 MiB.  ``solve_p_cut`` moves the record onto the
#: booted checkpoint's own header bytes (pp_cut_launch.stage_fixed_checkpoint_
#: delta, one-sided); a profile without an entry keeps its record unchanged
#: and says so.  Measured reason: boot dkr27bint4arrc4bar1int4ar10241254 (BF16
#: embedding) realised PP0 226647 tokens where the record priced 301650.
P_PP_STAGE_FIXED_REF_NONLAYER_MIB: Dict[str, Tuple[str, float, float]] = {
    "qwen27b": ("Qwen3.8-27B-INT8-gdncov-vocabembed", 1212.97, 2425.0),
}
#: Task #114 (24.09.), fnFL2 H41: group P's prefill TRANSIENT per stage as a
#: FUNCTION OF THE CHUNK -- measured support points, linear between them,
#: REFUSED (W131) outside, never extrapolated
#: (``planner.p_card_chunk.P_TRANSIENT_SUPPORT_FNFL2``; the boot's own
#: ``[vram-peak]`` instrument, peak minus allocated of the full chunk):
#:   chunk    PP0 (5090)  PP1 (3080)  PP2 (3080)
#:     512      0.13        0.13        0.13      GiB  x130, x134
#:    4096      0.92        0.82        0.85           x12, x14, x101, x107
#:    8192      1.82        1.63        1.60           x113, x116
#:   16384      3.61        3.21        3.19           x118, x141, x145, x146
#: fnFL2x121/x122 (5090 stage at residency 0.45/0.40, chunk 16384) passed
#: every launch gate and died in the FIRST chunk's forward (CUDA OOM, 320 and
#: 384 MiB short). #114 (add357d7c2) carried per-token slopes and read PP2 at
#: 2.43 GiB -- the DRAFT forward's line after a peak reset; the target forward
#: on PP2 draws 3.19 GiB (x118 02:36:24, and x145/x146 without a draft on P).
P_PREFILL_TRANSIENT_SUPPORT = _p_card.P_TRANSIENT_SUPPORT_FNFL2
#: #242: the profile's OWN measured prefill transient per P stage, MiB, at the
#: chunk :data:`P_ACTIVATION_RECORD_CHUNK` (``profile_records_data/<profile>
#: .json``). Where a profile carries it, the record REPLACES the builtin
#: support point of that chunk (H94: RECORD > BUILTIN): the runtime's post
#: (``P_PREFILL_TRANSIENT_ENV``) and the pool model book the SAME bytes per
#: stage. The builtin point (fnFL2x118..x146, 3697/3287/3267) was taken with
#: one sequence per chunk; the NF Dauerlauf of 28.09. packs two to five
#: sequences into one 16k chunk, and PP0 then draws 4594 MiB while PP1/PP2 stay
#: at 2945/2941 (planner definition peak - allocated, every PDFLIP-VRAM-PEAK
#: chunk window of 15 boots).
P_ACTIVATION_RECORD = "P_ACTIVATION_MIB"
#: THE MEASURED COMPOSITION OF A REAL WEG-2 STORE, from which the sidecar
#: headroom factor below is DERIVED rather than picked.  Provenance, both
#: halves on this rig and both quoted rather than recalled:
#:
#: * FILE COUNTS -- boot ``weg2sb5g``'s own ``W9 store census (shard-walked)``
#:   line, 2026-09-09T07:12:38Z, store dir ``/spinning/hicache-weg2-ram/store``:
#:   ``files=76733 kv=50651 mamba=42 draft=26040``.
#: * BYTE WEIGHTS -- ``stat`` on the store residue ``/spinning/hicache-l3``
#:   (same model, identity ``b023832e91bf5745``), 2026-09-09: a KV page is
#:   32,768 B (three samples, all exact); the three DCP shards of ONE mamba
#:   node (``feb21ed4a261...``) are 39,223,296 + 19,611,648 + 19,611,648 B,
#:   summing to 78,446,592 B = 74.81 MiB of GDN state per radix node -- which
#:   is what the #706 CANONICAL blob holds in one file, geometry-neutral.
#:   A draft page is :data:`host_ledger.DRAFT_PAGE_BYTES` = 2,048 B.
#:
#: THE HONEST LIMIT OF THIS RATIO, stated because a factor invites being read
#: as a constant of nature: it is ONE snapshot of ONE boot's mix.  The draft
#: half is structurally bounded (2,048 B beside every 32,768 B page = 1/16, and
#: the census sat below that at 0.51 draft pages per KV page).  The MAMBA half
#: is not: a blob is written per RADIX NODE, so its share moves with chunk
#: granularity and prefix reuse, and at this census it was the DOMINANT term --
#: 3.29 GB against 1.66 GB of KV.  That is also the arithmetic that explains
#: sb5g's 1,324 refusals without any further hypothesis: kv + mamba + draft =
#: 5.01 GB against 5 GiB of usable tmpfs, i.e. the store was already at its
#: wall.  The factor is therefore a SIZING INPUT with a flag
#: (``--store-sidecar-factor``), not a law, and on disk it is cheap to be
#: generous with -- which is the point of moving off RAM.
STORE_CENSUS_PROVENANCE = _PROFILE_27B.constant("STORE_CENSUS_PROVENANCE")
STORE_CENSUS_KV_PAGES = _PROFILE_27B.constant("STORE_CENSUS_KV_PAGES")
STORE_CENSUS_MAMBA_BLOBS = _PROFILE_27B.constant("STORE_CENSUS_MAMBA_BLOBS")
STORE_CENSUS_DRAFT_PAGES = _PROFILE_27B.constant("STORE_CENSUS_DRAFT_PAGES")
STORE_CENSUS_KV_PAGE_BYTES = _PROFILE_27B.constant("STORE_CENSUS_KV_PAGE_BYTES")
#: 39,223,296 + 19,611,648 + 19,611,648 measured on /spinning/hicache-l3.
STORE_MAMBA_BLOB_BYTES = 39_223_296 + 19_611_648 + 19_611_648
_CENSUS_KV_BYTES = STORE_CENSUS_KV_PAGES * STORE_CENSUS_KV_PAGE_BYTES
_CENSUS_MAMBA_BYTES = STORE_CENSUS_MAMBA_BLOBS * STORE_MAMBA_BLOB_BYTES
_CENSUS_DRAFT_BYTES = STORE_CENSUS_DRAFT_PAGES * host_ledger.DRAFT_PAGE_BYTES
#: (kv + mamba + draft) / kv at that census = 3.02.  Multiply the P group's KV
#: pool bytes by this to size the WHOLE store, sidecars included.
STORE_SIDECAR_FACTOR = (
    _CENSUS_KV_BYTES + _CENSUS_MAMBA_BYTES + _CENSUS_DRAFT_BYTES
) / float(_CENSUS_KV_BYTES)
#: ``min_free_space`` FOR THE DISK, and the reason it is not the old 1 GiB:
#: that 1 GiB was 1/6th of a 6 GiB tmpfs and it latched the write stop after
#: 5 GiB.  Here the filesystem is the CONTAINER'S ROOT (``/`` is
#: ``spinning/subvol-999-disk-0``; ``df`` 2026-09-09: 2.3 T, 622 G free), so
#: the floor protects the whole box, not the store: below it this rig's own
#: boot logs (a P+D+front triple measured ~2.5 GB on weg2sb4), the evidence
#: tree and the model cache start failing.  32 GiB is ~13 such boots of
#: headroom.  It is NOT a ZFS performance-knee margin -- the pool is already
#: 74 % full, so that knee is behind us and is a different conversation.
STORE_DISK_MIN_FREE_GIB = 32.0
L3_CAP_FILE = "L3_CAP.json"
_RESIDENT_RE = tolerant_compile(r"PDFLIP DRAFT-KV-PRODUCER armed .*resident_mib=(-?\d+(?:\.\d+)?)")
_HEAD_RELEASED_RE = tolerant_compile(r"PDFLIP DRAFT-KV-PRODUCER armed .*head_released_mib=(-?\d+(?:\.\d+)?)")
#: #66 (fnFL2v72): the fourth term -- what the private TAG POOLS still cache.
#: empty_cache cannot reach them (#65 of the same day), so NVML counts those
#: bytes while residue and release cannot see them: 5334.0 against 2202.6 +
#: 1212.5 left 1918.9 MiB unexplained and W11b refused a build that was fine.
_TAG_POOL_INACTIVE_RE = tolerant_compile(r"PDFLIP DRAFT-KV-PRODUCER armed .*tag_pool_inactive_mib=(-?\d+(?:\.\d+)?)")
#: #66: die zwei Terme, die W11b fehlten, als 533,7 MiB unerklaert blieben.
_OUTSIDE_TORCH_RE = tolerant_compile(r"PDFLIP DRAFT-KV-PRODUCER armed .*outside_torch_mib=(-?\d+(?:\.\d+)?)")
_DEFAULT_POOL_RE = tolerant_compile(r"PDFLIP DRAFT-KV-PRODUCER armed .*default_pool_inactive_mib=(-?\d+(?:\.\d+)?)")
_NVML_DELTA_RE = tolerant_compile(r"PDFLIP DRAFT-KV-PRODUCER armed .*nvml_delta_mib=(-?\d+(?:\.\d+)?)")
#: #66: der lebende Nicht-Modell-Posten (Attention-Workspace voran) und der
#: freie Rand der Karte, an dem gemessen wird, ob ein Rest ueberhaupt
#: gefaehrlich waere.
_OTHER_LIVE_RE = tolerant_compile(r"PDFLIP DRAFT-KV-PRODUCER armed .*other_live_mib=(-?\d+(?:\.\d+)?)")
_CARD_FREE_RE = tolerant_compile(r"PDFLIP DRAFT-KV-PRODUCER armed .*card_free_mib=(-?\d+(?:\.\d+)?)")
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
_STATE_MISSING_LOGGED: set = set()
#: Ops #164 (22.10.): the working directory the groups (P, D) and the front are
#: spawned in. The kernel writes a crash core to "core.<pid>" RELATIVE TO THE
#: CRASHING PROCESS'S CWD (proxmox: kernel.core_pattern=core, core_uses_pid=1;
#: docker's LimitCORE=infinity reaches every container), and the cwd used to be
#: the code tree. A core in the tree makes `git status --porcelain` dirty, and the
#: container's next `docker start` was REFUSED TREE by the entrypoint's
#: preflight_tree (27B night 22.10. 00:47Z, workaround recreate). The cwd is now
#: <evidence>/core/<boot stem>/ on the evidence volume: the core survives as
#: evidence there, the tree stays clean. One directory per boot, because the
#: kernel UNLINKS an existing core of the same name and container pids restart
#: low on every start (core.967 of one boot would replace core.967 of the last).
#: Nothing reads relative to the cwd: argv and env carry absolute paths, the
#: package comes from PYTHONPATH. "0" keeps the tree (the old cwd).
CORE_CWD_ENV = "FLLIPER_PDFLIP_CORE_CWD"
#: 29.09. (Planer post COLD_TIER_SHM): filesystems whose pages are shmem --
#: unevictable without swap, charged to the cgroup that touched them.
COLD_TIER_SHM_FS = ("tmpfs", "ramfs")
COLD_TIER_MARKER = "PDFLIP-HOST-LEDGER COLD-TIER"

def dc_measured_d_mib(card: Card, weight_source: str) -> int:
    """Group D's MEASURED dormant residue for this card, on THIS form.

    THE ONE SELECTOR, so there is no second bookkeeping and no parallel
    reserve object: every consumer of `DC_MEASURED_D_*` goes through here, which
    is also why group P's per-card budget follows the form automatically
    (`budgets_from_dc` subtracts this term).  Under the exchange arm P's budget
    therefore shrinks by 666 MiB on each 3080 and 856 MiB on the 5090 -- the
    residency is real and the budget that ignored it is what tripped W19.

    An unrecognised board REFUSES rather than borrowing a number: both triples
    name this rig's inventory only, and a third model's residue is not a
    property either of them has.
    """
    name = str(getattr(card, "name", ""))
    xchg = str(weight_source) == WEIGHT_SOURCE_EXCHANGE
    # HW-GENERIC 1002: selected by CALIBRATION CLASS (model + arch + VRAM
    # tier, pdflip/card_identity.py), never by a name substring -- a 10 GB RTX
    # 3080 used to borrow the 20 GB card's residue here.
    cls = card_identity.calibration_class(card)
    sel = DC_MEASURED_D_BY_CLASS.get(cls or "")
    if sel is None:
        # AP1 1006 (HW-BORROWED by arch, Q-710: flags instead of code): a board of
        # NO calibrated class is refused as before -- but as the VALUE refusal
        # HW-UNCALIBRATED (``refusals.refuse_value``), so ``--force`` can pass it and
        # the boot takes the figure of the calibrated class of the SAME ARCH (sm_86 ->
        # RTX3080, sm_120 -> RTX5090; ``card_identity.arch_twin_class``), named
        # UNMEASURED on this card (FORCED-PAST line). No twin of the arch (sm_89):
        # the refusal stays hard. A calibrated card never gets here (its own class
        # wins above): the reference rig is byte for byte what it was.
        twin = card_identity.arch_twin_class(card)
        twin_sel = DC_MEASURED_D_BY_CLASS.get(twin or "")
        text = (
            f"W19 dormant-residue reserve ({card_identity.CODE_UNCALIBRATED}): board {name!r} (nvml"
            f"{getattr(card, 'nvml_index', '?')}, {card_identity.card_key(card)}) belongs to no "
            "calibrated class of DC_MEASURED_D_* / DC_MEASURED_D_XCHG_* (measured on: "
            + ", ".join(sorted(DC_MEASURED_D_BY_CLASS)) + ") -- a third card's dormant residue "
            "is not a property either of them has.  Measure it, do not borrow it."
        )
        if twin_sel is None:
            raise PdFlipLaunchRefused(text)
        _refuse_value_once(
            card_identity.CODE_UNCALIBRATED,
            text + f"  --force BORROWS the dormant residue of class {twin} (same arch "
            f"{card_identity._sm(card_identity.props_of(card).cc)}) for this card: "
            "UNMEASURED here (HW-UNCALIBRATED).")
        sel = twin_sel
    # UNIFY S3: the published profile's row (pdflip/form.py), not the 27B alias.
    const_name, xchg_index = sel
    return _pconst("DC_MEASURED_D_XCHG_MIB")[xchg_index] if xchg else _pconst(const_name)

def p_chunked_prefill_tokens() -> int:
    """Group P's --chunked-prefill-size: the graph bucket when the P prefill
    graph is on, the common constant otherwise (byte-identical default).
    Under --p-chunk-policy dynamic it is the plan's CEILING (--p-chunk-max):
    the per-forward width comes from the plan, and every allocation sized
    from this value (corridor, frames, barlink buffers) must fund the widest
    chunk the plan may choose."""
    spec = _P_CHUNK.get("spec")
    if spec is not None:
        return int(spec.limits.max_tokens)
    # UNIFY S7: the common constant is group P's own (P_CHUNKED_PREFILL_TOKENS,
    # the NF form's FLLIPER_PDFLIP_P_CHUNKED_PREFILL_TOKENS; default = the
    # constant, byte-identical on the 27B line).
    return p_prefill_graph_bucket() or P_CHUNKED_PREFILL_TOKENS

def p_stage_fixed_ctx_delta_record(ns, n_stages: int) -> Tuple[Optional[Tuple[float, ...]], str]:
    """The profile's MEASURED stage_fixed increment for group P's context, where it has one:
    ``P_PP_STAGE_FIXED_YARN2_DELTA_MIB`` for a P context of 2 x 262144 (YaRN x2). ``(None, '')``
    otherwise -- the RoPE estimate (p_rope_context_delta_mib) stands."""
    ctx, _src = group_context_tokens(ns, "p")
    if int(ctx) != 2 * CONTEXT_LENGTH_TOKENS:
        return None, ""
    try:
        vals = tuple(float(v) for v in _pconst(P_STAGE_FIXED_YARN2_DELTA_RECORD, getattr(ns, "profile", None)))
    except KeyError:
        return None, ""
    if len(vals) != int(n_stages):
        return None, ""
    return vals, (f"PP-CUT STAGE-FIXED KONTEXT (gemessen): P context {ctx} -> stage_fixed += "
                  f"{[round(v, 1) for v in vals]} MiB ({P_STAGE_FIXED_YARN2_DELTA_RECORD}, boots "
                  f"{_pconst_boots(P_STAGE_FIXED_YARN2_DELTA_RECORD, getattr(ns, 'profile', None))}) "
                  f"statt der RoPE-Schaetzung")

def p_draft_kv_cell_bytes_by_stage(ns, n_stages: int, target_text_cfg: dict,
                                   kv_dtype_bytes: int = 1) -> Tuple[Tuple[int, ...], Optional[str]]:
    """PP-CUT DRAFT-ZELLE (YaRN x2 27B, 29.09.): the per-token bytes group P's draft KV adds to the pool
    cell of the stage that hosts it -- the LAST P stage (pool_configurator.draft_kv_pool_on_this_rank),
    whenever P carries the draft runner (the resolved ``draft_kv_on_p``; ``cold`` builds its pool too).
    From the draft's OWN geometry: DFLASH = the external draft's layers x 2 x kv heads x head_dim,
    NEXTN = one layer of the target's geometry; fp8 KV (the solver's own kv dtype). Boot
    dkr27browauthorityyarn2bar1fs09292006: PP2 cell 18432 = 4 x 2048 + 5 x 2 x 8 x 128.
    ``((), None)`` without a draft on P: the pool model is byte-identical."""
    resolved = _SPEC_FORM.get("draft_kv_on_p")
    on_p = bool(resolved) if resolved is not None else bool(envs.FLLIPER_PDFLIP_DRAFT_ON_P.get())
    if not on_p or int(n_stages) <= 0:
        return (), None
    if spec_form_is_dflash():
        dpath = str(getattr(ns, "dflash_draft_path", "") or DFLASH_DRAFT_PATH_DEFAULT)
        dcfg = _model_config(dpath)
        dcfg = dcfg.get("text_config") or dcfg
        layers = int(dcfg["num_hidden_layers"])
        src = f"DFLASH {os.path.basename(dpath.rstrip('/'))}"
    else:
        dcfg, layers, src = target_text_cfg, 1, "NEXTN (one target-geometry layer)"
    heads = int(dcfg.get("num_key_value_heads") or dcfg["num_attention_heads"])
    head_dim = int(dcfg.get("head_dim") or int(dcfg["hidden_size"]) // int(dcfg["num_attention_heads"]))
    v_head_dim = int(dcfg.get("v_head_dim") or head_dim)
    cell = int(layers) * heads * (head_dim + v_head_dim) * int(kv_dtype_bytes)
    by_stage = tuple([0] * (int(n_stages) - 1) + [cell])
    return by_stage, (f"PP-CUT DRAFT-ZELLE: {src}: {layers} layer(s) x {heads} kv heads x "
                      f"({head_dim}+{v_head_dim}) x {kv_dtype_bytes} B = {cell} B per token on the last P stage "
                      f"(its pool cell = attention layers x cell + this; pool_configurator cell_size_with_draft)")

def p_group_has_draft_producer() -> bool:
    """THE ONE detection of "does group P produce draft pages", read off P's
    FORM -- never off whether D has a draft model (D always has one).

    UNIFY S6: one def for both lines (the probe merge carried two bodies).
    True only when group P carries the draft-KV producer -- the H25-resolved
    ``draft_kv_on_p`` (``p_draft != none``: FLLIPER_PDFLIP_DRAFT_ON_P /
    --draft-kv-on-p / the profile, resolved ONCE by pdflip/form.py) and HiCache
    enabled -- AND that producer computes: under --spec-form DFLASH that is
    --dflash-produce-on-p on (``p_draft=compute``; ``cold`` builds the producer
    and computes nothing, 27B 2026-09-24); the NEXTN/MTP producer always
    computes. Before main resolved it (desk, tests) the H25 env is read, whose
    default is 0 = no producer (NF Nutzer-Order 24.09. 08:25Z)."""
    resolved = _SPEC_FORM.get("draft_kv_on_p")
    on_p = bool(resolved) if resolved is not None else bool(envs.FLLIPER_PDFLIP_DRAFT_ON_P.get())
    if not on_p:
        return False
    if spec_form_is_dflash():
        return dflash_produce_on_p()
    return True

def p_produces_draft_pages() -> bool:
    """Does group P WRITE draft pages this boot (given it carries the
    producer flags)? False only under --spec-form DFLASH with
    --dflash-produce-on-p off (p_draft=cold, the 27B 2026-09-24 form)."""
    return not (spec_form_is_dflash() and not dflash_produce_on_p())

def d_verify_window() -> int:
    """Group D's widest verify window: the draft block under DFLASH; under
    NEXTN the ``--speculative-num-draft-tokens`` D actually runs.

    NF line (H64): the Next-Flash arm ships D's MTP depth through --extra-d
    (``--speculative-num-steps 3 ... --speculative-num-draft-tokens 4``), which
    argv_d appends AFTER the constants' spec family (SPEC_NUM_DRAFT_TOKENS 3),
    and argparse takes the last value -- so the window is --extra-d's value
    when it names one (installed by :func:`apply_spec_form`), else the
    constant."""
    if spec_form_is_dflash():
        return int(_SPEC_FORM["block"])
    raw = _SPEC_FORM.get("d_extra_draft_tokens")
    try:
        extra = int(str(raw)) if raw is not None else 0
    except ValueError:
        extra = 0
    return extra if extra > 0 else int(SPEC_NUM_DRAFT_TOKENS)

def d_pool_peak_record(profile: Optional[str] = None
                       ) -> Tuple[Optional[List[int]], Optional[List[int]], str]:
    """``D_POOL_PEAK_NONRES_ROWS`` (the measured maximum of the MOE-POOL-DEMAND
    ``max_nonres_per_step`` per D rank) with ``D_POOL_PEAK_NONRES_SPAN_ROWS``
    (E - R of the form it was measured on), or ``(None, None, "")`` -- no
    floor, the stage solve stays byte-identical (27B has no such record)."""
    try:
        peak = [int(v) for v in _pconst("D_POOL_PEAK_NONRES_ROWS", profile)]
        span = [int(v) for v in _pconst("D_POOL_PEAK_NONRES_SPAN_ROWS", profile)]
    except (KeyError, TypeError, ValueError):
        return None, None, ""
    if not peak or len(peak) != len(span):
        return None, None, ""
    return peak, span, _pconst_boots("D_POOL_PEAK_NONRES_ROWS", profile)

def p_activation_record(profile: Optional[str] = None) -> Tuple[Optional[Tuple[float, ...]], str]:
    """``(P_ACTIVATION_MIB per P stage, 'boot ...')`` of ``profile``, or
    ``(None, "")`` where the profile never measured it (27B today: its
    support, env and pool model stay byte-identical)."""
    try:
        vals = tuple(float(v) for v in _pconst(P_ACTIVATION_RECORD, profile))
    except KeyError:
        return None, ""
    return vals, _pconst_boots(P_ACTIVATION_RECORD, profile)

def _p_chunk_nf_limits(nf_args: Dict[str, object], stages: int):
    from flliper.srt.pdflip import p_chunk_nf as _nf
    from flliper.srt.pdflip import p_chunk_policy as _pcp

    try:
        support = int(max(P_PREFILL_TRANSIENT_SUPPORT.chunks))
    except (AttributeError, TypeError, ValueError):
        support = 0
    _mg = nf_args.get("min_gain")
    return _nf.nf_limits(
        ceiling=P_CHUNKED_PREFILL_TOKENS,
        max_tokens=int(nf_args.get("max") or 0),
        min_tokens=int(nf_args.get("min") or 0),
        fixed_tokens=int(nf_args.get("fixed") or 0),
        raster=int(nf_args.get("raster") or _nf.NF_RASTER_TOKENS),
        transient_support_max=support,
        min_gain=float(_pcp.DEFAULT_MIN_GAIN if _mg is None else _mg),
        dynamic_min_tokens=int(nf_args.get("dynamic_min", -1)),
        stages=stages,
    )

def host_preflight(log: Log, tag: str, dry: bool) -> None:
    """The host preflight: the script, then the launcher's own 40 GiB MemAvailable floor.

    HOSTWARN2 (13.10.2026, user rule 06.10. "KEINE Verweigerung wegen Host-RAM ... man laesst
    es darauf ankommen, ob es OOM geht"): BOTH host-RAM floors WARN now (``HOST-RAM-WARN
    HOST-MEM ...``, full numbers, the boot goes on). What still refuses: a disk verdict of the
    script (``HOST-DISK``, forceable) and a script that failed without any verdict line
    (cannot be told from a broken tool; ``PdFlipLaunchRefused``). VRAM and card refusals are not here.
    """
    if dry:
        log("DRY-RUN: host_ledger_preflight.sh not run")
        return
    pre_log = f"{GPU_ARB}/preflight_weg2_{tag}.log"
    rc = subprocess.run([HOST_PREFLIGHT, pre_log]).returncode
    warns = 0
    if rc != 0:
        try:
            with open(pre_log, encoding="utf-8", errors="replace") as fh:
                ram_lines, other_lines = split_preflight_refusals(fh.read())
        except OSError:
            ram_lines, other_lines = [], []
        if other_lines:
            refusals.refuse_value("HOST-DISK", "BOOT REFUSED by the host-ledger preflight (disk floor, NOT host RAM) -- "
                                  + " | ".join(other_lines) + f" -- see {pre_log}", PdFlipLaunchRefused, log)
        if not ram_lines and not other_lines:
            raise PdFlipLaunchRefused(
                f"host-ledger preflight {HOST_PREFLIGHT} exited {rc} without a REFUSED verdict in {pre_log}: "
                "a broken or missing script is not a host-RAM verdict and does not boot silently")
        if ram_lines:
            warns += 1
            log(f"{host_ledger.HOST_WARN_MARKER} HOST-MEM PreflightScriptFloor -- {' | '.join(ram_lines)} "
                f"-- WARNING ONLY: the boot goes on (user rule 06.10.: no refusal on host RAM; an OOM kill is "
                f"the accepted risk); the script's own log: {pre_log}")
    mi = host_ledger.read_meminfo()
    avail_gib = mi["MemAvailable"] / host_ledger.GIB
    if avail_gib < 40:
        warns += 1
        top = subprocess.run(["ps", "-eo", "rss,pid,comm", "--sort=-rss"], capture_output=True, text=True).stdout.splitlines()[:8]
        log(f"{host_ledger.HOST_WARN_MARKER} HOST-MEM Preflight40GiB -- free -g available {avail_gib:.1f} GiB < 40 GiB; "
            "top RSS holders (NOT killed):\n" + "\n".join(top)
            + " -- WARNING ONLY: the boot goes on (user rule 06.10.: no refusal on host RAM; an OOM kill is the accepted risk)")
    oom = open("/sys/fs/cgroup/memory.events").read()
    log(f"host preflight {'PASS' if not warns else 'CONTINUED (host-RAM warnings=%d)' % warns}: "
        f"MemAvailable {avail_gib:.1f} GiB; cgroup memory.events baseline: {' '.join(oom.split())}")
    # SWAP READINESS (26.09.): the ledger's currencies assume a swapless cgroup;
    # name the swap state once per boot and WARN when pages already moved out.
    swap_line, _swap_warn = host_ledger.swap_state_line()
    log(swap_line)

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
            raise PdFlipLaunchRefused(
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

def plan_store(
    tag: str,
    p_pool_tokens: float,
    attn_counts: Sequence[int],
    kv_mib_per_token_per_attn_layer: float,
    sidecar_factor: float = STORE_SIDECAR_FACTOR,
    min_free_gib: float = STORE_DISK_MIN_FREE_GIB,
    root: Optional[str] = None,
    store_max_gb: float = 0.0,
    directory_name: Optional[str] = None,
    dry: Optional[bool] = None,
) -> StoreDiskPlan:
    """Size the store from THE P POOL, and check the DISK can fund it.

    ``max_size = P pool bytes x sidecar_factor``.  The P pool bytes come from
    the SAME source the KV pool sizing itself uses -- the ``PP-POOL-JOIN``
    line's own ``pool_tokens`` and ``cell_bytes``, i.e. :class:`PCutFacts`
    straight out of :func:`solve_p_cut` -- never a hand-typed 32 KiB.  That
    matters for the same reason #1286 F1 gave for the join key: a literal here
    would keep printing a plausible number on any config with a different KV
    dtype or head geometry while the pool itself had moved.

    WHY ``>= P pool`` IS THE RIGHT FLOOR (#1236): under the Weg-2 phase law
    group P prefills the WHOLE backlog into its pool and the store is the
    carrier every one of those prefixes travels through to reach group D.  A
    store smaller than the pool cannot hold what one prefill leg produces, so
    the writes at the end of the leg are refused -- which is exactly the shape
    boot weg2sb5g logged 1,324 times.

    ``min_free_space`` is checked against the DISK the directory lives on
    (``statvfs`` of ``root``), which is what
    ``LRUFileEvictor._enforce_free_space_locked`` will itself read at runtime;
    there is no second reading and no second bookkeeping.  Raises W57 when the
    filesystem cannot fund ``max_size + min_free`` -- fail fast at preflight
    rather than let the evictor latch its write stop mid-flip.
    """
    cell_bytes = int(round(sum(int(a) for a in attn_counts)
                           * float(kv_mib_per_token_per_attn_layer) * MIB))
    pool_bytes = int(round(float(p_pool_tokens) * cell_bytes))
    # USER-SET ABSOLUTE SIZE takes precedence over the derived one.
    # User order 2026-09-10, verbatim: "der l3 auf der platte hatte ich mal auf
    # 150gb groesse festgelegt, nicht 28gb". The derived figure
    # (`pool_bytes x sidecar_factor`) answers "how big must the store be to hold
    # one prefill leg plus its sidecars" -- a FLOOR, not a cap, and it cannot
    # express "the operator gave this store 150 GB of disk". In GB (10^9), the
    # unit the user used and the unit HiCacheFile's max_size already speaks.
    #
    # The #1236 floor still binds: a store below the P pool cannot hold what one
    # prefill leg produces (boot weg2sb5g logged that refusal 1,324 times), so a
    # user value under it is REFUSED BY NAME rather than silently clamped -- an
    # operator asking for less than the law allows should be told, not overridden.
    if float(store_max_gb) > 0:
        max_size = int(round(float(store_max_gb) * 1_000_000_000))
        if max_size < pool_bytes:
            raise PdFlipStoreDiskRefused(
                f"W57 PdFlipStoreDiskRefused: --store-max-gb {store_max_gb} GB = "
                f"{max_size / host_ledger.GIB:.2f} GiB is BELOW this boot's P pool "
                f"({int(p_pool_tokens)} tokens x {cell_bytes} B/token = "
                f"{pool_bytes / host_ledger.GIB:.2f} GiB). #1236's law is store >= P pool: "
                f"the store is the carrier every prefix travels through from P to D, so one "
                f"below the pool refuses the writes at the end of a prefill leg. Raise "
                f"--store-max-gb to at least "
                f"{int(math.ceil(pool_bytes / 1_000_000_000))} GB, or drop the flag to use "
                f"the derived size."
            )
    else:
        max_size = int(math.ceil(pool_bytes * float(sidecar_factor)))
    min_free = int(round(float(min_free_gib) * host_ledger.GIB))
    if root is None:
        # W57 (28.09.): the rig default is a path on the ZFS pool; a launcher
        # that was not told where its store lives measures THAT disk, not the
        # store's (the NF store is a bind of XFS [/l3/nf] into the container).
        if not STORE_ROOT_TOLD:
            mnt, typ, src = fs_of(STORE_ROOT)
            raise PdFlipStoreDiskRefused(
                f"W57 PdFlipStoreDiskRefused: no store root told -- FLLIPER_PDFLIP_STORE_ROOT "
                f"is unset, and the launcher's default {STORE_ROOT} lives on fs={typ or '?'} "
                f"mount={mnt or '?'} ({src or '?'}), which need not be the disk the store "
                f"is on; nothing was measured. Set FLLIPER_PDFLIP_STORE_ROOT to the store's "
                f"own root (the container: /var/lib/htsglang/hicache-weg2 with the store "
                f"volume mounted)."
            )
        root = STORE_ROOT
    os.makedirs(root, exist_ok=True)
    directory = f"{root}/{directory_name or tag}"
    # W57: measure where the store IS -- its own directory when it exists (a
    # store directory may itself be a mount), else the root it will be made in
    probe = directory if os.path.isdir(directory) else root
    st = os.statvfs(probe)
    free = st.f_bavail * st.f_frsize
    total = st.f_blocks * st.f_frsize
    fs_mount, fs_type, _fs_src = fs_of(probe)
    # L3P: a persistent directory's own bytes are inside its cap -- the disk
    # has to fund only what the store may still GROW by (``growth_bytes``).
    reused = 0
    if directory_name and os.path.isdir(directory):
        reused = _tree_bytes(directory)[0]
    plan = StoreDiskPlan(
        directory=directory,
        p_pool_tokens=int(p_pool_tokens),
        cell_bytes=cell_bytes,
        p_pool_bytes=pool_bytes,
        sidecar_factor=float(sidecar_factor),
        max_size_bytes=max_size,
        min_free_bytes=min_free,
        fs_free_bytes=free,
        fs_total_bytes=total,
        reused_bytes=reused,
        fs_path=probe,
        fs_type=fs_type,
        fs_mount=fs_mount,
        fs_refused=(dry is not None
                    and fs_type in set(envs.FLLIPER_PDFLIP_STORE_REFUSE_FS.get() or ())),
    )
    if plan.fs_refused:
        # W57 (user 28.09.: "L3 gehoert auf XFS, nie ZFS"): the container
        # layer or the ZFS pool is not the told store disk -- its free space
        # says nothing about the store. A dry run warns (the caller's line);
        # ``dry=None`` (not a launch: a sizing question) never gets here.
        if not dry:
            raise PdFlipStoreDiskRefused(
                f"W57 PdFlipStoreDiskRefused: store on {fs_type}, not the told disk -- "
                f"{plan.disk_line('REFUSED')}. FLLIPER_PDFLIP_STORE_ROOT={root} resolves to "
                f"fs={fs_type} (FLLIPER_PDFLIP_STORE_REFUSE_FS="
                f"{','.join(envs.FLLIPER_PDFLIP_STORE_REFUSE_FS.get() or ())}); mount the store "
                f"volume (XFS) at the store root, or empty the variable to allow it."
            )
        return plan
    if plan.needed_bytes > free:
        raise PdFlipStoreDiskRefused(
            f"W57 PdFlipStoreDiskRefused: the filesystem hosting {root} cannot fund this "
            f"boot's store. {plan.disk_line('REFUSED')}. "
            f"NEEDED {plan.needed_bytes / host_ledger.GIB:.2f} GiB = growth "
            f"{plan.growth_bytes / host_ledger.GIB:.2f} GiB of max_size "
            f"{max_size / host_ledger.GIB:.2f} GiB (P pool "
            f"{plan.p_pool_tokens} tokens x {cell_bytes} B/token = "
            f"{pool_bytes / host_ledger.GIB:.2f} GiB x sidecar factor "
            f"{sidecar_factor:.2f}) + min_free {min_free / host_ledger.GIB:.2f} GiB, "
            f"AVAILABLE {free / host_ledger.GIB:.2f} GiB of {total / host_ledger.GIB:.2f} "
            f"GiB. This is the '#1236 Store >= P pool' law meeting a disk that cannot "
            f"hold it -- refusing at preflight rather than letting HiCacheFile latch its "
            f"write stop mid-flip the way boot weg2sb5g did 1,324 times. The levers, "
            f"named: free space on {root}; --store-sidecar-factor (the sidecar headroom, "
            f"default {STORE_SIDECAR_FACTOR:.2f} from {STORE_CENSUS_PROVENANCE}); "
            f"--store-disk-min-free-gib (the floor protecting this box's own root "
            f"filesystem, default {STORE_DISK_MIN_FREE_GIB:.0f} GiB); or a smaller P "
            f"pool via --pp-solve-objective. NOT a lever: host RAM -- this store is "
            f"disk, and the host ledger no longer prices it."
        )
    return plan

def l3_revoke_window(directory: str, from_unix: float, to_unix: float, reason: str) -> dict:
    """L3-REVOKE: add a revoked write window to the store's record (atomic
    replace). Pages are MOVED at the next attach, not here -- a store in use
    by a running boot is never walked from outside."""
    if not float(from_unix) <= float(to_unix):
        raise ValueError(f"from {from_unix} after to {to_unix}")
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)
    wins = l3_revoked_windows(directory)
    rec = {"windows": [{"from_unix": a, "to_unix": b, "reason": r} for a, b, r in wins]}
    rec["windows"].append({
        "from_unix": float(from_unix), "to_unix": float(to_unix), "reason": str(reason),
        "from_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(from_unix))),
        "to_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(to_unix))),
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    path = _l3_revoked_path(directory)
    tmp = path + ".w"
    with open(tmp, "w") as f:
        json.dump(rec, f, sort_keys=True, indent=1)
    os.replace(tmp, path)
    return rec

def _l3_attach_from_index(log: Log, directory: str, dry: bool, epoch: float,
                          known: Optional[Tuple[str, ...]],
                          revoked: List[Tuple[float, float, str]]
                          ) -> Optional[Tuple[int, int, int]]:
    """``l3_persist_attach`` without the walk: the pages, their bytes and
    mtimes come from the persistent index (store_journal.load_index, the load
    the ranks make). Same removals, derived instead of found:

    - staging files: a write that died leaves its ``<final>.tmp.<uuid>`` only
      after its ``R`` line with no ``C``/``A`` -- the open intents; their
      staging names are globbed, nothing else is looked at;
    - ORPHAN pages: indexed stems in no group's suffix record;
    - revoked pages: indexed mtime inside a revoked window.

    Every page removed or moved gets an ``E`` line in a launcher journal filed
    under the PREVIOUS epoch, so the ranks' load (which replays every earlier
    boot's journal) and the D owner's compaction see it -- the index never
    keeps an entry the launcher unlinked. ``None`` = the index is unusable:
    the caller walks, as the ranks will.
    """
    import glob

    from flliper.srt.mem_cache.storage.file import store_journal as _sj

    t0 = time.monotonic()

    def _stat(stem):
        try:
            st = os.lstat(_sj.page_path(directory, stem))
        except (OSError, ValueError):  # ValueError: a stem os.* cannot name (NUL byte)
            return None
        return (st.st_mtime, st.st_blocks * 512)

    intents: List[str] = []
    items, why = _sj.load_index(directory, stat_size=_stat, open_intents=intents)
    if items is None:
        log(f"L3-PERSIST index unusable ({why}) -- ONE walk of dir={directory}")
        return None
    jn = None  # opened at the first E line: an attach that removes nothing files no journal
    now = time.time()

    def _gone(size, stem):
        nonlocal jn
        if jn is None:
            jn = _sj.JournalWriter(directory, "launcher", epoch=int(epoch) - 1)
        jn.write("E", now, size, stem)
    files = nbytes = removed = orphans = orphan_bytes = 0
    rev_n = rev_bytes = rev_failed = 0
    # 30.09. (27B dual1f 9pb9ex, launcher 101 % CPU in glob for minutes): one glob per open intent
    # re-listed the intent's whole shard (~3200 names) -- 1.4M journal lines x 256 shards. List every
    # directory ONCE and look the staging names up there; same files removed, O(entries) not O(intents x shard).
    _listing: Dict[str, Dict[str, List[str]]] = {}

    def _staging(final_path: str) -> List[str]:
        d, base = os.path.split(final_path)
        by_final = _listing.get(d)
        if by_final is None:
            by_final = {}
            try:
                names = os.listdir(d)
            except OSError:
                names = []
            for n in names:
                i = n.find(".tmp.")
                if i > 0:
                    by_final.setdefault(n[:i], []).append(os.path.join(d, n))
            _listing[d] = by_final
        return by_final.get(base, [])

    def _shard_final(stem: str) -> str:
        # the backend's shard rule WITHOUT page_path's existence probes: an open intent's page
        # usually does not exist (the write died), and page_path then globs ALL shards per stem
        pre = stem[:2]
        shard = pre if len(pre) == 2 and all(c in "0123456789abcdef" for c in pre) else "zz"
        return os.path.join(directory, shard, f"{stem}.bin")

    for stem in intents:
        final = _shard_final(stem)
        for p in {final, os.path.join(directory, f"{stem}.bin")}:
            for tmp in _staging(p):
                if not dry:
                    try:
                        os.unlink(tmp)
                        removed += 1
                    except OSError:
                        pass
                    if jn is None:  # the shard changed: a journal write after it
                        jn = _sj.JournalWriter(directory, "launcher", epoch=int(epoch) - 1)
    dropped: List[str] = []
    for stem, (mtime, size) in items.items():
        if revoked and any(a <= mtime <= b for a, b, _r in revoked):
            path = _sj.page_path(directory, stem)
            if _l3_move_revoked(directory, os.path.dirname(path), os.path.basename(path), dry):
                rev_n += 1
                rev_bytes += size
                if not dry:
                    _gone(size, stem)
                    dropped.append(stem)
            else:
                rev_failed += 1
            continue
        if known is not None and not stem.endswith(known):
            orphans += 1
            orphan_bytes += size
            if not dry:
                try:
                    os.unlink(_sj.page_path(directory, stem))
                    removed += 1
                except FileNotFoundError:
                    pass
                except OSError:
                    continue  # still on disk: its entry stays
                _gone(size, stem)
                dropped.append(stem)
            continue
        files += 1
        nbytes += size
    if jn is not None:
        jn.sync()
    snap_note = ""
    if not dry:
        # the launcher is the boot's snapshot writer: the loaded map minus
        # what it just removed, earlier boots' journals folded in -- before
        # any rank starts, so every rank loads ONE snapshot and replays nothing
        for stem in dropped:
            items.pop(stem, None)
        try:
            with _sj.snapshot_lock(directory):
                _n, _b, _jg = _sj.persist(
                    directory, ((m, st, b) for st, (m, b) in items.items()), len(items),
                    int(epoch))
            snap_note = f" snapshot rewritten ({_n} files, {_jg} journal(s) folded in)"
        except (OSError, ValueError) as exc:
            snap_note = (f" snapshot NOT rewritten ({type(exc).__name__}: {exc}; the E lines "
                         f"above carry the removals)")
    log(
        f"L3-PERSIST {'reuse' if files else 'fresh'} dir={directory} files={files} bytes={nbytes} "
        f"({nbytes / host_ledger.GIB:.2f} GiB) orphans={orphans} ({orphan_bytes / host_ledger.GIB:.2f} GiB, "
        + ("no suffix record yet: counted, kept" if known is None else
           f"suffix in no group's record: {'would be ' if dry else ''}removed")
        + f") staging+orphans_removed={removed} (staging of {len(intents)} open intent(s))"
        + (f" revoked={rev_n} ({rev_bytes / host_ledger.GIB:.2f} GiB, "
           f"{'would be ' if dry else ''}moved to {directory.rstrip('/') + L3_REVOKED_SUFFIX}"
           f"{f', {rev_failed} NOT moved (kept out of the count, still on disk)' if rev_failed else ''})"
           if revoked else "")
        + f" index_s={time.monotonic() - t0:.2f} source=index ({why}), no walk{snap_note}"
        + f" epoch={epoch:.3f}"
        + (" (DRY-RUN: nothing removed)" if dry else "")
        + " -- the ranks seed the L3 index and the evictor ledger from it at attach"
    )
    return files, nbytes, removed

def l3_persist_disk_sum(log: Log, root: str, directory: str, max_size_bytes: int,
                        min_free_bytes: int, own_bytes: int, dry: bool) -> int:
    """L3P, user 2026-09-27 "L3, jeder 150gb": every model's store has its own
    cap, and they share ONE disk. Records this store's cap and size
    (``L3_CAP.json``) and checks that what ALL stores may still grow by fits
    the disk: sum over stores of (cap - bytes at its last attach) + min_free
    <= free now. The other stores' bytes are as of THEIR last boot (named, not
    re-walked). Over budget is a named WARNING with the numbers, never a
    silent cap: a sibling store may never grow to its cap. Returns the
    shortfall in bytes (0 = fits)."""
    rec = {"max_size_bytes": int(max_size_bytes), "bytes_at_attach": int(own_bytes),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not dry:
        try:
            tmp = os.path.join(directory, L3_CAP_FILE + ".w")
            with open(tmp, "w") as f:
                json.dump(rec, f)
            os.replace(tmp, os.path.join(directory, L3_CAP_FILE))
        except OSError as exc:
            log(f"L3-PERSIST cap record not written ({type(exc).__name__}: {exc})")
    growth = max(0, int(max_size_bytes) - int(own_bytes))
    parts = [f"own={os.path.basename(directory)} grow={growth / 1e9:.1f} GB"]
    try:
        names = sorted(os.listdir(root))
    except OSError:
        names = []
    for n in names:
        d = os.path.join(root, n)
        if not n.startswith(L3_PERSIST_PREFIX) or d == directory:
            continue
        if n.endswith(L3_REVOKED_SUFFIX):
            parts.append(f"{n}=revoked pages held aside (no store; the user deletes)")
            continue
        try:
            with open(os.path.join(d, L3_CAP_FILE)) as f:
                o = json.load(f)
            g = max(0, int(o["max_size_bytes"]) - int(o["bytes_at_attach"]))
        except (OSError, ValueError, KeyError, TypeError):
            parts.append(f"{n}=no cap record")
            continue
        growth += g
        parts.append(f"{n} grow={g / 1e9:.1f} GB (as of {o.get('utc', '?')})")
    try:
        st = os.statvfs(root)
        free = st.f_bavail * st.f_frsize
    except OSError:
        log("L3-PERSIST DISK-SUM unreadable (statvfs failed) -- NOT checked")
        return 0
    need = growth + int(min_free_bytes)
    short = max(0, need - free)
    log(
        f"L3-PERSIST DISK-SUM {'WARN' if short else 'ok'}: all L3 stores may still grow "
        f"{growth / 1e9:.1f} GB + min_free {min_free_bytes / 1e9:.1f} GB = {need / 1e9:.1f} GB "
        f"against {free / 1e9:.1f} GB free on {root}"
        + (f" -- SHORT by {short / 1e9:.1f} GB: the disk cannot hold every store at its cap; "
           f"lower --store-max-gb or free space (nothing is capped silently)" if short else "")
        + f" [{'; '.join(parts)}]"
    )
    return short

def arc_preflight_line(
    arc: Dict[str, Optional[int]],
    plan: StoreDiskPlan,
    margin_gib: Optional[float],
) -> str:
    """ONE line pricing the ZFS ARC beside the reap margin, and W58 if it must.

    THE ASSUMPTION, stated rather than assumed: the ARC is the cache ZFS keeps
    for this dataset, it is RECLAIMABLE under memory pressure (the kernel
    shrinks it before the reaper kills), and it is HOST kernel memory -- it is
    not charged to this container's cgroup at all.

    WHAT WAS CHECKED ON THIS BOX rather than guessed (2026-09-09), because the
    briefing asked whether ``MemAvailable`` includes the ARC: ``/proc/meminfo``
    here is ``fuse.lxcfs``, and its numbers are this CGROUP's.  Measured at one
    reading: meminfo ``Cached`` 31,287,292 kB against cgroup ``memory.stat
    file`` 32,038,137,856 B -- the same 29.84 GiB -- and meminfo
    ``SReclaimable`` 792,860 kB against cgroup ``slab_reclaimable``
    811,390,688 B, the same 0.76 GiB, with ``MemAvailable`` equal to
    ``MemFree + Cached + SReclaimable`` to within 0.15 GiB.  The ARC's own
    4.92 GiB (``size`` 5,288,088,320 B) appears in NEITHER.  So: the ARC is
    invisible to the reap ledger, and invisible in the SAFE direction -- it
    cannot inflate ``memory.current``, which is the quantity the reaper acts
    on (#1233 fix 5).  What it can do is compete for host RAM outside our
    cgroup, which is why the cap is checked at all.

    THE REFUSAL is narrow ON PURPOSE: only an UNBOUNDED ARC (``zfs_arc_max``
    0 or unreadable) combined with a reap margin thinner than this store's own
    byte budget.  A capped ARC -- 5 GiB here, already sitting at 4.92 GiB, so
    at cap and evicting -- can absorb nothing further and is not a hazard.
    """
    def _g(v: Optional[int]) -> str:
        return "unreadable" if v is None else f"{v / host_ledger.GIB:.2f} GiB"

    arc_max = arc.get("arc_max")
    unbounded = arc_max is None or arc_max == 0
    # THE STORE'S OWN BYTES, not `needed_bytes`: `min_free` is free space the
    # store never writes into, so counting it here would over-state both the
    # RAM this move freed and the most an uncapped ARC could absorb from us.
    store_budget_gib = plan.max_size_bytes / host_ledger.GIB
    head = (
        f"PDFLIP-STORE ARC: zfs_arc_max={_g(arc_max)}"
        + ("" if arc_max else " (0/unreadable = NO EXPLICIT CAP)")
        + f" arcstats c_max={_g(arc.get('c_max'))} size={_g(arc.get('size'))} "
        f"vs this boot's store budget {store_budget_gib:.2f} GiB and reap margin "
        + ("unreadable" if margin_gib is None else f"{margin_gib:.2f} GiB")
        + ". The ARC is HOST kernel memory and reclaimable under pressure; measured on "
        "this box 2026-09-09, meminfo here is fuse.lxcfs and its Cached/SReclaimable "
        "are this CGROUP's own file/slab_reclaimable to the byte, and the ARC is in "
        "NEITHER -- so it cannot move memory.current, the quantity the reaper acts on."
    )
    if unbounded and margin_gib is not None and margin_gib < store_budget_gib:
        raise PdFlipStoreArcRefused(
            f"W58 PdFlipStoreArcRefused: the ZFS ARC has NO explicit cap "
            f"(zfs_arc_max={_g(arc_max)}) and the reap margin {margin_gib:.2f} GiB is "
            f"thinner than this store's own byte budget {store_budget_gib:.2f} GiB. An "
            f"uncapped ARC can grow to hold what this store writes, and this boot has "
            f"not left room for it. Set /sys/module/zfs/parameters/zfs_arc_max to a "
            f"finite value, or widen the margin. NOT refused when the ARC is capped: a "
            f"capped ARC evicts instead of growing. {head}"
        )
    return head + (
        "  VERDICT: capped, no refusal."
        if not unbounded
        else "  VERDICT: uncapped, but the reap margin covers this store's budget -- "
             "stated, not refused."
    )

def p_stage_fixed_for_checkpoint(
    ns, stage_fixed: Sequence[float], terms, *, text_cfg: Optional[Dict] = None,
    model: Optional[str] = None,
) -> Tuple[Tuple[float, ...], str]:
    """PPCUT-INT4AR 24.10.: group P's per-stage fixed posts for THIS checkpoint.

    Three branches, each with its own line:

    * ``--pp-cut-stage-fixed-mib`` given on the command line
      (``ns.pp_cut_stage_fixed_given``, set by :func:`apply_profile_arg_defaults`)
      -> the operator's measurement of this checkpoint, unchanged;
    * the profile registers no reference checkpoint in
      :data:`P_PP_STAGE_FIXED_REF_NONLAYER_MIB` -> the record, unchanged;
    * otherwise the record is moved onto this checkpoint's RESIDENT non-layer
      bytes by ``pp_cut_launch.stage_fixed_checkpoint_delta`` (first stage +=
      embedding delta, last stage += lm_head delta, one-sided).

    RESIDENT, not header bytes, where the two differ: a GGUF checkpoint loaded
    with ``FLLIPER_GGUF_DENSE_VOCAB=1`` dequantizes token_embd/output to BF16,
    so the header's quant size would under-book the first stage by the
    dequantization growth (review #222 SOLL-2).  There the dense BF16 size
    ``vocab_size x hidden_size x 2`` from the config is used for both.
    """
    fixed = tuple(float(v) for v in stage_fixed)
    if getattr(ns, "pp_cut_stage_fixed_given", False):
        return fixed, ("PP-CUT STAGE-FIXED CHECKPOINT: --pp-cut-stage-fixed-mib given on the command "
                       "line -> taken as this checkpoint's measurement, not moved")
    profile = str(getattr(ns, "profile", None) or PROFILE_QWEN27B)
    ref = P_PP_STAGE_FIXED_REF_NONLAYER_MIB.get(profile)
    if ref is None:
        return fixed, ("PP-CUT STAGE-FIXED CHECKPOINT: profile %r registers no reference checkpoint for "
                       "its P_PP_STAGE_FIXED_MIB record (P_PP_STAGE_FIXED_REF_NONLAYER_MIB) -> record "
                       "unchanged" % (profile,))
    from flliper.srt.planner import pp_cut as _pp_cut
    from flliper.srt.planner import pp_cut_launch as _cut

    emb_mib = float(terms.embedding_weight_bytes) / _pp_cut.MIB
    head_mib = float(terms.lm_head_weight_bytes) / _pp_cut.MIB
    note = ""
    if _p_gguf_dense_vocab(ns, model):
        cfg = text_cfg or {}
        vocab, hidden = int(cfg.get("vocab_size") or 0), int(cfg.get("hidden_size") or 0)
        if vocab > 0 and hidden > 0:
            dense = vocab * hidden * 2 / _pp_cut.MIB
            note = ("; GGUF %s=1: token_embd/output dequantized to BF16 at load -> resident "
                    "%.1f MiB each (header %.1f / %.1f)" % (GGUF_DENSE_VOCAB_ENV, dense, emb_mib, head_mib))
            emb_mib, head_mib = max(emb_mib, dense), max(head_mib, dense)
        else:
            note = ("; GGUF %s=1 but the config states no vocab_size/hidden_size -- header bytes "
                    "used, the first stage may be UNDER-booked by the dequantization growth"
                    % (GGUF_DENSE_VOCAB_ENV,))
    out, line = _cut.stage_fixed_checkpoint_delta(
        fixed,
        ref_label=ref[0],
        ref_embedding_mib=float(ref[1]),
        ref_lm_head_mib=float(ref[2]),
        embedding_mib=emb_mib,
        lm_head_mib=head_mib,
    )
    return out, line + note

def load_xchg_census_for_cards(
    census_path: str,
    cards: Sequence[Card],
    log: Optional[Log] = None,
    census_map: Optional[str] = None,
) -> "xchg_residency.XchgCensus":
    """AP4 1526: ``load_census`` + one row for EVERY live card, or a named refusal.

    The census is a measurement of an inventory; the live inventory is whatever
    this boot found.  A live card that is a key of the census keeps its row
    (the reference rig and any inventory the census was measured on: the SAME
    census object, nothing printed).  A card that is not gets a BORROWED row
    (``xchg_residency.resolve_census``) -- and the borrow is the forcebar value
    refusal ``HW-BORROWED`` (``refusals.refuse_value``): without ``--force`` a
    named W71, with ``--force`` the boot starts and prints
    ``FORCED-PAST HW-BORROWED``.  A row the operator named with
    ``--pdflip-xchg-census-map`` is an assertion, not a borrow: one
    ``PDFLIP-XCHG-CENSUS-MAP`` line and no refusal.  The peak is still graded
    against the card's LIVE NVML total by ``xchg_residency.solve``."""
    census = xchg_residency.load_census(census_path)   # raises W71 by name
    # AP2b 1006: on a foreign inventory the launcher derives the donor of every
    # missing row ITSELF (no --pdflip-xchg-census-map needed): a census row without
    # a card_class field is labelled from the reference rig's card registry
    # (UUID -> board), a live card borrows the row of its own class, else of its
    # arch twin (AP1 W19 rule).  Still a named HW-BORROWED borrow (forcebar).
    census, borrows, notes = xchg_residency.resolve_census(
        census, cards, census_map=_XCHG_CENSUS_MAP[0] if census_map is None else census_map,
        known_classes=xchg_residency.reference_census_classes())
    if log is not None:
        for ln in notes:
            log(ln)
    seen = {(d["code"], d["text"]) for d in refusals.forced_list()}
    for b in borrows:
        text = "W71 PdFlipXchgResidencyUnarmable: " + b.text(census.path)
        if ("HW-BORROWED", text) in seen:
            continue        # the second reader of the same census: already past
        refusals.refuse_value(
            "HW-BORROWED", text, xchg_residency.PdFlipXchgResidencyUnarmable, log=log)
    return census

def lane_coverage_dump_dir(ns) -> str:
    """#1348: where the per-rank coverage dumps go, or ``""`` when disarmed.

    THE SAME DIRECTORY #1292 WRITES ITS FOOTPRINT DUMPS INTO, deliberately and
    in that order of preference: an operator who armed the footprint probe has
    already named a dump directory for this boot, and a second directory for a
    second per-rank dump is one more thing to pass to the ingest and one more
    place for half a boot's evidence to sit. Absent that, the evidence dir the
    launcher already writes every log into.

    Returns ``""`` unless ``--xchg-coverage-diff`` was passed, which is what
    makes the OFF state byte-identical rather than merely cheap: ``build_env``
    POPS the variable on an empty string, so nothing downstream can read a
    value that was never asked for.
    """
    if not getattr(ns, "xchg_coverage_diff", False):
        return ""
    return os.environ.get("FLLIPER_PHASE_FOOTPRINT_DUMP") or getattr(
        ns, "evidence_dir", EVIDENCE_DIR
    )

def check_draft_resident(log_p: str, budget_mib: float = None, tol_mib: float = None,
                         rope_ctx_mib: float = 0.0) -> Dict[str, object]:
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
    budget = (float(_pconst("P_DRAFT_RESIDENT_BUDGET_MIB")) if budget_mib is None
              else float(budget_mib))
    # YaRN x2 27B (heu5g3, 29.09.): the draft's eager RoPE cache grows with the context BEYOND the budget's
    # 262144 (p_rope_draft_delta_mib): resident +128 MiB at 524288 (262528 -> 524672 rows x 128 cols fp32), and
    # its build writes it once more through the allocator (NVML +254 MiB, unaccounted 149.5 -> 275.5). The budget
    # takes the delta, the accounting explains the same delta once. 0 at x1: byte-identical.
    rope_ctx = max(0.0, float(rope_ctx_mib or 0.0))
    budget += rope_ctx
    tol = P_DRAFT_RESIDENT_TOL_MIB if tol_mib is None else float(tol_mib)
    out: Dict[str, object] = {
        "resident_mib": None, "budget_mib": budget, "tol_mib": tol, "over_mib": None,
        "head_released_mib": None, "nvml_delta_mib": None, "unaccounted_mib": None,
        "tag_pool_inactive_mib": None, "outside_torch_mib": None,
        "default_pool_inactive_mib": None, "cache_term": None,
        "other_live_mib": None, "card_free_mib": None,
        "accounting_tol_mib": P_DRAFT_BUILD_ACCOUNTING_TOL_MIB,
        "resident_ok": False, "accounted": False, "ok": False,
    }
    if rope_ctx > 0:
        out["rope_ctx_mib"] = rope_ctx
    try:
        with open(log_p, errors="replace") as f:
            for line in f:
                for key, rx in (
                    ("resident_mib", _RESIDENT_RE),
                    ("head_released_mib", _HEAD_RELEASED_RE),
                    ("tag_pool_inactive_mib", _TAG_POOL_INACTIVE_RE),
                    ("outside_torch_mib", _OUTSIDE_TORCH_RE),
                    ("default_pool_inactive_mib", _DEFAULT_POOL_RE),
                    ("nvml_delta_mib", _NVML_DELTA_RE),
                    ("other_live_mib", _OTHER_LIVE_RE),
                    ("card_free_mib", _CARD_FREE_RE),
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
        # #66: the tag pools' own cache is the THIRD explaining term, measured
        # (tag_pool_occupancy) rather than absorbed into a wider tolerance --
        # empty_cache cannot reach a private pool (#65 of the same day), so
        # NVML counts those bytes while residue and release cannot see them.
        pooled = out.get("tag_pool_inactive_mib")
        pooled = 0.0 if pooled is None or float(pooled) < 0 else float(pooled)
        # #66 (fnFL2v86): die drei Terme oben liessen 533,7 MiB unerklaert bei
        # 256 MiB Toleranz -- eine Verweigerung, die nicht sagen konnte, WO die
        # Bytes sind. Beide neuen Terme beantworten genau das und sie sind
        # ECHTE Posten, keine Toleranzerweiterung:
        #   outside_torch          was auf der Karte waechst, ohne dass torch
        #                          es bucht: CUDA-Kontext, cuBLAS-Workspaces,
        #                          JIT-Kernel. NVML sieht es, torch nie.
        #   default_pool_inactive  was der DEFAULT-Allokator cached, nachdem
        #                          empty_cache lief -- der Tag-Pool-Term sieht
        #                          nur die getaggten Pools.
        # Ein negativer Wert heisst "nicht messbar" und wird zu 0, nie geraten.
        outside = out.get("outside_torch_mib")
        outside = 0.0 if outside is None or float(outside) < 0 else float(outside)
        torch_cached = out.get("default_pool_inactive_mib")
        torch_cached = (
            -1.0 if torch_cached is None else float(torch_cached)
        )
        # WELCHER CACHE-TERM GILT -- EINER, NIE BEIDE (fnFL2v89).
        #
        # `torch.cuda.memory_reserved()` zaehlt die privaten MemPools MIT.
        # Gemessen: torch_cached=2597,7 gegen pooled=1385,2 und released=1212,5,
        # und 1385,2 + 1212,5 = 2597,7 auf die Nachkommastelle. Alle drei zu
        # addieren zaehlt dieselben Bytes zweimal und trieb W11b auf -2064,0 --
        # das Spiegelbild der +533,7, mit denen es anfing. Das Gate verweigert
        # beide Richtungen, und zu Recht.
        #
        # Also: liegt der Gesamtwert vor, ERSETZT er die beiden Teilterme.
        # Fehlt er (-1), bleibt die alte Rechnung, damit ein Boot ohne diese
        # Messung liest wie vorher.
        # #66 (fnFL2v92): DER POSTEN, DER VIER BOOTS LANG FEHLTE. Die Bilanz
        # kannte Modell (`r`) und Cache -- aber nicht die dritte Klasse: was
        # LEBT, ohne dem Modell zu gehoeren. Der Build ist kein nackter
        # `nn.Module`, sondern ein ganzer ModelRunner, und der legt seinen
        # Attention-Workspace an (der FlashInfer-Default allein sind 512 MiB,
        # `flashinfer_backend.py:1125`), dazu Sampler- und Rotary-Puffer.
        # Diese Bytes sind ALLOCATED: der Cache-Term zieht sie gerade ab, und
        # `resident_mib` sieht sie nicht, weil sie weder Parameter noch Buffer
        # sind. Genau deshalb blieben 533,7 MiB uebrig, und genau deshalb hat
        # kein Messpunkt-Fix sie bewegt -- gemessen wurde an den falschen zwei
        # Zustaenden. Der devindex-Graph nennt die Allokationsstelle je
        # Backend; geraten werden musste nichts.
        other_live = out.get("other_live_mib")
        other_live = 0.0 if other_live is None or float(other_live) < 0 else float(other_live)
        if torch_cached >= 0:
            out["unaccounted_mib"] = delta - (r + other_live + torch_cached + outside + rope_ctx)
            out["cache_term"] = "torch_total"
        else:
            out["unaccounted_mib"] = delta - (r + other_live + released + pooled + outside + rope_ctx)
            out["cache_term"] = "tag_pool+head_released"
        out["accounted"] = abs(out["unaccounted_mib"]) <= P_DRAFT_BUILD_ACCOUNTING_TOL_MIB
    out["ok"] = bool(out["resident_ok"] and out["accounted"])
    return out

def measured_record_path() -> str:
    """The sidecar this line writes its own dormant-image measurements into."""
    return f"{EVIDENCE_DIR}/{host_ledger.MEASURED_RECORD_NAME}"

def hostsample_csv_path(tag: str) -> str:
    """The per-boot cushion sampler's CSV -- ONE PRODUCER of the path.

    #1378 Posten 4. The convention host_ledger documents above
    ``ENV_HOSTSAMPLE_CSV``: ``<evidence>/<tag>_<yymmdd>/hostsample_<tag>.csv``
    -- the same dir shape every FLIP-record boot's sampler actually wrote
    (``weg2zwerg_0913/``, ``weg2ring1_0914/``). Published unconditionally
    into both groups' env by :func:`build_env`: a boot whose sampler never
    started reads a NAMED absence at the record (``cushion_min_and_reason``),
    which is evidence -- a silent null told nobody anything, and a 0.0 would
    have made the cross-boot gate look armed while it read nothing.
    """
    return (f"{EVIDENCE_DIR}/{tag}_{time.strftime('%m%d')}/"
            f"hostsample_{tag}.csv")

def boot_state_write(log: Log, state: Optional[str] = None, *, cause: Optional[dict] = None,
                     fields: Optional[dict] = None, event: Optional[Tuple[str, dict]] = None) -> None:
    """Write the launcher's own fields of state.json through the ONE shared writer
    (pdflip/state_file.py, writer=launcher: groups.*, invariants.*, loading/ready,
    dead with origin launcher|rank). The host writer creates the file; while it
    does not exist the launcher says so once and writes nothing. A write the
    order forbids (another writer is already further) is a no-op there, never an
    error here."""
    d = boot_state_dir()
    if not d:
        return
    if not os.path.exists(os.path.join(d, "state.json")):
        if d not in _STATE_MISSING_LOGGED:
            _STATE_MISSING_LOGGED.add(d)
            log(f"IPC STATE: {d}/state.json missing -- the host writer creates it; the launcher writes no state")
        return
    try:
        if state is not None or cause is not None or fields:
            state_file_mod.transition(d, state, cause=cause, fields=fields, writer="launcher")
        if event is not None:
            state_file_mod.add_event(d, event[0], event[1], writer="launcher")
    except OSError as e:
        log(f"IPC STATE WRITE FAILED ({type(e).__name__}): {e}")

def core_cwd_for(group_log: str, tree: str, log: Log) -> str:
    """cwd for a spawned group or the front: ``<EVIDENCE_DIR>/core/<boot stem>``,
    the boot stem being the log name without ``.log`` and without the group
    suffix (``boot_weg2_<tag>_<rev>_<date>.P.log`` -> ``boot_weg2_<tag>_<rev>_<date>``),
    so P, D and the front of one boot share one directory. Falls back to the
    tree, loudly, when the directory cannot be made."""
    if (os.environ.get(CORE_CWD_ENV) or "1").strip() == "0":
        return tree
    stem = os.path.basename(group_log)
    if stem.endswith(".log"):
        stem = stem[: -len(".log")]
    boot = stem.rsplit(".", 1)[0] if "." in stem else stem
    d = os.path.join(EVIDENCE_DIR, "core", boot or "unknown")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        log(f"CORE-CWD FALLBACK {d}: {type(e).__name__}: {e} -- cwd stays {tree}; a crash "
            f"core lands in the tree and the drei init moves it at the next start (ops #164)")
        return tree
    return d

def helper_registry_dir(tag: str) -> str:
    return f"{EVIDENCE_DIR}/helpers_{tag}"

def memts_csv_path(tag: str) -> str:
    return f"{GPU_ARB}/memts_weg2_{tag}.csv"

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
    ``main`` published a flip-order map and a PDFLIP-PP-SPLIT line for a layout
    group P was not launched with -- complete, confident, and wrong, which is
    precisely the failure mode fix 5 names and the one
    ``interleave_pause_order`` cannot refuse.  The pair is a PARAMETER now and
    every caller states which cut it is asking about; the constants remain the
    default because they are the incumbent, and ``solve_p_cut`` round-trips
    its own candidate through this same seam.
    """
    from flliper.srt.distributed.utils import derive_pp_layer_split

    return derive_pp_layer_split(
        list(P_PP_STAGE_RATIO_SCORES if scores is None else scores),
        is_full_attention=list(is_full_attention),
        attn_scores=list(
            P_PP_ATTN_STAGE_RATIO_SCORES if attn_scores is None else attn_scores
        ),
    )

def build_tms_preload(tree: str, venv: str, log: Log) -> str:
    """Build (or reuse) the patched torch_memory_saver preload hook."""
    script = os.path.join(tree, "scripts", "pdflip", "tms", "build_tms_preload.sh")
    r = subprocess.run([script, "--venv", venv], capture_output=True, text=True)
    if r.returncode != 0:
        raise PdFlipLaunchRefused(f"torch_memory_saver preload build failed: {r.stderr[-800:]}")
    so = r.stdout.strip().splitlines()[-1]
    log(f"torch_memory_saver 0.0.9.post1 preload hook REBUILT from python/flliper/srt/pdflip/tms_csrc (PATCH.md: cpu backup "
        f"freed after resume) -> {so}; FLLIPER_PDFLIP_TMS_PRELOAD_SO set for both groups")
    return so

def d_awake_rest(cards: List[Card], profile: Optional[str] = None) -> Tuple[Optional[List[Optional[int]]], str]:
    """``D_AWAKE_REST_MIB`` of the profile per D rank (``None`` entries: not
    measured on that rank), or ``(None, "")`` where the profile never measured
    it (27B today: its D budget stays byte-identical). rc12c: the rest an awake
    D rank holds beyond its BOOKED form (pdflip/d_awake_rest.py)."""
    try:
        vals = list(_pconst("D_AWAKE_REST_MIB", profile))
    except KeyError:
        return None, ""
    if len(vals) != len(cards):
        raise PdFlipLaunchRefused(
            f"D_AWAKE_REST_MIB has {len(vals)} entries for {len(cards)} cards; "
            f"re-measure with python -m flliper.srt.pdflip.d_awake_rest")
    return [None if v is None else int(v) for v in vals], _pconst_boots("D_AWAKE_REST_MIB", profile)

def d_admit_need_record(profile: Optional[str] = None, n: Optional[int] = None,
                        cards: Optional[Sequence[Card]] = None,
                        model: Optional[str] = None) -> Tuple[Optional[List[float]], str]:
    """PDFLIP-ADMIT-BOOK: the profile's ``D_ADMIT_NEED_MIB`` per card (one value covers every card) and its boots,
    or ``(None, why)`` -- no record, the switch off (``FLLIPER_PDFLIP_ADMIT_BOOK=0``), a vector that does not fit
    ``n`` cards, or a FORM the record was not measured on (checkpoint ``model``, else the published form's, and the D
    cards' inventory against ``budget_rest.ADMIT_NEED_FORMS``; unknown checkpoint or no cards = not this form, fail
    closed: nothing booked)."""
    if not _budget_rest.admit_book_on():
        return None, f"{_budget_rest.ADMIT_ENV}=0"
    try:
        vals = list(_pconst(_budget_rest.ADMIT_RECORD, profile))
    except (KeyError, TypeError):
        return None, f"no {_budget_rest.ADMIT_RECORD} record"
    if not vals or any(v is None for v in vals):
        return None, f"{_budget_rest.ADMIT_RECORD} {vals!r} has an unmeasured card"
    prof = profile if profile is not None else pdflip_form.DEFAULT_PROFILE
    row = pdflip_form.profile_row(prof)
    rec = row.constants.get(_budget_rest.ADMIT_RECORD) if row is not None else None
    boots = tuple(getattr(rec, "boots", ()) or ()) if rec is not None else ()
    if model is None:
        form = pdflip_form.current_form()
        model = form.model if form is not None else ""
    try:
        inventory = card_identity.inventory_signature(list(cards)) if cards else ()
    except Exception:  # noqa: BLE001 -- an unreadable inventory is not this form (fail closed)
        inventory = ()
    why = _budget_rest.admit_form_mismatch(boots, pdflip_form.model_key(model or ""), inventory)
    if why is not None:
        return None, why
    if n is not None and len(vals) not in (1, int(n)):
        out = [float(vals[0])] * int(n) if len(set(vals)) == 1 else None
        if out is None:
            return None, f"{_budget_rest.ADMIT_RECORD} {vals!r} does not fit {n} cards"
        return out, _pconst_boots(_budget_rest.ADMIT_RECORD, profile)
    out = [float(v) for v in vals]
    if n is not None and len(out) == 1:
        out = out * int(n)
    return out, _pconst_boots(_budget_rest.ADMIT_RECORD, profile)

def owned_miss_record_root() -> str:
    """The records root the D ranks WRITE to. What the pre-rename tree wrote lies next to it under the old subsystem name
    and is READ as well (``pool_miss_cost.record_dirs_for``, F0-F); nothing is moved."""
    return os.path.join(EVIDENCE_DIR, "records", "pdflip")

def d_fixed_record(profile: Optional[str] = None) -> Tuple[Optional[List[Optional[float]]], str]:
    """``D_FIXED_MIB`` of the profile (the #145 fixed post measured on the
    form that runs today), or ``(None, "")`` -- the builtin reference stands."""
    try:
        vals = list(_pconst("D_FIXED_MIB", profile))
    except KeyError:
        return None, ""
    return [None if v is None else float(v) for v in vals], _pconst_boots("D_FIXED_MIB", profile)

def d_activation_record(profile: Optional[str] = None) -> Tuple[Optional[List[Optional[float]]], str]:
    """rc12e: ``D_ACTIVATION_MIB`` of the profile (the D extend's measured
    transient maximum per rank), or ``(None, "")`` -- the reference stands."""
    try:
        vals = list(_pconst("D_ACTIVATION_MIB", profile))
    except KeyError:
        return None, ""
    return [None if v is None else float(v) for v in vals], _pconst_boots("D_ACTIVATION_MIB", profile)

def d_overshoot_record(profile: Optional[str] = None) -> Tuple[Optional[List[int]], str]:
    """``D_OVERSHOOT_MIB`` (peak - budget, the qwen27b record), or ``(None, "")``
    for a profile that prices D's awake excess as ``D_AWAKE_REST_MIB`` instead
    (rc12c: peak - budget has no fixpoint once the budget stops binding)."""
    try:
        return list(_pconst("D_OVERSHOOT_MIB", profile)), _pconst_boots("D_OVERSHOOT_MIB", profile)
    except KeyError:
        return None, ""

def d_extend_growth_record(profile: Optional[str] = None) -> Tuple[Optional[List[Optional[float]]], str]:
    """rc12f: ``D_EXTEND_GROWTH_MIB`` of the profile (what one D extend adds
    to torch reserved, measured), or ``(None, "")``."""
    try:
        vals = list(_pconst("D_EXTEND_GROWTH_MIB", profile))
    except KeyError:
        return None, ""
    return [None if v is None else float(v) for v in vals], _pconst_boots("D_EXTEND_GROWTH_MIB", profile)

def d_extend_growth_per_row_record(profile: Optional[str] = None) -> Tuple[Optional[List[Optional[float]]], str]:
    """rc12g: ``D_EXTEND_GROWTH_PER_ROW_MIB`` of the profile (reserved growth of
    one D extend per row, measured), or ``(None, "")`` -- no chunk cap."""
    try:
        vals = list(_pconst("D_EXTEND_GROWTH_PER_ROW_MIB", profile))
    except KeyError:
        return None, ""
    return [None if v is None else float(v) for v in vals], _pconst_boots("D_EXTEND_GROWTH_PER_ROW_MIB", profile)

def log_boot_state(log, msg: str) -> None:
    """#233 review: a BOOT-STATE line (the frozen corridor pointer, a group's
    recipe) goes into the launcher log FILE and stderr, never stdout. The
    dry-run plan dump (``propose_oracle``: the golden is the normalised stdout
    of ``main``) is a PLAN and must not move with the box's pointer file or a
    recipe hash (``test_planer_referenz_n3_1006`` goldens); a boot's
    launcher.log still carries the line for the metal check."""
    line = f"[{_now()}] PDFLIP-LAUNCH {msg}"
    print(line, file=sys.stderr, flush=True)
    fh = getattr(log, "fh", None)
    if fh:
        fh.write(line + "\n")
        fh.flush()

def _alive_not_zombie(pid: int, proc_root: str = "/proc") -> bool:
    """``_alive`` that does not count a zombie (a daemonized MPS control is reparented to PID 1 of the
    container, which may never reap it)."""
    if not _alive(pid):
        return False
    try:
        with open(f"{proc_root}/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True

def p_cut_calibration_line(model_path: str, quant_method: str) -> Optional[str]:
    """A NAMED line when a non-incumbent checkpoint has no P-cut calibration
    record: ``argv_p`` otherwise hands it the INT8 incumbent vector unasked
    (a 64-layer checkpoint without a record is not refused there). None for
    the incumbent family (compressed-tensors) and for a checkpoint that HAS a
    record -- default boots print nothing new."""
    qm = str(quant_method or "").lower()
    if qm in ("compressed-tensors", ""):
        return None
    dg, why = host_ledger.checkpoint_digest(model_path)
    if not dg:
        return ("PDFLIP P-CUT UNCALIBRATED %s checkpoint: its content digest is unreadable "
                "(%s) -- the P cut is the pinned one or the INT8 incumbent, NOT a "
                "measurement of this checkpoint" % (qm, why))
    rec, rec_why = host_ledger.read_pp_calibration(dg)
    if rec is not None:
        return None
    return ("PDFLIP P-CUT UNCALIBRATED %s checkpoint %s: %s. The P cut of this boot is "
            "the pinned --pp-layer-ratio / --pp-stage-ratio or the INT8 incumbent "
            "%s -- per-layer costs of %s differ per card (under --fp8-uniform-marlin "
            "every rank runs Marlin: FP8 W8A16, NVFP4 W4A16), so measure it: #PGAP ladder, "
            "then pdflip/tools/pcut_refit.py or --pp-cut-stage-fit with THIS "
            "checkpoint's P log" % (qm, dg[:12], rec_why, _csv(P_PP_STAGE_RATIO_SCORES), qm))

def calib_identity_of(ns) -> Optional["pdflip_form.CalibrationIdentity"]:
    """UNIFY S3: this boot's ONE calibration identity (pdflip_form
    .calibration_identity): checkpoint AND form, AND -- where the profile's
    records row names it (27B, line_identity 76e87ac3b2) -- a boot commit that
    is an ancestor of the tree this launcher runs. None without a resolved
    form (a desk caller, a teardown)."""
    form = getattr(ns, "pdflip_boot_form", None)
    if form is None:
        return None
    tree = getattr(ns, "tree", "") or ""
    return pdflip_form.calibration_identity(
        getattr(ns, "model", ""), EVIDENCE_DIR, form,
        repo=os.path.abspath(tree) if tree else "",
        power_limits=calib_power_limits(form))

def cold_tier_shm_post(ns, model: str, p_stage_layers=None, karte_path: str = "",
                       mounts_text: Optional[str] = None) -> Tuple[float, str]:
    """29.09. Planer post COLD_TIER_SHM -> ledger term ``cold_tier_shm_gib``
    (GiB, both moments): the expert store when its directory is on a tmpfs.

    MEASURED z30w-park (29.09. 09:12Z, Docker memory.max 84 GiB): shmem 53.36
    GiB of which /dev/shm 7.9 -- the store (/mnt/nf-experts, a per-boot tmpfs
    mount, map 'Store ~39.2 GiB') was the largest host post and priced
    nowhere; the ledger said 47.72 against 79.46 nonreclaim.

    Bytes the tmpfs already holds are credited: they are in the ledger's
    origin reading (nonreclaim now) and must not be charged twice."""
    store_dir = expert_store_dir_of(ns)
    if not store_dir:
        return 0.0, (f"{COLD_TIER_MARKER} post cold_tier_shm 0.00 GiB: no expert store "
                     f"(FLLIPER_MOE_EXPERT_STORE_DIR not in --env-d/--env-p/env)")
    mnt, fs, _src = fs_of(store_dir, mounts_text)
    if fs not in COLD_TIER_SHM_FS:
        return 0.0, (f"{COLD_TIER_MARKER} post cold_tier_shm 0.00 GiB: store {store_dir} on "
                     f"fs={fs or '?'} mount={mnt or '?'} -- page cache, reclaimable, not a post")
    planned, how = planned_expert_store_bytes(ns, model, p_stage_layers, karte_path)
    if planned is None:
        return 0.0, (f"{COLD_TIER_MARKER} post cold_tier_shm UNPRICED: store {store_dir} is "
                     f"{fs} (shmem) but {how}")
    resident = _tree_bytes(store_dir)[0] if os.path.exists(store_dir) else 0
    post = max(0, planned - resident) / host_ledger.GIB
    return post, (f"{COLD_TIER_MARKER} post cold_tier_shm {post:.2f} GiB = planned "
                  f"{planned / host_ledger.GIB:.2f} GiB ({how}) - already resident "
                  f"{resident / host_ledger.GIB:.2f} GiB (in the origin reading); store "
                  f"{store_dir} on {fs} mount={mnt}, both moments")

def derived_family_cost(ns, cards: Sequence["Card"], families, chunk_tokens: int):
    """HW-P1c 1003: the P-cut family cost (linear / attention ms per layer per
    stage) for a P group whose cards are a SUBSET of the calibration rig's.

    The split is a closed form of ONE measured cut (``MEASURED_MS_PER_LAYER`` at
    ``P_PP_STAGE_RATIO_SCORES``) and the deep anchor; it is fitted on THAT
    reference basis exactly as on the reference rig and then mapped to the live
    stages by calibration class (a stage on a 3080 takes the slowest cost the
    reference measured on a 3080: conservative). The family RATIO is, by the
    function's own assumption, a property of the model shape and not of the
    card, so it is kept; only the per-stage scale follows the live cards."""
    from flliper.srt.planner import pp_cut as _pp_cut
    from flliper.srt.pdflip import profile_records as _pr

    prof = str(getattr(ns, "profile", None) or pdflip_form.DEFAULT_PROFILE)
    cal, _ = records_inventory(prof)
    row = pdflip_form.profile_row(prof)
    ref_ms = _csv_floats(str(row.constant("MEASURED_MS_PER_LAYER")))
    ref_counts = list(P_PP_STAGE_RATIO_SCORES)
    ref_attn = _pp_cut.attention_counts(families, ref_counts)
    anchor = next((i for i, c in enumerate(cal) if c == ATTN_ANCHOR_CARD_CLASS), None)
    if anchor is None or len(ref_ms) != len(cal):
        raise PdFlipLaunchRefused(
            f"{card_identity.CODE_UNCALIBRATED}: the P-cut family cost of profile {prof!r} has no "
            f"reference basis for the calibrated inventory [{', '.join(cal)}]")
    ref_cost, ref_prov = _pp_cut.family_costs_from_measurement(
        measured_ms_per_layer=ref_ms,
        measured_counts=ref_counts,
        measured_attn_counts=ref_attn,
        chunk_tokens=int(chunk_tokens),
        ref_prefix_tokens=float(ns.pp_cut_calibration_prefix_tokens),
        anchor_stage=anchor,
        anchor_attn_ms_per_layer=float(ns.pp_cut_attn_anchor_ms),
        anchor_prefix_tokens=float(ns.pp_cut_attn_anchor_prefix_tokens),
    )
    live = tuple(card_identity.inventory_signature(cards))
    lin, att = [], []
    for j, c in enumerate(live):
        idx = [i for i, k in enumerate(cal) if k == c]
        if not idx:
            text = (f"{card_identity.CODE_UNCALIBRATED}: stage card class {c} has no measured twin in "
                    f"[{', '.join(cal)}]; the P-cut family cost is not derived across classes")
            # AP2 1006 (the AP1 W19 rule, Q-710): --force borrows the cost of the
            # calibrated class of the SAME ARCH (sm_86 -> RTX3080, sm_120 -> RTX5090),
            # named UNMEASURED; no twin of the arch (sm_89) stays refused.
            twin = card_identity.arch_twin_class(cards[j]) if j < len(cards) else None
            idx = [i for i, k in enumerate(cal) if twin and k == twin]
            if not idx:
                raise PdFlipLaunchRefused(text)
            _refuse_value_once(
                card_identity.CODE_UNCALIBRATED,
                text + f"  --force BORROWS the per-layer cost of class {twin} (same arch) for this stage: "
                "UNMEASURED here (HW-UNCALIBRATED).")
        lin.append(max(ref_cost.linear_ms_per_layer[i] for i in idx))
        att.append(max(ref_cost.attn_ms_per_layer_at_ref[i] for i in idx))
    cost = _pp_cut.FamilyDepthCost(
        linear_ms_per_layer=tuple(lin), attn_ms_per_layer_at_ref=tuple(att),
        chunk_tokens=int(chunk_tokens), ref_prefix_tokens=float(ns.pp_cut_calibration_prefix_tokens))
    return cost, (f"HW-DERIVE family cost for [{', '.join(live)}] from the reference basis "
                  f"[{', '.join(cal)}] by card class (slowest same-class stage): " + ref_prov)

def apply_profile_d_bs_default(ns, argv: Sequence[str]) -> int:
    """H91b/H95 (Nutzer-Design 25.09.): ``--profile nextflash`` without an
    explicit ``--d-bs`` runs D with up to ``DEFAULT_D_BS_NEXTFLASH`` (6) seats
    -- the UPPER BOUND of the dynamic seat count n = 1..6 per D phase
    (``d_seats.phase_seats`` from the wake's handoff_n); H91b's fixed 2 is the
    case n = 2. An explicit ``--d-bs`` always wins and stays the hard bound
    (``bs_source``: a told value is a choice, even when it equals a default).
    Returns the resolved value."""
    if (profile_standard_form(ns)
            and bs_source("--d-bs", argv) == "default"):
        ns.d_bs = DEFAULT_D_BS_NEXTFLASH
    return int(ns.d_bs)

def apply_profile_d_pool_waves_default(ns) -> Optional[str]:
    """H95: ``--profile nextflash`` runs D's expert pool with
    ``DEFAULT_D_POOL_WAVES_NEXTFLASH`` overflow waves unless ``--env-d``
    states FLLIPER_OPT_MOE_POOL_OVERFLOW_WAVES itself (0 = the H91b bound).
    Written INTO ``ns.env_d`` so the D-FRACTION-SOLVE (which reads --env-d)
    and the D group's environment (build_env applies --env-d last) see ONE
    value. Returns the line naming it, or None when nothing was added."""
    from flliper.srt.planner.expert_residency import POOL_OVERFLOW_WAVES_ENV

    if not profile_standard_form(ns):
        return None
    env_d = str(getattr(ns, "env_d", "") or "")
    if POOL_OVERFLOW_WAVES_ENV in parse_group_env(env_d):
        return None
    item = "%s=%d" % (POOL_OVERFLOW_WAVES_ENV, DEFAULT_D_POOL_WAVES_NEXTFLASH)
    ns.env_d = (env_d.rstrip(";") + ";" + item) if env_d.strip() else item
    # rc12e: the launcher's own default -- the D solve may raise it to what
    # the solved form needs (OVERFLOW-WAVES); a value told in --env-d is kept.
    ns.d_pool_waves_derived = True
    return ("D-POOL-WELLEN (H95): --profile nextflash -> --env-d %s (Ueberlaufwellen: "
            "der bs1-Scratch traegt jede Sitzzahl 1..--d-bs; --env-d %s=0 = die "
            "H91b-Schranke)" % (item, POOL_OVERFLOW_WAVES_ENV))

def apply_profile_d_seat_vram_default(ns) -> Optional[str]:
    """H95c: ``--profile nextflash`` backs D's seat posts only for the
    phase's occupied seats (FLLIPER_OPT_PDFLIP_D_SEAT_VRAM=1 into ``ns.env_d``)
    unless ``--env-d`` states the switch itself (=0: H95 B's fixed posts).
    Returns the line naming it, or None when nothing was added."""
    if not profile_standard_form(ns) or not DEFAULT_D_SEAT_VRAM_NEXTFLASH:
        return None
    env_d = str(getattr(ns, "env_d", "") or "")
    if "FLLIPER_OPT_PDFLIP_D_SEAT_VRAM" in parse_group_env(env_d):
        return None
    item = "FLLIPER_OPT_PDFLIP_D_SEAT_VRAM=1"
    ns.env_d = (env_d.rstrip(";") + ";" + item) if env_d.strip() else item
    return ("D-SITZ-VRAM (H95c): --profile nextflash -> --env-d %s (Sitz-Posten nur fuer "
            "die besetzten Sitze der Phase, der Rest als Experten-LRU-Zeilen auf TP0; "
            "--env-d FLLIPER_OPT_PDFLIP_D_SEAT_VRAM=0 = H95 B)" % item)
