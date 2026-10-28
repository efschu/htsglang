"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
import os
import re
import time

from flliper.srt.compat_shims import name_variants as _name_variants
from flliper.srt.name_compat import canonical_env
from flliper.srt.pdflip import admin_key as admin_key_mod
from flliper.srt.pdflip import card_identity
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import refusals
from flliper.srt.pdflip import state_file as state_file_mod
from flliper.srt.pdflip.launcher_d_leaf import EARLY_READ_FACTS
from flliper.srt.pdflip.launcher_types import BootState
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import PCutFacts
from flliper.srt.pdflip.launcher_types import PFrCap
from flliper.srt.pdflip.launcher_types import PdFlipDKvStageMaxRefused
from flliper.srt.pdflip.launcher_types import PdFlipDKvStageWavesRefused
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_types import PdFlipStoreDiskRefused
from flliper.srt.planner import p_card_chunk as _p_card
from flliper.srt.planner import power_limit as _power
from flliper.srt.registry import nvml as nvml_registry
import hashlib
import json
import math
import shlex
import subprocess

#: --p-prefill-graph (27B line, 2026-09-24; user: P on 512-token chunks, and
#: for the fixed chunk a prefill CUDA graph -- "ja unbedingt"). 0 = off, the
#: default: argv_p, the P form key and every ledger input are byte-identical
#: to a tree without the switch. N > 0 = group P captures ONE full prefill
#: graph of N tokens (backend 'full', the only prefill backend that can put
#: its capture pool in the memory saver's cuda_graph tag -- the same
#: sleep/wake the decode graphs on D ride) and runs its chunks at N tokens,
#: because a bucket below the chunk size would replay only the rest chunk.
#: Measured motive (xsn422 #PGAP): the host launch of one P forward is
#: 63/42/74 ms per stage, ~24 ms fixed + ~0.9 ms per layer, independent of the
#: token count, against ~48-71 ms of compute for a 512-token chunk.
P_PREFILL_GRAPH_DEFAULT = 0
#: --p-prefill-graph-policy (27B line, user order 26.09. ~05:15Z: "cuda graphen
#: im prefill nur anschalten wenn es was bringt, wenn es negativ ist,
#: ausschalten"). The rule is pdflip/p_graph_policy.py; 'auto' without a
#: calibration table captures exactly the pre-switch set (argv byte-identical).
P_PREFILL_GRAPH_POLICY_DEFAULT = "auto"
_P_PREFILL_GRAPH: Dict[str, object] = {
    "bucket": P_PREFILL_GRAPH_DEFAULT, "tiny": (), "captured": (), "legacy": (),
    "policy": P_PREFILL_GRAPH_POLICY_DEFAULT, "verdicts": (), "calibration": None, "lines": (),
    "extras": (), "min_gain": None, "cut_check": None,
}
#: --p-deep-split-from (27B line, 2026-09-24): 0 = off, the default.
P_DEEP_SPLIT_FROM_DEFAULT = 0
#: --p-chunk-policy (27B + NF line, user orders 25.09. ~21:40Z/21:41Z:
#: "dynamische chunkgroesse brauchen wir. umsetzen"). The rule lives in
#: pdflip/p_chunk_policy.py (pure; the NF line takes the same interface). This
#: block only turns the launcher's flags into a PolicySpec, prices P's
#: --chunked-prefill-size at the plan's CEILING and hands the spec to group P.
#: 'fixed' (the default for this release candidate) changes NOTHING: no env,
#: argv_p and the P form key byte-identical.
P_CHUNK_POLICY_DEFAULT = "fixed"
#: The plan's ceiling. 2048 = 4x the 512 graph chunk and half the 4096 P ran
#: eager before the graph (so inside the envelope P's corridor already funded);
#: the planner prices group P's activation corridor, frames and barlink
#: buffers at it, because under 'dynamic' it IS P's --chunked-prefill-size.
P_CHUNK_MAX_DEFAULT = 2048
P_CHUNK_MODEL_DEFAULT = "builtin-int8"
#: BUILTIN INT8 stage model (desk, 25.09., MEASURED where stated):
#: a_ms / b_ms_per_1k = pgap_stage_fit.fit_rank_lines over prefixes [0, 32768)
#: of boot weg2rc7c (a54f21cda0, INT8, cut 42/11/11, 512 graph, 1076-1082
#: device-bound chunks per rank, sd 2.1-2.5 ms): per 512 chunk,
#: t = a + b * (prefix + 256) / 1000.
#: fwd_overhead_ms = PP0's mean #PGAP gpu_gap_ms below 50 ms (3.73, n=2101):
#: the bottleneck stage's per-forward idle, i.e. a per-CHUNK cost; PP1/PP2's
#: gaps are pipeline waiting, which the flow shop produces itself.
#: eager_floor_ms = the host launch of an EAGER P forward, xsn422 #PGAP
#: (63/42/74 ms per stage, before the graph): a chunk above the largest graph
#: bucket runs eager and is paced by it when its device time is shorter.
#: UNIFY S9: a RECORD of the 27B profile (profile_records_data/qwen27b.json,
#: P_CHUNK_BUILTIN fmt=int8) carrying its power limit (400/230 W, row 27) and
#: the P cut it was fitted on (42,11,11: a per-stage table holds only for its
#: cut, p_chunk_model_cut_check). Values unchanged.
_P_CHUNK_BUILTIN_RECORD = pdflip_form.profile_record(
    "P_CHUNK_BUILTIN", pdflip_form.PROFILE_QWEN27B, fmt="int8").record
P_CHUNK_BUILTIN_INT8 = tuple(dict(row) for row in _P_CHUNK_BUILTIN_RECORD.value)
#: FP8 (27.10., desk/fp8-optima-1027): the same record kind for the W8A16-Marlin checkpoint (``--fp8-uniform-marlin``),
#: fitted from the 512-token #PGAP lines of boot dkr27bfp8bar1drei10271317 on cut 44/11/9. Its width curve is UNMEASURED
#: (only 512-token forwards exist), so a profile uses it with ``--p-chunk-mscale flat`` until the P sweep measures it.
_P_CHUNK_BUILTIN_RECORD_FP8 = pdflip_form.profile_record(
    "P_CHUNK_BUILTIN", pdflip_form.PROFILE_QWEN27B, fmt="fp8").record
P_CHUNK_BUILTIN_FP8 = tuple(dict(row) for row in _P_CHUNK_BUILTIN_RECORD_FP8.value)
#: ``--p-chunk-model`` names that are a record of this module: name -> (rows, record, label of the fit)
P_CHUNK_BUILTIN_SOURCES = {
    "builtin-int8": (P_CHUNK_BUILTIN_INT8, _P_CHUNK_BUILTIN_RECORD, "weg2rc7c fit [0,32k)"),
    "builtin-fp8": (P_CHUNK_BUILTIN_FP8, _P_CHUNK_BUILTIN_RECORD_FP8, "dkr27bfp8bar1drei10271317 fit [0,32k)"),
}
_P_CHUNK: Dict[str, object] = {"policy": P_CHUNK_POLICY_DEFAULT, "spec": None, "power_lines": (),
                             "profile": None, "ns": None}
#: The card class of each P stage OF THE CALIBRATION RIG the builtin chunk
#: model was fitted on (PP0 on the 5090, PP1/PP2 on the 3080s) -- a property
#: of that RECORD, not of the live rig. HW-GENERIC 1002: the live stages'
#: classes come from :func:`p_stage_live_classes` (order_cards + card
#: identity); a stage whose live class differs from its calibration class is
#: named by :func:`p_stage_class_mismatch_lines`.
P_STAGE_CARD_CLASSES = card_identity.REFERENCE_INVENTORY
#: --p-layer-split (27B, user order 26.09. ~05:35Z: "dynamisches layersplit ...
#: ohne neue ranks ... umsetzen"). The rule lives in pdflip/p_layer_split.py
#: (pure); design and evidence /spinning/gpu-arb/docs/DYN_LAYER_SPLIT.md.
#: 'static' (the default until the metal ladder) changes NOTHING: no env, no
#: line, argv and the P form key byte-identical. 'dynamic' builds and prints
#: the spec, the swing slab and its dry-run plans and hands group P the spec
#: env; the rank-side executor is pdflip/p_layer_split_runtime.py (resident
#: swing window under tag weights_swing, write-back on the proxy frame, the
#: row on the #791 decision, prefix pull + weight refill upstream, graph
#: bypass for non-home cuts). With PREFIX PULL by default (sec. 5: without
#: pull there is no gain); --p-layer-split-pull-gbps 0 is the V1 frontier.
P_LAYER_SPLIT_DEFAULT = "static"
_P_LAYER_SPLIT: Dict[str, object] = {"policy": P_LAYER_SPLIT_DEFAULT, "spec": None, "lines": ()}
#: --p-gc-freeze (NF 13.10., boot b60520f01d: a gen-2 GC over the ~1.12 M never-frozen
#: boot objects stood PP0/PP1 still 4.4-5.4 s inside a flip once the cold heap was
#: swapped out; ``done/nf-flip-langsam-analyse-1013.md``).  'on' = every P scheduler
#: calls gc.freeze() once after boot, as --d-gc-freeze does for D.  'off' (default) =
#: no env from this flag.  The nextflash and qwen27b rows turn the switch on for BOTH
#: groups by default (form.*_GROUP_SWITCH_DEFAULTS -> --env-p/--env-d
#: FLLIPER_PDFLIP_GC_FREEZE=1; ``--env-p/--env-d FLLIPER_PDFLIP_GC_FREEZE=0`` = the old
#: path; RC4 fix round 1, user order 14.10.: new features default ON).
P_GC_FREEZE_DEFAULT = "off"
#: #251c: the --env-d keys of D's KV stage form (the launcher's form values)
D_KV_STAGE_KEYS = ("FLLIPER_PDFLIP_D_KV_STAGE_TOKENS", "FLLIPER_PDFLIP_D_KV_STAGE_ROWS",
                   "FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS")
D_KV_STAGE_LRU_FLOOR_MARKER = "LRU-UNTERGRENZE (#251c, gemessener Spitzenbedarf)"
#: D's prefill chunk width.  Named once because it is X's FLOOR (K5): a bound
#: below one chunk would refuse work D must be able to do to make progress.
CHUNKED_PREFILL_TOKENS = 4096
#: PDFLIP-FORM (24.09.): the checkpoint(s) the #114 slopes above were MEASURED
#: on -- fnFL2x113/x116/x118 all ran Qwen3.8-Flash-Next-INT4. A transient is a
#: property of one model's prefill (hidden size, expert fan-out, attention
#: form), so the slopes are published only for that checkpoint.
#: UNIFY S3: read off the registry rows (ModelProfile.prefill_transient_checkpoints).
P_PREFILL_TRANSIENT_CALIBRATION_MODELS: Tuple[str, ...] = tuple(
    c for _row in pdflip_form.PROFILES.values() for c in _row.prefill_transient_checkpoints
)
#: H87: ``{model_key of this boot: (calibration model, why)}`` -- set ONCE by
#: :func:`note_calibration_footprint_alias` in ``main`` when this boot's
#: checkpoint is not a calibration model by NAME but is one by MEMORY FOOTPRINT
#: (an abliterated derivative). Keyed by the boot's own model name, so a stale
#: entry can never answer for another checkpoint. ``build_env`` sees only the
#: form's model NAME, which is why the verdict is taken where the path is known.
_CALIBRATION_FOOTPRINT_ALIAS: Dict[str, Tuple[str, str]] = {}
#: H92 (NF, user order 25.09. ~21:40Z via 27B): the --p-chunk-policy of the
#: nextflash profile -- ONE flag, ONE installer (:func:`apply_p_chunk_policy`),
#: which delegates here for --profile nextflash. The NF side (profile by model
#: key, hard limits, FR_P coupling, stream budget) lives in pdflip/p_chunk_nf.py.
#: 'fixed' changes NOTHING. Under 'dynamic' the plan's CEILING is group P's
#: --chunked-prefill-size (P_CHUNKED_PREFILL_TOKENS): argv_p stays
#: byte-identical, only the P group's env carries the spec.
#: The shared flags keep the 27B defaults in argparse; for NF a flag the
#: operator did NOT write reads NF's own default (model 'auto', max 0 = the P
#: chunk, dynamic-min -1 = the P chunk) -- decided by the argv, not by value.
P_CHUNK_NF_MODEL_DEFAULT = "auto"
WEIGHT_SOURCE_EXCHANGE = "exchange"
#: H92c: the runtime's HARD FLOOR per running P request on P's posture
#: (``mem_cache/mamba_pool_floor.mamba_slots_per_running_req``): 1 active + 1
#: ping-pong (P runs --disable-overlap-schedule) + 1 donation + 1 pinned
#: checkpoint (``--hicache-write-policy write_back``: no #755 reorder, so the
#: two do not share). Measured, fnFL2h91bb2 P log:
#: ``MAMBA-FLOOR pool=32 floor=32 retention_budget=0 (8 running requests x 4 =
#: 32)``. Everything above ``seats x this`` is the anchor-retention budget; at 0
#: every host backup of an anchor is refused. The planner only NAMES it on the
#: budget-posts line (the launch/runtime riegel is AW's); a test holds this
#: constant against the runtime function on P's posture.
P_MAMBA_FLOOR_SLOTS_PER_SEAT = 4
#: refusal code of a --cards value that names no card / a card twice
CODE_CARDS = "HW-CARDS"
#: The module the stock server is started as.  A process is a server only when
#: its argv carries ``-m`` followed by EXACTLY this token -- see
#: :func:`live_launch_servers` for why a substring of the command line is not.
LAUNCH_SERVER_MODULE = "flliper.launch_server"
#: Rename transition (compat_shims): this tree's module FIRST, then the other package
#: generation's.  A server started by a launcher of the other generation holds cards and
#: /dev/shm exactly like one of ours; without its spelling here the #1217 census would not
#: even look at it (FL5, RENAME_PLAN 8.14 item 1).
LAUNCH_SERVER_MODULES: Tuple[str, ...] = _name_variants(LAUNCH_SERVER_MODULE)
#: L3P (user 2026-09-27: "der l3 hicache wird jedesmal verworfen, oder? der soll
#: natürlich persistent sein"): the disk store is the RETENTION tier, so it is
#: one directory per model that every boot reattaches to, not one per boot tag.
#: Named ``l3-<model>-<hash of the model path>``; the prefix is what the sweep
#: spares. What keeps a reused page honest is the KEY, not the directory: every
#: page key carries compute_model_identity_hash (model path, revision, dtype,
#: quantization, kv_cache_dtype, uneven-TP vectors), so a page written in
#: another byte format is a clean miss, never a wrong hit.
L3_PERSIST_PREFIX = "l3-"
L3_IDENTITY_FILE = "L3_IDENTITY.json"
#: L3-REVOKE (28.09., rc12z17): a persistent store cannot tell a page computed
#: with broken weights from a good one -- the identity separates what was
#: CONFIGURED, not what a running boot COMPUTED. rc12z17 (103bfdf29a) wiped
#: D-TP0's live expert bank at its first S1 wake (10:51:24Z, n=4, rows +33 ->
#: +23: tms_set_spans(now=True) released the one extent that held every row)
#: and served garbage until the stop; every page it wrote in that window had
#: the store's own identity and would have been reused as a hit. The operator
#: names such a window in ``L3_REVOKED.json`` (``l3_revoke_window``); the next
#: attach, before any rank starts, MOVES every ``.bin`` written inside it (by
#: mtime -- a page is written once, whole, via tmp+rename) into the sibling
#: ``<store>.revoked``. Moved, never deleted (deleting is the user's).
L3_REVOKED_FILE = "L3_REVOKED.json"
L3_REVOKED_SUFFIX = ".revoked"
L3_SUFFIX_REGISTRY = "L3_SUFFIXES"
#: 27B FP8 (Qwen/Qwen3.8-27B-FP8, block 128x128) on the pdflip flip, the simple
#: variant: ONE byte layout on every card. The kernel choice is RANK-LOCAL
#: (fp8.py: ``use_marlin = force_marlin or can_auto_enable_marlin_fp8()``,
#: sm80..88): a 3080 repacks to Marlin ``int32 [K/16, 4N]`` + group scales
#: ``[K/128, N]``, a 5090 keeps native ``fp8 [N, K]`` + ``weight_scale_inv
#: [N/128, K/128]`` -- and the exchange copies bytes between the cards, so the
#: layouts must agree. Forcing Marlin everywhere makes them agree (the geometry
#: is the one compressed-tensors pack-quantized already moves: MIXED_FUSED_COLS,
#: weg2xsn258) and costs the 5090 its native FP8 GEMM. Marlin's per-card lock
#: workspace stays on the module: the exchange coverage books it as
#: local_scratch and every weights resume re-zeroes it
#: (weight_exchange.zero_local_scratch, adopted from the NF line).
FP8_UNIFORM_MARLIN_ENV = {
    "FLLIPER_FORCE_FP8_MARLIN": "1",
}
#: The checkpoint families --fp8-uniform-marlin makes one byte layout for:
#: ``fp8`` (Qwen/Qwen3.8-27B-FP8) and ``modelopt`` -- NVIDIA ModelOpt exports,
#: e.g. RadixArk Qwen3.8-27B-NVFP4 (MIXED_PRECISION: FP8 per-tensor attention
#: and GDN projections, NVFP4 g16 MLP and lm_head). Under ``modelopt`` the FP8
#: linears take Marlin through FLLIPER_FORCE_FP8_MARLIN (ModelOptFp8LinearMethod,
#: upstream #31340) and the NVFP4 linears through the server argument below; on
#: the 5090 both would otherwise run native kernels with other byte layouts
#: (fp8 [N,K] + scalar scale; NVFP4 CUTLASS with 128x4-swizzled block scales).
UNIFORM_MARLIN_QUANT_METHODS = ("fp8", "modelopt")
#: The server argument that puts every NVFP4 linear of a ModelOpt checkpoint on
#: Marlin (fp4_utils.initialize_fp4_gemm_config: ``auto`` resolves per rank --
#: cutlass on sm_120, marlin on sm_80..89).
FP4_UNIFORM_MARLIN_ARGV = ("--fp4-gemm-backend", "marlin")
#: Backlog #38 (--fp4-native-mixed): the NVFP4 linears keep the NATIVE byte
#: layout on every rank and pick their kernel per rank (sm_120 W4A4 CUTLASS,
#: sm_86 Marlin W4A16 in place) -- docs/NVFP4_NATIVE_LAYOUT_CONTRACT.md.
FP4_NATIVE_MIXED_ARGV = ("--fp4-gemm-backend", "native-mixed")
_P_LOG_MODEL_RE = re.compile(r"server_args=ServerArgs\(model_path='([^']*)'")
EXPERT_MAP_DIR_ENV = "FLLIPER_PDFLIP_EXPERT_MAP_DIR"

def _pconst(name: str, profile: Optional[str] = None):
    """A measured constant of ``profile`` (default: the published PDFLIP-FORM's
    profile, else the launcher's default profile) -- pdflip_form.profile_constant."""
    return pdflip_form.profile_constant(name, profile)

def p_prefill_graph_calibration_of(ns):
    """The graph calibration table: --p-prefill-graph-calibration PATH, else the
    'graph_calibration' key of a --p-chunk-model JSON file, else None."""
    from flliper.srt.pdflip import p_graph_policy as _pgp

    path = str(getattr(ns, "p_prefill_graph_calibration", "") or "").strip()
    try:
        if path:
            return _pgp.load_calibration(path)
        src = str(getattr(ns, "p_chunk_model", "") or "").strip()
        if src and src.endswith(".json") and os.path.isfile(src):
            with open(src) as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("graph_calibration"), dict):
                return _pgp.Calibration.from_json(data["graph_calibration"])
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--p-prefill-graph-calibration {path or src}: {exc}")
    return None

def p_prefill_graph_captured() -> List[int]:
    """The buckets group P really captures (the policy's answer), ascending;
    [] = no graph (switch off, or 'auto' found the graph slower everywhere)."""
    return list(_P_PREFILL_GRAPH.get("captured", ()) or ())

def p_prefill_graph_policy_lines() -> List[str]:
    return list(_P_PREFILL_GRAPH.get("lines", ()) or ())

def p_prefill_graph_tiny_of(ns, bucket: int) -> Tuple[int, ...]:
    """--p-prefill-graph-tiny as extra, SMALLER capture buckets (27B line,
    24.09., default off = ()). Why: group P ends every prompt with the 1-token
    END-OF-PREFILL ANCHOR chunk (schedule_policy._pdflip_end_anchor_split,
    FLLIPER_PDFLIP_END_ANCHOR=1), and the runner pads every batch up to the
    smallest captured bucket that holds it (base_cuda_graph_runner.
    _pad_to_bucket) -- with bs [512] alone that one token replays a full
    512-row graph: measured xsn430/xsn433 #PGAP tokens=1 gpu_fwd 37.1 / 35.0 /
    38.0 ms (PP0/PP1/PP2) against ~41 / 34 / 38 ms for a real 512 chunk. A
    16-token bucket is weight-bandwidth bound instead (~12 / 7 / 7 ms
    estimated). Buckets must be positive and below the main bucket."""
    raw = str(getattr(ns, "p_prefill_graph_tiny", "") or "").strip()
    if not raw:
        return ()
    if not bucket:
        raise SystemExit("--p-prefill-graph-tiny needs --p-prefill-graph N (the main bucket)")
    try:
        vals = sorted({int(x) for x in raw.split(",") if x.strip()})
    except ValueError:
        raise SystemExit(f"--p-prefill-graph-tiny {raw!r}: comma list of token counts expected")
    bad = [v for v in vals if v <= 0 or v >= int(bucket)]
    if bad:
        raise SystemExit(
            f"--p-prefill-graph-tiny {raw!r}: every bucket must be in (0, {int(bucket)}), got {bad}"
        )
    return tuple(vals)

def p_prefill_graph_bucket() -> int:
    """The P prefill graph's token bucket, 0 when the switch is off."""
    return int(_P_PREFILL_GRAPH["bucket"])

def p_deep_split_env(ns) -> Dict[str, str]:
    """Group P's environment for --p-deep-split-from; {} when off."""
    from flliper.srt.layers.attention import fi_prefill_wave_split as _fiws

    n = int(getattr(ns, "p_deep_split_from", P_DEEP_SPLIT_FROM_DEFAULT) or 0)
    if n < 0:
        raise SystemExit(f"--p-deep-split-from must be >= 0, got {n}")
    return _fiws.launcher_env_p_deep_split(n)

def p_chunk_model_power_limits(src: str, stages: int) -> Tuple[Optional[List[Optional[float]]], str]:
    """(calibration power limit per P stage, where it came from) of a
    --p-chunk-model source; ``None`` = the model does not know its limit (an
    old JSON, or a fit log from before row 27): warned, never refused.

    builtin-int8: the weg2rc7c fit (25.09.) ran under the rig limits of
    24.09. (power_limit.RIG_POWER_LIMIT_W_0924, bracketing readings);
    fit:<P.log>: that log's own ``POWER-LIMIT rank`` lines (PP<r>);
    JSON: its ``power_limit_w`` field."""
    from flliper.srt.pdflip import power_limit as _pl

    src = str(src or P_CHUNK_MODEL_DEFAULT).strip()
    classes = list(P_STAGE_CARD_CLASSES[:stages]) + [P_STAGE_CARD_CLASSES[-1]] * max(0, stages - 3)
    if src in P_CHUNK_BUILTIN_SOURCES:
        # UNIFY S9: the record's own limit (int8: the rig limits of 24.09.; fp8: its own boot's POWER-LIMIT lines)
        record = P_CHUNK_BUILTIN_SOURCES[src][1]
        lim = record.power_limits() or {}
        return ([lim.get(c) for c in classes],
                f"builtin rig limits ({record.power_limit_source or 'record'})")
    if src.startswith("fit:"):
        path = src[4:]
        try:
            with open(path, "rb") as fh:
                found = _pl.parse_boot_lines(
                    raw.decode("utf-8", "replace") for raw in fh if b"POWER-LIMIT rank " in raw)
        except OSError:
            found = {}
        vals = [found.get(f"PP{r}", {}).get("power_limit_w") for r in range(stages)]
        if not any(v is not None for v in vals):
            return None, f"fit log {os.path.basename(path)} has no POWER-LIMIT rank line (predates row 27)"
        return [None if v is None else float(v) for v in vals], f"fit log {os.path.basename(path)} POWER-LIMIT lines"
    try:
        with open(src) as fh:
            data = json.load(fh)
        vals = _pl.stage_limits_from_json(data, stages)
    except (OSError, ValueError, _pl.PowerLimitError) as exc:
        return None, f"json:{os.path.basename(src)} power_limit_w unreadable ({exc})"
    if vals is None:
        return None, f"json:{os.path.basename(src)} has no power_limit_w (old JSON)"
    return vals, f"json:{os.path.basename(src)} power_limit_w"

def _p_chunk_interp(curve: Dict[int, float], m: int) -> float:
    """``curve`` in log2(M), flat beyond its ends (a missing point is no gain)."""
    keys = sorted(curve)
    if m <= keys[0]:
        return float(curve[keys[0]])
    if m >= keys[-1]:
        return float(curve[keys[-1]])
    for lo, hi in zip(keys, keys[1:]):
        if lo <= m <= hi:
            f = (math.log2(m) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
            return float(curve[lo]) + f * (float(curve[hi]) - float(curve[lo]))
    return 1.0

def p_chunk_policy_spec():
    """The installed PolicySpec, None under 'fixed'."""
    return _P_CHUNK.get("spec")

def p_chunk_model_cut(src: str) -> Tuple[Tuple[int, ...], str]:
    """(the P cut the --p-chunk-model table was measured/fitted on, where that
    came from); () = the table does not say (an older JSON, a fit log)."""
    from flliper.srt.pdflip import p_graph_policy as _pgp

    src = str(src or P_CHUNK_MODEL_DEFAULT).strip()
    if src in P_CHUNK_BUILTIN_SOURCES:
        return tuple(P_CHUNK_BUILTIN_SOURCES[src][1].pp_layer_ratio), "record P_CHUNK_BUILTIN"
    if src.startswith("fit:"):
        return (), f"fit:{os.path.basename(src[4:])} (the fit log's own cut, not stated)"
    try:
        with open(src) as fh:
            data = json.load(fh)
        return _pgp.parse_cut((data or {}).get("pp_layer_ratio")), f"json:{os.path.basename(src)}"
    except (OSError, ValueError, AttributeError, _pgp.GraphPolicyError) as exc:
        return (), f"json:{os.path.basename(src)} pp_layer_ratio unreadable ({exc})"

def p_chunk_power_lines() -> List[str]:
    """Row 27: the chunk model's power-limit check (one line per P stage, loud
    on a > 5 % mismatch, a warning for a model without power_limit_w) plus the
    SCALE line when --p-power-scale rescaled it. [] under 'fixed'."""
    return ["PDFLIP " + ln for ln in (_P_CHUNK.get("power_lines") or ())]

def _int_list(raw: str, what: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(raw).split(",") if x.strip() != "")
    except ValueError:
        raise SystemExit(f"{what} {raw!r}: expected comma-separated integers")

def p_layer_split_swing_by_stage(n_stages: int) -> Tuple[Tuple[int, int], ...]:
    """The planner's swing post: (attention, linear) swing layers per P stage;
    () under static or when the spec's stage count is not P's."""
    from flliper.srt.pdflip import p_layer_split as _pls

    spec = _P_LAYER_SPLIT.get("spec")
    if spec is None or spec.geometry.stages != int(n_stages):
        return ()
    g = spec.geometry
    out = []
    for s in range(g.stages):
        win = tuple(g.swing_window(s))
        na = sum(1 for i in win if g.families[i] == _pls.FAMILY_ATTENTION)
        out.append((na, len(win) - na))
    return tuple(out)

def p_layer_split_slab_lines(spec) -> List[str]:
    """The swing slab per stage as the planner's post (weights from the rc9
    checkpoint census, DYN_LAYER_SPLIT.md sec. 2; the rank-side sizer prices
    the mirrors exactly, pool_configurator swing_extra_layer_counts)."""
    from flliper.srt.pdflip import p_layer_split as _pls

    g = spec.geometry
    out = []
    for s in range(g.stages - 1):
        win = tuple(g.swing_window(s))
        na = sum(1 for i in win if g.families[i] == _pls.FAMILY_ATTENTION)
        out.append(f"{_pls.LOG_TAG} slab stage={s} window={list(win)} attention={na} gdn={len(win) - na} "
                   f"(KV mirror = {na} x pool x 2048 B/token, state mirror = {len(win) - na} x mamba slots "
                   f"x 1.56 MiB, weights resident under tag weights_swing)")
    return out

def p_layer_split_env() -> Dict[str, str]:
    """Group P's environment for the layer split; {} under 'static'."""
    from flliper.srt.pdflip import p_layer_split as _pls

    spec = _P_LAYER_SPLIT.get("spec")
    if spec is None:
        return {}
    return {_pls.POLICY_ENV: _pls.POLICY_DYNAMIC, _pls.SPEC_ENV: spec.to_json()}

def p_gc_env(ns) -> Dict[str, str]:
    """Group P's environment for --p-gc-freeze; {} under 'off'."""
    from flliper.srt.pdflip import gc_instrument as _gci

    if str(getattr(ns, "p_gc_freeze", P_GC_FREEZE_DEFAULT) or P_GC_FREEZE_DEFAULT) != "on":
        return {}
    return {_gci.FREEZE_ENV: "1"}

def p_pool_posts_as_booked(ns) -> bool:
    """PP-POSTEN (27B, 29.09.): does the profile's registry row price group P's pool with the posts the
    P runtime BOOKS (``ModelProfile.p_pool_posts_as_booked``)? Then the corridor holdback is 0 (the P
    budget line already subtracts corridor = reserve + awake overshoot; every P rank logs 'KV budget
    holdback ... holdback=0.0') and the activation reserve is 0 unless pinned (the runtime books P's
    prefill transient only for a checkpoint with measured slopes, p_prefill_transient_for; the 27B has
    none -- boots 09291750/09291831/09292006: 'mamba state pool + speculative intermediate state + prefill
    activation reserve' = 37.41 MiB x linear layers, the mamba pool alone). Off = byte-identical."""
    row = pdflip_form.profile_row(getattr(ns, "profile", None))
    return bool(row is not None and getattr(row, "p_pool_posts_as_booked", False))

def kv_stage_wave_cap(ns, er, tab, max_by: Sequence[int], env: Mapping[str, str],
                      head: str) -> Tuple[Optional[int], List[str]]:
    """#251d: the overflow-wave cap the captured decode steps need under the
    highest stage per seat count ``max_by``. A form that lowers a batch's
    capture floor adds waves (``tab.extra_waves``); the step needs
    max(``tab.capture_waves``) of them, and below that its own bound refuses
    the capture on TP0 ('Step ids exceed the LRU rows plus the staging rows')
    -- rc12z22's FLLIPER_OPT_MOE_POOL_OVERFLOW_WAVES=2 against 3 for S1 at
    n=5/6. The launcher's own cap (derived, or not written at all) is raised;
    a cap told in --env-d is refused by name (W169). (None, []) = nothing to
    change: no extra wave, or the cap already carries it (the default table
    adds none)."""
    if not any(int(e) > 0 for e in tab.extra_waves(max_by)):
        return None, []
    need = max(tab.capture_waves(max_by))
    have = er.pool_overflow_waves(env)
    if need <= have:
        return None, []
    if er.POOL_OVERFLOW_WAVES_ENV in env and not getattr(ns, "d_pool_waves_derived", False):
        raise PdFlipDKvStageWavesRefused(
            "W169 PdFlipDKvStageWavesRefused: the stage form (highest stage per seat count %s) "
            "requires %d Ueberlaufwellen je bs %s, --env-d nennt %s=%s -- der Capture "
            "verweigerte auf TP0 ('Step ids exceed ...'); keine Stufenform geschrieben"
            % (list(max_by), need, list(tab.capture_waves(max_by)),
               er.POOL_OVERFLOW_WAVES_ENV, env.get(er.POOL_OVERFLOW_WAVES_ENV)))
    return need, ["%s: WELLEN (#251d) %s %d -> %d (Launcher-Ableitung): die Stufenform senkt "
                  "den Capture-Floor, Wellen je bs %s statt %s -- ohne die Anhebung verweigert "
                  "der Capture auf TP0" % (head, er.POOL_OVERFLOW_WAVES_ENV, have, need,
                                          list(tab.capture_waves(max_by)), list(tab.waves))]

def kv_stage_lru_floor(group, rows, fits, max_by: Sequence[int], peak: Sequence[int],
                       span: Sequence[int], source: str, head: str
                       ) -> Tuple[Dict[int, int], List[str]]:
    """(3) 29.09.: the measured peak pool demand as the LRU floor of the stage
    solve. A KV rank's stage rows are LRU rows in S0; a wake to S_j unmaps
    ``stage_rows[j]`` of them. The wave rule prices the step at
    min(b x verify x top_k, E - R) with waves -- the MEASURED peak of
    non-resident ids per step (MOE-POOL-DEMAND, rc12z30r3: TP2 43 of C 48)
    is the demand the extra wave hits. Floor per KV rank: the fewest rows any
    allowed (n, stage) of the written form leaves (``capacity[n][max_by[n]]``)
    >= the peak, scaled from the measured span E - R to this form's
    (first order: the non-resident ids of a step grow with the non-resident
    span). Short by d rows -> the scratch grows by ceil(d / (1 - p)) with
    p = peak / span: a row moved from resident to scratch widens the span by
    one row, so the need grows by p per row. Returns ({rank: rows to add},
    lines); an empty dict when every KV rank holds its peak."""
    last = rows[-1]
    fit_by = {int(f.rank): f for f in fits}
    cap = len(rows)
    add: Dict[int, int] = {}
    parts: List[str] = []
    for t in group.tables:
        r = int(t.host_rank)
        if r >= len(peak) or r >= len(last.scratch_given):
            continue
        f = fit_by.get(r)
        E = int(f.local_experts) if f is not None else 0
        C = int(last.scratch_given[r])
        span_now = max(E - (int(last.max_rows[r]) - C), 0)
        p = min(float(peak[r]) / int(span[r]), 1.0) if int(span[r]) > 0 else 1.0
        need = min(int(math.ceil(p * span_now - 1e-9)), span_now, int(last.ids_per_step))
        j_by = [min(int(max_by[i]), len(t.tokens) - 1) for i in range(cap)]
        lru = min(int(t.capacity[i][j_by[i]]) for i in range(cap))
        text = ("rang%d Peak %d bei Spanne %d -> Bedarf %d bei Spanne %d (E %d - R %d), "
                "LRU bis S%d min %d" % (r, int(peak[r]), int(span[r]), need, span_now, E,
                                        E - span_now, max(j_by), lru))
        if need > lru:
            k = int(math.ceil((need - lru) / max(1.0 - p, 0.05) - 1e-9))
            add[r] = k
            text += " < %d -> Scratch +%d (Bedarf +%.3f je Zeile)" % (need, k, p)
        else:
            text += " >= %d, haelt" % need
        parts.append(text)
    return add, ["%s: %s (%s aus %s): %s" % (
        head, D_KV_STAGE_LRU_FLOOR_MARKER, "D_POOL_PEAK_NONRES_ROWS", source or "Record",
        "; ".join(parts) or "kein KV-Rang")]

def operator_kv_stage_max(raw: Optional[str], cap: int, stages: int) -> Optional[Tuple[int, ...]]:
    """The operator's highest stage per seat count, or None when --env-d does
    not name it. Refuses by name what the table cannot take."""
    if raw is None or not str(raw).strip():
        return None
    try:
        vals = tuple(int(x) for x in str(raw).split(",") if x.strip())
    except ValueError:
        vals = ()
    if len(vals) != int(cap) or any(v < 0 or v >= int(stages) for v in vals):
        raise PdFlipDKvStageMaxRefused(
            "W168 PdFlipDKvStageMaxRefused: --env-d %s=%s passt nicht zur Stufentabelle "
            "(%d Sitzzahlen, Stufen 0..%d) -- keine Stufenform geschrieben"
            % (D_KV_STAGE_KEYS[2], raw, int(cap), int(stages) - 1))
    return vals

def kv_stage_cut_workers(kv_token_shares, host_rank: int = 0) -> List[int]:
    """#239 S3g (28.09.): the D ranks other than the attention host that hold
    full-attention KV under the token cut -- every worker with a share > 0,
    and (a cut the planner still solves: 'owned', 'maxmin', 'joint') every
    worker, since any of them may get one. [] without a cut."""
    if kv_token_shares is None:
        return []
    if isinstance(kv_token_shares, str):
        vec = None
        if kv_token_shares.startswith("owned:"):
            try:
                vec = [float(x) for x in kv_token_shares[len("owned:"):].split(",") if x.strip()]
            except ValueError:
                vec = None
        if vec is None:
            return [-1]  # unsolved: some worker, not yet known which
    else:
        vec = [float(x) for x in kv_token_shares]
    return [r for r, s in enumerate(vec) if r != int(host_rank) and float(s) > 0.0]

def hicache_disabled_line() -> str:
    """#1386: the one line the launcher prints when --pdflip-disable-hicache is set.

    Same reason as :func:`draft_kv_off_line`: an A-B arm that reads like the
    default is how a measurement gets attributed to the wrong tree. This says,
    in the boot's own argv-adjacent log, that BOTH groups' argv omit every
    HiCache flag and the ledger charged 0.00 for it -- the ledger's own
    ``PDFLIP-HICACHE-DISABLED`` line (host_ledger.choose) is the other half of
    this same fact, printed at the other end of the one bool.
    """
    return (
        "PDFLIP HICACHE-DISABLED: on -- both groups boot without "
        "--enable-hierarchical-cache and every --hicache-* flag (Minimalform "
        "order #1386: 0 requests served means 0 prefixes to reuse); the host "
        "ledger prices anchors_gib/rings_gib/overhead_gib at 0.00 for the "
        "identical reason (see PDFLIP-HICACHE-DISABLED in this boot's ARM lines)"
    )

def _median(xs: List[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])

def floor_source_for_shipped_line(decision) -> str:
    """The ``pool_floor_source=`` value :func:`shipped_line` prints."""
    src = str(getattr(decision, "pool_floor_source", "none") or "none")
    cut = getattr(decision, "pool_floor_cut", None)
    if src == "default-from-ordered-cut" and cut is not None:
        return (
            "default-from-ordered-cut %s (user order 2026-09-09: '39,13,12 mit bs2 "
            "im pp layout soll standard werden vorerst')"
            % (",".join(str(n) for n in cut),)
        )
    if src == "flag":
        return "flag (--pp-solve-pool-floor, operator override)"
    return "none (--pp-solve-pool-floor 0, operator override: OFF)"

def _live_stage_support(base):
    """AP2 1006: the builtin transient support (measured per P stage on the
    3-stage reference rig) for the P stage count of THIS launch. Only while an
    inventory view is installed (``apply_inventory_derivation``: a subset of the
    calibrated cards, or a boot forced past HW-UNCALIBRATED) whose card count --
    P = PP<N> -- differs from the support's: each live stage takes the measured
    points of the reference stage of the same ROLE (``p_card_chunk.role_stage_map``:
    first, middle, last). Everywhere else (the reference rig, desk callers
    without a view) ``base`` comes back unchanged."""
    from flliper.srt.pdflip import inventory_view as _iv

    act = _iv.active()
    if act is None or not base.points or len(act[2]) == base.n_stages:
        return base
    smap = _p_card.role_stage_map(len(act[2]), base.n_stages)
    if smap is None:
        return base
    tag = " RESTAGE %d->%d Stufen %s" % (base.n_stages, len(act[2]), ",".join(str(m) for m in smap))
    return _p_card.TransientSupport(model=base.model, points=tuple(
        _p_card.TransientPoint(chunk=p.chunk, mib=tuple(p.mib[m] for m in smap), source=p.source + tag)
        for p in base.points))

def note_calibration_footprint_alias(model_path: str) -> Optional[str]:
    """H87: record whether ``model_path`` is a footprint-identical derivative
    of a #114/H41 calibration checkpoint; returns the printed line (or None)."""
    name = pdflip_form.model_key(model_path)
    _CALIBRATION_FOOTPRINT_ALIAS.pop(name, None)
    if not name or name in P_PREFILL_TRANSIENT_CALIBRATION_MODELS:
        return None
    for ref in P_PREFILL_TRANSIENT_CALIBRATION_MODELS:
        ok, why = pdflip_form.reference_model_verdict(model_path, ref)
        if ok:
            _CALIBRATION_FOOTPRINT_ALIAS[name] = (ref, why)
            return (f"H87 CALIBRATION-IDENTITY {name} -> {ref} by memory footprint "
                    f"({why}): the #114/H41 prefill transient measured on {ref} applies")
    return None

def _p_chunk_nf_args(ns, argv_words: Sequence[str]) -> Dict[str, object]:
    """The NF reading of the shared --p-chunk-* flags (see above)."""
    def given(flag: str) -> bool:
        return pdflip_form.flag_given(list(argv_words or ()), flag)

    return {
        "model": (str(getattr(ns, "p_chunk_model", "") or P_CHUNK_NF_MODEL_DEFAULT)
                  if given("--p-chunk-model") else P_CHUNK_NF_MODEL_DEFAULT),
        "max": int(getattr(ns, "p_chunk_max", 0) or 0) if given("--p-chunk-max") else 0,
        "min": int(getattr(ns, "p_chunk_min", 0) or 0),
        "fixed": int(getattr(ns, "p_chunk_fixed", 0) or 0),
        "raster": int(getattr(ns, "p_chunk_raster", 0) or 0),
        "min_gain": getattr(ns, "p_chunk_min_gain", None),
        "dynamic_min": (int(getattr(ns, "p_chunk_dynamic_min_tokens", -1))
                        if given("--p-chunk-dynamic-min-tokens") else -1),
    }

#: FOLLOWS THE RESERVE SINCE #1257c. The runtime post this stands for
#: (``model_runner_kv_cache_mixin._gapped_corridor_holdback``) IS the user
#: reserve, and the reserve now defaults to 0 -- so on a default boot the
#: runtime charges ``gapped corridor holdback=0.000`` and a pool model still
#: carrying 1024.0 over-charges group P by a gibibyte PER RANK. The measured
#: 1024.0 above is what BOTH reference boots emitted, and it is exactly what
#: this expression still returns whenever the reserve is 1024. Derived, never
#: re-typed: ``--pp-cut-corridor-holdback-mib`` takes it as its default and
#: the operator can still pin it.
def p_corridor_holdback_default_mib(user_reserve_mib=None) -> float:
    """The 'gapped corridor holdback' post the boot will actually charge."""
    from flliper.srt.mem_ledger.terms import DEFAULT_USER_RESERVE_MIB

    if user_reserve_mib is None:
        user_reserve_mib = DEFAULT_USER_RESERVE_MIB
    return float(max(0, int(user_reserve_mib)))

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def parse_card_selection(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    """``"1,0"`` -> ``(1, 0)``; empty/None -> None (every card). Refuses BY
    NAME (HW-CARDS) a malformed entry or a duplicate index."""
    t = str(text or "").strip()
    if not t:
        return None
    out: List[int] = []
    for part in t.split(","):
        p = part.strip()
        if not p.isdigit():
            raise PdFlipLaunchRefused(
                f"{CODE_CARDS}: --cards {t!r}: {p!r} is not an NVML index (expected e.g. '1,0')")
        if int(p) in out:
            raise PdFlipLaunchRefused(f"{CODE_CARDS}: --cards {t!r} names NVML index {p} twice")
        out.append(int(p))
    return tuple(out)

def _forced_record_borrow(prof: str, cal: Sequence[str], live: Sequence[str],
                          cards: Sequence["Card"]) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str]]:
    """AP2 1006: the positional records of ``prof`` that :func:`inventory_view.borrow_vector`
    can place on the live cards (name -> (old text, borrowed text)) and the arch twins it
    used (live class label -> calibrated class, AP1 ``card_identity.arch_twin_class``).
    Only for a boot forced past HW-UNCALIBRATED; a record with no borrow is not listed."""
    from flliper.srt.pdflip import inventory_view as _iv
    from flliper.srt.pdflip import profile_records as _pr

    twins: Dict[str, str] = {}
    for c in cards:
        label = card_identity.class_label(c)
        if label not in cal:
            twin = card_identity.calibration_class(c) or card_identity.arch_twin_class(c)
            if twin:
                twins[label] = twin
    out: Dict[str, Tuple[str, str]] = {}
    for r in _pr.records(prof):
        if not _pr.is_positional(r.value, len(cal)):
            continue
        pol = _iv.RECORD_POLICY.get(r.name)
        if pol in (_iv.CONSUMER, _iv.CUT_GATED, _iv.ADVISORY, _iv.CUT_PIN):
            continue
        got = _iv.borrow_vector(pol, r.value, tuple(cal), tuple(live), twins)
        if got is None:
            continue

        def _t(v):
            return v if isinstance(v, str) else ",".join(str(x) for x in v)

        out[r.name] = (_t(r.value), _t(got))
    return out, twins

def order_cards(cards: List[Card], expect_count: Optional[int] = None) -> List[Card]:
    """CUDA ordinal order (rank 0 / PP0 / TP0 first), from NVML PROPERTIES
    (pdflip/card_identity.py): biggest NVML total first, then nameplate DRAM
    bandwidth, then NVML index. On the reference rig that is the 5090 first,
    then the 3080s by NVML index -- the order the name-based predecessor
    produced, byte for byte. Never a fixed index, never a name substring.

    HW-P1a 1003: no count gate by default -- the count is the inventory's and
    :func:`topology_check_line` refuses an unrunnable one BY NAME with its
    blockers. ``expect_count`` (a caller that needs exactly N) still refuses
    HW-COUNT. The arch gate (HW-ARCH) runs once in :func:`resolve_cards`."""
    try:
        return card_identity.order_cards(cards, expect_count, gate=False)
    except card_identity.CardInventoryRefused as exc:
        # PROFIL-EDITOR S1: --force lifts the COUNT refusal (a value refusal); the arch gate does not run here
        refusals.refuse_value("HW-COUNT", str(exc), PdFlipLaunchRefused, cause=exc)
        return card_identity.order_cards(cards, None, gate=False)

def log_power_limits(state: BootState, cards: List[Card], log, *,
                     reading: Optional["_power.PowerReading"] = None) -> "_power.PowerReading":
    """fnFL2 H57 (Nutzer 24.09. 18:53Z: "... im powerlimit bei 400 und 230 ...
    da ich spaeter das powerlimit ggf. erhoehen werde").

    Druckt ``POWER-LIMIT nvml0=230/320W nvml1=400/600W nvml2=230/320W
    sm_clock_max=...`` (NVML, nur lesend), legt dasselbe in ``state.power_limits``
    (-> ``boot_<TAG>.json``) und datiert jede Zeit-/Raten-Referenz des Planers
    dagegen: weicht ihr Limit ab, ``REFERENZ VERALTET (Power-Limit ...)`` mit
    Namen. VERWEIGERT NIE -- die VRAM-Riegel (W122/W130/W132/W126) lesen die
    Lesung nicht, und ein unlesbares NVML macht die Referenzen ``undatiert``,
    nicht den Boot unmoeglich. ``cards`` in CUDA-Ordinal-Reihenfolge (die
    Kartenraten-Bibliothek wird je Stufe gestempelt, wie der Solver sie liest).
    """
    reading = _power.read_card_power() if reading is None else reading
    state.power_limits = _power.state_dict(reading)
    log(_power.launch_line(reading))
    library_ref = _power.card_library_reference(reading, cards)
    for line in _power.reference_lines(
            reading, _power.TIMED_REFERENCES + ((library_ref,) if library_ref else ())):
        log(line)
    return reading

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
        raise PdFlipLaunchRefused(
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
    # NS 26.09.: the same row parser as the front's flip reading -- a row that is
    # not 'pid, used, uuid' (e.g. "No running processes found", "[N/A]") is
    # skipped and counted, never an unpack error (boot dkr27bbar1i8h109261950).
    from flliper.srt.pdflip.front import parse_compute_apps

    return parse_compute_apps(out, pids)

def is_launch_server_argv(argv: Sequence[str]) -> bool:
    """True only for ``<python...> [opts] -m flliper.launch_server ...``.

    argv[0]'s basename must start with ``python`` and the module must be a
    WHOLE argument after ``-m`` -- never a piece of a longer string.  A shell
    whose ``-c`` script merely MENTIONS the module (``bash -c 'pgrep -fc
    "flliper.launch_server"; bash start_when_free.sh ...'``) is not a server,
    and neither is ``python -m flliper.srt.pdflip.launcher``.
    """
    if not argv or not os.path.basename(argv[0]).startswith("python"):
        return False
    # rename transition: either generation's module (LAUNCH_SERVER_MODULES)
    for i, a in enumerate(argv[1:], start=1):
        if a == "-m" and i + 1 < len(argv) and argv[i + 1] in LAUNCH_SERVER_MODULES:
            return True
        if a.startswith("-m") and a[2:] in LAUNCH_SERVER_MODULES:
            return True
    return False

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

def fs_of(path: str, mounts_text: Optional[str] = None) -> Tuple[str, str, str]:
    """W57: (mount point, fs type, source) of the filesystem holding ``path``
    -- the longest /proc/mounts mount point above its nearest existing
    ancestor. ("", "", "") when /proc/mounts is unreadable."""
    p = os.path.abspath(str(path or "/"))
    while p != "/" and not os.path.exists(p):
        p = os.path.dirname(p)
    real = os.path.realpath(p)
    if mounts_text is None:
        try:
            with open("/proc/mounts") as fh:
                mounts_text = fh.read()
        except OSError:
            return "", "", ""

    def _unesc(v: str) -> str:
        return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), v)

    best = ("", "", "")
    for ln in mounts_text.splitlines():
        parts = ln.split()
        if len(parts) < 3:
            continue
        src, mnt, typ = _unesc(parts[0]), _unesc(parts[1]), parts[2]
        inside = real == mnt or real.startswith(mnt.rstrip("/") + "/") or mnt == "/"
        if inside and len(mnt) >= len(best[0]):
            best = (mnt, typ, src)
    return best

def l3_persist_enabled(env=None) -> bool:
    """L3P: ``FLLIPER_PDFLIP_L3_PERSIST`` (default on; 0 = the per-boot store,
    swept at the next boot, exactly as before)."""
    e = os.environ if env is None else env
    raw = (e.get("FLLIPER_PDFLIP_L3_PERSIST", "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")

def _l3_extra_flag(extra: str, flag: str) -> str:
    """The value argparse will keep for ``flag`` in a shell-split EXTRA string
    (the LAST occurrence), or ""."""
    try:
        toks = shlex.split(extra or "")
    except ValueError:
        return str(extra or "")
    val = ""
    for i, t in enumerate(toks):
        if t == flag and i + 1 < len(toks):
            val = toks[i + 1]
        elif t.startswith(flag + "="):
            val = t[len(flag) + 1:]
    return val

def l3_weights_fingerprint(model: str) -> str:
    """L3P N4: sha1 over (name, size, mtime_ns) of the checkpoint's weight
    files -- a checkpoint swapped IN PLACE at the same path is a new identity.
    Top level of the model directory (or the file itself for a GGUF path);
    stat only, nothing is read."""
    real = os.path.realpath(str(model or ""))
    rows = []
    try:
        if os.path.isfile(real):
            st = os.stat(real)
            rows.append((os.path.basename(real), st.st_size, st.st_mtime_ns))
        else:
            with os.scandir(real) as it:
                for e in it:
                    if e.name.endswith((".safetensors", ".gguf", ".bin", ".pt", ".pth")) and e.is_file():
                        st = e.stat()
                        rows.append((e.name, st.st_size, st.st_mtime_ns))
    except OSError:
        return ""
    rows.sort()
    return hashlib.sha1(repr(rows).encode()).hexdigest()

def l3_persistent_store(directory: str) -> bool:
    """L3P: ``directory`` is a persistent L3 store (its ``l3-`` name or its
    identity file) -- no teardown and no sweep may remove it."""
    d = str(directory or "").rstrip("/")
    return bool(d) and (
        os.path.basename(d).startswith(L3_PERSIST_PREFIX)
        or os.path.isfile(os.path.join(d, L3_IDENTITY_FILE))
    )

def l3_persist_dir_name(identity: dict) -> str:
    """L3P: ``l3-<profile>-<model>-<identity10>``; the readable part names the
    model, the digest separates everything else the identity carries."""
    base = os.path.basename(str(identity.get("model_path", "")).rstrip("/")) or "model"
    label = "-".join(x for x in (identity.get("profile") or "", base) if x)
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", label)[:80]
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    return f"{L3_PERSIST_PREFIX}{label}-{digest}"

def l3_persist_check_identity(directory: str, identity: dict, dry: bool) -> str:
    """L3P: the directory's own identity file must match; returns "new" /
    "match". A mismatch is REFUSED BY NAME -- a store is never shared by two
    identities, whatever its name says."""
    path = os.path.join(directory, L3_IDENTITY_FILE)
    try:
        with open(path) as f:
            have = json.load(f)
    except FileNotFoundError:
        have = None
    except (OSError, ValueError) as exc:
        raise PdFlipStoreDiskRefused(
            f"W57 PdFlipStoreDiskRefused: L3-PERSIST identity file {path} unreadable "
            f"({type(exc).__name__}: {exc}); refusing to reuse a store whose owner "
            f"cannot be proven. Move the directory aside or set FLLIPER_PDFLIP_L3_PERSIST=0.")
    if have is None:
        if not dry:
            os.makedirs(directory, exist_ok=True)
            tmp = path + ".w"
            with open(tmp, "w") as f:
                json.dump(identity, f, sort_keys=True, indent=1)
            os.replace(tmp, path)
        return "new"
    if have != identity:
        diff = sorted(k for k in set(have) | set(identity) if have.get(k) != identity.get(k))
        raise PdFlipStoreDiskRefused(
            f"W57 PdFlipStoreDiskRefused: L3-PERSIST store {directory} belongs to another "
            f"identity (differs in {diff}: store {[have.get(k) for k in diff]} vs this boot "
            f"{[identity.get(k) for k in diff]}). Two models never share an L3 store "
            f"(user 2026-09-27). Move the directory aside or set FLLIPER_PDFLIP_L3_PERSIST=0.")
    return "match"

def _l3_revoked_path(directory: str) -> str:
    return os.path.join(directory, L3_REVOKED_FILE)

def _l3_rope_view(override: str) -> str:
    """The rope/position part of a ``--json-model-override-args`` value
    (``rope_scaling``/``rope_parameters``/``max_position_embeddings``, top
    level and ``text_config``), canonical JSON; "" without one."""
    if not override:
        return ""
    try:
        args = json.loads(override)
    except (TypeError, ValueError):
        return override  # unparseable: compared verbatim
    if not isinstance(args, dict):
        return ""
    keys = ("rope_scaling", "rope_parameters", "max_position_embeddings")
    view = {k: args[k] for k in keys if k in args}
    txt = args.get("text_config")
    if isinstance(txt, dict):
        sub = {k: txt[k] for k in keys if k in txt}
        if sub:
            view["text_config"] = sub
    return json.dumps(view, sort_keys=True) if view else ""

def _l3_move_revoked(directory: str, root: str, name: str, dry: bool) -> bool:
    """Move one revoked page into ``<directory>.revoked`` (same relative path)."""
    if dry:
        return True
    src = os.path.join(root, name)
    rel = os.path.relpath(src, directory)
    dst = os.path.join(directory.rstrip("/") + L3_REVOKED_SUFFIX, rel)
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        return True
    except OSError:
        return False

def l3_registered_suffixes(directory: str) -> Optional[Tuple[str, ...]]:
    """L3P N1: the union of every group's scan suffixes the ranks recorded in
    ``<dir>/L3_SUFFIXES.<group>.json`` (written by HiCacheFile at attach), or
    None when no group has recorded yet (first boot of this store)."""
    out: List[str] = []
    found = False
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    for n in names:
        if not (n.startswith(L3_SUFFIX_REGISTRY + ".") and n.endswith(".json")):
            continue
        try:
            with open(os.path.join(directory, n)) as f:
                out.extend(str(x) for x in json.load(f).get("suffixes", []))
            found = True
        except (OSError, ValueError, AttributeError):
            continue
    return tuple(dict.fromkeys(x for x in out if x)) if found else None

def _last_flag_value(argv: Sequence[str], flag: str) -> Optional[str]:
    """The value argparse KEEPS for ``flag`` in a token list: the LAST one,
    in either spelling (``--flag v`` / ``--flag=v``). ``None`` when absent.

    The list twin of :func:`_argv_scalar` (which takes the --extra-* STRING):
    ``argv_p`` holds ``extra`` already split, and the finished argv is a list.
    """
    found = None
    toks = list(argv or ())
    for i, tok in enumerate(toks):
        tok = str(tok)
        if tok == flag and i + 1 < len(toks):
            found = str(toks[i + 1])
        elif tok.startswith(flag + "="):
            found = tok[len(flag) + 1:]
    return found

def _import_duplex_gate() -> float:
    from flliper.srt.managers import pdflip_memory_saver

    return float(pdflip_memory_saver.DUPLEX_SPLIT_MIN_RATIO)

def _front_leg_form() -> str:
    """``front.FLIP_LEG_FORM`` -- read, never restated.

    The launch check has to know how the flip orders its legs, and the only
    honest source is the module that does the ordering.  A copy here would be a
    second bookkeeping of the same fact and would go stale the moment C9 lands.
    """
    from flliper.srt.pdflip import front

    return front.FLIP_LEG_FORM

def pdflip_boot_token(ns) -> str:
    """#1348/#1395: ONE boot identity, shared by BOTH groups -- the single
    producer :func:`lane_coverage_boot_token` reuses rather than a second
    ``time.time()`` computation.

    Memoised on the namespace because `_env_knobs` is called once per group
    and the two groups must carry the SAME token -- a per-call `time.time()`
    would give P and D different identities and any reader keying on it
    would see two boots instead of one.

    UNCONDITIONAL AND SAFE TO ALWAYS PUBLISH -- unlike
    `FLLIPER_PDFLIP_LANE_COVERAGE_TOKEN` (see `lane_coverage_boot_token`'s own
    docstring: that variable's mere PRESENCE arms `coverage.py` line tracing
    elsewhere, 2x-10x on the traced modules, so it stays a gated FLAG output
    on purpose), this token decides nothing and arms nothing by existing --
    it is read-only identity, the same shape #1292's own dump directory
    reuse problem needs regardless of which instrument is armed. #1395
    reads it in `mem_ledger.activation_probe.boot_token()` via
    `FLLIPER_PDFLIP_BOOT_TOKEN` (below) for exactly the sibling defect
    `lane_coverage_boot_token`'s own docstring already names: "the dump
    directory is #1292's and #1292's is reused across boots".
    """
    tok = getattr(ns, "_pdflip_boot_token", "")
    if not tok:
        tok = f"{getattr(ns, 'tag', 'notag')}:{int(time.time())}:{os.getpid()}"
        setattr(ns, "_pdflip_boot_token", tok)
    return tok

def parse_group_env(spec: str) -> Dict[str, str]:
    """'KEY=VAL;KEY=VAL' -> dict; empty -> {}. Entries are ';'-separated because
    the values themselves are comma vectors (FLLIPER_MOE_SCRATCH_SLOTS=74,48,48).
    A malformed entry is refused by name rather than dropped (#781 class)."""
    out: Dict[str, str] = {}
    for item in [x for x in str(spec or "").split(";") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"--env-p/--env-d entry {item!r} is not KEY=VAL")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    # Rename 1b: an arm may still spell a name the old (or already the new)
    # way; callers read the dict by the tree's own name.
    return canonical_env(out)

def _gib(v) -> str:
    """#1361: bytes as GiB, or ``unreadable`` -- never 0.00 for an absence."""
    return "unreadable" if v is None else f"{int(v) / host_ledger.GIB:.2f} GiB"

def pdflip_weights_cpu_backup_ring_kw(
    ring_bytes: int, ring_span1_bytes: int,
    ring_absent_by_design_base: bool,
    weights_cpu_backup_armed: bool,
) -> Tuple[int, int, bool]:
    """``(ring_bytes, ring_span1_bytes, ring_absent_by_design)`` for the ledger.

    #1369: ONE function for BOTH halves of the same fact, because
    ``host_ledger.price``'s own contradiction guard REFUSES
    ``ring_absent_by_design=True`` beside a non-zero ring (boot weg2xsn13's
    own lesson, see :func:`ring_absent_by_design`'s docstring) -- a caller
    that ORs the DECLARATION into ``True`` without also zeroing the BYTES
    would turn "the ring is gone" into a boot-time crash instead of a price.
    Kept here, one call, rather than at each of the two former call sites,
    so the two halves cannot drift apart the way #1358's slot size/count
    once did.

    ``weights_cpu_backup_armed`` False -- the operator's explicit
    ``--pdflip-weights-cpu-backup off``, OR ``auto`` under an armed,
    authoritative exchange -- zeroes both byte counts and forces the
    declaration True, REGARDLESS of ``ring_absent_by_design_base`` (the
    pre-#1369 predicate: today only true under exchange+authoritative).
    True passes every input through unchanged -- byte-identical to every
    boot before this flag existed.
    """
    if weights_cpu_backup_armed:
        return (
            int(ring_bytes), int(ring_span1_bytes),
            bool(ring_absent_by_design_base),
        )
    return 0, 0, True

def group_rank_count(argv: Sequence[str]) -> int:
    """The rank count of a group, from the argv the launcher built for it:
    ``--tp-size`` x ``--pp-size`` (last occurrence wins, as in the server).

    27B NVFP4 duo 23.10.: the W7/W10 canonical gate was called with a literal
    3 (the reference rig's card count); on a 2-card launch P is TP1 x PP2 and
    D is TP2, so a clean group was refused as "group reports 1x2 ranks,
    launcher expects 3" (boot dkr27bnvfp4duobar1duo10230518)."""
    from flliper.srt.pdflip import carrier_census as _cc
    tp = _cc.tp_size_of(argv, default=1) or 1
    pp = 1
    items = list(argv)
    for i, a in enumerate(items):
        v = None
        if a == "--pp-size" and i + 1 < len(items):
            v = items[i + 1]
        elif a.startswith("--pp-size="):
            v = a.split("=", 1)[1]
        if v is not None:
            try:
                pp = int(v)
            except ValueError:
                pass
    return tp * pp

def group_form_block(env) -> Optional[dict]:
    """DASHBOARD-AUS-IPC (e): the boot form this group was handed (the one
    :data:`pdflip_form.FORM_ENV` every rank reads), as ``groups.<G>.form`` -- the
    rigdash's start form then comes from state.json, not from the log's
    PDFLIP-FORM line. ``None`` = no pdflip form published (teardown, desk test)."""
    f = pdflip_form.current_form(env)
    if f is None:
        return None
    return {**f.axes(), "profile": f.profile, "model": f.model, "describe": f.describe()}

def front_launch_fields(argv: Sequence[str], env: Dict[str, str]) -> dict:
    """Nutzer 29.09. (rigdash „komplette Startflags inkl. ENV“): state.json
    launch.front = das exec'te argv der Front (redigiert wie im Log) + ihre Umgebung."""
    return {"launch.front": state_file_mod.launch_snapshot(admin_key_mod.redact_argv(argv), env)}

def memts_owner_pattern(tag: str) -> str:
    """argv pattern of the processes whose life the logger is bound to: the
    launcher (until it returns after LAUNCHED) and the front (from before the
    launcher returns until the arm's teardown). Escaped, so the supervisor's
    own argv -- which carries this string -- never matches it."""
    return rf"flliper\.srt\.pdflip\.(launcher|front) .*--tag {re.escape(tag)}( |$)"

def flip_weights_tags(ns, family: List[str]) -> List[str]:
    """The weights tags the launcher sleeps/resumes: the family, or NOTHING
    under ``--flip-weights resident`` (fnFL2 v7, 20.09.: the post-READY
    sleep(P) named weights_0..7 + weights, the server's W4 guard refused a
    weights tag on a resident boot, three ranks disagreed -> W29 crash).
    The front derives its own empty list from ``--weights-resident``; this
    is the launcher half of the same rule."""
    if getattr(ns, "flip_weights", "family") == "resident":
        return []
    return list(family)

def _group_env_value(name: str, group_envs, default: str) -> str:
    """29.09.: ``name`` as the RANKS see it -- the group env (--env-p/--env-d,
    applied over the launcher env by build_env) wins; when both groups set it
    differently the larger one prices (the ONE arena is shared). The launcher
    env only when no group sets it (ENV-LAUNCHER != RANG)."""
    vals = [str(e[name]) for e in (group_envs or ()) if e and str(e.get(name, "") or "")]
    if vals:
        return max(vals, key=float)
    return str(os.environ.get(name, default) or default)

def model_config_path(model: str) -> str:
    """The ``config.json`` that describes ``model``, by the SERVER's own rule
    (``server_args.declared_config_path_for``, the body of
    ``ServerArgs.declared_config_path``) -- never a second copy of it.

    27B line G2 (2026-09-25): a GGUF launch names the ``.gguf`` FILE, whose
    config is the SIBLING ``config.json``; ``os.path.join(model, "config.json")``
    found ``<...>.gguf/config.json`` and every launcher reader died on it. A
    directory resolves exactly as before. None found: ``FileNotFoundError`` (the
    type a missing config always raised here, so callers that catch ``OSError``
    keep their behaviour), naming every place that was looked at."""
    from flliper.srt.server_args import (
        declared_config_path_candidates,
        declared_config_path_for,
    )

    path = declared_config_path_for(model)
    if path is None:
        looked = declared_config_path_candidates(model) or [
            os.path.join(str(model), "config.json")
        ]
        raise FileNotFoundError(
            2, "no config.json describes %s (looked at %s)" % (model, ", ".join(looked)),
            looked[-1],
        )
    return path

def model_is_gguf_file(model: str) -> bool:
    """Whether ``model`` names a GGUF checkpoint FILE -- the server's own
    predicate (``check_gguf_file``: a file with the ``.gguf`` suffix or the GGUF
    magic), the one that turns ``--load-format auto`` into ``gguf`` and the
    quantization into ``gguf`` (arg_groups/overrides._gguf_quantization)."""
    from flliper.srt.utils.hf_transformers_utils import check_gguf_file

    return bool(model) and check_gguf_file(model)

def gguf_backbone_depth(model: str) -> Optional[int]:
    """A GGUF file's backbone depth as the SERVER reconciles it
    (``gguf_registry.reconcile_sibling_config``): ``block_count`` minus the
    NEXTN/MTP draft blocks it counts (``nextn_predict_layers``). None where the
    metadata does not say. Read once per file (host_ledger.gguf_header_facts)."""
    return host_ledger.gguf_header_facts(model).backbone_depth

def parse_user_reserve(raw, cards) -> Dict[str, int]:
    """``--user-reserve-mib`` -> ``{card_uuid: MiB}``. Refuses, never guesses.

    #1257c. A scalar applies to every card; a list is in **RANK / CUDA-ORDINAL
    ORDER** -- the order :func:`order_cards` returns, i.e. the 5090 FIRST and
    then the 3080s by NVML index -- and must have exactly one entry per card.
    KEYED BY UUID on the way out for the reason every other per-card table in
    this file is: NVML enumeration is not stable across boots, and an
    index-keyed reserve would silently move to a different card the day the
    order shifts.

    CORRECTED #1342: this docstring, and the flag's help text, both used to say
    "NVML-ordinal order". THE CODE WAS ALWAYS RIGHT AND THE PROSE WAS WRONG --
    the values are zipped against ``cards``, which is
    ``order_cards(resolve_cards())`` (built at ``launcher.py:9082``, consumed
    at ``:9237``), so position 0 is the 5090 on this rig while NVML index 0 is
    a 3080. Boot weg2xsn17 proved it by controlled A/B: passing the DOCUMENTED
    ``1400,1800,1400`` put 1400 MiB on the 5090 and 1800 on a 3080, while
    ``1800,1400,1400`` -- the standing form in every order -- reproduced the
    reference boot byte-identically.

    SO THE DANGEROUS REPAIR IS THE OTHER ONE: "fixing" the code to NVML order
    would silently hand the 5090 the 1400 meant for a 3080, and the 5090 is the
    card that OOMed in #1312. The standing form ``1800,1400,1400`` is correct
    and matches the user setting (5090 1800 / 3080s 1400); it is pinned by
    ``test_the_standing_reserve_form_puts_1800_on_the_5090``.
    """
    text = str(raw if raw is not None else 0).strip()
    if not text:
        return {c.uuid: 0 for c in cards}
    try:
        values = [int(part) for part in text.split(",")]
    except ValueError:
        raise SystemExit(
            f"--user-reserve-mib must be an integer or a comma-separated list "
            f"of integers, got {text!r}."
        ) from None
    if any(v < 0 for v in values):
        raise SystemExit(f"--user-reserve-mib values must be >= 0, got {values}.")
    if len(values) == 1:
        values = values * len(cards)
    if len(values) != len(cards):
        raise SystemExit(
            f"--user-reserve-mib has {len(values)} entries but this boot has "
            f"{len(cards)} card(s); pass one value or exactly one per card."
        )
    return {c.uuid: v for c, v in zip(cards, values)}

def _owned_miss_min_paired() -> int:
    from flliper.srt.planner import expert_residency as _er

    return int(_er.OWNED_MISS_MIN_PAIRED_FORWARDS)

def _pconst_boots(name: str, profile: Optional[str] = None) -> str:
    """``boot <tag>[,<tag>...]`` of the record behind ``name`` for ``profile``
    (H94: a record names the boots it was measured on, so the budget line says
    whose measurement it charges); the provenance text where no boot is named."""
    prof = profile if profile is not None else pdflip_form.DEFAULT_PROFILE
    row = pdflip_form.profile_row(prof)
    rec = row.constants.get(name) if row is not None else None
    if rec is None:
        return ""
    boots = tuple(getattr(rec, "boots", ()) or ())
    if boots:
        own = "" if getattr(rec, "measured_on", prof) == prof else f" measured_on={rec.measured_on}"
        return "boot " + ",".join(boots) + own
    return str(getattr(rec, "provenance", ""))

def _mps_pids_on_pipe(pipe_dir: str, proc_root: str = "/proc") -> Dict[str, List[int]]:
    """The MPS control daemon and server processes of THIS boot: argv0 basename names the program, the
    environment must carry this boot's ``CUDA_MPS_PIPE_DIRECTORY`` (another boot's daemon is never touched)."""
    want = f"CUDA_MPS_PIPE_DIRECTORY={pipe_dir}".encode()
    out: Dict[str, List[int]] = {"daemon": [], "server": []}
    try:
        names = os.listdir(proc_root)
    except OSError:
        return out
    for n in names:
        if not n.isdigit():
            continue
        try:
            with open(f"{proc_root}/{n}/cmdline", "rb") as f:
                argv0 = os.path.basename(f.read().split(b"\0", 1)[0].decode(errors="replace"))
            kind = {"nvidia-cuda-mps-control": "daemon", "nvidia-cuda-mps-server": "server"}.get(argv0)
            if kind is None:
                continue
            with open(f"{proc_root}/{n}/environ", "rb") as f:
                if want in f.read().split(b"\0"):
                    out[kind].append(int(n))
        except OSError:
            continue
    return out

def p_mamba_mib_per_slot_by_stage(kinds: Sequence[str], stage_layers: Sequence[int],
                                  mib_per_linear_layer_per_slot: float) -> Tuple[float, ...]:
    """H92c: MiB one mamba slot costs on each P stage = price x the stage's
    LINEAR layers, counted off the checkpoint's own ``layer_types`` over the
    contiguous cut (Next Flash 29,11,8 -> 22/8/6 linear -> 34.29/12.47/9.35)."""
    out: List[float] = []
    start = 0
    for n in stage_layers:
        seg = list(kinds[start:start + int(n)])
        out.append(float(mib_per_linear_layer_per_slot)
                   * sum(1 for k in seg if str(k) != "full_attention"))
        start += int(n)
    return tuple(out)

def p_mamba_floor_text(slots: int, seats: int) -> str:
    """H92c: the runtime's hard floor against P's slots, as a named clause of
    the budget-posts line. A WARNING, never a refusal (the riegel is AW's):
    at retention 0 every anchor host backup is refused (fnFL2h91bb2)."""
    floor = int(seats) * int(P_MAMBA_FLOOR_SLOTS_PER_SEAT)
    retention = int(slots) - floor
    text = ("mamba floor %d seats x %d = %d, retention budget %d"
            % (int(seats), int(P_MAMBA_FLOOR_SLOTS_PER_SEAT), floor, retention))
    if retention <= 0:
        text += (" -- WARNUNG MAMBA-RETENTION: --max-mamba-cache-size %d <= Sitze x %d; "
                 "jeder Host-Backup eines Ankers wird verweigert (fnFL2h91bb2 "
                 "'MAMBA-FLOOR pool=32 floor=32 retention_budget=0'); mindestens "
                 "Sitze x %d + Retention-Budget geben" % (
                     int(slots), int(P_MAMBA_FLOOR_SLOTS_PER_SEAT),
                     int(P_MAMBA_FLOOR_SLOTS_PER_SEAT)))
    return text

def _gcd_reduce(values: Sequence[int]) -> Tuple[int, ...]:
    vals = [max(1, int(v)) for v in values]
    g = math.gcd(*vals) if len(vals) > 1 else vals[0]
    return tuple(v // max(1, g) for v in vals)

def _group_fact_value(group: str, flag: str) -> Optional[str]:
    """The value EARLY_READ_FACTS states for ``flag`` on ``group`` -- the same
    row build_env exports and the argv builders emit."""
    for fact in EARLY_READ_FACTS:
        if fact.groups in ("both", group) and len(fact.flag) >= 2 and fact.flag[0] == flag:
            return str(fact.flag[1])
    return None

def _infeasible_breakdown(pcm, mlp, attn_units, budgets, cap, cards=()) -> str:
    """The per-rank arithmetic behind a W64 feasible=False, as one clause.

    A refusal that names only its verdict makes the reader re-derive the
    reason by hand -- three boots of this flip paid that price, because
    "does not leave a positive KV pool on at least one rank" does not say
    WHICH rank, nor by how much, nor which term ate the budget.  Every number
    here comes from the same model that produced the verdict (no second
    accounting): the weight shards and the mamba pool from ``pcm``, the
    predicted per-rank tokens straight off ``cap["p"]``, and the two flat
    reserves from ``uneven_perf``'s own constants -- so the four terms add up
    to the budget the gate used and the shortfall is readable without a
    second run.
    """
    THIS_GROUP = "D"   # this breakdown only ever prices group D's vectors
    try:
        from flliper.srt.uneven_perf import (
            _PREDICT_MAMBA_ACT_RESERVE_MIB,
            _PREDICT_MIN_RANK_TOKENS,
            _PREDICT_OVERHEAD_MIB,
        )

        MI = float(2 ** 20)
        attn = list(attn_units) if attn_units else None
        wb = pcm.per_rank_weight_bytes(list(mlp), attn)
        # #62: the same subtraction predict_capacity makes -- what the host
        # expert store holds is not on the card. Printed as its own term so
        # the line cannot read as "the card carries all of it" again.
        off = pcm.per_rank_offloaded_weight_bytes(list(mlp))
        mb = pcm.mamba_pool_bytes_for(attn)
        cell = float(pcm.kv_cell_bytes)
        overhead = float(_PREDICT_OVERHEAD_MIB + _PREDICT_MAMBA_ACT_RESERVE_MIB)
        ptok = list(cap.get("p") or [])
        rows = []
        for r in range(len(budgets)):
            free = (
                float(budgets[r]) - (wb[r] - off[r]) / MI - mb[r] / MI - overhead
            )
            # THE KEYS, so a reader can JOIN this line to the measurement.
            # Without them the number is unattachable: the census speaks
            # `pp<PP>tp<TP>`, the budget lines speak `card=`/`nvml_idx=`, and
            # this line spoke neither -- a peer tool built on it could not
            # tell which rank, which card, or even which group it was reading
            # (reported 2026-09-21). The group is constant here and said out
            # loud rather than left to be inferred.
            _card = cards[r] if r < len(cards) else None
            _key = "group=%s r%d" % (THIS_GROUP, r)
            if _card is not None:
                _key += " nvml%s %s" % (
                    getattr(_card, "nvml_index", "?"),
                    getattr(_card, "name", "?"),
                )
            rows.append(
                "%s budget=%d - weights=%.0f (of which %.0f in the host "
                "expert store) - mamba=%.0f - reserves=%.0f "
                "=> free=%.0f MiB / kv_cell=%.0f B = %d tokens%s"
                % (
                    _key, int(budgets[r]), (wb[r] - off[r]) / MI, off[r] / MI,
                    mb[r] / MI, overhead, free,
                    cell, int(ptok[r]) if r < len(ptok) else -1,
                    "  <-- BELOW the minimum" if (
                        r < len(ptok) and ptok[r] < _PREDICT_MIN_RANK_TOKENS
                    ) else "",
                )
            )
        return (
            "  THE ARITHMETIC (same model, no second accounting; a rank needs "
            ">= %d tokens): %s"
            % (_PREDICT_MIN_RANK_TOKENS, "; ".join(rows))
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        return "  (per-rank breakdown unavailable: %s)" % exc

def fp4_native_mixed_env(um_argv: Sequence[str]) -> Dict[str, str]:
    """The environment --fp4-native-mixed adds (launcher + both groups), ONE
    boot-uniform switch read by (a) the exchange class names -- the NVFP4 native
    parameters become leaves of their linear's class (contract L4) -- and (b)
    the planner's NVFP4 lanes (uneven_perf.NVFP4_LANE_RECORD, L7). Empty for
    every other argv, so no other boot changes."""
    if tuple(um_argv or ()) == FP4_NATIVE_MIXED_ARGV:
        return {"FLLIPER_FP4_NATIVE_MIXED_BOOT": "1"}
    return {}

def fp4_native_mixed_refusal(
    quant_method: str, uniform_marlin: bool, fp4_native_mixed: bool
) -> Optional[str]:
    """W160 text when --fp4-native-mixed cannot hold: it only re-routes the
    NVFP4 linears of a ModelOpt checkpoint, and the FP8 linears of that same
    checkpoint still need --fp8-uniform-marlin for one layout. None = fine."""
    if not fp4_native_mixed:
        return None
    if str(quant_method or "").lower() != "modelopt":
        return ("W160 PdFlipFp8LayoutRefused: --fp4-native-mixed applies to a ModelOpt "
                "NVFP4 checkpoint only; this checkpoint's quant_method is %r"
                % (quant_method or "none"))
    if not uniform_marlin:
        return ("W160 PdFlipFp8LayoutRefused: --fp4-native-mixed moves only the NVFP4 "
                "linears to the native layout; the FP8 linears of the same ModelOpt "
                "checkpoint still need --fp8-uniform-marlin (one layout on every rank)")
    return None

def fp8_layout_decision(
    quant_method: str, weight_source: str, uniform_marlin: bool
) -> Tuple[Dict[str, str], Optional[str]]:
    """``(env for BOTH groups, log line or None)`` for --fp8-uniform-marlin;
    raises W160 for the one combination that cannot flip correctly.

    * not an FP8 checkpoint: ``{}``; the flag is inert (named once if given);
    * FP8 + the flag: :data:`FP8_UNIFORM_MARLIN_ENV` on every rank;
    * FP8 under ``--pdflip-weight-source exchange`` WITHOUT the flag: REFUSED --
      the 5090 and the 3080s would hold different bytes for one tensor and
      the exchange would move one card's layout onto the other's;
    * FP8 under the ring (each group refills ITS OWN bytes on the same card):
      native per-card layouts never meet, nothing to force; named, not forced.
    """
    qm = str(quant_method or "").lower()
    if qm not in UNIFORM_MARLIN_QUANT_METHODS:
        if uniform_marlin:
            return {}, ("PDFLIP FP8-UNIFORM-MARLIN inert: the checkpoint's quant_method is %r, "
                        "not one of %s -- no environment added"
                        % (qm or "none", "/".join(UNIFORM_MARLIN_QUANT_METHODS)))
        return {}, None
    if uniform_marlin:
        if qm == "modelopt":
            layout = ("FP8 linears Marlin int32 [K/16,4N] + per-channel scales [1,N], "
                      "NVFP4 linears Marlin int32 [K/16,2N] + fp8 scales [K/16,N] via "
                      "server argv %s on both groups; the 5090 gives up its native "
                      "FP8 and NVFP4 CUTLASS GEMMs for it" % " ".join(FP4_UNIFORM_MARLIN_ARGV))
        else:
            layout = ("Marlin int32 [K/16,4N] + group scales [K/128,N] on the 5090 as on "
                      "the 3080s; the 5090 gives up its native block-FP8 GEMM for it")
        return dict(FP8_UNIFORM_MARLIN_ENV), (
            "PDFLIP FP8-UNIFORM-MARLIN on (--fp8-uniform-marlin, %s checkpoint): %s on "
            "every rank of BOTH groups -- one byte layout for the flip exchange (%s); "
            "the Marlin lock workspace is booked local_scratch and re-zeroed after "
            "every weights resume (rank line 'PDFLIP-RESUME local-scratch zeroed=')"
            % (qm, " ".join("%s=%s" % kv for kv in sorted(FP8_UNIFORM_MARLIN_ENV.items())),
               layout))
    if str(weight_source) == WEIGHT_SOURCE_EXCHANGE:
        if qm == "modelopt":
            raise PdFlipLaunchRefused(
                "W160 PdFlipFp8LayoutRefused: the checkpoint is a ModelOpt export "
                "(quant_method modelopt: FP8 and/or NVFP4 linears) and --pdflip-weight-source "
                "exchange moves weight BYTES card to card, but both kernel choices are "
                "rank-local: a 3080 repacks to Marlin (FP8 int32 [K/16,4N], NVFP4 int32 "
                "[K/16,2N] + fp8 scales [K/16,N]), the 5090 keeps native layouts (fp8 "
                "[N,K] + scalar scale; NVFP4 CUTLASS with swizzled block scales). One "
                "tensor would have two layouts and the flip would copy one onto the "
                "other. Pass --fp8-uniform-marlin (Marlin on every rank, one layout), or "
                "run --pdflip-weight-source ring.")
        raise PdFlipLaunchRefused(
            "W160 PdFlipFp8LayoutRefused: the checkpoint is FP8 (quant_method fp8) and "
            "--pdflip-weight-source exchange moves weight BYTES card to card, but the "
            "FP8 kernel choice is rank-local: a 3080 repacks to Marlin int32 "
            "[K/16,4N] + group scales, the 5090 keeps native fp8 [N,K] + "
            "weight_scale_inv. One tensor would have two layouts and the flip would "
            "copy one onto the other. Pass --fp8-uniform-marlin (Marlin on every "
            "rank, one layout), or run --pdflip-weight-source ring.")
    return {}, ("PDFLIP %s native per-card layouts under --pdflip-weight-source %s: each "
                "group refills its own bytes on its own card, no byte crosses a card "
                "boundary, nothing forced" % (qm.upper(), weight_source))

def p_attn_head_split_env(cfg) -> Dict[str, str]:
    """Group P's env for the split; {} when off (byte-identical)."""
    from flliper.srt.pdflip import attn_head_split as _ah

    return {} if cfg is None else {_ah.ENV: cfg.to_env()}

def p_attn_head_split_line(cfg, pp_size: int) -> str:
    from flliper.srt.pdflip import attn_head_split as _ah

    return (
        "PDFLIP P-ATTN-HEAD-SPLIT: on %s (owner:helper:layers:G|hA-B), min_w %d, mirror cap %d "
        "tokens, max_w %d, deadline %.1f ms, ab %s; post '%s' MiB per stage = %s (booked by the "
        "ranks and by the cut's pool model), barlink p2p %.2f MiB/pair, pinned host %.1f MiB "
        "(ATTN_HEAD_SPLIT.md, P_MICROBATCH_AH2.md)"
        % (",".join(d.text() for d in cfg.delegations), cfg.min_w, cfg.cap_tokens, cfg.max_w,
           cfg.deadline_ms if any(d.downstream for d in cfg.delegations) else 0.0, cfg.ab or "off",
           _ah.POST_NAME, ",".join("%.1f" % v for v in _ah.stage_post_vector(cfg, pp_size)),
           cfg.p2p_bytes() / 2**20, _ah.host_pinned_mib(cfg, pp_size))
    )

def p_trim_end_anchor_env(on: bool) -> Dict[str, str]:
    """Group P's environment for ``--p-trim-end-anchor``: {} when off (the
    byte-identity guarantee). The variable NAME lives in
    pdflip/p_trim_end_anchor.py, the one module the ranks read it from."""
    from flliper.srt.pdflip import p_trim_end_anchor as _pt

    return {_pt.TRIM_ENV: "1"} if on else {}

def fork_anchor_env(token: Optional[int], p_trim: bool) -> Dict[str, str]:
    """``--fork-anchor-token``: the SAME environment for group P and group D
    ({} when off -- the byte-identity guarantee). The variable NAME lives in
    pdflip/fork_anchor.py, the one module the ranks read it from.

    Refused without ``--p-trim-end-anchor``: D's store read stops at the fork
    and must meet a P leg whose END anchor sits there; P's split path anchors
    at N-1 and would leave D's capped read without an anchor in range."""
    from flliper.srt.pdflip import fork_anchor as _fa

    if token is None:
        return {}
    if int(token) <= 0:
        raise SystemExit(f"--fork-anchor-token must be a positive token id, got {token}")
    if not p_trim:
        raise SystemExit(
            "--fork-anchor-token needs --p-trim-end-anchor: group D stops its store "
            "read at the prompt's fork, which only a fork-cut P leg anchors")
    return {_fa.TOKEN_ENV: str(int(token))}

def p_host_overlap_env(overlap: bool, hostgap: bool) -> Dict[str, str]:
    """Group P's extra environment for ``--p-host-overlap`` / ``--p-hostgap``.

    Empty when both are off -- that is the byte-identity guarantee, pinned by
    test_pdflip_p_host_overlap.py. The variable NAMES live in
    managers/pdflip_p_overlap.py, the one module the runtime reads them from."""
    from flliper.srt.managers import pdflip_p_overlap as _pov

    env: Dict[str, str] = {}
    if overlap:
        env.update(_pov.launcher_env_p_host_overlap())
    if hostgap:
        env.update(_pov.launcher_env_p_hostgap())
    return env

def p_recv_stream_env(on: bool) -> Dict[str, str]:
    """Group P's extra environment for ``--p-recv-stream``; empty when off
    (byte-identity, pinned by test_pdflip_p_recv_stream.py). The NAME lives in
    managers/pdflip_p_overlap.py, the one module the runtime reads it from."""
    from flliper.srt.managers import pdflip_p_overlap as _pov

    return _pov.launcher_env_p_recv_stream() if on else {}

def _p_chunk_dynamic_runnable(ns) -> bool:
    """Whether --p-chunk-policy dynamic can run with ``ns``'s other flags --
    the refusals of :func:`apply_p_chunk_policy`, read from the parsed flags
    (the graph is installed later): the plan's baseline width (--p-chunk-fixed,
    else the --p-prefill-graph bucket, else CHUNKED_PREFILL_TOKENS 4096) and its
    floor (--p-chunk-min, else that width) must not exceed --p-chunk-max, and
    the graph bucket neither. The measured form (27b.env, chunkab rc9j) is
    dynamic WITH --p-prefill-graph 512; without a graph the baseline 4096 is
    above the default max 2048 and the launcher would refuse the boot -- so a
    registry default of dynamic applies only where it is runnable."""
    bucket = int(getattr(ns, "p_prefill_graph", 0) or 0)
    cap = int(getattr(ns, "p_chunk_max", P_CHUNK_MAX_DEFAULT) or P_CHUNK_MAX_DEFAULT)
    fixed = int(getattr(ns, "p_chunk_fixed", 0) or 0) or bucket or CHUNKED_PREFILL_TOKENS
    low = int(getattr(ns, "p_chunk_min", 0) or 0) or fixed
    return 0 < low <= cap and not (bucket and cap < bucket)

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

def p_log_model_path(path: str, max_lines: int = 20000) -> str:
    """The ``model_path`` a P log's boot ran (its first server_args line); ``""``
    when the log names none within ``max_lines``."""
    try:
        with open(path, errors="replace") as fh:
            for n, line in enumerate(fh):
                if "server_args=ServerArgs(" in line:
                    m = _P_LOG_MODEL_RE.search(line)
                    if m:
                        return m.group(1)
                if n >= max_lines:
                    break
    except OSError:
        return ""
    return ""

def forced_lane_borrow(cards: Sequence[Card], lanes: Sequence[Optional[int]],
                       log=None) -> List[Optional[int]]:
    """AP2 1006: the PCIe width per ordinal the crossing price is looked up at.
    The measured link table (``MEASURED_GBPS_BY_LANES``) holds this rig's widths
    only (x4, x8); a card at another width (x16 on most boards) or one NVML does
    not report used to drop every candidate crossing it as UNPRICED, so a
    foreign inventory ended in W40 "not one cut ... priceable" even with --force.

    Unchanged whenever every width has a measured entry (the reference rig: the
    list comes back as it went in). Otherwise the value refusal HW-UNCALIBRATED:
    without --force refused by name; with it the ordinal is priced at the
    largest MEASURED width not above its own (x16 -> x8: never faster than
    measured), an unreported width at the smallest measured one, named
    UNMEASURED. A width below every measured one (x1, x2) gets no borrow -- any
    measured price would be optimistic -- and stays unpriced."""
    from flliper.srt.distributed.pp_crossing_transport import MEASURED_GBPS_BY_LANES

    have = sorted(int(k) for k in MEASURED_GBPS_BY_LANES)
    out = [None if x is None else int(x) for x in lanes]
    moved = []
    for i, w in enumerate(out):
        if w is not None and w in have:
            continue
        below = [k for k in have if w is None or k <= w]
        if not below:
            continue
        new = below[0] if w is None else below[-1]
        moved.append(f"ordinal {i} ({card_identity.class_label(cards[i]) if i < len(cards) else '?'}) "
                     f"x{w if w is not None else '?'} -> x{new}")
        out[i] = new
    if not moved:
        return list(lanes)
    refusals.refuse_value(
        card_identity.CODE_UNCALIBRATED,
        f"{card_identity.CODE_UNCALIBRATED}: PP-CUT crossing prices are measured at PCIe widths "
        f"{', '.join('x%d' % k for k in have)} only; " + "; ".join(moved)
        + " -- --force prices these at the nearest measured width not faster than their own: UNMEASURED "
        "on this board (a crossing measurement of this rig writes the table)",
        PdFlipLaunchRefused, log=log)
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
    from flliper.srt.distributed.pp_crossing_transport import MEASURED_GBPS_BY_LANES

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

def _fmt_gib(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2f}"

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

def expert_map_dir(evidence_dir: str, log) -> str:
    """Where the expert maps (#107 Platztausch, V1) are written: the evidence
    dir when this process can write it, else a NAMED writable fallback
    (``FLLIPER_PDFLIP_EXPERT_MAP_DIR``, default ``<tmp>/pdflip-expert-maps``).

    Dry runs against a container's evidence tree (owned by the container's
    user) printed 'PDFLIP-EXPERT-MAP failed: PermissionError' and published no
    map -- so the dry run never showed the map the boot would build. Both
    groups read the map by the path in their env, so any directory this
    launcher can write and the ranks can read is correct; the redirect is
    logged so a reader never looks for the map in the wrong place."""
    try:
        os.makedirs(evidence_dir, exist_ok=True)
    except OSError:
        pass
    if os.path.isdir(evidence_dir) and os.access(evidence_dir, os.W_OK | os.X_OK):
        return evidence_dir
    import tempfile

    alt = (os.environ.get(EXPERT_MAP_DIR_ENV)
           or os.path.join(tempfile.gettempdir(), "pdflip-expert-maps"))
    os.makedirs(alt, exist_ok=True)
    log(f"PDFLIP-EXPERT-MAP-DIR {evidence_dir} is not writable by this process -> "
        f"maps go to {alt} ({EXPERT_MAP_DIR_ENV}); both groups read them by the "
        f"path in their env")
    return alt

def p_card_fr_cap(fits, fracs: Sequence[float], num_experts: int,
                  lru_rows: Sequence[int]) -> PFrCap:
    """FR_P als Obergrenze: jede Stufe, die die Karte reisst, auf ihre
    Karten-Decke (``PCardFit.ceiling_fraction`` = groesste Fraction, deren
    Puffer in die tragbaren Zeilen passt); jede andere Stufe EXAKT wie gegeben.

    Untergrenze ist die bestehende Mindestresidenz der Stufe:
    ``largest_fraction_for_rows`` gibt None, wenn die Karte nicht einmal EINEN
    residenten Experten plus ihren Scratch traegt -- dann bleibt W132 mit Namen
    (``refusal``), statt auf 0 zu kappen. Ebenso, wenn die Decke nicht unter
    der gegebenen Fraction liegt (dann liegt der Mangel nicht an den Zeilen,
    und Kappen haelfe nicht)."""
    from flliper.srt.planner import expert_residency as _er

    new = [float(x) for x in fracs]
    stages, lines, why = [], [], []
    for fit in fits:
        if not fit.refused:
            continue
        s = int(fit.stage)
        short = float(fit.near_oom_mib) - float(fit.headroom_mib)
        ceil = fit.ceiling_fraction
        if ceil is None:
            why.append(
                "stage%d (%s): keine Kappung -- die Karte traegt nicht einmal einen "
                "residenten Experten plus Scratch %d (Mindestresidenz, "
                "largest_fraction_for_rows), es fehlen %.0f MiB"
                % (s, fit.card, int(lru_rows[s]), short))
            continue
        if float(ceil) >= float(fracs[s]):
            why.append(
                "stage%d (%s): keine Kappung -- die Karten-Decke f %.3f liegt nicht "
                "unter der gegebenen f %.4f, der Mangel (%.0f MiB) liegt nicht an den "
                "Experten-Zeilen" % (s, fit.card, float(ceil), float(fracs[s]), short))
            continue
        new[s] = float(ceil)
        stages.append(s)
        rows_new = _er.buffer_rows(local_experts=int(num_experts), fraction=float(ceil),
                                   scratch_rows=int(lru_rows[s]))
        lines.append(
            "stage%d (%s): f %.4f -> %.3f (%d -> %d Zeilen, Decke %d Zeilen): Kopfraum "
            "am letzten Chunk %.0f MiB < near-OOM %.0f, es fehlen %.0f MiB (Chunk-Index "
            "%d, Prompt %d); FR_P ist die Obergrenze, der Planer kappt"
            % (s, fit.card, float(fracs[s]), float(ceil), int(fit.buffer_rows), rows_new,
               int(fit.ceiling_max_rows), float(fit.headroom_mib), float(fit.near_oom_mib),
               short, int(fit.last_chunk_index), int(fit.prompt_tokens)))
    return PFrCap(tuple(new), tuple(stages), tuple(lines), "; ".join(why))

def n_stage_incumbent(n_stages: int, n_layers: int, ms_per_layer: Sequence[float]) -> List[int]:
    """HW-P1c 1003 (PP-CUT-PIN): the solver's starting cut for a P group of
    ``n_stages`` stages: layers in proportion to each stage's speed (1 / its
    measured ms per layer), every stage >= 1 layer, the sum exact. A STARTING
    point (the cost model's incumbent); the cut that ships is the solver's."""
    n = int(n_stages)
    rates = [1.0 / max(1e-9, float(m)) for m in list(ms_per_layer)[:n]]
    if len(rates) != n:
        rates = [1.0] * n
    tot = sum(rates)
    counts = [max(1, int(round(r / tot * int(n_layers)))) for r in rates]
    counts[-1] = int(n_layers) - sum(counts[:-1])
    while counts[-1] < 1:   # the last stage must keep a layer: take from the largest
        j = max(range(n - 1), key=lambda i: counts[i])
        counts[j] -= 1
        counts[-1] += 1
    return counts
