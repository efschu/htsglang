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
from flliper.srt.name_compat import tolerant_compile
from flliper.srt.pdflip import DEFAULT_PP_ORDERED_CUT
from flliper.srt.pdflip import DEFAULT_P_BS
from flliper.srt.pdflip import card_identity
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import pp_ordered_cut_for as pdflip_pp_ordered_cut_for
from flliper.srt.pdflip import refusals
from flliper.srt.pdflip import state_file as state_file_mod
from flliper.srt.pdflip.launcher_d_leaf import P_COVERS_D_SESSION_MARK
from flliper.srt.pdflip.launcher_d_leaf import P_DORMANT_EXPECT_N
from flliper.srt.pdflip.launcher_d_leaf import _argv_scalar
from flliper.srt.pdflip.launcher_d_leaf import _argv_vector
from flliper.srt.pdflip.launcher_d_leaf import _credit_counter_rows
from flliper.srt.pdflip.launcher_d_leaf import _csv
from flliper.srt.pdflip.launcher_d_leaf import d_expect_from_p_records
from flliper.srt.pdflip.launcher_d_leaf import d_kv_token_cut
from flliper.srt.pdflip.launcher_d_leaf import dormant_max_from_records
from flliper.srt.pdflip.launcher_d_mid import D_RANK_SOLVE_MARKER
from flliper.srt.pdflip.launcher_p_l3_leaf import P_CHUNK_BUILTIN_SOURCES
from flliper.srt.pdflip.launcher_p_l3_leaf import P_CHUNK_MODEL_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import _l3_extra_flag
from flliper.srt.pdflip.launcher_p_l3_leaf import _l3_revoked_path
from flliper.srt.pdflip.launcher_p_l3_leaf import _l3_rope_view
from flliper.srt.pdflip.launcher_p_l3_leaf import _last_flag_value
from flliper.srt.pdflip.launcher_p_l3_leaf import _median
from flliper.srt.pdflip.launcher_p_l3_leaf import _p_chunk_interp
from flliper.srt.pdflip.launcher_p_l3_leaf import incumbent_candidate
from flliper.srt.pdflip.launcher_p_l3_leaf import l3_weights_fingerprint
from flliper.srt.pdflip.launcher_p_l3_leaf import model_config_path
from flliper.srt.pdflip.launcher_p_l3_leaf import p_chunk_model_cut
from flliper.srt.pdflip.launcher_p_l3_leaf import p_chunk_policy_spec
from flliper.srt.pdflip.launcher_p_l3_leaf import p_host_overlap_env
from flliper.srt.pdflip.launcher_p_l3_leaf import p_prefill_graph_bucket
from flliper.srt.pdflip.launcher_p_l3_leaf import p_prefill_graph_captured
from flliper.srt.pdflip.launcher_p_l3_leaf import p_recv_stream_env
from flliper.srt.pdflip.launcher_p_l3_leaf import parse_group_env
from flliper.srt.pdflip.launcher_p_l3_leaf import pdflip_boot_token
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _proc_ppid
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _resolve_keys_in_rank_tree
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _same_vector
from flliper.srt.pdflip.launcher_refuse_proc_leaf import read_mean_prefill_prefix
from flliper.srt.pdflip.launcher_refuse_proc_leaf import read_pp_bubble
from flliper.srt.pdflip.launcher_refuse_proc_leaf import records_inventory
from flliper.srt.pdflip.launcher_refuse_proc_leaf import standard_form_resolved
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import PdFlipDEarlyStartUnreviewed
from flliper.srt.pdflip.launcher_types import PdFlipDFormOffTheMap
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_types import PdFlipStoreDiskRefused
from flliper.srt.pdflip.launcher_xchg_vision_leaf import XCHG_D_VECTOR_DEFAULT
from flliper.srt.pdflip.launcher_xchg_vision_leaf import _w_code_of
from flliper.srt.planner import p_card_chunk as _p_card
import hashlib
import json
import math

KV_CACHE_DTYPE = "fp8_e4m3"
#: The fixed part of a GRAPH forward that does not scale with the chunk
#: (replay launch, stage in/out), ms. Small by the #PGAP host lines (launch 3).
P_CHUNK_GRAPH_FIXED_MS = 2.0
#: The dry-run ladder the launcher prints its plan for (the Messplan rungs).
P_CHUNK_DRY_RUN_TOKENS = (2048, 8192, 32768, 131072)
P_CHUNK_CUT_TAG = "P-CHUNK-MODEL"
_RE_FLIP = re.compile(r"flip_total=(\d+) ms")
_RE_LEG1 = tolerant_compile(
    r"PDFLIP-SERVED group=P leg=1 .*?prompt_tokens=(\d+) cached_tokens=(\d+) wall=([0-9.]+)s"
)
#: #1271: the front's own P-drain window -- `prefilled` legs over `drain_s`
#: seconds. This is the denominator r_P was always meant to have.
_RE_DRAIN = tolerant_compile(r"PDFLIP P-DRAIN epoch=(\d+) .*?drain_s=([0-9.]+)")
_RE_LEG2 = tolerant_compile(
    r"PDFLIP-SERVED group=D leg=2 .*?uncached=(\d+) verdict=(\S+) wall=([0-9.]+)s"
)
#: The runtime's OWN safety margin on the demand-driven mamba pool
#: (model_runner_kv_cache_mixin.py:177 ``MAMBA_AUTO_SAFETY_MARGIN``). Restated
#: here rather than imported because importing that module drags torch into a
#: launcher that has no business loading it; the restatement is not free-hand
#: -- ``test_pp_cut_boot_sizing_1286`` reads the constant out of the runtime's
#: source and fails when the two drift.
P_MAMBA_AUTO_SAFETY_MARGIN = 1.25
#: #239 S3f: the marker of the published ownership.
D_OWNER_MARKER = "D-EIGENTUM (#239 S3f)"
#: L3P: bump when the ON-DISK page format changes incompatibly (the #706
#: geometry-neutral canonical page is generation 706); a new generation gets
#: a new directory instead of a MixedGenerationError on a reused one.
L3_PERSIST_GENERATION = "706"
#: HANDBACK N-1 (01.10.): profiles whose anchor keying switched to the exact
#: bigram form. Their L3 identity names it, so a store written under the
#: upstream keying -- whose persisted Mamba anchors at k units carry the state
#: after k+1 tokens -- becomes a new directory instead of a silently wrong
#: anchor (the page KV keys themselves are unchanged). NF ran the exact form
#: already; its identity and directory stay byte for byte.
L3_ANCHOR_KEYING_PROFILES = {"qwen27b": "exact-n1"}
#: YaRN x2: how the runtime applies a rope override (hf_transformers.config.
#: apply_model_override_args, nested sub-config merge). Bump when that changes.
L3_ROPE_APPLY = "merge-v1"
#: fnFL2 H37 (Task #118): the scheduler flag that bounds how many requests ONE
#: prefill forward of group P may carry.
P_MICRO_BATCH_FLAG = "--pp-max-micro-batch-size"
GGUF_DENSE_VOCAB_ENV = "FLLIPER_GGUF_DENSE_VOCAB"
_CAUSE_RE = re.compile(r"\b(W\d+[a-z]?)\s+(PdFlip[A-Za-z0-9]+)")
#: The env the D ranks read FR_D from (resident_fraction.resident_fraction_for_rank).
D_FR_ENV = "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION"
#: MAMBA-ARENA-FLOOR (16.10.): the most mamba state slots one running request can
#: hold -- MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO 3 + the overlap ping-pong 2
#: (model_runner_kv_cache_mixin._calculate_mamba_ratio), and the most
#: mamba_pool_floor.mamba_slots_per_running_req returns (1 + 2 + 1 + 1). With the
#: runtime's own safety margin, ceil(bs x 5 x 1.25) is an UPPER bound on the
#: demand-driven device pool (_auto_mamba_demand_size = max(ceil(t*ratio*1.25),
#: ratio, t*slots_per_req), then min(.., fit_cap)): measured D 1016_103428
#: target_concurrency=6 ratio=5 safety=1.25 -> 38 = ceil(6 x 6.25). An upper bound
#: over-sizes the arena, the safe direction (host RAM only warns).
MAMBA_ARENA_SLOTS_PER_REQUEST_UPPER = 5

def p_dormant_from_records(
    recs: Optional[Sequence[Dict[str, object]]], cards: Sequence[Card], weight_source: str,
    n: int = P_DORMANT_EXPECT_N,
) -> Tuple[Optional[Dict[str, int]], str]:
    """Group P's measured dormant residue for D's expectation budget
    (:func:`dormant_max_from_records` over group-P records)."""
    return dormant_max_from_records(recs, cards, weight_source, n, group="P")

def p_prefill_graph_buckets() -> List[int]:
    """Every captured prefill bucket, ascending; [] when no graph is captured
    (the switch off, or --p-prefill-graph-policy auto dropped every bucket)."""
    if not p_prefill_graph_bucket():
        return []
    return p_prefill_graph_captured()

def _p_chunk_stage(a_ms: float, b_ms_per_1k: float, fwd_overhead_ms: float,
                   eager_floor_ms: float, mscale: Dict[int, float], name: str):
    """One stage of the policy model from a 512-chunk line and a scale curve."""
    from flliper.srt.pdflip import p_chunk_policy as _pcp

    ref = 512
    fixed = min(P_CHUNK_GRAPH_FIXED_MS, float(a_ms))
    grid = sorted(set(mscale) | {256, 512, 1024, 2048, 4096})
    pts = []
    for m in grid:
        r = _p_chunk_interp(mscale, m)
        pts.append((m, float(fwd_overhead_ms) + fixed + (float(a_ms) - fixed) * (m / ref) * r))
    return _pcp.StageModel(tuple(pts), float(b_ms_per_1k) / ref, float(eager_floor_ms), name)

def p_chunk_model_cut_check(ns, cut, log) -> str:
    """UNIFY S9 (Agent PG: a per-stage table holds only for its P cut): the
    --p-chunk-model table's cut against the cut this boot SHIPS, once, right
    after the cut is final (after the graph table's own check). Nothing under
    'fixed'. The table names no cut: a WARNING, used as before. A different
    cut: a JSON table (an export for ONE cut) is REFUSED -- there is no
    table-less dynamic plan to fall back to, and switching to 'fixed' would
    change P's chunk ceiling behind the operator's back; re-export it for the
    cut (p_stage_model --cut), pin --pp-stage-ratio to its cut, or run
    --p-chunk-policy fixed. The builtin INT8 model (fitted at 42,11,11, a desk
    model for every cut until now) only warns. Returns the verdict."""
    if p_chunk_policy_spec() is None:
        return ""
    src = str(getattr(ns, "p_chunk_model", P_CHUNK_MODEL_DEFAULT) or P_CHUNK_MODEL_DEFAULT).strip()
    table, where = p_chunk_model_cut(src)
    counts = tuple(int(x) for x in (getattr(cut, "layer_counts", ()) or ()))
    gapped = bool(getattr(cut, "gapped", False))
    boot = (f"gapped({getattr(cut, 'layer_set', '') or ','.join(map(str, counts))})" if gapped
            else ",".join(map(str, counts)) or "?")
    if not table:
        log(f"PDFLIP {P_CHUNK_CUT_TAG} CUT-UNKNOWN WARNING table={where} names no pp_layer_ratio, "
            f"boot={boot} -> used as before; a per-stage table is only valid for the cut it was "
            f"measured on")
        return "unknown"
    tcut = ",".join(map(str, table))
    if not gapped and counts == tuple(table):
        log(f"PDFLIP {P_CHUNK_CUT_TAG} CUT-MATCH table={tcut} boot={boot} -> used ({where})")
        return "match"
    if src in P_CHUNK_BUILTIN_SOURCES:
        log(f"PDFLIP {P_CHUNK_CUT_TAG} CUT-MISMATCH WARNING table={tcut} boot={boot} ({where}): the "
            f"{src} model was fitted on {tcut}; its per-stage times do not describe this cut "
            f"-- the plan is a HOCHRECHNUNG here (not refused: the builtin is the desk default); "
            f"export a model for this cut (p_stage_model) to plan on a measurement")
        return "mismatch"
    msg = (f"{P_CHUNK_CUT_TAG} CUT-MISMATCH table={tcut} boot={boot} ({where}): the table was "
           f"measured on another P cut and is not valid for this one -- re-export it for {boot} "
           f"(p_stage_model), pin --pp-stage-ratio to {tcut}, or run --p-chunk-policy fixed")
    log("PDFLIP " + msg + " -- REFUSED")
    raise SystemExit(msg)

def p_chunk_policy_env() -> Dict[str, str]:
    """Group P's environment for the policy; {} under 'fixed' (byte-identical)."""
    from flliper.srt.pdflip import p_chunk_policy as _pcp

    spec = p_chunk_policy_spec()
    if spec is None:
        return {}
    return {_pcp.POLICY_ENV: _pcp.POLICY_DYNAMIC, _pcp.SPEC_ENV: spec.to_json()}

def p_chunk_policy_lines() -> List[str]:
    """The launcher's lines: the armed spec plus the plan it WOULD pick for the
    dry-run rungs (a prediction on the model, printed so the metal ladder is
    read against a stated expectation). [] under 'fixed'."""
    from flliper.srt.pdflip import p_chunk_policy as _pcp

    spec = p_chunk_policy_spec()
    if spec is None:
        return []
    out = ["PDFLIP " + _pcp.armed_line(spec, "group=P")
           + f" -- group P's --chunked-prefill-size is the ceiling {spec.limits.max_tokens}"]
    for n in P_CHUNK_DRY_RUN_TOKENS:
        if spec.limits.dynamic_min_tokens and n <= spec.limits.dynamic_min_tokens:
            out.append(f"PDFLIP {_pcp.LOG_TAG} plan policy=dynamic key=dry-run-{n} tokens={n} "
                       f"BYPASS (<= --p-chunk-dynamic-min-tokens {spec.limits.dynamic_min_tokens}): "
                       f"fixed width {spec.limits.fixed_tokens}, no plan")
            continue
        res = _pcp.plan_detail(n, len(spec.stages), spec.stages, spec.limits)
        out.append("PDFLIP " + _pcp.plan_line(res, key=f"dry-run-{n}", start=0, end=n)
                   + " (HOCHRECHNUNG on the model, not a measurement)")
    return out

def measure_x_inputs(log_path: str, floor_tokens: int) -> Optional[Tuple[float, float, float, int, int, int]]:
    """(flip_s, r_D, r_P, n_flips, n_leg2, n_leg1) from ONE front log, or None.

    Reads the front's OWN instruments -- ``flip_total=`` and the two
    ``PDFLIP-SERVED`` lines -- so every input to X is a measurement this rig
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

def publish_d_owner_ratio(ns, plan, label: str, log) -> List[float]:
    """#239 S3f: the ownership vector the planner solved under
    ``--d-kv-token-cut owned`` goes to ``--extra-d --rank-moe-ratio`` (the one
    place it lives for group D; argparse keeps the last copy, so every copy is
    replaced), and the before/after into ``ns._d_owner_solve`` for the
    record. Returns the new vector as floats for the caller's later readers."""
    from flliper.srt.pdflip import draft_post as _dp

    before = _argv_vector(getattr(ns, "extra_d", ""), "--rank-moe-ratio")
    if getattr(ns, "_d_owner_stated", None) is None:
        # the STATED vector, kept once: a second solve pass (dry-run and real
        # call, the d-only expectation) starts from it again, never from its
        # own published answer (Form A baseline of the x1 rule stays fixed)
        ns._d_owner_stated = [str(x) for x in before]
    before = list(ns._d_owner_stated)
    new = [int(x) for x in plan.solved_owner_ratio]
    ns.extra_d = _dp.replace_vector_flag(getattr(ns, "extra_d", "") or "",
                                         "--rank-moe-ratio", new)
    rec = dict(plan.owner_record)
    rec["label"] = label
    rec["stated_ratios"] = [str(x) for x in before]
    ns._d_owner_solve = rec
    log(f"{D_RANK_SOLVE_MARKER} {label} {D_OWNER_MARKER} veroeffentlicht: --extra-d "
        f"--rank-moe-ratio {','.join(str(x) for x in new)} (vorher "
        f"{','.join(str(x) for x in before)}; der Planer setzt das Eigentum, kein Handwert)")
    return [float(x) for x in new]

def pin_d_form_for_map(ns, log) -> Optional[dict]:
    """#239 rc12z29c: freeze the D form the Platztausch map is built from.

    rc12z29b (f833fcbb2d, -st-cut) died at D's load, TP1: 'Platztausch-Karte:
    Layer 0 Phase D nennt Ids ausserhalb dieses Rangs (lokal [-33, ...],
    lo=226)' -- the map was built (publish_expert_map, before P starts) from
    the STATED --rank-moe-ratio 183,137,168, the owned solve (after P's sleep)
    published 215,113,160 into --extra-d, and D's expert window followed the
    published one. The map cannot follow afterwards: P loaded its common
    prefix and store slots from it. So the form is solved once before the map
    (``d_form_solved_by_planner``), the map reads the published vectors, and
    every later D solve checks exactly that form (fixed cut, fixed ownership,
    fixed FR_D) against the measured budgets -- a form that no longer fits is
    refused by name (W122/W130), never re-solved under a map that describes
    another one. Returns the pinned form (also on ``ns._d_map_form``)."""
    ratios = _argv_vector(getattr(ns, "extra_d", ""), "--rank-moe-ratio")
    fr = _argv_vector(getattr(ns, "extra_d", ""), "--rank-moe-resident-fraction")
    cut = getattr(ns, "_d_solved_cut", None)
    # R2: a form without a token cut (the P1c pin) pins cut None -- only a
    # cut that was ASKED for and not published leaves the form unpinned
    if not ratios or not fr or (not cut and d_kv_token_cut(ns) is not None):
        log(f"{D_RANK_SOLVE_MARKER} KARTE-FORM (#239 rc12z29c) NICHT GEPINNT: the Solve "
            f"vor der Karte hat keinen Schnitt/Vektor veroeffentlicht (ratio {ratios}, "
            f"FR_D {fr}, Schnitt {cut}) -- die Karte liest die Vektoren, die jetzt in "
            f"--extra-d stehen")
        return None
    form = {"ratios": [str(x) for x in ratios], "fractions": [str(x) for x in fr],
            "cut": tuple(float(x) for x in cut) if cut else None}
    ns._d_map_form = form
    log(f"{D_RANK_SOLVE_MARKER} KARTE-FORM (#239 rc12z29c) GEPINNT: die Platztausch-Karte "
        f"baut Phase D aus --rank-moe-ratio {','.join(form['ratios'])}, FR_D "
        f"{','.join(form['fractions'])}, Token-Schnitt "
        f"{list(form['cut']) if form['cut'] else 'keiner'} -- dieselben "
        f"Vektoren, die D's Expertenfenster liest; jeder spaetere D-Solve prueft genau "
        f"diese Form gegen sein Budget und loest sie nicht neu")
    return form

def _proc_ancestors(pid: int, proc_root: str) -> List[int]:
    chain: List[int] = []
    seen = {pid}
    p = _proc_ppid(pid, proc_root)
    while p > 1 and p not in seen:
        chain.append(p)
        seen.add(p)
        p = _proc_ppid(p, proc_root)
    return chain

def l3_persist_identity(model: str, profile: str = "", form_kv: str = "",
                        kv_cache_dtype: str = KV_CACHE_DTYPE, extra_p: str = "",
                        extra_d: str = "", vision: str = "") -> dict:
    """L3P: what a persistent store is FOR. User 2026-09-27: "aber natürlich
    zwischen 27b und nf verschiedene L3 caches. sonst knallts" -- one model,
    one directory. Beside the checkpoint (path, config sha, weight-file stat
    fingerprint) it carries every launcher-side input that changes the BYTES
    of a page without changing its key: the KV dtype (the line's constant and
    an EXTRA override of either group), the effective
    ``--json-model-override-args`` of both groups (rope_scaling / YaRN: the
    key hashes token ids only), and the vision mode. The rank adds what only
    it knows (dtype, quantization, revision) in ``L3_RANK_IDENTITY.<group>``
    and refuses a mismatch there."""
    real = os.path.realpath(str(model or ""))
    cfg_sha = ""
    try:
        with open(model_config_path(model), "rb") as f:
            cfg_sha = hashlib.sha1(f.read()).hexdigest()
    except Exception:  # noqa: BLE001 -- a GGUF or unreadable config: the path still separates
        pass
    kv = str(kv_cache_dtype or "")
    kv_p = _l3_extra_flag(extra_p, "--kv-cache-dtype") or kv
    kv_d = _l3_extra_flag(extra_d, "--kv-cache-dtype") or kv

    def _sha(v: str) -> str:
        return hashlib.sha1(v.encode()).hexdigest()[:16] if v else ""

    ident = {
        "model_path": real,
        "model_config_sha1": cfg_sha,
        "weights_fp": l3_weights_fingerprint(model),
        "profile": str(profile or ""),
        "form_kv": str(form_kv or ""),
        "kv_cache_dtype": kv if kv_p == kv_d == kv else f"P={kv_p},D={kv_d}",
        "override_p": _sha(_l3_extra_flag(extra_p, "--json-model-override-args")),
        "override_d": _sha(_l3_extra_flag(extra_d, "--json-model-override-args")),
        "vision": str(vision or ""),
        "generation": L3_PERSIST_GENERATION,
    }
    _keying = L3_ANCHOR_KEYING_PROFILES.get(str(profile or ""))
    if _keying:
        ident["anchor_keying"] = _keying
    # YaRN x2 (28.09.): the rope the pages were rotated with, by name -- the
    # override's rope/position part (rope_scaling, rope_parameters,
    # max_position_embeddings; top level and text_config), canonical, and the
    # way the runtime applies it (L3_ROPE_APPLY: f37d83617d merges a nested
    # text_config override key by key; before it, the same string replaced the
    # sub-config). Only when an override names a rope -- a store without one
    # keeps its identity (and its directory) byte for byte.
    rope_p = _l3_rope_view(_l3_extra_flag(extra_p, "--json-model-override-args"))
    rope_d = _l3_rope_view(_l3_extra_flag(extra_d, "--json-model-override-args"))
    if rope_p or rope_d:
        ident["rope"] = rope_p if rope_p == rope_d else f"P={rope_p},D={rope_d}"
        ident["rope_apply"] = L3_ROPE_APPLY
    return ident

def l3_revoked_windows(directory: str) -> List[Tuple[float, float, str]]:
    """L3-REVOKE: the store's revoked write windows ``(from, to, reason)`` in
    unix seconds, [] without a record. An unreadable or malformed record is
    REFUSED BY NAME (W57): a revocation that cannot be read must not turn
    into silent reuse of the pages it names."""
    path = _l3_revoked_path(directory)
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        raise PdFlipStoreDiskRefused(
            f"W57 PdFlipStoreDiskRefused: L3-REVOKE record {path} unreadable "
            f"({type(exc).__name__}: {exc}); the pages it names cannot be told apart. "
            f"Fix the record (python -m flliper.srt.pdflip.l3_store_audit revoke) or move "
            f"the directory aside.")
    out: List[Tuple[float, float, str]] = []
    try:
        for w in raw["windows"]:
            t0, t1 = float(w["from_unix"]), float(w["to_unix"])
            if not t0 <= t1:
                raise ValueError(f"from {t0} after to {t1}")
            out.append((t0, t1, str(w.get("reason", ""))))
    except (KeyError, TypeError, ValueError) as exc:
        raise PdFlipStoreDiskRefused(
            f"W57 PdFlipStoreDiskRefused: L3-REVOKE record {path} malformed "
            f"({type(exc).__name__}: {exc}); expected "
            f'{{"windows": [{{"from_unix": s, "to_unix": s, "reason": "..."}}]}}.')
    return out

def l3_persist_check_rope(extra_p: str, extra_d: str) -> None:
    """L3-REVOKE companion (#259): P and D write and read the SAME pages (one
    L2, one L3; the key hashes token ids only). A rope override on one group
    and not the other (YaRN x2 on D alone) gives both groups the same keys for
    KV rotated at different positions -- refused by name before the store is
    touched, instead of a directory identity that records both and shares
    them."""
    p = _l3_rope_view(_l3_extra_flag(extra_p, "--json-model-override-args"))
    d = _l3_rope_view(_l3_extra_flag(extra_d, "--json-model-override-args"))
    if p != d:
        raise PdFlipStoreDiskRefused(
            f"W57 PdFlipStoreDiskRefused: L3-PERSIST P and D rotate differently "
            f"(P rope {p or 'config'} vs D rope {d or 'config'}); they share every "
            f"L2/L3 page by token-id key, so one group would read the other's KV at "
            f"the wrong positions. Give both --extra-p and --extra-d the same rope "
            f"override, or set FLLIPER_PDFLIP_L3_PERSIST=0.")

def p_micro_batch_flags(p_bs: int, extra: Sequence[str]) -> List[str]:
    """fnFL2 H37: group P's ``--pp-max-micro-batch-size``, or nothing.

    WHAT IT LIFTS. Unset, the scheduler derives the width as
    ``max_running_requests // pp_size`` (scheduler.default_pp_micro_batch_size)
    and ``get_num_allocatable_reqs`` caps every forward at it -- on PP3 that is
    ONE request per forward for ``--p-bs`` 1..5, so ``--p-bs 3`` buys three
    requests in flight on three stages but never two in one forward (and every
    END-ANCHOR tail of 1-4 tokens runs as a forward of its own). For a
    prefill-only group whose requests finish at the end of their prefill, the
    forward is bounded by the chunk TOKEN budget (``rem_chunk_tokens`` is a
    per-batch budget in ``PrefillAdder``), not by a request count; the total
    stays bounded by ``req_to_token_pool`` (= ``--max-running-requests``).

    The width is P's EFFECTIVE ``--max-running-requests`` -- the one argparse
    keeps, i.e. an ``--extra-p`` value if the arm doubled the flag -- so the two
    never disagree. An ``--extra-p`` ``--pp-max-micro-batch-size`` wins and
    nothing is emitted (same rule as ``--max-mamba-cache-size``).

    Off (the default): ``[]``, the argv is byte-identical to before H37.
    """
    if not envs.FLLIPER_PDFLIP_ENABLE_P_UNDIVIDED_MICRO_BATCH.get():
        return []
    if _last_flag_value(extra, P_MICRO_BATCH_FLAG) is not None:
        return []
    mrr = _last_flag_value(extra, "--max-running-requests")
    width = int(mrr) if mrr is not None else int(p_bs)
    return [P_MICRO_BATCH_FLAG, str(max(1, width))]

def p_micro_batch_line(argv_of_p: Sequence[str]) -> str:
    """fnFL2 H37: ONE line at launch that says how many requests one P
    forward may carry, and why -- the reading the burst probe is checked
    against (``Prefill batch, #new-seq: k`` on PP0 cannot exceed it).

    Also names a DOUBLED ``--max-running-requests`` (launcher value and an
    ``--extra-p`` value that differ): argparse keeps the last, so ``--p-bs``
    then no longer states P's concurrency -- the front still dispatches
    ``--p-bs`` + P_QUEUE_AHEAD.
    """
    toks = [str(t) for t in argv_of_p]
    mrr_all = [toks[i + 1] for i, t in enumerate(toks)
               if t == "--max-running-requests" and i + 1 < len(toks)]
    mrr_all += [t.split("=", 1)[1] for t in toks if t.startswith("--max-running-requests=")]
    try:
        mrr = int(_last_flag_value(toks, "--max-running-requests") or 0)
        pp = int(_last_flag_value(toks, "--pp-size") or 1)
    except ValueError:
        return ("P-MICRO-BATCH unreadable argv "
                "(--max-running-requests/--pp-size not integers)")
    stock = max(mrr // max(pp, 1), 1) if mrr > 0 else 1
    doubled = (f" DOUBLED --max-running-requests {'/'.join(mrr_all)}: argparse keeps {mrr}"
               if len(set(mrr_all)) > 1 else "")
    mb = _last_flag_value(toks, P_MICRO_BATCH_FLAG)
    if mb is None:
        return (f"P-MICRO-BATCH mode=stock width={stock} (= --max-running-requests "
                f"{mrr} // pp {pp}): one P forward admits at most {stock} request(s); "
                f"FLLIPER_PDFLIP_ENABLE_P_UNDIVIDED_MICRO_BATCH=1 lifts it to {max(mrr, 1)}{doubled}")
    if envs.FLLIPER_PDFLIP_ENABLE_P_MULTI_ANCHOR_TAILS.get():
        # fnFL2 H42: every whole-fit prompt's END-ANCHOR body becomes a tail
        # of its own (managers/anchor_tails.py); the tails end together.
        tails = ("FLLIPER_PDFLIP_ENABLE_P_MULTI_ANCHOR_TAILS=1: every END-ANCHOR body is an "
                 "anchor tail of its own request, so up to that many requests REACH THEIR END "
                 "per forward (their tails [N-1', N) ride together in the next pass)")
    else:
        tails = ("END-ANCHOR still mints ONE continuation per pass (#959/#996), so at "
                 "most one request REACHES ITS END per forward "
                 "(FLLIPER_PDFLIP_ENABLE_P_MULTI_ANCHOR_TAILS=1 lifts it)")
    return (f"P-MICRO-BATCH mode=undivided width={mb} (--max-running-requests {mrr}, "
            f"stock would be {stock}): one P forward admits up to {mb} request(s) within the "
            f"chunk budget; {tails}{doubled}")

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
    ``with self._pdflip_pcie_lock(...)`` around the pause loop, wake-H2D around
    the resume loop), and on a card whose measured duplex ratio does not reach
    R17's gate both directions resolve to the SAME key.  There the gathered pair
    is serialised again -- W's per-tag releases cannot fund S's acquires,
    because W is queued behind S on the key, or S behind W.

    The predecessor computed the AND over cards, printed it, and gated nothing;
    its justification was that "a card that keeps one key costs time on that
    card, and nothing else".  The metal refuted that 20 minutes later: nvml0
    (1.316, DUPLEX-NULL) passed R5 on both directions, armed, and wedged
    120.189 s into the FIRST flip -- W31 PdFlipHostRingExhausted need=24 free=12
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
    ``pdflip_memory_saver`` cannot see a stale worktree on ``PYTHONPATH`` or a
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
                f"PDFLIP-HOST-RING KEY-PROBE skipped: leg_form={leg_form!r} is not "
                "'interleave', so every card is serialised whatever its key does")
        return list(card_uuids)
    if py and tree:
        counts, why = _resolve_keys_in_rank_tree(card_uuids, published, py, tree)
        if not counts:
            if notes is not None:
                notes.append(
                    f"PDFLIP-HOST-RING KEY-PROBE FAILED in the RANK TREE {tree} "
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
                f"PDFLIP-HOST-RING KEY-PROBE resolver=RANK TREE {tree} ({py}) "
                + ", ".join(
                    f"{u}={counts.get(u, (0, 0))[0]}/{counts.get(u, (0, 0))[1]} "
                    "key(s) per direction" for u in card_uuids)
                + f" -- serialised={serialised or 'none'}")
        return serialised
    from flliper.srt.managers import pdflip_memory_saver

    if notes is not None:
        notes.append(
            "PDFLIP-HOST-RING KEY-PROBE resolver=IN-PROCESS (desk path: no --tree "
            "given) -- this cannot see a stale tree on the ranks' PYTHONPATH")
    serialised: List[str] = []
    saved = os.environ.get(pdflip_memory_saver.PCIE_DUPLEX_ENV)
    os.environ[pdflip_memory_saver.PCIE_DUPLEX_ENV] = published
    try:
        for uuid in card_uuids:
            try:
                keys = {
                    pdflip_memory_saver.pcie_lock_path(uuid, direction=d)
                    for d in pdflip_memory_saver.PCIE_DIRECTIONS
                }
            except TypeError:
                # A saver whose pcie_lock_path has no direction parameter at
                # all: one key, both legs, by construction.
                keys = {pdflip_memory_saver.pcie_lock_path(uuid)}
            except getattr(pdflip_memory_saver, "PdFlipDuplexDecisionRefused", ()):
                # W36 raised while resolving.  It is already a named refusal;
                # here it means SERIALISED, so the launcher's own W34 arm gives
                # the operator the whole picture in one refusal instead of a
                # traceback out of a helper.
                keys = {uuid}
            if len(keys) < len(pdflip_memory_saver.PCIE_DIRECTIONS):
                serialised.append(uuid)
    finally:
        if saved is None:
            os.environ.pop(pdflip_memory_saver.PCIE_DUPLEX_ENV, None)
        else:
            os.environ[pdflip_memory_saver.PCIE_DUPLEX_ENV] = saved
    return serialised

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
    ``.pdflip-vram-credit-*`` in the directory, so a teardown of boot A would
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

def p_covers_d_session(argv_p_shipped: Sequence[str], d_session: int) -> Tuple[bool, str]:
    """LAW (user 02.10. ~07:55Z): "kann nie passieren, weil P ja bis 262k
    prefillen kann und eine einzelne D session nur maximal 262k haben kann.
    das selbe gilt fuer yarn ... deswegen ja die stufen in P". P's KV stage
    (its ``--max-total-tokens`` pool) and its per-request cap
    (``--max-kv-per-request``) must each hold D's maximum session, so no
    request is ever D's to prefill for a capacity reason (the CARRIER-EXCEEDS
    route is deleted). Returns ``(ok, line)``; a term the shipped P argv does
    not name is printed as unnamed, never assumed."""
    d = int(d_session)
    cap = _last_flag_value(argv_p_shipped, "--max-kv-per-request")
    pool = _last_flag_value(argv_p_shipped, "--max-total-tokens")
    short = []
    for name, raw in (("P --max-kv-per-request", cap), ("P --max-total-tokens (KV stage)", pool)):
        if raw is None:
            continue
        try:
            if int(raw) < d:
                short.append(f"{name}={int(raw)} < D session {d}")
        except ValueError:
            continue
    line = (f"{P_COVERS_D_SESSION_MARK} D session (--max-kv-per-request)={d} "
            f"P cap={cap if cap is not None else 'unnamed'} "
            f"P stage pool={pool if pool is not None else 'unnamed'}")
    if short:
        return False, (line + " REFUSED: " + "; ".join(short) + " -- P must prefill every "
                       "request D can hold (no D prefill for capacity, law 02.10.); raise P's "
                       "stage (YaRN 524k / 786k) or lower D's --max-kv-per-request")
    return True, line + " ok"

def _p_gguf_dense_vocab(ns, model: Optional[str]) -> bool:
    """True when group P loads a GGUF checkpoint with the legacy dense vocab
    (``model_loader.gguf_qwen35.gguf_dense_vocab``: token_embd/output are
    dequantized to BF16 at load) -- read where P reads it: the exported env,
    overridden by an ``--env-p`` entry.  The name is restated here only so this
    module does not import torch; the reader's spelling is pinned by a test."""
    if not model or not os.path.isfile(str(model)):
        return False
    from flliper.srt.pdflip import gguf_census as _gguf_census

    if not _gguf_census.is_gguf_checkpoint(str(model)):
        return False
    val = os.environ.get(GGUF_DENSE_VOCAB_ENV, "0")
    try:
        val = parse_group_env(str(getattr(ns, "env_p", "") or "")).get(GGUF_DENSE_VOCAB_ENV, val)
    except ValueError:
        pass
    return str(val).strip() == "1"

def lane_coverage_boot_token(ns) -> str:
    """#1348: WHICH BOOT this run's coverage dumps belong to, or ``""`` when
    the coverage instrument itself is not armed.

    The GATE stays exactly as it was -- `FLLIPER_PDFLIP_LANE_COVERAGE_TOKEN`'s
    presence in a rank's environment is what tells `lane_coverage.py`
    whether to trace lines at all, so an ambient value in the operator's own
    shell must never arm it by accident (that popping discipline is why the
    switch is a flag and not an ambient variable in the first place). The
    VALUE, once armed, is :func:`pdflip_boot_token`'s -- one producer, so a
    boot's coverage dumps and its phase-footprint dumps (#1395) agree on
    which boot they belong to when both instruments are armed together.
    """
    if not getattr(ns, "xchg_coverage_diff", False):
        return ""
    return pdflip_boot_token(ns)

def refusal_cause(e: BaseException) -> dict:
    """A launcher refusal as §2.2 cause: code = W-number + name (never the bare
    W-number, 27B requirement 6), origin launcher, the full text."""
    msg = str(e)
    m = _CAUSE_RE.search(msg)
    if m:
        code, name = f"{m.group(1)}_{m.group(2)}", m.group(2)
    else:
        name = type(e).__name__
        code = f"{_w_code_of(e)}_{name}"
    return state_file_mod.make_cause(code, "launcher", msg, name=name, exception_type=type(e).__name__)

def _model_config(model: str) -> dict:
    with open(model_config_path(model)) as f:
        return json.load(f)

def refuse_d_early_start_unreviewed(ns) -> None:
    """W185, called before group P starts: a named refusal, never a silent off.

    27B (29.09.): with ``d_expect_from_p_records`` the 27B plans the early D
    from P's records, so an explicit 'on' boots (the proof boot); 'auto'
    stays off there until ``d_early_start_proven``. The 27B review's reader
    concern (ns.env_d/_d_owner_solve between launch_group(P) and #5) holds on
    this tree without a reader: only the early branch itself and the
    refusal's restore touch them in that window."""
    if (str(getattr(ns, "pdflip_d_early_start", "off")) == "on"
            and not d_expect_from_p_records(ns.profile)):
        raise PdFlipDEarlyStartUnreviewed(
            f"W185 PdFlipDEarlyStartUnreviewed: --pdflip-d-early-start on with profile "
            f"{ns.profile or pdflip_form.DEFAULT_PROFILE!r}, whose D expectation is the legacy "
            f"term (registry d_expect_from_p_records=False): the early D would be planned "
            f"from D's own reserve, not P's measured residue. Boot with 'off' (byte-identical).")

def p_seats(ns) -> int:
    """Group P's SEATS = its effective ``--max-running-requests``: a value in
    ``--extra-p`` beats ``--p-bs`` (argparse keeps the last; the same rule as
    ``p_micro_batch_flags``). One reader for the P card and the mamba floor."""
    _mrr = _argv_scalar(getattr(ns, "extra_p", ""), "--max-running-requests")
    try:
        return int(_mrr) if _mrr is not None else int(getattr(ns, "p_bs", DEFAULT_P_BS) or DEFAULT_P_BS)
    except (TypeError, ValueError):
        return int(getattr(ns, "p_bs", DEFAULT_P_BS) or DEFAULT_P_BS)

def p_host_overlap_lines(overlap: bool, hostgap: bool) -> List[str]:
    """The provenance lines for the two switches -- none when both are off."""
    lines: List[str] = []
    if overlap:
        lines.append(
            "PDFLIP P-HOST-OVERLAP: on (--p-host-overlap) -- group P gets %s: a "
            "middle prefill chunk exchanges no output (upstream's pure-chunk "
            "skip), the last rank fences its schedule stream on each forward "
            "device-side, and the chunked-prefill HiCache publish runs after "
            "the next launch instead of inside the plan. Reason: weg2xsn420 "
            "4x98k PP2 pass 873 ms against 733 gpu-ms, ~95-140 ms card idle "
            "per 4096 chunk from host work serialised behind the stage's own "
            "forward (managers/pdflip_p_overlap.py)."
            % " ".join("%s=%s" % kv for kv in sorted(p_host_overlap_env(True, False).items()))
        )
    if hostgap:
        lines.append(
            "PDFLIP P-HOSTGAP: on (--p-hostgap) -- group P prints one #PGAP line "
            "per forward: gpu_gap_ms = card idle before it (timing events, no "
            "sync), host phases since the previous launch."
        )
    return lines

def p_recv_stream_lines(on: bool) -> List[str]:
    """The provenance line for ``--p-recv-stream`` -- none when off."""
    if not on:
        return []
    return [
        "PDFLIP P-RECV-STREAM: on (--p-recv-stream) -- group P gets %s: the proxy "
        "frame's NCCL receive runs on a side stream without the fence on the "
        "running forward, the schedule stream waits for it device-side. Reason: "
        "#PGAP 26.09. (i8B/i8drt) PP1/PP2 card idle per chunk = frame transfer "
        "serialised behind the previous forward (PP2 2048: ~11 ms of ~462 ms, "
        "0.1 ms on PP0 which receives no frame) (managers/pdflip_p_overlap.py)."
        % " ".join("%s=%s" % kv for kv in sorted(p_recv_stream_env(True).items()))
    ]

def newest_bubble_log(
    evidence_dir: str, accept: Optional[Callable[[str], bool]] = None
) -> Optional[str]:
    """Newest ``*.P.log`` in ``evidence_dir`` that actually CARRIES the line.

    ``accept`` (PDFLIP-FORM, 24.09.): only logs of this boot's checkpoint
    (``pdflip_form.same_model_log``); None = every log, as before.

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
        if accept is not None and not accept(path):
            continue
        if read_pp_bubble(path) is not None:
            return path
    return None

def newest_prefill_census_log(
    evidence_dir: str, accept: Optional[Callable[[str], bool]] = None
) -> Optional[str]:
    """Newest ``*.P.log`` that actually CARRIES a prefill census.

    ``accept``: as :func:`newest_bubble_log` (PDFLIP-FORM calibration identity).

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
        if accept is not None and not accept(path):
            continue
        if read_mean_prefill_prefix(path) is not None:
            return path
    return None

def region_form_lane_price_cap(d_vector: str = XCHG_D_VECTOR_DEFAULT,
                               legs: str = "both",
                               n_stages: Optional[int] = None) -> int:
    """#1464b: the lane count the ledger charges in the REGION form for EVERY
    cut -- the measured record of the ordered default cut (39,13,12 -> 5),
    0 when that record does not exist (then nothing is capped and the old
    per-cut count prices, as before)."""
    if n_stages is not None and pdflip_pp_ordered_cut_for(n_stages) is None:
        # HW-P1c 1003: the incumbent record is the THREE-stage ordered cut's;
        # another stage count has no incumbent -> its own worst-case lane count
        # (host_ledger.resolve_xchg_lanes, N_PAIRS + N_CARDS) stands.
        return 0
    rec = host_ledger.XCHG_LANES_BY_CUT.get(
        host_ledger.xchg_cut_key(_csv(DEFAULT_PP_ORDERED_CUT), str(d_vector), str(legs)))
    try:
        return max(0, int((rec or {}).get("lanes", 0) or 0))
    except (TypeError, ValueError):
        return 0

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
    raise PdFlipLaunchRefused(
        "W40 PdFlipPPCutRefused: --pp-solve-objective incumbent asks for the cut "
        f"{','.join(str(n) for n in want_layers)} / attention "
        f"{','.join(str(a) for a in want_attn)}, and the solver did not rank it "
        f"as a choosable candidate ({len(decision.ranked)} ranked). Refusing "
        "rather than shipping the ranking's winner under the incumbent's name "
        "-- that substitution is exactly what would desynchronise argv from the "
        "flip-order map."
    )

def refuse_d_form_off_the_map(ns, log) -> None:
    """W170: the D form a D start carries must be the map's (#239 rc12z29c).

    Called right before every D env/argv is built (d-only, dry run, real). A
    no-op without a published map. Reads what D will actually read: the
    ratio and FR in --extra-d and FR in --env-d. Whatever moved it after the
    map (a later solve, the P1c FR-D ceiling, an operator edit) is refused by
    name here, never discovered by D's loader."""
    form = getattr(ns, "_expert_map_d_form", None)
    if not form:
        return
    ratios = _argv_vector(getattr(ns, "extra_d", ""), "--rank-moe-ratio")
    fr = _argv_vector(getattr(ns, "extra_d", ""), "--rank-moe-resident-fraction")
    env_fr_text = parse_group_env(getattr(ns, "env_d", "") or "").get(D_FR_ENV)
    moved = []
    if not _same_vector(ratios or [], form["ratios"]):
        moved.append(f"--rank-moe-ratio {ratios} (Karte {form['ratios']})")
    if not _same_vector(fr or [], form["fractions"]):
        moved.append(f"--rank-moe-resident-fraction {fr} (Karte {form['fractions']})")
    if env_fr_text is not None and not _same_vector(
            [x for x in str(env_fr_text).split(",") if x.strip()], form["fractions"]):
        moved.append(f"--env-d {D_FR_ENV}={env_fr_text} (Karte {form['fractions']})")
    if moved:
        raise PdFlipDFormOffTheMap(
            "W170 PdFlipDFormOffTheMap: D would start with a form the Platztausch map "
            "does not describe -- " + "; ".join(moved) + ". P loaded its prefix and "
            "store slots from the map; D would die at load. Fix the solve that "
            "moved it, not the map (#239 rc12z29c)")
    log(f"KARTE-FORM (#239 rc12z29c) D-START geprueft: --rank-moe-ratio "
        f"{','.join(form['ratios'])}, FR_D {','.join(form['fractions'])} = die Form der Karte")

def _restage_p_card_reference(ns, cards, reference, support, co_tenant, stage_layers, log, *,
                              layer_kinds, dense_mib_per_layer, mamba_mib_per_linear_layer_per_slot):
    """AP2 1006 (W167 at N != 3): the P-card reference (built in, or from
    ``--p-card-reference-logs``) was measured on its own stage count -- the NF
    built-in one on 3 stages of [RTX5090, RTX3080, RTX3080]. A launch of another
    card count used to stop at W167 ("Stufenzahl 2 gegen 3") even with --force.

    Now the reference is RE-STAGED (``p_card_chunk.restage_reference``): each
    live stage takes the measured row of the reference stage of the SAME ROLE
    (first / middle / last, ``role_stage_map``) and its K0 is shifted by its
    layer count exactly as a same-count recut. When every live stage sits on a
    card of the class its reference stage was measured on (5090 + 3080 from the
    reference rig) the row is the class's own measurement: one ``PP-CUT RESTAGE``
    line, DERIVED. When a stage borrows another class's row (a middle 5090 of
    2 x 5090 + 2 x 3080, any 3090) it is the value refusal HW-UNCALIBRATED:
    without ``--force`` refused by name as before, with it FORCED-PAST and named
    UNMEASURED. A reference that cannot be re-staged (layer types / Dense /
    Mamba price unknown, one stage) stays W167."""
    smap = _p_card.role_stage_map(len(stage_layers), len(reference.stage_layers))
    if smap is None:
        _p_card.recut_check(reference.stage_layers, stage_layers, layer_kinds, _p_card.CARD_MARKER)
        raise _p_card.PCutRecutRefused(   # pragma: no cover - recut_check refuses every count mismatch
            f"{_p_card.RECUT_REFUSAL_CODE}: {_p_card.CARD_MARKER} -- no stage role to place")
    ref_new, sup_new, co_new = _p_card.restage_reference(
        reference, support, co_tenant, stage_layers, smap, layer_kinds=layer_kinds,
        dense_mib_per_layer=dense_mib_per_layer,
        mamba_mib_per_linear_layer_per_slot=mamba_mib_per_linear_layer_per_slot)
    prof = str(getattr(ns, "profile", None) or pdflip_form.DEFAULT_PROFILE)
    ref_classes, _ = records_inventory(prof)
    ref_classes = tuple(ref_classes) if len(ref_classes) == len(reference.stage_layers) else ()
    live = [card_identity.class_label(cards[s]) if s < len(cards) else "?" for s in range(len(stage_layers))]
    rows = "; ".join(
        f"stage{s} ({live[s]}) <- reference stage{m} ({ref_classes[m] if ref_classes else '?'}, "
        f"{int(reference.stage_layers[m])} -> {int(stage_layers[s])} layers, K0 "
        f"{float(reference.headroom0_mib[m]):.0f} -> {float(ref_new.headroom0_mib[s]):.0f} MiB)"
        for s, m in enumerate(smap))
    head = (f"{_p_card.CARD_MARKER} RESTAGE ref={_p_card.split_text(reference.stage_layers)} "
            f"({reference.source}) -> {_p_card.split_text(stage_layers)}: {rows}")
    borrowed = [s for s, m in enumerate(smap) if not ref_classes or live[s] != ref_classes[m]]
    if not borrowed:
        log(head + " -- every stage on a card of the class its reference stage was measured on: "
            "DERIVED by role and layer count, not measured (the first boot of this cut is the measurement)")
        return ref_new, sup_new, co_new
    refusals.refuse_value(
        card_identity.CODE_UNCALIBRATED,
        f"{_p_card.RECUT_REFUSAL_CODE} {card_identity.CODE_UNCALIBRATED}: {head} -- stage(s) "
        + ", ".join(str(s) for s in borrowed)
        + " sit on a card of ANOTHER class than the reference stage they take the row from "
        f"(reference cards [{', '.join(ref_classes) or 'unknown'}]); --force BORROWS that row: UNMEASURED on "
        "this card. Without --force: give boots of this cut on these cards (--p-card-reference-logs).",
        PdFlipLaunchRefused, log=log)
    return ref_new, sup_new, co_new

def profile_standard_form(ns) -> bool:
    """UNIFY (operator 26.09.): the NF H91 standard form for this launch (D
    seats 1..6 per phase, pool waves, seat posts): an explicit
    FLLIPER_PDFLIP_STANDARD_FORM in the launcher's env wins (build_env hands the
    same env to P, D and the front), else the registry row's
    ``standard_form`` for ``ns.profile``; an unknown profile is off."""
    return standard_form_resolved(ns)[0]

def prefix_switches_announce(ns, boot_form, environ: Optional[Mapping[str, str]] = None) -> str:
    """RG (operator 26.09.): the PDFLIP-PREFIX-SWITCHES line of this boot (value
    and source -- env or profile -- per prefix switch, pdflip/form.py
    PREFIX_SWITCHES), computed on a COPY of the launcher env: the writers are
    build_env (P, D) and the front spawn. Refuses (PdFlipLaunchRefused) when
    --env-p/--env-d would run MZ differently on P and D."""
    env = dict(os.environ if environ is None else environ)
    rows = pdflip_form.publish_prefix_switches(env, boot_form.profile)
    split = pdflip_form.prefix_p_eq_d_mismatch(
        env, parse_group_env(getattr(ns, "env_p", "") or ""),
        parse_group_env(getattr(ns, "env_d", "") or ""))
    if split:
        raise PdFlipLaunchRefused(split)
    return pdflip_form.prefix_switches_line(rows)

def mamba_arena_group_floor(extra: str, bs_fallback) -> Tuple[int, int, int, str]:
    """(floor, device_slots, max_running, source) of ONE group's share of the mamba
    arena: device_slots + max_running + 1 -- the host anchor pool's law
    (memory_pool_host.mamba_anchor_slot_floor), applied to the arena every
    rank of both groups shares. device_slots = the group's own
    --max-mamba-cache-size when its argv names one (the last value, as argparse),
    else the upper bound of the demand-driven pool. (0, 0, 0, why) when the
    group's concurrency is not knowable here."""
    bs = _argv_scalar(extra or "", "--max-running-requests")
    try:
        bs = int(bs) if bs is not None else int(bs_fallback or 0)
    except (TypeError, ValueError):
        bs = 0
    if bs <= 0:
        return 0, 0, 0, "max_running_requests unknown"
    mmc = _argv_scalar(extra or "", "--max-mamba-cache-size")
    try:
        dev, src = int(mmc), "--max-mamba-cache-size"
    except (TypeError, ValueError):
        dev = int(math.ceil(bs * MAMBA_ARENA_SLOTS_PER_REQUEST_UPPER * P_MAMBA_AUTO_SAFETY_MARGIN))
        src = "demand upper bound ceil(%d x %d x %.2f)" % (bs, MAMBA_ARENA_SLOTS_PER_REQUEST_UPPER, P_MAMBA_AUTO_SAFETY_MARGIN)
    return dev + bs + 1, dev, bs, src
