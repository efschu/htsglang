"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import os

from flliper.srt.pdflip import DEFAULT_D_BS
from flliper.srt.pdflip import budget_rest as _budget_rest
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip.launcher_d_leaf import FR_D_CEILING_ENV
from flliper.srt.pdflip.launcher_d_leaf import KV_TOKEN_CUT_MARKER
from flliper.srt.pdflip.launcher_d_leaf import TORCH_CACHE_CAP_ENV
from flliper.srt.pdflip.launcher_d_leaf import _argv_int_list
from flliper.srt.pdflip.launcher_d_leaf import _argv_scalar
from flliper.srt.pdflip.launcher_d_leaf import _argv_vector
from flliper.srt.pdflip.launcher_d_mid import CONTEXT_LENGTH_TOKENS
from flliper.srt.pdflip.launcher_d_mid import D_RANK_SOLVE_MARKER
from flliper.srt.pdflip.launcher_d_mid import census_checkpoint
from flliper.srt.pdflip.launcher_d_mid import checkpoint_quant_method
from flliper.srt.pdflip.launcher_d_mid import d_draft_share_embed
from flliper.srt.pdflip.launcher_d_mid import d_early_start_proven
from flliper.srt.pdflip.launcher_d_mid import d_effective_seats
from flliper.srt.pdflip.launcher_d_mid import d_flip_trim_lines
from flliper.srt.pdflip.launcher_d_mid import d_form_solved_by_planner
from flliper.srt.pdflip.launcher_d_mid import d_kv_stage_tokens_named
from flliper.srt.pdflip.launcher_d_mid import group_context_tokens
from flliper.srt.pdflip.launcher_p_l3_leaf import CHUNKED_PREFILL_TOKENS
from flliper.srt.pdflip.launcher_p_l3_leaf import LAUNCH_SERVER_MODULES
from flliper.srt.pdflip.launcher_p_l3_leaf import P_CHUNK_BUILTIN_INT8
from flliper.srt.pdflip.launcher_p_l3_leaf import P_CHUNK_MAX_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import WEIGHT_SOURCE_EXCHANGE
from flliper.srt.pdflip.launcher_p_l3_leaf import _group_env_value
from flliper.srt.pdflip.launcher_p_l3_leaf import _int_list
from flliper.srt.pdflip.launcher_p_l3_leaf import gguf_backbone_depth
from flliper.srt.pdflip.launcher_p_l3_leaf import is_launch_server_argv
from flliper.srt.pdflip.launcher_p_l3_leaf import model_config_path
from flliper.srt.pdflip.launcher_p_l3_leaf import model_is_gguf_file
from flliper.srt.pdflip.launcher_p_l3_leaf import p_chunk_policy_spec
from flliper.srt.pdflip.launcher_p_l3_leaf import p_prefill_graph_bucket
from flliper.srt.pdflip.launcher_p_l3_leaf import parse_group_env
from flliper.srt.pdflip.launcher_p_mid import _model_config
from flliper.srt.pdflip.launcher_p_mid import _proc_ancestors
from flliper.srt.pdflip.launcher_p_mid import mamba_arena_group_floor
from flliper.srt.pdflip.launcher_p_mid import p_prefill_graph_buckets
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _proc_argv
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _proc_boot_tag
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _proc_has_cuda_context
from flliper.srt.pdflip.launcher_refuse_proc_leaf import set_group_env
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_vision_spec_mid import tokenizer_path_decision
from flliper.srt.pdflip.launcher_xchg_vision_leaf import _TOKENIZER_PATH
import json
import shlex

#: the measured x4-3080 BAR1 rate that bounds both 27B P edges (card probe
#: 14.4 / 6.5 / 13.3 GB/s) -- the planner's price of a pull
P_LAYER_SPLIT_PULL_GBPS_DEFAULT = 6.5
DRAFT_PARK_UNPRICED = "W128 PdFlipDraftParkUnpriced"
#: --p-attn-head-split (27B, release table row 22, ATTN_HEAD_SPLIT.md phase 2):
#: 'off' (default) = no env, the pool model and argv byte-identical.
P_ATTN_HEAD_SPLIT_DEFAULT = "off"
#: A split chunk must be EAGER: min_w above every captured prefill bucket.
P_ATTN_HEAD_SPLIT_MIN_W_DEFAULT = 1024
#: --p-prefill-graph-split (27B line, 2026-09-24): 0 = off, the default.
P_PREFILL_GRAPH_SPLIT_DEFAULT = 0
#: 27B int4ar 23.10.: the line of the flip arm's trim threshold
EXTEND_TRIM_FLIP_MARKER = "EXTEND-TRIM-FLIP"
EXTEND_TRIM_ENV = "FLLIPER_PDFLIP_EXTEND_TRIM_MIB"
#: #201: the per-rank provenance of the flip arm's trim threshold (card law or the named stated-law fallback)
TRIM_LAW_MARKER = "TRIM-LAW"
MAMBA_ARENA_ENV = "FLLIPER_HICACHE_ARENA_MAMBA_SLOTS"

def p_layer_split_spec_from_ns(ns):
    """The SplitSpec a 'dynamic' boot would hand group P (no side effects).
    The stage model is SG's per-layer-type model (pdflip/p_stage_model.py):
    a JSON path or a name under p_stage_model_data/ (27b_int8_rc9j,
    27b_nvfp4_rc9j)."""
    from flliper.srt.pdflip import p_chunk_policy as _pcp
    from flliper.srt.pdflip import p_layer_split as _pls
    from flliper.srt.pdflip import p_stage_model as _psm

    home = _int_list(getattr(ns, "p_layer_split_home", "") or "", "--p-layer-split-home")
    window = _int_list(getattr(ns, "p_layer_split_window", "") or "", "--p-layer-split-window")
    max_attn = _int_list(getattr(ns, "p_layer_split_max_attn", "") or "", "--p-layer-split-max-attn")
    src = str(getattr(ns, "p_layer_split_model", "") or "")
    if not home or not window or not src:
        raise SystemExit("--p-layer-split dynamic needs --p-layer-split-home, "
                         "--p-layer-split-window and --p-layer-split-model")
    path = src if os.path.sep in src or src.endswith(".json") else os.path.join(
        os.path.dirname(_psm.__file__), "p_stage_model_data", src + ".json")
    try:
        model = _psm.load_model(path)
    except (OSError, ValueError, KeyError, _psm.StageModelError) as exc:
        raise SystemExit(f"--p-layer-split-model {src}: {exc}")
    floors = tuple(float(r["eager_floor_ms"]) / n for r, n in zip(P_CHUNK_BUILTIN_INT8, (42, 11, 11)))
    try:
        geom = _pls.SplitGeometry(_pls.hybrid_families(model.n_layers, 4), home, window, max_attn)
        chunk = p_chunk_policy_spec()
        limits = chunk.limits if chunk is not None else _pcp.ChunkLimits(
            max_tokens=P_CHUNK_MAX_DEFAULT, min_tokens=p_prefill_graph_bucket() or 512,
            fixed_tokens=p_prefill_graph_bucket() or 512, page=1,
            graph_buckets=tuple(p_prefill_graph_buckets()), eager=True)
        _pull_raw = getattr(ns, "p_layer_split_pull_gbps", None)
        pull = P_LAYER_SPLIT_PULL_GBPS_DEFAULT if _pull_raw is None else float(_pull_raw)
        extra = _pls.ExtraCosts(
            _pls.WritebackPrice(2048, 1634304, tuple(6.5 for _ in home), True),
            floors[: len(home) + 1], tuple(pull for _ in home) if pull > 0 else ())
        spec = _pls.SplitSpec(geom, model, limits,
                              float(getattr(ns, "p_layer_split_min_gain", _pls.DEFAULT_MIN_GAIN)),
                              "every", os.path.basename(path), extra)
        _pls.check_layout(geom, model)
        return spec
    except (_pls.LayerSplitError, _pcp.ChunkPolicyError) as exc:
        raise SystemExit(f"--p-layer-split dynamic: {exc}")

def d_reshard_format(model_path: str) -> str:
    """'int8' | 'nvfp4' | '' -- the checkpoint formats the D model is calibrated for."""
    qm = checkpoint_quant_method(str(model_path))
    if qm == "compressed-tensors":
        return "int8"
    if qm == "modelopt":
        return "nvfp4"
    return ""

def d_stated_seats(ns) -> Optional[int]:
    """H91b: D's seats when this namespace STATES them (the launcher's own
    parse always does: --d-bs has a default; --extra-d may carry
    --max-running-requests), else None -- a hand-built namespace that says
    nothing about seats is priced as its reference was, byte-identical."""
    extra = str(getattr(ns, "extra_d", "") or "")
    if not hasattr(ns, "d_bs") and _argv_scalar(extra, "--max-running-requests") is None:
        return None
    return d_effective_seats(ns)

def d_seat_lines(ns) -> List[str]:
    """H91b: what the seat count of this boot implies for D's argv, named.

    * --d-bs (the front's D seats) against D's effective
      --max-running-requests -- unequal means the front hands D requests it
      queues, or leaves a seat idle;
    * --cuda-graph-bs-decode must capture every batch 1..seats, else the
      uncovered batch runs EAGER (correct, slow: the pool step's host
      rendezvous per layer).
    Findings, not refusals: the boot still starts."""
    from flliper.srt.pdflip.d_seats import graph_bs_covers

    extra = str(getattr(ns, "extra_d", "") or "")
    d_bs = int(getattr(ns, "d_bs", DEFAULT_D_BS) or DEFAULT_D_BS)
    seats = d_effective_seats(ns)
    graph = _argv_int_list(extra, "--cuda-graph-bs-decode")
    out = [
        "D-SITZE (H91b): --d-bs %d (Front-Sitze), D --max-running-requests %d (wirksam), "
        "--cuda-graph-bs-decode %s%s"
        % (d_bs, seats, "Runtime-Default" if graph is None else " ".join(str(b) for b in graph),
           # H95: the seat count is the phase's handoff_n, this is its bound
           (" -- H95: %d ist die OBERGRENZE, D faehrt je Phase n = 1..%d Sitze "
            "(handoff_n beim Flip P->D, PDFLIP D-PHASE-SEATS), die Sitz-Posten sind "
            "fuer %d gebucht" % (seats, seats, seats)) if seats > 1 else "")
    ]
    if seats != d_bs:
        out.append(
            "D-SITZE (H91b) BEFUND: --d-bs %d != D --max-running-requests %d (--extra-d "
            "gewinnt) -- die Front und D zaehlen verschiedene Sitze" % (d_bs, seats)
        )
    if not graph_bs_covers(graph, seats):
        out.append(
            "D-SITZE (H91b) BEFUND: --cuda-graph-bs-decode %s faengt nicht jede Batch "
            "1..%d ein -- die fehlende laeuft EAGER; fuer %d Sitze '--cuda-graph-bs-decode %s'"
            % (
                " ".join(str(b) for b in graph),
                seats,
                seats,
                " ".join(str(b) for b in range(1, seats + 1)),
            )
        )
    return out

def d_draft_park_term(ns, draft_on_p: bool) -> Tuple[float, str]:
    """``(MiB, provenance)`` of D's parked draft (ledger post ``d_draft_host``).

    Non-zero only in the one form where D parks: the exchange arm (under
    ``ring`` the draft rides the base tag's ring backup), weights flipping
    (under ``resident`` nothing sleeps) and group P without a draft.  Priced
    from the draft checkpoint's own headers (``draft_post.d_draft_host_mib``).
    D with a draft that cannot be priced is refused by name (W128): an
    unpriced pinned image is host RAM the ledger promised to somebody else.
    """
    if draft_on_p:
        return 0.0, "none (group P carries the draft; the draft tag flips)"
    if str(ns.pdflip_weight_source) != WEIGHT_SOURCE_EXCHANGE:
        return 0.0, f"none (weight source {ns.pdflip_weight_source!r}: the draft rides the base tag)"
    if str(getattr(ns, "flip_weights", "family")) == "resident":
        return 0.0, "none (--flip-weights resident: nothing sleeps)"
    extra_d = str(getattr(ns, "extra_d", "") or "")
    if _argv_scalar(extra_d, "--speculative-algorithm") is None:
        return 0.0, "none (group D runs no speculative decoding)"
    from flliper.srt.pdflip import draft_post as _dp

    placement = _argv_scalar(extra_d, "--speculative-draft-placement")
    path = _argv_scalar(extra_d, "--speculative-draft-model-path")
    mib = _dp.d_draft_host_mib(str(path), share_embed=d_draft_share_embed(ns)) if path else None
    if placement != "solo" or mib is None:
        raise PdFlipLaunchRefused(
            f"{DRAFT_PARK_UNPRICED}: group P carries no draft, so D parks its draft "
            "in pinned host RAM while P runs, and that image must be in the host "
            f"ledger -- but D's argv names placement={placement!r} draft "
            f"path={path!r} ({'no *.safetensors there' if path else 'no path'}). "
            "Only '--speculative-draft-placement solo' with an explicit "
            "--speculative-draft-model-path is priced (rank 0 holds the draft)."
        )
    return float(mib), (f"D rank 0 parks its solo draft: checkpoint {path} minus the "
                        f"target-shared tables (share_embed={d_draft_share_embed(ns)}) "
                        f"+ buffers {_dp.DRAFT_RUNNER_BUFFER_MIB} = {mib:.0f} MiB pinned")

def d_fr_ceiling_before_map(ns) -> bool:
    """R2 (28.09.): the P1c FR-D ceiling moves FR_D, and FR_D is a function
    the Platztausch map is built from -- so where P1c is ARMED (default on,
    P0 ``FLLIPER_PDFLIP_TORCH_CACHE_CAP=1`` in --env-d, a MoE D form with an
    ownership and an FR vector in --extra-d) the D form is solved and pinned
    BEFORE the map, from the expectation budgets, exactly like an owned /
    joint / maxmin form (#239 rc12z29c). After the map it would be refused
    (a514587c52, W170) and NF stayed on the profile's FR_D with ~1.3 GiB idle
    on nv0. A form the planner solves itself is already pinned there."""
    if d_form_solved_by_planner(ns):
        return False
    env_d = parse_group_env(getattr(ns, "env_d", "") or "")
    if str(env_d.get(FR_D_CEILING_ENV, "1")).strip() == "0":
        return False
    if str(env_d.get(TORCH_CACHE_CAP_ENV, "0")).strip() != "1":
        return False
    extra_d = getattr(ns, "extra_d", "") or ""
    return bool(_argv_vector(extra_d, "--rank-moe-ratio")
                and _argv_vector(extra_d, "--rank-moe-resident-fraction"))

def live_launch_servers(
    tag: str, proc_root: str = "/proc", self_pid: Optional[int] = None
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """#1217, fnFL2 H26: the LIVE ``flliper.launch_server`` processes that must
    refuse a residue sweep -- and, separately, the ones that only look like one.

    WHY NOT ``pgrep -f "flliper[.]launch_server"`` (what this replaces): it
    matches a SUBSTRING of every command line on the box.  Metal 24.09.
    08:56-08:57Z, boot fnFL2x140: three starts, three refusals naming pids
    3122930 / 3157795 / 3189930, each gone seconds later.  Each was the
    operator's own Bash-tool shell, ``bash -c '... pgrep -fc
    "flliper.launch_server"; ... bash start_when_free.sh fnFL2x139
    boot_nf_x140.sh fnFL2x140 ...'`` -- the ANCESTOR of this very launcher
    (tool shell -> start_when_free.sh -> boot_nf_x140.sh -> boot_nf_cd_long.sh
    -> arm_fnFL2_long.sh -> timeout -> ``python -m flliper.srt.pdflip.launcher
    ... --dry-run``), alive exactly as long as the start it was waiting on.
    The trap only armed because two dead 16 KB ``flliper_loads_*`` flags of
    another boot sat in /dev/shm; without own entries pgrep never ran.

    The rules:

    * this process and its whole parent chain are never counted;
    * only a real server argv counts (:func:`is_launch_server_argv`); any
      other process whose command line merely contains the module name goes
      to the second list, which the caller logs;
    * a real server counts when it holds a CUDA context, or when its boot tag
      (``FLLIPER_PDFLIP_BOOT_TOKEN``) is not ``tag`` -- a stock server without a
      token is foreign and counts even before its CUDA init.  Only a server of
      THIS tag with no CUDA context (nothing it could be holding) is skipped.
    """
    me = os.getpid() if self_pid is None else self_pid
    mine = {me, *_proc_ancestors(me, proc_root)}
    live: List[Dict[str, object]] = []
    ignored: List[Dict[str, object]] = []
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return live, ignored
    for entry in sorted((e for e in entries if e.isdigit()), key=int):
        argv = _proc_argv(entry, proc_root)
        if not argv:
            continue
        server = is_launch_server_argv(argv)
        if not server and not any(m in a for a in argv for m in LAUNCH_SERVER_MODULES):
            continue
        pid = int(entry)
        head = " ".join(argv)[:160]
        if pid in mine:
            ignored.append({"pid": pid, "why": "self/ancestor", "argv": head})
            continue
        if not server:
            ignored.append({"pid": pid, "why": "not a server argv", "argv": head})
            continue
        cuda = _proc_has_cuda_context(entry, proc_root)
        boot_tag = _proc_boot_tag(entry, proc_root)
        rec: Dict[str, object] = {"pid": pid, "cuda": cuda, "tag": boot_tag, "argv": head}
        if cuda or boot_tag != tag:
            live.append(rec)
        else:
            ignored.append({**rec, "why": "own tag, no CUDA context"})
    return live, ignored

def group_kv_tokens(ns, group: str) -> Tuple[int, str]:
    """YaRN x2 (27.09.): the KV tokens the form of group ``group`` must hold
    -- its context (#77: the whole context is the duty), or its
    ``--max-total-tokens`` pool where that is larger. Replaces the fixed
    CONTEXT_LENGTH_TOKENS of the D solve: a 524288 form priced at 262144
    'fitted' by 3.5 GiB on TP0 it does not have."""
    ctx, src = group_context_tokens(ns, group)
    raw = _argv_scalar(getattr(ns, f"extra_{group}", "") or "", "--max-total-tokens")
    try:
        pool = int(raw) if raw is not None else 0
    except ValueError:
        pool = 0
    if pool > ctx:
        return pool, f"--extra-{group} --max-total-tokens (> context {ctx} from {src})"
    return ctx, src

def model_num_layers(model: str) -> int:
    """The backbone depth, through the SERVER's own probe order (fix 6).

    For a GGUF file the loader's authority is the FILE: a depth that differs
    from the sibling config is reconciled there (the file wins, layer_types
    rebuilt), and every layer map this launcher derives from the config would
    then describe another model -- W162, before any card is touched."""
    from flliper.srt.server_args import declared_num_hidden_layers_from_config

    n = declared_num_hidden_layers_from_config(_model_config(model))
    if not n or n <= 0:
        raise PdFlipLaunchRefused(f"num_hidden_layers not found in {model_config_path(model)}")
    if model_is_gguf_file(model):
        depth = gguf_backbone_depth(model)
        if depth is not None and depth != int(n):
            raise PdFlipLaunchRefused(
                f"W162 PdFlipGgufDepthRefused: {model} carries {depth} backbone "
                f"blocks, its sibling {model_config_path(model)} declares "
                f"num_hidden_layers={n}. The loader runs the FILE's depth "
                "(gguf_registry.reconcile_sibling_config) while the chunk map, "
                "the P cut and the layer kinds of this launcher come from the "
                "config -- two sides of one seam on different models. Supply "
                "the config.json of exactly this GGUF next to it."
            )
    return int(n)

def apply_tokenizer_path(ns) -> Optional[str]:
    """Install --tokenizer-path once, before any argv is built (the install-once
    shape of apply_p_prefill_graph); returns the log line, if any."""
    path, line = tokenizer_path_decision(
        str(getattr(ns, "model", "") or ""),
        str(getattr(ns, "tokenizer_path", "") or ""),
    )
    _TOKENIZER_PATH["path"] = path
    return line

def apply_profile_d_kv_token_cut_default(ns, argv_words: Sequence[str]) -> Optional[str]:
    """#239 LEISTUNGSSCHALTER: an UNSET ``--d-kv-token-cut`` takes the registry
    row's ``d_kv_token_cut`` (nextflash ``owned``; qwen27b ``off``). Applied
    BEFORE the form is resolved (the kv axis derives from the flag), and only
    where the boot can run the cut -- else the code default ``off`` stays:
    D states Form A worker roles (``derive_kv`` refuses the cut without them),
    no other ``--form-kv`` is stated, and a flip boot has its host tier and at
    most one operator-named KV stage (``refuse_flip_under_token_cut``). Returns
    the line naming the default, or None."""
    if getattr(ns, "teardown", False):
        return None
    if pdflip_form.flag_given(argv_words, "--d-kv-token-cut"):
        return None
    if str(getattr(ns, "d_kv_token_cut", pdflip_form.KV_TOKEN_CUT_OFF)
           or pdflip_form.KV_TOKEN_CUT_OFF) != pdflip_form.KV_TOKEN_CUT_OFF:
        return None
    row = pdflip_form.profile_row(getattr(ns, "profile", None))
    want = str(getattr(row, "d_kv_token_cut", pdflip_form.KV_TOKEN_CUT_OFF)
               or pdflip_form.KV_TOKEN_CUT_OFF) if row is not None else pdflip_form.KV_TOKEN_CUT_OFF
    if want == pdflip_form.KV_TOKEN_CUT_OFF:
        return None
    stated_kv = getattr(ns, "form_kv", None)
    if stated_kv and stated_kv != "qsa_forma_dcp":
        return None
    roles = pdflip_form.flag_values(shlex.split(str(getattr(ns, "extra_d", "") or "")), "--rank-role")
    if not roles or "worker" not in [r.strip() for r in roles[-1].split(",")]:
        return None
    if not getattr(ns, "d_only", False):
        if getattr(ns, "pdflip_disable_hicache", False):
            return None
        stages = d_kv_stage_tokens_named(ns)
        if stages is not None and len(stages) != 1:
            return None
    ns.d_kv_token_cut = want
    return (f"{KV_TOKEN_CUT_MARKER} registry {row.id}: --d-kv-token-cut {want} "
            "(default on after the metal proof; --d-kv-token-cut off = the uncut Form A)")

def _d_early_start_armed(ns) -> bool:
    """BOOTZEIT 3: the one resolution of ``--pdflip-d-early-start``.

    'auto' (the default since the series 29.09., dearly z30x2) is on for the
    profiles whose row carries a metal proof (``d_early_start_proven``: Next
    Flash) and off for every other one, so the 27B line boots serial exactly
    as before until its own proof boot ran with an explicit 'on'."""
    mode = str(getattr(ns, "pdflip_d_early_start", "off") or "off")
    if getattr(ns, "dual_layout", False):
        # DUAL-TP3PP3: D is sized from P's AWAKE footprint, measured after P's
        # READY -- an early D would be planned before that number exists.
        return False
    if mode == "auto":
        return d_early_start_proven(getattr(ns, "profile", None))
    return mode == "on"

def p_attn_head_split_cfg(ns, pp_size: int, chunk_tokens: int, model: str):
    """The ONE AHConfig for group P, or None when off. Called by solve_p_cut
    (pool model post) and by main (group P's env) with the same inputs, so
    the planner and the ranks price and book one number."""
    from flliper.srt.pdflip import attn_head_split as _ah

    text = str(getattr(ns, "p_attn_head_split", P_ATTN_HEAD_SPLIT_DEFAULT) or "off")
    try:
        dels = _ah.parse_spec(text)
    except _ah.AHSpecError as exc:
        raise SystemExit(str(exc))
    if not dels:
        return None
    with open(model_config_path(model)) as fh:
        cfg = json.load(fh)
    tc = cfg.get("text_config") or cfg
    num_q, num_kv = int(tc["num_attention_heads"]), int(tc["num_key_value_heads"])
    head_dim = int(tc.get("head_dim") or int(tc["hidden_size"]) // num_q)
    min_w = int(getattr(ns, "p_attn_head_split_min_w", P_ATTN_HEAD_SPLIT_MIN_W_DEFAULT)
                or P_ATTN_HEAD_SPLIT_MIN_W_DEFAULT)
    buckets = list(p_prefill_graph_buckets())
    if buckets and min_w <= max(buckets):
        raise SystemExit(f"--p-attn-head-split-min-w {min_w}: a split chunk must run eager, "
                         f"above the captured prefill bucket(s) {buckets}")
    spec = p_chunk_policy_spec()
    max_w = max(int(chunk_tokens), int(spec.limits.max_tokens) if spec is not None else 0)
    if min_w > max_w:
        raise SystemExit(f"--p-attn-head-split: min_w {min_w} > the widest P chunk {max_w} -- "
                         "the split would never fire")
    try:
        _ah.validate(dels, pp_size=int(pp_size), num_kv_heads=num_kv, gqa=max(1, num_q // num_kv))
    except _ah.AHSpecError as exc:
        raise SystemExit(str(exc))
    if num_q % num_kv:
        raise SystemExit(f"--p-attn-head-split: {num_q} q heads over {num_kv} kv heads")
    ab = str(getattr(ns, "p_attn_head_split_ab", "off") or "off")
    ab = "" if ab == "off" else ab
    deadline = float(getattr(ns, "p_attn_head_split_deadline_ms", _ah.DEADLINE_MS_DEFAULT))
    if not any(d.downstream for d in dels) and deadline != _ah.DEADLINE_MS_DEFAULT:
        raise SystemExit("--p-attn-head-split-deadline-ms: only a downstream (V2) owner has a deadline; "
                         "the spec has upstream helpers only")
    return _ah.AHConfig(dels, min_w, int(getattr(ns, "max_kv_per_request", 0) or CONTEXT_LENGTH_TOKENS),
                        max_w, head_dim, num_q // num_kv, num_kv, deadline_ms=deadline, ab=ab)

def p_graph_split_env(ns) -> Dict[str, str]:
    """Group P's environment for --p-prefill-graph-split; {} when off or when
    the prefill graph itself is off (nothing to split inside)."""
    from flliper.srt.layers.attention import fi_graph_split as _fgs

    n = int(getattr(ns, "p_prefill_graph_split", P_PREFILL_GRAPH_SPLIT_DEFAULT) or 0)
    if n < 0 or n == 1:
        raise SystemExit(f"--p-prefill-graph-split must be 0 (off) or >= 2, got {n}")
    if not p_prefill_graph_buckets():
        return {}
    return _fgs.launcher_env_p_graph_split(n)

def census_checkpoint_decision(census_path: str, model_path: str,
                               evidence_dirs: Sequence[str],
                               allow_foreign: bool) -> Optional[str]:
    """The census must price THIS checkpoint (weg2xsn441: the FP8 boot ran on the
    INT8 census of weg2xsn246, the flip's VRAM was priced for the other
    checkpoint's bytes, and nothing said so before the metal). None when it
    does; a NAMED line when it cannot be verified or is allowed foreign;
    W161 otherwise -- in the dry run, before any card is touched."""
    src, how = census_checkpoint(census_path, evidence_dirs)
    if not src:
        return ("PDFLIP XCHG-CENSUS checkpoint UNVERIFIED: %s -- the flip's VRAM peaks "
                "are priced from %s for a checkpoint this launcher cannot name"
                % (how, census_path))
    if os.path.realpath(src) == os.path.realpath(str(model_path)):
        return None
    what = ("the census %s was measured on %s (%s); this boot runs %s -- its "
            "per-tag bytes, dormant residue and flip peaks are the other "
            "checkpoint's (format, scale geometry, embedding dtype, load residue)"
            % (census_path, src, how, model_path))
    if allow_foreign:
        return ("PDFLIP XCHG-CENSUS FOREIGN (allowed by --pdflip-xchg-census-foreign): "
                + what + " -- this boot MEASURES its own census, it is not priced by one")
    raise PdFlipLaunchRefused(
        "W161 PdFlipXchgCensusForeign: " + what + ". Produce this checkpoint's census "
        "(pdflip/xchg_census.py --ring-table-boot <a boot of it>), or pass "
        "--pdflip-xchg-census-foreign for the boot that measures it.")

def _d_extend_flip_trim_env(ns, log, label: str, rtext: str,
                            cards: Optional[Sequence[Card]] = None) -> Optional[str]:
    """27B int4ar 23.10. (D TP2 2x3080, boot dkr27bint4arbar1int4ar10231400 a45112428e, D.log _1023_140032): wherever
    the flip arm armed the vote (Q-694 rate + Q-710 floor), arm the rc12e trim too -- a value named in --env-d wins.

    The vote funds the chunk from DRIVER free alone (a4f4b8fec1). The 27B launcher never wrote the trim itself (only
    the NF #145 ledger path does, :func:`d_extend_trim_env`); the INT8 profile carried it by hand
    (``27b.env``: ``--env-d FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,1100,800``), the duo profile and the int4ar profile
    derived from it did not. Metal without it: 0 trims, D TP1 at 40-170 MiB driver free from 14:11Z on (near-OOM under
    the corridor rule), reserved 18.4 GiB vs 17.2 GiB allocated, every vote 1 row, the Q-710 floor raised it to 256
    (128-135 at 40 MiB free) -- 21003 D tokens in 137 extends, ~426 tok/s wall.

    Threshold per rank = ``ceil(line of the rank's card + floor_rows(CHUNKED_PREFILL_TOKENS) x rate)``
    (:func:`extend_trim.flip_trim_thresholds`): the driver free at which the vote funds at least the floor chunk
    above the line; under it the extend first hands the fully free cache segments back. #201 TRIM-LAW (24.10.): the
    line is the rank's OWN card's (:func:`d_flip_trim_lines`, D rank -> card UUID from ``cards``), not the stated law
    for every rank: a MEASURED card whose line is under 1024 trims no earlier than its law needs, one over 1024 is no
    longer priced under its line (the OOM direction). Unmeasured / foreign-form / no cards: the stated law, named.
    Review MUSS 1 (ADMIT-BASIS): the written value is never under ``budget_rest.ADMIT_TRIM_BASIS`` (1104, the
    threshold the admission records were measured under -- the admission trim reads this same list), named per rank,
    until a re-measurement under the lower threshold replaces the records. No reserve, no new term: the line, the floor chunk and the rate are the ones the vote itself uses. Dual layout /
    P0 / Next Flash never reach this (they never reach :func:`_d_extend_flip_rate_env`'s armed branch)."""
    from flliper.srt.pdflip import extend_trim as _et

    n = len([x for x in str(rtext or "").split(",") if x.strip()])
    if n <= 0:
        return None
    lines, srcs = d_flip_trim_lines(ns, cards, n)
    law_value = _et.flip_trim_thresholds(rtext, CHUNKED_PREFILL_TOKENS, 1, lines)
    if not law_value:
        return None
    # #201 review MUSS 1 (ADMIT-BASIS): the same list is the admission trim's threshold, and the admission records
    # (D_ADMIT_NEED_MIB, the duo non-booking) were measured with it at 1104 -- never WRITE a computed value under that
    # basis (budget_rest.ADMIT_TRIM_BASIS; measured regime, not a reserve) until a re-measurement replaces it
    _bound = _budget_rest.admit_trim_bound([float(x) for x in law_value.split(",")])
    value = ",".join(str(int(v)) for v, _held in _bound)
    _basis_name, _basis_mib = _budget_rest.ADMIT_TRIM_BASIS
    _env_d = getattr(ns, "env_d", "") or ""
    _have = parse_group_env(_env_d)
    # a value other than our own came from the user (a second solve pass sees its own write)
    _given = EXTEND_TRIM_ENV in _have and str(_have.get(EXTEND_TRIM_ENV, "")).strip() != value
    if not _given:
        ns.env_d = set_group_env(_env_d, EXTEND_TRIM_ENV, value)
    _now = str(parse_group_env(getattr(ns, "env_d", "") or "").get(EXTEND_TRIM_ENV, "")).strip()
    _rows = _et.floor_rows(CHUNKED_PREFILL_TOKENS, 1)
    _per = " | ".join(
        f"rang{r}" + (f" nvml{cards[r].nvml_index}" if cards is not None and len(list(cards)) == n else "")
        + f": {srcs[r]} -> {law_value.split(',')[r]}"
        + (f"; {_budget_rest.ADMIT_TRIM_BASIS_MARKER} {_basis_mib} haelt (Admissions-Records unter {_basis_name}="
           f"{_basis_mib} gemessen -- neu messen, bevor die Schwelle darunter darf) -> {value.split(',')[r]}"
           if _bound[r][1] else "")
        for r in range(n))
    log(f"{D_RANK_SOLVE_MARKER} {label} {EXTEND_TRIM_FLIP_MARKER} {EXTEND_TRIM_ENV}={_now}"
        f"{' (aus --env-d gesetzt, unveraendert; berechnet ' + value + ')' if _given else ''} (je Rang ceil(Linie der "
        f"eigenen Karte + Bodenchunk {_rows} x Rate); darunter gibt D vor dem Extend die ganz freien Cache-Segmente an "
        f"den Treiber zurueck, damit die Breitenabstimmung den echten Kartenstand liest) [{TRIM_LAW_MARKER} {_per}]")
    if _given:
        # review R1 (#201): a hand value under the basis stays in force (the user's regime, not overruled) but is
        # named: the admission records were measured at the basis, so an untrimmed pass can enter the admission under
        # 400 + A_peak on that rank
        try:
            _hv = [float(x) for x in _now.split(",") if x.strip()]
            _hv = (_hv * n) if len(_hv) == 1 else _hv
            for _r, _v in _budget_rest.admit_trim_hand_under_basis(_hv):
                log(f"{D_RANK_SOLVE_MARKER} {label} WARNUNG {TRIM_LAW_MARKER} {_budget_rest.ADMIT_TRIM_HAND_MARKER} "
                    f"rank={_r} value={_v:.0f} basis={_basis_mib}: der Handwert aus --env-d {EXTEND_TRIM_ENV} liegt "
                    f"unter der Basis, unter der die Admissions-Records gemessen sind (Schwelle >= 400 + A_peak 522 = "
                    f"922) -- bleibt unveraendert wirksam (Nutzer-Handwert, nicht ueberstimmt); eine ADMIT-NEED-Form "
                    f"(int4ar/duo) laeuft damit ausserhalb ihres gemessenen Regimes")
        except ValueError:
            pass  # an unreadable hand value is the rank's own parse_thresholds warning (trim off), not ours
    return _now

def log_draft_swap(ns, emap, *, fr_pp_before, total, ratios, fr_tp, layer_stage,
                   row_mib: float, log) -> None:
    """H25 (Praezisierung 08:30Z "ein systemram gegen vram tausch"): the host
    balance of the swap in ONE line -- the Platztausch store before and after
    the draft post (the map is rebuilt with the arm's own FR_P for "before"),
    plus D's parked draft. Only when group P carries no draft."""
    if str(getattr(ns, "draft_kv_on_p", "on")) != "off":
        return
    from flliper.srt.layers.moe import expert_map as _em
    from flliper.srt.pdflip import draft_post as _dp

    slots_after = int(emap["slots"])
    slots_before = slots_after
    if fr_pp_before:
        slots_before = int(_em.build_nested(
            total=total, ratios=[int(float(x)) for x in ratios],
            fr_pp=[float(x) for x in fr_pp_before], fr_tp=[float(x) for x in fr_tp],
            p_layer_stage=layer_stage, pad_tp=1)["slots"])
    path = _argv_scalar(getattr(ns, "extra_d", ""), "--speculative-draft-model-path") or ""
    host = _dp.d_draft_host_mib(str(path), share_embed=d_draft_share_embed(ns)) if path else None
    log(_dp.draft_swap_line(slots_before=slots_before, slots_after=slots_after,
                            layers=len(layer_stage), row_mib=float(row_mib),
                            d_draft_host=host))

def mamba_arena_slots_into_groups(ns, group_envs) -> str:
    """MAMBA-ARENA-FLOOR (16.10., boot 1016_124240: the shared mamba arena held 32
    slots from a profile env, complete=32 of 32, D lost anchors at=evict
    why=mamba_claim backuped=False). The ONE mamba arena is shared by every rank of
    both groups, so its slot count must hold both groups' floors at once (P's
    parked anchors and D's sleep flush coexist): floor = sum over the groups of
    device_slots + max_running + 1. A value under it -- profile env, launcher
    default, anything -- is RAISED to it in both group envs (the ranks and the
    ledger then read the same number); never a refusal (host RAM warns). Returns
    the line to log."""
    groups = [("D", getattr(ns, "extra_d", "") or "", getattr(ns, "d_bs", None))]
    if not bool(getattr(ns, "d_only", False)):
        groups.insert(0, ("P", getattr(ns, "extra_p", "") or "", getattr(ns, "p_bs", None)))
    parts, floor = [], 0
    for name, extra, bs_fb in groups:
        f, dev, bs, src = mamba_arena_group_floor(extra, bs_fb)
        parts.append("%s: %s" % (name, ("device %d [%s] + running %d + 1 = %d" % (dev, src, bs, f)) if f else src))
        floor += f
    named = any(e and str(e.get(MAMBA_ARENA_ENV, "") or "") for e in (group_envs or ())) or bool(
        str(os.environ.get(MAMBA_ARENA_ENV, "") or ""))
    cur = int(float(_group_env_value(MAMBA_ARENA_ENV, group_envs, "140")))
    if floor <= 0:
        return "PDFLIP-MAMBA-ARENA-FLOOR skipped: %s (slots %d kept)" % ("; ".join(parts), cur)
    if not named:
        # nobody named a count: the arena gets the floor itself (computed, not the
        # fixed launcher default 140 -- host RAM next to NF is the scarce side)
        val = str(int(floor))
        for attr in ("env_d", "env_p"):
            setattr(ns, attr, set_group_env(str(getattr(ns, attr, "") or ""), MAMBA_ARENA_ENV, val))
        for e in group_envs or ():
            if e is not None:
                e[MAMBA_ARENA_ENV] = val
        return "PDFLIP-MAMBA-ARENA-FLOOR COMPUTED: %s=%d in both group envs (%s; one arena for both groups)" % (
            MAMBA_ARENA_ENV, floor, "; ".join(parts))
    if cur >= floor:
        return "PDFLIP-MAMBA-ARENA-FLOOR ok: %s=%d >= floor %d (%s; one arena for both groups)" % (
            MAMBA_ARENA_ENV, cur, floor, "; ".join(parts))
    val = str(int(floor))
    for attr in ("env_d", "env_p"):
        setattr(ns, attr, set_group_env(str(getattr(ns, attr, "") or ""), MAMBA_ARENA_ENV, val))
    for e in group_envs or ():
        if e is not None:
            e[MAMBA_ARENA_ENV] = val
    return ("PDFLIP-MAMBA-ARENA-FLOOR RAISED: %s %d -> %d in both group envs (%s; one arena for both groups, "
            "a count under the sum loses anchors at=evict why=mamba_claim). Host RAM only warns, it never "
            "refuses this." % (MAMBA_ARENA_ENV, cur, floor, "; ".join(parts)))
