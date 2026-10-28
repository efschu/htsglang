"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import os

from flliper.srt.pdflip.launcher_d_leaf import DFLASH_DRAFT_PATH_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import VISION_OFF
from flliper.srt.pdflip.launcher_d_leaf import VISION_TRANSIENT
from flliper.srt.pdflip.launcher_d_leaf import build_expert_karte
from flliper.srt.pdflip.launcher_d_mid import d_kv_stage_tokens_named
from flliper.srt.pdflip.launcher_d_mid import expert_map_vectors
from flliper.srt.pdflip.launcher_p_l3_leaf import D_KV_STAGE_KEYS
from flliper.srt.pdflip.launcher_p_l3_leaf import model_config_path
from flliper.srt.pdflip.launcher_p_l3_leaf import model_is_gguf_file
from flliper.srt.pdflip.launcher_p_mid import _model_config
from flliper.srt.pdflip.launcher_p_mid import p_prefill_graph_buckets
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _rope_cache_cols
from flliper.srt.pdflip.launcher_types import PdFlipTokenCutFlipNotWired
from flliper.srt.pdflip.launcher_vision_spec_mid import W111_VISION_PLACE_DUAL
from flliper.srt.pdflip.launcher_vision_spec_mid import vision_tower_dir
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_PLACE_AUTO
import json

#: #239 S3h: the marker of the flip riegel line (refusal and allowance).
KV_TOKEN_CUT_FLIP_MARKER = "#239 S3h FLIP UNTER KV-TOKEN-SCHNITT"

def p_prefill_graph_flags() -> List[str]:
    """The ONE argv term of the P prefill graph ([] when off).

    ``--cuda-graph-config`` JSON rather than the convenience flags because it
    states all three keys in one token and wins over every other source
    (server_args._parse_cuda_graph_config), and an EXPLICIT prefill backend is
    also what skips the breakable-only auto-disable cascade (multimodal /
    memory-saver rules) that turned P's prefill graph off until now:
      * backend 'full' -- captures into the memory saver's cuda_graph tag
        (full_cuda_graph_backend.py), released by P's sleep like D's graphs;
      * bs [bucket]    -- one shape; a shorter rest chunk pads up to it;
      * full_prefill_max_req 1 -- one request slot: a chunk holding two
        requests runs eager (named by the runner's PREFILL-GRAPH eager line),
        and the captured linear-attention metadata never sees a sentinel row.
    """
    if not p_prefill_graph_buckets():
        return []
    return [
        "--cuda-graph-config",
        json.dumps(
            {
                "prefill": {
                    "backend": "full",
                    "bs": p_prefill_graph_buckets(),
                    "full_prefill_max_req": 1,
                }
            },
            separators=(",", ":"),
        ),
    ]

def p_prefill_graph_env(
    pool_mib: Sequence[float], max_prefix: Optional[int] = None
) -> Dict[str, str]:
    """Group P's environment under --p-prefill-graph ({} when off, so the
    default env stays byte-identical). ``max_prefix`` is
    --p-prefill-graph-max-prefix: None (the default) sets nothing, i.e. no
    depth threshold."""
    if not p_prefill_graph_buckets():
        return {}
    env = {
        # model_runner_kv_cache_mixin.PREFILL_GRAPH_POOL_ENV: the runtime
        # books the same post the pool model priced.
        "FLLIPER_KV_BUDGET_PREFILL_GRAPH_MIB": ",".join("%.1f" % v for v in pool_mib),
        # flashinfer_backend._full_cg_prefill_float_workspace: P captures no
        # decode graph (#1233 FIX 4), so the full-CG prefill wrappers may
        # share the float workspace instead of a second split-kv buffer.
        "FLLIPER_FULL_CG_PREFILL_SHARED_WORKSPACE": "1",
    }
    if max_prefix is not None:
        if int(max_prefix) < 0:
            raise SystemExit(f"--p-prefill-graph-max-prefix must be >= 0, got {max_prefix}")
        # prefill_cuda_graph_runner.PREFILL_GRAPH_MAX_PREFIX_ENV: a chunk whose
        # prefix exceeds it runs eager (census reason 'deep_split').
        env["FLLIPER_PREFILL_GRAPH_MAX_PREFIX"] = str(int(max_prefix))
    return env

def refuse_flip_under_token_cut(ns, boot_form) -> Optional[str]:
    """#239 S3h: the flip boot (P and D) of ``kv=qsa_forma_dcp``.

    None for every other form and for ``--d-only``. A flip boot with a host
    tier is the wired form: the line names what carries the cut and is
    returned. A flip boot without one is refused by name (a dry run prints
    the line with 'would refuse' and returns it; a real boot raises
    :class:`PdFlipTokenCutFlipNotWired`).

    The KV stages (29.09., S3h lifted where S3g holds): ONE operator-named
    stage passes as before. Stages the operator does not name are the
    launcher's to write (#251c + S3g: stage rows per KV rank, each from its
    own trim cell) -- this riegel runs before the plan, so it lets them
    through provisionally and ``refuse_flip_stage_form_without_trim_cells``
    decides after the FRACTION-SOLVE. More than one operator-named stage stays
    refused: the launcher writes no per-rank rows for them, a KV worker would
    meet the top stage untrimmed."""
    if boot_form is None or boot_form.kv != "qsa_forma_dcp" or getattr(ns, "d_only", False):
        return None
    from flliper.srt import rank_role

    why = []
    missing = rank_role.unwired_token_cut_seams()
    if missing:
        why.append("seam(s) %s unwired (%s)" % (
            ", ".join(missing), "; ".join(rank_role.SEAMS[m].what for m in missing)))
    if getattr(ns, "pdflip_disable_hicache", False):
        why.append(
            "--pdflip-disable-hicache: the P->D hand-off under the cut rides the "
            "arena L2 and the canonical store (the D workers read their owner "
            "rows of P's pages), and this boot has neither")
    stages = d_kv_stage_tokens_named(ns)
    if stages is not None and len(stages) != 1:
        why.append(
            "the D KV stage form is operator-named with %d stages %s -- the "
            "launcher writes no stage rows per KV rank for it (S3g: each KV "
            "worker trims only with its own trim cell in the plan), a KV-holding "
            "worker would build its pool for the top stage; name ONE stage "
            "(--env-d %s=<tokens>) or leave it to the launcher"
            % (len(stages), list(stages), D_KV_STAGE_KEYS[0]))
    if why:
        msg = (f"{KV_TOKEN_CUT_FLIP_MARKER}: --d-kv-token-cut {ns.d_kv_token_cut} "
               "(form kv=qsa_forma_dcp) in a FLIP boot refused -- "
               + " | ".join(why) + ". The cut boots with --d-only either way.")
        if getattr(ns, "dry_run", False):
            print(msg + " (dry run: would refuse)", flush=True)
            return msg
        raise PdFlipTokenCutFlipNotWired(msg)
    stage_text = (
        f"({D_KV_STAGE_KEYS[0]}={stages[0]}) " if stages is not None else
        "(KV-Stufen vom Launcher: Stufenzeilen je KV-Rang aus seiner Trim-Zelle, "
        "S3g -- Entscheid nach dem FRACTION-SOLVE, ohne Trim-Zelle je KV-Worker "
        "benannte Verweigerung) ")
    msg = (f"{KV_TOKEN_CUT_FLIP_MARKER} ERLAUBT: --d-kv-token-cut {ns.d_kv_token_cut} "
           + stage_text +
           "im Flip-Boot -- Traeger: das eine Arena-L2 und der kanonische "
           "Seitenspeicher (P schreibt ganze Seiten, die KV-Worker lesen ihre "
           "Owner-Zeilen, Tail-Adopt je Owner), Claim = Gruppen-MIN (Worker nur "
           "im MIN-Arm), Wake-Kredit je Rang gegen die eigene Karte. Metall-Marker: "
           "'#239 F14 KV-WORKER-WINDOW' auf TP1/TP2, 'PDFLIP-WAKE-KV-FIRST' je Rang, "
           "kein 'DRAFT-DISAGREE'.")
    print(msg, flush=True)
    return msg

def refuse_flip_stage_form_without_trim_cells(ns, boot_form, log) -> Optional[str]:
    """#239 S3h after the FRACTION-SOLVE (29.09.): the flip boot under the
    token cut whose KV stages the operator left to the launcher.

    Called right before every flip D start (dry run, real), after the last
    D solve wrote -- or dropped -- the stage form. None for every other form,
    for ``--d-only`` and for an operator-named stage (the pre-plan riegel
    decided those). Otherwise every rank the cut gives FA KV must carry its
    own stage rows in the written form (``_d_kv_stage_written``: the rows
    from its trim cell, ``kv_stage_trim_cell``); a dropped form or a KV
    worker without rows is refused by name -- the flip boot does not fall
    back to fixed tokens silently. A dry run prints 'would refuse' and returns
    the line; a real boot raises :class:`PdFlipTokenCutFlipNotWired`."""
    if boot_form is None or boot_form.kv != "qsa_forma_dcp" or getattr(ns, "d_only", False):
        return None
    w = getattr(ns, "_d_kv_stage_written", None)
    if not w and d_kv_stage_tokens_named(ns) is not None:
        return None  # operator-named (one stage, the riegel let it through)
    why = None
    if not w:
        why = ("keine Stufenform geschrieben (Zeile 'D-KV-STUFEN (#251c): entfaellt' im "
               "FRACTION-SOLVE nennt den Grund) -- die KV-Worker haben keine Trim-Zelle im "
               "Plan; feste Token nur mit genau EINER benannten Stufe (--env-d %s=<tokens>)"
               % D_KV_STAGE_KEYS[0])
    else:
        cut = [int(r) for r in (w.get("cut_workers") or ())]
        rows = {int(r): int(x) for r, x in (w.get("worker_rows") or {}).items()}
        if -1 in cut:
            why = "der Schnitt ist ungeloest -- welche Worker KV halten, ist nicht bekannt"
        else:
            lacking = [r for r in cut if rows.get(r, 0) <= 0]
            if lacking:
                why = ("KV-Worker Rang %s ohne eigene Stufenzeilen (Trim-Zelle) in der "
                       "geschriebenen Form %s -- er baute seinen FA-KV-Pool auf der obersten "
                       "Stufe ungetrimmt" % (",".join(str(r) for r in lacking), rows))
    if why:
        msg = (f"{KV_TOKEN_CUT_FLIP_MARKER}: --d-kv-token-cut {ns.d_kv_token_cut} "
               "(form kv=qsa_forma_dcp) in a FLIP boot refused -- KV-Stufen: " + why
               + ". The cut boots with --d-only either way.")
        if getattr(ns, "dry_run", False):
            log(msg + " (dry run: would refuse)")
            return msg
        raise PdFlipTokenCutFlipNotWired(msg)
    per_rank = " ".join("rang%d: %d" % (r, rows[r]) for r in sorted(rows))
    msg = (f"{KV_TOKEN_CUT_FLIP_MARKER} STUFEN ERLAUBT: jeder KV-Worker %s traegt seine "
           "Stufenzeilen aus der eigenen Trim-Zelle (%s; TP0 %d) -- S3g greift im Flip-Boot. "
           "Metall-Marker: '#251c KV-STAGE form=' mit trims_here=True auf TP0 UND den "
           "KV-Workern, 'D-KV-STUFEN RESIDENZ (#239 S3g) rang1/rang2' im FRACTION-SOLVE, "
           "kein 'PdFlipDSeatVramRefused ... holds full-attention KV', kein W171."
           % (",".join(str(r) for r in cut), per_rank, int(w["rows"])))
    log(msg)
    return msg

def _rope_cols(ns) -> Tuple[int, int, Optional[str]]:
    """Cos/sin cache columns of the target and (DFLASH) the draft -- context-free; read from the configs."""
    try:
        tcols = _rope_cache_cols(_model_config(str(ns.model)))
    except (OSError, ValueError, KeyError):
        tcols = 0
    dcols, dname = 0, None
    if str(getattr(ns, "spec_form", "") or "").upper() == "DFLASH":
        dpath = str(getattr(ns, "dflash_draft_path", "") or DFLASH_DRAFT_PATH_DEFAULT)
        try:
            dcols = _rope_cache_cols(_model_config(dpath))
            dname = os.path.basename(dpath.rstrip("/"))
        except (OSError, ValueError, KeyError):
            dcols = 0
    return tcols, dcols, dname

def planned_expert_store_bytes(ns, model: str, p_stage_layers=None,
                               karte_path: str = "") -> Tuple[Optional[int], str]:
    """The expert store's size from its DIMENSIONING source, never from df:
    store slots per layer x MoE layers x row bytes. ``karte_path`` = the map
    as written (after the D form is pinned); else the map is dry-built from
    the same vectors :func:`publish_expert_map` reads. (None, why) when no
    map could be built -- the ranks then size their store themselves."""
    from flliper.srt.planner import pp_cut as _pp_cut

    _terms = []

    def terms():
        if not _terms:
            _terms.append(_pp_cut.checkpoint_weight_terms(model))
        return _terms[0]

    if karte_path:
        with open(karte_path) as fh:
            emap = json.load(fh)
        src = f"map {karte_path}"
    else:
        ratios, fr_tp, fr_pp = expert_map_vectors(ns)
        if not (ratios and fr_tp and fr_pp):
            return None, "no map vectors (--rank-moe-ratio / FR_D / FR_P) -- no map, no sized store"
        _mirror = str(os.environ.get("PDFLIP_EXPERT_MAP_MIRROR", "0")).strip() in ("1", "true", "True", "on")
        emap, _ls = build_expert_karte(int(terms().num_experts), ratios, fr_pp, fr_tp,
                                        p_stage_layers, _mirror)
        src = "map dry-built from the vectors publish_expert_map reads"
    # the row the P cut priced (ns._expert_row_mib, same checkpoint terms) --
    # read only once a map exists: a dense checkpoint has no rows to price
    row = float(getattr(ns, "_expert_row_mib", 0.0) or 0.0) * _pp_cut.MIB
    if row <= 0:
        row = float(terms().expert_layer_weight_bytes) / max(1, int(terms().num_experts))
    slots = int(emap["slots"])
    layers = len(emap.get("p_layer_stage") or []) or int(terms().n_layers)
    return int(slots * layers * row), (f"slots {slots} x {layers} layers x "
                                       f"{row / _pp_cut.MIB:.3f} MiB/row ({src})")

def vision_gguf_tower_refusal(ns) -> Optional[str]:
    """VISION-GGUF (09.10.): ``--pdflip-vision transient`` with a GGUF ``--model`` reads the tower from ``--tokenizer-path``. Refused by name
    before any card is taken when that directory holds no single-shard tower in a safetensors index, or when its ``vision_config`` differs
    from the GGUF's (the rank builds the tower from the GGUF's config and fills it from the other checkpoint; ``model_type`` aside, the
    two must agree). None = consistent, or not this case (a safetensors model, or not transient: unchanged)."""
    if str(ns.pdflip_vision) != VISION_TRANSIENT or not model_is_gguf_file(str(ns.model)):
        return None
    from flliper.srt.planner.vision_stage_load import find_tower_shard

    try:
        tower_dir = vision_tower_dir(ns)
        find_tower_shard(tower_dir)
        configs = []
        for path in (str(ns.model), tower_dir):
            with open(model_config_path(path)) as fh:
                configs.append(json.load(fh).get("vision_config") or {})
    except Exception as exc:  # noqa: BLE001 -- every path out is the named refusal
        return f"{W111_VISION_PLACE_DUAL}: --pdflip-vision transient on a GGUF --model: {type(exc).__name__}: {exc}"
    mine, theirs = configs
    differ = sorted(k for k in set(mine) | set(theirs) if k != "model_type" and mine.get(k) != theirs.get(k))
    if not mine or differ:
        return (f"{W111_VISION_PLACE_DUAL}: --pdflip-vision transient on a GGUF --model: the vision_config of {ns.model} "
                f"{'is missing' if not mine else 'differs in ' + ', '.join(differ)} against --tokenizer-path {tower_dir}, "
                "whose checkpoint the tower is read from")
    return None

def vision_victim_host_term(ns) -> Tuple[float, str]:
    """VISION-WEIGHTS AP4: ``(MiB, provenance)`` of the host image of the displaced weights (ledger post ``vision_victim_host``).

    Non-zero only with ``--pdflip-vision transient`` + ``--pdflip-vision-place weights`` on a dense (27B) form: the victim bytes (= the tower bytes,
    plan section 2) wait in an anonymous pageable host image while one image is encoded and are freed after the give-back (TRANSIENT, 0 between
    images). NF (experts): 0, the victim rows have their copy in the expert store. A tower size that cannot be read is said (0 MiB, unmeasured),
    never guessed. HOST RAM IS A WARNING ONLY (user rule 2026-10-06): this term never refuses."""
    from flliper.srt.pdflip import vision_victim_plan as _vvp

    if not _vvp.active(getattr(ns, "pdflip_vision", VISION_OFF), getattr(ns, "pdflip_vision_place", VISION_PLACE_AUTO)):
        return 0.0, "none (--pdflip-vision-place is not 'weights')"
    form = getattr(ns, "pdflip_boot_form", None)
    if getattr(form, "arch", None) == "moe":
        return 0.0, "none (victim = resident expert rows; their copy is in the expert store, already booked)"
    try:
        tower_dir = vision_tower_dir(ns)
    except Exception as exc:  # noqa: BLE001 -- unmeasured is said (vision_gguf_tower_refusal stops the boot first)
        return 0.0, f"UNMEASURED ({type(exc).__name__}: {exc})"
    nbytes, src = _vvp.tower_bytes_from_checkpoint(tower_dir)
    if nbytes is None:
        return 0.0, f"UNMEASURED ({src})"
    kind = _vvp.KIND_PP_ONLY if bool(getattr(ns, "dual_layout", False)) else _vvp.KIND_DENSE
    return nbytes / (1 << 20), (f"victim={kind}: the tower bytes wait in a pageable host image per image request, freed after the give-back "
                                f"({src}); transient, 0 between images; printed only, in no ledger sum")
