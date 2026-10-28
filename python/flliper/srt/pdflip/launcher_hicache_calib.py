"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import List
from typing import Optional
from typing import Tuple
import os

from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip.launcher_d_leaf import D_RESHARD_DEFAULT
from flliper.srt.pdflip.launcher_d_leaf import _D_RESHARD
from flliper.srt.pdflip.launcher_d_leaf import _d_reshard_seg_placement
from flliper.srt.pdflip.launcher_d_leaf import build_expert_karte
from flliper.srt.pdflip.launcher_d_leaf import d_reshard_depth
from flliper.srt.pdflip.launcher_d_mid import CONTEXT_LENGTH_TOKENS
from flliper.srt.pdflip.launcher_d_mid import checkpoint_quant_method
from flliper.srt.pdflip.launcher_d_mid import expert_map_vectors
from flliper.srt.pdflip.launcher_d_mid import group_context_tokens
from flliper.srt.pdflip.launcher_d_p_apply import d_reshard_format
from flliper.srt.pdflip.launcher_d_p_apply import log_draft_swap
from flliper.srt.pdflip.launcher_d_p_apply import model_num_layers
from flliper.srt.pdflip.launcher_d_p_apply import p_attn_head_split_cfg
from flliper.srt.pdflip.launcher_d_p_apply import p_layer_split_spec_from_ns
from flliper.srt.pdflip.launcher_p_l3_leaf import P_LAYER_SPLIT_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import _P_LAYER_SPLIT
from flliper.srt.pdflip.launcher_p_l3_leaf import _int_list
from flliper.srt.pdflip.launcher_p_l3_leaf import expert_map_dir
from flliper.srt.pdflip.launcher_p_l3_leaf import model_config_path
from flliper.srt.pdflip.launcher_p_l3_leaf import p_layer_split_slab_lines
from flliper.srt.pdflip.launcher_p_mid import _model_config
from flliper.srt.pdflip.launcher_refuse_proc_leaf import PP_LAYER_SET_ENV
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _refuse_unbuilt_platztausch_buffers
from flliper.srt.pdflip.launcher_resolve_stop import _rope_cols
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
import json

P_LAYER_SPLIT_DRY_RUN_TOKENS = (32768, 131072)
D_RESHARD_DRY_RUN_LOADS = (("decode", 1, 32768), ("decode", 4, 32768), ("decode", 1, 131072),
                           ("prefill", 1, 4096))

def apply_p_layer_split(ns) -> None:
    """Install --p-layer-split once, after --p-chunk-policy (joint plan)."""
    from flliper.srt.pdflip import p_layer_split as _pls

    policy = str(getattr(ns, "p_layer_split", P_LAYER_SPLIT_DEFAULT) or P_LAYER_SPLIT_DEFAULT)
    if policy not in _pls.POLICIES:
        raise SystemExit(f"--p-layer-split {policy!r}: one of {list(_pls.POLICIES)}")
    _P_LAYER_SPLIT["policy"] = policy
    _P_LAYER_SPLIT["spec"] = None
    _P_LAYER_SPLIT["lines"] = ()
    if policy == _pls.POLICY_STATIC:
        return
    if os.environ.get(PP_LAYER_SET_ENV, "").strip() or getattr(ns, "pp_layer_set", None):
        raise SystemExit("--p-layer-split dynamic: REFUSED -- a layer SET (FLLIPER_PP_LAYER_SET / "
                         "--pp-layer-set) is not the contiguous home cut the split mirrors")
    spec = p_layer_split_spec_from_ns(ns)
    ratio = str(getattr(ns, "pp_stage_ratio", "") or "")
    if ratio:
        counts = _int_list(ratio, "--pp-stage-ratio")
        ends = tuple(sum(counts[: i + 1]) for i in range(len(counts) - 1))
        if ends != spec.geometry.home_cuts:
            raise SystemExit(f"--p-layer-split dynamic: REFUSED -- --pp-stage-ratio {ratio} cuts at "
                             f"{list(ends)} but --p-layer-split-home is {list(spec.geometry.home_cuts)}; "
                             f"the home cut IS the P partition (one cut, one author)")
    lines = ["PDFLIP " + _pls.armed_line(spec, "group=P")]
    lines += ["PDFLIP " + x for x in p_layer_split_slab_lines(spec)]
    for n in P_LAYER_SPLIT_DRY_RUN_TOKENS:
        res = _pls.plan_split(n, spec.geometry, spec.model, spec.limits, min_gain=spec.min_gain,
                              extra=spec.extra)
        lines.append("PDFLIP " + _pls.plan_line(res, key=f"dry-run-{n}", end=n)
                     + " (HOCHRECHNUNG on the model, not a measurement)")
    _P_LAYER_SPLIT["spec"] = spec
    _P_LAYER_SPLIT["lines"] = tuple(lines)
    for line in lines:
        print(line, flush=True)

def apply_d_reshard(ns) -> None:
    """Install --d-reshard once, before any argv is built."""
    from flliper.srt.pdflip import d_reshard as _dr

    policy = str(getattr(ns, "d_reshard", D_RESHARD_DEFAULT) or D_RESHARD_DEFAULT)
    if policy not in _dr.POLICIES + (_dr.POLICY_WAKE_SEG,):
        raise SystemExit(f"--d-reshard {policy!r}: one of {list(_dr.POLICIES) + [_dr.POLICY_WAKE_SEG]}")
    _D_RESHARD.update(policy=policy, spec=None, lines=())
    if policy == _dr.POLICY_OFF:
        return
    # 'wake-seg' (SB 26.09.): 'wake' whose preset minimises the SEGMENT round
    # (sum over layer segments of the slowest rank, d_reshard sec. "Segment
    # model"), not the per-rank sums -- rc9meas refuted the sum objective.
    objective = _dr.OBJECTIVE_SUM
    if policy == _dr.POLICY_WAKE_SEG:
        policy, objective = _dr.POLICY_WAKE, _dr.OBJECTIVE_SEGMENT
    model = str(getattr(ns, "model", "") or "")
    try:
        with open(model_config_path(model)) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--d-reshard {policy}: cannot read the model config of {model!r}: {exc}")
    try:
        _dr.check_profile(_dr.profile_of_config(cfg), policy)
    except _dr.ReshardProfileRefused as exc:
        raise SystemExit(str(exc))
    fmt = d_reshard_format(model)
    if fmt not in ("int8", "nvfp4"):
        raise SystemExit(f"--d-reshard {policy}: REFUSED -- no D cost model for checkpoint format "
                         f"{checkpoint_quant_method(model)!r} (calibrated: INT8 compressed-tensors, "
                         f"NVFP4 modelopt; DYN_D_RESHARD.md sec. 1)")
    raw_presets = str(getattr(ns, "d_reshard_presets", "auto") or "auto").strip()
    seg_points, seg_place, seg_shares = (), "capacity", None
    try:
        if objective == _dr.OBJECTIVE_SEGMENT:
            seg_points = _dr.seg_ladder(d_reshard_depth(getattr(ns, "d_reshard_depth", "robust")))
            seg_place, seg_shares = _d_reshard_seg_placement(ns)
        if objective == _dr.OBJECTIVE_SEGMENT and raw_presets == "auto":
            vec, _worst, _reg = _dr.seg_choose(fmt, seg_points, _dr.rc9_geometry(fmt).mlp_units,
                                               seg_place, seg_shares)
            presets = (_dr.Preset("seg", tuple(vec)),)
        else:
            presets = _dr.parse_presets(raw_presets, fmt)
        spec = _dr.ReshardSpec(policy, tuple(_dr.RC9_BASE), presets, presets[0].name,
                               float(getattr(ns, "d_reshard_min_gain", _dr.DEFAULT_MIN_GAIN)), fmt,
                               objective)
        spec.validate(_dr.rc9_geometry(fmt))
    except (_dr.ReshardError, ValueError) as exc:
        raise SystemExit(f"--d-reshard {policy}: {exc}")
    loads = [_dr.LoadClass(k, b, c) for k, b, c in D_RESHARD_DRY_RUN_LOADS]
    lines = ["PDFLIP " + _dr.armed_line(spec, "group=D")]
    if objective == _dr.OBJECTIVE_SEGMENT:
        lines += ["PDFLIP " + x + " (HOCHRECHNUNG on the segment model, not a measurement)"
                  for x in _dr.seg_plan_lines(fmt, spec.presets[0].mlp, seg_points, seg_place, seg_shares)]
    else:
        lines += ["PDFLIP " + x + " (HOCHRECHNUNG on the model, not a measurement)"
                  for x in _dr.plan_lines(spec, fmt, loads)]
    if policy == _dr.POLICY_LIVE:
        raise SystemExit("\n".join(lines) + "\n--d-reshard live: REFUSED -- a live switch moves the "
                         "MLP units over BAR1 (~0.2-0.3 s each way) for at most 33 ms (INT8) / 111 ms "
                         "(NVFP4) per 4096-token D-prefill chunk, and D prefills at most 3 chunks under "
                         "the X ceiling (DYN_D_RESHARD.md sec. 2.4).")
    if len(spec.presets) > 1:
        raise SystemExit("\n".join(lines) + "\n--d-reshard wake: REFUSED -- more than one preset "
                         "needs the rank-side executor (per-preset MLP views, graph sets, manifest "
                         "rows, spread posten), which is not wired (DYN_D_RESHARD.md sec. 7).")
    _D_RESHARD.update(spec=spec, lines=tuple(lines))

def _rope_context_terms(ns, group: str):
    """The ONE reading behind every RoPE-context delta (P pool floor, W11/W11b, D's dormant reserve):
    ``(ctx, src, extra_rows, target_cols, draft_cols, draft_name)`` for group ``group``; extra_rows <= 0 at or
    below 262144 (the rig's posts and records were measured there), and then no config is read."""
    ctx, src = group_context_tokens(ns, group)
    extra_rows = int(ctx) - CONTEXT_LENGTH_TOKENS
    if extra_rows <= 0:
        return ctx, src, extra_rows, 0, 0, None
    tcols, dcols, dname = _rope_cols(ns)
    return ctx, src, extra_rows, tcols, dcols, dname

def d_rope_context_delta_mib(ns, from_ctx: int = CONTEXT_LENGTH_TOKENS) -> Tuple[float, Optional[str]]:
    """YaRN x2 27B (29.09., zw4skn W19 at the first P wake): group D's per-rank share of the RoPE context
    growth, from the SAME reading as :func:`p_rope_context_delta_mib`. D is TP -- every rank builds the whole
    target's cos/sin cache AND the DFlash draft's -- and both stay resident while D sleeps, so D's dormant
    residue (the x1 record behind ``dc_expect_d``) grows by (target + draft cols) x extra rows x 4 B on every
    card. MEASURED dkr27browauthorityyarn2bar1fs09291825 (z30y3g, 524288): D_c(D) 2146 / 1718 / 1718 against the
    reserve 2106 / 1678 / 1678 built from the x1 record 1786 / 1358 / 1358 + 256 + 64 -> W19. ``(0.0, None)`` at
    or below 262144: every x1 reserve is byte-identical.

    ``from_ctx`` = the context the residue it is added to was MEASURED at (a record stamps
    ``vram_residue_context_tokens`` when it is not 262144; the constants are 262144): the delta is the
    difference, so a record measured at 524288 prices 0 for a 524288 boot and a NEGATIVE delta for an x1 boot
    (its own residue already carries the larger caches)."""
    ctx, src = group_context_tokens(ns, "d")
    if int(ctx) == int(from_ctx) or (int(ctx) <= CONTEXT_LENGTH_TOKENS and int(from_ctx) <= CONTEXT_LENGTH_TOKENS):
        return 0.0, None
    tcols, dcols, dname = _rope_cols(ns)
    rows = int(ctx) - int(from_ctx)
    per_col = rows * 4 / (1024.0 * 1024.0)
    delta = (tcols + dcols) * per_col
    return delta, (f"D ROPE-KONTEXT (YaRN x2 27B): D context {ctx} (from {src}) vs {int(from_ctx)} (the dormant "
                   f"residue's measured context) -> {rows:+d} rows x 4 B x (target {tcols} + draft "
                   f"{dname or '-'} {dcols}) cols = {delta:+.1f} MiB per D rank, resident in the dormant image")

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
    from flliper.srt.server_args import declared_layer_kinds_from_config

    return declared_layer_kinds_from_config(_model_config(model), model_num_layers(model))

def _p_ah_post_vector(ns, pp_size: int, chunk_tokens: int, model: str) -> Tuple[float, ...]:
    from flliper.srt.pdflip import attn_head_split as _ah

    return _ah.stage_post_vector(p_attn_head_split_cfg(ns, pp_size, chunk_tokens, model), pp_size)

def publish_expert_map(ns, model: str, evidence_dir: str, log,
                       p_stage_layers=None,
                       chunk_layers: Optional[int] = None) -> str:
    """Die EXPERTEN-KARTE bauen und ablegen; Pfad zurueck, sonst "".

    #107, Nutzer-Gesetz 22.09.: *"alles was geshardet wird requires ne
    karte"*. `expert_map.build` gab es seit dem Tag, `expert_store.expert_map`
    liest sie aus ``FLLIPER_MOE_EXPERT_MAP`` -- und NIEMAND hat sie je
    geschrieben (gemessen 22.09.: 2 Leser-Dateien, 0 Schreiber, in keinem
    Boot-argv). Ein Leser ohne Schreiber ist kein Riegel, er ist ein
    Kommentar (Memory ``riegel-hinter-dem-was-er-sichert``).

    Was das kostete: ohne Karte faellt ``_expert_store_rows_for`` auf die
    globale Residenz-Id-Datei zurueck. Die kennt EINE Zahl fuer alle Karten,
    also muss sie im Schnitt aller Stufen und Raenge liegen -- bei FR_P
    0.377/0.700/0.442 und FR_D 0.006/... bleibt genau EINE Id uebrig, die
    78+79 residenten Experten der beiden 3080er fallen heraus. Der Store
    wuchs dadurch von 420 auf 511 Slots (+22 %), liegt in tmpfs (= shmem =
    Host-RAM), und der Kernel toetete D-Rang 0 (w128, oom_kill 30 -> 31).
    Die Karte rechnet fuer dieselbe Form 354 Slots.

    KONSERVATIV: fehlt ein Vektor oder widerspricht sich die Karte, wird
    NICHTS geschrieben und der Lauf rechnet wie bisher -- eine halbe Karte
    waere eine falsche Slot-Zuordnung, und die ist Datenverlust.
    """
    import json as _json

    _skip = pdflip_form.gate_skip_line("#107 PDFLIP-EXPERT-MAP", getattr(ns, "pdflip_boot_form", None))
    if _skip:
        log(_skip)
        return ""
    try:
        # LOKAL importieren wie jede andere Funktion hier: `_pp_cut` ist im
        # Modulraum NICHT gebunden (nur in solve_p_cut/solve_d_ranks). Ohne
        # diese Zeile haette der `except BaseException` unten den NameError
        # geschluckt und "failed" geloggt -- ein Fix, der seine eigene
        # Fehlerklasse reproduziert. Der Test hat es gefangen.
        from flliper.srt.planner import pp_cut as _pp_cut
        from flliper.srt.layers.moe import expert_map as _em

        ratios, fr_tp, fr_pp = expert_map_vectors(ns)
        gaps = []
        if not ratios:
            gaps.append("--rank-moe-ratio (extra_d)")
        if not fr_tp:
            gaps.append("--rank-moe-resident-fraction (extra_d)")
        if not fr_pp:
            gaps.append("--pp-cut-expert-device-fraction / extra_p")
        if gaps:
            log("#107 EXPERTEN-KARTE ENTFAELLT: %s fehlt. Ohne sie rechnet "
                "jede Gruppe wieder selbst und die Residenz der jeweils "
                "anderen Raenge zaehlt nicht mit." % ", ".join(gaps))
            return ""
        total = int(_pp_cut.checkpoint_weight_terms(model).num_experts)
        # #159: `mirror` laesst P dieselben Ids waehlen wie D. Der Austausch
        # kann nur Bytes umhaengen, die BEIDE Seiten als denselben Tensor
        # beschreiben; ohne den Spiegel ist die Schnittmenge gemessen ZWEI
        # von 193 (fnFL2w132). Env-Schalter, damit die alte Form fuer einen
        # A/B byte-identisch erreichbar bleibt.
        # DEFAULT AUS (22.09. 21:0xZ, fnFL2w133): der Spiegel ist erst zur
        # HAELFTE gebaut. Er aendert die KARTE, aber die Residenz stellt im
        # Rang weiter `resident_fraction_for_rank()` her -- "die ersten N".
        # Karte und Rang widersprechen sich dann, und der Rang refused beim
        # Laden: "#107: 113 eigene kalte Experten haben in der KARTE keinen
        # Platz (erste: [236,237,238,239])" -- Id 236 ist in der Karte
        # resident, im Rang kalt. w133 starb daran, waehrend w132 mit
        # derselben Kette bis zum Wake kam.
        #
        # Was fehlt, ist `expert_map.resident_of()` im Ladepfad: die Funktion
        # existiert seit #107 und hat NULL Aufrufer -- dieselbe Klasse, die
        # heute schon #140, #156 und #107 selbst getroffen hat. Erst wenn der
        # Rang seine Residenz AUS DER KARTE nimmt, darf der Spiegel an.
        _mirror = str(os.environ.get("PDFLIP_EXPERT_MAP_MIRROR", "0")).strip() \
            in ("1", "true", "True", "on")
        # PLATZTAUSCH (Nutzer-Entscheid 22.09. 22:20Z): die Version-2-Karte.
        # Jede Phase haelt ihre eigene Anzahl; koordiniert wird nur die WAHL
        # der Ids (geschachtelt). Sie braucht den REALISIERTEN P-Layer-Split,
        # weil P's Residenz je STUFE gilt und ein Rang seinen Layer ueber die
        # Stufe adressiert (unter tp=1 ist moe_tp_rank fuer alle Stufen 0).
        emap, _layer_stage = build_expert_karte(total, ratios, fr_pp, fr_tp,
                                                 p_stage_layers, _mirror)
        if _layer_stage is None and not _mirror:
            _stages = [int(x) for x in (p_stage_layers or []) if int(x) > 0]
            log("#107 PLATZTAUSCH-KARTE ENTFAELLT: P-Layer-Split %s passt "
                "nicht zu %d P-Fractions -- Version-1-Karte wie bisher"
                % (_stages or "unbekannt", len(fr_pp)))
        reason = _em.refuse_if_inconsistent(emap)
        if reason:
            log("#107 EXPERTEN-KARTE VERWORFEN (not written): %s" % reason)
            return ""
        _refuse_unbuilt_platztausch_buffers(emap, chunk_layers=chunk_layers)
        file_path = os.path.join(expert_map_dir(evidence_dir, log), f"expert_map_{ns.tag}.json")
        with open(file_path, "w") as fh:
            _json.dump(emap, fh)
        # #239 rc12z29c-Blocker 2: die D-Form, die diese Karte beschreibt --
        # jeder D-Start wird dagegen geprueft (refuse_d_form_off_the_map)
        ns._expert_map_d_form = {"ratios": [str(x) for x in ratios],
                                 "fractions": [str(x) for x in fr_tp]}
        if _em.is_nested(emap):
            _pp = emap["phases"]["P"]
            _dp = emap["phases"]["D"]
            _row_mib = float(getattr(ns, "_expert_row_mib", 0.0) or 0.0)
            log(
                "PLATZTAUSCH-KARTE %s: %d experts, Store %d slots per Layer "
                "(= Komplement der Schnittmenge %d, in BEIDEN Phasen dieselbe "
                "Belegung, beim Flip wird nichts zurueckgeschrieben); P je Stufe "
                "resident %s davon gemeinsam %s; D je Rang resident %s. Der "
                "Austausch bewegt je Layer den gemeinsamen Praefix, das Extra "
                "holt der aufwachende Rang aus dem Store."
                % (file_path, total, int(emap["slots"]), int(emap["shared_resident"]),
                   [len(x) for x in _pp["resident"]],
                   [len(x) for x in _pp["common"]],
                   [len(x) for x in _dp["resident"]])
                + (" Store ~%.1f GiB." % (int(emap["slots"]) * _row_mib
                                          * len(emap.get("p_layer_stage", []))
                                          / 1024.0) if _row_mib else "")
            )
            _nj = _em.nested_join_verdict(emap)
            if _nj:
                log("PLATZTAUSCH FLIP-JOIN UNMOEGLICH: " + " | ".join(_nj))
            log_draft_swap(ns, emap, fr_pp_before=getattr(ns, "_h25_fr_p_before", None),
                           total=total, ratios=ratios, fr_tp=fr_tp,
                           layer_stage=_layer_stage, row_mib=_row_mib, log=log)
            return file_path
        _join = _em.join_verdict(emap)
        if _join:
            log("#159 FLIP-JOIN IMPOSSIBLE -- this boot cannot flip:")
            for _z in _join:
                log("     " + _z)
            log("     The swap is an OWNER CHANGE: what the one "
                "Gruppe nicht resident haelt, kann die andere nicht "
                "uebernehmen. Gleiche ANZAHL genuegt nicht -- P nimmt die "
                "ersten N von 0..total, D je Rang die ersten N SEINES Bandes; "
                "bei 158 gegen 158 ist die Schnittmenge EINS. Gemessen "
                "fnFL2w130/w131/w132: jedes Mal W68 PdFlipXchgPlanDisagree beim "
                "Wake, 40 Minuten nach dem Start, alle drei PP-Raenge tot.")
        else:
            log("#159 FLIP-JOIN ok: beide Gruppen halten je Layer dieselben Ids")
        log(
            "#107 EXPERTEN-KARTE %s: %d Experten, %d Store-Plaetze, P haelt je "
            "Stufe %s, D je Rang %s; der Flip bewegt %d Zeilen, %d bleiben auf "
            "den Karten liegen. BEIDE Gruppen lesen diese Datei -- vorher "
            "rechnete jede selbst und die Residenz der anderen Raenge fiel "
            "unter den Tisch."
            % (file_path, total, int(emap["slots"]),
               [len(x) for x in emap["phases"]["P"]["resident"]],
               [len(x) for x in emap["phases"]["D"]["resident"]],
               int(emap["moves"]), int(emap["shared_resident"]))
        )
        return file_path
    except PdFlipLaunchRefused:
        # W120 kippt den Boot -- mit Absicht, und als EINZIGE Ausnahme der
        # Regel darunter: eine Karte, die ein Rang nicht baut, flippt nicht
        # (x100), und ohne Karte flippt der Lauf erst recht nicht.
        raise
    except BaseException as _exc:  # noqa: BLE001 -- eine Karte kippt nie den Boot
        log("#107 EXPERTEN-KARTE failed: %s: %s" % (type(_exc).__name__, _exc))
        return ""
