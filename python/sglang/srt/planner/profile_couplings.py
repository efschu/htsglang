# SPDX-License-Identifier: Apache-2.0
"""PROFIL-EDITOR S4a (Auftrag 1431): die Kopplungs-Engine -- reine Funktionen, keine GPU, kein Server.

Der Profil-Editor zeigt, was ein geaenderter Wert nach sich zieht.  Dieses Modul rechnet es:
``(Hardwareprofil flliper.hardware/1, Modellprofil flliper.model/1, Einstellungen) -> Terme je Karte``
(PLAN_PROFIL_EDITOR_1003.md §3.4/§3.5).  Es ruft die vorhandenen Rechner, es baut keine zweite Rechnung:

* ``pp_cut.kv_cell_bytes_per_attention_layer`` / ``pp_cut.kv_reserve_mib_per_stage`` -- KV-Zelle und KV-Preis je Stufe
  (Metall fnFL2w123: 1088 B je Attention-Layer, Zellen 8704 / 4352 / 3264 mit Draft-Layer);
* ``pp_cut.solve_expert_fraction_per_stage`` (-> ``expert_residency.solve_stage_fraction_by_buffer_rule``) -- die groesste
  Experten-Fraction je Stufe, die ins Budget passt (Pufferregel ``min(R + Scratch, E)``);
* ``expert_residency.buffer_rows`` -- GPU-Zeilen je Layer bei einer gesetzten Fraction;
* ``model_profile.stage_weight_bytes`` / ``expert_buffer_fraction`` -- Gewichtsbytes je Stufe aus dem Modellprofil.

Die Kopplungen
--------------
C1  Layer-Split <-> KV/Kontext <-> Stufenleistung          (``c1_layer_split``, ``c1_move_layers``)
C2  MoE-Experten <-> KV                                    (``c2_expert_residency``)
C3  Chunkgroesse <-> Aktivierung <-> KV                    (``c3_chunk``)
C4  Kontextziel <-> KV-Gesamt                              (``c4_context_target``)
C5  Draft/speculation <-> VRAM P/D                         (nur Kante, noch ohne Rechnung)
C6  HiCache/L2/L3 <-> host RAM                             (nur Kante)
C7  Card choice/order <-> everything                       (nur Kante)

EHRLICHKEIT (Memory HOCHRECHNUNG != MESSUNG, INDIKATOR-GESETZ).  Jeder Term traegt seine Quelle (``src``): aus dem Profil
uebernommen (``config`` | ``Index`` | ``geschaetzt`` | ``gemessen`` | ``NVML``) oder ``Eingabe`` (vom Nutzer gesetzt) oder
``Standard`` (Annahme dieses Moduls, benannt).  Posten, die nur am Metall zu messen sind (CUDA-Kontext, Graphen, Allokator-Reste,
Seam-Staging), stehen in ``fixed_overhead_mib`` und sind OHNE Eingabe NULL -- dann ist ``free_mib`` eine OBERGRENZE und das Ergebnis
sagt es (``warnings``).  Die Stufenzeit ist eine Roofline-Naeherung (Decode, Batch 1, Gewichtsbandbreite ``mem_gbs.gemv``); sie
ersetzt nicht ``pp_cut.stage_costs`` (der braucht eine am Metall kalibrierte Census) und rankt keine Schnitte (#1019).

Dual-Form (``--dual-share``), Belege je Posten (Datei:Zeile im Baum 173161c595, Einzelheiten in ``_dual_share_p_terms``): Gewichte/Experten von P liegen im
Union-Image von D (weg2/launcher.py:22707-22712 Hilfe ``--dual-share``, :14784 ``UNION_ROLES=main``) -> Referenz, nicht im P-Budget; der Festposten
``--dual-p-overhead-mib`` liegt AUSSERHALB des P-Budgets (launcher.py:22713-22716, :14976 ``dual_share_planned_dc``); KV bei ``--dual-unified-kv on`` im
Karten-Ledger (weg2/card_kv_ledger.py Kopf); Mamba-Zustand und Draft sind P-eigen; was sich nicht an D binden laesst (Diff), ist nicht gerechnet.

Eingaenge der Launcher-Budgetrechnung der D-Phase (AP-H2 Fix-Runde 5).  ``launcher.budgets_from_dc`` (launcher.py:15252-15394) bekommt vom Launcher in
``_d_spec_from`` (launcher.py:27207-27214) die Eingaenge unten; jede Zeile sagt, ob der Balken sie rechnet (Datei:Zeile hier) oder ob sie ``nicht gerechnet``
ist -- dann nennt der Tooltip des Budgets sie so (``gaps`` aus ``launcher_d_budgets``).  Der Test ``TestLauncherFormulaForEveryPhaseAndForm`` ruft
``budgets_from_dc`` mit ALLEN diesen Argumenten auf (Release-Flip 27b-base, Release-Dual, NF abl, d_only) und haelt den Balken dagegen::

    Eingang (Launcher)                                       | im Balken gerechnet                                | nicht gerechnet
    ---------------------------------------------------------+----------------------------------------------------+------------------------------------------
    dc je Form (dc_src): Flip/d_only = --d-foreign-context-  | ``d_stage_terms`` pre_fixed: profile_couplings.py:1389           | im echten Lauf der gemessene Rest von P
      mib + --d-nontorch-mib; dual-share = P-Budget +        | ``dual_p_plan``: profile_couplings.py:1060                     |   nach sleep(P), nicht der Profilwert
      --dual-p-overhead-mib (dual_share_planned_dc)          |                                                    |
    user_reserve_by_card (--user-reserve-mib, Ordinal-       | ``parse_user_reserve``: profile_couplings.py:1083,      | im gebuchten Pfad bucht der Launcher sie
      Reihenfolge, 5090 zuerst; auch geerbt per ``source``)  | ``launcher_d_budgets`` Boden + Reserve: profile_couplings.py:1200  |   nicht (Tooltip sagt es)
    Korridor-Boden transient (CorridorFloor.mib = transient  | ``launcher_d_budgets``: stated law 1024            | gemessener Boden (Digest) und
      + reserve)                                             |   (``D_CORRIDOR_STATED_LAW_MIB``)                  |   SGLANG_CORRIDOR_LAW_FLOOR_MIB
    eingebauter Wach-Ueberschuss 404 (D_AWAKE_OVERSHOOT_MIB) | ``launcher_d_budgets``: profile_couplings.py:1198                  | --
    Overshoot-Record D_OVERSHOOT_MIB (overshoot_mib)         | ``launcher_d_budgets``: profile_couplings.py:1197 (Registerzeile)     | --
    Wach-Rest-Record D_AWAKE_REST_MIB (awake_rest_mib)       | ``launcher_d_budgets``: profile_couplings.py:1196 (ersetzt die 404)   | --
    P_DORMANT_SERVED_GROWTH_MIB (dormant_growth_mib)         | ``launcher_d_budgets``: profile_couplings.py:1149                     | --
    gebuchter Rest D_AWAKE_REST_BOOKED_MIB / ..._CAPPED_MIB  | ``launcher_d_budgets``: profile_couplings.py:1154 (Registerzeile    | Umgebung SGLANG_WEG2_BUDGET_REST_RECORD
      (booked_rest_kwargs; Torch-Cache-Kappe aus --env-d)    |   ``budget_rest_from_records``, Kappe aus env_d)   |   des Launchers
    Treiber-Carve (charge_driver_carve, driver_carve_min_    | ``launcher_d_budgets``: profile_couplings.py:1166, nur wenn die      | sonst: das Hardwareprofil traegt Card.
      total_mib, Card.reserved_mib)                          |   Hardwarekarte ``driver_reserved_mib`` traegt      |   reserved_mib nicht -> Tooltip nennt es
    Rundung auf 8 MiB (``// 8 * 8``)                         | ``launcher_d_budgets``: profile_couplings.py:1202                    | --
    corridor_constrain=True + corridor_sample_path (Pass     | --                                                 | nicht gerechnet (Messprobe; kann nur senken)
      mit Messprobe, launcher.py:15425-15462)                |                                                    |
    l15_mib (L1.5-Posten)                                    | kein Eingang der D-Phase (im Aufruf 27207-27214    | P-Aufruf (launcher.py:25663-25669):
                                                             |   nicht uebergeben)                                |   nicht gerechnet
    --d-reserve-mib (Verfuegbar-Vergleich, pp_cut.d_rank_    | ``contract_bar`` over_avail: profile_couplings.py:1665          | nicht Teil des Budgets (Launcher
      available_mib), NICHT budgets_from_dc                  |                                                    |   zieht es nur vom Verfuegbaren ab)
    --rank-user-reserve-mib / --rank-auto-reserve-mib        | --                                                 | Rang-Argumente der Laufzeit (server_args.py:
                                                             |                                                    |   2651, 2700), kein Eingang von
                                                             |                                                    |   budgets_from_dc: nicht gerechnet
    min mit --extra-p-Budget (dual_share_planned_dc,         | ``dual_p_plan``: Wert der Gruppe P als P-Budget    | eigenes P-Budget des Launchers
      launcher.py:14976-14992)                               |                                                    |   (budgets_from_dc('P'), :25663): nicht
    Records fuer eine andere Kartenzahl (inventory_view)     | --                                                 | nicht gerechnet, Tooltip nennt es
    explizites --rank-gpu-memory-mib der Gruppe D            | ``d_stage_terms`` has_budget: profile_couplings.py:1397         | --user-reserve-mib dort nicht gerechnet
                                                             |                                                    |   (Tooltip sagt es)

Einstellungen (``settings``, alles optional ausser ``stage_layers``)::

    stage_layers        [int, ...]     Layer je Stufe, Summe = n_layers (``--pp-stage-ratio``)
    budget_mib          [float, ...]   je Karte (``--rank-gpu-memory-mib``); sonst Kartengroesse - corridor_mib
    corridor_mib        float = 1024   freier VRAM je Karte im Ruhestand (Reserve-Semantik)
    kv_dtype            "auto"|"fp8_e4m3"   Variante aus ``model.kv.variants`` (sonst ``kv.chosen``)
    context_tokens      int = 262144   KV-Ziel je Stufe (Pflicht-Kontext des Nutzers)
    chunk_tokens        int = 2048     Prefill-Chunk (``--chunked-prefill-size`` / ``--p-chunk-max``)
    extend_rate_mib     float          MiB je Chunk-Zeile; sonst ``model.activation.extend_rate_mib_per_row``
    mamba_slots         int = 1        Mamba/GDN-Slots (Zustand je Linear-Layer und Slot)
    ssm_dtype           str            Zustands-Dtype (Schluessel von ``model.state.variants_mib``, z. B. bfloat16); sonst der der Config
    activation_mib      float | [..]   GEMESSENE Aktivierungsspitze je Karte statt Chunk x Rate (Eingabe, z. B. aus einem Boot)
    moe_resident_fraction  float | [..]   residenter Expertenanteil je Stufe (1.0 = alle)
    scratch_rows        int | [..] = 0   Scratch-Zeilen je Layer (Pufferregel)
    draft               bool = False   Draft/MTP-Layer: Gewicht auf der letzten Stufe, eine Attention-Zeile KV je Stufe
    fixed_overhead_mib  float | [..] = 0   am Metall gemessene feste Posten je Karte
    replicated          [str, ...]     Gewichtsposten, die jede Stufe traegt (``visual``, ``mtp``)
    attn_layers         [int, ...]     Attention-Layer je Stufe, wenn gepinnt (``--pp-attn-stage-ratio``); sonst aus den Familien
"""

from __future__ import annotations

import math
import json
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.planner import expert_residency as _er
from sglang.srt.planner import pp_cut as _pp
from sglang.srt.weg2 import model_profile as _mp

__all__ = [
    "SCHEMA",
    "EDGES",
    "CouplingError",
    "compute",
    "c1_layer_split",
    "c1_move_layers",
    "c2_expert_residency",
    "c3_chunk",
    "c4_context_target",
    "settings_from_server",
    "synthetic_hardware",
    "run",
    "main",
    "BAR_SEGMENTS",
    "stage_bar",
    "bars_for",
    "approx_payload",
    "approx_terms",
    "BALKEN_SCHEMA",
    "BALKEN_SEGMENTS",
    "phase_bars",
    "contract_bar",
    "d_stage_terms",
    "d_phase_config",
    "p_phase_settings",
    "detect_form",
]

SCHEMA = "flliper.couplings/1"
MIB = 1024.0 * 1024.0

#: Quellen, die diese Engine vergibt.  Alles andere ist die Quelle aus dem Profil (config/Index/geschaetzt/gemessen/NVML).
SRC_INPUT = "Input"
SRC_DEFAULT = "Default"
SRC_DERIVED = "calculated"


class CouplingError(ValueError):
    """Eine Eingabe, die sich nicht rechnen laesst (benannt, nie still repariert)."""


# ---------------------------------------------------------------------------
# C5-C7: benannte Kanten, noch ohne Rechnung
# ---------------------------------------------------------------------------

EDGES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "C5",
        "name": "Draft/speculation <-> VRAM P/D",
        "touches": ["--spec-form", "--draft-model-path", "P_DRAFT_RESIDENT_BUDGET_MIB"],
        "computes_with": ["expert_residency.draft_vocab_mib", "Records"],
        "computed": False,
        "text": "The draft occupies VRAM in every group that carries it; without a draft more remains for KV, but the spec gain is lost.",
        "now": "not computed: only ``draft`` (weight of the MTP layers on the last stage, one KV row per stage) enters C1-C4",
    },
    {
        "id": "C6",
        "name": "HiCache/L2/L3 <-> host RAM",
        "touches": ["--store-max-gb", "SGLANG_HICACHE_*", "PROFILE_SHM_MIN_GIB", "PROFILE_MEMAVAIL_MIN_GIB"],
        "computes_with": ["weg2/host_ledger.py", "Hardware profile host"],
        "computed": False,
        "text": "The cache stages live in host RAM, not in VRAM: thresholds from model bytes and card count (HWGEN P9 / K10).",
        "now": "not computed: a host RAM bar instead of VRAM follows",
    },
    {
        "id": "C7",
        "name": "Card choice/order <-> everything",
        "touches": ["Card list", "Host ordinal (form A)", "all vector values"],
        "computes_with": ["card_identity.order_cards", "topology.plan_topology"],
        "computed": False,
        "text": "Adding or removing a card changes the length of all vector values (layers, budgets, shares).",
        "now": "not computed: ``compute`` demands as many stages as cards and refuses otherwise (``vector_length``)",
    },
)


# ---------------------------------------------------------------------------
# kleine Helfer
# ---------------------------------------------------------------------------


def _val(node: Any, default: Any = None) -> Any:
    """Der Wert eines Profilknotens ``{"v": .., "src": ..}`` (oder der Wert selbst)."""
    if isinstance(node, Mapping) and "v" in node:
        v = node["v"]
        return default if v is None else v
    return default if node is None else node


def _src(node: Any, default: str = SRC_DEFAULT) -> str:
    return str(node["src"]) if isinstance(node, Mapping) and node.get("src") else default


def _per_card(x: Any, n: int, name: str, default: float) -> List[float]:
    if x is None:
        return [float(default)] * n
    if isinstance(x, (int, float)):
        return [float(x)] * n
    xs = [float(v) for v in x]
    if len(xs) != n:
        raise CouplingError("vector_length: %s has %d values, there are %d cards/stages" % (name, len(xs), n))
    return xs


def _term(v: float, src: str, note: str = "") -> Dict[str, Any]:
    out = {"v": round(float(v), 3), "src": src}
    if note:
        out["note"] = note
    return out


def _cards(hw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cards = list(hw.get("cards") or [])
    if not cards:
        raise CouplingError("Hardware profile without cards")
    return sorted(cards, key=lambda c: int(c.get("ord", 0)))


def _label(c: Mapping[str, Any]) -> str:
    return "Card %s (%s)" % (c.get("ord", "?"), str(c.get("name", "?")).replace("NVIDIA GeForce ", ""))


def _families(model: Mapping[str, Any]) -> List[str]:
    return [str(f) for f in _val(model["arch"]["layer_families"])]


def _cell_bytes(model: Mapping[str, Any], kv_dtype: Optional[str]) -> Tuple[float, str]:
    """Bytes je Token und Attention-Layer (Nutzlast + Skalenpuffer) der gewaehlten KV-Variante."""
    kv = model["kv"]
    if kv_dtype:
        var = (kv.get("variants") or {}).get(kv_dtype)
        if var is None:
            raise CouplingError("kv_dtype %r is no variant of the model profile (%s)" % (kv_dtype, ", ".join((kv.get("variants") or {}))))
        return float(_val(var["payload_bytes"])) + float(_val(var.get("scale_bytes"), 0.0)), _src(var["payload_bytes"])
    node = kv["cell_bytes_per_attn_layer_token"]
    return float(_val(node)), _src(node)


def _kv_geometry(model: Mapping[str, Any], kv_dtype: Optional[str], cell: float) -> Dict[str, Any]:
    """Geometrie fuer ``pp_cut.kv_*``: Koepfe, Kopfbreite, Dtype-Bytes der gewaehlten KV-Variante (fp8 = 1 B, sonst Modell-Dtype)."""
    a = model["arch"]
    chosen = kv_dtype or str(_val(model["kv"].get("chosen"), "auto"))
    dt = 1.0 if chosen.startswith("fp8") else 2.0
    hd = int(_val(a["head_dim"]))
    return {"kv_heads": int(_val(a["heads_kv"])), "head_dim": hd, "v_head_dim": hd, "kv_dtype_bytes": dt}


def _stage_bounds(counts: Sequence[int]) -> List[Tuple[int, int]]:
    out, start = [], 0
    for c in counts:
        out.append((start, start + int(c)))
        start += int(c)
    return out


# ---------------------------------------------------------------------------
# Terme je Stufe
# ---------------------------------------------------------------------------


def _stage_terms(hw: Mapping[str, Any], model: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
    cards = _cards(hw)
    n = len(cards)
    counts = [int(c) for c in (settings.get("stage_layers") or [])]
    if not counts:
        raise CouplingError("stage_layers is missing (--pp-stage-ratio)")
    if len(counts) != n:
        raise CouplingError("vector_length: stage_layers has %d stages, the hardware profile %d cards" % (len(counts), n))
    fams = _families(model)
    if sum(counts) != len(fams):
        raise CouplingError("stage_layers sums to %d, the model has %d layers" % (sum(counts), len(fams)))
    if any(c < 0 for c in counts):
        raise CouplingError("stage_layers contains negative values")

    warnings: List[str] = []
    corridor = float(settings.get("corridor_mib", 1024.0))
    ctx = int(settings.get("context_tokens", 262144))
    chunk = int(settings.get("chunk_tokens", 2048))
    slots = int(settings.get("mamba_slots", 1))
    draft = bool(settings.get("draft", False))
    replicated = tuple(settings.get("replicated") or ())

    totals = [float(_val(c.get("vram_total_mib"), 0.0)) for c in cards]
    if any(t <= 0 for t in totals):
        raise CouplingError("Hardware profile: vram_total_mib is missing on a card")
    budgets = _per_card(settings.get("budget_mib"), n, "budget_mib", 0.0) if settings.get("budget_mib") is not None else [t - corridor for t in totals]
    budget_src = SRC_INPUT if settings.get("budget_mib") is not None else SRC_DERIVED
    fixed = _per_card(settings.get("fixed_overhead_mib"), n, "fixed_overhead_mib", 0.0)
    fixed_src = SRC_INPUT if settings.get("fixed_overhead_mib") is not None else SRC_DEFAULT
    if settings.get("fixed_overhead_mib") is None:
        warnings.append("fixed_overhead_mib not set (CUDA context, graphs, allocator remainders, seam staging can be measured only on the hardware): free_mib is an UPPER BOUND")

    # Experten-Residenz je Stufe
    frac_in = settings.get("moe_resident_fraction", 1.0)
    fracs = _per_card(frac_in, n, "moe_resident_fraction", 1.0)
    scratch = _per_card(settings.get("scratch_rows"), n, "scratch_rows", 0.0)
    E = int(_val(model.get("experts", {}).get("n"), 0) or 0)
    moe = E > 0
    buf_fracs = [_mp.expert_buffer_fraction(E, f, int(s)) if moe else 1.0 for f, s in zip(fracs, scratch)]

    # Gewichte
    w = model["weights"]
    weights_full = _mp.stage_weight_bytes(model, counts, expert_fractions=buf_fracs, replicated=replicated)
    weights_dense = _mp.stage_weight_bytes(model, counts, expert_fractions=[0.0] * n, replicated=replicated)
    layer_b = _val(w["layer_bytes"])
    exp_b = _val(w["layer_expert_bytes"])
    bounds = _stage_bounds(counts)
    mtp_b = float(_val(w.get("mtp_bytes"), 0.0)) if draft else 0.0
    draft_src = _src(w.get("mtp_bytes"))
    draft_note = "MTP layers on the last stage" if draft else ""
    if draft and settings.get("draft_mib") is not None:
        # AP-H2: das Gewicht des Drafts kommt aus dem Profil des Draft-Verzeichnisses (model["draft"]["external"]), nicht aus dem Zielmodell
        mtp_b = float(settings["draft_mib"]) * MIB
        draft_src = str(settings.get("draft_src") or draft_src)
        draft_note = str(settings.get("draft_note") or draft_note)
    if "mtp" in replicated:
        mtp_b = 0.0        # schon als replizierter Posten in den Gewichten
    draft_bytes = [0.0] * n
    if draft:
        draft_bytes[-1] = mtp_b
    weights_b = [f + d for f, d in zip(weights_full, draft_bytes)]

    # Attention / Linear je Stufe
    pinned = settings.get("attn_layers")
    if pinned is not None:
        attn = [int(a) for a in pinned]
        if len(attn) != n:
            raise CouplingError("vector_length: attn_layers has %d values, there are %d stages" % (len(attn), n))
        if sum(attn) != sum(1 for f in fams if f == "attn"):
            raise CouplingError("attn_layers sums to %d, the model has %d attention layers" % (sum(attn), sum(1 for f in fams if f == "attn")))
    else:
        attn = [sum(1 for f in fams[a:b] if f == "attn") for a, b in bounds]
    lin = [c - a for c, a in zip(counts, attn)] if pinned is not None else [sum(1 for f in fams[a:b] if f != "attn") for a, b in bounds]
    draft_attn = [1 if draft else 0] * n

    cell, cell_src = _cell_bytes(model, settings.get("kv_dtype"))
    cell_mib = cell / MIB
    geo = _kv_geometry(model, settings.get("kv_dtype"), cell)
    cross = _pp.kv_cell_bytes_per_attention_layer(**geo)
    if abs(cross - cell) > 0.5:
        warnings.append("KV cell: model profile %.0f B, pp_cut geometry %.0f B per attention layer (the model profile applies)" % (cell, cross))
        geo = dict(geo, kv_dtype_bytes=geo["kv_dtype_bytes"] * cell / cross)
    state_node = model["state"].get("per_linear_layer_per_slot_mib")
    state_per = float(_val(state_node, 0.0))
    ssm = settings.get("ssm_dtype")
    if ssm:
        variants = model["state"].get("variants_mib") or {}
        if ssm not in variants:
            raise CouplingError("ssm_dtype %r is no variant of the model profile (%s)" % (ssm, ", ".join(variants)))
        state_per = float(variants[ssm])
        state_node = {"v": state_per, "src": SRC_INPUT}
    act_in = settings.get("activation_mib")
    act_vec = _per_card(act_in, n, "activation_mib", 0.0) if act_in is not None else None
    rate_node = (model.get("activation") or {}).get("extend_rate_mib_per_row")
    rate = float(settings["extend_rate_mib"]) if settings.get("extend_rate_mib") is not None else float(_val(rate_node, 0.0))
    rate_src = SRC_INPUT if settings.get("extend_rate_mib") is not None else _src(rate_node)

    stages = []
    for i, c in enumerate(cards):
        kv_mib = _pp.kv_reserve_mib_per_stage(
            tokens=ctx, attn_layers_by_stage=[attn[i]], draft_attn_layers_by_stage=[draft_attn[i]], **geo)[0]
        state_mib = lin[i] * state_per * slots
        act_mib = act_vec[i] if act_vec is not None else chunk * rate
        dense_mib = weights_dense[i] / MIB
        experts_mib = (weights_full[i] - weights_dense[i]) / MIB
        draft_mib = draft_bytes[i] / MIB
        w_mib = dense_mib + experts_mib + draft_mib
        resident = w_mib + state_mib + act_mib + fixed[i]
        needs = resident + kv_mib
        free = budgets[i] - needs
        kv_cap = None
        if attn[i] + draft_attn[i] > 0 and cell_mib > 0:
            kv_cap = max(0.0, (budgets[i] - resident) / ((attn[i] + draft_attn[i]) * cell_mib))
        # Rundenzeit (Decode, Batch 1): aktive Gewichtsbytes / Bandbreite -- Naeherung
        top_k = int(_val(model.get("experts", {}).get("top_k"), 0) or 0)
        active = sum(layer_b[bounds[i][0]:bounds[i][1]])
        if moe and top_k:
            active += sum(exp_b[bounds[i][0]:bounds[i][1]]) * min(1.0, top_k / float(E))
        if i == n - 1:
            active += float(_val(w.get("lm_head_bytes"), 0.0))
        gemv = _val((c.get("mem_gbs") or {}).get("gemv"))
        decode_ms = (active / (float(gemv) * 1e9) * 1e3) if gemv else None
        stages.append({
            "ord": c.get("ord", i),
            "label": _label(c),
            "layers": counts[i],
            "attn_layers": attn[i],
            "linear_layers": lin[i],
            "budget_mib": _term(budgets[i], budget_src, ("Card size - corridor %.0f MiB (assumption; the P budget of the launcher is calculated by budgets_from_dc('P', ...) with dc_expect_d, P_OVERSHOOT_MIB, --user-reserve-mib and the L1.5 item, launcher.py:25663-25669: not calculated here)" % corridor) if budget_src == SRC_DERIVED else ""),
            "terms": {
                "weights": _term(dense_mib, _src(w["total_bytes"]) if "total_bytes" in w else SRC_DEFAULT,
                                 "dense weights of the layers + embedding/lm_head of the role"),
                "experts": _term(experts_mib, _src(w["total_bytes"]) if "total_bytes" in w else SRC_DEFAULT,
                                 "resident expert rows %.0f %% (buffer rule)" % (100 * buf_fracs[i]) if moe else "no experts"),
                "draft": _term(draft_mib, draft_src if draft else SRC_DEFAULT, draft_note if draft_mib else "no draft"),
                "kv": _term(kv_mib, cell_src, "%d tokens x %d attention layers%s x %.0f B" % (ctx, attn[i], " + Draft" if draft_attn[i] else "", cell)),
                "state": _term(state_mib, _src(state_node), "%d linear layers x %.4f MiB x %d slot(s)" % (lin[i], state_per, slots)),
                "activation": _term(act_mib, SRC_INPUT if act_vec is not None else rate_src,
                                    "measured peak (input)" if act_vec is not None else "%d rows x %.4f MiB" % (chunk, rate)),
                "fixed": _term(fixed[i], fixed_src, "" if fixed_src == SRC_INPUT else "not measured"),
            },
            "draft_note": draft_note if i == n - 1 and draft else "",
            "needs_mib": round(needs, 3),
            "free_mib": round(free, 3),
            "overflow_mib": round(max(0.0, -free), 3),
            "kv_capacity_tokens": None if kv_cap is None else int(kv_cap),
            "decode_ms": None if decode_ms is None else round(decode_ms, 3),
            "decode_src": "Approximation (roofline, batch 1, mem_gbs.gemv)" if decode_ms is not None else "not measured (mem_gbs.gemv is missing)",
            "total_mib": totals[i],
            "kv_cell_mib": cell_mib,
            "weights_bytes": weights_b[i],
            "weights_total_mib": round(w_mib, 3),
        })
    capped = [s["kv_capacity_tokens"] for s in stages if s["kv_capacity_tokens"] is not None]
    timed = [s["decode_ms"] for s in stages if s["decode_ms"] is not None]
    return {
        "stages": stages,
        "context_floor_tokens": min(capped) if capped else None,
        "makespan_ms": max(timed) if len(timed) == n else None,
        "overflow_cards": [s["ord"] for s in stages if s["overflow_mib"] > 0],
        "warnings": warnings,
        "_ctx": {"cell_mib": cell_mib, "counts": counts, "attn": attn, "lin": lin, "draft_attn": draft_attn, "budgets": budgets,
                 "fixed": fixed, "rate": rate, "chunk": chunk, "slots": slots, "state_per": state_per, "E": E,
                 "buf_fracs": buf_fracs, "scratch": scratch, "act_vec": act_vec},
    }


# ---------------------------------------------------------------------------
# C1: Layer-Split <-> KV/Kontext <-> Stufenleistung
# ---------------------------------------------------------------------------


def c1_layer_split(hw: Mapping[str, Any], model: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Terme, Kontext-Boden (Minimum ueber die Stufen: jede PP-Stufe haelt KV fuer ALLE Token ihrer Layer) und Rundenzeit je Stufe."""
    t = _stage_terms(hw, model, settings)
    out = {k: v for k, v in t.items() if k != "_ctx"}
    out["id"] = "C1"
    return out


def c1_move_layers(hw: Mapping[str, Any], model: Mapping[str, Any], settings: Mapping[str, Any], *,
                   src: int, dst: int, n: int = 1) -> Dict[str, Any]:
    """``n`` Layer von Stufe ``src`` auf die BENACHBARTE Stufe ``dst`` verschieben: vorher/nachher je Karte und der Klartext.

    Der Schnitt ist zusammenhaengend; nur Nachbarstufen tauschen Layer (der Rand-Layer wechselt den Besitzer)."""
    counts = [int(c) for c in settings.get("stage_layers") or []]
    if abs(int(src) - int(dst)) != 1:
        raise CouplingError("Layers move only between ADJACENT STAGES (src=%s dst=%s)" % (src, dst))
    if not (0 <= src < len(counts) and 0 <= dst < len(counts)):
        raise CouplingError("Stage outside 0..%d" % (len(counts) - 1))
    if n < 1 or counts[src] < n:
        raise CouplingError("Stage %d has only %d layers, %d requested" % (src, counts[src], n))
    after_counts = list(counts)
    after_counts[src] -= n
    after_counts[dst] += n
    s2 = dict(settings)
    s2["stage_layers"] = after_counts
    if "attn_layers" in settings:
        s2.pop("attn_layers")   # gepinnte Attention-Zaehlung gilt fuer den alten Schnitt; neu aus den Familien
    before = _stage_terms(hw, model, settings)
    after = _stage_terms(hw, model, s2)
    rows = []
    for b, a in zip(before["stages"], after["stages"]):
        rows.append({
            "ord": b["ord"], "label": b["label"], "layers": [b["layers"], a["layers"]],
            "attn_layers": [b["attn_layers"], a["attn_layers"]],
            "weights_mib": [b["weights_total_mib"], a["weights_total_mib"]],
            "free_mib": [b["free_mib"], a["free_mib"]],
            "kv_capacity_tokens": [b["kv_capacity_tokens"], a["kv_capacity_tokens"]],
            "decode_ms": [b["decode_ms"], a["decode_ms"]],
            "overflow_mib": [b["overflow_mib"], a["overflow_mib"]],
        })
    hints = []
    bs, bd, as_, ad = before["stages"][src], before["stages"][dst], after["stages"][src], after["stages"][dst]
    hints.append(
        "You move %d layers from %s to %s (attention layers %d -> %d and %d -> %d). %s: %+.0f MiB free%s. %s: %+.0f MiB free%s." % (
            n, bs["label"], bd["label"], bs["attn_layers"], as_["attn_layers"], bd["attn_layers"], ad["attn_layers"],
            bs["label"], as_["free_mib"] - bs["free_mib"], _tok_delta(bs, as_),
            bd["label"], ad["free_mib"] - bd["free_mib"], _tok_delta(bd, ad)))
    if before["makespan_ms"] is not None and after["makespan_ms"] is not None:
        hints.append("Round time (approximation, decode batch 1): cycle %.2f -> %.2f ms (the slowest stage determines it)." % (
            before["makespan_ms"], after["makespan_ms"]))
    for st in after["stages"]:
        if st["overflow_mib"] > 0:
            hints.append("%s: %.0f MiB OVER the budget. The planner refuses; with force it starts anyway, an OOM is to be expected at loading." % (
                st["label"], st["overflow_mib"]))
    fb, fa = before["context_floor_tokens"], after["context_floor_tokens"]
    if fb is not None and fa is not None:
        hints.append("Context floor (minimum over the stages): %d -> %d tokens." % (fb, fa))
    return {
        "id": "C1", "move": {"src": src, "dst": dst, "n": n}, "before_layers": counts, "after_layers": after_counts, "rows": rows,
        "context_floor_tokens": [fb, fa], "makespan_ms": [before["makespan_ms"], after["makespan_ms"]], "hints": hints,
        "warnings": before["warnings"],
    }


def _tok_delta(b: Mapping[str, Any], a: Mapping[str, Any]) -> str:
    x, y = b["kv_capacity_tokens"], a["kv_capacity_tokens"]
    if x is None or y is None:
        return ""
    return " (KV capacity %+d tokens)" % (y - x)


# ---------------------------------------------------------------------------
# C2: MoE-Experten <-> KV
# ---------------------------------------------------------------------------


def c2_expert_residency(hw: Mapping[str, Any], model: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Je Stufe die GROESSTE Experten-Fraction, die nach KV-Preis (Kontextziel), Zustand, Aktivierung und Festposten noch ins Budget passt
    (``pp_cut.solve_expert_fraction_per_stage``), gegen die gesetzte Fraction.  Ohne Experten (dichtes Modell): keine Kopplung."""
    E = int(_val(model.get("experts", {}).get("n"), 0) or 0)
    if E <= 0:
        return {"id": "C2", "applicable": False, "text": "Dense model: no experts, no coupling."}
    t = _stage_terms(hw, model, settings)
    ctx = t["_ctx"]
    w = model["weights"]
    counts = ctx["counts"]
    n = len(counts)
    total_layers = sum(counts)
    # Gewicht ohne Expertenzeilen je Layer: Mittel; Experten je Layer: Mittel (fuer die Loesungsfunktion skalar)
    dense_mean = sum(_val(w["layer_bytes"])) / float(total_layers) / MIB
    exp_layer = sum(_val(w["layer_expert_bytes"])) / float(total_layers) / MIB
    nonlayer = [0.0] * n
    nonlayer[0] += float(_val(w.get("embed_bytes"), 0.0)) / MIB
    nonlayer[-1] += float(_val(w.get("lm_head_bytes"), 0.0)) / MIB
    if settings.get("draft"):
        nonlayer[-1] += float(_val(w.get("mtp_bytes"), 0.0)) / MIB
    for r in settings.get("replicated") or ():
        if r != "mtp":
            nonlayer = [x + float(_val(w.get(r + "_bytes"), 0.0)) / MIB for x in nonlayer]
    reserve = []
    for i, s in enumerate(t["stages"]):
        reserve.append(s["terms"]["kv"]["v"] + s["terms"]["state"]["v"] + s["terms"]["activation"]["v"] + s["terms"]["fixed"]["v"] + nonlayer[i])
    budgets = [s["budget_mib"]["v"] for s in t["stages"]]
    fr_max = _pp.solve_expert_fraction_per_stage(
        budgets_mib=budgets, stage_layers=counts, mean_layer_mib=dense_mean, expert_layer_mib=exp_layer,
        num_experts=E, lru_rows=ctx["scratch"], reserve_mib_by_stage=reserve)
    fr_set = _per_card(settings.get("moe_resident_fraction", 1.0), n, "moe_resident_fraction", 1.0)
    rows = []
    hints = []
    for i, s in enumerate(t["stages"]):
        rows_set = _er.buffer_rows(local_experts=E, fraction=fr_set[i], scratch_rows=int(ctx["scratch"][i]))
        rows_max = _er.buffer_rows(local_experts=E, fraction=fr_max[i], scratch_rows=int(ctx["scratch"][i]))
        rows.append({"ord": s["ord"], "label": s["label"], "fraction_set": fr_set[i], "fraction_max": round(fr_max[i], 4),
                     "gpu_rows_set": rows_set, "gpu_rows_max": rows_max, "of_rows": E,
                     "fits": fr_set[i] <= fr_max[i] + 1e-9})
        if fr_max[i] <= 0.0:
            hints.append("%s: not even the dense weights plus KV price fit into the budget (fraction 0)." % s["label"])
        elif fr_set[i] > fr_max[i] + 1e-9:
            hints.append("%s: fraction %.2f is too large; at most %.2f (%d of %d rows per layer on the card) at %d tokens of context." % (
                s["label"], fr_set[i], fr_max[i], rows_max, E, settings.get("context_tokens", 262144)))
        else:
            hints.append("%s: free up to fraction %.2f (set %.2f). More resident experts = fewer host accesses in decode, but less KV." % (
                s["label"], fr_max[i], fr_set[i]))
    return {"id": "C2", "applicable": True, "rows": rows, "hints": hints, "warnings": t["warnings"],
            "note": "Upper bound: draft KV producer and seam staging only via fixed_overhead_mib"}


# ---------------------------------------------------------------------------
# C3: Chunkgroesse <-> Aktivierung <-> KV
# ---------------------------------------------------------------------------


def c3_chunk(hw: Mapping[str, Any], model: Mapping[str, Any], settings: Mapping[str, Any], *, new_chunk_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Aktivierungsspitze = Zeilen x Extend-Rate je Karte; ein anderer Chunk verschiebt MiB zwischen Aktivierung und KV-Kapazitaet."""
    before = _stage_terms(hw, model, settings)
    rate = before["_ctx"]["rate"]
    chunk = before["_ctx"]["chunk"]
    rows, hints = [], []
    after = before
    if new_chunk_tokens is not None and int(new_chunk_tokens) != chunk:
        s2 = dict(settings)
        s2["chunk_tokens"] = int(new_chunk_tokens)
        after = _stage_terms(hw, model, s2)
    for b, a in zip(before["stages"], after["stages"]):
        rows.append({"ord": b["ord"], "label": b["label"], "activation_mib": [b["terms"]["activation"]["v"], a["terms"]["activation"]["v"]],
                     "kv_capacity_tokens": [b["kv_capacity_tokens"], a["kv_capacity_tokens"]],
                     "free_mib": [b["free_mib"], a["free_mib"]]})
    if rate <= 0:
        hints.append("Extend rate unknown (0): the activation is not calculated.")
    elif new_chunk_tokens is not None and int(new_chunk_tokens) != chunk:
        d = after["stages"][0]["terms"]["activation"]["v"] - before["stages"][0]["terms"]["activation"]["v"]
        hints.append("Chunk %d -> %d: %+.0f MiB of activation on each card, correspondingly less KV capacity; a larger chunk gives more prefill throughput, a higher peak and less context." % (chunk, int(new_chunk_tokens), d))
    return {"id": "C3", "chunk_tokens": [chunk, after["_ctx"]["chunk"]], "extend_rate_mib": rate, "rows": rows, "hints": hints,
            "warnings": before["warnings"]}


# ---------------------------------------------------------------------------
# C4: Kontextziel <-> KV-Gesamt
# ---------------------------------------------------------------------------


def c4_context_target(hw: Mapping[str, Any], model: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Braucht das Ziel (``context_tokens``) mehr KV, als die Stufen tragen?  Je Stufe: noetige KV-MiB, vorhandene, fehlende."""
    t = _stage_terms(hw, model, settings)
    ctx = int(settings.get("context_tokens", 262144))
    rows, hints = [], []
    worst_missing = 0.0
    for s in t["stages"]:
        need = s["terms"]["kv"]["v"]
        have = s["budget_mib"]["v"] - (s["needs_mib"] - need)
        miss = max(0.0, need - have)
        worst_missing = max(worst_missing, miss)
        rows.append({"ord": s["ord"], "label": s["label"], "kv_needed_mib": round(need, 3), "kv_available_mib": round(have, 3),
                     "missing_mib": round(miss, 3), "fits": miss <= 0.0, "capacity_tokens": s["kv_capacity_tokens"]})
        if miss > 0:
            hints.append("%s: %d tokens need %.0f MiB of KV, available %.0f MiB: %.0f MiB are missing (capacity %s tokens)." % (
                s["label"], ctx, need, have, miss, s["kv_capacity_tokens"]))
    if not hints:
        hints.append("The context goal of %d tokens fits on every stage." % ctx)
    return {"id": "C4", "context_tokens": ctx, "fits": worst_missing <= 0.0, "rows": rows, "hints": hints, "warnings": t["warnings"]}


# ---------------------------------------------------------------------------
# alles zusammen
# ---------------------------------------------------------------------------


def compute(hw: Mapping[str, Any], model: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Alle rechenbaren Kopplungen (C1-C4) plus die benannten Kanten C5-C7; ``hints`` ist die Klartextliste fuer den Editor."""
    c1 = c1_layer_split(hw, model, settings)
    c2 = c2_expert_residency(hw, model, settings)
    c3 = c3_chunk(hw, model, settings)
    c4 = c4_context_target(hw, model, settings)
    hints: List[str] = []
    for st in c1["stages"]:
        if st["overflow_mib"] > 0:
            hints.append("%s: +%.0f MiB over the budget. The planner refuses (VRAM bar); with force it starts anyway, OOM is to be expected at loading or graph building." % (st["label"], st["overflow_mib"]))
    hints += c4["hints"] + c2.get("hints", [])
    return {
        "schema": SCHEMA,
        "settings": {k: settings[k] for k in sorted(settings)},
        "model_id": model.get("id"),
        "hardware_id": hw.get("id"),
        "c1": c1, "c2": c2, "c3": c3, "c4": c4,
        "edges": [dict(e) for e in EDGES],
        "hints": hints,
        "warnings": c1["warnings"],
    }


# ---------------------------------------------------------------------------
# Einstellungen aus dem Serverprofil, synthetische Hardware (Tests)
# ---------------------------------------------------------------------------


def _ints(s: str) -> Optional[List[int]]:
    try:
        return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]
    except ValueError:
        return None


def settings_from_server(args: Mapping[str, str], model: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Einstellungen aus ``profile_json.args_dict(doc)`` (Flag -> letzter Wert).  Was nicht gesetzt ist, bleibt weg (dann gilt der Standard)."""
    out: Dict[str, Any] = {}
    sr = _ints(args.get("--pp-stage-ratio", ""))
    if sr:
        out["stage_layers"] = sr
    ar = _ints(args.get("--pp-attn-stage-ratio", ""))
    if ar:
        out["attn_layers"] = ar
    bm = args.get("--rank-gpu-memory-mib")
    if bm:
        try:
            out["budget_mib"] = [float(x) for x in bm.replace(" ", "").split(",") if x]
        except ValueError:
            pass
    for flag in ("--chunked-prefill-size", "--p-chunk-max"):
        v = args.get(flag)
        if v and v.isdigit():
            out["chunk_tokens"] = int(v)
            break
    for flag in ("--max-kv-per-request", "--context-length"):
        v = args.get(flag)
        if v and v.isdigit():
            out["context_tokens"] = int(v)
            break
    kd = args.get("--kv-cache-dtype")
    if kd and model is not None and kd in (model["kv"].get("variants") or {}):
        out["kv_dtype"] = kd
    mf = args.get("--rank-moe-resident-fraction")
    if mf:
        try:
            xs = [float(x) for x in mf.replace(" ", "").split(",") if x]
            out["moe_resident_fraction"] = xs if len(xs) > 1 else xs[0]
        except ValueError:
            pass
    return out


def synthetic_hardware(cards: Sequence[Tuple[str, float, Optional[float]]]) -> Dict[str, Any]:
    """Ein Hardwareprofil ``flliper.hardware/1`` im Mindestumfang fuer Tests/Entwurf: ``(Name, vram_total_mib, mem_gemv_gbs | None)`` je Karte."""
    out = []
    for i, (name, total, gemv) in enumerate(cards):
        out.append({
            "ord": i, "nvml_index": i, "name": name, "class_key": name,
            "vram_total_mib": {"v": float(total), "src": "NVML"},
            "mem_gbs": {"gemv": {"v": gemv, "src": "gemessen" if gemv else "not measured"}},
        })
    return {"schema": "flliper.hardware/1", "id": "synthetic", "cards": out}


# ---------------------------------------------------------------------------
# S4b (Auftrag 1432): Balken je Karte und Phase, Browser-Naeherung
# ---------------------------------------------------------------------------

#: Segmente eines Balkens in Zeichenreihenfolge: Schluessel (Term), Beschriftung, Erklaerung fuer den Tooltip
BAR_SEGMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("weights", "Weights", "dense weights of the layers of this stage plus embedding (first stage) or lm_head (last stage)"),
    ("experts", "Experts (resident)", "expert rows on the card by buffer rule min(R + scratch, E) per layer"),
    ("draft", "Draft/MTP", "weight of the MTP layers on the last stage"),
    ("kv", "KV", "context goal x attention layers of the stage (+ one draft row) x KV cell"),
    ("state", "Mamba/GDN state", "linear layers of the stage x state per layer and slot x slots"),
    ("activation", "Activation", "chunk rows x extend rate (peak at prefill)"),
    ("fixed", "Fixed items", "CUDA context, graphs, allocator remainders, seam staging: measurable only on the hardware (0 without input)"),
)

_ORIGIN = {SRC_INPUT: "Input (user/profile)", SRC_DEFAULT: "Assumption of this calculation", SRC_DERIVED: "computed"}


def _origin(src: str) -> str:
    return _ORIGIN.get(src, "Model profile/hardware profile (%s)" % src)


def _clip_to_budget(segs: List[Dict[str, Any]], budget: float) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    """Segmente bis ``budget`` behalten; was darueber liegt, kommt (von hinten abgeschnitten) in die Overflowliste.  Die Aufteilung
    des Overflows auf Posten ist Darstellung, die Summe ist exakt."""
    kept: List[Dict[str, Any]] = []
    cut: List[Dict[str, Any]] = []
    at = 0.0
    for s in segs:
        room = max(0.0, budget - at)
        inside = min(s["mib"], room)
        if inside > 0:
            kept.append(dict(s, mib=round(inside, 3)))
        beyond = s["mib"] - inside
        if beyond > 1e-9:
            cut.append({"key": s["key"], "label": s["label"], "mib": round(beyond, 3)})
        at += s["mib"]
    return kept, round(sum(c["mib"] for c in cut), 3), cut


def stage_bar(stage: Mapping[str, Any]) -> Dict[str, Any]:
    """Ein Karten-Balken aus einer Stufe von ``c1_layer_split``: Posten, Rest im Budget, Korridor, **Overflow als eigenes rotes Segment**.

    Zeichenfolge: Posten (bis zum Budget) | Rest im Budget | Corridor/reserve (Kartengroesse - Budget).  Ueberschreitet die Summe das
    Budget, steht der Teil darueber als Segment ``overflow`` (rot) VOR der Reserve; der Balken waechst dann ueber die Kartenkante, wenn
    der Overflow die Reserve ueberschreitet.  Kein stilles Beschneiden: ``overflow_mib`` und die betroffenen Posten stehen im Segment."""
    budget = float(stage["budget_mib"]["v"])
    total = float(stage["total_mib"])
    segs = []
    for key, label, what in BAR_SEGMENTS:
        term = stage["terms"][key]
        if term["v"] > 0:
            segs.append({"key": key, "label": label, "mib": term["v"], "src": term["src"], "origin": _origin(term["src"]),
                         "what": what + (" -- " + term["note"] if term.get("note") else "")})
    kept, overflow, cut = _clip_to_budget(segs, budget)
    free = max(0.0, budget - sum(s["mib"] for s in kept))
    out = list(kept)
    if overflow > 0:
        out.append({"key": "overflow", "label": "Overflow", "mib": overflow, "src": SRC_DERIVED, "origin": "computed",
                    "what": "Items over the budget (%.0f MiB): %s" % (budget, ", ".join("%s %.0f MiB" % (c["label"], c["mib"]) for c in cut)),
                    "cut": cut})
    elif free > 0:
        out.append({"key": "free_in_budget", "label": "Rest in the budget", "mib": round(free, 3), "src": SRC_DERIVED, "origin": "computed",
                    "what": "Budget - items (upper bound as long as fixed items are not measured)"})
    reserve = max(0.0, total - budget)
    if reserve > 0:
        out.append({"key": "corridor", "label": "Corridor/reserve", "mib": round(reserve, 3), "src": stage["budget_mib"]["src"],
                    "origin": _origin(stage["budget_mib"]["src"]), "what": "Card size - budget: stays free (reserve semantics)"})
    over_text = ""
    if overflow > 0:
        over_text = ("%s: +%.0f MiB over the budget. The planner refuses; with force it starts anyway, OOM is to be expected at loading or graph building." % (stage["label"], overflow))
    return {"ord": stage["ord"], "label": stage["label"], "total_mib": total, "budget_mib": budget, "segments": out,
            "free_mib": round(free, 3), "overflow_mib": overflow, "needs_mib": stage["needs_mib"], "over_text": over_text}


def bars_for(hw: Mapping[str, Any], model: Mapping[str, Any], settings: Mapping[str, Any],
             phases: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """Balken je Karte, je Phase.  ``phases`` = ``{"P": {Einstellungen, die gelten sollen}, "D": {...}}`` (jeweils ueber ``settings`` gelegt);
    ohne Angabe eine Phase ``alle``.  ``Spitze`` = je Karte die Phase mit dem groessten Bedarf (gleichzeitig belegt wird nie mehr)."""
    ph_in = dict(phases) if phases else {"alle": {}}
    per: Dict[str, Any] = {}
    for name, over in ph_in.items():
        s = dict(settings)
        s.update(over or {})
        c1 = c1_layer_split(hw, model, s)
        per[name] = {"bars": [stage_bar(st) for st in c1["stages"]], "context_floor_tokens": c1["context_floor_tokens"],
                     "makespan_ms": c1["makespan_ms"], "warnings": c1["warnings"]}
    out: Dict[str, Any] = {"phases": per}
    if len(per) > 1:
        names = list(per)
        peak = []
        for i in range(len(per[names[0]]["bars"])):
            best = max(names, key=lambda n: per[n]["bars"][i]["needs_mib"])
            peak.append(dict(per[best]["bars"][i], from_phase=best))
        out["Spitze"] = {"bars": peak}
    hints = []
    for name, ph in per.items():
        for b in ph["bars"]:
            if b["over_text"]:
                hints.append(("%s phase: " % name if name != "alle" else "") + b["over_text"])
    out["hints"] = hints
    return out


def approx_payload(hw: Mapping[str, Any], model: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Alles, was der Browser fuer die sofortige Naeherung braucht (``profil_balken.js::approx``): lineare Arithmetik, kein Solver.
    Gilt fuer eine Aenderung des LAYER-SCHNITTS bei sonst gleichen Einstellungen; Experten-Anteil, Kontext, Chunk, Budget bleiben die des Servers."""
    t = _stage_terms(hw, model, settings)
    ctx = t["_ctx"]
    w = model["weights"]
    fams = _families(model)
    replicated = tuple(settings.get("replicated") or ())
    rep_mib = sum(float(_val(w.get(r + "_bytes"), 0.0)) for r in replicated) / MIB
    return {
        "n_stages": len(ctx["counts"]),
        "stage_layers": ctx["counts"],
        "layer_dense_mib": [round(x / MIB, 4) for x in _val(w["layer_bytes"])],
        "layer_expert_mib": [round(x / MIB, 4) for x in _val(w["layer_expert_bytes"])],
        "layer_attn": [1 if f == "attn" else 0 for f in fams],
        "embed_mib": round(float(_val(w.get("embed_bytes"), 0.0)) / MIB, 4),
        "lm_head_mib": round(float(_val(w.get("lm_head_bytes"), 0.0)) / MIB, 4),
        "replicated_mib": round(rep_mib, 4),
        "draft_mib": round(float(_val(w.get("mtp_bytes"), 0.0)) / MIB, 4) if settings.get("draft") and "mtp" not in replicated else 0.0,
        "draft_layers": 1 if settings.get("draft") else 0,
        "buf_fracs": [round(x, 6) for x in ctx["buf_fracs"]],
        "cell_mib": ctx["cell_mib"],
        "context_tokens": int(settings.get("context_tokens", 262144)),
        "chunk_rows": ctx["chunk"],
        "extend_rate_mib": ctx["rate"],
        "activation_mib": ctx["act_vec"],
        "state_per_layer_mib": ctx["state_per"],
        "slots": ctx["slots"],
        "fixed_mib": ctx["fixed"],
        "budget_mib": ctx["budgets"],
        "total_mib": [s["total_mib"] for s in t["stages"]],
    }


def approx_terms(pl: Mapping[str, Any], stage_layers: Sequence[int]) -> List[Dict[str, float]]:
    """Die Naeherung, Python-Referenz zu ``profil_balken.js::approx`` (dieselbe Arithmetik, Zeile fuer Zeile): je Stufe Posten und frei."""
    n = int(pl["n_stages"])
    counts = [int(c) for c in stage_layers]
    if len(counts) != n or sum(counts) != len(pl["layer_dense_mib"]) or min(counts) < 0:
        raise CouplingError("approx: stage_layers does not match the model (%s)" % counts)
    out, start = [], 0
    for i, c in enumerate(counts):
        sl = slice(start, start + c)
        dense = sum(pl["layer_dense_mib"][sl]) + pl["replicated_mib"]
        if i == 0:
            dense += pl["embed_mib"]
        if i == n - 1:
            dense += pl["lm_head_mib"]
        experts = pl["buf_fracs"][i] * sum(pl["layer_expert_mib"][sl])
        draft = pl["draft_mib"] if i == n - 1 else 0.0
        attn = sum(pl["layer_attn"][sl])
        lin = c - attn
        kv = pl["context_tokens"] * (attn + pl["draft_layers"]) * pl["cell_mib"]
        state = lin * pl["state_per_layer_mib"] * pl["slots"]
        act = pl["activation_mib"][i] if pl.get("activation_mib") is not None else pl["chunk_rows"] * pl["extend_rate_mib"]
        fixed = pl["fixed_mib"][i]
        need = dense + experts + draft + kv + state + act + fixed
        out.append({"weights": dense, "experts": experts, "draft": draft, "kv": kv, "state": state, "activation": act, "fixed": fixed,
                    "needs": need, "free": pl["budget_mib"][i] - need})
        start += c
    return out


# ---------------------------------------------------------------------------
# AP-H2: Balken je Karte und Phase im Datenvertrag ``flliper.balken/1``
# ---------------------------------------------------------------------------
#
# EIN zusammenhaengender Balken je Karte und Phase in logischer Reihenfolge
#     Gewichte | Experten | Draft | KV | Mamba | (Aktivierung | Festposten) | Reserve | Frei
# Phasen: ``P`` (Prefill, Pipeline-Stufen) und ``D`` (Decode, TP-Raenge); Dual zeigt beide zugleich, nur-TP nur ``D``, Einzelkarte eine
# Phase ``alle``.  Der Vertrag steht im Modulkopf von ``static/profil_balken.js``; ein Orakel (propose/Dry-Run, AP-C/AP-D) kann Werte in
# DIESELBE Form liefern, ohne dass die Darstellung sich aendert.
#
# EHRLICHKEIT.  Ein Term, der sich aus den Profilzeilen und dem Modellprofil nicht belegen laesst, hat ``mib: None`` und
# ``herkunft: "nicht gerechnet"`` mit Grund (``detail``); er zaehlt nicht in die Summe, ``Frei`` ist dann eine Obergrenze.  Nichts wird
# geraten (Hochrechnung != Messung).  Werte "Naeherung" sind Rechnungen dieses Moduls, nicht die des Loesers/Launchers.

BALKEN_SCHEMA = "flliper.balken/1"
SRC_PROFILE = "Profile row"
SRC_APPROX = "Approximation"
SRC_NONE = "not calculated"
FORMS = ("single", "d_only", "flip", "dual")

BALKEN_SEGMENTS: Tuple[Tuple[str, str, str], ...] = BAR_SEGMENTS + (
    ("reserve", "Reserve", "Card size - budget: stays free (reserve semantics); is consumed by items above the budget"),
    ("free", "Free", "Budget - items (upper bound as long as items are not calculated)"),
)
#: Erklaerung je Posten der D-Phase (TP-Raenge): die Texte von BAR_SEGMENTS beschreiben die P-Stufen (Layer-Schnitt, letzte Stufe)
D_WHAT = {
    "weights": "dense weights, embedding and lm_head of this rank (TP share)",
    "experts": "expert rows of this rank by buffer rule min(R + scratch, own experts) per layer",
    "draft": "weight of the draft on this rank",
    "kv": "KV share of this rank for the context goal",
    "state": "Mamba/GDN state share of this rank: linear layers x state per layer and slot x slots x TP share",
    "activation": "decode activation of this rank",
    "fixed": "CUDA context of the sleeping phase and VRAM outside the torch allocator (D side)",
}
#: Erklaertext des D-Festpostens unter --dual-share: dort ist er P's wacher Plan (dc = P-Budget + Overhead), NICHT der CUDA-Kontext einer schlafenden Phase
DUAL_D_FIXED_WHAT = ("P's plan on this card under --dual-share (P budget + --dual-p-overhead-mib): P is awake and holds this when D is sized as union owner; the launcher deducts it as dormant_other (launcher.py:14976, :27289-27301)")
_ORIGIN.update({SRC_PROFILE: "Profile row", SRC_APPROX: "Approximation (not the solver)", SRC_NONE: "not calculated"})

_SPEC_FLAGS = ("--speculative-algorithm", "--speculative-draft-model-path", "--dflash-draft-path", "--spec-form")


def _fl(s: Any) -> Optional[List[float]]:
    """``"1,0,0"`` -> ``[1.0, 0.0, 0.0]``; leer oder unlesbar -> ``None``."""
    try:
        out = [float(x) for x in str(s).replace(" ", "").split(",") if x != ""]
    except ValueError:
        return None
    return out or None


def _merged(args: Mapping[str, str], phase_args: Optional[Mapping[str, Any]], phase: str) -> Dict[str, str]:
    """Flag -> Wert einer Phase: die Zeilen des Profils, darueber die ``--extra-p`` / ``--extra-d`` der Gruppe (argparse: Gruppe gewinnt)."""
    out = dict(args or {})
    out.update(dict((phase_args or {}).get(phase) or {}))
    return out


def _draft_info(model: Mapping[str, Any], all_args: Mapping[str, str]) -> Dict[str, Any]:
    """Was der Draft ist und was er wiegt -- nur aus Profilzeilen und Modellprofil.

    ``kind``: ``none`` | ``nextn`` (MTP-Kopf, launcher.SPEC_FORM_DEFAULT = NEXTN) | ``dflash`` (Platzierung nicht gerechnet).
    ``p_mib``/``d_mib``: Gewichte auf P (teilt ``lm_head`` mit dem Ziel, ``draft_post.P_SHARED_WITH_TARGET``) bzw. D (teilt Einbettung und
    ``lm_head``, ``draft_post.D_SHARED_WITH_TARGET``), ohne Laufzeitpuffer; ``None`` = nicht belegbar."""
    w = model.get("weights") or {}
    mtp = float(_val(w.get("mtp_bytes"), 0.0) or 0.0) / MIB
    ext = (model.get("draft") or {}).get("external")
    flagged = any(all_args.get(f) for f in _SPEC_FLAGS)
    form = str(all_args.get("--spec-form") or "").upper()
    algo = str(all_args.get("--speculative-algorithm") or "").upper()
    info: Dict[str, Any] = {"kind": "none", "p_mib": 0.0, "d_mib": 0.0, "src": SRC_DEFAULT, "p_note": "", "d_note": "", "reason": "",
                            "attn_layers": 1, "placement": str(all_args.get("--speculative-draft-placement") or "split"),
                            "gpu": all_args.get("--speculative-draft-gpu")}
    if all_args.get("--dflash-draft-path") or form == "DFLASH" or algo == "DFLASH":
        info.update(kind="dflash", p_mib=None, d_mib=None,
                    reason="DFlash2 draft: placement and weight per group are not calculated (oracle/AP-D)")
        return info
    if not (flagged or mtp > 0):
        info["reason"] = "no draft: no --speculative-*/--dflash-* flag in the profile and no MTP head in the model"
        return info
    info["kind"] = "nextn"
    if ext and "bytes_without_lm_head" in ext:
        info["p_mib"] = float(_val(ext["bytes_without_lm_head"])) / MIB
        info["d_mib"] = float(_val(ext["bytes_without_embed_lm_head"])) / MIB if "bytes_without_embed_lm_head" in ext else None
        info["src"] = _src(ext["bytes_without_lm_head"])
        info["p_note"] = "Draft directory without lm_head (P shares it with the goal), without runtime buffers"
        info["d_note"] = "Draft directory without embedding and lm_head (D shares them with the goal), without runtime buffers"
        if info["d_mib"] is None:
            info["reason"] = "Draft directory without a split of embedding/lm_head in the profile"
        info["attn_layers"] = int(_val((ext.get("kv") or {}).get("attn_layers"), 1) or 1)
    elif mtp > 0:
        info["p_mib"] = info["d_mib"] = mtp
        info["src"] = _src(w.get("mtp_bytes"))
        info["p_note"] = info["d_note"] = "MTP head of the target checkpoint (mtp.*)"
    else:
        info.update(p_mib=None, d_mib=None, reason="Draft directory is named in the profile but not profiled (no model profile of the draft)")
    return info


def p_phase_settings(args: Mapping[str, str], env: Mapping[str, str], model: Mapping[str, Any], n: int, draft: Mapping[str, Any],
                     *, carries_draft: bool) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Einstellungen fuer ``_stage_terms`` aus den Zeilen der P-Phase (oder der Einzelkarte) und die Liste der gelesenen Eingaben."""
    s = settings_from_server(args, model)
    seen: List[Dict[str, str]] = []

    def note(what: str, value: Any, herkunft: str) -> None:
        seen.append({"was": what, "wert": str(value), "herkunft": herkunft})

    for key, flag in (("stage_layers", "--pp-stage-ratio"), ("attn_layers", "--pp-attn-stage-ratio"), ("budget_mib", "--rank-gpu-memory-mib"),
                      ("chunk_tokens", "--chunked-prefill-size"), ("context_tokens", "--max-kv-per-request")):
        if key in s:
            note(key, args.get(flag, ""), "Profile row " + flag)
    if n == 1 and "stage_layers" not in s:
        s["stage_layers"] = [len(_families(model))]
        note("stage_layers", s["stage_layers"][0], "Single card: all layers on the one card")
    if "moe_resident_fraction" not in s:
        fr = _fl(args.get("--pp-cut-expert-device-fraction")) or _fl(env.get("SGLANG_MOE_RESIDENT_EXPERT_FRACTION"))
        if fr:
            s["moe_resident_fraction"] = fr if len(fr) > 1 else fr[0]
            note("moe_resident_fraction", ",".join("%g" % x for x in fr),
                 "Profile row --pp-cut-expert-device-fraction" if args.get("--pp-cut-expert-device-fraction")
                 else "Environment SGLANG_MOE_RESIDENT_EXPERT_FRACTION (--env-p)")
    else:
        note("moe_resident_fraction", args.get("--rank-moe-resident-fraction", ""), "Profile row --rank-moe-resident-fraction")
    sc = _fl(args.get("--pp-cut-expert-lru-rows")) or _fl(env.get("SGLANG_MOE_SCRATCH_SLOTS"))
    if sc:
        s["scratch_rows"] = [int(x) for x in sc] if len(sc) > 1 else int(sc[0])
        note("scratch_rows", ",".join("%d" % x for x in sc),
             "Profile row --pp-cut-expert-lru-rows" if args.get("--pp-cut-expert-lru-rows") else "Environment SGLANG_MOE_SCRATCH_SLOTS (--env-p)")
    slots = args.get("--max-mamba-cache-size")
    if slots and str(slots).isdigit():
        s["mamba_slots"] = int(slots)
        note("mamba_slots", slots, "Profile row --max-mamba-cache-size")
    ssm = args.get("--mamba-ssm-dtype")
    if ssm and ssm in ((model.get("state") or {}).get("variants_mib") or {}):
        s["ssm_dtype"] = ssm
        note("ssm_dtype", ssm, "Profile row --mamba-ssm-dtype")
    if "budget_mib" not in s:
        note("budget_mib", "Card size - 1024 MiB", "Assumption of this calculation (no --rank-gpu-memory-mib in the profile)")
    for key, default, why in (("chunk_tokens", 2048, "no --chunked-prefill-size/--p-chunk-max in the profile"),
                              ("context_tokens", 262144, "no --max-kv-per-request/--context-length in the profile"),
                              ("mamba_slots", 1, "no --max-mamba-cache-size in the profile")):
        if key not in s:
            note(key, default, "Assumption of this calculation (%s)" % why)
    # Draft: P traegt den MTP-Kopf nur mit --draft-kv-on-p on (Standard on; launcher.py --draft-kv-on-p Hilfe)
    s["draft"] = False
    if carries_draft and draft["kind"] == "nextn" and draft["p_mib"] is not None:
        s["draft"] = True
        s["draft_mib"] = draft["p_mib"]
        s["draft_src"] = draft["src"]
        s["draft_note"] = draft["p_note"]
    return s, seen


def _p_terms_for_contract(stage: Mapping[str, Any], draft: Mapping[str, Any], carries_draft: bool, is_last: bool,
                          settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Posten der ``_stage_terms``-Stufe im Vertragsformat: nicht belegbare Posten werden ``None`` (nicht gerechnet), nie 0."""
    t = {k: dict(v) for k, v in stage["terms"].items()}
    if settings.get("fixed_overhead_mib") is None:
        t["fixed"] = {"v": None, "src": SRC_NONE, "note": "CUDA context, graphs, allocator remainders, seam staging: measurable only on the hardware"}
    if settings.get("activation_mib") is None and not stage.get("_rate_known", True):
        t["activation"] = {"v": None, "src": SRC_NONE, "note": "Extend rate of the model unknown"}
    if is_last and carries_draft and draft["kind"] != "none" and not settings.get("draft"):
        t["draft"] = {"v": None, "src": SRC_NONE, "note": draft["reason"] or "Draft weight cannot be verified"}
    return t


#: Launcher-Standard fuer --dual-p-overhead-mib (launcher.py:22713, ``default=1500``)
DUAL_P_OVERHEAD_DEFAULT_MIB = 1500
#: D-Korridor ohne Messrecord, wie ihn ``launcher.budgets_from_dc`` abzieht: ``corridor = cf.mib + D_AWAKE_OVERSHOOT_MIB`` mit dem Boden des
#: stated law 1024 (``corridor_budget.floors_for_cards``, Quelle UNMEASURED-FALLBACK) und dem eingebauten Wach-Ueberschuss 404
#: (launcher.py:266-269 ``CORRIDOR_MIB = 1024 + D_AWAKE_OVERSHOOT_MIB``).  Gemessene Records (D_AWAKE_REST_MIB, Korridor-Floor) aendern ihn.
D_CORRIDOR_STATED_LAW_MIB = 1024
D_AWAKE_OVERSHOOT_MIB = 404
D_CORRIDOR_ASSUMED_MIB = D_CORRIDOR_STATED_LAW_MIB + D_AWAKE_OVERSHOOT_MIB


def dual_p_plan(p_args: Mapping[str, str], n: int) -> Dict[str, Any]:
    """Was P unter ``--dual-share`` je Karte nach seinem PLAN haelt, aus den Zeilen der Gruppe P.

    Der Launcher bemisst D in dieser Form aus P's Plan (launcher.py:27289-27301): ``dc = dual_share_planned_dc(cards, budgets_p, extra_p,
    --dual-p-overhead-mib)`` = effektives P-Budget + Overhead je Karte (launcher.py:14976-14992; ein ``--rank-gpu-memory-mib`` in ``--extra-p`` senkt
    das Launcher-Budget: ``min``), danach ``budgets_from_dc(cards, dc, ...)`` (launcher.py:27207).  Das eigene P-Budget des Launchers ist ohne Boot
    nicht zu belegen; fehlt die Zeile in der Gruppe P, ist ``budget`` ``None`` (nicht gerechnet)."""
    raw = p_args.get("--rank-gpu-memory-mib")
    vec = _fl(raw) if raw not in (None, "") else None
    if vec is not None and len(vec) != n:
        raise CouplingError("vector_length: --rank-gpu-memory-mib of group P has %d values, there are %d cards" % (len(vec), n))
    ov_raw = p_args.get("--dual-p-overhead-mib")
    try:
        ov = float(ov_raw) if ov_raw not in (None, "") else float(DUAL_P_OVERHEAD_DEFAULT_MIB)
    except (TypeError, ValueError):
        ov, ov_raw = float(DUAL_P_OVERHEAD_DEFAULT_MIB), None
    return {"budget": vec, "budget_raw": raw, "overhead": ov, "overhead_given": ov_raw not in (None, "")}


#: Umgebungsschalter der Torch-Cache-Kappe (launcher.py:5487); steht er in ``--env-d``, gilt er, sonst der Standard der Registerzeile
TORCH_CACHE_CAP_ENV = "SGLANG_WEG2_TORCH_CACHE_CAP"


def parse_user_reserve(raw: Any, n: int) -> List[int]:
    """``--user-reserve-mib`` -> MiB je Karte in RANG-/CUDA-ORDINAL-Reihenfolge (5090 zuerst) -- Spiegel von ``launcher.parse_user_reserve``
    (launcher.py:14118-14165): ein Skalar gilt fuer jede Karte, eine Liste braucht genau einen Wert je Karte, Werte >= 0; leer = 0.
    Der Launcher bricht bei einem Fehler ab (SystemExit); hier ist es ein ``CouplingError`` (die Phase steht dann mit ``ok: False`` da)."""
    text = str(raw if raw is not None else 0).strip()
    if not text:
        return [0] * n
    try:
        vals = [int(p) for p in text.split(",")]
    except ValueError:
        raise CouplingError("--user-reserve-mib must be an integer or a comma-separated list, is %r (launcher.py:14150)" % text)
    if any(v < 0 for v in vals):
        raise CouplingError("--user-reserve-mib: values must be >= 0, are %s (launcher.py:14155)" % vals)
    if len(vals) == 1:
        vals = vals * n
    if len(vals) != n:
        raise CouplingError("vector_length: --user-reserve-mib has %d values, there are %d cards (launcher.py:14160)" % (len(vals), n))
    return vals


def launcher_d_budgets(totals: Sequence[float], dc: Sequence[float], *, profile: str, user_reserve: Sequence[int],
                       carve: Optional[Sequence[Optional[float]]] = None, env_d: Optional[Mapping[str, str]] = None,
                       stated_law_mib: float = float(D_CORRIDOR_STATED_LAW_MIB)) -> Dict[str, Any]:
    """Das D-Budget je Karte, wie ``launcher.budgets_from_dc`` es fuer die Gruppe D rechnet (launcher.py:15252-15394), mit denselben Eingaengen.

    Zwei Pfade je Karte, wie im Launcher:

    * **gebuchter Rest** (Record ``D_AWAKE_REST_BOOKED_MIB`` des Profils, Registerzeile ``budget_rest_from_records``; launcher.py:15318-15353):
      ``(Karte - Carve - dc - Wachstum - Rest) // 8 * 8``.  Der Rest enthaelt den Korridor-Boden; Nutzerreserve, 404 und Ueberschuss werden NICHT
      daneben gebucht (launcher.py:15316-15322).
    * **Boden + Reserve** (kein Record fuer diese Karte): ``corridor = Boden + Nutzerreserve + (404 ohne Wach-Rest-Record)``,
      ``(Karte - corridor - dc - Wachstum - Ueberschuss - Carve - Wach-Rest) // 8 * 8`` (launcher.py:15373-15394).

    Records kommen aus der Registerzeile des Profils (``weg2/form.py``, dieselbe Quelle wie ``launcher._pconst``).  Nicht gerechnet (und in
    ``gaps`` benannt, damit der Tooltip es sagt): gemessener Korridor-Boden und ``SGLANG_CORRIDOR_LAW_FLOOR_MIB`` (hier das stated law 1024), der
    Korridor-Pass mit Messprobe (launcher.py:15425-15462; er kann das Budget nur senken), der Treiber-Carve (``Card.reserved_mib``, NVML; das
    Hardwareprofil traegt ihn nicht), der Umgebungsschalter ``SGLANG_WEG2_BUDGET_REST_RECORD`` des Launchers.
    ``carve`` ist je Karte MiB oder ``None`` (unbekannt)."""
    n = len(totals)
    gaps: List[str] = []
    seen: List[Dict[str, str]] = []
    form_mod = None
    row = None
    try:
        from sglang.srt.weg2 import form as form_mod  # noqa: F811
        row = form_mod.profile_row(profile)
    except Exception as exc:  # pragma: no cover - ohne weg2-Baum nicht pruefbar
        gaps.append("Profile records not readable (%s: %s)" % (type(exc).__name__, exc))
    if row is None and form_mod is not None:
        gaps.append("Profile %r is unknown in the registry: records (growth, awake residue, excess, booked rest) not calculated" % profile)

    def record(name: str, conv: Any) -> Tuple[Optional[List[Any]], str]:
        if row is None:
            return None, ""
        try:
            vals = list(form_mod.profile_constant(name, profile))
        except KeyError:
            return None, ""
        if len(vals) != n:
            gaps.append("Record %s has %d entries, there are %d cards (the launcher derives them for the subset, weg2/inventory_view.py): not calculated"
                        % (name, len(vals), n))
            return None, ""
        rec = row.constants.get(name)
        boots = tuple(getattr(rec, "boots", ()) or ()) if rec is not None else ()
        return [None if v is None else conv(v) for v in vals], ("%d boots, first %s" % (len(boots), boots[0])) if boots else "Record of the profile"

    grow, grow_src = record("P_DORMANT_SERVED_GROWTH_MIB", int)
    rest, rest_src = record("D_AWAKE_REST_MIB", int)
    over_rec, over_src = record("D_OVERSHOOT_MIB", int)
    booked: Optional[List[Optional[int]]] = None
    booked_src = ""
    if row is not None and bool(getattr(row, "budget_rest_from_records", False)):
        capped_env = (env_d or {}).get(TORCH_CACHE_CAP_ENV)
        capped = (str(capped_env).strip() == "1") if capped_env is not None else bool(getattr(row, "torch_cache_cap", False))
        name = "D_AWAKE_REST_BOOKED_MIB"
        if capped:
            cap_vals, cap_src = record("D_AWAKE_REST_CAPPED_MIB", int)
            if cap_vals is not None:
                booked, booked_src = cap_vals, "D_AWAKE_REST_CAPPED_MIB " + cap_src
        if booked is None:
            booked, booked_src = record(name, int)
            booked_src = ("%s %s" % (name, booked_src)) if booked is not None else ""
        gaps.append("Launcher environment SGLANG_WEG2_BUDGET_REST_RECORD (=0 switches the booked rest off; the launcher environment is not visible in the profile)")
    carve_on = row is not None and bool(getattr(row, "budget_charges_driver_carve", False))
    if carve_on or booked is not None:
        min_total = int(getattr(row, "driver_carve_min_total_mib", 0) or 0) if row is not None else 0
        if carve is None or any(c is None for c in carve):
            gaps.append("Driver carve (Card.reserved_mib from NVML; profile %s books it %s%s): the hardware profile does not carry it, not calculated -- the budget is higher than in the launcher by the carve (518 MiB 5090 / 425 MiB 3080, kartenplan_catalog.py:24-25)" % (
                            profile, ("from %d MiB card" % min_total) if (carve_on and min_total) else "on every card",
                            "; in the booked-rest path on every card, launcher.py:15331" if booked is not None else ""))
        charged = [carve_on and (min_total <= t or _uncalibrated(totals[i])) for i, t in enumerate(totals)]
    else:
        charged = [False] * n
    gaps.append("measured corridor floor and SGLANG_CORRIDOR_LAW_FLOOR_MIB (corridor_guard.corridor_floor_mib, launcher.py:15323): here the stated law %d MiB" % int(stated_law_mib))
    gaps.append("Corridor pass with measurement sample (--corridor-budget-sample, launcher.py:15425-15462): can only lower the budget")

    budgets: List[float] = []
    notes: List[str] = []
    for i in range(n):
        t, d = float(totals[i]), float(dc[i])
        g = float(grow[i]) if grow and grow[i] is not None else 0.0
        cvk = float(carve[i]) if (carve is not None and carve[i] is not None) else 0.0
        cv = cvk if charged[i] else 0.0
        carve_txt = (" - Carve %.0f" % (cvk if (booked is not None and booked[i] is not None) else cv)) if (cv or (cvk and booked is not None and booked[i] is not None)) else ""
        grow_txt = (" - growth %.0f (P_DORMANT_SERVED_GROWTH_MIB, %s)" % (g, grow_src)) if g else ""
        if booked is not None and booked[i] is not None:
            rb = float(booked[i])
            b = (int(t - cvk - d - g - rb) // 8) * 8                  # der gebuchte-Rest-Pfad bucht den Carve auf JEDER Karte (launcher.py:15331)
            notes.append("Launcher formula budgets_from_dc, booked rest (launcher.py:15318-15353): card %.0f%s - dc %.0f%s - booked rest %.0f (%s), rounded down to 8 MiB. The rest contains the corridor floor; --user-reserve-mib (%d), 404 and excess are not booked alongside (launcher.py:15316-15322)"
                         % (t, carve_txt, d, grow_txt, rb, booked_src, user_reserve[i]))
        else:
            rs = rest[i] if rest and rest[i] is not None else None
            ov = 0.0 if rs is not None else (float(over_rec[i]) if over_rec and over_rec[i] is not None else 0.0)
            builtin = 0.0 if rs is not None else float(D_AWAKE_OVERSHOOT_MIB)
            res = float(user_reserve[i])
            corridor = stated_law_mib + res + builtin
            awake = float(rs) if rs is not None else 0.0
            b = (int(t - corridor - d - g - ov - cv - awake) // 8) * 8
            parts = "Corridor %.0f (floor %.0f + user reserve %.0f (--user-reserve-mib)%s)" % (
                corridor, stated_law_mib, res, (" + built-in awake excess %d" % D_AWAKE_OVERSHOOT_MIB) if rs is None else "")
            notes.append("Launcher formula budgets_from_dc (launcher.py:15373-15394): card %.0f - %s - dc %.0f%s%s%s%s, rounded down to 8 MiB"
                         % (t, parts, d, grow_txt, (" - measured excess %.0f (D_OVERSHOOT_MIB, %s)" % (ov, over_src)) if ov else "", carve_txt,
                            (" - awake residue %.0f (D_AWAKE_REST_MIB, %s; replaces the 404)" % (awake, rest_src)) if rs is not None else ""))
        budgets.append(float(max(b, 0)))
    for what, vec, src in (("P_DORMANT_SERVED_GROWTH_MIB", grow, grow_src), ("D_AWAKE_REST_MIB", rest, rest_src),
                           ("D_OVERSHOOT_MIB", over_rec, over_src)):
        if vec is not None:
            seen.append({"was": what, "wert": ",".join("-" if v is None else str(v) for v in vec),
                         "herkunft": "Record of profile %s (%s; source weg2/form.py, like launcher._pconst)" % (profile, src)})
    if booked is not None:
        seen.append({"was": "D_AWAKE_REST_BOOKED_MIB", "wert": ",".join("-" if v is None else str(v) for v in booked),
                     "herkunft": "Record of profile %s (%s); booked instead of floor + reserve + 404 (launcher.py:15318-15353)" % (profile, booked_src)})
    return {"budgets": budgets, "notes": notes, "gaps": gaps, "seen": seen, "profile": profile}


def _uncalibrated(total_mib: float) -> bool:
    """Karte ohne Kalibrierklasse bucht den Carve immer (launcher.driver_carve_charged, launcher.py:14642): im Profil steht nur die Groesse, ein Modell
    gibt es nicht -- die beiden kalibrierten Klassen (3080 20 GB, 5090 32 GB) haben ihre Groesse; jede andere Groesse gilt als unkalibriert."""
    return not any(abs(total_mib - c) <= c * 0.02 for c in (20480.0, 32607.0))


def d_phase_config(args: Mapping[str, str], env: Mapping[str, str], model: Mapping[str, Any], n: int, draft: Mapping[str, Any],
                   dual_plan: Optional[Mapping[str, Any]] = None, launcher_args: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Einstellungen der D-Phase (TP-Raenge) aus den Zeilen der Gruppe D; ``seen`` = gelesene Eingaben mit Herkunft.

    ``dual_plan`` (``dual_p_plan``, nur Form dual mit ``--dual-share``): der Launcher bemisst D dann aus P's Plan, nicht aus
    ``--rank-gpu-memory-mib`` / ``--d-foreign-context-mib`` / ``--d-nontorch-mib`` der Gruppe D (launcher.py:27289-27301).
    ``launcher_args``: die Zeilen des Profils fuer den LAUNCHER selbst (``--user-reserve-mib``, ``--profile``), ohne ``--extra-d``: argparse des
    Launchers liest sie, nicht der Rang."""
    seen: List[Dict[str, str]] = []

    def note(what: str, value: Any, herkunft: str) -> None:
        seen.append({"was": what, "wert": str(value), "herkunft": herkunft})

    cfg: Dict[str, Any] = {"n": n, "draft": draft}
    la = dict(launcher_args or {})
    prof = str(la.get("--profile") or "").strip()
    if not prof:
        try:
            from sglang.srt.weg2 import form as _form
            prof = str(_form.DEFAULT_PROFILE)
        except Exception:  # pragma: no cover
            prof = "qwen27b"
        note("profile", prof, "Launcher default (--profile, launcher.py:22562; not set in the profile)")
    else:
        note("profile", prof, "Profile row --profile")
    cfg["profile"] = prof
    cfg["user_reserve_card"] = parse_user_reserve(la.get("--user-reserve-mib"), n)
    note("user_reserve_card", ",".join("%d" % v for v in cfg["user_reserve_card"]),
         "Profile row --user-reserve-mib (rank/ordinal order, 5090 first; raises the corridor floor, lowers the D budget by exactly this amount, launcher.py:15316-15323)" if la.get("--user-reserve-mib") not in (None, "") else
         "Launcher default 0 (no --user-reserve-mib in the profile)")
    cfg["env_d"] = dict(env or {})
    tp = str(args.get("--rank-tp-ratio") or "").strip()
    if tp in ("auto", "auto-performance"):
        cfg["tp_ratio"] = tp
        note("tp_ratio", tp, "Profile row --rank-tp-ratio")
    elif _fl(tp):
        cfg["tp_ratio"] = _fl(tp)
        note("tp_ratio", tp, "Profile row --rank-tp-ratio")
    mr = str(args.get("--rank-moe-ratio") or "").strip()
    if mr and _fl(mr):
        cfg["moe_ratio"] = _fl(mr)
        note("moe_ratio", mr, "Profile row --rank-moe-ratio")
    elif mr:
        cfg["moe_ratio"] = mr
        note("moe_ratio", mr, "Profile row --rank-moe-ratio")
    fr = _fl(args.get("--rank-moe-resident-fraction")) or _fl(env.get("SGLANG_MOE_RESIDENT_EXPERT_FRACTION"))
    if fr:
        cfg["moe_fraction"] = fr if len(fr) > 1 else fr[0]
        note("moe_fraction", ",".join("%g" % x for x in fr),
             "Profile row --rank-moe-resident-fraction" if args.get("--rank-moe-resident-fraction")
             else "Environment SGLANG_MOE_RESIDENT_EXPERT_FRACTION (--env-d)")
    sc = _fl(env.get("SGLANG_MOE_SCRATCH_SLOTS"))
    if sc:
        cfg["scratch"] = [int(x) for x in sc] if len(sc) > 1 else int(sc[0])
        note("scratch", ",".join("%d" % x for x in sc), "Environment SGLANG_MOE_SCRATCH_SLOTS (--env-d)")
    bud = _fl(args.get("--rank-gpu-memory-mib"))
    if dual_plan is not None:
        # --dual-share: D = Union-Owner, bemessen aus P's PLAN (dc = P-Budget + --dual-p-overhead-mib je Karte, launcher.py:27289-27301)
        cfg["dual_share"] = True
        cfg["dual_p_budget"] = dual_plan.get("budget")
        cfg["dual_overhead_mib"] = float(dual_plan.get("overhead", DUAL_P_OVERHEAD_DEFAULT_MIB))
        cfg["dual_overhead_given"] = bool(dual_plan.get("overhead_given"))
        if dual_plan.get("budget") is not None:
            note("dual_p_budget", dual_plan.get("budget_raw"), "Profile row --rank-gpu-memory-mib of group P (launcher.py:14976 dual_share_planned_dc)")
        else:
            note("dual_p_budget", "not in the profile", "no --rank-gpu-memory-mib in group P: the launcher solves P's budget, not calculated here")
        note("dual_p_overhead_mib", "%g" % cfg["dual_overhead_mib"],
             "Profile row --dual-p-overhead-mib" if cfg["dual_overhead_given"] else "Launcher default (--dual-p-overhead-mib 1500, launcher.py:22713)")
        if bud:
            note("budget_mib", args.get("--rank-gpu-memory-mib"),
                 "ignored under --dual-share: the launcher sizes D from P's plan (launcher.py:27289-27301), not from this row")
    elif bud:
        cfg["budget_mib"] = bud
        note("budget_mib", args.get("--rank-gpu-memory-mib"), "Profile row --rank-gpu-memory-mib")
    else:
        note("budget_mib", "Card size - fixed items - corridor %d MiB" % D_CORRIDOR_ASSUMED_MIB,
             "Assumption of this calculation (no --rank-gpu-memory-mib in group D; corridor = %d stated law + %d built-in awake excess, launcher.py:266-269)" % (D_CORRIDOR_STATED_LAW_MIB, D_AWAKE_OVERSHOOT_MIB))
    for flag in ("--max-kv-per-request", "--context-length"):
        v = args.get(flag)
        if v and str(v).isdigit():
            cfg["context_tokens"] = int(v)
            note("context_tokens", v, "Profile row " + flag)
            break
    slots = args.get("--max-mamba-cache-size")
    if slots and str(slots).isdigit():
        cfg["mamba_slots"] = int(slots)
        note("mamba_slots", slots, "Profile row --max-mamba-cache-size")
    for key, default, why in (("context_tokens", 262144, "no --max-kv-per-request/--context-length in group D"),
                              ("mamba_slots", 1, "no --max-mamba-cache-size in group D")):
        if key not in cfg:
            note(key, default, "Assumption of this calculation (%s)" % why)
    kd = args.get("--kv-cache-dtype")
    if kd and kd in ((model.get("kv") or {}).get("variants") or {}):
        cfg["kv_dtype"] = kd
        note("kv_dtype", kd, "Profile row --kv-cache-dtype")
    ssm = args.get("--mamba-ssm-dtype")
    if ssm and ssm in ((model.get("state") or {}).get("variants_mib") or {}):
        cfg["ssm_dtype"] = ssm
        note("ssm_dtype", ssm, "Profile row --mamba-ssm-dtype")
    kr = str(args.get("--rank-kv-ratio") or "coupled").strip()
    cfg["kv_ratio"] = _fl(kr) if _fl(kr) else kr
    if args.get("--rank-kv-ratio"):
        note("kv_ratio", kr, "Profile row --rank-kv-ratio")
    cfg["kv_token_cut"] = bool(args.get("--d-kv-token-cut")) or bool(env.get("SGLANG_UNEVEN_TOKEN_VECTOR"))
    if args.get("--d-kv-token-cut"):
        note("kv_token_cut", args.get("--d-kv-token-cut"), "Profile row --d-kv-token-cut")
    fo, nt = _fl(args.get("--d-foreign-context-mib")), _fl(args.get("--d-nontorch-mib"))
    if dual_plan is not None:
        if fo or nt:
            note("fixed_mib", "%s + %s" % (args.get("--d-foreign-context-mib") or "-", args.get("--d-nontorch-mib") or "-"),
                 "not part of the sizing under --dual-share: the launcher deducts P's plan (dc = P budget + overhead, launcher.py:27289-27301)")
    elif fo and nt and len(fo) == len(nt):
        cfg["fixed_mib"] = [a + b for a, b in zip(fo, nt)]
        cfg["fixed_parts"] = {"fremd": fo, "nichttorch": nt}
        note("fixed_mib", "%s + %s" % (args.get("--d-foreign-context-mib"), args.get("--d-nontorch-mib")),
             "Profile rows --d-foreign-context-mib (CUDA context of the sleeping phase) + --d-nontorch-mib (VRAM outside the torch allocator)")
    elif fo or nt:
        cfg["fixed_mib"] = fo or nt
        cfg["fixed_parts"] = {"fremd": fo, "nichttorch": nt}
        note("fixed_mib", args.get("--d-foreign-context-mib") or args.get("--d-nontorch-mib"),
             "only one of the profile rows --d-foreign-context-mib / --d-nontorch-mib")
    rs = _fl(args.get("--d-reserve-mib"))
    if rs:
        cfg["user_reserve_mib"] = rs
        note("user_reserve_mib", args.get("--d-reserve-mib"),
             "Profile row --d-reserve-mib (the launcher deducts it from the available: launcher.py:19886, pp_cut.d_rank_available_mib)")
    return dict(cfg, seen=seen)


def _norm_shares(ratio: Sequence[float], n: int, name: str) -> List[float]:
    if len(ratio) != n:
        raise CouplingError("vector_length: %s has %d values, there are %d ranks" % (name, len(ratio), n))
    tot = float(sum(ratio))
    if tot <= 0:
        raise CouplingError("%s: sum of the weights is 0" % name)
    return [float(r) / tot for r in ratio]


def d_stage_terms(hw: Mapping[str, Any], model: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Posten je Karte der D-Phase (TP-Raenge) im selben Stufenformat wie ``_stage_terms`` (``terms[k] = {v|None, src, note}``).

    Gewichte, Experten und Zustand folgen den Anteilen aus ``--rank-tp-ratio`` / ``--rank-moe-ratio`` (ohne Zeile: gleichmaessiger TP);
    Experten halten je Rang ``buffer_rows(owned, FR_D, Scratch)`` Zeilen je Layer (Pufferregel).  KV nur dort, wo sich die Verteilung
    belegen laesst (gleichmaessiger TP oder ein ausdruecklicher ``--rank-kv-ratio``-Vektor); sonst ``None``."""
    cards = _cards(hw)
    n = len(cards)
    totals = [float(_val(c.get("vram_total_mib"), 0.0)) for c in cards]
    if any(t <= 0 for t in totals):
        raise CouplingError("Hardware profile: vram_total_mib is missing on a card")
    corridor = float(cfg.get("corridor_mib", D_CORRIDOR_ASSUMED_MIB))
    has_budget = cfg.get("budget_mib") is not None
    # Launcher-Formel (budgets_from_dc, launcher.py:15393-15394): Budget = (Karte - Korridor - dormant_other - ...) // 8 * 8.  ``dormant_other`` (dc)
    # ist in der Flip-Form der CUDA-Kontext der schlafenden Phase (--d-foreign-context-mib + --d-nontorch-mib), in der Form dual mit --dual-share
    # P's PLAN (P-Budget + --dual-p-overhead-mib, launcher.py:27289-27301).  Ohne --rank-gpu-memory-mib ist das Budget die Assumption of this calculation.
    # Die Festposten liegen AUSSERHALB des Budgets (pp_cut.d_rank_available_mib), die Annahme "Karte - Korridor" wuerde sie doppelt zaehlen.
    dual = bool(cfg.get("dual_share"))
    dual_dc: Optional[List[float]] = None
    if dual and cfg.get("dual_p_budget") is not None:
        dual_dc = [b + float(cfg.get("dual_overhead_mib", DUAL_P_OVERHEAD_DEFAULT_MIB))
                   for b in _per_card(cfg.get("dual_p_budget"), n, "dual_p_budget", 0.0)]
    if dual_dc is not None:
        pre_fixed = dual_dc
    elif cfg.get("fixed_mib") is not None and not dual:
        pre_fixed = _per_card(cfg.get("fixed_mib"), n, "fixed_mib", 0.0)
    else:
        pre_fixed = [0.0] * n
    override = "corridor_mib" in cfg                      # Eingabe des Nutzers (Uebersteuerung): ersetzt die Launcher-Formel
    lb_res: Optional[Dict[str, Any]] = None
    if has_budget:
        budgets = _per_card(cfg.get("budget_mib"), n, "budget_mib", 0.0)
    elif override:
        budgets = [float(int(max(t - f - corridor, 0.0)) // 8 * 8) for t, f in zip(totals, pre_fixed)]
    else:
        carve = [(float(_val(c.get("driver_reserved_mib"))) if c.get("driver_reserved_mib") is not None else None) for c in cards]
        lb_res = launcher_d_budgets(totals, pre_fixed, profile=str(cfg.get("profile") or "qwen27b"),
                                    user_reserve=cfg.get("user_reserve_card") or [0] * n, carve=carve, env_d=cfg.get("env_d"))
        budgets = lb_res["budgets"]
    budget_src = SRC_PROFILE if has_budget else (SRC_INPUT if override else SRC_DERIVED)
    gaps_txt = ("; NOT CALCULATED (the launcher may land lower): " + "; ".join(lb_res["gaps"])) if lb_res is not None else ""
    ur_all = list(cfg.get("user_reserve_card") or [0] * n)
    corr_txt = "Corridor %.0f MiB (input, replaces the launcher formula)" % corridor if override else "Corridor by launcher formula"
    budget_notes: List[str]
    if has_budget:
        res_txt = ("; --user-reserve-mib %s raises the corridor floor in the launcher, is not calculated in the explicit budget" % ",".join("%d" % v for v in ur_all)
                   if any(ur_all) else "")
        budget_notes = ["--rank-gpu-memory-mib" + res_txt] * n
    elif override:
        budget_notes = ["Card - fixed items - %s, rounded down to 8 MiB" % corr_txt] * n
    else:
        assert lb_res is not None
        if dual_dc is not None:
            head = ("from P's plan (launcher.py:14976 dual_share_planned_dc, :27289-27301; P budget = value of group P, the launcher takes min(own P budget, value), whose budget is not calculated here): dc = P budget + --dual-p-overhead-mib. ")
        elif dual:
            head = ("UPPER BOUND: under --dual-share the launcher deducts P's plan (P budget + --dual-p-overhead-mib) as dc, but the profile has no --rank-gpu-memory-mib of group P: dc here 0, not calculated. ")
        elif any(pre_fixed):
            head = "dc = fixed items (--d-foreign-context-mib + --d-nontorch-mib; in the real run the measured rest of P after sleep(P)). "
        else:
            head = "dc = 0 (no --d-foreign-context-mib/--d-nontorch-mib in the profile: not booked, not calculated). "
        budget_notes = [head + x + gaps_txt for x in lb_res["notes"]]
    d_inputs = list(lb_res["seen"]) if lb_res is not None else []
    fams = _families(model)
    w = model["weights"]
    lb, le = list(_val(w["layer_bytes"])), list(_val(w["layer_expert_bytes"]))
    dense_total = (sum(lb) + float(_val(w.get("embed_bytes"), 0.0)) + float(_val(w.get("lm_head_bytes"), 0.0))) / MIB
    E = int(_val(model.get("experts", {}).get("n"), 0) or 0)
    ctx = int(cfg.get("context_tokens", 262144))
    slots = int(cfg.get("mamba_slots", 1))

    def none(note: str) -> Dict[str, Any]:
        return {"v": None, "src": SRC_NONE, "note": note}

    # --- Anteile ------------------------------------------------------------------------------------------------------------------
    tp = cfg.get("tp_ratio")
    shares: Optional[List[float]]
    if isinstance(tp, str):
        shares, share_note = None, "--rank-tp-ratio %s: the launcher solves the weights, not calculated" % tp
    elif tp:
        shares, share_note = _norm_shares(tp, n, "--rank-tp-ratio"), "Share by --rank-tp-ratio %s" % ",".join("%g" % x for x in tp)
    else:
        shares, share_note = [1.0 / n] * n, "even TP (no --rank-tp-ratio in the profile)"
    share_src = SRC_PROFILE if (tp and not isinstance(tp, str)) else SRC_APPROX

    # --- Experten ---------------------------------------------------------------------------------------------------------------------
    exp_terms: List[Dict[str, Any]] = []
    mr = cfg.get("moe_ratio")
    if E <= 0:
        exp_terms = [_term(0.0, SRC_DEFAULT, "no experts") for _ in range(n)]
    elif isinstance(mr, str):
        exp_terms = [none("--rank-moe-ratio %s: the launcher solves the allocation" % mr) for _ in range(n)]
    else:
        fracs = _per_card(cfg.get("moe_fraction"), n, "moe_fraction", 1.0)
        scr = _per_card(cfg.get("scratch"), n, "scratch", 0.0)
        per_expert = sum(le) / float(E) / MIB
        if mr:
            ratio = [float(x) for x in mr]
            if len(ratio) != n:
                raise CouplingError("vector_length: --rank-moe-ratio has %d values, there are %d ranks" % (len(ratio), n))
            as_counts = abs(sum(ratio) - E) < 1e-9
            counts = [int(x) for x in ratio] if as_counts else [int(round(E * r / sum(ratio))) for r in ratio]
            how = ("Ownership %s of %d experts" % (",".join(str(c) for c in counts), E)) if as_counts else \
                "Ownership normalised from the ratio to %d experts: %s" % (E, ",".join(str(c) for c in counts))
            for i in range(n):
                try:
                    rows = _er.buffer_rows(local_experts=counts[i], fraction=fracs[i], scratch_rows=int(scr[i])) if counts[i] > 0 else 0
                except ValueError as exc:
                    exp_terms.append(none(str(exc)))
                    continue
                exp_terms.append({"v": rows * per_expert, "src": SRC_APPROX,
                                  "note": "%s; %d of %d own rows per layer resident (FR %.3g, scratch %d), %.3f MiB per row across all layers"
                                          % (how, rows, counts[i], fracs[i], int(scr[i]), per_expert)})
        elif shares is not None:
            for i in range(n):
                bf = _mp.expert_buffer_fraction(E, fracs[i], int(scr[i]))
                exp_terms.append({"v": shares[i] * sum(le) / MIB * bf, "src": SRC_APPROX,
                                  "note": "%s; buffer rule %.0f %% (FR %.3g, scratch %d); no --rank-moe-ratio" % (share_note, 100 * bf, fracs[i], int(scr[i]))})
        else:
            exp_terms = [none(share_note) for _ in range(n)]

    # --- Zustand (Mamba/GDN) ------------------------------------------------------------------------------------------------------------
    attn_total = sum(1 for f in fams if f == "attn")
    lin_total = len(fams) - attn_total
    state_per = float(_val((model.get("state") or {}).get("per_linear_layer_per_slot_mib"), 0.0))
    if cfg.get("ssm_dtype"):
        state_per = float(model["state"]["variants_mib"][cfg["ssm_dtype"]])
    state_total = lin_total * state_per * slots

    # --- KV ------------------------------------------------------------------------------------------------------------------------------
    draft = cfg.get("draft") or {"kind": "none"}
    cell, cell_src = _cell_bytes(model, cfg.get("kv_dtype"))
    cell_mib = cell / MIB
    kv_total = ctx * attn_total * cell_mib
    kvr = cfg.get("kv_ratio")
    kv_shares: Optional[List[float]] = None
    if isinstance(kvr, list):
        kv_shares = _norm_shares(kvr, n, "--rank-kv-ratio")
        kv_note = "Token ownership by --rank-kv-ratio %s (each rank holds all KV heads of its tokens)" % ",".join("%g" % x for x in kvr)
    elif kvr not in (None, "coupled"):
        kv_note = "--rank-kv-ratio %s: the solver distributes the tokens weighted by capacity" % kvr
    elif cfg.get("kv_token_cut"):
        kv_note = "--d-kv-token-cut/SGLANG_UNEVEN_TOKEN_VECTOR set: the planner solves the token ownership"
    elif tp:
        kv_note = ("uneven TP (--rank-tp-ratio): the distribution of KV heads/tokens follows the plan of the runtime (KV heads tab), not calculated")
    else:
        h = int(_val(model["arch"].get("heads_kv"), 0) or 0)
        if h and h >= n and h % n == 0:
            kv_shares, kv_note = [1.0 / n] * n, "even TP: %d KV heads / %d ranks" % (h, n)
        else:
            kv_note = "%d KV heads on %d ranks: standard path of the runtime (replication/token axis), not calculated" % (h, n)

    # --- Draft ------------------------------------------------------------------------------------------------------------------------------
    host: Optional[int] = 0
    if draft.get("gpu") not in (None, ""):
        match = [i for i, c in enumerate(cards) if str(c.get("nvml_index")) == str(draft["gpu"])]
        host = match[0] if match else None
    draft_terms: List[Dict[str, Any]] = [_term(0.0, SRC_DEFAULT, draft.get("reason") or "no draft") for _ in range(n)]
    draft_share: List[float] = [0.0] * n
    if draft["kind"] != "none":
        solo = draft.get("placement") == "solo"
        if draft["d_mib"] is None:
            draft_terms = [none(draft.get("reason") or "Draft weight cannot be verified") for _ in range(n)]
        elif solo:
            if host is None:
                draft_terms = [none("--speculative-draft-gpu %s: rank not derivable from the hardware profile" % draft.get("gpu")) for _ in range(n)]
            else:
                draft_terms = [_term(draft["d_mib"] if i == host else 0.0, draft["src"], "solo on rank %d: %s" % (host, draft["d_note"]))
                               for i in range(n)]
                draft_share = [1.0 if i == host else 0.0 for i in range(n)]
        elif shares is not None:
            draft_terms = [_term(draft["d_mib"] * shares[i], SRC_APPROX,
                                 "split (--speculative-draft-placement split) by TP share: %s" % draft["d_note"]) for i in range(n)]
            draft_share = list(shares)
        else:
            draft_terms = [none(share_note) for _ in range(n)]

    fixed = cfg.get("fixed_mib")
    fixed_vec = _per_card(fixed, n, "fixed_mib", 0.0) if fixed is not None else None
    act = cfg.get("activation_mib")
    act_vec = _per_card(act, n, "activation_mib", 0.0) if act is not None else None

    stages = []
    for i, c in enumerate(cards):
        t: Dict[str, Any] = {}
        t["weights"] = none(share_note) if shares is None else {
            "v": shares[i] * dense_total, "src": share_src,
            "note": "%s (vocabulary follows the TP share here, not --rank-vocab-ratio)" % share_note}
        t["experts"] = exp_terms[i]
        t["draft"] = draft_terms[i]
        if kv_shares is None:
            t["kv"] = none(kv_note)
        else:
            drows = draft.get("attn_layers", 1) if (draft["kind"] == "nextn" and draft_share[i] > 0) else 0
            kv_i = kv_shares[i] * kv_total + (ctx * drows * cell_mib * draft_share[i] if drows else 0.0)
            t["kv"] = {"v": kv_i, "src": SRC_APPROX, "note": "%s: %d tokens x %d attention layers x %.0f B%s" % (
                kv_note, ctx, attn_total, cell, " + draft KV" if drows else "")}
        t["state"] = none(share_note) if shares is None else {
            "v": shares[i] * state_total, "src": SRC_APPROX,
            "note": "%s; %d linear layers x %.4f MiB x %d slot(s)" % (share_note, lin_total, state_per, slots)}
        t["activation"] = _term(act_vec[i], SRC_INPUT, "measured peak (input)") if act_vec is not None else none(
            "Decode activation and graphs of the D phase: measurable only on the hardware")
        if dual_dc is not None:
            t["fixed"] = {"v": dual_dc[i], "src": SRC_PROFILE if cfg.get("dual_overhead_given") else SRC_APPROX, "outside_budget": True,
                          "what": DUAL_D_FIXED_WHAT,
                          "note": "P's plan on this card (dormant_other, OUTSIDE D's budget): P budget %g + --dual-p-overhead-mib %g%s (from P plan, launcher.py:14976 dual_share_planned_dc, :27289-27301)"
                                  % (float(_per_card(cfg.get("dual_p_budget"), n, "dual_p_budget", 0.0)[i]), float(cfg.get("dual_overhead_mib")),
                                     "" if cfg.get("dual_overhead_given") else " (launcher default, not set in the profile)")}
        elif dual:
            t["fixed"] = dict(what=DUAL_D_FIXED_WHAT, **none("under --dual-share the launcher sizes D from P's plan (P budget + --dual-p-overhead-mib, launcher.py:27289-27301); the profile lacks --rank-gpu-memory-mib of group P, the launcher solves P's budget: not calculated"))
        elif fixed_vec is not None:
            parts = cfg.get("fixed_parts") or {}
            f_note = "; ".join("%s %s" % (k, ",".join("%g" % x for x in v)) for k, v in parts.items() if v)
            # Launcher-Semantik (pp_cut.d_rank_available_mib): verfuegbar = Karte - fremd - nichttorch - reserve; gefragt = Budget
            # (--rank-gpu-memory-mib).  Die Festposten liegen also AUSSERHALB des Budgets -> ``outside_budget`` (contract_bar rechnet sie
            # nicht gegen das Budget und zieht sie von der Reserve ab; sonst stuende derselbe Betrag zweimal im Balken).
            t["fixed"] = {"v": fixed_vec[i], "src": SRC_PROFILE, "outside_budget": True,
                          "note": "--d-foreign-context-mib (CUDA context of the sleeping phase) + --d-nontorch-mib (lie OUTSIDE the budget, as in the launcher): " + f_note}
        else:
            t["fixed"] = none("CUDA context, graphs, allocator remainders: measurable only on the hardware (no --d-foreign-context-mib/--d-nontorch-mib in the profile)")
        st = {"ord": c.get("ord", i), "label": _label(c), "total_mib": totals[i],
              "budget_mib": {"v": budgets[i], "src": budget_src, "note": budget_notes[i]}, "terms": t}
        if cfg.get("user_reserve_mib") is not None:
            st["user_reserve_mib"] = _per_card(cfg.get("user_reserve_mib"), n, "user_reserve_mib", 0.0)[i]
        stages.append(st)
    hints = ["D: KV total for %d tokens of context across all ranks: %.0f MiB (%s)" % (ctx, kv_total, cell_src)]
    if cfg.get("user_reserve_mib") is not None:
        hints.append("D: --d-reserve-mib %s enters the availability comparison (launcher: available = card - foreign - nontorch - reserve), but is not a segment of its own in the bar: the launcher holds KV pool, draft and activation in it (pp_cut.d_rank_available_mib), which are shown individually here."
                     % ",".join("%g" % x for x in _per_card(cfg.get("user_reserve_mib"), n, "user_reserve_mib", 0.0)))
    return {"stages": stages, "warnings": [], "hints": hints, "inputs": d_inputs}


def contract_bar(stage: Mapping[str, Any], phase: str) -> Dict[str, Any]:
    """Eine Stufe (``terms[k] = {v|None, src, note}``) -> Karten-Balken im Vertrag ``flliper.balken/1``.

    Posten in logischer Reihenfolge, dann ``Reserve`` (verfuegbar - Budget, von einem Overflow ueber das Budget aufgezehrt) und ``Frei``
    (Budget - Posten im Budget).  Posten mit ``outside_budget`` (D-Phase: ``--d-foreign-context-mib`` + ``--d-nontorch-mib``, Launcher:
    verfuegbar = Karte - fremd - nichttorch - reserve) liegen AUSSERHALB des Budgets: sie zaehlen nicht gegen das Budget, verkleinern aber das
    Verfuegbare (Segment ``ausserhalb_budget: true``).  Die Segmente SIND der Balken: Summe = max(Kartengroesse, Posten ausserhalb + max(Budget,
    Posten im Budget)); ragt sie ueber ``total_mib``, waechst der Balken ueber die Kartengrenze (``beyond_card_mib``) -- nichts wird abgeschnitten.
    ``mib: None`` = nicht gerechnet.  Ist das Budget groesser als das Verfuegbare (``budget_over_available_mib``), meldet der Launcher DARUEBER
    und startet trotzdem (launcher.py:19902-19958; kein raise).  ``stage["user_reserve_mib"]`` (``--d-reserve-mib``, launcher.py:19886) geht in
    den Verfuegbar-Vergleich ein (``pp_cut.d_rank_available_mib``), wird aber nicht als eigenes Segment gezeichnet: der Launcher haelt darin KV-Pool,
    Draft und Aktivierung, die hier schon einzeln stehen.

    Terme mit ``ref`` (Dual-Form, ``--dual-share``) sind REFERENZ ohne Budgetverbrauch: sie stehen in ``shared_with_d`` (nicht in ``segments``,
    nicht in der Summe).  ``ref = "shared"``: Bytes liegen in D's Union-Image bzw. im Karten-KV-Pool; ``ref = "in_festposten"``: der Betrag steckt
    schon im Festposten ``--dual-p-overhead-mib``.  ``stage["ref_extra"]`` traegt Referenzposten ohne Zahl (``mib: None``, nicht gerechnet)."""
    total = float(stage["total_mib"])
    budget = float(stage["budget_mib"]["v"])
    segs: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []
    missing: List[str] = []
    inside = 0.0
    outside = 0.0
    for key, label, what in BAR_SEGMENTS:
        t = stage["terms"].get(key)
        if t is None:
            continue
        if phase == "D":
            what = D_WHAT.get(key, what)
            if t.get("what"):                       # Text je Form/Herkunft (Beispiel: Festposten unter --dual-share = P's Plan, nicht CUDA-Kontext)
                what = str(t["what"])
        v = t.get("v")
        if t.get("ref"):
            if v is not None and v > 0:
                refs.append({"name": key, "label": label, "mib": round(float(v), 3), "ref": t["ref"], "herkunft": _origin(t["src"]),
                             "detail": what + (" -- " + t["note"] if t.get("note") else ""), "gerechnet": True})
            continue
        out_b = bool(t.get("outside_budget"))
        if v is None:
            segs.append({"name": key, "label": label, "mib": None, "herkunft": SRC_NONE, "detail": t.get("note") or what, "gerechnet": False})
            missing.append(label)
        elif v > 0:
            if out_b:
                outside += float(v)
            else:
                inside += float(v)
            seg = {"name": key, "label": label, "mib": round(float(v), 3), "herkunft": _origin(t["src"]),
                   "detail": what + (" -- " + t["note"] if t.get("note") else ""), "gerechnet": True}
            if out_b:
                seg["ausserhalb_budget"] = True
            segs.append(seg)
    for x in stage.get("ref_extra") or ():
        refs.append(dict(x))
        if x.get("mib") is None:
            missing.append(x["label"])
    known = inside + outside
    available = total - outside
    user_reserve = float(stage.get("user_reserve_mib") or 0.0)
    overflow = max(0.0, inside - budget)
    beyond = max(0.0, known - total)
    over_avail = max(0.0, budget - (available - user_reserve))
    reserve = max(0.0, available - max(budget, inside))
    free = max(0.0, min(budget, available) - inside)
    bsrc = stage["budget_mib"]
    if reserve > 0:
        wish = (" -- wish %.0f MiB, of which %.0f MiB consumed" % (available - budget, overflow)) if overflow > 0 else ""
        segs.append({"name": "reserve", "label": "Reserve", "mib": round(reserve, 3), "herkunft": _origin(bsrc["src"]),
                     "detail": BALKEN_SEGMENTS[-2][2] + (" -- " + bsrc["note"] if bsrc.get("note") else "") + wish, "gerechnet": True})
    if free > 0:
        obergrenze = (" -- UPPER BOUND: not calculated are " + ", ".join(missing)) if missing else ""
        segs.append({"name": "free", "label": "Free", "mib": round(free, 3), "herkunft": "gerechnet",
                     "detail": BALKEN_SEGMENTS[-1][2] + obergrenze, "gerechnet": True})
    return {"card": stage.get("ord"), "label": stage["label"], "phase": phase, "total_mib": total, "budget_mib": budget,
            "budget_herkunft": _origin(bsrc["src"]), "segments": segs, "posts_mib": round(known, 3), "free_mib": round(free, 3),
            "overflow_mib": round(overflow, 3), "beyond_card_mib": round(beyond, 3), "outside_budget_mib": round(outside, 3),
            "available_mib": round(available, 3), "budget_over_available_mib": round(over_avail, 3), "user_reserve_mib": round(user_reserve, 3),
            "shared_with_d": refs, "not_computed": missing,
            "over_text": _over_text(stage["label"], phase, overflow, beyond, budget, total, over_avail, outside, user_reserve)}


def _over_text(label: str, phase: str, overflow: float, beyond: float, budget: float, total: float,
               over_avail: float = 0.0, outside: float = 0.0, user_reserve: float = 0.0) -> str:
    # Belegt ist nur, was der Launcher tut: bei Budget > Verfuegbar protokolliert er "DARUEBER" und bootet weiter (launcher.py:19902-19958,
    # log_d_rank_vram_solve: kein raise).  Ob der Planer (propose) ablehnt, ist nicht Sache dieses Balkens -> kein "lehnt ab" ohne Beleg.
    if beyond > 0:
        return ("%s (%s): items %.0f MiB over the CARD (%.0f MiB); OOM is to be expected at loading or graph building." % (label, phase, beyond, total))
    if overflow > 0:
        return ("%s (%s): items %.0f MiB over the budget (%.0f MiB); the reserve is consumed." % (label, phase, overflow, budget))
    if over_avail > 0:
        res = (" - user reserve %.0f MiB (--d-reserve-mib)" % user_reserve) if user_reserve > 0 else ""
        return ("%s (%s): budget %.0f MiB is %.0f MiB larger than what is available (card %.0f - fixed items %.0f MiB outside the budget%s). The launcher reports THIS and starts anyway." % (label, phase, budget, over_avail, total, outside, res))
    return ""


def detect_form(args: Mapping[str, str], tokens: Sequence[str], n: int, form: Optional[str] = None) -> str:
    """Betriebsform aus dem Profil: ``single`` (eine Karte) | ``d_only`` (``--d-only``) | ``dual`` (``--dual-*``) | ``flip`` (Standard)."""
    if form in FORMS:
        return str(form)
    if n == 1:
        return "single"
    if any(str(k).startswith("--dual-") for k in args) or any(str(t).startswith("--dual-") for t in tokens):
        return "dual"
    if "--d-only" in tokens or "--d-only" in args:
        return "d_only"
    return "flip"


def _dual_share_p_terms(stage: Dict[str, Any], args: Mapping[str, str], seen: Optional[List[Dict[str, str]]]) -> None:
    """P-Stufe unter ``--dual-share``: was gegen das P-Budget zaehlt und was nur Referenz ist (AP-H2 Fix-Runde 3, Befund 1).

    Belegt aus dem Launcher (Datei:Zeile im Baum 173161c595):

    * ``--dual-share`` Hilfe (weg2/launcher.py:22707-22712): "P's stage computes on D's TP shards (shells over D's three shards of its layers) and binds
      the shard of the D rank on its card to D's bytes via the union image. D boots right after P as the union OWNER, sized from P's PLANNED budget
      (+ --dual-p-overhead-mib)".  Die Gewichte (und damit die residenten Experten, UNION_ROLES = ``main``, launcher.py:14784) liegen im Union-Image
      von D; P haelt davon nichts im eigenen Budget -> Referenz ``shared`` (Gewichte doppelt zu zaehlen waere falsch).  Was sich NICHT an D's
      Bytes binden laesst, behaelt P (weg2/union_arena_bind.py, Kopf: "What it cannot prove, it keeps"); die Menge ist ohne Boot nicht zu belegen
      -> Referenzposten ``diff`` "nicht gerechnet".
    * ``--dual-p-overhead-mib`` (launcher.py:22713-22716, Standard 1500): "what P holds on a card outside its --rank-gpu-memory-mib budget (CUDA
      context, graphs, activations)" -> Festposten AUSSERHALB des Budgets; die Aktivierung steckt darin (Referenz ``in_festposten``, kein
      zweiter Abzug).  D wird aus "P's PLAN" bemessen: Budget + dieser Wert (launcher.py:14976 ``dual_share_planned_dc``, :27291).
    * ``--dual-unified-kv on`` (weg2/card_kv_ledger.py Kopf: "the REST of the card is ONE KV budget K_c, with no split", P und D fragen ein Ledger;
      weg2/dual_p_kv_stage.py Kopf: P-Pool "born trimmed at 0 tokens", Bytes aus dem Karten-Ledger): P's KV liegt im Karten-KV-Pool, nicht im
      P-Budget -> Referenz ``shared``.  Ohne ``on`` ist das nicht belegt -> KV zaehlt gegen das P-Budget.
    * Mamba/GDN-Zustand (card_kv_ledger.py Kopf: "The boot fixes weights, contexts, mamba, graph pools and activation transients") und der Draft
      (UNION_ROLES = ``main``: der Draft ist nicht geteilt, launcher.py:14781-14784) bleiben P-eigen und zaehlen gegen das P-Budget.
    """
    terms = stage["terms"]

    def ref(key: str, kind: str, note: str) -> None:
        t = terms.get(key)
        if t is not None and t.get("v") is not None:
            t["ref"] = kind
            t["note"] = note + ((" -- " + t["note"]) if t.get("note") else "")

    ref("weights", "shared", "Dual share: P computes on D's TP shards and binds them to D's bytes via the union image (launcher.py:22707-22712); does not count against the P budget")
    ref("experts", "shared", "Dual share: resident experts live in the union image of D (UNION_ROLES=main, launcher.py:14784); does not count against the P budget")
    if str(args.get("--dual-unified-kv", "off")).lower() == "on":
        ref("kv", "shared", "--dual-unified-kv on: one KV pool per card for P and D (weg2/card_kv_ledger.py), P pool virtual; does not count against the P budget")
    ref("activation", "in_festposten", "Activation is part of --dual-p-overhead-mib (launcher.py:22713-22716: CUDA context, graphs, activations)")
    ov_raw = args.get("--dual-p-overhead-mib")
    try:
        ov = float(ov_raw) if ov_raw not in (None, "") else float(DUAL_P_OVERHEAD_DEFAULT_MIB)
    except (TypeError, ValueError):
        ov = float(DUAL_P_OVERHEAD_DEFAULT_MIB)
        ov_raw = None
    given = ov_raw not in (None, "")
    terms["fixed"] = {"v": ov, "src": SRC_INPUT if given else SRC_DEFAULT, "outside_budget": True,
                      "note": "--dual-p-overhead-mib: what P holds per card OUTSIDE its --rank-gpu-memory-mib (context, graphs, activation; launcher.py:22713-22716)"
                              + ("" if given else "; launcher default %d MiB (not set in the profile)" % DUAL_P_OVERHEAD_DEFAULT_MIB)}
    stage["ref_extra"] = [{"name": "diff", "label": "Diff of the P weights", "mib": None, "ref": "shared", "herkunft": SRC_NONE, "gerechnet": False,
                           "detail": "What cannot be bound to D's bytes, P keeps itself (union_arena_bind.py head: \"What it cannot prove, it keeps\"); "
                                     "the amount cannot be verified without a boot"}]
    if seen is not None:
        seen.append({"was": "dual_p_overhead_mib", "wert": "%g" % ov, "herkunft": ("Profile row --dual-p-overhead-mib" if given else
                     "Launcher default (--dual-p-overhead-mib 1500, launcher.py:22713)")})
        seen.append({"was": "dual_share", "wert": "an", "herkunft": "Profile row --dual-share: weights/experts (and KV with --dual-unified-kv on) do not count against the P budget"})


def phase_bars(hw: Mapping[str, Any], model: Mapping[str, Any], args: Mapping[str, str], phase_args: Optional[Mapping[str, Any]] = None,
               phase_env: Optional[Mapping[str, Any]] = None, form: Optional[str] = None, tokens: Sequence[str] = (),
               overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Balken je Karte und Phase im Vertrag ``flliper.balken/1`` aus dem Serverprofil (Flag -> Wert, Gruppenzeilen getrennt).

    Phase ``P``: Pipeline-Stufen (``_stage_terms``).  Phase ``D``: TP-Raenge (``d_stage_terms``).  Dual: beide zugleich auf denselben Karten --
    eine Summe wird NICHT gebildet (``--dual-share``: P rechnet auf den Shards von D, Gewichte doppelt zu zaehlen waere falsch; die
    Dual-Passung ist Aufgabe des Planers AP-E).  Eine Phase, die sich nicht rechnen laesst, steht mit ``ok: False`` und Grund da."""
    n = len(_cards(hw))
    form = detect_form(args, tokens, n, form)
    over = dict(overrides or {})
    penv = dict(phase_env or {})
    out_phases: Dict[str, Any] = {}
    hints: List[str] = []
    all_args: Dict[str, str] = dict(args or {})
    for ph in ("P", "D"):
        all_args.update(dict((phase_args or {}).get(ph) or {}))
    draft = _draft_info(model, all_args)
    carries_p = str(all_args.get("--draft-kv-on-p", "on")).lower() != "off"
    approx: Optional[Dict[str, Any]] = None
    dual_share = form == "dual" and ("--dual-share" in all_args or "--dual-share" in tokens)

    def run_p(name: str, scope: str) -> Dict[str, Any]:
        nonlocal approx
        a = _merged(args, phase_args, scope) if scope != "-" else dict(args)
        carries = carries_p or form == "single"
        settings, seen = p_phase_settings(a, dict(penv.get("P") or {}), model, n, draft, carries_draft=carries)
        settings.update({k: v for k, v in over.items() if k != "what"})
        if "stage_layers" not in settings:
            raise CouplingError("--pp-stage-ratio is missing in the profile: the layer cut of the P phase is unknown")
        t = _stage_terms(hw, model, settings)
        stages = []
        for i, st in enumerate(t["stages"]):
            st2 = dict(st)
            st2["_rate_known"] = t["_ctx"]["rate"] > 0
            st2["terms"] = _p_terms_for_contract(st2, draft, carries, i == len(t["stages"]) - 1, settings)
            if dual_share and name == "P":
                _dual_share_p_terms(st2, all_args, seen if i == 0 else None)
            stages.append(st2)
        try:
            approx = approx_payload(hw, model, settings)
        except (CouplingError, KeyError, ValueError, TypeError):
            approx = None
        if name == "P" and draft["kind"] != "none" and not carries:
            seen.append({"was": "draft", "wert": "aus", "herkunft": "Profile row --draft-kv-on-p off: P carries no MTP head"})
        label = "P phase (prefill, pipeline stages)" if name == "P" else "Single card (one phase)"
        # Kontext-Boden = Token, die nach den Gewichten ins Budget passen: unter --dual-share zaehlen die Gewichte nicht gegen das P-Budget, die Zahl
        # waere eine andere Frage -> nicht gerechnet (None) statt einer falschen 0
        return {"ok": True, "label": label, "bars": [contract_bar(s, name) for s in stages], "inputs": seen,
                "context_floor_tokens": None if (dual_share and name == "P") else t["context_floor_tokens"], "warnings": t["warnings"]}

    def run_d() -> Dict[str, Any]:
        a = _merged(args, phase_args, "D")
        # --dual-share: der Launcher bemisst D aus P's PLAN (launcher.py:27289-27301); die Zeilen der Gruppe P liefern ihn, D erbt sie nicht
        plan = dual_p_plan(_merged(args, phase_args, "P"), n) if dual_share else None
        cfg = d_phase_config(a, dict(penv.get("D") or {}), model, n, draft, plan, launcher_args=args)
        for k in ("context_tokens", "mamba_slots", "ssm_dtype", "kv_dtype", "corridor_mib", "activation_mib"):
            if k in over:
                cfg[k] = over[k]
        t = d_stage_terms(hw, model, cfg)
        return {"ok": True, "label": "D phase (decode, TP ranks)", "bars": [contract_bar(s, "D") for s in t["stages"]],
                "inputs": cfg["seen"] + t.get("inputs", []), "hints": t["hints"], "warnings": t["warnings"]}

    plan = {"single": [("alle", lambda: run_p("alle", "-"))], "d_only": [("D", run_d)],
            "flip": [("P", lambda: run_p("P", "P")), ("D", run_d)], "dual": [("P", lambda: run_p("P", "P")), ("D", run_d)]}[form]
    for name, fn in plan:
        try:
            out_phases[name] = fn()
        except (CouplingError, KeyError, ValueError, TypeError) as exc:
            out_phases[name] = {"ok": False, "label": name, "error": "%s: %s" % (type(exc).__name__, exc), "bars": [], "inputs": []}
    for ph in out_phases.values():
        for b in ph["bars"]:
            if b["over_text"]:
                hints.append(b["over_text"])
        hints.extend(ph.get("hints") or [])
    if form == "dual":
        hints.append("Dual: P and D run at the same time on the same cards. The sum of both bars is not calculated (--dual-share: P computes on the shards of D, the dual fit is a planner calculation of AP-E, not hw_fit).")
        if dual_share:
            hints.append("Dual share: P's weights and experts live in the union image of D and do not count against the P budget (reference row below the bar); the P-own items count against the P budget, the fixed item --dual-p-overhead-mib lies outside the budget.")
            hints.append("Dual share, D side: D is the union owner and holds weights and experts in its budget; the launcher sizes D from P's plan (fixed item = P budget + --dual-p-overhead-mib per card, outside D's budget, launcher.py:27289-27301). Under --dual-unified-kv on D's KV pool grows virtually from the card pool (--dual-d-kv-max-tokens); the bar shows the context goal, not the pool.")
    return {"schema": BALKEN_SCHEMA, "form": form, "n_cards": n, "phases": out_phases, "hints": hints, "approx": approx,
            "draft": {k: draft[k] for k in ("kind", "placement", "reason")}}


# ---------------------------------------------------------------------------
# Brueckenaufruf: eine JSON-Anfrage -> eine JSON-Antwort (Dashboard-Kindprozess, Auftrag 1431/1432)
# ---------------------------------------------------------------------------


def run(req: Mapping[str, Any]) -> Dict[str, Any]:
    """``{"what": "compute"|"move"|"chunk"|"context"|"bars", "hardware": {..}, "model": {..}, "settings": {..}, ...}`` -> Ergebnis.

    ``settings`` darf stattdessen ``server_args`` (Flag -> Wert, ``profile_json.args_dict``) tragen; ``move`` braucht ``src``/``dst``/``n``,
    ``chunk`` optional ``new_chunk_tokens``.  Eine unrechenbare Eingabe kommt als ``{"ok": False, "error": ...}`` zurueck, nie als Absturz."""
    try:
        hw, model = req["hardware"], req["model"]
        settings = dict(req.get("settings") or {})
        if req.get("server_args"):
            if req.get("what") != "phase_bars":      # phase_bars liest die Zeilen selbst (je Gruppe getrennt)
                settings = dict(settings_from_server(req["server_args"], model), **settings)
        what = req.get("what", "compute")
        if what == "compute":
            res = compute(hw, model, settings)
        elif what == "move":
            res = c1_move_layers(hw, model, settings, src=int(req["src"]), dst=int(req["dst"]), n=int(req.get("n", 1)))
        elif what == "chunk":
            res = c3_chunk(hw, model, settings, new_chunk_tokens=req.get("new_chunk_tokens"))
        elif what == "context":
            res = c4_context_target(hw, model, settings)
        elif what == "bars":
            res = dict(bars_for(hw, model, settings, phases=req.get("phases")), approx=approx_payload(hw, model, settings))
        elif what == "phase_bars":
            # AP-H2: Vertrag flliper.balken/1.  ``settings`` (Eingaben des Nutzers) gelten als Uebersteuerung; die Zeilen des Profils kommen aus
            # ``server_args`` (Flag -> Wert) sowie ``phase_args`` / ``phase_env`` (je Gruppe P/D: --extra-p/-d und --env-p/-d)
            res = phase_bars(hw, model, dict(req.get("server_args") or {}), req.get("phase_args"), req.get("phase_env"), req.get("form"),
                             tuple(req.get("tokens") or ()), overrides=settings)
        else:
            raise CouplingError("unknown request %r (compute|move|chunk|context)" % what)
        return {"ok": True, "result": res}
    except (CouplingError, KeyError, ValueError, TypeError) as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m sglang.srt.planner.profile_couplings`` : JSON-Anfrage von stdin, JSON-Antwort nach stdout."""
    res = run(json.load(sys.stdin))
    json.dump(res, sys.stdout, default=str)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
