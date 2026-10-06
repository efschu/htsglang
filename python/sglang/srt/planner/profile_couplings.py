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
C5  Draft/Spekulation <-> VRAM P/D                         (nur Kante, noch ohne Rechnung)
C6  HiCache/L2/L3 <-> Host-RAM                             (nur Kante)
C7  Kartenwahl/Reihenfolge <-> alles                       (nur Kante)

EHRLICHKEIT (Memory HOCHRECHNUNG != MESSUNG, INDIKATOR-GESETZ).  Jeder Term traegt seine Quelle (``src``): aus dem Profil
uebernommen (``config`` | ``Index`` | ``geschaetzt`` | ``gemessen`` | ``NVML``) oder ``Eingabe`` (vom Nutzer gesetzt) oder
``Standard`` (Annahme dieses Moduls, benannt).  Posten, die nur am Metall zu messen sind (CUDA-Kontext, Graphen, Allokator-Reste,
Seam-Staging), stehen in ``fixed_overhead_mib`` und sind OHNE Eingabe NULL -- dann ist ``free_mib`` eine OBERGRENZE und das Ergebnis
sagt es (``warnings``).  Die Stufenzeit ist eine Roofline-Naeherung (Decode, Batch 1, Gewichtsbandbreite ``mem_gbs.gemv``); sie
ersetzt nicht ``pp_cut.stage_costs`` (der braucht eine am Metall kalibrierte Census) und rankt keine Schnitte (#1019).

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
SRC_INPUT = "Eingabe"
SRC_DEFAULT = "Standard"
SRC_DERIVED = "gerechnet"


class CouplingError(ValueError):
    """Eine Eingabe, die sich nicht rechnen laesst (benannt, nie still repariert)."""


# ---------------------------------------------------------------------------
# C5-C7: benannte Kanten, noch ohne Rechnung
# ---------------------------------------------------------------------------

EDGES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "C5",
        "name": "Draft/Spekulation <-> VRAM P/D",
        "touches": ["--spec-form", "--draft-model-path", "P_DRAFT_RESIDENT_BUDGET_MIB"],
        "computes_with": ["expert_residency.draft_vocab_mib", "Records"],
        "computed": False,
        "text": "Der Draft belegt VRAM in jeder Gruppe, die ihn traegt; ohne Draft bleibt mehr fuer KV, aber der Spec-Gewinn entfaellt.",
        "now": "nicht berechnet: nur ``draft`` (Gewicht der MTP-Layer auf der letzten Stufe, eine KV-Zeile je Stufe) geht in C1-C4 ein",
    },
    {
        "id": "C6",
        "name": "HiCache/L2/L3 <-> Host-RAM",
        "touches": ["--store-max-gb", "SGLANG_HICACHE_*", "PROFILE_SHM_MIN_GIB", "PROFILE_MEMAVAIL_MIN_GIB"],
        "computes_with": ["weg2/host_ledger.py", "Hardwareprofil host"],
        "computed": False,
        "text": "Die Cache-Stufen leben im Host-RAM, nicht im VRAM: Schwellen aus Modellbytes und Kartenzahl (HWGEN P9 / K10).",
        "now": "nicht berechnet: Balken Host-RAM statt VRAM folgt",
    },
    {
        "id": "C7",
        "name": "Kartenwahl/Reihenfolge <-> alles",
        "touches": ["Kartenliste", "Host-Ordinal (Form A)", "alle Vektorwerte"],
        "computes_with": ["card_identity.order_cards", "topology.plan_topology"],
        "computed": False,
        "text": "Eine Karte hinzufuegen oder entfernen aendert die Laenge aller Vektorwerte (Layer, Budgets, Anteile).",
        "now": "nicht berechnet: ``compute`` verlangt gleich viele Stufen wie Karten und verweigert sonst (``vector_length``)",
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
        raise CouplingError("vector_length: %s hat %d Werte, es gibt %d Karten/Stufen" % (name, len(xs), n))
    return xs


def _term(v: float, src: str, note: str = "") -> Dict[str, Any]:
    out = {"v": round(float(v), 3), "src": src}
    if note:
        out["note"] = note
    return out


def _cards(hw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cards = list(hw.get("cards") or [])
    if not cards:
        raise CouplingError("Hardwareprofil ohne Karten")
    return sorted(cards, key=lambda c: int(c.get("ord", 0)))


def _label(c: Mapping[str, Any]) -> str:
    return "Karte %s (%s)" % (c.get("ord", "?"), str(c.get("name", "?")).replace("NVIDIA GeForce ", ""))


def _families(model: Mapping[str, Any]) -> List[str]:
    return [str(f) for f in _val(model["arch"]["layer_families"])]


def _cell_bytes(model: Mapping[str, Any], kv_dtype: Optional[str]) -> Tuple[float, str]:
    """Bytes je Token und Attention-Layer (Nutzlast + Skalenpuffer) der gewaehlten KV-Variante."""
    kv = model["kv"]
    if kv_dtype:
        var = (kv.get("variants") or {}).get(kv_dtype)
        if var is None:
            raise CouplingError("kv_dtype %r ist keine Variante des Modellprofils (%s)" % (kv_dtype, ", ".join((kv.get("variants") or {}))))
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
        raise CouplingError("stage_layers fehlt (--pp-stage-ratio)")
    if len(counts) != n:
        raise CouplingError("vector_length: stage_layers hat %d Stufen, das Hardwareprofil %d Karten" % (len(counts), n))
    fams = _families(model)
    if sum(counts) != len(fams):
        raise CouplingError("stage_layers summiert zu %d, das Modell hat %d Layer" % (sum(counts), len(fams)))
    if any(c < 0 for c in counts):
        raise CouplingError("stage_layers enthaelt negative Werte")

    warnings: List[str] = []
    corridor = float(settings.get("corridor_mib", 1024.0))
    ctx = int(settings.get("context_tokens", 262144))
    chunk = int(settings.get("chunk_tokens", 2048))
    slots = int(settings.get("mamba_slots", 1))
    draft = bool(settings.get("draft", False))
    replicated = tuple(settings.get("replicated") or ())

    totals = [float(_val(c.get("vram_total_mib"), 0.0)) for c in cards]
    if any(t <= 0 for t in totals):
        raise CouplingError("Hardwareprofil: vram_total_mib fehlt auf einer Karte")
    budgets = _per_card(settings.get("budget_mib"), n, "budget_mib", 0.0) if settings.get("budget_mib") is not None else [t - corridor for t in totals]
    budget_src = SRC_INPUT if settings.get("budget_mib") is not None else SRC_DERIVED
    fixed = _per_card(settings.get("fixed_overhead_mib"), n, "fixed_overhead_mib", 0.0)
    fixed_src = SRC_INPUT if settings.get("fixed_overhead_mib") is not None else SRC_DEFAULT
    if settings.get("fixed_overhead_mib") is None:
        warnings.append("fixed_overhead_mib nicht gesetzt (CUDA-Kontext, Graphen, Allokator-Reste, Seam-Staging sind nur am Metall zu messen): "
                        "free_mib ist eine OBERGRENZE")

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
    draft_note = "MTP-Layer auf der letzten Stufe" if draft else ""
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
            raise CouplingError("vector_length: attn_layers hat %d Werte, es gibt %d Stufen" % (len(attn), n))
        if sum(attn) != sum(1 for f in fams if f == "attn"):
            raise CouplingError("attn_layers summiert zu %d, das Modell hat %d Attention-Layer" % (sum(attn), sum(1 for f in fams if f == "attn")))
    else:
        attn = [sum(1 for f in fams[a:b] if f == "attn") for a, b in bounds]
    lin = [c - a for c, a in zip(counts, attn)] if pinned is not None else [sum(1 for f in fams[a:b] if f != "attn") for a, b in bounds]
    draft_attn = [1 if draft else 0] * n

    cell, cell_src = _cell_bytes(model, settings.get("kv_dtype"))
    cell_mib = cell / MIB
    geo = _kv_geometry(model, settings.get("kv_dtype"), cell)
    cross = _pp.kv_cell_bytes_per_attention_layer(**geo)
    if abs(cross - cell) > 0.5:
        warnings.append("KV-Zelle: Modellprofil %.0f B, pp_cut-Geometrie %.0f B je Attention-Layer (Modellprofil gilt)" % (cell, cross))
        geo = dict(geo, kv_dtype_bytes=geo["kv_dtype_bytes"] * cell / cross)
    state_node = model["state"].get("per_linear_layer_per_slot_mib")
    state_per = float(_val(state_node, 0.0))
    ssm = settings.get("ssm_dtype")
    if ssm:
        variants = model["state"].get("variants_mib") or {}
        if ssm not in variants:
            raise CouplingError("ssm_dtype %r ist keine Variante des Modellprofils (%s)" % (ssm, ", ".join(variants)))
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
            "budget_mib": _term(budgets[i], budget_src, "Kartengroesse - Korridor %.0f MiB" % corridor if budget_src == SRC_DERIVED else ""),
            "terms": {
                "weights": _term(dense_mib, _src(w["total_bytes"]) if "total_bytes" in w else SRC_DEFAULT,
                                 "dichte Gewichte der Layer + Einbettung/lm_head der Rolle"),
                "experts": _term(experts_mib, _src(w["total_bytes"]) if "total_bytes" in w else SRC_DEFAULT,
                                 "residente Expertenzeilen %.0f %% (Pufferregel)" % (100 * buf_fracs[i]) if moe else "keine Experten"),
                "draft": _term(draft_mib, draft_src if draft else SRC_DEFAULT, draft_note if draft_mib else "kein Draft"),
                "kv": _term(kv_mib, cell_src, "%d Token x %d Attention-Layer%s x %.0f B" % (ctx, attn[i], " + Draft" if draft_attn[i] else "", cell)),
                "state": _term(state_mib, _src(state_node), "%d Linear-Layer x %.4f MiB x %d Slot(s)" % (lin[i], state_per, slots)),
                "activation": _term(act_mib, SRC_INPUT if act_vec is not None else rate_src,
                                    "gemessene Spitze (Eingabe)" if act_vec is not None else "%d Zeilen x %.4f MiB" % (chunk, rate)),
                "fixed": _term(fixed[i], fixed_src, "" if fixed_src == SRC_INPUT else "nicht gemessen"),
            },
            "draft_note": draft_note if i == n - 1 and draft else "",
            "needs_mib": round(needs, 3),
            "free_mib": round(free, 3),
            "overflow_mib": round(max(0.0, -free), 3),
            "kv_capacity_tokens": None if kv_cap is None else int(kv_cap),
            "decode_ms": None if decode_ms is None else round(decode_ms, 3),
            "decode_src": "Naeherung (Roofline, Batch 1, mem_gbs.gemv)" if decode_ms is not None else "nicht gemessen (mem_gbs.gemv fehlt)",
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
        raise CouplingError("Layer wechseln nur zwischen NACHBARSTUFEN (src=%s dst=%s)" % (src, dst))
    if not (0 <= src < len(counts) and 0 <= dst < len(counts)):
        raise CouplingError("Stufe ausserhalb 0..%d" % (len(counts) - 1))
    if n < 1 or counts[src] < n:
        raise CouplingError("Stufe %d hat nur %d Layer, %d verlangt" % (src, counts[src], n))
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
        "Du verschiebst %d Layer von %s auf %s (Attention-Layer %d -> %d bzw. %d -> %d). %s: %+.0f MiB frei%s. %s: %+.0f MiB frei%s." % (
            n, bs["label"], bd["label"], bs["attn_layers"], as_["attn_layers"], bd["attn_layers"], ad["attn_layers"],
            bs["label"], as_["free_mib"] - bs["free_mib"], _tok_delta(bs, as_),
            bd["label"], ad["free_mib"] - bd["free_mib"], _tok_delta(bd, ad)))
    if before["makespan_ms"] is not None and after["makespan_ms"] is not None:
        hints.append("Rundenzeit (Naeherung, Decode Batch 1): Takt %.2f -> %.2f ms (die langsamste Stufe bestimmt ihn)." % (
            before["makespan_ms"], after["makespan_ms"]))
    for st in after["stages"]:
        if st["overflow_mib"] > 0:
            hints.append("%s: %.0f MiB UEBER dem Budget. Der Planer lehnt ab; mit Force startet es trotzdem, zu erwarten ist ein OOM beim Laden." % (
                st["label"], st["overflow_mib"]))
    fb, fa = before["context_floor_tokens"], after["context_floor_tokens"]
    if fb is not None and fa is not None:
        hints.append("Kontext-Boden (Minimum ueber die Stufen): %d -> %d Token." % (fb, fa))
    return {
        "id": "C1", "move": {"src": src, "dst": dst, "n": n}, "before_layers": counts, "after_layers": after_counts, "rows": rows,
        "context_floor_tokens": [fb, fa], "makespan_ms": [before["makespan_ms"], after["makespan_ms"]], "hints": hints,
        "warnings": before["warnings"],
    }


def _tok_delta(b: Mapping[str, Any], a: Mapping[str, Any]) -> str:
    x, y = b["kv_capacity_tokens"], a["kv_capacity_tokens"]
    if x is None or y is None:
        return ""
    return " (KV-Kapazitaet %+d Token)" % (y - x)


# ---------------------------------------------------------------------------
# C2: MoE-Experten <-> KV
# ---------------------------------------------------------------------------


def c2_expert_residency(hw: Mapping[str, Any], model: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Je Stufe die GROESSTE Experten-Fraction, die nach KV-Preis (Kontextziel), Zustand, Aktivierung und Festposten noch ins Budget passt
    (``pp_cut.solve_expert_fraction_per_stage``), gegen die gesetzte Fraction.  Ohne Experten (dichtes Modell): keine Kopplung."""
    E = int(_val(model.get("experts", {}).get("n"), 0) or 0)
    if E <= 0:
        return {"id": "C2", "applicable": False, "text": "Dichtes Modell: keine Experten, keine Kopplung."}
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
            hints.append("%s: nicht einmal die dichten Gewichte plus KV-Preis passen ins Budget (Fraction 0)." % s["label"])
        elif fr_set[i] > fr_max[i] + 1e-9:
            hints.append("%s: Fraction %.2f ist zu gross; hoechstens %.2f (%d von %d Zeilen je Layer auf der Karte) bei %d Token Kontext." % (
                s["label"], fr_set[i], fr_max[i], rows_max, E, settings.get("context_tokens", 262144)))
        else:
            hints.append("%s: bis Fraction %.2f frei (gesetzt %.2f). Mehr Experten resident = weniger Host-Zugriffe im Decode, aber weniger KV." % (
                s["label"], fr_max[i], fr_set[i]))
    return {"id": "C2", "applicable": True, "rows": rows, "hints": hints, "warnings": t["warnings"],
            "note": "Obergrenze: Draft-KV-Produzent und Seam-Staging nur ueber fixed_overhead_mib"}


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
        hints.append("Extend-Rate unbekannt (0): die Aktivierung ist nicht gerechnet.")
    elif new_chunk_tokens is not None and int(new_chunk_tokens) != chunk:
        d = after["stages"][0]["terms"]["activation"]["v"] - before["stages"][0]["terms"]["activation"]["v"]
        hints.append("Chunk %d -> %d: %+.0f MiB Aktivierung auf jeder Karte, entsprechend weniger KV-Kapazitaet; ein groesserer Chunk gibt mehr "
                     "Prefill-Durchsatz, eine hoehere Spitze und weniger Kontext." % (chunk, int(new_chunk_tokens), d))
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
            hints.append("%s: %d Token brauchen %.0f MiB KV, vorhanden %.0f MiB: es fehlen %.0f MiB (Kapazitaet %s Token)." % (
                s["label"], ctx, need, have, miss, s["kv_capacity_tokens"]))
    if not hints:
        hints.append("Das Kontextziel von %d Token passt auf jeder Stufe." % ctx)
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
            hints.append("%s: +%.0f MiB ueber dem Budget. Der Planer lehnt ab (VRAM-Riegel); mit Force startet es trotzdem, zu erwarten ist OOM "
                         "beim Laden oder beim Graphenaufbau." % (st["label"], st["overflow_mib"]))
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
            "mem_gbs": {"gemv": {"v": gemv, "src": "gemessen" if gemv else "nicht gemessen"}},
        })
    return {"schema": "flliper.hardware/1", "id": "synthetic", "cards": out}


# ---------------------------------------------------------------------------
# S4b (Auftrag 1432): Balken je Karte und Phase, Browser-Naeherung
# ---------------------------------------------------------------------------

#: Segmente eines Balkens in Zeichenreihenfolge: Schluessel (Term), Beschriftung, Erklaerung fuer den Tooltip
BAR_SEGMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("weights", "Gewichte", "dichte Gewichte der Layer dieser Stufe plus Einbettung (erste Stufe) bzw. lm_head (letzte Stufe)"),
    ("experts", "Experten (resident)", "Expertenzeilen auf der Karte nach Pufferregel min(R + Scratch, E) je Layer"),
    ("draft", "Draft/MTP", "Gewicht der MTP-Layer auf der letzten Stufe"),
    ("kv", "KV", "Kontextziel x Attention-Layer der Stufe (+ eine Draft-Zeile) x KV-Zelle"),
    ("state", "Mamba/GDN-Zustand", "Linear-Layer der Stufe x Zustand je Layer und Slot x Slots"),
    ("activation", "Aktivierung", "Chunk-Zeilen x Extend-Rate (Spitze beim Prefill)"),
    ("fixed", "Festposten", "CUDA-Kontext, Graphen, Allokator-Reste, Seam-Staging: nur am Metall zu messen (ohne Eingabe 0)"),
)

_ORIGIN = {SRC_INPUT: "Eingabe (Nutzer/Profil)", SRC_DEFAULT: "Annahme dieser Rechnung", SRC_DERIVED: "gerechnet"}


def _origin(src: str) -> str:
    return _ORIGIN.get(src, "Modellprofil/Hardwareprofil (%s)" % src)


def _clip_to_budget(segs: List[Dict[str, Any]], budget: float) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    """Segmente bis ``budget`` behalten; was darueber liegt, kommt (von hinten abgeschnitten) in die Ueberlaufliste.  Die Aufteilung
    des Ueberlaufs auf Posten ist Darstellung, die Summe ist exakt."""
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
    """Ein Karten-Balken aus einer Stufe von ``c1_layer_split``: Posten, Rest im Budget, Korridor, **Ueberlauf als eigenes rotes Segment**.

    Zeichenfolge: Posten (bis zum Budget) | Rest im Budget | Korridor/Reserve (Kartengroesse - Budget).  Ueberschreitet die Summe das
    Budget, steht der Teil darueber als Segment ``overflow`` (rot) VOR der Reserve; der Balken waechst dann ueber die Kartenkante, wenn
    der Ueberlauf die Reserve ueberschreitet.  Kein stilles Beschneiden: ``overflow_mib`` und die betroffenen Posten stehen im Segment."""
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
        out.append({"key": "overflow", "label": "Ueberlauf", "mib": overflow, "src": SRC_DERIVED, "origin": "gerechnet",
                    "what": "Posten ueber dem Budget (%.0f MiB): %s" % (budget, ", ".join("%s %.0f MiB" % (c["label"], c["mib"]) for c in cut)),
                    "cut": cut})
    elif free > 0:
        out.append({"key": "free_in_budget", "label": "Rest im Budget", "mib": round(free, 3), "src": SRC_DERIVED, "origin": "gerechnet",
                    "what": "Budget - Posten (Obergrenze, solange Festposten nicht gemessen sind)"})
    reserve = max(0.0, total - budget)
    if reserve > 0:
        out.append({"key": "corridor", "label": "Korridor/Reserve", "mib": round(reserve, 3), "src": stage["budget_mib"]["src"],
                    "origin": _origin(stage["budget_mib"]["src"]), "what": "Kartengroesse - Budget: bleibt frei (Reserve-Semantik)"})
    over_text = ""
    if overflow > 0:
        over_text = ("%s: +%.0f MiB ueber dem Budget. Der Planer lehnt ab; mit Force startet es trotzdem, zu erwarten ist OOM beim Laden "
                     "oder beim Graphenaufbau." % (stage["label"], overflow))
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
                hints.append(("%s-Phase: " % name if name != "alle" else "") + b["over_text"])
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
        raise CouplingError("approx: stage_layers passt nicht zum Modell (%s)" % counts)
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
SRC_PROFILE = "Profilzeile"
SRC_APPROX = "Naeherung"
SRC_NONE = "nicht gerechnet"
FORMS = ("single", "d_only", "flip", "dual")

BALKEN_SEGMENTS: Tuple[Tuple[str, str, str], ...] = BAR_SEGMENTS + (
    ("reserve", "Reserve", "Kartengroesse - Budget: bleibt frei (Reserve-Semantik); wird von Posten ueber dem Budget aufgezehrt"),
    ("free", "Frei", "Budget - Posten (Obergrenze, solange Posten nicht gerechnet sind)"),
)
#: Erklaerung je Posten der D-Phase (TP-Raenge): die Texte von BAR_SEGMENTS beschreiben die P-Stufen (Layer-Schnitt, letzte Stufe)
D_WHAT = {
    "weights": "dichte Gewichte, Einbettung und lm_head dieses Ranges (TP-Anteil)",
    "experts": "Expertenzeilen dieses Ranges nach Pufferregel min(R + Scratch, eigene Experten) je Layer",
    "draft": "Gewicht des Drafts auf diesem Rang",
    "kv": "KV-Anteil dieses Ranges fuer das Kontextziel",
    "state": "Mamba/GDN-Zustand-Anteil dieses Ranges: Linear-Layer x Zustand je Layer und Slot x Slots x TP-Anteil",
    "activation": "Decode-Aktivierung dieses Ranges",
    "fixed": "CUDA-Kontext der schlafenden Phase und VRAM ausserhalb des Torch-Allokators (D-Seite)",
}
_ORIGIN.update({SRC_PROFILE: "Profilzeile", SRC_APPROX: "Naeherung (nicht der Loeser)", SRC_NONE: "nicht gerechnet"})

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
                    reason="DFlash2-Draft: Platzierung und Gewicht je Gruppe sind nicht gerechnet (Orakel/AP-D)")
        return info
    if not (flagged or mtp > 0):
        info["reason"] = "kein Draft: kein --speculative-*/--dflash-*-Flag im Profil und kein MTP-Kopf im Modell"
        return info
    info["kind"] = "nextn"
    if ext and "bytes_without_lm_head" in ext:
        info["p_mib"] = float(_val(ext["bytes_without_lm_head"])) / MIB
        info["d_mib"] = float(_val(ext["bytes_without_embed_lm_head"])) / MIB if "bytes_without_embed_lm_head" in ext else None
        info["src"] = _src(ext["bytes_without_lm_head"])
        info["p_note"] = "Draft-Verzeichnis ohne lm_head (P teilt ihn mit dem Ziel), ohne Laufzeitpuffer"
        info["d_note"] = "Draft-Verzeichnis ohne Einbettung und lm_head (D teilt sie mit dem Ziel), ohne Laufzeitpuffer"
        if info["d_mib"] is None:
            info["reason"] = "Draft-Verzeichnis ohne Aufteilung Einbettung/lm_head im Profil"
        info["attn_layers"] = int(_val((ext.get("kv") or {}).get("attn_layers"), 1) or 1)
    elif mtp > 0:
        info["p_mib"] = info["d_mib"] = mtp
        info["src"] = _src(w.get("mtp_bytes"))
        info["p_note"] = info["d_note"] = "MTP-Kopf des Ziel-Checkpoints (mtp.*)"
    else:
        info.update(p_mib=None, d_mib=None, reason="Draft-Verzeichnis ist im Profil genannt, aber nicht profiliert (kein Modellprofil des Drafts)")
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
            note(key, args.get(flag, ""), "Profilzeile " + flag)
    if n == 1 and "stage_layers" not in s:
        s["stage_layers"] = [len(_families(model))]
        note("stage_layers", s["stage_layers"][0], "Einzelkarte: alle Layer auf der einen Karte")
    if "moe_resident_fraction" not in s:
        fr = _fl(args.get("--pp-cut-expert-device-fraction")) or _fl(env.get("SGLANG_MOE_RESIDENT_EXPERT_FRACTION"))
        if fr:
            s["moe_resident_fraction"] = fr if len(fr) > 1 else fr[0]
            note("moe_resident_fraction", ",".join("%g" % x for x in fr),
                 "Profilzeile --pp-cut-expert-device-fraction" if args.get("--pp-cut-expert-device-fraction")
                 else "Umgebung SGLANG_MOE_RESIDENT_EXPERT_FRACTION (--env-p)")
    else:
        note("moe_resident_fraction", args.get("--rank-moe-resident-fraction", ""), "Profilzeile --rank-moe-resident-fraction")
    sc = _fl(args.get("--pp-cut-expert-lru-rows")) or _fl(env.get("SGLANG_MOE_SCRATCH_SLOTS"))
    if sc:
        s["scratch_rows"] = [int(x) for x in sc] if len(sc) > 1 else int(sc[0])
        note("scratch_rows", ",".join("%d" % x for x in sc),
             "Profilzeile --pp-cut-expert-lru-rows" if args.get("--pp-cut-expert-lru-rows") else "Umgebung SGLANG_MOE_SCRATCH_SLOTS (--env-p)")
    slots = args.get("--max-mamba-cache-size")
    if slots and str(slots).isdigit():
        s["mamba_slots"] = int(slots)
        note("mamba_slots", slots, "Profilzeile --max-mamba-cache-size")
    ssm = args.get("--mamba-ssm-dtype")
    if ssm and ssm in ((model.get("state") or {}).get("variants_mib") or {}):
        s["ssm_dtype"] = ssm
        note("ssm_dtype", ssm, "Profilzeile --mamba-ssm-dtype")
    if "budget_mib" not in s:
        note("budget_mib", "Kartengroesse - 1024 MiB", "Annahme dieser Rechnung (kein --rank-gpu-memory-mib im Profil)")
    for key, default, why in (("chunk_tokens", 2048, "kein --chunked-prefill-size/--p-chunk-max im Profil"),
                              ("context_tokens", 262144, "kein --max-kv-per-request/--context-length im Profil"),
                              ("mamba_slots", 1, "kein --max-mamba-cache-size im Profil")):
        if key not in s:
            note(key, default, "Annahme dieser Rechnung (%s)" % why)
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
        t["fixed"] = {"v": None, "src": SRC_NONE, "note": "CUDA-Kontext, Graphen, Allokator-Reste, Seam-Staging: nur am Metall zu messen"}
    if settings.get("activation_mib") is None and not stage.get("_rate_known", True):
        t["activation"] = {"v": None, "src": SRC_NONE, "note": "Extend-Rate des Modells unbekannt"}
    if is_last and carries_draft and draft["kind"] != "none" and not settings.get("draft"):
        t["draft"] = {"v": None, "src": SRC_NONE, "note": draft["reason"] or "Draftgewicht nicht belegbar"}
    return t


def d_phase_config(args: Mapping[str, str], env: Mapping[str, str], model: Mapping[str, Any], n: int, draft: Mapping[str, Any]) -> Dict[str, Any]:
    """Einstellungen der D-Phase (TP-Raenge) aus den Zeilen der Gruppe D; ``seen`` = gelesene Eingaben mit Herkunft."""
    seen: List[Dict[str, str]] = []

    def note(what: str, value: Any, herkunft: str) -> None:
        seen.append({"was": what, "wert": str(value), "herkunft": herkunft})

    cfg: Dict[str, Any] = {"n": n, "draft": draft}
    tp = str(args.get("--rank-tp-ratio") or "").strip()
    if tp in ("auto", "auto-performance"):
        cfg["tp_ratio"] = tp
        note("tp_ratio", tp, "Profilzeile --rank-tp-ratio")
    elif _fl(tp):
        cfg["tp_ratio"] = _fl(tp)
        note("tp_ratio", tp, "Profilzeile --rank-tp-ratio")
    mr = str(args.get("--rank-moe-ratio") or "").strip()
    if mr and _fl(mr):
        cfg["moe_ratio"] = _fl(mr)
        note("moe_ratio", mr, "Profilzeile --rank-moe-ratio")
    elif mr:
        cfg["moe_ratio"] = mr
        note("moe_ratio", mr, "Profilzeile --rank-moe-ratio")
    fr = _fl(args.get("--rank-moe-resident-fraction")) or _fl(env.get("SGLANG_MOE_RESIDENT_EXPERT_FRACTION"))
    if fr:
        cfg["moe_fraction"] = fr if len(fr) > 1 else fr[0]
        note("moe_fraction", ",".join("%g" % x for x in fr),
             "Profilzeile --rank-moe-resident-fraction" if args.get("--rank-moe-resident-fraction")
             else "Umgebung SGLANG_MOE_RESIDENT_EXPERT_FRACTION (--env-d)")
    sc = _fl(env.get("SGLANG_MOE_SCRATCH_SLOTS"))
    if sc:
        cfg["scratch"] = [int(x) for x in sc] if len(sc) > 1 else int(sc[0])
        note("scratch", ",".join("%d" % x for x in sc), "Umgebung SGLANG_MOE_SCRATCH_SLOTS (--env-d)")
    bud = _fl(args.get("--rank-gpu-memory-mib"))
    if bud:
        cfg["budget_mib"] = bud
        note("budget_mib", args.get("--rank-gpu-memory-mib"), "Profilzeile --rank-gpu-memory-mib")
    else:
        note("budget_mib", "Kartengroesse - 1024 MiB", "Annahme dieser Rechnung (kein --rank-gpu-memory-mib in der Gruppe D)")
    for flag in ("--max-kv-per-request", "--context-length"):
        v = args.get(flag)
        if v and str(v).isdigit():
            cfg["context_tokens"] = int(v)
            note("context_tokens", v, "Profilzeile " + flag)
            break
    slots = args.get("--max-mamba-cache-size")
    if slots and str(slots).isdigit():
        cfg["mamba_slots"] = int(slots)
        note("mamba_slots", slots, "Profilzeile --max-mamba-cache-size")
    for key, default, why in (("context_tokens", 262144, "kein --max-kv-per-request/--context-length in der Gruppe D"),
                              ("mamba_slots", 1, "kein --max-mamba-cache-size in der Gruppe D")):
        if key not in cfg:
            note(key, default, "Annahme dieser Rechnung (%s)" % why)
    kd = args.get("--kv-cache-dtype")
    if kd and kd in ((model.get("kv") or {}).get("variants") or {}):
        cfg["kv_dtype"] = kd
        note("kv_dtype", kd, "Profilzeile --kv-cache-dtype")
    ssm = args.get("--mamba-ssm-dtype")
    if ssm and ssm in ((model.get("state") or {}).get("variants_mib") or {}):
        cfg["ssm_dtype"] = ssm
        note("ssm_dtype", ssm, "Profilzeile --mamba-ssm-dtype")
    kr = str(args.get("--rank-kv-ratio") or "coupled").strip()
    cfg["kv_ratio"] = _fl(kr) if _fl(kr) else kr
    if args.get("--rank-kv-ratio"):
        note("kv_ratio", kr, "Profilzeile --rank-kv-ratio")
    cfg["kv_token_cut"] = bool(args.get("--d-kv-token-cut")) or bool(env.get("SGLANG_UNEVEN_TOKEN_VECTOR"))
    if args.get("--d-kv-token-cut"):
        note("kv_token_cut", args.get("--d-kv-token-cut"), "Profilzeile --d-kv-token-cut")
    fo, nt = _fl(args.get("--d-foreign-context-mib")), _fl(args.get("--d-nontorch-mib"))
    if fo and nt and len(fo) == len(nt):
        cfg["fixed_mib"] = [a + b for a, b in zip(fo, nt)]
        cfg["fixed_parts"] = {"fremd": fo, "nichttorch": nt}
        note("fixed_mib", "%s + %s" % (args.get("--d-foreign-context-mib"), args.get("--d-nontorch-mib")),
             "Profilzeilen --d-foreign-context-mib (CUDA-Kontext der schlafenden Phase) + --d-nontorch-mib (VRAM ausserhalb des Torch-Allokators)")
    elif fo or nt:
        cfg["fixed_mib"] = fo or nt
        cfg["fixed_parts"] = {"fremd": fo, "nichttorch": nt}
        note("fixed_mib", args.get("--d-foreign-context-mib") or args.get("--d-nontorch-mib"),
             "nur eine der Profilzeilen --d-foreign-context-mib / --d-nontorch-mib")
    return dict(cfg, seen=seen)


def _norm_shares(ratio: Sequence[float], n: int, name: str) -> List[float]:
    if len(ratio) != n:
        raise CouplingError("vector_length: %s hat %d Werte, es gibt %d Raenge" % (name, len(ratio), n))
    tot = float(sum(ratio))
    if tot <= 0:
        raise CouplingError("%s: Summe der Gewichte ist 0" % name)
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
        raise CouplingError("Hardwareprofil: vram_total_mib fehlt auf einer Karte")
    corridor = float(cfg.get("corridor_mib", 1024.0))
    has_budget = cfg.get("budget_mib") is not None
    # Ohne --rank-gpu-memory-mib: Budget = verfuegbar im Sinn des Launchers (pp_cut.d_rank_available_mib: Karte - fremd - nichttorch - reserve);
    # die Festposten liegen AUSSERHALB des Budgets, die Annahme "Karte - Korridor" wuerde sie doppelt zaehlen.
    pre_fixed = _per_card(cfg.get("fixed_mib"), n, "fixed_mib", 0.0) if cfg.get("fixed_mib") is not None else [0.0] * n
    budgets = _per_card(cfg.get("budget_mib"), n, "budget_mib", 0.0) if has_budget else [t - f - corridor for t, f in zip(totals, pre_fixed)]
    budget_src = SRC_PROFILE if has_budget else SRC_DERIVED
    budget_note = "--rank-gpu-memory-mib" if has_budget else (
        "Kartengroesse - Festposten (fremd + nichttorch) - Korridor %.0f MiB (Annahme)" % corridor if any(pre_fixed)
        else "Kartengroesse - Korridor %.0f MiB (Annahme)" % corridor)
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
        shares, share_note = None, "--rank-tp-ratio %s: die Gewichte loest der Launcher, nicht gerechnet" % tp
    elif tp:
        shares, share_note = _norm_shares(tp, n, "--rank-tp-ratio"), "Anteil nach --rank-tp-ratio %s" % ",".join("%g" % x for x in tp)
    else:
        shares, share_note = [1.0 / n] * n, "gleichmaessiger TP (kein --rank-tp-ratio im Profil)"
    share_src = SRC_PROFILE if (tp and not isinstance(tp, str)) else SRC_APPROX

    # --- Experten ---------------------------------------------------------------------------------------------------------------------
    exp_terms: List[Dict[str, Any]] = []
    mr = cfg.get("moe_ratio")
    if E <= 0:
        exp_terms = [_term(0.0, SRC_DEFAULT, "keine Experten") for _ in range(n)]
    elif isinstance(mr, str):
        exp_terms = [none("--rank-moe-ratio %s: die Zuteilung loest der Launcher" % mr) for _ in range(n)]
    else:
        fracs = _per_card(cfg.get("moe_fraction"), n, "moe_fraction", 1.0)
        scr = _per_card(cfg.get("scratch"), n, "scratch", 0.0)
        per_expert = sum(le) / float(E) / MIB
        if mr:
            ratio = [float(x) for x in mr]
            if len(ratio) != n:
                raise CouplingError("vector_length: --rank-moe-ratio hat %d Werte, es gibt %d Raenge" % (len(ratio), n))
            as_counts = abs(sum(ratio) - E) < 1e-9
            counts = [int(x) for x in ratio] if as_counts else [int(round(E * r / sum(ratio))) for r in ratio]
            how = ("Besitz %s von %d Experten" % (",".join(str(c) for c in counts), E)) if as_counts else \
                "Besitz aus dem Verhaeltnis auf %d Experten normiert: %s" % (E, ",".join(str(c) for c in counts))
            for i in range(n):
                try:
                    rows = _er.buffer_rows(local_experts=counts[i], fraction=fracs[i], scratch_rows=int(scr[i])) if counts[i] > 0 else 0
                except ValueError as exc:
                    exp_terms.append(none(str(exc)))
                    continue
                exp_terms.append({"v": rows * per_expert, "src": SRC_APPROX,
                                  "note": "%s; %d von %d eigenen Zeilen je Layer resident (FR %.3g, Scratch %d), %.3f MiB je Zeile ueber alle Layer"
                                          % (how, rows, counts[i], fracs[i], int(scr[i]), per_expert)})
        elif shares is not None:
            for i in range(n):
                bf = _mp.expert_buffer_fraction(E, fracs[i], int(scr[i]))
                exp_terms.append({"v": shares[i] * sum(le) / MIB * bf, "src": SRC_APPROX,
                                  "note": "%s; Pufferregel %.0f %% (FR %.3g, Scratch %d); kein --rank-moe-ratio" % (share_note, 100 * bf, fracs[i], int(scr[i]))})
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
        kv_note = "Token-Eigentum nach --rank-kv-ratio %s (jeder Rang haelt alle KV-Koepfe seiner Token)" % ",".join("%g" % x for x in kvr)
    elif kvr not in (None, "coupled"):
        kv_note = "--rank-kv-ratio %s: der Loeser verteilt die Token kapazitaetsgewichtet" % kvr
    elif cfg.get("kv_token_cut"):
        kv_note = "--d-kv-token-cut/SGLANG_UNEVEN_TOKEN_VECTOR gesetzt: das Token-Eigentum loest der Planer"
    elif tp:
        kv_note = ("ungleicher TP (--rank-tp-ratio): die Verteilung der KV-Koepfe/Token folgt dem Plan der Laufzeit (Reiter KV-Koepfe), "
                   "nicht gerechnet")
    else:
        h = int(_val(model["arch"].get("heads_kv"), 0) or 0)
        if h and h >= n and h % n == 0:
            kv_shares, kv_note = [1.0 / n] * n, "gleichmaessiger TP: %d KV-Koepfe / %d Raenge" % (h, n)
        else:
            kv_note = "%d KV-Koepfe auf %d Raenge: Standardpfad der Laufzeit (Replikation/Token-Achse), nicht gerechnet" % (h, n)

    # --- Draft ------------------------------------------------------------------------------------------------------------------------------
    host: Optional[int] = 0
    if draft.get("gpu") not in (None, ""):
        match = [i for i, c in enumerate(cards) if str(c.get("nvml_index")) == str(draft["gpu"])]
        host = match[0] if match else None
    draft_terms: List[Dict[str, Any]] = [_term(0.0, SRC_DEFAULT, draft.get("reason") or "kein Draft") for _ in range(n)]
    draft_share: List[float] = [0.0] * n
    if draft["kind"] != "none":
        solo = draft.get("placement") == "solo"
        if draft["d_mib"] is None:
            draft_terms = [none(draft.get("reason") or "Draftgewicht nicht belegbar") for _ in range(n)]
        elif solo:
            if host is None:
                draft_terms = [none("--speculative-draft-gpu %s: Rang nicht aus dem Hardwareprofil ableitbar" % draft.get("gpu")) for _ in range(n)]
            else:
                draft_terms = [_term(draft["d_mib"] if i == host else 0.0, draft["src"], "solo auf Rang %d: %s" % (host, draft["d_note"]))
                               for i in range(n)]
                draft_share = [1.0 if i == host else 0.0 for i in range(n)]
        elif shares is not None:
            draft_terms = [_term(draft["d_mib"] * shares[i], SRC_APPROX,
                                 "geteilt (--speculative-draft-placement split) nach TP-Anteil: %s" % draft["d_note"]) for i in range(n)]
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
            "note": "%s (Vokabular folgt hier dem TP-Anteil, nicht --rank-vocab-ratio)" % share_note}
        t["experts"] = exp_terms[i]
        t["draft"] = draft_terms[i]
        if kv_shares is None:
            t["kv"] = none(kv_note)
        else:
            drows = draft.get("attn_layers", 1) if (draft["kind"] == "nextn" and draft_share[i] > 0) else 0
            kv_i = kv_shares[i] * kv_total + (ctx * drows * cell_mib * draft_share[i] if drows else 0.0)
            t["kv"] = {"v": kv_i, "src": SRC_APPROX, "note": "%s: %d Token x %d Attention-Layer x %.0f B%s" % (
                kv_note, ctx, attn_total, cell, " + Draft-KV" if drows else "")}
        t["state"] = none(share_note) if shares is None else {
            "v": shares[i] * state_total, "src": SRC_APPROX,
            "note": "%s; %d Linear-Layer x %.4f MiB x %d Slot(s)" % (share_note, lin_total, state_per, slots)}
        t["activation"] = _term(act_vec[i], SRC_INPUT, "gemessene Spitze (Eingabe)") if act_vec is not None else none(
            "Decode-Aktivierung und Graphen der D-Phase: nur am Metall zu messen")
        if fixed_vec is not None:
            parts = cfg.get("fixed_parts") or {}
            f_note = "; ".join("%s %s" % (k, ",".join("%g" % x for x in v)) for k, v in parts.items() if v)
            # Launcher-Semantik (pp_cut.d_rank_available_mib): verfuegbar = Karte - fremd - nichttorch - reserve; gefragt = Budget
            # (--rank-gpu-memory-mib).  Die Festposten liegen also AUSSERHALB des Budgets -> ``outside_budget`` (contract_bar rechnet sie
            # nicht gegen das Budget und zieht sie von der Reserve ab; sonst stuende derselbe Betrag zweimal im Balken).
            t["fixed"] = {"v": fixed_vec[i], "src": SRC_PROFILE, "outside_budget": True,
                          "note": "--d-foreign-context-mib (CUDA-Kontext der schlafenden Phase) + --d-nontorch-mib (liegen AUSSERHALB des Budgets, "
                                  "wie im Launcher): " + f_note}
        else:
            t["fixed"] = none("CUDA-Kontext, Graphen, Allokator-Reste: nur am Metall zu messen "
                              "(keine --d-foreign-context-mib/--d-nontorch-mib im Profil)")
        stages.append({"ord": c.get("ord", i), "label": _label(c), "total_mib": totals[i],
                       "budget_mib": {"v": budgets[i], "src": budget_src, "note": budget_note}, "terms": t})
    hints = ["D: KV gesamt fuer %d Token Kontext ueber alle Raenge: %.0f MiB (%s)" % (ctx, kv_total, cell_src)]
    return {"stages": stages, "warnings": [], "hints": hints}


def contract_bar(stage: Mapping[str, Any], phase: str) -> Dict[str, Any]:
    """Eine Stufe (``terms[k] = {v|None, src, note}``) -> Karten-Balken im Vertrag ``flliper.balken/1``.

    Posten in logischer Reihenfolge, dann ``Reserve`` (verfuegbar - Budget, von einem Ueberlauf ueber das Budget aufgezehrt) und ``Frei``
    (Budget - Posten im Budget).  Posten mit ``outside_budget`` (D-Phase: ``--d-foreign-context-mib`` + ``--d-nontorch-mib``, Launcher:
    verfuegbar = Karte - fremd - nichttorch - reserve) liegen AUSSERHALB des Budgets: sie zaehlen nicht gegen das Budget, verkleinern aber das
    Verfuegbare (Segment ``ausserhalb_budget: true``).  Die Segmente SIND der Balken: Summe = max(Kartengroesse, Posten ausserhalb + max(Budget,
    Posten im Budget)); ragt sie ueber ``total_mib``, waechst der Balken ueber die Kartengrenze (``beyond_card_mib``) -- nichts wird abgeschnitten.
    ``mib: None`` = nicht gerechnet.  Ist das Budget groesser als das Verfuegbare (``budget_over_available_mib``), lehnt der Launcher ab."""
    total = float(stage["total_mib"])
    budget = float(stage["budget_mib"]["v"])
    segs: List[Dict[str, Any]] = []
    missing: List[str] = []
    inside = 0.0
    outside = 0.0
    for key, label, what in BAR_SEGMENTS:
        t = stage["terms"].get(key)
        if t is None:
            continue
        if phase == "D":
            what = D_WHAT.get(key, what)
        v = t.get("v")
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
    known = inside + outside
    available = total - outside
    overflow = max(0.0, inside - budget)
    beyond = max(0.0, known - total)
    over_avail = max(0.0, budget - available)
    reserve = max(0.0, available - max(budget, inside))
    free = max(0.0, min(budget, available) - inside)
    bsrc = stage["budget_mib"]
    if reserve > 0:
        wish = (" -- Wunsch %.0f MiB, davon %.0f MiB aufgezehrt" % (available - budget, overflow)) if overflow > 0 else ""
        segs.append({"name": "reserve", "label": "Reserve", "mib": round(reserve, 3), "herkunft": _origin(bsrc["src"]),
                     "detail": BALKEN_SEGMENTS[-2][2] + (" -- " + bsrc["note"] if bsrc.get("note") else "") + wish, "gerechnet": True})
    if free > 0:
        obergrenze = (" -- OBERGRENZE: nicht gerechnet sind " + ", ".join(missing)) if missing else ""
        segs.append({"name": "free", "label": "Frei", "mib": round(free, 3), "herkunft": "gerechnet",
                     "detail": BALKEN_SEGMENTS[-1][2] + obergrenze, "gerechnet": True})
    return {"card": stage.get("ord"), "label": stage["label"], "phase": phase, "total_mib": total, "budget_mib": budget,
            "budget_herkunft": _origin(bsrc["src"]), "segments": segs, "posts_mib": round(known, 3), "free_mib": round(free, 3),
            "overflow_mib": round(overflow, 3), "beyond_card_mib": round(beyond, 3), "outside_budget_mib": round(outside, 3),
            "available_mib": round(available, 3), "budget_over_available_mib": round(over_avail, 3), "not_computed": missing,
            "over_text": _over_text(stage["label"], phase, overflow, beyond, budget, total, over_avail, outside)}


def _over_text(label: str, phase: str, overflow: float, beyond: float, budget: float, total: float,
               over_avail: float = 0.0, outside: float = 0.0) -> str:
    if beyond > 0:
        return ("%s (%s): Posten %.0f MiB ueber der KARTE (%.0f MiB). Der Planer lehnt ab; mit Force startet es trotzdem, zu erwarten ist OOM "
                "beim Laden oder beim Graphenaufbau." % (label, phase, beyond, total))
    if overflow > 0:
        return ("%s (%s): Posten %.0f MiB ueber dem Budget (%.0f MiB); die Reserve wird aufgezehrt. Der Planer lehnt ab; mit Force startet es "
                "trotzdem." % (label, phase, overflow, budget))
    if over_avail > 0:
        return ("%s (%s): Budget %.0f MiB ist %.0f MiB groesser als das Verfuegbare (Karte %.0f - Festposten %.0f MiB ausserhalb des Budgets). "
                "Der Planer lehnt ab; mit Force startet es trotzdem." % (label, phase, budget, over_avail, total, outside))
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

    def run_p(name: str, scope: str) -> Dict[str, Any]:
        nonlocal approx
        a = _merged(args, phase_args, scope) if scope != "-" else dict(args)
        carries = carries_p or form == "single"
        settings, seen = p_phase_settings(a, dict(penv.get("P") or {}), model, n, draft, carries_draft=carries)
        settings.update({k: v for k, v in over.items() if k != "what"})
        if "stage_layers" not in settings:
            raise CouplingError("--pp-stage-ratio fehlt im Profil: der Layer-Schnitt der P-Phase ist unbekannt")
        t = _stage_terms(hw, model, settings)
        stages = []
        for i, st in enumerate(t["stages"]):
            st2 = dict(st)
            st2["_rate_known"] = t["_ctx"]["rate"] > 0
            st2["terms"] = _p_terms_for_contract(st2, draft, carries, i == len(t["stages"]) - 1, settings)
            stages.append(st2)
        try:
            approx = approx_payload(hw, model, settings)
        except (CouplingError, KeyError, ValueError, TypeError):
            approx = None
        if name == "P" and draft["kind"] != "none" and not carries:
            seen.append({"was": "draft", "wert": "aus", "herkunft": "Profilzeile --draft-kv-on-p off: P traegt keinen MTP-Kopf"})
        label = "P-Phase (Prefill, Pipeline-Stufen)" if name == "P" else "Einzelkarte (eine Phase)"
        return {"ok": True, "label": label, "bars": [contract_bar(s, name) for s in stages], "inputs": seen,
                "context_floor_tokens": t["context_floor_tokens"], "warnings": t["warnings"]}

    def run_d() -> Dict[str, Any]:
        a = _merged(args, phase_args, "D")
        cfg = d_phase_config(a, dict(penv.get("D") or {}), model, n, draft)
        for k in ("context_tokens", "mamba_slots", "ssm_dtype", "kv_dtype", "corridor_mib", "activation_mib"):
            if k in over:
                cfg[k] = over[k]
        t = d_stage_terms(hw, model, cfg)
        return {"ok": True, "label": "D-Phase (Decode, TP-Raenge)", "bars": [contract_bar(s, "D") for s in t["stages"]],
                "inputs": cfg["seen"], "hints": t["hints"], "warnings": t["warnings"]}

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
        hints.append("Dual: P und D laufen gleichzeitig auf denselben Karten. Die Summe beider Balken ist nicht gerechnet (--dual-share: P rechnet "
                     "auf den Shards von D, die Dual-Passung ist Planer-Rechnung des AP-E, nicht hw_fit).")
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
            raise CouplingError("unbekannte Anfrage %r (compute|move|chunk|context)" % what)
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
