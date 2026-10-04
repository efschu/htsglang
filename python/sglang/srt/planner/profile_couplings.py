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
    weights_b = _mp.stage_weight_bytes(model, counts, expert_fractions=buf_fracs, replicated=replicated)
    layer_b = _val(w["layer_bytes"])
    exp_b = _val(w["layer_expert_bytes"])
    bounds = _stage_bounds(counts)
    mtp_b = float(_val(w.get("mtp_bytes"), 0.0)) if draft else 0.0
    if draft and "mtp" not in replicated:
        weights_b = list(weights_b)
        weights_b[-1] += mtp_b
    draft_note = "MTP-Layer auf der letzten Stufe" if draft else ""

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
    rate_node = (model.get("activation") or {}).get("extend_rate_mib_per_row")
    rate = float(settings["extend_rate_mib"]) if settings.get("extend_rate_mib") is not None else float(_val(rate_node, 0.0))
    rate_src = SRC_INPUT if settings.get("extend_rate_mib") is not None else _src(rate_node)

    stages = []
    for i, c in enumerate(cards):
        kv_mib = _pp.kv_reserve_mib_per_stage(
            tokens=ctx, attn_layers_by_stage=[attn[i]], draft_attn_layers_by_stage=[draft_attn[i]], **geo)[0]
        state_mib = lin[i] * state_per * slots
        act_mib = chunk * rate
        w_mib = weights_b[i] / MIB
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
                "weights": _term(w_mib, _src(w["total_bytes"]) if "total_bytes" in w else SRC_DEFAULT,
                                 "Experten residente Zeilen %.0f %%" % (100 * buf_fracs[i]) if moe else ""),
                "kv": _term(kv_mib, cell_src, "%d Token x %d Attention-Layer%s x %.0f B" % (ctx, attn[i], " + Draft" if draft_attn[i] else "", cell)),
                "state": _term(state_mib, _src(state_node), "%d Linear-Layer x %.4f MiB x %d Slot(s)" % (lin[i], state_per, slots)),
                "activation": _term(act_mib, rate_src, "%d Zeilen x %.4f MiB" % (chunk, rate)),
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
                 "buf_fracs": buf_fracs, "scratch": scratch},
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
            "weights_mib": [b["terms"]["weights"]["v"], a["terms"]["weights"]["v"]],
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
# Brueckenaufruf: eine JSON-Anfrage -> eine JSON-Antwort (Dashboard-Kindprozess, Auftrag 1431/1432)
# ---------------------------------------------------------------------------


def run(req: Mapping[str, Any]) -> Dict[str, Any]:
    """``{"what": "compute"|"move"|"chunk"|"context", "hardware": {..}, "model": {..}, "settings": {..}, ...}`` -> Ergebnis.

    ``settings`` darf stattdessen ``server_args`` (Flag -> Wert, ``profile_json.args_dict``) tragen; ``move`` braucht ``src``/``dst``/``n``,
    ``chunk`` optional ``new_chunk_tokens``.  Eine unrechenbare Eingabe kommt als ``{"ok": False, "error": ...}`` zurueck, nie als Absturz."""
    try:
        hw, model = req["hardware"], req["model"]
        settings = dict(req.get("settings") or {})
        if req.get("server_args"):
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
