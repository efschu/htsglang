# SPDX-License-Identifier: Apache-2.0
"""AP-F (Planer-Workflow 06.10.2026, Plan §3 Zeile AP-F, R5b): Vorschlag fuer die FORM EINZELKARTE (N=1).

N=1 ist KEIN weg2-Launcher-Lauf: ``weg2/topology.py:50`` ``MIN_CARDS=2`` (``:140`` SINGLE-MODE) schliesst eine Karte aus.  Die Form
ist der normale sglang-Server (``python -m sglang.launch_server``), und seine Argumente sind ``ServerArgs``
(``python/sglang/srt/server_args.py``).  Darum gibt es hier KEIN Launcher-Orakel (Plan §2 Stufe B entfaellt fuer N=1):

* **Vorschlag** = Ableitung aus Modellprofil (``flliper.model/1``, ``weg2/model_profile.py``) und Kartengroesse;
* **Passung** = Planer-Rechnung (Gewichte + Draft + KV-Pflicht + Mamba-Pool + Reserve gegen das statische Budget);
* **Ausfuehrbarkeit** = ``ServerArgs``-Parse (:func:`serverargs_parse`, argparse im Kindprozess, ohne GPU) PLUS Passung.  Der Parse
  prueft Flag-Namen, Choices und Typen; ``ServerArgs.__post_init__`` (Geraeteerkennung, Kompatibilitaets-Refusals) braucht einen
  Beschleuniger und laeuft hier NICHT -- das steht in jedem Verdikt (``art``), nie als "geprueft" verkauft.
  Das Verdikt heisst deshalb **"Planer-Rechnung"** (``VERDICT_ART``), nicht "Launcher-Dry-Run".

STDLIB ONLY.  Das Dashboard (ohne sglang-Import) laedt die Datei per Pfad; der Parse laeuft im Kindprozess.

Datenstruktur (AP-C uebernimmt sie oder bildet sie ab; Plan §2 Stufe C ``flliper.server/1``)
=============================================================================================
``propose_single`` liefert ``{"schema": SCHEMA, "form": "einzelkarte", ...}`` mit

* ``flags``: Liste von Eintraegen ``{"flag", "wert", "herkunft", "zustand", "begruendung", "verdikt"}``

  - ``flag``        CLI-Name (``--kv-cache-dtype``); ``wert`` ``None`` = nicht gesetzt (ServerArgs-Standard gilt),
                    ``True`` = Schalter ohne Wert (``--enable-hierarchical-cache``);
  - ``herkunft``    woher der Wert kommt: ``Modellprofil (Index|config|geschätzt)``, ``Kartenprofil``, ``Ziel``,
                    ``Planer-Rechnung``, ``Doku (Datei:Zeile)``, ``ServerArgs-Standard``, ``übersteuert``, ``unbelegt``;
  - ``zustand``     ``vorgeschlagen`` | ``übersteuert`` | ``unbelegt`` (Plan §1.1 (3); ``vom Launcher gelöst`` gibt es bei N=1 nicht);
  - ``begruendung`` Text mit Beleg (Datei:Zeile) und Zahlen;
  - ``verdikt``     ``{"state": "geht" | "verweigert" | "unbelegt", "code": None | "FIT-STATIC" | ..., "grund": str,
                    "art": "Planer-Rechnung", "parse": "nicht geprüft" | "ok" | "Fehler" | "veraltet (Alias)"}``.
                    ``nur mit --force`` gibt es bei N=1 nicht (kein Refusal-Register ohne Launcher).
* ``fit``: ``{"passt", "posten": [{"name","mib","herkunft","formel"}], "statisch_summe_mib", "statisch_budget_mib",
  "frei_mib" | "fehlt_mib", "reserve_*", "pre_load_free_mib", "kontext_treiber_mib", "kontext_treiber_herkunft", "kontext_treiber_standard",
  "checks": [{"code","ok","text"}], "max_context_tokens_fit", "hinweise": [...]}``

  Statisches Budget = ``--mem-fraction-static`` x ``pre_load_free_mib`` (freier Speicher VOR dem Modellladen, wie die Runtime es rechnet,
  model_runner_kv_cache_mixin.py:959-962), NICHT x Karte.  Der Posten "CUDA-Kontext + Treiber" (``kontext_treiber_mib``, Standard 400 MiB,
  Herkunft unbelegt, ``goals['pre_load_free_mib']`` ersetzt ihn) steht in ``posten``, zaehlt aber nicht in ``statisch_summe_mib``.
* ``verdikt``: ``{"state": "passt" | "passt nicht" | "unbelegt", "art": "Planer-Rechnung", "text": str}``
* ``relaxations``: Schritte, die der Planer selbst gegangen ist, damit es passt (KV fp8, Draft aus), je mit Zahlen;
* ``argv``: die ServerArgs-Argumentliste (nur Eintraege mit ``wert`` != ``None``);
* ``unbelegt``: alles, was nicht belegt ist (Eingaben ohne Quelle, nicht geprueftes ``__post_init__``).

Jede Rechenregel nennt ihre Quelle (Datei:Zeile im Baum 173161c595).  Zahlen ohne Beleg stehen als ``unbelegt``, nie geraten.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "flliper.server.single/1"
FORM = "einzelkarte"
VERDICT_ART = "Planer-Rechnung"
MIB = float(1 << 20)

# --- Herkunft / Zustand / Verdikt (Vokabular) ---------------------------------------------------------------
H_MODEL = "Modellprofil"
H_CARD = "Kartenprofil"
H_GOAL = "Ziel"
H_RULE = "Planer-Rechnung"
H_DOC = "Doku"
H_DEFAULT = "ServerArgs-Standard"
H_OVERRIDE = "übersteuert"
H_UNKNOWN = "unbelegt"

Z_PROPOSED = "vorgeschlagen"
Z_OVERRIDDEN = "übersteuert"
Z_UNBELEGT = "unbelegt"

V_OK = "geht"
V_NO = "verweigert"
V_UNK = "unbelegt"

# --- Regeln mit Beleg -------------------------------------------------------------------------------------
#: Ziel-Kontext, wenn der Aufrufer keinen nennt: "mindestens 128K, damit das Denken erhalten bleibt"
#: (docs_new/cookbook/autoregressive/OpenBMB/MiniCPM-V-4_6.mdx:63, Qwen3.5-Familie); begrenzt durch max_position_embeddings des Modells.
DEFAULT_CONTEXT_TOKENS = 131072
#: Verhaeltnis Mamba-Zustand : voller KV-Pool im Rest (server_args.py:4405-4409, model_runner_kv_cache_mixin.py:2652-2653/2706-2707).
MAMBA_FULL_MEMORY_RATIO = 0.9
#: Mamba-Slots je laufender Anfrage mit Radix-Cache, ohne extra_buffer: 1 aktiv + 1 Donation + 1 gepinnter Checkpoint
#: (mem_cache/mamba_pool_floor.py:52,57,79,228-261; model_runner_kv_cache_mixin.py:152 ..._RATIO = 3).  ``mamba_hard_floor`` :264 = Anfragen x Slots.
MAMBA_SLOTS_PER_REQ_RADIX = 3
MAMBA_SLOTS_PER_REQ_NO_RADIX = 1
#: NEXTN/MTP-Form des Kochbuchs: docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx:245-248 (steps 3, topk 1, draft tokens 4).
NEXTN_STEPS, NEXTN_TOPK, NEXTN_DRAFT_TOKENS = 3, 1, 4
#: HiCache: ``--page-size 64`` zu HiCache (docs_new/docs/advanced_features/hicache_best_practices.mdx:16,90).
HICACHE_PAGE_SIZE = 64
#: Kleinster Host/Geraet-Anteil, ab dem HiCache sich lohnt; darunter lasse ich HiCache aus (Planer-Entscheid, unbelegt am Metall).
HICACHE_MIN_RATIO = 0.05
#: CUDA-Kontext + Treiber-Abzweig, der VOR dem Modellladen schon belegt ist.  Die Runtime misst ``pre_model_load_memory`` = freier Speicher nach
#: Kontext/Init (model_runner.py:2301, utils/common.py:849 ``mem_get_info``) und rechnet den KV-/Mamba-Pool als
#: ``frei_nach_Gewichten - pre_model_load_memory x (1 - mem_fraction_static)`` (model_runner_kv_cache_mixin.py:959-962, 987-992), also ist das
#: statische Budget ``fraction x pre_model_load_memory`` und NICHT ``fraction x Karte``; Kontext + Treiber gehen gegen das Budget, nicht gegen die Reserve.
#: Standardwert: ~400 MiB GEMESSEN am 5090 im weg2-Launcher-Format (Boot weg2onebackup2 06.09.2026, launcher.py:253: "CUDA context + BAR1 windows"),
#: am Einzelserver UNBELEGT -> der Vorschlag markiert die davon abhaengigen Flags ``unbelegt``; ``goals['pre_load_free_mib']`` ersetzt ihn durch eine Messung.
CONTEXT_OVERHEAD_DEFAULT_MIB = 400.0
#: ``--max-prefill-tokens`` Standard (server_args.py:1399-1407), wenn chunked_prefill_size <= 0.
_MAX_PREFILL_TOKENS_DEFAULT = 16384

#: Flag -> (Schluessel im Werte-Satz, Anzeigename).  Nur diese Flags kennt die Rechnung; andere Uebersteuerungen laufen als Durchgriff.
_FLAG_KEYS: Dict[str, str] = {
    "--model-path": "model_path",
    "--no-enable-multimodal": "no_vision",
    "--kv-cache-dtype": "kv_dtype",
    "--context-length": "context",
    "--max-total-tokens": "kv_tokens",
    "--max-running-requests": "seats",
    "--mem-fraction-static": "fraction",
    "--max-mamba-cache-size": "slots",
    "--chunked-prefill-size": "cps",
    "--cuda-graph-max-bs-decode": "max_bs",
    "--enable-hierarchical-cache": "hicache",
    "--hicache-ratio": "hicache_ratio",
    "--page-size": "page_size",
    "--speculative-algorithm": "spec_algo",
    "--speculative-num-steps": "spec_steps",
    "--speculative-eagle-topk": "spec_topk",
    "--speculative-num-draft-tokens": "spec_draft_tokens",
    "--speculative-draft-model-path": "spec_draft_path",
    "--speculative-dflash-block-size": "spec_block",
}
#: Reihenfolge der Ausgabe.
_ORDER: Tuple[str, ...] = tuple(_FLAG_KEYS)
#: Flags, die die Passungsrechnung bestimmen (ihr Verdikt ``FIT-*`` bekommt, wenn es nicht passt).
_FIT_FLAGS = ("--mem-fraction-static", "--max-total-tokens", "--max-mamba-cache-size", "--context-length")


class ProposeSingleError(ValueError):
    """Eingaben, aus denen kein Vorschlag werden kann (Modellprofil ohne Gewichtssumme, Karte ohne Groesse) -- benannt, nie geraten."""


# ===========================================================================
# Bausteine nach server_args.py
# ===========================================================================


def capacity_tier(gpu_mem_mib: Optional[float], tp_size: int = 1) -> Tuple[int, int]:
    """``(chunked_prefill_size, decode max_bs)`` nach Speicherklasse -- server_args.py:12767-12837 (``_apply_gpu_mem_capacity_defaults``)."""
    if gpu_mem_mib is None:
        return 4096, 160
    g = float(gpu_mem_mib)
    if g < 20 * 1024:
        return 2048, 8
    if g < 35 * 1024:
        return 2048, (24 if tp_size < 4 else 80)
    if g < 60 * 1024:
        return 4096, (32 if tp_size < 4 else 160)
    if g < 160 * 1024:
        return 8192, (256 if tp_size < 4 else 512)
    return 16384, 512


def prefill_graph_sizes(max_bs: int) -> List[int]:
    """Erfasste Prefill-Graph-Groessen -- server_args.py:14974-14991 (``_generate_prefill_cuda_graph_batch_sizes``)."""
    sizes = (list(range(4, 33, 4)) + list(range(48, 257, 16)) + list(range(288, 513, 32)) + list(range(576, 1024 + 1, 64))
             + list(range(1280, 4096 + 1, 256)) + list(range(4608, max_bs + 1, 512)))
    return [s for s in sizes if s <= max_bs]


def reserve_mib(total_mib: float, cps: int, max_bs_decode: int, prefill_graph_bs: int, *, cuda: bool = True, tp: int = 1, pp: int = 1) -> float:
    """Reserve fuer Aktivierungen und Graphen (MiB) -- die ServerArgs-Heuristik, server_args.py:14509-14536 und :14825-14870:

    ``512 + max(chunked_prefill_size, 2048) x 1.5 + tp x pp / 8 x 1024 + decode max_bs x 2 [+ Zahl der Prefill-Graph-Groessen x 8, nur CUDA]``,
    bei mehr als 60 GiB Karte mindestens 10 GiB.  Der Kommentar bei server_args.py:14487-14499 haelt fest, dass der Rig-Ledger sie am 05.08.
    als zu hoch widerlegt hat (gebucht 3968 MiB, frei 1766): sie ist hier die OBERE Schranke; ``goals['reserve_mib']`` ersetzt sie durch
    eine gemessene Zahl.  Der CUDA-Kontext gehoert NICHT zur Reserve: er liegt vor ``pre_model_load_memory`` und mindert das statische Budget
    (:func:`pre_load_free`, Posten "CUDA-Kontext + Treiber" in :func:`account`)."""
    act_tokens = max(int(cps), 2048) if int(cps) > 0 else max(_MAX_PREFILL_TOKENS_DEFAULT, 2048)
    r = 512.0 + act_tokens * 1.5 + tp * pp / 8.0 * 1024.0
    r += int(max_bs_decode) * 2.0
    if cuda:
        r += int(prefill_graph_bs) * 8.0
    if total_mib > 60 * 1024:
        r = max(r, 10.0 * 1024.0)
    return r


def _floor3(x: float) -> float:
    return math.floor(x * 1000.0 + 1e-9) / 1000.0


def _floor_to(n: float, mult: int) -> int:
    n = int(math.floor(n))
    return n - (n % mult) if mult > 1 else n


def _ceil_to(n: float, mult: int) -> int:
    n = int(math.ceil(n))
    return n + (-n % mult) if mult > 1 else n


def pre_load_free(card: Mapping[str, Any], goals: Mapping[str, Any]) -> Tuple[float, float, str, bool]:
    """``(pre_load_free_mib, kontext_treiber_mib, herkunft, ist_standard)``: freier Speicher VOR dem Modellladen (Nenner von ``--mem-fraction-static``
    in der Runtime, model_runner_kv_cache_mixin.py:959-962) und der Kontext/Treiber-Posten, der ihn von der Kartengroesse trennt.
    ``goals['pre_load_free_mib']`` (gemessen) ersetzt den Standard ``Karte - CONTEXT_OVERHEAD_DEFAULT_MIB``."""
    total = float(card["total_mib"])
    given = goals.get("pre_load_free_mib")
    if given is not None:
        pre = float(given)
        if pre <= 0 or pre > total:
            raise ProposeSingleError("pre_load_free_mib %s muss in (0, Karte %.0f MiB] liegen" % (given, total))
        return pre, total - pre, "%s (pre_load_free_mib)" % H_GOAL, False
    pre = total - CONTEXT_OVERHEAD_DEFAULT_MIB
    if pre <= 0:
        raise ProposeSingleError("Karte %.0f MiB kleiner als der Kontext/Treiber-Standard %.0f MiB" % (total, CONTEXT_OVERHEAD_DEFAULT_MIB))
    return pre, CONTEXT_OVERHEAD_DEFAULT_MIB, "unbelegt (Messung am Launcher-Format, launcher.py:253; am Einzelserver nicht gemessen)", True


# ===========================================================================
# Eingaben normalisieren
# ===========================================================================


def normalize_card(card: Mapping[str, Any]) -> Dict[str, Any]:
    """Eine Karte (``flliper.hardware/1``-Eintrag ODER schlichtes Dict) -> ``{name, total_mib, usable_mib, unified, platform, ...}``.

    * ``total_mib``  was der Treiber als Gesamtspeicher meldet (NVML); der Nenner von ``--mem-fraction-static``;
    * ``usable_mib`` die adressierbare Decke (APU: GTT-Grenze, Laptop 25600 MiB laut Memory ``laptop-efeu-tp14``); Standard = ``total_mib``;
    * ``unified``    Geraet und Host teilen den Speicher (APU): das HiCache-Hostpool zaehlt dann gegen ``usable_mib``;
    * ``platform``   ``cuda`` (Standard) oder ``rocm``: ohne CUDA entfaellt der Prefill-Graph-Posten der Reserve
      (``model_executor/cuda_graph_config.py``: ``default_prefill_backend`` ist nur auf CUDA ``breakable``)."""
    if not isinstance(card, Mapping):
        raise ProposeSingleError("Karte ist kein Dict")
    total = card.get("total_mib")
    src = card.get("total_src") or H_CARD
    if total is None and isinstance(card.get("vram_total_mib"), Mapping):  # flliper.hardware/1
        node = card["vram_total_mib"]
        total, src = node.get("v"), "Kartenprofil (%s)" % node.get("src", "?")
    usable = card.get("usable_mib")
    if total is None and usable is None:
        raise ProposeSingleError("Karte ohne Speichergroesse (total_mib / vram_total_mib)")
    total = float(total if total is not None else usable)
    usable = float(usable) if usable is not None else total
    bw = card.get("bandwidth_gbs")
    if bw is None and isinstance(card.get("mem_gbs"), Mapping):
        bw = ((card["mem_gbs"].get("nameplate") or {}).get("v"))
    return {"name": str(card.get("name") or "unbekannt"), "total_mib": total, "usable_mib": min(usable, total) if not card.get("unified") else usable,
            "unified": bool(card.get("unified", False)), "platform": str(card.get("platform") or "cuda"),
            "bandwidth_gbs": bw, "cc": card.get("cc"), "total_src": src, "usable_src": card.get("usable_src") or src}


def card_from_hardware(profile: Mapping[str, Any], ordinal: int = 0) -> Dict[str, Any]:
    """Die Karte ``ordinal`` (Reihenfolge ``cards[].ord``) aus einem ``flliper.hardware/1``-Profil."""
    cards = list(profile.get("cards") or [])
    for c in cards:
        if int(c.get("ord", -1)) == int(ordinal):
            return normalize_card(c)
    if 0 <= ordinal < len(cards):
        return normalize_card(cards[ordinal])
    raise ProposeSingleError("Hardwareprofil hat keine Karte %d (hat %d)" % (ordinal, len(cards)))


def _n(profile: Mapping[str, Any], path: str) -> Optional[Mapping[str, Any]]:
    cur: Any = profile
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, Mapping) and "v" in cur else None


def _val(profile: Mapping[str, Any], path: str, default: Any = None) -> Any:
    n = _n(profile, path)
    return default if n is None else n["v"]


def _src(profile: Mapping[str, Any], path: str) -> str:
    n = _n(profile, path)
    return "%s (%s)" % (H_MODEL, n.get("src", "?")) if n else H_UNKNOWN


def model_facts(profile: Mapping[str, Any], draft_profile: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Die Zahlen aus ``flliper.model/1`` (und einem optionalen getrennten Draft-Profil, ``model_profile.estimate_draft``), die die Rechnung braucht."""
    unb: List[str] = []
    total = _val(profile, "weights.total_bytes")
    if total is None:
        raise ProposeSingleError("Modellprofil ohne weights.total_bytes")
    counts = _val(profile, "arch.layer_counts", {}) or {}
    n_attn = int(_val(profile, "kv.attn_layers", counts.get("attn", 0)) or 0)
    n_lin = int((_val(profile, "state.linear_layers", 0)) or 0)
    cell: Dict[str, float] = {}
    for key in ("auto", "fp8_e4m3"):
        c = _val(profile, "kv.variants.%s.cell_bytes_per_attn_layer_token" % key)
        if c is not None:
            cell[key] = float(c)
    if n_attn and not cell:
        unb.append("KV-Zelle fehlt im Modellprofil (kv.variants)")
    per_req = _val(profile, "state.per_request_bytes")
    if n_lin and per_req is None:
        unb.append("Mamba-Zustand je Slot fehlt im Modellprofil (state.per_request_bytes)")
    mtp_layers = int(_val(profile, "draft.mtp_layers", 0) or 0)
    mtp_bytes = float(_val(profile, "weights.mtp_bytes", 0) or 0)
    mtp_found = bool(_val(profile, "draft.mtp_tensors_found", False)) or mtp_bytes > 0
    ext: Optional[Dict[str, Any]] = None
    dp = draft_profile if draft_profile is not None else (profile.get("draft") or {}).get("external")
    if isinstance(dp, Mapping):
        kind = _val(dp, "kind", "unbekannt")
        tb = _val(dp, "total_bytes")
        ext = {"path": dp.get("path"), "kind": kind, "total_bytes": float(tb) if tb is not None else None,
               "block_size": _val(dp, "dflash.block_size"), "n_attn": int(_val(dp, "kv.attn_layers", 0) or 0)}
        ck = (dp.get("kv") or {}).get("cell_bytes_per_attn_layer_token") or {}
        ext["cell"] = {k: float(n["v"]) for k, n in ck.items() if isinstance(n, Mapping) and n.get("v") is not None}
        sl = (dp.get("kv") or {}).get("sliding")
        ext["window"] = int(sl["window_tokens"]["v"]) if sl and sl.get("window_tokens") else None
        ext["window_layers"] = int(sl["layers"]["v"]) if sl and sl.get("layers") else None
        if tb is None:
            unb.append("Draft-Profil ohne total_bytes")
    kvs = (profile.get("kv") or {}).get("sliding")
    return {
        "path": profile.get("path"), "id": profile.get("id"), "format": _val(profile, "format"),
        "family": _val(profile, "arch.family", "unbekannt"), "hybrid": bool(_val(profile, "arch.hybrid", False)),
        "n_layers": int(_val(profile, "arch.n_layers", 0) or 0), "n_attn": n_attn, "n_lin": n_lin,
        "vision": bool(_val(profile, "arch.vision", False)),
        "total_bytes": float(total), "visual_bytes": float(_val(profile, "weights.visual_bytes", 0) or 0), "mtp_bytes": mtp_bytes,
        "mtp_layers": mtp_layers, "mtp_found": mtp_found and mtp_layers > 0,
        "cell": cell, "per_req": float(per_req) if per_req is not None else None, "maxpos": int(_val(profile, "context.max_position_embeddings", 0) or 0),
        "kv_sliding": bool(kvs), "ext": ext, "unbelegt": unb,
        "src": {"total": _src(profile, "weights.total_bytes"), "visual": _src(profile, "weights.visual_bytes"), "mtp": _src(profile, "weights.mtp_bytes"),
                "cell": _src(profile, "kv.variants.auto.cell_bytes_per_attn_layer_token"), "state": _src(profile, "state.per_request_bytes"),
                "maxpos": _src(profile, "context.max_position_embeddings")},
    }


# ===========================================================================
# Die Rechnung (Posten, Checks)
# ===========================================================================


def _cell_for(F: Mapping[str, Any], dtype: str) -> Tuple[Optional[float], str]:
    """KV-Zelle je Attention-Layer und Token fuer ``--kv-cache-dtype``; ``(None, Grund)`` wenn nicht belegt."""
    if dtype in ("auto",):
        return F["cell"].get("auto"), "auto = Modell-Dtype"
    if dtype in ("fp8_e4m3", "fp8_e5m2"):
        # fp8_e5m2 hat dieselbe Bytezahl; ob e4m3 den Skalenpuffer anders traegt als e5m2, ist im Profil nicht getrennt (unbelegt) -> e4m3-Zelle.
        return F["cell"].get("fp8_e4m3"), "fp8 (1 B + Skalenpuffer)" + ("; e5m2 = e4m3-Zelle (unbelegt)" if dtype == "fp8_e5m2" else "")
    if dtype in ("bf16", "bfloat16"):
        return F["cell"].get("auto"), "bf16 = auto-Zelle, gilt nur, wenn der Modell-Dtype bf16 ist (unbelegt, sonst)"
    return None, "kv_dtype %s: Zelle nicht im Modellprofil (unbelegt)" % dtype


def account(F: Mapping[str, Any], card: Mapping[str, Any], v: Mapping[str, Any], goals: Mapping[str, Any]) -> Dict[str, Any]:
    """Posten und Checks fuer EINEN Werte-Satz ``v`` (Schluessel wie ``_FLAG_KEYS``-Werte).  Wird fuer den Vorschlag UND fuer Uebersteuerungen genutzt."""
    posten: List[Dict[str, Any]] = []
    hinweise: List[str] = []
    unb: List[str] = []
    total = float(card["total_mib"])
    usable = float(card["usable_mib"])

    def add(name: str, nbytes: float, herkunft: str, formel: str) -> float:
        mib = nbytes / MIB
        posten.append({"name": name, "mib": round(mib, 1), "herkunft": herkunft, "formel": formel})
        return mib

    w_main_b = F["total_bytes"] - F["mtp_bytes"] - (F["visual_bytes"] if v.get("no_vision") else 0.0)
    w_main = add("Gewichte (Hauptmodell)", w_main_b, F["src"]["total"],
                 "Summe aller Tensoren - mtp (der Ziel-Lader ueberspringt 'mtp', models/qwen3_5.py:2127-2129)"
                 + (" - Sichttuerm (--no-enable-multimodal, server_args.py:1061-1076)" if v.get("no_vision") else ""))
    spec = v.get("spec_algo")
    draft_kind = None
    w_draft = kv_draft = 0.0
    if spec and str(spec).upper() == "NEXTN":
        draft_kind = "nextn"
        w_draft = add("Gewichte (Draft: MTP-Kopf)", F["mtp_bytes"], F["src"]["mtp"], "mtp-Tensoren des Checkpoints (Index)")
    elif spec:
        draft_kind = "external"
        e = F["ext"]
        if e and e.get("total_bytes") is not None:
            w_draft = add("Gewichte (Draft: %s)" % e.get("kind"), e["total_bytes"], "%s (%s)" % (H_MODEL, "Draft-Profil"),
                          "Gesamtbytes des Draft-Verzeichnisses; Teilen von Einbettung/lm_head mit dem Ziel ist im einzelnen Server unbelegt (obere Schranke)")
        else:
            unb.append("Draft %r ohne Draft-Profil: Gewichte nicht gerechnet" % spec)

    kv_tokens = int(v.get("kv_tokens") or 0)
    dtype = str(v.get("kv_dtype") or "auto")
    cell, cell_note = _cell_for(F, dtype)
    kv_main = 0.0
    kv_bpt = 0.0  # Bytes je Token, Haupt-Pool + Draft
    if F["n_attn"]:
        if cell is None:
            unb.append(cell_note)
        else:
            kv_bpt_main = cell * F["n_attn"]
            kv_main = add("KV-Pool (Hauptmodell)", kv_bpt_main * kv_tokens, F["src"]["cell"],
                          "%d Token x %d Attention-Layer x %.0f B (%s)" % (kv_tokens, F["n_attn"], cell, cell_note))
            kv_bpt += kv_bpt_main
            if F["kv_sliding"]:
                hinweise.append("Gleitfenster-Layer des Hauptmodells sind NICHT abgezogen (obere Schranke)")
    if draft_kind == "nextn" and cell is not None:
        per_tok = cell * F["mtp_layers"]
        kv_draft = add("KV-Pool (Draft: MTP)", per_tok * kv_tokens, F["src"]["cell"],
                       "%d Token x %d MTP-Layer x %.0f B (MTP-Layer = volle Attention-Schicht, model_profile.estimate_draft)" % (kv_tokens, F["mtp_layers"], cell))
        kv_bpt += per_tok
    elif draft_kind == "external" and F["ext"] and F["ext"].get("cell"):
        e = F["ext"]
        ec = e["cell"].get("fp8_e4m3" if dtype.startswith("fp8") else "auto")
        if ec is not None and e.get("n_attn"):
            tok = kv_tokens
            note = "%d Token" % kv_tokens
            if e.get("window") and e.get("window_layers") == e.get("n_attn"):
                tok = min(kv_tokens, int(e["window"]))
                note = "min(%d Token, Gleitfenster %d)" % (kv_tokens, e["window"])
            kv_draft = add("KV-Pool (Draft: %s)" % e.get("kind"), ec * e["n_attn"] * tok, "%s (Draft-Profil)" % H_MODEL,
                           "%s x %d Layer x %.0f B" % (note, e["n_attn"], ec))
            if tok == kv_tokens:
                kv_bpt += ec * e["n_attn"]
        else:
            unb.append("Draft-KV-Zelle nicht im Draft-Profil")

    slots = int(v.get("slots") or 0)
    mamba_main = mamba_spec = 0.0
    seats = int(v.get("seats") or 1)
    floor_slots = (MAMBA_SLOTS_PER_REQ_NO_RADIX if goals.get("disable_radix") else MAMBA_SLOTS_PER_REQ_RADIX) * seats
    if F["n_lin"] and F["per_req"] is not None:
        mamba_main = add("Mamba-Zustandspool", F["per_req"] * slots, F["src"]["state"],
                         "%d Slots x %.2f MiB (conv + ssm aller %d Linear-Layer, ein Slot)" % (slots, F["per_req"] / MIB, F["n_lin"]))
        if draft_kind:
            D = int(v.get("spec_draft_tokens") or v.get("spec_block") or 0)
            ratio = MAMBA_SLOTS_PER_REQ_NO_RADIX if goals.get("disable_radix") else MAMBA_SLOTS_PER_REQ_RADIX
            capped = min(seats, slots // ratio) if ratio else seats
            mamba_spec = add("Mamba-Zwischenzustaende (Spekulation)", F["per_req"] * capped * D, F["src"]["state"],
                             "min(max-running-requests %d, Slots/%d) x %d Draft-Token x %.2f MiB (model_runner_kv_cache_mixin.py:2736-2745)" % (seats, ratio, D, F["per_req"] / MIB))

    host = 0.0
    host_mib = goals.get("host_ram_mib")
    if card["unified"] and v.get("hicache") and host_mib:
        host = add("HiCache-Hostpool (gleicher Speicher)", float(host_mib) * MIB, H_GOAL,
                   "APU: Geraet und Host teilen den Speicher; das Host-Budget des Ziels zaehlt gegen die adressierbare Decke")

    static_sum = w_main + w_draft + kv_main + kv_draft + mamba_main + mamba_spec
    # Reserve und statisches Budget (server_args.py:14538-14541: fraction = (gpu_mem - reserved) / gpu_mem)
    cps = int(v.get("cps") or 0)
    pf_max = cps
    if v.get("context"):
        pf_max = min(pf_max, int(v["context"]))
    if kv_tokens:
        pf_max = min(pf_max, kv_tokens)
    reserve_need = goals.get("reserve_mib")
    reserve_src = H_GOAL
    if reserve_need is None:
        reserve_need = reserve_mib(total, cps, int(v.get("max_bs") or 1), len(prefill_graph_sizes(max(pf_max, 0))), cuda=card["platform"] == "cuda")
        reserve_src = "ServerArgs-Heuristik (server_args.py:14509-14536)"
    fraction = v.get("fraction")
    # Statisches Budget wie die Runtime es rechnet: fraction x freier Speicher VOR dem Laden (nicht x Karte); Kontext + Treiber liegen davor.
    pre, overhead, overhead_src, overhead_default = pre_load_free(card, goals)
    add("CUDA-Kontext + Treiber (vor dem Laden belegt, nicht im statischen Budget)", overhead * MIB, overhead_src,
        "Karte %.0f MiB - freier Speicher vor dem Laden %.0f MiB (model_runner.py:2301 pre_model_load_memory); mindert das Budget, nicht die Reserve" % (total, pre))
    budget = float(fraction) * pre if fraction is not None else None
    # APU: das Hostpool liegt im selben Speicher und verengt den Bruchteil (_plan_values); es ist KEINE Reserve.
    reserve_given = (pre - budget - host) if budget is not None else None
    checks: List[Dict[str, Any]] = []
    if budget is None:
        checks.append({"code": "FIT-RESERVE", "ok": False,
                       "text": "Reserve %.0f MiB (%s) laesst keinen Bruchteil des freien Speichers %.0f MiB (Karte %.0f MiB - Kontext/Treiber %.0f MiB) fuer --mem-fraction-static" % (reserve_need, reserve_src, pre, total, overhead)})
    else:
        ok_static = static_sum <= budget + 1e-6
        checks.append({"code": "FIT-STATIC", "ok": ok_static,
                       "text": ("Gewichte + Draft + KV + Mamba %.0f MiB %s statisches Budget %.0f MiB (= --mem-fraction-static %.3f x freier Speicher vor dem Laden %.0f MiB = Karte %.0f - Kontext/Treiber %.0f)%s"
                                % (static_sum, "<=" if ok_static else ">", budget, fraction, pre, total, overhead,
                                   "" if ok_static else "; es fehlen %.0f MiB" % (static_sum - budget)))})
        ok_res = reserve_given + 1e-6 >= reserve_need
        checks.append({"code": "FIT-RESERVE", "ok": ok_res,
                       "text": ("Reserve %.0f MiB (= (1 - fraction) x freier Speicher%s) %s Bedarf %.0f MiB (%s)" % (reserve_given, " - Hostpool" if host else "", ">=" if ok_res else "<", reserve_need, reserve_src))})
        usable_eff = usable - overhead - host
        if usable < total or host:
            ok_card = budget <= usable_eff + 1e-6
            checks.append({"code": "FIT-CARD", "ok": ok_card,
                           "text": "statisches Budget %.0f MiB %s adressierbare Decke %.0f MiB%s" % (
                               budget, "<=" if ok_card else ">", usable_eff, " (abzueglich Kontext/Treiber %.0f MiB%s)" % (overhead, ", Hostpool %.0f MiB" % host if host else ""))})
    if v.get("context") and kv_tokens:
        ok_ctx = int(v["context"]) <= kv_tokens
        checks.append({"code": "FIT-CTX", "ok": ok_ctx, "text": "Kontext %d Token %s KV-Pool %d Token" % (int(v["context"]), "<=" if ok_ctx else ">", kv_tokens)})
    if F["n_lin"]:
        ok_floor = slots >= floor_slots
        checks.append({"code": "MAMBA-FLOOR", "ok": ok_floor,
                       "text": "Mamba-Slots %d %s Untergrenze %d (%d Anfragen x %d Slots; mamba_pool_floor.py:264)" % (
                           slots, ">=" if ok_floor else "<", floor_slots, seats, floor_slots // max(seats, 1))})
    if v.get("context") and F["maxpos"] and int(v["context"]) > F["maxpos"]:
        hinweise.append("Kontext %d liegt ueber max_position_embeddings %d des Modells: Rope-Erweiterung ist nicht belegt" % (int(v["context"]), F["maxpos"]))
    if F["family"] == "moe":
        hinweise.append("MoE im einzelnen Server: alle Experten muessen auf der Karte liegen (kein Experten-Offload in dieser Form); Teilauslagerung ist Sache des weg2-Launchers (N>=2)")
    if overhead_default:
        hinweise.append("Kontext + Treiber %.0f MiB sind der Standardwert (am Einzelserver unbelegt): das statische Budget ist fraction x (Karte - Kontext/Treiber); "
                        "gemessenes pre_model_load_memory als goals['pre_load_free_mib'] setzen" % overhead)
    if card["unified"]:
        hinweise.append("APU: Geraet und Host teilen den Speicher; torch-Gesamtspeicher der APU ist hier = adressierbare Decke angenommen (unbelegt am Rig)")
    out = {"passt": all(c["ok"] for c in checks) and not unb, "posten": posten, "statisch_summe_mib": round(static_sum, 1),
           "statisch_budget_mib": round(budget, 1) if budget is not None else None,
           "reserve_bedarf_mib": round(reserve_need, 1), "reserve_herkunft": reserve_src,
           "reserve_gegeben_mib": round(reserve_given, 1) if reserve_given is not None else None,
           "pre_load_free_mib": round(pre, 1), "kontext_treiber_mib": round(overhead, 1), "kontext_treiber_herkunft": overhead_src,
           "kontext_treiber_standard": overhead_default,
           "checks": checks, "hinweise": hinweise, "unbelegt": unb,
           "kv_bytes_je_token": kv_bpt, "mamba_floor_slots": floor_slots,
           "_numbers": {"w_main": w_main, "w_draft": w_draft, "mamba_spec": mamba_spec, "kv_main": kv_main, "kv_draft": kv_draft, "mamba_main": mamba_main, "host": host}}
    if budget is not None:
        d = budget - static_sum
        out["frei_mib" if d >= 0 else "fehlt_mib"] = round(abs(d), 1)
    return out


# ===========================================================================
# Ableitung des Vorschlags
# ===========================================================================


def _defaults_goals(goals: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    g = {"context_tokens": None, "kv_tokens": None, "seats": 1, "draft": "auto", "kv_dtype": None, "vision": False, "host_ram_mib": None,
         "chunked_prefill_size": None, "reserve_mib": None, "disable_radix": False, "pre_load_free_mib": None}
    for k, val in (goals or {}).items():
        if k not in g:
            raise ProposeSingleError("unbekanntes Ziel %r (bekannt: %s)" % (k, ", ".join(sorted(g))))
        g[k] = val
    if int(g["seats"]) < 1:
        raise ProposeSingleError("seats muss >= 1 sein")
    return g


def _plan_values(F: Mapping[str, Any], card: Mapping[str, Any], g: Mapping[str, Any], dtype: str, draft: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Mindest-Werte (KV = Ziel-Kontext, Mamba = Untergrenze) fuer EINE Stufe der Leiter; passt es, wird der Rest verteilt."""
    total, usable = float(card["total_mib"]), float(card["usable_mib"])
    pre, overhead, _, _ = pre_load_free(card, g)
    cps_tier, bs_tier = capacity_tier(total)
    cps = int(g["chunked_prefill_size"] or cps_tier)
    seats = int(g["seats"])
    ctx = int(g["context_tokens"] or (min(F["maxpos"], DEFAULT_CONTEXT_TOKENS) if F["maxpos"] else DEFAULT_CONTEXT_TOKENS))
    host = g.get("host_ram_mib")
    hicache = bool(host) and float(host) > 0
    page = HICACHE_PAGE_SIZE if hicache else 1
    kv_goal = _ceil_to(int(g["kv_tokens"] or ctx), page)
    v: Dict[str, Any] = {
        "model_path": F["path"], "no_vision": bool(F["vision"] and not g["vision"]), "kv_dtype": dtype, "context": ctx, "kv_tokens": kv_goal, "seats": seats,
        "cps": cps, "max_bs": min(bs_tier, max(seats, 1)), "hicache": hicache, "page_size": page if hicache else None,
        "slots": (MAMBA_SLOTS_PER_REQ_NO_RADIX if g["disable_radix"] else MAMBA_SLOTS_PER_REQ_RADIX) * seats if F["n_lin"] else None,
        "spec_algo": None, "spec_steps": None, "spec_topk": None, "spec_draft_tokens": None, "spec_draft_path": None, "spec_block": None,
        "hicache_ratio": None,
    }
    if draft == "nextn":
        v.update(spec_algo="NEXTN", spec_steps=NEXTN_STEPS, spec_topk=NEXTN_TOPK, spec_draft_tokens=NEXTN_DRAFT_TOKENS)
    elif draft == "external":
        e = F["ext"] or {}
        block = e.get("block_size")
        v.update(spec_algo="DFLASH", spec_draft_path=e.get("path"), spec_block=int(block) if block else None)
    reserve = g["reserve_mib"]
    if reserve is None:
        reserve = reserve_mib(total, cps, v["max_bs"], len(prefill_graph_sizes(min(cps, ctx, kv_goal))), cuda=card["platform"] == "cuda")
    host_unified = float(host) if (card["unified"] and hicache) else 0.0
    # Die Runtime wendet den Bruchteil auf den freien Speicher VOR dem Laden an (pre_model_load_memory, model_runner_kv_cache_mixin.py:959-962), nicht auf
    # die Karte: Budget = f x pre, Reserve (Slack) = pre x (1 - f).  Gewollt: Budget = Decke - Kontext/Treiber - Hostpool - Reserve.
    f = _floor3(min(0.99, (usable - overhead - host_unified - reserve) / pre)) if pre > 0 else 0.0
    v["fraction"] = f if f > 0 else None
    return v, {"reserve": reserve, "pre": pre, "overhead": overhead, "ctx": ctx, "kv_goal": kv_goal, "page": page, "host_unified": host_unified}


def _spend_rest(F: Mapping[str, Any], card: Mapping[str, Any], g: Mapping[str, Any], v: Dict[str, Any], aux: Mapping[str, Any]) -> None:
    """Rest-Regel Dicht (K4, Plan §1.2): was ueber die Pflicht hinaus im statischen Budget bleibt, geht im ServerArgs-Verhaeltnis
    ``mamba_full_memory_ratio`` (0.9, server_args.py:4405) an Mamba-Slots und KV-Token; ohne Mamba alles an KV."""
    acc = account(F, card, v, g)
    if v.get("fraction") is None or not acc["passt"]:
        return
    slack = acc["statisch_budget_mib"] - acc["statisch_summe_mib"]
    if slack <= 0 or not acc["kv_bytes_je_token"]:
        return
    has_mamba = bool(F["n_lin"] and F["per_req"])
    kv_share = slack / (1.0 + MAMBA_FULL_MEMORY_RATIO) if has_mamba else slack
    extra = int(kv_share * MIB // acc["kv_bytes_je_token"])
    page = int(aux["page"])
    new_tokens = _floor_to(v["kv_tokens"] + extra, page)
    spent = (new_tokens - v["kv_tokens"]) * acc["kv_bytes_je_token"] / MIB
    v["kv_tokens"] = new_tokens
    if has_mamba:
        rest = slack - spent
        v["slots"] = int(v["slots"]) + int(rest * MIB // F["per_req"])


def derive(F: Mapping[str, Any], card: Mapping[str, Any], g: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Werte-Satz + Hilfsgroessen + Leiter-Schritte + Rechnung.  Leiter, wenn das Ziel nicht passt (in dieser Reihenfolge, nur wo das Ziel es erlaubt):
    (1) KV fp8 statt Modell-Dtype, (2) Draft weglassen.  Der Kontext wird NIE still gekuerzt (er ist das Ziel); ``max_context_tokens_fit`` nennt, was passt."""
    # Draft-Wahl
    draft_goal = g["draft"]
    nextn_ok = bool(F["mtp_found"])
    ext_ok = bool(F["ext"] and F["ext"].get("total_bytes") is not None and str(F["ext"].get("kind")) in ("dflash", "dflash2"))
    draft: Optional[str] = None
    if draft_goal in ("auto", "on", "nextn", "external"):
        if draft_goal in ("auto", "on", "external") and ext_ok:
            draft = "external"
        elif draft_goal in ("auto", "on", "nextn") and nextn_ok:
            draft = "nextn"
    dtype_goal = g["kv_dtype"]
    dtype = dtype_goal or "auto"
    relax: List[Dict[str, Any]] = []
    while True:
        v, aux = _plan_values(F, card, g, dtype, draft)
        acc = account(F, card, v, g)
        if acc["passt"]:
            break
        if dtype_goal is None and dtype == "auto" and F["cell"].get("fp8_e4m3") is not None:
            relax.append({"schritt": "KV-Dtype fp8_e4m3", "grund": "mit Modell-Dtype fehlen %.0f MiB (KV %.0f MiB)" % (acc.get("fehlt_mib", 0.0), acc["_numbers"]["kv_main"])})
            dtype = "fp8_e4m3"
            continue
        if draft is not None and draft_goal == "auto":
            relax.append({"schritt": "Draft weggelassen", "grund": "es fehlen %.0f MiB; der Draft kostet %.0f MiB Gewichte + %.0f MiB KV + %.0f MiB Spekulationszustand"
                          % (acc.get("fehlt_mib", 0.0), acc["_numbers"]["w_draft"], acc["_numbers"]["kv_draft"], acc["_numbers"]["mamba_spec"])})
            draft = None
            continue
        break
    if acc["passt"]:
        _spend_rest(F, card, g, v, aux)
    # HiCache-Verhaeltnis nach dem endgueltigen KV-Pool
    if v.get("hicache"):
        kv_main_mib = (v["kv_tokens"] * F["cell"].get(("fp8_e4m3" if str(v["kv_dtype"]).startswith("fp8") else "auto"), 0.0) * F["n_attn"]) / MIB
        host = float(g["host_ram_mib"])
        ratio = _floor_to(host / kv_main_mib * 100.0, 1) / 100.0 if kv_main_mib > 0 else 0.0
        if ratio < HICACHE_MIN_RATIO:
            v["hicache"] = False
            v["page_size"] = None
            v["hicache_ratio"] = None
            aux = dict(aux, hicache_skipped="Host-Budget %.0f MiB = %.3f x KV-Pool %.0f MiB < %.2f" % (host, ratio, kv_main_mib, HICACHE_MIN_RATIO))
        else:
            v["hicache_ratio"] = ratio
            aux = dict(aux, host_kv_mib=round(ratio * kv_main_mib, 1), kv_main_mib=round(kv_main_mib, 1))
        # (Seitengroesse war schon in die KV-Token eingerechnet; faellt HiCache aus, bleibt der Pool glatt durch 64 teilbar -- harmlos.)
    acc = account(F, card, v, g)
    return v, aux, relax, acc


def max_context_fit(F: Mapping[str, Any], card: Mapping[str, Any], v: Mapping[str, Any], g: Mapping[str, Any]) -> Optional[int]:
    """Groesster Kontext (Token), den der KV-Pool bei den uebrigen Werten von ``v`` und dem statischen Budget von ``v`` haelt; ``None`` = nicht rechenbar."""
    if v.get("fraction") is None:
        return None
    probe = dict(v, kv_tokens=0)
    acc = account(F, card, probe, g)
    if acc["unbelegt"] or not acc["kv_bytes_je_token"]:
        return None
    room = acc["statisch_budget_mib"] - acc["statisch_summe_mib"]
    tokens = max(0, int(room * MIB // acc["kv_bytes_je_token"]))
    page = int(v.get("page_size") or 1)
    tokens = _floor_to(tokens, page)
    return min(tokens, F["maxpos"]) if F["maxpos"] else tokens


# ===========================================================================
# Eintraege
# ===========================================================================


def _entry(flag: str, wert: Any, herkunft: str, begruendung: str, *, state: str = V_OK, code: Optional[str] = None, grund: str = "",
           zustand: str = Z_PROPOSED) -> Dict[str, Any]:
    return {"flag": flag, "wert": wert, "herkunft": herkunft, "zustand": zustand, "begruendung": begruendung,
            "verdikt": {"state": state, "code": code, "grund": grund, "art": VERDICT_ART, "parse": "nicht geprüft"}}


def _fmt(x: float) -> str:
    return "%.0f" % x


def build_entries(F: Mapping[str, Any], card: Mapping[str, Any], g: Mapping[str, Any], v: Mapping[str, Any], aux: Mapping[str, Any],
                  relax: Sequence[Mapping[str, Any]], acc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    n = acc["_numbers"]
    failing = {c["code"]: c["text"] for c in acc["checks"] if not c["ok"]}
    fit_state = V_NO if failing else (V_UNK if acc["unbelegt"] else V_OK)
    fit_code = "+".join(sorted(failing)) if failing else None
    fit_grund = " | ".join(failing.values()) if failing else ("; ".join(acc["unbelegt"]) if acc["unbelegt"] else "")
    out: Dict[str, Dict[str, Any]] = {}
    out["--model-path"] = _entry("--model-path", v["model_path"], "%s (%s)" % (H_MODEL, "Pfad"), "Modellordner des Profils %s (Format %s, %s)" % (F["id"], F["format"], F["family"]))
    if F["vision"]:
        out["--no-enable-multimodal"] = _entry(
            "--no-enable-multimodal", True if v["no_vision"] else None, H_GOAL if v["no_vision"] else H_DEFAULT,
            ("Sichttuerm aus: spart %s MiB Gewichte (%s); server_args.py:1061-1076 nennt 879 MiB gemessen (Boot weg2xsn27)" % (_fmt(F["visual_bytes"] / MIB), F["src"]["visual"]))
            if v["no_vision"] else "Ziel 'vision': Sichttuerm bleibt geladen (%s MiB)" % _fmt(F["visual_bytes"] / MIB))
    dtype = str(v["kv_dtype"])
    dreason = "Modell-Dtype (auto)" if dtype == "auto" else "fp8"
    if relax and any(r["schritt"].startswith("KV-Dtype") for r in relax):
        dreason = "Leiter: " + next(r["grund"] for r in relax if r["schritt"].startswith("KV-Dtype"))
    elif g["kv_dtype"]:
        dreason = "Ziel"
    out["--kv-cache-dtype"] = _entry("--kv-cache-dtype", dtype, H_GOAL if g["kv_dtype"] else H_RULE,
                                     "%s; KV-Zelle %s B je Attention-Layer und Token (%s)" % (dreason, _fmt(_cell_for(F, dtype)[0] or 0), F["src"]["cell"]))
    out["--context-length"] = _entry("--context-length", v["context"], H_GOAL if g["context_tokens"] else H_RULE,
                                     ("Ziel-Kontext" if g["context_tokens"] else "Standardziel min(max_position_embeddings %d, %d) (MiniCPM-V-4_6.mdx:63)" % (F["maxpos"], DEFAULT_CONTEXT_TOKENS)),
                                     state=V_NO if "FIT-CTX" in failing else V_OK, code="FIT-CTX" if "FIT-CTX" in failing else None, grund=failing.get("FIT-CTX", ""))
    out["--max-total-tokens"] = _entry("--max-total-tokens", v["kv_tokens"], H_RULE,
                                       "KV-Pool: Pflicht %d Token (Ziel) + Rest im Verhaeltnis %.1f zu Mamba (server_args.py:4405); Seitengroesse %d; %s MiB" % (
                                           aux["kv_goal"], MAMBA_FULL_MEMORY_RATIO, aux["page"], _fmt(n["kv_main"] + n["kv_draft"])),
                                       state=fit_state if v["kv_tokens"] else V_UNK, code=fit_code, grund=fit_grund)
    out["--max-running-requests"] = _entry("--max-running-requests", v["seats"], H_GOAL, "Ziel 'Sitze gleichzeitig' (Standard 1)")
    fr = v.get("fraction")
    out["--mem-fraction-static"] = _entry(
        "--mem-fraction-static", fr, H_RULE,
        ("(Decke %s MiB - Kontext/Treiber %s MiB - Reserve %s MiB) / freier Speicher vor dem Laden %s MiB, abgerundet auf 3 Stellen -- die Runtime wendet den Bruchteil "
         "auf pre_model_load_memory an (model_runner_kv_cache_mixin.py:959-962), nicht auf die Karte (Abweichung von server_args.py:14538-14541, das durch die Karte teilt); Reserve aus %s%s" % (
            _fmt(card["usable_mib"] - aux["host_unified"]), _fmt(aux["overhead"]), _fmt(aux["reserve"]), _fmt(aux["pre"]), acc["reserve_herkunft"],
            "; Hostpool %s MiB abgezogen (APU)" % _fmt(aux["host_unified"]) if aux["host_unified"] else ""))
        if fr is not None else "Reserve %s MiB (+ Kontext/Treiber %s MiB) uebersteigt die Karte: kein Bruchteil moeglich" % (_fmt(aux["reserve"]), _fmt(aux["overhead"])),
        state=fit_state if fr is not None else V_NO, code=fit_code if fr is not None else "FIT-RESERVE", grund=fit_grund if fr is not None else "Reserve groesser als die Karte")
    if F["n_lin"] and v.get("slots") is not None:
        out["--max-mamba-cache-size"] = _entry(
            "--max-mamba-cache-size", v["slots"], H_RULE,
            "Untergrenze %d (= %d Anfragen x %d Slots, mamba_pool_floor.py:264) + Rest; je Slot %s MiB (%s)" % (
                acc["mamba_floor_slots"], v["seats"], acc["mamba_floor_slots"] // max(v["seats"], 1), _fmt((F["per_req"] or 0) / MIB), F["src"]["state"]),
            state=fit_state, code=fit_code, grund=fit_grund)
    out["--chunked-prefill-size"] = _entry("--chunked-prefill-size", v["cps"], H_GOAL if g["chunked_prefill_size"] else H_DEFAULT,
                                           "Kartenklasse %s MiB (server_args.py:12767-12837); bestimmt die Aktivierungsreserve" % _fmt(card["total_mib"]))
    out["--cuda-graph-max-bs-decode"] = _entry("--cuda-graph-max-bs-decode", v["max_bs"], H_RULE,
                                               "min(Kartenklassen-Standard, max-running-requests): groessere Decode-Graphen wuerden nie benutzt, ihre Reserve waechst mit max_bs (server_args.py:14847); kanonischer Name, '--cuda-graph-max-bs' ist veralteter Alias (server_args.py:19969-19977)")
    out["--enable-hierarchical-cache"] = _entry(
        "--enable-hierarchical-cache", True if v["hicache"] else None, H_GOAL if v["hicache"] else H_UNKNOWN,
        "Host-Budget %s MiB vorhanden (Ziel)" % _fmt(float(g.get("host_ram_mib") or 0)) if v["hicache"] else
        ("HiCache aus: " + aux["hicache_skipped"] if aux.get("hicache_skipped") else "kein Host-RAM-Budget im Ziel (host_ram_mib): HiCache nicht vorgeschlagen"),
        zustand=Z_PROPOSED if v["hicache"] or aux.get("hicache_skipped") else Z_UNBELEGT)
    if v["hicache"]:
        out["--hicache-ratio"] = _entry("--hicache-ratio", v["hicache_ratio"], H_RULE,
                                        "Host-Budget %s MiB / Geraete-KV-Pool %s MiB (nur Hauptmodell; ob der Draft-Pool mitgerechnet wird, ist unbelegt) -> Host-KV %s MiB"
                                        % (_fmt(float(g.get("host_ram_mib") or 0)), _fmt(aux.get("kv_main_mib", 0.0)), _fmt(aux.get("host_kv_mib", 0.0))))
        out["--page-size"] = _entry("--page-size", v["page_size"], "%s (hicache_best_practices.mdx:16)" % H_DOC, "HiCache-Empfehlung der Doku: Seitengroesse 64; der KV-Pool ist auf ein Vielfaches gerundet")
    if v.get("spec_algo"):
        src_d = "Doku (Qwen3.5.mdx:245-248)" if str(v["spec_algo"]).upper() == "NEXTN" else "Draft-Profil"
        why = ("Draft: MTP-Kopf im Checkpoint (%s MiB, %s); NEXTN-Form des Kochbuchs" % (_fmt(F["mtp_bytes"] / MIB), F["src"]["mtp"])
               if str(v["spec_algo"]).upper() == "NEXTN" else "Draft: getrenntes Verzeichnis (%s)" % (F["ext"] or {}).get("kind"))
        out["--speculative-algorithm"] = _entry("--speculative-algorithm", v["spec_algo"], src_d, why)
        if str(v["spec_algo"]).upper() == "NEXTN":
            out["--speculative-num-steps"] = _entry("--speculative-num-steps", v["spec_steps"], src_d, "Kochbuch-Form (Qwen3.5.mdx:246)")
            out["--speculative-eagle-topk"] = _entry("--speculative-eagle-topk", v["spec_topk"], src_d, "lineare Kette, topk 1 (Qwen3.5.mdx:247)")
            out["--speculative-num-draft-tokens"] = _entry("--speculative-num-draft-tokens", v["spec_draft_tokens"], src_d, "Kochbuch-Form (Qwen3.5.mdx:248); Spekulations-Zwischenzustaende rechnen mit diesem D")
        else:
            out["--speculative-draft-model-path"] = _entry("--speculative-draft-model-path", v["spec_draft_path"], "%s (Draft-Profil)" % H_MODEL, "Pfad des Draft-Verzeichnisses")
            if v.get("spec_block"):
                out["--speculative-dflash-block-size"] = _entry("--speculative-dflash-block-size", v["spec_block"], "%s (dflash_config.block_size)" % H_MODEL, "Verifikationsfenster des Drafts")
    else:
        reason = ("kein Draft verfuegbar (MTP-Kopf/Draft-Profil fehlt)" if g["draft"] != "off" and not F["mtp_found"] and not F["ext"]
                  else ("Ziel: kein Draft" if g["draft"] == "off" else "Leiter: " + next((r["grund"] for r in relax if r["schritt"].startswith("Draft")), "nicht gewaehlt")))
        out["--speculative-algorithm"] = _entry("--speculative-algorithm", None, H_RULE, "kein Draft: " + reason, zustand=Z_PROPOSED)
    return [out[k] for k in _ORDER if k in out]


def _apply_overrides(entries: List[Dict[str, Any]], overrides: Mapping[str, Any], proposed: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Markiert die uebersteuerten Eintraege (``proposed`` = die vorgeschlagenen Werte je Flag); die Verdikte kommen aus der neu gerechneten Passung."""
    by = {e["flag"]: e for e in entries}
    for flag, val in overrides.items():
        if flag in by:
            e = by[flag]
            e["begruendung"] = "übersteuert (vorgeschlagen war %r): %s" % (proposed.get(flag), e["begruendung"])
            e["wert"], e["herkunft"], e["zustand"] = val, H_OVERRIDE, Z_OVERRIDDEN
        else:
            entries.append(_entry(flag, val, H_OVERRIDE, "übersteuert: dieses Flag kennt die Einzelkarten-Rechnung nicht; nur der ServerArgs-Parse prueft es",
                                  state=V_UNK, grund="nicht in der Passungsrechnung", zustand=Z_OVERRIDDEN))
    return entries


# ===========================================================================
# Oeffentliche Funktion
# ===========================================================================


def propose_single(model_profile: Mapping[str, Any], card: Mapping[str, Any], goals: Optional[Mapping[str, Any]] = None, *,
                   draft_profile: Optional[Mapping[str, Any]] = None, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Vorschlag der Form Einzelkarte aus ``model_profile`` (``flliper.model/1``) und ``card`` (:func:`normalize_card`).

    ``goals`` (alle optional): ``context_tokens`` (Standard min(Modell-Maximum, 131072)), ``kv_tokens`` (Standard = Kontext), ``seats`` (Standard 1),
    ``draft`` ``auto|on|off|nextn|external`` (Standard auto: MTP-Kopf bzw. Draft-Profil nutzen, wenn es passt), ``kv_dtype`` (``None`` = der Planer waehlt;
    ``auto`` = Modell-Dtype, ``fp8_e4m3`` ...), ``vision`` (Standard ``False``), ``host_ram_mib`` (HiCache-Host-Budget; ohne Angabe kein HiCache),
    ``chunked_prefill_size``, ``reserve_mib`` (gemessene Reserve statt der ServerArgs-Heuristik), ``disable_radix``,
    ``pre_load_free_mib`` (gemessenes ``pre_model_load_memory`` in MiB, 0 < Wert <= Karte; ohne Angabe Karte - ``CONTEXT_OVERHEAD_DEFAULT_MIB``, unbelegt).
    ``overrides``: ``{flag: wert}`` -- jeder Wert ist setzbar; die Passung wird mit den uebersteuerten Werten NEU gerechnet."""
    c = normalize_card(card)
    F = model_facts(model_profile, draft_profile)
    g = _defaults_goals(goals)
    v, aux, relax, acc = derive(F, c, g)
    entries = build_entries(F, c, g, v, aux, relax, acc)
    proposed = {e["flag"]: e["wert"] for e in entries}
    unb: List[str] = list(F["unbelegt"]) + list(acc["unbelegt"])
    if overrides:
        v2 = dict(v)
        for flag, val in overrides.items():
            key = _FLAG_KEYS.get(flag)
            if key:
                v2[key] = val
                if flag == "--enable-hierarchical-cache":
                    v2["hicache"] = bool(val)
        if "--speculative-algorithm" in overrides and not overrides["--speculative-algorithm"]:
            v2.update(spec_algo=None, spec_steps=None, spec_topk=None, spec_draft_tokens=None, spec_draft_path=None, spec_block=None)
        # Die Reserve folgt den uebersteuerten Werten in account(); der Bruchteil bleibt der vorgeschlagene, FIT-RESERVE zeigt eine Luecke.
        acc = account(F, c, v2, g)
        v = v2
        entries = build_entries(F, c, g, v, aux, relax, acc)
        entries = _apply_overrides(entries, overrides, proposed)
    borrowed = [k for k, lab in F["src"].items() if "unbelegt" in lab] + (["Karte"] if "unbelegt" in str(c.get("total_src")) + str(c.get("usable_src")) else [])
    if borrowed:
        for e in entries:
            if e["zustand"] == Z_PROPOSED and e["wert"] is not None and (e["flag"] in _FIT_FLAGS or e["flag"] in ("--kv-cache-dtype", "--hicache-ratio", "--speculative-algorithm") or "unbelegt" in e["herkunft"]):
                e["zustand"] = Z_UNBELEGT
                e["begruendung"] += " [geborgt, unbelegt: %s]" % ", ".join(borrowed)
    if acc["kontext_treiber_standard"]:
        # Der Kontext/Treiber-Posten ist ein geborgter Standardwert; alles, was am statischen Budget haengt, traegt das (nur wenn nicht schon uebersteuert).
        for e in entries:
            if e["flag"] in ("--mem-fraction-static", "--max-total-tokens", "--max-mamba-cache-size", "--hicache-ratio") and e["wert"] is not None:
                if e["zustand"] == Z_PROPOSED:
                    e["zustand"] = Z_UNBELEGT
                e["begruendung"] += " [Kontext + Treiber %s MiB angenommen, am Einzelserver unbelegt; goals pre_load_free_mib ersetzt ihn]" % _fmt(acc["kontext_treiber_mib"])
    max_ctx = max_context_fit(F, c, v, g)
    fit = {k: val for k, val in acc.items() if not k.startswith("_")}
    fit["max_context_tokens_fit"] = max_ctx
    fit["passt"] = bool(acc["passt"])
    nums = acc["_numbers"]
    if fit["passt"]:
        verdict = {"state": "passt", "art": VERDICT_ART,
                   "text": "passt: statisch %s von %s MiB (= fraction x frei vor dem Laden; Kontext/Treiber %s MiB%s; Gewichte %s, Draft %s, KV %s, Mamba %s + %s Spekulation), %s MiB frei; Reserve %s MiB; Kontext bis %s Token" % (
                       _fmt(fit["statisch_summe_mib"]), _fmt(fit["statisch_budget_mib"] or 0), _fmt(fit["kontext_treiber_mib"]),
                       " angenommen, unbelegt" if fit["kontext_treiber_standard"] else " (Ziel)", _fmt(nums["w_main"]), _fmt(nums["w_draft"]),
                       _fmt(nums["kv_main"] + nums["kv_draft"]), _fmt(nums["mamba_main"]), _fmt(nums["mamba_spec"]), _fmt(fit.get("frei_mib", 0.0)),
                       _fmt(fit["reserve_gegeben_mib"] or 0), max_ctx if max_ctx is not None else "?")}
    elif acc["unbelegt"] and not any(not ch["ok"] for ch in acc["checks"]):
        verdict = {"state": "unbelegt", "art": VERDICT_ART, "text": "nicht rechenbar: " + "; ".join(acc["unbelegt"])}
    else:
        failing = [ch["text"] for ch in acc["checks"] if not ch["ok"]]
        verdict = {"state": "passt nicht", "art": VERDICT_ART,
                   "text": "passt nicht: " + " | ".join(failing) + ("; Posten: Gewichte %s + Draft %s + KV %s + Mamba %s + Spekulation %s MiB; mit diesen Werten haelt der KV-Pool %s Token Kontext" % (
                       _fmt(nums["w_main"]), _fmt(nums["w_draft"]), _fmt(nums["kv_main"] + nums["kv_draft"]), _fmt(nums["mamba_main"]), _fmt(nums["mamba_spec"]),
                       max_ctx if max_ctx is not None else "?"))}
    if acc["kontext_treiber_standard"]:
        unb.append("CUDA-Kontext + Treiber %.0f MiB (Standardwert, Messung am Launcher-Format launcher.py:253): am Einzelserver unbelegt; goals['pre_load_free_mib'] = gemessenes pre_model_load_memory"
                   % acc["kontext_treiber_mib"])
    unb.append("ServerArgs.__post_init__ (Geraeteerkennung, Kompatibilitaets-Refusals) ist nicht gelaufen: braucht einen Beschleuniger; nur argparse-Parse (serverargs_parse)")
    if c["unified"]:
        unb.append("APU-Speichermodell (torch-Gesamtspeicher = adressierbare Decke) unbelegt am Rig")
    argv: List[str] = []
    for e in entries:
        if e["wert"] is None:
            continue
        argv.append(e["flag"])
        if e["wert"] is not True:
            argv.append(str(e["wert"]))
    return {"schema": SCHEMA, "form": FORM, "n_cards": 1, "verdikt_art": VERDICT_ART,
            "card": c, "ziele": {k: val for k, val in g.items()}, "modell": {"id": F["id"], "path": F["path"], "format": F["format"], "family": F["family"]},
            "flags": entries, "fit": fit, "verdikt": verdict, "relaxations": relax, "argv": argv, "parse": None, "unbelegt": unb}


# ===========================================================================
# Ausfuehrbarkeit: ServerArgs-Parse im Kindprozess
# ===========================================================================

_PARSE_SCRIPT = r'''
import argparse, json, sys
sys.path.insert(0, sys.argv[1])
batch = json.loads(sys.stdin.read())
def emit(obj):
    sys.stdout.write("\nFLLIPER_PARSE_JSON:" + json.dumps(obj) + "\n")
try:
    from sglang.srt.server_args import ServerArgs
except BaseException as e:
    emit({"available": False, "error": "%s: %s" % (type(e).__name__, e)}); sys.exit(0)
class _P(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)
res = []
for argv in batch:
    p = _P(prog="sglang serve")
    ServerArgs.add_cli_args(p)
    row = {"ok": True, "error": None, "flags": {}}
    try:
        ns = p.parse_args(argv)
        for tok in argv:
            if tok.startswith("--") and tok in p._option_string_actions:
                act = p._option_string_actions[tok]
                val = getattr(ns, act.dest, None)
                try:
                    json.dumps(val)
                except TypeError:
                    val = str(val)
                row["flags"][tok] = {"dest": act.dest, "value": val, "deprecated": type(act).__name__.startswith("Deprecated")}
    except BaseException as e:
        row = {"ok": False, "error": "%s: %s" % (type(e).__name__, e), "flags": {}}
    res.append(row)
emit({"available": True, "results": res})
'''


def _tree_python_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def serverargs_parse_batch(argvs: Sequence[Sequence[str]], *, tree_python: Optional[str] = None, python: Optional[str] = None,
                           timeout_s: float = 180.0) -> Dict[str, Any]:
    """argparse-Parse von ``ServerArgs.add_cli_args`` im KINDPROZESS (Import ~8 s, kein GPU-Zugriff: ``CUDA_VISIBLE_DEVICES`` leer).

    ``{"available": bool, "results": [{"ok", "error", "flags": {flag: {"dest","value","deprecated"}}}], "error"}``; ``available=False`` heisst
    "nicht geprueft" (sglang nicht importierbar, Timeout) -- nie "ok"."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    try:
        cp = subprocess.run([python or sys.executable, "-c", _PARSE_SCRIPT, tree_python or _tree_python_dir()], input=json.dumps([list(a) for a in argvs]),
                            capture_output=True, text=True, timeout=timeout_s, env=env)
    except (OSError, subprocess.SubprocessError) as e:
        return {"available": False, "error": "%s: %s" % (type(e).__name__, e), "results": []}
    for line in reversed(cp.stdout.splitlines()):
        if line.startswith("FLLIPER_PARSE_JSON:"):
            return json.loads(line[len("FLLIPER_PARSE_JSON:"):])
    return {"available": False, "error": "kein Parse-Ergebnis (rc=%s): %s" % (cp.returncode, (cp.stderr or "")[-300:]), "results": []}


def serverargs_parse(argv: Sequence[str], **kw: Any) -> Dict[str, Any]:
    """Ein einzelner Parse; ``{"available", "ok", "error", "flags"}``."""
    r = serverargs_parse_batch([argv], **kw)
    if not r.get("available") or not r.get("results"):
        return {"available": False, "ok": None, "error": r.get("error"), "flags": {}}
    row = r["results"][0]
    return {"available": True, "ok": row["ok"], "error": row["error"], "flags": row["flags"]}


def apply_parse(proposal: Dict[str, Any], parse: Mapping[str, Any]) -> Dict[str, Any]:
    """Traegt ein Parse-Ergebnis in die Verdikte ein: je Flag ``verdikt.parse`` = ok / Fehler / veraltet (Alias); ``proposal['parse']`` = Gesamtergebnis."""
    proposal["parse"] = {k: parse.get(k) for k in ("available", "ok", "error")}
    for e in proposal["flags"]:
        if e["wert"] is None:
            continue
        if not parse.get("available"):
            e["verdikt"]["parse"] = "nicht geprüft"
            continue
        info = (parse.get("flags") or {}).get(e["flag"])
        if parse.get("ok") and info:
            e["verdikt"]["parse"] = "veraltet (Alias)" if info.get("deprecated") else "ok"
        elif not parse.get("ok"):
            named = e["flag"] in str(parse.get("error") or "")
            e["verdikt"]["parse"] = "Fehler" if named else "nicht geprüft"
            if named:
                e["verdikt"].update(state=V_NO, code="PARSE", grund=str(parse.get("error")))
        else:
            e["verdikt"]["parse"] = "nicht geprüft"
    if parse.get("available") and parse.get("ok") is False:
        proposal["verdikt"] = {"state": "passt nicht", "art": VERDICT_ART + " + ServerArgs-Parse", "text": "ServerArgs-Parse verweigert: %s" % parse.get("error")}
    elif parse.get("available") and parse.get("ok"):
        proposal["verdikt"] = dict(proposal["verdikt"], art=VERDICT_ART + " + ServerArgs-Parse")
    return proposal


def check(proposal: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
    """Vorschlag + Parse in einem Aufruf (:func:`serverargs_parse`, dann :func:`apply_parse`)."""
    return apply_parse(proposal, serverargs_parse(proposal["argv"], **kw))


def format_text(proposal: Mapping[str, Any]) -> str:
    """Kurzer Text fuer Log/Issue: Verdikt, Posten, Flags."""
    lines = ["Einzelkarte %s: %s" % (proposal["card"]["name"], proposal["verdikt"]["text"]), "Posten (MiB):"]
    for p in proposal["fit"]["posten"]:
        lines.append("  %-42s %9.1f  %s" % (p["name"], p["mib"], p["herkunft"]))
    lines.append("Argumente: " + " ".join(proposal["argv"]))
    return "\n".join(lines)
