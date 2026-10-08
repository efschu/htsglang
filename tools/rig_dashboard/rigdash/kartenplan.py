"""Kartenplaner (Item 510): optimale Startkonfiguration für ein gewähltes Modell auf 1..6 gewählten Karten.

Quelle der Wahrheit ist UNSER Planer, nicht ein Nachbau:

  1. Das Urteil "geht / geht nicht" kommt aus den Planer-Funktionen card_identity.arch_gate / order_cards /
     uncalibrated_message und topology.plan_topology (kartenplan_gate), mit synthetischem Karteninventar,
     ohne GPU, ohne Launcher.  Die Meldungen sind die Originaltexte des Planers.
  2. Geht das Inventar durch, kommt der Plan aus der Planer-AUFZEICHNUNG des Referenz-Boots (kartenplan_data/):
     vram_plan.json, Budgetzeilen des Launchers, argv/env der Gruppen, Posten aus den Rank logs.  Dass der Planer
     diese Budgets heute noch genauso rechnet, steht je Record als ``planer_nachrechnung``
     (launcher.budgets_from_dc im Kindprozess, kartenplan_bridge).
  3. Geht es NICHT durch (andere Karten, andere Zahl), gibt es keinen Plan.  Es gibt die Gründe des Planers und
     eine klar benannte NÄHERUNG ("geschätzt"), die nichts startet.

Nichts hier startet einen Prozess, eine GPU-Arbeit oder den Launcher.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

from . import kartenplan_catalog as CAT
from . import kartenplan_gate as GATE
from . import kartenplan_transport as TR

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kartenplan_data")
SCHEMA = "kartenplan.record/1"
#: so viele Karten bietet die Seite an: dieselbe Grenze wie der Planer (weg2/topology.py MAX_CARDS_BAR1 = 8, MIN_CARDS = 2); was dort außerhalb liegt,
#: sagt der Planer selbst als HW-TOPOLOGY ab. test_max_cards_follows_the_planner hält beide Zahlen zusammen.
MAX_CARDS = 8


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def plan_id_ok(plan: dict) -> bool:
    """Wie weg2/vram_plan.compute_plan_id: sha256 über den Plan ohne plan_id (Veränderung der Aufzeichnung fällt auf)."""
    body = {k: v for k, v in plan.items() if k != "plan_id"}
    return plan.get("plan_id") == "sha256:" + hashlib.sha256(_canonical(body).encode()).hexdigest()


def load_record(record_id: str, data_dir: str = DATA_DIR) -> Optional[dict]:
    try:
        with open(os.path.join(data_dir, record_id + ".json")) as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    if rec.get("schema") != SCHEMA:
        raise ValueError("Record %s: schema %r unknown" % (record_id, rec.get("schema")))
    if rec.get("vram_plan") and not plan_id_ok(rec["vram_plan"]):
        rec["vram_plan_ok"] = False
    return rec
#: die Planer-Bäume, aus denen das Gate gelesen wird; ``KARTENPLAN_TREE`` (Pfad zu <baum>/python) überschreibt
TREE_CANDIDATES = (
    "/opt/rigdash/kartenplan/current/python",
    "/spinning/htsglang/.claude/worktrees/rigdash-zoom-1001/.wt-hwgen/python",
)

PHASE_LABEL = {"P": "P layout (prefill, PP3)", "D": "D layout (decode, TP3)"}


def _find_tree(explicit: Optional[str] = None) -> Optional[str]:
    for t in (explicit, os.environ.get("KARTENPLAN_TREE"), *TREE_CANDIDATES):
        if t and os.path.isfile(os.path.join(t, "sglang", "srt", "weg2", "card_identity.py")):
            return t
    return None


class Kartenplaner:
    def __init__(self, *, data_dir: str = DATA_DIR, tree: Optional[str] = None):
        self.data_dir = data_dir
        self.tree = _find_tree(tree)
        self._gate_mods = None
        self._rec_cache: Dict[str, dict] = {}

    # ------------------------------------------------------------------ Katalog
    def catalog(self) -> dict:
        profs = []
        for p in CAT.PROFILES:
            d = dict(p)
            rec = self._record(p["record"])
            d["has_record"] = rec is not None
            d["boot"] = {k: (rec or {}).get("boot", {}).get(k) for k in ("tag", "rev", "image", "boot_profile", "ipc_era")} if rec else None
            profs.append(d)
        return {"cards": CAT.catalog_public(), "cards_disabled": [c for c in CAT.catalog_public(True) if not c["enabled"]],
                "profiles": profs, "max_cards": MAX_CARDS,
                "pcie": {"gens": list(TR.GENS), "lanes": list(TR.LANES), "lane_gbs": TR.LANE_GBS},
                "rig_preset": {"cards": [
                    {"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 4, "rebar": False, "chipset": False}},
                    {"card": "rtx5090-32", "pcie": {"gen": 5, "lanes": 8, "rebar": False, "chipset": False}},
                    {"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 8, "rebar": False, "chipset": False}}],
                    "src": "Widths user-confirmed 17.08. (5090 x8, one 3080 x8, one 3080 x4, memory rig-interconnect-p2p); generations = maximum of the card, not read per slot on the rig (ASPM downclock at idle)"},
                "planner_tree": self.tree, "gate_ok": self.tree is not None}

    def _record(self, rid: str) -> Optional[dict]:
        if rid not in self._rec_cache:
            self._rec_cache[rid] = load_record(rid, self.data_dir)
        return self._rec_cache[rid]

    def _mods(self):
        if self._gate_mods is None:
            if not self.tree:
                raise GATE.GateUnavailable("no planner tree with card_identity.py found (KARTENPLAN_TREE or install_510.sh)")
            self._gate_mods = GATE.load_modules(self.tree)
        return self._gate_mods

    # ------------------------------------------------------------------ Anfrage
    def plan(self, req: dict) -> dict:
        prof = CAT.profile(req.get("profile"))
        if prof is None:
            raise ValueError("unknown profile %r (selectable: %s)" % (req.get("profile"), ", ".join(p["id"] for p in CAT.PROFILES)))
        raw_cards = req.get("cards") or []
        if not 1 <= len(raw_cards) <= MAX_CARDS:
            raise ValueError("select 1 to %d cards, not %d" % (MAX_CARDS, len(raw_cards)))
        host_patched = bool(req.get("host_patched", True))
        cards = []
        for i, rc in enumerate(raw_cards):
            e = CAT.card(rc.get("card"))
            if e is None:
                raise ValueError("Card %r not in the catalog" % rc.get("card"))
            link = TR.per_card_link(e, rc.get("pcie"))
            cards.append({"index": i, "entry": e, "label": CAT.label(e), "link": link,
                          "status": self._card_status(e, prof)})
        transport = TR.choose_transport([c["link"] for c in cards], [c["label"] for c in cards], host_patched=host_patched)
        gate = self._gate(cards)
        verdict = self._verdict(prof, cards, gate, transport)
        out = {"ok": True, "profile": {k: prof[k] for k in ("id", "label", "line", "format", "flip", "note")},
               "request": {"profile": prof["id"], "host_patched": host_patched,
                           "cards": [{"card": c["entry"]["id"], "pcie": c["link"]["slot"]} for c in cards]},
               "cards": [self._public_card(c) for c in cards], "transport": transport, "gate": gate, "verdict": verdict}
        rec = self._record(prof["record"])
        if rec is None:
            verdict["goes"] = False
            verdict["reasons"].append({"code": "KEIN-RECORD", "text": "There is no planner recording for this profile (kartenplan_data/%s.json)." % prof["record"],
                                       "source": "kartenplan_records"})
        elif verdict["goes"]:
            out["plan"] = self._plan_from_record(rec, prof, cards, gate, transport)
        if not verdict["goes"] and rec is not None:
            out["naeherung"] = self._naeherung(rec, prof, cards)
        out["alternatives"] = self._alternatives(prof, cards, gate)
        return out

    # ------------------------------------------------------------------ Karten
    @staticmethod
    def _card_status(e: dict, prof: dict) -> dict:
        if not e["enabled"]:
            return {"level": "gesperrt", "ok": False, "why": e["off_reason"]}
        return CAT.arch_status(e["cc"], prof["format"])

    @staticmethod
    def _public_card(c: dict) -> dict:
        e = c["entry"]
        return {"index": c["index"], "card": e["id"], "label": c["label"], "arch": e["arch"], "cc": e["cc"],
                "usable_mib": e["usable_mib"], "usable_src": e["usable_src"], "mem_bw_gbs": e["mem_bw_gbs"], "mem_bw_src": e["mem_bw_src"],
                "link": c["link"], "status": c["status"], "driver_reserved_mib": e["driver_reserved_mib"]}

    def _gate(self, cards: List[dict]) -> dict:
        rows = []
        for c in cards:
            e, l = c["entry"], c["link"]
            rows.append({"nvml_index": c["index"], "uuid": "synthetisch-%d" % c["index"], "name": e["nvml_name"],
                         "total_mib": e["usable_mib"], "cc": e["cc"], "bar1_total_mib": l["bar1_mib"],
                         "pcie_max_gen": l["effective"]["gen"], "pcie_max_width": l["effective"]["lanes"]})
        try:
            ci, tp = self._mods()
            g = GATE.gate(ci, tp, rows)
            g["available"] = True
            return g
        except GATE.GateUnavailable as exc:
            return {"available": False, "ok": False, "why": str(exc)}

    # ------------------------------------------------------------------ Urteil
    def _verdict(self, prof: dict, cards: List[dict], gate: dict, transport: dict) -> dict:
        reasons: List[dict] = []
        seen = {}
        for c in cards:
            if not c["status"]["ok"]:
                key = (c["entry"]["arch"], c["status"]["why"])
                if key in seen:
                    seen[key]["cards"].append(c["index"])
                    continue
                seen[key] = {"code": "ARCH-" + c["entry"]["arch"], "text": c["status"]["why"],
                             "source": "Catalog + HW gate (card_identity.arch_gate)", "cards": [c["index"]]}
                reasons.append(seen[key])
        for r in reasons:
            r["text"] = "%s (concerns card%s %s)" % (r["text"], "s" if len(r["cards"]) > 1 else "", ", ".join(str(i + 1) for i in r["cards"]))
        if gate.get("available"):
            archs = [pc for pc in gate["per_card"] if not pc["arch"]["ok"]]
            if archs:
                reasons.append({"code": "HW-ARCH", "text": archs[0]["arch"]["message"] + (
                    "  [same message for %d more card(s)]" % (len(archs) - 1) if len(archs) > 1 else ""),
                    "source": "weg2/card_identity.arch_gate", "cards": [a["nvml_index"] for a in archs]})
            if gate["count"] and gate["count"]["ok"] is False:
                reasons.append({"code": "HW-COUNT", "text": gate["count"]["message"], "source": "weg2/card_identity.order_cards"})
            if gate["topology"] and not gate["topology"]["ok"]:
                reasons.append({"code": "HW-TOPOLOGY", "text": gate["topology"]["message"], "source": "weg2/topology.plan_topology"})
            if gate["calibration"] and not gate["calibration"]["ok"]:
                reasons.append({"code": "HW-UNCALIBRATED", "text": gate["calibration"]["message"], "source": "weg2/card_identity.uncalibrated_message"})
        else:
            reasons.append({"code": "GATE-FEHLT", "text": gate.get("why", "Gate not available"), "source": "kartenplan_gate"})
        if transport["transport"] == "nccl":
            reasons.append({"code": "TRANSPORT-NCCL", "text": " ".join(transport["reasons"]),
                            "source": "kartenplan_transport (best form is barlink BAR1)"})
        goes = not reasons
        if goes:
            head = "Works: %s on %d cards, transport %s." % (prof["label"], len(cards), "barlink BAR1" if transport["transport"] == "bar1" else transport["transport"])
        else:
            head = "Does not work: %s." % reasons[0]["code"] if len(reasons) == 1 else \
                "Does not work: %s." % ", ".join(sorted({r["code"] for r in reasons}))
        return {"goes": goes, "headline": head, "reasons": reasons,
                "cards_ok": [bool(c["status"]["ok"]) for c in cards]}

    def _alternatives(self, prof: dict, cards: List[dict], gate: dict) -> List[dict]:
        out = []
        for p in CAT.PROFILES:
            if p["id"] == prof["id"]:
                continue
            bad = [c["label"] for c in cards if not CAT.arch_status(c["entry"]["cc"], p["format"])["ok"] or not c["entry"]["enabled"]]
            ok = gate.get("available") and gate.get("ok") and not bad
            out.append({"profile": p["id"], "label": p["label"], "goes": bool(ok),
                        "why": ("same gate verdict as above (inventory matches the recording)" if ok else
                                ("Card(s) not runnable: " + ", ".join(bad) if bad else "Planner gate refuses this inventory")),
                        "has_record": self._record(p["record"]) is not None})
        return out

    # ------------------------------------------------------------------ Plan aus der Aufzeichnung
    def _plan_from_record(self, rec: dict, prof: dict, cards: List[dict], gate: dict, transport: dict) -> dict:
        order = gate["order"]               # [{nvml_index, class}] in Planer-Reihenfolge (Ordinal 0 = größte Karte)
        ordinal_of = {o["nvml_index"]: i for i, o in enumerate(order)}
        by_ord = {ordinal_of[c["index"]]: c for c in cards}
        plan_cards = _plan_cards(rec)
        phases = {g: _phase_breakdown(rec, g, plan_cards, prof["id"].endswith("-dual")) for g in ("P", "D")}
        peak = _flip_peak(plan_cards, phases)
        flags = _flags(rec)
        ctx = _context(rec)
        dr = _docker_run(rec, prof, transport)
        bars = []
        for o in range(len(plan_cards)):
            cc = by_ord[o]
            bars.append({"ordinal": o, "card_label": cc["label"], "input_index": cc["index"], "total_mib": plan_cards[o]["total_mib"],
                         "P": phases["P"][o], "D": phases["D"][o], "peak": peak[o]})
        return {
            "source": {"record": rec["id"], "boot_tag": rec["boot"]["tag"], "rev": rec["boot"]["rev"], "image": rec["boot"].get("image"),
                       "boot_profile": rec["boot"]["boot_profile"], "ipc_era": rec["boot"]["ipc_era"],
                       "plan_id": (rec.get("vram_plan") or {}).get("plan_id"), "plan_pass": (rec.get("vram_plan") or {}).get("pass"),
                       "plan_id_ok": rec.get("vram_plan_ok"),
                       "nachrechnung": _nachrechnung_summary(rec),
                       "note": "Numbers = recording of the planner at the real boot on the reference rig; PCIe settings do not change them (flip prices are measured only for the rig, HW-GENERISCH K4)."},
            "einfach": {"docker_run": dr, "bars": bars, "context": ctx,
                        "legend": ["Weights", "Experts", "KV", "Mamba/State", "Draft", "Activation/graphs", "Sleep residue", "Driver", "Rest"]},
            "experte": {"flags": flags, "phases": phases, "peak": peak, "warnings": _warnings(rec, phases, transport),
                        "flip_note": ("The flip transition itself (exchange buffer, lane window, arena) is not booked as an item in the plan (flip_legs empty): unmeasured.  The peak below is the maximum of the two phases per card."),
                        "closure": (rec.get("vram_plan") or {}).get("closure", []),
                        "open": (rec.get("vram_plan") or {}).get("open", []),
                        "overrides": (rec.get("vram_plan") or {}).get("overrides", [])},
        }

    # ------------------------------------------------------------------ Näherung (kein Planer-Ergebnis)
    def _naeherung(self, rec: dict, prof: dict, cards: List[dict]) -> dict:
        """Eine NÄHERUNG für Inventare, die der Planer verweigert.  Rechnet nur mit gemessenen Posten des Referenz-Rigs
        (Rank logs), skaliert grob; jede Zeile ist 'geschätzt'.  Startet nichts, behauptet keinen Plan."""
        plan_cards = _plan_cards(rec)
        ref_non_budget = [pc["total_mib"] - pc["d_budget_mib"] for pc in plan_cards if pc.get("d_budget_mib")]
        if not ref_non_budget:
            return {"available": False, "why": "Reference record without D budgets"}
        fixed = round(sum(ref_non_budget) / len(ref_non_budget))
        usable = [c["entry"]["usable_mib"] for c in cards]
        budget_est = [max(0, u - fixed) for u in usable]
        notes = []
        posts = _reference_posts(rec)
        w = posts["weights_total_mib"]
        side = posts["side_per_rank_mib"] * len(cards)
        kv_room = sum(budget_est) - w - side
        cell = posts.get("kv_cell_bytes")
        tokens = int(kv_room * 1048576 / cell) if (cell and kv_room > 0) else None
        fits = kv_room > 0
        notes.append("Fixed deduction per card %d MiB = mean (card minus D budget) of the three reference cards, boot %s: estimated, not measured on cards of another size."
                     % (fixed, rec["boot"]["tag"]))
        notes.append("Weights %d MiB = sum of the measured load items (rank logs, D group) of the reference boot." % w
                     if posts["weights_src"] == "rank_log" else posts["weights_src"])
        ctx = (_context(rec).get("context_tokens") or 0)
        kv_for_ctx = round(ctx * cell / 1048576) if (cell and ctx) else None
        return {"available": True, "label": "Approximation (NOT a planner result, none of it is started)",
                "context_target_tokens": ctx or None, "kv_for_context_mib": kv_for_ctx,
                "remainder_after_context_mib": (kv_room - kv_for_ctx) if kv_for_ctx is not None else None,
                "fixed_deduction_mib": fixed, "per_card": [{"label": c["label"], "usable_mib": u, "budget_est_mib": b}
                                                           for c, u, b in zip(cards, usable, budget_est)],
                "weights_mib": w, "side_posts_mib": side, "kv_room_mib": kv_room, "kv_tokens_est": tokens,
                "fits": fits, "notes": notes + posts["notes"],
                "verdict": ("The weights fit by calculation (sum of the cards without fixed deduction ≥ weights + side items)." if fits else
                            "The weights do NOT fit by calculation: sum of the cards minus fixed deductions is smaller than weights + side items."),
                "unplannable": "A plan (PP/TP cut, experts, flags) is missing: the planner has measured records only for the reference inventory (HW-GENERISCH K1/K2) and the cut search for other card counts/mixes is not built (stage 2b, weeks)."}


# ====================================================================== Hilfsfunktionen auf Records
def _plan_cards(rec: dict) -> List[dict]:
    """Karten des Referenz-Boots in Planer-Ordinalen: total, carve, D-/P-Budget, Schlafreste."""
    plan = rec.get("vram_plan")
    cards = []
    if plan:
        for i, c in enumerate(plan["cards"]):
            cards.append({"ordinal": i, "uuid": c["uuid"], "name": c["class"], "total_mib": c["total_mib"], "carve_mib": c["driver_reserved_mib"],
                          "nvml": c["nvml"]})
        for g, grp in plan["groups"].items():
            ranks = sorted(grp["ranks"].items(), key=lambda kv: kv[0])
            for i, (_k, r) in enumerate(ranks):
                cards[i][g.lower() + "_budget_mib"] = r["budget_mib"]
        for g, vec in (plan.get("asleep") or {}).items():
            for c in cards:
                c["asleep_" + g] = vec.get(c["uuid"])
    else:
        bj = rec.get("boot_json") or {}
        for i, c in enumerate(bj.get("cards") or []):
            cards.append({"ordinal": i, "uuid": c["uuid"], "name": c["name"], "total_mib": c["total_mib"], "carve_mib": c["reserved_mib"],
                          "nvml": c["nvml_index"]})
        for g in ("P", "D"):
            bud = (rec.get("budgets_final") or {}).get(g) or (bj.get("budgets") or {}).get(g) or []
            for i, b in enumerate(bud):
                cards[i][g.lower() + "_budget_mib"] = b
        dcp, dce = bj.get("dc_measured_p") or {}, bj.get("dc_expect_d") or {}
        for c in cards:
            c["asleep_P"] = dcp.get(c["uuid"])
            c["asleep_D"] = dce.get(c["uuid"])
    return cards


def _seg(key, label, mib, src, kind="post"):
    return {"key": key, "label": label, "mib": int(round(mib)), "src": src, "kind": kind}


#: Ein-Satz-Erklärung je Posten-Schlüssel (Tooltip des VRAM-Balkens)
SEG_WHAT = {
    "weights": "Dense model weights of this group on this card (attention, norms, embeddings, dense MLP).",
    "runtime": "Runtime state next to the loaded weights (buffers, scales, metadata).",
    "experts": "MoE experts that stay on the card permanently.",
    "experts_lru": "Expert cache (LRU) that fills the otherwise free VRAM (basic law: free VRAM belongs to the experts).",
    "kv": "KV cache pool: keys and values of the running and the cached requests.",
    "state": "Mamba/GDN state pools: recurrent state per seat (no KV).",
    "draft": "Draft/MTP model for speculative decoding.",
    "graphs": "CUDA graphs (recorded decode/prefill runs) with their private memory pool.",
    "transient": "Activations and intermediate buffers during prefill/decode (peak value).",
    "act": "Activation reserve that the rank planner schedules for the prefill.",
    "free_in_budget": "Part of the rank budget that is not booked individually (allocator cache, workspaces, unused pool).",
    "carve": "Part of the card reserved by the driver (CUDA context), usable for nothing.",
    "asleep": "VRAM that the sleeping other group keeps on this card (sleep residue).",
    "corridor": "Corridor: measured activation peak that the planner keeps free so that peaks trigger no OOM.",
    "overshoot": "Measured awake excess over the plan, which the planner deducts in addition.",
    "awake_rest": "Awake residue: VRAM that the awake group occupies beyond its booked items (record).",
    "l15": "L1.5 cache on the card (item, trade against experts/KV).",
}
#: Reihenfolge der Posten INNERHALB eines Phasenblocks (stabil: gleicher Schlüssel behält die Planer-Reihenfolge)
SEG_ORDER = ["weights", "runtime", "experts", "experts_lru", "state", "draft", "kv", "graphs", "transient", "act",
             "corridor", "overshoot", "awake_rest", "l15", "free_in_budget"]
#: Blöcke von links nach rechts: gemeinsam (Treiber), P, D -- in der P-Zeile und der D-Zeile an derselben Stelle
PHASE_BLOCKS = ("gemeinsam", "P", "D")


def _seg_phase(key: str, group: str, other: str) -> str:
    if key == "carve":
        return "gemeinsam"
    return other if key == "asleep" else group


def _annotate_segments(segs: List[dict], group: str, dual: bool) -> List[dict]:
    """Posten mit Phase (P/D/gemeinsam), Herkunft und Erklärung versehen und je Phase zusammenhängend ordnen.

    Ein Balken ist die Karte in dem Zustand, in dem ``group`` wach ist: Block ``gemeinsam`` (Treiber), Block P, Block D.
    Der Block der wachen Gruppe trägt ihre Posten, der Block der anderen Gruppe ihren Schlafrest (im Dual-Profil sind
    beide wach).  Innerhalb eines Blocks nach SEG_ORDER, sonst in Planer-Reihenfolge."""
    other = "D" if group == "P" else "P"
    for s in segs:
        s["phase"] = _seg_phase(s["key"], group, other)
        measured = str(s["src"]).startswith("Rank log") or s["key"] == "carve"
        s["origin"] = "measured" if measured else "planner value"
        s["origin_note"] = ("NVML reservation of the card" if s["key"] == "carve" else ("Rank log of the reference boot" if measured
                            else "vram_plan or budget line of the launcher"))
        s["what"] = SEG_WHAT.get(s["key"], s["label"])
        if dual and s["key"] == "asleep":
            s["what"] += " In the dual profile this group is awake at the same time."
    order = {k: i for i, k in enumerate(SEG_ORDER)}
    idx = {id(s): i for i, s in enumerate(segs)}
    return sorted(segs, key=lambda s: (PHASE_BLOCKS.index(s["phase"]), order.get(s["key"], len(order)), idx[id(s)]))


def _phase_breakdown(rec: dict, group: str, plan_cards: List[dict], dual: bool = False) -> List[dict]:
    """Je Karte die Posten der Phase ``group`` (P = Prefill-Layout wach, D = Decode-Layout wach)."""
    other = "D" if group == "P" else "P"
    out = []
    plan = rec.get("vram_plan")
    ranks_plan = sorted(((plan or {}).get("groups", {}).get(group, {}).get("ranks") or {}).items()) if plan else []
    rk_posts = rec.get("rank_posts", {}).get(group, {})
    rank_keys = sorted(k for k in rk_posts)
    closure = {(c["card"], c["phase"]): c for c in (plan or {}).get("closure", [])} if plan else {}
    for i, pc in enumerate(plan_cards):
        segs: List[dict] = []
        notes: List[str] = []
        extras: dict = {}
        budget = pc.get(group.lower() + "_budget_mib")
        asleep = pc.get("asleep_" + other)
        carve = pc["carve_mib"]
        if group == "D" and plan:
            terms = (plan["budget_terms"].get("D") or next((v for k, v in plan["budget_terms"].items() if k.startswith("D(")), {})).get(pc["uuid"], {})
        else:
            terms = {}
        # --- Posten innerhalb des Rang-Budgets
        r = ranks_plan[i][1] if ranks_plan and i < len(ranks_plan) else None
        has_plan_posts = bool(r and (r.get("fixed") or r.get("elastic")))
        if has_plan_posts:
            f, e = r["fixed"], r["elastic"]
            if "weights" in f:
                segs.append(_seg("weights", "Weights (dense)", f["weights"], "vram_plan fixed.weights"))
            else:
                notes.append("Dense weights of this group are not shown in the plan (fixed); they are in the 'Rest in the rank budget'.")
            if f.get("experts_resident"):
                segs.append(_seg("experts", "Experts resident", f["experts_resident"], "vram_plan fixed.experts_resident"))
            if (e.get("experts_lru") or {}).get("mib"):
                segs.append(_seg("experts_lru", "Expert LRU (fills free VRAM)", e["experts_lru"]["mib"],
                                 "vram_plan elastic.experts_lru (%s rows)" % e["experts_lru"].get("rows")))
            elif (e.get("experts_lru") or {}).get("rows"):
                notes.append("Expert LRU: %s rows (MiB not shown in the plan)" % e["experts_lru"]["rows"])
            if f.get("state_pools"):
                segs.append(_seg("state", "Mamba/state pools", f["state_pools"], "vram_plan fixed.state_pools"))
            if f.get("draft"):
                segs.append(_seg("draft", "Draft", f["draft"], "vram_plan fixed.draft"))
            if (e.get("kv") or {}).get("mib"):
                segs.append(_seg("kv", ("KV (%s tokens)" % e["kv"]["tokens"]) if e["kv"].get("tokens") else "KV (pool)", e["kv"]["mib"], "vram_plan elastic.kv"))
            elif (e.get("kv") or {}).get("mib") == 0:
                pass
            for st, v in (r.get("transient_by_state") or {}).items():
                segs.append(_seg("transient", "Activation/transient (%s)" % st, v, "vram_plan transient_by_state", kind="transient"))
            cl = closure.get((pc["uuid"], "%s awake" % group))
            if cl:
                for k, v in cl["terms"].items():
                    notes.append("Closing %s: %s %d MiB" % (cl["phase"], k, v))
                notes.append("Closing rest %d MiB (rest_to=%s, bound_by=%s)" % (cl["rest_mib"], cl["rest_to"], cl["bound_by"] or "-"))
        elif i < len(rank_keys) and rk_posts[rank_keys[i]]:
            segs, extras = _rank_log_segments(rk_posts[rank_keys[i]], group)
        # --- Posten um das Budget herum (Terme der Planer-Budgetzeile)
        around = _outside_terms(rec, group, pc, terms, other)
        used_in_budget = sum(s["mib"] for s in segs)
        total = pc["total_mib"]
        bound = None
        free_in_budget = None
        if budget is not None:
            free_in_budget = budget - used_in_budget
            bound = {"budget_mib": budget, "used_in_budget_mib": used_in_budget, "free_in_budget_mib": free_in_budget,
                     "terms": {k: terms[k] for k in ("total", "carve", "floor", "dormant", "growth", "awake", "awake_source") if k in terms}}
        rest = total - used_in_budget - sum(s["mib"] for s in around) - (free_in_budget or 0)
        if free_in_budget is not None:
            res_mib = extras.get("reserve_mib") or 0
            if res_mib:
                take = max(0, min(res_mib, free_in_budget))
                segs = segs + [_seg("act", "Activation reserve (rank planner books %d)" % res_mib, take,
                                    "Rank planner 'prefill activation reserve'; is part of the rest of the budget", kind="reserve")]
                if take < res_mib:
                    notes.append("The rank planner reserved %d MiB of activation, only %d MiB remain in the budget after the measured items." % (res_mib, max(free_in_budget, 0)))
                free_in_budget_rest = free_in_budget - take
            else:
                free_in_budget_rest = free_in_budget
            segs = segs + [_seg("free_in_budget", "Rest in the rank budget (not booked individually)", free_in_budget_rest,
                                "Budget minus sum of the measured items (allocator cache, workspaces, unused pool)", kind="rest")]
            if extras.get("ctx_mib"):
                notes.append("CUDA context/NCCL init %d MiB (rank log 'Init torch distributed') lie before the rank budget." % extras["ctx_mib"])
        allsegs = _annotate_segments(segs + around, group, dual)
        # Summe der gezeichneten Posten gegen die Karte.  Ein negativer Rest in the rank budget (Posten > Budget, weil sie
        # überlappen bzw. in der anderen Phase gemessen wurden) ist keine Überfüllung: der Planer schließt die Karte
        # trotzdem (rest_mib ~ 0).  Der Überstand zeigt sich im Balken, wird aber getrennt als ``overlap_mib`` benannt.
        sum_mib = sum(max(0, x["mib"]) for x in allsegs)
        over = max(0, sum_mib - total)
        overlap = min(over, max(0, -(free_in_budget or 0)))
        out.append({"ordinal": pc["ordinal"], "segments": allsegs, "total_mib": total, "budget": bound,
                    "rest_mib": rest, "notes": notes, "group": group,
                    "sum_mib": sum_mib, "over_mib": over, "overlap_mib": overlap, "hard_over_mib": over - overlap})
    return out


def _outside_terms(rec: dict, group: str, pc: dict, terms: dict, other: str) -> List[dict]:
    """Die Posten außerhalb des Rang-Budgets, wie der Planer sie in der Budgetzeile abzieht (beim Bau in Zahlen zerlegt)."""
    lines = [ln for ln in rec.get("budget_lines", []) if ln["group"] == group and ln["label"] == group and ln["ordinal"] == pc["ordinal"]]
    out: List[dict] = []
    if lines:
        g = lines[-1]["parsed"]
        src = "Budget line of the launcher (front log)"
        if g["carve"]:
            out.append(_seg("carve", "Driver carve (NVML reserved)", g["carve"], src, kind="outside"))
        if g["dormant"] is not None:
            out.append(_seg("asleep", "Sleep residue %s group" % other + (" + growth %d" % g["growth"] if g["growth"] else ""),
                            g["dormant"] + (g["growth"] or 0), src, kind="outside"))
        if g["corridor"]:
            out.append(_seg("corridor", ("Corridor (floor %s, source %s, + reserve/excess 404)" % (g["floor"], g["floor_source"])
                                         if g["floor"] is not None else "Corridor"), g["corridor"], src, kind="outside"))
        if g["over"] and not g["booked"]:      # beim gebuchten Wach-Rest steht der Überschuss nur als "ersetzt" in der Zeile
            out.append(_seg("overshoot", "measured awake excess", g["over"], src, kind="outside"))
        if g["booked"]:
            out.append(_seg("awake_rest", "Awake residue (record D_AWAKE_REST_BOOKED)", g["booked"], src, kind="outside"))
        if g["awake_rest"]:
            out.append(_seg("awake_rest", "Awake residue (record D_AWAKE_REST_MIB)", g["awake_rest"], src, kind="outside"))
        if g["l15"]:
            out.append(_seg("l15", "L1.5 cache (item, trade against experts/KV)", g["l15"], src, kind="outside"))
        return out
    if terms:
        src = "vram_plan budget_terms"
        out.append(_seg("carve", "Driver carve (NVML reserved)", terms.get("carve", 0), src, kind="outside"))
        out.append(_seg("asleep", "Sleep residue %s group (incl. growth %s)" % (other, terms.get("growth", 0)), terms.get("dormant", 0), src, kind="outside"))
        out.append(_seg("corridor", "Corridor floor (measured activation peak)", terms.get("floor", 0), src, kind="outside"))
        out.append(_seg("awake_rest", "Awake residue/excess (%s)" % str(terms.get("awake_source", ""))[:40], terms.get("awake", 0), src, kind="outside"))
        return out
    out.append(_seg("carve", "Driver carve (NVML reserved)", pc["carve_mib"], "vram_plan cards.driver_reserved_mib", kind="outside"))
    if pc.get("asleep_" + other) is not None:
        out.append(_seg("asleep", "Sleep residue %s group" % other, pc["asleep_" + other], "vram_plan asleep", kind="outside"))
    return out


def _rank_log_segments(rp: dict, group: str):
    segs = []
    src = "Rank log (measured)"
    w = rp.get("weights") or []
    tgt = next((x for x in w if "Draft" not in x["type"] and "MTP" not in x["type"]), None)
    drf = next((x for x in w if "Draft" in x["type"] or "MTP" in x["type"]), None)
    kvp = rp.get("kv_posts") or {}
    posts = kvp.get("posts_mib") or {}
    wr = posts.get("weights + runtime state")
    if tgt:
        segs.append(_seg("weights", "Weights", tgt["mib"], src + " Load weight end"))
    if drf and drf["mib"]:
        segs.append(_seg("draft", "Draft", drf["mib"], src + " Load weight end (Draft)"))
    if wr is not None and tgt:
        runtime = wr - tgt["mib"] - (drf["mib"] if drf else 0)
        if runtime > 0:
            segs.append(_seg("runtime", "Runtime state", runtime, "Rank planner 'weights + runtime state' minus loaded weights"))
    if rp.get("mamba"):
        segs.append(_seg("state", "Mamba/state pool", rp["mamba"]["mib"], src + " Mamba Cache is allocated", kind="post"))
    kv = sum(p["k_mib"] + p["v_mib"] for p in rp.get("kv_pools") or [])
    if kv:
        segs.append(_seg("kv", "KV (%s)" % ((rp["kv_pools"][0]["dtype"] if rp["kv_pools"] else "").replace("torch.", "")), kv, src + " KV Cache is allocated (goal + draft)"))
    gr = sum(g["mib"] for g in rp.get("graphs") or []) or rp.get("prefill_graph_mib", 0)
    if gr:
        segs.append(_seg("graphs", "CUDA graphs", gr, src + " Capture ... CUDA graph end", kind="post"))
    return segs, {"reserve_mib": posts.get("prefill activation reserve"), "ctx_mib": rp.get("init_dist_mib")}


def _flip_peak(plan_cards: List[dict], phases: Dict[str, List[dict]]) -> List[dict]:
    out = []
    for i, pc in enumerate(plan_cards):
        used = {}
        for g in ("P", "D"):
            ph = phases[g][i]
            used[g] = ph["total_mib"] - ph["rest_mib"]
        peak_g = max(used, key=used.get)
        out.append({"ordinal": i, "used_P_mib": used["P"], "used_D_mib": used["D"], "peak_mib": used[peak_g], "peak_phase": peak_g,
                    "headroom_mib": pc["total_mib"] - used[peak_g], "total_mib": pc["total_mib"]})
    return out


def _context(rec: dict) -> dict:
    a = rec["launch"]["argv"]
    d, p = a.get("D") or [], a.get("P") or []

    def val(args, flag):
        return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None

    ctx = val(d, "--context-length")
    kv_tokens = None
    for k, v in (rec.get("rank_state") or {}).items():
        kt = (v.get("kv") or {}).get("kv_tokens")
        if k.startswith("D.") and kt:
            kv_tokens = kt
            break
    plan = rec.get("vram_plan") or {}
    seats_p = val(p, "--max-running-requests")
    for o in plan.get("overrides", []):
        if o["group"] == "P" and o["key"] == "--max-running-requests":
            seats_p = o["value"]
    return {"context_tokens": int(ctx) if ctx else None, "context_src": "argv D --context-length (state.json or rank log head)",
            "kv_pool_tokens": kv_tokens, "kv_pool_src": "rankstate D kv.kv_tokens (IPC)" if kv_tokens else None,
            "seats_d": int(val(d, "--max-running-requests") or 0) or None, "seats_p": int(seats_p) if seats_p else None,
            "seats_src": "argv D/P --max-running-requests"}


def _nachrechnung_summary(rec: dict) -> Optional[dict]:
    n = rec.get("planer_nachrechnung")
    if not n:
        return None
    return {"all_match": n.get("all_match"), "checked_utc": n.get("checked_utc"), "rev": n.get("rev"),
            "rows": [{"id": r["id"], "planer": r["got_mib"], "boot": r["expected_mib"], "match": r["match"]} for r in n.get("rows", [])]}


# ---------------------------------------------------------------------- Flags
#: Flags, deren Wert der Planer rechnet (nicht das Profil festlegt) -> Bindung
PLANNER_DERIVED = {
    "--rank-gpu-memory-mib": "Planner: budget per card = card - carve - sleep residue - awake residue/corridor (launcher.budgets_from_dc)",
    "--pp-stage-ratio": "Planner: PP cut solver (pp_cut) or profile pin; source see overrides",
    "--pp-attn-stage-ratio": "Planner: PP cut, attention layers per stage",
    "--rank-tp-ratio": "Planner: D-TP operating point (FRACTION-SOLVE)",
    "--rank-mlp-ratio": "Planner: D reshard preset (drq 652:218:218)",
    "--rank-moe-ratio": "Planner: D expert split per rank",
    "--rank-moe-resident-fraction": "Planner: experts fill free VRAM (basic law VRAM = experts)",
    "--dc-reserve": "Planner: sleep residue of the D group per card (W19 record)",
    "--measured-record": "Record file of the measured values",
}


def _flags(rec: dict) -> dict:
    docs = rec.get("flag_docs") or {}
    plan = rec.get("vram_plan") or {}
    ov = {}
    for o in plan.get("overrides", []):
        ov.setdefault(o["key"], []).append(o)

    def one(name, value, group):
        d = docs.get(name) or {}
        why = []
        if d.get("profile_comment"):
            why.append({"kind": "Profile", "text": d["profile_comment"]["text"], "source": d["profile_comment"]["source"]})
        if d.get("code_help"):
            why.append({"kind": "Code (argparse-help)", "text": d["code_help"], "source": rec["flag_docs_sources"]["code"]})
        bound = PLANNER_DERIVED.get(name)
        o = [x for x in ov.get(name, []) if x["group"] in (group, "launcher")]
        if o:
            bound = (bound + "; " if bound else "") + "set via " + ", ".join(sorted({x["source"] for x in o}))
        return {"name": name, "value": value, "why": why, "set_by": ("Planner" if name in PLANNER_DERIVED else ("Override" if o else "Profile")),
                "bound_by": bound or "Profile default (no planner term)", "explained": bool(why)}

    argv = rec["launch"]["argv"]
    res = {"groups": {}, "env": {}}
    for g in ("P", "D"):
        res["groups"][g] = _flag_list(argv.get(g) or [], lambda n, v, g=g: one(n, v, g))
    res["groups"]["front"] = _flag_list(rec["launch"].get("front_argv") or [], lambda n, v: one(n, v, "front"))
    for g in ("P", "D"):
        env = rec["launch"]["env"].get(g) or {}
        res["env"][g] = [{"name": k, "value": v, "why": ([{"kind": "Profile", "text": docs[k]["profile_comment"]["text"],
                                                              "source": docs[k]["profile_comment"]["source"]}] if k in docs and docs[k].get("profile_comment") else []),
                          "set_by": "Profile/launcher", "bound_by": "Profile default", "explained": k in docs and bool(docs[k])}
                         for k, v in sorted(env.items())]
    res["totals"] = {"flags": sum(len(v) for v in res["groups"].values()),
                     "explained": sum(1 for v in res["groups"].values() for f in v if f["explained"]),
                     "env": sum(len(v) for v in res["env"].values()),
                     "env_explained": sum(1 for v in res["env"].values() for f in v if f["explained"])}
    return res


def _flag_list(argv: List[str], mk) -> List[dict]:
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if "=" in a:
                n, v = a.split("=", 1)
                out.append(mk(n, v))
                i += 1
                continue
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out.append(mk(a, argv[i + 1]))
                i += 2
                continue
            out.append(mk(a, ""))
        i += 1
    return out


def _warnings(rec: dict, phases: Dict[str, List[dict]], transport: dict) -> List[str]:
    w = list(transport.get("warnings", []))
    plan = rec.get("vram_plan") or {}
    for o in plan.get("open", []):
        w.append("Planner open point: " + o)
    for c in plan.get("closure", []):
        if c.get("idle"):
            w.append("IDLE (planner): %s on nvml%s unused %d MiB (>= 1 expert row, no user goal binds)" % (c["phase"], "?", c["rest_mib"]))
    if not rec["boot"]["ipc_era"]:
        w.append("This record comes from a boot before the IPC state (26.09.): argv from the rank log head, no env, no vram_plan.")
    for g in ("P", "D"):
        for ph in phases[g]:
            if ph["rest_mib"] < 0:
                w.append("%s phase card %d: the measured items exceed the card by %d MiB (items overlap or were measured in another phase)"
                         % (g, ph["ordinal"], -ph["rest_mib"]))
    return w


# ---------------------------------------------------------------------- docker run
def _docker_run(rec: dict, prof: dict, transport: dict) -> dict:
    env = rec["launch"]["env"].get("D") or {}
    line = prof["line"]
    t = "bar1" if transport["transport"] == "bar1" else "nccl"
    pick = {k: env[k] for k in ("HTSGLANG_PROFILE", "HTSGLANG_TRANSPORT", "HTSGLANG_INSTRUMENTS", "HTSGLANG_MEMAVAIL_MIN_GIB",
                                "HTSGLANG_ALLOW_EXPERIMENTAL") if k in env}
    pick.setdefault("HTSGLANG_PROFILE", rec["boot"]["boot_profile"])
    pick["HTSGLANG_TRANSPORT"] = t
    shm = "48g" if line == "27b" else "16g"
    image = rec["boot"].get("image") or "<image>"
    tag = "dkr%s<name><MMDDHHMM>" % line
    lines = ["docker run -d --name htsglang-acc-%s-<name> -p 127.0.0.1:30030:30030 -e HTSGLANG_TAG=%s \\" % (line, tag),
             "  --gpus all --security-opt apparmor=unconfined -v /sys/devices:/sys/devices --shm-size=%s --ulimit memlock=-1:-1 --init \\" % shm,
             "  -v $MODELS:/spinning/llm_stuff/club-3090/models-cache:ro \\",
             "  -v $ACC/evidence:/var/lib/htsglang/evidence -v $ACC/arb:/var/lib/htsglang/arb -v $ACC/store:/var/lib/htsglang/hicache-weg2 \\",
             "  -v $ACC/sglang:/root/.cache/sglang -v $ACC/triton:/root/.triton \\",
             "  -e MODE=weg2 " + " ".join("-e %s=%s" % kv for kv in sorted(pick.items())) + " \\",
             "  --memory $CAP --memory-swap $CAP --oom-score-adj 500 --device /dev/dmabuf_holder \\"]
    if line == "nf":
        lines.append("  --mount type=tmpfs,dst=/mnt/nf-experts,tmpfs-size=77309411328,tmpfs-mode=1777 \\")
    lines.append("  %s serve" % image)
    host = ("CTX=<ctx> IMAGE=%s LINE=%s PROFILE=%s PROFILE_MOUNT=1 ALLOW_EXPERIMENTAL=1 HOUSE_GUARD=memlimit GPUQ_ID=<window> "
            "bash /spinning/gpu-arb/docker/host_acceptance.sh serve %s" % (image, line, pick["HTSGLANG_PROFILE"], t))
    return {"docker_run": "\n".join(lines), "host_line": host,
            "env": pick,
            "note": ("Derived from host_acceptance.sh run_args (structure) and the state.json of the reference boot (profile, image, env). $MODELS=<host>/spinning/llm_stuff/club-3090/models-cache, $ACC=<host>/spinning/docker-acceptance/%s, $CAP = house memory ceiling (host_acceptance.sh cap_for), SHM %s per line. It is started in a gpuq window, never from this page." % (line, shm)),
            "image": image}


# ---------------------------------------------------------------------- Näherung: Referenzposten
def _reference_posts(rec: dict) -> dict:
    notes = []
    rp = rec.get("rank_posts", {}).get("D", {})
    plan = rec.get("vram_plan") or {}
    ranks = plan.get("groups", {}).get("D", {}).get("ranks") or {}
    has_plan_posts = any(r.get("fixed") for r in ranks.values())
    if has_plan_posts:
        w = 0
        side = 0
        for r in ranks.values():
            w += r["fixed"].get("weights", 0) + r["fixed"].get("state_pools", 0)
            side += sum((r.get("transient_by_state") or {}).values())
        n = max(1, len(ranks))
        kv_cells = sum(d["kv_cell_bytes"] for d in ((plan.get("demand") or {}).get("D") or {}).get("ranks", {}).values()) or None
        notes.append("NF: experts live in host memory; dense weights and state pools are fixed on the cards, experts and KV share the rest (basic law VRAM = experts). 'Side items' = activation/transient from vram_plan.")
        return {"weights_total_mib": w, "weights_src": "vram_plan fixed.weights + fixed.state_pools (D group, sum)",
                "side_per_rank_mib": round(side / n), "kv_cell_bytes": kv_cells, "notes": notes}
    w = side = kvb = 0
    toks = 0
    for k, r in rp.items():
        for x in r.get("weights") or []:
            w += x["mib"]
        posts = (r.get("kv_posts") or {}).get("posts_mib") or {}
        side += (r.get("mamba") or {}).get("mib", 0) + sum(g["mib"] for g in r.get("graphs") or []) + posts.get("prefill activation reserve", 0) \
            + (r.get("init_dist_mib") or 0)
        for p in r.get("kv_pools") or []:
            kvb += p["k_mib"] + p["v_mib"]
    rs = rec.get("rank_state") or {}
    for k, v in rs.items():
        if k.startswith("D.") and (v.get("kv") or {}).get("kv_tokens"):
            toks = v["kv"]["kv_tokens"]
            break
    n = max(1, len(rp))
    cell = int(kvb * 1048576 / toks) if toks and kvb else None
    if cell:
        notes.append("KV cell %d B/token = measured KV pools (K+V) / pool tokens of the reference boot." % cell)
    return {"weights_total_mib": w, "weights_src": "rank_log", "side_per_rank_mib": round(side / n), "kv_cell_bytes": cell, "notes": notes}
