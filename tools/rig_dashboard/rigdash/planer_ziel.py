"""PLANER-ZIELBILD 1021: the planner page in the target picture of the user (21.10., verbatim):

  "man waehle karten und modell, danach kann man jenachdem was moeglich ist, sololayout (also nur TP), flip, und dual, auswaehlen. dann
   noch die max sitze und max token pro sitz. der rest muesste sich dann eigentlich automatisch ergeben/berechnen"
  "warum so viele profilnamen? das ist eh unsinnig so und verwirrt"

This module is the SERVER side of that page.  It computes nothing that the planner already computes; it sits on top of ``ProfilEditor.propose``
(the planner + the launcher dry run in the oracle child) and adds three things the old page left to the user:

1. **Input = model + cards (type, count).**  ``katalog()`` lists the models by NAME (the internal base profile of each model/layout stays in
   ``MODELS`` below, for the start scripts) and the cards of the card catalog with a count.
2. **Only possible layouts** (``layouts()``): solo (TP only), flip, dual.  Each layout carries a calculation line for what is impossible: a rough
   NECESSARY condition from the model fact sheet (``planer_data/steckbrief.json``, generated from the planner's own model profile) and the card
   data (``kartenplan_catalog``), the arch gate of the card catalog, the number of cards.  The planner's own computation decides afterwards
   (``plan()``): "possible" here means "not excluded by arithmetic", never "proven".
3. **Input afterwards = max seats + max tokens per seat only** (``plan()``).  The tokens per seat are a CEILING per request
   (``--max-kv-per-request``): the KV pool is shared by all seats, so seats x tokens is NOT a budget product (user 21.10.: "nicht jeder
   request braucht den vollen kv ... maximal pro request theoretisch 262k, aber nie alle gleichzeitig").  The rest (weight split, KV pool,
   expert rows, Mamba host MiB, chunk, X) is read out of the planner's proposal and the launcher's plan dump and shown with its source.

Hardware figures: when the chosen cards ARE the cards of this machine's hardware profile (``ProfilEditor._hw_cards_match``), the proposal runs on
the measured hardware ("measured"); otherwise on synthetic catalog cards and every figure is shown as an ESTIMATE until the hardware profile of
those cards exists.  Card counts other than 3 are honestly "not supported yet" (the parallel track NCARDS builds the topology/launcher for them;
the files of that track are not touched here).

NVFP4 runs on sm86 (3080/3090) with our own kernels (Marlin FP4 W4A16, W4A8 on the INT8 tensor cores; user 21.10.): the card gate below is
``kartenplan_catalog.arch_status`` (what the image is built for), never a datasheet "native FP4" flag.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from . import kartenplan_catalog as CAT

SCHEMA = "flliper.planer-ziel/1"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planer_data")
STECKBRIEF_FILE = os.path.join(DATA_DIR, "steckbrief.json")
MIB = float(1 << 20)

#: card counts the launcher/topology serve today (``topology.plan_topology``: N = 3 is the metal-proven shape); everything else is the track NCARDS
SUPPORTED_N = (3,)

#: the three layouts of the page.  ``form`` = the form name of ``ProfilEditor.propose``.  The sentences are the user's words made plain (see profil_planer.FORMS)
LAYOUTS = [
    {"id": "solo", "name": "Solo (TP only)", "form": "tp",
     "satz": "All cards work as one tensor-parallel group that prefills and decodes; no switching, no front."},
    {"id": "flip", "name": "Flip", "form": "flip",
     "satz": "The same cards work alternately as prefill group P (pipeline) and decode group D (tensor-parallel); the weights are exchanged through the host."},
    {"id": "dual", "name": "Dual", "form": "dual",
     "satz": "P and D are awake at the same time on the same cards; no switching, but both layouts hold weights at once."},
]
_LAYOUT_BY_ID = {l["id"]: l for l in LAYOUTS}

#: The models by NAME (what the user sees) and, per layout, the internal base profile that starts the planner (``None`` = no such layout for the model).
#: The profile names are internal: start scripts and ``FLLIPER_PROFILE=`` use them, the page never shows them.
MODELS = [
    {"id": "qwen38-27b-int8", "name": "Qwen3.8-27B INT8", "line": "27b", "format": "int8",
     "profiles": {"solo": "27b", "flip": "27b", "dual": None},
     "no_dual": "Dual is calibrated on the NVFP4 weights only (the launcher's dual form has no INT8 reference profile)."},
    {"id": "qwen38-27b-nvfp4", "name": "Qwen3.8-27B NVFP4", "line": "27b", "format": "nvfp4",
     "profiles": {"solo": "27b-nvfp4", "flip": "27b-nvfp4", "dual": "27b-nvfp4-dual"}},
    {"id": "qwen38-nf-int4", "name": "Qwen3.8-Flash-Next INT4", "line": "nf", "format": "int4",
     "profiles": {"solo": "nf-int4", "flip": "nf-int4", "dual": None},
     "no_dual": "Dual exists only in the 27B line (user decision 07.10.: the NF line has no dual form)."},
]
_MODEL_BY_ID = {m["id"]: m for m in MODELS}

#: Memory the planner keeps per card besides weights, KV and state (graphs, activations, user reserve, L1.5 posts, CUDA context).  DERIVED, not typed:
#: the dry run of 27B INT8 on 3 x RTX 3090 (24576 MiB) budgets the P group at 18584 MiB -> 24576 - 18584 = 5992 MiB are left outside (done/planer-zielbild-1021.md, section 3).
OVERHEAD_PER_CARD_MIB = 6000
#: defaults of the two inputs (the planner's own defaults, ``ProfilEditor.ZIELE_INT`` / the proposal's ``goals``)
DEFAULT_SEATS = 6
DEFAULT_TOKENS = 262144
GOALS = {"seats": [1, 256], "tokens": [1024, 8 << 20], "token_presets": [32768, 65536, 131072, 262144]}

#: ready-made card sets of the page ("our rig" is only offered when the machine has the cards; see ``katalog``)
SHORTCUTS = [
    {"id": "3x3090", "label": "3 x RTX 3090", "cards": [{"card": "rtx3090-24", "count": 3}]},
    {"id": "rig", "label": "Our rig (RTX 5090 + 2 x RTX 3080 20 GB)", "cards": [{"card": "rtx5090-32", "count": 1}, {"card": "rtx3080-20", "count": 2}]},
]


class ZielError(ValueError):
    """A request the page may not send (unknown model/layout/card, counts out of range): HTTP 400 with this text."""


def load_steckbrief(path: str = STECKBRIEF_FILE) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("schema") != "flliper.steckbrief/1":
        raise ValueError("steckbrief: schema %r unknown" % (doc.get("schema"),))
    return {m["id"]: m for m in doc["models"]}


def _short_name(name: str) -> str:
    for pre in ("NVIDIA ", "GeForce "):
        if name.startswith(pre):
            name = name[len(pre):]
    return name.replace("GeForce ", "")


def _fmt_gib(mib: float) -> str:
    return "%.1f GiB" % (mib / 1024.0)


class ZielPlaner:
    def __init__(self, editor, steckbrief: Optional[Dict[str, dict]] = None, steckbrief_file: str = STECKBRIEF_FILE):
        self.ed = editor
        self._steck = steckbrief
        self._steck_file = steckbrief_file

    @property
    def steck(self) -> Dict[str, dict]:
        if self._steck is None:
            self._steck = load_steckbrief(self._steck_file)
        return self._steck

    # ------------------------------------------------------------------ Eingaben pruefen
    def _model(self, mid) -> dict:
        m = _MODEL_BY_ID.get(str(mid))
        if m is None:
            raise ZielError("unknown model %r (known: %s)" % (mid, ", ".join(_MODEL_BY_ID)))
        return m

    def _layout(self, lid) -> dict:
        l = _LAYOUT_BY_ID.get(str(lid))
        if l is None:
            raise ZielError("unknown layout %r (solo, flip, dual)" % (lid,))
        return l

    def _cards(self, spec) -> List[dict]:
        """``[{card, count}]`` -> one catalog entry per physical card (order: as given)."""
        if not isinstance(spec, list) or not spec:
            raise ZielError("choose at least one card: cards = [{card, count}]")
        out: List[dict] = []
        for row in spec:
            if not isinstance(row, dict):
                raise ZielError("cards must be [{card, count}]")
            e = CAT.card(row.get("card"))
            if e is None or not e["enabled"]:
                raise ZielError("card %r is not in the card catalog" % (row.get("card"),))
            try:
                k = int(row.get("count", 1))
            except (TypeError, ValueError):
                raise ZielError("count of %s must be a whole number" % e["id"])
            if not 1 <= k <= 8:
                raise ZielError("count of %s must be between 1 and 8" % e["id"])
            out += [e] * k
        if len(out) > 8:
            raise ZielError("at most 8 cards")
        return out

    @staticmethod
    def _int(v, name: str, lo: int, hi: int) -> Optional[int]:
        if v in (None, ""):
            return None
        try:
            x = int(v)
        except (TypeError, ValueError):
            raise ZielError("%s must be a whole number" % name)
        if not lo <= x <= hi:
            raise ZielError("%s must be between %d and %d" % (name, lo, hi))
        return x

    # ------------------------------------------------------------------ Hardware (gemessen / geschaetzt)
    def _hw_cards(self) -> List[dict]:
        """The cards of THIS machine as catalog entries (hardware profile -> ``match_nvml``); [] when there is no hardware profile."""
        hw = self.ed._hardware_profile() if self.ed is not None else None
        out = []
        for c in (hw or {}).get("cards") or []:
            tot = c.get("vram_total_mib")
            tot = tot.get("v") if isinstance(tot, dict) else tot
            e = CAT.match_nvml(c.get("name"), tot, c.get("cc"))
            if e is None:
                return []                           # one unknown card: no honest "your machine" shortcut
            out.append(e)
        return out

    def _measured(self, cards: List[dict]) -> bool:
        hw = self.ed._hardware_profile() if self.ed is not None else None
        if hw is None:
            return False
        return bool(self.ed._hw_cards_match(hw, [{"entry": e} for e in cards]))

    def _hw_info(self, cards: List[dict]) -> dict:
        if self._measured(cards):
            return {"quelle": "gemessen", "label": "measured",
                    "text": "These are the cards of this machine: the hardware profile (NVML, measured rates) is used."}
        return {"quelle": "geschaetzt", "label": "estimated",
                "text": "No measured hardware profile for these cards yet: memory sizes and bandwidth are catalog (datasheet) values, rates are borrowed from the "
                        "nearest measured card. Every figure below is an estimate; the first boot on this hardware is the measurement."}

    # ------------------------------------------------------------------ Katalog
    def katalog(self) -> dict:
        mine = self._hw_cards()
        counts: Dict[str, int] = {}
        for e in mine:
            counts[e["id"]] = counts.get(e["id"], 0) + 1
        cards = [{"id": c["id"], "label": c["label"], "arch": c["arch"], "mib": c["usable_mib"], "preset": c["preset"], "origin": c["origin"],
                  "origin_label": c["origin_label"], "measured": bool(c.get("measured_on_rig"))} for c in CAT.catalog_public()]
        models = []
        for m in MODELS:
            models.append({"id": m["id"], "name": m["name"], "format": m["format"],
                           "layouts": [l["id"] for l in LAYOUTS if m["profiles"].get(l["id"])],
                           "weights_mib": self._weights_mib(m["id"])})
        return {"ok": True, "schema": SCHEMA, "models": models, "cards": cards, "layouts": [dict(l) for l in LAYOUTS], "supported_n": list(SUPPORTED_N),
                "machine": ({"cards": [{"card": k, "count": v} for k, v in counts.items()], "n": len(mine)} if mine else None),
                "shortcuts": [dict(x) for x in SHORTCUTS], "defaults": {"seats": DEFAULT_SEATS, "tokens": DEFAULT_TOKENS}, "goals": GOALS,
                "overhead_per_card_mib": OVERHEAD_PER_CARD_MIB}

    # ------------------------------------------------------------------ Layouts (Rechenbegruendung)
    def _weights_mib(self, mid: str) -> Optional[int]:
        s = self.steck.get(mid)
        return s.get("weights_total_mib") if s else None

    def _resident_weights_mib(self, mid: str, layout: str, n: int) -> dict:
        """Weights that sit in VRAM at once: ``total`` and the formula, per model/layout.

        * dense: every weight (the draft included) once across the cards -- solo and flip hold ONE layout's weights at a time (flip exchanges them through the host);
        * dual: D holds the TP shard (W/N of every layer), P holds its pipeline stage minus the part that is D's shard: (N-1)/N of W/N per card
          -> W/N + (N-1)/N * W/N per card = W * (2N-1)/N^2 per card = W * (2N-1)/N in total (``propose_dual`` module head);
        * MoE with experts in the host (NF): the non-expert layers + lm_head + draft in VRAM; the experts stay in the host (rows are loaned on demand)."""
        s = self.steck[mid]
        draft = s.get("draft_mib") or 0
        if s.get("experts"):
            base = (s["weights_dense_mib"] + s["lm_head_mib"]) + draft
            how = "non-expert layers %d + lm_head %d + draft %d MiB (the %d experts of every MoE layer stay in the host)" % (
                s["weights_dense_mib"], s["lm_head_mib"], draft, s["experts"]["n"])
        else:
            base = s["weights_total_mib"] + draft
            how = "weights %d + draft %d MiB" % (s["weights_total_mib"], draft)
        if layout == "dual":
            tot = base * (2 * n - 1) / float(n)
            how += " x (2N-1)/N = x %d/%d (D's tensor-parallel shard and P's pipeline stage are resident at once)" % (2 * n - 1, n)
        else:
            tot = float(base)
        return {"mib": tot, "how": how}

    def _calc(self, model: dict, layout: str, cards: List[dict], seats: int, tokens: int) -> dict:
        s = self.steck[model["id"]]
        n = len(cards)
        have = sum(c["usable_mib"] for c in cards)
        w = self._resident_weights_mib(model["id"], layout, n)
        kv = tokens * s["kv_bytes_per_token_fp8"] / MIB
        state = seats * s["state_mib_per_seat_bf16"]
        over = n * OVERHEAD_PER_CARD_MIB
        need = w["mib"] + kv + state + over
        lines = [
            "cards: %d card%s = %s of VRAM in total (%s)" % (n, "" if n == 1 else "s", _fmt_gib(have), " + ".join(sorted("%s %s" % (CAT.label(c), _fmt_gib(c["usable_mib"])) for c in cards))),
            "weights: %s = %s" % (w["how"], _fmt_gib(w["mib"])),
            "KV: ONE request at the ceiling must fit the shared pool: %d tokens x %d B (fp8, %d attention layers) = %s (never seats x tokens)" % (
                tokens, s["kv_bytes_per_token_fp8"], s["attn_layers"], _fmt_gib(kv)),
            "state: %d seats x %.1f MiB recurrent state (%d linear layers) = %s" % (seats, s["state_mib_per_seat_bf16"], s["linear_layers"], _fmt_gib(state)),
            "per card outside the pool: %d x %d MiB (graphs, activations, reserves; derived from the dry run) = %s" % (n, OVERHEAD_PER_CARD_MIB, _fmt_gib(over)),
            "needed %s against %s available: %s" % (_fmt_gib(need), _fmt_gib(have), ("margin %s" % _fmt_gib(have - need)) if have >= need else ("short by %s" % _fmt_gib(need - have))),
        ]
        small = [CAT.label(c) for c in cards if c["usable_mib"] <= OVERHEAD_PER_CARD_MIB + 1024]
        return {"need_mib": round(need), "have_mib": have, "margin_mib": round(have - need), "lines": lines, "too_small": small}

    def _gate(self, model: dict, cards: List[dict]) -> List[str]:
        """The arch gate of the image, per distinct card (``kartenplan_catalog.arch_status``): the reasons of the planner, not a datasheet flag."""
        seen: Dict[str, str] = {}
        for c in cards:
            st = CAT.arch_status(c["cc"], model["format"])
            if not st["ok"]:
                seen[CAT.label(c)] = st["why"]
        return ["%s: %s" % (k, v) for k, v in seen.items()]

    def layouts(self, body: dict) -> dict:
        model = self._model(body.get("model"))
        cards = self._cards(body.get("cards"))
        seats = self._int(body.get("seats"), "seats", *GOALS["seats"]) or DEFAULT_SEATS
        tokens = self._int(body.get("tokens"), "tokens", *GOALS["tokens"]) or DEFAULT_TOKENS
        n = len(cards)
        s = self.steck.get(model["id"])
        if s is None:
            raise ZielError("no model fact sheet for %s (planer_data/steckbrief.json)" % model["id"])
        gate = self._gate(model, cards)
        out = []
        for l in LAYOUTS:
            calc = self._calc(model, l["id"], cards, seats, tokens)
            why: List[str] = []
            status = "ok"
            profile = model["profiles"].get(l["id"])
            if n not in SUPPORTED_N:
                status = "n_unsupported"
                why.append("%d card%s: the launcher and topology serve exactly 3 cards today; other counts are being built (not supported yet)." % (n, "" if n == 1 else "s"))
            if gate:
                status = "blocked"
                why += gate
            if not profile:
                status = "blocked"
                why.append(model.get("no_dual") or "no reference profile for this layout")
            if calc["too_small"]:
                status = "blocked"
                why.append("too little memory on %s: %d MiB outside the pool alone are more than the card offers" % (", ".join(sorted(set(calc["too_small"]))), OVERHEAD_PER_CARD_MIB))
            if calc["margin_mib"] < 0:
                status = "blocked" if status in ("ok", "tight") else status
                why.append("does not fit by the rough calculation: short by %s" % _fmt_gib(-calc["margin_mib"]))
            elif status == "ok" and calc["margin_mib"] < 0.05 * calc["have_mib"]:
                status = "tight"
                why.append("tight: margin %s (%.0f%% of the cards' memory)" % (_fmt_gib(calc["margin_mib"]), 100.0 * calc["margin_mib"] / calc["have_mib"]))
            possible = status in ("ok", "tight")
            out.append({"id": l["id"], "name": l["name"], "satz": l["satz"], "status": status, "possible": possible, "why": why, "calc": calc["lines"],
                        "margin_mib": calc["margin_mib"]})
        return {"ok": True, "schema": SCHEMA, "model": {"id": model["id"], "name": model["name"]}, "n": n, "layouts": out,
                "hw": self._hw_info(cards), "seats": seats, "tokens": tokens,
                "note": "A rough necessary condition from the model fact sheet and the card data. The planner's computation (next step) decides; "
                        "a layout shown as possible can still be refused there, a blocked one cannot be forced here."}

    # ------------------------------------------------------------------ Vorschlag
    def plan(self, body: dict) -> dict:
        lay = self.layouts(body)
        model = self._model(body.get("model"))
        layout = self._layout(body.get("layout"))
        li = next(x for x in lay["layouts"] if x["id"] == layout["id"])
        cards = self._cards(body.get("cards"))
        base = {"ok": True, "schema": SCHEMA, "model": lay["model"], "layout": {"id": layout["id"], "name": layout["name"]}, "n": lay["n"], "hw": lay["hw"],
                "layout_info": li}
        if li["status"] in ("n_unsupported", "blocked"):
            return dict(base, status=li["status"], felder=[], verdikt=None, hinweise=li["why"])
        seats = self._int(body.get("seats"), "seats", *GOALS["seats"])
        tokens = self._int(body.get("tokens"), "tokens", *GOALS["tokens"])
        goals: Dict[str, Any] = {}
        if seats is not None:
            goals["seats"] = seats
        if tokens is not None:
            goals["kv_tokens"] = tokens
        req = {"basis": {"kind": "release", "name": model["profiles"][layout["id"]]}, "form": layout["form"],
               "inventory": [{"card": e["id"], "pcie": {"gen": e["pcie_native"]["gen"], "lanes": e["pcie_native"]["lanes"]}} for e in cards],
               "goals": goals}
        res = self.ed.propose(req)
        out = dict(base, status="ok" if res.get("ok") else "error")
        if not res.get("ok"):
            return dict(out, felder=[], verdikt=None, hinweise=[res.get("error") or "the planner gave no proposal"] + list(res.get("notes") or []))
        out.update(self._summary(model, layout["id"], cards, res, seats, tokens))
        out["intern"] = {"basis": model["profiles"][layout["id"]], "form": layout["form"], "oracle": res.get("oracle"),
                         "startprofil": {"doc": (res.get("startprofil") or {}).get("doc"), "name": (res.get("startprofil") or {}).get("name")}}
        return out

    # ------------------------------------------------------------------ Zusammenfassung (Rest berechnet)
    @staticmethod
    def _csv(v) -> Optional[List[str]]:
        if v in (None, ""):
            return None
        parts = [x.strip() for x in str(v).split(",")]
        return parts if len(parts) >= 2 else None

    def _summary(self, model: dict, layout: str, cards: List[dict], res: dict, seats, tokens) -> dict:
        s = self.steck[model["id"]]
        prop = res.get("proposal") or {}
        vd = res.get("verdict") or {}
        plan = vd.get("plan") or {}
        gf = plan.get("group_flags") or {}
        P, D, F = gf.get("P") or {}, gf.get("D") or {}, gf.get("front") or {}
        pc = plan.get("pp_cut") or {}
        names = [c.get("name") or "" for c in prop.get("cards") or []]
        short = [_short_name(x) for x in names]
        vals = {w["label"]: w for w in res.get("values") or []}
        goals = prop.get("goals") or {}
        has_p = layout in ("flip", "dual")
        rows: List[dict] = []

        def row(group: str, key: str, label: str, value=None, per_card=None, unit="", source="launcher dry run", why="", estimate=False, raw=""):
            if value in (None, "") and not per_card:
                return
            rows.append({"gruppe": group, "key": key, "label": label, "wert": None if value is None else str(value), "je_karte": per_card, "einheit": unit,
                         "quelle": source, "warum": why, "schaetzung": bool(estimate), "roh": raw})

        def pv(label):                                  # the planner's own value with its reason, when it proposed one
            w = vals.get(label)
            return (w.get("reason") or "") if w else ""

        # --- Seats and ceiling (the two inputs, echoed as the planner resolved them)
        row("seats", "seats", "Max concurrent seats (decode group)", goals.get("seats"), source="your input / planner default",
            why="--max-running-requests of D and --d-bs of the front", raw="goals.seats")
        row("seats", "tokens", "Max tokens per seat (ceiling for ONE request)", goals.get("kv_tokens"), source="your input / planner default",
            why="--max-kv-per-request: a ceiling per request; the KV pool is shared by all seats, so this is not multiplied by the seats", raw="--max-kv-per-request")
        # --- Weight split
        if has_p:
            row("weights", "p_layers", "Prefill split: layers per card (group P, pipeline)", pc.get("layers"), per_card=self._csv(pc.get("layers")), unit="layers",
                why="solved by the launcher's cut solver from the cards' rates and memory; attention layers per stage: %s" % (pc.get("attn") or "-"), raw="plan.pp_cut.layers")
        tp = D.get("--rank-tp-ratio")
        if self._csv(tp):
            row("weights", "d_split", "Decode split: share of the dense weights per card (group D)", per_card=self._csv(tp), source="launcher dry run",
                why="--rank-tp-ratio: the ratio of the dense (non-expert) weights each card holds; 1,0,0 = rank 0 (host role) holds them all, the experts are spread by the MoE ratio",
                raw="--rank-tp-ratio")
        else:
            row("weights", "d_split", "Decode split (group D, tensor-parallel)", tp, source="launcher dry run",
                why="'auto' = the launcher shares every layer across the %d cards by their decode-group memory budgets" % len(cards), raw="--rank-tp-ratio")
        if has_p:
            row("weights", "p_mem", "Memory budget per card, prefill group", per_card=self._csv(P.get("--rank-gpu-memory-mib")) or ([P.get("--rank-gpu-memory-mib")] if P.get("--rank-gpu-memory-mib") else None),
                unit="MiB", why="--rank-gpu-memory-mib of group P (what the pipeline stage may use on the card)", raw="--rank-gpu-memory-mib (P)")
        row("weights", "d_mem", "Memory budget per card, decode group", per_card=self._csv(D.get("--rank-gpu-memory-mib")) or ([D.get("--rank-gpu-memory-mib")] if D.get("--rank-gpu-memory-mib") else None),
            unit="MiB", why="--rank-gpu-memory-mib of group D (weights shard + KV pool + state of the tensor-parallel group)", raw="--rank-gpu-memory-mib (D)")
        # --- KV pool
        if has_p and pc.get("chosen_pool"):
            row("kv", "p_pool", "KV pool, prefill group (shared by all seats)", pc["chosen_pool"], unit="tokens",
                why="solved by the launcher" + ("; floor %s = one request at the ceiling + one chunk" % pc["pool_floor"] if pc.get("pool_floor") else ""), raw="plan.pp_cut.chosen_pool")
        d_est = self._d_pool_estimate(s, layout, cards, D, seats if seats else goals.get("seats"))
        if d_est:
            row("kv", "d_pool", "KV pool, decode group (shared by all seats)", d_est["tokens"], unit="tokens", source="estimate", estimate=True,
                why=d_est["why"], raw="D budgets - weights - state, / KV bytes per token")
        # --- Experts (MoE only)
        if s.get("experts"):
            ex = s["experts"]
            row("experts", "expert_total", "Experts of the model (in the host)", "%d per layer x %d layers, top-%d" % (ex["n"], ex["moe_layers"], ex["top_k"]),
                source="model fact sheet", why="the expert weights (%s) stay in host memory; VRAM holds the non-expert layers and a row cache" % _fmt_gib(ex["total_mib"]))
            for lab, flag in (("Expert share per card (MoE ratio, group D)", "--rank-moe-ratio"), ("Expert rows resident per card (device cache)", "--pp-cut-expert-lru-rows"), ("Resident expert fraction per card", "--rank-moe-resident-fraction"),
                              ("Expert share on the device per card", "--pp-cut-expert-device-fraction")):
                self._flag_row(rows, "experts", flag, lab, res, P, D, pv)
            env_all = dict((res.get("launch") or {}).get("env") or {})
            for lab, name in (("Expert scratch slots per card", "FLLIPER_MOE_SCRATCH_SLOTS"), ("Expert residency (env)", "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION")):
                if env_all.get(name):
                    row("experts", name, lab, per_card=self._csv(env_all[name]) or None, value=None if self._csv(env_all[name]) else env_all[name], source="launcher dry run", raw=name)
        # --- Host memory
        row("host", "mamba_host", "Mamba (recurrent state) host memory", D.get("--hicache-mamba-host-mib") or P.get("--hicache-mamba-host-mib"), unit="MiB",
            source="launcher dry run", why="--hicache-mamba-host-mib: where the recurrent state of parked/evicted seats lives in the host", raw="--hicache-mamba-host-mib")
        hc = ", ".join("%s %s GiB" % (g, f["--hicache-size"]) for g, f in (("P", P), ("D", D)) if f.get("--hicache-size"))
        row("host", "host_cache", "Host KV cache (L2) per group", hc or None, source="launcher dry run",
            why="--hicache-size of the groups: the host tier that catches the KV of parked seats", raw="--hicache-size")
        # --- Chunk and X
        row("chunk", "chunk_p", "Prefill chunk (group P)", P.get("--chunked-prefill-size"), unit="tokens", why="--chunked-prefill-size of P", raw="--chunked-prefill-size (P)")
        row("chunk", "chunk_d", "Prefill chunk (group D)", D.get("--chunked-prefill-size"), unit="tokens", why="--chunked-prefill-size of D", raw="--chunked-prefill-size (D)")
        if layout == "flip":
            x = F.get("--x-ceiling-tokens") or D.get("--tp-prefill-max-tokens")
            row("chunk", "x", "X: longest prefill the decode group takes itself", x, unit="tokens", source="planner",
                why=pv("--x-ceiling-tokens") or "--x-ceiling-tokens: requests up to X tokens are prefilled on D, longer ones go to P", raw="--x-ceiling-tokens")

        groups = []
        titles = {"seats": "Your inputs", "weights": "Weight split and memory per card", "kv": "KV pool", "experts": "Experts", "host": "Host memory", "chunk": "Chunk and X"}
        for gid in ("seats", "weights", "kv", "experts", "host", "chunk"):
            zs = [r for r in rows if r["gruppe"] == gid]
            if zs:
                groups.append({"id": gid, "titel": titles[gid], "zeilen": zs})
        cardinfo = [{"name": short[i] if i < len(short) else "", "mib": (prop.get("cards") or [{}] * (i + 1))[i].get("total_mib"),
                     "rate": (prop.get("cards") or [{}] * (i + 1))[i].get("tflops_src")} for i in range(len(prop.get("cards") or []))]
        return {"felder": groups, "defaults": {"seats": goals.get("seats"), "tokens": goals.get("kv_tokens")}, "goals": goals, "karten": cardinfo,
                "verdikt": self._verdict(vd, prop), "hinweise": [str(x) for x in (prop.get("notes") or []) + (res.get("notes") or []) if x]}

    def _flag_row(self, rows, group, flag, label, res, P, D, pv) -> None:
        v = D.get(flag) or P.get(flag)
        if v in (None, ""):
            for w in res.get("values") or []:
                if w.get("label") == flag and w.get("value") not in (None, ""):
                    v = w["value"]
        if v in (None, ""):
            return
        per = self._csv(v)
        rows.append({"gruppe": group, "key": flag, "label": label, "wert": None if per else str(v), "je_karte": per, "einheit": "", "quelle": "planner / launcher",
                     "warum": pv(flag), "schaetzung": False, "roh": flag})

    @staticmethod
    def _d_pool_estimate(s: dict, layout: str, cards: List[dict], D: dict, seats) -> Optional[dict]:
        """Decode KV pool in tokens: what the D budgets leave after the weights and the state.  An ESTIMATE (the launcher sizes the pool at boot)."""
        budgets = D.get("--rank-gpu-memory-mib")
        if not budgets:
            return None
        try:
            tot = sum(int(x) for x in str(budgets).split(","))
        except ValueError:
            return None
        n = len(cards)
        draft = s.get("draft_mib") or 0
        w = (s["weights_dense_mib"] + s["lm_head_mib"] + draft) if s.get("experts") else (s["weights_total_mib"] + draft)
        state = (seats or DEFAULT_SEATS) * s["state_mib_per_seat_bf16"]
        free = tot - w - state
        if free <= 0:
            return None
        toks = int(free * MIB / s["kv_bytes_per_token_fp8"])
        return {"tokens": toks, "why": "(D budgets %d MiB - weights %d MiB - state %.0f MiB) / %d B per token; an estimate, the launcher sizes the pool at boot" % (
            tot, w, state, s["kv_bytes_per_token_fp8"])}

    @staticmethod
    def _verdict(vd: dict, prop: dict) -> dict:
        outcome = vd.get("outcome")
        items: List[dict] = []
        seen: Dict[str, dict] = {}
        for x in vd.get("verdikte") or []:
            if x.get("level") in ("run", "crash", "oracle", "blocker", "planer"):
                blocked = x.get("forcebar") is False or x.get("force_state") == "is_blocked" or x.get("level") == "crash"
                code = x.get("code")
                if code in seen:
                    seen[code]["count"] += 1          # the launcher repeats a code per card/rank: one line with a count
                    continue
                seen[code] = {"code": code, "text": (x.get("reason") or x.get("title") or "")[:300], "blocked": bool(blocked), "force": bool(x.get("forcebar")), "count": 1}
                items.append(seen[code])
        text = {"ok": "The launcher's dry run passes without force.",
                "ok_with_force": "The launcher's dry run passes only with force: it does not trust numbers that were not measured on these cards (see the notes). "
                                 "The first boot is the measurement.",
                "verweigert": "The launcher refuses this combination (see the reasons).", "crash": "The launcher's dry run failed (see the reasons)."}.get(outcome, "No launcher verdict (%s)." % (outcome,))
        fit = prop.get("fit") or {}
        return {"outcome": outcome, "text": text, "items": items, "fit": {"level": fit.get("level"), "margin_mib": fit.get("margin_mib")}}
