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
import re
from typing import Any, Dict, List, Optional

from . import kartenplan_catalog as CAT

SCHEMA = "flliper.planer-ziel/1"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planer_data")
STECKBRIEF_FILE = os.path.join(DATA_DIR, "steckbrief.json")
DE_FILE = os.path.join(DATA_DIR, "planer_ziel_de.json")
TM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "i18n_de.json")
MIB = float(1 << 20)

#: The card counts the planning core is CHECKED for (user 22.10.: "2 karten und 4 karten, und 5 karten und 6 karten und 7 karten gehen"): NCARDS 1021 ships 12 release
#: profiles ``{27b, 27b-nvfp4, 27b-nvfp4-dual, nf}-{2,3,4}x3090`` with a launcher dry-run golden each, release 1022 the same for 5, 6 and 7 (the NCARDS goldens ``fixtures/ncards_1021/golden`` of the launcher tests; pinned by ``test_planer_release_1022``).  Fewer (one card: nothing to flip between) or
#: more (5..8: no profile, no golden) are not calculated here and say why.  Also the FALLBACK when the topology cannot be asked (no editor / no planner gate).
SUPPORTED_N = (2, 3, 4, 5, 6, 7)
#: The counts a release boot has PROVEN on metal (``PROVEN_CARD_COUNTS`` of the topology module).  The other counts of ``SUPPORTED_N`` are calculated by the launcher dry run
#: but carry the named blocker METAL-UNPROVEN (the launcher accepts them only with force; the first boot is the proof) -- the page shows that, it never hides it.
PROVEN_N = (3,)

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
     "profiles": {"solo": "27b", "flip": "27b", "dual": None}, "ncards": {"solo": "27b", "flip": "27b"},
     "no_dual": "Dual is calibrated on the NVFP4 weights only (the launcher's dual form has no INT8 reference profile)."},
    {"id": "qwen38-27b-nvfp4", "name": "Qwen3.8-27B NVFP4", "line": "27b", "format": "nvfp4",
     "profiles": {"solo": "27b-nvfp4", "flip": "27b-nvfp4", "dual": "27b-nvfp4-dual"}, "ncards": {"solo": "27b-nvfp4", "flip": "27b-nvfp4", "dual": "27b-nvfp4-dual"}},
    {"id": "qwen38-nf-int4", "name": "Qwen3.8-Flash-Next INT4", "line": "nf", "format": "int4",
     "profiles": {"solo": "nf-int4", "flip": "nf-int4", "dual": None}, "ncards": {"solo": "nf", "flip": "nf"},
     "no_dual": "Dual exists only in the 27B line (user decision 07.10.: the NF line has no dual form)."},
]
_MODEL_BY_ID = {m["id"]: m for m in MODELS}

#: Memory per card outside weights, KV and state that a NECESSARY condition may take as a LOWER bound (graphs, activations, CUDA context).  DERIVED, not typed:
#: the dry run of 27B INT8 on 3 x RTX 3090 (24576 MiB) leaves 24576 - 19792 = 4784 MiB per card outside the D group budget (the P group leaves 5992..7784, but
#: that residue contains the user reserve and the L1.5 post; a necessary condition takes the SMALLEST residue of the same run and never a reserve; review A6/A6b,
#: done/planer-zielbild-1021-review.md).  What the launcher really keeps is the planner's answer, never this pre-check's.
OVERHEAD_PER_CARD_MIB = 4784
#: Fixed D-group posts per card that a pool ESTIMATE subtracts besides weights and state (graphs, activations, draft KV, L1.5 posts).  CALIBRATED on the one measured
#: pool of our rig, 27B INT8 flip, boot dkr27bbar1drei10210113 (21.10. 01:13Z, D log max_total_num_tokens 245185 + 252097 + 281698 = 778980 tokens):
#: D budgets 27792 + 17384 + 17168 = 62344 MiB - weights 28179 - draft 2072 - state 6 x 74.8 - pool 778980 x 34816 B (25866 MiB) = 5778 MiB = 3 x 1926 MiB.
#: Only used where the launcher sets no pool flag (27B flip/solo); a pool flag in the argv always wins (review A1/A4).
D_FIXED_PER_CARD_MIB = 1926
#: defaults of the two inputs (the planner's own defaults, ``ProfilEditor.ZIELE_INT`` / the proposal's ``goals``)
DEFAULT_SEATS = 6
DEFAULT_TOKENS = 262144
GOALS = {"seats": [1, 256], "tokens": [1024, 8 << 20], "token_presets": [32768, 65536, 131072, 262144]}

#: ready-made card sets of the page.  "Reference rig" is the rig of the developers (the cards of the golden measurements), offered on every machine by that name;
#: "This machine" (``katalog()["machine"]``) is the hardware profile of THIS machine and only offered when there is one (review B1)
SHORTCUTS = [
    {"id": "3x3090", "label": "3 x RTX 3090", "cards": [{"card": "rtx3090-24", "count": 3}]},
    {"id": "rig", "label": "Reference rig (RTX 5090 + 2 x RTX 3080 20 GB)", "cards": [{"card": "rtx5090-32", "count": 1}, {"card": "rtx3080-20", "count": 2}]},
]


# ---------------------------------------------------------------------------------------------------------------------------- German (review B2)
_DE_RULES: Optional[list] = None


def _de_rules() -> list:
    """The German texts as (exact dict, [(regex, template)]): ``planer_data/planer_ziel_de.json`` first (a template with printf placeholders becomes a regex), the
    dashboard's own translation memory (``static/i18n_de.json``, whole texts only) as the second source -- the planner's reason texts that the page already knows
    (e.g. the X bound) need no second copy."""
    global _DE_RULES
    if _DE_RULES is None:
        de: Dict[str, str] = {}
        for path, key in ((TM_FILE, "de"), (DE_FILE, "de")):           # the page's own file last: it wins on a clash
            try:
                with open(path, encoding="utf-8") as fh:
                    de.update(json.load(fh).get(key) or {})
            except (OSError, ValueError):
                pass
        exact: Dict[str, str] = {}
        pats: List[tuple] = []
        for en, g in de.items():
            parts = re.split(r"(%%|%[-.\d]*[dsrf])", en)
            if any(x.startswith("%") and x != "%%" for x in parts):
                rx = ""
                for x in parts:
                    if x == "%%":
                        rx += "%"
                    elif x.startswith("%"):
                        rx += r"(-?\d+)" if x.endswith("d") else r"(-?[\d.]+)" if x.endswith("f") else "(.*)"
                    else:
                        rx += re.escape(x)
                pats.append((re.compile("^" + rx + "$", re.S), g))
            else:
                exact[en.replace("%%", "%")] = g
        _DE_RULES = [exact, pats]
    return _DE_RULES


def _de_one(s: str) -> str:
    """German for one whole text: exact entry, else a template; ``s`` itself when there is none."""
    exact, pats = _de_rules()
    if s in exact:
        return exact[s]
    for rx, g in pats:
        m = rx.match(s)
        if m:
            try:
                return g.format(*[_de_text(x) for x in m.groups()])
            except (IndexError, KeyError, ValueError):
                return s
    return s


def _split_parts(s: str) -> List[str]:
    """``s`` cut into [part, separator, part, ...] at "; " and at ". " before a capital letter or "(" (the planner joins sub-sentences of one reason text with these:
    each part has its own German text, review B2); no regex: this page reads no log."""
    out: List[str] = []
    i = last = 0
    while i < len(s) - 1:
        if s.startswith("; ", i) or (s.startswith(". ", i) and i + 2 < len(s) and (s[i + 2].isupper() or s[i + 2] == "(")):
            out += [s[last:i], s[i:i + 2]]
            last = i = i + 2
        else:
            i += 1
    out.append(s[last:])
    return out


def _de_text(s: str) -> str:
    out = _de_one(s)
    if out != s or not isinstance(s, str):
        return out
    # a composed text matches no template as a whole: translate it part by part and keep the separators (a part without a German text stays English, so the
    # test over the real planner outputs names it)
    bits = _split_parts(s)
    if len(bits) < 3:
        return s
    got = []
    for i, b in enumerate(bits):
        if i % 2:                                           # a separator
            got.append(b)
            continue
        t = _de_one(b)
        if t == b and i:                                    # entries that start with the separator ("; floor %s = ...")
            t2 = _de_one(bits[i - 1] + b)
            if t2 != bits[i - 1] + b and t2.startswith(bits[i - 1]):
                t = t2[len(bits[i - 1]):]
        got.append(t)
    return "".join(got) if any(got[i] != bits[i] for i in range(0, len(bits), 2)) else s


def localize(obj, lang: Optional[str]):
    """The answer of the page API in the page language: every string with a German text in ``planer_ziel_de.json`` is replaced (exact text or template), anything
    else stays English (never blank).  ``lang`` other than ``de`` returns ``obj`` unchanged.  The structural keys (``id``, ``key``, ``roh`` = the raw flag, ``intern`` = the start profile that goes back to the server) are not touched."""
    if lang != "de":
        return obj
    if isinstance(obj, str):
        return _de_text(obj)
    if isinstance(obj, list):
        return [localize(x, lang) for x in obj]
    if isinstance(obj, dict):
        return {k: (v if k in ("id", "key", "roh", "code", "schema", "intern", "form", "outcome", "status", "was", "proof") else localize(v, lang)) for k, v in obj.items()}
    return obj


def _lang(body) -> Optional[str]:
    l = (body or {}).get("lang") if isinstance(body, dict) else None
    return l if l in ("de", "en") else None


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


def _cut_text(s: str, n: int) -> str:
    """``s`` cut to at most ``n`` characters at the end of a sentence when there is one (a half sentence has no German text and reads as an error); else hard, with an ellipsis."""
    s = str(s)
    if len(s) <= n:
        return s
    k = max(s.rfind(". ", 0, n), s.rfind("; ", 0, n))
    return s[:k + 1] if k >= n // 2 else s[:n - 1] + "\u2026"


def _fmt_gib(mib: float) -> str:
    return "%.1f GiB" % (mib / 1024.0)


class ZielPlaner:
    def __init__(self, editor, steckbrief: Optional[Dict[str, dict]] = None, steckbrief_file: str = STECKBRIEF_FILE):
        self.ed = editor
        self._steck = steckbrief
        self._steck_file = steckbrief_file
        self._n_states: Optional[Dict[int, dict]] = None

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

    @staticmethod
    def _has_measured_node(node) -> bool:
        """A value node ``{v, src}`` of the hardware profile that was MEASURED (``src == "gemessen"``), anywhere below ``node``."""
        if isinstance(node, dict):
            if node.get("src") == "gemessen" and node.get("v") is not None:
                return True
            return any(ZielPlaner._has_measured_node(x) for x in node.values())
        if isinstance(node, list):
            return any(ZielPlaner._has_measured_node(x) for x in node)
        return False

    def _rates_measured(self, hw: dict) -> bool:
        """True when EVERY card of the hardware profile has measured rates (memory bandwidth / compute: ``mem_gbs``, ``compute``, ``gemm_tflops``).  A profile that
        was only CAPTURED (NVML names and sizes) has none: its figures are catalog values and borrowed rates, not measurements (review A5)."""
        cards = hw.get("cards") or []
        return bool(cards) and all(any(self._has_measured_node(c.get(k)) for k in ("mem_gbs", "compute", "gemm_tflops")) for c in cards)

    def _measured(self, cards: List[dict]) -> bool:
        hw = self.ed._hardware_profile() if self.ed is not None else None
        if hw is None:
            return False
        return bool(self.ed._hw_cards_match(hw, [{"entry": e} for e in cards])) and self._rates_measured(hw)

    def _hw_info(self, cards: List[dict]) -> dict:
        hw = self.ed._hardware_profile() if self.ed is not None else None
        same = bool(hw) and bool(self.ed._hw_cards_match(hw, [{"entry": e} for e in cards]))
        if same and self._rates_measured(hw):
            return {"quelle": "gemessen", "label": "measured",
                    "text": "These are the cards of this machine: the hardware profile (NVML, measured rates) is used."}
        if same:
            idx = [c.get("nvml_index") for c in hw.get("cards") or [] if isinstance(c.get("nvml_index"), int) and not isinstance(c.get("nvml_index"), bool)]
            return {"quelle": "erfasst", "label": "rates estimated",
                    "text": "These are the cards of this machine (NVML names and sizes captured), but their rates (memory bandwidth, compute) were never measured: the planner "
                            "borrows the rates of the nearest measured card. Every figure below is an estimate until the rates are measured (button below; it books a GPU window).",
                    "measure": {"api": "api/hwprofil/measure", "cards": idx}}
        return {"quelle": "geschaetzt", "label": "estimated",
                "text": "No measured hardware profile for these cards yet: memory sizes and bandwidth are catalog (datasheet) values, rates are borrowed from the "
                        "nearest measured card. Every figure below is an estimate; the first boot on this hardware is the measurement."}

    # ------------------------------------------------------------------ Topologie (B4)
    def n_states(self) -> Dict[int, dict]:
        """The state of every card count 1..8 on the page, once per process: ``{"state": "proven" | "unproven" | "off" | "refused", "text": ...}``.

        * ``proven``   -- in ``SUPPORTED_N`` and the topology gate lets it through (a release boot proved it);
        * ``unproven`` -- in ``SUPPORTED_N``, the gate refuses with the ONE blocker METAL-UNPROVEN: the launcher dry run calculates the layout, the launcher takes it only
          with force, the first boot is the proof (``topology.plan_topology``: ``HW-COUNT``);
        * ``refused``  -- in ``SUPPORTED_N`` but the gate names another blocker (never offered; the gate text is the reason);
        * ``off``      -- outside ``SUPPORTED_N``: not calculated here (one card: nothing to flip between; 8 cards: no profile and no dry-run golden), with its reason.

        Without a planner gate at all the constants decide; a gate that cannot be asked for ONE count does not offer it (``refused``, its notes as the text): ``PROVEN_N`` proven, the rest of
        ``SUPPORTED_N`` unproven."""
        if self._n_states is not None:
            return self._n_states
        out: Dict[int, dict] = {}
        tp = None
        try:
            _ci, tp = self.ed.kp._mods()
        except Exception:                                   # noqa: BLE001 -- no gate to ask: the constants below
            tp = None
        for n in range(1, 9):
            if n not in SUPPORTED_N:
                out[n] = {"state": "off", "text": self._off_reason(n)}
                continue
            notes: List[str] = []
            refused = None
            if tp is not None:
                try:
                    refused = self.ed._topology_verdict(n, tp, notes)
                except Exception:                           # noqa: BLE001 -- a broken gate is "not checkable", never a 500
                    notes.append("topology gate failed")
            if tp is None:
                out[n] = {"state": "proven" if n in PROVEN_N else "unproven", "text": ""}
            elif notes:                                     # the gate exists but could not be asked for this count: not offered (review B4)
                out[n] = {"state": "refused", "text": "; ".join(notes)[:200]}
            elif not refused:
                out[n] = {"state": "proven", "text": ""}
            elif self._blocker_codes(refused) == {"METAL-UNPROVEN"}:
                out[n] = {"state": "unproven", "text": ""}
            else:
                out[n] = {"state": "refused", "text": refused}
        self._n_states = out
        return out

    @staticmethod
    def _blocker_codes(text: str) -> set:
        """The blocker codes ``[CODE]`` that a topology refusal names (``[3]`` in "(proven: [3])" is not a code)."""
        out = set()
        for piece in str(text).split("[")[1:]:
            code = piece.split("]")[0]
            if code[:1].isalpha() and code == code.upper() and all(ch.isalnum() or ch == "-" for ch in code):
                out.add(code)
        return out

    @staticmethod
    def _off_reason(n: int) -> str:
        if n < SUPPORTED_N[0]:
            return ("%d card: a flip or dual needs two groups over at least two cards, and the single-group mode for one card (the decode group alone behind the front, "
                    "with a model fit check before load) does not exist yet (plan P4) (not supported yet)." % n)
        return ("%d cards: the planner is checked for %d to %d cards (a launcher dry run per card count); more cards are not calculated here yet (not supported yet)."
                % (n, SUPPORTED_N[0], SUPPORTED_N[-1]))

    def supported_n(self) -> tuple:
        """The card counts the page calculates: the counts of ``SUPPORTED_N`` that the topology gate does not refuse by name (proven or only METAL-UNPROVEN, see
        ``n_states``).  Without a planner gate: ``SUPPORTED_N``."""
        got = tuple(n for n, st in sorted(self.n_states().items()) if st["state"] in ("proven", "unproven"))
        return got if got else SUPPORTED_N

    def _release_block(self, model: dict, layout: str, cards: List[dict]) -> Optional[str]:
        """The named open defect of the release profile for N identical RTX 3090 (``<stem>-<N>x3090.env``): a profile with ``PROFILE_STATUS=blockiert-<code>`` is not
        startable (NCARDS 1021: NF on 2 and 4 cards, the launcher's D-EIGENTUM solver finds no sustainable form and the D-BUDGET refusal only passes by name with force).  The
        page does not offer a layout that its own release profile says cannot start; the answer is the profile's status, never a guess.  ``None`` = no such profile / not blocked."""
        stem = (model.get("ncards") or {}).get(layout)
        rdir = getattr(self.ed, "release_dir", None)
        if not stem or not rdir or len({c["id"] for c in cards}) != 1 or cards[0]["id"] != "rtx3090-24":
            return None
        code = ""
        try:
            with open(os.path.join(rdir, "%s-%dx3090.env" % (stem, len(cards))), encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("PROFILE_STATUS=blockiert-"):
                        code = line.split()[0][len("PROFILE_STATUS=blockiert-"):]
                        break
        except OSError:
            return None
        if not code:
            return None
        return ("The release profile for this model on %d cards of this kind is marked blocked (%s): the launcher's decode-ownership solver (D-EIGENTUM, #239 S3f) finds no "
                "sustainable distribution, and even with force it only passes the D-BUDGET refusal by name. Not startable yet; the cause is open." % (len(cards), code.upper()))

    # ------------------------------------------------------------------ Katalog
    def katalog(self, lang: Optional[str] = None) -> dict:
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
        return localize({"ok": True, "schema": SCHEMA, "models": models, "cards": cards, "layouts": [dict(l) for l in LAYOUTS], "supported_n": list(self.supported_n()), "n_states": {str(n): st["state"] for n, st in sorted(self.n_states().items())},
                "machine": ({"cards": [{"card": k, "count": v} for k, v in counts.items()], "n": len(mine)} if mine else None),
                "shortcuts": [dict(x) for x in SHORTCUTS], "defaults": {"seats": DEFAULT_SEATS, "tokens": DEFAULT_TOKENS}, "goals": GOALS,
                "overhead_per_card_mib": OVERHEAD_PER_CARD_MIB}, lang)

    # ------------------------------------------------------------------ Layouts (Rechenbegruendung)
    def _weights_mib(self, mid: str) -> Optional[int]:
        """The weights of the model in MiB as the cards must hold them.  For a MoE model with experts in the host that is the sum of its parts (non-expert layers +
        lm_head + experts); the sheet's raw ``weights_total_mib`` of such a checkpoint (NF: 167112) is the model profile's own total and is NOT additive with its
        parts (the stored safetensors of the NF checkpoint are 69455 MiB), so it is never shown (review B5)."""
        s = self.steck.get(mid)
        if not s:
            return None
        if s.get("experts"):
            return int(s["weights_dense_mib"] + s["lm_head_mib"] + s["experts"]["total_mib"])
        return s.get("weights_total_mib")

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
        return localize(self._layouts(body), _lang(body))

    def _layouts(self, body: dict) -> dict:
        model = self._model(body.get("model"))
        cards = self._cards(body.get("cards"))
        seats = self._int(body.get("seats"), "seats", *GOALS["seats"]) or DEFAULT_SEATS
        tokens = self._int(body.get("tokens"), "tokens", *GOALS["tokens"]) or DEFAULT_TOKENS
        n = len(cards)
        s = self.steck.get(model["id"])
        if s is None:
            raise ZielError("no model fact sheet for %s (planer_data/steckbrief.json)" % model["id"])
        gate = self._gate(model, cards)
        sup = self.supported_n()
        nst = self.n_states().get(n) or {"state": "off", "text": self._off_reason(n)}
        out = []
        for l in LAYOUTS:
            calc = self._calc(model, l["id"], cards, seats, tokens)
            why: List[str] = []
            status = "ok"
            profile = model["profiles"].get(l["id"])
            if n not in sup:
                status = "n_unsupported"
                if nst["state"] == "refused":
                    why.append("%d cards: the topology gate refuses this card count by name (%s)." % (n, nst["text"][:200]))
                else:
                    why.append(nst["text"] or self._off_reason(n))
            elif nst["state"] == "unproven":
                why.append("%d cards: no release boot has proven this card count on metal yet (the named blocker METAL-UNPROVEN). The numbers come from the launcher dry run; "
                           "the launcher takes this layout only with force, and the first boot is the proof." % n)
            if gate:
                status = "blocked"
                why += gate
            if not profile:
                status = "blocked"
                why.append(model.get("no_dual") or "no reference profile for this layout")
            blocked_by_profile = self._release_block(model, l["id"], cards) if status == "ok" else None
            if blocked_by_profile:
                status = "blocked"
                why.append(blocked_by_profile)
                if calc["margin_mib"] >= 0:                         # the numbers of the calculation below: it is not the memory
                    why.append("Memory is not the limit: the rough calculation below leaves a margin of %s, and the launcher's per-rank memory checks of the profile (W122/W130) "
                               "pass on every rank. What fails is the ownership search of the decode group: no form with an edge on every rank and without more "
                               "misses of a worker than form A." % _fmt_gib(calc["margin_mib"]))
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
                        "margin_mib": calc["margin_mib"], "proof": "metal" if nst["state"] == "proven" else "dry_run"})
        return {"ok": True, "schema": SCHEMA, "model": {"id": model["id"], "name": model["name"]}, "n": n, "layouts": out,
                "hw": self._hw_info(cards), "seats": seats, "tokens": tokens,
                "note": "A rough necessary condition from the model fact sheet and the card data. The planner's computation (next step) decides; "
                        "a layout shown as possible can still be refused there, a blocked one cannot be forced here."}

    # ------------------------------------------------------------------ Vorschlag
    def plan(self, body: dict) -> dict:
        return localize(self._plan(body), _lang(body))

    def _plan(self, body: dict) -> dict:
        lay = self._layouts(body)
        model = self._model(body.get("model"))
        layout = self._layout(body.get("layout"))
        li = next(x for x in lay["layouts"] if x["id"] == layout["id"])
        cards = self._cards(body.get("cards"))
        base = {"ok": True, "schema": SCHEMA, "model": lay["model"], "layout": {"id": layout["id"], "name": layout["name"]}, "n": lay["n"], "hw": lay["hw"],
                "layout_info": li}
        if li["status"] in ("n_unsupported", "blocked"):
            # dict literals, not keyword arguments: the release rename (ident map) renames a ``verdikt=`` keyword but never a string key, which would split the
            # JSON contract of the page in the renamed tree (review B6)
            return dict(base, **{"status": li["status"], "felder": [], "verdikt": None, "hinweise": li["why"]})
        seats = self._int(body.get("seats"), "seats", *GOALS["seats"])
        tokens = self._int(body.get("tokens"), "tokens", *GOALS["tokens"])
        res = self._propose(model, layout, cards, seats, tokens)
        out = dict(base, status="ok" if res.get("ok") else "error")
        if not res.get("ok"):
            return dict(out, **{"felder": [], "verdikt": None, "hinweise": [res.get("error") or "the planner gave no proposal"] + list(res.get("notes") or [])})
        out.update(self._summary(model, layout["id"], cards, res, seats, tokens))
        out["intern"] = {"basis": model["profiles"][layout["id"]], "form": layout["form"], "oracle": res.get("oracle"),
                         "startprofil": {"doc": (res.get("startprofil") or {}).get("doc"), "name": (res.get("startprofil") or {}).get("name")}}
        # the ceiling per request is the one input that the cards can refuse outright (W40: no servable pipeline cut holds ONE full-context prompt, e.g. 27B INT8 on
        # 2 x 24 GB).  Then the page does not stop at the refusal: it asks the launcher again at the next lower ceiling presets and shows the largest one it takes -- real
        # dry runs, never an estimate -- and says so (``anpassung``).
        asked = int(tokens or (out.get("goals") or {}).get("kv_tokens") or DEFAULT_TOKENS)       # what the user asked, else what the planner resolved as the default
        if self._ceiling_refused(out):
            for t in sorted((x for x in GOALS["token_presets"] if x < asked), reverse=True):
                res2 = self._propose(model, layout, cards, seats, t)
                if not res2.get("ok"):
                    continue
                out2 = dict(base, status="ok")
                out2.update(self._summary(model, layout["id"], cards, res2, seats, t))
                if self._ceiling_refused(out2):
                    continue
                out2["intern"] = {"basis": model["profiles"][layout["id"]], "form": layout["form"], "oracle": res2.get("oracle"),
                                  "startprofil": {"doc": (res2.get("startprofil") or {}).get("doc"), "name": (res2.get("startprofil") or {}).get("name")}}
                out2["anpassung"] = {"gefragt": asked, "erreicht": t, "was": "tokens",
                                     "text": "%d tokens per seat do not fit these cards (the launcher refuses with W40: no servable cut holds one full-context prompt). "
                                             "The figures below are for %d tokens per seat, the largest ceiling the launcher takes (dry runs at the ceiling presets)." % (asked, t)}
                return out2
            out["anpassung"] = {"gefragt": asked, "erreicht": None, "was": "tokens",
                                "text": "%d tokens per seat do not fit these cards, and none of the lower ceiling presets does either (the launcher refuses each with W40)." % asked}
        return out

    def _propose(self, model: dict, layout: dict, cards: List[dict], seats, tokens) -> dict:
        goals: Dict[str, Any] = {}
        if seats is not None:
            goals["seats"] = seats
        if tokens is not None:
            goals["kv_tokens"] = tokens
        req = {"basis": {"kind": "release", "name": model["profiles"][layout["id"]]}, "form": layout["form"],
               "inventory": [{"card": e["id"], "pcie": {"gen": e["pcie_native"]["gen"], "lanes": e["pcie_native"]["lanes"]}} for e in cards],
               "goals": goals}
        return self.ed.propose(req)

    @staticmethod
    def _ceiling_refused(out: dict) -> bool:
        """The launcher refused with a blocked ``PP-CUT`` (W40) item: the ceiling per request is more than a pipeline cut over these cards can serve."""
        vd = out.get("verdikt") or {}
        return vd.get("outcome") in ("verweigert", "crash") and any(i.get("code") == "PP-CUT" and i.get("blocked") for i in vd.get("items") or [])

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
        # --- KV pool: the EFFECTIVE value is what the launcher applies (a pool flag); the cut solver's pool and our own estimate are derivations only (review A1-A4)
        pp = self._pool_p(layout, res, P, pc)
        if pp:
            row("kv", "p_pool", "KV pool, prefill group (shared by all seats)", pp["tokens"], unit="tokens", source=pp["source"], why=pp["why"], raw=pp["flag"])
        dp = self._pool_d(s, layout, cards, res, D, seats if seats else goals.get("seats"))
        if dp:
            row("kv", "d_pool", "KV pool, decode group (shared by all seats)", dp["tokens"], unit="tokens", source=dp["source"], estimate=dp["estimate"],
                why=dp["why"], raw=dp["flag"])
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
    def _argv_flag(res: dict, flag: str) -> Optional[str]:
        """The value of ``flag`` in the launcher's argv of the dry run (``launch.argv``); ``None`` when it is not there."""
        argv = [str(x) for x in ((res.get("launch") or {}).get("argv") or [])]
        for i, a in enumerate(argv):
            if a == flag and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                return argv[i + 1]
            if a.startswith(flag + "="):
                return a.split("=", 1)[1]
        return None

    def _pool_p(self, layout: str, res: dict, P: dict, pc: dict) -> Optional[dict]:
        """Prefill KV pool in tokens as the launcher APPLIES it: dual ``--dual-p-kv-max-tokens`` (argv), flip ``--max-total-tokens`` of group P.  The cut solver's
        ``chosen_pool`` is only the derivation (for 27B it equals the flag; for NF the flag is the ceiling and the solver priced far more)."""
        if layout not in ("flip", "dual"):
            return None
        flag = "--dual-p-kv-max-tokens" if layout == "dual" else "--max-total-tokens"
        v = self._argv_flag(res, flag) if layout == "dual" else None
        v = v or P.get(flag)
        solver = pc.get("chosen_pool")
        floor = ("; floor %s = one request at the ceiling + one chunk" % pc["pool_floor"]) if pc.get("pool_floor") not in (None, "", "none") else ""
        if v not in (None, ""):
            why = "%s of the launcher run" % flag
            if solver and str(solver) != str(v):
                why += "; the cut solver priced %s tokens, the flag caps the pool" % solver
            return {"tokens": str(v), "source": "launcher dry run (%s)" % flag, "why": why + floor, "flag": flag}
        if solver:
            return {"tokens": str(solver), "source": "launcher cut solver (derivation)", "why": "no pool flag in the launcher run; the cut solver's priced pool" + floor,
                    "flag": "plan.pp_cut.chosen_pool"}
        return None

    def _pool_d(self, s: dict, layout: str, cards: List[dict], res: dict, D: dict, seats) -> Optional[dict]:
        """Decode KV pool in tokens: the launcher's flag when it sets one (dual ``--dual-d-kv-max-tokens``, NF ``--max-total-tokens``), else an ESTIMATE."""
        flag = "--dual-d-kv-max-tokens" if layout == "dual" else "--max-total-tokens"
        v = (self._argv_flag(res, flag) if layout == "dual" else None) or D.get(flag)
        if v not in (None, ""):
            return {"tokens": str(v), "source": "launcher dry run (%s)" % flag, "estimate": False, "flag": flag,
                    "why": "%s of the launcher run: the pool is shared by all seats" % flag}
        est = self._d_pool_estimate(s, layout, cards, D, seats)
        if est:
            return {"tokens": est["tokens"], "source": "estimate", "estimate": True, "why": est["why"], "flag": "D budgets - weights - state - fixed posts, / KV bytes per token"}
        return None

    @staticmethod
    def _resident_experts_mib(s: dict, D: dict) -> Optional[float]:
        """Expert weights that sit in the D budgets (MoE with experts in the host): sum over the cards of share x resident fraction x all experts
        (``--rank-moe-ratio`` x ``--rank-moe-resident-fraction``); ``None`` when the launcher run does not say."""
        try:
            ratio = [float(x) for x in str(D.get("--rank-moe-ratio") or "").split(",") if x.strip()]
            frac = [float(x) for x in str(D.get("--rank-moe-resident-fraction") or "").split(",") if x.strip()]
        except ValueError:
            return None
        if not ratio or len(ratio) != len(frac) or sum(ratio) <= 0:
            return None
        return s["experts"]["total_mib"] * sum(r / sum(ratio) * f for r, f in zip(ratio, frac))

    @staticmethod
    def _d_pool_estimate(s: dict, layout: str, cards: List[dict], D: dict, seats) -> Optional[dict]:
        """Decode KV pool in tokens when the launcher sets no pool flag: the D budgets minus weights, the resident experts (MoE), the state and the fixed D posts
        (``D_FIXED_PER_CARD_MIB`` per card, calibrated on the measured pool of our rig).  An ESTIMATE (the launcher sizes the pool at boot)."""
        budgets = D.get("--rank-gpu-memory-mib")
        if not budgets:
            return None
        try:
            tot = sum(int(x) for x in str(budgets).split(","))
        except ValueError:
            return None
        n = len(cards)
        draft = s.get("draft_mib") or 0
        extra, extra_txt = 0.0, ""
        if s.get("experts"):
            w = s["weights_dense_mib"] + s["lm_head_mib"] + draft
            ex = ZielPlaner._resident_experts_mib(s, D)
            if ex is None:
                return None                                 # experts sit in the same budgets: no honest estimate without their residency
            extra, extra_txt = ex, " - resident experts %.0f MiB" % ex
        else:
            w = s["weights_total_mib"] + draft
        state = (seats or DEFAULT_SEATS) * s["state_mib_per_seat_bf16"]
        fixed = n * D_FIXED_PER_CARD_MIB
        free = tot - w - extra - state - fixed
        if free <= 0:
            return None
        toks = int(free * MIB / s["kv_bytes_per_token_fp8"])
        return {"tokens": toks, "why": "(D budgets %d MiB - weights %d MiB%s - state %.0f MiB - fixed posts %d x %d MiB) / %d B per token; an estimate calibrated on the measured pool of "
                                       "our rig (27B INT8 flip: 778980 tokens), the launcher sizes the pool at boot" % (
            tot, w, extra_txt, state, n, D_FIXED_PER_CARD_MIB, s["kv_bytes_per_token_fp8"])}

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
                seen[code] = {"code": code, "text": _cut_text(x.get("reason") or x.get("title") or "", 300), "blocked": bool(blocked), "force": bool(x.get("forcebar")), "count": 1}
                items.append(seen[code])
        text = {"geht": "The launcher's dry run passes without force.", "ok": "The launcher's dry run passes without force.",
                "ok_with_force": "The launcher's dry run passes only with force: it does not trust numbers that were not measured on these cards (see the notes). "
                                 "The first boot is the measurement.",
                "verweigert": "The launcher refuses this combination (see the reasons).", "crash": "The launcher's dry run failed (see the reasons)."}.get(outcome, "No launcher verdict (%s)." % (outcome,))
        fit = prop.get("fit") or {}
        return {"outcome": outcome, "text": text, "items": items, "fit": {"level": fit.get("level"), "margin_mib": fit.get("margin_mib")}}
