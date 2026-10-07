"""AP-D of the profile planner (plan PLAN-PROFIL-PLANER-1006 section 2 stage B/C, section 3 row AP-D): the ORACLE made productive and
the VERDICTS it hands to the editor.

``propose_oracle`` (AP0) runs the launcher's real ``main`` with ``--dry-run`` on a replayed inventory; ``propose`` (AP-C) derives the
candidate argv.  THIS module asks the oracle about a launch (a release profile, a user profile or a proposal) and turns what the
launcher did into a document ``flliper.verdikt/1``:

* one verdict per THING the launcher said, in the one structure ``{code, ebene, forcebar, force_state, grund, konsequenz, ...}``:
  every ``FORCED-PAST`` line (the value refusals Force would pass), the refusal or the CRASH that ended the dry run, the vector
  blockers of the HW gate (``PROFILE-VECTORS``, ``RECORDS-NVEC``, ``METAL-UNPROVEN`` ... named inside the ``HW-COUNT`` text), the fit
  bound of ``hw_fit`` (``FIT``) and every value that is borrowed (``HW-BORROWED``) or uncalibrated (``HW-UNCALIBRATED``);
* ``forcebar`` comes from ``refusals.by_code`` (the register), ``force_state`` from the register's ``wired`` flags exactly as the
  dashboard reads them (``force`` | ``blockiert`` | ``ungeprueft``) plus ``geht`` / ``hinweis`` for what is no refusal;
* a CRASH of the dry run (an ``IndexError`` of ``overshoot_mib[i]`` on four cards, any exception that is not a launcher refusal) is a
  verdict ``ORAKEL-ABSTURZ`` with the place it happened -- never an exception of the oracle, never a 500 of the dashboard;
* the document carries the PROFILE HASH (plan 4c: live profiles drift): the sha256 of the profile FILE when there is one and of the
  launch input the oracle really got, plus the hash of the argv/env and of the sources of the oracle (launcher, refusals, hw_fit), so a
  verdict can always be attributed to the profile state it was asked about.

The SINGLE CARD form (AP-F, ``run_propose_single``) has no launcher (``topology.py`` MIN_CARDS=2) and so no oracle run: its document is the same
shape (values, ``flliper.verdikt/1``, verdicts per value, ``launch``) but the verdict is a PLANER-RECHNUNG of ``propose_single`` plus the ServerArgs parse
(``ausgang`` passt | passt_nicht | unbelegt, ``lauf`` None, ``laeufe`` 0, no Force), and every verdict of it says so.

Nothing in here judges a value or re-implements a solver (R1/Q-710): every number in a verdict is a number the launcher or ``hw_fit``
printed.  A verdict the oracle cannot give (the harness itself failed) is ``ORAKEL-FEHLER`` and says so.  GPU-free, NVML-free, Docker-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "flliper.verdikt/1"
#: the document the worker returns for a proposal (proposal of AP-C + verdicts of this module)
PROPOSE_SCHEMA = "flliper.propose-d/1"

#: ``force_state`` values: the three of the dashboard's ``force_verdict`` + two for what is no refusal
FORCE, BLOCKED, UNCHECKED, GOES, HINT = "force", "blockiert", "ungeprueft", "geht", "hinweis"

#: codes of THIS module that are not in ``refusals.REGISTER``: ``ebene`` says where they come from, ``parent`` the register code whose
#: text names them (a blocker inside the ``HW-COUNT`` text is judged by its parent's class), ``titel`` / ``konsequenz`` are the fixed words
OWN_CODES: Dict[str, Dict[str, Any]] = {
    "FIT": {"ebene": "fit", "parent": None, "titel": "Passung nach hw_fit (notwendige Bedingung)",
            "konsequenz": "hw_fit prueft nur die NOTWENDIGE Bedingung (Gewichte + KV + Mamba-Slots + Posten gegen Karte minus Residuum); "
                          "ob der Start laeuft, sagt der Launcher-Lauf (die uebrigen Verdikte)."},
    "DUAL-PASSUNG": {"ebene": "fit", "parent": None, "titel": "Dual-Passung: Planer-Rechnung, nicht hw_fit",
                     "konsequenz": "hw_fit rechnet das Dual nicht (es druckt 'Dual ... NOT modelled'); der Planer rechnet je Karte P-Budget + Overhead + "
                                   "Ruhe-Posten von D + D-Gewichte (mit Draft) gegen die Karte. Das ist eine NOTWENDIGE Bedingung aus Modellgroessen und "
                                   "Records, keine Messung; ob der Start laeuft, sagt der Launcher-Lauf (die uebrigen Verdikte)."},
    "DUAL-PFLICHT": {"ebene": "fit", "parent": None, "titel": "P-KV-Pflicht der Dual-Form je Karte (Planer-Rechnung)",
                     "konsequenz": "P muss die KV-Pflicht (Standard 262144 Token) als EINEN Prompt tragen: Deckel, Level des P-Pools und der gemeinsame Pool "
                                   "jeder Karte. Der Pool je Karte ist nur fuer die Karten und das Modell des Referenzboots geeicht; sonst steht dort "
                                   "'nicht gerechnet'. Einen Dual-Pool-Riegel im Launcher gibt es nach dem 27B-Sitz nicht (done/dual-schnitt-262k-1006.md Abschnitt 5): diese Rechnung ist der einzige Schutz davor."},
    "EINZEL-PASSUNG": {"ebene": "fit", "parent": None, "titel": "Einzelkarte: Passung als Planer-Rechnung (kein Launcher-Lauf)",
                       "konsequenz": "Die Einzelkarte hat keinen weg2-Launcher (topology.py MIN_CARDS=2): der Planer rechnet Gewichte + Draft + KV-Pflicht + Mamba-Pool "
                                     "+ Reserve gegen das statische Budget (Bruchteil x freier Speicher vor dem Laden). Das ist eine NOTWENDIGE Bedingung aus "
                                     "Modellgroessen, keine Messung; ein Force gibt es bei N=1 nicht."},
    "EINZEL-PARSE": {"ebene": "fit", "parent": None, "titel": "Einzelkarte: ServerArgs-Parse (nur argparse, ohne Geraet)",
                     "konsequenz": "Der Parse prueft Flag-Namen, Auswahlwerte und Typen von ServerArgs.add_cli_args; ServerArgs.__post_init__ (Geraeteerkennung, "
                                   "Kompatibilitaets-Refusals) braucht einen Beschleuniger und ist NICHT gelaufen."},
    "FIT-STATIC": {"ebene": "fit", "parent": None, "titel": "Gewichte + Draft + KV + Mamba gegen das statische Budget",
                   "konsequenz": "Der Pool ist groesser als --mem-fraction-static x freier Speicher vor dem Laden; die Runtime wuerde beim Laden oder Poolbau am Speicher scheitern."},
    "FIT-RESERVE": {"ebene": "fit", "parent": None, "titel": "Reserve ausserhalb des statischen Budgets zu klein",
                    "konsequenz": "Aktivierungen, CUDA-Graphen und Fragmentierung brauchen mehr als (1 - Bruchteil) des freien Speichers: ein Lauf kann an der Reserve scheitern."},
    "FIT-CARD": {"ebene": "fit", "parent": None, "titel": "Statisches Budget groesser als die adressierbare Decke der Karte",
                 "konsequenz": "Auf einer APU oder bei Hostpool teilen Geraet und Host den Speicher; das Budget liegt ueber der Decke."},
    "FIT-CTX": {"ebene": "fit", "parent": None, "titel": "Kontext groesser als der KV-Pool",
                "konsequenz": "Ein Prompt der Kontextlaenge passt nicht in den KV-Pool."},
    "MAMBA-FLOOR": {"ebene": "fit", "parent": None, "titel": "Mamba-Pool unter der Untergrenze",
                    "konsequenz": "Weniger Slots als Anfragen x Slots je Anfrage (mamba_pool_floor.py): der Server verweigert oder haelt Anfragen nicht."},
    "PROFILE-VECTORS": {"ebene": "blocker", "parent": "HW-COUNT", "titel": "Positionsvektoren des Profils nicht fuer dieses Inventar",
                        "konsequenz": "Das Profil traegt Vektoren (je Karte ein Eintrag) mit anderer Laenge als die Kartenzahl; der Launcher kann sie "
                                      "nicht ableiten. Mit Vorschlag des Planers haben alle Vektoren genau N Eintraege."},
    "RECORDS-NVEC": {"ebene": "blocker", "parent": "HW-COUNT", "titel": "Gemessene Records sind Vektoren eines anderen Inventars",
                     "konsequenz": "Die Records des Profils (weg2/profile_records_data) sind Vektoren eines Inventars und fuer dieses nicht "
                                   "ableitbar; ein Kalibrierboot dieser Karten muss sie schreiben. Im Launcher kann das spaeter als Absturz oder "
                                   "Verweigerung auftreten (siehe ORAKEL-ABSTURZ)."},
    "METAL-UNPROVEN": {"ebene": "blocker", "parent": "HW-COUNT", "titel": "Kartenzahl am Metall nicht bewiesen",
                       "konsequenz": "Es gibt kein Release-Boot mit dieser Kartenzahl; die Rechnung des Planers ist Hochrechnung, keine Messung."},
    "HW-BORROWED": {"ebene": "wert", "parent": None, "titel": "Wert von einer anderen Karte oder einem anderen Profil geborgt",
                    "konsequenz": "Kein Ablehnungscode: der Wert ist von einer Zwillingskarte der Architektur oder einem anderen Profil geliehen "
                                  "(unbelegt) und am Metall dieser Karten nicht gemessen."},
    "UNBELEGT": {"ebene": "wert", "parent": None, "titel": "Wert ist Planer-Rechnung, nicht gemessen",
                 "konsequenz": "Kein Ablehnungscode: der Wert ist Hochrechnung des Planers und am Metall dieser Karten nicht belegt."},
    "PLANER": {"ebene": "planer", "parent": None, "titel": "Der Vorschlag des Planers kann diese Form nicht belegen",
               "konsequenz": "Stufe A (propose) hat fuer diese Karten keine passende Aufteilung gefunden; der Launcher-Lauf zeigt, was er daraus macht."},
    "ORAKEL-ABSTURZ": {"ebene": "absturz", "parent": None, "titel": "Der Launcher-Trockenlauf stuerzte ab",
                       "konsequenz": "Kein Urteil ueber die Werte, sondern ein Fehler des Launchers fuer diese Form (Ausnahme statt Verweigerung). "
                                     "Force aendert daran nichts; der echte Start bricht an derselben Stelle ab."},
    "ORAKEL-FEHLER": {"ebene": "orakel", "parent": None, "titel": "Das Orakel konnte nicht fragen",
                      "konsequenz": "Der Trockenlauf kam nicht zustande (Harness, Modellpfad, Kindprozess): es gibt KEIN Urteil, weder 'geht' noch 'verweigert'."},
}

#: launcher codes WITHOUT a row in ``refusals.REGISTER`` (R1: the launcher and the register are not changed from here).  Without this table the
#: dashboard showed them as ``LAUNCHER-UNKLASSIFIZIERT``.  Key = the launcher W-code; ``code`` is the name the verdict carries; the class is
#: ``nicht_forcebar`` and ``forcebar`` is False for both: neither is passed by ``refuse_value`` (the launcher raises them as plain refusals),
#: and Force changes neither (a missing census / a missing measurement is not a number Force could start with).  ``grund`` is the launcher's
#: own wording with the place it is printed (file:line of THIS tree, pinned by a test against the source text).
SUPPLEMENT_CODES: Dict[str, Dict[str, Any]] = {
    "W71": {"code": "W71-CENSUS", "klass": "nicht_forcebar", "forcebar": False,
            "titel": "W71 UUID-gebundener Exchange-Census: Residenz-Rechnung nicht belegt",
            "quelle": "weg2/xchg_residency.py:711-723 (refusal_head), :313-387 (load_census)",
            "klasse_grund": "Der Launcher verweigert W71 als Weg2XchgResidencyUnarmable ohne Force-Pfad: die Rechnung braucht einen gemessenen Census "
                            "dieser Karten (je UUID), und ohne ihn gibt es keinen Wert, mit dem ein erzwungener Start laufen koennte.",
            "grund": "W71 Weg2XchgResidencyUnarmable: the exchange's predicted VRAM residency does not fit (or the census file is missing/unreadable); "
                     "there is no fallback that makes an over-committed card fit: the boot REFUSES by name and exits 2, BEFORE either group starts. "
                     "Run --weg2-weight-source ring, or re-cut the schedule. [Launcher-Text xchg_residency.py:714-723]",
            "konsequenz": "Bleibt auch mit Force bestehen. Der Census ist je UUID gemessen und an die Karten gebunden; fremde Karten haben keinen. "
                          "Ausweg laut Launcher: --weg2-weight-source ring oder den Schnitt neu legen."},
    "W64": {"code": "W64-OPPOINT", "klass": "nicht_forcebar", "forcebar": False,
            "titel": "W64 Betriebspunkt: das Modell ergibt keinen positiven KV-Pool",
            "quelle": "launcher.py:16986-16993 (Weg2TpOperatingPointInfeasible), :17349-17352 (fatal fuer die ausgelieferte Position, auch ohne dual_layout)",
            "klasse_grund": "Der Launcher verweigert W64 als Urteil des Modells (PerfCostModel.predict_capacity feasible=False) ohne Sicherheitsfaktor und "
                            "ohne Force-Pfad: es gibt keinen Wert, mit dem ein erzwungener Start laufen koennte.",
            "grund": "W64 Weg2TpOperatingPointInfeasible: the derived D weights are marked feasible=False against this boot's budgets -- the weight shards "
                     "plus the mamba pool plus the reserves do not leave a positive KV pool on at least one rank. Refused. This is the model's own verdict, "
                     "not a margin chosen here. [Launcher-Text launcher.py:16986-16993]",
            "konsequenz": "Bleibt auch mit Force bestehen. Das Urteil ist das des Modells; hilft nur, Budgets, Gewichtsaufteilung oder Kartenzahl zu aendern.",
            # nur wenn die Meldung 'W64-DUAL:' traegt (der Launcher haengt es nur bei dual_layout an, launcher.py:17231/17242): Dual-Wortlaut
            "dual": {"titel": "W64 Dual-D: Betriebspunkt ohne gemessenes Dual-D-Log nicht belegt",
                     "quelle": "launcher.py:16986-16993 (Weg2TpOperatingPointInfeasible), :17242-17243 (kein gemessenes Dual-D-Log)",
                     "klasse_grund": "Der Launcher verweigert W64 als Urteil des Modells (PerfCostModel.predict_capacity feasible=False) ohne Sicherheitsfaktor; "
                                     "im Dual hebt nur ein gemessenes Dual-D-Log der Gewichte die Ablehnung auf (dual_w64.find_dual_d_measurement), nicht Force.",
                     "grund": "W64 Weg2TpOperatingPointInfeasible: the derived D weights are marked feasible=False against this boot's budgets -- the weight shards "
                              "plus the mamba pool plus the reserves do not leave a positive KV pool on at least one rank. Refused. This is the model's own verdict, "
                              "not a margin chosen here. W64-DUAL: no measured dual-share D log of the model with these weights; the model verdict stands. "
                              "[Launcher-Text launcher.py:16986-16993, :17242-17243]",
                     "konsequenz": "Bleibt auch mit Force bestehen. Aufgehoben wird es nur durch ein gemessenes Dual-D-Log dieser Gewichte (Evidence-Verzeichnis); "
                                   "ohne Messung gilt das Urteil des Modells."},
            # 'W64-DUAL: the refusal names no weight vector' (launcher.py:17235): die Meldung nennt keinen Gewichtsvektor, es gab also keine Suche
            "dual_ohne_gewichte": {"titel": "W64 Dual: Betriebspunkt nicht belegt (Meldung ohne Gewichtsvektor, kein Dual-D-Log gesucht)",
                     "quelle": "launcher.py:16986-16993 (Weg2TpOperatingPointInfeasible), :17235 (the refusal names no weight vector)",
                     "klasse_grund": "Der Launcher verweigert W64 als Urteil des Modells ohne Sicherheitsfaktor; im Dual konnte er das Dual-D-Log nicht suchen, weil "
                                     "die Meldung keinen Gewichtsvektor nennt (dual_w64.find_dual_d_measurement braucht ihn).",
                     "grund": "W64 Weg2TpOperatingPointInfeasible: the derived D weights are marked feasible=False against this boot's budgets. "
                              "W64-DUAL: the refusal names no weight vector; the model verdict stands. [Launcher-Text launcher.py:16986-16993, :17235]",
                     "konsequenz": "Bleibt auch mit Force bestehen. Es wurde kein Dual-D-Log gesucht (kein Gewichtsvektor in der Meldung); das Urteil des Modells gilt."},
            # 'W64-DUAL MEASURED (...) -> INFEASIBLE' (dual_w64.py:117, angehaengt launcher.py:17343-17347): ein Dual-D-Log WURDE gefunden und urteilt selbst
            "dual_gemessen": {"titel": "W64 Dual-D: gemessenes Dual-D-Log bestaetigt: kein ausreichender KV-Pool",
                     "quelle": "launcher.py:16986-16993 (Weg2TpOperatingPointInfeasible), :17343-17347 (Messung angehaengt), dual_w64.py:117 (judge)",
                     "klasse_grund": "Der Launcher verweigert W64 als Urteil des Modells; im Dual wurde ein gemessenes Dual-D-Log dieser Gewichte gefunden "
                                     "(dual_w64.find_dual_d_measurement) und urteilt auf den eigenen Posten des Laufs INFEASIBLE. Force aendert die Messung nicht.",
                     "grund": "W64 Weg2TpOperatingPointInfeasible: the derived D weights are marked feasible=False against this boot's budgets. "
                              "W64-DUAL MEASURED (...) -> INFEASIBLE: a measured dual-share D log of these weights also leaves less than the minimum tokens on at least "
                              "one rank. [Launcher-Text launcher.py:16986-16993, dual_w64.py:117]",
                     "konsequenz": "Bleibt auch mit Force bestehen. Die Messung dieser Gewichte bestaetigt das Urteil; hilft nur, Budgets, Gewichtsaufteilung oder "
                                   "Kartenzahl zu aendern."}},
}

def _supp_view(supp: Mapping[str, Any], text: Any) -> Dict[str, Any]:
    """The supplement row in the wording the refusal text earns.  The launcher prints W64 in four shapes (``dual_layout`` adds the last three): no Dual
    marker (form-neutral base); ``W64-DUAL MEASURED (...) -> INFEASIBLE`` (a measured log was found, dual_w64.py:117, launcher.py:17343-17347);
    ``W64-DUAL: the refusal names no weight vector`` (no search happened, :17235); ``W64-DUAL: no measured dual-share D log`` (searched, none, :17242)."""
    t = str(text or "")
    if "W64-DUAL MEASURED" in t and supp.get("dual_gemessen"):
        return {**supp, **supp["dual_gemessen"]}
    if "W64-DUAL: the refusal names no weight vector" in t and supp.get("dual_ohne_gewichte"):
        return {**supp, **supp["dual_ohne_gewichte"]}
    if "W64-DUAL:" in t and supp.get("dual"):
        return {**supp, **supp["dual"]}
    return supp


#: launcher exception CLASS name -> register code (the class names of ``launcher.py`` / ``weg2/__init__``); a name not here is judged by
#: its W-code or stays ``LAUNCHER-UNKLASSIFIZIERT``
_CLASS_CODE = {"Weg2PPCutRefused": "PP-CUT", "Weg2DepthUnfunded": "PP-CUT", "Weg2DepthGapped": "PP-CUT"}
#: launcher W-code -> register code (``refusals.REGISTER`` row ``PP-CUT``: "W40/W42/W43")
_W_CODE = {"W40": "PP-CUT", "W42": "PP-CUT", "W43": "PP-CUT"}

_SPLIT_BLOCKERS_RX = re.compile(r";\s+(?=\[[A-Z])")
_VEC_ITEM_RX = re.compile(r"^(\S.*?) \((\d+) entries\)$")
#: the launcher W-code at the START of a refusal text ("W10 Weg2DrafterIdentityMismatch: ...")
_W_START_RX = re.compile(r"^\s*W(\d+[a-z]?)\b(?![-.\w])")
#: a W-code as a whole token inside a refusal text: not part of a path or name (``-W8-``, ``/W8``, ``xW8``)
_W_TOKEN_RX = re.compile(r"(?<![\w/.-])W(\d+[a-z]?)(?![\w/.-])")
_NAMED_RX = re.compile(r"\((HW-[A-Z]+)\)")


# ---------------------------------------------------------------------------
# hashes (plan 4c: live profiles drift)
# ---------------------------------------------------------------------------

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def file_sha256(path: str) -> Optional[str]:
    """sha256 of a file, None when it cannot be read (a profile that is not a file, an image path)."""
    try:
        with open(path, "rb") as fh:
            return _sha(fh.read())
    except OSError:
        return None


def launch_hash(argv: Sequence[str], env: Mapping[str, Any]) -> str:
    """sha256 of the argv and the (sorted) environment of a launch: what the cache keys on and what a verdict is attributed to."""
    return _sha(canonical({"argv": [str(t) for t in argv], "env": {str(k): str(v) for k, v in sorted(env.items())}}).encode())


def profile_identity(li: Any) -> Dict[str, Any]:
    """The profile state the oracle was asked about: ``quelle`` (file name), ``datei_sha256`` (the profile FILE, None when the launch
    input is not from a file), ``eingabe_sha256`` (argv + env + variables of the launch input the oracle really got), model and draft."""
    src = str(getattr(li, "source", "") or "")
    vars_ = dict(getattr(li, "vars", {}) or {})
    return {"quelle": os.path.basename(src) if src else "", "datei_sha256": file_sha256(src) if src and os.path.isfile(src) else None,
            "eingabe_sha256": _sha(canonical({"argv": list(li.argv), "env": dict(li.env), "vars": vars_}).encode()),
            "model": str(vars_.get("PROFILE_MODEL", "")), "draft": str(vars_.get("PROFILE_DRAFT", "")),
            "name": str(vars_.get("PROFILE_NAME", "")), "status": str(vars_.get("PROFILE_STATUS", ""))}


def oracle_version() -> Dict[str, Optional[str]]:
    """sha256 (first 16) of the sources a verdict depends on: the launcher, the refusal register, ``hw_fit``, ``topology``, the oracle."""
    from sglang.srt.weg2 import hw_fit, launcher, propose_oracle, refusals, topology

    out: Dict[str, Optional[str]] = {}
    for name, mod in (("launcher", launcher), ("refusals", refusals), ("hw_fit", hw_fit), ("topology", topology),
                      ("propose_oracle", propose_oracle)):
        h = file_sha256(getattr(mod, "__file__", ""))
        out[name] = h[:16] if h else None
    return out


# ---------------------------------------------------------------------------
# the register as the verdicts read it
# ---------------------------------------------------------------------------

_REG_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}


def register_rows() -> Dict[str, Dict[str, Any]]:
    """``{code: row}`` of ``refusals.public_register`` with the ``wired`` flags of THIS tree's launcher (read once per process)."""
    if "rows" not in _REG_CACHE:
        from sglang.srt.weg2 import launcher, refusals

        with open(launcher.__file__, encoding="utf-8") as fh:
            wired = refusals.wired_codes(fh.read())
        _REG_CACHE["rows"] = {r["code"]: r for r in refusals.public_register(wired)}
    return _REG_CACHE["rows"]


def force_state_of(row: Optional[Mapping[str, Any]]) -> Tuple[str, Optional[str]]:
    """``(force_state, via)`` of a register row: the SAME reading as ``rigdash.profil.force_verdict`` (a test pins the two together)."""
    if not row:
        return BLOCKED, None
    if not row.get("forcebar"):
        return BLOCKED, None
    if row.get("wired"):
        return FORCE, "launcher"
    ep = row.get("wired_entrypoint")
    if ep is None:
        ep = row.get("enforced_by") == "entrypoint"
    if ep:
        return FORCE, "entrypoint"
    if row.get("enforced_by") == "planner-gate":
        return UNCHECKED, None
    return BLOCKED, None


def _clip(text: Any, n: int = 700) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip(" ,;:") + "…"


def verdikt(code: str, *, ebene: str = "lauf", text: str = "", grund: Optional[str] = None, werte: Sequence[str] = (),
            parent: Optional[str] = None, force_state: Optional[str] = None, extra: Optional[Mapping[str, Any]] = None,
            konsequenz: Optional[str] = None) -> Dict[str, Any]:
    """ONE verdict ``{code, ebene, forcebar, force_state, grund, konsequenz, text, ...}``.

    A register code takes ``forcebar`` / ``konsequenz`` / ``klasse_grund`` from ``refusals.by_code``; a sub-blocker (``parent``) takes the
    class of its parent and its OWN words; a code that is no refusal (``FIT``, ``HW-BORROWED`` ...) has ``forcebar: None``.  ``force_state``
    is derived from the register unless the caller gives it (``geht`` / ``hinweis`` for what is no refusal, or a run-specific state)."""
    reg = register_rows()
    supp = next((x for x in SUPPLEMENT_CODES.values() if x["code"] == code), None)
    if supp is not None and code not in reg:
        supp = _supp_view(supp, grund if grund is not None else text)
        out = {"code": code, "ebene": ebene, "titel": supp["titel"], "forcebar": False, "force_state": BLOCKED, "force_via": None,
               "klasse": supp["klass"], "klasse_grund": supp["klasse_grund"], "wired_at": None, "force_scope": None,
               "konsequenz": konsequenz if konsequenz is not None else supp["konsequenz"],
               "grund": _clip(grund if grund is not None else text) or _clip(supp["grund"]), "text": str(text or ""), "werte": list(werte),
               "quelle": supp["quelle"], "ergaenzung": True}
        if force_state is not None:
            out["force_state"] = force_state
        if extra:
            out.update(extra)
        return out
    own = OWN_CODES.get(code) or {}
    parent = parent or own.get("parent")
    row = reg.get(code) or (reg.get(parent) if parent else None)
    out: Dict[str, Any] = {"code": code, "ebene": own.get("ebene", ebene), "titel": own.get("titel") or (reg.get(code) or {}).get("title") or code}
    if row is not None:
        state, via = force_state_of(row)
        out.update({"forcebar": bool(row.get("forcebar")), "force_state": state, "force_via": via,
                    "klasse": row.get("klass"), "klasse_grund": row.get("why_class"), "wired_at": row.get("wired_at"),
                    "force_scope": row.get("force_scope")})
        kons = own.get("konsequenz") or row.get("consequence") or ""
    else:
        out.update({"forcebar": None, "force_state": HINT, "force_via": None, "klasse": None, "klasse_grund": None, "wired_at": None,
                    "force_scope": None})
        kons = own.get("konsequenz") or ""
    if force_state is not None:
        out["force_state"] = force_state
    out["konsequenz"] = konsequenz if konsequenz is not None else kons
    out["grund"] = _clip(grund if grund is not None else text)
    out["text"] = str(text or "")
    out["werte"] = list(werte)
    if parent:
        out["parent"] = parent
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# what the launcher said -> verdicts
# ---------------------------------------------------------------------------

def _is_refusal_class(exc_type: str, exc_mro: Optional[Sequence[str]]) -> bool:
    """Is the exception a launcher refusal?  By the MRO when the harness sent it (``Weg2LaunchRefused`` and every subclass, ``SystemExit``
    refusals of the ``Weg2*`` family), else by the launcher's own naming (``Weg2*``, ``*Refused``, ``*Unproven``).  A stdlib exception
    (``KeyError``, ``IndexError``, ``FileNotFoundError`` ...) is never a refusal, whatever its text says."""
    names = [str(n) for n in (exc_mro or ())]
    if names:
        return any(n == "Weg2LaunchRefused" or n.startswith("Weg2") or n.endswith("Refused") or n.endswith("Unproven") for n in names)
    return exc_type.startswith("Weg2") or exc_type.endswith("Refused") or exc_type.endswith("Unproven")


def classify_exception(exc_type: Optional[str], exc_msg: str, exc_mro: Optional[Sequence[str]] = None) -> Dict[str, Optional[str]]:
    """The register code and the launcher W-code of the exception that ended a dry run.

    ``kind``: ``ablehnung`` (a launcher refusal: ``Weg2LaunchRefused`` and subclasses), ``optionen`` (argparse / ``SystemExit``), ``absturz``
    (anything that is not a refusal: ``IndexError``, ``KeyError``, ``FileNotFoundError``, ... -- a defect of the launcher for this form, not a
    judgement of values).  The W-code is read at the START of the message (``W10 Weg2DrafterIdentityMismatch: ...``); only a refusal may
    carry it further inside, and then only as a whole token (``Qwen3.8-27B-DFlash2-W8-lued`` is a path, not the code W8)."""
    from sglang.srt.weg2 import refusals

    msg = str(exc_msg or "")
    if exc_type is None:
        return {"kind": None, "code": None, "launcher_code": None}
    if exc_type == "SystemExit" or msg.startswith("usage:"):
        mw = _W_START_RX.match(msg)
        return {"kind": "optionen", "code": "OPTIONEN", "launcher_code": "W" + mw.group(1) if mw else None}
    if not _is_refusal_class(exc_type, exc_mro):
        return {"kind": "absturz", "code": "ORAKEL-ABSTURZ", "launcher_code": None}
    mw = _W_START_RX.match(msg) or _W_TOKEN_RX.search(msg)
    wcode = "W" + mw.group(1) if mw else None
    hw = refusals.classify(msg)
    if hw:
        return {"kind": "ablehnung", "code": hw, "launcher_code": wcode}
    if exc_type in _CLASS_CODE:
        return {"kind": "ablehnung", "code": _CLASS_CODE[exc_type], "launcher_code": wcode}
    if wcode in _W_CODE:
        return {"kind": "ablehnung", "code": _W_CODE[wcode], "launcher_code": wcode}
    if wcode in SUPPLEMENT_CODES:
        return {"kind": "ablehnung", "code": SUPPLEMENT_CODES[wcode]["code"], "launcher_code": wcode}
    return {"kind": "ablehnung", "code": "LAUNCHER-UNKLASSIFIZIERT", "launcher_code": wcode}


def blockers_of(text: str) -> List[Dict[str, str]]:
    """The ``[CODE] what (where)`` blockers a ``HW-COUNT`` / ``HW-TOPOLOGY`` text names (``topology.Blocker.text``, joined by ``"; "``),
    in order; ``where`` is the LAST parenthesis of an item."""
    t = str(text or "")
    k = t.find("blocker(s): ")
    if k < 0:
        k = t.find("Blockers: ")
        tail = t[k + len("Blockers: "):] if k >= 0 else ""
    else:
        tail = t[k + len("blocker(s): "):]
    tail = tail.split(" || ")[0]               # the launcher appends `` || visible: <the cards it saw>`` to the text
    out = []
    for item in _SPLIT_BLOCKERS_RX.split(tail.strip()):
        item = item.strip()
        if not item.startswith("[") or "]" not in item:
            continue
        code, rest = item[1:].split("]", 1)
        rest = rest.strip()
        where = ""
        if rest.endswith(")") and " (" in rest:
            rest, where = rest.rsplit(" (", 1)
            where = where[:-1]
        out.append({"code": code, "what": " ".join(rest.split()), "where": " ".join(where.split())})
    return out


def vector_keys_of(text: str) -> List[str]:
    """The ``name`` of every ``name (K entries)`` a PROFILE-VECTORS blocker names (the launcher's own names: the flag, or the token of a
    ``--extra-p/-d`` / ``--env-p/-d`` word without its group)."""
    t = str(text or "")
    k = t.find("vector): ")
    if k >= 0:
        t = t[k + len("vector): "):]
    t = t.split("; this launch has")[0]
    keys = []
    for item in t.split(", "):
        m = _VEC_ITEM_RX.match(item.strip())
        if m:
            keys.append(m.group(1))
    return sorted(set(keys))


def _run_summary(res: Any) -> Dict[str, Any]:
    cls = classify_exception(res.exc_type, res.exc_msg, getattr(res, "exc_mro", None))
    return {"rc": res.rc, "exc_type": res.exc_type, "exc_msg": _clip(res.exc_msg, 1200), "exc_where": getattr(res, "exc_where", ""),
            "code": cls["code"], "launcher_code": cls["launcher_code"], "kind": cls["kind"], "forced": [dict(f) for f in res.forced],
            "zeilen": res.text.count("\n")}


def plan_summary(res: Any) -> Dict[str, Any]:
    """The resolved values of the plan the launcher printed (``propose_oracle.parse_plan_dump``): cut, budgets, group flags, W-codes."""
    from sglang.srt.weg2 import propose_oracle as O

    p = O.parse_plan_dump(res.dump())
    return {"pp_cut": p["pp_cut"], "budgets": p["budgets"], "group_flags": p["group_flags"], "w_codes": p["w_codes"],
            "zeilen": len(p["lines"])}


def build_verdikt(n: int, first: Any, second: Any = None, *, lens: Optional[Mapping[str, int]] = None,
                  vorschlag: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The verdict document of one oracle question.

    ``first`` = the dry run WITHOUT force (what the launcher says as it is), ``second`` = the dry run WITH ``--force`` (every value
    refusal Force would pass, listed, and what still stops the start) or None when the first run needed none.  ``lens`` = the vector
    lengths of the launch (``propose.vector_lengths``), ``vorschlag`` = the AP-C proposal when the question was one."""
    final = second if second is not None else first
    fin = _run_summary(final)
    v_list: List[Dict[str, Any]] = []
    # (1) every refusal Force passed: the value refusals, in the order the launcher raised them
    for f in final.forced:
        code = str(f["code"])
        v_list.append(verdikt(code, ebene="lauf", text=f.get("text", ""), grund=f.get("text", ""), force_state=FORCE if (register_rows().get(code) or {}).get("forcebar") else None,
                              extra={"durchgelassen": True}))
        if code == "HW-COUNT":
            for b in blockers_of(f.get("text", "")):
                keys = vector_keys_of(b["what"]) if b["code"] == "PROFILE-VECTORS" else []
                v_list.append(verdikt(b["code"], ebene="blocker", text=b["what"], grund=b["what"], werte=keys, parent="HW-COUNT",
                                      force_state=FORCE, konsequenz=(None if b["code"] in OWN_CODES else b["what"]),
                                      extra={"wo": b["where"], "blocker": b["code"], "durchgelassen": True}))
    # (2) what ended the run
    if fin["kind"] is not None:
        code = fin["code"]
        extra = {"launcher_code": fin["launcher_code"], "exc_type": fin["exc_type"], "durchgelassen": False}
        if fin["kind"] == "absturz":
            extra["wo"] = fin["exc_where"]
            blk = [b for f in final.forced for b in blockers_of(f.get("text", ""))]
            if any(b["code"] == "RECORDS-NVEC" for b in blk):
                extra["hangt_an"] = "RECORDS-NVEC"
            txt = "%s: %s%s" % (fin["exc_type"], fin["exc_msg"], (" (" + fin["exc_where"] + ")") if fin["exc_where"] else "")
            v_list.append(verdikt("ORAKEL-ABSTURZ", ebene="absturz", text=txt, grund=txt, force_state=BLOCKED, extra=extra))
        else:
            row = register_rows().get(code)
            state = force_state_of(row)[0] if row else BLOCKED
            if row is not None and row.get("forcebar") and state == FORCE:
                state = BLOCKED            # a forceable code that STILL stopped the run: the run was not forced for it
            nennt = _NAMED_RX.search(fin["exc_msg"])
            if nennt and nennt.group(1) != code:
                extra["nennt"] = nennt.group(1)           # ``W19 dormant-residue reserve (HW-UNCALIBRATED): ...``: the launcher names the register code it belongs to
            v_list.append(verdikt(code, ebene="lauf", text=fin["exc_msg"], grund=fin["exc_msg"], force_state=state if row else BLOCKED,
                                  extra=extra))
    elif fin["rc"] not in (0, None):
        v_list.append(verdikt("LAUNCHER-UNKLASSIFIZIERT", ebene="lauf", text="Rueckgabewert %r ohne Ausnahme" % (fin["rc"],),
                              force_state=BLOCKED, extra={"durchgelassen": False}))
    # (3) the vectors of the launch: the launcher DERIVES a vector for a live subset of the cards where every card has a measured twin (the
    # HW gate lists ``PROFILE-VECTORS`` only for the vectors it cannot derive), so a vector that is not N entries long is DATA here (``vektoren``),
    # never a verdict of its own; a PROPOSAL that carries one is a defect of the planner's own result and is said so (below)
    n_bad = {k: int(c) for k, c in (lens or {}).items() if int(c) != int(n)}
    # (4) the proposal's own verdicts: fit bound, blockers of stage A, borrowed values
    if vorschlag:
        fit = vorschlag.get("fit") or {}
        lvl = str(fit.get("level") or "")
        if lvl:
            st = {"ja": GOES, "knapp": HINT, "nein": BLOCKED}.get(lvl, HINT)
            if vorschlag.get("form") == "dual":
                st = HINT          # hw_fit does not model the Dual: its Flip bound neither clears nor blocks it (DUAL-PASSUNG below does)
            txt = "hw_fit%s: %s%s%s" % (" (Flip-Schranke, Dual nicht modelliert)" if vorschlag.get("form") == "dual" else "", lvl, (" (Rest %s MiB)" % fit["margin_mib"]) if fit.get("margin_mib") is not None else "",
                                      ("; " + str(fit["first"])) if fit.get("first") else "")
            v_list.append(verdikt("FIT", ebene="fit", text=txt, grund=txt, force_state=st, extra={"stufe": lvl, "rest_mib": fit.get("margin_mib"),
                                                                                              "zeilen": list(fit.get("lines") or [])[:12]}))
        for m in fit.get("marks") or []:
            if str(m).startswith("HW-BORROWED") or "HW-BORROWED" in str(m):
                v_list.append(verdikt("HW-BORROWED", ebene="fit", text=str(m), grund=str(m), force_state=HINT, extra={"quelle": "hw_fit"}))
        # the Dual form: its own coupling (``propose_dual``), labelled as a Planer-Rechnung and never as hw_fit
        for dv in (vorschlag.get("dual") or {}).get("verdikte") or []:
            stufe = str(dv.get("stufe") or "")
            v_list.append(verdikt(str(dv["code"]), ebene="fit", text=str(dv.get("text", "")), grund=str(dv.get("text", "")),
                                  force_state={"ja": GOES, "nein": BLOCKED if dv["code"] == "DUAL-PASSUNG" else HINT}.get(stufe, HINT),
                                  extra={"stufe": stufe, "etikett": dv.get("etikett"), "rest_mib": dv.get("rest_mib")}))
        for b in vorschlag.get("blocker") or []:
            v_list.append(verdikt("PLANER", ebene="planer", text=str(b), grund=str(b), force_state=BLOCKED))
        falsch = {k: int(c) for k, c in (vorschlag.get("vektoren_falsch") or {}).items()}
        if falsch and not any(v["code"] == "PROFILE-VECTORS" for v in v_list):
            txt = "Der Vorschlag traegt Vektoren mit anderer Laenge als die Kartenzahl %d: %s" % (n, ", ".join("%s (%d)" % kv for kv in sorted(falsch.items())))
            v_list.append(verdikt("PROFILE-VECTORS", ebene="blocker", text=txt, grund=txt, werte=sorted(falsch), parent="HW-COUNT",
                                  extra={"durchgelassen": False, "wo": "propose.vector_lengths (Vorschlag)"}))
    doc = {"schema": SCHEMA, "n": int(n), "verdikte": v_list, "lauf": fin,
           "vektoren": {"laengen": {k: int(c) for k, c in (lens or {}).items()}, "nicht_n": n_bad},
           "ohne_force": _run_summary(first), "mit_force": _run_summary(second) if second is not None else None,
           "forced": [dict(f) for f in final.forced], "plan": plan_summary(final)}
    refus = [v for v in v_list if v["ebene"] in ("lauf", "absturz") or (v.get("parent") == "HW-COUNT" and v.get("durchgelassen"))]
    clean_first = first.exc_type is None and first.rc in (0, None) and not first.forced
    if clean_first:
        doc["ausgang"] = "geht"
    elif fin["kind"] == "absturz":
        doc["ausgang"] = "absturz"
    elif fin["kind"] is not None or fin["rc"] not in (0, None):
        doc["ausgang"] = "verweigert"
    else:
        doc["ausgang"] = "geht_mit_force"
    doc["geht"] = doc["ausgang"] == "geht"
    doc["geht_mit_force"] = doc["ausgang"] in ("geht", "geht_mit_force")
    doc["zaehlung"] = {s: sum(1 for v in refus if v["force_state"] == s) for s in (FORCE, BLOCKED, UNCHECKED)}
    return doc


def _tail(key: str) -> str:
    """The name of a value label without its group (``--extra-d --rank-moe-ratio`` -> ``--rank-moe-ratio``): the launcher names a vector by it."""
    return str(key).split()[-1].rstrip("=") if str(key).split() else str(key)


def slim_verdikt(v: Mapping[str, Any], grund: Optional[str] = None) -> Dict[str, Any]:
    """A verdict as it hangs on ONE value: without the (long) ``text`` and ``werte`` fields."""
    d = {k: v.get(k) for k in ("code", "ebene", "forcebar", "force_state", "konsequenz", "titel")}
    d["grund"] = _clip(grund if grund is not None else v.get("grund"), 400)
    return d


def werte_verdikte(vorschlag: Mapping[str, Any], verdikt_doc: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """``{wert-key: [verdikt, ...]}`` for every value of a proposal (``vorschlag["werte"]``): an empty list = the oracle has no objection.

    A value gets: ``PROFILE-VECTORS`` when it is a vector of another length than N; ``HW-UNCALIBRATED`` when it is a per-card class
    value and the launcher passed that refusal; ``HW-BORROWED`` when its origin says it was borrowed; ``UNBELEGT`` for any other value
    the planner marked unbelegt.  The verdicts are copies without the (long) ``text`` and ``werte`` fields."""
    vl = verdikt_doc.get("verdikte") or []
    by_code: Dict[str, Dict[str, Any]] = {}
    for v in vl:
        by_code.setdefault(v["code"], v)
    bad_keys = set()
    for v in vl:
        if v["code"] == "PROFILE-VECTORS":
            bad_keys |= set(v.get("werte") or [])
    bad_tails = {_tail(k) for k in bad_keys}
    uncal = by_code.get("HW-UNCALIBRATED")
    out: Dict[str, List[Dict[str, Any]]] = {}

    slim = slim_verdikt

    for w in vorschlag.get("werte") or []:
        key = str(w.get("key"))
        lst: List[Dict[str, Any]] = []
        if key in bad_keys or _tail(key) in bad_tails:
            lst.append(slim(by_code["PROFILE-VECTORS"], "Vektor mit anderer Laenge als die Kartenzahl: %s" % key))
        if uncal is not None and w.get("policy") == "class":
            lst.append(slim(uncal))
        herk = "%s %s" % (w.get("herkunft", ""), w.get("grund", ""))
        if w.get("zustand") == "unbelegt":
            if "geborgt" in herk or "HW-BORROWED" in herk:
                lst.append(slim(verdikt("HW-BORROWED", ebene="wert", text=herk, grund=w.get("herkunft"), force_state=HINT)))
            elif not lst:
                lst.append(slim(verdikt("UNBELEGT", ebene="wert", text=herk, grund=w.get("herkunft"), force_state=HINT)))
        out[key] = lst
    return out


# ---------------------------------------------------------------------------
# asking the oracle
# ---------------------------------------------------------------------------

def devices_from_cards(cards: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Replay rows (``propose_oracle.replay_row``) for synthetic dashboard cards: ``[{"entry": <card catalog row>, "link": {"bar1_mib",
    "effective": {"gen", "lanes"}}}, ...]``, NVML index = position.  Every field is a catalog value or the chosen link: the card is a
    DATASHEET card (its UUID is synthetic) -- the verdicts about it are not measurements."""
    from sglang.srt.weg2 import propose_oracle as O

    rows = []
    for i, c in enumerate(cards):
        e = c["entry"]
        link = c.get("link") or {}
        eff = link.get("effective") or {}
        rows.append(O.replay_row(
            i, uuid=O._fake_uuid("%d:%s" % (i, e.get("id") or e.get("nvml_name"))), name=str(e["nvml_name"]), total_mib=int(e["usable_mib"]),
            cc=e.get("cc"), bar1_total_mib=link.get("bar1_mib"), pcie_max_gen=eff.get("gen"), pcie_max_width=eff.get("lanes"),
            mem_bus_width_bits=e.get("bus_bits"), mem_clock_max_mhz=O.catalog_mem_clock_mhz(e.get("mem_bw_gbs"), e.get("bus_bits"))))
    return rows


def launch_input_of(argv: Sequence[str], env: Mapping[str, str], basis: Any, source: str = "") -> Any:
    """A ``LaunchInput`` of a proposal (its argv + env) on the variables of the profile it was derived from."""
    from sglang.srt.weg2 import propose_oracle as O

    return O.LaunchInput(list(argv), dict(env), dict(getattr(basis, "vars", {}) or {}), [], source or "propose:" + str(getattr(basis, "source", "")),
                         str(getattr(basis, "instruments", "0")))


def ask(li: Any, devices: Sequence[Mapping[str, Any]], *, tree: str, form: Optional[str] = None, vorschlag: Optional[Mapping[str, Any]] = None,
        snapshots: Optional[Mapping[str, str]] = None, evidence_dir: Optional[str] = None, scratch: Optional[str] = None,
        extra_args: Sequence[str] = (), **run_kw: Any) -> Dict[str, Any]:
    """Ask the launcher what it makes of the launch ``li`` (``propose_oracle.LaunchInput``) on ``devices``; returns the ``flliper.verdikt/1``
    document.  NEVER raises for a refusal or a crash of the dry run, nor for a failure of the harness: those are verdicts
    (``verweigert`` / ``absturz`` / ``orakel_fehler``).

    Runs the dry run as it is (no force); when the launcher refuses it runs it again with ``--force`` so that EVERY value refusal Force
    passes is listed and what still stops the start is named.  A launcher CRASH on the first run is final (a second run would crash the same)."""
    from sglang.srt.weg2 import propose as P
    from sglang.srt.weg2 import propose_oracle as O

    t0 = time.time()
    runs = 0
    n = len(devices)
    ident = profile_identity(li)
    head = {"schema": SCHEMA, "form": form, "n": n,
            "inventar": [{"index": d.get("index"), "name": d.get("name"), "total_mib": int(d.get("total_bytes", 0)) >> 20,
                          "cc": [d.get("cc_major"), d.get("cc_minor")]} for d in devices],
            "profil": ident, "argv_sha256": launch_hash(li.argv, li.env)}
    try:
        r1 = O.run_profile("", devices, tree=tree, force=False, launch_input=li, snapshots=snapshots, evidence_dir=evidence_dir,
                           scratch=scratch, extra_args=extra_args, **run_kw)
        res1, notes = r1.result, list(r1.notes)
        res2 = None
        runs = 1
        if res1.exc_type is not None and classify_exception(res1.exc_type, res1.exc_msg, getattr(res1, "exc_mro", None))["kind"] != "absturz":
            r2 = O.run_profile("", devices, tree=tree, force=True, launch_input=li, snapshots=snapshots, evidence_dir=evidence_dir,
                               scratch=scratch, extra_args=extra_args, **run_kw)
            res2 = r2.result
            runs = 2
        try:
            lens = P.vector_lengths(r1.argv, li.env)
        except Exception:  # noqa: BLE001 -- the vector count is a help, the dry run is the authority
            lens = {}
        doc = build_verdikt(n, res1, res2, lens=lens, vorschlag=vorschlag)
        doc["argv_final_sha256"] = launch_hash(r1.argv, li.env)
    except Exception as exc:  # noqa: BLE001 -- a failure of the harness is a verdict, never an exception
        doc = {"schema": SCHEMA, "n": n, "ausgang": "orakel_fehler", "geht": False, "geht_mit_force": False,
               "verdikte": [verdikt("ORAKEL-FEHLER", ebene="orakel", text="%s: %s" % (type(exc).__name__, exc), force_state=BLOCKED)],
               "lauf": None, "ohne_force": None, "mit_force": None, "forced": [], "plan": {}, "zaehlung": {FORCE: 0, BLOCKED: 1, UNCHECKED: 0}}
        notes = []
    doc.update({k: v for k, v in head.items() if k not in doc})
    doc["orakel"] = {"version": oracle_version(), "dauer_s": round(time.time() - t0, 2), "notizen": notes,
                     "laeufe": 0 if doc["ausgang"] == "orakel_fehler" else runs}
    return doc


# ---------------------------------------------------------------------------
# propose + ask (the worker's job)
# ---------------------------------------------------------------------------

def _basis_from(req: Mapping[str, Any], scratch: str) -> Any:
    """The profile the proposal is based on, as a ``LaunchInput``: ``basis.env_path`` (a release profile file) or ``basis.env_text``
    (a profile the editor rendered); evaluated by bash like the entrypoint does (``propose_oracle.profile_launch_input``)."""
    from sglang.srt.weg2 import propose_oracle as O

    b = req.get("basis") or {}
    kw = {"instruments": str(req.get("instruments", "0"))}
    if b.get("env_path"):
        return O.profile_launch_input(str(b["env_path"]), **kw)
    if b.get("env_text") is not None:
        p = os.path.join(scratch, "basis.env")
        with open(p, "w", encoding="utf-8", errors="surrogateescape") as fh:
            fh.write(str(b["env_text"]))
        li = O.profile_launch_input(p, **kw)
        if b.get("source"):
            li.source = str(b["source"])
        return li
    raise ValueError("basis braucht env_path oder env_text")


def _devices_of(req: Mapping[str, Any]) -> List[Dict[str, Any]]:
    from sglang.srt.weg2 import propose_oracle as O

    inv = req.get("inventar") or {}
    if inv.get("hardware"):
        return O.replay_from_hardware_profile(inv["hardware"])
    if inv.get("devices"):
        return [dict(d) for d in inv["devices"]]
    if inv.get("cards"):
        return devices_from_cards(inv["cards"])
    raise ValueError("inventar braucht hardware, devices oder cards")


def _model_profiles(li: Any, req: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
    """``(modell, draft, notes)`` of the profile's model and draft directory (read like the oracle reads them: an empty mount point is
    replaced by its header snapshot / sibling stub, ``propose_oracle.ensure_model_dir``)."""
    from sglang.srt.weg2 import model_profile as MP
    from sglang.srt.weg2 import propose_oracle as O

    notes: List[str] = []
    snaps = dict(req.get("snapshots") or {})
    sib = O.DEFAULT_MODEL_SIBLINGS

    def stub(path: str) -> str:
        bn = os.path.basename(path.rstrip("/"))
        try:
            return O.ensure_model_dir(path, siblings=sib.get(bn, ()), farm_root=O.DEFAULT_FARM_ROOT, snapshot=snaps.get(bn, ""))
        except FileNotFoundError as exc:
            notes.append(str(exc))
            return path

    mpath = str(req.get("model_path") or li.model or "")
    modell = None
    if mpath:
        mp = MP.estimate_or_state(stub(mpath))
        if mp.get("ok"):
            modell = mp["profile"]
        else:
            notes.append("Modellprofil: %s (%s)" % (mp.get("reason") or mp.get("state") or mp, mp.get("state")))
    draft = None
    dpath = str(req.get("draft_path") or li.draft or "")
    if dpath:
        try:
            draft = MP.estimate_draft(stub(dpath))
        except Exception as exc:  # noqa: BLE001 -- no draft profile: propose() runs without the draft term, and says so
            notes.append("Draft-Profil nicht lesbar (%s: %s)" % (type(exc).__name__, exc))
    return modell, draft, notes


# ---------------------------------------------------------------------------
# the SINGLE CARD form (AP-F): no launcher, so no oracle run -- the verdicts are a Planer-Rechnung and say so
# ---------------------------------------------------------------------------

#: the names of the single-card form in a request (the dashboard sends ``single``, the plan says ``einzel``)
SINGLE_FORMS = ("single", "einzel", "einzelkarte")
#: dashboard goal -> goal of ``propose_single`` (the "Kontext" regulator of the page is the context per request); the rest of the dashboard's goals
#: (p_cut, d_objective, draft_kv_on_p, force_rules) are group-P/D knobs and do not exist for one card
_SINGLE_GOALS = (("seats", "seats"), ("kv_tokens", "context_tokens"), ("kv_dtype", "kv_dtype"), ("draft", "draft"), ("host_ram_mib", "host_ram_mib"),
                 ("pre_load_free_mib", "pre_load_free_mib"), ("reserve_mib", "reserve_mib"))
_SINGLE_STATE = {"passt": ("passt", GOES), "passt nicht": ("passt_nicht", BLOCKED), "unbelegt": ("unbelegt", HINT)}


class _NoLaunch:
    """What ``_model_profiles`` reads of a launch input when the single-card request has no base profile."""

    def __init__(self, model: str = "") -> None:
        self.model, self.draft = model, ""


def _single_card(req: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    """``(card for propose_single, note)``: the ONE card of the request.  ``inventar.hardware`` takes card ``inventar.karte`` (ordinal, default 0) of the
    hardware profile; ``devices`` / ``cards`` must be exactly one."""
    from sglang.srt.weg2 import propose_single as PS

    inv = req.get("inventar") or {}
    if inv.get("hardware"):
        ordinal = int(inv.get("karte") or 0)
        card = PS.card_from_hardware(inv["hardware"], ordinal)
        return card, "Karte %d des Hardwareprofils: %s, %d MiB (%s)" % (ordinal, card["name"], card["total_mib"], card["total_src"])
    rows = _devices_of(req)
    if len(rows) != 1:
        raise PS.ProposeSingleError("die Einzelkarte braucht genau eine Karte, nicht %d" % len(rows))
    d = rows[0]
    total = int(d.get("total_bytes", 0)) >> 20
    card = PS.normalize_card({"name": d.get("name"), "total_mib": total, "total_src": "Kartenkatalog (usable_mib, Datenblatt-Eintrag der Karte)",
                              "cc": [d.get("cc_major"), d.get("cc_minor")]})
    return card, "Karte: %s, %d MiB (Kartenkatalog)" % (card["name"], total)


def _single_werte(p: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The flags of ``propose_single`` as the list of values of the propose document (``propose.py`` ``werte``): there is no profile before, so every value that
    is set is new (``geaendert``); a switch is the empty value (like ``--d-only``)."""
    out = []
    for e in p["flags"]:
        wert = e["wert"]
        sval = None if wert is None else ("" if wert is True else str(wert))
        out.append({"key": e["flag"], "group": "-", "policy": "single", "alt": None, "wert": sval, "eintraege": 1, "zustand": e["zustand"],
                    "herkunft": e["herkunft"], "grund": e["begruendung"], "in_argv": wert is not None, "geaendert": wert is not None})
    return out


def single_verdikt(p: Mapping[str, Any], card_note: str, notes: Sequence[str], profil: Mapping[str, Any], t0: float) -> Dict[str, Any]:
    """The ``flliper.verdikt/1`` document of a single-card proposal: the fit as a Planer-Rechnung, the failing checks, the ServerArgs parse.  There is NO
    launcher run (``lauf`` / ``ohne_force`` / ``mit_force`` are None, ``laeufe`` 0) and no Force (nothing here is a register code)."""
    v = p["verdikt"]
    ausgang, st = _SINGLE_STATE.get(v["state"], ("unbelegt", HINT))
    fit = p["fit"]
    v_list: List[Dict[str, Any]] = [verdikt("EINZEL-PASSUNG", ebene="fit", text=v["text"], grund=v["text"], force_state=st,
                                            extra={"stufe": {"passt": "ja", "passt nicht": "nein"}.get(v["state"], "unbelegt"), "art": v.get("art"),
                                                   "etikett": "Planer-Rechnung", "rest_mib": fit.get("frei_mib")})]
    for ch in fit.get("checks") or []:
        if not ch["ok"]:
            v_list.append(verdikt(str(ch["code"]), ebene="fit", text=ch["text"], grund=ch["text"], force_state=BLOCKED, extra={"etikett": "Planer-Rechnung"}))
    for u in fit.get("unbelegt") or []:
        v_list.append(verdikt("UNBELEGT", ebene="fit", text=str(u), grund=str(u), force_state=HINT, extra={"etikett": "Planer-Rechnung"}))
    par = p.get("parse") or {}
    if par.get("available") and par.get("ok") is False:
        v_list.append(verdikt("EINZEL-PARSE", ebene="fit", text="ServerArgs-Parse verweigert: %s" % par.get("error"), grund=str(par.get("error")), force_state=BLOCKED))
    elif par.get("available") and par.get("ok"):
        v_list.append(verdikt("EINZEL-PARSE", ebene="fit", text="ServerArgs-Parse ok (argparse; __post_init__ nicht gelaufen)", force_state=GOES))
    elif par:
        v_list.append(verdikt("EINZEL-PARSE", ebene="fit", text="ServerArgs-Parse nicht geprueft: %s" % (par.get("error") or "nicht verfuegbar"), force_state=HINT))
    n_block = sum(1 for x in v_list if x["force_state"] == BLOCKED)
    argv = list(p["argv"])
    return {"schema": SCHEMA, "n": 1, "form": "einzel", "art": v.get("art"), "verdikte": v_list, "lauf": None,
            "vektoren": {"laengen": {}, "nicht_n": {}}, "ohne_force": None, "mit_force": None, "forced": [], "plan": {},
            "ausgang": ausgang, "geht": ausgang == "passt", "geht_mit_force": ausgang == "passt",
            "zaehlung": {FORCE: 0, BLOCKED: n_block, UNCHECKED: 0}, "profil": dict(profil), "argv_sha256": launch_hash(argv, {}),
            "inventar": [{"index": 0, "name": p["card"]["name"], "total_mib": int(p["card"]["total_mib"])}],
            "orakel": {"version": {"propose_single": (file_sha256(_single_file()) or "")[:16] or None}, "dauer_s": round(time.time() - t0, 2),
                       "notizen": [card_note] + list(notes), "laeufe": 0, "art": v.get("art")}}


def _num(x: Any) -> Any:
    return int(x) if isinstance(x, str) and x.isdigit() else x


def _single_file() -> str:
    from sglang.srt.weg2 import propose_single as PS

    return getattr(PS, "__file__", "") or ""


def single_je_wert(p: Mapping[str, Any], vorschlag: Mapping[str, Any], verd: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Verdicts per flag: ``werte_verdikte`` (borrowed / unbelegt values) plus what the fit check and the parse said about the flag itself."""
    out = werte_verdikte(vorschlag, verd)
    for e in p["flags"]:
        vd = e["verdikt"]
        lst = out.setdefault(e["flag"], [])
        if vd["state"] == "verweigert" and vd.get("code"):
            for code in str(vd["code"]).split("+"):
                lst.append(slim_verdikt(verdikt(code, ebene="fit", text=vd["grund"], grund=vd["grund"], force_state=BLOCKED)))
        if vd.get("parse") == "veraltet (Alias)":
            lst.append(slim_verdikt(verdikt("EINZEL-PARSE", ebene="fit", text="ServerArgs kennt dieses Flag nur als veralteten Alias", force_state=HINT)))
    return out


def run_propose_single(req: Mapping[str, Any]) -> Dict[str, Any]:
    """propose() of the form single card (``propose_single``, plan row AP-F) in the shape of ``run_propose``: ``vorschlag`` (the values in the list of
    ``propose.py``), ``verdikt`` (Planer-Rechnung, no launcher run), ``je_wert``, ``launch`` ({argv, env: {}}: the arguments of ``python -m sglang.launch_server``).
    Model: ``model_path`` or the base profile's ``PROFILE_MODEL``; a draft only when ``draft_path`` is given (the base profile's draft is a weg2 draft)."""
    from sglang.srt.weg2 import propose_single as PS

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="apd-") as scratch:
        li = _basis_from(req, scratch) if req.get("basis") else None
        mreq = dict(req)
        mreq["draft_path"] = req.get("draft_path") or ""
        modell, draft, notes = _model_profiles(_NoLaunch(str(getattr(li, "model", "") or "")), mreq)
        if modell is None:
            return {"ok": False, "error": "kein Modellprofil: %s" % ("; ".join(notes) or "Modellpfad fehlt"), "notizen": notes}
        z = dict(req.get("ziele") or {})
        goals = {dst: z[src] for src, dst in _SINGLE_GOALS if z.get(src) not in (None, "")}
        ignored = sorted(k for k in z if k not in {s for s, _ in _SINGLE_GOALS} and z.get(k) not in (None, "", False))
        if ignored:
            notes.append("Ziele ohne Bedeutung fuer eine Karte (ignoriert): %s" % ", ".join(ignored))
        try:
            card, card_note = _single_card(req)
            p = PS.propose_single(modell, card, goals, draft_profile=draft)
        except (PS.ProposeSingleError, ValueError, KeyError) as exc:
            return {"ok": False, "error": "Einzelkarte: %s" % exc, "notizen": notes}
        if req.get("parse", True):
            p = PS.check(p)
        else:
            p["parse"] = {"available": False, "ok": None, "error": "nicht angefragt"}
        werte = _single_werte(p)
        fit = p["fit"]
        lvl = {"passt": "ja", "passt nicht": "nein"}.get(p["verdikt"]["state"], "unbelegt")
        flag_of = {w["key"]: w["wert"] for w in werte}.get          # what the proposal settled on (the goals may be defaults)
        hinweise = [str(r.get("grund") or r.get("schritt")) for r in p.get("relaxations") or []] + list(fit.get("hinweise") or [])
        vorschlag = {"schema": "flliper.propose-a/1", "form": "einzel", "n": 1, "werte": werte,
                     "cards": [{"name": p["card"]["name"], "total_mib": int(p["card"]["total_mib"]), "tflops_src": None}],
                     "inventory": {"gleich_wie_profil": False, "n": 1}, "seeds": {},
                     "fit": {"level": lvl, "first": p["verdikt"]["text"], "margin_mib": fit.get("frei_mib"), "lines": [], "marks": [],
                             "art": p["verdikt"].get("art") or PS.VERDICT_ART},
                     "ziele": {"seats": _num(flag_of("--max-running-requests")), "kv_tokens": _num(flag_of("--context-length")), "kv_dtype": flag_of("--kv-cache-dtype")},
                     "unbelegt": list(p.get("unbelegt") or []), "hinweise": hinweise, "blocker": [], "vektorlaengen": {}, "vektoren_ok": True,
                     "vektoren_falsch": {}, "basis": os.path.basename(str(getattr(li, "source", "") or "")) or "(kein Profil)",
                     "argv": list(p["argv"]), "env": {}, "einzelkarte": p}
        profil = dict(profile_identity(li), rolle="basis") if li is not None else {"rolle": "keines", "quelle": "", "datei_sha256": None, "eingabe_sha256": None}
        profil["vorschlag_sha256"] = launch_hash(p["argv"], {})
        verd = single_verdikt(p, card_note, notes, profil, t0)
        return {"ok": True, "schema": PROPOSE_SCHEMA, "vorschlag": vorschlag, "verdikt": verd, "je_wert": single_je_wert(p, vorschlag, verd),
                "launch": {"argv": list(p["argv"]), "env": {}}, "notizen": notes}


def run_propose(req: Mapping[str, Any], *, tree: str) -> Dict[str, Any]:
    """propose() (stage A) + the oracle (stage B) + the verdicts per value (stage C) for one request of the dashboard worker.

    ``req``: ``basis`` {env_path | env_text}, ``inventar`` {hardware | devices | cards}, ``form`` flip|tp|dual|single, ``ziele``, ``model_path``,
    ``draft_path``, ``snapshots`` {registry name: header snapshot dir}, ``instruments``.  Returns ``{"ok": True, "schema": PROPOSE_SCHEMA,
    "vorschlag": <flliper.propose-a/1>, "verdikt": <flliper.verdikt/1>, "je_wert": {key: [verdikt]}, "launch": {argv, env}}`` or
    ``{"ok": False, "error": ...}`` for a request the planner cannot answer at all (no model profile, wrong form)."""
    from sglang.srt.weg2 import propose as P

    if str(req.get("form") or "").lower() in SINGLE_FORMS:
        return run_propose_single(req)                     # N=1: no launcher, no oracle run (topology.py MIN_CARDS=2)
    with tempfile.TemporaryDirectory(prefix="apd-") as scratch:
        li = _basis_from(req, scratch)
        devices = _devices_of(req)
        modell, draft, notes = _model_profiles(li, req)
        if modell is None:
            return {"ok": False, "error": "kein Modellprofil: %s" % ("; ".join(notes) or "Modellpfad fehlt"), "notizen": notes}
        try:
            v = P.propose(devices, modell, str(req.get("form") or "flip"), dict(req.get("ziele") or {}), basis=li, draft=draft,
                          rates=req.get("rates"))
        except P.ProposeError as exc:
            return {"ok": False, "error": str(exc), "notizen": notes}
        pli = launch_input_of(v["argv"], v["env"], li, "propose:" + v["basis"])
        verd = ask(pli, devices, tree=tree, form=str(v["form"]), vorschlag=v, snapshots=req.get("snapshots"), scratch=None)
        verd["profil"] = dict(profile_identity(li), **{"rolle": "basis"})
        verd["profil"]["vorschlag_sha256"] = launch_hash(v["argv"], v["env"])
        return {"ok": True, "schema": PROPOSE_SCHEMA, "vorschlag": v, "verdikt": verd, "je_wert": werte_verdikte(v, verd),
                "launch": {"argv": list(v["argv"]), "env": dict(v["env"])}, "notizen": notes}


def run_verdikt(req: Mapping[str, Any], *, tree: str) -> Dict[str, Any]:
    """The oracle for ONE launch (no proposal): ``basis`` {env_path | env_text} on ``inventar``; the dashboard's dry run."""
    with tempfile.TemporaryDirectory(prefix="apd-") as scratch:
        li = _basis_from(req, scratch)
        devices = _devices_of(req)
        return {"ok": True, "verdikt": ask(li, devices, tree=tree, form=req.get("form"), snapshots=req.get("snapshots"))}
