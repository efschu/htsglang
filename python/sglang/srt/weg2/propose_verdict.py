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

#: launcher exception CLASS name -> register code (the class names of ``launcher.py`` / ``weg2/__init__``); a name not here is judged by
#: its W-code or stays ``LAUNCHER-UNKLASSIFIZIERT``
_CLASS_CODE = {"Weg2PPCutRefused": "PP-CUT", "Weg2DepthUnfunded": "PP-CUT", "Weg2DepthGapped": "PP-CUT"}
#: launcher W-code -> register code (``refusals.REGISTER`` row ``PP-CUT``: "W40/W42/W43")
_W_CODE = {"W40": "PP-CUT", "W42": "PP-CUT", "W43": "PP-CUT"}

_SPLIT_BLOCKERS_RX = re.compile(r";\s+(?=\[[A-Z])")
_VEC_ITEM_RX = re.compile(r"^(\S.*?) \((\d+) entries\)$")
_W_RX = re.compile(r"\bW(\d+[a-z]?)\b")
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

def classify_exception(exc_type: Optional[str], exc_msg: str) -> Dict[str, Optional[str]]:
    """The register code and the launcher W-code of the exception that ended a dry run.

    ``kind``: ``ablehnung`` (a launcher refusal), ``optionen`` (argparse / ``SystemExit``), ``absturz`` (anything that is not a refusal:
    ``IndexError``, ``KeyError``, ``AssertionError``, ... -- a defect of the launcher for this form, not a judgement of values)."""
    from sglang.srt.weg2 import refusals

    msg = str(exc_msg or "")
    mw = _W_RX.search(msg)
    wcode = "W" + mw.group(1) if mw else None
    if exc_type is None:
        return {"kind": None, "code": None, "launcher_code": None}
    if exc_type == "SystemExit" or msg.startswith("usage:"):
        return {"kind": "optionen", "code": "OPTIONEN", "launcher_code": wcode}
    hw = refusals.classify(msg)
    if hw:
        return {"kind": "ablehnung", "code": hw, "launcher_code": wcode}
    if exc_type in _CLASS_CODE:
        return {"kind": "ablehnung", "code": _CLASS_CODE[exc_type], "launcher_code": wcode}
    if wcode in _W_CODE:
        return {"kind": "ablehnung", "code": _W_CODE[wcode], "launcher_code": wcode}
    if exc_type.startswith("Weg2") or exc_type.endswith("Refused") or exc_type.endswith("Unproven") or wcode:
        return {"kind": "ablehnung", "code": "LAUNCHER-UNKLASSIFIZIERT", "launcher_code": wcode}
    return {"kind": "absturz", "code": "ORAKEL-ABSTURZ", "launcher_code": wcode}


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
    cls = classify_exception(res.exc_type, res.exc_msg)
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
            txt = "hw_fit: %s%s%s" % (lvl, (" (Rest %s MiB)" % fit["margin_mib"]) if fit.get("margin_mib") is not None else "",
                                      ("; " + str(fit["first"])) if fit.get("first") else "")
            v_list.append(verdikt("FIT", ebene="fit", text=txt, grund=txt, force_state=st, extra={"stufe": lvl, "rest_mib": fit.get("margin_mib"),
                                                                                              "zeilen": list(fit.get("lines") or [])[:12]}))
        for m in fit.get("marks") or []:
            if str(m).startswith("HW-BORROWED") or "HW-BORROWED" in str(m):
                v_list.append(verdikt("HW-BORROWED", ebene="fit", text=str(m), grund=str(m), force_state=HINT, extra={"quelle": "hw_fit"}))
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

    def slim(v: Mapping[str, Any], grund: Optional[str] = None) -> Dict[str, Any]:
        d = {k: v.get(k) for k in ("code", "ebene", "forcebar", "force_state", "konsequenz", "titel")}
        d["grund"] = _clip(grund if grund is not None else v.get("grund"), 400)
        return d

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
        if res1.exc_type is not None and classify_exception(res1.exc_type, res1.exc_msg)["kind"] != "absturz":
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


def run_propose(req: Mapping[str, Any], *, tree: str) -> Dict[str, Any]:
    """propose() (stage A) + the oracle (stage B) + the verdicts per value (stage C) for one request of the dashboard worker.

    ``req``: ``basis`` {env_path | env_text}, ``inventar`` {hardware | devices | cards}, ``form`` flip|tp, ``ziele``, ``model_path``,
    ``draft_path``, ``snapshots`` {registry name: header snapshot dir}, ``instruments``.  Returns ``{"ok": True, "schema": PROPOSE_SCHEMA,
    "vorschlag": <flliper.propose-a/1>, "verdikt": <flliper.verdikt/1>, "je_wert": {key: [verdikt]}, "launch": {argv, env}}`` or
    ``{"ok": False, "error": ...}`` for a request the planner cannot answer at all (no model profile, wrong form)."""
    from sglang.srt.weg2 import propose as P

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
