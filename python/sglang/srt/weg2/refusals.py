"""PROFIL-EDITOR S1: the refusal register and the ONE Force switch.

Nutzer-Entscheid 03.10. ~20:15Z (verbatim): "das dashboard soll ein profil erstellen, beim serverstart soll der
user ein profil angeben das geladen wird und es soll einen force flag beim serverstart geben dass die ablehnung
der werte aufhebt und trotzdem startet".

* ONE switch, no list of codes: ``--force`` on the launcher (the container entrypoint maps ``FLLIPER_FORCE=1``
  / ``HTSGLANG_FORCE=1`` to it).  It lifts every VALUE refusal that is wired through :func:`refuse_value`;
  each lifted refusal is printed as ``FORCED-PAST <CODE> <reason>``; and a forced boot WRITES NO RECORDS
  (``host_ledger.append_measured_record`` returns without writing): what it measures lies outside the
  promise the planner made, and must never calibrate a later boot.
* Two classes, each with its reason (see :data:`REGISTER`):
    - ``wert``            a VALUE refusal: the planner or the launcher judges numbers (capacity, VRAM, host
                          memory, calibration of the inventory, card count).  Force starts with these numbers.
    - ``nicht_forcebar``  NOT a value judgement: someone else's occupation of a card or of /dev/shm, a missing or
                          broken file, an impossible build.  Force does not touch these.
* ``wired`` says whether the launcher of THIS tree consults the switch for that code today
  (:func:`wired_codes` greps ``refuse_value("CODE"`` in ``launcher.py``; a test pins the register to it, so the
  register cannot claim more than the launcher does).  A ``wert`` code that is not wired yet is listed as such
  and keeps refusing.

PURE: stdlib only.  The dashboard loads this file by path to show, per profile, which refusals the planner would
raise and whether Force passes them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: set by the launcher in ITS environment when ``--force`` is on, so the front and the groups (separate processes)
#: inherit it; read by :func:`forced_boot`
ENV_FORCED_BOOT = "SGLANG_WEG2_FORCED_BOOT"
#: container side: the entrypoint turns ``FLLIPER_FORCE=1`` (mapped to this) into ``--force``
ENV_FORCE_REQUEST = "HTSGLANG_FORCE"
MARKER = "FORCED-PAST"

CLASS_VALUE = "wert"
CLASS_HARD = "nicht_forcebar"
CLASS_LABEL = {CLASS_VALUE: "Wert-Ablehnung (forcebar)", CLASS_HARD: "nicht forcebar"}


@dataclass(frozen=True)
class Refusal:
    code: str
    klass: str               # CLASS_VALUE | CLASS_HARD
    title: str
    why_class: str           # the reason for the classification (Nutzer: "begründe jede Zuordnung")
    source: str              # where it is raised
    enforced_by: str         # launcher | entrypoint | planner-gate
    consequence: str = ""    # what a forced start means for this code

    @property
    def forcebar(self) -> bool:
        return self.klass == CLASS_VALUE


def _v(code, title, why, source, by, consequence):
    return Refusal(code, CLASS_VALUE, title, why, source, by, consequence)


def _h(code, title, why, source, by):
    return Refusal(code, CLASS_HARD, title, why, source, by, "")


REGISTER: Tuple[Refusal, ...] = (
    # ------------------------------------------------------------------ Wert-Ablehnungen
    _v("HW-COUNT", "Kartenzahl ist nicht die bewiesene / nicht zum Profil",
       "Das Profil nennt die Kartenzahl (PROFILE_CARD_COUNT, Topologie P = PPn / D = TPn) bzw. der Planer kennt fuer N nur den Beweisstand "
       "(topology.plan_topology, blockers); er vergleicht eine Zahl mit der sichtbaren Zahl. Ein Wert-Urteil ueber Zahlen und Beweisstand, "
       "kein Fehler der Hardware.",
       "weg2/topology.py plan_topology (HW-COUNT mit Blockern); weg2/card_identity.py order_cards; launcher.topology_check_line, launcher.order_cards", "launcher",
       "Der Start läuft mit den sichtbaren Karten. Die genannten Blocker (Positions-Vektoren, BAR1-Fenster, PP-Schnitt-Boden, Records) bleiben bestehen; "
       "die nächste Verweigerung kommt aus dem ersten davon, den der Launcher prüft."),
    _v("HW-UNCALIBRATED", "Karteninventar ist nicht das gemessene",
       "Die Positions-Messwerte des Profils (Budgets, Raten, Rest) gelten für ein anderes Inventar. Der Planer weigert sich, "
       "fremde Messungen zu leihen. Das ist ein Urteil über die Güte der Zahlen, nicht über die Möglichkeit zu starten.",
       "weg2/card_identity.py uncalibrated_message; launcher.inventory_check_line", "launcher",
       "Die Zahlen des Profils werden auf das fremde Inventar angewendet, als wären sie dafür gemessen (Boot-Log trägt FORCED-PAST). "
       "Zu erwarten: Fehlbudgets, im schlechten Fall OOM beim Laden oder Graph-Aufbau."),
    _v("HOST-MEM", "Host-Speicher unter der Schwelle",
       "Eine Schwelle (host_ledger-Preflight, MemAvailable >= 40 GiB) gegen einen gemessenen Wert: Kapazität, kein Fehler.",
       "launcher.host_preflight", "launcher",
       "Der Start läuft mit weniger freiem Host-Speicher. Zu erwarten: OOM-Kill des Containers, wenn der Cache wächst."),
    _v("D-BUDGET", "Gruppe D: Speicherplan unlösbar (Rang-Solver)",
       "Der D-Rang-Solver hat gerechnet und meldet, dass seine Zielgrößen (KV, Experten, Reste) nicht in die Karten passen: "
       "Kapazität, Zahlen bleiben für den Start verfügbar.",
       "launcher (D_RANK_SOLVE, plan.refusal)", "launcher",
       "D startet mit dem gerechneten, nicht passenden Plan. Zu erwarten: OOM bei Pool-Anlage oder Graphen."),
    _v("WAKE-CREDIT", "Aufwach-Kredit reicht nicht (P↔D-Wechsel)",
       "Der Wake-Credit-Solver gibt ein Kapazitätsurteil über den Gruppenwechsel (Zeit/Bytes je Karte); die Zahlen existieren.",
       "launcher (wake_credit plan.refusal / log_wake_credit_solve_pd)", "launcher",
       "Der Wechsel P↔D läuft mit unzureichendem Kredit. Zu erwarten: Wartezeit oder OOM im Flip."),
    _v("P-CARD", "Gruppe P: Karten-Budget reicht nicht (Chunk/Experten)",
       "Der P-Karten-Solver urteilt, dass der Chunk mit den Experten-Brüchen nicht in die Karte passt: Kapazitätsurteil über Zahlen.",
       "launcher.p_card_verdict (p_card_refusal_text)", "launcher",
       "P startet mit dem nicht passenden Chunk-/Experten-Plan. Zu erwarten: OOM im ersten Prefill-Chunk."),
    _v("PP-CUT", "PP-Schnitt / Pipeline-Tiefe nicht finanzierbar",
       "W40/W42/W43: der Schnitt-Solver findet keinen Schnitt unter den Pool-Böden bzw. die Tiefe wird nicht vom KV-Pool getragen: Zahlenurteil.",
       "launcher (Weg2PPCutRefused, Weg2DepthUnfunded, Weg2DepthGapped)", "launcher",
       "Noch nicht verdrahtet: Es gibt keinen Plan, mit dem weitergestartet werden könnte (der Solver liefert keinen Schnitt)."),
    _v("PROFIL-STATUS", "Profil ist nicht abgenommen",
       "PROFILE_STATUS (experimentell/formnachweis/vorbereitet/geplant) sagt, wie weit das Profil bewiesen ist: Urteil über den Beweisstand.",
       "docker/entrypoint.sh (PROFILE_STATUS)", "entrypoint",
       "Wie HTSGLANG_ALLOW_EXPERIMENTAL=1, aber auch für vorbereitet/geplant. Der Entrypoint liegt außerhalb des Repos (gestagter Patch)."),
    _v("SHM", "/dev/shm zu klein",
       "Schwelle (PROFILE_SHM_MIN_GIB) gegen die Größe des gemounteten shm: Kapazität.",
       "docker/entrypoint.sh (SHM)", "entrypoint",
       "Der Start läuft mit kleinerem shm. Zu erwarten: Bus-/Allokationsfehler beim Anlegen der Arena."),
    _v("STORE", "Experten-Store (tmpfs) zu klein oder nicht tmpfs",
       "Schwelle (PROFILE_TMPFS_GIB) gegen die Größe des Store-Mounts: Kapazität.",
       "docker/entrypoint.sh (STORE)", "entrypoint",
       "Der Start läuft mit zu kleinem Store. Zu erwarten: Absturz bei cudaHostRegister bzw. voller Store."),
    _v("MEMAVAIL", "MemAvailable unter PROFILE_MEMAVAIL_MIN_GIB",
       "Schwelle des Profils gegen den gemessenen Host-Wert: Kapazität.",
       "docker/entrypoint.sh (MEMORY)", "entrypoint",
       "Der Start läuft mit weniger freiem Host-Speicher (siehe HOST-MEM)."),
    # ------------------------------------------------------------------ nicht forcebar
    _h("HW-TOPOLOGY", "Kartenzahl ausserhalb von 2..8: es gibt gar keine Flip-Topologie",
       "Keine Wert-Ablehnung: topology.plan_topology kennt fuer N ausserhalb [MIN_CARDS, MAX_CARDS_BAR1] keine Topologie (P/D-Form, BAR1-Fenster, "
       "Pipeline), es gibt nichts, womit weitergestartet werden koennte. Ein N innerhalb des Bereichs, das nur noch nicht bewiesen ist, heisst "
       "HW-COUNT und ist forcebar.",
       "weg2/topology.py plan_topology; launcher.topology_check_line", "launcher"),
    _h("HW-ARCH", "Compute-Capability ohne Kernel im Image",
       "Keine Wert-Ablehnung: das Image trägt Code für sm_86 und sm_120 (Wheel 86;120a) und lässt sm_89 über die Binärkompatibilität "
       "der sm_86-Cubins plus JIT zu (sm_89 ist unkalibriert, siehe HW-UNCALIBRATED). Jede andere Architektur (sm_80, sm_90, sm_100, "
       "sm_121, nicht gemeldete cc) hat keinen ausführbaren Kernel, und der Entrypoint verweigert sie schon heute ausdrücklich immer. "
       "Force würde einen Absturz im ersten Kernel statt einer Meldung liefern.",
       "weg2/card_identity.py arch_gate; launcher.resolve_cards", "launcher"),
    _h("KARTE-BELEGT", "Fremder Prozess oder fremdes Fenster auf der Karte",
       "Keine Wert-Ablehnung, sondern Schutz anderer Nutzer: die Belegungsprüfung (NVML-Fremdnutzung, gpuq-Fenster) darf nie "
       "übergangen werden, sonst trifft der Start die Daten eines anderen.",
       "launcher.cards_free_check; gpuq-Fenster", "launcher"),
    _h("SHM-BELEGT", "Lebender Halter auf eigenen /dev/shm-Einträgen",
       "Fremder lebender Prozess hält Einträge des Launchers: Belegung, nicht Wert. Wegräumen würde ihn zerstören.",
       "launcher.shm_residue_sweep (#1217/#1233)", "launcher"),
    _h("PORT-BELEGT", "Front-Port ist schon gebunden",
       "Echter Fehler: ein anderer Listener hält den Port. Ein Start würde scheitern oder den fremden Listener stören.",
       "launcher.refuse_if_front_unbindable", "launcher"),
    _h("MODELL-FEHLT", "Modell, Draft, Tokenizer oder vom Profil verlangter Pfad fehlt oder ist kaputt",
       "Echter Fehler, kein Wert: ohne die Datei gibt es nichts zu laden (Entrypoint MODEL, Launcher W162/W163).",
       "docker/entrypoint.sh (MODEL); launcher (Weg2GgufDepthRefused, Weg2TokenizerRefused)", "entrypoint"),
    _h("FORMAT", "Format des Checkpoints passt nicht zum Profil",
       "Echter Fehler: der Checkpoint ist in einem anderen Format als das Profil annimmt (Entrypoint detect_format, Launcher W160).",
       "docker/entrypoint.sh (MODEL); launcher (Weg2Fp8LayoutRefused)", "entrypoint"),
    _h("BAUM", "Code-Stand unsauber oder nicht das Image",
       "Herkunftsprüfung: ein Boot aus einem anderen oder geänderten Baum wäre nicht reproduzierbar und nichts darüber belegbar.",
       "launcher (tree is not clean); docker/entrypoint.sh (TREE)", "launcher"),
    _h("OPTIONEN", "Widersprüchliche oder unbekannte Optionen",
       "Der Aufruf widerspricht sich (z. B. --pp-stage-ratio ohne --pp-attn-stage-ratio, Draft-Schalter gegen Env): ein Bedienfehler, "
       "kein Wert-Urteil. Der Start hätte keine eindeutige Bedeutung.",
       "launcher (W40 Pflichtpaare, W44, W104, W151/W152, Argparse)", "launcher"),
    _h("LAUF-FEHLER", "Gruppe starb oder wurde nicht READY",
       "Laufzeitfehler nach dem Start: es gibt nichts zu übergehen.",
       "launcher.wait_ready", "launcher"),
    _h("LAUNCHER-UNKLASSIFIZIERT", "Übrige Weg2LaunchRefused-Stellen des Launchers",
       "Die Stellen sind noch nicht einzeln klassifiziert (Stand: siehe launcher_raise_sites()). Bis dahin bleiben sie hart; "
       "ehrliche Teilabdeckung, kein Pauschalfreibrief.",
       "launcher (Weg2LaunchRefused)", "launcher"),
)

_BY_CODE: Dict[str, Refusal] = {r.code: r for r in REGISTER}


def by_code(code: str) -> Optional[Refusal]:
    return _BY_CODE.get(code)


def value_codes() -> List[str]:
    return [r.code for r in REGISTER if r.forcebar]


def hard_codes() -> List[str]:
    return [r.code for r in REGISTER if not r.forcebar]


# ---------------------------------------------------------------------------
# the switch

_FORCED: List[List[object]] = []        # [code, text, logged]
_ARMED: Dict[str, object] = {"on": False, "sink": None}
_UNSET = object()


def arm(on: bool, sink: Optional[Callable[[str, str], None]] = None) -> None:
    """The launcher calls this once from ``--force``; the environment marks the boot for the processes it spawns.
    ``sink(code, text)`` (optional) is told of every refusal passed, e.g. to write it to the boot state."""
    _ARMED["on"] = bool(on)
    _ARMED["sink"] = sink if on else None
    if on:
        os.environ[ENV_FORCED_BOOT] = "1"
    _FORCED.clear()


def forced_boot() -> bool:
    """True in the launcher after ``arm(True)`` and in every process that inherited its environment."""
    return bool(_ARMED["on"]) or os.environ.get(ENV_FORCED_BOOT, "") == "1"


def forced_list() -> List[Dict[str, str]]:
    """What this boot went past, in order: ``[{code, text}]`` (for the boot state and the log)."""
    return [{"code": str(c), "text": str(t)} for c, t, _l in _FORCED]


def forced_line(code: str, text: str) -> str:
    return "%s %s %s" % (MARKER, code, " ".join(str(text).split()))


def refuse_value(code: str, text: str, make_exc: Callable[[str], BaseException],
                 log: Optional[Callable[[str], None]] = None, cause: object = _UNSET) -> None:
    """Raise ``make_exc(text)`` -- unless the boot is forced AND ``code`` is a value refusal: then remember
    ``FORCED-PAST <CODE> <reason>`` (printed through ``log`` now, or by :func:`flush`) and return.  Without
    ``--force`` this is exactly ``raise make_exc(text)`` (``from cause`` when given): the text is unchanged."""
    r = _BY_CODE.get(code)
    if r is None or not r.forcebar or not forced_boot():
        if cause is _UNSET:
            raise make_exc(text)
        raise make_exc(text) from cause         # type: ignore[misc]
    entry: List[object] = [code, str(text), False]
    _FORCED.append(entry)
    sink = _ARMED.get("sink")
    if sink is not None:
        try:
            sink(code, str(text))
        except Exception:        # noqa: BLE001 -- a lost state event must never turn a forced start into a crash
            pass
    if log is not None:
        log(forced_line(code, text))
        entry[2] = True


def flush(log: Callable[[str], None]) -> int:
    """Print the forced-past lines that no ``log`` saw yet (sites without a logger); returns how many."""
    n = 0
    for entry in _FORCED:
        if not entry[2]:
            log(forced_line(str(entry[0]), str(entry[1])))
            entry[2] = True
            n += 1
    return n


def records_allowed() -> bool:
    """False in a forced boot: it measures outside the promise, so nothing it measures becomes a record."""
    return not forced_boot()


# ---------------------------------------------------------------------------
# honesty about the wiring

_RX_CALL = re.compile(r"""refuse_value\(\s*["']([A-Z0-9-]+)["']""")


def wired_codes(launcher_source: str) -> List[str]:
    """Codes the given launcher source consults the switch for (``refuse_value("CODE", ...)``)."""
    return sorted(set(_RX_CALL.findall(launcher_source)))


def launcher_raise_sites(launcher_source: str) -> int:
    """Number of ``raise Weg2LaunchRefused`` statements in the launcher source."""
    return len(re.findall(r"raise Weg2LaunchRefused", launcher_source))


def classify(text: str) -> Optional[str]:
    """Best-effort code of a refusal message by its own prefix (``HW-COUNT: ...``); None if it carries none."""
    m = re.match(r"\s*(HW-[A-Z]+)\b", str(text))
    if m and m.group(1) in _BY_CODE:
        return m.group(1)
    return None


def public_register(wired: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """The register as data for the dashboard (``wired``: the codes the launcher of the planner tree consults)."""
    w = set(wired or ())
    out = []
    for r in REGISTER:
        out.append({"code": r.code, "klass": r.klass, "klass_label": CLASS_LABEL[r.klass], "forcebar": r.forcebar,
                    "title": r.title, "why_class": r.why_class, "source": r.source, "enforced_by": r.enforced_by,
                    "consequence": r.consequence, "wired": (r.code in w) if r.forcebar else None})
    return out
