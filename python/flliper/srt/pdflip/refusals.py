"""PROFIL-EDITOR S1: the refusal register and the ONE Force switch.

Nutzer-Entscheid 03.10. ~20:15Z (verbatim): "das dashboard soll ein profil erstellen, beim serverstart soll der
user ein profil angeben das geladen wird und es soll einen force flag beim serverstart geben dass die refusal
der values aufhebt und trotzdem startet".

* ONE switch, no list of codes: ``--force`` on the launcher (the container entrypoint maps ``FLLIPER_FORCE=1``
  / ``HTSGLANG_FORCE=1`` to it).  It lifts every VALUE refusal that is wired through :func:`refuse_value`;
  each lifted refusal is printed as ``FORCED-PAST <CODE> <reason>``; and a forced boot WRITES NO RECORDS
  (``host_ledger.append_measured_record`` returns without writing): what it measures lies outside the
  promise the planner made, and must never calibrate a later boot.
  #227 (24.10.) narrows that last rule to what it protects: a forced boot whose EVERY passed refusal is a
  PROOF GAP of its own form (:data:`RECORD_NEUTRAL`: HW-COUNT with the sole blocker METAL-UNPROVEN, the PP-CUT
  crossing-width HW-UNCALIBRATED) runs its own checkpoint, form, inventory and tree -- the non-forceable
  refusals (MODELL-FEHLT, FORMAT, TREE, HW-ARCH) and the unforced inventory check guarantee that -- and its
  measurement is exactly what closes the gap, so it writes records (stamped ``forced_past``).  Any other
  passed refusal (a foreign inventory's borrowed records, a borrowed residue/census/anchor/overshoot, a
  capacity verdict, an entrypoint threshold) keeps the boot record-free.  See :func:`records_verdict`.
* Three classes, each with its reason (see :data:`REGISTER`; ``warnt`` is HOSTWARN2, 13.10.2026: a code that used to
  refuse host RAM and now only prints ``HOST-RAM-WARN``, never consulted by :func:`refuse_value`):
    - ``value``            a VALUE refusal: the planner or the launcher judges numbers (capacity, VRAM, host
                          memory, calibration of the inventory, card count).  Force starts with these numbers.
    - ``nicht_forcebar``  NOT a value judgement: someone else's occupation of a card or of /dev/shm, a missing or
                          broken file, an impossible build.  Force does not touch these.
* ``wired`` says whether the launcher of THIS tree consults the switch for that code today
  (:func:`wired_codes` greps ``refuse_value("CODE"`` in ``launcher.py``; a test pins the register to it, so the
  register cannot claim more than the launcher does).  A ``value`` code that is not wired yet is listed as such
  and keeps refusing.
* The container ENTRYPOINT is a second place that consults the switch (``FLLIPER_FORCE=1`` -> ``refuse_value TAG CODE text``
  in ``entrypoint.sh``, before the launcher exists): PROFIL-STATUS, SHM, STORE, MEMAVAIL and the GPU gate's HW-COUNT /
  HW-UNCALIBRATED.  :data:`ENTRYPOINT_FORCE_CODES` names them (a test pins the tuple to the entrypoint source), and
  :func:`public_register` reports ``wired_at`` = launcher | entrypoint | launcher+entrypoint | none.  A code wired only in
  the entrypoint is forceable in the Docker start and NOT in a bare launcher call -- the dashboard must say exactly that,
  never "stays with Force" (wrong for Docker) and never "Force passes it" (wrong for the bare launcher).

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
ENV_FORCED_BOOT = "FLLIPER_PDFLIP_FORCED_BOOT"
#: container side: the entrypoint turns ``FLLIPER_FORCE=1`` (mapped to this) into ``--force``
ENV_FORCE_REQUEST = "HTSGLANG_FORCE"
#: both spellings the entrypoint mirrors onto each other (entrypoint.sh compat block, ``FORCE`` in its key list)
ENV_FORCE_REQUEST_NAMES: Tuple[str, ...] = (ENV_FORCE_REQUEST, "FLLIPER_FORCE")
#: #227: what the ENTRYPOINT passed before the launcher existed, one ``<CODE>\t<text>`` per line (entrypoint.sh
#: exports it, possibly empty, whenever it runs with ``HTSGLANG_FORCE=1``).  Absent under ``HTSGLANG_FORCE=1`` =
#: unknown = the boot writes no records (an entrypoint that does not list what it passed cannot be judged).
ENV_FORCED_PAST_EP = "FLLIPER_PDFLIP_FORCED_PAST_EP"
#: #227: the launcher's records verdict, handed to the FRONT's environment at its spawn (``allow;<classes>``);
#: anything else -- or nothing -- in a forced child is "no records".  Never trusted from the launcher's own
#: environment (:func:`arm` drops an inherited value).
ENV_FORCED_RECORDS = "FLLIPER_PDFLIP_FORCED_RECORDS"
RECORDS_ALLOW = "allow"
MARKER = "FORCED-PAST"

CLASS_VALUE = "value"
CLASS_HARD = "nicht_forcebar"
#: HOSTWARN2 (13.10.2026, user rule 06.10. "KEINE Verweigerung wegen Host-RAM"): the code is NOT a refusal any more --
#: the launcher prints ``HOST-RAM-WARN <code> ...`` with the full numbers and the boot goes on. Listed so a reader of
#: the register finds the old code and what became of it; never consulted by :func:`refuse_value`.
CLASS_WARN = "warnt"
CLASS_LABEL = {CLASS_VALUE: "value refusal (forceable)", CLASS_HARD: "not forceable",
               CLASS_WARN: "warns only (no refusal)"}


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


def _w(code, title, why, source, by, consequence):
    return Refusal(code, CLASS_WARN, title, why, source, by, consequence)


REGISTER: Tuple[Refusal, ...] = (
    # ------------------------------------------------------------------ value refusals
    _v("HW-COUNT", "Card count is not the proven one / not the one of the profile",
       "The profile names the card count (PROFILE_CARD_COUNT, topology P = PPn / D = TPn), or the planner knows only the proof state for N "
       "(topology.plan_topology, blockers); it compares one number with the visible number. A value judgement about numbers and proof state, "
       "not a fault of the hardware.",
       "pdflip/topology.py plan_topology (HW-COUNT with blockers); pdflip/card_identity.py order_cards; launcher.topology_check_line, launcher.order_cards", "launcher",
       "The start runs with the visible cards. The named blockers (positional vectors, BAR1 window, PP cut floor, records) remain; "
       "the next refusal comes from the first of them that the launcher checks."),
    _v("HW-UNCALIBRATED", "Card inventory is not the measured one",
       "The positional measured values of the profile (budgets, rates, rest) apply to another inventory. The planner refuses "
       "to borrow foreign measurements. This is a judgement about the quality of the numbers, not about the possibility to start.",
       "pdflip/card_identity.py uncalibrated_message; launcher.inventory_check_line; "
       "AP1 1006: launcher.dc_measured_d_mib (W19, borrows the arch twin's residue), launcher.attn_anchor_stage "
       "(borrows the reference stage), launcher._unpin_foreign_cut (drops an infeasible profile pin)", "launcher",
       "The numbers of the profile are applied to the foreign inventory as if they had been measured for it (the boot log carries FORCED-PAST). "
       "To be expected: wrong budgets, in the worst case OOM at loading or graph building."),
    _v("HW-BORROWED", "Census row of a card is borrowed from another card",
       "AP4 1526: the W71 residency census (--pdflip-xchg-census) is measured per card UUID. A card of the running inventory that is not "
       "in the census (foreign rig, other card count, other class) gets the row of the heaviest census card of the same class, or "
       "(without class) the heaviest row at all. A borrowed measurement is a judgement about the quality of the numbers, not about the "
       "possibility to start; the peak is still checked against the LIVE NVML sum of the card.",
       "pdflip/xchg_residency.py resolve_census; launcher.load_xchg_census_for_cards", "launcher",
       "The exchange peak is calculated from the borrowed row (the boot log carries FORCED-PAST HW-BORROWED). To be expected: peak "
       "over- or underestimated; a card that is too small keeps refusing through W71 (peak against the NVML sum)."),
    _v("HOST-DISK", "Free disk under the preflight floor",
       "host_ledger_preflight.sh refuses when / has under 100 GiB free (the HiCache file backend writes there): a threshold against a "
       "measured value, capacity of the DISK. HOSTWARN2: split off from HOST-MEM, which was the same exit code for RAM and disk; the "
       "user rule 06.10. frees host RAM, not disk.",
       "launcher.host_preflight (REFUSED: only N GiB free on /)", "launcher",
       "The start runs with less free disk. To be expected: a full disk under the HiCache store."),
    _v("D-BUDGET", "Group D: memory plan unsolvable (rank solver)",
       "The D rank solver has calculated and reports that its target sizes (KV, experts, rests) do not fit into the cards: "
       "capacity; the numbers stay available for the start.",
       "launcher (D_RANK_SOLVE, plan.refusal)", "launcher",
       "D starts with the calculated plan that does not fit. To be expected: OOM at pool allocation or graphs."),
    _v("WAKE-CREDIT", "Wake credit is not enough (P↔D switch)",
       "The wake credit solver gives a capacity judgement about the group switch (time/bytes per card); the numbers exist.",
       "launcher (wake_credit plan.refusal / log_wake_credit_solve_pd)", "launcher",
       "The P↔D switch runs with insufficient credit. To be expected: waiting time or OOM in the flip."),
    _v("P-CARD", "Group P: card budget is not enough (chunk/experts)",
       "The P card solver judges that the chunk with the expert fractions does not fit into the card: a capacity judgement about numbers.",
       "launcher.p_card_verdict (p_card_refusal_text)", "launcher",
       "P starts with the chunk/expert plan that does not fit. To be expected: OOM in the first prefill chunk."),
    _v("PP-CUT", "PP cut / pipeline depth not fundable",
       "W40/W42/W43: the cut solver finds no cut under the pool floors, or the depth is not carried by the KV pool: a judgement about numbers.",
       "launcher (PdFlipPPCutRefused, PdFlipDepthUnfunded, PdFlipDepthGapped)", "launcher",
       "Not wired yet: there is no plan with which the start could continue (the solver delivers no cut)."),
    _v("PROFIL-STATUS", "Profile is not accepted",
       "PROFILE_STATUS (experimentell/formnachweis/vorbereitet/geplant) says how far the profile is proven: a judgement about the proof state.",
       "docker/entrypoint.sh (PROFILE_STATUS)", "entrypoint",
       "Like HTSGLANG_ALLOW_EXPERIMENTAL=1, but also for vorbereitet/geplant. Forceable only in the Docker start (entrypoint): the launcher does not know this gate (entrypoint.sh lies outside this tree)."),
    _v("SHM", "/dev/shm too small",
       "Threshold (PROFILE_SHM_MIN_GIB) against the size of the mounted shm: capacity.",
       "docker/entrypoint.sh (SHM)", "entrypoint",
       "The start runs with a smaller shm. To be expected: bus/allocation error when the arena is created."),
    _v("STORE", "Expert store (tmpfs) too small or not tmpfs",
       "Threshold (PROFILE_TMPFS_GIB) against the size of the store mount: capacity.",
       "docker/entrypoint.sh (STORE)", "entrypoint",
       "The start runs with a store that is too small. To be expected: crash at cudaHostRegister or a full store."),
    _v("MEMAVAIL", "MemAvailable unter PROFILE_MEMAVAIL_MIN_GIB",
       "Threshold of the profile against the measured host value: capacity.",
       "docker/entrypoint.sh (MEMORY)", "entrypoint",
       "The start runs with less free host memory (see HOST-MEM)."),
    # ------------------------------------------------------------------ warns only
    _w("HOST-MEM", "Host memory below the threshold (WARNS since 13.10.2026)",
       "User rule 06.10.: no refusal on host RAM. The host_ledger preflight (script floor) and the launcher's own MemAvailable >= 40 GiB "
       "floor print HOST-RAM-WARN HOST-MEM with the numbers; the host ledger itself (W20 / W21 / W87 / W97) and the pinned-host "
       "registry warn the same way. A threshold against a measured value is capacity, and the OOM killer is the arbiter.",
       "launcher.host_preflight; host_ledger.choose(warn_only=True); mem_cache/pinned_host_budget.warn_pinned_host", "launcher",
       "The start runs with less free host memory. To be expected: OOM kill of the container when the cache grows."),
    # ------------------------------------------------------------------ not forceable
    _h("HW-TOPOLOGY", "Card count outside 2..8: there is no flip topology at all",
       "Not a value refusal: topology.plan_topology knows no topology for N outside [MIN_CARDS, MAX_CARDS_BAR1] (P/D form, BAR1 window, "
       "pipeline), there is nothing with which the start could continue. An N inside the range that is merely not proven yet is called "
       "HW-COUNT and is forceable.",
       "pdflip/topology.py plan_topology; launcher.topology_check_line", "launcher"),
    _h("HW-ARCH", "Compute capability without a kernel in the image",
       "Not a value refusal: the image carries code for sm_86 and sm_120 (wheel 86;120a) and admits sm_89 through the binary compatibility "
       "of the sm_86 cubins plus JIT (sm_89 is uncalibrated, see HW-UNCALIBRATED). Every other architecture (sm_80, sm_90, sm_100, "
       "sm_121, unreported cc) has no executable kernel, and the entrypoint expressly always refuses it already today. "
       "Force would deliver a crash in the first kernel instead of a message.",
       "pdflip/card_identity.py arch_gate; launcher.resolve_cards", "launcher"),
    _h("KARTE-BELEGT", "Foreign process or foreign window on the card",
       "Not a value refusal but protection of other users: the occupancy check (NVML foreign use, gpuq window) must never "
       "be passed, otherwise the start hits the data of someone else.",
       "launcher.cards_free_check; gpuq window", "launcher"),
    _h("SHM-BELEGT", "Live holder on own /dev/shm entries",
       "A foreign live process holds entries of the launcher: occupancy, not a value. Sweeping them away would destroy it.",
       "launcher.shm_residue_sweep (#1217/#1233)", "launcher"),
    _h("PORT-BELEGT", "Front port is already bound",
       "A real fault: another listener holds the port. A start would fail or disturb the foreign listener.",
       "launcher.refuse_if_front_unbindable", "launcher"),
    _h("MODELL-FEHLT", "Model, draft, tokenizer or a path demanded by the profile is missing or broken",
       "A real fault, not a value: without the file there is nothing to load (entrypoint MODEL, launcher W162/W163).",
       "docker/entrypoint.sh (MODEL); launcher (PdFlipGgufDepthRefused, PdFlipTokenizerRefused)", "entrypoint"),
    _h("FORMAT", "Format of the checkpoint does not match the profile",
       "A real fault: the checkpoint is in another format than the profile assumes (entrypoint detect_format, launcher W160).",
       "docker/entrypoint.sh (MODEL); launcher (PdFlipFp8LayoutRefused)", "entrypoint"),
    _h("TREE", "Code state unclean or not the image",
       "Provenance check: a boot from another or modified tree would not be reproducible and nothing about it could be verified.",
       "launcher (tree is not clean); docker/entrypoint.sh (TREE)", "launcher"),
    _h("OPTIONEN", "Contradictory or unknown options",
       "The call contradicts itself (e.g. --pp-stage-ratio without --pp-attn-stage-ratio, draft switch against env): an operating error, "
       "not a value judgement. The start would have no unambiguous meaning.",
       "launcher (W40 mandatory pairs, W44, W104, W151/W152, argparse)", "launcher"),
    _h("LAUF-FEHLER", "Group died or did not become READY",
       "Runtime error after the start: there is nothing to pass.",
       "launcher.wait_ready", "launcher"),
    _h("LAUNCHER-UNKLASSIFIZIERT", "Remaining PdFlipLaunchRefused sites of the launcher",
       "The sites are not classified individually yet (state: see launcher_raise_sites()). Until then they stay hard; "
       "honest partial coverage, no blanket pass.",
       "launcher (PdFlipLaunchRefused)", "launcher"),
)

_BY_CODE: Dict[str, Refusal] = {r.code: r for r in REGISTER}

#: value refusals the container entrypoint passes with ``FLLIPER_FORCE=1`` (entrypoint.sh: ``refuse_value <TAG> <CODE>`` and the
#: GPU gate's ``FORCED-PAST HW-COUNT|HW-UNCALIBRATED``); pinned to the entrypoint source by test_refusals_force_1003
ENTRYPOINT_FORCE_CODES: Tuple[str, ...] = ("HW-COUNT", "HW-UNCALIBRATED", "MEMAVAIL", "PROFIL-STATUS", "SHM", "STORE")
#: the exact scope sentence for a code that only the entrypoint lifts (shown by the dashboard verbatim)
SCOPE_ENTRYPOINT_ONLY = "forceable in the Docker start (entrypoint), not in a plain launcher call"


def by_code(code: str) -> Optional[Refusal]:
    return _BY_CODE.get(code)


def value_codes() -> List[str]:
    return [r.code for r in REGISTER if r.forcebar]


def hard_codes() -> List[str]:
    return [r.code for r in REGISTER if r.klass == CLASS_HARD]


def warn_codes() -> List[str]:
    """Codes that used to refuse and now only warn (HOSTWARN2)."""
    return [r.code for r in REGISTER if r.klass == CLASS_WARN]


# ---------------------------------------------------------------------------
# the switch

_FORCED: List[List[object]] = []        # [code, text, logged]
_ARMED: Dict[str, object] = {"on": False, "sink": None}
#: #227: records a forced launcher wanted to write before its verdict was final (``(path, record)``)
_HELD: List[Tuple[str, Dict[str, object]]] = []
#: #227: the launcher's final records verdict (:func:`seal_records`)
_SEAL: Dict[str, object] = {"sealed": False, "allow": False, "why": "", "classes": ()}
_UNSET = object()


def arm(on: bool, sink: Optional[Callable[[str, str], None]] = None) -> None:
    """The launcher calls this once from ``--force``; the environment marks the boot for the processes it spawns.
    ``sink(code, text)`` (optional) is told of every refusal passed, e.g. to write it to the boot state."""
    _ARMED["on"] = bool(on)
    _ARMED["sink"] = sink if on else None
    if on:
        os.environ[ENV_FORCED_BOOT] = "1"
    os.environ.pop(ENV_FORCED_RECORDS, None)    # #227: the launcher seals; an inherited verdict is never trusted
    _FORCED.clear()
    _HELD.clear()
    _SEAL.update(sealed=False, allow=False, why="", classes=())


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
    if _SEAL["sealed"] and _SEAL["allow"] and records_class(code, str(text)) is None:
        # #227: main() spawns the front last, after every refusal site; a pass after the seal is a wiring change
        # that must not ride on a verdict already handed out -- this process stops writing and says why
        _SEAL.update(allow=False, why="%s passed AFTER the records seal (%s) -- no further records from this "
                     "process; a front spawned before it holds the earlier verdict" % (code, _one_line(text, 160)))
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
    """May this process write a measured record NOW?  True in an unforced boot (unchanged).  In a forced boot
    (#227): the launcher only after :func:`seal_records` judged every passed refusal record-neutral; a child
    (front, groups) only when the launcher handed it ``ENV_FORCED_RECORDS=allow;...``.  Everything else -- a
    forced boot that passed a non-neutral refusal, an unsealed launcher, a child without the seal -- writes none:
    what it measures lies outside the promise."""
    if not forced_boot():
        return True
    if _ARMED["on"]:
        return bool(_SEAL["sealed"]) and bool(_SEAL["allow"])
    return _env_seal() is not None


def records_pending() -> bool:
    """#227: a forced LAUNCHER before its seal -- a record it writes now is HELD (:func:`hold`) and written or
    dropped by the verdict, never written on a verdict that is not final yet."""
    return forced_boot() and bool(_ARMED["on"]) and not bool(_SEAL["sealed"])


def hold(path: str, rec: Dict[str, object]) -> None:
    _HELD.append((str(path), dict(rec)))


def take_held() -> List[Tuple[str, Dict[str, object]]]:
    out = list(_HELD)
    _HELD.clear()
    return out


# ---------------------------------------------------------------------------
# #227 FORCED-RECORDS: which forced passes leave the form identity intact
#
# Measured 24.10.: every rotation boot since 23.10. 05:18Z (45 of 45, int8/dual/int4ar/duo; nf likewise) runs with
# HTSGLANG_FORCE=1 and passed exactly two refusals -- HW-COUNT [METAL-UNPROVEN] (2-card forms) and the PP-CUT
# crossing-width HW-UNCALIBRATED (the PCIe switch gives x16, the crossing table holds x4/x8).  Both say "this has
# not been measured on this rig yet"; the second literally says "a crossing measurement of this rig writes the
# table".  With "forced = no records" no boot could ever write that measurement: the 27B sidecar has no sample
# since 22.10. 02:04Z and int4ar's D residue stayed the 2652 MiB constant against 1258 MiB measured at its sleep.

_RX_BLOCKERS = re.compile(r"not yet runnable, (\d+) blocker\(s\): (.*?)(?: \|\| visible: .*)?$", re.S)
_RX_BLOCKER_TAG = re.compile(r"(?:^|; )\[([A-Z][A-Z0-9-]*)\] ")
_RX_CROSSING = re.compile(
    r"HW-UNCALIBRATED: PP-CUT crossing prices are measured at PCIe widths x\d+(?:, x\d+)* only; "
    r"ordinal \d+ \([^()]*\) x(?:\d+|\?) -> x\d+(?:; ordinal \d+ \([^()]*\) x(?:\d+|\?) -> x\d+)*"
    r" -- --force prices these at the nearest measured width not faster than their own: UNMEASURED "
    r"on this board \(a crossing measurement of this rig writes the table\)")


def _hw_count_metal_only(text: str) -> bool:
    """HW-COUNT whose blocker list is exactly N x ``[METAL-UNPROVEN]`` (topology.plan_topology's text): the card
    count is runnable, only no release boot on N cards exists yet.  A profile/visible count mismatch
    (card_identity ``--expect-count``, order_cards) or any further blocker is a foreign configuration."""
    m = _RX_BLOCKERS.search(str(text))
    if m is None:
        return False
    tags = _RX_BLOCKER_TAG.findall(m.group(2))
    return bool(tags) and len(tags) == int(m.group(1)) and all(t == "METAL-UNPROVEN" for t in tags)


def _hw_uncal_crossing_only(text: str) -> bool:
    """HW-UNCALIBRATED of ``launcher.forced_lane_borrow`` alone (full match, no ``||`` join with an inventory
    message): only the TIME price of a PP crossing is borrowed, from a width not faster than the card's own; no
    VRAM budget, residue, record or vector of another inventory enters the boot."""
    return _RX_CROSSING.fullmatch(" ".join(str(text).split())) is not None


#: ``(code, class name, predicate on the refusal text)``: a forced pass that is a PROOF GAP of this very form.
#: Deliberately two entries; extend only with a reason why the pass changes nothing the boot then measures.
RECORD_NEUTRAL: Tuple[Tuple[str, str, Callable[[str], bool]], ...] = (
    ("HW-COUNT", "METAL-UNPROVEN", _hw_count_metal_only),
    ("HW-UNCALIBRATED", "pp-crossing-width", _hw_uncal_crossing_only),
)


def records_class(code: str, text: str) -> Optional[str]:
    """``<CODE>/<class>`` when this forced pass leaves the boot's form identity intact, None otherwise."""
    for c, name, pred in RECORD_NEUTRAL:
        if str(code) == c and pred(str(text)):
            return f"{c}/{name}"
    return None


def _one_line(text: object, n: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3] + "..."


def entrypoint_forced(environ: Optional[Dict[str, str]] = None) -> Optional[List[Tuple[str, str]]]:
    """What the entrypoint passed (``[(code, text)]``); None = unknown (``HTSGLANG_FORCE=1`` without the list);
    ``[]`` = nothing, or no entrypoint (no ``HTSGLANG_FORCE=1``: a bare launcher call)."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_FORCED_PAST_EP)
    if raw is None:
        asked = any(str(env.get(k, "")).strip() == "1" for k in ENV_FORCE_REQUEST_NAMES)
        return None if asked else []
    out: List[Tuple[str, str]] = []
    for line in str(raw).splitlines():
        if not line.strip():
            continue
        code, _, text = line.partition("\t")
        out.append((code.strip(), text))
    return out


def records_verdict(environ: Optional[Dict[str, str]] = None) -> Tuple[bool, str, Tuple[str, ...]]:
    """``(allow, why, classes)`` of the forced passes so far (launcher) -- every one record-neutral, the
    entrypoint's included; ``classes`` the sorted record-neutral classes passed.  Unforced: ``(True, ..., ())``."""
    if not forced_boot():
        return True, "not forced", ()
    if not _ARMED["on"]:
        v = _env_seal(environ)
        if v is None:
            return False, "forced child without the launcher's records seal", ()
        return True, "sealed by the launcher", v
    passed = [(str(c), str(t), "launcher") for c, t, _l in _FORCED]
    ep = entrypoint_forced(environ)
    if ep is None:
        return False, ("the entrypoint ran with %s=1 but did not list what it passed (%s absent): cannot be judged"
                       % (ENV_FORCE_REQUEST, ENV_FORCED_PAST_EP)), ()
    passed += [(c, t, "entrypoint") for c, t in ep]
    bad = [(c, t, w) for c, t, w in passed if records_class(c, t) is None]
    classes = tuple(sorted({records_class(c, t) for c, t, _w in passed} - {None}))
    if bad:
        return False, "; ".join("%s (%s): %s" % (c, w, _one_line(t, 140)) for c, t, w in bad), classes
    return True, ("every passed refusal is a proof gap of this form: " + (", ".join(classes) or "none passed")), classes


def seal_records(environ: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """#227: the launcher's FINAL verdict, taken once right before the front (the record writer) spawns, after
    every refusal site of ``main``.  Returns ``(allow, why)``; then :func:`records_allowed` answers by it and
    ``host_ledger.flush_held_records`` writes or drops what was held."""
    allow, why, classes = records_verdict(environ)
    if not forced_boot():
        _SEAL.update(sealed=True, allow=True, why=why, classes=())
        return True, why
    _SEAL.update(sealed=True, allow=bool(allow), why=why, classes=tuple(classes))
    return bool(allow), why


def records_verdict_line() -> str:
    """The launcher's one line about it (``FORCED-RECORDS WRITES|NONE ...``)."""
    if not forced_boot():
        return "FORCED-RECORDS n/a: not a forced boot -- records as always"
    if not _SEAL["sealed"]:
        return "FORCED-RECORDS PENDING: verdict not sealed yet -- records held"
    if _SEAL["allow"]:
        return ("FORCED-RECORDS WRITES: " + str(_SEAL["why"]) + " -- identity: checkpoint, form, card inventory and "
                "tree are this boot's own (non-forceable MODELL-FEHLT/FORMAT/TREE/HW-ARCH, inventory check not "
                "forced); records carry forced_past=[" + ", ".join(_SEAL["classes"]) + "]")
    return ("FORCED-RECORDS NONE: this boot writes NO records -- " + str(_SEAL["why"])
            + " (a measurement outside the promise must not calibrate a later boot)")


def export_records_verdict(env: Dict[str, str]) -> None:
    """Hand the sealed verdict to a child's environment (the front): ``allow;<classes>`` or nothing."""
    if forced_boot() and _SEAL["sealed"] and _SEAL["allow"]:
        env[ENV_FORCED_RECORDS] = RECORDS_ALLOW + ";" + ",".join(_SEAL["classes"])
    else:
        env.pop(ENV_FORCED_RECORDS, None)


def forced_record_stamp() -> Optional[List[str]]:
    """The ``forced_past`` stamp of a record written by a forced boot (its record-neutral classes); None unforced."""
    if not forced_boot():
        return None
    if _ARMED["on"]:
        return list(_SEAL["classes"])
    v = _env_seal()
    return list(v) if v is not None else []


def _env_seal(environ: Optional[Dict[str, str]] = None) -> Optional[Tuple[str, ...]]:
    env = os.environ if environ is None else environ
    v = str(env.get(ENV_FORCED_RECORDS, "") or "")
    if v != RECORDS_ALLOW and not v.startswith(RECORDS_ALLOW + ";"):
        return None
    rest = v[len(RECORDS_ALLOW) + 1:] if v.startswith(RECORDS_ALLOW + ";") else ""
    return tuple(x for x in rest.split(",") if x)


# ---------------------------------------------------------------------------
# honesty about the wiring

_RX_CALL = re.compile(r"""refuse_value\(\s*["']([A-Z0-9-]+)["']""")


def wired_codes(launcher_source: str) -> List[str]:
    """Codes the given launcher source consults the switch for (``refuse_value("CODE", ...)``)."""
    return sorted(set(_RX_CALL.findall(launcher_source)))


_RX_EP_CALL = re.compile(r"""refuse_value\s+[A-Z_]+\s+([A-Z][A-Z0-9-]+)""")
_RX_EP_PAST = re.compile(r"""FORCED-PAST\s+([A-Z][A-Z0-9-]+)""")


def entrypoint_wired_codes(entrypoint_source: str) -> List[str]:
    """Codes the given ``entrypoint.sh`` passes under ``HTSGLANG_FORCE=1``: ``refuse_value <TAG> <CODE> ...`` calls and literal
    ``FORCED-PAST <CODE>`` lines (the shell function's own ``$2`` is not a code and is not matched)."""
    return sorted(set(_RX_EP_CALL.findall(entrypoint_source)) | set(_RX_EP_PAST.findall(entrypoint_source)))


def launcher_raise_sites(launcher_source: str) -> int:
    """Number of ``raise PdFlipLaunchRefused`` statements in the launcher source."""
    return len(re.findall(r"raise PdFlipLaunchRefused", launcher_source))


def classify(text: str) -> Optional[str]:
    """Best-effort code of a refusal message by its own prefix (``HW-COUNT: ...``); None if it carries none."""
    m = re.match(r"\s*(HW-[A-Z]+)\b", str(text))
    if m and m.group(1) in _BY_CODE:
        return m.group(1)
    return None


def public_register(wired: Optional[Sequence[str]] = None,
                    entrypoint_wired: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """The register as data for the dashboard.

    ``wired``: the codes the launcher of the planner tree consults.  ``entrypoint_wired``: the codes the container entrypoint
    consults (default :data:`ENTRYPOINT_FORCE_CODES`).  Per forceable code: ``wired`` (launcher, unchanged meaning),
    ``wired_entrypoint`` and ``wired_at`` (``launcher`` | ``entrypoint`` | ``launcher+entrypoint`` | ``none``); for a hard code
    all three are ``None``.  ``force_scope`` is the one-line truth about WHERE Force lifts it."""
    w = set(wired or ())
    e = set(ENTRYPOINT_FORCE_CODES if entrypoint_wired is None else entrypoint_wired)
    out = []
    for r in REGISTER:
        in_l, in_e = r.code in w, r.code in e
        if r.forcebar:
            at = "launcher+entrypoint" if (in_l and in_e) else "launcher" if in_l else "entrypoint" if in_e else "none"
            scope = {"launcher+entrypoint": "forceable in the Docker start (entrypoint) and in the launcher",
                     "launcher": "forceable in the launcher (also in the Docker start, which calls the launcher)",
                     "entrypoint": SCOPE_ENTRYPOINT_ONLY,
                     "none": "forceable by class, but wired nowhere: the start keeps refusing"}[at]
        else:
            at, scope = None, "not forceable"
        out.append({"code": r.code, "klass": r.klass, "klass_label": CLASS_LABEL[r.klass], "forcebar": r.forcebar,
                    "title": r.title, "why_class": r.why_class, "source": r.source, "enforced_by": r.enforced_by,
                    "consequence": r.consequence, "wired": in_l if r.forcebar else None,
                    "wired_entrypoint": in_e if r.forcebar else None, "wired_at": at, "force_scope": scope})
    return out
