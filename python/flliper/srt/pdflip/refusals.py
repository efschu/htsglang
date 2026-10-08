"""PROFIL-EDITOR S1: the refusal register and the ONE Force switch.

Nutzer-Entscheid 03.10. ~20:15Z (verbatim): "das dashboard soll ein profil erstellen, beim serverstart soll der
user ein profil angeben das geladen wird und es soll einen force flag beim serverstart geben dass die refusal
der values aufhebt und trotzdem startet".

* ONE switch, no list of codes: ``--force`` on the launcher (the container entrypoint maps ``FLLIPER_FORCE=1``
  / ``HTSGLANG_FORCE=1`` to it).  It lifts every VALUE refusal that is wired through :func:`refuse_value`;
  each lifted refusal is printed as ``FORCED-PAST <CODE> <reason>``; and a forced boot WRITES NO RECORDS
  (``host_ledger.append_measured_record`` returns without writing): what it measures lies outside the
  promise the planner made, and must never calibrate a later boot.
* Two classes, each with its reason (see :data:`REGISTER`):
    - ``value``            a VALUE refusal: the planner or the launcher judges numbers (capacity, VRAM, host
                          memory, calibration of the inventory, card count).  Force starts with these numbers.
    - ``nicht_forcebar``  NOT a value judgement: someone else's occupation of a card or of /dev/shm, a missing or
                          broken file, an impossible build.  Force does not touch these.
* ``wired`` says whether the launcher of THIS tree consults the switch for that code today
  (:func:`wired_codes` greps ``refuse_value("CODE"`` in ``launcher.py``; a test pins the register to it, so the
  register cannot claim more than the launcher does).  A ``value`` code that is not wired yet is listed as such
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
ENV_FORCED_BOOT = "FLLIPER_PDFLIP_FORCED_BOOT"
#: container side: the entrypoint turns ``FLLIPER_FORCE=1`` (mapped to this) into ``--force``
ENV_FORCE_REQUEST = "HTSGLANG_FORCE"
MARKER = "FORCED-PAST"

CLASS_VALUE = "value"
CLASS_HARD = "nicht_forcebar"
CLASS_LABEL = {CLASS_VALUE: "value refusal (forceable)", CLASS_HARD: "not forceable"}


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
    _v("HOST-MEM", "Host memory below the threshold",
       "A threshold (host_ledger preflight, MemAvailable >= 40 GiB) against a measured value: capacity, not a fault.",
       "launcher.host_preflight", "launcher",
       "The start runs with less free host memory. To be expected: OOM kill of the container when the cache grows."),
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
       "Like HTSGLANG_ALLOW_EXPERIMENTAL=1, but also for vorbereitet/geplant. The entrypoint lies outside the repo (staged patch)."),
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
    # ------------------------------------------------------------------ not forceable
    _h("HW-TOPOLOGY", "Card count outside 2..8: there is no flip topology at all",
       "Not a value refusal: topology.plan_topology knows no topology for N outside [MIN_CARDS, MAX_CARDS_BAR1] (P/D form, BAR1 window, "
       "pipeline), there is nothing with which the start could continue. An N inside the range that is merely not proven yet is called "
       "HW-COUNT and is forceable.",
       "pdflip/topology.py plan_topology; launcher.topology_check_line", "launcher"),
    _h("HW-ARCH", "Compute capability without a kernel in the image",
       "Not a value refusal: the image carries only code for sm_86 and sm_120 (wheel 86;120a). Another architecture has no "
       "executable kernel, and the entrypoint expressly always refuses it already today. Force would deliver a crash in the first "
       "kernel instead of a message.",
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
    """Number of ``raise PdFlipLaunchRefused`` statements in the launcher source."""
    return len(re.findall(r"raise PdFlipLaunchRefused", launcher_source))


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
