#!/usr/bin/env python3
"""The step table -- the single authoritative description of the battery.

BATTERY.md, run_step.sh, the plan/resume machinery and the dry tests all read
THIS table. A step that exists here and nowhere else still runs correctly; a
step described only in prose does not exist. Keeping one table is what makes
"never deviate from the step list" enforceable rather than a request.

Fields per step:

  step_id          directory name under the run dir and the verdict key
  title            one line, human
  model            which executor model is meant to drive it: haiku for pure
                   script steps, sonnet for steps that boot a server and have
                   to read a log when something goes sideways
  script           the step script, relative to scripts/gpu_battery/
  check            the check script, relative to scripts/gpu_battery/checks/
  timeout_s        HARD wall for the step script. Exceeding it is a STOP, not
                   a reason to wait longer. Always well above expected_min so
                   that a slow load is not mistaken for a hang.
  expected_min     what it should take when nothing is wrong
  retryable        may the executor re-run it once, unattended? True only
                   where a retry costs minutes and no boot budget.
  deps             steps whose ARTIFACTS this step consumes. Not the running
                   order -- that is the table order. Kept minimal on purpose
                   so that a resume can run a late step without re-running
                   boots that have nothing to do with it.
  needs_cards      does it touch a GPU at all
  locks            who takes /tmp/gpu-card-N.lock:
                     "battery"  run_step.sh takes them for the step
                     "tool"     the step's own tool takes them (p2p run_all.sh)
                                -- run_step.sh must NOT hold them, or the tool
                                aborts on its own lock acquisition
                     "none"     CPU-only or query-only, nothing taken
  report_gate      after this step the executor stops and reports even on PASS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Step:
    step_id: str
    title: str
    model: str
    script: str
    check: str
    timeout_s: int
    expected_min: int
    retryable: bool = False
    deps: Tuple[str, ...] = ()
    needs_cards: bool = True
    locks: str = "battery"
    report_gate: bool = False


STEPS: Tuple[Step, ...] = (
    Step(
        step_id="s00_preflight",
        title="Preflight: Karten-Identitaet (PCI/UUID), Korridor, Locks, Pflichtdateien",
        model="haiku",
        script="s00_preflight.sh",
        check="check_s00_preflight.py",
        timeout_s=300,
        expected_min=3,
        retryable=True,
        deps=(),
        needs_cards=True,
        locks="none",
    ),
    Step(
        step_id="s01_p2p_reprobe",
        title="P2P-Re-Probe nach Treiber-Update (capability matrix, d2d, NCCL-Transport)",
        model="haiku",
        script="s01_p2p_reprobe.sh",
        check="check_s01_p2p_reprobe.py",
        timeout_s=1800,
        expected_min=10,
        retryable=True,
        deps=("s00_preflight",),
        needs_cards=True,
        locks="tool",
    ),
    Step(
        step_id="s02_boot_a",
        title="R7c Boot A -- FP8-Referenz, der Ein-Achsen-Falsifikator",
        model="sonnet",
        script="s02_boot_a.sh",
        check="check_s02_boot_a.py",
        timeout_s=3600,
        expected_min=35,
        retryable=False,
        deps=("s00_preflight",),
        needs_cards=True,
        locks="battery",
        report_gate=True,
    ),
    Step(
        step_id="s03_boot_b",
        title="R7c Boot B -- AWQ mit BF16-Kopf, die Kopf-vs-Ziel-Achse",
        model="sonnet",
        script="s03_boot_b.sh",
        check="check_s03_boot_b.py",
        timeout_s=3900,
        expected_min=40,
        retryable=False,
        deps=("s00_preflight",),
        needs_cards=True,
        locks="battery",
    ),
    Step(
        step_id="s04_boot_c",
        title="R7c Boot C -- DFLASH-Q8_0-Drafter solo auf einer 3080",
        model="sonnet",
        script="s04_boot_c.sh",
        check="check_s04_boot_c.py",
        timeout_s=4200,
        expected_min=45,
        retryable=False,
        deps=("s00_preflight",),
        needs_cards=True,
        locks="battery",
    ),
    Step(
        step_id="s05_boot_d",
        title="R7c Boot D -- Lane-Re-Seed A/B, die bekannte Konfiguration",
        model="sonnet",
        script="s05_boot_d.sh",
        check="check_s05_boot_d.py",
        timeout_s=3000,
        expected_min=30,
        retryable=False,
        deps=("s00_preflight",),
        needs_cards=True,
        locks="battery",
    ),
    Step(
        step_id="s06_nccl_reference",
        title="NCCL/System-RAM-Referenzmessung im #279-Format (p50+p99, idle+Last)",
        model="haiku",
        script="s06_nccl_reference.sh",
        check="check_s06_nccl_reference.py",
        timeout_s=1800,
        expected_min=15,
        retryable=True,
        deps=("s00_preflight",),
        needs_cards=True,
        locks="battery",
    ),
    Step(
        step_id="s07_offload_register_gpu",
        title="Offload-Register auf GPU: CudaDeviceOps, echte Posten-Groessen, Rueckhol-Latenzen",
        model="haiku",
        script="s07_offload_register_gpu.sh",
        check="check_s07_offload_register_gpu.py",
        timeout_s=1200,
        expected_min=12,
        retryable=True,
        deps=("s00_preflight",),
        needs_cards=True,
        locks="battery",
    ),
    Step(
        step_id="s08_dispatcher_tables",
        title="Dispatcher-Ratentabellen laden + Placeholder-Neutralitaet nachpruefen (CPU)",
        model="haiku",
        script="s08_dispatcher_tables.sh",
        check="check_s08_dispatcher_tables.py",
        timeout_s=600,
        expected_min=3,
        retryable=True,
        deps=("s01_p2p_reprobe", "s06_nccl_reference"),
        needs_cards=False,
        locks="none",
    ),
    Step(
        step_id="s09_sensor_smoke",
        title="gdn-/KV-Druck-Leiter-Smoke: Flags booten, Sensor frisst echte Belegung",
        model="sonnet",
        script="s09_sensor_smoke.sh",
        check="check_s09_sensor_smoke.py",
        timeout_s=1800,
        expected_min=15,
        retryable=True,
        deps=("s00_preflight",),
        needs_cards=True,
        locks="battery",
    ),
)

STEPS_BY_ID: Dict[str, Step] = {s.step_id: s for s in STEPS}
STEP_ORDER: List[str] = [s.step_id for s in STEPS]

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_STOP = "STOP"
VERDICT_SKIP = "SKIP"
VERDICTS = (VERDICT_PASS, VERDICT_FAIL, VERDICT_STOP, VERDICT_SKIP)


def total_expected_min(step_ids: Optional[List[str]] = None) -> int:
    ids = step_ids if step_ids is not None else STEP_ORDER
    return sum(STEPS_BY_ID[i].expected_min for i in ids)


def resolve_ids(spec: str) -> List[str]:
    """Turn a comma list of step ids or unambiguous prefixes into step ids.

    Prefixes exist so the operator can type s02 instead of s02_boot_a; an
    ambiguous or unknown token is an error, never a guess.
    """
    out: List[str] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token in STEPS_BY_ID:
            out.append(token)
            continue
        matches = [i for i in STEP_ORDER if i.startswith(token)]
        if len(matches) == 1:
            out.append(matches[0])
        elif not matches:
            raise KeyError(f"unbekannter Schritt: {token!r}")
        else:
            raise KeyError(f"mehrdeutiger Schritt {token!r}: {matches}")
    return out
