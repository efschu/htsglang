#!/usr/bin/env python3
"""weg1_host_sizing.py -- #1068 WEG 1 (slice 5): size both phase host pools
from the DEMAND formula, cap them by the host ledger BY NAME, refuse when the
formula is not solvable. Never a ratio, never a fallback nobody chose.

WHY THIS LIVES IN THE TREE (slice 5 fix, operator decision 6 of 2026-09-02).
The first cut of this module sat in /spinning/gpu-arb/devtools (unversioned),
so the in-tree test pinned only files outside the tree and was green on every
tree state; nothing in a commit could make it red. The consumer of S and M is
the tree (server_args.py _refuse_incomplete_phase_flip_hicache_sizing_1068,
slice 1, refuses a ratio under --phase-flip-rebind-hicache and demands the
three-flag contract), so the producer of S and M is versioned beside it. The
launcher (boot_855_train0901.sh, operator dir) runs it as
`python -m sglang.srt.planner.weg1_host_sizing` under PYTHONPATH=$TREE/python:
a tree WITHOUT this module refuses by construction, and the launcher probes
for the module and for the L13 refusal symbol on $TREE before it composes
the command line, naming whatever is missing.

WHAT THIS REPLACES. boot_855_train0901.sh:405-416 (pre-slice-5) emitted
`--hicache-ratio ${HICACHE_RATIO:-1.5}` and printed a provenance line whose
last branch read "HARDCODED FALLBACK -- nobody chose this number".

THE ARITHMETIC (WEG1_BUILD_SPEC_0901.md section 5 "Launcher-Formel" and
section 11 A11.1/A11.2, both binding for slice 5):

  demand_rows = (n_resident + n_queue + chain_lag) x prompt_max
      n_resident = max_running_requests   (config: the TP batch cap)
      n_queue    = WEG1_QUEUE_DEPTH       (knob, default = max_running_requests)
      chain_lag  = 1                      (section 5: +1 occupant on PP1/PP2)
      prompt_max = WEG1_PROMPT_MAX_TOKENS (knob, default = the load's prompt
                                           class, 39365 on the Boot-2 load)
  S_demand  = ceil(demand_rows x cell_pp0 / 1e9)   [GB, absolute]
      cell_pp0 = the PP0 host cell in bytes per token: the LARGEST cell over
                 the ranks, because pool_host/base.py:140-147 sizes each rank
                 as int(S*1e9 // cell_r) and sync_fixed_hicache_size takes the
                 group MIN, i.e. the rank with the largest cell binds. The
                 spec's literal "cell_pp0 from the LAST HOST-LEDGER line"
                 would read PP2's 8192 on this cut and size PP0 short; the
                 max over the lines is the binding term. Read from the
                 previous boot's HOST-LEDGER POST lines (field `cell_pp=%d B`,
                 L8 as built), the knob WEG1_CELL_PP0_BYTES takes precedence
                 over the log, else 16384 (section 5, log 1329/1350).
  rows(S)   = S*1e9 // cell_pp0 + 1                  (base.py:140-147)

  Ledger cap: 8 x S + anchors <= MemAvailable - 16 GiB - 10 GiB - 27 GiB,
  where 8 x S is the KV of both phases (PP: S + S/2 + S/2 = 2S, TP pin
  rows-coupled to the PP pool: 3 x 2S = 6S), the anchors are 2 phases x ranks
  x M_MIB, 16 GiB is the #721 floor AS THE GATE APPLIES IT
  (host_ledger_preflight.sh FLOOR_G=16, in GiB; the spec's section 5 writes
  "16 GB" and that is a unit slip, operator decision 1 of 2026-09-02 -- the
  launcher formula and the gate now use the same 16 GiB and both the GiB
  floor and the resulting cap are printed), 10 GiB is
  PINNED_HOST_RESERVE_BYTES (pinned_host_budget.py:59) and 27 GiB is the
  weights-load transient (#721), subtracted as a safety term although the
  pools are allocated after the load.
      S_ledger  = floor((cap - anchors) / 8e9)
      S         = min(S_demand, S_ledger)
  A11.2: demand_rows > rows(S_ledger) is a NAMED degradation (one line,
  verbatim below), never a silent size. Refusal (exit 2) only when
  rows(S_ledger) < max_running_requests x chunked_prefill_size (the in-tree
  floor G10 would refuse anyway) or when S_ledger < 1.

  Boot-2 configuration (MemAvailable 119 GB, 8+8+1 x 39365): demand 669205
  rows (S_demand 11) against cap 62.09 GB - anchors 15.10 GB = 46.99 GB over
  8 x S -> S_ledger 5 (305176 rows, spans_at_prompt_max 7.75): the A11.2
  degradation line prints and S = 5. With the spec's 16e9 floor the same
  configuration gave S_ledger 6 (kv budget 48.17 GB); the 1.18 GB the GiB
  floor costs sits exactly on that boundary. Spec section 11 R1 names S=5
  as the fallback value, so the result stays inside the spec's own space.

  M_MIB (section 5): formula = ceil((device_slots + max_running_requests + 1
  + 2 x device_slots) x per_slot_rank0_MiB) = 69 x 37.41 = 2582 -> 2600;
  the acceptance value is 2400 (64 slots = 3.2 x device_slots 20; the risk
  is named in section 11 R4). Knob WEG1_MAMBA_HOST_MIB, default 2400. The
  local slot count on rank 0, floor(M_MIB / per_slot_rank0), must be >=
  device_slots + max_running_requests + 1 (the in-tree anchor floor of
  phase_flip_boot.py); below it this refuses (exit 2).

UNITS, SAID ONCE. GB means 1e9 bytes (the unit of --hicache-size and of
base.py:140), GiB means 2**30 (the unit of the floor, the reserve and the
transient), MiB means 2**20 (the unit of --hicache-mamba-host-mib).
MemAvailable is read from /proc/meminfo in kB and converted with x1024.

INTERFACE. `size_host_pools(...)` is the importable form (the slice-5 test
pins it); `main()` is the launcher's form: config through flags, rig terms
through WEG1_* environment knobs, human lines on stdout, machine lines
`WEG1_<KEY>=<value>` on stdout for the launcher to read, exit 2 on refusal
with '#1068 WEG1 SIZING REFUSED: ...' naming the terms. The provenance line
of spec section 4.6 ("requested/ledger/EFFECTIVE") is built as
"demand=/ledger= -> EFFECTIVE=" for S and "formula=/knob= -> EFFECTIVE=" for
M: the spec's "requested" is the demand (S) or the knob (M); the rename is
deliberate because "requested" would not say WHO requested.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

GB = 1e9
GIB = 2**30
MIB = 2**20
# #721 floor as host_ledger_preflight.sh FLOOR_G applies it: 16 GiB, not the
# spec's "16 GB" literal (unit slip; operator decision 2026-09-02).
FLOOR_BYTES = 16 * GIB
RESERVE_BYTES = 10 * GIB  # PINNED_HOST_RESERVE_BYTES, pinned_host_budget.py:59
LOAD_TRANSIENT_BYTES = 27 * GIB  # weights-load transient, #721
CHAIN_LAG = 1  # spec section 5: +1 queue occupant on PP1/PP2 at every cutover
KV_MULTIPLE_OF_S = 8  # PP 2S + TP pin 6S (rows-coupled), spec section 5
DEFAULT_PROMPT_MAX = 39365  # Boot-2 load prompt class, spec section 5
DEFAULT_CELL_PP0 = 16384  # PP0 host cell B/token, spec section 5 (log 1329/1350)
DEFAULT_MAMBA_MIB = 2400  # spec section 5 acceptance value (64 slots)
DEFAULT_PER_SLOT_RANK0_MIB = 37.41  # log 1351 (PP0 mamba host per_slot)
DEFAULT_DEVICE_SLOTS = 20  # log 1351 device_slots=20
HOST_LEDGER_TOKEN = "HOST-LEDGER POST #847/#871 phase-flip staging pin"
_CELL_PP_RE = re.compile(r"\bcell_pp=(\d+) B\b")

# A11.2, verbatim (the launcher and the acceptance grep this prefix).
DEMAND_EXCEEDS_LEDGER = (
    "#1068 HOST POOL DEMAND EXCEEDS LEDGER demand_rows=%d (n_resident=%d "
    "n_queue=%d chain_lag=1 prompt_max=%d) ledger_rows=%d "
    "spans_at_prompt_max=%.2f -> pool sized to the ledger; requests beyond "
    "the pool land sequentially via evict_host or are truncated-named (L2)"
)
# A12.3 correction to the line above: the RATE gate defers (A12.2), it never
# declines; only the POOL gate lands-after-evict_host or truncates.
DEMAND_EXCEEDS_LEDGER_A123 = (
    "#1068 HOST POOL DEMAND EXCEEDS LEDGER (A12.3): the sentence above names "
    "the host_pool_exhausted gate only; a rate_limited verdict is DEFERRED "
    "(A12.2, '#1068 PREFETCH DEFERRED' -> 'LANDED'), never a decline into "
    "recompute"
)
REFUSED_PREFIX = "#1068 WEG1 SIZING REFUSED"


class SizingRefused(RuntimeError):
    """The formula is not solvable on this box / config. Exit 2, no fallback."""


@dataclass
class Sizing:
    n_resident: int
    n_queue: int
    prompt_max: int
    cell_pp0: int
    demand_rows: int
    s_demand_gb: int
    memavail_bytes: int
    cap_bytes: int
    anchors_bytes: int
    s_ledger_gb: int
    ledger_rows: int
    s_gb: int
    pool_rows: int
    spans_at_prompt_max: float
    m_mib: int
    m_formula_mib: int
    m_formula_rounded_mib: int
    anchor_slots_rank0: int
    anchor_floor_slots: int
    degraded: bool
    lines: list = field(default_factory=list)
    refusal: Optional[str] = None


def rows_of(s_gb: int, cell: int) -> int:
    """pool_host/base.py:140-147: int(S*1e9 // cell) rows, +1 page (page_size 1)."""
    return int(s_gb * GB // cell) + 1


def read_memavail_bytes() -> int:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise SizingRefused(f"{REFUSED_PREFIX}: /proc/meminfo carries no MemAvailable line")


def cell_pp0_from_log(path: str):
    """The LARGEST `cell_pp=%d B` over the HOST-LEDGER POST lines of a boot
    log (the binding rank under the group MIN sync), with a provenance
    string. None when the log has no such field (pre-#1068 L8 form) or does
    not exist."""
    if not path or not os.path.isfile(path):
        return None, f"no previous log at {path or '<unset>'}"
    cells = []
    n_ledger = 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if HOST_LEDGER_TOKEN in line:
                n_ledger += 1
                m = _CELL_PP_RE.search(line)
                if m:
                    cells.append(int(m.group(1)))
    if not cells:
        return None, (
            f"previous log {path} carries {n_ledger} HOST-LEDGER POST line(s) "
            "without a cell_pp field (pre-#1068 form)"
        )
    return max(cells), (
        f"max cell_pp over {len(cells)} HOST-LEDGER POST line(s) of {path} "
        f"(cells seen: {sorted(set(cells))})"
    )


def size_host_pools(
    max_running_requests: int,
    chunked_prefill_size: int,
    ranks: int,
    memavail_bytes: int,
    cell_pp0_bytes: int,
    prompt_max_tokens: int,
    n_queue: int,
    mamba_host_mib: int,
    per_slot_rank0_mib: float,
    device_slots: int,
    provenance: Optional[dict] = None,
) -> Sizing:
    """The pure arithmetic. `provenance` maps term -> origin string for the
    A11.1 terms line; missing entries print as 'unstated'."""
    prov = dict(provenance or {})

    def p(term: str) -> str:
        return prov.get(term, "unstated")

    if max_running_requests <= 0 or chunked_prefill_size <= 0 or ranks <= 0:
        raise SizingRefused(
            f"{REFUSED_PREFIX}: config terms must be positive "
            f"(max_running_requests={max_running_requests} "
            f"chunked_prefill_size={chunked_prefill_size} ranks={ranks})"
        )
    if cell_pp0_bytes <= 0 or prompt_max_tokens <= 0 or per_slot_rank0_mib <= 0:
        raise SizingRefused(
            f"{REFUSED_PREFIX}: rig terms must be positive "
            f"(cell_pp0={cell_pp0_bytes} prompt_max={prompt_max_tokens} "
            f"per_slot_rank0={per_slot_rank0_mib})"
        )
    n_resident = int(max_running_requests)
    n_queue = int(n_queue)
    demand_rows = (n_resident + n_queue + CHAIN_LAG) * int(prompt_max_tokens)
    s_demand = int(math.ceil(demand_rows * cell_pp0_bytes / GB))

    # Anchor pools first: they are a term of the ledger cap.
    m_formula = int(
        math.ceil(
            (device_slots + max_running_requests + 1 + 2 * device_slots)
            * float(per_slot_rank0_mib)
        )
    )
    m_formula_rounded = int(math.ceil(m_formula / 100.0) * 100)
    m_mib = int(mamba_host_mib)
    anchor_slots_rank0 = int(m_mib * MIB // int(round(per_slot_rank0_mib * MIB)))
    anchor_floor = int(device_slots + max_running_requests + 1)
    anchors_bytes = 2 * ranks * m_mib * MIB

    cap_bytes = int(memavail_bytes) - FLOOR_BYTES - RESERVE_BYTES - LOAD_TRANSIENT_BYTES
    kv_budget = cap_bytes - anchors_bytes
    s_ledger = int(math.floor(kv_budget / (KV_MULTIPLE_OF_S * GB))) if kv_budget > 0 else 0

    floor_txt = f"{FLOOR_BYTES / GIB:.0f} GiB ({FLOOR_BYTES / GB:.2f} GB)"
    lines: list[str] = []
    lines.append(
        "#1068 WEG1 SIZING TERMS "
        f"n_resident={n_resident} ({p('n_resident')}) "
        f"n_queue={n_queue} ({p('n_queue')}) "
        f"chain_lag={CHAIN_LAG} (spec section 5) "
        f"prompt_max={prompt_max_tokens} ({p('prompt_max')}) "
        f"cell_pp0={cell_pp0_bytes} B ({p('cell_pp0')}) "
        f"memavail={memavail_bytes / GB:.2f} GB ({p('memavail')}) "
        f"floor={floor_txt} (#721 as host_ledger_preflight.sh FLOOR_G gates it; spec section 5 '16 GB' is a unit slip) "
        f"reserve={RESERVE_BYTES / GIB:.0f} GiB (pinned_host_budget.py:59) "
        f"load_transient={LOAD_TRANSIENT_BYTES / GIB:.0f} GiB (#721) "
        f"ranks={ranks} ({p('ranks')}) "
        f"m_mib={m_mib} ({p('m_mib')}) "
        f"per_slot_rank0={per_slot_rank0_mib:.2f} MiB ({p('per_slot_rank0')}) "
        f"device_slots={device_slots} ({p('device_slots')}) "
        f"chunked_prefill_size={chunked_prefill_size} ({p('chunked_prefill_size')}) "
        "-- provenance in parentheses: config | knob | default | previous log"
    )
    lines.append(
        f"#1068 WEG1 DEMAND rows={demand_rows} = ({n_resident} + {n_queue} + "
        f"{CHAIN_LAG}) x {prompt_max_tokens}; S_demand={s_demand} GB = "
        f"ceil({demand_rows} x {cell_pp0_bytes} B / 1e9)"
    )
    lines.append(
        f"#1068 WEG1 LEDGER cap={cap_bytes / GB:.2f} GB = memavail "
        f"{memavail_bytes / GB:.2f} GB - floor {floor_txt} - reserve "
        f"{RESERVE_BYTES / GB:.2f} GB (10 GiB) - load transient "
        f"{LOAD_TRANSIENT_BYTES / GB:.2f} GB (27 GiB); anchors={anchors_bytes / GB:.2f} GB "
        f"(2 phases x {ranks} ranks x {m_mib} MiB); kv_budget={kv_budget / GB:.2f} GB "
        f"over {KV_MULTIPLE_OF_S} x S (PP 2S + TP pin 6S) -> S_ledger={s_ledger} GB"
    )
    if s_ledger < 1:
        raise SizingRefused(
            f"{REFUSED_PREFIX}: ledger cap leaves no KV budget: memavail "
            f"{memavail_bytes / GB:.2f} GB - floor {floor_txt} - "
            f"reserve {RESERVE_BYTES / GB:.2f} GB - load transient "
            f"{LOAD_TRANSIENT_BYTES / GB:.2f} GB - anchors {anchors_bytes / GB:.2f} GB "
            f"= {kv_budget / GB:.2f} GB < {KV_MULTIPLE_OF_S} GB (one GB of S costs "
            f"{KV_MULTIPLE_OF_S} GB pinned). No fallback: free host RAM or lower "
            "WEG1_MAMBA_HOST_MIB by name."
        )
    ledger_rows = rows_of(s_ledger, cell_pp0_bytes)
    floor_rows = int(max_running_requests) * int(chunked_prefill_size)
    if ledger_rows < floor_rows:
        raise SizingRefused(
            f"{REFUSED_PREFIX}: ledger_rows={ledger_rows} (S_ledger={s_ledger} GB "
            f"at cell_pp0 {cell_pp0_bytes} B) < max_running_requests "
            f"{max_running_requests} x chunked_prefill_size {chunked_prefill_size} "
            f"= {floor_rows} rows (the in-tree floor G10 refuses this too); memavail "
            f"{memavail_bytes / GB:.2f} GB, floor {floor_txt}, reserve "
            f"{RESERVE_BYTES / GB:.2f} GB, load transient {LOAD_TRANSIENT_BYTES / GB:.2f} GB, "
            f"anchors {anchors_bytes / GB:.2f} GB"
        )
    if anchor_slots_rank0 < anchor_floor:
        raise SizingRefused(
            f"{REFUSED_PREFIX}: --hicache-mamba-host-mib {m_mib} buys "
            f"{anchor_slots_rank0} slots on rank 0 (per_slot {per_slot_rank0_mib:.2f} MiB) "
            f"< device_slots {device_slots} + max_running_requests "
            f"{max_running_requests} + 1 = {anchor_floor} (the in-tree anchor floor "
            f"refuses this too); formula value {m_formula} MiB (rounded "
            f"{m_formula_rounded})"
        )

    degraded = demand_rows > ledger_rows
    s_gb = min(s_demand, s_ledger)
    pool_rows = rows_of(s_gb, cell_pp0_bytes)
    spans = pool_rows / float(prompt_max_tokens)
    if degraded:
        lines.append(
            DEMAND_EXCEEDS_LEDGER
            % (demand_rows, n_resident, n_queue, prompt_max_tokens, ledger_rows, ledger_rows / float(prompt_max_tokens))
        )
        lines.append(DEMAND_EXCEEDS_LEDGER_A123)
        s_prov = (
            f"ledger cap (A11.2 named degradation: demand {s_demand} GB / "
            f"{demand_rows} rows > ledger {s_ledger} GB / {ledger_rows} rows)"
        )
    else:
        s_prov = (
            f"demand formula (A11.1: {demand_rows} rows fit the ledger "
            f"{ledger_rows} rows)"
        )
    # The provenance line survives (spec section 4.6: requested/ledger/EFFECTIVE;
    # 'requested' is spelled 'demand' here, see the module docstring).
    lines.append(
        f"  hicache_size: demand={s_demand} ledger={s_ledger} -> EFFECTIVE={s_gb} GB "
        f"({pool_rows} rows at cell_pp0 {cell_pp0_bytes} B, "
        f"spans_at_prompt_max={spans:.2f}) [provenance: {s_prov}]"
    )
    lines.append(
        f"  hicache_mamba_host_mib: formula={m_formula} (= ({device_slots} + "
        f"{max_running_requests} + 1 + 2 x {device_slots}) x {per_slot_rank0_mib:.2f} MiB, "
        f"rounded {m_formula_rounded}) knob={prov.get('m_mib_knob', '<unset>')} -> "
        f"EFFECTIVE={m_mib} MiB ({anchor_slots_rank0} local slots on rank 0 >= floor "
        f"{anchor_floor} = device_slots + max_running_requests + 1; the group MIN "
        f"binds, see '#1035 ANCHOR-POOL PROVENANCE ... synced_from_local=') "
        f"[provenance: {p('m_mib')}]"
    )
    return Sizing(
        n_resident=n_resident,
        n_queue=n_queue,
        prompt_max=int(prompt_max_tokens),
        cell_pp0=int(cell_pp0_bytes),
        demand_rows=demand_rows,
        s_demand_gb=s_demand,
        memavail_bytes=int(memavail_bytes),
        cap_bytes=cap_bytes,
        anchors_bytes=anchors_bytes,
        s_ledger_gb=s_ledger,
        ledger_rows=ledger_rows,
        s_gb=s_gb,
        pool_rows=pool_rows,
        spans_at_prompt_max=spans,
        m_mib=m_mib,
        m_formula_mib=m_formula,
        m_formula_rounded_mib=m_formula_rounded,
        anchor_slots_rank0=anchor_slots_rank0,
        anchor_floor_slots=anchor_floor,
        degraded=degraded,
        lines=lines,
    )


def _env_int(name: str, default: int):
    v = os.environ.get(name, "")
    if v == "":
        return default, f"default ({name} unset)"
    try:
        return int(v), f"knob {name}={v}"
    except ValueError as e:
        raise SizingRefused(f"{REFUSED_PREFIX}: {name}={v!r} is not an integer") from e


def _env_float(name: str, default: float):
    v = os.environ.get(name, "")
    if v == "":
        return default, f"default ({name} unset)"
    try:
        return float(v), f"knob {name}={v}"
    except ValueError as e:
        raise SizingRefused(f"{REFUSED_PREFIX}: {name}={v!r} is not a number") from e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-running-requests", type=int, required=True)
    ap.add_argument("--chunked-prefill-size", type=int, required=True)
    ap.add_argument("--ranks", type=int, required=True)
    ap.add_argument("--prev-log", default="", help="previous boot log; cell_pp0 is read from its HOST-LEDGER POST lines when present")
    args = ap.parse_args(argv)
    try:
        prov = {
            "n_resident": "config --max-running-requests",
            "ranks": "config --pp-size x --tp-size",
            "chunked_prefill_size": "config --chunked-prefill-size",
        }
        n_queue, prov["n_queue"] = _env_int("WEG1_QUEUE_DEPTH", args.max_running_requests)
        if "unset" in prov["n_queue"]:
            prov["n_queue"] = "default = max_running_requests (knob WEG1_QUEUE_DEPTH)"
        prompt_max, prov["prompt_max"] = _env_int("WEG1_PROMPT_MAX_TOKENS", DEFAULT_PROMPT_MAX)
        if "unset" in prov["prompt_max"]:
            prov["prompt_max"] = "default = Boot-2 load prompt class, spec section 5 (knob WEG1_PROMPT_MAX_TOKENS)"
        # cell_pp0: knob > previous log > default.
        cell_knob = os.environ.get("WEG1_CELL_PP0_BYTES", "")
        if cell_knob:
            cell_pp0, prov["cell_pp0"] = _env_int("WEG1_CELL_PP0_BYTES", DEFAULT_CELL_PP0)
        else:
            cell_pp0, why = cell_pp0_from_log(args.prev_log)
            if cell_pp0 is None:
                cell_pp0 = DEFAULT_CELL_PP0
                prov["cell_pp0"] = f"default 16384 B, spec section 5 log 1329/1350; {why}"
            else:
                prov["cell_pp0"] = f"previous log: {why}"
        mem_knob = os.environ.get("WEG1_MEMAVAIL_GB", "")
        if mem_knob:
            mem_gb, _ = _env_float("WEG1_MEMAVAIL_GB", 0.0)
            memavail = int(mem_gb * GB)
            prov["memavail"] = f"knob WEG1_MEMAVAIL_GB={mem_knob}"
        else:
            memavail = read_memavail_bytes()
            prov["memavail"] = f"live /proc/meminfo MemAvailable {memavail // 1024} kB x 1024"
        m_mib, m_prov = _env_int("WEG1_MAMBA_HOST_MIB", DEFAULT_MAMBA_MIB)
        prov["m_mib_knob"] = os.environ.get("WEG1_MAMBA_HOST_MIB", "") or "<unset>"
        prov["m_mib"] = (
            m_prov if "unset" not in m_prov
            else "default 2400 = spec section 5 acceptance value (64 slots = 3.2 x device_slots; R4 named), knob WEG1_MAMBA_HOST_MIB"
        )
        per_slot, prov["per_slot_rank0"] = _env_float("WEG1_PER_SLOT_RANK0_MIB", DEFAULT_PER_SLOT_RANK0_MIB)
        if "unset" in prov["per_slot_rank0"]:
            prov["per_slot_rank0"] = "default 37.41 MiB, log 1351 PP0 (knob WEG1_PER_SLOT_RANK0_MIB)"
        dev_slots, prov["device_slots"] = _env_int("WEG1_DEVICE_SLOTS", DEFAULT_DEVICE_SLOTS)
        if "unset" in prov["device_slots"]:
            prov["device_slots"] = "default 20, log 1351 device_slots (knob WEG1_DEVICE_SLOTS)"
        r = size_host_pools(
            max_running_requests=args.max_running_requests,
            chunked_prefill_size=args.chunked_prefill_size,
            ranks=args.ranks,
            memavail_bytes=memavail,
            cell_pp0_bytes=cell_pp0,
            prompt_max_tokens=prompt_max,
            n_queue=n_queue,
            mamba_host_mib=m_mib,
            per_slot_rank0_mib=per_slot,
            device_slots=dev_slots,
            provenance=prov,
        )
    except SizingRefused as e:
        print(str(e))
        return 2
    for ln in r.lines:
        print(ln)
    # Machine lines for the launcher (grep '^WEG1_').
    print(f"WEG1_S_GB={r.s_gb}")
    print(f"WEG1_M_MIB={r.m_mib}")
    print(f"WEG1_PROMPT_MAX={r.prompt_max}")
    print(f"WEG1_DEMAND_ROWS={r.demand_rows}")
    print(f"WEG1_LEDGER_ROWS={r.ledger_rows}")
    print(f"WEG1_POOL_ROWS={r.pool_rows}")
    print(f"WEG1_SPANS_AT_PROMPT_MAX={r.spans_at_prompt_max:.2f}")
    print(f"WEG1_DEGRADED={'1' if r.degraded else '0'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
