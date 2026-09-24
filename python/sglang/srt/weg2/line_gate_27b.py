"""27B LINE DESK GATE: the arm's own launcher calls, dry, on this tree -- HARD.

User order 2026-09-24 11:4xZ: the 27B runs its own line (desk/27b-up-line-0924
from 829ebd09f8, the weg2xsn411 tree) and a desk gate with the REAL arm argv
must show the standard before any boot; a deviation aborts.

What it does (no GPU, no rank, no boot; ``CUDA_VISIBLE_DEVICES=""``, the dry
run reads NVML and the host only):

1. READS THE ARM ITSELF (``--arm /spinning/gpu-arb/weg2/arm_xsn4xx.sh``): the
   arm is reduced to a side-effect-free bash program -- its assignments, its
   exports, the if-blocks that only assign, and its launcher calls rewritten to
   PRINT their argv and exported environment -- and run under ``env -i``.
   Service stops, mounts, git, loops, the boot itself: dropped. ``WT`` is
   forced to the tree under test. Result: every launcher invocation the arm
   makes (gate probes 0b, the dry-run trio, the launch), in its order.
2. RUNS each probe and the launch as ``--dry-run`` on this tree with that
   argv and environment, and judges the probes by the arm's OWN criteria
   (rc, gate_assert / gate_refuse strings read off its gates_run body).
3. HOLDS THE LAUNCH to the standard (user orders 08.09. / 16.09., restated
   24.09.): P cut ``--p-cut``/``--p-attn`` (default 42,11,11 / 10,3,3,
   makespan), pool floor = cap+chunk with the cap at 262144 (one request),
   host ledger FUNDABLE at the arm's pinned M, and group P/D argv equal to the
   last good boot's (weg2xsn411) except the flags the launcher SOLVES per boot.

Exit 0 = PASS, 1 = FAIL, 2 = could not run.

    python <tree>/python/sglang/srt/weg2/line_gate_27b.py \\
        --arm /spinning/gpu-arb/weg2/arm_xsn411.sh --work-dir DIR

Run BY PATH: the parent imports nothing from sglang (``import sglang`` is
612 MiB RSS; the dry run child needs its own ~1 GiB in the agents' 3 GiB
cgroup).
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

EVIDENCE_DIR = "/spinning/evidence-665-f1"
VENV_PY = "/spinning/htsglang-gpu/.venv/bin/python"
GOOD_REF_TAG = "weg2xsn411"
#: the last good boot's OWN dry run of the same arm argv: a dry run prices D's
#: budget from an EXPECTATION of P's dormant residue (no P exists yet), so a
#: dry run is compared with a dry run -- never with the boot's measured budgets.
GOOD_REF_DRY = "/spinning/evidence-665-f1/weg2xsn411_0917/dry_deviation.log"
#: tolerance for per-rank budgets solved from measured residues: the #1444
#: record's own margin (launcher.DC_RECORD_MARGIN_MIB) -- the residue of one
#: boot to the next moves by tens of MiB (xsn410 -> xsn411: 2094 -> 2070).
BUDGET_TOL_MIB = 256
STANDARD_P_CUT = "42,11,11"
STANDARD_P_ATTN = "10,3,3"
STANDARD_CONTEXT_TOKENS = 262144

#: flags the launcher SOLVES per boot from measured inputs (budgets, pools):
#: a changed value is the solve seeing other inputs, not another form.
SOLVED_FLAGS = frozenset({"--max-total-tokens", "--rank-gpu-memory-mib"})
PER_BOOT_FLAGS = frozenset({"--admin-api-key"})


# --------------------------------------------------------------------------
# bash text helpers (quote-aware)
# --------------------------------------------------------------------------


def _quote_state_after(s: str, q: Optional[str] = None) -> Optional[str]:
    """The open quote (``'``/``"``) at the end of ``s``, starting in ``q``."""
    i = 0
    while i < len(s):
        c = s[i]
        if q is None:
            if c == "\\":
                i += 2
                continue
            if c == "#" and (i == 0 or s[i - 1] in " \t;"):
                return None  # comment to end of line
            if c in ("'", '"'):
                q = c
        elif q == "'":
            if c == "'":
                q = None
        else:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                q = None
        i += 1
    return q


def _strip_comment(line: str) -> str:
    q = None
    i = 0
    while i < len(line):
        c = line[i]
        if q is None:
            if c == "\\":
                i += 2
                continue
            if c == "#" and (i == 0 or line[i - 1] in " \t;"):
                return line[:i].rstrip()
            if c in ("'", '"'):
                q = c
        elif q == "'":
            if c == "'":
                q = None
        else:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                q = None
        i += 1
    return line.rstrip()


def _split_top(s: str, sep: str = ";") -> List[str]:
    parts, cur, q, i = [], [], None, 0
    while i < len(s):
        c = s[i]
        if q is None and c == "\\" and i + 1 < len(s):
            cur.append(s[i:i + 2])
            i += 2
            continue
        if q is None:
            if c in ("'", '"'):
                q = c
            elif c == sep and not (sep == ";" and i + 1 < len(s) and s[i + 1] == ";"):
                parts.append("".join(cur))
                cur = []
                i += 1
                continue
        elif q == "'" and c == "'":
            q = None
        elif q == '"':
            if c == "\\" and i + 1 < len(s):
                cur.append(s[i:i + 2])
                i += 2
                continue
            if c == '"':
                q = None
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


def _cut_redirect(cmd: str) -> Tuple[str, str]:
    q, i = None, 0
    while i < len(cmd):
        c = cmd[i]
        if q is None:
            if c == "\\":
                i += 2
                continue
            if c in ("'", '"'):
                q = c
            elif c == ">":
                rest = cmd[i + 1:].strip().split()
                return cmd[:i].rstrip(), (rest[0] if rest else "")
        elif q == "'" and c == "'":
            q = None
        elif q == '"' and c == '"':
            q = None
        i += 1
    return cmd, ""


def _statements(text: str) -> List[str]:
    """Logical statements: backslash continuations joined, multi-line quoted
    strings kept whole, comments stripped."""
    out: List[str] = []
    cur, q = "", None
    for raw in text.split("\n"):
        if q is None and not cur and raw.lstrip().startswith("#"):
            continue
        piece = raw if q is not None else _strip_comment(raw)
        if q is None and piece.endswith("\\"):
            cur += piece[:-1] + " "
            continue
        cur += piece
        q = _quote_state_after(piece, q)
        if q is not None:
            cur += "\n"
            continue
        if cur.strip():
            out.append(cur.strip())
        cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


# --------------------------------------------------------------------------
# the arm reader
# --------------------------------------------------------------------------

_LAUNCHER_CMD = "-m sglang.srt.weg2.launcher"
_ASSIGN_ONLY_RE = re.compile(r"^(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*=")
_FUNC_ONE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{(.*)\}$", re.S)
_FUNC_OPEN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{$")
_REPLAYED = ("gate_run", "gate_run3", "run", "gates_run")
_COND_OK_RE = re.compile(r"^(?:if|elif)\s+(?:!\s*)?(?:\[\[?\s|test\s|grep\s)")
_ENV_SKIP = {"PATH", "HOME", "PWD", "OLDPWD", "SHLVL", "_"}


def _is_pure_assignment(stmt: str) -> bool:
    if "$(" in stmt or "`" in stmt:
        return False
    parts = [p.strip() for p in _split_top(stmt) if p.strip()]
    return bool(parts) and all(_ASSIGN_ONLY_RE.match(p) for p in parts)


def _opens_loop(stmt: str) -> int:
    """net do/done (and case/esac) nesting a statement opens, outside quotes."""
    text = []
    q, i = None, 0
    while i < len(stmt):
        c = stmt[i]
        if q is None and c in ("'", '"'):
            q = c
        elif q is not None and c == q:
            q = None
        elif q is None:
            text.append(c)
        i += 1
    t = " " + "".join(text).replace(";", " ; ") + " "
    opens = len(re.findall(r"[\s;]do[\s;]", t)) + len(re.findall(r"\scase\s", t))
    closes = len(re.findall(r"[\s;]done[\s;]", t)) + len(re.findall(r"[\s;]esac[\s;]", t))
    if re.match(r"^\s*(for|while|until)\s", t) and not re.search(r"[\s;]do[\s;]", t):
        opens += 1  # `for x in ...` with `do` on the next line
    return opens - closes


def _rewrite_launcher_stmt(stmt: str, kind: str) -> Optional[str]:
    k = stmt.find(_LAUNCHER_CMD)
    if k < 0:
        return None
    args, target = _cut_redirect(stmt[k + len(_LAUNCHER_CMD):])
    args = args.rstrip().rstrip("&").rstrip()
    return f"__emit {kind} {target or kind} {args}"


def _rewrite_function(name: str, body: str) -> str:
    kind = "probe" if name.startswith("gate_run") else "dry"
    keep = []
    for st in (p.strip() for p in _split_top(body)):
        if not st:
            continue
        if _LAUNCHER_CMD in st:
            keep.append(_rewrite_launcher_stmt(st, kind))
        elif _is_pure_assignment(st) or st == "shift":
            keep.append(st)
    return f"{name}(){{ {'; '.join(k for k in keep if k)}; }}"


def _gate_criteria(body: Sequence[str]) -> Dict[str, Dict[str, object]]:
    crit: Dict[str, Dict[str, object]] = {}
    logvar: Dict[str, str] = {}
    rcvar: Dict[str, str] = {}
    current = None
    for st in body:
        m = re.match(r"^(gate_run3|gate_run)\b([^;]*);\s*([A-Za-z_]\w*)=\$\?", st)
        if m:
            current = m.group(3)
            continue
        m = re.match(r'^([A-Za-z_]\w*)="?\$EVIDPROBE/([^"\s]+)\.log"?$', st)
        if m and current:
            logvar[m.group(1)] = m.group(2)
            rcvar[current] = m.group(2)
            crit[m.group(2)] = {"rc0": False, "have": [], "not": []}
            continue
        m = re.match(r'^\[\s*"\$([A-Za-z_]\w*)"\s*=\s*"0"\s*\]\s*\|\|\s*die', st)
        if m and m.group(1) in rcvar:
            crit[rcvar[m.group(1)]]["rc0"] = True
            continue
        m = re.match(r'^gate_(assert|refuse)\s+"\$([A-Za-z_]\w*)"\s+(.*)$', st)
        if m and m.group(2) in logvar:
            key = "have" if m.group(1) == "assert" else "not"
            crit[logvar[m.group(2)]][key] += shlex.split(m.group(3))
    return crit


def build_arm_replay(arm_text: str, tree: str, positional: Sequence[str]) -> Tuple[str, Dict[str, Dict[str, object]]]:
    stmts = _statements(arm_text)
    out: List[str] = []
    placeholders: List[str] = []
    criteria: Dict[str, Dict[str, object]] = {}
    i, n = 0, len(stmts)
    while i < n:
        st = stmts[i]
        i += 1
        m1 = _FUNC_ONE_RE.match(st)
        if m1:
            if m1.group(1) in _REPLAYED:
                out.append(_rewrite_function(m1.group(1), m1.group(2)))
            continue
        mo = _FUNC_OPEN_RE.match(st)
        if mo:
            body = []
            while i < n and stmts[i] != "}" and not stmts[i].startswith("}"):
                body.append(stmts[i])
                i += 1
            i += 1
            if mo.group(1) == "gates_run":
                criteria = _gate_criteria(body)
                calls = []
                for b in body:
                    fm = _FUNC_ONE_RE.match(b)
                    if fm and fm.group(1) in _REPLAYED:
                        calls.append(_rewrite_function(fm.group(1), fm.group(2)))
                        continue
                    cm = re.match(r"^(gate_run3|gate_run)\b([^;]*)", b)
                    if cm:
                        calls.append((cm.group(1) + cm.group(2)).strip())
                out.append("gates_run(){ " + "; ".join(calls) + "; }")
            continue
        depth = _opens_loop(st)
        if depth > 0 or re.match(r"^(for|while|until|case)\s", st):
            while i < n and depth > 0:
                depth += _opens_loop(stmts[i])
                i += 1
            continue
        if re.match(r"^if\s", st):
            block = [st]
            nest = 1 if not re.search(r"(^|;)\s*fi\s*$", st) else 0
            while i < n and nest > 0:
                b = stmts[i]
                i += 1
                if re.match(r"^if\s", b):
                    nest += 1
                if b == "fi" or re.search(r"(^|;)\s*fi\s*;?$", b):
                    nest -= 1
                block.append(b)
            ok = bool(_COND_OK_RE.match(block[0])) and "$(" not in block[0] and "|" not in block[0]
            kept = [block[0]]
            for b in block[1:]:
                if not ok:
                    break
                if b.startswith("elif "):
                    ok = bool(_COND_OK_RE.match(b)) and "$(" not in b and "|" not in b
                    kept.append(b)
                elif b in ("else", "fi", "then") or b.startswith("fi"):
                    kept.append(b)
                elif re.match(r"^(say|die|echo)\b", b):
                    kept.append(":")
                elif _is_pure_assignment(b) or re.match(r"^(gates_run|gate_run3|gate_run|run)(\s|$)", b):
                    kept.append(b)
                else:
                    ok = False
            if ok:
                out.extend(kept)
            continue
        if _ASSIGN_ONLY_RE.match(st):
            if _is_pure_assignment(st):
                name = re.match(r"^(?:export\s+)?([A-Za-z_]\w*)=", st).group(1)
                if name == "WT":
                    out.append(f"WT={shlex.quote(tree)}")
                else:
                    out.append(st)
            else:
                for p in _split_top(st):
                    mm = re.match(r"^\s*(?:export\s+)?([A-Za-z_]\w*)=", p)
                    if mm:
                        placeholders.append(mm.group(1))
            continue
        if re.match(r"^\[\[?\s.*\]\]?\s*&&\s*[A-Za-z_]\w*=", st) and "$(" not in st:
            out.append(st)
            continue
        if _LAUNCHER_CMD in st and "$BASE" in st:
            rw = _rewrite_launcher_stmt(st, "launch")
            if rw:
                out.append(rw)
            continue
        if re.match(r"^(gates_run|run)(\s|$)", st):
            out.append(st)
    prelude = [
        "set +e",
        "say(){ :; }",
        "die(){ :; }",
        "__emit(){ local kind=\"$1\" name=\"$2\"; shift 2; printf 'CALL\\0%s\\0%s\\0' \"$kind\" \"$name\"; "
        "for __a in \"$@\"; do printf 'ARG\\0%s\\0' \"$__a\"; done; printf 'ENV\\0'; env -0; printf 'END\\0'; }",
        "set -- " + " ".join(shlex.quote(p) for p in positional),
        f"WT_OVERRIDE={shlex.quote(tree)}",
    ] + [f"{p}=__GATE_{p}__" for p in dict.fromkeys(placeholders) if p != "WT"]
    return "\n".join(prelude + out) + "\n", criteria


@dataclass
class ArmCall:
    kind: str
    name: str
    argv: List[str]
    env: Dict[str, str]
    must_rc0: bool = False
    must_have: Tuple[str, ...] = ()
    must_not: Tuple[str, ...] = ()


def read_arm(arm_path: str, tree: str, positional: Sequence[str] = ("GATE-SHA", "GATE-CUSHION")) -> List[ArmCall]:
    with open(arm_path, errors="replace") as f:
        program, criteria = build_arm_replay(f.read(), tree, positional)
    with tempfile.TemporaryDirectory(prefix="arm_replay_") as td:
        res = subprocess.run(["bash", "--noprofile", "--norc", "-c", program], cwd=td,
                             env={"PATH": "/usr/bin:/bin", "HOME": td},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    toks = res.stdout.decode("utf-8", "replace").split("\0")
    calls: List[ArmCall] = []
    k = 0
    while k < len(toks):
        if toks[k] != "CALL":
            k += 1
            continue
        kind, name = toks[k + 1], toks[k + 2]
        k += 3
        argv: List[str] = []
        while k < len(toks) and toks[k] == "ARG":
            argv.append(toks[k + 1])
            k += 2
        env: Dict[str, str] = {}
        if k < len(toks) and toks[k] == "ENV":
            k += 1
            while k < len(toks) and toks[k] != "END":
                if "=" in toks[k]:
                    key, val = toks[k].split("=", 1)
                    if key not in _ENV_SKIP:
                        env[key] = val
                k += 1
            k += 1
        base = os.path.basename(name)
        base = base[:-4] if base.endswith(".log") else base
        if kind == "launch":
            base = "launch"
        c = criteria.get(base, {})
        calls.append(ArmCall(kind, base, argv, env, bool(c.get("rc0")),
                             tuple(c.get("have", ())), tuple(c.get("not", ()))))
    if not calls:
        raise RuntimeError(f"arm replay of {arm_path} produced no launcher call "
                           f"(bash rc={res.returncode}: {res.stderr.decode('utf-8', 'replace')[-500:]})")
    return calls


# --------------------------------------------------------------------------
# the box: live, or the declared quiet snapshot
# --------------------------------------------------------------------------

GIB = 1 << 30
#: THE DECLARED QUIET BOX (``--box quiet``, the default): this container read
#: with NO boot running, 2026-09-24 ~11:50Z (memory.current 10.07 GiB, anon
#: 7.52, file 2.14, shmem 0.00 GiB). The gate judges the CODE -- which cut,
#: which floor, which arm the ledger funds -- and must not be decided by
#: whichever seat happens to hold the box at the moment it runs (the Next-Flash
#: seat's boot reads 94.5 GiB and a live launch_server, and every dry run then
#: refuses at the #1217 preflight). slab_reclaimable is 0 on purpose: less
#: reclaimable = a HIGHER non-reclaimable origin, the conservative direction.
#: ``--box live`` reads the real box, exactly as the arm's own preflight does.
QUIET_BOX = {"current": 10.07, "anon": 7.52, "file": 2.14, "shmem": 0.0, "slab_reclaimable": 0.0}


def write_quiet_box(root: str) -> Tuple[str, str, str]:
    """``(meminfo path, cgroup dir, description)`` of the declared quiet box;
    MemTotal and the oom_kill baseline are the live box's own."""
    os.makedirs(os.path.join(root, "cgroup"), exist_ok=True)
    total_kib = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total_kib = int(line.split()[1])
    qb = {k: int(v * GIB) for k, v in QUIET_BOX.items()}
    avail_kib = total_kib - (qb["current"] - qb["file"]) // 1024
    meminfo = os.path.join(root, "meminfo")
    with open(meminfo, "w") as f:
        f.write(f"MemTotal:       {total_kib} kB\nMemFree:        {avail_kib} kB\n"
                f"MemAvailable:   {avail_kib} kB\nCached:         {qb['file'] // 1024} kB\n"
                f"Shmem:          {qb['shmem'] // 1024} kB\nSwapTotal:      0 kB\n")
    oom = 0
    try:
        with open("/sys/fs/cgroup/memory.events") as f:
            m = re.search(r"^oom_kill (\d+)", f.read(), re.M)
            oom = int(m.group(1)) if m else 0
    except OSError:
        pass
    cg = os.path.join(root, "cgroup")
    for name, text in (("memory.current", str(qb["current"])), ("memory.peak", str(qb["current"])),
                       ("memory.max", "max"),
                       ("memory.events", f"low 0\nhigh 0\nmax 0\noom 0\noom_kill {oom}\n"),
                       ("memory.stat", "".join(f"{k} {qb[k]}\n" for k in
                                               ("anon", "file", "shmem", "slab_reclaimable"))
                        + "unevictable 0\n")):
        with open(os.path.join(cg, name), "w") as f:
            f.write(text + ("" if text.endswith("\n") else "\n"))
    return meminfo, cg, (f"declared quiet box (no boot running, 2026-09-24 ~11:50Z): memory.current "
                         f"{QUIET_BOX['current']:.2f} GiB, anon {QUIET_BOX['anon']:.2f}, file "
                         f"{QUIET_BOX['file']:.2f}, shmem {QUIET_BOX['shmem']:.2f}; MemTotal and the oom_kill "
                         f"baseline live; /dev/shm and the process table private (unshare)")


_CHILD_FLAG = "--_launcher-child"


def _child_main(argv: Sequence[str]) -> int:
    """In the sandbox: the launcher with the quiet box's meminfo/cgroup."""
    meminfo, cgroup, rest = argv[0], argv[1], list(argv[2:])
    from sglang.srt.weg2 import launcher as L

    L.MEMINFO_PATH = meminfo
    L.CGROUP_ROOT = cgroup
    return int(L.cli(rest) or 0)


# --------------------------------------------------------------------------
# dry runs and verdicts
# --------------------------------------------------------------------------


def run_call(call: ArmCall, tree: str, work_dir: str, reuse: bool = False, timeout_s: int = 900,
             box: Optional[Tuple[str, str]] = None) -> Tuple[int, str]:
    log_path = os.path.join(work_dir, f"{call.kind}_{call.name}.log")
    if reuse and os.path.exists(log_path):
        with open(log_path, errors="replace") as f:
            m = re.search(r"^# rc=(-?\d+)$", f.read(), re.M)
        if m:
            return int(m.group(1)), log_path
    largv = list(call.argv)
    if "--dry-run" not in largv:
        largv.append("--dry-run")
    if box is None:
        argv = [VENV_PY, "-m", "sglang.srt.weg2.launcher"] + largv
    else:
        argv = ["unshare", "--mount", "--pid", "--fork", "--mount-proc", "sh", "-c",
                'mount -t tmpfs -o size=64m tmpfs /dev/shm && exec "$0" "$@"',
                VENV_PY, os.path.abspath(__file__), _CHILD_FLAG, box[0], box[1]] + largv
    env = dict(os.environ)
    env.update(call.env)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.path.join(tree, "python")
    with open(log_path, "w") as fh:
        fh.write("# " + " ".join(shlex.quote(a) for a in argv) + "\n")
        fh.write("# env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(call.env.items())) + "\n")
        fh.flush()
        try:
            rc = subprocess.run(argv, cwd=tree, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = 124
        fh.write(f"# rc={rc}\n")
    return rc, log_path


def flag_map(argv: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, List[str]] = {}
    i, argv = 0, list(argv)
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            if "=" in t:
                k, v = t.split("=", 1)
                out.setdefault(k, []).append(v)
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                vals, j = [argv[i + 1]], i + 2
                while j < len(argv) and not argv[j].startswith("--"):
                    vals.append(argv[j])
                    j += 1
                out.setdefault(t, []).append(" ".join(vals))
                i = j
            else:
                out.setdefault(t, []).append("")
                i += 1
        else:
            i += 1
    return {k: tuple(v) for k, v in out.items()}


_ARGV_RE = re.compile(r"WEG2-LAUNCH group (?P<g>[PD]) argv: (?P<argv>.*)$")


def group_argv(log_path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            m = _ARGV_RE.search(line.rstrip("\n"))
            if m:
                out[m.group("g")] = shlex.split(m.group("argv"))
    return out


def judge_probe(call: ArmCall, rc: int, log_path: str) -> Tuple[bool, str]:
    body = open(log_path, errors="replace").read()
    why = []
    if call.must_rc0 and rc != 0:
        refused = re.findall(r"WEG2-LAUNCH REFUSED: .*", body)
        why.append(f"rc={rc} (the arm dies on it): {refused[-1][:400] if refused else body[-300:]}")
    why += [f"emitter '{w}' missing" for w in call.must_have if w not in body]
    why += [f"'{w}' must not appear" for w in call.must_not if w in body]
    return (not why), "; ".join(why) or f"rc={rc}, {list(call.must_have)} present, {list(call.must_not)} absent"


_FLOOR_RE = re.compile(r"PP-CUT POOL FLOOR RULE: source=cap\+chunk \(p_bs=1[^)]*\) floor=(\d+) = "
                       r"max_kv_per_request (\d+) \+ chunk (\d+)")
_OBJ_RE = re.compile(r"PP-CUT solver: objective=(\w+) layers=([\d,]+) attn=([\d,]+)")
_CHOSEN_RE = re.compile(r"WEG2-HOST-LEDGER CHOSEN S=(\d+) GB .*? M=(\d+) MiB")
_PEAK_RE = re.compile(r"reap_headroom=([-\d.]+) GiB \(hard bound ([\d.]+) - predicted run peak ([\d.]+) GiB\)")


def launch_problems(log_path: str, good_argv: Dict[str, List[str]], pinned_m: Optional[str],
                    p_cut: str, p_attn: str) -> Tuple[List[str], List[str]]:
    """``(problems, facts)`` of the launch dry run against the standard."""
    body = open(log_path, errors="replace").read()
    problems: List[str] = []
    facts: List[str] = []
    if "DRY-RUN complete" not in body:
        refused = re.findall(r"WEG2-LAUNCH REFUSED: .*", body)
        return [f"dry run did not complete: {refused[-1][:500] if refused else body[-400:]}"], facts
    new = group_argv(log_path)
    fp = flag_map(new.get("P", []))
    cut = (fp.get("--pp-stage-ratio") or ("?",))[0]
    attn = (fp.get("--pp-attn-stage-ratio") or ("?",))[0]
    facts.append(f"P cut {cut} attn {attn}")
    if cut != p_cut or attn != p_attn:
        problems.append(f"P cut {cut} / attn {attn} is not the standard {p_cut} / {p_attn}")
    o = _OBJ_RE.search(body)
    if not o or o.group(1) != "makespan":
        problems.append(f"PP-CUT objective {o.group(1) if o else '?'} != makespan")
    f = _FLOOR_RE.search(body)
    if not f:
        rule = re.search(r"PP-CUT POOL FLOOR RULE: (source=[^\n]{0,90})", body)
        problems.append("P pool floor is not cap+chunk (p_bs=1): " + (rule.group(1) if rule else "no rule line"))
    else:
        facts.append(f"P pool floor {f.group(1)} = cap {f.group(2)} + chunk {f.group(3)}")
        if int(f.group(2)) != STANDARD_CONTEXT_TOKENS:
            problems.append(f"P pool floor cap {f.group(2)} != {STANDARD_CONTEXT_TOKENS} (one full-context request)")
    ch = _CHOSEN_RE.search(body)
    pk = _PEAK_RE.search(body)
    if not ch:
        problems.append("no WEG2-HOST-LEDGER CHOSEN line: the host ledger funded no arm")
    else:
        facts.append(f"host ledger CHOSEN S={ch.group(1)} M={ch.group(2)}"
                     + (f", predicted run peak {pk.group(3)} GiB vs hard bound {pk.group(2)} "
                        f"(headroom {pk.group(1)})" if pk else ""))
        if pinned_m is not None and ch.group(2) != str(pinned_m):
            problems.append(f"host ledger M={ch.group(2)} != the arm's pinned M={pinned_m}")
    ptok = (fp.get("--max-total-tokens") or (None,))[0]
    if ptok is None or int(ptok) < STANDARD_CONTEXT_TOKENS:
        problems.append(f"P --max-total-tokens {ptok} holds no full-context request ({STANDARD_CONTEXT_TOKENS})")
    for g in ("P", "D"):
        a, b = flag_map(good_argv.get(g, [])), flag_map(new.get(g, []))
        if not a:
            problems.append(f"reference {GOOD_REF_TAG} carries no group {g} argv")
            continue
        for k in sorted(set(a) | set(b)):
            if k == "--model-path" or a.get(k) == b.get(k):
                continue
            va, vb = " ".join(a.get(k, ("<absent>",))), " ".join(b.get(k, ("<absent>",)))
            if g == "P" and k in ("--pp-stage-ratio", "--pp-attn-stage-ratio"):
                continue  # judged against the standard above
            if k == "--rank-gpu-memory-mib":
                try:
                    da = [int(x) for x in va.split(",")]
                    db = [int(x) for x in vb.split(",")]
                    worst = max(abs(x - y) for x, y in zip(da, db)) if len(da) == len(db) else None
                except ValueError:
                    worst = None
                if worst is None or worst > BUDGET_TOL_MIB:
                    problems.append(f"group {g} {k}: {GOOD_REF_TAG} {va} -> {vb} (beyond +-{BUDGET_TOL_MIB} MiB)")
                else:
                    facts.append(f"{g} {k}: {va} -> {vb} (solved, within +-{BUDGET_TOL_MIB} MiB)")
            elif k in SOLVED_FLAGS or k in PER_BOOT_FLAGS:
                facts.append(f"{g} {k}: {va} -> {vb} (solved per boot)")
            else:
                problems.append(f"group {g} argv {k}: {GOOD_REF_TAG} {va!r} -> {vb!r}")
    return problems, facts


def good_ref_log(tag: str = GOOD_REF_TAG) -> Optional[str]:
    hits = [os.path.join(EVIDENCE_DIR, n) for n in os.listdir(EVIDENCE_DIR)
            if n.startswith(f"boot_weg2_{tag}_") and n.endswith(".front.log")]
    return max(hits, key=os.path.getmtime) if hits else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == _CHILD_FLAG:
        return _child_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, help="the arm script (arm_xsn4xx.sh) whose launcher calls are gated")
    ap.add_argument("--tree", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--positional", default="GATE-SHA GATE-CUSHION",
                    help="the arm's positional arguments (shell words); defaults leave every optional one at its default")
    ap.add_argument("--p-cut", default=STANDARD_P_CUT)
    ap.add_argument("--p-attn", default=STANDARD_P_ATTN)
    ap.add_argument("--good-ref", default=GOOD_REF_TAG)
    ap.add_argument("--good-ref-dry", default=GOOD_REF_DRY,
                    help="the reference arm's own dry run of the same argv (dry is compared with dry)")
    ap.add_argument("--skip-probes", action="store_true", help="gate only the launch call")
    ap.add_argument("--reuse", action="store_true", help="re-read finished dry-run logs in --work-dir")
    ap.add_argument("--box", choices=["quiet", "live"], default="quiet",
                    help="quiet (default): the declared quiet-box snapshot, private /dev/shm and process "
                         "table -- judges the code; live: the real box, as the arm's own preflight")
    ns = ap.parse_args(argv)
    tree = os.path.abspath(ns.tree)
    work = ns.work_dir or tempfile.mkdtemp(prefix="line_gate_27b_")
    os.makedirs(work, exist_ok=True)
    try:
        calls = read_arm(ns.arm, tree, shlex.split(ns.positional))
    except Exception as e:  # noqa: BLE001
        print(f"LINE-GATE-27B could not read the arm: {e}")
        return 2
    box = None
    if ns.box == "quiet":
        meminfo, cg, box_desc = write_quiet_box(os.path.join(work, "box"))
        box = (meminfo, cg)
    else:
        box_desc = "LIVE box (the real /proc, /dev/shm and cgroup, as the arm's preflight reads them)"
    print(f"LINE-GATE-27B arm={ns.arm} tree={tree} work={work}")
    print(f"  box: {box_desc}")
    print("  arm calls: " + ", ".join(f"{c.kind}:{c.name}" for c in calls))
    ok = True
    for c in calls:
        if c.kind == "probe" and not ns.skip_probes:
            t0 = time.time()
            rc, log = run_call(c, tree, work, reuse=ns.reuse, box=box)
            good, why = judge_probe(c, rc, log)
            ok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] probe {c.name} ({time.time() - t0:.0f} s): {why}")
    launch = [c for c in calls if c.kind == "launch"]
    if not launch:
        print("  [FAIL] the arm makes no launch call this reader could see")
        return 1
    c = launch[-1]
    t0 = time.time()
    rc, log = run_call(c, tree, work, reuse=ns.reuse, box=box)
    gref = ns.good_ref_dry if ns.good_ref_dry and os.path.exists(ns.good_ref_dry) else good_ref_log(ns.good_ref)
    good_argv = group_argv(gref) if gref else {}
    print(f"  reference: {gref}")
    pinned = (flag_map(c.argv).get("--pin-ledger-arm-m") or (None,))[-1]
    problems, facts = launch_problems(log, good_argv, pinned, ns.p_cut, ns.p_attn)
    print(f"  launch dry run rc={rc} ({time.time() - t0:.0f} s) -> {log}")
    with open(log, errors="replace") as f:
        for line in f:
            if any(k in line for k in ("WEG2-27B-LINE", "#1444 DC-RESIDUE", "PP-CUT depth axis",
                                       "PP-CUT POOL FLOOR:", "WEG2 DFLASH-PRODUCE-ON-P")):
                print("  | " + line.split("WEG2-LAUNCH ", 1)[-1].rstrip()[:300])
    for fct in facts:
        print(f"  - {fct}")
    for p in problems:
        print(f"  [FAIL] {p}")
    ok &= not problems and rc == 0
    print(f"LINE-GATE-27B {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
